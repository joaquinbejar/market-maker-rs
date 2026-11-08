"""Thin Gymnasium environments over the native worker.

Each environment holds one worker thread, and every number that matters comes from it:
the `f32` reward is the kernel's derived value, and the exact `Decimal` reward and net
result come back as strings in `info`. An error from the worker ends the environment,
and nothing continues after it.
"""

import json
import math

import gymnasium as gym
import numpy as np

from ._native import NativeEnv

_SEED_LIMIT = 2**64


class MarketMakingEnv(gym.Env):
    """The observation, the reward and the end of an episode, for any action encoding.

    The observation holds each feature's normalized value, then whether it has a value at
    this decision. `terminated` means the data ended and `truncated` that `max_steps`
    decisions were taken, as the specification states.
    """

    metadata = {"render_modes": []}

    def __init__(self, config, rows, expected_spec=None, seed=0):
        super().__init__()
        self._config = dict(config)
        self._native = NativeEnv(json.dumps(self._config), list(rows), expected_spec)
        self.bridge_info = dict(self._native.info())
        self._features = self.bridge_info["features"]
        self._offsets = self.bridge_info["offsets"]
        self._sizes = self.bridge_info["sizes"]
        # A hedge takes its size from the same grid of sizes as a quote, and `hedge_mask` has
        # one bit per size.
        self._hedge = bool(self.bridge_info["hedge_actions"])
        self._hedge_mask = 0
        # The cancel-only action, in the action space of version 4.
        self._cancel = bool(self.bridge_info["cancel_action"])
        # The lives of an expiring quote, in the action space of version 5.
        self._lives = len(self.bridge_info["quote_lives_ns"])
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self._features,),
            dtype=np.float32,
        )
        self._next_seed = seed
        self._mask = None

    @property
    def config(self):
        """A copy of the configuration this environment was built from."""
        return json.loads(json.dumps(self._config))

    @property
    def spec_id(self):
        """The specification this environment runs, as `expected_spec` takes it."""
        return tuple(self.bridge_info["spec"])

    def _observation(self, observed):
        values = np.asarray(observed["values"], dtype=np.float32)
        present = np.asarray(
            [raw is not None for raw in observed["raw"]], dtype=np.float32
        )
        return np.concatenate((values, present))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed
        self._next_seed = (seed + 1) % _SEED_LIMIT
        reply = self._native.reset(seed)
        if reply["done"]:
            raise ValueError("the data holds no decision")
        self._mask = reply["mask"]
        self._hedge_mask = reply["hedge_mask"]
        observed = reply["observation"]
        info = {"row": observed["row"], "raw_observation": observed["raw"], "seed": seed}
        return self._observation(observed), info

    def _decode(self, action):
        """Returns `(kind, indices)` for the worker, from the policy's action."""
        raise NotImplementedError

    def step(self, action):
        kind, indices = self._decode(action)
        return self.act(kind, indices, action)

    def act(self, kind, indices=(), action=None):
        """Takes the decoded action `(kind, indices)`: what `step` does after decoding.

        A wrapper that acts for the policy (`ExpiringQuote`) uses it, with `action` the
        policy's own action for the record, if any.
        """
        indices = tuple(indices)
        if kind == "hold":
            reply = self._native.step_hold()
        elif kind == "quote":
            reply = self._native.step_quote(*indices)
        elif kind == "hedge":
            reply = self._native.step_hedge(*indices)
        elif kind == "cancel":
            reply = self._native.step_cancel()
        elif kind == "expiring_quote":
            reply = self._native.step_expiring_quote(*indices)
        else:
            raise ValueError(f"unknown action kind {kind!r}")
        if action is None:
            action = [kind, *indices]
        extra = {
            "decoded_action": {"kind": kind, "indices": list(indices)},
            "applied_action": (
                {"kind": kind, "indices": list(indices)}
                if reply["applied"]
                else {"kind": "hold", "indices": []}
            ),
        }
        return self._transition(reply, action, extra)

    def step_hold(self):
        """A hold, whatever the encoding: the resting orders stay and nothing is sent."""
        reply = self._native.step_hold()
        hold = {"kind": "hold", "indices": []}
        return self._transition(
            reply, None, {"decoded_action": hold, "applied_action": hold}
        )

    def _transition(self, reply, action, extra):
        self._mask = reply["mask"]
        self._hedge_mask = reply["hedge_mask"]
        observed = reply["observation"]
        info = {
            "proposed_action": _plain(action),
            "refused_because": reply["refused_because"],
            "reward_exact": reply["reward"],
            "ledger_change": reply["ledger_change"],
            "terminal_close": reply["terminal_close"],
            "penalty": reply["penalty"],
            "net": reply["net"],
            "penalties": reply["penalties"],
            "close": reply["close"],
            "orders": dict(reply["orders"]),
            "gates": {check: dict(counts) for check, counts in reply["gates"].items()},
            "filled_lots": dict(reply["filled_lots"]),
            "row": observed["row"],
            "raw_observation": observed["raw"],
            **extra,
        }
        return (
            self._observation(observed),
            float(reply["reward_f32"]),
            bool(reply["terminated"]),
            bool(reply["truncated"]),
            info,
        )

    def attribution(self):
        """The episode's components in the evaluation's definitions, reconciled with the reward."""
        return self._native.attribution()

    def replace_rows(self, rows):
        """Replaces the rows under the running episode and replays it on them (#282). The new
        rows must keep the rows already read; a ValueError leaves the episode as it was."""
        self._native.replace_rows([tuple(row) for row in rows])
        self.bridge_info = dict(self._native.info())

    def close(self):
        self._native.close()

    @property
    def is_closed(self):
        """True once the worker has ended and freed its dataset."""
        return self._native.is_closed()


def _plain(action):
    return np.asarray(action).tolist()


def _bits(mask, count):
    return np.asarray([(mask >> index) & 1 == 1 for index in range(count)], dtype=bool)


class MaskableMarketMakingEnv(MarketMakingEnv):
    """A discrete encoding for maskable PPO: `(hold or quote, bid, ask, bid size, ask size)`.

    With `cancel_action` in the configuration the first value has a third choice, 2, the
    cancel: every resting or pending order is cancelled and nothing is sent.

    With `hedge_actions` in the configuration a sixth value is the hedge: 0 is no hedge, and
    `i + 1` hedges with the grid's size `i`. A hedge reduces the delivered position with a
    marketable order at the observed touch, on the side that reduces it, as the kernel's
    `HedgeActions::MarketableAtTouch` states; it replaces the resting orders, so when the sixth
    value is above 0 the first five are not read. Its mask comes from `hedge_mask`, which
    allows no hedge at a flat position or larger than the position, so only 0 is open there.

    With `quote_lives_ns` in the configuration a last value is the quote's life: 0 is no
    expiry, a plain quote, and `i + 1` is life `i`. Every value is always open, and only a
    quote reads it.

    Requires the mask policy, so a trainer that leaves the mask ends the environment.
    """

    def __init__(self, config, rows, expected_spec=None, seed=0):
        if config["invalid_action"] != "mask":
            raise ValueError("the maskable encoding needs the mask policy")
        super().__init__(config, rows, expected_spec, seed)
        dims = [3 if self._cancel else 2, self._offsets, self._offsets, self._sizes, self._sizes]
        if self._hedge:
            dims.append(1 + self._sizes)
        if self._lives:
            dims.append(1 + self._lives)
        self.action_space = gym.spaces.MultiDiscrete(dims)

    def action_masks(self):
        """The admissible values per dimension, concatenated.

        A quote needs a value in each dimension. When one has none, a quote is not offered
        and that dimension is left open, since a hold ignores it and an empty dimension
        cannot be sampled.
        """
        mask = self._mask
        bid = _bits(mask["bid"], self._offsets)
        ask = _bits(mask["ask"], self._offsets)
        size = _bits(mask["size"], self._sizes)
        quote = bool(mask["quote_ok"]) and bid.any() and ask.any() and size.any()
        # A cancel is allowed whenever a hold is.
        kinds = [True, quote, True] if self._cancel else [True, quote]
        parts = [np.asarray(kinds, dtype=bool)]
        for part in (bid, ask, size, size):
            parts.append(part if part.any() else np.ones_like(part))
        if self._hedge:
            # No hedge is always allowed.
            parts.append(
                np.concatenate(([True], _bits(self._hedge_mask, self._sizes)))
            )
        if self._lives:
            parts.append(np.ones(1 + self._lives, dtype=bool))
        return np.concatenate(parts)

    def _decode(self, action):
        values = [int(value) for value in np.asarray(action).ravel()]
        expected = 5 + int(self._hedge) + int(self._lives > 0)
        if len(values) != expected:
            raise ValueError(f"an action has {expected} values, not {len(values)}")
        if self._hedge and values[5] > 0:
            return "hedge", (values[5] - 1,)
        if values[0] == 0:
            return "hold", ()
        if self._cancel and values[0] == 2:
            return "cancel", ()
        if self._lives and values[-1] > 0:
            return "expiring_quote", (*values[1:5], values[-1] - 1)
        return "quote", tuple(values[1:5])


class BoxMarketMakingEnv(MarketMakingEnv):
    """A continuous encoding for SAC, each value in `[-1, 1]`.

    The first value above zero asks for a quote, and the others pick a bin of the bid
    offset, the ask offset, the bid size and the ask size. A Box cannot state a mask, so
    this requires the reject policy: what falls outside the mask is a charged hold. A value
    that is not finite or is out of range is an error and ends nothing but this call.

    With `hedge_actions` in the configuration a sixth value above zero asks for a hedge, of the
    grid's size at bin `(2 v - 1)` of the sizes, and the first five are not read; at or below
    zero they decide as without it. `2 v - 1` maps `(0, 1]` onto `(-1, 1]`, where the bin rule
    of the quote sizes applies, so a `v` just above zero is the smallest size and, with two
    sizes, `v` of 0.5 or more is the largest. A hedge reduces the delivered position on the side
    that reduces it; one outside `hedge_mask` (a flat position, or a size above the position) is
    a charged hold, as anything outside the mask is.

    With `cancel_action` in the configuration the last value above zero asks for a cancel,
    after a hedge and before a quote: every resting or pending order is cancelled and
    nothing is sent. It is the value after the hedge's, if any.

    With `quote_lives_ns` in the configuration a last value picks the quote's life, by the
    same bin rule over `1 + lives` bins: the first bin is no expiry, a plain quote, and bin
    `i + 1` is life `i`. Only a quote reads it.
    """

    def __init__(self, config, rows, expected_spec=None, seed=0):
        if config["invalid_action"] != "reject":
            raise ValueError("the continuous encoding needs the reject policy")
        super().__init__(config, rows, expected_spec, seed)
        self._length = 5 + int(self._hedge) + int(self._cancel) + int(self._lives > 0)
        shape = (self._length,)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    @staticmethod
    def _bin(value, count):
        return min(int((value + 1.0) / 2.0 * count), count - 1)

    def _decode(self, action):
        values = [float(value) for value in np.asarray(action).ravel()]
        expected = self._length
        if len(values) != expected:
            raise ValueError(f"an action has {expected} values, not {len(values)}")
        for value in values:
            if not math.isfinite(value) or abs(value) > 1.0:
                raise ValueError(f"an action value of {value} is not in [-1, 1]")
        if self._hedge and values[5] > 0.0:
            return "hedge", (self._bin(2.0 * values[5] - 1.0, self._sizes),)
        if self._cancel and values[5 + int(self._hedge)] > 0.0:
            return "cancel", ()
        if values[0] <= 0.0:
            return "hold", ()
        quote = (
            self._bin(values[1], self._offsets),
            self._bin(values[2], self._offsets),
            self._bin(values[3], self._sizes),
            self._bin(values[4], self._sizes),
        )
        life = self._bin(values[-1], 1 + self._lives) if self._lives else 0
        if life > 0:
            return "expiring_quote", (*quote, life - 1)
        return "quote", quote


class ResidualMarketMakingEnv(MarketMakingEnv):
    """A correction to a classical quote, for SAC: `(bid away, ask away, bid size, ask size)`.

    Each value is a fraction of its bound in `[-1, 1]`, and zero leaves the classical action
    as it is. A value that is not finite refuses the proposal and the classical action
    stands, as the log in `info["residual"]` says: a policy's output is never an error here,
    and the limits apply in the kernel to whatever comes out.
    """

    def __init__(self, config, rows, expected_spec=None, seed=0):
        if config.get("residual") is None:
            raise ValueError("a residual environment needs a residual configuration")
        if config.get("hedge_actions", False):
            raise ValueError("a residual environment takes no hedge actions: drop hedge_actions")
        if config.get("cancel_action", False):
            raise ValueError("a residual environment takes no cancel action: drop cancel_action")
        if config.get("quote_lives_ns") is not None:
            raise ValueError("a residual environment takes no quote life: drop quote_lives_ns")
        super().__init__(config, rows, expected_spec, seed)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def step(self, action):
        values = [float(value) for value in np.asarray(action).ravel()]
        if len(values) != 4:
            raise ValueError(f"a proposal has 4 values, not {len(values)}")
        reply = self._native.step_residual(*values)
        return self._transition(reply, action, {"residual": reply["residual"]})
