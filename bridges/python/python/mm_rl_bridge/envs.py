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
        observed = reply["observation"]
        info = {"row": observed["row"], "raw_observation": observed["raw"], "seed": seed}
        return self._observation(observed), info

    def _decode(self, action):
        """Returns `(kind, indices)` for the worker, from the policy's action."""
        raise NotImplementedError

    def step(self, action):
        kind, indices = self._decode(action)
        if kind == "hold":
            reply = self._native.step_hold()
        elif kind == "quote":
            reply = self._native.step_quote(*indices)
        else:
            raise ValueError(f"unknown action kind {kind!r}")
        self._mask = reply["mask"]
        observed = reply["observation"]
        info = {
            "proposed_action": _plain(action),
            "decoded_action": {"kind": kind, "indices": list(indices)},
            "applied_action": (
                {"kind": kind, "indices": list(indices)}
                if reply["applied"]
                else {"kind": "hold", "indices": []}
            ),
            "refused_because": reply["refused_because"],
            "reward_exact": reply["reward"],
            "ledger_change": reply["ledger_change"],
            "terminal_close": reply["terminal_close"],
            "penalty": reply["penalty"],
            "net": reply["net"],
            "penalties": reply["penalties"],
            "close": reply["close"],
            "row": observed["row"],
            "raw_observation": observed["raw"],
        }
        return (
            self._observation(observed),
            float(reply["reward_f32"]),
            bool(reply["terminated"]),
            bool(reply["truncated"]),
            info,
        )

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

    Requires the mask policy, so a trainer that leaves the mask ends the environment.
    """

    def __init__(self, config, rows, expected_spec=None, seed=0):
        if config["invalid_action"] != "mask":
            raise ValueError("the maskable encoding needs the mask policy")
        super().__init__(config, rows, expected_spec, seed)
        self.action_space = gym.spaces.MultiDiscrete(
            [2, self._offsets, self._offsets, self._sizes, self._sizes]
        )

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
        parts = [np.asarray([True, quote], dtype=bool)]
        for part in (bid, ask, size, size):
            parts.append(part if part.any() else np.ones_like(part))
        return np.concatenate(parts)

    def _decode(self, action):
        values = [int(value) for value in np.asarray(action).ravel()]
        if len(values) != 5:
            raise ValueError(f"an action has 5 values, not {len(values)}")
        if values[0] == 0:
            return "hold", ()
        return "quote", tuple(values[1:])


class BoxMarketMakingEnv(MarketMakingEnv):
    """A continuous encoding for SAC, each value in `[-1, 1]`.

    The first value above zero asks for a quote, and the others pick a bin of the bid
    offset, the ask offset, the bid size and the ask size. A Box cannot state a mask, so
    this requires the reject policy: what falls outside the mask is a charged hold. A value
    that is not finite or is out of range is an error and ends nothing but this call.
    """

    def __init__(self, config, rows, expected_spec=None, seed=0):
        if config["invalid_action"] != "reject":
            raise ValueError("the continuous encoding needs the reject policy")
        super().__init__(config, rows, expected_spec, seed)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    @staticmethod
    def _bin(value, count):
        return min(int((value + 1.0) / 2.0 * count), count - 1)

    def _decode(self, action):
        values = [float(value) for value in np.asarray(action).ravel()]
        if len(values) != 5:
            raise ValueError(f"an action has 5 values, not {len(values)}")
        for value in values:
            if not math.isfinite(value) or abs(value) > 1.0:
                raise ValueError(f"an action value of {value} is not in [-1, 1]")
        if values[0] <= 0.0:
            return "hold", ()
        return "quote", (
            self._bin(values[1], self._offsets),
            self._bin(values[2], self._offsets),
            self._bin(values[3], self._sizes),
            self._bin(values[4], self._sizes),
        )
