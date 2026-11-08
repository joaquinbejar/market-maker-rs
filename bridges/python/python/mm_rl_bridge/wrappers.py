"""Wrappers for experiments on decision frequency and memory.

`DecisionInterval` changes how often the policy decides, not the market: the rows, the
latencies and the fill model are those of the environment, and between two decisions the
resting orders stay. The policy sees only the observation of the row it decides on.
`EmaFeatures` and `history` give a policy memory over its own decisions; both start from
zero at every reset, so nothing crosses from one episode to the next.
"""

from decimal import Decimal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStackObservation

from .envs import MarketMakingEnv

ORDER_KINDS = ("accepted", "rejected", "canceled", "cancel_rejected", "fills")
# Positions in the observation's values, as `FeatureId` orders them.
MID_CHANGE = 1
IMBALANCE = 5


class DecisionInterval(gym.Wrapper):
    """A decision every `every` rows: the decided step, then holds until the next decision.

    The reward is the sum over the rows the decision covers, and `info` holds the exact sum
    (`reward_exact`), the last row's net result, the order reports summed over those rows
    and the number of rows (`rows`). The gate's cuts and refusals (`gates`) are the decided
    step's: a hold submits nothing, so the gate decides nothing on the rows held. An episode that ends between decisions ends there.
    """

    def __init__(self, env, every):
        if not isinstance(env.unwrapped, MarketMakingEnv):
            raise TypeError("a decision interval needs a market making environment")
        if int(every) != every or every < 1:
            raise ValueError(f"a decision interval is a whole number of rows, not {every}")
        super().__init__(env)
        self.every = int(every)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        total = Decimal(info["reward_exact"])
        orders = {kind: info["orders"][kind] for kind in ORDER_KINDS}
        rows = 1
        while rows < self.every and not (terminated or truncated):
            observation, held, terminated, truncated, last = self.env.unwrapped.step_hold()
            reward += held
            total += Decimal(last["reward_exact"])
            for kind in ORDER_KINDS:
                orders[kind] += last["orders"][kind]
            for key in ("net", "penalties", "close", "row", "raw_observation"):
                info[key] = last[key]
            rows += 1
        info["reward_exact"] = str(total)
        info["orders"] = orders
        info["rows"] = rows
        return observation, reward, terminated, truncated, info


class EmaFeatures(gym.ObservationWrapper):
    """The observation, then exponential moving averages over the decisions of the mid
    change and the imbalance, one per half-life in decisions.

    Each average starts at zero at reset and moves by `1 - 0.5 ** (1 / half_life)` of the
    way to the new value at each decision. Values stay in `[-1, 1]`, as the features do.
    """

    def __init__(self, env, half_lives=(2, 8, 32)):
        if not half_lives or any(h <= 0 for h in half_lives):
            raise ValueError(f"half-lives must be positive, not {half_lives}")
        super().__init__(env)
        self.weights = np.asarray([1 - 0.5 ** (1 / h) for h in half_lives], dtype=np.float64)
        width = 2 * len(half_lives)
        base = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((base.low, -np.ones(width, dtype=np.float32))),
            high=np.concatenate((base.high, np.ones(width, dtype=np.float32))),
            dtype=np.float32,
        )
        self._averages = np.zeros((2, len(half_lives)))

    def reset(self, **kwargs):
        self._averages = np.zeros_like(self._averages)
        return super().reset(**kwargs)

    def observation(self, observation):
        latest = np.asarray([observation[MID_CHANGE], observation[IMBALANCE]])[:, None]
        self._averages += self.weights[None, :] * (latest - self._averages)
        extra = np.clip(self._averages.ravel(), -1.0, 1.0).astype(np.float32)
        return np.concatenate((np.asarray(observation, dtype=np.float32), extra))


def history(env, length):
    """The last `length` decision observations, oldest first, flattened; zeros before the
    first decision of an episode."""
    if int(length) != length or length < 1:
        raise ValueError(f"a history is a whole number of decisions, not {length}")
    return FlattenObservation(FrameStackObservation(env, int(length), padding_type="zero"))


class ChosenDuration(gym.Wrapper):
    """The policy chooses with each action how many rows its quote rests before it decides
    again: one more value in `[-1, 1]` at the end of the action, binned onto `choices`.

    The rest of the action goes to the environment; then, as `DecisionInterval` does, the
    resting orders stay for the chosen number of rows, the reward is the sum over them, and
    `info` holds `rows` and the chosen `duration`. A cancel takes effect only after the
    environment's cancel latency, so fills in flight stand. An episode that ends inside a
    duration ends there.
    """

    def __init__(self, env, choices=(1, 5, 10, 30)):
        if not isinstance(env.unwrapped, MarketMakingEnv):
            raise TypeError("a chosen duration needs a market making environment")
        if not choices or any(int(c) != c or c < 1 for c in choices):
            raise ValueError(f"durations are positive whole numbers of rows, not {choices}")
        base = env.action_space
        if not isinstance(base, gym.spaces.Box):
            raise TypeError("a chosen duration extends a Box action")
        super().__init__(env)
        self.choices = tuple(int(c) for c in choices)
        self.action_space = gym.spaces.Box(
            low=np.concatenate((base.low, [-1.0])).astype(np.float32),
            high=np.concatenate((base.high, [1.0])).astype(np.float32),
            dtype=np.float32,
        )

    def duration_of(self, value):
        """The choice for a value in `[-1, 1]`: equal bins, the last one closed."""
        index = int((float(value) + 1.0) / 2.0 * len(self.choices))
        return self.choices[min(max(index, 0), len(self.choices) - 1)]

    def step(self, action):
        values = np.asarray(action, dtype=np.float32).ravel()
        if values.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"an action has {self.action_space.shape[0]} values")
        duration = self.duration_of(values[-1])
        observation, reward, terminated, truncated, info = self.env.step(values[:-1])
        total = Decimal(info["reward_exact"])
        orders = {kind: info["orders"][kind] for kind in ORDER_KINDS}
        rows = 1
        while rows < duration and not (terminated or truncated):
            observation, held, terminated, truncated, last = self.env.unwrapped.step_hold()
            reward += held
            total += Decimal(last["reward_exact"])
            for kind in ORDER_KINDS:
                orders[kind] += last["orders"][kind]
            for key in ("net", "penalties", "close", "row", "raw_observation"):
                info[key] = last[key]
            rows += 1
        info.update(reward_exact=str(total), orders=orders, rows=rows, duration=duration)
        return observation, reward, terminated, truncated, info
