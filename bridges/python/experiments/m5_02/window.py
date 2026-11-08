"""The bridge's observation as a window of past decisions, for the extractors of `nets` (#107).

The bridge observes the 11 features of the M4 specification at each decision, as normalized
values then presence flags. This wrapper keeps the last `L` decisions' market features
(`MARKET`), newest first, with their ages in decisions, and passes the agent's own state
(`AGENT`) through as scalars, in the `nets.Layout` order. A missing market feature is 0 in
the window, and its decision is still valid: the presence flag of each market feature goes in
as a value of its own, so nothing is invented and nothing is dropped.

Nothing outside the bridge's observation enters: the window is the specification's own
features at earlier decisions, and it is cleared at every reset, so no memory crosses
episodes or splits.
"""

import collections

import gymnasium as gym
import numpy as np

# Indices in the M4 specification's 11 features (`market_maker_rs::rl::spec`).
MARKET = (0, 1, 2, 3, 4, 5)  # spread, mid change, row gap, bid size, ask size, imbalance
AGENT = (6, 7, 8, 9, 10)  # position, entry gap, open orders, awaiting cancel, steps left
FEATURES = 11


def layout_dims(buffer):
    channels = 2 * len(MARKET)
    scalars = 2 * len(AGENT)
    return {"buffer": buffer, "channels": channels, "scalars": scalars}


class WindowEnv(gym.Wrapper):
    """`[ages, valid, values (market values and flags), scalars (agent values and flags)]`."""

    def __init__(self, env, buffer):
        super().__init__(env)
        if env.observation_space.shape != (2 * FEATURES,):
            raise ValueError("the wrapped environment must observe the 11 M4 features")
        self.buffer = buffer
        dims = layout_dims(buffer)
        size = (2 + dims["channels"]) * buffer + dims["scalars"]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self._items = collections.deque(maxlen=buffer)

    def _market(self, obs):
        values, flags = obs[:FEATURES], obs[FEATURES:]
        return np.concatenate([values[list(MARKET)], flags[list(MARKET)]])

    def _scalars(self, obs):
        values, flags = obs[:FEATURES], obs[FEATURES:]
        return np.concatenate([values[list(AGENT)], flags[list(AGENT)]])

    def _window(self, obs):
        L = self.buffer
        channels = 2 * len(MARKET)
        out = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i, item in enumerate(reversed(self._items)):
            out[i] = float(i)
            out[L + i] = 1.0
            out[2 * L + i * channels : 2 * L + (i + 1) * channels] = item
        out[(2 + channels) * L :] = self._scalars(obs)
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._items.clear()
        self._items.append(self._market(obs))
        return self._window(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._items.append(self._market(obs))
        return self._window(obs), reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()
