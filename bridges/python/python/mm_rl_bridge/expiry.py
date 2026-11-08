"""A quote that expires by itself, from the cancel-only action.

`ExpiringQuote` lets a policy place a quote that is cancelled `rows` decisions later with
no decision of the policy in between. The environment decides on every row, so a quote
sent at the decision on row `t` rests while the wrapper holds, and the wrapper sends the
cancel at the decision on row `t + rows`. The policy decides again on the next row.

The timing, in venue time: the quote's orders arrive after the submission latency and its
cancel after the cancel latency, so the quote rests `rows` rows plus the cancel latency less
the submission latency. With equal latencies it rests exactly `rows` rows: with none, it can
fill on rows `t + 1` to `t + rows`, and not after. A fill before the cancel arrives stands,
and a cancel that reaches an order already filled is rejected (`CancelRejected`).
"""

import gymnasium as gym

from .envs import MarketMakingEnv


class ExpiringQuote(gym.Wrapper):
    """Cancels each quote `rows` decisions after it was sent, without asking the policy.

    One step of the wrapper is one decision of the policy. A quote the environment applied
    is followed by `rows - 1` holds and a cancel, taken by the wrapper; any other action is
    a single step. The reward is the sum of the steps' rewards, and the episode ends as
    soon as one of them ends it. `info["expiry"]` records the steps the wrapper took: their
    kinds, exact rewards, and whether the cancel was sent.

    Needs `cancel_action` in the configuration, and the environment itself, not a wrapper of
    it, since the wrapper's own steps bypass the policy's encoding.
    """

    def __init__(self, env, rows):
        if not isinstance(env, MarketMakingEnv):
            raise TypeError("ExpiringQuote wraps a MarketMakingEnv directly")
        if not env.bridge_info["cancel_action"]:
            raise ValueError("an expiring quote needs cancel_action in the configuration")
        if env.bridge_info["quote_lives_ns"]:
            raise ValueError("drop quote_lives_ns: its quotes already expire at the venue")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(f"rows must be a positive integer, got {rows!r}")
        super().__init__(env)
        self.rows = rows

    def action_masks(self):
        """The environment's mask: the policy's decision is the environment's."""
        return self.env.action_masks()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        steps = [{"kind": info["applied_action"]["kind"], "reward_exact": info["reward_exact"]}]
        cancelled = False
        if info["applied_action"]["kind"] == "quote":
            for held in range(self.rows):
                if terminated or truncated:
                    break
                kind = "cancel" if held == self.rows - 1 else "hold"
                observation, more, terminated, truncated, info = self.env.act(kind)
                reward += more
                steps.append({"kind": kind, "reward_exact": info["reward_exact"]})
                cancelled = kind == "cancel"
        info = dict(info)
        info["expiry"] = {"rows": self.rows, "steps": steps, "cancelled": cancelled}
        return observation, reward, terminated, truncated, info
