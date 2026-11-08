"""The external constrained optimizer on a toy CMDP whose optimum is known (#105).

    MUJOCO_GL=disable python experiments/m4_08/toy_cmdp.py <out dir> <algorithm> <steps> <seed>

One state, an action `a` and `x = 1 / (1 + exp(-a))`, a reward of `sqrt(x) - 0.8 x` and a cost
of `x` per step, over episodes of 20 steps. The reward peaks inside the action space, at
`x = 1 / 1.6 ** 2` (0.39, a cost of 7.8 per episode), so no action runs to a bound where the
gradient vanishes. With a cost limit of 6 per episode the constrained optimum is `x = 0.3` at
every step: a return of `20 * (sqrt(0.3) - 0.24)` (6.15) and a cost of 6, with a Lagrange
multiplier of `1 / (2 * sqrt(0.3)) - 0.8` (0.113). `PPOLag` and `CPO` must end near a cost of
6, `PPO` near 7.8. Runs in the experiment's own
venv (`omnisafe-lock.txt`), not the bridge's.
"""

from __future__ import annotations

import json
import sys
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register

EPISODE = 20
COST_LIMIT = 6.0


@env_register
class Toy(CMDP):
    """A reward of `sqrt(x) - 0.8 x` and a cost of `x = 1 / (1 + exp(-a))` per step."""

    _support_envs: ClassVar[list[str]] = ["Toy-v0"]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1
    metadata: ClassVar[dict[str, int]] = {}

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self._action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,))
        self._max_episode_steps = EPISODE
        self._count = 0
        self.env_spec_log = {}

    def step(self, action):
        self._count += 1
        a = float(torch.as_tensor(action).reshape(-1)[0])
        x = 1.0 / (1.0 + np.exp(-a))
        reward = torch.as_tensor(x ** 0.5 - 0.8 * x, dtype=torch.float32)
        cost = torch.as_tensor(x, dtype=torch.float32)
        obs = torch.zeros(1, dtype=torch.float32)
        truncated = torch.as_tensor(self._count >= EPISODE)
        return obs, reward, cost, torch.as_tensor(False), truncated, {
            "final_observation": obs}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._count = 0
        return torch.zeros(1, dtype=torch.float32), {}

    def set_seed(self, seed: int) -> None:
        np.random.seed(seed)

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None

    @property
    def max_episode_steps(self) -> int:
        return EPISODE


def main():
    import omnisafe

    out, algorithm, steps, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    # Optional `key=value` overrides of `algo_cfgs`, for the configuration checks.
    extra = dict(arg.split("=", 1) for arg in sys.argv[5:])
    cfgs = {
        "seed": seed,
        "train_cfgs": {"total_steps": steps, "device": "cpu", "torch_threads": 1,
                       "vector_env_nums": 1, "parallel": 1},
        "algo_cfgs": {"steps_per_epoch": 2000, "update_iters": 10},
        "logger_cfgs": {"use_wandb": False, "use_tensorboard": False, "log_dir": out},
    }
    for key, value in extra.items():
        cfgs["algo_cfgs"][key] = json.loads(value)
    if algorithm == "CPO":
        cfgs["algo_cfgs"]["cost_limit"] = COST_LIMIT
    elif algorithm != "PPO":
        cfgs["lagrange_cfgs"] = {"cost_limit": COST_LIMIT}
    agent = omnisafe.Agent(algorithm, "Toy-v0", custom_cfgs=cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()
    print(json.dumps({"algorithm": algorithm, "steps": steps, "seed": seed, "overrides": extra,
                      "episode_return": float(ep_ret), "episode_cost": float(ep_cost),
                      "episode_length": float(ep_len), "cost_limit": COST_LIMIT,
                      "constrained_optimum": {"return": EPISODE * (0.3 ** 0.5 - 0.24),
                                              "cost": COST_LIMIT},
                      "unconstrained_optimum": {"return": EPISODE * (0.625 - 0.8 * 0.390625),
                                                "cost": EPISODE * 0.390625}}))


if __name__ == "__main__":
    main()
