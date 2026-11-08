"""The market making environment as an OmniSafe CMDP, for the comparison of #105.

Runs in the experiment's own venv (`omnisafe-lock.txt`). The job comes from the JSON file
named by `M4_08_JOB`: the days and their rows' directory, the environment's configuration,
the seed and, for the penalized arm, `penalty`. Each reset draws one of the days from the
seed. The reward is the kernel's (less `penalty` times the cost for the penalized arm), and
the cost is `c_inventory + c_notional` of `design.json`, measured after the step from the
delivered position and the observed row's mid. Every cost is also in `info` apart.
"""

from __future__ import annotations

import json
import os
import pathlib
from decimal import Decimal
from typing import Any, ClassVar

import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register

from mm_rl_bridge import BoxMarketMakingEnv

from costs import costs  # noqa: E402  (the experiment's own module)

def load_job():
    return json.loads(pathlib.Path(os.environ["M4_08_JOB"]).read_text())


@env_register
class MarketMaking(CMDP):
    """One environment per day of the job; a reset draws a day."""

    _support_envs: ClassVar[list[str]] = ["MarketMaking-v0"]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1
    metadata: ClassVar[dict[str, int]] = {}

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        job = load_job()
        self._penalty = float(job.get("penalty") or 0.0)
        self._tick = Decimal(job["config"]["tick_size"])
        self._rng = np.random.default_rng(job["seed"])
        self._days = []
        for split, regime, day, name in job["days"]:
            path = pathlib.Path(job["data_dir"]) / split / regime / name
            rows = [tuple(row) for row in json.loads(path.read_text())]
            env = BoxMarketMakingEnv(job["config"], rows)
            if env.bridge_info["dataset_digest"] != job["digests"][f"{split}/{regime}/{name}"]:
                raise ValueError(f"{path} has other rows than its digest")
            self._days.append((env, rows))
        env = self._days[0][0]
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._max_episode_steps = job["config"]["max_steps"]
        self._current = self._days[0]
        self.env_spec_log = {}

    def step(self, action):
        env, rows = self._current
        values = np.clip(np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
                         -1.0, 1.0)
        observation, reward, terminated, truncated, info = env.step(values)
        cost = costs(info, rows, self._tick)
        constrained = cost["c_inventory"] + cost["c_notional"]
        obs = torch.as_tensor(observation, dtype=torch.float32)
        out = {**cost, "net": info["net"], "refused_because": info["refused_because"],
               "final_observation": obs}
        return (obs, torch.as_tensor(reward - self._penalty * constrained, dtype=torch.float32),
                torch.as_tensor(constrained, dtype=torch.float32),
                torch.as_tensor(terminated), torch.as_tensor(truncated), out)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._current = self._days[int(self._rng.integers(len(self._days)))]
        observation, info = self._current[0].reset(seed=int(self._rng.integers(2**63)))
        return torch.as_tensor(observation, dtype=torch.float32), info

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def render(self) -> Any:
        return None

    def close(self) -> None:
        for env, _ in self._days:
            env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps
