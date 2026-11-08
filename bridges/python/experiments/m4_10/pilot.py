"""Timing pilot of the quote duration comparison of #110, on training days only; nothing is
selected.

    python experiments/m4_10/pilot.py <data dir> <decisions> <seed>

trains the full arm's policy shape (PPO, a Box action with the chosen duration) for
`decisions` decisions on calm training day 1 at each latency of `design.json`, and prints the
wall time, the rows covered and the durations chosen.
"""

import json
import pathlib
import sys
import time
from collections import Counter

import torch
from stable_baselines3 import PPO

from mm_rl_bridge import BoxMarketMakingEnv
from mm_rl_bridge.wrappers import ChosenDuration

HERE = pathlib.Path(__file__).resolve().parent


def config(design, latency):
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": design["simulator"]["rows_per_day"], "fee_rate": "0.0002",
        "initial_capital": run["initial_capital"], "latency": latency,
        "fill_model": run["fill_model"], "limits": run["limits"],
        "bid_offsets": grid["bid_offsets"], "ask_offsets": grid["ask_offsets"],
        "sizes": grid["sizes"], "scales": design["observation_scales"], "f32_unit": "1",
        "invalid_action": "reject", "invalid_action_penalty": "0", "terminal": run["terminal"],
    }


class Counting(ChosenDuration):
    def __init__(self, env, choices):
        super().__init__(env, choices)
        self.rows = 0
        self.chosen = Counter()

    def step(self, action):
        out = super().step(action)
        self.rows += out[4]["rows"]
        self.chosen[out[4]["duration"]] += 1
        return out


def main():
    data_dir, decisions, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    torch.set_num_threads(1)
    design = json.loads((HERE / "design.json").read_text())
    rows = [tuple(r) for r in json.loads(
        (pathlib.Path(data_dir) / "train" / "calm" / "day-1.json").read_text())]
    h = design["hyperparameters"]
    kwargs = {k: h[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef")}
    for name, latency in design["execution"]["latencies"].items():
        env = Counting(BoxMarketMakingEnv(config(design, latency), rows, seed=seed),
                       design["duration"]["choices_rows"])
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kwargs)
        started = time.perf_counter()
        model.learn(total_timesteps=decisions)
        seconds = time.perf_counter() - started
        env.close()
        print(json.dumps({"latency_ms": name, "decisions": decisions, "rows": env.rows,
                          "seconds": round(seconds, 1),
                          "per_decision_ms": round(1000 * seconds / decisions, 3),
                          "estimate_2m_minutes": round(seconds / decisions * 2_000_000 / 60, 1),
                          "durations": dict(sorted(env.chosen.items()))}), flush=True)


if __name__ == "__main__":
    main()
