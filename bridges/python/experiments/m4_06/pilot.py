"""Timing pilot of the comparison of #103, on training days only; nothing is selected.

    python experiments/m4_06/pilot.py <data dir> <arm> <decisions> <seed> [<every>]

trains one arm (`mlp` or `lstm`) for `decisions` decisions at a decision every `every` rows
(1 by default), on calm
training day 1, and prints the wall time. The fill model is the queue position one at one lot
per millisecond, because the bridge does not take the rational rate yet (#269); the time per
decision does not depend on it.
"""

import json
import pathlib
import sys
import time

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mm_rl_bridge import BoxMarketMakingEnv
from mm_rl_bridge.wrappers import DecisionInterval

HERE = pathlib.Path(__file__).resolve().parent
SCALES = {"spread_ticks": "10", "mid_change_ticks": "10", "row_gap_ms": "2000",
          "size_lots": "10", "position": "20", "entry_gap": "1"}


def config(design):
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": design["simulator"]["rows_per_day"], "fee_rate": "0.0002",
        "initial_capital": run["initial_capital"], "latency": run["latency"],
        "fill_model": {"kind": "queue_position", "rate": "1", "min_queue_ms": 5000},
        "limits": run["limits"], "bid_offsets": grid["bid_offsets"],
        "ask_offsets": grid["ask_offsets"], "sizes": grid["sizes"], "scales": SCALES,
        "f32_unit": "1", "invalid_action": "reject", "invalid_action_penalty": "0",
        "terminal": run["terminal"],
    }


def main():
    data_dir, arm, decisions, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    every = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    torch.set_num_threads(1)
    design = json.loads((HERE / "design.json").read_text())
    rows = [tuple(r) for r in json.loads(
        (pathlib.Path(data_dir) / "train" / "calm" / "day-1.json").read_text())]
    env = DecisionInterval(BoxMarketMakingEnv(config(design), rows, seed=seed), every)
    h = design["hyperparameters"]
    kwargs = {k: h[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef")}
    algorithm, policy = (RecurrentPPO, "MlpLstmPolicy") if arm == "lstm" else (PPO, "MlpPolicy")
    model = algorithm(policy, env, seed=seed, device="cpu", verbose=0, **kwargs)
    started = time.perf_counter()
    model.learn(total_timesteps=decisions)
    seconds = time.perf_counter() - started
    env.close()
    print(json.dumps({"arm": arm, "every": every, "decisions": decisions, "seconds": round(seconds, 1),
                      "per_decision_ms": round(1000 * seconds / decisions, 3),
                      "estimate_2m_minutes": round(seconds / decisions * 2_000_000 / 60, 1)}))


if __name__ == "__main__":
    main()
