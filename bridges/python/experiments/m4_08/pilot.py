"""Timing pilot of the constrained comparison of #105, on training days only.

    MUJOCO_GL=disable python experiments/m4_08/pilot.py <#102 data dir> <out dir> <algorithm> <steps> <seed>

trains one OmniSafe algorithm (`PPO`, `PPOLag` or `CPO`, configured as `design.json` says) on
the pooled training days and prints the wall time; nothing is selected. The fill model is
the queue position one at one lot per millisecond, because the bridge does not take the
rational rate yet (#269); the time per step does not depend on it.
"""

import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def config(design, fee="0.0002"):
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": 8640, "fee_rate": fee, "initial_capital": run["initial_capital"],
        "latency": run["latency"],
        "fill_model": {"kind": "queue_position", "rate": "1", "min_queue_ms": 5000},
        "limits": run["limits"], "bid_offsets": grid["bid_offsets"],
        "ask_offsets": grid["ask_offsets"], "sizes": grid["sizes"],
        "scales": design["observation_scales"], "f32_unit": "1",
        "invalid_action": "reject", "invalid_action_penalty": "0", "terminal": run["terminal"],
    }


def main():
    data, out, algorithm, steps, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    design = json.loads((HERE / "design.json").read_text())
    base = json.loads((HERE.parent / "m4_05" / "digests.json").read_text())
    train = [r for r in base["days"] if r["split"] == "train"]
    job = {"data_dir": data, "config": config(design), "seed": seed, "penalty": None,
           "days": [["train", r["regime"], r["day"], f"day-{r['day']}.json"] for r in train],
           "digests": {f"train/{r['regime']}/day-{r['day']}.json": r["dataset_digest"] for r in train}}
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    job_path = pathlib.Path(out) / "job.json"
    job_path.write_text(json.dumps(job))
    os.environ["M4_08_JOB"] = str(job_path)
    sys.path.insert(0, str(HERE))
    import cmdp  # noqa: F401  (registers the environment)
    import omnisafe

    cfgs = {"seed": seed,
            "train_cfgs": {"total_steps": steps, "device": "cpu", "torch_threads": 1,
                           "vector_env_nums": 1, "parallel": 1},
            "algo_cfgs": {"steps_per_epoch": design["algorithm_cfgs"]["steps_per_epoch"],
                          "update_iters": design["algorithm_cfgs"]["update_iters"]},
            "logger_cfgs": {"use_wandb": False, "use_tensorboard": False, "log_dir": out}}
    if algorithm == "PPOLag":
        cfgs["algo_cfgs"].update({"standardized_rew_adv": False, "standardized_cost_adv": False})
        cfgs["lagrange_cfgs"] = {"cost_limit": 0.25 * 8640}
    elif algorithm == "CPO":
        cfgs["algo_cfgs"]["cost_limit"] = 0.25 * 8640
    started = time.perf_counter()
    agent = omnisafe.Agent(algorithm, "MarketMaking-v0", custom_cfgs=cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()
    seconds = time.perf_counter() - started
    print(json.dumps({"algorithm": algorithm, "steps": steps, "seconds": round(seconds, 1),
                      "per_step_ms": round(1000 * seconds / steps, 3),
                      "estimate_2m_minutes": round(seconds / steps * 2_000_000 / 60, 1),
                      "last_epoch": {"return": float(ep_ret), "cost": float(ep_cost),
                                     "length": float(ep_len)}}))


if __name__ == "__main__":
    main()
