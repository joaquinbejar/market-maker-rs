"""Runs a step of the out-of-sample comparison of #102 from its committed design.

`python experiments/m4_05/run.py <step> <data dir> <out dir> <commit>`, from `bridges/python`,
after `qr_days` wrote the rows. Every day's rows are checked against `digests.json` before
use. This file implements the reduced step, which trains on the training days and scores on
the validation days only; it never opens the holdout's directory.

Everything here is synthetic. One synthetic simulator, made-up parameters and 24 days per
regime bound what it can show.
"""

import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
import pathlib
import sys
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from mm_rl_bridge import MaskableMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import ZeroResidual, evaluate, versions

HERE = pathlib.Path(__file__).resolve().parent
DAY_MS = 86_400_000
SCALES = {
    "spread_ticks": "10",
    "mid_change_ticks": "10",
    "row_gap_ms": "20000",
    "size_lots": "10",
    "position": "20",
    "entry_gap": "1",
}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_design():
    design = json.loads((HERE / "design.json").read_text())
    digests = json.loads((HERE / "digests.json").read_text())
    if digests["design_version"] != design["design_version"]:
        raise ValueError("the digests belong to another design")
    by_day = {(d["regime"], d["day"]): d for d in digests["days"]}
    return design, by_day


def days_of(design, split):
    first, last = design["splits"][split]
    return list(range(first, last + 1))


def read_rows(data_dir, design, digests, regime, day, split):
    if split == "holdout":
        raise PermissionError("this step does not read the holdout")
    record = digests[(regime, day)]
    if record["split"] != split:
        raise ValueError(f"{regime} day {day} is in {record['split']}, not {split}")
    path = pathlib.Path(data_dir) / split / regime / f"day-{day}.json"
    rows = [tuple(row) for row in json.loads(path.read_text())]
    return rows, record


def execution(design, variant):
    chosen = dict(design["execution"]["main"])
    if variant != "main":
        chosen.update(
            {k: v for k, v in design["execution"]["ablations"][variant].items() if k in chosen}
        )
    return chosen


def base_config(design, variant, policy, grid):
    kernel = design["kernel"]
    run = execution(design, variant)
    return {
        "tick_size": kernel["tick_size"],
        "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"],
        "max_order_size": kernel["max_order_size"],
        "max_steps": design["simulator"]["rows_per_day"],
        "fee_rate": design["execution"]["fee_rate"],
        "initial_capital": design["execution"]["initial_capital"],
        "latency": run["latency"],
        "fill_model": run["fill_model"],
        "limits": design["execution"]["limits"],
        "bid_offsets": grid["bid_offsets"],
        "ask_offsets": grid["ask_offsets"],
        "sizes": grid["sizes"],
        "scales": SCALES,
        "f32_unit": "1",
        "invalid_action": policy,
        "invalid_action_penalty": "0",
        "terminal": design["execution"]["terminal"],
    }


def baseline_config(kind, params, day):
    """A classical strategy's configuration for one day, its session starting with the day."""
    start = (day - 1) * DAY_MS
    if kind == "avellaneda_stoikov":
        return {
            "kind": "avellaneda_stoikov",
            "risk_aversion": params["risk_aversion"],
            "volatility": params["volatility"],
            "order_intensity": params["order_intensity"],
            "session_start_ms": start,
            "horizon_ms": params["horizon_ms"],
            "size": params["size"],
        }
    return {
        "kind": "glft",
        "risk_aversion": params["risk_aversion"],
        "order_intensity_k": params["order_intensity_k"],
        "intensity_scale_a": params["intensity_scale_a"],
        "max_inventory_lots": params["max_inventory_lots"],
        "terminal_time": params["terminal_time"],
        "min_spread": params["min_spread"],
        "volatility": params["volatility"],
        "session_start_ms": start,
    }


def residual_config(design, variant, kind, params, day, bounds):
    grid = {"bid_offsets": [0], "ask_offsets": [0], "sizes": [1]}
    config = base_config(design, variant, "reject", grid)
    config["residual"] = {
        "baseline": baseline_config(kind, params, day),
        "price_ticks": bounds["price_ticks"],
        "size_lots": bounds["size_lots"],
        "default_order_size": "1",
    }
    return config


SECONDS_PER_YEAR = 31_536_000


def volatility(data_dir, design, digests, regime):
    """The design's rule, over the training days only: absolute price units per square root
    of a 365 day year, from the changes of the mid between consecutive rows of a day."""
    tick = Decimal(design["kernel"]["tick_size"])
    changes = []
    for day in days_of(design, "train"):
        rows, _ = read_rows(data_dir, design, digests, regime, day, "train")
        mids = [Decimal(row[1] + row[3]) / 2 * tick for row in rows]
        changes.extend(float(b - a) for a, b in zip(mids, mids[1:]))
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)
    seconds = design["simulator"]["sample_ms"] / 1000
    return f"{(variance * SECONDS_PER_YEAR / seconds) ** 0.5:.4f}"


def grid_points(grid, sigma=None):
    keys = sorted(grid)
    points = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    for point in points:
        if point.get("volatility") == "from the volatility rule":
            point["volatility"] = sigma
    return points


def checked(env, record):
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{record['regime']} day {record['day']} has other rows than digests.json")
    return env


class DayCycle(gym.Env):
    """One environment per training day; every reset draws a day from the run's seed."""

    metadata = {"render_modes": []}

    def __init__(self, envs, seed):
        super().__init__()
        self._envs = envs
        self._rng = np.random.default_rng(seed)
        self._current = envs[0]
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current = self._envs[int(self._rng.integers(len(self._envs)))]
        return self._current.reset(seed=int(self._rng.integers(2**63)))

    def step(self, action):
        return self._current.step(action)

    def action_masks(self):
        return self._current.action_masks()

    def close(self):
        for env in self._envs:
            env.close()


class Constant:
    """A fixed action at every decision: a hold, or one quote of the grid."""

    def __init__(self, action):
        self._action = np.asarray(action)

    def predict(self, observation, deterministic=True, action_masks=None):
        return self._action, None


def build_env(job, data_dir, design, digests, day, split):
    rows, record = read_rows(data_dir, design, digests, job["regime"], day, split)
    if job["encoding"] == "maskable":
        env = MaskableMarketMakingEnv(job["config"], rows)
    else:
        config = residual_config(
            design, job["variant"], job["baseline_kind"], job["baseline"], day, job["bounds"]
        )
        env = ResidualMarketMakingEnv(config, rows)
    return checked(env, record)


def train_job(job):
    """One training run, in its own process on one thread."""
    torch.set_num_threads(1)
    design, digests = load_design()
    set_random_seed(job["seed"])
    days = days_of(design, "train")
    envs = [build_env(job, job["data_dir"], design, digests, day, "train") for day in days]
    cycle = DayCycle(envs, job["seed"])
    model_class = MaskablePPO if job["algorithm"] == "maskable_ppo" else SAC
    start = time.perf_counter()
    model = model_class(
        "MlpPolicy",
        DummyVecEnv([lambda: Monitor(cycle)]),
        seed=job["seed"],
        device="cpu",
        **job["hyperparameters"],
    )
    model.learn(total_timesteps=job["steps"], progress_bar=False)
    seconds = time.perf_counter() - start
    path = pathlib.Path(job["path"])
    model.save(path)
    first = envs[0]
    metadata = {
        "algorithm": job["algorithm"],
        "regime": job["regime"],
        "variant": job["variant"],
        "seed": job["seed"],
        "steps": job["steps"],
        "hyperparameters": job["hyperparameters"],
        "spec": list(first.spec_id),
        "kernel_fingerprint": first.bridge_info["kernel_fingerprint"],
        "config": first.config,
        "training_days": days,
        "training_digests": [env.bridge_info["dataset_digest"] for env in envs],
        "commit": job["commit"],
        "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(),
        "train_seconds": round(seconds, 2),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")),
    }
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def same_but_session(a, b):
    """True when two configurations differ in nothing but the classical session's start."""

    def strip(config):
        config = json.loads(json.dumps(config))
        baseline = (config.get("residual") or {}).get("baseline")
        if baseline is not None:
            baseline.pop("session_start_ms", None)
        return config

    return strip(a) == strip(b)


def load_frozen(metadata, env):
    path = pathlib.Path(metadata["path"])
    if sha256(path.with_suffix(".zip")) != metadata["checkpoint_sha256"]:
        raise ValueError("a checkpoint changed after training")
    if tuple(metadata["spec"]) != env.spec_id:
        raise ValueError("a checkpoint of another specification")
    if metadata["kernel_fingerprint"] != env.bridge_info["kernel_fingerprint"]:
        raise ValueError("a checkpoint of other execution rules")
    if not same_but_session(metadata["config"], env.config):
        raise ValueError("a checkpoint of another configuration")
    model_class = MaskablePPO if metadata["algorithm"] == "maskable_ppo" else SAC
    model = model_class.load(path.with_suffix(".zip"), device="cpu")
    model.policy.set_training_mode(False)
    return model


def evaluate_job(job):
    """One arm on one day, over the evaluation seeds."""
    torch.set_num_threads(1)
    design, digests = load_design()
    env = build_env(job, job["data_dir"], design, digests, job["day"], job["split"])
    try:
        if job["policy"]["kind"] == "constant":
            policy = Constant(job["policy"]["action"])
        elif job["policy"]["kind"] == "zero_residual":
            policy = ZeroResidual()
        else:
            policy = load_frozen(job["policy"]["metadata"], env)
        results = evaluate(policy, env, design["evaluation"]["evaluation_seeds"])
    finally:
        env.close()
    nets = [r.net for r in results]
    return {
        "arm": job["arm"],
        "candidate": job["candidate"],
        "regime": job["regime"],
        "day": job["day"],
        "split": job["split"],
        "nets": [str(n) for n in nets],
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)),
        "attribution": [r.attribution for r in results],
        "penalties": [str(r.penalties) for r in results],
        "steps": [r.steps for r in results],
    }


def pool(jobs, worker, parallel, label):
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for done, _ in enumerate(concurrent.futures.as_completed(futures), start=1):
            elapsed = round(time.perf_counter() - started)
            print(f"{label}: {done}/{len(jobs)} after {elapsed} s", file=sys.stderr, flush=True)
        return [future.result() for future in futures]


def mean_daily(rows):
    values = [Decimal(r["daily_value"]) for r in rows]
    return sum(values, Decimal(0)) / len(values)


def reduced(data_dir, out_dir, commit, dry_budget=None):
    torch.set_num_threads(1)
    design, digests = load_design()
    step = dict(next(s for s in design["steps"] if s["name"] == "reduced"))
    if dry_budget is not None:
        # A check of the code only: its results are not the step's and are not reported.
        step["budget_steps"] = dry_budget
    parallel = design["compute"]["max_parallel_runs"]
    out = pathlib.Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    report = {
        "step": "reduced",
        "commit": commit,
        "data": design["data"],
        "bounds": design["bounds"],
        "design_sha256": sha256(HERE / "design.json"),
        "digests_sha256": sha256(HERE / "digests.json"),
        "versions": versions(),
        "regimes": step["regimes"],
        "training_seeds": step["training_seeds"],
        "budget_steps": step["budget_steps"],
        "holdout": "not read",
        "dry_run": dry_budget is not None,
    }
    clock = {}
    arms = design["arms"]
    ppo_grid = arms["maskable_ppo"]["grid"]
    grid_config = base_config(design, "main", "mask", ppo_grid)
    validation = days_of(design, "validation")
    started = time.perf_counter()

    sigmas = {r: volatility(data_dir, design, digests, r) for r in step["regimes"]}
    report["volatility_from_training_days"] = sigmas

    # Baselines and classical arms, scored on validation, every grid point reported.
    jobs = []
    for regime in step["regimes"]:
        for day in validation:
            common = {"regime": regime, "day": day, "split": "validation", "data_dir": data_dir,
                      "variant": "main"}
            jobs.append({**common, "arm": "no_trade", "candidate": "hold",
                         "encoding": "maskable", "config": grid_config,
                         "policy": {"kind": "constant", "action": [0, 0, 0, 0, 0]}})
            offsets = ppo_grid["bid_offsets"]
            for point in grid_points(arms["fixed_spread"]["grid"]):
                index = offsets.index(point["offset_ticks"])
                jobs.append({**common, "arm": "fixed_spread",
                             "candidate": json.dumps(point, sort_keys=True),
                             "encoding": "maskable", "config": grid_config,
                             "policy": {"kind": "constant", "action": [1, index, index, 0, 0]}})
            for kind in ("avellaneda_stoikov", "glft"):
                for point in grid_points(arms[kind]["grid"], sigmas[regime]):
                    jobs.append({**common, "arm": kind,
                                 "candidate": json.dumps(point, sort_keys=True),
                                 "encoding": "residual", "baseline_kind": kind,
                                 "baseline": point,
                                 "bounds": arms["sac_residual_avellaneda_stoikov"]["bounds"],
                                 "policy": {"kind": "zero_residual"}})
    # The slow ones first, so that the pool ends together.
    jobs.sort(key=lambda job: job["arm"] != "glft")
    classical = pool(jobs, evaluate_job, parallel, "classical")
    clock["classical_validation_seconds"] = round(time.perf_counter() - started, 1)
    report["classical_validation"] = classical

    selected = {}
    for regime in step["regimes"]:
        for kind in ("avellaneda_stoikov", "glft"):
            scores = {}
            for row in classical:
                if row["arm"] == kind and row["regime"] == regime:
                    scores.setdefault(row["candidate"], []).append(row)
            best = max(scores, key=lambda c: (mean_daily(scores[c]), c))
            selected[(regime, kind)] = json.loads(best)
    report["classical_selected"] = {f"{r}/{k}": v for (r, k), v in selected.items()}

    # Learned arms, trained on the training days with the first hyperparameter set.
    started = time.perf_counter()
    train = []
    for regime in step["regimes"]:
        for seed in step["training_seeds"]:
            common = {"regime": regime, "seed": seed, "steps": step["budget_steps"],
                      "data_dir": data_dir, "variant": "main", "commit": commit}
            train.append({**common, "algorithm": "maskable_ppo", "encoding": "maskable",
                          "config": grid_config,
                          "hyperparameters": arms["maskable_ppo"]["hyperparameter_sets"][0],
                          "path": str(out / "checkpoints" / f"ppo-{regime}-{seed}")})
            residual = arms["sac_residual_avellaneda_stoikov"]
            train.append({**common, "algorithm": "sac_residual", "encoding": "residual",
                          "baseline_kind": "avellaneda_stoikov",
                          "baseline": selected[(regime, "avellaneda_stoikov")],
                          "bounds": residual["bounds"],
                          "hyperparameters": residual["hyperparameter_sets"][0],
                          "path": str(out / "checkpoints" / f"sac-as-{regime}-{seed}")})
    trained = pool(train, train_job, parallel, "training")
    clock["training_seconds"] = round(time.perf_counter() - started, 1)
    for job, metadata in zip(train, trained, strict=True):
        metadata["path"] = job["path"]
    report["training"] = trained

    started = time.perf_counter()
    jobs = []
    for job, metadata in zip(train, trained, strict=True):
        for day in validation:
            jobs.append({**{k: v for k, v in job.items() if k not in ("seed", "steps", "path")},
                         "day": day, "split": "validation",
                         "arm": "maskable_ppo" if job["algorithm"] == "maskable_ppo"
                         else "sac_residual_avellaneda_stoikov",
                         "candidate": f"seed-{job['seed']}",
                         "policy": {"kind": "frozen", "metadata": metadata}})
    learned = pool(jobs, evaluate_job, parallel, "learned")
    clock["learned_validation_seconds"] = round(time.perf_counter() - started, 1)
    report["learned_validation"] = learned

    summary = {}
    for row in classical + learned:
        key = f"{row['regime']}/{row['arm']}/{row['candidate']}"
        summary.setdefault(key, []).append(row)
    report["mean_daily_net_on_validation"] = {
        key: str(mean_daily(rows)) for key, rows in sorted(summary.items())
    }
    report["seconds"] = clock
    (out / "reduced.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    step, data_dir, out_dir, commit = sys.argv[1:5]
    dry_budget = int(sys.argv[6]) if sys.argv[5:6] == ["--dry-run"] else None
    if step != "reduced":
        raise SystemExit(f"step {step!r} is not implemented yet")
    started = time.perf_counter()
    report = reduced(data_dir, out_dir, commit, dry_budget)
    print(json.dumps({"seconds": report["seconds"], "total": round(time.perf_counter() - started, 1),
                      "mean_daily_net_on_validation": report["mean_daily_net_on_validation"]},
                     indent=2))


if __name__ == "__main__":
    main()
