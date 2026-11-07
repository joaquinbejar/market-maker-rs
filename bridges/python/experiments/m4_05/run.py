"""Runs a step of the out-of-sample comparison of #102 from its committed design.

From `bridges/python`, after `qr_days` wrote the rows:

    python experiments/m4_05/run.py <step> <data dir> <out dir> <commit>

Steps: `check_ppo_hold`, `selection`, `ablations`, then `holdout`. The reduced step ran at
commit da14f8f with the runner of that commit. Every day's rows are checked against
`digests.json` before use. Only `holdout` reads the holdout, and only once every checkpoint
and configuration is frozen: it takes the frozen manifest that `freeze` writes, and refuses
any checkpoint, configuration or design whose hash differs.

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
from mm_rl_bridge.training import ZeroResidual, versions

HERE = pathlib.Path(__file__).resolve().parent
DAY_MS = 86_400_000
SECONDS_PER_YEAR = 31_536_000
POSITION_FEATURE = 6
SCALES = {
    "spread_ticks": "10",
    "mid_change_ticks": "10",
    "row_gap_ms": "20000",
    "size_lots": "10",
    "position": "20",
    "entry_gap": "1",
}
LEARNED = {
    "maskable_ppo": {"encoding": "maskable", "algorithm": "maskable_ppo"},
    "sac_residual_avellaneda_stoikov": {"encoding": "residual", "algorithm": "sac_residual"},
}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_design():
    design = json.loads((HERE / "design.json").read_text())
    digests = json.loads((HERE / "digests.json").read_text())
    if digests["design_version"] != design["design_version"]:
        raise ValueError("the digests belong to another design")
    by_day = {(d["regime"], d["day"]): d for d in digests["days"]}
    return design, by_day


def regimes_of(design):
    return [regime["name"] for regime in design["simulator"]["regimes"]]


def days_of(design, split):
    first, last = design["splits"][split]
    return list(range(first, last + 1))


def read_rows(data_dir, digests, regime, day, split, holdout_allowed=False):
    if split == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    record = digests[(regime, day)]
    if record["split"] != split:
        raise ValueError(f"{regime} day {day} is in {record['split']}, not {split}")
    path = pathlib.Path(data_dir) / split / regime / f"day-{day}.json"
    rows = [tuple(row) for row in json.loads(path.read_text())]
    return rows, record


def execution(design, variant):
    chosen = dict(design["execution"]["main"])
    if variant != "main":
        for key, value in design["execution"]["ablations"][variant].items():
            if key in chosen:
                chosen[key] = value
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


def volatility(data_dir, design, digests, regime):
    """The design's rule, over the training days only: absolute price units per square root
    of a 365 day year, from the changes of the mid between consecutive rows of a day."""
    tick = Decimal(design["kernel"]["tick_size"])
    changes = []
    for day in days_of(design, "train"):
        rows, _ = read_rows(data_dir, digests, regime, day, "train")
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

    def __init__(self, envs, seed, penalty=None):
        super().__init__()
        self._envs = envs
        self._rng = np.random.default_rng(seed)
        self._current = envs[0]
        self._penalty = penalty
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current = self._envs[int(self._rng.integers(len(self._envs)))]
        return self._current.reset(seed=int(self._rng.integers(2**63)))

    def step(self, action):
        observation, reward, terminated, truncated, info = self._current.step(action)
        if self._penalty is not None:
            # Training only: the exact net in `info` is untouched, and evaluation never
            # sees this term.
            position = info["raw_observation"][POSITION_FEATURE]
            if position is not None:
                reward -= self._penalty * float(position) ** 2
        return observation, reward, terminated, truncated, info

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


def build_env(job, digests, regime, day, split, holdout_allowed=False):
    rows, record = read_rows(job["data_dir"], digests, regime, day, split, holdout_allowed)
    design, _ = load_design()
    if job["encoding"] == "maskable":
        env = MaskableMarketMakingEnv(job["config"], rows)
    else:
        config = residual_config(
            design,
            job["variant"],
            job["baseline_kind"],
            job["baselines"][regime],
            day,
            job["bounds"],
        )
        env = ResidualMarketMakingEnv(config, rows)
    return checked(env, record)


def train_job(job):
    """One training run on the pooled training days, in its own process on one thread."""
    torch.set_num_threads(1)
    design, digests = load_design()
    set_random_seed(job["seed"])
    pairs = [(regime, day) for regime in job["regimes"] for day in days_of(design, "train")]
    envs = [build_env(job, digests, regime, day, "train") for regime, day in pairs]
    penalty = job.get("penalty")
    cycle = DayCycle(envs, job["seed"], None if penalty is None else float(penalty))
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
    configs = {}
    for (regime, _), env in zip(pairs, envs, strict=True):
        configs.setdefault(regime, env.config)
    metadata = {
        "algorithm": job["algorithm"],
        "arm": job["arm"],
        "variant": job["variant"],
        "penalty": penalty,
        "set": job["set"],
        "seed": job["seed"],
        "steps": job["steps"],
        "hyperparameters": job["hyperparameters"],
        "spec": list(envs[0].spec_id),
        "kernel_fingerprint": envs[0].bridge_info["kernel_fingerprint"],
        "configs": configs,
        "training_days": [list(pair) for pair in pairs],
        "training_digests": [env.bridge_info["dataset_digest"] for env in envs],
        "commit": job["commit"],
        "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(),
        "train_seconds": round(seconds, 2),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")),
        "path": str(path),
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


def load_frozen(metadata, env, regime):
    path = pathlib.Path(metadata["path"])
    if sha256(path.with_suffix(".zip")) != metadata["checkpoint_sha256"]:
        raise ValueError("a checkpoint changed after training")
    if tuple(metadata["spec"]) != env.spec_id:
        raise ValueError("a checkpoint of another specification")
    if metadata["kernel_fingerprint"] != env.bridge_info["kernel_fingerprint"]:
        raise ValueError("a checkpoint of other execution rules")
    if not same_but_session(metadata["configs"][regime], env.config):
        raise ValueError("a checkpoint of another configuration")
    model_class = MaskablePPO if metadata["algorithm"] == "maskable_ppo" else SAC
    model = model_class.load(path.with_suffix(".zip"), device="cpu")
    model.policy.set_training_mode(False)
    return model


def time_beyond(attribution, lots):
    """The share of the window with an absolute position above `lots`."""
    start, end = attribution["window_ms"]
    if end <= start:
        return 0.0
    beyond, since, position = 0, start, Decimal(0)
    for timestamp, _, after in attribution["positions"]:
        until = min(max(timestamp, start), end)
        if abs(position) > lots:
            beyond += until - since
        since, position = until, Decimal(after)
    if abs(position) > lots:
        beyond += end - since
    return beyond / (end - start)


def episode(policy, env, seed, design):
    """One evaluated day: the exact net result, its attribution, and what the gate refused."""
    observation, _ = env.reset(seed=seed)
    refusals = fallbacks = clipped = steps = 0
    masked = isinstance(env, MaskableMarketMakingEnv)
    while True:
        if masked:
            action, _ = policy.predict(
                observation, deterministic=True, action_masks=env.action_masks()
            )
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        refusals += info["refused_because"] is not None
        residual = info.get("residual")
        if residual is not None:
            fallbacks += residual["fallback"] is not None
            clipped += any(residual["clips"])
        if terminated or truncated:
            break
    attribution = dict(env.attribution())
    limits = design["execution"]["limits"]
    peak_notional = max(
        (abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
        default=Decimal(0),
    )
    return {
        "seed": seed,
        "steps": steps,
        "net": info["net"],
        "penalties": info["penalties"],
        "attribution": {k: v for k, v in attribution.items() if k != "positions"},
        "refusals": refusals,
        "residual_fallbacks": fallbacks,
        "residual_clipped": clipped,
        "peak_notional": str(peak_notional),
        "time_beyond_5_lots": time_beyond(attribution, 5),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak_notional > Decimal(limits["max_notional"]),
    }


def evaluate_job(job):
    """One arm on one day, over the evaluation seeds."""
    torch.set_num_threads(1)
    design, digests = load_design()
    env = build_env(
        job, digests, job["regime"], job["day"], job["split"], job["split"] == "holdout"
    )
    try:
        if job["policy"]["kind"] == "constant":
            policy = Constant(job["policy"]["action"])
        elif job["policy"]["kind"] == "zero_residual":
            policy = ZeroResidual()
        else:
            policy = load_frozen(job["policy"]["metadata"], env, job["regime"])
        days = [episode(policy, env, seed, design) for seed in job["evaluation_seeds"]]
    finally:
        env.close()
    nets = [Decimal(d["net"]) for d in days]
    return {
        "arm": job["arm"],
        "candidate": job["candidate"],
        "variant": job["variant"],
        "regime": job["regime"],
        "day": job["day"],
        "split": job["split"],
        "ablation": job.get("ablation"),
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)),
        "episodes": days,
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


class Context:
    """What every step shares: the design, the data, the output and the commit."""

    def __init__(self, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.digests = load_design()
        self.data_dir = data_dir
        self.out = pathlib.Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.parallel = self.design["compute"]["max_parallel_runs"]
        self.seeds = self.design["evaluation"]["evaluation_seeds_from_step_2"]
        self.regimes = regimes_of(self.design)
        self.arms = self.design["arms"]
        self.grid = self.arms["maskable_ppo"]["grid"]
        self.sigmas = {r: volatility(data_dir, self.design, self.digests, r) for r in self.regimes}

    def header(self, step):
        return {
            "step": step,
            "commit": self.commit,
            "data": self.design["data"],
            "bounds": self.design["bounds"],
            "design_sha256": sha256(HERE / "design.json"),
            "digests_sha256": sha256(HERE / "digests.json"),
            "versions": versions(),
            "evaluation_seeds": self.seeds,
            "volatility_from_training_days": self.sigmas,
        }

    def grid_config(self, variant):
        return base_config(self.design, variant, "mask", self.grid)

    def common(self, variant, regime, day, split):
        return {"regime": regime, "day": day, "split": split, "data_dir": self.data_dir,
                "variant": variant, "evaluation_seeds": self.seeds}

    def classical_jobs(self, variant, split, days, chosen=None):
        """Baselines and classical arms; every grid point, or only the chosen ones."""
        jobs = []
        offsets = self.grid["bid_offsets"]
        bounds = self.arms["sac_residual_avellaneda_stoikov"]["bounds"]
        for regime in self.regimes:
            for day in days:
                common = self.common(variant, regime, day, split)
                maskable = {**common, "encoding": "maskable", "config": self.grid_config(variant)}
                jobs.append({**maskable, "arm": "no_trade", "candidate": "hold",
                             "policy": {"kind": "constant", "action": [0, 0, 0, 0, 0]}})
                for point in grid_points(self.arms["fixed_spread"]["grid"]):
                    candidate = json.dumps(point, sort_keys=True)
                    if chosen is not None and chosen[(regime, "fixed_spread")] != point:
                        continue
                    index = offsets.index(point["offset_ticks"])
                    jobs.append({**maskable, "arm": "fixed_spread", "candidate": candidate,
                                 "policy": {"kind": "constant",
                                            "action": [1, index, index, 0, 0]}})
                for kind in ("avellaneda_stoikov", "glft"):
                    for point in grid_points(self.arms[kind]["grid"], self.sigmas[regime]):
                        if chosen is not None and chosen[(regime, kind)] != point:
                            continue
                        jobs.append({**common, "arm": kind,
                                     "candidate": json.dumps(point, sort_keys=True),
                                     "encoding": "residual", "baseline_kind": kind,
                                     "baselines": {regime: point}, "bounds": bounds,
                                     "policy": {"kind": "zero_residual"}})
        # The slow ones first, so that the pool ends together.
        jobs.sort(key=lambda job: job["arm"] != "glft")
        return jobs

    def train_jobs(self, arm, sets, variant, baselines, penalty=None, tag="main"):
        jobs = []
        spec = self.arms[arm]
        for index in sets:
            for seed in self.design["training_seeds"]:
                name = f"{arm}-{tag}-set{index}-seed{seed}"
                job = {"arm": arm, "algorithm": LEARNED[arm]["algorithm"],
                       "encoding": LEARNED[arm]["encoding"], "regimes": self.regimes,
                       "seed": seed, "steps": spec["budget_steps"], "set": index,
                       "hyperparameters": spec["hyperparameter_sets"][index],
                       "data_dir": self.data_dir, "variant": variant, "penalty": penalty,
                       "commit": self.commit, "path": str(self.out / "checkpoints" / name)}
                if LEARNED[arm]["encoding"] == "maskable":
                    job["config"] = self.grid_config(variant)
                else:
                    job.update({"baseline_kind": "avellaneda_stoikov", "baselines": baselines,
                                "bounds": spec["bounds"]})
                jobs.append(job)
        return jobs

    def learned_jobs(self, runs, variant, split, days):
        jobs = []
        for metadata in runs:
            for regime in self.regimes:
                for day in days:
                    job = {**self.common(variant, regime, day, split), "arm": metadata["arm"],
                           "candidate": f"set{metadata['set']}-seed{metadata['seed']}",
                           "encoding": LEARNED[metadata["arm"]]["encoding"],
                           "policy": {"kind": "frozen", "metadata": metadata}}
                    if job["encoding"] == "maskable":
                        job["config"] = self.grid_config(variant)
                    else:
                        baselines = {}
                        for r, config in metadata["configs"].items():
                            baselines[r] = {k: v for k, v in config["residual"]["baseline"].items()
                                            if k not in ("kind", "session_start_ms")}
                        job.update({"baseline_kind": "avellaneda_stoikov",
                                    "baselines": baselines,
                                    "bounds": self.arms[metadata["arm"]]["bounds"]})
                    jobs.append(job)
        return jobs

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path


def best(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def selection(ctx):
    report = ctx.header("selection")
    clock, started = {}, time.perf_counter()
    validation = days_of(ctx.design, "validation")
    classical = pool(ctx.classical_jobs("main", "validation", validation), evaluate_job,
                     ctx.parallel, "classical")
    clock["classical_validation_seconds"] = round(time.perf_counter() - started, 1)
    chosen = {}
    for (regime, arm), rows in best(classical, lambda r: (r["regime"], r["arm"])).items():
        if arm == "no_trade":
            continue
        scores = best(rows, lambda r: r["candidate"])
        top = max(scores, key=lambda c: (mean_daily(scores[c]), c))
        chosen[(regime, arm)] = json.loads(top)
    report["classical_validation"] = classical
    report["classical_selected"] = {f"{r}/{a}": v for (r, a), v in sorted(chosen.items())}

    baselines = {r: chosen[(r, "avellaneda_stoikov")] for r in ctx.regimes}
    started = time.perf_counter()
    sets = {arm: range(len(ctx.arms[arm]["hyperparameter_sets"])) for arm in LEARNED}
    train = [job for arm in LEARNED for job in ctx.train_jobs(arm, sets[arm], "main", baselines)]
    trained = pool(train, train_job, ctx.parallel, "training")
    clock["training_seconds"] = round(time.perf_counter() - started, 1)
    report["training"] = trained

    started = time.perf_counter()
    learned = pool(ctx.learned_jobs(trained, "main", "validation", validation), evaluate_job,
                   ctx.parallel, "learned")
    clock["learned_validation_seconds"] = round(time.perf_counter() - started, 1)
    report["learned_validation"] = learned
    selected_sets = {}
    for arm in LEARNED:
        rows = [r for r in learned if r["arm"] == arm]
        by_set = best(rows, lambda r: r["candidate"].split("-")[0])
        selected_sets[arm] = max(by_set, key=lambda s: (mean_daily(by_set[s]), s))
    report["learned_selected_sets"] = selected_sets
    report["seconds"] = clock
    ctx.write("selection", report)
    return report


def ablations(ctx, selection_report):
    """Each ablation retrains the selected set of each learned arm, and evaluates the
    selected classical points, under its execution, on validation."""
    report = ctx.header("ablations")
    clock = {}
    validation = days_of(ctx.design, "validation")
    chosen = {tuple(k.split("/")): v for k, v in selection_report["classical_selected"].items()}
    baselines = {r: chosen[(r, "avellaneda_stoikov")] for r in ctx.regimes}
    sets = {arm: int(s.removeprefix("set"))
            for arm, s in selection_report["learned_selected_sets"].items()}
    for variant, spec in ctx.design["execution"]["ablations"].items():
        started = time.perf_counter()
        execution_variant = "main" if spec.get("training_only") else variant
        penalty = spec.get("penalty_per_squared_lot_per_step") if spec.get("training_only") else None
        train = [job for arm in LEARNED
                 for job in ctx.train_jobs(arm, [sets[arm]], execution_variant, baselines,
                                           penalty, tag=variant)]
        trained = pool(train, train_job, ctx.parallel, f"{variant} training")
        jobs = ctx.learned_jobs(trained, execution_variant, "validation", validation)
        if not spec.get("training_only"):
            jobs += ctx.classical_jobs(variant, "validation", validation, chosen)
        rows = pool(jobs, evaluate_job, ctx.parallel, f"{variant} evaluation")
        for row in rows:
            row["ablation"] = variant
        report[variant] = {"training": trained, "validation": rows}
        clock[variant] = round(time.perf_counter() - started, 1)
    report["seconds"] = clock
    ctx.write("ablations", report)
    return report


def check_ppo_hold(ctx):
    """Whether PPO's hold is an artefact of the mask or of the reward's scale, on training
    days only: how often the mask offers a quote, and what a reward 100 times larger gives."""
    report = ctx.header("check_ppo_hold")
    job = {"encoding": "maskable", "config": ctx.grid_config("main"), "data_dir": ctx.data_dir,
           "variant": "main"}
    offered = decisions = 0
    for regime in ctx.regimes:
        env = build_env(job, ctx.digests, regime, 1, "train")
        env.reset(seed=1)
        while True:
            mask = env.action_masks()
            decisions += 1
            offered += bool(mask[1])
            _, _, terminated, truncated, _ = env.step(np.array([0, 0, 0, 0, 0]))
            if terminated or truncated:
                break
        env.close()
    report["mask_offers_a_quote"] = {"decisions": decisions, "offered": offered}
    scaled = ctx.grid_config("main")
    scaled["f32_unit"] = "0.01"
    train = ctx.train_jobs("maskable_ppo", [0], "main", {}, tag="check-reward-x100")
    for job in train[:2]:
        job["config"] = scaled
        job["steps"] = 50_000
    trained = pool(train[:2], train_job, ctx.parallel, "check training")
    rows = []
    for metadata in trained:
        jobs = []
        for regime in ctx.regimes:
            job = {**ctx.common("main", regime, 2, "train"), "arm": "maskable_ppo",
                   "candidate": f"x100-seed{metadata['seed']}", "encoding": "maskable",
                   "config": scaled, "policy": {"kind": "frozen", "metadata": metadata}}
            jobs.append(job)
        rows += pool(jobs, evaluate_job, ctx.parallel, "check evaluation")
    report["reward_x100_on_training_day_2"] = rows
    report["training"] = trained
    ctx.write("check_ppo_hold", report)
    return report


def frozen_runs(selection_report, ablations_report):
    """The learned runs the holdout evaluates: the selected set of each arm, and every
    ablation run."""
    sets = selection_report["learned_selected_sets"]
    runs = [m for m in selection_report["training"] if f"set{m['set']}" == sets[m["arm"]]]
    for variant, part in ablations_report.items():
        if isinstance(part, dict) and "training" in part:
            runs.extend(dict(m, ablation=variant) for m in part["training"])
    return runs


def freeze(ctx):
    """Records the hash of every checkpoint, configuration and choice the holdout uses,
    before it is read."""
    selection_report = json.loads((ctx.out / "selection.json").read_text())
    ablations_report = json.loads((ctx.out / "ablations.json").read_text())
    runs = frozen_runs(selection_report, ablations_report)
    for metadata in runs:
        if sha256(pathlib.Path(metadata["path"]).with_suffix(".zip")) != metadata["checkpoint_sha256"]:
            raise ValueError(f"{metadata['path']} changed after training")
    manifest = {
        **ctx.header("freeze"),
        "selection_sha256": sha256(ctx.out / "selection.json"),
        "ablations_sha256": sha256(ctx.out / "ablations.json"),
        "classical_selected": selection_report["classical_selected"],
        "learned_selected_sets": selection_report["learned_selected_sets"],
        "runs": runs,
    }
    path = ctx.write("frozen", manifest)
    return {"frozen_sha256": sha256(path), "runs": len(runs)}


def holdout(ctx, frozen_sha256):
    """The holdout, once: every arm as frozen, under its execution."""
    path = ctx.out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    if (ctx.out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    frozen = json.loads(path.read_text())
    for key, name in (("design_sha256", "design.json"), ("digests_sha256", "digests.json")):
        if frozen[key] != sha256(HERE / name):
            raise ValueError(f"{name} changed after the freeze")
    for name in ("selection", "ablations"):
        if frozen[f"{name}_sha256"] != sha256(ctx.out / f"{name}.json"):
            raise ValueError(f"{name}.json changed after the freeze")
    report = ctx.header("holdout")
    report["frozen_sha256"] = frozen_sha256
    started = time.perf_counter()
    days = days_of(ctx.design, "holdout")
    chosen = {tuple(k.split("/")): v for k, v in frozen["classical_selected"].items()}
    jobs = []
    for variant in ["main"] + [v for v, s in ctx.design["execution"]["ablations"].items()
                               if not s.get("training_only")]:
        for job in ctx.classical_jobs(variant, "holdout", days, chosen):
            jobs.append(job)
    for metadata in frozen["runs"]:
        variant = metadata["variant"]
        for job in ctx.learned_jobs([metadata], variant, "holdout", days):
            job["ablation"] = metadata.get("ablation")
            jobs.append(job)
    rows = pool(jobs, evaluate_job, ctx.parallel, "holdout")
    report["rows"] = rows
    report["seconds"] = {"holdout_seconds": round(time.perf_counter() - started, 1)}
    ctx.write("holdout", report)
    return report


def main():
    step, data_dir, out_dir, commit = sys.argv[1:5]
    ctx = Context(data_dir, out_dir, commit)
    started = time.perf_counter()
    if step == "check_ppo_hold":
        report = check_ppo_hold(ctx)
    elif step == "selection":
        report = selection(ctx)
    elif step == "ablations":
        report = ablations(ctx, json.loads((ctx.out / "selection.json").read_text()))
    elif step == "freeze":
        report = freeze(ctx)
        print(json.dumps(report))
    elif step == "holdout":
        report = holdout(ctx, sys.argv[5])
    else:
        raise SystemExit(f"step {step!r} is not implemented yet")
    print(json.dumps({"step": step, "seconds": report.get("seconds"),
                      "total": round(time.perf_counter() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
