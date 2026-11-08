"""Runs a step of the decision frequency and memory comparison of #103 from its design.

From `bridges/python`, after `qr_days` wrote the rows of `design.json`:

    python experiments/m4_06/run.py <step> <data dir> <out dir> <commit> [<frozen sha256>]

Steps, in order: `fee_rule`, `classical_selection`, `training`, `freeze`, `holdout`. Every
day's rows are checked against `digests.json` before use. Only `holdout` reads the holdout,
once, from the manifest `freeze` writes, and it refuses any checkpoint, configuration or
design whose hash differs. Every row of every report carries its arm, interval, seed and fee.

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
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from mm_rl_bridge import BoxMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import ZeroResidual, versions
from mm_rl_bridge.wrappers import ORDER_KINDS, DecisionInterval, EmaFeatures, history

HERE = pathlib.Path(__file__).resolve().parent
DAY_MS = 86_400_000
SECONDS_PER_YEAR = 31_536_000
LEARNED = ("mlp", "ema", "history", "lstm")
CLASSICAL = ("avellaneda_stoikov", "glft")
HOLD = [-1.0, 0.0, 0.0, 0.0, 0.0]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_design():
    design = json.loads((HERE / "design.json").read_text())
    digests = json.loads((HERE / "digests.json").read_text())
    if digests["design_version"] != design["design_version"]:
        raise ValueError("the digests belong to another design")
    return design, {(d["regime"], d["day"]): d for d in digests["days"]}


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
    return [tuple(row) for row in json.loads(path.read_text())], record


def base_config(design, fee, policy="reject"):
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"],
        "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"],
        "max_order_size": kernel["max_order_size"],
        "max_steps": design["simulator"]["rows_per_day"],
        "fee_rate": fee,
        "initial_capital": run["initial_capital"],
        "latency": run["latency"],
        "fill_model": run["fill_model"],
        "limits": run["limits"],
        "bid_offsets": grid["bid_offsets"],
        "ask_offsets": grid["ask_offsets"],
        "sizes": grid["sizes"],
        "scales": design["observation_scales"],
        "f32_unit": "1",
        "invalid_action": policy,
        "invalid_action_penalty": design["action_space"]["invalid_action_penalty"],
        "terminal": run["terminal"],
    }


def baseline_config(kind, params, day):
    """A classical strategy for one day, its session starting with the day."""
    start = (day - 1) * DAY_MS
    if kind == "avellaneda_stoikov":
        return {"kind": kind, "risk_aversion": params["risk_aversion"],
                "volatility": params["volatility"], "order_intensity": params["order_intensity"],
                "session_start_ms": start, "horizon_ms": params["horizon_ms"],
                "size": params["size"]}
    return {"kind": kind, "risk_aversion": params["risk_aversion"],
            "order_intensity_k": params["order_intensity_k"],
            "intensity_scale_a": params["intensity_scale_a"],
            "max_inventory_lots": params["max_inventory_lots"],
            "terminal_time": params["terminal_time"], "min_spread": params["min_spread"],
            "volatility": params["volatility"], "session_start_ms": start,
            "form": "closed_form"}


def volatility(data_dir, design, digests, regime):
    """The design's rule over the training days: absolute price units per square root of a
    365 day year, from the changes of the mid between consecutive rows of a day."""
    tick = Decimal(design["kernel"]["tick_size"])
    count, total, squares = 0, 0.0, 0.0
    for day in days_of(design, "train"):
        rows, _ = read_rows(data_dir, digests, regime, day, "train")
        mids = [float(Decimal(row[1] + row[3]) / 2 * tick) for row in rows]
        for a, b in zip(mids, mids[1:]):
            change = b - a
            count, total, squares = count + 1, total + change, squares + change * change
    mean = total / count
    variance = squares / count - mean * mean
    seconds = design["simulator"]["sample_ms"] / 1000
    return f"{(variance * SECONDS_PER_YEAR / seconds) ** 0.5:.4f}"


def grid_points(grid, sigma):
    keys = sorted(grid)
    points = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    for point in points:
        if point.get("volatility") == "from the volatility rule":
            point["volatility"] = sigma
    return points


def build_env(job, digests, regime, day, split, holdout_allowed=False):
    """The environment of one day, wrapped for the job's arm and decision interval."""
    design, _ = load_design()
    rows, record = read_rows(job["data_dir"], digests, regime, day, split, holdout_allowed)
    config = base_config(design, job["fee"])
    if job["arm"] in CLASSICAL:
        config["residual"] = {"baseline": baseline_config(job["arm"], job["point"], day),
                              "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        env = ResidualMarketMakingEnv(config, rows)
    else:
        env = BoxMarketMakingEnv(config, rows)
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{regime} day {day} has other rows than digests.json")
    env = DecisionInterval(env, job["every"])
    if job["arm"] == "ema":
        env = EmaFeatures(env, tuple(design["arms"]["ema"]["half_lives_decisions"]))
    elif job["arm"] == "history":
        env = history(env, design["arms"]["history"]["length"])
    return env


class DayCycle(gym.Env):
    """One wrapped environment per training day; every reset draws a day from the seed."""

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

    def close(self):
        for env in self._envs:
            env.close()


def model_class(arm):
    return (RecurrentPPO, "MlpLstmPolicy") if arm == "lstm" else (PPO, "MlpPolicy")


def train_job(job):
    """One training run on the pooled training days, in its own process on one thread."""
    torch.set_num_threads(1)
    design, digests = load_design()
    set_random_seed(job["seed"])
    pairs = [(regime, day) for regime in regimes_of(design) for day in days_of(design, "train")]
    envs = [build_env(job, digests, regime, day, "train") for regime, day in pairs]
    cycle = DayCycle(envs, job["seed"])
    algorithm, policy = model_class(job["arm"])
    started = time.perf_counter()
    model = algorithm(policy, DummyVecEnv([lambda: Monitor(cycle)]), seed=job["seed"],
                      device="cpu", verbose=0, **job["hyperparameters"])
    model.learn(total_timesteps=job["decisions"], progress_bar=False)
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    model.save(path)
    metadata = {
        "arm": job["arm"], "every": job["every"], "seed": job["seed"], "fee": job["fee"],
        "decisions": job["decisions"], "hyperparameters": job["hyperparameters"],
        "spec": list(envs[0].unwrapped.spec_id),
        "kernel_fingerprint": envs[0].unwrapped.bridge_info["kernel_fingerprint"],
        "config": envs[0].unwrapped.config,
        "training_digests": [env.unwrapped.bridge_info["dataset_digest"] for env in envs],
        "commit": job["commit"], "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(), "train_seconds": round(seconds, 1),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path),
    }
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def load_frozen(metadata, env):
    """A trained policy, after checking that it is the one trained and that the environment
    is the one it was trained for, but for its fee and its day."""
    path = pathlib.Path(metadata["path"])
    if sha256(path.with_suffix(".zip")) != metadata["checkpoint_sha256"]:
        raise ValueError("a checkpoint changed after training")
    base = env.unwrapped
    if tuple(metadata["spec"]) != base.spec_id:
        raise ValueError("a checkpoint of another specification")
    same = {k: v for k, v in metadata["config"].items() if k != "fee_rate"}
    if same != {k: v for k, v in base.config.items() if k != "fee_rate"}:
        raise ValueError("a checkpoint of another configuration")
    algorithm, _ = model_class(metadata["arm"])
    model = algorithm.load(path.with_suffix(".zip"), device="cpu")
    model.policy.set_training_mode(False)
    return model


class Constant:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)

    def predict(self, observation, **_):
        return self._action, None


def policy_of(job, env):
    kind = job["policy"]["kind"]
    if kind == "hold":
        return Constant(HOLD)
    if kind == "zero_residual":
        return ZeroResidual()
    return load_frozen(job["policy"]["metadata"], env)


def applied_kind(info):
    """`hold` or the kind of what the decision sent; a residual step names its composed
    action, which a refusal turns into a hold."""
    if "applied_action" in info:
        return info["applied_action"]["kind"]
    if info["refused_because"] is not None:
        return "hold"
    return info["residual"]["composed"]["kind"]


def time_beyond(attribution, lots):
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


def episode(policy, env, seed, recurrent, design):
    """One evaluated day. A recurrent policy starts from a zero state: `episode_start`."""
    observation, _ = env.reset(seed=seed)
    state, start = None, np.ones((1,), dtype=bool)
    orders = dict.fromkeys(ORDER_KINDS, 0)
    decisions = quotes = refusals = rows = 0
    decide_seconds = 0.0
    while True:
        started = time.perf_counter()
        if recurrent:
            action, state = policy.predict(observation, state=state, episode_start=start,
                                           deterministic=True)
            start = np.zeros((1,), dtype=bool)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        decide_seconds += time.perf_counter() - started
        observation, _, terminated, truncated, info = env.step(action)
        decisions += 1
        rows += info["rows"]
        quotes += applied_kind(info) != "hold"
        refusals += info["refused_because"] is not None
        for kind in ORDER_KINDS:
            orders[kind] += info["orders"][kind]
        if terminated or truncated:
            break
    attribution = dict(env.unwrapped.attribution())
    limits = design["execution"]["limits"]
    peak = max((abs(Decimal(a)) * Decimal(p) for _, p, a in attribution["positions"]),
               default=Decimal(0))
    return {
        "seed": seed, "decisions": decisions, "rows": rows, "quote_transactions": quotes,
        "refusals": refusals, "orders": orders, "net": info["net"],
        "decide_microseconds": round(1e6 * decide_seconds / decisions, 2),
        "attribution": {k: v for k, v in attribution.items() if k != "positions"},
        "peak_notional": str(peak), "time_beyond_5_lots": time_beyond(attribution, 5),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak > Decimal(limits["max_notional"]),
    }


def evaluate_job(job):
    """One arm on one day at one interval and fee."""
    torch.set_num_threads(1)
    design, digests = load_design()
    env = build_env(job, digests, job["regime"], job["day"], job["split"],
                    job["split"] == "holdout")
    try:
        policy = policy_of(job, env)
        days = [episode(policy, env, seed, job["arm"] == "lstm", design)
                for seed in design["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets = [Decimal(d["net"]) for d in days]
    return {key: job[key] for key in ("arm", "candidate", "every", "fee", "regime", "day",
                                      "split")} | {
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)), "episodes": days}


def pool(jobs, worker, parallel, label):
    """Every job in its own process; results in the order of `jobs`."""
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


def group(rows, key):
    out = {}
    for row in rows:
        out.setdefault(key(row), []).append(row)
    return out


class Context:
    """What every step shares: the design, the data, the output and the commit."""

    def __init__(self, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.digests = load_design()
        self.data_dir, self.commit = data_dir, commit
        self.out = pathlib.Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.parallel = self.design["compute"]["max_parallel_runs"]
        self.regimes = regimes_of(self.design)
        self.intervals = self.design["frequency"]["decision_every_rows"]
        self.sigmas = {r: volatility(data_dir, self.design, self.digests, r)
                       for r in self.regimes}

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "digests_sha256": sha256(HERE / "digests.json"), "versions": versions(),
                "volatility_from_training_days": self.sigmas}

    def job(self, arm, every, fee, regime, day, split, candidate, policy, point=None):
        return {"arm": arm, "every": every, "fee": fee, "regime": regime, "day": day,
                "split": split, "candidate": candidate, "policy": policy, "point": point,
                "data_dir": self.data_dir}

    def classical_jobs(self, fee, split, days, intervals, chosen=None, arms=CLASSICAL):
        jobs = []
        for every in intervals:
            for regime in self.regimes:
                for day in days:
                    jobs.append(self.job("no_trade", every, fee, regime, day, split, "hold",
                                         {"kind": "hold"}))
                    for arm in arms:
                        for point in grid_points(self.design["arms"][arm]["grid"],
                                                 self.sigmas[regime]):
                            if chosen is not None and chosen[f"{regime}/{arm}/{every}"] != point:
                                continue
                            jobs.append(self.job(arm, every, fee, regime, day, split,
                                                 json.dumps(point, sort_keys=True),
                                                 {"kind": "zero_residual"}, point))
        return jobs

    def learned_jobs(self, runs, fee, split, days):
        return [self.job(m["arm"], m["every"], fee, regime, day, split, f"seed{m['seed']}",
                         {"kind": "frozen", "metadata": m})
                for m in runs for regime in self.regimes for day in days]

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path


def fee_rule(ctx):
    """The design's fee rule, on training days only."""
    report = ctx.header("fee_rule")
    started = time.perf_counter()
    rule = ctx.design["execution"]["fee_rule"]
    rows = []
    for fee in rule["candidates"]:
        rows += pool(ctx.classical_jobs(fee, "train", days_of(ctx.design, "train"), [1],
                                        arms=("avellaneda_stoikov",)),
                     evaluate_job, ctx.parallel, f"fee {fee}")
    best = {}
    for fee, by_fee in group([r for r in rows if r["arm"] == "avellaneda_stoikov"],
                             lambda r: r["fee"]).items():
        points = group(by_fee, lambda r: r["candidate"])
        top = max(points, key=lambda c: (mean_daily(points[c]), c))
        best[fee] = {"point": json.loads(top), "mean_daily_net": str(mean_daily(points[top]))}
    positive = [fee for fee in rule["candidates"] if Decimal(best[fee]["mean_daily_net"]) > 0]
    chosen = max(positive, key=Decimal) if positive else ctx.design["execution"]["secondary_fee_rate"]
    report.update({"rows": rows, "best_per_fee": best, "fee": chosen,
                   "profitable": bool(positive),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("fee_rule", report)
    return report


def classical_selection(ctx, fee):
    report = ctx.header("classical_selection")
    started = time.perf_counter()
    rows = pool(ctx.classical_jobs(fee, "validation", days_of(ctx.design, "validation"),
                                   ctx.intervals), evaluate_job, ctx.parallel, "classical")
    chosen = {}
    for (regime, arm, every), part in group(
            [r for r in rows if r["arm"] in CLASSICAL],
            lambda r: (r["regime"], r["arm"], r["every"])).items():
        points = group(part, lambda r: r["candidate"])
        chosen[f"{regime}/{arm}/{every}"] = json.loads(
            max(points, key=lambda c: (mean_daily(points[c]), c)))
    report.update({"fee": fee, "rows": rows, "selected": chosen,
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("classical_selection", report)
    return report


def training(ctx, fee):
    report = ctx.header("training")
    started = time.perf_counter()
    jobs = []
    for arm in LEARNED:
        for every in ctx.intervals:
            for seed in ctx.design["training_seeds"]:
                name = f"{arm}-every{every}-seed{seed}"
                jobs.append({"arm": arm, "every": every, "seed": seed, "fee": fee,
                             "decisions": ctx.design["budget_decisions"],
                             "hyperparameters": {k: v for k, v in
                                                 ctx.design["hyperparameters"].items()
                                                 if k != "note"},
                             "data_dir": ctx.data_dir, "commit": ctx.commit,
                             "path": str(ctx.out / "checkpoints" / name)})
    # The slow ones first, so that the pool ends together.
    jobs.sort(key=lambda job: (job["arm"] != "lstm", -job["every"]))
    report.update({"fee": fee, "runs": pool(jobs, train_job, ctx.parallel, "training"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("training", report)
    return report


def freeze(ctx):
    names = ("fee_rule", "classical_selection", "training")
    reports = {name: json.loads((ctx.out / f"{name}.json").read_text()) for name in names}
    for metadata in reports["training"]["runs"]:
        if sha256(pathlib.Path(metadata["path"]).with_suffix(".zip")) != metadata["checkpoint_sha256"]:
            raise ValueError(f"{metadata['path']} changed after training")
    manifest = {**ctx.header("freeze"),
                **{f"{name}_sha256": sha256(ctx.out / f"{name}.json") for name in names},
                "fee": reports["fee_rule"]["fee"],
                "classical_selected": reports["classical_selection"]["selected"],
                "runs": reports["training"]["runs"]}
    path = ctx.write("frozen", manifest)
    return {"frozen_sha256": sha256(path), "runs": len(manifest["runs"])}


def holdout(ctx, frozen_sha256):
    """The holdout, once: every arm as frozen, at the chosen fee and at the secondary one."""
    path = ctx.out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    if (ctx.out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    frozen = json.loads(path.read_text())
    for key, name in (("design_sha256", "design.json"), ("digests_sha256", "digests.json")):
        if frozen[key] != sha256(HERE / name):
            raise ValueError(f"{name} changed after the freeze")
    for name in ("fee_rule", "classical_selection", "training"):
        if frozen[f"{name}_sha256"] != sha256(ctx.out / f"{name}.json"):
            raise ValueError(f"{name}.json changed after the freeze")
    report = ctx.header("holdout")
    report["frozen_sha256"] = frozen_sha256
    started = time.perf_counter()
    days = days_of(ctx.design, "holdout")
    fees = list(dict.fromkeys([frozen["fee"], ctx.design["execution"]["secondary_fee_rate"]]))
    jobs = []
    for fee in fees:
        jobs += ctx.classical_jobs(fee, "holdout", days, ctx.intervals,
                                   frozen["classical_selected"])
        jobs += ctx.learned_jobs(frozen["runs"], fee, "holdout", days)
    report.update({"fees": fees, "rows": pool(jobs, evaluate_job, ctx.parallel, "holdout"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("holdout", report)
    return report


def main():
    step, data_dir, out_dir, commit = sys.argv[1:5]
    ctx = Context(data_dir, out_dir, commit)
    started = time.perf_counter()
    read = lambda name: json.loads((ctx.out / f"{name}.json").read_text())  # noqa: E731
    if step == "fee_rule":
        report = fee_rule(ctx)
    elif step == "classical_selection":
        report = classical_selection(ctx, read("fee_rule")["fee"])
    elif step == "training":
        report = training(ctx, read("fee_rule")["fee"])
    elif step == "freeze":
        report = freeze(ctx)
        print(json.dumps(report))
    elif step == "holdout":
        report = holdout(ctx, sys.argv[5])
    else:
        raise SystemExit(f"unknown step {step!r}")
    print(json.dumps({"step": step, "seconds": report.get("seconds"),
                      "total": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
