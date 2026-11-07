"""Regime adaptation and transfer (#112), on the days of `examples/regime_days.rs`.

From `bridges/python`, after `regime_days` wrote the days and `regime_alarm` their alarms:

    python experiments/m5_08/run.py check <data dir> <out dir> <commit>
    python experiments/m5_08/run.py classical <data dir> <out dir> <commit> <parallel>
    python experiments/m5_08/run.py train <data dir> <out dir> <commit> <parallel> <arm> <#103 validation sha256>
    python experiments/m5_08/run.py freeze <data dir> <out dir>
    python experiments/m5_08/run.py holdout <data dir> <out dir> <commit> <frozen sha256> <parallel>
    python experiments/m5_08/run.py report <data dir> <out dir> <intervals binary> [<name>]

Everything #103 fixes comes from its runner (`experiments/m4_06/run.py`): the configuration, the
Box encoding with the reject policy, the classical baselines as zero residuals, the learned arms
and their training loop over pooled days, the evaluated episode, the volatility rule. This study
adds the scheduled baseline (#296) for the adaptive and oracle arms, the detector's alarm as an
observation for `learned_detector`, and the comparisons of its design: data shift, execution
change and simulator error.

- `check`: before any training, on training day 1 of each simulator, the fixed arm fills under
  the design's fill model and its net differs from immediate fill (the queue binds).
- `classical`: on validation only, the fixed point's volatility, the adaptive arm's stressed
  point and the coordinate-search oracle's points, by #103's rule.
- `train`: `learned` and `learned_detector` per simulator and seed on its training days.
- `freeze`: hashes the design, the classical selection, every checkpoint and every alarm file.
- `holdout`: refuses anything whose hash differs, reads the holdout once, refuses a second run.
- `report`: intervals, the go rule, transfer and the detector's metrics.

Every day's rows are checked against `digests.json` before use. SYNTHETIC: the simulators and
their parameters are made up.
"""

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from mm_rl_bridge import BoxMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import versions
from mm_rl_bridge.wrappers import DecisionInterval, EmaFeatures, history

HERE = pathlib.Path(__file__).resolve().parent

_SPEC = importlib.util.spec_from_file_location("m4_06_run", HERE.parent / "m4_06" / "run.py")
m6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m6)

SIMULATORS = ("s1", "s2")
REGIMES = ("calm", "stressed", "thin")
AS = "avellaneda_stoikov"
EVERY = 1
CLASSICAL_ARMS = ("fixed", "adaptive", "oracle")
LEARNED_ARMS = ("learned", "learned_detector")
EXECUTIONS = ("design", "immediate")
RATIO_CAP = 5.0
MARKOUT_MS = (1000, 10_000)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    """This design and #103's, with #103's design and #107's checked against the pins."""
    design = json.loads((HERE / "design.json").read_text())
    pins = design["pins"]
    for name, path in (("m4_06_design", HERE.parent / "m4_06" / "design.json"),
                       ("m5_02_design", HERE.parent / "m5_02" / "design.json")):
        if sha256(path) != pins[name]:
            raise ValueError(f"{name} is not the pinned one")
    base, _ = m6.load_design()
    return design, base


def digests():
    records = json.loads((HERE / "digests.json").read_text())["days"]
    return {(r["simulator"], r["day"]): r for r in records}


def days_of(design, split):
    first, last = design["splits"][split]
    return list(range(first, last + 1))


def split_of(design, day):
    return next(s for s in ("train", "validation", "holdout") if day in days_of(design, s))


def day_path(data_dir, simulator, day, design, suffix=""):
    return pathlib.Path(data_dir) / split_of(design, day) / simulator / f"day-{day}{suffix}.json"


def read_rows(data_dir, simulator, day, design, holdout_allowed=False):
    split = split_of(design, day)
    if split == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    record = digests()[(simulator, day)]
    rows = [tuple(r) for r in json.loads(day_path(data_dir, simulator, day, design).read_text())]
    return rows, record


def read_alarm(data_dir, fitted, simulator, day, design, holdout_allowed=False):
    """Per row: the alarm (0 or 1) and the ratio, None before the detector has its window."""
    if split_of(design, day) == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    path = day_path(data_dir, simulator, day, design, f".alarm-{fitted}")
    return [(int(a), None if r is None else float(r)) for a, r, _ in json.loads(path.read_text())]


def latent(design, day, rows):
    """The latent regime index of each row: row `r` is second `r` of the day."""
    regime = [0] * rows
    for name, start, end in design["schedule"]["segments"].get(str(day), []):
        for second in range(start, end):
            regime[second] = REGIMES.index(name)
    return regime


def volatility(data_dir, design, base, simulator):
    """#103's volatility rule over this simulator's training days."""
    tick = Decimal(base["kernel"]["tick_size"])
    changes = []
    for day in days_of(design, "train"):
        rows, _ = read_rows(data_dir, simulator, day, design)
        mids = [Decimal(r[1] + r[3]) / 2 * tick for r in rows]
        changes.extend(float(b - a) for a, b in zip(mids, mids[1:]))
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)
    seconds = base["simulator"]["sample_ms"] / 1000
    return f"{(variance * m6.SECONDS_PER_YEAR / seconds) ** 0.5:.4f}"


def fixed_point(design, sigma):
    """#103's calm Avellaneda-Stoikov point at one row, as #107's design pins it, at `sigma`."""
    m5_02 = json.loads((HERE.parent / "m5_02" / "design.json").read_text())
    return {**m5_02["references"]["points"]["calm"][AS], "volatility": sigma}


def changes_of(labels):
    """`[row, index]` at every row where the index differs from the one before (index 0 first)."""
    out, before = [], 0
    for row, index in enumerate(labels):
        if index != before:
            out.append([row, index])
            before = index
    return out


def scheduled(points, labels, day):
    """The scheduled baseline of #296 over Avellaneda-Stoikov points, by a label per row."""
    return {"kind": "scheduled",
            "points": [m6.baseline_config(AS, p, day) for p in points],
            "changes": changes_of(labels)}


class DetectorFeatures(gym.Wrapper):
    """The observation, then the alarm and `min(ratio, 5) / 5` at the row the agent observed;
    both 0 before the detector has its window."""

    def __init__(self, env, alarm):
        super().__init__(env)
        self._alarm = alarm
        low, high = env.observation_space.low, env.observation_space.high
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((low, np.zeros(2, dtype=np.float32))),
            high=np.concatenate((high, np.ones(2, dtype=np.float32))), dtype=np.float32)

    def _extend(self, observation, info):
        alarm, ratio = self._alarm[info["row"]] if info.get("row") is not None else (0, None)
        extra = [float(alarm), 0.0 if ratio is None else min(ratio, RATIO_CAP) / RATIO_CAP]
        return np.concatenate((np.asarray(observation, dtype=np.float32),
                               np.asarray(extra, dtype=np.float32)))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._extend(observation, info), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._extend(observation, info), reward, terminated, truncated, info


def build_env(job, holdout_allowed=False):
    """One day of `job["simulator"]` for one arm, calibrated on `job["fitted"]`."""
    design, base = load()
    sim, day = job["simulator"], job["day"]
    rows, record = read_rows(job["data_dir"], sim, day, design, holdout_allowed)
    config = m6.base_config(base, job["fee"])
    execution = job.get("execution", "design")
    config["fill_model"] = dict(design["execution"]["fill_model"] if execution == "design"
                                else design["execution"]["execution_change"]["fill_model"])
    arm = job["arm"]
    if arm in CLASSICAL_ARMS:
        points = job["points"]
        if arm == "fixed":
            baseline = m6.baseline_config(AS, points[0], day)
        elif arm == "adaptive":
            alarm = read_alarm(job["data_dir"], job["fitted"], sim, day, design, holdout_allowed)
            baseline = scheduled(points, [a for a, _ in alarm], day)
        else:
            baseline = scheduled(points, latent(design, day, len(rows)), day)
        config["residual"] = {"baseline": baseline, "price_ticks": 4, "size_lots": 1,
                              "default_order_size": "1"}
        env = ResidualMarketMakingEnv(config, rows)
    else:
        env = BoxMarketMakingEnv(config, rows)
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{sim} day {day} has other rows than digests.json")
    env = DecisionInterval(env, EVERY)
    if arm in LEARNED_ARMS:
        if arm == "learned_detector":
            env = DetectorFeatures(env, read_alarm(job["data_dir"], job["fitted"], sim, day,
                                                   design, holdout_allowed))
        memory = job["memory"]
        if memory == "ema":
            env = EmaFeatures(env, tuple(base["arms"]["ema"]["half_lives_decisions"]))
        elif memory == "history":
            env = history(env, base["arms"]["history"]["length"])
    return env, rows


def markouts(positions, rows, tick):
    """Signed markouts of the fills at each horizon, summed over the day, in price units: a buy
    gains when the mid rises after it. A fill with no row at its horizon is left out."""
    times = np.asarray([r[0] // 1_000_000 for r in rows], dtype=np.int64)
    mids = [Decimal(r[1] + r[3]) / 2 * tick for r in rows]
    out = {str(h): Decimal(0) for h in MARKOUT_MS}
    before = Decimal(0)
    for timestamp, price, after in positions:
        size = Decimal(after) - before
        before = Decimal(after)
        for h in MARKOUT_MS:
            at = np.searchsorted(times, timestamp + h)
            if at < len(rows):
                out[str(h)] += size * (mids[at] - Decimal(price))
    return {k: str(v) for k, v in out.items()}


def run_episode(policy, env, rows, recurrent, base, seed):
    day = m6.episode(policy, env, seed, recurrent, base)
    attribution = dict(env.unwrapped.attribution())
    day["markouts"] = markouts(attribution["positions"], rows, Decimal(base["kernel"]["tick_size"]))
    return day


def frozen_policy(meta, env):
    path = pathlib.Path(meta["path"]).with_suffix(".zip")
    if sha256(path) != meta["checkpoint_sha256"]:
        raise ValueError(f"{path} is not the checkpoint its metadata names")
    base_env = env.unwrapped
    if tuple(meta["spec"]) != base_env.spec_id:
        raise ValueError("a checkpoint of another specification")
    keep = ("fee_rate", "fill_model")
    if ({k: v for k, v in meta["config"].items() if k not in keep}
            != {k: v for k, v in base_env.config.items() if k not in keep}):
        raise ValueError("a checkpoint of another configuration")
    algorithm, _ = m6.model_class(meta["memory"])
    model = algorithm.load(path, device="cpu")
    model.policy.set_training_mode(False)
    return model


def evaluate_job(job):
    torch.set_num_threads(1)
    _, base = load()
    env, rows = build_env(job, job["split"] == "holdout")
    try:
        kind = job["policy"]["kind"]
        if kind == "hold":
            policy = m6.Constant(m6.HOLD)
        elif kind == "zero_residual":
            policy = m6.ZeroResidual()
        else:
            policy = frozen_policy(job["policy"]["metadata"], env)
        recurrent = job.get("memory") == "lstm"
        days = [run_episode(policy, env, rows, recurrent, base, s)
                for s in base["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets_ = [Decimal(d["net"]) for d in days]
    keys = ("arm", "candidate", "fee", "fitted", "simulator", "execution", "day", "split")
    return {key: job.get(key) for key in keys} | {
        "daily_value": str(sum(nets_, Decimal(0)) / len(nets_)), "episodes": days}


def train_job(job):
    torch.set_num_threads(1)
    design, base = load()
    set_random_seed(job["seed"])
    envs = [build_env({**job, "day": day, "simulator": job["fitted"]})[0]
            for day in days_of(design, "train")]
    cycle = m6.DayCycle(envs, job["seed"])
    algorithm, policy = m6.model_class(job["memory"])
    hyper = base["hyperparameters"]
    kwargs = {k: hyper[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef")}
    started = time.perf_counter()
    model = algorithm(policy, DummyVecEnv([lambda: Monitor(cycle)]), seed=job["seed"],
                      device="cpu", verbose=0, **kwargs)
    model.learn(total_timesteps=base["budget_decisions"], progress_bar=False)
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    model.save(path)
    metadata = {
        "arm": job["arm"], "memory": job["memory"], "fitted": job["fitted"], "seed": job["seed"],
        "fee": job["fee"], "decisions": base["budget_decisions"], "every": EVERY,
        "m4_06_validation_sha256": job["m4_06_validation_sha256"],
        "spec": list(envs[0].unwrapped.spec_id), "config": envs[0].unwrapped.config,
        "training_digests": [e.unwrapped.bridge_info["dataset_digest"] for e in envs],
        "commit": job["commit"], "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(), "train_seconds": round(seconds, 1),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path),
    }
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def mean_daily(rows):
    values = [Decimal(r["daily_value"]) for r in rows]
    return sum(values, Decimal(0)) / len(values)


class Context:
    def __init__(self, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.base = load()
        self.data_dir = data_dir
        self.out = pathlib.Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.fee = self.design["fee"]["value"]
        self.secondary_fee = self.design["fee"]["secondary"]

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "fee": self.fee, "design_sha256": sha256(HERE / "design.json"),
                "digests_sha256": sha256(HERE / "digests.json"), "versions": versions()}

    def job(self, arm, fitted, simulator, day, candidate, policy, points=None, fee=None,
            execution="design", memory=None):
        return {"arm": arm, "fitted": fitted, "simulator": simulator, "day": day,
                "split": split_of(self.design, day), "candidate": candidate, "policy": policy,
                "points": points, "fee": fee or self.fee, "execution": execution,
                "memory": memory, "data_dir": self.data_dir}

    def write(self, name, report):
        path = self.out / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; a step runs once")
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return sha256(path)

    def classical(self):
        return json.loads((self.out / "classical.json").read_text())

    def arm_points(self, fitted):
        chosen = self.classical()["selected"][fitted]
        return {"fixed": [chosen["fixed"]], "adaptive": [chosen["fixed"], chosen["stressed"]],
                "oracle": [chosen["oracle"][r] for r in REGIMES]}

    def train_jobs(self, memory=None, validation_sha=None):
        jobs = []
        for fitted in SIMULATORS:
            for arm in LEARNED_ARMS:
                for seed in self.base["training_seeds"]:
                    jobs.append({"arm": arm, "memory": memory, "fitted": fitted, "seed": seed,
                                 "fee": self.fee, "data_dir": self.data_dir,
                                 "commit": self.commit, "m4_06_validation_sha256": validation_sha,
                                 "path": str(self.out / "checkpoints" / f"{arm}-{fitted}-seed{seed}")})
        return jobs

    def runs(self):
        out = []
        for path in sorted((self.out / "checkpoints").glob("*.json")):
            out.append(json.loads(path.read_text()))
        expected = len(SIMULATORS) * len(LEARNED_ARMS) * len(self.base["training_seeds"])
        if len(out) != expected:
            raise FileNotFoundError(f"{len(out)} of {expected} runs are trained")
        return out

    def alarm_files(self):
        """The frozen detector's files only, one per simulator fitted on: not the tagged files of
        a post hoc sensitivity run."""
        root = pathlib.Path(self.data_dir)
        return sorted(p for fit in SIMULATORS for p in root.glob(f"*/*/day-*.alarm-{fit}.json"))


def check(ctx):
    """The queue standard: the fixed arm fills under the design's model on training day 1 of
    each simulator, and its net differs from immediate fill."""
    day = days_of(ctx.design, "train")[0]
    report = {}
    for sim in SIMULATORS:
        point = fixed_point(ctx.design, volatility(ctx.data_dir, ctx.design, ctx.base, sim))
        row = {"point": point}
        for execution in EXECUTIONS:
            job = ctx.job("fixed", sim, sim, day, "check", {"kind": "zero_residual"}, [point],
                          execution=execution)
            env, rows = build_env(job)
            try:
                episode = m6.episode(m6.ZeroResidual(), env, 101, False, ctx.base)
            finally:
                env.close()
            row[execution] = {"net": episode["net"], "fills": episode["orders"]["fills"]}
        row["binds"] = row["design"]["net"] != row["immediate"]["net"]
        row["passes"] = row["binds"] and row["design"]["fills"] > 0
        report[sim] = row
        print(sim, json.dumps(row), flush=True)
    print(ctx.write("check", {**ctx.header("check"), "day": day, "rows": report}))
    if not all(r["passes"] for r in report.values()):
        raise SystemExit("the queue does not bind, or the fixed arm does not fill")


def classical(ctx, parallel):
    """On validation: the adaptive arm's stressed point and the coordinate-search oracle's
    points, each by #103's rule (the highest mean daily net, ties by the candidate's text)."""
    days = days_of(ctx.design, "validation")
    zero = {"kind": "zero_residual"}
    jobs, sigmas, grids, fixed = [], {}, {}, {}
    for sim in SIMULATORS:
        sigmas[sim] = volatility(ctx.data_dir, ctx.design, ctx.base, sim)
        fixed[sim] = fixed_point(ctx.design, sigmas[sim])
        grids[sim] = m6.grid_points(ctx.base["arms"][AS]["grid"], sigmas[sim])
        for point in grids[sim]:
            text = json.dumps(point, sort_keys=True)
            jobs += [ctx.job("adaptive", sim, sim, d, f"stressed={text}", zero, [fixed[sim], point])
                     for d in days]
            for index, regime in enumerate(REGIMES):
                points = [fixed[sim]] * len(REGIMES)
                points[index] = point
                jobs += [ctx.job("oracle", sim, sim, d, f"{regime}={text}", zero, points)
                         for d in days]
    rows = m6.pool(jobs, evaluate_job, parallel, "classical")
    selected = {}
    for sim in SIMULATORS:

        def best(arm, prefix):
            mine = [r for r in rows if r["fitted"] == sim and r["arm"] == arm
                    and r["candidate"].startswith(prefix)]
            candidates = sorted({r["candidate"] for r in mine})
            top = max(candidates,
                      key=lambda c: (mean_daily([r for r in mine if r["candidate"] == c]), c))
            return json.loads(top[len(prefix):])

        selected[sim] = {"fixed": fixed[sim], "stressed": best("adaptive", "stressed="),
                         "oracle": {r: best("oracle", f"{r}=") for r in REGIMES}}
    print(json.dumps(selected), ctx.write("classical", {
        **ctx.header("classical"), "volatility": sigmas, "selected": selected, "rows": rows,
        "oracle": "a coordinate-search oracle: each regime's point chosen with the others at "
                  "the fixed point; not the joint optimum"}))


def train(ctx, parallel, memory, validation_sha):
    if memory not in m6.LEARNED:
        raise SystemExit(f"the learned arm is one of {m6.LEARNED}")
    if (ctx.out / "frozen.json").exists():
        raise SystemExit("already frozen")
    todo = [j for j in ctx.train_jobs(memory, validation_sha)
            if not pathlib.Path(j["path"]).with_suffix(".json").exists()]
    todo.sort(key=lambda j: j["arm"] != "learned_detector")
    for meta in m6.pool(todo, train_job, parallel, "train"):
        print(meta["arm"], meta["fitted"], meta["seed"], meta["train_seconds"], flush=True)


def freeze(ctx):
    entries = {}
    for meta in ctx.runs():
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if sha256(zip_path) != meta["checkpoint_sha256"]:
            raise SystemExit(f"{zip_path} changed")
        if meta["design_sha256"] != sha256(HERE / "design.json"):
            raise SystemExit(f"{zip_path} was trained under another design")
        entries[zip_path.stem] = {"checkpoint": meta["checkpoint_sha256"],
                                  "metadata": sha256(zip_path.with_suffix(".json"))}
    alarms = {str(p.relative_to(ctx.data_dir)): sha256(p) for p in ctx.alarm_files()}
    if len(alarms) != len(SIMULATORS) ** 2 * ctx.design["days"]:
        raise SystemExit(f"{len(alarms)} alarm files")
    frozen = {"design_sha256": sha256(HERE / "design.json"),
              "digests_sha256": sha256(HERE / "digests.json"),
              "classical_sha256": sha256(ctx.out / "classical.json"),
              "runs": entries, "alarms": alarms}
    print(ctx.write("frozen", frozen))


def holdout_jobs(ctx, runs):
    """Every arm on every holdout day of both simulators under both executions, calibrated on
    each simulator, at the fee; at the secondary fee beside it, on the own simulator under the
    design's execution only."""
    days = days_of(ctx.design, "holdout")
    hold = {"kind": "hold"}
    zero = {"kind": "zero_residual"}
    jobs = []
    for fee in dict.fromkeys([ctx.fee, ctx.secondary_fee]):
        for fitted in SIMULATORS:
            points = ctx.arm_points(fitted)
            for sim in SIMULATORS:
                for execution in EXECUTIONS:
                    if fee != ctx.fee and (sim != fitted or execution != "design"):
                        continue
                    for day in days:
                        common = {"fee": fee, "execution": execution}
                        jobs.append(ctx.job("no_trade", fitted, sim, day, "hold", hold, **common))
                        for arm in CLASSICAL_ARMS:
                            jobs.append(ctx.job(arm, fitted, sim, day, arm, zero, points[arm],
                                                **common))
                        for meta in runs:
                            if meta["fitted"] == fitted:
                                jobs.append(ctx.job(meta["arm"], fitted, sim, day,
                                                    f"seed{meta['seed']}",
                                                    {"kind": "frozen", "metadata": meta},
                                                    memory=meta["memory"], **common))
    return jobs


def holdout(ctx, frozen_sha, parallel):
    if sha256(ctx.out / "frozen.json") != frozen_sha:
        raise SystemExit("the frozen manifest differs")
    frozen = json.loads((ctx.out / "frozen.json").read_text())
    for name, path in (("design_sha256", HERE / "design.json"),
                       ("digests_sha256", HERE / "digests.json"),
                       ("classical_sha256", ctx.out / "classical.json")):
        if frozen[name] != sha256(path):
            raise SystemExit(f"{path.name} differs from the frozen one")
    runs = ctx.runs()
    for meta in runs:
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        entry = frozen["runs"][zip_path.stem]
        if (entry["checkpoint"] != sha256(zip_path)
                or entry["metadata"] != sha256(zip_path.with_suffix(".json"))):
            raise SystemExit(f"{zip_path.stem} changed after the freeze")
    alarms = {str(p.relative_to(ctx.data_dir)): sha256(p) for p in ctx.alarm_files()}
    if alarms != frozen["alarms"]:
        raise SystemExit("the alarm files differ from the frozen ones")
    if (ctx.out / "holdout.json").exists():
        raise SystemExit("the holdout runs once")
    rows = m6.pool(holdout_jobs(ctx, runs), evaluate_job, parallel, "holdout")
    print(ctx.write("holdout", {**ctx.header("holdout"), "frozen_sha256": frozen_sha,
                                "secondary_fee": ctx.secondary_fee, "rows": rows}))


def detector_metrics(ctx, fitted, sim):
    """Delay to each holdout change, missed changes, false alarms per calm day and time in each
    alarm state against the latent regime, from the alarm files."""
    out = {"changes": [], "calm_onsets": 0, "seconds": {}}
    calm_days = 0
    for day in days_of(ctx.design, "holdout"):
        alarm = [a for a, _ in read_alarm(ctx.data_dir, fitted, sim, day, ctx.design, True)]
        labels = latent(ctx.design, day, len(alarm))
        onsets = [r for r in range(len(alarm)) if alarm[r] and (r == 0 or not alarm[r - 1])]
        for regime, state in zip(labels, alarm):
            key = f"{REGIMES[regime]}/{'on' if state else 'off'}"
            out["seconds"][key] = out["seconds"].get(key, 0) + 1
        segments = ctx.design["schedule"]["segments"].get(str(day), [])
        if not segments:
            calm_days += 1
            out["calm_onsets"] += len(onsets)
        for name, start, end in segments:
            # Design change (h): the first onset at or after the start; an alarm already on at
            # the start (latched from a false alarm) is not a detection.
            inside = [r for r in onsets if start <= r < end]
            delay = inside[0] - start if inside else None
            out["changes"].append({"day": day, "regime": name, "start": start, "delay_s": delay,
                                   "missed": delay is None, "on_at_start": bool(alarm[start])})
    out["false_alarms_per_calm_day"] = out["calm_onsets"] / calm_days
    return out


def daily(rows, arm, fitted, sim, execution, fee, days):
    """An arm's daily net; a learned arm's day is the mean over its seeds, as in #103."""
    out = []
    for day in days:
        values = [Decimal(r["daily_value"]) for r in rows
                  if r["arm"] == arm and r["fitted"] == fitted and r["simulator"] == sim
                  and r["execution"] == execution and r["fee"] == fee and r["day"] == day]
        out.append(sum(values, Decimal(0)) / len(values))
    return out


def episodes(rows, arm, fitted, sim, execution, fee):
    return [e for r in rows if r["arm"] == arm and r["fitted"] == fitted
            and r["simulator"] == sim and r["execution"] == execution and r["fee"] == fee
            for e in r["episodes"]]


def summary(rows, arm, fitted, sim, execution, fee, days):
    values = daily(rows, arm, fitted, sim, execution, fee, days)
    eps = episodes(rows, arm, fitted, sim, execution, fee)

    def mean(key):
        items = [Decimal(key(e)) for e in eps]
        return str(sum(items, Decimal(0)) / len(items))

    worst = max(1, -(-len(values) * 5 // 100))
    return {
        "mean_daily_net": str(sum(values, Decimal(0)) / len(values)),
        "cvar_5": str(sum(sorted(values)[:worst], Decimal(0)) / worst),
        "mean_max_drawdown": mean(lambda e: e["attribution"]["max_drawdown"]),
        "mean_fees": mean(lambda e: e["attribution"]["quote_fees"]),
        "mean_spread_capture": mean(lambda e: e["attribution"]["spread_capture"]),
        "mean_inventory_pnl": mean(lambda e: e["attribution"]["inventory_pnl"]),
        "markout": {str(h): mean(lambda e, h=h: e["markouts"][str(h)]) for h in MARKOUT_MS},
        "peak_notional": str(max(Decimal(e["peak_notional"]) for e in eps)),
        "breach": any(e["limit_breach"] for e in eps),
        "fills": sum(e["orders"]["fills"] for e in eps),
    }


def report(ctx, binary, name="report"):
    design, rule = ctx.design, ctx.design["decision_rule"]
    rows = json.loads((ctx.out / "holdout.json").read_text())["rows"]
    days = days_of(design, "holdout")
    pairs = [("adaptive", "fixed"), ("learned_detector", "learned")]
    series, keys = {}, []
    for fee in dict.fromkeys([ctx.fee, ctx.secondary_fee]):
        for fitted in SIMULATORS:
            for sim in SIMULATORS:
                for execution in EXECUTIONS:
                    if not any(r["fee"] == fee and r["fitted"] == fitted and r["simulator"] == sim
                               and r["execution"] == execution for r in rows):
                        continue
                    group = f"{fee}/{fitted}->{sim}/{execution}"
                    keys.append((fee, fitted, sim, execution, group))
                    none = daily(rows, "no_trade", fitted, sim, execution, fee, days)
                    for arm in ("fixed", "adaptive", "oracle", "learned", "learned_detector"):
                        mine = daily(rows, arm, fitted, sim, execution, fee, days)
                        series[f"{group}/{arm}-minus-no_trade"] = [str(a - b) for a, b in zip(mine, none)]
                    for a, b in pairs:
                        mine = daily(rows, a, fitted, sim, execution, fee, days)
                        theirs = daily(rows, b, fitted, sim, execution, fee, days)
                        series[f"{group}/{a}-minus-{b}"] = [str(x - y) for x, y in zip(mine, theirs)]
    path = ctx.out / "series.json"
    path.write_text(json.dumps({"bootstrap": ctx.base["evaluation"]["bootstrap"], "series": series}))
    intervals = json.loads(subprocess.run([binary, str(path)], check=True, capture_output=True,
                                          text=True).stdout)
    detectors = {f"{f}->{s}": detector_metrics(ctx, f, s) for f in SIMULATORS for s in SIMULATORS}
    groups = {}
    for fee, fitted, sim, execution, group in keys:
        arms = {arm: summary(rows, arm, fitted, sim, execution, fee, days)
                for arm in ("no_trade",) + CLASSICAL_ARMS + LEARNED_ARMS}
        entry = {"arms": arms, "intervals": {k[len(group) + 1:]: v for k, v in intervals.items()
                                             if k.startswith(group + "/")}}
        if fitted != sim:
            entry["transfer_failures"] = [
                arm for arm in CLASSICAL_ARMS + LEARNED_ARMS
                if Decimal(arms[arm]["mean_daily_net"]) < Decimal(arms["no_trade"]["mean_daily_net"])]
        groups[group] = entry
    decisions = {}
    factor = Decimal("1.2")
    for fee, fitted, sim, execution, group in keys:
        if fee != ctx.fee or fitted != sim:
            continue
        detector = detectors[f"{fitted}->{sim}"]
        stress = [c for c in detector["changes"] if c["regime"] == "stressed"]
        timely = all(not c["missed"] and c["delay_s"] <= 300 for c in stress)
        quiet = detector["false_alarms_per_calm_day"] <= 2
        for a, b in pairs:
            interval = intervals[f"{group}/{a}-minus-{b}"]
            mine, theirs = groups[group]["arms"][a], groups[group]["arms"][b]
            checks = {
                "interval_above_zero": Decimal(interval["lower"]) > 0,
                "drawdown_ok": Decimal(mine["mean_max_drawdown"])
                <= factor * Decimal(theirs["mean_max_drawdown"]),
                "no_breach": not mine["breach"],
                "stress_detected_within_300_s": timely,
                "false_alarms_ok": quiet,
            }
            decisions.setdefault(f"{a}-vs-{b}", {})[f"{sim}/{execution}"] = {
                **checks, "interval": interval, "go": all(checks.values())}
    verdict = {name: all(g["go"] for g in by.values()) for name, by in decisions.items()}
    result = {**ctx.header("report"), "rule": rule, "groups": groups, "detectors": detectors,
              "decisions": decisions, "go": verdict,
              "not_run": {"S3": "#106 had not adopted its neural simulator at the freeze"}}
    print(json.dumps(verdict), ctx.write(name, result))


def main():
    step = sys.argv[1]
    if step == "check":
        check(Context(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif step == "classical":
        classical(Context(sys.argv[2], sys.argv[3], sys.argv[4]), int(sys.argv[5]))
    elif step == "train":
        train(Context(sys.argv[2], sys.argv[3], sys.argv[4]), int(sys.argv[5]), sys.argv[6],
              sys.argv[7])
    elif step == "freeze":
        freeze(Context(sys.argv[2], sys.argv[3], None))
    elif step == "holdout":
        holdout(Context(sys.argv[2], sys.argv[3], sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    elif step == "report":
        report(Context(sys.argv[2], sys.argv[3], None), sys.argv[4], *sys.argv[5:6])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
