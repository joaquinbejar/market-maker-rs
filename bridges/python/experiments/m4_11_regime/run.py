"""Runs a step of #309's offline study with the regime in the input, from its committed design.

From `bridges/python`, with this study's new days and #102's run directory:

    python experiments/m4_11_regime/run.py <step> <data dir> <#102 run dir> <attempts root> \
        <attempt> <commit> [args]

Steps, in order:
- `pilot`: writes budget.json with the time estimates, on training days;
- `gate`: the queue gate and the determinism check, on validation days;
- `logs <dataset> <split>`: a dataset's logs, on train or validation days;
- `audit`: the aliasing diagnosis and the input's range and precision, on training logs;
- `train <dataset> bc`, then `sanity <dataset>`: standard_eps30 first, then standard_eps10;
- `train <dataset> cql <alpha>`: only after both checks;
- `ppo <regime|blind>`: the interactive comparators;
- `select`: CQL's alpha, by interactive runs on validation days;
- `freeze`: the hashes of the design, digests, budget, code, choice, logs and checkpoints;
- `holdout <frozen sha256>`: every arm on the holdout days, once;
- `report`: the paired intervals of the holdout.

Everything here is synthetic: #102's simulator on new days, whose rows do not react to orders.
"""

import concurrent.futures
import importlib.util
import json
import math
import multiprocessing
import os
import pathlib
import sys
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = _module("m4_11_run", EXPERIMENTS / "m4_11" / "run.py")
regime_of = _module("m4_11_regime_input", HERE / "regime.py")
base, m4_07, est, offline = prior.base, prior.m4_07, prior.est, prior.offline
sha256 = prior.sha256

from mm_rl_bridge import MaskableMarketMakingEnv  # noqa: E402

SEED = 101
POSITION = 6
CODE = ("run.py", "regime.py")
SHARED_CODE = (("m4_11", "offline.py"), ("m4_11", "run.py"), ("m4_07", "run.py"),
               ("m4_07", "estimate.py"), ("m4_05", "run.py"))
PRIMARY, SECONDARY = "standard_eps30", "standard_eps10"
BLOCK, REPLICATES, BOOTSTRAP_SEED = 2.0, 2000, 309
THRESHOLD = 0.9
OVERRUN = 1.5
PILOT_GRADIENT_STEPS = 5000
PILOT_PPO_STEPS = 20000


def load():
    design = json.loads((HERE / "design.json").read_text())
    reused = design["reuses"]
    for path, key in ((EXPERIMENTS / "m4_05" / "design.json", "m4_05_design_sha256"),
                      (EXPERIMENTS / "m4_07" / "design.json", "m4_07_design_sha256")):
        if sha256(path) != reused[key]:
            raise ValueError(f"{path.parent.name}'s design is not the one this study reuses")
    m4_05 = json.loads((EXPERIMENTS / "m4_05" / "design.json").read_text())
    digests = json.loads((HERE / "digests.json").read_text())
    return design, m4_05, {(d["regime"], d["day"]): d for d in digests["days"]}


def days_input_matches(days_input, m4_05, offset):
    """Whether the generator's input is #102's design with only the seed offset added."""
    rest = {k: v for k, v in days_input.items() if k != "seed_offset"}
    return rest == m4_05 and days_input.get("seed_offset") == offset


class RegimeObservation:
    """An environment whose observations carry the day's regime; everything else is its own."""

    def __init__(self, env, regime):
        self._env, self._regime = env, regime
        space = env.observation_space
        ones = np.ones(len(regime_of.REGIMES), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.concatenate([space.low, 0 * ones]),
                                                np.concatenate([space.high, ones]),
                                                dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        observation, info = self._env.reset(**kwargs)
        return regime_of.with_regime(observation, self._regime), info

    def step(self, action):
        observation, *rest = self._env.step(action)
        return (regime_of.with_regime(observation, self._regime), *rest)


def make_env(job):
    _, m4_05, digests = load()
    rows, record = base.read_rows(job["data_dir"], digests, job["regime"], job["day"], job["split"],
                                  holdout_allowed=job.get("holdout_allowed", False))
    return base.checked(MaskableMarketMakingEnv(job["config"], rows), record), m4_05


def policy_of(job, env, grid, m4_05):
    """The evaluated arm on (observation, raw, mask), and whether its support is measured."""
    spec, regime = job["policy"], job["regime"]
    if spec["kind"] == "rule":
        return prior.rule(job, regime, grid, m4_05, spec["name"]), False
    if spec["kind"] == "ppo":
        model = base.load_frozen(spec["metadata"], env, regime)
        if spec["with_regime"]:
            def act(observation, raw, mask):
                return model.predict(regime_of.with_regime(observation, regime),
                                     deterministic=True, action_masks=mask)[0]
            return act, False
        return m4_07.frozen_ppo(model), False
    actions = offline.Actions(len(grid.offsets), len(grid.sizes))
    trained = offline.Policy(prior.network_of(spec["metadata"]), actions)
    return (lambda observation, raw, mask: trained.predict(
        regime_of.with_regime(observation, regime), action_masks=mask)[0]), True


def support_bins(observations, positions, edges, regime):
    """#280's bin crossed with the regime."""
    bins = prior.state_bins(observations, positions, edges)
    return bins * len(regime_of.REGIMES) + regime_of.REGIMES.index(regime)


def run_job(job):
    """One arm, one day, at the environment seed: the exact net and, for an offline arm, the
    support of its decisions."""
    torch.set_num_threads(1)
    env, m4_05 = make_env(job)
    grid = m4_07.grid_of(job["config"])
    try:
        policy, measured = policy_of(job, env, grid, m4_05)
        behaviour = None
        if measured:
            greedy_rule = prior.rule(job, job["regime"], grid, m4_05, "as_grid")
            behaviour = m4_07.Behaviour(grid, greedy_rule, job["epsilon"], 0)
        observation, info = env.reset(seed=job["seed"])
        steps = refusals = 0
        seen = {"observation": [], "code": [], "position": [], "propensity": []}
        scale = float(job["config"]["scales"]["position"])
        while True:
            raw, mask = info["raw_observation"], env.action_masks()
            action = grid.canonical(policy(observation, raw, mask))
            if behaviour is not None:
                greedy = grid.canonical(behaviour.greedy(observation, raw, mask))
                seen["observation"].append(observation)
                seen["code"].append(grid.code(action))
                seen["position"].append(float(observation[POSITION]) * scale)
                seen["propensity"].append(behaviour.probability(mask, greedy, action))
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            steps += 1
            refusals += info["refused_because"] is not None
            if terminated or truncated:
                break
        attribution = dict(env.attribution())
    finally:
        env.close()
    limits = job["config"]["limits"]
    peak = max((abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
               default=Decimal(0))
    row = {"arm": job["arm"], "seed_of_arm": job.get("seed_of_arm"), "regime": job["regime"],
           "day": job["day"], "split": job["split"], "execution": job["execution"],
           "environment_seed": job["seed"], "steps": steps, "net": info["net"],
           "fills": attribution.get("fills"), "max_abs_position": attribution.get("max_abs_position"),
           "refusals": refusals, "peak_notional": str(peak),
           "time_beyond_5_lots": base.time_beyond(attribution, 5),
           "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
           or peak > Decimal(limits["max_notional"]),
           "label": "[simulated]"}
    if measured:
        row["support_raw"] = {k: np.asarray(v).tolist() for k, v in seen.items()
                              if k != "observation"}
        row["support_bins"] = support_bins(np.asarray(seen["observation"]),
                                           np.asarray(seen["position"]), job["edges"],
                                           job["regime"]).tolist()
    return row


def log_job(job):
    """#280's log of one behaviour, seed and day, plus the raw position and steps left that
    as_grid reads, which the aliasing diagnosis keys on."""
    torch.set_num_threads(1)
    env, m4_05 = make_env(job)
    grid = m4_07.grid_of(job["config"])
    try:
        greedy_rule = prior.rule(job, job["regime"], grid, m4_05, "as_grid")
        behaviour = m4_07.Behaviour(grid, greedy_rule, job["epsilon"], job["seed"])
        observation, info = env.reset(seed=job["seed"])
        rows = {k: [] for k in ("observation", "mask", "action", "propensity", "reward", "refused",
                                "target_code", "target_mu", "raw_position", "raw_steps_left")}
        while True:
            mask, raw = env.action_masks(), info["raw_observation"]
            action, greedy, propensity = behaviour.act(observation, raw, mask)
            rows["observation"].append(observation)
            rows["mask"].append(mask)
            rows["action"].append(grid.code(action))
            rows["propensity"].append(propensity)
            rows["target_code"].append(grid.code(greedy))
            rows["target_mu"].append(behaviour.probability(mask, greedy, greedy))
            rows["raw_position"].append(
                float(Decimal(raw[m4_07.POSITION])) if raw[m4_07.POSITION] is not None else 0.0)
            rows["raw_steps_left"].append(int(raw[m4_07.STEPS_LEFT]))
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            rows["reward"].append(float(Decimal(info["reward_exact"])))
            rows["refused"].append(info["refused_because"] is not None)
            if terminated or truncated:
                break
    finally:
        env.close()
    path = pathlib.Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, observation=np.asarray(rows["observation"], dtype=np.float32),
        mask=np.asarray(rows["mask"], dtype=bool), action=np.asarray(rows["action"], dtype=np.int16),
        propensity=np.asarray(rows["propensity"]), reward=np.asarray(rows["reward"]),
        refused=np.asarray(rows["refused"], dtype=bool), target_names=np.asarray(["as_grid"]),
        target_code=np.asarray([rows["target_code"]], dtype=np.int16),
        target_mu=np.asarray([rows["target_mu"]]),
        raw_position=np.asarray(rows["raw_position"]),
        raw_steps_left=np.asarray(rows["raw_steps_left"], dtype=np.int64))
    return {"regime": job["regime"], "day": job["day"], "split": job["split"],
            "behaviour": job["behaviour"], "seed": job["seed"], "steps": len(rows["action"]),
            "net": info["net"], "path": str(path), "sha256": sha256(path), "label": "[observed]"}


def days_with_regime(records):
    """The logged days, checked by sha256, with the regime appended to the observations."""
    days = prior.read_logs(records)
    for record, day in zip(records, days, strict=True):
        day["observation"] = regime_of.with_regime(day["observation"], record["regime"])
    return days


def train_job(job):
    torch.set_num_threads(1)
    data = offline.Transitions(offline.Actions(job["k"], job["m"]), days_with_regime(job["logs"]),
                               job["reward_scale"])
    started = time.perf_counter()
    network, history, finished = offline.train(job["arm"], data, job["seed"], job["settings"])
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)
    metadata = {"dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
                "settings": job["settings"], "reward_scale": job["reward_scale"],
                "inputs": int(data.observation.shape[1]), "outputs": int(data.mask.shape[1]),
                "transitions": len(data), "finished": finished, "history": history,
                "train_seconds": round(seconds, 2), "path": str(path),
                "checkpoint_sha256": sha256(path), "commit": job["commit"],
                "design_sha256": sha256(HERE / "design.json")}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def ppo_job(job):
    """#102's maskable PPO recipe on this study's training days, with or without the regime."""
    torch.set_num_threads(1)
    _, _, digests = load()
    set_random_seed(job["seed"])
    envs, configs, keys = [], {}, []
    for regime in job["regimes"]:
        for day in job["days"]:
            rows, record = base.read_rows(job["data_dir"], digests, regime, day, "train")
            env = base.checked(MaskableMarketMakingEnv(job["config"], rows), record)
            configs.setdefault(regime, env.config)
            keys.append(env.bridge_info["dataset_digest"])
            envs.append(RegimeObservation(env, regime) if job["with_regime"] else env)
    cycle = base.DayCycle(envs, job["seed"])
    start = time.perf_counter()
    model = MaskablePPO("MlpPolicy", DummyVecEnv([lambda: Monitor(cycle)]), seed=job["seed"],
                        device="cpu", **job["hyperparameters"])
    model.learn(total_timesteps=job["steps"], progress_bar=False)
    seconds = time.perf_counter() - start
    path = pathlib.Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    metadata = {"algorithm": "maskable_ppo", "arm": job["arm"], "with_regime": job["with_regime"],
                "seed": job["seed"], "steps": job["steps"], "hyperparameters": job["hyperparameters"],
                "spec": list(envs[0].spec_id), "kernel_fingerprint": envs[0].bridge_info["kernel_fingerprint"],
                "configs": configs, "training_digests": keys, "commit": job["commit"],
                "design_sha256": sha256(HERE / "design.json"), "train_seconds": round(seconds, 2),
                "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path)}
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def marker_path(root, design_sha256):
    return pathlib.Path(root) / f"holdout-{design_sha256}.started"


def refuse_if_holdout_started(root, design_sha256):
    if marker_path(root, design_sha256).exists():
        raise PermissionError("the holdout of this design has started: no further attempt")


def input_precision(days):
    """Per day group: the observation columns' ranges, and whether each distinct raw position and
    steps left maps to a distinct float32 input."""
    observation = np.concatenate([d["observation"] for d in days])
    def distinct(raw, column):
        pairs = set(zip(np.concatenate(raw).tolist(), observation[:, column].tolist(), strict=True))
        return len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})
    return {"column_min": observation.min(axis=0).tolist(),
            "column_max": observation.max(axis=0).tolist(),
            "position_one_to_one": distinct([d["raw_position"] for d in days], POSITION),
            "steps_left_one_to_one": distinct([d["raw_steps_left"] for d in days],
                                              m4_07.STEPS_LEFT)}


def agreement_by_regime(pairs):
    """`pairs`: (regime, hits array). The pooled agreement and the agreement per regime."""
    by = {}
    for regime, hits in pairs:
        agree, total = by.get(regime, (0, 0))
        by[regime] = (agree + int(np.sum(hits)), total + len(hits))
    total = sum(n for _, n in by.values())
    return (sum(a for a, _ in by.values()) / total,
            {r: a / n for r, (a, n) in sorted(by.items())})


def sanity_stops(agreements):
    return not all(a >= THRESHOLD for a in agreements)


def paired_interval(series_by_regime):
    """`series_by_regime`: regime to the ordered daily differences. Per regime, and pooled by
    resampling each regime's days independently and averaging the regimes' means."""
    regimes = sorted(series_by_regime)
    out, draws = {}, []
    for offset, regime in enumerate(regimes):
        series = np.asarray(series_by_regime[regime], dtype=float)
        index = est.stationary_bootstrap_indices(len(series), BLOCK, REPLICATES,
                                                 BOOTSTRAP_SEED + offset)
        means = series[index].mean(axis=1)
        draws.append(means)
        out[regime] = {"mean": float(series.mean()),
                       "low": float(np.quantile(means, 0.05)),
                       "high": float(np.quantile(means, 0.95)), "descriptive": True}
    pooled = np.mean(draws, axis=0)
    out["pooled"] = {"mean": float(np.mean([out[r]["mean"] for r in regimes])),
                     "low": float(np.quantile(pooled, 0.05)),
                     "high": float(np.quantile(pooled, 0.95)), "unadjusted": True}
    out["pooled"]["excludes_zero"] = out["pooled"]["low"] > 0 or out["pooled"]["high"] < 0
    return out


def daily_values(rows, arm):
    """Regime to day to the arm's daily value: the mean over its seeds."""
    collected = {}
    for r in rows:
        if r["arm"] == arm:
            collected.setdefault(r["regime"], {}).setdefault(r["day"], []).append(float(r["net"]))
    return {reg: {d: float(np.mean(v)) for d, v in days.items()} for reg, days in collected.items()}


class Study:
    def __init__(self, data_dir, run_dir, root, attempt, commit):
        torch.set_num_threads(1)
        self.design, self.m4_05, self.digests = load()
        self.data_dir, self.run_dir = data_dir, pathlib.Path(run_dir)
        self.root = pathlib.Path(root).resolve()
        self.out = self.root / f"attempt-{int(attempt)}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        if sha256(self.run_dir / "frozen.json") != self.design["reuses"]["m4_05_frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one this study reuses")
        self.frozen = json.loads((self.run_dir / "frozen.json").read_text())
        self.regimes = base.regimes_of(self.m4_05)
        grid_spec = self.m4_05["arms"]["maskable_ppo"]["grid"]
        self.main_config = base.base_config(self.m4_05, "main", "mask", grid_spec)
        self.grid = m4_07.grid_of(self.main_config)
        self.parallel = self.design["compute"]["max_parallel_runs"]
        self.design_sha256 = sha256(HERE / "design.json")

    def config(self, execution):
        return prior.config_of(self.main_config, execution, self.design)

    def code_hashes(self):
        out = {name: sha256(HERE / name) for name in CODE}
        out.update({f"{d}/{n}": sha256(EXPERIMENTS / d / n) for d, n in SHARED_CODE})
        return out

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "disclosure": self.design["disclosure"],
                "design_sha256": self.design_sha256, "code_sha256": self.code_hashes(),
                "versions": base.versions()}

    def days(self, split):
        first, last = self.design["splits"][split]
        return list(range(first, last + 1))

    def write(self, name, report):
        (self.out / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        return report

    def read(self, name):
        return json.loads((self.out / f"{name}.json").read_text())

    def job(self, regime, day, split, execution, arm, policy, seed=SEED, **extra):
        return {"regime": regime, "day": day, "split": split, "execution": execution,
                "data_dir": self.data_dir, "config": self.config(execution),
                "selected": self.frozen["classical_selected"], "arm": arm, "policy": policy,
                "seed": seed, "holdout_allowed": split == "holdout", **extra}

    def logs(self, dataset, split):
        return self.read(f"logs-{dataset}-{split}")["logs"]

    def epsilon(self, dataset):
        return prior.m4_07_epsilon("as_grid_eps10" if dataset.endswith("eps10") else "as_grid_eps30")

    def edges(self, dataset):
        return self.edges_of(self.logs(dataset, "train"))

    def edges_of(self, records):
        observations = np.concatenate([d["observation"] for d in prior.read_logs(records)])
        return np.quantile(observations[observations[:, 16] > 0, 5], [0.2, 0.4, 0.6, 0.8]).tolist()


def require(study, name, why):
    return prior.require(study, name, why)


def rule_policy(name):
    return {"kind": "rule", "name": name}


def ledger_add(path, label, process_seconds):
    """Adds a step's process seconds to the ledger and returns the total."""
    ledger = json.loads(path.read_text()) if path.exists() else {"steps": [], "total": 0.0}
    ledger["steps"].append({"step": label, "process_seconds": process_seconds})
    ledger["total"] += process_seconds
    path.write_text(json.dumps(ledger, indent=2))
    return ledger["total"]


def overran(elapsed, estimate):
    """The design's compute rule: more than 50 % over the estimate."""
    return elapsed > OVERRUN * estimate


def job_seconds(study, kind):
    """A job's estimated seconds, from the pilot's budget.json."""
    budget = study.read("budget")
    design = study.design
    if kind == "rule_day":
        return budget["rule_day_seconds"]
    if kind == "log_day":
        return budget["log_day_seconds"]
    if kind == "ppo":
        return budget["ppo_seconds_per_step"] * study.m4_05["arms"]["maskable_ppo"]["budget_steps"]
    if kind == "offline_day":
        return budget["offline_day_seconds"]
    return budget[kind]["seconds_per_gradient_step"] * design["arms"][kind]["settings"]["gradient_steps"]


def timed_call(worker, job):
    """A job's result and its own seconds."""
    started = time.perf_counter()
    result = worker(job)
    return result, time.perf_counter() - started


def rounds(jobs, workers):
    """Rounds of whole jobs on the workers."""
    return math.ceil(jobs / workers)


def run_pool(jobs, worker, parallel, limit, label):
    """The jobs in spawned processes, stopped at `limit` seconds or at a job's failure. Returns
    (rows or None, the finished count, elapsed, process seconds, the failure or None). At the limit the workers are terminated, and
    each job still running is charged the time since the start."""
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    executor = concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context)
    futures = [executor.submit(timed_call, worker, job) for job in jobs]
    pending = set(futures)

    def stopped(error):
        # `_processes` is private; the Python of the run is pinned by its environment.
        workers = list((executor._processes or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in workers:
            process.terminate()
        for process in workers:
            process.join()
        elapsed = time.perf_counter() - started
        finished = [f.result()[1] for f in futures if f.done() and not f.cancelled()
                    and f.exception() is None]
        charged = sum(finished) + min(len(workers), len(jobs) - len(finished)) * elapsed
        return None, len(finished), elapsed, charged, error

    try:
        while pending:
            timeout = None if limit is None else max(limit - (time.perf_counter() - started), 0.0)
            done, pending = concurrent.futures.wait(
                pending, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                if future.exception() is not None:
                    return stopped(future.exception())
            print(f"{label}: {len(jobs) - len(pending)}/{len(jobs)} after "
                  f"{round(time.perf_counter() - started)} s", file=sys.stderr, flush=True)
            if pending and not done:
                return stopped(None)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    outcomes = [future.result() for future in futures]
    return ([row for row, _ in outcomes], len(jobs), time.perf_counter() - started,
            sum(seconds for _, seconds in outcomes), None)


def step_limit(estimate, remaining_process_seconds, slots):
    """The smaller of 1.5 times the step's estimate and its share of the ceiling's remainder."""
    return min(OVERRUN * estimate, max(remaining_process_seconds, 0.0) / slots)


def may_continue(budget, has_go):
    """Whether a step may run: a pilot that did not stop, and a recorded compute go."""
    return not budget.get("stop") and has_go


def record_go(study, reference):
    budget = study.read("budget")
    if budget.get("stop"):
        raise PermissionError("the pilot stopped the study")
    return study.write("compute-go", {"reference": reference,
                                      "ceiling_process_hours": budget["ceiling_process_hours"],
                                      "budget_sha256": sha256(study.out / "budget.json")})


def job_count_ok(jobs, planned, enforce):
    """A step's jobs equal its planned count; the holdout may drop a failed secondary dataset."""
    return len(jobs) == planned if enforce else len(jobs) <= planned


def pool(jobs, worker, study, label, kind, planned, enforce=True):
    """The jobs in parallel under the step's limit. Only the holdout, read once, is not stopped:
    it records an excess instead."""
    if not job_count_ok(jobs, planned, enforce):
        raise ValueError(f"{label} has {len(jobs)} jobs, the ceiling planned {planned}")
    budget = study.read("budget")
    if not may_continue(budget, (study.out / "compute-go.json").exists()):
        raise PermissionError("no step runs after a stopped pilot or without the recorded compute go")
    ledger_path = study.out / "ledger.json"
    spent = json.loads(ledger_path.read_text())["total"] if ledger_path.exists() else 0.0
    slots = min(study.parallel, len(jobs))
    estimate = job_seconds(study, kind) * rounds(len(jobs), study.parallel)
    limit = step_limit(estimate, budget["ceiling_process_hours"] * 3600 - spent, slots)
    rows, finished, elapsed, process_seconds, failure = run_pool(jobs, worker, study.parallel, limit if enforce else None, label)
    total = ledger_add(ledger_path, label, process_seconds)
    name = label.replace(" ", "-")
    if failure is not None:
        raise failure
    if rows is None:
        study.write(f"overrun-{name}", {"step": label, "seconds": elapsed, "limit": limit,
                                        "finished_jobs": finished, "jobs": len(jobs)})
        raise RuntimeError(f"{label} reached its limit of {limit:.0f} s with {finished}/{len(jobs)} jobs")
    if elapsed > limit or total > budget["ceiling_process_hours"] * 3600:
        study.write(f"overrun-{name}", {"step": label, "seconds": elapsed, "limit": limit,
                                        "process_seconds_total": total})
    return rows


STEP_JOBS = {"gate": 63, "logs_train": 210, "logs_validation": 75, "train": 5, "ppo": 5,
             "select": 225, "holdout": 495}
PLANNED = {"rule_day": STEP_JOBS["gate"], "log_day": 2 * (STEP_JOBS["logs_train"] + STEP_JOBS["logs_validation"]),
           "bc": 2 * STEP_JOBS["train"], "cql": 4 * STEP_JOBS["train"], "ppo": 2 * STEP_JOBS["ppo"],
           "offline_day": STEP_JOBS["select"] + STEP_JOBS["holdout"]}


def ceiling_hours(seconds_by_kind, pilot_process_seconds=0.0):
    """The design's total: every planned job's estimated seconds and the pilot's, times 1.5, in
    hours."""
    return (sum(PLANNED[k] * seconds_by_kind[k] for k in PLANNED) + pilot_process_seconds) * OVERRUN / 3600


def pilot_job(job):
    """One timed unit of a kind; the concurrent workers are the load."""
    torch.set_num_threads(1)
    started = time.perf_counter()
    if job["kind"] == "train":
        train_job(job["train"])
    elif job["kind"] == "ppo":
        ppo_job(job["ppo"])
    elif job["kind"] == "log":
        log_job(job["log"])
    else:
        run_job(job["run"])
    return time.perf_counter() - started


def pilot(study):
    """Times under real load, on training days and attempt 1's training logs."""
    records = json.loads((study.root / "attempt-1" / f"logs-{PRIMARY}-train.json").read_text())["logs"]
    scale = reward_scale_of(records)
    workers = range(1, study.parallel + 1)
    day = 1
    k, m = len(study.grid.offsets), len(study.grid.sizes)
    out, trained = {}, None
    spent = [0.0]

    def timed(jobs):
        seconds = prior.pool(jobs, pilot_job, study.parallel, "pilot")
        spent[0] += float(np.sum(seconds))
        return float(np.mean(seconds))

    for arm in ("bc", "cql"):
        full = study.design["arms"][arm]["settings"]["gradient_steps"]
        settings = dict(study.design["arms"][arm]["settings"], gradient_steps=PILOT_GRADIENT_STEPS)
        if arm == "cql":
            settings["alpha"] = 0.1
        paths = [study.out / "pilot" / f"{arm}-{w}.pt" for w in workers]
        seconds = timed([{"kind": "train", "train": {
            "dataset": PRIMARY, "arm": arm, "seed": w, "settings": settings, "logs": records,
            "reward_scale": scale, "k": k, "m": m, "commit": study.commit, "path": str(path)}}
            for w, path in zip(workers, paths, strict=True)])
        out[arm] = {"seconds_per_gradient_step": seconds / PILOT_GRADIENT_STEPS,
                    "estimate_seconds": seconds / PILOT_GRADIENT_STEPS * full}
        if arm == "bc":
            trained = json.loads(paths[0].with_suffix(".json").read_text())
    edges = study.edges_of(records)
    out["log_day_seconds"] = timed([{"kind": "log", "log": study.job(
        study.regimes[w % 3], day, "train", "standard", "as_grid_eps30", None, seed=w, behaviour="as_grid_eps30",
        epsilon=0.3, path=str(study.out / "pilot" / f"log-{w}.npz"))} for w in workers])
    out["rule_day_seconds"] = timed([{"kind": "run", "run": study.job(
        study.regimes[w % 3], day, "train", "standard", "as_grid", rule_policy("as_grid"))} for w in workers])
    out["offline_day_seconds"] = timed([{"kind": "run", "run": study.job(
        study.regimes[w % 3], day, "train", "standard", "bc", {"kind": "offline", "metadata": trained},
        seed_of_arm=1, epsilon=0.3, edges=edges)} for w in workers])
    arm = study.m4_05["arms"]["maskable_ppo"]
    ppo = timed([{"kind": "ppo", "ppo": {
        "arm": "pilot", "with_regime": True, "seed": w, "steps": PILOT_PPO_STEPS,
        "hyperparameters": arm["hyperparameter_sets"][0], "config": study.config("standard"),
        "regimes": study.regimes, "days": study.days("train"), "data_dir": study.data_dir,
        "commit": study.commit, "path": str(study.out / "pilot" / f"ppo-{w}")}} for w in workers])
    out["ppo_seconds_per_step"] = ppo / PILOT_PPO_STEPS
    report = study.header("pilot")
    report.update(out)
    report["ceiling_process_hours"] = ceiling_hours({
        "rule_day": out["rule_day_seconds"], "log_day": out["log_day_seconds"],
        "offline_day": out["offline_day_seconds"], "bc": out["bc"]["estimate_seconds"],
        "cql": out["cql"]["estimate_seconds"],
        "ppo": out["ppo_seconds_per_step"] * arm["budget_steps"]}, spent[0])
    report["pilot_process_seconds"] = spent[0]
    maximum = study.design["compute"]["maximum_process_hours"]
    report["maximum_process_hours"] = maximum
    report["stop"] = report["ceiling_process_hours"] > maximum
    ledger_add(study.out / "ledger.json", "pilot", spent[0])
    return study.write("budget", report)


def gate(study):
    jobs = [study.job(regime, day, "validation", execution, name, rule_policy(name))
            for execution in ("standard", "immediate_reference")
            for name in ("as_grid", "fixed_spread_4")
            for regime in study.regimes for day in study.days("validation")]
    first = study.days("validation")[0]
    repeats = [study.job(regime, first, "validation", "standard", "fixed_spread_4",
                         rule_policy("fixed_spread_4")) for regime in study.regimes]
    rows = pool(jobs + repeats, run_job, study, "gate", "rule_day", STEP_JOBS["gate"])
    runs, again = rows[:len(jobs)], rows[len(jobs):]
    summary = prior.gate_summary(runs)
    reference = {(r["regime"], r["day"]): r["net"] for r in runs
                 if r["arm"] == "fixed_spread_4" and r["execution"] == "standard"}
    determinism = [{"regime": r["regime"], "day": r["day"], "first": reference[(r["regime"], r["day"])],
                    "again": r["net"]} for r in again]
    deterministic = all(d["first"] == d["again"] for d in determinism)
    report = study.header("gate")
    report.update({"rows": runs, "summary": summary, "determinism": determinism,
                   "deterministic": deterministic,
                   "stop": not (summary["as_grid"]["passes"] and deterministic)})
    return study.write("gate", report)


def run_logs(study, dataset, split):
    if dataset not in study.design["datasets"]:
        raise ValueError(f"no dataset {dataset}")
    if split not in ("train", "validation"):
        raise PermissionError("no log is generated on the holdout")
    require(study, "gate", "the logs wait for the queue gate")
    behaviour = "as_grid_eps10" if dataset.endswith("eps10") else "as_grid_eps30"
    seeds = json.loads((EXPERIMENTS / "m4_07" / "design.json").read_text())["seeds"]["behaviour_seeds"]
    jobs = [study.job(regime, day, split, "standard", behaviour, None, seed=seed,
                      behaviour=behaviour, epsilon=study.epsilon(dataset),
                      path=str(study.out / "logs" / dataset / split / regime / f"day-{day}" / f"seed{seed}.npz"))
            for regime in study.regimes for day in study.days(split) for seed in seeds]
    report = study.header(f"logs {dataset} {split}")
    report["logs"] = pool(jobs, log_job, study, f"logs {dataset} {split}", "log_day", STEP_JOBS[f"logs_{split}"])
    return study.write(f"logs-{dataset}-{split}", report)


def audit(study):
    """The aliasing diagnosis and the input's range and precision, on the primary training logs."""
    records = study.logs(PRIMARY, "train")
    days = prior.read_logs(records)
    observations = np.concatenate([d["observation"] for d in days])
    edges = np.quantile(observations[observations[:, 16] > 0, 5], [0.2, 0.4, 0.6, 0.8]).tolist()
    rows_per_day = study.m4_05["simulator"]["rows_per_day"]
    prepared = []
    for record, day in zip(records, days, strict=True):
        target = day["target_code"][list(day["target_names"]).index("as_grid")]
        prepared.append({"regime": record["regime"], "position": day["raw_position"],
                         "steps_left": np.asarray(day["raw_steps_left"], dtype=float),
                         "steps_fraction": np.asarray(day["raw_steps_left"], dtype=float) / rows_per_day,
                         "mask": day["mask"], "state_bin": est.bin_states(day["observation"], edges),
                         "target": target})
    report = study.header("audit")
    report.update(regime_of.aliasing(prepared))
    report["input_precision"] = input_precision(days)
    return study.write("audit", report)


def reward_scale_of(records):
    return prior.reward_scale(prior.read_logs(records))


def run_train(study, dataset, arm, alpha=None):
    require(study, "audit", "training waits for the audit")
    settings = dict(study.design["arms"][arm]["settings"])
    tag = arm
    if arm == "cql":
        require(study, f"sanity-{PRIMARY}", "CQL waits for BC's check on the primary dataset")
        if not (study.out / f"sanity-{SECONDARY}.json").exists():
            raise PermissionError("CQL waits for BC's check on the secondary dataset")
        if dataset == SECONDARY:
            require(study, f"sanity-{SECONDARY}", "the secondary dataset failed its check")
        if alpha is None or float(alpha) not in study.design["arms"]["cql"]["alpha_choices"]:
            raise ValueError("CQL needs an alpha from the design's choices")
        settings["alpha"] = float(alpha)
        tag = f"cql-alpha{float(alpha)}"
    elif dataset == SECONDARY:
        require(study, f"sanity-{PRIMARY}", "the secondary dataset waits for the primary's check")
    records = study.logs(dataset, "train")
    scale = reward_scale_of(records)
    jobs = [{"dataset": dataset, "arm": arm, "seed": seed, "settings": settings, "logs": records,
             "reward_scale": scale, "k": len(study.grid.offsets), "m": len(study.grid.sizes),
             "commit": study.commit,
             "path": str(study.out / "checkpoints" / dataset / f"{tag}-seed{seed}.pt")}
            for seed in study.design["arms"]["training_seeds"]]
    rows = pool(jobs, train_job, study, f"train {dataset} {tag}", arm, STEP_JOBS["train"])
    report = study.header(f"train {dataset} {tag}")
    report.update({"runs": rows, "reward_scale": scale,
                   "stop": not all(r["finished"] for r in rows)})
    return study.write(f"train-{dataset}-{tag}", report)


def sanity(study, dataset):
    if dataset == SECONDARY:
        require(study, f"sanity-{PRIMARY}", "the secondary check waits for the primary's")
    trained = study.read(f"train-{dataset}-bc")
    records = study.logs(dataset, "validation")
    days = days_with_regime(records)
    actions = offline.Actions(len(study.grid.offsets), len(study.grid.sizes))
    out = []
    for run in trained["runs"]:
        network = prior.network_of(run).eval()
        pairs = []
        for record, day in zip(records, days, strict=True):
            target = day["target_code"][list(day["target_names"]).index("as_grid")]
            joint = torch.as_tensor(actions.joint_mask(day["mask"]))
            with torch.no_grad():
                predicted = offline.masked(network(torch.as_tensor(day["observation"])), joint).argmax(1).numpy()
            pairs.append((record["regime"], predicted == target))
        pooled, by_regime = agreement_by_regime(pairs)
        out.append({"seed": run["seed"], "agreement": pooled, "by_regime": by_regime})
    report = study.header(f"sanity {dataset}")
    report.update({"runs": out, "threshold": THRESHOLD, "label": "[observed]",
                   "stop": sanity_stops([r["agreement"] for r in out])})
    return study.write(f"sanity-{dataset}", report)


def run_ppo(study, variant):
    require(study, "gate", "the PPO comparators wait for the queue gate")
    with_regime = {"regime": True, "blind": False}[variant]
    arm = study.m4_05["arms"]["maskable_ppo"]
    name = f"maskable_ppo_{variant}"
    jobs = [{"arm": name, "with_regime": with_regime, "seed": seed, "steps": arm["budget_steps"],
             "hyperparameters": arm["hyperparameter_sets"][0], "config": study.config("standard"),
             "regimes": study.regimes, "days": study.days("train"), "data_dir": study.data_dir,
             "commit": study.commit, "path": str(study.out / "checkpoints" / name / f"seed{seed}")}
            for seed in study.design["arms"]["training_seeds"]]
    report = study.header(name)
    report["runs"] = pool(jobs, ppo_job, study, name, "ppo", STEP_JOBS["ppo"])
    return study.write(name, report)


def select(study):
    require(study, f"sanity-{PRIMARY}", "the choice waits for the BC sanity rule")
    edges = study.edges(PRIMARY)
    jobs = []
    for alpha in study.design["arms"]["cql"]["alpha_choices"]:
        tag = f"cql-alpha{float(alpha)}"
        trained = study.read(f"train-{PRIMARY}-{tag}")
        if trained.get("stop"):
            continue
        for run in trained["runs"]:
            jobs.extend(study.job(regime, day, "validation", "standard", tag,
                                  {"kind": "offline", "metadata": run}, seed_of_arm=run["seed"],
                                  epsilon=study.epsilon(PRIMARY), edges=edges)
                        for regime in study.regimes for day in study.days("validation"))
    rows = pool(jobs, run_job, study, "select", "offline_day", STEP_JOBS["select"])
    for row in rows:
        row.pop("support_raw", None)
        row.pop("support_bins", None)
    means = prior.mean_by(rows, "arm")
    chosen = prior.choose_alpha(means, study.design["arms"]["cql"]["alpha_choices"])
    report = study.header("select")
    report.update({"rows": rows, "mean_daily_net": {k: str(v) for k, v in means.items()},
                   "alpha": chosen, "stop": chosen is None})
    return study.write("select", report)


def active_datasets(study):
    """The primary, and the secondary while its BC check passes."""
    if study.read(f"sanity-{SECONDARY}").get("stop"):
        return [PRIMARY]
    return [PRIMARY, SECONDARY]


def freeze(study):
    refuse_if_holdout_started(study.root, study.design_sha256)
    for name in ("gate", "audit", f"sanity-{PRIMARY}", "select"):
        require(study, name, "the freeze waits for every check")
    alpha = study.read("select")["alpha"]
    cql_tag = f"cql-alpha{float(alpha)}"
    datasets = active_datasets(study)
    tags = {dataset: ["bc", cql_tag] for dataset in datasets}
    checkpoints = {}
    for dataset, names in tags.items():
        for tag in names:
            report = study.read(f"train-{dataset}-{tag}")
            if report.get("stop"):
                raise PermissionError(f"the {dataset} {tag} training stopped")
            for run in report["runs"]:
                checkpoints[run["path"]] = sha256(run["path"])
    for name in ("maskable_ppo_regime", "maskable_ppo_blind"):
        for run in study.read(name)["runs"]:
            path = pathlib.Path(run["path"]).with_suffix(".zip")
            if sha256(path) != run["checkpoint_sha256"]:
                raise ValueError(f"{path} is not the PPO checkpoint its record names")
            checkpoints[str(path)] = run["checkpoint_sha256"]
    logs = {r["path"]: r["sha256"] for dataset in datasets for split in ("train", "validation")
            for r in study.logs(dataset, split)}
    reports = {p.name: sha256(p) for p in sorted(study.out.glob("*.json")) if p.name != "frozen.json"}
    report = study.header("freeze")
    report.update({"alpha": alpha, "tags": tags, "checkpoints": checkpoints, "logs": logs,
                   "reports": reports, "root": str(study.root),
                   "digests_sha256": sha256(HERE / "digests.json"),
                   "budget_sha256": sha256(study.out / "budget.json"),
                   "ledger_sha256": sha256(study.out / "ledger.json")})
    return study.write("frozen", report)


def check_frozen(study, frozen):
    if frozen["root"] != str(study.root):
        raise PermissionError("the holdout runs only under the attempts root the freeze records")
    if frozen["design_sha256"] != study.design_sha256:
        raise ValueError("design.json changed after the freeze")
    if frozen["digests_sha256"] != sha256(HERE / "digests.json"):
        raise ValueError("digests.json changed after the freeze")
    if frozen["budget_sha256"] != sha256(study.out / "budget.json"):
        raise ValueError("budget.json changed after the freeze")
    if frozen["code_sha256"] != study.code_hashes():
        raise ValueError("the code changed after the freeze")
    for path, digest in {**frozen["checkpoints"], **frozen["logs"]}.items():
        if sha256(path) != digest:
            raise ValueError(f"{path} changed after the freeze")
    for name, digest in frozen["reports"].items():
        if sha256(study.out / name) != digest:
            raise ValueError(f"{name} changed after the freeze")


def holdout_jobs(study, frozen):
    jobs, days = [], study.days("holdout")
    for name in ("no_trade", "fixed_spread_4", "as_grid"):
        jobs.extend(study.job(regime, day, "holdout", "standard", name, rule_policy(name))
                    for regime in study.regimes for day in days)
    for name, with_regime in (("maskable_ppo_regime", True), ("maskable_ppo_blind", False)):
        for run in study.read(name)["runs"]:
            jobs.extend(study.job(regime, day, "holdout", "standard", name,
                                  {"kind": "ppo", "metadata": run, "with_regime": with_regime},
                                  seed_of_arm=run["seed"])
                        for regime in study.regimes for day in days)
    for dataset, names in frozen["tags"].items():
        edges = study.edges(dataset)
        for tag in names:
            arm = "bc" if tag == "bc" else "cql"
            for run in study.read(f"train-{dataset}-{tag}")["runs"]:
                jobs.extend(study.job(regime, day, "holdout", "standard", f"{arm}:{dataset}",
                                      {"kind": "offline", "metadata": run}, seed_of_arm=run["seed"],
                                      dataset=dataset, epsilon=study.epsilon(dataset), edges=edges)
                            for regime in study.regimes for day in days)
    return jobs


def support_report(study, rows):
    out, keys_by_dataset = {}, {}
    scale = float(study.main_config["scales"]["position"])
    for row in rows:
        if "support_raw" not in row:
            continue
        dataset = row["dataset"]
        if dataset not in keys_by_dataset:
            edges, keys = study.edges(dataset), []
            records = study.logs(dataset, "train")
            for record, d in zip(records, prior.read_logs(records), strict=True):
                bins = support_bins(d["observation"], d["observation"][:, POSITION] * scale, edges,
                                    record["regime"])
                keys.extend(zip(bins.tolist(), d["action"].astype(int).tolist(), strict=True))
            keys_by_dataset[dataset] = keys
        counts = offline.support_counts(keys_by_dataset[dataset],
                                        list(zip(row["support_bins"], row["support_raw"]["code"], strict=True)))
        propensity = np.asarray(row["support_raw"]["propensity"])
        entry = out.setdefault(f"{row['arm']}-seed{row['seed_of_arm']}",
                               {"decisions": 0, "propensity_zero": 0, "propensity_under_1e-4": 0,
                                "pair_count_zero": 0, "pair_count_under_10": 0})
        entry["decisions"] += len(propensity)
        entry["propensity_zero"] += int((propensity == 0).sum())
        entry["propensity_under_1e-4"] += int((propensity < 1e-4).sum())
        entry["pair_count_zero"] += int((counts == 0).sum())
        entry["pair_count_under_10"] += int((counts < 10).sum())
    return out


def holdout(study, frozen_sha256):
    frozen_path = study.out / "frozen.json"
    if not frozen_path.exists() or sha256(frozen_path) != frozen_sha256:
        raise PermissionError("the holdout runs only against the frozen manifest named")
    refuse_if_holdout_started(study.root, study.design_sha256)
    frozen = json.loads(frozen_path.read_text())
    check_frozen(study, frozen)
    jobs = holdout_jobs(study, frozen)
    descriptor = os.open(marker_path(study.root, study.design_sha256),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as marker:
        marker.write(json.dumps({"commit": study.commit, "frozen_sha256": frozen_sha256,
                                 "attempt": study.out.name, "jobs": len(jobs), "at": time.time()}))
    rows = pool(jobs, run_job, study, "holdout", "offline_day", STEP_JOBS["holdout"], enforce=False)
    for job, row in zip(jobs, rows, strict=True):
        row["dataset"] = job.get("dataset")
    support = support_report(study, rows)
    for row in rows:
        row.pop("support_raw", None)
        row.pop("support_bins", None)
    report = study.header("holdout")
    report.update({"frozen_sha256": frozen_sha256, "rows": rows, "support": support})
    return study.write("holdout", report)


def report(study):
    rows = study.read("holdout")["rows"]
    tags = study.read("frozen")["tags"]
    out = {}
    for dataset in tags:
        names = {"cql": f"cql:{dataset}", "bc": f"bc:{dataset}"}
        for left, right in study.design["evaluation"]["comparisons"]:
            a, b = names.get(left, left), names.get(right, right)
            values_a, values_b = daily_values(rows, a), daily_values(rows, b)
            series = {regime: [values_a[regime][d] - values_b[regime][d]
                               for d in sorted(values_a[regime])] for regime in values_a}
            out[f"{a} - {b}"] = paired_interval(series)
    result = study.header("report")
    result.update({"comparisons": out, "label": "[simulated]",
                   "claims": study.design["evaluation"]["claims"]})
    return study.write("report", result)


def main():
    step, data_dir, run_dir, root, attempt, commit = sys.argv[1:7]
    args = sys.argv[7:]
    study = Study(data_dir, run_dir, root, attempt, commit)
    steps = {"pilot": lambda: pilot(study), "go": lambda: record_go(study, args[0]), "gate": lambda: gate(study),
             "logs": lambda: run_logs(study, args[0], args[1]), "audit": lambda: audit(study),
             "train": lambda: run_train(study, args[0], args[1], args[2] if len(args) > 2 else None),
             "sanity": lambda: sanity(study, args[0]), "ppo": lambda: run_ppo(study, args[0]),
             "select": lambda: select(study), "freeze": lambda: freeze(study),
             "holdout": lambda: holdout(study, args[0]), "report": lambda: report(study)}
    started = time.perf_counter()
    out = steps[step]()
    print(json.dumps({"step": step, "seconds": round(time.perf_counter() - started, 1),
                      "stop": out.get("stop")}))


if __name__ == "__main__":
    main()
