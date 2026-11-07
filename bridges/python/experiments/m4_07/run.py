"""Runs a step of #104's offline evaluation support study from its committed design.

From `bridges/python`, with #102's rows and frozen runs:

    python experiments/m4_07/run.py <step> <data dir> <m4_05 run dir> <out dir> <commit> [args]

Steps:
- `rate`: the step rate on one training day, before the full run;
- `consistency`: interactive runs on validation days that must equal #102's;
- `logs <split>`: the behaviours' logs;
- `truth <split>`: each target's interactive run;
- `validation`: the estimator settings, the validation diagnostics and the stop rule;
- `freeze`: the hashes of the design, the code, the settings and every log;
- `holdout <frozen sha256>`: the estimates and the decision on the holdout logs, once.

Everything here is synthetic: #102's queue-reactive days, whose rows do not react to orders.
Only `logs holdout`, `truth holdout` and `holdout` read the holdout, and only after `freeze`.
"""

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
M4_05 = HERE.parent / "m4_05"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module("m4_05_run", M4_05 / "run.py")
est = _module("m4_07_estimate", HERE / "estimate.py")
rule = _module("m4_07_decide", HERE / "decide.py")

from mm_rl_bridge import MaskableMarketMakingEnv  # noqa: E402

YEAR_MS = 31_536_000_000
POSITION = 6
SPREAD = 0
MID_CHANGE = 1
IMBALANCE = 5
STEPS_LEFT = 10
HOLD = (0, 0, 0, 0, 0)
TRUTH_SEED = 101


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    design = json.loads((HERE / "design.json").read_text())
    reused = design["reuses"]
    if sha256(M4_05 / "design.json") != reused["design_sha256"]:
        raise ValueError("#102's design is not the one this study reuses")
    if sha256(M4_05 / "digests.json") != reused["digests_sha256"]:
        raise ValueError("#102's digests are not the ones this study reuses")
    m4_05, digests = base.load_design()
    return design, m4_05, digests


# The action space.


class Grid:
    """#102's maskable grid, and the layout of its mask: kind, bid, ask, bid size, ask size."""

    def __init__(self, grid):
        self.offsets = list(grid["bid_offsets"])
        if list(grid["ask_offsets"]) != self.offsets:
            raise ValueError("the study assumes the same offsets on both sides")
        self.sizes = list(grid["sizes"])
        k, m = len(self.offsets), len(self.sizes)
        self.slices = [slice(0, 2), slice(2, 2 + k), slice(2 + k, 2 + 2 * k),
                       slice(2 + 2 * k, 2 + 2 * k + m), slice(2 + 2 * k + m, 2 + 2 * k + 2 * m)]

    def canonical(self, action):
        """A joint action as a tuple; every hold is the same action."""
        action = tuple(int(v) for v in np.asarray(action).ravel())
        return HOLD if action[0] == 0 else action

    def code(self, action):
        """0 for a hold, and 1 + the quote's index otherwise."""
        kind, bid, ask, bid_size, ask_size = self.canonical(action)
        if kind == 0:
            return 0
        k, m = len(self.offsets), len(self.sizes)
        return 1 + ((bid * k + ask) * m + bid_size) * m + ask_size

    def allowed(self, mask):
        """Whether a quote is allowed, and the allowed indices of each quote dimension."""
        parts = [np.flatnonzero(mask[s]) for s in self.slices]
        return bool(mask[1]), parts[1:]

    def nearest(self, distance_ticks):
        """The index of the grid offset nearest a distance, the smaller on a tie, 0 below 0."""
        distance = max(distance_ticks, 0.0)
        return min(range(len(self.offsets)), key=lambda i: (abs(self.offsets[i] - distance), i))


class Behaviour:
    """A greedy rule, mixed epsilon-uniformly: with probability epsilon a hold with
    probability 1/2, otherwise a quote uniform over the quotes the mask allows."""

    def __init__(self, grid, greedy, epsilon, seed):
        self.grid = grid
        self.greedy = greedy
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def uniform(self, mask, action):
        quote_ok, dims = self.grid.allowed(mask)
        if not quote_ok:
            return 1.0 if action == HOLD else 0.0
        if action == HOLD:
            return 0.5
        count = math.prod(len(d) for d in dims)
        inside = all(index in d for index, d in zip(action[1:], dims, strict=True))
        return 0.5 / count if inside else 0.0

    def probability(self, mask, greedy, action):
        return (1.0 - self.epsilon) * (action == greedy) + self.epsilon * self.uniform(mask, action)

    def act(self, observation, raw, mask):
        greedy = self.grid.canonical(self.greedy(observation, raw, mask))
        if self.rng.random() >= self.epsilon:
            action = greedy
        else:
            quote_ok, dims = self.grid.allowed(mask)
            if not quote_ok or self.rng.random() < 0.5:
                action = HOLD
            else:
                action = (1, *(int(self.rng.choice(d)) for d in dims))
        return action, greedy, self.probability(mask, greedy, action)


class AsGrid:
    """Avellaneda-Stoikov at #102's selected point, projected on the grid (design `as_grid`)."""

    def __init__(self, grid, point, tick, sample_ms, max_steps):
        self.grid = grid
        self.gamma = float(point["risk_aversion"])
        self.k = float(point["order_intensity"])
        self.sigma = float(point["volatility"])
        self.horizon_ms = float(point["horizon_ms"])
        self.tick = float(tick)
        self.sample_ms = sample_ms
        self.max_steps = max_steps

    def distances(self, position, steps_left):
        elapsed = (self.max_steps - steps_left) * self.sample_ms
        years = max(self.horizon_ms - elapsed, 0.0) / YEAR_MS
        risk = self.gamma * self.sigma**2 * years
        half = (risk + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)) / 2.0
        skew = position * risk
        return (half + skew) / self.tick, (half - skew) / self.tick

    def __call__(self, observation, raw, mask):
        position = float(Decimal(raw[POSITION])) if raw[POSITION] is not None else 0.0
        steps_left = int(raw[STEPS_LEFT])
        bid_d, ask_d = self.distances(position, steps_left)
        action = (1, self.grid.nearest(bid_d), self.grid.nearest(ask_d), 0, 0)
        quote_ok, dims = self.grid.allowed(mask)
        if not quote_ok or not all(i in d for i, d in zip(action[1:], dims, strict=True)):
            return HOLD
        return action


def grid_of(config):
    return Grid({key: config[key] for key in ("bid_offsets", "ask_offsets", "sizes")})


def constant(action):
    return lambda observation, raw, mask: action


def frozen_ppo(model):
    def act(observation, raw, mask):
        action, _ = model.predict(observation, deterministic=True, action_masks=mask)
        return action

    return act


def load_frozen(metadata, env, regime):
    """#102's loader, with one difference stated in the design: the frozen runs were trained
    at action space version 1, before hedge actions (#247) made it version 2, and an
    environment with hedge actions off is unchanged by that. So the specification may differ in
    the action version 1 -> 2, and in the identity that hashes it, and in nothing else; the
    kernel fingerprint and the configuration must still be equal. The consistency step checks
    that the runs give #102's validation values exactly."""
    spec = list(metadata["spec"])
    here = list(env.spec_id)
    same = spec[0] == here[0] and spec[2:4] == here[2:4]
    if spec != here and not (same and spec[1] == 1 and here[1] == 2):
        raise ValueError("a checkpoint of another specification")
    if env.config.get("hedge_actions"):
        raise ValueError("a version 1 checkpoint runs only with hedge actions off")
    return base.load_frozen({**metadata, "spec": here}, env, regime)


def ppo_runs(frozen, run_dir):
    """#102's frozen main maskable PPO runs, their paths moved to the copy given."""
    runs = []
    for metadata in frozen["runs"]:
        if metadata["arm"] != "maskable_ppo" or metadata.get("ablation") or metadata["set"] != 0:
            continue
        moved = dict(metadata)
        moved["path"] = str(pathlib.Path(run_dir) / "checkpoints" / pathlib.Path(metadata["path"]).name)
        runs.append(moved)
    runs.sort(key=lambda m: m["seed"])
    if [m["seed"] for m in runs] != [1, 2, 3, 4, 5]:
        raise ValueError("the frozen manifest does not hold PPO set 0 seeds 1 to 5")
    return runs


class Study:
    """What every step shares."""

    def __init__(self, data_dir, run_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.m4_05, self.digests = load()
        self.data_dir = data_dir
        self.run_dir = pathlib.Path(run_dir)
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        frozen_path = self.run_dir / "frozen.json"
        if sha256(frozen_path) != self.design["reuses"]["frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one this study reuses")
        self.frozen = json.loads(frozen_path.read_text())
        self.selected = {tuple(k.split("/")): v for k, v in self.frozen["classical_selected"].items()}
        self.regimes = base.regimes_of(self.m4_05)
        self.grid_spec = self.m4_05["arms"]["maskable_ppo"]["grid"]
        self.config = base.base_config(self.m4_05, "main", "mask", self.grid_spec)
        self.ppo = ppo_runs(self.frozen, run_dir)

    def header(self, step):
        return {
            "step": step,
            "commit": self.commit,
            "data": self.design["data"],
            "bounds": self.design["bounds"],
            "disclosure": self.design["disclosure"],
            "design_sha256": sha256(HERE / "design.json"),
            "code_sha256": {name: sha256(HERE / name)
                            for name in ("run.py", "estimate.py", "decide.py")},
            "versions": base.versions(),
        }

    def days(self, split):
        return base.days_of(self.m4_05, split)

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path


def make_env(job, regime, day, split):
    design, m4_05, digests = load()
    rows, record = base.read_rows(job["data_dir"], digests, regime, day, split,
                                  holdout_allowed=job.get("holdout_allowed", False))
    return base.checked(MaskableMarketMakingEnv(job["config"], rows), record), m4_05


def targets_for(job, regime, env, grid, m4_05):
    """Every target of the design, as callables on (observation, raw, mask)."""
    offsets = grid.offsets
    point = job["selected"][f"{regime}/avellaneda_stoikov"]
    fixed = job["selected"][f"{regime}/fixed_spread"]
    index = offsets.index(fixed["offset_ticks"])
    sizes = grid.sizes.index(int(fixed["size_lots"]))
    as_grid = AsGrid(grid, point, m4_05["kernel"]["tick_size"],
                     m4_05["simulator"]["sample_ms"], m4_05["simulator"]["rows_per_day"])
    targets = {"no_trade": constant(HOLD), "fixed_spread_4": constant((1, index, index, sizes, sizes)),
               "as_grid": as_grid}
    for metadata in job["ppo"]:
        model = load_frozen(metadata, env, regime)
        targets[f"maskable_ppo-seed{metadata['seed']}"] = frozen_ppo(model)
    return targets


def log_job(job):
    """One behaviour, one seed, one day: the log, with every target's action at each state."""
    torch.set_num_threads(1)
    regime, day, split = job["regime"], job["day"], job["split"]
    env, m4_05 = make_env(job, regime, day, split)
    grid = grid_of(job["config"])
    try:
        targets = targets_for(job, regime, env, grid, m4_05)
        behaviour = Behaviour(grid, targets["as_grid"], job["epsilon"], job["seed"])
        observation, info = env.reset(seed=job["seed"])
        rows = {"observation": [], "mask": [], "action": [], "propensity": [], "reward": [],
                "refused": []}
        target_code = {name: [] for name in targets}
        target_mu = {name: [] for name in targets}
        while True:
            mask = env.action_masks()
            raw = info["raw_observation"]
            action, greedy, propensity = behaviour.act(observation, raw, mask)
            for name, target in targets.items():
                chosen = grid.canonical(target(observation, raw, mask))
                target_code[name].append(grid.code(chosen))
                target_mu[name].append(behaviour.probability(mask, greedy, chosen))
            rows["observation"].append(observation)
            rows["mask"].append(mask)
            rows["action"].append(grid.code(action))
            rows["propensity"].append(propensity)
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            rows["reward"].append(float(Decimal(info["reward_exact"])))
            rows["refused"].append(info["refused_because"] is not None)
            if terminated or truncated:
                break
        path = pathlib.Path(job["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observation=np.asarray(rows["observation"], dtype=np.float32),
            mask=np.asarray(rows["mask"], dtype=bool),
            action=np.asarray(rows["action"], dtype=np.int16),
            propensity=np.asarray(rows["propensity"]),
            reward=np.asarray(rows["reward"]),
            refused=np.asarray(rows["refused"], dtype=bool),
            target_names=np.asarray(list(targets)),
            target_code=np.asarray([target_code[n] for n in targets], dtype=np.int16),
            target_mu=np.asarray([target_mu[n] for n in targets]),
        )
        return {"regime": regime, "day": day, "split": split, "behaviour": job["behaviour"],
                "seed": job["seed"], "steps": len(rows["action"]), "net": info["net"],
                "path": str(path), "sha256": sha256(path)}
    finally:
        env.close()


def truth_job(job):
    """One target on one day, interactively at the truth seed: its exact net."""
    torch.set_num_threads(1)
    regime, day, split = job["regime"], job["day"], job["split"]
    env, m4_05 = make_env(job, regime, day, split)
    grid = grid_of(job["config"])
    try:
        target = targets_for(job, regime, env, grid, m4_05)[job["target"]]
        observation, info = env.reset(seed=TRUTH_SEED)
        steps = 0
        while True:
            action = grid.canonical(target(observation, info["raw_observation"], env.action_masks()))
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            steps += 1
            if terminated or truncated:
                break
        attribution = dict(env.attribution())
        return {"regime": regime, "day": day, "split": split, "target": job["target"],
                "steps": steps, "net": info["net"], "fills": attribution.get("fills"),
                "label": "[simulated]"}
    finally:
        env.close()


def pool(jobs, worker, parallel, label):
    return base.pool(jobs, worker, parallel, label)


def common_job(study, regime, day, split):
    return {"regime": regime, "day": day, "split": split, "data_dir": study.data_dir,
            "config": study.config, "selected": study.frozen["classical_selected"],
            "ppo": study.ppo, "holdout_allowed": split == "holdout"}


def target_names(study):
    return ["no_trade", "fixed_spread_4", "as_grid"] + [
        f"maskable_ppo-seed{m['seed']}" for m in study.ppo]


def run_logs(study, split):
    if split == "holdout" and not (study.out / "frozen.json").exists():
        raise PermissionError("the holdout logs wait for the freeze")
    jobs = []
    for name, spec in study.design["behaviours"].items():
        if "epsilon" not in spec:
            continue
        for regime in study.regimes:
            for day in study.days(split):
                for seed in study.design["seeds"]["behaviour_seeds"]:
                    path = study.out / "logs" / split / regime / f"day-{day}" / f"{name}-seed{seed}.npz"
                    jobs.append({**common_job(study, regime, day, split), "behaviour": name,
                                 "epsilon": spec["epsilon"], "seed": seed, "path": str(path)})
    parallel = study.design["compute"]["max_parallel_runs"]
    rows = pool(jobs, log_job, parallel, f"logs {split}")
    report = study.header(f"logs {split}")
    report["logs"] = rows
    study.write(f"logs-{split}", report)
    return report


def run_truth(study, split):
    if split == "holdout" and not (study.out / "frozen.json").exists():
        raise PermissionError("the holdout truth waits for the freeze")
    jobs = [{**common_job(study, regime, day, split), "target": target}
            for target in target_names(study) for regime in study.regimes
            for day in study.days(split)]
    rows = pool(jobs, truth_job, study.design["compute"]["max_parallel_runs"], f"truth {split}")
    report = study.header(f"truth {split}")
    report["rows"] = rows
    study.write(f"truth-{split}", report)
    return report


def consistency(study):
    """The design's first stop rule: this study's interactive runs of the fixed spread and of
    each PPO seed on validation days equal #102's validation values, day by day."""
    selection = json.loads((study.run_dir / "selection.json").read_text())
    expected = {}
    for row in selection["classical_validation"]:
        if row["arm"] == "fixed_spread" and json.loads(row["candidate"])["offset_ticks"] == 4:
            expected[("fixed_spread_4", row["regime"], row["day"])] = Decimal(row["daily_value"])
    for row in selection["learned_validation"]:
        if row["arm"] == "maskable_ppo" and row["candidate"].startswith("set0-"):
            seed = row["candidate"].split("seed")[1]
            expected[(f"maskable_ppo-seed{seed}", row["regime"], row["day"])] = Decimal(
                row["daily_value"])
    jobs = [{**common_job(study, regime, day, "validation"), "target": target}
            for target, regime, day in sorted(expected)]
    rows = pool(jobs, truth_job, study.design["compute"]["max_parallel_runs"], "consistency")
    mismatches = [
        {"target": r["target"], "regime": r["regime"], "day": r["day"], "here": r["net"],
         "m4_05": str(expected[(r["target"], r["regime"], r["day"])])}
        for r in rows if Decimal(r["net"]) != expected[(r["target"], r["regime"], r["day"])]
    ]
    report = study.header("consistency")
    report.update({"compared": len(rows), "mismatches": mismatches, "rows": rows,
                   "stop": bool(mismatches)})
    study.write("consistency", report)
    return report


def rate(study):
    """Steps per second of one log and one truth run on training day 1 of the first regime."""
    regime = study.regimes[0]
    out = {}
    started = time.perf_counter()
    path = study.out / "rate" / "log.npz"
    row = log_job({**common_job(study, regime, 1, "train"), "behaviour": "as_grid_eps30",
                   "epsilon": 0.3, "seed": 1, "path": str(path)})
    out["log_steps_per_second"] = round(row["steps"] / (time.perf_counter() - started), 1)
    started = time.perf_counter()
    row = truth_job({**common_job(study, regime, 1, "train"), "target": "maskable_ppo-seed1"})
    out["ppo_truth_steps_per_second"] = round(row["steps"] / (time.perf_counter() - started), 1)
    started = time.perf_counter()
    row = truth_job({**common_job(study, regime, 1, "train"), "target": "as_grid"})
    out["rule_truth_steps_per_second"] = round(row["steps"] / (time.perf_counter() - started), 1)
    report = study.header("rate")
    report.update(out)
    study.write("rate", report)
    return report


def main():
    step, data_dir, run_dir, out_dir, commit = sys.argv[1:6]
    study = Study(data_dir, run_dir, out_dir, commit)
    started = time.perf_counter()
    if step == "rate":
        report = rate(study)
    elif step == "consistency":
        report = consistency(study)
    elif step == "logs":
        report = run_logs(study, sys.argv[6])
    elif step == "truth":
        report = run_truth(study, sys.argv[6])
    else:
        raise SystemExit(f"step {step!r} is not implemented yet")
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps({"step": step, "total_seconds": round(time.perf_counter() - started, 1),
                      **{k: v for k, v in summary.items() if k not in ("data", "bounds")}},
                     indent=2, default=str))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
