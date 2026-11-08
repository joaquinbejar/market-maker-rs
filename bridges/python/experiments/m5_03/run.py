"""Runs a step of #108's learned robustness study from its committed design.

From `bridges/python`, with #102's frozen runs:

    python experiments/m5_03/run.py <step> <m4_05 run dir> <data dir> <out dir> <commit> [args]

Steps:
- `days`: writes every scenario's design, simulates its days with `qr_days`, and records the
  digests in `experiments/m5_03/digests/`, to be committed before any policy runs;
- `rate`: the step rate of each arm on one day;
- `selection`: every arm on every grid point's selection days, and the adversary's choice;
- `freeze`: the hashes of the design, the code, the digests and the selection;
- `holdout <frozen sha256>`: every arm under the four conditions, once.

Everything here is synthetic: perturbed copies of #102's queue-reactive regimes.
"""

import concurrent.futures
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
BRIDGE = HERE.parent.parent
M4_05 = HERE.parent / "m4_05"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module("m4_05_run", M4_05 / "run.py")

from mm_rl_bridge import MaskableMarketMakingEnv, ResidualMarketMakingEnv  # noqa: E402
from mm_rl_bridge.training import ZeroResidual  # noqa: E402

SPLITS = ("selection", "holdout")
ARMS = ("no_trade", "fixed_spread", "avellaneda_stoikov", "glft", "maskable_ppo", "residual_sac")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    design = json.loads((HERE / "design.json").read_text())
    if sha256(M4_05 / "design.json") != design["reuses"]["design_sha256"]:
        raise ValueError("#102's design is not the one this study reuses")
    m4_05 = json.loads((M4_05 / "design.json").read_text())
    return design, m4_05


# The scenarios.


def points(design):
    """Every grid point, as a name and its multipliers, in a fixed order."""
    p = design["perturbations"]
    keys = ("market_rate_multiplier", "cancel_rate_multiplier", "theta_multiplier",
            "limit_rate_multiplier")
    out = []
    for values in itertools.product(*(p[k] for k in keys)):
        point = dict(zip(keys, values, strict=True))
        out.append((point_name(point), point))
    return out


def point_name(point):
    return "m{market_rate_multiplier}-c{cancel_rate_multiplier}-t{theta_multiplier}-l{limit_rate_multiplier}".format(
        **point)


def perturbed(m4_05, point, split, design):
    """#102's design with every regime's rates and theta multiplied, and this study's days."""
    out = copy.deepcopy(m4_05)
    for regime in out["simulator"]["regimes"]:
        regime["theta"] = regime["theta"] * point["theta_multiplier"]
        if not regime["theta"] < 1.0:
            raise ValueError(f"theta of {regime['name']} reaches 1 at {point}")
        for table in regime["rates"]:
            for row in table:
                row[0] = row[0] * point["limit_rate_multiplier"]
                row[1] = row[1] * point["cancel_rate_multiplier"]
                row[2] = row[2] * point["market_rate_multiplier"]
    days = design["days"][split]
    out["days"] = days["per_regime"]
    out["seed_offset"] = days["seed_offset"]
    out["splits"] = {"train": [1, days["per_regime"]],
                     "validation": [days["per_regime"] + 1, days["per_regime"] + 1],
                     "holdout": [days["per_regime"] + 2, days["per_regime"] + 2]}
    out["study"] = {"issue": 108, "point": point, "split": split}
    return out


def days_step(design, m4_05, data_dir):
    """Every scenario's days, and their digests, written once."""
    digests_dir = HERE / "digests"
    digests_dir.mkdir(exist_ok=True)
    scenarios = pathlib.Path(data_dir) / "scenarios"
    scenarios.mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        for name, point in points(design):
            scenario = f"{split}-{name}"
            design_path = scenarios / f"{scenario}.json"
            design_path.write_text(json.dumps(perturbed(m4_05, point, split, design), indent=2))
            out = pathlib.Path(data_dir) / scenario
            result = subprocess.run(
                ["cargo", "run", "-q", "--release", "--example", "qr_days", "--",
                 str(design_path), str(out)],
                cwd=BRIDGE, capture_output=True, text=True, check=True)
            record = json.loads(result.stdout)
            record["scenario"] = scenario
            record["point"] = point
            record["design_sha256"] = sha256(design_path)
            (digests_dir / f"{scenario}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            written.append(scenario)
            print(f"days: {scenario}", file=sys.stderr, flush=True)
    return {"scenarios": len(written)}


def digests_of(scenario):
    record = json.loads((HERE / "digests" / f"{scenario}.json").read_text())
    return {(d["regime"], d["day"]): d for d in record["days"]}


def rows_of(data_dir, scenario, regime, day):
    path = pathlib.Path(data_dir) / scenario / "train" / regime / f"day-{day}.json"
    return [tuple(row) for row in json.loads(path.read_text())]


# The arms.


def load_frozen(metadata, env, regime):
    """#102's loader across the action space version 1 to 2 change with hedges off, as #104:
    the specification may differ in the action version and its identity only, and the kernel
    fingerprint and the configuration are still checked."""
    spec, here = list(metadata["spec"]), list(env.spec_id)
    same = spec[0] == here[0] and spec[2:4] == here[2:4]
    if spec != here and not (same and spec[1] == 1 and here[1] == 2):
        raise ValueError("a checkpoint of another specification")
    if env.config.get("hedge_actions"):
        raise ValueError("a version 1 checkpoint runs only with hedge actions off")
    return base.load_frozen({**metadata, "spec": here}, env, regime)


def learned_runs(frozen, run_dir):
    sets = frozen["learned_selected_sets"]
    runs = {"maskable_ppo": [], "residual_sac": []}
    names = {"maskable_ppo": "maskable_ppo", "sac_residual_avellaneda_stoikov": "residual_sac"}
    for metadata in frozen["runs"]:
        if metadata.get("ablation") or f"set{metadata['set']}" != sets[metadata["arm"]]:
            continue
        moved = dict(metadata)
        moved["path"] = str(pathlib.Path(run_dir) / "checkpoints" / pathlib.Path(metadata["path"]).name)
        runs[names[metadata["arm"]]].append(moved)
    for arm, found in runs.items():
        found.sort(key=lambda m: m["seed"])
        if [m["seed"] for m in found] != [1, 2, 3, 4, 5]:
            raise ValueError(f"the frozen manifest does not hold {arm} seeds 1 to 5")
    return runs


def execution_variant(config, stress):
    if stress is None:
        return config
    config = json.loads(json.dumps(config))
    config["fill_model"] = stress
    return config


def run_arm(job):
    """One arm (or one seed of a learned arm) on one day of one scenario: the episode as #102
    measures it. Under the execution stress a learned policy is loaded and checked against
    #102's main configuration, then run under the stressed one."""
    torch.set_num_threads(1)
    design, m4_05 = load()
    frozen = json.loads((pathlib.Path(job["run_dir"]) / "frozen.json").read_text())
    selected = {tuple(k.split("/")): v for k, v in frozen["classical_selected"].items()}
    regime, day = job["regime"], job["day"]
    rows = rows_of(job["data_dir"], job["scenario"], regime, day)
    record = digests_of(job["scenario"])[(regime, day)]
    grid = m4_05["arms"]["maskable_ppo"]["grid"]
    stress = job.get("execution")
    arm = job["arm"]

    def maskable(config):
        return base.checked(MaskableMarketMakingEnv(config, rows), record)

    def residual(config):
        return base.checked(ResidualMarketMakingEnv(config, rows), record)

    main_mask = base.base_config(m4_05, "main", "mask", grid)
    bounds = m4_05["arms"]["sac_residual_avellaneda_stoikov"]["bounds"]
    if arm in ("no_trade", "fixed_spread", "maskable_ppo"):
        env = maskable(execution_variant(main_mask, stress))
    else:
        kind = "glft" if arm == "glft" else "avellaneda_stoikov"
        point = selected[(regime, kind)]
        config = base.residual_config(m4_05, "main", kind, point, day, bounds)
        env = residual(execution_variant(config, stress))
    try:
        if arm == "no_trade":
            policy = base.Constant([0, 0, 0, 0, 0])
        elif arm == "fixed_spread":
            index = grid["bid_offsets"].index(selected[(regime, "fixed_spread")]["offset_ticks"])
            policy = base.Constant([1, index, index, 0, 0])
        elif arm in ("avellaneda_stoikov", "glft"):
            policy = ZeroResidual()
        else:
            metadata = job["metadata"]
            if stress is None:
                policy = load_frozen(metadata, env, regime)
            else:
                checker = (maskable(main_mask) if arm == "maskable_ppo" else
                           residual(base.residual_config(m4_05, "main", "avellaneda_stoikov",
                                                         selected[(regime, "avellaneda_stoikov")],
                                                         day, bounds)))
                try:
                    policy = load_frozen(metadata, checker, regime)
                finally:
                    checker.close()
        result = base.episode(policy, env, design["evaluation_seed"], m4_05)
        # The day's largest absolute position and the price it was reached at, for the margin
        # measure; the first on a tie.
        peak, peak_price = Decimal(0), None
        for _, price, after in dict(env.attribution())["positions"]:
            if abs(Decimal(after)) > abs(peak):
                peak, peak_price = Decimal(after), price
    finally:
        env.close()
    attribution = result["attribution"]
    return {"scenario": job["scenario"], "condition": job.get("condition"), "arm": arm,
            "seed": job.get("metadata", {}).get("seed"), "regime": regime, "day": day,
            "net": result["net"], "max_drawdown": attribution.get("max_drawdown"),
            "max_abs_position": attribution.get("max_abs_position"),
            "mean_abs_position": attribution.get("mean_abs_position"),
            "peak_notional": result["peak_notional"], "refusals": result["refusals"],
            "residual_fallbacks": result["residual_fallbacks"],
            "residual_clipped": result["residual_clipped"],
            "limit_breach": result["limit_breach"],
            "peak_position": str(peak), "peak_position_price": peak_price}


def arm_jobs(design, m4_05, runs, scenario, days, common):
    jobs = []
    regimes = [r["name"] for r in m4_05["simulator"]["regimes"]]
    for regime in regimes:
        for day in days:
            for arm in ARMS:
                if arm in runs:
                    for metadata in runs[arm]:
                        jobs.append({**common, "scenario": scenario, "regime": regime,
                                     "day": day, "arm": arm, "metadata": metadata})
                else:
                    jobs.append({**common, "scenario": scenario, "regime": regime, "day": day,
                                 "arm": arm})
    return jobs


def job_key(job):
    seed = job.get("metadata", {}).get("seed")
    execution = json.dumps(job.get("execution"), sort_keys=True)
    return (job["scenario"], job.get("condition"), execution, job["arm"], seed, job["regime"],
            job["day"])


def pool(jobs, parallel, label, journal):
    """Runs the jobs, appending each result to `journal` as it ends, and skips the jobs a
    previous run of the same step already journaled. A result does not depend on the order or
    the number of processes, so a step can be stopped and resumed with fewer of them."""
    journal = pathlib.Path(journal)
    done_rows = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            done_rows[tuple(json.loads(entry["key"]))] = entry["row"]
    keys = [job_key(job) for job in jobs]
    todo = [(k, j) for k, j in zip(keys, jobs, strict=True) if k not in done_rows]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    print(f"{label}: {len(jobs) - len(todo)} of {len(jobs)} already done; {parallel} processes",
          file=sys.stderr, flush=True)
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as executor, \
            journal.open("a") as out:
        futures = {executor.submit(run_arm, job): key for key, job in todo}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            key = futures[future]
            done_rows[key] = row
            out.write(json.dumps({"key": json.dumps(list(key)), "row": row}) + "\n")
            out.flush()
            if done % 50 == 0 or done == len(todo):
                print(f"{label}: {done}/{len(todo)} after {round(time.perf_counter() - started)} s",
                      file=sys.stderr, flush=True)
    return [done_rows[key] for key in keys]


def daily(rows):
    """Per (arm, regime, day): the daily net, a learned arm's as the mean over its seeds."""
    out = {}
    for row in rows:
        out.setdefault((row["arm"], row["regime"], row["day"]), []).append(Decimal(row["net"]))
    return {k: sum(v, Decimal(0)) / len(v) for k, v in out.items()}


def cvar_loss(nets, share):
    """The mean loss of the worst ceil(share x n) days."""
    losses = sorted((-float(n) for n in nets), reverse=True)
    k = max(1, math.ceil(share * len(losses)))
    return sum(losses[:k]) / k


def choose(by_point, share=0.05):
    """The adversary's rule: per arm, the point with the largest CVaR of the daily loss, the
    first in grid order on a tie."""
    chosen = {}
    for arm in ARMS:
        best = None
        for name, nets in by_point.items():
            value = cvar_loss(nets[arm], share)
            if best is None or value > best[1]:
                best = (name, value)
        chosen[arm] = {"point": best[0], "cvar_loss": best[1]}
    return chosen


class Study:
    def __init__(self, run_dir, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.m4_05 = load()
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        frozen_path = pathlib.Path(run_dir) / "frozen.json"
        if sha256(frozen_path) != self.design["reuses"]["frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one this study reuses")
        self.frozen = json.loads(frozen_path.read_text())
        self.runs = learned_runs(self.frozen, run_dir)
        # The team's shared compute cap may lower the design's process count; it changes no
        # result.
        cap = self.design["compute"]["max_parallel_runs"]
        self.parallel = min(cap, int(os.environ.get("M5_03_PROCESSES", cap)))

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "code_sha256": sha256(HERE / "run.py"), "versions": base.versions()}

    def common(self):
        return {"run_dir": self.run_dir, "data_dir": self.data_dir}

    def write(self, name, report):
        (self.out / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def selection(study):
    days = list(range(1, study.design["days"]["selection"]["per_regime"] + 1))
    jobs = []
    for name, _ in points(study.design):
        jobs += arm_jobs(study.design, study.m4_05, study.runs, f"selection-{name}", days,
                         study.common())
    rows = pool(jobs, study.parallel, "selection", study.out / "selection.journal")
    by_point = {}
    for name, _ in points(study.design):
        scoped = daily([r for r in rows if r["scenario"] == f"selection-{name}"])
        by_point[name] = {arm: [v for (a, _, _), v in scoped.items() if a == arm] for arm in ARMS}
    report = study.header("selection")
    report["rows"] = rows
    report["chosen"] = choose(by_point)
    study.write("selection", report)
    return report


def freeze(study):
    selection_path = study.out / "selection.json"
    manifest = study.header("freeze")
    manifest["selection_sha256"] = sha256(selection_path)
    manifest["digests_sha256"] = digest_hashes(HERE / "digests")
    manifest["chosen"] = json.loads(selection_path.read_text())["chosen"]
    path = study.out / "frozen.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"frozen_sha256": sha256(path)}


def digest_hashes(digests_dir):
    return {p.name: sha256(p) for p in sorted(pathlib.Path(digests_dir).glob("*.json"))}


def check_frozen_digests(frozen, digests_dir):
    """Refuses digests that are not the frozen set: a file added, removed or changed since
    the freeze, so regenerated days cannot pass under the frozen manifest."""
    here = digest_hashes(digests_dir)
    frozen_set = frozen.get("digests_sha256")
    if not isinstance(frozen_set, dict) or not frozen_set:
        raise ValueError("the frozen manifest holds no digests")
    if set(here) != set(frozen_set):
        moved = sorted(set(here) ^ set(frozen_set))
        raise ValueError(f"the digest files changed after the freeze: {moved}")
    changed = sorted(name for name, digest in here.items() if frozen_set[name] != digest)
    if changed:
        raise ValueError(f"digests changed after the freeze: {changed}")


def holdout(study, frozen_sha256):
    path = study.out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    if (study.out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    frozen = json.loads(path.read_text())
    here = study.header("holdout")
    for key in ("design_sha256", "code_sha256"):
        if frozen[key] != here[key]:
            raise ValueError(f"{key} changed after the freeze")
    if frozen["selection_sha256"] != sha256(study.out / "selection.json"):
        raise ValueError("selection.json changed after the freeze")
    # Before any job runs or any journaled row is reused.
    check_frozen_digests(frozen, HERE / "digests")
    days = list(range(1, study.design["days"]["holdout"]["per_regime"] + 1))
    unperturbed = point_name({k: 1.0 for k in study.design["fixed_stress"]})
    stress = point_name(study.design["fixed_stress"])
    jobs = []
    for condition, scenario, execution in (
        ("unperturbed", unperturbed, None),
        ("fixed_stress", stress, None),
        ("execution_stress", unperturbed, study.design["execution_stress"]["fill_model"]),
    ):
        for job in arm_jobs(study.design, study.m4_05, study.runs, f"holdout-{scenario}", days,
                            study.common()):
            jobs.append({**job, "condition": condition, "execution": execution})
    for arm, choice in frozen["chosen"].items():
        for job in arm_jobs(study.design, study.m4_05, study.runs, f"holdout-{choice['point']}",
                            days, study.common()):
            if job["arm"] == arm:
                jobs.append({**job, "condition": "adversary"})
    rows = pool(jobs, study.parallel, "holdout", study.out / "holdout.journal")
    here["frozen_sha256"] = frozen_sha256
    here["rows"] = rows
    study.write("holdout", here)
    return here


def rate(study):
    """Steps per second of each arm on one selection day, before the full run."""
    out = {}
    scenario = f"selection-{points(study.design)[0][0]}"
    regime = study.m4_05["simulator"]["regimes"][0]["name"]
    for arm in ARMS:
        job = {**study.common(), "scenario": scenario, "regime": regime, "day": 1, "arm": arm}
        if arm in study.runs:
            job["metadata"] = study.runs[arm][0]
        started = time.perf_counter()
        run_arm(job)
        out[arm] = round(8640 / (time.perf_counter() - started), 1)
    report = study.header("rate")
    report["steps_per_second"] = out
    study.write("rate", report)
    return report


def main():
    step, run_dir, data_dir, out_dir, commit = sys.argv[1:6]
    started = time.perf_counter()
    if step == "days":
        design, m4_05 = load()
        report = days_step(design, m4_05, data_dir)
    else:
        study = Study(run_dir, data_dir, out_dir, commit)
        if step == "rate":
            report = rate(study)
        elif step == "selection":
            report = selection(study)
        elif step == "freeze":
            report = freeze(study)
        elif step == "holdout":
            report = holdout(study, sys.argv[6])
        else:
            raise SystemExit(f"unknown step {step!r}")
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps({"step": step, "seconds": round(time.perf_counter() - started, 1),
                      **summary}, indent=2, default=str))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
