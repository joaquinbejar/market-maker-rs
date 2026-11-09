"""#282: #108's robustness study with the agent's own price impact, in rows that react to its
fills. The design is design.json, fixed before any run.

    python experiments/m5_04/run.py rate <#102 run dir> <out dir> <commit>
    python experiments/m5_04/run.py check <#102 run dir> <out dir> <commit>
    python experiments/m5_04/run.py selection <#102 run dir> <out dir> <commit> <#108 selection.json>
    python experiments/m5_04/run.py freeze <#102 run dir> <out dir> <commit>
    python experiments/m5_04/run.py holdout <#102 run dir> <out dir> <commit> <frozen sha256>

`rate` times one #102 training day (calm, day 1: neither a selection nor a holdout day) per arm
at kappa 0 and at 2 kappa0, with the reactions it makes, before any slot is asked for. `check`
runs that day at 2 kappa0 with the worker's replay and with the Python one, per arm, and
requires the same reactions, moves, net, attribution and positions.

Synthetic data only.
"""

import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
M5_03 = HERE.parent / "m5_03"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m503 = _module("m5_03_run", M5_03 / "run.py")
base = m503.base

from mm_rl_bridge import (  # noqa: E402
    MaskableMarketMakingEnv,
    ReactiveQrMarket,
    ResidualMarketMakingEnv,
)
from mm_rl_bridge.training import ZeroResidual  # noqa: E402

MASK64 = (1 << 64) - 1
DAY_MS = 86_400_000
SEED_STRIDE = 1_000_003


# The reaction, as design.json states it.


def fnv1a64(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & MASK64
    return value


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def unit(self):
        return (self.next_u64() >> 11) / float(1 << 53)


def impact_rng(day_seed):
    return SplitMix64(fnv1a64(day_seed.to_bytes(8, "little") + b"impact-282"))


def moves_for(kappa, filled, rng):
    """Reference moves for one step's fills: netted, against the agent's net position change,
    kappa * |n| ticks with the fraction drawn; one draw per step with n != 0."""
    n = filled["sell"] - filled["buy"]
    if n == 0:
        return 0
    total = kappa * abs(n)
    whole = math.floor(total)
    if rng.unit() < total - whole:
        whole += 1
    return whole if n > 0 else -whole


def day_market(regime, m4_05, seed, day):
    sim = m4_05["simulator"]
    episode = {"seed": seed, "reference_ticks": sim["reference_ticks"],
               "start_ms": (day - 1) * DAY_MS, "sample_ms": sim["sample_ms"],
               "rows": sim["rows_per_day"]}
    return ReactiveQrMarket(json.dumps(regime), json.dumps(m4_05["kernel"]), json.dumps(episode))


def predict(policy, env, observation):
    if isinstance(env, MaskableMarketMakingEnv):
        action, _ = policy.predict(observation, deterministic=True,
                                   action_masks=env.action_masks())
    else:
        action, _ = policy.predict(observation, deterministic=True)
    return action


class ReplayMismatch(RuntimeError):
    """A replay that does not reach the pre-reaction step: the run stops."""


def reactive_episode(policy, make_env, market, kappa, rng, seed, m4_05, replay="native"):
    """One evaluated day in rows that react to the agent's fills, with base.episode's result
    and the reactions. After a reaction the episode is replayed on the new rows without
    querying the policy, and must reach the same step: by default in the bridge's worker
    (replace_rows, which checks the whole state, the observation and the masks); with
    replay="python", by rebuilding the environment and stepping the logged actions here,
    checking the observation, net and penalties."""
    env = make_env(market.rows())
    log = []
    observation, _ = env.reset(seed=seed)
    refusals = fallbacks = clipped = 0
    reactions = moves = replayed = 0
    emptied = Decimal(0)
    try:
        while True:
            action = predict(policy, env, observation)
            observation, _, terminated, truncated, info = env.step(action)
            log.append(action)
            refusals += info["refused_because"] is not None
            residual = info.get("residual")
            if residual is not None:
                fallbacks += residual["fallback"] is not None
                clipped += any(residual["clips"])
            done = terminated or truncated
            if kappa > 0 and not done:
                step_moves = moves_for(kappa, info["filled_lots"], rng)
                if step_moves:
                    emptied += Decimal(market.react(info["row"], step_moves))
                    if replay == "native":
                        try:
                            env.replace_rows(market.rows())
                        except ValueError as error:
                            raise ReplayMismatch(f"step {len(log)}: {error}") from error
                    else:
                        env.close()
                        env = make_env(market.rows())
                        again, _ = env.reset(seed=seed)
                        for logged in log:
                            again, _, _, _, check = env.step(logged)
                        if not (np.array_equal(again, observation)
                                and check["net"] == info["net"]
                                and check["penalties"] == info["penalties"]):
                            raise ReplayMismatch(f"step {len(log)} after a reaction")
                        observation = again
                    reactions += 1
                    moves += abs(step_moves)
                    replayed += len(log)
            if done:
                break
        attribution = dict(env.attribution())
    finally:
        env.close()
    limits = m4_05["execution"]["limits"]
    peak_notional = max(
        (abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
        default=Decimal(0),
    )
    return {"steps": len(log), "net": info["net"], "penalties": info["penalties"],
            "attribution": {k: v for k, v in attribution.items() if k != "positions"},
            "positions": attribution["positions"], "refusals": refusals,
            "residual_fallbacks": fallbacks, "residual_clipped": clipped,
            "peak_notional": str(peak_notional), "reactions": reactions, "moves": moves,
            "emptied": str(emptied),
            "replayed_steps": replayed, "stopped": market.stopped(),
            "limit_breach": Decimal(attribution["max_abs_position"])
            > Decimal(limits["max_position"]) or peak_notional > Decimal(limits["max_notional"])}


# The arms, as #108 runs them.


def arm_policy(arm, regime, day, rows, frozen, runs, m4_05, stress=None, metadata=None):
    """The arm's policy and the environment factory over given rows, as #108's run_arm: under
    the execution stress a learned policy is loaded against #102's main configuration and run
    under the stressed one."""
    selected = {tuple(k.split("/")): v for k, v in frozen["classical_selected"].items()}
    grid = m4_05["arms"]["maskable_ppo"]["grid"]
    main_mask = base.base_config(m4_05, "main", "mask", grid)
    bounds = m4_05["arms"]["sac_residual_avellaneda_stoikov"]["bounds"]
    if arm in ("no_trade", "fixed_spread", "maskable_ppo"):
        main = main_mask

        def factory(config):
            return lambda new_rows: MaskableMarketMakingEnv(config, [tuple(r) for r in new_rows])
    else:
        kind = "glft" if arm == "glft" else "avellaneda_stoikov"
        main = base.residual_config(m4_05, "main", kind, selected[(regime, kind)], day, bounds)

        def factory(config):
            return lambda new_rows: ResidualMarketMakingEnv(config, [tuple(r) for r in new_rows])
    make_env = factory(m503.execution_variant(main, stress))
    if arm == "no_trade":
        policy = base.Constant([0, 0, 0, 0, 0])
    elif arm == "fixed_spread":
        index = grid["bid_offsets"].index(selected[(regime, "fixed_spread")]["offset_ticks"])
        policy = base.Constant([1, index, index, 0, 0])
    elif arm in ("avellaneda_stoikov", "glft"):
        policy = ZeroResidual()
    else:
        if arm == "maskable_ppo":
            checker_config = main
        else:
            checker_config = base.residual_config(m4_05, "main", "avellaneda_stoikov",
                                                  selected[(regime, "avellaneda_stoikov")], day,
                                                  bounds)
        probe = factory(checker_config)(rows)
        try:
            policy = m503.load_frozen(metadata or runs[arm][0], probe, regime)
        finally:
            probe.close()
    return policy, make_env


def rate(run_dir, out_dir, commit):
    torch.set_num_threads(1)
    design = json.loads((HERE / "design.json").read_text())
    m4_05 = json.loads((HERE.parent / "m4_05" / "design.json").read_text())
    frozen = json.loads((pathlib.Path(run_dir) / "frozen.json").read_text())
    runs = m503.learned_runs(frozen, run_dir)
    regime = m4_05["simulator"]["regimes"][0]
    day = 1
    seed = SEED_STRIDE * 1 + day
    kappa0 = design["impact_scale"]["values"][regime["name"]]["kappa0"]
    out = {}
    for arm in m503.ARMS:
        rows = day_market(regime, m4_05, seed, day).rows()
        policy, make_env = arm_policy(arm, regime["name"], day, rows, frozen, runs, m4_05)
        for label, kappa in (("kappa 0", 0.0), ("kappa 2 kappa0", 2 * kappa0)):
            market = day_market(regime, m4_05, seed, day)
            started = time.perf_counter()
            result = reactive_episode(policy, make_env, market, kappa, impact_rng(seed),
                                      design["evaluation_seed"], m4_05)
            seconds = time.perf_counter() - started
            out.setdefault(arm, {})[label] = {
                "seconds": round(seconds, 2), "steps": result["steps"],
                "reactions": result["reactions"], "moves": result["moves"],
                "replayed_steps": result["replayed_steps"], "net": result["net"]}
            print(arm, label, out[arm][label], file=sys.stderr, flush=True)
    report = {"step": "rate", "commit": commit, "regime": regime["name"], "day": day,
              "seed": seed, "kappa0": kappa0, "arms": out, "versions": base.versions()}
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / "rate.json").write_text(json.dumps(report, indent=2))
    return report


def check(run_dir, out_dir, commit):
    torch.set_num_threads(1)
    design = json.loads((HERE / "design.json").read_text())
    m4_05 = json.loads((HERE.parent / "m4_05" / "design.json").read_text())
    frozen = json.loads((pathlib.Path(run_dir) / "frozen.json").read_text())
    runs = m503.learned_runs(frozen, run_dir)
    regime = m4_05["simulator"]["regimes"][0]
    day, seed = 1, SEED_STRIDE + 1
    kappa = 2 * design["impact_scale"]["values"][regime["name"]]["kappa0"]
    out = {}
    for arm in m503.ARMS:
        rows = day_market(regime, m4_05, seed, day).rows()
        policy, make_env = arm_policy(arm, regime["name"], day, rows, frozen, runs, m4_05)
        results = {}
        for path in ("native", "python"):
            started = time.perf_counter()
            result = reactive_episode(policy, make_env, day_market(regime, m4_05, seed, day),
                                      kappa, impact_rng(seed), design["evaluation_seed"], m4_05,
                                      replay=path)
            results[path] = (round(time.perf_counter() - started, 2), result)
        (native_s, native), (python_s, python) = results["native"], results["python"]
        same = all(native[k] == python[k] for k in
                   ("reactions", "moves", "net", "attribution", "positions", "steps"))
        out[arm] = {"same": same, "reactions": native["reactions"], "moves": native["moves"],
                    "net": native["net"], "seconds_native": native_s,
                    "seconds_python": python_s}
        print(arm, out[arm], file=sys.stderr, flush=True)
        if not same:
            raise SystemExit(f"{arm}: the replays differ")
    report = {"step": "check", "commit": commit, "regime": regime["name"], "day": day,
              "seed": seed, "kappa": kappa, "arms": out, "versions": base.versions()}
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / "check.json").write_text(json.dumps(report, indent=2))
    return report


# The study.

M5_03_DESIGN = json.loads((M5_03 / "design.json").read_text())
# Design version 4: kappa0 (index 1) left the grid after the overrun stop; every holdout impact
# condition is at 2 kappa0. The k1 rows already journaled are never used.
KAPPA_INDICES = (0, 2)
CEILING_PROCESS_HOURS = 97.5
HOLDOUT_RESERVE_PROCESS_HOURS = 9.0


class Overrun(RuntimeError):
    """The study's process-hour ceiling was reached: the step stops, and is reported."""


def spent(out):
    """Process-seconds spent so far, from budget.json in the output directory."""
    path = pathlib.Path(out) / "budget.json"
    return json.loads(path.read_text())["spent_process_seconds"] if path.exists() else None


def charge(out, label, seconds, parallel):
    path = pathlib.Path(out) / "budget.json"
    budget = json.loads(path.read_text())
    budget["spent_process_seconds"] += seconds * parallel
    budget.setdefault("log", []).append({"step": label, "wall_seconds": round(seconds, 1),
                                         "processes": parallel})
    path.write_text(json.dumps(budget, indent=2) + "\n")


def limit_for(out, parallel, reserve_hours):
    """Wall seconds left before the ceiling, keeping `reserve_hours` for later steps."""
    used = spent(out)
    if used is None:
        raise ValueError("no budget.json: the spent process-seconds must be recorded first")
    left = (CEILING_PROCESS_HOURS - reserve_hours) * 3600 - used
    if left <= 0:
        raise Overrun(f"the ceiling of {CEILING_PROCESS_HOURS} process-hours is reached")
    return left / parallel
FIELDS_108 = ("net", "max_drawdown", "max_abs_position", "mean_abs_position", "peak_notional",
              "refusals", "residual_fallbacks", "residual_clipped", "limit_breach",
              "peak_position", "peak_position_price")


def sha256(path):
    return m503.sha256(path)


def load():
    design = json.loads((HERE / "design.json").read_text())
    if sha256(M5_03 / "design.json") != design["reuses"]["design_sha256"]:
        raise ValueError("#108's design is not the one this study reuses")
    m4_05 = json.loads((HERE.parent / "m4_05" / "design.json").read_text())
    return design, m4_05


def grid_points():
    """Every point, as a name, its flow point's name and multipliers, and the kappa index."""
    return [(f"{name}-k{k}", name, point, k) for name, point in m503.points(M5_03_DESIGN)
            for k in KAPPA_INDICES]


def unit_flow():
    return m503.point_name({k: 1.0 for k in M5_03_DESIGN["fixed_stress"]})


def regime_day(m4_05, flow_point, split, regime_name, day):
    """The perturbed regime, its index and the day's seed, as #108's days."""
    perturbed = m503.perturbed(m4_05, flow_point, split, M5_03_DESIGN)
    names = [r["name"] for r in perturbed["simulator"]["regimes"]]
    index = names.index(regime_name)
    seed = SEED_STRIDE * (index + 1) + day + perturbed["seed_offset"]
    return perturbed["simulator"]["regimes"][index], seed


def run_job(job):
    """One arm (or one learned seed) on one day at one point: the reacting episode."""
    torch.set_num_threads(1)
    design, m4_05 = load()
    frozen = json.loads((pathlib.Path(job["run_dir"]) / "frozen.json").read_text())
    runs = m503.learned_runs(frozen, job["run_dir"])
    flows = dict(m503.points(M5_03_DESIGN))
    regime, day, split = job["regime"], job["day"], job["split"]
    regime_dict, seed = regime_day(m4_05, flows[job["flow"]], split, regime, day)
    market = day_market(regime_dict, m4_05, seed, day)
    record = m503.digests_of(f"{split}-{job['flow']}")[(regime, day)]
    if market.digest() != record["dataset_digest"]:
        raise ValueError(f"{split}-{job['flow']} {regime} day {day}: not #108's rows")
    kappa = job["kappa_index"] * design["impact_scale"]["values"][regime]["kappa0"]
    policy, make_env = arm_policy(job["arm"], regime, day, market.rows(), frozen, runs, m4_05,
                                  job.get("execution"), job.get("metadata"))
    result = reactive_episode(policy, make_env, market, kappa, impact_rng(seed),
                              design["evaluation_seed"], m4_05)
    peak, peak_price = Decimal(0), None
    for _, price, after in result["positions"]:
        if abs(Decimal(after)) > abs(peak):
            peak, peak_price = Decimal(after), price
    attribution = result["attribution"]
    return {"scenario": f"{split}-{job['flow']}", "point": job["point"],
            "kappa_index": job["kappa_index"], "kappa": kappa,
            "condition": job.get("condition"), "arm": job["arm"],
            "seed": job.get("metadata", {}).get("seed"), "regime": regime, "day": day,
            "net": result["net"], "max_drawdown": attribution.get("max_drawdown"),
            "max_abs_position": attribution.get("max_abs_position"),
            "mean_abs_position": attribution.get("mean_abs_position"),
            "peak_notional": result["peak_notional"], "refusals": result["refusals"],
            "residual_fallbacks": result["residual_fallbacks"],
            "residual_clipped": result["residual_clipped"],
            "limit_breach": result["limit_breach"], "peak_position": str(peak),
            "peak_position_price": peak_price, "reactions": result["reactions"],
            "moves": result["moves"], "emptied": result["emptied"],
            "replayed_steps": result["replayed_steps"], "stopped": result["stopped"]}


def jobs_for(points, split, days, runs, common, extra=None):
    m4_05 = json.loads((HERE.parent / "m4_05" / "design.json").read_text())
    jobs = []
    for name, flow, _, k in points:
        for regime in [r["name"] for r in m4_05["simulator"]["regimes"]]:
            for day in days:
                for arm in m503.ARMS:
                    base_job = {**common, "point": name, "flow": flow, "kappa_index": k,
                                "split": split, "regime": regime, "day": day, "arm": arm,
                                **(extra or {})}
                    if arm in runs:
                        jobs.extend({**base_job, "metadata": m} for m in runs[arm])
                    else:
                        jobs.append(base_job)
    return jobs


def job_key(job):
    seed = job.get("metadata", {}).get("seed")
    execution = json.dumps(job.get("execution"), sort_keys=True)
    return (job["split"], job["point"], job.get("condition"), execution, job["arm"], seed,
            job["regime"], job["day"])


def pool(jobs, parallel, label, journal, limit=None, out=None):
    """As #108's: each result is journaled as it ends, and a resumed step skips those. Past
    `limit` wall seconds the workers are killed (by their own pids), what ended stays
    journaled, and Overrun is raised; the wall time is charged to budget.json in `out`."""
    import concurrent.futures
    import multiprocessing
    journal = pathlib.Path(journal)
    done_rows = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            done_rows[tuple(json.loads(entry["key"]))] = entry["row"]
    keys = [job_key(job) for job in jobs]
    todo = [(k, j) for k, j in zip(keys, jobs, strict=True) if k not in done_rows]
    started = time.perf_counter()
    print(f"{label}: {len(jobs) - len(todo)} of {len(jobs)} already done; {parallel} processes",
          file=sys.stderr, flush=True)
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context)
    try:
        with journal.open("a") as written:
            futures = {executor.submit(run_job, job): key for key, job in todo}
            timeout = None if limit is None else max(limit, 0)
            try:
                for done, future in enumerate(
                        concurrent.futures.as_completed(futures, timeout=timeout), start=1):
                    row = future.result()
                    key = futures[future]
                    done_rows[key] = row
                    written.write(json.dumps({"key": json.dumps(list(key)), "row": row}) + "\n")
                    written.flush()
                    if done % 50 == 0 or done == len(todo):
                        print(f"{label}: {done}/{len(todo)} after "
                              f"{round(time.perf_counter() - started)} s", file=sys.stderr,
                              flush=True)
            except concurrent.futures.TimeoutError:
                for process in list(getattr(executor, "_processes", {}).values()):
                    if process.is_alive():
                        process.kill()
                raise Overrun(f"{label} reached the process-hour ceiling after "
                              f"{round(time.perf_counter() - started)} s") from None
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if out is not None and todo:
            charge(out, label, time.perf_counter() - started, parallel)
    return [done_rows[key] for key in keys]


def identity(rows, selection_108):
    """Every kappa-0 row against #108's selection row of the same arm, seed, regime, day and
    flow point, field by field. The mismatches, empty when it holds."""
    theirs = {(r["scenario"], r["arm"], r["seed"], r["regime"], r["day"]): r
              for r in selection_108["rows"]}
    wrong = []
    for row in rows:
        other = theirs.get((row["scenario"], row["arm"], row["seed"], row["regime"], row["day"]))
        if other is None or any(row[f] != other[f] for f in FIELDS_108):
            wrong.append((row["scenario"], row["arm"], row["seed"], row["regime"], row["day"]))
    return wrong


class Study:
    def __init__(self, run_dir, out_dir, commit):
        torch.set_num_threads(1)
        import os
        self.design, self.m4_05 = load()
        self.run_dir = run_dir
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        frozen_path = pathlib.Path(run_dir) / "frozen.json"
        if sha256(frozen_path) != M5_03_DESIGN["reuses"]["frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one #108 reuses")
        self.frozen = json.loads(frozen_path.read_text())
        self.runs = m503.learned_runs(self.frozen, run_dir)
        cap = self.design["compute"]["max_parallel_runs"]
        self.parallel = min(cap, int(os.environ.get("M5_04_PROCESSES", cap)))
        if self.parallel < 1:
            raise ValueError("no processes: the design's compute cap is 0")

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "code_sha256": sha256(HERE / "run.py"), "versions": base.versions()}

    def common(self):
        return {"run_dir": self.run_dir}

    def write(self, name, report):
        (self.out / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def selection(study, selection_108_path):
    """The 72 points on the selection days: kappa 0 first, checked against #108's rows, then
    kappa > 0; then the adversary's choice per arm."""
    selection_108 = json.loads(pathlib.Path(selection_108_path).read_text())
    if sha256(selection_108_path) != study.design["identity"]["selection_108_sha256"]:
        raise ValueError("#108's selection.json is not the one the design names")
    days = list(range(1, M5_03_DESIGN["days"]["selection"]["per_regime"] + 1))
    points = grid_points()
    zero = [p for p in points if p[3] == 0]
    rows = pool(jobs_for(zero, "selection", days, study.runs, study.common()), study.parallel,
                "selection kappa 0", study.out / "selection.journal")
    wrong = identity(rows, selection_108)
    if wrong:
        report = study.header("selection")
        report["identity_failures"] = wrong
        study.write("identity_failure", report)
        raise SystemExit(f"stop: kappa 0 differs from #108 in {len(wrong)} rows")
    rest = [p for p in points if p[3] != 0]
    limit = limit_for(study.out, study.parallel, HOLDOUT_RESERVE_PROCESS_HOURS)
    rows += pool(jobs_for(rest, "selection", days, study.runs, study.common()), study.parallel,
                 "selection kappa > 0", study.out / "selection.journal", limit, study.out)
    by_point = {}
    for name, *_ in points:
        scoped = m503.daily([r for r in rows if r["point"] == name])
        by_point[name] = {arm: [v for (a, _, _), v in scoped.items() if a == arm]
                          for arm in m503.ARMS}
    report = study.header("selection")
    report["rows"] = rows
    report["identity"] = "every kappa-0 row equals #108's selection row"
    report["chosen"] = m503.choose(by_point)
    study.write("selection", report)
    return report


def freeze(study):
    selection_path = study.out / "selection.json"
    manifest = study.header("freeze")
    manifest["selection_sha256"] = sha256(selection_path)
    manifest["digests_sha256"] = m503.digest_hashes(M5_03 / "digests")
    manifest["chosen"] = json.loads(selection_path.read_text())["chosen"]
    path = study.out / "frozen.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"frozen_sha256": sha256(path)}


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
    m503.check_frozen_digests(frozen, M5_03 / "digests")
    days = list(range(1, M5_03_DESIGN["days"]["holdout"]["per_regime"] + 1))
    unit = unit_flow()
    stress_flow = m503.point_name(M5_03_DESIGN["fixed_stress"])
    by_name = {p[0]: p for p in grid_points()}
    jobs = []
    for condition, point, execution in (
        ("unperturbed", f"{unit}-k0", None),
        ("impact_only", f"{unit}-k2", None),
        ("fixed_stress", f"{stress_flow}-k2", None),
        ("execution_stress", f"{unit}-k0", M5_03_DESIGN["execution_stress"]["fill_model"]),
    ):
        jobs += jobs_for([by_name[point]], "holdout", days, study.runs, study.common(),
                         {"condition": condition, "execution": execution})
    for arm, choice in frozen["chosen"].items():
        jobs += [j for j in jobs_for([by_name[choice["point"]]], "holdout", days, study.runs,
                                     study.common(), {"condition": "adversary"})
                 if j["arm"] == arm]
    limit = limit_for(study.out, study.parallel, 0.0)
    # Before any holdout day is read: a second or concurrent attempt is refused.
    try:
        marker = os.open(study.out / "holdout.started", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("the holdout was attempted already") from error
    with os.fdopen(marker, "w") as handle:
        handle.write(json.dumps({"started": time.time(), "pid": os.getpid(),
                                 "frozen_sha256": frozen_sha256}) + "\n")
    rows = pool(jobs, study.parallel, "holdout", study.out / "holdout.journal", limit, study.out)
    here["frozen_sha256"] = frozen_sha256
    here["rows"] = rows
    study.write("holdout", here)
    return here


def main():
    step = sys.argv[1]
    if step in ("selection", "freeze", "holdout"):
        study = Study(*sys.argv[2:5])
        started = time.perf_counter()
        if step == "selection":
            report = selection(study, sys.argv[5])
        elif step == "freeze":
            report = freeze(study)
        else:
            report = holdout(study, sys.argv[5])
        summary = {k: v for k, v in report.items() if not isinstance(v, list)}
        print(json.dumps({"step": step, "seconds": round(time.perf_counter() - started, 1),
                          **summary}, indent=2, default=str))
        return
    if step == "rate":
        report = rate(*sys.argv[2:5])
        print(json.dumps(report["arms"], indent=2))
    elif step == "check":
        report = check(*sys.argv[2:5])
        print(json.dumps(report["arms"], indent=2))
    else:
        raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
