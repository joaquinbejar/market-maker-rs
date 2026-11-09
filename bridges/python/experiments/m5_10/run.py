"""The steps of #108 part 2 (part 1 of its design): selection, freeze and a single holdout.

SYNTHETIC. The job lists and guards are plain Python; `evaluate` imports #107's runner lazily, so
loading this module trains and reads nothing. The marker directory is an argument, kept outside
the output directory.
"""

import functools
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import grid  # noqa: E402
import stats  # noqa: E402

SETTINGS = {"impact_0": 0.0, "impact_low": 0.25, "impact_high": 1.0}
CLASSICAL = ("avellaneda_stoikov", "glft")
LEARNED = ("mlp", "lstm")
SEEDS = (1, 2, 3, 4, 5)
SELECTION_DAYS, HOLDOUT_DAYS = 10, 30
OFFSETS = {"selection": 100000, "holdout": 300000}
MAX_RECOVERIES = 1
FEE = "-0.0001"


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def policies(setting):
    """The 13 policy runs of a setting."""
    out = [{"arm": "no_trade", "setting": setting, "seed": None}]
    out += [{"arm": a, "setting": setting, "seed": None} for a in CLASSICAL]
    out += [{"arm": a, "setting": setting, "seed": s} for a in LEARNED for s in SEEDS]
    return out


def point_id(point):
    return f"m{point['market']}-l{point['limit']}-b{point['branching']}-i{point['impact']}"


def selection_jobs(settings=SETTINGS):
    return [{"split": "selection", "policy": p, "point": pt, "day": d}
            for s, impact in settings.items() for pt in grid.points(impact)
            for p in policies(s) for d in range(1, SELECTION_DAYS + 1)]


def condition_points(impact, selected, removed=()):
    """unperturbed, fixed stress, the arm's own adversary point and impact only.

    A removed named point leaves its condition not run, with no substitute; a removed
    unperturbed point voids the setting.
    """
    named = {"unperturbed": grid.points(impact)[0], "fixed_stress": grid.fixed_stress(impact),
             "adversary": selected, "impact_only": grid.impact_only(impact)}
    if point_id(named["unperturbed"]) in removed:
        return {}
    return {name: pt for name, pt in named.items() if point_id(pt) not in removed}


def policy_key(policy):
    """The seeds of a learned arm are one policy."""
    return f"{policy['setting']}/{policy['arm']}"


def holdout_jobs(selected_by_policy, settings=SETTINGS, removed=None):
    jobs = []
    for s, impact in settings.items():
        if point_id(grid.points(impact)[0]) in (removed or {}).get(s, ()):
            continue  # a void setting: it has no selection and is reported as not run
        for p in policies(s):
            conditions = condition_points(impact, selected_by_policy[policy_key(p)]["point"],
                                          (removed or {}).get(s, ()))
            for name, pt in conditions.items():
                jobs += [{"split": "holdout", "condition": name, "policy": p, "point": pt, "day": d}
                         for d in range(1, HOLDOUT_DAYS + 1)]
    return jobs


def check_classical(classical_path, expected_sha, settings=SETTINGS):
    """#107's classical points, by hash; a missing arm for a setting fails loudly."""
    if sha256(classical_path) != expected_sha:
        raise SystemExit(f"{classical_path} is not the file #107 froze")
    selected = json.loads(pathlib.Path(classical_path).read_text())["selected"]
    for setting in settings:
        for arm in CLASSICAL:
            if arm not in selected.get(setting, {}):
                raise SystemExit(f"no {arm} point for {setting}")
    return selected


def job_key(job):
    p = job["policy"]
    return "|".join(str(x) for x in (job["split"], job.get("condition", ""), p["setting"], p["arm"],
                                     p["seed"], point_id(job["point"]), job["day"]))


def run_journaled(jobs, journal, evaluate):
    """Runs the jobs not yet in the journal, appending each result; a restart skips the done ones."""
    journal = pathlib.Path(journal)
    done = load_journal(journal)
    with journal.open("a") as handle:
        for job in jobs:
            key = job_key(job)
            if key not in done:
                done[key] = {"key": key, "result": evaluate(job)}
                handle.write(json.dumps(done[key]) + "\n")
                handle.flush()
    return [done[job_key(job)] for job in jobs]


def select(rows, settings=SETTINGS):
    """{policy_key: the point and its selection loss}: mean daily loss, seeds of an arm averaged.

    Candidates are taken in grid order, so a tie goes to the first point of the grid.
    """
    by = {}
    for job, row in rows:
        key = policy_key(job["policy"])
        pid = point_id(job["point"])
        by.setdefault(key, {}).setdefault(pid, {}).setdefault(job["day"], []).append(-row["net"])
    out = {}
    for key, per_point in by.items():
        points = [p for p in grid.points(settings[key.split("/")[0]]) if point_id(p) in per_point]
        losses = {i: [sum(v) / len(v) for v in per_point[point_id(p)].values()] for i, p in enumerate(points)}
        best, means = stats.pick_adversary(points, losses)
        out[key] = {"point": points[best], "selection_loss": means[best]}
    return out


def claim(marker_dir, design_sha, journal=None, recover=None):
    """One attempt per design; a recovery only after an external kill, its journal prefix recorded.

    On a recovery the torn last line is repaired, then the journal is hashed into the marker, and
    `verify_prefix` checks at the end that those bytes did not change.
    """
    marker = pathlib.Path(marker_dir) / f"holdout-{design_sha}.jsonl"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not recover:
            raise SystemExit("a holdout attempt already started; a retry needs a recovery reason") from None
        lines = [json.loads(x) for x in marker.read_text().splitlines()]
        if sum(x["kind"] == "recovery" for x in lines) >= MAX_RECOVERIES:
            raise SystemExit("too many recoveries")
        load_journal(journal)
        data = pathlib.Path(journal).read_bytes()
        with marker.open("a") as handle:
            handle.write(json.dumps({"kind": "recovery", "reason": recover, "prefix_bytes": len(data),
                                     "prefix_sha256": hashlib.sha256(data).hexdigest()}) + "\n")
    else:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"kind": "attempt", "design_sha256": design_sha}) + "\n")
    return marker


def verify_prefix(marker, journal):
    """Raises if a recovery's recorded journal prefix changed by the end of the run."""
    data = pathlib.Path(journal).read_bytes()
    for line in pathlib.Path(marker).read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "recovery" and hashlib.sha256(data[: row["prefix_bytes"]]).hexdigest() != row["prefix_sha256"]:
            raise SystemExit("the journal's recorded prefix changed")


def day_path(data_dir, split, setting, point, day):
    return pathlib.Path(data_dir) / split / setting / point_id(point) / "holdout" / "p" / f"day-{day}.json"


def write_day_configs(causal, out, settings=SETTINGS):
    """One hawkes_days input per split, setting and point; run the example on each."""
    out = pathlib.Path(out)
    written = []
    for split, days in (("selection", SELECTION_DAYS), ("holdout", HOLDOUT_DAYS)):
        for setting, impact in settings.items():
            for point in grid.points(impact):
                path = out / split / setting / f"{point_id(point)}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                config = grid.day_config(causal, point, "p", days, OFFSETS[split])
                path.write_text(json.dumps({"causal": config}))
                written.append(path)
    return written


@functools.lru_cache(maxsize=1)
def _runner():
    """#107's runner, whose environment, wrappers and checkpoint checks are reused unchanged."""
    import importlib.util

    path = HERE.parent / "m5_02" / "run_market.py"
    spec = importlib.util.spec_from_file_location("m5_02_run_market", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(rm, job, rows, classical):
    design, base = rm.load()
    config = rm.m6.base_config(base, FEE)
    config["fill_model"] = dict(design["execution"]["fill_model"])
    arm, setting = job["policy"]["arm"], job["policy"]["setting"]
    if arm in CLASSICAL:
        config["residual"] = {"baseline": rm.m6.baseline_config(arm, classical[setting][arm], job["day"]),
                              "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        return base, rm.DecisionInterval(rm.ResidualMarketMakingEnv(config, rows), rm.EVERY)
    return base, rm.DecisionInterval(rm.BoxMarketMakingEnv(config, rows), rm.EVERY)


def evaluate(job, data_dir, checkpoints, classical, digests):
    """One day of one policy under #107's execution; the rows are checked against their digest."""
    rm = _runner()
    rm.torch.set_num_threads(1)
    policy = job["policy"]
    path = day_path(data_dir, job["split"], policy["setting"], job["point"], job["day"])
    if sha256(path) != digests[str(path.relative_to(data_dir))]:
        raise ValueError(f"{path} is not the committed day")
    rows = [tuple(r) for r in json.loads(path.read_text())]
    base, env = _env(rm, job, rows, classical)
    try:
        if policy["arm"] == "no_trade":
            model = rm.m6.Constant(rm.m6.HOLD)
        elif policy["arm"] in CLASSICAL:
            model = rm.m6.ZeroResidual()
        else:
            name = f"{policy['arm']}-{policy['setting']}-seed{policy['seed']}"
            model = rm.frozen_policy(json.loads((pathlib.Path(checkpoints) / f"{name}.json").read_text()), env)
        days = [rm.m6.episode(model, env, s, policy["arm"] == "lstm", base)
                for s in base["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    if len(days) != 1:
        raise ValueError("the design evaluates one seed per day")
    return {"net": float(days[0]["net"]), "episode": days[0]}


PILOT_SETTING = "impact_high"
PILOT_OFFSET = 500000
PILOT_DAYS = 2
# Selection: 24 points x 10 days x 3 settings. Holdout: at most 8 points per setting (the three
# named conditions and each of the 5 arms' own pick) x 30 days x 3 settings.
GENERATED_DAYS = 24 * SELECTION_DAYS * 3 + 8 * HOLDOUT_DAYS * 3


def pilot_jobs():
    """2 days per arm at the unperturbed and the most active point, on seeds outside every split."""
    impact = SETTINGS[PILOT_SETTING]
    return [{"split": "pilot", "policy": p, "point": pt, "day": d}
            for pt in (grid.points(impact)[0], grid.fixed_stress(impact))
            for p in policies(PILOT_SETTING) for d in range(1, PILOT_DAYS + 1)]


def project(rows, generation_seconds_per_day=0.0, ceiling_hours=90.0):
    """Process-hours for the whole count from the pilot's per-day seconds, the slowest point's."""
    per_arm = {}
    for job, seconds in rows:
        per_arm.setdefault(job["policy"]["arm"], {}).setdefault(point_id(job["point"]), []).append(seconds)
    worst = {arm: max(sum(v) / len(v) for v in points.values()) for arm, points in per_arm.items()}
    runs = {"no_trade": 1, "avellaneda_stoikov": 1, "glft": 1, "mlp": 5, "lstm": 5}
    per_setting_day = sum(worst[arm] * n for arm, n in runs.items())
    days = (24 * SELECTION_DAYS + 4 * HOLDOUT_DAYS) * len(SETTINGS)
    hours = per_setting_day * days / 3600
    generation = generation_seconds_per_day * GENERATED_DAYS / 3600
    total = hours + generation
    return {"worst_seconds_per_day": worst, "policy_hours": hours, "generation_seconds_per_day": generation_seconds_per_day,
            "generated_days": GENERATED_DAYS, "generation_hours": generation, "projected_hours": total,
            "over_ceiling": total > ceiling_hours}


def spectral_radii(weights, points):
    """{point id: radius} and the ids failing the design's check; those are removed and reported."""
    radii = {point_id(p): grid.spectral_radius(grid.branching_matrix(weights, p["branching"])) for p in points}
    return radii, sorted(point_id(p) for p in points if not grid.plausible(weights, p))


def spectral_check(weights, points):
    """The pilot's two points: it cannot time a point the study would remove, so it stops."""
    radii, failed = spectral_radii(weights, points)
    if failed:
        raise SystemExit(f"{failed[0]} is not subcritical")
    return radii


def generate_pilot_days(causal, data_dir, binary):
    """Writes the pilot's days with examples/hawkes_days and returns their sha256 by relative path."""
    import subprocess

    generation = {}
    data_dir = pathlib.Path(data_dir)
    impact = SETTINGS[PILOT_SETTING]
    digests = {}
    points = (grid.points(impact)[0], grid.fixed_stress(impact))
    radii = spectral_check(causal["flow"]["weights"], points)
    for point in points:
        out = data_dir / "pilot" / PILOT_SETTING / point_id(point)
        out.mkdir(parents=True, exist_ok=True)
        config = out / "config.json"
        config.write_text(json.dumps({"causal": grid.day_config(causal, point, "p", PILOT_DAYS, PILOT_OFFSET)}))
        start = time.perf_counter()
        subprocess.run([str(binary), str(config), str(out)], check=True, stdout=subprocess.DEVNULL)
        generation[point_id(point)] = (time.perf_counter() - start) / PILOT_DAYS
        for day in range(1, PILOT_DAYS + 1):
            path = day_path(data_dir, "pilot", PILOT_SETTING, point, day)
            digests[str(path.relative_to(data_dir))] = sha256(path)
    return digests, radii, generation


def pilot_step(argv):
    """python run.py pilot <out> <hawkes_days binary> <#107 causal output dir>"""
    out, binary, causal_out = argv
    out, causal_out = pathlib.Path(out), pathlib.Path(causal_out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((causal_out / "frozen.json").read_text())
    classical = check_classical(causal_out / "classical.json", frozen["classical_sha256"])
    causal = json.loads((HERE.parent / "m5_02" / "design.json").read_text())["causal"]
    if (out / "pilot.jsonl").exists():
        raise SystemExit("a pilot output already exists; one pilot per output directory")
    digests, radii, generation = generate_pilot_days(causal, out, binary)
    _runner()  # imported once here, so its one-off cost is outside every job's time
    rows = []
    with (out / "pilot.jsonl").open("x") as handle:
        for job in pilot_jobs():
            start = time.perf_counter()
            result = evaluate(job, out, causal_out / "checkpoints", classical, digests)
            seconds = time.perf_counter() - start
            rows.append((job, seconds))
            handle.write(json.dumps({"key": job_key(job), "seconds": seconds, "net": result["net"]}) + "\n")
            handle.flush()
    report = {"projection": project(rows, max(generation.values())), "generation_seconds_per_day": generation,
              "first_job_seconds": rows[0][1], "spectral_radius": radii, "load": os.getloadavg()}
    (out / "pilot.json").write_text(json.dumps(report, indent=1))


def causal_section():
    return json.loads((HERE.parent / "m5_02" / "design.json").read_text())["causal"]


def generate(causal, out, binary, split, setting, point, days):
    """Writes one point's days and returns {relative path: sha256}."""
    import subprocess

    out = pathlib.Path(out)
    target = out / split / setting / point_id(point)
    target.mkdir(parents=True, exist_ok=True)
    config = target / "config.json"
    config.write_text(json.dumps({"causal": grid.day_config(causal, point, "p", days, OFFSETS[split])}))
    subprocess.run([str(binary), str(config), str(target)], check=True, stdout=subprocess.DEVNULL)
    paths = []
    for d in range(1, days + 1):
        base = day_path(out, split, setting, point, d)
        paths += [base, base.with_name(f"day-{d}.flow.json"), base.with_name(f"day-{d}.queues.json")]
    return {str(p.relative_to(out)): sha256(p) for p in paths}


def days_selection(out, binary, digests_path, pilot_jsonl, settings=SETTINGS):
    """Every point's selection days, after the spectral check of every point; digests to commit.

    A point failing that check is reported and generates no day; a failing unperturbed point
    voids its setting.
    """
    budget = Budget(out, pilot_jsonl)
    start = time.perf_counter()
    causal = causal_section()
    radii, removed, digests = {}, {}, {}
    for setting, impact in settings.items():
        radii[setting], removed[setting] = spectral_radii(causal["flow"]["weights"], grid.points(impact))
        if point_id(grid.points(impact)[0]) in removed[setting]:
            continue
        for point in grid.points(impact):
            if point_id(point) not in removed[setting]:
                digests.update(generate(causal, out, binary, "selection", setting, point, SELECTION_DAYS))
    pathlib.Path(digests_path).write_text(json.dumps(
        {"digests": digests, "spectral_radius": radii, "spectral_removed": removed}, indent=0))
    budget.charge("days-selection", (time.perf_counter() - start) / 3600)


def point_statistics(out, split, setting, point, days):
    """Queue shares, market rate and impact slope of one point's days, with no policy."""
    import numpy as np

    cap = causal_section()["flow"]["queue_cap"]
    below, queue, open_, rate, changes, signed = [], [], [], [], [], []
    for day in range(1, days + 1):
        base = day_path(out, split, setting, point, day)
        queues = np.asarray(json.loads(base.with_name(f"day-{day}.queues.json").read_text()), float)
        flow = np.asarray(json.loads(base.with_name(f"day-{day}.flow.json").read_text()), float)
        rows = np.asarray(json.loads(base.read_text()), float)
        below.append(float((queues < cap).mean()))
        queue.append(float(queues.mean()))
        open_.append(float((queues > 0).mean()))
        rate.append(float(flow.sum(axis=1).mean() / 2))
        # Row r's orders move the mid of row r + 1; mids are in ticks, the bid being floor(mid).
        mids = (rows[:, 1] + rows[:, 3]) / 2
        changes.append(mids[1:] - mids[:-1])
        signed.append(flow[:-1, 0] - flow[:-1, 1])
    return {"shares": {"below_cap": float(np.mean(below)), "mean_queue": float(np.mean(queue)),
                       "non_empty": float(np.mean(open_))},
            "cap_share": 1 - float(np.mean(below)), "empty_share": 1 - float(np.mean(open_)),
            "market_rate": float(np.mean(rate)),
            "slope": grid.slope(np.concatenate(changes), np.concatenate(signed))}


def identify(out, report_path, digests_path, settings=SETTINGS):
    """The identifiability rules on the selection days; failing points go in `removed`."""
    causal = causal_section()
    rates, weights = causal["flow"]["rates"], causal["flow"]["weights"]
    record = json.loads(pathlib.Path(digests_path).read_text())
    spectral = record["spectral_removed"]
    for relative, expected in record["digests"].items():
        if sha256(pathlib.Path(out) / relative) != expected:
            raise SystemExit(f"{relative} is not the committed day")
    report = {}
    for setting, impact in settings.items():
        points = grid.points(impact)
        removed = list(spectral[setting])
        rows = {}
        if point_id(points[0]) not in removed:
            base_stats = point_statistics(out, "selection", setting, points[0], SELECTION_DAYS)
            base_formula = grid.rate_formula(rates, weights, points[0], base_stats["shares"])
            for point in points:
                if point_id(point) in spectral[setting]:
                    continue
                st = point_statistics(out, "selection", setting, point, SELECTION_DAYS)
                formula = grid.rate_formula(rates, weights, point, st["shares"])
                rate_ok = grid.rate_passes(st["market_rate"], base_stats["market_rate"], formula, base_formula)
                impact_ok = grid.impact_passes(st["slope"], point["impact"])
                rows[point_id(point)] = {**st, "formula": formula, "rate_ok": rate_ok, "impact_ok": impact_ok}
                if not (rate_ok and impact_ok):
                    removed.append(point_id(point))
        report[setting] = {"points": rows, "removed": removed, "spectral_removed": spectral[setting],
                           "void": point_id(points[0]) in removed}
    pathlib.Path(report_path).write_text(json.dumps(report, indent=1))


def _worker(args):
    job, ctx = args
    return job, evaluate(job, ctx["out"], ctx["checkpoints"], ctx["classical"], ctx["digests"])


CEILING_HOURS = 90.0
STOP_FACTOR = 1.5


class Budget:
    """The process-hours ledger of the whole study; every step is charged, the pilot first."""

    def __init__(self, out, pilot_jsonl):
        self.path = pathlib.Path(out) / "budget.jsonl"
        if not self.path.exists():
            lines = pathlib.Path(pilot_jsonl).read_text().splitlines()
            seconds = sum(json.loads(x)["seconds"] for x in lines)
            self.charge("pilot", seconds / 3600)

    def spent(self):
        return sum(json.loads(x)["process_hours"] for x in self.path.read_text().splitlines())

    def remaining(self):
        return CEILING_HOURS - self.spent()

    def charge(self, step, process_hours):
        with self.path.open("a") as handle:
            handle.write(json.dumps({"step": step, "process_hours": process_hours}) + "\n")


def step_limit_seconds(jobs, worst_seconds, processes, remaining_hours):
    """The wall limit of a step: 1.5 x its pilot-based estimate, or what the ceiling leaves."""
    estimate = sum(worst_seconds[j["policy"]["arm"]] for j in jobs) / processes
    return min(STOP_FACTOR * estimate, remaining_hours * 3600 / processes), estimate


def run_pool(step, jobs, journal, processes, ctx, budget, worst_seconds):
    """As run_journaled, with a pool; only the parent writes the journal.

    Stops at the step's limit: the workers are terminated, the time is charged, and it reports.
    """
    import multiprocessing

    journal = pathlib.Path(journal)
    done = load_journal(journal)
    todo = [j for j in jobs if job_key(j) not in done]
    limit, estimate = step_limit_seconds(todo, worst_seconds, processes, budget.remaining())
    start = time.perf_counter()
    stopped = False
    try:
        with multiprocessing.get_context("spawn").Pool(processes) as pool, journal.open("a") as handle:
            results = pool.imap_unordered(_worker, [(j, ctx) for j in todo])
            for _ in todo:
                left = limit - (time.perf_counter() - start)
                try:
                    job, result = results.next(timeout=max(left, 0.001))
                except multiprocessing.TimeoutError:
                    stopped = True
                    pool.terminate()
                    break
                row = {"key": job_key(job), "result": result}
                done[row["key"]] = row
                handle.write(json.dumps(row) + "\n")
                handle.flush()
    finally:
        budget.charge(step, (time.perf_counter() - start) * processes / 3600)
    wall = time.perf_counter() - start
    if stopped:
        raise SystemExit(f"{step} stopped at {wall:.0f} s, its limit (estimate {estimate:.0f} s): reported, no cut")
    return [(j, done[job_key(j)]["result"]) for j in jobs]


def load_journal(journal):
    journal = pathlib.Path(journal)
    if not journal.exists():
        return {}
    lines = journal.read_text().splitlines()
    done = {}
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            journal.write_text("".join(x + "\n" for x in lines[:-1]))
            break
        done[row["key"]] = row
    return done


def check_checkpoints(causal_out, frozen, settings=SETTINGS):
    """Each learned checkpoint and its metadata against #107's frozen manifest."""
    for setting in settings:
        for arm in LEARNED:
            for seed in SEEDS:
                name = f"{arm}-{setting}-seed{seed}"
                run = frozen["runs"][name]
                if sha256(causal_out / "checkpoints" / f"{name}.zip") != run["checkpoint"] \
                        or sha256(causal_out / "checkpoints" / f"{name}.json") != run["metadata"]:
                    raise SystemExit(f"{name} is not the checkpoint #107 froze")


def context(out, causal_out, digests_path):
    causal_out = pathlib.Path(causal_out)
    frozen = json.loads((causal_out / "frozen.json").read_text())
    check_checkpoints(causal_out, frozen)
    return {"out": str(out), "checkpoints": str(causal_out / "checkpoints"),
            "classical": check_classical(causal_out / "classical.json", frozen["classical_sha256"]),
            "digests": json.loads(pathlib.Path(digests_path).read_text())["digests"]}


def removed_points(identify_path):
    report = json.loads(pathlib.Path(identify_path).read_text())
    return {s: set(v["removed"]) for s, v in report.items()}


def worst_seconds_of(pilot_dir):
    return json.loads((pathlib.Path(pilot_dir) / "pilot.json").read_text())["projection"]["worst_seconds_per_day"]


def selection_step(out, causal_out, digests_path, identify_path, pilot_dir, processes):
    report = json.loads(pathlib.Path(identify_path).read_text())
    removed = removed_points(identify_path)
    jobs = [j for j in selection_jobs()
            if not report[j["policy"]["setting"]]["void"] and point_id(j["point"]) not in removed[j["policy"]["setting"]]]
    budget, worst = Budget(out, pathlib.Path(pilot_dir) / "pilot.jsonl"), worst_seconds_of(pilot_dir)
    rows = run_pool("selection", jobs, pathlib.Path(out) / "selection.jsonl", processes,
                    context(out, causal_out, digests_path), budget, worst)
    pathlib.Path(out, "selection.json").write_text(json.dumps(select(rows), indent=1))


def code_hashes():
    paths = [HERE / "run.py", HERE / "grid.py", HERE / "stats.py", HERE.parent / "m5_02" / "run_market.py"]
    return {p.name: sha256(p) for p in paths}


def secondary_selection_step(out, causal_out, digests_path, identify_path, pilot_dir, processes):
    """The removed points' selection rows, after the primary's selection is written.

    The secondary's own pick is over every point; it reads the primary's rows, never writes them.
    """
    out = pathlib.Path(out)
    if not (out / "selection.json").exists():
        raise SystemExit("the primary's selection comes first")
    report = json.loads(pathlib.Path(identify_path).read_text())
    removed = removed_points(identify_path)
    spectral = {s: set(v["spectral_removed"]) for s, v in report.items()}
    all_jobs = [j for j in selection_jobs() if not report[j["policy"]["setting"]]["void"]
                and point_id(j["point"]) not in spectral[j["policy"]["setting"]]]
    jobs = [j for j in all_jobs if point_id(j["point"]) in removed[j["policy"]["setting"]]]
    budget, worst = Budget(out, pathlib.Path(pilot_dir) / "pilot.jsonl"), worst_seconds_of(pilot_dir)
    rows = run_pool("selection-secondary", jobs, out / "selection_secondary.jsonl", processes,
                    context(out, causal_out, digests_path), budget, worst)
    primary = load_journal(out / "selection.jsonl")
    primary_rows = [(j, primary[job_key(j)]["result"]) for j in all_jobs if job_key(j) in primary]
    (out / "selection_secondary.json").write_text(json.dumps(select(primary_rows + rows), indent=1))


def secondary_jobs(picks, removed, primary_picks, settings=SETTINGS):
    """Labelled secondary holdout jobs: the fixed stress and the pick over every point."""
    jobs = []
    for s, impact in settings.items():
        if point_id(grid.points(impact)[0]) in removed.get(s, ()):
            continue
        for p in policies(s):
            named = {}
            if point_id(grid.fixed_stress(impact)) in removed.get(s, ()):
                named["secondary_fixed_stress"] = grid.fixed_stress(impact)
            pick = picks[policy_key(p)]["point"]
            if pick != primary_picks[policy_key(p)]["point"]:
                named["secondary_adversary"] = pick  # the primary's own pick is reused in the report
            jobs += [{"split": "holdout", "condition": c, "policy": p, "point": pt, "day": d}
                     for c, pt in named.items() for d in range(1, HOLDOUT_DAYS + 1)]
    return jobs


def freeze(out, design_path, digests_path, identify_path, pilot_paths, m107_frozen, binary):
    """The manifest the holdout runs from: every input by hash, and each policy's pick."""
    out = pathlib.Path(out)
    selected = json.loads((out / "selection.json").read_text())
    manifest = {"design_sha256": sha256(design_path), "selection_digests_sha256": sha256(digests_path),
                "identify_sha256": sha256(identify_path), "selection_sha256": sha256(out / "selection.json"),
                "selection_secondary_sha256": sha256(out / "selection_secondary.json"),
                "selected_secondary": json.loads((out / "selection_secondary.json").read_text()),
                "pilot_sha256": {str(p): sha256(p) for p in pilot_paths}, "m107_frozen_sha256": sha256(m107_frozen),
                "m107_frozen_path": str(m107_frozen), "generator_sha256": sha256(binary), "code_sha256": code_hashes(),
                "selected": selected}
    path = out / "frozen.json"
    if path.exists():
        raise SystemExit("already frozen")
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return sha256(path)


def verify_manifest(out, design_path, selection_digests, identify_path, holdout_digests, frozen_sha):
    """Every frozen input and the code, checked before the claim."""
    out = pathlib.Path(out)
    if sha256(out / "frozen.json") != frozen_sha:
        raise SystemExit("frozen.json is not the freeze named")
    m = json.loads((out / "frozen.json").read_text())
    checks = {"design": (design_path, m["design_sha256"]),
              "selection digests": (selection_digests, m["selection_digests_sha256"]),
              "identify": (identify_path, m["identify_sha256"]),
              "selection": (out / "selection.json", m["selection_sha256"]),
              "secondary selection": (out / "selection_secondary.json", m["selection_secondary_sha256"])}
    checks["#107 freeze"] = (m["m107_frozen_path"], m["m107_frozen_sha256"])
    checks.update({f"pilot {p}": (p, h) for p, h in m["pilot_sha256"].items()})
    for name, (path, expected) in checks.items():
        if sha256(path) != expected:
            raise SystemExit(f"{name} is not the frozen one")
    if code_hashes() != m["code_sha256"]:
        raise SystemExit("the code is not the frozen one")
    days = json.loads((out / "holdout_days.json").read_text())
    if days["frozen_sha256"] != frozen_sha or days["digests_sha256"] != sha256(holdout_digests):
        raise SystemExit("the holdout digests are not those written for this freeze")
    return m


def days_holdout(out, binary, digests_path, identify_path, pilot_jsonl, settings=SETTINGS):
    """The holdout days of the frozen points only; digests to commit before the holdout."""
    out = pathlib.Path(out)
    budget = Budget(out, pilot_jsonl)
    start = time.perf_counter()
    manifest = json.loads((out / "frozen.json").read_text())
    if sha256(binary) != manifest["generator_sha256"]:
        raise SystemExit("the generator is not the frozen one")
    selected = manifest["selected"]
    report = json.loads(pathlib.Path(identify_path).read_text())
    causal, digests = causal_section(), {}
    for setting, impact in settings.items():
        if report[setting]["void"]:
            continue
        picks = [v["point"] for k, v in selected.items() if k.startswith(f"{setting}/")]
        wanted = {point_id(p): p for p in picks}
        for point in (grid.points(impact)[0], grid.fixed_stress(impact), grid.impact_only(impact)):
            wanted[point_id(point)] = point
        for v in [x for k, x in manifest["selected_secondary"].items() if k.startswith(f"{setting}/")]:
            wanted[point_id(v["point"])] = v["point"]
        spectral = report[setting]["spectral_removed"]
        for pid, point in wanted.items():
            if pid not in spectral:
                digests.update(generate(causal, out, binary, "holdout", setting, point, HOLDOUT_DAYS))
    pathlib.Path(digests_path).write_text(json.dumps({"digests": digests}, indent=0))
    (out / "holdout_days.json").write_text(json.dumps(
        {"frozen_sha256": sha256(out / "frozen.json"), "digests_sha256": sha256(digests_path)}))
    budget.charge("days-holdout", (time.perf_counter() - start) / 3600)


def write_rows(path, rows):
    pathlib.Path(path).write_text(json.dumps(
        [{"key": job_key(j), "condition": j["condition"], "policy": j["policy"], "point": j["point"],
          "day": j["day"], "result": r} for j, r in rows]))


def holdout_step(out, causal_out, selection_digests, holdout_digests, identify_path, pilot_dir, frozen_sha,
                 marker_dir, processes, recover=None):
    out = pathlib.Path(out)
    manifest = verify_manifest(out, HERE / "design.json", selection_digests, identify_path, holdout_digests, frozen_sha)
    journal = out / "holdout.jsonl"
    if journal.exists() and not recover:
        raise SystemExit("holdout.jsonl exists; only a recovery may resume it")
    # Everything that needs no holdout row fails here, before the single attempt is claimed.
    removed = removed_points(identify_path)
    jobs = holdout_jobs(manifest["selected"], removed=removed)
    ctx = context(out, causal_out, holdout_digests)
    marker = claim(marker_dir, manifest["design_sha256"], journal if recover else None, recover)
    with marker.open("a") as handle:
        handle.write(json.dumps({"kind": "holdout_digests", "sha256": sha256(holdout_digests)}) + "\n")
    budget = Budget(out, pathlib.Path(pilot_dir) / "pilot.jsonl")
    rows = run_pool("holdout", jobs, journal, processes, ctx, budget,
                    worst_seconds_of(pilot_dir))
    verify_prefix(marker, journal)
    write_rows(out / "holdout.json", rows)
    # The secondary runs after the primary is written, so a stop cannot cost the primary.
    secondary = secondary_jobs(manifest["selected_secondary"], removed, manifest["selected"])
    try:
        rows = run_pool("holdout-secondary", secondary, out / "holdout_secondary.jsonl", processes, ctx, budget,
                        worst_seconds_of(pilot_dir))
    except (Exception, SystemExit) as error:  # the primary stays valid; the secondary's failure is reported
        (out / "holdout_secondary_failed.json").write_text(json.dumps({"error": repr(error)}))
        return
    write_rows(out / "holdout_secondary.json", rows)


def main(argv):
    step, rest = argv[0], argv[1:]
    if step == "pilot":
        pilot_step(rest)
    elif step == "days-selection":
        days_selection(*rest)
    elif step == "identify":
        identify(*rest)
    elif step == "selection":
        selection_step(*rest[:5], int(rest[5]))
    elif step == "selection-secondary":
        secondary_selection_step(*rest[:5], int(rest[5]))
    elif step == "freeze":
        print(freeze(rest[0], HERE / "design.json", rest[1], rest[2], rest[5:], rest[3], rest[4]))
    elif step == "days-holdout":
        days_holdout(*rest)
    elif step == "holdout":
        holdout_step(*rest[:8], int(rest[8]), *rest[9:])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main(sys.argv[1:])
