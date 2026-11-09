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


def claim(marker_dir, frozen_sha, journal=None, recover=None, journal_prefix_sha=None):
    """One attempt per freeze; a recovery only after an external kill, with the prefix checked."""
    marker = pathlib.Path(marker_dir) / f"holdout-{frozen_sha}.jsonl"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not recover:
            raise SystemExit("a holdout attempt already started; a retry needs a recovery reason") from None
        lines = [json.loads(x) for x in marker.read_text().splitlines()]
        if sum(x["kind"] == "recovery" for x in lines) >= MAX_RECOVERIES:
            raise SystemExit("too many recoveries")
        if journal is None or journal_prefix_sha is None or sha256(journal) != journal_prefix_sha:
            raise SystemExit("the journal prefix does not match its recorded hash")
        with marker.open("a") as handle:
            handle.write(json.dumps({"kind": "recovery", "reason": recover}) + "\n")
    else:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"kind": "attempt", "frozen_sha256": frozen_sha}) + "\n")
    return marker


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
        raise SystemExit(f"{path} is not the committed day")
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
        raise SystemExit("the design evaluates one seed per day")
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


def spectral_check(weights, points):
    """The design's plausibility check, before any day is generated: stops if a point fails."""
    radii = {}
    for point in points:
        radii[point_id(point)] = grid.spectral_radius(grid.branching_matrix(weights, point["branching"]))
        if not grid.plausible(weights, point):
            raise SystemExit(f"{point_id(point)} is not subcritical")
    return radii


def generate_pilot_days(causal, data_dir, binary):
    """Writes the pilot's days with examples/hawkes_days and returns their sha256 by relative path."""
    import subprocess
    import time

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
    import time

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
    paths = [day_path(out, split, setting, point, d) for d in range(1, days + 1)]
    return {str(p.relative_to(out)): sha256(p) for p in paths}


def days_selection(out, binary, digests_path, settings=SETTINGS):
    """Every point's selection days, after the spectral check of every point; digests to commit."""
    causal = causal_section()
    radii, digests = {}, {}
    for setting, impact in settings.items():
        radii[setting] = spectral_check(causal["flow"]["weights"], grid.points(impact))
        for point in grid.points(impact):
            digests.update(generate(causal, out, binary, "selection", setting, point, SELECTION_DAYS))
    pathlib.Path(digests_path).write_text(json.dumps({"digests": digests, "spectral_radius": radii}, indent=0))


def point_statistics(out, split, setting, point, days):
    """Queue shares, market rate and impact slope of one point's days, with no policy."""
    import numpy as np

    below, queue, open_, rate, changes, signed = [], [], [], [], [], []
    for day in range(1, days + 1):
        base = day_path(out, split, setting, point, day)
        queues = np.asarray(json.loads(base.with_name(f"day-{day}.queues.json").read_text()), float)
        flow = np.asarray(json.loads(base.with_name(f"day-{day}.flow.json").read_text()), float)
        rows = np.asarray(json.loads(base.read_text()), float)
        below.append(float((queues < QUEUE_CAP).mean()))
        queue.append(float(queues.mean()))
        open_.append(float((queues > 0).mean()))
        rate.append(float(flow.sum(axis=1).mean() / 2))
        # Row r's orders move the mid of row r + 1; mids are in ticks, the bid being floor(mid).
        mids = rows[:, 1]
        changes.append(mids[1:] - mids[:-1])
        signed.append(flow[:-1, 0] - flow[:-1, 1])
    return {"shares": {"below_cap": float(np.mean(below)), "mean_queue": float(np.mean(queue)),
                       "non_empty": float(np.mean(open_))},
            "cap_share": 1 - float(np.mean(below)), "empty_share": 1 - float(np.mean(open_)),
            "market_rate": float(np.mean(rate)),
            "slope": grid.slope(np.concatenate(changes), np.concatenate(signed))}


QUEUE_CAP = 20


def identify(out, report_path, settings=SETTINGS):
    """The identifiability rules on the selection days; failing points go in `removed`."""
    causal = causal_section()
    rates, weights = causal["flow"]["rates"], causal["flow"]["weights"]
    report = {}
    for setting, impact in settings.items():
        points = grid.points(impact)
        base_stats = point_statistics(out, "selection", setting, points[0], SELECTION_DAYS)
        base_formula = grid.rate_formula(rates, weights, points[0], base_stats["shares"])
        rows, removed = {}, []
        for point in points:
            st = point_statistics(out, "selection", setting, point, SELECTION_DAYS)
            formula = grid.rate_formula(rates, weights, point, st["shares"])
            rate_ok = grid.rate_passes(st["market_rate"], base_stats["market_rate"], formula, base_formula)
            impact_ok = grid.impact_passes(st["slope"], point["impact"])
            rows[point_id(point)] = {**st, "formula": formula, "rate_ok": rate_ok, "impact_ok": impact_ok}
            if not (rate_ok and impact_ok):
                removed.append(point_id(point))
        report[setting] = {"points": rows, "removed": removed}
    pathlib.Path(report_path).write_text(json.dumps(report, indent=1))


def _worker(args):
    job, ctx = args
    return job, evaluate(job, ctx["out"], ctx["checkpoints"], ctx["classical"], ctx["digests"])


def run_pool(jobs, journal, processes, ctx):
    """As run_journaled, with a pool; only the parent writes the journal."""
    import multiprocessing

    journal = pathlib.Path(journal)
    done = load_journal(journal)
    todo = [j for j in jobs if job_key(j) not in done]
    with multiprocessing.get_context("spawn").Pool(processes) as pool, journal.open("a") as handle:
        for job, result in pool.imap_unordered(_worker, [(j, ctx) for j in todo]):
            row = {"key": job_key(job), "result": result}
            done[row["key"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
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


def context(out, causal_out, digests_path):
    causal_out = pathlib.Path(causal_out)
    frozen = json.loads((causal_out / "frozen.json").read_text())
    return {"out": str(out), "checkpoints": str(causal_out / "checkpoints"),
            "classical": check_classical(causal_out / "classical.json", frozen["classical_sha256"]),
            "digests": json.loads(pathlib.Path(digests_path).read_text())["digests"]}


def removed_points(identify_path):
    report = json.loads(pathlib.Path(identify_path).read_text())
    return {s: set(v["removed"]) for s, v in report.items()}


def selection_step(out, causal_out, digests_path, identify_path, processes):
    removed = removed_points(identify_path)
    jobs = [j for j in selection_jobs() if point_id(j["point"]) not in removed[j["policy"]["setting"]]]
    rows = run_pool(jobs, pathlib.Path(out) / "selection.jsonl", processes, context(out, causal_out, digests_path))
    pathlib.Path(out, "selection.json").write_text(json.dumps(select(rows), indent=1))


def freeze(out, design_path, digests_path, identify_path, pilot_paths):
    """The manifest the holdout runs from: every input by hash, and each policy's pick."""
    out = pathlib.Path(out)
    selected = json.loads((out / "selection.json").read_text())
    manifest = {"design_sha256": sha256(design_path), "selection_digests_sha256": sha256(digests_path),
                "identify_sha256": sha256(identify_path), "selection_sha256": sha256(out / "selection.json"),
                "pilot_sha256": {str(p): sha256(p) for p in pilot_paths}, "selected": selected}
    path = out / "frozen.json"
    if path.exists():
        raise SystemExit("already frozen")
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return sha256(path)


def days_holdout(out, binary, digests_path, identify_path, settings=SETTINGS):
    """The holdout days of the frozen points only; digests to commit before the holdout."""
    out = pathlib.Path(out)
    selected = json.loads((out / "frozen.json").read_text())["selected"]
    removed = removed_points(identify_path)
    causal, digests = causal_section(), {}
    for setting, impact in settings.items():
        picks = [v["point"] for k, v in selected.items() if k.startswith(f"{setting}/")]
        wanted = {point_id(p): p for p in picks}
        for point in (grid.points(impact)[0], grid.fixed_stress(impact), grid.impact_only(impact)):
            wanted[point_id(point)] = point
        for pid, point in wanted.items():
            if pid not in removed[setting]:
                digests.update(generate(causal, out, binary, "holdout", setting, point, HOLDOUT_DAYS))
    pathlib.Path(digests_path).write_text(json.dumps({"digests": digests}, indent=0))


def holdout_step(out, causal_out, digests_path, identify_path, frozen_sha, marker_dir, processes, recover=None,
                 journal_prefix_sha=None):
    out = pathlib.Path(out)
    if sha256(out / "frozen.json") != frozen_sha:
        raise SystemExit("frozen.json is not the freeze named")
    journal = out / "holdout.jsonl"
    claim(marker_dir, frozen_sha, journal if recover else None, recover, journal_prefix_sha)
    selected = json.loads((out / "frozen.json").read_text())["selected"]
    removed = removed_points(identify_path)
    jobs = holdout_jobs(selected, removed=removed)
    rows = run_pool(jobs, journal, processes, context(out, causal_out, digests_path))
    (out / "holdout.json").write_text(json.dumps(
        [{"key": job_key(j), "condition": j["condition"], "policy": j["policy"], "point": j["point"],
          "day": j["day"], "result": r} for j, r in rows]))


def main(argv):
    step, rest = argv[0], argv[1:]
    if step == "pilot":
        pilot_step(rest)
    elif step == "days-selection":
        days_selection(*rest)
    elif step == "identify":
        identify(*rest)
    elif step == "selection":
        selection_step(*rest[:4], int(rest[4]))
    elif step == "freeze":
        print(freeze(rest[0], HERE / "design.json", rest[1], rest[2], rest[3:]))
    elif step == "days-holdout":
        days_holdout(*rest)
    elif step == "holdout":
        holdout_step(*rest[:5], rest[5], int(rest[6]), *(rest[7:] or []))
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main(sys.argv[1:])
