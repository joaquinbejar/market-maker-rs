"""The steps of #108 part 2 (part 1 of its design): selection, freeze and a single holdout.

SYNTHETIC. The job lists and guards are plain Python; `evaluate` imports #107's runner lazily, so
loading this module trains and reads nothing. The marker directory is an argument, kept outside
the output directory.
"""

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


def condition_points(setting, impact, selected):
    """unperturbed, fixed stress, the arm's own adversary point and impact only."""
    return {"unperturbed": grid.points(impact)[0], "fixed_stress": grid.fixed_stress(impact),
            "adversary": selected, "impact_only": grid.impact_only(impact)}


def policy_key(policy):
    """The seeds of a learned arm are one policy."""
    return f"{policy['setting']}/{policy['arm']}"


def holdout_jobs(selected_by_policy, settings=SETTINGS):
    jobs = []
    for s, impact in settings.items():
        for p in policies(s):
            conditions = condition_points(s, impact, selected_by_policy[policy_key(p)])
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
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            done[row["key"]] = row
    with journal.open("a") as handle:
        for job in jobs:
            key = job_key(job)
            if key not in done:
                done[key] = {"key": key, "result": evaluate(job)}
                handle.write(json.dumps(done[key]) + "\n")
                handle.flush()
    return [done[job_key(job)] for job in jobs]


def select(rows):
    """{policy_key: grid index} from selection rows: mean daily loss, seeds of an arm averaged."""
    by = {}
    for job, row in rows:
        key = policy_key(job["policy"])
        pid = point_id(job["point"])
        by.setdefault(key, {}).setdefault(pid, {}).setdefault(job["day"], []).append(-row["net"])
    out = {}
    for key, per_point in by.items():
        ids = list(per_point)
        losses = {i: [sum(v) / len(v) for v in per_point[pid].values()] for i, pid in enumerate(ids)}
        best, means = stats.pick_adversary(ids, losses)
        out[key] = {"point": ids[best], "selection_loss": means[best]}
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
    return {"net": float(days[0]["net"]), "episode": days[0]}


PILOT_SETTING = "impact_high"
PILOT_OFFSET = 500000
PILOT_DAYS = 2


def pilot_jobs():
    """2 days per arm at the unperturbed and the most active point, on seeds outside every split."""
    impact = SETTINGS[PILOT_SETTING]
    return [{"split": "pilot", "policy": p, "point": pt, "day": d}
            for pt in (grid.points(impact)[0], grid.fixed_stress(impact))
            for p in policies(PILOT_SETTING) for d in range(1, PILOT_DAYS + 1)]


def project(rows, ceiling_hours=90.0):
    """Process-hours for the whole count from the pilot's per-day seconds, the slowest point's."""
    per_arm = {}
    for job, seconds in rows:
        per_arm.setdefault(job["policy"]["arm"], {}).setdefault(point_id(job["point"]), []).append(seconds)
    worst = {arm: max(sum(v) / len(v) for v in points.values()) for arm, points in per_arm.items()}
    runs = {"no_trade": 1, "avellaneda_stoikov": 1, "glft": 1, "mlp": 5, "lstm": 5}
    per_setting_day = sum(worst[arm] * n for arm, n in runs.items())
    days = (24 * SELECTION_DAYS + 4 * HOLDOUT_DAYS) * len(SETTINGS)
    hours = per_setting_day * days / 3600
    return {"worst_seconds_per_day": worst, "projected_hours": hours, "over_ceiling": hours > ceiling_hours}
