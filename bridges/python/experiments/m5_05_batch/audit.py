"""A diagnostic after #289's holdout read: the frozen learned arms on the validation days, with
the position read at every clearing.

    python experiments/m5_05_batch/audit.py <out dir> <commit> <parallel> <frozen sha256>

It reads no holdout day, decides nothing and changes no reported result. It checks the frozen
manifest and every checkpoint against it, runs each learned run on validation days 15 to 19 at
each latency and the chosen fee, and writes `audit.json`: per arm and latency, the days whose
position went past its limit at a clearing, and the largest position.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402


def summarise(rows):
    """Per arm and latency: the runs' days, the breach days and the largest position."""
    out = {}
    for row in rows:
        key = f"{row['arm']}/{row['latency']}ms"
        entry = out.setdefault(key, {"days": 0, "breach_days": 0, "peak_position": 0})
        entry["days"] += 1
        entry["breach_days"] += int(bool(row["breach"]))
        entry["peak_position"] = max(entry["peak_position"], int(row["peak_position"]))
    return dict(sorted(out.items()))


def audit_jobs(ctx, runs, frozen):
    jobs = []
    coefficients = ctx.read("predictor")["coefficients"]
    for meta in runs:
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        entry = frozen["runs"][path.stem]
        if entry["checkpoint"] != run.sha256(path):
            raise SystemExit(f"{path.stem} is not the frozen checkpoint")
        for day in run.days_of(ctx.design, "validation"):
            jobs.append(ctx.job(meta["arm"], day, meta["latency"], ctx.fee(), f"seed{meta['seed']}",
                                {"kind": "frozen", "metadata": meta}, coefficients))
    return jobs


def main():
    out, commit, parallel, frozen_sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    ctx = run.Context(out, commit, parallel)
    if run.sha256(ctx.out / "frozen.json") != frozen_sha:
        raise SystemExit("the frozen manifest differs")
    frozen = json.loads((ctx.out / "frozen.json").read_text())
    jobs = audit_jobs(ctx, ctx.runs(), frozen)
    rows = run.pool(jobs, run.evaluate_job, ctx.parallel, "audit")
    if any(row["split"] != "validation" for row in rows):
        raise SystemExit("the audit read a day outside validation")
    report = {**ctx.header("audit"), "frozen_sha256": frozen_sha, "split": "validation",
              "fee": ctx.fee(), "diagnostic": "decides nothing; reads no holdout day",
              "by_arm": summarise(rows), "rows": rows}
    print(json.dumps(report["by_arm"]), ctx.write("audit", report))


if __name__ == "__main__":
    main()
