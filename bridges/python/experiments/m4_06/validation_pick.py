"""The learned arm at one row that #103's validation days pick, for a study that builds on it.

Added after the freeze and before any holdout read, and outside the frozen design: #103 itself
compares every learned arm on the holdout and selects none. This reads no holdout day.

From `bridges/python`:

    python experiments/m4_06/validation_pick.py <data dir> <out dir> <commit> <frozen sha256> [<processes>]

It checks the frozen manifest, the design, the digests and every checkpoint against their
hashes. It evaluates each learned arm at one row, with its five training seeds, on the
validation days of every regime, at the frozen fee and environment seed 101. It picks #103's way:
the highest mean daily net, with a tie going to the greater name. It writes
`validation_pick.json`.

Everything here is synthetic.
"""

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# Imported by name, so that the spawned workers of `run.pool` can find `run.evaluate_job`.
sys.path.insert(0, str(HERE))
import run  # noqa: E402


def checked_frozen(out, frozen_sha256):
    """The frozen manifest, after every hash it rests on matches."""
    path = out / "frozen.json"
    if run.sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    frozen = json.loads(path.read_text())
    for key, name in (("design_sha256", "design.json"), ("digests_sha256", "digests.json")):
        if frozen[key] != run.sha256(HERE / name):
            raise ValueError(f"{name} changed after the freeze")
    for metadata in frozen["runs"]:
        zipped = pathlib.Path(metadata["path"]).with_suffix(".zip")
        if run.sha256(zipped) != metadata["checkpoint_sha256"]:
            raise ValueError(f"{zipped} is not the frozen checkpoint")
    return frozen


def one_row_runs(frozen):
    return [m for m in frozen["runs"] if m["every"] == 1]


def pick(rows):
    """The arm with the highest mean daily net over its rows; a tie goes to the greater name."""
    by_arm = run.group(rows, lambda r: r["arm"])
    means = {arm: run.mean_daily(part) for arm, part in by_arm.items()}
    best = max(means, key=lambda arm: (means[arm], arm))
    return best, {arm: str(value) for arm, value in sorted(means.items())}


def main():
    data_dir, out_dir, commit, frozen_sha256 = sys.argv[1:5]
    ctx = run.Context(data_dir, out_dir, commit)
    # Fewer processes than the design's, to run beside the holdout within the shared cap.
    parallel = int(sys.argv[5]) if len(sys.argv) > 5 else ctx.parallel
    frozen = checked_frozen(ctx.out, frozen_sha256)
    runs = one_row_runs(frozen)
    report = ctx.header("validation_pick")
    report.update({"frozen_sha256": frozen_sha256, "fee": frozen["fee"], "every": 1,
                   "note": "after the freeze and before any holdout read; outside the frozen "
                           "design, which selects no learned arm", "processes": parallel})
    started = time.perf_counter()
    jobs = ctx.learned_jobs(runs, frozen["fee"], "validation",
                            run.days_of(ctx.design, "validation"))
    rows = run.pool(jobs, run.evaluate_job, parallel, "validation pick")
    best, means = pick(rows)
    report.update({"rows": rows, "mean_daily_net": means, "selected": best,
                   "checkpoints": {f"{m['arm']}-seed{m['seed']}": m["checkpoint_sha256"]
                                   for m in runs if m["arm"] == best},
                   "seconds": round(time.perf_counter() - started, 1)})
    path = ctx.write("validation_pick", report)
    print(json.dumps({"selected": best, "mean_daily_net": means,
                      "validation_pick_sha256": run.sha256(path)}))


if __name__ == "__main__":
    main()
