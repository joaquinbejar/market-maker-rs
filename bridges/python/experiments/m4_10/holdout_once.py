"""Runs #110's holdout once, from `bridges/python`:

    python experiments/m4_10/holdout_once.py <data dir> <out dir> <commit> <frozen sha256> [<processes>]

`run.py holdout` refuses only once `holdout.json` exists, which it writes after every job, so a
failed or concurrent attempt could read holdout days again. This wrapper, added after the
freeze and leaving `run.py` as frozen, creates `<out dir>/holdout.started` exclusively
(`O_EXCL`) before any job, keeps it if the holdout fails, and refuses to start while it or
`holdout.json` exists: a failed attempt is never run again by script, and a recovery is by
hand, recorded and declared.
"""

import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# Imported by name, so that the spawned workers of the pool can find `run.evaluate_job`.
sys.path.insert(0, str(HERE))
import run  # noqa: E402

MARKER = "holdout.started"


def once(out, frozen_sha256, commit, start):
    """Calls `start` after creating the marker; refuses while the marker or a result exists."""
    out = pathlib.Path(out)
    if (out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    if (out / MARKER).exists():
        raise ValueError("the holdout was started already; a failed or concurrent attempt "
                         "is not run again, and recovery is by hand")
    descriptor = os.open(out / MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as marker:
        marker.write(json.dumps({"commit": commit, "frozen_sha256": frozen_sha256,
                                 "at": time.time()}))
    return start()


def main():
    data_dir, out_dir, commit, frozen_sha256 = sys.argv[1:5]
    ctx = run.Context(data_dir, out_dir, commit)
    if len(sys.argv) > 5:
        ctx.parallel = int(sys.argv[5])
    report = once(ctx.out, frozen_sha256, commit, lambda: run.holdout(ctx, frozen_sha256))
    print(json.dumps({"step": "holdout", "seconds": report.get("seconds"),
                      "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
