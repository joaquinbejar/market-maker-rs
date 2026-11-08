"""A single evaluation attempt per freeze, for part A's evaluation and parts B and C's holdout.

Before any evaluation is dispatched, `claim` creates an attempt marker next to the frozen manifest
with `O_EXCL`, so a second attempt, after a crash or a failure before the result is written,
finds it and stops. The explicit recovery policy: a retry runs only with a stated reason, which
is appended to the marker and carried in the result, so a reader sees every attempt.

The guard was added after #107's evaluation and holdout had been read; it protects later runs.
"""

import json
import os
import pathlib
import time


def _record(kind, frozen_sha, **extra):
    return {"kind": kind, "frozen_sha256": frozen_sha,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(), **extra}


def claim(marker, frozen_sha, recover=None):
    """Claims the evaluation for this freeze and returns every attempt recorded, this one last.

    Raises SystemExit when an earlier attempt left the marker and no recovery reason is given.
    """
    marker = pathlib.Path(marker)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not recover:
            raise SystemExit(f"an evaluation attempt already started ({marker}); a retry needs "
                             "an explicit recovery reason") from None
        with marker.open("a") as handle:
            handle.write(json.dumps(_record("recovery", frozen_sha, reason=recover)) + "\n")
    else:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(_record("attempt", frozen_sha)) + "\n")
    return [json.loads(line) for line in marker.read_text().splitlines() if line.strip()]


def recovery_reason(argv):
    """The reason after `--recover` in the arguments, or None."""
    if "--recover" not in argv:
        return None
    index = argv.index("--recover")
    if index + 1 >= len(argv) or not argv[index + 1].strip():
        raise SystemExit("--recover needs a reason")
    return argv[index + 1]
