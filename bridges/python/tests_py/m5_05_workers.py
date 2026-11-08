"""Workers for #281's runner tests that must run in a real spawned process pool."""

import time


def sleeping(job):
    """A training run that outlasts any test's limit."""
    time.sleep(30)
    return {}


def deterministic(job):
    """A holdout job whose row depends on the job alone, slowly enough to be cut mid-run."""
    time.sleep(0.05)
    net = (job["seed"] * 31 + job["day"] * 7) % 11 - 5
    return {"condition": job["condition"], "arm": job["arm"], "seed": job["seed"],
            "regime": job["regime"], "day": job["day"], "net": str(net)}
