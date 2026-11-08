"""Workers for #281's runner tests that must run in a real spawned process pool."""

import time


def sleeping(job):
    """A training run that outlasts any test's limit."""
    time.sleep(30)
    return {}
