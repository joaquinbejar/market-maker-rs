"""The one resume of #281's holdout after an attempt killed from outside (design.json, changes).

From `bridges/python`, detached from any terminal session:

    nohup setsid python -I experiments/m5_05/resume.py <m4_05 run dir> <out dir> <commit> \
        <frozen sha256> > <out dir>/resume.stdout 2> <out dir>/resume.stderr &

`run.py` is left as frozen: this file imports it and reuses its checks, its job list and its
pool. The resume runs only when the first attempt's marker exists, its report does not, and no
resume was made; it records its own start with O_EXCL, naming the first attempt, before any
job runs. It drops a torn last line of the journal, then runs only the jobs the journal does
not hold. A second death stops the study, reported: there is no further resume.
"""

import hashlib
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# Importable by name, so the pool's spawned workers find run_policy.
sys.path.insert(0, str(HERE))
import run  # noqa: E402


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def repair_journal(path):
    """Drops a torn last line (a write cut by the kill); every other line must parse. Returns
    the number of complete lines and whether a torn one was dropped."""
    path = pathlib.Path(path)
    text = path.read_text()
    lines = text.split("\n")
    complete = [line for line in lines[:-1]]
    tail = lines[-1]
    for line in complete:
        json.loads(json.loads(line)["key"])
    torn = False
    if tail:
        try:
            json.loads(json.loads(tail)["key"])
            complete.append(tail)
        except (ValueError, KeyError):
            torn = True
    if torn or not text.endswith("\n") and text:
        path.write_text("".join(line + "\n" for line in complete))
    return len(complete), torn


def environment():
    """What the run executes beside run.py: the bridge's native extension, the bridge's
    Cargo.lock and the Python packages, by hash, with the extension's build time."""
    import mm_rl_bridge._native as native

    extension = pathlib.Path(native.__file__)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=False).stdout
    if not freeze.strip():
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable],
                                capture_output=True, text=True, check=False).stdout
    lock = HERE.parent.parent / "Cargo.lock"
    return {"extension_sha256": sha256(extension), "extension_mtime": extension.stat().st_mtime,
            "cargo_lock_sha256": sha256(lock) if lock.exists() else None,
            "packages_sha256": hashlib.sha256(freeze.encode()).hexdigest()}


def prefix_sha256(path, nbytes):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read(nbytes)).hexdigest()


def first_attempt_seconds(out):
    """The first attempt's running time: its marker's start to its journal's last write."""
    started = json.loads((out / "holdout.started").read_text())["started"]
    return max(0.0, (out / "holdout.journal").stat().st_mtime - started)


def resume(ctx, out, commit, frozen_sha256):
    out = pathlib.Path(out)
    if not (out / "holdout.started").exists():
        raise ValueError("no first attempt to resume")
    if (out / "holdout.json").exists():
        raise ValueError("the holdout finished: nothing to resume")
    path = out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    first = json.loads((out / "holdout.started").read_text())
    if first["frozen_sha256"] != frozen_sha256:
        raise ValueError("the first attempt ran from another manifest")
    frozen = json.loads(path.read_text())
    here = ctx.header("holdout", commit)
    for key in ("design_sha256", "code_sha256"):
        if frozen[key] != here[key]:
            raise ValueError(f"{key} changed after the freeze")
    if frozen["selection_sha256"] != sha256(out / "selection.json"):
        raise ValueError("selection.json changed after the freeze")
    if frozen["frozen_models_sha256"] != sha256(run.FROZEN_MODELS):
        raise ValueError("frozen_models.json changed after the freeze")
    run.m503.check_frozen_digests(frozen, run.DIGESTS)
    if run.impact_provenance(ctx) != frozen["impact"]:
        raise ValueError("#282's runner changed after the freeze")
    models = run.check_models(ctx, out)
    jobs = run.holdout_jobs(ctx, out, frozen, models)
    # The overrun rule spans both attempts.
    limit = run.limit_seconds(out, "holdout") - first_attempt_seconds(out)
    try:
        marker = os.open(out / "holdout.resumed", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("the holdout was resumed already: no further resume") from error
    complete, torn = repair_journal(out / "holdout.journal")
    journal = out / "holdout.journal"
    before = {"lines": complete, "bytes": journal.stat().st_size,
              "sha256": sha256(journal)}
    with os.fdopen(marker, "w") as handle:
        handle.write(json.dumps({"started": time.time(), "pid": os.getpid(),
                                 "frozen_sha256": frozen_sha256, "first_attempt": first,
                                 "resume_sha256": sha256(__file__),
                                 "run_sha256": sha256(HERE / "run.py"),
                                 "journal_before": before, "torn_line_dropped": torn,
                                 "environment": environment(),
                                 "limit_seconds": limit}) + "\n")
    rows = run.pool(jobs, run.run_policy, ctx.parallel, "holdout resume", journal, limit)
    # The first attempt's lines are untouched: the journal only grew after them.
    if prefix_sha256(journal, before["bytes"]) != before["sha256"]:
        raise ValueError("the first attempt's journal lines changed during the resume")
    here["frozen_sha256"] = frozen_sha256
    here["chosen"] = frozen["chosen"]
    here["impact_run"] = frozen["impact"] is not None
    here["resumed"] = {"journal_before": before, "torn_line_dropped": torn,
                       "first_lines_unchanged": True, "resume_sha256": sha256(__file__),
                       "environment": environment()}
    here["rows"] = rows
    ctx.write(out, "holdout", here)
    return here


def main():
    run_dir, out_dir, commit, frozen_sha256 = sys.argv[1:5]
    started = time.perf_counter()
    report = resume(run.Context(run_dir), out_dir, commit, frozen_sha256)
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps({"step": "resume", "seconds": round(time.perf_counter() - started, 1),
                      **summary}, indent=2, default=str))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
