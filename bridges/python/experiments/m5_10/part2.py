"""Part 2 of #108: #281's declared and unrun impact condition, once. SYNTHETIC.

    python experiments/m5_10/part2.py <#102 run dir> <#281 out dir> <part 2 out dir> <marker dir> <processes>

#281's frozen A and B (10 models) at kappa = 2 kappa0 per regime, through #282's reactive runner
(on main since e88bdf6), on #281's 30 holdout days. Those base days were already read for
#281's other 13 conditions, so this is a one-off added condition with no bearing on #281's
decision; it is reported apart. Estimands: #281's D_tail (CVaR5 loss, seed-averaged and per
seed) and D_net at impact, A - B, with #281's stratified stationary bootstrap. No primary claim.
"""

import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RUNNER_MERGE_COMMIT = "e88bdf6b7d021f7b6db7e016bc682fb11347895f"
BUDGET_LEFT_HOURS = 90.0 - 47.7  # part 1's ledger at its end
PART2_CEILING_HOURS = 6.0


def load_runner():
    """#281's runner and report, imported by their own names from their directory, so the
    pickled jobs resolve in spawn workers (which inherit sys.path). Loading by path under a
    new name broke attempt 1: the workers could not import that name."""
    directory = str(HERE.parent / "m5_05")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import report  # noqa: E402
    import run  # noqa: E402

    for module in (run, report):
        if pathlib.Path(module.__file__).resolve().parent != HERE.parent / "m5_05":
            raise SystemExit(f"another study's {module.__name__} module shadows #281's")
    return run, report


def check_runner_commit(repo):
    """#282's runner here is the one merged at RUNNER_MERGE_COMMIT, an ancestor of HEAD."""
    import subprocess

    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", RUNNER_MERGE_COMMIT, "HEAD"], check=True)
    blob = subprocess.run(["git", "-C", str(repo), "show", f"{RUNNER_MERGE_COMMIT}:bridges/python/experiments/m5_04/run.py"],
                          check=True, capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != sha256(HERE.parent / "m5_04" / "run.py"):
        raise SystemExit("m5_04/run.py is not the one merged at the runner's commit")


ATTEMPT_1 = {"attempt": 1, "pid": 86547, "head": "7ff4aa15e7bdb5a759643e576075f5d939015c38",
             "cause": "every spawn worker failed with ModuleNotFoundError: No module named 'm5_05_run'",
             "rows": 0, "days_read": 0}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def marker_path(marker_dir, attempt=2):
    suffix = "" if attempt == 1 else f"-attempt{attempt}"
    return pathlib.Path(marker_dir) / f"part2-{sha256(HERE / 'design.json')}{suffix}.jsonl"


def claim(marker_dir, manifest):
    """One attempt: an O_EXCL marker before any day is read; a failure is never rerun."""
    if not marker_path(marker_dir, 1).exists():
        raise SystemExit("attempt 1's marker is missing; this run is attempt 2 only")
    path = marker_path(marker_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit("part 2's last attempt was made already; it is not rerun") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"kind": "previous", **ATTEMPT_1}) + "\n")
        handle.write(json.dumps({"kind": "attempt", "attempt": 2, "at": time.time(), **manifest}) + "\n")
    return path


def jobs(m505, ctx, out281, models, kappa):
    return m505.policy_jobs(ctx, out281, "impact", "holdout", m505.unperturbed_name(), arms=m505.TRAINED,
                            frozen_models=models, extra=lambda regime: {"impact": True, "kappa": kappa[regime]})


def limit_seconds(processes):
    return min(PART2_CEILING_HOURS, BUDGET_LEFT_HOURS) * 3600 / processes


def estimands(report, rows):
    """D_tail and D_net at impact, A - B, with #281's intervals."""
    indices = report.stratified_indices(report.DAYS)
    a, b = report.matrix(rows, "adversarial", "impact"), report.matrix(rows, "baseline", "impact")
    return {s: report.difference(s, a, b, indices) for s in ("cvar_averaged", "cvar_per_seed", "mean_net")}


def main(run_dir, out281, out, marker_dir, processes):
    check_runner_commit(HERE)
    m505, report = load_runner()
    out, out281 = pathlib.Path(out), pathlib.Path(out281)
    out.mkdir(parents=True, exist_ok=True)
    ctx = m505.Context(run_dir)
    ctx.parallel = int(processes)
    provenance = m505.impact_provenance(ctx)
    if provenance is None:
        raise SystemExit("#282's runner is not here")
    models = m505.check_models(ctx, out281)
    manifest = {"runner_merge_commit": RUNNER_MERGE_COMMIT, "impact": provenance,
                "frozen_models_sha256": sha256(m505.FROZEN_MODELS), "design_sha256": sha256(HERE / "design.json"),
                "code_sha256": {p.name: sha256(p) for p in (pathlib.Path(__file__), HERE.parent / "m5_05" / "run.py")}}
    todo = jobs(m505, ctx, out281, models, provenance["kappa"])
    claim(marker_dir, manifest)
    started = time.perf_counter()
    try:
        rows = m505.pool(todo, m505.run_policy, ctx.parallel, "part2", out / "part2.journal",
                         limit_seconds(ctx.parallel))
    finally:
        (out / "part2_charge.json").write_text(json.dumps(
            {"process_hours": (time.perf_counter() - started) * ctx.parallel / 3600}))
    result = {"label": "one-off added condition; its base days were read for #281's other 13 conditions; "
                       "no bearing on #281's decision; 14 conditions in all, unadjusted",
              "manifest": manifest, "estimands": estimands(report, rows), "rows": rows}
    (out / "part2.json").write_text(json.dumps(result, indent=1, default=str))


def selftest():
    """Run as a script with --selftest: a function only #281's runner has, pickled to a real
    spawn worker, resolves there to #281's runner."""
    import concurrent.futures
    import multiprocessing

    run, _ = load_runner()
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        name = pool.submit(run.unperturbed_name).result()
    print(json.dumps({"worker_unperturbed_name": name, "parent": run.unperturbed_name()}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(*sys.argv[1:])
