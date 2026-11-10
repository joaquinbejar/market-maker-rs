"""Part 2 of #108: #281's declared and unrun impact condition, once. SYNTHETIC.

    python experiments/m5_10/part2.py <#102 run dir> <#281 out dir> <part 2 out dir> <marker dir> <processes>

#281's frozen A and B (10 models) at kappa = 2 kappa0 per regime, through #282's reactive runner
(on main since e88bdf6), on #281's 30 holdout days. Those base days were already read for
#281's other 13 conditions, so this is a one-off added condition with no bearing on #281's
decision; it is reported apart. Estimands: #281's D_tail (CVaR5 loss, seed-averaged and per
seed) and D_net at impact, A - B, with #281's stratified stationary bootstrap. No primary claim.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RUNNER_MERGE_COMMIT = "e88bdf6b7d021f7b6db7e016bc682fb11347895f"
BUDGET_LEFT_HOURS = 90.0 - 47.7  # part 1's ledger at its end
PART2_CEILING_HOURS = 6.0


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def marker_path(marker_dir):
    return pathlib.Path(marker_dir) / f"part2-{sha256(HERE / 'design.json')}.jsonl"


def claim(marker_dir, manifest):
    """One attempt: an O_EXCL marker before any day is read; a failure is never rerun."""
    path = marker_path(marker_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit("part 2 was attempted already; it is not rerun") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"kind": "attempt", "at": time.time(), **manifest}) + "\n")
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
    m505 = _module("m5_05_run", HERE.parent / "m5_05" / "run.py")
    report = _module("m5_05_report", HERE.parent / "m5_05" / "report.py")
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


if __name__ == "__main__":
    main(*sys.argv[1:])
