"""The relabelling of the comparison's holdout rows (#102), on synthetic rows."""

import hashlib
import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_05_report", HERE.parent / "experiments" / "m4_05" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

DESIGN = {
    "simulator": {"regimes": [{"name": "calm"}, {"name": "thin"}]},
    "splits": {"holdout": [20, 21]},
}
RUNS = [
    {"arm": "maskable_ppo", "set": 0, "seed": 1, "variant": "main"},
    {"arm": "maskable_ppo", "set": 0, "seed": 1, "variant": "fill_immediate",
     "ablation": "fill_immediate"},
    {"arm": "maskable_ppo", "set": 0, "seed": 1, "variant": "main",
     "ablation": "reward_inventory_penalty"},
]


def row(arm, candidate, variant, regime, day):
    return {"arm": arm, "candidate": candidate, "variant": variant, "regime": regime,
            "day": day}


def written(tmp_path, rows, runs=RUNS):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"runs": runs}))
    return {"frozen_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "rows": rows}


def submitted():
    """The rows in the order the runner submits the jobs: classical first, then each run
    over the regimes and the days."""
    rows = [row("glft", "{}", "main", "calm", 20)]
    for run in RUNS:
        for regime in ("calm", "thin"):
            for day in (20, 21):
                rows.append(row(run["arm"], "set0-seed1", run["variant"], regime, day))
    return rows


def test_relabel_gives_each_learned_row_the_ablation_of_its_run(tmp_path):
    labelled = report.relabel(tmp_path, written(tmp_path, submitted()), DESIGN)
    labels = [r["ablation"] for r in labelled]
    assert labels == [None] + [None] * 4 + ["fill_immediate"] * 4 + [
        "reward_inventory_penalty"] * 4
    # The main and the penalty-trained runs differ only by order: the main tables keep 4 rows.
    main = [r for r in labelled if r["ablation"] is None and r["variant"] == "main"]
    assert sum(r["arm"] == "maskable_ppo" for r in main) == 4


def test_relabel_refuses_rows_out_of_submission_order(tmp_path):
    rows = submitted()
    rows[5], rows[6] = rows[6], rows[5]
    with pytest.raises(ValueError, match="not the submitted job"):
        report.relabel(tmp_path, written(tmp_path, rows), DESIGN)


def test_relabel_refuses_another_manifest_a_split_block_and_a_wrong_label(tmp_path):
    holdout = written(tmp_path, submitted())
    holdout["frozen_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the manifest"):
        report.relabel(tmp_path, holdout, DESIGN)
    rows = submitted()
    rows.insert(3, row("glft", "{}", "main", "thin", 20))
    with pytest.raises(ValueError, match="one block"):
        report.relabel(tmp_path, written(tmp_path, rows), DESIGN)
    rows = submitted()
    rows[0]["ablation"] = "fill_immediate"
    with pytest.raises(ValueError, match="classical row"):
        report.relabel(tmp_path, written(tmp_path, rows), DESIGN)
    rows = submitted()
    rows[-1]["ablation"] = None
    with pytest.raises(ValueError, match="is labelled None"):
        report.relabel(tmp_path, written(tmp_path, rows), DESIGN)
