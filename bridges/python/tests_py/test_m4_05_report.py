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


def test_a_series_without_an_interval_is_an_error_not_a_decision():
    short = {"insufficient": True, "days": 2, "required": 4}
    with pytest.raises(ValueError, match="no interval"):
        report.decide(short, report.Decimal(0), report.Decimal(1), 0)
    with pytest.raises(ValueError, match="no interval"):
        report.includes_zero(short)
    interval = {"mean": 1.0, "lower": 0.5, "upper": 1.5, "days": 5}
    assert report.decide(interval, report.Decimal(1), report.Decimal(1), 0) == "continue"
    assert report.decide(dict(interval, lower=-0.5), report.Decimal(1), report.Decimal(1), 0) == (
        "revise")
    assert report.decide(dict(interval, mean=-1.0, lower=-2.0), report.Decimal(1),
                         report.Decimal(1), 0) == "discard"
    assert not report.includes_zero(interval) and report.includes_zero(dict(interval, lower=0.0))


def seed_day(day, seed, net, drawdown):
    attribution = {"max_drawdown": drawdown, "spread_capture": "0", "inventory_pnl": "0",
                   "quote_fees": "0", "fills": 0, "mean_abs_position": "0",
                   "max_abs_position": "0"}
    episode = {"attribution": attribution, "steps": 10, "peak_notional": "0",
               "time_beyond_5_lots": 0.0, "refusals": 0, "residual_fallbacks": 0,
               "residual_clipped": 0, "limit_breach": False}
    return {"arm": "maskable_ppo", "candidate": f"set0-seed{seed}", "regime": "calm",
            "day": day, "daily_value": net, "episodes": [episode]}


def test_a_learned_arms_day_is_the_mean_over_its_seeds_in_the_table():
    # Every day one seed loses 100 and four gain 25: each day's mean is 0.
    rows = [seed_day(day, seed, net, dd)
            for day in range(20, 25)
            for seed, (net, dd) in enumerate(
                [("-100", "100"), ("25", "0"), ("25", "0"), ("25", "0"), ("25", "0")], 1)]
    row = report.summary(rows)
    assert (row["days"], row["seed_days"]) == (5, 25)
    assert report.Decimal(row["mean_daily_net"]) == 0
    assert report.Decimal(row["cvar_5_daily_net"]) == 0
    assert report.Decimal(row["mean_max_drawdown"]) == 20
    assert report.Decimal(row["worst_max_drawdown"]) == 20
