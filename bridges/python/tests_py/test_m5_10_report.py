"""The report of #108 part 2 on toy rows."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from m5_10_load import module  # noqa: E402

rep = module("report")


def row(arm, condition, day, value, seed=None):
    episode = {"attribution": {"max_abs_position": str(seed or 3), "mean_abs_position": "1", "max_drawdown": "2",
                               "spread_capture": "1", "inventory_pnl": "0"},
               "peak_notional": "100", "limit_breach": False, "refusals": 1}
    return {"policy": {"setting": "s", "arm": arm, "seed": seed}, "condition": condition, "day": day,
            "result": {"net": value, "episode": episode}}


def rows():
    out = []
    for d in range(1, 31):
        out += [row("no_trade", c, d, 0.0) for c in ("unperturbed", "adversary")]
        out += [row("mlp", "unperturbed", d, 1.0, s) for s in (1, 2)]
        out += [row("mlp", "adversary", d, v, s) for s, v in ((1, -1.0), (2, -3.0))]
    return out


def selected():
    return {"s/mlp": {"selection_loss": 1.5}, "s/no_trade": {"selection_loss": 0.0}}


def test_seeds_average_and_the_stress_reads_below_zero():
    table = {(r["arm"], r["condition"]): r for r in rep.tables(rows(), selected())}
    adv = table[("mlp", "adversary")]
    assert adv["mean_net"] == -2.0
    assert adv["against_unperturbed"]["interval"] == [-3.0, -3.0]
    assert adv["against_unperturbed"]["reading"] == "below 0"
    assert adv["against_no_trade"]["reading"] == "below 0"


def test_requirement_selection_and_holdout_loss():
    adv = {(r["arm"], r["condition"]): r for r in rep.tables(rows(), selected())}[("mlp", "adversary")]
    assert adv["margin_requirement_max"] == pytest.approx(5.0)
    assert adv["selection_loss"] == 1.5 and adv["holdout_mean_loss"] == 2.0
    assert adv["limit_breaches"] == 0


def test_notes_state_the_limits():
    text = " ".join(rep.NOTES)
    assert "SYNTHETIC" in text and "GLFT replaces" in text and "not measured" in text


def test_series_is_ordered_by_day():
    by_day = {2: [{"v": 2.0}], 1: [{"v": 1.0}]}
    assert np.array_equal(rep.series(by_day, lambda r: r["v"]), [1.0, 2.0])


def test_largest_is_over_seeds_and_not_averaged():
    table = {(r["arm"], r["condition"]): r for r in rep.tables(rows(), selected())}
    assert table[("mlp", "adversary")]["max_abs_position"] == 2.0
    assert table[("mlp", "adversary")]["refusals"] == 60


def test_not_run_names_conditions():
    fixed = rep.point_id(rep.grid.fixed_stress(0.0))
    identify = {"impact_0": {"void": False, "removed": [fixed], "spectral_removed": [],
                             "points": {"y": {"empty_share": 0.1, "cap_share": 0.2}}},
                "impact_low": {"void": True, "removed": ["u"], "spectral_removed": [], "points": {}}}
    out = rep.not_run(identify)
    assert out["impact_0"]["conditions_not_run"] == ["fixed_stress"]
    assert len(out["impact_low"]["conditions_not_run"]) == 4
    assert out["impact_0"]["shares"]["y"] == {"empty": 0.1, "cap": 0.2}


def test_primary_table_refuses_secondary_rows():
    with pytest.raises(ValueError):
        rep.tables(rows() + [row("mlp", "secondary_adversary", 1, 0.0)], selected())


def test_secondary_is_labelled_and_apart():
    extra = [row("mlp", "secondary_fixed_stress", d, 0.0, 1) for d in range(1, 31)]
    out = rep.secondary_table(extra, rows())
    assert out[0]["label"].startswith("not identified") and out[0]["reading"] == "below 0"


def test_a_reused_pick_is_relabelled_only_in_the_secondary():
    base = rows()
    picks = {"s/mlp": {"point": 1}, "s/no_trade": {"point": 2}}
    primary = {"s/mlp": {"point": 1}, "s/no_trade": {"point": 3}}
    out = rep.secondary_table([], base, None, picks, primary)
    assert [(r["arm"], r["condition"]) for r in out] == [("mlp", "secondary_adversary")]
    assert out[0]["reused_primary_rows"] is True
    assert all(r["condition"] != "secondary_adversary" for r in base)


def test_secondary_shares_the_primary_metrics():
    extra = [row("mlp", "secondary_fixed_stress", d, 0.0, 1) for d in range(1, 31)]
    r = rep.secondary_table(extra, rows())[0]
    for key in ("max_abs_position", "mean_abs_position", "max_drawdown", "refusals", "attribution_mean",
                "margin_requirement_max", "limit_breaches"):
        assert key in r


def test_a_failed_secondary_prints_no_table(tmp_path):
    import json
    (tmp_path / "holdout.json").write_text(json.dumps(rows()))
    (tmp_path / "frozen.json").write_text(json.dumps({"selected": selected(), "selected_secondary": {}}))
    (tmp_path / "holdout_secondary_failed.json").write_text(json.dumps({"error": "x"}))
    (tmp_path / "i").write_text("{}")
    rep.main(tmp_path, tmp_path / "i")
    out = json.loads((tmp_path / "report.json").read_text())
    assert out["secondary"] is None and out["secondary_failure"] == {"error": "x"}
