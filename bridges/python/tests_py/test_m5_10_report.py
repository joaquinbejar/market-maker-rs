"""The report of #108 part 2 on toy rows."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
import report as rep  # noqa: E402


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


def test_not_run_lists_removals_and_shares():
    identify = {"s": {"void": False, "removed": ["x"], "spectral_removed": [],
                      "points": {"y": {"empty_share": 0.1, "cap_share": 0.2}}}}
    assert rep.not_run(identify)["s"]["shares"]["y"] == {"empty": 0.1, "cap": 0.2}
