"""#282's report on made-up rows: every condition is paired with its unperturbed days."""

import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m5_04_report", HERE.parent / "experiments" / "m5_04" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def rows(net_of):
    out = []
    for condition in report.CONDITIONS:
        for arm in report.ARMS:
            for regime in ("calm", "stressed", "thin"):
                for day in range(1, 11):
                    out.append({"condition": condition, "arm": arm, "regime": regime, "day": day,
                                "net": str(net_of(condition, arm, day)), "max_drawdown": "1",
                                "limit_breach": False, "reactions": 3, "moves": 4,
                                "emptied": "2.5"})
    return out


def run_report(tmp_path, capsys, net_of):
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({"data": "made up", "bounds": "none", "rows": rows(net_of)}))
    sys.argv = ["report.py", str(path)]
    report.main()
    return capsys.readouterr().out


def test_an_impact_that_costs_every_day_moves_the_interval(tmp_path, capsys):
    out = run_report(tmp_path, capsys,
                     lambda c, a, d: -1 if c == "impact_only" and a != "no_trade" else 0)
    assert "| fixed_spread | impact_only | -1.00 [-1.00, -1.00]" in out
    assert "Stop rule" not in out
    assert "| impact_only | glft | -1.00 |" in out
    assert "| 3.0 | 4.0 | 2.50 |" in out


def test_no_change_anywhere_reads_the_stop_rule(tmp_path, capsys):
    out = run_report(tmp_path, capsys, lambda c, a, d: d % 3)
    assert "Stop rule: no condition changed" in out
    assert "never read as equivalence" in out


def test_a_condition_on_other_days_is_refused(tmp_path, capsys):
    path = tmp_path / "holdout.json"
    data = rows(lambda c, a, d: 0)
    data = [r for r in data if not (r["condition"] == "adversary" and r["day"] == 10)]
    path.write_text(json.dumps({"data": "x", "bounds": "y", "rows": data}))
    sys.argv = ["report.py", str(path)]
    with pytest.raises(ValueError, match="other days"):
        report.main()
