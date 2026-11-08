"""#108's report: the margin measure, the tail and the paired days."""

import importlib.util
import pathlib
from decimal import Decimal

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m5_03_report", HERE.parent / "experiments" / "m5_03" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_maintenance_is_five_percent_of_notional_rounded_up_at_scale_six():
    assert report.maintenance("-4", "50.0") == Decimal("10.000000")
    assert report.maintenance("3", "0.0000101") == Decimal("0.000002")
    assert report.maintenance("0", None) == 0


def test_cvar_is_the_mean_of_the_worst_days():
    assert report.cvar([-9.0, -7.0] + [1.0] * 28) == pytest.approx(-8.0)
    assert report.cvar([5.0, -3.0, 2.0]) == pytest.approx(-3.0)


def test_a_learned_arms_day_is_the_mean_over_its_seeds_and_days_stay_ordered():
    rows = [
        {"condition": "unperturbed", "arm": "maskable_ppo", "regime": "thin", "day": 2, "v": 1.0},
        {"condition": "unperturbed", "arm": "maskable_ppo", "regime": "thin", "day": 2, "v": 3.0},
        {"condition": "unperturbed", "arm": "maskable_ppo", "regime": "calm", "day": 1, "v": 5.0},
    ]
    days, values = report.by_day(rows, lambda r: r["v"])[("unperturbed", "maskable_ppo")]
    assert days == [("calm", 1), ("thin", 2)]
    assert values == [5.0, 2.0]


def test_a_constant_difference_has_a_point_interval():
    assert report.stationary([2.0] * 10) == (2.0, 2.0, 2.0)
