"""The go or no-go rule of #104, fixed before any log exists, on hand-made diagnostics."""

import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m4_07"
SPEC = importlib.util.spec_from_file_location("m4_07_decide", STUDY / "decide.py")
rule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rule)

CLEAN = {
    "propensity_recorded": True,
    "out_of_support_by_construction": False,
    "zero_propensity": 0,
    "effective_sample": 0.2,
    "max_weight": 0.01,
    "cross_check": True,
}


def with_(**changes):
    return {**CLEAN, **changes}


def test_clean_diagnostics_are_a_go():
    assert rule.decide(CLEAN) == ("go", [])


def test_each_reason_alone_is_a_no_go():
    cases = {
        "no_recorded_propensity": with_(propensity_recorded=False),
        "out_of_support": with_(zero_propensity=1),
        "low_effective_sample": with_(effective_sample=0.049),
        "heavy_weights": with_(max_weight=0.051),
        "estimators_disagree": with_(cross_check=False),
    }
    for reason, diagnostics in cases.items():
        assert rule.decide(diagnostics) == ("no_go", [reason])
    assert rule.decide(with_(out_of_support_by_construction=True)) == (
        "no_go",
        ["out_of_support"],
    )


def test_the_thresholds_are_inclusive_where_the_design_says():
    assert rule.decide(with_(effective_sample=0.05, max_weight=0.05)) == ("go", [])


def test_what_was_not_measured_cannot_pass():
    assert rule.decide(with_(effective_sample=None)) == ("no_go", ["low_effective_sample"])
    assert rule.decide(with_(cross_check=None)) == ("no_go", ["no_cross_check"])
    # With another reason, a missing cross check adds nothing.
    assert rule.decide(with_(cross_check=None, max_weight=0.5)) == ("no_go", ["heavy_weights"])
    # A weight nobody measured is not a reason by itself.
    assert rule.decide(with_(max_weight=None)) == ("go", [])


def test_reasons_come_in_a_fixed_order():
    everything = with_(
        propensity_recorded=False,
        zero_propensity=3,
        effective_sample=0.0,
        max_weight=1.0,
        cross_check=False,
    )
    assert rule.decide(everything) == (
        "no_go",
        [
            "no_recorded_propensity",
            "out_of_support",
            "low_effective_sample",
            "heavy_weights",
            "estimators_disagree",
        ],
    )


def test_the_validation_stop_rule_ignores_the_behaviours_own_rule():
    assert rule.stop_on_validation({"as_grid": 0.9, "no_trade": 0.01, "maskable_ppo-1": None})
    assert not rule.stop_on_validation({"as_grid": 0.9, "no_trade": 0.05})


def test_the_rule_matches_the_committed_design():
    design = json.loads((STUDY / "design.json").read_text())
    reasons = design["decision_rule"]["reasons"]
    assert tuple(reasons) == rule.REASONS
    assert "effective_sample < 0.05" in reasons["low_effective_sample"]
    assert rule.MIN_EFFECTIVE_SAMPLE == 0.05
    assert "max_weight > 0.05" in reasons["heavy_weights"]
    assert rule.MAX_WEIGHT == 0.05
    assert "under 0.0001" in design["diagnostics"]["low_propensity"]
    assert rule.LOW_PROPENSITY == 0.0001
