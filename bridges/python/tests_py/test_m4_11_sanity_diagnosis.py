"""How the diagnosis of #280's BC sanity stop counts agreement. On made-up codes."""

import importlib.util
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m4_11"
sys.modules.pop("run", None)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnosis = load("m4_11_sanity_diagnosis", "sanity_diagnosis.py")
offline = load("m4_11_offline_for_diagnosis", "offline.py")
ACTIONS = offline.Actions(3, 2)


def code(bid, ask, bid_size=0, ask_size=0):
    return ACTIONS.code((1, bid, ask, bid_size, ask_size))


def test_agreement_is_counted_per_regime_and_pooled_by_decisions():
    days = [("calm", [1, 2, 3, 4], [1, 2, 9, 9], [1, 0, 0, 0]),
            ("thin", [5, 6], [5, 6], [5, 6]),
            ("calm", [7, 7], [7, 8], [7, 7])]
    out = diagnosis.breakdown(days, ACTIONS)
    assert out["decisions"] == 8
    assert out["by_regime"] == {"calm": 3 / 6, "thin": 1.0}
    # Pooled over decisions, not the mean of the regimes' shares.
    assert out["agreement"] == 5 / 8
    assert out["logged_equals_as_grid"] == 4 / 8
    assert out["as_grid_hold_share"] == 0


def test_a_quote_miss_is_near_when_the_offsets_differ_by_one_step_at_most():
    target = [code(1, 1), code(1, 1), code(1, 1), code(0, 0), 0, code(2, 2)]
    predicted = [code(1, 1, 1, 1),  # sizes only: near
                 code(2, 1),        # one bid step: near
                 code(2, 2),        # two steps: far
                 0,                 # a hold against a quote: not a quote miss
                 code(0, 0),        # a quote against a hold: not a quote miss
                 0]                 # a hold against a quote: not a quote miss
    out = diagnosis.breakdown([("calm", predicted, target, target)], ACTIONS)
    assert out["quote_misses"] == 3
    assert out["quote_misses_offsets_within_1"] == 2
    # as_grid's holds, not BC's.
    assert out["as_grid_hold_share"] == 1 / 6


def test_codes_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="different lengths"):
        diagnosis.breakdown([("calm", [1, 2], [1, 3, 4], [1, 2, 3])], ACTIONS)
    with pytest.raises(ValueError, match="different lengths"):
        diagnosis.breakdown([("calm", [1, 2], [1, 3], [1])], ACTIONS)
