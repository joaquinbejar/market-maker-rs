"""The regime's one-hot and the aliasing diagnosis of #309, on made-up decisions."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m4_11_regime"
SPEC = importlib.util.spec_from_file_location("m4_11_regime", EXPERIMENT / "regime.py")
regime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regime)


def day(name, targets, positions=(0.0, 1.0), steps=(0.95, 0.05)):
    return {"regime": name, "position": list(positions), "steps_left": list(steps), "steps_fraction": list(steps),
            "mask": np.ones((len(targets), 3), dtype=bool), "state_bin": [0] * len(targets),
            "target": list(targets)}


def test_the_one_hot_follows_the_design_order():
    assert regime.one_hot("stressed").tolist() == [0.0, 1.0, 0.0]
    design = json.loads((EXPERIMENT / "design.json").read_text())
    assert design["regime_input"]["choice"].index("calm") < design["regime_input"]["choice"].index("thin")


def test_with_regime_appends_three_columns_to_one_or_many():
    assert regime.with_regime(np.zeros(4), "thin").tolist() == [0, 0, 0, 0, 0, 0, 1]
    many = regime.with_regime(np.zeros((2, 4)), "calm")
    assert many.shape == (2, 7) and many[:, 4].tolist() == [1, 1]


def test_the_ceiling_counts_each_keys_most_frequent_action():
    out = regime.ceiling(["a", "a", "a", "b"], [1, 1, 2, 3])
    assert out["agreement_ceiling"] == 0.75 and out["ambiguous_share"] == 0.75 and out["keys"] == 2


def test_the_ceiling_refuses_rows_that_do_not_match():
    with pytest.raises(ValueError):
        regime.ceiling(["a"], [1, 2])


def test_aliasing_is_seen_without_the_regime_and_resolved_with_it():
    out = regime.aliasing([day("calm", [1, 2]), day("thin", [3, 4])])
    assert out["exact"]["agreement_ceiling"] == 0.5 and out["exact"]["ambiguous_share"] == 1.0
    assert out["exact_with_regime"]["agreement_ceiling"] == 1.0 and not out["stop"]
    assert out["binned_with_regime"]["agreement_ceiling"] == 1.0


def test_labels_the_regime_does_not_determine_stop_the_study():
    assert regime.aliasing([day("calm", [1, 2]), day("calm", [3, 2])])["stop"]


def test_no_aliasing_is_reported_and_does_not_stop():
    out = regime.aliasing([day("calm", [1, 2]), day("thin", [1, 2])])
    assert out["exact"]["agreement_ceiling"] == 1.0 and not out["stop"]


def test_steps_left_fall_in_ten_bins_with_the_top_edge_in_the_last():
    assert regime.steps_bin([0.0, 0.09, 0.1, 1.0]).tolist() == [0, 0, 1, 9]
