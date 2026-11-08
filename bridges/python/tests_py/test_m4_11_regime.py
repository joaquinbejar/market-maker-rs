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


RUN_SPEC = importlib.util.spec_from_file_location("m4_11_regime_run", EXPERIMENT / "run.py")
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


def test_the_generator_input_is_the_earlier_design_with_only_the_offset():
    m4_05 = json.loads((HERE.parent / "experiments" / "m4_05" / "design.json").read_text())
    days_input = json.loads((EXPERIMENT / "days_input.json").read_text())
    design = json.loads((EXPERIMENT / "design.json").read_text())
    assert run.days_input_matches(days_input, m4_05, design["days"]["seed_offset"])
    assert not run.days_input_matches({**days_input, "days": 25}, m4_05, design["days"]["seed_offset"])


def test_a_started_holdout_refuses_any_further_attempt(tmp_path):
    run.refuse_if_holdout_started(tmp_path, "abc")
    run.marker_path(tmp_path, "abc").write_text("{}")
    with pytest.raises(PermissionError):
        run.refuse_if_holdout_started(tmp_path, "abc")
    run.refuse_if_holdout_started(tmp_path, "other")


def test_the_holdout_refuses_another_attempts_root():
    class Study:
        root = pathlib.Path("/elsewhere")

    with pytest.raises(PermissionError, match="root"):
        run.check_frozen(Study(), {"root": "/fixed"})


def test_the_bc_rule_needs_every_seed_at_the_threshold():
    assert not run.sanity_stops([0.9, 0.95])
    assert run.sanity_stops([0.9, 0.8999])


def test_agreement_is_pooled_by_decisions_and_given_per_regime():
    pooled, by = run.agreement_by_regime([("calm", np.array([True, False])), ("thin", np.array([True] * 2))])
    assert pooled == 0.75 and by == {"calm": 0.5, "thin": 1.0}


def test_the_compute_rule_is_more_than_half_over():
    assert not run.overran(150.0, 100.0)
    assert run.overran(150.1, 100.0)


def test_the_pooled_interval_averages_regimes_with_equal_weight():
    out = run.paired_interval({"calm": [1.0] * 5, "thin": [3.0] * 5})
    assert out["pooled"]["mean"] == 2.0 and out["pooled"]["low"] == 2.0
    assert out["pooled"]["excludes_zero"] and out["calm"]["descriptive"]
    zero = run.paired_interval({"calm": [-1.0, 1.0, -1.0, 1.0, 0.0]})
    assert not zero["pooled"]["excludes_zero"]


def test_input_precision_flags_two_raw_values_on_one_input():
    observation = np.zeros((2, 12), dtype=np.float32)
    day = {"observation": observation, "raw_position": np.array([0.0, 1.0]),
           "raw_steps_left": np.array([1, 1])}
    out = run.input_precision([day])
    assert not out["position_one_to_one"] and out["steps_left_one_to_one"]
