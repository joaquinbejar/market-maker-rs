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


def test_the_ceiling_is_every_planned_job_with_half_again():
    seconds = dict.fromkeys(run.PLANNED, 0.0)
    assert run.ceiling_hours(seconds) == 0.0
    seconds["bc"] = 3600.0
    assert run.ceiling_hours(seconds) == run.PLANNED["bc"] * 1.5


def test_the_ledger_accumulates_process_seconds(tmp_path):
    path = tmp_path / "ledger.json"
    assert run.ledger_add(path, "a", 10.0) == 10.0
    assert run.ledger_add(path, "b", 5.0) == 15.0
    assert len(json.loads(path.read_text())["steps"]) == 2


def test_a_step_is_limited_by_its_estimate_or_the_ceilings_remainder():
    assert run.step_limit(100.0, 4000.0, 4) == 150.0
    assert run.step_limit(100.0, 400.0, 4) == 100.0
    assert run.step_limit(100.0, -5.0, 4) == 0.0


def test_the_job_count_must_equal_the_plan_except_at_the_holdout():
    assert run.job_count_ok([1, 2], 2, True) and not run.job_count_ok([1], 2, True)
    assert run.job_count_ok([1], 2, False) and not run.job_count_ok([1, 2, 3], 2, False)


def test_the_planned_jobs_are_the_steps_counts():
    assert run.PLANNED["log_day"] == 570 and run.PLANNED["cql"] == 20
    assert run.PLANNED["offline_day"] == 225 + 495


def _sleep(seconds):
    import time
    time.sleep(seconds)
    return seconds


def timed_here(worker, job):
    """`run.timed_call`, importable by the spawned workers of a test: the runner's module is
    loaded from a path here, and is importable only as a script's `__main__`."""
    import time
    started = time.perf_counter()
    return worker(job), time.perf_counter() - started


@pytest.fixture
def importable_timer(monkeypatch):
    monkeypatch.setattr(run, "timed_call", timed_here)


def test_the_timed_call_returns_the_result_and_its_seconds():
    result, seconds = run.timed_call(_sleep, 0.0)
    assert result == 0.0 and 0.0 <= seconds < 1.0


def test_a_pool_at_its_limit_stops_running_workers(importable_timer):
    import time
    started = time.perf_counter()
    rows, finished, _, charged, failure = run.run_pool([0.0, 30.0], sleep_worker, 2, 8.0, "t")
    assert failure is None and rows is None and finished == 1 and 7.0 < charged < 25 and time.perf_counter() - started < 25


sleep_worker = _sleep


def test_five_jobs_on_four_workers_are_two_rounds():
    assert run.rounds(5, 4) == 2 and run.rounds(4, 4) == 1


def test_process_seconds_are_each_jobs_own(importable_timer):
    rows, finished, _, charged, _failure = run.run_pool([0.0, 0.0], sleep_worker, 2, None, "t")
    assert rows == [0.0, 0.0] and finished == 2 and charged < 1.0


def _fail_or_sleep(seconds):
    if seconds < 0:
        raise ValueError("job failed")
    return _sleep(seconds)


def test_a_failed_job_stops_the_others_and_is_charged(importable_timer):
    import time
    started = time.perf_counter()
    rows, _, _, charged, failure = run.run_pool([-1.0, 30.0], _fail_or_sleep, 2, None, "t")
    assert rows is None and isinstance(failure, ValueError) and charged >= 0.0
    assert time.perf_counter() - started < 25


def test_a_step_needs_a_pilot_that_did_not_stop_and_the_recorded_go():
    assert run.may_continue({"stop": False}, True)
    assert not run.may_continue({"stop": True}, True)
    assert not run.may_continue({"stop": False}, False)
