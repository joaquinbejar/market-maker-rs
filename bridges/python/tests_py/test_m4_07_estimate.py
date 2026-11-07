"""The estimators and diagnostics of #104, on hand-made logs whose values are known."""

import importlib.util
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_07_estimate", HERE.parent / "experiments" / "m4_07" / "estimate.py"
)
est = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(est)


def test_a_ratio_is_the_inverse_propensity_where_the_target_took_the_logged_action():
    rho = est.ratios([True, False, True, True], [0.5, 0.25, 0.1, 0.0])
    assert rho.tolist() == [2.0, 0.0, 10.0, 0.0]


def test_window_weights_multiply_the_last_ratios_in_logs():
    rho = [2.0, 3.0, 0.5, 4.0]
    assert np.exp(est.log_window_weights(rho)) == pytest.approx([2.0, 6.0, 3.0, 12.0])
    assert np.exp(est.log_window_weights(rho, 1)) == pytest.approx(rho)
    assert np.exp(est.log_window_weights(rho, 2)) == pytest.approx([2.0, 6.0, 1.5, 2.0])
    # A zero ratio zeroes every window that holds it, and only those.
    zeroed = np.exp(est.log_window_weights([2.0, 0.0, 3.0, 5.0], 2))
    assert zeroed == pytest.approx([2.0, 0.0, 0.0, 15.0])
    assert np.exp(est.log_window_weights([2.0, 0.0, 3.0])).tolist() == [2.0, 0.0, 0.0]
    # A product over thousands of decisions stays finite.
    assert np.isfinite(est.log_window_weights(np.full(10_000, 700.0))).all()
    with pytest.raises(ValueError):
        est.log_window_weights(rho, 0)


def logs(weights):
    with np.errstate(divide="ignore"):
        return [np.log(np.asarray(w, dtype=float)) for w in weights]


def test_pdwis_on_policy_is_the_mean_daily_reward():
    rewards = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 0.0, 1.0])]
    weights = [np.ones(3), np.ones(3)]
    assert est.pdwis(rewards, logs(weights)) == pytest.approx(5.0)


def test_pdwis_by_hand():
    # Decision 0: weights 2 and 0 -> 1.0; decision 1: weights 1 and 3 -> (1*4 + 3*0) / 4 = 1.0;
    # decision 2: only the first trajectory is that long -> 5.0.
    rewards = [np.array([1.0, 4.0, 5.0]), np.array([7.0, 0.0])]
    weights = [np.array([2.0, 1.0, 1.0]), np.array([0.0, 3.0])]
    assert est.pdwis(rewards, logs(weights)) == pytest.approx(7.0)
    # A decision where every weight is zero adds nothing.
    assert est.pdwis([np.array([5.0])], logs([[0.0]])) == 0.0
    # Weights of any common scale at a decision give the same estimate.
    huge = [w + 900.0 for w in logs(weights)]
    assert est.pdwis(rewards, huge) == pytest.approx(7.0)


def test_doubly_robust_with_no_model_is_pdwis_and_with_an_exact_model_is_direct():
    rng = np.random.default_rng(7)
    rewards = [rng.normal(size=6) for _ in range(4)]
    weights = [rng.uniform(0.1, 3.0, size=6) for _ in range(4)]
    zero = [np.zeros(6) for _ in range(4)]
    assert est.doubly_robust(rewards, logs(weights), zero, zero) == pytest.approx(
        est.pdwis(rewards, logs(weights))
    )
    # A model that predicts every logged reward exactly leaves only the target term, which
    # with weights of one is the direct method.
    ones = [np.ones(6) for _ in range(4)]
    target = [rng.normal(size=6) for _ in range(4)]
    assert est.doubly_robust(rewards, logs(ones), rewards, target) == pytest.approx(
        est.direct(target)
    )


def test_doubly_robust_by_hand():
    # Two trajectories of two decisions, weights (cumulative) [1, 2] and [3, 0]:
    # t=0: w (1, 3) -> mean(r - q) = (1*1 + 3*2) / 4 = 1.75; w_prev (1, 1) -> mean(qt) = 1.5;
    # t=1: w (2, 0) -> r - q of the first = 3; w_prev (1, 3) -> (1*4 + 3*0) / 4 = 1.0.
    rewards = [np.array([2.0, 5.0]), np.array([4.0, 1.0])]
    logged = [np.array([1.0, 2.0]), np.array([2.0, 1.0])]
    target = [np.array([1.0, 4.0]), np.array([2.0, 0.0])]
    weights = logs([[1.0, 2.0], [3.0, 0.0]])
    assert est.doubly_robust(rewards, weights, logged, target) == pytest.approx(7.25)


def test_the_effective_sample_and_the_largest_weight_per_decision():
    assert est.effective_sample_share(logs([np.ones(4), np.ones(4)])) == pytest.approx(1.0)
    # Decision 0: weights (1, 1) -> 1.0; decision 1: (1, 0) -> 1 of 2 -> 0.5.
    assert est.effective_sample_share(logs([[1.0, 1.0], [1.0, 0.0]])) == pytest.approx(0.75)
    # A decision with no weight counts 0.
    assert est.effective_sample_share(logs([[0.0, 1.0]])) == pytest.approx(0.5)
    assert est.effective_sample_share(logs([np.zeros(3)])) == 0.0
    # Decision 0: 1 of 1 + 3 = 0.25 and 0.75; decision 1: 3 and 4 -> 4/7.
    assert est.max_normalized_weight(logs([[1.0, 3.0], [3.0, 4.0]])) == pytest.approx(0.75)
    assert est.max_normalized_weight(logs([np.zeros(2)])) is None


def test_ridge_recovers_a_linear_reward_and_leaves_the_intercept_free():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    y = x @ np.array([5.0, 2.0, -1.0])
    assert est.ridge_fit(x, y, 1e-9) == pytest.approx([5.0, 2.0, -1.0], abs=1e-6)
    # A large penalty shrinks the slopes but not the intercept.
    beta = est.ridge_fit(x, y, 1e9)
    assert beta[0] == pytest.approx(np.mean(y), abs=0.05)
    assert abs(beta[1]) < 1e-3
    # The squared error from the moments equals the direct one.
    noisy = y + rng.normal(size=200)
    beta = est.ridge_fit(x, noisy, 1.0)
    direct = float(np.sum((noisy - x @ beta) ** 2))
    assert est.squared_error_from_moments(beta, x.T @ x, x.T @ noisy, noisy @ noisy) == (
        pytest.approx(direct)
    )


def test_the_stationary_bootstrap_is_seeded_and_continues_blocks():
    a = est.stationary_bootstrap_indices(10, 2.0, 50, 104)
    b = est.stationary_bootstrap_indices(10, 2.0, 50, 104)
    assert (a == b).all()
    assert a.shape == (50, 10)
    assert a.min() >= 0 and a.max() <= 9
    # With a very long mean block, a draw is one wrapped run.
    run = est.stationary_bootstrap_indices(6, 1e12, 1, 1)[0]
    assert all((run[k] + 1) % 6 == run[k + 1] for k in range(5))


def test_an_interval_of_a_constant_is_that_constant_and_intervals_overlap_by_bounds():
    units = [[np.ones(2)] for _ in range(5)]
    point, low, high = est.interval(lambda ts: 3.0, units, 2.0, 20, 104)
    assert (point, low, high) == (3.0, 3.0, 3.0)
    assert est.overlap((0, 1.0, 2.0), (0, 2.0, 3.0))
    assert not est.overlap((0, 1.0, 2.0), (0, 2.1, 3.0))
