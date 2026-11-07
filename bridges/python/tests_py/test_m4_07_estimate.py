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
    # decision 2: weights 1 and 1 -> (5 + 1) / 2 = 3.0.
    rewards = [np.array([1.0, 4.0, 5.0]), np.array([7.0, 0.0, 1.0])]
    weights = [np.array([2.0, 1.0, 1.0]), np.array([0.0, 3.0, 1.0])]
    assert est.pdwis(rewards, logs(weights)) == pytest.approx(5.0)
    # A decision where every weight is zero adds nothing.
    assert est.pdwis([np.array([5.0])], logs([[0.0]])) == 0.0
    # Weights of any common scale at a decision give the same estimate.
    huge = [w + 900.0 for w in logs(weights)]
    assert est.pdwis(rewards, huge) == pytest.approx(5.0)


def test_unequal_lengths_are_refused_rather_than_conditioned_on_survivors():
    # On policy, rewards [1, 2] and [3] have a true mean daily value of 3; normalizing at each
    # decision over the trajectories still running would give 2 + 2 = 4.
    rewards = [np.array([1.0, 2.0]), np.array([3.0])]
    zero = [np.zeros(2), np.zeros(1)]
    with pytest.raises(ValueError, match="unequal"):
        est.pdwis(rewards, zero)
    with pytest.raises(ValueError, match="unequal"):
        est.doubly_robust(rewards, zero, rewards, rewards)
    with pytest.raises(ValueError, match="unequal"):
        est.effective_sample_share(zero)


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


def test_the_window_is_the_largest_effective_sample_that_passes_or_the_shortest():
    assert est.choose_window({1: 0.4, 10: 0.2, 100: 0.01}, 0.05) == 1
    assert est.choose_window({1: 0.1, 10: 0.3, 100: 0.01}, 0.05) == 10
    assert est.choose_window({1: 0.01, 10: 0.02, 100: None}, 0.05) == 1
    # A tie goes to the shorter window.
    assert est.choose_window({1: 0.3, 10: 0.3}, 0.05) == 1


OFFSETS = [0, 1, 2, 4, 8]
SIZES = [1, 2]


def test_codes_decode_to_the_grid():
    assert est.decode(0, OFFSETS, SIZES) == (0, 0, 0, 0, 0)
    # code = 1 + ((bid * 5 + ask) * 2 + bid size) * 2 + ask size
    assert est.decode(1 + ((3 * 5 + 3) * 2 + 0) * 2 + 0, OFFSETS, SIZES) == (1, 4, 4, 1, 1)
    assert est.decode(1 + ((4 * 5 + 0) * 2 + 1) * 2 + 1, OFFSETS, SIZES) == (1, 8, 0, 2, 2)
    assert est.decode(100, OFFSETS, SIZES) == (1, 8, 8, 2, 2)


def test_the_direct_methods_features():
    obs = np.zeros((2, 22))
    obs[:, 6] = [0.5, -0.25]
    obs[:, 0] = [0.2, 0.4]
    codes = [0, 1 + ((4 * 5 + 0) * 2 + 1) * 2 + 1]
    x = est.features(obs, codes, OFFSETS, SIZES)
    assert x.shape == (2, 1 + 22 + 5 + 3)
    assert x[0, 23:].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]
    # quote, bid 8/8, ask 0/8, sizes 2/2; quote*pos, (bid-ask)*pos, quote*spread.
    assert x[1, 23:].tolist() == pytest.approx([1, 1, 0, 1, 1, -0.25, -0.25, 0.4])


def test_bins_and_counted_propensities():
    obs = np.zeros((4, 22))
    obs[:, 0] = [0.1, 0.2, 0.4, 0.9]  # 1, 2, 4 and 9 ticks
    obs[:, 1] = [0.0, 0.1, 0.3, 0.6]
    obs[:, 12] = [0, 1, 1, 1]  # the first has no mid change
    obs[:, 5] = [-0.9, 0.0, 0.5, 0.9]
    obs[:, 16] = [1, 1, 1, 0]  # the last has no imbalance
    edges = [-0.5, 0.0, 0.5, 0.75]
    bins = est.bin_states(obs, edges)
    # (spread bin * 5 + change bin) * 6 + imbalance bin
    assert bins.tolist() == [(0 * 5 + 4) * 6 + 0, (1 * 5 + 1) * 6 + 2,
                             (2 * 5 + 2) * 6 + 3, (4 * 5 + 3) * 6 + 5]
    train_bins = [7, 7, 7, 9]
    train_codes = [0, 0, 5, 3]
    mu = est.counted_propensity(train_bins, train_codes, [7, 7, 9, 11, 9], [0, 5, 3, 0, 4])
    assert mu.tolist() == pytest.approx([2 / 3, 1 / 3, 1.0, 0.0, 0.0])


def test_the_row_estimators_equal_the_list_ones():
    rng = np.random.default_rng(11)
    lengths = [5, 5, 5]
    rewards = [rng.normal(size=n) for n in lengths]
    log_w = [np.log(rng.uniform(0.1, 3.0, size=n)) for n in lengths]
    logged = [rng.normal(size=n) for n in lengths]
    target = [rng.normal(size=n) for n in lengths]
    r = est.stacked(rewards)
    lw = est.stacked(log_w, fill=-np.inf)
    prev = est.stacked([np.concatenate(([0.0], x[:-1])) for x in log_w], fill=-np.inf)
    present = est.stacked([np.ones(n) for n in lengths])
    assert est.pdwis_rows(r, lw) == pytest.approx(est.pdwis(rewards, log_w))
    assert est.doubly_robust_rows(
        r, lw, prev, est.stacked(logged), est.stacked(target), present
    ) == pytest.approx(est.doubly_robust(rewards, log_w, logged, target))
    assert est.direct_rows(est.stacked(target)) == pytest.approx(est.direct(target))
    units = [[0], [1, 2]]
    point, low, high = est.interval_rows(lambda rows: est.pdwis_rows(r[rows], lw[rows]),
                                         units, 2.0, 50, 104)
    assert point == pytest.approx(est.pdwis(rewards, log_w))
    assert low <= high


def test_the_previous_weight_is_the_targets_window_without_its_own_ratio():
    rho = [2.0, 3.0, 0.5, 4.0]
    # Full window: the product of every ratio before the decision.
    assert np.exp(est.log_previous_weights(rho)) == pytest.approx([1.0, 2.0, 6.0, 3.0])
    # A window of 1 holds the decision's own ratio only, so nothing comes before it.
    assert np.exp(est.log_previous_weights(rho, 1)) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    # A window of 2: the one ratio before the decision.
    assert np.exp(est.log_previous_weights(rho, 2)) == pytest.approx([1.0, 2.0, 3.0, 0.5])
    # A window of 3: the two before it.
    assert np.exp(est.log_previous_weights(rho, 3)) == pytest.approx([1.0, 2.0, 6.0, 1.5])
    with pytest.raises(ValueError):
        est.log_previous_weights(rho, 0)


def test_a_window_one_doubly_robust_weights_the_target_term_by_one():
    # Rewards 1, a model of 0 for the logged action and 10 then 20 for the target's: at the
    # second decision both trajectories' previous weights are 1, so the target term is 15,
    # where the shifted window weights 2 and 0 would give 10.
    rho = [np.array([2.0, 2.0]), np.array([0.0, 2.0])]
    rewards = [np.ones(2), np.ones(2)]
    logged = [np.zeros(2), np.zeros(2)]
    target = [np.array([0.0, 10.0]), np.array([0.0, 20.0])]
    log_w = [est.log_window_weights(r, 1) for r in rho]
    log_prev = [est.log_previous_weights(r, 1) for r in rho]
    assert est.doubly_robust(rewards, log_w, logged, target, log_prev) == pytest.approx(17.0)
    r = est.stacked(rewards)
    present = est.stacked([np.ones(2), np.ones(2)])
    assert est.doubly_robust_rows(r, est.stacked(log_w), est.stacked(log_prev), est.stacked(logged),
                                  est.stacked(target), present) == pytest.approx(17.0)
