"""The off-policy estimators and diagnostics of #104, on logs held as numpy arrays.

A log is a list of trajectories, one per day and behaviour seed. Each trajectory holds, per
decision `t`, the reward `r[t]`, the behaviour's probability `mu[t]` of the logged action and,
per target, whether the target's action equals the logged one (`match[t]`). The targets here are
deterministic, so a ratio is `match / mu`. Every estimate is of the target's value over a day:
the sum over decisions of the expected reward per decision, averaged over trajectories.

Nothing here reads a target's interactive value.
"""

import numpy as np


def ratios(match, mu):
    """Per-decision ratios of a deterministic target: `1 / mu` where the target took the logged
    action, and 0 elsewhere."""
    match = np.asarray(match, dtype=bool)
    mu = np.asarray(mu, dtype=float)
    out = np.zeros_like(mu)
    np.divide(1.0, mu, out=out, where=match & (mu > 0))
    return out


def log_window_weights(rho, window=None):
    """The log of the product of the last `window` ratios at each decision, or of all of them
    when `window` is None; minus infinity where a ratio in the window is zero. Logs keep a
    product over thousands of decisions finite."""
    rho = np.asarray(rho, dtype=float)
    if window is not None and window < 1:
        raise ValueError("a window has at least one decision")
    zero = rho <= 0
    logs = np.where(zero, 0.0, np.log(np.where(zero, 1.0, rho)))
    sums = np.concatenate(([0.0], np.cumsum(logs)))
    zeros = np.concatenate(([0], np.cumsum(zero)))
    t = np.arange(1, len(rho) + 1)
    start = np.zeros_like(t) if window is None else np.maximum(t - window, 0)
    out = sums[t] - sums[start]
    out[(zeros[t] - zeros[start]) > 0] = -np.inf
    return out


def _stack(arrays, fill=0.0):
    """Trajectories of unequal length as one matrix, padded with `fill`."""
    length = max(len(a) for a in arrays)
    out = np.full((len(arrays), length), fill)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def per_decision(log_weights):
    """Weights scaled per decision so the largest across trajectories is 1 (0 where every
    weight is zero). Every estimator and diagnostic here normalizes at each decision, so this
    scale changes none of them."""
    logw = _stack(log_weights, fill=-np.inf)
    top = logw.max(axis=0)
    finite = np.isfinite(top)
    shifted = np.where(finite, logw - np.where(finite, top, 0.0), -np.inf)
    return np.exp(shifted)


def _mean_by(weight, values):
    total = weight.sum(axis=0)
    return np.divide((weight * values).sum(axis=0), total, out=np.zeros_like(total), where=total > 0)


def pdwis(rewards, log_weights):
    """Per-decision weighted importance sampling: at each decision, the weighted mean reward
    across trajectories, summed over decisions. A decision where every weight is zero adds
    nothing, which is a bias toward zero that the effective sample shows."""
    return float(_mean_by(per_decision(log_weights), _stack(rewards)).sum())


def direct(predicted_target):
    """The direct method: the model's reward for the target's action, summed over the logged
    decisions of each trajectory, averaged over trajectories."""
    return float(np.mean([np.sum(p) for p in predicted_target]))


def doubly_robust(rewards, log_weights, predicted_logged, predicted_target):
    """Per-decision doubly robust with a one-step reward model, the weights normalized at each
    decision across trajectories:
    sum over t of mean_w(r - q(s, a)) + mean_w_prev(q(s, pi(s))),
    where `w_prev` is the weight before the decision's own ratio (1 at the first decision)."""
    r = _stack(rewards)
    w = per_decision(log_weights)
    q_logged = _stack(predicted_logged)
    q_target = _stack(predicted_target)
    present = _stack([np.ones(len(x)) for x in rewards])
    # The weight before each decision's own ratio, normalized at that decision.
    log_prev = [np.concatenate(([0.0], np.asarray(lw, dtype=float)[:-1])) for lw in log_weights]
    w_prev = per_decision(log_prev) * present
    return float((_mean_by(w, r - q_logged) + _mean_by(w_prev, q_target)).sum())


def effective_sample_share(log_weights):
    """The mean over decisions of (sum of weights)^2 / sum of squared weights across the
    trajectories present, as a share of them; a decision where every weight is zero counts 0."""
    w = per_decision(log_weights)
    present = _stack([np.ones(len(x)) for x in log_weights])
    total = w.sum(axis=0)
    square = (w * w).sum(axis=0)
    ess = np.divide(total * total, square, out=np.zeros_like(total), where=square > 0)
    return float(np.mean(ess / present.sum(axis=0)))


def max_normalized_weight(log_weights):
    """The largest weight of any trajectory at any decision, as a share of the weights at that
    decision; None when no weight is positive anywhere."""
    w = per_decision(log_weights)
    total = w.sum(axis=0)
    if not (total > 0).any():
        return None
    shares = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    return float(shares.max())


def ridge_from_moments(xtx, xty, alpha):
    """Ridge regression from `X'X` and `X'y`, with an unpenalized intercept in column 0, so a
    fit over millions of decisions streams instead of holding them."""
    xtx = np.asarray(xtx, dtype=float)
    penalty = alpha * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, np.asarray(xty, dtype=float))


def ridge_fit(features, targets, alpha):
    """Ridge regression with an unpenalized intercept in column 0."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    return ridge_from_moments(x.T @ x, x.T @ y, alpha)


def squared_error_from_moments(beta, xtx, xty, yty):
    """The sum of squared residuals of `beta`, from the moments alone."""
    beta = np.asarray(beta, dtype=float)
    return float(yty - 2.0 * beta @ np.asarray(xty) + beta @ np.asarray(xtx) @ beta)


def stationary_bootstrap_indices(n, mean_block, replicates, seed):
    """Politis and Romano's stationary bootstrap over `n` ordered units: blocks of geometric
    length with mean `mean_block`, wrapping around."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty((replicates, n), dtype=int)
    for b in range(replicates):
        i = int(rng.integers(n))
        for k in range(n):
            if k > 0:
                i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
            out[b, k] = i
    return out


def interval(estimator, units, mean_block, replicates, seed, confidence=0.90):
    """The point estimate on every unit, and the percentile interval of `estimator` over the
    stationary bootstrap of the ordered `units` (each a list of trajectories, as one day)."""
    point = estimator([t for unit in units for t in unit])
    draws = []
    for index in stationary_bootstrap_indices(len(units), mean_block, replicates, seed):
        draws.append(estimator([t for i in index for t in units[i]]))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return point, float(low), float(high)


def overlap(a, b):
    """Whether two intervals `(point, low, high)` overlap."""
    return a[1] <= b[2] and b[1] <= a[2]
