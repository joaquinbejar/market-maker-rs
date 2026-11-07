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


def log_previous_weights(rho, window=None):
    """The log of the product of the ratios in the target's window before each decision's own
    ratio: over decisions t - window + 1 to t - 1, or 0 to t - 1 when `window` is None; 0 (a
    weight of 1) where that span is empty, as at the first decision or for a window of 1."""
    rho = np.asarray(rho, dtype=float)
    if window is not None and window < 1:
        raise ValueError("a window has at least one decision")
    before = np.concatenate(([1.0], rho[:-1]))
    if window is None:
        out = log_window_weights(before)
        out[0] = 0.0
        return out
    if window == 1:
        return np.zeros(len(rho))
    out = log_window_weights(before, window - 1)
    out[0] = 0.0
    return out


def _stack(arrays, fill=0.0):
    """Trajectories as one matrix. Every estimator here normalizes at each decision across
    trajectories, so a trajectory that ended earlier would leave the denominator and the
    estimate would condition on the trajectories still running; unequal lengths are refused
    instead. Every day of #102's data has the same number of decisions. `fill` is unused and
    kept for the callers."""
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"trajectories of unequal lengths {sorted(lengths)} are refused")
    return np.asarray([np.asarray(a, dtype=float) for a in arrays])


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


def doubly_robust(rewards, log_weights, predicted_logged, predicted_target, log_previous=None):
    """Per-decision doubly robust with a one-step reward model, the weights normalized at each
    decision across trajectories:
    sum over t of mean_w(r - q(s, a)) + mean_w_prev(q(s, pi(s))),
    where `w_prev` is the weight before the decision's own ratio (1 at the first decision):
    `log_previous` from `log_previous_weights` with the same window; without it, the weights
    are taken as cumulative from the first decision, and shifted by one."""
    r = _stack(rewards)
    w = per_decision(log_weights)
    q_logged = _stack(predicted_logged)
    q_target = _stack(predicted_target)
    present = _stack([np.ones(len(x)) for x in rewards])
    # The weight before each decision's own ratio, normalized at that decision.
    if log_previous is None:
        log_previous = [np.concatenate(([0.0], np.asarray(lw, dtype=float)[:-1]))
                        for lw in log_weights]
    w_prev = per_decision(log_previous) * present
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


def choose_window(ess_by_window, minimum):
    """The design's rule: the window with the largest effective sample among those at or above
    `minimum`, and otherwise the shortest. `ess_by_window` maps a window to its effective sample
    share; ties go to the shorter window."""
    windows = sorted(ess_by_window)
    passing = [w for w in windows if ess_by_window[w] is not None and ess_by_window[w] >= minimum]
    if not passing:
        return windows[0]
    return max(passing, key=lambda w: (ess_by_window[w], -w))


def decode(code, offsets, sizes):
    """A joint action code as (quote, bid offset, ask offset, bid size, ask size) in ticks and
    lots; a hold is all zeros."""
    if code == 0:
        return (0, 0, 0, 0, 0)
    index = code - 1
    k, m = len(offsets), len(sizes)
    ask_size = index % m
    index //= m
    bid_size = index % m
    index //= m
    ask = index % k
    bid = index // k
    return (1, offsets[bid], offsets[ask], sizes[bid_size], sizes[ask_size])


def action_features(codes, offsets, sizes):
    """Per action: quote, bid and ask offset over the largest offset, bid and ask size over the
    largest size. A hold is all zeros."""
    top_offset, top_size = max(offsets), max(sizes)
    table = np.zeros((1 + len(offsets) ** 2 * len(sizes) ** 2, 5))
    for code in range(1, len(table)):
        quote, bid, ask, bid_size, ask_size = decode(code, offsets, sizes)
        table[code] = (quote, bid / top_offset, ask / top_offset, bid_size / top_size,
                       ask_size / top_size)
    return table[np.asarray(codes, dtype=int)]


def features(observations, codes, offsets, sizes, position=6, spread=0):
    """The direct method's features of (observation, action): an intercept, the observation as
    the policy sees it, the action's features, and the action's interactions with the position
    and the spread."""
    obs = np.asarray(observations, dtype=float)
    act = action_features(codes, offsets, sizes)
    quote, bid, ask = act[:, 0], act[:, 1], act[:, 2]
    pos, spr = obs[:, position], obs[:, spread]
    return np.column_stack(
        [np.ones(len(obs)), obs, act, quote * pos, (bid - ask) * pos, quote * spr]
    )


def bin_states(observations, edges_imbalance, spread=0, mid_change=1, imbalance=5,
               spread_scale=10.0, mid_scale=10.0, present_offset=11):
    """The design's coverage and counting bins, from the observation as the policy sees it: the
    spread in ticks (1, 2, 3-4, 5-8, more), the absolute mid change in ticks (0, 1, 2-4, more, or
    none), and the imbalance by the given quintile edges (or none)."""
    obs = np.asarray(observations, dtype=float)
    ticks = np.rint(obs[:, spread] * spread_scale)
    spread_bin = np.digitize(ticks, [1.5, 2.5, 4.5, 8.5])
    change = np.abs(np.rint(obs[:, mid_change] * mid_scale))
    change_bin = np.where(obs[:, present_offset + mid_change] > 0,
                          np.digitize(change, [0.5, 1.5, 4.5]), 4)
    imb_bin = np.where(obs[:, present_offset + imbalance] > 0,
                       np.digitize(obs[:, imbalance], edges_imbalance), len(edges_imbalance) + 1)
    return (spread_bin * 5 + change_bin) * (len(edges_imbalance) + 2) + imb_bin


def counted_propensity(train_bins, train_codes, bins, codes):
    """The behaviour's probability of `codes` at `bins`, counted from training logs; 0 for a bin
    or an action the training logs never hold."""
    pairs = {}
    totals = {}
    for b, c in zip(np.asarray(train_bins).tolist(), np.asarray(train_codes).tolist(), strict=True):
        pairs[(b, c)] = pairs.get((b, c), 0) + 1
        totals[b] = totals.get(b, 0) + 1
    return np.asarray([pairs.get((b, c), 0) / totals[b] if b in totals else 0.0
                       for b, c in zip(np.asarray(bins).tolist(), np.asarray(codes).tolist(),
                                       strict=True)])


def stacked(arrays, fill=0.0):
    """Trajectories as one matrix, padded with `fill`, for estimators on row subsets."""
    return _stack(arrays, fill)


def _normalized(log_w):
    top = log_w.max(axis=0)
    finite = np.isfinite(top)
    return np.exp(np.where(finite, log_w - np.where(finite, top, 0.0), -np.inf))


def pdwis_rows(r, log_w):
    """`pdwis` on stacked matrices (rows are trajectories)."""
    return float(_mean_by(_normalized(log_w), r).sum())


def doubly_robust_rows(r, log_w, log_w_prev, q_logged, q_target, present):
    """`doubly_robust` on stacked matrices; `log_w_prev` is the log weight before each
    decision's own ratio (0 at the first decision), `present` marks the real decisions."""
    w = _normalized(log_w)
    w_prev = _normalized(log_w_prev) * present
    return float((_mean_by(w, r - q_logged) + _mean_by(w_prev, q_target)).sum())


def direct_rows(q_target):
    """`direct` on a stacked matrix padded with zeros."""
    return float(q_target.sum(axis=1).mean())


def interval_rows(estimator, units, mean_block, replicates, seed, confidence=0.90):
    """The estimate on every row, and the percentile interval over the stationary bootstrap of
    the ordered `units`, each a list of row indices (one day's trajectories)."""
    point = estimator(np.concatenate([np.asarray(u) for u in units]))
    draws = [estimator(np.concatenate([np.asarray(units[i]) for i in index]))
             for index in stationary_bootstrap_indices(len(units), mean_block, replicates, seed)]
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return point, float(low), float(high)
