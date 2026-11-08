"""The adversary's pick, the tail and the intervals of #108 part 2. SYNTHETIC."""

import math

import numpy as np

MEAN_BLOCK_DAYS = 2
REPLICATES = 2000
SEED = 108


def pick_adversary(points, losses_by_point):
    """The point with the largest mean daily loss; a tie takes the first in grid order."""
    means = [float(np.mean(losses_by_point[i])) for i in range(len(points))]
    return int(np.argmax(means)), means


def cvar_loss(daily_net, level=0.05):
    """The mean loss of the worst ceil(level x days) days: the worst 2 of 30."""
    losses = np.sort(-np.asarray(daily_net, float))[::-1]
    return float(losses[: math.ceil(level * len(losses))].mean())


def stationary_indices(n, rng, mean_block=MEAN_BLOCK_DAYS):
    """One stationary bootstrap resample of 0..n-1, wrapping at the end."""
    out = np.empty(n, int)
    out[0] = rng.integers(n)
    for t in range(1, n):
        out[t] = rng.integers(n) if rng.random() < 1 / mean_block else (out[t - 1] + 1) % n
    return out


def paired_interval(differences, replicates=REPLICATES, seed=SEED):
    """The 90 % stationary block bootstrap interval of the mean paired daily difference."""
    values = np.asarray(differences, float)
    rng = np.random.default_rng(seed)
    means = [values[stationary_indices(len(values), rng)].mean() for _ in range(replicates)]
    return [float(np.percentile(means, 5)), float(np.percentile(means, 95))]


def reading(interval):
    """Never equivalence: an interval holding 0 only shows no difference."""
    if interval[0] > 0:
        return "above 0"
    if interval[1] < 0:
        return "below 0"
    return "no difference shown"
