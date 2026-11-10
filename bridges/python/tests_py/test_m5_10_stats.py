"""The statistics of #108 part 2 on toy numbers."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from m5_10_load import module  # noqa: E402

s = module("stats")


def test_adversary_takes_the_largest_mean_loss_and_the_first_on_a_tie():
    assert s.pick_adversary("abc", {0: [1, 1], 1: [3, 1], 2: [2, 2]})[0] == 1


def test_cvar_is_the_worst_two_of_thirty():
    net = np.arange(30.0) - 5
    assert s.cvar_loss(net) == 4.5


def test_stationary_indices_are_in_range_and_keep_blocks():
    rng = np.random.default_rng(1)
    idx = s.stationary_indices(1000, rng)
    assert idx.min() >= 0 and idx.max() < 1000
    continued = np.mean((idx[1:] - idx[:-1]) % 1000 == 1)
    assert 0.45 < continued < 0.55


def test_interval_is_seeded_and_constant_series_collapses():
    assert s.paired_interval(np.ones(30), replicates=50) == [1.0, 1.0]
    x = np.random.default_rng(2).normal(size=30)
    assert s.paired_interval(x, replicates=50) == s.paired_interval(x, replicates=50)


def test_reading_never_claims_equivalence():
    assert s.reading([-1, 1]) == "no difference shown"
    assert s.reading([1, 2]) == "above 0" and s.reading([-2, -1]) == "below 0"
