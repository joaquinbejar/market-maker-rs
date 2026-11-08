"""#106's S2 metrics and rule, on made-up values only (SYNTHETIC)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"))
import s2  # noqa: E402


@pytest.mark.parametrize("queue,bucket", [(0, 0), (1, 0), (2, 1), (4, 1), (5, 2), (17, 3), (64, 3), (65, 4)])
def test_size_buckets(queue, bucket):
    assert s2.size_bucket(queue) == bucket


@pytest.mark.parametrize("bid,ask,bucket", [(1, 0, 4), (0, 1, 0), (1, 1, 2), (0, 0, None)])
def test_imbalance_buckets(bid, ask, bucket):
    assert s2.imbalance_bucket(bid, ask) == bucket


def test_a_conditional_tv_weights_by_the_datas_share():
    data = {0: {"a": 3}, 1: {"a": 1}}
    model = {0: {"a": 1}, 1: {"b": 1}}
    assert s2._conditional_tv(data, model) == pytest.approx(0.25)


def test_a_move_reads_the_sample_strictly_before_it():
    samples = [[0, [1, 0, 0], [9, 0, 0], 1], [1000, [9, 0, 0], [1, 0, 0], 1]]
    trajectory = {"samples": samples, "events": [], "references": [[1000, True, False]], "directions": [1]}
    assert s2.per_block(trajectory)[0]["moves"] == [(0, 1)]


def test_an_unmoved_reference_is_no_move():
    trajectory = {"samples": [[0, [1, 0, 0], [1, 0, 0], 1]], "events": [],
                  "references": [[500, False, False]], "directions": [0]}
    assert s2.per_block(trajectory)[0]["moves"] == []


def test_a_block_undefined_on_either_arm_is_dropped_for_both():
    windows, lost = s2.paired([[1.0, np.nan, 3.0]], [[0.0, 1.0, np.nan]])
    assert windows[0].tolist() == [1.0] and lost == pytest.approx(2 / 3)


def test_the_interval_of_a_constant_difference_is_that_constant():
    low, high = s2.interval([np.full(12, -0.1), np.full(12, -0.3)], replicates=50)
    assert low == pytest.approx(-0.2) and high == pytest.approx(-0.2)


def table(better):
    arms = {}
    for arm in ("mdqr", "qr", "qr_count", "qr_constant"):
        value = 0.1 if arm == "mdqr" and better else 0.2
        arms[arm] = {m: [[value] * 12] * 3 for m in
                     [f"queue_tv_{i}" for i in range(3)] + list(s2.METRICS)}
    return arms


def test_a_clear_win_is_adopted():
    assert s2.decide(table(True), {"qr": 0.9, "qr_count": 0.8}, 0)["decision"].startswith("adopt")


def test_a_failed_draw_or_a_low_tau_or_no_win_discards():
    assert s2.decide(table(True), {"qr": 0.9, "qr_count": 0.8}, 1)["decision"] == "discard"
    assert s2.decide(table(True), {"qr": 0.5, "qr_count": 0.8}, 0)["decision"] == "discard"
    assert s2.decide(table(False), {"qr": 0.9, "qr_count": 0.8}, 0)["decision"] == "discard"


def test_tau_b_handles_ties_and_order():
    import aggregate
    assert aggregate.tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert aggregate.tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert aggregate.tau_b([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    assert np.isnan(aggregate.tau_b([1, 1], [1, 2]))
