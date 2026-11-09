"""#106's block clock and size head on made-up windows (no capture data)."""

import pathlib
import sys

import numpy as np
import pytest

EXPERIMENT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"
sys.path.insert(0, str(EXPERIMENT))
import blocks  # noqa: E402
import mdqr  # noqa: E402

T0 = 1_766_491_200_000


def obs(ms, side, depth, kind, queue, size, sub=0):
    return {"timestamp_ns": ms * 1_000_000 + sub, "timestamp_ms": ms, "side": side,
            "depth": depth, "kind": kind, "queue_before": queue, "size": size}


def sample(ms, bid, ask):
    return {"timestamp_ms": ms, "bid": bid + [0, 0], "ask": ask + [0, 0], "spread_ticks": 1}


def window(observations, trades=(), references=(), samples=()):
    return {"window": {"start_ms": T0, "end_ms": T0 + 7_200_000}, "observations": list(observations),
            "samples": list(samples), "references": list(references), "trades": list(trades),
            "history": {"start_ms": T0 - 900_000, "last_sample": sample(T0 - 1000, [5, 6, 7], [8, 9, 10]),
                        "observations": [], "references": [], "trades": []}}


@pytest.mark.parametrize("size,cls", [(0.1, 0), (1.4, 0), (1.5, 1), (63.0, 62), (64.0, 63), (900.0, 63)])
def test_sizes_map_to_their_class(size, cls):
    assert blocks.size_class(size) == cls


def test_a_block_is_its_events_and_an_end_token():
    w = window([obs(T0 + 10, "bid", 0, "limit", 5, 2.0), obs(T0 + 10, "ask", 1, "cancel", 9, 1.0),
                obs(T0 + 30, "bid", 0, "market", 7, 1.0)])
    r = blocks.rows_of(w)
    assert len(r["wait_dt"]) == 2
    assert r["wait_dt"].tolist() == pytest.approx([0.01, 0.02])
    assert r["seq_y"].tolist() == [blocks.cell_of(0, 0, 0), blocks.cell_of(1, 1, 1), blocks.END,
                                   blocks.cell_of(0, 0, 2), blocks.END]
    assert r["seq_block"].tolist() == [0, 0, 0, 1, 1]
    assert r["size_class"].tolist() == [1, 0, 0]


def test_the_state_never_sees_the_events_own_queue():
    a = window([obs(T0 + 10, "bid", 0, "limit", 5, 1.0)])
    b = window([obs(T0 + 10, "bid", 0, "limit", 400, 1.0)])
    assert np.array_equal(blocks.rows_of(a)["seq_x"][0], blocks.rows_of(b)["seq_x"][0])


def test_the_wait_row_does_not_see_the_wait():
    a = window([obs(T0 + 10, "bid", 0, "limit", 5, 1.0), obs(T0 + 20, "bid", 0, "limit", 6, 1.0)])
    b = window([obs(T0 + 10, "bid", 0, "limit", 5, 1.0), obs(T0 + 90_000, "bid", 0, "limit", 6, 1.0)])
    assert np.array_equal(blocks.rows_of(a)["wait_x"][1], blocks.rows_of(b)["wait_x"][1])


def test_a_trade_in_the_block_is_not_in_its_imbalance():
    event = [obs(T0 + 10, "bid", 0, "market", 5, 1.0)]
    same = {"timestamp_ns": (T0 + 10) * 1_000_000, "timestamp_ms": T0 + 10, "side": "buy", "size": "1", "price": "1"}
    assert np.array_equal(blocks.rows_of(window(event))["seq_x"][0],
                          blocks.rows_of(window(event, trades=[same]))["seq_x"][0])


def test_a_later_sample_does_not_reach_the_block_before_it():
    event = [obs(T0 + 10, "bid", 0, "limit", 5, 1.0)]
    assert np.array_equal(blocks.rows_of(window(event))["seq_x"][0],
                          blocks.rows_of(window(event, samples=[sample(T0 + 10, [99, 1, 1], [1, 1, 1])]))["seq_x"][0])


def test_imbalance_counts_each_horizon():
    trades = [{"timestamp_ns": 0, "size": "1", "side": "buy"},
              {"timestamp_ns": int(100e9), "size": "1", "side": "sell"}]
    assert blocks.WindowImbalance(trades).before(int(110e9)) == [-1.0, -1.0, 0.0, 0.0]


def mechanics(cap=100):
    law = [0.0] * 6 + [1.0]
    return mdqr.Mechanics(1.0, 0.0, [law] * 3, [1, 1, 1], cap)


def flat_model(end_logit=0.0, tails=None, log_rate=0.0, tail_logit=-50.0):
    n = blocks.feature_count()
    event_b = np.full(1 + blocks.CELLS + 1, 0.0)
    event_b[0], event_b[-1] = log_rate, end_logit
    size_b = np.full(blocks.SIZE_CLASSES, -50.0)
    size_b[0], size_b[blocks.TAIL] = 0.0, tail_logit
    return {"mean": [0.0] * n, "scale": [1.0] * n,
            "event": [(np.zeros((1 + blocks.CELLS + 1, n)), event_b)],
            "size": [(np.zeros((blocks.SIZE_CLASSES, n + blocks.CELLS)), size_b)],
            "tails": tails or {}}


def generator(model, cap=100, start=([3, 3, 3], [3, 3, 3])):
    return blocks.Generator(model, mechanics(cap), 7, start)


def test_a_block_runs_past_a_single_event_and_is_never_empty():
    g = generator(flat_model())
    sizes = [len(g.block()) for _ in range(200)]
    assert min(sizes) >= 1 and max(sizes) > 1


def test_the_budget_is_a_failure_not_a_truncation():
    g = generator(flat_model(end_logit=-200.0), cap=10_000_000)
    with pytest.raises(blocks.BudgetExceeded):
        g.block()


def test_a_large_cancel_empties_the_queue_and_is_counted():
    g = generator(flat_model())
    s, d, k, q, done, _ = g._apply(blocks.cell_of(0, 1, 1), 50)
    assert (q, done, g.book.queues[0][1]) == (3, 3, 0) and g.clipped_events == 1
    assert g.clipped_volume == 47


def test_the_tail_is_masked_without_training_sizes():
    g = generator(flat_model(tail_logit=50.0, tails={}))
    x = np.zeros(blocks.feature_count())
    assert g._size(x, 0) == 1


def test_the_tail_draws_training_sizes():
    g = generator(flat_model(tail_logit=50.0, tails={"0": [70, 80]}))
    assert g._size(np.zeros(blocks.feature_count()), 0) in (70, 80) and g.tail_draws == 1


def test_an_empty_queue_cannot_be_cancelled_and_a_full_one_cannot_grow():
    g = generator(flat_model(), cap=3, start=([0, 3, 3], [3, 3, 3]))
    mask = g._mask(first=True)
    assert not mask[blocks.cell_of(0, 0, 1)] and not mask[blocks.cell_of(0, 0, 2)]
    assert not mask[blocks.cell_of(0, 1, 0)] and not mask[blocks.END]


def test_the_same_seed_is_the_same_run():
    a, b = generator(flat_model()), generator(flat_model())
    assert [a.block() for _ in range(50)] == [b.block() for _ in range(50)]


def made_up(n=6000, seed=3):
    """Blocks with a 2-event law, one size, a 0.1 s wait: what fit must recover."""
    rng = np.random.default_rng(seed)
    t, out = T0 * 1_000_000, []
    for _ in range(n):
        t += int(rng.exponential(0.1) * 1e9) + 1
        for _ in range(2):
            out.append({"timestamp_ns": t, "timestamp_ms": t // 1_000_000, "side": "bid", "depth": 0,
                        "kind": "limit", "queue_before": 5, "size": 3.0})
    return window(out)


def test_fit_recovers_a_known_rate_law_and_size():
    torch = pytest.importorskip("torch")
    rows = blocks.rows_of(made_up())
    held = blocks.rows_of(made_up(1000, 4))
    model, history = blocks.fit(rows, held, {}, 1, epochs=8, batch=256)
    assert history[-1]["validation"] > history[0]["validation"]
    x = (held["wait_x"][100] - np.asarray(model["mean"])) / np.asarray(model["scale"])
    rate = np.exp(blocks._forward(model["event"], x)[0])
    assert rate == pytest.approx(10.0, rel=0.15)
    h = np.concatenate([(held["size_x"][100] - model["mean"]) / model["scale"], np.eye(blocks.CELLS)[0]])
    assert int(np.argmax(blocks._forward(model["size"], h))) == 2
    assert torch is not None
