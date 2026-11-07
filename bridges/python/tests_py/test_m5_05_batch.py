"""The batch auction toy of #289 against hand-computed clearings. All data here is synthetic."""

import importlib.util
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "m5_05_batch_fba", HERE.parent / "experiments" / "m5_05_batch" / "fba.py")
fba = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fba)


def _order(order_id, side, price, lots, arrival_ms=0):
    return fba.Order(order_id, side, price, lots, arrival_ms)


def test_a_cross_clears_at_the_volume_maximising_price():
    # D(99) = 5, S(99) = 1; D(100) = 5, S(100) = 3; D(101) = 2, S(101) = 3; D(102) = 0.
    orders = [_order(1, 1, 101, 2), _order(2, 1, 100, 3), _order(3, -1, 99, 1),
              _order(4, -1, 100, 2), _order(5, -1, 102, 5)]
    out = fba.clear(orders, reference=100)
    assert (out.price, out.volume, out.imbalance) == (100, 3, 2)
    # The 101 bid fills in full, the 100 bid gets the one lot left; both offers at or below fill.
    assert out.fills == {1: 2, 2: 1, 3: 1, 4: 2}


def test_no_cross_is_no_trade():
    out = fba.clear([_order(1, 1, 99, 1), _order(2, -1, 100, 1)], reference=100)
    assert out == fba.Clearing(None, 0, 0, {})
    assert fba.clear([], reference=100).price is None
    assert fba.clear([_order(1, 1, 99, 1)], reference=100).price is None


def test_a_volume_tie_goes_to_the_smaller_imbalance():
    # D(100) = 4, S(100) = 2: volume 2, imbalance 2; D(101) = 2, S(101) = 3: volume 2, imbalance 1.
    orders = [_order(1, 1, 101, 2), _order(2, 1, 100, 2), _order(3, -1, 100, 2),
              _order(4, -1, 101, 1)]
    out = fba.clear(orders, reference=100)
    assert (out.price, out.volume, out.imbalance) == (101, 2, -1)
    assert out.fills == {1: 2, 3: 2}


def test_then_the_price_nearest_the_reference_then_the_lower():
    # Volume 2 and no imbalance at 100 and at 102.
    orders = [_order(1, 1, 102, 2), _order(2, -1, 100, 2)]
    assert fba.clear(orders, reference=102).price == 102
    assert fba.clear(orders, reference=99).price == 100
    assert fba.clear(orders, reference=101).price == 100


def test_the_marginal_level_shares_pro_rata_and_the_remainder_by_arrival():
    # Seven lots bid at 100 for four offered: 4 x 3 // 7 = 1, 4 x 2 // 7 = 1 twice, one lot left,
    # which goes to the earliest arrival, order 2.
    orders = [_order(1, 1, 100, 3, arrival_ms=5), _order(2, 1, 100, 2, arrival_ms=1),
              _order(3, 1, 100, 2, arrival_ms=3), _order(4, -1, 100, 4)]
    out = fba.clear(orders, reference=100)
    assert (out.price, out.volume) == (100, 4)
    assert out.fills == {1: 1, 2: 2, 3: 1, 4: 4}
    assert sum(v for k, v in out.fills.items() if k != 4) == out.volume


def test_fills_balance_respect_each_order_and_keep_price_priority():
    import random

    rng = random.Random(7)
    for _ in range(500):
        orders = [_order(i, rng.choice((1, -1)), rng.randint(95, 105), rng.randint(1, 4),
                         rng.randint(0, 499)) for i in range(rng.randint(1, 12))]
        out = fba.clear(orders, reference=100)
        by_id = {o.order_id: o for o in orders}
        bought = sum(v for k, v in out.fills.items() if by_id[k].side == 1)
        sold = sum(v for k, v in out.fills.items() if by_id[k].side == -1)
        assert bought == sold == out.volume
        for k, v in out.fills.items():
            o = by_id[k]
            assert 0 < v <= o.lots
            assert (o.price >= out.price) if o.side == 1 else (o.price <= out.price)
        # Price priority: an order fills only once every more aggressive order on its side has
        # filled in full. (Strictly better than the price is not enough: with a tie on volume
        # and imbalance, the reference may pick a price at which better orders are rationed.)
        for o in orders:
            if out.fills.get(o.order_id):
                for other in orders:
                    if other.side == o.side and other.side * (other.price - o.price) > 0:
                        assert out.fills.get(other.order_id) == other.lots


def test_a_message_acts_in_the_batch_whose_close_is_after_its_arrival():
    assert fba.batch_of(0, 500) == 0
    assert fba.batch_of(499, 500) == 0
    assert fba.batch_of(500, 500) == 1
    # Sent in a batch's last 100 ms row: in time at 30 and 60 ms, late at 100 ms.
    sent = 400
    assert [fba.batch_of(sent + d, 500) for d in (30, 60, 100)] == [0, 0, 1]


def test_orders_are_checked():
    with pytest.raises(ValueError):
        fba.Order(1, 0, 100, 1, 0)
    with pytest.raises(ValueError):
        fba.Order(1, 1, 100, 0, 0)
    with pytest.raises(ValueError):
        fba.clear([_order(1, 1, 100, 1), _order(1, -1, 100, 1)], reference=100)


_ENV_SPEC = importlib.util.spec_from_file_location(
    "m5_05_batch_env", HERE.parent / "experiments" / "m5_05_batch" / "env.py")
batch_env = importlib.util.module_from_spec(_ENV_SPEC)
_ENV_SPEC.loader.exec_module(batch_env)

import numpy as np  # noqa: E402
from decimal import Decimal  # noqa: E402

HOLD = np.asarray([-1.0, 0, 0, 0, 0, -1.0], dtype=np.float32)  # no quote, one row
# A bid and an ask at offset 0 (one tick from the signal), one lot each, one row.
QUOTE = np.asarray([1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def _fixed(orders, rows=10):
    """A flat signal at 10000 and the given background orders `(arrival, side, price, lots)`."""
    flat = np.full(rows, 10_000.0)
    return batch_env.Background(flat, flat, [(*o, "limit") for o in orders])


def _env(latency, orders, rows=10):
    return batch_env.BatchEnv(seed=0, rows=rows, latency_ms=latency, fee_rate="0",
                              background=_fixed(orders, rows))


def _run(env, actions):
    env.reset()
    for action in actions:
        env.step(action)
    return env


@pytest.mark.parametrize("latency,fills", [(30, 1), (60, 1), (100, 0)])
def test_the_same_last_row_update_makes_the_batch_at_30_ms_and_misses_it_at_100_ms(latency, fills):
    # A sell at 9999 arrives at 450 ms, in batch 0; the agent's bid at 9999 is sent at 400 ms,
    # the start of the batch's last row. It arrives at 430 or 460 ms, in time, or at 500 ms,
    # the close, which puts it in batch 1, where the sell has expired.
    env = _run(_env(latency, [(450, -1, 9_999, 1)]), [HOLD] * 4 + [QUOTE] + [HOLD] * 5)
    assert len(env.fills) == fills
    if fills:
        assert env.fills[0] == (500, 1, 9_999, 1)
        assert env.position == 1


@pytest.mark.parametrize("latency,fills", [(30, 0), (100, 1)])
def test_a_cancel_that_misses_the_close_does_not_stop_the_fill(latency, fills):
    # The bid rests from row 0. At 400 ms the agent holds, which cancels it: at 30 ms the cancel
    # lands at 430, before the sell at 450; at 100 ms it lands at the close and acts in batch 1,
    # so the bid is still in batch 0 and fills.
    env = _run(_env(latency, [(450, -1, 9_999, 1)]),
               [QUOTE, QUOTE, QUOTE, QUOTE, HOLD] + [HOLD] * 5)
    assert len(env.fills) == fills


def test_the_rewards_of_a_day_sum_to_its_net_and_the_mark_is_the_last_clearing():
    env = batch_env.BatchEnv(seed=3, rows=600, latency_ms=60, fee_rate="0.0002")
    env.reset()
    total, rng = Decimal(0), np.random.default_rng(1)
    while True:
        _, _, terminated, _, info = env.step(rng.uniform(-1, 1, 6).astype(np.float32))
        total += Decimal(info["reward_exact"])
        if terminated:
            break
    assert total == Decimal(info["net"])
    cash = Decimal(0)
    for _, side, price, lots in env.fills:
        notional = Decimal(lots) * Decimal(price) * env.tick
        cash -= side * notional + env.fee_rate * notional
    assert Decimal(info["net"]) == cash + env.position * Decimal(env.reference) * env.tick
    assert len(env.fills) > 0
    assert abs(env.position) <= env.max_position


def test_a_day_is_its_seed():
    a = batch_env.draw_background(batch_env.Flow(), 5, 200, 100)
    b = batch_env.draw_background(batch_env.Flow(), 5, 200, 100)
    c = batch_env.draw_background(batch_env.Flow(), 6, 200, 100)
    assert a.orders == b.orders and (a.signal == b.signal).all()
    assert a.orders != c.orders


def test_a_duration_holds_the_quote_for_its_rows():
    env = batch_env.BatchEnv(seed=0, rows=40, latency_ms=30, fee_rate="0")
    env.reset()
    for code, rows in ((-1.0, 1), (-0.4, 5), (0.2, 10), (0.9, 30)):
        action = np.asarray([1.0, 0, 0, 0, 0, code], dtype=np.float32)
        start = env.row
        _, _, terminated, _, info = env.step(action)
        assert info["duration"] == rows and env.row == min(start + rows, 40)
        if terminated:
            break
    fixed = batch_env.BatchEnv(seed=0, rows=40, latency_ms=30, fee_rate="0", with_duration=False)
    assert fixed.action_space.shape == (5,)
    fixed.reset()
    assert fixed.step(np.zeros(5, dtype=np.float32))[4]["duration"] == 1


def test_the_quote_never_takes_the_position_past_its_limit():
    env = _env(30, [(50 + 100 * r, -1, 9_999, 3) for r in range(0, 200, 5)], rows=200)
    env.max_position = 4
    env.reset()
    for _ in range(200):
        env.step(np.asarray([1.0, -1, -1, 1, 1, -1], dtype=np.float32))
    assert env.position == 4
