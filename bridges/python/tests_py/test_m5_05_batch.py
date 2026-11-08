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
    assert env.late["orders"] == (2 if latency == 100 else 0)
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


_RUN_SPEC = importlib.util.spec_from_file_location(
    "m5_05_batch_run", HERE.parent / "experiments" / "m5_05_batch" / "run.py")
batch_run = importlib.util.module_from_spec(_RUN_SPEC)
_RUN_SPEC.loader.exec_module(batch_run)


def test_markouts_follow_the_reference_after_each_fill():
    env = _env(30, [])
    env.reset()
    env.batches = [(500, 100, 1), (1000, 103, 1), (1500, None, 0), (2000, 98, 1)]
    env.fills = [(500, 1, 100, 2), (1000, -1, 103, 1)]
    out = batch_run.markouts(env)
    tick = env.tick
    # At 1 s: the buy at 500 sees the reference of 1500, still 103; the sell at 1000 sees 98.
    assert Decimal(out["1000"]) == 2 * (103 - 100) * tick + -1 * (98 - 103) * tick
    # At 5 s both horizons pass the last close and are left out.
    assert Decimal(out["5000"]) == 0


def test_the_drawdown_is_the_largest_fall_from_a_peak():
    values = [Decimal(v) for v in ("1", "3", "2", "5", "1", "4")]
    assert batch_run.drawdown(values) == Decimal(4)
    assert batch_run.drawdown([Decimal("-2")]) == Decimal(2)


def test_the_best_point_is_the_highest_mean_net_with_ties_by_text():
    rows = [{"candidate": '{"a": 1}', "net": "1"}, {"candidate": '{"a": 1}', "net": "3"},
            {"candidate": '{"a": 2}', "net": "2"}, {"candidate": '{"a": 2}', "net": "2"}]
    assert batch_run.best(rows) == ({"a": 2}, "2")


def test_an_episode_reports_its_day():
    env = batch_env.BatchEnv(seed=2, rows=500, latency_ms=100, fee_rate="0.0002")
    day = batch_run.episode(batch_run.as_policy({"risk_aversion": "0.001", "order_intensity": "100"},
                                                0.003, batch_run.load()), env)
    assert day["fills"] == len(env.fills) and day["decisions"] == 500
    assert day["durations"] == {"1": 500}
    assert set(day["markouts"]) == {"1000", "5000", "30000"}
    assert day["late"]["orders"] > 0


def test_the_teacher_is_cloned_into_the_policy():
    from stable_baselines3 import PPO

    design = batch_run.load()
    predictor = batch_run.Predictor([0.0] * 5)
    envs = [batch_env.BatchEnv(seed=s, rows=200, latency_ms=30, fee_rate="0", predictor=predictor)
            for s in (1, 2)]
    model = PPO("MlpPolicy", envs[0], seed=1, device="cpu", n_steps=64, batch_size=32)
    teacher = {"risk_aversion": "0.001", "order_intensity": "100"}
    design = {**design, "teacher_fit": {**design["teacher_fit"], "epochs": 30}}
    losses = batch_run.clone(model, envs, teacher, 0.003, design, 400, seed=1)
    assert losses[-1] < losses[0] / 4


def test_holdout_days_are_refused_outside_their_step():
    design = batch_run.load()
    with pytest.raises(PermissionError):
        batch_run.make_env(design, 20, 30, "0", "full")
    assert batch_run.make_env(design, 15, 30, "0", "no_duration").action_space.shape == (5,)


REPORTS = ("predictor", "fee_rule", "check", "classical", "pilot")


def frozen_context(tmp_path, monkeypatch, pool):
    import json

    ctx = batch_run.Context(tmp_path, "c0ffee", 1)
    for name in REPORTS:
        (tmp_path / f"{name}.json").write_text("{}")
    manifest = {"design_sha256": batch_run.sha256(batch_run.HERE / "design.json"),
                "code_sha256": {n: batch_run.sha256(batch_run.HERE / n) for n in batch_run.CODE},
                **{f"{n}_sha256": batch_run.sha256(tmp_path / f"{n}.json") for n in REPORTS},
                "runs": {}}
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(ctx, "runs", lambda: [])
    monkeypatch.setattr(ctx, "header", lambda step: {"step": step})
    monkeypatch.setattr(batch_run, "holdout_jobs", lambda ctx, runs: [{"day": 20}])
    monkeypatch.setattr(batch_run, "pool", pool)
    return ctx, manifest, batch_run.sha256(tmp_path / "frozen.json")


def test_the_freeze_records_the_code_and_the_holdout_refuses_changed_code(tmp_path, monkeypatch):
    import json

    calls = []
    ctx, manifest, good = frozen_context(
        tmp_path, monkeypatch, lambda jobs, worker, parallel, label: calls.append(1) or [{}])
    assert set(batch_run.CODE) == {"run.py", "env.py", "fba.py"}
    for name in batch_run.CODE:
        changed = {**manifest["code_sha256"], name: "0" * 64}
        (tmp_path / "frozen.json").write_text(json.dumps(manifest | {"code_sha256": changed}))
        with pytest.raises(SystemExit, match=f"{name} changed after the freeze"):
            batch_run.holdout(ctx, batch_run.sha256(tmp_path / "frozen.json"))
        assert not (tmp_path / batch_run.HOLDOUT_MARKER).exists() and calls == []
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    batch_run.holdout(ctx, good)
    assert calls == [1] and (tmp_path / "holdout.json").exists()
    # The freeze writes the digests of the code the holdout checks.
    fresh = batch_run.Context(tmp_path / "fresh", "c0ffee", 1)
    for name in REPORTS:
        (fresh.out / f"{name}.json").write_text("{}")
    monkeypatch.setattr(fresh, "runs", lambda: [])
    batch_run.freeze(fresh)
    written = json.loads((fresh.out / "frozen.json").read_text())
    assert written["code_sha256"] == manifest["code_sha256"]


def test_a_failed_or_concurrent_holdout_is_never_run_again(tmp_path, monkeypatch):
    import json
    import pathlib as paths

    calls = []

    def failing(jobs, worker, parallel, label):
        calls.append(len(jobs))
        raise RuntimeError("a worker failed after reading days")

    ctx, _, good = frozen_context(tmp_path, monkeypatch, failing)
    with pytest.raises(RuntimeError, match="worker failed"):
        batch_run.holdout(ctx, good)
    marker = tmp_path / batch_run.HOLDOUT_MARKER
    assert marker.exists() and not (tmp_path / "holdout.json").exists()
    assert json.loads(marker.read_text())["frozen_sha256"] == good
    with pytest.raises(SystemExit, match="started already"):
        batch_run.holdout(ctx, good)
    assert calls == [1]
    # The marker is created exclusively: an attempt that loses the race fails on creation.
    marker.unlink()
    real = batch_run.os.open

    def racing(path, flags, *args):
        if str(path).endswith(batch_run.HOLDOUT_MARKER) and not batch_run.os.path.exists(path):
            paths.Path(path).write_text("{}")
        return real(path, flags, *args)

    monkeypatch.setattr(batch_run.os, "open", racing)
    with pytest.raises(FileExistsError):
        batch_run.holdout(ctx, good)
    assert calls == [1]


def test_every_pool_worker_is_this_runners_own_function():
    # A spawned worker imports its function from the main module: the pool must be given
    # functions this runner defines, never another module's.
    import ast

    tree = ast.parse((batch_run.HERE / "run.py").read_text())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    workers = [call.args[1] for call in ast.walk(tree)
               if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "pool"]
    assert workers and all(isinstance(w, ast.Name) and w.id in defined for w in workers), [
        ast.unparse(w) for w in workers]


def _scripted(actions):
    """A policy that takes `actions[row]` at the decision on that row, and holds otherwise."""
    def policy(env, obs):
        return actions.get(env.row, HOLD)
    return policy


# A bid and an ask at one tick from the signal, one lot each, resting 30 rows.
LONG_QUOTE = np.asarray([1.0, -1.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float32)


def test_a_step_of_several_closes_is_measured_at_each_clearing():
    # At 100 ms the bid sent on row 4 arrives at the 500 ms close, so it acts in the next batch,
    # while the bid sent on row 3 is still live at 500 ms: both fill, and the position reaches
    # 2 against a limit of 1. The ask of the long quote brings it back to 1 at 1500 ms, inside
    # the same step, so a position read only at the step's end would see 1 and no breach.
    background = [(450, -1, 9_999, 1), (950, -1, 9_999, 1), (1_450, 1, 10_001, 1)]
    env = _env(100, background, rows=35)
    env.max_position = 1
    out = batch_run.episode(_scripted({3: QUOTE, 4: LONG_QUOTE}), env)
    assert [(close, side) for close, side, _, _ in env.fills] == [(500, 1), (1000, 1), (1500, -1)]
    assert env.position == 1
    assert out["peak_position"] == 2 and out["breach"] is True
    # The drawdown reads the equity at each close too: one per batch, then the day's end.
    assert len(env.marks) == 7
    worst = batch_run.drawdown([mark for _, _, mark in env.marks] + [Decimal(out["net"])])
    assert Decimal(out["max_drawdown"]) == worst


def _aggressive_peaks(latency):
    """Per seed, the peak position and the breach of a policy that quotes one tick away at
    the largest size on every row it decides, for a random duration, against a limit of 2."""
    rng = np.random.default_rng(latency)
    out = []
    for seed in range(6):
        env = batch_env.BatchEnv(seed=seed, rows=600, latency_ms=latency, fee_rate="0")
        env.max_position = 2
        actions = {row: np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, rng.uniform(-1, 1)],
                                   dtype=np.float32) for row in range(600)}
        day = batch_run.episode(_scripted(actions), env)
        assert len(env.fills) > 15
        out.append((day["peak_position"], day["breach"]))
    return out


def test_below_a_row_of_delay_no_quote_takes_the_position_past_its_limit():
    # A replacement's cancel and its new orders arrive together, before the close, and each
    # order is sized on the position at its submission: at most one order a side is live at a
    # close, so the limit holds at every clearing whatever the durations.
    for latency in (30, 60):
        assert all(peak <= 2 and not breach for peak, breach in _aggressive_peaks(latency))
    # At a delay of one row a bid sent on a batch's last row arrives at the close and acts in
    # the next batch while the previous bid is still live: the same play breaches.
    assert any(breach for _, breach in _aggressive_peaks(100))


def test_the_drawdown_reads_a_dip_inside_a_long_step():
    # One long quote: its bid fills at 9999 at 500 ms; the background then clears at 9990 at
    # 1000 ms and at 10000 at 1500 ms. The mark falls 9 ticks and recovers inside the step,
    # which a drawdown over the steps' ends would miss.
    background = [(450, -1, 9_999, 1), (950, 1, 9_990, 1), (960, -1, 9_990, 1),
                  (1_450, 1, 10_000, 1), (1_460, -1, 10_000, 1)]
    env = _env(30, background, rows=35)
    out = batch_run.episode(_scripted({0: LONG_QUOTE}), env)
    assert [(close, side) for close, side, _, _ in env.fills] == [(500, 1)]
    assert Decimal(out["max_drawdown"]) == Decimal(9) * env.tick


def test_the_audit_counts_breach_days_and_the_largest_position_per_arm_and_latency():
    import sys as _sys

    _sys.modules.pop("run", None)
    audit_spec = importlib.util.spec_from_file_location(
        "m5_05_batch_audit", HERE.parent / "experiments" / "m5_05_batch" / "audit.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    rows = [{"arm": "full", "latency": 30, "breach": False, "peak_position": 20},
            {"arm": "full", "latency": 30, "breach": True, "peak_position": 21},
            {"arm": "full", "latency": 100, "breach": False, "peak_position": 19},
            {"arm": "no_duration", "latency": 30, "breach": False, "peak_position": 20}]
    assert audit.summarise(rows) == {
        "full/100ms": {"days": 1, "breach_days": 0, "peak_position": 19},
        "full/30ms": {"days": 2, "breach_days": 1, "peak_position": 21},
        "no_duration/30ms": {"days": 1, "breach_days": 0, "peak_position": 20},
    }
