"""The runner of the regime adaptation and transfer study (#112). All data here is synthetic."""

import importlib.util
import pathlib
from decimal import Decimal

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "m5_08_run", HERE.parent / "experiments" / "m5_08" / "run.py")
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)

POINT = {"risk_aversion": "0.0001", "volatility": "40", "order_intensity": "50",
         "horizon_ms": 86400000, "size": "1"}
WIDE = {**POINT, "risk_aversion": "0.01", "order_intensity": "100"}


def _rows(count=300):
    """A synthetic touch that walks a tick now and then, in kernel ticks and lots."""
    rng = np.random.default_rng(0)
    mid, rows = 10000, []
    for i in range(count):
        mid += int(rng.integers(-1, 2))
        rows.append((14 * 86_400_000_000_000 + i * 1_000_000_000, mid - 1, 3, mid + 1, 4))
    return rows


@pytest.fixture
def synthetic_day(monkeypatch):
    """Validation day 15 of either simulator: synthetic rows, and an alarm on rows 100 to 199."""
    _, base = run.load()
    rows = _rows()
    probe = run.BoxMarketMakingEnv(run.m6.base_config(base, "0.0002"), rows)
    record = {"dataset_digest": probe.bridge_info["dataset_digest"]}
    probe.close()
    alarm = [(1 if 100 <= r < 200 else 0, None if r < 10 else 2.0) for r in range(len(rows))]
    state = {"alarm": alarm}
    monkeypatch.setattr(run, "read_rows", lambda *a, **k: (rows, record))
    monkeypatch.setattr(run, "read_alarm", lambda *a, **k: state["alarm"])
    return state


def _job(arm, points=None, memory=None, execution="design", policy=None):
    kind = {"no_trade": "hold"}.get(arm, "zero_residual")
    return {"arm": arm, "fitted": "s1", "simulator": "s2", "day": 15, "split": "validation",
            "candidate": arm, "policy": policy or {"kind": kind}, "points": points,
            "fee": "0.0002", "execution": execution, "memory": memory, "data_dir": None}


def test_the_design_pins_the_protocols_it_builds_on():
    design, base = run.load()
    assert design["fee"]["value"] == "-0.0001"
    assert design["execution"]["fill_model"]["per_ms"] == 500
    assert base["training_seeds"] == design["seeds"]["training"]


def test_changes_mark_each_switch_once():
    assert run.changes_of([0, 0, 1, 1, 0, 2, 2]) == [[2, 1], [4, 0], [5, 2]]
    assert run.changes_of([0, 0]) == []
    assert run.changes_of([1, 1, 0]) == [[0, 1], [2, 0]]


def test_the_latent_regime_follows_the_schedule():
    design, _ = run.load()
    day17 = run.latent(design, 17, 86_400)
    assert day17[35_999] == 0 and day17[36_000] == 1 and day17[50_399] == 1 and day17[50_400] == 0
    assert set(run.latent(design, 15, 86_400)) == {0}
    assert run.latent(design, 23, 86_400)[21_600] == run.REGIMES.index("thin")


def test_markouts_sign_buys_and_sells_against_the_later_mid():
    tick = Decimal("0.005")
    rows = [(t * 1_000_000_000, 100, 1, 102, 1) for t in range(12)]
    rows[1] = (1_000_000_000, 104, 1, 106, 1)  # mid 105 one second after the buy at 0
    rows[10] = (10_000_000_000, 96, 1, 98, 1)  # mid 97 ten seconds after it
    positions = [(0, "0.5", "1"), (5_000, "0.5", "0")]
    out = run.markouts(positions, rows, tick)
    # The buy: +1 x (105 - 100) ticks at 1 s and (97 - 100) at 10 s; the sell at 5 s: -1 x
    # (101 - 100) at 1 s, and nothing at 10 s, past the last row.
    assert Decimal(out["1000"]) == (Decimal(105) * tick - Decimal("0.5")) - (Decimal(101) * tick - Decimal("0.5"))
    assert Decimal(out["10000"]) == Decimal(97) * tick - Decimal("0.5")


@pytest.mark.parametrize("arm,memory", [("no_trade", None), ("fixed", None), ("adaptive", None),
                                        ("oracle", None), ("learned", "mlp"), ("learned", "ema"),
                                        ("learned", "history"), ("learned", "lstm"),
                                        ("learned_detector", "mlp"),
                                        ("learned_detector", "history")])
def test_every_arm_builds_and_evaluates_a_day(synthetic_day, arm, memory):
    design, base = run.load()
    points = {"fixed": [POINT], "adaptive": [POINT, WIDE], "oracle": [POINT, WIDE, WIDE]}.get(arm)
    env, rows = run.build_env(_job(arm, points, memory))
    try:
        assert env.unwrapped.config["fill_model"] == design["execution"]["fill_model"]
        if arm in run.LEARNED_ARMS:
            algorithm, policy_name = run.m6.model_class(memory)
            policy = algorithm(policy_name, env, seed=1, device="cpu", n_steps=64, batch_size=32)
        elif arm == "no_trade":
            policy = run.m6.Constant(run.m6.HOLD)
        else:
            policy = run.m6.ZeroResidual()
        day = run.run_episode(policy, env, rows, memory == "lstm", base, 101)
        assert day["rows"] == day["decisions"] == len(rows) - 1
        assert set(day["markouts"]) == {"1000", "10000"}
    finally:
        env.close()


def test_the_immediate_execution_changes_only_the_fill_model(synthetic_day):
    design, _ = run.load()
    a, _ = run.build_env(_job("fixed", [POINT]))
    b, _ = run.build_env(_job("fixed", [POINT], execution="immediate"))
    try:
        assert b.unwrapped.config["fill_model"] == {"kind": "immediate"}
        same = lambda env: {k: v for k, v in env.unwrapped.config.items() if k != "fill_model"}
        assert same(a) == same(b)
    finally:
        a.close()
        b.close()


def test_adaptive_without_an_alarm_is_the_fixed_arm(synthetic_day):
    _, base = run.load()
    synthetic_day["alarm"] = [(0, 1.0)] * 300
    nets = []
    for arm, points in (("fixed", [POINT]), ("adaptive", [POINT, WIDE])):
        env, rows = run.build_env(_job(arm, points))
        try:
            nets.append(run.run_episode(run.m6.ZeroResidual(), env, rows, False, base, 101))
        finally:
            env.close()
    assert nets[0]["net"] == nets[1]["net"] and nets[0]["markouts"] == nets[1]["markouts"]


def test_adaptive_quotes_the_alarm_point_while_the_alarm_is_on(synthetic_day):
    env, _ = run.build_env(_job("adaptive", [POINT, {**POINT, "size": "2"}]))
    try:
        _, info = env.reset(seed=101)
        zero = np.zeros(4, dtype=np.float32)
        while True:
            seen = info["row"]
            _, _, terminated, truncated, info = env.step(zero)
            classical = info["residual"]["classical"]
            if classical["kind"] == "quote":
                assert classical["bid_lots"] == (2 if 100 <= seen < 200 else 1), seen
            if terminated or truncated:
                break
    finally:
        env.close()


def test_the_detector_features_are_those_of_the_observed_row(synthetic_day):
    env, _ = run.build_env(_job("learned_detector", memory="mlp"))
    try:
        obs, info = env.reset(seed=101)
        assert obs[-2:].tolist() == [0.0, 0.0]
        hold = np.asarray(run.m6.HOLD, dtype=np.float32)
        while True:
            obs, _, terminated, truncated, info = env.step(hold)
            row = info["row"]
            alarm, ratio = synthetic_day["alarm"][row]
            expected = [float(alarm), 0.0 if ratio is None else ratio / run.RATIO_CAP]
            assert obs[-2:].tolist() == pytest.approx(expected)
            if terminated or truncated:
                break
        assert env.observation_space.shape[0] == obs.shape[0]
    finally:
        env.close()


def test_the_detector_metrics_count_delays_misses_and_false_alarms(monkeypatch):
    design, _ = run.load()
    rows = 86_400
    alarms = {}
    for day in run.days_of(design, "holdout"):
        alarm = [0] * rows
        if day == 20:
            alarm[1000:1010] = [1] * 10
            alarm[5000:5001] = [1]
        if day == 22:  # stress from 54000: on at 54120
            alarm[54_120:60_000] = [1] * (60_000 - 54_120)
        if day == 23:  # thin from 21600: latched on from before it, never a new onset inside
            alarm[21_000:30_000] = [1] * 9_000
        alarms[day] = [(a, 1.0) for a in alarm]
    monkeypatch.setattr(run, "read_alarm", lambda d, f, s, day, *a: alarms[day])

    class Ctx:
        pass

    ctx = Ctx()
    ctx.design, ctx.data_dir = design, None
    out = run.detector_metrics(ctx, "s1", "s1")
    stress = [c for c in out["changes"] if c["regime"] == "stressed"]
    thin = [c for c in out["changes"] if c["regime"] == "thin"]
    assert stress == [{"day": 22, "regime": "stressed", "start": 54_000, "delay_s": 120,
                       "missed": False, "on_at_start": False}]
    # An alarm already on at the change is not a detection: no onset inside, so missed.
    assert thin[0]["missed"] and thin[0]["delay_s"] is None and thin[0]["on_at_start"]
    # Two onsets over the calm days 20, 21 and 24.
    assert out["false_alarms_per_calm_day"] == pytest.approx(2 / 3)
    assert out["seconds"]["stressed/on"] == 60_000 - 54_120
    assert out["seconds"]["stressed/off"] == 14_400 - (60_000 - 54_120)


def test_the_frozen_alarm_files_leave_out_tagged_sensitivity_files(tmp_path):
    for name in ("day-20.alarm-s1.json", "day-20.alarm-s2.json", "day-20.alarm-s1-wallclock.json"):
        path = tmp_path / "holdout" / "s1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")

    class Ctx:
        data_dir = str(tmp_path)

    found = [p.name for p in run.Context.alarm_files(Ctx())]
    assert found == ["day-20.alarm-s1.json", "day-20.alarm-s2.json"]


def test_the_holdout_is_refused_outside_its_step(tmp_path):
    design, _ = run.load()
    with pytest.raises(PermissionError):
        run.read_rows(str(tmp_path), "s1", 20, design)
    with pytest.raises(PermissionError):
        run.read_alarm(str(tmp_path), "s1", "s1", 22, design)


def test_the_summary_takes_the_worst_day_as_the_cvar():
    episode = {"attribution": {"max_drawdown": "1", "quote_fees": "0", "spread_capture": "0",
                               "inventory_pnl": "0"},
               "markouts": {"1000": "0", "10000": "0"}, "peak_notional": "5",
               "limit_breach": False, "orders": {"fills": 1}}
    rows = [{"arm": "fixed", "fitted": "s1", "simulator": "s1", "execution": "design",
             "fee": "f", "day": d, "daily_value": v, "episodes": [episode]}
            for d, v in zip(range(20, 25), ["3", "-2", "1", "0", "5"])]
    out = run.summary(rows, "fixed", "s1", "s1", "design", "f", list(range(20, 25)))
    assert out["cvar_5"] == "-2"
    assert Decimal(out["mean_daily_net"]) == Decimal("1.4")
