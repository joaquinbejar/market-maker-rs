"""The window over the bridge's observation and the runner of parts B and C of #107.

All data here is synthetic.
"""

import importlib.util
import json
import pathlib
import sys
from decimal import Decimal

import gymnasium as gym
import numpy as np
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m5_02"
sys.path.insert(0, str(EXPERIMENT))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m5_02_{name}", EXPERIMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


window = _load("window")
nets = _load("nets")
run_market = _load("run_market")


class Scripted(gym.Env):
    """Observations `values = step + feature / 100`, flags 1 except mid change at reset."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(22,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 5, 5, 2, 2])
        self.t = 0

    def _obs(self):
        values = np.asarray([self.t + f / 100 for f in range(11)], dtype=np.float32)
        flags = np.ones(11, dtype=np.float32)
        if self.t == 0:
            flags[1] = 0.0
            values[1] = 0.0
        return np.concatenate([values, flags])

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(16, dtype=bool)


def _split(env, obs):
    lay = nets.Layout(**window.layout_dims(env.buffer))
    return [x[0].numpy() for x in lay.split(torch.as_tensor(obs[None, :]))]


def test_the_window_holds_past_market_features_newest_first():
    env = window.WindowEnv(Scripted(), buffer=4)
    env.reset()
    for _ in range(5):
        obs, *_ = env.step(np.zeros(5))
    ages, valid, values, scalars = _split(env, obs)
    assert ages.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert valid.tolist() == [1.0] * 4
    # Newest is step 5, then 4, 3, 2; channel 0 is the spread's value, channel 6 its flag.
    assert values[:, 0].tolist() == pytest.approx([5.0, 4.0, 3.0, 2.0])
    assert values[:, 6].tolist() == [1.0] * 4
    # The agent's state is the current one: position (feature 6) at step 5, and its flags.
    assert scalars[:5].tolist() == pytest.approx([5.06, 5.07, 5.08, 5.09, 5.10], abs=1e-5)
    assert scalars[5:].tolist() == [1.0] * 5


def test_a_missing_feature_is_zero_with_its_flag_and_the_decision_stays_valid():
    env = window.WindowEnv(Scripted(), buffer=4)
    obs, _ = env.reset()
    ages, valid, values, _ = _split(env, obs)
    assert valid.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert values[0, 1] == 0.0 and values[0, 7] == 0.0
    assert values[0, 0] == pytest.approx(0.0) and values[0, 6] == 1.0


def test_reset_clears_the_window():
    env = window.WindowEnv(Scripted(), buffer=4)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(5))
    obs, _ = env.reset()
    _, valid, values, _ = _split(env, obs)
    assert valid.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.all(values[1:] == 0.0)


def test_the_window_refuses_another_observation():
    class Other(Scripted):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-1, 1, shape=(20,), dtype=np.float32)

    with pytest.raises(ValueError):
        window.WindowEnv(Other(), buffer=4)


def test_one_frame_is_the_current_observation_alone():
    env = window.WindowEnv(Scripted(), buffer=4)
    env.reset()
    obs, *_ = env.step(np.zeros(5))
    lay = nets.Layout(**window.layout_dims(4))
    space = gym.spaces.Box(-np.inf, np.inf, shape=(lay.size,), dtype=np.float32)
    out = nets.FrameStack(space, lay, frames=1)(torch.as_tensor(obs[None, :]))[0].numpy()
    raw = Scripted()
    raw.reset()
    current, *_ = raw.step(np.zeros(5))
    market = np.concatenate([current[:11][:6], current[11:][:6]])
    agent = np.concatenate([current[:11][6:], current[11:][6:]])
    assert out.tolist() == pytest.approx([0.0, *market, *agent], abs=1e-6)


INTERVAL = lambda mean, lower, upper: {"mean": str(mean), "lower": str(lower), "upper": str(upper)}


@pytest.mark.parametrize(
    "against, against_none, breach, hsn_dd, control_dd, expected",
    [
        (INTERVAL(2, 1, 3), INTERVAL(2, 1, 3), False, 1, 1, "hsn_helps"),
        (INTERVAL(2, 1, 3), INTERVAL(0.1, -1, 1), False, 1, 1, "revise"),
        (INTERVAL(2, 1, 3), INTERVAL(2, 1, 3), True, 1, 1, "revise"),
        (INTERVAL(2, 1, 3), INTERVAL(2, 1, 3), False, Decimal("1.3"), 1, "revise"),
        (INTERVAL(2, 1, 3), INTERVAL(2, 1, 3), False, Decimal("1.2"), 1, "hsn_helps"),
        (INTERVAL(1, -1, 3), INTERVAL(2, 1, 3), False, 1, 1, "revise"),
        (INTERVAL(0, -1, 1), INTERVAL(2, 1, 3), False, 1, 1, "no_gain"),
        (INTERVAL(-1, -2, 0), INTERVAL(2, 1, 3), False, 1, 1, "no_gain"),
        # Above the control but wholly below no-trade is not a gain.
        (INTERVAL(2, 1, 3), INTERVAL(-2, -3, -1), False, 1, 1, "revise"),
    ],
)
def test_the_decision_rule(against, against_none, breach, hsn_dd, control_dd, expected):
    out = run_market.decide(against, against_none, breach, Decimal(hsn_dd), Decimal(control_dd),
                            Decimal("1.2"))
    assert out["decision"] == expected


def test_rows_refuse_the_holdout_and_a_wrong_split(tmp_path):
    digests = {("calm", 20): {"split": "holdout", "group": "calm", "day": 20},
               ("calm", 3): {"split": "train", "group": "calm", "day": 3}}
    with pytest.raises(PermissionError):
        run_market.read_rows("market", tmp_path, digests, "calm", 20, "holdout")
    with pytest.raises(ValueError):
        run_market.read_rows("market", tmp_path, digests, "calm", 3, "validation")


def test_design_widths_follow_the_parameter_rule():
    design, base = run_market.load()
    for name, (width, count) in run_market.check_widths(design, base).items():
        target = design["parameters"]["target"]
        assert abs(count - target) <= design["parameters"]["tolerance"] * target


def test_the_design_pins_the_protocol_it_builds_on():
    design, base = run_market.load()
    assert base["issue"] == 103
    assert set(design["arms"]) == {"mlp", "ema", "history", "lstm", "gru_window", "hsn"}
    assert all(design["arms"][a]["from"] == "#103, verbatim" for a in ("mlp", "ema", "history", "lstm"))


def test_hsn_and_the_window_gru_see_the_current_observation():
    lay = nets.Layout(**window.layout_dims(4))
    space = gym.spaces.Box(-np.inf, np.inf, shape=(lay.size,), dtype=np.float32)
    env = window.WindowEnv(Scripted(), buffer=4)
    env.reset()
    obs, *_ = env.step(np.zeros(5))
    x = torch.as_tensor(obs[None, :])
    _, _, values, scalars = lay.split(x)
    hsn = nets.Hsn(space, lay, alphas=[0.2, 0.4], betas=[0.3, 0.08], floor=1e-4, with_current=True)
    out = hsn(x)[0]
    p = 2 * lay.channels
    assert out[p : p + lay.channels].tolist() == pytest.approx(values[0, 0].tolist())
    assert out[p + lay.channels :].tolist() == pytest.approx(scalars[0].tolist())
    torch.manual_seed(0)
    gru = nets.Recurrent(space, lay, cell="gru", embed=3, hidden=5, with_current=True)
    out = gru(x)[0]
    assert out[5 : 5 + lay.channels].tolist() == pytest.approx(values[0, 0].tolist())


def _rows(count=300):
    """A synthetic touch that walks a tick now and then, in kernel ticks and lots."""
    rng = np.random.default_rng(0)
    mid, rows = 10000, []
    for i in range(count):
        mid += int(rng.integers(-1, 2))
        rows.append((i * 1_000_000_000, mid - 1, 3, mid + 1, 4))
    return rows


@pytest.fixture
def synthetic_day(monkeypatch):
    design, base = run_market.load()
    rows = _rows()
    probe = run_market.BoxMarketMakingEnv(run_market.m6.base_config(base, "0.0002"), rows)
    digest = probe.bridge_info["dataset_digest"]
    probe.close()
    record = {"split": "validation", "group": "calm", "day": 15, "dataset_digest": digest}
    monkeypatch.setattr(run_market, "read_rows", lambda *args, **kwargs: (rows, record))
    return {("calm", 15): record}


@pytest.mark.parametrize("arm", ["mlp", "ema", "history", "lstm", "gru_window", "hsn",
                                 "avellaneda_stoikov"])
def test_every_arm_builds_steps_and_evaluates_a_day(synthetic_day, arm):
    design, base = run_market.load()
    job = {"part": "market", "arm": arm, "fee": "0.0002", "data_dir": None,
           "point": {"risk_aversion": "0.01", "volatility": "0.5", "order_intensity": "50",
                     "horizon_ms": 86400000, "size": "1"}}
    env = run_market.build_env(job, synthetic_day, "calm", 15, "validation")
    try:
        assert env.unwrapped.config["fill_model"] == design["execution"]["fill_model"]
        assert env.unwrapped.config["fill_model"]["min_queue_ms"] == 0
        if arm in run_market.WINDOWED:
            assert env.observation_space.shape == (run_market.layout(design).size,)
            model = run_market.model_of(design, base, arm, env, seed=1)
            assert nets.parameter_count(model.policy) == pytest.approx(
                design["parameters"]["target"], rel=design["parameters"]["tolerance"])
            policy = model
        elif arm == "avellaneda_stoikov":
            policy = run_market.m6.ZeroResidual()
        else:
            policy = run_market.model_of(design, base, arm, env, seed=1)
        day = run_market.m6.episode(policy, env, 101, arm == "lstm", base)
        # One decision per row but the last, which only ends the day.
        assert day["rows"] == day["decisions"] == len(_rows()) - 1
    finally:
        env.close()


def test_the_window_starts_empty_on_every_day(synthetic_day):
    design, _ = run_market.load()
    job = {"part": "market", "arm": "hsn", "fee": "0.0002", "data_dir": None}
    env = run_market.build_env(job, synthetic_day, "calm", 15, "validation")
    try:
        lay = run_market.layout(design)
        env.reset(seed=101)
        for _ in range(40):
            env.step(np.asarray(run_market.m6.HOLD, dtype=np.float32))
        obs, _ = env.reset(seed=101)
        _, valid, _, _ = lay.split(torch.as_tensor(obs[None, :]))
        assert valid[0].sum().item() == 1.0
    finally:
        env.close()


def test_an_evaluation_is_claimed_once_per_freeze(tmp_path):
    attempt = _load("attempt")
    marker = tmp_path / "holdout.attempt"
    first = attempt.claim(marker, "abc")
    assert [r["kind"] for r in first] == ["attempt"] and first[0]["frozen_sha256"] == "abc"
    # A second attempt, after a crash before the result was written, stops.
    with pytest.raises(SystemExit, match="already started"):
        attempt.claim(marker, "abc")
    # The explicit recovery appends its reason, and the result carries every attempt.
    again = attempt.claim(marker, "abc", recover="the pool died at job 3")
    assert [r["kind"] for r in again] == ["attempt", "recovery"]
    assert again[1]["reason"] == "the pool died at job 3"
    assert attempt.recovery_reason(["holdout", "x", "--recover", "why"]) == "why"
    assert attempt.recovery_reason(["holdout", "x"]) is None
    with pytest.raises(SystemExit):
        attempt.recovery_reason(["holdout", "--recover"])


def test_the_holdout_claims_its_attempt_before_dispatching(tmp_path, monkeypatch):
    """A failure inside the holdout's pool leaves the marker, so a rerun with the same freeze
    stops before evaluating anything."""
    frozen = tmp_path / "market" / "frozen.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}")

    class Ctx:
        part, out, fee, secondary_fee = "market", tmp_path / "market", "f", "f"
        design, base = {}, {}

        def runs(self):
            return []

        def reference_jobs(self, *a):
            return [{"job": 1}]

        def learned_jobs(self, *a):
            return []

    monkeypatch.setattr(run_market, "sha256", lambda path: "same")
    monkeypatch.setattr(run_market.json, "loads", lambda text: {
        "design_sha256": "same", "validation_sha256": "same", "runs": {}})
    monkeypatch.setattr(run_market, "days_of", lambda *a: [20])
    dispatched = []

    def crash(jobs, *a):
        dispatched.append(len(jobs))
        raise RuntimeError("the pool died")

    monkeypatch.setattr(run_market.m6, "pool", crash)
    with pytest.raises(RuntimeError):
        run_market.holdout(Ctx(), "same", 1)
    assert dispatched == [1] and (tmp_path / "market" / "holdout.attempt").exists()
    with pytest.raises(SystemExit, match="already started"):
        run_market.holdout(Ctx(), "same", 1)
    assert dispatched == [1]
