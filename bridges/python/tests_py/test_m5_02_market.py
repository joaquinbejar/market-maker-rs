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


def test_every_encoding_runs_the_design_execution():
    design, base = run_market.load()
    for encoding in ("maskable", "residual"):
        config = run_market.config_of(design, base, encoding)
        assert config["fill_model"] == design["execution"]["fill_model"]
        assert config["latency"] == design["execution"]["latency"]
        assert config["fee_rate"] == base["execution"]["fee_rate"]
        assert config["limits"] == base["execution"]["limits"]


def test_design_widths_follow_the_parameter_rule():
    design = json.loads((EXPERIMENT / "design.json").read_text())
    for name, (width, count) in run_market.check_widths(design).items():
        target = design["parameters"]["target"]
        assert abs(count - target) <= design["parameters"]["tolerance"] * target
