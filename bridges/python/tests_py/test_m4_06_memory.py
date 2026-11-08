"""Memory hygiene of the comparison of #103: no state crosses episodes, and a reloaded
checkpoint acts as the trained policy. On the conftest walk, not the experiment's data."""

import importlib.util
import pathlib

import numpy as np
from conftest import config, walk
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mm_rl_bridge import BoxMarketMakingEnv
from mm_rl_bridge.wrappers import DecisionInterval, EmaFeatures, history

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_06_run", HERE.parent / "experiments" / "m4_06" / "run.py"
)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

SMALL = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}
DESIGN = {"execution": {"limits": {"max_position": "20", "max_notional": "100000"}}}


def box(every=2):
    return DecisionInterval(BoxMarketMakingEnv(config(policy="reject", max_steps=1000), walk()),
                            every)


def actions(model, env, recurrent, decisions, state=None, start=True):
    observation, _ = env.reset(seed=4)
    out, flags = [], np.asarray([start])
    for _ in range(decisions):
        if recurrent:
            action, state = model.predict(observation, state=state, episode_start=flags,
                                          deterministic=True)
            flags = np.asarray([False])
        else:
            action, _ = model.predict(observation, deterministic=True)
        out.append(np.asarray(action).copy())
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.asarray(out), state


def test_a_recurrent_policy_forgets_the_last_episode_at_its_start():
    env = box()
    model = RecurrentPPO("MlpLstmPolicy", env, seed=1, device="cpu", **SMALL)
    model.learn(64)
    fresh, carried = actions(model, env, True, 20)
    # Starting a new episode with the last episode's state and `episode_start` gives the
    # fresh episode's actions: the state is reset, not carried.
    again, _ = actions(model, env, True, 20, state=carried, start=True)
    assert np.array_equal(fresh, again)
    # Without the flag the carried state would be used, which acts differently here.
    leaked, _ = actions(model, env, True, 20, state=carried, start=False)
    assert not np.array_equal(fresh, leaked)
    env.close()


def test_a_reloaded_checkpoint_acts_as_the_trained_policy(tmp_path):
    for algorithm, policy, recurrent in ((PPO, "MlpPolicy", False),
                                         (RecurrentPPO, "MlpLstmPolicy", True)):
        env = history(EmaFeatures(box()), 3)
        model = algorithm(policy, env, seed=2, device="cpu", **SMALL)
        model.learn(64)
        trained, _ = actions(model, env, recurrent, 30)
        model.save(tmp_path / policy)
        loaded = algorithm.load(tmp_path / f"{policy}.zip", device="cpu")
        reloaded, _ = actions(loaded, env, recurrent, 30)
        assert np.array_equal(trained, reloaded)
        env.close()


def test_an_evaluated_day_does_not_depend_on_the_day_before():
    env = box()
    model = RecurrentPPO("MlpLstmPolicy", env, seed=3, device="cpu", **SMALL)
    model.learn(64)
    first = run.episode(model, env, 9, True, DESIGN)
    second = run.episode(model, env, 9, True, DESIGN)
    for result in (first, second):
        result.pop("decide_microseconds")
    assert first == second and first["decisions"] > 0
    env.close()
