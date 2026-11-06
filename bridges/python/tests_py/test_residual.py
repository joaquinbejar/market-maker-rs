"""A residual policy through the bridge. All data here is synthetic."""

from decimal import Decimal

import numpy as np
import pytest
from conftest import config, walk
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from mm_rl_bridge import BridgeFailure, ResidualMarketMakingEnv
from mm_rl_bridge.training import (
    Protocol,
    ZeroResidual,
    evaluate,
    load_for_inference,
    train,
)

SAC_SMALL = {"learning_starts": 16, "batch_size": 16, "buffer_size": 512}


def residual_config(max_steps=1000, price_ticks=4, size_lots=1):
    settings = config(policy="reject", max_steps=max_steps)
    settings["residual"] = {
        "baseline": {
            "kind": "avellaneda_stoikov",
            "risk_aversion": "0.1",
            "volatility": "0.2",
            "order_intensity": "1.5",
            "session_start_ms": 0,
            "horizon_ms": 100_000_000,
            "size": "1",
        },
        "price_ticks": price_ticks,
        "size_lots": size_lots,
        "default_order_size": "1",
    }
    return settings


def env_on(rows=None, settings=None):
    return ResidualMarketMakingEnv(settings or residual_config(), rows or walk())


def test_a_residual_environment_needs_its_configuration():
    with pytest.raises(ValueError, match="residual configuration"):
        ResidualMarketMakingEnv(config(policy="reject"), walk())


def test_interface_validator_accepts_it():
    env = env_on()
    sb3_check_env(env, warn=True)
    env.close()


def test_each_action_dimension_reaches_its_side_at_both_edges():
    moves = {
        0: lambda c, q, v: q["bid"] == max(c["bid"] - v * 4, 1),
        1: lambda c, q, v: q["ask"] == c["ask"] + v * 4,
        2: lambda c, q, v: q["bid_lots"] == max(c["bid_lots"] + v, 1),
        3: lambda c, q, v: q["ask_lots"] == max(c["ask_lots"] + v, 1),
    }
    for slot, check in moves.items():
        for value in (-1, 1):
            env = env_on()
            env.reset(seed=1)
            action = np.zeros(4, dtype=np.float32)
            action[slot] = value
            _, _, _, _, info = env.step(action)
            log = info["residual"]
            classical, composed = log["classical"], log["composed"]
            if log["fallback"] is None:
                assert check(classical, composed, value), (slot, value, log)
            else:
                assert composed == classical
            assert info["proposed_action"] == action.tolist()
            env.close()


def test_zero_residual_equals_the_classical_action_at_every_decision():
    env = env_on()
    env.reset(seed=1)
    done = False
    while not done:
        _, _, a, b, info = env.step(np.zeros(4, dtype=np.float32))
        log = info["residual"]
        assert log["composed"] == log["classical"]
        assert log["delta"] == (0, 0, 0, 0)
        assert log["clips"] == (False, False, False)
        assert log["fallback"] is None
        done = a or b
    assert Decimal(env.attribution()["net_pnl"]) == Decimal(info["net"])
    env.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_value_that_is_not_finite_falls_back_and_goes_on(bad):
    env = env_on()
    env.reset(seed=1)
    for slot in range(4):
        action = np.zeros(4, dtype=np.float64)
        action[slot] = bad
        _, _, _, _, info = env.step(action)
        assert info["residual"]["fallback"] == "NonFinite"
        assert info["residual"]["composed"] == info["residual"]["classical"]
    env.close()


def test_a_value_beyond_its_bound_is_clipped_and_logged():
    env = env_on()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([5.0, 0.0, 0.0, 0.0]))
    assert info["residual"]["delta"][0] == 4
    assert info["residual"]["clips"][0] is True
    env.close()


def test_a_grid_action_is_not_a_residual_and_ends_the_environment():
    env = env_on()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="4 values"):
        env.step(np.zeros(5))
    with pytest.raises(BridgeFailure, match="grid action"):
        env._native.step_hold()
    with pytest.raises(BridgeFailure, match="closed"):
        env.step(np.zeros(4))


def test_limits_hold_for_any_output():
    settings = residual_config(price_ticks=40, size_lots=9)
    settings["limits"] = {"max_position": "3", "max_notional": "100000", "scaling_factor": "1"}
    free_settings = residual_config(price_ticks=40, size_lots=9)
    rng = np.random.default_rng(5)
    capped, free = env_on(settings=settings), env_on(settings=free_settings)
    capped.reset(seed=1)
    free.reset(seed=1)
    largest_capped, largest_free = Decimal(0), Decimal(0)
    done = False
    while not done:
        action = rng.choice([-1.0, 1.0, 1e300, -1e300, float("nan")], size=4)
        action[0], action[1] = -1.0, -1.0
        _, _, a, b, info = capped.step(action)
        _, _, _, _, open_info = free.step(action)
        largest_capped = max(largest_capped, abs(Decimal(info["raw_observation"][6])))
        largest_free = max(largest_free, abs(Decimal(open_info["raw_observation"][6])))
        done = a or b
    assert largest_capped <= 3
    assert largest_free > 3
    capped.close()
    free.close()


def test_checkpoint_reproduces_inference_on_saved_observations(tmp_path):
    env = env_on()
    path = tmp_path / "model"
    train("sac_residual", env, 2, 64, path, "abc", SAC_SMALL)
    model, _ = load_for_inference(path, env)
    observation, _ = env.reset(seed=4)
    saved, proposed = [], []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        saved.append(observation.copy())
        proposed.append(action.copy())
        observation, _, a, b, _ = env.step(action)
        done = a or b
    env.close()
    # A fresh process would have only the file and the rows: no environment steps here.
    other = env_on()
    again, metadata = load_for_inference(path, other)
    other.close()
    assert metadata["config"]["residual"]["price_ticks"] == 4
    for observation, action in zip(saved, proposed, strict=True):
        replayed, _ = again.predict(observation, deterministic=True)
        assert np.array_equal(replayed, action)


def test_protocol_compares_the_policy_with_its_zero_baseline_on_the_same_window(tmp_path):
    rows = walk(600)
    ns = lambda ms: ms * 1_000_000  # noqa: E731
    settings = residual_config()
    protocol = Protocol(
        settings,
        "sac_residual",
        rows,
        ns(1000 + 300 * 25),
        ns(1000 + 450 * 25),
        tmp_path,
        "abc",
    )
    protocol.train(1, 64, SAC_SMALL)
    table = protocol.select([1])
    assert table[0]["results"][0].attribution["net_pnl"] == str(table[0]["results"][0].net)
    env = ResidualMarketMakingEnv(settings, rows[300:450])
    baseline = evaluate(ZeroResidual, env, [1])
    env.close()
    assert baseline[0].steps == table[0]["results"][0].steps == 150
    assert baseline[0].attribution["fills"] > 0
    for field in ("spread_capture", "inventory_pnl", "quote_fees", "base_fees", "net_pnl"):
        assert field in baseline[0].attribution
