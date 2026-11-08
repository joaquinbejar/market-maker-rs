"""A residual policy through the bridge. All data here is synthetic."""

import json
from decimal import Decimal

import numpy as np
import pytest
from conftest import config, walk
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from mm_rl_bridge import BridgeFailure, ResidualMarketMakingEnv
from mm_rl_bridge._native import RESIDUAL_VERSION
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
    # A hold is allowed (a decision interval holds between decisions); a quote is not.
    with pytest.raises(BridgeFailure, match="grid action"):
        env._native.step_quote(0, 0, 1, 0)
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


GLFT = {
    "kind": "glft",
    "risk_aversion": "0.01",
    "order_intensity_k": "0.3",
    "intensity_scale_a": "1.5",
    "max_inventory_lots": 10,
    "terminal_time": 100_000_000,
    "min_spread": "0.5",
    "volatility": "0.2",
    "session_start_ms": 0,
}
OTHER_RESIDUALS = {
    "price bound": lambda r: {**r, "price_ticks": 40},
    "size bound": lambda r: {**r, "size_lots": 2},
    "default size": lambda r: {**r, "default_order_size": "2"},
    "baseline parameter": lambda r: {**r, "baseline": {**r["baseline"], "risk_aversion": "0.2"}},
    "baseline kind": lambda r: {**r, "baseline": GLFT},
    "one number written two ways": lambda r: {
        **r,
        "baseline": {**r["baseline"], "volatility": "0.20"},
    },
}


@pytest.fixture(scope="module")
def residual_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("residual") / "model"
    env = env_on()
    train("sac_residual", env, 2, 32, path, "abc", SAC_SMALL)
    env.close()
    return path


@pytest.mark.parametrize("change", OTHER_RESIDUALS.values(), ids=OTHER_RESIDUALS.keys())
def test_loading_refuses_another_residual_configuration(residual_checkpoint, monkeypatch, change):
    settings = residual_config()
    settings["residual"] = change(settings["residual"])
    other = env_on(settings=settings)
    try:
        # The specification and the execution rules alone cannot tell them apart.
        _, metadata = load_for_inference(residual_checkpoint, env_on())
        assert tuple(metadata["spec"]) == other.spec_id
        assert metadata["kernel_fingerprint"] == other.bridge_info["kernel_fingerprint"]

        def never(*_args, **_kwargs):
            raise AssertionError("loaded the zip, which runs code, before refusing it")

        # The refusal comes before the zip is opened.
        monkeypatch.setattr(SAC, "load", never)
        with pytest.raises(ValueError, match="another residual configuration"):
            load_for_inference(residual_checkpoint, other)
    finally:
        other.close()


def never_load(*_args, **_kwargs):
    raise AssertionError("loaded the zip, which runs code, before refusing it")


def test_loading_refuses_another_composition_version(residual_checkpoint, tmp_path, monkeypatch):
    path = tmp_path / "model"
    for suffix in (".zip", ".json"):
        path.with_suffix(suffix).write_bytes(residual_checkpoint.with_suffix(suffix).read_bytes())
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["residual_version"] == RESIDUAL_VERSION
    env = env_on()
    try:
        for version in (RESIDUAL_VERSION + 1, None):
            metadata["residual_version"] = version
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with monkeypatch.context() as patch:
                patch.setattr(SAC, "load", never_load)
                with pytest.raises(ValueError, match="another version"):
                    load_for_inference(path, env)
        metadata["residual_version"] = RESIDUAL_VERSION
        path.with_suffix(".json").write_text(json.dumps(metadata))
        _, loaded = load_for_inference(path, env)
        assert loaded == metadata
    finally:
        env.close()


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


def scheduled_config(changes):
    settings = residual_config()
    plain = settings["residual"]["baseline"]
    settings["residual"]["baseline"] = {
        "kind": "scheduled",
        "points": [plain, dict(plain, size="2")],
        "changes": changes,
    }
    return settings


def test_a_scheduled_baseline_quotes_from_the_point_of_the_row_it_observed():
    env = env_on(settings=scheduled_config([[50, 1], [120, 0]]))
    _, info = env.reset(seed=1)
    zero = np.zeros(4, dtype=np.float32)
    sizes = {}
    while True:
        seen = info["row"]
        _, _, terminated, truncated, info = env.step(zero)
        classical = info["residual"]["classical"]
        if classical["kind"] == "quote":
            expected = 2 if 50 <= seen < 120 else 1
            assert classical["bid_lots"] == expected, (seen, classical)
            sizes[expected] = sizes.get(expected, 0) + 1
        if terminated or truncated:
            break
    env.close()
    assert sizes.get(1, 0) > 0 and sizes.get(2, 0) > 0


def test_a_schedule_without_changes_is_its_first_point():
    plain, scheduled = env_on(), env_on(settings=scheduled_config([]))
    plain.reset(seed=4)
    scheduled.reset(seed=4)
    zero = np.zeros(4, dtype=np.float32)
    while True:
        a = plain.step(zero)
        b = scheduled.step(zero)
        assert a[1:4] == b[1:4]
        assert a[4]["residual"] == b[4]["residual"] and a[4]["net"] == b[4]["net"]
        if a[2] or a[3]:
            break
    assert plain.attribution() == scheduled.attribution()
    plain.close()
    scheduled.close()


@pytest.mark.parametrize("changes", [[[5, 2]], [[5, 0]], [[5, 1], [5, 0]], [[5, 1], [6, 1]]])
def test_a_malformed_schedule_is_refused(changes):
    with pytest.raises(ValueError, match="scheduled baseline"):
        env_on(settings=scheduled_config(changes))
