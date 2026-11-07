"""The environments against the specification and the hand computed trajectory."""

import gc
import resource

import gymnasium as gym
import numpy as np
import pytest
from conftest import LONG_ROWS, config, row, walk
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from mm_rl_bridge import (
    BoxMarketMakingEnv,
    BridgeFailure,
    MaskableMarketMakingEnv,
    NativeEnv,
)
import json
import sys

# `ru_maxrss` is in bytes on macOS and in kibibytes on Linux.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def test_hand_computed_trajectory_through_gymnasium():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS)
    observation, info = env.reset(seed=1)
    assert info["row"] == 0
    # Spread 2 ticks of 10; no mid change or row gap at the first decision.
    assert observation[0] == pytest.approx(0.2)
    assert observation.dtype == np.float32
    assert observation.shape == (22,)
    assert list(observation[11:13]) == [1.0, 0.0]

    observation, reward, terminated, truncated, info = env.step([1, 1, 2, 1, 0])
    assert reward == pytest.approx(0.801)
    assert info["reward_exact"] == "0.801"
    assert info["applied_action"] == {"kind": "quote", "indices": [1, 2, 1, 0]}
    assert info["refused_because"] is None
    assert not terminated and not truncated

    _, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    assert info["reward_exact"] == "-2"
    assert not terminated and not truncated

    _, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    assert info["reward_exact"] == "1.8"
    assert info["ledger_change"] == "3"
    assert info["terminal_close"] == "-1.2"
    assert info["net"] == "0.601"
    assert info["close"] == {
        "side": "Sell",
        "price": "100",
        "quantity": "2",
        "fee": "0.2",
        "beyond_touch": "0",
    }
    assert terminated and not truncated
    # The run is over, so only a hold is on offer.
    assert env.action_masks()[:2].tolist() == [True, False]
    env.close()


def test_attribution_reports_the_position_and_drawdown_of_the_hand_trajectory():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS)
    env.reset(seed=1)
    env.step([1, 1, 2, 1, 0])
    env.step([0, 0, 0, 0, 0])
    env.step([0, 0, 0, 0, 0])
    a = env.attribution()
    # Two lots bought at 99.5 at 2 s, sold at 100 by the close at 3 s.
    assert a["positions"] == [(2000, "99.5", "2"), (3000, "100", "0")]
    assert (a["max_abs_position"], a["final_position"]) == ("2", "0")
    # Two lots held for 1 s of the 2 s between the first and the last mark.
    assert a["mean_abs_position"] == "1"
    # Cash -199.199 after the buy and its fee, and 2 x 99 at the mark of 2 s.
    assert a["max_drawdown"] == "1.199"
    assert a["window_ms"] == [1000, 3001]
    env.close()


def test_truncation_is_not_termination():
    env = MaskableMarketMakingEnv(config(max_steps=2), LONG_ROWS)
    env.reset(seed=1)
    env.step([1, 1, 2, 1, 0])
    _, _, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    assert truncated and not terminated
    assert info["close"] is None
    assert info["net"] == "-1.199"
    env.close()


def test_vector_env_contract_done_truncated_flag_and_terminal_observation():
    for max_steps, truncated in ((2, True), (100, False)):
        env = MaskableMarketMakingEnv(config(max_steps=max_steps), LONG_ROWS)
        direct = MaskableMarketMakingEnv(config(max_steps=max_steps), LONG_ROWS)
        direct.reset(seed=1)
        vec = DummyVecEnv([lambda: env])
        vec.seed(1)
        vec.reset()
        hold = np.zeros((1, 5), dtype=np.int64)
        done = False
        while not done:
            observations, _, dones, infos = vec.step(hold)
            last, _, end_a, end_b, _ = direct.step([0, 0, 0, 0, 0])
            done = bool(dones[0])
            assert done == (end_a or end_b)
        assert infos[0]["TimeLimit.truncated"] == truncated
        assert np.array_equal(infos[0]["terminal_observation"], last)
        # A new episode is at its first row.
        assert not np.array_equal(observations[0], last)
        vec.close()
        direct.close()


def test_interface_validators_accept_both_encodings():
    maskable = MaskableMarketMakingEnv(config(), walk())
    sb3_check_env(maskable, warn=True)
    gym_check_env(maskable, skip_render_check=True)
    maskable.close()
    box = BoxMarketMakingEnv(config(policy="reject"), walk())
    sb3_check_env(box, warn=True)
    gym_check_env(box, skip_render_check=True)
    box.close()


def test_mask_shape_and_values_follow_the_grid():
    grid = config()
    grid["bid_offsets"] = [1, 500, 60000]
    grid["sizes"] = [1, 11]
    env = MaskableMarketMakingEnv(grid, LONG_ROWS)
    env.reset(seed=1)
    masks = env.action_masks()
    assert masks.tolist() == [
        True, True,
        True, False, False,
        True, True, True,
        True, False,
        True, False,
    ]
    assert list(env.action_space.nvec) == [2, 3, 3, 2, 2]
    env.close()


def test_mask_offers_only_a_hold_when_a_dimension_is_empty():
    grid = config()
    grid["sizes"] = [11, 12]
    env = MaskableMarketMakingEnv(grid, LONG_ROWS)
    env.reset(seed=1)
    masks = env.action_masks()
    assert masks[:2].tolist() == [True, False]
    # The dimension left open has something to sample.
    assert masks[8:].all()
    _, _, _, _, info = env.step(_sample_masked(env))
    assert info["applied_action"] == {"kind": "hold", "indices": []}
    env.close()


def _sample_masked(env):
    masks = env.action_masks()
    action, start = [], 0
    for count in env.action_space.nvec:
        action.append(int(np.flatnonzero(masks[start : start + count])[0]))
        start += count
    return action


def test_mask_policy_ends_the_environment_outside_the_mask_and_nothing_continues():
    grid = config()
    grid["sizes"] = [1, 11]
    env = MaskableMarketMakingEnv(grid, LONG_ROWS)
    env.reset(seed=1)
    with pytest.raises(BridgeFailure, match="outside the mask"):
        env.step([1, 1, 2, 1, 0])
    assert env.is_closed
    with pytest.raises(BridgeFailure, match="closed"):
        env.step([0, 0, 0, 0, 0])
    with pytest.raises(BridgeFailure, match="closed"):
        env.reset(seed=1)


def test_reject_policy_reports_proposed_and_applied_action_and_the_charge():
    grid = config(policy="reject")
    grid["sizes"] = [1, 11]
    env = BoxMarketMakingEnv(grid, LONG_ROWS)
    env.reset(seed=1)
    proposed = np.asarray([1.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    _, reward, _, _, info = env.step(proposed)
    assert info["proposed_action"] == proposed.tolist()
    assert info["decoded_action"] == {"kind": "quote", "indices": [1, 2, 1, 0]}
    assert info["applied_action"] == {"kind": "hold", "indices": []}
    assert info["refused_because"] == "SizeRange"
    assert info["penalty"] == "0.25"
    assert info["penalties"] == "0.25"
    assert info["reward_exact"] == "-0.25"
    assert info["net"] == "0"
    assert reward == pytest.approx(-0.25)
    env.close()


def test_box_bins_reach_both_ends_and_a_quote_needs_a_positive_first_value():
    env = BoxMarketMakingEnv(config(policy="reject"), LONG_ROWS)
    env.reset(seed=1)
    assert env._decode([1.0, -1.0, 1.0, -1.0, 1.0]) == ("quote", (0, 2, 0, 1))
    assert env._decode([0.5, 0.0, -0.34, 0.01, -0.01]) == ("quote", (1, 0, 1, 0))
    assert env._decode([0.0, 1.0, 1.0, 1.0, 1.0]) == ("hold", ())
    assert env._decode([-1.0, 1.0, 1.0, 1.0, 1.0]) == ("hold", ())
    env.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1.5, -1.5])
def test_box_refuses_a_value_that_is_not_finite_or_in_range(bad):
    env = BoxMarketMakingEnv(config(policy="reject"), LONG_ROWS)
    env.reset(seed=1)
    for slot in range(5):
        action = np.zeros(5, dtype=np.float64)
        action[slot] = bad
        with pytest.raises(ValueError, match="not in"):
            env.step(action)
    env.close()


def test_encodings_refuse_the_other_policy():
    with pytest.raises(ValueError, match="mask policy"):
        MaskableMarketMakingEnv(config(policy="reject"), LONG_ROWS)
    with pytest.raises(ValueError, match="reject policy"):
        BoxMarketMakingEnv(config(policy="mask"), LONG_ROWS)


def test_unknown_specification_fails_before_any_step():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS)
    spec = env.spec_id
    env.close()
    same = MaskableMarketMakingEnv(config(), LONG_ROWS, expected_spec=spec)
    same.close()
    wrong = spec[:4] + (f"{int(spec[4], 16) ^ 1:032x}",)
    with pytest.raises(ValueError, match="unknown rl specification"):
        MaskableMarketMakingEnv(config(), LONG_ROWS, expected_spec=wrong)
    later = (spec[0], spec[1], spec[2] + 1, spec[3], spec[4])
    with pytest.raises(ValueError):
        MaskableMarketMakingEnv(config(), LONG_ROWS, expected_spec=later)
    other = config()
    other["scales"]["position"] = "5"
    with pytest.raises(ValueError):
        MaskableMarketMakingEnv(other, LONG_ROWS, expected_spec=spec)


def test_no_decision_in_the_data_is_an_error_at_reset():
    env = MaskableMarketMakingEnv(config(), [])
    with pytest.raises(ValueError, match="no decision"):
        env.reset(seed=1)
    env.close()


def test_seeds_decide_the_latency_and_repeat_exactly():
    jitter = config(max_steps=1000)
    jitter["latency"].update(observation_q=1, submit_q=1, cancel_q=1, ack_q=1, jitter_q=2)
    def run(seed):
        env = MaskableMarketMakingEnv(jitter, walk(), seed=0)
        observation, _ = env.reset(seed=seed)
        trace = [observation.tobytes()]
        done = False
        step = 0
        while not done:
            action = [1, step % 2, step % 2, 0, 0] if step % 3 == 0 else [0, 0, 0, 0, 0]
            observation, _, a, b, info = env.step(action)
            trace.append((observation.tobytes(), info["reward_exact"], info["net"]))
            done = a or b
            step += 1
        env.close()
        return trace
    assert run(5) == run(5)
    assert run(5) != run(6)


def test_reset_without_a_seed_moves_to_the_next_one():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS, seed=41)
    _, first = env.reset()
    _, second = env.reset()
    _, chosen = env.reset(seed=7)
    _, after = env.reset()
    assert (first["seed"], second["seed"], chosen["seed"], after["seed"]) == (41, 42, 7, 8)
    env.close()


def test_close_ends_the_thread_and_frees_the_memory():
    # 400 000 rows are about 16 MB in the worker, against a heap that stays put.
    rows = [row(1000 + i, 199, 201) for i in range(400_000)]
    text = json.dumps(config(max_steps=10))
    env = NativeEnv(text, rows, None)
    env.close()
    assert env.is_closed()
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(40):
        env = NativeEnv(text, rows, None)
        env.reset(1)
        env.step_hold()
        env.close()
        assert env.is_closed()
        with pytest.raises(BridgeFailure):
            env.step_hold()
    gc.collect()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # 40 datasets kept would be 640 MB. The peak may move by a few of them.
    assert (after - before) * RSS_UNIT < 160 * 1024 * 1024


def test_observation_space_holds_every_observation():
    env = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    observation, _ = env.reset(seed=3)
    done = False
    step = 0
    while not done:
        assert env.observation_space.contains(observation)
        action = [1, 0, 0, 0, 0] if step % 3 == 0 else [0, 0, 0, 0, 0]
        observation, _, a, b, _ = env.step(action)
        done = a or b
        step += 1
    assert env.observation_space.contains(observation)
    env.close()


def test_a_rational_queue_rate_fills_over_coarse_rows():
    # Rows ten seconds apart, the ask resting at 99.5 with one lot from the second row on, so
    # the bid of 2 at 99.5 waits behind one lot. Immediate fill takes all of it at once; one
    # lot every 9 s clears none for it after 10 s, one after 20 s and the rest after 30 s.
    rows = [row(1_000, 199, 201, 1, 1)] + [
        row(ms, 197, 199, 1, 1) for ms in (11_000, 21_000, 31_000, 41_000)
    ]

    def positions(fill_model):
        cfg = config(terminal="mark")
        cfg["fill_model"] = fill_model
        env = MaskableMarketMakingEnv(cfg, rows)
        env.reset(seed=1)
        out = []
        action = [1, 1, 2, 1, 0]
        for _ in range(3):
            observation, _, terminated, truncated, info = env.step(action)
            # The delivered position over its scale of 4.
            out.append(float(observation[6]) * 4)
            action = [0, 0, 0, 0, 0]
        return out

    assert positions({"kind": "immediate"}) == [2.0, 2.0, 2.0]
    slow = {"kind": "queue_position_rational", "lots": 1, "per_ms": 9_000, "min_queue_ms": 0}
    assert positions(slow) == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        cfg = config()
        cfg["fill_model"] = dict(slow, lots=0)
        MaskableMarketMakingEnv(cfg, rows)
