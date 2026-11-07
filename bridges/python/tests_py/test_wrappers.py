"""Decision interval and memory wrappers (#103), on the hand computed rows and a walk."""

from decimal import Decimal

import numpy as np
import pytest
from conftest import LONG_ROWS, config, walk

from mm_rl_bridge import MaskableMarketMakingEnv
from mm_rl_bridge.wrappers import DecisionInterval, EmaFeatures, history

QUOTE = [1, 1, 2, 1, 0]
HOLD = [0, 0, 0, 0, 0]


def test_each_step_reports_the_venue_orders_it_caused():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS)
    env.reset(seed=1)
    _, _, _, _, info = env.step(QUOTE)
    assert info["orders"] == {"accepted": 2, "rejected": 0, "canceled": 0,
                              "cancel_rejected": 0, "fills": 1}
    _, _, _, _, info = env.step([1, 2, 2, 0, 0])
    assert (info["orders"]["accepted"], info["orders"]["canceled"]) == (2, 1)
    env.close()


def test_an_interval_of_one_is_the_environment():
    plain = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    wrapped = DecisionInterval(MaskableMarketMakingEnv(config(max_steps=1000), walk()), 1)
    a, _ = plain.reset(seed=3)
    b, _ = wrapped.reset(seed=3)
    assert np.array_equal(a, b)
    for step in range(40):
        action = QUOTE if step % 3 == 0 else HOLD
        one = plain.step(action)
        two = wrapped.step(action)
        assert np.array_equal(one[0], two[0]) and one[1:4] == two[1:4]
        assert one[4]["reward_exact"] == two[4]["reward_exact"] and two[4]["rows"] == 1
    plain.close()
    wrapped.close()


def test_an_interval_holds_between_decisions_and_sums_the_rows():
    direct = MaskableMarketMakingEnv(config(), LONG_ROWS)
    direct.reset(seed=1)
    first = direct.step(QUOTE)
    second = direct.step(HOLD)
    third = direct.step(HOLD)
    env = DecisionInterval(MaskableMarketMakingEnv(config(), LONG_ROWS), 2)
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(QUOTE)
    # The decision and one hold: the observation, reward and orders of both rows.
    assert np.array_equal(observation, second[0])
    assert reward == pytest.approx(first[1] + second[1])
    total = Decimal(first[4]["reward_exact"]) + Decimal(second[4]["reward_exact"])
    assert Decimal(info["reward_exact"]) == total
    assert info["net"] == second[4]["net"] and info["rows"] == 2
    assert info["orders"]["accepted"] == 2 and info["orders"]["fills"] == 1
    assert not (terminated or truncated)
    # The data ends after one more row: the decision covers only that row.
    _, reward, terminated, _, info = env.step(QUOTE)
    assert terminated and info["rows"] == 1
    # A quote on the last row cannot fill before the data ends: the end is the direct one's.
    assert info["net"] == third[4]["net"]
    direct.close()
    env.close()


def test_an_interval_must_be_a_positive_whole_number_of_rows():
    env = MaskableMarketMakingEnv(config(), LONG_ROWS)
    for every in (0, -1, 1.5):
        with pytest.raises(ValueError):
            DecisionInterval(env, every)
    env.close()


def test_ema_features_start_at_zero_each_episode_and_follow_the_decisions():
    env = EmaFeatures(MaskableMarketMakingEnv(config(max_steps=1000), walk()), (1, 4))
    assert env.observation_space.shape == (env.env.observation_space.shape[0] + 4,)
    first, _ = env.reset(seed=1)
    base = len(first) - 4
    w = [0.5, 1 - 0.5 ** 0.25]
    expected = [w[0] * first[1], w[1] * first[1], w[0] * first[5], w[1] * first[5]]
    assert first[base:] == pytest.approx(expected)
    second, *_ = env.step(HOLD)
    after = [expected[0] + w[0] * (second[1] - expected[0]),
             expected[1] + w[1] * (second[1] - expected[1])]
    assert second[base:base + 2] == pytest.approx(after)
    # A new episode forgets the last: its first averages are its first observation's.
    again, _ = env.reset(seed=1)
    assert np.array_equal(again, first)
    assert env.observation_space.contains(second)
    env.close()


def test_history_is_oldest_first_and_zero_before_the_first_decision():
    env = history(MaskableMarketMakingEnv(config(max_steps=1000), walk()), 3)
    width = env.unwrapped.observation_space.shape[0]
    assert env.observation_space.shape == (3 * width,)
    first, _ = env.reset(seed=1)
    assert not first[: 2 * width].any() and np.array_equal(first[2 * width:], first[-width:])
    second, *_ = env.step(HOLD)
    assert np.array_equal(second[width: 2 * width], first[2 * width:])
    env.step(HOLD)
    env.step(HOLD)
    again, _ = env.reset(seed=1)
    assert np.array_equal(again, first)
    env.close()


def test_an_interval_is_the_decision_then_direct_holds_on_a_walk():
    every = 3
    direct = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    wrapped = DecisionInterval(MaskableMarketMakingEnv(config(max_steps=1000), walk()), every)
    direct.reset(seed=5)
    wrapped.reset(seed=5)
    held_fills = 0
    for decision in range(60):
        action = QUOTE if decision % 2 == 0 else [1, 0, 0, 1, 1]
        observation, reward, terminated, truncated, info = wrapped.step(action)
        rows = [direct.step(action)]
        while len(rows) < every and not (rows[-1][2] or rows[-1][3]):
            rows.append(direct.step(HOLD))
        held_fills += sum(r[4]["orders"]["fills"] for r in rows[1:])
        assert np.array_equal(observation, rows[-1][0]) and info["rows"] == len(rows)
        assert reward == pytest.approx(sum(r[1] for r in rows))
        assert Decimal(info["reward_exact"]) == sum(Decimal(r[4]["reward_exact"]) for r in rows)
        for kind in info["orders"]:
            assert info["orders"][kind] == sum(r[4]["orders"][kind] for r in rows)
        if terminated or truncated:
            break
    # Fills happened between decisions, so the sums above were tested on them.
    assert held_fills > 0
    direct.close()
    wrapped.close()


def test_an_interval_holds_a_residual_policys_orders_between_its_decisions():
    from mm_rl_bridge import ResidualMarketMakingEnv
    from test_residual import residual_config

    env = DecisionInterval(ResidualMarketMakingEnv(residual_config(), walk()), 4)
    env.reset(seed=2)
    zero = np.zeros(4, dtype=np.float32)
    placed = 0
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(zero)
        placed += info["orders"]["accepted"]
        assert info["rows"] == 4 or terminated or truncated
        if terminated or truncated:
            break
    assert placed > 0
    env.close()
