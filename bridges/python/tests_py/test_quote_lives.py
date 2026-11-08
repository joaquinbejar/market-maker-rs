"""Quotes that expire at the venue, through the bridge (#297). All data here is synthetic."""

import pytest
from conftest import LONG_ROWS, config, row

from mm_rl_bridge import BoxMarketMakingEnv, ExpiringQuote, MaskableMarketMakingEnv
from mm_rl_bridge.envs import ResidualMarketMakingEnv

MS = 1_000_000
BID = [1, 1, 2, 1, 0]


def living(lives, policy="mask", cancel=False):
    cfg = config(policy=policy)
    cfg["quote_lives_ns"] = list(lives)
    cfg["cancel_action"] = cancel
    return cfg


def crossing(index=2, count=8):
    return [
        row(1000 * (i + 1), 197, 199) if i == index else row(1000 * (i + 1), 200, 202)
        for i in range(count)
    ]


def filled(lives, actions):
    env = MaskableMarketMakingEnv(living(lives), crossing())
    env.reset(seed=1)
    fills = 0
    for action in actions:
        _, _, _, _, info = env.step(action)
        fills += info["orders"]["fills"]
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step_hold()
        fills += info["orders"]["fills"]
        done = terminated or truncated
    env.close()
    return fills


def test_lives_alone_with_cancel_and_none_are_different_spaces():
    plain = MaskableMarketMakingEnv(config(), LONG_ROWS)
    alone = MaskableMarketMakingEnv(living([MS]), LONG_ROWS)
    both = MaskableMarketMakingEnv(living([MS], cancel=True), LONG_ROWS)
    assert plain.spec_id[1] == 2 and alone.spec_id[1] == 5 and both.spec_id[1] == 5
    assert alone.spec_id[4] != both.spec_id[4]
    assert plain.bridge_info["quote_lives_ns"] == []
    assert alone.bridge_info["quote_lives_ns"] == [MS]
    with pytest.raises(Exception, match="refused|versions|specification"):
        MaskableMarketMakingEnv(config(), LONG_ROWS, expected_spec=alone.spec_id)
    for env in (plain, alone, both):
        env.close()


def test_the_maskable_life_is_a_last_dimension_with_no_expiry_first():
    env = MaskableMarketMakingEnv(living([MS, 2 * MS]), LONG_ROWS)
    env.reset(seed=1)
    assert list(env.action_space.nvec) == [2, 3, 3, 2, 2, 3]
    mask = env.action_masks()
    assert len(mask) == sum(env.action_space.nvec)
    assert list(mask[-3:]) == [True, True, True]
    assert env._decode([*BID, 0]) == ("quote", (1, 2, 1, 0))
    assert env._decode([*BID, 1]) == ("expiring_quote", (1, 2, 1, 0, 0))
    assert env._decode([*BID, 2]) == ("expiring_quote", (1, 2, 1, 0, 1))
    assert env._decode([0, 1, 2, 1, 0, 2]) == ("hold", ())
    env.close()


@pytest.mark.parametrize(
    "value,kind,life",
    [
        (-1.0, "quote", None),
        (-1.0 / 3.0 - 1e-6, "quote", None),
        (-1.0 / 3.0 + 1e-6, "expiring_quote", 0),
        (1.0 / 3.0 - 1e-6, "expiring_quote", 0),
        (1.0 / 3.0 + 1e-6, "expiring_quote", 1),
        (1.0, "expiring_quote", 1),
    ],
)
def test_the_box_picks_the_life_by_the_bin_rule(value, kind, life):
    env = BoxMarketMakingEnv(living([MS, 2 * MS], policy="reject"), LONG_ROWS)
    assert env.action_space.shape == (6,)
    decoded = env._decode([1.0, -1.0, -1.0, -1.0, -1.0, value])
    assert decoded[0] == kind
    assert (decoded[1][4] if kind == "expiring_quote" else None) == life
    env.close()


def test_the_box_reads_the_cancel_before_the_life():
    env = BoxMarketMakingEnv(living([MS], policy="reject", cancel=True), LONG_ROWS)
    assert env.action_space.shape == (7,)
    assert env._decode([1.0, 0, 0, 0, 0, 1.0, -1.0]) == ("cancel", ())
    assert env._decode([1.0, 0, 0, 0, 0, -1.0, 1.0])[0] == "expiring_quote"
    env.close()


def test_the_box_reads_hedge_cancel_and_life_at_their_own_places():
    cfg = living([MS], policy="reject", cancel=True)
    cfg["hedge_actions"] = True
    env = BoxMarketMakingEnv(cfg, LONG_ROWS)
    assert env.action_space.shape == (8,)
    quote = [1.0, -1.0, -1.0, -1.0, -1.0]
    assert env._decode([*quote, 1.0, 1.0, 1.0])[0] == "hedge"
    assert env._decode([*quote, -1.0, 1.0, 1.0]) == ("cancel", ())
    assert env._decode([*quote, -1.0, -1.0, 1.0]) == ("expiring_quote", (0, 0, 0, 0, 0))
    assert env._decode([*quote, -1.0, -1.0, -1.0]) == ("quote", (0, 0, 0, 0))
    env.close()


def test_expiry_is_exact_at_the_life_after_activation():
    assert filled([2000 * MS], [[*BID, 1]]) == 0
    assert filled([2000 * MS + 1], [[*BID, 1]]) == 1


def test_a_quote_replaced_before_its_life_does_not_expire_the_replacement():
    assert filled([1500 * MS], [[*BID, 1], [*BID, 0]]) == 1
    assert filled([1500 * MS, 1500 * MS + 1], [[*BID, 1], [*BID, 2]]) == 1


def test_the_no_expiry_choice_rests_past_every_life():
    assert filled([MS], [[*BID, 0]]) == 1
    assert filled([MS], [[*BID, 1]]) == 0


@pytest.mark.parametrize("lives", [[], [0], [5, 5], [5, 3], list(range(1, 10))])
def test_invalid_lives_are_refused_at_construction(lives):
    with pytest.raises(Exception, match="quote lives"):
        MaskableMarketMakingEnv(living(lives), LONG_ROWS)


def test_a_residual_environment_refuses_lives():
    cfg = living([MS])
    cfg["residual"] = {"kind": "placeholder"}
    with pytest.raises(ValueError, match="quote_lives_ns"):
        ResidualMarketMakingEnv(cfg, LONG_ROWS)


def test_the_cancel_wrapper_refuses_lives():
    env = MaskableMarketMakingEnv(living([MS], cancel=True), LONG_ROWS)
    with pytest.raises(ValueError, match="quote_lives_ns"):
        ExpiringQuote(env, rows=2)
    env.close()
