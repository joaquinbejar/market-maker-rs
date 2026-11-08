"""The cancel-only action and the expiring quote through the bridge (#288). All data here is
synthetic."""

import numpy as np
import pytest
from conftest import LONG_ROWS, config, row

from mm_rl_bridge import BoxMarketMakingEnv, ExpiringQuote, MaskableMarketMakingEnv
from mm_rl_bridge.envs import ResidualMarketMakingEnv

# The position's raw value in the observation.
POSITION = 6
# A bid of 2 lots at 99.5 (offset 1) and a far ask: it fills when a row's ask reaches 99.5.
BID = [1, 1, 2, 1, 0]


def cancelling(policy="mask", **kwargs):
    cfg = config(policy=policy, **kwargs)
    cfg["cancel_action"] = True
    return cfg


def test_without_the_flag_nothing_changes_and_with_it_the_space_is_version_4():
    plain = MaskableMarketMakingEnv(config(), LONG_ROWS)
    off = config()
    off["cancel_action"] = False
    explicit = MaskableMarketMakingEnv(off, LONG_ROWS)
    assert plain.spec_id == explicit.spec_id
    assert plain.bridge_info["cancel_action"] is False
    assert list(plain.action_space.nvec) == [2, 3, 3, 2, 2]
    with_flag = MaskableMarketMakingEnv(cancelling(), LONG_ROWS)
    assert with_flag.bridge_info["cancel_action"] is True
    assert list(with_flag.action_space.nvec) == [3, 3, 3, 2, 2]
    assert plain.spec_id[1] == 2 and with_flag.spec_id[1] == 4
    assert with_flag.spec_id[4] != plain.spec_id[4]
    # A checkpoint of one space is refused by the other.
    with pytest.raises(Exception, match="refused|versions|specification"):
        MaskableMarketMakingEnv(config(), LONG_ROWS, expected_spec=with_flag.spec_id)
    for env in (plain, explicit, with_flag):
        env.close()


def test_the_maskable_cancel_is_the_third_kind_and_always_allowed():
    env = MaskableMarketMakingEnv(cancelling(), LONG_ROWS)
    env.reset(seed=1)
    mask = env.action_masks()
    # One value per choice of every dimension, the cancel among the kinds.
    assert len(mask) == sum(env.action_space.nvec)
    assert list(mask[:3]) == [True, True, True]
    assert env._decode([2, 0, 0, 0, 0]) == ("cancel", ())
    assert env._decode([0, 2, 2, 1, 1]) == ("hold", ())
    assert env._decode(BID)[0] == "quote"
    env.close()


def test_the_box_cancel_is_the_last_value_after_a_hedge():
    env = BoxMarketMakingEnv(cancelling(policy="reject"), LONG_ROWS)
    assert env.action_space.shape == (6,)
    quote = [1.0, 0.0, 0.5, 0.5, -1.0]
    assert env._decode(quote + [0.0])[0] == "quote"
    assert env._decode(quote + [0.01]) == ("cancel", ())
    env.close()
    both = cancelling(policy="reject")
    both["hedge_actions"] = True
    env = BoxMarketMakingEnv(both, LONG_ROWS)
    assert env.action_space.shape == (7,)
    assert env._decode(quote + [0.5, 0.5])[0] == "hedge"
    assert env._decode(quote + [0.0, 0.5]) == ("cancel", ())
    env.close()


def test_a_residual_environment_refuses_the_cancel_action():
    cfg = cancelling()
    cfg["residual"] = {"kind": "placeholder"}
    with pytest.raises(ValueError, match="cancel_action"):
        ResidualMarketMakingEnv(cfg, LONG_ROWS)


def rows_crossing_at(index, count=8):
    """Rows a second apart at 100 / 101, and at `index` an ask of 99.5 that fills the bid."""
    return [
        row(1000 * (i + 1), 197, 199) if i == index else row(1000 * (i + 1), 200, 202)
        for i in range(count)
    ]


def expiring(rows, cancel_ms=0, submit_ms=0, policy="mask"):
    cfg = cancelling(policy=policy)
    cfg["latency"]["cancel_q"] = cancel_ms
    cfg["latency"]["submit_q"] = submit_ms
    return MaskableMarketMakingEnv(cfg, rows)


def position_after_expiry(cross_row, duration, cancel_ms=0, submit_ms=0):
    env = ExpiringQuote(
        expiring(rows_crossing_at(cross_row), cancel_ms, submit_ms), rows=duration
    )
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(BID)
    kinds = [s["kind"] for s in info["expiry"]["steps"]]
    assert kinds == ["quote"] + ["hold"] * (duration - 1) + ["cancel"]
    assert info["expiry"]["cancelled"]
    # One more decision, so that a fill on the row after the cancel would be seen.
    _, _, _, _, after = env.step([0, 0, 0, 0, 0])
    env.close()
    # Raw values are exact decimals, as strings.
    return float(after["raw_observation"][POSITION])


def test_a_quote_rests_exactly_its_rows_with_no_latency():
    # Sent on row 0 and cancelled at the decision on row 3: it can fill on rows 1 to 3.
    assert position_after_expiry(cross_row=3, duration=3) == 2.0
    assert position_after_expiry(cross_row=4, duration=3) == 0.0
    assert position_after_expiry(cross_row=1, duration=1) == 2.0
    assert position_after_expiry(cross_row=2, duration=1) == 0.0


def test_a_cancel_latency_beyond_the_submission_latency_lengthens_the_rest():
    # A cancel 1.5 rows late: the fill on row 4 lands before it, and stands.
    assert position_after_expiry(cross_row=4, duration=3, cancel_ms=1500) == 2.0
    assert position_after_expiry(cross_row=5, duration=3, cancel_ms=1500) == 0.0


def test_the_wrapper_sums_the_rewards_and_passes_other_actions_through():
    env = ExpiringQuote(expiring(rows_crossing_at(2)), rows=2)
    direct = expiring(rows_crossing_at(2))
    env.reset(seed=1)
    direct.reset(seed=1)
    _, reward, _, _, info = env.step(BID)
    steps = [direct.step(BID), direct.act("hold"), direct.act("cancel")]
    assert reward == pytest.approx(sum(s[1] for s in steps))
    assert [s["reward_exact"] for s in info["expiry"]["steps"]] == [
        s[4]["reward_exact"] for s in steps
    ]
    # A hold is one step, and nothing is cancelled.
    _, _, _, _, info = env.step([0, 0, 0, 0, 0])
    assert [s["kind"] for s in info["expiry"]["steps"]] == ["hold"]
    assert not info["expiry"]["cancelled"]
    assert np.array_equal(env.action_masks(), env.env.action_masks())
    env.close()
    direct.close()


def test_the_wrapper_needs_the_cancel_action_and_a_positive_duration():
    with pytest.raises(ValueError, match="cancel_action"):
        ExpiringQuote(MaskableMarketMakingEnv(config(), LONG_ROWS), rows=2)
    for rows in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            ExpiringQuote(expiring(LONG_ROWS), rows=rows)
