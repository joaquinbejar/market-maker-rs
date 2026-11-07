"""Hedge actions through the bridge (#258), on the hand computed trajectory's rows."""

import numpy as np
import pytest
from conftest import LONG_ROWS, config, walk

from mm_rl_bridge import BoxMarketMakingEnv, MaskableMarketMakingEnv
from mm_rl_bridge.envs import ResidualMarketMakingEnv
from mm_rl_bridge.training import load_for_inference, train

PPO_SMALL = {"n_steps": 16, "batch_size": 16, "n_epochs": 1}


def hedged(policy="mask", **kwargs):
    cfg = config(policy=policy, **kwargs)
    cfg["hedge_actions"] = True
    return cfg


def test_without_the_flag_nothing_changes():
    plain = MaskableMarketMakingEnv(config(), LONG_ROWS)
    off = config()
    off["hedge_actions"] = False
    explicit = MaskableMarketMakingEnv(off, LONG_ROWS)
    assert plain.spec_id == explicit.spec_id
    assert plain.bridge_info["hedge_actions"] is False
    assert list(plain.action_space.nvec) == [2, 3, 3, 2, 2]
    with_flag = MaskableMarketMakingEnv(hedged(), LONG_ROWS)
    assert with_flag.spec_id[:4] == plain.spec_id[:4]
    assert with_flag.spec_id[4] != plain.spec_id[4]
    for env in (plain, explicit, with_flag):
        env.close()


def test_the_maskable_hedge_dimension_is_the_sizes_and_follows_the_hedge_mask():
    env = MaskableMarketMakingEnv(hedged(), LONG_ROWS)
    sizes = env.bridge_info["sizes"]
    assert list(env.action_space.nvec) == [2, 3, 3, sizes, sizes, 1 + sizes]
    env.reset(seed=1)
    # Flat: no hedge but "no hedge".
    assert list(env.action_masks()[-(1 + sizes):]) == [True, False, False]
    # Bid 2 at 99.5, filled on the next row: long 2, so both sizes reduce it.
    env.step([1, 1, 2, 1, 0, 0])
    assert list(env.action_masks()[-(1 + sizes):]) == [True, True, True]
    env.close()


def test_a_hedge_ignores_the_quote_values_and_is_the_hedge_step_itself():
    through = MaskableMarketMakingEnv(hedged(), LONG_ROWS)
    direct = MaskableMarketMakingEnv(hedged(), LONG_ROWS)
    for env in (through, direct):
        env.reset(seed=1)
        env.step([1, 1, 2, 1, 0, 0])
    # Quote values that would mean something, and a hedge of size 0 (1 lot).
    observation, reward, terminated, truncated, info = through.step([1, 2, 0, 1, 1, 1])
    reply = direct._native.step_hedge(0)
    assert info["decoded_action"] == {"kind": "hedge", "indices": [0]}
    assert info["applied_action"] == {"kind": "hedge", "indices": [0]}
    assert info["reward_exact"] == reply["reward"]
    assert info["net"] == reply["net"]
    assert info["refused_because"] == reply["refused_because"] is None
    assert list(observation[: len(reply["observation"]["values"])]) == pytest.approx(
        reply["observation"]["values"]
    )
    assert terminated == reply["terminated"] and truncated == reply["truncated"]
    through.close()
    direct.close()


def test_the_box_hedge_threshold_and_bins():
    env = BoxMarketMakingEnv(hedged(policy="reject"), LONG_ROWS)
    assert env.action_space.shape == (6,)
    quote = [1.0, 0.0, 0.5, 0.5, -1.0]
    assert env._decode(quote + [0.0])[0] == "quote"
    assert env._decode(quote + [-1.0])[0] == "quote"
    assert env._decode([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == ("hold", ())
    assert env._decode(quote + [0.01]) == ("hedge", (0,))
    assert env._decode(quote + [0.5]) == ("hedge", (1,))
    assert env._decode(quote + [1.0]) == ("hedge", (1,))
    with pytest.raises(ValueError, match="6 values"):
        env._decode(quote)
    # A hedge at a flat position is outside the mask: a charged hold.
    env.reset(seed=1)
    _, _, _, _, info = env.step(quote + [1.0])
    assert info["applied_action"] == {"kind": "hold", "indices": []}
    assert info["refused_because"] is not None
    assert info["penalty"] == "0.25"
    env.close()


def test_a_residual_environment_refuses_the_flag():
    cfg = hedged()
    cfg["residual"] = {
        "baseline": {
            "kind": "avellaneda_stoikov",
            "risk_aversion": "0.1",
            "volatility": "0.2",
            "order_intensity": "1.5",
            "session_start_ms": 0,
            "horizon_ms": 100000,
            "size": "1",
        },
        "price_ticks": 4,
        "size_lots": 1,
        "default_order_size": "1",
    }
    with pytest.raises(ValueError, match="hedge_actions"):
        ResidualMarketMakingEnv(cfg, walk())


def test_a_checkpoint_loads_only_with_the_same_flag(tmp_path):
    plain_path = tmp_path / "plain"
    plain = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    metadata = train("maskable_ppo", plain, 3, 32, plain_path, "abc", PPO_SMALL)
    plain.close()
    # A checkpoint from before the flag has no key, and loads into a plain environment.
    assert "hedge_actions" not in metadata["config"]
    again = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    load_for_inference(plain_path, again)
    again.close()
    with_flag = MaskableMarketMakingEnv(hedged(max_steps=1000), walk())
    with pytest.raises(ValueError, match="hedge_actions=False, this environment has hedge_actions=True"):
        load_for_inference(plain_path, with_flag)
    hedged_path = tmp_path / "hedged"
    train("maskable_ppo", with_flag, 3, 32, hedged_path, "abc", PPO_SMALL)
    with_flag.close()
    without = MaskableMarketMakingEnv(config(max_steps=1000), walk())
    with pytest.raises(ValueError, match="hedge_actions=True, this environment has hedge_actions=False"):
        load_for_inference(hedged_path, without)
    without.close()
    back = MaskableMarketMakingEnv(hedged(max_steps=1000), walk())
    model, _ = load_for_inference(hedged_path, back)
    assert np.asarray(model.action_space.nvec).tolist()[-1] == 1 + back.bridge_info["sizes"]
    back.close()
