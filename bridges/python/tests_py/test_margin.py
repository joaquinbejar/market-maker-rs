"""The kernel's balance gate through the bridge (#105): a policy asking for more than the
balance holds never gets it executed, and each step reports what the gate cut or refused.
All data is synthetic."""

import pytest
from conftest import LONG_ROWS, config, walk

from mm_rl_bridge import MaskableMarketMakingEnv
from mm_rl_bridge.wrappers import DecisionInterval

# A bid of 2 lots at 99.5 and an ask of 2 lots at 100.5 (offsets 1 and 1).
BOTH = [1, 1, 1, 1, 1]
HOLD = [0, 0, 0, 0, 0]
# The position's raw value in the observation.
POSITION = 6


def gated(base_deposit="1", capital="150", max_steps=100):
    cfg = config(max_steps=max_steps)
    cfg["initial_capital"] = capital
    if base_deposit is not None:
        cfg["margin"] = {"base_deposit": base_deposit}
    return cfg


def position_after(cfg, rows=LONG_ROWS):
    env = MaskableMarketMakingEnv(cfg, rows)
    env.reset(seed=1)
    margin = {"reduced": 0, "rejected": 0}
    while True:
        *_, terminated, truncated, info = env.step(BOTH)
        for verdict in margin:
            margin[verdict] += info["gates"]["margin"][verdict]
        if terminated or truncated:
            break
    position = float(info["raw_observation"][POSITION] or 0)
    final = env.attribution()["max_abs_position"]
    env.close()
    return position, final, margin


def test_a_policy_beyond_the_balance_never_gets_it_executed_and_the_cuts_are_reported():
    # The balance pays for one lot at 99.5: the bid is cut, then refused.
    _, held, margin = position_after(gated())
    assert held == "1"
    # The kernel's own decisions for this policy and these rows; the worker tests check the
    # counts against the kernel's events step by step.
    assert margin == {"reduced": 5, "rejected": 1}
    # Without the gate the same policy holds 2, and nothing is cut.
    _, held, margin = position_after(gated(base_deposit=None))
    assert held == "2" and margin == {"reduced": 0, "rejected": 0}


def test_each_step_reports_every_check_and_the_limits_apart_from_the_margin():
    cfg = gated()
    cfg["limits"] = {"max_position": "1", "max_notional": "100000", "scaling_factor": "1"}
    env = MaskableMarketMakingEnv(cfg, LONG_ROWS)
    env.reset(seed=1)
    _, _, _, _, info = env.step(BOTH)
    assert set(info["gates"]) == {"limits", "combined", "margin"}
    assert all(set(c) == {"reduced", "rejected"} for c in info["gates"].values())
    assert info["gates"]["combined"] == {"reduced": 0, "rejected": 0}
    # The limits decide first: the 2 lot quote is cut there, then the margin sees 1 lot.
    assert info["gates"]["limits"]["reduced"] + info["gates"]["limits"]["rejected"] >= 1
    _, _, _, _, info = env.step(HOLD)
    assert info["gates"] == {k: {"reduced": 0, "rejected": 0} for k in info["gates"]}
    env.close()


def test_an_interval_reports_the_decided_steps_gates_since_a_hold_submits_nothing():
    rows = walk(60)
    direct = MaskableMarketMakingEnv(gated(max_steps=1000), rows)
    wrapped = DecisionInterval(MaskableMarketMakingEnv(gated(max_steps=1000), rows), 3)
    direct.reset(seed=2)
    wrapped.reset(seed=2)
    seen = 0
    for _ in range(10):
        *_, info = wrapped.step(BOTH)
        steps = [direct.step(BOTH)] + [direct.step(HOLD) for _ in range(info["rows"] - 1)]
        assert info["gates"] == steps[0][4]["gates"]
        for held in steps[1:]:
            assert held[4]["gates"] == {k: {"reduced": 0, "rejected": 0} for k in held[4]["gates"]}
        seen += sum(c for counts in info["gates"].values() for c in counts.values())
    assert seen > 0
    direct.close()
    wrapped.close()


def test_a_margin_with_another_field_or_a_deposit_off_the_grid_is_refused():
    cfg = gated()
    cfg["margin"]["leverage"] = "20"
    with pytest.raises(Exception, match="unknown field|leverage"):
        MaskableMarketMakingEnv(cfg, LONG_ROWS)
    with pytest.raises(Exception):
        MaskableMarketMakingEnv(gated(base_deposit="0.5"), LONG_ROWS)


def test_hedge_actions_run_under_the_gate():
    # The bridge's hedges are in the quoted instrument, so the gate decides on them too; the
    # two instrument hedged kernel, which refuses the gate, is not one the bridge drives.
    cfg = gated()
    cfg["hedge_actions"] = True
    env = MaskableMarketMakingEnv(cfg, LONG_ROWS)
    env.reset(seed=1)
    *_, info = env.step(BOTH + [0])
    assert info["gates"]["margin"]["reduced"] == 2
    env.close()
