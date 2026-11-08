"""The hard limits of the constrained comparison (#105) on the RL path: limits derived from
the margin rule, `max_notional = capital / maintenance rate`, are the kernel's gate, and a
policy proposing beyond them never gets the order executed. The margin gate itself is not
wired into the kernel (#283); this is the stated stand-in."""

from decimal import Decimal

import numpy as np
from conftest import config, walk

from mm_rl_bridge import BoxMarketMakingEnv

CAPITAL = Decimal("125")
MAINTENANCE_RATE = Decimal("0.05")


def margin_limits(capital, rate, max_position):
    return {"max_position": str(max_position), "max_notional": str(capital / rate),
            "scaling_factor": "1"}


def run(settings, action, decisions=399):
    env = BoxMarketMakingEnv(settings, walk())
    env.reset(seed=1)
    refused, worst_position, worst_notional = 0, Decimal(0), Decimal(0)
    for _ in range(decisions):
        _, _, terminated, truncated, info = env.step(action)
        # The gate refuses order legs, which the venue reports as rejected.
        refused += info["orders"]["rejected"]
        if terminated or truncated:
            break
    attribution = env.attribution()
    for _, price, after in attribution["positions"]:
        worst_position = max(worst_position, abs(Decimal(after)))
        worst_notional = max(worst_notional, abs(Decimal(after)) * Decimal(price))
    env.close()
    return refused, worst_position, worst_notional


def test_a_policy_proposing_beyond_the_margin_limits_never_gets_past_the_gate():
    # Always bid the largest size at the touch and ask the smallest far away: it keeps buying.
    greedy = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    # A notional the walk's prices (about 95 to 100) reach at three lots.
    capital, rate = Decimal("14.5"), MAINTENANCE_RATE
    settings = config(policy="reject", max_steps=1000)
    settings["limits"] = margin_limits(capital, rate, 20)
    refused, position, notional = run(settings, greedy)
    assert notional <= capital / rate
    assert refused > 0, "the gate never had to refuse an order"
    # Without the limits the same policy goes past them: the fixture is not vacuous.
    free = config(policy="reject", max_steps=1000)
    free["limits"] = None
    _, free_position, free_notional = run(free, greedy)
    assert free_notional > capital / rate and free_position > position


def test_the_design_limits_are_the_margin_rule_of_its_capital():
    limits = margin_limits(CAPITAL, MAINTENANCE_RATE, 20)
    assert Decimal(limits["max_notional"]) == Decimal("2500")
