"""Sign, scale and accounting of the costs of the constrained comparison (#105)."""

import importlib.util
import pathlib
from decimal import Decimal

import pytest
from conftest import LONG_ROWS, config

from mm_rl_bridge import BoxMarketMakingEnv

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_08_costs", HERE.parent / "experiments" / "m4_08" / "costs.py"
)
costs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(costs)

TICK = Decimal("0.5")


def info(position, row, net="0"):
    raw = [None] * 11
    raw[costs.POSITION] = position
    return {"raw_observation": raw, "row": row, "net": net}


def test_costs_follow_their_formulas_on_hand_values():
    rows = [(0, 199, 5, 201, 5)]  # mid 200 ticks of 0.5: 100
    # Flat: nothing.
    assert costs.costs(info("0", 0), rows, TICK) == {
        "c_inventory": 0.0, "c_notional": 0.0, "c_margin": 0.0}
    # 4 lots: under the soft limit, notional 400 / 2500, margin 0.05 * 400 / 125.
    four = costs.costs(info("4", 0), rows, TICK)
    assert four["c_inventory"] == 0.0
    assert four["c_notional"] == pytest.approx(0.16)
    assert four["c_margin"] == pytest.approx(0.16)
    # Short 8 lots: the sign does not matter; 3 lots over the soft limit of 5.
    short = costs.costs(info("-8", 0), rows, TICK)
    assert short["c_inventory"] == pytest.approx(3 / 15)
    assert short["c_notional"] == pytest.approx(0.32)
    # Losses shrink equity, so the margin proxy rises; at no equity it is 1.
    assert costs.costs(info("4", 0, net="-62.5"), rows, TICK)["c_margin"] == pytest.approx(0.32)
    assert costs.costs(info("4", 0, net="-125"), rows, TICK)["c_margin"] == 1.0
    assert costs.costs(info("20", 0, net="-100"), rows, TICK)["c_margin"] == 1.0


def test_costs_read_the_delivered_position_and_the_observed_row_of_a_step():
    env = BoxMarketMakingEnv(config(policy="reject"), LONG_ROWS)
    env.reset(seed=1)
    # Bid 2 lots at the touch; it fills on the next row (the hand computed trajectory).
    _, _, _, _, step = env.step([1.0, 0.0, 0.5, 1.0, -1.0])
    position = Decimal(str(step["raw_observation"][costs.POSITION]))
    assert position == 2
    mid = Decimal(LONG_ROWS[step["row"]][1] + LONG_ROWS[step["row"]][3]) / 2 * TICK
    measured = costs.costs(step, LONG_ROWS, TICK)
    assert measured["c_notional"] == pytest.approx(float(2 * mid / 2500))
    assert measured["c_inventory"] == 0.0
    env.close()
