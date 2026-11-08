"""The costs of the constrained comparison of #105, apart from any optimizer.

Per step, from the delivered position after it, the observed row's mid and the net result:
`c_inventory = max(0, |q| - 5) / 15`, `c_notional = |q| * mid / 2500`, and the declared
margin proxy `c_margin`: the margin-rs rule `linear-im/1` maintenance requirement
`0.05 * |q| * mid` over equity `125 + net`, capped at 1, and 1 at or below zero equity.
"""

from decimal import Decimal

POSITION = 6
SOFT_LOTS = 5
INVENTORY_SPAN = 15
NOTIONAL_SCALE = Decimal("2500")
MAINTENANCE_RATE = Decimal("0.05")
CAPITAL = Decimal("125")


def costs(info, rows, tick):
    """The design's costs of one step: from the delivered position after it, the observed
    row's mid, and the net result."""
    raw = info["raw_observation"][POSITION]
    position = abs(Decimal(str(raw))) if raw is not None else Decimal(0)
    row = rows[info["row"]] if info["row"] is not None else rows[0]
    mid = Decimal(row[1] + row[3]) / 2 * tick
    inventory = max(Decimal(0), position - SOFT_LOTS) / INVENTORY_SPAN
    notional = position * mid / NOTIONAL_SCALE
    equity = CAPITAL + Decimal(info["net"])
    margin = Decimal(1) if equity <= 0 else min(Decimal(1), MAINTENANCE_RATE * position * mid / equity)
    return {"c_inventory": float(inventory), "c_notional": float(notional),
            "c_margin": float(margin)}
