"""The report of #108's holdout, as markdown.

    python experiments/m5_03/report.py <out dir>/holdout.json

A learned arm's day is the mean over its five seeds, as #102. Intervals are paired 90 % stationary
block bootstraps (geometric blocks of mean length 2, as #102) over the 30 holdout days of the daily
difference. design.json calls this a moving block bootstrap; it is the stationary one, and
design.json is left as committed.
"""

import importlib.util
import json
import math
import pathlib
import sys
from decimal import ROUND_CEILING, Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ARMS = ("no_trade", "fixed_spread", "avellaneda_stoikov", "glft", "maskable_ppo", "residual_sac")
CONDITIONS = ("unperturbed", "adversary", "fixed_stress", "execution_stress")
CAPITAL = Decimal(250)
# margin-rs 0.2's `linear-im/1`: one tier, maintenance 5 % of notional, rounded up at scale 6.
MAINTENANCE_RATE = Decimal("0.05")


def maintenance(position, price):
    if price is None:
        return Decimal(0)
    notional = abs(Decimal(position)) * Decimal(price)
    return (notional * MAINTENANCE_RATE).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)


def by_day(rows, key):
    """Per (condition, arm): the ordered days and the per-day value of `key`, a learned arm's
    as the mean over its seeds."""
    out = {}
    for row in rows:
        out.setdefault((row["condition"], row["arm"]), {}).setdefault(
            (row["regime"], row["day"]), []).append(key(row))
    return {k: (sorted(v), [float(np.mean(v[d])) for d in sorted(v)]) for k, v in out.items()}


def stationary(values, mean_block=2.0, replicates=2000, seed=108):
    n = len(values)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    draws = []
    for _ in range(replicates):
        i = int(rng.integers(n))
        total = 0.0
        for k in range(n):
            if k > 0:
                i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
            total += values[i]
        draws.append(total / n)
    low, high = np.quantile(draws, [0.05, 0.95])
    return float(np.mean(values)), float(low), float(high)


def cvar(values, share=0.05):
    worst = sorted(values)[: max(1, math.ceil(share * len(values)))]
    return float(np.mean(worst))


def main():
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    rows = report["rows"]
    net = by_day(rows, lambda r: float(Decimal(r["net"])))
    drawdown = by_day(rows, lambda r: float(Decimal(r["max_drawdown"] or 0)))
    mean_pos = by_day(rows, lambda r: float(Decimal(r["mean_abs_position"] or 0)))
    peak = {}
    breaches = {}
    for r in rows:
        k = (r["condition"], r["arm"])
        req = maintenance(r["peak_position"], r["peak_position_price"])
        peak[k] = max(peak.get(k, Decimal(0)), req)
        breaches[k] = breaches.get(k, 0) + int(bool(r["limit_breach"]))
    print("## #108 holdout (synthetic)\n")
    print(report["data"] + "\n")
    print(report["bounds"] + "\n")
    print("| condition | arm | net/day | CVaR 5 % | mean max DD | worst DD | mean abs pos | breaches "
          "| maintenance at peak | shortfall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        for arm in ARMS:
            k = (condition, arm)
            if k not in net:
                continue
            days, values = net[k]
            print(f"| {condition} | {arm} | {np.mean(values):.2f} | {cvar(values):.2f} "
                  f"| {np.mean(drawdown[k][1]):.2f} | {max(drawdown[k][1]):.2f} "
                  f"| {np.mean(mean_pos[k][1]):.3f} | {breaches[k]} | {peak[k]:.2f} "
                  f"| {max(Decimal(0), peak[k] - CAPITAL):.2f} |")
    print("\n### Paired intervals of the daily net, 90 %\n")
    print("| arm | condition | against its unperturbed run | against no-trade |")
    print("|---|---|---|---|")
    moved_any = False
    for arm in ARMS:
        base_days, base = net[("unperturbed", arm)]
        for condition in CONDITIONS[1:]:
            days, values = net[(condition, arm)]
            if days != base_days:
                raise ValueError(f"{arm} {condition} has other days than its unperturbed run")
            own = stationary([a - b for a, b in zip(values, base, strict=True)])
            _, trade = net[(condition, "no_trade")]
            versus = stationary([a - b for a, b in zip(values, trade, strict=True)])
            moved_any |= not (own[1] <= 0.0 <= own[2])
            print(f"| {arm} | {condition} | {own[0]:.2f} [{own[1]:.2f}, {own[2]:.2f}] "
                  f"| {versus[0]:.2f} [{versus[1]:.2f}, {versus[2]:.2f}] |")
    print()
    if not moved_any:
        print("Stop rule: no condition changed any arm's daily net beyond its interval.")
    print("An interval that holds 0 is reported as such; it is never read as equivalence or as an "
          "absence of risk.")
    print(f"\nMargin: the maintenance requirement of margin-rs `linear-im/1` (5 % of notional) at the "
          f"day's largest absolute position, against a made-up capital of {CAPITAL}. #102's notional "
          f"limit of 2500 caps the requirement at 125, so the shortfall is 0 by construction; the "
          f"requirement itself is the measure to read.")


if __name__ == "__main__":
    main()
