"""The report of #282's holdout, as markdown.

    python experiments/m5_04/report.py <out dir>/holdout.json [<#108 out dir>/holdout.json]

With #108's holdout rows, the unperturbed condition is checked against them field by field, as
the design requires (kappa 0 on #108's unperturbed holdout days must give #108's rows).

As #108's report: a learned arm's day is the mean over its five seeds, and intervals are paired
90 % stationary block bootstraps (geometric blocks of mean length 2) over the 30 holdout days of
the daily difference, here with seed 282. The impact columns are per day: reactions, imposed
moves and the first-limit volume the moves emptied (modelled consumption, not the agent's
trades).
"""

import importlib.util
import json
import pathlib
import sys
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("m5_03_report", HERE.parent / "m5_03" / "report.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ARMS = base.ARMS
CONDITIONS = ("unperturbed", "impact_only", "adversary", "fixed_stress", "execution_stress")
SEED = 282
# The fields #108 records per row; run.py's FIELDS_108.
FIELDS_108 = ("net", "max_drawdown", "max_abs_position", "mean_abs_position", "peak_notional",
              "refusals", "residual_fallbacks", "residual_clipped", "limit_breach",
              "peak_position", "peak_position_price")


def identity_108(rows, rows_108):
    """The unperturbed rows against #108's unperturbed holdout rows of the same arm, seed,
    regime and day, field by field: the mismatches, empty when it holds, and a row missing on
    either side is a mismatch."""
    def keyed(source):
        return {(r["arm"], r["seed"], r["regime"], r["day"]): r for r in source
                if r["condition"] == "unperturbed"}
    ours, theirs = keyed(rows), keyed(rows_108)
    wrong = sorted(set(ours) ^ set(theirs), key=str)
    for key in sorted(set(ours) & set(theirs), key=str):
        if any(ours[key][f] != theirs[key][f] for f in FIELDS_108):
            wrong.append(key)
    return wrong


def totals(rows):
    """Per (condition, arm): refusals, residual fallbacks and clips summed over days and seeds,
    the largest absolute position, and the margin maintenance at the day's peak position."""
    out = {}
    for r in rows:
        k = (r["condition"], r["arm"])
        t = out.setdefault(k, {"refusals": 0, "fallbacks": 0, "clipped": 0,
                               "max_abs_position": Decimal(0), "maintenance": Decimal(0)})
        t["refusals"] += int(r["refusals"])
        t["fallbacks"] += int(r["residual_fallbacks"])
        t["clipped"] += int(r["residual_clipped"])
        t["max_abs_position"] = max(t["max_abs_position"], abs(Decimal(r["max_abs_position"] or 0)))
        t["maintenance"] = max(t["maintenance"],
                               base.maintenance(r["peak_position"], r["peak_position_price"]))
    return out


def interval(values):
    return base.stationary(values, seed=SEED)


def main():
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    rows = report["rows"]
    mean_pos = base.by_day(rows, lambda r: float(Decimal(r["mean_abs_position"] or 0)))
    counts = totals(rows)
    net = base.by_day(rows, lambda r: float(Decimal(r["net"])))
    drawdown = base.by_day(rows, lambda r: float(Decimal(r["max_drawdown"] or 0)))
    reactions = base.by_day(rows, lambda r: float(r["reactions"]))
    moves = base.by_day(rows, lambda r: float(r["moves"]))
    emptied = base.by_day(rows, lambda r: float(Decimal(r["emptied"])))
    breaches = {}
    for r in rows:
        k = (r["condition"], r["arm"])
        breaches[k] = breaches.get(k, 0) + int(bool(r["limit_breach"]))
    print("## #282 holdout (synthetic)\n")
    print(report["data"] + "\n")
    print(report["bounds"] + "\n")
    print("| condition | arm | net/day | CVaR 5 % | mean max DD | worst DD | breaches "
          "| reactions/day | moves/day | emptied/day |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        for arm in ARMS:
            k = (condition, arm)
            if k not in net:
                continue
            values = net[k][1]
            print(f"| {condition} | {arm} | {np.mean(values):.2f} | {base.cvar(values):.2f} "
                  f"| {np.mean(drawdown[k][1]):.2f} | {max(drawdown[k][1]):.2f} "
                  f"| {breaches[k]} | {np.mean(reactions[k][1]):.1f} "
                  f"| {np.mean(moves[k][1]):.1f} | {np.mean(emptied[k][1]):.2f} |")
    print("\n### Positions, gate and margin\n")
    print("| condition | arm | mean abs pos | largest abs pos | refusals | residual fallbacks "
          "| residual clips | maintenance at peak | shortfall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        for arm in ARMS:
            k = (condition, arm)
            if k not in counts:
                continue
            c = counts[k]
            print(f"| {condition} | {arm} | {np.mean(mean_pos[k][1]):.3f} "
                  f"| {c['max_abs_position']} | {c['refusals']} | {c['fallbacks']} "
                  f"| {c['clipped']} | {c['maintenance']:.2f} "
                  f"| {max(Decimal(0), c['maintenance'] - base.CAPITAL):.2f} |")
    if report.get("chosen"):
        print("\nThe adversary's points, chosen per arm on the selection days: " + "; ".join(
            f"{arm} {choice['point']}" for arm, choice in sorted(report["chosen"].items())))
    print("\n### Paired intervals of the daily net, 90 %\n")
    print("| arm | condition | against its unperturbed run | against no-trade |")
    print("|---|---|---|---|")
    moved_any = False
    for arm in ARMS:
        base_days, base_values = net[("unperturbed", arm)]
        for condition in CONDITIONS[1:]:
            days, values = net[(condition, arm)]
            if days != base_days:
                raise ValueError(f"{arm} {condition} has other days than its unperturbed run")
            own = interval([a - b for a, b in zip(values, base_values, strict=True)])
            _, trade = net[(condition, "no_trade")]
            versus = interval([a - b for a, b in zip(values, trade, strict=True)])
            moved_any |= not (own[1] <= 0.0 <= own[2])
            print(f"| {arm} | {condition} | {own[0]:.2f} [{own[1]:.2f}, {own[2]:.2f}] "
                  f"| {versus[0]:.2f} [{versus[1]:.2f}, {versus[2]:.2f}] |")
    print()
    if not moved_any:
        print("Stop rule: no condition changed any arm's daily net beyond its interval.")
    print("An interval that holds 0 is reported as such; it is never read as equivalence or as an "
          "absence of risk. The frozen policies were trained without their own impact: this "
          "measures their robustness to it, not an adapted policy.")
    if len(sys.argv) > 2:
        wrong = identity_108(rows, json.loads(pathlib.Path(sys.argv[2]).read_text())["rows"])
        if wrong:
            print(f"\nIdentity with #108's unperturbed holdout rows FAILS on {len(wrong)} rows, "
                  f"first {wrong[:5]}.")
        else:
            print("\nIdentity with #108's unperturbed holdout rows holds on every row, field by "
                  "field.")
    print(f"\nMargin: the maintenance requirement of margin-rs `linear-im/1` (5 % of notional) at "
          f"the day's largest absolute position, against a made-up capital of {base.CAPITAL}; "
          "measured, not enforced.")


if __name__ == "__main__":
    main()
