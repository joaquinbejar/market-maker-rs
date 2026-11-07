"""Tables, intervals and decisions of the comparison of #103, from the runner's holdout report.

    python experiments/m4_06/report.py <out dir> <intervals binary>

reads `frozen.json` and `holdout.json` from `<out dir>`, writes the series to bootstrap to
`<out dir>/series.json`, runs the intervals binary (the example `m4_05_intervals`, the
evaluation's block bootstrap, with the design's settings) on it, and writes `report.json`.

This file fixes, before the holdout is read, how the design's metrics and decision rule are
read:
- A day is a regime and holdout day. Its value is the day's exact net result; a learned arm's
  is the mean over its five training seeds, and each seed is also reported apart.
- Pooled means the days of every regime together, 15 days.
- Every comparison is paired, day by day, at the same fee, and bootstrapped over days.
- Memory: at each decision interval, each of ema, history and lstm against mlp. Memory helps
  when the interval's lower bound is above 0 and neither arm breached a limit on any day.
- Frequency: for every arm but no_trade, the 5 and 10 row intervals against the 1 row
  interval, by the same rule.
- No trade: each learned arm at each interval against no_trade at that interval; an interval
  that includes 0 is reported as equivalent to not trading.
- Classical: each learned arm against the better of avellaneda_stoikov and glft at the same
  interval, the one with the higher mean daily net in that scope, reported beside.
- Max drawdown is the mean over days of each day's largest drawdown. CVaR 5 % of the daily net
  is the mean of the worst ceil(5 % of the days), the worst day for 5 or 15 days.
- The fee the rule chose is the primary row; 0.0002 is the fixed secondary row, read the same
  way and labelled as such.

The validation pick of the learned arm at one row (`validation_pick.py`) was made after the
freeze and before any holdout read, for the studies that build on #103; it is not part of
this decision rule and is reported beside it.
"""

import hashlib
import json
import math
import pathlib
import subprocess
import sys
from decimal import Decimal

LEARNED = ("ema", "history", "lstm", "mlp")
MEMORY = ("ema", "history", "lstm")
CLASSICAL = ("avellaneda_stoikov", "glft")
QUOTING = LEARNED + CLASSICAL


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values, Decimal(0)) / len(values) if values else None


def cvar(values, share=Decimal("0.05")):
    ordered = sorted(values)
    return mean(ordered[: max(1, math.ceil(len(ordered) * share))])


def daily(rows):
    """Day key to the arm's daily value; a learned arm's is the mean over its seeds."""
    by_day = {}
    for row in rows:
        by_day.setdefault((row["regime"], row["day"]), []).append(Decimal(row["daily_value"]))
    return {key: mean(values) for key, values in sorted(by_day.items())}


def drawdowns(rows):
    by_day = {}
    for row in rows:
        value = Decimal(row["episodes"][0]["attribution"]["max_drawdown"])
        by_day.setdefault((row["regime"], row["day"]), []).append(value)
    return {key: mean(values) for key, values in sorted(by_day.items())}


def breaches(rows):
    return sum(bool(r["episodes"][0]["limit_breach"]) for r in rows)


def summary(rows):
    """One arm at one interval over some days; per-day counts are over every seed's day."""
    episodes = [r["episodes"][0] for r in rows]
    nets = list(daily(rows).values())
    dds = list(drawdowns(rows).values())
    steps = sum(e["decisions"] for e in episodes)

    def per_day(value):
        return str(mean(Decimal(value(e)) for e in episodes))

    return {
        "days": len(nets),
        "seed_days": len(rows),
        "mean_daily_net": str(mean(nets)),
        "spread_capture": per_day(lambda e: e["attribution"]["spread_capture"]),
        "inventory_pnl": per_day(lambda e: e["attribution"]["inventory_pnl"]),
        "fees": per_day(lambda e: e["attribution"]["quote_fees"]),
        "fills_per_day": per_day(lambda e: e["attribution"]["fills"]),
        "mean_max_drawdown": str(mean(dds)),
        "worst_max_drawdown": str(max(dds)),
        "cvar_5_daily_net": str(cvar(nets)),
        "mean_abs_position": per_day(lambda e: e["attribution"]["mean_abs_position"] or 0),
        "max_abs_position": str(max(Decimal(e["attribution"]["max_abs_position"])
                                    for e in episodes)),
        "peak_notional": str(max(Decimal(e["peak_notional"]) for e in episodes)),
        "time_beyond_5_lots": max(e["time_beyond_5_lots"] for e in episodes),
        "limit_breaches": breaches(rows),
        "refusal_rate": sum(e["refusals"] for e in episodes) / steps,
        "decisions_per_day": per_day(lambda e: e["decisions"]),
        "quote_transactions_per_day": per_day(lambda e: e["quote_transactions"]),
        "orders_per_day": {kind: per_day(lambda e, k=kind: e["orders"][k])
                           for kind in ("accepted", "canceled", "cancel_rejected", "rejected")},
        "decide_microseconds": sum(e["decide_microseconds"] * e["decisions"]
                                   for e in episodes) / steps,
    }


def checked(out):
    """The holdout report, after its manifest and the run's commit match."""
    holdout = json.loads((out / "holdout.json").read_text())
    if sha256(out / "frozen.json") != holdout["frozen_sha256"]:
        raise ValueError("frozen.json is not the manifest the holdout was evaluated from")
    return holdout


def scopes_of(rows):
    regimes = sorted({r["regime"] for r in rows})
    scopes = {regime: [regime] for regime in regimes}
    scopes["pooled"] = regimes
    return scopes


def difference(a, b):
    if a.keys() != b.keys():
        raise ValueError("the two arms do not have the same days")
    return [str(a[day] - b[day]) for day in a]


def build(holdout):
    """Tables and the series to bootstrap, per fee and scope; `name` keys the comparisons."""
    tables, series, plan = {}, {}, []
    for fee in holdout["fees"]:
        at_fee = [r for r in holdout["rows"] if r["fee"] == fee]
        intervals = sorted({r["every"] for r in at_fee})
        for scope, members in scopes_of(at_fee).items():
            chosen = [r for r in at_fee if r["regime"] in members]
            rows_of = {(arm, every): [r for r in chosen if r["arm"] == arm and r["every"] == every]
                       for arm in sorted({r["arm"] for r in chosen}) for every in intervals}
            rows_of = {key: rows for key, rows in rows_of.items() if rows}
            table = {f"{arm}/every{every}": summary(rows) for (arm, every), rows in rows_of.items()}
            for (arm, every), rows in rows_of.items():
                if arm in LEARNED:
                    table[f"{arm}/every{every}"]["per_seed"] = {
                        seed: summary([r for r in rows if r["candidate"] == seed])
                        for seed in sorted({r["candidate"] for r in rows})}
            tables.setdefault(fee, {})[scope] = table
            values = {key: daily(rows) for key, rows in rows_of.items()}
            prefix = f"{fee}/{scope}"
            for (arm, every), value in values.items():
                series[f"{prefix}/{arm}/every{every}"] = [str(v) for v in value.values()]

            def compare(kind, a, b):
                name = f"{prefix}/{a[0]}/every{a[1]}-minus-{b[0]}/every{b[1]}"
                series[name] = difference(values[a], values[b])
                plan.append({"kind": kind, "fee": fee, "scope": scope, "name": name,
                             "arm": a, "against": b,
                             "breaches": breaches(rows_of[a]) + breaches(rows_of[b])})

            for every in intervals:
                for arm in MEMORY:
                    compare("memory", (arm, every), ("mlp", every))
                best = max(CLASSICAL, key=lambda c: (mean(values[(c, every)].values()), c))
                for arm in LEARNED:
                    compare("no_trade", (arm, every), ("no_trade", every))
                    compare("classical", (arm, every), (best, every))
            for arm in QUOTING:
                for every in intervals:
                    if every != 1:
                        compare("frequency", (arm, every), (arm, 1))
    return tables, series, plan


def bounds(interval):
    if any(interval.get(key) is None for key in ("mean", "lower", "upper")):
        raise ValueError(f"no interval to decide on: {interval}")
    return Decimal(str(interval["mean"])), Decimal(str(interval["lower"])), \
        Decimal(str(interval["upper"]))


def decide(item, interval):
    _, lower, upper = bounds(interval)
    if item["kind"] in ("memory", "frequency"):
        return "helps" if lower > 0 and item["breaches"] == 0 else "not shown"
    if item["kind"] == "no_trade":
        return "equivalent to not trading" if lower <= 0 <= upper else (
            "better than not trading" if lower > 0 else "worse than not trading")
    return "reported beside"


def main():
    out = pathlib.Path(sys.argv[1])
    binary = sys.argv[2]
    design = json.loads((pathlib.Path(__file__).resolve().parent / "design.json").read_text())
    holdout = checked(out)
    tables, series, plan = build(holdout)
    payload = {"bootstrap": design["evaluation"]["bootstrap"], "series": series}
    (out / "series.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    intervals = json.loads(subprocess.run([binary, str(out / "series.json")], check=True,
                                          capture_output=True, text=True).stdout)
    decisions = [item | {"interval": intervals[item["name"]],
                         "decision": decide(item, intervals[item["name"]])} for item in plan]
    pick = out / "validation_pick.json"
    report = {
        "commit": holdout["commit"], "data": holdout["data"], "bounds": holdout["bounds"],
        "frozen_sha256": holdout["frozen_sha256"], "holdout_sha256": sha256(out / "holdout.json"),
        "primary_fee": holdout["fees"][0], "secondary_fee": holdout["fees"][-1],
        "validation_pick": ({"sha256": sha256(pick),
                             "selected": json.loads(pick.read_text())["selected"]}
                            if pick.exists() else None),
        "tables": tables, "intervals": intervals, "decisions": decisions,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"decisions": len(decisions), "report_sha256": sha256(out / "report.json")}))


if __name__ == "__main__":
    main()
