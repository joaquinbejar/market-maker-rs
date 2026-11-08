"""Tables, intervals and decisions of the quote duration comparison of #110, from the
runner's holdout report.

    python experiments/m4_10/report.py <out dir> <intervals binary>

reads `frozen.json` and `holdout.json` from `<out dir>`, writes the series to bootstrap to
`<out dir>/series.json`, runs the intervals binary (the example `m4_05_intervals`, the
evaluation's block bootstrap, with the design's settings) on it, and writes `report.json`.

This file fixes, before the holdout is read, how the design's metrics and decision rule are
read:
- A day is a regime and holdout day. Its value is the day's exact net result; a learned arm's
  is the unweighted mean over its five training seeds, and each seed is also reported apart.
- Each latency is read apart, per regime and pooled (every regime, 15 days); each fee the
  holdout ran is read apart, the fee rule's first.
- Every comparison is paired, day by day, at the same latency and fee, and bootstrapped over
  days.
- Duration: full against no_duration at each latency. Duration helps when the interval's
  lower bound is above 0 and neither arm breached a limit on any day.
- Ablations: full against no_teacher and against no_predictor, the same way ("helps").
- No trade: every learned arm against no_trade at that latency (no_trade deciding every row):
  an interval that includes 0 is reported as equivalent to not trading.
- Avellaneda-Stoikov, at decision intervals of 1 and 10 rows, is reported beside, with no
  decision.
- Descriptive, with no decision: the no_duration checkpoints under immediate fill. Under the
  design's queue an arm deciding every row fills only by holding its quote for at least 8
  rows, so its result reads as the cost of having no duration action under this queue.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
from decimal import Decimal

LEARNED = ("full", "no_teacher", "no_duration", "no_predictor")
ORDER_KINDS = ("accepted", "rejected", "canceled", "cancel_rejected", "fills")
MARKOUT_MS = ("1000", "5000", "30000")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values, Decimal(0)) / len(values) if values else None


def daily(rows):
    """Day key to the arm's daily value; a learned arm's is the mean over its seeds."""
    by_day = {}
    for row in rows:
        by_day.setdefault((row["regime"], row["day"]), []).append(Decimal(row["daily_value"]))
    return {key: mean(values) for key, values in sorted(by_day.items())}


def breaches(rows):
    return sum(bool(e["limit_breach"]) for r in rows for e in r["episodes"])


def summary(rows):
    """One arm in one latency, fee and scope; per day figures are over every seed's day."""
    episodes = [e for r in rows for e in r["episodes"]]

    def per_day(value):
        return str(mean(Decimal(str(value(e))) for e in episodes))

    durations = {}
    for e in episodes:
        for duration, count in e["durations"].items():
            durations[duration] = durations.get(duration, 0) + count
    return {
        "days": len(daily(rows)),
        "seed_days": len(rows),
        "mean_daily_net": str(mean(daily(rows).values())),
        "per_hour": {k: per_day(lambda e, k=k: e["per_hour"][k]) for k in ORDER_KINDS},
        "markouts_ticks_per_day": {h: per_day(lambda e, h=h: e["markouts_ticks"][h]["ticks"])
                                   for h in MARKOUT_MS},
        "mean_abs_position": per_day(lambda e: e["attribution"]["mean_abs_position"] or 0),
        "max_abs_position": str(max(Decimal(e["attribution"]["max_abs_position"])
                                    for e in episodes)),
        "time_beyond_5_lots": per_day(lambda e: e["time_beyond_5_lots"]),
        "decide_microseconds": per_day(lambda e: e["decide_microseconds"]),
        "holds_per_decision": per_day(lambda e: e["holds_per_decision"]),
        "durations_chosen": dict(sorted(durations.items(), key=lambda kv: int(kv[0]))),
        "attribution": {k: per_day(lambda e, k=k: e["attribution"][k] or 0)
                        for k in ("spread_capture", "inventory_pnl", "quote_fees",
                                  "max_drawdown", "fills")},
        "limit_breaches": breaches(rows),
    }


def checked(out):
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


def key_of(row):
    """An arm as the report names it: a classical or no-trade arm with its interval."""
    if row["arm"] in LEARNED:
        return row["arm"]
    return f"{row['arm']}/every{row['every']}"


PLAN = ([("duration", "full", "no_duration")]
        + [("ablation", "full", other) for other in ("no_teacher", "no_predictor")]
        + [("no_trade", arm, "no_trade/every1") for arm in LEARNED])


def build(holdout):
    tables, series, plan, descriptive = {}, {}, [], {}
    queue = [r for r in holdout["rows"] if r.get("fill", "queue") == "queue"]
    immediate = [r for r in holdout["rows"] if r.get("fill", "queue") == "immediate"]
    for fee in holdout["fees"]:
        for latency in sorted({r["latency"] for r in queue}, key=int):
            chosen_rows = [r for r in queue if r["fee"] == fee and r["latency"] == latency]
            for scope, members in scopes_of(chosen_rows).items():
                by_arm = {}
                for r in chosen_rows:
                    if r["regime"] in members:
                        by_arm.setdefault(key_of(r), []).append(r)
                table = {arm: summary(rows) for arm, rows in sorted(by_arm.items())}
                for arm in LEARNED:
                    if arm in by_arm:
                        table[arm]["per_seed"] = {
                            seed: summary([r for r in by_arm[arm] if r["candidate"] == seed])
                            for seed in sorted({r["candidate"] for r in by_arm[arm]})}
                tables.setdefault(fee, {}).setdefault(latency, {})[scope] = table
                values = {arm: daily(rows) for arm, rows in by_arm.items()}
                prefix = f"{fee}/{latency}/{scope}"
                for arm, value in values.items():
                    series[f"{prefix}/{arm}"] = [str(v) for v in value.values()]
                for kind, a, b in PLAN:
                    name = f"{prefix}/{a}-minus-{b}"
                    series[name] = difference(values[a], values[b])
                    plan.append({"kind": kind, "fee": fee, "latency": latency, "scope": scope,
                                 "name": name, "arm": a, "against": b,
                                 "breaches": breaches(by_arm[a]) + breaches(by_arm[b])})
    for latency in sorted({r["latency"] for r in immediate}, key=int):
        rows = [r for r in immediate if r["latency"] == latency]
        descriptive[latency] = {scope: summary([r for r in rows if r["regime"] in members])
                                for scope, members in scopes_of(rows).items()}
    return tables, series, plan, descriptive


def decide(item, interval):
    if any(interval.get(key) is None for key in ("mean", "lower", "upper")):
        raise ValueError(f"no interval to decide on: {interval}")
    lower, upper = Decimal(str(interval["lower"])), Decimal(str(interval["upper"]))
    if item["kind"] in ("duration", "ablation"):
        return "helps" if lower > 0 and item["breaches"] == 0 else "not shown"
    return "equivalent to not trading" if lower <= 0 <= upper else (
        "better than not trading" if lower > 0 else "worse than not trading")


def main():
    out = pathlib.Path(sys.argv[1])
    binary = sys.argv[2]
    design = json.loads((pathlib.Path(__file__).resolve().parent / "design.json").read_text())
    holdout = checked(out)
    tables, series, plan, descriptive = build(holdout)
    payload = {"bootstrap": design["evaluation"]["bootstrap"], "series": series}
    (out / "series.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    intervals = json.loads(subprocess.run([binary, str(out / "series.json")], check=True,
                                          capture_output=True, text=True).stdout)
    decisions = [item | {"interval": intervals[item["name"]],
                         "decision": decide(item, intervals[item["name"]])} for item in plan]
    report = {
        "commit": holdout["commit"], "data": holdout["data"], "bounds": holdout["bounds"],
        "frozen_sha256": holdout["frozen_sha256"], "holdout_sha256": sha256(out / "holdout.json"),
        "fees": holdout["fees"], "tables": tables, "intervals": intervals,
        "decisions": decisions,
        "descriptive_immediate_fill": {"what": "the no_duration checkpoints under immediate "
                                               "fill, at the chosen fee; no decision",
                                       "tables": descriptive},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"decisions": len(decisions), "report_sha256": sha256(out / "report.json")}))


if __name__ == "__main__":
    main()
