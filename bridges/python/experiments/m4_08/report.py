"""Tables, intervals and decisions of the constrained comparison of #105, from the runner's
holdout report.

    python experiments/m4_08/report.py <out dir> <intervals binary>

reads `frozen.json` and `holdout.json` from `<out dir>`, writes the series to bootstrap to
`<out dir>/series.json`, runs the intervals binary (the example `m4_05_intervals`, the
evaluation's block bootstrap, with the design's settings) on it, and writes `report.json`.

This file fixes, before the holdout is read, how the design's metrics and decision rule are
read:
- A day is a regime and holdout day. Its net is the day's exact net result and its
  constrained cost is the day's mean per step of c_inventory + c_notional; a learned arm's
  day is the mean over its five training seeds, and each seed is also reported apart.
- The rows of the evaluation, each read apart: the unshocked holdout days at the fee the rule
  chose; the shocked holdout days at that fee, one row per shock; and the unshocked days at
  the secondary fee 0.0002, when the rule chose another.
- Each row is read per regime (5 days) and pooled (every regime, 15 days).
- Every comparison is paired, day by day, in the same row, and bootstrapped over days, both
  on the daily net and on the daily constrained cost.
- Constrained better: cpo against ppo_penalized. It holds when the net interval includes or
  is above 0 (its upper bound is at least 0), the cost interval is below 0 (its upper bound is
  below 0), and neither arm breached a limit on any day.
- Also: ppo, ppo_penalized and cpo against no_trade, and ppo_penalized and cpo against ppo,
  read by the same rule and labelled "better by the rule" when it holds. No trading has no
  cost, so no arm can be below it on cost: those read as not shown by construction.
- Costs are reported apart, each as a mean per step per day against its budget, with the
  episodes over budget; c_margin is reported, not constrained.
- Nothing here is a safety guarantee.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
from decimal import Decimal

LEARNED = ("ppo", "ppo_penalized", "cpo")
COSTS = ("c_inventory", "c_notional", "c_margin")
CONSTRAINED = ("c_inventory", "c_notional")
ATTRIBUTION = ("spread_capture", "inventory_pnl", "quote_fees", "fills", "max_drawdown",
               "mean_abs_position", "max_abs_position")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values, Decimal(0)) / len(values) if values else None


def constrained_cost(row):
    """The day's mean constrained cost per step, over its episodes."""
    return mean(sum(Decimal(str(e["cost_means"][k])) for k in CONSTRAINED)
                for e in row["episodes"])


def daily(rows, value):
    """Day key to the arm's value; a learned arm's is the mean over its seeds."""
    by_day = {}
    for row in rows:
        by_day.setdefault((row["regime"], row["day"]), []).append(value(row))
    return {key: mean(values) for key, values in sorted(by_day.items())}


def net(row):
    return Decimal(row["daily_value"])


def breaches(rows):
    return sum(bool(e["limit_breach"]) for r in rows for e in r["episodes"])


def summary(rows, budgets):
    """One arm in one row and scope; per day counts are over every seed's day."""
    episodes = [e for r in rows for e in r["episodes"]]
    refused = {}
    for e in episodes:
        for reason, count in e["actions_refused"].items():
            refused[reason] = refused.get(reason, 0) + count

    def per_day(value):
        return str(mean(Decimal(str(value(e))) for e in episodes))

    return {
        "days": len(daily(rows, net)),
        "seed_days": len(rows),
        "mean_daily_net": str(mean(daily(rows, net).values())),
        "mean_daily_constrained_cost": str(mean(daily(rows, constrained_cost).values())),
        "costs": {k: {"mean_per_step": per_day(lambda e, k=k: e["cost_means"][k]),
                      "budget": budgets.get(k),
                      "episodes_over_budget": sum(bool(e["over_budget"].get(k))
                                                  for e in episodes) if k in budgets else None}
                  for k in COSTS},
        "legs_rejected_per_day": per_day(lambda e: e["legs_rejected"]),
        "actions_refused": dict(sorted(refused.items())),
        "limit_breaches": breaches(rows),
        "peak_notional": str(max(Decimal(e["peak_notional"]) for e in episodes)),
        "attribution": {k: per_day(lambda e, k=k: e["attribution"][k] or 0)
                        for k in ATTRIBUTION},
        "orders_per_day": {k: per_day(lambda e, k=k: e["orders"][k])
                           for k in ("accepted", "canceled", "cancel_rejected", "rejected",
                                     "fills")},
    }


def checked(out):
    """The holdout report, after its manifest matches."""
    holdout = json.loads((out / "holdout.json").read_text())
    if sha256(out / "frozen.json") != holdout["frozen_sha256"]:
        raise ValueError("frozen.json is not the manifest the holdout was evaluated from")
    return holdout


def evaluation_rows(holdout, fee, secondary):
    """The design's rows: name to the holdout rows in it."""
    rows = holdout["rows"]
    out = {"unshocked": [r for r in rows if r["fee"] == fee and r["shock"] is None]}
    for shock in sorted({r["shock"] for r in rows if r["shock"] is not None}):
        out[f"shock {shock}"] = [r for r in rows if r["fee"] == fee and r["shock"] == shock]
    if secondary != fee:
        out["secondary fee"] = [r for r in rows if r["fee"] == secondary and r["shock"] is None]
    return out


def scopes_of(rows):
    regimes = sorted({r["regime"] for r in rows})
    scopes = {regime: [regime] for regime in regimes}
    scopes["pooled"] = regimes
    return scopes


def difference(a, b):
    if a.keys() != b.keys():
        raise ValueError("the two arms do not have the same days")
    return [str(a[day] - b[day]) for day in a]


PLAN = ([("constrained", "cpo", "ppo_penalized")]
        + [("against no_trade", arm, "no_trade") for arm in LEARNED]
        + [("against ppo", arm, "ppo") for arm in ("ppo_penalized", "cpo")])


def build(holdout, fee, secondary, budgets):
    """Tables, and the series to bootstrap per row and scope; `name` keys the comparisons."""
    tables, series, plan = {}, {}, []
    for row_name, chosen_rows in evaluation_rows(holdout, fee, secondary).items():
        for scope, members in scopes_of(chosen_rows).items():
            chosen = [r for r in chosen_rows if r["regime"] in members]
            by_arm = {}
            for r in chosen:
                by_arm.setdefault(r["arm"], []).append(r)
            table = {arm: summary(rows, budgets) for arm, rows in sorted(by_arm.items())}
            for arm in LEARNED:
                if arm in by_arm:
                    table[arm]["per_seed"] = {
                        seed: summary([r for r in by_arm[arm] if r["candidate"] == seed], budgets)
                        for seed in sorted({r["candidate"] for r in by_arm[arm]})}
            tables.setdefault(row_name, {})[scope] = table
            prefix = f"{row_name}/{scope}"
            for kind, a, b in PLAN:
                name = f"{prefix}/{a}-minus-{b}"
                series[f"{name}/net"] = difference(daily(by_arm[a], net), daily(by_arm[b], net))
                series[f"{name}/cost"] = difference(daily(by_arm[a], constrained_cost),
                                                    daily(by_arm[b], constrained_cost))
                plan.append({"kind": kind, "row": row_name, "scope": scope, "name": name,
                             "arm": a, "against": b,
                             "breaches": breaches(by_arm[a]) + breaches(by_arm[b])})
    return tables, series, plan


def bounds(interval):
    if any(interval.get(key) is None for key in ("mean", "lower", "upper")):
        raise ValueError(f"no interval to decide on: {interval}")
    return Decimal(str(interval["lower"])), Decimal(str(interval["upper"]))


def decide(item, net_interval, cost_interval):
    """The design's rule: net not below 0, cost below 0, no breach."""
    _, net_upper = bounds(net_interval)
    _, cost_upper = bounds(cost_interval)
    holds = net_upper >= 0 and cost_upper < 0 and item["breaches"] == 0
    if item["kind"] == "constrained":
        return "constrained better" if holds else "not shown"
    return "better by the rule" if holds else "not shown"


def main():
    out = pathlib.Path(sys.argv[1])
    binary = sys.argv[2]
    design = json.loads((pathlib.Path(__file__).resolve().parent / "design.json").read_text())
    holdout = checked(out)
    frozen = json.loads((out / "frozen.json").read_text())
    budgets = {k: Decimal(v) for k, v in
               design["costs"]["budgets_per_step_mean_over_an_episode"].items()}
    fee, secondary = frozen["fee"], design["execution"]["secondary_fee_rate"]
    tables, series, plan = build(holdout, fee, secondary, {k: str(v) for k, v in budgets.items()})
    payload = {"bootstrap": design["evaluation"]["bootstrap"], "series": series}
    (out / "series.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    intervals = json.loads(subprocess.run([binary, str(out / "series.json")], check=True,
                                          capture_output=True, text=True).stdout)
    decisions = [item | {"net_interval": intervals[f"{item['name']}/net"],
                         "cost_interval": intervals[f"{item['name']}/cost"],
                         "decision": decide(item, intervals[f"{item['name']}/net"],
                                            intervals[f"{item['name']}/cost"])}
                 for item in plan]
    report = {
        "commit": holdout["commit"], "data": holdout["data"], "bounds": holdout["bounds"],
        "frozen_sha256": holdout["frozen_sha256"], "holdout_sha256": sha256(out / "holdout.json"),
        "fee": fee, "secondary_fee": secondary, "penalty_selected": frozen["penalty_selected"],
        "tables": tables, "intervals": intervals, "decisions": decisions,
        "note": "nothing here is a safety guarantee",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"decisions": len(decisions), "report_sha256": sha256(out / "report.json")}))


if __name__ == "__main__":
    main()
