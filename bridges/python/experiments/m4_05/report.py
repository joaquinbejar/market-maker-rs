"""Tables, intervals and decisions of the comparison of #102, from the runner's reports.

    python experiments/m4_05/report.py <out dir> <intervals binary>

reads `selection.json`, `ablations.json` and `holdout.json` from `<out dir>`, writes the
series to bootstrap to `<out dir>/series.json`, runs the intervals binary (the example
`m4_05_intervals`, the evaluation's block bootstrap) on it, and writes `report.json`.

This file fixes, before the holdout is read, how the design's metrics and decision rule are
read:
- A day's value is the day's exact net result. A learned arm's day is the mean over its five
  training seeds, and each seed is also reported apart.
- Max drawdown is the mean over days of each day's largest drawdown, and the worst day is
  reported beside it. The rule's drawdown condition compares those means.
- CVaR 5 % of the daily net is the mean of the worst ceil(5 % of the days), which with five
  holdout days per regime is the worst day.
- Pooled means the days of every regime together.
- The best classical arm is Avellaneda-Stoikov or GLFT, whichever has the higher mean daily
  net on the holdout, per regime and pooled; GLFT's quote rate is reported with it.

An amendment, made after the validation results were seen and before the holdout was read
(see AMENDMENT): each learned arm is also compared with no-trade, and with
Avellaneda-Stoikov.
"""

import json
import math
import pathlib
import subprocess
import sys
from decimal import Decimal

LEARNED = ("maskable_ppo", "sac_residual_avellaneda_stoikov")
CLASSICAL = ("avellaneda_stoikov", "glft")
AGAINST = CLASSICAL + ("no_trade",)
AMENDMENT = (
    "Made after the validation results of the selection step were seen, and before the "
    "holdout was read. Read literally, the decision rule calls a learned arm that never "
    "trades a success whenever the best classical arm loses, which it did on validation. So "
    "each learned arm is also given a paired interval against no-trade: when that interval "
    "includes 0, the arm is reported as equivalent to not trading, and its decision is "
    "revise at best. The rule against the best classical arm is unchanged. Since the best "
    "classical arm on validation was GLFT, which mostly does not quote (#264), each learned "
    "arm is also reported against Avellaneda-Stoikov, the classical arm that quotes."
)


def mean(values):
    values = list(values)
    return sum(values, Decimal(0)) / len(values) if values else None


def cvar(values, share=Decimal("0.05")):
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * share))
    return mean(ordered[:count])


def summary(rows):
    """The table row of one arm over some days."""
    episodes = [r["episodes"][0] for r in rows]
    nets = [Decimal(r["daily_value"]) for r in rows]
    drawdowns = [Decimal(e["attribution"]["max_drawdown"]) for e in episodes]
    steps = sum(e["steps"] for e in episodes)
    return {
        "days": len(rows),
        "mean_daily_net": str(mean(nets)),
        "spread_capture": str(mean(Decimal(e["attribution"]["spread_capture"]) for e in episodes)),
        "inventory_pnl": str(mean(Decimal(e["attribution"]["inventory_pnl"]) for e in episodes)),
        "fees": str(mean(Decimal(e["attribution"]["quote_fees"]) for e in episodes)),
        "fills_per_day": str(mean(Decimal(e["attribution"]["fills"]) for e in episodes)),
        "mean_max_drawdown": str(mean(drawdowns)),
        "worst_max_drawdown": str(max(drawdowns)),
        "cvar_5_daily_net": str(cvar(nets)),
        "mean_abs_position": str(mean(Decimal(e["attribution"]["mean_abs_position"] or 0)
                                      for e in episodes)),
        "max_abs_position": str(max(Decimal(e["attribution"]["max_abs_position"])
                                    for e in episodes)),
        "peak_notional": str(max(Decimal(e["peak_notional"]) for e in episodes)),
        "time_beyond_5_lots": max(e["time_beyond_5_lots"] for e in episodes),
        "refusal_rate": sum(e["refusals"] for e in episodes) / steps,
        "residual_fallback_rate": sum(e["residual_fallbacks"] for e in episodes) / steps,
        "residual_clipped_rate": sum(e["residual_clipped"] for e in episodes) / steps,
        "limit_breaches": sum(bool(e["limit_breach"]) for e in episodes),
    }


def daily(rows):
    """Day key to the arm's daily value; a learned arm's is the mean over its seeds."""
    by_day = {}
    for row in rows:
        by_day.setdefault((row["regime"], row["day"]), []).append(Decimal(row["daily_value"]))
    return {key: mean(values) for key, values in sorted(by_day.items())}


def drawdown_by_day(rows):
    by_day = {}
    for row in rows:
        value = Decimal(row["episodes"][0]["attribution"]["max_drawdown"])
        by_day.setdefault((row["regime"], row["day"]), []).append(value)
    return {key: mean(values) for key, values in sorted(by_day.items())}


def build(out):
    holdout = json.loads((out / "holdout.json").read_text())
    rows = [r for r in holdout["rows"] if r.get("ablation") is None and r["variant"] == "main"]
    regimes = sorted({r["regime"] for r in rows})
    scopes = {regime: [regime] for regime in regimes}
    scopes["pooled"] = regimes
    arms = sorted({r["arm"] for r in rows})
    tables, series, comparisons = {}, {}, {}
    for scope, members in scopes.items():
        chosen = [r for r in rows if r["regime"] in members]
        tables[scope] = {arm: summary([r for r in chosen if r["arm"] == arm]) for arm in arms}
        for arm in LEARNED:
            seeds = sorted({r["candidate"] for r in chosen if r["arm"] == arm})
            tables[scope][arm]["per_seed"] = {
                seed: summary([r for r in chosen if r["arm"] == arm and r["candidate"] == seed])
                for seed in seeds
            }
        values = {arm: daily([r for r in chosen if r["arm"] == arm]) for arm in arms}
        for arm in arms:
            series[f"{scope}/{arm}"] = [str(v) for v in values[arm].values()]
        best = max(CLASSICAL, key=lambda a: mean(values[a].values()))
        comparisons[scope] = {"best_classical": best}
        for arm in LEARNED:
            for against in AGAINST:
                difference = [values[arm][day] - values[against][day] for day in values[arm]]
                series[f"{scope}/{arm}-minus-{against}"] = [str(d) for d in difference]
    return holdout, tables, series, comparisons, rows


def includes_zero(interval):
    return interval.get("lower") is not None and interval["lower"] <= 0 <= interval["upper"]


def decide(interval, learned, classical, breaches):
    lower = interval.get("lower")
    point = interval.get("mean")
    drawdown_ok = learned <= Decimal("1.2") * classical
    if lower is not None and lower > 0 and breaches == 0 and drawdown_ok:
        return "continue"
    if point is not None and point > 0:
        return "revise"
    return "discard"


def main():
    out = pathlib.Path(sys.argv[1])
    binary = sys.argv[2]
    design = json.loads((pathlib.Path(__file__).resolve().parent / "design.json").read_text())
    holdout, tables, series, comparisons, rows = build(out)
    payload = {"bootstrap": design["evaluation"]["bootstrap"], "series": series}
    (out / "series.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    intervals = json.loads(
        subprocess.run([binary, str(out / "series.json")], check=True, capture_output=True,
                       text=True).stdout
    )
    decisions = {}
    for scope, comparison in comparisons.items():
        best = comparison["best_classical"]
        members = [scope] if scope != "pooled" else sorted(k for k in tables if k != "pooled")
        for arm in LEARNED:
            chosen = [r for r in rows if r["regime"] in members]
            learned_dd = mean(drawdown_by_day([r for r in chosen if r["arm"] == arm]).values())
            classical_dd = mean(drawdown_by_day([r for r in chosen if r["arm"] == best]).values())
            breaches = tables[scope][arm]["limit_breaches"]
            interval = intervals[f"{scope}/{arm}-minus-{best}"]
            decision = decide(interval, learned_dd, classical_dd, breaches)
            against_none = intervals[f"{scope}/{arm}-minus-no_trade"]
            equivalent = includes_zero(against_none)
            if equivalent and decision == "continue":
                decision = "revise"
            as_dd = mean(drawdown_by_day(
                [r for r in chosen if r["arm"] == "avellaneda_stoikov"]).values())
            against_as = intervals[f"{scope}/{arm}-minus-avellaneda_stoikov"]
            decisions[f"{scope}/{arm}"] = {
                "against": best,
                "interval": interval,
                "mean_max_drawdown": [str(learned_dd), str(classical_dd)],
                "limit_breaches": breaches,
                "against_no_trade": against_none,
                "equivalent_to_not_trading": equivalent,
                "decision": decision,
                "against_avellaneda_stoikov": {
                    "interval": against_as,
                    "mean_max_drawdown": [str(learned_dd), str(as_dd)],
                    "by_the_same_rule": decide(against_as, learned_dd, as_dd, breaches),
                },
            }
    report = {
        "commit": holdout["commit"],
        "data": holdout["data"],
        "bounds": holdout["bounds"],
        "frozen_sha256": holdout["frozen_sha256"],
        "amendment": AMENDMENT,
        "tables": tables,
        "intervals": intervals,
        "decisions": decisions,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(decisions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
