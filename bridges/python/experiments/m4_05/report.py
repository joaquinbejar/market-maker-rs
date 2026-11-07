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
Avellaneda-Stoikov. The bounds the ablations put on the execution are in EXECUTION_BOUNDS.

A correction, made after report.py was first run on the holdout (see FIRST_READ): the rows of
`holdout.json` carry no ablation label, so the learned runs are labelled from the order the
runner submitted them in (see `relabel`), and every row must match it.
"""

import hashlib
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
EXECUTION_BOUNDS = (
    "Found on validation by the ablations, before the holdout was read; the design was not "
    "changed. (1) The main execution is in effect immediate fill: the queue model's rate "
    "cannot be below one lot per millisecond (#267), so an order clears about 9975 lots by "
    "the next 10 s row, more than any touch (8 lots at most on training and validation), and "
    "the fill_immediate ablation gave the same result as the main execution on every "
    "validation row. Queue position is therefore absent from this comparison. (2) The "
    "latency_one_row ablation is degenerate: an order that arrives exactly on the next row "
    "cannot match on it, and the cancel sent by that row's quote arrives on the row after, "
    "before matching (#268), so no arm fills. It does not measure latency. Conclusions hold "
    "only for immediate fills at 10 s rows with a latency well under a row."
)
FIRST_READ = (
    "The holdout was evaluated once. The first run of this report on it counted the learned "
    "runs trained with the reward_inventory_penalty ablation together with the main runs, "
    "because the runner wrote no ablation label into the holdout rows: the learned arms' tables "
    "had 150 pooled days instead of 75, and their decisions were made on that mixture. What it "
    "showed: maskable PPO 'continue' in calm (interval against GLFT [0.0016, 3.91], against "
    "no-trade [-6.65, -1.95]) and 'revise', equivalent to not trading, in stressed, thin and "
    "pooled; the residual SAC 'discard' everywhere against GLFT and better than "
    "Avellaneda-Stoikov everywhere; the PPO run that traded heavily was a penalty-trained one. "
    "The bug was then found and the rows relabelled from the order the runner submitted them "
    "in, with no new evaluation and no change to any rule or threshold; the tables and "
    "decisions here are from the relabelled rows only."
)
TRAINING_ONLY = "reward_inventory_penalty"


def relabel(out, holdout, design):
    """Labels the holdout's learned rows with their ablation, from submission order.

    The runner submits every classical job first, then each frozen learned run, in the order
    of `frozen.json`, over the regimes and then the holdout days, and keeps the results in
    that order. Every row must match the job it is labelled from on arm, candidate,
    execution variant, regime and day, or nothing is reported.
    """
    path = out / "frozen.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != holdout["frozen_sha256"]:
        raise ValueError("frozen.json is not the manifest the holdout was evaluated from")
    frozen = json.loads(path.read_text())
    regimes = [r["name"] for r in design["simulator"]["regimes"]]
    first, last = design["splits"]["holdout"]
    expected = [
        (m["arm"], f"set{m['set']}-seed{m['seed']}", m["variant"], regime, day,
         m.get("ablation"))
        for m in frozen["runs"]
        for regime in regimes
        for day in range(first, last + 1)
    ]
    rows = holdout["rows"]
    learned = [i for i, row in enumerate(rows) if row["arm"] in LEARNED]
    start = len(rows) - len(expected)
    if learned != list(range(start, len(rows))):
        raise ValueError("the learned rows are not the last rows, in one block")
    for row in rows[:start]:
        if row.get("ablation") is not None:
            raise ValueError(f"a classical row carries an ablation: {row['arm']}")
        row["ablation"] = None
    for row, (arm, candidate, variant, regime, day, ablation) in zip(rows[start:], expected):
        seen = (row["arm"], row["candidate"], row["variant"], row["regime"], row["day"])
        submitted = (arm, candidate, variant, regime, day)
        if seen != submitted:
            raise ValueError(f"row {seen} is not the submitted job {submitted}")
        if row.get("ablation", ablation) != ablation:
            raise ValueError(f"row {seen} is labelled {row['ablation']}, submitted {ablation}")
        row["ablation"] = ablation
    return rows


def mean(values):
    values = list(values)
    return sum(values, Decimal(0)) / len(values) if values else None


def cvar(values, share=Decimal("0.05")):
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * share))
    return mean(ordered[:count])


def summary(rows):
    """The table row of one arm over some days.

    A day is a regime and day; a learned arm's day is the mean over its seeds, so the
    daily net, its CVaR and the drawdowns are over those days. Means of the attribution and
    positions are over every seed's day, which is the same for equal seeds per day; the
    largest position, peak notional and time beyond 5 lots are the largest of any seed's day.
    """
    episodes = [r["episodes"][0] for r in rows]
    nets = list(daily(rows).values())
    drawdowns = list(drawdown_by_day(rows).values())
    steps = sum(e["steps"] for e in episodes)
    return {
        "days": len(nets),
        "seed_days": len(rows),
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


def ablation_tables(rows, regimes):
    """Per ablation, the tables of every arm it evaluated on the holdout. An execution
    ablation has the classical arms and the learned runs retrained under it; the training-only
    one has the learned runs it trained, under the main execution."""
    labels = sorted({r["variant"] for r in rows if r["variant"] != "main"}) + [TRAINING_ONLY]
    out = {}
    for label in labels:
        if label == TRAINING_ONLY:
            chosen = [r for r in rows if r["ablation"] == label]
        else:
            chosen = [r for r in rows if r["variant"] == label]
        scopes = {regime: [regime] for regime in regimes}
        scopes["pooled"] = regimes
        out[label] = {
            scope: {
                arm: summary([r for r in chosen if r["arm"] == arm and r["regime"] in members])
                for arm in sorted({r["arm"] for r in chosen})
            }
            for scope, members in scopes.items()
        }
    return out


def build(out, design):
    holdout = json.loads((out / "holdout.json").read_text())
    labelled = relabel(out, holdout, design)
    rows = [r for r in labelled if r["ablation"] is None and r["variant"] == "main"]
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
    return holdout, tables, series, comparisons, rows, ablation_tables(labelled, regimes)


def bounds(interval):
    """`(mean, lower, upper)` of an interval; a series without one is an error, not a decision."""
    if any(interval.get(key) is None for key in ("mean", "lower", "upper")):
        raise ValueError(f"no interval to decide on: {interval}")
    return interval["mean"], interval["lower"], interval["upper"]


def includes_zero(interval):
    _, lower, upper = bounds(interval)
    return lower <= 0 <= upper


def decide(interval, learned, classical, breaches):
    point, lower, _ = bounds(interval)
    drawdown_ok = learned <= Decimal("1.2") * classical
    if lower > 0 and breaches == 0 and drawdown_ok:
        return "continue"
    if point > 0:
        return "revise"
    return "discard"


def main():
    out = pathlib.Path(sys.argv[1])
    binary = sys.argv[2]
    design = json.loads((pathlib.Path(__file__).resolve().parent / "design.json").read_text())
    holdout, tables, series, comparisons, rows, ablations = build(out, design)
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
        "first_read": FIRST_READ,
        "amendment": AMENDMENT,
        "execution_bounds": EXECUTION_BOUNDS,
        "tables": tables,
        "ablation_tables": ablations,
        "intervals": intervals,
        "decisions": decisions,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(decisions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
