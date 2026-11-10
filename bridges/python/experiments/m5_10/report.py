"""The holdout tables of #108 part 2, from holdout.json and frozen.json. SYNTHETIC.

    python experiments/m5_10/report.py <out dir> <identify.json>

Descriptive per arm and setting: an interval holding 0 is 'no difference shown', never
equivalence or robustness. Margin shortfall is not measured: only the requirement is reported.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import grid  # noqa: E402
import stats  # noqa: E402
from run import SETTINGS, point_id  # noqa: E402

MAINTENANCE = 0.05  # margin-rs linear-im/1, as #108's part 1
NOTES = ["SYNTHETIC: made-up Hawkes parameters; says nothing about a market.",
         "The impact is the market orders' on the mid; the agent's own fills do not move it.",
         "The classical set changed from the first design: GLFT replaces fixed spread, since #107 part C selected "
         "only Avellaneda-Stoikov and GLFT.",
         "The mean-loss adversary is not comparable with #295's CVaR adversary.",
         "The primary's adversary searches only the points that passed the frozen rate test, likely the milder "
         "flows; a loss or no difference under it is limited to those points, and fixed stress was not run.",
         "The secondary's adversary searches every point; never compare it with the primary's as a robustness test.",
         "Margin shortfall: not measured; this simulator has no declared capital."]


def daily(rows):
    """{(setting, arm, condition): {day: value of a day}}, the seeds of an arm averaged."""
    grouped = {}
    for row in rows:
        p = row["policy"]
        key = (p["setting"], p["arm"], row["condition"])
        grouped.setdefault(key, {}).setdefault(row["day"], []).append(row)
    return grouped


def series(by_day, field):
    days = sorted(by_day)
    return np.array([np.mean([field(r) for r in by_day[d]]) for d in days])


def net(row):
    return float(row["result"]["net"])


def attribution(row, name):
    return float(row["result"]["episode"]["attribution"][name])


def largest(by_day, field):
    """The largest over every seed and day, so seed averaging cannot understate it."""
    return max(field(r) for day in by_day.values() for r in day)


def requirement(row):
    return MAINTENANCE * float(row["result"]["episode"]["peak_notional"])


def breach(row):
    return bool(row["result"]["episode"]["limit_breach"])


def _pid(point):
    return point_id(point)


def check_primary(rows):
    """The primary's table refuses a secondary row."""
    if any(r["condition"].startswith("secondary_") for r in rows):
        raise ValueError("a secondary row in the primary's table")


def secondary_table(rows, primary_rows, identify=None, secondary_picks=None, primary_picks=None):
    """Exploratory, 'not identified by the frozen rate test'; feeds no decision.

    A policy whose pick over every point is its primary pick reuses those days, relabelled here.
    """
    base = daily(primary_rows)
    reused = []
    for row in primary_rows:
        key = f"{row['policy']['setting']}/{row['policy']['arm']}"
        if row["condition"] == "adversary" and secondary_picks and secondary_picks[key]["point"] == primary_picks[key]["point"]:
            reused.append({**row, "condition": "secondary_adversary"})
    reused_keys = {(r["policy"]["setting"], r["policy"]["arm"], "secondary_adversary") for r in reused}
    grouped = daily(list(rows) + reused)
    out = []
    for (setting, arm, condition), by_day in sorted(grouped.items()):
        values = series(by_day, net)
        interval = stats.paired_interval(values - series(base[(setting, arm, "unperturbed")], net))
        hold = grouped.get((setting, "no_trade", condition))
        against_hold = None
        if hold is not None and arm != "no_trade":
            against_hold = stats.paired_interval(values - series(hold, net))
        ratios = {}
        if identify:
            points = identify[setting]["points"]
            base_point = points[next(iter(points))]
            for pid in sorted({_pid(r["point"]) for d in by_day.values() for r in d}):
                p = points[pid]
                ratios[pid] = {"realised_over_formula": (p["market_rate"] / base_point["market_rate"])
                               / (p["formula"] / base_point["formula"]),
                               "empty_share": p["empty_share"], "cap_share": p["cap_share"]}
        out.append({"label": "not identified by the frozen rate test, exploratory", "setting": setting, "arm": arm,
                    "condition": condition, "reused_primary_rows": (setting, arm, condition) in reused_keys,
                    **metrics(by_day), "against_unperturbed": interval, "against_no_trade": against_hold,
                    "reading": stats.reading(interval), "points_rate_test": ratios,
                    "selection_loss": (secondary_picks or {}).get(f"{setting}/{arm}", {}).get("selection_loss")
                    if condition == "secondary_adversary" else None,
                    "holdout_mean_loss": -float(values.mean()) if condition == "secondary_adversary" else None})
    return out


def metrics(by_day):
    """The metrics shared by the primary's and the secondary's rows."""
    values = series(by_day, net)
    return {"mean_net": float(values.mean()), "cvar5_loss": stats.cvar_loss(values),
            "max_abs_position": largest(by_day, lambda r: attribution(r, "max_abs_position")),
            "mean_abs_position": float(series(by_day, lambda r: attribution(r, "mean_abs_position")).mean()),
            "max_drawdown": largest(by_day, lambda r: attribution(r, "max_drawdown")),
            "refusals": int(sum(r["result"]["episode"]["refusals"] for day in by_day.values() for r in day)),
            "attribution_mean": {k: float(series(by_day, lambda r: attribution(r, k)).mean())
                                 for k in ("spread_capture", "inventory_pnl")},
            "margin_requirement_max": largest(by_day, requirement),
            "limit_breaches": int(sum(breach(r) for day in by_day.values() for r in day))}


def tables(rows, selected):
    check_primary(rows)
    grouped = daily(rows)
    out = []
    for (setting, arm, condition), by_day in sorted(grouped.items()):
        values = series(by_day, net)
        entry = {"setting": setting, "arm": arm, "condition": condition, **metrics(by_day)}
        base = grouped.get((setting, arm, "unperturbed"))
        if base is not None and condition != "unperturbed":
            interval = stats.paired_interval(values - series(base, net))
            entry["against_unperturbed"] = {"interval": interval, "reading": stats.reading(interval)}
        hold = grouped.get((setting, "no_trade", condition))
        if hold is not None and arm != "no_trade":
            interval = stats.paired_interval(values - series(hold, net))
            entry["against_no_trade"] = {"interval": interval, "reading": stats.reading(interval)}
        if condition == "adversary":
            pick = selected[f"{setting}/{arm}"]
            entry["selection_loss"] = pick["selection_loss"]
            entry["holdout_mean_loss"] = -float(values.mean())
        out.append(entry)
    return out


def not_run(identify):
    """Per setting, the conditions not run because of a removed point, with the queue shares."""
    out = {}
    for setting, v in identify.items():
        impact = SETTINGS[setting]
        named = {"fixed_stress": grid.fixed_stress(impact), "impact_only": grid.impact_only(impact)}
        conditions = ["unperturbed", "fixed_stress", "adversary", "impact_only"] if v["void"] else \
            [name for name, point in named.items() if point_id(point) in v["removed"]]
        out[setting] = {"void": v["void"], "conditions_not_run": conditions, "removed_points": v["removed"],
                        "spectral_removed": v["spectral_removed"],
                        "shares": {k: {"empty": p["empty_share"], "cap": p["cap_share"]} for k, p in v["points"].items()}}
    return out


def main(out, identify_path):
    out = pathlib.Path(out)
    rows = json.loads((out / "holdout.json").read_text())
    secondary_path = out / "holdout_secondary.json"
    failed_path = out / "holdout_secondary_failed.json"
    failure = json.loads(failed_path.read_text()) if failed_path.exists() else None
    secondary = json.loads(secondary_path.read_text()) if secondary_path.exists() else []
    frozen = json.loads((out / "frozen.json").read_text())
    selected = frozen["selected"]
    identify = json.loads(pathlib.Path(identify_path).read_text())
    (out / "report.json").write_text(json.dumps({"notes": NOTES, "rows": tables(rows, selected),
                                                     "secondary_failure": failure,
                                                     "secondary": None if failure else secondary_table(
                                                         secondary, rows, identify, frozen["selected_secondary"], selected),
                                                     "not_run": not_run(json.loads(pathlib.Path(identify_path).read_text()))}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
