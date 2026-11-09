"""The holdout tables of #108 part 2, from holdout.json and frozen.json. SYNTHETIC.

    python experiments/m5_10/report.py <out dir>

Descriptive per arm and setting: an interval holding 0 is 'no difference shown', never
equivalence or robustness. Margin shortfall is not measured: only the requirement is reported.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import stats  # noqa: E402

MAINTENANCE = 0.05  # margin-rs linear-im/1, as #108's part 1
NOTES = ["SYNTHETIC: made-up Hawkes parameters; says nothing about a market.",
         "The impact is the market orders' on the mid; the agent's own fills do not move it.",
         "The classical set changed from the first design: GLFT replaces fixed spread, since #107 part C selected "
         "only Avellaneda-Stoikov and GLFT.",
         "The mean-loss adversary is not comparable with #295's CVaR adversary.",
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


def peak_position(row):
    return float(row["result"]["episode"]["attribution"]["max_abs_position"])


def requirement(row):
    return MAINTENANCE * float(row["result"]["episode"]["peak_notional"])


def breach(row):
    return bool(row["result"]["episode"]["limit_breach"])


def tables(rows, selected):
    grouped = daily(rows)
    out = []
    for (setting, arm, condition), by_day in sorted(grouped.items()):
        values = series(by_day, net)
        entry = {"setting": setting, "arm": arm, "condition": condition, "mean_net": float(values.mean()),
                 "cvar5_loss": stats.cvar_loss(values), "max_abs_position": float(series(by_day, peak_position).max()),
                 "margin_requirement_max": float(series(by_day, requirement).max()),
                 "limit_breaches": int(sum(breach(r) for day in by_day.values() for r in day))}
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


def main(out):
    out = pathlib.Path(out)
    rows = json.loads((out / "holdout.json").read_text())
    selected = json.loads((out / "frozen.json").read_text())["selected"]
    (out / "report.json").write_text(json.dumps({"notes": NOTES, "rows": tables(rows, selected)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
