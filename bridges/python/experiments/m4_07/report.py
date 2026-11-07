"""The report of #104 from a step's output, as markdown.

    python experiments/m4_07/report.py <out dir>/holdout.json

Every value carries its label: [observed], [estimated] or [simulated].
"""

import json
import sys
from decimal import Decimal


def fmt(value, places=2):
    if value is None:
        return "n/a"
    return f"{float(value):.{places}f}"


def interval(triple):
    point, low, high = triple
    return f"{fmt(point)} [{fmt(low)}, {fmt(high)}]"


def rows_table(report):
    truth = report["truth"]
    lines = [
        "| behaviour | target | window | PDWIS [estimated] | DR [estimated] | DM [estimated] "
        "| truth [simulated] | ESS | max weight | decision | reasons |",
        "|---|---|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        d = row["diagnostics"]
        t = truth.get(row["target"], {}).get("mean_daily_net")
        e = row["estimates"]
        lines.append(
            f"| {row['behaviour']} | {row['target']} | {row['window']} "
            f"| {interval(e['pdwis_window'])} | {interval(e['dr'])} | {interval(e['dm'])} "
            f"| {fmt(Decimal(t)) if t is not None else 'n/a'} | {fmt(d['effective_sample'], 4)} "
            f"| {fmt(d['max_weight'], 4)} | {row['decision']} | {', '.join(row['reasons']) or '-'} |"
        )
    for target, why in report["out_of_support_by_construction"].items():
        lines.append(f"| any | {target} | - | - | - | - | - | - | - | no_go | out_of_support: {why} |")
    return "\n".join(lines)


def calibration(report):
    """How often a go's interval held the simulated truth, and how often a no-go's did: the
    check of the diagnostics the design promises, read only after the decision."""
    counts = {"go": [0, 0], "no_go": [0, 0]}
    for row in report["rows"]:
        t = report["truth"].get(row["target"], {}).get("mean_daily_net")
        if t is None:
            continue
        _, low, high = row["estimates"]["pdwis_window"]
        held = low <= float(t) <= high
        counts[row["decision"]][0] += held
        counts[row["decision"]][1] += 1
    return counts


def main():
    report = json.loads(open(sys.argv[1]).read())
    print(f"## #104 offline evaluation support: {report['step']}\n")
    print(report["data"] + "\n")
    print(report["bounds"] + "\n")
    print(report["disclosure"] + "\n")
    print(rows_table(report) + "\n")
    counts = calibration(report)
    for decision, (held, total) in counts.items():
        print(f"- {decision}: the PDWIS interval held the [simulated] value in {held} of {total} rows.")
    if "real_data" in report:
        print("\n**Real data:** " + report["real_data"])


if __name__ == "__main__":
    main()
