"""Tables, intervals and decisions of #105's second, margin-stress holdout.

    python experiments/m4_08/report_stress.py <run out dir> <stress out dir> <intervals binary>

reads the first holdout's `frozen.json` and `stress_holdout.json`, writes the series to
bootstrap to `<stress out dir>/stress_series.json`, runs the intervals binary (the example
`m4_05_intervals`) on it, and writes `stress_report.json`. This file fixes, before the
second holdout is read, how it is read (`stress_design.json`):
- A day is a regime and new holdout day; its value is the share of its steps not Healthy
  under margin-rs's `linear-im/1`; a learned arm's day is the mean over its seeds, and each
  seed is also reported apart.
- The rows, read apart: unshocked, and each shock; each per regime and pooled.
- Fewer margin states: the paired 90 % block bootstrap interval of the daily difference is
  below 0 (its upper bound is), and neither arm breached a limit on any day.
- The comparisons: cpo against ppo_penalized; cpo and ppo against the selected
  Avellaneda-Stoikov point; ppo_penalized and cpo against ppo.
- An arm without a fill in a row and scope is Healthy by construction there: a comparison
  against it is flagged as trading against not trading.
- It is the second holdout and does not revise the first's decisions. Nothing here is a
  safety guarantee.
"""

import json
import pathlib
import subprocess
import sys
from decimal import Decimal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report import breaches, difference, mean, scopes_of, sha256  # noqa: E402

LEARNED = ("ppo", "ppo_penalized", "cpo")
STATES = ("healthy", "reduce_only", "default")
PLAN = (("cpo", "ppo_penalized"), ("cpo", "avellaneda_stoikov"),
        ("ppo", "avellaneda_stoikov"), ("ppo_penalized", "ppo"), ("cpo", "ppo"))


def share(row):
    """A day's share of steps not Healthy, over its episodes."""
    return mean(Decimal(str(e["margin"]["not_healthy_share"])) for e in row["episodes"])


def daily(rows, value):
    by_day = {}
    for row in rows:
        by_day.setdefault((row["regime"], row["day"]), []).append(value(row))
    return {key: mean(values) for key, values in sorted(by_day.items())}


def fills(rows):
    return sum(int(e["attribution"]["fills"]) for r in rows for e in r["episodes"])


def summary(rows):
    """One arm in one row and scope; per day figures are over every seed's day."""
    episodes = [e for r in rows for e in r["episodes"]]

    def per_day(value):
        return str(mean(Decimal(str(value(e))) for e in episodes))

    return {
        "days": len(daily(rows, share)),
        "seed_days": len(rows),
        "mean_not_healthy_share": str(mean(daily(rows, share).values())),
        "steps_per_day": {s: per_day(lambda e, s=s: e["margin"]["steps"][s]) for s in STATES},
        "entries_per_day": per_day(lambda e: e["margin"]["entries"]),
        "default_entries_per_day": per_day(lambda e: e["margin"]["entries_default"]),
        "largest_shortfall": str(max(Decimal(e["margin"]["largest_shortfall"])
                                     for e in episodes)),
        "max_abs_position": str(max(Decimal(e["margin"]["max_abs_position"])
                                    for e in episodes)),
        "fills_per_day": per_day(lambda e: e["attribution"]["fills"]),
        "mean_daily_net": str(mean(Decimal(r["daily_value"]) for r in rows)),
        "limit_breaches": breaches(rows),
    }


def evaluation_rows(stress):
    rows = stress["rows"]
    out = {"unshocked": [r for r in rows if r["shock"] is None]}
    for shock in sorted({r["shock"] for r in rows if r["shock"] is not None}):
        out[f"shock {shock}"] = [r for r in rows if r["shock"] == shock]
    return out


def build(stress):
    tables, series, plan = {}, {}, []
    for row_name, chosen_rows in evaluation_rows(stress).items():
        for scope, members in scopes_of(chosen_rows).items():
            by_arm = {}
            for r in chosen_rows:
                if r["regime"] in members:
                    by_arm.setdefault(r["arm"], []).append(r)
            table = {arm: summary(rows) for arm, rows in sorted(by_arm.items())}
            for arm in LEARNED:
                if arm in by_arm:
                    table[arm]["per_seed"] = {
                        seed: summary([r for r in by_arm[arm] if r["candidate"] == seed])
                        for seed in sorted({r["candidate"] for r in by_arm[arm]})}
            tables.setdefault(row_name, {})[scope] = table
            for a, b in PLAN:
                name = f"{row_name}/{scope}/{a}-minus-{b}"
                series[name] = difference(daily(by_arm[a], share), daily(by_arm[b], share))
                plan.append({"row": row_name, "scope": scope, "name": name, "arm": a,
                             "against": b,
                             "breaches": breaches(by_arm[a]) + breaches(by_arm[b]),
                             "against_traded": fills(by_arm[b]) > 0,
                             "arm_traded": fills(by_arm[a]) > 0})
    return tables, series, plan


def decide(item, interval):
    if any(interval.get(key) is None for key in ("mean", "lower", "upper")):
        raise ValueError(f"no interval to decide on: {interval}")
    fewer = Decimal(str(interval["upper"])) < 0 and item["breaches"] == 0
    return "fewer margin states" if fewer else "not shown"


def checked(run_out, stress_out):
    stress = json.loads((stress_out / "stress_holdout.json").read_text())
    if sha256(run_out / "frozen.json") != stress["frozen_sha256"]:
        raise ValueError("frozen.json is not the manifest the stress holdout was evaluated from")
    for key, name in (("stress_design_sha256", "stress_design.json"),
                      ("stress_digests_sha256", "stress_digests.json"),
                      ("stress_shocks_sha256", "stress_shocks.json")):
        if stress[key] != sha256(HERE / name):
            raise ValueError(f"{name} is not the one the stress holdout was evaluated with")
    return stress


def main():
    run_out, stress_out, binary = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    design = json.loads((HERE / "stress_design.json").read_text())
    stress = checked(run_out, stress_out)
    tables, series, plan = build(stress)
    payload = {"bootstrap": design["evaluation"]["bootstrap"], "series": series}
    (stress_out / "stress_series.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    intervals = json.loads(subprocess.run([binary, str(stress_out / "stress_series.json")],
                                          check=True, capture_output=True, text=True).stdout)
    decisions = []
    for item in plan:
        interval = intervals[item["name"]]
        note = None if item["against_traded"] and item["arm_traded"] else \
            "trading against not trading: an arm without a fill is Healthy by construction"
        decisions.append(item | {"interval": interval, "decision": decide(item, interval),
                                 "note": note})
    report = {"label": "the second, margin-stress holdout; it does not revise the first",
              "commit": stress["commit"], "frozen_sha256": stress["frozen_sha256"],
              "stress_holdout_sha256": sha256(stress_out / "stress_holdout.json"),
              "fee": stress["fee"], "days": stress["days"], "tables": tables,
              "intervals": intervals, "decisions": decisions,
              "note": "nothing here is a safety guarantee"}
    (stress_out / "stress_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"decisions": len(decisions),
                      "report_sha256": sha256(stress_out / "stress_report.json")}))


if __name__ == "__main__":
    main()
