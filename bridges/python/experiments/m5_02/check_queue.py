"""Before any training of parts B and C of #107: the queue binds and a classical arm fills.

    python experiments/m5_02/check_queue.py <part> <data dir> <out dir>

Avellaneda-Stoikov as a zero residual at every point of #103's grid (volatility by #103's rule
over the part's training days) on training day 1 of every group, at the secondary fee 0.0002,
under the design's fill model and under immediate fill. A group passes when some point fills
under the design's model and its net differs from immediate fill. It reads training days only.
SYNTHETIC.
"""

import importlib.util
import json
import pathlib
import sys
from decimal import Decimal

HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("m5_02_run_market", HERE / "run_market.py")
rm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rm)


def volatility(part, data_dir, design, base, digests, group):
    return rm.volatility(part, data_dir, design, base, digests, group)


def main():
    part, data_dir, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    design, base = rm.load()
    digests = rm.digests_of(part)
    fee = base["execution"]["secondary_fee_rate"]
    day = rm.days_of(design, base, part, "train")[0]
    report = {}
    for group in rm.groups_of(design, base, part):
        sigma = volatility(part, data_dir, design, base, digests, group)
        points = rm.m6.grid_points(base["arms"]["avellaneda_stoikov"]["grid"], sigma)
        rows = []
        for point in points:
            row = {"point": point}
            for name, fill in (("design", design["execution"]["fill_model"]),
                               ("immediate", {"kind": "immediate"})):
                job = {"part": part, "arm": "avellaneda_stoikov", "fee": fee,
                       "data_dir": data_dir, "point": point, "fill_model": fill}
                env = rm.build_env(job, digests, group, day, "train")
                try:
                    episode = rm.m6.episode(rm.m6.ZeroResidual(), env, 101, False, base)
                finally:
                    env.close()
                row[name] = {"net": episode["net"], "fills": episode["orders"]["fills"]}
            row["binds"] = row["design"]["net"] != row["immediate"]["net"]
            row["passes"] = row["binds"] and row["design"]["fills"] > 0
            rows.append(row)
        report[group] = {"points": rows, "passes": any(r["passes"] for r in rows)}
        print(group, report[group]["passes"],
              [(r["point"]["risk_aversion"], r["point"]["order_intensity"],
                r["design"]["fills"], r["immediate"]["fills"]) for r in rows], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"check_queue-{part}.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if not all(r["passes"] for r in report.values()):
        raise SystemExit("the queue does not bind, or the classical arm does not fill")


if __name__ == "__main__":
    main()
