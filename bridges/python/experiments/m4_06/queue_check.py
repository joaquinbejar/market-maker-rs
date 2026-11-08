"""Whether the queue binds on a training day: the fee rule's Avellaneda-Stoikov point, deciding
every row, under the design's queue and under immediate fill.

    python experiments/m4_06/queue_check.py <data dir> <out dir> [<regime> <day>]

reads `fee_rule.json` from `<out dir>` for the fee and the point, runs the point on one training
day (calm day 1 unless given) at a decision every row under each fill model, and prints the
fills, the orders and the exact net of each. It reads no validation or holdout day, and changes
no step's output. Added after the holdout was read, for the review; it is reported beside the
design's own check before training, which used validation day 15.

Everything here is synthetic.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from mm_rl_bridge import ResidualMarketMakingEnv  # noqa: E402
from mm_rl_bridge.training import ZeroResidual  # noqa: E402
from mm_rl_bridge.wrappers import DecisionInterval  # noqa: E402

IMMEDIATE = {"kind": "immediate"}


def day_under(fill_model, design, digests, data_dir, regime, day, fee, point):
    rows, record = run.read_rows(data_dir, digests, regime, day, "train")
    config = run.base_config(design, fee)
    config["fill_model"] = fill_model
    config["residual"] = {"baseline": run.baseline_config("avellaneda_stoikov", point, day),
                          "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
    env = ResidualMarketMakingEnv(config, rows)
    try:
        if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
            raise ValueError(f"{regime} day {day} has other rows than digests.json")
        result = run.episode(ZeroResidual(), DecisionInterval(env, 1),
                             design["evaluation"]["evaluation_seeds"][0], False, design)
    finally:
        env.close()
    return {"fill_model": fill_model, "fills": result["attribution"]["fills"],
            "orders": result["orders"], "net": result["net"]}


def main():
    data_dir, out = sys.argv[1], pathlib.Path(sys.argv[2])
    regime, day = (sys.argv[3], int(sys.argv[4])) if len(sys.argv) > 4 else ("calm", 1)
    design, digests = run.load_design()
    if day not in run.days_of(design, "train"):
        raise ValueError(f"day {day} is not a training day")
    fee_rule = json.loads((out / "fee_rule.json").read_text())
    fee = fee_rule["fee"]
    # The fee rule chose a point per regime: its volatility is that regime's.
    point = dict(fee_rule["best_per_fee"][fee]["point"])
    point["volatility"] = fee_rule["volatility_from_training_days"][regime]
    rows = [day_under(model, design, digests, data_dir, regime, day, fee, point)
            for model in (design["execution"]["fill_model"], IMMEDIATE)]
    print(json.dumps({"regime": regime, "day": day, "fee": fee, "point": point,
                      "fee_rule_sha256": run.sha256(out / "fee_rule.json"), "every": 1,
                      "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
