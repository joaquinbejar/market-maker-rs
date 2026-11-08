"""Shocked copies of #102's validation and holdout days, for the stress rows of #105.

    python experiments/m4_08/shocks.py <#102 data dir> <out dir> > experiments/m4_08/shocks.json

For every regime, validation or holdout day and shock of `design.json`, the rows are the
day's own until `at_row`, and from there on the bid and the ask move by `round(mid * shock)`
kernel ticks, `mid` the mid of row `at_row` in ticks: an asset shock in the sense of
margin-rs's `Scenario::shock`, as a gap in the rows. Spread, sizes and times are unchanged.
The source rows are checked against #102's digests first. It prints, per copy, the dataset
digest the environment reports for its rows. The rows are only written here; no run reads a
shocked holdout day before the freeze.
"""

import json
import pathlib
import sys
from decimal import ROUND_HALF_EVEN, Decimal

from mm_rl_bridge import BoxMarketMakingEnv

HERE = pathlib.Path(__file__).resolve().parent


def digest_of(design, rows):
    """The dataset digest the bridge reports for `rows`; it depends on the rows only."""
    kernel = design["kernel"]
    config = {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": len(rows), "fee_rate": "0", "initial_capital": "0",
        "latency": {"quantum_ns": 1000000, "observation_q": 0, "submit_q": 0, "cancel_q": 0,
                    "ack_q": 0, "jitter_q": 0},
        "fill_model": {"kind": "immediate"}, "limits": None,
        "bid_offsets": [0], "ask_offsets": [0], "sizes": [1],
        "scales": design["observation_scales"], "f32_unit": "1", "invalid_action": "reject",
        "invalid_action_penalty": "0", "terminal": "mark",
    }
    env = BoxMarketMakingEnv(config, rows)
    try:
        return env.bridge_info["dataset_digest"]
    finally:
        env.close()


def shocked(rows, at_row, shock):
    bid, ask = rows[at_row][1], rows[at_row][3]
    delta = int((Decimal(bid + ask) / 2 * Decimal(shock)).quantize(Decimal(1), ROUND_HALF_EVEN))
    out = []
    for index, (ts, b, bl, a, al) in enumerate(rows):
        if index >= at_row:
            b, a = b + delta, a + delta
            if b <= 0 or a <= b:
                raise ValueError(f"a shock of {shock} leaves row {index} without a book")
        out.append((ts, b, bl, a, al))
    return out, delta


def main():
    source, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    design = json.loads((HERE / "design.json").read_text())
    base = json.loads((HERE.parent / "m4_05" / "digests.json").read_text())
    spec = design["shocks"]
    records = []
    for record in base["days"]:
        if record["split"] not in spec["applied_to"]:
            continue
        path = source / record["split"] / record["regime"] / f"day-{record['day']}.json"
        rows = [tuple(row) for row in json.loads(path.read_text())]
        if digest_of(design, rows) != record["dataset_digest"]:
            raise ValueError(f"{path} is not the day of #102's digests")
        for shock in spec["shocks"]:
            copy, delta = shocked(rows, spec["at_row"], shock)
            target = out / record["split"] / record["regime"] / f"day-{record['day']}-shock{shock}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([list(row) for row in copy]))
            records.append({"regime": record["regime"], "day": record["day"],
                            "split": record["split"], "shock": shock, "delta_ticks": delta,
                            "source_digest": record["dataset_digest"],
                            "dataset_digest": digest_of(design, copy), "rows": len(copy)})
    print(json.dumps({"design_version": design["design_version"], "shocks": records}, indent=2))


if __name__ == "__main__":
    main()
