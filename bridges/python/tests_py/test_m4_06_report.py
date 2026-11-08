"""How the report of #103 reads the holdout: days, comparisons, the best classical arm and the
decisions. On made-up rows, not the experiment's."""

import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_06_report", HERE.parent / "experiments" / "m4_06" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

ARMS = ("no_trade", "avellaneda_stoikov", "glft", "ema", "history", "lstm", "mlp")


def episode(net, breach=False):
    return {"attribution": {"spread_capture": "0", "inventory_pnl": "0", "quote_fees": "0",
                            "fills": 1, "max_drawdown": "1", "mean_abs_position": "0",
                            "max_abs_position": "0"},
            "decisions": 10, "quote_transactions": 5, "refusals": 1, "limit_breach": breach,
            "peak_notional": "0", "time_beyond_5_lots": 0, "decide_microseconds": 2.0,
            "orders": {"accepted": 1, "canceled": 1, "cancel_rejected": 0, "rejected": 0},
            "net": str(net)}


def holdout(value, fees=("-0.0001", "0.0002"), breach=None):
    rows = []
    for fee in fees:
        for regime in ("calm", "thin"):
            for day in (20, 21):
                for every in (1, 5):
                    for arm in ARMS:
                        seeds = ("seed1", "seed2") if arm in report.LEARNED else ("point",)
                        for seed in seeds:
                            net = value(arm, every, regime, day, seed)
                            rows.append({"arm": arm, "every": every, "fee": fee,
                                         "regime": regime, "day": day, "candidate": seed,
                                         "daily_value": str(net),
                                         "episodes": [episode(net, breach == (arm, seed))]})
    return {"fees": list(fees), "rows": rows}


def flat(arm, every, regime, day, seed):
    return 0


def test_a_learned_day_is_the_mean_of_its_seeds_and_pooled_holds_every_regime():
    def value(arm, every, regime, day, seed):
        return {"seed1": 1, "seed2": 3}.get(seed, 0)

    tables, series, _ = report.build(holdout(value))
    assert tables["-0.0001"]["calm"]["mlp/every1"]["mean_daily_net"] == "2"
    assert tables["-0.0001"]["calm"]["mlp/every1"]["days"] == 2
    assert tables["-0.0001"]["pooled"]["mlp/every1"]["days"] == 4
    assert set(tables["-0.0001"]["calm"]["mlp/every1"]["per_seed"]) == {"seed1", "seed2"}
    assert series["-0.0001/pooled/mlp/every1"] == ["2", "2", "2", "2"]


def test_every_comparison_of_the_rule_is_made_at_every_fee_and_scope():
    _, series, plan = report.build(holdout(flat))
    kinds = {}
    for item in plan:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    # 2 fees x 3 scopes; per scope: memory 3 x 2 intervals, no_trade and classical 4 x 2,
    # frequency 6 quoting arms x 1 longer interval.
    assert kinds == {"memory": 36, "no_trade": 48, "classical": 48, "frequency": 36}
    assert "-0.0001/calm/ema/every5-minus-mlp/every5" in series
    assert "0.0002/pooled/glft/every5-minus-glft/every1" in series


def test_the_classical_comparison_is_against_the_better_classical_arm_in_its_scope():
    def value(arm, every, regime, day, seed):
        return {"glft": 2 if regime == "calm" else -1, "avellaneda_stoikov": 1}.get(arm, 0)

    _, _, plan = report.build(holdout(value))
    against = {(i["fee"], i["scope"]): i["against"][0] for i in plan if i["kind"] == "classical"}
    assert against[("-0.0001", "calm")] == "glft"
    assert against[("-0.0001", "thin")] == "avellaneda_stoikov"
    assert against[("-0.0001", "pooled")] == "avellaneda_stoikov"


def test_a_breach_in_either_arm_stops_memory_from_helping():
    _, _, plan = report.build(holdout(flat, breach=("lstm", "seed2")))
    item = next(i for i in plan if i["kind"] == "memory" and i["arm"] == ("lstm", 1)
                and i["scope"] == "calm" and i["fee"] == "-0.0001")
    assert item["breaches"] == 2
    above = {"mean": 1, "lower": 0.5, "upper": 2}
    assert report.decide(item, above) == "not shown"
    assert report.decide(item | {"breaches": 0}, above) == "helps"
    assert report.decide(item | {"breaches": 0}, {"mean": 1, "lower": 0, "upper": 2}) == "not shown"
    # A breach of the arm compared against counts too.
    _, _, plan = report.build(holdout(flat, breach=("mlp", "seed1")))
    item = next(i for i in plan if i["kind"] == "memory" and i["arm"] == ("lstm", 1)
                and i["scope"] == "calm" and i["fee"] == "-0.0001")
    assert item["breaches"] == 2


def test_against_no_trade_an_interval_with_zero_is_equivalent_to_not_trading():
    item = {"kind": "no_trade", "breaches": 0}
    assert report.decide(item, {"mean": 0, "lower": -1, "upper": 1}) == "equivalent to not trading"
    assert report.decide(item, {"mean": 1, "lower": 0, "upper": 2}) == "equivalent to not trading"
    assert report.decide(item, {"mean": 2, "lower": 1, "upper": 3}) == "better than not trading"
    assert report.decide(item, {"mean": -2, "lower": -3, "upper": -1}) == "worse than not trading"
    with pytest.raises(ValueError, match="no interval"):
        report.decide(item, {"mean": None, "lower": None, "upper": None, "days": 3})


def test_unpaired_days_and_another_manifest_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same days"):
        report.difference({("calm", 20): 1}, {("calm", 21): 1})
    (tmp_path / "frozen.json").write_text("{}")
    (tmp_path / "holdout.json").write_text(json.dumps({"frozen_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the manifest"):
        report.checked(tmp_path)
