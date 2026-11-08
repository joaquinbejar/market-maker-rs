"""How #110's report reads the holdout, and the wrapper that runs the holdout once. On
made-up rows, not the experiment's."""

import importlib.util
import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m4_10"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("m4_10_report", "report.py")
sys.path.insert(0, str(EXPERIMENT))
# The experiments' runners share the module name run: another study's, imported by an
# earlier test, must not stand in for this one's.
sys.modules.pop("run", None)
once = load("m4_10_holdout_once", "holdout_once.py")

LATENCIES = ("25", "500")


def episode(net, breach=False):
    return {"net": str(net), "limit_breach": breach, "durations": {"1": 3, "10": 1},
            "per_hour": dict.fromkeys(report.ORDER_KINDS, "2"),
            "markouts_ticks": {h: {"ticks": "1", "fills": 1} for h in report.MARKOUT_MS},
            "attribution": {"mean_abs_position": "1", "max_abs_position": "3",
                            "spread_capture": "0", "inventory_pnl": "0", "quote_fees": "0",
                            "max_drawdown": "1", "fills": 2},
            "time_beyond_5_lots": 0.0, "decide_microseconds": 5.0, "holds_per_decision": 0.5}


def holdout(value, breach=None, fees=("0.0002",), immediate=True):
    rows = []
    for fee in fees:
        for latency in LATENCIES:
            for regime in ("calm", "thin"):
                for day in (20, 21):
                    arms = [(a, None, s) for a in report.LEARNED for s in ("seed1", "seed2")]
                    arms += [(a, every, "point") for a in ("no_trade", "avellaneda_stoikov")
                             for every in (1, 10)]
                    for arm, every, seed in arms:
                        net = value(arm, every, fee, latency, regime, day, seed)
                        rows.append({"arm": arm, "every": every, "fee": fee, "latency": latency,
                                     "regime": regime, "day": day, "candidate": seed,
                                     "fill": "queue", "daily_value": str(net),
                                     "episodes": [episode(net, breach == (arm, seed))]})
    if immediate:
        for latency in LATENCIES:
            for regime in ("calm", "thin"):
                for day in (20, 21):
                    rows.append({"arm": "no_duration", "every": None, "fee": fees[0],
                                 "latency": latency, "regime": regime, "day": day,
                                 "candidate": "seed1", "fill": "immediate",
                                 "daily_value": "99", "episodes": [episode(99)]})
    return {"fees": list(fees), "rows": rows}


def flat(*key):
    return 0


def test_every_planned_comparison_is_made_per_fee_latency_and_scope():
    _, series, plan, _ = report.build(holdout(flat))
    kinds = {}
    for item in plan:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    # 2 latencies x 3 scopes: 1 duration, 2 ablations, 4 against no_trade.
    assert kinds == {"duration": 6, "ablation": 12, "no_trade": 24}
    assert "0.0002/500/pooled/full-minus-no_duration" in series
    assert "0.0002/25/thin/no_predictor-minus-no_trade/every1" in series
    assert {i["fee"] for i in plan} == {"0.0002"}
    # Each fee the holdout ran is read apart.
    _, _, plan, _ = report.build(holdout(flat, fees=("0", "0.0002")))
    assert {i["fee"] for i in plan} == {"0", "0.0002"} and len(plan) == 84


def test_a_learned_day_is_the_mean_of_its_seeds_and_classical_arms_keep_their_interval():
    def value(arm, every, fee, latency, regime, day, seed):
        if arm == "avellaneda_stoikov":
            return 5 if every == 10 else 1
        return {"seed1": 1, "seed2": 3}.get(seed, 0)

    tables, series, _, _ = report.build(holdout(value))
    calm = tables["0.0002"]["25"]["calm"]
    assert calm["full"]["mean_daily_net"] == "2" and set(calm["full"]["per_seed"]) == {
        "seed1", "seed2"}
    assert calm["avellaneda_stoikov/every10"]["mean_daily_net"] == "5"
    assert calm["avellaneda_stoikov/every1"]["mean_daily_net"] == "1"
    assert calm["full"]["durations_chosen"] == {"1": 12, "10": 4}
    assert series["0.0002/25/pooled/full-minus-no_trade/every1"] == ["2", "2", "2", "2"]


def test_the_immediate_fill_rows_are_descriptive_and_kept_out_of_the_decisions():
    tables, series, plan, descriptive = report.build(holdout(flat))
    assert set(descriptive) == set(LATENCIES)
    assert descriptive["25"]["pooled"]["mean_daily_net"] == "99"
    assert tables["0.0002"]["25"]["pooled"]["no_duration"]["mean_daily_net"] == "0"
    assert all("99" not in values for values in series.values())


def test_the_rules_read_helps_and_not_trading_as_the_design_says():
    helps = {"kind": "duration", "breaches": 0}
    assert report.decide(helps, {"mean": 1, "lower": 0.1, "upper": 2}) == "helps"
    assert report.decide(helps, {"mean": 1, "lower": 0, "upper": 2}) == "not shown"
    assert report.decide(helps | {"breaches": 1}, {"mean": 1, "lower": 0.1, "upper": 2}) == \
        "not shown"
    ablation = {"kind": "ablation", "breaches": 0}
    assert report.decide(ablation, {"mean": 1, "lower": 0.5, "upper": 2}) == "helps"
    trade = {"kind": "no_trade", "breaches": 0}
    assert report.decide(trade, {"mean": 0, "lower": 0, "upper": 1}) == \
        "equivalent to not trading"
    assert report.decide(trade, {"mean": 0, "lower": -1, "upper": 0}) == \
        "equivalent to not trading"
    assert report.decide(trade, {"mean": 2, "lower": 1, "upper": 3}) == "better than not trading"
    assert report.decide(trade, {"mean": -2, "lower": -3, "upper": -1}) == \
        "worse than not trading"
    with pytest.raises(ValueError, match="no interval"):
        report.decide(trade, {"mean": None, "lower": None, "upper": None})


def test_a_breach_in_either_arm_counts():
    _, _, plan, _ = report.build(holdout(flat, breach=("no_duration", "seed2")))
    item = next(i for i in plan if i["kind"] == "duration" and i["latency"] == "25"
                and i["scope"] == "calm")
    assert item["breaches"] == 2
    _, _, plan, _ = report.build(holdout(flat, breach=("full", "seed1")))
    item = next(i for i in plan if i["kind"] == "ablation" and i["against"] == "no_teacher"
                and i["latency"] == "500" and i["scope"] == "pooled")
    assert item["breaches"] == 4


def test_unpaired_days_and_another_manifest_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same days"):
        report.difference({("calm", 20): 1}, {("calm", 21): 1})
    (tmp_path / "frozen.json").write_text("{}")
    (tmp_path / "holdout.json").write_text(json.dumps({"frozen_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the manifest"):
        report.checked(tmp_path)


def test_the_holdout_runs_once_and_a_failed_or_concurrent_attempt_never_again(tmp_path):
    calls = []
    assert once.once(tmp_path, "f" * 64, "c0ffee", lambda: calls.append(1) or {"ok": 1}) == {
        "ok": 1}
    marker = tmp_path / once.MARKER
    assert json.loads(marker.read_text())["frozen_sha256"] == "f" * 64
    with pytest.raises(ValueError, match="started already"):
        once.once(tmp_path, "f" * 64, "c0ffee", lambda: calls.append(2))
    (tmp_path / "holdout.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluated already"):
        once.once(tmp_path, "f" * 64, "c0ffee", lambda: calls.append(3))
    assert calls == [1]
    # A failure keeps the marker, so the attempt is not run again.
    failing = tmp_path / "failing"
    failing.mkdir()

    def fail():
        raise RuntimeError("a worker failed after reading days")

    with pytest.raises(RuntimeError, match="worker failed"):
        once.once(failing, "f" * 64, "c0ffee", fail)
    assert (failing / once.MARKER).exists()
    with pytest.raises(ValueError, match="started already"):
        once.once(failing, "f" * 64, "c0ffee", lambda: calls.append(4))
    assert calls == [1]


def test_the_marker_is_created_exclusively(tmp_path, monkeypatch):
    real = os.open

    def racing(path, flags, *args):
        # Another attempt creates the marker between the check and the creation.
        if str(path).endswith(once.MARKER) and not os.path.exists(path):
            pathlib.Path(path).write_text("{}")
        return real(path, flags, *args)

    monkeypatch.setattr(os, "open", racing)
    with pytest.raises(FileExistsError):
        once.once(tmp_path, "f" * 64, "c0ffee", lambda: pytest.fail("ran"))
