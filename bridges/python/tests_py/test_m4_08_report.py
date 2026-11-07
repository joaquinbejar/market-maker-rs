"""How the report of #105 reads the holdout: its rows, days, costs, comparisons and the
decision rule. On made-up rows, not the experiment's."""

import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_08_report", HERE.parent / "experiments" / "m4_08" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

ARMS = ("no_trade", "avellaneda_stoikov", "ppo", "ppo_penalized", "cpo")
BUDGETS = {"c_inventory": "0.05", "c_notional": "0.2"}


def episode(net, cost=(0, 0, 0), breach=False, refused=None):
    inventory, notional, margin = cost
    return {"cost_means": {"c_inventory": inventory, "c_notional": notional, "c_margin": margin},
            "over_budget": {"c_inventory": inventory > 0.05, "c_notional": notional > 0.2},
            "actions_refused": refused or {}, "legs_rejected": 1, "limit_breach": breach,
            "peak_notional": "10", "net": str(net),
            "orders": {"accepted": 2, "canceled": 1, "cancel_rejected": 0, "rejected": 1,
                       "fills": 1},
            "attribution": {k: "0" for k in report.ATTRIBUTION} | {"fills": 1}}


def holdout(value, cost=lambda *a: (0, 0, 0), fees=("-0.0001", "0.0002"), breach=None,
            shocks=(None, "-0.05", "0.05")):
    rows = []
    for fee in fees:
        for shock in (shocks if fee == fees[0] else (None,)):
            for regime in ("calm", "thin"):
                for day in (20, 21):
                    for arm in ARMS:
                        seeds = ("seed1", "seed2") if arm in report.LEARNED else ("point",)
                        for seed in seeds:
                            key = (arm, fee, shock, regime, day, seed)
                            rows.append({"arm": arm, "fee": fee, "shock": shock,
                                         "regime": regime, "day": day, "candidate": seed,
                                         "daily_value": str(value(*key)),
                                         "episodes": [episode(value(*key), cost(*key),
                                                              breach == (arm, seed))]})
    return {"rows": rows}


def flat(*key):
    return 0


def build(h, fee="-0.0001"):
    return report.build(h, fee, "0.0002", BUDGETS)


def test_the_rows_are_the_unshocked_days_each_shock_and_the_secondary_fee():
    tables, _, _ = build(holdout(flat))
    assert set(tables) == {"unshocked", "shock -0.05", "shock 0.05", "secondary fee"}
    assert set(tables["unshocked"]) == {"calm", "thin", "pooled"}
    assert tables["unshocked"]["pooled"]["cpo"]["days"] == 4
    assert tables["unshocked"]["calm"]["cpo"]["seed_days"] == 4
    # When the rule chose the secondary fee, there is no secondary row.
    tables, _, _ = build(holdout(flat, fees=("0.0002",)), fee="0.0002")
    assert "secondary fee" not in tables


def test_a_learned_day_is_the_mean_of_its_seeds_for_net_and_for_cost():
    def value(arm, fee, shock, regime, day, seed):
        return {"seed1": 1, "seed2": 3}.get(seed, 0)

    def cost(arm, fee, shock, regime, day, seed):
        if arm == "no_trade":
            return (0, 0, 0)
        return (0.1, 0.3, 0.9) if seed == "seed1" else (0.0, 0.1, 0.5)

    tables, series, _ = build(holdout(value, cost))
    cpo = tables["unshocked"]["calm"]["cpo"]
    assert cpo["mean_daily_net"] == "2"
    # (0.4 + 0.1) / 2: c_margin is not in the constrained cost.
    assert cpo["mean_daily_constrained_cost"] == "0.25"
    assert cpo["costs"]["c_margin"]["mean_per_step"] == "0.7"
    assert cpo["costs"]["c_inventory"]["episodes_over_budget"] == 2
    assert cpo["costs"]["c_margin"]["episodes_over_budget"] is None
    assert set(cpo["per_seed"]) == {"seed1", "seed2"}
    assert series["unshocked/pooled/cpo-minus-no_trade/net"] == ["2", "2", "2", "2"]
    assert series["unshocked/pooled/cpo-minus-no_trade/cost"] == ["0.25"] * 4


def test_every_comparison_of_the_rule_is_made_in_every_row_and_scope():
    _, series, plan = build(holdout(flat))
    kinds = {}
    for item in plan:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    # 4 rows x 3 scopes; per scope 1 constrained, 3 against no_trade, 2 against ppo.
    assert kinds == {"constrained": 12, "against no_trade": 36, "against ppo": 24}
    assert "shock 0.05/thin/cpo-minus-ppo_penalized/cost" in series
    assert not any("ppo-minus-ppo/" in name for name in series)


def test_the_rule_needs_net_not_below_cost_below_and_no_breach():
    item = {"kind": "constrained", "breaches": 0}
    below = {"mean": -1, "lower": -2, "upper": -0.5}
    assert report.decide(item, {"mean": 0, "lower": -1, "upper": 0}, below) == "constrained better"
    assert report.decide(item, {"mean": 2, "lower": 1, "upper": 3}, below) == "constrained better"
    assert report.decide(item, {"mean": -2, "lower": -3, "upper": -0.1}, below) == "not shown"
    assert report.decide(item, {"mean": 0, "lower": -1, "upper": 1},
                         {"mean": 0, "lower": -1, "upper": 0}) == "not shown"
    assert report.decide(item | {"breaches": 1}, {"mean": 0, "lower": -1, "upper": 1},
                         below) == "not shown"
    other = {"kind": "against ppo", "breaches": 0}
    assert report.decide(other, {"mean": 0, "lower": -1, "upper": 1}, below) == \
        "better by the rule"
    with pytest.raises(ValueError, match="no interval"):
        report.decide(item, {"mean": None, "lower": None, "upper": None}, below)


def test_a_breach_in_either_arm_counts():
    _, _, plan = build(holdout(flat, breach=("ppo_penalized", "seed2")))
    item = next(i for i in plan if i["kind"] == "constrained" and i["row"] == "unshocked"
                and i["scope"] == "calm")
    assert item["breaches"] == 2
    item = next(i for i in plan if i["kind"] == "against no_trade" and i["arm"] == "cpo"
                and i["row"] == "unshocked" and i["scope"] == "calm")
    assert item["breaches"] == 0


def test_refusals_are_summed_by_reason_and_the_shocked_rows_hold_only_their_shock():
    def value(arm, fee, shock, regime, day, seed):
        return 5 if shock == "0.05" else 0

    h = holdout(value)
    for row in h["rows"]:
        row["episodes"][0]["actions_refused"] = {"limits": 2} if row["arm"] == "cpo" else {}
    tables, _, _ = build(h)
    assert tables["shock 0.05"]["pooled"]["ppo"]["mean_daily_net"] == "5"
    assert tables["unshocked"]["pooled"]["ppo"]["mean_daily_net"] == "0"
    assert tables["unshocked"]["calm"]["cpo"]["actions_refused"] == {"limits": 8}


def test_unpaired_days_and_another_manifest_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same days"):
        report.difference({("calm", 20): 1}, {("calm", 21): 1})
    (tmp_path / "frozen.json").write_text("{}")
    (tmp_path / "holdout.json").write_text(json.dumps({"frozen_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the manifest"):
        report.checked(tmp_path)


def test_the_secondary_fee_row_holds_only_unshocked_days():
    h = holdout(flat)
    stray = dict(next(r for r in h["rows"] if r["fee"] == "0.0002" and r["arm"] == "cpo"),
                 shock="0.05", daily_value="9")
    h["rows"].append(stray)
    rows = report.evaluation_rows(h, "-0.0001", "0.0002")
    assert stray not in rows["secondary fee"] and stray not in rows["shock 0.05"]
