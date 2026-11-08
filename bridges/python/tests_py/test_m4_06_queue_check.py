"""#103's check that the queue binds: one training day under the queue and under immediate fill.
On a made-up walk, not the experiment's days."""

import importlib.util
import json
import pathlib
import sys

import pytest
from conftest import walk

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_06_queue_check", HERE.parent / "experiments" / "m4_06" / "queue_check.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

POINT = {"risk_aversion": "0.01", "order_intensity": "50", "size": "1", "horizon_ms": 86400000,
         "volatility": "0.5"}


@pytest.fixture
def walk_day(monkeypatch):
    rows = walk(400, spacing_ms=1000)
    reads = []
    design, _ = check.run.load_design()
    probe = check.ResidualMarketMakingEnv(
        check.run.base_config(design, "0") | {"residual": {
            "baseline": check.run.baseline_config("avellaneda_stoikov", POINT, 1),
            "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}}, rows)
    record = {"dataset_digest": probe.bridge_info["dataset_digest"]}
    probe.close()

    def read_rows(data_dir, digests, regime, day, split, holdout_allowed=False):
        reads.append((regime, day, split))
        return rows, record

    monkeypatch.setattr(check.run, "read_rows", read_rows)
    return design, reads


def test_the_same_day_runs_under_each_fill_model_and_immediate_fills_at_least_as_often(walk_day):
    design, reads = walk_day
    queue = check.day_under(design["execution"]["fill_model"], design, {}, "x", "calm", 1, "0",
                            POINT)
    immediate = check.day_under(check.IMMEDIATE, design, {}, "x", "calm", 1, "0", POINT)
    assert reads == [("calm", 1, "train")] * 2
    assert queue["fill_model"] == design["execution"]["fill_model"]
    assert immediate["fill_model"] == {"kind": "immediate"}
    assert immediate["fills"] > queue["fills"] and queue["orders"]["accepted"] > 0
    assert immediate["orders"]["fills"] == immediate["fills"]


def test_only_a_training_day_is_read_and_with_its_regimes_volatility(walk_day, tmp_path,
                                                                      monkeypatch, capsys):
    design, reads = walk_day
    fee_rule = {"fee": "0", "best_per_fee": {"0": {"point": POINT | {"volatility": "9"}}},
                "volatility_from_training_days": {"calm": "0.5", "thin": "0.7"}}
    (tmp_path / "fee_rule.json").write_text(json.dumps(fee_rule))
    monkeypatch.setattr(sys, "argv", ["queue_check.py", "x", str(tmp_path), "thin", "2"])
    check.main()
    out = json.loads(capsys.readouterr().out)
    assert out["point"]["volatility"] == "0.7" and out["fee"] == "0"
    assert [r["fill_model"]["kind"] for r in out["rows"]] == [
        design["execution"]["fill_model"]["kind"], "immediate"]
    assert reads == [("thin", 2, "train")] * 2
    monkeypatch.setattr(sys, "argv", ["queue_check.py", "x", str(tmp_path), "calm", "15"])
    with pytest.raises(ValueError, match="not a training day"):
        check.main()
