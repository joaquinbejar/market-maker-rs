"""The validation pick of #103's learned arm at one row: its rule and the hashes it checks."""

import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_06_pick", HERE.parent / "experiments" / "m4_06" / "validation_pick.py"
)
pick = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pick)


def rows(values):
    return [{"arm": arm, "daily_value": str(v)} for arm, vs in values.items() for v in vs]


def test_the_pick_is_the_highest_mean_daily_net_and_a_tie_goes_to_the_greater_name():
    best, means = pick.pick(rows({"mlp": [1, 3], "lstm": [5, -2], "ema": [0, 1]}))
    assert best == "mlp" and means == {"ema": "0.5", "lstm": "1.5", "mlp": "2"}
    best, _ = pick.pick(rows({"history": [2], "ema": [2], "lstm": [1]}))
    assert best == "history"


def test_only_the_runs_at_one_row_are_evaluated():
    runs = [{"arm": "mlp", "every": e, "seed": 1} for e in (1, 5, 10)]
    assert pick.one_row_runs({"runs": runs}) == [runs[0]]


def frozen_dir(tmp_path):
    zipped = tmp_path / "mlp-every1-seed1.zip"
    zipped.write_bytes(b"weights")
    manifest = {"design_sha256": pick.run.sha256(pick.HERE / "design.json"),
                "digests_sha256": pick.run.sha256(pick.HERE / "digests.json"),
                "runs": [{"path": str(zipped.with_suffix("")), "every": 1,
                          "checkpoint_sha256": pick.run.sha256(zipped)}]}
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    return zipped, manifest


def test_the_manifest_the_design_and_every_checkpoint_must_match_their_hashes(tmp_path):
    zipped, manifest = frozen_dir(tmp_path)
    good = pick.run.sha256(tmp_path / "frozen.json")
    assert pick.checked_frozen(tmp_path, good)["runs"] == manifest["runs"]
    with pytest.raises(ValueError, match="not the one given"):
        pick.checked_frozen(tmp_path, "0" * 64)
    zipped.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="not the frozen checkpoint"):
        pick.checked_frozen(tmp_path, good)
    manifest["design_sha256"] = "0" * 64
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="design.json changed"):
        pick.checked_frozen(tmp_path, pick.run.sha256(tmp_path / "frozen.json"))
