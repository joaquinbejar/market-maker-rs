"""What #105's freeze, holdout and loading refuse: a checkpoint or a configuration that is not
the one training recorded. On made-up runs, not the experiment's."""

import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m4_08"
sys.path.insert(0, str(EXPERIMENT))
SPEC = importlib.util.spec_from_file_location("m4_08_run", EXPERIMENT / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class Ctx:
    """What the freeze and the holdout read of a context."""

    def __init__(self, out):
        self.out = pathlib.Path(out)

    def header(self, step):
        return {"step": step, "design_sha256": run.sha256(run.HERE / "design.json"),
                "shocks_sha256": run.sha256(run.HERE / "shocks.json"),
                "lock_sha256": run.sha256(run.HERE / "omnisafe-lock.txt")}

    write = run.Context.write
    read = run.Context.read


def trained(tmp_path):
    run_dir = tmp_path / "runs" / "cpo-seed1"
    (run_dir / "torch_save").mkdir(parents=True, exist_ok=True)
    (run_dir / "torch_save" / "epoch-1.pt").write_bytes(b"weights")
    (run_dir / "config.json").write_text(json.dumps({"algo_cfgs": {"obs_normalize": True}}))
    metadata = {"arm": "cpo", "penalty": None, "seed": 1, "run_dir": str(run_dir),
                "model": "epoch-1.pt",
                "checkpoint_sha256": run.sha256(run_dir / "torch_save" / "epoch-1.pt"),
                "config_sha256": run.sha256(run_dir / "config.json")}
    for name in ("fee_rule", "classical_selection"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"fee": "0.0002", "selected": {}}))
    (tmp_path / "penalty_selection.json").write_text(json.dumps({"selected": "0.1"}))
    (tmp_path / "training.json").write_text(json.dumps({"runs": [metadata]}))
    return run_dir


def test_the_freeze_refuses_a_configuration_or_checkpoint_changed_after_training(tmp_path):
    run_dir = trained(tmp_path)
    assert run.freeze(Ctx(tmp_path))["runs"] == 1
    (run_dir / "config.json").write_text(json.dumps({"algo_cfgs": {"obs_normalize": False}}))
    with pytest.raises(ValueError, match="config.json changed after training"):
        run.freeze(Ctx(tmp_path))
    trained(tmp_path)
    (run_dir / "torch_save" / "epoch-1.pt").write_bytes(b"retrained")
    with pytest.raises(ValueError, match="changed after training"):
        run.freeze(Ctx(tmp_path))


def test_the_holdout_refuses_a_configuration_changed_after_the_freeze(tmp_path):
    run_dir = trained(tmp_path)
    frozen = run.freeze(Ctx(tmp_path))["frozen_sha256"]
    (run_dir / "config.json").write_text(json.dumps({"algo_cfgs": {"obs_normalize": False}}))
    with pytest.raises(ValueError, match="config.json changed after the freeze"):
        run.holdout(Ctx(tmp_path), frozen)
    assert not (tmp_path / "holdout.json").exists()


def test_a_policy_loads_only_from_the_configuration_it_was_trained_with(tmp_path):
    run_dir = trained(tmp_path)
    recorded = run.sha256(run_dir / "config.json")
    (run_dir / "config.json").write_text(json.dumps({"model_cfgs": {"actor": {}}}))
    with pytest.raises(ValueError, match="config.json changed after training"):
        run.Learned(str(run_dir), "epoch-1.pt", recorded)


def test_every_learned_job_carries_the_recorded_configuration_hash():
    ctx = run.Context.__new__(run.Context)
    ctx.regimes, ctx.paths = ["calm"], {}
    m = {"arm": "cpo", "seed": 1, "penalty": None, "run_dir": "r", "model": "m",
         "config_sha256": "c" * 64}
    jobs = ctx.learned_jobs([m], "0.0002", "validation", [15])
    assert jobs and all(j["policy"]["config_sha256"] == "c" * 64 for j in jobs)
