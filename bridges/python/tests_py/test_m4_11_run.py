"""The rules of #280's runner that read no environment: executions, the queue gate, the choice of
alpha, the reward scale, the support bins and the checks on logs and reports."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_11_run", HERE.parent / "experiments" / "m4_11" / "run.py"
)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

DESIGN = json.loads((HERE.parent / "experiments" / "m4_11" / "design.json").read_text())


def main_config():
    return {"fill_model": {"kind": "queue_position", "rate": "1", "min_queue_ms": 5000},
            "latency": {"quantum_ns": 1000000, "submit_q": 25}, "fee_rate": "0.0002"}


def test_executions_change_the_fill_model_only():
    config = main_config()
    standard = run.config_of(config, "standard", DESIGN)
    assert standard["fill_model"] == {"kind": "queue_position_rational", "lots": 1,
                                      "per_ms": 5000, "min_queue_ms": 0}
    assert standard["latency"] == config["latency"] and standard["fee_rate"] == "0.0002"
    assert run.config_of(config, "immediate_reference", DESIGN)["fill_model"] == {"kind": "immediate"}
    assert run.config_of(config, "m4_05_main", DESIGN) == config
    # The study's configuration is not changed in place.
    assert config["fill_model"]["kind"] == "queue_position"


def gate_rows(as_fills, as_standard, as_immediate, fixed=("3", "-1", "-2")):
    rows = []
    for arm, (fills, standard, immediate) in (("as_grid", (as_fills, as_standard, as_immediate)),
                                              ("fixed_spread_4", fixed)):
        for day in (15, 16):
            rows.append({"arm": arm, "execution": "standard", "fills": fills, "net": standard,
                         "day": day})
            rows.append({"arm": arm, "execution": "immediate_reference", "fills": "9",
                         "net": immediate, "day": day})
    return rows


def test_the_gate_needs_fills_and_a_net_unlike_immediate_fill():
    passing = run.gate_summary(gate_rows("2", "-5.5", "-7"))
    assert passing["as_grid"]["passes"] and passing["as_grid"]["standard_fills"] == 4
    assert passing["as_grid"]["standard_mean_daily_net"] == "-5.5"
    assert not run.gate_summary(gate_rows("0", "-5.5", "-7"))["as_grid"]["passes"]
    assert not run.gate_summary(gate_rows("2", "-7", "-7.0"))["as_grid"]["passes"]
    # fixed_spread_4 is reported beside it, by the same rule.
    assert not run.gate_summary(gate_rows("2", "-5.5", "-7", ("1", "-2", "-2")))[
        "fixed_spread_4"]["passes"]


def test_the_gate_refuses_a_rule_missing_an_execution():
    rows = [r for r in gate_rows("2", "-5", "-7") if not (
        r["arm"] == "as_grid" and r["execution"] == "immediate_reference")]
    with pytest.raises(ValueError, match="both executions"):
        run.gate_summary(rows)


def test_alpha_is_the_best_trained_one_and_the_smaller_on_a_tie():
    choices = [0.1, 1.0, 10.0]
    assert run.choose_alpha({"cql-alpha0.1": 1, "cql-alpha1.0": 3, "cql-alpha10.0": 2}, choices) == 1.0
    assert run.choose_alpha({"cql-alpha0.1": 3, "cql-alpha1.0": 3, "cql-alpha10.0": 3}, choices) == 0.1
    # A stopped alpha has no mean, and is skipped.
    assert run.choose_alpha({"cql-alpha10.0": -1}, choices) == 10.0
    assert run.choose_alpha({}, choices) is None


def test_the_reward_scale_is_the_training_rewards_spread():
    days = [{"reward": np.array([1.0, -1.0])}, {"reward": np.array([1.0, -1.0])}]
    assert run.reward_scale(days) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no spread"):
        run.reward_scale([{"reward": np.zeros(3)}])


def test_state_bins_cross_104s_bin_with_the_position():
    obs = np.zeros((3, 22), dtype=np.float32)
    obs[:, 0] = 0.1  # one tick of spread at a scale of 10
    obs[:, 11] = 1.0
    bins = run.state_bins(obs, np.array([-7.0, 0.0, 7.0]), [0.0, 0.1, 0.2, 0.3])
    base = run.est.bin_states(obs, [0.0, 0.1, 0.2, 0.3])
    assert bins.tolist() == (base * 5 + np.array([0, 2, 4])).tolist()
    assert run.state_bins(np.zeros((0, 22)), np.zeros(0), [0.0, 0.1, 0.2, 0.3]).size == 0


def test_logs_are_read_only_when_their_hash_matches(tmp_path):
    path = tmp_path / "day.npz"
    np.savez_compressed(path, action=np.zeros(2, dtype=np.int16), reward=np.ones(2))
    days = run.read_logs([{"path": str(path), "sha256": run.sha256(path)}])
    assert days[0]["reward"].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="not the log"):
        run.read_logs([{"path": str(path), "sha256": "0" * 64}])


class Out:
    def __init__(self, path):
        self.out = path


def test_a_step_waits_for_a_report_that_exists_and_did_not_stop(tmp_path):
    study = Out(tmp_path)
    with pytest.raises(PermissionError, match="run `gate` first"):
        run.require(study, "gate", "waits")
    (tmp_path / "gate.json").write_text(json.dumps({"stop": True}))
    with pytest.raises(PermissionError, match="stopped the study"):
        run.require(study, "gate", "waits")
    (tmp_path / "gate.json").write_text(json.dumps({"stop": False, "x": 1}))
    assert run.require(study, "gate", "waits")["x"] == 1


def test_the_behaviours_epsilons_are_104s():
    assert run.m4_07_epsilon("as_grid_eps30") == 0.3
    assert run.m4_07_epsilon("as_grid_eps10") == 0.1


def test_the_design_fixes_the_reused_files():
    # load() checks #102's design and digests and #104's design against the hashes pinned here.
    design, _, _ = run.load()
    assert design["issue"] == 280
    assert design["arms"]["cql"]["alpha_choices"] == [0.1, 1.0, 10.0]


def test_mean_by_averages_each_arms_daily_nets():
    rows = [{"arm": "a", "net": "1"}, {"arm": "a", "net": "2"}, {"arm": "b", "net": "-1"}]
    means = run.mean_by(rows, "arm")
    assert str(means["a"]) == "1.5" and str(means["b"]) == "-1"


class Frozen:
    """What the holdout reads of a study: its output directory and commit."""

    def __init__(self, out):
        self.out, self.commit, self.parallel = pathlib.Path(out), "c0ffee", 1


def frozen_study(tmp_path):
    checkpoint = tmp_path / "bc-seed1.pt"
    checkpoint.write_bytes(b"weights")
    (tmp_path / "gate.json").write_text("{}")
    manifest = {"design_sha256": run.sha256(run.HERE / "design.json"),
                "code_sha256": {name: run.sha256(run.HERE / name) for name in run.CODE},
                "checkpoints": {str(checkpoint): run.sha256(checkpoint)}, "logs": {},
                "reports": {"gate.json": run.sha256(tmp_path / "gate.json")}, "tags": {}}
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    return Frozen(tmp_path), run.sha256(tmp_path / "frozen.json"), checkpoint


def test_the_holdout_runs_once_against_its_manifest_and_unchanged_design_and_code(
        tmp_path, monkeypatch):
    study, good, checkpoint = frozen_study(tmp_path)
    jobs = [{"comparison": "primary", "dataset": None}]
    monkeypatch.setattr(run, "holdout_jobs", lambda study, frozen: list(jobs))
    monkeypatch.setattr(run, "pool", lambda jobs, worker, parallel, label: [{} for _ in jobs])
    monkeypatch.setattr(run, "support_report", lambda study, rows: {})
    study.header = lambda step: {"step": step}
    study.write = lambda name, report: (study.out / f"{name}.json").write_text(json.dumps(report))
    with pytest.raises(PermissionError, match="frozen manifest named"):
        run.holdout(study, "0" * 64)
    manifest = json.loads((tmp_path / "frozen.json").read_text())
    for key, changed in (("design_sha256", "0" * 64),
                         ("code_sha256", {**manifest["code_sha256"], "run.py": "0" * 64})):
        (tmp_path / "frozen.json").write_text(json.dumps(manifest | {key: changed}))
        with pytest.raises(ValueError, match="changed after the freeze"):
            run.holdout(study, run.sha256(tmp_path / "frozen.json"))
        assert not (tmp_path / run.HOLDOUT_MARKER).exists()
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    report = run.holdout(study, good)
    assert report["frozen_sha256"] == good and (tmp_path / run.HOLDOUT_MARKER).exists()
    with pytest.raises(PermissionError, match="evaluated already"):
        run.holdout(study, good)
    (tmp_path / "holdout.json").unlink()
    # The marker stays: with the result gone, the holdout is still not read again.
    with pytest.raises(PermissionError, match="started already"):
        run.holdout(study, good)


def test_a_failed_or_concurrent_holdout_is_never_run_again(tmp_path, monkeypatch):
    study, good, _ = frozen_study(tmp_path)
    calls = []
    monkeypatch.setattr(run, "holdout_jobs", lambda study, frozen: [{"comparison": "primary"}])

    def failing(jobs, worker, parallel, label):
        calls.append(len(jobs))
        raise RuntimeError("a worker failed after reading days")

    monkeypatch.setattr(run, "pool", failing)
    with pytest.raises(RuntimeError, match="worker failed"):
        run.holdout(study, good)
    assert (tmp_path / run.HOLDOUT_MARKER).exists() and not (tmp_path / "holdout.json").exists()
    with pytest.raises(PermissionError, match="started already"):
        run.holdout(study, good)
    assert calls == [1]
    # The marker is created exclusively: an attempt that loses the race fails on creation.
    (tmp_path / run.HOLDOUT_MARKER).unlink()
    real = run.os.open

    def racing(path, flags, *args):
        if str(path).endswith(run.HOLDOUT_MARKER) and not run.os.path.exists(path):
            pathlib.Path(path).write_text("{}")
        return real(path, flags, *args)

    monkeypatch.setattr(run.os, "open", racing)
    with pytest.raises(FileExistsError):
        run.holdout(study, good)
    assert calls == [1]
