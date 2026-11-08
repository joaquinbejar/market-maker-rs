"""#105's second, margin-stress holdout: its days, its margin reading per step, what the
holdout runs and refuses, and how the report reads it. On made-up rows and values."""

import importlib.util
import json
import pathlib
import sys
from decimal import Decimal

import numpy as np
import pytest
from conftest import walk

from mm_rl_bridge import BoxMarketMakingEnv, maintenance_path

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m4_08"
sys.path.insert(0, str(EXPERIMENT))
# The experiments' runners share module names (run, report): another study's, imported by an
# earlier test, must not stand in for this one's.
for name in ("run", "report", "shocks", "costs"):
    sys.modules.pop(name, None)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stress = load("m4_08_stress", "stress.py")
reporting = load("m4_08_report_stress", "report_stress.py")
run = stress.run


def day(regime, n, digest="d"):
    return {"regime": regime, "day": n, "seed": n, "rows": 1, "dataset_digest": digest,
            "external_digest": "e", "parameters_fingerprint": "p", "split": "x"}


def test_the_new_days_are_after_102s_which_must_be_the_same():
    base = {"days": [day("calm", 1), day("calm", 2)]}
    generated = {"days": [day("calm", 1), day("calm", 2), day("calm", 3)]}
    assert [d["day"] for d in stress.new_days(generated, base)] == [3]
    changed = {"days": [day("calm", 1), day("calm", 2, "other"), day("calm", 3)]}
    with pytest.raises(ValueError, match="not #102's"):
        stress.new_days(changed, base)
    with pytest.raises(ValueError, match="do not cover"):
        stress.new_days({"days": [day("calm", 1), day("calm", 3)]}, base)


def test_the_committed_days_are_25_to_29_of_every_regime():
    days = json.loads(stress.STRESS_DIGESTS.read_text())["days"]
    assert sorted({d["day"] for d in days}) == [25, 26, 27, 28, 29]
    assert {d["regime"] for d in days} == {"calm", "stressed", "thin"} and len(days) == 15
    design, digests = stress.load_stress()
    assert design["days"] == [25, 29]
    assert len(digests) == 15 * 3 and all(k[0] == "holdout" for k in digests)


def test_a_days_margin_summary_counts_states_entries_and_the_worst_shortfall():
    path = [("healthy", "1", "0"), ("reduce_only", "1", "0.5"), ("default", "1", "2"),
            ("healthy", "1", "0"), ("reduce_only", "1", "0.25"), ("reduce_only", "1", "1")]
    out = stress.summary(path, [Decimal("1"), Decimal("-7"), Decimal("3")])
    assert out["steps"] == {"healthy": 2, "reduce_only": 3, "default": 1}
    assert out["not_healthy_share"] == pytest.approx(4 / 6)
    # Only a step out of Healthy is an entry: Healthy to ReduceOnly twice.
    assert out["entries"] == 2 and out["entries_default"] == 1
    # Healthy straight to Default is one entry and one Default entry; a stay is neither.
    direct = stress.summary([("default", "1", "3"), ("default", "1", "3"),
                             ("healthy", "1", "0"), ("default", "1", "1")], [])
    assert (direct["entries"], direct["entries_default"]) == (2, 2)
    assert out["largest_shortfall"] == "2" and out["max_abs_position"] == "7"


def test_the_margin_inputs_are_the_signed_position_the_observed_mid_and_the_net():
    rows = [(0, 199, 5, 201, 5), (1, 197, 5, 199, 5)]
    info = {"raw_observation": [None] * 6 + ["-3"], "row": 1, "net": "-2.5"}
    assert stress.margin_inputs(info, rows, Decimal("0.5")) == (
        Decimal("-3"), Decimal("99"), Decimal("-2.5"))
    # The terminal close of the last step is not an economic result of the day.
    last = info | {"terminal_close": "-0.5"}
    assert stress.margin_inputs(last, rows, Decimal("0.5"))[2] == Decimal("-2")
    flat = {"raw_observation": [None] * 7, "row": None, "net": "0", "terminal_close": None}
    assert stress.margin_inputs(flat, rows, Decimal("0.5"))[:2] == (Decimal(0), Decimal(100))


def test_the_helper_reads_maintenance_as_margin_rs_does():
    path = maintenance_path("125", 2, [("20", "100", "-25"), ("-20", "100", "-25.01"),
                                       ("1", "100", "-126"), ("0", "100", "0")])
    assert [s for s, _, _ in path] == ["healthy", "reduce_only", "default", "healthy"]
    assert path[1][1:] == ("100.00", "0.01")
    with pytest.raises(Exception, match="position"):
        maintenance_path("125", 2, [("x", "100", "0")])
    # An equity beyond a decimal is the bridge's error, not a panic.
    from mm_rl_bridge import BridgeFailure
    with pytest.raises(BridgeFailure, match="overflows"):
        maintenance_path("79228162514264337593543950335", 6, [("0", "1", "1")])


class Quoting:
    """Quotes on the first decision, then holds."""

    def __init__(self):
        self.calls = 0

    def predict(self, observation):
        self.calls += 1
        first = self.calls == 1
        return np.asarray([1.0 if first else -1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)


class Always:
    """A bid at the touch and a far ask on every decision."""

    def predict(self, observation):
        return np.asarray([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)


def test_an_evaluated_day_reads_every_step_and_leaves_the_environment_as_it_was():
    design, _ = run.load_design()
    stress_design, _ = stress.load_stress()
    rows = walk(60, spacing_ms=10_000)
    env = BoxMarketMakingEnv(run.base_config(design, "0"), rows)
    day = stress.episode(Quoting(), env, rows, 1, design, stress_design)
    assert "step" not in vars(env)
    margin = day["margin"]
    assert sum(margin["steps"].values()) == day["steps"]
    assert margin["steps"]["healthy"] == day["steps"] and margin["entries"] == 0
    # On a capital of 0.5, any position the quote fills leaves Healthy: the reading follows
    # the delivered position, and its largest is the attribution's.
    filling = BoxMarketMakingEnv(run.base_config(design, "0") | {
        "fill_model": {"kind": "immediate"}}, rows)
    thin = stress.episode(Always(), filling, rows, 1, design,
                          {"account": {"capital": "0.5", "scale": 6}})
    filling.close()
    assert Decimal(thin["attribution"]["max_abs_position"]) > 0
    assert thin["margin"]["max_abs_position"] == thin["attribution"]["max_abs_position"]
    assert thin["margin"]["steps"]["healthy"] < thin["steps"] and thin["margin"]["entries"] >= 1
    # A second day on the same environment records each step once.
    again = stress.episode(Quoting(), env, rows, 1, design, stress_design)
    assert sum(again["margin"]["steps"].values()) == again["steps"]
    env.close()


class Ctx:
    def __init__(self):
        self.commit, self.parallel, self.regimes = "c0ffee", 1, ["calm"]
        self.paths = {"data": "d", "shocks": "s"}
        self.sigmas = {"calm": "0.5"}
        self.design, _ = run.load_design()

    classical_jobs = run.Context.classical_jobs
    learned_jobs = run.Context.learned_jobs
    job = run.Context.job


def frozen(tmp_path):
    run_dir = tmp_path / "runs" / "cpo-seed1"
    (run_dir / "torch_save").mkdir(parents=True, exist_ok=True)
    (run_dir / "torch_save" / "m.pt").write_bytes(b"w")
    (run_dir / "config.json").write_text("{}")
    point = {"risk_aversion": "0.01", "order_intensity": "50", "size": "1",
             "horizon_ms": 86400000, "volatility": "0.5"}
    manifest = {"design_sha256": run.sha256(run.HERE / "design.json"),
                "shocks_sha256": run.sha256(run.HERE / "shocks.json"),
                "lock_sha256": run.sha256(run.HERE / "omnisafe-lock.txt"),
                "fee": "0.0002", "classical_selected": {"calm": point},
                "runs": [{"arm": "cpo", "seed": 1, "penalty": None, "run_dir": str(run_dir),
                          "model": "m.pt", "checkpoint_sha256": run.sha256(run_dir / "torch_save" / "m.pt"),
                          "config_sha256": run.sha256(run_dir / "config.json")}]}
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    return run.sha256(tmp_path / "frozen.json"), run_dir


def test_the_stress_holdout_runs_every_frozen_arm_on_the_new_days_once(tmp_path, monkeypatch):
    good, run_dir = frozen(tmp_path)
    seen = []
    monkeypatch.setattr(run, "pool", lambda jobs, w, p, l: seen.extend(jobs) or [
        {"arm": j["arm"]} for j in jobs])
    out = tmp_path / "stress"
    with pytest.raises(ValueError, match="not the one given"):
        stress.holdout(Ctx(), tmp_path, out, "0" * 64)
    report = stress.holdout(Ctx(), tmp_path, out, good)
    assert {j["day"] for j in seen} == {25, 26, 27, 28, 29}
    assert {j["shock"] for j in seen} == {None, "-0.05", "0.05"}
    assert {j["fee"] for j in seen} == {"0.0002"} and {j["split"] for j in seen} == {"holdout"}
    assert {j["arm"] for j in seen} == {"no_trade", "avellaneda_stoikov", "cpo"}
    assert len(seen) == 5 * 3 * 3
    assert report["stress_design_sha256"] == run.sha256(stress.STRESS_DESIGN)
    assert report["code_sha256"] == run.sha256(stress.HERE / "stress.py")
    with pytest.raises(ValueError, match="evaluated already"):
        stress.holdout(Ctx(), tmp_path, out, good)
    (out / "stress_holdout.json").unlink()
    # The marker of the read stays: even with the result gone, it is not read again.
    with pytest.raises(ValueError, match="started already"):
        stress.holdout(Ctx(), tmp_path, out, good)
    (out / "stress_holdout.started").unlink()
    (run_dir / "config.json").write_text('{"changed": 1}')
    with pytest.raises(ValueError, match="config.json changed"):
        stress.holdout(Ctx(), tmp_path, out, good)


def stress_rows(share, fills=lambda arm: 1, breach=None):
    rows = []
    for shock in (None, "-0.05", "0.05"):
        for regime in ("calm", "thin"):
            for n in (25, 26):
                for arm in ("no_trade", "avellaneda_stoikov", "ppo", "ppo_penalized", "cpo"):
                    seeds = ("seed1", "seed2") if arm in reporting.LEARNED else ("point",)
                    for seed in seeds:
                        value = share(arm, shock, regime, n, seed)
                        rows.append({"arm": arm, "shock": shock, "regime": regime, "day": n,
                                     "candidate": seed, "daily_value": "0", "episodes": [{
                                         "margin": {"not_healthy_share": value,
                                                    "steps": {"healthy": 1, "reduce_only": 0,
                                                              "default": 0},
                                                    "entries": 1, "entries_default": 2,
                                                    "largest_shortfall": "0",
                                                    "max_abs_position": "0"},
                                         "attribution": {"fills": fills(arm)},
                                         "limit_breach": breach == (arm, seed)}]})
    return {"rows": rows}


def test_the_report_compares_every_planned_pair_in_every_row_and_scope():
    _, series, plan = reporting.build(stress_rows(lambda *k: 0.0))
    assert len(plan) == 3 * 3 * len(reporting.PLAN)
    assert "shock 0.05/pooled/cpo-minus-avellaneda_stoikov" in series
    assert {(i["arm"], i["against"]) for i in plan} == set(reporting.PLAN)


def test_a_learned_day_is_the_mean_of_its_seeds_and_a_non_trading_arm_is_flagged():
    def share(arm, shock, regime, n, seed):
        return {"seed1": 0.2, "seed2": 0.6}.get(seed, 0.0)

    tables, series, plan = reporting.build(stress_rows(
        share, fills=lambda arm: 0 if arm == "ppo_penalized" else 3))
    assert tables["unshocked"]["calm"]["cpo"]["mean_not_healthy_share"] == "0.4"
    assert tables["unshocked"]["calm"]["cpo"]["entries_per_day"] == "1"
    assert tables["unshocked"]["calm"]["cpo"]["default_entries_per_day"] == "2"
    assert series["unshocked/pooled/cpo-minus-avellaneda_stoikov"] == ["0.4"] * 4
    item = next(i for i in plan if (i["arm"], i["against"]) == ("cpo", "ppo_penalized"))
    assert not item["against_traded"] and item["arm_traded"]
    other = next(i for i in plan if (i["arm"], i["against"]) == ("ppo", "avellaneda_stoikov"))
    assert other["against_traded"] and other["arm_traded"]


def test_fewer_margin_states_needs_an_interval_below_zero_and_no_breach():
    item = {"breaches": 0}
    assert reporting.decide(item, {"mean": -1, "lower": -2, "upper": -0.1}) == \
        "fewer margin states"
    assert reporting.decide(item, {"mean": -1, "lower": -2, "upper": 0}) == "not shown"
    assert reporting.decide(item | {"breaches": 1},
                            {"mean": -1, "lower": -2, "upper": -0.1}) == "not shown"
    with pytest.raises(ValueError, match="no interval"):
        reporting.decide(item, {"mean": None, "lower": None, "upper": None})
    _, _, plan = reporting.build(stress_rows(lambda *k: 0.0, breach=("ppo", "seed2")))
    item = next(i for i in plan if (i["arm"], i["against"]) == ("ppo", "avellaneda_stoikov")
                and i["scope"] == "calm" and i["row"] == "unshocked")
    # ppo's seed 2 breached on both of the scope's days.
    assert item["breaches"] == 2
    # A breach of the arm compared against counts too.
    against = next(i for i in plan if (i["arm"], i["against"]) == ("cpo", "ppo")
                   and i["scope"] == "calm" and i["row"] == "unshocked")
    assert against["breaches"] == 2


def test_the_report_refuses_a_stress_holdout_of_another_manifest_or_design(tmp_path):
    (tmp_path / "frozen.json").write_text("{}")
    out = tmp_path / "s"
    out.mkdir()
    base = {"frozen_sha256": run.sha256(tmp_path / "frozen.json"),
            "stress_design_sha256": run.sha256(stress.STRESS_DESIGN),
            "stress_digests_sha256": run.sha256(stress.STRESS_DIGESTS),
            "stress_shocks_sha256": run.sha256(stress.STRESS_SHOCKS)}
    (out / "stress_holdout.json").write_text(json.dumps(base))
    assert reporting.checked(tmp_path, out) == base
    (out / "stress_holdout.json").write_text(json.dumps(base | {"frozen_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the manifest"):
        reporting.checked(tmp_path, out)
    (out / "stress_holdout.json").write_text(json.dumps(base | {"stress_design_sha256": "0"}))
    with pytest.raises(ValueError, match="stress_design.json is not"):
        reporting.checked(tmp_path, out)


def test_a_failed_or_concurrent_stress_holdout_is_never_run_again(tmp_path, monkeypatch):
    good, _ = frozen(tmp_path)
    calls = []

    def failing(jobs, worker, parallel, label):
        calls.append(len(jobs))
        raise RuntimeError("a worker failed after reading days")

    monkeypatch.setattr(run, "pool", failing)
    out = tmp_path / "stress"
    with pytest.raises(RuntimeError, match="worker failed"):
        stress.holdout(Ctx(), tmp_path, out, good)
    assert (out / "stress_holdout.started").exists()
    assert not (out / "stress_holdout.json").exists()
    with pytest.raises(ValueError, match="started already"):
        stress.holdout(Ctx(), tmp_path, out, good)
    assert len(calls) == 1
    # A concurrent attempt finds the marker another one created, and stops before any job.
    other = tmp_path / "other"
    other.mkdir()
    (other / "stress_holdout.started").write_text("{}")
    with pytest.raises(ValueError, match="started already"):
        stress.holdout(Ctx(), tmp_path, other, good)
    assert len(calls) == 1
    # The marker is created exclusively: an attempt that loses the race fails on creation.
    racing = tmp_path / "racing"
    real_exists = pathlib.Path.exists

    def late(self):
        if self.name == "stress_holdout.started" and not real_exists(self):
            self.write_text("{}")
            return False
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", late)
    with pytest.raises(FileExistsError):
        stress.holdout(Ctx(), tmp_path, racing, good)
    assert len(calls) == 1
