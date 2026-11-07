"""The grid, the perturbation and the adversary's rule of #108, on hand-made inputs."""

import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m5_03"
SPEC = importlib.util.spec_from_file_location("m5_03_run", STUDY / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

DESIGN = json.loads((STUDY / "design.json").read_text())
M4_05 = json.loads((HERE.parent / "experiments" / "m4_05" / "design.json").read_text())


def test_the_grid_has_the_designs_24_points_with_the_unperturbed_one_first():
    grid = run.points(DESIGN)
    assert len(grid) == 24
    assert len({name for name, _ in grid}) == 24
    assert grid[0][1] == {"market_rate_multiplier": 1.0, "cancel_rate_multiplier": 1.0,
                          "theta_multiplier": 1.0, "limit_rate_multiplier": 1.0}
    names = [name for name, _ in grid]
    assert run.point_name(DESIGN["fixed_stress"]) in names


def test_a_perturbation_multiplies_its_own_column_and_theta_only():
    point = {"market_rate_multiplier": 2.0, "cancel_rate_multiplier": 1.5,
             "theta_multiplier": 1.2, "limit_rate_multiplier": 0.7}
    out = run.perturbed(M4_05, point, "holdout", DESIGN)
    for before, after in zip(M4_05["simulator"]["regimes"], out["simulator"]["regimes"], strict=True):
        assert after["theta"] == pytest.approx(before["theta"] * 1.2)
        assert after["theta"] < 1.0
        for table_b, table_a in zip(before["rates"], after["rates"], strict=True):
            for row_b, row_a in zip(table_b, table_a, strict=True):
                assert row_a == pytest.approx([row_b[0] * 0.7, row_b[1] * 1.5, row_b[2] * 2.0])
        assert after["event_sizes"] == before["event_sizes"]
        assert after["tick_size"] == before["tick_size"]
    assert out["days"] == DESIGN["days"]["holdout"]["per_regime"]
    assert out["seed_offset"] == DESIGN["days"]["holdout"]["seed_offset"]
    assert out["splits"]["train"] == [1, out["days"]]
    # #102's design is left as it was.
    assert M4_05["simulator"]["regimes"][0]["theta"] == 0.6


def test_a_theta_that_reaches_one_is_refused():
    point = {"market_rate_multiplier": 1.0, "cancel_rate_multiplier": 1.0,
             "theta_multiplier": 1.3, "limit_rate_multiplier": 1.0}
    with pytest.raises(ValueError, match="theta"):
        run.perturbed(M4_05, point, "selection", DESIGN)


def test_cvar_is_the_mean_loss_of_the_worst_days():
    # 30 days: ceil(5 % x 30) = 2 worst losses, 9 and 7.
    nets = [-9, -7, -1] + [1] * 27
    assert run.cvar_loss(nets, 0.05) == pytest.approx(8.0)
    # Fewer days than 20 still take one.
    assert run.cvar_loss([-3, 2, 5], 0.05) == pytest.approx(3.0)


def test_the_adversary_picks_per_arm_the_largest_tail_loss_and_the_first_on_a_tie():
    flat = [0.0] * 30
    by_point = {
        "a": {arm: list(flat) for arm in run.ARMS},
        "b": {arm: list(flat) for arm in run.ARMS},
        "c": {arm: list(flat) for arm in run.ARMS},
    }
    by_point["b"]["glft"] = [-5.0] * 2 + [0.0] * 28
    by_point["c"]["glft"] = [-4.0] * 30
    by_point["c"]["maskable_ppo"] = [-1.0] + [0.0] * 29
    chosen = run.choose(by_point)
    assert chosen["glft"]["point"] == "b"
    assert chosen["glft"]["cvar_loss"] == pytest.approx(5.0)
    assert chosen["maskable_ppo"]["point"] == "c"
    # Every point ties for no-trade: the first in order.
    assert chosen["no_trade"]["point"] == "a"


def test_a_learned_arms_day_is_the_mean_over_its_seeds():
    rows = [{"arm": "maskable_ppo", "regime": "calm", "day": 1, "net": n} for n in ("1", "2", "6")]
    rows.append({"arm": "no_trade", "regime": "calm", "day": 1, "net": "0"})
    out = run.daily(rows)
    assert out[("maskable_ppo", "calm", 1)] == 3
    assert out[("no_trade", "calm", 1)] == 0


def test_the_holdout_refuses_digests_other_than_the_frozen_set(tmp_path):
    for name, body in (("a.json", "1"), ("b.json", "2")):
        (tmp_path / name).write_text(body)
    frozen = {"digests_sha256": run.digest_hashes(tmp_path)}
    run.check_frozen_digests(frozen, tmp_path)
    (tmp_path / "b.json").write_text("3")
    with pytest.raises(ValueError, match="digests changed"):
        run.check_frozen_digests(frozen, tmp_path)
    (tmp_path / "b.json").write_text("2")
    (tmp_path / "c.json").write_text("4")
    with pytest.raises(ValueError, match="digest files changed"):
        run.check_frozen_digests(frozen, tmp_path)
    (tmp_path / "c.json").unlink()
    (tmp_path / "a.json").unlink()
    with pytest.raises(ValueError, match="digest files changed"):
        run.check_frozen_digests(frozen, tmp_path)
    with pytest.raises(ValueError, match="no digests"):
        run.check_frozen_digests({}, tmp_path)


def test_the_holdout_checks_the_digests_before_any_job(tmp_path, monkeypatch):
    calls = []

    def refuse(*_):
        raise ValueError("digests changed")

    monkeypatch.setattr(run, "check_frozen_digests", refuse)
    monkeypatch.setattr(run, "pool", lambda *a, **k: calls.append(a))
    out = tmp_path
    (out / "selection.json").write_text("{}")
    header = {"design_sha256": "d", "code_sha256": "c"}
    frozen = {**header, "selection_sha256": run.sha256(out / "selection.json"), "chosen": {}}
    (out / "frozen.json").write_text(json.dumps(frozen))

    class Study:
        pass

    study = Study()
    study.out = out
    study.header = lambda step: dict(header)
    with pytest.raises(ValueError, match="digests changed"):
        run.holdout(study, run.sha256(out / "frozen.json"))
    assert calls == []
