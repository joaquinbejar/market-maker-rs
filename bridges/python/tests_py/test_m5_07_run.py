"""The runner of #111 on toy inputs; the paper's examples are not run here."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_07"))
import run_stage_a as r  # noqa: E402


def test_shape_error_ignores_a_constant():
    v = np.array([1.0, 2.0, 4.0])
    assert r.shape_error(v, v - 7.0, np.ones(3, bool)) == 0


def test_shape_error_keeps_the_fitted_cell():
    v, tested = np.array([1.0, 2.0]), np.ones(2, bool)
    assert r.shape_error(v, np.array([0.0, 2.0]), tested) == pytest.approx(1.0)


def test_tested_cells_rules():
    p = r.build("example_1", "0.01", 0.2267)
    v = np.ones(p.shape)
    tested, excluded = r.tested_cells("example_1", p, v)
    assert tested[15, 15] and not tested[2, 15] and tested[3, 15]
    assert excluded["boundary"] > 0 and excluded["small_v"] == 0


def test_tested_cells_example_2_band_is_two_max_steps():
    p = r.build("example_2", "1", 0.2267)
    tested, _ = r.tested_cells("example_2", p, np.ones(p.shape))
    half = 5000
    edge = half - 2 * 533
    assert tested[half + edge - 1] and not tested[half + edge]


def test_central_block():
    assert (r.central(np.arange(5), (3,)) == [1, 2, 3]).all()


def test_bootstrap_is_paired_and_seeded():
    x = np.ones(50)
    assert r.bootstrap(x) == [1.0, 1.0]


def test_example_2_grid_matches_the_rust_table_size():
    assert r.build("example_2", "1", 0.2267).shape == (10001,)
    assert r.build("example_1", "0.01", 0.2267).shape == (31, 31)


def test_decide_rests_on_example_2_only():
    ok = {"status": "pass"}
    result = {"example_2": {"check_1": ok, "check_2": ok, "primary": {"1": {"verdict": "better"}}}}
    assert r.decide(result)["adopt"]
    result["example_2"]["check_2"] = {"status": "inconclusive"}
    decision = r.decide(result)
    assert not decision["adopt"] and decision["example_2_label"] == "control not validated"
    assert "unit choice" in decision["scope"]
    result["example_2"].update(check_2=ok, primary={"1": {"verdict": "no difference shown"}})
    assert not r.decide(result)["adopt"]


def test_example_3_uni_arms_sum_to_the_shared_noise():
    multi, uni = r.example_3_arms("0.01", 0.2267)
    assert multi.shape == (31,) * 4 and len(multi.moves) == 8 and all(u.shape == (31,) for u in uni)
    assert all((u.penalty[15] == 0) and u.moves[0].lam == 0.05 for u in uni)
    steps, paths = 200, 8
    normals, uniforms = np.random.default_rng(1).standard_normal((paths, steps)), np.random.default_rng(2).random((paths, steps, 8))
    always = [np.full((1, 31), 3, np.uint8) for _ in uni]
    total = r.run_uni(uni, always, normals, uniforms, 0.2267)
    alone = sum(np.array([float(x) for x in r.c.path_pnl(p, *r.c.backtest(p, a, normals, uniforms[:, :, [2 * j, 2 * j + 1]], r.DT, 0.2267)[:2])])
                for j, (p, a) in enumerate(zip(uni, always)))
    assert total == pytest.approx(alone)


def test_multi_asset_policy_posts_at_flat_inventory():
    multi, _ = r.example_3_arms("0.01", 0.2267)
    policy = r.multi_asset_policy(multi, "0.01", 0.2267)
    assert policy.shape == (1, 31, 31, 31, 31) and policy[0, 15, 15, 15, 15] == 255


def test_decoupled_v_hat_ignores_the_second_asset():
    p = r.build("example_1", "0.01", 0.2267)
    v = r.v_hat("example_1", p, "0.01", 0.2267, delta=r.DECOUPLED_DELTA)
    assert v[15, 0] == pytest.approx(v[15, 15], abs=1e-9)


def test_claim_refuses_an_altered_design(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "DESIGN_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="freeze"):
        r.claim(tmp_path, None)


def test_marker_is_written_before_any_evaluation(tmp_path, monkeypatch):
    def boom(*_):
        assert (tmp_path / "attempt.jsonl").exists()
        raise RuntimeError

    monkeypatch.setattr(r, "check_1", boom)
    with pytest.raises(RuntimeError):
        r.evaluate(tmp_path, tmp_path)
    with pytest.raises(SystemExit, match="already started"):
        r.evaluate(tmp_path, tmp_path)


def test_a_recovery_under_another_freeze_is_refused(tmp_path):
    marker = tmp_path / "attempt.jsonl"
    marker.write_text(json.dumps({"kind": "attempt", "frozen_sha256": "0" * 64}) + "\n")
    before = marker.read_text()
    with pytest.raises(SystemExit, match="another frozen_sha256"):
        r.claim(tmp_path, "crash")
    assert marker.read_text() == before


def test_recoveries_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "check_1", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(RuntimeError):
        r.evaluate(tmp_path, tmp_path)
    with pytest.raises(RuntimeError):
        r.evaluate(tmp_path, tmp_path, recover="crash")
    with pytest.raises(SystemExit, match="too many"):
        r.evaluate(tmp_path, tmp_path, recover="again")
    lines = (tmp_path / "attempt.jsonl").read_text().splitlines()
    assert json.loads(lines[1])["reason"] == "crash"
