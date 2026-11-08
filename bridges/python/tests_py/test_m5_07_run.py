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
