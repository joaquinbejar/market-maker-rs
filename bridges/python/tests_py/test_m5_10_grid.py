"""The grid and identifiability rules of #108 part 2 on toy numbers."""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parents[1] / "experiments"
sys.path.insert(0, str(ROOT / "m5_10"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from m5_10_load import module  # noqa: E402

g = module("grid")

WEIGHTS = json.loads((ROOT / "m5_02" / "design.json").read_text())["causal"]["flow"]["weights"]
RATES = {"limit": 1.5, "cancel_per_lot": 0.05, "market": 0.3}
OPEN = {"below_cap": 0.9, "mean_queue": 8.0, "non_empty": 0.95}


def test_grid_order_and_size():
    p = g.points(0.25)
    assert len(p) == 24
    assert p[0] == {"market": 1, "limit": 1, "branching": 1, "impact": 0.25}
    assert p[1]["impact"] == 1.0
    assert g.fixed_stress(0.25) in p and g.impact_only(0.25) in p


def test_every_design_point_is_subcritical():
    assert all(g.plausible(WEIGHTS, p) for p in g.points(0.0))


def test_a_supercritical_point_is_refused():
    assert not g.plausible([[[0, 0, 0], [0, 0, 0], [0, 0, 1.0]]], {"branching": 1.2})


def test_unperturbed_ratio_is_one():
    base = g.points(0.0)[0]
    f = g.rate_formula(RATES, WEIGHTS, base, OPEN)
    assert g.rate_passes(3.0, 3.0, f, f)


def test_rate_tolerance_edges():
    assert g.rate_passes(1.09, 1.0, 1.0, 1.0) and not g.rate_passes(1.11, 1.0, 1.0, 1.0)


def test_formula_includes_cross_excitation():
    point = g.points(0.0)[0]
    alone = RATES["market"] * OPEN["non_empty"]
    assert g.rate_formula(RATES, WEIGHTS, point, OPEN) > alone


def test_slope_and_impact_rule():
    x = np.array([1.0, -1, 2, 0, -2])
    assert abs(g.slope(0.25 * x + 3, x) - 0.25) < 1e-12
    assert g.impact_passes(0.26, 0.25) and not g.impact_passes(0.30, 0.25)
    assert g.impact_passes(0.02, 0.0) and not g.impact_passes(0.03, 0.0)


def test_day_config_scales_only_the_point_and_leaves_the_input():
    causal = json.loads((ROOT / "m5_02" / "design.json").read_text())["causal"]
    before = json.dumps(causal)
    out = g.day_config(causal, g.fixed_stress(0.0), "p", 10, 100000)
    assert json.dumps(causal) == before
    assert np.allclose([out["flow"]["rates"]["market"], out["flow"]["rates"]["limit"]], [0.6, 1.05])
    assert out["flow"]["rates"]["cancel_per_lot"] == 0.05
    assert np.isclose(out["flow"]["weights"][0][2][2], 0.36)
    assert out["settings"] == [{"name": "p", "impact_ticks": 0.25}]
    assert out["seed_offset"] == 100000 and out["splits"]["holdout"] == [1, 10]
