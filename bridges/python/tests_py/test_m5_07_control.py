"""The control of #111 on toy parameters; the paper's examples are not run here."""

import pathlib
import sys
from decimal import Decimal as D

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_07"))
import control as c  # noqa: E402


def toy(kappa=D("0.01")):
    return c.inventory_problem([D(1), D("-0.5")], [D("0.2"), D("0.1")], [D(1), D(2)], [3, 3],
                               [1.0, 0.5], [kappa / 2, kappa / 2], kappa, 0.3)


def test_sigma_units():
    assert abs(c.sigma_per_second(3000, 0.18) - 0.22665) < 1e-4


def test_hjb_no_penalty_is_spread_times_intensity():
    p = c.inventory_problem([D(1)], [D("0.2")], [D(1)], [2], [1.0], [D(0)], D(0), 0.3)
    v, _ = c.solve_hjb(p, 1.0, 0.01)
    # at the centre both sides are open while the value is flat: earning at most 2 x 0.1 per second
    assert 0 < v[2] <= 0.2 + 1e-9


def test_hjb_penalised_centre_beats_edge_and_is_symmetric():
    p = c.inventory_problem([D(1)], [D("0.2")], [D(1)], [3], [1.0], [D(0)], D("0.1"), 1.0)
    v, _ = c.solve_hjb(p, 5.0, 0.01)
    assert v[3] == v.max()
    assert abs(v[0] - v[-1]) < 1e-9


def test_policy_never_posts_off_the_grid():
    p = toy()
    _, policy = c.solve_hjb(p, 1.0, 0.1, keep_policy_until=0.5)
    assert policy.shape[0] == 5
    bid_bit = 1 << 0
    assert not (policy[:, -1, :] & bid_bit).any()  # asset 0 at its top bound cannot buy


def test_dt_refinement_converges():
    p = toy()
    v1, _ = c.solve_hjb(p, 5.0, 0.02)
    v2, _ = c.solve_hjb(p, 5.0, 0.01)
    assert np.max(np.abs(v1 - v2)) < 1e-2


def test_a_is_symmetric_psd_and_b_is_zero_when_symmetric():
    a, b = c.asymptotic_ab([1, -0.028], [1, 20], [1, 0.1], [1, 0.1], [1, 5], [0.005, 0.005],
                           0.01, 0.2267, 0.18)
    assert np.allclose(a, a.T)
    assert np.linalg.eigvalsh(a).min() >= -1e-12
    assert np.allclose(b, 0)


def test_one_asset_a_matches_closed_form():
    a, _ = c.asymptotic_ab([1.0], [1.0], [2.0], [2.0], [0.5], [0.0], 0.04, 0.3, 0.18)
    # Gamma = sqrt(delta^2 / D_plus), D_plus = 4 x 0.5 = 2
    assert abs(a[0, 0] - 0.3 / 2 * 0.2 * np.sqrt(1 / 2)) < 1e-12


def test_net_risk_d_matches_a_for_uncoupled_asset():
    a, _ = c.asymptotic_ab([0.5], [2.0], [1.0], [1.0], [0.25], [0.0], 0.04, 0.3, 0.18)
    net = c.net_risk_coefficient([0.5], [2.0], [1.0], [0.25], 0.04, 0.3)
    # theta_hat = -D r^2 and v_hat = -A q^2 agree with r = delta q
    assert abs(net * 0.25 - a[0, 0]) < 1e-12


def test_greedy_matches_the_paper_closed_form():
    p = toy()
    a, b = np.array([[0.02, -0.003], [-0.003, 0.01]]), np.zeros(2)
    axes = [np.arange(-3, 4) * 1.0, np.arange(-3, 4) * 2.0]
    mask = c.greedy_from_v_hat(p, c.v_hat_inventory(p, a, b, axes))
    q = np.array([1.0, -2.0])
    idx = (4, 2)
    for i, mv in enumerate(p.moves):
        e = np.eye(2)[mv.asset]
        lhs = 2 * mv.phi * e @ a @ q + float(mv.m) * e @ a @ e + mv.phi * e @ b
        assert bool((mask[idx] >> i) & 1) == (lhs <= float(mv.d) / 2)


def test_net_risk_lattice_is_exact_and_rejects_off_lattice():
    p = c.net_risk_problem([D("0.533"), D("-0.014")], [D("0.1"), D("0.05")], [D(2), D(2)],
                           D(10), [0.05, 0.05], D(1), 0.2)
    assert p.shape == (10001,)
    assert p.moves[0].shift == 533 and p.moves[2].shift == -14
    try:
        c.net_risk_problem([D("0.0005")], [D("0.1")], [D(1)], D(10), [0.05], D(1), 0.2)
    except ValueError:
        return
    raise AssertionError


def test_common_random_numbers_do_not_depend_on_control():
    p = toy()
    n1, u1 = c.draw_path(101, 7, 50, len(p.moves))
    n2, u2 = c.draw_path(101, 7, 50, len(p.moves))
    assert (n1 == n2).all() and (u1 == u2).all()
    n3, _ = c.draw_path(101, 8, 50, len(p.moves))
    assert not (n1 == n3).all()


def test_backtest_stays_on_grid_and_revenue_is_exact():
    p = toy()
    always = np.full((1, *p.shape), 0b1111, np.uint8)
    pairs = [c.draw_path(1, k, 200, len(p.moves)) for k in range(20)]
    n, u = np.stack([x[0] for x in pairs]), np.stack([x[1] for x in pairs])
    counts, noise, state = c.backtest(p, always, n, u, 0.1, 0.3)
    assert all((s >= 0).all() and (s < 7).all() for s in state)
    pnl = c.path_pnl(p, counts, np.zeros_like(noise))
    expected = int(counts[0][0] + counts[0][1]) * D("0.1") + int(counts[0][2] + counts[0][3]) * D("0.1")
    assert pnl[0] == expected


def test_backtest_no_quote_has_no_fills():
    p = toy()
    never = np.zeros((1, *p.shape), np.uint8)
    n, u = c.draw_path(1, 0, 100, len(p.moves))
    counts, noise, _ = c.backtest(p, never, n[None], u[None], 0.1, 0.3)
    assert counts.sum() == 0 and noise[0] == 0


def test_solve_hjb_refuses_an_unstable_step():
    problem = c.net_risk_problem([D("0.5")], [D("0.1")], [D("2")], D("1"), [10.0], D("1"), 0.2)
    with pytest.raises(ValueError, match="explicit Euler"):
        c.solve_hjb(problem, 1.0, 0.1)
