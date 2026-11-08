"""The paper's setting and the extractors of #107, on synthetic episodes."""

import importlib.util
import json
import math
import pathlib
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m5_02"
sys.path.insert(0, str(EXPERIMENT))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"m5_02_{name}", EXPERIMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper_env = _load("paper_env")
nets = _load("nets")
run_paper = _load("run_paper")

EXP = paper_env.ExponentialKernel.from_branching(0.9, [5.0, 0.3])
POWER = paper_env.PowerLawKernel(0.7, 0.15)


def test_kernels_carry_the_stated_branching_ratio():
    assert EXP.norm() == pytest.approx(0.9, abs=1e-12)
    assert POWER.norm() == pytest.approx(0.7, abs=1e-12)
    assert POWER.c == pytest.approx(0.7 * 0.15, abs=1e-15)


def _expected_count(baseline, kernel, horizon, steps=4000):
    """E N(T) from the renewal equation m(t) = mu + int_0^t g(t - s) m(s) ds, by quadrature."""
    h = horizon / steps
    grid = np.arange(steps + 1) * h
    m = np.zeros(steps + 1)
    g = kernel(grid)
    for i in range(steps + 1):
        m[i] = baseline + (h * np.dot(g[1 : i + 1][::-1], m[:i]) if i else 0.0)
    return float(np.trapezoid(m, grid))


@pytest.mark.parametrize("kernel", [EXP, POWER], ids=["exponential", "power_law"])
def test_thinning_matches_the_mean_count(kernel):
    rng = np.random.default_rng(11)
    horizon, baseline = 2.0, 1.0
    counts = [len(paper_env.simulate_channel(rng, baseline, kernel, horizon)) for _ in range(20000)]
    expected = _expected_count(baseline, kernel, horizon)
    se = np.std(counts) / math.sqrt(len(counts))
    assert abs(np.mean(counts) - expected) < 4 * se, (np.mean(counts), expected, se)


def test_a_zero_kernel_is_poisson():
    rng = np.random.default_rng(3)
    flat = paper_env.ExponentialKernel([0.0], [1.0])
    counts = [len(paper_env.simulate_channel(rng, 1.5, flat, 2.0)) for _ in range(20000)]
    assert np.mean(counts) == pytest.approx(3.0, abs=4 * math.sqrt(3.0 / 20000))
    assert np.var(counts) == pytest.approx(3.0, rel=0.05)


def _run(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        obs, reward, done, _, info = env.step(policy(obs))
        total += reward
    return total, info


def test_rewards_sum_to_the_objective():
    env = paper_env.PaperEnv(EXP, 0.5)
    rng = np.random.default_rng(5)
    for seed in range(300):
        total, info = _run(env, seed, lambda _: rng.uniform(-10, 10, size=2))
        assert total == pytest.approx(info["objective"], abs=1e-12)


def test_an_episode_does_not_depend_on_the_policy():
    first = paper_env.simulate_episode(42, POWER, 0.5)
    env = paper_env.PaperEnv(POWER, 0.5)
    env.reset(seed=42)
    for a, b in zip(first, env._episode):
        np.testing.assert_array_equal(a, b)
    wide, _ = _run(env, 42, lambda _: np.array([10.0, 10.0]))
    env.reset(seed=42)
    for a, b in zip(first, env._episode):
        np.testing.assert_array_equal(a, b)


def _scripted(env, times, sides, uniforms):
    env.reset(seed=0)
    env._episode = (
        np.asarray(times, dtype=np.float64),
        np.asarray(sides, dtype=np.int64),
        np.zeros(len(times) + 1),
        np.asarray(uniforms, dtype=np.float64),
    )


def test_the_inventory_cap_stops_fills_but_not_observation():
    env = paper_env.PaperEnv(EXP, 0.5)
    _scripted(env, [0.01 * (i + 1) for i in range(7)], [-1] * 7, [0.0] * 7)
    fills = []
    for _ in range(8):
        _, _, done, _, info = env.step(np.array([-10.0, -10.0]))
        fills.append(info["filled"])
    assert done
    assert fills[:5] == ["bid"] * 5 and fills[5:7] == [None, None]
    assert info["inventory"] == 5
    assert len(env._seen) == 7


def test_a_fill_earns_its_spread_and_moves_inventory():
    env = paper_env.PaperEnv(EXP, 0.5)
    _scripted(env, [0.1, 0.2], [+1, -1], [0.0, 1.0])
    _, reward, _, _, info = env.step(np.array([0.0, 0.0]))
    ask = paper_env.spread_of(0.0)
    assert info["filled"] == "ask" and info["inventory"] == -1
    assert reward == pytest.approx(ask, abs=1e-15)
    assert info["cash"] == pytest.approx(paper_env.S0 + ask, abs=1e-12)
    _, _, _, _, info = env.step(np.array([0.0, 0.0]))
    assert info["filled"] is None


def test_the_observation_layout():
    env = paper_env.PaperEnv(EXP, 0.5)
    _scripted(env, [0.1, 0.25], [+1, -1], [1.0, 1.0])
    env.step(np.zeros(2))
    obs, *_ = env.step(np.zeros(2))
    layout = nets.Layout(paper_env.BUFFER, paper_env.CHANNELS, paper_env.SCALARS)
    ages, valid, values, scalars = layout.split(torch.as_tensor(obs[None, :]))
    assert ages[0, :3].tolist() == pytest.approx([0.0, 0.15, 0.0], abs=1e-6)
    assert valid[0, :3].tolist() == [1.0, 1.0, 0.0]
    assert values[0, 0].tolist() == [0.0, 1.0] and values[0, 1].tolist() == [1.0, 0.0]
    assert scalars[0].tolist() == pytest.approx([0.0, 0.0, 0.5], abs=1e-6)


LAYOUT = nets.Layout(4, 2, 1)
SPACE = gym.spaces.Box(-np.inf, np.inf, shape=(LAYOUT.size,), dtype=np.float32)


def _buffer(ages, sides, scalar=0.5, garbage=0.0):
    """Valid items first, newest first; invalid slots filled with `garbage`."""
    obs = np.full(LAYOUT.size, garbage, dtype=np.float32)
    L, C = LAYOUT.buffer, LAYOUT.channels
    obs[L : 2 * L] = 0.0
    for i, (age, side) in enumerate(zip(ages, sides)):
        obs[i] = age
        obs[L + i] = 1.0
        obs[2 * L + i * C : 2 * L + (i + 1) * C] = [1.0, 0.0] if side > 0 else [0.0, 1.0]
    obs[(2 + C) * L :] = scalar
    return torch.as_tensor(obs[None, :])


def test_hsn_is_the_mixture_of_exponentials():
    hsn = nets.Hsn(SPACE, LAYOUT, alphas=[1.5, 0.1], betas=[6.0, 0.4], floor=1e-4)
    out = hsn(_buffer([0.1, 0.3, 0.7], [+1, -1, +1]))
    a, b = [1.5, 0.1], [6.0, 0.4]
    buy = [a[j] * (math.exp(-b[j] * 0.1) + math.exp(-b[j] * 0.7)) for j in range(2)]
    sell = [a[j] * math.exp(-b[j] * 0.3) for j in range(2)]
    assert out[0].tolist() == pytest.approx(buy + sell + [0.5], rel=1e-5)


def test_hsn_decays_learn_and_a_fixed_ema_does_not():
    hsn = nets.Hsn(SPACE, LAYOUT, alphas=[1.5, 0.1], betas=[6.0, 0.4], floor=1e-4)
    hsn(_buffer([0.1, 0.3], [+1, -1])).sum().backward()
    assert hsn.raw_betas.grad is not None and hsn.raw_betas.grad.abs().sum() > 0
    ema = nets.FixedEma(SPACE, LAYOUT, decays=[0.4, 6.0])
    assert sum(p.numel() for p in ema.parameters()) == 0
    out = ema(_buffer([0.1, 0.3], [+1, -1]))
    assert out[0].tolist() == pytest.approx(
        [math.exp(-0.04), math.exp(-0.6), math.exp(-0.12), math.exp(-1.8), 0.5], rel=1e-5
    )


def test_hsn_stays_positive_at_any_raw_value():
    hsn = nets.Hsn(SPACE, LAYOUT, alphas=[1.5, 0.1], betas=[6.0, 0.4], floor=1e-4)
    with torch.no_grad():
        hsn.raw_betas.fill_(-1e4)
        hsn.raw_alphas.fill_(-1e4)
    assert (hsn.betas() >= 1e-4).all() and (hsn.alphas() >= 1e-4).all()


@pytest.mark.parametrize("make", [
    lambda: nets.Recurrent(SPACE, LAYOUT, cell="gru", embed=3, hidden=5),
    lambda: nets.Recurrent(SPACE, LAYOUT, cell="lstm", embed=3, hidden=5),
    lambda: nets.FrameStack(SPACE, LAYOUT, frames=3),
    lambda: nets.Hsn(SPACE, LAYOUT, alphas=[1.0, 0.5], betas=[2.0, 0.3], floor=1e-4),
    lambda: nets.FixedEma(SPACE, LAYOUT, decays=[1.0]),
], ids=["gru", "lstm", "frame_stack", "hsn", "fixed_ema"])
def test_invalid_slots_do_not_leak(make):
    torch.manual_seed(0)
    net = make()
    clean = net(_buffer([0.1, 0.4], [+1, -1]))
    dirty = net(_buffer([0.1, 0.4], [+1, -1], garbage=7.0))
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_recurrent_with_no_items_is_zero_and_reads_oldest_first(cell):
    torch.manual_seed(0)
    net = nets.Recurrent(SPACE, LAYOUT, cell=cell, embed=3, hidden=5)
    empty = net(_buffer([], []))
    assert empty[0, :5].abs().sum() == 0 and empty[0, 5] == 0.5
    items = _buffer([0.1, 0.4], [+1, -1])
    ages, valid, values, _ = LAYOUT.split(items)
    seq = torch.cat([ages.unsqueeze(-1), values], -1)[:, [1, 0]]
    _, state = net.rnn(torch.tanh(net.embed(seq)))
    last = (state[0] if cell == "lstm" else state)[-1]
    assert torch.allclose(net(items)[0, :5], last[0], atol=1e-6)


def test_welch_matches_a_known_value():
    assert run_paper.betainc(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-12)
    # Two-sided p of t = 2 with 10 degrees of freedom.
    p = run_paper.betainc(5.0, 0.5, 10.0 / (10.0 + 4.0))
    assert p == pytest.approx(0.0733880, abs=1e-6)
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    t, df, p = run_paper.welch(x, y)
    assert t == pytest.approx(-1.8973666, abs=1e-6)
    assert df == pytest.approx(5.8823529, abs=1e-6)
    assert p == pytest.approx(0.1075, abs=5e-4)


def test_design_widths_follow_the_parameter_rule():
    design, _ = run_paper.load_design()
    found = run_paper.check_widths(design)
    for name, (width, count) in found.items():
        assert abs(count - design["paper"]["parameters"]["target"]) <= 0.1 * 2600
