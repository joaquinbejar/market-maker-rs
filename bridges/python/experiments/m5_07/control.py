"""The control of #111 under the paper's Approximation 3.1: exact HJB, asymptotic quadratic, backtest.

An inventory problem is a rectangular grid with a list of moves. Example 1 has one axis per asset;
the net-risk Example 2 has one axis, the exact lattice of r in steps of 0.002. Bid is phi = +1.
"""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

SECONDS_PER_YEAR = 365 * 86400
LATTICE = Decimal("0.002")


def sigma_per_second(s0, v0, seconds_per_year=SECONDS_PER_YEAR):
    return s0 * np.sqrt(v0 / seconds_per_year)


@dataclass(frozen=True)
class Move:
    axis: int
    shift: int
    phi: int
    asset: int
    m: Decimal
    d: Decimal
    lam: float


@dataclass(frozen=True)
class Problem:
    moves: tuple
    penalty: np.ndarray  # running cost per second at each state
    risk: np.ndarray  # r = delta . q at each state
    origin: tuple  # index of q = 0

    @property
    def shape(self):
        return self.penalty.shape


def _slices(move, n_axes):
    """The (source, target) slices of the states from which a move stays on the grid."""
    src, dst = [slice(None)] * n_axes, [slice(None)] * n_axes
    if move.shift > 0:
        src[move.axis], dst[move.axis] = slice(None, -move.shift), slice(move.shift, None)
    else:
        src[move.axis], dst[move.axis] = slice(-move.shift, None), slice(None, move.shift)
    return tuple(src), tuple(dst)


def _moves_for(asset, axis, step, m, d, lam):
    return [Move(axis, step, 1, asset, m, d, lam), Move(axis, -step, -1, asset, m, d, lam)]


def inventory_problem(delta, d, m, levels, lam, kappa_j, kappa, sigma):
    """Per-asset inventory bounds: asset j takes q = m_j * i for |i| <= levels[j]."""
    delta = np.asarray([float(x) for x in delta])
    axes = [float(m[j]) * np.arange(-n, n + 1) for j, n in enumerate(levels)]
    q = np.meshgrid(*axes, indexing="ij")
    exposure = [delta[j] * q[j] for j in range(len(q))]
    risk = sum(exposure)
    penalty = 0.5 * sigma**2 * (sum(float(k) * e**2 for k, e in zip(kappa_j, exposure))
                                + float(kappa) * risk**2)
    moves = [mv for j in range(len(q)) for mv in _moves_for(j, j, 1, m[j], d[j], lam[j])]
    return Problem(tuple(moves), penalty, risk, tuple(levels))


def net_risk_problem(delta, d, m, bound, lam, kappa, sigma):
    """Example 2's state: r on the lattice of multiples of 0.002 in [-bound, bound]."""
    half = bound / LATTICE
    if half != half.to_integral_value():
        raise ValueError("bound is not on the lattice")
    half = int(half)
    moves = []
    for j, delta_j in enumerate(delta):
        step = m[j] * delta_j / LATTICE
        if step != step.to_integral_value():
            raise ValueError("m delta is not on the lattice")
        step = int(step)
        # phi is the side of the order; a negative delta moves r the other way.
        moves += [Move(0, step, 1, j, m[j], d[j], lam[j]), Move(0, -step, -1, j, m[j], d[j], lam[j])]
    r = float(LATTICE) * np.arange(-half, half + 1)
    return Problem(tuple(moves), 0.5 * float(kappa) * sigma**2 * r**2, r, (half,))


def _mask_type(problem):
    return np.uint8 if len(problem.moves) <= 8 else np.uint16


def solve_hjb(problem, horizon, dt, keep_policy_until=0.0):
    """Explicit backward Euler of the HJB. Returns v(0, .) and the policy bitmask by step.

    The policy array has one row per step of length dt from t = 0 up to keep_policy_until; bit i
    says move i is posted, and only where it stays on the grid.
    """
    if dt * sum(mv.lam for mv in problem.moves) >= 0.5:
        raise ValueError("explicit Euler needs dt times the total intensity well below 1")
    steps = int(round(horizon / dt))
    keep = int(round(keep_policy_until / dt))
    n_axes = len(problem.shape)
    policy = np.zeros((keep, *problem.shape), _mask_type(problem)) if keep else None
    slices = [_slices(mv, n_axes) for mv in problem.moves]
    half = [float(mv.d) / 2 for mv in problem.moves]
    v = np.zeros(problem.shape)
    for step in range(steps - 1, -1, -1):
        rate = -problem.penalty.copy()
        mask = np.zeros(problem.shape, _mask_type(problem)) if step < keep else None
        for i, (mv, (src, dst)) in enumerate(zip(problem.moves, slices)):
            m = float(mv.m)
            p = (v[src] - v[dst]) / m
            post = p <= half[i]
            rate[src] += m * mv.lam * post * (half[i] - p)
            if mask is not None:
                mask[src] |= (post.astype(mask.dtype) << i)
        if mask is not None:
            policy[step] = mask
        v = v + dt * rate
    return v, policy


def asymptotic_ab(delta, m, lam_b, lam_a, alpha, kappa_j, kappa, sigma, v0, mu=0.0):
    """The paper's T -> infinity A and B. V0 scales only the term of B that mu switches on."""
    delta = np.asarray(delta, float)
    z, lam_b, lam_a, alpha = (np.asarray(x, float) for x in (m, lam_b, lam_a, alpha))
    d_plus = (lam_b + lam_a) * alpha * z
    d_minus = np.diag((lam_b - lam_a) * alpha * z)
    v_minus = (lam_a - lam_b) * z
    sigma_m = np.diag(np.asarray(kappa_j, float) * delta**2 / kappa) + np.outer(delta, delta)
    root = np.sqrt(d_plus)
    inner = _sqrtm((root[:, None] * sigma_m) * root)
    gamma = inner / np.outer(root, root)
    a = sigma / 2 * np.sqrt(kappa) * gamma
    a_hat = np.sqrt(kappa * v0) * inner
    pinv = np.linalg.pinv(a_hat)
    b = -(pinv * root[:, None] / root[None, :]) @ delta * mu \
        - ((a_hat @ pinv) / np.outer(root, root)) @ (v_minus + sigma / 2 * np.sqrt(kappa) * d_minus @ np.diag(gamma))
    return a, b


def _sqrtm(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def v_hat_inventory(problem, a, b, m_axes):
    """-q'Aq - q'B on the grid of inventory_problem, with C left at 0."""
    q = np.stack(np.meshgrid(*m_axes, indexing="ij"), axis=-1)
    return -np.einsum("...i,ij,...j", q, a, q) - q @ b


def net_risk_coefficient(delta, m, lam, alpha, kappa, sigma):
    """D of the paper's net-risk formula, so that theta_hat(r) = -D r^2 - C."""
    delta, m, lam, alpha = (np.asarray(x, float) for x in (delta, m, lam, alpha))
    return sigma / 2 * np.sqrt(kappa / (2 * np.sum(delta**2 * lam * m * alpha)))


def greedy_from_v_hat(problem, v_hat):
    """Bitmask of the greedy rule: post if (v_hat(s) - v_hat(s + move)) / m <= D / 2, on the grid."""
    mask = np.zeros(problem.shape, _mask_type(problem))
    for i, mv in enumerate(problem.moves):
        src, dst = _slices(mv, len(problem.shape))
        post = (v_hat[src] - v_hat[dst]) / float(mv.m) <= float(mv.d) / 2
        mask[src] |= post.astype(mask.dtype) << i
    return mask


def fill_probabilities(problem, dt):
    return np.array([1.0 - np.exp(-mv.lam * dt) for mv in problem.moves])


def draw_path(seed, path, steps, n_moves):
    """The control-independent randomness of one path: dW normals and one uniform per move, step."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, path]))
    return rng.standard_normal(steps), rng.random((steps, n_moves))


def backtest(problem, policy, normals, uniforms, dt, sigma):
    """Runs paths (rows of normals and uniforms). policy has one row per step, or one row in all.

    A move is posted from the state at the step's start and fills against its uniform, in move
    order, and only if it still stays on the grid. Returns (fill counts per path and move, the
    float sigma dW term per path, final state indices).
    """
    paths, steps = normals.shape
    moves = problem.moves
    p = fill_probabilities(problem, dt)
    if len(problem.shape) == 1:
        state = [np.full(paths, problem.origin[0])]
    else:
        state = [np.full(paths, o) for o in problem.origin]
    counts = np.zeros((paths, len(moves)), np.int64)
    noise = np.zeros(paths)
    root = sigma * np.sqrt(dt)
    for step in range(steps):
        idx = tuple(state)
        noise += problem.risk[idx] * root * normals[:, step]
        table = policy[step if len(policy) > 1 else 0]
        mask = table[idx]
        for i, mv in enumerate(moves):
            fill = ((mask >> i) & 1).astype(bool) & (uniforms[:, step, i] < p[i])
            target = state[mv.axis] + mv.shift
            fill &= (target >= 0) & (target < problem.shape[mv.axis])
            state[mv.axis] = np.where(fill, target, state[mv.axis])
            counts[:, i] += fill
    return counts, noise, state


def path_pnl(problem, counts, noise):
    """Exact fill revenue in Decimal plus the float sigma dW term, one Decimal per path."""
    per_fill = [mv.m * mv.d / 2 for mv in problem.moves]
    return [sum((int(c) * r for c, r in zip(row, per_fill)), Decimal(0)) + Decimal(float(n))
            for row, n in zip(counts, noise)]
