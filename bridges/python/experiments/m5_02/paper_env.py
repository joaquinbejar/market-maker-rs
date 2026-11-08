"""The single-asset setting of the Hawkes-Structured Network paper, rebuilt from its text (#107).

Acero, Plank and Zhang, *Reinforcement Learning with Hawkes-Structured Networks for Market
Making* (arXiv:2610.16524v1), single-asset experiments. SYNTHETIC: two independent Hawkes
processes of market orders, a Brownian mid with zero direct impact, and an exponential fill
probability. Nothing here touches the crate's simulator or kernel, and nothing here says
anything about a market.

What the paper states, and is used as stated:

- buy market orders `M+` and sell market orders `M-` are two independent Hawkes processes,
  baselines 1.0 and 0.7, sharing one kernel;
- `S_t = S0 + sigma W_t` with `S0 = 100`, `sigma = 0.1`, and no impact;
- a buy market order fills the agent's ask at distance `delta+` with probability
  `exp(-10 delta+)`, a sell market order its bid likewise;
- spreads in `[0.005, 0.30]`, set at the start and after each market order;
- the objective `X_T + Q_T (S_T - kappa Q_T) - gamma int Q_t^2 dt`, `gamma = 0.10`,
  `kappa = 0.001`, and `|Q| <= 5`.

What the paper leaves open, and is chosen here (the design records each choice):

- one fill is one unit;
- at `Q = 5` the bid does not fill, at `Q = -5` the ask does not; the order is still
  observed;
- the reward of a step is the change of `X + Q S` over it minus `gamma Q^2 dt`, and the last
  step adds `-kappa Q_T^2`, so an episode's rewards sum exactly to the objective;
- events are simulated by Ogata thinning, exact for these non-increasing kernels;
- every random draw of an episode (event times, the mid's increments, the fill draws) comes
  from its seed before the first step, so the path does not depend on the policy and every
  arm meets the same episodes.

The observation follows `nets.Layout`: the ages `t - tau` of the last `L` market orders,
newest first, their validity flags, their sides one-hot (buy, sell), then
`(S - S0) / (sigma sqrt T)`, `Q / 5` and `(T - t) / T`.
"""

import math

import gymnasium as gym
import numpy as np

S0 = 100.0
SIGMA = 0.1
FILL_DECAY = 10.0
SPREAD_LOW = 0.005
SPREAD_HIGH = 0.30
GAMMA_RUNNING = 0.10
KAPPA_TERMINAL = 0.001
MAX_INVENTORY = 5
BASELINES = {+1: 1.0, -1: 0.7}
BUFFER = 32
RAW_BOUND = 10.0
CHANNELS = 2
SCALARS = 3


class ExponentialKernel:
    """`g(t) = sum_j alpha_j exp(-beta_j t)`."""

    def __init__(self, alphas, betas):
        if len(alphas) != len(betas) or not alphas:
            raise ValueError("one alpha per beta, at least one")
        if any(a < 0 for a in alphas) or any(b <= 0 for b in betas):
            raise ValueError("alpha >= 0 and beta > 0")
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.betas = np.asarray(betas, dtype=np.float64)

    @classmethod
    def from_branching(cls, n, betas):
        """`alpha_j = (n / p) beta_j`, so that each mode carries `n / p` of the norm."""
        p = len(betas)
        return cls([n / p * b for b in betas], betas)

    def __call__(self, ages):
        ages = np.asarray(ages, dtype=np.float64)
        return (self.alphas * np.exp(-np.multiply.outer(ages, self.betas))).sum(axis=-1)

    def norm(self):
        return float((self.alphas / self.betas).sum())


class PowerLawKernel:
    """`g(t) = c (t + tau0)^-(1 + eps)`, with `c = n eps tau0^eps`, so its norm is `n`."""

    def __init__(self, n, eps, tau0=1.0):
        if not (0 < n < 1) or eps <= 0 or tau0 <= 0:
            raise ValueError("0 < n < 1, eps > 0, tau0 > 0")
        self.n, self.eps, self.tau0 = n, eps, tau0
        self.c = n * eps * tau0**eps

    def __call__(self, ages):
        ages = np.asarray(ages, dtype=np.float64)
        return self.c * (ages + self.tau0) ** (-(1.0 + self.eps))

    def norm(self):
        return self.c * self.tau0 ** (-self.eps) / self.eps


def kernel_of(setting):
    kind = setting["kernel"]
    if kind == "exponential":
        return ExponentialKernel.from_branching(setting["n"], setting["betas"])
    if kind == "power_law":
        return PowerLawKernel(setting["n"], setting["eps"], setting.get("tau0", 1.0))
    raise ValueError(f"unknown kernel {kind}")


def intensity(baseline, kernel, events, t):
    """`lambda(t) = mu + sum over events strictly before or at t of g(t - tau)`."""
    past = np.asarray([e for e in events if e <= t], dtype=np.float64)
    if past.size == 0:
        return baseline
    return baseline + float(kernel(t - past).sum())


def simulate_channel(rng, baseline, kernel, horizon):
    """Ogata thinning on `[0, horizon)`; exact because the kernel never increases."""
    events = []
    t = 0.0
    while True:
        bound = intensity(baseline, kernel, events, t)
        t += rng.exponential(1.0 / bound)
        if t >= horizon:
            return events
        if rng.uniform() * bound <= intensity(baseline, kernel, events, t):
            events.append(t)


def simulate_episode(seed, kernel, horizon):
    """Every random draw of one episode, before its first step."""
    rng = np.random.default_rng(seed)
    times, sides = [], []
    for side in (+1, -1):
        channel = simulate_channel(rng, BASELINES[side], kernel, horizon)
        times += channel
        sides += [side] * len(channel)
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    times = np.asarray(times, dtype=np.float64)[order]
    sides = np.asarray(sides, dtype=np.int64)[order]
    # One mid increment per interval (events + the last one to the horizon), one fill draw
    # per event.
    normals = rng.standard_normal(len(times) + 1)
    uniforms = rng.uniform(size=len(times))
    return times, sides, normals, uniforms


def spread_of(raw):
    """The policy's unbounded output to a spread in `[0.005, 0.30]` through a sigmoid."""
    raw = float(np.clip(raw, -60.0, 60.0))
    return SPREAD_LOW + (SPREAD_HIGH - SPREAD_LOW) / (1.0 + math.exp(-raw))


class PaperEnv(gym.Env):
    """One episode is `[0, T]`; a step runs from one decision to the next market order.

    The action is `(raw ask, raw bid)`, each mapped to a spread by `spread_of`. `info` carries
    the exact components: cash, inventory, mid, fills, and the episode's objective at its end.
    """

    metadata = {"render_modes": []}

    def __init__(self, kernel, horizon, buffer=BUFFER):
        super().__init__()
        self.kernel = kernel
        self.horizon = float(horizon)
        self.buffer = buffer
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=((2 + CHANNELS) * buffer + SCALARS,), dtype=np.float32
        )
        # The trainer needs a bounded box and clips to it; a raw value of 10 is within
        # 5e-5 of either end of the spread range.
        self.action_space = gym.spaces.Box(-RAW_BOUND, RAW_BOUND, shape=(2,), dtype=np.float32)
        self._episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**63 - 1))
        self._episode = simulate_episode(seed, self.kernel, self.horizon)
        self._next = 0
        self._t = 0.0
        self._mid = S0
        self._cash = 0.0
        self._inventory = 0
        self._running = 0.0
        self._fills = 0
        self._seen = []
        return self._observation(), {"seed": seed}

    def _observation(self):
        L = self.buffer
        obs = np.zeros((2 + CHANNELS) * L + SCALARS, dtype=np.float64)
        recent = self._seen[::-1][:L]
        for i, (tau, side) in enumerate(recent):
            obs[i] = self._t - tau
            obs[L + i] = 1.0
            obs[2 * L + i * CHANNELS + (0 if side == +1 else 1)] = 1.0
        base = (2 + CHANNELS) * L
        obs[base] = (self._mid - S0) / (SIGMA * math.sqrt(self.horizon))
        obs[base + 1] = self._inventory / MAX_INVENTORY
        obs[base + 2] = (self.horizon - self._t) / self.horizon
        return obs.astype(np.float32)

    def step(self, action):
        if self._episode is None:
            raise RuntimeError("reset before step")
        times, sides, normals, uniforms = self._episode
        ask, bid = spread_of(action[0]), spread_of(action[1])
        k = self._next
        last = k == len(times)
        until = self.horizon if last else float(times[k])
        dt = until - self._t
        mid = self._mid + SIGMA * math.sqrt(dt) * float(normals[k])
        penalty = GAMMA_RUNNING * self._inventory**2 * dt
        reward = self._inventory * (mid - self._mid) - penalty
        self._running += penalty
        self._mid, self._t = mid, until
        filled = None
        if not last:
            side = int(sides[k])
            if side == +1 and self._inventory > -MAX_INVENTORY:
                if uniforms[k] < math.exp(-FILL_DECAY * ask):
                    self._cash += mid + ask
                    self._inventory -= 1
                    reward += ask
                    filled = "ask"
            elif side == -1 and self._inventory < MAX_INVENTORY:
                if uniforms[k] < math.exp(-FILL_DECAY * bid):
                    self._cash -= mid - bid
                    self._inventory += 1
                    reward += bid
                    filled = "bid"
            if filled:
                self._fills += 1
            self._seen.append((until, side))
            self._next = k + 1
        else:
            reward -= KAPPA_TERMINAL * self._inventory**2
        info = {
            "cash": self._cash,
            "inventory": self._inventory,
            "mid": self._mid,
            "filled": filled,
            "spreads": (ask, bid),
        }
        if last:
            info["objective"] = self.objective()
            info["fills"] = self._fills
        return self._observation(), float(reward), last, False, info

    def objective(self):
        """`X_T + Q_T (S_T - kappa Q_T) - gamma int Q^2 dt`, at the time reached."""
        q = self._inventory
        return self._cash + q * (self._mid - KAPPA_TERMINAL * q) - self._running
