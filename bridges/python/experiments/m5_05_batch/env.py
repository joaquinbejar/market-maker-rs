"""A market making environment over the batch auction toy of #289. SYNTHETIC.

Time runs in rows of `row_ms` (100 ms) inside batches of `batch_ms` (500 ms). At the start of
each row the agent may decide; a decision replaces its quote: the cancel of the old orders and the
new orders are sent at the row's start and arrive `latency_ms` later (the same delay for both, so
the switch is clean at the arrival). An order or cancel acts in the batch whose close is strictly
after its arrival (`fba.batch_of`). A decision holds for `duration` rows, after which the agent
decides again; between decisions nothing is sent and the quote rests.

The background flow of a day is drawn from its seed before the day starts and does not depend on
the agent: a fundamental that moves every row, limit orders around it, informed orders priced at
the fundamental at their arrival (which cross a stale agent quote after a move), and noise orders
through the reference. Background orders act in their arrival's batch and expire at its clearing.
The agent's orders rest until cancelled or filled.

The agent observes, at each decision and with no delay, the last clearing price, a public signal
(the fundamental plus noise), the rows left to the close, the last batch's volume and imbalance,
its position and its resting quote. Its quote is priced from the signal: the bid `offset + 1`
ticks below it, the ask `offset + 1` above. Money is `Decimal`: cash, fees and the mark at the
last clearing price, so the reward of a day sums to its net.
"""

import math
import pathlib
import sys
from dataclasses import dataclass, field
from decimal import Decimal

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fba  # noqa: E402

OFFSETS = (0, 1, 2, 4, 8)
SIZES = (1, 2)
DURATIONS = (1, 5, 10, 30)


@dataclass(frozen=True)
class Flow:
    """Made-up parameters of the background flow, per row."""

    reference_ticks: int = 10_000
    fundamental_sd: float = 0.3
    signal_sd: float = 1.0
    limit_rate: float = 3.0
    limit_depth_p: float = 0.5
    limit_lots: int = 3
    informed_rate: float = 0.4
    informed_share: float = 0.7
    noise_rate: float = 0.2
    noise_reach: int = 5
    market_lots: int = 2


@dataclass(frozen=True)
class Background:
    """A day's background: the fundamental and the signal at each row's start, and the orders
    as `(arrival_ms, side, price, lots)`, in arrival order. Informed and noise orders carry
    their side's rule; `reference` resolves at the batch."""

    fundamental: np.ndarray
    signal: np.ndarray
    orders: list = field(default_factory=list)


def draw_background(flow, seed, rows, row_ms):
    """The background of one day from its seed; noise orders are priced against the reference
    at their batch, so their price is resolved when the batch clears (`price` None, `reach`)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, flow.fundamental_sd, rows)
    fundamental = flow.reference_ticks + np.concatenate(([0.0], np.cumsum(steps[:-1])))
    signal = fundamental + rng.normal(0.0, flow.signal_sd, rows)
    orders = []
    for row in range(rows):
        start = row * row_ms
        f = fundamental[row]
        for _ in range(rng.poisson(flow.limit_rate)):
            side = 1 if rng.random() < 0.5 else -1
            depth = int(rng.geometric(flow.limit_depth_p))
            price = int(round(f)) - side * depth
            orders.append((start + int(rng.integers(row_ms)), side, price,
                           int(rng.integers(1, flow.limit_lots + 1)), "limit"))
        for _ in range(rng.poisson(flow.informed_rate)):
            informed = rng.random() < flow.informed_share
            orders.append((start + int(rng.integers(row_ms)), 0 if informed else
                           (1 if rng.random() < 0.5 else -1), int(round(f)),
                           int(rng.integers(1, flow.market_lots + 1)),
                           "informed" if informed else "noise"))
        for _ in range(rng.poisson(flow.noise_rate)):
            side = 1 if rng.random() < 0.5 else -1
            orders.append((start + int(rng.integers(row_ms)), side, None,
                           int(rng.integers(1, flow.market_lots + 1)), "noise"))
    orders.sort(key=lambda o: o[0])
    return Background(fundamental, signal, orders)


def resolve(order, reference, flow):
    """A background order as the auction sees it, against the batch's reference."""
    arrival, side, price, lots, kind = order
    if kind == "informed":
        # It buys below the fundamental and sells above it; at the reference it stays out.
        side = 1 if price > reference else -1 if price < reference else 0
        return None if side == 0 else (arrival, side, price, lots)
    if kind == "noise":
        price = reference + side * flow.noise_reach if price is None else price
        return arrival, side, price, lots
    return arrival, side, price, lots


@dataclass
class Resting:
    order_id: int
    side: int
    price: int
    lots: int
    arrival_ms: int
    cancel_ms: int | None = None


class BatchEnv(gym.Env):
    """One day of the batch auction for one agent. `with_duration` adds the duration value to
    the action; `predictor` (a callable on the market part of the observation) adds its value."""

    metadata = {"render_modes": []}

    def __init__(self, *, seed, rows, latency_ms, fee_rate, flow=Flow(), row_ms=100,
                 batch_ms=500, tick=Decimal("0.01"), max_position=20, with_duration=True,
                 predictor=None, background=None):
        super().__init__()
        if batch_ms % row_ms:
            raise ValueError("a batch is a whole number of rows")
        self.flow, self.rows, self.row_ms, self.batch_ms = flow, rows, row_ms, batch_ms
        self.latency_ms, self.fee_rate, self.tick = latency_ms, Decimal(fee_rate), tick
        self.max_position, self.with_duration, self.predictor = max_position, with_duration, predictor
        self.day_seed = seed
        self.background = background or draw_background(flow, seed, rows, row_ms)
        width = 6 if with_duration else 5
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(width,), dtype=np.float32)
        features = 9 + (1 if predictor is not None else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(features,), dtype=np.float32)

    # The state of a day.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.row = 0
        self.next_order = 0
        self.ids = 0
        self.resting = []
        self.reference = self.flow.reference_ticks
        self.last_volume = 0
        self.last_imbalance = 0
        self.position = 0
        self.cash = Decimal(0)
        self.fees = Decimal(0)
        self.fills = []
        self.batches = []
        # At each close: the position and the marked equity after the clearing, where both
        # change; a step of several rows spans several closes.
        self.marks = []
        self.late = {"orders": 0, "cancels": 0}
        self.net = Decimal(0)
        return self._observe(), self._info()

    def _mark(self):
        return self.cash + Decimal(self.position) * Decimal(self.reference) * self.tick

    def market_features(self):
        signal = float(self.background.signal[min(self.row, self.rows - 1)])
        rows_per_batch = self.batch_ms // self.row_ms
        left = rows_per_batch - (self.row % rows_per_batch)
        return np.asarray([left / rows_per_batch, (signal - self.reference) / 4.0,
                           self.last_volume / 10.0, self.last_imbalance / 10.0], dtype=np.float64)

    def _observe(self):
        live = [o for o in self.resting if o.cancel_ms is None]
        bid = next((o for o in live if o.side == 1), None)
        ask = next((o for o in live if o.side == -1), None)
        quote = [1.0 if live else 0.0,
                 0.0 if bid is None else (bid.price - self.reference) / 8.0,
                 0.0 if ask is None else (ask.price - self.reference) / 8.0,
                 self.position / self.max_position]
        out = list(self.market_features()) + [len(live) / 2.0] + quote
        if self.predictor is not None:
            out.append(float(self.predictor(self.market_features())))
        return np.asarray(out, dtype=np.float32)

    def _info(self):
        return {"row": self.row, "net": str(self.net), "position": self.position,
                "fees": str(self.fees), "reference": self.reference}

    def decode(self, action):
        """The quote an action asks for: `None` to hold none, or the offsets, sizes and duration."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        def pick(value, choices):
            return choices[min(int((value + 1.0) / 2.0 * len(choices)), len(choices) - 1)]

        duration = pick(a[5], DURATIONS) if self.with_duration else 1
        if a[0] < 0:
            return None, duration
        return (pick(a[1], OFFSETS), pick(a[2], OFFSETS), pick(a[3], SIZES), pick(a[4], SIZES)), duration

    def _replace(self, quote, now):
        arrival = now + self.latency_ms
        late = fba.batch_of(arrival, self.batch_ms) != fba.batch_of(now, self.batch_ms)
        for o in self.resting:
            if o.cancel_ms is None:
                o.cancel_ms = arrival
                self.late["cancels"] += late
        if quote is None:
            return
        bid_off, ask_off, bid_lots, ask_lots = quote
        centre = int(round(float(self.background.signal[self.row])))
        for side, offset, lots in ((1, bid_off, bid_lots), (-1, ask_off, ask_lots)):
            # Never quote a size that could take the position past its limit.
            lots = min(lots, self.max_position - side * self.position)
            if lots <= 0:
                continue
            self.ids += 1
            self.late["orders"] += late
            self.resting.append(Resting(-self.ids, side, centre - side * (offset + 1), lots, arrival))

    def _clear(self, close):
        """Clears the batch that closes at `close`, with every background order that arrived in
        it and every agent order live at the close."""
        orders, start = [], close - self.batch_ms
        index = self.next_order
        background = self.background.orders
        while index < len(background) and background[index][0] < close:
            resolved = resolve(background[index], self.reference, self.flow)
            if resolved is not None and resolved[0] >= start:
                arrival, side, price, lots = resolved
                orders.append(fba.Order(index, side, price, lots, arrival))
            index += 1
        self.next_order = index
        live = [o for o in self.resting if o.arrival_ms < close
                and (o.cancel_ms is None or o.cancel_ms >= close)]
        orders += [fba.Order(o.order_id, o.side, o.price, o.lots, o.arrival_ms, "agent")
                   for o in live]
        out = fba.clear(orders, self.reference)
        for o in live:
            lots = out.fills.get(o.order_id, 0)
            if lots:
                notional = Decimal(lots) * Decimal(out.price) * self.tick
                fee = self.fee_rate * notional
                self.cash -= o.side * notional + fee
                self.fees += fee
                self.position += o.side * lots
                o.lots -= lots
                self.fills.append((close, o.side, out.price, lots))
        self.resting = [o for o in self.resting if o.lots > 0
                        and (o.cancel_ms is None or o.cancel_ms >= close)]
        if out.price is not None:
            self.reference = out.price
        self.last_volume, self.last_imbalance = out.volume, out.imbalance
        self.batches.append((close, out.price, out.volume))
        self.marks.append((close, self.position, self._mark()))

    def _advance_row(self):
        """Ends the current row: clears the batch if the row's end is a close."""
        end = (self.row + 1) * self.row_ms
        if end % self.batch_ms == 0:
            self._clear(end)
        self.row += 1

    def step(self, action):
        quote, duration = self.decode(action)
        before = self._mark()
        self._replace(quote, self.row * self.row_ms)
        rows = 0
        while rows < duration and self.row < self.rows:
            self._advance_row()
            rows += 1
        self.net = self._mark()
        terminated = self.row >= self.rows
        info = self._info() | {"rows": rows, "duration": duration, "reward_exact": str(self.net - before)}
        return self._observe(), float(self.net - before), terminated, False, info


def avellaneda_stoikov(env, risk_aversion, intensity, volatility, horizon_rows):
    """The Avellaneda-Stoikov quote at the agent's state, on the action grid, in price units as
    the crate takes it: the reservation price `s - q gamma sigma^2 tau` and the half spread
    `gamma sigma^2 tau / 2 + ln(1 + gamma / k) / gamma`, with `sigma` in price units per square
    root of a row and `tau` the rows left; then in ticks from the signal, the offsets nearest on
    the grid (an offset `o` quotes `o + 1` ticks away). Returns the action that asks for it."""
    tau = max(horizon_rows - env.row, 0)
    variance = volatility ** 2 * tau
    tick = float(env.tick)
    reservation = -env.position * risk_aversion * variance / tick
    half = (risk_aversion * variance / 2.0
            + math.log(1.0 + risk_aversion / intensity) / risk_aversion) / tick
    bid_away = max(half - reservation - 1.0, 0.0)
    ask_away = max(half + reservation - 1.0, 0.0)

    def code(away):
        index = int(np.argmin([abs(o - away) for o in OFFSETS]))
        return -1.0 + (2.0 * index + 1.0) / len(OFFSETS)

    action = [1.0, code(bid_away), code(ask_away), -0.5, -0.5]
    if env.with_duration:
        action.append(-0.75)
    return np.asarray(action, dtype=np.float32)
