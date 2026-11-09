"""MDQR-small for #106: the queue-reactive book of `src/backtest/queue_reactive.rs` with its
intensities from a neural network of the state (after Bodor and Carlier, arXiv
2501.08822v1, reimplemented from the text; no code was released with it).

The book is the simulator's, step for step: the same cells in the same order, the same
xorshift64* generator, the same reference move, shift with the event-size renormalization
and redraw. With the truth's own rate table as its intensities, `simulate` is the Rust
simulator event for event, which a test checks against `mdqr_qr truth`.

Everything here runs on CPU in float64, torch with 1 thread.
"""

import math
from fractions import Fraction

import numpy as np

RNG_MULTIPLIER = 0x2545F4914F6CDD1D
ZERO_SEED = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
UNIT_SCALE = 1.0 / 9_007_199_254_740_992.0
SIDES = ("bid", "ask")
KINDS = ("limit", "cancel", "market")
NO_KIND = 3


class Rng:
    """xorshift64*, as the simulator's `unit`."""

    def __init__(self, seed):
        self.state = ZERO_SEED if seed == 0 else seed & MASK

    def unit(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        self.state = x
        return (((x * RNG_MULTIPLIER) & MASK) >> 11) * UNIT_SCALE


class Mechanics:
    """What the book does besides its intensities: theta, theta_reinit, the invariant laws,
    the event sizes and the cap, per limit."""

    def __init__(self, theta, theta_reinit, invariant, event_sizes, max_queue):
        self.theta = float(theta)
        self.theta_reinit = float(theta_reinit)
        self.invariant = [list(map(float, law)) for law in invariant]
        self.event_sizes = [Fraction(str(s)) for s in event_sizes]
        self.max_queue = int(max_queue)
        self.limits = len(self.invariant)
        if len(self.event_sizes) != self.limits:
            raise ValueError("one event size a limit")

    def to_json(self):
        return {"theta": self.theta, "theta_reinit": self.theta_reinit,
                "invariant": self.invariant, "event_sizes": [str(s) for s in self.event_sizes],
                "max_queue": self.max_queue}

    @classmethod
    def from_json(cls, value):
        return cls(value["theta"], value["theta_reinit"], value["invariant"],
                   value["event_sizes"], value["max_queue"])


def allowed(queue, kind, max_queue):
    """Whether an event of `kind` can happen at a queue of `queue`."""
    if kind == 0:
        return queue < max_queue
    return queue > 0


class Book:
    """The simulator's state and step, with the intensities given by `rates(book)`, a list
    of `(side, depth, kind, rate)` in the simulator's order."""

    def __init__(self, mechanics, seed, reference_ticks):
        self.m = mechanics
        k = mechanics.limits
        self.queues = [[0] * k, [0] * k]
        self.reference_ticks = reference_ticks
        self.time = 0.0
        self.clock = 0.0
        self.last = [[0.0] * k, [0.0] * k]
        self.last_kind = [[NO_KIND] * k, [NO_KIND] * k]
        self.rng = Rng(seed)
        self.redraw()

    def unit(self):
        return self.rng.unit()

    def draw_invariant(self, depth):
        pick = self.unit()
        last = 0
        for n, p in enumerate(self.m.invariant[depth]):
            if p > 0.0:
                last = n
                pick -= p
                if pick < 0.0:
                    break
        return last

    def redraw(self):
        for depth in range(self.m.limits):
            self.queues[0][depth] = self.draw_invariant(depth)
            self.queues[1][depth] = self.draw_invariant(depth)

    def best_offset(self, side):
        for depth, q in enumerate(self.queues[side]):
            if q > 0:
                return depth
        return self.m.limits

    def spread_ticks(self):
        return self.best_offset(0) + self.best_offset(1) + 1

    def mid_half_ticks(self):
        return self.best_offset(1) - self.best_offset(0)

    def renormalized(self, queue, frm, to):
        value = Fraction(queue) * self.m.event_sizes[frm] / self.m.event_sizes[to]
        return min(math.ceil(value), self.m.max_queue)

    def shift(self, up):
        k = self.m.limits
        frm, to = (1, 0) if up else (0, 1)
        old_from, old_to = list(self.queues[frm]), list(self.queues[to])
        new_from = [0] * k
        new_to = [0] * k
        for depth in range(k):
            nxt = depth + 1
            new_from[depth] = (self.renormalized(old_from[nxt], nxt, depth) if nxt < k
                               else self.draw_invariant(depth))
        new_to[0] = self.renormalized(old_from[0], 0, 0)
        for depth in range(1, k):
            new_to[depth] = self.renormalized(old_to[depth - 1], depth - 1, depth)
        self.queues[frm] = new_from
        self.queues[to] = new_to

    def reference_move(self, before, after):
        up = after > before and self.queues[1][0] == 0
        down = after < before and self.queues[0][0] == 0
        if not up and not down:
            return None
        moved = self.unit() < self.m.theta
        reinitialized = False
        if moved:
            self.shift(up)
            self.reference_ticks += 1 if up else -1
            if self.unit() < self.m.theta_reinit:
                self.redraw()
                reinitialized = True
            k = self.m.limits
            self.last = [[self.time] * k, [self.time] * k]
            self.last_kind = [[NO_KIND] * k, [NO_KIND] * k]
        return moved, reinitialized

    def step(self, rates):
        """One event, or None if nothing can happen. Returns `(side, depth, kind, queue
        before, waiting seconds, reference outcome)`."""
        cells = [c for c in rates(self) if c[3] > 0.0]
        total = 0.0
        for c in cells:
            total += c[3]
        if total <= 0.0:
            return None
        waited = -math.log(1.0 - self.unit()) / total
        pick = self.unit() * total
        chosen = cells[-1]
        for cell in cells:
            pick -= cell[3]
            if pick < 0.0:
                chosen = cell
                break
        side, depth, kind, _ = chosen
        now = self.time + waited
        if now <= self.time:
            return None
        self.time = now
        self.clock = max(self.clock, now)
        before = self.queues[side][depth]
        waiting = now - self.last[side][depth]
        mid_before = self.mid_half_ticks()
        self.queues[side][depth] = before + 1 if kind == 0 else before - 1
        if self.queues[side][depth] < 0:
            raise ValueError("a queue went below zero")
        self.last[side][depth] = now
        self.last_kind[side][depth] = kind
        reference = self.reference_move(mid_before, self.mid_half_ticks())
        return side, depth, kind, before, waiting, reference


def table_rates(table, max_queue):
    """The truth's intensities: `table[depth][n] = (limit, cancel, market)`, as the
    simulator's `rates_at` with every size supported."""

    def rates(book):
        out = []
        for side in (0, 1):
            for depth in range(book.m.limits):
                n = book.queues[side][depth]
                row = table[depth][n]
                for kind in range(3):
                    out.append((side, depth, kind, float(row[kind])))
        return out

    return rates


def milliseconds(seconds):
    return int(math.floor(seconds * 1000.0))


def simulate(mechanics, rates, seed, reference_ticks, horizon, sample_s, max_events):
    """A run as the simulator's `run_limited`, in the `.trajectory.json` format."""
    book = Book(mechanics, seed, reference_ticks)
    k = mechanics.limits
    samples, events, references = [], [], []
    next_sample = book.clock + sample_s
    end = book.clock + horizon
    stopped = None
    while True:
        if len(events) >= max_events:
            stopped = "EventLimit"
            break
        before_queues = [list(book.queues[0]), list(book.queues[1])]
        before_spread = book.spread_ticks()
        snapshot = book_state(book)
        result = book.step(rates)
        if result is None:
            stopped = "Stalled"
            break
        while next_sample <= book.time and next_sample <= end:
            samples.append([milliseconds(next_sample), before_queues[0], before_queues[1],
                            before_spread])
            next_sample += sample_s
        if book.time > end:
            restore(book, snapshot)
            book.clock = end
            break
        side, depth, kind, before, waiting, reference = result
        ms = milliseconds(book.time)
        events.append([ms, side, depth, kind, before, waiting])
        if reference is not None:
            references.append([ms, reference[0], reference[1]])
    return {"limits": k, "samples": samples, "events": events, "references": references,
            "stopped": stopped}


def book_state(book):
    return (tuple(map(tuple, book.queues)), book.reference_ticks, book.time, book.clock,
            tuple(map(tuple, book.last)), tuple(map(tuple, book.last_kind)),
            book.rng.state)


def restore(book, state):
    (queues, book.reference_ticks, book.time, book.clock, last, last_kind,
     book.rng.state) = state
    book.queues = [list(q) for q in queues]
    book.last = [list(t) for t in last]
    book.last_kind = [list(k) for k in last_kind]


# The state the network reads.

def feature_count(limits):
    queues = 2 * limits
    return queues + 1 + queues * 4 + queues


def features(queues, spread, last_kind, since):
    """`queues[side][depth]`, the spread in ticks, the kind of the last event at each queue
    (3 for none) and the seconds since it, as of the previous event."""
    flat_q = [q for side in queues for q in side]
    flat_k = [k for side in last_kind for k in side]
    flat_s = [s for side in since for s in side]
    one_hot = []
    for k in flat_k:
        one_hot.extend(1.0 if k == i else 0.0 for i in range(4))
    return ([math.log1p(q) for q in flat_q] + [float(spread)] + one_hot
            + [math.log1p(max(s, 0.0)) for s in flat_s])


def training_rows(record, limits, max_queue):
    """From an `.events.json` record: for each event, the state as of the previous event,
    the time since it, the index of the event's cell, and the mask of allowed cells.
    The cell index is `side * limits * 3 + depth * 3 + kind`."""
    events = record["events"]
    moves = {}
    for t, moved, _ in record["references"]:
        moves[t] = moved
    k = limits
    last = [[0.0] * k, [0.0] * k]
    last_kind = [[NO_KIND] * k, [NO_KIND] * k]
    previous = 0.0
    queues = [list(record["start"]["bid"]), list(record["start"]["ask"])]
    x, dt, cell, mask = [], [], [], []
    for t, side, depth, kind, bid, ask in events:
        before = [list(bid), list(ask)]
        # The book before this event is the book after the previous one: the record's.
        queues = before
        spread = offset(queues[0], k) + offset(queues[1], k) + 1
        since = [[previous - s for s in last[i]] for i in (0, 1)]
        x.append(features(queues, spread, last_kind, since))
        dt.append(t - previous)
        cell.append(side * k * 3 + depth * 3 + kind)
        mask.append([allowed(queues[s][d], kd, max_queue)
                     for s in (0, 1) for d in range(k) for kd in range(3)])
        last[side][depth] = t
        last_kind[side][depth] = kind
        if moves.get(t):
            last = [[t] * k, [t] * k]
            last_kind = [[NO_KIND] * k, [NO_KIND] * k]
        previous = t
    return (np.asarray(x, dtype=np.float64), np.asarray(dt, dtype=np.float64),
            np.asarray(cell, dtype=np.int64), np.asarray(mask, dtype=bool))


def offset(queues, limits):
    for depth, q in enumerate(queues):
        if q > 0:
            return depth
    return limits


def estimate_mechanics(trajectories, records, event_sizes):
    """theta, theta_reinit and the invariant laws from the training windows: the share of
    eligible moves that moved, of moves that redrew, and the queue sizes at the samples,
    per limit, sides pooled."""
    eligible = sum(len(r["references"]) for r in records)
    moved = sum(1 for r in records for _, m, _ in r["references"] if m)
    redrawn = sum(1 for r in records for _, m, re in r["references"] if m and re)
    if eligible == 0 or moved == 0:
        raise ValueError("no reference moves to estimate theta from")
    limits = trajectories[0]["limits"]
    top = 0
    counts = [dict() for _ in range(limits)]
    for trajectory in trajectories:
        for _, bid, ask, _ in trajectory["samples"]:
            for depth in range(limits):
                for q in (bid[depth], ask[depth]):
                    counts[depth][q] = counts[depth].get(q, 0) + 1
                    top = max(top, q)
    invariant = []
    for depth in range(limits):
        total = sum(counts[depth].values())
        invariant.append([counts[depth].get(n, 0) / total for n in range(top + 1)])
    return Mechanics(moved / eligible, redrawn / moved, invariant, event_sizes, top)


class Network:
    """The intensity network as numpy, for the simulation: tanh layers, then the log
    intensities of every cell."""

    def __init__(self, weights, limits, max_queue, mean, scale):
        self.weights = weights
        self.limits = limits
        self.max_queue = max_queue
        self.mean = mean
        self.scale = scale

    def log_intensities(self, x):
        h = (np.asarray(x) - self.mean) / self.scale
        for i, (w, b) in enumerate(self.weights):
            h = h @ w.T + b
            if i < len(self.weights) - 1:
                h = np.tanh(h)
        return h

    def rates(self, book):
        k = self.limits
        x = features(book.queues, book.spread_ticks(), book.last_kind,
                     [[book.time - t for t in book.last[s]] for s in (0, 1)])
        log_rates = self.log_intensities(x)
        out = []
        i = 0
        for side in (0, 1):
            for depth in range(k):
                q = book.queues[side][depth]
                for kind in range(3):
                    rate = math.exp(float(log_rates[i])) if allowed(q, kind, self.max_queue) else 0.0
                    out.append((side, depth, kind, rate))
                    i += 1
        return out


def fit(rows, limits, max_queue, seed, hidden=(128, 64), lr=0.001, batch=4096, epochs=8):
    """Maximum likelihood of the marked point process with piecewise constant intensities:
    the sum over events of the log intensity of the event's cell, less the total allowed
    intensity times the time since the previous event."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x, dt, cell, mask = rows
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    xs = torch.tensor((x - mean) / scale, dtype=torch.float64)
    dts = torch.tensor(dt, dtype=torch.float64)
    cells = torch.tensor(cell)
    masks = torch.tensor(mask)
    layers = []
    width = x.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(width, h, dtype=torch.float64), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, 2 * limits * 3, dtype=torch.float64))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = xs.shape[0]
    history = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            log_rates = net(xs[idx])
            rates = torch.exp(log_rates) * masks[idx]
            loglik = (log_rates.gather(1, cells[idx][:, None]).squeeze(1)
                      - rates.sum(dim=1) * dts[idx])
            loss = -loglik.sum() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * n
        history.append(-total / n)
    weights = [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy())
               for m in net if isinstance(m, torch.nn.Linear)]
    return Network(weights, limits, max_queue, mean, scale), history
