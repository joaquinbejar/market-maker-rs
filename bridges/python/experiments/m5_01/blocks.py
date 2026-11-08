"""MDQR-small's block clock and size head for #106's main study (`design.json`, main_study
clock, state, fit and size_head). REAL DATA when given the capture's windows; the tests use
made-up windows only.

Events at one timestamp form a block. A network reads the state and gives the log block rate
and the logits of the next event (18 cells) or of the end of the block; a second network reads
the state and the event's cell and gives its size class (1 to 63, or the tail 64+).

Three rules the design left open, fixed on the training windows only (`design.json` changes):
a size is `max(1, round(size))` event sizes, as the replay rounds a queue; the feasibility
mask is applied in simulation and not in the likelihood, because the data has cancels and
trades at queues that round to zero; the state before an event is tracked from the past only
(the last book sample strictly before it and earlier events), never from the event's own
`queue_before`, which would reveal its cell.
"""

import bisect
import math
from collections import deque

import numpy as np

from mdqr import KINDS, NO_KIND, SIDES, Book, offset

LIMITS = 3
CELLS = 2 * LIMITS * 3
END = CELLS
SIZE_CLASSES = 64
TAIL = SIZE_CLASSES - 1
BLOCK_BUDGET = 2048
HORIZONS = (20.0, 60.0, 300.0, 900.0)
NS = 1e9


def feature_count():
    return 6 + 1 + len(HORIZONS) + 6 * 4 + 6 + 2


def size_class(size):
    """The class of a size in event sizes: 0 is one event size, TAIL is 64 or more."""
    units = max(1, int(math.floor(size + 0.5)))
    return min(units, SIZE_CLASSES) - 1


def cell_of(side, depth, kind):
    return side * LIMITS * 3 + depth * 3 + kind


def uncell(cell):
    return cell // (LIMITS * 3), (cell % (LIMITS * 3)) // 3, cell % 3


def hour_features(seconds):
    angle = 2.0 * math.pi * (seconds % 86400.0) / 86400.0
    return [math.sin(angle), math.cos(angle)]


def state_features(queues, last_kind, since, imbalance, seconds):
    """The design's state: 6 queues, spread, 4 imbalances, last kinds, times since, hour."""
    flat_q = [q for side in queues for q in side]
    one_hot = []
    for side in last_kind:
        for k in side:
            one_hot.extend(1.0 if k == i else 0.0 for i in range(4))
    spread = offset(queues[0], LIMITS) + offset(queues[1], LIMITS) + 1
    return ([math.log1p(q) for q in flat_q] + [float(spread)] + list(imbalance) + one_hot
            + [math.log1p(max(s, 0.0)) for side in since for s in side]
            + hour_features(seconds))


class Imbalance:
    """Signed trade imbalance (buys less sells over their sum, 0 if none) per horizon, from
    trades added in time order."""

    def __init__(self):
        self.trades = deque()

    def add(self, seconds, signed):
        self.trades.append((seconds, signed))

    def at(self, seconds):
        while self.trades and self.trades[0][0] < seconds - HORIZONS[-1]:
            self.trades.popleft()
        out = []
        for horizon in HORIZONS:
            net = gross = 0.0
            for t, s in reversed(self.trades):
                if t < seconds - horizon:
                    break
                net += s
                gross += abs(s)
            out.append(net / gross if gross > 0.0 else 0.0)
        return out


class WindowImbalance:
    """`Imbalance` for a recorded series: prefix sums, trades strictly before a time."""

    def __init__(self, trades):
        rows = sorted((t["timestamp_ns"], float(t["size"]) * (1 if t["side"] == "buy" else -1))
                      for t in trades)
        self.times = [r[0] for r in rows]
        self.net = [0.0]
        self.gross = [0.0]
        for _, s in rows:
            self.net.append(self.net[-1] + s)
            self.gross.append(self.gross[-1] + abs(s))

    def before(self, ns, inclusive=False):
        find = bisect.bisect_right if inclusive else bisect.bisect_left
        end = find(self.times, ns)
        out = []
        for horizon in HORIZONS:
            start = bisect.bisect_left(self.times, ns - int(horizon * NS))
            gross = self.gross[end] - self.gross[start] if end > start else 0.0
            out.append((self.net[end] - self.net[start]) / gross if gross > 0.0 else 0.0)
        return out


def rows_of(window):
    """A captured window as training rows. `wait` has one row a block: the state after the
    previous block at that block's time, and the wait. `seq` has one row per event and per end
    of block: the state before it at the block's time, its target, its block. `size` has the
    state and cell of each event and its class."""
    start_ns = window["window"]["start_ms"] * 1_000_000
    trades = WindowImbalance(window["history"]["trades"] + window["trades"])
    samples = [window["history"]["last_sample"]] + window["samples"]
    sample_ms = [s["timestamp_ms"] for s in samples]
    side_of = {name: i for i, name in enumerate(SIDES)}
    kind_of = {name: i for i, name in enumerate(KINDS)}
    moves = sorted(r["timestamp_ms"] for r in window["references"] if r["moved"])
    queues = [list(samples[0]["bid"][:LIMITS]), list(samples[0]["ask"][:LIMITS])]
    # History: the last event at each queue before the window.
    last = [[None] * LIMITS for _ in SIDES]
    last_kind = [[NO_KIND] * LIMITS for _ in SIDES]
    for o in window["history"]["observations"]:
        if o["depth"] < LIMITS:
            s, d = side_of[o["side"]], o["depth"]
            last[s][d] = o["timestamp_ns"]
            last_kind[s][d] = kind_of[o["kind"]]
    first_move = 0
    for r in window["history"]["references"]:
        if r["moved"]:
            for s in range(2):
                for d in range(LIMITS):
                    if last[s][d] is not None and last[s][d] <= r["timestamp_ms"] * 1_000_000:
                        last[s][d] = r["timestamp_ms"] * 1_000_000
                        last_kind[s][d] = NO_KIND

    def since(at):
        return [[(at - (start_ns - int(HORIZONS[-1] * NS) if t is None else t)) / NS for t in side]
                for side in last]

    wait_x, wait_dt, seq_x, seq_y, seq_block, size_x, size_c = [], [], [], [], [], [], []
    previous = start_ns
    observations = window["observations"]
    i, block = 0, 0
    next_sample = 0
    while i < len(observations):
        now = observations[i]["timestamp_ns"]
        wait_x.append(state_features(queues, last_kind, since(previous),
                                     trades.before(previous, inclusive=previous != start_ns),
                                     previous / NS))
        wait_dt.append((now - previous) / NS)
        now_ms = now // 1_000_000
        # Book samples strictly before the block resynchronize the tracked queues.
        at = bisect.bisect_left(sample_ms, now_ms) - 1
        if at >= next_sample:
            queues = [list(samples[at]["bid"][:LIMITS]), list(samples[at]["ask"][:LIMITS])]
            next_sample = at + 1
        while first_move < len(moves) and moves[first_move] < now_ms:
            moved_ns = moves[first_move] * 1_000_000
            last = [[moved_ns] * LIMITS for _ in SIDES]
            last_kind = [[NO_KIND] * LIMITS for _ in SIDES]
            first_move += 1
        imbalance = trades.before(now)
        seconds = now / NS
        while i < len(observations) and observations[i]["timestamp_ns"] == now:
            o = observations[i]
            x = state_features(queues, last_kind, since(now), imbalance, seconds)
            s, d, k = side_of[o["side"]], o["depth"], kind_of[o["kind"]]
            c = cell_of(s, d, k)
            seq_x.append(x)
            seq_y.append(c)
            seq_block.append(block)
            size_x.append(x)
            cls = size_class(o["size"])
            size_c.append((c, cls))
            units = cls + 1
            queues[s][d] = (o["queue_before"] + units if k == 0
                            else max(o["queue_before"] - units, 0))
            last[s][d] = now
            last_kind[s][d] = k
            i += 1
        seq_x.append(state_features(queues, last_kind, since(now), imbalance, seconds))
        seq_y.append(END)
        seq_block.append(block)
        previous = now
        block += 1
    return {"wait_x": np.asarray(wait_x), "wait_dt": np.asarray(wait_dt),
            "seq_x": np.asarray(seq_x), "seq_y": np.asarray(seq_y, dtype=np.int64),
            "seq_block": np.asarray(seq_block, dtype=np.int64),
            "size_x": np.asarray(size_x), "size_cell": np.asarray([c for c, _ in size_c], dtype=np.int64),
            "size_class": np.asarray([k for _, k in size_c], dtype=np.int64)}


def join(parts):
    """Rows of several windows, block numbers kept distinct."""
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != "seq_block"}
    offset_blocks, blocks = 0, []
    for p in parts:
        blocks.append(p["seq_block"] + offset_blocks)
        offset_blocks += len(p["wait_dt"])
    out["seq_block"] = np.concatenate(blocks)
    return out


def tails_of(windows):
    """The training sizes of 64 or more, per cell, in event sizes."""
    side_of = {name: i for i, name in enumerate(SIDES)}
    kind_of = {name: i for i, name in enumerate(KINDS)}
    tails = {}
    for window in windows:
        for o in window["observations"]:
            if size_class(o["size"]) == TAIL:
                key = cell_of(side_of[o["side"]], o["depth"], kind_of[o["kind"]])
                tails.setdefault(key, []).append(max(64, int(math.floor(o["size"] + 0.5))))
    return {key: sorted(v) for key, v in tails.items()}


def _mlp(width, outputs, torch, hidden=(128, 64)):
    layers = []
    for h in hidden:
        layers += [torch.nn.Linear(width, h, dtype=torch.float64), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, outputs, dtype=torch.float64))
    return torch.nn.Sequential(*layers)


def _nll(event_net, size_net, rows, tensors, torch, blocks=None):
    """Summed negative log-likelihood of the blocks in `blocks` (all if None), and its count
    of blocks. Within-block logits are normalized over the 19 outcomes."""
    wait_x, wait_dt, seq_x, seq_y, seq_block, size_x, size_c, size_seq, tail_mask = tensors
    if blocks is None:
        w_idx, s_idx, z_idx = None, None, None
    else:
        w_idx = blocks
        chosen = torch.zeros(wait_x.shape[0], dtype=torch.bool)
        chosen[blocks] = True
        s_idx = chosen[seq_block]
        z_idx = chosen[size_seq]
    def pick(t, i):
        return t if i is None else t[i]
    log_rate = event_net(pick(wait_x, w_idx))[:, 0]
    wait = (log_rate - torch.exp(log_rate) * pick(wait_dt, w_idx)).sum()
    logits = event_net(pick(seq_x, s_idx))[:, 1:]
    seq = torch.log_softmax(logits, dim=1).gather(1, pick(seq_y, s_idx)[:, None]).sum()
    size_logits = size_net(pick(size_x, z_idx)).masked_fill(pick(tail_mask, z_idx), -1e30)
    size = torch.log_softmax(size_logits, dim=1).gather(1, pick(size_c, z_idx)[:, None]).sum()
    return -(wait + seq + size)


def _tensors(rows, mean, scale, tails, torch):
    def standard(x):
        return torch.tensor((x - mean) / scale, dtype=torch.float64)

    one_hot = torch.nn.functional.one_hot(torch.tensor(rows["size_cell"]), CELLS).to(torch.float64)
    mask = torch.zeros(len(rows["size_cell"]), SIZE_CLASSES, dtype=torch.bool)
    mask[:, TAIL] = torch.tensor([int(c) not in tails for c in rows["size_cell"]])
    # An event's position in the block's sequence is not needed; its block is.
    seq_block = torch.tensor(rows["seq_block"])
    size_seq = seq_block[torch.tensor(rows["seq_y"]) != END]
    return (standard(rows["wait_x"]), torch.tensor(rows["wait_dt"], dtype=torch.float64),
            standard(rows["seq_x"]), torch.tensor(rows["seq_y"]), seq_block,
            torch.cat([standard(rows["size_x"]), one_hot], dim=1),
            torch.tensor(rows["size_class"]), size_seq, mask)


def fit(train, validation, tails, seed, lr=0.001, batch=4096, epochs=8):
    """Adam on minibatches of blocks; keeps the epoch with the best validation likelihood.
    Returns the model as a plain dict and the per-block log-likelihoods per epoch."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x = np.concatenate([train["wait_x"], train["seq_x"]])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0.0] = 1.0
    data = _tensors(train, mean, scale, tails, torch)
    held = _tensors(validation, mean, scale, tails, torch)
    width = train["wait_x"].shape[1]
    event_net = _mlp(width, 1 + CELLS + 1, torch)
    size_net = _mlp(width + CELLS, SIZE_CLASSES, torch)
    params = list(event_net.parameters()) + list(size_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = data[0].shape[0]
    history, best, best_ll = [], None, -math.inf
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            loss = _nll(event_net, size_net, train, data, torch, idx) / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        with torch.no_grad():
            ll = -float(_nll(event_net, size_net, validation, held, torch)) / held[0].shape[0]
        history.append({"epoch": epoch, "train": -total / n, "validation": ll})
        if not math.isfinite(ll):
            raise ValueError(f"the likelihood is not finite at epoch {epoch}")
        if ll > best_ll:
            best_ll = ll
            best = {"epoch": epoch, "event": _weights(event_net, torch),
                    "size": _weights(size_net, torch)}
    return {"mean": mean.tolist(), "scale": scale.tolist(), "epoch": best["epoch"],
            "event": best["event"], "size": best["size"],
            "tails": {str(k): v for k, v in tails.items()}}, history


def _weights(net, torch):
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy())
            for m in net if isinstance(m, torch.nn.Linear)]


def _forward(weights, h):
    for i, (w, b) in enumerate(weights):
        h = w @ h + b
        if i < len(weights) - 1:
            h = np.tanh(h)
    return h


class BudgetExceeded(Exception):
    """A block reached the event budget: a failed draw, never a truncation."""


class Generator:
    """Simulates the model on the QR book's mechanics, in sized events."""

    def __init__(self, model, mechanics, seed, start, history_events=(), history_trades=(),
                 start_seconds=0.0, event_size_btc=None):
        self.model = model
        self.m = mechanics
        self.book = Book(mechanics, seed, 0)
        self.book.queues = [list(start[0]), list(start[1])]
        self.book.time = start_seconds
        self.book.clock = start_seconds
        self.book.last = [[start_seconds - HORIZONS[-1]] * LIMITS for _ in SIDES]
        self.imbalance = Imbalance()
        for t, signed in history_trades:
            self.imbalance.add(t, signed)
        for t, s, d, k in history_events:
            self.book.last[s][d] = t
            self.book.last_kind[s][d] = k
        self.mean = np.asarray(model["mean"])
        self.scale = np.asarray(model["scale"])
        self.event_size_btc = event_size_btc or [1.0] * LIMITS
        self.tails = {int(k): v for k, v in model["tails"].items()}
        self.clipped_events = self.clipped_volume = self.volume = 0.0
        self.tail_draws = self.at_cap = self.events = 0
        self.directions = []

    def _x(self, imbalance):
        book = self.book
        since = [[book.time - t for t in side] for side in book.last]
        x = state_features(book.queues, book.last_kind, since, imbalance, book.time)
        return (np.asarray(x) - self.mean) / self.scale

    def _categorical(self, logits):
        weights = np.exp(logits - logits.max())
        pick = self.book.unit() * float(weights.sum())
        for i, w in enumerate(weights):
            pick -= float(w)
            if pick < 0.0 and w > 0.0:
                return i
        return int(np.flatnonzero(weights > 0.0)[-1])

    def _mask(self, first):
        book = self.book
        out = np.zeros(CELLS + 1, dtype=bool)
        for c in range(CELLS):
            s, d, k = uncell(c)
            q = book.queues[s][d]
            out[c] = q < self.m.max_queue if k == 0 else q > 0
        out[END] = not first
        return out

    def _size(self, x, cell):
        h = np.concatenate([x, np.eye(CELLS)[cell]])
        logits = _forward(self.model["size"], h)
        if cell not in self.tails:
            logits[TAIL] = -np.inf
        cls = self._categorical(logits)
        if cls == TAIL:
            self.tail_draws += 1
            sizes = self.tails[cell]
            return sizes[min(int(self.book.unit() * len(sizes)), len(sizes) - 1)]
        return cls + 1

    def _apply(self, cell, units):
        book = self.book
        s, d, k = uncell(cell)
        q = book.queues[s][d]
        mid_before = book.mid_half_ticks()
        if k == 0:
            after = min(q + units, self.m.max_queue)
            done = after - q
        else:
            done = min(units, q)
            after = q - done
            if units > q:
                self.clipped_events += 1
                self.clipped_volume += units - q
        self.volume += units
        book.queues[s][d] = after
        book.last[s][d] = book.time
        book.last_kind[s][d] = k
        self.events += 1
        if after >= self.m.max_queue:
            self.at_cap += 1
        ticks_before = book.reference_ticks
        reference = book.reference_move(mid_before, book.mid_half_ticks())
        if reference is not None:
            self.directions.append(book.reference_ticks - ticks_before)
        return s, d, k, q, done, reference

    def block(self):
        """One block: advances the clock and returns its events."""
        book = self.book
        x = self._x(self.imbalance.at(book.time))
        rate = math.exp(float(_forward(self.model["event"], x)[0]))
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("the block rate is not finite")
        book.time += -math.log(1.0 - book.unit()) / rate
        book.clock = book.time
        imbalance = self.imbalance.at(book.time)
        out, trades = [], []
        first = True
        while True:
            x = self._x(imbalance)
            logits = _forward(self.model["event"], x)[1:]
            if not np.all(np.isfinite(logits)):
                raise ValueError("the event logits are not finite")
            logits[~self._mask(first)] = -np.inf
            choice = self._categorical(logits)
            if choice == END:
                break
            if len(out) >= BLOCK_BUDGET:
                raise BudgetExceeded(f"a block reached {BLOCK_BUDGET} events")
            units = self._size(x, choice)
            event = self._apply(choice, units)
            out.append(event)
            if event[2] == 2 and event[4] > 0:
                btc = event[4] * self.event_size_btc[event[1]]
                trades.append(-btc if event[0] == 0 else btc)
            first = False
        for signed in trades:
            self.imbalance.add(book.time, signed)
        return out
