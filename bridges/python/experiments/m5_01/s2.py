"""The metrics and stop rule S2 of #106's main study (`design.json`, stop_rules). REAL DATA when
given the capture's windows and simulations; the tests use made-up values only.

A trajectory here is a dict: `samples` [ms, bid, ask, spread], `events` [ms, side, depth, kind,
queue_before, waiting], `references` [ms, moved, reinitialized] and `directions`, one sign per
reference, all with times from the window's start. Each statistic is per 10 min block.
"""

import numpy as np

BLOCK_MS, BLOCKS, LIMITS = 600_000, 12, 3
SIZE_EDGES = (1, 2, 5, 17, 65)
IMBALANCE_EDGES = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)
MARGIN, SEED, REPLICATES, MEAN_BLOCK = 0.02, 106, 2000, 2.0
PAIRS = ("qr", "qr_count", "qr_constant")
QR_ARMS = ("qr", "qr_count")
METRICS = ("correlation", "kind", "direction", "gamma", "cap")


def size_bucket(queue):
    """The design's buckets 1, 2-4, 5-16, 17-64, 65+; a queue of 0 goes with the first."""
    return max(int(np.searchsorted(SIZE_EDGES, queue, side="right")) - 1, 0)


def imbalance_bucket(bid, ask):
    """The touch imbalance's bucket, the last closed; None for an empty touch."""
    if bid + ask == 0:
        return None
    value = (bid - ask) / (bid + ask)
    return min(int(np.searchsorted(IMBALANCE_EDGES, value, side="right")) - 1, 4)


def from_window(window):
    """A capture window as a trajectory."""
    start = window["window"]["start_ms"]
    sides = {"bid": 0, "ask": 1}
    kinds = {"limit": 0, "cancel": 1, "market": 2}
    return {"samples": [[s["timestamp_ms"] - start, s["bid"], s["ask"], s["spread_ticks"]]
                        for s in window["samples"]],
            "events": [[o["timestamp_ms"] - start, sides[o["side"]], o["depth"], kinds[o["kind"]],
                        o["queue_before"], o["waiting_seconds"]]
                       for o in window["observations"] if o["depth"] < LIMITS],
            "references": [[r["timestamp_ms"] - start, r["moved"], r["reinitialized"]]
                           for r in window["references"]],
            "directions": window["reference_directions"]}


def per_block(trajectory):
    """Splits a trajectory into its 12 blocks."""
    out = [{"samples": [], "events": [], "moves": []} for _ in range(BLOCKS)]
    for sample in trajectory["samples"]:
        if 0 <= sample[0] // BLOCK_MS < BLOCKS:
            out[sample[0] // BLOCK_MS]["samples"].append(sample)
    for event in trajectory["events"]:
        if 0 <= event[0] // BLOCK_MS < BLOCKS:
            out[event[0] // BLOCK_MS]["events"].append(event)
    samples = trajectory["samples"]
    times = [s[0] for s in samples]
    for reference, sign in zip(trajectory["references"], trajectory["directions"], strict=True):
        if not reference[1] or not 0 <= reference[0] // BLOCK_MS < BLOCKS:
            continue
        at = int(np.searchsorted(times, reference[0], side="left")) - 1
        if at >= 0:
            bucket = imbalance_bucket(samples[at][1][0], samples[at][2][0])
            if bucket is not None:
                out[reference[0] // BLOCK_MS]["moves"].append((bucket, sign))
    return out


def _tv(a, b):
    keys = set(a) | set(b)
    total_a, total_b = sum(a.values()), sum(b.values())
    if total_a == 0 or total_b == 0:
        return np.nan
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b) for k in keys)


def _counts(values):
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _conditional_tv(data, model):
    """Mean over the data's conditions of the per-condition TV, weighted by the data's share.
    `data` and `model` map a condition to the outcomes' counts."""
    total = sum(sum(v.values()) for v in data.values())
    if total == 0:
        return np.nan
    out = 0.0
    for key, counts in data.items():
        if sum(model.get(key, {}).values()) == 0:
            out += sum(counts.values()) / total  # no model mass where the data has some: TV 1
            continue
        out += sum(counts.values()) / total * _tv(counts, model[key])
    return out


def _queues(block, limit):
    return [q for s in block["samples"] for q in (s[1][limit], s[2][limit])]


def _kinds(block):
    out = {}
    for _, _, depth, kind, queue, _ in block["events"]:
        table = out.setdefault((size_bucket(queue), depth), {})
        table[kind] = table.get(kind, 0) + 1
    return out


def _directions(block):
    out = {}
    for bucket, sign in block["moves"]:
        table = out.setdefault(bucket, {})
        table[sign] = table.get(sign, 0) + 1
    return out


def _correlation(block):
    if len(block["samples"]) < 2:
        return np.nan
    bid, ask = ([float(s[i][0]) for s in block["samples"]] for i in (1, 2))
    if np.std(bid) == 0.0 or np.std(ask) == 0.0:
        return np.nan
    return float(np.corrcoef(bid, ask)[0, 1])


def _gamma(values):
    """The gamma shape by moments, mean^2 / variance."""
    if not values or np.var(values) == 0.0:
        return np.nan
    return float(np.mean(values) ** 2 / np.var(values))


def block_metrics(data, model, cap):
    """One block's metrics, model against data; lower is better for all."""
    out = {f"queue_tv_{limit}": _tv(_counts(_queues(data, limit)), _counts(_queues(model, limit)))
           for limit in range(LIMITS)}
    out["correlation"] = abs(_correlation(model) - _correlation(data))
    out["kind"] = _conditional_tv(_kinds(data), _kinds(model))
    out["direction"] = _conditional_tv(_directions(data), _directions(model))
    out["gamma"] = float(np.mean([abs(_gamma(_queues(model, limit)) - _gamma(_queues(data, limit)))
                                  for limit in range(LIMITS)]))
    events = model["events"]
    out["cap"] = sum(e[4] >= cap for e in events) / len(events) if events else np.nan
    return out


def stationary_indices(n, rng):
    """One stationary bootstrap sample of indices 0 to n - 1, mean block `MEAN_BLOCK`."""
    out = [int(rng.integers(n))]
    while len(out) < n:
        out.append(int(rng.integers(n)) if rng.random() < 1.0 / MEAN_BLOCK else (out[-1] + 1) % n)
    return out


def interval(windows, rng=None, replicates=REPLICATES):
    """The 90 % percentile interval of the mean paired difference: each window resampled alone,
    windows weighted equally. `windows` holds one array of block differences per window, with
    the blocks undefined for any arm already removed."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    means = np.empty(replicates)
    for r in range(replicates):
        means[r] = np.mean([np.mean(w[stationary_indices(len(w), rng)]) for w in windows])
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def paired(mdqr, other):
    """Per-window block differences, dropping a block undefined on either, and the share lost."""
    windows, lost, total = [], 0, 0
    for a, b in zip(mdqr, other, strict=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        keep = np.isfinite(a) & np.isfinite(b)
        lost += int((~keep).sum())
        total += len(a)
        windows.append(a[keep] - b[keep])
    return windows, lost / total


def align_undefined(table):
    """A block undefined on any arm is dropped for every arm alike, as S2_undefined says."""
    out = {arm: {k: [list(w) for w in v] for k, v in metrics.items()}
           for arm, metrics in table.items()}
    for metric in next(iter(out.values())):
        for w in range(len(next(iter(out.values()))[metric])):
            for b in range(BLOCKS):
                if any(not np.isfinite(out[arm][metric][w][b]) for arm in out):
                    for arm in out:
                        out[arm][metric][w][b] = float("nan")
    return out


def decide(table, tau, failed_draws, undefined=None):
    """The design's S2. `table[arm][metric]` is one list of 12 block values per window; `tau`
    is the tau-b against each QR arm, or None if the transfer did not run."""
    primary, dependencies = {}, {}
    table = align_undefined(table)
    undefined = undefined or {}

    def read(metric, other):
        windows, lost = paired(table["mdqr"][metric], table[other][metric])
        lost = max(lost, undefined.get("mdqr", {}).get(metric, 0.0),
                   undefined.get(other, {}).get(metric, 0.0))
        if lost > 0.25 or any(len(w) == 0 for w in windows):
            return {"undefined_share": lost, "passes": False}
        low, high = interval(windows)
        return {"low": low, "high": high, "undefined_share": lost}

    for limit in range(LIMITS):
        for other in PAIRS:
            result = read(f"queue_tv_{limit}", other)
            result.setdefault("passes", result.get("high", 1.0) < 0.0)
            primary[f"{limit + 1}/{other}"] = result
    for metric in METRICS:
        for other in QR_ARMS:
            result = read(metric, other)
            result.setdefault("passes", result.get("high", 1.0) <= MARGIN)
            dependencies[f"{metric}/{other}"] = result
    transfer = tau is not None and all(tau[arm] >= 0.6 for arm in QR_ARMS)
    passes = (all(r["passes"] for r in primary.values())
              and all(r["passes"] for r in dependencies.values()) and transfer
              and failed_draws == 0)
    return {"primary": primary, "dependencies": dependencies, "tau": tau,
            "failed_draws": failed_draws,
            "decision": "adopt provisionally on this epoch" if passes else "discard"}
