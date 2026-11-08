"""Stop rule S1 of #106's main study (`design.json`, main_study stop_rules). REAL DATA.

    python experiments/m5_01/evaluate.py s1 <capture dir> <fits dir> <out dir> <seed> <window>

Simulates 24 h from a validation window's starting book for one training seed, with
simulation seed 107000 + 100 w + seed, w being 1 or 2. Reads no holdout file. Writes
`s1-<seed>-<w>.json`; `s1-decide <out dir>` applies the rule to the 10 runs.

    python experiments/m5_01/evaluate.py claim-holdout <out dir>
    python experiments/m5_01/evaluate.py draws <capture dir> <fits dir> <out dir> <set> <seed>

`claim-holdout` creates the exclusive marker once, before any holdout window is read, and fails
if it exists; `draws` refuses the holdout without it. `draws` simulates MDQR's 10 draws per
window for one training seed and writes each draw's block metrics and 10 s book.
"""

import json
import math
import os
import pathlib
import pickle
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blocks  # noqa: E402
import mdqr  # noqa: E402
import s2  # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
VALIDATION = (1, 2)
S1_SECONDS = 86400.0
S1_CAP_SHARE = 0.01


def start_of(window):
    """The book before the window, and its history as the generator takes it: trades as
    (seconds, signed BTC) and the last event at each queue, a reference move resetting them
    as `blocks.rows_of` does."""
    side_of = {name: i for i, name in enumerate(mdqr.SIDES)}
    kind_of = {name: i for i, name in enumerate(mdqr.KINDS)}
    history = window["history"]
    sample = history["last_sample"]
    start = (list(sample["bid"][:blocks.LIMITS]), list(sample["ask"][:blocks.LIMITS]))
    last = {}
    for o in history["observations"]:
        if o["depth"] < blocks.LIMITS:
            last[(side_of[o["side"]], o["depth"])] = (o["timestamp_ns"] / blocks.NS,
                                                      kind_of[o["kind"]])
    for r in history["references"]:
        if r["moved"]:
            moved = r["timestamp_ms"] / 1000.0
            last = {key: (moved, mdqr.NO_KIND) if t <= moved else (t, k)
                    for key, (t, k) in last.items()}
    events = [(t, s, d, k) for (s, d), (t, k) in sorted(last.items())]
    trades = [(t["timestamp_ns"] / blocks.NS,
               float(t["size"]) * (1 if t["side"] == "buy" else -1))
              for t in sorted(history["trades"], key=lambda t: t["timestamp_ns"])]
    return start, events, trades


def s1_run(model, window, simulation_seed, seconds=S1_SECONDS, progress=None):
    """One S1 simulation. Returns the counts; a NaN or a budget hit is a recorded failure."""
    start, events, trades = start_of(window)
    begin = window["window"]["start_ms"] / 1000.0
    generator = blocks.Generator(model, mdqr.Mechanics.from_json(model["mechanics"]),
                                 simulation_seed, start, events, trades, begin,
                                 [float(s) for s in window["event_sizes"]])
    failure, count = None, 0
    try:
        while generator.book.time < begin + seconds:
            generator.block()
            count += 1
            if progress and count % 20000 == 0:
                progress(generator.book.time - begin, count, generator.events)
    except blocks.BudgetExceeded as error:
        failure = f"budget: {error}"
    except ValueError as error:
        failure = f"nan: {error}"
    share = generator.at_cap / generator.events if generator.events else 0.0
    return {"failure": failure, "blocks": count, "events": generator.events,
            "at_cap": generator.at_cap, "cap_share": share,
            "simulated_s": generator.book.time - begin,
            "clipped_events": generator.clipped_events, "tail_draws": generator.tail_draws}


def s1_failed(run):
    """Why a run discards MDQR, or None."""
    if run["failure"] is not None:
        return run["failure"]
    if not math.isfinite(run["cap_share"]) or run["cap_share"] > S1_CAP_SHARE:
        return f"cap share {run['cap_share']}"
    return None


def decide(runs):
    """S1: a single failure in any seed and window discards MDQR; no seed is dropped."""
    failures = {key: s1_failed(run) for key, run in runs.items() if s1_failed(run)}
    missing = [f"{s}-{w}" for s in SEEDS for w in VALIDATION if f"{s}-{w}" not in runs]
    if missing:
        raise ValueError(f"S1 needs all 10 runs, missing {missing}")
    return {"failures": failures, "decision": "discard MDQR" if failures else "pass"}


DRAW_SECONDS, DRAWS = 7200.0, 10
HOLDOUT_MARKER = "holdout.started"


def claim_holdout(out):
    """Creates the marker exclusively; a second claim raises FileExistsError."""
    out.mkdir(parents=True, exist_ok=True)
    os.close(os.open(out / HOLDOUT_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY))


def simulate_window(model, window, simulation_seed):
    """One draw of the window's 2 h as an `s2` trajectory, or raises on a failed draw."""
    start, events, trades = start_of(window)
    begin = window["window"]["start_ms"] / 1000.0
    g = blocks.Generator(model, mdqr.Mechanics.from_json(model["mechanics"]), simulation_seed,
                         start, events, trades, begin, [float(s) for s in window["event_sizes"]])
    end = begin + DRAW_SECONDS
    samples, rows, references, directions = [], [], [], []
    next_sample = begin + 1.0
    while True:
        book = ([list(q) for q in g.book.queues], g.book.spread_ticks())
        emitted = g.block()
        now = g.book.time
        while next_sample <= min(now, end):
            samples.append([int(round((next_sample - begin) * 1000)), book[0][0], book[0][1], book[1]])
            next_sample += 1.0
        if now > end:
            break
        ms = int((now - begin) * 1000)
        for s, d, k, q, _, reference in emitted:
            rows.append([ms, s, d, k, q, 0.0])
            if reference is not None:
                references.append([ms, reference[0], reference[1]])
        directions = g.directions[:len(references)]
    return {"samples": samples, "events": rows, "references": references, "directions": directions}


def score(data_blocks, trajectory, cap):
    """The 12 blocks' metrics of a draw, as one list per metric."""
    per = [s2.block_metrics(d, m, cap) for d, m in zip(data_blocks, s2.per_block(trajectory), strict=True)]
    return {key: [row[key] for row in per] for key in per[0]}


def run_draws(capture, fits, out, set_name, seed):
    count = {"validation": 2, "holdout": 3}[set_name]
    if set_name == "holdout" and not (out / HOLDOUT_MARKER).exists():
        raise SystemExit("the holdout is not claimed")
    with open(fits / f"model-{seed}.pkl", "rb") as f:
        model = pickle.load(f)
    cap = mdqr.Mechanics.from_json(model["mechanics"]).max_queue
    for w in range(1, count + 1):
        window = json.loads((capture / set_name / f"window-{w - 1}.json").read_text())
        data = s2.per_block(s2.from_window(window))
        for j in range(1, DRAWS + 1):
            path = out / set_name / f"mdqr-{seed}-w{w}-d{j}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if set_name == "holdout" and path.exists():
                raise SystemExit(f"{path} exists: the holdout runs once")
            try:
                trajectory = simulate_window(model, window, 106000 + 100 * w + j)
                row = {"failure": None, "metrics": score(data, trajectory, cap),
                       "book_10s": trajectory["samples"][9::10],
                       "references": trajectory["references"],
                       "directions": trajectory["directions"]}
            except (blocks.BudgetExceeded, ValueError) as error:
                row = {"failure": str(error)}
            path.write_text(json.dumps(row) + "\n")


def main():
    step = sys.argv[1]
    if step == "claim-holdout":
        claim_holdout(pathlib.Path(sys.argv[2]))
        return
    if step == "draws":
        capture, fits, out = (pathlib.Path(p) for p in sys.argv[2:5])
        run_draws(capture, fits, out, sys.argv[5], int(sys.argv[6]))
        return
    if step == "s1":
        capture, fits, out = (pathlib.Path(p) for p in sys.argv[2:5])
        seed, w = int(sys.argv[5]), int(sys.argv[6])
        if seed not in SEEDS or w not in VALIDATION:
            raise SystemExit("seed 1 to 5, window 1 or 2")
        out.mkdir(parents=True, exist_ok=True)
        with open(fits / f"model-{seed}.pkl", "rb") as f:
            model = pickle.load(f)
        window = json.loads((capture / "validation" / f"window-{w - 1}.json").read_text())
        begun = time.time()

        def progress(simulated, count, events):
            (out / f"s1-{seed}-{w}.progress").write_text(
                f"{simulated:.0f} s, {count} blocks, {events} events, {time.time() - begun:.0f} s\n")

        run = s1_run(model, window, 107000 + 100 * w + seed, progress=progress)
        run.update(label="REAL DATA: Hyperliquid BTC, December 2025, validation window",
                   seed=seed, window=w, wall_s=time.time() - begun)
        (out / f"s1-{seed}-{w}.json").write_text(json.dumps(run, indent=1) + "\n")
    elif step == "s1-decide":
        out = pathlib.Path(sys.argv[2])
        runs = {f"{s}-{w}": json.loads((out / f"s1-{s}-{w}.json").read_text())
                for s in SEEDS for w in VALIDATION if (out / f"s1-{s}-{w}.json").exists()}
        result = decide(runs)
        (out / "s1.json").write_text(json.dumps(result, indent=1) + "\n")
        print(result)
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
