"""Trains MDQR-small for #106 on the captured windows. REAL DATA.

    python experiments/m5_01/train.py <capture dir> <out dir> <seed> [training windows]

Reads only the mdqr_train and validation sets, never the holdout. `training windows` limits
the training set, for the timing pilot. Writes `model-<seed>.pkl` and `fit-<seed>.json`.
"""

import json
import pathlib
import pickle
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blocks  # noqa: E402
import mdqr  # noqa: E402

TRAIN_WINDOWS, VALIDATION_WINDOWS = 5, 2


def load(capture, set_name, index):
    return json.loads((pathlib.Path(capture) / set_name / f"window-{index}.json").read_text())


def mechanics_of(window_summaries, event_sizes):
    trajectories = [{"limits": blocks.LIMITS, "samples": s} for s, _ in window_summaries]
    records = [{"references": r} for _, r in window_summaries]
    return mdqr.estimate_mechanics(trajectories, records, event_sizes)


def main():
    capture, out, seed = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else TRAIN_WINDOWS
    out.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    parts, summaries, tails, sizes = [], [], {}, None
    for i in range(count):
        window = load(capture, "mdqr_train", i)
        sizes = window["event_sizes"]
        parts.append(blocks.rows_of(window))
        summaries.append(([[s["timestamp_ms"], s["bid"][:blocks.LIMITS], s["ask"][:blocks.LIMITS],
                            s["spread_ticks"]] for s in window["samples"]],
                          [[r["timestamp_ms"], r["moved"], r["reinitialized"]]
                           for r in window["references"]]))
        for key, values in blocks.tails_of([window]).items():
            tails.setdefault(key, []).extend(values)
        del window
    tails = {key: sorted(v) for key, v in tails.items()}
    held = [blocks.rows_of(load(capture, "validation", i)) for i in range(VALIDATION_WINDOWS)]
    train, validation = blocks.join(parts), blocks.join(held)
    del parts, held
    prepared = time.time()
    model, history = blocks.fit(train, validation, tails, seed)
    model["mechanics"] = mechanics_of(summaries, sizes).to_json()
    with open(out / f"model-{seed}.pkl", "wb") as f:
        pickle.dump(model, f)
    (out / f"fit-{seed}.json").write_text(json.dumps({
        "label": "REAL DATA: Hyperliquid BTC, December 2025", "seed": seed, "windows": count,
        "epoch_kept": model["epoch"], "history": history, "blocks": len(train["wait_dt"]),
        "events": len(train["size_class"]), "prepare_s": prepared - begun,
        "fit_s": time.time() - prepared}, indent=1) + "\n")


if __name__ == "__main__":
    main()
