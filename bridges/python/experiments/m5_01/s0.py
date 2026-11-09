"""Stage S0 of #106: MDQR fitted on synthetic queue-reactive windows must recover the truth.

SYNTHETIC: the truth is the queue-reactive simulator at #102's calm parameters
(`design.json` s0.truth), not market data.

From `bridges/python`:

    cargo run --release --example mdqr_qr -- truth experiments/m5_01/design.json <data dir>
    cargo run --release --example mdqr_qr -- parameters experiments/m5_01/design.json > <out dir>/truth.json
    python experiments/m5_01/s0.py check <data dir> <out dir>
    python experiments/m5_01/s0.py fit <data dir> <out dir> <training seed>
    python experiments/m5_01/s0.py simulate <data dir> <out dir> <training seed>
    python experiments/m5_01/s0.py compare <data dir> <out dir> <compare binary>
    python experiments/m5_01/s0.py decide <out dir>

`check` records the sha256 of every window and the design's pins in `data.json`; every later
step refuses a window whose hash differs. `fit` writes `fit-<seed>.json` (the mechanics, the
network and its log-likelihood per epoch); `simulate` writes the model's evaluation windows;
`compare` runs `mdqr_qr compare` on every model and truth_again window against the evaluation
window of the same index; `decide` applies the design's stop rule and writes `s0.json`.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mdqr  # noqa: E402

SETS = ("train", "evaluation", "truth_again")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_design():
    return json.loads((HERE / "design.json").read_text())


def windows(design, data_dir, set_name, suffix):
    count = len(design["s0"]["seeds"][set_name])
    return [pathlib.Path(data_dir) / set_name / f"window-{i}.{suffix}.json"
            for i in range(1, count + 1)]


def write(out, name, value):
    path = pathlib.Path(out) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")
    return path


def read_checked(out, path):
    """A window, refused unless its hash is the one `check` recorded."""
    data = json.loads((pathlib.Path(out) / "data.json").read_text())
    expected = data["windows"].get(str(path))
    if expected is None or sha256(path) != expected:
        raise ValueError(f"{path} is not the window check recorded")
    return json.loads(pathlib.Path(path).read_text())


def check(data_dir, out):
    design = load_design()
    pin = sha256(HERE.parent / "m4_06" / "design.json")
    if pin != design["pins"]["m4_06_design_sha256"]:
        raise ValueError("experiments/m4_06/design.json is not the pinned one")
    recorded = {}
    for set_name in SETS:
        for suffix in ("trajectory", "events"):
            for path in windows(design, data_dir, set_name, suffix):
                recorded[str(path)] = sha256(path)
    truth = json.loads((pathlib.Path(out) / "truth.json").read_text())
    return write(out, "data", {"design_sha256": sha256(HERE / "design.json"),
                               "truth_fingerprint": truth["fingerprint"],
                               "windows": recorded})


def model_spec(design):
    model = design["s0"]["model"]
    return {"hidden": (128, 64), "lr": 0.001, "batch": 4096, "epochs": 8,
            "seeds": model["training_seeds"]}


def fit_seed(data_dir, out, seed):
    design = load_design()
    spec = model_spec(design)
    if seed not in spec["seeds"]:
        raise ValueError(f"seed {seed} is not a training seed of the design")
    started = time.perf_counter()
    trajectories = [read_checked(out, p) for p in windows(design, data_dir, "train", "trajectory")]
    records = [read_checked(out, p) for p in windows(design, data_dir, "train", "events")]
    mechanics = mdqr.estimate_mechanics(trajectories, records,
                                        design["s0"]["truth"]["event_sizes"])
    parts = [mdqr.training_rows(r, mechanics.limits, mechanics.max_queue) for r in records]
    rows = tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    network, history = mdqr.fit(rows, mechanics.limits, mechanics.max_queue, seed,
                                spec["hidden"], spec["lr"], spec["batch"], spec["epochs"])
    return write(out, f"fit-{seed}", {
        "seed": seed,
        "design_sha256": sha256(HERE / "design.json"),
        "events": int(rows[0].shape[0]),
        "mechanics": mechanics.to_json(),
        "network": network_json(network),
        "loglik_per_event_by_epoch": history,
        "seconds": time.perf_counter() - started,
    })


def network_json(network):
    return {"weights": [[w.tolist(), b.tolist()] for w, b in network.weights],
            "limits": network.limits, "max_queue": network.max_queue,
            "mean": network.mean.tolist(), "scale": network.scale.tolist()}


def network_of(value):
    return mdqr.Network([(np.asarray(w), np.asarray(b)) for w, b in value["weights"]],
                        value["limits"], value["max_queue"], np.asarray(value["mean"]),
                        np.asarray(value["scale"]))


def simulate_seed(data_dir, out, seed):
    design = load_design()
    s0 = design["s0"]
    fitted = json.loads((pathlib.Path(out) / f"fit-{seed}.json").read_text())
    if fitted["design_sha256"] != sha256(HERE / "design.json"):
        raise ValueError(f"fit-{seed} was fitted under another design")
    mechanics = mdqr.Mechanics.from_json(fitted["mechanics"])
    network = network_of(fitted["network"])
    paths = []
    for i, window_seed in enumerate(s0["seeds"]["evaluation"], start=1):
        started = time.perf_counter()
        trajectory = mdqr.simulate(mechanics, network.rates, window_seed,
                                   s0["reference_ticks"], s0["window_seconds"],
                                   s0["sample_ms"] / 1000.0, s0["max_events_per_window"])
        trajectory["seconds"] = time.perf_counter() - started
        trajectory["fit_sha256"] = sha256(pathlib.Path(out) / f"fit-{seed}.json")
        paths.append(write(out, f"model-{seed}/window-{i}.trajectory", trajectory))
    return paths


def compare_all(data_dir, out, binary):
    design = load_design()
    evaluation = windows(design, data_dir, "evaluation", "trajectory")
    for path in evaluation:
        read_checked(out, path)
    rows = []

    def run(kind, seed, model, data):
        result = subprocess.run([binary, "compare", str(HERE / "design.json"), str(data),
                                 str(model)], check=True, capture_output=True, text=True)
        rows.append({"kind": kind, "seed": seed, "window": data.name,
                     "report": json.loads(result.stdout)})

    for i, data in enumerate(evaluation, start=1):
        again = windows(design, data_dir, "truth_again", "trajectory")[i - 1]
        read_checked(out, again)
        run("truth_again", None, again, data)
        for seed in design["s0"]["model"]["training_seeds"]:
            run("mdqr", seed, pathlib.Path(out) / f"model-{seed}" / f"window-{i}.trajectory.json",
                data)
    return write(out, "compare", {"design_sha256": sha256(HERE / "design.json"), "rows": rows})


def decide(out):
    """The design's stop rule: for every training seed, the mean over the evaluation windows
    of the queue size TV at each limit and of the flow law TV for each kind is at most the
    truth's self-discrepancy, the same mean, plus the margin."""
    design = load_design()
    margin = 0.05
    rows = json.loads((pathlib.Path(out) / "compare.json").read_text())["rows"]

    def means(selected):
        queue = np.mean([r["report"]["queue_size"] for r in selected], axis=0)
        flow = np.mean([r["report"]["flow_law"] for r in selected], axis=0)
        return queue.tolist(), flow.tolist()

    truth_q, truth_f = means([r for r in rows if r["kind"] == "truth_again"])
    seeds = {}
    for seed in design["s0"]["model"]["training_seeds"]:
        q, f = means([r for r in rows if r["kind"] == "mdqr" and r["seed"] == seed])
        seeds[str(seed)] = {
            "queue_size_tv": q, "flow_law_tv": f,
            "passes": (all(a <= b + margin for a, b in zip(q, truth_q))
                       and all(a <= b + margin for a, b in zip(f, truth_f))),
        }
    passed = all(s["passes"] for s in seeds.values())
    return write(out, "s0", {
        "data": "SYNTHETIC: queue-reactive truth at #102's calm parameters",
        "truth_self": {"queue_size_tv": truth_q, "flow_law_tv": truth_f},
        "margin": margin, "seeds": seeds,
        "decision": "pass" if passed else "stop: MDQR cannot recover the truth",
    })


def main():
    step = sys.argv[1]
    if step == "check":
        print(check(sys.argv[2], sys.argv[3]))
    elif step == "fit":
        print(fit_seed(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif step == "simulate":
        print(simulate_seed(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif step == "compare":
        print(compare_all(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif step == "decide":
        print(decide(sys.argv[2]))
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
