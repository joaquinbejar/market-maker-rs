"""Assembles #106's S2 from the holdout's draws and transfer runs, and applies the rule. REAL DATA.

    python experiments/m5_01/aggregate.py score-qr <capture> <arms dir> <out dir>
    python experiments/m5_01/aggregate.py decide <out dir> <transfer json>...

`score-qr` scores the fitted arms' draws against the data; `decide` writes `s2.json`.
"""

import glob
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402
import s2  # noqa: E402

WINDOWS, DRAWS, SEEDS = 3, 10, 5
ARMS = ("qr", "qr_count", "qr_constant")


def tau_b(x, y):
    """Kendall's tau-b; NaN if either is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    concordant = tied_x = tied_y = total = 0.0
    for i in range(len(x)):
        dx, dy = np.sign(x[i + 1:] - x[i]), np.sign(y[i + 1:] - y[i])
        concordant += float((dx * dy).sum())
        tied_x += float((dx == 0).sum())
        tied_y += float((dy == 0).sum())
        total += len(dx)
    denominator = np.sqrt((total - tied_x) * (total - tied_y))
    return concordant / denominator if denominator else float("nan")


def tau_interval(a, b, rng, replicates=s2.REPLICATES):
    """90 % interval of tau-b, resampling each window's blocks of per-block nets, the same
    blocks for both arms. `a` and `b`: policies x windows x blocks."""
    out = []
    for _ in range(replicates):
        picks = [s2.stationary_indices(a.shape[2], rng) for _ in range(a.shape[1])]
        means = [np.mean([array[:, w, pick].sum(axis=1) for w, pick in enumerate(picks)], axis=0)
                 for array in (a, b)]
        out.append(tau_b(*means))
    return float(np.nanpercentile(out, 5)), float(np.nanpercentile(out, 95))


def nets(transfer, names):
    """policies x windows x blocks, the mean over the draws of each key in `transfer`, or None
    if a draw failed. `names` maps a window (1 to 3) to its result rows."""
    policies = sorted(next(iter(transfer[names[1][0]].values())) and transfer[names[1][0]])
    out = np.empty((len(policies), WINDOWS, s2.BLOCKS))
    for w in range(1, WINDOWS + 1):
        rows = [transfer[n] for n in names[w]]
        if any("failure" in r for r in rows):
            return None, policies
        for p, policy in enumerate(policies):
            out[p, w - 1] = np.mean([r[policy]["blocks"] for r in rows], axis=0)
    return out, policies


def score_qr(capture, arms_dir, out, cap):
    """One JSON of the block metrics per QR draw, under `out/holdout`."""
    if not (out / evaluate.HOLDOUT_MARKER).exists():
        raise SystemExit("the holdout is not claimed")
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    for w in range(1, WINDOWS + 1):
        window = json.loads((capture / "holdout" / f"window-{w - 1}.json").read_text())
        data = s2.per_block(s2.from_window(window))
        for arm in ARMS:
            for j in range(1, DRAWS + 1):
                value = json.loads((arms_dir / f"{arm}-w{w}-d{j}.json").read_text())
                value["directions"] = value["reference_directions"]
                row = {"failure": value["stopped"],
                       "metrics": evaluate.score(data, value, cap)}
                path = out / "holdout" / f"{arm}-w{w}-d{j}.json"
                if path.exists():
                    raise SystemExit(f"{path} exists: the holdout runs once")
                path.write_text(json.dumps(row) + "\n")


def table_of(out):
    """Each arm's metric per window and block, mean over draws (and MDQR's seeds); failed draws."""
    table, failed = {}, 0
    keys = None
    dropped = {}
    def draws(prefix, w):
        nonlocal failed, keys
        rows = []
        for path in sorted(glob.glob(str(out / "holdout" / f"{prefix}-w{w}-d*.json"))):
            row = json.loads(pathlib.Path(path).read_text())
            if row.get("failure"):
                failed += prefix.startswith("mdqr")
                continue
            rows.append(row["metrics"])
        keys = keys or list(rows[0])
        dropped[prefix.split("-")[0]] = dropped.get(prefix.split("-")[0], 0) + sum(
            int(np.isnan(np.asarray(r[k], float)).sum()) for r in rows for k in keys)
        return {k: np.nanmean([r[k] for r in rows], axis=0) for k in keys}
    for arm in ARMS:
        table[arm] = [draws(arm, w) for w in range(1, WINDOWS + 1)]
    seeds = [[draws(f"mdqr-{s}", w) for w in range(1, WINDOWS + 1)] for s in range(1, SEEDS + 1)]
    table["mdqr"] = [{k: np.nanmean([s[i][k] for s in seeds], axis=0) for k in keys}
                     for i in range(WINDOWS)]
    return {arm: {k: [windows[i][k].tolist() for i in range(WINDOWS)] for k in keys}
            for arm, windows in table.items()}, failed, dropped


def names_of(transfer, arm):
    """The draws of an arm per window, matched on the exact file name; a wrong count stops."""
    expected = DRAWS * (SEEDS if arm == "mdqr" else 1)
    out = {}
    for w in range(1, WINDOWS + 1):
        pattern = re.compile(rf"{arm}(-\d+)?-w{w}-d\d+\.json")
        out[w] = sorted(p for p in transfer if pattern.fullmatch(pathlib.Path(p).name))
        if len(out[w]) != expected:
            raise ValueError(f"{arm} window {w}: {len(out[w])} draws, not {expected}")
    return out


def decide(out, transfer_paths):
    transfer = {}
    for path in transfer_paths:
        transfer.update(json.loads(pathlib.Path(path).read_text()))
    table, failed, dropped = table_of(out)
    arrays = {}
    for arm in ARMS + ("mdqr",):
        arrays[arm], policies = nets(transfer, names_of(transfer, arm))
    rng = np.random.default_rng(s2.SEED)
    tau, intervals, policy_means = {}, {}, {}
    if arrays["mdqr"] is not None:
        mdqr = arrays["mdqr"].sum(axis=2).mean(axis=1)
        policy_means["mdqr"] = dict(zip(policies, mdqr.tolist(), strict=True))
        for arm in s2.QR_ARMS:
            if arrays[arm] is not None:
                tau[arm] = tau_b(mdqr, arrays[arm].sum(axis=2).mean(axis=1))
                intervals[arm] = tau_interval(arrays["mdqr"], arrays[arm], rng)
    usable = {k: v for k, v in tau.items() if np.isfinite(v)}
    result = s2.decide(table, usable if len(usable) == len(s2.QR_ARMS) else None, failed)
    result.update(label="REAL DATA: Hyperliquid BTC, December 2025; a reused holdout, exploratory",
                  tau_intervals=intervals, policy_means=policy_means,
                  undefined_values_per_arm=dropped)
    (out / "s2.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    if sys.argv[1] == "score-qr":
        fits = json.loads((pathlib.Path(sys.argv[2]) / "fits.json").read_text())
        caps = {fits["arms"][arm]["parameters"]["cap"] for arm in ARMS}
        if len(caps) != 1:
            raise SystemExit("the QR arms differ in their cap")
        score_qr(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]),
                 caps.pop())
    else:
        print(decide(pathlib.Path(sys.argv[2]), sys.argv[3:])["decision"])


if __name__ == "__main__":
    main()
