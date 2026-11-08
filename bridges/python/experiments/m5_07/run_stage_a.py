"""Stage A of #111: checks 1 and 2 and the primary comparison, once per freeze.

    python experiments/m5_07/run_stage_a.py <out dir> <dir of gate tables> [--recover <reason>]

The gate tables come from `examples/qrh_gate_table`, named `<example>_<scale>.txt`.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from decimal import Decimal as D

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402

DESIGN_SHA256 = "be16a4564c207e9d55df3f4144bb774d1bee96f25b8c389ebb0a889b0e09bf16"
MAX_RECOVERIES = 1
DT = 0.1
SEEDS = (101, 102, 103, 104, 105)
PATHS = 1000
TRADING_YEAR = 252 * 6.5 * 3600
EXAMPLES = {
    "example_1": {"delta": ["1", "-0.028"], "d": ["0.25", "0.05"], "m": ["1", "20"], "levels": [15, 15],
                  "lam": [1.0, 0.1], "T": 300.0, "Tb": 150.0, "kappa": "0.01"},
    "example_2": {"delta": ["0.533", "0.134", "-0.014", "-0.013"], "d": ["0.1", "0.1", "0.05", "0.05"],
                  "m": ["2", "2", "2", "2"], "R": "10", "lam": [0.05] * 4, "T": 600.0, "Tb": 300.0,
                  "kappa": "1"},
}
KAPPAS = ("0.0001", "0.001", "0.01", "0.03", "0.1", "0.3", "1")


def decimals(values):
    return [D(v) for v in values]


def build(name, kappa, sigma, factor=1):
    ex = EXAMPLES[name]
    kappa = D(kappa)
    if name == "example_1":
        return c.inventory_problem(decimals(ex["delta"]), decimals(ex["d"]), decimals(ex["m"]),
                                   [n * factor for n in ex["levels"]], ex["lam"],
                                   [kappa / 2, kappa / 2], kappa, sigma)
    return c.net_risk_problem(decimals(ex["delta"]), decimals(ex["d"]), decimals(ex["m"]),
                              D(ex["R"]) * factor, ex["lam"], kappa, sigma)


def alphas(name):
    return [1 / (4 * float(x)) for x in EXAMPLES[name]["d"]]


def v_hat(name, problem, kappa, sigma, v0=0.18):
    ex, kappa = EXAMPLES[name], float(kappa)
    delta, m = [float(x) for x in ex["delta"]], [float(x) for x in ex["m"]]
    if name == "example_2":
        return -c.net_risk_coefficient(delta, m, ex["lam"], alphas(name), kappa, sigma) * problem.risk**2
    a, b = c.asymptotic_ab(delta, m, ex["lam"], ex["lam"], alphas(name), [kappa / 2] * 2, kappa, sigma, v0)
    half = [(n - 1) // 2 for n in problem.shape]
    axes = [m[j] * np.arange(-half[j], half[j] + 1) for j in range(2)]
    return c.v_hat_inventory(problem, a, b, axes)


def tested_cells(name, problem, v):
    """(tested mask, shares excluded by each rule); the boundary band is reported apart."""
    if name == "example_1":
        band = np.ones(problem.shape, bool)
        band[3:-3, 3:-3] = False
    else:
        step = max(abs(mv.shift) for mv in problem.moves)
        band = np.zeros(problem.shape, bool)
        band[:2 * step + 1] = band[-2 * step - 1:] = True
    small = np.abs(v) < 1e-3 * np.abs(v).max()
    return ~band & ~small, {"small_v": float(small.mean()), "boundary": float(band.mean())}


def shape_error(v, approx, tested):
    """Max |v_hat - v| / |v| on tested cells, with C set so the maxima over them agree."""
    fitted = approx + (v[tested].max() - approx[tested].max())
    return float((np.abs(fitted - v)[tested] / np.abs(v)[tested]).max())


def central(array, parent_shape):
    """The block of `array` matching a grid of parent_shape, centred."""
    cuts = tuple(slice((a - b) // 2, (a - b) // 2 + b) for a, b in zip(array.shape, parent_shape))
    return array[cuts]


def check_1(name, sigma):
    ex = EXAMPLES[name]
    problem = build(name, ex["kappa"], sigma)
    v, _ = c.solve_hjb(problem, ex["T"], DT)
    tested, excluded = tested_cells(name, problem, v)
    fine, _ = c.solve_hjb(problem, ex["T"], DT / 2)
    wide, _ = c.solve_hjb(build(name, ex["kappa"], sigma, 2), ex["T"], DT)
    change_dt = float((np.abs(fine - v)[tested] / np.abs(v)[tested]).max())
    change_q = float((np.abs(central(wide, v.shape) - v)[tested] / np.abs(v)[tested]).max())
    error = shape_error(v, v_hat(name, problem, ex["kappa"], sigma), tested)
    share = float(tested.mean())
    refined = change_dt < 0.005 and change_q < 0.005
    status = "inconclusive" if share < 0.5 or not refined else ("pass" if error <= 0.05 else "fail")
    return {"status": status, "label": "control validated (shape)" if status == "pass" else None,
            "max_relative_error": error, "tested_share": share, "excluded_share": excluded,
            "refinement": {"dt_half": change_dt, "double_bound": change_q, "under_0_5pct": refined}}


def draws(name, problem, seed):
    steps = int(round(EXAMPLES[name]["Tb"] / DT))
    pairs = [c.draw_path(seed, path, steps, len(problem.moves)) for path in range(PATHS)]
    return np.stack([x[0] for x in pairs]), np.stack([x[1] for x in pairs])


def run_arm(name, problem, policy, seeds, sigma):
    out = []
    for seed in seeds:
        normals, uniforms = draws(name, problem, seed)
        counts, noise, _ = c.backtest(problem, policy, normals, uniforms, DT, sigma)
        out += c.path_pnl(problem, counts, noise)
    return np.array([float(x) for x in out])


def bootstrap(values, seed=1111, replicates=2000):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), (replicates, len(values)))
    means = values[picks].mean(axis=1)
    return [float(np.percentile(means, 5)), float(np.percentile(means, 95))]


def greedy(name, problem, kappa, sigma):
    return c.greedy_from_v_hat(problem, v_hat(name, problem, kappa, sigma))[None]


def check_2(name, sigma, seeds=SEEDS):
    ex, rows = EXAMPLES[name], []
    for kappa in KAPPAS:
        problem = build(name, kappa, sigma)
        _, exact = c.solve_hjb(problem, ex["T"], DT, keep_policy_until=ex["Tb"])
        exact_pnl = run_arm(name, problem, exact, seeds, sigma)
        approx_pnl = run_arm(name, problem, greedy(name, problem, kappa, sigma), seeds, sigma)
        interval = bootstrap(approx_pnl - exact_pnl)
        exact_interval = bootstrap(exact_pnl)
        testable = not exact_interval[0] <= 0 <= exact_interval[1]
        bound = 0.05 * abs(float(exact_pnl.mean()))
        rows.append({"kappa": kappa, "exact_mean": float(exact_pnl.mean()),
                     "approx_mean": float(approx_pnl.mean()), "exact_interval": exact_interval,
                     "paired_interval": interval, "testable": testable,
                     "pass": testable and -bound <= interval[0] and interval[1] <= bound,
                     "exact_std": float(exact_pnl.std()), "approx_std": float(approx_pnl.std())})
    tested = [r for r in rows if r["testable"]]
    status = "inconclusive" if len(tested) < 4 else ("pass" if all(r["pass"] for r in tested) else "fail")
    return {"status": status, "rows": rows}


def load_gate(path, shape):
    table = np.loadtxt(path, dtype=np.uint16, ndmin=2)
    return (table[:, 0].astype(np.uint8).reshape(shape)[None],
            float((table[:, 1] != 0).mean()))


def primary(name, sigma, tables, seeds=SEEDS):
    ex = EXAMPLES[name]
    problem = build(name, ex["kappa"], sigma)
    qrh = run_arm(name, problem, greedy(name, problem, ex["kappa"], sigma), seeds, sigma)
    out = {}
    for scale in ("1", "0.5", "2"):
        gate, scaled = load_gate(pathlib.Path(tables) / f"{name}_{scale}.txt", problem.shape)
        pnl = run_arm(name, problem, gate, seeds, sigma)
        diff = qrh - pnl
        lo, hi = bootstrap(diff)
        verdict = "better" if lo > 0 else ("worse" if hi < 0 else "no difference shown")
        out[scale] = {"qrh_mean": float(qrh.mean()), "gate_mean": float(pnl.mean()),
                      "paired_interval": [lo, hi], "verdict": verdict, "scaled_share": scaled,
                      "qrh_std": float(qrh.std()), "gate_std": float(pnl.std())}
    return out


def _attempt():
    spec = importlib.util.spec_from_file_location("attempt", HERE.parent / "m5_02" / "attempt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def claim(out, recover):
    """Checks the freeze and claims the single attempt, before any evaluation."""
    digest = hashlib.sha256((HERE / "design.json").read_bytes()).hexdigest()
    if digest != DESIGN_SHA256:
        raise SystemExit(f"design.json does not match the freeze: {digest}")
    attempts = _attempt().claim(pathlib.Path(out) / "attempt.jsonl", digest, recover)
    if sum(a["kind"] == "recovery" for a in attempts) > MAX_RECOVERIES:
        raise SystemExit("too many recoveries")
    return attempts


def evaluate(out, tables, recover=None):
    attempts = claim(out, recover)
    sigma = float(c.sigma_per_second(3000, 0.18))
    result = {"attempts": attempts, "sigma": sigma}
    first = check_1("example_1", sigma)
    result["example_1"] = {"check_1": first}
    if first["status"] == "fail":
        result["stopped"] = "check 1 failed on example 1"
    else:
        result["example_2"] = {"check_1": check_1("example_2", sigma)}
        slow = float(c.sigma_per_second(3000, 0.18, TRADING_YEAR))
        for name in EXAMPLES:
            result[name]["check_2"] = check_2(name, sigma)
            result[name]["primary"] = primary(name, sigma, tables)
            result[name]["primary_trading_year"] = primary(name, slow, tables)
    (pathlib.Path(out) / "stage_a.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("tables")
    parser.add_argument("--recover")
    args = parser.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    evaluate(args.out, args.tables, args.recover)
