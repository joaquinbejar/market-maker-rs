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

DESIGN_SHA256 = "48c8ea5aa0602ff54409f812b22907d4d27f2ab3699eb98998e240a8799642fa"
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
EXAMPLE_3 = {"kappas": ("0.0001", "0.001", "0.01", "0.1"), "levels": 15, "T": 4000.0, "Tb": 2000.0}
DECOUPLED_DELTA = ["1", "0"]
NOT_RUNNABLE = ("not runnable: the state is the net risk r under the joint delta, and an SPX-only risk "
                "is not a function of r, so a decoupled rule cannot be evaluated on this grid")


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


def v_hat(name, problem, kappa, sigma, v0=0.18, delta=None):
    ex, kappa = EXAMPLES[name], float(kappa)
    delta, m = [float(x) for x in (delta or ex["delta"])], [float(x) for x in ex["m"]]
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


def draws(horizon, problem, seed):
    steps = int(round(horizon / DT))
    pairs = [c.draw_path(seed, path, steps, len(problem.moves)) for path in range(PATHS)]
    return np.stack([x[0] for x in pairs]), np.stack([x[1] for x in pairs])


def run_arm(name, problem, policy, seeds, sigma):
    out = []
    for seed in seeds:
        normals, uniforms = draws(EXAMPLES[name]["Tb"], problem, seed)
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


def paired(reference, other):
    """The paired (reference - other) P&L interval and each arm's mean and standard deviation."""
    return {"reference_mean": float(reference.mean()), "reference_std": float(reference.std()),
            "other_mean": float(other.mean()), "other_std": float(other.std()),
            "paired_interval": bootstrap(reference - other)}


def decoupled_example_1(sigma, tables, seeds=SEEDS):
    """Example 1's other cells, each scored with the joint delta, against (greedy, joint). Secondary."""
    ex = EXAMPLES["example_1"]
    joint = build("example_1", ex["kappa"], sigma)
    reference = run_arm("example_1", joint, greedy("example_1", joint, ex["kappa"], sigma), seeds, sigma)
    gate, _ = load_gate(pathlib.Path(tables) / "example_1_1.txt", joint.shape)
    decoupled_gate, scaled = load_gate(pathlib.Path(tables) / "example_1_decoupled_1.txt", joint.shape)
    decoupled_value = v_hat("example_1", joint, ex["kappa"], sigma, delta=DECOUPLED_DELTA)
    policies = {"gate_joint": gate, "greedy_decoupled": c.greedy_from_v_hat(joint, decoupled_value)[None],
                "gate_decoupled": decoupled_gate}
    return {"scaled_share_gate_decoupled": scaled,
            **{cell: paired(reference, run_arm("example_1", joint, policy, seeds, sigma))
               for cell, policy in policies.items()}}


def example_3_arms(kappa, sigma):
    """(joint inventory problem for the multi-asset arm, one problem per asset with kappa = 0)."""
    ex = EXAMPLES["example_2"]
    delta, d, m = decimals(ex["delta"]), decimals(ex["d"]), decimals(ex["m"])
    n, half = len(delta), D(kappa) / 2
    multi = c.inventory_problem(delta, d, m, [EXAMPLE_3["levels"]] * n, ex["lam"], [half] * n, D(kappa), sigma)
    uni = [c.inventory_problem([delta[j]], [d[j]], [m[j]], [EXAMPLE_3["levels"]], [ex["lam"][j]], [half], D(0), sigma)
           for j in range(n)]
    return multi, uni


def multi_asset_policy(problem, kappa, sigma):
    ex = EXAMPLES["example_2"]
    delta, m = [float(x) for x in ex["delta"]], [float(x) for x in ex["m"]]
    a, b = c.asymptotic_ab(delta, m, ex["lam"], ex["lam"], alphas("example_2"), [float(kappa) / 2] * len(m),
                           float(kappa), sigma, 0.18)
    half = [(n - 1) // 2 for n in problem.shape]
    axes = [m[j] * np.arange(-half[j], half[j] + 1) for j in range(len(m))]
    return c.greedy_from_v_hat(problem, c.v_hat_inventory(problem, a, b, axes))[None]


def run_uni(problems, policies, normals, uniforms, sigma):
    """Each asset alone on its own two uniforms and the shared dW; the P&L is summed per path."""
    total = None
    for j, (problem, policy) in enumerate(zip(problems, policies)):
        counts, noise, _ = c.backtest(problem, policy, normals, uniforms[:, :, [2 * j, 2 * j + 1]], DT, sigma)
        pnl = np.array([float(x) for x in c.path_pnl(problem, counts, noise)])
        total = pnl if total is None else total + pnl
    return total


def example_3(sigma, seeds=SEEDS):
    """Multi-asset greedy against the uni-asset benchmark, delta frozen. Secondary."""
    rows = []
    for kappa in EXAMPLE_3["kappas"]:
        multi, uni = example_3_arms(kappa, sigma)
        policy = multi_asset_policy(multi, kappa, sigma)
        exact = [c.solve_hjb(p, EXAMPLE_3["T"], DT, keep_policy_until=EXAMPLE_3["Tb"])[1] for p in uni]
        multi_pnl, uni_pnl = [], []
        for seed in seeds:
            normals, uniforms = draws(EXAMPLE_3["Tb"], multi, seed)
            counts, noise, _ = c.backtest(multi, policy, normals, uniforms, DT, sigma)
            multi_pnl += [float(x) for x in c.path_pnl(multi, counts, noise)]
            uni_pnl += list(run_uni(uni, exact, normals, uniforms, sigma))
        rows.append({"kappa": kappa, **paired(np.array(multi_pnl), np.array(uni_pnl))})
    return {"status": "run", "rows": rows}


def checks_passed(result, name):
    return result[name]["check_1"]["status"] == "pass" and result[name]["check_2"]["status"] == "pass"


def decide(result):
    """The design's rule: only example 2's primary at x1, with both its checks passing, can adopt."""
    verdict = result["example_2"]["primary"]["1"]["verdict"]
    validated = checks_passed(result, "example_2")
    return {"adopt": validated and verdict == "better", "example_2_primary": verdict,
            "example_2_label": None if validated else "control not validated",
            "scope": "control under Approximation 3.1 with the paper's delta on SPX/VIX parameters; sigma 0.2267 is our unit choice"}


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
    marker = pathlib.Path(out) / "attempt.jsonl"
    if marker.exists():
        earlier = [json.loads(line) for line in marker.read_text().splitlines() if line.strip()]
        if any(a["frozen_sha256"] != digest for a in earlier):
            raise SystemExit("an earlier attempt ran under another frozen_sha256")
        if recover and sum(a["kind"] == "recovery" for a in earlier) >= MAX_RECOVERIES:
            raise SystemExit("too many recoveries")
    return _attempt().claim(marker, digest, recover)


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
        result["example_1"]["secondary"] = decoupled_example_1(sigma, tables)
        result["example_2"]["secondary"] = {"status": NOT_RUNNABLE}
        result["example_3"] = (example_3(sigma) if all(checks_passed(result, n) for n in EXAMPLES)
                               else {"status": "not run: checks 1 and 2 did not pass on examples 1 and 2"})
        result["decision"] = decide(result)
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
