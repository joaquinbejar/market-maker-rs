"""Part A of #107: the paper's single-asset setting, reproduced apart from our simulator.

From `bridges/python`:

    python experiments/m5_02/run_paper.py widths
    python experiments/m5_02/run_paper.py train <out dir> <commit> <parallel>
    python experiments/m5_02/run_paper.py freeze <out dir>
    python experiments/m5_02/run_paper.py evaluate <out dir> <frozen sha256> [--recover <reason>]
    python experiments/m5_02/run_paper.py report <out dir>

`widths` checks that the head widths in `design_paper.json` are the ones its parameter target
gives. `train` refuses to run unless they are. `evaluate` reads only checkpoints whose
hashes `freeze` recorded, refuses a different design, and runs once.

SYNTHETIC: the paper's setting as its text gives it, with the choices `paper_env` lists.
"""

import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import pathlib
import platform
import sys
import time

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from torch.utils.flop_counter import FlopCounterMode

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attempt  # noqa: E402
import nets  # noqa: E402
import paper_env  # noqa: E402


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_design():
    path = HERE / "design_paper.json"
    return {"paper": json.loads(path.read_text())}, sha256(path)


def layout_of(part):
    return nets.Layout(**part["layout"])


def env_of(part, setting):
    return paper_env.PaperEnv(
        paper_env.kernel_of(setting), setting["horizon"], buffer=part["layout"]["buffer"]
    )


def model_of(part, setting, arm, width, seed, env=None):
    env = env or env_of(part, setting)
    ppo = dict(part["ppo"])
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=nets.policy_kwargs(arm, layout_of(part), width),
        device="cpu",
        seed=seed,
        verbose=0,
        **ppo,
    )


def arm_for(part, setting, name):
    """An arm's extractor arguments, with the HSN initialization of its setting."""
    arm = dict(part["arms"][name])
    if arm["extractor"] == "hsn":
        init = part["hsn_init"][setting["hsn_init"]]
        arm["alphas"], arm["betas"] = init["alphas"], init["betas"]
    return arm


def matched_width(part, setting, name):
    target, tolerance = part["parameters"]["target"], part["parameters"]["tolerance"]
    best = None
    for width in range(part["parameters"]["min_width"], part["parameters"]["max_width"] + 1):
        count = nets.parameter_count(model_of(part, setting, arm_for(part, setting, name), width, 0).policy)
        if best is None or abs(count - target) < abs(best[1] - target):
            best = (width, count)
    if abs(best[1] - target) > tolerance * target:
        raise ValueError(f"{name}: no width within {tolerance} of {target} ({best})")
    return best


def check_widths(design):
    part = design["paper"]
    setting = part["settings"][0]
    found = {name: matched_width(part, setting, name) for name in part["arms"]}
    for name, (width, count) in found.items():
        if part["widths"].get(name) != width:
            raise SystemExit(f"design width of {name} is {part['widths'].get(name)}, target gives {width}")
    return found


def versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "device": "cpu",
        "threads": 1,
    }


def jobs_of(design):
    part = design["paper"]
    return [
        {"setting": s["name"], "arm": a, "seed": seed}
        for s in part["settings"]
        for a in part["arms"]
        for seed in part["training_seeds"]
    ]


def name_of(job):
    return f"{job['setting']}__{job['arm']}__seed{job['seed']}"


def setting_named(part, name):
    return next(s for s in part["settings"] if s["name"] == name)


def train_job(args):
    job, out, commit, design_sha = args
    torch.set_num_threads(1)
    design, sha = load_design()
    if sha != design_sha:
        raise RuntimeError("design changed during the run")
    part = design["paper"]
    setting = setting_named(part, job["setting"])
    width = part["widths"][job["arm"]]
    model = model_of(part, setting, arm_for(part, setting, job["arm"]), width, job["seed"])
    start = time.monotonic()
    model.learn(part["budget_steps"])
    wall = time.monotonic() - start
    path = pathlib.Path(out) / "models" / f"{name_of(job)}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    meta = {
        **job,
        "width": width,
        "parameters": nets.parameter_count(model.policy),
        "budget_steps": part["budget_steps"],
        "wall_s": wall,
        "commit": commit,
        "design_sha256": design_sha,
        "versions": versions(),
        "zip_sha256": sha256(path),
    }
    extractor = model.policy.features_extractor
    if isinstance(extractor, nets.Hsn):
        meta["learned"] = {
            "alphas": extractor.alphas().detach().tolist(),
            "betas": extractor.betas().detach().tolist(),
        }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return name_of(job), wall


def train(out, commit, parallel):
    design, sha = load_design()
    check_widths(design)
    out = pathlib.Path(out)
    if (out / "frozen.json").exists():
        raise SystemExit("already frozen")
    todo = [j for j in jobs_of(design) if not (out / "models" / f"{name_of(j)}.json").exists()]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as pool:
        for name, wall in pool.map(train_job, [(j, out, commit, sha) for j in todo]):
            print(f"{name} {wall:.1f} s", flush=True)


def freeze(out):
    design, sha = load_design()
    out = pathlib.Path(out)
    entries = {}
    for job in jobs_of(design):
        zip_path = out / "models" / f"{name_of(job)}.zip"
        meta_path = zip_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        if meta["design_sha256"] != sha or meta["zip_sha256"] != sha256(zip_path):
            raise SystemExit(f"{name_of(job)} does not match the design or its zip")
        entries[name_of(job)] = {"zip": sha256(zip_path), "metadata": sha256(meta_path)}
    frozen = out / "frozen.json"
    if frozen.exists():
        raise SystemExit("already frozen")
    frozen.write_text(json.dumps({"design_sha256": sha, "runs": entries}, indent=1, sort_keys=True))
    print(sha256(frozen))


def audit_observations(part, setting, count):
    """A fixed set of observations: the paper env under a fixed uniform random policy."""
    env = env_of(part, setting)
    rng = np.random.default_rng(part["audit"]["seed"])
    observations = []
    episode = 0
    while len(observations) < count:
        obs, _ = env.reset(seed=part["audit"]["episode_seed_base"] + episode)
        episode += 1
        done = False
        while not done and len(observations) < count:
            observations.append(obs)
            action = rng.uniform(-3, 3, size=2).astype(np.float32)
            obs, _, done, _, _ = env.step(action)
    return np.asarray(observations, dtype=np.float32)


def inference_audit(part, setting, path, observations):
    first = PPO.load(path, device="cpu")
    second = PPO.load(path, device="cpu")
    a, _ = first.predict(observations, deterministic=True)
    b, _ = second.predict(observations, deterministic=True)
    one = torch.as_tensor(observations[:1])
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            first.policy._predict(one, deterministic=True)
    timings = []
    with torch.no_grad():
        for row in observations[: part["audit"]["timed"]]:
            start = time.perf_counter_ns()
            first.policy._predict(torch.as_tensor(row[None, :]), deterministic=True)
            timings.append(time.perf_counter_ns() - start)
    return {
        "identical": bool(np.array_equal(a, b)),
        "flops_per_decision": int(counter.get_total_flops()),
        "median_inference_us": float(np.median(timings)) / 1000.0,
    }


def evaluate(out, frozen_sha, recover=None):
    torch.set_num_threads(1)
    design, sha = load_design()
    part = design["paper"]
    out = pathlib.Path(out)
    frozen_path = out / "frozen.json"
    if sha256(frozen_path) != frozen_sha:
        raise SystemExit("frozen manifest differs")
    frozen = json.loads(frozen_path.read_text())
    if frozen["design_sha256"] != sha:
        raise SystemExit("design differs from the frozen one")
    result_path = out / "evaluation.json"
    if result_path.exists():
        raise SystemExit("already evaluated; the evaluation runs once")
    for job in jobs_of(design):
        name = name_of(job)
        if sha256(out / "models" / f"{name}.zip") != frozen["runs"][name]["zip"]:
            raise SystemExit(f"{name} changed after the freeze")
    # One attempt per freeze, claimed before any checkpoint is evaluated (attempt.py).
    attempts = attempt.claim(out / "evaluation.attempt", frozen_sha, recover)
    seeds = list(range(part["evaluation"]["seed_base"], part["evaluation"]["seed_base"] + part["evaluation"]["episodes"]))
    audits = {s["name"]: audit_observations(part, s, part["audit"]["observations"]) for s in part["settings"]}
    rows = []
    for job in jobs_of(design):
        name = name_of(job)
        zip_path = out / "models" / f"{name}.zip"
        if sha256(zip_path) != frozen["runs"][name]["zip"]:
            raise SystemExit(f"{name} changed after the freeze")
        setting = setting_named(part, job["setting"])
        model = PPO.load(zip_path, device="cpu")
        env = env_of(part, setting)
        objectives, fills = [], []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            objectives.append(info["objective"])
            fills.append(info["fills"])
        meta = json.loads(zip_path.with_suffix(".json").read_text())
        rows.append(
            {
                **job,
                "mean_objective": float(np.mean(objectives)),
                "objectives": objectives,
                "mean_fills": float(np.mean(fills)),
                "parameters": meta["parameters"],
                "train_wall_s": meta["wall_s"],
                "learned": meta.get("learned"),
                "audit": inference_audit(part, setting, zip_path, audits[job["setting"]]),
            }
        )
        print(name, f"{rows[-1]['mean_objective']:.5f}", flush=True)
    result_path.write_text(json.dumps({"frozen_sha256": frozen_sha, "attempts": attempts,
                                       "rows": rows}, indent=1))
    print(sha256(result_path))


# Welch's t-test, as the paper reports; the regularized incomplete beta by continued fraction.
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def welch(x, y):
    """Two-sided Welch t-test: (t, degrees of freedom, p)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    p = betainc(df / 2.0, 0.5, df / (df + t * t))
    return float(t), float(df), float(p)


def report(out):
    design, _ = load_design()
    part = design["paper"]
    out = pathlib.Path(out)
    rows = json.loads((out / "evaluation.json").read_text())["rows"]
    alpha = part["reproduction"]["significance"]
    tables = {}
    for setting in part["settings"]:
        per_arm = {}
        for arm in part["arms"]:
            values = [r["mean_objective"] for r in rows if r["setting"] == setting["name"] and r["arm"] == arm]
            per_arm[arm] = {
                "mean": float(np.mean(values)),
                "sem": float(np.std(values, ddof=1) / math.sqrt(len(values))),
                "seeds": len(values),
            }
        controls = [a for a in part["arms"] if a != "hsn"]
        best = max(controls, key=lambda a: per_arm[a]["mean"])
        hsn = [r["mean_objective"] for r in rows if r["setting"] == setting["name"] and r["arm"] == "hsn"]
        tests = {}
        for arm in controls:
            other = [r["mean_objective"] for r in rows if r["setting"] == setting["name"] and r["arm"] == arm]
            t, df, p = welch(hsn, other)
            tests[arm] = {"t": t, "df": df, "p": p}
        expected = setting["paper"]["expected"]
        p_best = tests[best]["p"]
        if expected == "tie":
            observed = "tie" if p_best >= alpha else ("hsn_better" if per_arm["hsn"]["mean"] > per_arm[best]["mean"] else "hsn_worse")
        else:
            observed = "hsn_better" if (per_arm["hsn"]["mean"] > per_arm[best]["mean"] and p_best < alpha) else (
                "tie" if p_best >= alpha else "hsn_worse")
        tables[setting["name"]] = {
            "arms": per_arm,
            "best_control": best,
            "welch_vs_controls": tests,
            "paper": setting["paper"],
            "observed": observed,
            "reproduced": observed == expected,
        }
    path = out / "report.json"
    path.write_text(json.dumps(tables, indent=1, sort_keys=True))
    print(json.dumps(tables, indent=1, sort_keys=True))
    print(sha256(path))


def main():
    step = sys.argv[1]
    if step == "widths":
        design, _ = load_design()
        print(json.dumps(check_widths(design)))
    elif step == "train":
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif step == "freeze":
        freeze(sys.argv[2])
    elif step == "evaluate":
        evaluate(sys.argv[2], sys.argv[3], attempt.recovery_reason(sys.argv))
    elif step == "report":
        report(sys.argv[2])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
