"""Runs a step of the constrained comparison of #105 from its design, in the experiment's
own venv (`omnisafe-lock.txt`, Python 3.11, `MUJOCO_GL=disable`):

    python experiments/m4_08/run.py <step> <#102 data dir> <shocks dir> <out dir> <commit> [<frozen sha256>]

Steps, in order: `fee_rule`, `classical_selection`, `training`, `penalty_selection`,
`freeze`, `holdout`. Every day's rows are checked against #102's digests or `shocks.json`
before use. Only `holdout` reads holdout days, shocked or not, once, from the manifest
`freeze` writes. The kernel's limits are the only gate and the same for every arm; the
costs are measured beside the reward and never change what the kernel admits.

Everything here is synthetic. One synthetic simulator, made-up parameters and 24 days per
regime bound what it can show.
"""

import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
from decimal import Decimal

import numpy as np
import torch

from mm_rl_bridge import BoxMarketMakingEnv, ResidualMarketMakingEnv

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from costs import costs  # noqa: E402

DAY_MS = 86_400_000
SECONDS_PER_YEAR = 31_536_000
LEARNED = ("ppo", "cpo", "ppo_penalized")
ALGORITHM = {"ppo": "PPO", "cpo": "CPO", "ppo_penalized": "PPO"}
HOLD = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_design():
    design = json.loads((HERE / "design.json").read_text())
    base = json.loads((HERE.parent / "m4_05" / "digests.json").read_text())
    shocks = json.loads((HERE / "shocks.json").read_text())
    if shocks["design_version"] != design["design_version"]:
        raise ValueError("the shocks belong to another design")
    digests = {(d["split"], d["regime"], d["day"], None): d["dataset_digest"] for d in base["days"]}
    for s in shocks["shocks"]:
        digests[(s["split"], s["regime"], s["day"], s["shock"])] = s["dataset_digest"]
    return design, digests


def regimes_of(design):
    return ["calm", "stressed", "thin"]


def days_of(design, split):
    first, last = design["splits"][split]
    return list(range(first, last + 1))


def day_file(paths, split, regime, day, shock):
    if shock is None:
        return pathlib.Path(paths["data"]) / split / regime / f"day-{day}.json"
    return pathlib.Path(paths["shocks"]) / split / regime / f"day-{day}-shock{shock}.json"


def read_rows(paths, digests, split, regime, day, shock, holdout_allowed=False):
    if split == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    path = day_file(paths, split, regime, day, shock)
    return [tuple(r) for r in json.loads(path.read_text())], digests[(split, regime, day, shock)]


def base_config(design, fee):
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": 8640, "fee_rate": fee, "initial_capital": run["initial_capital"],
        "latency": run["latency"], "fill_model": run["fill_model"], "limits": run["limits"],
        "bid_offsets": grid["bid_offsets"], "ask_offsets": grid["ask_offsets"],
        "sizes": grid["sizes"], "scales": design["observation_scales"], "f32_unit": "1",
        "invalid_action": "reject",
        "invalid_action_penalty": design["action_space"]["invalid_action_penalty"],
        "terminal": run["terminal"],
    }


def volatility(paths, design, digests, regime):
    """#102's rule over the training days, at 10 s rows."""
    tick = Decimal(design["kernel"]["tick_size"])
    count, total, squares = 0, 0.0, 0.0
    for day in days_of(design, "train"):
        rows, _ = read_rows(paths, digests, "train", regime, day, None)
        mids = [float(Decimal(r[1] + r[3]) / 2 * tick) for r in rows]
        for a, b in zip(mids, mids[1:]):
            count, total, squares = count + 1, total + (b - a), squares + (b - a) ** 2
    mean = total / count
    return f"{((squares / count - mean * mean) * SECONDS_PER_YEAR / 10) ** 0.5:.4f}"


def grid_points(grid, sigma):
    keys = sorted(grid)
    points = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    for point in points:
        if point.get("volatility") == "from the volatility rule":
            point["volatility"] = sigma
    return points


def build_env(job, digests, holdout_allowed=False):
    design, _ = load_design()
    rows, digest = read_rows(job["paths"], digests, job["split"], job["regime"], job["day"],
                             job["shock"], holdout_allowed)
    config = base_config(design, job["fee"])
    if job["arm"] == "avellaneda_stoikov":
        p = job["point"]
        config["residual"] = {"baseline": {
            "kind": "avellaneda_stoikov", "risk_aversion": p["risk_aversion"],
            "volatility": p["volatility"], "order_intensity": p["order_intensity"],
            "session_start_ms": (job["day"] - 1) * DAY_MS, "horizon_ms": p["horizon_ms"],
            "size": p["size"]}, "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        env = ResidualMarketMakingEnv(config, rows)
    else:
        env = BoxMarketMakingEnv(config, rows)
    if env.bridge_info["dataset_digest"] != digest:
        env.close()
        raise ValueError(f"{job['split']} {job['regime']} {job['day']} {job['shock']}: other rows")
    return env, rows


class Learned:
    """An OmniSafe policy as saved: its actor and observation normalizer; it acts on the
    actor's mean."""

    def __init__(self, run_dir, model, config_sha256=None):
        # The configuration sets the actor and the normalizer: it must be the one trained.
        if config_sha256 is not None and \
                sha256(pathlib.Path(run_dir) / "config.json") != config_sha256:
            raise ValueError(f"{run_dir}/config.json changed after training")
        from omnisafe.common.normalizer import Normalizer
        from omnisafe.models.actor import ActorBuilder
        import gymnasium

        cfgs = json.loads((pathlib.Path(run_dir) / "config.json").read_text())
        params = torch.load(pathlib.Path(run_dir) / "torch_save" / model, map_location="cpu")
        width = int(params["pi"][next(k for k in params["pi"] if k.endswith("0.weight"))].shape[1])
        obs_space = gymnasium.spaces.Box(-1.0, 1.0, (width,), dtype=np.float32)
        act_space = gymnasium.spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        actor = cfgs["model_cfgs"]["actor"]
        self._actor = ActorBuilder(obs_space=obs_space, act_space=act_space,
                                   hidden_sizes=actor["hidden_sizes"],
                                   activation=actor["activation"],
                                   weight_initialization_mode=cfgs["model_cfgs"][
                                       "weight_initialization_mode"],
                                   ).build_actor(cfgs["model_cfgs"]["actor_type"])
        self._actor.load_state_dict(params["pi"])
        self._norm = None
        if cfgs["algo_cfgs"]["obs_normalize"]:
            self._norm = Normalizer(shape=obs_space.shape, clip=5)
            self._norm.load_state_dict(params["obs_normalizer"])

    def predict(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32)
        if self._norm is not None:
            obs = self._norm.normalize(obs)
        with torch.no_grad():
            action = self._actor.predict(obs, deterministic=True)
        return np.clip(action.numpy().reshape(-1), -1.0, 1.0).astype(np.float32)


class Constant:
    def __init__(self, action):
        self._action = action

    def predict(self, observation):
        return self._action


def episode(policy, env, rows, seed, design):
    """One evaluated day: net, costs against their budgets, refusals and limits."""
    tick = Decimal(design["kernel"]["tick_size"])
    budgets = {k: float(v) for k, v in design["costs"]["budgets_per_step_mean_over_an_episode"].items()}
    observation, _ = env.reset(seed=seed)
    sums = dict.fromkeys(("c_inventory", "c_notional", "c_margin"), 0.0)
    steps = 0
    refused = {}
    orders = dict.fromkeys(("accepted", "rejected", "canceled", "cancel_rejected", "fills"), 0)
    while True:
        observation, _, terminated, truncated, info = env.step(policy.predict(observation))
        steps += 1
        for key, value in costs(info, rows, tick).items():
            sums[key] += value
        for kind in orders:
            orders[kind] += info["orders"][kind]
        reason = info["refused_because"]
        if reason is not None:
            refused[reason] = refused.get(reason, 0) + 1
        if terminated or truncated:
            break
    attribution = dict(env.attribution())
    limits = design["execution"]["limits"]
    peak = max((abs(Decimal(a)) * Decimal(p) for _, p, a in attribution["positions"]),
               default=Decimal(0))
    means = {k: v / steps for k, v in sums.items()}
    return {
        "seed": seed, "steps": steps, "net": info["net"], "cost_means": means,
        "over_budget": {k: means[k] > budgets[k] for k in budgets},
        "actions_refused": refused, "orders": orders,
        # Every leg the kernel refused: by the hard limits, or by its admission (range).
        "legs_rejected": orders["rejected"],
        "attribution": {k: v for k, v in attribution.items() if k != "positions"},
        "peak_notional": str(peak),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak > Decimal(limits["max_notional"]),
    }


class ZeroResidual:
    def predict(self, observation):
        return np.zeros(4, dtype=np.float32)


def evaluate_job(job):
    torch.set_num_threads(1)
    design, digests = load_design()
    env, rows = build_env(job, digests, job["split"] == "holdout")
    try:
        kind = job["policy"]["kind"]
        policy = (Constant(HOLD) if kind == "hold" else ZeroResidual() if kind == "zero_residual"
                  else Learned(job["policy"]["run_dir"], job["policy"]["model"],
                               job["policy"]["config_sha256"]))
        days = [episode(policy, env, rows, seed, design)
                for seed in design["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets = [Decimal(d["net"]) for d in days]
    keep = ("arm", "candidate", "fee", "regime", "day", "split", "shock", "penalty")
    return {k: job.get(k) for k in keep} | {
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)), "episodes": days}


def train_job(job):
    """One OmniSafe run in its own process, on the pooled training days."""
    design, digests = load_design()
    out = pathlib.Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    days = [["train", r, d, f"day-{d}.json"] for r in regimes_of(design)
            for d in days_of(design, "train")]
    spec = {"data_dir": job["paths"]["data"], "config": base_config(design, job["fee"]),
            "seed": job["seed"], "penalty": job.get("penalty"), "days": days,
            "digests": {f"train/{r}/day-{d}.json": digests[("train", r, d, None)]
                        for _, r, d, _ in days}}
    (out / "job.json").write_text(json.dumps(spec))
    algo = design["algorithm_cfgs"]
    budget = sum(float(v) for v in design["costs"]["budgets_per_step_mean_over_an_episode"].values())
    cfgs = {"seed": job["seed"],
            "train_cfgs": {"total_steps": design["budget_steps"], "device": "cpu",
                           "torch_threads": 1, "vector_env_nums": 1, "parallel": 1},
            "algo_cfgs": {"steps_per_epoch": algo["steps_per_epoch"],
                          "update_iters": algo["update_iters"]},
            "logger_cfgs": {"use_wandb": False, "use_tensorboard": False,
                            "log_dir": str(out / "omnisafe")}}
    if job["arm"] == "cpo":
        cfgs["algo_cfgs"]["cost_limit"] = budget * 8640
    (out / "cfgs.json").write_text(json.dumps(cfgs))
    started = time.perf_counter()
    script = (f"import sys, json, os; sys.path.insert(0, {str(HERE)!r}); import cmdp, omnisafe; "
              f"a = omnisafe.Agent({ALGORITHM[job['arm']]!r}, 'MarketMaking-v0', "
              f"custom_cfgs=json.load(open({str(out / 'cfgs.json')!r}))); a.learn()")
    env = {**os.environ, "M4_08_JOB": str(out / "job.json"), "MUJOCO_GL": "disable"}
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    (out / "train.log").write_text(done.stdout[-20000:] + done.stderr[-20000:])
    if done.returncode != 0:
        raise RuntimeError(f"{out.name} failed: {done.stderr[-2000:]}")
    run_dir = next((out / "omnisafe").glob("*/seed-*"))
    models = sorted((run_dir / "torch_save").glob("epoch-*.pt"),
                    key=lambda p: int(p.stem.split("-")[1]))
    model = models[-1]
    return {"arm": job["arm"], "seed": job["seed"], "penalty": job.get("penalty"),
            "fee": job["fee"], "run_dir": str(run_dir), "model": model.name,
            "checkpoint_sha256": sha256(model), "config_sha256": sha256(run_dir / "config.json"),
            "job_sha256": sha256(out / "job.json"), "commit": job["commit"],
            "design_sha256": sha256(HERE / "design.json"),
            "lock_sha256": sha256(HERE / "omnisafe-lock.txt"),
            "train_seconds": round(time.perf_counter() - started, 1)}


def pool(jobs, worker, parallel, label):
    """Every job in its own process; results in the order of `jobs`."""
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for done, _ in enumerate(concurrent.futures.as_completed(futures), start=1):
            print(f"{label}: {done}/{len(jobs)} after {round(time.perf_counter() - started)} s",
                  file=sys.stderr, flush=True)
        return [future.result() for future in futures]


def mean_daily(rows):
    return sum((Decimal(r["daily_value"]) for r in rows), Decimal(0)) / len(rows)


def group(rows, key):
    out = {}
    for row in rows:
        out.setdefault(key(row), []).append(row)
    return out


class Context:
    def __init__(self, data, shocks, out, commit):
        self.design, self.digests = load_design()
        self.paths = {"data": data, "shocks": shocks}
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.parallel = self.design["compute"]["max_parallel_runs"]
        self.regimes = regimes_of(self.design)
        self.sigmas = {r: volatility(self.paths, self.design, self.digests, r) for r in self.regimes}

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "shocks_sha256": sha256(HERE / "shocks.json"),
                "lock_sha256": sha256(HERE / "omnisafe-lock.txt"),
                "volatility_from_training_days": self.sigmas}

    def job(self, arm, fee, split, regime, day, shock, candidate, policy, point=None, penalty=None):
        return {"arm": arm, "fee": fee, "split": split, "regime": regime, "day": day,
                "shock": shock, "candidate": candidate, "policy": policy, "point": point,
                "penalty": penalty, "paths": self.paths}

    def classical_jobs(self, fee, split, days, shocks=(None,), chosen=None):
        jobs = []
        for shock in shocks:
            for regime in self.regimes:
                for day in days:
                    jobs.append(self.job("no_trade", fee, split, regime, day, shock, "hold",
                                         {"kind": "hold"}))
                    for point in grid_points(self.design["arms"]["avellaneda_stoikov"]["grid"],
                                             self.sigmas[regime]):
                        if chosen is not None and chosen[regime] != point:
                            continue
                        jobs.append(self.job("avellaneda_stoikov", fee, split, regime, day, shock,
                                             json.dumps(point, sort_keys=True),
                                             {"kind": "zero_residual"}, point))
        return jobs

    def learned_jobs(self, runs, fee, split, days, shocks=(None,)):
        return [self.job(m["arm"], fee, split, regime, day, shock,
                         f"seed{m['seed']}" + (f"-penalty{m['penalty']}" if m["penalty"] else ""),
                         {"kind": "learned", "run_dir": m["run_dir"], "model": m["model"],
                          "config_sha256": m["config_sha256"]},
                         penalty=m["penalty"])
                for m in runs for shock in shocks for regime in self.regimes for day in days]

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path

    def read(self, name):
        return json.loads((self.out / f"{name}.json").read_text())


def fee_rule(ctx):
    report = ctx.header("fee_rule")
    started = time.perf_counter()
    rule = ctx.design["execution"]
    rows = []
    for fee in rule["fee_candidates"]:
        rows += pool([j for j in ctx.classical_jobs(fee, "train", days_of(ctx.design, "train"))
                      if j["arm"] == "avellaneda_stoikov"], evaluate_job, ctx.parallel, f"fee {fee}")
    best = {}
    for fee, part in group(rows, lambda r: r["fee"]).items():
        points = group(part, lambda r: r["candidate"])
        top = max(points, key=lambda c: (mean_daily(points[c]), c))
        best[fee] = {"point": json.loads(top), "mean_daily_net": str(mean_daily(points[top]))}
    positive = [f for f in rule["fee_candidates"] if Decimal(best[f]["mean_daily_net"]) > 0]
    fee = max(positive, key=Decimal) if positive else rule["secondary_fee_rate"]
    report.update({"rows": rows, "best_per_fee": best, "fee": fee, "profitable": bool(positive),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("fee_rule", report)
    return report


def classical_selection(ctx):
    report = ctx.header("classical_selection")
    started = time.perf_counter()
    fee = ctx.read("fee_rule")["fee"]
    rows = pool(ctx.classical_jobs(fee, "validation", days_of(ctx.design, "validation")),
                evaluate_job, ctx.parallel, "classical")
    chosen = {}
    for regime, part in group([r for r in rows if r["arm"] == "avellaneda_stoikov"],
                              lambda r: r["regime"]).items():
        points = group(part, lambda r: r["candidate"])
        chosen[regime] = json.loads(max(points, key=lambda c: (mean_daily(points[c]), c)))
    report.update({"fee": fee, "rows": rows, "selected": chosen,
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("classical_selection", report)
    return report


def training(ctx):
    report = ctx.header("training")
    started = time.perf_counter()
    fee = ctx.read("fee_rule")["fee"]
    jobs = []
    for arm in LEARNED:
        penalties = ctx.design["arms"]["ppo_penalized"]["lambda_grid"] if arm == "ppo_penalized" else [None]
        for penalty in penalties:
            for seed in ctx.design["training_seeds"]:
                name = f"{arm}-seed{seed}" + (f"-penalty{penalty}" if penalty else "")
                jobs.append({"arm": arm, "seed": seed, "penalty": penalty, "fee": fee,
                             "paths": ctx.paths, "commit": ctx.commit,
                             "out": str(ctx.out / "runs" / name)})
    report.update({"fee": fee, "runs": pool(jobs, train_job, ctx.parallel, "training"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("training", report)
    return report


def penalty_selection(ctx):
    report = ctx.header("penalty_selection")
    started = time.perf_counter()
    fee = ctx.read("fee_rule")["fee"]
    runs = [m for m in ctx.read("training")["runs"] if m["arm"] == "ppo_penalized"]
    rows = pool(ctx.learned_jobs(runs, fee, "validation", days_of(ctx.design, "validation")),
                evaluate_job, ctx.parallel, "penalties")
    budget = sum(float(v) for v in ctx.design["costs"]["budgets_per_step_mean_over_an_episode"].values())
    summary = {}
    for penalty, part in group(rows, lambda r: r["penalty"]).items():
        cost = np.mean([sum(e["cost_means"][k] for k in ("c_inventory", "c_notional"))
                        for r in part for e in r["episodes"]])
        summary[penalty] = {"mean_daily_net": str(mean_daily(part)), "mean_cost_per_step": cost,
                            "within_budget": bool(cost <= budget)}
    within = [p for p, s in summary.items() if s["within_budget"]]
    chosen = (max(within, key=lambda p: Decimal(summary[p]["mean_daily_net"])) if within
              else min(summary, key=lambda p: summary[p]["mean_cost_per_step"]))
    report.update({"fee": fee, "rows": rows, "summary": summary, "selected": chosen,
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("penalty_selection", report)
    return report


STEPS = ("fee_rule", "classical_selection", "training", "penalty_selection")


def check_run(m, when):
    """A trained run's checkpoint and configuration, against the hashes training recorded."""
    run_dir = pathlib.Path(m["run_dir"])
    if sha256(run_dir / "torch_save" / m["model"]) != m["checkpoint_sha256"]:
        raise ValueError(f"{m['run_dir']} changed after {when}")
    if sha256(run_dir / "config.json") != m["config_sha256"]:
        raise ValueError(f"{m['run_dir']}/config.json changed after {when}")


def freeze(ctx):
    reports = {name: ctx.read(name) for name in STEPS}
    for m in reports["training"]["runs"]:
        check_run(m, "training")
    chosen = reports["penalty_selection"]["selected"]
    runs = [m for m in reports["training"]["runs"]
            if m["arm"] != "ppo_penalized" or m["penalty"] == chosen]
    manifest = {**ctx.header("freeze"),
                **{f"{name}_sha256": sha256(ctx.out / f"{name}.json") for name in STEPS},
                "fee": reports["fee_rule"]["fee"],
                "classical_selected": reports["classical_selection"]["selected"],
                "penalty_selected": chosen, "runs": runs}
    path = ctx.write("frozen", manifest)
    return {"frozen_sha256": sha256(path), "runs": len(runs)}


def holdout(ctx, frozen_sha256):
    path = ctx.out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    if (ctx.out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    frozen = json.loads(path.read_text())
    for key, name in (("design_sha256", "design.json"), ("shocks_sha256", "shocks.json"),
                      ("lock_sha256", "omnisafe-lock.txt")):
        if frozen[key] != sha256(HERE / name):
            raise ValueError(f"{name} changed after the freeze")
    for name in STEPS:
        if frozen[f"{name}_sha256"] != sha256(ctx.out / f"{name}.json"):
            raise ValueError(f"{name}.json changed after the freeze")
    for m in frozen["runs"]:
        check_run(m, "the freeze")
    report = ctx.header("holdout")
    report["frozen_sha256"] = frozen_sha256
    started = time.perf_counter()
    days = days_of(ctx.design, "holdout")
    shocks = [None] + ctx.design["shocks"]["shocks"]
    fee, secondary = frozen["fee"], ctx.design["execution"]["secondary_fee_rate"]
    jobs = ctx.classical_jobs(fee, "holdout", days, shocks, frozen["classical_selected"])
    jobs += ctx.learned_jobs(frozen["runs"], fee, "holdout", days, shocks)
    if secondary != fee:
        jobs += ctx.classical_jobs(secondary, "holdout", days, (None,), frozen["classical_selected"])
        jobs += ctx.learned_jobs(frozen["runs"], secondary, "holdout", days)
    report.update({"rows": pool(jobs, evaluate_job, ctx.parallel, "holdout"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("holdout", report)
    return report


def main():
    step, data, shocks, out, commit = sys.argv[1:6]
    ctx = Context(data, shocks, out, commit)
    started = time.perf_counter()
    if step == "freeze":
        report = freeze(ctx)
        print(json.dumps(report))
    elif step == "holdout":
        report = holdout(ctx, sys.argv[6])
    elif step in STEPS:
        report = globals()[step](ctx)
    else:
        raise SystemExit(f"unknown step {step!r}")
    print(json.dumps({"step": step, "seconds": report.get("seconds"),
                      "total": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
