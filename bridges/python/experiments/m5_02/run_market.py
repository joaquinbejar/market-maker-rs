"""Parts B and C of #107: HSN against temporal controls in our simulator and in our kernel.

From `bridges/python`, with `<part>` either `market` (B, the #102 queue-reactive days) or
`causal` (C, the Hawkes days of `examples/hawkes_days.rs`):

    python experiments/m5_02/run_market.py train <part> <data dir> <out dir> <commit> <parallel>
    python experiments/m5_02/run_market.py select <part> <data dir> <out dir> <commit> <parallel>
    python experiments/m5_02/run_market.py freeze <part> <out dir>
    python experiments/m5_02/run_market.py holdout <part> <data dir> <out dir> <commit> <frozen sha256> <parallel>
    python experiments/m5_02/run_market.py report <part> <out dir> <intervals binary>

Every day's rows are checked against their digests before use. `select` reads validation
only. `freeze` hashes every checkpoint, its metadata, the selection and the design, and
`holdout` refuses anything whose hash differs, reads the holdout once and refuses a second
run. The execution, limits and fees are the same for every arm.

SYNTHETIC: B reuses the #102 days, whose holdout #102 has already read; C's flow and impact
are made up.
"""

import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import pathlib
import subprocess
import sys
import time
from decimal import Decimal

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.flop_counter import FlopCounterMode

from mm_rl_bridge import MaskableMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import ZeroResidual, versions

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nets  # noqa: E402
import window  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("m4_05_run", HERE.parent / "m4_05" / "run.py")
m4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m4)

HOLD = [0, 0, 0, 0, 0]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    """This design, the #102 design it builds on, and the digests of the part's days."""
    design = json.loads((HERE / "design.json").read_text())
    base, _ = m4.load_design()
    pins = design["market"]["pins"]
    for name, path in (("m4_05_design", HERE.parent / "m4_05" / "design.json"),
                       ("m4_05_digests", HERE.parent / "m4_05" / "digests.json")):
        if sha256(path) != pins[name]:
            raise ValueError(f"{name} is not the pinned one")
    return design, base


def digests_of(part):
    if part == "market":
        _, by_day = m4.load_design()
        return {(regime, day): {**r, "group": regime} for (regime, day), r in by_day.items()}
    records = json.loads((HERE / "causal_digests.json").read_text())
    return {(r["setting"], r["day"]): {**r, "group": r["setting"]} for r in records["days"]}


def section(design, part):
    return design[part]


def groups_of(design, base, part):
    if part == "market":
        return m4.regimes_of(base)
    return [s["name"] for s in design["causal"]["settings"]]


def days_of(design, base, part, split):
    splits = base["splits"] if part == "market" else design["causal"]["splits"]
    first, last = splits[split]
    return list(range(first, last + 1))


def read_rows(part, data_dir, digests, group, day, split, holdout_allowed=False):
    if split == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    record = digests[(group, day)]
    if record["split"] != split:
        raise ValueError(f"{group} day {day} is in {record['split']}, not {split}")
    path = pathlib.Path(data_dir) / split / group / f"day-{day}.json"
    return [tuple(row) for row in json.loads(path.read_text())], record


def config_of(design, base, encoding):
    """The #102 configuration with this design's execution: the same for every arm."""
    shared = design["execution"]
    grid = base["arms"]["maskable_ppo"]["grid"]
    if encoding == "maskable":
        config = m4.base_config(base, "main", "mask", grid)
    else:
        config = m4.base_config(base, "main", "reject", {"bid_offsets": [0], "ask_offsets": [0], "sizes": [1]})
    config["fill_model"] = dict(shared["fill_model"])
    config["latency"] = dict(shared["latency"])
    return config


def checked(env, record):
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{record['group']} day {record['day']} has other rows than its digest")
    return env


def build_env(job, digests, group, day, split, holdout_allowed=False):
    design, base = load()
    rows, record = read_rows(job["part"], job["data_dir"], digests, group, day, split, holdout_allowed)
    if job["encoding"] == "residual":
        config = config_of(design, base, "residual")
        config["residual"] = {
            "baseline": m4.baseline_config(job["baseline_kind"], job["baselines"][group], day),
            # A zero residual over the classical quote: the bounds are #102's and never bind.
            "price_ticks": 4,
            "size_lots": 1,
            "default_order_size": "1",
        }
        return checked(ResidualMarketMakingEnv(config, rows), record)
    env = checked(MaskableMarketMakingEnv(config_of(design, base, "maskable"), rows), record)
    return window.WindowEnv(env, design["window"]["buffer"])


def layout(design):
    return nets.Layout(**window.layout_dims(design["window"]["buffer"]))


def arm_spec(design, name):
    return dict(design["arms"][name])


def train_job(job):
    torch.set_num_threads(1)
    design, base = load()
    digests = digests_of(job["part"])
    set_random_seed(job["seed"])
    pairs = [(g, d) for g in job["groups"] for d in days_of(design, base, job["part"], "train")]
    envs = [build_env(job, digests, g, d, "train") for g, d in pairs]
    cycle = m4.DayCycle(envs, job["seed"])
    arm = arm_spec(design, job["arm"])
    ppo = design["ppo"]
    start = time.perf_counter()
    model = MaskablePPO(
        "MlpPolicy",
        DummyVecEnv([lambda: Monitor(cycle)]),
        seed=job["seed"],
        device="cpu",
        policy_kwargs=nets.policy_kwargs(arm, layout(design), design["widths"][job["arm"]]),
        **ppo,
    )
    model.learn(total_timesteps=design["budget_steps"], progress_bar=False)
    seconds = time.perf_counter() - start
    path = pathlib.Path(job["path"])
    model.save(path)
    extractor = model.policy.features_extractor
    metadata = {
        "part": job["part"],
        "arm": job["arm"],
        "groups": job["groups"],
        "seed": job["seed"],
        "steps": design["budget_steps"],
        "width": design["widths"][job["arm"]],
        "parameters": nets.parameter_count(model.policy),
        "training_days": [list(p) for p in pairs],
        "training_digests": [e.unwrapped.bridge_info["dataset_digest"] for e in envs],
        "commit": job["commit"],
        "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(),
        "train_seconds": round(seconds, 2),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")),
        "path": str(path),
    }
    if isinstance(extractor, nets.Hsn):
        metadata["learned"] = {"alphas": extractor.alphas().detach().tolist(),
                               "betas": extractor.betas().detach().tolist()}
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def episode(policy, env, seed, limits):
    """One evaluated day: the exact net result, its attribution, and what the gate refused."""
    observation, _ = env.reset(seed=seed)
    refusals = steps = quotes = 0
    masked = hasattr(env, "action_masks") and not isinstance(env, ResidualMarketMakingEnv)
    while True:
        if masked:
            action, _ = policy.predict(observation, deterministic=True, action_masks=env.action_masks())
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        quotes += int(np.asarray(action).ravel()[0]) if masked else 1
        refusals += info["refused_because"] is not None
        if terminated or truncated:
            break
    attribution = dict(env.unwrapped.attribution())
    peak = max((abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
               default=Decimal(0))
    return {
        "seed": seed,
        "steps": steps,
        "quote_share": quotes / steps,
        "net": info["net"],
        "attribution": {k: v for k, v in attribution.items() if k != "positions"},
        "refusals": refusals,
        "peak_notional": str(peak),
        "time_beyond_5_lots": m4.time_beyond(attribution, 5),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak > Decimal(limits["max_notional"]),
    }


def evaluate_job(job):
    torch.set_num_threads(1)
    design, base = load()
    digests = digests_of(job["part"])
    env = build_env(job, digests, job["group"], job["day"], job["split"], job["split"] == "holdout")
    try:
        kind = job["policy"]["kind"]
        if kind == "hold":
            policy = m4.Constant(HOLD)
        elif kind == "zero_residual":
            policy = ZeroResidual()
        else:
            meta = job["policy"]["metadata"]
            path = pathlib.Path(meta["path"]).with_suffix(".zip")
            if sha256(path) != meta["checkpoint_sha256"]:
                raise ValueError(f"{path} is not the checkpoint its metadata names")
            policy = MaskablePPO.load(path, device="cpu")
            policy.policy.set_training_mode(False)
        days = [episode(policy, env, s, base["execution"]["limits"]) for s in design["evaluation_seeds"]]
    finally:
        env.close()
    nets_ = [Decimal(d["net"]) for d in days]
    return {
        "arm": job["arm"],
        "candidate": job["candidate"],
        "group": job["group"],
        "day": job["day"],
        "split": job["split"],
        "daily_value": str(sum(nets_, Decimal(0)) / len(nets_)),
        "episodes": days,
    }


def pool(jobs, worker, parallel, label):
    return m4.pool(jobs, worker, parallel, label)


class Context:
    def __init__(self, part, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.part = part
        self.design, self.base = load()
        self.digests = digests_of(part)
        self.data_dir = data_dir
        self.out = pathlib.Path(out_dir) / part
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.groups = groups_of(self.design, self.base, part)
        self._sigmas = None

    @property
    def sigmas(self):
        if self._sigmas is None:
            self._sigmas = {g: self.volatility(g) for g in self.groups}
        return self._sigmas

    def volatility(self, group):
        """#102's volatility rule over this part's training days."""
        tick = Decimal(self.base["kernel"]["tick_size"])
        changes = []
        for day in days_of(self.design, self.base, self.part, "train"):
            rows, _ = read_rows(self.part, self.data_dir, self.digests, group, day, "train")
            mids = [Decimal(row[1] + row[3]) / 2 * tick for row in rows]
            changes.extend(float(b - a) for a, b in zip(mids, mids[1:]))
        mean = sum(changes) / len(changes)
        variance = sum((c - mean) ** 2 for c in changes) / len(changes)
        seconds = self.base["simulator"]["sample_ms"] / 1000
        return f"{(variance * m4.SECONDS_PER_YEAR / seconds) ** 0.5:.4f}"

    def header(self, step):
        return {
            "step": step,
            "part": self.part,
            "commit": self.commit,
            "data": self.design[self.part]["data"],
            "design_sha256": sha256(HERE / "design.json"),
            "versions": versions(),
            "evaluation_seeds": self.design["evaluation_seeds"],
            "volatility_from_training_days": self._sigmas,
        }

    def training_groups(self):
        """B trains one policy on every regime pooled; C one policy per impact setting."""
        if self.part == "market":
            return [self.groups]
        return [[g] for g in self.groups]

    def train_jobs(self):
        jobs = []
        for groups in self.training_groups():
            for arm in self.design["arms"]:
                for seed in self.design["training_seeds"]:
                    tag = "pooled" if self.part == "market" else groups[0]
                    name = f"{arm}-{tag}-seed{seed}"
                    jobs.append({"part": self.part, "arm": arm, "groups": groups, "seed": seed,
                                 "encoding": "maskable", "data_dir": self.data_dir,
                                 "commit": self.commit,
                                 "path": str(self.out / "checkpoints" / name)})
        return jobs

    def runs(self):
        out = []
        for job in self.train_jobs():
            meta = pathlib.Path(job["path"]).with_suffix(".json")
            if not meta.exists():
                raise FileNotFoundError(f"{meta} has not been trained")
            out.append(json.loads(meta.read_text()))
        return out

    def reference_jobs(self, split, days):
        jobs = []
        refs = self.design["references"]
        for group in self.groups:
            for day in days:
                common = {"part": self.part, "group": group, "day": day, "split": split,
                          "data_dir": self.data_dir}
                jobs.append({**common, "arm": "no_trade", "candidate": "hold",
                             "encoding": "maskable", "policy": {"kind": "hold"}})
                for kind in ("avellaneda_stoikov", "glft"):
                    point = {**refs[kind], "volatility": self.sigmas[group]}
                    jobs.append({**common, "arm": kind, "candidate": json.dumps(point, sort_keys=True),
                                 "encoding": "residual", "baseline_kind": kind,
                                 "baselines": {group: point}, "policy": {"kind": "zero_residual"}})
        jobs.sort(key=lambda job: job["arm"] != "glft")
        return jobs

    def learned_jobs(self, runs, split, days):
        jobs = []
        for meta in runs:
            for group in meta["groups"]:
                for day in days:
                    jobs.append({"part": self.part, "group": group, "day": day, "split": split,
                                 "data_dir": self.data_dir, "arm": meta["arm"],
                                 "candidate": f"seed{meta['seed']}", "encoding": "maskable",
                                 "policy": {"kind": "frozen", "metadata": meta}})
        return jobs

    def write(self, name, report):
        path = self.out / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; a step runs once")
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return sha256(path)


def check_widths(design):
    """Each arm's width is the one the parameter rule gives, within its tolerance."""
    import gymnasium as gym
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    lay = layout(design)
    space = gym.spaces.Box(-np.inf, np.inf, shape=(lay.size,), dtype=np.float32)
    actions = gym.spaces.MultiDiscrete(design["action_dims"])
    rule = design["parameters"]
    found = {}
    for name in design["arms"]:
        best = None
        for width in range(rule["min_width"], rule["max_width"] + 1):
            policy = MaskableActorCriticPolicy(space, actions, lambda _: 3e-4,
                                               **nets.policy_kwargs(arm_spec(design, name), lay, width))
            count = nets.parameter_count(policy)
            if best is None or abs(count - rule["target"]) < abs(best[1] - rule["target"]):
                best = (width, count)
        if abs(best[1] - rule["target"]) > rule["tolerance"] * rule["target"]:
            raise ValueError(f"{name}: no width within the tolerance ({best})")
        found[name] = best
        if design["widths"].get(name) != best[0]:
            raise SystemExit(f"design width of {name} is {design['widths'].get(name)}, rule gives {best[0]}")
    return found


def mean_daily(rows):
    values = [Decimal(r["daily_value"]) for r in rows]
    return sum(values, Decimal(0)) / len(values)


def train(ctx, parallel):
    check_widths(ctx.design)
    if (ctx.out / "frozen.json").exists():
        raise SystemExit("already frozen")
    todo = [j for j in ctx.train_jobs() if not pathlib.Path(j["path"]).with_suffix(".json").exists()]
    for meta in pool(todo, train_job, parallel, f"train {ctx.part}"):
        print(meta["arm"], meta["groups"], meta["seed"], meta["train_seconds"], flush=True)


def select(ctx, parallel):
    """Every arm on validation; the best control is chosen here, before the holdout."""
    days = days_of(ctx.design, ctx.base, ctx.part, "validation")
    rows = pool(ctx.reference_jobs("validation", days) + ctx.learned_jobs(ctx.runs(), "validation", days),
                evaluate_job, parallel, f"validation {ctx.part}")
    controls = [a for a in ctx.design["arms"] if a != "hsn"]
    best = {}
    for groups in ctx.training_groups():
        key = "pooled" if ctx.part == "market" else groups[0]
        scores = {a: mean_daily([r for r in rows if r["arm"] == a and r["group"] in groups]) for a in controls}
        best[key] = max(controls, key=lambda a: (scores[a], a))
    report = {**ctx.header("select"), "rows": rows, "best_control": best}
    print(json.dumps(best), ctx.write("validation", report))


def freeze(ctx):
    runs = ctx.runs()
    entries = {}
    for meta in runs:
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if sha256(zip_path) != meta["checkpoint_sha256"]:
            raise SystemExit(f"{zip_path} changed")
        if meta["design_sha256"] != sha256(HERE / "design.json"):
            raise SystemExit(f"{zip_path} was trained under another design")
        entries[pathlib.Path(meta["path"]).name] = {
            "checkpoint": meta["checkpoint_sha256"],
            "metadata": sha256(zip_path.with_suffix(".json")),
        }
    frozen = {
        "design_sha256": sha256(HERE / "design.json"),
        "validation_sha256": sha256(ctx.out / "validation.json"),
        "runs": entries,
    }
    print(ctx.write("frozen", frozen))


def holdout(ctx, frozen_sha, parallel):
    if sha256(ctx.out / "frozen.json") != frozen_sha:
        raise SystemExit("the frozen manifest differs")
    frozen = json.loads((ctx.out / "frozen.json").read_text())
    if frozen["design_sha256"] != sha256(HERE / "design.json"):
        raise SystemExit("the design differs from the frozen one")
    if frozen["validation_sha256"] != sha256(ctx.out / "validation.json"):
        raise SystemExit("the selection differs from the frozen one")
    runs = ctx.runs()
    for meta in runs:
        name = pathlib.Path(meta["path"]).name
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if frozen["runs"][name]["checkpoint"] != sha256(zip_path):
            raise SystemExit(f"{name} changed after the freeze")
        if frozen["runs"][name]["metadata"] != sha256(zip_path.with_suffix(".json")):
            raise SystemExit(f"{name}'s metadata changed after the freeze")
    if (ctx.out / "holdout.json").exists():
        raise SystemExit("the holdout runs once")
    days = days_of(ctx.design, ctx.base, ctx.part, "holdout")
    rows = pool(ctx.reference_jobs("holdout", days) + ctx.learned_jobs(runs, "holdout", days),
                evaluate_job, parallel, f"holdout {ctx.part}")
    print(ctx.write("holdout", {**ctx.header("holdout"), "frozen_sha256": frozen_sha, "rows": rows}))


def audit(ctx):
    """Two loads give the same actions on a fixed set of observations; cost per decision."""
    design = ctx.design
    first_group = ctx.groups[0]
    day = days_of(design, ctx.base, ctx.part, "validation")[0]
    job = {"part": ctx.part, "data_dir": ctx.data_dir, "encoding": "maskable"}
    env = build_env(job, ctx.digests, first_group, day, "validation")
    observations, masks = [], []
    obs, _ = env.reset(seed=design["evaluation_seeds"][0])
    while len(observations) < design["audit"]["observations"]:
        observations.append(obs)
        masks.append(env.action_masks())
        obs, _, terminated, truncated, _ = env.step(HOLD)
        if terminated or truncated:
            obs, _ = env.reset(seed=design["evaluation_seeds"][0])
    env.close()
    observations = np.asarray(observations, dtype=np.float32)
    masks = np.asarray(masks)
    out = {}
    for meta in ctx.runs():
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        a = MaskablePPO.load(path, device="cpu")
        b = MaskablePPO.load(path, device="cpu")
        acts_a, _ = a.predict(observations, deterministic=True, action_masks=masks)
        acts_b, _ = b.predict(observations, deterministic=True, action_masks=masks)
        one = torch.as_tensor(observations[:1])
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            a.policy.get_distribution(one)
        timings = []
        with torch.no_grad():
            for row in observations[: design["audit"]["timed"]]:
                start = time.perf_counter_ns()
                a.policy.get_distribution(torch.as_tensor(row[None, :]))
                timings.append(time.perf_counter_ns() - start)
        out[path.stem] = {"identical": bool(np.array_equal(acts_a, acts_b)),
                          "flops_per_decision": int(counter.get_total_flops()),
                          "median_inference_us": float(np.median(timings)) / 1000.0,
                          "parameters": meta["parameters"],
                          "train_seconds": meta["train_seconds"],
                          "learned": meta.get("learned")}
    return out


def decide(against, against_none, breach, hsn_drawdown, control_drawdown, factor):
    """The design's decision rule, from the intervals and the risk checks."""
    equivalent = Decimal(against_none["lower"]) <= 0 <= Decimal(against_none["upper"])
    dd_ok = hsn_drawdown <= factor * control_drawdown
    if Decimal(against["lower"]) > 0 and not breach and dd_ok and not equivalent:
        decision = "hsn_helps"
    elif Decimal(against["mean"]) > 0:
        decision = "revise"
    else:
        decision = "no_gain"
    return {"against_control": against, "against_no_trade": against_none,
            "equivalent_to_no_trade": equivalent, "breach": breach, "drawdown_ok": dd_ok,
            "decision": decision}


def report(ctx, binary):
    """Tables, paired intervals and the design's decision, per group and pooled."""
    design = ctx.design
    holdout_rows = json.loads((ctx.out / "holdout.json").read_text())["rows"]
    best = json.loads((ctx.out / "validation.json").read_text())["best_control"]
    days = days_of(design, ctx.base, ctx.part, "holdout")

    def daily(arm, group):
        """A learned arm's day is the mean over its seeds, as in #102."""
        out = []
        for day in days:
            values = [Decimal(r["daily_value"]) for r in holdout_rows
                      if r["arm"] == arm and r["group"] == group and r["day"] == day]
            out.append(sum(values, Decimal(0)) / len(values))
        return out

    scopes = [([g], g) for g in ctx.groups]
    if ctx.part == "market":
        scopes.append((ctx.groups, "pooled"))
    series = {}
    for groups, scope in scopes:
        control = best["pooled"] if ctx.part == "market" else best[groups[0]]
        hsn = [v for g in groups for v in daily("hsn", g)]
        for other in (control, "no_trade"):
            theirs = [v for g in groups for v in daily(other, g)]
            series[f"{scope}/hsn-minus-{other}"] = [str(a - b) for a, b in zip(hsn, theirs)]
    (ctx.out / "series.json").write_text(json.dumps({"bootstrap": design["bootstrap"], "series": series}))
    intervals = json.loads(subprocess.run([binary, str(ctx.out / "series.json")], check=True,
                                          capture_output=True, text=True).stdout)
    rule = design["decision_rule"]
    decisions = {}
    for groups, scope in scopes:
        control = best["pooled"] if ctx.part == "market" else best[groups[0]]
        against = intervals[f"{scope}/hsn-minus-{control}"]
        against_none = intervals[f"{scope}/hsn-minus-no_trade"]
        rows = [r for r in holdout_rows if r["group"] in groups]
        breach = any(e["limit_breach"] for r in rows if r["arm"] == "hsn" for e in r["episodes"])

        def drawdown(arm):
            values = [Decimal(e["attribution"]["max_drawdown"]) for r in rows if r["arm"] == arm
                      for e in r["episodes"]]
            return sum(values, Decimal(0)) / len(values)

        decisions[scope] = {"control": control,
                            **decide(against, against_none, breach, drawdown("hsn"),
                                     drawdown(control), Decimal(rule["drawdown_factor"]))}
    out = {**ctx.header("report"), "decisions": decisions, "intervals": intervals,
           "audit": audit(ctx)}
    print(json.dumps(decisions, indent=2), ctx.write("report", out))


def main():
    step, part = sys.argv[1], sys.argv[2]
    if part not in ("market", "causal"):
        raise SystemExit("part is market or causal")
    if step == "train":
        train(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), int(sys.argv[6]))
    elif step == "select":
        select(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), int(sys.argv[6]))
    elif step == "freeze":
        freeze(Context(part, None, sys.argv[3], None))
    elif step == "holdout":
        holdout(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), sys.argv[6], int(sys.argv[7]))
    elif step == "report":
        report(Context(part, None, sys.argv[3], None), sys.argv[4])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
