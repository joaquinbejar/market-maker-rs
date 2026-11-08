"""Parts B and C of #107: HSN against temporal controls, on the protocol of #103.

From `bridges/python`, with `<part>` either `market` (B, the 1 s queue-reactive days of
#103) or `causal` (C, the Hawkes days of `examples/hawkes_days.rs`):

    python experiments/m5_02/run_market.py classical causal <data dir> <out dir> <commit> <parallel>
    python experiments/m5_02/run_market.py train <part> <data dir> <out dir> <commit> <parallel>
    python experiments/m5_02/run_market.py select <part> <data dir> <out dir> <commit> <parallel>
    python experiments/m5_02/run_market.py freeze <part> <out dir>
    python experiments/m5_02/run_market.py holdout <part> <data dir> <out dir> <commit> <frozen sha256> <parallel> [--recover <reason>]
    python experiments/m5_02/run_market.py report <part> <data dir> <out dir> <intervals binary>

Everything #103 fixes is taken from its runner (`experiments/m4_06/run.py`) and its wrappers
(`mm_rl_bridge.wrappers`), not copied: the configuration, the Box encoding with the reject
policy, the decision interval, the arms `mlp`, `ema`, `history` and `lstm`, the training loop
over pooled days, the evaluated episode. This part adds two arms, `gru_window` and `hsn`, over
a window of the last decisions, at one decision per row.

Every day's rows are checked against their digests before use. `select` reads validation
only. `freeze` hashes every checkpoint, its metadata, the selection and the design;
`holdout` refuses anything whose hash differs, reads the holdout once and refuses a second
run. SYNTHETIC: B's days are #103's, whose holdout #103 reads first; C's flow is made up.
"""

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from decimal import Decimal

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.flop_counter import FlopCounterMode

from mm_rl_bridge import BoxMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import versions
from mm_rl_bridge.wrappers import DecisionInterval, EmaFeatures, history

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attempt  # noqa: E402
import nets  # noqa: E402
import window  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("m4_06_run", HERE.parent / "m4_06" / "run.py")
m6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m6)

WINDOWED = ("gru_window", "hsn")
EVERY = 1


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    """This design and #103's, with #103's design and digests checked against the pins."""
    design = json.loads((HERE / "design.json").read_text())
    pins = design["pins"]
    for name, path in (("m4_06_design", HERE.parent / "m4_06" / "design.json"),
                       ("m4_06_digests", HERE.parent / "m4_06" / "digests.json")):
        if sha256(path) != pins[name]:
            raise ValueError(f"{name} is not the pinned one")
    base, _ = m6.load_design()
    return design, base


def digests_of(part):
    if part == "market":
        _, by_day = m6.load_design()
        return {key: {**r, "group": key[0]} for key, r in by_day.items()}
    records = json.loads((HERE / "causal_digests.json").read_text())
    return {(r["setting"], r["day"]): {**r, "group": r["setting"]} for r in records["days"]}


def groups_of(design, base, part):
    if part == "market":
        return m6.regimes_of(base)
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


def volatility(part, data_dir, design, base, digests, group):
    """#103's volatility rule over the part's training days."""
    tick = Decimal(base["kernel"]["tick_size"])
    changes = []
    for day in days_of(design, base, part, "train"):
        rows, _ = read_rows(part, data_dir, digests, group, day, "train")
        mids = [Decimal(r[1] + r[3]) / 2 * tick for r in rows]
        changes.extend(float(b - a) for a, b in zip(mids, mids[1:]))
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)
    seconds = base["simulator"]["sample_ms"] / 1000
    return f"{(variance * m6.SECONDS_PER_YEAR / seconds) ** 0.5:.4f}"


def build_env(job, digests, group, day, split, holdout_allowed=False):
    """One day for one arm: #103's configuration and wrappers, then the window if windowed."""
    design, base = load()
    rows, record = read_rows(job["part"], job["data_dir"], digests, group, day, split,
                             holdout_allowed)
    config = m6.base_config(base, job["fee"])
    # The queue standard for 1 s rows, over #103's minimum time in queue (see the design).
    config["fill_model"] = dict(job.get("fill_model") or design["execution"]["fill_model"])
    arm = job["arm"]
    if arm in m6.CLASSICAL:
        config["residual"] = {"baseline": m6.baseline_config(arm, job["point"], day),
                              "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        env = ResidualMarketMakingEnv(config, rows)
    else:
        env = BoxMarketMakingEnv(config, rows)
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{group} day {day} has other rows than its digest")
    env = DecisionInterval(env, EVERY)
    if arm == "ema":
        env = EmaFeatures(env, tuple(base["arms"]["ema"]["half_lives_decisions"]))
    elif arm == "history":
        env = history(env, base["arms"]["history"]["length"])
    elif arm in WINDOWED:
        env = window.WindowEnv(env, design["window"]["buffer"])
    return env


def layout(design):
    return nets.Layout(**window.layout_dims(design["window"]["buffer"]))


def model_of(design, base, arm, env, seed):
    hyper = base["hyperparameters"]
    kwargs = {k: hyper[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef")}
    if arm in WINDOWED:
        policy_kwargs = nets.policy_kwargs(design["arms"][arm], layout(design), design["widths"][arm])
        return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                   policy_kwargs=policy_kwargs, **kwargs)
    algorithm, policy = m6.model_class(arm)
    return algorithm(policy, env, seed=seed, device="cpu", verbose=0, **kwargs)


def load_model(arm, path):
    algorithm = PPO if arm in WINDOWED else m6.model_class(arm)[0]
    model = algorithm.load(path, device="cpu")
    model.policy.set_training_mode(False)
    return model


def train_job(job):
    torch.set_num_threads(1)
    design, base = load()
    digests = digests_of(job["part"])
    set_random_seed(job["seed"])
    pairs = [(g, d) for g in job["groups"] for d in days_of(design, base, job["part"], "train")]
    envs = [build_env(job, digests, g, d, "train") for g, d in pairs]
    cycle = m6.DayCycle(envs, job["seed"])
    model = model_of(design, base, job["arm"], DummyVecEnv([lambda: Monitor(cycle)]), job["seed"])
    started = time.perf_counter()
    model.learn(total_timesteps=base["budget_decisions"], progress_bar=False)
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    model.save(path)
    metadata = {
        "part": job["part"], "arm": job["arm"], "groups": job["groups"], "seed": job["seed"],
        "fee": job["fee"], "decisions": base["budget_decisions"], "every": EVERY,
        "parameters": nets.parameter_count(model.policy),
        "spec": list(envs[0].unwrapped.spec_id),
        "config": envs[0].unwrapped.config,
        "training_digests": [e.unwrapped.bridge_info["dataset_digest"] for e in envs],
        "commit": job["commit"], "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(), "train_seconds": round(seconds, 1),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path),
    }
    extractor = model.policy.features_extractor
    if isinstance(extractor, nets.Hsn):
        metadata["learned"] = {"alphas": extractor.alphas().detach().tolist(),
                               "betas": extractor.betas().detach().tolist()}
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def frozen_policy(meta, env):
    """A checkpoint, after checking it is the one trained, for this specification and this
    configuration but for its fee."""
    path = pathlib.Path(meta["path"]).with_suffix(".zip")
    if sha256(path) != meta["checkpoint_sha256"]:
        raise ValueError(f"{path} is not the checkpoint its metadata names")
    base_env = env.unwrapped
    if tuple(meta["spec"]) != base_env.spec_id:
        raise ValueError("a checkpoint of another specification")
    same = {k: v for k, v in meta["config"].items() if k != "fee_rate"}
    if same != {k: v for k, v in base_env.config.items() if k != "fee_rate"}:
        raise ValueError("a checkpoint of another configuration")
    return load_model(meta["arm"], path)


def evaluate_job(job):
    torch.set_num_threads(1)
    design, base = load()
    digests = digests_of(job["part"])
    env = build_env(job, digests, job["group"], job["day"], job["split"], job["split"] == "holdout")
    try:
        kind = job["policy"]["kind"]
        if kind == "hold":
            policy = m6.Constant(m6.HOLD)
        elif kind == "zero_residual":
            policy = m6.ZeroResidual()
        else:
            policy = frozen_policy(job["policy"]["metadata"], env)
        days = [m6.episode(policy, env, s, job["arm"] == "lstm", base)
                for s in base["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets_ = [Decimal(d["net"]) for d in days]
    return {key: job[key] for key in ("arm", "candidate", "fee", "group", "day", "split")} | {
        "daily_value": str(sum(nets_, Decimal(0)) / len(nets_)), "episodes": days}


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
        self.fee = self.design["fee"]["value"]
        self.secondary_fee = self.base["execution"]["secondary_fee_rate"]

    def header(self, step):
        return {"step": step, "part": self.part, "commit": self.commit,
                "data": self.design[self.part]["data"], "fee": self.fee,
                "design_sha256": sha256(HERE / "design.json"), "versions": versions()}

    def training_groups(self):
        """B trains one policy on every regime pooled, as #103; C one per impact setting."""
        return [self.groups] if self.part == "market" else [[g] for g in self.groups]

    def train_jobs(self):
        jobs = []
        for groups in self.training_groups():
            tag = "pooled" if self.part == "market" else groups[0]
            for arm in self.design["arms"]:
                for seed in self.base["training_seeds"]:
                    jobs.append({"part": self.part, "arm": arm, "groups": groups, "seed": seed,
                                 "fee": self.fee, "data_dir": self.data_dir, "commit": self.commit,
                                 "path": str(self.out / "checkpoints" / f"{arm}-{tag}-seed{seed}")})
        # The slow ones first, so that the pool ends together.
        jobs.sort(key=lambda job: job["arm"] != "lstm")
        return jobs

    def runs(self):
        out = []
        for job in self.train_jobs():
            meta = pathlib.Path(job["path"]).with_suffix(".json")
            if not meta.exists():
                raise FileNotFoundError(f"{meta} has not been trained")
            out.append(json.loads(meta.read_text()))
        return out

    def job(self, arm, fee, group, day, split, candidate, policy, point=None):
        return {"part": self.part, "arm": arm, "fee": fee, "group": group, "day": day,
                "split": split, "candidate": candidate, "policy": policy, "point": point,
                "data_dir": self.data_dir}

    def reference_points(self):
        """B: the points #103 selected at one row, pinned in the design. C: this part's own
        selection on its validation days, by #103's rule."""
        if self.part == "market":
            return self.design["references"]["points"]
        return json.loads((self.out / "classical.json").read_text())["selected"]

    def reference_jobs(self, fee, split, days):
        refs = self.reference_points()
        jobs = []
        for group in self.groups:
            for day in days:
                jobs.append(self.job("no_trade", fee, group, day, split, "hold", {"kind": "hold"}))
                for arm in m6.CLASSICAL:
                    point = refs[group][arm]
                    jobs.append(self.job(arm, fee, group, day, split,
                                         json.dumps(point, sort_keys=True),
                                         {"kind": "zero_residual"}, point))
        return jobs

    def learned_jobs(self, runs, fee, split, days):
        return [self.job(m["arm"], fee, g, d, split, f"seed{m['seed']}",
                         {"kind": "frozen", "metadata": m})
                for m in runs for g in m["groups"] for d in days]

    def write(self, name, report):
        path = self.out / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; a step runs once")
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return sha256(path)


def check_widths(design, base):
    """Each windowed arm's width is the one the parameter rule gives: the count of #103's
    `mlp`, within the tolerance."""
    import gymnasium as gym
    from stable_baselines3.common.policies import ActorCriticPolicy

    lay = layout(design)
    rule = design["parameters"]
    actions = gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
    raw = gym.spaces.Box(-1, 1, shape=(2 * window.FEATURES,), dtype=np.float32)
    target = nets.parameter_count(ActorCriticPolicy(raw, actions, lambda _: 3e-4))
    if target != rule["target"]:
        raise SystemExit(f"#103's mlp has {target} parameters, the design says {rule['target']}")
    space = gym.spaces.Box(-np.inf, np.inf, shape=(lay.size,), dtype=np.float32)
    found = {}
    for name in WINDOWED:
        best = None
        for width in range(rule["min_width"], rule["max_width"] + 1):
            policy = ActorCriticPolicy(space, actions, lambda _: 3e-4,
                                       **nets.policy_kwargs(design["arms"][name], lay, width))
            count = nets.parameter_count(policy)
            if best is None or abs(count - target) < abs(best[1] - target):
                best = (width, count)
        if abs(best[1] - target) > rule["tolerance"] * target:
            raise ValueError(f"{name}: no width within the tolerance ({best})")
        if design["widths"].get(name) != best[0]:
            raise SystemExit(f"design width of {name} is {design['widths'].get(name)}, rule gives {best[0]}")
        found[name] = best
    return found


def classical(ctx, parallel):
    """C only: #103's classical grids on C's validation days at the fee, and its rule."""
    if ctx.part != "causal":
        raise SystemExit("B takes #103's classical selection")
    days = days_of(ctx.design, ctx.base, ctx.part, "validation")
    jobs = []
    sigmas = {}
    for group in ctx.groups:
        sigmas[group] = volatility(ctx.part, ctx.data_dir, ctx.design, ctx.base, ctx.digests, group)
        for arm in m6.CLASSICAL:
            for point in m6.grid_points(ctx.base["arms"][arm]["grid"], sigmas[group]):
                for day in days:
                    jobs.append(ctx.job(arm, ctx.fee, group, day, "validation",
                                        json.dumps(point, sort_keys=True),
                                        {"kind": "zero_residual"}, point))
    rows = m6.pool(jobs, evaluate_job, parallel, "classical causal")
    selected = {}
    for group in ctx.groups:
        selected[group] = {}
        for arm in m6.CLASSICAL:
            mine = [r for r in rows if r["group"] == group and r["arm"] == arm]
            candidates = sorted({r["candidate"] for r in mine})
            top = max(candidates, key=lambda c: (mean_daily([r for r in mine if r["candidate"] == c]), c))
            selected[group][arm] = json.loads(top)
    print(json.dumps(selected), ctx.write("classical", {**ctx.header("classical"), "rows": rows,
                                                        "volatility": sigmas, "selected": selected}))


def train(ctx, parallel):
    check_widths(ctx.design, ctx.base)
    if (ctx.out / "frozen.json").exists():
        raise SystemExit("already frozen")
    todo = [j for j in ctx.train_jobs() if not pathlib.Path(j["path"]).with_suffix(".json").exists()]
    for meta in m6.pool(todo, train_job, parallel, f"train {ctx.part}"):
        print(meta["arm"], meta["groups"], meta["seed"], meta["train_seconds"], flush=True)


def mean_daily(rows):
    values = [Decimal(r["daily_value"]) for r in rows]
    return sum(values, Decimal(0)) / len(values)


def select(ctx, parallel):
    """Every arm on validation; the best control is chosen here, before the holdout."""
    days = days_of(ctx.design, ctx.base, ctx.part, "validation")
    rows = m6.pool(ctx.reference_jobs(ctx.fee, "validation", days)
                   + ctx.learned_jobs(ctx.runs(), ctx.fee, "validation", days),
                   evaluate_job, parallel, f"validation {ctx.part}")
    controls = [a for a in ctx.design["arms"] if a != "hsn"]
    best = {}
    for groups in ctx.training_groups():
        key = "pooled" if ctx.part == "market" else groups[0]
        scores = {a: mean_daily([r for r in rows if r["arm"] == a and r["group"] in groups])
                  for a in controls}
        best[key] = max(controls, key=lambda a: (scores[a], a))
    print(json.dumps(best), ctx.write("validation", {**ctx.header("select"), "rows": rows,
                                                     "best_control": best}))


def freeze(ctx):
    entries = {}
    for meta in ctx.runs():
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if sha256(zip_path) != meta["checkpoint_sha256"]:
            raise SystemExit(f"{zip_path} changed")
        if meta["design_sha256"] != sha256(HERE / "design.json"):
            raise SystemExit(f"{zip_path} was trained under another design")
        entries[zip_path.stem] = {"checkpoint": meta["checkpoint_sha256"],
                                  "metadata": sha256(zip_path.with_suffix(".json"))}
    frozen = {"design_sha256": sha256(HERE / "design.json"),
              "validation_sha256": sha256(ctx.out / "validation.json"), "runs": entries}
    if ctx.part == "causal":
        frozen["classical_sha256"] = sha256(ctx.out / "classical.json")
    print(ctx.write("frozen", frozen))


def holdout(ctx, frozen_sha, parallel, recover=None):
    if sha256(ctx.out / "frozen.json") != frozen_sha:
        raise SystemExit("the frozen manifest differs")
    frozen = json.loads((ctx.out / "frozen.json").read_text())
    if frozen["design_sha256"] != sha256(HERE / "design.json"):
        raise SystemExit("the design differs from the frozen one")
    if frozen["validation_sha256"] != sha256(ctx.out / "validation.json"):
        raise SystemExit("the selection differs from the frozen one")
    if ctx.part == "causal" and frozen["classical_sha256"] != sha256(ctx.out / "classical.json"):
        raise SystemExit("the classical selection differs from the frozen one")
    runs = ctx.runs()
    for meta in runs:
        zip_path = pathlib.Path(meta["path"]).with_suffix(".zip")
        entry = frozen["runs"][zip_path.stem]
        if entry["checkpoint"] != sha256(zip_path) or entry["metadata"] != sha256(zip_path.with_suffix(".json")):
            raise SystemExit(f"{zip_path.stem} changed after the freeze")
    if (ctx.out / "holdout.json").exists():
        raise SystemExit("the holdout runs once")
    days = days_of(ctx.design, ctx.base, ctx.part, "holdout")
    jobs = []
    for fee in dict.fromkeys([ctx.fee, ctx.secondary_fee]):
        jobs += ctx.reference_jobs(fee, "holdout", days) + ctx.learned_jobs(runs, fee, "holdout", days)
    # One attempt per freeze, claimed before any job is dispatched (attempt.py).
    attempts = attempt.claim(ctx.out / "holdout.attempt", frozen_sha, recover)
    rows = m6.pool(jobs, evaluate_job, parallel, f"holdout {ctx.part}")
    print(ctx.write("holdout", {**ctx.header("holdout"), "frozen_sha256": frozen_sha,
                                "secondary_fee": ctx.secondary_fee, "attempts": attempts,
                                "rows": rows}))


def decide(against, against_none, breach, hsn_drawdown, control_drawdown, factor):
    """The design's decision rule, from the intervals and the risk checks. HSN helps only when
    it is above no-trade as well: an interval wholly below no-trade is not "not equivalent to not
    trading" in the rule's sense (a latent gap found in review, after the holdout; no decision
    of #107 changes)."""
    equivalent = Decimal(against_none["lower"]) <= 0 <= Decimal(against_none["upper"])
    above_none = Decimal(against_none["lower"]) > 0
    dd_ok = hsn_drawdown <= factor * control_drawdown
    if Decimal(against["lower"]) > 0 and not breach and dd_ok and above_none:
        decision = "hsn_helps"
    elif Decimal(against["mean"]) > 0:
        decision = "revise"
    else:
        decision = "no_gain"
    return {"against_control": against, "against_no_trade": against_none,
            "equivalent_to_no_trade": equivalent, "breach": breach, "drawdown_ok": dd_ok,
            "decision": decision}


def audit(ctx):
    """Two loads give the same actions on a fixed set of observations; cost per decision."""
    rule = ctx.design["audit"]
    day = days_of(ctx.design, ctx.base, ctx.part, "validation")[0]
    seed = ctx.base["evaluation"]["evaluation_seeds"][0]
    sets = {}
    for arm in ctx.design["arms"]:
        job = {"part": ctx.part, "arm": arm, "fee": ctx.fee, "data_dir": ctx.data_dir}
        env = build_env(job, ctx.digests, ctx.groups[0], day, "validation")
        observations = []
        obs, _ = env.reset(seed=seed)
        while len(observations) < rule["observations"]:
            observations.append(obs)
            obs, _, terminated, truncated, _ = env.step(np.asarray(m6.HOLD, dtype=np.float32))
            if terminated or truncated:
                obs, _ = env.reset(seed=seed)
        env.close()
        sets[arm] = np.asarray(observations, dtype=np.float32)
    out = {}
    for meta in ctx.runs():
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        observations = sets[meta["arm"]]
        recurrent = meta["arm"] == "lstm"
        first, second = load_model(meta["arm"], path), load_model(meta["arm"], path)

        def act(model):
            if not recurrent:
                return model.predict(observations, deterministic=True)[0]
            state, start, actions = None, np.ones((1,), dtype=bool), []
            for row in observations:
                action, state = model.predict(row, state=state, episode_start=start,
                                              deterministic=True)
                start = np.zeros((1,), dtype=bool)
                actions.append(action)
            return np.asarray(actions)

        timings = []
        for row in observations[: rule["timed"]]:
            started = time.perf_counter_ns()
            if recurrent:
                first.predict(row, state=None, episode_start=np.ones((1,), dtype=bool),
                              deterministic=True)
            else:
                first.predict(row, deterministic=True)
            timings.append(time.perf_counter_ns() - started)
        flops = None
        if not recurrent:
            with FlopCounterMode(display=False) as counter, torch.no_grad():
                first.policy.get_distribution(torch.as_tensor(observations[:1]))
            flops = int(counter.get_total_flops())
        out[path.stem] = {"identical": bool(np.array_equal(act(first), act(second))),
                          "flops_per_decision": flops,
                          "median_predict_us": float(np.median(timings)) / 1000.0,
                          "parameters": meta["parameters"], "train_seconds": meta["train_seconds"],
                          "learned": meta.get("learned")}
    return out


def report(ctx, binary):
    """Paired intervals and the design's decision, per group and pooled, at the chosen fee and
    at the secondary fee beside it."""
    design = ctx.design
    rows_all = json.loads((ctx.out / "holdout.json").read_text())["rows"]
    best = json.loads((ctx.out / "validation.json").read_text())["best_control"]
    days = days_of(design, ctx.base, ctx.part, "holdout")
    scopes = [([g], g) for g in ctx.groups]
    if ctx.part == "market":
        scopes.append((ctx.groups, "pooled"))
    out = {}
    for fee in dict.fromkeys([ctx.fee, ctx.secondary_fee]):
        rows = [r for r in rows_all if r["fee"] == fee]

        def daily(arm, group):
            """A learned arm's day is the mean over its seeds, as in #103."""
            values_by_day = []
            for day in days:
                values = [Decimal(r["daily_value"]) for r in rows
                          if r["arm"] == arm and r["group"] == group and r["day"] == day]
                values_by_day.append(sum(values, Decimal(0)) / len(values))
            return values_by_day

        def control_of(groups):
            return best["pooled"] if ctx.part == "market" else best[groups[0]]

        series = {}
        for groups, scope in scopes:
            hsn = [v for g in groups for v in daily("hsn", g)]
            for other in (control_of(groups), "no_trade"):
                theirs = [v for g in groups for v in daily(other, g)]
                series[f"{scope}/hsn-minus-{other}"] = [str(a - b) for a, b in zip(hsn, theirs)]
        path = ctx.out / f"series-{fee}.json"
        path.write_text(json.dumps({"bootstrap": ctx.base["evaluation"]["bootstrap"],
                                    "series": series}))
        intervals = json.loads(subprocess.run([binary, str(path)], check=True,
                                              capture_output=True, text=True).stdout)
        decisions = {}
        for groups, scope in scopes:
            control = control_of(groups)
            scoped = [r for r in rows if r["group"] in groups]
            breach = any(e["limit_breach"] for r in scoped if r["arm"] == "hsn"
                         for e in r["episodes"])

            def drawdown(arm):
                values = [Decimal(e["attribution"]["max_drawdown"]) for r in scoped
                          if r["arm"] == arm for e in r["episodes"]]
                return sum(values, Decimal(0)) / len(values)

            decisions[scope] = {"control": control, **decide(
                intervals[f"{scope}/hsn-minus-{control}"],
                intervals[f"{scope}/hsn-minus-no_trade"], breach, drawdown("hsn"),
                drawdown(control), Decimal(design["decision_rule"]["drawdown_factor"]))}
        out[fee] = {"decisions": decisions, "intervals": intervals}
    result = {**ctx.header("report"), "by_fee": out, "audit": audit(ctx)}
    print(json.dumps({fee: v["decisions"] for fee, v in out.items()}, indent=2),
          ctx.write("report", result))


def main():
    step, part = sys.argv[1], sys.argv[2]
    if part not in ("market", "causal"):
        raise SystemExit("part is market or causal")
    if step == "train":
        train(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), int(sys.argv[6]))
    elif step == "classical":
        classical(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), int(sys.argv[6]))
    elif step == "select":
        select(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), int(sys.argv[6]))
    elif step == "freeze":
        freeze(Context(part, None, sys.argv[3], None))
    elif step == "holdout":
        holdout(Context(part, sys.argv[3], sys.argv[4], sys.argv[5]), sys.argv[6], int(sys.argv[7]),
                attempt.recovery_reason(sys.argv))
    elif step == "report":
        report(Context(part, sys.argv[3], sys.argv[4], None), sys.argv[5])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
