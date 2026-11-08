"""The batch auction study of #289, from its design. SYNTHETIC, reported apart from the order book.

From `bridges/python`:

    python experiments/m5_05_batch/run.py <step> <out dir> <commit> <parallel> [<frozen sha256>]

Steps, in order: `predictor`, `fee_rule`, `check`, `classical`, `pilot`, `train`, `freeze`,
`holdout`, `report <intervals binary>`. Every step writes one file and refuses to run twice; only
`holdout` reads holdout days, once, from the manifest `freeze` writes, and it refuses anything
whose hash differs. The days are drawn from their seeds (`design.json`), so no data is stored.
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

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env as batch  # noqa: E402

LEARNED = ("full", "no_duration")
CODE = ("run.py", "env.py", "fba.py")
HOLDOUT_MARKER = "holdout.started"
HOLD = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
MARKOUT_MS = (1000, 5000, 30_000)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    return json.loads((HERE / "design.json").read_text())


def days_of(design, split):
    first, last = design["splits"][split]
    return list(range(first, last + 1))


def split_of(design, day):
    return next(s for s in ("train", "validation", "holdout") if day in days_of(design, s))


def flow_of(design):
    return batch.Flow(**design["toy"]["flow"])


class Predictor:
    """The fixed linear forecast: `coefficients . (market features, 1)`."""

    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def __call__(self, features):
        return float(self.coefficients @ np.append(features, 1.0))


def make_env(design, day, latency, fee, arm, holdout_allowed=False, predictor=None):
    if split_of(design, day) == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    toy = design["toy"]
    with_duration = arm != "no_duration"
    return batch.BatchEnv(
        seed=int(design["day_seed"].split()[0]) + day, rows=toy["rows_per_day"],
        latency_ms=latency, fee_rate=fee, flow=flow_of(design), row_ms=toy["row_ms"],
        batch_ms=toy["batch_ms"], tick=Decimal(toy["tick"]),
        max_position=toy["max_position_lots"], with_duration=with_duration,
        predictor=predictor if arm in LEARNED else None)


def as_policy(point, sigma, design):
    gamma, k = float(point["risk_aversion"]), float(point["order_intensity"])
    horizon = design["arms"]["avellaneda_stoikov"]["horizon_rows"]
    return lambda env, obs: batch.avellaneda_stoikov(env, gamma, k, sigma, horizon)


def hold_policy(env, obs):
    return HOLD if env.with_duration else HOLD[:5]


def drawdown(equity):
    peak, worst = Decimal(0), Decimal(0)
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def markouts(env):
    """Signed markouts of the day's fills against the reference `h` later (the last clearing
    price at or before then), in price units; a fill whose horizon passes the day's last close
    is left out."""
    closes = np.asarray([b[0] for b in env.batches], dtype=np.int64)
    prices, ref = [], env.flow.reference_ticks
    for _, price, _ in env.batches:
        ref = price if price is not None else ref
        prices.append(ref)
    out = {}
    for h in MARKOUT_MS:
        total = Decimal(0)
        for close, side, price, lots in env.fills:
            if len(closes) == 0 or close + h > closes[-1]:
                continue
            at = int(np.searchsorted(closes, close + h, side="right")) - 1
            total += side * lots * (Decimal(prices[at]) - Decimal(price)) * env.tick
        out[str(h)] = str(total)
    return out


def episode(policy, env):
    """One day: `policy(env, obs)` gives the action. The day's metrics, money as text.

    The position, its peak and the equity behind the drawdown are read at every clearing,
    where they change, not once per step: a step of several rows spans several closes."""
    obs, _ = env.reset(seed=0)
    durations, decisions = {}, 0
    while True:
        obs, _, terminated, _, info = env.step(policy(env, obs))
        decisions += 1
        durations[info["duration"]] = durations.get(info["duration"], 0) + 1
        if terminated:
            break
    peak_position = max((abs(position) for _, position, _ in env.marks), default=0)
    equity = [mark for _, _, mark in env.marks] + [Decimal(info["net"])]
    hours = Decimal(env.rows * env.row_ms) / Decimal(3_600_000)
    return {"net": info["net"], "fees": str(env.fees), "fills": len(env.fills),
            "fills_per_hour": str(Decimal(len(env.fills)) / hours), "decisions": decisions,
            "max_drawdown": str(drawdown(equity)), "peak_position": peak_position,
            "final_position": env.position, "breach": peak_position > env.max_position,
            "durations": {str(k): v for k, v in sorted(durations.items())},
            "late": dict(env.late), "markouts": markouts(env)}


def load_model(path):
    model = PPO.load(path, device="cpu")
    model.policy.set_training_mode(False)
    return model


def model_policy(model):
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def predictor_of(out):
    return Predictor(json.loads((out / "predictor.json").read_text())["coefficients"])


def evaluate_job(job):
    """One arm on one day, at one latency and fee."""
    torch.set_num_threads(1)
    design = load()
    predictor = Predictor(job["predictor"]) if job.get("predictor") else None
    env = make_env(design, job["day"], job["latency"], job["fee"], job["arm"],
                   job["split"] == "holdout", predictor)
    kind = job["policy"]["kind"]
    if kind == "hold":
        policy = hold_policy
    elif kind == "as":
        policy = as_policy(job["policy"]["point"], job["policy"]["sigma"], design)
    else:
        meta = job["policy"]["metadata"]
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if sha256(path) != meta["checkpoint_sha256"]:
            raise ValueError(f"{path} is not the checkpoint its metadata names")
        policy = model_policy(load_model(path))
    day = episode(policy, env)
    return {k: job[k] for k in ("arm", "candidate", "latency", "fee", "day", "split")} | day


def pool(jobs, worker, parallel, label):
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for done, _ in enumerate(concurrent.futures.as_completed(futures), start=1):
            print(f"{label}: {done}/{len(jobs)} after {round(time.perf_counter() - started)} s",
                  file=sys.stderr, flush=True)
        return [f.result() for f in futures]


def mean_net(rows):
    values = [Decimal(r["net"]) for r in rows]
    return sum(values, Decimal(0)) / len(values)


def grid(design):
    g = design["arms"]["avellaneda_stoikov"]["grid"]
    return [{"risk_aversion": a, "order_intensity": k}
            for a, k in itertools.product(g["risk_aversion"], g["order_intensity"])]


class DayCycle(gym.Env):
    """One environment per training day; each reset draws a day from the seed."""

    metadata = {"render_modes": []}

    def __init__(self, envs, seed):
        super().__init__()
        self._envs, self._rng = envs, np.random.default_rng(seed)
        self._current = envs[0]
        self.observation_space, self.action_space = envs[0].observation_space, envs[0].action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current = self._envs[int(self._rng.integers(len(self._envs)))]
        return self._current.reset()

    def step(self, action):
        return self._current.step(action)


def clone(model, envs, teacher, sigma, design, decisions, seed):
    """Behaviour cloning of the teacher: its actions on training days, then the policy's mean
    action fitted to them by mean squared error."""
    observations, actions = [], []
    rng = np.random.default_rng(seed)
    policy = as_policy(teacher, sigma, design)
    while len(observations) < decisions:
        env = envs[int(rng.integers(len(envs)))]
        obs, _ = env.reset()
        while len(observations) < decisions:
            action = policy(env, obs)
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
    x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    rule = design["teacher_fit"]
    optimiser = torch.optim.Adam(model.policy.parameters(), lr=rule["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(rule["epochs"]):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), rule["batch_size"]):
            index = order[start:start + rule["batch_size"]]
            mean = model.policy.get_distribution(x[index]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, y[index])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        losses.append(total / len(x))
    return losses


class Curve(BaseCallback):
    """The deterministic policy's mean daily net on the given days every `every` decisions."""

    def __init__(self, design, days, latency, fee, arm, predictor, every):
        super().__init__()
        self.design, self.days, self.latency, self.fee, self.arm = design, days, latency, fee, arm
        self.predictor, self.every, self.points = predictor, every, []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            nets = [Decimal(episode(model_policy(self.model),
                                    make_env(self.design, d, self.latency, self.fee, self.arm,
                                             predictor=self.predictor))["net"])
                    for d in self.days]
            self.points.append([self.num_timesteps, str(sum(nets, Decimal(0)) / len(nets))])
        return True


def train_job(job):
    torch.set_num_threads(1)
    design = load()
    set_random_seed(job["seed"])
    predictor = Predictor(job["predictor"])
    envs = [make_env(design, d, job["latency"], job["fee"], job["arm"], predictor=predictor)
            for d in days_of(design, "train")]
    cycle = DayCycle(envs, job["seed"])
    hyper = design["hyperparameters"]
    kwargs = {k: hyper[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef")}
    model = PPO("MlpPolicy", DummyVecEnv([lambda: Monitor(cycle)]), seed=job["seed"],
                device="cpu", verbose=0, **kwargs)
    started = time.perf_counter()
    losses = clone(model, envs, job["teacher"], job["sigma"], design,
                   design["teacher_fit"]["decisions"], job["seed"])
    callback = None
    if job.get("curve"):
        callback = Curve(design, job["curve"]["days"], job["latency"], job["fee"], job["arm"],
                         predictor, job["curve"]["every"])
    model.learn(total_timesteps=design["budget_decisions"], progress_bar=False, callback=callback)
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    model.save(path)
    meta = {"arm": job["arm"], "latency": job["latency"], "seed": job["seed"], "fee": job["fee"],
            "teacher": job["teacher"], "clone_losses": losses,
            "decisions": design["budget_decisions"], "commit": job["commit"],
            "design_sha256": sha256(HERE / "design.json"), "train_seconds": round(seconds, 1),
            "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path)}
    if callback is not None:
        meta["curve"] = callback.points
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


class Context:
    def __init__(self, out, commit, parallel):
        torch.set_num_threads(1)
        self.design = load()
        self.out = pathlib.Path(out)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.commit, self.parallel = commit, parallel
        self.latencies = self.design["latencies_ms"]

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "design_sha256": sha256(HERE / "design.json")}

    def write(self, name, report):
        path = self.out / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; a step runs once")
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return sha256(path)

    def read(self, name):
        return json.loads((self.out / f"{name}.json").read_text())

    def fee(self):
        return self.read("fee_rule")["fee"]

    def sigma(self):
        return self.read("predictor")["sigma"]

    def job(self, arm, day, latency, fee, candidate, policy, predictor=None):
        return {"arm": arm, "day": day, "split": split_of(self.design, day), "latency": latency,
                "fee": fee, "candidate": candidate, "policy": policy, "predictor": predictor}

    def as_jobs(self, days, latencies, fees):
        sigma = self.sigma()
        return [self.job("avellaneda_stoikov", d, lat, fee, json.dumps(p, sort_keys=True),
                         {"kind": "as", "point": p, "sigma": sigma})
                for fee in fees for lat in latencies for p in grid(self.design) for d in days]

    def train_jobs(self):
        fee, sigma = self.fee(), self.sigma()
        teachers = self.read("classical")["teachers"]
        coefficients = self.read("predictor")["coefficients"]
        return [{"arm": arm, "latency": lat, "seed": seed, "fee": fee, "sigma": sigma,
                 "teacher": teachers[str(lat)], "predictor": coefficients, "commit": self.commit,
                 "path": str(self.out / "checkpoints" / f"{arm}-{lat}ms-seed{seed}")}
                for arm in LEARNED for lat in self.latencies
                for seed in self.design["training_seeds"]]

    def runs(self):
        out = []
        for job in self.train_jobs():
            meta = pathlib.Path(job["path"]).with_suffix(".json")
            if not meta.exists():
                raise FileNotFoundError(f"{meta} has not been trained")
            out.append(json.loads(meta.read_text()))
        return out


def best(rows):
    points = {}
    for r in rows:
        points.setdefault(r["candidate"], []).append(r)
    top = max(points, key=lambda c: (mean_net(points[c]), c))
    return json.loads(top), str(mean_net(points[top]))


def predictor_step(ctx):
    """The no-agent training days: the volatility of the clearing price, and the least squares
    forecast of the reference's change over the next 5 rows from the market features."""
    design = ctx.design
    xs, ys, changes = [], [], []
    for day in days_of(design, "train"):
        env = make_env(design, day, ctx.latencies[0], "0", "avellaneda_stoikov")
        env.reset()
        features, refs = [], []
        while True:
            features.append(np.append(env.market_features(), 1.0))
            refs.append(env.reference)
            _, _, terminated, _, _ = env.step(HOLD)
            if terminated:
                break
        refs.append(env.reference)
        refs = np.asarray(refs, dtype=np.float64)
        changes.extend(np.diff(refs))
        horizon = 5
        for i in range(len(features) - horizon):
            xs.append(features[i])
            ys.append(refs[i + horizon] - refs[i])
    coefficients, *_ = np.linalg.lstsq(np.asarray(xs), np.asarray(ys), rcond=None)
    tick = float(design["toy"]["tick"])
    sigma = float(np.std(np.asarray(changes) * tick))
    print(ctx.write("predictor", {**ctx.header("predictor"), "coefficients": coefficients.tolist(),
                                  "sigma": sigma, "rows": len(ys)}))


def fee_rule(ctx):
    fees = ["0.0002", "0", "-0.0001"]
    rows = pool(ctx.as_jobs(days_of(ctx.design, "train"), [ctx.latencies[0]], fees),
                evaluate_job, ctx.parallel, "fee rule")
    by_fee = {fee: best([r for r in rows if r["fee"] == fee]) for fee in fees}
    positive = [fee for fee in fees if Decimal(by_fee[fee][1]) > 0]
    chosen = max(positive, key=Decimal) if positive else ctx.design["fee"]["secondary"]
    print(chosen, ctx.write("fee_rule", {**ctx.header("fee_rule"), "rows": rows,
                                         "best_per_fee": by_fee, "fee": chosen}))


def check(ctx):
    point = {"risk_aversion": "0.001", "order_intensity": "100"}
    day = days_of(ctx.design, "train")[0]
    rows = pool([ctx.job("avellaneda_stoikov", day, lat, ctx.fee(), "check",
                         {"kind": "as", "point": point, "sigma": ctx.sigma()}) for lat in (30, 100)],
                evaluate_job, 2, "check")
    fills = {r["latency"]: r["fills"] for r in rows}
    nets = {r["latency"]: r["net"] for r in rows}
    passes = all(v > 0 for v in fills.values()) and nets[30] != nets[100]
    print(passes, fills, nets, ctx.write("check", {**ctx.header("check"), "rows": rows,
                                                   "passes": passes}))
    if not passes:
        raise SystemExit("the classical arm does not fill or the delay does not bind")


def classical(ctx):
    """Per latency: the teacher (training-day best at the fee) and the classical arm (validation
    best at the fee)."""
    fee = ctx.fee()
    train = pool(ctx.as_jobs(days_of(ctx.design, "train"), ctx.latencies, [fee]),
                 evaluate_job, ctx.parallel, "teachers")
    valid = pool(ctx.as_jobs(days_of(ctx.design, "validation"), ctx.latencies, [fee]),
                 evaluate_job, ctx.parallel, "classical")
    teachers = {str(lat): best([r for r in train if r["latency"] == lat])[0] for lat in ctx.latencies}
    selected = {str(lat): best([r for r in valid if r["latency"] == lat])[0] for lat in ctx.latencies}
    print(json.dumps({"teachers": teachers, "selected": selected}),
          ctx.write("classical", {**ctx.header("classical"), "train_rows": train,
                                  "validation_rows": valid, "teachers": teachers,
                                  "selected": selected}))


def pilot(ctx):
    rule = ctx.design["pilot"]
    job = {**next(j for j in ctx.train_jobs() if j["arm"] == "full" and j["latency"] == 60),
           "seed": 0, "path": str(ctx.out / "pilot-full-60ms-seed0"),
           "curve": {"days": [1, 2], "every": 50_000}}
    meta = pool([job], train_job, 1, "pilot")[0]
    values = [Decimal(v) for _, v in meta["curve"]]
    last, before = values[-3:], values[-6:-3]
    m_last, m_before = sum(last) / 3, sum(before) / 3
    plateaued = abs(m_last - m_before) <= Decimal("0.1") * max(abs(m_before), Decimal(1))
    print(plateaued, ctx.write("pilot", {**ctx.header("pilot"), "rule": rule, "curve": meta["curve"],
                                         "plateaued": plateaued, "train_seconds": meta["train_seconds"]}))


def train(ctx):
    if (ctx.out / "frozen.json").exists():
        raise SystemExit("already frozen")
    todo = [j for j in ctx.train_jobs() if not pathlib.Path(j["path"]).with_suffix(".json").exists()]
    for meta in pool(todo, train_job, ctx.parallel, "train"):
        print(meta["arm"], meta["latency"], meta["seed"], meta["train_seconds"], flush=True)


def freeze(ctx):
    entries = {}
    for meta in ctx.runs():
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        if sha256(path) != meta["checkpoint_sha256"] or meta["design_sha256"] != sha256(HERE / "design.json"):
            raise SystemExit(f"{path} changed or was trained under another design")
        entries[path.stem] = {"checkpoint": meta["checkpoint_sha256"],
                              "metadata": sha256(path.with_suffix(".json"))}
    print(ctx.write("frozen", {"design_sha256": sha256(HERE / "design.json"),
                               "code_sha256": {name: sha256(HERE / name) for name in CODE},
                               **{f"{n}_sha256": sha256(ctx.out / f"{n}.json")
                                  for n in ("predictor", "fee_rule", "check", "classical", "pilot")},
                               "runs": entries}))


def holdout(ctx, frozen_sha):
    if sha256(ctx.out / "frozen.json") != frozen_sha:
        raise SystemExit("the frozen manifest differs")
    frozen = json.loads((ctx.out / "frozen.json").read_text())
    if frozen["design_sha256"] != sha256(HERE / "design.json"):
        raise SystemExit("the design differs")
    for name, digest in frozen["code_sha256"].items():
        if sha256(HERE / name) != digest:
            raise SystemExit(f"{name} changed after the freeze")
    for n in ("predictor", "fee_rule", "check", "classical", "pilot"):
        if frozen[f"{n}_sha256"] != sha256(ctx.out / f"{n}.json"):
            raise SystemExit(f"{n}.json differs")
    runs = ctx.runs()
    for meta in runs:
        path = pathlib.Path(meta["path"]).with_suffix(".zip")
        entry = frozen["runs"][path.stem]
        if entry["checkpoint"] != sha256(path) or entry["metadata"] != sha256(path.with_suffix(".json")):
            raise SystemExit(f"{path.stem} changed after the freeze")
    if (ctx.out / "holdout.json").exists():
        raise SystemExit("the holdout runs once")
    if (ctx.out / HOLDOUT_MARKER).exists():
        raise SystemExit("the holdout was started already; a failed or concurrent attempt is not "
                         "run again, and recovery is by hand")
    jobs = holdout_jobs(ctx, runs)
    # Before any job reads a holdout day: created only if absent, so a concurrent attempt fails
    # here, and kept if the evaluation fails, so it is never run again on its own.
    descriptor = os.open(ctx.out / HOLDOUT_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as marker:
        marker.write(json.dumps({"commit": ctx.commit, "frozen_sha256": frozen_sha,
                                 "jobs": len(jobs), "at": time.time()}))
    rows = pool(jobs, evaluate_job, ctx.parallel, "holdout")
    print(ctx.write("holdout", {**ctx.header("holdout"), "frozen_sha256": frozen_sha, "rows": rows}))


def holdout_jobs(ctx, runs):
    """Every arm on the holdout days, at each latency, at the fee and the secondary fee."""
    selected = ctx.read("classical")["selected"]
    coefficients = ctx.read("predictor")["coefficients"]
    jobs = []
    for fee in dict.fromkeys([ctx.fee(), ctx.design["fee"]["secondary"]]):
        for lat in ctx.latencies:
            for day in days_of(ctx.design, "holdout"):
                jobs.append(ctx.job("no_trade", day, lat, fee, "hold", {"kind": "hold"}))
                point = selected[str(lat)]
                jobs.append(ctx.job("avellaneda_stoikov", day, lat, fee, json.dumps(point, sort_keys=True),
                                    {"kind": "as", "point": point, "sigma": ctx.sigma()}))
                for meta in runs:
                    if meta["latency"] == lat:
                        jobs.append(ctx.job(meta["arm"], day, lat, fee, f"seed{meta['seed']}",
                                            {"kind": "frozen", "metadata": meta}, coefficients))
    return jobs


def daily(rows, arm, latency, fee, days):
    out = []
    for day in days:
        values = [Decimal(r["net"]) for r in rows if r["arm"] == arm and r["latency"] == latency
                  and r["fee"] == fee and r["day"] == day]
        out.append(sum(values, Decimal(0)) / len(values))
    return out


def report(ctx, binary):
    rows = ctx.read("holdout")["rows"]
    days = days_of(ctx.design, "holdout")
    arms = ("no_trade", "avellaneda_stoikov") + LEARNED
    series, keys = {}, []
    for fee in dict.fromkeys([ctx.fee(), ctx.design["fee"]["secondary"]]):
        for lat in ctx.latencies:
            group = f"{fee}/{lat}ms"
            keys.append((fee, lat, group))
            none = daily(rows, "no_trade", lat, fee, days)
            for arm in arms[1:]:
                mine = daily(rows, arm, lat, fee, days)
                series[f"{group}/{arm}-minus-no_trade"] = [str(a - b) for a, b in zip(mine, none)]
            full, fixed = daily(rows, "full", lat, fee, days), daily(rows, "no_duration", lat, fee, days)
            series[f"{group}/full-minus-no_duration"] = [str(a - b) for a, b in zip(full, fixed)]
    path = ctx.out / "series.json"
    path.write_text(json.dumps({"bootstrap": ctx.design["evaluation"]["bootstrap"], "series": series}))
    intervals = json.loads(subprocess.run([binary, str(path)], check=True, capture_output=True,
                                          text=True).stdout)
    out = {}
    for fee, lat, group in keys:
        scoped = [r for r in rows if r["fee"] == fee and r["latency"] == lat]

        def mean_of(arm, key):
            values = [Decimal(r[key]) for r in scoped if r["arm"] == arm]
            return str(sum(values, Decimal(0)) / len(values))

        summary = {arm: {"mean_daily_net": mean_of(arm, "net"), "fees": mean_of(arm, "fees"),
                         "fills_per_hour": mean_of(arm, "fills_per_hour"),
                         "max_drawdown": mean_of(arm, "max_drawdown"),
                         "markouts": {str(h): str(sum((Decimal(r["markouts"][str(h)]) for r in scoped
                                                       if r["arm"] == arm), Decimal(0))
                                                  / sum(1 for r in scoped if r["arm"] == arm))
                                      for h in MARKOUT_MS},
                         "breach": any(r["breach"] for r in scoped if r["arm"] == arm)}
                   for arm in arms}
        interval = intervals[f"{group}/full-minus-no_duration"]
        helps = Decimal(interval["lower"]) > 0 and not any(summary[a]["breach"] for a in arms)
        equivalent = {arm: Decimal(intervals[f"{group}/{arm}-minus-no_trade"]["lower"]) <= 0
                      <= Decimal(intervals[f"{group}/{arm}-minus-no_trade"]["upper"])
                      for arm in arms[1:]}
        out[group] = {"arms": summary, "intervals": {k[len(group) + 1:]: v for k, v in intervals.items()
                                                     if k.startswith(group + "/")},
                      "duration_helps": helps, "equivalent_to_no_trade": equivalent}
    result = {**ctx.header("report"), "batch_only": ctx.design["reported_apart"],
              "not_run": ctx.design["not_run"], "pilot": ctx.read("pilot"), "groups": out}
    print(json.dumps({g: v["duration_helps"] for g, v in out.items()}), ctx.write("report", result))


def main():
    step, out, commit = sys.argv[1], sys.argv[2], sys.argv[3]
    parallel = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ctx = Context(out, commit, parallel)
    steps = {"predictor": predictor_step, "fee_rule": fee_rule, "check": check,
             "classical": classical, "pilot": pilot, "train": train, "freeze": freeze}
    if step in steps:
        steps[step](ctx)
    elif step == "holdout":
        holdout(ctx, sys.argv[5])
    elif step == "report":
        report(ctx, sys.argv[5])
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main()
