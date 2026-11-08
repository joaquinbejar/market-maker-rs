"""Runs a step of the quote duration comparison of #110 from its design.

From `bridges/python`, once the rows of `digests.json` are written:

    python experiments/m4_10/run.py <step> <data dir> <out dir> <commit> [<frozen sha256>]

Steps, in order: `fee_rule`, `predictor_fit`, `classical_selection`, `teacher`, `training`,
`freeze`, `holdout`. Every day's rows are checked against `digests.json` before use. Only
`holdout` reads the holdout, once, from the manifest `freeze` writes, and it refuses any
checkpoint, predictor, teacher, report or design whose hash differs. Every row of every
report carries its arm, latency, interval, seed and fee.

The choices the design left open are its version 3 (`runner_choices` in `design.json`):
the predictor's target and inputs, the teacher's projection of the classical quote onto the
grid and its cloning loss, and the arm without a duration.

Everything here is synthetic. One synthetic simulator, made-up parameters and 24 days per
regime bound what it can show.
"""

import importlib.util
import json
import pathlib
import sys
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from mm_rl_bridge import BoxMarketMakingEnv, ResidualMarketMakingEnv
from mm_rl_bridge.training import ZeroResidual, versions
from mm_rl_bridge.wrappers import ORDER_KINDS, ChosenDuration, DecisionInterval

HERE = pathlib.Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# #103's runner, for the helpers whose rules this design takes from it unchanged.
base = _module("m4_06_run", HERE.parent / "m4_06" / "run.py")
sha256, read_rows, grid_points = base.sha256, base.read_rows, base.grid_points
days_of, regimes_of, baseline_config = base.days_of, base.regimes_of, base.baseline_config
time_beyond, applied_kind, pool = base.time_beyond, base.applied_kind, base.pool
mean_daily, group, Constant, DayCycle = base.mean_daily, base.group, base.Constant, base.DayCycle

LEARNED = ("full", "no_teacher", "no_duration", "no_predictor")
CLASSICAL = ("avellaneda_stoikov",)
HOLD = [-1.0, 0.0, 0.0, 0.0, 0.0]
CLASSICAL_INTERVALS = (1, 10)
MARKOUT_MS = (1000, 5000, 30000)
FORECAST_ROWS = 5
IMMEDIATE = {"kind": "immediate"}


def load_design():
    design = json.loads((HERE / "design.json").read_text())
    digests = json.loads((HERE / "digests.json").read_text())
    if digests["design_version"] != design["data_design_version"]:
        raise ValueError("the digests belong to another design")
    return design, {(d["regime"], d["day"]): d for d in digests["days"]}


def config(design, fee, latency, policy="reject"):
    """The environment's configuration at one fee and one of the design's latencies."""
    kernel, run = design["kernel"], design["execution"]
    grid = design["action_space"]["grid"]
    return {
        "tick_size": kernel["tick_size"], "lot_size": kernel["lot_size"],
        "max_price": kernel["max_price"], "max_order_size": kernel["max_order_size"],
        "max_steps": design["simulator"]["rows_per_day"], "fee_rate": fee,
        "initial_capital": run["initial_capital"], "latency": run["latencies"][latency],
        "fill_model": run["fill_model"], "limits": run["limits"],
        "bid_offsets": grid["bid_offsets"], "ask_offsets": grid["ask_offsets"],
        "sizes": grid["sizes"], "scales": design["observation_scales"], "f32_unit": "1",
        "invalid_action": policy,
        "invalid_action_penalty": design["action_space"]["invalid_action_penalty"],
        "terminal": run["terminal"],
    }


class Forecast(gym.ObservationWrapper):
    """The observation, then the fixed linear forecast of the mid change over the next rows,
    clipped to `[-1, 1]`: `coefficients` weigh the observation's values, then the intercept."""

    def __init__(self, env, coefficients):
        super().__init__(env)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        base_space = env.observation_space
        if self.coefficients.shape != (base_space.shape[0] + 1,):
            raise ValueError(f"a forecast of {base_space.shape[0]} values takes "
                             f"{base_space.shape[0] + 1} coefficients")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((base_space.low, [-1.0])).astype(np.float32),
            high=np.concatenate((base_space.high, [1.0])).astype(np.float32),
            dtype=np.float32,
        )

    def forecast(self, observation):
        value = float(np.dot(self.coefficients[:-1], observation) + self.coefficients[-1])
        return min(max(value, -1.0), 1.0)

    def observation(self, observation):
        return np.concatenate((np.asarray(observation, dtype=np.float32),
                               np.asarray([self.forecast(observation)], dtype=np.float32)))


def arm_shape(design, arm):
    return design["arms"][arm]


def wrap(env, design, arm, predictor):
    """The learned arm's environment: a chosen duration and a forecast, as the arm has them."""
    shape = arm_shape(design, arm)
    if shape["duration"]:
        env = ChosenDuration(env, design["duration"]["choices_rows"])
    if shape["predictor"]:
        env = Forecast(env, predictor["coefficients"])
    return env


def build_env(job, digests, regime, day, split, holdout_allowed=False):
    """The environment of one day for the job's arm, latency and interval."""
    design, _ = load_design()
    rows, record = read_rows(job["data_dir"], digests, regime, day, split, holdout_allowed)
    cfg = config(design, job["fee"], job["latency"])
    if job.get("fill") == "immediate":
        cfg["fill_model"] = IMMEDIATE
    if job["arm"] in CLASSICAL:
        cfg["residual"] = {"baseline": baseline_config(job["arm"], job["point"], day),
                           "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        env = ResidualMarketMakingEnv(cfg, rows)
    else:
        env = BoxMarketMakingEnv(cfg, rows)
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError(f"{regime} day {day} has other rows than digests.json")
    if job["arm"] in LEARNED:
        return wrap(env, design, job["arm"], job.get("predictor"))
    return DecisionInterval(env, job["every"])


def mid2(row):
    """Twice the mid of a row `(ts_ns, bid, bid_lots, ask, ask_lots)`, in ticks."""
    return row[1] + row[3]


def nearest(value, choices):
    """The choice nearest `value`; a tie goes to the larger choice."""
    return min(choices, key=lambda c: (abs(c - value), -c))


def bin_centre(index, count):
    """The value in `[-1, 1]` at the centre of bin `index` of `count` equal bins."""
    return -1.0 + (2 * index + 1) / count


PROJECTION_NOTES = ("quotes", "holds", "one_sided", "inside_mid", "beyond_grid", "off_grid",
                    "error_ticks")


def teacher_target(classical, row, grid, durations, notes=None, duration=None):
    """The Box action, at its bins' centres, nearest the classical action on the grid: offsets
    from the observed row's mid clamped to the grid, sizes to the nearest, a missing side at
    the farthest offset and the smallest size, and `duration` (the shortest when None).

    `notes`, when given, counts what the projection did: quotes and holds, one-sided quotes
    whose missing side it made live, sides inside the mid clamped to 0, sides beyond the
    grid clamped to its farthest offset, sides snapped off their offset, and the ticks they
    moved, clamping included."""
    notes = notes if notes is not None else dict.fromkeys(PROJECTION_NOTES, 0)
    offsets, sizes = grid["bid_offsets"], grid["sizes"]
    if grid["ask_offsets"] != offsets:
        raise ValueError("the teacher's projection takes one offset grid for both sides")
    m = mid2(row)
    sides = {}
    if classical["kind"] == "quote":
        sides = {"bid": (classical["bid"], classical["bid_lots"]),
                 "ask": (classical["ask"], classical["ask_lots"])}
    elif classical["kind"] in ("bid", "ask"):
        sides = {classical["kind"]: (classical["price"], classical["lots"])}
    if not sides:
        notes["holds"] += 1
        target = [-0.5, 0.0, 0.0, 0.0, 0.0]
    else:
        notes["quotes"] += 1
        notes["one_sided"] += len(sides) == 1

        def side(name):
            if name not in sides:
                return len(offsets) - 1, 0
            price, lots = sides[name]
            away = m // 2 - price if name == "bid" else price - (m + 1) // 2
            notes["inside_mid"] += away < 0
            notes["beyond_grid"] += away > max(offsets)
            offset = nearest(min(max(away, 0), max(offsets)), offsets)
            notes["off_grid"] += offset != away
            notes["error_ticks"] += abs(offset - away)
            return offsets.index(offset), sizes.index(nearest(lots, sizes))

        (bid, bid_size), (ask, ask_size) = side("bid"), side("ask")
        target = [0.5, bin_centre(bid, len(offsets)), bin_centre(ask, len(offsets)),
                  bin_centre(bid_size, len(sizes)), bin_centre(ask_size, len(sizes))]
    if durations is not None:
        chosen = durations.index(duration) if duration is not None else 0
        target.append(bin_centre(chosen, len(durations)))
    return target


def mid_changes(rows, ahead):
    """For each row, the mid change over the next `ahead` rows in ticks, None near the end."""
    return [(mid2(rows[i + ahead]) - mid2(rows[i])) / 2 if i + ahead < len(rows) else None
            for i in range(len(rows))]


def fit_predictor(samples):
    """Least squares of the targets on the observations and an intercept."""
    x = np.asarray([s[0] for s in samples], dtype=np.float64)
    y = np.asarray([s[1] for s in samples], dtype=np.float64)
    design_matrix = np.hstack((x, np.ones((len(x), 1))))
    # The minimum norm solution: a value constant under the hold policy gets no weight of
    # its own beyond what the intercept takes.
    coefficients, _, rank, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    fitted = design_matrix @ coefficients
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else 0.0
    return [float(c) for c in coefficients], r2, int(rank)


def predictor_samples(job):
    """One training day under the hold policy: each observation and the scaled mid change
    over the next rows from the row it observed."""
    design, digests = load_design()
    rows, record = read_rows(job["data_dir"], digests, job["regime"], job["day"], "train")
    env = BoxMarketMakingEnv(config(design, job["fee"], job["latency"]), rows)
    if env.bridge_info["dataset_digest"] != record["dataset_digest"]:
        env.close()
        raise ValueError("other rows than digests.json")
    scale = float(design["observation_scales"]["mid_change_ticks"])
    changes = mid_changes(rows, FORECAST_ROWS)
    observation, info = env.reset(seed=design["evaluation"]["evaluation_seeds"][0])
    samples = []
    try:
        while True:
            if changes[info["row"]] is not None:
                samples.append((observation.tolist(), changes[info["row"]] / scale))
            observation, _, terminated, truncated, info = env.step(HOLD)
            if terminated or truncated:
                break
    finally:
        env.close()
    return samples


def teacher_samples(job):
    """The classical point's own decisions on training days drawn from the seed, until the
    design's number: each observation (with the forecast when the arm has one) and its
    target on the arm's Box."""
    design, digests = load_design()
    grid = design["action_space"]["grid"]
    durations = design["duration"]["choices_rows"] if arm_shape(design, job["arm"])["duration"] \
        else None
    coefficients = (np.asarray(job["predictor"]["coefficients"])
                    if arm_shape(design, job["arm"])["predictor"] else None)
    every = design["runner_choices"]["teacher"]["every_rows"]
    rng = np.random.default_rng(job["seed"])
    days = [(regime, day) for regime in regimes_of(design) for day in days_of(design, "train")]
    observations, targets = [], []
    notes = dict.fromkeys(PROJECTION_NOTES, 0) | {"fills": 0}
    while len(targets) < job["decisions"]:
        regime, day = days[int(rng.integers(len(days)))]
        rows, record = read_rows(job["data_dir"], digests, regime, day, "train")
        cfg = config(design, job["fee"], job["latency"])
        point = job["point"] | {"volatility": job["sigmas"][regime]}
        cfg["residual"] = {"baseline": baseline_config("avellaneda_stoikov", point, day),
                           "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
        inner = ResidualMarketMakingEnv(cfg, rows)
        env = DecisionInterval(inner, every)
        try:
            if inner.bridge_info["dataset_digest"] != record["dataset_digest"]:
                raise ValueError("other rows than digests.json")
            observation, info = env.reset(seed=int(rng.integers(2**31)))
            while len(targets) < job["decisions"]:
                observed = rows[info["row"]]
                step = env.step([0.0, 0.0, 0.0, 0.0])
                classical = step[4]["residual"]["classical"]
                value = observation.astype(np.float64)
                if coefficients is not None:
                    forecast = float(np.dot(coefficients[:-1], value) + coefficients[-1])
                    value = np.append(value, min(max(forecast, -1.0), 1.0))
                observations.append(value.tolist())
                targets.append(teacher_target(classical, observed, grid, durations, notes,
                                              every if durations is not None else None))
                observation, info = step[0], step[4]
                if step[2] or step[3]:
                    break
            notes["fills"] += inner.attribution()["fills"]
        finally:
            env.close()
    return observations, targets, notes


def clone(model, observations, targets, settings, seed):
    """Fits the policy's mean action to the targets by mean squared error; the value head and
    the log standard deviation stay as they are. Returns the loss of each epoch."""
    torch.manual_seed(seed)
    policy = model.policy
    x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    parameters = list(policy.mlp_extractor.policy_net.parameters()) \
        + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            features = policy.extract_features(x[batch], policy.pi_features_extractor)
            mean = policy.action_net(policy.mlp_extractor.forward_actor(features))
            loss = torch.nn.functional.mse_loss(mean, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(x))
    return losses


def train_job(job):
    """One training run on the pooled training days, from the teacher's weights when the arm
    has one, in its own process on one thread."""
    torch.set_num_threads(1)
    design, digests = load_design()
    set_random_seed(job["seed"])
    pairs = [(regime, day) for regime in regimes_of(design) for day in days_of(design, "train")]
    envs = [build_env(job, digests, regime, day, "train") for regime, day in pairs]
    cycle = DayCycle(envs, job["seed"])
    model = PPO("MlpPolicy", DummyVecEnv([lambda: Monitor(cycle)]), seed=job["seed"],
                device="cpu", verbose=0, **job["hyperparameters"])
    losses = None
    started = time.perf_counter()
    if job.get("teacher") is not None:
        data = json.loads(pathlib.Path(job["teacher"]["path"]).read_text())
        if sha256(job["teacher"]["path"]) != job["teacher"]["sha256"]:
            raise ValueError("the teacher's samples changed after they were drawn")
        losses = clone(model, data["observations"], data["targets"],
                       design["runner_choices"]["cloning"], job["seed"])
    clone_seconds = time.perf_counter() - started
    model.learn(total_timesteps=job["decisions"], progress_bar=False)
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    model.save(path)
    metadata = {
        "arm": job["arm"], "latency": job["latency"], "seed": job["seed"], "fee": job["fee"],
        "decisions": job["decisions"], "hyperparameters": job["hyperparameters"],
        "teacher": job.get("teacher"), "cloning_losses": losses,
        "predictor": job.get("predictor"),
        "spec": list(envs[0].unwrapped.spec_id),
        "kernel_fingerprint": envs[0].unwrapped.bridge_info["kernel_fingerprint"],
        "config": envs[0].unwrapped.config,
        "training_digests": [env.unwrapped.bridge_info["dataset_digest"] for env in envs],
        "commit": job["commit"], "design_sha256": sha256(HERE / "design.json"),
        "versions": versions(), "clone_seconds": round(clone_seconds, 1),
        "train_seconds": round(seconds, 1),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path),
    }
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def load_frozen(metadata, env, also=()):
    """A trained policy, after checking that it is the one trained and that the environment
    is the one it was trained for, but for its fee, its day and the keys in `also`."""
    path = pathlib.Path(metadata["path"])
    if sha256(path.with_suffix(".zip")) != metadata["checkpoint_sha256"]:
        raise ValueError("a checkpoint changed after training")
    if tuple(metadata["spec"]) != env.unwrapped.spec_id:
        raise ValueError("a checkpoint of another specification")
    free = {"fee_rate", *also}
    same = {k: v for k, v in metadata["config"].items() if k not in free}
    if same != {k: v for k, v in env.unwrapped.config.items() if k not in free}:
        raise ValueError("a checkpoint of another configuration")
    model = PPO.load(path.with_suffix(".zip"), device="cpu")
    model.policy.set_training_mode(False)
    return model


def policy_of(job, env):
    kind = job["policy"]["kind"]
    if kind == "hold":
        return Constant(HOLD)
    if kind == "zero_residual":
        return ZeroResidual()
    # The descriptive rows under immediate fill load the checkpoints trained under the queue.
    also = ("fill_model",) if job.get("fill") == "immediate" else ()
    return load_frozen(job["policy"]["metadata"], env, also)


def markouts(positions, rows, horizons=MARKOUT_MS):
    """Per horizon, the sum over fills of the signed quantity times the mid that many
    milliseconds after the fill less the fill price, in ticks; a fill with no row that late
    is left out. `positions` are `(ms, price in ticks, position after)`."""
    times = [row[0] // 1_000_000 for row in rows]
    out = {}
    for horizon in horizons:
        total, counted, before, index = Decimal(0), 0, Decimal(0), 0
        for ms, price, after in positions:
            after = Decimal(after)
            quantity, before = after - before, after
            target = ms + horizon
            if target > times[-1]:
                continue
            while index + 1 < len(times) and times[index + 1] <= target:
                index += 1
            while index > 0 and times[index] > target:
                index -= 1
            total += quantity * (Decimal(mid2(rows[index])) / 2 - Decimal(price))
            counted += 1
        out[str(horizon)] = {"ticks": str(total), "fills": counted}
    return out


def episode(policy, env, seed, design, rows):
    """One evaluated day."""
    observation, _ = env.reset(seed=seed)
    orders = dict.fromkeys(ORDER_KINDS, 0)
    decisions = quotes = refusals = covered = holds = 0
    durations = {}
    decide_seconds = 0.0
    while True:
        started = time.perf_counter()
        action, _ = policy.predict(observation, deterministic=True)
        decide_seconds += time.perf_counter() - started
        observation, _, terminated, truncated, info = env.step(action)
        decisions += 1
        # A plain Box decides every row.
        covered += info.get("rows", 1)
        kind = applied_kind(info)
        quotes += kind != "hold"
        holds += kind == "hold"
        refusals += info["refused_because"] is not None
        duration = info.get("duration", info.get("rows", 1))
        durations[str(duration)] = durations.get(str(duration), 0) + 1
        for kind in ORDER_KINDS:
            orders[kind] += info["orders"][kind]
        if terminated or truncated:
            break
    attribution = dict(env.unwrapped.attribution())
    limits = design["execution"]["limits"]
    peak = max((abs(Decimal(a)) * Decimal(p) for _, p, a in attribution["positions"]),
               default=Decimal(0))
    tick = Decimal(design["kernel"]["tick_size"])
    in_ticks = [(ms, Decimal(p) / tick, a) for ms, p, a in attribution["positions"]]
    hours = Decimal(covered * design["simulator"]["sample_ms"]) / 3_600_000
    return {
        "seed": seed, "decisions": decisions, "rows": covered, "quote_transactions": quotes,
        "refusals": refusals, "orders": orders, "net": info["net"], "durations": durations,
        "holds_per_decision": holds / decisions,
        "per_hour": {kind: str((Decimal(orders[kind]) / hours).quantize(Decimal("0.0001")))
                     for kind in ORDER_KINDS},
        "markouts_ticks": markouts(in_ticks, rows),
        "decide_microseconds": round(1e6 * decide_seconds / decisions, 2),
        "attribution": {k: v for k, v in attribution.items() if k != "positions"},
        "peak_notional": str(peak), "time_beyond_5_lots": time_beyond(attribution, 5),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak > Decimal(limits["max_notional"]),
    }


def evaluate_job(job):
    """One arm on one day at one latency, interval and fee."""
    torch.set_num_threads(1)
    design, digests = load_design()
    env = build_env(job, digests, job["regime"], job["day"], job["split"],
                    job["split"] == "holdout")
    rows, _ = read_rows(job["data_dir"], digests, job["regime"], job["day"], job["split"],
                        job["split"] == "holdout")
    try:
        policy = policy_of(job, env)
        days = [episode(policy, env, seed, design, rows)
                for seed in design["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets = [Decimal(d["net"]) for d in days]
    keys = ("arm", "candidate", "latency", "every", "fee", "regime", "day", "split")
    return {key: job[key] for key in keys} | {"fill": job.get("fill", "queue")} | {
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)), "episodes": days}


class Context:
    """What every step shares: the design, the data, the output and the commit."""

    def __init__(self, data_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.digests = load_design()
        self.data_dir, self.commit = data_dir, commit
        self.out = pathlib.Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "teacher").mkdir(parents=True, exist_ok=True)
        self.parallel = self.design["compute"]["max_parallel_runs"]
        self.regimes = regimes_of(self.design)
        self.latencies = list(self.design["execution"]["latencies"])
        self.sigmas = {r: base.volatility(data_dir, self.design, self.digests, r)
                       for r in self.regimes}

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "digests_sha256": sha256(HERE / "digests.json"), "versions": versions(),
                "volatility_from_training_days": self.sigmas}

    def job(self, arm, latency, every, fee, regime, day, split, candidate, policy, point=None):
        return {"arm": arm, "latency": latency, "every": every, "fee": fee, "regime": regime,
                "day": day, "split": split, "candidate": candidate, "policy": policy,
                "point": point, "data_dir": self.data_dir}

    def classical_jobs(self, fee, split, days, latencies, intervals, chosen=None):
        jobs = []
        grid = self.design["arms"]["avellaneda_stoikov"]["grid"]
        for latency in latencies:
            for every in intervals:
                for regime in self.regimes:
                    for day in days:
                        jobs.append(self.job("no_trade", latency, every, fee, regime, day, split,
                                             "hold", {"kind": "hold"}))
                        for point in grid_points(grid, self.sigmas[regime]):
                            key = f"{regime}/avellaneda_stoikov/{latency}/{every}"
                            if chosen is not None and chosen[key] != point:
                                continue
                            jobs.append(self.job("avellaneda_stoikov", latency, every, fee,
                                                 regime, day, split,
                                                 json.dumps(point, sort_keys=True),
                                                 {"kind": "zero_residual"}, point))
        return jobs

    def learned_jobs(self, runs, fee, split, days):
        return [self.job(m["arm"], m["latency"], None, fee, regime, day, split,
                         f"seed{m['seed']}", {"kind": "frozen", "metadata": m})
                | {"predictor": m["predictor"]}
                for m in runs for regime in self.regimes for day in days]

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path

    def read(self, name):
        return json.loads((self.out / f"{name}.json").read_text())


def best_point(rows, key=lambda r: r["candidate"]):
    """The candidate with the highest mean daily net; a tie goes to the greater candidate."""
    points = group(rows, key)
    top = max(points, key=lambda c: (mean_daily(points[c]), c))
    return json.loads(top), mean_daily(points[top])


def without_volatility(row):
    """A point's parameters but its volatility, which the volatility rule sets per regime."""
    point = json.loads(row["candidate"])
    point.pop("volatility")
    return json.dumps(point, sort_keys=True)


def fee_rule(ctx):
    """#103's fee rule on these training days, deciding every row at the shortest latency."""
    report = ctx.header("fee_rule")
    started = time.perf_counter()
    rule = ctx.design["runner_choices"]["fee_rule"]
    rows = []
    for fee in rule["candidates"]:
        rows += pool(ctx.classical_jobs(fee, "train", days_of(ctx.design, "train"),
                                        [rule["latency"]], [rule["every_rows"]]),
                     evaluate_job, ctx.parallel, f"fee {fee}")
    best = {}
    for fee, by_fee in group([r for r in rows if r["arm"] == "avellaneda_stoikov"],
                             lambda r: r["fee"]).items():
        point, value = best_point(by_fee, without_volatility)
        best[fee] = {"point": point, "mean_daily_net": str(value)}
    positive = [fee for fee in rule["candidates"] if Decimal(best[fee]["mean_daily_net"]) > 0]
    chosen = max(positive, key=Decimal) if positive else \
        ctx.design["execution"]["secondary_fee_rate"]
    point = json.dumps(best[chosen]["point"], sort_keys=True)
    point_fills = sum(r["episodes"][0]["attribution"]["fills"] for r in rows
                      if r["arm"] == "avellaneda_stoikov" and r["fee"] == chosen
                      and without_volatility(r) == point)
    report.update({"rows": rows, "best_per_fee": best, "fee": chosen,
                   "teacher_point": best[chosen]["point"], "teacher_point_fills": point_fills,
                   "profitable": bool(positive),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("fee_rule", report)
    return report


def predictor_fit(ctx, fee):
    """The forecast's coefficients, from every row of every training day under the hold
    policy, every regime pooled."""
    report = ctx.header("predictor_fit")
    started = time.perf_counter()
    latency = ctx.design["runner_choices"]["fee_rule"]["latency"]
    jobs = [{"regime": regime, "day": day, "fee": fee, "latency": latency,
             "data_dir": ctx.data_dir}
            for regime in ctx.regimes for day in days_of(ctx.design, "train")]
    parts = pool(jobs, predictor_samples, ctx.parallel, "predictor")
    samples = [s for part in parts for s in part]
    coefficients, r2, rank = fit_predictor(samples)
    report.update({"fee": fee, "latency": latency, "samples": len(samples),
                   "coefficients": coefficients, "r2": r2, "rank": rank,
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("predictor", report)
    return report


def classical_selection(ctx, fee):
    report = ctx.header("classical_selection")
    started = time.perf_counter()
    rows = pool(ctx.classical_jobs(fee, "validation", days_of(ctx.design, "validation"),
                                   ctx.latencies, CLASSICAL_INTERVALS),
                evaluate_job, ctx.parallel, "classical")
    chosen = {}
    for (regime, latency, every), part in group(
            [r for r in rows if r["arm"] in CLASSICAL],
            lambda r: (r["regime"], r["latency"], r["every"])).items():
        chosen[f"{regime}/avellaneda_stoikov/{latency}/{every}"] = best_point(part)[0]
    report.update({"fee": fee, "rows": rows, "selected": chosen,
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("classical_selection", report)
    return report


def teacher_job(job):
    torch.set_num_threads(1)
    started = time.perf_counter()
    observations, targets, notes = teacher_samples(job)
    path = pathlib.Path(job["path"])
    path.write_text(json.dumps({"observations": observations, "targets": targets}))
    return {key: job[key] for key in ("arm", "latency", "seed", "decisions")} | {
        "path": str(path), "sha256": sha256(path), "projection": notes,
        "seconds": round(time.perf_counter() - started, 1)}


def teacher(ctx, fee_report, predictor):
    """The classical point's decisions, per teacher arm, latency and seed."""
    if fee_report["teacher_point_fills"] == 0:
        raise ValueError("the fee rule's point does not fill on the training days: no teacher")
    report = ctx.header("teacher")
    started = time.perf_counter()
    jobs = [{"arm": arm, "latency": latency, "seed": seed, "fee": fee_report["fee"],
             "point": fee_report["teacher_point"], "sigmas": ctx.sigmas,
             "decisions": ctx.design["runner_choices"]["teacher"]["decisions"],
             "data_dir": ctx.data_dir,
             "predictor": predictor,
             "path": str(ctx.out / "teacher" / f"{arm}-{latency}-seed{seed}.json")}
            for arm in LEARNED if ctx.design["arms"][arm]["teacher"]
            for latency in ctx.latencies for seed in ctx.design["training_seeds"]]
    report.update({"fee": fee_report["fee"], "point": fee_report["teacher_point"],
                   "samples": pool(jobs, teacher_job, ctx.parallel, "teacher"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("teacher", report)
    return report


def training(ctx, fee, predictor, teacher_report):
    report = ctx.header("training")
    started = time.perf_counter()
    empty = [s["path"] for s in teacher_report["samples"] if s["projection"]["fills"] == 0]
    if empty:
        raise ValueError(f"a teacher's own rollout never filled: {empty}")
    samples = {(s["arm"], s["latency"], s["seed"]): s for s in teacher_report["samples"]}
    hyperparameters = {k: v for k, v in ctx.design["hyperparameters"].items() if k != "note"}
    jobs = []
    for arm in LEARNED:
        for latency in ctx.latencies:
            for seed in ctx.design["training_seeds"]:
                name = f"{arm}-{latency}-seed{seed}"
                jobs.append({"arm": arm, "latency": latency, "seed": seed, "fee": fee,
                             "decisions": ctx.design["budget_decisions"],
                             "hyperparameters": hyperparameters, "data_dir": ctx.data_dir,
                             "commit": ctx.commit, "predictor": predictor, "every": None,
                             "teacher": samples.get((arm, latency, seed)),
                             "path": str(ctx.out / "checkpoints" / name)})
    # The slow ones first, so that the pool ends together: a chosen duration at a short
    # latency.
    jobs.sort(key=lambda job: (not ctx.design["arms"][job["arm"]]["duration"],
                               int(job["latency"])))
    report.update({"fee": fee, "runs": pool(jobs, train_job, ctx.parallel, "training"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("training", report)
    return report


FROZEN_REPORTS = ("fee_rule", "predictor", "classical_selection", "teacher", "training")
# The code the holdout runs: this runner and #103's, whose helpers it takes.
FROZEN_CODE = {"code_sha256": HERE / "run.py",
               "base_code_sha256": HERE.parent / "m4_06" / "run.py"}


def freeze(ctx):
    reports = {name: ctx.read(name) for name in FROZEN_REPORTS}
    for metadata in reports["training"]["runs"]:
        if sha256(pathlib.Path(metadata["path"]).with_suffix(".zip")) != \
                metadata["checkpoint_sha256"]:
            raise ValueError(f"{metadata['path']} changed after training")
    for sample in reports["teacher"]["samples"]:
        if sha256(sample["path"]) != sample["sha256"]:
            raise ValueError(f"{sample['path']} changed after it was drawn")
    manifest = {**ctx.header("freeze"),
                **{f"{name}_sha256": sha256(ctx.out / f"{name}.json") for name in FROZEN_REPORTS},
                **{key: sha256(path) for key, path in FROZEN_CODE.items()},
                "fee": reports["fee_rule"]["fee"],
                "classical_selected": reports["classical_selection"]["selected"],
                "runs": reports["training"]["runs"]}
    path = ctx.write("frozen", manifest)
    return {"frozen_sha256": sha256(path), "runs": len(manifest["runs"])}


def holdout(ctx, frozen_sha256):
    """The holdout, once: every arm as frozen, at the chosen fee and at the secondary one."""
    path = ctx.out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    if (ctx.out / "holdout.json").exists():
        raise ValueError("the holdout was evaluated already")
    frozen = json.loads(path.read_text())
    files = {"design_sha256": HERE / "design.json", "digests_sha256": HERE / "digests.json",
             **FROZEN_CODE}
    for key, path in files.items():
        if frozen[key] != sha256(path):
            raise ValueError(f"{path.parent.name}/{path.name} changed after the freeze")
    for name in FROZEN_REPORTS:
        if frozen[f"{name}_sha256"] != sha256(ctx.out / f"{name}.json"):
            raise ValueError(f"{name}.json changed after the freeze")
    report = ctx.header("holdout")
    report["frozen_sha256"] = frozen_sha256
    started = time.perf_counter()
    days = days_of(ctx.design, "holdout")
    fees = list(dict.fromkeys([frozen["fee"], ctx.design["execution"]["secondary_fee_rate"]]))
    jobs = []
    for fee in fees:
        jobs += ctx.classical_jobs(fee, "holdout", days, ctx.latencies, CLASSICAL_INTERVALS,
                                   frozen["classical_selected"])
        jobs += ctx.learned_jobs(frozen["runs"], fee, "holdout", days)
    # Descriptive, with no decision attached: the arm without a duration under immediate fill.
    without = [m for m in frozen["runs"] if not ctx.design["arms"][m["arm"]]["duration"]]
    jobs += [job | {"fill": "immediate"}
             for job in ctx.learned_jobs(without, frozen["fee"], "holdout", days)]
    report.update({"fees": fees, "rows": pool(jobs, evaluate_job, ctx.parallel, "holdout"),
                   "seconds": round(time.perf_counter() - started, 1)})
    ctx.write("holdout", report)
    return report


def main():
    step, data_dir, out_dir, commit = sys.argv[1:5]
    ctx = Context(data_dir, out_dir, commit)
    if len(sys.argv) > 6:
        ctx.parallel = int(sys.argv[6])
    started = time.perf_counter()
    if step == "fee_rule":
        report = fee_rule(ctx)
    elif step == "predictor_fit":
        report = predictor_fit(ctx, ctx.read("fee_rule")["fee"])
    elif step == "classical_selection":
        report = classical_selection(ctx, ctx.read("fee_rule")["fee"])
    elif step == "teacher":
        report = teacher(ctx, ctx.read("fee_rule"), ctx.read("predictor"))
    elif step == "training":
        report = training(ctx, ctx.read("fee_rule")["fee"], ctx.read("predictor"),
                          ctx.read("teacher"))
    elif step == "freeze":
        report = freeze(ctx)
        print(json.dumps(report))
    elif step == "holdout":
        report = holdout(ctx, sys.argv[5])
    else:
        raise SystemExit(f"unknown step {step!r}")
    print(json.dumps({"step": step, "seconds": report.get("seconds"),
                      "total": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
