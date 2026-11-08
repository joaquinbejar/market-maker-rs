"""Runs a step of #281's adversarial training study from its committed design.

From `bridges/python`, with #102's frozen runs:

    python experiments/m5_05/run.py <step> <m4_05 run dir> <out dir> <commit> [args]

Steps, in order:
- `days`: the digests of the training points' days and of every holdout condition's exogenous
  rows, written to `experiments/m5_05/digests/` to be committed before any policy runs;
- `rate`: one adversarial run's first 10000 steps and one scoring pass, timed;
- `train [run names]`: the 10 runs, adversarial (A) and baseline (B), seeds 1 to 5; a named
  run that was killed from outside is rerun once;
- `validate`: every policy on #102's validation days, and B's reproduction check;
- `freeze_models`: the checkpoints' hashes, written to `experiments/m5_05/frozen_models.json`;
- `selection`: the held-out points on #108's selection days, and each arm's worst point;
- `freeze`: the hashes of the design, the code, the digests, the models and the selection;
- `holdout <frozen sha256>`: every condition once.

Rows are simulated in process by the bridge's `ReactiveQrMarket`, whose rows and digest are
`qr_days`', and every day is checked against a committed digest before use. Everything here is
synthetic: #102's queue-reactive regimes and #108's perturbed copies of them.
"""

import concurrent.futures
import functools
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import pathlib
import sys
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).resolve().parent
M4_05 = HERE.parent / "m4_05"
M5_03 = HERE.parent / "m5_03"
M5_04 = HERE.parent / "m5_04"
DIGESTS = HERE / "digests"
FROZEN_MODELS = HERE / "frozen_models.json"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module("m4_05_run", M4_05 / "run.py")
m503 = _module("m5_03_run", M5_03 / "run.py")

from mm_rl_bridge import ReactiveQrMarket, ResidualMarketMakingEnv  # noqa: E402

DAY_MS = 86_400_000
SEED_STRIDE = 1_000_003
POINT_KEYS = ("market_rate_multiplier", "cancel_rate_multiplier", "theta_multiplier",
              "limit_rate_multiplier")
TRAINED = ("adversarial", "baseline")
ARMS = (*TRAINED, "reference")
SAC_ARM = "sac_residual_avellaneda_stoikov"
# design.json, in words: "evaluation seed 101" and "training days 1 and 8 of each regime".
EVALUATION_SEED = 101
SCORING_DAYS = (1, 8)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    """This study's design and the two it reuses, refusing any that changed."""
    design = json.loads((HERE / "design.json").read_text())
    reuses = design["reuses"]
    if sha256(M4_05 / "design.json") != reuses["m4_05"]["design_sha256"]:
        raise ValueError("#102's design is not the one this study reuses")
    if sha256(M5_03 / "design.json") != reuses["m5_03"]["design_sha256"]:
        raise ValueError("#108's design is not the one this study reuses")
    m4_05 = json.loads((M4_05 / "design.json").read_text())
    m5_03 = json.loads((M5_03 / "design.json").read_text())
    return design, m4_05, m5_03


# The grid.


def points(m5_03):
    """#108's 24 points, as (name, multipliers), in #108's order."""
    return m503.points(m5_03)


def training_points(m5_03):
    return [(n, p) for n, p in points(m5_03) if p["limit_rate_multiplier"] == 1.0]


def held_out_points(m5_03):
    return [(n, p) for n, p in points(m5_03) if p["limit_rate_multiplier"] != 1.0]


def unperturbed_name():
    return m503.point_name({k: 1.0 for k in POINT_KEYS})


def regime_at(m4_05, m5_03, point, regime_name):
    """A regime of #102's under a point's multipliers, as #108 perturbs it, and its number."""
    perturbed = m503.perturbed(m4_05, point, "selection", m5_03)
    names = [r["name"] for r in perturbed["simulator"]["regimes"]]
    index = names.index(regime_name)
    return perturbed["simulator"]["regimes"][index], index + 1


def split_days(design, m4_05, m5_03, split):
    """A split's seed offset and days: #102's training and validation days, #108's selection
    days, and this study's fresh holdout days."""
    if split in ("train", "validation"):
        first, last = m4_05["splits"][split]
        return 0, list(range(first, last + 1))
    days = m5_03["days"]["selection"] if split == "selection" else design["holdout"]["days"]
    return days["seed_offset"], list(range(1, days["per_regime"] + 1))


def day_seed(regime_number, day, offset):
    return SEED_STRIDE * regime_number + day + offset


def day_market(m4_05, regime, seed, day):
    sim = m4_05["simulator"]
    episode = {"seed": seed, "reference_ticks": sim["reference_ticks"],
               "start_ms": (day - 1) * DAY_MS, "sample_ms": sim["sample_ms"],
               "rows": sim["rows_per_day"]}
    return ReactiveQrMarket(json.dumps(regime), json.dumps(m4_05["kernel"]), json.dumps(episode))


# The committed digests.


@functools.lru_cache(maxsize=None)
def _digest_file(path):
    record = json.loads(pathlib.Path(path).read_text())
    return {(d["regime"], d["day"]): d["dataset_digest"] for d in record["days"]}


def expected_digest(split, name, regime, day):
    """The committed digest of a day: this study's for training points and the holdout, #102's
    for validation, #108's for selection."""
    if split in ("train", "holdout"):
        path = DIGESTS / f"{split}-{name}.json"
    elif split == "validation":
        if name != unperturbed_name():
            raise ValueError("validation is unperturbed only")
        path = M4_05 / "digests.json"
    elif split == "selection":
        path = M5_03 / "digests" / f"selection-{name}.json"
    else:
        raise ValueError(f"unknown split {split!r}")
    return _digest_file(str(path))[(regime, day)]


def market_for(ctx, split, name, regime, day, holdout_allowed=False):
    """A day's market, its rows checked against the committed digest, and the day's seed."""
    if split == "holdout" and not holdout_allowed:
        raise PermissionError("only the holdout step reads the holdout")
    offset, days = split_days(ctx.design, ctx.m4_05, ctx.m5_03, split)
    if day not in days:
        raise ValueError(f"day {day} is not in {split}")
    point = dict(ctx.points)[name]
    regime_dict, number = regime_at(ctx.m4_05, ctx.m5_03, point, regime)
    seed = day_seed(number, day, offset)
    market = day_market(ctx.m4_05, regime_dict, seed, day)
    if market.digest() != expected_digest(split, name, regime, day):
        raise ValueError(f"{split} {name} {regime} day {day}: other rows than the digest")
    return market, seed


def days_step(ctx):
    """The digests of every training point's days and of every holdout point's days, written
    once, with #102's unperturbed training days and #108's selection days checked on the way."""
    DIGESTS.mkdir(exist_ok=True)
    written = 0
    plan = [("train", n, p) for n, p in training_points(ctx.m5_03)]
    plan += [("holdout", n, p) for n, p in points(ctx.m5_03)]
    for split, name, point in plan:
        path = DIGESTS / f"{split}-{name}.json"
        if path.exists():
            raise FileExistsError(f"{path.name} was written already")
        offset, days = split_days(ctx.design, ctx.m4_05, ctx.m5_03, split)
        records = []
        for regime in ctx.regimes:
            regime_dict, number = regime_at(ctx.m4_05, ctx.m5_03, point, regime)
            for day in days:
                seed = day_seed(number, day, offset)
                market = day_market(ctx.m4_05, regime_dict, seed, day)
                records.append({"regime": regime, "day": day, "seed": seed,
                                "rows": len(market.rows()), "dataset_digest": market.digest()})
        if split == "train" and name == unperturbed_name():
            m4_05 = _digest_file(str(M4_05 / "digests.json"))
            for record in records:
                if record["dataset_digest"] != m4_05[(record["regime"], record["day"])]:
                    raise ValueError(f"unperturbed {record['regime']} day {record['day']} is not "
                                     "#102's training day")
        path.write_text(json.dumps({"split": split, "point_name": name, "point": point,
                                    "seed_offset": offset, "days": records},
                                   indent=2, sort_keys=True) + "\n")
        written += 1
        print(f"days: {path.name}", file=sys.stderr, flush=True)
    checked = 0
    for name, _ in held_out_points(ctx.m5_03):
        for regime in ctx.regimes:
            for day in split_days(ctx.design, ctx.m4_05, ctx.m5_03, "selection")[1]:
                market_for(ctx, "selection", name, regime, day)
                checked += 1
    return {"digest_files": written, "selection_days_checked": checked}


# Training.


def uniform_mixture(n):
    return [1.0 / n] * n


def adversary_mixture(worst, n, unperturbed=0):
    """Phases 2 to 5: the worst point 0.50, each other 0.25 / (n - 1), the unperturbed point
    another 0.25."""
    mixture = [0.25 / (n - 1)] * n
    mixture[worst] = 0.50
    mixture[unperturbed] += 0.25
    return mixture


def worst_point(scores, names):
    """The point with the largest score; the first in grid order on a tie."""
    best = None
    for index, name in enumerate(names):
        if best is None or scores[name]["worst_loss"] > scores[names[best]]["worst_loss"]:
            best = index
    return best


def worst_loss(nets):
    """The score: the largest daily loss (loss = -net) of the scoring days."""
    return max(-float(n) for n in nets)


def residual_env(ctx, market, regime, day, stress=None):
    config = base.residual_config(ctx.m4_05, "main", "avellaneda_stoikov", ctx.baselines[regime],
                                  day, ctx.bounds)
    env = ResidualMarketMakingEnv(m503.execution_variant(config, stress),
                                  [tuple(r) for r in market.rows()])
    if env.bridge_info["dataset_digest"] != market.digest():
        env.close()
        raise ValueError("the environment holds other rows than its market")
    return env


def training_env(ctx, name, regime, day):
    market, _ = market_for(ctx, "train", name, regime, day)
    return residual_env(ctx, market, regime, day)


class Days(gym.Env):
    """#102's DayCycle: every reset draws a training day from the run's seed, by the same
    generator and in the same order. The adversarial arm then draws the episode's point from
    the phase's mixture by a generator of its own; the baseline runs every day unperturbed."""

    metadata = {"render_modes": []}

    def __init__(self, ctx, arm, seed, make=training_env):
        super().__init__()
        self._ctx, self._arm, self._seed, self._make = ctx, arm, seed, make
        self._names = [n for n, _ in training_points(ctx.m5_03)]
        self._pairs = [(r, d) for r in ctx.regimes
                       for d in split_days(ctx.design, ctx.m4_05, ctx.m5_03, "train")[1]]
        self._rng = np.random.default_rng(seed)
        self._points = None
        self.phase, self.mixture = 0, None
        self.episodes = []
        self._current = make(ctx, self._names[0], *self._pairs[0])
        self.observation_space = self._current.observation_space
        self.action_space = self._current.action_space

    def set_phase(self, phase, mixture):
        self.phase = phase
        self.mixture = None if mixture is None else list(mixture)
        self._points = np.random.default_rng([281, self._seed, phase])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        regime, day = self._pairs[int(self._rng.integers(len(self._pairs)))]
        reset_seed = int(self._rng.integers(2**63))
        if self._arm == "adversarial":
            name = self._names[int(self._points.choice(len(self._names), p=self.mixture))]
        else:
            name = self._names[0]
        self._current.close()
        self._current = self._make(self._ctx, name, regime, day)
        self.episodes.append({"phase": self.phase, "regime": regime, "day": day, "point": name,
                              "return": 0.0, "steps": 0})
        return self._current.reset(seed=reset_seed)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._current.step(action)
        if self.episodes:
            self.episodes[-1]["return"] += float(reward)
            self.episodes[-1]["steps"] += 1
        return observation, reward, terminated, truncated, info

    def close(self):
        self._current.close()


def score(model, ctx, make=training_env):
    """The current policy on every training point over the scoring days: each point's daily
    nets and its worst daily loss."""
    out = {}
    for name in [n for n, _ in training_points(ctx.m5_03)]:
        nets = []
        for regime in ctx.regimes:
            for day in SCORING_DAYS:
                env = make(ctx, name, regime, day)
                try:
                    nets.append(base.episode(model, env, EVALUATION_SEED, ctx.m4_05)["net"])
                finally:
                    env.close()
        out[name] = {"nets": nets, "worst_loss": worst_loss(nets)}
    return out


def run_phases(model, cycle, arm, names, phases, steps_each, scorer):
    """The five phases: before phases 2 to 5 the policy as it stands is scored, and the
    adversarial arm's mixture moves to the worst point. One model throughout: buffer,
    optimisers, entropy coefficient, targets and step count carry across."""
    log = []
    train_seconds = score_seconds = 0.0
    for phase in range(1, phases + 1):
        entry = {"phase": phase}
        mixture = uniform_mixture(len(names)) if arm == "adversarial" else None
        if phase > 1:
            started = time.perf_counter()
            scores = scorer(model)
            score_seconds += time.perf_counter() - started
            worst = worst_point(scores, names)
            entry.update({"scores": scores, "worst": names[worst]})
            if arm == "adversarial":
                mixture = adversary_mixture(worst, len(names))
        cycle.set_phase(phase, mixture)
        entry["mixture"] = mixture
        started = time.perf_counter()
        model.learn(total_timesteps=steps_each, reset_num_timesteps=False, progress_bar=False)
        train_seconds += time.perf_counter() - started
        entry["num_timesteps"] = int(model.num_timesteps)
        log.append(entry)
    return log, train_seconds, score_seconds


def hyperparameters(ctx):
    """#102's set 2, as this design states it."""
    sets = ctx.m4_05["arms"][SAC_ARM]["hyperparameter_sets"]
    chosen = sets[2]
    stated = {k: v for k, v in ctx.design["arms"]["hyperparameters"].items() if k != "other"}
    if chosen != stated:
        raise ValueError("#102's set 2 is not the design's hyperparameters")
    return chosen


def train_job(job):
    """One training run, in its own process on one thread."""
    torch.set_num_threads(1)
    ctx = Context(job["run_dir"])
    seed, arm = job["seed"], job["arm"]
    set_random_seed(seed)
    cycle = Days(ctx, arm, seed)
    model = SAC("MlpPolicy", DummyVecEnv([lambda: Monitor(cycle)]), seed=seed, device="cpu",
                **hyperparameters(ctx))
    path = pathlib.Path(job["path"])
    model.set_logger(configure(str(path.with_suffix(".sb3")), ["csv"]))
    names = [n for n, _ in training_points(ctx.m5_03)]
    phases = ctx.design["training"]["phases"]
    log, train_seconds, score_seconds = run_phases(
        model, cycle, arm, names, phases["count"], job.get("steps_each", phases["steps_each"]),
        lambda m: score(m, ctx))
    model.save(path)
    configs, spec, fingerprint = {}, None, None
    for regime in ctx.regimes:
        env = training_env(ctx, unperturbed_name(), regime, 1)
        configs[regime] = env.config
        spec, fingerprint = list(env.spec_id), env.bridge_info["kernel_fingerprint"]
        env.close()
    metadata = {
        "algorithm": "sac_residual", "arm": SAC_ARM, "study_arm": arm, "seed": seed,
        "steps": int(model.num_timesteps), "hyperparameters": hyperparameters(ctx),
        "spec": spec, "kernel_fingerprint": fingerprint, "configs": configs,
        "phases": log, "episodes": cycle.episodes, "commit": job["commit"],
        "design_sha256": sha256(HERE / "design.json"), "code_sha256": sha256(HERE / "run.py"),
        "versions": base.versions(), "train_seconds": round(train_seconds, 2),
        "score_seconds": round(score_seconds, 2),
        "score_episodes": (phases["count"] - 1) * len(names) * len(ctx.regimes)
        * len(SCORING_DAYS),
        "checkpoint_sha256": sha256(path.with_suffix(".zip")), "path": str(path),
    }
    cycle.close()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return {k: v for k, v in metadata.items() if k not in ("episodes", "phases", "configs")}


def run_name(arm, seed):
    return f"{arm}-seed{seed}"


class Overrun(RuntimeError):
    """A step ran past 1.5 times its estimate: it stops, and is reported."""


def limit_seconds(out, step):
    """The design's stop rule: 1.5 times the step's estimate, from `estimates.json` in the
    output directory, which is written from the rate step before the step runs."""
    path = pathlib.Path(out) / "estimates.json"
    if not path.exists():
        raise ValueError(f"no estimate for {step}: write estimates.json from the rate step first")
    estimates = json.loads(path.read_text())
    if step not in estimates:
        raise ValueError(f"estimates.json has no estimate for {step}")
    return 1.5 * float(estimates[step])


def stop_workers(executor):
    """Kills this pool's own worker processes, each by its exact pid, and nothing else."""
    for process in list(getattr(executor, "_processes", {}).values()):
        if process.is_alive():
            process.kill()
    executor.shutdown(wait=True, cancel_futures=True)


def outcomes(attempts):
    """Per run: its starts, and how its last attempt ended. "failed" is an error inside the run
    (a program or numerical error), "interrupted" a worker killed from outside, "stopped" the
    overrun rule; a start with no end was killed with the step."""
    out = {}
    if pathlib.Path(attempts).exists():
        for line in pathlib.Path(attempts).read_text().splitlines():
            entry = json.loads(line)
            record = out.setdefault(entry["run"], {"starts": 0, "end": None})
            if "started" in entry:
                record["starts"] += 1
                record["end"] = None
            else:
                record["end"] = entry
    return out


def rerunnable(name, record, rerun):
    """A run that was started may run again only when it was killed from outside (no end, or
    an interrupted worker), when it is named, and once."""
    if record is None:
        return True
    end = record["end"]
    if end is not None and end["outcome"] in ("failed", "stopped"):
        raise ValueError(f"{name} ended {end['outcome']}: {end.get('error')}; it is a failed "
                         "seed, reported, and never rerun")
    if name not in rerun or record["starts"] > 1:
        raise ValueError(f"{name} was started before and did not end; it is rerun only when "
                         "it was killed from outside, named, and once")
    return True


def train(ctx, out, commit, rerun=(), limit=None, worker=None):
    """The 10 runs. Each start and each end is journaled in train.attempts."""
    out = pathlib.Path(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    attempts = out / "train.attempts"
    limit = limit_seconds(out, "train") if limit is None else limit
    worker = worker or train_job
    known = outcomes(attempts)
    jobs = []
    for arm in TRAINED:
        for seed in ctx.design["arms"]["training_seeds"]:
            name = run_name(arm, seed)
            path = out / "checkpoints" / name
            if path.with_suffix(".json").exists():
                continue
            rerunnable(name, known.get(name), rerun)
            jobs.append({"run_dir": ctx.run_dir, "arm": arm, "seed": seed, "path": str(path),
                         "commit": commit})
    context = multiprocessing.get_context("spawn")
    results = []
    started = time.perf_counter()
    executor = concurrent.futures.ProcessPoolExecutor(ctx.parallel, mp_context=context)
    ended = set()
    with attempts.open("a") as journal:

        def end(name, outcome, error=None):
            ended.add(name)
            journal.write(json.dumps({"run": name, "ended": time.time(), "outcome": outcome,
                                      "error": error}) + "\n")
            journal.flush()

        futures = {}
        for job in jobs:
            journal.write(json.dumps({"run": pathlib.Path(job["path"]).name,
                                      "started": time.time(), "pid": os.getpid()}) + "\n")
            journal.flush()
            futures[executor.submit(worker, job)] = pathlib.Path(job["path"]).name
        try:
            remaining = limit - (time.perf_counter() - started)
            for future in concurrent.futures.as_completed(futures, timeout=max(remaining, 0)):
                name = futures[future]
                try:
                    results.append(future.result())
                    end(name, "done")
                    print(f"train: {name} done", file=sys.stderr, flush=True)
                except concurrent.futures.process.BrokenProcessPool as error:
                    end(name, "interrupted", f"{type(error).__name__}: {error}")
                    results.append({"run": name, "interrupted": str(error)})
                except Exception as error:  # a failed run is reported, never dropped
                    end(name, "failed", f"{type(error).__name__}: {error}")
                    results.append({"run": name, "failed": f"{type(error).__name__}: {error}"})
                    print(f"train: {name} FAILED: {error}", file=sys.stderr, flush=True)
        except concurrent.futures.TimeoutError:
            # Every run without a journaled end is stopped, before the kill: a killed worker's
            # future then reads as done with BrokenProcessPool, which is not its outcome.
            for name in futures.values():
                if name not in ended:
                    end(name, "stopped", f"over 1.5 times the estimate ({limit:.0f} s)")
            stop_workers(executor)
            raise Overrun(f"train ran past {limit:.0f} s; the unfinished runs are stopped")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return {"step": "train", "commit": commit, "runs": results}


# Evaluation.


def trained_runs(out):
    """The trained runs' metadata, A and B, seeds 1 to 5 each: None for a run that failed in
    training (a failed seed, kept), and an error for one that never ended."""
    known = outcomes(pathlib.Path(out) / "train.attempts")
    runs = {}
    for arm in TRAINED:
        found = []
        for seed in (1, 2, 3, 4, 5):
            name = run_name(arm, seed)
            path = pathlib.Path(out) / "checkpoints" / f"{name}.json"
            if path.exists():
                found.append(json.loads(path.read_text()))
                continue
            end = (known.get(name) or {}).get("end")
            if end is None or end["outcome"] not in ("failed", "stopped"):
                raise ValueError(f"{name} has no checkpoint and did not finish: train it first")
            found.append(None)
        runs[arm] = found
    return runs


def policies(ctx, out):
    """Every policy: (arm, seed, metadata), the reference's from #102's frozen runs; a failed
    seed's metadata is None, and its days are missing."""
    out_list = []
    for arm, found in trained_runs(out).items():
        for seed, metadata in zip((1, 2, 3, 4, 5), found, strict=True):
            out_list.append((arm, seed, metadata))
    for metadata in m503.learned_runs(ctx.frozen, ctx.run_dir)["residual_sac"]:
        out_list.append(("reference", metadata["seed"], metadata))
    return out_list


def finite(net):
    try:
        return Decimal(net).is_finite()
    except Exception:
        return False


def evaluate(ctx, policy, market, seed, regime, day, stress=None, impact=None):
    """One day: the episode as #108 measures it, under the execution stress if given, or in
    rows that react to the agent if `impact` holds #282's runner and kappa."""
    if impact is None:
        env = residual_env(ctx, market, regime, day, stress)
        try:
            result = base.episode(policy, env, EVALUATION_SEED, ctx.m4_05)
            positions = dict(env.attribution())["positions"]
        finally:
            env.close()
        extra = {}
    else:
        module, kappa = impact
        config = base.residual_config(ctx.m4_05, "main", "avellaneda_stoikov",
                                      ctx.baselines[regime], day, ctx.bounds)

        def make_env(rows):
            return ResidualMarketMakingEnv(config, [tuple(r) for r in rows])

        result = module.reactive_episode(policy, make_env, market, kappa,
                                         module.impact_rng(seed),
                                         EVALUATION_SEED, ctx.m4_05)
        positions = result["positions"]
        extra = {"kappa": kappa, "reactions": result["reactions"], "moves": result["moves"],
                 "emptied": result["emptied"], "replayed_steps": result["replayed_steps"],
                 "stopped": result["stopped"], "reacted_digest": market.digest()}
    peak, peak_price = Decimal(0), None
    for _, price, after in positions:
        if abs(Decimal(after)) > abs(peak):
            peak, peak_price = Decimal(after), price
    attribution = result["attribution"]
    return {"net": result["net"], "max_drawdown": attribution.get("max_drawdown"),
            "max_abs_position": attribution.get("max_abs_position"),
            "mean_abs_position": attribution.get("mean_abs_position"),
            "fills": attribution.get("fills"),
            "peak_notional": result["peak_notional"], "refusals": result["refusals"],
            "residual_fallbacks": result["residual_fallbacks"],
            "residual_clipped": result["residual_clipped"],
            "limit_breach": result["limit_breach"], "peak_position": str(peak),
            "peak_position_price": peak_price, **extra}


def run_policy(job):
    """One policy seed on one day of one condition. The rows and the checkpoint are checked
    first, and a mismatch stops the step. An episode that then errors, or ends with a net that
    is not finite, leaves the seed-day missing, with the error."""
    torch.set_num_threads(1)
    ctx = Context(job["run_dir"])
    split, name, regime, day = job["split"], job["point"], job["regime"], job["day"]
    market, seed = market_for(ctx, split, name, regime, day, split == "holdout")
    metadata = job["metadata"]
    row = {"condition": job["condition"], "split": split, "point": name, "arm": job["arm"],
           "seed": job["seed"], "regime": regime, "day": day, "day_seed": seed,
           "dataset_digest": market.digest()}
    if metadata is None:
        return {**row, "net": None, "error": "the seed failed in training"}
    if job.get("frozen_sha256") and sha256(pathlib.Path(metadata["path"]).with_suffix(".zip")) \
            != job["frozen_sha256"]:
        raise ValueError(f"{job['arm']} seed {job['seed']}: the checkpoint changed")
    probe = residual_env(ctx, market, regime, day)
    try:
        policy = m503.load_frozen(metadata, probe, regime)
    finally:
        probe.close()
    impact = None
    if job.get("impact"):
        module = impact_module()
        if module is None:
            raise ValueError("the impact condition needs #282's runner")
        impact = (module, job["kappa"])
    try:
        result = evaluate(ctx, policy, market, seed, regime, day, job.get("execution"), impact)
    except Exception as error:  # the seed-day is missing, never dropped
        return {**row, "net": None, "error": f"{type(error).__name__}: {error}"}
    if not finite(result["net"]):
        return {**row, **result, "net": None, "error": f"non-finite net {result['net']}"}
    return {**row, **result}


def job_key(job):
    return (job["condition"], job["arm"], job["seed"], job["regime"], job["day"])


def pool(jobs, worker, parallel, label, journal, limit):
    """Runs the jobs, appending each result to `journal` as it ends, and skips the jobs a
    previous run of the same step journaled. A result does not depend on the order or the
    number of processes. Past `limit` seconds the step stops (the design's overrun rule): its
    workers are killed, what ended stays journaled, and Overrun is raised."""
    journal = pathlib.Path(journal)
    done_rows = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            done_rows[tuple(json.loads(entry["key"]))] = entry["row"]
    keys = [job_key(job) for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("two jobs share a key")
    todo = [(k, j) for k, j in zip(keys, jobs, strict=True) if k not in done_rows]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    print(f"{label}: {len(jobs) - len(todo)} of {len(jobs)} already done; {parallel} processes",
          file=sys.stderr, flush=True)
    executor = concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context)
    with journal.open("a") as out:
        futures = {executor.submit(worker, job): key for key, job in todo}
        try:
            remaining = max(limit - (time.perf_counter() - started), 0)
            for done, future in enumerate(
                    concurrent.futures.as_completed(futures, timeout=remaining), start=1):
                row = future.result()
                key = futures[future]
                done_rows[key] = row
                out.write(json.dumps({"key": json.dumps(list(key)), "row": row}) + "\n")
                out.flush()
                if done % 50 == 0 or done == len(todo):
                    print(f"{label}: {done}/{len(todo)} after "
                          f"{round(time.perf_counter() - started)} s", file=sys.stderr, flush=True)
        except concurrent.futures.TimeoutError:
            stop_workers(executor)
            raise Overrun(f"{label} ran past {limit:.0f} s; {len(done_rows)} of {len(jobs)} "
                          "jobs ended and are journaled") from None
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return [done_rows[key] for key in keys]


def policy_jobs(ctx, out, condition, split, name, arms=ARMS, frozen_models=None, extra=None):
    jobs = []
    _, days = split_days(ctx.design, ctx.m4_05, ctx.m5_03, split)
    for arm, seed, metadata in policies(ctx, out):
        if arm not in arms:
            continue
        frozen = None
        if frozen_models is not None:
            frozen = frozen_models[f"{arm}-seed{seed}"].get("checkpoint_sha256")
        for regime in ctx.regimes:
            for day in days:
                job = {"run_dir": ctx.run_dir, "condition": condition, "split": split,
                       "point": name, "arm": arm, "seed": seed, "regime": regime, "day": day,
                       "metadata": metadata, "frozen_sha256": frozen}
                if extra is not None:
                    job.update(extra(regime))
                jobs.append(job)
    return jobs


def seed_averaged(rows, arm):
    """Per (regime, day), the mean net over the arm's seeds; None when a seed-day is missing."""
    out = {}
    for row in rows:
        if row["arm"] == arm:
            out.setdefault((row["regime"], row["day"]), []).append(row["net"])
    return {k: (None if any(v is None for v in vs) else
                sum((Decimal(v) for v in vs), Decimal(0)) / len(vs)) for k, vs in out.items()}


def cvar_loss(nets, share=0.05):
    """The mean loss of the worst ceil(share x n) days; None when a day is missing."""
    if any(n is None for n in nets):
        return None
    return m503.cvar_loss(nets, share)


def reproduction(rows, check):
    """B's reproduction check: the mean over B's seeds of each seed's mean validation daily net,
    against #102's, within the design's tolerance. A missing seed-day fails it."""
    per_seed = {}
    for row in rows:
        if row["arm"] == "baseline":
            per_seed.setdefault(row["seed"], []).append(row["net"])
    reference = check["reference"]
    tolerance = 2 * reference["sample_sd"] * math.sqrt(2 / 5)
    if sorted(per_seed) != [1, 2, 3, 4, 5] or any(
            n is None for nets in per_seed.values() for n in nets):
        return {"passed": False, "reason": "a seed-day of B is missing", "tolerance": tolerance}
    means = {s: float(sum(Decimal(n) for n in nets) / len(nets)) for s, nets in per_seed.items()}
    statistic = sum(means.values()) / len(means)
    difference = statistic - reference["mean"]
    return {"passed": abs(difference) <= tolerance, "statistic": statistic,
            "per_seed": means, "reference_mean": reference["mean"], "difference": difference,
            "tolerance": tolerance}


def validate(ctx, out, commit):
    jobs = policy_jobs(ctx, out, "validation", "validation", unperturbed_name())
    rows = pool(jobs, run_policy, ctx.parallel, "validate", pathlib.Path(out) / "validate.journal",
                limit_seconds(out, "validate"))
    report = ctx.header("validate", commit)
    report["rows"] = rows
    report["reproduction"] = reproduction(rows, ctx.design["validation"]["reproduction_check"])
    ctx.write(out, "validation", report)
    return report


def freeze_models(ctx, out, commit):
    """The checkpoints' hashes, once, after B passed its check and before any selection run."""
    validation = json.loads((pathlib.Path(out) / "validation.json").read_text())
    if not validation["reproduction"]["passed"]:
        raise ValueError("B did not pass its reproduction check: the study stops")
    if FROZEN_MODELS.exists():
        raise FileExistsError("the models were frozen already")
    models = {}
    for arm, found in trained_runs(out).items():
        for seed, metadata in zip((1, 2, 3, 4, 5), found, strict=True):
            if metadata is None:
                models[run_name(arm, seed)] = {"failed": "the seed failed in training"}
                continue
            path = pathlib.Path(metadata["path"])
            if sha256(path.with_suffix(".zip")) != metadata["checkpoint_sha256"]:
                raise ValueError(f"{run_name(arm, seed)} changed after training")
            models[run_name(arm, seed)] = {
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "metadata_sha256": sha256(path.with_suffix(".json"))}
    for metadata in m503.learned_runs(ctx.frozen, ctx.run_dir)["residual_sac"]:
        path = pathlib.Path(metadata["path"]).with_suffix(".zip")
        models[f"reference-seed{metadata['seed']}"] = {"checkpoint_sha256": sha256(path)}
    manifest = {"commit": commit, "validation_sha256": sha256(pathlib.Path(out) / "validation.json"),
                "models": models}
    FROZEN_MODELS.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"frozen_models_sha256": sha256(FROZEN_MODELS)}


def check_models(ctx, out):
    """The frozen models, each checkpoint and trained metadata unchanged."""
    if not FROZEN_MODELS.exists():
        raise ValueError("the models are not frozen")
    models = json.loads(FROZEN_MODELS.read_text())["models"]
    for arm, seed, metadata in policies(ctx, out):
        frozen = models[f"{arm}-seed{seed}"]
        if (metadata is None) != ("failed" in frozen):
            raise ValueError(f"{arm} seed {seed}: failed or not, it is not as frozen")
        if metadata is None:
            continue
        path = pathlib.Path(metadata["path"])
        if sha256(path.with_suffix(".zip")) != frozen["checkpoint_sha256"]:
            raise ValueError(f"{arm} seed {seed}: the checkpoint changed after the freeze")
        if "metadata_sha256" in frozen and sha256(path.with_suffix(".json")) \
                != frozen["metadata_sha256"]:
            raise ValueError(f"{arm} seed {seed}: the metadata changed after the freeze")
    return models


def choose(by_point, arms=ARMS, share=0.05):
    """Per arm, the held-out point with the largest CVaR of the seed-averaged daily loss, the
    first in grid order on a tie. A point with a missing seed-day leaves the arm's choice
    failed, with no point."""
    chosen = {}
    for arm in arms:
        best, failed = None, []
        for name, nets in by_point.items():
            value = cvar_loss(nets[arm], share)
            if value is None:
                failed.append(name)
                continue
            if best is None or value > best[1]:
                best = (name, value)
        if failed:
            chosen[arm] = {"point": None, "failed": failed}
        else:
            chosen[arm] = {"point": best[0], "cvar_loss": best[1]}
    return chosen


def selection(ctx, out, commit):
    models = check_models(ctx, out)
    jobs = []
    for name, _ in held_out_points(ctx.m5_03):
        jobs += policy_jobs(ctx, out, f"held_out:{name}", "selection", name,
                            frozen_models=models)
    rows = pool(jobs, run_policy, ctx.parallel, "selection",
                pathlib.Path(out) / "selection.journal", limit_seconds(out, "selection"))
    by_point = {}
    for name, _ in held_out_points(ctx.m5_03):
        scoped = [r for r in rows if r["point"] == name]
        by_point[name] = {}
        for arm in ARMS:
            averaged = seed_averaged(scoped, arm)
            by_point[name][arm] = [averaged[k] for k in sorted(averaged)]
    report = ctx.header("selection", commit)
    report["rows"] = rows
    report["chosen"] = choose(by_point)
    ctx.write(out, "selection", report)
    return report


# The impact condition, through #282's runner when it is on main.


def impact_module():
    path = M5_04 / "run.py"
    if not path.exists():
        return None
    return _module("m5_04_run", path)


def impact_provenance(ctx):
    """#282's runner and its kappa0, checked against this design; None when the runner is not
    here, which leaves the condition not run."""
    if not (M5_04 / "run.py").exists():
        return None
    m5_04 = json.loads((M5_04 / "design.json").read_text())
    measured = m5_04["impact_scale"]["measurement"]
    stated = ctx.design["reuses"]["m5_04"]["kappa0_measurement"]
    if measured["commit"] != stated["commit"] or measured["stdout_sha256"] != stated["stdout_sha256"]:
        raise ValueError("#282's kappa0 is not the measurement this design states")
    kappa = {r: 2 * m5_04["impact_scale"]["values"][r]["kappa0"] for r in ctx.regimes}
    return {"run_sha256": sha256(M5_04 / "run.py"), "design_sha256": sha256(M5_04 / "design.json"),
            "kappa": kappa}


def digest_hashes():
    return m503.digest_hashes(DIGESTS)


def freeze(ctx, out, commit):
    out = pathlib.Path(out)
    check_models(ctx, out)
    selection_path = out / "selection.json"
    manifest = ctx.header("freeze", commit)
    manifest["selection_sha256"] = sha256(selection_path)
    manifest["frozen_models_sha256"] = sha256(FROZEN_MODELS)
    manifest["digests_sha256"] = digest_hashes()
    manifest["chosen"] = json.loads(selection_path.read_text())["chosen"]
    manifest["impact"] = impact_provenance(ctx)
    path = out / "frozen.json"
    if path.exists():
        raise FileExistsError("the study was frozen already")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"frozen_sha256": sha256(path)}


def holdout_jobs(ctx, out, frozen, models):
    """Every holdout condition's jobs: A, B and the reference at the unperturbed rows, each
    held-out point, the execution stress and the impact (when #282's runner is here); A and B
    at the perturbed training points, in sample."""
    unperturbed = unperturbed_name()
    jobs = policy_jobs(ctx, out, "unperturbed", "holdout", unperturbed, frozen_models=models)
    for name, _ in held_out_points(ctx.m5_03):
        jobs += policy_jobs(ctx, out, f"held_out:{name}", "holdout", name, frozen_models=models)
    stress = ctx.design["holdout"]["conditions"]["execution_stress"]
    fill_model = ctx.m5_03["execution_stress"]["fill_model"]
    if "queue_position_rational" not in stress:
        raise ValueError("the execution stress is not #108's")
    jobs += policy_jobs(ctx, out, "execution_stress", "holdout", unperturbed,
                        frozen_models=models, extra=lambda _: {"execution": fill_model})
    if frozen["impact"] is not None:
        kappa = frozen["impact"]["kappa"]
        jobs += policy_jobs(ctx, out, "impact", "holdout", unperturbed, frozen_models=models,
                            extra=lambda regime: {"impact": True, "kappa": kappa[regime]})
    for name, _ in training_points(ctx.m5_03):
        if name != unperturbed:
            jobs += policy_jobs(ctx, out, f"in_sample:{name}", "holdout", name, arms=TRAINED,
                                frozen_models=models)
    return jobs


def holdout(ctx, out, commit, frozen_sha256):
    out = pathlib.Path(out)
    path = out / "frozen.json"
    if sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    frozen = json.loads(path.read_text())
    here = ctx.header("holdout", commit)
    for key in ("design_sha256", "code_sha256"):
        if frozen[key] != here[key]:
            raise ValueError(f"{key} changed after the freeze")
    if frozen["selection_sha256"] != sha256(out / "selection.json"):
        raise ValueError("selection.json changed after the freeze")
    if frozen["frozen_models_sha256"] != sha256(FROZEN_MODELS):
        raise ValueError("frozen_models.json changed after the freeze")
    m503.check_frozen_digests(frozen, DIGESTS)
    if impact_provenance(ctx) != frozen["impact"]:
        raise ValueError("#282's runner changed after the freeze")
    models = check_models(ctx, out)
    jobs = holdout_jobs(ctx, out, frozen, models)
    limit = limit_seconds(out, "holdout")
    # Before any day is read: a second or concurrent attempt is refused, and a failed one is
    # never rerun.
    try:
        marker = os.open(out / "holdout.started", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("the holdout was attempted already") from error
    with os.fdopen(marker, "w") as handle:
        handle.write(json.dumps({"started": time.time(), "pid": os.getpid(),
                                 "frozen_sha256": frozen_sha256}) + "\n")
    rows = pool(jobs, run_policy, ctx.parallel, "holdout", out / "holdout.journal", limit)
    here["frozen_sha256"] = frozen_sha256
    here["chosen"] = frozen["chosen"]
    here["impact_run"] = frozen["impact"] is not None
    here["rows"] = rows
    ctx.write(out, "holdout", here)
    return here


def rate(ctx, out, commit):
    """One adversarial run's first 10000 steps and one scoring pass, timed, on training days."""
    torch.set_num_threads(1)
    set_random_seed(1)
    cycle = Days(ctx, "adversarial", 1)
    cycle.set_phase(1, uniform_mixture(len(training_points(ctx.m5_03))))
    model = SAC("MlpPolicy", DummyVecEnv([lambda: Monitor(cycle)]), seed=1, device="cpu",
                **hyperparameters(ctx))
    started = time.perf_counter()
    model.learn(total_timesteps=10_000, reset_num_timesteps=False, progress_bar=False)
    learn = time.perf_counter() - started
    started = time.perf_counter()
    score(model, ctx)
    scoring = time.perf_counter() - started
    cycle.close()
    report = ctx.header("rate", commit)
    report.update({"learn_10000_seconds": round(learn, 1), "score_pass_seconds": round(scoring, 1)})
    ctx.write(out, "rate", report)
    return report


class Context:
    """What every step shares: the designs, #102's frozen runs and the compute cap."""

    def __init__(self, run_dir):
        torch.set_num_threads(1)
        self.design, self.m4_05, self.m5_03 = load()
        self.run_dir = str(run_dir)
        frozen_path = pathlib.Path(run_dir) / "frozen.json"
        if sha256(frozen_path) != self.design["reuses"]["m4_05"]["frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one this study reuses")
        self.frozen = json.loads(frozen_path.read_text())
        self.regimes = [r["name"] for r in self.m4_05["simulator"]["regimes"]]
        self.baselines = {r: self.frozen["classical_selected"][f"{r}/avellaneda_stoikov"]
                          for r in self.regimes}
        self.bounds = self.m4_05["arms"][SAC_ARM]["bounds"]
        self.points = points(self.m5_03)
        # The team's shared compute cap may lower the design's process count; it changes no
        # result.
        cap = self.design["compute"]["max_parallel_runs"]
        self.parallel = min(cap, int(os.environ.get("M5_05_PROCESSES", cap)))

    def header(self, step, commit):
        return {"step": step, "commit": commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "design_sha256": sha256(HERE / "design.json"),
                "code_sha256": sha256(HERE / "run.py"), "versions": base.versions()}

    def write(self, out, name, report):
        path = pathlib.Path(out)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def main():
    step, run_dir, out_dir, commit = sys.argv[1:5]
    started = time.perf_counter()
    ctx = Context(run_dir)
    if step == "days":
        report = days_step(ctx)
    elif step == "rate":
        report = rate(ctx, out_dir, commit)
    elif step == "train":
        report = train(ctx, out_dir, commit, tuple(sys.argv[5:]))
        ctx.write(out_dir, f"train-{int(time.time())}", report)
    elif step == "validate":
        report = validate(ctx, out_dir, commit)
    elif step == "freeze_models":
        report = freeze_models(ctx, out_dir, commit)
    elif step == "selection":
        report = selection(ctx, out_dir, commit)
    elif step == "freeze":
        report = freeze(ctx, out_dir, commit)
    elif step == "holdout":
        report = holdout(ctx, out_dir, commit, sys.argv[5])
    else:
        raise SystemExit(f"unknown step {step!r}")
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps({"step": step, "seconds": round(time.perf_counter() - started, 1),
                      **summary}, indent=2, default=str))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
