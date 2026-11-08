"""Runs a step of #280's offline training study from its committed design.

From `bridges/python`, with #102's rows and run directory and #104's output directory:

    python experiments/m4_11/run.py <step> <data dir> <m4_05 run dir> <m4_07 out dir> <out dir> <commit> [args]

Steps, in order:
- `pilot`: times per gradient step, per log step and per evaluation step, on training days;
- `gate`: the queue gate and the determinism check, on validation days;
- `logs <dataset> <split>`: a standard dataset's logs, on train or validation days;
- `ppo`: the maskable PPO comparator, retrained under the standard execution;
- `train <dataset> <arm> [alpha]`: five seeds of BC, or of CQL at an alpha;
- `sanity <dataset>`: BC's agreement with as_grid on the dataset's validation logs;
- `select`: CQL's alpha, by interactive runs on validation days;
- `consistency`: the secondary dataset's check against #102's validation values;
- `freeze`: the hashes of the design, the code, the choice, every log and every checkpoint;
- `holdout <frozen sha256>`: every arm on the holdout days, once.

Everything here is synthetic: #102's queue-reactive days, whose rows do not react to orders.
Only `holdout` reads the holdout, and only after `freeze`.
"""

import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import os
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module("m4_05_run", EXPERIMENTS / "m4_05" / "run.py")
m4_07 = _module("m4_07_run", EXPERIMENTS / "m4_07" / "run.py")
est = _module("m4_07_estimate", EXPERIMENTS / "m4_07" / "estimate.py")
offline = _module("m4_11_offline", HERE / "offline.py")

from mm_rl_bridge import MaskableMarketMakingEnv  # noqa: E402

SEED = 101
POSITION = 6
CODE = ("run.py", "offline.py")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load():
    design = json.loads((HERE / "design.json").read_text())
    reused = design["reuses"]
    for path, key in ((EXPERIMENTS / "m4_05" / "design.json", "m4_05_design_sha256"),
                      (EXPERIMENTS / "m4_05" / "digests.json", "m4_05_digests_sha256"),
                      (EXPERIMENTS / "m4_07" / "design.json", "m4_07_design_sha256")):
        if sha256(path) != reused[key]:
            raise ValueError(f"{path.name} of {path.parent.name} is not the one this study reuses")
    m4_05, digests = base.load_design()
    return design, m4_05, digests


# Configurations and policies.


def config_of(study_config, execution, design):
    """#102's maskable configuration with the execution's fill model; latency and everything
    else stay #102's main."""
    config = dict(study_config)
    if execution == "m4_05_main":
        return config
    config["fill_model"] = dict(design["executions"][execution]["fill_model"])
    return config


def rule(job, regime, grid, m4_05, name):
    """A comparator rule as a callable on (observation, raw, mask)."""
    if name == "no_trade":
        return m4_07.constant(m4_07.HOLD)
    fixed = job["selected"][f"{regime}/fixed_spread"]
    if name == "fixed_spread_4":
        index = grid.offsets.index(fixed["offset_ticks"])
        size = grid.sizes.index(int(fixed["size_lots"]))
        return m4_07.constant((1, index, index, size, size))
    if name == "as_grid":
        point = job["selected"][f"{regime}/avellaneda_stoikov"]
        return m4_07.AsGrid(grid, point, m4_05["kernel"]["tick_size"],
                            m4_05["simulator"]["sample_ms"], m4_05["simulator"]["rows_per_day"])
    raise ValueError(f"no rule {name}")


def network_of(metadata):
    network = offline.mlp(metadata["inputs"], metadata["outputs"], metadata["settings"]["hidden"])
    path = pathlib.Path(metadata["path"])
    if sha256(path) != metadata["checkpoint_sha256"]:
        raise ValueError(f"{path} is not the checkpoint its record names")
    network.load_state_dict(torch.load(path, weights_only=True))
    return network


def policy_of(job, regime, env, grid, m4_05):
    """The evaluated arm as a callable on (observation, raw, mask), and whether it is an
    offline arm, whose support is measured."""
    spec = job["policy"]
    if spec["kind"] == "rule":
        return rule(job, regime, grid, m4_05, spec["name"]), False
    if spec["kind"] == "frozen_ppo":
        return m4_07.frozen_ppo(m4_07.load_frozen(spec["metadata"], env, regime)), False
    if spec["kind"] == "ppo":
        return m4_07.frozen_ppo(base.load_frozen(spec["metadata"], env, regime)), False
    actions = offline.Actions(len(grid.offsets), len(grid.sizes))
    trained = offline.Policy(network_of(spec["metadata"]), actions)
    return (lambda observation, raw, mask: trained.predict(observation, action_masks=mask)[0]), True


# Jobs, each in its own process on one thread.


def make_env(job):
    design, m4_05, digests = load()
    rows, record = base.read_rows(job["data_dir"], digests, job["regime"], job["day"],
                                  job["split"], holdout_allowed=job.get("holdout_allowed", False))
    return base.checked(MaskableMarketMakingEnv(job["config"], rows), record), design, m4_05


def run_job(job):
    """One arm on one day at the environment seed: the exact net, its attribution and, for an
    offline arm, the support of its decisions."""
    torch.set_num_threads(1)
    env, design, m4_05 = make_env(job)
    grid = m4_07.grid_of(job["config"])
    try:
        policy, measured = policy_of(job, job["regime"], env, grid, m4_05)
        behaviour = None
        if measured:
            greedy_rule = rule(job, job["regime"], grid, m4_05, "as_grid")
            behaviour = m4_07.Behaviour(grid, greedy_rule, job["epsilon"], 0)
        observation, info = env.reset(seed=job["seed"])
        steps = refusals = 0
        seen = {"observation": [], "code": [], "position": [], "propensity": []}
        scale = float(job["config"]["scales"]["position"])
        while True:
            raw, mask = info["raw_observation"], env.action_masks()
            action = grid.canonical(policy(observation, raw, mask))
            if behaviour is not None:
                greedy = grid.canonical(behaviour.greedy(observation, raw, mask))
                seen["observation"].append(observation)
                seen["code"].append(grid.code(action))
                seen["position"].append(float(observation[POSITION]) * scale)
                seen["propensity"].append(behaviour.probability(mask, greedy, action))
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            steps += 1
            refusals += info["refused_because"] is not None
            if terminated or truncated:
                break
        attribution = dict(env.attribution())
    finally:
        env.close()
    limits = job["config"]["limits"]
    peak = max((abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
               default=Decimal(0))
    row = {
        "arm": job["arm"], "seed_of_arm": job.get("seed_of_arm"), "regime": job["regime"],
        "day": job["day"], "split": job["split"], "execution": job["execution"],
        "environment_seed": job["seed"], "steps": steps, "net": info["net"],
        "fills": attribution.get("fills"), "max_abs_position": attribution.get("max_abs_position"),
        "refusals": refusals, "peak_notional": str(peak),
        "time_beyond_5_lots": base.time_beyond(attribution, 5),
        "limit_breach": Decimal(attribution["max_abs_position"]) > Decimal(limits["max_position"])
        or peak > Decimal(limits["max_notional"]),
        "label": "[simulated]",
    }
    if measured:
        row["support_raw"] = {k: np.asarray(v).tolist() for k, v in seen.items()
                              if k != "observation"}
        row["support_bins"] = state_bins(np.asarray(seen["observation"]),
                                         np.asarray(seen["position"]), job["edges"]).tolist()
    return row


def state_bins(observations, positions, edges):
    """#104's state bin crossed with the signed position bin."""
    if len(observations) == 0:
        return np.zeros(0, dtype=int)
    return est.bin_states(observations, edges) * 5 + offline.position_bin(positions)


def log_job(job):
    """One behaviour, one seed, one day of a standard dataset: #104's log with as_grid as the
    only target, whose code the BC sanity rule reads."""
    torch.set_num_threads(1)
    env, design, m4_05 = make_env(job)
    grid = m4_07.grid_of(job["config"])
    try:
        greedy_rule = rule(job, job["regime"], grid, m4_05, "as_grid")
        behaviour = m4_07.Behaviour(grid, greedy_rule, job["epsilon"], job["seed"])
        observation, info = env.reset(seed=job["seed"])
        rows = {k: [] for k in ("observation", "mask", "action", "propensity", "reward", "refused",
                                "target_code", "target_mu")}
        while True:
            mask = env.action_masks()
            action, greedy, propensity = behaviour.act(observation, info["raw_observation"], mask)
            rows["observation"].append(observation)
            rows["mask"].append(mask)
            rows["action"].append(grid.code(action))
            rows["propensity"].append(propensity)
            rows["target_code"].append(grid.code(greedy))
            rows["target_mu"].append(behaviour.probability(mask, greedy, greedy))
            observation, _, terminated, truncated, info = env.step(np.asarray(action))
            rows["reward"].append(float(Decimal(info["reward_exact"])))
            rows["refused"].append(info["refused_because"] is not None)
            if terminated or truncated:
                break
    finally:
        env.close()
    path = pathlib.Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observation=np.asarray(rows["observation"], dtype=np.float32),
        mask=np.asarray(rows["mask"], dtype=bool),
        action=np.asarray(rows["action"], dtype=np.int16),
        propensity=np.asarray(rows["propensity"]),
        reward=np.asarray(rows["reward"]),
        refused=np.asarray(rows["refused"], dtype=bool),
        target_names=np.asarray(["as_grid"]),
        target_code=np.asarray([rows["target_code"]], dtype=np.int16),
        target_mu=np.asarray([rows["target_mu"]]),
    )
    return {"regime": job["regime"], "day": job["day"], "split": job["split"],
            "behaviour": job["behaviour"], "seed": job["seed"], "steps": len(rows["action"]),
            "net": info["net"], "path": str(path), "sha256": sha256(path), "label": "[observed]"}


def train_job(job):
    """One offline training run: the dataset's training logs, its reward scale, one seed."""
    torch.set_num_threads(1)
    days = read_logs(job["logs"])
    actions = offline.Actions(job["k"], job["m"])
    data = offline.Transitions(actions, days, job["reward_scale"])
    started = time.perf_counter()
    network, history, finished = offline.train(job["arm"], data, job["seed"], job["settings"])
    seconds = time.perf_counter() - started
    path = pathlib.Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)
    metadata = {
        "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
        "settings": job["settings"], "reward_scale": job["reward_scale"],
        "inputs": int(data.observation.shape[1]), "outputs": int(data.mask.shape[1]),
        "transitions": len(data), "finished": finished, "history": history,
        "train_seconds": round(seconds, 2), "path": str(path), "checkpoint_sha256": sha256(path),
        "commit": job["commit"], "design_sha256": sha256(HERE / "design.json"),
        "code_sha256": {name: sha256(HERE / name) for name in CODE},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def read_logs(records):
    """The logged days, each checked against its recorded sha256."""
    days = []
    for record in records:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"{record['path']} is not the log its record names")
        with np.load(record["path"], allow_pickle=False) as z:
            days.append({key: z[key] for key in z.files})
    return days


def pool(jobs, worker, parallel, label):
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(parallel, mp_context=context) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for done, _ in enumerate(concurrent.futures.as_completed(futures), start=1):
            print(f"{label}: {done}/{len(jobs)} after {round(time.perf_counter() - started)} s",
                  file=sys.stderr, flush=True)
        return [future.result() for future in futures]


# The study.


class Study:
    def __init__(self, data_dir, run_dir, m4_07_dir, out_dir, commit):
        torch.set_num_threads(1)
        self.design, self.m4_05, self.digests = load()
        self.data_dir = data_dir
        self.run_dir = pathlib.Path(run_dir)
        self.m4_07_dir = pathlib.Path(m4_07_dir)
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        reused = self.design["reuses"]
        if sha256(self.run_dir / "frozen.json") != reused["m4_05_frozen_manifest_sha256"]:
            raise ValueError("#102's frozen manifest is not the one this study reuses")
        self.frozen = json.loads((self.run_dir / "frozen.json").read_text())
        self.regimes = base.regimes_of(self.m4_05)
        grid_spec = self.m4_05["arms"]["maskable_ppo"]["grid"]
        self.main_config = base.base_config(self.m4_05, "main", "mask", grid_spec)
        self.grid = m4_07.grid_of(self.main_config)
        self.parallel = self.design["compute"]["max_parallel_runs"]

    def config(self, execution):
        return config_of(self.main_config, execution, self.design)

    def header(self, step):
        return {"step": step, "commit": self.commit, "data": self.design["data"],
                "bounds": self.design["bounds"], "disclosure": self.design["disclosure"],
                "design_sha256": sha256(HERE / "design.json"),
                "code_sha256": {name: sha256(HERE / name) for name in CODE},
                "versions": base.versions()}

    def days(self, split):
        return base.days_of(self.m4_05, split)

    def write(self, name, report):
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return path

    def read(self, name):
        return json.loads((self.out / f"{name}.json").read_text())

    def job(self, regime, day, split, execution, arm, policy, seed=SEED, **extra):
        return {"regime": regime, "day": day, "split": split, "execution": execution,
                "data_dir": self.data_dir, "config": self.config(execution),
                "selected": self.frozen["classical_selected"], "arm": arm, "policy": policy,
                "seed": seed, "holdout_allowed": split == "holdout", **extra}

    def execution_of(self, dataset):
        return self.design["datasets"][dataset]["execution"]

    def epsilon_of(self, dataset):
        name = "as_grid_eps10" if dataset.endswith("eps10") else "as_grid_eps30"
        return m4_07_epsilon(name)

    def logs(self, dataset, split):
        """The dataset's log records of a split, checked against their manifests."""
        if dataset == "m4_07_eps30":
            frozen_path = self.m4_07_dir / "frozen.json"
            if sha256(frozen_path) != self.design["reuses"]["m4_07_frozen_manifest_sha256"]:
                raise ValueError("#104's frozen manifest is not the one this study reuses")
            frozen = json.loads(frozen_path.read_text())
            report_path = self.m4_07_dir / f"logs-{split}.json"
            if sha256(report_path) != frozen["logs"][split]:
                raise ValueError(f"#104's {split} logs are not the frozen ones")
            rows = json.loads(report_path.read_text())["logs"]
            return [r for r in rows if r["behaviour"] == "as_grid_eps30"]
        return self.read(f"logs-{dataset}-{split}")["logs"]

    def edges(self, dataset):
        """The imbalance quintile edges of the dataset's training logs, as #104 bins them."""
        observations = np.concatenate([d["observation"] for d in read_logs(self.logs(dataset, "train"))])
        present = observations[:, 11 + 5] > 0
        return np.quantile(observations[present, 5], [0.2, 0.4, 0.6, 0.8]).tolist()


def m4_07_epsilon(name):
    design = json.loads((EXPERIMENTS / "m4_07" / "design.json").read_text())
    return design["behaviours"][name]["epsilon"]


def rule_policy(name):
    return {"kind": "rule", "name": name}


def gate_summary(runs):
    """Per classical rule: its fills over the days under the standard execution, and its mean
    daily net under the standard execution and under immediate fill. It passes the gate with
    more than 0 fills and a mean daily net different from immediate fill's."""
    summary = {}
    for name in ("as_grid", "fixed_spread_4"):
        by = {e: [r for r in runs if r["arm"] == name and r["execution"] == e]
              for e in ("standard", "immediate_reference")}
        if not all(by.values()):
            raise ValueError(f"the gate needs {name} under both executions")
        fills = sum(int(r["fills"]) for r in by["standard"])
        means = {e: sum((Decimal(r["net"]) for r in v), Decimal(0)) / len(v) for e, v in by.items()}
        summary[name] = {"standard_fills": fills,
                         "standard_mean_daily_net": str(means["standard"]),
                         "immediate_mean_daily_net": str(means["immediate_reference"]),
                         "passes": fills > 0 and means["standard"] != means["immediate_reference"]}
    return summary


def choose_alpha(means, choices):
    """The alpha of the largest mean daily net among those trained, a tie to the smaller; None
    when none was."""
    candidates = [a for a in sorted(choices) if f"cql-alpha{float(a)}" in means]
    if not candidates:
        return None
    return max(candidates, key=lambda a: (means[f"cql-alpha{float(a)}"], -a))


def gate(study):
    """The queue gate and the determinism check, on validation days at the environment seed."""
    jobs = [study.job(regime, day, "validation", execution, name, rule_policy(name))
            for execution in ("standard", "immediate_reference")
            for name in ("as_grid", "fixed_spread_4")
            for regime in study.regimes for day in study.days("validation")]
    first = study.days("validation")[0]
    repeats = [study.job(regime, first, "validation", "standard", "fixed_spread_4",
                         rule_policy("fixed_spread_4")) for regime in study.regimes]
    rows = pool(jobs + repeats, run_job, study.parallel, "gate")
    runs, again = rows[:len(jobs)], rows[len(jobs):]
    summary = gate_summary(runs)
    reference = {(r["regime"], r["day"]): r["net"] for r in runs
                 if r["arm"] == "fixed_spread_4" and r["execution"] == "standard"}
    determinism = [{"regime": r["regime"], "day": r["day"], "first": reference[(r["regime"], r["day"])],
                    "again": r["net"]} for r in again]
    deterministic = all(d["first"] == d["again"] for d in determinism)
    report = study.header("gate")
    report.update({"rows": runs, "summary": summary, "determinism": determinism,
                   "deterministic": deterministic,
                   "stop": not (summary["as_grid"]["passes"] and deterministic)})
    study.write("gate", report)
    return report


def require(study, name, why):
    path = study.out / f"{name}.json"
    if not path.exists():
        raise PermissionError(f"{why}: run `{name}` first")
    report = json.loads(path.read_text())
    if report.get("stop"):
        raise PermissionError(f"`{name}` stopped the study")
    return report


def run_logs(study, dataset, split):
    if dataset not in ("standard_eps30", "standard_eps10"):
        raise ValueError("only the standard datasets are generated here")
    if split not in ("train", "validation"):
        raise PermissionError("no log is generated on the holdout")
    require(study, "gate", "the logs wait for the queue gate")
    behaviour = "as_grid_eps10" if dataset.endswith("eps10") else "as_grid_eps30"
    jobs = []
    for regime in study.regimes:
        for day in study.days(split):
            for seed in json.loads((EXPERIMENTS / "m4_07" / "design.json").read_text())[
                    "seeds"]["behaviour_seeds"]:
                path = study.out / "logs" / dataset / split / regime / f"day-{day}" / f"seed{seed}.npz"
                jobs.append(study.job(regime, day, split, "standard", behaviour, None, seed=seed,
                                      behaviour=behaviour, epsilon=m4_07_epsilon(behaviour),
                                      path=str(path)))
    rows = pool(jobs, log_job, study.parallel, f"logs {dataset} {split}")
    report = study.header(f"logs {dataset} {split}")
    report["logs"] = rows
    study.write(f"logs-{dataset}-{split}", report)
    return report


def reward_scale(days):
    rewards = np.concatenate([np.asarray(d["reward"], dtype=float) for d in days])
    scale = float(np.std(rewards))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the training rewards have no spread to scale by")
    return scale


def run_train(study, dataset, arm, alpha=None):
    if study.execution_of(dataset) == "standard":
        require(study, "gate", "training waits for the queue gate")
    records = study.logs(dataset, "train")
    scale = reward_scale(read_logs(records))
    settings = dict(study.design["arms"][arm]["settings"])
    tag = arm
    if arm == "cql":
        if alpha is None or float(alpha) not in study.design["arms"]["cql"]["alpha_choices"]:
            raise ValueError("CQL needs an alpha from the design's choices")
        settings["alpha"] = float(alpha)
        tag = f"cql-alpha{float(alpha)}"
    jobs = [{"dataset": dataset, "arm": arm, "seed": seed, "settings": settings,
             "logs": records, "reward_scale": scale, "k": len(study.grid.offsets),
             "m": len(study.grid.sizes), "commit": study.commit,
             "path": str(study.out / "checkpoints" / dataset / f"{tag}-seed{seed}.pt")}
            for seed in study.design["arms"]["training_seeds"]]
    rows = pool(jobs, train_job, study.parallel, f"train {dataset} {tag}")
    report = study.header(f"train {dataset} {tag}")
    report.update({"runs": rows, "reward_scale": scale,
                   "stop": not all(r["finished"] for r in rows)})
    study.write(f"train-{dataset}-{tag}", report)
    return report


def ppo_job(job):
    """#102's training of one maskable PPO run, from this module: the pool's spawned workers
    import a worker by its module's name, which the spec-loaded #102 runner does not have."""
    return base.train_job(job)


def run_ppo(study):
    require(study, "gate", "the PPO comparator waits for the queue gate")
    arm = study.m4_05["arms"]["maskable_ppo"]
    hyperparameters = arm["hyperparameter_sets"][0]
    jobs = [{"algorithm": "maskable_ppo", "arm": "maskable_ppo", "variant": "standard",
             "set": 0, "seed": seed, "steps": arm["budget_steps"],
             "hyperparameters": hyperparameters, "encoding": "maskable",
             "config": study.config("standard"), "regimes": study.regimes,
             "data_dir": study.data_dir, "commit": study.commit,
             "path": str(study.out / "checkpoints" / "ppo" / f"maskable_ppo-seed{seed}")}
            for seed in study.design["arms"]["training_seeds"]]
    rows = pool(jobs, ppo_job, study.parallel, "ppo")
    report = study.header("ppo")
    report["runs"] = rows
    study.write("ppo", report)
    return report


def sanity(study, dataset, tag="bc"):
    """BC's agreement with as_grid's action on the dataset's validation logs, per seed."""
    report_train = study.read(f"train-{dataset}-{tag}")
    days = read_logs(study.logs(dataset, "validation"))
    actions = offline.Actions(len(study.grid.offsets), len(study.grid.sizes))
    out = []
    for run in report_train["runs"]:
        policy = offline.Policy(network_of(run), actions)
        agree = total = 0
        for d in days:
            names = list(d["target_names"])
            target = d["target_code"][names.index("as_grid")]
            for observation, mask, code in zip(d["observation"], d["mask"], target, strict=True):
                agree += policy.code(observation, mask) == int(code)
                total += 1
        out.append({"seed": run["seed"], "agreement": agree / total, "decisions": total})
    report = study.header(f"sanity {dataset}")
    report.update({"runs": out, "threshold": 0.9,
                   "stop": not all(r["agreement"] >= 0.9 for r in out)})
    study.write(f"sanity-{dataset}", report)
    return report


def offline_policy(run):
    return {"kind": "offline", "metadata": run}


def mean_by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(Decimal(r["net"]))
    return {k: sum(v, Decimal(0)) / len(v) for k, v in out.items()}


def select(study):
    """CQL's alpha on the primary dataset: the largest mean daily net over the validation days of
    every regime and the five seeds, interactively at the environment seed; a tie to the smaller."""
    require(study, "sanity-standard_eps30", "the choice waits for the BC sanity rule")
    dataset = "standard_eps30"
    edges = study.edges(dataset)
    jobs = []
    for alpha in study.design["arms"]["cql"]["alpha_choices"]:
        trained = study.read(f"train-{dataset}-cql-alpha{float(alpha)}")
        if trained.get("stop"):
            continue
        for run in trained["runs"]:
            for regime in study.regimes:
                for day in study.days("validation"):
                    jobs.append(study.job(regime, day, "validation", "standard", f"cql-alpha{float(alpha)}",
                                          offline_policy(run), seed_of_arm=run["seed"],
                                          epsilon=study.epsilon_of(dataset), edges=edges))
    rows = pool(jobs, run_job, study.parallel, "select")
    for row in rows:
        row.pop("support_raw", None)
        row.pop("support_bins", None)
    means = mean_by(rows, "arm")
    chosen = choose_alpha(means, study.design["arms"]["cql"]["alpha_choices"])
    report = study.header("select")
    report.update({"rows": rows, "mean_daily_net": {k: str(v) for k, v in means.items()},
                   "alpha": chosen, "stop": chosen is None})
    study.write("select", report)
    return report


def consistency(study):
    """The secondary dataset's determinism check: fixed_spread_4 on the validation days under
    #102's main execution equals #102's validation values exactly, as in #104."""
    selection = json.loads((study.run_dir / "selection.json").read_text())
    expected = {}
    for row in selection["classical_validation"]:
        if row["arm"] == "fixed_spread" and json.loads(row["candidate"])["offset_ticks"] == 4:
            expected[(row["regime"], row["day"])] = Decimal(row["daily_value"])
    jobs = [study.job(regime, day, "validation", "m4_05_main", "fixed_spread_4",
                      rule_policy("fixed_spread_4")) for regime, day in sorted(expected)]
    rows = pool(jobs, run_job, study.parallel, "consistency")
    mismatches = [{"regime": r["regime"], "day": r["day"], "here": r["net"],
                   "m4_05": str(expected[(r["regime"], r["day"])])}
                  for r in rows if Decimal(r["net"]) != expected[(r["regime"], r["day"])]]
    report = study.header("consistency")
    report.update({"compared": len(rows), "mismatches": mismatches, "stop": bool(mismatches)})
    study.write("consistency", report)
    return report


def trained_runs(study, dataset, tag):
    report = study.read(f"train-{dataset}-{tag}")
    if report.get("stop"):
        raise PermissionError(f"the {dataset} {tag} training stopped")
    return report["runs"]


def freeze(study):
    for name in ("gate", "sanity-standard_eps30", "sanity-standard_eps10", "sanity-m4_07_eps30",
                 "select", "consistency"):
        require(study, name, "the freeze waits for every check")
    alpha = study.read("select")["alpha"]
    tag = f"cql-alpha{float(alpha)}"
    tags = {"standard_eps30": ["bc", tag], "standard_eps10": ["bc", tag], "m4_07_eps30": ["bc", tag]}
    checkpoints = {}
    for dataset, names in tags.items():
        for tag in names:
            for run in trained_runs(study, dataset, tag):
                checkpoints[run["path"]] = sha256(run["path"])
    for run in study.read("ppo")["runs"]:
        # #102's training saves the model at its path with stable-baselines3's ".zip" and hashes
        # that file: the manifest names the file the holdout loads and checks.
        path = pathlib.Path(run["path"]).with_suffix(".zip")
        if sha256(path) != run["checkpoint_sha256"]:
            raise ValueError(f"{path} is not the PPO checkpoint its record names")
        checkpoints[str(path)] = run["checkpoint_sha256"]
    logs = {}
    for dataset in ("standard_eps30", "standard_eps10", "m4_07_eps30"):
        for split in ("train", "validation"):
            for record in study.logs(dataset, split):
                logs[record["path"]] = record["sha256"]
    reports = {p.name: sha256(p) for p in sorted(study.out.glob("*.json")) if p.name != "frozen.json"}
    report = study.header("freeze")
    report.update({"alpha": alpha, "tags": tags, "checkpoints": checkpoints, "logs": logs,
                   "reports": reports})
    study.write("frozen", report)
    return report


def holdout_jobs(study, frozen):
    """Every arm of the design on the holdout days, as the frozen manifest names them."""
    jobs = []
    days = study.days("holdout")
    for comparison, execution, datasets in (("primary", "standard", ["standard_eps30", "standard_eps10"]),
                                            ("secondary_m4_07", "m4_05_main", ["m4_07_eps30"])):
        for name in ("no_trade", "fixed_spread_4", "as_grid"):
            for regime in study.regimes:
                for day in days:
                    jobs.append(study.job(regime, day, "holdout", execution, name, rule_policy(name),
                                          comparison=comparison))
        if execution == "standard":
            ppo = [(run["seed"], {"kind": "ppo", "metadata": run}) for run in study.read("ppo")["runs"]]
        else:
            ppo = [(m["seed"], {"kind": "frozen_ppo", "metadata": m})
                   for m in m4_07.ppo_runs(study.frozen, study.run_dir)]
        for seed, policy in ppo:
            for regime in study.regimes:
                for day in days:
                    jobs.append(study.job(regime, day, "holdout", execution, "maskable_ppo", policy,
                                          seed_of_arm=seed, comparison=comparison))
        for dataset in datasets:
            edges = study.edges(dataset)
            for tag in frozen["tags"][dataset]:
                arm = "bc" if tag == "bc" else "cql"
                for run in trained_runs(study, dataset, tag):
                    for regime in study.regimes:
                        for day in days:
                            jobs.append(study.job(
                                regime, day, "holdout", execution, f"{arm}:{dataset}",
                                offline_policy(run), seed_of_arm=run["seed"],
                                comparison=comparison, dataset=dataset,
                                epsilon=study.epsilon_of(dataset), edges=edges))
    return jobs


HOLDOUT_MARKER = "holdout.started"


def holdout(study, frozen_sha256):
    frozen_path = study.out / "frozen.json"
    if not frozen_path.exists() or sha256(frozen_path) != frozen_sha256:
        raise PermissionError("the holdout runs only against the frozen manifest named")
    if (study.out / "holdout.json").exists():
        raise PermissionError("the holdout was evaluated already")
    if (study.out / HOLDOUT_MARKER).exists():
        raise PermissionError("the holdout was started already; a failed or concurrent attempt "
                              "is not run again, and recovery is by hand")
    frozen = json.loads(frozen_path.read_text())
    if frozen["design_sha256"] != sha256(HERE / "design.json"):
        raise ValueError("design.json changed after the freeze")
    for name, digest in frozen["code_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"{name} changed after the freeze")
    for path, digest in {**frozen["checkpoints"], **frozen["logs"]}.items():
        if sha256(path) != digest:
            raise ValueError(f"{path} changed after the freeze")
    for name, digest in frozen["reports"].items():
        if sha256(study.out / name) != digest:
            raise ValueError(f"{name} changed after the freeze")
    jobs = holdout_jobs(study, frozen)
    # Before any job reads a holdout day: created only if absent, so a concurrent attempt
    # fails here, and kept if the evaluation fails, so it is never run again on its own.
    descriptor = os.open(study.out / HOLDOUT_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as marker:
        marker.write(json.dumps({"commit": study.commit, "frozen_sha256": frozen_sha256,
                                 "jobs": len(jobs), "at": time.time()}))
    rows = pool(jobs, run_job, study.parallel, "holdout")
    for job, row in zip(jobs, rows, strict=True):
        row["comparison"] = job["comparison"]
        row["dataset"] = job.get("dataset")
    support = support_report(study, rows)
    for row in rows:
        row.pop("support_raw", None)
        row.pop("support_bins", None)
    report = study.header("holdout")
    report.update({"frozen_sha256": frozen_sha256, "rows": rows, "support": support})
    study.write("holdout", report)
    return report


def support_report(study, rows):
    """Per offline arm: the shares of its holdout decisions at behaviour propensity 0 and under
    0.0001, and with a (state bin, action) pair its training logs hold 0 and under 10 times."""
    out = {}
    keys_by_dataset = {}
    for row in rows:
        if "support_raw" not in row:
            continue
        dataset = row["dataset"]
        if dataset not in keys_by_dataset:
            edges = study.edges(dataset)
            keys = []
            for d in read_logs(study.logs(dataset, "train")):
                positions = d["observation"][:, POSITION] * float(
                    study.main_config["scales"]["position"])
                bins = state_bins(d["observation"], positions, edges)
                keys.extend(zip(bins.tolist(), d["action"].astype(int).tolist(), strict=True))
            keys_by_dataset[dataset] = keys
        counts = offline.support_counts(keys_by_dataset[dataset],
                                        list(zip(row["support_bins"], row["support_raw"]["code"],
                                                 strict=True)))
        propensity = np.asarray(row["support_raw"]["propensity"])
        entry = out.setdefault(f"{row['arm']}-seed{row['seed_of_arm']}",
                               {"decisions": 0, "propensity_zero": 0, "propensity_under_1e-4": 0,
                                "pair_count_zero": 0, "pair_count_under_10": 0})
        entry["decisions"] += len(propensity)
        entry["propensity_zero"] += int((propensity == 0).sum())
        entry["propensity_under_1e-4"] += int((propensity < 1e-4).sum())
        entry["pair_count_zero"] += int((counts == 0).sum())
        entry["pair_count_under_10"] += int((counts < 10).sum())
    for entry in out.values():
        n = max(entry["decisions"], 1)
        for key in ("propensity_zero", "propensity_under_1e-4", "pair_count_zero",
                    "pair_count_under_10"):
            entry[f"{key}_share"] = entry[key] / n
    return out


def pilot(study):
    """Times on training days only: gradient steps of BC and CQL on the secondary dataset's
    training logs, one standard log day, one evaluation day. Nothing is selected."""
    records = study.logs("m4_07_eps30", "train")
    days = read_logs(records[:5])
    actions = offline.Actions(len(study.grid.offsets), len(study.grid.sizes))
    data = offline.Transitions(actions, days, reward_scale(days))
    out = {}
    for arm in ("bc", "cql"):
        settings = dict(study.design["arms"][arm]["settings"], gradient_steps=500)
        if arm == "cql":
            settings["alpha"] = 1.0
        started = time.perf_counter()
        offline.train(arm, data, 1, settings)
        per = (time.perf_counter() - started) / 500
        full = study.design["arms"][arm]["settings"]["gradient_steps"]
        out[arm] = {"seconds_per_gradient_step": round(per, 5),
                    "estimate_minutes_per_run": round(per * full / 60, 1)}
    regime = study.regimes[0]
    started = time.perf_counter()
    log_job(study.job(regime, 1, "train", "standard", "as_grid_eps30", None, seed=1,
                      behaviour="as_grid_eps30", epsilon=0.3,
                      path=str(study.out / "pilot" / "log.npz")))
    out["log_day_seconds"] = round(time.perf_counter() - started, 1)
    started = time.perf_counter()
    run_job(study.job(regime, 1, "train", "standard", "as_grid", rule_policy("as_grid")))
    out["rule_day_seconds"] = round(time.perf_counter() - started, 1)
    report = study.header("pilot")
    report.update(out)
    study.write("pilot", report)
    return report


def main():
    step, data_dir, run_dir, m4_07_dir, out_dir, commit = sys.argv[1:7]
    args = sys.argv[7:]
    study = Study(data_dir, run_dir, m4_07_dir, out_dir, commit)
    steps = {
        "pilot": lambda: pilot(study),
        "gate": lambda: gate(study),
        "logs": lambda: run_logs(study, args[0], args[1]),
        "ppo": lambda: run_ppo(study),
        "train": lambda: run_train(study, args[0], args[1], args[2] if len(args) > 2 else None),
        "sanity": lambda: sanity(study, args[0]),
        "select": lambda: select(study),
        "consistency": lambda: consistency(study),
        "freeze": lambda: freeze(study),
        "holdout": lambda: holdout(study, args[0]),
    }
    started = time.perf_counter()
    report = steps[step]()
    print(json.dumps({"step": step, "seconds": round(time.perf_counter() - started, 1),
                      "stop": report.get("stop")}))


if __name__ == "__main__":
    main()
