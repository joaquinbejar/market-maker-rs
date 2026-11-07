"""Training, selection and frozen evaluation with a pinned external trainer.

A checkpoint is the trainer's zip and a metadata file with what produced it. Loading one
checks that the environment runs the same specification under the same execution rules, and
that the zip is the one that was saved. The trainer's zip runs code when loaded, so load only
checkpoints of your own: the hash catches a change, and is no security boundary. There is
no observation normalization outside the specification, so a checkpoint has none to carry.
`Protocol` makes the steps consume each other, so that a selection never reads
the holdout and the holdout is evaluated once, on checkpoints frozen before it.
"""

import dataclasses
import hashlib
import json
import pathlib
import platform
from decimal import Decimal

import gymnasium
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from ._native import BRIDGE_VERSION, LIBRARY_VERSION, RESIDUAL_VERSION
from .envs import BoxMarketMakingEnv, MaskableMarketMakingEnv, ResidualMarketMakingEnv

TRAINER_VERSION = "2.9.2"
METADATA_VERSION = 1

_ALGORITHMS = {
    "maskable_ppo": (MaskablePPO, MaskableMarketMakingEnv),
    "sac": (SAC, BoxMarketMakingEnv),
    "sac_residual": (SAC, ResidualMarketMakingEnv),
}


def versions():
    """The versions of everything a run depends on, the library and this bridge included."""
    return {
        "market_maker_rs": LIBRARY_VERSION,
        "bridge": BRIDGE_VERSION,
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }


def _check_algorithm(algorithm, env):
    try:
        _, expected = _ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError(f"unknown algorithm {algorithm!r}") from None
    if not isinstance(env, expected):
        raise TypeError(f"{algorithm} needs a {expected.__name__}, not a {type(env).__name__}")


def _hash(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _check_inputs(path, seed, total_timesteps, commit, hyperparameters):
    """Refuses, before a run, what could only fail when the run is saved."""
    if path.suffix not in ("", ".zip"):
        raise ValueError(f"a checkpoint's path ends in .zip or nothing, not {path.suffix!r}")
    inputs = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "commit": commit,
        "hyperparameters": hyperparameters,
    }
    try:
        json.dumps(inputs)
    except TypeError as error:
        raise ValueError(f"a checkpoint's metadata must be JSON: {error}") from error


def train(algorithm, env, seed, total_timesteps, path, commit, hyperparameters=None):
    """Trains on one thread of the CPU and saves a checkpoint with its metadata.

    `commit` is declared by the caller, and not checked. Replaces what `path` held. Raises
    `ValueError` before training for a path with another suffix, or for inputs the metadata
    cannot hold as JSON. Returns the metadata.
    """
    _check_algorithm(algorithm, env)
    if stable_baselines3.__version__ != TRAINER_VERSION:
        raise RuntimeError(f"the trainer is pinned to {TRAINER_VERSION}")
    path = pathlib.Path(path)
    hyperparameters = dict(hyperparameters or {})
    _check_inputs(path, seed, total_timesteps, commit, hyperparameters)
    torch.set_num_threads(1)
    set_random_seed(seed)
    model_class, _ = _ALGORITHMS[algorithm]
    env.reset(seed=seed)
    wrapped = DummyVecEnv([lambda: Monitor(env)])
    model = model_class("MlpPolicy", wrapped, seed=seed, device="cpu", **hyperparameters)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(path)
    zip_path = path.with_suffix(".zip")
    metadata = {
        "metadata_version": METADATA_VERSION,
        "algorithm": algorithm,
        "spec": list(env.spec_id),
        "config": env.config,
        "kernel_fingerprint": env.bridge_info["kernel_fingerprint"],
        "residual_version": _residual_version(env.config),
        "train_dataset_digest": env.bridge_info["dataset_digest"],
        "train_rows": env.bridge_info["rows"],
        "seed": seed,
        "total_timesteps": total_timesteps,
        "hyperparameters": hyperparameters,
        "commit": commit,
        "versions": versions(),
        "optimizer": type(_optimizer(model)).__name__,
        "checkpoint_sha256": _hash(zip_path),
    }
    _metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _residual_version(config):
    return None if config.get("residual") is None else RESIDUAL_VERSION


def _optimizer(model):
    policy = model.policy
    if hasattr(policy, "optimizer"):
        return policy.optimizer
    return policy.actor.optimizer


def _metadata_path(path):
    return pathlib.Path(path).with_suffix(".json")


def load_for_inference(path, env):
    """Loads a checkpoint to act only, after checking it belongs to this environment.

    Raises `ValueError` for another specification, other execution rules (fees, latency,
    fills or limits), another residual (its baseline, its bounds or the version of its
    composition, none of which the specification covers) or a zip that changed. Residual
    configurations are compared as written, so one number written two ways is refused too.
    """
    path = pathlib.Path(path)
    metadata = json.loads(_metadata_path(path).read_text())
    _check_algorithm(metadata["algorithm"], env)
    # A checkpoint from before hedge actions has no such key, and had none.
    trained = bool(metadata["config"].get("hedge_actions", False))
    here = bool(env.config.get("hedge_actions", False))
    if trained != here:
        raise ValueError(
            f"the checkpoint was trained with hedge_actions={trained}, "
            f"this environment has hedge_actions={here}"
        )
    if tuple(metadata["spec"]) != env.spec_id:
        raise ValueError(f"the checkpoint is for {metadata['spec']}, this is {env.spec_id}")
    if metadata["kernel_fingerprint"] != env.bridge_info["kernel_fingerprint"]:
        raise ValueError("the checkpoint was trained under other execution rules")
    if metadata.get("residual_version") != _residual_version(env.config):
        raise ValueError("the checkpoint composes its residual under another version")
    if metadata["config"].get("residual") != env.config.get("residual"):
        raise ValueError("the checkpoint was trained over another residual configuration")
    zip_path = path.with_suffix(".zip")
    if _hash(zip_path) != metadata["checkpoint_sha256"]:
        raise ValueError("the checkpoint's zip is not the one that was saved")
    model_class, _ = _ALGORITHMS[metadata["algorithm"]]
    model = model_class.load(zip_path, device="cpu")
    model.policy.set_training_mode(False)
    return model, metadata


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    """One evaluated episode. `net` is exact, and `trace` holds every exact reward."""

    seed: int
    steps: int
    net: Decimal
    penalties: Decimal
    terminated: bool
    truncated: bool
    trace: tuple
    attribution: dict


class ZeroResidual:
    """The classical baseline as a policy: no correction at any decision."""

    @staticmethod
    def predict(observation, deterministic=True):
        return np.zeros(4, dtype=np.float32), None


def evaluate(model, env, seeds):
    """Runs the model with no learning, and returns one exact result per seed."""
    results = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        trace = []
        while True:
            if isinstance(env, MaskableMarketMakingEnv):
                action, _ = model.predict(
                    observation, deterministic=True, action_masks=env.action_masks()
                )
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            trace.append(info["reward_exact"])
            if terminated or truncated:
                break
        results.append(
            EpisodeResult(
                seed=seed,
                steps=len(trace),
                net=Decimal(info["net"]),
                penalties=Decimal(info["penalties"]),
                terminated=terminated,
                truncated=truncated,
                trace=tuple(trace),
                attribution=dict(env.attribution()),
            )
        )
    return results


def chronological_split(rows, train_end_ns, validation_end_ns):
    """Splits rows by time, into train, validation and holdout."""
    if not train_end_ns < validation_end_ns:
        raise ValueError("the train window must end before the validation window")
    train_rows = [row for row in rows if row[0] < train_end_ns]
    validation = [row for row in rows if train_end_ns <= row[0] < validation_end_ns]
    holdout = [row for row in rows if validation_end_ns <= row[0]]
    if not train_rows or not validation or not holdout:
        raise ValueError("every window needs at least one row")
    return train_rows, validation, holdout


class Protocol:
    """Training, then selection on validation, then one evaluation on the holdout."""

    def __init__(self, config, algorithm, rows, train_end_ns, validation_end_ns, directory, commit):
        _, self._env_class = _ALGORITHMS[algorithm]
        self._config = dict(config)
        self._algorithm = algorithm
        self._train, self._validation, self._holdout = chronological_split(
            rows, train_end_ns, validation_end_ns
        )
        self._directory = pathlib.Path(directory)
        self._commit = commit
        self._candidates = []
        self._selected = None
        self._holdout_used = False

    def train(self, seed, total_timesteps, hyperparameters=None):
        """Trains one candidate on the training window. Every candidate stays listed.

        A candidate's files are named by its place in the list, so that candidates with one
        seed and other hyperparameters stay apart. Raises `FileExistsError` before training
        if those files exist, so that a directory used before keeps its checkpoints.
        """
        if self._selected is not None:
            raise RuntimeError("training after the selection")
        path = self._directory / f"{self._algorithm}-{len(self._candidates)}-{seed}"
        for existing in (path.with_suffix(".zip"), _metadata_path(path)):
            if existing.exists():
                raise FileExistsError(f"{existing} holds a candidate of an earlier run")
        env = self._env_class(self._config, self._train, seed=seed)
        try:
            metadata = train(
                self._algorithm,
                env,
                seed,
                total_timesteps,
                path,
                self._commit,
                hyperparameters,
            )
        finally:
            env.close()
        self._candidates.append((path, metadata))
        return metadata

    def select(self, evaluation_seeds):
        """Evaluates every candidate on validation and keeps the best mean net result.

        The earlier candidate wins a tie. Returns one row per candidate, the unselected ones
        included.
        """
        if not self._candidates:
            raise RuntimeError("there is nothing to select from")
        if self._selected is not None:
            raise RuntimeError("the selection was made")
        rows = []
        for path, metadata in self._candidates:
            env = self._env_class(self._config, self._validation)
            try:
                model, _ = load_for_inference(path, env)
                results = evaluate(model, env, evaluation_seeds)
            finally:
                env.close()
            mean = sum((r.net for r in results), Decimal(0)) / len(results)
            rows.append(
                {"path": path, "seed": metadata["seed"], "mean_net": mean, "results": results}
            )
        self._selected = max(rows, key=lambda row: row["mean_net"])["path"]
        return rows

    def run_holdout(self, evaluation_seeds):
        """Evaluates the selected checkpoint on the holdout, once."""
        if self._selected is None:
            raise RuntimeError("the holdout comes after the selection")
        if self._holdout_used:
            raise RuntimeError("the holdout was already evaluated")
        self._holdout_used = True
        env = self._env_class(self._config, self._holdout)
        try:
            model, metadata = load_for_inference(self._selected, env)
            return metadata, evaluate(model, env, evaluation_seeds)
        finally:
            env.close()
