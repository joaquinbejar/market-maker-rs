"""A small report of both algorithms on synthetic rows: `python tests_py/smoke_report.py <commit>`.

It separates what the environment guarantees from what training happened to do. A smoke run
checks the contract, and says nothing about whether a policy earns anything.
"""

import json
import pathlib
import sys
import tempfile
import time

import torch
from conftest import config, walk

from mm_rl_bridge import BoxMarketMakingEnv, MaskableMarketMakingEnv
from mm_rl_bridge.training import evaluate, load_for_inference, train, versions

RUNS = {
    "maskable_ppo": (
        MaskableMarketMakingEnv,
        "mask",
        {"n_steps": 256, "batch_size": 64, "n_epochs": 2},
    ),
    "sac": (
        BoxMarketMakingEnv,
        "reject",
        {"learning_starts": 256, "batch_size": 64, "buffer_size": 4096},
    ),
}
TRAIN_STEPS = 4096
TRAIN_SEEDS = [1, 2, 3]
EVALUATION_SEEDS = [101, 102]


def main(commit):
    torch.set_num_threads(1)
    rows = walk(1000)
    report = {"commit": commit, "data": "synthetic", "versions": versions(), "runs": []}
    with tempfile.TemporaryDirectory() as directory:
        for algorithm, (make, policy, hyperparameters) in RUNS.items():
            settings = config(policy=policy, max_steps=2000)
            for seed in TRAIN_SEEDS:
                env = make(settings, rows)
                path = pathlib.Path(directory) / f"{algorithm}-{seed}"
                start = time.perf_counter()
                metadata = train(
                    algorithm, env, seed, TRAIN_STEPS, path, commit, hyperparameters
                )
                elapsed = time.perf_counter() - start
                twin = pathlib.Path(directory) / f"{algorithm}-{seed}-again"
                train(algorithm, env, seed, TRAIN_STEPS, twin, commit, hyperparameters)
                model, _ = load_for_inference(path, env)
                other, _ = load_for_inference(twin, env)
                same_parameters = all(
                    bool((a == b).all())
                    for a, b in zip(
                        model.policy.parameters(), other.policy.parameters(), strict=True
                    )
                )
                first = evaluate(model, env, EVALUATION_SEEDS)
                second = evaluate(model, env, EVALUATION_SEEDS)
                env.close()
                report["runs"].append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "train_seconds": round(elapsed, 2),
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                        "same_parameters_when_trained_twice": same_parameters,
                        "evaluation_repeats_exactly": first == second,
                        "net_by_evaluation_seed": {str(r.seed): str(r.net) for r in first},
                        "penalties_by_evaluation_seed": {
                            str(r.seed): str(r.penalties) for r in first
                        },
                    }
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unknown")
