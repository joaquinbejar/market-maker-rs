"""A small SAC budget against its zero residual: `python tests_py/residual_report.py <commit>`.

Synthetic rows and a small budget. Every seed is reported, in the evaluation's components.
It tests the path from training to a frozen evaluation, and is no evidence for or against a
residual policy: that comparison is a separate, frozen one.
"""

import json
import sys
import tempfile

import torch
from conftest import walk
from test_residual import residual_config

from mm_rl_bridge import ResidualMarketMakingEnv
from mm_rl_bridge.training import Protocol, ZeroResidual, evaluate, versions

TRAIN_SEEDS = [1, 2, 3, 4, 5]
EVALUATION_SEEDS = [101, 102, 103]
BUDGET = 4096
HYPERPARAMETERS = {"learning_starts": 256, "batch_size": 64, "buffer_size": 4096}
SPACING_MS = 25


def row(result):
    return {
        "seed": result.seed,
        "steps": result.steps,
        "net": str(result.net),
        **result.attribution,
    }


def baseline(settings, rows, seeds):
    env = ResidualMarketMakingEnv(settings, rows)
    try:
        return [row(r) for r in evaluate(ZeroResidual, env, seeds)]
    finally:
        env.close()


def main(commit):
    torch.set_num_threads(1)
    rows = walk(3000)
    settings = residual_config(max_steps=5000)
    train_end = (1000 + 1500 * SPACING_MS) * 1_000_000
    validation_end = (1000 + 2250 * SPACING_MS) * 1_000_000
    report = {"commit": commit, "data": "synthetic", "versions": versions(), "budget": BUDGET}
    with tempfile.TemporaryDirectory() as directory:
        protocol = Protocol(
            settings, "sac_residual", rows, train_end, validation_end, directory, commit
        )
        for seed in TRAIN_SEEDS:
            protocol.train(seed, BUDGET, HYPERPARAMETERS)
        table = protocol.select(EVALUATION_SEEDS)
        report["validation_zero_residual"] = baseline(settings, rows[1500:2250], EVALUATION_SEEDS)
        report["validation_candidates"] = [
            {
                "train_seed": entry["seed"],
                "mean_net": str(entry["mean_net"]),
                "results": [row(r) for r in entry["results"]],
            }
            for entry in table
        ]
        metadata, holdout = protocol.run_holdout(EVALUATION_SEEDS)
        report["selected_train_seed"] = metadata["seed"]
        report["holdout_selected"] = [row(r) for r in holdout]
        report["holdout_zero_residual"] = baseline(settings, rows[2250:], EVALUATION_SEEDS)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unknown")
