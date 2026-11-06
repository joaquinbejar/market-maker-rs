"""A pinned trainer over the environments: smoke runs, checkpoints and the protocol."""

import json
import pathlib
import tomllib
from decimal import Decimal

import numpy as np
import pytest
import sb3_contrib
import stable_baselines3
import torch
from conftest import config, walk

from mm_rl_bridge import BoxMarketMakingEnv, MaskableMarketMakingEnv
from mm_rl_bridge.training import (
    TRAINER_VERSION,
    EpisodeResult,
    Protocol,
    chronological_split,
    evaluate,
    load_for_inference,
    train,
    versions,
)

PPO_SMALL = {"n_steps": 32, "batch_size": 16, "n_epochs": 1}
SAC_SMALL = {"learning_starts": 16, "batch_size": 16, "buffer_size": 512}


def maskable(rows=None, **kwargs):
    return MaskableMarketMakingEnv(config(max_steps=1000), rows or walk(), **kwargs)


def boxed(rows=None, **kwargs):
    return BoxMarketMakingEnv(config(policy="reject", max_steps=1000), rows or walk(), **kwargs)


def test_versions_name_the_library_and_the_bridge_as_their_manifests_do():
    bridge = pathlib.Path(__file__).parents[1]
    manifests = [bridge.parents[1] / "Cargo.toml", bridge / "Cargo.toml"]
    library_version, bridge_version = (
        tomllib.loads(manifest.read_text())["package"]["version"] for manifest in manifests
    )
    assert library_version != bridge_version
    assert versions()["market_maker_rs"] == library_version
    assert versions()["bridge"] == bridge_version


def test_trainer_is_the_pinned_version():
    assert stable_baselines3.__version__ == TRAINER_VERSION
    assert sb3_contrib.__version__ == TRAINER_VERSION


@pytest.mark.parametrize(
    ("algorithm", "make", "hyperparameters"),
    [("maskable_ppo", maskable, PPO_SMALL), ("sac", boxed, SAC_SMALL)],
)
def test_smoke_training_gives_a_reusable_checkpoint(tmp_path, algorithm, make, hyperparameters):
    env = make()
    path = tmp_path / algorithm
    metadata = train(algorithm, env, 11, 64, path, "abc123", hyperparameters)
    assert metadata["algorithm"] == algorithm
    assert metadata["spec"] == list(env.spec_id)
    assert metadata["config"] == env.config
    assert metadata["train_rows"] == 400
    assert metadata["kernel_fingerprint"] == env.bridge_info["kernel_fingerprint"]
    assert metadata["residual_version"] is None
    assert metadata["train_dataset_digest"] == env.bridge_info["dataset_digest"]
    assert metadata["commit"] == "abc123"
    assert metadata["seed"] == 11
    assert metadata["hyperparameters"] == hyperparameters
    assert metadata["versions"] == versions()
    assert metadata["versions"]["stable_baselines3"] == TRAINER_VERSION
    assert metadata["versions"]["threads"] == 1
    assert metadata["optimizer"] == "Adam"
    assert json.loads(path.with_suffix(".json").read_text()) == metadata
    env.close()

    # The checkpoint evaluates with no contract error, and the same way twice.
    fresh = make()
    model, _ = load_for_inference(path, fresh)
    first = evaluate(model, fresh, [1, 2])
    second = evaluate(model, fresh, [1, 2])
    assert first == second
    for result in first:
        assert result.terminated and not result.truncated
        assert result.steps == 400
        assert sum((Decimal(r) for r in result.trace), Decimal(0)) == result.net - result.penalties
    fresh.close()


def test_reloaded_checkpoint_predicts_what_the_saved_model_predicts(tmp_path):
    env = maskable()
    path = tmp_path / "model"
    train("maskable_ppo", env, 3, 32, path, "abc", PPO_SMALL)
    observation, _ = env.reset(seed=1)
    masks = env.action_masks()
    model, _ = load_for_inference(path, env)
    again, _ = load_for_inference(path, env)
    a, _ = model.predict(observation, deterministic=True, action_masks=masks)
    b, _ = again.predict(observation, deterministic=True, action_masks=masks)
    assert np.array_equal(a, b)
    env.close()


def test_inference_does_not_change_the_model(tmp_path):
    env = maskable()
    path = tmp_path / "model"
    train("maskable_ppo", env, 3, 32, path, "abc", PPO_SMALL)
    model, _ = load_for_inference(path, env)
    before = [p.detach().clone() for p in model.policy.parameters()]
    assert not model.policy.training
    evaluate(model, env, [1])
    after = list(model.policy.parameters())
    assert all(bool((a == b).all()) for a, b in zip(before, after, strict=True))
    env.close()


def test_load_refuses_another_specification_and_a_changed_zip(tmp_path):
    env = maskable()
    path = tmp_path / "model"
    metadata = train("maskable_ppo", env, 3, 32, path, "abc", PPO_SMALL)
    env.close()

    other_config = config(max_steps=999)
    other = MaskableMarketMakingEnv(other_config, walk())
    with pytest.raises(ValueError, match="the checkpoint is for"):
        load_for_inference(path, other)
    other.close()

    fees = config(max_steps=1000)
    fees["fee_rate"] = "0.002"
    other_rules = MaskableMarketMakingEnv(fees, walk())
    assert other_rules.spec_id == tuple(metadata["spec"])
    with pytest.raises(ValueError, match="other execution rules"):
        load_for_inference(path, other_rules)
    other_rules.close()

    wrong_kind = boxed()
    with pytest.raises(TypeError, match="needs a MaskableMarketMakingEnv"):
        load_for_inference(path, wrong_kind)
    wrong_kind.close()

    zip_path = path.with_suffix(".zip")
    zip_path.write_bytes(zip_path.read_bytes() + b"x")
    same = maskable()
    with pytest.raises(ValueError, match="not the one that was saved"):
        load_for_inference(path, same)
    same.close()


def test_train_refuses_an_unknown_algorithm_or_the_wrong_encoding(tmp_path):
    env = maskable()
    with pytest.raises(ValueError, match="unknown algorithm"):
        train("dqn", env, 1, 8, tmp_path / "x", "abc")
    with pytest.raises(TypeError, match="needs a BoxMarketMakingEnv"):
        train("sac", env, 1, 8, tmp_path / "x", "abc")
    env.close()


def test_split_is_chronological_and_refuses_an_empty_window():
    rows = [(t, 1, 1, 2, 1) for t in range(10)]
    train_rows, validation, holdout = chronological_split(rows, 4, 7)
    assert [r[0] for r in train_rows] == [0, 1, 2, 3]
    assert [r[0] for r in validation] == [4, 5, 6]
    assert [r[0] for r in holdout] == [7, 8, 9]
    with pytest.raises(ValueError, match="must end before"):
        chronological_split(rows, 7, 7)
    with pytest.raises(ValueError, match="at least one row"):
        chronological_split(rows, 4, 100)
    with pytest.raises(ValueError, match="at least one row"):
        chronological_split(rows, 0, 7)
    with pytest.raises(ValueError, match="at least one row"):
        chronological_split(rows[:4] + rows[7:], 4, 7)


def test_protocol_orders_its_steps_and_uses_the_holdout_once(tmp_path):
    rows = walk(600)
    ns = lambda ms: ms * 1_000_000  # noqa: E731
    protocol = Protocol(
        config(max_steps=1000),
        "maskable_ppo",
        rows,
        ns(1000 + 300 * 25),
        ns(1000 + 450 * 25),
        tmp_path,
        "abc",
    )
    with pytest.raises(RuntimeError, match="nothing to select"):
        protocol.select([1])
    with pytest.raises(RuntimeError, match="after the selection"):
        protocol.run_holdout([1])
    protocol.train(1, 32, PPO_SMALL)
    protocol.train(2, 32, PPO_SMALL)
    table = protocol.select([1])
    # Every candidate stays in the report, and the selected one is the best on validation.
    assert [row["seed"] for row in table] == [1, 2]
    assert all(row["results"][0].steps == 150 for row in table)
    best = max(table, key=lambda row: row["mean_net"])
    with pytest.raises(RuntimeError, match="was made"):
        protocol.select([1])
    with pytest.raises(RuntimeError, match="after the selection"):
        protocol.train(3, 32, PPO_SMALL)
    metadata, results = protocol.run_holdout([1, 2])
    assert metadata["seed"] == best["seed"]
    assert [r.steps for r in results] == [150, 150]
    with pytest.raises(RuntimeError, match="already evaluated"):
        protocol.run_holdout([1])


def protocol_in(directory):
    ns = lambda ms: ms * 1_000_000  # noqa: E731
    return Protocol(
        config(max_steps=1000),
        "maskable_ppo",
        walk(600),
        ns(1000 + 300 * 25),
        ns(1000 + 450 * 25),
        directory,
        "abc",
    )


def files_in(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_protocol_keeps_candidates_with_one_seed_apart(tmp_path):
    protocol = protocol_in(tmp_path)
    other = {**PPO_SMALL, "n_epochs": 2}
    first = protocol.train(1, 32, PPO_SMALL)
    before = files_in(tmp_path)
    second = protocol.train(1, 32, other)
    after = files_in(tmp_path)
    assert len(before) == 2
    assert len(after) == 4
    assert {name: after[name] for name in before} == before
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    table = protocol.select([1])
    assert len({row["path"] for row in table}) == 2
    # Each row was evaluated on its own checkpoint, under its own metadata.
    env = MaskableMarketMakingEnv(config(max_steps=1000), walk(600))
    try:
        loaded = [load_for_inference(row["path"], env)[1] for row in table]
    finally:
        env.close()
    assert loaded == [first, second]
    assert [m["hyperparameters"] for m in loaded] == [PPO_SMALL, other]


@pytest.mark.parametrize("kept", [".zip", ".json"])
def test_protocol_refuses_a_candidates_files_before_training(tmp_path, monkeypatch, kept):
    protocol_in(tmp_path).train(1, 32, PPO_SMALL)
    for path in list(tmp_path.iterdir()):
        if path.suffix != kept:
            path.unlink()
    before = files_in(tmp_path)

    def never(*_args, **_kwargs):
        raise AssertionError("trained over an existing candidate")

    monkeypatch.setattr("mm_rl_bridge.training.train", never)
    later = protocol_in(tmp_path)
    with pytest.raises(FileExistsError, match="earlier run"):
        later.train(1, 32, {**PPO_SMALL, "n_epochs": 2})
    assert files_in(tmp_path) == before
    with pytest.raises(RuntimeError, match="nothing to select"):
        later.select([1])


SELECTIONS = {
    "best in the middle": ({1: ["1", "1"], 2: ["3", "5"], 3: ["2", "2"]}, 2),
    "by the mean, not one episode": ({1: ["9", "-9"], 2: ["1", "1"], 3: ["0", "0"]}, 2),
    "all negative": ({1: ["-5", "-5"], 2: ["-1", "-1"], 3: ["-9", "-9"]}, 2),
    "a tie goes to the earlier": ({1: ["2", "2"], 2: ["2", "2"], 3: ["1", "1"]}, 1),
}


@pytest.mark.parametrize("nets, expected", SELECTIONS.values(), ids=SELECTIONS.keys())
def test_protocol_sends_the_best_mean_net_to_the_holdout(tmp_path, monkeypatch, nets, expected):
    protocol = protocol_in(tmp_path)
    for seed in nets:
        protocol.train(seed, 32, PPO_SMALL)
    evaluated = []

    def load(path, env):
        metadata = json.loads(pathlib.Path(path).with_suffix(".json").read_text())
        return metadata, metadata

    def known(model, env, seeds):
        evaluated.append(env.bridge_info["dataset_digest"])
        return [
            EpisodeResult(seed, 1, Decimal(net), Decimal(0), True, False, (), {})
            for seed, net in zip(seeds, nets[model["seed"]])
        ]

    monkeypatch.setattr("mm_rl_bridge.training.load_for_inference", load)
    monkeypatch.setattr("mm_rl_bridge.training.evaluate", known)
    table = protocol.select([7, 8])
    assert [row["seed"] for row in table] == list(nets)
    assert [row["mean_net"] for row in table] == [
        sum(map(Decimal, values)) / 2 for values in nets.values()
    ]
    metadata, _ = protocol.run_holdout([7, 8])
    assert metadata["seed"] == expected
    # The selection read validation alone, and only the holdout's evaluation read the holdout.
    _, validation, holdout = chronological_split(
        walk(600), (1000 + 300 * 25) * 1_000_000, (1000 + 450 * 25) * 1_000_000
    )
    digests = []
    for window in (validation, holdout):
        env = MaskableMarketMakingEnv(config(max_steps=1000), window)
        digests.append(env.bridge_info["dataset_digest"])
        env.close()
    assert digests[0] != digests[1]
    assert evaluated == [digests[0]] * 3 + [digests[1]]


BAD_INPUTS = {
    "a suffix": ({"name": "v1.5"}, "ends in"),
    "two suffixes": ({"name": "model.tar.gz"}, "ends in"),
    "hyperparameters": (
        {"hyperparameters": {**PPO_SMALL, "policy_kwargs": {"activation_fn": torch.nn.Tanh}}},
        "must be JSON",
    ),
    "a commit": ({"commit": object()}, "must be JSON"),
    "a seed": ({"seed": np.int64(1)}, "must be JSON"),
    "a budget": ({"total_timesteps": np.int64(8)}, "must be JSON"),
}


@pytest.mark.parametrize("changed, message", BAD_INPUTS.values(), ids=BAD_INPUTS.keys())
def test_train_refuses_what_it_could_not_save_before_training(
    tmp_path, monkeypatch, changed, message
):
    def never(*_args, **_kwargs):
        raise AssertionError("trained on inputs it could not save")

    monkeypatch.setattr(sb3_contrib.MaskablePPO, "learn", never)
    inputs = {
        "name": "model",
        "seed": 1,
        "total_timesteps": 8,
        "commit": "abc",
        "hyperparameters": PPO_SMALL,
        **changed,
    }
    env = maskable()
    with pytest.raises(ValueError, match=message):
        train(
            "maskable_ppo",
            env,
            inputs["seed"],
            inputs["total_timesteps"],
            tmp_path / inputs["name"],
            inputs["commit"],
            inputs["hyperparameters"],
        )
    env.close()
    assert list(tmp_path.iterdir()) == []


def test_train_takes_a_path_with_or_without_the_zip_suffix(tmp_path):
    env = maskable()
    for name in ("plain", "named.zip"):
        metadata = train("maskable_ppo", env, 3, 32, tmp_path / name, "abc", PPO_SMALL)
        _, loaded = load_for_inference(tmp_path / name, env)
        assert loaded == metadata
    env.close()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "named.json",
        "named.zip",
        "plain.json",
        "plain.zip",
    ]


def test_report_separates_environment_determinism_from_training_variability(tmp_path):
    # Environment: the same actions give the same exact trace, which is asserted.
    # Training: two runs with one seed are compared and the result is only reported, as
    # bit exact optimizer reproducibility is not claimed.
    env = maskable()
    train("maskable_ppo", env, 5, 64, tmp_path / "a", "abc", PPO_SMALL)
    train("maskable_ppo", env, 5, 64, tmp_path / "b", "abc", PPO_SMALL)
    a, _ = load_for_inference(tmp_path / "a", env)
    b, _ = load_for_inference(tmp_path / "b", env)
    assert evaluate(a, env, [1]) == evaluate(a, env, [1])
    same = all(
        bool((x == y).all())
        for x, y in zip(a.policy.parameters(), b.policy.parameters(), strict=True)
    )
    print(f"training with one seed twice gave identical parameters: {same}")
    env.close()
