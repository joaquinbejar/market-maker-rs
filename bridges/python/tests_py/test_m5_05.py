"""#281's runner: the split of the grid, the training adversary, the phases, the checks and the
holdout's guards. Rows are synthetic: #102's simulator and hand-made fakes."""

import importlib.util
import json
import pathlib
import time
from decimal import Decimal

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m5_05"
SPEC = importlib.util.spec_from_file_location("m5_05_run", STUDY / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

DESIGN, M4_05, M5_03 = run.load()


class FakeCtx:
    """The parts of the study's context that need no frozen run."""

    def __init__(self):
        self.design, self.m4_05, self.m5_03 = DESIGN, M4_05, M5_03
        self.regimes = ["calm", "stressed", "thin"]
        self.points = run.points(M5_03)
        self.run_dir = "unused"
        self.parallel = 1


def test_the_grid_splits_into_12_training_and_12_held_out_points():
    training = run.training_points(M5_03)
    held_out = run.held_out_points(M5_03)
    assert len(training) == 12 and len(held_out) == 12
    assert training[0][0] == run.unperturbed_name()
    assert all(p["limit_rate_multiplier"] == 1.0 for _, p in training)
    assert all(p["limit_rate_multiplier"] == 0.7 for _, p in held_out)
    names = [n for n, _ in held_out]
    # #108's fixed stress and its choice for #102's residual SAC are held out.
    assert run.m503.point_name(M5_03["fixed_stress"]) in names
    assert "m2.0-c1.5-t1.2-l0.7" in names
    # In #108's order.
    order = [n for n, _ in run.points(M5_03)]
    assert [order.index(n) for n in names] == sorted(order.index(n) for n in names)


def test_the_constants_are_the_designs():
    assert "evaluation seed 101" in DESIGN["training"]["score"]["what"]
    assert M4_05["evaluation"]["evaluation_seeds_from_step_2"] == [run.EVALUATION_SEED]
    assert DESIGN["training"]["score"]["scoring_days"].startswith("training days 1 and 8")
    assert run.SCORING_DAYS == (1, 8)
    assert DESIGN["training"]["phases"] == {**DESIGN["training"]["phases"], "count": 5,
                                            "steps_each": 100000}


def test_the_splits_days_and_offsets():
    assert run.split_days(DESIGN, M4_05, M5_03, "train") == (0, list(range(1, 15)))
    assert run.split_days(DESIGN, M4_05, M5_03, "validation") == (0, list(range(15, 20)))
    assert run.split_days(DESIGN, M4_05, M5_03, "selection") == (100000, list(range(1, 11)))
    assert run.split_days(DESIGN, M4_05, M5_03, "holdout") == (300000, list(range(1, 11)))
    # Never one of #102's, #108's or #282's seeds.
    assert run.day_seed(3, 10, 300000) == 3 * 1_000_003 + 10 + 300000


def test_the_mixtures():
    assert run.uniform_mixture(12) == pytest.approx([1 / 12] * 12)
    mixture = run.adversary_mixture(5, 12)
    assert sum(mixture) == pytest.approx(1.0)
    assert mixture[5] == pytest.approx(0.50)
    assert mixture[0] == pytest.approx(0.25 + 0.25 / 11)
    assert all(m == pytest.approx(0.25 / 11) for i, m in enumerate(mixture) if i not in (0, 5))
    # The unperturbed point as the worst holds 0.75.
    mixture = run.adversary_mixture(0, 12)
    assert mixture[0] == pytest.approx(0.75)
    assert sum(mixture) == pytest.approx(1.0)


def test_the_score_is_the_worst_daily_loss_and_the_first_point_wins_a_tie():
    assert run.worst_loss(["3", "-7.5", "-2", "1", "0", "4"]) == pytest.approx(7.5)
    names = ["a", "b", "c"]
    scores = {"a": {"worst_loss": 1.0}, "b": {"worst_loss": 4.0}, "c": {"worst_loss": 4.0}}
    assert run.worst_point(scores, names) == 1


class FakeEnv(gym.Env):
    """A one-dimensional residual stand-in: the reward is the action against a target that
    the day and the point set, over 20 steps."""

    def __init__(self, name="p", regime="calm", day=1):
        super().__init__()
        self.name, self.regime, self.day = name, regime, day
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.closed = False
        self.reset_seed = None
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seed, self.t = seed, 0
        return np.array([self.day / 14, len(self.name) / 20], np.float32), {}

    def step(self, action):
        self.t += 1
        target = (self.day % 3 - 1) / 2
        reward = -float((float(action[0]) - target) ** 2)
        observation = np.array([self.day / 14, self.t / 20], np.float32)
        return observation, reward, self.t >= 20, False, {}

    def close(self):
        self.closed = True


def fake_make(made):
    def make(ctx, name, regime, day):
        env = FakeEnv(name, regime, day)
        made.append(env)
        return env
    return make


def test_the_baseline_draws_102s_days_and_the_adversary_the_same_days():
    made_b, made_a = [], []
    b = run.Days(FakeCtx(), "baseline", 3, fake_make(made_b))
    a = run.Days(FakeCtx(), "adversarial", 3, fake_make(made_a))
    b.set_phase(1, None)
    a.set_phase(1, run.uniform_mixture(12))
    for _ in range(30):
        b.reset()
        a.reset()
    # #102's DayCycle: a day index from the run's seed, then the reset seed.
    rng = np.random.default_rng(3)
    pairs = [(r, d) for r in ("calm", "stressed", "thin") for d in range(1, 15)]
    expected = []
    for _ in range(30):
        regime, day = pairs[int(rng.integers(len(pairs)))]
        expected.append((regime, day, int(rng.integers(2**63))))
    seen_b = [(e.regime, e.day, e.reset_seed) for e in made_b[1:]]
    seen_a = [(e.regime, e.day, e.reset_seed) for e in made_a[1:]]
    assert seen_b == expected and seen_a == expected
    # The baseline never leaves the unperturbed point; the adversary draws others.
    assert {e.name for e in made_b[1:]} == {run.unperturbed_name()}
    assert len({e.name for e in made_a[1:]}) > 1
    # The previous environment is closed at every reset.
    assert all(e.closed for e in made_a[:-1])


def test_the_adversarys_points_come_from_281_seed_phase_and_follow_the_mixture():
    made = []
    env = run.Days(FakeCtx(), "adversarial", 2, fake_make(made))
    names = [n for n, _ in run.training_points(M5_03)]
    mixture = run.adversary_mixture(7, 12)
    env.set_phase(3, mixture)
    for _ in range(400):
        env.reset()
    drawn = [e["point"] for e in env.episodes]
    rng = np.random.default_rng([281, 2, 3])
    assert drawn == [names[int(rng.choice(12, p=mixture))] for _ in range(400)]
    assert drawn.count(names[7]) / 400 == pytest.approx(0.5, abs=0.08)
    assert drawn.count(names[0]) / 400 == pytest.approx(0.25 + 0.25 / 11, abs=0.08)
    assert {e["phase"] for e in env.episodes} == {3}


def sac(env, seed=1):
    torch.manual_seed(seed)
    return SAC("MlpPolicy", DummyVecEnv([lambda: env]), seed=seed, device="cpu",
               learning_starts=30, batch_size=16, buffer_size=1000, learning_rate=3e-4,
               tau=0.02, policy_kwargs={"net_arch": [8]})


def params(model):
    return [p.detach().clone() for p in model.policy.parameters()]


def test_five_phases_carry_the_model_as_one_run_and_scoring_changes_nothing():
    """Five learn calls of 40 steps equal one of 200, with or without a policy scored between
    them: buffer, optimisers, entropy coefficient and step count carry across."""
    def phases(scorer):
        cycle = run.Days(FakeCtx(), "baseline", 1, lambda ctx, n, r, d: FakeEnv(n, r, d))
        model = sac(cycle)
        log, _, _ = run.run_phases(model, cycle, "baseline",
                                   [n for n, _ in run.training_points(M5_03)], 5, 40, scorer)
        return model, log

    def predicting(model):
        observation = np.zeros(2, np.float32)
        for _ in range(25):
            model.predict(observation, deterministic=True)
        return {n: {"worst_loss": 0.0} for n, _ in run.training_points(M5_03)}

    split, log = phases(predicting)
    assert [e["num_timesteps"] for e in log] == [40, 80, 120, 160, 200]
    assert log[0]["mixture"] is None and "scores" not in log[0]
    assert all("scores" in e for e in log[1:])
    # The segmented run on the same days equals one learn call of the same total.
    plain = sac(run.Days(FakeCtx(), "baseline", 1, lambda ctx, n, r, d: FakeEnv(n, r, d)))
    plain.env.envs[0].set_phase(1, None)
    plain.learn(total_timesteps=200, progress_bar=False)
    for a, b in zip(params(split), params(plain), strict=True):
        assert torch.equal(a, b)
    assert split.replay_buffer.pos == plain.replay_buffer.pos
    assert split.num_timesteps == plain.num_timesteps == 200


def test_the_adversarial_mixture_moves_to_the_worst_point_each_phase():
    names = [n for n, _ in run.training_points(M5_03)]
    worst = iter([4, 4, 9, 0])

    def scorer(_):
        w = next(worst)
        return {n: {"worst_loss": float(i == w)} for i, n in enumerate(names)}

    cycle = run.Days(FakeCtx(), "adversarial", 1, lambda ctx, n, r, d: FakeEnv(n, r, d))
    log, _, _ = run.run_phases(sac(cycle), cycle, "adversarial", names, 5, 30, scorer)
    assert log[0]["mixture"] == pytest.approx(run.uniform_mixture(12))
    assert [e["worst"] for e in log[1:]] == [names[4], names[4], names[9], names[0]]
    assert log[3]["mixture"] == pytest.approx(run.adversary_mixture(9, 12))
    assert log[4]["mixture"][0] == pytest.approx(0.75)


def test_markets_are_checked_against_the_committed_digests(monkeypatch):
    ctx = FakeCtx()
    # #102's validation day and #108's selection day, simulated here, match their digests.
    run.market_for(ctx, "validation", run.unperturbed_name(), "calm", 15)
    run.market_for(ctx, "selection", "m2.0-c1.5-t1.2-l0.7", "thin", 3)
    with pytest.raises(PermissionError):
        run.market_for(ctx, "holdout", run.unperturbed_name(), "calm", 1)
    with pytest.raises(ValueError, match="not in"):
        run.market_for(ctx, "validation", run.unperturbed_name(), "calm", 14)
    with pytest.raises(ValueError, match="unperturbed only"):
        run.market_for(ctx, "validation", "m2.0-c1.5-t1.2-l0.7", "calm", 15)
    monkeypatch.setattr(run, "expected_digest", lambda *a: "0" * 32)
    with pytest.raises(ValueError, match="other rows"):
        run.market_for(ctx, "selection", "m2.0-c1.5-t1.2-l0.7", "thin", 3)


def baseline_rows(nets_by_seed):
    rows = []
    for seed, nets in nets_by_seed.items():
        for day, net in enumerate(nets, start=1):
            rows.append({"arm": "baseline", "seed": seed, "regime": "calm", "day": day,
                         "net": net})
    return rows


def test_the_reproduction_check_is_numeric():
    check = DESIGN["validation"]["reproduction_check"]
    tolerance = 2 * 4.0847 * (2 / 5) ** 0.5
    mean = check["reference"]["mean"]
    inside = {s: [str(mean + tolerance - 0.01)] * 3 for s in range(1, 6)}
    result = run.reproduction(baseline_rows(inside), check)
    assert result["passed"] and result["tolerance"] == pytest.approx(5.1668, abs=1e-4)
    outside = {s: [str(mean - tolerance - 0.01)] * 3 for s in range(1, 6)}
    assert not run.reproduction(baseline_rows(outside), check)["passed"]
    # A seed's mean over its days, then the mean over seeds.
    spread = {1: ["-30", "-24"], 2: ["-27"] * 2, 3: ["-27"] * 2, 4: ["-27"] * 2, 5: ["-27"] * 2}
    assert run.reproduction(baseline_rows(spread), check)["statistic"] == pytest.approx(-27.0)
    missing = dict(inside)
    missing[3] = [None, str(mean), str(mean)]
    result = run.reproduction(baseline_rows(missing), check)
    assert not result["passed"] and "missing" in result["reason"]
    four = {s: v for s, v in inside.items() if s != 5}
    assert not run.reproduction(baseline_rows(four), check)["passed"]


def test_a_missing_seed_day_fails_the_average_and_the_choice():
    rows = [{"arm": "adversarial", "regime": "calm", "day": 1, "net": n} for n in ("1", "2", "6")]
    rows += [{"arm": "adversarial", "regime": "calm", "day": 2, "net": n} for n in ("1", None)]
    out = run.seed_averaged(rows, "adversarial")
    assert out[("calm", 1)] == 3 and out[("calm", 2)] is None
    assert run.cvar_loss([Decimal(1), None]) is None
    flat = [Decimal(0)] * 30
    by_point = {"a": {arm: list(flat) for arm in run.ARMS},
                "b": {arm: list(flat) for arm in run.ARMS}}
    by_point["b"]["baseline"] = [Decimal(-5)] * 2 + [Decimal(0)] * 28
    by_point["a"]["reference"] = [None] + [Decimal(0)] * 29
    chosen = run.choose(by_point)
    assert chosen["baseline"] == {"point": "b", "cvar_loss": pytest.approx(5.0)}
    assert chosen["adversarial"]["point"] == "a"
    assert chosen["reference"] == {"point": None, "failed": ["a"]}


def test_an_episode_error_or_a_non_finite_net_leaves_the_seed_day_missing(monkeypatch):
    class Market:
        def digest(self):
            return "d"

    monkeypatch.setattr(run, "Context", lambda _: FakeCtx())
    monkeypatch.setattr(run, "market_for", lambda *a: (Market(), 7))
    monkeypatch.setattr(run, "residual_env", lambda *a: FakeEnv())
    monkeypatch.setattr(run.m503, "load_frozen", lambda *a: "policy")
    job = {"run_dir": "x", "split": "selection", "point": "p", "regime": "calm", "day": 1,
           "metadata": {"path": "unused"}, "condition": "held_out:p", "arm": "baseline",
           "seed": 2}

    def failing(*_):
        raise RuntimeError("the worker died")

    monkeypatch.setattr(run, "evaluate", failing)
    row = run.run_policy(job)
    assert row["net"] is None and "worker died" in row["error"]
    monkeypatch.setattr(run, "evaluate", lambda *a: {"net": "NaN"})
    row = run.run_policy(job)
    assert row["net"] is None and "non-finite" in row["error"]
    monkeypatch.setattr(run, "evaluate", lambda *a: {"net": "-3.5"})
    assert run.run_policy(job)["net"] == "-3.5"

    # A changed checkpoint is not a missing day: it stops the step.
    def digest_mismatch(*_):
        raise ValueError("other rows than the digest")

    monkeypatch.setattr(run, "market_for", digest_mismatch)
    with pytest.raises(ValueError, match="other rows"):
        run.run_policy(job)


def test_the_models_freeze_only_after_b_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "FROZEN_MODELS", tmp_path / "frozen_models.json")
    (tmp_path / "validation.json").write_text(json.dumps({"reproduction": {"passed": False}}))
    with pytest.raises(ValueError, match="stops"):
        run.freeze_models(FakeCtx(), tmp_path, "c")
    assert not (tmp_path / "frozen_models.json").exists()
    with pytest.raises(ValueError, match="not frozen"):
        run.check_models(FakeCtx(), tmp_path)


def threads(monkeypatch):
    """Pools on threads, so a test's workers need no pickling; the runner's logic is the same."""
    import concurrent.futures as cf
    monkeypatch.setattr(run.concurrent.futures, "ProcessPoolExecutor",
                        lambda n, mp_context=None: cf.ThreadPoolExecutor(n))


def train_worker(fail=None, interrupt=None):
    def worker(job):
        name = pathlib.Path(job["path"]).name
        if name == fail:
            raise ValueError("loss became NaN")
        if name == interrupt:
            raise run.concurrent.futures.process.BrokenProcessPool("a worker was killed")
        pathlib.Path(job["path"]).with_suffix(".json").write_text(json.dumps({"seed": job["seed"]}))
        return {"run": name}
    return worker


def test_a_run_that_failed_inside_is_never_rerun_even_when_named(tmp_path, monkeypatch):
    threads(monkeypatch)
    ctx = FakeCtx()
    report = run.train(ctx, tmp_path, "c", limit=60, worker=train_worker(fail="adversarial-seed2"))
    assert {"run": "adversarial-seed2", "failed": "ValueError: loss became NaN"} in report["runs"]
    record = run.outcomes(tmp_path / "train.attempts")["adversarial-seed2"]
    assert record["starts"] == 1 and record["end"]["outcome"] == "failed"
    with pytest.raises(ValueError, match="never rerun"):
        run.train(ctx, tmp_path, "c", ("adversarial-seed2",), limit=60, worker=train_worker())
    # It is a failed seed, kept as such; the other runs stand.
    runs = run.trained_runs(tmp_path)
    assert runs["adversarial"][1] is None and runs["adversarial"][0] == {"seed": 1}


def test_a_killed_run_is_rerun_only_when_named_and_once(tmp_path, monkeypatch):
    threads(monkeypatch)
    ctx = FakeCtx()
    run.train(ctx, tmp_path, "c", limit=60, worker=train_worker(interrupt="baseline-seed4"))
    assert run.outcomes(tmp_path / "train.attempts")["baseline-seed4"]["end"]["outcome"] \
        == "interrupted"
    with pytest.raises(ValueError, match="did not finish"):
        run.trained_runs(tmp_path)
    with pytest.raises(ValueError, match="named, and once"):
        run.train(ctx, tmp_path, "c", limit=60, worker=train_worker())
    run.train(ctx, tmp_path, "c", ("baseline-seed4",), limit=60,
              worker=train_worker(interrupt="baseline-seed4"))
    with pytest.raises(ValueError, match="named, and once"):
        run.train(ctx, tmp_path, "c", ("baseline-seed4",), limit=60, worker=train_worker())
    # A start with no end (the step itself was killed) counts as killed too.
    (tmp_path / "train.attempts").write_text(json.dumps({"run": "baseline-seed4",
                                                         "started": 0}) + "\n")
    run.train(ctx, tmp_path, "c", ("baseline-seed4",), limit=60, worker=train_worker())
    assert run.trained_runs(tmp_path)["baseline"][3] == {"seed": 4}


def test_a_step_past_one_and_a_half_times_its_estimate_stops(tmp_path, monkeypatch):
    threads(monkeypatch)
    with pytest.raises(ValueError, match="no estimate"):
        run.limit_seconds(tmp_path, "selection")
    (tmp_path / "estimates.json").write_text(json.dumps({"selection": 10}))
    assert run.limit_seconds(tmp_path, "selection") == 15
    with pytest.raises(ValueError, match="no estimate for holdout"):
        run.limit_seconds(tmp_path, "holdout")

    def slow(job):
        if job["day"] > 1:
            time.sleep(1.0)
        return {"net": "1"}

    jobs = [{"condition": "c", "arm": "baseline", "seed": 1, "regime": "calm", "day": d}
            for d in (1, 2, 3)]
    with pytest.raises(run.Overrun, match="ran past"):
        run.pool(jobs, slow, 1, "selection", tmp_path / "selection.journal", 0.5)
    journaled = (tmp_path / "selection.journal").read_text().splitlines()
    assert len(journaled) == 1
    ctx = FakeCtx()

    def slow_train(job):
        time.sleep(1.0)
        return {}

    with pytest.raises(run.Overrun):
        run.train(ctx, tmp_path, "c", limit=0.3, worker=slow_train)
    ends = run.outcomes(tmp_path / "train.attempts")
    assert {r["end"]["outcome"] for r in ends.values() if r["end"]} == {"stopped"}
    # A stopped run is a failed seed, never rerun.
    with pytest.raises(ValueError, match="never rerun"):
        run.train(ctx, tmp_path, "c", ("adversarial-seed1",), limit=60, worker=train_worker())


def frozen_study(tmp_path, monkeypatch):
    """A frozen study on disk with no runs behind it, for the holdout's guards."""
    monkeypatch.setattr(run, "FROZEN_MODELS", tmp_path / "frozen_models.json")
    (tmp_path / "frozen_models.json").write_text("{}")
    (tmp_path / "selection.json").write_text("{}")

    class Ctx(FakeCtx):
        def header(self, step, commit):
            return {"design_sha256": "d", "code_sha256": "c"}

    frozen = {"design_sha256": "d", "code_sha256": "c", "impact": None, "chosen": {},
              "selection_sha256": run.sha256(tmp_path / "selection.json"),
              "frozen_models_sha256": run.sha256(tmp_path / "frozen_models.json"),
              "digests_sha256": {"x.json": "y"}}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    (tmp_path / "estimates.json").write_text(json.dumps({"holdout": 100}))
    monkeypatch.setattr(run, "impact_provenance", lambda ctx: None)
    monkeypatch.setattr(run, "check_models", lambda ctx, out: {})
    monkeypatch.setattr(run, "holdout_jobs", lambda *a: [])
    return Ctx(), run.sha256(tmp_path / "frozen.json")


def test_the_holdout_runs_once_and_marks_the_attempt_before_any_day(tmp_path, monkeypatch):
    ctx, frozen_sha = frozen_study(tmp_path, monkeypatch)
    monkeypatch.setattr(run.m503, "check_frozen_digests", lambda *a: None)
    seen = []

    def pool(jobs, *a):
        seen.append((tmp_path / "holdout.started").exists())
        raise RuntimeError("a day failed")

    monkeypatch.setattr(run, "pool", pool)
    with pytest.raises(RuntimeError):
        run.holdout(ctx, tmp_path, "c", frozen_sha)
    assert seen == [True]
    with pytest.raises(ValueError, match="attempted already"):
        run.holdout(ctx, tmp_path, "c", frozen_sha)
    assert len(seen) == 1


def test_the_holdout_refuses_any_change_after_the_freeze(tmp_path, monkeypatch):
    ctx, frozen_sha = frozen_study(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "pool", lambda *a: pytest.fail("a job ran"))
    with pytest.raises(ValueError, match="not the one given"):
        run.holdout(ctx, tmp_path, "c", "0" * 64)
    # The digests are checked before the marker.
    with pytest.raises(ValueError, match="digest"):
        run.holdout(ctx, tmp_path, "c", frozen_sha)
    monkeypatch.setattr(run.m503, "check_frozen_digests", lambda *a: None)
    (tmp_path / "frozen_models.json").write_text("{\"changed\": 1}")
    with pytest.raises(ValueError, match="frozen_models.json changed"):
        run.holdout(ctx, tmp_path, "c", frozen_sha)
    (tmp_path / "frozen_models.json").write_text("{}")
    monkeypatch.setattr(run, "impact_provenance", lambda ctx: {"run_sha256": "other"})
    with pytest.raises(ValueError, match="#282's runner changed"):
        run.holdout(ctx, tmp_path, "c", frozen_sha)
    assert not (tmp_path / "holdout.started").exists()


def test_the_holdout_conditions_and_their_runs(monkeypatch):
    ctx = FakeCtx()
    seeds = [(arm, s, {"path": f"{arm}{s}"}) for arm in run.ARMS for s in range(1, 6)]
    monkeypatch.setattr(run, "policies", lambda ctx, out: seeds)
    models = {f"{arm}-seed{s}": {"checkpoint_sha256": "h"} for arm, s, _ in seeds}
    kappa = {"calm": 0.2, "stressed": 0.3, "thin": 0.4}
    jobs = run.holdout_jobs(ctx, "out", {"impact": {"kappa": kappa}}, models)
    conditions = {}
    for job in jobs:
        conditions.setdefault(job["condition"], set()).add(job["arm"])
    assert len([c for c in conditions if c.startswith("held_out:")]) == 12
    assert len([c for c in conditions if c.startswith("in_sample:")]) == 11
    per_arm = {arm: sum(arm in arms for arms in conditions.values()) for arm in run.ARMS}
    assert per_arm == {"adversarial": 26, "baseline": 26, "reference": 15}
    assert len(jobs) == (10 * 26 + 5 * 15) * 30
    impact = [j for j in jobs if j["condition"] == "impact"]
    assert {(j["regime"], j["kappa"]) for j in impact} == set(kappa.items())
    stress = [j for j in jobs if j["condition"] == "execution_stress"]
    assert all(j["execution"] == M5_03["execution_stress"]["fill_model"] for j in stress)
    assert all(j["frozen_sha256"] == "h" and j["split"] == "holdout" for j in jobs)
    # Without #282's runner the impact condition is not run, and nothing replaces it.
    jobs = run.holdout_jobs(ctx, "out", {"impact": None}, models)
    assert "impact" not in {j["condition"] for j in jobs}
    assert len(jobs) == (10 * 25 + 5 * 14) * 30


def test_a_failed_seed_is_frozen_as_failed_and_its_days_are_missing(tmp_path, monkeypatch):
    class Market:
        def digest(self):
            return "d"

    monkeypatch.setattr(run, "Context", lambda _: FakeCtx())
    monkeypatch.setattr(run, "market_for", lambda *a: (Market(), 7))
    row = run.run_policy({"run_dir": "x", "split": "holdout", "point": "p", "regime": "thin",
                          "day": 4, "metadata": None, "condition": "unperturbed",
                          "arm": "adversarial", "seed": 3, "frozen_sha256": None})
    assert row["net"] is None and row["error"] == "the seed failed in training"
    monkeypatch.setattr(run, "FROZEN_MODELS", tmp_path / "frozen_models.json")
    (tmp_path / "frozen_models.json").write_text(json.dumps(
        {"models": {"adversarial-seed3": {"failed": "the seed failed in training"}}}))
    monkeypatch.setattr(run, "policies",
                        lambda ctx, out: [("adversarial", 3, {"path": str(tmp_path / "x")})])
    with pytest.raises(ValueError, match="not as frozen"):
        run.check_models(FakeCtx(), tmp_path)
    monkeypatch.setattr(run, "policies", lambda ctx, out: [("adversarial", 3, None)])
    assert run.check_models(FakeCtx(), tmp_path)["adversarial-seed3"]["failed"]


def test_an_overrun_in_a_real_process_pool_stops_every_unfinished_run(tmp_path):
    """Spawned workers, killed at the limit: every run without an end is journaled stopped,
    never left rerunnable."""
    import m5_05_workers

    ctx = FakeCtx()
    ctx.parallel = 2
    started = time.perf_counter()
    with pytest.raises(run.Overrun):
        run.train(ctx, tmp_path, "c", limit=3, worker=m5_05_workers.sleeping)
    assert time.perf_counter() - started < 20
    ends = run.outcomes(tmp_path / "train.attempts")
    assert len(ends) == 10
    assert {r["end"]["outcome"] for r in ends.values()} == {"stopped"}
    for name in ends:
        with pytest.raises(ValueError, match="never rerun"):
            run.rerunnable(name, ends[name], (name,))
