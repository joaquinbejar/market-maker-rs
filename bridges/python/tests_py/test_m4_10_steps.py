"""The rules of #110's runner: the teacher's projection and cloning, the predictor, the arms'
environments, the markouts, and what each step chooses, runs and refuses, with the evaluations
stubbed where they would read the experiment's days. On made-up rows and values."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest
from conftest import walk
from stable_baselines3 import PPO

from mm_rl_bridge import BoxMarketMakingEnv
from mm_rl_bridge.wrappers import ChosenDuration

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_10_run", HERE.parent / "experiments" / "m4_10" / "run.py"
)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

DESIGN, _ = run.load_design()
GRID = DESIGN["action_space"]["grid"]
DURATIONS = DESIGN["duration"]["choices_rows"]
# A row whose mid is 200.5 ticks: floor 200, ceil 201.
ODD = (0, 199, 5, 202, 5)


def quote(bid, ask, bid_lots=1, ask_lots=1):
    return {"kind": "quote", "bid": bid, "ask": ask, "bid_lots": bid_lots, "ask_lots": ask_lots}


def offsets_of(target):
    """The grid offsets and sizes a target's values fall in, by the Box's own bins."""
    bins = BoxMarketMakingEnv._bin
    n, m = len(GRID["bid_offsets"]), len(GRID["sizes"])
    return (GRID["bid_offsets"][bins(target[1], n)], GRID["ask_offsets"][bins(target[2], n)],
            GRID["sizes"][bins(target[3], m)], GRID["sizes"][bins(target[4], m)])


def test_a_quote_is_projected_from_the_observed_mid_onto_the_nearest_offsets():
    # Bid 2 below floor(mid), ask 4 above ceil(mid).
    target = run.teacher_target(quote(198, 205, 1, 2), ODD, GRID, DURATIONS)
    assert target[0] > 0 and offsets_of(target) == (2, 4, 1, 2)
    assert len(target) == 6 and target[5] == run.bin_centre(0, len(DURATIONS))
    assert ChosenDuration.duration_of(type("D", (), {"choices": tuple(DURATIONS)})(),
                                      target[5]) == DURATIONS[0]
    # An even mid: both sides measure from it.
    assert offsets_of(run.teacher_target(quote(199, 201), (0, 199, 5, 201, 5), GRID, None)) \
        == (1, 1, 1, 1)
    # The teacher's own duration, when given.
    assert run.teacher_target(quote(198, 205), ODD, GRID, DURATIONS, duration=10)[5] == \
        run.bin_centre(DURATIONS.index(10), len(DURATIONS))
    # Without a duration the target has the Box's five values.
    assert len(run.teacher_target(quote(198, 205), ODD, GRID, None)) == 5


def test_the_projection_clamps_snaps_to_the_farther_offset_and_the_larger_size():
    # 3 away sits between 2 and 4, 6 between 4 and 8: ties go to the farther.
    assert offsets_of(run.teacher_target(quote(197, 207), ODD, GRID, None))[:2] == (4, 8)
    # Inside the mid is 0; beyond the grid is its farthest offset.
    assert offsets_of(run.teacher_target(quote(201, 230), ODD, GRID, None))[:2] == (0, 8)
    # 3 lots is nearest 2; a tie between 1 and 2 would go to 2.
    assert offsets_of(run.teacher_target(quote(200, 201, 3, 1), ODD, GRID, None))[2:] == (2, 1)
    assert run.nearest(1.5, [1, 2]) == 2 and run.nearest(3, [2, 4]) == 4
    assert run.nearest(5, [4, 8]) == 4


def test_a_one_sided_quote_puts_the_missing_side_far_and_small_and_a_hold_is_negative():
    bid_only = run.teacher_target({"kind": "bid", "price": 198, "lots": 2}, ODD, GRID, None)
    assert bid_only[0] > 0 and offsets_of(bid_only) == (2, 8, 2, 1)
    ask_only = run.teacher_target({"kind": "ask", "price": 202, "lots": 1}, ODD, GRID, None)
    assert offsets_of(ask_only) == (8, 1, 1, 1)
    for kind in ({"kind": "hold"}, {"kind": "strategy_error", "code": 3}):
        target = run.teacher_target(kind, ODD, GRID, DURATIONS)
        assert target[0] < 0 and len(target) == 6
    other = dict(GRID, ask_offsets=[0, 1])
    with pytest.raises(ValueError, match="one offset grid"):
        run.teacher_target(quote(198, 205), ODD, other, None)


def test_the_projection_counts_what_it_clamped_snapped_and_made_live():
    notes = dict.fromkeys(run.PROJECTION_NOTES, 0)
    # Bid 1 inside the mid, ask 3 out: clamped to 0 and snapped to 4.
    run.teacher_target(quote(201, 204), ODD, GRID, None, notes)
    # Bid 12 out, beyond the grid: clamped to 8; ask exact.
    run.teacher_target(quote(188, 202), ODD, GRID, None, notes)
    run.teacher_target({"kind": "ask", "price": 202, "lots": 1}, ODD, GRID, None, notes)
    run.teacher_target({"kind": "hold"}, ODD, GRID, None, notes)
    # At the mid and at the farthest offset exactly: neither clamped nor snapped.
    run.teacher_target(quote(200, 209), ODD, GRID, None, notes)
    notes["quotes"] -= 1
    assert notes == {"quotes": 3, "holds": 1, "one_sided": 1, "inside_mid": 1,
                     "beyond_grid": 1, "off_grid": 3, "error_ticks": 1 + 1 + 4}


def test_bin_centres_fall_in_their_bins():
    assert [run.bin_centre(i, 4) for i in range(4)] == [-0.75, -0.25, 0.25, 0.75]
    assert run.teacher_target(quote(198, 205, 1, 2), ODD, GRID, DURATIONS) == pytest.approx(
        [0.5, 0.0, 0.4, -0.5, 0.5, -0.75])
    for count in (2, 4, 5):
        for index in range(count):
            assert BoxMarketMakingEnv._bin(run.bin_centre(index, count), count) == index


def test_the_mid_change_is_over_the_next_rows_and_none_near_the_end():
    rows = [(0, 100 + i, 1, 102 + i * 2, 1) for i in range(8)]
    changes = run.mid_changes(rows, 5)
    assert changes[:3] == [7.5, 7.5, 7.5] and changes[3:] == [None] * 5


def test_the_predictor_fit_recovers_a_linear_rule():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    y = x @ [0.5, -2.0, 0.0] + 0.25
    coefficients, r2, rank = run.fit_predictor(list(zip(x.tolist(), y.tolist())))
    assert coefficients == pytest.approx([0.5, -2.0, 0.0, 0.25])
    assert r2 == pytest.approx(1.0) and rank == 4
    _, flat, _ = run.fit_predictor([([float(i)], 1.0) for i in range(5)])
    assert flat == 0.0
    # A value constant over the samples gets the minimum norm share of the intercept.
    constant = [([1.0, float(i)], 2.0 * i + 1.0) for i in range(6)]
    coefficients, _, rank = run.fit_predictor(constant)
    assert rank == 2 and coefficients == pytest.approx([0.5, 2.0, 0.5])


def test_the_forecast_is_appended_clipped_and_needs_one_weight_per_value():
    env = run.Forecast(BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), walk(40)),
                       [0.0] * 22 + [0.3])
    observation, _ = env.reset(seed=1)
    assert observation.shape == (23,) and observation[-1] == pytest.approx(0.3)
    assert env.observation_space.contains(observation)
    assert env.forecast(np.ones(22)) == pytest.approx(0.3)
    loud = run.Forecast(env.env, [1.0] * 22 + [0.0])
    assert loud.forecast(np.ones(22)) == 1.0 and loud.forecast(-np.ones(22)) == -1.0
    with pytest.raises(ValueError, match="23 coefficients"):
        run.Forecast(env.env, [0.0] * 22)
    env.close()


def test_each_arm_has_its_duration_and_forecast():
    predictor = {"coefficients": [0.0] * 23}
    shapes = {}
    for arm in run.LEARNED:
        env = run.wrap(BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), walk(40)), DESIGN,
                       arm, predictor if DESIGN["arms"][arm]["predictor"] else None)
        shapes[arm] = (env.action_space.shape[0], env.observation_space.shape[0])
        env.close()
    assert shapes == {"full": (6, 23), "no_teacher": (6, 23), "no_duration": (5, 23),
                      "no_predictor": (6, 22)}


def test_a_markout_is_the_signed_quantity_times_the_later_mid_less_the_price():
    rows = [(ms * 1_000_000, 100, 1, 102, 1) for ms in (0, 1000, 2000)] + \
        [(ms * 1_000_000, 110, 1, 112, 1) for ms in (3000, 4000)]
    # A buy of 2 at 101 and a sell of 3 at 102, both before the mid rises to 111.
    positions = [(500, 101, "2"), (1500, 102, "-1")]
    out = run.markouts(positions, rows, (1000, 2000))
    # After 1 s: the buy sees 101 (row at 1000), the sell 101 (row at 2000): 0 + 3.
    assert out["1000"] == {"ticks": "3", "fills": 2}
    # After 2 s: the buy sees 101 (row at 2000), the sell 111 (row at 3000): 0 - 27.
    assert out["2000"] == {"ticks": "-27", "fills": 2}
    # A fill with no row that late is left out; one landing on the last row is counted.
    assert run.markouts(positions, rows, (2000000,))["2000000"] == {"ticks": "0", "fills": 0}
    assert run.markouts([(1000, 101, "1")], rows, (3000,))["3000"] == {"ticks": "10", "fills": 1}
    # A horizon landing on a row reads that row.
    assert run.markouts([(1000, 101, "1")], rows, (2000,))["2000"] == {"ticks": "10", "fills": 1}


def test_cloning_moves_the_mean_toward_the_targets_and_nothing_else():
    env = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), walk(40))
    model = PPO("MlpPolicy", env, seed=1, device="cpu", n_steps=64, batch_size=32)
    before = {name: p.detach().clone() for name, p in model.policy.named_parameters()}
    rng = np.random.default_rng(2)
    observations = rng.normal(size=(512, 22)).tolist()
    targets = [[0.5, 0.2, -0.2, -0.5, 0.5]] * 512
    settings = {"learning_rate": 3e-3, "batch_size": 64, "epochs": 20}
    losses = run.clone(model, observations, targets, settings, seed=3)
    assert len(losses) == 20 and losses[-1] < losses[0] / 10
    moved = {name for name, p in model.policy.named_parameters()
             if not np.array_equal(p.detach().numpy(), before[name].numpy())}
    assert moved and all(name.startswith(("mlp_extractor.policy_net", "action_net"))
                         for name in moved)
    mean = model.policy.get_distribution(
        model.policy.obs_to_tensor(np.asarray(observations[:4], dtype=np.float32))[0]
    ).distribution.mean.detach().numpy()
    assert mean == pytest.approx(np.asarray(targets[:4]), abs=0.05)
    # The same seed, the same fit.
    again = PPO("MlpPolicy", env, seed=1, device="cpu", n_steps=64, batch_size=32)
    assert run.clone(again, observations, targets, settings, seed=3) == losses
    env.close()


def walk_on(day, count=60):
    """The short walk, moved to its UTC day."""
    shift = (day - 1) * 86_400_000 * 1_000_000
    return [(ts + shift, *rest) for ts, *rest in walk(count)]


@pytest.fixture
def one_day(monkeypatch):
    """Every day is the same short walk on its own day, with its own digest."""
    digests = {}
    reads = []

    def read_rows(data_dir, digests_, regime, day, split, holdout_allowed=False):
        reads.append((regime, day, split))
        rows = walk_on(day)
        if day not in digests:
            probe = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), rows)
            digests[day] = {"dataset_digest": probe.bridge_info["dataset_digest"]}
            probe.close()
        return rows, digests[day]

    monkeypatch.setattr(run, "read_rows", read_rows)
    return walk_on(1), reads


def test_predictor_samples_pair_each_observation_with_its_rows_change(one_day):
    rows, reads = one_day
    samples = run.predictor_samples({"regime": "calm", "day": 1, "fee": "0", "latency": "25",
                                     "data_dir": "x"})
    assert reads == [("calm", 1, "train")]
    scale = float(DESIGN["observation_scales"]["mid_change_ticks"])
    changes = run.mid_changes(rows, run.FORECAST_ROWS)
    env = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), rows)
    observation, info = env.reset(seed=DESIGN["evaluation"]["evaluation_seeds"][0])
    expected = []
    while True:
        if changes[info["row"]] is not None:
            expected.append((observation.tolist(), changes[info["row"]] / scale))
        observation, _, terminated, truncated, info = env.step(run.HOLD)
        if terminated or truncated:
            break
    env.close()
    assert samples == expected and len(samples) > 40
    # The last rows of the day have no change to forecast.
    assert all(change is not None for _, change in samples)


def test_teacher_samples_are_the_classical_decisions_until_the_budget(one_day, monkeypatch):
    rows, reads = one_day
    sigmas = dict(zip(run.regimes_of(DESIGN), ("0.5", "0.6", "0.7")))
    volatilities, projected = [], []
    baseline, project = run.baseline_config, run.teacher_target

    def baseline_config(kind, point, day):
        volatilities.append(point["volatility"])
        return baseline(kind, point, day)

    def teacher_target(classical, row, grid, durations, notes=None, duration=None):
        projected.append((classical, row))
        return project(classical, row, grid, durations, notes, duration)

    monkeypatch.setattr(run, "baseline_config", baseline_config)
    monkeypatch.setattr(run, "teacher_target", teacher_target)
    point = {"risk_aversion": "0.01", "order_intensity": "50", "size": "1",
             "horizon_ms": 7200000}
    job = {"arm": "full", "latency": "25", "seed": 4, "fee": "0", "point": point,
           "sigmas": sigmas, "decisions": 130, "data_dir": "x",
           "predictor": {"coefficients": [0.0] * 22 + [0.25]}}
    observations, targets, notes = run.teacher_samples(job)
    assert len(observations) == len(targets) == 130
    # A day has fewer decisions than the budget: more than one day is drawn.
    assert len(reads) >= 3 and all(split == "train" for *_, split in reads)
    assert all(len(o) == 23 and o[-1] == 0.25 for o in observations)
    assert all(len(t) == 6 for t in targets) and any(t[0] > 0 for t in targets)
    every = DESIGN["runner_choices"]["teacher"]["every_rows"]
    assert every == 10 and all(t[5] == run.bin_centre(DURATIONS.index(every), len(DURATIONS))
                               for t in targets)
    assert notes["fills"] >= 0 and set(notes) == {*run.PROJECTION_NOTES, "fills"}
    # The same seed draws the same days and decisions.
    assert notes["quotes"] + notes["holds"] == 130 and notes["quotes"] > 0
    # Each day's point has its regime's volatility.
    assert volatilities == [sigmas[regime] for regime, *_ in reads]
    # The first day again, directly: each target is of the row its decision observed.
    regime, day, _ = reads[0]
    rng = np.random.default_rng(4)
    rng.integers(len(run.regimes_of(DESIGN)) * len(run.days_of(DESIGN, "train")))
    cfg = run.config(DESIGN, "0", "25")
    cfg["residual"] = {"baseline": baseline("avellaneda_stoikov",
                                            point | {"volatility": sigmas[regime]}, day),
                       "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}
    inner = run.ResidualMarketMakingEnv(cfg, walk_on(day))
    direct = run.DecisionInterval(inner, every)
    _, info = direct.reset(seed=int(rng.integers(2**31)))
    for k in range(4):
        observed = walk_on(day)[info["row"]]
        *_, info = direct.step([0.0, 0.0, 0.0, 0.0])
        assert projected[k] == (info["residual"]["classical"], observed)
        assert info["rows"] == every
    direct.close()
    projected.clear()
    assert run.teacher_samples(job) == (observations, targets, notes)
    loud = run.teacher_samples(job | {"predictor": {"coefficients": [0.0] * 22 + [3.0]}})
    assert all(o[-1] == 1.0 for o in loud[0])
    flat = run.teacher_samples(job | {"arm": "no_duration"})
    assert all(len(t) == 5 for t in flat[1]) and [t[:5] for t in targets] == flat[1]
    plain = run.teacher_samples(job | {"arm": "no_predictor"})
    assert all(len(o) == 22 for o in plain[0])
    assert plain[1] == targets and [o[:22] for o in observations] == plain[0]


class Ctx:
    """What the steps read of a context, with jobs that name their own daily value."""

    def __init__(self, out, values):
        self.design, self.digests = run.load_design()
        self.out = pathlib.Path(out)
        (self.out / "teacher").mkdir(exist_ok=True)
        self.parallel, self.data_dir, self.commit = 1, "x", "c0ffee"
        self.regimes = ["calm", "thin"]
        self.latencies = ["25", "500"]
        self.sigmas = {"calm": 0.1, "thin": 0.2}
        self.values = values

    header = run.Context.header
    job = run.Context.job
    classical_jobs = run.Context.classical_jobs
    learned_jobs = run.Context.learned_jobs
    write = run.Context.write
    read = run.Context.read


def header(self, step):
    return {"step": step, "design_sha256": run.sha256(run.HERE / "design.json"),
            "digests_sha256": run.sha256(run.HERE / "digests.json")}


Ctx.header = header


@pytest.fixture
def stubbed(monkeypatch):
    holder = {"pools": []}

    def pool(jobs, worker, parallel, label):
        holder["pools"].append((worker.__name__, len(jobs)))
        return [job | {"daily_value": str(holder["ctx"].values(job)),
                       "episodes": [{"attribution": {"fills": 1}}]} for job in jobs]

    monkeypatch.setattr(run, "pool", pool)
    return holder


def test_the_fee_rule_compares_points_over_every_regime_and_at_the_shortest_latency(
        tmp_path, stubbed):
    # In calm intensity 200 is best, in thin 50, and 100 is second in both: pooled, 100 wins
    # at fee 0 and -0.0001 only.
    def value(job):
        if job["arm"] == "no_trade":
            return 0
        if job["fee"] not in ("0", "-0.0001"):
            return -1
        scores = {"calm": {"200": 3, "100": 2, "50": -4}, "thin": {"50": 3, "100": 2, "200": -4}}
        bonus = 0.5 if job["point"]["risk_aversion"] == "0.001" else 0
        return scores[job["regime"]][job["point"]["order_intensity"]] + bonus

    ctx = Ctx(tmp_path, value)
    stubbed["ctx"] = ctx
    report = run.fee_rule(ctx)
    assert report["fee"] == "0" and report["profitable"]
    assert report["teacher_point"] == {"horizon_ms": 7200000, "order_intensity": "100",
                                       "risk_aversion": "0.001", "size": "1"}
    assert {(r["latency"], r["every"], r["split"]) for r in report["rows"]} == {("25", 10, "train")}
    # One fill per stubbed day: the chosen point's days at the chosen fee, every regime.
    assert report["teacher_point_fills"] == len(ctx.regimes) * len(run.days_of(DESIGN, "train"))
    assert {r["day"] for r in report["rows"]} == set(run.days_of(DESIGN, "train"))
    ctx = Ctx(tmp_path, lambda job: 0)
    stubbed["ctx"] = ctx
    report = run.fee_rule(ctx)
    assert report["fee"] == DESIGN["execution"]["secondary_fee_rate"] and not report["profitable"]


def test_the_classical_selection_picks_per_regime_latency_and_interval(tmp_path, stubbed):
    def value(job):
        if job["arm"] == "no_trade":
            return 0
        best = "50" if (job["regime"], job["latency"], job["every"]) == ("calm", "25", 1) \
            else "200"
        return 1 if job["point"]["order_intensity"] == best else 0

    ctx = Ctx(tmp_path, value)
    stubbed["ctx"] = ctx
    chosen = run.classical_selection(ctx, "0")["selected"]
    assert len(chosen) == 2 * 2 * 2
    assert chosen["calm/avellaneda_stoikov/25/1"]["order_intensity"] == "50"
    assert chosen["calm/avellaneda_stoikov/25/10"]["order_intensity"] == "200"
    assert chosen["calm/avellaneda_stoikov/500/1"]["order_intensity"] == "200"
    assert chosen["thin/avellaneda_stoikov/25/1"]["volatility"] == 0.2


def test_the_teacher_and_training_cover_their_arms_latencies_and_seeds(tmp_path, monkeypatch):
    ctx = Ctx(tmp_path, None)
    seen = {}

    def pool(jobs, worker, parallel, label):
        seen[label] = jobs
        if label == "teacher":
            return [{k: j[k] for k in ("arm", "latency", "seed")}
                    | {"path": j["path"], "projection": {"fills": 3}} for j in jobs]
        return [{"arm": j["arm"]} for j in jobs]

    monkeypatch.setattr(run, "pool", pool)
    fee_report = {"fee": "0", "teacher_point": {"order_intensity": "100"},
                  "teacher_point_fills": 5}
    predictor = {"coefficients": [0.0] * 23}
    taught = run.teacher(ctx, fee_report, predictor)
    seeds = DESIGN["training_seeds"]
    teachers = [a for a in run.LEARNED if DESIGN["arms"][a]["teacher"]]
    assert sorted(teachers) == ["full", "no_duration", "no_predictor"]
    assert {(j["arm"], j["latency"], j["seed"]) for j in seen["teacher"]} == {
        (a, lat, s) for a in teachers for lat in ctx.latencies for s in seeds}
    assert all(j["point"] == {"order_intensity": "100"} and j["sigmas"] == ctx.sigmas
               and j["decisions"] == DESIGN["runner_choices"]["teacher"]["decisions"]
               for j in seen["teacher"])
    with pytest.raises(ValueError, match="does not fill"):
        run.teacher(ctx, fee_report | {"teacher_point_fills": 0}, predictor)
    starved = dict(taught, samples=[s | {"projection": {"fills": 0}} if i == 1 else s
                                    for i, s in enumerate(taught["samples"])])
    with pytest.raises(ValueError, match="never filled"):
        run.training(ctx, "0", predictor, starved)
    run.training(ctx, "0", predictor, taught)
    jobs = seen["training"]
    assert len(jobs) == len(run.LEARNED) * len(ctx.latencies) * len(seeds)
    for job in jobs:
        has = DESIGN["arms"][job["arm"]]["teacher"]
        assert (job["teacher"] is not None) == has
        if has:
            assert (job["teacher"]["arm"], job["teacher"]["latency"],
                    job["teacher"]["seed"]) == (job["arm"], job["latency"], job["seed"])
        assert job["decisions"] == DESIGN["budget_decisions"] and job["predictor"] == predictor
    # A chosen duration at a short latency first; the arm without a duration last.
    assert jobs[0]["latency"] == "25" and DESIGN["arms"][jobs[0]["arm"]]["duration"]
    assert all(j["arm"] == "no_duration" for j in jobs[-len(ctx.latencies) * len(seeds):])


def frozen_run(tmp_path):
    zipped = tmp_path / "full-25-seed1.zip"
    zipped.write_bytes(b"weights")
    flat = tmp_path / "no_duration-25-seed1.zip"
    flat.write_bytes(b"other weights")
    samples = tmp_path / "teacher" / "full-25-seed1.json"
    samples.parent.mkdir(exist_ok=True)
    samples.write_text("{}")
    for name in ("fee_rule", "predictor"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"fee": "0", "coefficients": []}))
    grid = DESIGN["arms"]["avellaneda_stoikov"]["grid"]
    selected = {f"{regime}/avellaneda_stoikov/{latency}/{every}":
                run.grid_points(grid, sigma)[0]
                for regime, sigma in (("calm", 0.1), ("thin", 0.2)) for latency in ("25", "500")
                for every in run.CLASSICAL_INTERVALS}
    (tmp_path / "classical_selection.json").write_text(json.dumps({"selected": selected}))
    (tmp_path / "teacher.json").write_text(json.dumps({"samples": [
        {"path": str(samples), "sha256": run.sha256(samples)}]}))
    (tmp_path / "training.json").write_text(json.dumps({"runs": [
        {"arm": "full", "latency": "25", "seed": 1, "predictor": {"coefficients": [0.5]},
         "path": str(zipped.with_suffix("")), "checkpoint_sha256": run.sha256(zipped)},
        {"arm": "no_duration", "latency": "25", "seed": 1, "predictor": {"coefficients": [0.5]},
         "path": str(flat.with_suffix("")), "checkpoint_sha256": run.sha256(flat)}]}))
    return zipped, samples


def test_the_freeze_refuses_a_changed_checkpoint_or_teacher_sample(tmp_path):
    zipped, samples = frozen_run(tmp_path)
    ctx = Ctx(tmp_path, None)
    manifest = run.freeze(ctx)
    assert manifest["runs"] == 2
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    assert frozen["code_sha256"] == run.sha256(run.HERE / "run.py")
    assert frozen["base_code_sha256"] == run.sha256(run.HERE.parent / "m4_06" / "run.py")
    assert set(f"{n}_sha256" for n in run.FROZEN_REPORTS) <= set(frozen)
    samples.write_text('{"changed": 1}')
    with pytest.raises(ValueError, match="changed after it was drawn"):
        run.freeze(ctx)
    frozen_run(tmp_path)
    zipped.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="changed after training"):
        run.freeze(ctx)


def test_the_holdout_runs_once_from_its_manifest_at_both_fees(tmp_path, stubbed):
    frozen_run(tmp_path)
    ctx = Ctx(tmp_path, lambda job: 0)
    stubbed["ctx"] = ctx
    good = run.freeze(ctx)["frozen_sha256"]
    with pytest.raises(ValueError, match="not the one given"):
        run.holdout(ctx, "0" * 64)
    for name in run.FROZEN_REPORTS:
        original = (tmp_path / f"{name}.json").read_text()
        (tmp_path / f"{name}.json").write_text(original + " ")
        with pytest.raises(ValueError, match=f"{name}.json changed"):
            run.holdout(ctx, good)
        (tmp_path / f"{name}.json").write_text(original)
    manifest = json.loads((tmp_path / "frozen.json").read_text())
    for key, name in (("design_sha256", "m4_10/design.json"), ("digests_sha256", "m4_10/digests.json"),
                      ("code_sha256", "m4_10/run.py"), ("base_code_sha256", "m4_06/run.py")):
        changed = manifest | {key: "0" * 64}
        (tmp_path / "frozen.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=f"{name} changed"):
            run.holdout(ctx, run.sha256(tmp_path / "frozen.json"))
    (tmp_path / "frozen.json").write_text(json.dumps(manifest))
    report = run.holdout(ctx, run.sha256(tmp_path / "frozen.json"))
    assert report["fees"] == ["0", "0.0002"]
    rows = report["rows"]
    assert {r["split"] for r in rows} == {"holdout"}
    for fee in report["fees"]:
        assert {r["arm"] for r in rows if r["fee"] == fee and r.get("fill", "queue") == "queue"} == {
            "no_trade", "avellaneda_stoikov", "full", "no_duration"}
    # Descriptive: the arm without a duration under immediate fill, at the chosen fee only.
    immediate = [r for r in rows if r.get("fill") == "immediate"]
    assert {(r["arm"], r["fee"]) for r in immediate} == {("no_duration", "0")}
    assert len(immediate) == len(ctx.regimes) * len(run.days_of(DESIGN, "holdout"))
    # Only the chosen point, at each regime, latency and interval.
    classical = [r for r in rows if r["arm"] == "avellaneda_stoikov"]
    assert len(classical) == 2 * 2 * 2 * 2 * len(run.days_of(DESIGN, "holdout"))
    assert {(r["latency"], r["every"]) for r in rows if r["arm"] == "no_trade"} == {
        (lat, every) for lat in ctx.latencies for every in run.CLASSICAL_INTERVALS}
    # Each learned job carries its run's predictor.
    assert all(r["predictor"] == {"coefficients": [0.5]} for r in rows if r["arm"] == "full")
    with pytest.raises(ValueError, match="evaluated already"):
        run.holdout(ctx, run.sha256(tmp_path / "frozen.json"))
    # A chosen fee that is the secondary one is evaluated once.
    (tmp_path / "holdout.json").unlink()
    (tmp_path / "fee_rule.json").write_text(json.dumps({"fee": "0.0002"}))
    report = run.holdout(ctx, run.freeze(ctx)["frozen_sha256"])
    assert report["fees"] == ["0.0002"]


def test_an_evaluated_day_counts_its_durations_rows_and_orders_per_hour():
    env = run.wrap(BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), walk(80)), DESIGN,
                   "no_predictor", None)
    # A quote resting 5 rows, then holds of 1 row: every row is covered once.
    choice = {5: run.bin_centre(1, len(DURATIONS)), 1: -1.0}

    class Policy:
        def __init__(self):
            self.calls = 0

        def predict(self, observation, deterministic=True):
            self.calls += 1
            first = self.calls == 1
            return np.asarray([1.0 if first else -1.0, 0.0, 0.0, 0.0, 0.0,
                               choice[5 if first else 1]], dtype=np.float32), None

    class Longest(Policy):
        def predict(self, observation, deterministic=True):
            return np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), None

    rows = walk(80)
    day = run.episode(Policy(), env, 1, DESIGN, rows)
    env.close()
    assert day["durations"]["5"] == 1 and day["durations"]["1"] == day["decisions"] - 1
    assert day["holds_per_decision"] == (day["decisions"] - 1) / day["decisions"]
    assert day["rows"] == day["decisions"] + 4
    hours = day["rows"] * DESIGN["simulator"]["sample_ms"] / 3_600_000
    assert float(day["per_hour"]["accepted"]) == pytest.approx(day["orders"]["accepted"] / hours,
                                                               abs=1e-3)
    assert set(day["markouts_ticks"]) == {"1000", "5000", "30000"}
    # The last duration chosen is cut by the end of the day, and still counted as chosen.
    env = run.wrap(BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), rows), DESIGN,
                   "no_predictor", None)
    day = run.episode(Longest(), env, 1, DESIGN, rows)
    env.close()
    assert day["durations"] == {"30": day["decisions"]} and day["rows"] < 30 * day["decisions"]
    plain = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), rows)
    day = run.episode(run.Constant(run.HOLD), plain, 1, DESIGN, rows)
    plain.close()
    assert day["rows"] == day["decisions"] and day["durations"] == {"1": day["decisions"]}
    assert day["holds_per_decision"] == 1.0


def test_each_latency_has_its_own_delays():
    for latency, delays in DESIGN["execution"]["latencies"].items():
        cfg = run.config(DESIGN, "0.0002", latency)
        assert cfg["latency"] == delays and cfg["fee_rate"] == "0.0002"
        assert cfg["fill_model"] == DESIGN["execution"]["fill_model"]


def test_a_day_is_evaluated_on_its_own_rows_and_only_the_holdout_step_reads_the_holdout(
        one_day, monkeypatch):
    allowed = []
    plain = run.read_rows

    def read_rows(data_dir, digests, regime, day, split, holdout_allowed=False):
        allowed.append((split, holdout_allowed))
        return plain(data_dir, digests, regime, day, split, holdout_allowed)

    monkeypatch.setattr(run, "read_rows", read_rows)
    job = {"arm": "no_trade", "latency": "25", "every": 10, "fee": "0", "regime": "calm",
           "day": 20, "split": "holdout", "candidate": "hold", "policy": {"kind": "hold"},
           "point": None, "data_dir": "x"}
    row = run.evaluate_job(job)
    assert set(allowed) == {("holdout", True)} and row["fill"] == "queue"
    assert row["daily_value"] == "0" and row["latency"] == "25" and row["every"] == 10
    day = row["episodes"][0]
    assert day["durations"]["10"] >= 1 and sum(day["durations"].values()) == day["decisions"]
    allowed.clear()
    run.evaluate_job(job | {"split": "validation", "day": 15})
    assert set(allowed) == {("validation", False)}


def test_a_day_with_other_rows_than_its_digest_is_refused(monkeypatch):
    monkeypatch.setattr(run, "read_rows", lambda *a, **k: (walk(30), {"dataset_digest": "x"}))
    job = {"arm": "no_trade", "latency": "25", "every": 1, "fee": "0", "data_dir": "x"}
    with pytest.raises(ValueError, match="other rows than digests.json"):
        run.build_env(job, {}, "calm", 1, "train")


def test_a_frozen_checkpoint_loads_only_where_it_was_trained(tmp_path):
    env = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), walk(40, spacing_ms=100))
    PPO("MlpPolicy", env, seed=1, device="cpu", n_steps=64, batch_size=32).save(tmp_path / "m")
    metadata = {"path": str(tmp_path / "m"), "spec": list(env.spec_id), "config": env.config,
                "checkpoint_sha256": run.sha256(tmp_path / "m.zip")}
    assert run.load_frozen(metadata, env) is not None
    # Another fee is the comparison's secondary row; another latency is another environment.
    other_fee = BoxMarketMakingEnv(run.config(DESIGN, "0.0002", "25"), walk(40, spacing_ms=100))
    assert run.load_frozen(metadata, other_fee) is not None
    later = BoxMarketMakingEnv(run.config(DESIGN, "0", "500"), walk(40, spacing_ms=100))
    with pytest.raises(ValueError, match="another configuration"):
        run.load_frozen(metadata, later)
    # Another grid is another specification.
    wider = BoxMarketMakingEnv(run.config(DESIGN, "0", "25") | {"sizes": [1, 3]},
                               walk(40, spacing_ms=100))
    with pytest.raises(ValueError, match="another specification"):
        run.load_frozen(metadata, wider)
    wider.close()
    with pytest.raises(ValueError, match="changed after training"):
        run.load_frozen(metadata | {"checkpoint_sha256": "0" * 64}, env)
    # Under immediate fill only when the caller allows that one difference.
    instant = BoxMarketMakingEnv(run.config(DESIGN, "0", "25") | {"fill_model": run.IMMEDIATE},
                                 walk(40, spacing_ms=100))
    with pytest.raises(ValueError, match="another configuration"):
        run.load_frozen(metadata, instant)
    assert run.load_frozen(metadata, instant, ("fill_model",)) is not None
    instant.close()
    for e in (env, other_fee, later):
        e.close()


def test_a_learned_arm_decides_through_its_own_wrappers(one_day):
    job = {"arm": "full", "latency": "25", "fee": "0", "data_dir": "x",
           "predictor": {"coefficients": [0.0] * 23}, "every": None}
    env = run.build_env(job, {}, "calm", 1, "train")
    assert env.action_space.shape == (6,) and env.observation_space.shape == (23,)
    env.close()
    env = run.build_env(job | {"arm": "avellaneda_stoikov", "every": 10,
                               "point": {"risk_aversion": "0.01", "order_intensity": "50",
                                         "size": "1", "horizon_ms": 7200000,
                                         "volatility": "0.5"}}, {}, "calm", 1, "train")
    assert isinstance(env, run.DecisionInterval) and env.every == 10
    assert env.unwrapped.config["fill_model"] == DESIGN["execution"]["fill_model"]
    env.close()
    env = run.build_env(job | {"fill": "immediate"}, {}, "calm", 1, "train")
    assert env.unwrapped.config["fill_model"] == run.IMMEDIATE
    env.close()



def test_the_teacher_log_counts_its_rollouts_fills(monkeypatch):
    # Rows 100 ms apart, as the experiment's: a quote resting 10 rows clears a lot.
    def rows_on(day):
        shift = (day - 1) * 86_400_000 * 1_000_000
        return [(ts + shift, *rest) for ts, *rest in walk(300, spacing_ms=100)]

    digests = {}

    def read_rows(data_dir, digests_, regime, day, split, holdout_allowed=False):
        if day not in digests:
            probe = BoxMarketMakingEnv(run.config(DESIGN, "0", "25"), rows_on(day))
            digests[day] = {"dataset_digest": probe.bridge_info["dataset_digest"]}
            probe.close()
        return rows_on(day), digests[day]

    monkeypatch.setattr(run, "read_rows", read_rows)
    sigmas = dict.fromkeys(run.regimes_of(DESIGN), "0.5")
    point = {"risk_aversion": "0.01", "order_intensity": "50", "size": "1",
             "horizon_ms": 7200000}
    job = {"arm": "no_predictor", "latency": "25", "seed": 2, "fee": "0", "point": point,
           "sigmas": sigmas, "decisions": 60, "data_dir": "x", "predictor": None}
    _, _, notes = run.teacher_samples(job)
    assert notes["fills"] > 0
