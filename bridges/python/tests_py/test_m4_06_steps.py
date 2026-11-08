"""The rules of #103's steps, with the evaluations stubbed: the fee rule, the classical selection,
what the freeze and the holdout refuse, the jobs they run, and what an evaluated day counts. On
made-up values, not the experiment's."""

import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_06_run_steps", HERE.parent / "experiments" / "m4_06" / "run.py"
)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


A, B = json.dumps("a"), json.dumps("b")


class Ctx:
    """What the steps read of a context, with jobs that name their own daily value."""

    def __init__(self, out, values):
        self.design, self.digests = run.load_design()
        self.out = pathlib.Path(out)
        self.parallel = 1
        self.regimes = ["calm", "thin"]
        self.intervals = [1, 5]
        self.values = values
        self.written = {}

    def header(self, step):
        return {"step": step, "design_sha256": run.sha256(run.HERE / "design.json"),
                "digests_sha256": run.sha256(run.HERE / "digests.json")}

    def classical_jobs(self, fee, split, days, intervals, chosen=None, arms=run.CLASSICAL):
        # As the real ones: one hold per day beside the points.
        return [{"arm": arm, "every": every, "fee": fee, "regime": regime, "day": day,
                 "candidate": candidate}
                for every in intervals for regime in self.regimes for day in days
                for arm, candidate in [("no_trade", "hold")]
                + [(arm, c) for arm in arms for c in (A, B)]]

    def learned_jobs(self, runs, fee, split, days):
        return [{"arm": "mlp", "every": 1, "fee": fee, "regime": regime, "day": day,
                 "candidate": "seed1"} for regime in self.regimes for day in days]

    def write(self, name, report):
        self.written[name] = report
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(report))
        return path


@pytest.fixture
def stubbed(monkeypatch):
    def pool(jobs, worker, parallel, label):
        return [job | {"daily_value": str(ctx_values(job))} for job in jobs]

    holder = {}

    def ctx_values(job):
        return holder["ctx"].values(job)

    monkeypatch.setattr(run, "pool", pool)
    return holder


def test_the_fee_rule_takes_the_largest_profitable_fee_and_else_the_secondary(tmp_path, stubbed):
    def profitable_at_zero_or_below(job):
        return 1 if job["fee"] in ("0", "-0.0001") and job["candidate"] == B else -1

    ctx = Ctx(tmp_path, profitable_at_zero_or_below)
    stubbed["ctx"] = ctx
    report = run.fee_rule(ctx)
    assert report["fee"] == "0" and report["profitable"]
    assert report["best_per_fee"]["0"]["point"] == "b"
    ctx = Ctx(tmp_path, lambda job: -1)
    stubbed["ctx"] = ctx
    report = run.fee_rule(ctx)
    assert report["fee"] == ctx.design["execution"]["secondary_fee_rate"]
    assert not report["profitable"]
    # A mean of exactly 0 is not profitable.
    ctx = Ctx(tmp_path, lambda job: 0 if job["fee"] == "-0.0001" else -1)
    stubbed["ctx"] = ctx
    assert not run.fee_rule(ctx)["profitable"]


def test_the_fee_rule_reads_only_avellaneda_stoikov_on_training_days(tmp_path, stubbed):
    ctx = Ctx(tmp_path, lambda job: 1 if job["candidate"] == A else 0)
    stubbed["ctx"] = ctx
    report = run.fee_rule(ctx)
    assert {r["arm"] for r in report["rows"]} == {"avellaneda_stoikov", "no_trade"}
    assert {r["day"] for r in report["rows"]} == set(run.days_of(ctx.design, "train"))
    assert {r["every"] for r in report["rows"]} == {1}
    # The better point, though the other is the greater name.
    assert report["best_per_fee"]["0.0002"]["point"] == "a"
    assert report["best_per_fee"]["0.0002"]["mean_daily_net"] == "1"


def test_the_classical_selection_picks_per_regime_arm_and_interval(tmp_path, stubbed):
    def value(job):
        best = "a" if (job["regime"], job["every"]) == ("calm", 1) else "b"
        return 2 if job["candidate"] == json.dumps(best) else 0

    ctx = Ctx(tmp_path, value)
    stubbed["ctx"] = ctx
    chosen = run.classical_selection(ctx, "-0.0001")["selected"]
    assert chosen["calm/avellaneda_stoikov/1"] == "a"
    assert chosen["calm/glft/5"] == "b" and chosen["thin/glft/1"] == "b"
    # A tie goes to the greater candidate.
    ctx = Ctx(tmp_path, lambda job: 0)
    stubbed["ctx"] = ctx
    assert set(run.classical_selection(ctx, "-0.0001")["selected"].values()) == {"b"}


def frozen_run(tmp_path):
    zipped = tmp_path / "mlp-every1-seed1.zip"
    zipped.write_bytes(b"weights")
    for name in ("fee_rule", "classical_selection"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"fee": "-0.0001", "selected": {}}))
    (tmp_path / "training.json").write_text(json.dumps({"runs": [
        {"path": str(zipped.with_suffix("")), "checkpoint_sha256": run.sha256(zipped)}]}))
    return zipped


def test_the_freeze_refuses_a_checkpoint_that_changed_after_training(tmp_path):
    zipped = frozen_run(tmp_path)
    ctx = Ctx(tmp_path, None)
    assert run.freeze(ctx)["runs"] == 1
    zipped.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="changed after training"):
        run.freeze(ctx)


def test_the_holdout_runs_once_from_the_manifest_it_was_given(tmp_path, monkeypatch):
    frozen_run(tmp_path)
    ctx = Ctx(tmp_path, None)
    good = run.freeze(ctx)["frozen_sha256"]
    with pytest.raises(ValueError, match="not the one given"):
        run.holdout(ctx, "0" * 64)
    (tmp_path / "training.json").write_text("{}")
    with pytest.raises(ValueError, match="training.json changed"):
        run.holdout(ctx, good)
    (tmp_path / "holdout.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluated already"):
        run.holdout(ctx, good)


def test_the_holdout_refuses_a_design_other_than_the_frozen_one(tmp_path):
    frozen_run(tmp_path)
    ctx = Ctx(tmp_path, None)
    for key, name in (("design_sha256", "design.json"), ("digests_sha256", "digests.json")):
        run.freeze(ctx)
        manifest = json.loads((tmp_path / "frozen.json").read_text())
        manifest[key] = "0" * 64
        (tmp_path / "frozen.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=f"{name} changed after the freeze"):
            run.holdout(ctx, run.sha256(tmp_path / "frozen.json"))


def test_the_holdout_evaluates_every_arm_at_the_chosen_and_the_secondary_fee(tmp_path, stubbed):
    frozen_run(tmp_path)
    ctx = Ctx(tmp_path, lambda job: 0)
    stubbed["ctx"] = ctx
    seen = []
    plain = ctx.classical_jobs
    ctx.classical_jobs = lambda fee, split, days, intervals, chosen=None: (
        seen.append((fee, split, intervals, chosen)) or plain(fee, split, days, intervals))
    report = run.holdout(ctx, run.freeze(ctx)["frozen_sha256"])
    assert report["fees"] == ["-0.0001", "0.0002"]
    assert seen == [(fee, "holdout", [1, 5], {}) for fee in report["fees"]]
    rows = report["rows"]
    assert {r["day"] for r in rows} == set(run.days_of(ctx.design, "holdout"))
    for fee in report["fees"]:
        assert {r["arm"] for r in rows if r["fee"] == fee} == {*run.CLASSICAL, "no_trade", "mlp"}
    assert json.loads((tmp_path / "holdout.json").read_text())["fees"] == report["fees"]
    # A chosen fee that is the secondary one is evaluated once.
    (tmp_path / "holdout.json").unlink()
    (tmp_path / "fee_rule.json").write_text(json.dumps({"fee": "0.0002"}))
    report = run.holdout(ctx, run.freeze(ctx)["frozen_sha256"])
    assert report["fees"] == ["0.0002"]


def bare_context(tmp_path):
    """A context without its data: the design, two regimes and their volatilities."""
    ctx = run.Context.__new__(run.Context)
    ctx.design, ctx.digests = run.load_design()
    ctx.data_dir, ctx.commit, ctx.out = "data", "c0ffee", tmp_path
    ctx.regimes, ctx.intervals = ["calm", "thin"], [1, 5]
    ctx.sigmas = {"calm": 0.1, "thin": 0.2}
    return ctx


def test_the_jobs_cover_every_point_or_only_the_chosen_ones_and_one_hold_per_day(tmp_path):
    ctx = bare_context(tmp_path)
    jobs = ctx.classical_jobs("0", "validation", [3, 4], [1, 5])
    grids = {arm: len(run.grid_points(ctx.design["arms"][arm]["grid"], 0)) for arm in run.CLASSICAL}
    count = lambda arm: sum(j["arm"] == arm for j in jobs)  # noqa: E731
    assert count("no_trade") == 2 * 2 * 2
    for arm in run.CLASSICAL:
        assert count(arm) == 2 * 2 * 2 * grids[arm]
    assert {j["point"]["volatility"] for j in jobs if j["arm"] == "glft" and j["regime"] == "thin"} \
        == {0.2}
    hold = next(j for j in jobs if j["arm"] == "no_trade")
    assert hold["policy"] == {"kind": "hold"} and hold["candidate"] == "hold"
    chosen = {f"{r}/{a}/{e}": json.loads(next(
        j["candidate"] for j in jobs if (j["regime"], j["arm"], j["every"]) == (r, a, e)))
        for r in ctx.regimes for a in run.CLASSICAL for e in (1, 5)}
    picked = ctx.classical_jobs("0", "holdout", [3], [1, 5], chosen)
    assert len(picked) == 2 * 2 * (1 + len(run.CLASSICAL))
    assert all(j["point"] == chosen[f"{j['regime']}/{j['arm']}/{j['every']}"]
               for j in picked if j["arm"] != "no_trade")
    runs = [{"arm": "lstm", "every": 5, "seed": 3}, {"arm": "mlp", "every": 1, "seed": 4}]
    learned = ctx.learned_jobs(runs, "0", "holdout", [3, 4])
    assert len(learned) == 2 * 2 * 2
    assert {(j["arm"], j["every"], j["candidate"]) for j in learned} == {
        ("lstm", 5, "seed3"), ("mlp", 1, "seed4")}
    assert all(j["policy"] == {"kind": "frozen", "metadata": m}
               for m in runs for j in learned if j["candidate"] == f"seed{m['seed']}")


class Script:
    """An environment that replays steps, with a position path for the attribution."""

    def __init__(self, steps, positions, max_abs):
        self.steps, self.positions, self.max_abs = steps, positions, max_abs
        self.unwrapped = self

    def reset(self, seed):
        self.index = 0
        return 0, {}

    def step(self, action):
        info = self.steps[self.index]
        self.index += 1
        done = self.index == len(self.steps)
        return self.index, 0.0, done, False, info

    def attribution(self):
        return {"window_ms": (0, 100), "positions": self.positions,
                "max_abs_position": self.max_abs}


def step(rows, kind, refused=None, net="0", **orders):
    counts = dict.fromkeys(run.ORDER_KINDS, 0) | orders
    return {"rows": rows, "applied_action": {"kind": kind}, "refused_because": refused,
            "orders": counts, "net": net}


class Recording:
    def __init__(self):
        self.starts = []

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        self.starts.append(None if episode_start is None else bool(episode_start[0]))
        return [0], "state"


def test_an_evaluated_day_counts_decisions_rows_quotes_refusals_and_orders():
    design, _ = run.load_design()
    steps = [step(5, "quote", accepted=2), step(5, "hold", canceled=1, fills=1),
             step(3, "hold", refused="limits", net="7", rejected=1, cancel_rejected=1)]
    # At most 5 lots for the first half of the day, then 6: half the day beyond 5 lots.
    env = Script(steps, [(0, "100", "5"), (50, "100", "-6")], "6")
    policy = Recording()
    day = run.episode(policy, env, 101, True, design)
    assert policy.starts == [True, False, False]
    assert (day["decisions"], day["rows"], day["quote_transactions"], day["refusals"]) == \
        (3, 13, 1, 1)
    assert day["orders"] == {"accepted": 2, "rejected": 1, "canceled": 1,
                             "cancel_rejected": 1, "fills": 1}
    assert day["net"] == "7" and day["seed"] == 101
    assert day["time_beyond_5_lots"] == 0.5
    assert day["peak_notional"] == "600"
    assert not day["limit_breach"]
    assert "positions" not in day["attribution"]


def test_a_limit_breach_is_a_position_or_a_notional_beyond_the_limits():
    design, _ = run.load_design()
    limits = design["execution"]["limits"]
    at = Script([step(1, "hold")], [(0, "125", limits["max_position"])], limits["max_position"])
    policy = Recording()
    assert not run.episode(policy, at, 1, False, design)["limit_breach"]
    # A policy without memory is not given an episode start.
    assert policy.starts == [None]
    beyond = Script([step(1, "hold")], [(0, "1", "21")], "21")
    assert run.episode(Recording(), beyond, 1, False, design)["limit_breach"]
    notional = Script([step(1, "hold")], [(0, "200", "13")], "13")
    assert run.episode(Recording(), notional, 1, False, design)["limit_breach"]


def test_a_residual_step_names_its_composed_action_and_a_refusal_holds():
    composed = {"refused_because": None, "residual": {"composed": {"kind": "quote"}}}
    assert run.applied_kind(composed) == "quote"
    assert run.applied_kind(composed | {"refused_because": "limits"}) == "hold"
    assert run.applied_kind({"applied_action": {"kind": "hedge"}}) == "hedge"


def test_time_beyond_is_the_share_of_the_window_beyond_the_lots():
    window = {"window_ms": (100, 200)}
    assert run.time_beyond(window | {"positions": [(50, "1", "6"), (150, "1", "0")]}, 5) == 0.5
    assert run.time_beyond(window | {"positions": [(120, "1", "-6"), (300, "1", "0")]}, 5) == 0.8
    # A position held to the end of the window counts until the end.
    assert run.time_beyond(window | {"positions": [(150, "1", "7")]}, 5) == 0.5
    assert run.time_beyond({"window_ms": (5, 5), "positions": [(0, "1", "9")]}, 5) == 0.0
