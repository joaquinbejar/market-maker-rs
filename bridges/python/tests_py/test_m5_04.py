"""#282's reaction rules, as design.json states them. Synthetic only."""

import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m5_04"
SPEC = importlib.util.spec_from_file_location("m5_04_run", STUDY / "run.py")
run = importlib.util.module_from_spec(SPEC)
sys.modules["m5_04_run"] = run
SPEC.loader.exec_module(run)
DESIGN = json.loads((STUDY / "design.json").read_text())


def test_splitmix_and_fnv_are_the_standard_ones():
    # SplitMix64 from 0, and FNV-1a 64 of empty input and of "a".
    assert run.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF
    assert run.fnv1a64(b"") == 0xCBF29CE484222325
    assert run.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert 0.0 <= run.SplitMix64(7).unit() < 1.0


class Draws:
    def __init__(self, values):
        self.values = list(values)
        self.drawn = 0

    def unit(self):
        self.drawn += 1
        return self.values.pop(0)


def test_fills_are_netted_and_move_against_the_agent():
    rng = Draws([0.99] * 4)
    assert run.moves_for(2.0, {"buy": 3, "sell": 1}, rng) == -4, "bought 2 net: down"
    assert run.moves_for(2.0, {"buy": 0, "sell": 1}, rng) == 2, "sold 1: up"
    assert run.moves_for(2.0, {"buy": 2, "sell": 2}, rng) == 0
    assert run.moves_for(2.0, {"buy": 0, "sell": 0}, rng) == 0
    assert rng.drawn == 2, "one draw per step with a net, none otherwise"


def test_the_fraction_is_one_more_tick_when_the_draw_is_below_it():
    assert run.moves_for(0.25, {"buy": 0, "sell": 2}, Draws([0.49])) == 1
    assert run.moves_for(0.25, {"buy": 0, "sell": 2}, Draws([0.5])) == 0
    assert run.moves_for(0.25, {"buy": 2, "sell": 0}, Draws([0.1])) == -1
    # Drawn even when the fraction is 0.
    rng = Draws([0.0])
    assert run.moves_for(1.0, {"buy": 0, "sell": 3}, rng) == 3 and rng.drawn == 1


def test_the_rounding_has_the_fraction_as_its_frequency():
    rng = run.impact_rng(1_000_004)
    ups = sum(run.moves_for(0.3, {"buy": 0, "sell": 1}, rng) for _ in range(20_000))
    assert abs(ups / 20_000 - 0.3) < 0.01


def test_the_impact_rng_is_seeded_from_the_day_seed():
    a = [run.impact_rng(1_000_004).unit() for _ in range(1)]
    b = [run.impact_rng(1_000_004).unit() for _ in range(1)]
    c = [run.impact_rng(1_000_005).unit() for _ in range(1)]
    assert a == b and a != c
    expected = run.fnv1a64((1_000_004).to_bytes(8, "little") + b"impact-282")
    assert run.impact_rng(1_000_004).state == expected


def test_the_design_records_positive_kappa0_per_regime():
    values = DESIGN["impact_scale"]["values"]
    assert set(values) == {"calm", "stressed", "thin"}
    for regime in values.values():
        assert regime["kappa0_buy"] > 0 and regime["kappa0_sell"] > 0
        assert regime["kappa0"] == pytest.approx((regime["kappa0_buy"] + regime["kappa0_sell"]) / 2)


M4_05 = json.loads((HERE.parent / "experiments" / "m4_05" / "design.json").read_text())


def short_market(rows=400):
    regime = M4_05["simulator"]["regimes"][0]
    sim = M4_05["simulator"]
    episode = {"seed": 1_000_004, "reference_ticks": sim["reference_ticks"], "start_ms": 0,
               "sample_ms": sim["sample_ms"], "rows": rows}
    return run.ReactiveQrMarket(json.dumps(regime), json.dumps(M4_05["kernel"]),
                                json.dumps(episode))


def fixed_spread():
    grid = M4_05["arms"]["maskable_ppo"]["grid"]
    config = run.base.base_config(M4_05, "main", "mask", grid)

    def make_env(rows):
        return run.MaskableMarketMakingEnv(config, [tuple(r) for r in rows])

    return run.base.Constant([1, 1, 1, 0, 0]), make_env


def test_kappa_zero_is_the_non_reactive_episode():
    policy, make_env = fixed_spread()
    market = short_market()
    reactive = run.reactive_episode(policy, make_env, market, 0.0, run.impact_rng(1), 101, M4_05)
    env = make_env(short_market().rows())
    try:
        plain = run.base.episode(policy, env, 101, M4_05)
    finally:
        env.close()
    assert reactive["net"] == plain["net"] and reactive["steps"] == plain["steps"]
    assert reactive["attribution"] == plain["attribution"]
    assert reactive["reactions"] == 0 and reactive["replayed_steps"] == 0


def test_reactions_replay_to_the_same_step_and_change_the_day():
    policy, make_env = fixed_spread()
    calm = short_market()
    reacted = run.reactive_episode(policy, make_env, short_market(), 2.0, run.impact_rng(1), 101,
                                   M4_05)
    plain = run.reactive_episode(policy, make_env, calm, 0.0, run.impact_rng(1), 101, M4_05)
    assert reacted["reactions"] > 0 and reacted["moves"] >= reacted["reactions"]
    assert reacted["net"] != plain["net"]
    again = run.reactive_episode(policy, make_env, short_market(), 2.0, run.impact_rng(1), 101,
                                 M4_05)
    assert again["net"] == reacted["net"], "reproducible"


def test_the_native_replay_reaches_what_the_python_one_does():
    policy, make_env = fixed_spread()
    native = run.reactive_episode(policy, make_env, short_market(), 2.0, run.impact_rng(1), 101,
                                  M4_05)
    python = run.reactive_episode(policy, make_env, short_market(), 2.0, run.impact_rng(1), 101,
                                  M4_05, replay="python")
    assert native["reactions"] == python["reactions"] > 0
    assert native["moves"] == python["moves"]
    assert native["net"] == python["net"] and native["attribution"] == python["attribution"]
    assert native["positions"] == python["positions"]


def test_replacing_rows_that_change_what_was_read_is_refused_and_changes_nothing():
    policy, make_env = fixed_spread()
    market = short_market()
    rows = market.rows()
    env = make_env(rows)
    other = make_env(rows)
    try:
        observation, _ = env.reset(seed=101)
        other.reset(seed=101)
        for _ in range(30):
            observation, *_ = env.step([1, 1, 1, 0, 0])
            other.step([1, 1, 1, 0, 0])
        changed = [list(r) for r in rows]
        changed[3][1] += 2
        with pytest.raises(ValueError):
            env.replace_rows([tuple(r) for r in changed])
        assert env.bridge_info == other.bridge_info
        for _ in range(30):
            a = env.step([1, 1, 1, 0, 0])
            b = other.step([1, 1, 1, 0, 0])
            assert a[4]["net"] == b[4]["net"]
    finally:
        env.close()
        other.close()


def test_the_grid_is_108s_24_flow_points_times_kappa_0_and_2_kappa0():
    """Design version 4: kappa0 left the grid; no k1 point, job or key remains."""
    points = run.grid_points()
    assert len(points) == 48 and len({p[0] for p in points}) == 48
    assert {p[3] for p in points} == {0, 2}
    assert not any(p[0].endswith("-k1") for p in points)
    assert points[0][0] == f"{run.unit_flow()}-k0"


def test_each_day_has_108s_seed_and_rows():
    flows = dict(run.m503.points(run.M5_03_DESIGN))
    for flow in (run.unit_flow(), run.m503.point_name(run.M5_03_DESIGN["fixed_stress"])):
        for split in ("selection", "holdout"):
            records = run.m503.digests_of(f"{split}-{flow}")
            for (regime, day), record in list(records.items())[:3]:
                regime_dict, seed = run.regime_day(M4_05, flows[flow], split, regime, day)
                assert seed == record["seed"]
    regime_dict, seed = run.regime_day(M4_05, flows[run.unit_flow()], "selection", "thin", 2)
    market = run.day_market(regime_dict, M4_05, seed, 2)
    record = run.m503.digests_of(f"selection-{run.unit_flow()}")[("thin", 2)]
    assert market.digest() == record["dataset_digest"]


def test_the_identity_check_names_every_row_that_differs_or_is_missing():
    row = {"scenario": "selection-x", "arm": "glft", "seed": None, "regime": "calm", "day": 1,
           **{f: "1" for f in run.FIELDS_108}}
    theirs = {"rows": [dict(row)]}
    assert run.identity([row], theirs) == []
    changed = {**row, "max_drawdown": "2"}
    assert run.identity([changed], theirs) == [("selection-x", "glft", None, "calm", 1)]
    missing = {**row, "day": 2}
    assert run.identity([missing], theirs) == [("selection-x", "glft", None, "calm", 2)]
    extra = {**row, "reactions": 5, "emptied": "3"}
    assert run.identity([extra], theirs) == [], "only #108's fields are compared"


def test_job_keys_are_unique_across_the_selection():
    runs = {"maskable_ppo": [{"seed": s} for s in range(1, 6)],
            "residual_sac": [{"seed": s} for s in range(1, 6)]}
    jobs = run.jobs_for(run.grid_points(), "selection", [1, 2], runs, {})
    assert len(jobs) == 48 * 3 * 2 * 14
    assert len({run.job_key(j) for j in jobs}) == len(jobs)
    assert not any(j["kappa_index"] == 1 or "-k1" in j["point"] for j in jobs)
    assert not any("-k1" in str(run.job_key(j)) for j in jobs)


def test_the_ceiling_keeps_the_holdouts_reserve_and_stops_at_975(tmp_path):
    with pytest.raises(ValueError, match="no budget.json"):
        run.limit_for(tmp_path, 4, 9.0)
    (tmp_path / "budget.json").write_text(json.dumps({"spent_process_seconds": 51348}))
    assert run.limit_for(tmp_path, 4, 9.0) == pytest.approx(((97.5 - 9) * 3600 - 51348) / 4)
    run.charge(tmp_path, "x", 1000, 4)
    assert run.spent(tmp_path) == 55348
    (tmp_path / "budget.json").write_text(json.dumps({"spent_process_seconds": 97.5 * 3600}))
    with pytest.raises(run.Overrun):
        run.limit_for(tmp_path, 4, 0.0)


def test_the_holdout_marks_its_attempt_before_any_job_and_runs_once(tmp_path, monkeypatch):
    (tmp_path / "selection.json").write_text("{}")
    (tmp_path / "budget.json").write_text(json.dumps({"spent_process_seconds": 0}))
    frozen = {"design_sha256": "d", "code_sha256": "c", "chosen": {},
              "selection_sha256": run.sha256(tmp_path / "selection.json")}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(run.m503, "check_frozen_digests", lambda *a: None)
    monkeypatch.setattr(run, "jobs_for", lambda *a, **k: [])
    seen = []

    def pool(*a, **k):
        seen.append((tmp_path / "holdout.started").exists())
        raise RuntimeError("a day failed")

    monkeypatch.setattr(run, "pool", pool)

    class Study:
        out, parallel, runs = tmp_path, 4, {}

        def header(self, step):
            return {"design_sha256": "d", "code_sha256": "c"}

        def common(self):
            return {}

    sha = run.sha256(tmp_path / "frozen.json")
    with pytest.raises(RuntimeError):
        run.holdout(Study(), sha)
    assert seen == [True]
    with pytest.raises(ValueError, match="attempted already"):
        run.holdout(Study(), sha)
