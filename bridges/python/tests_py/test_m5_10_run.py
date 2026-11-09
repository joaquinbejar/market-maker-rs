"""The job lists and guards of #108 part 2; nothing is evaluated."""

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
import run as r  # noqa: E402


def selected():
    return {r.policy_key(p): {"point": r.grid.points(i)[3]} for s, i in r.SETTINGS.items() for p in r.policies(s)}


def test_counts_match_the_design():
    assert len(r.policies("impact_0")) == 13
    assert len(r.selection_jobs()) == 9360
    assert len(r.holdout_jobs(selected())) == 4680


def test_keys_are_unique():
    jobs = r.selection_jobs() + r.holdout_jobs(selected())
    assert len({r.job_key(j) for j in jobs}) == len(jobs)


def test_check_classical_hash_and_missing_arm(tmp_path):
    good = {s: {a: {} for a in r.CLASSICAL} for s in r.SETTINGS}
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"selected": good}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert r.check_classical(path, digest) == good
    with pytest.raises(SystemExit, match="not the file"):
        r.check_classical(path, "0" * 64)
    del good["impact_low"]["glft"]
    path.write_text(json.dumps({"selected": good}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="no glft"):
        r.check_classical(path, digest)


def test_journal_resumes_without_rerunning(tmp_path):
    jobs = r.selection_jobs()[:3]
    calls = []
    def ev(job):
        calls.append(r.job_key(job))
        return {"net": 1}
    r.run_journaled(jobs[:2], tmp_path / "j", ev)
    r.run_journaled(jobs, tmp_path / "j", ev)
    assert len(calls) == 3


def test_select_averages_seeds_and_breaks_ties_first():
    p = r.grid.points(0.0)
    policy = lambda seed: {"arm": "mlp", "setting": "impact_0", "seed": seed}
    rows = [({"policy": policy(s), "point": p[i], "day": 1}, {"net": n})
            for i, s, n in ((0, 1, -1), (0, 2, -3), (1, 1, -2), (1, 2, -2), (2, 1, 0), (2, 2, 0))]
    out = r.select(rows)["impact_0/mlp"]
    assert out["point"] == p[0] and out["selection_loss"] == 2


def test_claim_recovery_rules(tmp_path):
    r.claim(tmp_path / "m", "a" * 64)
    with pytest.raises(SystemExit, match="already started"):
        r.claim(tmp_path / "m", "a" * 64)
    journal = tmp_path / "j"
    journal.write_text("x")
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(SystemExit, match="prefix"):
        r.claim(tmp_path / "m", "a" * 64, journal, "killed", "0" * 64)
    r.claim(tmp_path / "m", "a" * 64, journal, "killed", digest)
    with pytest.raises(SystemExit, match="too many"):
        r.claim(tmp_path / "m", "a" * 64, journal, "again", digest)


def test_day_configs_one_per_point_and_split(tmp_path):
    causal = json.loads((pathlib.Path(r.HERE).parent / "m5_02" / "design.json").read_text())["causal"]
    written = r.write_day_configs(causal, tmp_path)
    assert len(written) == 2 * 3 * 24
    sample = json.loads(written[0].read_text())["causal"]
    assert sample["seed_offset"] == 100000 and sample["days"] == 10


def test_pilot_covers_every_arm_at_both_points():
    jobs = r.pilot_jobs()
    assert len(jobs) == 2 * 13 * 2
    assert {j["policy"]["arm"] for j in jobs} == {"no_trade", "avellaneda_stoikov", "glft", "mlp", "lstm"}
    assert r.point_id(jobs[-1]["point"]) == "m2-l0.7-b1.2-i2.0"


def test_project_uses_each_arm_slowest_point():
    rows = [(j, 20.0 if j["point"]["market"] == 2 else 10.0) for j in r.pilot_jobs()]
    out = r.project(rows)
    assert out["projected_hours"] == pytest.approx(20.0 * 13 * 360 * 3 / 3600)
    assert out["over_ceiling"] is False
    with_days = r.project(rows, 36.0)
    assert with_days["generation_hours"] == pytest.approx(14.4)
    assert with_days["projected_hours"] == pytest.approx(out["projected_hours"] + 14.4)


def test_spectral_check_records_and_stops():
    weights = [[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]]
    points = [r.grid.fixed_stress(1.0)]
    assert r.spectral_check(weights, points)["m2-l0.7-b1.2-i2.0"] == pytest.approx(0.12)
    with pytest.raises(SystemExit, match="subcritical"):
        r.spectral_check([[[0, 0, 0], [0, 0, 0], [0, 0, 1.0]]], points)


def test_pilot_refuses_an_existing_output(tmp_path):
    (tmp_path / "pilot.jsonl").write_text("")
    (tmp_path / "frozen.json").write_text(json.dumps({"classical_sha256": ""}))
    good = {s: {a: {} for a in r.CLASSICAL} for s in r.SETTINGS}
    (tmp_path / "classical.json").write_text(json.dumps({"selected": good}))
    digest = hashlib.sha256((tmp_path / "classical.json").read_bytes()).hexdigest()
    (tmp_path / "frozen.json").write_text(json.dumps({"classical_sha256": digest}))
    with pytest.raises(SystemExit, match="already exists"):
        r.pilot_step([str(tmp_path), "unused", str(tmp_path)])


def test_select_tie_uses_grid_order_whatever_the_row_order():
    p = r.grid.points(0.0)
    policy = {"arm": "glft", "setting": "impact_0", "seed": None}
    rows = [({"policy": policy, "point": p[i], "day": 1}, {"net": -1}) for i in (5, 2, 7)]
    assert r.select(rows)["impact_0/glft"]["point"] == p[2]


def test_removed_named_points_leave_no_substitute():
    impact = 0.25
    adv = r.grid.points(impact)[3]
    fixed = r.point_id(r.grid.fixed_stress(impact))
    assert "fixed_stress" not in r.condition_points(impact, adv, {fixed})
    assert r.condition_points(impact, adv, {r.point_id(r.grid.points(impact)[0])}) == {}


def test_torn_last_line_is_dropped_and_the_job_reruns(tmp_path):
    journal = tmp_path / "j"
    journal.write_text('{"key": "a", "result": 1}\n{"key": "b", "res')
    assert list(r.load_journal(journal)) == ["a"]
    assert journal.read_text() == '{"key": "a", "result": 1}\n'


def test_a_torn_middle_line_is_an_error(tmp_path):
    journal = tmp_path / "j"
    journal.write_text('{"key": "a"\n{"key": "b"}\n')
    with pytest.raises(json.JSONDecodeError):
        r.load_journal(journal)


def test_freeze_is_single_and_hashes_inputs(tmp_path):
    for name in ("selection.json", "d", "i", "p", "design"):
        (tmp_path / name).write_text("{}")
    digest = r.freeze(tmp_path, tmp_path / "design", tmp_path / "d", tmp_path / "i", [tmp_path / "p"])
    assert digest == r.sha256(tmp_path / "frozen.json")
    with pytest.raises(SystemExit, match="already"):
        r.freeze(tmp_path, tmp_path / "design", tmp_path / "d", tmp_path / "i", [])


def test_holdout_refuses_another_freeze(tmp_path):
    (tmp_path / "frozen.json").write_text("{}")
    with pytest.raises(SystemExit, match="not the freeze"):
        r.holdout_step(tmp_path, tmp_path, tmp_path, tmp_path, "0" * 64, tmp_path / "m", 1)
    assert not (tmp_path / "m").exists()


def test_point_statistics_on_a_toy_day(tmp_path):
    point = r.grid.points(0.0)[0]
    path = r.day_path(tmp_path, "selection", "impact_0", point, 1)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([[0, 10, 1, 11, 1], [1, 11, 1, 12, 1], [2, 11, 1, 12, 1]]))
    path.with_name("day-1.flow.json").write_text(json.dumps([[1, 0], [0, 0], [0, 0]]))
    path.with_name("day-1.queues.json").write_text(json.dumps([[0, 20], [5, 5], [5, 5]]))
    st = r.point_statistics(tmp_path, "selection", "impact_0", point, 1)
    assert st["empty_share"] == pytest.approx(1 / 6) and st["cap_share"] == pytest.approx(1 / 6)
    assert st["market_rate"] == pytest.approx(1 / 6) and st["slope"] == pytest.approx(1)


def test_holdout_jobs_drop_removed_conditions_per_setting():
    fixed = r.point_id(r.grid.fixed_stress(0.0))
    jobs = r.holdout_jobs(selected(), removed={"impact_0": {fixed}})
    assert len(jobs) == 4680 - 13 * 30
    assert not any(j["condition"] == "fixed_stress" and j["policy"]["setting"] == "impact_0" for j in jobs)
