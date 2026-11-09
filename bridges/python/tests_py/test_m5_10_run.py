"""The job lists and guards of #108 part 2; nothing is evaluated."""

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
import run as r  # noqa: E402


def selected():
    return {r.policy_key(p): r.grid.points(i)[3] for s, i in r.SETTINGS.items() for p in r.policies(s)}


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
    assert out["point"] == r.point_id(p[0]) and out["selection_loss"] == 2


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
