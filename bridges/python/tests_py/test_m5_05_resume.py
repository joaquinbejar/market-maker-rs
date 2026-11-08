"""#281's one resume of a holdout killed from outside: its guards, its marker and the journal."""

import importlib.util
import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m5_05"
SPEC = importlib.util.spec_from_file_location("m5_05_resume", STUDY / "resume.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)
run = resume.run


def line(i):
    return json.dumps({"key": json.dumps(["c", "baseline", 1, "calm", i]), "row": {"net": "1"}})


def test_a_torn_last_line_is_dropped_and_complete_ones_kept(tmp_path):
    path = tmp_path / "j"
    path.write_text(line(1) + "\n" + line(2) + "\n")
    assert resume.repair_journal(path) == (2, False)
    path.write_text(line(1) + "\n" + line(2) + "\n" + line(3)[:20])
    assert resume.repair_journal(path) == (2, True)
    assert path.read_text() == line(1) + "\n" + line(2) + "\n"
    path.write_text(line(1) + "\n" + "{broken\n" + line(2) + "\n")
    with pytest.raises(ValueError):
        resume.repair_journal(path)


class Ctx:
    parallel = 1

    def header(self, step, commit):
        return {"design_sha256": "d", "code_sha256": "c"}

    def write(self, out, name, report):
        (pathlib.Path(out) / f"{name}.json").write_text(json.dumps(report))


def study(tmp_path, monkeypatch, started=True):
    monkeypatch.setattr(run, "FROZEN_MODELS", tmp_path / "frozen_models.json")
    (tmp_path / "frozen_models.json").write_text("{}")
    (tmp_path / "selection.json").write_text("{}")
    frozen = {"design_sha256": "d", "code_sha256": "c", "impact": None, "chosen": {},
              "selection_sha256": resume.sha256(tmp_path / "selection.json"),
              "frozen_models_sha256": resume.sha256(tmp_path / "frozen_models.json")}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    sha = resume.sha256(tmp_path / "frozen.json")
    (tmp_path / "estimates.json").write_text(json.dumps({"holdout": 100}))
    (tmp_path / "holdout.journal").write_text(line(1) + "\n" + line(2)[:10])
    if started:
        (tmp_path / "holdout.started").write_text(json.dumps(
            {"started": os.stat(tmp_path / "holdout.journal").st_mtime - 30, "pid": 1,
             "frozen_sha256": sha}) + "\n")
    monkeypatch.setattr(run.m503, "check_frozen_digests", lambda *a: None)
    monkeypatch.setattr(run, "impact_provenance", lambda ctx: None)
    monkeypatch.setattr(run, "check_models", lambda ctx, out: {})
    monkeypatch.setattr(run, "holdout_jobs", lambda *a: ["job"])
    return sha


def test_the_resume_runs_once_after_a_first_attempt_and_marks_itself_first(tmp_path, monkeypatch):
    sha = study(tmp_path, monkeypatch)
    seen = []

    def pool(jobs, worker, parallel, label, journal, limit):
        marker = json.loads((tmp_path / "holdout.resumed").read_text())
        seen.append((marker, limit, (tmp_path / "holdout.journal").read_text()))
        return [{"net": "1"}]

    monkeypatch.setattr(run, "pool", pool)
    report = resume.resume(Ctx(), tmp_path, "c", sha)
    marker, limit, journal = seen[0]
    assert marker["first_attempt"]["frozen_sha256"] == sha
    assert marker["journal_before"]["lines"] == 1 and marker["torn_line_dropped"]
    assert marker["environment"]["extension_sha256"]
    # The overrun rule spans both attempts: 150 s less the first attempt's 30 s.
    assert limit == pytest.approx(120, abs=1)
    assert journal == line(1) + "\n"
    assert report["resumed"]["journal_before"]["lines"] == 1
    assert report["resumed"]["first_lines_unchanged"]
    (tmp_path / "holdout.json").unlink()
    with pytest.raises(ValueError, match="resumed already"):
        resume.resume(Ctx(), tmp_path, "c", sha)
    assert len(seen) == 1


def test_the_resume_refuses_without_a_first_attempt_after_a_finish_or_on_change(tmp_path,
                                                                               monkeypatch):
    sha = study(tmp_path, monkeypatch, started=False)
    monkeypatch.setattr(run, "pool", lambda *a: pytest.fail("a job ran"))
    with pytest.raises(ValueError, match="no first attempt"):
        resume.resume(Ctx(), tmp_path, "c", sha)
    (tmp_path / "holdout.started").write_text(json.dumps(
        {"started": 0, "pid": 1, "frozen_sha256": "other"}))
    with pytest.raises(ValueError, match="another manifest"):
        resume.resume(Ctx(), tmp_path, "c", sha)
    with pytest.raises(ValueError, match="not the one given"):
        resume.resume(Ctx(), tmp_path, "c", "0" * 64)
    (tmp_path / "holdout.started").write_text(json.dumps(
        {"started": 0, "pid": 1, "frozen_sha256": sha}))
    (tmp_path / "frozen_models.json").write_text("{\"x\": 1}")
    with pytest.raises(ValueError, match="frozen_models.json changed"):
        resume.resume(Ctx(), tmp_path, "c", sha)
    (tmp_path / "holdout.json").write_text("{}")
    with pytest.raises(ValueError, match="finished"):
        resume.resume(Ctx(), tmp_path, "c", sha)
    assert not (tmp_path / "holdout.resumed").exists()


def test_run_py_is_the_frozen_code():
    assert resume.sha256(STUDY / "run.py") == \
        "82afd334a00905f120597a6329df1e25e6a01b7f5fb7ead1017f337eabe1878a"
    assert "run" in sys.modules


def test_a_run_cut_mid_way_and_resumed_equals_an_uninterrupted_one(tmp_path):
    """Real spawned pools: the first is stopped part way (its workers killed), the resume runs
    with another process count, and the rows equal an uninterrupted run's byte for byte; the
    first attempt's journal lines are untouched."""
    import m5_05_workers

    jobs = [{"condition": c, "arm": a, "seed": s, "regime": "calm", "day": d}
            for c in ("unperturbed", "held_out:p") for a in ("adversarial", "baseline")
            for s in (1, 2) for d in range(1, 11)]
    whole = run.pool(jobs, m5_05_workers.deterministic, 2, "whole", tmp_path / "whole.journal",
                     600)
    cut = tmp_path / "cut.journal"
    with pytest.raises(run.Overrun):
        run.pool(jobs, m5_05_workers.deterministic, 1, "first", cut, 1.5)
    complete, _ = resume.repair_journal(cut)
    assert 0 < complete < len(jobs)
    size, digest = cut.stat().st_size, resume.sha256(cut)
    resumed = run.pool(jobs, m5_05_workers.deterministic, 3, "resume", cut, 600)
    assert json.dumps(resumed, sort_keys=True) == json.dumps(whole, sort_keys=True)
    assert resume.prefix_sha256(cut, size) == digest
