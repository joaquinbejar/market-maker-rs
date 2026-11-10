"""Part 2's guards on toy inputs; nothing is evaluated."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "experiments" / "m5_10"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "m5_10_part2", pathlib.Path(__file__).parents[1] / "experiments" / "m5_10" / "part2.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


def test_attempt_two_only_after_attempt_one(tmp_path):
    with pytest.raises(SystemExit, match="attempt 1"):
        p2.claim(tmp_path, {"x": 1})
    p2.marker_path(tmp_path, 1).write_text("")
    marker = p2.claim(tmp_path, {"x": 1})
    lines = marker.read_text().splitlines()
    assert '"previous"' in lines[0] and '"attempt": 2' in lines[1]
    with pytest.raises(SystemExit, match="not rerun"):
        p2.claim(tmp_path, {"x": 1})


def test_runner_commit_is_checked():
    """Needs the runner's merge commit in the clone; CI's shallow checkout lacks it."""
    import subprocess

    present = subprocess.run(["git", "-C", str(p2.HERE), "cat-file", "-e", f"{p2.RUNNER_MERGE_COMMIT}^{{commit}}"],
                             capture_output=True).returncode == 0
    if not present:
        pytest.skip("the runner's merge commit is not in this shallow clone")
    p2.check_runner_commit(p2.HERE)


def test_limit_is_within_the_budget():
    assert p2.limit_seconds(4) == pytest.approx(6.0 * 3600 / 4)


def test_jobs_are_a_and_b_only_at_impact():
    class M:
        TRAINED = ("adversarial", "baseline")

        @staticmethod
        def unperturbed_name():
            return "u"

        @staticmethod
        def policy_jobs(ctx, out, condition, split, name, arms, frozen_models, extra):
            return [{"condition": condition, "arms": arms, "split": split, **extra("calm")}]
    out = p2.jobs(M, None, None, {}, {"calm": 0.5})
    assert out == [{"condition": "impact", "arms": ("adversarial", "baseline"), "split": "holdout",
                    "impact": True, "kappa": 0.5}]


def test_a_spawn_worker_of_the_script_resolves_281s_runner():
    """part2.py started as a script, as the run starts it, so pytest's path cannot hide the bug."""
    import json
    import subprocess

    out = subprocess.run([sys.executable, str(p2.HERE / "part2.py"), "--selftest"], check=True,
                         capture_output=True, text=True, cwd=str(p2.HERE.parents[1])).stdout
    result = json.loads(out.strip().splitlines()[-1])
    assert result["worker_unperturbed_name"] == result["parent"]
