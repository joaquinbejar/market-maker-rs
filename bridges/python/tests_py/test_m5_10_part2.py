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


def test_one_attempt_only(tmp_path):
    p2.claim(tmp_path, {"x": 1})
    with pytest.raises(SystemExit, match="not rerun"):
        p2.claim(tmp_path, {"x": 1})


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
