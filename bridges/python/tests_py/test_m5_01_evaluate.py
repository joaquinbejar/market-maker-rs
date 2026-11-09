"""#106's S1 rule and starting state, on made-up values only (SYNTHETIC)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"))
import evaluate  # noqa: E402


def observation(ms, depth, kind, side="bid"):
    return {"timestamp_ns": ms * 1_000_000, "side": side, "depth": depth, "kind": kind}


def run(failure=None, cap_share=0.0):
    return {"failure": failure, "cap_share": cap_share}


def all_runs(**changed):
    runs = {f"{s}-{w}": run() for s in evaluate.SEEDS for w in evaluate.VALIDATION}
    runs.update(changed)
    return runs


def test_all_clean_runs_pass():
    assert evaluate.decide(all_runs())["decision"] == "pass"


@pytest.mark.parametrize("bad", [run("nan: x"), run("budget: x"), run(cap_share=0.011)])
def test_one_failure_in_one_seed_discards(bad):
    result = evaluate.decide(all_runs(**{"3-2": bad}))
    assert result["decision"] == "discard MDQR" and list(result["failures"]) == ["3-2"]


def test_exactly_the_limit_passes():
    assert evaluate.s1_failed(run(cap_share=0.01)) is None


def test_a_missing_run_is_refused():
    runs = all_runs()
    del runs["5-1"]
    with pytest.raises(ValueError):
        evaluate.decide(runs)


def test_a_reference_move_resets_earlier_events_only():
    kind = evaluate.mdqr.KINDS[0]
    window = {"history": {
        "last_sample": {"bid": [1, 2, 3, 9], "ask": [4, 5, 6, 9]},
        "observations": [observation(1000, 0, kind), observation(3000, 1, kind),
                         observation(1000, 7, kind)],
        "references": [{"timestamp_ms": 2000, "moved": True}],
        "trades": [{"timestamp_ns": 2_000_000_000, "size": "0.5", "side": "sell"},
                   {"timestamp_ns": 1_000_000_000, "size": "1", "side": "buy"}]}}
    start, events, trades = evaluate.start_of(window)
    assert start == ([1, 2, 3], [4, 5, 6])
    assert events == [(2.0, 0, 0, evaluate.mdqr.NO_KIND), (3.0, 0, 1, 0)]
    assert trades == [(1.0, 1.0), (2.0, -0.5)]


def test_the_holdout_is_claimed_once(tmp_path):
    evaluate.claim_holdout(tmp_path)
    with pytest.raises(FileExistsError):
        evaluate.claim_holdout(tmp_path)


def test_draws_refuse_an_unclaimed_holdout(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.run_draws(tmp_path, tmp_path, tmp_path, "holdout", 1)
