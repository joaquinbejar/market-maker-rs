"""The behaviours and the grid of #104, on hand-made masks."""

import importlib.util
import itertools
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent / "experiments" / "m4_07"
SPEC = importlib.util.spec_from_file_location("m4_07_run", STUDY / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

GRID = run.Grid({"bid_offsets": [0, 1, 2, 4, 8], "ask_offsets": [0, 1, 2, 4, 8], "sizes": [1, 2]})


def mask(quote=True, bid=(1, 1, 1, 1, 1), ask=(1, 1, 1, 1, 1), sizes=(1, 1)):
    return np.asarray([1, int(quote), *bid, *ask, *sizes, *sizes], dtype=bool)


def every_action():
    yield run.HOLD
    for b, a, bs, asz in itertools.product(range(5), range(5), range(2), range(2)):
        yield (1, b, a, bs, asz)


def test_codes_are_one_to_one_and_match_the_estimators_decoding():
    codes = [GRID.code(action) for action in every_action()]
    assert sorted(codes) == list(range(101))
    for action in every_action():
        decoded = run.est.decode(GRID.code(action), GRID.offsets, GRID.sizes)
        if action == run.HOLD:
            assert decoded == (0, 0, 0, 0, 0)
        else:
            assert decoded == (1, GRID.offsets[action[1]], GRID.offsets[action[2]],
                               GRID.sizes[action[3]], GRID.sizes[action[4]])
    # Every hold is the same action.
    assert GRID.canonical([0, 3, 2, 1, 0]) == run.HOLD


@pytest.mark.parametrize(
    "m",
    [mask(), mask(bid=(0, 1, 1, 0, 0)), mask(quote=False), mask(ask=(1, 0, 0, 0, 0), sizes=(1, 0))],
)
def test_the_behaviours_probabilities_sum_to_one_and_match_its_draws(m):
    greedy = (1, 1, 0, 0, 0) if m[1] else run.HOLD
    behaviour = run.Behaviour(GRID, lambda o, r, mk: greedy, 0.3, seed=4)
    total = sum(behaviour.probability(m, greedy, action) for action in every_action())
    assert total == pytest.approx(1.0)
    # The greedy action takes 0.7 plus its share of the exploration.
    assert behaviour.probability(m, greedy, greedy) == pytest.approx(
        0.7 + 0.3 * behaviour.uniform(m, greedy)
    )
    counts = {}
    draws = 20_000
    for _ in range(draws):
        action, _, propensity = behaviour.act(None, None, m)
        counts[action] = counts.get(action, 0) + 1
        assert propensity == pytest.approx(behaviour.probability(m, greedy, action))
        assert propensity > 0
    # The draws follow the stated probabilities, the greedy action's and the hold's.
    for action in {greedy, run.HOLD}:
        assert counts.get(action, 0) / draws == pytest.approx(
            behaviour.probability(m, greedy, action), abs=0.012
        )


def test_an_action_outside_the_mask_has_no_probability():
    m = mask(bid=(0, 1, 1, 1, 1))
    behaviour = run.Behaviour(GRID, lambda o, r, mk: run.HOLD, 0.3, seed=1)
    assert behaviour.probability(m, run.HOLD, (1, 0, 0, 0, 0)) == 0.0
    assert behaviour.probability(mask(quote=False), run.HOLD, (1, 1, 1, 0, 0)) == 0.0


def test_as_grid_skews_against_the_position_and_widens_with_time_left():
    point = {"risk_aversion": "0.01", "order_intensity": "50", "volatility": "37.7329",
             "horizon_ms": 86_400_000}
    rule = run.AsGrid(GRID, point, "0.005", 10_000, 8640)
    flat_bid, flat_ask = rule.distances(0.0, 8640)
    assert flat_bid == pytest.approx(flat_ask)
    long_bid, long_ask = rule.distances(2.0, 8640)
    assert long_bid > flat_bid > long_ask
    late_bid, _ = rule.distances(0.0, 10)
    assert late_bid < flat_bid
    # Nearest grid offsets, the smaller on a tie, 0 below 0.
    assert [GRID.nearest(d) for d in (-3.0, 0.4, 1.5, 3.0, 5.9, 6.0, 100.0)] == [0, 0, 1, 2, 3, 3, 4]
    raw = [None] * 11
    raw[run.POSITION], raw[run.STEPS_LEFT] = "0", "8640"
    action = rule(None, raw, mask())
    assert action[0] == 1 and action[1] == action[2] and action[3:] == (0, 0)
    assert rule(None, raw, mask(quote=False)) == run.HOLD


def test_a_holdout_refuses_training_logs_or_truth_changed_after_the_freeze(tmp_path):
    for name in ("logs-train.json", "logs-validation.json", "truth-train.json"):
        (tmp_path / name).write_text("{}")
    frozen = {"logs": {s: run.sha256(tmp_path / f"logs-{s}.json") for s in ("train", "validation")},
              "truth": {"train": run.sha256(tmp_path / "truth-train.json")}}
    run.check_frozen_inputs(tmp_path, frozen)
    (tmp_path / "logs-train.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="logs-train.json changed"):
        run.check_frozen_inputs(tmp_path, frozen)
    (tmp_path / "logs-train.json").write_text("{}")
    (tmp_path / "truth-train.json").write_text("[]")
    with pytest.raises(ValueError, match="truth-train.json changed"):
        run.check_frozen_inputs(tmp_path, frozen)
