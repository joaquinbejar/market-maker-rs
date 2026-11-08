"""#281's report on hand-made holdout rows: the stratified resampling, the CVaR difference, the
signs, the missing seed-days and the claim's rule. All numbers are made up."""

import importlib.util
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m5_05_report", HERE.parent / "experiments" / "m5_05" / "report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

DAYS = [(r, d) for r in report.REGIMES for d in range(1, 11)]


def rows_for(condition, arm, nets, seeds=(1, 2, 3, 4, 5)):
    """nets: a function of (seed, day index) giving the net, or None for a missing day."""
    out = []
    for seed in seeds:
        for i, (regime, day) in enumerate(DAYS):
            net = nets(seed, i)
            out.append({"condition": condition, "arm": arm, "seed": seed, "regime": regime,
                        "day": day, "net": None if net is None else str(net)})
    return out


def test_the_resampling_stays_within_each_regime_and_wraps_there():
    indices = report.stratified_indices(DAYS, replicates=300)
    assert indices.shape == (300, 30)
    for regime_index in range(3):
        block = indices[:, 10 * regime_index:10 * (regime_index + 1)]
        assert block.min() >= 10 * regime_index and block.max() < 10 * (regime_index + 1)
    # Blocks of mean length 2 continue to the next day, wrapping from the last to the first.
    steps = np.diff(indices[:, :10], axis=1)
    assert np.mean((steps == 1) | (steps == -9)) == pytest.approx(0.5 + 0.5 / 10, abs=0.03)
    assert np.array_equal(indices, report.stratified_indices(DAYS, replicates=300))


def test_the_tail_difference_is_a_difference_of_cvars_not_a_cvar_of_differences():
    # A loses 10 on day 0 and B loses 10 on day 1: each arm's CVaR5 (worst 2 of 30) is 5, so
    # the difference is 0; the CVaR of the pointwise differences would be 5.
    a = [[-10.0 if i == 0 else 0.0 for i in range(30)]]
    b = [[-10.0 if i == 1 else 0.0 for i in range(30)]]
    every = np.arange(30)[None, :]
    d = report.difference("cvar_averaged", a, b, every)
    assert d["point"] == pytest.approx(0.0)
    pointwise = [x - y for x, y in zip(a[0], b[0], strict=True)]
    assert report.cvar_loss(pointwise) == pytest.approx(5.0)


def test_seed_averaged_and_per_seed_tails_differ_and_are_both_kept():
    # Seed 1 loses 20 on day 0, seed 2 on day 1: the averaged arm loses 10 on each.
    m = [[-20.0 if i == 0 else 0.0 for i in range(30)],
         [-20.0 if i == 1 else 0.0 for i in range(30)]]
    every = np.arange(30)
    assert report.cvar_averaged(m, every) == pytest.approx(10.0)
    assert report.cvar_per_seed(m, every) == pytest.approx(10.0)
    m[1] = [-20.0 if i == 0 else 0.0 for i in range(30)]
    assert report.cvar_averaged(m, every) == pytest.approx(10.0)
    m[1] = [0.0] * 30
    assert report.cvar_averaged(m, every) == pytest.approx(5.0)
    assert report.cvar_per_seed(m, every) == pytest.approx(5.0)


def study(a_tail, b_tail, a_net=1.0, b_net=1.0, missing=None, impact=True):
    """Holdout rows: every arm flat at its net except a tail loss on the first two days of each
    regime at the selected points."""
    rows = []
    for arm, net in (("adversarial", a_net), ("baseline", b_net), ("reference", 0.0)):
        rows += rows_for("unperturbed", arm, lambda s, i, n=net: n)
        for c in ("held_out:p1", "held_out:p2", "execution_stress", *(["impact"] * impact)):
            tail = {"adversarial": a_tail, "baseline": b_tail, "reference": 3.0}[arm]

            def nets(s, i, tail=tail, c=c, arm=arm):
                if missing == (arm, c, s, i):
                    return None
                return -tail if i % 10 < 2 and c.startswith("held_out") else 0.5
            rows += rows_for(c, arm, nets)
    return rows


CHOSEN = {"adversarial": {"point": "p1"}, "baseline": {"point": "p2"},
          "reference": {"point": "p1"}}


def analyse(rows, chosen=CHOSEN, impact_run=True):
    return report.analyse(rows, chosen, impact_run, names=["p1", "p2"], in_sample_names=[])


def test_signs_and_the_claim():
    result = analyse(study(a_tail=2.0, b_tail=8.0, a_net=0.5, b_net=1.0))
    primary = result["primary"]
    # A's tail loss is smaller: negative, an improvement; A earns less: negative, a cost.
    assert primary["tail_at_p_A"]["cvar_averaged"]["point"] == pytest.approx(-6.0)
    assert primary["tail_at_p_B"]["cvar_averaged"]["point"] == pytest.approx(-6.0)
    assert primary["envelope"]["cvar_averaged"]["point"] == pytest.approx(-6.0)
    assert primary["cost"]["point"] == pytest.approx(-0.5)
    assert result["claim"] == "gain"
    assert result["conditions_evaluated"] == 4
    assert result["count_below_zero"] == 2
    assert result["count_excluding_zero"] == 2
    worse = analyse(study(a_tail=8.0, b_tail=2.0))
    assert worse["primary"]["tail_at_p_A"]["cvar_averaged"]["point"] == pytest.approx(6.0)
    assert worse["claim"] == "no gain shown"
    # Intervals wholly above 0 exclude 0 too: the design counts both directions.
    assert worse["count_below_zero"] == 0 and worse["count_excluding_zero"] == 2


def test_the_denominator_is_the_conditions_actually_evaluated():
    result = analyse(study(2.0, 8.0, impact=False), impact_run=False)
    assert result["conditions_evaluated"] == 3
    assert "impact" not in result["secondary"]
    assert "3 held-out conditions evaluated" in report.render(result).replace("of the ", "")
    text = report.render(result, holdout={"impact_run": False, "bounds": "B",
                                          "resumed": {"journal_before": {"lines": 7, "sha256": "h"},
                                                      "resume_sha256": "r"}})
    assert "declared and not run" in text and "resumed once" in text and "7 jobs" in text


def test_a_missing_seed_day_fails_its_estimators_and_the_claim():
    rows = study(2.0, 8.0, missing=("baseline", "held_out:p2", 3, 4))
    result = analyse(rows)
    primary = result["primary"]
    assert "failed" in primary["tail_at_p_B"]["cvar_averaged"]
    assert "failed" in primary["tail_at_p_B"]["cvar_per_seed"]
    assert "failed" in primary["envelope"]["cvar_averaged"]
    # The other tail still stands, but the claim needs both.
    assert "failed" not in primary["tail_at_p_A"]["cvar_averaged"]
    assert result["claim"] == "unavailable"
    entry = result["table"][("held_out:p2", "baseline")]
    assert entry["missing"] == 1 and entry["cvar_averaged"] is None
    text = report.render(result)
    assert "failed (a seed-day is missing)" in text
    # A failed selection leaves no point, and the claim unavailable.
    result = analyse(study(2.0, 8.0), chosen={**CHOSEN, "baseline": {"point": None}})
    assert result["claim"] == "unavailable"


def test_a_seeds_own_tail_can_exceed_the_averaged_arms():
    # Seed 1 loses 20 on days 0 and 1, seed 2 on days 2 and 3: the averaged arm loses 10 on
    # four days, while each deployed seed has a tail of 20.
    m = [[-20.0 if i in (0, 1) else 0.0 for i in range(30)],
         [-20.0 if i in (2, 3) else 0.0 for i in range(30)]]
    every = np.arange(30)
    assert report.cvar_averaged(m, every) == pytest.approx(10.0)
    assert report.cvar_per_seed(m, every) == pytest.approx(20.0)


def test_the_replicates_pair_both_arms_on_the_same_days():
    # A is B less 1 every day, B varying by day: every paired replicate's difference is -1.
    b = [[float(i % 7) for i in range(30)]]
    a = [[x - 1.0 for x in b[0]]]
    indices = report.stratified_indices(DAYS, replicates=200)
    d = report.difference("mean_net", a, b, indices)
    assert d == {"point": pytest.approx(-1.0), "low": pytest.approx(-1.0),
                 "high": pytest.approx(-1.0)}


def test_the_claim_needs_an_improvement_at_both_selected_points():
    rows = [r for r in study(a_tail=2.0, b_tail=8.0)
            if not (r["condition"] == "held_out:p2" and r["arm"] == "adversarial")]

    def worse_at_p2(seed, i):
        return -9.0 if i % 10 < 2 else 0.5

    rows += rows_for("held_out:p2", "adversarial", worse_at_p2)
    result = analyse(rows)
    assert result["primary"]["tail_at_p_A"]["cvar_averaged"]["high"] < 0
    assert result["primary"]["tail_at_p_B"]["cvar_averaged"]["low"] > 0
    assert result["claim"] == "no gain shown"


def test_an_absent_seed_or_day_is_missing_not_a_smaller_sample():
    rows = study(2.0, 8.0)
    no_seed = [r for r in rows if not (r["arm"] == "baseline" and r["seed"] == 5
                                       and r["condition"] == "held_out:p2")]
    result = analyse(no_seed)
    assert "failed" in result["primary"]["tail_at_p_B"]["cvar_averaged"]
    assert result["claim"] == "unavailable"
    assert result["table"][("held_out:p2", "baseline")]["missing"] == 30
    no_day = [r for r in rows if not (r["condition"] == "held_out:p1" and r["regime"] == "thin"
                                      and r["day"] == 10)]
    result = analyse(no_day)
    assert "failed" in result["primary"]["tail_at_p_A"]["cvar_averaged"]
    assert result["claim"] == "unavailable"
    # A condition with no row at all is missing whole, never left out.
    no_condition = [r for r in rows if r["condition"] != "execution_stress"]
    result = analyse(no_condition)
    assert "failed" in result["secondary"]["execution_stress"]["cvar_averaged"]
    assert result["conditions_evaluated"] == 4
    assert result["table"][("execution_stress", "adversarial")]["missing"] == 150


def test_rows_outside_the_grid_or_twice_are_refused():
    rows = study(2.0, 8.0)
    with pytest.raises(ValueError, match="two rows"):
        analyse(rows + [dict(rows[0])])
    with pytest.raises(ValueError, match="outside"):
        analyse(rows + [{**rows[0], "seed": 6}])
    with pytest.raises(ValueError, match="outside"):
        analyse(rows + [{**rows[0], "day": 11}])


def test_the_registered_conditions_are_the_runners():
    spec = importlib.util.spec_from_file_location(
        "m5_05_run_names", HERE.parent / "experiments" / "m5_05" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    _, _, m5_03 = run.load()
    assert report.grid_names(held_out=True) == [n for n, _ in run.held_out_points(m5_03)]
    assert report.grid_names(held_out=False) == [
        n for n, _ in run.training_points(m5_03) if n != run.unperturbed_name()]
    assert len(report.held_out_conditions(True)) == 14
    assert len(report.held_out_conditions(False)) == 13
