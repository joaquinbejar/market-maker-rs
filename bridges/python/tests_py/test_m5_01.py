"""#106's stage S0: the MDQR book is the queue-reactive simulator's, its training rows are
the states the events saw, the fit is deterministic, and the stop rule reads as designed.
On a fixture of the Rust simulator (SYNTHETIC) and on made-up comparisons."""

import json
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "experiments" / "m5_01"
sys.path.insert(0, str(EXPERIMENT))
import mdqr  # noqa: E402
import s0 as run  # noqa: E402

FIXTURE = json.loads((HERE / "fixtures" / "m5_01_qr_60s.json").read_text())
DESIGN = json.loads((EXPERIMENT / "design.json").read_text())


def truth_mechanics():
    p = FIXTURE["parameters"]
    return mdqr.Mechanics(p["theta"], p["theta_reinit"], p["invariant"],
                          DESIGN["s0"]["truth"]["event_sizes"], p["max_queue"])


def test_with_the_truths_rates_the_book_is_the_rust_simulator_event_for_event():
    p = FIXTURE["parameters"]
    got = mdqr.simulate(truth_mechanics(), mdqr.table_rates(p["rates"], p["max_queue"]),
                        FIXTURE["seed"], FIXTURE["reference_ticks"], FIXTURE["window_seconds"],
                        FIXTURE["sample_ms"] / 1000.0, 10**6)
    want = FIXTURE["trajectory"]
    assert len(want["events"]) > 1000
    assert got["events"] == want["events"]
    assert got["samples"] == want["samples"]
    assert [[t, m, r] for t, m, r in got["references"]] == want["references"]
    assert any(m for _, m, _ in want["references"]), "the fixture has no reference move"


def test_the_training_rows_are_the_states_before_each_event():
    p = FIXTURE["parameters"]
    record = FIXTURE["events"]
    k = p["limits"]
    x, dt, cell, mask = mdqr.training_rows(record, k, p["max_queue"])
    events = record["events"]
    assert x.shape == (len(events), mdqr.feature_count(k))
    assert dt[0] == events[0][0] and np.all(dt > 0)
    for i, (t, side, depth, kind, bid, ask) in enumerate(events):
        assert cell[i] == side * k * 3 + depth * 3 + kind
        queues = [bid, ask]
        assert np.allclose(x[i, :2 * k], np.log1p([q for s in queues for q in s]))
        # The event happened, so the state allowed it; and the mask is the queue rule.
        assert mask[i, cell[i]]
        for s in (0, 1):
            for d in range(k):
                assert mask[i, s * k * 3 + d * 3 + 1] == (queues[s][d] > 0)
                assert mask[i, s * k * 3 + d * 3 + 0] == (queues[s][d] < p["max_queue"])


def test_the_network_sees_in_simulation_the_state_the_training_rows_hold():
    # Rebuild the rows from a simulated run's own record: the features the network reads
    # while simulating must be the ones training saw for the same events.
    p = FIXTURE["parameters"]
    seen = []

    class Spy:
        def rates(self, book):
            seen.append(mdqr.features(book.queues, book.spread_ticks(), book.last_kind,
                                      [[book.time - t for t in book.last[s]] for s in (0, 1)]))
            return mdqr.table_rates(p["rates"], p["max_queue"])(book)

    mdqr.simulate(truth_mechanics(), Spy().rates, FIXTURE["seed"], FIXTURE["reference_ticks"],
                  FIXTURE["window_seconds"], 1.0, 10**6)
    x, _, _, _ = mdqr.training_rows(FIXTURE["events"], p["limits"], p["max_queue"])
    assert np.allclose(np.asarray(seen[:len(x)]), x)


def test_the_fit_is_deterministic_and_improves_the_likelihood():
    p = FIXTURE["parameters"]
    rows = mdqr.training_rows(FIXTURE["events"], p["limits"], p["max_queue"])
    a, history = mdqr.fit(rows, p["limits"], p["max_queue"], 3, batch=256, epochs=3)
    b, again = mdqr.fit(rows, p["limits"], p["max_queue"], 3, batch=256, epochs=3)
    assert history == again
    assert all(np.array_equal(wa, wb) and np.array_equal(ba, bb)
               for (wa, ba), (wb, bb) in zip(a.weights, b.weights))
    assert history[-1] > history[0]


def test_the_mechanics_estimate_reads_moves_redraws_and_sample_laws():
    trajectory = {"limits": 1, "samples": [[1000, [1], [3], 1], [2000, [1], [0], 2]]}
    record = {"references": [[0.1, True, True], [0.2, True, False], [0.3, False, False],
                             [0.4, True, False]]}
    m = mdqr.estimate_mechanics([trajectory], [record], ["1"])
    assert m.theta == 3 / 4 and m.theta_reinit == 1 / 3
    assert m.invariant == [[1 / 4, 2 / 4, 0.0, 1 / 4]] and m.max_queue == 3


def rows_of(truth, model, margin=0.0):
    rows = []
    for w in ("window-1", "window-2"):
        rows.append({"kind": "truth_again", "seed": None, "window": w,
                     "report": {"queue_size": truth[0], "flow_law": truth[1]}})
        for seed in DESIGN["s0"]["model"]["training_seeds"]:
            q, f = model.get(seed, truth)
            rows.append({"kind": "mdqr", "seed": seed, "window": w,
                         "report": {"queue_size": q, "flow_law": f}})
    return rows


@pytest.mark.parametrize("extra,decision", [(0.049, "pass"), (0.051, "stop")])
def test_the_stop_rule_allows_the_truths_own_discrepancy_plus_the_margin(tmp_path, extra,
                                                                          decision):
    truth = ([0.02, 0.03], [0.01, 0.01, 0.02])
    worse = ([0.02, 0.03 + extra], [0.01, 0.01, 0.02])
    (tmp_path / "compare.json").write_text(json.dumps({"rows": rows_of(truth, {4: worse})}))
    result = json.loads(run.decide(tmp_path).read_text())
    assert result["decision"].startswith(decision)
    assert result["seeds"]["4"]["passes"] == (decision == "pass")
    assert all(result["seeds"][str(s)]["passes"] for s in (1, 2, 3, 5))


def test_a_window_whose_hash_changed_is_refused(tmp_path):
    window = tmp_path / "w.json"
    window.write_text("[]")
    (tmp_path / "data.json").write_text(json.dumps({"windows": {str(window): run.sha256(window)}}))
    assert run.read_checked(tmp_path, window) == []
    window.write_text("[1]")
    with pytest.raises(ValueError, match="not the window check recorded"):
        run.read_checked(tmp_path, window)
