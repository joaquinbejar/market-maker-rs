"""Fixtures. All data here is synthetic."""

import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def config(terminal="close", policy="mask", max_steps=100, penalty="0.25"):
    """Tick 0.5, lot 1, fee 0.1 %, the grid of the crate's hand computed trajectories."""
    return {
        "tick_size": "0.5",
        "lot_size": "1",
        "max_price": "1000",
        "max_order_size": "10",
        "max_steps": max_steps,
        "fee_rate": "0.001",
        "initial_capital": "0",
        "latency": {
            "quantum_ns": 1_000_000,
            "observation_q": 0,
            "submit_q": 0,
            "cancel_q": 0,
            "ack_q": 0,
            "jitter_q": 0,
        },
        "fill_model": {"kind": "immediate"},
        "limits": None,
        "bid_offsets": [0, 1, 10],
        "ask_offsets": [0, 1, 10],
        "sizes": [1, 2],
        "scales": {
            "spread_ticks": "10",
            "mid_change_ticks": "4",
            "row_gap_ms": "4000",
            "size_lots": "10",
            "position": "4",
            "entry_gap": "2",
        },
        "f32_unit": "1",
        "invalid_action": policy,
        "invalid_action_penalty": penalty,
        "terminal": terminal,
    }


def row(ms, bid, ask, bid_lots=5, ask_lots=5):
    return (ms * 1_000_000, bid, bid_lots, ask, ask_lots)


LONG_ROWS = [row(1000, 199, 201), row(2000, 197, 199), row(3000, 200, 202)]


def walk(rows=400, spacing_ms=25):
    """A deterministic walk on the grid, in which a near quote fills now and then."""
    out = []
    for i in range(rows):
        bid = 190 + (i * 7) % 13
        out.append(row(1000 + i * spacing_ms, bid, bid + 2, 4, 3))
    return out
