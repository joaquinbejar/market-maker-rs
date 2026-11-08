"""The reactive market of #282 and the lots filled per step. All data here is synthetic."""

import json
import pathlib

import pytest

from conftest import config, walk
from mm_rl_bridge import BridgeFailure, MaskableMarketMakingEnv, ReactiveQrMarket

M4_05 = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "m4_05"
DESIGN = json.loads((M4_05 / "design.json").read_text())
DAY_MS = 86_400_000


def market(regime=0, day=1, rows=8640):
    sim = DESIGN["simulator"]
    episode = {
        "seed": 1_000_003 * (regime + 1) + day,
        "reference_ticks": sim["reference_ticks"],
        "start_ms": (day - 1) * DAY_MS,
        "sample_ms": sim["sample_ms"],
        "rows": rows,
    }
    return ReactiveQrMarket(
        json.dumps(sim["regimes"][regime]), json.dumps(DESIGN["kernel"]), json.dumps(episode)
    )


def test_a_design_day_has_qr_days_digest():
    record = json.loads((M4_05 / "digests.json").read_text())["days"][0]
    assert (record["regime"], record["day"]) == ("calm", 1)
    day = market()
    assert day.digest() == record["dataset_digest"]
    assert len(day.rows()) == 8640 and day.stopped() is None


def test_a_reaction_keeps_the_rows_up_to_it():
    day = market(regime=1, day=3, rows=500)
    before = day.rows()
    assert day.react(50, 0) == "0"
    assert day.rows() == before
    assert float(day.react(50, -2)) >= 0
    after = day.rows()
    assert after[:51] == before[:51] and after[51:] != before[51:]
    again = market(regime=1, day=3, rows=500)
    again.react(50, -2)
    assert again.rows() == after


def test_bad_input_raises_and_never_crashes():
    with pytest.raises(BridgeFailure):
        ReactiveQrMarket("{", json.dumps(DESIGN["kernel"]), "{}")
    day = market(rows=20)
    with pytest.raises(BridgeFailure):
        day.react(20, 1)
    with pytest.raises(BridgeFailure):
        day.react(5, -(2**63))
    with pytest.raises(OverflowError):
        day.react(-1, 1)


def test_each_step_reports_the_lots_filled_by_side():
    cfg = config()
    env = MaskableMarketMakingEnv(cfg, walk())
    env.reset(seed=1)
    total = {"buy": 0, "sell": 0}
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step([1, 0, 1, 0, 0])
        filled = info["filled_lots"]
        assert set(filled) == {"buy", "sell"}
        assert info["orders"]["fills"] == 0 or filled["buy"] + filled["sell"] > 0
        total = {side: total[side] + filled[side] for side in total}
        done = terminated or truncated
    assert total["buy"] > 0 and total["sell"] > 0, total


def test_replacing_rows_replays_as_a_fresh_run_on_them():
    """The worker's replay against the plain path: a fresh environment on the new rows,
    stepped with the same actions from the start."""
    grid = DESIGN["arms"]["maskable_ppo"]["grid"]
    cfg = {**config(), "tick_size": DESIGN["kernel"]["tick_size"],
           "lot_size": DESIGN["kernel"]["lot_size"], "bid_offsets": grid["bid_offsets"],
           "ask_offsets": grid["ask_offsets"], "sizes": grid["sizes"], "max_steps": 400}
    day = market(regime=0, day=2, rows=400)
    action = [1, 1, 1, 0, 0]
    env = MaskableMarketMakingEnv(cfg, [tuple(r) for r in day.rows()])
    replaced = 0
    try:
        env.reset(seed=3)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            if not done and info["filled_lots"]["sell"] != info["filled_lots"]["buy"] and replaced < 20:
                up = info["filled_lots"]["sell"] > info["filled_lots"]["buy"]
                day.react(info["row"], 1 if up else -1)
                digest = env.bridge_info["dataset_digest"]
                env.replace_rows(day.rows())
                assert env.bridge_info["dataset_digest"] != digest
                replaced += 1
        final = env.attribution()
    finally:
        env.close()
    assert replaced > 0
    fresh = MaskableMarketMakingEnv(cfg, [tuple(r) for r in day.rows()])
    try:
        fresh.reset(seed=3)
        for _ in range(steps):
            *_, info = fresh.step(action)
        assert fresh.attribution() == final
    finally:
        fresh.close()
