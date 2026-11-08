"""Cancel races under a chosen duration (#110): with a cancel latency, a replacing quote
cancels the old one only when the cancel takes effect, a fill landing before that stands,
and a cancel arriving after the fill is a cancel rejected."""

from conftest import config, row

from mm_rl_bridge import BoxMarketMakingEnv
from mm_rl_bridge.wrappers import ChosenDuration

# Rows 25 ms apart; on the fourth the ask comes down to 99.5, onto a bid at the touch.
ROWS = [row(1000, 199, 201), row(1025, 199, 201), row(1050, 199, 201),
        row(1075, 198, 199), row(1100, 199, 201), row(1125, 199, 201), row(1150, 199, 201)]
# Offsets count ticks from the mid (200 here): offset 1 is a bid at the touch of 199, offset 10
# a bid at 190 that the ask of 199 never reaches; the ask is 10 ticks out; one lot each.
BID_AT_TOUCH = [1.0, 0.0, 1.0, -1.0, -1.0]
BID_BELOW = [1.0, 1.0, 1.0, -1.0, -1.0]


def env(cancel_ms):
    settings = config(policy="reject")
    settings["latency"] = {"quantum_ns": 1_000_000, "observation_q": 0, "submit_q": 0,
                           "cancel_q": cancel_ms, "ack_q": 0, "jitter_q": 0}
    return ChosenDuration(BoxMarketMakingEnv(settings, ROWS), choices=(1, 2))


def run(cancel_ms):
    e = env(cancel_ms)
    e.reset(seed=1)
    steps = [e.step(BID_AT_TOUCH + [1.0])[4]]          # rests two rows, decides at 1050
    while True:
        _, _, terminated, truncated, info = e.step(BID_BELOW + [-1.0])
        steps.append(info)
        if terminated or truncated:
            break
    e.close()
    return steps


def test_a_fill_before_the_cancel_takes_effect_stands_and_the_cancel_is_rejected():
    steps = run(50)
    # The replacing quote at 1050 sends its cancel, effective at 1100; the old bid fills at
    # 1075 first, so its cancel arrives too late.
    fills = sum(s["orders"]["fills"] for s in steps)
    assert fills == 1
    assert sum(s["orders"]["cancel_rejected"] for s in steps) == 1
    assert sum(s["orders"]["canceled"] for s in steps[:2]) == 0


def test_without_the_latency_the_cancel_comes_first_and_nothing_fills():
    steps = run(0)
    assert sum(s["orders"]["fills"] for s in steps) == 0
    assert sum(s["orders"]["cancel_rejected"] for s in steps) == 0
    # The old bid is cancelled at the replacing decision itself.
    assert steps[1]["orders"]["canceled"] >= 1
