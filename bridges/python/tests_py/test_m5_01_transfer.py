"""#106's transfer rows, on made-up values only (SYNTHETIC)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"))
import transfer  # noqa: E402


def samples(n=720, bid=3, ask=0, spread=1):
    return [[(i + 1) * 10_000, [bid, 0, 0], [ask, 0, 0], spread] for i in range(n)]


def test_the_pins_formula():
    rows = transfer.rows_of(samples(), [[15_000, True, False], [16_000, False, False], [25_000, True, True]], [1, 0, -1])
    assert rows[0] == (10_000_000_000, 19_999, 3, 20_001, 1)
    assert rows[1][1] == 20_001 and rows[2][1] == 19_999


@pytest.mark.parametrize("draw", [samples(719), samples(bid=2.5)])
def test_a_short_or_fractional_draw_fails(draw):
    with pytest.raises(transfer.FailedDraw):
        transfer.rows_of(draw, [], [])


def test_a_non_positive_bid_fails():
    moves = [[1, True, False]] * 10_000
    with pytest.raises(transfer.FailedDraw):
        transfer.rows_of(samples(), moves, [-1] * 10_000)


def test_the_last_block_holds_the_terminal_mark():
    cumulative = [0.0] * 718 + [5.0]
    blocks = transfer.block_nets(cumulative)
    assert blocks[-1] == 5.0 and sum(blocks[:-1]) == 0.0


def test_a_policy_stopped_early_holds_its_net():
    assert transfer.block_nets([1.0] * 10)[0] == 1.0
    assert sum(transfer.block_nets([1.0] * 10)[1:]) == 0.0


def test_rows_read_the_ten_second_grid_only():
    one_second = [[(i + 1) * 1000, [1, 0, 0], [1, 0, 0], 1] for i in range(7200)]
    assert len(transfer.rows_of(one_second, [], [])) == 720
    assert transfer.rows_of(one_second, [], [])[0][0] == 10_000_000_000


def test_only_the_action_version_may_differ():
    assert transfer.spec_matches([1, 1, 1, 1, "a"], (1, 2, 1, 1, "b"))
    assert not transfer.spec_matches([1, 1, 2, 1, "a"], (1, 2, 1, 1, "b"))
    assert not transfer.spec_matches([2, 1, 1, 1, "a"], (1, 2, 1, 1, "b"))


def test_the_captures_grid_from_zero_gives_720_rows():
    from_zero = [[i * 1000, [1, 0, 0], [1, 0, 0], 1] for i in range(7200)]
    assert len(transfer.rows_of(from_zero, [], [], 0)) == 720
    with pytest.raises(transfer.FailedDraw):
        transfer.rows_of(from_zero, [], [])
