"""The masked joint actions, the losses and the training loop of #280's offline arms."""

import importlib.util
import math
import pathlib

import numpy as np
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_11_offline", HERE.parent / "experiments" / "m4_11" / "offline.py"
)
offline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(offline)

M4_07 = importlib.util.spec_from_file_location(
    "m4_07_estimate_for_m4_11", HERE.parent / "experiments" / "m4_07" / "estimate.py"
)
estimate = importlib.util.module_from_spec(M4_07)
M4_07.loader.exec_module(estimate)

OFFSETS, SIZES = [0, 1, 2, 4, 8], [1, 2]


def full_mask():
    return np.ones(16, dtype=bool)


def test_codes_follow_the_logged_layout_and_round_trip():
    actions = offline.Actions(5, 2)
    assert actions.count == 101
    assert actions.code((0, 3, 1, 1, 0)) == 0
    # bid index 2, ask index 3, bid size 1, ask size 0: 1 + ((2*5+3)*2+1)*2+0 = 55.
    assert actions.code((1, 2, 3, 1, 0)) == 55
    for code in range(actions.count):
        assert actions.code(actions.action(code)) == code
        # The same code as #104's logs: its decoding gives these indices' offsets and sizes.
        kind, bid, ask, bs, ask_s = actions.action(code)
        expected = (0, 0, 0, 0, 0) if code == 0 else (1, OFFSETS[bid], OFFSETS[ask], SIZES[bs],
                                                      SIZES[ask_s])
        assert estimate.decode(code, OFFSETS, SIZES) == expected


def test_the_joint_mask_allows_exactly_the_quotes_whose_indices_the_mask_allows():
    actions = offline.Actions(5, 2)
    mask = full_mask()
    assert actions.joint_mask(mask).all()
    mask[2 + 4] = False  # bid offset index 4 (8 ticks) refused
    mask[2 + 2 * 5 + 2 + 1] = False  # ask size index 1 refused
    joint = actions.joint_mask(mask)[0]
    for code in range(1, actions.count):
        _, bid, _, _, ask_size = actions.action(code)
        assert joint[code] == (bid != 4 and ask_size != 1)
    assert joint[0]
    no_quote = full_mask()
    no_quote[1] = False
    joint = actions.joint_mask(no_quote)[0]
    assert joint[0] and not joint[1:].any()
    no_hold = full_mask()
    no_hold[0] = False
    assert not actions.joint_mask(no_hold)[0][0]


def day(n, codes=None, reward=None):
    rng = np.random.default_rng(n)
    return {"observation": rng.normal(size=(n, 3)).astype(np.float32),
            "mask": np.ones((n, 16), dtype=bool),
            "action": np.zeros(n, dtype=np.int16) if codes is None else np.asarray(codes),
            "reward": np.ones(n) if reward is None else np.asarray(reward, dtype=float)}


def test_transitions_end_each_day_and_scale_rewards():
    data = offline.Transitions(offline.Actions(5, 2), [day(3), day(2)], reward_scale=4.0)
    assert len(data) == 5
    assert data.done.tolist() == [False, False, True, False, True]
    assert torch.allclose(data.reward, torch.full((5,), 0.25))
    # The next observation within a day is the following row; across days it never leaks.
    first = day(3)["observation"]
    assert torch.equal(data.next_observation[0], torch.as_tensor(first[1]))
    assert torch.equal(data.next_observation[2], torch.as_tensor(first[2]))


def test_transitions_refuse_a_logged_action_the_mask_forbids_or_ragged_days():
    bad = day(2, codes=[0, 7])
    bad["mask"][1, 1] = False  # no quote allowed at the second decision, yet code 7 logged
    with pytest.raises(ValueError, match="does not allow"):
        offline.Transitions(offline.Actions(5, 2), [bad], 1.0)
    ragged = day(3)
    ragged["reward"] = np.ones(2)
    with pytest.raises(ValueError, match="different lengths"):
        offline.Transitions(offline.Actions(5, 2), [ragged], 1.0)


def test_bc_loss_is_the_cross_entropy_over_allowed_actions_only():
    logits = torch.tensor([[1.0, 2.0, 30.0]])
    mask = torch.tensor([[True, True, False]])
    loss = offline.bc_loss(logits, mask, torch.tensor([1]))
    # Over {1, 2} only: -log(e^2 / (e^1 + e^2)).
    assert float(loss) == pytest.approx(-math.log(math.e**2 / (math.e + math.e**2)), rel=1e-6)


class Table(torch.nn.Module):
    """Q values that do not depend on the observation's values, only on its first entry."""

    def __init__(self, rows):
        super().__init__()
        self.rows = torch.as_tensor(rows, dtype=torch.float32)

    def forward(self, obs):
        return self.rows[obs[:, 0].long()]


def test_cql_loss_on_hand_values():
    # Two states (0 and 1), three actions. Online Q and target Q differ, so double DQN shows.
    q = Table([[1.0, 2.0, 3.0], [4.0, 0.0, 5.0]])
    target = Table([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    obs = torch.tensor([[0.0]])
    nxt = torch.tensor([[1.0]])
    mask = torch.tensor([[True, True, False]])
    nxt_mask = torch.tensor([[True, True, False]])
    batch = (obs, mask, torch.tensor([1]), torch.tensor([0.5]), nxt, nxt_mask,
             torch.tensor([False]))
    loss, td, conservative = offline.cql_loss(q, target, batch, gamma=0.9, alpha=2.0)
    # Next action: argmax of online Q over allowed {0, 1} at state 1 is 0 (4 > 0), not 2;
    # its target value is 10, so the target is 0.5 + 0.9 * 10 = 9.5; Q(s, 1) = 2; |error| 7.5.
    assert float(td) == pytest.approx(7.5 - 0.5)
    # logsumexp over the allowed {1, 2} at state 0, minus Q(s, a_logged) = 2.
    assert float(conservative) == pytest.approx(math.log(math.e + math.e**2) - 2.0)
    assert float(loss) == pytest.approx(7.0 + 2.0 * float(conservative))
    # The day's last decision does not bootstrap: target 0.5, |error| 1.5.
    last = batch[:6] + (torch.tensor([True]),)
    _, td_last, _ = offline.cql_loss(q, target, last, gamma=0.9, alpha=2.0)
    assert float(td_last) == pytest.approx(1.5 - 0.5)


def test_greedy_picks_the_best_allowed_action_and_the_smallest_code_on_a_tie():
    values = torch.tensor([1.0, 5.0, 9.0, 5.0])
    assert offline.greedy(values, torch.tensor([True, True, False, True])) == 1
    assert offline.greedy(values, torch.tensor([True, False, False, False])) == 0


def test_the_policy_acts_on_the_mask_and_returns_an_index_action():
    actions = offline.Actions(5, 2)
    net = torch.nn.Linear(3, actions.count)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.copy_(torch.arange(actions.count, dtype=torch.float32))
    policy = offline.Policy(net, actions)
    mask = full_mask()
    action, _ = policy.predict(np.zeros(3), action_masks=mask)
    assert tuple(action) == actions.action(100)
    mask[1] = False
    action, _ = policy.predict(np.zeros(3), action_masks=mask)
    assert tuple(action) == (0, 0, 0, 0, 0)


def settings(kind, steps=300):
    base = {"hidden": [32], "learning_rate": 0.01, "batch_size": 64, "gradient_steps": steps}
    if kind == "cql":
        base.update({"gamma": 0.5, "alpha": 1.0, "target_period": 50})
    return base


def test_bc_recovers_a_deterministic_logged_rule():
    # The logged rule quotes code 7 when the first feature is positive and holds otherwise.
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, 3)).astype(np.float32)
    codes = np.where(obs[:, 0] > 0, 7, 0)
    logged = {"observation": obs, "mask": np.ones((400, 16), dtype=bool), "action": codes,
              "reward": np.zeros(400)}
    actions = offline.Actions(5, 2)
    data = offline.Transitions(actions, [logged], 1.0)
    network, history, finished = offline.train("bc", data, 1, settings("bc"))
    assert finished
    policy = offline.Policy(network, actions)
    agree = np.mean([policy.code(o, np.ones(16, dtype=bool)) == c for o, c in zip(obs, codes)])
    assert agree > 0.95
    assert history[-1]["step"] == 300


def test_cql_prefers_the_logged_rewarding_action_over_one_never_logged():
    # One state; holding pays 0 and code 3 pays 1, both logged; no other action is logged.
    n = 400
    codes = np.where(np.arange(n) % 2 == 0, 0, 3)
    logged = {"observation": np.zeros((n, 3), dtype=np.float32),
              "mask": np.ones((n, 16), dtype=bool), "action": codes,
              "reward": (codes == 3).astype(float)}
    actions = offline.Actions(5, 2)
    data = offline.Transitions(actions, [logged], 1.0)
    network, _, finished = offline.train("cql", data, 1, settings("cql", 600))
    assert finished
    assert offline.Policy(network, actions).code(np.zeros(3), np.ones(16, dtype=bool)) == 3


def test_training_stops_on_a_loss_that_is_not_finite():
    logged = day(10, reward=[float("nan")] * 10)
    data = offline.Transitions(offline.Actions(5, 2), [logged], 1.0)
    _, history, finished = offline.train("cql", data, 1, settings("cql", 50))
    assert not finished
    assert history[-1]["stop"] == "not finite" and history[-1]["step"] == 1


def test_position_bins_and_support_counts():
    assert offline.position_bin([-9, -6, -5, -1, 0, 1, 5, 6]).tolist() == [0, 0, 1, 1, 2, 3, 3, 4]
    keys = [(1, 0), (1, 0), (2, 5)]
    assert offline.support_counts(keys, [(1, 0), (2, 5), (3, 1)]).tolist() == [2, 1, 0]


def test_cql_bootstraps_through_the_synced_target_network():
    # Days of two decisions: the first pays 0, the last pays 1 and ends the day. With alpha 0
    # and gamma 0.5, the first state's logged value is 0.5 * 1 only if the target network
    # follows the trained one; a target left at its initial weights gives 0.5 * Q_init.
    first = np.zeros(3, dtype=np.float32)
    last = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    logged = {"observation": np.tile(np.stack([first, last]), (100, 1)),
              "mask": np.ones((200, 16), dtype=bool), "action": np.zeros(200, dtype=np.int16),
              "reward": np.tile([0.0, 1.0], 100)}
    days = [{k: v[2 * i:2 * i + 2] for k, v in logged.items()} for i in range(100)]
    actions = offline.Actions(5, 2)
    data = offline.Transitions(actions, days, 1.0)
    run = dict(settings("cql", 1500), alpha=0.0)
    network, _, finished = offline.train("cql", data, 3, run)
    assert finished
    with torch.no_grad():
        values = network(torch.as_tensor(np.stack([first, last])))[:, 0]
    assert float(values[1]) == pytest.approx(1.0, abs=0.1)
    assert float(values[0]) == pytest.approx(0.5, abs=0.1)
