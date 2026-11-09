"""Offline training on logged decisions of #102's maskable grid (#280): behaviour cloning and a
discrete conservative Q-learning, both restricted by the mask.

A joint action is a code: 0 for a hold, and 1 + the quote's index otherwise, with the quote
indexed as (bid offset, ask offset, bid size, ask size), the last fastest. The mask has #102's
layout: kind (hold, quote), bid offset, ask offset, bid size, ask size.

Nothing here reads a log path or an environment; `run.py` does.
"""

import itertools
import math

import numpy as np
import torch
from torch import nn


class Actions:
    """The joint actions of a grid with `k` offsets and `m` sizes on each side."""

    def __init__(self, k, m):
        self.k, self.m = k, m
        self.count = 1 + k * k * m * m
        quotes = list(itertools.product(range(k), range(k), range(m), range(m)))
        self.quotes = np.asarray(quotes, dtype=int)

    def code(self, action):
        """The code of an action (kind, bid, ask, bid size, ask size) given by indices."""
        kind, bid, ask, bid_size, ask_size = (int(v) for v in np.asarray(action).ravel())
        if kind == 0:
            return 0
        return 1 + ((bid * self.k + ask) * self.m + bid_size) * self.m + ask_size

    def action(self, code):
        """The action, by indices, of a code; every hold is (0, 0, 0, 0, 0)."""
        code = int(code)
        if code == 0:
            return (0, 0, 0, 0, 0)
        return (1, *(int(v) for v in self.quotes[code - 1]))

    def joint_mask(self, masks):
        """The allowed codes of each mask row: a hold when the mask allows one, a quote when it
        allows a quote and each of the quote's indices."""
        masks = np.atleast_2d(np.asarray(masks, dtype=bool))
        k, m = self.k, self.m
        bid = masks[:, 2:2 + k]
        ask = masks[:, 2 + k:2 + 2 * k]
        bid_size = masks[:, 2 + 2 * k:2 + 2 * k + m]
        ask_size = masks[:, 2 + 2 * k + m:2 + 2 * k + 2 * m]
        q = self.quotes
        quote = (masks[:, [1]] & bid[:, q[:, 0]] & ask[:, q[:, 1]]
                 & bid_size[:, q[:, 2]] & ask_size[:, q[:, 3]])
        return np.concatenate([masks[:, [0]], quote], axis=1)


class Transitions:
    """Logged days as transitions: the observation, the joint mask, the logged code, the reward,
    the next observation and joint mask, and whether the decision was the day's last."""

    def __init__(self, actions, days, reward_scale):
        obs, mask, code, reward, nxt, nxt_mask, done = [], [], [], [], [], [], []
        for day in days:
            o = np.asarray(day["observation"], dtype=np.float32)
            jm = actions.joint_mask(day["mask"])
            n = len(o)
            if n < 1 or len(day["action"]) != n or len(day["reward"]) != n:
                raise ValueError("a logged day with rows of different lengths")
            obs.append(o)
            mask.append(jm)
            code.append(np.asarray(day["action"], dtype=np.int64))
            reward.append(np.asarray(day["reward"], dtype=np.float64) / reward_scale)
            # The day's last decision has no next state: its target is its reward alone.
            nxt.append(np.concatenate([o[1:], o[-1:]]))
            nxt_mask.append(np.concatenate([jm[1:], jm[-1:]]))
            last = np.zeros(n, dtype=bool)
            last[-1] = True
            done.append(last)
        self.observation = torch.as_tensor(np.concatenate(obs))
        self.mask = torch.as_tensor(np.concatenate(mask))
        self.code = torch.as_tensor(np.concatenate(code))
        self.reward = torch.as_tensor(np.concatenate(reward), dtype=torch.float32)
        self.next_observation = torch.as_tensor(np.concatenate(nxt))
        self.next_mask = torch.as_tensor(np.concatenate(nxt_mask))
        self.done = torch.as_tensor(np.concatenate(done))
        logged_allowed = self.mask[torch.arange(len(self.code)), self.code]
        if not bool(logged_allowed.all()):
            raise ValueError("a logged action the logged mask does not allow")

    def __len__(self):
        return len(self.code)

    def batch(self, generator, size):
        index = torch.randint(len(self), (size,), generator=generator)
        return (self.observation[index], self.mask[index], self.code[index], self.reward[index],
                self.next_observation[index], self.next_mask[index], self.done[index])


def mlp(inputs, outputs, hidden):
    layers, width = [], inputs
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


NEG = -1e9


def masked(values, mask):
    """`values` with the disallowed actions at a large negative number."""
    return values.masked_fill(~mask, NEG)


def bc_loss(logits, mask, code):
    """Cross-entropy of the logged code under a softmax over the allowed actions only."""
    return nn.functional.cross_entropy(masked(logits, mask), code)


def cql_loss(q, target_q, batch, gamma, alpha):
    """Double DQN's temporal difference loss plus alpha times the conservative term, both over
    the allowed actions only.

    The target of a decision is r + gamma * Q_target(s', argmax over the allowed a' of Q(s', a')),
    with no bootstrap after the day's last decision. The conservative term is
    logsumexp over the allowed a of Q(s, a), minus Q(s, a_logged)."""
    obs, mask, code, reward, nxt, nxt_mask, done = batch
    values = q(obs)
    chosen = values.gather(1, code[:, None]).squeeze(1)
    with torch.no_grad():
        best = masked(q(nxt), nxt_mask).argmax(dim=1, keepdim=True)
        bootstrap = target_q(nxt).gather(1, best).squeeze(1)
        target = reward + gamma * (~done).float() * bootstrap
    td = nn.functional.smooth_l1_loss(chosen, target)
    conservative = (torch.logsumexp(masked(values, mask), dim=1) - chosen).mean()
    return td + alpha * conservative, td, conservative


def greedy(values, mask):
    """The allowed action of the largest value, the smallest code on a tie."""
    return int(masked(values, mask).argmax())


class Policy:
    """A trained network as a deterministic policy with #102's `predict` interface."""

    def __init__(self, network, actions):
        self.network = network.eval()
        self.actions = actions

    def code(self, observation, mask):
        joint = torch.as_tensor(self.actions.joint_mask(mask)[0])
        with torch.no_grad():
            values = self.network(torch.as_tensor(np.asarray(observation, dtype=np.float32))[None])
        return greedy(values[0], joint)

    def predict(self, observation, deterministic=True, action_masks=None):
        return np.asarray(self.actions.action(self.code(observation, action_masks))), None


def train(kind, data, seed, settings, log_every=1000):
    """One training run. `kind` is "bc" or "cql"; `settings` holds the hidden widths, the
    learning rate, the batch size and the gradient steps, and for "cql" gamma, alpha and the
    target network's period. Returns the network and a history of the losses; stops on the
    first loss that is not finite, as the design's stop rule says."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    inputs, outputs = data.observation.shape[1], data.mask.shape[1]
    network = mlp(inputs, outputs, settings["hidden"])
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    target = None
    if kind == "cql":
        target = mlp(inputs, outputs, settings["hidden"])
        target.load_state_dict(network.state_dict())
    history = []
    for step in range(1, settings["gradient_steps"] + 1):
        batch = data.batch(generator, settings["batch_size"])
        if kind == "bc":
            loss = bc_loss(network(batch[0]), batch[1], batch[2])
            parts = {}
        else:
            loss, td, conservative = cql_loss(network, target, batch, settings["gamma"],
                                              settings["alpha"])
            parts = {"td": td.item(), "conservative": conservative.item()}
        value = loss.item()
        if not math.isfinite(value):
            history.append({"step": step, "loss": value, **parts, "stop": "not finite"})
            return network, history, False
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if target is not None and step % settings["target_period"] == 0:
            target.load_state_dict(network.state_dict())
        if step % log_every == 0 or step == settings["gradient_steps"]:
            history.append({"step": step, "loss": value, **parts})
    return network, history, True


def position_bin(position_lots):
    """Signed position in lots: at most -6, -5 to -1, 0, 1 to 5, at least 6."""
    return np.digitize(np.asarray(position_lots, dtype=float), [-5.5, -0.5, 0.5, 5.5])


def support_counts(train_keys, keys):
    """For each (state bin, code) key, how many training decisions hold it."""
    counts = {}
    for key in train_keys:
        counts[key] = counts.get(key, 0) + 1
    return np.asarray([counts.get(key, 0) for key in keys])
