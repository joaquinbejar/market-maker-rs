"""Feature extractors for the comparison of #107: the Hawkes-Structured Network and its controls.

Every extractor reads the same flat observation, described by `Layout`: a buffer of the last
`L` items (market orders in the paper's setting, decisions in ours), newest first, as their
ages, their validity flags and `C` values each, then `S` scalars that every arm receives as
they are. The extractors differ only in how they summarize the buffer:

- `ScalarOnly`: not at all;
- `FrameStack`: the newest `k` items as they are;
- `FixedEma`: per channel, `sum_i x_ic exp(-beta age_i)` for a fixed grid of `beta`;
- `Hsn`: per channel and mode, `alpha_cj sum_i x_ic exp(-beta_cj age_i)`, with `alpha` and
  `beta` positive (softplus plus a floor) and learned end to end, as in arXiv:2610.16524v1;
- `Recurrent`: a GRU or LSTM over the valid items, oldest first, after a linear embedding.

With `with_current`, `Hsn` and `Recurrent` also pass the newest item's values on as they are,
so that they see the current observation as a plain MLP does.

A fixed EMA is never called HSN: only `Hsn` has learnable decays, and its tests check that its
decays move under a gradient.
"""

import math

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional


class Layout:
    """`[ages (L), valid (L), values (L x C), scalars (S)]`."""

    def __init__(self, buffer, channels, scalars):
        self.buffer, self.channels, self.scalars = buffer, channels, scalars

    @property
    def size(self):
        return (2 + self.channels) * self.buffer + self.scalars

    def split(self, obs):
        L, C = self.buffer, self.channels
        ages = obs[:, :L]
        valid = obs[:, L : 2 * L]
        values = obs[:, 2 * L : (2 + C) * L].reshape(-1, L, C)
        scalars = obs[:, (2 + C) * L :]
        return ages, valid, values, scalars

    def as_dict(self):
        return {"buffer": self.buffer, "channels": self.channels, "scalars": self.scalars}


def _inverse_softplus(y):
    return y + math.log(-math.expm1(-y))


class _Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, layout, features_dim):
        if observation_space.shape != (layout.size,):
            raise ValueError(f"observation {observation_space.shape} is not layout {layout.size}")
        super().__init__(observation_space, features_dim)
        self.layout = layout


class ScalarOnly(_Extractor):
    def __init__(self, observation_space, layout):
        super().__init__(observation_space, layout, layout.scalars)

    def forward(self, obs):
        return self.layout.split(obs)[3]


class FrameStack(_Extractor):
    def __init__(self, observation_space, layout, frames):
        if not 1 <= frames <= layout.buffer:
            raise ValueError("1 <= frames <= buffer")
        super().__init__(
            observation_space, layout, frames * (1 + layout.channels) + layout.scalars
        )
        self.frames = frames

    def forward(self, obs):
        ages, valid, values, scalars = self.layout.split(obs)
        k = self.frames
        mask = valid[:, :k]
        items = torch.cat([(ages[:, :k] * mask).unsqueeze(-1), values[:, :k] * mask.unsqueeze(-1)], -1)
        return torch.cat([items.flatten(1), scalars], 1)


def _current(valid, values):
    """The newest item's values, zero if there is none."""
    return values[:, 0] * valid[:, :1]


def _filtered(ages, valid, values, alphas, betas):
    """`alpha_cj sum_i valid_i x_ic exp(-beta_cj age_i)`, shape (batch, C * p)."""
    # (batch, L, 1, 1) * (C, p) -> (batch, L, C, p)
    decay = torch.exp(-ages[:, :, None, None] * betas)
    weighted = (valid[:, :, None, None] * values[:, :, :, None]) * decay
    return (alphas * weighted.sum(1)).flatten(1)


class FixedEma(_Extractor):
    def __init__(self, observation_space, layout, decays):
        super().__init__(
            observation_space, layout, layout.channels * len(decays) + layout.scalars
        )
        betas = torch.tensor(decays, dtype=torch.float32).expand(layout.channels, -1).clone()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", torch.ones_like(betas))

    def forward(self, obs):
        ages, valid, values, scalars = self.layout.split(obs)
        return torch.cat([_filtered(ages, valid, values, self.alphas, self.betas), scalars], 1)


class Hsn(_Extractor):
    """The learnable mixture of exponentials, `p` modes per channel."""

    def __init__(self, observation_space, layout, alphas, betas, floor, with_current=False):
        if len(alphas) != len(betas) or not alphas:
            raise ValueError("one alpha per beta")
        if min(alphas) <= floor or min(betas) <= floor:
            raise ValueError("initial alpha and beta above the floor")
        p = len(betas)
        current = layout.channels if with_current else 0
        super().__init__(observation_space, layout, layout.channels * p + current + layout.scalars)
        self.floor = floor
        self.with_current = with_current

        def raw(values):
            start = torch.tensor([_inverse_softplus(v - floor) for v in values])
            return nn.Parameter(start.expand(layout.channels, -1).clone())

        self.raw_alphas = raw(alphas)
        self.raw_betas = raw(betas)

    def alphas(self):
        return functional.softplus(self.raw_alphas) + self.floor

    def betas(self):
        return functional.softplus(self.raw_betas) + self.floor

    def forward(self, obs):
        ages, valid, values, scalars = self.layout.split(obs)
        parts = [_filtered(ages, valid, values, self.alphas(), self.betas())]
        if self.with_current:
            parts.append(_current(valid, values))
        return torch.cat(parts + [scalars], 1)


class Recurrent(_Extractor):
    """A GRU or LSTM over the valid items, oldest first; no item gives a zero summary."""

    def __init__(self, observation_space, layout, cell, embed, hidden, with_current=False):
        current = layout.channels if with_current else 0
        super().__init__(observation_space, layout, hidden + current + layout.scalars)
        self.with_current = with_current
        self.embed = nn.Linear(1 + layout.channels, embed)
        kinds = {"gru": nn.GRU, "lstm": nn.LSTM}
        if cell not in kinds:
            raise ValueError(f"unknown cell {cell}")
        self.cell = cell
        self.rnn = kinds[cell](embed, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, obs):
        ages, valid, values, scalars = self.layout.split(obs)
        L = self.layout.buffer
        counts = valid.sum(1).round().long()
        # Valid items are the newest `n`, so oldest first is index n - 1 down to 0.
        steps = torch.arange(L, device=obs.device)
        index = (counts[:, None] - 1 - steps[None, :]).clamp(min=0)
        items = torch.cat([ages.unsqueeze(-1), values], -1)
        ordered = torch.gather(items, 1, index[:, :, None].expand(-1, -1, items.shape[-1]))
        lengths = counts.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            torch.tanh(self.embed(ordered)), lengths, batch_first=True, enforce_sorted=False
        )
        _, state = self.rnn(packed)
        last = (state[0] if self.cell == "lstm" else state)[-1]
        last = last * (counts > 0).to(last.dtype)[:, None]
        parts = [last, _current(valid, values)] if self.with_current else [last]
        return torch.cat(parts + [scalars], 1)


EXTRACTORS = {
    "mlp": ScalarOnly,
    "frame_stack": FrameStack,
    "fixed_ema": FixedEma,
    "hsn": Hsn,
    "recurrent": Recurrent,
}


def policy_kwargs(arm, layout, width):
    """The `policy_kwargs` of an arm: its extractor, shared by a policy and a value MLP."""
    kwargs = {k: v for k, v in arm.items() if k not in ("extractor", "name", "note")}
    return {
        "features_extractor_class": EXTRACTORS[arm["extractor"]],
        "features_extractor_kwargs": {"layout": layout, **kwargs},
        "share_features_extractor": True,
        "net_arch": {"pi": [width, width], "vf": [width, width]},
        "activation_fn": nn.Tanh,
    }


def parameter_count(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)
