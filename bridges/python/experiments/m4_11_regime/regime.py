"""The regime's one-hot and the direct diagnosis of aliasing of #309's design.

Nothing here reads a log path or an environment; the runner does.
"""

import numpy as np

REGIMES = ("calm", "stressed", "thin")
TEN_BINS = 10


def one_hot(regime):
    """The regime's one-hot, in the design's order."""
    return np.asarray([float(name == regime) for name in REGIMES], dtype=np.float32)


def with_regime(observations, regime):
    """The observations, one per row or a single one, with the regime's one-hot appended."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim == 1:
        return np.concatenate([observations, one_hot(regime)])
    columns = np.tile(one_hot(regime), (len(observations), 1))
    return np.concatenate([observations, columns], axis=1)


def ceiling(keys, actions):
    """For decisions given as a key and an action: the share whose key carries more than one
    action, and the highest agreement any classifier of the key can reach."""
    if len(keys) != len(actions):
        raise ValueError("keys and actions of different lengths")
    if not keys:
        raise ValueError("no decisions")
    by_key = {}
    for key, action in zip(keys, actions, strict=True):
        counts = by_key.setdefault(key, {})
        counts[int(action)] = counts.get(int(action), 0) + 1
    ambiguous = best = 0
    for counts in by_key.values():
        best += max(counts.values())
        if len(counts) > 1:
            ambiguous += sum(counts.values())
    return {"decisions": len(keys), "keys": len(by_key), "ambiguous_share": ambiguous / len(keys),
            "agreement_ceiling": best / len(keys)}


def steps_bin(steps_left_fraction):
    """Steps left, as a fraction of the day in [0, 1], in ten equal bins."""
    fraction = np.asarray(steps_left_fraction, dtype=float)
    return np.minimum((fraction * TEN_BINS).astype(int), TEN_BINS - 1)


def aliasing(days):
    """The design's diagnosis. Each day holds `regime`, and per decision `position`,
    `steps_left` (raw), `steps_fraction` (of the day), `mask`, `state_bin` (#104's) and `target`
    (as_grid's code). Returns the exact key's and the binned key's results, with and without
    the regime, and whether the exact key with the regime determines as_grid's action."""
    exact, exact_regime, binned, binned_regime, targets = [], [], [], [], []
    for day in days:
        regime = day["regime"]
        positions = np.asarray(day["position"], dtype=float).tolist()
        steps = np.asarray(day["steps_left"], dtype=float).tolist()
        bins = steps_bin(day["steps_fraction"]).tolist()
        states = np.asarray(day["state_bin"], dtype=int).tolist()
        masks = [row.tobytes() for row in np.asarray(day["mask"], dtype=bool)]
        for position, step, step_bin, state, mask in zip(positions, steps, bins, states, masks,
                                                         strict=True):
            key = (position, step, mask)
            exact.append(key)
            exact_regime.append((regime, *key))
            coarse = (state, position, step_bin)
            binned.append(coarse)
            binned_regime.append((regime, *coarse))
        targets.extend(np.asarray(day["target"], dtype=int).tolist())
    with_label = ceiling(exact_regime, targets)
    return {"exact": ceiling(exact, targets), "exact_with_regime": with_label,
            "binned": ceiling(binned, targets), "binned_with_regime": ceiling(binned_regime, targets),
            "label": "[observed]", "stop": with_label["agreement_ceiling"] != 1.0}
