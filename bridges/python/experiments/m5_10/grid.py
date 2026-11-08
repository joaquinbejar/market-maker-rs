"""The perturbation grid of #108 part 2 and its plausibility and identifiability rules.

SYNTHETIC. No day is generated here: the rules take statistics a caller measured.
"""

import itertools
import json

import numpy as np

MARKET, LIMIT, BRANCHING = (1, 1.5, 2), (1, 0.7), (1, 1.2)
NEXT_IMPACT = {0.0: 0.25, 0.25: 1.0, 1.0: 2.0}
TOLERANCE = 0.10
ZERO_IMPACT_SLOPE = 0.025
MARKET_INDEX = 2  # limit, cancel, market, as the crate's weight matrices


def points(impact):
    """The 24 points of a setting in the design's order, the unperturbed point first."""
    return [{"market": m, "limit": l, "branching": b, "impact": i}
            for m, l, b, i in itertools.product(MARKET, LIMIT, BRANCHING, (impact, NEXT_IMPACT[impact]))]


def fixed_stress(impact):
    return {"market": 2, "limit": 0.7, "branching": 1.2, "impact": NEXT_IMPACT[impact]}


def impact_only(impact):
    return {"market": 1, "limit": 1, "branching": 1, "impact": NEXT_IMPACT[impact]}


def branching_matrix(weights, scale=1.0):
    """Sum_u alpha_u, the matrix HawkesParameters::new checks."""
    return scale * np.sum(np.asarray(weights, float), axis=0)


def spectral_radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def plausible(weights, point):
    return spectral_radius(branching_matrix(weights, point["branching"])) < 1


def rate_formula(rates, weights, point, shares):
    """The market entry of (I - K)^-1 mu, with time-averaged open shares of the queue."""
    mu = np.array([rates["limit"] * point["limit"] * shares["below_cap"],
                   rates["cancel_per_lot"] * shares["mean_queue"],
                   rates["market"] * point["market"] * shares["non_empty"]])
    response = np.linalg.solve(np.eye(3) - branching_matrix(weights, point["branching"]), mu)
    return float(response[MARKET_INDEX])


def rate_passes(realised, realised_unperturbed, formula, formula_unperturbed):
    ratio = (realised / realised_unperturbed) / (formula / formula_unperturbed)
    return abs(ratio - 1) <= TOLERANCE


def slope(mid_changes, signed_orders):
    """OLS slope of the per-row mid change on the row's signed market orders."""
    x = np.asarray(signed_orders, float)
    y = np.asarray(mid_changes, float)
    x = x - x.mean()
    return float(x @ (y - y.mean()) / (x @ x))


def impact_passes(estimate, impact_ticks):
    if impact_ticks == 0:
        return abs(estimate) <= ZERO_IMPACT_SLOPE
    return abs(estimate / impact_ticks - 1) <= TOLERANCE


def day_config(causal, point, name, days, offset):
    """The `causal` section hawkes_days reads for one point: one setting, every day in `holdout`."""
    out = json.loads(json.dumps(causal))
    flow = out["flow"]
    flow["rates"]["market"] *= point["market"]
    flow["rates"]["limit"] *= point["limit"]
    flow["weights"] = (np.asarray(flow["weights"], float) * point["branching"]).tolist()
    out["settings"] = [{"name": name, "impact_ticks": point["impact"]}]
    out["days"] = days
    out["splits"] = {"train": [days + 1, days + 1], "validation": [days + 2, days + 2], "holdout": [1, days]}
    out["seed_offset"] = offset
    return out
