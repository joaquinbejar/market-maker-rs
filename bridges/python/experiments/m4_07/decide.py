"""The go or no-go of #104 for one target on one behaviour's log, fixed before any log exists.

It reads diagnostics only. A target's true value is never an input: the study reports it
beside the decision, as a check of the diagnostics, in a simulator whose rows do not react to
orders. A go says only that these checks found nothing against reading the estimates for that
target on that log. It does not say that they are right, or that the target is good.
"""

MIN_EFFECTIVE_SAMPLE = 0.05
MAX_WEIGHT = 0.05
LOW_PROPENSITY = 0.0001

REASONS = (
    "no_recorded_propensity",
    "out_of_support",
    "low_effective_sample",
    "heavy_weights",
    "estimators_disagree",
    "no_cross_check",
)


def decide(diagnostics):
    """`("go", [])` or `("no_go", reasons)`, the reasons in the order of `REASONS`.

    `diagnostics` holds:
    - `propensity_recorded`: whether the log records the behaviour's probabilities;
    - `out_of_support_by_construction`: a target with no probability against the behaviour;
    - `zero_propensity`: decisions where the target's action has behaviour probability 0;
    - `effective_sample`: the effective sample as a share of the decisions, or `None`;
    - `max_weight`: the largest normalized weight, or `None`;
    - `cross_check`: whether the importance and doubly robust intervals overlap, or `None`.
    """
    reasons = set()
    if not diagnostics["propensity_recorded"]:
        reasons.add("no_recorded_propensity")
    if diagnostics["out_of_support_by_construction"] or diagnostics["zero_propensity"] > 0:
        reasons.add("out_of_support")
    effective = diagnostics["effective_sample"]
    if effective is None or effective < MIN_EFFECTIVE_SAMPLE:
        reasons.add("low_effective_sample")
    weight = diagnostics["max_weight"]
    if weight is not None and weight > MAX_WEIGHT:
        reasons.add("heavy_weights")
    if diagnostics["cross_check"] is False:
        reasons.add("estimators_disagree")
    if diagnostics["cross_check"] is None and not reasons:
        reasons.add("no_cross_check")
    if not reasons:
        return "go", []
    return "no_go", [reason for reason in REASONS if reason in reasons]


def stop_on_validation(effective_by_target, behaviour_greedy="as_grid"):
    """The validation stop rule: True when no target other than the behaviour's greedy rule
    reaches the minimum effective sample, so the holdout logs stay closed."""
    return not any(
        effective is not None and effective >= MIN_EFFECTIVE_SAMPLE
        for target, effective in effective_by_target.items()
        if target != behaviour_greedy
    )
