"""The report of #281's holdout, as markdown, by design.json's estimands.

    python experiments/m5_05/report.py <out dir>

Reads `holdout.json`, `validation.json` and the trained runs' metadata in `<out dir>`.

- loss = -net. CVaR5 is the mean of the worst ceil(0.05 x 30) = 2 daily losses: a coarse tail.
- The seed-averaged arm's day is the mean of its 5 seeds' nets; its CVaR is that averaged arm's
  tail. Per seed, the CVaR of each seed's own days, averaged over the seeds, is the expected
  tail of one deployed seed. Both are reported.
- Intervals: 2000 replicates, seed 281. Each replicate draws, per regime, a stationary bootstrap
  sequence over that regime's 10 days (blocks of mean length 2, wrapping within the regime);
  the 30 indices are shared by every arm, seed and condition. Each arm's statistic is
  recomputed on them, and the replicate's difference is the difference of those statistics.
- D < 0 in a tail is an improvement of A over B; D < 0 in the net is a cost.
- A missing seed-day fails every estimator that needs it: no number is shown for it.

Everything here is synthetic.
"""

import itertools
import json
import math
import pathlib
import sys
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REGIMES = ("calm", "stressed", "thin")
ARMS = ("adversarial", "baseline", "reference")
SEEDS = (1, 2, 3, 4, 5)
DAYS_PER_REGIME = 10
# The registered grid: every arm's five seeds on the 30 holdout days. A seed or a day absent
# from the rows is missing, never a smaller sample.
DAYS = [(regime, day) for regime in REGIMES for day in range(1, DAYS_PER_REGIME + 1)]
SHARE = 0.05
REPLICATES = 2000
MEAN_BLOCK = 2.0
SEED = 281


class Missing(Exception):
    """A seed-day an estimator needs is missing."""


def grid_names(held_out):
    """#108's points, named as #108 names them: the 12 with the lower limit rate, or the 11
    perturbed ones with the limit rate unchanged."""
    m5_03 = json.loads((HERE.parent / "m5_03" / "design.json").read_text())
    p = m5_03["perturbations"]
    keys = ("market_rate_multiplier", "cancel_rate_multiplier", "theta_multiplier",
            "limit_rate_multiplier")
    names = []
    for values in itertools.product(*(p[k] for k in keys)):
        point = dict(zip(keys, values, strict=True))
        unchanged = point["limit_rate_multiplier"] == 1.0
        unperturbed = all(v == 1.0 for v in point.values())
        if (not unchanged) if held_out else (unchanged and not unperturbed):
            names.append("m{market_rate_multiplier}-c{cancel_rate_multiplier}-"
                         "t{theta_multiplier}-l{limit_rate_multiplier}".format(**point))
    return names


def check_cells(rows):
    """Refuses a row outside the registered grid, or two rows for one cell."""
    seen = set()
    for r in rows:
        if r["arm"] not in ARMS or r["seed"] not in SEEDS or (r["regime"], r["day"]) not in DAYS:
            raise ValueError(f"a row outside the registered grid: {r['arm']} seed {r['seed']} "
                             f"{r['regime']} day {r['day']}")
        key = (r["condition"], r["arm"], r["seed"], r["regime"], r["day"])
        if key in seen:
            raise ValueError(f"two rows for {key}")
        seen.add(key)


def matrix(rows, arm, condition, days=DAYS):
    """The arm's nets under the condition, a row per registered seed and a column per
    registered day; None where a seed-day is missing, so an absent seed or day is missing
    too. None when the arm has no row at all under the condition."""
    cells = {}
    for r in rows:
        if r["arm"] == arm and r["condition"] == condition:
            cells[(r["seed"], r["regime"], r["day"])] = (
                None if r["net"] is None else float(Decimal(r["net"])))
    if not cells:
        return None
    return [[cells.get((s, *d)) for d in days] for s in SEEDS]


def stratified_indices(days, replicates=REPLICATES, mean_block=MEAN_BLOCK, seed=SEED):
    """Per replicate, per regime, a stationary bootstrap sequence over that regime's days that
    wraps within the regime and never into another; concatenated in the days' order."""
    groups = [[i for i, d in enumerate(days) if d[0] == regime] for regime in REGIMES]
    groups = [g for g in groups if g]
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty((replicates, len(days)), dtype=int)
    for b in range(replicates):
        sequence = []
        for group in groups:
            m = len(group)
            i = int(rng.integers(m))
            for k in range(m):
                if k > 0:
                    i = int(rng.integers(m)) if rng.random() < p else (i + 1) % m
                sequence.append(group[i])
        out[b] = sequence
    return out


def cvar_loss(nets, share=SHARE):
    losses = sorted((-n for n in nets), reverse=True)
    k = max(1, math.ceil(share * len(losses)))
    return sum(losses[:k]) / k


def seed_averaged(m):
    if m is None or any(v is None for row in m for v in row):
        raise Missing
    return np.mean(np.asarray(m, dtype=float), axis=0)


def per_seed(m):
    if m is None or any(v is None for row in m for v in row):
        raise Missing
    return np.asarray(m, dtype=float)


def mean_net(m, idx):
    return float(np.mean(seed_averaged(m)[idx]))


def cvar_averaged(m, idx):
    return cvar_loss(list(seed_averaged(m)[idx]))


def cvar_per_seed(m, idx):
    return float(np.mean([cvar_loss(list(row[idx])) for row in per_seed(m)]))


STATISTICS = {"mean_net": mean_net, "cvar_averaged": cvar_averaged,
              "cvar_per_seed": cvar_per_seed}


def difference(statistic, a, b, indices):
    """statistic(a) - statistic(b), with its 90 % interval over the shared replicates; a failed
    estimator when a seed-day it needs is missing."""
    f = STATISTICS[statistic]
    every = np.arange(indices.shape[1])
    try:
        point = f(a, every) - f(b, every)
        draws = [f(a, idx) - f(b, idx) for idx in indices]
    except Missing:
        return {"failed": "a seed-day is missing"}
    low, high = np.quantile(draws, [0.05, 0.95])
    return {"point": point, "low": float(low), "high": float(high)}


def below_zero(d):
    return "failed" not in d and d["high"] < 0.0


def held_out_conditions(impact_run, names=None):
    """The held-out conditions the study evaluates: the 12 points, the execution stress, and
    the impact when #282's runner ran it."""
    names = grid_names(held_out=True) if names is None else names
    return [f"held_out:{n}" for n in names] + ["execution_stress"] + (
        ["impact"] if impact_run else [])


def analyse(rows, chosen, impact_run, indices=None, names=None, in_sample_names=None):
    check_cells(rows)
    days = DAYS
    if indices is None:
        indices = stratified_indices(days)

    def m(arm, condition):
        return matrix(rows, arm, condition, days)

    def tail(condition_a, condition_b=None):
        condition_b = condition_b or condition_a
        return {s: difference(s, m("adversarial", condition_a), m("baseline", condition_b),
                              indices) for s in ("cvar_averaged", "cvar_per_seed")}

    unavailable = {"cvar_averaged": {"failed": "no selected point"},
                   "cvar_per_seed": {"failed": "no selected point"}}
    p_a = (chosen.get("adversarial") or {}).get("point")
    p_b = (chosen.get("baseline") or {}).get("point")
    primary = {
        "cost": difference("mean_net", m("adversarial", "unperturbed"),
                           m("baseline", "unperturbed"), indices),
        "tail_at_p_A": tail(f"held_out:{p_a}") if p_a else unavailable,
        "tail_at_p_B": tail(f"held_out:{p_b}") if p_b else unavailable,
        "envelope": (tail(f"held_out:{p_a}", f"held_out:{p_b}") if p_a and p_b
                     else unavailable),
    }
    a, b = primary["tail_at_p_A"]["cvar_averaged"], primary["tail_at_p_B"]["cvar_averaged"]
    if "failed" in a or "failed" in b:
        claim = "unavailable"
    elif below_zero(a) and below_zero(b):
        claim = "gain"
    else:
        claim = "no gain shown"
    conditions = held_out_conditions(impact_run, names)
    secondary = {c: {**tail(c), "mean_net": difference(
        "mean_net", m("adversarial", c), m("baseline", c), indices)} for c in conditions}
    count = sum(below_zero(v["cvar_averaged"]) for v in secondary.values())
    excluding = sum("failed" not in v["cvar_averaged"]
                    and not (v["cvar_averaged"]["low"] <= 0.0 <= v["cvar_averaged"]["high"])
                    for v in secondary.values())
    if in_sample_names is None:
        in_sample_names = grid_names(held_out=False)
    in_sample = {f"in_sample:{n}": tail(f"in_sample:{n}") for n in in_sample_names}
    every = np.arange(len(days))
    table = {}
    for condition in ["unperturbed", *conditions, *in_sample]:
        for arm in ARMS:
            if arm == "reference" and condition.startswith("in_sample:"):
                continue
            x = m(arm, condition)
            if x is None:
                x = [[None] * len(days) for _ in SEEDS]
            missing = sum(v is None for row in x for v in row)
            entry = {"seeds": len(x), "missing": missing}
            for name, f in STATISTICS.items():
                try:
                    entry[name] = f(x, every)
                except Missing:
                    entry[name] = None
            table[(condition, arm)] = entry
    return {"days": len(days), "chosen": {"p_A": p_a, "p_B": p_b,
                                          "p_ref": (chosen.get("reference") or {}).get("point")},
            "primary": primary, "claim": claim, "secondary": secondary,
            "count_below_zero": count, "count_excluding_zero": excluding,
            "conditions_evaluated": len(conditions),
            "in_sample": in_sample, "table": table}


def fmt(d):
    if "failed" in d:
        return f"failed ({d['failed']})"
    return f"{d['point']:.2f} [{d['low']:.2f}, {d['high']:.2f}]"


def number(x):
    return "failed" if x is None else f"{x:.2f}"


def render(result, validation=None, runs=(), holdout=None):
    holdout = holdout or {}
    lines = ["## #281 holdout (synthetic)", "",
             holdout.get("bounds", "") , "",
             "Synthetic evidence from the same generator as the training days: these bounded "
             "perturbations of one made-up simulator, not a market. CVaR5 is the mean of the "
             f"worst 2 of {result['days']} daily losses, a coarse tail.", ""]
    if not holdout.get("impact_run", True):
        lines += ["The impact condition was declared and not run: #282's runner was not on main "
                  "at the freeze. Nothing replaces it.", ""]
    resumed = holdout.get("resumed")
    if resumed:
        before = resumed["journal_before"]
        lines += [f"The holdout was resumed once after its first attempt was killed from outside "
                  f"with {before['lines']} jobs journaled; those lines were unchanged after the "
                  f"resume (sha256 {before['sha256']}); resume.py sha256 "
                  f"{resumed['resume_sha256']}.", ""]
    chosen = result["chosen"]
    lines += [f"Selected held-out points (#108's selection days): p_A {chosen['p_A']}, "
              f"p_B {chosen['p_B']}, reference {chosen['p_ref']}.", ""]
    lines += ["### Primary: A - B, 90 % intervals", "",
              "| estimand | seed-averaged arm | mean per-seed |", "|---|---|---|"]
    primary = result["primary"]
    lines.append(f"| cost: mean daily net, unperturbed (< 0 a cost) | {fmt(primary['cost'])} | |")
    for key, label in (("tail_at_p_A", "CVaR5 loss at p_A (< 0 an improvement)"),
                       ("tail_at_p_B", "CVaR5 loss at p_B (< 0 an improvement)"),
                       ("envelope", "envelope: A at p_A against B at p_B, two selected "
                                    "envelopes, not a same-scenario effect")):
        lines.append(f"| {label} | {fmt(primary[key]['cvar_averaged'])} "
                     f"| {fmt(primary[key]['cvar_per_seed'])} |")
    lines += ["", f"Primary claim: **{result['claim']}**. It needs both tail intervals at p_A "
              "and p_B wholly below 0; the envelope never carries it alone. An interval that "
              "holds 0 is never read as robustness, equivalence or an absence of risk.", ""]
    lines += ["### Secondary: every held-out condition, A - B", "",
              "| condition | CVaR5 loss, averaged | CVaR5 loss, per seed | mean daily net |",
              "|---|---|---|---|"]
    for condition, d in result["secondary"].items():
        lines.append(f"| {condition} | {fmt(d['cvar_averaged'])} | {fmt(d['cvar_per_seed'])} "
                     f"| {fmt(d['mean_net'])} |")
    lines += ["", f"{result['count_excluding_zero']} of the {result['conditions_evaluated']} "
              "held-out conditions evaluated have a seed-averaged tail interval that excludes 0, "
              f"as the design counts them; {result['count_below_zero']} of them wholly below 0 "
              "(an improvement of A). No multiplicity correction.", ""]
    lines += ["### Every arm and condition", "",
              "| condition | arm | seeds | missing seed-days | mean net | CVaR5, averaged "
              "| CVaR5, per seed |", "|---|---|---:|---:|---:|---:|---:|"]
    for (condition, arm), e in result["table"].items():
        lines.append(f"| {condition} | {arm} | {e['seeds']} | {e['missing']} "
                     f"| {number(e['mean_net'])} | {number(e['cvar_averaged'])} "
                     f"| {number(e['cvar_per_seed'])} |")
    lines += ["", "The reference is #102's frozen residual SAC; no claim is attached to it.", ""]
    lines += ["### In sample: the perturbed training points (no claim)", "",
              "| point | CVaR5 loss, averaged | CVaR5 loss, per seed |", "|---|---|---|"]
    for condition, d in result["in_sample"].items():
        lines.append(f"| {condition} | {fmt(d['cvar_averaged'])} | {fmt(d['cvar_per_seed'])} |")
    if validation is not None:
        check = validation["reproduction"]
        lines += ["", "### B's reproduction check (validation days)", "",
                  f"Passed: {check['passed']}; statistic {check.get('statistic')}, #102's "
                  f"{check.get('reference_mean')}, tolerance {check['tolerance']:.4f}."]
    if runs:
        lines += ["", "### Training", "",
                  "| run | worst point per scoring | its worst daily loss | train s | scoring s |",
                  "|---|---|---|---:|---:|"]
        for run in runs:
            worst = [p.get("worst") for p in run["phases"][1:]]
            losses = [f"{p['scores'][p['worst']]['worst_loss']:.2f}"
                      for p in run["phases"][1:]]
            lines.append(f"| {run['study_arm']}-seed{run['seed']} | {', '.join(worst)} "
                         f"| {', '.join(losses)} | {run['train_seconds']} "
                         f"| {run['score_seconds']} |")
    return "\n".join(lines) + "\n"


def main():
    out = pathlib.Path(sys.argv[1])
    holdout = json.loads((out / "holdout.json").read_text())
    validation_path = out / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    runs = [json.loads(p.read_text()) for p in sorted((out / "checkpoints").glob("*.json"))]
    result = analyse(holdout["rows"], holdout["chosen"], holdout["impact_run"])
    print(render(result, validation, runs, holdout))


if __name__ == "__main__":
    main()
