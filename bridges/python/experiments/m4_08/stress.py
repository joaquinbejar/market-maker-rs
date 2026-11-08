"""The second, margin-stress holdout of #105: the frozen runs on new days, read under
margin-rs's maintenance rule. It never revises the first holdout's verdict.

In the experiment's own venv (`omnisafe-lock.txt`, `MUJOCO_GL=disable`), from `bridges/python`:

    python experiments/m4_08/stress.py days <qr_days output> > experiments/m4_08/stress_digests.json
    python experiments/m4_08/stress.py shocks <stress data dir> <shocks dir> > experiments/m4_08/stress_shocks.json
    python experiments/m4_08/stress.py holdout <stress data dir> <shocks dir> <run out dir> <stress out dir> <commit> <frozen sha256>

`days` takes the digests `qr_days` printed for `stress_days.json` (days 1 to 29 of #102's
regimes and seed rule), checks that days 1 to 24 are #102's, and keeps days 25 to 29.
`shocks` writes their shocked copies, as `shocks.py` does for #102's days. `holdout` reads
them once: every frozen run of the first holdout's manifest, unchanged, with no-trade and
the selected Avellaneda-Stoikov points, at the frozen fee, unshocked and under each shock.
Each step's delivered position, observed mid and net result go to
`mm_rl_bridge.maintenance_path`, which assesses the account under margin-rs's `linear-im/1`
(`stress_design.json`). Everything here is synthetic.
"""

import json
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import shocks as shocking  # noqa: E402
from costs import POSITION, costs  # noqa: E402
from mm_rl_bridge import maintenance_path  # noqa: E402

STRESS_DESIGN = HERE / "stress_design.json"
STRESS_DIGESTS = HERE / "stress_digests.json"
STRESS_SHOCKS = HERE / "stress_shocks.json"
SPLIT = "holdout"
STATES = ("healthy", "reduce_only", "default")


def load_stress():
    design = json.loads(STRESS_DESIGN.read_text())
    digests = {(SPLIT, d["regime"], d["day"], None): d["dataset_digest"]
               for d in json.loads(STRESS_DIGESTS.read_text())["days"]}
    for s in json.loads(STRESS_SHOCKS.read_text())["shocks"]:
        digests[(SPLIT, s["regime"], s["day"], s["shock"])] = s["dataset_digest"]
    return design, digests


def new_days(generated, base):
    """Days after #102's, after checking that every one of #102's days is the same."""
    old = {(d["regime"], d["day"]): d for d in base["days"]}
    keys = ("seed", "rows", "dataset_digest", "external_digest", "parameters_fingerprint")
    for d in generated["days"]:
        known = old.get((d["regime"], d["day"]))
        if known is not None and any(d[k] != known[k] for k in keys):
            raise ValueError(f"{d['regime']} day {d['day']} is not #102's")
    if not set(old) <= {(d["regime"], d["day"]) for d in generated["days"]}:
        raise ValueError("the generated days do not cover #102's")
    last = max(day for _, day in old)
    return [d for d in generated["days"] if d["day"] > last]


def summary(path, positions):
    """A day's margin reading: steps per state; entries, each step out of Healthy (into
    ReduceOnly or straight into Default), where a closeout would start; entries into
    Default from any other state; and the largest shortfall. A recovery to Healthy and a
    new step out of it is a new entry."""
    counts = dict.fromkeys(STATES, 0)
    entries = defaults = 0
    previous, worst = "healthy", Decimal(0)
    for state, _, shortfall in path:
        counts[state] += 1
        entries += previous == "healthy" and state != "healthy"
        defaults += previous != "default" and state == "default"
        previous = state
        worst = max(worst, Decimal(shortfall))
    steps = len(path)
    return {"steps": counts, "not_healthy_share": (steps - counts["healthy"]) / steps,
            "entries": entries, "entries_default": defaults, "largest_shortfall": str(worst),
            "max_abs_position": str(max((abs(p) for p in positions), default=Decimal(0)))}


def margin_inputs(info, rows, tick):
    """A step's delivered position (signed), the observed row's mid and the economic net
    result: fees included, penalties excluded (the kernel's net never holds them), and the
    terminal close, which only the last step has, taken out."""
    raw = info["raw_observation"][POSITION]
    position = Decimal(str(raw)) if raw is not None else Decimal(0)
    row = rows[info["row"]] if info["row"] is not None else rows[0]
    close = info.get("terminal_close")
    net = Decimal(info["net"]) - (Decimal(str(close)) if close is not None else Decimal(0))
    return position, Decimal(row[1] + row[3]) / 2 * tick, net


def episode(policy, env, rows, seed, design, stress):
    """One evaluated day of the first holdout's episode, with the margin reading beside it."""
    tick = Decimal(design["kernel"]["tick_size"])
    inputs = []
    recording = _Recording(policy, env, rows, tick, inputs)
    try:
        day = run.episode(recording, env, rows, seed, design)
    finally:
        recording.restore(env)
    account = stress["account"]
    path = maintenance_path(account["capital"], account["scale"],
                            [(str(q), str(m), str(n)) for q, m, n in inputs])
    return day | {"margin": summary(path, [q for q, _, _ in inputs])}


class _Recording:
    """The policy, unchanged, recording each step's margin inputs from the step's info."""

    def __init__(self, policy, env, rows, tick, inputs):
        self.policy, self.rows, self.tick, self.inputs = policy, rows, tick, inputs
        self._own = "step" in vars(env)
        self._step = env.step
        env.step = self._record

    def _record(self, action):
        out = self._step(action)
        self.inputs.append(margin_inputs(out[4], self.rows, self.tick))
        return out

    def restore(self, env):
        if self._own:
            env.step = self._step
        else:
            del env.step

    def predict(self, observation):
        return self.policy.predict(observation)


def evaluate_job(job):
    torch.set_num_threads(1)
    design, _ = run.load_design()
    stress, digests = load_stress()
    env, rows = run.build_env(job, digests, True)
    try:
        kind = job["policy"]["kind"]
        policy = (run.Constant(run.HOLD) if kind == "hold" else run.ZeroResidual()
                  if kind == "zero_residual" else
                  run.Learned(job["policy"]["run_dir"], job["policy"]["model"],
                              job["policy"]["config_sha256"]))
        days = [episode(policy, env, rows, seed, design, stress)
                for seed in design["evaluation"]["evaluation_seeds"]]
    finally:
        env.close()
    nets = [Decimal(d["net"]) for d in days]
    keep = ("arm", "candidate", "fee", "regime", "day", "split", "shock", "penalty")
    return {k: job.get(k) for k in keep} | {
        "daily_value": str(sum(nets, Decimal(0)) / len(nets)), "episodes": days}


def holdout(ctx, run_out, stress_out, frozen_sha256):
    """The second holdout, once, from the first's manifest and this design's hashes."""
    path = run_out / "frozen.json"
    if run.sha256(path) != frozen_sha256:
        raise ValueError("the frozen manifest is not the one given")
    stress_out.mkdir(parents=True, exist_ok=True)
    target = stress_out / "stress_holdout.json"
    marker = stress_out / "stress_holdout.started"
    if target.exists():
        raise ValueError("the margin-stress holdout was evaluated already")
    if marker.exists():
        raise ValueError("the margin-stress holdout was started already; a failed or "
                         "concurrent attempt is not run again, and recovery is by hand")
    frozen = json.loads(path.read_text())
    for key, name in (("design_sha256", "design.json"), ("shocks_sha256", "shocks.json"),
                      ("lock_sha256", "omnisafe-lock.txt")):
        if frozen[key] != run.sha256(HERE / name):
            raise ValueError(f"{name} changed after the freeze")
    for m in frozen["runs"]:
        run.check_run(m, "the freeze")
    stress, _ = load_stress()
    days = list(range(stress["days"][0], stress["days"][1] + 1))
    shocks = [None] + run.load_design()[0]["shocks"]["shocks"]
    fee = frozen["fee"]
    jobs = ctx.classical_jobs(fee, SPLIT, days, shocks, frozen["classical_selected"])
    jobs += ctx.learned_jobs(frozen["runs"], fee, SPLIT, days, shocks)
    # Before any job reads a holdout day: created only if absent, so a concurrent attempt
    # fails here, and kept if the evaluation fails, so it is never run again on its own.
    with open(marker, "x") as out:
        out.write(json.dumps({"commit": ctx.commit, "frozen_sha256": frozen_sha256,
                              "jobs": len(jobs), "at": time.time()}))
    started = time.perf_counter()
    report = {"step": "stress_holdout", "commit": ctx.commit, "frozen_sha256": frozen_sha256,
              "stress_design_sha256": run.sha256(STRESS_DESIGN),
              "stress_digests_sha256": run.sha256(STRESS_DIGESTS),
              "stress_shocks_sha256": run.sha256(STRESS_SHOCKS),
              "code_sha256": run.sha256(HERE / "stress.py"), "fee": fee, "days": days,
              "rows": run.pool(jobs, evaluate_job, ctx.parallel, "stress holdout")}
    report["seconds"] = round(time.perf_counter() - started, 1)
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def write_shocks(data, out, design, stress_digests):
    """The shocked copies of the new days, after checking each source day's digest."""
    spec = design["shocks"]
    records = []
    for record in stress_digests["days"]:
        source = pathlib.Path(data) / SPLIT / record["regime"] / f"day-{record['day']}.json"
        rows = [tuple(row) for row in json.loads(source.read_text())]
        if shocking.digest_of(design, rows) != record["dataset_digest"]:
            raise ValueError(f"{source} is not the day of stress_digests.json")
        for shock in spec["shocks"]:
            copy, delta = shocking.shocked(rows, spec["at_row"], shock)
            target = pathlib.Path(out) / SPLIT / record["regime"] / \
                f"day-{record['day']}-shock{shock}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([list(row) for row in copy]))
            records.append({"regime": record["regime"], "day": record["day"], "split": SPLIT,
                            "shock": shock, "delta_ticks": delta,
                            "source_digest": record["dataset_digest"],
                            "dataset_digest": shocking.digest_of(design, copy),
                            "rows": len(copy)})
    return records


def main():
    step = sys.argv[1]
    if step == "days":
        generated = json.loads(pathlib.Path(sys.argv[2]).read_text())
        base = json.loads((HERE.parent / "m4_05" / "digests.json").read_text())
        print(json.dumps({"days": new_days(generated, base)}, indent=2))
    elif step == "shocks":
        design, _ = run.load_design()
        stress_digests = json.loads(STRESS_DIGESTS.read_text())
        print(json.dumps({"shocks": write_shocks(sys.argv[2], sys.argv[3], design,
                                                 stress_digests)}, indent=2))
    elif step == "holdout":
        data, shocks_dir, run_out, stress_out, commit, frozen = sys.argv[2:8]
        ctx = run.Context(data, shocks_dir, run_out, commit)
        report = holdout(ctx, pathlib.Path(run_out), pathlib.Path(stress_out), frozen)
        print(json.dumps({"step": "stress_holdout", "seconds": report["seconds"],
                          "rows": len(report["rows"])}))
    else:
        raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()
