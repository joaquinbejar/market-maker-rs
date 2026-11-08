"""#282: #108's robustness study with the agent's own price impact, in rows that react to its
fills. The design is design.json, fixed before any run.

    python experiments/m5_04/run.py rate <#102 run dir> <out dir> <commit>

`rate` times one #102 training day (calm, day 1: neither a selection nor a holdout day) per arm
at kappa 0 and at 2 kappa0, with the reactions it makes, before any slot is asked for.

Synthetic data only.
"""

import importlib.util
import json
import math
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
M5_03 = HERE.parent / "m5_03"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m503 = _module("m5_03_run", M5_03 / "run.py")
base = m503.base

from mm_rl_bridge import (  # noqa: E402
    MaskableMarketMakingEnv,
    ReactiveQrMarket,
    ResidualMarketMakingEnv,
)
from mm_rl_bridge.training import ZeroResidual  # noqa: E402

MASK64 = (1 << 64) - 1
DAY_MS = 86_400_000
SEED_STRIDE = 1_000_003


# The reaction, as design.json states it.


def fnv1a64(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & MASK64
    return value


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def unit(self):
        return (self.next_u64() >> 11) / float(1 << 53)


def impact_rng(day_seed):
    return SplitMix64(fnv1a64(day_seed.to_bytes(8, "little") + b"impact-282"))


def moves_for(kappa, filled, rng):
    """Reference moves for one step's fills: netted, against the agent's net position change,
    kappa * |n| ticks with the fraction drawn; one draw per step with n != 0."""
    n = filled["sell"] - filled["buy"]
    if n == 0:
        return 0
    total = kappa * abs(n)
    whole = math.floor(total)
    if rng.unit() < total - whole:
        whole += 1
    return whole if n > 0 else -whole


def day_market(regime, m4_05, seed, day):
    sim = m4_05["simulator"]
    episode = {"seed": seed, "reference_ticks": sim["reference_ticks"],
               "start_ms": (day - 1) * DAY_MS, "sample_ms": sim["sample_ms"],
               "rows": sim["rows_per_day"]}
    return ReactiveQrMarket(json.dumps(regime), json.dumps(m4_05["kernel"]), json.dumps(episode))


def predict(policy, env, observation):
    if isinstance(env, MaskableMarketMakingEnv):
        action, _ = policy.predict(observation, deterministic=True,
                                   action_masks=env.action_masks())
    else:
        action, _ = policy.predict(observation, deterministic=True)
    return action


class ReplayMismatch(RuntimeError):
    """A replay that does not reach the pre-reaction step: the run stops."""


def reactive_episode(policy, make_env, market, kappa, rng, seed, m4_05):
    """One evaluated day in rows that react to the agent's fills, with base.episode's result
    and the reactions. After a reaction the environment is rebuilt on the new rows and the
    logged actions are replayed without querying the policy; the replay must reach the same
    observation, net and penalties."""
    env = make_env(market.rows())
    log = []
    observation, _ = env.reset(seed=seed)
    refusals = fallbacks = clipped = 0
    reactions = moves = replayed = 0
    try:
        while True:
            action = predict(policy, env, observation)
            observation, _, terminated, truncated, info = env.step(action)
            log.append(action)
            refusals += info["refused_because"] is not None
            residual = info.get("residual")
            if residual is not None:
                fallbacks += residual["fallback"] is not None
                clipped += any(residual["clips"])
            done = terminated or truncated
            if kappa > 0 and not done:
                step_moves = moves_for(kappa, info["filled_lots"], rng)
                if step_moves:
                    market.react(info["row"], step_moves)
                    env.close()
                    env = make_env(market.rows())
                    again, _ = env.reset(seed=seed)
                    for logged in log:
                        again, _, _, _, check = env.step(logged)
                    if not (np.array_equal(again, observation) and check["net"] == info["net"]
                            and check["penalties"] == info["penalties"]):
                        raise ReplayMismatch(f"step {len(log)} after a reaction")
                    observation = again
                    reactions += 1
                    moves += abs(step_moves)
                    replayed += len(log)
            if done:
                break
        attribution = dict(env.attribution())
    finally:
        env.close()
    limits = m4_05["execution"]["limits"]
    peak_notional = max(
        (abs(Decimal(after)) * Decimal(price) for _, price, after in attribution["positions"]),
        default=Decimal(0),
    )
    return {"steps": len(log), "net": info["net"], "penalties": info["penalties"],
            "attribution": {k: v for k, v in attribution.items() if k != "positions"},
            "positions": attribution["positions"], "refusals": refusals,
            "residual_fallbacks": fallbacks, "residual_clipped": clipped,
            "peak_notional": str(peak_notional), "reactions": reactions, "moves": moves,
            "replayed_steps": replayed, "stopped": market.stopped(),
            "limit_breach": Decimal(attribution["max_abs_position"])
            > Decimal(limits["max_position"]) or peak_notional > Decimal(limits["max_notional"])}


# The arms, as #108 runs them.


def arm_policy(arm, regime, day, rows, frozen, runs, m4_05):
    """The arm's policy and the environment factory over given rows, as #108's run_arm."""
    selected = {tuple(k.split("/")): v for k, v in frozen["classical_selected"].items()}
    grid = m4_05["arms"]["maskable_ppo"]["grid"]
    main_mask = base.base_config(m4_05, "main", "mask", grid)
    bounds = m4_05["arms"]["sac_residual_avellaneda_stoikov"]["bounds"]
    if arm in ("no_trade", "fixed_spread", "maskable_ppo"):
        def make_env(new_rows, config=main_mask):
            return MaskableMarketMakingEnv(config, [tuple(r) for r in new_rows])
    else:
        kind = "glft" if arm == "glft" else "avellaneda_stoikov"
        config = base.residual_config(m4_05, "main", kind, selected[(regime, kind)], day, bounds)

        def make_env(new_rows, config=config):
            return ResidualMarketMakingEnv(config, [tuple(r) for r in new_rows])
    if arm == "no_trade":
        policy = base.Constant([0, 0, 0, 0, 0])
    elif arm == "fixed_spread":
        index = grid["bid_offsets"].index(selected[(regime, "fixed_spread")]["offset_ticks"])
        policy = base.Constant([1, index, index, 0, 0])
    elif arm in ("avellaneda_stoikov", "glft"):
        policy = ZeroResidual()
    else:
        probe = make_env(rows)
        try:
            policy = m503.load_frozen(runs[arm][0], probe, regime)
        finally:
            probe.close()
    return policy, make_env


def rate(run_dir, out_dir, commit):
    torch.set_num_threads(1)
    design = json.loads((HERE / "design.json").read_text())
    m4_05 = json.loads((HERE.parent / "m4_05" / "design.json").read_text())
    frozen = json.loads((pathlib.Path(run_dir) / "frozen.json").read_text())
    runs = m503.learned_runs(frozen, run_dir)
    regime = m4_05["simulator"]["regimes"][0]
    day = 1
    seed = SEED_STRIDE * 1 + day
    kappa0 = design["impact_scale"]["values"][regime["name"]]["kappa0"]
    out = {}
    for arm in m503.ARMS:
        rows = day_market(regime, m4_05, seed, day).rows()
        policy, make_env = arm_policy(arm, regime["name"], day, rows, frozen, runs, m4_05)
        for label, kappa in (("kappa 0", 0.0), ("kappa 2 kappa0", 2 * kappa0)):
            market = day_market(regime, m4_05, seed, day)
            started = time.perf_counter()
            result = reactive_episode(policy, make_env, market, kappa, impact_rng(seed),
                                      design["evaluation_seed"], m4_05)
            seconds = time.perf_counter() - started
            out.setdefault(arm, {})[label] = {
                "seconds": round(seconds, 2), "steps": result["steps"],
                "reactions": result["reactions"], "moves": result["moves"],
                "replayed_steps": result["replayed_steps"], "net": result["net"]}
            print(arm, label, out[arm][label], file=sys.stderr, flush=True)
    report = {"step": "rate", "commit": commit, "regime": regime["name"], "day": day,
              "seed": seed, "kappa0": kappa0, "arms": out, "versions": base.versions()}
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / "rate.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    step = sys.argv[1]
    if step == "rate":
        report = rate(*sys.argv[2:5])
        print(json.dumps(report["arms"], indent=2))
    else:
        raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()
