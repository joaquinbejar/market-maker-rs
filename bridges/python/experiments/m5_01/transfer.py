"""The transfer of #102's frozen arms to #106's generators (`design.json`, main_study transfer
and its pins). REAL DATA fits and their simulations; #102's arms were trained on synthetic days.

    python experiments/m5_01/transfer.py run <frozen dir> <out json> <regime> <input.json>...

Each input is a trajectory or an MDQR draw file; each output row holds, per policy, the net of
the 12 blocks of 60 rows, the fills and the replaced configuration's fingerprint.
"""

import copy
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2  # noqa: E402

ROWS, BLOCK_ROWS, ROW_NS, MID_START = 720, 60, 10_000_000_000, 20_000
FILL_MODEL = {"kind": "queue_position_rational", "lots": 1, "per_ms": 5000, "min_queue_ms": 0}
FROZEN_SHA256 = "9de757b5bf76274882306c9a9c3d9257389deca64d24074948d1a95646c19458"
SELECTED = {"maskable_ppo": 0, "sac_residual_avellaneda_stoikov": 2}
REGIME = "calm"
REGIMES = ("calm", "stressed", "thin")
EVALUATION_SEED = 101


class FailedDraw(ValueError):
    """Rows the pins do not allow: a failed draw, never repaired."""


def rows_of(samples, references, directions, first=1):
    """The pins' rows from samples [ms, bid, ask, spread] and the signed moves. `first` is the
    grid's first multiple of 10 s: 0 for the capture, 1 for a generator."""
    samples = [s for s in samples if s[0] % 10_000 == 0][:ROWS]
    if [s[0] for s in samples] != [(i + first) * 10_000 for i in range(ROWS)]:
        raise FailedDraw(f"{len(samples)} rows on the 10 s grid, not {ROWS}")
    moves = [(r[0], d) for r, d in zip(references, directions, strict=True) if r[1]]
    out, at, signed = [], 0, 0
    for index, (ms, bid, ask, spread) in enumerate(samples[:ROWS]):
        while at < len(moves) and moves[at][0] <= ms:
            signed += 1 if moves[at][1] > 0 else -1
            at += 1
        for queue in (bid[0], ask[0], spread):
            if int(queue) != queue:
                raise FailedDraw(f"a value that is not an integer: {queue}")
        half = max(int(spread), 1)
        mid = MID_START + 2 * signed
        if mid - half <= 0:
            raise FailedDraw("a bid that is not positive")
        out.append(((index + 1) * ROW_NS, mid - half, max(int(bid[0]), 1), mid + half,
                    max(int(ask[0]), 1)))
    return out


def block_nets(cumulative):
    """Net per 60 rows from the cumulative net after each of the 719 steps a 720 row episode
    has (a step moves from one row to the next); the last block ends at the episode's last
    value, which holds the terminal mark."""
    if not cumulative:
        raise FailedDraw("an episode with no step")
    cumulative = list(cumulative) + [cumulative[-1]] * (ROWS - 1 - len(cumulative))
    ends = [cumulative[(k + 1) * BLOCK_ROWS - 2] for k in range(ROWS // BLOCK_ROWS)]
    return [float(b - a) for a, b in zip([0.0] + ends[:-1], ends, strict=True)]


def _m4_05():
    spec = importlib.util.spec_from_file_location("m4_05_run", HERE.parent / "m4_05" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spec_matches(trained, here):
    """Equal, or differing only in the action version 1 to 2 and the identity that hashes it,
    the change #104 and #108 accepted with hedge actions off."""
    trained, here = list(trained), list(here)
    return trained == here or (trained[0] == here[0] and trained[2:4] == here[2:4]
                               and trained[1] == 1 and here[1] == 2)


def replaced(config):
    out = copy.deepcopy(config)
    out["fill_model"] = dict(FILL_MODEL)
    return out


def run_policy(policy, env, masked):
    observation, _ = env.reset(seed=EVALUATION_SEED)
    cumulative = []
    while True:
        if masked:
            action, _ = policy.predict(observation, deterministic=True, action_masks=env.action_masks())
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        cumulative.append(float(info["net"]))
        if terminated or truncated:
            break
    attribution = dict(env.attribution())
    attribution.pop("positions", None)
    return {"blocks": block_nets(cumulative), "steps": len(cumulative), "attribution": attribution,
            "fingerprint": env.bridge_info["kernel_fingerprint"]}


def arms(frozen_dir, regime=REGIME):
    """The 13 policies as (name, config, masked, loader), at a regime's configuration."""
    run = _m4_05()
    manifest_path = pathlib.Path(frozen_dir) / "frozen.json"
    if run.sha256(manifest_path) != FROZEN_SHA256:
        raise ValueError("the frozen manifest is not the pinned one")
    frozen = json.loads(manifest_path.read_text())
    chosen = [r for r in frozen["runs"] if r["variant"] == "main" and r["penalty"] is None
              and r["set"] == SELECTED.get(r["arm"])]
    if len(chosen) != 10:
        raise ValueError(f"{len(chosen)} learned arms, not 10")
    ppo = next(r for r in chosen if r["arm"] == "maskable_ppo")["configs"][regime]
    sac = next(r for r in chosen if r["arm"] != "maskable_ppo")["configs"][regime]
    if sac["residual"]["baseline"]["session_start_ms"] != 0:
        raise ValueError("the frozen baseline's session does not start at the window")
    out = []
    fixed = frozen["classical_selected"][f"{regime}/fixed_spread"]
    index = ppo["bid_offsets"].index(fixed["offset_ticks"])
    out.append(("fixed_spread", replaced(ppo), True, lambda env: run.Constant([1, index, index, 0, 0])))
    for kind in ("avellaneda_stoikov", "glft"):
        config = replaced(sac)
        config["residual"]["baseline"] = run.baseline_config(
            kind, frozen["classical_selected"][f"{regime}/{kind}"], 1)
        out.append((kind, config, False, lambda env: run.ZeroResidual()))

    def loader(metadata):
        def load(env):
            path = pathlib.Path(frozen_dir) / "checkpoints" / pathlib.Path(metadata["path"]).name
            zipped = path.with_suffix(".zip")
            if run.sha256(zipped) != metadata["checkpoint_sha256"]:
                raise ValueError("a checkpoint changed")
            if not spec_matches(metadata["spec"], env.spec_id):
                raise ValueError("a checkpoint of another specification: stop and report")
            if env.config.get("hedge_actions"):
                raise ValueError("a version 1 checkpoint runs only with hedge actions off")
            kept = {k: v for k, v in metadata["configs"][regime].items() if k != "fill_model"}
            if kept != {k: v for k, v in env.config.items() if k != "fill_model"}:
                raise ValueError("a checkpoint of another configuration: stop and report")
            model_class = run.MaskablePPO if metadata["algorithm"] == "maskable_ppo" else run.SAC
            model = model_class.load(zipped, device="cpu")
            model.policy.set_training_mode(False)
            return model
        return load

    for metadata in chosen:
        out.append((f"{metadata['arm']}-{metadata['seed']}", replaced(metadata["configs"][regime]),
                    metadata["arm"] == "maskable_ppo", loader(metadata)))
    return run, out


def row_statistics(rows):
    """Descriptive: realised 10 s mid volatility in kernel ticks, median spread and touch lots."""
    mids = np.array([(r[1] + r[3]) / 2 for r in rows])
    return {"volatility_10s": float(np.std(np.diff(mids))),
            "median_spread": float(np.median([r[3] - r[1] for r in rows])),
            "median_touch_lots": float(np.median([r[2] for r in rows] + [r[4] for r in rows]))}


def evaluate(frozen_dir, rows, regime=REGIME):
    """Every policy on one set of rows, by name."""
    run, policies = arms(frozen_dir, regime)
    out = {}
    for name, config, masked, load in policies:
        make = run.MaskableMarketMakingEnv if masked else run.ResidualMarketMakingEnv
        env = make(config, rows)
        try:
            out[name] = run_policy(load(env), env, masked)
        finally:
            env.close()
    out["_rows"] = row_statistics(rows)
    return out


def consistency(frozen_dir, rows_dir):
    """One checkpoint per family at #102's own execution must give #102's recorded calm
    validation value exactly (design pins.consistency). Raises on any mismatch."""
    run = _m4_05()
    frozen_dir, rows_dir = pathlib.Path(frozen_dir), pathlib.Path(rows_dir)
    manifest = json.loads((frozen_dir / "frozen.json").read_text())
    selection = frozen_dir / "selection.json"
    if run.sha256(selection) != manifest["selection_sha256"]:
        raise ValueError("selection.json is not the frozen one")
    recorded = {(r["arm"], r["candidate"], r["day"]): r["daily_value"]
                for r in json.loads(selection.read_text())["learned_validation"]
                if r["regime"] == REGIME and r["variant"] == "main"}
    design, _ = run.load_design()
    out = []
    for arm, chosen_set in SELECTED.items():
        metadata = next(r for r in manifest["runs"] if r["arm"] == arm and r["seed"] == 1
                        and r["set"] == chosen_set and r["variant"] == "main" and r["penalty"] is None)
        days = sorted(d for a, c, d in recorded if a == arm and c == f"set{chosen_set}-seed1")
        if not days:
            raise ValueError(f"no recorded validation value for {arm}")
        masked = arm == "maskable_ppo"
        for day in days:
            config = copy.deepcopy(metadata["configs"][REGIME])
            if not masked:
                config["residual"]["baseline"]["session_start_ms"] = (day - 1) * run.DAY_MS
            rows = [tuple(r) for r in json.loads(
                (rows_dir / "validation" / REGIME / f"day-{day}.json").read_text())]
            make = run.MaskableMarketMakingEnv if masked else run.ResidualMarketMakingEnv
            env = make(config, rows)
            try:
                if env.bridge_info["kernel_fingerprint"] != metadata["kernel_fingerprint"]:
                    raise ValueError(f"{arm}: another kernel fingerprint")
                zipped = frozen_dir / "checkpoints" / f"{pathlib.Path(metadata['path']).name}.zip"
                if run.sha256(zipped) != metadata["checkpoint_sha256"]:
                    raise ValueError("a checkpoint changed")
                model_class = run.MaskablePPO if masked else run.SAC
                model = model_class.load(zipped, device="cpu")
                model.policy.set_training_mode(False)
                value = run.episode(model, env, EVALUATION_SEED, design)["net"]
            finally:
                env.close()
            expected = recorded[(arm, f"set{chosen_set}-seed1", day)]
            out.append({"arm": arm, "day": day, "recorded": expected, "value": value})
            if run.Decimal(value) != run.Decimal(expected):
                raise ValueError(f"inconsistent: {out[-1]}")
    return out


def inputs_of(path):
    value = json.loads(pathlib.Path(path).read_text())
    if "book_10s" in value:
        return value["book_10s"], value["references"], value["directions"], 1
    if "reference_directions" in value and "limits" in value:
        return value["samples"], value["references"], value["reference_directions"], 1
    if "window" in value:
        trajectory = s2.from_window(value)
        return trajectory["samples"], trajectory["references"], trajectory["directions"], 0
    raise ValueError(f"{path} is not a draw")


def main():
    frozen_dir, out = sys.argv[2], pathlib.Path(sys.argv[3])
    regime = sys.argv[4]
    if regime not in REGIMES:
        raise SystemExit(f"unknown regime {regime}")
    results = {"_regime": regime}
    for path in sys.argv[5:]:
        try:
            results[path] = evaluate(frozen_dir, rows_of(*inputs_of(path)), regime)
        except FailedDraw as error:
            results[path] = {"failure": str(error)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results) + "\n")


if __name__ == "__main__" and sys.argv[1] == "run":
    np.seterr(all="raise")
    main()
