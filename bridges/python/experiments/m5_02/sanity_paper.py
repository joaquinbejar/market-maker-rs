"""A post-freeze check of part A of #107: the paper's setting pays, and pays by spread.

    python experiments/m5_02/sanity_paper.py <out dir>

A uniform random raw action and two fixed spreads (0.10 and 0.02 on both sides), 2000
episodes per setting on seeds 5000000 to 5001999, which are neither training nor evaluation
seeds. It was run after the freeze of `design_paper.json` and changes nothing in it.
SYNTHETIC.
"""

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_env  # noqa: E402

SEEDS = range(5_000_000, 5_002_000)


def raw_for(spread):
    p = (spread - paper_env.SPREAD_LOW) / (paper_env.SPREAD_HIGH - paper_env.SPREAD_LOW)
    return math.log(p / (1 - p))


def main():
    design = json.loads((HERE / "design_paper.json").read_text())
    policies = {"random": None, "fixed_0.10": raw_for(0.10), "fixed_0.02": raw_for(0.02)}
    out = {}
    for setting in design["settings"]:
        env = paper_env.PaperEnv(paper_env.kernel_of(setting), setting["horizon"])
        rows = {}
        for name, raw in policies.items():
            rng = np.random.default_rng(3)
            objectives, fills = [], 0
            for seed in SEEDS:
                obs, _ = env.reset(seed=seed)
                done = False
                while not done:
                    action = rng.uniform(-10, 10, 2) if raw is None else np.array([raw, raw])
                    obs, _, done, _, info = env.step(action)
                objectives.append(info["objective"])
                fills += info["fills"]
            objectives = np.asarray(objectives)
            rows[name] = {
                "mean": float(objectives.mean()),
                "sem": float(objectives.std(ddof=1) / math.sqrt(len(objectives))),
                "share_nonzero": float((objectives != 0).mean()),
                "fills_per_episode": fills / len(objectives),
            }
        out[setting["name"]] = rows
    path = pathlib.Path(sys.argv[1]) / "sanity.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
