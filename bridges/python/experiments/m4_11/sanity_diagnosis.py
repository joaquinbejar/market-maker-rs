"""Why the BC sanity rule stopped the study: BC's agreement with as_grid, by regime and split.

    python experiments/m4_11/sanity_diagnosis.py <dataset> <output json> <#102 data dir> \
        <#102 run dir> <#104 out dir> <out dir> <commit>

reads the dataset's logs and its BC checkpoints, each checked against its recorded sha256, and
writes, per BC seed and per split (validation, then train): the agreement with as_grid's action
overall and per regime, how often the logged action is as_grid's, as_grid's share of holds,
and of the decisions where BC and as_grid both quote but differ, how many are within one
offset step. It reads no holdout day and writes nothing into the out dir.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent


def breakdown(days, actions):
    """`days`: (regime, predicted codes, as_grid's codes, logged codes) per day."""
    by_regime = {}
    logged = holds = total = near = quote_misses = 0
    for regime, predicted, target, logged_codes in days:
        predicted, target = np.asarray(predicted), np.asarray(target)
        if not len(predicted) == len(target) == len(logged_codes):
            raise ValueError(f"a {regime} day with codes of different lengths")
        hit = predicted == target
        agree, count = by_regime.get(regime, (0, 0))
        by_regime[regime] = (agree + int(hit.sum()), count + len(hit))
        logged += int((np.asarray(logged_codes) == target).sum())
        holds += int((target == 0).sum())
        total += len(target)
        for p, t in zip(predicted[~hit], target[~hit], strict=True):
            if p == 0 or t == 0:
                continue
            a, b = actions.action(int(p)), actions.action(int(t))
            quote_misses += 1
            near += int(abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])) <= 1)
    agree = sum(a for a, _ in by_regime.values())
    return {"decisions": total, "agreement": agree / total,
            "by_regime": {r: a / n for r, (a, n) in sorted(by_regime.items())},
            "logged_equals_as_grid": logged / total, "as_grid_hold_share": holds / total,
            "quote_misses": quote_misses, "quote_misses_offsets_within_1": near}


def main():
    dataset, output = sys.argv[1], pathlib.Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("m4_11_run", HERE / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    study = run.Study(*sys.argv[3:8])
    actions = run.offline.Actions(len(study.grid.offsets), len(study.grid.sizes))
    report = study.header(f"sanity diagnosis {dataset}")
    report["runs"] = []
    for split in ("validation", "train"):
        records = study.logs(dataset, split)
        days = run.read_logs(records)
        for meta in study.read(f"train-{dataset}-bc")["runs"]:
            network = run.network_of(meta).eval()
            rows = []
            for record, day in zip(records, days, strict=True):
                observation = torch.as_tensor(np.asarray(day["observation"], dtype=np.float32))
                joint = torch.as_tensor(actions.joint_mask(day["mask"]))
                with torch.no_grad():
                    predicted = run.offline.masked(network(observation), joint).argmax(1).numpy()
                target = day["target_code"][list(day["target_names"]).index("as_grid")]
                rows.append((record["regime"], predicted, target, day["action"]))
            report["runs"].append({"split": split, "seed": meta["seed"],
                                   **breakdown(rows, actions)})
    output.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
