# Offline training with the regime in the input (#309)

Everything here is synthetic: #102's queue-reactive simulator, with made-up parameters, on new
days generated for this study. Its rows do not react to orders. The logs' rewards are the
simulator's consequences of the logged actions, not observed ones.

This is a new experiment motivated by a validation result of #280, not a confirmatory
continuation of it. #280's stop at its BC sanity rule stays as the result of its design.

`design.json` fixes before any run the regime input, the days, the arms and their settings, the
comparators, the intervals, the stop rules, the attempts and the holdout rule, and says what had
been seen. `changes` records its versions. `days_input.json` is #102's design with only a seed
offset added, which `qr_days` reads to generate the days; `digests.json` holds their digests.

- `regime.py`: the one-hot and the aliasing diagnosis.
- `run.py`: the steps, in the order its docstring gives.

## Audit of how #280 built its logs

From reading `experiments/m4_11/run.py` and `experiments/m4_07/run.py`:
- `log_job` records, per decision, the observation, the mask from `env.action_masks()`, the
  behaviour's action and `target_code`, as_grid's greedy action coded by `grid.code`.
- as_grid reads only its per-regime point, the raw position, the raw steps left and the mask, so
  its action is a deterministic function of those four. The observation holds position and
  steps left (features 6 and 10) but no regime and no per-regime input.
- No fault was found by reading. This study's logs use the same path, and add the raw position
  and steps left that the aliasing diagnosis keys on.

The `audit` step reports, before any training, the aliasing diagnosis and whether position and
steps left reach the network unclipped and one to one. Its result is recorded here once it has
run.

## Status

The design was reviewed and fixed before any run. No day, log or training exists yet.
