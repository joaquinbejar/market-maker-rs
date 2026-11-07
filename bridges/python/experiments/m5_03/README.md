# Robustness of learned and classical policies to bounded flow perturbations (#108)

Everything here is synthetic. It uses perturbed copies of #102's queue-reactive regimes: a
simulator with made-up parameters, whose rows do not react to orders. Impact is out of reach
for that reason (#282).

- `design.json` fixes everything before any policy runs:
  - the perturbation grid, the fixed stress and the execution stress;
  - the selection and holdout days' seeds;
  - the adversary's rule;
  - the arms, the conditions, the metrics, the intervals and the stop rules.

  Its `changes` log records each amendment and what had been seen when it was made.
- `digests/` holds every scenario's day digests, written by `qr_days` with this study's seed
  offsets, committed before any policy ran.
- `run.py` runs the `days`, `rate`, `selection`, `freeze` and `holdout` steps. It journals each
  run, so a step resumes with fewer processes.
- `report.py` writes the holdout's tables and paired intervals.

The adversary sets parameters before a day and sees nothing of its path. It changes neither
matching, accounting nor limits. #102's frozen policies run under #102's main execution, which
is in effect an immediate fill on 10 s rows (#267). Adversarial training is #281.
