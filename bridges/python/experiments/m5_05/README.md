# Adversarial training against flow perturbations, tested on held-out stress (#281)

Everything here is synthetic: #102's queue-reactive regimes, a simulator with made-up
parameters, and #108's perturbed copies of them. The fresh holdout days come from the same
generator, so a result says nothing about a market.

- `design.json` fixes everything before any code or run:
  - the arms: #102's residual SAC over Avellaneda-Stoikov retrained without (B) and with (A) the
    training adversary, and #102's frozen seeds as a reference row with no claim;
  - the split of #108's grid: the 12 points with the limit rate unchanged train the adversary;
    the 12 with a lower limit rate, the execution stress and #282's own impact are held out;
  - the adversary's phases, mixture and score, the checkpoint (the last, for both arms), B's
    numeric reproduction check, the selection of each arm's worst held-out point, the fresh
    holdout days, the estimands, the intervals, the failures and the stop rules.

  `from_the_approved_note` lists what changed from the note the maintainer approved, and
  `changes` will record every later amendment and what had been seen when it was made.
- `digests/` will hold the day digests of the training points and of every holdout condition's
  exogenous rows, committed before any policy runs. The rows are simulated in process by the
  bridge's `ReactiveQrMarket`, whose rows and digest equal `qr_days`'; the `days` step checks
  that #102's training days and #108's selection days come out with their committed digests.
- `frozen_models.json` will hold the checkpoints' hashes, committed after training and B's
  check, before any held-out point is evaluated.
- `run.py` runs the `days`, `rate`, `train`, `validate`, `freeze_models`, `selection`, `freeze`
  and `holdout` steps. Selection and validation journal each run, so they resume with fewer
  processes. The holdout marks its attempt before any day is read and never runs twice.
- `train.attempts` journals every training run's start and end. A run that failed inside
  (a program or numerical error) or was stopped by the overrun rule is a failed seed: kept,
  reported, its days missing, and never rerun. A run killed from outside is rerun only when
  named, once.
- The overrun rule is enforced: `train`, `validate`, `selection` and `holdout` read their
  estimates from `estimates.json` in the output directory, written from the `rate` step before
  they run, and stop at 1.5 times the estimate, killing only their own workers. What ended stays
  journaled, and the stop is reported.
- `report.py` writes the holdout's estimands and intervals, every arm's table, the in-sample
  rows, B's check and the training scores and costs.

The impact condition runs only through #282's runner on main; without it, it is reported as not
run and nothing replaces it.
