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

## Results (SYNTHETIC)

The holdout ran once from frozen.json `4c8ca714...`, on 30 fresh days. Its first attempt was
killed from outside and resumed once, as disclosed below. All 9600 rows are present.

- **Primary claim: no gain shown.** Both arms' worst held-out point on the selection days is
  m2.0-c1.5-t1.2-l0.7, so the tail contrast at p_A, at p_B and the envelope are one number. A
  minus B in CVaR5 loss is +6.39 [-2.94, 13.98] for the seed-averaged arm and +10.02 [-1.62,
  19.19] per seed (90 %). The point estimate says A's tail is larger, and the interval holds 0.
- **Cost:** unperturbed, A's mean daily net is 3.32 below B's, [-4.45, -2.20].
- **Secondary:** on none of the 13 held-out conditions evaluated is A's tail interval wholly
  below B's. On two it is wholly above: m1.0-c1.0-t1.2-l0.7 (+9.13 [5.16, 11.49]) and
  m1.5-c1.5-t1.0-l0.7 (+28.45 [5.07, 46.52]). No multiplicity correction.
- **Reference:** retrained B reproduces #102's residual SAC exactly. Its validation daily nets
  match #102's to every printed digit, and its selection and holdout rows equal the reference's.
  The reference row adds nothing.

Disclosures:
- **Impact not run.** The impact condition was not run: #282's runner was not on main at the
  freeze, so 13 held-out conditions, not 14. #108's learned criterion on impact stays open.
- **frozen_models.json timing.** The file was written at 11:37:07Z, before the selection, and
  the selection checked every checkpoint against it. It was committed after the selection,
  unchanged.
- **Resumed once.** The first holdout attempt (12:02:57Z) was killed at about 12:13:40Z, 627 s
  in, with 2236 of 9600 jobs journaled, when the terminal session that started it ended.
  - The transcript of that session has no entry between about 12:05Z and 12:36Z, so the exact
    session exit time is not known.
  - It was resumed once by resume.py (amendments.json), detached, with run.py and design.json
    as frozen. Its marker records the journal's hash before the resume; the first 2236 lines
    were unchanged after it.
  - That nothing was read in between rests on the author's command log: a line count and a parse
    only. It is not provable.
  - "Same binary" rests on the native extension's mtime (08:14Z, before the first attempt): the
    first attempt recorded no hashes. The resume recorded them.
