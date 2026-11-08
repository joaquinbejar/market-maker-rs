# Adversarial training against flow perturbations, tested on held-out stress (#281)

Everything here is synthetic: #102's queue-reactive regimes, a simulator with made-up
parameters, and #108's perturbed copies of them. The fresh holdout days come from the same
generator, so a result says nothing about a market. #102's main execution is in effect an
immediate fill on 10 s rows (#267), and every conclusion carries that bound.

- `design.json` fixes everything before any code or run:
  - the arms: #102's residual SAC over Avellaneda-Stoikov retrained without (B) and with (A) the
    training adversary, and #102's frozen seeds as a reference row with no claim;
  - the split of #108's grid: the 12 points with the limit rate unchanged train the adversary;
    the 12 with a lower limit rate, the execution stress and #282's own impact are held out;
  - the adversary's phases, mixture and score, the checkpoint (the last, for both arms), B's
    numeric reproduction check, the selection of each arm's worst held-out point, the fresh
    holdout days, the estimands, the intervals, the failures and the stop rules.

  `from_the_approved_note` lists what changed from the note the maintainer approved, and
  `changes` records later amendments and what had been seen when it was made.
- `digests/` holds the day digests of the training points and of every holdout condition's
  exogenous rows, committed before any policy runs. The rows are simulated in process by the
  bridge's `ReactiveQrMarket`, whose rows and digest equal `qr_days`'; the `days` step checks
  that #102's training days and #108's selection days come out with their committed digests.
- `frozen_models.json` holds the checkpoints' hashes, written after training and B's check and
  before any held-out point was evaluated (committed after the selection, unchanged).
- `amendments.json` records the one amendment after the freeze (the resume), and `resume.py`
  carries it out.
- `run.py` runs the `days`, `rate`, `train`, `validate`, `freeze_models`, `selection`, `freeze`
  and `holdout` steps. Selection and validation journal each run, so they resume with fewer
  processes. The holdout marks its attempt before any day is read; a second plain run is refused, and the one resume after an external kill is resume.py's, recorded in amendments.json.
- `train.attempts` journals every training run's start and end. A run that failed inside
  (a program or numerical error) or was stopped by the overrun rule is a failed seed: kept,
  reported, its days missing, and never rerun. A run killed from outside is rerun only when
  named, once.
- The overrun rule is enforced: `train`, `validate`, `selection` and `holdout` read their
  estimates from `estimates.json` in the output directory. No step writes it: it was written by
  hand from the `rate` step's timings before training (contents below), and the freeze does not
  hash it. A step stops at 1.5 times its estimate, killing only its own workers; what ended
  stays journaled, and the stop is reported.
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
- **Secondary:** 2 of the 13 held-out conditions evaluated have a seed-averaged tail interval
  that excludes 0, as the design counts them, and both are wholly above 0 (A worse):
  m1.0-c1.0-t1.2-l0.7 (+9.13 [5.16, 11.49]) and m1.5-c1.5-t1.0-l0.7 (+28.45 [5.07, 46.52]).
  None is wholly below 0. No multiplicity correction. Correction, disclosed: the first report
  (stored as report.md, run before this was found in review) printed only the count wholly below
  0 ("0 of the 13"); report.py now prints the design's count as well, from the same holdout.json,
  with no rerun.
- **Reference:** the reproduction check (mean of B's per-seed validation means within 5.17 of
  #102's) passed, but it is a weak gate here: B reuses #102's seeds, so it is a replay, and the
  tolerance assumes independent seeds. The artifacts show more than the gate: B's rows equal
  the reference's in every recorded field (net, drawdown, positions, fills, peak notional,
  refusals, fallbacks, clips, breach, peak position and price, dataset digest) on all 75
  validation, 1800 selection and 2100 holdout seed-days compared. The reference row adds
  nothing.

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

## Full tables (report.py on holdout.json, SYNTHETIC)

### Primary: A - B, 90 % intervals

| estimand | seed-averaged arm | mean per-seed |
|---|---|---|
| cost: mean daily net, unperturbed (< 0 a cost) | -3.32 [-4.45, -2.20] | |
| CVaR5 loss at p_A (< 0 an improvement) | 6.39 [-2.94, 13.98] | 10.02 [-1.62, 19.19] |
| CVaR5 loss at p_B (< 0 an improvement) | 6.39 [-2.94, 13.98] | 10.02 [-1.62, 19.19] |
| envelope: A at p_A against B at p_B, two selected envelopes, not a same-scenario effect | 6.39 [-2.94, 13.98] | 10.02 [-1.62, 19.19] |

Primary claim: **no gain shown**. It needs both tail intervals at p_A and p_B wholly below 0; the envelope never carries it alone. An interval that holds 0 is never read as robustness, equivalence or an absence of risk.

### Secondary: every held-out condition, A - B

| condition | CVaR5 loss, averaged | CVaR5 loss, per seed | mean daily net |
|---|---|---|---|
| held_out:m1.0-c1.0-t1.0-l0.7 | 2.41 [-4.74, 5.70] | 2.82 [0.05, 5.14] | -2.78 [-4.54, -1.02] |
| held_out:m1.0-c1.0-t1.2-l0.7 | 9.13 [5.16, 11.49] | 9.86 [6.73, 13.96] | -3.30 [-5.08, -1.50] |
| held_out:m1.0-c1.5-t1.0-l0.7 | -0.28 [-2.96, 8.90] | 2.89 [-1.03, 10.34] | -4.10 [-6.97, -1.38] |
| held_out:m1.0-c1.5-t1.2-l0.7 | 0.90 [-6.86, 6.79] | 2.72 [-2.96, 6.25] | -4.08 [-5.98, -2.24] |
| held_out:m1.5-c1.0-t1.0-l0.7 | -1.06 [-3.94, 1.82] | 1.62 [0.32, 6.61] | -4.73 [-7.04, -2.34] |
| held_out:m1.5-c1.0-t1.2-l0.7 | -2.43 [-12.60, 8.97] | 2.93 [-7.12, 12.87] | -1.72 [-4.03, 0.78] |
| held_out:m1.5-c1.5-t1.0-l0.7 | 28.45 [5.07, 46.52] | 26.35 [5.82, 44.71] | -9.93 [-12.83, -7.03] |
| held_out:m1.5-c1.5-t1.2-l0.7 | -3.00 [-6.87, 8.65] | -0.33 [-5.86, 8.99] | -2.53 [-5.11, -0.06] |
| held_out:m2.0-c1.0-t1.0-l0.7 | -4.73 [-8.56, 3.65] | 0.98 [-6.07, 11.03] | -5.34 [-7.88, -2.90] |
| held_out:m2.0-c1.0-t1.2-l0.7 | -6.58 [-10.73, 4.17] | 3.31 [-7.52, 10.76] | -4.23 [-7.27, -1.21] |
| held_out:m2.0-c1.5-t1.0-l0.7 | 12.59 [-5.41, 22.75] | 11.87 [-0.76, 19.81] | -4.67 [-7.32, -2.10] |
| held_out:m2.0-c1.5-t1.2-l0.7 | 6.39 [-2.94, 13.98] | 10.02 [-1.62, 19.19] | -4.65 [-7.96, -1.65] |
| execution_stress | 1.57 [-0.62, 5.10] | 2.65 [1.64, 4.82] | -2.84 [-3.70, -1.96] |

2 of the 13 held-out conditions evaluated have a seed-averaged tail interval that excludes 0, as the design counts them; 0 of them wholly below 0 (an improvement of A). No multiplicity correction.

### Every arm and condition

| condition | arm | seeds | missing seed-days | mean net | CVaR5, averaged | CVaR5, per seed |
|---|---|---:|---:|---:|---:|---:|
| unperturbed | adversarial | 5 | 0 | -30.94 | 47.51 | 50.67 |
| unperturbed | baseline | 5 | 0 | -27.62 | 46.27 | 47.57 |
| unperturbed | reference | 5 | 0 | -27.62 | 46.27 | 47.57 |
| held_out:m1.0-c1.0-t1.0-l0.7 | adversarial | 5 | 0 | -57.28 | 92.78 | 95.26 |
| held_out:m1.0-c1.0-t1.0-l0.7 | baseline | 5 | 0 | -54.51 | 90.36 | 92.44 |
| held_out:m1.0-c1.0-t1.0-l0.7 | reference | 5 | 0 | -54.51 | 90.36 | 92.44 |
| held_out:m1.0-c1.0-t1.2-l0.7 | adversarial | 5 | 0 | -69.32 | 105.84 | 109.23 |
| held_out:m1.0-c1.0-t1.2-l0.7 | baseline | 5 | 0 | -66.02 | 96.71 | 99.37 |
| held_out:m1.0-c1.0-t1.2-l0.7 | reference | 5 | 0 | -66.02 | 96.71 | 99.37 |
| held_out:m1.0-c1.5-t1.0-l0.7 | adversarial | 5 | 0 | -105.27 | 164.35 | 167.53 |
| held_out:m1.0-c1.5-t1.0-l0.7 | baseline | 5 | 0 | -101.17 | 164.64 | 164.64 |
| held_out:m1.0-c1.5-t1.0-l0.7 | reference | 5 | 0 | -101.17 | 164.64 | 164.64 |
| held_out:m1.0-c1.5-t1.2-l0.7 | adversarial | 5 | 0 | -124.94 | 187.61 | 190.95 |
| held_out:m1.0-c1.5-t1.2-l0.7 | baseline | 5 | 0 | -120.86 | 186.71 | 188.23 |
| held_out:m1.0-c1.5-t1.2-l0.7 | reference | 5 | 0 | -120.86 | 186.71 | 188.23 |
| held_out:m1.5-c1.0-t1.0-l0.7 | adversarial | 5 | 0 | -95.85 | 159.00 | 161.96 |
| held_out:m1.5-c1.0-t1.0-l0.7 | baseline | 5 | 0 | -91.12 | 160.06 | 160.34 |
| held_out:m1.5-c1.0-t1.0-l0.7 | reference | 5 | 0 | -91.12 | 160.06 | 160.34 |
| held_out:m1.5-c1.0-t1.2-l0.7 | adversarial | 5 | 0 | -114.09 | 180.84 | 187.47 |
| held_out:m1.5-c1.0-t1.2-l0.7 | baseline | 5 | 0 | -112.37 | 183.27 | 184.54 |
| held_out:m1.5-c1.0-t1.2-l0.7 | reference | 5 | 0 | -112.37 | 183.27 | 184.54 |
| held_out:m1.5-c1.5-t1.0-l0.7 | adversarial | 5 | 0 | -146.49 | 250.69 | 250.69 |
| held_out:m1.5-c1.5-t1.0-l0.7 | baseline | 5 | 0 | -136.56 | 222.23 | 224.33 |
| held_out:m1.5-c1.5-t1.0-l0.7 | reference | 5 | 0 | -136.56 | 222.23 | 224.33 |
| held_out:m1.5-c1.5-t1.2-l0.7 | adversarial | 5 | 0 | -169.72 | 257.29 | 260.38 |
| held_out:m1.5-c1.5-t1.2-l0.7 | baseline | 5 | 0 | -167.19 | 260.29 | 260.71 |
| held_out:m1.5-c1.5-t1.2-l0.7 | reference | 5 | 0 | -167.19 | 260.29 | 260.71 |
| held_out:m2.0-c1.0-t1.0-l0.7 | adversarial | 5 | 0 | -131.41 | 211.63 | 218.30 |
| held_out:m2.0-c1.0-t1.0-l0.7 | baseline | 5 | 0 | -126.06 | 216.36 | 217.32 |
| held_out:m2.0-c1.0-t1.0-l0.7 | reference | 5 | 0 | -126.06 | 216.36 | 217.32 |
| held_out:m2.0-c1.0-t1.2-l0.7 | adversarial | 5 | 0 | -158.85 | 254.54 | 266.62 |
| held_out:m2.0-c1.0-t1.2-l0.7 | baseline | 5 | 0 | -154.63 | 261.12 | 263.31 |
| held_out:m2.0-c1.0-t1.2-l0.7 | reference | 5 | 0 | -154.63 | 261.12 | 263.31 |
| held_out:m2.0-c1.5-t1.0-l0.7 | adversarial | 5 | 0 | -178.72 | 295.49 | 297.19 |
| held_out:m2.0-c1.5-t1.0-l0.7 | baseline | 5 | 0 | -174.04 | 282.90 | 285.33 |
| held_out:m2.0-c1.5-t1.0-l0.7 | reference | 5 | 0 | -174.04 | 282.90 | 285.33 |
| held_out:m2.0-c1.5-t1.2-l0.7 | adversarial | 5 | 0 | -210.04 | 333.80 | 337.82 |
| held_out:m2.0-c1.5-t1.2-l0.7 | baseline | 5 | 0 | -205.39 | 327.41 | 327.80 |
| held_out:m2.0-c1.5-t1.2-l0.7 | reference | 5 | 0 | -205.39 | 327.41 | 327.80 |
| execution_stress | adversarial | 5 | 0 | -24.12 | 38.47 | 40.56 |
| execution_stress | baseline | 5 | 0 | -21.28 | 36.90 | 37.91 |
| execution_stress | reference | 5 | 0 | -21.28 | 36.90 | 37.91 |
| in_sample:m1.0-c1.0-t1.2-l1.0 | adversarial | 5 | 0 | -38.31 | 59.83 | 66.13 |
| in_sample:m1.0-c1.0-t1.2-l1.0 | baseline | 5 | 0 | -36.00 | 63.13 | 63.13 |
| in_sample:m1.0-c1.5-t1.0-l1.0 | adversarial | 5 | 0 | -69.16 | 95.63 | 100.36 |
| in_sample:m1.0-c1.5-t1.0-l1.0 | baseline | 5 | 0 | -65.50 | 97.23 | 98.01 |
| in_sample:m1.0-c1.5-t1.2-l1.0 | adversarial | 5 | 0 | -85.58 | 113.55 | 128.13 |
| in_sample:m1.0-c1.5-t1.2-l1.0 | baseline | 5 | 0 | -83.78 | 125.85 | 125.85 |
| in_sample:m1.5-c1.0-t1.0-l1.0 | adversarial | 5 | 0 | -59.98 | 94.84 | 98.42 |
| in_sample:m1.5-c1.0-t1.0-l1.0 | baseline | 5 | 0 | -56.86 | 84.10 | 86.25 |
| in_sample:m1.5-c1.0-t1.2-l1.0 | adversarial | 5 | 0 | -73.33 | 111.50 | 117.67 |
| in_sample:m1.5-c1.0-t1.2-l1.0 | baseline | 5 | 0 | -68.75 | 107.52 | 107.84 |
| in_sample:m1.5-c1.5-t1.0-l1.0 | adversarial | 5 | 0 | -104.86 | 158.46 | 167.60 |
| in_sample:m1.5-c1.5-t1.0-l1.0 | baseline | 5 | 0 | -101.55 | 158.87 | 159.61 |
| in_sample:m1.5-c1.5-t1.2-l1.0 | adversarial | 5 | 0 | -125.04 | 177.69 | 181.87 |
| in_sample:m1.5-c1.5-t1.2-l1.0 | baseline | 5 | 0 | -122.40 | 173.67 | 176.37 |
| in_sample:m2.0-c1.0-t1.0-l1.0 | adversarial | 5 | 0 | -93.12 | 152.56 | 158.73 |
| in_sample:m2.0-c1.0-t1.0-l1.0 | baseline | 5 | 0 | -88.99 | 152.25 | 152.50 |
| in_sample:m2.0-c1.0-t1.2-l1.0 | adversarial | 5 | 0 | -113.66 | 174.29 | 178.05 |
| in_sample:m2.0-c1.0-t1.2-l1.0 | baseline | 5 | 0 | -111.69 | 173.14 | 175.63 |
| in_sample:m2.0-c1.5-t1.0-l1.0 | adversarial | 5 | 0 | -143.59 | 226.43 | 232.84 |
| in_sample:m2.0-c1.5-t1.0-l1.0 | baseline | 5 | 0 | -137.55 | 214.20 | 218.39 |
| in_sample:m2.0-c1.5-t1.2-l1.0 | adversarial | 5 | 0 | -169.26 | 260.81 | 268.26 |
| in_sample:m2.0-c1.5-t1.2-l1.0 | baseline | 5 | 0 | -165.18 | 256.19 | 256.97 |

The reference is #102's frozen residual SAC; no claim is attached to it.

### In sample: the perturbed training points (no claim)

| point | CVaR5 loss, averaged | CVaR5 loss, per seed |
|---|---|---|
| in_sample:m1.0-c1.0-t1.2-l1.0 | -3.30 [-4.64, 0.42] | 3.00 [0.12, 5.64] |
| in_sample:m1.0-c1.5-t1.0-l1.0 | -1.61 [-9.18, 7.55] | 2.35 [-2.65, 8.76] |
| in_sample:m1.0-c1.5-t1.2-l1.0 | -12.30 [-15.13, 0.34] | 2.28 [-2.87, 9.46] |
| in_sample:m1.5-c1.0-t1.0-l1.0 | 10.74 [3.72, 17.81] | 12.17 [6.47, 17.25] |
| in_sample:m1.5-c1.0-t1.2-l1.0 | 3.98 [-0.61, 13.55] | 9.84 [3.83, 16.65] |
| in_sample:m1.5-c1.5-t1.0-l1.0 | -0.40 [-5.88, 7.76] | 7.99 [1.70, 14.99] |
| in_sample:m1.5-c1.5-t1.2-l1.0 | 4.02 [-4.19, 9.96] | 5.50 [0.13, 8.95] |
| in_sample:m2.0-c1.0-t1.0-l1.0 | 0.31 [-5.27, 7.30] | 6.22 [2.90, 10.50] |
| in_sample:m2.0-c1.0-t1.2-l1.0 | 1.15 [-8.01, 6.28] | 2.42 [-2.43, 6.77] |
| in_sample:m2.0-c1.5-t1.0-l1.0 | 12.23 [-0.03, 21.62] | 14.45 [3.92, 18.00] |
| in_sample:m2.0-c1.5-t1.2-l1.0 | 4.62 [0.11, 9.39] | 11.29 [4.40, 14.01] |

### B's reproduction check (validation days)

Passed: True; statistic -27.162139373333332, #102's -27.1621, tolerance 5.1668.

### Training

| run | worst point per scoring | its worst daily loss | train s | scoring s |
|---|---|---|---:|---:|
| adversarial-seed1 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 248.53, 243.77, 255.52, 244.85 | 1379.36 | 348.19 |
| adversarial-seed2 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 285.74, 254.36, 247.17, 243.82 | 1376.67 | 345.5 |
| adversarial-seed3 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 274.43, 258.56, 249.81, 254.45 | 1379.92 | 346.82 |
| adversarial-seed4 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 278.38, 266.86, 246.88, 233.08 | 1381.61 | 346.19 |
| adversarial-seed5 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 243.02, 249.52, 257.58, 235.50 | 1285.77 | 307.71 |
| baseline-seed1 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 252.73, 260.35, 247.68, 248.62 | 1292.04 | 310.37 |
| baseline-seed2 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 245.18, 240.59, 247.45, 271.27 | 1294.57 | 311.37 |
| baseline-seed3 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 280.74, 244.93, 245.92, 252.15 | 1292.99 | 309.19 |
| baseline-seed4 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 270.09, 229.38, 240.23, 243.74 | 1167.58 | 291.47 |
| baseline-seed5 | m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0 | 250.44, 245.15, 238.78, 253.01 | 1169.73 | 295.19 |


## Criterion 3: cost, seed stability, failures

### Training cost

| run | training s | scoring s | scoring episodes | steps |
|---|---:|---:|---:|---:|
| adversarial seed 1 | 1379.36 | 348.19 | 288 | 500000 |
| adversarial seed 2 | 1376.67 | 345.5 | 288 | 500000 |
| adversarial seed 3 | 1379.92 | 346.82 | 288 | 500000 |
| adversarial seed 4 | 1381.61 | 346.19 | 288 | 500000 |
| adversarial seed 5 | 1285.77 | 307.71 | 288 | 500000 |
| baseline seed 1 | 1292.04 | 310.37 | 288 | 500000 |
| baseline seed 2 | 1294.57 | 311.37 | 288 | 500000 |
| baseline seed 3 | 1292.99 | 309.19 | 288 | 500000 |
| baseline seed 4 | 1167.58 | 291.47 | 288 | 500000 |
| baseline seed 5 | 1169.73 | 295.19 | 288 | 500000 |

adversarial: training 1.89 process-hours, scoring 0.47 (counted apart from the 500 000 step budget).
baseline: training 1.73 process-hours, scoring 0.42 (counted apart from the 500 000 step budget).

Adversary's worst training point at each scoring (phases 2 to 5), A only:

- seed 1: m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0
- seed 2: m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0
- seed 3: m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0
- seed 4: m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0
- seed 5: m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0, m2.0-c1.5-t1.2-l1.0

### Seed stability (holdout, per seed)

| arm | seed | mean net unperturbed | CVaR5 loss unperturbed | mean net at m2.0-c1.5-t1.2-l0.7 | CVaR5 loss there |
|---|---:|---:|---:|---:|---:|
| adversarial | 1 | -25.81 | 41.71 | -207.63 | 332.50 |
| adversarial | 2 | -32.66 | 53.19 | -217.43 | 346.27 |
| adversarial | 3 | -34.96 | 57.30 | -214.55 | 344.46 |
| adversarial | 4 | -27.46 | 43.23 | -197.56 | 329.59 |
| adversarial | 5 | -33.82 | 57.93 | -213.02 | 336.31 |
| adversarial | spread (sd over seeds) | 4.06 | 7.73 | 7.83 | 7.31 |
| baseline | 1 | -27.38 | 47.48 | -211.93 | 334.35 |
| baseline | 2 | -34.30 | 55.50 | -212.17 | 334.15 |
| baseline | 3 | -24.85 | 41.84 | -197.84 | 314.84 |
| baseline | 4 | -27.60 | 50.86 | -205.92 | 336.31 |
| baseline | 5 | -23.97 | 42.19 | -199.07 | 319.37 |
| baseline | spread (sd over seeds) | 4.05 | 5.82 | 6.82 | 9.93 |

### Failures

- Training: 10 of 10 runs ended done; 0 failed, 0 interrupted, 0 stopped, 0 reruns.
- Validation 0 of 225, selection 0 of 5400, holdout 0 of 9600 seed-days missing; no non-finite net.
- Holdout attempts: 1 killed from outside, 1 resume (see Disclosures).

## Pinned files

- estimates.json sha256 `460a5826e95a7e3a492f55dab01343be70b3d72f1533d27141807c05fbf5ac6e`, written by
  hand from rate.json and not hashed by the freeze: train 9200 s, validate 300 s, selection
  4050 s, holdout 8900 s (limits 1.5 times), with the basis that every one of the 500 000 steps
  is a gradient step (an upper bound).

- amendments.json sha256 `409de61e3b25701ea1c395db28b4b52473ccd23d2919491707d4c5b50bb10147`
- resume.py sha256 `ba13d8e176ad8d3af676238e627481abfafc63f2a8c1120bcf13267119872f8d` (recorded in holdout.resumed and holdout.json)
- run.py sha256 `82afd334a00905f120597a6329df1e25e6a01b7f5fb7ead1017f337eabe1878a`, design.json `06861512bfa6c5c6fa75145207a41be2b818c9f60ae86c6570c76485989bc1ef`, frozen.json `4c8ca714b08b789922a954142063906d29c4851f59446cee2d9e54273bb0226d`
