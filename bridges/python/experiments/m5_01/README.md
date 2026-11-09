# MDQR as a neural order book generator (#106)

Two stages. S0 is SYNTHETIC: MDQR recovers a queue-reactive truth at #102's calm parameters, and
it passed. The main study is REAL DATA, the public Hyperliquid Order Flow Dataset (Zenodo
18184441, BTC perpetual, December 2025), and it is **exploratory on a reused holdout**: #286 had
already read this holdout for the queue-reactive arms, and MDQR's design was informed by that
failure. `design.json` holds the design and every change to it, in order.

## Decision: discard

MDQR is **not adopted**, by the frozen rule S2, for two independent reasons:

1. Dependency (a), the error in the correlation of the best bid and ask queues, fails against
   both queue-reactive arms: the upper bound of MDQR minus the arm is 0.058 against Drop and
   0.064 against Count, over the 0.02 margin.
2. The transfer rule cannot pass. Every one of the 13 policies has zero fills on the rows of all
   three queue-reactive arms, so their nets all tie at 0 and Kendall's tau-b is undefined. An
   undefined tau is read as not passing. This says that those arms' saturated touch queues do not
   let any order reach the front under the rational queue. It says nothing about the policies.

No draw failed: all 150 MDQR holdout draws, and S1's 10 runs, are valid.

| S2 | Result |
|---|---|
| Primary, queue size TV, MDQR minus each QR arm, 3 limits and 3 arms, 90 % | all 9 pass; upper bounds -0.18 to -0.76 |
| (a) correlation of best queues, Drop / Count | **fail**: [-0.031, 0.058] / [0.040, 0.064] |
| (b) P(kind given queue and limit) | pass: upper -0.361 / -0.155 |
| (c) next mid move direction given imbalance | pass: upper -0.580 / -0.604 |
| (d) gamma shape | pass, see the caveat on Drop |
| (e) share of events at the cap | pass: upper -0.393 / 0.003 |
| Transfer, tau-b at calm | **undefined**, so not a pass |

MDQR's generative fit is much better than the queue-reactive arms' on the queue sizes, and that
is a finding, but S2 asks more than that. The conclusions hold on this epoch only. MDQR trains on
one noon window a day and the controls on all of `queue_train`, so training exposure is unequal.

## What to read carefully

- **Neither QR arm is a strong comparator here.** Both are transient on this epoch (#290). Median
  touch lots on their rows are 200 (Drop, at the cap) and 132 (Count), and the mid barely moves on
  them: realised 10 s volatility is 0.010 and 0 kernel ticks. The primary's wins are largely wins
  over a saturating book. Against the stationary constant baseline MDQR also wins, with upper
  bounds of -0.27, -0.61 and -0.68 by limit.
- **Metric (e) favours MDQR structurally.** Each arm is measured at its own cap, 358 for MDQR and
  200 for QR, so saturation is rarer for MDQR by construction.
- **Metric (d) against Drop is degenerate:** its interval is around -700000 because a queue near
  the cap has almost no variance, which blows up the moment estimate. The pass carries no
  information about shape. Dispersion is large throughout: this is a moment estimate per block.
- **MDQR's mid is too volatile.** Realised 10 s volatility on its rows is 12.0 kernel ticks, and the
  median spread is 2.77 against 2 for the QR arms. Tau against the data's own rows is 0.60 at
  calm and 0.78 at stressed and thin, outside the rule. The rule is at calm, and passing there
  would not have changed under stressed or thin, since the undefined tau holds in all three.
- **Ties.** On MDQR's rows 5.99 of the 13 policies have zero fills on average, 4 tie exactly, and
  the mean net of the fixed spread is -385.8. GLFT and maskable PPO mostly do not trade. A tau
  over these policies would rest on the other seven; it is not a ranking of the six.
- The arms run outside their calibration: the mapped rows are not #102's calm regime, the episode
  is a session's first 720 rows and closes at the data's end, so policies never reach the
  end-of-session behaviour they trained on, and a move of several ticks counts as one step.

## Disclosures

- **Chronology.** Extraction and hash checks of the capture began at 07:14 UTC on 2026-11-08, after
  the design at bb524c1, committed at 07:08 UTC, was approved. No semantic parse, replay or inspection of the
  capture came before the later design agreement. Every extracted file matched the per-file
  manifest of #76.
- **Hawkes is a failed arm.** Its fit on `queue_train` returned `HawkesFailure::Unstable` at all 4
  decays. No retuning and no other stride was run. The inputs were thinned one in 10 here,
  against one in 31 in #263's month, and the failure cannot be attributed to either the stride or
  the epoch. The design's rule reads only the QR arms, so S2 is unaffected.
- **Second export.** S2 dependency (c) needs each mid move's direction, which the first export did
  not write; this surfaced after S1 passed. The 220 hours were replayed once more with the same
  files and the direction added, and the windows, fits and summary equal the first export's
  otherwise. The holdout hours were re-read mechanically for that export, with only the count of
  directions checked before the evaluation step.
- **Transfer.** Frozen arms are a copy of #102's, verified against its manifest (sha256
  `9de757b5bf76274882306c9a9c3d9257389deca64d24074948d1a95646c19458`). They are at action space
  version 1 and the crate at 2. Under #102's fill model the check reproduces each recorded
  validation value exactly: 5 of 5 for residual SAC seed 1. For maskable PPO, 5 of 5 as well, but
  all those values are 0, so the check says nothing about its inputs: **PPO is unverified.**
- Before the transfer's pins were complete, the runner was tested on the first validation window
  of the data's rows, and each policy's net and fills there were read. The holdout was claimed
  once, after that freeze, and read once, with no resume.
- The holdout's windows were read by the QR arms before (#286), and #290 read training.
- `design.json`'s text was edited after the holdout to remove internal references and local paths, with no change of meaning. The version under which the holdout ran is at commit `c1928b9`, sha256 `536e19b2db42326c6a2665d0c22215b9d26ebee8671f8a8b959cd1ece52c6ca0`. This study is merged as a squash; commit `c1928b9` stays retrievable from the commits of its pull request (#318).

## Files

S0's script is `s0.py`; its results were produced when it was named `run.py`, a rename made after the holdout so that its module does not shadow another study's.

`results/` holds S1, the consistency and gate checks, the three `s2-<regime>.json` and the run log.
The capture and its derived windows are not in the repository; the capture is public. From
`bridges/python`, `extract.py`, `examples/mdqr_capture` (at the root), `train.py`, `evaluate.py`,
`examples/mdqr_qr` `arms`, `aggregate.py` and `transfer.py` reproduce the run with the design's seeds.
