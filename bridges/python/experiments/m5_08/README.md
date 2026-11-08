# Regime adaptation and transfer (#112)

SYNTHETIC. Two simulators with made-up parameters, each day calm but for segments of a stressed
or a thin regime at times fixed in `design.json` before any run. Training days hold thin
segments but no stress, so the stress is unseen when it comes. Nothing here is about a market:
**#112 makes no real-data claim** (see "Not run").

`design.json` fixes the data, splits, change times, detector thresholds, arms, seeds, decision
rule and compute; its `changes` record every choice made later, before any run, and what was
seen when. The protocol is #103's (`../m4_06`): 1 s rows, its kernel, execution, fee rule result
(-0.0001, with 0.0002 beside it), Box encoding with the reject policy, hyperparameters, budget
(2M decisions), training seeds 1 to 5, evaluation seed 101 and bootstrap, under the queue
standard for 1 s rows (rational queue, 1 lot per 500 ms, no minimum time).

## Steps

From `bridges/python`:

    cargo run --release --example regime_days -- experiments/m5_08/design.json \
        experiments/m4_06/design.json experiments/m5_02/design.json <data dir>
    cargo run --release --example regime_alarm -- experiments/m5_08/design.json <data dir> 4
    python experiments/m5_08/run.py check <data dir> <out dir> <commit>
    python experiments/m5_08/run.py classical <data dir> <out dir> <commit> <parallel>
    python experiments/m5_08/run.py train <data dir> <out dir> <commit> <parallel> history <sha256 of #103's validation pick>
    python experiments/m5_08/run.py freeze <data dir> <out dir>
    python experiments/m5_08/run.py holdout <data dir> <out dir> <commit> <frozen sha256> <parallel>
    python experiments/m5_08/run.py report <data dir> <out dir> target/release/examples/m4_05_intervals

`regime_days` prints the digests in `digests.json`; every day's rows are checked against them.

- **S1**: #103's queue-reactive simulator, one path a day, its regimes spliced at the change
  times: the book is kept, its queues clipped to the new regime's cap and limits.
- **S2**: #107 C's Hawkes flow at impact 1.0, its rates changed at the change times as #237
  stresses it (market order baseline x 2, the buy side x 1.5 more, diffusion x 3; thin halves
  the limit order rate). Queues and excitation carry across a change.
- **Detector** (`regime_alarm`): `VolatilityEstimator`'s EWMA (lambda 0.9) over the last 300
  rows against the training days' volatility, and `VPINCalculator` over the market orders known
  at each row; alarm on at a ratio of 1.8 or toxic VPIN, off below 1.3 and not toxic. Causal:
  each row reads nothing after it. One file per day and per simulator it is fitted on.
- **Arms**: no-trade; `fixed`, Avellaneda-Stoikov at #103's calm point; `adaptive`, the same
  point while the alarm is off and a stressed point while it is on (the scheduled baseline of
  #296, which quotes by the row the agent observed); `oracle`, a point per latent regime, chosen
  by a **coordinate search** on validation (each regime's point with the others at the fixed
  point): a privileged reference, not a policy and not the joint optimum; `learned`, #103's
  validation pick (`history`), retrained per simulator; `learned_detector`, the same with the
  alarm and the capped ratio appended to its observation.

## Results

First and only read of the holdout (days 20 to 24), at the fee -0.0001: `report-v2.json`
(sha256 `a1b207ba...b0c7`), recomputed after review from the outputs already written (see
"Corrections after the holdout"). Mean daily net over the 5 days; intervals are 90 % block
bootstrap of the paired daily difference.

**Decision: NO-GO for both comparisons, on every simulator and execution group.**

| comparison | group | mean | 90 % interval | fails |
|---|---|---|---|---|
| adaptive − fixed | S1, queue | +0.14 | [-0.0000004, 0.28] | interval, false alarms |
| adaptive − fixed | S1, immediate | +0.15 | [-0.35, 0.64] | interval, false alarms |
| adaptive − fixed | S2, queue | +28.9 | [-4.5, 64.3] | interval, false alarms |
| adaptive − fixed | S2, immediate | +19.4 | [-11.9, 51.2] | interval, false alarms |
| learned_detector − learned | S1, queue | +18.7 | [-2.7, 40.1] | interval, false alarms |
| learned_detector − learned | S1, immediate | +16.4 | [-8.0, 43.0] | interval, false alarms |
| learned_detector − learned | S2, queue | -22.3 | [-53.6, 8.9] | interval, drawdown, false alarms |
| learned_detector − learned | S2, immediate | -31.6 | [-55.5, -8.0] | interval, drawdown, false alarms |

- On S1 under the queue, the lower bound of adaptive against fixed is below 0 in the seventh
  digit: that is a NO-GO, not a near-go, and the false alarms fail it on their own.
- The detector catches the holdout's stress in time (48 s on S1, 3 s on S2) but raises 64.7
  alarm onsets a calm holdout day on S1 and 27.7 on S2, against at most 2. Lambda 0.9, taken
  from #237's 1 minute rows, is about 10 s of memory on 1 s rows; the validation days showed it
  before the holdout, and it ran as frozen. The thin change of day 23, beside the rule, is
  caught on S1 (33 s) and missed on S2.
- Arms on their own simulator, under the queue (no-trade is 0):

  | | fixed | adaptive | oracle | learned | learned_detector |
  |---|---|---|---|---|---|
  | S1 | -0.01 | +0.13 | +0.05 | -6.91 | +11.81 |
  | S2 | -58.1 | -29.2 | -2.9 | -7.3 | -29.7 |

  S2's flow moves the mid with its market orders, and every arm loses to it.
- Transfer (an arm worse than no-trade on the simulator it was not fitted on): fitted on S1 and
  run on S2, every arm fails (fixed -82.9, adaptive -19.1, oracle -35.0, learned -49.8,
  learned_detector -117.7), its detector on for most of each day, already on at both changes,
  so it misses the stress and catches thin only 629 s in; fitted on S2 and run on S1,
  the classical arms fail by about 0.01 (they barely fill), the S2-fitted detector never fires
  and misses both changes, and the learned arms gain (+3.5, +3.8).
- At the secondary fee 0.0002, on the own simulator under the queue: S1 fixed -4.13, adaptive
  -3.95; S2 fixed -72.6, adaptive -43.4.

### Post hoc, outside the rule

Chosen after seeing the validation alarms, on days already read: the detector alone with lambda
`0.9^(1/60)` (#237's wall-clock memory on 1 s rows), every other setting frozen. No arm is
evaluated under it. `sensitivity-v2.json` (sha256 `1a9b122b...d212`, with the delay as corrected in (j)):

| fitted -> data | onsets a calm day | stress delay | thin delay |
|---|---|---|---|
| S1 -> S1 | 257.0 (64.7 as frozen) | 158 s | 133 s |
| S2 -> S2 | 91.7 (27.7) | 298 s | 159 s |
| S1 -> S2 | 1.0, on all day (29.3) | missed (on before it) | 634 s |
| S2 -> S1 | 0, never on (0) | missed | missed |

A longer memory makes it noisier, not quieter. `calculate_ewma` starts from the sample variance
of the window's first 5 returns, which keeps a weight of `lambda^300`, about 0.59 at this
lambda: over a sliding window, the ratio then follows those 5 returns. Neither lambda gives a
usable detector here; that is a finding about this detector on 1 s rows, not a reason to retune
inside the rule.

## Corrections after the holdout

From review, recorded as changes (j) and (k) in `design.json`, with no retraining, retuning or
new holdout read:

- **(j) Delay.** The first report credited an alarm already on at a change's start with a delay
  of 0, against change (h), which counts the first onset at or after the start. Only onsets
  count now; an alarm latched on from before a change is a miss. Recomputed from the same alarm
  files and holdout rows (`report.json`, sha256 `53dc509a...27dc`, kept beside it):

  | fitted -> data | change | first report | now |
  |---|---|---|---|
  | S1 -> S1 | stress / thin | 48 s / 33 s | 48 s / 33 s |
  | S2 -> S2 | stress / thin | 3 s / missed | 3 s / missed |
  | S1 -> S2 | stress / thin | 0 s / 0 s | missed (on at start) / 629 s |
  | S2 -> S1 | stress / thin | missed / missed | missed / missed |

  Only the S1 -> S2 transfer row changes. The go rule reads the own simulator's rows, which do
  not change: every verdict is the same NO-GO, and the intervals, nets and transfer failures
  are identical.
- **(k) Flow window.** The design lists a 600 s flow window under the detector, but the rule's
  alarm reads only the volatility ratio and VPIN, and VPIN runs over its 20 buckets of 50 lots,
  restarting each day; the window is not used. It came from #237, where it fed the adaptive
  arm's flow-based quote, which no arm here has. It stays unused, since implementing it now
  would mean recomputing the alarms after the holdout read.
- The frozen alarm files are now selected by name, so a tagged sensitivity run's files no
  longer enter `freeze` or `holdout`.

- **Once-only holdout.** `run.py holdout` refuses a second run only when `holdout.json` exists,
  so a failure before it was written would have allowed a second evaluation with the same
  freeze (unlike #107's later attempt marker). This session's command log shows exactly one
  holdout attempt: 2026-11-07, 21:13Z to 21:46Z, rc 0, `holdout.json` written once.

## Not run

- **S3**, #106's neural simulator: not adopted by the freeze.
- **H**, the Hyperliquid capture: the capture is not on the host that runs this study, and its
  prices do not fit #103's kernel; a mapping into S1's units would be a choice made after the
  freeze that changes the economics, so a result under S1's checkpoints would not be
  interpretable. #112 makes no claim about real data.

## Limits

One synthetic path a day per simulator, 5 holdout days, one stress and one thin change in the
holdout, made-up parameters, a detector carried over from a 1 minute setting, and #103's single
set of hyperparameters.
