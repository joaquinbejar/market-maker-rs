# HSN against temporal controls (#107)

SYNTHETIC. A learnable mixture-of-exponentials memory (HSN, arXiv 2610.16524) against MLP,
frame-stack, fixed-EMA, GRU, LSTM and parameter-matched controls, in three parts. Each design
was frozen before its runs (`design_paper.json` for A, `design.json` for B and C), with the
changes made later recorded in it; every holdout was read once.

## Part A: the paper's single-asset setting

`paper_env.py` rebuilds the paper's environment (two Hawkes channels, Ogata thinning, fill
probability `exp(-10 delta)`, inventory at most 5, spread in [0.005, 0.30]); `run_paper.py`
trains every arm with PPO on 30 seeds per row and reads the evaluation once.

| row | HSN | best control | observed | paper (HSN, best baseline) | paper's claim |
|---|---|---|---|---|---|
| exponential, n = 0.90 | 0.0282 | LSTM 0.0291 | tie (Welch p >= 0.12) | 0.0208, MLP + frame-stack 0.0211 | tie: **reproduced** |
| power law, eps = 0.15 | 0.0230 | MLP + frame-stack 0.0236 | tie (p >= 0.10) | 0.0216, MLP + frame-stack 0.0163 | HSN +32.5 %: **not reproduced** |

- Only the class of the result (tie or gain) is compared with the paper. The levels differ: ours
  are above the paper's for every arm (HSN 0.0282 against 0.0208, and 0.0230 against 0.0216;
  frame-stack 0.0287 against 0.0211, and 0.0236 against 0.0163). The cause is not separated:
  the rebuild follows the paper's description, and whatever it leaves out may account for it.
- The agent sees few market orders: 1.11 and 0.91 per episode on the 100 evaluation episodes
  (seed base 1000000), by the simulator's count. HSN's decays end where they started: the 30
  seeds' mean betas are 6.01 and 0.41 from 6.0 and 0.4 (exponential), and 4.96 and 0.30 from 5.0
  and 0.3 (power law), with standard deviations of 0.06 or less.
- A fixed spread of 0.10, with no learning, scores 0.0300 on the exponential row, above every
  learned arm, and 0.0230 on the power-law row, within 0.0006 below every learned arm
  (`sanity.json`, a post-freeze check on other seeds, base 5000000, so not paired with the arms).
  Both are above the paper's reported controls (0.0211 and 0.0163). This is the main finding of
  A: in this setting a fixed quote does as well as any memory.
- Two loads of every checkpoint give identical actions.

## Parts B and C: #103's protocol

1 s rows, #103's kernel, execution, Box encoding with the reject policy, hyperparameters, budget
(2M decisions), seeds 1 to 5 and bootstrap, under the queue standard for 1 s rows (rational
queue, 1 lot per 500 ms, no minimum time). `check_queue.py` showed, before training, that a
classical arm fills and the queue binds. The windowed arms (`gru_window`, `hsn`) are width-matched
to #103's MLP (11547 and 11719 parameters against 11659).

- **B**: #103's queue-reactive days (calm, stressed, thin), one policy per arm on every regime
  pooled. Its holdout days are #103's, which #103 read first.
- **C**: Hawkes order flow that moves the mid (`examples/hawkes_days.rs`), at impacts 0, 0.25 and
  1.0, one policy per arm and impact; its classical points were selected on its own validation.

The decision per scope is HSN against the control that validation chose, with a 90 % block
bootstrap of the paired daily net: `hsn_helps` needs the lower bound above 0, no breach, a
drawdown at most the design's factor of the control's, and HSN not equivalent to no-trade;
`revise` if the mean is above 0 otherwise; `no_gain` else. At the fee -0.0001 (the secondary fee
0.0002 beside it):

| part | scope | control | HSN − control | HSN − no-trade | decision (0.0002) |
|---|---|---|---|---|---|
| B | calm | history | -23.64 [-37.58, -9.71] | -8.67 [-13.56, -3.78] | no_gain (no_gain) |
| B | stressed | history | -4.03 [-30.91, 25.46] | -10.04 [-21.38, 2.17] | no_gain (no_gain) |
| B | thin | history | -7.45 [-22.50, 7.22] | -16.06 [-23.43, -9.19] | no_gain (no_gain) |
| B | pooled | history | -11.71 [-25.94, 3.41] | -11.59 [-16.63, -6.57] | no_gain (no_gain) |
| C | impact 0 | mlp | +1.23 [-7.62, 10.09] | -1.81 [-3.94, 0.13] | revise (revise) |
| C | impact 0.25 | lstm | +17.95 [-4.38, 45.00] | +0.76 [-3.07, 4.85] | revise (revise) |
| C | impact 1.0 | lstm | +1.15 [-26.28, 29.14] | -12.51 [-13.81, -11.20] | revise (no_gain) |

- **HSN helps nowhere.** On B its mean is below `history` in every scope, but only calm's
  interval excludes 0; against not trading, calm, thin and pooled exclude 0 and stressed does not.
  On C no interval against the control clears 0 at the design's fee.
- At the secondary fee, C at impact 0.25 has HSN above LSTM with the interval above 0 ([32.14,
  82.33]), but HSN there is equivalent to not trading ([-6.87, 1.06]): `revise`, not a gain. It
  is beside the rule, which reads the design's fee.
- No breach anywhere. Two loads of every checkpoint (30 in B, 90 in C) give identical actions.

## Corrections after the holdout

From review, after every holdout had been read; nothing was rerun:

- **One attempt per freeze.** `run_paper.py evaluate` and `run_market.py holdout` refused a
  second run only once their result file existed, so a failure before it was written would have
  allowed a second evaluation with the same freeze. Both now claim an attempt marker with
  `O_EXCL` before any evaluation is dispatched, and refuse a retry unless `--recover <reason>`
  is given, which is appended to the marker and carried in the result (`attempt.py`). The guard
  was added after the reads. This session's command log shows exactly one evaluation attempt
  for part A (2026-11-07 13:41Z) and exactly one holdout attempt for B and C together
  (2026-11-08 06:29Z to 08:30Z, both parts rc 0, each `holdout.json` written once).
- **The decision rule** let `hsn_helps` through for an HSN wholly below no-trade, since that is
  "not equivalent" to not trading. It now also needs HSN above no-trade. Applied to the stored
  intervals, no decision of B or C changes.

## Steps

From `bridges/python`:

    python experiments/m5_02/run_paper.py <step> ...        # part A, see its docstring
    python experiments/m5_02/check_queue.py <part> <data dir> <out dir>
    python experiments/m5_02/run_market.py <step> <part> ... # B (market) and C (causal)

The days of B are #103's (`examples/qr_days`); those of C come from `examples/hawkes_days` with
the digests in `causal_digests.json`.

## Limits

One synthetic simulator per part with made-up parameters, 5 holdout days per group, #103's
single set of hyperparameters, and a setting in A where the agent sees very few market orders.
