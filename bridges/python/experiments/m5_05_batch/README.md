# A frequent batch auction toy, reported apart (#289)

Everything here is **synthetic**. The study uses a batch auction toy with a made-up background
flow. Its execution rules are not those of the crate's continuous order book, so no number here
is compared with #110 or any order book result, nor the reverse.

The toy collects orders for 500 ms and clears them at one uniform price. Submit and cancel
delays are 30, 60 or 100 ms, and rows are 100 ms apart.
- `fba.py` states the clearing rule, the timing rule and the limits, with hand-computed fixtures
  in `tests_py/test_m5_05_batch.py`.
- `env.py` is the environment.
- `design.json` fixes the arms, the budget, the plateau criterion and the decision rule, before
  any run. Its `changes` list what changed before the first run.
- `run.py` holds the steps, in the order its docstring gives.

From `bridges/python`:

```
python experiments/m5_05_batch/run.py <step> <out dir> <commit> <parallel> [<frozen sha256> | <intervals binary>]
```

## The run

All steps ran at commit `d440433` on synthetic days drawn from their seeds (`2890000 + day`),
with no stored data.

| step | processes | time | estimate | result |
|---|---|---|---|---|
| predictor | 1 | 21 s | | σ 0.00565 per row; least squares on 1,007,930 rows |
| fee_rule | 3 | 3:59 | ~5 min | fee 0 (at 0.0002 the training-day best Avellaneda-Stoikov point loses) |
| check | 2 | 3 s | | passes: fills at 30 ms (4989) and 100 ms (5001), with nets 6.15 and 1.91 |
| classical | 3 | 5:52 | ~7 min | teachers and validation-selected points per latency |
| pilot | 1 | 2:23 | 15 to 25 min | **not plateaued** |
| train | 3 | 16:31 | 2.5 to 4 h | 30 of 30 runs |
| freeze | 1 | | | `frozen.json` |
| holdout | 3 | 8:37 | | 360 of 360 jobs, read once behind the `holdout.started` marker |

The estimates were the design's for training, and mine for the steps before it. No step overran
its estimate by more than the 50 % the design's stop rule allows. The design gives no estimate
for the holdout: it took 8:37 against my informal estimate of about 5 min.

**The pilot curve did not plateau.** It is the deterministic policy's mean daily net on training
days 1 and 2, every 50,000 decisions:

9.52, 20.35, 27.51, 30.41, 24.90, 23.04, 23.76, 23.89, 17.12, 18.45.

The last three average 19.82, against 23.90 for the three before; the gap of 4.08 is beyond 10 %.
By the design the budget stays at 500,000 decisions.

## Holdout results [simulated]

Holdout days 20 to 24. A learned arm's day is the mean over its five seeds.
- The intervals are 90 % stationary block bootstrap intervals of the paired daily difference
  (2,000 replicates, mean block 2 days, seed 102).
- The design's evaluation seed, 101, is not read by `run.py`: `episode` resets each day with seed
  0, and a day's flow comes from its day seed. The policies are deterministic, so the evaluation
  has no other randomness.

**At the chosen fee, 0:**

| latency | Avellaneda-Stoikov | full | no_duration | full − no_duration | the rule's verdict |
|---|---|---|---|---|---|
| 30 ms | 15.47 | 10.52 | 0.43 | 10.09 (0.41 to 19.41) | duration helps, weakly; see below |
| 60 ms | 15.06 | 8.61 | 0.43 | 8.18 (−0.12 to 16.48) | no claim |
| 100 ms | 13.52 | 4.09 | 1.88 | 2.22 (−2.18 to 6.61) | no claim; full breached its position limit |

The columns are mean daily nets. Against not trading:
- Avellaneda-Stoikov and full are above it at every latency.
- no_duration is equivalent to not trading at every latency.

**At the secondary fee, 0.0002**, every trading arm loses and is below not trading. Duration
does not help at any latency: full − no_duration is −42.82, −24.51 and −17.72 at 30, 60 and
100 ms, each interval below 0.

**Durations**, as shares of the full arm's decisions. Unlike #110's order book arms, the full arm
uses longer durations:

| latency | 1 row | 5 rows | 10 rows | 30 rows |
|---|---|---|---|---|
| 30 ms | 70.9 % | 6.4 % | 13.9 % | 8.8 % |
| 60 ms | 81.2 % | 7.9 % | 6.8 % | 4.0 % |
| 100 ms | 52.9 % | 17.4 % | 17.2 % | 12.5 % |

### How the position was measured, found after the results were known

**At the holdout, the position was read once per step.** `run.py` read the position, its peak and
the equity behind the drawdown at the end of each step, at commit `d440433`. A step of the full arm
lasts up to 30 rows, so up to six clearings. For the full arm, the holdout's peak position, breach
and maximum drawdown are therefore step-end samples, which understate what happened inside a step.
- Its breach counts at 100 ms (seed 1 on 5 days, seed 4 on 1) are a lower bound.
- no_duration and Avellaneda-Stoikov decide on every row. A row spans at most one clearing, so
  their figures are exact.
- Full's nets stand as measured. But at 100 ms they do not show that its limit held, and at no
  latency are its drawdowns comparable on risk with the arms read on every row.

Since then, `episode` reads all three at every clearing (`env.marks`), with tests. The holdout
was not read again.

**Why no breach was possible at 30 or 60 ms.** The rule's verdict at 30 ms needs that no arm
breached its limit. That condition is established here by the mechanism below and by tests,
written after the holdout read. It is not measured on the holdout.

In `env.py` at `d440433`:
- **A replacement lands all at once.** A replacement sent at time `t` cancels every resting
  order and places its new orders at the same instant, `a = t + delay`.
- **Only one of them is live at a close.** At a close, an order being cancelled is live only if
  `a` is at or after the close, and a new order only if `a` is before it. So the old and new
  orders are never live at the same close.
- **New orders are sized on a position that is still current when they arrive.** Each order is
  sized at submission so that it cannot take the position past its limit. Decisions fall on
  rows, multiples of 100 ms, and closes are multiples of 500 ms. At a delay of 30 or 60 ms no
  close falls in `(t, a]`, so the position the order was sized on is still the position when it
  arrives.
- **Fills and durations add nothing.** Later partial fills only shrink an order, since the
  clearing never fills an order beyond its lots
  (`test_fills_balance_respect_each_order_and_keep_price_priority`). A longer duration only
  skips decisions; it adds no order.

Each side therefore has at most one live order at a close, within the limit.

At 100 ms, `a` is the close itself when the decision is on a batch's last row. The order being
replaced is still live at that close and can fill there, while the new one, sized before that
fill, acts at the next close. So the position can pass the limit.

A test drives aggressive random play on six seeds. It finds no breach at 30 or 60 ms at any
clearing, and a breach at 100 ms, which shows the test can detect one.

**A diagnostic on validation days.** `audit.py` runs the frozen learned checkpoints, verified
against `frozen.json`, on validation days 15 to 19 at the chosen fee. It reads the position at
every clearing, at commit `0e67b57`. It reads no holdout day and decides nothing; it could only
have withdrawn the claim.

| arm | latency | days | breach days | largest position |
|---|---|---|---|---|
| full | 30 ms | 25 | 0 | 20 |
| full | 60 ms | 25 | 0 | 20 |
| full | 100 ms | 25 | **5** | 21 |
| no_duration | 30, 60, 100 ms | 25 each | 0 | 20 |

The limit is 20 lots. The audit agrees with the mechanism: no breach at 30 or 60 ms, and
breaches at 100 ms. `audit.json` sha256 is
`957fc145326b9f8fd5d36fb7ac8e0a48684d3c584f1110622cfa9eb105e3d6bd`.

**Read with these limits:**
- The duration effect at 30 ms is a weak result at one latency, simulated. It is one comparison
  of three at fee 0, with a lower bound of 0.41 on a difference of 10.09. No multiplicity rule was
  frozen, and its no-breach condition rests on the mechanism above, not on the holdout's
  measurement.
- At 100 ms the rule makes no duration claim.
- 30 and 60 ms are close to one treatment. Decisions sit on 100 ms rows, so at either delay an
  order lands in the batch it was sent in, before its close. The two differ only in where it
  lands among the background orders of that batch.
- At every latency the classical Avellaneda-Stoikov arm's mean exceeds both learned arms'.
- Late messages (orders and cancels that acted a batch after the one they were sent in) occur
  only at 100 ms.
- Not reported:
  - the design's spread capture against the mark, and the mean absolute position, were not
    computed by `run.py`;
  - the largest absolute position is reported only as the peak above;
  - the design's `not_run` arms (no teacher, no predictor) were left out for compute.

## Hashes

| file | sha256 |
|---|---|
| `predictor.json` | `dddb40551facba1fbfbe8fcd8911e9db1f24cefe9190721bea44e8bfe27e7362` |
| `fee_rule.json` | `75f413ba2e3edbfbd73f05b444e7dc70d638300f3ad56ecd40060cd14cfa9285` |
| `check.json` | `938b269484d0c8825563aac39cd21724dbe6353a68a6fdf24123146e0f599fbf` |
| `classical.json` | `d9cb4b9db21fe9dfd944169a8fc6e36d889111e292360d97a9c395a43ca7478d` |
| `pilot.json` | `a067e8b2f573fc8c1ca7b35cb311c4ce7023efe19b1f1674d8148eee2a957b82` |
| `frozen.json` | `e015d8feb23ab95a79a02c0b40dd26cc8c2eec66e38f8ed319f6d9c8bdcd586a` |
| `holdout.json` | `a3f03894bba75985530599db7ffe5eba4514ca2accf46da21ff88fc9adec55ad` |
| `report.json` | `c82a20bcb2a2565a355876063f36e43e1aac2d5fccac4d0c246c7cf73cae681f` |
| `audit.json` (validation, diagnostic) | `957fc145326b9f8fd5d36fb7ac8e0a48684d3c584f1110622cfa9eb105e3d6bd` |
