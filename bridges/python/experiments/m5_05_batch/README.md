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

Holdout days 20 to 24, evaluation seed 101. A learned arm's day is the mean over its five seeds.
The intervals are 90 % stationary block bootstrap intervals of the paired daily difference
(2,000 replicates, mean block 2 days, seed 102).

**At the chosen fee, 0:**

| latency | Avellaneda-Stoikov | full | no_duration | full − no_duration | duration helps |
|---|---|---|---|---|---|
| 30 ms | 15.47 | 10.52 | 0.43 | 10.09 (0.42 to 19.41) | **yes** |
| 60 ms | 15.06 | 8.61 | 0.43 | 8.18 (−0.12 to 16.48) | no |
| 100 ms | 13.52 | 4.09 | 1.88 | 2.21 (−2.18 to 6.61) | no; full breached its position limit |

The columns are mean daily nets. Against not trading:
- Avellaneda-Stoikov and full are above it at every latency.
- no_duration is equivalent to not trading at every latency.

**At the secondary fee, 0.0002**, every trading arm loses and is below not trading. Duration
does not help at any latency: full − no_duration is −42.82, −24.51 and −17.73 at 30, 60 and
100 ms, each interval below 0.

**Durations.** Unlike #110's order book arms, the full arm uses longer durations:

| latency | 1 row | 5 rows | 10 rows | 30 rows |
|---|---|---|---|---|
| 30 ms | 70.9 % | 6.4 % | 13.9 % | 8.8 % |
| 60 ms | 81.2 % | 7.9 % | 6.8 % | 4.0 % |
| 100 ms | 52.9 % | 17.4 % | 17.2 % | 12.5 % |

**Read with these limits:**
- The duration effect at 30 ms is one comparison of three at fee 0, with a lower bound near 0
  (0.42). No multiplicity rule was frozen.
- At 100 ms two of full's seeds breached the position limit: seed 1 on 5 days, seed 4 on 1. The
  rule therefore makes no duration claim there.
- At every latency the classical Avellaneda-Stoikov arm's mean exceeds both learned arms'.
- Late messages (orders and cancels that acted a batch after the one they were sent for) occur
  only at 100 ms.

The design's `not_run` arms (no teacher, no predictor) were left out for compute.

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
