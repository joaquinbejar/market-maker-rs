# Offline training against interactive training (#280)

Everything here is synthetic. It uses #102's queue-reactive days: a simulator with made-up
parameters, whose rows do not react to orders. The logs' rewards are therefore the simulator's
consequences of the logged actions, not observed ones. Those days, the simulator and its
parameters bound what this study can show.

The question is what a policy trained from a fixed log achieves, against one trained
interactively on the same simulator, days and execution.

- `design.json` fixes the following before any log, training or run:
  - the executions and the datasets;
  - the arms (behaviour cloning, and a discrete conservative Q-learning, both restricted by the
    mask) with their settings, and the comparators;
  - the evaluation, the intervals and the support diagnostics;
  - the stop rules and the holdout rule.
- `offline.py` holds the arms: the joint actions and their mask, the losses and the training
  loop.
- `run.py` holds the steps, in the order its docstring gives.

## Datasets

- **Primary.** Logs generated here by #104's behaviours (as_grid, epsilon-greedy at 0.3 and
  0.1) under the queue standard for 10 s rows: queue position cleared at one lot per 5000 ms,
  with no minimum time in queue. A queue gate comes first: on validation days, as_grid must
  fill and must end with a net different from immediate fill's.
- **Secondary.** #104's frozen logs, as they are. They were made under #102's main execution,
  which is immediate fill in effect.

Every value is labelled `[observed]` (read from a log) or `[simulated]` (an interactive run in
the same simulator). Nothing here is an off-policy estimate.

From `bridges/python`:

```
python experiments/m4_11/run.py <step> <#102 data dir> <#102 run dir> <#104 out dir> <out dir> <commit> [args]
```

No run reads the holdout days before `freeze` has recorded the hashes of the design, the code,
the chosen alpha, every log and every checkpoint. No log is generated on them.

## Result: the study stopped at the BC sanity rule

The design's BC sanity rule says that on each dataset's validation logs, every BC seed must
choose as_grid's action on at least 90 % of the logged decisions; otherwise the study stops.
On the primary dataset at epsilon 0.3 (`standard_eps30`), no seed reached it:

| BC seed | agreement, validation | calm | stressed | thin | agreement, train |
|---|---|---|---|---|---|
| 1 | 0.8132 | 0.7015 | 0.8392 | 0.8988 | 0.8113 |
| 2 | 0.8144 | 0.6077 | 0.9006 | 0.9349 | 0.8118 |
| 3 | 0.8197 | 0.6740 | 0.8690 | 0.9163 | 0.8168 |
| 4 | 0.8194 | 0.6800 | 0.8626 | 0.9156 | 0.8160 |
| 5 | 0.8187 | 0.7220 | 0.8357 | 0.8984 | 0.8161 |

These are shares of 647,925 validation decisions and 1,814,190 training decisions per seed
`[observed]`, all in the logs; the per-regime columns are validation. The study stopped there:
- the sanity checks of the other two datasets did not run;
- neither did the choice of CQL's alpha, the consistency check or the freeze;
- the holdout was never read.

The CQL runs were trained but not evaluated. Nothing here compares an offline arm with an
interactive one.

`sanity_diagnosis.py` gives the breakdown above. It reads only the logs and the BC
checkpoints, each checked against its recorded sha256, and finds no fault in the pipeline:
- The logs encode the target consistently. The logged action is as_grid's on 70.06 % of the
  decisions, as epsilon 0.3 implies, and as_grid never holds.
- BC does not overfit: its agreement on the training logs is within 0.004 of validation.
- What it misses depends on the regime, and the seeds trade one regime against another
  (seed 1: calm 0.70, stressed 0.84; seed 2: calm 0.61, stressed 0.90).
- About 39 % of the misses where both quote are within one offset step of as_grid's quote.

The likely reading is that the pooled observation does not determine as_grid's choice; the
difference between regimes is consistent with this, but does not by itself prove it. #102's
as_grid takes its point and its volatility per regime. The observation has no regime or
volatility feature: the 11 row features (spread, mid change, row gap, the two sizes,
imbalance, position, entry gap, open orders, orders awaiting a cancel, steps left) and their
presence flags. A single classifier over the three regimes' logs therefore compromises
between them. On this reading, offline BC from these pooled logs cannot clone a
regime-dependent behaviour policy whose observation lacks the regime. By the design's rule, a BC that is not as_grid's
clone means the pipeline cannot answer the question, and the study stops without changing the
design after seeing the data.

A study that gives the offline arms the regime, or as_grid's per-regime inputs, would be a
new experiment motivated by this validation result, not a continuation of this design: a
fresh design and its own holdout policy, left to #309. This stop stays as the result of the
registered design.
