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
