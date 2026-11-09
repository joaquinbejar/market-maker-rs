# #282: policies against a market that reacts to their orders: study not completed

**Status: the study did not complete. There is no result for any policy under impact, and no
claim is made.** All data here is synthetic.

This directory holds the design (`design.json`, version 4), the runner (`run.py`) and the
holdout report (`report.py`). They are committed so the study can be read, reviewed and rerun.

## What happened

| Step | Outcome |
| --- | --- |
| Selection at kappa = 0 | Ran (4185 s on 4 processes). |
| First kappa > 0 selection | Stopped under the overrun rule on a contended machine (8652 s on 4 processes). |
| Design version 4 | Written after that stop: kappa0 left the adversary's grid, and a hard ceiling of 97.5 process-hours for the whole study was set, 9 of them reserved for the holdout. |
| Second kappa > 0 selection | Stopped by its limit at 66813 s on 4 processes, with 7617 of its 8981 jobs written. |
| `selection.json`, freeze, holdout | Not produced. Nothing was frozen and the holdout was never run. |

Spent: 318600 process-seconds, 88.5 process-hours. The ceiling was kept and the study was not
restarted past it: finishing the selection and running the holdout would have cost about 108
process-hours. The pace of the second selection, 8.8 s per job, was slower than its projection on
the maintainer's machine, which other work was also using for part of the run. How much of the
gap was that load and how much the job mix was not separated.

## What can and cannot be read from this

- The counts above come from the journal's keys and the budget file. No kappa > 0 selection
  value was read or summarized, for this report or for any decision about stopping.
- Design version 4 was a change made after the first selection had started. Its 1260 rows at
  kappa0 were discarded unread, as the design's `changes` entry states.
- The adversary's grid of version 4 covers only {0, 2 kappa0}. That reduction was never tested.
- The question of #282, how the learned policies behave when the market reacts to their orders,
  stays open. Answering it needs a new budget, and a selection run on it from a recorded design;
  the partial journal is not a result.
- The simulator's opt-in reaction to the agent's orders, already merged with its own tests, is
  not affected by this.

## Running it again

From `bridges/python`, detached from any terminal session:

    python experiments/m5_04/run.py selection <run dir> <out> <commit> <selection of #108>
    python experiments/m5_04/run.py freeze <run dir> <out> <commit>
    python experiments/m5_04/run.py holdout <run dir> <out> <commit> <frozen sha256>

The ceiling in `run.py` and `budget.json` in the output directory apply to the whole study,
restarts included. A selection journal resumes, skipping the jobs it holds. The holdout records
its attempt with an exclusive marker before any job and runs once.
