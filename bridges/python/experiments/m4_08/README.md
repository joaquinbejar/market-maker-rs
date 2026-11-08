# Constrained learning against a reward penalty (#105)

Synthetic: #102's queue-reactive regimes (calm, stressed, thin) on its own days, sampled every
10 seconds, with shocked copies of the validation and holdout days for the stress rows. One
simulator, made-up parameters and 24 days per regime bound what this can show. Nothing here
is a safety guarantee.

`design.json` fixes the data, splits, execution, costs and budgets, arms, seeds, decision rule
and compute before any run, and records every later change in `changes` with what had been
seen.

- **Costs**, measured after each step apart from the reward (`costs.py`): `c_inventory` and
  `c_notional` are constrained, with budgets per step; `c_margin`, a margin-rs maintenance
  proxy, is reported and not constrained.
- **Hard limits:** the kernel's `RiskLimits`, with `max_notional` from the margin rule of the
  capital (125 / 0.05 = 2500), the same for every arm and the final authority. The costs never
  change what the kernel admits. `tests_py/test_m4_08_limits.py` shows that a policy
  proposing beyond them never gets the order past the gate.
- **Arms:** no-trade; Avellaneda-Stoikov selected on validation per regime; OmniSafe 0.5.0
  PPO on the kernel's reward; PPO with the reward less λ times the constrained cost, λ chosen on
  validation; OmniSafe CPO with the cost limit of the budgets. PPOLag failed the toy gate
  (`toy_cmdp.py`, `toy_gate.json`) and is not an arm.
- **Decision rule:** CPO against the penalized PPO, a paired 90 % block bootstrap over days of
  the daily net and of the daily constrained cost: constrained better when the net interval is
  not below 0, the cost interval is below 0 and neither arm breached a limit. Every learned arm
  is also read against no-trade and against PPO, the same way.

From `bridges/python`, in the experiment's own venv (`omnisafe-lock.txt`, Python 3.11,
`MUJOCO_GL=disable`):

    python experiments/m4_08/shocks.py <#102 data dir> <shocks dir> > experiments/m4_08/shocks.json
    python experiments/m4_08/run.py <step> <#102 data dir> <shocks dir> <out dir> <commit> [<frozen sha256>]

with the steps `fee_rule`, `classical_selection`, `training`, `penalty_selection`, `freeze`
and `holdout`, then, with the bridge's venv,

    python experiments/m4_08/report.py <out dir> target/release/examples/m4_05_intervals

`report.py` fixes how the holdout is read and was committed (70a4322) before it ran on the
holdout.

## The run

- Training: 25 runs (PPO and CPO with 5 seeds each, the penalized PPO with 3 λ and 5 seeds), none
  stopped, 184 minutes against an estimate of 270.
- `shocks.py` also writes shocked copies of the validation days, as the design lists them, but
  no step reads them: the classical and penalty selections run on the unshocked validation
  days, and only the holdout reads shocked days.
- The freeze kept the 15 runs of PPO, CPO and the penalized PPO at the chosen λ.
- The holdout evaluated 765 jobs, every one planned: the 15 runs, no-trade and the selected
  Avellaneda-Stoikov point, each on 5 holdout days of 3 regimes, unshocked and under each of
  the two shocks (15 x 45 + 2 x 45). The fee rule chose 0.0002, the secondary fee, so there is
  no separate secondary row.
- Found in review after the holdout: the run (925928b) checked each checkpoint's hash at the
  freeze and the holdout but not its OmniSafe `config.json`, which sets the actor and the
  observation normalizer and whose hash training recorded. The runner now checks both before
  any policy loads, at the freeze, the holdout and every evaluation. For this run, every one of
  the 25 configurations still matches the hash training recorded, and none was written after
  training (the last at 23:21 on 7 November; the freeze and the holdout came after).
- The holdout took 6 minutes against an estimate of 180. The estimates of the evaluation steps
  were set before any evaluation was timed, as 60 minutes on 6 processes scaled by 3 for 2;
  an evaluated day is 8640 steps, and a job took about 0.9 seconds on the holdout (0.23 for
  the classical jobs of the fee rule). The fee rule (2 minutes against 90), the classical
  selection and the penalty selection were as far below theirs. Only training, which the
  pilot timed, was near its estimate. No step overran.

The results are withheld here on purpose, as #103's are: another study's holdout is pending.
They come from `report.json`, whose hash the pull request commits to.

## The second, margin-stress holdout

`stress_design.json` fixes, before it runs, a second holdout that reads the frozen runs under
margin-rs's maintenance rule; it does not revise the first holdout's decisions. Its days are
days 25 to 29 of #102's regimes, by the same seed rule (`stress_days.json`): generated with days
1 to 24, which reproduce #102's digests, and recorded in `stress_digests.json`, with their
shocked copies in `stress_shocks.json`. The frozen runs, no-trade and the selected
Avellaneda-Stoikov points run on them at the frozen fee, unshocked and under each shock, and
after each step the account (one linear perpetual position on 125 of collateral) is assessed
by `margin_rs::maintenance::assess` under `linear-im/1`, from the previous step's state
(`bridges/python/src/maintenance.rs`, `mm_rl_bridge.maintenance_path`). A state is recorded,
never acted on. `report_stress.py` reads the share of steps not Healthy per day, with the
entries into ReduceOnly or Default (where a closeout would start), the largest shortfall, and
the positions and fills beside them; an arm with no fill is Healthy by construction, and a
comparison against it is flagged as trading against not trading.

    python experiments/m4_08/stress.py days <qr_days output for stress_days.json>
    python experiments/m4_08/stress.py shocks <stress data dir> <shocks dir>
    python experiments/m4_08/stress.py holdout <stress data dir> <shocks dir> <out dir> <stress out dir> <commit> <frozen sha256>
    python experiments/m4_08/report_stress.py <out dir> <stress out dir> target/release/examples/m4_05_intervals

## Bounds

The comparisons are many: three rows, three regimes and pooled, three kinds; each is a 90 %
interval on 5 or 15 days, with no correction for their number. The margin gate is not wired
into the bridge, so the margin is represented by limits derived from its rule and by the
`c_margin` proxy; margin-rs#7's maintenance margin and liquidation scenarios are not used.
