# Offline training with the regime in the input (#309)

Everything here is synthetic: #102's queue-reactive simulator, with made-up parameters, on new
days generated for this study. Its rows do not react to orders. The logs' rewards are the
simulator's consequences of the logged actions, not observed ones.

This is a new experiment motivated by a validation result of #280, not a confirmatory
continuation of it. #280's stop at its BC sanity rule stays as the result of its design.

`design.json` fixes before any run the regime input, the days, the arms and their settings, the
comparators, the intervals, the stop rules, the attempts and the holdout rule, and says what had
been seen. `changes` records its versions. `days_input.json` is #102's design with only a seed
offset added, which `qr_days` reads to generate the days; `digests.json` holds their digests.

- `regime.py`: the one-hot and the aliasing diagnosis.
- `run.py`: the steps, in the order its docstring gives.

## Audit of how #280 built its logs

From reading `experiments/m4_11/run.py` and `experiments/m4_07/run.py`:
- `log_job` records, per decision, the observation, the mask from `env.action_masks()`, the
  behaviour's action and `target_code`, as_grid's greedy action coded by `grid.code`.
- as_grid reads only its per-regime point, the raw position, the raw steps left and the mask, so
  its action is a deterministic function of those four. The observation holds position and
  steps left (features 6 and 10) but no regime and no per-regime input.
- No fault was found by reading. This study's logs use the same path, and add the raw position
  and steps left that the aliasing diagnosis keys on.

The `audit` step reports, before any training, the aliasing diagnosis and whether position and
steps left reach the network unclipped and one to one.

The audit found position and steps left one to one with the input, inside the columns' observed
ranges (position -0.95 to 1.0 in units of 20 lots, steps left up to 1.0), so neither is clipped
in these logs.

## Result

Run at commit 249ad33975012a00f52736ee080584fbc4763e2b. Attempt 1 stopped at the compute rule
(see the design's disclosure); attempt 2 is the study's last, from scratch. It used 4.40 process
hours of a ceiling of 8.43. The holdout took 171 s against a step limit of 161 s, which is recorded
as an excess and, by the design, did not stop it.

**Aliasing, on the training logs `[observed]`.** The highest agreement any classifier of the key
(position, steps left, mask) can reach with as_grid is 0.8146 without the regime and 1.0 with it.
With #104's state bin, position and ten bins of steps left it is 0.7931 and 0.9240. A network sees
other features too, so the first figure is not a bound on BC.

**BC sanity, threshold 90 % `[observed]`.** Every seed passes: validation agreement 0.9768 to 0.9818
on `standard_eps30` and 0.9835 to 0.9865 on `standard_eps10`. The input and the days both changed
from #280, so this does not by itself prove that aliasing caused #280's stop.

**Choice of alpha `[simulated]`, on validation.** Mean daily net: alpha 0.1, 0; alpha 1.0, -101.24;
alpha 10, -103.20. The rule picks alpha 0.1, which never trades and is the only one that does not
lose: the choice selected not trading rather than a strength of CQL.

**Holdout, mean daily net `[simulated]`, environment seed 101, 5 days per regime.**

| Arm | calm | stressed | thin |
|---|---|---|---|
| no_trade | 0 | 0 | 0 |
| maskable_ppo_regime | 0 | 0 | 0 |
| maskable_ppo_blind | 0 | 0 | 0 |
| cql on standard_eps30 | 0 | 0 | 0 |
| cql on standard_eps10 | -32.51 | -105.76 | -123.39 |
| bc on standard_eps30 | -41.69 | -125.03 | -140.52 |
| bc on standard_eps10 | -42.04 | -125.56 | -140.90 |
| as_grid | -42.71 | -125.03 | -141.20 |
| fixed_spread_4 | -74.11 | -178.90 | -170.65 |

No arm made money. Both PPO comparators and CQL on the primary dataset held at every decision: 0
fills on every holdout day. So the comparison of CQL with PPO at equal information on the primary
dataset compares two policies that do not trade, and says nothing about offline against
interactive training beyond that. There were no gate refusals or limit breaches.

**Paired differences of the pooled mean, 90 % interval, unadjusted over seven comparisons on
each dataset, 13 intervals in all `[simulated]`.** Only pooled intervals carry a claim; the design fixes why.

| Comparison | standard_eps30 | standard_eps10 |
|---|---|---|
| cql - maskable_ppo_regime | 0 [0, 0] | -87.22 [-89.03, -85.36] |
| bc - maskable_ppo_regime | -102.41 [-103.20, -101.59] | -102.84 [-103.64, -101.96] |
| cql - bc | 102.41 [101.59, 103.20] | 15.62 [14.38, 16.79] |
| cql - as_grid | 102.98 [102.04, 103.86] | 15.76 [14.30, 17.14] |
| bc - as_grid | 0.56 [0.17, 0.95] | 0.14 [-0.30, 0.57] |
| cql - no_trade | 0 [0, 0] | -87.22 [-89.03, -85.36] |

maskable_ppo_regime - maskable_ppo_blind is 0 [0, 0]: neither trades, so the value of the input to
PPO could not be measured here. CQL's alpha was chosen on validation and PPO's recipe was not
tuned here, so the comparisons are at equal information only.

bc - as_grid on `standard_eps30` excludes zero, at 0.56 a day [0.17, 0.95]: a tiny effect, unadjusted.

**Limit on the PPO comparators.** Why they hold at every decision was not diagnosed: this study does
not tell a learned hold from a training that failed. Their zero fills are not a finding about PPO,
and the comparisons against them say only that those policies, as trained here, did not trade.

No offline policy is better than an interactive one on these days: the only intervals excluding
zero against maskable_ppo_regime are worse for the offline arm. CQL's better net against as_grid
and BC reflects trading less in a setting where every trading arm loses. No claim is made about
any market or about an estimated regime.

**Support `[observed]`.** Every holdout decision of every offline arm has a behaviour propensity
above 0, as designed. The share of decisions whose (state bin, action) pair the training logs
hold 0 times is at most 0.0016, and fewer than 10 times at most 0.0082.

## Evidence

`results/` holds the reports of attempt 2, with local directories replaced by placeholders, so
their hashes differ from the run's. The run used commit 249ad33975012a00f52736ee080584fbc4763e2b,
whose `design.json` has sha256
fb90fae88399e37307a7aa0efa3f68e9b67ede8eb4017fedf889804d4307c274, and whose `frozen.json` had
sha256 69f04c7f690e7133d377272745244b84fc5d6e71db17dca26dabc3bc3613cb56. After the run, the words
of `changes` in `design.json` were edited without changing their meaning.
