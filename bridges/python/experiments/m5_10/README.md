# Robustness to bounded Hawkes flow and flow impact (#108 part 2)

SYNTHETIC. Everything here comes from #107 part C's Hawkes simulator with made-up parameters, and
says nothing about any real venue. Impact is the market orders' impact on the mid; the agent's
own fills do not move it. The policies are #107 part C's frozen ones; nothing was trained here.

`design.json` (v7) fixed the study before any policy ran; its `changes` log records each amendment
and what had been seen when it was made. #295 (part 1 of #108, `m5_03`) already ran bounded flow
perturbations of #102's queue-reactive regimes; this part adds Hawkes flow, flow impact together
with it, and #281's unrun impact condition.

## Part 1: perturbed Hawkes days

24 points per impact setting (market rate x1, 1.5, 2; limit rate x1, 0.7; branching x1, 1.2;
the setting's impact or the next higher). Arms: no-trade, Avellaneda-Stoikov and GLFT at #107's
points, #107's frozen mlp and lstm (5 seeds each, one policy). The adversary picks, per policy,
the point with the largest mean daily loss on 10 selection days; the holdout is 30 fresh days,
read once. Intervals are paired 90 % stationary block bootstraps (mean block 2 days, 2000
replicates, seed 108).

**The identifiability test removed 14 of 24 points per setting.** Before any policy ran, the frozen rate
test removed them, the fixed stress included; every impact slope test passed. Realised over formula rate falls as the empty-queue share grows (0.97 at 3 %, 0.70 at
19 %); that this is the suppression at the empty queue the design named is untested. The rule was
not changed. So the **primary** analysis (the only one feeding a reading) searches the 10
surviving points only, and fixed stress is reported not run in every setting. Its worst case is
the worst over those points, not over the declared grid. A **secondary**,
exploratory analysis of the 14 removed points, labelled "not identified by the frozen rate test",
was decided after `identify.json` was seen and before any policy ran; it feeds no decision and
never changed the primary's pick. Never compare the two adversaries as a robustness test of each
other.

### Primary

Daily net per day, against the same arm unperturbed and against no-trade under the same condition.

| setting | arm | condition | mean daily net | CVaR5 loss | against unperturbed, 90 % | against no-trade, 90 % |
|---|---|---|---|---|---|---|
| impact_0 | avellaneda_stoikov | adversary | -7.8 | 48.4 | [-10.6, +0.4] | [-12.4, -3.1] |
| impact_0 | avellaneda_stoikov | impact_only | -6.5 | 48.0 | [-9.6, +1.3] | [-12.1, -1.3] |
| impact_0 | glft | adversary | -1.3 | 11.8 | [-3.2, +1.0] | [-2.9, +0.3] |
| impact_0 | glft | impact_only | -1.1 | 10.6 | [-2.9, +1.2] | [-2.6, +0.5] |
| impact_0 | lstm | adversary | -18.5 | 38.0 | [-13.0, -8.7] | [-22.2, -14.6] |
| impact_0 | lstm | impact_only | -14.3 | 37.5 | [-9.2, -4.3] | [-18.3, -10.4] |
| impact_0 | mlp | adversary | -0.8 | 32.0 | [-4.8, +1.0] | [-6.5, +5.0] |
| impact_0 | mlp | impact_only | +0.6 | 25.2 | [-3.1, +1.9] | [-4.2, +5.5] |
| impact_high | avellaneda_stoikov | adversary | -131.7 | 152.6 | [-133.5, -126.7] | [-135.1, -128.5] |
| impact_high | avellaneda_stoikov | impact_only | -68.6 | 78.8 | [-68.3, -65.4] | [-70.1, -67.2] |
| impact_high | glft | adversary | -415.6 | 457.7 | [-410.1, -395.5] | [-422.6, -408.0] |
| impact_high | glft | impact_only | -246.3 | 283.5 | [-238.0, -229.1] | [-251.5, -241.1] |
| impact_high | lstm | adversary | -147.7 | 356.2 | [-158.7, -115.5] | [-178.2, -114.8] |
| impact_high | lstm | impact_only | -76.0 | 268.6 | [-80.3, -52.4] | [-105.9, -45.8] |
| impact_high | mlp | adversary | -162.9 | 291.8 | [-173.2, -142.9] | [-181.6, -142.8] |
| impact_high | mlp | impact_only | -85.4 | 185.1 | [-89.4, -72.3] | [-99.8, -71.0] |
| impact_low | avellaneda_stoikov | adversary | -175.4 | 260.3 | [-185.2, -160.0] | [-186.3, -164.5] |
| impact_low | avellaneda_stoikov | impact_only | -94.7 | 168.0 | [-104.4, -80.1] | [-106.3, -83.4] |
| impact_low | glft | adversary | -148.3 | 178.3 | [-151.9, -142.7] | [-152.5, -144.1] |
| impact_low | glft | impact_only | -88.1 | 116.5 | [-91.4, -82.8] | [-91.9, -84.5] |
| impact_low | lstm | adversary | -148.5 | 307.8 | [-158.8, -122.9] | [-173.1, -122.2] |
| impact_low | lstm | impact_only | -86.0 | 243.4 | [-96.3, -62.0] | [-110.9, -61.2] |
| impact_low | mlp | adversary | -126.2 | 170.5 | [-130.9, -117.6] | [-132.8, -119.6] |
| impact_low | mlp | impact_only | -75.9 | 105.0 | [-78.7, -69.6] | [-80.7, -71.5] |


- At impact 0, the surviving stresses cost little: Avellaneda-Stoikov, GLFT and mlp show no
  difference against unperturbed; lstm loses (adversary, impact only) with intervals below 0.
- At impact 0.25 and 1.0, every quoting arm loses heavily under its adversary and under impact
  alone, every interval wholly below 0, and below no-trade.
- No limit breach anywhere; positions reach the 20-lot (and GLFT's 5-lot) limits.
- Margin shortfall is **not measured**: this simulator has no capital, and none was made up. The
  margin-rs maintenance requirement at the peak position is in `report.json`.

### Secondary (exploratory, not identified by the frozen rate test)

| setting | arm | condition | mean daily net | daily net against unperturbed, 90 % | rows |
|---|---|---|---|---|---|
| impact_0 | avellaneda_stoikov | adversary | -10.1 | [-12.2, -3.3] | own |
| impact_0 | avellaneda_stoikov | fixed_stress | -5.7 | [-6.6, +0.2] | own |
| impact_0 | glft | adversary | -1.3 | [-3.2, +1.0] | reused from the primary |
| impact_0 | glft | fixed_stress | +0.6 | [-1.2, +2.8] | own |
| impact_0 | lstm | adversary | -18.5 | [-13.0, -8.7] | reused from the primary |
| impact_0 | lstm | fixed_stress | -12.5 | [-6.6, -3.1] | own |
| impact_0 | mlp | adversary | -0.8 | [-4.8, +1.0] | reused from the primary |
| impact_0 | mlp | fixed_stress | -0.5 | [-3.6, +0.3] | own |
| impact_high | avellaneda_stoikov | adversary | -132.0 | [-132.6, -127.9] | own |
| impact_high | avellaneda_stoikov | fixed_stress | -61.2 | [-60.9, -58.0] | own |
| impact_high | glft | adversary | -415.6 | [-410.1, -395.5] | reused from the primary |
| impact_high | glft | fixed_stress | -161.9 | [-156.3, -141.8] | own |
| impact_high | lstm | adversary | -147.7 | [-158.7, -115.5] | reused from the primary |
| impact_high | lstm | fixed_stress | -72.1 | [-80.1, -45.7] | own |
| impact_high | mlp | adversary | -162.9 | [-173.2, -142.9] | reused from the primary |
| impact_high | mlp | fixed_stress | -44.1 | [-45.5, -34.0] | own |
| impact_low | avellaneda_stoikov | adversary | -175.4 | [-185.2, -160.0] | reused from the primary |
| impact_low | avellaneda_stoikov | fixed_stress | -67.1 | [-72.0, -57.0] | own |
| impact_low | glft | adversary | -148.3 | [-151.9, -142.7] | reused from the primary |
| impact_low | glft | fixed_stress | -56.3 | [-58.9, -51.4] | own |
| impact_low | lstm | adversary | -148.5 | [-158.8, -122.9] | reused from the primary |
| impact_low | lstm | fixed_stress | -84.4 | [-87.0, -68.0] | own |
| impact_low | mlp | adversary | -126.2 | [-130.9, -117.6] | reused from the primary |
| impact_low | mlp | fixed_stress | -18.9 | [-22.5, -11.6] | own |

What was measured: for 13 of 15 policies (3 of them no-trade) the pick over all 24 points was
the surviving primary pick. Avellaneda-Stoikov at impact 0 and 1.0 picked a removed point, which
lost more than its primary pick by 2.3 and 0.3 a day. The fixed stress lost less than the
primary picks in every row. Much of the loss comes with the impact step, which is among the
surviving points: impact alone costs about half of the adversary's loss. Rows marked
reused are the primary's own adversary days, because that policy's pick over every point was its
primary pick; they are not independent evidence.

## Part 2: #281's impact condition

**Not completed.** #281's frozen A and B at kappa = 2 kappa0, through #282's reactive runner
(merged at e88bdf6), on #281's 30 holdout days: a one-off added condition, since those base days
were read for #281's other 13 conditions, with no bearing on #281's decision.

- Attempt 2 stopped at its coded limit, 5400 s on 4 processes (the design's 6 process-hours),
  with 142 of 300 seed-days journaled. No estimand is drawn from the partial rows, and they were
  not read for results.
- The cause is the cost estimate: part 2 had no pilot, and its reactive episodes ran about 2
  process-minutes a day, some 6 times part 1's, so it needed about 10 process-hours.
- Attempt 2 was declared the last before it ran; there is no third attempt. Completing it would
  need a pilot of the reactive episodes and a budget of about 10 process-hours on its own.

## Disclosures

- Design: v1 to v7 before any policy ran. v4 replaced fixed spread by GLFT, as #107 part C
  selected only Avellaneda-Stoikov and GLFT. v5 and v6 added the secondary after `identify.json`
  (sha256 7346691a...cb62) was seen.
- Pilot: 2 days per arm on seeds outside every split, under load; it projected 47.7
  process-hours, under the 90 ceiling, so no cut. The study spent 47.7 in part 1.
- Single holdout attempt, no stop, no failure, no recovery. Freeze sha256 09102f07...1ef1
  (`frozen.sha256`); the freeze file holds local paths and stays out of the repository.
- Part 2 attempt 1 failed at start: every spawn worker raised ModuleNotFoundError because
  #281's runner was loaded under a new module name; the journal had 0 lines and no day was read.
  The code had been checked at its exact SHA without a real spawned run, which is how the fault
  was missed. Attempt 2, decided before any outcome existed, ran under a second marker recording attempt 1, with
  the import fixed and a script-level spawn test.
- Spend: 47.7 process-hours in part 1 and 6.0 in part 2, 53.7 of the 90 ceiling.
- Operating rules: the marker directory (`m5_10_markers`) is an argument and must be the same
  every time; a restart after a limit stop needs an explicit decision; a killed run's time is
  charged by hand from its logs; a second holdout_secondary journal is not refused.
- The evaluation used an older bridge virtual environment (#107's), whose `test_training`
  version check fails, as do its `test_quote_lives` tests (a stale bridge); #107's checkpoints
  were evaluated with it. `make pre-push`, which builds the current bridge, passes.

## Results in the repository

`results/` holds `report.json`, `selection.json`, `selection_secondary.json`, `budget.jsonl` and
the three markers. The holdout rows (`holdout.json`, 3.3 MB, and `holdout_secondary.json`) and the
days are kept outside the repository; their sha256 are in the PR and the report's inputs.

## Steps

From `bridges/python`, with `run.py <step>`: `pilot`, `days-selection`, `identify`, `selection`,
`selection-secondary`, `freeze`, `days-holdout`, `holdout`; then `report.py`; `part2.py` for #281.
