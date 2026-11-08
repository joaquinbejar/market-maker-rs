# Quote duration and latency aware learning (#110)

Synthetic: #103's queue-reactive regimes (calm, stressed, thin) sampled every 100 ms, for a
2 hour session per day, with new data and its own digests (`digests.json`). One simulator,
made-up parameters and 24 days per regime bound what this can show.

`design.json` fixes the data, splits, execution, arms, seeds, decision rule and compute before
any run, and records every later change in `changes` with what had been seen:

- **Version 1:** the design and decision rule, before any run.
- **Version 2:** 100 ms rows, so that latency below a second can be measured, and the queue rate (1 lot per 750 ms) that fills and binds at every latency.
- **Version 3:** the runner's choices the design had left open (`runner_choices`): the predictor, the teacher's projection of the classical quote onto the Box and its cloning, and the arm without a duration.
- **Version 4:** the fee rule and the teacher decide every 10 rows. At version 3 every fee rule point, deciding every row, had no fill on any training day: a quote replaced every 100 ms row clears no lot. Two checks in code refuse a teacher that does not fill. The version 3 outputs are kept apart with their hashes.

## Arms

- `full`: PPO choosing its quote and how many rows it rests (`ChosenDuration`, choices 1, 5, 10 and 30), with a linear forecast of the mid change as one more feature, warm started by cloning the Avellaneda-Stoikov point the fee rule chose.
- `no_teacher`, `no_predictor`: `full` without the warm start or without the forecast.
- `no_duration`: `full` deciding every row with the plain Box. Under this queue it fills only by holding its quote for at least 8 rows, which it must learn without a duration action.
- Comparators: no-trade, and Avellaneda-Stoikov selected on validation per regime, latency and interval (1 and 10 rows).
- Each learned arm is trained at each latency (25, 100 and 500 ms submit and cancel delays) with 5 seeds: 60 runs of 2M decisions.

## Running it

From `bridges/python`:

    python experiments/m4_10/run.py <step> <data dir> <out dir> <commit> - [<processes>]

with the steps `fee_rule`, `predictor_fit`, `classical_selection`, `teacher`, `training` and `freeze`. Then, once:

    python experiments/m4_10/holdout_once.py <data dir> <out dir> <commit> <frozen sha256> [<processes>]
    python experiments/m4_10/report.py <out dir> target/release/examples/m4_05_intervals

`holdout_once.py` was added after the freeze, leaving `run.py` as frozen. It creates `holdout.started` exclusively before any job and keeps it if the holdout fails, so a failed or concurrent attempt is never run again by script. `report.py` fixes how the holdout is read and was committed (f953f2c) before the holdout ran.

## The run

All runs used v4, at cf930bd, on 3 processes.

- **Setup steps:**
  - fee rule: 23 min;
  - predictor fit: under 1 min;
  - classical selection: 19 min;
  - teacher: 11 min.
- **Fee rule:** no fee was profitable on the training days, so the rule chose 0.0002, the secondary fee. The chosen point (risk aversion 0.01, intensity 50) filled 2405 times on them.
- **Teacher:** each of the 45 rollouts filled, between 556 and 2267 times. The projection log, the first run as an example:
  - 100000 quotes, with no holds and no one-sided quote;
  - no side inside the mid, and 875 sides beyond the grid;
  - 169118 of 200000 sides snapped off their exact distance, by 207656 ticks in all, about 1 tick per side.
- **Training:** 60 runs, each of 2M decisions (2,000,896 PPO timesteps), in 142 minutes, none stopped. The design's estimates are for 6 processes and the run used 3. As agreed before training reached any limit, the overrun rule was read against the estimate scaled to 3 processes: 480 minutes, so a stop at 720. `design.json` was not edited, since its hash is frozen. The runs took 6 to 9 minutes each, faster than the pilot's 22 to 30 minutes per 2M; the times are reported as measured.
- **Freeze:** `frozen.json` sha256 57fc4687e66053bfc50e2116c652ee18751d04271e31e27ee608612346a30455, 60 runs. It records the hashes of the design, the digests, this runner, #103's runner whose helpers it takes, every report, every checkpoint and every teacher sample.
- **Holdout:** read once through `holdout_once.py` at f953f2c: 1305 jobs in 62 minutes. That is 1080 at the fee the rule chose (the secondary fee, so one fee row), plus the 225 descriptive jobs under immediate fill.
- **Hashes:** `holdout.json` sha256 d84715e4a151f575776728b185b9a69b872c97914a3194884a08c56ad562ebb3; `report.json` sha256 25a9aae579a3599cecb51d2f26d8ad0879674b7c76509134e44c1c638ca54b6f (84 decisions).

Found in review after the read:
- The freeze hashes this runner and #103's, not the bridge's `wrappers.py` and `envs.py` (`ChosenDuration`, `DecisionInterval`). It did not bite: the whole bridge, Python and Rust, is byte-identical between the training run (cf930bd) and the holdout read (f953f2c). The merge of main after the read changed both files (#303's gate counts).
- `holdout_once.py` creates its marker before `run.holdout` checks the hashes, so a mismatch would lock the holdout with nothing read. That is the conservative failure, and it did not happen; any recovery would be by hand and declared.

The results are withheld here on purpose, as #103's and #105's are: another study's holdout is pending. They come from `report.json`, whose hash the pull request commits to.

## Bounds

- **Multiple comparisons:** three latencies, three regimes and pooled, three kinds. Each is a 90 % interval on 5 or 15 days, with no correction for their number.
- **What is compared:** a decision interval chosen with each action, not a quote that expires by itself in the kernel (#297). No batch auction result is compared (#289).
- **Teacher:** a two-sided surrogate of the classical point, so its missing sides are made live.
- **Predictor:** fitted under the hold policy, so its position and order features are not identified apart from the intercept.
