# Decision frequency and policy memory (#103)

Synthetic: one queue-reactive simulator with made-up parameters, the regimes of #102 sampled
every second. `design.json` fixes the data, splits, execution, arms, seeds, decision rule and
compute before any run, and records every later change in `changes`. `digests.json` holds the
dataset and external digest of every day, from

    cargo run --release --example qr_days -- experiments/m4_06/design.json <data dir>

The holdout days are written apart and are not read until every checkpoint and configuration is
frozen.

From `bridges/python`, in order:

    python experiments/m4_06/run.py <step> <data dir> <out dir> <commit> [<frozen sha256>]

with the steps `fee_rule`, `classical_selection`, `training`, `freeze` and `holdout`, then

    python experiments/m4_06/report.py <out dir> target/release/examples/m4_05_intervals

`report.py` fixes how the holdout is read (days, paired intervals and the decision rule of
`design.json`) and was committed before it ran on the holdout.

`validation_pick.py` was added after the freeze and before any holdout read, outside the frozen
design, which selects no learned arm. It picks the learned arm at one row on the validation
days by #103's rule, for the studies that build on #103, and checks every frozen hash first.

The comparisons are many (two fees, three regimes and pooled, four kinds), and each is a 90 %
interval on five or fifteen days with no correction for their number; a regime's five days
bound how much any one of them can show.
