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

How the report reads the classical comparator: per fee and scope, it compares each learned arm
with the better of Avellaneda-Stoikov and GLFT by their mean daily net on the holdout days of
that scope. That is the stronger of the two after the fact, so it is the harder comparator for a
learned arm, and the comparison is reported beside the decision rule, not in it. The points each
classical arm runs were chosen on validation (`classical_selection.json`); only the choice
between the two arms uses the holdout.

Whether the queue binds. The design's check before training (`changes`, version 2, validation
day 15, risk aversion 0.01 and intensity 50, fee 0.0002) compared the queue with immediate
fill: deciding every row, 22 fills and a net of -0.020169 against 64 fills and -0.31108; every 5
rows 736 and -19.103243 against 738 and -19.102291; every 10 rows 1399 and -38.680769 against
1399 and -38.650775. On a training day, `queue_check.py` runs the fee rule's point (risk
aversion 0.001, intensity 50, the calm volatility, fee -0.0001) every row on calm day 1 under
both: the queue 64 fills and a net of 2.2113975, immediate fill 255 fills and 4.0693375, with
172798 orders accepted under each. The queue binds deciding every row and barely at 5 or 10
rows. `queue_check.py` was added after the holdout was read, for the review; it reads one
training day and changes no step's output:

    python experiments/m4_06/queue_check.py <data dir> <out dir> [<regime> <day>]

The comparisons are many (two fees, three regimes and pooled, four kinds), and each is a 90 %
interval on five or fifteen days with no correction for their number; a regime's five days
bound how much any one of them can show.
