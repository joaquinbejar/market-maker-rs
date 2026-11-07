# Decision frequency and policy memory (#103)

Synthetic: one queue-reactive simulator with made-up parameters, the regimes of #102 sampled
every second. `design.json` fixes the data, splits, execution, arms, seeds, decision rule and
compute before any run, and records every later change in `changes`. `digests.json` holds the
dataset and external digest of every day, from

    cargo run --release --example qr_days -- experiments/m4_06/design.json <data dir>

The holdout days are written apart and are not read until every checkpoint and configuration is
frozen.
