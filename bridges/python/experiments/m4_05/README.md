# The out-of-sample comparison of #102

Everything here is synthetic: a queue-reactive simulator with made-up parameters, and no
market data. One synthetic simulator, made-up parameters and 24 days per regime bound what the
comparison can show.

- `design.json` fixes, before any training: the regimes and their parameters, the days and
  their seeds, the split into train, validation and holdout, the execution, the arms and their
  grids, the training seeds, the budgets, the bootstrap and the decision rule.
- `digests.json` holds, for every regime and day, the seed, the dataset digest the environment
  reports for its rows, and the external digest of the parameters, the seed and the reference.
  The holdout days are in it, so that their data is fixed before any run.

The rows are not in the repository. From `bridges/python`:

```
cargo run --release --example qr_days -- experiments/m4_05/design.json <data dir> > digests.json
```

writes them under `<data dir>/<split>/<regime>/day-<n>.json` and prints the digests, which
must equal `digests.json`. No run reads `<data dir>/holdout` until every checkpoint and
configuration is frozen and its hash recorded.
