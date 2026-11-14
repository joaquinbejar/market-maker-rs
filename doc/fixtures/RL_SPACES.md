# RL spaces fixtures

`rl_spaces.json` lets an environment outside this crate check its use of the `rl`
spaces. The crate embeds it as `market_maker_rs::rl::RL_SPACES_FIXTURES`. It is generated
by `tests/rl_external.rs`, which also checks it on every test run; rewrite it with
`MM_WRITE_RL_FIXTURES=1 cargo test --features rl --test rl_external` only after a
deliberate change of a space version or of `ACTION_HEAD_LAYOUT_VERSION`. That run skips the
comparison with the embedded constant, which still holds the old file until the crate is
rebuilt; run the test once more without the variable to check the embedded copy.

- `format`: the version of this file's layout (2).
- `head_layout_version`: `ACTION_HEAD_LAYOUT_VERSION`, the head order and kind codes the
  `actions` entries use.
- `profiles`: each declared profile as plain data (its `step_ns` is `null` when unset),
  with the hex of `RlSpec::to_bytes`, its `SpecId` (fingerprint as a decimal string) and
  `RlSpec::action_head_sizes`.
- `observations`: plain inputs of `observation_features` for one instrument of a profile
  (`instrument`, the observation, the previous row and that instrument's tick size) with
  the expected raw values and normalized `f32` values. With an instrument index head the
  policy's observation is these vectors concatenated in instrument order
  (`RlSpec::profile_observation`).
- `actions`: rows of head values with the action `RlSpec::decode_action` gives and its
  canonical row from `RlSpec::encode_action`, or `invalid_action` when it refuses the row:
  a kind not offered on that instrument (`RlSpec::offered_kinds`) or a head outside its
  grid. The `two_quoted` rows include a quote and both one sided orders on instrument 1.

Decimals are strings. Each `f32` is a string in its shortest round trip form.

## Observation inputs left out

An observation holds more than the features read. These fields are omitted because no
feature of the space depends on them, so any value gives the same vector:
`delivered_fills`, `delivered_local_rejects`, `status` and `rejected`. The lot size is not
an input either: book sizes and orders are already in lots, and the position in base units.
The hedge lane features of `RlEnv::new_hedged` are not part of a declared profile, and a book
depth past 1 is refused until the space has features for those levels, so no input describes
them.
