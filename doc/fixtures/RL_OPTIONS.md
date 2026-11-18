# RL options spaces fixtures

`rl_options.json` lets an environment outside this crate check its use of the options
quoting spaces (`rl::OptionsProfile`) and of the net delta (`rl::net_delta`,
`rl::black_scholes_delta`). The crate embeds it as `market_maker_rs::rl::RL_OPTIONS_FIXTURES`.
It is generated and checked by `tests/rl_options.rs`; CI checks the same file on `x86_64` and
`aarch64`, so every delta is bit identical on both. Rewrite it with
`MM_WRITE_RL_OPTIONS_FIXTURES=1 cargo test --features rl --test rl_options` only after a
deliberate change of `OPTIONS_OBSERVATION_SPACE_VERSION`, `OPTIONS_ACTION_SPACE_VERSION` or
`NET_DELTA_VERSION`, then run the test again without the variable to check the embedded copy.

- `format`: the version of this file's layout (1).
- `profiles`: each profile as plain data with the hex of `OptionsProfile::to_bytes`, its
  `SpecId` (fingerprint as a decimal string), the observation length and the action head
  sizes. The scales are those of `tests/rl_options.rs`; they enter the bytes.
- `slots`: a listing of `[expiry symbol, expiry ns, strike, style]` and, per index mark, the
  series each option slot names (`null` for an empty slot) as
  `OptionsProfile::resolve_slots` gives it. It covers two expiries at the same instant
  (ordered by symbol), an unlisted put, a strike tie (to the lower strike) and no index mark.
- `observation`: one decision of the `small` profile: raw values (`null` with no value) and
  normalized `f32` values in observation order (slots, hedge lane, agent).
- `actions`: head rows of the `small` profile with the decoded kind, lane and canonical row,
  or `invalid_action` when `OptionsProfile::decode_action` refuses the row.
- `deltas`: `[S, K, tau, sigma, style, delta]` rows; tau in years, delta rounded half to even
  to `1e-9`. Rows come in call and put pairs of the same inputs.
- `net_delta`: linear positions `[lots, lot size]`, option positions
  `[lots, base per lot, strike, style, expiry ns]`, the net delta and the hedge cap in lots.

Decimals are strings. Each `f32` is a string in its shortest round trip form.
