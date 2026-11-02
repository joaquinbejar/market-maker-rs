# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking

- The order direction is the shared `financial_types::Action`, which
  replaces both local `Side { Buy, Sell }` enums (`execution::Side` and
  `events::Side`). It is re-exported as `execution::Action`,
  `events::Action` and from the prelude. Replace `Side::Buy` and
  `Side::Sell` with `Action::Buy` and `Action::Sell`; `is_buy()` and
  `is_sell()` are unchanged, and `Side::opposite()` is removed.
- `events::OptionStyle` is the shared `financial_types::OptionStyle`, the
  same type as `optionstratlib::OptionStyle`.
- `Action` is `#[non_exhaustive]` and has an `Other` variant, so a `match`
  on it outside this crate needs a wildcard arm. This crate supports only
  `Buy` and `Sell`.
- New error variant `MMError::UnsupportedAction { action }`, returned for
  `Action::Other` before any state changes by:
  - `ExchangeConnector::submit_order` of `MockExchangeConnector` and
    `OrderBookConnector` (no order id is consumed);
  - `OrderManager::register_order`, `OrderManager::register_order_checked`
    (before the policy is asked) and `OrderManager::record_fill`;
  - the methods that became fallible, listed below.
- These methods now return `MMResult<Decimal>` instead of `Decimal`:
  - `Fill::net_value`;
  - `OrderManager::get_open_quantity` and
    `ThreadSafeOrderManager::get_open_quantity`;
  - `LimitsPolicy::live_quantity`;
  - `AccountingFill::signed_quantity` and `AccountingFill::cash_flow`.
- Methods that keep their signature handle `Action::Other` as follows:
  - `PretradePolicy::check` of `LimitsPolicy` and `MarginRiskPolicy`
    returns `PretradeDecision::Rejected`, before any state changes;
  - `LimitsPolicy::on_order_event` ignores a fill with that side (its
    execution id is not consumed); `MarginRiskPolicy` records it in
    `MarginAccount::errors` and books nothing;
  - `MarginLedgerAccountant::post_fill` returns the error and books
    nothing;
  - every built-in `FillModel` returns `FillResult::NoFill` (the
    probabilistic model does not advance its generator), and
    `ProbabilisticFillModel::calculate_probability` returns zero.
- Serialization:
  - the JSON of `MarketMakerEvent` is unchanged: sides are still
    `"buy"`/`"sell"` and option styles `"call"`/`"put"`, and other
    spellings are still rejected. Serializing an event whose side is
    `Action::Other` fails;
  - serializing `Action` or `OptionStyle` directly, outside an event, now
    gives `"Buy"`/`"Sell"` and `"Call"`/`"Put"`. This matches the former
    `execution::Side`, but not the former `events::Side` and
    `events::OptionStyle`, which serialized in lowercase.
- The crate declares `rust-version = "1.95"`. This is not a new
  requirement of this change: the dependency graph already needed Rust
  1.95 before it, because `orderbook-rs` 0.15.1 (a required dependency
  that declares no `rust-version`) uses atomic `try_update`, stable since
  1.95. `financial_types` itself needs only 1.88.

### Added

- Dependency `financial_types = "0.3"`. It derives `serde` unconditionally,
  so `serde` is now compiled in the default build too; the `serde` feature
  still gates this crate's own serialization support.
- `make msrv` checks the crate with the declared minimum Rust version, with
  no features and with all features.
