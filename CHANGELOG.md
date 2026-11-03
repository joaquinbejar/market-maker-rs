# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `examples/counterpart.rs` (feature `options`): a batch counterpart for an
  external harness. `calibrate` builds a surface from a quotes CSV with
  `calibrate_surface` and writes it as JSON (status, reasons, slices in
  exact round-trip text, canonical `params`, the report); `price` values a
  requests CSV on that surface under sticky moneyness or sticky strike and
  writes one response row per request (`ok` with price, total Greeks and
  the total variance read, or `error` with a class, `InvalidInput`,
  `OutOfDomain`, `Unsupported` or `Numerical`, and no number); `identity`
  prints the protocol and the build commit. Decimals are read exactly or
  refused. `price --vol-shift X` prices on the surface with its implied
  volatility moved by `X` (a parallel `ShockedSurface`), for finite
  differences in volatility. Its unit tests run with `cargo test`.
- `options::pricing::surface_point(&OptionInputs, Dynamics)`: the `(k, tau)`
  at which `value` reads the surface, computed by the same code (#95).
- Order by order book reconstruction under `l3` (new feature `l3`, no
  dependencies, #77): `BookReconstructor` rebuilds the visible book of one
  product from an `L3Snapshot` and its sequenced `L3Message`s, modelled on
  the Coinbase Exchange `full` channel, with the snapshot handshake. It fails
  closed and transactionally: a gap, an out of order sequence or any
  `Discrepancy` (unknown maker, missing match, size increase or price
  change, crossed book, capacity, checked arithmetic) ends the segment, and
  views error until the next validated snapshot. A `QualityManifest`
  records segments and counters; `priority_profile` the volume of unknown
  priority per level; `volume_ahead` the size ahead of an own or
  hypothetical order under declared `Assumption`s. The
  `l3_reconstruction` example replays a synthetic stream with injected
  faults. Replaying captured segments waits for #76.

- Surface calibration under `options::surface` (feature `options`, #96):
  quote cleaning with a reason for every rejected quote and a global error
  for future data (`clean_quotes`); implied total variance by safeguarded
  Newton on the undiscounted Black price (`implied_total_variance`,
  `extract`); the quasi-explicit SVI slice fit with typed failures and
  reported active constraints (`fit_slice`); `calibrate_surface` with a
  `QualityReport` (residuals by strike and expiry, diagnosis on every node,
  stability against the previous surface aged to the same instant) and a
  Healthy, Degraded or Failed status; a causal, append only `SurfaceStore`
  that ages surfaces at constant volatility per `k` and serves them under a
  `UsePolicy`; holdout validation against a baseline linear in `w`
  (`validate_holdout`) and a causal `replay`; and the
  `vol_surface_validation` example. Wiring the surface into the backtest
  engine waits for option instruments (#97).
- `options::pricing` (feature `options`): European option valuation on a
  `VolSurface`, `value(&OptionInputs, &dyn VolSurface, Dynamics)` with
  `Dynamics::{StickyMoneyness, StickyStrike { s_ref }}`, returning an
  `OptionValuation` (price, total delta and gamma, vega per 1.00 of a
  parallel implied volatility shift, theta per calendar year), the payoff
  exactly at expiry, `OptionPricer` and `SurfacePricer`, `PRICING_YEAR_DAYS`
  (365) and `tau_from_days`, and `black_undiscounted` / `black_w` for
  implied volatility extraction (on every finite positive `F`, `K`; a
  positive price below the smallest `f64` is `MMError::PriceUnderflow`,
  never 0). The normal distribution is a port of
  fdlibm's `erfc` (#95).
- `MMError::PriceUnderflow` (discriminant 10): a price that is positive in
  exact arithmetic but below the smallest positive `f64`, returned by
  `black_undiscounted` instead of a `NumericalError` with that message. Its
  `Display` is the former text ("numerical error: Black price is positive
  but below the smallest positive f64"), `is_numerical_error()` is true for
  it, and with `serde` it serializes as `{"type":"PriceUnderflow"}`, with no
  `details`.
- `options::adapter::ADAPTER_MS_PER_YEAR`: the 365.25-day year of the
  adapter's ms helpers, now named; their results are unchanged (#95).
- The financial environment kernel: `backtest::kernel` (feature `kernel`,
  no dependencies) with `FinancialKernel`,
  a deterministic, resumable re-implementation of the backtest engine loop
  over a fixed width, `Copy` `KernelState`, with `reset`, `step`,
  `observe` and `mask`, a canonical state encoding (`encode`; `decode`,
  which accepts only states satisfying every invariant `step` relies on;
  `fingerprint`), validated `KernelDataset`s, event sinks
  and a `QuoteAdapter` for Avellaneda-Stoikov and GLFT quotes (bound to one
  kernel; GLFT at an inventory bound posts a one-sided action). An action
  whose transaction fails is rolled back and reported as
  `Rejected(Terminal)` with the halt in the step status. Inside its
  subset (immediate fills, quote fees, optional `LimitsPolicy` limits,
  latency with jitter, ordered data) it reproduces engine runs exactly;
  `new` rejects configurations outside the exactness bounds and capacities
  below the dataset's needs. Examples `kernel_avellaneda` and `kernel_glft`
  (#83).
- `options::surface` (feature `options`): the `VolSurface` trait (total
  implied variance `w(k, tau)` and its derivatives at a log forward
  moneyness and a time to expiry), the validated `TotalVariance` value,
  `check_point` and a `FlatSurface`. Model arithmetic there is `f64`, an
  explicit exception documented in the module (#95, #96).

### Changed

- With the `margin` feature, a fill whose base fee breaks the fee
  declaration of the gate's limits is recorded in `MarginAccount::errors`
  as a fee breach (the gate then rejects new orders on the instrument),
  and the message for a fill that is not clean includes its shortfalls.
  The `backtest::margin` docs give the declaration that matches
  `FeeCurrency::Base`.
- `BacktestEngine::faults` keeps only the first `MAX_RECORDED_FAULTS` (16)
  errors; `BacktestEngine::fault_count` reports the total. Admission still
  stops at the first fault (#133).

### Fixed

- `ProbabilisticFillModel` draws its uniform number from the top 32 bits
  of the generator state scaled by 2^32, so it covers [0, 1). The draw used
  to stay below 0.5, so a fill with probability p happened with
  probability min(1, 2p). Backtests that use this model now fill less
  often, and their output for a given seed changes (#143).
- `OrderBookConnector::cancel_all_orders` no longer clears the whole order
  mapping when some cancels fail: an order whose cancel failed while it
  still rests stays tracked, and the first failure is returned after every
  order was attempted. `MockExchangeConnector::cancel_all_orders` returns
  the first cancel error, `LockPoisoned` included, instead of dropping it.
  Calling again retries the orders that failed (#146).

### Breaking

- The `margin` feature depends on margin-rs 0.1 from crates.io instead of
  a git revision. Callers that build the `RiskGate` or `Ledger` passed to
  `MarginAccount` or `MarginLedgerAccountant` use margin-rs 0.1, whose
  order and execution direction is `financial_types::Action`, the type
  market-maker-rs already uses.
- `GLFTStrategy` computes the published model of Guéant, Lehalle and
  Fernandez-Tapia (arXiv:1105.3115) instead of an Avellaneda-Stoikov variant
  (#134). `GLFTConfig::new` takes `(risk_aversion, order_intensity_k,
  intensity_scale_a, max_inventory_lots, lot_size, terminal_time,
  min_spread)`: the fill intensity scale `A` per second, the inventory bound
  `Q` in lots (`1..=MAX_INVENTORY_LOTS`, 200) and the lot size are new, and
  `order_intensity` is renamed `order_intensity_k`. New `GLFTConfig::validate`.
  `GLFTStrategy::calculate_optimal_quotes` returns `MMResult<GlftQuotes>`
  (`bid` and `ask` are `Option<Decimal>`: no bid at `+Q`, no ask at `-Q`),
  requires the inventory to be a whole number of lots within `±Q` (otherwise
  `InvalidPositionUpdate`), returns `NumericalError` on any non finite or
  non convertible intermediate, and widens a spread below `min_spread`
  symmetrically. New `GLFTStrategy::closed_form_quotes` (Proposition 3).
  `GLFTStrategy::compare_with_avellaneda_stoikov` returns
  `(GlftQuotes, (Decimal, Decimal))`, its A-S side in lots with
  `γ · lot_size`. A quote needing more than 11 000 steps of the matrix
  exponential (about 50 ms at `Q = 200`) returns `NumericalError`. Removed: `PenaltyFunction`,
  `GLFTConfig::{terminal_penalty, dynamic_gamma, gamma_scaling_factor,
  penalty_function, with_dynamic_gamma, with_penalty_function}` and
  `GLFTStrategy::{calculate_reservation_price, calculate_optimal_spread,
  calculate_dynamic_gamma}`. The prelude exports `GlftQuotes` instead of
  `PenaltyFunction`. With `serde`, `GLFTConfig` denies unknown fields.
- Chain quotes come from the option pricer instead of a placeholder
  (intrinsic value plus `spot / 50`, which ignored volatility and expiry):
  `ChainMarketMaker::refresh_all_quotes(underlying_price, &ChainPricing)`
  returns one `ChainQuoteOutcome` per contract, `Quote(ChainQuoteUpdate)`
  or `NotQuoted { strike, style, reason }` (for example a theoretical value
  below one price unit), and fails as a whole only when the shared inputs
  do (`OptionPricer::check_market`: spot, time to expiry, rates and their
  derived factors; the surface is never evaluated there, so a surface error
  at a contract's own `k` is that contract's `NotQuoted`). `ChainPricing` carries the pricer, the time to expiry and the rates of
  the chain's expiration. `generate_quote` takes the `OptionValuation`
  (its theo is the model price rounded half to even to the price unit) and
  returns `MMResult`; `ChainQuoteUpdate` has a `valuation` field and
  `ChainQuoteUpdate::new` takes it and returns `MMResult` (zero theo, a
  crossed quote or an overflowing spread are errors);
  `ChainQuoteUpdate::spread` returns `Option<u64>` (`None` when crossed);
  `get_spread_multiplier` and `calculate_quote_size` return `MMResult`. The
  quote path uses checked arithmetic (#95).
- `MMError::UnsupportedContract { reason }`: a contract the model cannot
  value (for example a non-European option given to the
  Black-Scholes-Merton pricer) (#95).
- No panics in library code (#135). New `MMError::LockPoisoned`; these now
  return `MMResult` instead of panicking on a poisoned lock:
  `ThreadSafeOrderManager::{read, write, get_stats, open_order_count,
  check_timeouts, cleanup}`, `CollectingAlertHandler::{alerts, count, clear}`,
  `MockExchangeConnector::{set_balance, open_order_count, all_order_ids}` and
  `OrderBookConnector::set_balance`. `AlertHandler::handle` returns
  `MMResult<()>`; `AlertManager::{alert, alert_with_message}` deliver to every
  handler and return the first handler error. `GridConfig::new` rejects
  `levels_per_side > i32::MAX`; `AccountingFill::cash_flow` and
  `LimitsPolicy::live_quantity` return `NumericalError` on overflow; position,
  PnL, notional and slippage values saturate at the `Decimal` bounds.
  `OrderBookConnector::{with_config, from_orderbook}` return `MMResult<Self>`
  and reject a price or quantity precision above the new `MAX_PRECISION`
  (19), also checked by the new `OrderBookConnectorConfig::validate`; orders
  with a non positive price, a negative quantity or an out of range value
  return `NumericalError` instead of being saturated. `LiveMetrics::{
  update_position, update_pnl, update_realized_pnl, update_unrealized_pnl,
  add_realized_pnl, reset}` return `MMResult<()>`, `LiveMetrics::{
  get_position, get_realized_pnl, get_unrealized_pnl}` return
  `MMResult<Decimal>`, `LiveMetrics::snapshot` returns
  `MMResult<MetricsSnapshot>` and `MetricsBridge::sync` returns
  `MMResult<()>`: a poisoned lock is `LockPoisoned` and a PnL overflow is
  `NumericalError`. A poisoned `MarginAccount` rejects orders and returns
  `LockPoisoned` from event and fill booking until reset, and `fault()`
  reports it. `VolatilityEstimator::{calculate_simple, calculate_ewma}`
  need at least 3 prices; `RiskAversionCalibrator::calibrate_from_history`
  rejects an empty history. `RiskLimits::check_order` and
  `ManagedOrder::record_fill` return `NumericalError` on overflow;
  `MarketImpactFillModel` fills return `NumericalError` when the impact
  overflows. With `max_position` not positive, `RiskLimits::scale_order_size`
  returns zero and `position_utilization` returns `Decimal::MAX`.
  `AlertManager` deduplicates per handler: a handler that delivered an
  alert type inside the window is skipped and one that failed is retried.
  `OrderBookConnector` accepts every raw price that converts exactly to a
  `Decimal` and rejects one that would be rounded, and holds the order mapping lock across book updates.

- Backtest fees can be charged in the base asset or a third asset:
  `BacktestConfig::fee_currency` (`FeeCurrency::{Quote, Base, Other}`,
  default `Quote`, unchanged behaviour). `FeeAsset` moved to
  `types::fee` (still re-exported from `backtest`), gained
  `Other(String)` and is no longer `Copy`. `OrderEvent::Fill` has a new
  `fee_asset` field. `MarginLedgerAccountant` and the margin policy book
  base fees on the base asset instead of refusing them. `FeeAsset` is
  `#[non_exhaustive]`. A base fee reduces the base held: the engine,
  strategy and `LimitsPolicy` positions are net of base fees, which are
  marked at the current mid in `total_fees`. `SimulatedFill` gains
  `base_fee` and `base_delta()`, and `AccountingSnapshot` gains
  `base_fees`, whose `fees` now covers quote fees only (#127).
- `BacktestResult` has a new public field `accounting: AccountingStatus`
  (accountant description and the execution ids it refused), and
  `SimulatedFill` has a new public field `execution_id: Option<u64>`
  (`None` from its constructors, set by the engine); struct literals of
  either type need the new field (#126).
- The order direction is the shared `financial_types::Action`, which
  replaces both local `Side { Buy, Sell }` enums (`execution::Side` and
  `events::Side`). It is re-exported as `execution::Action`,
  `events::Action` and from the prelude. Replace `Side::Buy` and
  `Side::Sell` with `Action::Buy` and `Action::Sell`; `is_buy()`,
  `is_sell()` and `opposite()` keep working (`Action::opposite()` maps
  `Other` to itself).
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
  - `SimOrder::validate`, `SimOrder::accept` and `SimOrder::apply_fill`
    (`SimOrder::reject` and `SimOrder::cancel` accept any side, to take
    such an order out);
  - `MarginLedgerAccountant::post_fill`;
  - the methods whose signature changed, listed below.
- Signature changes:
  - now `MMResult<Decimal>` instead of `Decimal`: `Fill::net_value`,
    `OrderManager::get_open_quantity`,
    `ThreadSafeOrderManager::get_open_quantity`,
    `LimitsPolicy::live_quantity`, `AccountingFill::signed_quantity`,
    `AccountingFill::cash_flow` and
    `ProbabilisticFillModel::calculate_probability`;
  - `ManagedOrder::record_fill` returns `MMResult<()>` (it checks the
    fill's side and the order's own side);
  - `FillModel::simulate_fill` returns `MMResult<FillResult>`; the built-in
    models return `UnsupportedAction` for an unsupported side without
    touching their state (the probabilistic model does not advance its
    generator);
  - `PretradePolicy::on_order_event` returns `MMResult<()>` (the default
    implementation returns `Ok(())`).
- Pretrade policies and `Action::Other`:
  - `PretradePolicy::check` of `LimitsPolicy` and `MarginRiskPolicy`
    returns `PretradeDecision::Rejected`, before any state changes;
  - a fill with that side is unknown exposure: `on_order_event` returns
    `UnsupportedAction`, books nothing and does not consume the execution
    id, and latches a fault (`LimitsPolicy::fault`,
    `MarginAccount::fault`). While the fault is set, every `check` is
    rejected; `reset` clears it.
- `BacktestEngine` records any error of the fill model or the pretrade
  policy in `BacktestEngine::faults` and, from the first one, refuses every
  new quote (reported as `Rejected`, without asking the policy) until
  `reset`. Resting orders keep working.
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

- `options::surface` (feature `options`): raw SVI slices and surfaces
  (`SviSlice`, `SviSurface`), linear in total variance between expiries,
  with a declared extrapolation policy (none beyond the last expiry, an
  opt-in constant volatility below the first, SVI wings in `k` marked
  `Region::Extrapolated`); static diagnostics on a declared grid
  (`diagnose`: butterfly `g(k)`, calendar slope and gaps); and named shock
  scenarios (`SurfaceShock`, `ShockedSurface`, `run_scenarios`), each with
  its own diagnostics. The model is evaluated in `f64`, an explicit
  exception for option models: every value is checked finite (#96).
- `BacktestEngine::faults`, `LimitsPolicy::fault` and
  `MarginAccount::fault`.
- Dependency `financial_types = "0.3"`. It derives `serde` unconditionally,
  so `serde` is now compiled in the default build too; the `serde` feature
  still gates this crate's own serialization support.
- `make msrv` checks the crate with the declared minimum Rust version, with
  no features and with all features.
