# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `strategy::signal_quote`: causal, bounded quote adjustment from imbalance,
  micro-price and VPIN features, with a declared policy for missing or stale
  values, composed with the Avellaneda-Stoikov quotes under the risk limits,
  and a simulation that ablates each feature on the same data, costs and fill
  rules, with markout, inventory and attributed PnL (#89).
- `analytics::micro_price`: Stoikov's micro-price (manuscript dated 18 April
  2018) from discretized imbalance and spread transitions, with typed
  fallbacks to the mid, a restorable calibrator, timestamped signals and an
  out-of-sample comparison of mid, weighted mid and micro-price at horizons
  fixed by the caller (#88).
- `strategy::walk_forward`: chronological splits, causal calibration of
  absolute volatility per square root of year, fill intensity `A` and `k`,
  and intraday seasonality, in `CalibrationRecord`s that carry the training
  interval, resolution and a hash of the training data, plus a walk-forward
  report against constant parameters (#87).
- Contract specifications under `instruments` (new feature `instruments`,
  off by default): `financial_types::instrument` re-exported as
  `instruments`. With `margin` as well, `instruments::margin::margin_instrument`
  converts an `InstrumentSpec` into a `margin-rs` `Instrument` with explicit
  `MarginScales`: exact field mapping, scales at least the digits of the
  ticks and the lot (`AdapterError::ScaleBelowGrid`), options
  `AdapterError::Unsupported`, constructor refusals `AdapterError::Margin`.
  The `financial_types` floor is now 0.3.1 (#93).
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

- `make test`, so `make pre-push` and the Run Tests workflow, also runs
  `cargo test --all-features`: the tests behind a feature such as `chain` or
  `options` did not run in it before (#183).
- `instruments::margin::margin_instrument` delegates to
  `margin_rs::Instrument::from_spec`, so the mapping has one implementation
  (#176). Its results and errors are unchanged. `margin` and `instruments`
  together now need `margin-rs` 0.1.1 or later: the dependency stays `"0.1"`,
  and Cargo refuses 0.1.0 when it resolves. A lockfile at 0.1.0 is not upgraded
  by itself: run `cargo update -p margin-rs`. `margin` alone still accepts
  0.1.0.
- `AdapterError`'s `From<MarginError>` is now a manual impl: the two new
  `margin-rs` errors `UnsupportedSpec` and `ScaleBelowGrid` convert to
  `AdapterError::Unsupported` and `AdapterError::ScaleBelowGrid`, not to
  `AdapterError::Margin`. Every other error still converts to
  `AdapterError::Margin`.
- `MarketTick::spread_bps` and `MarketSnapshot::spread_bps` return the bound
  with the spread's sign when the ratio overflows, and
  `ManagedOrder::fill_ratio` saturates at the bound on such an overfill. Both
  returned 0. Fill rate calibration propagates a failed logarithm instead of
  regressing on 0.
- `OrderFlowStats::imbalance`, `VolumeBucket` imbalances and VPIN keep their
  proportions when a side's volume is beyond the `Decimal` bounds. There,
  `net_flow` carries rounding error, so its sign is reliable only when the
  true net exceeds that error. `buy_vwap` and `sell_vwap` are `None` when their
  inputs overflow, and `volume_ratio` is `None` when a volume sits at the
  bound.
- `tests/contract_fixtures.rs`: the inverse future fixture uses the Deribit
  contract value, 10 USD (was 100), on 100 contracts, and the linear one the
  Binance terms (multiplier 1, lot step 0.001); both are cross-checked
  against `financial_types` formulas in separate tests (#128).
- With the `margin` feature, a fill whose base fee breaks the fee
  declaration of the gate's limits is recorded in `MarginAccount::errors`
  as a fee breach (the gate then rejects new orders on the instrument),
  and the message for a fill that is not clean includes its shortfalls.
  The `backtest::margin` docs give the declaration that matches
  `FeeCurrency::Base`.
- `BacktestEngine::faults` keeps only the first `MAX_RECORDED_FAULTS` (16)
  errors; `BacktestEngine::fault_count` reports the total. Admission still
  stops at the first fault (#133).
- CI installs Rust with `dtolnay/rust-toolchain`, pinned by commit, in
  place of the archived `actions-rs/toolchain`; toolchains, components and
  matrices are unchanged. `make lint` also runs clippy with
  `--no-default-features` (#155).
- The crate denies `clippy::unwrap_used`, `expect_used`,
  `indexing_slicing`, `panic` and `arithmetic_side_effects` outside tests
  (#145), with no file level exemption.
- Arithmetic is checked (#135, #144, #166): a function
  that returns `MMResult` reports an overflow or a zero divisor as
  `MMError::NumericalError`; derived display and report values (mid
  prices, spreads, notionals, imbalances, analytics statistics, backtest
  metrics, persisted record totals, multi underlying aggregates) saturate
  at the `Decimal` bounds and say so in their docs; an overflowing ratio
  that returns `Option` is `None`. Several former panics on overflow or a
  zero divisor (grid inventory scaling with a zero `max_position`, the
  weighted book imbalance at a zero mid, every Decimal `sum`, which
  panics on overflow, and the order intensity square root of the smallest
  positive Decimal) are now errors or saturating values.
  `OrderManager::get_open_quantity` reports an overflowing total as
  `MMError::NumericalError`.
- `OrderBookSnapshot::mid_price`, `spread` and `spread_bps` (execution)
  return `None` when the arithmetic overflows.
- `OHLCVBar::typical_price` and `vwap` are an overflow safe mean of high,
  low and close, so they stay within the bar's range near the bounds.
- `OrderIntensityEstimator::fill_probability` is 1, not `None`, when the
  intensity times the horizon overflows.
- Backtest CAGR computes the exponential of a negative argument as
  `1 / exp(-x)`, so a short losing run annualizes toward -100% instead of
  far below it.
- Mid prices (`MarketTick`, `Quote`, data feed quotes, the VPIN quote
  rule) are exact near the `Decimal` bounds instead of saturating the sum,
  and every book and trade flow imbalance is invariant under a common
  scale of its sides, even when their sum is not representable.

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
  Calling again retries the orders that failed. An order the book no
  longer knows (filled or cancelled outside the connector) is unmapped and
  left out of the cancelled responses instead of failing every later
  call, and an order re-mapped by a concurrent modify is cancelled through
  its current book id (#146).

### Breaking

These signature changes are breaking and ship in 0.7.0, not in a patch
release.

- `OptionsAdapter::time_to_terminal_ms` returns `MMResult<u64>`: a failed
  conversion to years is an error, not 0, which reads as already expired.
- Checked arithmetic in the legacy options and chain files (#166). These now
  report an overflow or a zero divisor as `MMError::NumericalError`, where
  they used to panic:
  - `OptionsAdapter::moneyness` returns `MMResult<Decimal>`, and `is_atm`
    returns `MMResult<bool>`. A zero strike is now an error, not a moneyness
    of 0, which reads as far out of the money;
  - `PositionGreeks::scale` returns `MMResult<Self>`;
  - `PortfolioGreeks::add` and `remove` return `MMResult<()>` and leave the
    portfolio unchanged on error; `dollar_delta`, `dollar_gamma`,
    `dollar_vega`, `dollar_theta` and `shares_to_hedge` return
    `MMResult<Decimal>`;
  - `PortfolioGreeks` no longer implements `Add`, `AddAssign`, `Sub` and
    `SubAssign`, which cannot report an overflow. Use the new `checked_add`
    and `checked_sub`, which return `MMResult<Self>`;
  - `OptionsMarketMaker::calculate_skew_adjustment` returns
    `MMResult<(Decimal, Decimal)>` (implementors change their signature);
  - `GreeksCircuitBreaker::check` returns `MMResult<()>`. A check that cannot
    be computed opens a closed breaker;
  - `AutoHedger::needs_hedge` returns `MMResult<bool>`, `hedge_urgency`
    `MMResult<HedgeUrgency>` and `calculate_hedge`
    `MMResult<Option<HedgeOrder>>`;
  - `LimitUtilization::calculate` returns `MMResult<Self>`;
  - `GreeksRiskManager::check_order` returns `MMResult<OrderDecision>`,
    `limit_utilization` `MMResult<LimitUtilization>`, `needs_hedge`
    `MMResult<bool>`, `calculate_hedge_order` `MMResult<Option<HedgeOrder>>`
    and `hedge_urgency` `MMResult<HedgeUrgency>`;
  - `GreeksRiskManager::update_on_fill` and `update_on_close` return
    `MMResult<()>`. When the fill cannot be booked, the Greeks are unchanged
    and so miss it: the result is the new `MMError::FillNotBooked`, and the
    circuit breaker opens and stays open until `reset` or
    `reset_circuit_breaker`. Meanwhile `needs_hedge`, `calculate_hedge_order`,
    `hedge_urgency` and `limit_utilization` return the new
    `MMError::GreeksMissFill`. A fill that is booked when the check after it
    fails is the new `MMError::FillBookedCheckFailed`, and must not be booked
    again;
  - `ChainRiskManager::add_position_greeks` and `remove_position_greeks`
    return `MMResult<()>`, `calculate_hedge` `MMResult<Vec<HedgeOrder>>`,
    `dollar_delta`, `dollar_gamma`, `dollar_vega` and `dollar_theta`
    `MMResult<Decimal>`, and `should_hedge` `MMResult<bool>`. When a fill
    cannot be booked, the result is `MMError::FillNotBooked` and the manager
    fails closed until `reset` or the new `reconcile`: `is_risk_breached` is
    true, `breached_limits` has an entry for it, and `should_hedge`,
    `calculate_hedge` and the dollar values return `MMError::GreeksMissFill`.
    `update_greeks` does not clear it;
  - `ChainMarketMaker::check_chain_risk` returns `MMResult<RiskStatus>`, which
    is `Breached` while a fill is missing, so `can_quote` is false, and
    `calculate_hedge` `MMResult<Vec<HedgeOrder>>`. The quoting methods do not
    call it, so stopping is up to the caller;
  - both managers gain `has_unbooked_fill`, and `GreeksRiskManager` gains
    `reconcile` as well, which installs recomputed Greeks and reruns the
    breaker. A later fill that books, or a successful retry, does not clear
    the state: `MMError::FillNotBooked` only promises that booking again
    cannot count that call twice.
- `AccountingStatus` has a new `snapshot_failures` field (`serde(default)`):
  the execution ids after whose post the accountant's snapshot failed, so
  the engine kept stale balances. `is_consistent` is false while it is not
  empty.
- `QueuePositionFillModel::update_queue` returns `MMResult<()>` and
  rejects a negative bid or ask size with `MMError::InvalidMarketState`,
  leaving its estimates unchanged; `simulate_fill` rejects such a tick the
  same way. Invalid sizes are never read as an empty queue, which would
  fill orders at once (#144).
- Checked arithmetic (#144): these now return `MMResult` and report an
  overflow as `MMError::NumericalError`:
  - `InventoryPosition::update_fill` returns `MMResult<()>` and leaves the
    position unchanged when the quantity or the exact average entry price
    is not representable (it no longer saturates the average);
  - `FillAccountant::snapshot` returns `MMResult<AccountingSnapshot>`
    (implementors change their signature), and
    `BacktestEngine::accounting_snapshot` returns
    `Option<MMResult<AccountingSnapshot>>`; a failed snapshot is recorded
    as an engine fault;
  - `DepthBasedOffering::calculate_ask_size` and `calculate_bid_size`;
  - `GridStrategy::generate_grid`, `generate_grid_with_inventory`,
    `calculate_price`, `calculate_level_size`, `price_range` and
    `max_notional_exposure`;
  - `AdaptiveSpread::bid_price` and `ask_price`, and
    `AdaptiveSpreadCalculator::calculate_spread`,
    `calculate_spread_with_volatility`,
    `calculate_weighted_orderbook_imbalance`,
    `calculate_orderbook_imbalance` and `calculate_tradeflow_imbalance`
    (a saturated depth or volume would distort the signal);
  - `RiskAversionCalibrator::calibrate_from_halflife`,
    `ParameterOptimizer::optimize_quick` and
    `OptimizedParameters::adjusted_gamma`;
  - `MultiUnderlyingManager::reallocate_capital` returns `MMResult<()>`
    and computes every allocation before applying any.
  - `PortfolioPosition::total_absolute_position` and `net_position`
    return `MMResult<Decimal>`; their Decimal sums used to panic on
    overflow.
- `LimitsPolicy` latches a fault and rejects every later check when a fill
  would overflow its position, instead of saturating it.
- The backtest engine's built in bookkeeping validates each fill's
  position, cash flow, realized PnL and fee totals before it touches the
  order, the policy or itself; a fill that would overflow any of them is
  recorded as a fault and changes nothing, instead of saturating.
- Values documented as report only (notionals, totals, utilizations,
  analytics statistics, backtest metrics) saturate at the `Decimal` bounds;
  every value that feeds a quote, a mark, a signal or state is computed
  overflow safe or returns a typed error. A `ProbabilisticFillModel` time
  component too large to represent saturates to 1, its clamp.
- `SimulatedFill::base_delta`, `ProbabilisticFillModel` probabilities and
  `MarginLedgerAccountant` base fees report overflow instead of
  saturating.

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
