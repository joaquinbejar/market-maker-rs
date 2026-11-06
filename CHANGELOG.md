# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Example `hyperliquid_validation` (Refs #214), behind `serde`: evaluates on the Hyperliquid
  capture the criteria of #87, #88, #89, #90 and #91 that the existing interfaces can evaluate,
  with no new library code. It reports walk-forward errors per day against constant parameters
  and their sensitivity to the training window; out-of-sample errors and markouts of the mid,
  weighted mid and micro-price at fixed horizons, overall and by spread and imbalance; quoting
  markout, inventory and attributed PnL over a grid of declared fees and latencies; queue-reactive
  intensities per state, event sizes and estimates per day; and Poisson, queue-reactive and Hawkes
  flow on the holdout, the Hawkes decay chosen on validation. The protocol is fixed in the code
  before any data is read: the longest stretch with no reset, split by time into training,
  validation and holdout, the holdout used for no choice, no randomness, fees declared and
  funding not applied. It streams the files, keeps only thinned inputs and says what it keeps.
  Not evaluated: #90 (b) and (c), which need a data trajectory the replay does not expose; #77,
  #180 and #91 (b) do not apply to this capture. No data is in the repository.
- Hedge actions in the RL action space (feature `rl`, #247): with
  `RlConfig::with_hedge_actions`, `rl::RlAction::Hedge` reduces the delivered position with a
  marketable limit order at the observed touch, of a size from the order size grid and never
  larger than that position, as `HedgeActions::MarketableAtTouch` states. `RlEnv::hedge_mask`
  and `Transition::hedge_mask` give the sizes allowed; a hedge outside them follows the
  declared `InvalidActionPolicy`, and `rl::ERR_HEDGE_REFUSED` marks one taken as a hold. It is
  an order like any other: the kernel's admission, gate and limits decide on it, and its cost
  reaches the reward only through the kernel's ledger. A hedge in another instrument is not
  modelled (#256). Checked on synthetic data only.
- `backtest::evaluation::block_bootstrap` (#255): the harness's stationary block bootstrap of a
  mean over a daily series, the one `Experiment` gives on the mean daily net result and on the
  daily difference to a baseline, for runs that do not go through `Experiment`.
- `options::surface::upstream` (#96), behind `options`: `UpstreamSurface` reads OptionStratLib's
  calibrated `ImpliedVolatilitySurface` and its derivatives as a `VolSurface`, so the pricer and
  the chain take it unchanged; the interpolation, wings and short and long expiries are
  upstream's. What upstream reports and does not refuse is refused by default under
  `UpstreamPolicy`: a surface whose static arbitrage report did not pass, a read whose value or
  slope spans an expiry with no slice, and fitted expiries closer than a second. A surface served
  after its as-of time is rolled down at constant implied volatility per log moneyness.
  `UpstreamStore` keeps every calibration attempt, failed ones included, with the time it became
  available, and serves at an instant only what was available then, with its age from the data.
  `SurfaceQuality` reports residuals by strike and expiry, rejected quotes and unfitted expiries
  by reason, and the static arbitrage violations. Example `upstream_surface_replay` runs it over
  synthetic daily snapshots, out of sample, against a flat volatility per expiry.
- `rl::LIBRARY_VERSION` (#100): the crate's version, which the Python bridge records in a
  checkpoint's metadata next to its own.
- Example `regime_adaptation` (#112): fixed parameters against a simple causal adaptation over
  synthetic regime changes on queue-reactive and Hawkes flow, with detection delay, false alarms,
  a labelled oracle, transfer between the simulators and a go or no-go rule fixed before the
  holdout, which says no-go here. No new public API, and nothing learned.
- `backtest::hyperliquid` (#214), behind `serde`: decoders for the three streams of the
  Hyperliquid "An Open Book" capture, a book replay, and conversions to `TopOfBook`,
  `MarketEvent`, `PriceSample`, `SizedEvent`, `QueueObservation`, `ReferenceObservation`,
  `HawkesSegment` and `QuoteExposure`, with no new dependency. Book diffs have no timestamp, so
  their times are inferred and marked. Example `hyperliquid_hour` reports what could and could
  not be established. No data is in the repository, and the book starts empty.
  Consecutive files of a stream overlap by minutes, the later one writing the earlier one's
  tail again: `RepeatFilter` and the readers `decode_order_statuses_after`,
  `read_book_diffs_after` and `read_trades_after` leave out a line of the file before too, and
  count it.
  A silence longer than a declared `ReplayConfig::reset_gap_ns`, in the diffs or the trades, empties
  the book and restarts the warm-up, with `Coverage` counting the resets and the orders cut, and
  `BookEvent::epoch` keeping events apart.
  Readers leave out a line or record exactly equal to an earlier one of its file, as the capture
  re-emits stretches of its streams, and count it. A trade stamped before the last is skipped and
  counted.
- `options::settlement` (#97), behind `options` and `margin`: `OptionBook` keeps option positions
  beside a `margin-rs` ledger, and books an expiry, exercise or assignment computed by
  OptionStratLib as one transaction: open orders' reservations released, cash as a settlement or
  the underlying as a spot execution at the strike, a lapse as a settlement of zero, the position
  extinguished. An obligation is booked even if it leaves a shortfall, which is reported. Any error leaves all of it as it was.
  Fixtures are synthetic.
- A true expiry in the example `quote_duration` (#110): `Hold::Expire` decides at the rate of
  `Hold::For` and cancels each quote after a shorter life through
  `BacktestStrategy::cancels_when_silent`, leaving the book empty until the next decision.
  Fixtures cover the book left empty, a fill before the cancel takes effect, a cancel too
  late for a fill and an order still in flight. Synthetic, with no new public API and nothing
  learned. On this tape it gave fewer fills with a worse markout than the refresh at the same
  rate, on every seed and delay reported.
- Initial margin before a derivative order in a backtest (features `margin` and `instruments`,
  #230): `backtest::DerivativeMarginAccount` shares a `margin-rs` gate and its reservation
  service between a pretrade policy and an accountant, in contracts and on the contract's
  notional. An order beyond the margin is refused whole. `BacktestEngine::with_instrument` now
  accepts a policy that declares the new `PretradePolicy::in_one_contract`, which defaults to
  `in_contracts`, and still refuses any other.
- Test `constraint_costs` (#105), behind `margin`: inventory, margin and drawdown costs with
  sign and scale checked by hand, reported apart from the violations per episode and from the
  decisions of the hard gate. Scripted proposers ask for more than the account holds, and the
  unchanged `margin-rs` gate cuts or refuses them, with a control that has no gate. Synthetic,
  with made-up budgets and no new public API. Nothing is learned, so constrained learning and a
  penalty are not compared.
- `Experiment::with_regimes` takes a description, recorded in `Manifest::regime_labeller`
  (#92), so a manifest covers its regime report as far as the caller declares it.
- Example `chain_pricing_evaluation` (#95): measures the chain's replaced pricing with the
  evaluation harness, through the real `ChainMarketMaker`, with no new public API. On synthetic
  paths the surface quotes nearer the market than the previous formula when the surface is the
  market. Net PnL shows no such gain when a crossing quote takes the touch, only when fills are
  booked at the quote's own price, and positions are unhedged. It reports three fill rules,
  volatility off by a quarter each way, and a market with a premium and a skew.
- Residual corrections to a classical quote (feature `rl`, #101): `rl::ResidualEnv` puts a
  bounded correction in ticks and lots on an Avellaneda-Stoikov or GLFT quote, discretized,
  validated against the grid and then given to the kernel's gate. A zero residual gives the
  classical action unchanged, and `rl::ResidualDecision` logs each component, what was clipped
  and why a proposal fell back. That this helps is a hypothesis of this crate: no comparison
  here says so. The Python bridge trains it with `SAC`. Checked on synthetic data only.
- Example `consistent_time_travel` (#109): chronological replay as the baseline against
  time travel as a separate labelled mode, on a small synthetic environment whose true
  values are exact, with no new public API. It reports discontinuities, reuse of the tape
  and unsupported actions, and shows a negative result: a search on the state mistakes an
  action for an informed natural move. A replay never leaves its window. No market data
  was read and no policy was trained.
- Example `offline_rl_support` (#104): which diagnostics can support an offline evaluation,
  on a small synthetic environment whose true values are exact, with no new public API. It
  covers the shape of a log, action and state coverage, actions out of support, a hidden
  confounder that passes the support checks, a log of fills only, and a go or no-go from
  reproducible diagnostics, with the truth never an input. No market log was read and no
  policy was trained, so the comparison with interactive simulation stays open.
- Calibrated parameters in the causal replay (#87): `backtest::calibrated::CalibratedQuoter`
  quotes the validated Avellaneda-Stoikov baseline in `BacktestEngine` from a
  `ParameterSchedule`, built from a `WalkForwardReport` or from constant parameters to
  compare under the same rules. Each fold's parameters hold only during its test window, and
  outside one, or in a fold that could not calibrate, it is silent and cancels what rests,
  although an order already out can still fill until then, later under latency. New provided
  `BacktestStrategy::cancels_when_silent`, false by default, so other strategies do not
  change, and outside what the financial kernel reproduces when true. The intensity scale and
  the seasonality do not enter these quotes. Checked on synthetic processes only.
- Example `quote_duration` (#110): holding a quote for a chosen time in the continuous causal
  engine, against requoting at every tick, under several delays, with fills, cancels, markout
  and inventory, and with race fixtures showing that a fill before the effective cancel stands.
  Synthetic only, with no new public API. No teacher or predictor is learned, nothing here is a
  batch market, and its delays are inputs, not measurements of a venue.
- `strategy::signal_quote::simulate_gated` replays a signal quote simulation with a
  `PretradePolicy` as a gate after the raw limits, with its reductions, rejections and
  faults on `SimulationReport` (#89). `simulate` and `simulate_detailed` give the same
  results as before.
- `backtest::queue_reactive::estimate_event_sizes` estimates the event size of each limit
  from the sizes in the data, inside a window, for `estimate_parameters` (#90). New
  `SizedEvent`, `EventSizeRecord` and `QrFailure::TooFewSizedEvents`.
- Seasonality for `backtest::hawkes` (#91): `SeasonalProfile`, a time change of the whole
  flow, `fit_seasonality` from training segments, and `score_flow_seasonal`, with at most
  `MAX_SEASONAL_BUCKETS` buckets and serde for the profile.
- `Scenario::with_fee_rate` (#92): one experiment can vary the fee rate, recorded in
  `ScenarioRecord::fee_rate`.
- The queue-reactive simulator as the market of an RL episode (feature `rl`, #90):
  `rl::qr_market` samples `QrSimulator` onto the kernel's grid, so `RlEnv` can run on it.
  The market does not react to the agent, and fills come from the kernel's fill model. A
  simulator that stops ends the rows early with its reason. New
  `QrParameters::tick_size`. Checked on simulated data only.
- A Python bridge for an external RL trainer (`bridges/python`, #100): a nested crate outside
  the root build and not published, with a worker thread per environment, thin Gymnasium
  environments for `MaskablePPO` and `SAC`, checkpoints with metadata, inference only
  loading and a protocol of training, selection and one holdout evaluation. The trainer is
  pinned to `stable-baselines3` and `sb3-contrib` 2.9.2, on the CPU with one thread. Nothing in
  the library or its features changes. Checked on synthetic data only.
- Versioned observation, action and reward spaces (feature `rl`, #99): `rl::RlEnv` drives one
  kernel for an external RL trainer or engine, and `rl::RlSpec` describes its spaces, with a
  version per space and a fingerprint that `RlSpec::ensure` checks before any step.
  Features hold only what a real agent could know at its decision, normalized by constants
  from the configuration. An action outside the mask is an error or a charged hold, as
  declared, and the mask reads no hard limit. The reward is `Decimal`, and the `f32` is
  derived from it: its ledger part is the change of `MetricId::NetPnl` at every decision,
  which `RlEnv::reconcile` checks against `backtest::evaluation`. The end of the data marks
  the inventory or closes it at the touch, as declared, and `max_steps` closes nothing. No
  hedge actions, since the kernel holds one instrument. Checked on synthetic data only;
  agreement with an external RL engine's adapter is not checked here.
- `backtest::evaluation::check_contract` (feature `instruments`, #92): refuses a contract
  the attribution cannot value, such as an inverse one, since its records carry no contract.
- Perpetual funding in `backtest::contracts` (#207): `ContractEngine::with_perpetual`
  takes a new `instruments::perpetual::PerpetualTerms`, with the declared rate and mark of
  each instant on the funding grid. Each instant books through the `margin-rs` ledger,
  rounded half to even once at the settlement asset's scale, and a position pays the
  instants in `(opened, closed]`. `ContractRunResult::total_funding` reports it apart from
  `realized_pnl` and the fees, `fundings` lists the instants, and an instant of the run with
  no declared rate is in `ContractSummary::funding_missing` and makes the run not consistent.
- A fold of `strategy::walk_forward` whose predicted fill rate or score is above the
  range of `Decimal` no longer fails the whole report (#179): `walk_forward` leaves that
  fold's intensity score out and names the exposure with the highest rate in the new
  `OutOfSampleError::intensity_out_of_support`, with
  `WalkForwardReport::out_of_support_folds` to count them. The rate is not capped, and
  `CalibratedParameters::rate_per_second` keeps its typed error for it.
- Venues that close, in `strategy::walk_forward` (#180):
  `CalibrationData::with_sessions` declares when the market is open. Each session gets
  its own price grid and gap limit, and no return crosses a closure, so an overnight
  stop neither fails a window nor dilutes the volatility with zero returns. The
  volatility is then per square root of a year of open time, which leaves the move
  across a closure out of it. A silence during a declared session is still a `PriceGap`.
  An intraday bucket reached only while closed has a multiplier of 0, and an exposure
  stamped at the close counts in the session's last bucket. New
  `CalibrationFailure::TooFewReturns`, and `CalibrationRecord::open_ms` for the time
  the market was declared open. Without sessions the results do not change. Checked on
  synthetic prices only: validation on real sessions is pending.
- Closed-form joint quotes for correlated assets (feature `multi-underlying`, #98):
  `multi_underlying::multi_asset::MultiAssetModel` implements the asymptotic quotes of
  section 3.3.3 of Bergault, Evangelista, Gueant and Vieira, arXiv:1810.04383v5, for
  symmetric exponential intensities, with `RiskView::Diagonal` as the comparison without
  correlations. With `margin` too, `MultiAssetQuoter` submits every quote to a
  `margin-rs` `RiskGate`, shares one capital budget across instruments and reports the
  cause of each active constraint. Spot instruments only, with no option logic and
  no hedging stage: a withheld side is reported and hedging is left to the caller.
- Joint option quotes (features `multi-underlying`, `options` and `margin`, #98):
  `multi_underlying::multi_asset::OptionQuoter` turns each contract's delta and vega under a
  declared factor covariance into the inputs of `MultiAssetModel`, so a hedged book is skewed less
  than a concentrated one. Greek and scenario limits hold on every way live orders can fill, one
  budget is shared across contracts, every quote holds a `margin-rs` ledger reservation, each change
  to a side is logged with its cause, and a hedge is sized and reported. A buy holds its premium,
  and a sale and a short the initial margin of `margin-rs`'s rule `short-option-im/1`, that
  crate's own specification and not a venue's margin. Example `option_quoting`, on synthetic
  data.
- Metric definitions and attribution for comparing strategies (#92):
  `backtest::evaluation` measures one window from plain marks and fills, with no
  engine types in its arithmetic. Net result is split into spread capture, inventory
  and fees, exactly in `Decimal`: a sum that would round is an error, and
  `BucketedAttribution::reconcile` compares the result with the books to the last digit.
  Buckets by UTC day or by a caller's label sum exactly to the run. It reports signed
  markout, realized and effective spread, price impact, fills, volume, time weighted
  inventory and drawdown, each with a declared unit, sign, horizon and capital
  denominator (`MetricId::definition`). It covers linear spot positions. Hedge costs
  are reported as `HedgeCosts::NotModelled`, not as zero.
- Experiments that compare strategies under one set of rules (#92):
  `backtest::evaluation::Experiment` holds the fees, fill model, latency and windows, and
  a `Candidate` brings only a strategy built from the training ticks. A simple baseline
  is required. Validation, selection and the holdout consume each other, so one
  experiment looks at its holdout once. Every run reconciles with the engine's books; a
  run that does not, or that has a fault, stays in the report as a failed row. Scenarios
  after the first are sensitivity runs on validation only. Reports are per UTC day and per
  regime, with stationary bootstrap intervals, paired against each baseline, and a
  manifest of seeds, parameters and data hashes. Example: `baseline_evaluation`.
- Several dated futures in one backtest run (features `margin` and `instruments`,
  #199): `backtest::contracts::ContractEngine` gives each contract its own data, fill
  model and orders, on one clock and one `margin-rs` ledger, with one sided orders and
  a `ContractStrategy` that may answer any report with commands. It values each contract
  at its own latest mid, closes a contract at its expiry and books a declared
  `SettlementPrice` when published, and a position past its expiry with none makes the
  run not consistent. There is no margin requirement or liquidation. New provided
  `PretradePolicy::in_contracts`, false by default and true for `RollGuard`: it refuses
  any other policy. `BacktestEngine` and `with_instrument` do not change, and a test
  holds them equal on one contract.
- A roll through the backtest (features `arbitrage` and `margin`, #199):
  `instruments::roll::run::RollStrategy` runs a `RollPlan` on `ContractEngine`, with a
  `SharedRollGuard` as its policy, and agrees with the same plan booked directly in a
  ledger. The plan does not price an unwind, so it crosses that leg's latest observed
  book, within an age the caller gives, or it is a fault. A plan for positions the
  ledger does not hold is a fault and sends nothing. New `ContractView::now_ms`, the
  time of a decision, on every `ContractStrategy` callback, and new provided
  `ContractStrategy::next_wake_ms`, `on_wake` and `take_fault`, with
  `MAX_STRATEGY_FAULTS_PER_CALLBACK`.
- The net result of a calendar spread under an assumed exit (feature `arbitrage`,
  #202): `instruments::calendar::evaluate_calendar` buys one linear dated future and
  sells another of the same underlying. As two futures do not converge to each other,
  the exit spread and its date, before the near expiry, are required inputs, and the
  result reports the entry spread and that assumption apart, with entry fees, exit
  costs and carry itemized. Settlement of the near leg is not modelled.
  A non-positive entry spread can pass. New `BasisError::ExitOutsideWindow`.
- The lifecycle of a dated future (feature `instruments`, #94):
  `instruments::futures::FutureTerms` joins a specification with how it settles
  and who publishes its settlement price, with a phase by the clock and a cash
  settlement against a `SettlementPrice` that declares its contract, source,
  expiry and publication time. With `margin` too, `settle_in_ledger` books it and
  `phase_in_ledger` reports `Settled` from the ledger's record. `FutureCalendar`
  orders declared contracts of one product. The published price is taken as given:
  its averaging method is not verified or reproduced. Delivery is not supported.
- What may be quoted and reserved during a roll (feature `arbitrage`, #94):
  `instruments::roll::RollGuard`, a `PretradePolicy` for the live path that makes the
  near contract close only and caps the far one at the plan, with a limit on the
  exposure of both legs. With `margin` too, `RollReserves` holds both legs' margin in
  a ledger and releases the near contract's orders once it settled, and
  `FutureTerms::settle_in_gate` settles in a risk gate's ledger. The backtest engine
  does not run them; `instruments::roll::run` does, through `ContractEngine`.
- A two contract roll plan (feature `arbitrage`, #94):
  `instruments::roll::RollPlan` moves a position from a dated future to a later one,
  preserving `contracts * size`: the underlying on a linear contract, the contract
  value on an inverse one. It keeps every observed fill itself, so positions and the
  exposure gap stay right when one leg fills alone or a fill does not fit its intents,
  which then fault. With `margin` too, `reconcile` compares it with a ledger, per fill
  and per leg, reporting what is unexplained as a quantity. It only emits intents, and
  `instruments::roll::run::RollStrategy` runs it through
  `backtest::contracts::ContractEngine`, with `SharedRollGuard` as the engine's policy.
- The net edge of a basis trade (new feature `arbitrage`, off by default, #94):
  `instruments::basis::evaluate_basis` buys one leg and sells the other, a spot and
  a linear dated future, through `arbitrage-rs`, whose types stay out of this
  crate's public API. It itemizes entry fees, required exit costs and carry, and
  reports the convergence date and an estimate of the capital given by the caller's
  rate. Selling spot needs a declared borrow. Expired, missing, stale or off tick
  quotes, too little size and too little edge are rejections. Two futures are not
  evaluated, as they do not converge to each other.
- Derivative contracts in the backtest engine (#191, features `margin` and
  `instruments`): `BacktestEngine::with_instrument(spec, accountant)` runs a
  linear or inverse future or perpetual, with `MarginLedgerAccountant` on it, in
  contracts and in the settlement asset, BTC for an inverse contract, so the
  initial capital must be too. The run is refused for a pretrade policy, whose
  limits read `price * quantity`, or for an accountant of other terms, and the
  checks run again at the start of every run. New `FeeCurrency::Settlement` and
  `FeeAsset::Settlement`, and `FillAccountant::unrealized_at`,
  `requires_instrument` and `check_instrument`, all with defaults. Funding,
  expiry and settlement are not simulated, as funding is #207 and `ContractEngine` runs
  a dated future's expiry and settlement (#199), and neither are margin
  requirements or liquidation. `AccountingStatus` gains `valuation_failures` and
  `setup_refused`, and is not consistent with either. A derivative needs a ledger
  with no other open position. New provided
  `FillAccountant::begin_from_current_state`, which the engine calls when it is
  handed an accountant: `MarginLedgerAccountant` then numbers its executions
  after those already in the ledger, in any of its journals and on spot too, so
  a fill of a run can no longer collide with them (#196), and a replay onto a
  ledger handed over again books as new fills, not as duplicates, so it is no
  way to recover a run. `AccountingStatus::dropped_fills` counts fills
  dropped because their fee or booking could not be computed, which used to
  leave `is_consistent` true.
- Orders and fills against a contract specification (feature `instruments`,
  #93): `instruments::orders::check_order` refuses an order for another
  symbol, off the tick or the lot, or with a missing or unexpected price, with
  a typed `OrderSpecError`, and `fill_notional` gives a fill's notional in the
  asset of the contract, where `Fill::notional` is only right for a linear
  contract of multiplier one. With `margin` too,
  `instruments::margin::fill_execution` converts a fill into a `margin-rs`
  execution.
- `backtest::hawkes`: the QRH-I queue-reactive Hawkes model of Wu, Rambaldi, Muzy
  and Bacry for one queue, with exponential kernels, a stability check, excitation
  in a restorable state, expectation maximization, and a comparison of Poisson,
  queue-reactive and Hawkes models by likelihood, time-rescaled residuals, flow
  and, reported apart, a simple fill measure (#91).
- `backtest::queue_reactive`: a seedable simulator of aggregated queues after
  Huang, Lehalle and Rosenbaum (Model III over Model I), with maximum likelihood
  intensities per queue size, exportable parameters, bounded restorable state,
  a declared policy for unsupported states, and distribution discrepancies with
  state coverage against a constant intensity baseline (#90).
- The kernel can fill over several rows: `KernelFillModel::QueuePosition`
  (with `KernelConfig::with_fill_model`) mirrors `QueuePositionFillModel` on
  the lot grid, report for report, and produces `ExecKind::PartialFill`.
  `CapacityNeed` has a new field, `partial_reports_per_order`: `DLV` must be at
  least `(3 + partial_reports_per_order) * ORD`. It is zero for
  `KernelFillModel::Immediate`, whose behaviour, encoding and recorded
  trajectories are unchanged. `decode` checks that the queued fill reports of
  each order add up (#177).
- `tests/kernel_margin.rs` (features `kernel` and `margin`): margin-rs ledger
  reservations driven by the kernel's events, through restore, forks and worker
  assignment. The kernel gate itself keeps using `RiskLimits` (#177).
- The hedges of the kernel's replay gate come from arbitrage-rs
  (`plan_band_hedge`), a new dev-dependency; the recorded trajectories are
  unchanged (#177).
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

- `optionstratlib` requirement 0.24.1, for the surface's derivatives (#96).
- `optionstratlib` 0.24 and `option-chain-orderbook` 0.15 (breaking, as their types are in this
  crate's public API). `ExpirationDate`'s conversions from the wall clock now need the `clock`
  feature, which `options` turns on, so results are unchanged.
- `MarginLedgerAccountant::snapshot` keeps running sums instead of walking the
  journal, so a run costs linear rather than quadratic time in its fills. The
  values and the errors are the same: a sum that overflows stays an error until
  a reset (#195).
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

- `MarginLedgerAccountant` adds a replayed fill's base fee only once. A duplicate
  used to add it again, to `base_fees` and against the position; the engine never
  posts twice, so this reached direct callers only (#195).
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

- `rl::ACTION_SPACE_VERSION` is 2 (#247), so every `rl::SpecId` changes and a checkpoint
  trained under action space 1 is refused by `RlSpec::ensure`. `rl::RlAction` gains `Hedge`,
  `rl::RlConfig` gains `hedge_actions`, and `rl::HedgeActions` gains `MarketableAtTouch`.
  Behaviour with hedge actions off is unchanged. The Python bridge is built for action
  space 2.
- `margin-rs` 0.2, whose types are in this crate's public API (#97, #98). `options::settlement`
  now settles an option held in the ledger as a registered `OptionInstrument`, through
  `Ledger::settle_option`: `OptionPosition::new` takes the option's `InstrumentId` instead of a
  cash asset, `OptionBook::open` takes the ledger and checks the instrument's strike, right,
  exercise style and multiplier and the ledger's position against the option, and
  `SettlementRecord` gains `outcome` and `realized_pnl`. A cash option out of the money settles
  in cash for zero, as the ledger has no lapse for one. The premium is booked by the ledger at
  each fill and stays in the option's realized PnL. Inverse options are refused.
  `instruments::margin_instrument` now converts an option specification, as
  `Instrument::from_spec` does in 0.2; `fill_execution` still refuses an option's fill.
- `OptionQuoter` holds a short's capital at `margin-rs` 0.2's short option initial margin
  (#98), computed by `margin-rs`'s `rules::ShortOptionIm` and no longer from the declared
  scenarios. A sale reserves the rule's initial margin for one short contract at the round's
  spot, per contract, instead of its worst scenario value less its price, and a short position
  counts in `OptionCapital::deployed` at the rule's requirement instead of its worst scenario
  loss. A buy still holds its premium, and the part of a sale that only unwinds a long still
  holds nothing. The default rule is `short-option-im/1` in the cash asset at its ledger scale:
  per short contract, `max(15 % of the spot less the amount out of the money, 10 % of the
  spot)`, rounded up to the rule's scale. `OptionQuoter::with_short_margin` takes a rule with
  other rates, built with `ShortOptionIm::new` under its own rule id, and refuses one in another
  asset than the cash. The rule covers the initial margin of each short contract alone, at the
  round's spot; it does not net between contracts or against the underlying or the hedge, adds
  no stress beyond its formula, and leaves out the short's value, its cost to close, which
  `margin-rs` leaves to collateral and this quoter does not count against the budget; its
  maintenance margin is not used. Its rates are `margin-rs`'s own specification, not
  a venue's rule. The scenarios now feed only the scenario loss limit, so a sale's reservation
  and the budget's sizing change for every contract.
- `ContractRunResult` and `ContractSummary` have new public fields for funding (#207):
  `realized_pnl`, `total_funding` and `fundings`; `funding` and `funding_missing`. A
  `ContractStrategy` is unchanged.
These signature changes are breaking and ship in 0.7.0, not in a patch
release.

- `FeeCurrency` has a new variant, `Settlement`, and is now `#[non_exhaustive]`.
  `AccountingStatus` has three new fields, so a struct literal must set them; with
  `serde` they default. `FeeCurrency::charge` refuses `Settlement`, which needs
  the contract. `MarginLedgerAccountant` posted to before it is handed to an
  engine now resets to the state at the handover, not to the one it was wrapped
  with (#191).
- `Fill` has a new field, `symbol`, the instrument traded, so a struct literal
  must set it. With `serde` it defaults to empty, so a record written before it
  still decodes. `instruments::orders::fill_notional` and
  `instruments::margin::fill_execution` now refuse a fill whose symbol is not the
  identifier of the specification (`OrderSpecError::SymbolMismatch`, the new
  `AdapterError::SymbolMismatch`), an empty one included, where a wrong
  specification used to give a wrong amount silently (#191).
- `MockExchangeConnector::next_trade` names the symbol of `subscribe_trades`, and
  without a subscription it is an error rather than a trade of no instrument;
  `subscribe_trades` refuses an empty symbol (#191).
- `ChainMarketMaker::generate_quote` and `refresh_all_quotes` enforce the chain
  risk status: while `RiskStatus::can_quote` is false they refuse with the new
  `MMError::QuotingHalted`, before anything is valued, and when the status
  cannot be computed they return its error. This is a behaviour change, with
  the signatures unchanged: a breached limit or a failed booking stops quotes
  until it clears, and a delta above `delta_hedge_threshold` stops them until
  the caller hedges (#182).
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
    `calculate_hedge` `MMResult<Vec<HedgeOrder>>`;
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
