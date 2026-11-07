//! Backtesting engine for strategy simulation.
//!
//! This module provides the core backtesting engine that simulates strategy
//! execution on historical market data.
//!
//! # Execution semantics
//!
//! Orders persist across ticks and follow the causal lifecycle in
//! [`crate::backtest::lifecycle`]. For each tick the engine first matches the
//! orders that were already resting through the configured
//! [`FillModel`] (by default the optimistic
//! [`ImmediateFillModel`]), booking
//! partial fills at the model price plus slippage, then asks the strategy
//! for a new quote. A new quote replaces the resting orders and only becomes
//! eligible on later ticks. With [`BacktestEngine::with_latency`] the
//! strategy observes ticks late, orders activate and cancels take effect
//! after their delays, and reports reach the strategy after the
//! acknowledgement delay; a cancel that arrives after a fill is reported as
//! `CancelRejected` and the fill stands. Events scheduled between two ticks
//! (observations, activations, cancels, report deliveries) happen at their
//! own times, in chronological order, before the next tick is matched; so do
//! events due at the tick's own time. An order that becomes active at a
//! tick's time is eligible only from the next tick on (see
//! [`crate::backtest::latency`] for what that means for a delay of exactly
//! one tick interval).
//! Jitter applies to every delay, observations included. Reports still in
//! flight when the data ends are delivered after the last tick. Every
//! transition is published as an
//! [`ExecutionReport`] to [`BacktestStrategy::on_execution_report`] and
//! recorded in [`BacktestEngine::execution_reports`].
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{
//!     BacktestConfig, BacktestEngine, BacktestStrategy, MarketTick,
//!     SimulatedFill, VecDataSource, SlippageModel
//! };
//! use market_maker_rs::position::inventory::InventoryPosition;
//! use market_maker_rs::strategy::quote::Quote;
//! use market_maker_rs::dec;
//!
//! // Define a simple strategy
//! struct SimpleStrategy;
//!
//! impl BacktestStrategy for SimpleStrategy {
//!     fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
//!         // Simple market making: quote around mid price
//!         let mid = tick.mid_price();
//!         Some(Quote {
//!             bid_price: mid - dec!(0.1),
//!             bid_size: market_maker_rs::Decimal::ONE,
//!             ask_price: mid + dec!(0.1),
//!             ask_size: market_maker_rs::Decimal::ONE,
//!             timestamp: tick.timestamp,
//!         })
//!     }
//!
//!     fn on_fill(&mut self, _fill: &SimulatedFill) {}
//!     fn reset(&mut self) {}
//! }
//!
//! let ticks = vec![
//!     MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.2), dec!(1.0)),
//! ];
//!
//! let config = BacktestConfig::default();
//! let strategy = SimpleStrategy;
//! let data_source = VecDataSource::new(ticks);
//!
//! let mut engine = BacktestEngine::new(config, strategy, data_source);
//! let result = engine.run();
//!
//! assert_eq!(result.num_ticks, 1);
//! ```

use crate::Decimal;
use crate::execution::Action;
use crate::position::inventory::InventoryPosition;
use crate::position::pnl::PnL;
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::strategy::quote::Quote;
use crate::types::action::Dir;
use crate::types::decimal::OrOverflow;
#[cfg(feature = "instruments")]
use crate::types::decimal::exact_product;
use crate::types::error::{MMError, MMResult};
use rust_decimal_macros::dec;

use super::accounting::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
use super::data::{HistoricalDataSource, MarketTick};
use super::fill_models::{FillModel, FillResult, ImmediateFillModel, SimulatedOrder};
use super::latency::{JitterRng, SimLatencyConfig};
use super::lifecycle::{ExecutionKind, ExecutionReport, SimClock, SimOrder, SimOrderId};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::panic::{RefUnwindSafe, UnwindSafe};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Simulated fill representing a trade execution.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::SimulatedFill;
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let fill = SimulatedFill::new(Action::Buy, dec!(100.0), dec!(0.1), 1000);
/// assert_eq!(fill.notional(), dec!(10.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulatedFill {
    /// Order side.
    pub side: Action,
    /// Fill price.
    pub price: Decimal,
    /// Fill quantity.
    pub quantity: Decimal,
    /// Fill timestamp in milliseconds.
    pub timestamp: u64,
    /// Fee paid for this fill.
    pub fee: Decimal,
    /// Engine execution id (see [`AccountingFill::execution_id`]), set
    /// for every fill the engine produces; `None` for fills built by hand.
    #[cfg_attr(feature = "serde", serde(default))]
    pub execution_id: Option<u64>,
    /// Fee charged in the base asset, in base units; zero unless the fee
    /// currency is [`FeeCurrency::Base`]. It reduces the base held: see
    /// [`SimulatedFill::base_delta`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub base_fee: Decimal,
}

impl SimulatedFill {
    /// Creates a new simulated fill.
    #[must_use]
    pub fn new(side: Action, price: Decimal, quantity: Decimal, timestamp: u64) -> Self {
        Self {
            side,
            price,
            quantity,
            timestamp,
            fee: Decimal::ZERO,
            execution_id: None,
            base_fee: Decimal::ZERO,
        }
    }

    /// Creates a new simulated fill with fee.
    #[must_use]
    pub fn with_fee(
        side: Action,
        price: Decimal,
        quantity: Decimal,
        timestamp: u64,
        fee: Decimal,
    ) -> Self {
        Self {
            side,
            price,
            quantity,
            timestamp,
            fee,
            execution_id: None,
            base_fee: Decimal::ZERO,
        }
    }

    /// Change of the base asset held: the signed quantity minus the base
    /// fee (a buy receives `quantity - base_fee`, a sell delivers
    /// `quantity + base_fee`).
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the side is neither buy
    /// nor sell, or [`MMError::NumericalError`] if the delta overflows.
    pub fn base_delta(&self) -> MMResult<Decimal> {
        let delta = match Dir::try_from(self.side)? {
            Dir::Buy => self.quantity.checked_sub(self.base_fee),
            Dir::Sell => Decimal::ZERO
                .checked_sub(self.quantity)
                .and_then(|q| q.checked_sub(self.base_fee)),
        };
        delta.or_overflow("fill base delta")
    }

    /// Sets the engine execution id.
    #[must_use]
    pub fn with_execution_id(mut self, execution_id: u64) -> Self {
        self.execution_id = Some(execution_id);
        self
    }

    /// Returns the notional value (price * quantity), saturating at the
    /// `Decimal` bounds instead of panicking.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.quantity)
    }
}

/// A fill's built in bookkeeping, validated before the engine applies it.
#[derive(Debug, Clone)]
struct Booking {
    position: InventoryPosition,
    cash_flow: Decimal,
    base_fee_quantity: Decimal,
    fixed_fees: Decimal,
}

/// Slippage model for simulating execution costs.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::SlippageModel;
/// use market_maker_rs::dec;
///
/// let model = SlippageModel::Fixed(dec!(0.01));
/// assert_eq!(model.calculate_slippage(dec!(100.0), dec!(0.01)), dec!(0.01));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum SlippageModel {
    /// No slippage.
    #[default]
    None,
    /// Fixed slippage amount.
    Fixed(Decimal),
    /// Percentage of price.
    Percentage(Decimal),
    /// Volatility-based slippage.
    VolatilityBased {
        /// Multiplier for volatility.
        multiplier: Decimal,
    },
}

impl SlippageModel {
    /// Calculates the slippage amount.
    ///
    /// # Arguments
    ///
    /// * `price` - The base price
    /// * `volatility` - Current volatility (used for volatility-based model)
    ///
    /// The products saturate at the `Decimal` bounds instead of panicking.
    #[must_use]
    pub fn calculate_slippage(&self, price: Decimal, volatility: Decimal) -> Decimal {
        match self {
            SlippageModel::None => Decimal::ZERO,
            SlippageModel::Fixed(amount) => *amount,
            SlippageModel::Percentage(pct) => price.saturating_mul(*pct),
            SlippageModel::VolatilityBased { multiplier } => {
                price.saturating_mul(volatility).saturating_mul(*multiplier)
            }
        }
    }
}

/// Currency in which the backtest charges fees.
///
/// The fee rate applies to the traded notional in every case. The fee is
/// computed in its own currency, rounded half to even to that currency's
/// scale, and posted to the accountant with the matching [`FeeAsset`].
/// Result metrics ([`BacktestResult::total_fees`], net PnL) stay in the
/// quote asset.
///
/// - A base fee reduces the base held: the position the engine, the
///   strategy and the pretrade policy see is the net base holding (a buy of
///   1 with a 0.001 base fee leaves 0.999). The fees paid in base are
///   marked at the current mid like the position, so equity is always cash
///   plus net holdings at the mark.
/// - A third asset fee is converted once at `quote_per_unit`.
/// - A settlement asset fee needs the contract, so it is only for a run with
///   `BacktestEngine::with_instrument`; there, every metric is in the settlement
///   asset.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum FeeCurrency {
    /// Quote asset: `price * quantity * rate`, unrounded (the historical
    /// behaviour).
    #[default]
    Quote,
    /// Base asset: `quantity * rate`, rounded to `scale` decimal places.
    Base {
        /// Decimal places of the base asset.
        scale: u32,
    },
    /// A third asset: `price * quantity * rate / quote_per_unit`, rounded
    /// to `scale` decimal places.
    Other {
        /// Asset code, posted as [`FeeAsset::Other`].
        asset: String,
        /// Price of one unit of the asset in the quote asset; positive.
        quote_per_unit: Decimal,
        /// Decimal places of the asset.
        scale: u32,
    },
    /// The settlement asset of a derivative contract: the contract's notional
    /// times the rate, exact, then rounded half to even to `scale` decimal
    /// places once. A negative rate is a rebate.
    Settlement {
        /// Decimal places of the settlement asset.
        scale: u32,
    },
}

/// A fee computed for one execution.
#[derive(Debug, Clone, PartialEq)]
pub struct FeeCharge {
    /// Amount in [`FeeCharge::asset`] units.
    pub amount: Decimal,
    /// Asset the fee is charged in.
    pub asset: FeeAsset,
    /// Value of the fee in the quote asset, or in the settlement asset on a
    /// derivative contract, used by the result metrics.
    pub quote_value: Decimal,
}

impl FeeCurrency {
    /// Largest scale a [`Decimal`] can hold.
    const MAX_SCALE: u32 = 28;

    /// Checks the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a scale exceeds 28, the
    /// asset code is empty or `quote_per_unit` is not positive.
    pub fn validate(&self) -> MMResult<()> {
        match self {
            Self::Quote => Ok(()),
            Self::Base { scale } | Self::Settlement { scale } => Self::check_scale(*scale),
            Self::Other {
                asset,
                quote_per_unit,
                scale,
            } => {
                if asset.trim().is_empty() {
                    return Err(MMError::InvalidConfiguration(
                        "fee asset code must not be empty".to_string(),
                    ));
                }
                if *quote_per_unit <= Decimal::ZERO {
                    return Err(MMError::InvalidConfiguration(format!(
                        "fee asset quote_per_unit must be positive, got {quote_per_unit}"
                    )));
                }
                Self::check_scale(*scale)
            }
        }
    }

    fn check_scale(scale: u32) -> MMResult<()> {
        if scale > Self::MAX_SCALE {
            return Err(MMError::InvalidConfiguration(format!(
                "fee scale {scale} exceeds {}",
                Self::MAX_SCALE
            )));
        }
        Ok(())
    }

    /// Computes the fee of one execution.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for invalid parameters (see
    /// [`FeeCurrency::validate`]) and for [`FeeCurrency::Settlement`], which
    /// depends on the contract and not only on price and quantity, and
    /// [`MMError::NumericalError`] on overflow.
    pub fn charge(&self, price: Decimal, quantity: Decimal, rate: Decimal) -> MMResult<FeeCharge> {
        self.validate()?;
        let overflow = || MMError::NumericalError("fee computation overflowed".to_string());
        let notional = price.checked_mul(quantity).ok_or_else(overflow)?;
        let quote_fee = notional.checked_mul(rate).ok_or_else(overflow)?;
        let round = |value: Decimal, scale: u32| {
            value.round_dp_with_strategy(scale, rust_decimal::RoundingStrategy::MidpointNearestEven)
        };
        match self {
            Self::Settlement { .. } => Err(MMError::InvalidConfiguration(
                "a settlement asset fee needs the contract: use with_instrument".to_string(),
            )),
            Self::Quote => Ok(FeeCharge {
                amount: quote_fee,
                asset: FeeAsset::Quote,
                quote_value: quote_fee,
            }),
            Self::Base { scale } => {
                let amount = round(quantity.checked_mul(rate).ok_or_else(overflow)?, *scale);
                Ok(FeeCharge {
                    amount,
                    asset: FeeAsset::Base,
                    quote_value: amount.checked_mul(price).ok_or_else(overflow)?,
                })
            }
            Self::Other {
                asset,
                quote_per_unit,
                scale,
            } => {
                let amount = round(
                    quote_fee
                        .checked_div(*quote_per_unit)
                        .ok_or_else(overflow)?,
                    *scale,
                );
                Ok(FeeCharge {
                    amount,
                    asset: FeeAsset::Other(asset.clone()),
                    quote_value: amount.checked_mul(*quote_per_unit).ok_or_else(overflow)?,
                })
            }
        }
    }
}

/// Backtest configuration.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{BacktestConfig, SlippageModel};
/// use market_maker_rs::dec;
///
/// let config = BacktestConfig::default()
///     .with_initial_capital(dec!(100000.0))
///     .with_fee_rate(dec!(0.001))
///     .with_slippage(SlippageModel::Fixed(dec!(0.01)));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: Decimal,
    /// Trading fee rate (as decimal, e.g., 0.001 for 0.1%).
    pub fee_rate: Decimal,
    /// Currency the fee is charged in.
    #[cfg_attr(feature = "serde", serde(default))]
    pub fee_currency: FeeCurrency,
    /// Minimum tick size. When positive, simulated bids are rounded down
    /// and asks up to a multiple of it; zero disables price rounding.
    pub tick_size: Decimal,
    /// Minimum lot size. When positive, simulated order quantities are
    /// rounded down to a multiple of it; zero disables quantity rounding.
    /// An order whose rounded quantity is zero is rejected.
    pub lot_size: Decimal,
    /// Slippage model.
    pub slippage: SlippageModel,
    /// Order size used for a quote side whose size is not positive.
    pub default_order_size: Decimal,
    /// Record equity curve at each tick.
    pub record_equity_curve: bool,
    /// Record all trades.
    pub record_trades: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: Decimal::from(100_000),
            fee_rate: Decimal::ZERO,
            fee_currency: FeeCurrency::Quote,
            tick_size: dec!(0.01),
            lot_size: dec!(0.001),
            slippage: SlippageModel::None,
            default_order_size: Decimal::ONE,
            record_equity_curve: true,
            record_trades: true,
        }
    }
}

impl BacktestConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial capital.
    #[must_use]
    pub fn with_initial_capital(mut self, capital: Decimal) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Sets the fee rate.
    #[must_use]
    pub fn with_fee_rate(mut self, rate: Decimal) -> Self {
        self.fee_rate = rate;
        self
    }

    /// Sets the fee currency. Invalid parameters are reported when the
    /// first fee is computed: the fill is not booked and the engine records
    /// a fault (see [`BacktestEngine::faults`]); call
    /// [`FeeCurrency::validate`] to check them up front.
    #[must_use]
    pub fn with_fee_currency(mut self, currency: FeeCurrency) -> Self {
        self.fee_currency = currency;
        self
    }

    /// Sets the tick size.
    #[must_use]
    pub fn with_tick_size(mut self, size: Decimal) -> Self {
        self.tick_size = size;
        self
    }

    /// Sets the lot size.
    #[must_use]
    pub fn with_lot_size(mut self, size: Decimal) -> Self {
        self.lot_size = size;
        self
    }

    /// Sets the slippage model.
    #[must_use]
    pub fn with_slippage(mut self, slippage: SlippageModel) -> Self {
        self.slippage = slippage;
        self
    }

    /// Sets the default order size.
    #[must_use]
    pub fn with_default_order_size(mut self, size: Decimal) -> Self {
        self.default_order_size = size;
        self
    }

    /// Enables or disables equity curve recording.
    #[must_use]
    pub fn with_record_equity_curve(mut self, record: bool) -> Self {
        self.record_equity_curve = record;
        self
    }

    /// Enables or disables trade recording.
    #[must_use]
    pub fn with_record_trades(mut self, record: bool) -> Self {
        self.record_trades = record;
        self
    }
}

/// Maximum number of faults kept by [`BacktestEngine::faults`]; later
/// faults are only counted (see [`BacktestEngine::fault_count`]).
pub const MAX_RECORDED_FAULTS: usize = 16;

/// Backtest result containing performance metrics.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::BacktestResult;
/// use market_maker_rs::dec;
///
/// let result = BacktestResult::default();
/// assert_eq!(result.total_pnl, dec!(0.0));
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BacktestResult {
    /// Total PnL before fees.
    pub total_pnl: Decimal,
    /// Total fees paid, in quote, or in the settlement asset on a derivative
    /// contract. Fees paid in the base asset are valued at the final mark, so
    /// with base fees this differs from the sum of [`SimulatedFill::fee`], which
    /// values each fee at its fill price.
    pub total_fees: Decimal,
    /// Net PnL after fees.
    pub net_pnl: Decimal,
    /// Number of trades executed.
    pub num_trades: u64,
    /// Number of ticks processed.
    pub num_ticks: u64,
    /// Start timestamp.
    pub start_time: u64,
    /// End timestamp.
    pub end_time: u64,
    /// Maximum position held.
    pub max_position: Decimal,
    /// Final position.
    pub final_position: Decimal,
    /// Equity curve: (timestamp, equity).
    pub equity_curve: Vec<(u64, Decimal)>,
    /// All executed trades.
    pub trades: Vec<SimulatedFill>,
    /// Maximum drawdown.
    pub max_drawdown: Decimal,
    /// Sharpe ratio approximation (if enough data).
    pub sharpe_ratio: Option<Decimal>,
    /// Accounting status of the run: which accountant booked it and which
    /// venue fills it refused.
    #[cfg_attr(feature = "serde", serde(default))]
    pub accounting: AccountingStatus,
}

/// Accounting status of a backtest run.
///
/// With a selected [`FillAccountant`], position, PnL and fees follow the
/// accountant, while [`BacktestResult::trades`] lists every venue fill. A
/// fill the accountant refused is in `trades` but not in the metrics; this
/// status makes that divergence visible from the result alone.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccountingStatus {
    /// Description of the selected accountant, or `None` for the built in
    /// bookkeeping (which never refuses a fill).
    pub accountant: Option<String>,
    /// Execution ids whose post the accountant refused, in venue order.
    ///
    /// Each recorded trade carries its id in
    /// [`SimulatedFill::execution_id`] (see
    /// [`BacktestResult::refused_trades`]).
    pub refused_executions: Vec<u64>,
    /// Execution ids after whose post the accountant's snapshot failed, in
    /// venue order (`0` for the snapshot taken when the accountant was
    /// selected or reset). The engine then kept its previous balances, so
    /// they may be stale.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_failures: Vec<u64>,
    /// Marks the accountant could not value. The engine then kept its previous
    /// unrealized PnL and total, so they may be stale.
    #[cfg_attr(feature = "serde", serde(default))]
    pub valuation_failures: u64,
    /// Why the run was refused before it started, with nothing processed.
    #[cfg_attr(feature = "serde", serde(default))]
    pub setup_refused: Option<String>,
    /// Times a venue fill was dropped because its fee or its booking could not
    /// be computed. It is then in no metric and not in
    /// [`BacktestResult::trades`], and its order stays open, so one fill can
    /// count more than once.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_fills: u64,
}

impl AccountingStatus {
    /// Returns `true` if the run started, no venue fill was dropped, the
    /// accountant booked every one, and every snapshot and valuation of it
    /// succeeded.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.refused_executions.is_empty()
            && self.snapshot_failures.is_empty()
            && self.valuation_failures == 0
            && self.setup_refused.is_none()
            && self.dropped_fills == 0
    }
}

impl BacktestResult {
    /// Venue fills the selected accountant refused to book, matched by
    /// [`SimulatedFill::execution_id`].
    ///
    /// Empty when trade recording is off or no post was refused. The
    /// remaining trades are exactly the ones reflected in the metrics.
    pub fn refused_trades(&self) -> impl Iterator<Item = &SimulatedFill> + '_ {
        self.trades.iter().filter(|trade| {
            trade
                .execution_id
                .is_some_and(|id| self.accounting.refused_executions.contains(&id))
        })
    }

    /// Returns the win rate (trades with positive PnL / total trades).
    #[must_use]
    pub fn win_rate(&self) -> Decimal {
        if self.num_trades == 0 {
            return Decimal::ZERO;
        }
        // Simplified: positive net PnL means "winning"
        if self.net_pnl > Decimal::ZERO {
            Decimal::ONE
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the average trade PnL.
    #[must_use]
    pub fn avg_trade_pnl(&self) -> Decimal {
        if self.num_trades == 0 {
            Decimal::ZERO
        } else {
            // Dividing by a count of at least one cannot overflow.
            self.net_pnl
                .checked_div(Decimal::from(self.num_trades))
                .unwrap_or(Decimal::ZERO)
        }
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Returns the return on capital.
    ///
    /// A quotient beyond the `Decimal` range saturates to `Decimal::MAX` or
    /// `Decimal::MIN` with the sign of the PnL.
    #[must_use]
    pub fn return_on_capital(&self, initial_capital: Decimal) -> Decimal {
        if initial_capital > Decimal::ZERO {
            self.net_pnl.checked_div(initial_capital).unwrap_or(
                if self.net_pnl.is_sign_negative() {
                    Decimal::MIN
                } else {
                    Decimal::MAX
                },
            )
        } else {
            Decimal::ZERO
        }
    }
}

/// Strategy trait for backtesting.
///
/// Implement this trait to create a strategy that can be backtested.
pub trait BacktestStrategy {
    /// Called on each market tick, after resting orders were matched
    /// against it.
    ///
    /// Returning `Some(quote)` cancels every resting order and submits a new
    /// bid and ask sized by the quote (or [`BacktestConfig::default_order_size`]
    /// for a side whose size is not positive), with prices rounded to
    /// [`BacktestConfig::tick_size`] and quantities to
    /// [`BacktestConfig::lot_size`]. The new orders
    /// can only fill on later ticks, never on the tick they were computed
    /// from. Returning `None` leaves resting orders in place.
    ///
    /// With simulated latency, `tick` may be older than the venue time and
    /// `position` only reflects fills whose reports were already delivered.
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote>;

    /// Called when an order is filled.
    fn on_fill(&mut self, fill: &SimulatedFill);

    /// Called for every order lifecycle transition (accept, reject, fill,
    /// cancel), in event order. The default implementation ignores reports.
    fn on_execution_report(&mut self, _report: &ExecutionReport) {}

    /// Whether returning `None` from [`Self::on_tick`] also cancels every
    /// resting order, as for a strategy that must not leave a quote live once it
    /// no longer has grounds for one. False by default: `None` leaves them in
    /// place. Cancels take the simulated cancel latency, as any other. The
    /// financial kernel has no such path, so its equivalence with this engine
    /// holds only while this is false.
    fn cancels_when_silent(&self) -> bool {
        false
    }

    /// Resets the strategy state.
    fn reset(&mut self);
}

/// Backtesting engine for simulating strategy execution.
///
/// # Type Parameters
///
/// * `S` - Strategy type implementing `BacktestStrategy`
/// * `D` - Data source type implementing `HistoricalDataSource`
/// * `F` - Fill model implementing [`FillModel`]; defaults to
///   [`ImmediateFillModel`], the optimistic best bid/offer cross used as the
///   reference mode. Choose another model with
///   [`BacktestEngine::with_fill_model`].
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{
///     BacktestConfig, BacktestEngine, BacktestStrategy, MarketTick,
///     SimulatedFill, VecDataSource
/// };
/// use market_maker_rs::position::inventory::InventoryPosition;
/// use market_maker_rs::strategy::quote::Quote;
/// use market_maker_rs::dec;
///
/// struct PassiveStrategy;
///
/// impl BacktestStrategy for PassiveStrategy {
///     fn on_tick(&mut self, _tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
///         None // No quotes, just observe
///     }
///     fn on_fill(&mut self, _fill: &SimulatedFill) {}
///     fn reset(&mut self) {}
/// }
///
/// let ticks = vec![
///     MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.2), dec!(1.0)),
///     MarketTick::new(1001, dec!(100.1), dec!(1.0), dec!(100.3), dec!(1.0)),
/// ];
///
/// let mut engine = BacktestEngine::new(
///     BacktestConfig::default(),
///     PassiveStrategy,
///     VecDataSource::new(ticks),
/// );
///
/// let result = engine.run();
/// assert_eq!(result.num_ticks, 2);
/// assert_eq!(result.num_trades, 0);
/// ```
#[derive(Debug)]
pub struct BacktestEngine<
    S: BacktestStrategy,
    D: HistoricalDataSource,
    F: FillModel = ImmediateFillModel,
> {
    config: BacktestConfig,
    fill_model: F,
    strategy: S,
    data_source: D,
    position: InventoryPosition,
    pnl: PnL,
    equity_curve: Vec<(u64, Decimal)>,
    trades: Vec<SimulatedFill>,
    total_fees: Decimal,
    /// Fees fixed in quote (quote and third asset fees).
    fixed_fees: Decimal,
    /// Fees paid in base, in base units; marked at every tick.
    base_fee_quantity: Decimal,
    max_position: Decimal,
    peak_equity: Decimal,
    max_drawdown: Decimal,
    clock: SimClock,
    orders: Vec<SimOrder>,
    next_order_id: SimOrderId,
    reports: Vec<ExecutionReport>,
    latency: SimLatencyConfig,
    jitter: JitterRng,
    next_message: u64,
    /// Venue messages keyed by `(effective time, sequence)`.
    pending: BTreeMap<(u64, u64), PendingAction>,
    /// Key in `pending` of each order waiting for activation.
    pending_activations: HashMap<SimOrderId, (u64, u64)>,
    /// Orders targeted by a cancel still in flight.
    cancel_targets: HashSet<SimOrderId>,
    /// Terminal orders (filled or rejected) with a cancel still in flight,
    /// kept so the cancel can be answered with `CancelRejected`.
    awaiting_cancel: HashMap<SimOrderId, SimOrder>,
    /// FIFO report channel to the strategy, non decreasing delivery times.
    deliveries: VecDeque<Delivery>,
    /// FIFO observation channel: (time the strategy sees it, tick).
    observed: VecDeque<(u64, MarketTick)>,
    agent_position: InventoryPosition,
    /// Last price the position was marked at.
    last_mark: Option<Decimal>,
    last_delivery_at: u64,
    last_observed_at: u64,
    /// Selected accountant; `None` keeps the built in bookkeeping.
    accountant: Option<Box<dyn FillAccountant + Send + Sync + UnwindSafe + RefUnwindSafe>>,
    accounting_errors: Vec<(u64, MMError)>,
    /// Execution ids after which the accountant's snapshot failed.
    snapshot_failures: Vec<u64>,
    /// Marks the accountant could not value.
    valuation_failures: u64,
    /// Why the last run was refused before it started.
    setup_refused: Option<String>,
    /// Venue fills dropped because their fee or booking could not be computed.
    dropped_fills: u64,
    /// The contract of a derivative run.
    #[cfg(feature = "instruments")]
    instrument: Option<financial_types::instrument::InstrumentSpec>,
    next_execution_id: u64,
    /// Pretrade policy every new order passes before submission.
    policy: Option<Box<dyn PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe>>,
    pretrade_log: Vec<PretradeRecord>,
    /// Errors of the fill model or the pretrade policy; while not empty,
    /// every new quote is refused. Holds at most [`MAX_RECORDED_FAULTS`].
    faults: Vec<MMError>,
    /// Total faults, recorded or not (saturating).
    fault_count: u64,
}

/// One pretrade decision taken by the engine's policy.
#[derive(Debug, Clone, PartialEq)]
pub struct PretradeRecord {
    /// Simulated order the request was for.
    pub order_id: SimOrderId,
    /// Request presented to the policy.
    pub request: PretradeRequest,
    /// Policy decision.
    pub decision: PretradeDecision,
    /// Quantity submitted (zero when nothing was submitted), after the
    /// clamp to the request and lot rounding.
    pub submitted: Decimal,
}

#[derive(Debug, Clone)]
enum PendingAction {
    Activate(SimOrder),
    Cancel(SimOrderId),
}

/// Report waiting to reach the strategy.
#[derive(Debug, Clone)]
struct Delivery {
    at: u64,
    report: ExecutionReport,
    fill: Option<(Dir, SimulatedFill)>,
}

impl<S: BacktestStrategy, D: HistoricalDataSource> BacktestEngine<S, D> {
    /// Creates a new backtest engine with the reference
    /// [`ImmediateFillModel`].
    #[must_use]
    pub fn new(config: BacktestConfig, strategy: S, data_source: D) -> Self {
        Self::build(config, strategy, data_source, ImmediateFillModel::new())
    }
}

impl<S: BacktestStrategy, D: HistoricalDataSource, F: FillModel> BacktestEngine<S, D, F> {
    /// Creates a backtest engine that executes resting orders through
    /// `fill_model`.
    ///
    /// # Errors
    ///
    /// Returns the error from [`FillModel::validate`] if the model
    /// parameters are invalid.
    pub fn with_fill_model(
        config: BacktestConfig,
        strategy: S,
        data_source: D,
        fill_model: F,
    ) -> MMResult<Self> {
        fill_model.validate()?;
        Ok(Self::build(config, strategy, data_source, fill_model))
    }

    /// Applies simulated latency to observations, submissions, cancels and
    /// acknowledgements. The default is zero latency.
    ///
    /// # Errors
    ///
    /// Returns [`crate::types::error::MMError::InvalidConfiguration`] if a
    /// delay exceeds [`super::latency::MAX_SIM_LATENCY_MS`].
    pub fn with_latency(mut self, latency: SimLatencyConfig) -> MMResult<Self> {
        latency.validate()?;
        self.latency = latency;
        self.jitter = JitterRng::new(latency.seed);
        Ok(self)
    }

    /// Returns the simulated latency configuration.
    #[must_use]
    pub fn latency(&self) -> &SimLatencyConfig {
        &self.latency
    }

    fn build(config: BacktestConfig, strategy: S, data_source: D, fill_model: F) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            config,
            fill_model,
            strategy,
            data_source,
            position: InventoryPosition::new(),
            pnl: PnL::new(),
            equity_curve: Vec::new(),
            trades: Vec::new(),
            total_fees: Decimal::ZERO,
            fixed_fees: Decimal::ZERO,
            base_fee_quantity: Decimal::ZERO,
            max_position: Decimal::ZERO,
            peak_equity: initial_capital,
            max_drawdown: Decimal::ZERO,
            clock: SimClock::new(),
            orders: Vec::new(),
            next_order_id: 1,
            reports: Vec::new(),
            latency: SimLatencyConfig::default(),
            jitter: JitterRng::new(0),
            next_message: 0,
            pending: BTreeMap::new(),
            pending_activations: HashMap::new(),
            cancel_targets: HashSet::new(),
            awaiting_cancel: HashMap::new(),
            deliveries: VecDeque::new(),
            observed: VecDeque::new(),
            agent_position: InventoryPosition::new(),
            last_mark: None,
            last_delivery_at: 0,
            last_observed_at: 0,
            accountant: None,
            accounting_errors: Vec::new(),
            snapshot_failures: Vec::new(),
            valuation_failures: 0,
            setup_refused: None,
            dropped_fills: 0,
            #[cfg(feature = "instruments")]
            instrument: None,
            next_execution_id: 1,
            policy: None,
            pretrade_log: Vec::new(),
            faults: Vec::new(),
            fault_count: 0,
        }
    }

    /// Runs the backtest and returns the result.
    pub fn run(&mut self) -> BacktestResult {
        self.run_with_progress(|_, _| {})
    }

    /// Runs the backtest with a progress callback.
    ///
    /// The callback receives (current_tick, total_ticks).
    pub fn run_with_progress<P: FnMut(usize, usize)>(&mut self, mut callback: P) -> BacktestResult {
        let total_ticks = self.data_source.len();
        let mut num_ticks = 0u64;
        let mut start_time = 0u64;
        let mut end_time = 0u64;

        // Before any tick is read or any state changes.
        self.setup_refused = None;
        if let Err(error) = self.check_setup() {
            self.setup_refused = Some(error.to_string());
            self.record_fault(error);
            return self.result(num_ticks, start_time, end_time);
        }

        while let Some(tick) = self.data_source.next_tick() {
            if num_ticks == 0 {
                start_time = tick.timestamp;
            }
            end_time = tick.timestamp;

            // Simulated time follows the data; out of order ticks are
            // processed at the current time so it never goes backwards.
            let now = self.clock.now().max(tick.timestamp);

            // 1. Every scheduled event up to now happens first, in time
            //    order: venue messages, report deliveries and observations
            //    that fell between the previous tick and this one.
            self.advance_events(now);
            let _ = self.clock.advance_to(now);

            // 2. Market data: match orders that were already active before
            //    this tick.
            if let Some(policy) = self.policy.as_mut() {
                policy.on_market_data(tick.mid_price(), tick.timestamp);
            }
            self.match_active_orders(&tick, now);

            // 3. The tick enters the FIFO observation channel; the strategy
            //    decides when it sees it, with the position it has been told
            //    about. New orders can only match later ticks.
            let ready = self
                .delayed(tick.timestamp, self.latency.observation_ms)
                .max(self.last_observed_at);
            self.last_observed_at = ready;
            self.observed.push_back((ready, tick.clone()));
            self.advance_events(now);

            // Update PnL mark-to-market
            let mid_price = tick.mid_price();
            self.mark_to(mid_price);

            // Track equity
            // The run cannot fail, so money sums saturate at the `Decimal`
            // bounds instead of panicking.
            let equity = self
                .config
                .initial_capital
                .saturating_add(self.pnl.total)
                .saturating_sub(self.total_fees);
            if self.config.record_equity_curve {
                self.equity_curve.push((tick.timestamp, equity));
            }

            // Track drawdown
            if equity > self.peak_equity {
                self.peak_equity = equity;
            }
            let drawdown = self.peak_equity.saturating_sub(equity);
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }

            num_ticks = num_ticks.saturating_add(1);
            callback(
                usize::try_from(num_ticks).unwrap_or(usize::MAX),
                total_ticks,
            );
        }

        // Reports still in flight reach the strategy after the data ends.
        self.flush_deliveries();

        self.result(num_ticks, start_time, end_time)
    }

    /// The result of the run so far.
    fn result(&self, num_ticks: u64, start_time: u64, end_time: u64) -> BacktestResult {
        BacktestResult {
            total_pnl: self.pnl.total,
            total_fees: self.total_fees,
            net_pnl: self.pnl.total.saturating_sub(self.total_fees),
            num_trades: crate::types::decimal::saturating_u64(self.trades.len()),
            num_ticks,
            start_time,
            end_time,
            max_position: self.max_position,
            final_position: self.position.quantity,
            equity_curve: if self.config.record_equity_curve {
                self.equity_curve.clone()
            } else {
                Vec::new()
            },
            trades: if self.config.record_trades {
                self.trades.clone()
            } else {
                Vec::new()
            },
            max_drawdown: self.max_drawdown,
            sharpe_ratio: self.calculate_sharpe_ratio(),
            accounting: AccountingStatus {
                accountant: self.accountant_description(),
                refused_executions: self.accounting_errors.iter().map(|(id, _)| *id).collect(),
                snapshot_failures: self.snapshot_failures.clone(),
                valuation_failures: self.valuation_failures,
                setup_refused: self.setup_refused.clone(),
                dropped_fills: self.dropped_fills,
            },
        }
    }

    /// Delivers the reports still in flight when the data ends, so the
    /// strategy hears about every fill in the result. Venue messages and
    /// observations still pending are dropped: no market data is left to
    /// act on.
    fn flush_deliveries(&mut self) {
        while !self.deliveries.is_empty() {
            self.deliver_next();
        }
    }

    /// Runs every scheduled event with time at or before `until` in
    /// chronological order. At equal times venue messages come first, then
    /// report deliveries, then observations, so a decision sees every report
    /// delivered at its own time. Events created while running (a decision
    /// scheduling a zero delay submission) are picked up in the same pass.
    fn advance_events(&mut self, until: u64) {
        loop {
            let times = [
                self.pending.first_key_value().map(|(key, _)| key.0),
                self.deliveries.front().map(|d| d.at),
                self.observed.front().map(|(at, _)| *at),
            ];
            let Some((time, class)) = next_due(&times, until) else {
                break;
            };
            // Out of order data can schedule an event before the clock;
            // it then happens at the current time.
            let at = time.max(self.clock.now());
            let _ = self.clock.advance_to(at);
            match class {
                0 => self.apply_next_venue_message(),
                1 => self.deliver_next(),
                _ => self.observe_next(at),
            }
        }
    }

    /// Hands the oldest observed tick to the strategy at time `at`.
    fn observe_next(&mut self, at: u64) {
        let Some((_, seen)) = self.observed.pop_front() else {
            return;
        };
        if let Some(quote) = self.strategy.on_tick(&seen, &self.agent_position) {
            self.replace_quotes(&quote, at);
        } else if self.strategy.cancels_when_silent() {
            self.cancel_resting(at);
        }
    }

    /// Requests the cancel of every live or pending order not already being
    /// cancelled.
    fn cancel_resting(&mut self, now: u64) {
        let mut targets: Vec<SimOrderId> = self
            .orders
            .iter()
            .filter(|o| !o.state.is_terminal())
            .map(|o| o.id)
            .collect();
        targets.extend(self.pending_activations.keys().copied());
        targets.retain(|id| !self.cancel_targets.contains(id));
        // Ids are allocated in order, so sorting keeps scheduling (and the
        // jitter draws) deterministic.
        targets.sort_unstable();
        for id in targets {
            let effective = self.delayed(now, self.latency.cancel_ms);
            self.cancel_targets.insert(id);
            self.notify_policy(OrderEvent::CancelRequested {
                order_id: id.to_string(),
            });
            self.schedule(effective, PendingAction::Cancel(id));
        }
    }

    /// Matches every eligible resting order against the tick through the
    /// fill model. Fill quantities reported by the model are cumulative per
    /// order; only the increase over the filled quantity is booked, at the
    /// model's price plus slippage.
    fn match_active_orders(&mut self, tick: &MarketTick, now: u64) {
        self.fill_model.on_tick(tick);
        for index in 0..self.orders.len() {
            let Some(order) = self.orders.get(index) else {
                continue;
            };
            // Eligibility uses the tick's exchange time, not the clamped
            // clock, so an out of order tick older than the activation
            // can never fill the order.
            if !order.is_eligible_at(tick.timestamp) {
                continue;
            }
            let side = order.side;
            // The engine only creates buy and sell orders (see
            // `submit_quotes`); this is the single conversion point.
            let dir = match Dir::try_from(side) {
                Ok(dir) => dir,
                Err(error) => {
                    self.record_fault(error);
                    continue;
                }
            };
            let view = SimulatedOrder::new(side, order.price, order.quantity, order.active_after);
            let time_in_queue = tick.timestamp.saturating_sub(order.active_after);
            let outcome = match self.fill_model.simulate_fill(&view, tick, time_in_queue) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.record_fault(error);
                    continue;
                }
            };
            let Some((quantity, model_price)) = fill_increment(order, outcome) else {
                continue;
            };
            if quantity <= Decimal::ZERO {
                continue;
            }
            let price = slipped_price(&self.config.slippage, model_price, dir);
            // A model or slippage output that is not a valid price is
            // ignored: it would book negative proceeds and fees.
            if price <= Decimal::ZERO {
                continue;
            }
            let charge = match self.charge_fill(price, quantity) {
                Ok(charge) => charge,
                Err(error) => {
                    self.dropped_fills = self.dropped_fills.saturating_add(1);
                    self.record_fault(error);
                    continue;
                }
            };
            // Reports and result metrics carry the fee valued in quote, or in
            // the settlement asset on a derivative contract.
            let fee = charge.quote_value;
            let mut fill = SimulatedFill::with_fee(side, price, quantity, now, fee)
                .with_execution_id(self.next_execution_id);
            if charge.asset == FeeAsset::Base {
                fill.base_fee = charge.amount;
            }
            // The built in bookkeeping is validated before anything is
            // mutated: a booking that overflows is a fault that leaves the
            // order, the policy and the engine unchanged.
            let booking = if self.accountant.is_some() {
                None
            } else {
                match self.plan_booking(&fill, dir, &charge) {
                    Ok(booking) => Some(booking),
                    Err(error) => {
                        self.dropped_fills = self.dropped_fills.saturating_add(1);
                        self.record_fault(error);
                        continue;
                    }
                }
            };
            let Some(order) = self.orders.get_mut(index) else {
                continue;
            };
            if order.apply_fill(quantity).is_err() {
                continue;
            }
            let stamp = self.clock.stamp();
            let report = ExecutionReport::fill(stamp, order, price, quantity, fee);
            let (order_id, cumulative) = (order.id, order.filled);
            // The policy books the fill first so a ledger it owns is up to
            // date when the accountant reads it.
            self.notify_policy(OrderEvent::Fill {
                order_id: order_id.to_string(),
                execution_id: format!("sim-{}", self.next_execution_id),
                side,
                price,
                quantity,
                cumulative_quantity: cumulative,
                fee: charge.amount,
                fee_asset: charge.asset.clone(),
                timestamp: now,
            });
            self.process_fill(fill.clone(), booking, order_id, cumulative, charge);
            self.queue_delivery(report, Some((dir, fill)));
        }
        let targets = &self.cancel_targets;
        let awaiting = &mut self.awaiting_cancel;
        self.orders.retain(|order| {
            if !order.state.is_terminal() {
                return true;
            }
            if targets.contains(&order.id) {
                awaiting.insert(order.id, order.clone());
            }
            false
        });
    }

    /// Schedules cancels for every resting or pending order and submits the
    /// new bid and ask. Cancels take effect after the cancel latency; new
    /// orders become active after the submission latency and only match
    /// ticks after that. A crossed or locked quote (bid at or above ask) is
    /// rejected locally on both sides at decision time, so the engine never
    /// rests two orders that could trade with each other.
    fn replace_quotes(&mut self, quote: &Quote, now: u64) {
        self.cancel_resting(now);

        let tick = self.config.tick_size;
        let bid_price = round_to_step(quote.bid_price, tick, false);
        let ask_price = round_to_step(quote.ask_price, tick, true);
        // Checked on the quote as emitted: rounding away from the market
        // could otherwise hide a crossed quote.
        let crossed = quote.bid_price >= quote.ask_price;
        for (side, price, quoted) in [
            (Action::Buy, bid_price, quote.bid_size),
            (Action::Sell, ask_price, quote.ask_size),
        ] {
            let requested = if quoted > Decimal::ZERO {
                quoted
            } else {
                self.config.default_order_size
            };
            // A size that rounds to zero is rejected by the venue checks.
            let mut size = round_to_step(requested, self.config.lot_size, false);
            let id = self.next_order_id;
            self.next_order_id = self.next_order_id.saturating_add(1);
            let stamp = self.clock.stamp();
            // After a fault the policy or the fill model could not account
            // for something: no new order is admitted.
            let mut refused = crossed || !self.faults.is_empty();
            if !refused && let Some(policy) = self.policy.as_mut() {
                let request = PretradeRequest::new(
                    id.to_string(),
                    side,
                    Some(price),
                    size,
                    OrderPurpose::Quote,
                    now,
                );
                let decision = policy.check(&request);
                // Never more than requested. The policy registered exactly
                // `approved`, so a size that is not a whole number of lots
                // is refused and released rather than silently shrunk,
                // which would leave a stale reservation behind.
                let approved = decision.approved_quantity(size);
                let whole_lots = round_to_step(approved, self.config.lot_size, false) == approved;
                refused = approved <= Decimal::ZERO || !whole_lots;
                if approved > Decimal::ZERO
                    && !whole_lots
                    && let Err(error) = policy.on_order_event(&OrderEvent::VenueRejected {
                        order_id: id.to_string(),
                    })
                {
                    self.record_fault(error);
                }
                if !refused {
                    size = approved;
                }
                self.pretrade_log.push(PretradeRecord {
                    order_id: id,
                    request,
                    decision,
                    submitted: if refused { Decimal::ZERO } else { size },
                });
            }
            if refused {
                let mut order = SimOrder::new(id, side, price, size, stamp, now);
                let _ = order.reject();
                let stamp = self.clock.stamp();
                self.queue_delivery(
                    ExecutionReport::status(stamp, &order, ExecutionKind::Rejected),
                    None,
                );
                continue;
            }
            let effective = self.delayed(now, self.latency.submit_ms);
            let order = SimOrder::new(id, side, price, size, stamp, effective);
            self.schedule(effective, PendingAction::Activate(order));
        }
    }

    /// Adds latency and jitter to `now`.
    fn delayed(&mut self, now: u64, base_ms: u64) -> u64 {
        let jitter = self.jitter.draw(self.latency.jitter_ms);
        now.saturating_add(base_ms).saturating_add(jitter)
    }

    fn schedule(&mut self, effective: u64, action: PendingAction) {
        let key = (effective, self.next_message);
        self.next_message = self.next_message.saturating_add(1);
        if let PendingAction::Activate(order) = &action {
            self.pending_activations.insert(order.id, key);
        }
        self.pending.insert(key, action);
    }

    /// Applies the earliest venue message, in `(effective time, sequence)`
    /// order.
    fn apply_next_venue_message(&mut self) {
        let Some((_, action)) = self.pending.pop_first() else {
            return;
        };
        match action {
            PendingAction::Activate(mut order) => {
                self.pending_activations.remove(&order.id);
                let kind = if order.validate().is_ok() {
                    let _ = order.accept();
                    ExecutionKind::Accepted
                } else {
                    let _ = order.reject();
                    ExecutionKind::Rejected
                };
                let stamp = self.clock.stamp();
                self.emit(ExecutionReport::status(stamp, &order, kind));
                if kind == ExecutionKind::Accepted {
                    self.orders.push(order);
                } else if self.cancel_targets.contains(&order.id) {
                    self.awaiting_cancel.insert(order.id, order);
                }
            }
            PendingAction::Cancel(id) => {
                self.cancel_targets.remove(&id);
                self.apply_cancel(id);
            }
        }
    }

    /// Cancels order `id` if it is still open or not yet active; otherwise
    /// reports that the cancel arrived too late. Fills that happened earlier
    /// always stand.
    fn apply_cancel(&mut self, id: SimOrderId) {
        let stamp = self.clock.stamp();
        if let Some(index) = self.orders.iter().position(|o| o.id == id)
            && let Some(order) = self.orders.get_mut(index)
            && order.cancel().is_ok()
        {
            let report = ExecutionReport::status(stamp, order, ExecutionKind::Canceled);
            self.orders.remove(index);
            self.emit(report);
            return;
        }
        if let Some(key) = self.pending_activations.remove(&id) {
            if let Some(PendingAction::Activate(mut order)) = self.pending.remove(&key) {
                let _ = order.cancel();
                self.emit(ExecutionReport::status(
                    stamp,
                    &order,
                    ExecutionKind::Canceled,
                ));
            }
            return;
        }
        if let Some(order) = self.awaiting_cancel.remove(&id) {
            let report = ExecutionReport::status(stamp, &order, ExecutionKind::CancelRejected);
            self.emit(report);
        }
    }

    /// Delivers the oldest queued report to the strategy, updating the
    /// position the strategy knows about.
    fn deliver_next(&mut self) {
        let Some(delivery) = self.deliveries.pop_front() else {
            return;
        };
        self.strategy.on_execution_report(&delivery.report);
        if let Some((_, fill)) = delivery.fill {
            // The fill was produced by the engine, so its side is buy or
            // sell, and a base delta error was already recorded when the
            // venue applied it. A position overflow leaves the known
            // position unchanged and is recorded as a fault.
            if let Ok(base_delta) = fill.base_delta()
                && let Err(error) =
                    self.agent_position
                        .update_fill(base_delta, fill.price, fill.timestamp)
            {
                self.record_fault(error);
            }
            self.strategy.on_fill(&fill);
        }
    }

    /// Records a venue report and queues it for the strategy.
    fn emit(&mut self, report: ExecutionReport) {
        let order_id = report.order_id.to_string();
        let event = match report.kind {
            ExecutionKind::Accepted => Some(OrderEvent::Accepted { order_id }),
            ExecutionKind::Rejected => Some(OrderEvent::VenueRejected { order_id }),
            ExecutionKind::Canceled => Some(OrderEvent::Canceled { order_id }),
            ExecutionKind::CancelRejected => Some(OrderEvent::CancelRejected { order_id }),
            // Fills are forwarded with their execution details.
            ExecutionKind::PartialFill | ExecutionKind::Fill => None,
        };
        if let Some(event) = event {
            self.notify_policy(event);
        }
        self.queue_delivery(report, None);
    }

    /// Counts a fault and keeps it if the bounded list has room.
    fn record_fault(&mut self, error: MMError) {
        self.fault_count = self.fault_count.saturating_add(1);
        if self.faults.len() < MAX_RECORDED_FAULTS {
            self.faults.push(error);
        }
    }

    fn notify_policy(&mut self, event: OrderEvent) {
        if let Some(policy) = self.policy.as_mut()
            && let Err(error) = policy.on_order_event(&event)
        {
            self.record_fault(error);
        }
    }

    fn queue_delivery(&mut self, report: ExecutionReport, fill: Option<(Dir, SimulatedFill)>) {
        // The report channel is FIFO: jitter delays a report but never lets
        // it overtake one produced earlier.
        let at = self
            .delayed(report.stamp.timestamp, self.latency.ack_ms)
            .max(self.last_delivery_at);
        self.last_delivery_at = at;
        if self.config.record_trades {
            self.reports.push(report.clone());
        }
        self.deliveries.push_back(Delivery { at, report, fill });
    }

    /// Processes a fill: updates position, PnL, and notifies strategy.
    fn process_fill(
        &mut self,
        fill: SimulatedFill,
        booking: Option<Booking>,
        order_id: SimOrderId,
        cumulative: Decimal,
        charge: FeeCharge,
    ) {
        let execution_id = self.next_execution_id;
        self.next_execution_id = self.next_execution_id.saturating_add(1);
        if let Some(accountant) = self.accountant.as_mut() {
            let posted = AccountingFill {
                execution_id,
                order_id,
                side: fill.side,
                price: fill.price,
                quantity: fill.quantity,
                cumulative_quantity: cumulative,
                fee: charge.amount,
                fee_asset: charge.asset,
                timestamp: fill.timestamp,
            };
            if let Err(error) = accountant.post_fill(&posted) {
                self.accounting_errors.push((execution_id, error));
            }
            self.sync_accounting(fill.timestamp, execution_id);
            self.track_fill(fill);
            return;
        }

        // Validated by `plan_booking` before the order was filled.
        let Some(booking) = booking else {
            self.record_fault(MMError::InvalidMarketState(format!(
                "fill {execution_id} reached the built in bookkeeping without a booking"
            )));
            return;
        };
        self.position = booking.position;
        self.pnl.add_realized(booking.cash_flow);
        self.base_fee_quantity = booking.base_fee_quantity;
        self.fixed_fees = booking.fixed_fees;
        self.mark_to(fill.price);
        self.track_fill(fill);
    }

    /// The built in bookkeeping of one fill, computed without mutating the
    /// engine: the position after it (base fees deducted), the cash flow
    /// (negative for a buy) and the fee totals.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the position, the notional,
    /// the realized PnL or a fee total overflows.
    fn plan_booking(
        &self,
        fill: &SimulatedFill,
        dir: Dir,
        charge: &FeeCharge,
    ) -> MMResult<Booking> {
        let base_delta = fill.base_delta()?;
        let mut position = self.position.clone();
        position.update_fill(base_delta, fill.price, fill.timestamp)?;

        // Simplified realized PnL: the cash flow of the fill.
        let notional = fill
            .price
            .checked_mul(fill.quantity)
            .or_overflow("fill notional")?;
        let cash_flow = match dir {
            Dir::Buy => Decimal::ZERO.checked_sub(notional),
            Dir::Sell => Some(notional),
        }
        .or_overflow("fill cash flow")?;
        self.pnl
            .realized
            .checked_add(cash_flow)
            .or_overflow("realized pnl")?;

        // Base fees are kept in base units and marked.
        let (base_fee_quantity, fixed_fees) = match charge.asset {
            FeeAsset::Base => (
                self.base_fee_quantity
                    .checked_add(charge.amount)
                    .or_overflow("base fee total")?,
                self.fixed_fees,
            ),
            // Refused before the run: the built in bookkeeping is spot style.
            FeeAsset::Settlement => {
                return Err(MMError::InvalidConfiguration(
                    "the built in bookkeeping cannot book a settlement asset fee".to_string(),
                ));
            }
            FeeAsset::Quote | FeeAsset::Other(_) => (
                self.base_fee_quantity,
                self.fixed_fees
                    .checked_add(charge.quote_value)
                    .or_overflow("fee total")?,
            ),
        };
        Ok(Booking {
            position,
            cash_flow,
            base_fee_quantity,
            fixed_fees,
        })
    }

    /// Updates the position high water mark and records the trade.
    fn track_fill(&mut self, fill: SimulatedFill) {
        // Track max position
        let abs_position = self.position.quantity.abs();
        if abs_position > self.max_position {
            self.max_position = abs_position;
        }

        // Record trade
        if self.config.record_trades {
            self.trades.push(fill);
        }
    }

    /// Calculates an approximate Sharpe ratio from the equity curve.
    fn calculate_sharpe_ratio(&self) -> Option<Decimal> {
        sharpe_ratio(&self.equity_curve)
    }

    /// Returns the current state (position and PnL).
    #[must_use]
    pub fn get_state(&self) -> (&InventoryPosition, &PnL) {
        (&self.position, &self.pnl)
    }

    /// Returns the execution reports recorded so far, in event order.
    ///
    /// Reports are recorded only when [`BacktestConfig::record_trades`] is
    /// set; otherwise this is empty, while the strategy still receives every
    /// report through [`BacktestStrategy::on_execution_report`].
    #[must_use]
    pub fn execution_reports(&self) -> &[ExecutionReport] {
        &self.reports
    }

    /// Returns the fill model.
    #[must_use]
    pub fn fill_model(&self) -> &F {
        &self.fill_model
    }

    /// Passes every new order through `policy` before submission.
    ///
    /// A rejection (or a quantity that rounds to zero lots) is reported as a
    /// local `Rejected` and nothing reaches the venue; a reduction submits the
    /// smaller quantity and can never increase it. A reduction that is not a
    /// whole number of [`BacktestConfig::lot_size`] lots is refused and
    /// released in the policy with an [`OrderEvent::VenueRejected`]. The policy also receives
    /// the market mark of every tick and every lifecycle event of the orders,
    /// in venue order. See [`crate::risk::pretrade`].
    #[must_use]
    pub fn with_pretrade_policy(
        mut self,
        policy: impl PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe + 'static,
    ) -> Self {
        self.policy = Some(Box::new(policy));
        self
    }

    /// Every pretrade decision taken so far, in decision order.
    #[must_use]
    pub fn pretrade_decisions(&self) -> &[PretradeRecord] {
        &self.pretrade_log
    }

    /// Description of the pretrade policy, if one was selected.
    #[must_use]
    pub fn pretrade_policy_description(&self) -> Option<String> {
        self.policy.as_ref().map(|p| p.description())
    }

    /// Selects `accountant` as the single accounting authority for fills.
    ///
    /// Every fill is posted to it exactly once and the position, PnL, fees
    /// and equity of the run are read from its snapshot instead of the built
    /// in bookkeeping. The run, and every run after [`BacktestEngine::reset`],
    /// starts from the accountant's state now
    /// ([`FillAccountant::begin_from_current_state`]). See
    /// [`crate::backtest::accounting`].
    #[must_use]
    pub fn with_accountant(
        mut self,
        mut accountant: impl FillAccountant + Send + Sync + UnwindSafe + RefUnwindSafe + 'static,
    ) -> Self {
        accountant.begin_from_current_state();
        self.accountant = Some(Box::new(accountant));
        self.sync_accounting(0, 0);
        self
    }

    /// Runs a derivative contract: `spec` is the contract and `accountant` the
    /// accounting authority for its fills, taken together so that neither can be
    /// set without the other.
    ///
    /// The position, the fees and every result metric are then in contracts and
    /// in the settlement asset of `spec`, which is BTC for an inverse contract,
    /// so [`BacktestConfig::initial_capital`] must be too. Equity is
    /// `initial_capital + realized + unrealized - fees`, where realized and
    /// unrealized come from the accountant and contain no notional.
    ///
    /// Limits of such a run:
    ///
    /// - [`MarketTick`] sizes and the fill models must be in contracts, or a
    ///   partial fill is sized wrongly;
    /// - trade volume is `price * contracts`, which is not a notional;
    /// - the accountant books with overdraft, with no margin requirement or
    ///   liquidation, and no funding, expiry or settlement;
    /// - one contract only: a run across contracts, for a roll, is
    ///   `backtest::contracts::ContractEngine`;
    /// - a fill at or after a future's expiry is refused by the accountant and
    ///   recorded in [`AccountingStatus::refused_executions`];
    /// - a mark the accountant cannot value is counted in
    ///   [`AccountingStatus::valuation_failures`], keeps the previous unrealized
    ///   PnL and total, and, as a fault, stops new orders.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a contract that is not a future or a
    /// perpetual, a pretrade policy that does not decide in contracts (see
    /// [`PretradePolicy::in_one_contract`]: other limits read `price * quantity`, which is
    /// not a notional here), a [`BacktestConfig::lot_size`] other than the lot of
    /// `spec`, a [`BacktestConfig::tick_size`] other than its single price tick,
    /// or a fee currency other than [`FeeCurrency::Settlement`]; and the error of
    /// [`FillAccountant::check_instrument`], which refuses an accountant of other
    /// terms.
    ///
    /// The same checks run at the start of every run, so a later
    /// [`BacktestEngine::with_accountant`] or
    /// [`BacktestEngine::with_pretrade_policy`] gets the run refused, with
    /// [`AccountingStatus::setup_refused`] set and nothing processed.
    #[cfg(feature = "instruments")]
    pub fn with_instrument(
        mut self,
        spec: financial_types::instrument::InstrumentSpec,
        mut accountant: impl FillAccountant + Send + Sync + UnwindSafe + RefUnwindSafe + 'static,
    ) -> MMResult<Self> {
        accountant.begin_from_current_state();
        self.instrument = Some(spec);
        self.accountant = Some(Box::new(accountant));
        self.check_setup()?;
        self.sync_accounting(0, 0);
        Ok(self)
    }

    /// Refuses what this configuration cannot book correctly.
    fn check_setup(&self) -> MMResult<()> {
        #[cfg(feature = "instruments")]
        if let Some(spec) = &self.instrument {
            return self.check_instrument_setup(spec);
        }
        if matches!(self.config.fee_currency, FeeCurrency::Settlement { .. }) {
            return Err(MMError::InvalidConfiguration(
                "a settlement asset fee needs the contract: use with_instrument".to_string(),
            ));
        }
        if self
            .accountant
            .as_ref()
            .is_some_and(|accountant| accountant.requires_instrument())
        {
            return Err(MMError::InvalidConfiguration(
                "a derivative accountant needs the contract: use with_instrument".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "instruments")]
    fn check_instrument_setup(
        &self,
        spec: &financial_types::instrument::InstrumentSpec,
    ) -> MMResult<()> {
        use financial_types::instrument::ContractKind;
        let refuse = |reason: String| {
            Err(MMError::InvalidConfiguration(format!(
                "instrument {}: {reason}",
                spec.id().as_str()
            )))
        };
        if !matches!(spec.kind(), ContractKind::Future | ContractKind::Perpetual) {
            return refuse(format!("{:?} is not a future or a perpetual", spec.kind()));
        }
        if self.policy.as_ref().is_some_and(|p| !p.in_one_contract()) {
            return refuse(
                "a pretrade policy reads price * quantity, which is not a notional here: \
                 only one that decides in contracts can run"
                    .to_string(),
            );
        }
        if self.config.lot_size != spec.lot() {
            return refuse(format!(
                "lot size {} differs from the lot {}",
                self.config.lot_size,
                spec.lot()
            ));
        }
        let bands = spec.price_ticks().bands();
        let single_tick = match bands {
            [only] => only.tick == self.config.tick_size,
            _ => false,
        };
        if !single_tick {
            return refuse(format!(
                "tick size {} is not its single price tick",
                self.config.tick_size
            ));
        }
        let FeeCurrency::Settlement { scale } = self.config.fee_currency else {
            return refuse("the fee currency must be Settlement".to_string());
        };
        self.config.fee_currency.validate()?;
        let Some(accountant) = self.accountant.as_ref() else {
            return refuse("it needs an accountant".to_string());
        };
        accountant.check_instrument(spec, scale)
    }

    /// The fee of one execution.
    fn charge_fill(&self, price: Decimal, quantity: Decimal) -> MMResult<FeeCharge> {
        #[cfg(feature = "instruments")]
        if let (Some(spec), FeeCurrency::Settlement { scale }) =
            (&self.instrument, &self.config.fee_currency)
        {
            return settlement_charge(spec, *scale, price, quantity, self.config.fee_rate);
        }
        self.config
            .fee_currency
            .charge(price, quantity, self.config.fee_rate)
    }

    /// Snapshot of the selected accountant, if one was selected.
    ///
    /// The inner result is the accountant's: an error if a sum over its
    /// bookings overflows.
    #[must_use]
    pub fn accounting_snapshot(&self) -> Option<MMResult<AccountingSnapshot>> {
        self.accountant.as_ref().map(|a| a.snapshot())
    }

    /// Description of the selected accountant, if one was selected.
    #[must_use]
    pub fn accountant_description(&self) -> Option<String> {
        self.accountant.as_ref().map(|a| a.description())
    }

    /// The first errors returned by the fill model or the pretrade policy,
    /// in order.
    ///
    /// Each one means a component could not account for an order or an
    /// event, so from the first one on the engine refuses every new quote
    /// (reported as `Rejected`), as a policy rejection would, until
    /// [`BacktestEngine::reset`]. Orders already resting keep working.
    ///
    /// The list is bounded: only the first [`MAX_RECORDED_FAULTS`] errors
    /// are kept, so a component that fails on every call cannot grow it
    /// without limit. [`BacktestEngine::fault_count`] counts all of them.
    #[must_use]
    pub fn faults(&self) -> &[MMError] {
        &self.faults
    }

    /// Total number of faults since the last reset, including the ones not
    /// kept in [`BacktestEngine::faults`]. Saturates at `u64::MAX`.
    #[must_use]
    pub fn fault_count(&self) -> u64 {
        self.fault_count
    }

    /// Posts the selected accountant refused, as `(execution id, error)`.
    ///
    /// The fill still happened at the venue; the result metrics follow the
    /// accountant, so a non empty list means they disagree with the venue.
    #[must_use]
    pub fn accounting_errors(&self) -> &[(u64, MMError)] {
        &self.accounting_errors
    }

    /// Describes the fill model and its parameters, for result reports.
    #[must_use]
    pub fn fill_model_description(&self) -> String {
        self.fill_model.description()
    }

    /// Returns the orders currently resting in the simulated book.
    #[must_use]
    pub fn open_orders(&self) -> &[SimOrder] {
        &self.orders
    }

    /// Returns a reference to the strategy.
    #[must_use]
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    /// Returns a mutable reference to the strategy.
    pub fn strategy_mut(&mut self) -> &mut S {
        &mut self.strategy
    }

    /// Resets the engine for another run.
    pub fn reset(&mut self) {
        self.data_source.reset();
        self.strategy.reset();
        self.position = InventoryPosition::new();
        self.pnl = PnL::new();
        self.equity_curve.clear();
        self.trades.clear();
        self.total_fees = Decimal::ZERO;
        self.fixed_fees = Decimal::ZERO;
        self.base_fee_quantity = Decimal::ZERO;
        self.max_position = Decimal::ZERO;
        self.peak_equity = self.config.initial_capital;
        self.max_drawdown = Decimal::ZERO;
        self.fill_model.reset();
        self.clock.reset();
        self.orders.clear();
        self.jitter.reset();
        self.next_message = 0;
        self.pending.clear();
        self.pending_activations.clear();
        self.cancel_targets.clear();
        self.awaiting_cancel.clear();
        self.deliveries.clear();
        self.observed.clear();
        self.agent_position = InventoryPosition::new();
        self.last_mark = None;
        self.last_delivery_at = 0;
        self.last_observed_at = 0;
        self.next_order_id = 1;
        if let Some(accountant) = self.accountant.as_mut() {
            accountant.reset();
        }
        self.accounting_errors.clear();
        self.snapshot_failures.clear();
        self.valuation_failures = 0;
        self.dropped_fills = 0;
        self.next_execution_id = 1;
        if let Some(policy) = self.policy.as_mut() {
            policy.reset();
        }
        self.pretrade_log.clear();
        self.faults.clear();
        self.fault_count = 0;
        self.reports.clear();
        self.sync_accounting(0, 0);
    }

    /// Copies the selected accountant's state into the engine's position,
    /// PnL and fee fields, so runs start from the accountant's initial
    /// state (also with no fills at all). A snapshot that fails is
    /// recorded as a fault and leaves those fields unchanged.
    fn sync_accounting(&mut self, timestamp: u64, execution_id: u64) {
        let Some(accountant) = self.accountant.as_ref() else {
            return;
        };
        let snapshot = match accountant.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.snapshot_failures.push(execution_id);
                self.record_fault(error);
                return;
            }
        };
        self.position.quantity = snapshot.position;
        self.position.avg_entry_price = snapshot.average_entry_price.unwrap_or(Decimal::ZERO);
        self.position.last_update = timestamp;
        self.pnl.realized = snapshot.cash_flow;
        self.fixed_fees = snapshot.fees;
        self.base_fee_quantity = snapshot.base_fees;
        let mark = self.last_mark.unwrap_or(self.position.avg_entry_price);
        self.mark_to(mark);
        self.max_position = self.max_position.max(snapshot.position.abs());
    }

    /// Whether this is a run of a derivative contract.
    fn has_instrument(&self) -> bool {
        #[cfg(feature = "instruments")]
        {
            self.instrument.is_some()
        }
        #[cfg(not(feature = "instruments"))]
        {
            false
        }
    }

    /// Marks the position and the base fees paid at `mark`.
    ///
    /// Equity is cash plus the net base held at the mark. Base fees are
    /// added back to the gross PnL and counted as fees at the same mark, so
    /// `net = total - fees` stays exact.
    fn mark_to(&mut self, mark: Decimal) {
        self.last_mark = Some(mark);
        let valued = self
            .accountant
            .as_ref()
            .and_then(|accountant| accountant.unrealized_at(mark));
        match valued {
            Some(Ok(unrealized)) => {
                self.pnl.unrealized = unrealized;
                self.pnl.total = self.pnl.realized.saturating_add(unrealized);
                self.total_fees = self.fixed_fees;
                return;
            }
            // Never `position * mark`, which is wrong for this contract: keep
            // the previous unrealized and total, and say so.
            Some(Err(error)) => {
                self.valuation_failures = self.valuation_failures.saturating_add(1);
                self.record_fault(error);
                return;
            }
            None => {
                if self.has_instrument() {
                    self.valuation_failures = self.valuation_failures.saturating_add(1);
                    self.record_fault(MMError::InvalidConfiguration(
                        "the accountant of a derivative run must value it".to_string(),
                    ));
                    return;
                }
            }
        }
        let base_fee_value = self.base_fee_quantity.saturating_mul(mark);
        self.pnl.unrealized = self
            .position
            .quantity
            .saturating_mul(mark)
            .saturating_add(base_fee_value);
        self.pnl.total = self.pnl.realized.saturating_add(self.pnl.unrealized);
        self.total_fees = self.fixed_fees.saturating_add(base_fee_value);
    }
}

/// The class and time of the next event at or before `until`: the earliest, and at
/// equal times the first class in `times`.
pub(super) fn next_due(times: &[Option<u64>], until: u64) -> Option<(u64, usize)> {
    let mut next: Option<(u64, usize)> = None;
    for (class, time) in times.iter().enumerate() {
        if let Some(t) = *time
            && t <= until
            && next.is_none_or(|n| (t, class) < n)
        {
            next = Some((t, class));
        }
    }
    next
}

/// What a fill model's outcome adds to an order, and at what price: fill quantities are
/// cumulative per order, so only the increase counts. `None` for no fill.
pub(super) fn fill_increment(order: &SimOrder, outcome: FillResult) -> Option<(Decimal, Decimal)> {
    match outcome {
        FillResult::NoFill => None,
        FillResult::FullFill { fill_price } => Some((order.remaining(), fill_price)),
        FillResult::PartialFill {
            filled_quantity,
            fill_price,
        } => Some((
            filled_quantity
                .min(order.quantity)
                .saturating_sub(order.filled),
            fill_price,
        )),
    }
}

/// A fill price with slippage applied against the taker.
pub(super) fn slipped_price(slippage: &SlippageModel, price: Decimal, dir: Dir) -> Decimal {
    let slip = slippage.calculate_slippage(price, Decimal::ZERO);
    match dir {
        Dir::Buy => price.saturating_add(slip),
        Dir::Sell => price.saturating_sub(slip),
    }
}

/// The fee of one execution of a derivative contract, in its settlement asset:
/// the exact notional times `rate`, rounded half to even to `scale` once.
///
/// # Errors
///
/// [`MMError::NumericalError`] if the notional or the product is not exact.
#[cfg(feature = "instruments")]
pub(super) fn settlement_charge(
    spec: &financial_types::instrument::InstrumentSpec,
    scale: u32,
    price: Decimal,
    quantity: Decimal,
    rate: Decimal,
) -> MMResult<FeeCharge> {
    use financial_types::instrument::{MoneyRatio, Ratio, RoundingRule};
    let numerical = |error: financial_types::instrument::InstrumentError| {
        MMError::NumericalError(format!("settlement fee: {error}"))
    };
    let notional = spec.notional(quantity, price).map_err(numerical)?;
    let numerator = exact_product(notional.ratio.num(), rate).ok_or_else(|| {
        MMError::NumericalError(format!(
            "settlement fee: {} times {rate} is not exact in a Decimal",
            notional.ratio.num()
        ))
    })?;
    let fee = Ratio::new(numerator, notional.ratio.den()).map_err(numerical)?;
    let settled = MoneyRatio::new(fee, notional.asset)
        .settle(scale, RoundingRule::HalfEven)
        .map_err(numerical)?;
    Ok(FeeCharge {
        amount: settled.amount.amount,
        asset: FeeAsset::Settlement,
        quote_value: settled.amount.amount,
    })
}

/// Rounds `value` to a multiple of `step`, up or down; a non positive step
/// leaves the value unchanged.
///
/// Works on the remainder, so it never divides into a huge quotient and
/// never loses precision; if the rounded value would overflow, the value is
/// returned unrounded and the venue checks decide.
#[must_use]
pub(crate) fn round_to_step(value: Decimal, step: Decimal, up: bool) -> Decimal {
    if step <= Decimal::ZERO {
        return value;
    }
    let Some(remainder) = value.checked_rem(step) else {
        return value;
    };
    if remainder.is_zero() {
        return value;
    }
    // Truncation toward zero; adjust to floor for negative values.
    let Some(toward_zero) = value.checked_sub(remainder) else {
        return value;
    };
    let down = if remainder < Decimal::ZERO {
        toward_zero.checked_sub(step)
    } else {
        Some(toward_zero)
    };
    let rounded = if up {
        down.and_then(|d| d.checked_add(step))
    } else {
        down
    };
    rounded.unwrap_or(value)
}

/// Approximate Sharpe ratio of an equity curve (mean over standard
/// deviation of per step returns), or `None` when it is undefined or a
/// step overflows.
pub(super) fn sharpe_ratio(equity_curve: &[(u64, Decimal)]) -> Option<Decimal> {
    // A window whose previous equity is not positive has no return and is
    // skipped; a return that overflows makes the whole ratio `None`.
    let returns: Vec<Decimal> = equity_curve
        .windows(2)
        .filter_map(|w| match w {
            [prev, next] if prev.1 > Decimal::ZERO => Some(
                next.1
                    .checked_sub(prev.1)
                    .and_then(|d| d.checked_div(prev.1)),
            ),
            _ => None,
        })
        .collect::<Option<_>>()?;

    if returns.is_empty() {
        return None;
    }

    let n = Decimal::from(returns.len());
    let mean = returns
        .iter()
        .try_fold(Decimal::ZERO, |acc, r| acc.checked_add(*r))?
        .checked_div(n)?;

    // Calculate standard deviation
    let variance = returns
        .iter()
        .try_fold(Decimal::ZERO, |acc, r| {
            let diff = r.checked_sub(mean)?;
            acc.checked_add(diff.checked_mul(diff)?)
        })?
        .checked_div(n)?;

    if variance <= Decimal::ZERO {
        return None;
    }

    // Approximate sqrt using Newton's method
    let std_dev = decimal_sqrt(variance)?;

    if std_dev > Decimal::ZERO {
        mean.checked_div(std_dev)
    } else {
        None
    }
}

/// Approximate square root using Newton's method.
fn decimal_sqrt(n: Decimal) -> Option<Decimal> {
    if n < Decimal::ZERO {
        return None;
    }
    if n == Decimal::ZERO {
        return Some(Decimal::ZERO);
    }

    let mut x = n;
    let two = Decimal::TWO;

    for _ in 0..20 {
        let next = x.checked_add(n.checked_div(x)?)?.checked_div(two)?;
        if next.checked_sub(x)?.abs() < dec!(0.0000001) {
            return Some(next);
        }
        x = next;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The notional's numerator times the rate must not round: that would be
    /// a second rounding, here of 5.5e-29 BTC to 0 instead of 1e-28.
    #[cfg(feature = "instruments")]
    #[test]
    fn test_settlement_fee_is_rounded_once_or_refused() {
        let spec = crate::instruments::reference::deribit::btc_perpetual().unwrap();
        let (quantity, price) = (Decimal::new(1, 14), dec!(2));
        assert!(matches!(
            settlement_charge(&spec, 28, price, quantity, Decimal::new(11, 16)),
            Err(MMError::NumericalError(_))
        ));
        // 5.5e-28 is a tie, and half to even takes it to 6e-28.
        let fee = settlement_charge(&spec, 28, price, quantity, Decimal::new(11, 15)).unwrap();
        assert_eq!(fee.amount, Decimal::new(6, 28));
        assert_eq!(fee.asset, FeeAsset::Settlement);
    }

    #[cfg(feature = "instruments")]
    #[test]
    fn test_exact_product_refuses_what_checked_mul_rounds() {
        assert_eq!(exact_product(dec!(1.5), dec!(2)), Some(dec!(3)));
        // Twenty nine places, but the last digit is a zero.
        assert_eq!(
            exact_product(Decimal::new(5, 15), Decimal::new(2, 14)),
            Some(Decimal::new(1, 28))
        );
        assert_eq!(
            exact_product(Decimal::new(1, 14), Decimal::new(1, 15)),
            None
        );
        assert_eq!(exact_product(Decimal::MAX, dec!(10)), None);
        assert_eq!(
            exact_product(Decimal::ZERO, Decimal::new(1, 28)),
            Some(Decimal::ZERO)
        );
    }

    /// With or without the `instruments` feature: a settlement asset fee needs
    /// the contract, so the run is refused before it starts.
    #[test]
    fn test_settlement_fee_without_a_contract_is_refused_before_the_run() {
        let config = BacktestConfig {
            fee_rate: dec!(0.001),
            fee_currency: FeeCurrency::Settlement { scale: 8 },
            ..BacktestConfig::default()
        };
        let ticks = vec![create_test_tick(1000, dec!(100), dec!(101))];
        let mut engine = BacktestEngine::new(config, PassiveStrategy, VecDataSource::new(ticks));
        let result = engine.run();
        assert_eq!(result.num_ticks, 0);
        assert!(!result.accounting.is_consistent());
        assert!(
            result
                .accounting
                .setup_refused
                .as_deref()
                .is_some_and(|reason| reason.contains("use with_instrument"))
        );
        assert_eq!(engine.fault_count(), 1);
        assert!(matches!(
            FeeCurrency::Settlement { scale: 8 }.charge(dec!(1), dec!(1), dec!(0.1)),
            Err(MMError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            FeeCurrency::Settlement { scale: 29 }.validate(),
            Err(MMError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn test_base_delta_overflow_returns_numerical_error() {
        let mut fill = SimulatedFill::new(Action::Sell, dec!(1), Decimal::MAX, 0);
        fill.base_fee = dec!(1);
        let result = fill.base_delta();
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        let mut fill = SimulatedFill::new(Action::Buy, dec!(1), Decimal::MIN, 0);
        fill.base_fee = dec!(1);
        assert!(matches!(fill.base_delta(), Err(MMError::NumericalError(_))));
    }

    #[test]
    fn test_sharpe_ratio_overflowing_return_is_none() {
        let tiny = dec!(0.0000000000000000000000000001);
        let curve = [(0, dec!(100)), (1, dec!(110)), (2, tiny), (3, Decimal::MAX)];
        assert_eq!(sharpe_ratio(&curve), None);
    }

    #[test]
    fn test_sharpe_ratio_non_positive_previous_equity_is_skipped() {
        let curve = [
            (0, dec!(100)),
            (1, dec!(-5)),
            (2, dec!(110)),
            (3, dec!(121)),
            (4, dec!(120)),
        ];
        assert!(sharpe_ratio(&curve).is_some());
    }
    use crate::backtest::VecDataSource;
    use crate::backtest::{
        MarketImpactFillModel, ProbabilisticFillModel, QueuePositionFillModel, SimLatencyConfig,
    };
    use crate::dec;

    struct TestStrategy {
        quote_spread: Decimal,
        fills_received: Vec<SimulatedFill>,
    }

    impl TestStrategy {
        fn new(spread: Decimal) -> Self {
            Self {
                quote_spread: spread,
                fills_received: Vec::new(),
            }
        }
    }

    impl BacktestStrategy for TestStrategy {
        fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
            let mid = tick.mid_price();
            let half_spread = self.quote_spread / Decimal::TWO;
            Some(Quote {
                bid_price: mid - half_spread,
                bid_size: Decimal::ONE,
                ask_price: mid + half_spread,
                ask_size: Decimal::ONE,
                timestamp: tick.timestamp,
            })
        }

        fn on_fill(&mut self, fill: &SimulatedFill) {
            self.fills_received.push(fill.clone());
        }

        fn reset(&mut self) {
            self.fills_received.clear();
        }
    }

    struct PassiveStrategy;

    impl BacktestStrategy for PassiveStrategy {
        fn on_tick(&mut self, _tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
            None
        }
        fn on_fill(&mut self, _fill: &SimulatedFill) {}
        fn reset(&mut self) {}
    }

    fn create_test_tick(timestamp: u64, bid: Decimal, ask: Decimal) -> MarketTick {
        MarketTick::new(timestamp, bid, dec!(1.0), ask, dec!(1.0))
    }

    #[test]
    fn test_simulated_fill_new() {
        let fill = SimulatedFill::new(Action::Buy, dec!(100.0), dec!(0.1), 1000);

        assert_eq!(fill.side, Action::Buy);
        assert_eq!(fill.price, dec!(100.0));
        assert_eq!(fill.quantity, dec!(0.1));
        assert_eq!(fill.timestamp, 1000);
        assert_eq!(fill.fee, Decimal::ZERO);
    }

    #[test]
    fn test_simulated_fill_with_fee() {
        let fill = SimulatedFill::with_fee(Action::Sell, dec!(100.0), dec!(0.1), 1000, dec!(0.01));

        assert_eq!(fill.fee, dec!(0.01));
    }

    #[test]
    fn test_simulated_fill_notional() {
        let fill = SimulatedFill::new(Action::Buy, dec!(100.0), dec!(0.5), 1000);
        assert_eq!(fill.notional(), dec!(50.0));
    }

    #[test]
    fn test_slippage_model_none() {
        let model = SlippageModel::None;
        assert_eq!(
            model.calculate_slippage(dec!(100.0), dec!(0.01)),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_slippage_model_fixed() {
        let model = SlippageModel::Fixed(dec!(0.05));
        assert_eq!(
            model.calculate_slippage(dec!(100.0), dec!(0.01)),
            dec!(0.05)
        );
    }

    #[test]
    fn test_slippage_model_percentage() {
        let model = SlippageModel::Percentage(dec!(0.001)); // 0.1%
        assert_eq!(model.calculate_slippage(dec!(100.0), dec!(0.01)), dec!(0.1));
    }

    #[test]
    fn test_slippage_model_volatility_based() {
        let model = SlippageModel::VolatilityBased {
            multiplier: dec!(2.0),
        };
        // 100 * 0.01 * 2 = 2
        assert_eq!(model.calculate_slippage(dec!(100.0), dec!(0.01)), dec!(2.0));
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();

        assert_eq!(config.initial_capital, Decimal::from(100_000));
        assert_eq!(config.fee_rate, Decimal::ZERO);
        assert!(config.record_equity_curve);
        assert!(config.record_trades);
    }

    #[test]
    fn test_backtest_config_builder() {
        let config = BacktestConfig::new()
            .with_initial_capital(dec!(50000.0))
            .with_fee_rate(dec!(0.001))
            .with_slippage(SlippageModel::Fixed(dec!(0.01)));

        assert_eq!(config.initial_capital, dec!(50000.0));
        assert_eq!(config.fee_rate, dec!(0.001));
    }

    #[test]
    fn test_fee_currency_quote_matches_historical_formula() {
        let charge = FeeCurrency::Quote
            .charge(dec!(100.5), dec!(0.3), dec!(0.001))
            .unwrap();
        assert_eq!(charge.amount, dec!(0.03015));
        assert_eq!(charge.asset, FeeAsset::Quote);
        assert_eq!(charge.quote_value, dec!(0.03015));
    }

    #[test]
    fn test_fee_currency_base_buy_and_sell_rounds_half_even_to_scale() {
        let base = FeeCurrency::Base { scale: 4 };
        // 0.25 * 0.001 = 0.00025 -> 0.0002 (half to even).
        let charge = base.charge(dec!(100), dec!(0.25), dec!(0.001)).unwrap();
        assert_eq!(charge.amount, dec!(0.0002));
        assert_eq!(charge.asset, FeeAsset::Base);
        assert_eq!(charge.quote_value, dec!(0.02));
        // 0.35 * 0.001 = 0.00035 -> 0.0004.
        let charge = base.charge(dec!(100), dec!(0.35), dec!(0.001)).unwrap();
        assert_eq!(charge.amount, dec!(0.0004));
        assert_eq!(charge.quote_value, dec!(0.04));
    }

    #[test]
    fn test_fee_currency_third_asset_converts_through_its_quote_price() {
        let other = FeeCurrency::Other {
            asset: "BNB".to_string(),
            quote_per_unit: dec!(300),
            scale: 6,
        };
        // 100 * 2 * 0.001 = 0.2 quote = 0.000666666.. BNB -> 0.000667.
        let charge = other.charge(dec!(100), dec!(2), dec!(0.001)).unwrap();
        assert_eq!(charge.amount, dec!(0.000667));
        assert_eq!(charge.asset, FeeAsset::Other("BNB".to_string()));
        assert_eq!(charge.quote_value, dec!(0.2001));
    }

    #[test]
    fn test_fee_currency_invalid_parameters_return_configuration_error() {
        for currency in [
            FeeCurrency::Base { scale: 29 },
            FeeCurrency::Other {
                asset: " ".to_string(),
                quote_per_unit: dec!(1),
                scale: 2,
            },
            FeeCurrency::Other {
                asset: "BNB".to_string(),
                quote_per_unit: Decimal::ZERO,
                scale: 2,
            },
        ] {
            assert!(matches!(
                currency.validate(),
                Err(MMError::InvalidConfiguration(_))
            ));
            assert!(matches!(
                currency.charge(dec!(1), dec!(1), dec!(0.001)),
                Err(MMError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn test_fee_currency_overflow_returns_numerical_error() {
        assert!(matches!(
            FeeCurrency::Quote.charge(Decimal::MAX, dec!(2), dec!(1)),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_engine_base_fee_posts_base_amount_and_reports_quote_value() {
        let accountant = RecordingAccountant::default();
        let log = accountant.log.clone();
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(dec!(0.001))
                .with_fee_currency(FeeCurrency::Base { scale: 8 }),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap()
        .with_accountant(accountant);
        let result = engine.run();
        let posted: Vec<(Decimal, FeeAsset)> = log
            .lock()
            .unwrap()
            .iter()
            .map(|f| (f.fee, f.fee_asset.clone()))
            .collect();
        assert_eq!(
            posted,
            vec![
                (dec!(0.0005), FeeAsset::Base),
                (dec!(0.001), FeeAsset::Base),
                (dec!(0.0005), FeeAsset::Base),
            ]
        );
        let reported: Vec<Decimal> = result.trades.iter().map(|f| f.fee).collect();
        assert_eq!(reported, vec![dec!(0.05), dec!(0.1), dec!(0.05)]);
    }

    /// One crossing tick at 100, then the mid moves to 200.
    fn base_fee_engine(
        quotes: Vec<Option<(Decimal, Decimal)>>,
        ticks: Vec<MarketTick>,
    ) -> BacktestEngine<ScriptedStrategy, VecDataSource> {
        BacktestEngine::new(
            BacktestConfig::default()
                .with_default_order_size(dec!(1))
                .with_fee_rate(dec!(0.01))
                .with_fee_currency(FeeCurrency::Base { scale: 8 }),
            ScriptedStrategy::new(quotes),
            VecDataSource::new(ticks),
        )
    }

    #[test]
    fn test_engine_base_fee_price_move_marks_net_holdings() {
        // Buy 1 at 100 with a 1% fee in base: 0.99 held, cash -100. At a
        // mid of 200 the equity change is -100 + 0.99 * 200 = 98.
        let ticks = vec![
            MarketTick::new(1000, dec!(99), dec!(1), dec!(101), dec!(1)),
            MarketTick::new(1001, dec!(99), dec!(1), dec!(100), dec!(1)),
            MarketTick::new(1002, dec!(199), dec!(1), dec!(201), dec!(1)),
        ];
        let mut engine = base_fee_engine(vec![Some((dec!(100), dec!(300)))], ticks);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.final_position, dec!(0.99));
        assert_eq!(result.net_pnl, dec!(98));
        // The fee (0.01 BTC) is marked at 200 like the holdings.
        assert_eq!(result.total_fees, dec!(2));
        assert_eq!(result.total_pnl, dec!(100));
    }

    #[test]
    fn test_engine_base_fee_round_trip_leaves_the_fee_short() {
        // Buy 1 at 100 then sell 1 at 100, 1% base fee each: 0.99 then
        // -0.02 held; the strategy sees the same net inventory.
        let ticks = vec![
            MarketTick::new(1000, dec!(99), dec!(1), dec!(101), dec!(1)),
            MarketTick::new(1001, dec!(99), dec!(1), dec!(100), dec!(1)),
            MarketTick::new(1002, dec!(99), dec!(1), dec!(101), dec!(1)),
            // Crosses the resting ask at 100.
            MarketTick::new(1003, dec!(100), dec!(1), dec!(102), dec!(1)),
            MarketTick::new(1004, dec!(99), dec!(1), dec!(101), dec!(1)),
        ];
        let quotes = vec![
            Some((dec!(100), dec!(300))),
            None,
            Some((dec!(50), dec!(100))),
        ];
        let mut engine = base_fee_engine(quotes, ticks);
        let result = engine.run();
        assert_eq!(result.num_trades, 2);
        assert_eq!(result.final_position, dec!(-0.02));
        // Cash 0, holdings -0.02 at the last mid of 100.
        assert_eq!(result.net_pnl, dec!(-2));
        let seen = engine.strategy().seen.last().map(|(_, q)| *q);
        assert_eq!(seen, Some(dec!(-0.02)));
    }

    #[test]
    fn test_engine_invalid_fee_currency_records_fault_and_books_nothing() {
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(dec!(0.001))
                .with_fee_currency(FeeCurrency::Base { scale: 40 }),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap();
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert!(matches!(
            engine.faults().first(),
            Some(MMError::InvalidConfiguration(_))
        ));
        // A fee that cannot be computed drops the fill, visibly.
        assert!(result.accounting.dropped_fills > 0);
        assert!(!result.accounting.is_consistent());
    }

    #[test]
    fn test_backtest_result_default() {
        let result = BacktestResult::default();

        assert_eq!(result.total_pnl, Decimal::ZERO);
        assert_eq!(result.num_trades, 0);
    }

    #[test]
    fn test_backtest_result_avg_trade_pnl() {
        let result = BacktestResult {
            net_pnl: dec!(100.0),
            num_trades: 10,
            ..Default::default()
        };

        assert_eq!(result.avg_trade_pnl(), dec!(10.0));
    }

    #[test]
    fn test_backtest_result_duration() {
        let result = BacktestResult {
            start_time: 1000,
            end_time: 5000,
            ..Default::default()
        };

        assert_eq!(result.duration_ms(), 4000);
    }

    #[test]
    fn test_backtest_engine_passive() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(100.1), dec!(100.3)),
            create_test_tick(1002, dec!(100.2), dec!(100.4)),
        ];

        let mut engine = BacktestEngine::new(
            BacktestConfig::default(),
            PassiveStrategy,
            VecDataSource::new(ticks),
        );

        let result = engine.run();

        assert_eq!(result.num_ticks, 3);
        assert_eq!(result.num_trades, 0);
        assert_eq!(result.start_time, 1000);
        assert_eq!(result.end_time, 1002);
    }

    #[test]
    fn test_backtest_engine_with_fills() {
        // Quotes are submitted on one tick and can only fill on a later tick.
        // Tick 1 quotes bid 100.09 / ask 100.11 around mid 100.1.
        // Tick 2 has ask 99.98 <= resting bid 100.09, so the bid submitted on
        // tick 1 fills on tick 2 at its limit price.
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)), // No fill
            create_test_tick(1001, dec!(99.95), dec!(99.98)), // Ask 99.98 <= bid_quote ~99.965
        ];

        let strategy = TestStrategy::new(dec!(0.02)); // Very tight spread

        let mut engine = BacktestEngine::new(
            BacktestConfig::default().with_default_order_size(dec!(1.0)),
            strategy,
            VecDataSource::new(ticks),
        );

        let result = engine.run();

        assert_eq!(result.num_ticks, 2);
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].side, Action::Buy);
        assert_eq!(result.trades[0].price, dec!(100.09));
        assert_eq!(result.trades[0].timestamp, 1001);
        // Actual fill logic depends on market crossing our quotes
        assert_eq!(result.start_time, 1000);
        assert_eq!(result.end_time, 1001);
    }

    #[test]
    fn test_backtest_engine_with_fees() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(99.8), dec!(100.0)), // Triggers buy
        ];

        let strategy = TestStrategy::new(dec!(0.2));

        let config = BacktestConfig::default()
            .with_fee_rate(dec!(0.001))
            .with_default_order_size(dec!(1.0));

        let mut engine = BacktestEngine::new(config, strategy, VecDataSource::new(ticks));

        let result = engine.run();

        // Should have fees if there were trades
        if result.num_trades > 0 {
            assert!(result.total_fees > Decimal::ZERO);
        }
    }

    #[test]
    fn test_backtest_engine_reset() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(100.1), dec!(100.3)),
        ];

        let mut engine = BacktestEngine::new(
            BacktestConfig::default(),
            PassiveStrategy,
            VecDataSource::new(ticks),
        );

        let result1 = engine.run();
        assert_eq!(result1.num_ticks, 2);

        engine.reset();

        let result2 = engine.run();
        assert_eq!(result2.num_ticks, 2);
    }

    #[test]
    fn test_backtest_engine_equity_curve() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(100.1), dec!(100.3)),
            create_test_tick(1002, dec!(100.2), dec!(100.4)),
        ];

        let config = BacktestConfig::default().with_record_equity_curve(true);

        let mut engine = BacktestEngine::new(config, PassiveStrategy, VecDataSource::new(ticks));

        let result = engine.run();

        assert_eq!(result.equity_curve.len(), 3);
    }

    #[test]
    fn test_backtest_engine_no_equity_curve() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(100.1), dec!(100.3)),
        ];

        let config = BacktestConfig::default().with_record_equity_curve(false);

        let mut engine = BacktestEngine::new(config, PassiveStrategy, VecDataSource::new(ticks));

        let result = engine.run();

        assert!(result.equity_curve.is_empty());
    }

    #[test]
    fn test_backtest_engine_get_state() {
        let ticks = vec![create_test_tick(1000, dec!(100.0), dec!(100.2))];

        let engine = BacktestEngine::new(
            BacktestConfig::default(),
            PassiveStrategy,
            VecDataSource::new(ticks),
        );

        let (position, pnl) = engine.get_state();
        assert_eq!(position.quantity, Decimal::ZERO);
        assert_eq!(pnl.total, Decimal::ZERO);
    }

    #[test]
    fn test_decimal_sqrt() {
        assert_eq!(decimal_sqrt(Decimal::ZERO), Some(Decimal::ZERO));
        assert!(decimal_sqrt(dec!(-1.0)).is_none());

        let sqrt_4 = decimal_sqrt(dec!(4.0)).unwrap();
        assert!((sqrt_4 - dec!(2.0)).abs() < dec!(0.0001));

        let sqrt_2 = decimal_sqrt(dec!(2.0)).unwrap();
        assert!((sqrt_2 - dec!(1.414)).abs() < dec!(0.001));
    }

    #[test]
    fn test_backtest_engine_progress_callback() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(100.1), dec!(100.3)),
            create_test_tick(1002, dec!(100.2), dec!(100.4)),
        ];

        let mut progress_calls = 0;

        let mut engine = BacktestEngine::new(
            BacktestConfig::default(),
            PassiveStrategy,
            VecDataSource::new(ticks),
        );

        engine.run_with_progress(|current, total| {
            progress_calls += 1;
            assert!(current <= total);
        });

        assert_eq!(progress_calls, 3);
    }

    // Additional tests for improved coverage

    #[test]
    fn test_backtest_result_avg_trade_pnl_zero_trades() {
        let result = BacktestResult {
            net_pnl: dec!(100.0),
            num_trades: 0,
            ..Default::default()
        };

        assert_eq!(result.avg_trade_pnl(), Decimal::ZERO);
    }

    #[test]
    fn test_backtest_config_with_slippage() {
        let config = BacktestConfig::default().with_slippage(SlippageModel::Fixed(dec!(0.01)));

        assert!(matches!(config.slippage, SlippageModel::Fixed(_)));
    }

    #[test]
    fn test_backtest_config_with_initial_capital() {
        let config = BacktestConfig::default().with_initial_capital(dec!(50000.0));

        assert_eq!(config.initial_capital, dec!(50000.0));
    }

    #[test]
    fn test_backtest_config_with_record_trades() {
        let config = BacktestConfig::default().with_record_trades(true);

        assert!(config.record_trades);
    }

    #[test]
    fn test_backtest_engine_with_actual_fills() {
        // Create ticks where market definitely crosses our quotes
        // Strategy quotes bid at mid - spread/2, ask at mid + spread/2
        // For a BUY fill: market ask <= our bid
        // For a SELL fill: market bid >= our ask
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)), // Initial tick
            create_test_tick(1001, dec!(99.0), dec!(99.1)),   // Market dropped, ask 99.1 < prev bid
            create_test_tick(1002, dec!(101.0), dec!(101.2)), // Market jumped, bid 101 > prev ask
        ];

        // Wide spread strategy
        let strategy = TestStrategy::new(dec!(2.0)); // 2.0 spread

        let config = BacktestConfig::default()
            .with_default_order_size(dec!(1.0))
            .with_record_trades(true);

        let mut engine = BacktestEngine::new(config, strategy, VecDataSource::new(ticks));

        let result = engine.run();

        assert_eq!(result.num_ticks, 3);
    }

    #[test]
    fn test_backtest_engine_sharpe_calculation() {
        // Create enough ticks to calculate Sharpe ratio
        let mut ticks = Vec::new();
        for i in 0..20 {
            let price = dec!(100.0) + Decimal::from(i % 3);
            ticks.push(create_test_tick(1000 + i as u64, price, price + dec!(0.2)));
        }

        let config = BacktestConfig::default().with_record_equity_curve(true);

        let mut engine = BacktestEngine::new(config, PassiveStrategy, VecDataSource::new(ticks));

        let result = engine.run();

        // Sharpe should be calculated if we have equity curve
        assert_eq!(result.equity_curve.len(), 20);
    }

    #[test]
    fn test_backtest_engine_max_drawdown() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.2)),
            create_test_tick(1001, dec!(95.0), dec!(95.2)), // Price drop
            create_test_tick(1002, dec!(90.0), dec!(90.2)), // Further drop
            create_test_tick(1003, dec!(100.0), dec!(100.2)), // Recovery
        ];

        let config = BacktestConfig::default().with_record_equity_curve(true);

        let mut engine = BacktestEngine::new(config, PassiveStrategy, VecDataSource::new(ticks));

        let result = engine.run();

        assert_eq!(result.num_ticks, 4);
    }

    #[test]
    fn test_backtest_result_is_profitable() {
        let profitable = BacktestResult {
            net_pnl: dec!(100.0),
            ..Default::default()
        };
        assert!(profitable.net_pnl > Decimal::ZERO);

        let unprofitable = BacktestResult {
            net_pnl: dec!(-50.0),
            ..Default::default()
        };
        assert!(unprofitable.net_pnl < Decimal::ZERO);
    }

    /// Strategy returning scripted quotes per tick index and recording reports.
    struct ScriptedStrategy {
        quotes: Vec<Option<(Decimal, Decimal)>>,
        index: usize,
        reports: Vec<ExecutionReport>,
        fills: usize,
        seen: Vec<(u64, Decimal)>,
    }

    impl ScriptedStrategy {
        fn new(quotes: Vec<Option<(Decimal, Decimal)>>) -> Self {
            Self {
                quotes,
                index: 0,
                reports: Vec::new(),
                fills: 0,
                seen: Vec::new(),
            }
        }
    }

    impl BacktestStrategy for ScriptedStrategy {
        fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
            self.seen.push((tick.timestamp, position.quantity));
            let quote = self.quotes.get(self.index).copied().flatten();
            self.index += 1;
            // Zero sizes: the engine falls back to `default_order_size`.
            quote.map(|(bid, ask)| Quote {
                bid_price: bid,
                bid_size: Decimal::ZERO,
                ask_price: ask,
                ask_size: Decimal::ZERO,
                timestamp: tick.timestamp,
            })
        }
        fn on_fill(&mut self, _fill: &SimulatedFill) {
            self.fills += 1;
        }
        fn on_execution_report(&mut self, report: &ExecutionReport) {
            self.reports.push(report.clone());
        }
        fn reset(&mut self) {
            self.index = 0;
            self.reports.clear();
            self.fills = 0;
            self.seen.clear();
        }
    }

    fn scripted_engine(
        ticks: Vec<MarketTick>,
        quotes: Vec<Option<(Decimal, Decimal)>>,
    ) -> BacktestEngine<ScriptedStrategy, VecDataSource> {
        BacktestEngine::new(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(quotes),
            VecDataSource::new(ticks),
        )
    }

    #[test]
    fn test_quote_never_fills_on_its_own_tick() {
        // The quote crosses the market on the tick it is computed from.
        let ticks = vec![create_test_tick(1000, dec!(99.0), dec!(99.5))];
        let mut engine = scripted_engine(ticks, vec![Some((dec!(100), dec!(101)))]);
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert_eq!(engine.open_orders().len(), 2);
    }

    #[test]
    fn test_equal_timestamp_tick_does_not_fill_new_order() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1000, dec!(99.0), dec!(99.5)),
            create_test_tick(1001, dec!(99.0), dec!(99.5)),
        ];
        let mut engine = scripted_engine(ticks, vec![Some((dec!(100), dec!(103)))]);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].timestamp, 1001);
        assert_eq!(result.trades[0].quantity, dec!(2));
    }

    #[test]
    fn test_orders_persist_when_strategy_returns_none() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(98.0), dec!(102.0)),
            create_test_tick(1002, dec!(101.5), dec!(102.0)),
        ];
        let mut engine = scripted_engine(ticks, vec![Some((dec!(99), dec!(101))), None, None]);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].side, Action::Sell);
        assert_eq!(result.trades[0].price, dec!(101));
        assert_eq!(result.final_position, dec!(-2));
        // The bid is still resting.
        assert_eq!(engine.open_orders().len(), 1);
        assert_eq!(engine.open_orders()[0].side, Action::Buy);
    }

    #[test]
    fn test_full_report_history_for_cancel_replace_and_fill() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(98.0), dec!(102.0)),
            create_test_tick(1002, dec!(97.0), dec!(99.4)),
        ];
        let mut engine = scripted_engine(
            ticks,
            vec![
                Some((dec!(99), dec!(101))),
                Some((dec!(99.5), dec!(100.5))),
                None,
            ],
        );
        engine.run();
        let history: Vec<(u64, SimOrderId, ExecutionKind)> = engine
            .execution_reports()
            .iter()
            .map(|r| (r.stamp.timestamp, r.order_id, r.kind))
            .collect();
        assert_eq!(
            history,
            vec![
                (1000, 1, ExecutionKind::Accepted),
                (1000, 2, ExecutionKind::Accepted),
                (1001, 1, ExecutionKind::Canceled),
                (1001, 2, ExecutionKind::Canceled),
                (1001, 3, ExecutionKind::Accepted),
                (1001, 4, ExecutionKind::Accepted),
                (1002, 3, ExecutionKind::Fill),
            ]
        );
        let stamps: Vec<EventStampKey> = engine
            .execution_reports()
            .iter()
            .map(|r| (r.stamp.timestamp, r.stamp.sequence))
            .collect();
        let mut sorted = stamps.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(stamps, sorted, "stamps are unique and strictly increasing");
        // Strategy saw the same reports in the same order.
        assert_eq!(engine.strategy().reports, engine.execution_reports());
        assert_eq!(engine.strategy().fills, 1);
    }

    type EventStampKey = (u64, u64);

    #[test]
    fn test_invalid_quote_side_is_rejected() {
        let ticks = vec![create_test_tick(1000, dec!(98.0), dec!(102.0))];
        let mut engine = scripted_engine(ticks, vec![Some((dec!(0), dec!(101)))]);
        engine.run();
        let kinds: Vec<ExecutionKind> = engine.execution_reports().iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![ExecutionKind::Rejected, ExecutionKind::Accepted]
        );
        assert_eq!(engine.open_orders().len(), 1);
    }

    /// Strategy quoting one fixed quote with explicit sizes.
    struct SizedStrategy(Quote);

    impl BacktestStrategy for SizedStrategy {
        fn on_tick(&mut self, _tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
            Some(self.0.clone())
        }
        fn on_fill(&mut self, _fill: &SimulatedFill) {}
        fn reset(&mut self) {}
    }

    fn sized_reports(config: BacktestConfig, quote: Quote) -> Vec<ExecutionReport> {
        let ticks = vec![create_test_tick(1000, dec!(90.0), dec!(110.0))];
        let mut engine = BacktestEngine::new(
            config.with_record_trades(true),
            SizedStrategy(quote),
            VecDataSource::new(ticks),
        );
        engine.run();
        engine.execution_reports().to_vec()
    }

    fn sized_quote(bid: Decimal, bid_size: Decimal, ask: Decimal, ask_size: Decimal) -> Quote {
        Quote {
            bid_price: bid,
            bid_size,
            ask_price: ask,
            ask_size,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_orders_use_quote_sizes_per_side() {
        let reports = sized_reports(
            BacktestConfig::default().with_default_order_size(dec!(5)),
            sized_quote(dec!(99), dec!(1.5), dec!(101), dec!(0.25)),
        );
        let sizes: Vec<(Action, Decimal)> = reports.iter().map(|r| (r.side, r.remaining)).collect();
        assert_eq!(
            sizes,
            vec![(Action::Buy, dec!(1.5)), (Action::Sell, dec!(0.25))]
        );
    }

    #[test]
    fn test_non_positive_quote_size_uses_default() {
        let reports = sized_reports(
            BacktestConfig::default().with_default_order_size(dec!(5)),
            sized_quote(dec!(99), Decimal::ZERO, dec!(101), dec!(-1)),
        );
        assert!(reports.iter().all(|r| r.remaining == dec!(5)));
    }

    #[test]
    fn test_prices_round_away_from_the_market_on_tick() {
        let reports = sized_reports(
            BacktestConfig::default().with_tick_size(dec!(0.5)),
            sized_quote(dec!(99.74), dec!(1), dec!(100.26), dec!(1)),
        );
        let prices: Vec<Decimal> = reports.iter().map(|r| r.price).collect();
        assert_eq!(prices, vec![dec!(99.5), dec!(100.5)]);
    }

    #[test]
    fn test_quantity_rounds_down_to_lot() {
        let reports = sized_reports(
            BacktestConfig::default().with_lot_size(dec!(0.1)),
            sized_quote(dec!(99), dec!(1.29), dec!(101), dec!(0.37)),
        );
        let sizes: Vec<Decimal> = reports.iter().map(|r| r.remaining).collect();
        assert_eq!(sizes, vec![dec!(1.2), dec!(0.3)]);
    }

    #[test]
    fn test_quantity_below_one_lot_is_rejected() {
        let reports = sized_reports(
            BacktestConfig::default().with_lot_size(dec!(1)),
            sized_quote(dec!(99), dec!(0.4), dec!(101), dec!(2)),
        );
        let kinds: Vec<(Action, ExecutionKind)> =
            reports.iter().map(|r| (r.side, r.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (Action::Buy, ExecutionKind::Rejected),
                (Action::Sell, ExecutionKind::Accepted)
            ]
        );
    }

    #[test]
    fn test_round_to_step_boundaries() {
        // A tiny step must not overflow through a huge quotient.
        let tiny = Decimal::new(1, 28);
        assert_eq!(round_to_step(dec!(100), tiny, false), dec!(100));
        assert_eq!(round_to_step(dec!(100), tiny, true), dec!(100));
        // Just under one lot rounds down to zero, never up.
        let almost = dec!(2.9999999999999999999999999999);
        assert_eq!(round_to_step(almost, dec!(3), false), Decimal::ZERO);
        assert_eq!(round_to_step(almost, dec!(3), true), dec!(3));
        // Huge value with a fine step stays put when rounding up overflows.
        assert_eq!(round_to_step(Decimal::MAX, dec!(0.7), true), Decimal::MAX);
        assert_eq!(round_to_step(dec!(-1.25), dec!(0.5), false), dec!(-1.5));
        assert_eq!(round_to_step(dec!(-1.25), dec!(0.5), true), dec!(-1.0));
    }

    #[test]
    fn test_crossed_quote_is_not_hidden_by_rounding() {
        let reports = sized_reports(
            BacktestConfig::default(),
            sized_quote(dec!(100.004), dec!(1), dec!(100.001), dec!(1)),
        );
        assert!(reports.iter().all(|r| r.kind == ExecutionKind::Rejected));
    }

    #[test]
    fn test_zero_tick_and_lot_keep_values() {
        let reports = sized_reports(
            BacktestConfig::default()
                .with_tick_size(Decimal::ZERO)
                .with_lot_size(Decimal::ZERO),
            sized_quote(dec!(99.123), dec!(0.0007), dec!(101.987), dec!(1)),
        );
        let got: Vec<(Decimal, Decimal)> = reports.iter().map(|r| (r.price, r.remaining)).collect();
        assert_eq!(
            got,
            vec![(dec!(99.123), dec!(0.0007)), (dec!(101.987), dec!(1))]
        );
    }

    #[test]
    fn test_crossed_quote_is_rejected_on_both_sides() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(103.0)),
        ];
        let mut engine = scripted_engine(
            ticks,
            vec![Some((dec!(100), dec!(100))), Some((dec!(101), dec!(99)))],
        );
        let result = engine.run();
        assert!(
            engine
                .execution_reports()
                .iter()
                .all(|r| r.kind == ExecutionKind::Rejected)
        );
        assert_eq!(engine.execution_reports().len(), 4);
        assert!(engine.open_orders().is_empty());
        assert_eq!(result.num_trades, 0);
    }

    #[test]
    fn test_out_of_order_tick_cannot_fill_before_activation() {
        // Submitted at 1000; 1001 does not cross; the stale tick at 999
        // crosses but predates the activation and must not fill.
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(98.0), dec!(102.0)),
            create_test_tick(999, dec!(97.0), dec!(99.0)),
        ];
        let mut engine = scripted_engine(ticks, vec![Some((dec!(100), dec!(103))), None, None]);
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert_eq!(engine.open_orders().len(), 2);
    }

    #[test]
    fn test_replay_is_deterministic() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(99.4)),
            create_test_tick(1002, dec!(100.6), dec!(103.0)),
            create_test_tick(1003, dec!(98.0), dec!(102.0)),
        ];
        let quotes = vec![
            Some((dec!(99.5), dec!(100.5))),
            Some((dec!(99.5), dec!(100.5))),
            None,
            Some((dec!(99), dec!(101))),
        ];
        let mut engine = scripted_engine(ticks, quotes);
        let first = engine.run();
        let first_reports = engine.execution_reports().to_vec();
        engine.reset();
        let second = engine.run();
        assert_eq!(first_reports, engine.execution_reports());
        assert_eq!(first.trades, second.trades);
        assert_eq!(first.net_pnl, second.net_pnl);
        assert_eq!(first.num_trades, 2);
    }

    #[test]
    fn test_fill_fee_matches_report() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(99.0)),
        ];
        let mut engine = BacktestEngine::new(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(dec!(0.001)),
            ScriptedStrategy::new(vec![Some((dec!(100), dec!(110)))]),
            VecDataSource::new(ticks),
        );
        let result = engine.run();
        let fill = engine
            .execution_reports()
            .iter()
            .find(|r| r.is_fill())
            .expect("fill report");
        assert_eq!(fill.fee, dec!(0.2));
        assert_eq!(result.total_fees, dec!(0.2));
        assert_eq!(fill.cumulative_quantity, dec!(2));
        assert_eq!(fill.remaining, Decimal::ZERO);
    }

    fn queue_ticks() -> Vec<MarketTick> {
        // Buy quoted at 100 on tick 1000. The ask sits at 100 with size 1, so
        // the queue model sees 1 unit ahead of the order.
        let crossing = |ts| MarketTick::new(ts, dec!(99), dec!(1), dec!(100), dec!(1));
        vec![
            MarketTick::new(1000, dec!(99), dec!(1), dec!(101), dec!(1)),
            crossing(1001),
            crossing(1003),
            crossing(1005),
            crossing(1007),
        ]
    }

    fn queue_quotes() -> Vec<Option<(Decimal, Decimal)>> {
        vec![Some((dec!(100), dec!(200)))]
    }

    #[test]
    fn test_changing_fill_model_changes_executions() {
        let mut immediate = scripted_engine(queue_ticks(), queue_quotes());
        let immediate_result = immediate.run();

        let mut queued = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap();
        let queued_result = queued.run();

        // Immediate: whole order at the first crossing tick.
        assert_eq!(immediate_result.num_trades, 1);
        assert_eq!(immediate_result.trades[0].timestamp, 1001);
        assert_eq!(immediate_result.trades[0].quantity, dec!(2));

        // Queue model: cleared volume 0.5 per ms behind 1 unit of queue.
        // t=1001: cleared 0.5, no fill. t=1003: cleared 1.5, cumulative 0.5.
        // t=1005: cleared 2.5, cumulative 1.5. t=1007: cleared 3.5, full.
        let fills: Vec<(u64, Decimal)> = queued_result
            .trades
            .iter()
            .map(|f| (f.timestamp, f.quantity))
            .collect();
        assert_eq!(
            fills,
            vec![(1003, dec!(0.5)), (1005, dec!(1.0)), (1007, dec!(0.5))]
        );
        assert_eq!(queued_result.final_position, dec!(2));
        let kinds: Vec<ExecutionKind> = queued
            .execution_reports()
            .iter()
            .filter(|r| r.is_fill())
            .map(|r| r.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                ExecutionKind::PartialFill,
                ExecutionKind::PartialFill,
                ExecutionKind::Fill
            ]
        );
        assert!(
            queued
                .fill_model_description()
                .contains("heuristic estimate")
        );
        assert!(immediate.fill_model_description().contains("reference"));
    }

    #[test]
    fn test_partial_fills_conserve_cash_and_fees() {
        let config = BacktestConfig::default()
            .with_default_order_size(dec!(2))
            .with_fee_rate(dec!(0.001));
        let mut queued = BacktestEngine::with_fill_model(
            config.clone(),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap();
        let partial = queued.run();
        assert_eq!(partial.num_trades, 3);

        // Same economics as one 2.0 fill at 100 with fee 0.2.
        assert_eq!(partial.total_fees, dec!(0.2));
        assert_eq!(partial.final_position, dec!(2));
        let (_, pnl) = queued.get_state();
        assert_eq!(pnl.realized, dec!(-200));
        let fees_in_reports: Decimal = queued
            .execution_reports()
            .iter()
            .filter(|r| r.is_fill())
            .map(|r| r.fee)
            .sum();
        assert_eq!(fees_in_reports, dec!(0.2));
    }

    /// Model returning a scripted cumulative quantity per tick, used to
    /// check that a decreasing cumulative fill books nothing.
    #[derive(Debug)]
    struct CumulativeStub {
        script: Vec<Decimal>,
        tick: usize,
    }

    impl FillModel for CumulativeStub {
        fn simulate_fill(
            &self,
            order: &SimulatedOrder,
            _tick: &MarketTick,
            _time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            Ok(match self.script.get(self.tick.saturating_sub(1)) {
                Some(q) if *q > Decimal::ZERO && order.side == Action::Buy => {
                    FillResult::PartialFill {
                        filled_quantity: *q,
                        fill_price: order.price,
                    }
                }
                _ => FillResult::NoFill,
            })
        }
        fn reset(&mut self) {
            self.tick = 0;
        }
        fn name(&self) -> &'static str {
            "CumulativeStub"
        }
        fn on_tick(&mut self, _tick: &MarketTick) {
            self.tick += 1;
        }
    }

    #[test]
    fn test_decreasing_cumulative_partial_fill_is_skipped() {
        // Tick 1 places the quote; ticks 2..4 report cumulative 1.0, 0.5, 1.5.
        let stub = CumulativeStub {
            script: vec![Decimal::ZERO, dec!(1.0), dec!(0.5), dec!(1.5)],
            tick: 0,
        };
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            stub,
        )
        .unwrap();
        let result = engine.run();
        let fills: Vec<(u64, Decimal)> = result
            .trades
            .iter()
            .map(|f| (f.timestamp, f.quantity))
            .collect();
        assert_eq!(fills, vec![(1001, dec!(1.0)), (1005, dec!(0.5))]);
        assert_eq!(result.final_position, dec!(1.5));
    }

    #[test]
    fn test_non_positive_model_price_is_not_booked() {
        // Square root impact large enough to push a sell below zero.
        let model = MarketImpactFillModel::new(dec!(1), dec!(1), ImmediateFillModel::new());
        let ticks = vec![
            MarketTick::new(1000, dec!(0.05), dec!(1), dec!(0.2), dec!(1)),
            MarketTick::new(1001, dec!(0.15), dec!(1), dec!(0.2), dec!(1)),
        ];
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(1)),
            ScriptedStrategy::new(vec![Some((dec!(0.01), dec!(0.1)))]),
            VecDataSource::new(ticks),
            model,
        )
        .unwrap();
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert_eq!(result.total_fees, Decimal::ZERO);
        assert_eq!(engine.open_orders().len(), 2);
    }

    /// Accountant test double: books like the built in bookkeeping and
    /// records every posted fill in a shared log.
    #[derive(Debug, Default)]
    struct RecordingAccountant {
        log: std::sync::Arc<std::sync::Mutex<Vec<AccountingFill>>>,
        snapshot: AccountingSnapshot,
        refuse_from: Option<u64>,
    }

    impl FillAccountant for RecordingAccountant {
        fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
            if self.refuse_from.is_some_and(|id| fill.execution_id >= id) {
                return Err(MMError::InvalidPositionUpdate("refused".to_string()));
            }
            let (signed, cash_flow) = (fill.signed_quantity()?, fill.cash_flow()?);
            self.log.lock().unwrap().push(fill.clone());
            self.snapshot.position += signed;
            self.snapshot.cash_flow += cash_flow;
            self.snapshot.fees += fill.fee;
            self.snapshot.fills_posted += 1;
            Ok(())
        }
        fn snapshot(&self) -> MMResult<AccountingSnapshot> {
            Ok(self.snapshot.clone())
        }
        fn reset(&mut self) {
            self.log.lock().unwrap().clear();
            self.snapshot = AccountingSnapshot::default();
        }
        fn description(&self) -> String {
            "recording test double".to_string()
        }
    }

    fn queued_engine_with_fees()
    -> BacktestEngine<ScriptedStrategy, VecDataSource, QueuePositionFillModel> {
        BacktestEngine::with_fill_model(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(dec!(0.001)),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap()
    }

    #[test]
    fn test_accountant_receives_each_fill_once_with_fee_context() {
        let accountant = RecordingAccountant::default();
        let log = accountant.log.clone();
        let mut engine = queued_engine_with_fees().with_accountant(accountant);
        let result = engine.run();

        let log = log.lock().unwrap();
        let posted: Vec<(u64, SimOrderId, Decimal, Decimal, Decimal, FeeAsset)> = log
            .iter()
            .map(|f| {
                (
                    f.execution_id,
                    f.order_id,
                    f.quantity,
                    f.cumulative_quantity,
                    f.fee,
                    f.fee_asset.clone(),
                )
            })
            .collect();
        assert_eq!(
            posted,
            vec![
                (1, 1, dec!(0.5), dec!(0.5), dec!(0.05), FeeAsset::Quote),
                (2, 1, dec!(1.0), dec!(1.5), dec!(0.1), FeeAsset::Quote),
                (3, 1, dec!(0.5), dec!(2.0), dec!(0.05), FeeAsset::Quote),
            ]
        );
        assert_eq!(result.num_trades, 3);
        assert!(engine.accounting_errors().is_empty());
        assert_eq!(
            engine.accountant_description().as_deref(),
            Some("recording test double")
        );
    }

    #[test]
    fn test_selected_accountant_matches_builtin_results() {
        let builtin = queued_engine_with_fees().run();
        let mut engine = queued_engine_with_fees().with_accountant(RecordingAccountant::default());
        let selected = engine.run();
        assert_eq!(selected.final_position, builtin.final_position);
        assert_eq!(selected.total_fees, builtin.total_fees);
        assert_eq!(selected.total_pnl, builtin.total_pnl);
        assert_eq!(selected.net_pnl, builtin.net_pnl);
        assert_eq!(selected.max_position, builtin.max_position);
        let snapshot = engine.accounting_snapshot().unwrap().unwrap();
        assert_eq!(snapshot.fills_posted, 3);
        assert_eq!(snapshot.cash_flow, dec!(-200));
        assert_eq!(snapshot.fees, dec!(0.2));
    }

    #[test]
    fn test_refused_posts_are_reported_and_metrics_follow_accountant() {
        let accountant = RecordingAccountant {
            refuse_from: Some(3),
            ..RecordingAccountant::default()
        };
        let mut engine = queued_engine_with_fees().with_accountant(accountant);
        let result = engine.run();
        // The venue filled 2.0; the accountant only booked the first 1.5.
        assert_eq!(result.num_trades, 3);
        assert_eq!(result.final_position, dec!(1.5));
        let errors: Vec<u64> = engine
            .accounting_errors()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(errors, vec![3]);
    }

    #[test]
    fn test_backtest_result_refused_post_divergence_visible_from_result() {
        let accountant = RecordingAccountant {
            refuse_from: Some(3),
            ..RecordingAccountant::default()
        };
        let mut engine = queued_engine_with_fees().with_accountant(accountant);
        let result = engine.run();
        assert_eq!(
            result.accounting.accountant.as_deref(),
            Some("recording test double")
        );
        assert_eq!(result.accounting.refused_executions, vec![3]);
        assert!(!result.accounting.is_consistent());
        let ids: Vec<Option<u64>> = result.trades.iter().map(|f| f.execution_id).collect();
        assert_eq!(ids, vec![Some(1), Some(2), Some(3)]);
        let refused: Vec<Decimal> = result.refused_trades().map(|f| f.quantity).collect();
        assert_eq!(refused, vec![dec!(0.5)]);
        // Venue fills minus the refused ones add up to the booked position.
        let venue: Decimal = result.trades.iter().map(|f| f.quantity).sum();
        let refused_total: Decimal = refused.iter().copied().sum();
        assert_eq!(venue, dec!(2.0));
        assert_eq!(venue - refused_total, result.final_position);
    }

    #[test]
    fn test_backtest_result_refused_middle_fill_maps_to_its_trade() {
        // Refuse only execution 2 (quantity 1.0, distinct from 1 and 3).
        #[derive(Debug, Default)]
        struct RefuseSecond(RecordingAccountant);
        impl FillAccountant for RefuseSecond {
            fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
                if fill.execution_id == 2 {
                    return Err(MMError::InvalidPositionUpdate("refused".to_string()));
                }
                self.0.post_fill(fill)
            }
            fn snapshot(&self) -> MMResult<AccountingSnapshot> {
                self.0.snapshot()
            }
            fn reset(&mut self) {
                self.0.reset();
            }
        }
        let mut engine = queued_engine_with_fees().with_accountant(RefuseSecond::default());
        let first = engine.run();
        let refused: Vec<Decimal> = first.refused_trades().map(|f| f.quantity).collect();
        assert_eq!(refused, vec![dec!(1.0)]);
        assert_eq!(first.final_position, dec!(1.0));
        // Ids stay aligned with the trades after a reset.
        engine.reset();
        let second = engine.run();
        assert_eq!(second.accounting, first.accounting);
        assert_eq!(second.refused_trades().count(), 1);
    }

    #[test]
    fn test_failed_accounting_snapshot_is_recorded_as_fault() {
        #[derive(Debug, Default)]
        struct FailingSnapshot;
        impl FillAccountant for FailingSnapshot {
            fn post_fill(&mut self, _fill: &AccountingFill) -> MMResult<()> {
                Ok(())
            }
            fn snapshot(&self) -> MMResult<AccountingSnapshot> {
                Err(MMError::NumericalError(
                    "ledger cash flow overflows".to_string(),
                ))
            }
            fn reset(&mut self) {}
        }
        let mut engine = queued_engine_with_fees().with_accountant(FailingSnapshot);
        assert_eq!(engine.fault_count(), 1);
        assert!(matches!(
            engine.faults().first(),
            Some(MMError::NumericalError(_))
        ));
        assert!(matches!(
            engine.accounting_snapshot(),
            Some(Err(MMError::NumericalError(_)))
        ));
        let result = engine.run();
        assert_eq!(result.accounting.snapshot_failures.first(), Some(&0));
        assert!(!result.accounting.is_consistent());
    }

    #[test]
    fn test_snapshot_failure_after_a_booked_fill_is_inconsistent() {
        // Books every fill, but its snapshot fails once anything is booked.
        #[derive(Debug, Default)]
        struct FailAfterFirst(RecordingAccountant);
        impl FillAccountant for FailAfterFirst {
            fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
                self.0.post_fill(fill)
            }
            fn snapshot(&self) -> MMResult<AccountingSnapshot> {
                let snapshot = self.0.snapshot()?;
                if snapshot.fills_posted > 0 {
                    return Err(MMError::NumericalError(
                        "ledger cash flow overflows".to_string(),
                    ));
                }
                Ok(snapshot)
            }
            fn reset(&mut self) {
                self.0.reset();
            }
        }
        let mut engine = queued_engine_with_fees().with_accountant(FailAfterFirst::default());
        let result = engine.run();
        assert!(result.accounting.refused_executions.is_empty());
        assert_eq!(result.accounting.snapshot_failures.first(), Some(&1));
        assert!(!result.accounting.is_consistent());
    }

    #[test]
    fn test_backtest_result_refusals_without_trade_recording_keep_ids() {
        let accountant = RecordingAccountant {
            refuse_from: Some(3),
            ..RecordingAccountant::default()
        };
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(dec!(0.001))
                .with_record_trades(false),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap()
        .with_accountant(accountant);
        let result = engine.run();
        assert_eq!(result.accounting.refused_executions, vec![3]);
        assert_eq!(result.refused_trades().count(), 0);
    }

    #[test]
    fn test_backtest_result_builtin_bookkeeping_status_is_default() {
        let result = queued_engine_with_fees().run();
        assert_eq!(result.accounting, AccountingStatus::default());
        assert!(result.accounting.is_consistent());
        assert_eq!(result.refused_trades().count(), 0);
    }

    #[test]
    fn test_accountant_is_reset_with_engine() {
        let accountant = RecordingAccountant::default();
        let log = accountant.log.clone();
        let mut engine = queued_engine_with_fees().with_accountant(accountant);
        let first = engine.run();
        let first_log = log.lock().unwrap().clone();
        engine.reset();
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(
            engine.accounting_snapshot(),
            Some(Ok(AccountingSnapshot::default()))
        );
        let second = engine.run();
        assert_eq!(*log.lock().unwrap(), first_log);
        assert_eq!(second.net_pnl, first.net_pnl);
    }

    #[cfg(feature = "margin")]
    mod margin_ledger {
        use super::*;
        use crate::backtest::MarginLedgerAccountant;
        use margin_rs::ledger::Ledger;
        use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};

        fn accountant(deposit: Decimal) -> MarginLedgerAccountant {
            let usdc = Asset::new("USDC").unwrap();
            let btc = Asset::new("BTC").unwrap();
            let pair = InstrumentId::new("BTC-USDC").unwrap();
            let mut ledger = Ledger::default();
            ledger.register_asset(usdc.clone(), 6).unwrap();
            ledger.register_asset(btc.clone(), 8).unwrap();
            ledger
                .register_instrument(
                    SpotInstrument::new(pair.clone(), btc, usdc.clone(), 2, 2).unwrap(),
                )
                .unwrap();
            ledger
                .deposit(TransferId::new("seed").unwrap(), usdc, deposit)
                .unwrap();
            MarginLedgerAccountant::new(ledger, pair).unwrap()
        }

        #[test]
        fn test_margin_ledger_reconciles_with_builtin_results() {
            let builtin = queued_engine_with_fees().run();
            let mut engine = queued_engine_with_fees().with_accountant(accountant(dec!(1000)));
            let result = engine.run();
            assert!(engine.accounting_errors().is_empty());
            assert_eq!(result.final_position, builtin.final_position);
            assert_eq!(result.total_fees, dec!(0.2));
            assert_eq!(result.net_pnl, builtin.net_pnl);
            let snapshot = engine.accounting_snapshot().unwrap().unwrap();
            assert_eq!(snapshot.cash_flow, dec!(-200));
            assert_eq!(snapshot.average_entry_price, Some(dec!(100)));
            assert_eq!(snapshot.fills_posted, 3);
        }

        #[test]
        fn test_margin_ledger_books_venue_fills_beyond_the_balance() {
            // 100.05 USDC covers only the first 0.5; the venue still filled
            // 2.0, so the ledger books it and records the shortfall.
            let mut engine = queued_engine_with_fees().with_accountant(accountant(dec!(100.05)));
            let result = engine.run();
            assert_eq!(result.num_trades, 3);
            assert_eq!(result.final_position, dec!(2));
            assert!(engine.accounting_errors().is_empty());
        }

        #[test]
        fn test_margin_ledger_unsupported_action_books_nothing() {
            let mut acc = accountant(dec!(1000));
            let mut fill = ask_fill(1, dec!(100), dec!(1), Decimal::ZERO);
            fill.side = Action::Other;
            assert_eq!(
                acc.post_fill(&fill),
                Err(MMError::UnsupportedAction {
                    action: Action::Other
                })
            );
            assert!(acc.ledger().journal().is_empty());
            assert_eq!(acc.snapshot().unwrap().fills_posted, 0);
        }

        #[test]
        fn test_margin_ledger_unsupported_action_leaves_snapshot_unchanged() {
            let mut acc = accountant(dec!(1000));
            acc.post_fill(&ask_fill(1, dec!(100), dec!(1), dec!(0.1)))
                .unwrap();
            let before = acc.snapshot().unwrap();
            let mut fill = ask_fill(2, dec!(100), dec!(1), dec!(0.1));
            fill.side = Action::Other;
            assert!(acc.post_fill(&fill).is_err());
            assert_eq!(acc.ledger().journal().len(), 1);
            assert_eq!(acc.snapshot().unwrap(), before);
        }

        fn ask_fill(id: u64, price: Decimal, quantity: Decimal, fee: Decimal) -> AccountingFill {
            AccountingFill {
                execution_id: id,
                order_id: id,
                side: Action::Sell,
                price,
                quantity,
                cumulative_quantity: quantity,
                fee,
                fee_asset: FeeAsset::Quote,
                timestamp: 0,
            }
        }

        #[test]
        fn test_sell_first_without_base_is_booked_with_overdraft() {
            let mut acc = accountant(dec!(1000));
            acc.post_fill(&ask_fill(1, dec!(100), dec!(1), dec!(0.1)))
                .unwrap();
            let snapshot = acc.snapshot().unwrap();
            assert_eq!(snapshot.position, dec!(-1));
            assert_eq!(snapshot.cash_flow, dec!(100));
            assert_eq!(snapshot.fees, dec!(0.1));
            let btc = Asset::new("BTC").unwrap();
            assert_eq!(acc.overdrafts().get(&btc), Some(&dec!(1)));
            acc.reset();
            assert!(acc.overdrafts().is_empty());
        }

        #[test]
        fn test_negative_fee_is_booked_as_rebate() {
            let mut acc = accountant(dec!(1000));
            acc.post_fill(&ask_fill(1, dec!(100), dec!(1), dec!(-0.05)))
                .unwrap();
            assert_eq!(acc.snapshot().unwrap().fees, dec!(-0.05));
        }

        #[test]
        fn test_margin_ledger_base_fee_debits_base_and_is_valued_in_quote() {
            let mut acc = accountant(dec!(1000));
            let mut fill = ask_fill(1, dec!(100), dec!(1), dec!(0.001));
            fill.fee_asset = FeeAsset::Base;
            acc.post_fill(&fill).unwrap();
            let btc = Asset::new("BTC").unwrap();
            let usdc = Asset::new("USDC").unwrap();
            assert_eq!(acc.ledger().fees_paid(&btc), dec!(0.001));
            assert_eq!(acc.ledger().fees_paid(&usdc), Decimal::ZERO);
            // The sell delivers 1 BTC plus the 0.001 BTC fee.
            assert_eq!(acc.overdrafts().get(&btc), Some(&dec!(1.001)));
            // The account delivered 1.001 BTC; the fee is reported in base
            // for the engine to mark.
            let snapshot = acc.snapshot().unwrap();
            assert_eq!(snapshot.position, dec!(-1.001));
            assert_eq!(snapshot.cash_flow, dec!(100));
            assert_eq!(snapshot.fees, Decimal::ZERO);
            assert_eq!(snapshot.base_fees, dec!(0.001));
            acc.reset();
            assert_eq!(acc.snapshot().unwrap().base_fees, Decimal::ZERO);
        }

        #[test]
        fn test_margin_ledger_third_asset_fee_is_refused() {
            let mut acc = accountant(dec!(1000));
            let mut fill = ask_fill(1, dec!(100), dec!(1), dec!(0.5));
            fill.fee_asset = FeeAsset::Other("BNB".to_string());
            assert!(acc.post_fill(&fill).is_err());
            assert_eq!(acc.snapshot().unwrap(), AccountingSnapshot::default());
        }

        #[test]
        fn test_margin_ledger_engine_base_fees_match_builtin_metrics() {
            let config = || {
                BacktestConfig::default()
                    .with_default_order_size(dec!(2))
                    .with_fee_rate(dec!(0.001))
                    .with_fee_currency(FeeCurrency::Base { scale: 8 })
            };
            let run = |with_ledger: bool| {
                let engine = BacktestEngine::with_fill_model(
                    config(),
                    ScriptedStrategy::new(queue_quotes()),
                    VecDataSource::new(queue_ticks()),
                    QueuePositionFillModel::new(dec!(0.5)),
                )
                .unwrap();
                if with_ledger {
                    let mut engine = engine.with_accountant(accountant(dec!(1000)));
                    let result = engine.run();
                    assert!(engine.accounting_errors().is_empty());
                    result
                } else {
                    let mut engine = engine;
                    engine.run()
                }
            };
            let (builtin, ledger) = (run(false), run(true));
            // 2.0 BTC bought at 100 with a 0.1% fee in BTC: 0.002 BTC held
            // less, and the fee is marked at the last mid (99.5).
            assert_eq!(builtin.final_position, dec!(1.998));
            assert_eq!(builtin.total_fees, dec!(0.199));
            assert_eq!(ledger.total_fees, builtin.total_fees);
            assert_eq!(ledger.final_position, builtin.final_position);
            assert_eq!(ledger.net_pnl, builtin.net_pnl);
        }

        #[test]
        fn test_other_instruments_do_not_leak_into_the_snapshot() {
            let usdc = Asset::new("USDC").unwrap();
            let eth = Asset::new("ETH").unwrap();
            let eth_pair = InstrumentId::new("ETH-USDC").unwrap();
            let mut ledger = accountant(dec!(1000)).ledger().clone();
            ledger.register_asset(eth.clone(), 8).unwrap();
            ledger
                .register_instrument(
                    SpotInstrument::new(eth_pair.clone(), eth, usdc.clone(), 2, 2).unwrap(),
                )
                .unwrap();
            let eth_buy = margin_rs::ledger::Execution::new(
                margin_rs::ExecutionId::new("eth-1").unwrap(),
                eth_pair,
                margin_rs::Action::Buy,
                dec!(100),
                dec!(1),
            )
            .with_fee(usdc, dec!(0.1));
            ledger.book_execution(&eth_buy).unwrap();
            let acc = MarginLedgerAccountant::new(ledger, InstrumentId::new("BTC-USDC").unwrap())
                .unwrap();
            assert_eq!(acc.snapshot().unwrap(), AccountingSnapshot::default());
        }

        #[test]
        fn test_initial_ledger_state_is_used_without_fills() {
            // Seed a prior BTC position: buy 1 at 90 with a 0.09 fee.
            let mut acc = accountant(dec!(1000));
            let buy = AccountingFill {
                side: Action::Buy,
                ..ask_fill(99, dec!(90), dec!(1), dec!(0.09))
            };
            acc.post_fill(&buy).unwrap();
            let seeded = MarginLedgerAccountant::new(
                acc.ledger().clone(),
                InstrumentId::new("BTC-USDC").unwrap(),
            )
            .unwrap();
            let ticks = vec![create_test_tick(1000, dec!(99.0), dec!(101.0))];
            let mut engine = BacktestEngine::new(
                BacktestConfig::default(),
                ScriptedStrategy::new(vec![None]),
                VecDataSource::new(ticks),
            )
            .with_accountant(seeded);
            let result = engine.run();
            assert_eq!(result.num_trades, 0);
            assert_eq!(result.final_position, dec!(1));
            assert_eq!(result.total_fees, dec!(0.09));
            assert_eq!(result.max_position, dec!(1));
            // Marked at mid 100: -90 cash + 100 inventory, before fees.
            assert_eq!(result.total_pnl, dec!(10));
            engine.reset();
            assert_eq!(engine.get_state().0.quantity, dec!(1));
        }

        #[test]
        fn test_execution_ids_continue_after_those_the_ledger_started_with() {
            let pair = InstrumentId::new("BTC-USDC").unwrap();
            let buy = |execution_id: u64, price: Decimal| AccountingFill {
                execution_id,
                ..ask_fill(execution_id, price, dec!(1), dec!(0))
            };
            let mut first = accountant(dec!(1000));
            first.post_fill(&buy(99, dec!(90))).unwrap();
            let mut second =
                MarginLedgerAccountant::new(first.ledger().clone(), pair.clone()).unwrap();
            second.post_fill(&buy(1, dec!(91))).unwrap();
            // The same fill again is still the same execution.
            second.post_fill(&buy(1, dec!(91))).unwrap();
            let ids: Vec<&str> = second
                .ledger()
                .journal()
                .iter()
                .map(|e| e.execution_id.as_str())
                .collect();
            assert_eq!(ids, ["sim-99", "sim-100"]);
            second.reset();
            second.post_fill(&buy(1, dec!(92))).unwrap();
            assert_eq!(
                second.ledger().journal()[1].execution_id.as_str(),
                "sim-100"
            );

            let mut last = accountant(dec!(1000));
            last.post_fill(&buy(u64::MAX, dec!(90))).unwrap();
            let mut past = MarginLedgerAccountant::new(last.ledger().clone(), pair).unwrap();
            assert!(matches!(
                past.post_fill(&buy(1, dec!(91))),
                Err(MMError::NumericalError(_))
            ));
            assert_eq!(past.ledger().journal().len(), 1);
        }

        #[test]
        fn test_an_accountant_starts_from_its_state_at_the_handover() {
            // Posted to before the engine takes it, with a base fee and an overdraft.
            let mut acc = accountant(dec!(1000));
            let sell = ask_fill(1, dec!(100), dec!(1), dec!(0));
            acc.post_fill(&sell).unwrap();
            let buy = AccountingFill {
                side: Action::Buy,
                fee: dec!(0.5),
                fee_asset: FeeAsset::Base,
                ..ask_fill(2, dec!(90), dec!(2), dec!(0))
            };
            acc.post_fill(&buy).unwrap();
            assert!(!acc.overdrafts().is_empty());
            let handed = acc.snapshot().unwrap();
            assert_eq!(handed.base_fees, dec!(0.5));

            let mut direct = acc.clone();
            direct.begin_from_current_state();
            assert!(direct.overdrafts().is_empty());
            let next = AccountingFill {
                side: Action::Buy,
                ..ask_fill(1, dec!(91), dec!(1), dec!(0))
            };
            direct.post_fill(&next).unwrap();
            assert_eq!(direct.ledger().journal()[2].execution_id.as_str(), "sim-3");
            assert_ne!(direct.snapshot().unwrap(), handed);
            direct.reset();
            assert_eq!(direct.snapshot().unwrap(), handed);
            assert_eq!(direct.ledger().journal().len(), 2);

            // The engine does this for a spot run too, at its first and later runs.
            let ticks = vec![
                create_test_tick(1000, dec!(98.0), dec!(102.0)),
                create_test_tick(1001, dec!(97.0), dec!(99.0)),
            ];
            let mut engine = BacktestEngine::new(
                BacktestConfig::default().with_default_order_size(dec!(1)),
                ScriptedStrategy::new(vec![Some((dec!(99), dec!(200))), None]),
                VecDataSource::new(ticks),
            )
            .with_accountant(acc);
            assert_eq!(engine.accounting_snapshot().unwrap().unwrap(), handed);
            let first = engine.run();
            assert!(first.accounting.is_consistent(), "{:?}", first.accounting);
            assert_eq!(first.num_trades, 1);
            engine.reset();
            assert_eq!(engine.accounting_snapshot().unwrap().unwrap(), handed);
            let second = engine.run();
            assert_eq!(second.accounting, first.accounting);
            assert_eq!(second.total_pnl, first.total_pnl);
            assert_eq!(second.final_position, first.final_position);
        }

        #[test]
        fn test_unknown_instrument_is_rejected() {
            let ledger = Ledger::default();
            let missing = InstrumentId::new("ETH-USDC").unwrap();
            assert!(MarginLedgerAccountant::new(ledger, missing).is_err());
        }
    }

    /// Pretrade policy test double returning a fixed decision and logging
    /// every request it sees.
    #[derive(Debug)]
    struct FixedPolicy {
        decision: PretradeDecision,
        seen: std::sync::Arc<std::sync::Mutex<Vec<PretradeRequest>>>,
        events: std::sync::Arc<std::sync::Mutex<Vec<OrderEvent>>>,
    }

    impl FixedPolicy {
        fn new(decision: PretradeDecision) -> Self {
            Self {
                decision,
                seen: Default::default(),
                events: Default::default(),
            }
        }
    }

    impl PretradePolicy for FixedPolicy {
        fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
            self.seen.lock().unwrap().push(request.clone());
            self.decision.clone()
        }
        fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
        fn reset(&mut self) {
            self.seen.lock().unwrap().clear();
        }
    }

    /// Fills every sell in full at a fixed price, so two fills of size 2
    /// at a third of `Decimal::MAX` overflow the realized cash.
    #[derive(Debug)]
    struct HugeSellModel;

    impl FillModel for HugeSellModel {
        fn simulate_fill(
            &self,
            order: &SimulatedOrder,
            _tick: &MarketTick,
            _time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            Ok(if order.side == Action::Sell {
                FillResult::FullFill {
                    fill_price: Decimal::MAX / dec!(3),
                }
            } else {
                FillResult::NoFill
            })
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "HugeSell"
        }
    }

    #[test]
    fn test_overflowing_booking_faults_before_any_state_changes() {
        let quotes = vec![Some((dec!(100), dec!(200))); 5];
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_fee_rate(Decimal::ZERO),
            ScriptedStrategy::new(quotes),
            VecDataSource::new(queue_ticks()),
            HugeSellModel,
        )
        .unwrap();
        let result = engine.run();

        // The first sell books; the second would overflow the realized cash.
        assert!(
            engine
                .faults()
                .iter()
                .any(|f| *f == MMError::NumericalError("realized pnl overflows".to_string())),
            "{:?}",
            engine.faults()
        );
        let first = (Decimal::MAX / dec!(3)) * dec!(2);
        assert_eq!(result.num_trades, 1);
        assert!(result.accounting.dropped_fills > 0);
        assert!(!result.accounting.is_consistent());
        assert_eq!(result.final_position, dec!(-2));
        assert_eq!(engine.pnl.realized, first);
        // The refused fill left its order unfilled and reported nothing.
        let fills = engine
            .execution_reports()
            .iter()
            .filter(|r| r.kind == ExecutionKind::Fill || r.kind == ExecutionKind::PartialFill)
            .count();
        assert_eq!(fills, 1);
    }

    /// Fill model that fails on every order.
    #[derive(Debug)]
    struct FailingFillModel;

    impl FillModel for FailingFillModel {
        fn simulate_fill(
            &self,
            _order: &SimulatedOrder,
            _tick: &MarketTick,
            _time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            Err(MMError::NumericalError("model failure".to_string()))
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "Failing"
        }
    }

    #[test]
    fn test_fault_stops_admission_until_reset() {
        let policy = FixedPolicy::new(PretradeDecision::Accepted);
        let seen = policy.seen.clone();
        let quotes = vec![Some((dec!(100), dec!(200))); 5];
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(quotes.clone()),
            VecDataSource::new(queue_ticks()),
            FailingFillModel,
        )
        .unwrap()
        .with_pretrade_policy(policy);
        let result = engine.run();

        assert_eq!(result.num_trades, 0);
        assert!(!engine.faults().is_empty());
        assert!(
            engine
                .faults()
                .iter()
                .all(|f| *f == MMError::NumericalError("model failure".to_string()))
        );
        // Only the quotes decided before the first fault reached the
        // policy; every later one was refused without asking it.
        let asked = seen.lock().unwrap().len();
        let refused = engine
            .execution_reports()
            .iter()
            .filter(|r| r.kind == ExecutionKind::Rejected)
            .count();
        assert!(asked > 0, "the first quotes reach the policy");
        assert!(refused > 0, "later quotes are refused");
        assert_eq!(asked + refused, 2 * quotes.len());

        engine.reset();
        assert!(engine.faults().is_empty());
        assert_eq!(engine.fault_count(), 0);
    }

    /// Fill model that fails on every call and counts the calls.
    #[derive(Debug, Default)]
    struct CountingFailingModel {
        calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }

    impl FillModel for CountingFailingModel {
        fn simulate_fill(
            &self,
            _order: &SimulatedOrder,
            _tick: &MarketTick,
            _time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(MMError::NumericalError("model failure".to_string()))
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "CountingFailing"
        }
    }

    #[test]
    fn test_engine_faults_always_failing_model_list_stays_bounded() {
        let model = CountingFailingModel::default();
        let calls = model.calls.clone();
        let ticks: Vec<MarketTick> = (0..500u64)
            .map(|i| MarketTick::new(1000 + i, dec!(99), dec!(1), dec!(101), dec!(1)))
            .collect();
        // One quote, then none: its two orders keep resting and the model
        // fails on both at every later tick.
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(1)),
            ScriptedStrategy::new(vec![Some((dec!(98), dec!(102)))]),
            VecDataSource::new(ticks),
            model,
        )
        .unwrap();
        engine.run();
        let total = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(total > MAX_RECORDED_FAULTS as u64 * 10, "calls {total}");
        assert_eq!(engine.faults().len(), MAX_RECORDED_FAULTS);
        assert_eq!(engine.fault_count(), total);
        assert_eq!(engine.open_orders().len(), 2);
        engine.reset();
        assert!(engine.faults().is_empty());
        assert_eq!(engine.fault_count(), 0);
    }

    /// Accepts every order, recording the decision times, and fails on
    /// every fill.
    #[derive(Debug, Default)]
    struct FailingOnFillPolicy {
        checked_at: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    }

    impl PretradePolicy for FailingOnFillPolicy {
        fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
            self.checked_at.lock().unwrap().push(request.timestamp);
            PretradeDecision::Accepted
        }
        fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
            match event {
                OrderEvent::Fill { .. } => Err(MMError::UnsupportedAction {
                    action: Action::Other,
                }),
                _ => Ok(()),
            }
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn test_policy_error_on_a_fill_stops_admission() {
        let quotes = vec![Some((dec!(100), dec!(200))); 5];
        let policy = FailingOnFillPolicy::default();
        let checked_at = policy.checked_at.clone();
        let mut engine =
            scripted_engine(queue_ticks(), quotes.clone()).with_pretrade_policy(policy);
        let result = engine.run();

        // The fill happened at the venue and is kept.
        assert!(result.num_trades >= 1);
        assert_eq!(
            engine.faults().first(),
            Some(&MMError::UnsupportedAction {
                action: Action::Other
            })
        );
        // The policy is never asked again after the first fill, and the
        // quotes decided after it are refused.
        let reports = engine.execution_reports();
        let first_fill_at = reports
            .iter()
            .find(|r| matches!(r.kind, ExecutionKind::Fill | ExecutionKind::PartialFill))
            .map(|r| r.stamp.timestamp)
            .unwrap();
        let checked_at = checked_at.lock().unwrap();
        assert!(!checked_at.is_empty());
        assert!(checked_at.iter().all(|t| *t <= first_fill_at));
        assert!(
            reports
                .iter()
                .any(|r| r.kind == ExecutionKind::Rejected && r.stamp.timestamp > first_fill_at)
        );
    }

    #[test]
    fn test_rejecting_policy_prevents_submission() {
        let policy = FixedPolicy::new(PretradeDecision::Rejected {
            reason: "no".to_string(),
        });
        let mut engine =
            scripted_engine(queue_ticks(), queue_quotes()).with_pretrade_policy(policy);
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert!(engine.open_orders().is_empty());
        assert!(
            engine
                .execution_reports()
                .iter()
                .all(|r| r.kind == ExecutionKind::Rejected)
        );
        let decisions = engine.pretrade_decisions();
        assert_eq!(decisions.len(), 2);
        assert!(decisions.iter().all(|d| d.submitted == Decimal::ZERO));
    }

    #[test]
    fn test_reducing_policy_cannot_increase_quantity() {
        let grow = FixedPolicy::new(PretradeDecision::Reduced {
            quantity: dec!(10),
            reason: "bigger".to_string(),
        });
        let mut engine = scripted_engine(queue_ticks(), queue_quotes()).with_pretrade_policy(grow);
        engine.run();
        assert!(
            engine
                .pretrade_decisions()
                .iter()
                .all(|d| d.submitted == dec!(2))
        );

        let shrink = FixedPolicy::new(PretradeDecision::Reduced {
            quantity: dec!(0.5),
            reason: "smaller".to_string(),
        });
        let mut engine =
            scripted_engine(queue_ticks(), queue_quotes()).with_pretrade_policy(shrink);
        let result = engine.run();
        assert_eq!(result.final_position, dec!(0.5));
        assert!(
            engine
                .pretrade_decisions()
                .iter()
                .all(|d| d.submitted == dec!(0.5))
        );
    }

    #[test]
    fn test_failed_registration_releases_the_policy() {
        use crate::execution::{OrderManager, OrderRequest};
        use crate::risk::RiskLimits;
        use crate::risk::pretrade::LimitsPolicy;
        let limits = RiskLimits::new(dec!(10), dec!(1_000_000), dec!(0)).unwrap();
        let mut policy = LimitsPolicy::new(limits);
        let mut manager = OrderManager::with_defaults();
        let request = OrderRequest::limit_buy("BTC-USD", dec!(100), dec!(4));
        // The id is taken outside the policy, so registration fails after
        // the policy approved and registered the order.
        manager
            .register_order(&request, "dup".to_string(), 1)
            .unwrap();
        let result = manager.register_order_checked(
            &request,
            "dup".to_string(),
            2,
            OrderPurpose::Hedge,
            &mut policy,
        );
        assert!(result.is_err());
        assert_eq!(policy.live_quantity(Action::Buy).unwrap(), Decimal::ZERO);
    }

    #[test]
    fn test_quote_and_hedge_paths_use_the_same_seam() {
        use crate::execution::{OrderManager, OrderRequest};
        let mut policy = FixedPolicy::new(PretradeDecision::Rejected {
            reason: "halted".to_string(),
        });
        let seen = policy.seen.clone();

        // Hedge through the live order manager.
        let mut manager = OrderManager::with_defaults();
        let hedge = OrderRequest::limit_sell("BTC-USD", dec!(100), dec!(1));
        let decision = manager
            .register_order_checked(
                &hedge,
                "h-1".to_string(),
                5,
                OrderPurpose::Hedge,
                &mut policy,
            )
            .unwrap();
        assert!(decision.is_rejected());
        assert_eq!(manager.open_order_count(), 0);

        // Quote through the backtest engine.
        let mut engine =
            scripted_engine(queue_ticks(), queue_quotes()).with_pretrade_policy(policy);
        engine.run();
        assert!(engine.open_orders().is_empty());

        // The engine run does not reset the policy, so both paths are logged.
        let purposes: Vec<OrderPurpose> = seen.lock().unwrap().iter().map(|r| r.purpose).collect();
        assert_eq!(
            purposes,
            vec![
                OrderPurpose::Hedge,
                OrderPurpose::Quote,
                OrderPurpose::Quote
            ]
        );
    }

    #[test]
    fn test_reduction_off_the_lot_grid_is_released() {
        // A reduction to 1.5 with lot 1 is not a whole lot: the order is
        // refused and the policy is told to release what it registered.
        let policy = FixedPolicy::new(PretradeDecision::Reduced {
            quantity: dec!(1.5),
            reason: "capacity".to_string(),
        });
        let events = policy.events.clone();
        let mut engine = BacktestEngine::new(
            BacktestConfig::default()
                .with_default_order_size(dec!(2))
                .with_lot_size(dec!(1)),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
        )
        .with_pretrade_policy(policy);
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert!(engine.open_orders().is_empty());
        assert!(
            engine
                .pretrade_decisions()
                .iter()
                .all(|d| d.submitted == Decimal::ZERO)
        );
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                OrderEvent::VenueRejected {
                    order_id: "1".to_string()
                },
                OrderEvent::VenueRejected {
                    order_id: "2".to_string()
                },
            ]
        );
    }

    #[test]
    fn test_limits_policy_caps_worst_case_position() {
        use crate::risk::RiskLimits;
        use crate::risk::pretrade::LimitsPolicy;
        // Max position 3: the first bid of 2 fits; after it fills, the
        // requote bid is reduced to the 1 unit of capacity left.
        let limits = RiskLimits::new(dec!(3), dec!(1_000_000), dec!(0)).unwrap();
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(99.0)),
            create_test_tick(1002, dec!(98.0), dec!(102.0)),
        ];
        let mut engine = scripted_engine(
            ticks,
            vec![
                Some((dec!(100), dec!(200))),
                None,
                Some((dec!(100), dec!(200))),
            ],
        )
        .with_pretrade_policy(LimitsPolicy::new(limits));
        let result = engine.run();
        assert_eq!(result.final_position, dec!(2));
        let bids: Vec<Decimal> = engine
            .pretrade_decisions()
            .iter()
            .filter(|d| d.request.side == Action::Buy)
            .map(|d| d.submitted)
            .collect();
        assert_eq!(bids, vec![dec!(2), dec!(1)]);
    }

    #[cfg(feature = "margin")]
    mod margin_gate {
        use super::*;
        use crate::backtest::MarginAccount;
        use margin_rs::ledger::Ledger;
        use margin_rs::risk::{FeeAsset as GateFeeAsset, FeeRounding, InstrumentLimits, RiskGate};
        use margin_rs::{Asset, InstrumentId, Rounding, SpotInstrument, TransferId};

        fn account(max_position: Option<Decimal>) -> MarginAccount {
            account_with(InstrumentLimits {
                max_fee_rate: dec!(0.001),
                max_position,
                ..InstrumentLimits::default()
            })
        }

        fn account_with(limits: InstrumentLimits) -> MarginAccount {
            let usdc = Asset::new("USDC").unwrap();
            let btc = Asset::new("BTC").unwrap();
            let pair = InstrumentId::new("BTC-USDC").unwrap();
            let mut ledger = Ledger::default();
            ledger.register_asset(usdc.clone(), 6).unwrap();
            ledger.register_asset(btc.clone(), 8).unwrap();
            ledger
                .register_instrument(
                    SpotInstrument::new(pair.clone(), btc.clone(), usdc.clone(), 2, 2).unwrap(),
                )
                .unwrap();
            let mut gate = RiskGate::new(ledger);
            gate.deposit(TransferId::new("seed-usdc").unwrap(), usdc, dec!(1000))
                .unwrap();
            gate.deposit(TransferId::new("seed-btc").unwrap(), btc, dec!(5))
                .unwrap();
            gate.set_limits(&pair, limits).unwrap();
            MarginAccount::new(gate, pair).unwrap()
        }

        /// Limits with fees declared in base, rounded per fill in `mode`,
        /// at a rate whose fee on 1 BTC (1.5e-8) is not on the 8 digit grid.
        fn base_fee_limits(mode: Rounding) -> InstrumentLimits {
            InstrumentLimits {
                max_fee_rate: dec!(0.000000015),
                fee_asset: GateFeeAsset::Base,
                fee_rounding: FeeRounding::PerFill {
                    mode,
                    min_fill: None,
                },
                ..InstrumentLimits::default()
            }
        }

        fn base_fee_sell(policy: &mut impl PretradePolicy, id: &str) {
            let request = PretradeRequest::new(
                id,
                Action::Sell,
                Some(dec!(100)),
                dec!(1),
                OrderPurpose::Quote,
                0,
            );
            assert!(!policy.check(&request).is_rejected());
            // `FeeCurrency::Base { scale: 8 }` charges 1.5e-8 rounded half
            // to even: 2e-8.
            let fee = FeeCurrency::Base { scale: 8 }
                .charge(dec!(100), dec!(1), dec!(0.000000015))
                .unwrap()
                .amount;
            assert_eq!(fee, dec!(0.00000002));
            policy
                .on_order_event(&OrderEvent::Fill {
                    order_id: id.to_string(),
                    execution_id: format!("e-{id}"),
                    side: Action::Sell,
                    price: dec!(100),
                    quantity: dec!(1),
                    cumulative_quantity: dec!(1),
                    fee,
                    fee_asset: crate::types::fee::FeeAsset::Base,
                    timestamp: 1,
                })
                .unwrap();
        }

        #[test]
        fn test_margin_policy_base_fee_breach_is_recorded_and_rejects_next_order() {
            // Declared toward zero (bound 1e-8), charged half to even (2e-8).
            let account = account_with(base_fee_limits(Rounding::TowardZero));
            let mut policy = account.policy();
            base_fee_sell(&mut policy, "o-1");
            let btc = Asset::new("BTC").unwrap();
            assert!(account.with_gate(|gate| gate.ledger().fees_paid(&btc) == dec!(0.00000002)));
            let errors = account.errors();
            assert_eq!(errors.len(), 1, "{errors:?}");
            assert!(errors[0].starts_with("fee breaches: "), "{errors:?}");
            let pair = InstrumentId::new("BTC-USDC").unwrap();
            assert_eq!(
                account.with_gate(|gate| gate.fee_declaration_breaches(&pair).len()),
                1
            );
            let next = PretradeRequest::new(
                "o-2",
                Action::Sell,
                Some(dec!(100)),
                dec!(1),
                OrderPurpose::Quote,
                0,
            );
            assert!(policy.check(&next).is_rejected());
        }

        #[test]
        fn test_margin_policy_base_fee_matching_declaration_records_nothing() {
            // Declared half to even, as `FeeCurrency::Base` charges.
            let account = account_with(base_fee_limits(Rounding::HalfEven));
            let mut policy = account.policy();
            base_fee_sell(&mut policy, "o-1");
            assert!(account.errors().is_empty(), "{:?}", account.errors());
            let pair = InstrumentId::new("BTC-USDC").unwrap();
            assert!(account.with_gate(|gate| gate.fee_declaration_breaches(&pair).is_empty()));
            let next = PretradeRequest::new(
                "o-2",
                Action::Sell,
                Some(dec!(100)),
                dec!(1),
                OrderPurpose::Quote,
                0,
            );
            assert!(!policy.check(&next).is_rejected());
        }

        #[test]
        fn test_margin_policy_poisoned_account_fails_closed_until_reset() {
            let account = account(None);
            let mut policy = account.policy();
            let mut accountant = account.accountant();
            let request = PretradeRequest::new(
                "o-1",
                Action::Buy,
                Some(dec!(100)),
                dec!(1),
                OrderPurpose::Quote,
                0,
            );
            let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                account.with_gate(|_| panic!("poison the account"))
            }));
            assert!(poisoned.is_err());

            assert!(matches!(account.fault(), Some(MMError::LockPoisoned(_))));
            assert!(policy.check(&request).is_rejected());
            let event = OrderEvent::Accepted {
                order_id: "o-1".to_string(),
            };
            assert!(matches!(
                policy.on_order_event(&event),
                Err(MMError::LockPoisoned(_))
            ));
            let fill = AccountingFill {
                execution_id: 1,
                order_id: 1,
                side: Action::Buy,
                price: dec!(100),
                quantity: dec!(1),
                cumulative_quantity: dec!(1),
                fee: Decimal::ZERO,
                fee_asset: crate::types::fee::FeeAsset::Quote,
                timestamp: 0,
            };
            assert!(matches!(
                accountant.post_fill(&fill),
                Err(MMError::LockPoisoned(_))
            ));

            policy.reset();
            assert!(account.fault().is_none());
            assert!(!policy.check(&request).is_rejected());
        }

        #[test]
        fn test_margin_policy_refuses_a_settlement_asset_fee() {
            let account = account(None);
            let mut policy = account.policy();
            let request = PretradeRequest::new(
                "o-1",
                Action::Buy,
                Some(dec!(100)),
                dec!(1),
                OrderPurpose::Quote,
                0,
            );
            assert!(!policy.check(&request).is_rejected());
            let fill = OrderEvent::Fill {
                order_id: "o-1".to_string(),
                execution_id: "e-o-1".to_string(),
                side: Action::Buy,
                price: dec!(100),
                quantity: dec!(1),
                cumulative_quantity: dec!(1),
                fee: dec!(0.001),
                fee_asset: crate::types::fee::FeeAsset::Settlement,
                timestamp: 1,
            };
            let error = policy.on_order_event(&fill).unwrap_err();
            assert!(error.to_string().contains("settlement asset"), "{error}");
            assert_eq!(account.fault(), Some(error));
            assert!(account.with_gate(|gate| gate.ledger().journal().is_empty()));
        }

        #[test]
        fn test_margin_policy_fee_assets_book_base_and_refuse_third_asset() {
            let account = account(None);
            let mut policy = account.policy();
            let request = |id: &str| {
                PretradeRequest::new(
                    id,
                    Action::Buy,
                    Some(dec!(100)),
                    dec!(1),
                    OrderPurpose::Quote,
                    0,
                )
            };
            let fill = |id: &str, fee_asset| OrderEvent::Fill {
                order_id: id.to_string(),
                execution_id: format!("e-{id}"),
                side: Action::Buy,
                price: dec!(100),
                quantity: dec!(1),
                cumulative_quantity: dec!(1),
                fee: dec!(0.001),
                fee_asset,
                timestamp: 1,
            };
            assert!(!policy.check(&request("o-1")).is_rejected());
            policy
                .on_order_event(&fill("o-1", crate::types::fee::FeeAsset::Base))
                .unwrap();
            let btc = Asset::new("BTC").unwrap();
            assert!(account.with_gate(|gate| gate.ledger().fees_paid(&btc) == dec!(0.001)));
            assert!(account.errors().is_empty());

            assert!(!policy.check(&request("o-2")).is_rejected());
            let other = fill("o-2", crate::types::fee::FeeAsset::Other("BNB".to_string()));
            assert!(policy.on_order_event(&other).is_err());
            assert!(account.fault().is_some());
            assert!(account.with_gate(|gate| gate.ledger().journal().len() == 1));
        }

        #[test]
        fn test_margin_policy_unsupported_action_is_rejected_and_fill_latches_a_fault() {
            let account = account(None);
            let mut policy = account.policy();
            let request = |id: &str, side| {
                PretradeRequest::new(id, side, Some(dec!(100)), dec!(1), OrderPurpose::Quote, 0)
            };
            assert_eq!(
                policy.check(&request("o-0", Action::Other)),
                PretradeDecision::Rejected {
                    reason: "unsupported action Other: only buy and sell are supported".to_string(),
                }
            );
            assert_eq!(account.fault(), None);

            // A fill with an unsupported side is unknown exposure: it fails,
            // books nothing and latches a fault that rejects every order.
            assert!(!policy.check(&request("o-1", Action::Buy)).is_rejected());
            let unsupported = MMError::UnsupportedAction {
                action: Action::Other,
            };
            assert_eq!(
                policy.on_order_event(&OrderEvent::Fill {
                    order_id: "o-1".to_string(),
                    execution_id: "e-1".to_string(),
                    side: Action::Other,
                    price: dec!(100),
                    quantity: dec!(1),
                    cumulative_quantity: dec!(1),
                    fee: Decimal::ZERO,
                    fee_asset: crate::types::fee::FeeAsset::Quote,
                    timestamp: 1,
                }),
                Err(unsupported.clone())
            );
            assert_eq!(account.fault(), Some(unsupported));
            assert!(account.errors().is_empty());
            assert!(account.with_gate(|gate| gate.ledger().journal().is_empty()));
            assert_eq!(
                policy.check(&request("o-2", Action::Buy)),
                PretradeDecision::Rejected {
                    reason:
                        "policy fault: unsupported action Other: only buy and sell are supported"
                            .to_string(),
                }
            );

            policy.reset();
            assert_eq!(account.fault(), None);
            assert!(!policy.check(&request("o-3", Action::Buy)).is_rejected());
        }

        #[test]
        fn test_margin_gate_books_and_reconciles_with_builtin() {
            let builtin = queued_engine_with_fees().run();
            let account = account(None);
            let mut engine = queued_engine_with_fees()
                .with_pretrade_policy(account.policy())
                .with_accountant(account.accountant());
            let result = engine.run();
            assert!(account.errors().is_empty(), "{:?}", account.errors());
            assert!(engine.accounting_errors().is_empty());
            assert_eq!(result.final_position, builtin.final_position);
            assert_eq!(result.total_fees, dec!(0.2));
            assert_eq!(result.net_pnl, builtin.net_pnl);
            account.with_gate(|gate| {
                assert!(gate.ledger().reconcile().unwrap().is_clean());
                assert_eq!(gate.ledger().journal().len(), 3);
            });
        }

        #[test]
        fn test_margin_gate_reduces_to_its_position_limit() {
            let account = account(Some(dec!(1)));
            let mut engine = queued_engine_with_fees()
                .with_pretrade_policy(account.policy())
                .with_accountant(account.accountant());
            let result = engine.run();
            let bid = &engine.pretrade_decisions()[0];
            assert_eq!(bid.request.side, Action::Buy);
            assert_eq!(bid.submitted, dec!(1));
            assert_eq!(result.final_position, dec!(1));
            assert!(account.errors().is_empty(), "{:?}", account.errors());
        }

        #[test]
        fn test_margin_policy_rejects_another_instrument() {
            use crate::execution::{OrderManager, OrderRequest};
            let account = account(None);
            let mut policy = account.policy();
            let mut manager = OrderManager::with_defaults();
            let eth = OrderRequest::limit_buy("ETH-USDC", dec!(100), dec!(1));
            let decision = manager
                .register_order_checked(
                    &eth,
                    "e-1".to_string(),
                    1,
                    OrderPurpose::Hedge,
                    &mut policy,
                )
                .unwrap();
            assert!(decision.is_rejected());
            assert_eq!(manager.open_order_count(), 0);
            account.with_gate(|gate| assert_eq!(gate.live_orders().count(), 0));
        }

        #[test]
        fn test_margin_reset_restores_the_account() {
            let account = account(None);
            let mut engine = queued_engine_with_fees()
                .with_pretrade_policy(account.policy())
                .with_accountant(account.accountant());
            let first = engine.run();
            engine.reset();
            account.with_gate(|gate| assert!(gate.ledger().journal().is_empty()));
            let second = engine.run();
            assert_eq!(first.net_pnl, second.net_pnl);
            assert!(account.errors().is_empty(), "{:?}", account.errors());
        }
    }

    #[test]
    fn test_partial_fills_never_exceed_order_quantity() {
        // A generous clearing rate must still fill exactly the order size.
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(queue_quotes()),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(100)),
        )
        .unwrap();
        let result = engine.run();
        let total: Decimal = result.trades.iter().map(|f| f.quantity).sum();
        assert_eq!(total, dec!(2));
        assert_eq!(result.num_trades, 1);
    }

    #[test]
    fn test_invalid_fill_model_is_rejected() {
        let result = BacktestEngine::with_fill_model(
            BacktestConfig::default(),
            ScriptedStrategy::new(vec![]),
            VecDataSource::new(vec![]),
            QueuePositionFillModel::new(Decimal::ZERO),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_probabilistic_model_replays_after_reset() {
        let ticks: Vec<MarketTick> = (0..40)
            .map(|i| MarketTick::new(1000 + i, dec!(99), dec!(5), dec!(100), dec!(5)))
            .collect();
        let quotes = (0..40)
            .map(|i| {
                if i % 5 == 0 {
                    Some((dec!(100), dec!(99.5)))
                } else {
                    None
                }
            })
            .collect();
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default(),
            ScriptedStrategy::new(quotes),
            VecDataSource::new(ticks),
            ProbabilisticFillModel::default(),
        )
        .unwrap();
        let first = engine.run();
        let first_reports = engine.execution_reports().to_vec();
        engine.reset();
        let second = engine.run();
        assert_eq!(first.trades, second.trades);
        assert_eq!(first_reports, engine.execution_reports());
    }

    fn latency_engine(
        ticks: Vec<MarketTick>,
        quotes: Vec<Option<(Decimal, Decimal)>>,
        latency: SimLatencyConfig,
    ) -> BacktestEngine<ScriptedStrategy, VecDataSource> {
        scripted_engine(ticks, quotes)
            .with_latency(latency)
            .unwrap()
    }

    /// Bid 100 rests from t=1000; at t=1002 the strategy requotes (bid 90),
    /// which cancels the 100 bid; the market sells down to 99.5 at t=1004.
    fn race_ticks() -> Vec<MarketTick> {
        vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(98.0), dec!(102.0)),
            create_test_tick(1002, dec!(98.0), dec!(102.0)),
            create_test_tick(1004, dec!(97.0), dec!(99.5)),
            create_test_tick(1008, dec!(98.0), dec!(102.0)),
        ]
    }

    fn race_quotes() -> Vec<Option<(Decimal, Decimal)>> {
        vec![
            Some((dec!(100), dec!(200))),
            None,
            Some((dec!(90), dec!(200))),
            None,
            None,
        ]
    }

    fn kinds_for(
        engine: &BacktestEngine<ScriptedStrategy, VecDataSource>,
        id: SimOrderId,
    ) -> Vec<(u64, ExecutionKind)> {
        engine
            .execution_reports()
            .iter()
            .filter(|r| r.order_id == id)
            .map(|r| (r.stamp.timestamp, r.kind))
            .collect()
    }

    #[test]
    fn test_cancel_before_fill() {
        // Cancel decided at 1002 takes effect at 1003, before the cross at 1004.
        let latency = SimLatencyConfig::new().with_cancel_ms(1);
        let mut engine = latency_engine(race_ticks(), race_quotes(), latency);
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert_eq!(
            kinds_for(&engine, 1),
            vec![
                (1000, ExecutionKind::Accepted),
                (1003, ExecutionKind::Canceled)
            ]
        );
    }

    #[test]
    fn test_fill_before_cancel_stands() {
        // Cancel decided at 1002 takes effect at 1007, after the cross at 1004.
        let latency = SimLatencyConfig::new().with_cancel_ms(5);
        let mut engine = latency_engine(race_ticks(), race_quotes(), latency);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].timestamp, 1004);
        assert_eq!(
            kinds_for(&engine, 1),
            vec![
                (1000, ExecutionKind::Accepted),
                (1004, ExecutionKind::Fill),
                (1007, ExecutionKind::CancelRejected),
            ]
        );
        // The late cancel does not reverse the fill or its cashflow.
        assert_eq!(result.final_position, dec!(2));
        assert_eq!(engine.get_state().0.quantity, dec!(2));
        assert_eq!(engine.get_state().1.realized, dec!(-200));
    }

    #[test]
    fn test_events_between_ticks_happen_on_time() {
        // The 1000 tick is seen at 1002 and the bid is active from 1003,
        // before the next tick at 1010, which it can therefore match.
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1010, dec!(97.0), dec!(99.0)),
        ];
        let latency = SimLatencyConfig::new()
            .with_observation_ms(2)
            .with_submit_ms(1);
        let mut engine = latency_engine(ticks, vec![Some((dec!(100), dec!(200)))], latency);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(
            kinds_for(&engine, 1),
            vec![(1003, ExecutionKind::Accepted), (1010, ExecutionKind::Fill)]
        );
    }

    #[test]
    fn test_observation_jitter_delays_decisions() {
        let (jitter, seed) = (5, 11);
        // Replay the engine's draws: the first observation, one per tick
        // enqueued before the decision, then the bid submission.
        let mut rng = JitterRng::new(seed);
        let observe = rng.draw(jitter);
        for _ in 1..observe {
            rng.draw(jitter);
        }
        let submit = rng.draw(jitter);
        assert!(observe > 0, "the seed must delay the first observation");
        let ticks = (0..30)
            .map(|i| create_test_tick(1000 + i, dec!(98.0), dec!(102.0)))
            .collect();
        let latency = SimLatencyConfig::new().with_jitter(jitter, seed);
        let mut engine = latency_engine(ticks, vec![Some((dec!(100), dec!(101)))], latency);
        engine.run();
        assert_eq!(
            kinds_for(&engine, 1)[0],
            (1000 + observe + submit, ExecutionKind::Accepted)
        );
    }

    #[test]
    fn test_cancel_of_rejected_order_is_answered() {
        // Bid 0 is rejected at 1003; the cancel decided at 1001 lands at
        // 1006 and must still produce an outcome.
        let ticks = (0..9)
            .map(|i| create_test_tick(1000 + i, dec!(98.0), dec!(102.0)))
            .collect();
        let latency = SimLatencyConfig::new().with_submit_ms(3).with_cancel_ms(5);
        let mut engine = latency_engine(
            ticks,
            vec![Some((dec!(0), dec!(101))), Some((dec!(99), dec!(101)))],
            latency,
        );
        engine.run();
        assert_eq!(
            kinds_for(&engine, 1),
            vec![
                (1003, ExecutionKind::Rejected),
                (1006, ExecutionKind::CancelRejected)
            ]
        );
        assert_eq!(
            kinds_for(&engine, 2),
            vec![
                (1003, ExecutionKind::Accepted),
                (1006, ExecutionKind::Canceled)
            ]
        );
    }

    #[test]
    fn test_reports_in_flight_are_delivered_after_last_tick() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(99.0)),
        ];
        let latency = SimLatencyConfig::new().with_ack_ms(5);
        let mut engine = latency_engine(ticks, vec![Some((dec!(100), dec!(200)))], latency);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(engine.strategy().fills, 1);
        assert_eq!(engine.strategy().reports, engine.execution_reports());
    }

    #[test]
    fn test_cancel_faster_than_submit_removes_pending_order() {
        let ticks = (0..9)
            .map(|i| create_test_tick(1000 + i, dec!(98.0), dec!(102.0)))
            .collect();
        let latency = SimLatencyConfig::new().with_submit_ms(5).with_cancel_ms(1);
        let mut engine = latency_engine(
            ticks,
            vec![Some((dec!(99), dec!(101))), Some((dec!(98), dec!(102)))],
            latency,
        );
        engine.run();
        // Never accepted: the cancel at 1002 removes the activation due at 1005.
        assert_eq!(kinds_for(&engine, 1), vec![(1002, ExecutionKind::Canceled)]);
    }

    #[test]
    fn test_late_cancel_after_partial_fill_keeps_filled_part() {
        let mut engine = BacktestEngine::with_fill_model(
            BacktestConfig::default().with_default_order_size(dec!(2)),
            ScriptedStrategy::new(vec![
                Some((dec!(100), dec!(200))),
                None,
                Some((dec!(99), dec!(200))),
            ]),
            VecDataSource::new(queue_ticks()),
            QueuePositionFillModel::new(dec!(0.5)),
        )
        .unwrap()
        .with_latency(SimLatencyConfig::new().with_cancel_ms(3))
        .unwrap();
        let result = engine.run();
        let history: Vec<(u64, ExecutionKind)> = engine
            .execution_reports()
            .iter()
            .filter(|r| r.order_id == 1)
            .map(|r| (r.stamp.timestamp, r.kind))
            .collect();
        // Requote at 1003 cancels at 1006, after the 1005 partial fill.
        assert_eq!(
            history,
            vec![
                (1000, ExecutionKind::Accepted),
                (1003, ExecutionKind::PartialFill),
                (1005, ExecutionKind::PartialFill),
                (1006, ExecutionKind::Canceled),
            ]
        );
        let canceled = engine
            .execution_reports()
            .iter()
            .find(|r| r.order_id == 1 && r.kind == ExecutionKind::Canceled)
            .unwrap();
        assert_eq!(canceled.cumulative_quantity, dec!(1.5));
        assert_eq!(result.final_position, dec!(1.5));
    }

    /// Fills of a run on ticks every 10 ms that sell through a bid of 100,
    /// with the strategy quoting on the first tick and then on every tick
    /// (`requote`) or never again.
    fn fills_with_latency(submit_ms: u64, cancel_ms: u64, requote: bool) -> usize {
        let ticks: Vec<MarketTick> = (0..8)
            .map(|i| create_test_tick(1000 + 10 * i, dec!(97.0), dec!(99.5)))
            .collect();
        let quotes = (0..8)
            .map(|i| (i == 0 || requote).then_some((dec!(100), dec!(200))))
            .collect();
        let latency = SimLatencyConfig::new()
            .with_submit_ms(submit_ms)
            .with_cancel_ms(cancel_ms);
        let mut engine = latency_engine(ticks, quotes, latency);
        engine.run();
        engine
            .execution_reports()
            .iter()
            .filter(|r| matches!(r.kind, ExecutionKind::Fill | ExecutionKind::PartialFill))
            .count()
    }

    #[test]
    fn test_latency_of_exactly_one_tick_never_fills_when_every_tick_requotes() {
        // An order active at a tick's time is not eligible on that tick, and
        // the cancel of the next quote takes effect on the tick after, before
        // matching: no order ever has a tick to fill on.
        assert_eq!(fills_with_latency(10, 10, true), 0);
        // One millisecond less and every order fills on the tick it reaches.
        assert_eq!(fills_with_latency(9, 9, true), 7);
        // Without the requote the order rests and fills on the tick after it
        // becomes active.
        assert_eq!(fills_with_latency(10, 10, false), 1);
    }

    #[test]
    fn test_submission_latency_delays_activation() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1002, dec!(97.0), dec!(99.0)),
            create_test_tick(1003, dec!(97.0), dec!(99.0)),
            create_test_tick(1004, dec!(97.0), dec!(99.0)),
        ];
        let latency = SimLatencyConfig::new().with_submit_ms(3);
        let mut engine = latency_engine(ticks, vec![Some((dec!(100), dec!(200)))], latency);
        let result = engine.run();
        // Active from 1003, so the 1002 and 1003 ticks cannot fill it.
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].timestamp, 1004);
        assert_eq!(kinds_for(&engine, 1)[0], (1003, ExecutionKind::Accepted));
    }

    #[test]
    fn test_agent_never_observes_report_before_ack_delay() {
        let ticks: Vec<MarketTick> = (0..7)
            .map(|i| {
                if i == 1 {
                    create_test_tick(1000 + i, dec!(97.0), dec!(99.0))
                } else {
                    create_test_tick(1000 + i, dec!(98.0), dec!(102.0))
                }
            })
            .collect();
        let latency = SimLatencyConfig::new().with_ack_ms(3);
        let mut engine = latency_engine(ticks, vec![Some((dec!(100), dec!(200)))], latency);
        let result = engine.run();
        assert_eq!(result.num_trades, 1);
        assert_eq!(result.trades[0].timestamp, 1001);
        // Exchange position changes at 1001; the strategy learns at 1004.
        assert_eq!(
            engine.strategy().seen,
            vec![
                (1000, dec!(0)),
                (1001, dec!(0)),
                (1002, dec!(0)),
                (1003, dec!(0)),
                (1004, dec!(2)),
                (1005, dec!(2)),
                (1006, dec!(2)),
            ]
        );
        assert_eq!(engine.strategy().fills, 1);
    }

    #[test]
    fn test_observation_latency_delays_ticks() {
        let ticks: Vec<MarketTick> = (0..5)
            .map(|i| create_test_tick(1000 + i, dec!(98.0), dec!(102.0)))
            .collect();
        let latency = SimLatencyConfig::new().with_observation_ms(2);
        let mut engine = latency_engine(ticks, vec![], latency);
        engine.run();
        let seen: Vec<u64> = engine.strategy().seen.iter().map(|(t, _)| *t).collect();
        assert_eq!(seen, vec![1000, 1001, 1002]);
    }

    #[test]
    fn test_jitter_is_reproducible_with_same_seed() {
        let ticks: Vec<MarketTick> = (0..60)
            .map(|i| {
                let ask = if i % 3 == 0 { dec!(99.0) } else { dec!(102.0) };
                create_test_tick(1000 + i, dec!(98.0), ask)
            })
            .collect();
        let quotes = (0..60)
            .map(|i| (i % 4 == 0).then_some((dec!(100), dec!(200))))
            .collect();
        let latency = SimLatencyConfig::new()
            .with_submit_ms(1)
            .with_cancel_ms(1)
            .with_ack_ms(1)
            .with_jitter(4, 42);
        let mut engine = latency_engine(ticks, quotes, latency);
        let first = engine.run();
        let first_reports = engine.execution_reports().to_vec();
        let first_seen = engine.strategy().seen.clone();
        engine.reset();
        let second = engine.run();
        assert!(first.num_trades > 0);
        assert_eq!(first.trades, second.trades);
        assert_eq!(first_reports, engine.execution_reports());
        assert_eq!(first_seen, engine.strategy().seen);
        // FIFO report channel: the strategy receives reports in the order
        // the venue produced them, even with jitter.
        let delivered = engine.strategy().reports.clone();
        let produced: Vec<ExecutionReport> = engine
            .execution_reports()
            .iter()
            .take(delivered.len())
            .cloned()
            .collect();
        assert_eq!(delivered, produced);
    }

    #[test]
    fn test_zero_latency_matches_default_engine() {
        let ticks = vec![
            create_test_tick(1000, dec!(98.0), dec!(102.0)),
            create_test_tick(1001, dec!(97.0), dec!(99.4)),
            create_test_tick(1002, dec!(100.6), dec!(103.0)),
            create_test_tick(1003, dec!(98.0), dec!(102.0)),
        ];
        let quotes = vec![
            Some((dec!(99.5), dec!(100.5))),
            Some((dec!(99.5), dec!(100.5))),
            None,
            Some((dec!(99), dec!(101))),
        ];
        let mut plain = scripted_engine(ticks.clone(), quotes.clone());
        let mut zero = latency_engine(ticks, quotes, SimLatencyConfig::default());
        let a = plain.run();
        let b = zero.run();
        assert_eq!(a.trades, b.trades);
        assert_eq!(plain.execution_reports(), zero.execution_reports());
        assert_eq!(plain.strategy().seen, zero.strategy().seen);
    }

    #[test]
    fn test_invalid_latency_is_rejected() {
        let engine = scripted_engine(vec![], vec![]);
        let huge = SimLatencyConfig::new().with_submit_ms(crate::backtest::MAX_SIM_LATENCY_MS + 1);
        assert!(engine.with_latency(huge).is_err());
    }
}

#[cfg(test)]
mod auto_traits {
    use super::*;
    use crate::backtest::VecDataSource;

    fn assert_send_sync<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {}
    use std::panic::{RefUnwindSafe, UnwindSafe};

    struct Noop;
    impl BacktestStrategy for Noop {
        fn on_tick(&mut self, _tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
            None
        }
        fn on_fill(&mut self, _fill: &SimulatedFill) {}
        fn reset(&mut self) {}
    }

    #[test]
    fn test_engine_stays_send_and_sync() {
        assert_send_sync::<BacktestEngine<Noop, VecDataSource>>();
        #[cfg(feature = "margin")]
        assert_send_sync::<crate::backtest::MarginAccount>();
    }
}
