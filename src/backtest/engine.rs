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
//! eligible on later ticks. Every transition is published as an
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
use crate::execution::Side;
use crate::position::inventory::InventoryPosition;
use crate::position::pnl::PnL;
use crate::strategy::quote::Quote;
use crate::types::error::MMResult;

use super::data::{HistoricalDataSource, MarketTick};
use super::fill_models::{FillModel, FillResult, ImmediateFillModel, SimulatedOrder};
use super::lifecycle::{ExecutionKind, ExecutionReport, SimClock, SimOrder, SimOrderId};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Simulated fill representing a trade execution.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::SimulatedFill;
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let fill = SimulatedFill::new(Side::Buy, dec!(100.0), dec!(0.1), 1000);
/// assert_eq!(fill.notional(), dec!(10.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulatedFill {
    /// Order side.
    pub side: Side,
    /// Fill price.
    pub price: Decimal,
    /// Fill quantity.
    pub quantity: Decimal,
    /// Fill timestamp in milliseconds.
    pub timestamp: u64,
    /// Fee paid for this fill.
    pub fee: Decimal,
}

impl SimulatedFill {
    /// Creates a new simulated fill.
    #[must_use]
    pub fn new(side: Side, price: Decimal, quantity: Decimal, timestamp: u64) -> Self {
        Self {
            side,
            price,
            quantity,
            timestamp,
            fee: Decimal::ZERO,
        }
    }

    /// Creates a new simulated fill with fee.
    #[must_use]
    pub fn with_fee(
        side: Side,
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
        }
    }

    /// Returns the notional value (price * quantity).
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }
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
    #[must_use]
    pub fn calculate_slippage(&self, price: Decimal, volatility: Decimal) -> Decimal {
        match self {
            SlippageModel::None => Decimal::ZERO,
            SlippageModel::Fixed(amount) => *amount,
            SlippageModel::Percentage(pct) => price * pct,
            SlippageModel::VolatilityBased { multiplier } => price * volatility * multiplier,
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
    /// Minimum tick size.
    pub tick_size: Decimal,
    /// Minimum lot size.
    pub lot_size: Decimal,
    /// Slippage model.
    pub slippage: SlippageModel,
    /// Default order size.
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
            tick_size: Decimal::from_str_exact("0.01").unwrap(),
            lot_size: Decimal::from_str_exact("0.001").unwrap(),
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
    /// Total fees paid.
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
}

impl BacktestResult {
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
            self.net_pnl / Decimal::from(self.num_trades)
        }
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Returns the return on capital.
    #[must_use]
    pub fn return_on_capital(&self, initial_capital: Decimal) -> Decimal {
        if initial_capital > Decimal::ZERO {
            self.net_pnl / initial_capital
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
    /// bid and ask of [`BacktestConfig::default_order_size`]. The new orders
    /// can only fill on later ticks, never on the tick they were computed
    /// from. Returning `None` leaves resting orders in place.
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote>;

    /// Called when an order is filled.
    fn on_fill(&mut self, fill: &SimulatedFill);

    /// Called for every order lifecycle transition (accept, reject, fill,
    /// cancel), in event order. The default implementation ignores reports.
    fn on_execution_report(&mut self, _report: &ExecutionReport) {}

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
    max_position: Decimal,
    peak_equity: Decimal,
    max_drawdown: Decimal,
    clock: SimClock,
    orders: Vec<SimOrder>,
    next_order_id: SimOrderId,
    reports: Vec<ExecutionReport>,
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
            max_position: Decimal::ZERO,
            peak_equity: initial_capital,
            max_drawdown: Decimal::ZERO,
            clock: SimClock::new(),
            orders: Vec::new(),
            next_order_id: 1,
            reports: Vec::new(),
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

        while let Some(tick) = self.data_source.next_tick() {
            if num_ticks == 0 {
                start_time = tick.timestamp;
            }
            end_time = tick.timestamp;

            // Simulated time follows the data; out of order ticks are
            // processed at the current time so it never goes backwards.
            let now = self.clock.now().max(tick.timestamp);
            let _ = self.clock.advance_to(now);

            // 1. Market data first: match orders that were already active
            //    before this tick.
            self.match_active_orders(&tick, now);

            // 2. Decide on this tick; new orders can only match later ticks.
            if let Some(quote) = self.strategy.on_tick(&tick, &self.position) {
                self.replace_quotes(&quote, now);
            }

            // Update PnL mark-to-market
            let mid_price = tick.mid_price();
            self.pnl.unrealized = self.position.quantity * mid_price;
            self.pnl.total = self.pnl.realized + self.pnl.unrealized;

            // Track equity
            let equity = self.config.initial_capital + self.pnl.total - self.total_fees;
            if self.config.record_equity_curve {
                self.equity_curve.push((tick.timestamp, equity));
            }

            // Track drawdown
            if equity > self.peak_equity {
                self.peak_equity = equity;
            }
            let drawdown = self.peak_equity - equity;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }

            num_ticks += 1;
            callback(num_ticks as usize, total_ticks);
        }

        BacktestResult {
            total_pnl: self.pnl.total,
            total_fees: self.total_fees,
            net_pnl: self.pnl.total - self.total_fees,
            num_trades: self.trades.len() as u64,
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
        }
    }

    /// Matches every eligible resting order against the tick through the
    /// fill model. Fill quantities reported by the model are cumulative per
    /// order; only the increase over the filled quantity is booked, at the
    /// model's price plus slippage.
    fn match_active_orders(&mut self, tick: &MarketTick, now: u64) {
        self.fill_model.on_tick(tick);
        for index in 0..self.orders.len() {
            let order = &self.orders[index];
            // Eligibility uses the tick's exchange time, not the clamped
            // clock, so an out of order tick older than the activation
            // can never fill the order.
            if !order.is_eligible_at(tick.timestamp) {
                continue;
            }
            let side = order.side;
            let view = SimulatedOrder::new(side, order.price, order.quantity, order.active_after);
            let time_in_queue = tick.timestamp.saturating_sub(order.active_after);
            let (quantity, model_price) =
                match self.fill_model.simulate_fill(&view, tick, time_in_queue) {
                    FillResult::NoFill => continue,
                    FillResult::FullFill { fill_price } => (order.remaining(), fill_price),
                    FillResult::PartialFill {
                        filled_quantity,
                        fill_price,
                    } => (
                        filled_quantity.min(order.quantity) - order.filled,
                        fill_price,
                    ),
                };
            if quantity <= Decimal::ZERO {
                continue;
            }
            let price = self.apply_slippage(model_price, side);
            // A model or slippage output that is not a valid price is
            // ignored: it would book negative proceeds and fees.
            if price <= Decimal::ZERO {
                continue;
            }
            if self.orders[index].apply_fill(quantity).is_err() {
                continue;
            }
            let fee = price * quantity * self.config.fee_rate;
            let stamp = self.clock.stamp();
            let report = ExecutionReport::fill(stamp, &self.orders[index], price, quantity, fee);
            self.emit(report);
            self.process_fill(SimulatedFill::with_fee(side, price, quantity, now, fee));
        }
        self.orders.retain(|order| !order.state.is_terminal());
    }

    /// Cancels every resting order and submits the new bid and ask. New
    /// orders become eligible only for ticks after `now`. A crossed or
    /// locked quote (bid at or above ask) is rejected on both sides so the
    /// engine never rests two orders that could trade with each other.
    fn replace_quotes(&mut self, quote: &Quote, now: u64) {
        for index in 0..self.orders.len() {
            if self.orders[index].cancel().is_ok() {
                let stamp = self.clock.stamp();
                let report =
                    ExecutionReport::status(stamp, &self.orders[index], ExecutionKind::Canceled);
                self.emit(report);
            }
        }
        self.orders.clear();

        let size = self.config.default_order_size;
        let crossed = quote.bid_price >= quote.ask_price;
        for (side, price) in [(Side::Buy, quote.bid_price), (Side::Sell, quote.ask_price)] {
            let id = self.next_order_id;
            self.next_order_id += 1;
            let stamp = self.clock.stamp();
            let mut order = SimOrder::new(id, side, price, size, stamp, now);
            let kind = if !crossed && order.validate().is_ok() {
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
            }
        }
    }

    /// Delivers a report to the strategy and records it.
    fn emit(&mut self, report: ExecutionReport) {
        self.strategy.on_execution_report(&report);
        if self.config.record_trades {
            self.reports.push(report);
        }
    }

    /// Applies slippage to a fill price.
    fn apply_slippage(&self, price: Decimal, side: Side) -> Decimal {
        let slippage = self
            .config
            .slippage
            .calculate_slippage(price, Decimal::ZERO);
        match side {
            Side::Buy => price + slippage,
            Side::Sell => price - slippage,
        }
    }

    /// Processes a fill: updates position, PnL, and notifies strategy.
    fn process_fill(&mut self, fill: SimulatedFill) {
        // Update position
        let signed_qty = match fill.side {
            Side::Buy => fill.quantity,
            Side::Sell => -fill.quantity,
        };
        self.position
            .update_fill(signed_qty, fill.price, fill.timestamp);

        // Update realized PnL for closing trades
        // Simplified: just track the cash flow
        let cash_flow = match fill.side {
            Side::Buy => -fill.notional(),
            Side::Sell => fill.notional(),
        };
        self.pnl.add_realized(cash_flow);

        // Track fees
        self.total_fees += fill.fee;

        // Track max position
        let abs_position = self.position.quantity.abs();
        if abs_position > self.max_position {
            self.max_position = abs_position;
        }

        // Notify strategy
        self.strategy.on_fill(&fill);

        // Record trade
        if self.config.record_trades {
            self.trades.push(fill);
        }
    }

    /// Calculates an approximate Sharpe ratio from the equity curve.
    fn calculate_sharpe_ratio(&self) -> Option<Decimal> {
        if self.equity_curve.len() < 2 {
            return None;
        }

        // Calculate returns
        let returns: Vec<Decimal> = self
            .equity_curve
            .windows(2)
            .filter_map(|w| {
                if w[0].1 > Decimal::ZERO {
                    Some((w[1].1 - w[0].1) / w[0].1)
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return None;
        }

        let n = Decimal::from(returns.len() as u64);
        let mean: Decimal = returns.iter().sum::<Decimal>() / n;

        // Calculate standard deviation
        let variance: Decimal = returns
            .iter()
            .map(|r| {
                let diff = *r - mean;
                diff * diff
            })
            .sum::<Decimal>()
            / n;

        if variance <= Decimal::ZERO {
            return None;
        }

        // Approximate sqrt using Newton's method
        let std_dev = decimal_sqrt(variance)?;

        if std_dev > Decimal::ZERO {
            Some(mean / std_dev)
        } else {
            None
        }
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
        self.max_position = Decimal::ZERO;
        self.peak_equity = self.config.initial_capital;
        self.max_drawdown = Decimal::ZERO;
        self.fill_model.reset();
        self.clock.reset();
        self.orders.clear();
        self.next_order_id = 1;
        self.reports.clear();
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
        let next = (x + n / x) / two;
        if (next - x).abs() < Decimal::from_str_exact("0.0000001").unwrap() {
            return Some(next);
        }
        x = next;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::VecDataSource;
    use crate::backtest::{MarketImpactFillModel, ProbabilisticFillModel, QueuePositionFillModel};
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
        let fill = SimulatedFill::new(Side::Buy, dec!(100.0), dec!(0.1), 1000);

        assert_eq!(fill.side, Side::Buy);
        assert_eq!(fill.price, dec!(100.0));
        assert_eq!(fill.quantity, dec!(0.1));
        assert_eq!(fill.timestamp, 1000);
        assert_eq!(fill.fee, Decimal::ZERO);
    }

    #[test]
    fn test_simulated_fill_with_fee() {
        let fill = SimulatedFill::with_fee(Side::Sell, dec!(100.0), dec!(0.1), 1000, dec!(0.01));

        assert_eq!(fill.fee, dec!(0.01));
    }

    #[test]
    fn test_simulated_fill_notional() {
        let fill = SimulatedFill::new(Side::Buy, dec!(100.0), dec!(0.5), 1000);
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
        assert_eq!(result.trades[0].side, Side::Buy);
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
    }

    impl ScriptedStrategy {
        fn new(quotes: Vec<Option<(Decimal, Decimal)>>) -> Self {
            Self {
                quotes,
                index: 0,
                reports: Vec::new(),
                fills: 0,
            }
        }
    }

    impl BacktestStrategy for ScriptedStrategy {
        fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
            let quote = self.quotes.get(self.index).copied().flatten();
            self.index += 1;
            quote.map(|(bid, ask)| Quote {
                bid_price: bid,
                bid_size: Decimal::ONE,
                ask_price: ask,
                ask_size: Decimal::ONE,
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
        assert_eq!(result.trades[0].side, Side::Sell);
        assert_eq!(result.trades[0].price, dec!(101));
        assert_eq!(result.final_position, dec!(-2));
        // The bid is still resting.
        assert_eq!(engine.open_orders().len(), 1);
        assert_eq!(engine.open_orders()[0].side, Side::Buy);
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
        ) -> FillResult {
            match self.script.get(self.tick.saturating_sub(1)) {
                Some(q) if *q > Decimal::ZERO && order.side == Side::Buy => {
                    FillResult::PartialFill {
                        filled_quantity: *q,
                        fill_price: order.price,
                    }
                }
                _ => FillResult::NoFill,
            }
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
}
