//! Realistic fill models for backtesting.
//!
//! This module provides various fill models with increasing realism for
//! more accurate backtesting simulations.
//!
//! # Overview
//!
//! Fill models determine how and when orders get filled during backtesting:
//!
//! - **ImmediateFillModel**: Fills at quote price when market crosses (baseline)
//! - **QueuePositionFillModel**: Simulates queue priority based on time and size
//! - **ProbabilisticFillModel**: Fill probability based on depth and time
//! - **MarketImpactFillModel**: Price impact proportional to order size
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{
//!     ImmediateFillModel, FillModel, SimulatedOrder, MarketTick, FillResult
//! };
//! use market_maker_rs::execution::Action;
//! use market_maker_rs::dec;
//!
//! let model = ImmediateFillModel::new();
//! let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
//! let tick = MarketTick::new(1001, dec!(99.9), dec!(1.0), dec!(100.0), dec!(1.0));
//!
//! // Check if order fills
//! let result = model.simulate_fill(&order, &tick, 1)?;
//! match result {
//!     FillResult::FullFill { fill_price } => println!("Filled at {}", fill_price),
//!     FillResult::PartialFill { filled_quantity, fill_price } => {
//!         println!("Partial fill: {} at {}", filled_quantity, fill_price)
//!     }
//!     FillResult::NoFill => println!("No fill"),
//! }
//! # Ok::<(), market_maker_rs::types::error::MMError>(())
//! ```

use crate::Decimal;
use crate::execution::Action;
use crate::types::action::Dir;
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::data::MarketTick;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Simulated order for fill model evaluation.
///
/// Represents an order that the fill model will evaluate for potential fills.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::SimulatedOrder;
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
/// assert_eq!(order.side, Action::Buy);
/// assert_eq!(order.price, dec!(100.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulatedOrder {
    /// Order side. Only buy and sell are supported: fill models never fill
    /// an order with any other action.
    pub side: Action,
    /// Limit price.
    pub price: Decimal,
    /// Order quantity.
    pub quantity: Decimal,
    /// Timestamp when order was submitted, in milliseconds.
    pub submitted_at: u64,
}

impl SimulatedOrder {
    /// Creates a new simulated order.
    #[must_use]
    pub fn new(side: Action, price: Decimal, quantity: Decimal, submitted_at: u64) -> Self {
        Self {
            side,
            price,
            quantity,
            submitted_at,
        }
    }

    /// Returns the notional value of the order, saturating at the
    /// `Decimal` bounds.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.quantity)
    }

    /// Returns true if this is a buy order.
    #[must_use]
    pub fn is_buy(&self) -> bool {
        self.side == Action::Buy
    }

    /// Returns true if this is a sell order.
    #[must_use]
    pub fn is_sell(&self) -> bool {
        self.side == Action::Sell
    }

    /// The supported direction.
    fn dir(&self) -> MMResult<Dir> {
        Dir::try_from(self.side)
    }
}

/// Result of a fill simulation.
///
/// Represents the outcome of evaluating an order against market conditions.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::FillResult;
/// use market_maker_rs::dec;
///
/// let result = FillResult::FullFill { fill_price: dec!(100.0) };
/// assert!(result.is_filled());
/// assert_eq!(result.fill_price(), Some(dec!(100.0)));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum FillResult {
    /// No fill occurred.
    #[default]
    NoFill,
    /// Partial fill with specified quantity and price.
    PartialFill {
        /// Quantity that was filled.
        filled_quantity: Decimal,
        /// Price at which the fill occurred.
        fill_price: Decimal,
    },
    /// Complete fill at specified price.
    FullFill {
        /// Price at which the fill occurred.
        fill_price: Decimal,
    },
}

impl FillResult {
    /// Returns true if any fill occurred.
    #[must_use]
    pub fn is_filled(&self) -> bool {
        !matches!(self, FillResult::NoFill)
    }

    /// Returns true if the order was completely filled.
    #[must_use]
    pub fn is_full_fill(&self) -> bool {
        matches!(self, FillResult::FullFill { .. })
    }

    /// Returns true if the order was partially filled.
    #[must_use]
    pub fn is_partial_fill(&self) -> bool {
        matches!(self, FillResult::PartialFill { .. })
    }

    /// Returns the fill price if any fill occurred.
    #[must_use]
    pub fn fill_price(&self) -> Option<Decimal> {
        match self {
            FillResult::NoFill => None,
            FillResult::PartialFill { fill_price, .. } => Some(*fill_price),
            FillResult::FullFill { fill_price } => Some(*fill_price),
        }
    }

    /// Returns the filled quantity.
    #[must_use]
    pub fn filled_quantity(&self, order_quantity: Decimal) -> Decimal {
        match self {
            FillResult::NoFill => Decimal::ZERO,
            FillResult::PartialFill {
                filled_quantity, ..
            } => *filled_quantity,
            FillResult::FullFill { .. } => order_quantity,
        }
    }
}

/// Trait for fill model implementations.
///
/// Fill models determine how and when orders get filled during backtesting.
/// Different models provide varying levels of realism.
///
/// # Engine contract
///
/// [`BacktestEngine`](super::BacktestEngine) calls [`FillModel::on_tick`]
/// once per tick, then `simulate_fill` for every resting order. It passes the
/// order's original quantity and the time since the order became active. A
/// `PartialFill { filled_quantity, .. }` is read as the cumulative quantity
/// the order would have filled by now: the engine books only the increase
/// over what is already filled, so the same volume is never filled twice. A
/// `FullFill` fills whatever remains.
pub trait FillModel: Send + Sync {
    /// Simulates whether and how an order fills given market state.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to evaluate
    /// * `tick` - Current market tick data
    /// * `time_in_queue_ms` - Time the order has been in queue, in milliseconds
    ///
    /// # Returns
    ///
    /// A `FillResult` indicating no fill, partial fill, or full fill.
    ///
    /// # Errors
    ///
    /// The built-in models return [`MMError::UnsupportedAction`] for an
    /// order whose side is neither buy nor sell, without touching their
    /// state; implementations may return their own errors.
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult>;

    /// Resets the model state (e.g., queue positions).
    fn reset(&mut self);

    /// Returns the model name for identification.
    fn name(&self) -> &'static str;

    /// Observes a market tick before any order is evaluated against it.
    ///
    /// Models that keep market state (for example queue depth estimates)
    /// update it here. The default implementation does nothing.
    fn on_tick(&mut self, _tick: &MarketTick) {}

    /// Checks the model parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a parameter is out of
    /// range. The default implementation accepts any configuration.
    fn validate(&self) -> MMResult<()> {
        Ok(())
    }

    /// Describes the model and its parameters for result reports.
    ///
    /// Heuristic models must say so; no description may claim exact queue
    /// position reconstruction unless the model replays order level data.
    fn description(&self) -> String {
        self.name().to_string()
    }
}

/// Immediate fill model - the simplest fill model.
///
/// Fills orders immediately at the quote price when the market crosses.
/// This is the baseline model that tends to overestimate performance.
///
/// # Fill Logic
///
/// - Buy orders fill when market ask <= order price
/// - Sell orders fill when market bid >= order price
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{ImmediateFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let model = ImmediateFillModel::new();
/// let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
///
/// // Market ask at 99.9 <= order price 100.0, should fill
/// let tick = MarketTick::new(1001, dec!(99.8), dec!(1.0), dec!(99.9), dec!(1.0));
/// let result = model.simulate_fill(&order, &tick, 1)?;
/// assert!(result.is_full_fill());
/// # Ok::<(), market_maker_rs::types::error::MMError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct ImmediateFillModel;

impl ImmediateFillModel {
    /// Creates a new immediate fill model.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ImmediateFillModel {
    /// The fill of a supported order: the model proper.
    fn fill_for(
        &self,
        dir: Dir,
        order: &SimulatedOrder,
        tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> FillResult {
        match dir {
            Dir::Buy => {
                // Buy fills when market ask <= order price
                if tick.ask_price <= order.price {
                    FillResult::FullFill {
                        fill_price: order.price,
                    }
                } else {
                    FillResult::NoFill
                }
            }
            Dir::Sell => {
                // Sell fills when market bid >= order price
                if tick.bid_price >= order.price {
                    FillResult::FullFill {
                        fill_price: order.price,
                    }
                } else {
                    FillResult::NoFill
                }
            }
        }
    }
}

impl FillModel for ImmediateFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        let dir = order.dir()?;
        Ok(self.fill_for(dir, order, tick, time_in_queue_ms))
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "ImmediateFill"
    }

    fn description(&self) -> String {
        "ImmediateFill: reference best bid/offer cross, full fill at limit price (optimistic)"
            .to_string()
    }
}

/// Queue position fill model.
///
/// Simulates queue priority based on time in queue and estimated queue depth.
/// More realistic than immediate fill as it accounts for orders ahead in the queue.
///
/// # Model Assumptions
///
/// - Orders join the back of the queue at their price level
/// - Queue clears at a configurable fill rate (volume per millisecond)
/// - Order fills when enough volume has cleared ahead of it
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{QueuePositionFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 units per ms
/// let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(10.0), dec!(100.1), dec!(10.0));
///
/// // Update queue from market data
/// model.update_queue(&tick).unwrap();
///
/// // Simulate fill after some time in queue
/// let result = model.simulate_fill(&order, &tick, 100);
/// ```
#[derive(Debug, Clone)]
pub struct QueuePositionFillModel {
    /// Estimated queue depth at each price level.
    queue_depth: HashMap<String, Decimal>,
    /// Fill rate: volume per millisecond that clears the queue.
    fill_rate: Decimal,
    /// Minimum time in queue before any fill is possible, in milliseconds.
    min_queue_time_ms: u64,
}

impl QueuePositionFillModel {
    /// Creates a new queue position fill model.
    ///
    /// # Arguments
    ///
    /// * `fill_rate` - Volume per millisecond that clears the queue
    #[must_use]
    pub fn new(fill_rate: Decimal) -> Self {
        Self {
            queue_depth: HashMap::new(),
            fill_rate,
            min_queue_time_ms: 0,
        }
    }

    /// Creates a new queue position fill model with minimum queue time.
    ///
    /// # Arguments
    ///
    /// * `fill_rate` - Volume per millisecond that clears the queue
    /// * `min_queue_time_ms` - Minimum time before fills are possible
    #[must_use]
    pub fn with_min_queue_time(fill_rate: Decimal, min_queue_time_ms: u64) -> Self {
        Self {
            queue_depth: HashMap::new(),
            fill_rate,
            min_queue_time_ms,
        }
    }

    /// Updates queue depth estimates from market data.
    ///
    /// Uses the bid/ask sizes as estimates of queue depth at those levels.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if a size is negative; the
    /// estimates are then unchanged. Invalid data is never read as an empty
    /// queue, which would fill orders at once.
    pub fn update_queue(&mut self, tick: &MarketTick) -> MMResult<()> {
        check_queue_sizes(tick)?;
        // Normalized string keys: 100, 100.0 and 100.00 are the same level.
        let bid_key = tick.bid_price.normalize().to_string();
        let ask_key = tick.ask_price.normalize().to_string();

        self.queue_depth.insert(bid_key, tick.bid_size);
        self.queue_depth.insert(ask_key, tick.ask_size);
        Ok(())
    }

    /// Returns the estimated queue depth at a price level.
    #[must_use]
    pub fn get_queue_depth(&self, price: Decimal) -> Decimal {
        self.queue_depth
            .get(&price.normalize().to_string())
            .copied()
            .unwrap_or(Decimal::ZERO)
    }

    /// Calculates volume cleared based on time in queue.
    ///
    /// Returns `None` when the product overflows: a saturated value is not a
    /// real volume, so the caller treats it conservatively as no fill.
    fn volume_cleared(&self, time_in_queue_ms: u64) -> Option<Decimal> {
        self.fill_rate.checked_mul(Decimal::from(time_in_queue_ms))
    }
}

impl QueuePositionFillModel {
    /// The fill of a supported order: the model proper.
    fn fill_for(
        &self,
        dir: Dir,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult {
        // Check minimum queue time
        if time_in_queue_ms < self.min_queue_time_ms {
            return FillResult::NoFill;
        }

        // First check if market has crossed our price
        let market_crossed = match dir {
            Dir::Buy => tick.ask_price <= order.price,
            Dir::Sell => tick.bid_price >= order.price,
        };

        if !market_crossed {
            return FillResult::NoFill;
        }

        // When the market trades strictly through our price, the level has
        // been consumed and nothing is ahead; only an at touch level keeps
        // its estimated queue.
        let through = match dir {
            Dir::Buy => tick.ask_price < order.price,
            Dir::Sell => tick.bid_price > order.price,
        };
        let queue_ahead = if through {
            Decimal::ZERO
        } else {
            self.get_queue_depth(order.price)
        };

        // Calculate how much volume has cleared
        let Some(cleared) = self.volume_cleared(time_in_queue_ms) else {
            // Overflowed volume is not real volume: fail conservative.
            return FillResult::NoFill;
        };
        if cleared <= queue_ahead {
            // Still waiting in queue
            return FillResult::NoFill;
        }
        // Cannot overflow: cleared > queue_ahead >= 0 here.
        let remainder = cleared.saturating_sub(queue_ahead);
        if remainder >= order.quantity {
            // Full fill - we've cleared the queue plus our order
            FillResult::FullFill {
                fill_price: order.price,
            }
        } else {
            // Partial fill - some of our order filled
            FillResult::PartialFill {
                filled_quantity: remainder,
                fill_price: order.price,
            }
        }
    }
}

impl FillModel for QueuePositionFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        let dir = order.dir()?;
        // A tick with a negative size has no valid queue estimate.
        check_queue_sizes(tick)?;
        Ok(self.fill_for(dir, order, tick, time_in_queue_ms))
    }

    fn reset(&mut self) {
        self.queue_depth.clear();
    }

    fn name(&self) -> &'static str {
        "QueuePosition"
    }

    /// Keeps only the current top of book levels, so memory stays bounded
    /// on long streams and no stale depth survives a price move.
    ///
    /// A tick with a negative size leaves no estimate; `simulate_fill`
    /// rejects that tick, so it never fills against an empty queue.
    fn on_tick(&mut self, tick: &MarketTick) {
        self.queue_depth.clear();
        if self.update_queue(tick).is_err() {
            self.queue_depth.clear();
        }
    }

    fn validate(&self) -> MMResult<()> {
        if self.fill_rate <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "QueuePosition fill_rate must be positive, got {}",
                self.fill_rate
            )));
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!(
            "QueuePosition: heuristic estimate, queue ahead from top of book size, \
             constant clearing rate {} per ms, min queue time {} ms (not order level queue)",
            self.fill_rate, self.min_queue_time_ms
        )
    }
}

impl Default for QueuePositionFillModel {
    fn default() -> Self {
        Self::new(dec!(0.01))
    }
}

/// Probabilistic fill model.
///
/// Fills orders based on a probability that depends on market depth and time in queue.
/// Uses a seeded RNG for reproducible backtests.
///
/// # Probability Formula
///
/// ```text
/// P(fill) = base_prob * exp(-depth_factor * relative_depth) * (1 - exp(-time_factor * time_ms))
/// ```
///
/// Where:
/// - `base_prob` is the base fill probability
/// - `depth_factor` controls sensitivity to market depth
/// - `relative_depth` is order size relative to available liquidity
/// - `time_factor` controls how probability increases over time
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{ProbabilisticFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let model = ProbabilisticFillModel::new(
///     dec!(0.5),  // 50% base probability
///     dec!(1.0),  // depth factor
///     dec!(0.001), // time factor
///     42,         // seed for reproducibility
/// );
///
/// let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));
///
/// // Probability depends on depth and time
/// let prob = model.calculate_probability(&order, &tick, 100);
/// ```
#[derive(Debug)]
pub struct ProbabilisticFillModel {
    /// Base fill probability per tick (0.0 to 1.0).
    base_probability: Decimal,
    /// Depth factor: higher = less likely to fill in deep book.
    depth_factor: Decimal,
    /// Time factor: higher = more likely to fill over time.
    time_factor: Decimal,
    /// Seed for random number generation.
    seed: u64,
    /// Current state for deterministic random generation (thread-safe).
    state: AtomicU64,
}

impl ProbabilisticFillModel {
    /// Creates a new probabilistic fill model.
    ///
    /// # Arguments
    ///
    /// * `base_probability` - Base fill probability (0.0 to 1.0)
    /// * `depth_factor` - Sensitivity to market depth
    /// * `time_factor` - How probability increases over time
    /// * `seed` - Random seed for reproducibility
    #[must_use]
    pub fn new(
        base_probability: Decimal,
        depth_factor: Decimal,
        time_factor: Decimal,
        seed: u64,
    ) -> Self {
        Self {
            base_probability,
            depth_factor,
            time_factor,
            seed,
            state: AtomicU64::new(seed),
        }
    }

    /// Calculates fill probability for an order.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to evaluate
    /// * `tick` - Current market tick
    /// * `time_in_queue_ms` - Time in queue in milliseconds
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the order's side is neither
    /// buy nor sell, or [`MMError::NumericalError`] if the probability
    /// overflows.
    pub fn calculate_probability(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<Decimal> {
        self.probability_for(order.dir()?, order, tick, time_in_queue_ms)
    }

    /// The fill probability of a supported order.
    fn probability_for(
        &self,
        dir: Dir,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<Decimal> {
        // Get available liquidity at our side
        let available_liquidity = match dir {
            Dir::Buy => tick.ask_size,
            Dir::Sell => tick.bid_size,
        };

        // Calculate relative depth (order size / available liquidity)
        let relative_depth = if available_liquidity > Decimal::ZERO {
            order
                .quantity
                .checked_div(available_liquidity)
                .or_overflow("relative depth")?
        } else {
            Decimal::ONE // Max depth if no liquidity
        };

        // Depth component: exp(-depth_factor * relative_depth)
        // Approximated as 1 / (1 + depth_factor * relative_depth) for simplicity
        let depth_component = self
            .depth_factor
            .checked_mul(relative_depth)
            .and_then(|x| Decimal::ONE.checked_add(x))
            .and_then(|x| Decimal::ONE.checked_div(x))
            .or_overflow("depth component")?;

        // Time component: 1 - exp(-time_factor * time_ms)
        // Approximated as min(1, time_factor * time_ms) for simplicity
        // Clamped to 1 anyway, so a product too large to represent is 1;
        // a negative factor (rejected by `validate`) stays below zero.
        let time_component = self
            .time_factor
            .checked_mul(Decimal::from(time_in_queue_ms))
            .unwrap_or(if self.time_factor.is_sign_negative() {
                Decimal::MIN
            } else {
                Decimal::ONE
            })
            .min(Decimal::ONE);

        // Final probability
        self.base_probability
            .checked_mul(depth_component)
            .and_then(|x| x.checked_mul(time_component))
            .or_overflow("fill probability")
    }

    /// Generates a deterministic pseudo-random number, uniform in [0, 1).
    fn next_random(&self) -> Decimal {
        // Simple LCG for deterministic randomness
        let current = self.state.load(Ordering::Relaxed);
        let next = current.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state.store(next, Ordering::Relaxed);
        // The top 32 bits of the state, scaled by 2^32, give a uniform draw
        // in [0, 1). The high bits of an LCG are its best distributed ones.
        let value = next >> 32;
        let scale = Decimal::from(1u64 << 32);
        Decimal::from(value)
            .checked_div(scale)
            .unwrap_or(Decimal::ZERO)
    }
}

impl ProbabilisticFillModel {
    /// The fill of a supported order: the model proper.
    fn fill_for(
        &self,
        dir: Dir,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        // First check if market has crossed our price
        let market_crossed = match dir {
            Dir::Buy => tick.ask_price <= order.price,
            Dir::Sell => tick.bid_price >= order.price,
        };

        if !market_crossed {
            return Ok(FillResult::NoFill);
        }

        // Calculate fill probability
        let prob = self.probability_for(dir, order, tick, time_in_queue_ms)?;

        // Generate random number and compare
        let random = self.next_random();

        Ok(if random < prob {
            FillResult::FullFill {
                fill_price: order.price,
            }
        } else {
            FillResult::NoFill
        })
    }
}

impl FillModel for ProbabilisticFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        // Before the random draw, so the generator state is untouched.
        let dir = order.dir()?;
        self.fill_for(dir, order, tick, time_in_queue_ms)
    }

    fn reset(&mut self) {
        self.state.store(self.seed, Ordering::Relaxed);
    }

    fn name(&self) -> &'static str {
        "Probabilistic"
    }

    fn validate(&self) -> MMResult<()> {
        if self.base_probability < Decimal::ZERO || self.base_probability > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(format!(
                "Probabilistic base_probability must be in [0, 1], got {}",
                self.base_probability
            )));
        }
        if self.depth_factor < Decimal::ZERO || self.time_factor < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "Probabilistic depth_factor and time_factor must be non negative".to_string(),
            ));
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!(
            "Probabilistic: heuristic estimate, base probability {}, depth factor {}, \
             time factor {}, seed {}",
            self.base_probability, self.depth_factor, self.time_factor, self.seed
        )
    }
}

impl Default for ProbabilisticFillModel {
    fn default() -> Self {
        Self::new(dec!(0.5), Decimal::ONE, dec!(0.001), 42)
    }
}

/// Market impact fill model.
///
/// Applies price impact proportional to order size using the square-root model.
/// Wraps another fill model and adjusts the fill price based on market impact.
///
/// # Impact Formula
///
/// ```text
/// impact = coefficient * sqrt(size / average_daily_volume)
/// ```
///
/// For buy orders, the fill price is increased by the impact.
/// For sell orders, the fill price is decreased by the impact.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{
///     MarketImpactFillModel, ImmediateFillModel, FillModel, SimulatedOrder, MarketTick
/// };
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let base_model = ImmediateFillModel::new();
/// let model = MarketImpactFillModel::new(
///     dec!(0.1),      // impact coefficient
///     dec!(1000000.0), // average daily volume
///     base_model,
/// );
///
/// let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(100.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(1000.0), dec!(100.0), dec!(1000.0));
///
/// // Calculate expected impact
/// let impact = model.calculate_impact(dec!(100.0));
/// ```
#[derive(Debug, Clone)]
pub struct MarketImpactFillModel<M: FillModel> {
    /// Impact coefficient for square-root model.
    impact_coefficient: Decimal,
    /// Average daily volume for normalization.
    average_daily_volume: Decimal,
    /// Underlying fill model.
    base_model: M,
}

impl<M: FillModel> MarketImpactFillModel<M> {
    /// Creates a new market impact fill model.
    ///
    /// # Arguments
    ///
    /// * `impact_coefficient` - Coefficient for impact calculation
    /// * `average_daily_volume` - ADV for normalization
    /// * `base_model` - Underlying fill model to wrap
    #[must_use]
    pub fn new(impact_coefficient: Decimal, average_daily_volume: Decimal, base_model: M) -> Self {
        Self {
            impact_coefficient,
            average_daily_volume,
            base_model,
        }
    }

    /// Calculates price impact for a given order size.
    ///
    /// Uses the square-root model: `impact = coeff * sqrt(size / ADV)`
    ///
    /// An impact beyond the `Decimal` range saturates to `Decimal::MAX`;
    /// fills report it as [`MMError::NumericalError`] instead.
    #[must_use]
    pub fn calculate_impact(&self, size: Decimal) -> Decimal {
        self.checked_impact(size).unwrap_or(Decimal::MAX)
    }

    /// Square root impact with checked arithmetic.
    fn checked_impact(&self, size: Decimal) -> MMResult<Decimal> {
        if self.average_daily_volume <= Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }
        let overflow = || {
            MMError::NumericalError(format!(
                "market impact of size {size} over volume {} overflows",
                self.average_daily_volume
            ))
        };

        let size_ratio = size
            .checked_div(self.average_daily_volume)
            .ok_or_else(overflow)?;
        // A negative size has no impact.
        if size_ratio < Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        // Approximate sqrt using Newton's method
        let sqrt_ratio = decimal_sqrt(size_ratio).ok_or_else(overflow)?;

        self.impact_coefficient
            .checked_mul(sqrt_ratio)
            .ok_or_else(overflow)
    }

    /// Moves `fill_price` against the order by the impact of `size`.
    fn impacted_price(&self, dir: Dir, fill_price: Decimal, size: Decimal) -> MMResult<Decimal> {
        let impact = self.checked_impact(size)?;
        match dir {
            Dir::Buy => fill_price.checked_add(impact),
            Dir::Sell => fill_price.checked_sub(impact),
        }
        .ok_or_else(|| {
            MMError::NumericalError(format!(
                "fill price {fill_price} with impact {impact} overflows"
            ))
        })
    }

    /// Returns a reference to the base model.
    #[must_use]
    pub fn base_model(&self) -> &M {
        &self.base_model
    }

    /// Returns a mutable reference to the base model.
    pub fn base_model_mut(&mut self) -> &mut M {
        &mut self.base_model
    }
}

impl<M: FillModel> FillModel for MarketImpactFillModel<M> {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        // Checked here too: a custom base model may not reject an
        // unsupported side itself.
        let dir = order.dir()?;
        // First get result from base model
        let base_result = self
            .base_model
            .simulate_fill(order, tick, time_in_queue_ms)?;

        // Apply market impact to fill price
        Ok(match base_result {
            FillResult::NoFill => FillResult::NoFill,
            FillResult::PartialFill {
                filled_quantity,
                fill_price,
            } => {
                let adjusted_price = self.impacted_price(dir, fill_price, filled_quantity)?;
                FillResult::PartialFill {
                    filled_quantity,
                    fill_price: adjusted_price,
                }
            }
            FillResult::FullFill { fill_price } => {
                let adjusted_price = self.impacted_price(dir, fill_price, order.quantity)?;
                FillResult::FullFill {
                    fill_price: adjusted_price,
                }
            }
        })
    }

    fn reset(&mut self) {
        self.base_model.reset();
    }

    fn name(&self) -> &'static str {
        "MarketImpact"
    }

    fn on_tick(&mut self, tick: &MarketTick) {
        self.base_model.on_tick(tick);
    }

    fn validate(&self) -> MMResult<()> {
        if self.impact_coefficient < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "MarketImpact impact_coefficient must be non negative, got {}",
                self.impact_coefficient
            )));
        }
        if self.average_daily_volume <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "MarketImpact average_daily_volume must be positive, got {}",
                self.average_daily_volume
            )));
        }
        self.base_model.validate()
    }

    fn description(&self) -> String {
        format!(
            "MarketImpact: square root impact, coefficient {}, average daily volume {}, over [{}]",
            self.impact_coefficient,
            self.average_daily_volume,
            self.base_model.description()
        )
    }
}

/// Approximate square root using Newton's method; `None` for a negative
/// input or a step that overflows.
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

/// Rejects a tick whose bid or ask size is negative.
fn check_queue_sizes(tick: &MarketTick) -> MMResult<()> {
    if tick.bid_size.is_sign_negative() || tick.ask_size.is_sign_negative() {
        return Err(MMError::InvalidMarketState(format!(
            "negative queue size at {}: bid {} ask {}",
            tick.timestamp, tick.bid_size, tick.ask_size
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn create_test_tick(bid: Decimal, ask: Decimal) -> MarketTick {
        MarketTick::new(1000, bid, dec!(10.0), ask, dec!(10.0))
    }

    // SimulatedOrder tests
    #[test]
    fn test_simulated_order_new() {
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);

        assert_eq!(order.side, Action::Buy);
        assert_eq!(order.price, dec!(100.0));
        assert_eq!(order.quantity, dec!(1.0));
        assert_eq!(order.submitted_at, 1000);
    }

    #[test]
    fn test_simulated_order_notional() {
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(2.5), 1000);
        assert_eq!(order.notional(), dec!(250.0));
    }

    #[test]
    fn test_simulated_order_is_buy_sell() {
        let buy = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let sell = SimulatedOrder::new(Action::Sell, dec!(100.0), dec!(1.0), 1000);

        assert!(buy.is_buy());
        assert!(!buy.is_sell());
        assert!(!sell.is_buy());
        assert!(sell.is_sell());
    }

    // FillResult tests
    #[test]
    fn test_fill_result_no_fill() {
        let result = FillResult::NoFill;

        assert!(!result.is_filled());
        assert!(!result.is_full_fill());
        assert!(!result.is_partial_fill());
        assert!(result.fill_price().is_none());
        assert_eq!(result.filled_quantity(dec!(1.0)), Decimal::ZERO);
    }

    #[test]
    fn test_fill_result_full_fill() {
        let result = FillResult::FullFill {
            fill_price: dec!(100.0),
        };

        assert!(result.is_filled());
        assert!(result.is_full_fill());
        assert!(!result.is_partial_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
        assert_eq!(result.filled_quantity(dec!(5.0)), dec!(5.0));
    }

    #[test]
    fn test_fill_result_partial_fill() {
        let result = FillResult::PartialFill {
            filled_quantity: dec!(0.5),
            fill_price: dec!(100.0),
        };

        assert!(result.is_filled());
        assert!(!result.is_full_fill());
        assert!(result.is_partial_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
        assert_eq!(result.filled_quantity(dec!(1.0)), dec!(0.5));
    }

    // ImmediateFillModel tests
    #[test]
    fn test_immediate_fill_buy_fills() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 99.9 <= order price 100.0, should fill
        let tick = create_test_tick(dec!(99.8), dec!(99.9));
        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        assert!(result.is_full_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
    }

    #[test]
    fn test_immediate_fill_buy_no_fill() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 100.1 > order price 100.0, should not fill
        let tick = create_test_tick(dec!(100.0), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        assert!(!result.is_filled());
    }

    #[test]
    fn test_immediate_fill_sell_fills() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Action::Sell, dec!(100.0), dec!(1.0), 1000);

        // Bid at 100.1 >= order price 100.0, should fill
        let tick = create_test_tick(dec!(100.1), dec!(100.2));
        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        assert!(result.is_full_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
    }

    #[test]
    fn test_immediate_fill_sell_no_fill() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Action::Sell, dec!(100.0), dec!(1.0), 1000);

        // Bid at 99.9 < order price 100.0, should not fill
        let tick = create_test_tick(dec!(99.9), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        assert!(!result.is_filled());
    }

    #[test]
    fn test_immediate_fill_name() {
        let model = ImmediateFillModel::new();
        assert_eq!(model.name(), "ImmediateFill");
    }

    // QueuePositionFillModel tests
    #[test]
    fn test_queue_position_new() {
        let model = QueuePositionFillModel::new(dec!(0.1));
        assert_eq!(model.fill_rate, dec!(0.1));
    }

    #[test]
    fn test_queue_position_update_queue() {
        let mut model = QueuePositionFillModel::new(dec!(0.1));
        let tick = MarketTick::new(1000, dec!(100.0), dec!(50.0), dec!(100.1), dec!(30.0));

        model.update_queue(&tick).unwrap();

        assert_eq!(model.get_queue_depth(dec!(100.0)), dec!(50.0));
        assert_eq!(model.get_queue_depth(dec!(100.1)), dec!(30.0));
    }

    #[test]
    fn test_queue_position_negative_size_is_rejected() {
        let mut model = QueuePositionFillModel::new(dec!(0.1));
        let valid = MarketTick::new(1000, dec!(100.0), dec!(50.0), dec!(100.1), dec!(30.0));
        model.update_queue(&valid).unwrap();
        let invalid = MarketTick::new(1001, dec!(100.0), dec!(-50.0), dec!(100.1), dec!(30.0));
        let result = model.update_queue(&invalid);
        assert!(
            matches!(result, Err(MMError::InvalidMarketState(_))),
            "{result:?}"
        );
        // The estimates are unchanged.
        assert_eq!(model.get_queue_depth(dec!(100.0)), dec!(50.0));

        // Through the engine hook the invalid tick never fills: it is
        // rejected, not read as an empty queue.
        model.on_tick(&invalid);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.1), dec!(1.0), 1000);
        let result = model.simulate_fill(&order, &invalid, 1_000_000);
        assert!(
            matches!(result, Err(MMError::InvalidMarketState(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_queue_position_no_fill_market_not_crossed() {
        let model = QueuePositionFillModel::new(dec!(0.1));
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 100.1 > order price, market not crossed
        let tick = create_test_tick(dec!(99.9), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 1000).unwrap();

        assert!(!result.is_filled());
    }

    #[test]
    fn test_queue_position_max_queue_and_max_cleared_returns_no_fill() {
        let mut model = QueuePositionFillModel::new(Decimal::MAX);
        let tick = MarketTick::new(1000, dec!(1), dec!(1), dec!(2), Decimal::MAX);
        model.update_queue(&tick).unwrap();
        let order = SimulatedOrder::new(Action::Buy, dec!(2), dec!(1), 1000);
        // cleared == queue_ahead == MAX: only the queue ahead cleared.
        let result = model.simulate_fill(&order, &tick, 1).unwrap();
        assert!(!result.is_filled());
    }

    #[test]
    fn test_queue_position_overflowing_cleared_volume_returns_no_fill() {
        let mut model = QueuePositionFillModel::new(Decimal::MAX);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10), dec!(100), dec!(5));
        model.update_queue(&tick).unwrap();
        let order = SimulatedOrder::new(Action::Buy, dec!(100), dec!(1), 1000);
        let result = model.simulate_fill(&order, &tick, 2).unwrap();
        assert!(!result.is_filled());
    }

    #[test]
    fn test_queue_position_fill_after_queue_clears() {
        let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 per ms
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);

        // Set up queue with 5 units ahead
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(5.0));
        model.update_queue(&tick).unwrap();

        // After 60ms: cleared 6 units, queue was 5, order is 1 -> full fill
        let result = model.simulate_fill(&order, &tick, 60).unwrap();
        assert!(result.is_full_fill());
    }

    #[test]
    fn test_queue_position_partial_fill() {
        let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 per ms
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(2.0), 1000);

        // Set up queue with 5 units ahead
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(5.0));
        model.update_queue(&tick).unwrap();

        // After 60ms: cleared 6 units, queue was 5, partial fill of 1 unit
        let result = model.simulate_fill(&order, &tick, 60).unwrap();
        assert!(result.is_partial_fill());
        assert_eq!(result.filled_quantity(dec!(2.0)), dec!(1.0));
    }

    #[test]
    fn test_queue_position_min_queue_time() {
        let model = QueuePositionFillModel::with_min_queue_time(dec!(0.1), 100);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        // Before min time
        let result = model.simulate_fill(&order, &tick, 50).unwrap();
        assert!(!result.is_filled());

        // After min time
        let _result = model.simulate_fill(&order, &tick, 100).unwrap();
        // May or may not fill depending on queue, but min time check passed
    }

    #[test]
    fn test_queue_position_reset() {
        let mut model = QueuePositionFillModel::new(dec!(0.1));
        let tick = create_test_tick(dec!(100.0), dec!(100.1));
        model.update_queue(&tick).unwrap();

        assert!(model.get_queue_depth(dec!(100.0)) > Decimal::ZERO);

        model.reset();

        assert_eq!(model.get_queue_depth(dec!(100.0)), Decimal::ZERO);
    }

    // ProbabilisticFillModel tests
    #[test]
    fn test_probabilistic_huge_time_factor_saturates_the_time_component() {
        let model = ProbabilisticFillModel::new(dec!(0.5), Decimal::ZERO, Decimal::MAX, 42);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));
        let prob = model.calculate_probability(&order, &tick, 1_000).unwrap();
        assert_eq!(prob, dec!(0.5));
    }

    #[test]
    fn test_probabilistic_overflowing_probability_returns_numerical_error() {
        let model = ProbabilisticFillModel::new(dec!(0.5), Decimal::MAX, dec!(0.01), 42);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(10.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(1.0));
        let result = model.calculate_probability(&order, &tick, 100);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        let crossed = create_test_tick(dec!(99.9), dec!(99.9));
        let result = model.simulate_fill(&order, &crossed, 100);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_probabilistic_new() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.001), 42);

        assert_eq!(model.base_probability, dec!(0.5));
        assert_eq!(model.depth_factor, dec!(1.0));
        assert_eq!(model.seed, 42);
    }

    #[test]
    fn test_probabilistic_calculate_probability() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));

        let prob = model.calculate_probability(&order, &tick, 100).unwrap();

        // Probability should be between 0 and base_probability
        assert!(prob >= Decimal::ZERO);
        assert!(prob <= dec!(0.5));
    }

    #[test]
    fn test_probabilistic_probability_increases_with_time() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));

        let prob_early = model.calculate_probability(&order, &tick, 10).unwrap();
        let prob_late = model.calculate_probability(&order, &tick, 100).unwrap();

        assert!(prob_late > prob_early);
    }

    #[test]
    fn test_probabilistic_deterministic() {
        let model1 = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let model2 = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);

        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        // Same seed should produce same results
        let result1 = model1.simulate_fill(&order, &tick, 100).unwrap();
        let result2 = model2.simulate_fill(&order, &tick, 100).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_probabilistic_reset() {
        let mut model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9));

        let result1 = model.simulate_fill(&order, &tick, 100).unwrap();
        model.reset();
        let result2 = model.simulate_fill(&order, &tick, 100).unwrap();

        // After reset, should get same result
        assert_eq!(result1, result2);
    }

    /// Fills `n` crossed orders whose fill probability is exactly `p` and
    /// returns the empirical fill rate.
    fn empirical_fill_rate(p: Decimal, seed: u64, n: u32) -> Decimal {
        // depth_factor 0 makes the depth component 1; time_factor 1 with a
        // queue time of 1000 ms saturates the time component at 1.
        let model = ProbabilisticFillModel::new(p, Decimal::ZERO, Decimal::ONE, seed);
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9));
        assert_eq!(model.calculate_probability(&order, &tick, 1000).unwrap(), p);
        let fills = (0..n)
            .filter(|_| {
                model
                    .simulate_fill(&order, &tick, 1000)
                    .unwrap()
                    .is_filled()
            })
            .count();
        Decimal::from(fills) / Decimal::from(n)
    }

    #[test]
    fn test_probabilistic_fill_rate_matches_probability() {
        // 20_000 draws: the binomial standard deviation is at most 0.0036, so
        // a tolerance of 0.02 is more than five standard deviations.
        let tolerance = dec!(0.02);
        for p in [dec!(0.1), dec!(0.3), dec!(0.5), dec!(0.7), dec!(0.9)] {
            for seed in [1u64, 42, 12_345] {
                let rate = empirical_fill_rate(p, seed, 20_000);
                assert!(
                    (rate - p).abs() <= tolerance,
                    "p {p}, seed {seed}: empirical fill rate {rate}"
                );
            }
        }
    }

    #[test]
    fn test_probabilistic_draw_reaches_upper_half() {
        // The draw used to stay below 0.5, so p = 0.5 always filled.
        let rate = empirical_fill_rate(dec!(0.5), 7, 2_000);
        assert!(rate < dec!(0.6), "empirical fill rate {rate}");
    }

    // MarketImpactFillModel tests
    #[test]
    fn test_market_impact_calculate_impact() {
        let model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        // Impact = 0.1 * sqrt(100 / 1000000) = 0.1 * 0.01 = 0.001
        let impact = model.calculate_impact(dec!(100.0));
        assert!(impact > Decimal::ZERO);
        assert!(impact < dec!(0.01));
    }

    #[test]
    fn test_market_impact_buy_increases_price() {
        let model = MarketImpactFillModel::new(dec!(1.0), dec!(1000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(100.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        // Buy should have higher fill price due to impact
        if let FillResult::FullFill { fill_price } = result {
            assert!(fill_price > dec!(100.0));
        } else {
            panic!("Expected full fill");
        }
    }

    #[test]
    fn test_market_impact_sell_decreases_price() {
        let model = MarketImpactFillModel::new(dec!(1.0), dec!(1000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Action::Sell, dec!(100.0), dec!(100.0), 1000);
        let tick = create_test_tick(dec!(100.1), dec!(100.2)); // Market crossed

        let result = model.simulate_fill(&order, &tick, 0).unwrap();

        // Sell should have lower fill price due to impact
        if let FillResult::FullFill { fill_price } = result {
            assert!(fill_price < dec!(100.0));
        } else {
            panic!("Expected full fill");
        }
    }

    #[test]
    fn test_market_impact_ratio_overflow_returns_err() {
        let model = MarketImpactFillModel::new(
            dec!(1.0),
            dec!(0.0000000000000000000000000001),
            ImmediateFillModel::new(),
        );
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(10000000000), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9));

        let result = model.simulate_fill(&order, &tick, 0);
        assert!(matches!(result, Err(MMError::NumericalError(_))));
        assert_eq!(model.calculate_impact(dec!(10000000000)), Decimal::MAX);
    }

    #[test]
    fn test_market_impact_coefficient_overflow_returns_err() {
        let model = MarketImpactFillModel::new(Decimal::MAX, dec!(1.0), ImmediateFillModel::new());
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(4.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9));

        let result = model.simulate_fill(&order, &tick, 0);
        assert!(matches!(result, Err(MMError::NumericalError(_))));
    }

    #[test]
    fn test_market_impact_no_fill_passes_through() {
        let model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(100.1)); // Market not crossed

        let result = model.simulate_fill(&order, &tick, 0).unwrap();
        assert!(!result.is_filled());
    }

    #[test]
    fn test_market_impact_reset() {
        let mut model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        model.reset(); // Should not panic
        assert_eq!(model.name(), "MarketImpact");
    }

    // decimal_sqrt tests
    #[test]
    fn test_decimal_sqrt_positive() {
        let result = decimal_sqrt(dec!(4.0)).unwrap();
        assert!((result - dec!(2.0)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_decimal_sqrt_zero() {
        assert_eq!(decimal_sqrt(Decimal::ZERO), Some(Decimal::ZERO));
    }

    #[test]
    fn test_decimal_sqrt_negative() {
        assert!(decimal_sqrt(dec!(-1.0)).is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_simulated_order_serialization() {
        let order = SimulatedOrder::new(Action::Buy, dec!(100.0), dec!(1.0), 1000);
        let json = serde_json::to_string(&order).unwrap();
        let deserialized: SimulatedOrder = serde_json::from_str(&json).unwrap();
        assert_eq!(order, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_fill_result_serialization() {
        let result = FillResult::FullFill {
            fill_price: dec!(100.0),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: FillResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }

    fn all_models() -> Vec<Box<dyn FillModel>> {
        vec![
            Box::new(ImmediateFillModel::new()),
            Box::new(QueuePositionFillModel::new(dec!(0.5))),
            Box::new(ProbabilisticFillModel::default()),
            Box::new(MarketImpactFillModel::new(
                dec!(0.1),
                dec!(1000),
                ImmediateFillModel::new(),
            )),
        ]
    }

    #[test]
    fn test_contract_partial_fill_never_exceeds_quantity() {
        let crossing = MarketTick::new(10, dec!(99), dec!(3), dec!(100), dec!(3));
        for mut model in all_models() {
            model.on_tick(&crossing);
            for (side, price) in [(Action::Buy, dec!(100)), (Action::Sell, dec!(99))] {
                let order = SimulatedOrder::new(side, price, dec!(2), 0);
                for t in [0u64, 1, 5, 50, 5000] {
                    if let FillResult::PartialFill {
                        filled_quantity, ..
                    } = model.simulate_fill(&order, &crossing, t).unwrap()
                    {
                        assert!(filled_quantity > Decimal::ZERO, "{}", model.name());
                        assert!(filled_quantity <= order.quantity, "{}", model.name());
                    }
                }
            }
        }
    }

    #[test]
    fn test_contract_no_fill_without_cross() {
        // Buy below the ask and sell above the bid never fill.
        let tick = MarketTick::new(10, dec!(99), dec!(3), dec!(101), dec!(3));
        for mut model in all_models() {
            model.on_tick(&tick);
            let buy = SimulatedOrder::new(Action::Buy, dec!(100), dec!(1), 0);
            let sell = SimulatedOrder::new(Action::Sell, dec!(100), dec!(1), 0);
            if model.name() != "Probabilistic" {
                assert_eq!(
                    model.simulate_fill(&buy, &tick, 100).unwrap(),
                    FillResult::NoFill
                );
                assert_eq!(
                    model.simulate_fill(&sell, &tick, 100).unwrap(),
                    FillResult::NoFill
                );
            }
        }
    }

    #[test]
    fn test_contract_descriptions_are_honest() {
        for model in all_models() {
            let description = model.description();
            assert!(description.starts_with(model.name()), "{description}");
            assert!(!description.contains("L3"), "{description}");
        }
        assert!(
            QueuePositionFillModel::default()
                .description()
                .contains("heuristic estimate")
        );
        assert!(
            ProbabilisticFillModel::default()
                .description()
                .contains("heuristic estimate")
        );
    }

    #[test]
    fn test_contract_validation() {
        for model in all_models() {
            assert!(model.validate().is_ok(), "{}", model.name());
        }
        assert!(QueuePositionFillModel::new(dec!(-1)).validate().is_err());
        assert!(
            ProbabilisticFillModel::new(dec!(1.5), dec!(1), dec!(0.1), 1)
                .validate()
                .is_err()
        );
        assert!(
            MarketImpactFillModel::new(dec!(0.1), Decimal::ZERO, ImmediateFillModel::new())
                .validate()
                .is_err()
        );
        assert!(
            MarketImpactFillModel::new(
                dec!(0.1),
                dec!(10),
                QueuePositionFillModel::new(Decimal::ZERO)
            )
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_queue_model_updates_on_tick_hook() {
        let mut model = QueuePositionFillModel::new(dec!(1));
        let tick = MarketTick::new(1, dec!(99), dec!(4), dec!(100), dec!(2));
        model.on_tick(&tick);
        assert_eq!(model.get_queue_depth(dec!(100)), dec!(2));
        assert_eq!(model.get_queue_depth(dec!(99)), dec!(4));
    }

    #[test]
    fn test_queue_depth_ignores_decimal_scale() {
        let mut model = QueuePositionFillModel::new(dec!(1));
        model.on_tick(&MarketTick::new(
            1,
            dec!(99.0),
            dec!(4),
            dec!(100.0),
            dec!(2),
        ));
        assert_eq!(model.get_queue_depth(dec!(100)), dec!(2));
        assert_eq!(model.get_queue_depth(dec!(100.00)), dec!(2));
        assert_eq!(model.get_queue_depth(dec!(99)), dec!(4));
    }

    fn unsupported() -> MMError {
        MMError::UnsupportedAction {
            action: Action::Other,
        }
    }

    #[test]
    fn test_unsupported_action_returns_err() {
        // Crosses both a buy at 100 and a sell at 99.
        let crossing = MarketTick::new(10, dec!(99), dec!(3), dec!(100), dec!(3));
        let order = SimulatedOrder::new(Action::Other, dec!(100), dec!(2), 0);
        for mut model in all_models() {
            model.on_tick(&crossing);
            for t in [0u64, 1, 5, 50, 5000] {
                assert_eq!(
                    model.simulate_fill(&order, &crossing, t),
                    Err(unsupported()),
                    "{} t={t}",
                    model.name()
                );
            }
        }
    }

    #[test]
    fn test_probabilistic_unsupported_action_keeps_generator_state() {
        let model = ProbabilisticFillModel::default();
        let crossing = MarketTick::new(10, dec!(99), dec!(3), dec!(100), dec!(3));
        let order = SimulatedOrder::new(Action::Other, dec!(100), dec!(2), 0);
        let before = model.state.load(Ordering::Relaxed);
        assert_eq!(
            model.simulate_fill(&order, &crossing, 5000),
            Err(unsupported())
        );
        assert_eq!(
            model.calculate_probability(&order, &crossing, 5000),
            Err(unsupported())
        );
        assert_eq!(model.state.load(Ordering::Relaxed), before);
    }

    #[test]
    fn test_market_impact_unsupported_action_returns_err_despite_base_model() {
        /// Fills every order in full at its limit price.
        struct AlwaysFill;
        impl FillModel for AlwaysFill {
            fn simulate_fill(
                &self,
                order: &SimulatedOrder,
                _tick: &MarketTick,
                _time_in_queue_ms: u64,
            ) -> MMResult<FillResult> {
                Ok(FillResult::FullFill {
                    fill_price: order.price,
                })
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "AlwaysFill"
            }
        }

        let model = MarketImpactFillModel::new(dec!(0.1), dec!(1000), AlwaysFill);
        let tick = create_test_tick(dec!(99), dec!(101));
        let other = SimulatedOrder::new(Action::Other, dec!(100), dec!(2), 0);
        assert_eq!(model.simulate_fill(&other, &tick, 0), Err(unsupported()));
        let buy = SimulatedOrder::new(Action::Buy, dec!(100), dec!(2), 0);
        assert!(model.simulate_fill(&buy, &tick, 0).unwrap().is_full_fill());
    }
}
