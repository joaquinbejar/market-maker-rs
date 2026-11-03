//! Adaptive spread calculation based on order book and trade flow imbalance.
//!
//! This module provides tools for dynamically adjusting bid-ask spreads based on
//! market microstructure signals like order book depth imbalance and trade flow.

use crate::Decimal;
use crate::types::decimal::{OrOverflow, imbalance_ratio};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Order book imbalance measurement.
///
/// Measures the ratio of bid vs ask depth to detect directional pressure.
///
/// # Imbalance Interpretation
///
/// - Positive imbalance (+1.0): All depth on bid side, expect price rise
/// - Zero imbalance (0.0): Balanced book
/// - Negative imbalance (-1.0): All depth on ask side, expect price fall
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::adaptive_spread::OrderBookImbalance;
/// use market_maker_rs::dec;
///
/// let imbalance = OrderBookImbalance::new(dec!(1000.0), dec!(500.0), 5);
/// // (1000 - 500) / (1000 + 500) = 0.333...
/// assert!(imbalance.imbalance > dec!(0.33) && imbalance.imbalance < dec!(0.34));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookImbalance {
    /// Imbalance ratio: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    /// Range: -1.0 (all asks) to +1.0 (all bids).
    pub imbalance: Decimal,

    /// Total bid depth considered in units.
    pub bid_depth: Decimal,

    /// Total ask depth considered in units.
    pub ask_depth: Decimal,

    /// Number of price levels analyzed.
    pub levels: u32,
}

impl OrderBookImbalance {
    /// Creates a new `OrderBookImbalance` from bid and ask depths.
    ///
    /// # Arguments
    ///
    /// * `bid_depth` - Total bid depth in units
    /// * `ask_depth` - Total ask depth in units
    /// * `levels` - Number of price levels analyzed
    ///
    /// The imbalance is invariant under a common scale of the depths, even
    /// when their sum is beyond the `Decimal` bounds.
    #[must_use]
    pub fn new(bid_depth: Decimal, ask_depth: Decimal, levels: u32) -> Self {
        let imbalance = if bid_depth.saturating_add(ask_depth) > Decimal::ZERO {
            imbalance_ratio(bid_depth, ask_depth)
        } else {
            Decimal::ZERO
        };

        Self {
            imbalance,
            bid_depth,
            ask_depth,
            levels,
        }
    }

    /// Returns the total depth (bid + ask), saturating at the `Decimal`
    /// bounds.
    #[must_use]
    pub fn total_depth(&self) -> Decimal {
        self.bid_depth.saturating_add(self.ask_depth)
    }

    /// Returns true if the book is bid-heavy (positive imbalance).
    #[must_use]
    pub fn is_bid_heavy(&self) -> bool {
        self.imbalance > Decimal::ZERO
    }

    /// Returns true if the book is ask-heavy (negative imbalance).
    #[must_use]
    pub fn is_ask_heavy(&self) -> bool {
        self.imbalance < Decimal::ZERO
    }

    /// Returns the absolute imbalance value.
    #[must_use]
    pub fn abs_imbalance(&self) -> Decimal {
        self.imbalance.abs()
    }
}

/// Trade flow imbalance measurement.
///
/// Measures the ratio of buy vs sell volume in a time window.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::adaptive_spread::TradeFlowImbalance;
/// use market_maker_rs::dec;
///
/// let flow = TradeFlowImbalance::new(dec!(100.0), dec!(50.0));
/// assert_eq!(flow.net_flow, dec!(50.0));
/// // (100 - 50) / (100 + 50) = 0.333...
/// assert!(flow.imbalance > dec!(0.33) && flow.imbalance < dec!(0.34));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TradeFlowImbalance {
    /// Buy volume in the analysis window.
    pub buy_volume: Decimal,

    /// Sell volume in the analysis window.
    pub sell_volume: Decimal,

    /// Net flow: buy_volume - sell_volume.
    pub net_flow: Decimal,

    /// Imbalance ratio: (buy - sell) / (buy + sell).
    /// Range: -1.0 (all sells) to +1.0 (all buys).
    pub imbalance: Decimal,
}

impl TradeFlowImbalance {
    /// Creates a new `TradeFlowImbalance` from buy and sell volumes.
    ///
    /// # Arguments
    ///
    /// * `buy_volume` - Total buy volume in window
    /// * `sell_volume` - Total sell volume in window
    ///
    /// The imbalance is invariant under a common scale of the volumes, even
    /// when their sum is beyond the `Decimal` bounds; `net_flow` is a
    /// report value that saturates.
    #[must_use]
    pub fn new(buy_volume: Decimal, sell_volume: Decimal) -> Self {
        let net_flow = buy_volume.saturating_sub(sell_volume);
        let imbalance = if buy_volume.saturating_add(sell_volume) > Decimal::ZERO {
            imbalance_ratio(buy_volume, sell_volume)
        } else {
            Decimal::ZERO
        };

        Self {
            buy_volume,
            sell_volume,
            net_flow,
            imbalance,
        }
    }

    /// Returns the total volume (buy + sell), saturating at the `Decimal`
    /// bounds.
    #[must_use]
    pub fn total_volume(&self) -> Decimal {
        self.buy_volume.saturating_add(self.sell_volume)
    }

    /// Returns true if flow is buy-dominated.
    #[must_use]
    pub fn is_buy_dominated(&self) -> bool {
        self.imbalance > Decimal::ZERO
    }

    /// Returns true if flow is sell-dominated.
    #[must_use]
    pub fn is_sell_dominated(&self) -> bool {
        self.imbalance < Decimal::ZERO
    }
}

/// A trade record for flow analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trade {
    /// Trade price.
    pub price: Decimal,
    /// Trade size in units.
    pub size: Decimal,
    /// True if buyer was the aggressor (market buy).
    pub is_buyer_maker: bool,
    /// Trade timestamp in milliseconds.
    pub timestamp: u64,
}

impl Trade {
    /// Creates a new trade record.
    #[must_use]
    pub fn new(price: Decimal, size: Decimal, is_buyer_maker: bool, timestamp: u64) -> Self {
        Self {
            price,
            size,
            is_buyer_maker,
            timestamp,
        }
    }

    /// Returns the notional value of the trade.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.size)
    }
}

/// Configuration for adaptive spread calculation.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::adaptive_spread::AdaptiveSpreadConfig;
/// use market_maker_rs::dec;
///
/// let config = AdaptiveSpreadConfig::new(
///     dec!(0.001),  // 0.1% base spread
///     dec!(2.0),    // max 2x adjustment
///     dec!(0.5),    // 50% orderbook sensitivity
///     dec!(0.3),    // 30% tradeflow sensitivity
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdaptiveSpreadConfig {
    /// Base spread from underlying strategy (as decimal, e.g., 0.001 for 0.1%).
    pub base_spread: Decimal,

    /// Maximum spread adjustment factor (e.g., 2.0 = can double spread).
    pub max_adjustment: Decimal,

    /// Sensitivity to order book imbalance (0.0 to 1.0).
    pub orderbook_sensitivity: Decimal,

    /// Sensitivity to trade flow imbalance (0.0 to 1.0).
    pub tradeflow_sensitivity: Decimal,
}

impl AdaptiveSpreadConfig {
    /// Creates a new `AdaptiveSpreadConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `base_spread` - Base spread as decimal (must be positive)
    /// * `max_adjustment` - Maximum adjustment factor (must be >= 1.0)
    /// * `orderbook_sensitivity` - Sensitivity to orderbook (0.0 to 1.0)
    /// * `tradeflow_sensitivity` - Sensitivity to trade flow (0.0 to 1.0)
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    pub fn new(
        base_spread: Decimal,
        max_adjustment: Decimal,
        orderbook_sensitivity: Decimal,
        tradeflow_sensitivity: Decimal,
    ) -> MMResult<Self> {
        if base_spread <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "base_spread must be positive".to_string(),
            ));
        }

        if max_adjustment < Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "max_adjustment must be >= 1.0".to_string(),
            ));
        }

        if orderbook_sensitivity < Decimal::ZERO || orderbook_sensitivity > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "orderbook_sensitivity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if tradeflow_sensitivity < Decimal::ZERO || tradeflow_sensitivity > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "tradeflow_sensitivity must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            base_spread,
            max_adjustment,
            orderbook_sensitivity,
            tradeflow_sensitivity,
        })
    }
}

/// Calculated adaptive spread with asymmetric bid/ask distances.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::adaptive_spread::AdaptiveSpread;
/// use market_maker_rs::dec;
///
/// let spread = AdaptiveSpread::new(dec!(0.0008), dec!(0.0012));
/// assert_eq!(spread.total_spread, dec!(0.002));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdaptiveSpread {
    /// Distance from mid price for bid (as decimal).
    pub bid_spread: Decimal,

    /// Distance from mid price for ask (as decimal).
    pub ask_spread: Decimal,

    /// Total spread (bid_spread + ask_spread).
    pub total_spread: Decimal,
}

impl AdaptiveSpread {
    /// Creates a new `AdaptiveSpread`.
    ///
    /// `total_spread` is report only: it saturates at the `Decimal`
    /// bounds.
    #[must_use]
    pub fn new(bid_spread: Decimal, ask_spread: Decimal) -> Self {
        Self {
            bid_spread,
            ask_spread,
            total_spread: bid_spread.saturating_add(ask_spread),
        }
    }

    /// Creates a symmetric spread.
    #[must_use]
    pub fn symmetric(half_spread: Decimal) -> Self {
        Self::new(half_spread, half_spread)
    }

    /// Returns the bid price given a mid price.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the price overflows.
    pub fn bid_price(&self, mid_price: Decimal) -> MMResult<Decimal> {
        Decimal::ONE
            .checked_sub(self.bid_spread)
            .and_then(|f| mid_price.checked_mul(f))
            .or_overflow("adaptive bid price")
    }

    /// Returns the ask price given a mid price.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the price overflows.
    pub fn ask_price(&self, mid_price: Decimal) -> MMResult<Decimal> {
        Decimal::ONE
            .checked_add(self.ask_spread)
            .and_then(|f| mid_price.checked_mul(f))
            .or_overflow("adaptive ask price")
    }

    /// Returns the skew (positive = wider ask, negative = wider bid),
    /// saturating at the `Decimal` bounds.
    #[must_use]
    pub fn skew(&self) -> Decimal {
        self.ask_spread.saturating_sub(self.bid_spread)
    }

    /// Returns true if the spread is symmetric.
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        self.bid_spread == self.ask_spread
    }
}

/// Calculator for adaptive spreads based on market microstructure.
///
/// Adjusts spreads based on order book imbalance and trade flow to reduce
/// adverse selection and improve quote positioning.
///
/// # Adjustment Logic
///
/// - **Positive imbalance** (more bids): Expect price rise → widen ask spread
/// - **Negative imbalance** (more asks): Expect price fall → widen bid spread
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::adaptive_spread::{
///     AdaptiveSpreadCalculator, AdaptiveSpreadConfig, OrderBookImbalance,
/// };
/// use market_maker_rs::dec;
///
/// let config = AdaptiveSpreadConfig::new(
///     dec!(0.001), dec!(2.0), dec!(0.5), dec!(0.3)
/// ).unwrap();
/// let calculator = AdaptiveSpreadCalculator::new(config);
///
/// // Balanced book → symmetric spread
/// let balanced = OrderBookImbalance::new(dec!(100.0), dec!(100.0), 5);
/// let spread = calculator.calculate_spread(&balanced, None).unwrap();
/// assert!(spread.is_symmetric());
///
/// // Bid-heavy book → wider ask spread
/// let bid_heavy = OrderBookImbalance::new(dec!(150.0), dec!(50.0), 5);
/// let spread = calculator.calculate_spread(&bid_heavy, None).unwrap();
/// assert!(spread.ask_spread > spread.bid_spread);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdaptiveSpreadCalculator {
    /// Configuration for spread calculation.
    config: AdaptiveSpreadConfig,
}

impl AdaptiveSpreadCalculator {
    /// Creates a new `AdaptiveSpreadCalculator`.
    #[must_use]
    pub fn new(config: AdaptiveSpreadConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveSpreadConfig {
        &self.config
    }

    /// Calculates order book imbalance from depth data.
    ///
    /// # Arguments
    ///
    /// * `bid_depths` - Bid levels as (price, size) pairs, best bid first
    /// * `ask_depths` - Ask levels as (price, size) pairs, best ask first
    /// * `levels` - Number of levels to consider (0 = all)
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::adaptive_spread::AdaptiveSpreadCalculator;
    /// use market_maker_rs::dec;
    ///
    /// let bids = vec![
    ///     (dec!(100.0), dec!(10.0)),
    ///     (dec!(99.0), dec!(20.0)),
    /// ];
    /// let asks = vec![
    ///     (dec!(101.0), dec!(15.0)),
    ///     (dec!(102.0), dec!(25.0)),
    /// ];
    ///
    /// let imbalance = AdaptiveSpreadCalculator::calculate_orderbook_imbalance(&bids, &asks, 2).unwrap();
    /// // bid_depth = 30, ask_depth = 40
    /// // imbalance = (30 - 40) / (30 + 40) = -0.142...
    /// assert!(imbalance.imbalance < dec!(0.0));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if a side's depth overflows: a
    /// saturated depth would distort the signal.
    pub fn calculate_orderbook_imbalance(
        bid_depths: &[(Decimal, Decimal)],
        ask_depths: &[(Decimal, Decimal)],
        levels: u32,
    ) -> MMResult<OrderBookImbalance> {
        let levels_usize = if levels == 0 {
            usize::MAX
        } else {
            usize::try_from(levels).unwrap_or(usize::MAX)
        };

        // A saturated depth would distort the signal, so the sums are checked.
        let depth = |levels: &[(Decimal, Decimal)]| {
            levels
                .iter()
                .take(levels_usize)
                .try_fold(Decimal::ZERO, |acc, (_, size)| acc.checked_add(*size))
                .or_overflow("book depth")
        };
        let bid_depth = depth(bid_depths)?;
        let ask_depth = depth(ask_depths)?;

        let actual_levels = u32::try_from(bid_depths.len().min(ask_depths.len()).min(levels_usize))
            .unwrap_or(u32::MAX);

        Ok(OrderBookImbalance::new(bid_depth, ask_depth, actual_levels))
    }

    /// Calculates volume-weighted order book imbalance.
    ///
    /// Weights depth by distance from mid price, giving more importance
    /// to levels closer to the best bid/ask.
    ///
    /// # Arguments
    ///
    /// * `bid_depths` - Bid levels as (price, size) pairs
    /// * `ask_depths` - Ask levels as (price, size) pairs
    /// * `mid_price` - Current mid price for weighting
    /// * `levels` - Number of levels to consider
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if `mid_price` is zero or a
    /// weighted depth overflows.
    pub fn calculate_weighted_orderbook_imbalance(
        bid_depths: &[(Decimal, Decimal)],
        ask_depths: &[(Decimal, Decimal)],
        mid_price: Decimal,
        levels: u32,
    ) -> MMResult<OrderBookImbalance> {
        let levels_usize = if levels == 0 {
            usize::MAX
        } else {
            usize::try_from(levels).unwrap_or(usize::MAX)
        };

        // Weight by inverse distance from mid: size / (1 + 10 * |p - mid| / mid)
        let weighted = |levels: &[(Decimal, Decimal)]| {
            levels
                .iter()
                .take(levels_usize)
                .try_fold(Decimal::ZERO, |acc, (price, size)| {
                    let distance = mid_price
                        .checked_sub(*price)?
                        .abs()
                        .checked_div(mid_price)?;
                    let weight = distance
                        .checked_mul(Decimal::TEN)
                        .and_then(|x| Decimal::ONE.checked_add(x))
                        .and_then(|x| Decimal::ONE.checked_div(x))?;
                    acc.checked_add(size.checked_mul(weight)?)
                })
                .or_overflow("weighted depth")
        };
        let bid_depth = weighted(bid_depths)?;
        let ask_depth = weighted(ask_depths)?;

        let actual_levels = u32::try_from(bid_depths.len().min(ask_depths.len()).min(levels_usize))
            .unwrap_or(u32::MAX);

        Ok(OrderBookImbalance::new(bid_depth, ask_depth, actual_levels))
    }

    /// Calculates trade flow imbalance from recent trades.
    ///
    /// # Arguments
    ///
    /// * `trades` - List of recent trades
    /// * `window_ms` - Time window in milliseconds
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::adaptive_spread::{AdaptiveSpreadCalculator, Trade};
    /// use market_maker_rs::dec;
    ///
    /// let trades = vec![
    ///     Trade::new(dec!(100.0), dec!(10.0), false, 1000), // buy
    ///     Trade::new(dec!(100.0), dec!(5.0), true, 1500),   // sell
    ///     Trade::new(dec!(100.0), dec!(8.0), false, 2000),  // buy
    /// ];
    ///
    /// let flow = AdaptiveSpreadCalculator::calculate_tradeflow_imbalance(&trades, 5000, 3000).unwrap();
    /// // buy_volume = 18, sell_volume = 5
    /// assert!(flow.is_buy_dominated());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if a side's volume overflows:
    /// a saturated volume would distort the signal.
    pub fn calculate_tradeflow_imbalance(
        trades: &[Trade],
        window_ms: u64,
        current_time: u64,
    ) -> MMResult<TradeFlowImbalance> {
        let cutoff = current_time.saturating_sub(window_ms);

        let (buy_volume, sell_volume) = trades
            .iter()
            .filter(|t| t.timestamp >= cutoff)
            .try_fold((Decimal::ZERO, Decimal::ZERO), |(buy, sell), trade| {
                Some(if trade.is_buyer_maker {
                    // Seller was aggressor (market sell)
                    (buy, sell.checked_add(trade.size)?)
                } else {
                    // Buyer was aggressor (market buy)
                    (buy.checked_add(trade.size)?, sell)
                })
            })
            .or_overflow("trade flow volume")?;

        Ok(TradeFlowImbalance::new(buy_volume, sell_volume))
    }

    /// Calculates adaptive spread based on imbalances.
    ///
    /// # Arguments
    ///
    /// * `orderbook_imbalance` - Current order book imbalance
    /// * `tradeflow_imbalance` - Optional trade flow imbalance
    ///
    /// # Returns
    ///
    /// `AdaptiveSpread` with asymmetric bid/ask distances.
    ///
    /// # Adjustment Logic
    ///
    /// - Positive imbalance (bid-heavy): Widen ask spread, tighten bid
    /// - Negative imbalance (ask-heavy): Widen bid spread, tighten ask
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the spread overflows.
    pub fn calculate_spread(
        &self,
        orderbook_imbalance: &OrderBookImbalance,
        tradeflow_imbalance: Option<&TradeFlowImbalance>,
    ) -> MMResult<AdaptiveSpread> {
        let half_spread = self
            .config
            .base_spread
            .checked_div(Decimal::TWO)
            .or_overflow("adaptive half spread")?;

        // Calculate combined imbalance signal
        let mut combined_imbalance = orderbook_imbalance
            .imbalance
            .checked_mul(self.config.orderbook_sensitivity)
            .or_overflow("combined imbalance")?;

        if let Some(flow) = tradeflow_imbalance {
            combined_imbalance = flow
                .imbalance
                .checked_mul(self.config.tradeflow_sensitivity)
                .and_then(|x| combined_imbalance.checked_add(x))
                .or_overflow("combined imbalance")?;
        }

        // Clamp combined imbalance to [-1, 1]
        let combined_imbalance = combined_imbalance
            .max(Decimal::NEGATIVE_ONE)
            .min(Decimal::ONE);

        // Calculate adjustment factor (0 to max_adjustment - 1)
        let adjustment = self
            .config
            .max_adjustment
            .checked_sub(Decimal::ONE)
            .and_then(|range| combined_imbalance.abs().checked_mul(range))
            .or_overflow("spread adjustment")?;
        let half = Decimal::new(5, 1); // 0.5
        let widen = Decimal::ONE
            .checked_add(adjustment)
            .or_overflow("spread adjustment")?;
        let tighten = adjustment
            .checked_mul(half)
            .and_then(|x| Decimal::ONE.checked_sub(x))
            .or_overflow("spread adjustment")?
            .max(half); // Min 0.5x on tighten side

        // Apply asymmetric adjustment
        // Positive imbalance → widen ask, tighten bid
        // Negative imbalance → widen bid, tighten ask
        let (bid_adjustment, ask_adjustment) = if combined_imbalance > Decimal::ZERO {
            // Bid-heavy: expect price rise, widen ask
            (tighten, widen)
        } else if combined_imbalance < Decimal::ZERO {
            // Ask-heavy: expect price fall, widen bid
            (widen, tighten)
        } else {
            (Decimal::ONE, Decimal::ONE)
        };

        let bid_spread = half_spread
            .checked_mul(bid_adjustment)
            .or_overflow("adaptive bid spread")?;
        let ask_spread = half_spread
            .checked_mul(ask_adjustment)
            .or_overflow("adaptive ask spread")?;

        Ok(AdaptiveSpread::new(bid_spread, ask_spread))
    }

    /// Calculates spread with volatility adjustment.
    ///
    /// Higher volatility increases the base spread before imbalance adjustments.
    ///
    /// # Arguments
    ///
    /// * `orderbook_imbalance` - Current order book imbalance
    /// * `tradeflow_imbalance` - Optional trade flow imbalance
    /// * `current_volatility` - Current volatility estimate
    /// * `baseline_volatility` - Normal/baseline volatility
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the spread overflows.
    pub fn calculate_spread_with_volatility(
        &self,
        orderbook_imbalance: &OrderBookImbalance,
        tradeflow_imbalance: Option<&TradeFlowImbalance>,
        current_volatility: Decimal,
        baseline_volatility: Decimal,
    ) -> MMResult<AdaptiveSpread> {
        // Adjust base spread by volatility ratio; a ratio too large to
        // represent is clamped to the 2x cap below like any large ratio.
        let vol_ratio = if baseline_volatility > Decimal::ZERO {
            current_volatility
                .checked_div(baseline_volatility)
                .unwrap_or(Decimal::TWO)
        } else {
            Decimal::ONE
        };

        // Clamp volatility adjustment
        let vol_adjustment = vol_ratio.max(Decimal::new(5, 1)).min(Decimal::TWO);

        // Create temporary config with adjusted base spread
        let adjusted_config = AdaptiveSpreadConfig {
            base_spread: self
                .config
                .base_spread
                .checked_mul(vol_adjustment)
                .or_overflow("volatility adjusted spread")?,
            ..self.config.clone()
        };

        let adjusted_calculator = AdaptiveSpreadCalculator::new(adjusted_config);
        adjusted_calculator.calculate_spread(orderbook_imbalance, tradeflow_imbalance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_weighted_imbalance_zero_mid_returns_numerical_error() {
        let bids = [(dec!(99), dec!(1))];
        let asks = [(dec!(101), dec!(1))];
        let result = AdaptiveSpreadCalculator::calculate_weighted_orderbook_imbalance(
            &bids,
            &asks,
            Decimal::ZERO,
            1,
        );
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_spread_overflow_returns_numerical_error() {
        let config = AdaptiveSpreadConfig::new(Decimal::MAX, dec!(4), dec!(1), dec!(1)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);
        let book = OrderBookImbalance::new(dec!(10), dec!(0), 1);
        let result = calculator.calculate_spread(&book, None);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_adaptive_prices_overflow_returns_numerical_error() {
        let spread = AdaptiveSpread::new(dec!(-1), dec!(1));
        assert!(matches!(
            spread.bid_price(Decimal::MAX),
            Err(MMError::NumericalError(_))
        ));
        assert!(matches!(
            spread.ask_price(Decimal::MAX),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_orderbook_imbalance_balanced() {
        let imbalance = OrderBookImbalance::new(dec!(100.0), dec!(100.0), 5);

        assert_eq!(imbalance.imbalance, dec!(0.0));
        assert_eq!(imbalance.total_depth(), dec!(200.0));
        assert!(!imbalance.is_bid_heavy());
        assert!(!imbalance.is_ask_heavy());
    }

    #[test]
    fn test_orderbook_imbalance_bid_heavy() {
        let imbalance = OrderBookImbalance::new(dec!(150.0), dec!(50.0), 5);

        // (150 - 50) / (150 + 50) = 0.5
        assert_eq!(imbalance.imbalance, dec!(0.5));
        assert!(imbalance.is_bid_heavy());
        assert!(!imbalance.is_ask_heavy());
    }

    #[test]
    fn test_orderbook_imbalance_ask_heavy() {
        let imbalance = OrderBookImbalance::new(dec!(50.0), dec!(150.0), 5);

        // (50 - 150) / (50 + 150) = -0.5
        assert_eq!(imbalance.imbalance, dec!(-0.5));
        assert!(!imbalance.is_bid_heavy());
        assert!(imbalance.is_ask_heavy());
    }

    #[test]
    fn test_orderbook_imbalance_empty() {
        let imbalance = OrderBookImbalance::new(dec!(0.0), dec!(0.0), 0);
        assert_eq!(imbalance.imbalance, dec!(0.0));
    }

    #[test]
    fn test_tradeflow_imbalance_buy_dominated() {
        let flow = TradeFlowImbalance::new(dec!(100.0), dec!(50.0));

        assert_eq!(flow.net_flow, dec!(50.0));
        assert!(flow.is_buy_dominated());
        assert!(!flow.is_sell_dominated());
    }

    #[test]
    fn test_tradeflow_imbalance_sell_dominated() {
        let flow = TradeFlowImbalance::new(dec!(50.0), dec!(100.0));

        assert_eq!(flow.net_flow, dec!(-50.0));
        assert!(!flow.is_buy_dominated());
        assert!(flow.is_sell_dominated());
    }

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(dec!(100.0), dec!(10.0), false, 12345);

        assert_eq!(trade.price, dec!(100.0));
        assert_eq!(trade.size, dec!(10.0));
        assert!(!trade.is_buyer_maker);
        assert_eq!(trade.timestamp, 12345);
        assert_eq!(trade.notional(), dec!(1000.0));
    }

    #[test]
    fn test_config_valid() {
        let config = AdaptiveSpreadConfig::new(dec!(0.001), dec!(2.0), dec!(0.5), dec!(0.3));
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_invalid_base_spread() {
        let config = AdaptiveSpreadConfig::new(dec!(0.0), dec!(2.0), dec!(0.5), dec!(0.3));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_max_adjustment() {
        let config = AdaptiveSpreadConfig::new(dec!(0.001), dec!(0.5), dec!(0.5), dec!(0.3));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_sensitivity() {
        let config = AdaptiveSpreadConfig::new(dec!(0.001), dec!(2.0), dec!(1.5), dec!(0.3));
        assert!(config.is_err());

        let config = AdaptiveSpreadConfig::new(dec!(0.001), dec!(2.0), dec!(0.5), dec!(-0.1));
        assert!(config.is_err());
    }

    #[test]
    fn test_adaptive_spread_symmetric() {
        let spread = AdaptiveSpread::symmetric(dec!(0.001));

        assert_eq!(spread.bid_spread, dec!(0.001));
        assert_eq!(spread.ask_spread, dec!(0.001));
        assert_eq!(spread.total_spread, dec!(0.002));
        assert!(spread.is_symmetric());
        assert_eq!(spread.skew(), dec!(0.0));
    }

    #[test]
    fn test_adaptive_spread_asymmetric() {
        let spread = AdaptiveSpread::new(dec!(0.0008), dec!(0.0012));

        assert!(!spread.is_symmetric());
        assert_eq!(spread.skew(), dec!(0.0004));
    }

    #[test]
    fn test_adaptive_spread_prices() {
        let spread = AdaptiveSpread::new(dec!(0.001), dec!(0.001));
        let mid = dec!(100.0);

        assert_eq!(spread.bid_price(mid).unwrap(), dec!(99.9));
        assert_eq!(spread.ask_price(mid).unwrap(), dec!(100.1));
    }

    #[test]
    fn test_calculate_orderbook_imbalance() {
        let bids = vec![(dec!(100.0), dec!(10.0)), (dec!(99.0), dec!(20.0))];
        let asks = vec![(dec!(101.0), dec!(15.0)), (dec!(102.0), dec!(25.0))];

        let imbalance =
            AdaptiveSpreadCalculator::calculate_orderbook_imbalance(&bids, &asks, 2).unwrap();

        assert_eq!(imbalance.bid_depth, dec!(30.0));
        assert_eq!(imbalance.ask_depth, dec!(40.0));
        assert_eq!(imbalance.levels, 2);
        assert!(imbalance.is_ask_heavy());
    }

    #[test]
    fn test_calculate_orderbook_imbalance_limited_levels() {
        let bids = vec![
            (dec!(100.0), dec!(10.0)),
            (dec!(99.0), dec!(20.0)),
            (dec!(98.0), dec!(30.0)),
        ];
        let asks = vec![
            (dec!(101.0), dec!(15.0)),
            (dec!(102.0), dec!(25.0)),
            (dec!(103.0), dec!(35.0)),
        ];

        let imbalance =
            AdaptiveSpreadCalculator::calculate_orderbook_imbalance(&bids, &asks, 1).unwrap();

        assert_eq!(imbalance.bid_depth, dec!(10.0));
        assert_eq!(imbalance.ask_depth, dec!(15.0));
    }

    #[test]
    fn test_calculate_tradeflow_imbalance() {
        let trades = vec![
            Trade::new(dec!(100.0), dec!(10.0), false, 1000), // buy
            Trade::new(dec!(100.0), dec!(5.0), true, 1500),   // sell
            Trade::new(dec!(100.0), dec!(8.0), false, 2000),  // buy
        ];

        let flow =
            AdaptiveSpreadCalculator::calculate_tradeflow_imbalance(&trades, 5000, 3000).unwrap();

        assert_eq!(flow.buy_volume, dec!(18.0));
        assert_eq!(flow.sell_volume, dec!(5.0));
        assert!(flow.is_buy_dominated());
    }

    #[test]
    fn test_calculate_tradeflow_imbalance_window() {
        let trades = vec![
            Trade::new(dec!(100.0), dec!(10.0), false, 1000), // outside window
            Trade::new(dec!(100.0), dec!(5.0), true, 2500),   // inside window
            Trade::new(dec!(100.0), dec!(8.0), false, 2800),  // inside window
        ];

        let flow =
            AdaptiveSpreadCalculator::calculate_tradeflow_imbalance(&trades, 1000, 3000).unwrap();

        // Only trades from 2000-3000 should be included
        assert_eq!(flow.buy_volume, dec!(8.0));
        assert_eq!(flow.sell_volume, dec!(5.0));
    }

    #[test]
    fn test_calculate_spread_balanced() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.002), dec!(2.0), dec!(0.5), dec!(0.3)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);

        let imbalance = OrderBookImbalance::new(dec!(100.0), dec!(100.0), 5);
        let spread = calculator.calculate_spread(&imbalance, None).unwrap();

        assert!(spread.is_symmetric());
        assert_eq!(spread.bid_spread, dec!(0.001));
        assert_eq!(spread.ask_spread, dec!(0.001));
    }

    #[test]
    fn test_calculate_spread_bid_heavy() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.002), dec!(2.0), dec!(1.0), dec!(0.0)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);

        // Strong bid imbalance
        let imbalance = OrderBookImbalance::new(dec!(200.0), dec!(0.0), 5);
        let spread = calculator.calculate_spread(&imbalance, None).unwrap();

        // Should widen ask, tighten bid
        assert!(spread.ask_spread > spread.bid_spread);
    }

    #[test]
    fn test_calculate_spread_ask_heavy() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.002), dec!(2.0), dec!(1.0), dec!(0.0)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);

        // Strong ask imbalance
        let imbalance = OrderBookImbalance::new(dec!(0.0), dec!(200.0), 5);
        let spread = calculator.calculate_spread(&imbalance, None).unwrap();

        // Should widen bid, tighten ask
        assert!(spread.bid_spread > spread.ask_spread);
    }

    #[test]
    fn test_calculate_spread_with_tradeflow() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.002), dec!(2.0), dec!(0.5), dec!(0.5)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);

        let orderbook = OrderBookImbalance::new(dec!(100.0), dec!(100.0), 5);
        let tradeflow = TradeFlowImbalance::new(dec!(100.0), dec!(0.0)); // All buys

        let spread = calculator
            .calculate_spread(&orderbook, Some(&tradeflow))
            .unwrap();

        // Buy-dominated flow should widen ask
        assert!(spread.ask_spread > spread.bid_spread);
    }

    #[test]
    fn test_calculate_spread_with_volatility() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.002), dec!(2.0), dec!(0.5), dec!(0.3)).unwrap();
        let calculator = AdaptiveSpreadCalculator::new(config);

        let imbalance = OrderBookImbalance::new(dec!(100.0), dec!(100.0), 5);

        // High volatility should increase spread
        let spread_high_vol = calculator
            .calculate_spread_with_volatility(
                &imbalance,
                None,
                dec!(0.02), // 2% current vol
                dec!(0.01), // 1% baseline
            )
            .unwrap();

        let spread_normal = calculator.calculate_spread(&imbalance, None).unwrap();

        assert!(spread_high_vol.total_spread > spread_normal.total_spread);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config =
            AdaptiveSpreadConfig::new(dec!(0.001), dec!(2.0), dec!(0.5), dec!(0.3)).unwrap();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptiveSpreadConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
