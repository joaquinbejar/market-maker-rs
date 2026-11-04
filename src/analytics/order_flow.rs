//! Order flow imbalance analysis.
//!
//! This module provides tools to analyze order flow and detect directional
//! pressure in the market. Order flow analysis helps market makers understand
//! the balance between buying and selling pressure.
//!
//! # Key Metrics
//!
//! - **Buy/Sell Volume**: Aggregate volume by trade direction
//! - **Order Flow Imbalance (OFI)**: Net buying vs selling pressure
//! - **Volume-Weighted Average Price (VWAP)**: By side
//! - **Trade Intensity**: Trades per unit time
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::order_flow::{OrderFlowAnalyzer, Trade, TradeSide};
//! use market_maker_rs::dec;
//!
//! let mut analyzer = OrderFlowAnalyzer::new(5000); // 5 second window
//!
//! // Add some trades
//! analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
//! analyzer.add_trade(Trade::new(dec!(100.1), dec!(5.0), TradeSide::Sell, 2000));
//! analyzer.add_trade(Trade::new(dec!(100.2), dec!(8.0), TradeSide::Buy, 3000));
//!
//! let stats = analyzer.get_stats(4000);
//! assert!(stats.imbalance > dec!(0.0)); // More buying pressure
//! ```

use std::collections::VecDeque;

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trade side classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TradeSide {
    /// Buyer-initiated trade (market buy).
    Buy,
    /// Seller-initiated trade (market sell).
    Sell,
}

impl TradeSide {
    /// Returns true if this is a buy trade.
    #[must_use]
    pub fn is_buy(&self) -> bool {
        matches!(self, TradeSide::Buy)
    }

    /// Returns true if this is a sell trade.
    #[must_use]
    pub fn is_sell(&self) -> bool {
        matches!(self, TradeSide::Sell)
    }
}

/// A single trade record for order flow analysis.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
/// use market_maker_rs::dec;
///
/// let trade = Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1234567890);
/// assert_eq!(trade.notional(), dec!(1000.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trade {
    /// Trade execution price.
    pub price: Decimal,
    /// Trade size in base units.
    pub size: Decimal,
    /// Trade side (buy or sell).
    pub side: TradeSide,
    /// Trade timestamp in milliseconds.
    pub timestamp: u64,
}

impl Trade {
    /// Creates a new trade record.
    ///
    /// # Arguments
    ///
    /// * `price` - Trade execution price
    /// * `size` - Trade size in base units
    /// * `side` - Trade side (buy or sell)
    /// * `timestamp` - Trade timestamp in milliseconds
    #[must_use]
    pub fn new(price: Decimal, size: Decimal, side: TradeSide, timestamp: u64) -> Self {
        Self {
            price,
            size,
            side,
            timestamp,
        }
    }

    /// Returns the notional value of the trade (price * size).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.size)
    }

    /// Returns true if this is a buy trade.
    #[must_use]
    pub fn is_buy(&self) -> bool {
        self.side.is_buy()
    }

    /// Returns true if this is a sell trade.
    #[must_use]
    pub fn is_sell(&self) -> bool {
        self.side.is_sell()
    }
}

/// Order flow statistics for a time window.
///
/// Contains comprehensive metrics about trading activity including
/// volume, trade counts, imbalance, and VWAP by side.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderFlowStats {
    /// Total buy volume in the window.
    pub buy_volume: Decimal,

    /// Total sell volume in the window.
    pub sell_volume: Decimal,

    /// Number of buy trades in the window.
    pub buy_count: u64,

    /// Number of sell trades in the window.
    pub sell_count: u64,

    /// Order flow imbalance: (buy - sell) / (buy + sell).
    /// Range: -1.0 (all sells) to +1.0 (all buys).
    pub imbalance: Decimal,

    /// Net flow: buy_volume - sell_volume.
    pub net_flow: Decimal,

    /// Volume-weighted average price for buy trades.
    /// None if no buy trades in window.
    pub buy_vwap: Option<Decimal>,

    /// Volume-weighted average price for sell trades.
    /// None if no sell trades in window.
    pub sell_vwap: Option<Decimal>,

    /// Total notional traded (buy + sell).
    pub total_notional: Decimal,

    /// Window start timestamp in milliseconds.
    pub window_start: u64,

    /// Window end timestamp in milliseconds.
    pub window_end: u64,
}

/// The volume weighted average price of one side's trades, or `None` when
/// the side has no volume or its notional or volume overflows.
fn side_vwap<'a>(mut trades: impl Iterator<Item = &'a Trade>) -> Option<Decimal> {
    let (notional, volume) =
        trades.try_fold((Decimal::ZERO, Decimal::ZERO), |(notional, volume), t| {
            Some((
                notional.checked_add(t.price.checked_mul(t.size)?)?,
                volume.checked_add(t.size)?,
            ))
        })?;
    if volume > Decimal::ZERO {
        notional.checked_div(volume)
    } else {
        None
    }
}

impl OrderFlowStats {
    /// Returns the total volume (buy + sell).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn total_volume(&self) -> Decimal {
        self.buy_volume.saturating_add(self.sell_volume)
    }

    /// Returns the total trade count, saturating at `u64::MAX`.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.buy_count.saturating_add(self.sell_count)
    }

    /// Returns true if flow is bullish (positive imbalance).
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.imbalance > Decimal::ZERO
    }

    /// Returns true if flow is bearish (negative imbalance).
    #[must_use]
    pub fn is_bearish(&self) -> bool {
        self.imbalance < Decimal::ZERO
    }

    /// Returns the buy/sell volume ratio.
    /// Returns None if sell_volume is zero, a volume sits at the `Decimal`
    /// bound (it may have saturated, so the ratio would be wrong) or the
    /// ratio overflows.
    #[must_use]
    pub fn volume_ratio(&self) -> Option<Decimal> {
        if self.buy_volume == Decimal::MAX || self.sell_volume == Decimal::MAX {
            return None;
        }
        if self.sell_volume > Decimal::ZERO {
            self.buy_volume.checked_div(self.sell_volume)
        } else {
            None
        }
    }

    /// Returns the window duration in milliseconds.
    #[must_use]
    pub fn window_duration_ms(&self) -> u64 {
        self.window_end.saturating_sub(self.window_start)
    }
}

impl Default for OrderFlowStats {
    fn default() -> Self {
        Self {
            buy_volume: Decimal::ZERO,
            sell_volume: Decimal::ZERO,
            buy_count: 0,
            sell_count: 0,
            imbalance: Decimal::ZERO,
            net_flow: Decimal::ZERO,
            buy_vwap: None,
            sell_vwap: None,
            total_notional: Decimal::ZERO,
            window_start: 0,
            window_end: 0,
        }
    }
}

/// Order flow analyzer with rolling window.
///
/// Tracks trades in a rolling time window and calculates order flow
/// statistics including volume, imbalance, and VWAP.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::order_flow::{OrderFlowAnalyzer, Trade, TradeSide};
/// use market_maker_rs::dec;
///
/// let mut analyzer = OrderFlowAnalyzer::new(5000); // 5 second window
///
/// analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
/// analyzer.add_trade(Trade::new(dec!(99.5), dec!(15.0), TradeSide::Sell, 2000));
///
/// let stats = analyzer.get_stats(3000);
/// println!("Imbalance: {}", stats.imbalance);
/// println!("Net flow: {}", stats.net_flow);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderFlowAnalyzer {
    /// Rolling window duration in milliseconds.
    window_ms: u64,
    /// Trade buffer using VecDeque for efficient front removal.
    trades: VecDeque<Trade>,
    /// Maximum number of trades to keep (memory management).
    max_trades: usize,
}

impl OrderFlowAnalyzer {
    /// Default maximum number of trades to keep.
    pub const DEFAULT_MAX_TRADES: usize = 10_000;

    /// Creates a new `OrderFlowAnalyzer` with the specified window duration.
    ///
    /// # Arguments
    ///
    /// * `window_ms` - Rolling window duration in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::analytics::order_flow::OrderFlowAnalyzer;
    ///
    /// let analyzer = OrderFlowAnalyzer::new(5000); // 5 second window
    /// ```
    #[must_use]
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            trades: VecDeque::new(),
            max_trades: Self::DEFAULT_MAX_TRADES,
        }
    }

    /// Creates a new `OrderFlowAnalyzer` with custom max trades limit.
    ///
    /// # Arguments
    ///
    /// * `window_ms` - Rolling window duration in milliseconds
    /// * `max_trades` - Maximum number of trades to keep in buffer
    #[must_use]
    pub fn with_max_trades(window_ms: u64, max_trades: usize) -> Self {
        Self {
            window_ms,
            trades: VecDeque::new(),
            max_trades,
        }
    }

    /// Returns the window duration in milliseconds.
    #[must_use]
    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    /// Returns the number of trades currently in the buffer.
    #[must_use]
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// Adds a new trade to the analyzer.
    ///
    /// Trades are added to the buffer and will be included in statistics
    /// until they fall outside the rolling window.
    ///
    /// # Arguments
    ///
    /// * `trade` - The trade to add
    pub fn add_trade(&mut self, trade: Trade) {
        self.trades.push_back(trade);

        // Enforce max trades limit
        while self.trades.len() > self.max_trades {
            self.trades.pop_front();
        }
    }

    /// Adds a trade from components.
    ///
    /// Convenience method to add a trade without creating a Trade struct.
    ///
    /// # Arguments
    ///
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `side` - Trade side
    /// * `timestamp` - Trade timestamp in milliseconds
    pub fn add_trade_components(
        &mut self,
        price: Decimal,
        size: Decimal,
        side: TradeSide,
        timestamp: u64,
    ) {
        self.add_trade(Trade::new(price, size, side, timestamp));
    }

    /// Gets current order flow statistics for the window.
    ///
    /// Calculates all metrics based on trades within the rolling window
    /// ending at `current_time`.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// `OrderFlowStats` containing all calculated metrics. The imbalance is
    /// exact in proportion even when a side's volume is beyond the `Decimal`
    /// bounds. The volumes, notionals and `net_flow` are report only values
    /// that saturate there, and a VWAP whose inputs overflow is `None`.
    #[must_use]
    pub fn get_stats(&self, current_time: u64) -> OrderFlowStats {
        let window_start = current_time.saturating_sub(self.window_ms);

        let in_window = self
            .trades
            .iter()
            .filter(|t| t.timestamp >= window_start && t.timestamp <= current_time);
        let buys = in_window.clone().filter(|t| t.side == TradeSide::Buy);
        let sells = in_window.filter(|t| t.side == TradeSide::Sell);
        let buy_sizes = buys.clone().map(|t| t.size);
        let sell_sizes = sells.clone().map(|t| t.size);

        let saturated_sum = |sizes: &mut dyn Iterator<Item = Decimal>| {
            sizes.fold(Decimal::ZERO, |acc, x| acc.saturating_add(x))
        };
        let buy_volume = saturated_sum(&mut buy_sizes.clone());
        let sell_volume = saturated_sum(&mut sell_sizes.clone());
        let buy_notional = saturated_sum(&mut buys.clone().map(|t| t.notional()));
        let sell_notional = saturated_sum(&mut sells.clone().map(|t| t.notional()));
        let buy_count = u64::try_from(buys.clone().count()).unwrap_or(u64::MAX);
        let sell_count = u64::try_from(sells.clone().count()).unwrap_or(u64::MAX);

        let net_flow = crate::types::decimal::net_of_sides(buy_sizes.clone(), sell_sizes.clone());
        let imbalance = crate::types::decimal::imbalance_of_sides(buy_sizes, sell_sizes);
        let buy_vwap = side_vwap(buys);
        let sell_vwap = side_vwap(sells);

        OrderFlowStats {
            buy_volume,
            sell_volume,
            buy_count,
            sell_count,
            imbalance,
            net_flow,
            buy_vwap,
            sell_vwap,
            total_notional: buy_notional.saturating_add(sell_notional),
            window_start,
            window_end: current_time,
        }
    }

    /// Gets the imbalance value directly.
    ///
    /// Convenience method to get just the imbalance without full stats.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// Imbalance value in range [-1.0, 1.0].
    #[must_use]
    pub fn get_imbalance(&self, current_time: u64) -> Decimal {
        self.get_stats(current_time).imbalance
    }

    /// Checks if flow is significantly bullish.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum imbalance to consider bullish (e.g., 0.3)
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// True if imbalance exceeds the threshold.
    #[must_use]
    pub fn is_bullish(&self, threshold: Decimal, current_time: u64) -> bool {
        self.get_imbalance(current_time) >= threshold
    }

    /// Checks if flow is significantly bearish.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum negative imbalance to consider bearish (e.g., -0.3)
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// True if imbalance is below the threshold.
    #[must_use]
    pub fn is_bearish(&self, threshold: Decimal, current_time: u64) -> bool {
        self.get_imbalance(current_time) <= threshold
    }

    /// Clears old trades outside the window.
    ///
    /// Call this periodically to manage memory usage.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    pub fn cleanup(&mut self, current_time: u64) {
        let window_start = current_time.saturating_sub(self.window_ms);
        while let Some(trade) = self.trades.front() {
            if trade.timestamp < window_start {
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Gets trade intensity (trades per second).
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// Number of trades per second in the current window.
    #[must_use]
    pub fn trade_intensity(&self, current_time: u64) -> Decimal {
        let stats = self.get_stats(current_time);
        let window_seconds = window_seconds(self.window_ms);
        if window_seconds > Decimal::ZERO {
            // A `u64` count over at least 0.001 seconds cannot overflow.
            Decimal::from(stats.total_count())
                .checked_div(window_seconds)
                .unwrap_or(Decimal::MAX)
        } else {
            Decimal::ZERO
        }
    }

    /// Gets volume intensity (volume per second).
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// Total volume per second in the current window. A report only
    /// value: it saturates at the `Decimal` bounds.
    #[must_use]
    pub fn volume_intensity(&self, current_time: u64) -> Decimal {
        let stats = self.get_stats(current_time);
        let window_seconds = window_seconds(self.window_ms);
        if window_seconds > Decimal::ZERO {
            stats
                .total_volume()
                .checked_div(window_seconds)
                .unwrap_or(Decimal::MAX)
        } else {
            Decimal::ZERO
        }
    }

    /// Clears all trades from the analyzer.
    pub fn clear(&mut self) {
        self.trades.clear();
    }

    /// Gets the most recent trade if any.
    #[must_use]
    pub fn last_trade(&self) -> Option<&Trade> {
        self.trades.back()
    }

    /// Gets the oldest trade in the buffer if any.
    #[must_use]
    pub fn first_trade(&self) -> Option<&Trade> {
        self.trades.front()
    }
}

/// Builder for `OrderFlowAnalyzer` with validation.
#[derive(Debug, Clone)]
pub struct OrderFlowAnalyzerBuilder {
    window_ms: u64,
    max_trades: usize,
}

impl OrderFlowAnalyzerBuilder {
    /// Creates a new builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window_ms: 5000,
            max_trades: OrderFlowAnalyzer::DEFAULT_MAX_TRADES,
        }
    }

    /// Sets the window duration in milliseconds.
    #[must_use]
    pub fn window_ms(mut self, window_ms: u64) -> Self {
        self.window_ms = window_ms;
        self
    }

    /// Sets the maximum number of trades to keep.
    #[must_use]
    pub fn max_trades(mut self, max_trades: usize) -> Self {
        self.max_trades = max_trades;
        self
    }

    /// Builds the `OrderFlowAnalyzer` with validation.
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if window_ms is 0.
    pub fn build(self) -> MMResult<OrderFlowAnalyzer> {
        if self.window_ms == 0 {
            return Err(MMError::InvalidConfiguration(
                "window_ms must be positive".to_string(),
            ));
        }

        if self.max_trades == 0 {
            return Err(MMError::InvalidConfiguration(
                "max_trades must be positive".to_string(),
            ));
        }

        Ok(OrderFlowAnalyzer {
            window_ms: self.window_ms,
            trades: VecDeque::new(),
            max_trades: self.max_trades,
        })
    }
}

impl Default for OrderFlowAnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A window length in milliseconds as seconds.
fn window_seconds(window_ms: u64) -> Decimal {
    // Dividing by the nonzero constant 1000 shrinks the value: it cannot fail.
    Decimal::from(window_ms)
        .checked_div(Decimal::ONE_THOUSAND)
        .unwrap_or(Decimal::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000);

        assert_eq!(trade.price, dec!(100.0));
        assert_eq!(trade.size, dec!(10.0));
        assert_eq!(trade.side, TradeSide::Buy);
        assert_eq!(trade.timestamp, 1000);
        assert_eq!(trade.notional(), dec!(1000.0));
        assert!(trade.is_buy());
        assert!(!trade.is_sell());
    }

    #[test]
    fn test_trade_side() {
        assert!(TradeSide::Buy.is_buy());
        assert!(!TradeSide::Buy.is_sell());
        assert!(!TradeSide::Sell.is_buy());
        assert!(TradeSide::Sell.is_sell());
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = OrderFlowAnalyzer::new(5000);
        let stats = analyzer.get_stats(10_000);

        assert_eq!(stats.buy_volume, Decimal::ZERO);
        assert_eq!(stats.sell_volume, Decimal::ZERO);
        assert_eq!(stats.buy_count, 0);
        assert_eq!(stats.sell_count, 0);
        assert_eq!(stats.imbalance, Decimal::ZERO);
        assert_eq!(stats.net_flow, Decimal::ZERO);
        assert!(stats.buy_vwap.is_none());
        assert!(stats.sell_vwap.is_none());
    }

    #[test]
    fn test_single_buy_trade() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));

        let stats = analyzer.get_stats(3000);

        assert_eq!(stats.buy_volume, dec!(10.0));
        assert_eq!(stats.sell_volume, Decimal::ZERO);
        assert_eq!(stats.buy_count, 1);
        assert_eq!(stats.sell_count, 0);
        assert_eq!(stats.imbalance, Decimal::ONE);
        assert_eq!(stats.net_flow, dec!(10.0));
        assert_eq!(stats.buy_vwap, Some(dec!(100.0)));
        assert!(stats.sell_vwap.is_none());
    }

    #[test]
    fn test_single_sell_trade() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Sell, 1000));

        let stats = analyzer.get_stats(3000);

        assert_eq!(stats.buy_volume, Decimal::ZERO);
        assert_eq!(stats.sell_volume, dec!(10.0));
        assert_eq!(stats.imbalance, -Decimal::ONE);
        assert_eq!(stats.net_flow, dec!(-10.0));
        assert!(stats.buy_vwap.is_none());
        assert_eq!(stats.sell_vwap, Some(dec!(100.0)));
    }

    #[test]
    fn test_balanced_flow() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Sell, 2000));

        let stats = analyzer.get_stats(3000);

        assert_eq!(stats.buy_volume, dec!(10.0));
        assert_eq!(stats.sell_volume, dec!(10.0));
        assert_eq!(stats.imbalance, Decimal::ZERO);
        assert_eq!(stats.net_flow, Decimal::ZERO);
    }

    #[test]
    fn test_imbalanced_flow_bullish() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(15.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(5.0), TradeSide::Sell, 2000));

        let stats = analyzer.get_stats(3000);

        // (15 - 5) / (15 + 5) = 10 / 20 = 0.5
        assert_eq!(stats.imbalance, dec!(0.5));
        assert_eq!(stats.net_flow, dec!(10.0));
        assert!(stats.is_bullish());
        assert!(!stats.is_bearish());
    }

    #[test]
    fn test_imbalanced_flow_bearish() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(5.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(15.0), TradeSide::Sell, 2000));

        let stats = analyzer.get_stats(3000);

        // (5 - 15) / (5 + 15) = -10 / 20 = -0.5
        assert_eq!(stats.imbalance, dec!(-0.5));
        assert_eq!(stats.net_flow, dec!(-10.0));
        assert!(!stats.is_bullish());
        assert!(stats.is_bearish());
    }

    #[test]
    fn test_window_expiration() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        // Trade at t=1000
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        // Trade at t=8000 (outside 5s window from t=10000)
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(5.0), TradeSide::Sell, 8000));

        // At t=10000, window is [5000, 10000], so only second trade is included
        let stats = analyzer.get_stats(10_000);

        assert_eq!(stats.buy_volume, Decimal::ZERO);
        assert_eq!(stats.sell_volume, dec!(5.0));
        assert_eq!(stats.buy_count, 0);
        assert_eq!(stats.sell_count, 1);
    }

    #[test]
    fn test_vwap_calculation() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        // Buy trades: 10 @ 100, 20 @ 102
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(102.0), dec!(20.0), TradeSide::Buy, 2000));

        // Sell trades: 5 @ 99, 15 @ 101
        analyzer.add_trade(Trade::new(dec!(99.0), dec!(5.0), TradeSide::Sell, 3000));
        analyzer.add_trade(Trade::new(dec!(101.0), dec!(15.0), TradeSide::Sell, 4000));

        let stats = analyzer.get_stats(5000);

        // Buy VWAP: (10*100 + 20*102) / 30 = (1000 + 2040) / 30 = 3040 / 30 = 101.333...
        let expected_buy_vwap = dec!(3040) / dec!(30);
        assert_eq!(stats.buy_vwap, Some(expected_buy_vwap));

        // Sell VWAP: (5*99 + 15*101) / 20 = (495 + 1515) / 20 = 2010 / 20 = 100.5
        assert_eq!(stats.sell_vwap, Some(dec!(100.5)));
    }

    #[test]
    fn test_cleanup() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 2000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 8000));

        assert_eq!(analyzer.trade_count(), 3);

        // Cleanup at t=10000, window is [5000, 10000]
        analyzer.cleanup(10_000);

        // Only trade at t=8000 should remain
        assert_eq!(analyzer.trade_count(), 1);
    }

    #[test]
    fn test_is_bullish_bearish() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(20.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(5.0), TradeSide::Sell, 2000));

        // Imbalance = (20-5)/(20+5) = 0.6
        assert!(analyzer.is_bullish(dec!(0.5), 3000));
        assert!(!analyzer.is_bullish(dec!(0.7), 3000));
        assert!(!analyzer.is_bearish(dec!(-0.5), 3000));
    }

    #[test]
    fn test_trade_intensity() {
        let mut analyzer = OrderFlowAnalyzer::new(5000); // 5 second window

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 2000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Sell, 3000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Sell, 4000));

        // 4 trades in 5 seconds = 0.8 trades/second
        let intensity = analyzer.trade_intensity(5000);
        assert_eq!(intensity, dec!(0.8));
    }

    #[test]
    fn test_volume_intensity() {
        let mut analyzer = OrderFlowAnalyzer::new(5000); // 5 second window

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(15.0), TradeSide::Sell, 2000));

        // 25 volume in 5 seconds = 5 volume/second
        let intensity = analyzer.volume_intensity(5000);
        assert_eq!(intensity, dec!(5));
    }

    #[test]
    fn test_add_trade_components() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        analyzer.add_trade_components(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000);

        let stats = analyzer.get_stats(2000);
        assert_eq!(stats.buy_volume, dec!(10.0));
    }

    #[test]
    fn test_max_trades_limit() {
        let mut analyzer = OrderFlowAnalyzer::with_max_trades(5000, 3);

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(1.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(2.0), TradeSide::Buy, 2000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(3.0), TradeSide::Buy, 3000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(4.0), TradeSide::Buy, 4000));

        // Should only keep last 3 trades
        assert_eq!(analyzer.trade_count(), 3);

        // First trade (size=1) should be removed
        let stats = analyzer.get_stats(5000);
        assert_eq!(stats.buy_volume, dec!(9.0)); // 2 + 3 + 4
    }

    #[test]
    fn test_builder_valid() {
        let analyzer = OrderFlowAnalyzerBuilder::new()
            .window_ms(10_000)
            .max_trades(5000)
            .build();

        assert!(analyzer.is_ok());
        let analyzer = analyzer.unwrap();
        assert_eq!(analyzer.window_ms(), 10_000);
    }

    #[test]
    fn test_builder_invalid_window() {
        let result = OrderFlowAnalyzerBuilder::new().window_ms(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_max_trades() {
        let result = OrderFlowAnalyzerBuilder::new().max_trades(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_order_flow_stats_helpers() {
        let stats = OrderFlowStats {
            buy_volume: dec!(100.0),
            sell_volume: dec!(50.0),
            buy_count: 10,
            sell_count: 5,
            imbalance: dec!(0.333),
            net_flow: dec!(50.0),
            buy_vwap: Some(dec!(100.0)),
            sell_vwap: Some(dec!(99.0)),
            total_notional: dec!(15000.0),
            window_start: 1000,
            window_end: 6000,
        };

        assert_eq!(stats.total_volume(), dec!(150.0));
        assert_eq!(stats.total_count(), 15);
        assert!(stats.is_bullish());
        assert!(!stats.is_bearish());
        assert_eq!(stats.volume_ratio(), Some(dec!(2.0)));
        assert_eq!(stats.window_duration_ms(), 5000);
    }

    #[test]
    fn test_last_first_trade() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        assert!(analyzer.last_trade().is_none());
        assert!(analyzer.first_trade().is_none());

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(101.0), dec!(20.0), TradeSide::Sell, 2000));

        assert_eq!(analyzer.first_trade().unwrap().price, dec!(100.0));
        assert_eq!(analyzer.last_trade().unwrap().price, dec!(101.0));
    }

    #[test]
    fn test_clear() {
        let mut analyzer = OrderFlowAnalyzer::new(5000);

        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
        analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Sell, 2000));

        assert_eq!(analyzer.trade_count(), 2);

        analyzer.clear();

        assert_eq!(analyzer.trade_count(), 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let trade = Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000);

        let json = serde_json::to_string(&trade).unwrap();
        let deserialized: Trade = serde_json::from_str(&json).unwrap();

        assert_eq!(trade, deserialized);
    }
}
