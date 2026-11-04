//! Market data types and sources for backtesting.
//!
//! This module provides data structures and traits for historical market data
//! used in backtesting simulations.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{MarketTick, VecDataSource, HistoricalDataSource};
//! use market_maker_rs::dec;
//!
//! let ticks = vec![
//!     MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0)),
//!     MarketTick::new(1001, dec!(100.1), dec!(1.0), dec!(100.2), dec!(1.0)),
//! ];
//!
//! let mut source = VecDataSource::new(ticks);
//! assert_eq!(source.len(), 2);
//!
//! let tick = source.next_tick().unwrap();
//! assert_eq!(tick.timestamp, 1000);
//! ```

use crate::Decimal;
use crate::types::decimal::midpoint;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Market tick data representing a point-in-time snapshot of the order book.
///
/// Contains bid/ask prices and sizes, plus optional last trade information.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::MarketTick;
/// use market_maker_rs::dec;
///
/// let tick = MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0));
/// assert_eq!(tick.mid_price(), dec!(100.05));
/// assert_eq!(tick.spread(), dec!(0.1));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MarketTick {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Best bid price.
    pub bid_price: Decimal,
    /// Best bid size.
    pub bid_size: Decimal,
    /// Best ask price.
    pub ask_price: Decimal,
    /// Best ask size.
    pub ask_size: Decimal,
    /// Last trade price (if available).
    pub last_price: Option<Decimal>,
    /// Last trade size (if available).
    pub last_size: Option<Decimal>,
}

impl MarketTick {
    /// Creates a new market tick.
    #[must_use]
    pub fn new(
        timestamp: u64,
        bid_price: Decimal,
        bid_size: Decimal,
        ask_price: Decimal,
        ask_size: Decimal,
    ) -> Self {
        Self {
            timestamp,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            last_price: None,
            last_size: None,
        }
    }

    /// Creates a new market tick with last trade information.
    #[must_use]
    pub fn with_last_trade(
        timestamp: u64,
        bid_price: Decimal,
        bid_size: Decimal,
        ask_price: Decimal,
        ask_size: Decimal,
        last_price: Decimal,
        last_size: Decimal,
    ) -> Self {
        Self {
            timestamp,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            last_price: Some(last_price),
            last_size: Some(last_size),
        }
    }

    /// Returns the mid price.
    ///
    /// Exact: computed without overflow, never saturated.
    #[must_use]
    pub fn mid_price(&self) -> Decimal {
        midpoint(self.bid_price, self.ask_price)
    }

    /// Returns the bid-ask spread.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn spread(&self) -> Decimal {
        self.ask_price.saturating_sub(self.bid_price)
    }

    /// Returns the spread as a percentage of mid price.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn spread_bps(&self) -> Decimal {
        let mid = self.mid_price();
        if mid > Decimal::ZERO {
            // An overflow is an extreme spread, never a zero one.
            crate::types::decimal::saturating_div(self.spread(), mid)
                .saturating_mul(Decimal::from(10000))
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the total available liquidity (bid + ask size).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn total_liquidity(&self) -> Decimal {
        self.bid_size.saturating_add(self.ask_size)
    }

    /// Returns the imbalance ratio: (bid_size - ask_size) / (bid_size + ask_size).
    ///
    /// Scale invariant: exact even when the sides sum beyond the `Decimal` bounds; zero when there is no liquidity.
    #[must_use]
    pub fn imbalance(&self) -> Decimal {
        let total = self.total_liquidity();
        if total > Decimal::ZERO {
            crate::types::decimal::imbalance_ratio(self.bid_size, self.ask_size)
        } else {
            Decimal::ZERO
        }
    }
}

/// OHLCV bar data for candlestick-based analysis.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::OHLCVBar;
/// use market_maker_rs::dec;
///
/// let bar = OHLCVBar::new(
///     1000,
///     dec!(100.0),
///     dec!(105.0),
///     dec!(99.0),
///     dec!(102.0),
///     dec!(1000.0),
/// );
///
/// assert_eq!(bar.range(), dec!(6.0));
/// assert!(bar.is_bullish());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OHLCVBar {
    /// Bar start timestamp in milliseconds.
    pub timestamp: u64,
    /// Opening price.
    pub open: Decimal,
    /// Highest price.
    pub high: Decimal,
    /// Lowest price.
    pub low: Decimal,
    /// Closing price.
    pub close: Decimal,
    /// Total volume.
    pub volume: Decimal,
}

impl OHLCVBar {
    /// Creates a new OHLCV bar.
    #[must_use]
    pub fn new(
        timestamp: u64,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Returns the price range (high - low).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn range(&self) -> Decimal {
        self.high.saturating_sub(self.low)
    }

    /// Returns the body size (|close - open|).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn body(&self) -> Decimal {
        if self.close > self.open {
            self.close.saturating_sub(self.open)
        } else {
            self.open.saturating_sub(self.close)
        }
    }

    /// Returns true if the bar is bullish (close > open).
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Returns true if the bar is bearish (close < open).
    #[must_use]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Returns the typical price (high + low + close) / 3.
    ///
    /// Overflow safe: the mean stays within the bar's range even when the
    /// sum is beyond the `Decimal` bounds.
    #[must_use]
    pub fn typical_price(&self) -> Decimal {
        crate::types::decimal::mean3(self.high, self.low, self.close)
    }

    /// Returns the VWAP approximation using typical price.
    #[must_use]
    pub fn vwap(&self) -> Decimal {
        self.typical_price()
    }
}

/// Trait for historical data sources.
///
/// Provides an iterator-like interface for consuming market data.
pub trait HistoricalDataSource {
    /// Returns the next tick and advances the cursor.
    fn next_tick(&mut self) -> Option<MarketTick>;

    /// Peeks at the next tick without advancing.
    fn peek_tick(&self) -> Option<&MarketTick>;

    /// Resets the data source to the beginning.
    fn reset(&mut self);

    /// Returns the total number of ticks.
    fn len(&self) -> usize;

    /// Returns true if there are no ticks.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of remaining ticks.
    fn remaining(&self) -> usize;
}

/// Vector-based historical data source.
///
/// Stores ticks in memory for fast iteration.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{MarketTick, VecDataSource, HistoricalDataSource};
/// use market_maker_rs::dec;
///
/// let ticks = vec![
///     MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0)),
///     MarketTick::new(1001, dec!(100.1), dec!(1.0), dec!(100.2), dec!(1.0)),
///     MarketTick::new(1002, dec!(100.2), dec!(1.0), dec!(100.3), dec!(1.0)),
/// ];
///
/// let mut source = VecDataSource::new(ticks);
///
/// // Iterate through ticks
/// while let Some(tick) = source.next_tick() {
///     println!("Tick at {}: mid = {}", tick.timestamp, tick.mid_price());
/// }
///
/// // Reset and iterate again
/// source.reset();
/// assert_eq!(source.remaining(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct VecDataSource {
    ticks: Vec<MarketTick>,
    index: usize,
}

impl VecDataSource {
    /// Creates a new vector data source from a list of ticks.
    #[must_use]
    pub fn new(ticks: Vec<MarketTick>) -> Self {
        Self { ticks, index: 0 }
    }

    /// Creates an empty data source.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ticks: Vec::new(),
            index: 0,
        }
    }

    /// Returns the current index position.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.index
    }

    /// Returns a reference to all ticks.
    #[must_use]
    pub fn ticks(&self) -> &[MarketTick] {
        &self.ticks
    }

    /// Returns the tick at a specific index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&MarketTick> {
        self.ticks.get(index)
    }

    /// Adds a tick to the data source.
    pub fn push(&mut self, tick: MarketTick) {
        self.ticks.push(tick);
    }

    /// Returns the time range of the data.
    #[must_use]
    pub fn time_range(&self) -> Option<(u64, u64)> {
        match (self.ticks.first(), self.ticks.last()) {
            (Some(first), Some(last)) => Some((first.timestamp, last.timestamp)),
            _ => None,
        }
    }
}

impl HistoricalDataSource for VecDataSource {
    fn next_tick(&mut self) -> Option<MarketTick> {
        let tick = self.ticks.get(self.index)?.clone();
        self.index = self.index.saturating_add(1);
        Some(tick)
    }

    fn peek_tick(&self) -> Option<&MarketTick> {
        self.ticks.get(self.index)
    }

    fn reset(&mut self) {
        self.index = 0;
    }

    fn len(&self) -> usize {
        self.ticks.len()
    }

    fn remaining(&self) -> usize {
        self.ticks.len().saturating_sub(self.index)
    }
}

impl Default for VecDataSource {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn create_test_tick(timestamp: u64, bid: Decimal, ask: Decimal) -> MarketTick {
        MarketTick::new(timestamp, bid, dec!(1.0), ask, dec!(1.0))
    }

    #[test]
    fn test_market_tick_new() {
        let tick = MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0));

        assert_eq!(tick.timestamp, 1000);
        assert_eq!(tick.bid_price, dec!(100.0));
        assert_eq!(tick.ask_price, dec!(100.1));
        assert!(tick.last_price.is_none());
    }

    #[test]
    fn test_market_tick_with_last_trade() {
        let tick = MarketTick::with_last_trade(
            1000,
            dec!(100.0),
            dec!(1.0),
            dec!(100.1),
            dec!(1.0),
            dec!(100.05),
            dec!(0.5),
        );

        assert_eq!(tick.last_price, Some(dec!(100.05)));
        assert_eq!(tick.last_size, Some(dec!(0.5)));
    }

    #[test]
    fn test_market_tick_mid_price() {
        let tick = create_test_tick(1000, dec!(100.0), dec!(100.2));
        assert_eq!(tick.mid_price(), dec!(100.1));
    }

    #[test]
    fn test_market_tick_spread() {
        let tick = create_test_tick(1000, dec!(100.0), dec!(100.2));
        assert_eq!(tick.spread(), dec!(0.2));
    }

    #[test]
    fn test_market_tick_spread_bps() {
        let tick = create_test_tick(1000, dec!(100.0), dec!(100.1));
        // Spread = 0.1, mid = 100.05, bps = (0.1 / 100.05) * 10000 ≈ 9.995
        let bps = tick.spread_bps();
        assert!(bps > dec!(9.0) && bps < dec!(10.0));
    }

    #[test]
    fn test_market_tick_imbalance() {
        let tick = MarketTick::new(1000, dec!(100.0), dec!(2.0), dec!(100.1), dec!(1.0));
        // Imbalance = (2 - 1) / (2 + 1) = 1/3
        let imbalance = tick.imbalance();
        assert!(imbalance > dec!(0.33) && imbalance < dec!(0.34));
    }

    #[test]
    fn test_ohlcv_bar_new() {
        let bar = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(102.0),
            dec!(1000.0),
        );

        assert_eq!(bar.timestamp, 1000);
        assert_eq!(bar.open, dec!(100.0));
        assert_eq!(bar.high, dec!(105.0));
        assert_eq!(bar.low, dec!(99.0));
        assert_eq!(bar.close, dec!(102.0));
        assert_eq!(bar.volume, dec!(1000.0));
    }

    #[test]
    fn test_ohlcv_bar_range() {
        let bar = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(102.0),
            dec!(1000.0),
        );
        assert_eq!(bar.range(), dec!(6.0));
    }

    #[test]
    fn test_ohlcv_bar_body() {
        let bullish = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(103.0),
            dec!(1000.0),
        );
        assert_eq!(bullish.body(), dec!(3.0));

        let bearish = OHLCVBar::new(
            1000,
            dec!(103.0),
            dec!(105.0),
            dec!(99.0),
            dec!(100.0),
            dec!(1000.0),
        );
        assert_eq!(bearish.body(), dec!(3.0));
    }

    #[test]
    fn test_ohlcv_bar_bullish_bearish() {
        let bullish = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(103.0),
            dec!(1000.0),
        );
        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());

        let bearish = OHLCVBar::new(
            1000,
            dec!(103.0),
            dec!(105.0),
            dec!(99.0),
            dec!(100.0),
            dec!(1000.0),
        );
        assert!(!bearish.is_bullish());
        assert!(bearish.is_bearish());
    }

    #[test]
    fn test_ohlcv_bar_typical_price_near_the_bounds() {
        let bar = OHLCVBar::new(
            0,
            Decimal::MAX,
            Decimal::MAX,
            Decimal::MAX,
            Decimal::MAX,
            dec!(1),
        );
        assert_eq!(bar.typical_price(), Decimal::MAX);
        assert_eq!(bar.vwap(), Decimal::MAX);
        let low = Decimal::MAX - dec!(3);
        let bar = OHLCVBar::new(0, low, Decimal::MAX, low, low, dec!(1));
        assert_eq!(bar.typical_price(), Decimal::MAX - dec!(2));
    }

    #[test]
    fn test_ohlcv_bar_typical_price() {
        let bar = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(102.0),
            dec!(1000.0),
        );
        // Typical = (105 + 99 + 102) / 3 = 102
        assert_eq!(bar.typical_price(), dec!(102.0));
    }

    #[test]
    fn test_vec_data_source_new() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(1001, dec!(100.1), dec!(100.2)),
        ];

        let source = VecDataSource::new(ticks);
        assert_eq!(source.len(), 2);
        assert!(!source.is_empty());
    }

    #[test]
    fn test_vec_data_source_empty() {
        let source = VecDataSource::empty();
        assert_eq!(source.len(), 0);
        assert!(source.is_empty());
    }

    #[test]
    fn test_vec_data_source_next_tick() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(1001, dec!(100.1), dec!(100.2)),
        ];

        let mut source = VecDataSource::new(ticks);

        let tick1 = source.next_tick().unwrap();
        assert_eq!(tick1.timestamp, 1000);

        let tick2 = source.next_tick().unwrap();
        assert_eq!(tick2.timestamp, 1001);

        assert!(source.next_tick().is_none());
    }

    #[test]
    fn test_vec_data_source_peek_tick() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(1001, dec!(100.1), dec!(100.2)),
        ];

        let mut source = VecDataSource::new(ticks);

        // Peek doesn't advance
        let peeked = source.peek_tick().unwrap();
        assert_eq!(peeked.timestamp, 1000);

        let peeked2 = source.peek_tick().unwrap();
        assert_eq!(peeked2.timestamp, 1000);

        // Next advances
        source.next_tick();
        let peeked3 = source.peek_tick().unwrap();
        assert_eq!(peeked3.timestamp, 1001);
    }

    #[test]
    fn test_vec_data_source_reset() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(1001, dec!(100.1), dec!(100.2)),
        ];

        let mut source = VecDataSource::new(ticks);

        source.next_tick();
        source.next_tick();
        assert_eq!(source.remaining(), 0);

        source.reset();
        assert_eq!(source.remaining(), 2);
        assert_eq!(source.current_index(), 0);
    }

    #[test]
    fn test_vec_data_source_remaining() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(1001, dec!(100.1), dec!(100.2)),
            create_test_tick(1002, dec!(100.2), dec!(100.3)),
        ];

        let mut source = VecDataSource::new(ticks);

        assert_eq!(source.remaining(), 3);
        source.next_tick();
        assert_eq!(source.remaining(), 2);
        source.next_tick();
        assert_eq!(source.remaining(), 1);
        source.next_tick();
        assert_eq!(source.remaining(), 0);
    }

    #[test]
    fn test_vec_data_source_time_range() {
        let ticks = vec![
            create_test_tick(1000, dec!(100.0), dec!(100.1)),
            create_test_tick(2000, dec!(100.1), dec!(100.2)),
            create_test_tick(3000, dec!(100.2), dec!(100.3)),
        ];

        let source = VecDataSource::new(ticks);
        let (start, end) = source.time_range().unwrap();
        assert_eq!(start, 1000);
        assert_eq!(end, 3000);
    }

    #[test]
    fn test_vec_data_source_time_range_empty() {
        let source = VecDataSource::empty();
        assert!(source.time_range().is_none());
    }

    #[test]
    fn test_vec_data_source_push() {
        let mut source = VecDataSource::empty();
        source.push(create_test_tick(1000, dec!(100.0), dec!(100.1)));
        source.push(create_test_tick(1001, dec!(100.1), dec!(100.2)));

        assert_eq!(source.len(), 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_market_tick_serialization() {
        let tick = MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0));
        let json = serde_json::to_string(&tick).unwrap();
        let deserialized: MarketTick = serde_json::from_str(&json).unwrap();
        assert_eq!(tick, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ohlcv_bar_serialization() {
        let bar = OHLCVBar::new(
            1000,
            dec!(100.0),
            dec!(105.0),
            dec!(99.0),
            dec!(102.0),
            dec!(1000.0),
        );
        let json = serde_json::to_string(&bar).unwrap();
        let deserialized: OHLCVBar = serde_json::from_str(&json).unwrap();
        assert_eq!(bar, deserialized);
    }
}
