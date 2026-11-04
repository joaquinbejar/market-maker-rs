//! Data feed types and structures.
//!
//! This module defines the core types used for market data feeds.

use crate::Decimal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Price update event from market data feed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceUpdate {
    /// Trading symbol.
    pub symbol: String,
    /// Current price.
    pub price: Decimal,
    /// Update timestamp in milliseconds.
    pub timestamp: u64,
    /// Data source identifier.
    pub source: DataSource,
}

impl PriceUpdate {
    /// Creates a new price update.
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        price: Decimal,
        timestamp: u64,
        source: DataSource,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            price,
            timestamp,
            source,
        }
    }

    /// Creates a price update from an exchange source.
    #[must_use]
    pub fn from_exchange(
        symbol: impl Into<String>,
        price: Decimal,
        timestamp: u64,
        exchange: impl Into<String>,
    ) -> Self {
        Self::new(
            symbol,
            price,
            timestamp,
            DataSource::Exchange(exchange.into()),
        )
    }

    /// Creates a simulated price update.
    #[must_use]
    pub fn simulated(symbol: impl Into<String>, price: Decimal, timestamp: u64) -> Self {
        Self::new(symbol, price, timestamp, DataSource::Simulated)
    }
}

/// Trade event from market data feed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trade {
    /// Trading symbol.
    pub symbol: String,
    /// Trade price.
    pub price: Decimal,
    /// Trade quantity.
    pub quantity: Decimal,
    /// Trade side (buy/sell).
    pub side: TradeSide,
    /// Trade timestamp in milliseconds.
    pub timestamp: u64,
    /// Unique trade identifier.
    pub trade_id: String,
}

impl Trade {
    /// Creates a new trade.
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        price: Decimal,
        quantity: Decimal,
        side: TradeSide,
        timestamp: u64,
        trade_id: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            price,
            quantity,
            side,
            timestamp,
            trade_id: trade_id.into(),
        }
    }

    /// Returns the notional value of the trade.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.quantity)
    }

    /// Returns true if this is a buy trade.
    #[must_use]
    pub fn is_buy(&self) -> bool {
        matches!(self.side, TradeSide::Buy)
    }

    /// Returns true if this is a sell trade.
    #[must_use]
    pub fn is_sell(&self) -> bool {
        matches!(self.side, TradeSide::Sell)
    }
}

/// Trade side indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum TradeSide {
    /// Buy/bid side.
    Buy,
    /// Sell/ask side.
    Sell,
    /// Unknown side.
    #[default]
    Unknown,
}

/// Data source identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum DataSource {
    /// Data from a specific exchange.
    Exchange(String),
    /// Aggregated data from multiple sources.
    Aggregated,
    /// Simulated/mock data.
    #[default]
    Simulated,
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Exchange(name) => write!(f, "Exchange({})", name),
            DataSource::Aggregated => write!(f, "Aggregated"),
            DataSource::Simulated => write!(f, "Simulated"),
        }
    }
}

/// Implied volatility surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IVSurface {
    /// Underlying symbol.
    pub symbol: String,
    /// Surface timestamp in milliseconds.
    pub timestamp: u64,
    /// IV points on the surface.
    pub points: Vec<IVPoint>,
}

impl IVSurface {
    /// Creates a new IV surface.
    #[must_use]
    pub fn new(symbol: impl Into<String>, timestamp: u64) -> Self {
        Self {
            symbol: symbol.into(),
            timestamp,
            points: Vec::new(),
        }
    }

    /// Creates an IV surface with points.
    #[must_use]
    pub fn with_points(symbol: impl Into<String>, timestamp: u64, points: Vec<IVPoint>) -> Self {
        Self {
            symbol: symbol.into(),
            timestamp,
            points,
        }
    }

    /// Adds a point to the surface.
    pub fn add_point(&mut self, point: IVPoint) {
        self.points.push(point);
    }

    /// Gets IV for a specific strike and expiration.
    ///
    /// Returns the exact match if found, otherwise None.
    #[must_use]
    pub fn get_iv(&self, strike: Decimal, expiration_days: u32) -> Option<Decimal> {
        self.points
            .iter()
            .find(|p| p.strike == strike && p.expiration_days == expiration_days)
            .map(|p| p.iv)
    }

    /// Gets all points for a specific expiration.
    #[must_use]
    pub fn points_for_expiration(&self, expiration_days: u32) -> Vec<&IVPoint> {
        self.points
            .iter()
            .filter(|p| p.expiration_days == expiration_days)
            .collect()
    }

    /// Gets all unique expirations in the surface.
    #[must_use]
    pub fn expirations(&self) -> Vec<u32> {
        let mut exps: Vec<u32> = self.points.iter().map(|p| p.expiration_days).collect();
        exps.sort_unstable();
        exps.dedup();
        exps
    }

    /// Returns the number of points in the surface.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the surface has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Single point on the IV surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IVPoint {
    /// Strike price.
    pub strike: Decimal,
    /// Days to expiration.
    pub expiration_days: u32,
    /// Implied volatility (as decimal, e.g., 0.30 for 30%).
    pub iv: Decimal,
}

impl IVPoint {
    /// Creates a new IV point.
    #[must_use]
    pub fn new(strike: Decimal, expiration_days: u32, iv: Decimal) -> Self {
        Self {
            strike,
            expiration_days,
            iv,
        }
    }
}

/// IV surface update event.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IVSurfaceUpdate {
    /// Underlying symbol.
    pub symbol: String,
    /// Update timestamp in milliseconds.
    pub timestamp: u64,
    /// Updated IV points.
    pub updated_points: Vec<IVPoint>,
}

impl IVSurfaceUpdate {
    /// Creates a new IV surface update.
    #[must_use]
    pub fn new(symbol: impl Into<String>, timestamp: u64, updated_points: Vec<IVPoint>) -> Self {
        Self {
            symbol: symbol.into(),
            timestamp,
            updated_points,
        }
    }

    /// Returns the number of updated points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.updated_points.len()
    }

    /// Returns true if no points were updated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.updated_points.is_empty()
    }
}

/// Market snapshot with current prices and volume.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MarketSnapshot {
    /// Trading symbol.
    pub symbol: String,
    /// Best bid price.
    pub bid: Decimal,
    /// Best ask price.
    pub ask: Decimal,
    /// Last trade price.
    pub last: Decimal,
    /// 24h trading volume.
    pub volume: Decimal,
    /// Snapshot timestamp in milliseconds.
    pub timestamp: u64,
}

impl MarketSnapshot {
    /// Creates a new market snapshot.
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        bid: Decimal,
        ask: Decimal,
        last: Decimal,
        volume: Decimal,
        timestamp: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            bid,
            ask,
            last,
            volume,
            timestamp,
        }
    }

    /// Returns the mid price.
    ///
    /// Exact: computed without overflow, never saturated.
    #[must_use]
    pub fn mid_price(&self) -> Decimal {
        crate::types::decimal::midpoint(self.bid, self.ask)
    }

    /// Returns the spread.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn spread(&self) -> Decimal {
        self.ask.saturating_sub(self.bid)
    }

    /// Returns the spread in basis points.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_market_snapshot_spread_bps_overflow_is_a_signed_bound() {
        let wide = |bid, ask| MarketSnapshot::new("BTC", bid, ask, dec!(0), dec!(0), 0);
        let (low, high) = (dec!(-5), dec!(5.0000000000000000000000000002));
        assert_eq!(wide(low, high).spread_bps(), Decimal::MAX);
        assert_eq!(wide(high, low).spread_bps(), Decimal::MIN);
    }

    #[test]
    fn test_price_update_new() {
        let update = PriceUpdate::new("BTC", dec!(50000.0), 1000, DataSource::Simulated);
        assert_eq!(update.symbol, "BTC");
        assert_eq!(update.price, dec!(50000.0));
        assert_eq!(update.timestamp, 1000);
    }

    #[test]
    fn test_price_update_from_exchange() {
        let update = PriceUpdate::from_exchange("BTC", dec!(50000.0), 1000, "Binance");
        assert_eq!(update.source, DataSource::Exchange("Binance".to_string()));
    }

    #[test]
    fn test_trade_new() {
        let trade = Trade::new(
            "BTC",
            dec!(50000.0),
            dec!(1.0),
            TradeSide::Buy,
            1000,
            "trade-1",
        );
        assert_eq!(trade.notional(), dec!(50000.0));
        assert!(trade.is_buy());
        assert!(!trade.is_sell());
    }

    #[test]
    fn test_iv_surface() {
        let mut surface = IVSurface::new("BTC", 1000);
        surface.add_point(IVPoint::new(dec!(50000.0), 30, dec!(0.60)));
        surface.add_point(IVPoint::new(dec!(55000.0), 30, dec!(0.55)));
        surface.add_point(IVPoint::new(dec!(50000.0), 60, dec!(0.58)));

        assert_eq!(surface.len(), 3);
        assert_eq!(surface.get_iv(dec!(50000.0), 30), Some(dec!(0.60)));
        assert_eq!(surface.get_iv(dec!(50000.0), 60), Some(dec!(0.58)));
        assert_eq!(surface.get_iv(dec!(45000.0), 30), None);

        let exps = surface.expirations();
        assert_eq!(exps, vec![30, 60]);

        let points_30 = surface.points_for_expiration(30);
        assert_eq!(points_30.len(), 2);
    }

    #[test]
    fn test_market_snapshot() {
        let snapshot = MarketSnapshot::new(
            "BTC",
            dec!(49990.0),
            dec!(50010.0),
            dec!(50000.0),
            dec!(1000.0),
            1000,
        );
        assert_eq!(snapshot.mid_price(), dec!(50000.0));
        assert_eq!(snapshot.spread(), dec!(20.0));
        assert_eq!(snapshot.spread_bps(), dec!(4.0)); // 20/50000 * 10000 = 4 bps
    }

    #[test]
    fn test_data_source_display() {
        assert_eq!(
            DataSource::Exchange("Binance".to_string()).to_string(),
            "Exchange(Binance)"
        );
        assert_eq!(DataSource::Aggregated.to_string(), "Aggregated");
        assert_eq!(DataSource::Simulated.to_string(), "Simulated");
    }
}
