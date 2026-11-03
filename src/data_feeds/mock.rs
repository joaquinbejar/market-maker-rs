//! Mock data feed for testing.
//!
//! This module provides a mock implementation of `MarketDataFeed` for testing
//! without requiring real exchange connectivity.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{RwLock, mpsc};

use crate::Decimal;
use crate::data_feeds::feed::MarketDataFeed;
use crate::data_feeds::types::{
    DataSource, IVPoint, IVSurface, IVSurfaceUpdate, MarketSnapshot, PriceUpdate, Trade,
};
use crate::types::error::{MMError, MMResult};

// Type aliases for cleaner code
type PriceMap = HashMap<String, Decimal>;
type IVSurfaceMap = HashMap<String, IVSurface>;
type SnapshotMap = HashMap<String, MarketSnapshot>;
type PriceSubsMap = HashMap<String, Vec<mpsc::Sender<PriceUpdate>>>;
type TradeSubsMap = HashMap<String, Vec<mpsc::Sender<Trade>>>;
type IVSubsMap = HashMap<String, Vec<mpsc::Sender<IVSurfaceUpdate>>>;

/// Mock data feed for testing.
///
/// Provides a simulated market data feed that can be controlled programmatically
/// for testing purposes.
pub struct MockDataFeed {
    prices: Arc<RwLock<PriceMap>>,
    iv_surfaces: Arc<RwLock<IVSurfaceMap>>,
    snapshots: Arc<RwLock<SnapshotMap>>,
    price_subs: Arc<RwLock<PriceSubsMap>>,
    trade_subs: Arc<RwLock<TradeSubsMap>>,
    iv_subs: Arc<RwLock<IVSubsMap>>,
    connected: AtomicBool,
    buffer_size: usize,
}

impl Default for MockDataFeed {
    fn default() -> Self {
        Self::new()
    }
}

impl MockDataFeed {
    /// Creates a new mock data feed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prices: Arc::new(RwLock::new(HashMap::new())),
            iv_surfaces: Arc::new(RwLock::new(HashMap::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            price_subs: Arc::new(RwLock::new(HashMap::new())),
            trade_subs: Arc::new(RwLock::new(HashMap::new())),
            iv_subs: Arc::new(RwLock::new(HashMap::new())),
            connected: AtomicBool::new(true),
            buffer_size: 100,
        }
    }

    /// Creates a mock data feed with custom buffer size.
    #[must_use]
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            ..Self::new()
        }
    }

    /// Sets the current price for a symbol.
    pub async fn set_price(&self, symbol: &str, price: Decimal) {
        let mut prices: tokio::sync::RwLockWriteGuard<'_, PriceMap> = self.prices.write().await;
        prices.insert(symbol.to_string(), price);
    }

    /// Gets the current price for a symbol.
    pub async fn get_price(&self, symbol: &str) -> Option<Decimal> {
        let prices: tokio::sync::RwLockReadGuard<'_, PriceMap> = self.prices.read().await;
        prices.get(symbol).copied()
    }

    /// Sets the IV surface for a symbol.
    pub async fn set_iv_surface(&self, symbol: &str, surface: IVSurface) {
        let mut surfaces: tokio::sync::RwLockWriteGuard<'_, IVSurfaceMap> =
            self.iv_surfaces.write().await;
        surfaces.insert(symbol.to_string(), surface);
    }

    /// Sets the market snapshot for a symbol.
    pub async fn set_snapshot(&self, symbol: &str, snapshot: MarketSnapshot) {
        let mut snapshots: tokio::sync::RwLockWriteGuard<'_, SnapshotMap> =
            self.snapshots.write().await;
        snapshots.insert(symbol.to_string(), snapshot);
    }

    /// Simulates a price update.
    pub async fn simulate_price_update(&self, symbol: &str, price: Decimal) {
        let timestamp: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);

        let update = PriceUpdate::new(symbol, price, timestamp, DataSource::Simulated);
        self.set_price(symbol, price).await;

        let subs: tokio::sync::RwLockReadGuard<'_, PriceSubsMap> = self.price_subs.read().await;
        if let Some(subscribers) = subs.get(symbol) {
            for sender in subscribers {
                let _ = sender.try_send(update.clone());
            }
        }
    }

    /// Simulates a trade event.
    pub async fn simulate_trade(&self, trade: Trade) {
        let symbol = trade.symbol.clone();
        let subs: tokio::sync::RwLockReadGuard<'_, TradeSubsMap> = self.trade_subs.read().await;
        if let Some(subscribers) = subs.get(&symbol) {
            for sender in subscribers {
                let _ = sender.try_send(trade.clone());
            }
        }
    }

    /// Simulates an IV surface update.
    pub async fn simulate_iv_update(&self, symbol: &str, updated_points: Vec<IVPoint>) {
        let timestamp: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);

        let update = IVSurfaceUpdate::new(symbol, timestamp, updated_points.clone());

        {
            let mut surfaces: tokio::sync::RwLockWriteGuard<'_, IVSurfaceMap> =
                self.iv_surfaces.write().await;
            if let Some(surface) = surfaces.get_mut(symbol) {
                for point in updated_points {
                    if let Some(existing) = surface.points.iter_mut().find(|p| {
                        p.strike == point.strike && p.expiration_days == point.expiration_days
                    }) {
                        existing.iv = point.iv;
                    } else {
                        surface.points.push(point);
                    }
                }
                surface.timestamp = timestamp;
            }
        }

        let subs: tokio::sync::RwLockReadGuard<'_, IVSubsMap> = self.iv_subs.read().await;
        if let Some(subscribers) = subs.get(symbol) {
            for sender in subscribers {
                let _ = sender.try_send(update.clone());
            }
        }
    }

    /// Sets the connection status.
    pub fn set_connected(&self, connected: bool) {
        self.connected.store(connected, Ordering::SeqCst);
    }

    /// Clears all subscriptions.
    pub async fn clear_subscriptions(&self) {
        let mut price_subs: tokio::sync::RwLockWriteGuard<'_, PriceSubsMap> =
            self.price_subs.write().await;
        price_subs.clear();
        drop(price_subs);

        let mut trade_subs: tokio::sync::RwLockWriteGuard<'_, TradeSubsMap> =
            self.trade_subs.write().await;
        trade_subs.clear();
        drop(trade_subs);

        let mut iv_subs: tokio::sync::RwLockWriteGuard<'_, IVSubsMap> = self.iv_subs.write().await;
        iv_subs.clear();
    }

    /// Returns the number of price subscribers for a symbol.
    pub async fn price_subscriber_count(&self, symbol: &str) -> usize {
        let subs: tokio::sync::RwLockReadGuard<'_, PriceSubsMap> = self.price_subs.read().await;
        subs.get(symbol).map_or(0, Vec::len)
    }

    /// Returns the number of trade subscribers for a symbol.
    pub async fn trade_subscriber_count(&self, symbol: &str) -> usize {
        let subs: tokio::sync::RwLockReadGuard<'_, TradeSubsMap> = self.trade_subs.read().await;
        subs.get(symbol).map_or(0, Vec::len)
    }
}

#[async_trait]
impl MarketDataFeed for MockDataFeed {
    async fn subscribe_underlying(&self, symbol: &str) -> MMResult<mpsc::Receiver<PriceUpdate>> {
        if !self.is_connected() {
            return Err(MMError::ConnectionError("Feed not connected".to_string()));
        }

        let (tx, rx) = mpsc::channel(self.buffer_size);
        let mut subs: tokio::sync::RwLockWriteGuard<'_, PriceSubsMap> =
            self.price_subs.write().await;
        subs.entry(symbol.to_string()).or_default().push(tx);
        Ok(rx)
    }

    async fn subscribe_trades(&self, symbol: &str) -> MMResult<mpsc::Receiver<Trade>> {
        if !self.is_connected() {
            return Err(MMError::ConnectionError("Feed not connected".to_string()));
        }

        let (tx, rx) = mpsc::channel(self.buffer_size);
        let mut subs: tokio::sync::RwLockWriteGuard<'_, TradeSubsMap> =
            self.trade_subs.write().await;
        subs.entry(symbol.to_string()).or_default().push(tx);
        Ok(rx)
    }

    async fn get_iv_surface(&self, symbol: &str) -> MMResult<IVSurface> {
        if !self.is_connected() {
            return Err(MMError::ConnectionError("Feed not connected".to_string()));
        }

        let surfaces: tokio::sync::RwLockReadGuard<'_, IVSurfaceMap> =
            self.iv_surfaces.read().await;
        surfaces
            .get(symbol)
            .cloned()
            .ok_or_else(|| MMError::InvalidMarketState(format!("No IV surface for {}", symbol)))
    }

    async fn subscribe_iv_surface(
        &self,
        symbol: &str,
    ) -> MMResult<mpsc::Receiver<IVSurfaceUpdate>> {
        if !self.is_connected() {
            return Err(MMError::ConnectionError("Feed not connected".to_string()));
        }

        let (tx, rx) = mpsc::channel(self.buffer_size);
        let mut subs: tokio::sync::RwLockWriteGuard<'_, IVSubsMap> = self.iv_subs.write().await;
        subs.entry(symbol.to_string()).or_default().push(tx);
        Ok(rx)
    }

    async fn get_snapshot(&self, symbol: &str) -> MMResult<MarketSnapshot> {
        if !self.is_connected() {
            return Err(MMError::ConnectionError("Feed not connected".to_string()));
        }

        let snapshots: tokio::sync::RwLockReadGuard<'_, SnapshotMap> = self.snapshots.read().await;
        snapshots
            .get(symbol)
            .cloned()
            .ok_or_else(|| MMError::InvalidMarketState(format!("No snapshot for {}", symbol)))
    }

    async fn disconnect(&self) -> MMResult<()> {
        self.set_connected(false);
        self.clear_subscriptions().await;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_feeds::types::TradeSide;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_mock_feed_new() {
        let feed = MockDataFeed::new();
        assert!(feed.is_connected());
    }

    #[tokio::test]
    async fn test_mock_feed_set_get_price() {
        let feed = MockDataFeed::new();
        feed.set_price("BTC", dec!(50000.0)).await;
        assert_eq!(feed.get_price("BTC").await, Some(dec!(50000.0)));
        assert_eq!(feed.get_price("ETH").await, None);
    }

    #[tokio::test]
    async fn test_mock_feed_subscribe_underlying() {
        let feed = MockDataFeed::new();
        let mut rx = feed.subscribe_underlying("BTC").await.unwrap();
        assert_eq!(feed.price_subscriber_count("BTC").await, 1);

        feed.simulate_price_update("BTC", dec!(50000.0)).await;
        let update = rx.recv().await.unwrap();
        assert_eq!(update.symbol, "BTC");
        assert_eq!(update.price, dec!(50000.0));
    }

    #[tokio::test]
    async fn test_mock_feed_subscribe_trades() {
        let feed = MockDataFeed::new();
        let mut rx = feed.subscribe_trades("BTC").await.unwrap();
        assert_eq!(feed.trade_subscriber_count("BTC").await, 1);

        let trade = Trade::new(
            "BTC",
            dec!(50000.0),
            dec!(1.0),
            TradeSide::Buy,
            1000,
            "trade-1",
        );
        feed.simulate_trade(trade).await;

        let received = rx.recv().await.unwrap();
        assert_eq!(received.symbol, "BTC");
        assert!(received.is_buy());
    }

    #[tokio::test]
    async fn test_mock_feed_iv_surface() {
        let feed = MockDataFeed::new();
        let mut surface = IVSurface::new("BTC", 1000);
        surface.add_point(IVPoint::new(dec!(50000.0), 30, dec!(0.60)));
        feed.set_iv_surface("BTC", surface).await;

        let retrieved = feed.get_iv_surface("BTC").await.unwrap();
        assert_eq!(retrieved.symbol, "BTC");
        assert_eq!(retrieved.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_feed_snapshot() {
        let feed = MockDataFeed::new();
        let snapshot = MarketSnapshot::new(
            "BTC",
            dec!(49990.0),
            dec!(50010.0),
            dec!(50000.0),
            dec!(1000.0),
            1000,
        );
        feed.set_snapshot("BTC", snapshot).await;

        let retrieved = feed.get_snapshot("BTC").await.unwrap();
        assert_eq!(retrieved.mid_price(), dec!(50000.0));
    }

    #[tokio::test]
    async fn test_mock_feed_disconnect() {
        let feed = MockDataFeed::new();
        let _ = feed.subscribe_underlying("BTC").await.unwrap();
        assert_eq!(feed.price_subscriber_count("BTC").await, 1);

        feed.disconnect().await.unwrap();
        assert!(!feed.is_connected());
        assert_eq!(feed.price_subscriber_count("BTC").await, 0);
    }

    #[tokio::test]
    async fn test_mock_feed_not_connected_error() {
        let feed = MockDataFeed::new();
        feed.set_connected(false);
        let result = feed.subscribe_underlying("BTC").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_feed_multiple_subscribers() {
        let feed = MockDataFeed::new();
        let mut rx1 = feed.subscribe_underlying("BTC").await.unwrap();
        let mut rx2 = feed.subscribe_underlying("BTC").await.unwrap();
        assert_eq!(feed.price_subscriber_count("BTC").await, 2);

        feed.simulate_price_update("BTC", dec!(50000.0)).await;

        let update1 = rx1.recv().await.unwrap();
        let update2 = rx2.recv().await.unwrap();
        assert_eq!(update1.price, dec!(50000.0));
        assert_eq!(update2.price, dec!(50000.0));
    }
}
