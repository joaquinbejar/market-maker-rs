//! Market data feed trait and configuration.
//!
//! This module defines the abstract `MarketDataFeed` trait for different data providers.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::data_feeds::types::{IVSurface, IVSurfaceUpdate, MarketSnapshot, PriceUpdate, Trade};
use crate::types::error::MMResult;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reconnection configuration for data feeds.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReconnectConfig {
    /// Initial delay before reconnection attempt in milliseconds.
    pub initial_delay_ms: u64,
    /// Maximum delay between reconnection attempts in milliseconds.
    pub max_delay_ms: u64,
    /// Backoff multiplier for exponential backoff.
    pub backoff_multiplier: f64,
    /// Maximum number of reconnection attempts (None = infinite).
    pub max_attempts: Option<u32>,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            max_attempts: None,
        }
    }
}

impl ReconnectConfig {
    /// Creates a new reconnect configuration.
    #[must_use]
    pub fn new(
        initial_delay_ms: u64,
        max_delay_ms: u64,
        backoff_multiplier: f64,
        max_attempts: Option<u32>,
    ) -> Self {
        Self {
            initial_delay_ms,
            max_delay_ms,
            backoff_multiplier,
            max_attempts,
        }
    }

    /// Calculates the delay for a given attempt number.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let exponent = i32::try_from(attempt).unwrap_or(i32::MAX);
        // Stats boundary, not money: a delay in ms is exact below 2^53, and
        // the float to int `as` cast saturates (NaN maps to 0) before the cap.
        let delay = self.initial_delay_ms as f64 * self.backoff_multiplier.powi(exponent);
        (delay as u64).min(self.max_delay_ms)
    }

    /// Returns true if another attempt should be made.
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        match self.max_attempts {
            Some(max) => attempt < max,
            None => true,
        }
    }
}

/// Abstract trait for market data providers.
///
/// This trait defines the interface for subscribing to and receiving
/// real-time market data updates from various sources.
///
/// # Example
///
/// ```rust,ignore
/// use market_maker_rs::data_feeds::{MarketDataFeed, MockDataFeed};
///
/// let feed = MockDataFeed::new();
///
/// // Subscribe to price updates
/// let mut rx = feed.subscribe_underlying("BTC").await?;
///
/// // Process updates
/// while let Some(update) = rx.recv().await {
///     println!("Price: {}", update.price);
/// }
/// ```
#[async_trait]
pub trait MarketDataFeed: Send + Sync {
    /// Subscribe to underlying price updates.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol to subscribe to
    ///
    /// # Returns
    ///
    /// A receiver channel for price updates.
    async fn subscribe_underlying(&self, symbol: &str) -> MMResult<mpsc::Receiver<PriceUpdate>>;

    /// Subscribe to trade feed.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol to subscribe to
    ///
    /// # Returns
    ///
    /// A receiver channel for trade events.
    async fn subscribe_trades(&self, symbol: &str) -> MMResult<mpsc::Receiver<Trade>>;

    /// Get current implied volatility surface.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The underlying symbol
    ///
    /// # Returns
    ///
    /// The current IV surface for the symbol.
    async fn get_iv_surface(&self, symbol: &str) -> MMResult<IVSurface>;

    /// Subscribe to IV surface updates.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The underlying symbol
    ///
    /// # Returns
    ///
    /// A receiver channel for IV surface updates.
    async fn subscribe_iv_surface(&self, symbol: &str)
    -> MMResult<mpsc::Receiver<IVSurfaceUpdate>>;

    /// Get current market snapshot.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    ///
    /// # Returns
    ///
    /// The current market snapshot.
    async fn get_snapshot(&self, symbol: &str) -> MMResult<MarketSnapshot>;

    /// Disconnect from data feed.
    ///
    /// Closes all subscriptions and releases resources.
    async fn disconnect(&self) -> MMResult<()>;

    /// Check if the feed is connected.
    fn is_connected(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconnect_config_default() {
        let config = ReconnectConfig::default();
        assert_eq!(config.initial_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30000);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert_eq!(config.max_attempts, None);
    }

    #[test]
    fn test_reconnect_config_delay_for_attempt() {
        let config = ReconnectConfig::default();

        assert_eq!(config.delay_for_attempt(0), 1000);
        assert_eq!(config.delay_for_attempt(1), 2000);
        assert_eq!(config.delay_for_attempt(2), 4000);
        assert_eq!(config.delay_for_attempt(3), 8000);
        assert_eq!(config.delay_for_attempt(4), 16000);
        assert_eq!(config.delay_for_attempt(5), 30000); // Capped at max
        assert_eq!(config.delay_for_attempt(10), 30000); // Still capped
    }

    #[test]
    fn test_reconnect_config_should_retry() {
        let unlimited = ReconnectConfig::default();
        assert!(unlimited.should_retry(0));
        assert!(unlimited.should_retry(100));

        let limited = ReconnectConfig::new(1000, 30000, 2.0, Some(3));
        assert!(limited.should_retry(0));
        assert!(limited.should_retry(2));
        assert!(!limited.should_retry(3));
        assert!(!limited.should_retry(10));
    }
}
