//! Mock exchange connector for testing.
//!
//! This module provides a mock implementation of the `ExchangeConnector` trait
//! for use in unit and integration tests.
//!
//! # Features
//!
//! - Configurable latency simulation
//! - Failure injection for testing error handling
//! - Order tracking and state management
//! - Simulated order book
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     MockExchangeConnector, MockConfig, OrderRequest
//! };
//! use market_maker_rs::dec;
//!
//! let config = MockConfig::default();
//! let connector = MockExchangeConnector::new(config);
//!
//! let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
//! // In an async context: connector.submit_order(request).await
//! ```

use async_trait::async_trait;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

use super::connector::{
    BookLevel, ExchangeConnector, Fill, MarketDataStream, OrderBookSnapshot, OrderId, OrderRequest,
    OrderResponse, OrderStatus,
};
use crate::types::action::{Action, Dir};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the mock exchange connector.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::MockConfig;
///
/// let config = MockConfig::default()
///     .with_latency_ms(10)
///     .with_failure_rate(0.01);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MockConfig {
    /// Simulated latency in milliseconds.
    pub latency_ms: u64,

    /// Probability of random failure (0.0 to 1.0).
    pub failure_rate: f64,

    /// Initial balances by asset.
    pub initial_balances: HashMap<String, Decimal>,

    /// Default order book depth.
    pub default_depth: usize,

    /// Base price for simulated order book.
    pub base_price: Decimal,

    /// Spread for simulated order book (as decimal, e.g., 0.001 = 0.1%).
    pub spread: Decimal,
}

/// Maps a poisoned lock to [`MMError::LockPoisoned`].
fn poisoned<T>(_: std::sync::PoisonError<T>) -> MMError {
    MMError::LockPoisoned("mock connector".to_string())
}

impl Default for MockConfig {
    fn default() -> Self {
        let mut balances = HashMap::new();
        balances.insert("USD".to_string(), Decimal::from(100_000));
        balances.insert("BTC".to_string(), Decimal::from(10));

        Self {
            latency_ms: 0,
            failure_rate: 0.0,
            initial_balances: balances,
            default_depth: 10,
            base_price: Decimal::from(50_000),
            spread: dec!(0.001),
        }
    }
}

impl MockConfig {
    /// Creates a new mock config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the simulated latency.
    #[must_use]
    pub fn with_latency_ms(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Sets the failure rate.
    #[must_use]
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the base price for the simulated order book.
    #[must_use]
    pub fn with_base_price(mut self, price: Decimal) -> Self {
        self.base_price = price;
        self
    }

    /// Sets the spread for the simulated order book.
    #[must_use]
    pub fn with_spread(mut self, spread: Decimal) -> Self {
        self.spread = spread;
        self
    }

    /// Sets an initial balance for an asset.
    #[must_use]
    pub fn with_balance(mut self, asset: impl Into<String>, balance: Decimal) -> Self {
        self.initial_balances.insert(asset.into(), balance);
        self
    }
}

/// Internal order state for tracking.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OrderState {
    request: OrderRequest,
    status: OrderStatus,
    order_id: OrderId,
    timestamp: u64,
}

/// Mock exchange connector for testing.
///
/// Provides a simulated exchange environment for testing strategies
/// without connecting to a real exchange.
///
/// # Thread Safety
///
/// This implementation is thread-safe and can be shared across async tasks.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{
///     MockExchangeConnector, MockConfig, OrderRequest
/// };
/// use market_maker_rs::dec;
///
/// let connector = MockExchangeConnector::new(MockConfig::default());
///
/// // Create an order request
/// let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
///
/// // In an async context, you would call:
/// // let response = connector.submit_order(request).await.unwrap();
/// ```
#[derive(Debug)]
pub struct MockExchangeConnector {
    config: MockConfig,
    orders: RwLock<HashMap<String, OrderState>>,
    balances: RwLock<HashMap<String, Decimal>>,
    order_counter: AtomicU64,
    current_time: AtomicU64,
    /// The symbol of the last trade subscription, which simulated trades name.
    trade_symbol: RwLock<Option<String>>,
}

impl MockExchangeConnector {
    /// Creates a new mock exchange connector.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the mock connector
    #[must_use]
    pub fn new(config: MockConfig) -> Self {
        let balances = RwLock::new(config.initial_balances.clone());
        Self {
            config,
            orders: RwLock::new(HashMap::new()),
            balances,
            order_counter: AtomicU64::new(1),
            current_time: AtomicU64::new(1_000_000),
            trade_symbol: RwLock::new(None),
        }
    }

    /// Creates a mock connector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(MockConfig::default())
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &MockConfig {
        &self.config
    }

    /// Sets the current simulated time.
    pub fn set_time(&self, time: u64) {
        self.current_time.store(time, Ordering::SeqCst);
    }

    /// Advances the simulated time.
    pub fn advance_time(&self, delta: u64) {
        self.current_time.fetch_add(delta, Ordering::SeqCst);
    }

    /// Gets the current simulated time.
    #[must_use]
    pub fn current_time(&self) -> u64 {
        self.current_time.load(Ordering::SeqCst)
    }

    /// Sets the balance for an asset.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn set_balance(&self, asset: &str, balance: Decimal) -> MMResult<()> {
        let mut balances = self.balances.write().map_err(poisoned)?;
        balances.insert(asset.to_string(), balance);
        Ok(())
    }

    /// Generates a new unique order ID.
    fn next_order_id(&self) -> OrderId {
        let id = self.order_counter.fetch_add(1, Ordering::SeqCst);
        OrderId::new(format!("mock-{}", id))
    }

    /// Simulates latency if configured.
    /// Note: Latency simulation is a no-op in the library.
    /// For actual latency simulation, use tokio::time::sleep in your test code.
    async fn simulate_latency(&self) {
        // Latency simulation is intentionally a no-op in the library.
        // Tests can add their own delays if needed.
        let _ = self.config.latency_ms;
    }

    /// Checks if a random failure should occur.
    fn should_fail(&self) -> bool {
        if self.config.failure_rate > 0.0 {
            // Simple deterministic "random" based on order counter
            let counter = self.order_counter.load(Ordering::SeqCst);
            // `bucket < trunc(rate * 100)` is `bucket + 1 <= rate * 100`
            // for an integer bucket, without a float to int cast.
            let bucket = u8::try_from(counter % 100).unwrap_or(u8::MAX);
            f64::from(bucket) + 1.0 <= self.config.failure_rate * 100.0
        } else {
            false
        }
    }

    /// Generates a simulated order book.
    ///
    /// A synthetic book for tests: its prices and sizes saturate at the
    /// `Decimal` bounds for extreme configurations.
    fn generate_orderbook(&self, symbol: &str, depth: usize) -> OrderBookSnapshot {
        let base_price = self.config.base_price;
        let half_spread = base_price
            .saturating_mul(self.config.spread)
            .saturating_mul(dec!(0.5));

        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);

        let tick_size = base_price.saturating_mul(dec!(0.0001));

        for i in 0..depth {
            let offset = tick_size.saturating_mul(Decimal::from(i));
            let qty = Decimal::ONE.saturating_add(Decimal::from(i).saturating_mul(dec!(0.5)));

            // Bid prices descending from mid - half_spread
            let bid_price = base_price
                .saturating_sub(half_spread)
                .saturating_sub(offset);
            let bid_qty = qty;
            bids.push(BookLevel::new(bid_price, bid_qty));

            // Ask prices ascending from mid + half_spread
            let ask_price = base_price
                .saturating_add(half_spread)
                .saturating_add(offset);
            let ask_qty = qty;
            asks.push(BookLevel::new(ask_price, ask_qty));
        }

        OrderBookSnapshot {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: self.current_time(),
        }
    }

    /// Simulates a fill for a market order.
    fn simulate_market_fill(&self, request: &OrderRequest, dir: Dir) -> (OrderStatus, Decimal) {
        let orderbook = self.generate_orderbook(&request.symbol, 1);

        let fill_price = match dir {
            Dir::Buy => orderbook.best_ask().unwrap_or(self.config.base_price),
            Dir::Sell => orderbook.best_bid().unwrap_or(self.config.base_price),
        };

        let status = OrderStatus::Filled {
            filled_qty: request.quantity,
            avg_price: fill_price,
        };

        (status, fill_price)
    }

    /// Gets the number of open orders.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn open_order_count(&self) -> MMResult<usize> {
        let orders = self.orders.read().map_err(poisoned)?;
        Ok(orders.values().filter(|o| o.status.is_open()).count())
    }

    /// Gets all order IDs.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn all_order_ids(&self) -> MMResult<Vec<OrderId>> {
        let orders = self.orders.read().map_err(poisoned)?;
        Ok(orders.values().map(|o| o.order_id.clone()).collect())
    }
}

#[async_trait]
impl ExchangeConnector for MockExchangeConnector {
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderResponse> {
        // Validated before any simulated latency, failure or id allocation.
        let dir = Dir::try_from(request.side)?;
        self.simulate_latency().await;

        if self.should_fail() {
            return Err(MMError::InvalidMarketState(
                "simulated exchange failure".to_string(),
            ));
        }

        let order_id = self.next_order_id();
        let timestamp = self.current_time();

        let status = if request.order_type.is_market() {
            let (status, _) = self.simulate_market_fill(&request, dir);
            status
        } else {
            OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            }
        };

        let state = OrderState {
            request: request.clone(),
            status: status.clone(),
            order_id: order_id.clone(),
            timestamp,
        };

        {
            let mut orders = self.orders.write().map_err(poisoned)?;
            orders.insert(order_id.as_str().to_string(), state);
        }

        Ok(OrderResponse {
            order_id,
            client_order_id: request.client_order_id,
            status,
            timestamp,
        })
    }

    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        self.simulate_latency().await;

        if self.should_fail() {
            return Err(MMError::InvalidMarketState(
                "simulated exchange failure".to_string(),
            ));
        }

        let mut orders = self.orders.write().map_err(poisoned)?;
        let state = orders
            .get_mut(order_id.as_str())
            .ok_or_else(|| MMError::InvalidMarketState(format!("order not found: {}", order_id)))?;

        if state.status.is_terminal() {
            return Err(MMError::InvalidMarketState(format!(
                "order already terminal: {}",
                order_id
            )));
        }

        let filled_qty = state.status.filled_qty();
        state.status = OrderStatus::Cancelled { filled_qty };

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: state.request.client_order_id.clone(),
            status: state.status.clone(),
            timestamp: self.current_time(),
        })
    }

    async fn modify_order(
        &self,
        order_id: &OrderId,
        new_price: Option<Decimal>,
        new_quantity: Option<Decimal>,
    ) -> MMResult<OrderResponse> {
        self.simulate_latency().await;

        // Cancel the existing order
        let cancelled = self.cancel_order(order_id).await?;

        // Get the original request
        let original_request = {
            let orders = self.orders.read().map_err(poisoned)?;
            let state = orders.get(order_id.as_str()).ok_or_else(|| {
                MMError::InvalidMarketState(format!("order not found: {}", order_id))
            })?;
            state.request.clone()
        };

        // Create a new order with modified parameters
        let new_request = OrderRequest {
            price: new_price.or(original_request.price),
            quantity: new_quantity.unwrap_or(original_request.quantity),
            ..original_request
        };

        // Submit the new order
        let mut response = self.submit_order(new_request).await?;

        // Preserve the client order ID
        response.client_order_id = cancelled.client_order_id;

        Ok(response)
    }

    async fn get_order_status(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        self.simulate_latency().await;

        let orders = self.orders.read().map_err(poisoned)?;
        let state = orders
            .get(order_id.as_str())
            .ok_or_else(|| MMError::InvalidMarketState(format!("order not found: {}", order_id)))?;

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: state.request.client_order_id.clone(),
            status: state.status.clone(),
            timestamp: self.current_time(),
        })
    }

    async fn get_open_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>> {
        self.simulate_latency().await;

        let orders = self.orders.read().map_err(poisoned)?;
        let open_orders: Vec<OrderResponse> = orders
            .values()
            .filter(|state| state.request.symbol == symbol && state.status.is_open())
            .map(|state| OrderResponse {
                order_id: state.order_id.clone(),
                client_order_id: state.request.client_order_id.clone(),
                status: state.status.clone(),
                timestamp: self.current_time(),
            })
            .collect();

        Ok(open_orders)
    }

    async fn cancel_all_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>> {
        self.simulate_latency().await;

        let order_ids: Vec<OrderId> = {
            let orders = self.orders.read().map_err(poisoned)?;
            orders
                .values()
                .filter(|state| state.request.symbol == symbol && state.status.is_open())
                .map(|state| state.order_id.clone())
                .collect()
        };

        let mut responses = Vec::new();
        let mut first_error = None;
        for order_id in order_ids {
            match self.cancel_order(&order_id).await {
                Ok(response) => responses.push(response),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(responses),
        }
    }

    async fn get_orderbook(&self, symbol: &str, depth: usize) -> MMResult<OrderBookSnapshot> {
        self.simulate_latency().await;

        if self.should_fail() {
            return Err(MMError::InvalidMarketState(
                "simulated exchange failure".to_string(),
            ));
        }

        Ok(self.generate_orderbook(symbol, depth))
    }

    async fn get_balance(&self, asset: &str) -> MMResult<Decimal> {
        self.simulate_latency().await;

        let balances = self.balances.read().map_err(poisoned)?;
        Ok(*balances.get(asset).unwrap_or(&Decimal::ZERO))
    }
}

#[async_trait]
impl MarketDataStream for MockExchangeConnector {
    async fn subscribe_orderbook(&self, _symbol: &str) -> MMResult<()> {
        // Mock implementation - always succeeds
        Ok(())
    }

    /// Simulated trades name `symbol`; a later subscription replaces it.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for an empty symbol, and
    /// [`MMError::LockPoisoned`] if the lock is poisoned.
    async fn subscribe_trades(&self, symbol: &str) -> MMResult<()> {
        if symbol.is_empty() {
            return Err(MMError::InvalidConfiguration(
                "cannot subscribe to the trades of an empty symbol".to_string(),
            ));
        }
        *self.trade_symbol.write().map_err(poisoned)? = Some(symbol.to_string());
        Ok(())
    }

    async fn next_orderbook_update(&self) -> MMResult<OrderBookSnapshot> {
        // Return a simulated order book
        Ok(self.generate_orderbook("BTC-USD", self.config.default_depth))
    }

    /// A simulated trade of the subscribed symbol.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] without a trade subscription, since the
    /// trade would have no instrument to name, and [`MMError::LockPoisoned`] if
    /// the lock is poisoned.
    async fn next_trade(&self) -> MMResult<Fill> {
        let symbol = self
            .trade_symbol
            .read()
            .map_err(poisoned)?
            .clone()
            .ok_or_else(|| {
                MMError::InvalidConfiguration(
                    "no trade subscription: subscribe_trades first".to_string(),
                )
            })?;
        Ok(Fill {
            symbol,
            order_id: OrderId::new("mock-trade"),
            trade_id: format!("trade-{}", self.current_time()),
            price: self.config.base_price,
            quantity: dec!(0.1),
            side: Action::Buy,
            timestamp: self.current_time(),
            fee: dec!(0.001),
            fee_currency: "USD".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[tokio::test]
    async fn test_next_trade_names_the_subscribed_symbol() {
        let connector = MockExchangeConnector::with_defaults();
        connector.subscribe_trades("ETH-USD").await.unwrap();
        assert_eq!(connector.next_trade().await.unwrap().symbol, "ETH-USD");
        // A later subscription replaces it.
        connector.subscribe_trades("SOL-USD").await.unwrap();
        assert_eq!(connector.next_trade().await.unwrap().symbol, "SOL-USD");
    }

    #[tokio::test]
    async fn test_next_trade_without_a_subscription_names_nothing() {
        let connector = MockExchangeConnector::with_defaults();
        assert!(matches!(
            connector.next_trade().await,
            Err(MMError::InvalidConfiguration(_))
        ));
    }

    #[tokio::test]
    async fn test_subscribe_trades_refuses_an_empty_symbol() {
        let connector = MockExchangeConnector::with_defaults();
        assert!(matches!(
            connector.subscribe_trades("").await,
            Err(MMError::InvalidConfiguration(_))
        ));
        // And the refusal subscribed nothing.
        assert!(connector.next_trade().await.is_err());
    }

    #[tokio::test]
    async fn test_submit_limit_order() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let response = connector.submit_order(request).await.unwrap();

        assert!(response.status.is_open());
        assert!(!response.order_id.as_str().is_empty());
    }

    #[tokio::test]
    async fn test_submit_market_order() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::market_buy("BTC-USD", dec!(0.1));
        let response = connector.submit_order(request).await.unwrap();

        assert!(response.status.is_terminal());
        match response.status {
            OrderStatus::Filled { filled_qty, .. } => {
                assert_eq!(filled_qty, dec!(0.1));
            }
            _ => panic!("expected filled status"),
        }
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let response = connector.submit_order(request).await.unwrap();

        let cancelled = connector.cancel_order(&response.order_id).await.unwrap();
        assert!(cancelled.status.is_terminal());
        match cancelled.status {
            OrderStatus::Cancelled { filled_qty } => {
                assert_eq!(filled_qty, Decimal::ZERO);
            }
            _ => panic!("expected cancelled status"),
        }
    }

    #[tokio::test]
    async fn test_cancel_nonexistent_order() {
        let connector = MockExchangeConnector::with_defaults();

        let result = connector.cancel_order(&OrderId::new("nonexistent")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_order_status() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let response = connector.submit_order(request).await.unwrap();

        let status = connector
            .get_order_status(&response.order_id)
            .await
            .unwrap();
        assert!(status.status.is_open());
    }

    #[tokio::test]
    async fn test_get_open_orders() {
        let connector = MockExchangeConnector::with_defaults();

        // Submit multiple orders
        connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1)))
            .await
            .unwrap();
        connector
            .submit_order(OrderRequest::limit_sell(
                "BTC-USD",
                dec!(51000.0),
                dec!(0.1),
            ))
            .await
            .unwrap();
        connector
            .submit_order(OrderRequest::limit_buy("ETH-USD", dec!(3000.0), dec!(1.0)))
            .await
            .unwrap();

        let btc_orders = connector.get_open_orders("BTC-USD").await.unwrap();
        assert_eq!(btc_orders.len(), 2);

        let eth_orders = connector.get_open_orders("ETH-USD").await.unwrap();
        assert_eq!(eth_orders.len(), 1);
    }

    #[tokio::test]
    async fn test_cancel_all_orders() {
        let connector = MockExchangeConnector::with_defaults();

        // Submit multiple orders
        connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1)))
            .await
            .unwrap();
        connector
            .submit_order(OrderRequest::limit_sell(
                "BTC-USD",
                dec!(51000.0),
                dec!(0.1),
            ))
            .await
            .unwrap();

        let cancelled = connector.cancel_all_orders("BTC-USD").await.unwrap();
        assert_eq!(cancelled.len(), 2);

        let open = connector.get_open_orders("BTC-USD").await.unwrap();
        assert!(open.is_empty());
    }

    #[tokio::test]
    async fn test_cancel_all_orders_reports_failed_cancels() {
        let mut connector = MockExchangeConnector::with_defaults();
        connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1)))
            .await
            .unwrap();
        connector
            .submit_order(OrderRequest::limit_sell(
                "BTC-USD",
                dec!(51000.0),
                dec!(0.1),
            ))
            .await
            .unwrap();
        // Every later exchange call fails.
        connector.config.failure_rate = 1.0;

        let error = connector.cancel_all_orders("BTC-USD").await.unwrap_err();
        assert!(matches!(error, MMError::InvalidMarketState(_)), "{error}");
        // Nothing was cancelled, so both orders are still open.
        assert_eq!(connector.get_open_orders("BTC-USD").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_cancel_all_orders_poisoned_lock() {
        let connector = MockExchangeConnector::with_defaults();
        connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1)))
            .await
            .unwrap();
        std::thread::scope(|scope| {
            let poisoner = scope.spawn(|| {
                let _guard = connector.orders.write().unwrap();
                panic!("poison the order table");
            });
            assert!(poisoner.join().is_err());
        });

        let error = connector.cancel_all_orders("BTC-USD").await.unwrap_err();
        assert!(matches!(error, MMError::LockPoisoned(_)), "{error}");
    }

    #[tokio::test]
    async fn test_modify_order() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let response = connector.submit_order(request).await.unwrap();

        let modified = connector
            .modify_order(&response.order_id, Some(dec!(49000.0)), None)
            .await
            .unwrap();

        assert!(modified.status.is_open());
        assert_ne!(modified.order_id, response.order_id); // New order ID
    }

    #[tokio::test]
    async fn test_get_orderbook() {
        let connector = MockExchangeConnector::with_defaults();

        let orderbook = connector.get_orderbook("BTC-USD", 5).await.unwrap();

        assert_eq!(orderbook.symbol, "BTC-USD");
        assert_eq!(orderbook.bids.len(), 5);
        assert_eq!(orderbook.asks.len(), 5);
        assert!(orderbook.best_bid().is_some());
        assert!(orderbook.best_ask().is_some());
        assert!(orderbook.spread().unwrap() > Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_get_balance() {
        let connector = MockExchangeConnector::with_defaults();

        let usd_balance = connector.get_balance("USD").await.unwrap();
        assert_eq!(usd_balance, Decimal::from(100_000));

        let btc_balance = connector.get_balance("BTC").await.unwrap();
        assert_eq!(btc_balance, Decimal::from(10));

        let unknown = connector.get_balance("UNKNOWN").await.unwrap();
        assert_eq!(unknown, Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_set_balance() {
        let connector = MockExchangeConnector::with_defaults();

        connector.set_balance("ETH", dec!(100.0)).unwrap();
        let balance = connector.get_balance("ETH").await.unwrap();
        assert_eq!(balance, dec!(100.0));
    }

    #[tokio::test]
    async fn test_time_management() {
        let connector = MockExchangeConnector::with_defaults();

        let initial_time = connector.current_time();
        connector.advance_time(1000);
        assert_eq!(connector.current_time(), initial_time + 1000);

        connector.set_time(5000);
        assert_eq!(connector.current_time(), 5000);
    }

    #[tokio::test]
    async fn test_client_order_id() {
        let connector = MockExchangeConnector::with_defaults();

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1))
            .with_client_order_id("my-order-1");

        let response = connector.submit_order(request).await.unwrap();
        assert_eq!(response.client_order_id, Some("my-order-1".to_string()));
    }

    #[tokio::test]
    async fn test_market_data_stream() {
        let connector = MockExchangeConnector::with_defaults();

        // Subscribe (always succeeds in mock)
        connector.subscribe_orderbook("BTC-USD").await.unwrap();
        connector.subscribe_trades("BTC-USD").await.unwrap();

        // Get updates
        let orderbook = connector.next_orderbook_update().await.unwrap();
        assert!(!orderbook.bids.is_empty());

        let trade = connector.next_trade().await.unwrap();
        assert!(trade.quantity > Decimal::ZERO);
    }

    #[test]
    fn test_mock_config_builder() {
        let config = MockConfig::new()
            .with_latency_ms(50)
            .with_failure_rate(0.1)
            .with_base_price(dec!(60000.0))
            .with_spread(dec!(0.002))
            .with_balance("ETH", dec!(50.0));

        assert_eq!(config.latency_ms, 50);
        assert_eq!(config.failure_rate, 0.1);
        assert_eq!(config.base_price, dec!(60000.0));
        assert_eq!(config.spread, dec!(0.002));
        assert_eq!(config.initial_balances.get("ETH"), Some(&dec!(50.0)));
    }

    #[tokio::test]
    async fn test_open_order_count() {
        let connector = MockExchangeConnector::with_defaults();

        assert_eq!(connector.open_order_count().unwrap(), 0);

        connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1)))
            .await
            .unwrap();

        assert_eq!(connector.open_order_count().unwrap(), 1);

        connector
            .submit_order(OrderRequest::market_buy("BTC-USD", dec!(0.1)))
            .await
            .unwrap();

        // Market order is filled immediately, so still 1 open
        assert_eq!(connector.open_order_count().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_submit_order_unsupported_action_returns_err_without_effects() {
        let connector = MockExchangeConnector::with_defaults();
        let fresh = MockExchangeConnector::with_defaults();

        let mut request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        request.side = Action::Other;
        assert_eq!(
            connector.submit_order(request).await,
            Err(MMError::UnsupportedAction {
                action: Action::Other
            })
        );
        assert!(
            connector
                .get_open_orders("BTC-USD")
                .await
                .unwrap()
                .is_empty()
        );

        // No order id was consumed.
        let valid = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let first = connector.submit_order(valid.clone()).await.unwrap();
        let expected = fresh.submit_order(valid).await.unwrap();
        assert_eq!(first.order_id, expected.order_id);
    }
}
