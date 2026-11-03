//! OrderBook-rs connector implementation.
//!
//! This module provides an `ExchangeConnector` implementation that connects
//! to the `orderbook-rs` library for order management and execution.
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::execution::{OrderBookConnector, OrderRequest, Action, OrderType};
//! use market_maker_rs::dec;
//!
//! // Create connector with an order book
//! let connector = OrderBookConnector::new("BTC-USD");
//!
//! // Submit a limit order
//! let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
//! let response = connector.submit_order(request).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use orderbook_rs::{
    DefaultOrderBook, OrderBook, OrderBookSnapshot as OBSnapshot, OrderId as OBOrderId,
    Side as OBSide, TimeInForce as OBTimeInForce,
};

use crate::Decimal;
use crate::execution::connector::{
    BookLevel, ExchangeConnector, OrderBookSnapshot, OrderId, OrderRequest, OrderResponse,
    OrderStatus, OrderType, TimeInForce,
};
use crate::types::action::{Action, Dir};
use crate::types::decimal::{decimal_to_u64_saturating, decimal_to_u128_saturating};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the OrderBook connector.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookConnectorConfig {
    /// Price precision (decimal places).
    pub price_precision: u32,
    /// Quantity precision (decimal places).
    pub quantity_precision: u32,
    /// Fee rate as a decimal (e.g., 0.001 for 0.1%).
    pub fee_rate: Decimal,
    /// Fee currency.
    pub fee_currency: String,
}

/// Maps a poisoned lock to [`MMError::LockPoisoned`].
fn poisoned<T>(_: std::sync::PoisonError<T>) -> MMError {
    MMError::LockPoisoned("orderbook connector".to_string())
}

impl Default for OrderBookConnectorConfig {
    fn default() -> Self {
        Self {
            price_precision: 2,
            quantity_precision: 8,
            fee_rate: Decimal::ZERO,
            fee_currency: "USD".to_string(),
        }
    }
}

/// Exchange connector implementation for OrderBook-rs.
///
/// This connector wraps an `orderbook-rs` `OrderBook` and implements
/// the `ExchangeConnector` trait for integration with market making strategies.
pub struct OrderBookConnector {
    /// The underlying order book.
    order_book: Arc<DefaultOrderBook>,
    /// Configuration.
    config: OrderBookConnectorConfig,
    /// Order ID counter for generating unique IDs.
    order_id_counter: AtomicU64,
    /// Mapping from our OrderId to orderbook-rs OrderId.
    order_mapping: std::sync::RwLock<HashMap<String, OBOrderId>>,
    /// Simulated balances for testing.
    balances: std::sync::RwLock<HashMap<String, Decimal>>,
}

impl OrderBookConnector {
    /// Creates a new OrderBook connector for the given symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol (e.g., "BTC-USD")
    ///
    /// # Returns
    ///
    /// A new `OrderBookConnector` instance.
    #[must_use]
    pub fn new(symbol: &str) -> Self {
        Self {
            order_book: Arc::new(OrderBook::new(symbol)),
            config: OrderBookConnectorConfig::default(),
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Creates a new OrderBook connector with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    /// * `config` - The connector configuration
    ///
    /// # Returns
    ///
    /// A new `OrderBookConnector` instance.
    #[must_use]
    pub fn with_config(symbol: &str, config: OrderBookConnectorConfig) -> Self {
        Self {
            order_book: Arc::new(OrderBook::new(symbol)),
            config,
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Creates a new OrderBook connector wrapping an existing order book.
    ///
    /// # Arguments
    ///
    /// * `order_book` - The existing order book to wrap
    /// * `config` - The connector configuration
    ///
    /// # Returns
    ///
    /// A new `OrderBookConnector` instance.
    #[must_use]
    pub fn from_orderbook(
        order_book: Arc<DefaultOrderBook>,
        config: OrderBookConnectorConfig,
    ) -> Self {
        Self {
            order_book,
            config,
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Returns a reference to the underlying order book.
    #[must_use]
    pub fn order_book(&self) -> &DefaultOrderBook {
        &self.order_book
    }

    /// Returns the symbol of this connector.
    #[must_use]
    pub fn symbol(&self) -> &str {
        self.order_book.symbol()
    }

    /// Sets a simulated balance for testing.
    ///
    /// # Arguments
    ///
    /// * `asset` - The asset symbol
    /// * `balance` - The balance amount
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn set_balance(&self, asset: &str, balance: Decimal) -> MMResult<()> {
        let mut balances = self.balances.write().map_err(poisoned)?;
        balances.insert(asset.to_string(), balance);
        Ok(())
    }

    /// Generates a new unique order ID.
    fn next_order_id(&self) -> (OrderId, OBOrderId) {
        let id = self.order_id_counter.fetch_add(1, Ordering::SeqCst);
        let order_id = OrderId::new(id.to_string());
        let ob_order_id = OBOrderId::sequential(id);
        (order_id, ob_order_id)
    }

    /// Generates a new unique orderbook-rs order ID from the shared counter.
    fn next_ob_order_id(&self) -> OBOrderId {
        OBOrderId::sequential(self.order_id_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Converts a supported direction to orderbook-rs Side.
    fn convert_side(dir: Dir) -> OBSide {
        match dir {
            Dir::Buy => OBSide::Buy,
            Dir::Sell => OBSide::Sell,
        }
    }

    /// Converts orderbook-rs Side to our Action.
    #[allow(dead_code)]
    fn convert_side_back(side: OBSide) -> Action {
        match side {
            OBSide::Buy => Action::Buy,
            OBSide::Sell => Action::Sell,
        }
    }

    /// Converts our TimeInForce to orderbook-rs TimeInForce.
    fn convert_tif(tif: TimeInForce) -> OBTimeInForce {
        match tif {
            TimeInForce::GoodTilCancel => OBTimeInForce::Gtc,
            TimeInForce::ImmediateOrCancel => OBTimeInForce::Ioc,
            TimeInForce::FillOrKill => OBTimeInForce::Fok,
            TimeInForce::GoodTilTime(ts) => OBTimeInForce::Gtd(ts),
        }
    }

    /// Converts a Decimal price to the raw `u128` used by orderbook-rs.
    fn price_to_u128(&self, price: Decimal) -> u128 {
        let multiplier = Decimal::from(10u64.pow(self.config.price_precision));
        let scaled = price * multiplier;
        decimal_to_u128_saturating(scaled)
    }

    /// Converts a raw `u128` price from orderbook-rs to Decimal.
    fn u128_to_price(&self, price: u128) -> Decimal {
        let divisor = Decimal::from(10u64.pow(self.config.price_precision));
        Decimal::from(price) / divisor
    }

    /// Converts a Decimal quantity to u64 for orderbook-rs.
    fn quantity_to_u64(&self, quantity: Decimal) -> u64 {
        let multiplier = Decimal::from(10u64.pow(self.config.quantity_precision));
        let scaled = quantity * multiplier;
        decimal_to_u64_saturating(scaled)
    }

    /// Converts a u64 quantity from orderbook-rs to Decimal.
    fn u64_to_quantity(&self, quantity: u64) -> Decimal {
        let divisor = Decimal::from(10u64.pow(self.config.quantity_precision));
        Decimal::from(quantity) / divisor
    }

    /// Gets the current timestamp in milliseconds.
    fn current_timestamp() -> u64 {
        orderbook_rs::current_time_millis()
    }
}

#[async_trait]
impl ExchangeConnector for OrderBookConnector {
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderResponse> {
        // Validated before an order id is consumed or the book is touched.
        let dir = Dir::try_from(request.side)?;
        let (order_id, ob_order_id) = self.next_order_id();
        let ob_side = Self::convert_side(dir);
        let ob_tif = Self::convert_tif(request.time_in_force);
        let quantity = self.quantity_to_u64(request.quantity);

        let result = match request.order_type {
            OrderType::Market => {
                // Submit market order
                self.order_book
                    .submit_market_order(ob_order_id, quantity, ob_side)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

                // Market orders are immediately filled or rejected
                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Filled {
                        filled_qty: request.quantity,
                        avg_price: request.price.unwrap_or(Decimal::ZERO),
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
            OrderType::Limit => {
                let price = request.price.ok_or_else(|| {
                    MMError::InvalidConfiguration("Limit order requires price".to_string())
                })?;
                let ob_price = self.price_to_u128(price);

                self.order_book
                    .add_limit_order(ob_order_id, ob_price, quantity, ob_side, ob_tif, None)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

                // Store mapping
                {
                    let mut mapping = self.order_mapping.write().map_err(poisoned)?;
                    mapping.insert(order_id.as_str().to_string(), ob_order_id);
                }

                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Open {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
            OrderType::PostOnly => {
                let price = request.price.ok_or_else(|| {
                    MMError::InvalidConfiguration("PostOnly order requires price".to_string())
                })?;
                let ob_price = self.price_to_u128(price);

                self.order_book
                    .add_post_only_order(ob_order_id, ob_price, quantity, ob_side, ob_tif, None)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

                // Store mapping
                {
                    let mut mapping = self.order_mapping.write().map_err(poisoned)?;
                    mapping.insert(order_id.as_str().to_string(), ob_order_id);
                }

                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Open {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
        };

        Ok(result)
    }

    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        let ob_order_id = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping.get(order_id.as_str()).copied().ok_or_else(|| {
                MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
            })?
        };

        self.order_book
            .cancel_order(ob_order_id)
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Remove from mapping
        {
            let mut mapping = self.order_mapping.write().map_err(poisoned)?;
            mapping.remove(order_id.as_str());
        }

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Cancelled {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn modify_order(
        &self,
        order_id: &OrderId,
        new_price: Option<Decimal>,
        new_quantity: Option<Decimal>,
    ) -> MMResult<OrderResponse> {
        let ob_order_id = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping.get(order_id.as_str()).copied().ok_or_else(|| {
                MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
            })?
        };

        // For order modification, we cancel and re-add the order
        // This is a common pattern when the underlying system doesn't support atomic modifications

        // Get the current order to extract its details
        let order = self.order_book.get_order(ob_order_id).ok_or_else(|| {
            MMError::InvalidPositionUpdate(format!("Order not found in book: {}", order_id))
        })?;

        // Extract current order details
        let (current_price, current_quantity, side, tif) = match order.as_ref() {
            orderbook_rs::OrderType::Standard {
                price,
                quantity,
                side,
                time_in_force,
                ..
            } => (price.as_u128(), quantity.as_u64(), *side, *time_in_force),
            orderbook_rs::OrderType::PostOnly {
                price,
                quantity,
                side,
                time_in_force,
                ..
            } => (price.as_u128(), quantity.as_u64(), *side, *time_in_force),
            _ => {
                return Err(MMError::InvalidPositionUpdate(
                    "Unsupported order type for modification".to_string(),
                ));
            }
        };

        // Cancel the existing order
        self.order_book
            .cancel_order(ob_order_id)
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Create new order with updated values
        let final_price = new_price
            .map(|p| self.price_to_u128(p))
            .unwrap_or(current_price);
        let final_quantity = new_quantity
            .map(|q| self.quantity_to_u64(q))
            .unwrap_or(current_quantity);

        // Generate new order ID for the replacement order
        let new_ob_order_id = self.next_ob_order_id();

        self.order_book
            .add_limit_order(
                new_ob_order_id,
                final_price,
                final_quantity,
                side,
                tif,
                None,
            )
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Update mapping with new order ID
        {
            let mut mapping = self.order_mapping.write().map_err(poisoned)?;
            mapping.insert(order_id.as_str().to_string(), new_ob_order_id);
        }

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn get_order_status(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        let ob_order_id = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping.get(order_id.as_str()).copied().ok_or_else(|| {
                MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
            })?
        };

        let order = self.order_book.get_order(ob_order_id).ok_or_else(|| {
            MMError::InvalidPositionUpdate(format!("Order not found in book: {}", order_id))
        })?;

        // Order exists in the book, so it's open
        let _ = order; // Acknowledge we have the order

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn get_open_orders(&self, _symbol: &str) -> MMResult<Vec<OrderResponse>> {
        let mapping = self.order_mapping.read().map_err(poisoned)?;
        let mut orders = Vec::new();

        for id_str in mapping.keys() {
            let id_string = id_str.to_string();
            orders.push(OrderResponse {
                order_id: OrderId::new(id_string),
                client_order_id: None,
                status: OrderStatus::Open {
                    filled_qty: Decimal::ZERO,
                },
                timestamp: Self::current_timestamp(),
            });
        }

        Ok(orders)
    }

    async fn cancel_all_orders(&self, _symbol: &str) -> MMResult<Vec<OrderResponse>> {
        let order_ids: Vec<(String, OBOrderId)> = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping
                .iter()
                .map(|(k, v): (&String, &OBOrderId)| (k.clone(), *v))
                .collect()
        };

        let mut responses = Vec::new();

        for (id_str, ob_id) in order_ids {
            if self.order_book.cancel_order(ob_id).is_ok() {
                responses.push(OrderResponse {
                    order_id: OrderId::new(id_str),
                    client_order_id: None,
                    status: OrderStatus::Cancelled {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                });
            }
        }

        // Clear mapping
        {
            let mut mapping = self.order_mapping.write().map_err(poisoned)?;
            mapping.clear();
        }

        Ok(responses)
    }

    async fn get_orderbook(&self, _symbol: &str, depth: usize) -> MMResult<OrderBookSnapshot> {
        let snapshot: OBSnapshot = self
            .order_book
            .create_snapshot(depth)
            .map_err(|e| MMError::InvalidMarketState(e.to_string()))?;

        let bids: Vec<BookLevel> = snapshot
            .bids
            .iter()
            .map(|level| {
                BookLevel::new(
                    self.u128_to_price(level.price().as_u128()),
                    self.u64_to_quantity(level.visible_quantity().as_u64()),
                )
            })
            .collect();

        let asks: Vec<BookLevel> = snapshot
            .asks
            .iter()
            .map(|level| {
                BookLevel::new(
                    self.u128_to_price(level.price().as_u128()),
                    self.u64_to_quantity(level.visible_quantity().as_u64()),
                )
            })
            .collect();

        Ok(OrderBookSnapshot {
            symbol: self.order_book.symbol().to_string(),
            bids,
            asks,
            timestamp: snapshot.timestamp,
        })
    }

    async fn get_balance(&self, asset: &str) -> MMResult<Decimal> {
        let balances = self.balances.read().map_err(poisoned)?;
        Ok(balances.get(asset).copied().unwrap_or(Decimal::ZERO))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[tokio::test]
    async fn test_connector_creation() {
        let connector = OrderBookConnector::new("BTC-USD");
        assert_eq!(connector.symbol(), "BTC-USD");
    }

    #[tokio::test]
    async fn test_submit_limit_order() {
        let connector = OrderBookConnector::new("BTC-USD");

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();

        assert!(response.status.is_open());
    }

    #[tokio::test]
    async fn test_submit_and_cancel_order() {
        let connector = OrderBookConnector::new("BTC-USD");

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();
        let order_id = response.order_id;

        let cancel_response = connector.cancel_order(&order_id).await.unwrap();
        assert!(cancel_response.status.is_terminal());
    }

    #[tokio::test]
    async fn test_get_orderbook() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Add some orders
        let buy_request = OrderRequest::limit_buy("BTC-USD", dec!(49000.0), dec!(1.0));
        connector.submit_order(buy_request).await.unwrap();

        let sell_request = OrderRequest::limit_sell("BTC-USD", dec!(51000.0), dec!(1.0));
        connector.submit_order(sell_request).await.unwrap();

        let snapshot = connector.get_orderbook("BTC-USD", 10).await.unwrap();

        assert_eq!(snapshot.symbol, "BTC-USD");
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[tokio::test]
    async fn test_balance() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Initially zero
        let balance = connector.get_balance("BTC").await.unwrap();
        assert_eq!(balance, Decimal::ZERO);

        // Set balance
        connector.set_balance("BTC", dec!(10.0)).unwrap();
        let balance = connector.get_balance("BTC").await.unwrap();
        assert_eq!(balance, dec!(10.0));
    }

    #[tokio::test]
    async fn test_cancel_all_orders() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Add multiple orders
        for i in 0..5 {
            let price = dec!(50000.0) - Decimal::from(i * 100);
            let request = OrderRequest::limit_buy("BTC-USD", price, dec!(1.0));
            connector.submit_order(request).await.unwrap();
        }

        let open_orders = connector.get_open_orders("BTC-USD").await.unwrap();
        assert_eq!(open_orders.len(), 5);

        let cancelled = connector.cancel_all_orders("BTC-USD").await.unwrap();
        assert_eq!(cancelled.len(), 5);

        let open_orders = connector.get_open_orders("BTC-USD").await.unwrap();
        assert!(open_orders.is_empty());
    }

    #[tokio::test]
    async fn test_submit_order_unsupported_action_returns_err_without_effects() {
        let connector = OrderBookConnector::new("BTC-USD");
        let fresh = OrderBookConnector::new("BTC-USD");

        for order_type in [OrderType::Limit, OrderType::PostOnly, OrderType::Market] {
            let mut request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
            request.order_type = order_type;
            request.side = Action::Other;
            assert_eq!(
                connector.submit_order(request).await,
                Err(MMError::UnsupportedAction {
                    action: Action::Other
                })
            );
        }
        assert!(
            connector
                .get_open_orders("BTC-USD")
                .await
                .unwrap()
                .is_empty()
        );
        let book = connector.get_orderbook("BTC-USD", 10).await.unwrap();
        assert!(book.bids.is_empty() && book.asks.is_empty());

        // No order id was consumed.
        let valid = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let first = connector.submit_order(valid.clone()).await.unwrap();
        let expected = fresh.submit_order(valid).await.unwrap();
        assert_eq!(first.order_id, expected.order_id);
    }
}
