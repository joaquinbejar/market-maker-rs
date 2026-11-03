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
use crate::types::error::{MMError, MMResult};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

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

/// Largest supported price or quantity precision: `10^19` is the largest
/// power of ten that fits in a `u64`.
pub const MAX_PRECISION: u32 = 19;

/// Returns `10^precision`.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] if `precision` is above
/// [`MAX_PRECISION`].
fn scale_factor(precision: u32) -> MMResult<Decimal> {
    10u64
        .checked_pow(precision)
        .map(Decimal::from)
        .ok_or_else(|| {
            MMError::InvalidConfiguration(format!(
                "precision must be at most {MAX_PRECISION}, got {precision}"
            ))
        })
}

impl OrderBookConnectorConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `price_precision` or
    /// `quantity_precision` is above [`MAX_PRECISION`].
    pub fn validate(&self) -> MMResult<()> {
        for (name, precision) in [
            ("price_precision", self.price_precision),
            ("quantity_precision", self.quantity_precision),
        ] {
            if precision > MAX_PRECISION {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be at most {MAX_PRECISION}, got {precision}"
                )));
            }
        }
        Ok(())
    }
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
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a precision is above
    /// [`MAX_PRECISION`].
    pub fn with_config(symbol: &str, config: OrderBookConnectorConfig) -> MMResult<Self> {
        Self::from_orderbook(Arc::new(OrderBook::new(symbol)), config)
    }

    /// Creates a new OrderBook connector wrapping an existing order book.
    ///
    /// # Arguments
    ///
    /// * `order_book` - The existing order book to wrap
    /// * `config` - The connector configuration
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a precision is above
    /// [`MAX_PRECISION`].
    pub fn from_orderbook(
        order_book: Arc<DefaultOrderBook>,
        config: OrderBookConnectorConfig,
    ) -> MMResult<Self> {
        config.validate()?;
        Ok(Self {
            order_book,
            config,
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        })
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
    ///
    /// Rejects a price that is not positive, that rounds to zero ticks, or
    /// that does not fit: a zero raw price would make a sell marketable.
    fn price_to_u128(&self, price: Decimal) -> MMResult<u128> {
        let scaled = price
            .checked_mul(scale_factor(self.config.price_precision)?)
            .map(|v| v.trunc())
            .ok_or_else(|| MMError::NumericalError(format!("price {price} overflows")))?;
        match scaled.to_u128() {
            Some(raw) if raw > 0 => Ok(raw),
            Some(_) | None => Err(MMError::NumericalError(format!(
                "price {price} is not a positive raw price at precision {}",
                self.config.price_precision
            ))),
        }
    }

    /// Converts a raw `u128` price from orderbook-rs to Decimal.
    fn u128_to_price(&self, price: u128) -> MMResult<Decimal> {
        let divisor = scale_factor(self.config.price_precision)?;
        Decimal::from_u128(price)
            .and_then(|p| p.checked_div(divisor))
            .ok_or_else(|| MMError::NumericalError(format!("raw price {price} out of range")))
    }

    /// Converts a Decimal quantity to u64 for orderbook-rs.
    ///
    /// Rejects a quantity that is negative or does not fit in a `u64`.
    fn quantity_to_u64(&self, quantity: Decimal) -> MMResult<u64> {
        let scaled = quantity
            .checked_mul(scale_factor(self.config.quantity_precision)?)
            .map(|v| v.trunc())
            .ok_or_else(|| MMError::NumericalError(format!("quantity {quantity} overflows")))?;
        if scaled.is_sign_negative() && !scaled.is_zero() {
            return Err(MMError::NumericalError(format!(
                "quantity {quantity} is negative"
            )));
        }
        scaled.to_u64().ok_or_else(|| {
            MMError::NumericalError(format!("quantity {quantity} does not fit in u64"))
        })
    }

    /// Converts a u64 quantity from orderbook-rs to Decimal.
    fn u64_to_quantity(&self, quantity: u64) -> MMResult<Decimal> {
        let divisor = scale_factor(self.config.quantity_precision)?;
        Decimal::from(quantity)
            .checked_div(divisor)
            .ok_or_else(|| MMError::NumericalError(format!("raw quantity {quantity} out of range")))
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
        let quantity = self.quantity_to_u64(request.quantity)?;

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
                let ob_price = self.price_to_u128(price)?;

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
                let ob_price = self.price_to_u128(price)?;

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

        // Convert the new values before touching the book, so a bad value
        // leaves the existing order resting.
        let final_price = match new_price {
            Some(p) => self.price_to_u128(p)?,
            None => current_price,
        };
        let final_quantity = match new_quantity {
            Some(q) => self.quantity_to_u64(q)?,
            None => current_quantity,
        };

        // Cancel the existing order
        self.order_book
            .cancel_order(ob_order_id)
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

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

        let bids = snapshot
            .bids
            .iter()
            .map(|level| {
                Ok(BookLevel::new(
                    self.u128_to_price(level.price().as_u128())?,
                    self.u64_to_quantity(level.visible_quantity().as_u64())?,
                ))
            })
            .collect::<MMResult<Vec<BookLevel>>>()?;
        let asks = snapshot
            .asks
            .iter()
            .map(|level| {
                Ok(BookLevel::new(
                    self.u128_to_price(level.price().as_u128())?,
                    self.u64_to_quantity(level.visible_quantity().as_u64())?,
                ))
            })
            .collect::<MMResult<Vec<BookLevel>>>()?;

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

    fn config(price_precision: u32, quantity_precision: u32) -> OrderBookConnectorConfig {
        OrderBookConnectorConfig {
            price_precision,
            quantity_precision,
            ..OrderBookConnectorConfig::default()
        }
    }

    #[test]
    fn test_with_config_precision_above_max_returns_err() {
        for cfg in [config(20, 8), config(2, 20)] {
            assert!(matches!(
                OrderBookConnector::with_config("BTC-USD", cfg),
                Err(MMError::InvalidConfiguration(_))
            ));
        }
        assert!(OrderBookConnector::with_config("BTC-USD", config(19, 19)).is_ok());
    }

    #[test]
    fn test_from_orderbook_precision_above_max_returns_err() {
        let book = Arc::new(OrderBook::new("BTC-USD"));
        assert!(matches!(
            OrderBookConnector::from_orderbook(book, config(u32::MAX, 8)),
            Err(MMError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn test_raw_to_decimal_beyond_range_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        assert!(matches!(
            connector.u128_to_price(u128::MAX),
            Err(MMError::NumericalError(_))
        ));
        assert_eq!(connector.u128_to_price(12345).unwrap(), dec!(123.45));
        assert_eq!(connector.u64_to_quantity(150_000_000).unwrap(), dec!(1.5));
    }

    #[test]
    fn test_price_to_u128_non_positive_or_huge_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        for price in [dec!(0), dec!(-1), dec!(0.001), Decimal::MAX] {
            assert!(matches!(
                connector.price_to_u128(price),
                Err(MMError::NumericalError(_))
            ));
        }
        assert_eq!(connector.price_to_u128(dec!(1.23)).unwrap(), 123);
    }

    #[test]
    fn test_quantity_to_u64_negative_or_huge_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        for quantity in [dec!(-1), dec!(1_000_000_000_000)] {
            assert!(matches!(
                connector.quantity_to_u64(quantity),
                Err(MMError::NumericalError(_))
            ));
        }
        assert_eq!(connector.quantity_to_u64(dec!(0)).unwrap(), 0);
    }

    #[tokio::test]
    async fn test_submit_order_sell_at_zero_price_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_sell("BTC-USD", dec!(0), dec!(1.0));
        assert!(matches!(
            connector.submit_order(request).await,
            Err(MMError::NumericalError(_))
        ));
    }

    #[tokio::test]
    async fn test_modify_order_invalid_price_keeps_order() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();
        let result = connector
            .modify_order(&response.order_id, Some(dec!(-5)), None)
            .await;
        assert!(matches!(result, Err(MMError::NumericalError(_))));
        let book = connector.get_orderbook("BTC-USD", 10).await.unwrap();
        assert_eq!(book.bids.len(), 1);
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
