//! Order Management System (OMS) for tracking order lifecycle.
//!
//! This module provides order tracking and lifecycle management including
//! order states, partial fills, and reconciliation.
//!
//! # Overview
//!
//! The `OrderManager` maintains the state of all orders and provides methods
//! for order lifecycle management:
//!
//! - Order registration and tracking
//! - Status updates from exchange responses
//! - Fill recording and average price calculation
//! - Timeout detection and cleanup
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     OrderManager, OrderManagerConfig, OrderRequest, Side
//! };
//! use market_maker_rs::dec;
//!
//! let config = OrderManagerConfig::default();
//! let mut manager = OrderManager::new(config);
//!
//! // Register an order before submission
//! let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
//! manager.register_order(&request, "client-1".to_string(), 1000).unwrap();
//!
//! // Check open orders
//! let open = manager.get_open_orders();
//! assert_eq!(open.len(), 1);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

use super::connector::{Fill, OrderId, OrderRequest, OrderResponse, OrderStatus, OrderType, Side};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Internal managed order state.
///
/// Tracks the complete lifecycle of an order including all fills.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{ManagedOrder, OrderId, Side, OrderType, OrderStatus};
/// use market_maker_rs::dec;
///
/// let order = ManagedOrder::new(
///     OrderId::new("12345"),
///     "client-1".to_string(),
///     "BTC-USD".to_string(),
///     Side::Buy,
///     OrderType::Limit,
///     dec!(50000.0),
///     dec!(0.1),
///     1000,
/// );
///
/// assert!(order.is_pending());
/// assert_eq!(order.remaining_quantity, dec!(0.1));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ManagedOrder {
    /// Exchange-assigned order ID.
    pub order_id: OrderId,
    /// Client-assigned order ID for tracking.
    pub client_order_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Order side.
    pub side: Side,
    /// Order type.
    pub order_type: OrderType,
    /// Original order price.
    pub original_price: Decimal,
    /// Original order quantity.
    pub original_quantity: Decimal,
    /// Total filled quantity.
    pub filled_quantity: Decimal,
    /// Remaining unfilled quantity.
    pub remaining_quantity: Decimal,
    /// Volume-weighted average fill price.
    pub average_fill_price: Decimal,
    /// Current order status.
    pub status: OrderStatus,
    /// Order creation timestamp in milliseconds.
    pub created_at: u64,
    /// Last update timestamp in milliseconds.
    pub updated_at: u64,
    /// List of all fills for this order.
    pub fills: Vec<Fill>,
}

impl ManagedOrder {
    /// Creates a new managed order.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        order_id: OrderId,
        client_order_id: String,
        symbol: String,
        side: Side,
        order_type: OrderType,
        price: Decimal,
        quantity: Decimal,
        timestamp: u64,
    ) -> Self {
        Self {
            order_id,
            client_order_id,
            symbol,
            side,
            order_type,
            original_price: price,
            original_quantity: quantity,
            filled_quantity: Decimal::ZERO,
            remaining_quantity: quantity,
            average_fill_price: Decimal::ZERO,
            status: OrderStatus::Pending,
            created_at: timestamp,
            updated_at: timestamp,
            fills: Vec::new(),
        }
    }

    /// Returns true if the order is pending submission.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self.status, OrderStatus::Pending)
    }

    /// Returns true if the order is open on the exchange.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.status.is_open()
    }

    /// Returns true if the order is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Returns the fill ratio (filled / original).
    #[must_use]
    pub fn fill_ratio(&self) -> Decimal {
        if self.original_quantity > Decimal::ZERO {
            self.filled_quantity / self.original_quantity
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the notional value of the original order.
    #[must_use]
    pub fn original_notional(&self) -> Decimal {
        self.original_price * self.original_quantity
    }

    /// Returns the notional value of filled quantity.
    #[must_use]
    pub fn filled_notional(&self) -> Decimal {
        self.average_fill_price * self.filled_quantity
    }

    /// Returns the age of the order in milliseconds.
    #[must_use]
    pub fn age_ms(&self, current_time: u64) -> u64 {
        current_time.saturating_sub(self.created_at)
    }

    /// Records a fill and updates quantities and average price.
    pub fn record_fill(&mut self, fill: &Fill, timestamp: u64) {
        let new_filled = self.filled_quantity + fill.quantity;

        // Calculate new VWAP
        if new_filled > Decimal::ZERO {
            let old_value = self.average_fill_price * self.filled_quantity;
            let new_value = fill.price * fill.quantity;
            self.average_fill_price = (old_value + new_value) / new_filled;
        }

        self.filled_quantity = new_filled;
        self.remaining_quantity = self.original_quantity - new_filled;
        self.updated_at = timestamp;
        self.fills.push(fill.clone());

        // Update status based on fill
        if self.remaining_quantity <= Decimal::ZERO {
            self.status = OrderStatus::Filled {
                filled_qty: self.filled_quantity,
                avg_price: self.average_fill_price,
            };
        } else {
            self.status = OrderStatus::PartiallyFilled {
                filled_qty: self.filled_quantity,
                remaining_qty: self.remaining_quantity,
            };
        }
    }

    /// Updates the order status.
    pub fn update_status(&mut self, status: OrderStatus, timestamp: u64) {
        self.status = status;
        self.updated_at = timestamp;
    }
}

/// Order manager configuration.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::OrderManagerConfig;
///
/// let config = OrderManagerConfig::default()
///     .with_order_timeout_ms(30_000)
///     .with_max_open_orders(100);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderManagerConfig {
    /// Order timeout in milliseconds (0 = no timeout).
    pub order_timeout_ms: u64,
    /// Maximum open orders per symbol (0 = unlimited).
    pub max_open_orders: usize,
    /// Enable duplicate order detection.
    pub detect_duplicates: bool,
}

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            order_timeout_ms: 0,
            max_open_orders: 0,
            detect_duplicates: true,
        }
    }
}

impl OrderManagerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the order timeout.
    #[must_use]
    pub fn with_order_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.order_timeout_ms = timeout_ms;
        self
    }

    /// Sets the maximum open orders per symbol.
    #[must_use]
    pub fn with_max_open_orders(mut self, max: usize) -> Self {
        self.max_open_orders = max;
        self
    }

    /// Sets duplicate detection.
    #[must_use]
    pub fn with_detect_duplicates(mut self, detect: bool) -> Self {
        self.detect_duplicates = detect;
        self
    }
}

/// Order manager statistics.
///
/// Provides summary statistics about order activity.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderManagerStats {
    /// Total number of orders tracked.
    pub total_orders: usize,
    /// Number of currently open orders.
    pub open_orders: usize,
    /// Number of filled orders.
    pub filled_orders: usize,
    /// Number of cancelled orders.
    pub cancelled_orders: usize,
    /// Total number of fills recorded.
    pub total_fills: usize,
    /// Number of pending orders.
    pub pending_orders: usize,
    /// Number of rejected orders.
    pub rejected_orders: usize,
}

/// Order manager for tracking order lifecycle.
///
/// Maintains the state of all orders and provides methods for
/// order lifecycle management.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{
///     OrderManager, OrderManagerConfig, OrderRequest, OrderResponse,
///     OrderId, OrderStatus
/// };
/// use market_maker_rs::dec;
///
/// let mut manager = OrderManager::new(OrderManagerConfig::default());
///
/// // Register order before submission
/// let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
/// manager.register_order(&request, "client-1".to_string(), 1000).unwrap();
///
/// // Update with exchange response
/// let response = OrderResponse::new(
///     OrderId::new("exchange-123"),
///     OrderStatus::Open { filled_qty: dec!(0.0) },
///     1001,
/// );
/// manager.update_order("client-1", &response, 1001).unwrap();
///
/// // Get order by client ID
/// let order = manager.get_order_by_client_id("client-1").unwrap();
/// assert!(order.is_open());
/// ```
#[derive(Debug)]
pub struct OrderManager {
    config: OrderManagerConfig,
    orders: HashMap<String, ManagedOrder>,
    orders_by_exchange_id: HashMap<String, String>,
    open_orders_by_symbol: HashMap<String, Vec<String>>,
}

impl OrderManager {
    /// Creates a new order manager.
    #[must_use]
    pub fn new(config: OrderManagerConfig) -> Self {
        Self {
            config,
            orders: HashMap::new(),
            orders_by_exchange_id: HashMap::new(),
            open_orders_by_symbol: HashMap::new(),
        }
    }

    /// Creates an order manager with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(OrderManagerConfig::default())
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &OrderManagerConfig {
        &self.config
    }

    /// Registers a new order before submission.
    ///
    /// # Arguments
    ///
    /// * `request` - The order request
    /// * `client_order_id` - Client-assigned order ID
    /// * `timestamp` - Current timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Duplicate client_order_id exists (if detection enabled)
    /// - Maximum open orders exceeded for the symbol
    pub fn register_order(
        &mut self,
        request: &OrderRequest,
        client_order_id: String,
        timestamp: u64,
    ) -> MMResult<()> {
        // Check for duplicates
        if self.config.detect_duplicates && self.orders.contains_key(&client_order_id) {
            return Err(MMError::InvalidMarketState(format!(
                "duplicate client_order_id: {}",
                client_order_id
            )));
        }

        // Check max open orders
        if self.config.max_open_orders > 0 {
            let symbol_orders = self
                .open_orders_by_symbol
                .get(&request.symbol)
                .map(|v| v.len())
                .unwrap_or(0);

            if symbol_orders >= self.config.max_open_orders {
                return Err(MMError::InvalidMarketState(format!(
                    "max open orders ({}) exceeded for symbol {}",
                    self.config.max_open_orders, request.symbol
                )));
            }
        }

        let order = ManagedOrder::new(
            OrderId::new(&client_order_id), // Temporary ID until exchange responds
            client_order_id.clone(),
            request.symbol.clone(),
            request.side,
            request.order_type,
            request.price.unwrap_or(Decimal::ZERO),
            request.quantity,
            timestamp,
        );

        // Add to symbol index
        self.open_orders_by_symbol
            .entry(request.symbol.clone())
            .or_default()
            .push(client_order_id.clone());

        self.orders.insert(client_order_id, order);

        Ok(())
    }

    /// Updates an order with exchange response.
    ///
    /// # Arguments
    ///
    /// * `client_order_id` - Client-assigned order ID
    /// * `response` - Exchange response
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn update_order(
        &mut self,
        client_order_id: &str,
        response: &OrderResponse,
        timestamp: u64,
    ) -> MMResult<()> {
        let (symbol, is_terminal) = {
            let order = self.orders.get_mut(client_order_id).ok_or_else(|| {
                MMError::InvalidMarketState(format!("order not found: {}", client_order_id))
            })?;

            // Update exchange order ID mapping
            let exchange_id = response.order_id.as_str().to_string();
            if order.order_id.as_str() != exchange_id {
                order.order_id = response.order_id.clone();
                self.orders_by_exchange_id
                    .insert(exchange_id, client_order_id.to_string());
            }

            order.update_status(response.status.clone(), timestamp);
            (order.symbol.clone(), order.is_terminal())
        };

        // Remove from open orders if terminal
        if is_terminal {
            self.remove_from_open_orders(&symbol, client_order_id);
        }

        Ok(())
    }

    /// Records a fill for an order.
    ///
    /// # Arguments
    ///
    /// * `fill` - Fill information
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn record_fill(&mut self, fill: &Fill, timestamp: u64) -> MMResult<()> {
        // Find order by exchange ID
        let client_id = self
            .orders_by_exchange_id
            .get(fill.order_id.as_str())
            .cloned()
            .ok_or_else(|| {
                MMError::InvalidMarketState(format!("order not found for fill: {}", fill.order_id))
            })?;

        let (symbol, is_terminal) = {
            let order = self.orders.get_mut(&client_id).ok_or_else(|| {
                MMError::InvalidMarketState(format!("order not found: {}", client_id))
            })?;

            order.record_fill(fill, timestamp);
            (order.symbol.clone(), order.is_terminal())
        };

        // Remove from open orders if fully filled
        if is_terminal {
            self.remove_from_open_orders(&symbol, &client_id);
        }

        Ok(())
    }

    /// Gets an order by exchange order ID.
    #[must_use]
    pub fn get_order(&self, order_id: &OrderId) -> Option<&ManagedOrder> {
        self.orders_by_exchange_id
            .get(order_id.as_str())
            .and_then(|client_id| self.orders.get(client_id))
    }

    /// Gets an order by client order ID.
    #[must_use]
    pub fn get_order_by_client_id(&self, client_order_id: &str) -> Option<&ManagedOrder> {
        self.orders.get(client_order_id)
    }

    /// Gets a mutable reference to an order by client order ID.
    pub fn get_order_by_client_id_mut(
        &mut self,
        client_order_id: &str,
    ) -> Option<&mut ManagedOrder> {
        self.orders.get_mut(client_order_id)
    }

    /// Gets all open orders.
    #[must_use]
    pub fn get_open_orders(&self) -> Vec<&ManagedOrder> {
        self.orders
            .values()
            .filter(|o| o.is_open() || o.is_pending())
            .collect()
    }

    /// Gets open orders for a specific symbol.
    #[must_use]
    pub fn get_open_orders_for_symbol(&self, symbol: &str) -> Vec<&ManagedOrder> {
        self.open_orders_by_symbol
            .get(symbol)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.orders.get(id))
                    .filter(|o| o.is_open() || o.is_pending())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Checks if an order exists by exchange order ID.
    #[must_use]
    pub fn has_order(&self, order_id: &OrderId) -> bool {
        self.orders_by_exchange_id.contains_key(order_id.as_str())
    }

    /// Checks if an order exists by client order ID.
    #[must_use]
    pub fn has_order_by_client_id(&self, client_order_id: &str) -> bool {
        self.orders.contains_key(client_order_id)
    }

    /// Gets total open quantity for a side.
    #[must_use]
    pub fn get_open_quantity(&self, symbol: &str, side: Side) -> Decimal {
        self.get_open_orders_for_symbol(symbol)
            .iter()
            .filter(|o| o.side == side)
            .map(|o| o.remaining_quantity)
            .sum()
    }

    /// Marks an order as cancelled.
    pub fn mark_cancelled(&mut self, client_order_id: &str, timestamp: u64) -> MMResult<()> {
        let symbol = {
            let order = self.orders.get_mut(client_order_id).ok_or_else(|| {
                MMError::InvalidMarketState(format!("order not found: {}", client_order_id))
            })?;

            let filled_qty = order.filled_quantity;
            order.update_status(OrderStatus::Cancelled { filled_qty }, timestamp);
            order.symbol.clone()
        };

        self.remove_from_open_orders(&symbol, client_order_id);

        Ok(())
    }

    /// Checks for timed out orders.
    ///
    /// Returns a list of client order IDs that have exceeded the timeout.
    #[must_use]
    pub fn check_timeouts(&self, current_time: u64) -> Vec<String> {
        if self.config.order_timeout_ms == 0 {
            return Vec::new();
        }

        self.orders
            .iter()
            .filter(|(_, order)| {
                (order.is_open() || order.is_pending())
                    && order.age_ms(current_time) > self.config.order_timeout_ms
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Removes completed/cancelled orders older than retention period.
    pub fn cleanup(&mut self, retention_ms: u64, current_time: u64) {
        let to_remove: Vec<String> = self
            .orders
            .iter()
            .filter(|(_, order)| {
                order.is_terminal() && current_time.saturating_sub(order.updated_at) > retention_ms
            })
            .map(|(id, _)| id.clone())
            .collect();

        for client_id in to_remove {
            if let Some(order) = self.orders.remove(&client_id) {
                self.orders_by_exchange_id.remove(order.order_id.as_str());
            }
        }
    }

    /// Gets order statistics.
    #[must_use]
    pub fn get_stats(&self) -> OrderManagerStats {
        let mut stats = OrderManagerStats::default();

        for order in self.orders.values() {
            stats.total_orders += 1;
            stats.total_fills += order.fills.len();

            match &order.status {
                OrderStatus::Pending => stats.pending_orders += 1,
                OrderStatus::Open { .. } | OrderStatus::PartiallyFilled { .. } => {
                    stats.open_orders += 1
                }
                OrderStatus::Filled { .. } => stats.filled_orders += 1,
                OrderStatus::Cancelled { .. } => stats.cancelled_orders += 1,
                OrderStatus::Rejected { .. } => stats.rejected_orders += 1,
            }
        }

        stats
    }

    /// Returns the total number of orders.
    #[must_use]
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Returns the number of open orders.
    #[must_use]
    pub fn open_order_count(&self) -> usize {
        self.orders
            .values()
            .filter(|o| o.is_open() || o.is_pending())
            .count()
    }

    /// Removes an order from the open orders index.
    fn remove_from_open_orders(&mut self, symbol: &str, client_order_id: &str) {
        if let Some(ids) = self.open_orders_by_symbol.get_mut(symbol) {
            ids.retain(|id| id != client_order_id);
        }
    }
}

/// Thread-safe wrapper for OrderManager.
///
/// Provides concurrent access to the order manager using `Arc<RwLock<>>`.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{
///     ThreadSafeOrderManager, OrderManagerConfig, OrderRequest
/// };
/// use market_maker_rs::dec;
///
/// let manager = ThreadSafeOrderManager::new(OrderManagerConfig::default());
///
/// // Can be cloned and shared across threads
/// let manager2 = manager.clone();
///
/// // Access with read lock
/// let stats = manager.read(|m| m.get_stats());
///
/// // Access with write lock
/// let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
/// manager.write(|m| m.register_order(&request, "client-1".to_string(), 1000)).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ThreadSafeOrderManager {
    inner: Arc<RwLock<OrderManager>>,
}

impl ThreadSafeOrderManager {
    /// Creates a new thread-safe order manager.
    #[must_use]
    pub fn new(config: OrderManagerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(OrderManager::new(config))),
        }
    }

    /// Creates a thread-safe order manager with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(OrderManagerConfig::default())
    }

    /// Executes a read operation on the order manager.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&OrderManager) -> R,
    {
        let guard = self.inner.read().unwrap();
        f(&guard)
    }

    /// Executes a write operation on the order manager.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut OrderManager) -> R,
    {
        let mut guard = self.inner.write().unwrap();
        f(&mut guard)
    }

    /// Registers a new order.
    pub fn register_order(
        &self,
        request: &OrderRequest,
        client_order_id: String,
        timestamp: u64,
    ) -> MMResult<()> {
        self.write(|m| m.register_order(request, client_order_id, timestamp))
    }

    /// Updates an order with exchange response.
    pub fn update_order(
        &self,
        client_order_id: &str,
        response: &OrderResponse,
        timestamp: u64,
    ) -> MMResult<()> {
        self.write(|m| m.update_order(client_order_id, response, timestamp))
    }

    /// Records a fill.
    pub fn record_fill(&self, fill: &Fill, timestamp: u64) -> MMResult<()> {
        self.write(|m| m.record_fill(fill, timestamp))
    }

    /// Gets order statistics.
    #[must_use]
    pub fn get_stats(&self) -> OrderManagerStats {
        self.read(|m| m.get_stats())
    }

    /// Gets the number of open orders.
    #[must_use]
    pub fn open_order_count(&self) -> usize {
        self.read(|m| m.open_order_count())
    }

    /// Gets total open quantity for a side.
    #[must_use]
    pub fn get_open_quantity(&self, symbol: &str, side: Side) -> Decimal {
        self.read(|m| m.get_open_quantity(symbol, side))
    }

    /// Checks for timed out orders.
    #[must_use]
    pub fn check_timeouts(&self, current_time: u64) -> Vec<String> {
        self.read(|m| m.check_timeouts(current_time))
    }

    /// Marks an order as cancelled.
    pub fn mark_cancelled(&self, client_order_id: &str, timestamp: u64) -> MMResult<()> {
        self.write(|m| m.mark_cancelled(client_order_id, timestamp))
    }

    /// Cleans up old orders.
    pub fn cleanup(&self, retention_ms: u64, current_time: u64) {
        self.write(|m| m.cleanup(retention_ms, current_time));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn create_test_request() -> OrderRequest {
        OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1))
    }

    fn create_test_fill(order_id: &OrderId, price: Decimal, quantity: Decimal) -> Fill {
        Fill {
            order_id: order_id.clone(),
            trade_id: "trade-1".to_string(),
            price,
            quantity,
            side: Side::Buy,
            timestamp: 1000,
            fee: dec!(0.01),
            fee_currency: "USD".to_string(),
        }
    }

    #[test]
    fn test_managed_order_new() {
        let order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Side::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(0.1),
            1000,
        );

        assert!(order.is_pending());
        assert!(!order.is_open());
        assert!(!order.is_terminal());
        assert_eq!(order.filled_quantity, Decimal::ZERO);
        assert_eq!(order.remaining_quantity, dec!(0.1));
        assert_eq!(order.fill_ratio(), Decimal::ZERO);
    }

    #[test]
    fn test_managed_order_record_fill() {
        let mut order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Side::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(1.0),
            1000,
        );

        // Partial fill
        let fill1 = create_test_fill(&order.order_id, dec!(50000.0), dec!(0.3));
        order.record_fill(&fill1, 1001);

        assert_eq!(order.filled_quantity, dec!(0.3));
        assert_eq!(order.remaining_quantity, dec!(0.7));
        assert_eq!(order.average_fill_price, dec!(50000.0));
        assert!(order.is_open());

        // Another partial fill at different price
        let fill2 = create_test_fill(&order.order_id, dec!(49900.0), dec!(0.2));
        order.record_fill(&fill2, 1002);

        assert_eq!(order.filled_quantity, dec!(0.5));
        assert_eq!(order.remaining_quantity, dec!(0.5));
        // VWAP: (50000 * 0.3 + 49900 * 0.2) / 0.5 = 49960
        assert_eq!(order.average_fill_price, dec!(49960.0));

        // Complete fill
        let fill3 = create_test_fill(&order.order_id, dec!(50100.0), dec!(0.5));
        order.record_fill(&fill3, 1003);

        assert_eq!(order.filled_quantity, dec!(1.0));
        assert_eq!(order.remaining_quantity, Decimal::ZERO);
        assert!(order.is_terminal());
        assert_eq!(order.fills.len(), 3);
    }

    #[test]
    fn test_order_manager_register() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        assert_eq!(manager.order_count(), 1);
        assert_eq!(manager.open_order_count(), 1);

        let order = manager.get_order_by_client_id("client-1").unwrap();
        assert!(order.is_pending());
        assert_eq!(order.symbol, "BTC-USD");
    }

    #[test]
    fn test_order_manager_duplicate_detection() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        let result = manager.register_order(&request, "client-1".to_string(), 1001);
        assert!(result.is_err());
    }

    #[test]
    fn test_order_manager_max_open_orders() {
        let config = OrderManagerConfig::default().with_max_open_orders(2);
        let mut manager = OrderManager::new(config);
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&request, "client-2".to_string(), 1001)
            .unwrap();

        let result = manager.register_order(&request, "client-3".to_string(), 1002);
        assert!(result.is_err());
    }

    #[test]
    fn test_order_manager_update_order() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        let response = OrderResponse::new(
            OrderId::new("exchange-123"),
            OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            1001,
        );

        manager.update_order("client-1", &response, 1001).unwrap();

        let order = manager.get_order_by_client_id("client-1").unwrap();
        assert!(order.is_open());
        assert_eq!(order.order_id.as_str(), "exchange-123");

        // Can also look up by exchange ID
        let order2 = manager.get_order(&OrderId::new("exchange-123")).unwrap();
        assert_eq!(order2.client_order_id, "client-1");
    }

    #[test]
    fn test_order_manager_record_fill() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        let exchange_id = OrderId::new("exchange-123");
        let response = OrderResponse::new(
            exchange_id.clone(),
            OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            1001,
        );
        manager.update_order("client-1", &response, 1001).unwrap();

        let fill = create_test_fill(&exchange_id, dec!(50000.0), dec!(0.1));
        manager.record_fill(&fill, 1002).unwrap();

        let order = manager.get_order_by_client_id("client-1").unwrap();
        assert!(order.is_terminal());
        assert_eq!(order.filled_quantity, dec!(0.1));
    }

    #[test]
    fn test_order_manager_get_open_orders() {
        let mut manager = OrderManager::with_defaults();

        // Register multiple orders
        let btc_request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let eth_request = OrderRequest::limit_buy("ETH-USD", dec!(3000.0), dec!(1.0));

        manager
            .register_order(&btc_request, "btc-1".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&btc_request, "btc-2".to_string(), 1001)
            .unwrap();
        manager
            .register_order(&eth_request, "eth-1".to_string(), 1002)
            .unwrap();

        let all_open = manager.get_open_orders();
        assert_eq!(all_open.len(), 3);

        let btc_open = manager.get_open_orders_for_symbol("BTC-USD");
        assert_eq!(btc_open.len(), 2);

        let eth_open = manager.get_open_orders_for_symbol("ETH-USD");
        assert_eq!(eth_open.len(), 1);
    }

    #[test]
    fn test_order_manager_get_open_quantity() {
        let mut manager = OrderManager::with_defaults();

        let buy_request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.5));
        let sell_request = OrderRequest::limit_sell("BTC-USD", dec!(51000.0), dec!(0.3));

        manager
            .register_order(&buy_request, "buy-1".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&buy_request, "buy-2".to_string(), 1001)
            .unwrap();
        manager
            .register_order(&sell_request, "sell-1".to_string(), 1002)
            .unwrap();

        let buy_qty = manager.get_open_quantity("BTC-USD", Side::Buy);
        assert_eq!(buy_qty, dec!(1.0));

        let sell_qty = manager.get_open_quantity("BTC-USD", Side::Sell);
        assert_eq!(sell_qty, dec!(0.3));
    }

    #[test]
    fn test_order_manager_mark_cancelled() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        manager.mark_cancelled("client-1", 1001).unwrap();

        let order = manager.get_order_by_client_id("client-1").unwrap();
        assert!(order.is_terminal());
        assert!(matches!(order.status, OrderStatus::Cancelled { .. }));

        assert_eq!(manager.open_order_count(), 0);
    }

    #[test]
    fn test_order_manager_check_timeouts() {
        let config = OrderManagerConfig::default().with_order_timeout_ms(1000);
        let mut manager = OrderManager::new(config);
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&request, "client-2".to_string(), 2000)
            .unwrap();

        // At time 2500, only client-1 should be timed out
        let timeouts = manager.check_timeouts(2500);
        assert_eq!(timeouts.len(), 1);
        assert!(timeouts.contains(&"client-1".to_string()));

        // At time 3500, both should be timed out
        let timeouts = manager.check_timeouts(3500);
        assert_eq!(timeouts.len(), 2);
    }

    #[test]
    fn test_order_manager_cleanup() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();
        manager.mark_cancelled("client-1", 1001).unwrap();

        // Not old enough to clean up
        manager.cleanup(1000, 1500);
        assert_eq!(manager.order_count(), 1);

        // Old enough now
        manager.cleanup(1000, 3000);
        assert_eq!(manager.order_count(), 0);
    }

    #[test]
    fn test_order_manager_stats() {
        let mut manager = OrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&request, "client-2".to_string(), 1001)
            .unwrap();

        // Update one to open
        let response = OrderResponse::new(
            OrderId::new("exchange-1"),
            OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            1002,
        );
        manager.update_order("client-1", &response, 1002).unwrap();

        // Cancel the other
        manager.mark_cancelled("client-2", 1003).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_orders, 2);
        assert_eq!(stats.open_orders, 1);
        assert_eq!(stats.cancelled_orders, 1);
    }

    #[test]
    fn test_thread_safe_order_manager() {
        let manager = ThreadSafeOrderManager::with_defaults();
        let request = create_test_request();

        manager
            .register_order(&request, "client-1".to_string(), 1000)
            .unwrap();

        assert_eq!(manager.open_order_count(), 1);

        let stats = manager.get_stats();
        assert_eq!(stats.total_orders, 1);
        assert_eq!(stats.pending_orders, 1);
    }

    #[test]
    fn test_order_age() {
        let order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Side::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(0.1),
            1000,
        );

        assert_eq!(order.age_ms(1500), 500);
        assert_eq!(order.age_ms(2000), 1000);
        assert_eq!(order.age_ms(500), 0); // Saturating sub
    }

    #[test]
    fn test_order_notional() {
        let order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Side::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(0.1),
            1000,
        );

        assert_eq!(order.original_notional(), dec!(5000.0));
        assert_eq!(order.filled_notional(), Decimal::ZERO);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config = OrderManagerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OrderManagerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.order_timeout_ms, deserialized.order_timeout_ms);
    }
}
