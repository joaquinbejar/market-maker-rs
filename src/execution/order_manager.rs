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
//!     OrderManager, OrderManagerConfig, OrderRequest, Action
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
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::types::error::{MMError, MMResult};

use super::connector::{
    Action, Fill, OrderId, OrderRequest, OrderResponse, OrderStatus, OrderType,
};
use crate::types::action::Dir;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Internal managed order state.
///
/// Tracks the complete lifecycle of an order including all fills.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{ManagedOrder, OrderId, Action, OrderType, OrderStatus};
/// use market_maker_rs::dec;
///
/// let order = ManagedOrder::new(
///     OrderId::new("12345"),
///     "client-1".to_string(),
///     "BTC-USD".to_string(),
///     Action::Buy,
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
    pub side: Action,
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
        side: Action,
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
    ///
    /// Report only: an overfill whose ratio overflows saturates at the
    /// `Decimal` bound, so it never reads as unfilled.
    #[must_use]
    pub fn fill_ratio(&self) -> Decimal {
        if self.original_quantity > Decimal::ZERO {
            crate::types::decimal::saturating_div(self.filled_quantity, self.original_quantity)
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the notional value of the original order, saturating at the
    /// `Decimal` bounds.
    #[must_use]
    pub fn original_notional(&self) -> Decimal {
        self.original_price.saturating_mul(self.original_quantity)
    }

    /// Returns the notional value of filled quantity, saturating at the
    /// `Decimal` bounds.
    #[must_use]
    pub fn filled_notional(&self) -> Decimal {
        self.average_fill_price.saturating_mul(self.filled_quantity)
    }

    /// Returns the age of the order in milliseconds.
    #[must_use]
    pub fn age_ms(&self, current_time: u64) -> u64 {
        current_time.saturating_sub(self.created_at)
    }

    /// Records a fill and updates quantities and average price.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the fill's side or the
    /// order's side is neither buy nor sell, or [`MMError::NumericalError`]
    /// if the filled quantity, remaining quantity or average price
    /// overflows; the order is unchanged in both cases.
    pub fn record_fill(&mut self, fill: &Fill, timestamp: u64) -> MMResult<()> {
        Dir::try_from(self.side)?;
        Dir::try_from(fill.side)?;
        let overflow = |what: &str| {
            MMError::NumericalError(format!(
                "fill of {} at {} overflows the {what}",
                fill.quantity, fill.price
            ))
        };
        let new_filled = self
            .filled_quantity
            .checked_add(fill.quantity)
            .ok_or_else(|| overflow("filled quantity"))?;

        // Calculate new VWAP
        let average_fill_price = if new_filled > Decimal::ZERO {
            self.average_fill_price
                .checked_mul(self.filled_quantity)
                .and_then(|old| old.checked_add(fill.price.checked_mul(fill.quantity)?))
                .and_then(|total| total.checked_div(new_filled))
                .ok_or_else(|| overflow("average price"))?
        } else {
            self.average_fill_price
        };
        let remaining_quantity = self
            .original_quantity
            .checked_sub(new_filled)
            .ok_or_else(|| overflow("remaining quantity"))?;

        self.average_fill_price = average_fill_price;
        self.filled_quantity = new_filled;
        self.remaining_quantity = remaining_quantity;
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
        Ok(())
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

    /// Passes the order through `policy` and registers it with the approved
    /// quantity.
    ///
    /// This is the same pretrade seam the backtest engine uses, so quotes and
    /// hedges sent through the live path are checked by the same policy.
    /// Nothing is registered on a rejection or a zero approved quantity; a
    /// reduction never increases the quantity. The caller forwards the
    /// order's later lifecycle to the policy with
    /// [`PretradePolicy::on_order_event`].
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if `request.side` is neither
    /// buy nor sell, before the policy is asked, and the
    /// [`OrderManager::register_order`] errors for an approved order. If
    /// registration fails and the policy then fails to release the
    /// reservation, the policy's error is returned instead.
    pub fn register_order_checked(
        &mut self,
        request: &OrderRequest,
        client_order_id: String,
        timestamp: u64,
        purpose: OrderPurpose,
        policy: &mut dyn PretradePolicy,
    ) -> MMResult<PretradeDecision> {
        // Before the policy reserves anything.
        Dir::try_from(request.side)?;
        let pretrade = PretradeRequest::new(
            client_order_id.clone(),
            request.side,
            request.price,
            request.quantity,
            purpose,
            timestamp,
        )
        .with_instrument(request.symbol.clone());
        let decision = policy.check(&pretrade);
        let approved = decision.approved_quantity(request.quantity);
        if approved <= Decimal::ZERO {
            return Ok(decision);
        }
        let mut approved_request = request.clone();
        approved_request.quantity = approved;
        if let Err(error) =
            self.register_order(&approved_request, client_order_id.clone(), timestamp)
        {
            // The policy reserved the approved quantity; nothing reached the
            // venue, so release it. A policy that cannot take the release
            // is the more serious failure and is reported instead.
            policy.on_order_event(&OrderEvent::VenueRejected {
                order_id: client_order_id,
            })?;
            return Err(error);
        }
        Ok(decision)
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
    /// - The side is neither buy nor sell ([`MMError::UnsupportedAction`])
    /// - Duplicate client_order_id exists (if detection enabled)
    /// - Maximum open orders exceeded for the symbol
    pub fn register_order(
        &mut self,
        request: &OrderRequest,
        client_order_id: String,
        timestamp: u64,
    ) -> MMResult<()> {
        Dir::try_from(request.side)?;
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
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The fill's side, or the matched order's side, is neither buy nor
    ///   sell ([`MMError::UnsupportedAction`]), before any state changes
    /// - No order matches the fill's order ID
    pub fn record_fill(&mut self, fill: &Fill, timestamp: u64) -> MMResult<()> {
        Dir::try_from(fill.side)?;
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

            order.record_fill(fill, timestamp)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if `side` is neither buy nor
    /// sell, or [`MMError::NumericalError`] if the total overflows.
    pub fn get_open_quantity(&self, symbol: &str, side: Action) -> MMResult<Decimal> {
        let side = Action::from(Dir::try_from(side)?);
        self.get_open_orders_for_symbol(symbol)
            .iter()
            .filter(|o| o.side == side)
            .try_fold(Decimal::ZERO, |acc, o| {
                acc.checked_add(o.remaining_quantity)
            })
            .ok_or_else(|| {
                MMError::NumericalError(format!("open {side} quantity on {symbol} overflows"))
            })
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
            stats.total_orders = stats.total_orders.saturating_add(1);
            stats.total_fills = stats.total_fills.saturating_add(order.fills.len());

            match &order.status {
                OrderStatus::Pending => {
                    stats.pending_orders = stats.pending_orders.saturating_add(1)
                }
                OrderStatus::Open { .. } | OrderStatus::PartiallyFilled { .. } => {
                    stats.open_orders = stats.open_orders.saturating_add(1)
                }
                OrderStatus::Filled { .. } => {
                    stats.filled_orders = stats.filled_orders.saturating_add(1)
                }
                OrderStatus::Cancelled { .. } => {
                    stats.cancelled_orders = stats.cancelled_orders.saturating_add(1)
                }
                OrderStatus::Rejected { .. } => {
                    stats.rejected_orders = stats.rejected_orders.saturating_add(1)
                }
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
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn read<F, R>(&self, f: F) -> MMResult<R>
    where
        F: FnOnce(&OrderManager) -> R,
    {
        let guard = self
            .inner
            .read()
            .map_err(|_| MMError::LockPoisoned("order manager".to_string()))?;
        Ok(f(&guard))
    }

    /// Executes a write operation on the order manager.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn write<F, R>(&self, f: F) -> MMResult<R>
    where
        F: FnOnce(&mut OrderManager) -> R,
    {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| MMError::LockPoisoned("order manager".to_string()))?;
        Ok(f(&mut guard))
    }

    /// Registers a new order.
    pub fn register_order(
        &self,
        request: &OrderRequest,
        client_order_id: String,
        timestamp: u64,
    ) -> MMResult<()> {
        self.write(|m| m.register_order(request, client_order_id, timestamp))?
    }

    /// Updates an order with exchange response.
    pub fn update_order(
        &self,
        client_order_id: &str,
        response: &OrderResponse,
        timestamp: u64,
    ) -> MMResult<()> {
        self.write(|m| m.update_order(client_order_id, response, timestamp))?
    }

    /// Records a fill.
    ///
    /// # Errors
    ///
    /// Returns the [`OrderManager::record_fill`] errors.
    pub fn record_fill(&self, fill: &Fill, timestamp: u64) -> MMResult<()> {
        self.write(|m| m.record_fill(fill, timestamp))?
    }

    /// Gets order statistics.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn get_stats(&self) -> MMResult<OrderManagerStats> {
        self.read(|m| m.get_stats())
    }

    /// Gets the number of open orders.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn open_order_count(&self) -> MMResult<usize> {
        self.read(|m| m.open_order_count())
    }

    /// Gets total open quantity for a side.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if `side` is neither buy nor
    /// sell.
    pub fn get_open_quantity(&self, symbol: &str, side: Action) -> MMResult<Decimal> {
        self.read(|m| m.get_open_quantity(symbol, side))?
    }

    /// Checks for timed out orders.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn check_timeouts(&self, current_time: u64) -> MMResult<Vec<String>> {
        self.read(|m| m.check_timeouts(current_time))
    }

    /// Marks an order as cancelled.
    pub fn mark_cancelled(&self, client_order_id: &str, timestamp: u64) -> MMResult<()> {
        self.write(|m| m.mark_cancelled(client_order_id, timestamp))?
    }

    /// Cleans up old orders.
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn cleanup(&self, retention_ms: u64, current_time: u64) -> MMResult<()> {
        self.write(|m| m.cleanup(retention_ms, current_time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_fill_ratio_overfill_beyond_bounds_saturates() {
        let id = OrderId::new("OVERFILL");
        let mut order = ManagedOrder::new(
            id.clone(),
            "client".to_string(),
            "BTC-USD".to_string(),
            Action::Buy,
            OrderType::Limit,
            dec!(1),
            dec!(0.0000000000000000000000000001),
            0,
        );
        order
            .record_fill(&create_test_fill(&id, dec!(1), dec!(10)), 1)
            .unwrap();
        assert_eq!(order.fill_ratio(), Decimal::MAX);
    }

    fn create_test_request() -> OrderRequest {
        OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1))
    }

    fn create_test_fill(order_id: &OrderId, price: Decimal, quantity: Decimal) -> Fill {
        Fill {
            symbol: "BTC-USD".to_string(),
            order_id: order_id.clone(),
            trade_id: "trade-1".to_string(),
            price,
            quantity,
            side: Action::Buy,
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
            Action::Buy,
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
    fn test_managed_order_record_fill_overflow_returns_err_unchanged() {
        let mut order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Action::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(10.0),
            1000,
        );
        let fill = create_test_fill(&order.order_id, Decimal::MAX, dec!(2));
        let result = order.record_fill(&fill, 1001);
        assert!(matches!(result, Err(MMError::NumericalError(_))));
        assert_eq!(order.filled_quantity, Decimal::ZERO);
        assert_eq!(order.remaining_quantity, dec!(10.0));
        assert!(order.fills.is_empty());
    }

    #[test]
    fn test_managed_order_record_fill() {
        let mut order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Action::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(1.0),
            1000,
        );

        // Partial fill
        let fill1 = create_test_fill(&order.order_id, dec!(50000.0), dec!(0.3));
        order.record_fill(&fill1, 1001).unwrap();

        assert_eq!(order.filled_quantity, dec!(0.3));
        assert_eq!(order.remaining_quantity, dec!(0.7));
        assert_eq!(order.average_fill_price, dec!(50000.0));
        assert!(order.is_open());

        // Another partial fill at different price
        let fill2 = create_test_fill(&order.order_id, dec!(49900.0), dec!(0.2));
        order.record_fill(&fill2, 1002).unwrap();

        assert_eq!(order.filled_quantity, dec!(0.5));
        assert_eq!(order.remaining_quantity, dec!(0.5));
        // VWAP: (50000 * 0.3 + 49900 * 0.2) / 0.5 = 49960
        assert_eq!(order.average_fill_price, dec!(49960.0));

        // Complete fill
        let fill3 = create_test_fill(&order.order_id, dec!(50100.0), dec!(0.5));
        order.record_fill(&fill3, 1003).unwrap();

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
    fn test_order_manager_open_quantity_overflow_returns_numerical_error() {
        let mut manager = OrderManager::with_defaults();
        let request = OrderRequest::limit_buy("BTC-USD", dec!(1), Decimal::MAX);
        manager
            .register_order(&request, "a".to_string(), 1000)
            .unwrap();
        manager
            .register_order(&request, "b".to_string(), 1001)
            .unwrap();
        let result = manager.get_open_quantity("BTC-USD", Action::Buy);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
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

        let buy_qty = manager.get_open_quantity("BTC-USD", Action::Buy).unwrap();
        assert_eq!(buy_qty, dec!(1.0));

        let sell_qty = manager.get_open_quantity("BTC-USD", Action::Sell).unwrap();
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

        assert_eq!(manager.open_order_count().unwrap(), 1);

        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.total_orders, 1);
        assert_eq!(stats.pending_orders, 1);
    }

    #[test]
    fn test_order_age() {
        let order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Action::Buy,
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
            Action::Buy,
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

    fn unsupported() -> MMError {
        MMError::UnsupportedAction {
            action: Action::Other,
        }
    }

    #[test]
    fn test_register_order_unsupported_action_registers_nothing() {
        let mut manager = OrderManager::with_defaults();
        let mut request = create_test_request();
        request.side = Action::Other;

        assert_eq!(
            manager.register_order(&request, "client-1".to_string(), 1000),
            Err(unsupported())
        );
        assert_eq!(manager.order_count(), 0);
        assert_eq!(manager.open_order_count(), 0);
        assert!(!manager.has_order_by_client_id("client-1"));
    }

    #[test]
    fn test_register_order_checked_unsupported_action_skips_the_policy() {
        use crate::risk::RiskLimits;
        use crate::risk::pretrade::LimitsPolicy;

        let mut manager = OrderManager::with_defaults();
        let mut policy =
            LimitsPolicy::new(RiskLimits::new(dec!(10), dec!(1_000_000), dec!(0)).unwrap());
        let mut request = create_test_request();
        request.side = Action::Other;

        assert_eq!(
            manager.register_order_checked(
                &request,
                "client-1".to_string(),
                1000,
                OrderPurpose::Quote,
                &mut policy,
            ),
            Err(unsupported())
        );
        assert_eq!(manager.order_count(), 0);
        // The policy reserved nothing.
        assert_eq!(policy.live_quantity(Action::Buy).unwrap(), Decimal::ZERO);
        assert_eq!(policy.live_quantity(Action::Sell).unwrap(), Decimal::ZERO);
    }

    #[test]
    fn test_record_fill_unsupported_action_leaves_order_unchanged() {
        let mut manager = OrderManager::with_defaults();
        manager
            .register_order(&create_test_request(), "client-1".to_string(), 1000)
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
        let before = manager.get_order_by_client_id("client-1").unwrap().clone();

        let mut fill = create_test_fill(&exchange_id, dec!(50000.0), dec!(0.1));
        fill.side = Action::Other;
        assert_eq!(manager.record_fill(&fill, 1002), Err(unsupported()));

        let after = manager.get_order_by_client_id("client-1").unwrap();
        assert_eq!(after.filled_quantity, before.filled_quantity);
        assert_eq!(after.status, before.status);
        assert!(after.fills.is_empty());
        assert_eq!(manager.open_order_count(), 1);
    }

    #[test]
    fn test_get_open_quantity_unsupported_action_returns_err() {
        let mut manager = OrderManager::with_defaults();
        manager
            .register_order(&create_test_request(), "client-1".to_string(), 1000)
            .unwrap();
        assert_eq!(
            manager.get_open_quantity("BTC-USD", Action::Other),
            Err(unsupported())
        );

        let shared = ThreadSafeOrderManager::with_defaults();
        assert_eq!(
            shared.get_open_quantity("BTC-USD", Action::Other),
            Err(unsupported())
        );
        assert_eq!(
            shared.get_open_quantity("BTC-USD", Action::Buy).unwrap(),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_managed_order_record_fill_unsupported_action_leaves_order_unchanged() {
        let order = ManagedOrder::new(
            OrderId::new("12345"),
            "client-1".to_string(),
            "BTC-USD".to_string(),
            Action::Buy,
            OrderType::Limit,
            dec!(50000.0),
            dec!(1.0),
            1000,
        );

        // An unsupported fill side.
        let mut direct = order.clone();
        let mut fill = create_test_fill(&order.order_id, dec!(50000.0), dec!(0.3));
        fill.side = Action::Other;
        assert_eq!(direct.record_fill(&fill, 1001), Err(unsupported()));

        // An order whose side the caller changed to an unsupported one.
        let mut altered = order.clone();
        altered.side = Action::Other;
        let fill = create_test_fill(&order.order_id, dec!(50000.0), dec!(0.3));
        assert_eq!(altered.record_fill(&fill, 1001), Err(unsupported()));

        for unchanged in [&direct, &altered] {
            assert_eq!(unchanged.filled_quantity, Decimal::ZERO);
            assert_eq!(unchanged.remaining_quantity, dec!(1.0));
            assert_eq!(unchanged.average_fill_price, Decimal::ZERO);
            assert_eq!(unchanged.status, OrderStatus::Pending);
            assert_eq!(unchanged.updated_at, 1000);
            assert!(unchanged.fills.is_empty());
        }
    }

    #[test]
    fn test_thread_safe_order_manager_poisoned_lock_returns_lock_poisoned() {
        let manager = ThreadSafeOrderManager::new(OrderManagerConfig::default());
        let clone = manager.clone();
        let joined = std::thread::spawn(move || {
            let _ = clone.write(|_| panic!("poison the lock"));
        })
        .join();
        assert!(joined.is_err());
        assert!(matches!(
            manager.open_order_count(),
            Err(MMError::LockPoisoned(_))
        ));
        assert!(matches!(manager.get_stats(), Err(MMError::LockPoisoned(_))));
    }
}
