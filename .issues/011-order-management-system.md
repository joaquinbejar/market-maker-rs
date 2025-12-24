# Implement Order Management System (OMS)

## Summary

Add order tracking and lifecycle management to handle order states, partial fills, and reconciliation.

## Motivation

A robust OMS is essential for:
- Tracking all open orders and their states
- Handling partial fills correctly
- Reconciling positions with exchange
- Preventing duplicate orders
- Managing order timeouts and expiry

## Detailed Description

Implement an `OrderManager` that maintains the state of all orders and provides methods for order lifecycle management.

### Proposed API

```rust
use std::collections::HashMap;

/// Internal order state
#[derive(Debug, Clone)]
pub struct ManagedOrder {
    pub order_id: OrderId,
    pub client_order_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub original_price: Decimal,
    pub original_quantity: Decimal,
    pub filled_quantity: Decimal,
    pub remaining_quantity: Decimal,
    pub average_fill_price: Decimal,
    pub status: OrderStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub fills: Vec<Fill>,
}

/// Order manager configuration
pub struct OrderManagerConfig {
    /// Order timeout in milliseconds (0 = no timeout)
    pub order_timeout_ms: u64,
    
    /// Maximum open orders per symbol
    pub max_open_orders: usize,
    
    /// Enable duplicate order detection
    pub detect_duplicates: bool,
}

/// Order manager
pub struct OrderManager {
    config: OrderManagerConfig,
    orders: HashMap<OrderId, ManagedOrder>,
    orders_by_client_id: HashMap<String, OrderId>,
    open_orders_by_symbol: HashMap<String, Vec<OrderId>>,
}

impl OrderManager {
    pub fn new(config: OrderManagerConfig) -> Self;
    
    /// Register a new order (before submission)
    pub fn register_order(&mut self, request: &OrderRequest, client_order_id: String, timestamp: u64) -> MMResult<()>;
    
    /// Update order with exchange response
    pub fn update_order(&mut self, response: &OrderResponse, timestamp: u64) -> MMResult<()>;
    
    /// Record a fill for an order
    pub fn record_fill(&mut self, fill: &Fill, timestamp: u64) -> MMResult<()>;
    
    /// Get order by exchange order ID
    pub fn get_order(&self, order_id: &OrderId) -> Option<&ManagedOrder>;
    
    /// Get order by client order ID
    pub fn get_order_by_client_id(&self, client_order_id: &str) -> Option<&ManagedOrder>;
    
    /// Get all open orders
    pub fn get_open_orders(&self) -> Vec<&ManagedOrder>;
    
    /// Get open orders for a symbol
    pub fn get_open_orders_for_symbol(&self, symbol: &str) -> Vec<&ManagedOrder>;
    
    /// Check if order exists
    pub fn has_order(&self, order_id: &OrderId) -> bool;
    
    /// Get total open quantity for a side
    pub fn get_open_quantity(&self, symbol: &str, side: Side) -> Decimal;
    
    /// Mark order as cancelled
    pub fn mark_cancelled(&mut self, order_id: &OrderId, timestamp: u64) -> MMResult<()>;
    
    /// Check for timed out orders
    pub fn check_timeouts(&mut self, current_time: u64) -> Vec<OrderId>;
    
    /// Remove completed/cancelled orders older than retention period
    pub fn cleanup(&mut self, retention_ms: u64, current_time: u64);
    
    /// Get order statistics
    pub fn get_stats(&self) -> OrderManagerStats;
}

pub struct OrderManagerStats {
    pub total_orders: usize,
    pub open_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub total_fills: usize,
}
```

## Acceptance Criteria

- [ ] `ManagedOrder` struct with full order state
- [ ] `OrderManagerConfig` with timeout and limits
- [ ] `OrderManager` with HashMap-based storage
- [ ] `register_order()` creates pending order entry
- [ ] `update_order()` handles status transitions
- [ ] `record_fill()` updates quantities and average price
- [ ] `get_open_orders()` and filtering by symbol
- [ ] `get_open_quantity()` for position calculation
- [ ] `check_timeouts()` identifies stale orders
- [ ] `cleanup()` removes old completed orders
- [ ] Duplicate order detection (optional)
- [ ] Thread-safe version with `Arc<RwLock<>>` wrapper
- [ ] Unit tests covering:
  - Order lifecycle (new → open → filled)
  - Partial fills
  - Cancellation
  - Timeout detection
  - Duplicate detection
- [ ] Documentation with usage examples

## Technical Notes

- Use `HashMap` for O(1) order lookup
- Maintain separate index for client_order_id lookups
- Average fill price: `sum(fill_price * fill_qty) / total_filled_qty`
- Consider adding event emission for state changes
- Thread-safe wrapper: `pub struct ThreadSafeOrderManager(Arc<RwLock<OrderManager>>)`

## Labels

`enhancement`, `execution`, `priority:high`
