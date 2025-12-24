# Define Exchange Connector Trait

## Summary

Create an abstract trait for exchange connectivity to enable multiple venue support and facilitate testing.

## Motivation

A well-defined exchange connector interface allows:
- Supporting multiple exchanges with the same strategy code
- Easy mocking for unit and integration tests
- Clean separation between strategy logic and execution
- Future support for smart order routing

## Detailed Description

Define traits and types for exchange interaction including order submission, cancellation, and market data subscription.

### Proposed API

```rust
use async_trait::async_trait;

/// Unique identifier for an order
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderId(pub String);

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    PostOnly,
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    GoodTilCancel,
    ImmediateOrCancel,
    FillOrKill,
    GoodTilTime(u64),
}

/// Order status
#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Pending,
    Open { filled_qty: Decimal },
    PartiallyFilled { filled_qty: Decimal, remaining_qty: Decimal },
    Filled { filled_qty: Decimal, avg_price: Decimal },
    Cancelled { filled_qty: Decimal },
    Rejected { reason: String },
}

/// Order request
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Option<Decimal>,
    pub quantity: Decimal,
    pub time_in_force: TimeInForce,
    pub client_order_id: Option<String>,
}

/// Order response
#[derive(Debug, Clone)]
pub struct OrderResponse {
    pub order_id: OrderId,
    pub client_order_id: Option<String>,
    pub status: OrderStatus,
    pub timestamp: u64,
}

/// Trade/fill information
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: OrderId,
    pub trade_id: String,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: Side,
    pub timestamp: u64,
    pub fee: Decimal,
    pub fee_currency: String,
}

/// Order book level
#[derive(Debug, Clone)]
pub struct BookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

/// Order book snapshot
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub symbol: String,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub timestamp: u64,
}

/// Exchange connector trait
#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Submit a new order
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderResponse>;
    
    /// Cancel an existing order
    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<OrderResponse>;
    
    /// Modify an existing order (cancel + replace)
    async fn modify_order(
        &self,
        order_id: &OrderId,
        new_price: Option<Decimal>,
        new_quantity: Option<Decimal>,
    ) -> MMResult<OrderResponse>;
    
    /// Get order status
    async fn get_order_status(&self, order_id: &OrderId) -> MMResult<OrderResponse>;
    
    /// Get all open orders for a symbol
    async fn get_open_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>>;
    
    /// Cancel all open orders for a symbol
    async fn cancel_all_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>>;
    
    /// Get current order book snapshot
    async fn get_orderbook(&self, symbol: &str, depth: usize) -> MMResult<OrderBookSnapshot>;
    
    /// Get account balance for an asset
    async fn get_balance(&self, asset: &str) -> MMResult<Decimal>;
}

/// Market data stream trait
#[async_trait]
pub trait MarketDataStream: Send + Sync {
    /// Subscribe to order book updates
    async fn subscribe_orderbook(&self, symbol: &str) -> MMResult<()>;
    
    /// Subscribe to trade stream
    async fn subscribe_trades(&self, symbol: &str) -> MMResult<()>;
    
    /// Get next order book update (blocking)
    async fn next_orderbook_update(&self) -> MMResult<OrderBookSnapshot>;
    
    /// Get next trade (blocking)
    async fn next_trade(&self) -> MMResult<Fill>;
}
```

## Acceptance Criteria

- [ ] All type definitions: `OrderId`, `Side`, `OrderType`, `TimeInForce`, `OrderStatus`
- [ ] `OrderRequest` and `OrderResponse` structs
- [ ] `Fill` struct for trade information
- [ ] `BookLevel` and `OrderBookSnapshot` for market data
- [ ] `ExchangeConnector` async trait with all methods
- [ ] `MarketDataStream` trait for real-time data
- [ ] `MockExchangeConnector` implementation for testing
- [ ] Comprehensive documentation for each type and method
- [ ] Unit tests for mock implementation
- [ ] Example showing trait usage

## Technical Notes

- Use `async_trait` crate for async trait methods
- All methods return `MMResult<T>` for consistent error handling
- Consider adding `Clone` bounds where needed for Arc usage
- Mock implementation should support configurable latency and failure injection
- Symbol format should be exchange-agnostic (e.g., "BTC-USD")

## Labels

`enhancement`, `execution`, `priority:high`
