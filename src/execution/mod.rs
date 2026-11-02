//! Execution module for exchange connectivity and order management.
//!
//! This module provides traits and types for exchange interaction, enabling
//! multiple venue support and facilitating testing through mock implementations.
//!
//! # Overview
//!
//! The execution module defines:
//!
//! - **Order types**: `OrderRequest`, `OrderResponse`, `OrderStatus`
//! - **Market data types**: `BookLevel`, `OrderBookSnapshot`, `Fill`
//! - **Connector traits**: `ExchangeConnector`, `MarketDataStream`
//! - **Mock implementation**: `MockExchangeConnector` for testing
//! - **Order management**: `OrderManager`, `ManagedOrder` for order lifecycle
//! - **Latency tracking**: `LatencyTracker`, `LatencyStats` for performance monitoring
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     OrderRequest, Action, OrderType, TimeInForce, ExchangeConnector
//! };
//! use market_maker_rs::dec;
//!
//! // Create an order request
//! let request = OrderRequest::new(
//!     "BTC-USD",
//!     Action::Buy,
//!     OrderType::Limit,
//!     Some(dec!(50000.0)),
//!     dec!(0.1),
//! );
//!
//! // In practice, you would use a real exchange connector
//! // let response = connector.submit_order(request).await?;
//! ```

/// Exchange connector trait and types.
pub mod connector;

/// Mock exchange connector for testing.
pub mod mock;

/// Order management system.
pub mod order_manager;

/// Latency tracking and metrics.
pub mod latency;

/// OrderBook-rs connector implementation.
pub mod orderbook_connector;

pub use connector::{
    Action, BookLevel, ExchangeConnector, Fill, MarketDataStream, OrderBookSnapshot, OrderId,
    OrderRequest, OrderResponse, OrderStatus, OrderType, TimeInForce,
};
pub use latency::{
    Histogram, LatencyMeasurement, LatencyMetric, LatencyStats, LatencyTracker,
    LatencyTrackerConfig,
};
pub use mock::{MockConfig, MockExchangeConnector};
pub use order_manager::{
    ManagedOrder, OrderManager, OrderManagerConfig, OrderManagerStats, ThreadSafeOrderManager,
};
pub use orderbook_connector::{OrderBookConnector, OrderBookConnectorConfig};
