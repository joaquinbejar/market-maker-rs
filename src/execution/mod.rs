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
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     OrderRequest, Side, OrderType, TimeInForce, ExchangeConnector
//! };
//! use market_maker_rs::dec;
//!
//! // Create an order request
//! let request = OrderRequest::new(
//!     "BTC-USD",
//!     Side::Buy,
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

pub use connector::{
    BookLevel, ExchangeConnector, Fill, MarketDataStream, OrderBookSnapshot, OrderId, OrderRequest,
    OrderResponse, OrderStatus, OrderType, Side, TimeInForce,
};
pub use mock::{MockConfig, MockExchangeConnector};
pub use order_manager::{
    ManagedOrder, OrderManager, OrderManagerConfig, OrderManagerStats, ThreadSafeOrderManager,
};
