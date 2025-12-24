//! Risk management module for position limits and exposure control.
//!
//! This module provides tools for managing risk in market making operations,
//! including position limits, notional exposure limits, and order scaling.
//!
//! # Overview
//!
//! Market makers must carefully manage their inventory to avoid excessive
//! exposure to price movements. This module provides:
//!
//! - **Position Limits**: Maximum absolute position size (units)
//! - **Notional Limits**: Maximum exposure in currency terms
//! - **Order Scaling**: Automatic reduction of order sizes near limits
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::risk::RiskLimits;
//! use market_maker_rs::dec;
//!
//! let limits = RiskLimits::new(
//!     dec!(100.0),  // max 100 units position
//!     dec!(10000.0), // max $10,000 notional
//!     dec!(0.5),    // 50% scaling factor
//! ).unwrap();
//!
//! // Check if an order is allowed
//! let current_position = dec!(50.0);
//! let order_size = dec!(10.0);
//! let price = dec!(100.0);
//!
//! assert!(limits.check_order(current_position, order_size, price).unwrap());
//! ```

mod limits;

pub use limits::RiskLimits;
