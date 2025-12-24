//! Risk management module for position limits, exposure control, circuit breakers, and drawdown tracking.
//!
//! This module provides tools for managing risk in market making operations,
//! including position limits, notional exposure limits, order scaling,
//! circuit breakers, and drawdown monitoring.
//!
//! # Overview
//!
//! Market makers must carefully manage their inventory to avoid excessive
//! exposure to price movements. This module provides:
//!
//! - **Position Limits**: Maximum absolute position size (units)
//! - **Notional Limits**: Maximum exposure in currency terms
//! - **Order Scaling**: Automatic reduction of order sizes near limits
//! - **Circuit Breakers**: Automatic trading halts on adverse conditions
//! - **Drawdown Tracking**: Monitor and limit decline from peak equity
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::risk::{RiskLimits, DrawdownTracker, CircuitBreaker, CircuitBreakerConfig};
//! use market_maker_rs::dec;
//!
//! // Position limits
//! let limits = RiskLimits::new(
//!     dec!(100.0),  // max 100 units position
//!     dec!(10000.0), // max $10,000 notional
//!     dec!(0.5),    // 50% scaling factor
//! ).unwrap();
//!
//! assert!(limits.check_order(dec!(50.0), dec!(10.0), dec!(100.0)).unwrap());
//!
//! // Circuit breaker
//! let config = CircuitBreakerConfig::new(
//!     dec!(1000.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000
//! ).unwrap();
//! let breaker = CircuitBreaker::new(config);
//! assert!(breaker.is_trading_allowed());
//!
//! // Drawdown tracking
//! let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
//! tracker.update(dec!(9000.0), 1000);
//! assert_eq!(tracker.current_drawdown(), dec!(0.1)); // 10% drawdown
//! ```

mod circuit_breaker;
mod drawdown;
mod limits;

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, TriggerReason,
};
pub use drawdown::{DrawdownRecord, DrawdownTracker};
pub use limits::RiskLimits;
