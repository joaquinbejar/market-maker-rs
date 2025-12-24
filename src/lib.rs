//! Market Making Library
//!
//! A Rust library implementing quantitative market making strategies, starting with the
//! Avellaneda-Stoikov model. This library provides the mathematical foundations and domain models
//! necessary for building automated market making systems for financial markets.
//!
//! # Overview
//!
//! Market making is the practice of simultaneously providing buy (bid) and sell (ask) quotes
//! in a financial market. The market maker profits from the bid-ask spread while providing
//! liquidity to the market.
//!
//! ## Key Challenges
//!
//! - **Inventory Risk**: Holding positions exposes the market maker to price movements
//! - **Adverse Selection**: Informed traders may trade against you when they have better information
//! - **Optimal Pricing**: Balance between execution probability and profitability
//!
//! # The Avellaneda-Stoikov Model
//!
//! The Avellaneda-Stoikov model (2008) solves the optimal market making problem using
//! stochastic control theory. It determines optimal bid and ask prices given:
//!
//! - Current market price and volatility
//! - Current inventory position
//! - Risk aversion
//! - Time remaining in trading session
//! - Order arrival dynamics
//!
//! # Modules
//!
//! - [`strategy`]: Pure mathematical calculations for quote generation
//! - [`position`]: Inventory tracking and PnL management
//! - [`market_state`]: Market data representation
//! - [`risk`]: Position limits and exposure control
//! - [`types`]: Common types and error definitions
//! - [`prelude`]: Convenient re-exports of commonly used types
//!
//! # Quick Start
//!
//! Import commonly used types with the prelude:
//!
//! ```rust
//! use market_maker_rs::prelude::*;
//! ```
//!
//! # Examples
//!
//! Examples will be added once core functionality is implemented.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

// Re-export Decimal for use throughout the library
pub use rust_decimal::Decimal;
pub use rust_decimal_macros::dec;

/// Market state module containing market data representations.
pub mod market_state;

/// Position tracking module for inventory and PnL management.
pub mod position;

/// Strategy module containing pure mathematical calculations for market making.
///
/// This module implements the Avellaneda-Stoikov model calculations:
/// - Reservation price computation
/// - Optimal spread calculation
/// - Bid/ask quote generation
pub mod strategy;

/// Risk management module for position limits and exposure control.
///
/// This module provides tools for managing risk in market making operations:
/// - Position limits (maximum inventory size)
/// - Notional exposure limits (maximum value at risk)
/// - Order scaling (automatic size reduction near limits)
pub mod risk;

/// Common types and errors.
pub mod types;

/// Prelude module for convenient imports.
pub mod prelude;
