//! OptionStratLib integration module for options market making.
//!
//! This module provides integration with the `optionstratlib` library for:
//! - Options pricing using Black-Scholes and other models
//! - Greeks calculation (delta, gamma, theta, vega, rho)
//! - Expiration date handling
//! - Volatility model integration
//!
//! # Feature Flag
//!
//! This module is only available when the `options` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! market-maker-rs = { version = "0.2", features = ["options"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::options::{PortfolioGreeks, OptionsAdapter};
//! use optionstratlib::{Options, ExpirationDate, OptionStyle, pos};
//!
//! // Create an option using OptionStratLib
//! let option = Options::new(
//!     OptionType::European,
//!     Side::Long,
//!     "BTC".to_string(),
//!     pos!(50000.0),
//!     ExpirationDate::Days(pos!(30.0)),
//!     pos!(0.6),
//!     pos!(1.0),
//!     pos!(48000.0),
//!     0.05,
//!     OptionStyle::Call,
//!     0.0,
//!     None,
//! );
//!
//! // Calculate Greeks
//! let greeks = OptionsAdapter::calculate_greeks(&option).unwrap();
//! println!("Delta: {}", greeks.delta);
//! ```

/// Adapter functions for OptionStratLib integration.
pub mod adapter;

/// Portfolio Greeks aggregation and tracking.
pub mod greeks;

/// Options-specific market making functionality.
pub mod market_maker;

pub use adapter::OptionsAdapter;
pub use greeks::{PortfolioGreeks, PositionGreeks};
pub use market_maker::{
    GreeksLimits, HedgeOrder, HedgeType, OptionsMarketMaker, OptionsMarketMakerConfig,
    OptionsMarketMakerImpl,
};
