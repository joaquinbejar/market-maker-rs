//! Strategy module containing pure mathematical calculations for market making.
//!
//! This module implements various market making strategies:
//! - Avellaneda-Stoikov model using stochastic control theory
//! - Grid trading for ranging markets
//! - Depth-based offering
//!
//! # Key Formulas (Avellaneda-Stoikov)
//!
//! ## Reservation Price
//! ```text
//! r = s - q * γ * σ² * (T - t)
//! ```
//!
//! ## Optimal Spread
//! ```text
//! spread = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
//! ```
//!
//! ## Optimal Quotes
//! ```text
//! bid = reservation_price - spread/2
//! ask = reservation_price + spread/2
//! ```

/// Core Avellaneda-Stoikov model calculations.
pub mod avellaneda_stoikov;

/// Quote generation logic.
pub mod quote;

/// Strategy configuration.
pub mod config;
pub mod interface;

/// Depth-based offering strategy.
pub mod depth_based;

/// Grid trading strategy.
pub mod grid;
