//! Common types and error definitions for the market making library.
//!
//! This module contains:
//! - Error types using `thiserror`
//! - Type aliases for domain concepts
//! - Common enums and shared data structures

/// Error types and result handling.
pub mod error;

/// Decimal arithmetic helpers.
pub mod decimal;

/// Common type aliases for prices, quantities, and time.
pub mod primitives;

/// Trade direction and option style shared with the other crates.
pub mod action;

/// Fee currency of an execution.
pub mod fee;
