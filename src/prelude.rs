//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types, traits, and functions
//! from the market making library. Users can import everything they need with:
//!
//! ```rust
//! use market_maker_rs::prelude::*;
//! ```

// Re-export Decimal and helper functions
pub use crate::types::decimal::{decimal_ln, decimal_powi, decimal_sqrt};
pub use crate::{Decimal, dec};

// Re-export types module
pub use crate::types::error::{MMError, MMResult};
pub use crate::types::primitives::{
    OrderIntensity, Price, Quantity, RiskAversion, Timestamp, Volatility,
};

// Re-export strategy types
pub use crate::strategy::config::StrategyConfig;
pub use crate::strategy::quote::Quote;

// Re-export position types
pub use crate::position::inventory::InventoryPosition;
pub use crate::position::pnl::PnL;

// Re-export market state types
pub use crate::market_state::snapshot::MarketState;

// Re-export risk types
pub use crate::risk::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, DrawdownRecord, DrawdownTracker,
    RiskLimits, TriggerReason,
};
