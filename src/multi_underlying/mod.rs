//! Multi-underlying support for managing multiple assets simultaneously.
//!
//! This module provides support for market making across multiple underlyings:
//! - Cross-asset correlation tracking
//! - Capital allocation strategies
//! - Unified risk view across all positions
//! - Per-underlying configuration and limits
//! - Cross-asset hedging suggestions
//! - Closed-form joint quotes, in the `multi_asset` module
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::multi_underlying::{
//!     MultiUnderlyingManager, UnderlyingConfig, CapitalAllocation,
//! };
//!
//! let mut manager = MultiUnderlyingManager::new(dec!(1_000_000.0));
//!
//! // Add underlyings
//! manager.add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)));
//! manager.add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)));
//!
//! // Get unified risk view
//! let risk = manager.get_unified_risk();
//! ```

mod config;
mod manager;
pub mod multi_asset;
mod risk;
mod types;

pub use config::{CapitalAllocationStrategy, UnderlyingConfig};
pub use manager::MultiUnderlyingManager;
pub use risk::{CrossAssetHedge, HedgeType, UnifiedGreeks, UnifiedRisk};
pub use types::{CorrelationEntry, UnderlyingState, UnderlyingStatus};
