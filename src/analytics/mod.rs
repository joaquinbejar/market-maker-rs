//! Analytics module for market data analysis.
//!
//! This module provides tools for analyzing market microstructure data,
//! including order flow analysis, trade flow metrics, toxicity detection,
//! and dynamic parameter estimation.
//!
//! # Submodules
//!
//! - `order_flow`: Order flow imbalance analysis and trade tracking
//! - `vpin`: VPIN (Volume-Synchronized Probability of Informed Trading) calculation
//! - `intensity`: Dynamic order intensity estimation for A-S model
//! - `live_metrics`: Real-time operational metrics tracking
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::order_flow::{OrderFlowAnalyzer, Trade, TradeSide};
//! use market_maker_rs::dec;
//!
//! let mut analyzer = OrderFlowAnalyzer::new(5000);
//! analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
//!
//! let stats = analyzer.get_stats(2000);
//! println!("Imbalance: {}", stats.imbalance);
//! ```

/// Order flow imbalance analysis.
pub mod order_flow;

/// VPIN (Volume-Synchronized Probability of Informed Trading) calculation.
pub mod vpin;

/// Dynamic order intensity estimation.
pub mod intensity;

/// Live metrics tracking for real-time monitoring.
pub mod live_metrics;

pub use intensity::{
    FillObservation, FillSide, IntensityEstimate, ObservationStats, OrderIntensityConfig,
    OrderIntensityEstimator,
};
pub use live_metrics::{Counter, Gauge, LiveMetrics, MetricsSnapshot, SharedLiveMetrics};
pub use order_flow::{
    OrderFlowAnalyzer, OrderFlowAnalyzerBuilder, OrderFlowStats, Trade, TradeSide,
};
pub use vpin::{BucketStats, TradeClassifier, VPINCalculator, VPINConfig, VolumeBucket};
