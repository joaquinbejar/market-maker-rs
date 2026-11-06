//! Backtesting engine for strategy validation on historical data.
//!
//! This module provides an event-driven backtesting engine that simulates
//! strategy execution on historical market data.
//!
//! # Overview
//!
//! The backtesting module includes:
//!
//! - **Data types**: `MarketTick`, `OHLCVBar` for market data
//! - **Data sources**: `HistoricalDataSource` trait and `VecDataSource` implementation
//! - **Strategy trait**: `BacktestStrategy` for strategy integration
//! - **Engine**: `BacktestEngine` for running simulations
//! - **Results**: `BacktestResult` with comprehensive metrics
//! - **Accounting port**: `FillAccountant` to select the accounting authority for fills
//! - **Fill models**: Realistic fill simulation with queue position and market impact
//! - **Performance metrics**: Comprehensive metrics calculation (Sharpe, Sortino, etc.)
//! - **Financial kernel** (feature `kernel`): the engine loop as a resumable,
//!   `Copy` state machine for external RL trainers
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{
//!     BacktestConfig, BacktestEngine, MarketTick, VecDataSource, SlippageModel
//! };
//! use market_maker_rs::dec;
//!
//! // Create sample market data
//! let ticks = vec![
//!     MarketTick::new(1000, dec!(100.0), dec!(1.0), dec!(100.1), dec!(1.0)),
//!     MarketTick::new(1001, dec!(100.1), dec!(1.0), dec!(100.2), dec!(1.0)),
//! ];
//!
//! let data_source = VecDataSource::new(ticks);
//! let config = BacktestConfig::default();
//!
//! // In practice, you would implement BacktestStrategy for your strategy
//! // let mut engine = BacktestEngine::new(config, strategy, data_source);
//! // let result = engine.run();
//! ```

pub mod accounting;

pub mod calibrated;

#[cfg(all(feature = "margin", feature = "instruments"))]
pub mod contracts;

#[cfg(feature = "serde")]
pub mod hyperliquid;

/// `margin-rs` risk gate behind the pretrade and accounting ports.
#[cfg(feature = "margin")]
pub mod margin;

/// Initial margin for a future or a perpetual, behind the same ports.
#[cfg(all(feature = "margin", feature = "instruments"))]
pub mod derivative_margin;

/// Data types for market data.
pub mod data;

pub mod evaluation;

/// Backtesting engine implementation.
pub mod engine;

/// Realistic fill models for backtesting.
pub mod fill_models;

#[cfg(feature = "kernel")]
pub mod kernel;

pub mod latency;

pub mod lifecycle;

pub mod queue_reactive;

pub mod hawkes;

/// Performance metrics calculator.
pub mod metrics;

#[cfg(feature = "margin")]
pub use accounting::MarginLedgerAccountant;
pub use accounting::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
pub use data::{HistoricalDataSource, MarketTick, OHLCVBar, VecDataSource};
#[cfg(all(feature = "margin", feature = "instruments"))]
pub use derivative_margin::{
    DerivativeMarginAccount, DerivativeMarginAccountant, DerivativeMarginPolicy,
};
pub use engine::{
    AccountingStatus, BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy,
    PretradeRecord, SimulatedFill, SlippageModel,
};
pub use fill_models::{
    FillModel, FillResult, ImmediateFillModel, MarketImpactFillModel, ProbabilisticFillModel,
    QueuePositionFillModel, SimulatedOrder,
};
pub use latency::{MAX_SIM_LATENCY_MS, SimLatencyConfig};
pub use lifecycle::{
    EventStamp, ExecutionKind, ExecutionReport, SimClock, SimOrder, SimOrderId, SimOrderState,
};
#[cfg(feature = "margin")]
pub use margin::{MarginAccount, MarginGateAccountant, MarginRiskPolicy};
pub use metrics::{EquityPoint, MetricsCalculator, MetricsConfig, PerformanceMetrics, TradeRecord};
