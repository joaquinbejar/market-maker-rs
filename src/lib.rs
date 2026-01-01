//! # Market Making Library
//!
//! A comprehensive Rust library implementing quantitative market making strategies based on
//! the Avellaneda-Stoikov model and extensions. This library provides production-ready
//! components for building automated market making systems for financial markets.
//!
//! ## Overview
//!
//! Market making is the practice of simultaneously providing buy (bid) and sell (ask) quotes
//! in a financial market. The market maker profits from the bid-ask spread while providing
//! liquidity to the market.
//!
//! ### Key Challenges Addressed
//!
//! - **Inventory Risk**: Dynamic quote skewing based on position
//! - **Adverse Selection**: Order flow toxicity detection with VPIN
//! - **Optimal Pricing**: Stochastic control theory for spread optimization
//! - **Risk Management**: Circuit breakers, drawdown limits, and position controls
//! - **Multi-Asset**: Correlation-aware portfolio risk management
//! - **Options Market Making**: Greeks-aware quoting with delta hedging
//!
//! ## Features
//!
//! ### Strategy Models
//!
//! - **Avellaneda-Stoikov**: Classic optimal market making with reservation price
//! - **GLFT Extension**: Guéant-Lehalle-Fernandez-Tapia with terminal penalties
//! - **Grid Trading**: Multi-level order placement with geometric/arithmetic spacing
//! - **Adaptive Spread**: Dynamic spread adjustment based on order book imbalance
//! - **Depth-Based Offering**: Size adjustment based on market depth
//!
//! ### Risk Management
//!
//! - **Position Limits**: Maximum inventory size controls
//! - **Notional Limits**: Maximum value at risk
//! - **Circuit Breakers**: Automatic trading halts on adverse conditions
//! - **Drawdown Tracking**: Peak-to-trough monitoring with configurable limits
//! - **Alert System**: Configurable alerts for critical events
//! - **Portfolio Risk**: Correlation matrix and multi-asset VaR
//!
//! ### Analytics
//!
//! - **Order Flow Analysis**: Trade flow imbalance and toxicity metrics
//! - **VPIN Calculator**: Volume-synchronized probability of informed trading
//! - **Order Intensity Estimation**: Fill rate modeling for parameter calibration
//! - **Live Metrics**: Real-time operational metrics with atomic counters
//! - **Prometheus Export**: Optional metrics export with Grafana dashboard
//!
//! ### Backtesting
//!
//! - **Event-Driven Engine**: Tick-by-tick simulation
//! - **Fill Models**: Immediate, queue position, probabilistic, market impact
//! - **Performance Metrics**: Sharpe, Sortino, Calmar, max drawdown, profit factor
//! - **Slippage Models**: Fixed, percentage, volatility-based
//!
//! ### Execution
//!
//! - **Exchange Connector Trait**: Abstract interface for any exchange
//! - **Order Manager**: Order lifecycle management with state tracking
//! - **Latency Tracking**: Histogram-based latency measurement
//! - **Mock Connector**: Testing without real exchange connectivity
//! - **OrderBook-rs Connector**: Integration with lock-free order book
//!
//! ### Options Market Making (Feature: `options`)
//!
//! - **Greeks Calculation**: Delta, gamma, theta, vega, rho via OptionStratLib
//! - **Portfolio Greeks**: Aggregation across multiple positions
//! - **Greeks-Aware Quoting**: Spread adjustment based on gamma exposure
//! - **Delta Hedging**: Automatic hedge order generation
//! - **Risk Management**: Greeks-based limits and circuit breakers
//! - **Auto-Hedging**: Configurable triggers for delta neutralization
//!
//! ### Option Chain Integration (Feature: `chain`)
//!
//! - **Multi-Strike Quoting**: Quote all strikes in an expiration
//! - **Chain-Level Risk**: Aggregate Greeks across the chain
//! - **ATM Detection**: Automatic spread adjustment for ATM options
//! - **Chain Risk Manager**: Chain-wide limits and hedging
//!
//! ### Parameter Calibration
//!
//! - **Risk Aversion (γ)**: Calibration from inventory half-life
//! - **Order Intensity (k)**: Estimation from historical fill rates
//! - **Volatility Regimes**: Automatic detection and parameter adjustment
//!
//! ## The Avellaneda-Stoikov Model
//!
//! The Avellaneda-Stoikov model (2008) solves the optimal market making problem using
//! stochastic control theory. Key formulas:
//!
//! ### Reservation Price
//! ```text
//! r = s - q × γ × σ² × (T - t)
//! ```
//!
//! ### Optimal Spread
//! ```text
//! spread = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)
//! ```
//!
//! Where:
//! - `s`: Mid price
//! - `q`: Current inventory
//! - `γ`: Risk aversion parameter
//! - `σ`: Volatility
//! - `T - t`: Time remaining
//! - `k`: Order arrival intensity
//!
//! ## Modules
//!
//! - [`strategy`]: Quote generation algorithms (A-S, GLFT, Grid, Adaptive)
//! - [`position`]: Inventory tracking and PnL management
//! - [`market_state`]: Market data and volatility estimation
//! - [`risk`]: Limits, circuit breakers, alerts, and portfolio risk
//! - [`analytics`]: Order flow, VPIN, intensity estimation, live metrics
//! - [`execution`]: Exchange connectivity, order management, latency tracking
//! - [`backtest`]: Historical simulation with fill models and metrics
//! - [`types`]: Common types, decimals, and error definitions
//! - [`prelude`]: Convenient re-exports of commonly used types
//! - `options`: Options pricing, Greeks, and market making (feature: `options`)
//! - `chain`: Option chain integration and multi-strike quoting (feature: `chain`)
//!
//! ## Quick Start
//!
//! ```rust
//! use market_maker_rs::prelude::*;
//!
//! // Calculate optimal quotes using Avellaneda-Stoikov
//! let mid_price = dec!(100.0);
//! let inventory = dec!(5.0);
//! let risk_aversion = dec!(0.5);    // γ
//! let volatility = dec!(0.02);
//! let time_to_terminal_ms = 3600_000; // 1 hour
//! let order_intensity = dec!(1.5);  // k
//!
//! let (bid, ask) = market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes(
//!     mid_price,
//!     inventory,
//!     risk_aversion,
//!     volatility,
//!     time_to_terminal_ms,
//!     order_intensity,
//! ).unwrap();
//!
//! println!("Bid: {}, Ask: {}", bid, ask);
//! ```
//!
//! ## Feature Flags
//!
//! - `prometheus`: Enable Prometheus metrics export (adds `prometheus`, `hyper`, `tokio` dependencies)
//! - `serde`: Enable serialization/deserialization for all types
//! - `options`: Enable OptionStratLib integration for options pricing and Greeks calculation
//! - `chain`: Enable Option-Chain-OrderBook integration (includes `options`)
//!
//! ## Examples
//!
//! ### Risk Management
//!
//! ```rust
//! use market_maker_rs::prelude::*;
//!
//! // Set up position limits
//! let limits = RiskLimits::new(
//!     dec!(100.0),   // max 100 units position
//!     dec!(10000.0), // max $10,000 notional
//!     dec!(0.5),     // 50% scaling factor
//! ).unwrap();
//!
//! // Check if order is allowed
//! let allowed = limits.check_order(dec!(50.0), dec!(10.0), dec!(100.0)).unwrap();
//!
//! // Circuit breaker for automatic trading halts
//! let config = CircuitBreakerConfig::new(
//!     dec!(1000.0), // max daily loss
//!     dec!(0.05),   // max loss per trade (5%)
//!     5,            // max consecutive losses
//!     dec!(0.10),   // max drawdown (10%)
//!     300_000,      // cooldown period (5 min)
//!     60_000,       // loss window (1 min)
//! ).unwrap();
//! let breaker = CircuitBreaker::new(config);
//! ```
//!
//! ### Backtesting
//!
//! ```rust,ignore
//! use market_maker_rs::prelude::*;
//!
//! // Configure backtest
//! let config = BacktestConfig::default()
//!     .with_initial_capital(dec!(100000.0))
//!     .with_fee_rate(dec!(0.001))
//!     .with_slippage(SlippageModel::Fixed(dec!(0.01)));
//!
//! // Run backtest with your strategy
//! let mut engine = BacktestEngine::new(config, strategy, data_source);
//! let result = engine.run();
//!
//! println!("Net PnL: {}", result.net_pnl);
//! println!("Sharpe Ratio: {:?}", result.sharpe_ratio);
//! println!("Max Drawdown: {}", result.max_drawdown);
//! ```
//!
//! ### Portfolio Risk
//!
//! ```rust
//! use market_maker_rs::risk::portfolio::*;
//! use market_maker_rs::dec;
//!
//! // Create correlation matrix
//! let btc = AssetId::new("BTC");
//! let eth = AssetId::new("ETH");
//! let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
//! matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
//!
//! // Calculate portfolio risk
//! let mut portfolio = PortfolioPosition::new();
//! portfolio.set_position(btc, dec!(1.0), dec!(0.05));
//! portfolio.set_position(eth, dec!(10.0), dec!(0.08));
//!
//! let calculator = PortfolioRiskCalculator::new(matrix);
//! let vol = calculator.portfolio_volatility(&portfolio).unwrap();
//! ```
//!
//! ### Options Greeks (Feature: `options`)
//!
//! ```rust,ignore
//! use market_maker_rs::options::{OptionsAdapter, PortfolioGreeks, PositionGreeks};
//! use optionstratlib::model::option::Options;
//!
//! // Calculate Greeks for an option
//! let greeks = OptionsAdapter::calculate_greeks(&option).unwrap();
//! println!("Delta: {}, Gamma: {}", greeks.delta, greeks.gamma);
//!
//! // Aggregate portfolio Greeks
//! let mut portfolio = PortfolioGreeks::new();
//! portfolio.add(&greeks, dec!(10.0)); // 10 contracts
//!
//! // Check delta neutrality
//! let shares_to_hedge = portfolio.shares_to_hedge(dec!(100.0));
//! ```
//!
//! ### Options Market Making (Feature: `options`)
//!
//! ```rust,ignore
//! use market_maker_rs::options::{
//!     OptionsMarketMaker, OptionsMarketMakerImpl, OptionsMarketMakerConfig,
//!     GreeksLimits, PortfolioGreeks,
//! };
//!
//! // Create market maker with Greeks-aware quoting
//! let config = OptionsMarketMakerConfig::default();
//! let market_maker = OptionsMarketMakerImpl::new(config);
//!
//! // Calculate Greeks-adjusted quotes
//! let (bid, ask) = market_maker.calculate_greeks_adjusted_quotes(
//!     &option,
//!     &portfolio_greeks,
//!     &risk_limits,
//! ).unwrap();
//!
//! // Get delta hedge suggestions
//! let hedges = market_maker.calculate_delta_hedge(
//!     &portfolio_greeks,
//!     underlying_price,
//!     "BTC",
//! ).unwrap();
//! ```
//!
//! ### Greeks Risk Management (Feature: `options`)
//!
//! ```rust,ignore
//! use market_maker_rs::options::{
//!     GreeksRiskManager, AutoHedgerConfig, GreeksLimits, OrderDecision,
//! };
//!
//! // Create risk manager with auto-hedging
//! let limits = GreeksLimits::default();
//! let hedger_config = AutoHedgerConfig::default();
//! let mut risk_manager = GreeksRiskManager::new("BTC", limits, hedger_config);
//!
//! // Check if order is allowed
//! let decision = risk_manager.check_order(&option_greeks, dec!(10.0));
//! match decision {
//!     OrderDecision::Allowed => { /* proceed */ },
//!     OrderDecision::Scaled { new_size, .. } => { /* use scaled size */ },
//!     OrderDecision::Rejected { reason } => { /* reject */ },
//! }
//!
//! // Check if hedging is needed
//! if risk_manager.needs_hedge() {
//!     let hedge = risk_manager.calculate_hedge_order(underlying_price);
//! }
//! ```
//!
//! ### Option Chain Market Making (Feature: `chain`)
//!
//! ```rust,ignore
//! use market_maker_rs::chain::{ChainMarketMaker, ChainMarketMakerConfig};
//! use option_chain_orderbook::orderbook::ExpirationOrderBook;
//! use std::sync::Arc;
//!
//! // Create chain market maker
//! let chain = Arc::new(ExpirationOrderBook::new("BTC", expiration));
//! let config = ChainMarketMakerConfig::default();
//! let mm = ChainMarketMaker::new(chain, config);
//!
//! // Refresh all quotes across the chain
//! let quotes = mm.refresh_all_quotes(underlying_price).unwrap();
//!
//! // Check chain risk status
//! let status = mm.check_chain_risk();
//! if !status.can_quote() {
//!     // Stop quoting or hedge
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

// Re-export Decimal for use throughout the library
pub use rust_decimal::Decimal;
pub use rust_decimal_macros::dec;

/// Market state module containing market data representations.
///
/// Provides:
/// - Market snapshots with bid/ask prices
/// - Volatility estimation (simple, EWMA, Parkinson)
pub mod market_state;

/// Position tracking module for inventory and PnL management.
///
/// Provides:
/// - Inventory position tracking with average cost
/// - Realized and unrealized PnL calculation
pub mod position;

/// Strategy module containing quote generation algorithms.
///
/// Implements multiple market making strategies:
/// - **Avellaneda-Stoikov**: Optimal quotes using stochastic control
/// - **GLFT**: Extension with terminal inventory penalties
/// - **Grid Trading**: Multi-level orders with configurable spacing
/// - **Adaptive Spread**: Dynamic adjustment based on order book imbalance
/// - **Depth-Based**: Size adjustment based on market depth
/// - **Parameter Calibration**: Tools for γ and k estimation
pub mod strategy;

/// Risk management module for comprehensive risk control.
///
/// Provides:
/// - **Position Limits**: Maximum inventory size controls
/// - **Notional Limits**: Maximum value at risk
/// - **Circuit Breakers**: Automatic trading halts on adverse conditions
/// - **Drawdown Tracking**: Peak-to-trough monitoring
/// - **Alert System**: Configurable alerts with multiple handlers
/// - **Portfolio Risk**: Correlation matrix and multi-asset VaR
/// - **Hedge Calculator**: Cross-asset hedging ratios
pub mod risk;

/// Common types and error definitions.
///
/// Provides:
/// - Decimal helper functions (ln, sqrt, powi)
/// - Error types with thiserror
/// - Primitive type aliases
pub mod types;

/// Analytics module for market microstructure analysis.
///
/// Provides:
/// - **Order Flow Analysis**: Trade flow imbalance and toxicity
/// - **VPIN Calculator**: Volume-synchronized probability of informed trading
/// - **Order Intensity**: Fill rate estimation for parameter calibration
/// - **Live Metrics**: Real-time operational metrics with atomic counters
/// - **Prometheus Export**: Optional metrics server with Grafana dashboard
pub mod analytics;

/// Execution module for exchange connectivity and order management.
///
/// Provides:
/// - **Exchange Connector**: Abstract trait for any exchange
/// - **Order Manager**: Order lifecycle with state tracking
/// - **Latency Tracking**: Histogram-based measurement
/// - **Mock Connector**: Testing without real connectivity
pub mod execution;

/// Backtesting module for strategy validation on historical data.
///
/// Provides:
/// - **Event-Driven Engine**: Tick-by-tick simulation
/// - **Data Sources**: Ticks and OHLCV bars
/// - **Fill Models**: Immediate, queue position, probabilistic, market impact
/// - **Performance Metrics**: Sharpe, Sortino, Calmar, profit factor
/// - **Slippage Models**: Fixed, percentage, volatility-based
pub mod backtest;

/// Real-time market data feeds module.
///
/// This module is only available when the `data-feeds` feature is enabled.
/// It provides:
/// - **MarketDataFeed Trait**: Abstract interface for data providers
/// - **Price Updates**: Real-time underlying price streaming
/// - **Trade Feed**: Trade events for order flow analysis
/// - **IV Surface**: Implied volatility surface updates
/// - **Mock Feed**: Testing without real connectivity
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.3", features = ["data-feeds"] }
/// ```
#[cfg(feature = "data-feeds")]
pub mod data_feeds;

/// Prelude module for convenient imports.
///
/// Import all commonly used types with:
/// ```rust
/// use market_maker_rs::prelude::*;
/// ```
pub mod prelude;

/// OptionStratLib integration module for options market making.
///
/// This module is only available when the `options` feature is enabled.
/// It provides:
/// - Greeks calculation and portfolio aggregation
/// - Options pricing adapter
/// - Time conversion utilities
/// - Moneyness calculations
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.2", features = ["options"] }
/// ```
#[cfg(feature = "options")]
pub mod options;

/// Option-Chain-OrderBook integration module.
///
/// This module is only available when the `chain` feature is enabled.
/// It provides:
/// - Multi-strike quoting across option chains
/// - Chain-level Greeks aggregation
/// - Chain risk management
/// - ATM strike detection
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.3", features = ["chain"] }
/// ```
#[cfg(feature = "chain")]
pub mod chain;

/// REST/WebSocket API module for market maker control and monitoring.
///
/// This module is only available when the `api` feature is enabled.
/// It provides:
/// - **REST Endpoints**: Status, configuration, quotes, positions, Greeks, P&L
/// - **OpenAPI Documentation**: Auto-generated API docs with utoipa
/// - **CORS Support**: Configurable cross-origin resource sharing
/// - **State Management**: Thread-safe shared state for the market maker
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.3", features = ["api"] }
/// ```
#[cfg(feature = "api")]
pub mod api;
