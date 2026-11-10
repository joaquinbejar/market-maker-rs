//! # Market Making Library
//!
//! A comprehensive Rust library implementing quantitative market making strategies based on
//! the Avellaneda-Stoikov model and extensions. It provides components for building
//! automated market making systems, causal backtests of them, and a deterministic kernel
//! that an external reinforcement learning trainer can step.
//!
//! A live deployment still needs its own exchange connectivity, monitoring and validation:
//! the results in this repository are simulations and replays, not evidence of profit.
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
//! - **GLFT**: Guéant-Lehalle-Fernandez-Tapia optimal quotes with a bounded inventory (exact solution and closed form)
//! - **Grid Trading**: Multi-level order placement with geometric/arithmetic spacing
//! - **Adaptive Spread**: Dynamic spread adjustment based on order book imbalance
//! - **Depth-Based Offering**: Size adjustment based on market depth
//! - **Signal Quotes**: Causal, bounded quote adjustment from order book imbalance (`strategy::signal_quote`)
//! - **Walk-Forward Calibration**: Chronological splits and causal calibration, with venues that close (`strategy::walk_forward`)
//! - **Joint Quotes**: Closed-form quotes for correlated assets (feature `multi-underlying`)
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
//! - **Micro-Price**: Stoikov's micro-price (`analytics::micro_price`)
//!
//! ### Backtesting
//!
//! - **Event-Driven Engine**: Tick-by-tick simulation
//! - **Fill Models**: Immediate, queue position, probabilistic, market impact
//! - **Performance Metrics**: Sharpe, Sortino, Calmar, max drawdown, profit factor
//! - **Slippage Models**: Fixed, percentage, volatility-based
//! - **Evaluation Harness**: Experiments that compare strategies under one set of rules, metric
//!   attribution, fee scenarios and block bootstrap intervals (`backtest::evaluation`)
//! - **Order Flow Simulators**: A seedable queue-reactive simulator, with an opt-in reference that
//!   reacts to the agent (`backtest::queue_reactive`), and the QRH-I queue-reactive Hawkes model
//!   with seasonality (`backtest::hawkes`)
//! - **Hyperliquid Replay**: Decoders for its streams and a rebuilt book sampled on a time grid
//!   (`backtest::hyperliquid`, feature `serde`)
//! - **Derivative Contracts**: Several dated futures against one ledger, perpetual funding,
//!   contract lifecycle and initial margin before an order (features `margin` and `instruments`)
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
//! ### Reinforcement Learning Support (Features: `kernel`, `rl`)
//!
//! - **Financial Kernel**: The backtest loop as a resumable, `Copy` state machine with a canonical
//!   encoding, exactly equivalent to the engine inside its documented subset
//! - **Versioned Spaces**: Observations, actions, masks and a `Decimal` reward that reconciles with
//!   the evaluation harness
//! - **Quote Control**: Cancel-only actions and quotes that expire at the venue, exact in venue time
//! - **Hedging**: A hedge lane in a second instrument, and a balance gate through `margin-rs`
//! - **Residual Policies**: Corrections on top of a classical quote
//! - **Python Bridge**: `bridges/python`, a nested crate outside this package, for external trainers
//!
//! ### Options, Surfaces and Settlement (Feature: `options`)
//!
//! - **Pricing**: European option valuation (`options::pricing`)
//! - **Volatility Surfaces**: SVI slices and surfaces, calibration and upstream surfaces (`options::surface`)
//! - **Settlement**: Option positions held and settled in a `margin-rs` ledger (`options::settlement`, with `margin`)
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
//! - `options`: Options pricing, surfaces, Greeks, settlement and market making (feature: `options`)
//! - `chain`: Option chain integration and multi-strike quoting (feature: `chain`)
//! - `api`: REST/WebSocket API layer (feature: `api`)
//! - `persistence`: Data persistence layer (feature: `persistence`)
//! - `multi_underlying`: Multi-asset management and closed-form joint quotes (feature: `multi-underlying`; with `margin`, each quote through a `margin-rs` gate)
//! - `events`: Event broadcasting system (feature: `events`)
//! - `data_feeds`: Real-time market data feeds (feature: `data-feeds`)
//! - `l3`: Order by order book reconstruction with a quality manifest and
//!   declared queue limits (feature: `l3`)
//! - `instruments`: Contract specifications shared with `margin-rs` and
//!   `arbitrage-rs`, and their conversion into `margin-rs` instruments
//!   (feature: `instruments`, with `margin` for the conversion), and the net edge
//!   of a spot and a dated future, a calendar spread under an assumed exit and a
//!   roll plan (feature: `arbitrage`)
//! - `rl`: Versioned observation, action and reward spaces over the kernel (feature: `rl`)
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
//! ## Shared Types
//!
//! The order direction and the option style are the shared enums of the
//! `financial_types` crate, also used by `margin-rs`, `arbitrage-rs` and
//! OptionStratLib, and re-exported from the [`prelude`]:
//!
//! - `Action` (`Buy`/`Sell`) for every order, fill and query side. It also
//!   has an `Other` variant, which this crate does not support: methods that
//!   take or process an order, a fill or a side return
//!   `MMError::UnsupportedAction` for it, before changing anything; a
//!   pretrade check rejects the order; and a pretrade policy that receives a
//!   fill with that side also latches a fault and rejects every later order
//!   until it is reset.
//! - `OptionStyle` (`Call`/`Put`).
//!
//! `financial_types` derives `serde` unconditionally, so `serde` is compiled
//! even with no features enabled; the `serde` feature still gates this
//! crate's own serialization support.
//!
//! ## Feature Flags
//!
//! - `prometheus`: Enable Prometheus metrics export (adds `prometheus`, `hyper`, `tokio` dependencies)
//! - `serde`: Enable serialization/deserialization for all types
//! - `options`: Enable OptionStratLib integration for options pricing and Greeks calculation
//! - `chain`: Enable Option-Chain-OrderBook integration (includes `options`)
//! - `api`: Enable REST/WebSocket API layer with OpenAPI documentation
//! - `persistence`: Enable persistence layer for market maker data
//! - `multi-underlying`: Enable multi-asset management with correlation tracking
//! - `events`: Enable event broadcasting system for real-time updates
//! - `margin`: Use `margin-rs` in the backtest: its exact ledger as the fill
//!   accountant (`backtest::MarginLedgerAccountant`) and its risk gate as the
//!   pretrade policy (`backtest::MarginAccount`)
//! - `data-feeds`: Enable real-time market data feed abstractions
//! - `l3`: Enable order by order (L3) book reconstruction: snapshot and
//!   sequenced messages, fail closed on gaps and inconsistencies, a quality
//!   manifest, and volume ahead under declared hypotheses (no dependencies)
//! - `instruments`: Contract specifications from `financial_types`
//!   (`instruments` module): validated specifications, exact money and
//!   dated venue snapshots, and orders and fills checked against them
//!   (`instruments::orders`); with `margin` too, their conversion into
//!   `margin-rs` instruments and executions (`instruments::margin`), and several
//!   dated futures in one backtest run against one ledger
//!   (`backtest::contracts`). Both
//!   together need
//!   `margin-rs` 0.1.1 or later: Cargo does not upgrade a lockfile at 0.1.0
//!   by itself, so run `cargo update -p margin-rs`
//! - `arbitrage`: The net edge of a spot and a linear dated future
//!   (`instruments::basis`), through `arbitrage-rs`, whose types stay out of this
//!   crate's public API, a calendar spread under an assumed exit
//!   (`instruments::calendar`), and a two contract roll plan (`instruments::roll`),
//!   checked against a ledger and run in a backtest (`instruments::roll::run`) with
//!   `margin` too; implies `instruments`
//! - `kernel`: The financial environment kernel (`backtest::kernel`): the
//!   backtest engine's loop as a resumable, `Copy` state machine with a
//!   canonical encoding, an agent-safe observation and an admission mask,
//!   exactly equivalent to the engine inside its documented subset. Meant to
//!   be stepped by an external RL trainer through a bridge crate. No extra
//!   dependencies.
//! - `rl`: Versioned observation, action and reward spaces over the kernel (`rl`),
//!   for an external RL trainer or engine: features a real agent could know, masks
//!   apart from the hard limits, and a `Decimal` reward that reconciles with
//!   `backtest::evaluation`, with the `f32` derived from it. Implies `kernel`.
//!
//! ## Research Status
//!
//! The 0.7.0 cycle added an out-of-sample research program under `bridges/python/experiments`,
//! which is not part of the published crate. The studies fix their designs in advance, disclose
//! any exception, such as a holdout that had been read before, and report negative results. So
//! far no learned policy has been shown to beat the best classical comparator out of sample, so
//! the crate recommends none. Most of these studies are synthetic, and some remain open: treat
//! their conclusions as limited to the setting each report states.
//!
//! The consolidated report, with the evidence and product matrices and the open gaps, is
//! `doc/RESEARCH_CONSOLIDATION.md` in the repository.
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
//! portfolio.add(&greeks, dec!(10.0)).unwrap(); // 10 contracts
//!
//! // Check delta neutrality
//! let shares_to_hedge = portfolio.shares_to_hedge(dec!(100.0)).unwrap();
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
//! let decision = risk_manager.check_order(&option_greeks, dec!(10.0)).unwrap();
//! match decision {
//!     OrderDecision::Allowed => { /* proceed */ },
//!     OrderDecision::Scaled { new_size, .. } => { /* use scaled size */ },
//!     OrderDecision::Rejected { reason } => { /* reject */ },
//! }
//!
//! // Check if hedging is needed
//! if risk_manager.needs_hedge().unwrap() {
//!     let hedge = risk_manager.calculate_hedge_order(underlying_price).unwrap();
//! }
//! ```
//!
//! ### Option Chain Market Making (Feature: `chain`)
//!
//! ```rust,ignore
//! use market_maker_rs::chain::{ChainMarketMaker, ChainMarketMakerConfig, ChainPricing, ChainQuoteOutcome};
//! use market_maker_rs::options::{Dynamics, FlatSurface, SurfacePricer, tau_from_days};
//! use option_chain_orderbook::orderbook::ExpirationOrderBook;
//! use std::sync::Arc;
//!
//! // Create chain market maker
//! let chain = Arc::new(ExpirationOrderBook::new("BTC", expiration));
//! let config = ChainMarketMakerConfig::default();
//! let mm = ChainMarketMaker::new(chain, config);
//!
//! // Valuation inputs for this expiration (surface, dynamics, rates, tau)
//! let pricer = SurfacePricer::new(Arc::new(FlatSurface::new(dec!(0.6))?), Dynamics::StickyMoneyness)?;
//! let pricing = ChainPricing::new(Arc::new(pricer), tau_from_days(dec!(30))?, dec!(0), dec!(0))?;
//!
//! // Refresh all quotes across the chain: one outcome per contract
//! for outcome in mm.refresh_all_quotes(underlying_price, &pricing)? {
//!     match outcome {
//!         ChainQuoteOutcome::Quote(quote) => { /* publish */ }
//!         ChainQuoteOutcome::NotQuoted { reason, .. } => { /* withdraw any resting quote */ }
//!         _ => {}
//!     }
//! }
//!
//! // Check chain risk status. While it does not allow quoting, the refresh
//! // above refuses with `MMError::QuotingHalted`.
//! let status = mm.check_chain_risk().unwrap();
//! if !status.can_quote() {
//!     // Withdraw resting quotes, and hedge or reconcile
//! }
//! ```
//!
//! ### Multi-Underlying Management (Feature: `multi-underlying`)
//!
//! ```rust,ignore
//! use market_maker_rs::multi_underlying::{
//!     MultiUnderlyingManager, UnderlyingConfig, CapitalAllocationStrategy,
//! };
//! use market_maker_rs::dec;
//!
//! // Create manager with $1M capital
//! let mut manager = MultiUnderlyingManager::new(dec!(1_000_000))
//!     .with_allocation_strategy(CapitalAllocationStrategy::RiskParity)
//!     .with_max_total_delta(dec!(50000));
//!
//! // Add underlyings with target weights
//! manager.add_underlying(UnderlyingConfig::new("BTC", dec!(0.40))).unwrap();
//! manager.add_underlying(UnderlyingConfig::new("ETH", dec!(0.30))).unwrap();
//!
//! // Set correlations
//! manager.set_correlation("BTC", "ETH", dec!(0.85));
//!
//! // Update market data
//! manager.update_price("BTC", dec!(45000));
//! manager.update_greeks("BTC", dec!(5.5), dec!(0.02), dec!(1500));
//!
//! // Get unified risk view
//! let risk = manager.get_unified_risk();
//! println!("Total delta: {}", risk.greeks.total_dollar_delta);
//!
//! // Get cross-asset hedge suggestions
//! let hedges = manager.get_cross_asset_hedges();
//! ```
//!
//! ### Event Broadcasting (Feature: `events`)
//!
//! ```rust,ignore
//! use market_maker_rs::events::{
//!     EventBroadcaster, EventBroadcasterConfig, EventFilter, EventType,
//!     MarketMakerEvent, Action,
//! };
//! use std::sync::Arc;
//!
//! // Create broadcaster
//! let config = EventBroadcasterConfig::default();
//! let broadcaster = Arc::new(EventBroadcaster::new(config));
//!
//! // Subscribe to all events
//! let mut all_rx = broadcaster.subscribe();
//!
//! // Subscribe with filter (fills only)
//! let filter = EventFilter::new()
//!     .with_event_types([EventType::OrderFilled])
//!     .with_symbols(["BTC".to_string()]);
//! let mut filtered_rx = broadcaster.subscribe_filtered(filter);
//!
//! // Broadcast an event
//! broadcaster.broadcast(MarketMakerEvent::OrderFilled {
//!     order_id: "ORD-001".to_string(),
//!     symbol: "BTC".to_string(),
//!     instrument: "BTC-PERP".to_string(),
//!     side: Action::Buy,
//!     quantity: 10,
//!     price: 50000,
//!     fee: 5,
//!     edge: 100,
//!     timestamp: 1234567890,
//! }).await;
//!
//! // Get history for reconnection
//! let missed = broadcaster.get_reconnection_history(last_sequence).await;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]
// Zero panics in library code: no unwrap, expect, unchecked indexing or
// panic, and no arithmetic that can overflow or divide by zero. Tests are
// exempt.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::arithmetic_side_effects
    )
)]

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
/// - **GLFT**: Guéant-Lehalle-Fernandez-Tapia quotes with a bounded inventory
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
/// market-maker-rs = { version = "0.5", features = ["data-feeds"] }
/// ```
#[cfg(feature = "data-feeds")]
pub mod data_feeds;

/// Order by order (L3) book reconstruction and queue limits.
///
/// Rebuilds the visible book from a snapshot and a sequenced message
/// stream, fails closed on gaps and inconsistencies, records a quality
/// manifest, and states what the data supports about queue priority.
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.6", features = ["l3"] }
/// ```
#[cfg(feature = "l3")]
pub mod l3;

/// Contract specifications from `financial_types`, and their conversion
/// into `margin-rs` instruments.
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.6", features = ["instruments"] }
/// ```
///
/// The conversion (`instruments::margin`) also needs the `margin` feature.
#[cfg(feature = "instruments")]
pub mod instruments;

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
/// market-maker-rs = { version = "0.5", features = ["options"] }
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
/// market-maker-rs = { version = "0.5", features = ["chain"] }
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
/// market-maker-rs = { version = "0.5", features = ["api"] }
/// ```
#[cfg(feature = "api")]
pub mod api;

/// Persistence layer for market maker data.
///
/// This module is only available when the `persistence` feature is enabled.
/// It provides abstractions for persisting market maker data including:
/// - **Fill Records**: Trade executions with price, quantity, fees
/// - **Position Snapshots**: Point-in-time position state
/// - **Daily P&L**: Aggregated daily profit/loss records
/// - **Configuration**: Persistent settings storage
/// - **Event Logs**: System events and alerts
///
/// The module includes an in-memory implementation for testing.
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.5", features = ["persistence"] }
/// ```
#[cfg(feature = "persistence")]
pub mod persistence;

/// Multi-underlying support module for managing multiple assets simultaneously.
///
/// This module is only available when the `multi-underlying` feature is enabled.
/// It provides:
/// - **Cross-Asset Correlation**: Track correlations between underlyings
/// - **Capital Allocation**: Multiple strategies (equal, risk parity, volatility-weighted)
/// - **Unified Risk View**: Aggregate Greeks and P&L across all positions
/// - **Per-Underlying Limits**: Individual risk limits per asset
/// - **Cross-Asset Hedging**: Suggestions for hedging across correlated assets
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.5", features = ["multi-underlying"] }
/// ```
#[cfg(feature = "multi-underlying")]
pub mod multi_underlying;

/// Event system for broadcasting market maker events.
///
/// This module is only available when the `events` feature is enabled.
/// It provides a structured event handling system for broadcasting
/// market maker events to the frontend and other consumers, with support for
/// event history and reconnection.
///
/// # Features
///
/// - **Event Broadcasting**: Distribute real-time events to multiple consumers
/// - **Event Filtering**: Subscribe to specific event types or symbols
/// - **Event History**: Buffer for reconnection scenarios
/// - **Event Batching**: Aggregate high-frequency updates
/// - **Heartbeat**: Connection keep-alive mechanism
///
/// # Feature Flag
///
/// Enable with:
/// ```toml
/// [dependencies]
/// market-maker-rs = { version = "0.5", features = ["events"] }
/// ```
#[cfg(feature = "events")]
pub mod events;

#[cfg(feature = "rl")]
pub mod rl;
