//! Option-Chain-OrderBook integration module.
//!
//! This module provides integration with the `option-chain-orderbook` library
//! for multi-strike quoting, chain-level risk aggregation, and hierarchical
//! option chain management.
//!
//! # Feature Flag
//!
//! This module is only available when the `chain` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! market-maker-rs = { version = "0.3", features = ["chain"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::chain::{ChainMarketMaker, ChainMarketMakerConfig};
//! use option_chain_orderbook::orderbook::ExpirationOrderBook;
//!
//! let config = ChainMarketMakerConfig::default();
//! let chain = ExpirationOrderBook::new("BTC", expiration);
//! let mm = ChainMarketMaker::new(chain, config);
//!
//! // Valuation inputs for this expiration, then one outcome per contract
//! let pricer = SurfacePricer::new(Arc::new(FlatSurface::new(dec!(0.6))?), Dynamics::StickyMoneyness)?;
//! let pricing = ChainPricing::new(Arc::new(pricer), tau_from_days(dec!(30))?, dec!(0), dec!(0))?;
//! let outcomes = mm.refresh_all_quotes(underlying_price, &pricing)?;
//! ```

/// Chain market maker implementation.
pub mod market_maker;

/// Chain risk management.
pub mod risk;

pub use market_maker::{
    ChainMarketMaker, ChainMarketMakerConfig, ChainPricing, ChainQuoteOutcome, ChainQuoteUpdate,
    RiskStatus,
};
pub use risk::{ChainRiskLimits, ChainRiskManager};
