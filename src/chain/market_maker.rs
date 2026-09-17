//! Chain market maker implementation.
//!
//! This module provides the `ChainMarketMaker` for multi-strike quoting
//! across an entire option chain.

use std::sync::Arc;

use option_chain_orderbook::orderbook::ExpirationOrderBook;
use optionstratlib::OptionStyle;
use rust_decimal_macros::dec;

use crate::Decimal;
use crate::chain::risk::{ChainRiskLimits, ChainRiskManager};
use crate::options::greeks::PortfolioGreeks;
use crate::options::market_maker::{HedgeOrder, OptionsMarketMakerConfig};
use crate::types::error::MMResult;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for chain market maker.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainMarketMakerConfig {
    /// Base spread in basis points.
    pub base_spread_bps: u64,
    /// ATM spread multiplier (spreads are wider at ATM due to gamma).
    pub atm_spread_multiplier: Decimal,
    /// OTM spread multiplier.
    pub otm_spread_multiplier: Decimal,
    /// Maximum delta per strike.
    pub max_delta_per_strike: Decimal,
    /// Maximum total chain delta.
    pub max_chain_delta: Decimal,
    /// Quote size base.
    pub base_quote_size: u64,
    /// Contract multiplier.
    pub contract_multiplier: Decimal,
    /// ATM tolerance for strike detection (as percentage).
    pub atm_tolerance_pct: Decimal,
    /// Minimum spread in basis points.
    pub min_spread_bps: u64,
    /// Maximum spread in basis points.
    pub max_spread_bps: u64,
}

impl Default for ChainMarketMakerConfig {
    fn default() -> Self {
        Self {
            base_spread_bps: 200, // 2%
            atm_spread_multiplier: dec!(1.5),
            otm_spread_multiplier: dec!(1.0),
            max_delta_per_strike: dec!(50.0),
            max_chain_delta: dec!(500.0),
            base_quote_size: 10,
            contract_multiplier: dec!(100.0),
            atm_tolerance_pct: dec!(0.02), // 2%
            min_spread_bps: 50,            // 0.5%
            max_spread_bps: 1000,          // 10%
        }
    }
}

impl ChainMarketMakerConfig {
    /// Creates a new configuration with custom base spread.
    #[must_use]
    pub fn with_base_spread(base_spread_bps: u64) -> Self {
        Self {
            base_spread_bps,
            ..Default::default()
        }
    }

    /// Converts the options market maker config for individual options.
    #[must_use]
    pub fn to_options_config(&self) -> OptionsMarketMakerConfig {
        OptionsMarketMakerConfig {
            base_spread_pct: Decimal::from(self.base_spread_bps) / dec!(10000.0),
            min_spread_pct: Decimal::from(self.min_spread_bps) / dec!(10000.0),
            max_spread_pct: Decimal::from(self.max_spread_bps) / dec!(10000.0),
            gamma_spread_multiplier: self.atm_spread_multiplier,
            vega_spread_multiplier: dec!(0.5),
            theta_spread_multiplier: dec!(0.1),
            atm_tolerance: self.atm_tolerance_pct,
            contract_multiplier: self.contract_multiplier,
            put_call_skew_factor: dec!(1.0),
        }
    }
}

/// Quote update for a single option.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainQuoteUpdate {
    /// Strike price.
    pub strike: u64,
    /// Option style (Call or Put).
    pub style: OptionStyle,
    /// Bid price.
    pub bid_price: u64,
    /// Ask price.
    pub ask_price: u64,
    /// Bid size.
    pub bid_size: u64,
    /// Ask size.
    pub ask_size: u64,
    /// Theoretical value.
    pub theo: u64,
    /// Spread in basis points.
    pub spread_bps: u64,
}

impl ChainQuoteUpdate {
    /// Creates a new quote update.
    #[must_use]
    pub fn new(
        strike: u64,
        style: OptionStyle,
        bid_price: u64,
        ask_price: u64,
        bid_size: u64,
        ask_size: u64,
        theo: u64,
    ) -> Self {
        let spread_bps = ((ask_price - bid_price) * 10000)
            .checked_div(theo)
            .unwrap_or(0);

        Self {
            strike,
            style,
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            theo,
            spread_bps,
        }
    }

    /// Returns the mid price.
    #[must_use]
    pub fn mid_price(&self) -> u64 {
        (self.bid_price + self.ask_price) / 2
    }

    /// Returns the spread.
    #[must_use]
    pub fn spread(&self) -> u64 {
        self.ask_price - self.bid_price
    }
}

/// Risk status for the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RiskStatus {
    /// All risk limits are within bounds.
    Normal,
    /// Approaching risk limits (warning).
    Warning,
    /// Risk limits breached (should stop quoting).
    Breached,
    /// Hedge required.
    HedgeRequired,
}

impl RiskStatus {
    /// Returns true if quoting should continue.
    #[must_use]
    pub fn can_quote(&self) -> bool {
        matches!(self, RiskStatus::Normal | RiskStatus::Warning)
    }

    /// Returns true if immediate action is required.
    #[must_use]
    pub fn requires_action(&self) -> bool {
        matches!(self, RiskStatus::Breached | RiskStatus::HedgeRequired)
    }
}

/// Market maker for an entire option chain.
///
/// Provides multi-strike quoting, chain-level Greeks aggregation,
/// and risk management for an option expiration.
pub struct ChainMarketMaker {
    /// The option chain.
    chain: Arc<ExpirationOrderBook>,
    /// Configuration.
    config: ChainMarketMakerConfig,
    /// Risk manager.
    risk_manager: ChainRiskManager,
    /// Underlying symbol.
    underlying_symbol: String,
}

impl ChainMarketMaker {
    /// Creates a new chain market maker.
    ///
    /// # Arguments
    ///
    /// * `chain` - The expiration order book
    /// * `config` - Market maker configuration
    #[must_use]
    pub fn new(chain: Arc<ExpirationOrderBook>, config: ChainMarketMakerConfig) -> Self {
        let underlying_symbol = chain.underlying().to_string();
        let risk_limits = ChainRiskLimits {
            max_chain_delta: config.max_chain_delta,
            max_delta_per_strike: config.max_delta_per_strike,
            ..Default::default()
        };

        Self {
            chain,
            risk_manager: ChainRiskManager::new(
                &underlying_symbol,
                risk_limits,
                config.contract_multiplier,
            ),
            config,
            underlying_symbol,
        }
    }

    /// Creates a new chain market maker with default configuration.
    #[must_use]
    pub fn with_defaults(chain: Arc<ExpirationOrderBook>) -> Self {
        Self::new(chain, ChainMarketMakerConfig::default())
    }

    /// Returns a reference to the chain.
    #[must_use]
    pub fn chain(&self) -> &ExpirationOrderBook {
        &self.chain
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ChainMarketMakerConfig {
        &self.config
    }

    /// Returns a reference to the risk manager.
    #[must_use]
    pub fn risk_manager(&self) -> &ChainRiskManager {
        &self.risk_manager
    }

    /// Returns a mutable reference to the risk manager.
    pub fn risk_manager_mut(&mut self) -> &mut ChainRiskManager {
        &mut self.risk_manager
    }

    /// Returns the underlying symbol.
    #[must_use]
    pub fn underlying_symbol(&self) -> &str {
        &self.underlying_symbol
    }

    /// Gets the ATM strike for the given spot price.
    ///
    /// # Arguments
    ///
    /// * `spot_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// The ATM strike price, or an error if no strikes exist.
    pub fn get_atm_strike(&self, spot_price: u64) -> MMResult<u64> {
        self.chain
            .atm_strike(spot_price)
            .map_err(|e| crate::types::error::MMError::InvalidMarketState(e.to_string()))
    }

    /// Calculates the spread multiplier based on moneyness.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `spot_price` - Current underlying price
    /// * `is_atm` - Whether this is the ATM strike
    #[must_use]
    pub fn get_spread_multiplier(&self, strike: u64, spot_price: u64, is_atm: bool) -> Decimal {
        if is_atm {
            self.config.atm_spread_multiplier
        } else {
            // Calculate moneyness
            let moneyness = if spot_price > 0 {
                Decimal::from(strike) / Decimal::from(spot_price)
            } else {
                dec!(1.0)
            };

            // Further OTM options get slightly wider spreads
            let otm_factor = (moneyness - dec!(1.0)).abs();
            self.config.otm_spread_multiplier + otm_factor * dec!(0.5)
        }
    }

    /// Calculates the quote size based on strike characteristics.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `spot_price` - Current underlying price
    /// * `is_atm` - Whether this is the ATM strike
    #[must_use]
    pub fn calculate_quote_size(&self, _strike: u64, _spot_price: u64, is_atm: bool) -> u64 {
        if is_atm {
            // ATM options typically have more liquidity
            self.config.base_quote_size * 2
        } else {
            self.config.base_quote_size
        }
    }

    /// Generates a quote for a single option.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `style` - Option style (Call or Put)
    /// * `spot_price` - Current underlying price
    /// * `theo_price` - Theoretical price
    /// * `is_atm` - Whether this is the ATM strike
    #[must_use]
    pub fn generate_quote(
        &self,
        strike: u64,
        style: OptionStyle,
        spot_price: u64,
        theo_price: u64,
        is_atm: bool,
    ) -> ChainQuoteUpdate {
        let spread_mult = self.get_spread_multiplier(strike, spot_price, is_atm);

        // Calculate spread in basis points
        let base_spread_bps = Decimal::from(self.config.base_spread_bps);
        let adjusted_spread_bps = base_spread_bps * spread_mult;

        // Clamp spread
        let clamped_spread_bps = if adjusted_spread_bps < Decimal::from(self.config.min_spread_bps)
        {
            Decimal::from(self.config.min_spread_bps)
        } else if adjusted_spread_bps > Decimal::from(self.config.max_spread_bps) {
            Decimal::from(self.config.max_spread_bps)
        } else {
            adjusted_spread_bps
        };

        // Calculate half spread
        let half_spread = (Decimal::from(theo_price) * clamped_spread_bps / dec!(20000.0))
            .to_string()
            .parse::<f64>()
            .map(|f| f as u64)
            .unwrap_or(1);

        let bid_price = theo_price.saturating_sub(half_spread);
        let ask_price = theo_price.saturating_add(half_spread);

        let quote_size = self.calculate_quote_size(strike, spot_price, is_atm);

        ChainQuoteUpdate::new(
            strike, style, bid_price, ask_price, quote_size, quote_size, theo_price,
        )
    }

    /// Refreshes all quotes based on current underlying price.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying (in price units)
    ///
    /// # Returns
    ///
    /// A vector of quote updates for all strikes and styles.
    pub fn refresh_all_quotes(&self, underlying_price: u64) -> MMResult<Vec<ChainQuoteUpdate>> {
        let atm_strike = self.get_atm_strike(underlying_price).ok();
        let strikes = self.chain.strike_prices();
        let mut updates = Vec::with_capacity(strikes.len() * 2);

        for strike in strikes {
            let is_atm = atm_strike == Some(strike);

            // Generate quotes for both call and put
            for style in [OptionStyle::Call, OptionStyle::Put] {
                // For now, use a simple theoretical price calculation
                // In production, this would use the volatility surface
                let theo_price = self.estimate_theo_price(strike, underlying_price, style);

                let quote =
                    self.generate_quote(strike, style, underlying_price, theo_price, is_atm);
                updates.push(quote);
            }
        }

        Ok(updates)
    }

    /// Estimates theoretical price (simplified).
    ///
    /// In production, this would use the volatility surface and proper pricing.
    fn estimate_theo_price(&self, strike: u64, spot: u64, style: OptionStyle) -> u64 {
        let intrinsic = match style {
            OptionStyle::Call => spot.saturating_sub(strike),
            OptionStyle::Put => strike.saturating_sub(spot),
        };

        // Add some time value (simplified)
        let time_value = spot / 50; // ~2% time value

        intrinsic + time_value
    }

    /// Gets the aggregated Greeks for the entire chain.
    #[must_use]
    pub fn get_chain_greeks(&self) -> &PortfolioGreeks {
        self.risk_manager.current_greeks()
    }

    /// Checks the chain-level risk status.
    #[must_use]
    pub fn check_chain_risk(&self) -> RiskStatus {
        if self.risk_manager.is_risk_breached() {
            RiskStatus::Breached
        } else if self.risk_manager.should_hedge() {
            RiskStatus::HedgeRequired
        } else {
            // Check if approaching limits (80% threshold)
            let greeks = self.risk_manager.current_greeks();
            let limits = self.risk_manager.limits();

            let delta_util = greeks.delta.abs() / limits.max_chain_delta;
            let gamma_util = greeks.gamma.abs() / limits.max_chain_gamma;
            let vega_util = greeks.vega.abs() / limits.max_chain_vega;

            if delta_util > dec!(0.8) || gamma_util > dec!(0.8) || vega_util > dec!(0.8) {
                RiskStatus::Warning
            } else {
                RiskStatus::Normal
            }
        }
    }

    /// Calculates hedge orders to neutralize chain delta.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying
    #[must_use]
    pub fn calculate_hedge(&self, underlying_price: Decimal) -> Vec<HedgeOrder> {
        self.risk_manager.calculate_hedge(underlying_price)
    }

    /// Returns the number of strikes in the chain.
    #[must_use]
    pub fn strike_count(&self) -> usize {
        self.chain.strike_count()
    }

    /// Returns all strike prices (sorted).
    pub fn strike_prices(&self) -> Vec<u64> {
        self.chain.strike_prices()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optionstratlib::ExpirationDate;
    use optionstratlib::prelude::pos_or_panic;

    fn create_test_chain() -> Arc<ExpirationOrderBook> {
        let exp = ExpirationDate::Days(pos_or_panic!(30.0));
        let chain = ExpirationOrderBook::new("BTC", exp);

        // Add some strikes
        chain.get_or_create_strike(45000);
        chain.get_or_create_strike(50000);
        chain.get_or_create_strike(55000);

        Arc::new(chain)
    }

    #[test]
    fn test_chain_market_maker_config_default() {
        let config = ChainMarketMakerConfig::default();
        assert_eq!(config.base_spread_bps, 200);
        assert_eq!(config.atm_spread_multiplier, dec!(1.5));
    }

    #[test]
    fn test_chain_market_maker_new() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        assert_eq!(mm.underlying_symbol(), "BTC");
        assert_eq!(mm.strike_count(), 3);
    }

    #[test]
    fn test_get_atm_strike() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let atm = mm.get_atm_strike(50000).unwrap();
        assert_eq!(atm, 50000);

        let atm = mm.get_atm_strike(48000).unwrap();
        assert_eq!(atm, 50000); // Closest strike
    }

    #[test]
    fn test_get_spread_multiplier() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        // ATM should have higher multiplier
        let atm_mult = mm.get_spread_multiplier(50000, 50000, true);
        let otm_mult = mm.get_spread_multiplier(55000, 50000, false);

        assert!(atm_mult > otm_mult);
    }

    #[test]
    fn test_generate_quote() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let quote = mm.generate_quote(50000, OptionStyle::Call, 50000, 1000, true);

        assert_eq!(quote.strike, 50000);
        assert_eq!(quote.style, OptionStyle::Call);
        assert!(quote.bid_price < quote.ask_price);
        assert!(quote.bid_size > 0);
    }

    #[test]
    fn test_refresh_all_quotes() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let quotes = mm.refresh_all_quotes(50000).unwrap();

        // 3 strikes * 2 styles = 6 quotes
        assert_eq!(quotes.len(), 6);

        // Check all quotes have valid prices
        for quote in &quotes {
            assert!(quote.bid_price < quote.ask_price);
        }
    }

    #[test]
    fn test_check_chain_risk_normal() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let status = mm.check_chain_risk();
        assert_eq!(status, RiskStatus::Normal);
        assert!(status.can_quote());
    }

    #[test]
    fn test_risk_status_can_quote() {
        assert!(RiskStatus::Normal.can_quote());
        assert!(RiskStatus::Warning.can_quote());
        assert!(!RiskStatus::Breached.can_quote());
        assert!(!RiskStatus::HedgeRequired.can_quote());
    }

    #[test]
    fn test_chain_quote_update() {
        let quote = ChainQuoteUpdate::new(50000, OptionStyle::Call, 980, 1020, 10, 10, 1000);

        assert_eq!(quote.mid_price(), 1000);
        assert_eq!(quote.spread(), 40);
        assert_eq!(quote.spread_bps, 400); // 4%
    }
}
