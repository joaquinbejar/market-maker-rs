//! Options-specific market making functionality.
//!
//! This module provides Greeks-aware quoting, spread adjustment based on
//! option characteristics, and delta hedging logic for options market making.
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::options::{OptionsMarketMaker, OptionsMarketMakerConfig};
//! use market_maker_rs::options::{PortfolioGreeks, PositionGreeks};
//!
//! let config = OptionsMarketMakerConfig::default();
//! let market_maker = OptionsMarketMakerImpl::new(config);
//!
//! // Calculate Greeks-adjusted quotes
//! let (bid, ask) = market_maker.calculate_greeks_adjusted_quotes(
//!     &option,
//!     &portfolio_greeks,
//!     &greeks_limits,
//! )?;
//! ```

use crate::Decimal;
use crate::options::adapter::OptionsAdapter;
use crate::options::greeks::{PortfolioGreeks, PositionGreeks};
use crate::types::error::MMResult;
use optionstratlib::model::option::Options;
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Greeks limits for risk management.
///
/// Defines thresholds for each Greek that trigger spread widening
/// or hedging actions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GreeksLimits {
    /// Maximum absolute delta before hedging is required.
    pub max_delta: Decimal,
    /// Maximum absolute gamma exposure.
    pub max_gamma: Decimal,
    /// Maximum absolute vega exposure.
    pub max_vega: Decimal,
    /// Maximum daily theta decay allowed.
    pub max_theta: Decimal,
    /// Delta threshold that triggers hedge suggestions.
    pub delta_hedge_threshold: Decimal,
}

impl Default for GreeksLimits {
    fn default() -> Self {
        Self {
            max_delta: dec!(100.0),
            max_gamma: dec!(50.0),
            max_vega: dec!(1000.0),
            max_theta: dec!(-500.0),
            delta_hedge_threshold: dec!(10.0),
        }
    }
}

impl GreeksLimits {
    /// Creates new Greeks limits with specified values.
    #[must_use]
    pub fn new(
        max_delta: Decimal,
        max_gamma: Decimal,
        max_vega: Decimal,
        max_theta: Decimal,
        delta_hedge_threshold: Decimal,
    ) -> Self {
        Self {
            max_delta,
            max_gamma,
            max_vega,
            max_theta,
            delta_hedge_threshold,
        }
    }

    /// Checks if delta exceeds the hedge threshold.
    #[must_use]
    pub fn should_hedge_delta(&self, portfolio: &PortfolioGreeks) -> bool {
        portfolio.delta.abs() > self.delta_hedge_threshold
    }

    /// Checks if any Greek limit is breached.
    #[must_use]
    pub fn is_any_limit_breached(&self, portfolio: &PortfolioGreeks) -> bool {
        portfolio.delta.abs() > self.max_delta
            || portfolio.gamma.abs() > self.max_gamma
            || portfolio.vega.abs() > self.max_vega
            || portfolio.theta < self.max_theta
    }
}

/// Hedge order suggestion for delta neutralization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HedgeOrder {
    /// Symbol to trade for hedging.
    pub symbol: String,
    /// Quantity to trade (positive = buy, negative = sell).
    pub quantity: Decimal,
    /// Suggested price for the hedge order.
    pub price: Decimal,
    /// Whether this is an underlying or option hedge.
    pub hedge_type: HedgeType,
    /// Expected delta reduction from this hedge.
    pub delta_impact: Decimal,
}

impl HedgeOrder {
    /// Creates a new hedge order.
    #[must_use]
    pub fn new(
        symbol: String,
        quantity: Decimal,
        price: Decimal,
        hedge_type: HedgeType,
        delta_impact: Decimal,
    ) -> Self {
        Self {
            symbol,
            quantity,
            price,
            hedge_type,
            delta_impact,
        }
    }

    /// Creates an underlying hedge order.
    #[must_use]
    pub fn underlying(symbol: String, quantity: Decimal, price: Decimal) -> Self {
        let delta_impact = -quantity; // Underlying has delta of 1
        Self {
            symbol,
            quantity,
            price,
            hedge_type: HedgeType::Underlying,
            delta_impact,
        }
    }
}

/// Type of hedge instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HedgeType {
    /// Hedge using the underlying asset.
    Underlying,
    /// Hedge using another option.
    Option,
    /// Hedge using futures.
    Futures,
}

/// Configuration for options market making.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptionsMarketMakerConfig {
    /// Base spread in percentage (e.g., 0.01 for 1%).
    pub base_spread_pct: Decimal,
    /// Minimum spread in percentage.
    pub min_spread_pct: Decimal,
    /// Maximum spread in percentage.
    pub max_spread_pct: Decimal,
    /// Gamma multiplier for spread adjustment.
    /// Higher gamma options get wider spreads.
    pub gamma_spread_multiplier: Decimal,
    /// Vega multiplier for spread adjustment.
    /// Higher vega options get wider spreads in volatile markets.
    pub vega_spread_multiplier: Decimal,
    /// Time decay factor for spread adjustment.
    /// Options near expiration may have wider spreads.
    pub theta_spread_multiplier: Decimal,
    /// ATM tolerance for gamma-based spread widening.
    pub atm_tolerance: Decimal,
    /// Contract multiplier for dollar calculations.
    pub contract_multiplier: Decimal,
    /// Skew adjustment factor for puts vs calls.
    pub put_call_skew_factor: Decimal,
}

impl Default for OptionsMarketMakerConfig {
    fn default() -> Self {
        Self {
            base_spread_pct: dec!(0.02), // 2% base spread
            min_spread_pct: dec!(0.005), // 0.5% minimum
            max_spread_pct: dec!(0.10),  // 10% maximum
            gamma_spread_multiplier: dec!(2.0),
            vega_spread_multiplier: dec!(0.5),
            theta_spread_multiplier: dec!(0.1),
            atm_tolerance: dec!(0.02),        // 2% ATM tolerance
            contract_multiplier: dec!(100.0), // Standard equity options
            put_call_skew_factor: dec!(1.0),  // No skew by default
        }
    }
}

impl OptionsMarketMakerConfig {
    /// Creates a new configuration with custom base spread.
    #[must_use]
    pub fn with_base_spread(base_spread_pct: Decimal) -> Self {
        Self {
            base_spread_pct,
            ..Default::default()
        }
    }
}

/// Trait for options-specific market making.
///
/// Provides methods for calculating Greeks-aware quotes, spread adjustments,
/// and delta hedging suggestions.
pub trait OptionsMarketMaker {
    /// Calculates quotes adjusted for portfolio Greeks.
    ///
    /// # Arguments
    ///
    /// * `option` - The option to quote
    /// * `portfolio_greeks` - Current portfolio Greek exposure
    /// * `risk_limits` - Greeks limits for risk management
    ///
    /// # Returns
    ///
    /// A tuple of (bid_price, ask_price) adjusted for Greeks exposure.
    fn calculate_greeks_adjusted_quotes(
        &self,
        option: &Options,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> MMResult<(Decimal, Decimal)>;

    /// Calculates spread based on option characteristics.
    ///
    /// # Arguments
    ///
    /// * `option` - The option to calculate spread for
    /// * `option_greeks` - Greeks of the specific option
    ///
    /// # Returns
    ///
    /// The spread as a decimal (e.g., 0.02 for 2%).
    fn calculate_options_spread(
        &self,
        option: &Options,
        option_greeks: &PositionGreeks,
    ) -> MMResult<Decimal>;

    /// Calculates hedge orders to neutralize delta.
    ///
    /// # Arguments
    ///
    /// * `portfolio_greeks` - Current portfolio Greek exposure
    /// * `underlying_price` - Current price of the underlying
    /// * `underlying_symbol` - Symbol of the underlying asset
    ///
    /// # Returns
    ///
    /// A vector of hedge orders to neutralize delta exposure.
    fn calculate_delta_hedge(
        &self,
        portfolio_greeks: &PortfolioGreeks,
        underlying_price: Decimal,
        underlying_symbol: &str,
    ) -> MMResult<Vec<HedgeOrder>>;

    /// Calculates the skew adjustment for bid/ask based on portfolio exposure.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the option being quoted
    /// * `portfolio_greeks` - Current portfolio Greek exposure
    /// * `risk_limits` - Greeks limits for risk management
    ///
    /// # Returns
    ///
    /// A tuple of (bid_adjustment, ask_adjustment) as multipliers.
    fn calculate_skew_adjustment(
        &self,
        option_greeks: &PositionGreeks,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> (Decimal, Decimal);
}

/// Implementation of options market making.
pub struct OptionsMarketMakerImpl {
    config: OptionsMarketMakerConfig,
}

impl OptionsMarketMakerImpl {
    /// Creates a new options market maker with the given configuration.
    #[must_use]
    pub fn new(config: OptionsMarketMakerConfig) -> Self {
        Self { config }
    }

    /// Creates a new options market maker with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(OptionsMarketMakerConfig::default())
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &OptionsMarketMakerConfig {
        &self.config
    }

    /// Calculates gamma-based spread adjustment.
    ///
    /// ATM options have higher gamma and require wider spreads.
    fn gamma_spread_adjustment(&self, option: &Options, gamma: Decimal) -> Decimal {
        let underlying = option.underlying_price.to_dec();
        let strike = option.strike_price.to_dec();

        // Check if ATM
        let is_atm = OptionsAdapter::is_atm(underlying, strike, self.config.atm_tolerance);

        if is_atm {
            // ATM options get wider spreads due to higher gamma risk
            gamma.abs() * self.config.gamma_spread_multiplier * dec!(0.01)
        } else {
            // OTM/ITM options have lower gamma, smaller adjustment
            gamma.abs() * self.config.gamma_spread_multiplier * dec!(0.005)
        }
    }

    /// Calculates vega-based spread adjustment.
    ///
    /// Higher vega options need wider spreads in volatile markets.
    fn vega_spread_adjustment(&self, vega: Decimal) -> Decimal {
        vega.abs() * self.config.vega_spread_multiplier * dec!(0.001)
    }

    /// Calculates theta-based spread adjustment.
    ///
    /// Options with high theta decay may need adjusted spreads.
    fn theta_spread_adjustment(&self, theta: Decimal) -> Decimal {
        // Theta is typically negative, so we use absolute value
        theta.abs() * self.config.theta_spread_multiplier * dec!(0.01)
    }

    /// Clamps spread to configured min/max bounds.
    fn clamp_spread(&self, spread: Decimal) -> Decimal {
        if spread < self.config.min_spread_pct {
            self.config.min_spread_pct
        } else if spread > self.config.max_spread_pct {
            self.config.max_spread_pct
        } else {
            spread
        }
    }
}

impl OptionsMarketMaker for OptionsMarketMakerImpl {
    fn calculate_greeks_adjusted_quotes(
        &self,
        option: &Options,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> MMResult<(Decimal, Decimal)> {
        // Get theoretical value
        let theo = OptionsAdapter::theoretical_value(option)?;

        // Get option Greeks
        let option_greeks = OptionsAdapter::calculate_greeks(option)?;

        // Calculate base spread
        let spread_pct = self.calculate_options_spread(option, &option_greeks)?;

        // Calculate skew adjustments based on portfolio exposure
        let (bid_adj, ask_adj) =
            self.calculate_skew_adjustment(&option_greeks, portfolio_greeks, risk_limits);

        // Calculate half spread
        let half_spread = theo * spread_pct / dec!(2.0);

        // Apply skew adjustments
        let bid = theo - half_spread * bid_adj;
        let ask = theo + half_spread * ask_adj;

        // Ensure bid < ask and both are positive
        let final_bid = if bid > Decimal::ZERO { bid } else { dec!(0.01) };
        let final_ask = if ask > final_bid {
            ask
        } else {
            final_bid + dec!(0.01)
        };

        Ok((final_bid, final_ask))
    }

    fn calculate_options_spread(
        &self,
        option: &Options,
        option_greeks: &PositionGreeks,
    ) -> MMResult<Decimal> {
        // Start with base spread
        let mut spread = self.config.base_spread_pct;

        // Add gamma-based adjustment
        spread += self.gamma_spread_adjustment(option, option_greeks.gamma);

        // Add vega-based adjustment
        spread += self.vega_spread_adjustment(option_greeks.vega);

        // Add theta-based adjustment
        spread += self.theta_spread_adjustment(option_greeks.theta);

        // Apply put/call skew if it's a put
        if option.option_style == optionstratlib::OptionStyle::Put {
            spread *= self.config.put_call_skew_factor;
        }

        // Clamp to bounds
        Ok(self.clamp_spread(spread))
    }

    fn calculate_delta_hedge(
        &self,
        portfolio_greeks: &PortfolioGreeks,
        underlying_price: Decimal,
        underlying_symbol: &str,
    ) -> MMResult<Vec<HedgeOrder>> {
        let mut hedges = Vec::new();

        // Calculate shares needed to hedge
        let shares_to_hedge = portfolio_greeks.shares_to_hedge(self.config.contract_multiplier);

        if shares_to_hedge.abs() < dec!(1.0) {
            // No significant hedge needed
            return Ok(hedges);
        }

        // Create underlying hedge order
        let hedge = HedgeOrder::underlying(
            underlying_symbol.to_string(),
            shares_to_hedge,
            underlying_price,
        );

        hedges.push(hedge);

        Ok(hedges)
    }

    fn calculate_skew_adjustment(
        &self,
        option_greeks: &PositionGreeks,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> (Decimal, Decimal) {
        let mut bid_adj = dec!(1.0);
        let mut ask_adj = dec!(1.0);

        // If we're long delta and this option adds more long delta,
        // widen the bid (less aggressive buying) and tighten the ask (more aggressive selling)
        let delta_utilization = portfolio_greeks.delta / risk_limits.max_delta;

        if delta_utilization.abs() > dec!(0.5) {
            // We're using more than 50% of delta capacity
            if portfolio_greeks.delta > Decimal::ZERO && option_greeks.delta > Decimal::ZERO {
                // Long delta portfolio, long delta option - widen bid
                bid_adj += delta_utilization.abs() * dec!(0.5);
                ask_adj -= delta_utilization.abs() * dec!(0.2);
            } else if portfolio_greeks.delta < Decimal::ZERO && option_greeks.delta < Decimal::ZERO
            {
                // Short delta portfolio, short delta option - widen ask
                ask_adj += delta_utilization.abs() * dec!(0.5);
                bid_adj -= delta_utilization.abs() * dec!(0.2);
            }
        }

        // Similar logic for gamma
        let gamma_utilization = portfolio_greeks.gamma / risk_limits.max_gamma;
        if gamma_utilization.abs() > dec!(0.5) {
            // High gamma exposure - widen both sides
            let gamma_adj = gamma_utilization.abs() * dec!(0.3);
            bid_adj += gamma_adj;
            ask_adj += gamma_adj;
        }

        // Ensure adjustments are positive
        bid_adj = if bid_adj > dec!(0.5) {
            bid_adj
        } else {
            dec!(0.5)
        };
        ask_adj = if ask_adj > dec!(0.5) {
            ask_adj
        } else {
            dec!(0.5)
        };

        (bid_adj, ask_adj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optionstratlib::model::ExpirationDate;
    use optionstratlib::model::types::{OptionStyle, OptionType, Side};
    use optionstratlib::pos;

    fn create_test_option() -> Options {
        Options::new(
            OptionType::European,
            Side::Long,
            "TEST".to_string(),
            pos!(100.0),
            ExpirationDate::Days(pos!(30.0)),
            pos!(0.2),
            pos!(1.0),
            pos!(100.0),
            dec!(0.05),
            OptionStyle::Call,
            pos!(0.0),
            None,
        )
    }

    #[test]
    fn test_greeks_limits_default() {
        let limits = GreeksLimits::default();
        assert_eq!(limits.max_delta, dec!(100.0));
        assert_eq!(limits.delta_hedge_threshold, dec!(10.0));
    }

    #[test]
    fn test_greeks_limits_should_hedge() {
        let limits = GreeksLimits::default();

        let mut portfolio = PortfolioGreeks::new();
        portfolio.delta = dec!(5.0);
        assert!(!limits.should_hedge_delta(&portfolio));

        portfolio.delta = dec!(15.0);
        assert!(limits.should_hedge_delta(&portfolio));

        portfolio.delta = dec!(-15.0);
        assert!(limits.should_hedge_delta(&portfolio));
    }

    #[test]
    fn test_hedge_order_underlying() {
        let hedge = HedgeOrder::underlying("SPY".to_string(), dec!(100.0), dec!(450.0));

        assert_eq!(hedge.symbol, "SPY");
        assert_eq!(hedge.quantity, dec!(100.0));
        assert_eq!(hedge.price, dec!(450.0));
        assert_eq!(hedge.hedge_type, HedgeType::Underlying);
        assert_eq!(hedge.delta_impact, dec!(-100.0));
    }

    #[test]
    fn test_options_market_maker_config_default() {
        let config = OptionsMarketMakerConfig::default();
        assert_eq!(config.base_spread_pct, dec!(0.02));
        assert_eq!(config.min_spread_pct, dec!(0.005));
        assert_eq!(config.max_spread_pct, dec!(0.10));
    }

    #[test]
    fn test_calculate_options_spread() {
        let mm = OptionsMarketMakerImpl::default_config();
        let option = create_test_option();
        let greeks = OptionsAdapter::calculate_greeks(&option).unwrap();

        let spread = mm.calculate_options_spread(&option, &greeks).unwrap();

        // Spread should be between min and max
        assert!(spread >= mm.config.min_spread_pct);
        assert!(spread <= mm.config.max_spread_pct);
    }

    #[test]
    fn test_calculate_greeks_adjusted_quotes() {
        let mm = OptionsMarketMakerImpl::default_config();
        let option = create_test_option();
        let portfolio = PortfolioGreeks::new();
        let limits = GreeksLimits::default();

        let (bid, ask) = mm
            .calculate_greeks_adjusted_quotes(&option, &portfolio, &limits)
            .unwrap();

        // Bid should be less than ask
        assert!(bid < ask);
        // Both should be positive
        assert!(bid > Decimal::ZERO);
        assert!(ask > Decimal::ZERO);
    }

    #[test]
    fn test_calculate_delta_hedge() {
        let mm = OptionsMarketMakerImpl::default_config();

        let mut portfolio = PortfolioGreeks::new();
        portfolio.delta = dec!(10.0); // Long 10 delta

        let hedges = mm
            .calculate_delta_hedge(&portfolio, dec!(100.0), "SPY")
            .unwrap();

        assert_eq!(hedges.len(), 1);
        let hedge = &hedges[0];
        assert_eq!(hedge.symbol, "SPY");
        // Should sell shares to hedge long delta
        assert!(hedge.quantity < Decimal::ZERO);
    }

    #[test]
    fn test_calculate_delta_hedge_no_hedge_needed() {
        let mm = OptionsMarketMakerImpl::default_config();

        let portfolio = PortfolioGreeks::new(); // Zero delta

        let hedges = mm
            .calculate_delta_hedge(&portfolio, dec!(100.0), "SPY")
            .unwrap();

        assert!(hedges.is_empty());
    }

    #[test]
    fn test_skew_adjustment_high_delta() {
        let mm = OptionsMarketMakerImpl::default_config();
        let limits = GreeksLimits::default();

        // High long delta portfolio
        let mut portfolio = PortfolioGreeks::new();
        portfolio.delta = dec!(80.0); // 80% of max

        // Long delta option
        let option_greeks = PositionGreeks::new(
            dec!(0.5),   // delta
            dec!(0.02),  // gamma
            dec!(-0.05), // theta
            dec!(0.15),  // vega
            dec!(0.08),  // rho
        );

        let (bid_adj, _ask_adj) = mm.calculate_skew_adjustment(&option_greeks, &portfolio, &limits);

        // Bid should be widened (higher multiplier) when adding more long delta
        assert!(bid_adj > dec!(1.0));
    }

    #[test]
    fn test_gamma_spread_adjustment_atm() {
        let mm = OptionsMarketMakerImpl::default_config();
        let option = create_test_option(); // ATM option

        let atm_adjustment = mm.gamma_spread_adjustment(&option, dec!(0.05));

        // Create OTM option
        let mut otm_option = create_test_option();
        otm_option.strike_price = pos!(120.0); // 20% OTM

        let otm_adjustment = mm.gamma_spread_adjustment(&otm_option, dec!(0.05));

        // ATM should have higher adjustment than OTM
        assert!(atm_adjustment > otm_adjustment);
    }

    #[test]
    fn test_clamp_spread() {
        let mm = OptionsMarketMakerImpl::default_config();

        // Below minimum
        assert_eq!(mm.clamp_spread(dec!(0.001)), mm.config.min_spread_pct);

        // Above maximum
        assert_eq!(mm.clamp_spread(dec!(0.20)), mm.config.max_spread_pct);

        // Within bounds
        assert_eq!(mm.clamp_spread(dec!(0.05)), dec!(0.05));
    }
}
