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
use crate::types::decimal::{OrOverflow, negate};
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
        let delta_impact = negate(quantity); // Underlying has delta of 1
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// for a zero delta or gamma limit, and when a utilization overflows.
    fn calculate_skew_adjustment(
        &self,
        option_greeks: &PositionGreeks,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> MMResult<(Decimal, Decimal)>;
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
    fn gamma_spread_adjustment(&self, option: &Options, gamma: Decimal) -> MMResult<Decimal> {
        let underlying = option.underlying_price.to_dec();
        let strike = option.strike_price.to_dec();

        // Check if ATM
        let is_atm = OptionsAdapter::is_atm(underlying, strike, self.config.atm_tolerance)?;

        // ATM options get wider spreads due to higher gamma risk; OTM/ITM
        // options have lower gamma, smaller adjustment
        let rate = if is_atm { dec!(0.01) } else { dec!(0.005) };
        gamma
            .abs()
            .checked_mul(self.config.gamma_spread_multiplier)
            .and_then(|x| x.checked_mul(rate))
            .or_overflow("gamma spread adjustment")
    }

    /// Calculates vega-based spread adjustment.
    ///
    /// Higher vega options need wider spreads in volatile markets.
    fn vega_spread_adjustment(&self, vega: Decimal) -> MMResult<Decimal> {
        vega.abs()
            .checked_mul(self.config.vega_spread_multiplier)
            .and_then(|x| x.checked_mul(dec!(0.001)))
            .or_overflow("vega spread adjustment")
    }

    /// Calculates theta-based spread adjustment.
    ///
    /// Options with high theta decay may need adjusted spreads.
    fn theta_spread_adjustment(&self, theta: Decimal) -> MMResult<Decimal> {
        // Theta is typically negative, so we use absolute value
        theta
            .abs()
            .checked_mul(self.config.theta_spread_multiplier)
            .and_then(|x| x.checked_mul(dec!(0.01)))
            .or_overflow("theta spread adjustment")
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

/// Makes a quote positive and uncrossed: a bid that is not positive becomes
/// one cent, and an ask not above the bid becomes one cent above it.
///
/// # Errors
///
/// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
/// when no representable ask is above the bid.
fn positive_uncrossed(bid: Decimal, ask: Decimal) -> MMResult<(Decimal, Decimal)> {
    let final_bid = if bid > Decimal::ZERO { bid } else { dec!(0.01) };
    if ask > final_bid {
        return Ok((final_bid, ask));
    }
    // The filter catches a cent rounded away at the top of the range.
    let final_ask = final_bid
        .checked_add(dec!(0.01))
        .filter(|ask| *ask > final_bid)
        .or_overflow("ask above the bid")?;
    Ok((final_bid, final_ask))
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
            self.calculate_skew_adjustment(&option_greeks, portfolio_greeks, risk_limits)?;

        // Calculate half spread
        let half_spread = theo
            .checked_mul(spread_pct)
            .and_then(|x| x.checked_div(dec!(2.0)))
            .or_overflow("half spread")?;

        // Apply skew adjustments
        let bid = half_spread
            .checked_mul(bid_adj)
            .and_then(|x| theo.checked_sub(x))
            .or_overflow("bid")?;
        let ask = half_spread
            .checked_mul(ask_adj)
            .and_then(|x| theo.checked_add(x))
            .or_overflow("ask")?;

        // Ensure bid < ask and both are positive
        positive_uncrossed(bid, ask)
    }

    fn calculate_options_spread(
        &self,
        option: &Options,
        option_greeks: &PositionGreeks,
    ) -> MMResult<Decimal> {
        let gamma_adj = self.gamma_spread_adjustment(option, option_greeks.gamma)?;
        let vega_adj = self.vega_spread_adjustment(option_greeks.vega)?;
        let theta_adj = self.theta_spread_adjustment(option_greeks.theta)?;

        // Start with base spread, then add the gamma, vega and theta based
        // adjustments
        let mut spread = self
            .config
            .base_spread_pct
            .checked_add(gamma_adj)
            .and_then(|x| x.checked_add(vega_adj))
            .and_then(|x| x.checked_add(theta_adj))
            .or_overflow("options spread")?;

        // Apply put/call skew if it's a put
        if option.option_style == optionstratlib::OptionStyle::Put {
            spread = spread
                .checked_mul(self.config.put_call_skew_factor)
                .or_overflow("options spread")?;
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
        let shares_to_hedge = portfolio_greeks.shares_to_hedge(self.config.contract_multiplier)?;

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
    ) -> MMResult<(Decimal, Decimal)> {
        let mut bid_adj = dec!(1.0);
        let mut ask_adj = dec!(1.0);

        // A utilization is at most `Decimal::MAX`, so an adjustment is at most
        // `1 + 0.5 * MAX + 0.3 * MAX`: the saturating operations below never
        // saturate. Only the utilizations themselves can fail.

        // If we're long delta and this option adds more long delta,
        // widen the bid (less aggressive buying) and tighten the ask (more aggressive selling)
        let delta_utilization = portfolio_greeks
            .delta
            .checked_div(risk_limits.max_delta)
            .or_overflow("delta utilization")?;

        if delta_utilization.abs() > dec!(0.5) {
            // We're using more than 50% of delta capacity
            let widen = delta_utilization.abs().saturating_mul(dec!(0.5));
            let tighten = delta_utilization.abs().saturating_mul(dec!(0.2));
            if portfolio_greeks.delta > Decimal::ZERO && option_greeks.delta > Decimal::ZERO {
                // Long delta portfolio, long delta option - widen bid
                bid_adj = bid_adj.saturating_add(widen);
                ask_adj = ask_adj.saturating_sub(tighten);
            } else if portfolio_greeks.delta < Decimal::ZERO && option_greeks.delta < Decimal::ZERO
            {
                // Short delta portfolio, short delta option - widen ask
                ask_adj = ask_adj.saturating_add(widen);
                bid_adj = bid_adj.saturating_sub(tighten);
            }
        }

        // Similar logic for gamma
        let gamma_utilization = portfolio_greeks
            .gamma
            .checked_div(risk_limits.max_gamma)
            .or_overflow("gamma utilization")?;
        if gamma_utilization.abs() > dec!(0.5) {
            // High gamma exposure - widen both sides
            let gamma_adj = gamma_utilization.abs().saturating_mul(dec!(0.3));
            bid_adj = bid_adj.saturating_add(gamma_adj);
            ask_adj = ask_adj.saturating_add(gamma_adj);
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

        Ok((bid_adj, ask_adj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::error::MMError;
    use optionstratlib::model::ExpirationDate;
    use optionstratlib::model::types::{OptionStyle, OptionType, Side};
    use optionstratlib::prelude::pos_or_panic;

    fn create_test_option() -> Options {
        Options::new(
            OptionType::European,
            Side::Long,
            "TEST".to_string(),
            pos_or_panic!(100.0),
            ExpirationDate::Days(pos_or_panic!(30.0)),
            pos_or_panic!(0.2),
            pos_or_panic!(1.0),
            pos_or_panic!(100.0),
            dec!(0.05),
            OptionStyle::Call,
            pos_or_panic!(0.0),
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

        let (bid_adj, _ask_adj) = mm
            .calculate_skew_adjustment(&option_greeks, &portfolio, &limits)
            .unwrap();

        // Bid should be widened (higher multiplier) when adding more long delta
        assert!(bid_adj > dec!(1.0));
    }

    #[test]
    fn test_gamma_spread_adjustment_atm() {
        let mm = OptionsMarketMakerImpl::default_config();
        let option = create_test_option(); // ATM option

        let atm_adjustment = mm.gamma_spread_adjustment(&option, dec!(0.05)).unwrap();

        // Create OTM option
        let mut otm_option = create_test_option();
        otm_option.strike_price = pos_or_panic!(120.0); // 20% OTM

        let otm_adjustment = mm.gamma_spread_adjustment(&otm_option, dec!(0.05)).unwrap();

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

    fn overflow(what: &str) -> MMError {
        MMError::NumericalError(format!("{what} overflows"))
    }

    fn with_config(config: OptionsMarketMakerConfig) -> OptionsMarketMakerImpl {
        OptionsMarketMakerImpl::new(config)
    }

    fn unit_greeks() -> PositionGreeks {
        PositionGreeks::new(dec!(1), dec!(2), dec!(-2), dec!(2), dec!(0))
    }

    #[test]
    fn test_hedge_order_underlying_negates_at_the_bound() {
        let hedge = HedgeOrder::underlying("SPY".to_string(), Decimal::MIN, dec!(1));
        assert_eq!(hedge.delta_impact, Decimal::MAX);
    }

    #[test]
    fn test_spread_adjustments_overflow() {
        let mm = with_config(OptionsMarketMakerConfig {
            gamma_spread_multiplier: Decimal::MAX,
            vega_spread_multiplier: Decimal::MAX,
            theta_spread_multiplier: Decimal::MAX,
            ..Default::default()
        });
        assert_eq!(
            mm.gamma_spread_adjustment(&create_test_option(), dec!(2)),
            Err(overflow("gamma spread adjustment"))
        );
        assert_eq!(
            mm.vega_spread_adjustment(dec!(2)),
            Err(overflow("vega spread adjustment"))
        );
        assert_eq!(
            mm.theta_spread_adjustment(dec!(-2)),
            Err(overflow("theta spread adjustment"))
        );
    }

    #[test]
    fn test_calculate_options_spread_propagates_adjustment_overflow() {
        let option = create_test_option();
        for (config, what) in [
            (
                OptionsMarketMakerConfig {
                    gamma_spread_multiplier: Decimal::MAX,
                    ..Default::default()
                },
                "gamma spread adjustment",
            ),
            (
                OptionsMarketMakerConfig {
                    vega_spread_multiplier: Decimal::MAX,
                    ..Default::default()
                },
                "vega spread adjustment",
            ),
            (
                OptionsMarketMakerConfig {
                    theta_spread_multiplier: Decimal::MAX,
                    ..Default::default()
                },
                "theta spread adjustment",
            ),
        ] {
            assert_eq!(
                with_config(config).calculate_options_spread(&option, &unit_greeks()),
                Err(overflow(what))
            );
        }
    }

    #[test]
    fn test_calculate_options_spread_sum_overflow() {
        let mm = with_config(OptionsMarketMakerConfig {
            base_spread_pct: Decimal::MAX,
            ..Default::default()
        });
        // One addend at a time, each large enough not to round away at the
        // bound: gamma 4, vega 1 and theta 2 with the default multipliers.
        for greeks in [
            PositionGreeks::new(dec!(0), dec!(200), dec!(0), dec!(0), dec!(0)),
            PositionGreeks::new(dec!(0), dec!(0), dec!(0), dec!(2000), dec!(0)),
            PositionGreeks::new(dec!(0), dec!(0), dec!(-2000), dec!(0), dec!(0)),
        ] {
            assert_eq!(
                mm.calculate_options_spread(&create_test_option(), &greeks),
                Err(overflow("options spread"))
            );
        }
    }

    #[test]
    fn test_calculate_options_spread_put_skew_overflow() {
        let mm = with_config(OptionsMarketMakerConfig {
            base_spread_pct: dec!(2),
            put_call_skew_factor: Decimal::MAX,
            max_spread_pct: Decimal::MAX,
            ..Default::default()
        });
        let mut put = create_test_option();
        put.option_style = OptionStyle::Put;
        assert_eq!(
            mm.calculate_options_spread(&put, &unit_greeks()),
            Err(overflow("options spread"))
        );
    }

    /// A config whose spread is as large as `spread` whatever the Greeks.
    fn fixed_spread(spread: Decimal) -> OptionsMarketMakerImpl {
        with_config(OptionsMarketMakerConfig {
            base_spread_pct: spread,
            min_spread_pct: spread,
            max_spread_pct: spread,
            ..Default::default()
        })
    }

    #[test]
    fn test_quotes_half_spread_overflow() {
        // The theoretical value is about 2.5, so its product with half the
        // bound overflows.
        let quotes = fixed_spread(Decimal::MAX / dec!(2)).calculate_greeks_adjusted_quotes(
            &create_test_option(),
            &PortfolioGreeks::new(),
            &GreeksLimits::default(),
        );
        assert_eq!(quotes, Err(overflow("half spread")));
    }

    #[test]
    fn test_quotes_bid_overflow() {
        // The theoretical value is about 2.5, so the half spread is about
        // 1/8 of the bound and a skew of 51 overflows.
        let portfolio = PortfolioGreeks {
            delta: dec!(10000),
            ..PortfolioGreeks::new()
        };
        let quotes = fixed_spread(Decimal::MAX / dec!(10)).calculate_greeks_adjusted_quotes(
            &create_test_option(),
            &portfolio,
            &GreeksLimits::default(),
        );
        assert_eq!(quotes, Err(overflow("bid")));
    }

    #[test]
    fn test_quotes_ask_overflow() {
        // A short delta portfolio and a short delta option widen the ask
        // only, and the bid adjustment stays at its floor.
        let portfolio = PortfolioGreeks {
            delta: dec!(-10000),
            ..PortfolioGreeks::new()
        };
        let mut put = create_test_option();
        put.option_style = OptionStyle::Put;
        let quotes = fixed_spread(Decimal::MAX / dec!(10)).calculate_greeks_adjusted_quotes(
            &put,
            &portfolio,
            &GreeksLimits::default(),
        );
        assert_eq!(quotes, Err(overflow("ask")));
    }

    #[test]
    fn test_positive_uncrossed() {
        assert_eq!(positive_uncrossed(dec!(1), dec!(2)), Ok((dec!(1), dec!(2))));
        // A bid that is not positive becomes a cent.
        assert_eq!(
            positive_uncrossed(dec!(-1), dec!(2)),
            Ok((dec!(0.01), dec!(2)))
        );
        // A crossed ask becomes a cent above the bid.
        assert_eq!(
            positive_uncrossed(dec!(2), dec!(1)),
            Ok((dec!(2), dec!(2.01)))
        );
    }

    #[test]
    fn test_positive_uncrossed_no_ask_above_the_bid() {
        assert_eq!(
            positive_uncrossed(Decimal::MAX, dec!(1)),
            Err(overflow("ask above the bid"))
        );
    }

    #[test]
    fn test_calculate_delta_hedge_overflow() {
        let mm = with_config(OptionsMarketMakerConfig {
            contract_multiplier: Decimal::MAX,
            ..Default::default()
        });
        let portfolio = PortfolioGreeks {
            delta: dec!(2),
            ..PortfolioGreeks::new()
        };
        assert_eq!(
            mm.calculate_delta_hedge(&portfolio, dec!(100), "SPY")
                .map(|hedges| hedges.len()),
            Err(overflow("shares to hedge"))
        );
    }

    #[test]
    fn test_skew_adjustment_zero_limits_are_errors() {
        let mm = OptionsMarketMakerImpl::default_config();
        let portfolio = PortfolioGreeks::new();
        let greeks = unit_greeks();
        let zero_delta = GreeksLimits {
            max_delta: Decimal::ZERO,
            ..Default::default()
        };
        assert_eq!(
            mm.calculate_skew_adjustment(&greeks, &portfolio, &zero_delta),
            Err(overflow("delta utilization"))
        );
        let zero_gamma = GreeksLimits {
            max_gamma: Decimal::ZERO,
            ..Default::default()
        };
        assert_eq!(
            mm.calculate_skew_adjustment(&greeks, &portfolio, &zero_gamma),
            Err(overflow("gamma utilization"))
        );
    }

    #[test]
    fn test_skew_adjustment_utilization_overflow() {
        let mm = OptionsMarketMakerImpl::default_config();
        let tiny = Decimal::new(1, 28);
        let limits = GreeksLimits {
            max_delta: tiny,
            ..Default::default()
        };
        let portfolio = PortfolioGreeks {
            delta: Decimal::MAX,
            ..PortfolioGreeks::new()
        };
        assert_eq!(
            mm.calculate_skew_adjustment(&unit_greeks(), &portfolio, &limits),
            Err(overflow("delta utilization"))
        );
    }

    #[test]
    fn test_skew_adjustment_at_maximum_utilization_does_not_saturate() {
        // Delta and gamma both at the bound with unit limits: the bid
        // adjustment is 1 + 0.5 MAX + 0.3 MAX.
        let mm = OptionsMarketMakerImpl::default_config();
        let limits = GreeksLimits {
            max_delta: dec!(1),
            max_gamma: dec!(1),
            ..Default::default()
        };
        let portfolio = PortfolioGreeks {
            delta: Decimal::MAX,
            gamma: Decimal::MAX,
            ..PortfolioGreeks::new()
        };
        let (bid_adj, ask_adj) = mm
            .calculate_skew_adjustment(&unit_greeks(), &portfolio, &limits)
            .unwrap();
        assert!(bid_adj < Decimal::MAX);
        assert!(bid_adj > ask_adj);
    }
}
