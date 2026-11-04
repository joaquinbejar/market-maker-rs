//! Chain-level risk management.
//!
//! This module provides risk management functionality for option chains,
//! including Greeks aggregation, risk limits, and hedge calculations.

use crate::Decimal;
use crate::options::greeks::PortfolioGreeks;
use crate::options::market_maker::HedgeOrder;
use crate::types::error::MMResult;
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Risk limits for an entire option chain.
///
/// Defines thresholds that trigger risk actions at the chain level.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainRiskLimits {
    /// Maximum absolute delta for the entire chain.
    pub max_chain_delta: Decimal,
    /// Maximum absolute gamma for the entire chain.
    pub max_chain_gamma: Decimal,
    /// Maximum absolute vega for the entire chain.
    pub max_chain_vega: Decimal,
    /// Maximum daily theta decay allowed.
    pub max_chain_theta: Decimal,
    /// Maximum delta per individual strike.
    pub max_delta_per_strike: Decimal,
    /// Delta threshold that triggers hedge suggestions.
    pub delta_hedge_threshold: Decimal,
    /// Maximum notional exposure.
    pub max_notional: Decimal,
}

impl Default for ChainRiskLimits {
    fn default() -> Self {
        Self {
            max_chain_delta: dec!(500.0),
            max_chain_gamma: dec!(100.0),
            max_chain_vega: dec!(5000.0),
            max_chain_theta: dec!(-2000.0),
            max_delta_per_strike: dec!(50.0),
            delta_hedge_threshold: dec!(25.0),
            max_notional: dec!(1000000.0),
        }
    }
}

impl ChainRiskLimits {
    /// Creates new chain risk limits with specified values.
    #[must_use]
    pub fn new(
        max_chain_delta: Decimal,
        max_chain_gamma: Decimal,
        max_chain_vega: Decimal,
        max_chain_theta: Decimal,
        max_delta_per_strike: Decimal,
        delta_hedge_threshold: Decimal,
        max_notional: Decimal,
    ) -> Self {
        Self {
            max_chain_delta,
            max_chain_gamma,
            max_chain_vega,
            max_chain_theta,
            max_delta_per_strike,
            delta_hedge_threshold,
            max_notional,
        }
    }

    /// Checks if delta exceeds the hedge threshold.
    #[must_use]
    pub fn should_hedge_delta(&self, greeks: &PortfolioGreeks) -> bool {
        greeks.delta.abs() > self.delta_hedge_threshold
    }

    /// Checks if any chain-level limit is breached.
    #[must_use]
    pub fn is_any_limit_breached(&self, greeks: &PortfolioGreeks) -> bool {
        greeks.delta.abs() > self.max_chain_delta
            || greeks.gamma.abs() > self.max_chain_gamma
            || greeks.vega.abs() > self.max_chain_vega
            || greeks.theta < self.max_chain_theta
    }

    /// Returns which limits are breached.
    #[must_use]
    pub fn breached_limits(&self, greeks: &PortfolioGreeks) -> Vec<String> {
        let mut breached = Vec::new();

        if greeks.delta.abs() > self.max_chain_delta {
            breached.push(format!(
                "Delta: {} > {}",
                greeks.delta.abs(),
                self.max_chain_delta
            ));
        }
        if greeks.gamma.abs() > self.max_chain_gamma {
            breached.push(format!(
                "Gamma: {} > {}",
                greeks.gamma.abs(),
                self.max_chain_gamma
            ));
        }
        if greeks.vega.abs() > self.max_chain_vega {
            breached.push(format!(
                "Vega: {} > {}",
                greeks.vega.abs(),
                self.max_chain_vega
            ));
        }
        if greeks.theta < self.max_chain_theta {
            breached.push(format!(
                "Theta: {} < {}",
                greeks.theta, self.max_chain_theta
            ));
        }

        breached
    }
}

/// Chain-level risk manager.
///
/// Tracks and manages risk across an entire option chain.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainRiskManager {
    /// Risk limits for the chain.
    limits: ChainRiskLimits,
    /// Current aggregated Greeks for the chain.
    current_greeks: PortfolioGreeks,
    /// Contract multiplier for dollar calculations.
    contract_multiplier: Decimal,
    /// Underlying symbol.
    underlying_symbol: String,
}

impl ChainRiskManager {
    /// Creates a new chain risk manager.
    ///
    /// # Arguments
    ///
    /// * `underlying_symbol` - The underlying asset symbol
    /// * `limits` - Risk limits for the chain
    /// * `contract_multiplier` - Contract multiplier for dollar calculations
    #[must_use]
    pub fn new(
        underlying_symbol: impl Into<String>,
        limits: ChainRiskLimits,
        contract_multiplier: Decimal,
    ) -> Self {
        Self {
            limits,
            current_greeks: PortfolioGreeks::new(),
            contract_multiplier,
            underlying_symbol: underlying_symbol.into(),
        }
    }

    /// Creates a new chain risk manager with default limits.
    #[must_use]
    pub fn with_defaults(underlying_symbol: impl Into<String>) -> Self {
        Self::new(underlying_symbol, ChainRiskLimits::default(), dec!(100.0))
    }

    /// Returns a reference to the current Greeks.
    #[must_use]
    pub fn current_greeks(&self) -> &PortfolioGreeks {
        &self.current_greeks
    }

    /// Returns a reference to the risk limits.
    #[must_use]
    pub fn limits(&self) -> &ChainRiskLimits {
        &self.limits
    }

    /// Updates the current Greeks.
    pub fn update_greeks(&mut self, greeks: PortfolioGreeks) {
        self.current_greeks = greeks;
    }

    /// Adds Greeks from a position.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// on overflow; the current Greeks are unchanged.
    pub fn add_position_greeks(
        &mut self,
        greeks: &crate::options::greeks::PositionGreeks,
        quantity: Decimal,
    ) -> MMResult<()> {
        self.current_greeks.add(greeks, quantity)
    }

    /// Removes Greeks from a position.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// on overflow; the current Greeks are unchanged.
    pub fn remove_position_greeks(
        &mut self,
        greeks: &crate::options::greeks::PositionGreeks,
        quantity: Decimal,
    ) -> MMResult<()> {
        self.current_greeks.remove(greeks, quantity)
    }

    /// Checks if the chain should hedge delta.
    #[must_use]
    pub fn should_hedge(&self) -> bool {
        self.limits.should_hedge_delta(&self.current_greeks)
    }

    /// Checks if any risk limit is breached.
    #[must_use]
    pub fn is_risk_breached(&self) -> bool {
        self.limits.is_any_limit_breached(&self.current_greeks)
    }

    /// Returns which limits are breached.
    #[must_use]
    pub fn breached_limits(&self) -> Vec<String> {
        self.limits.breached_limits(&self.current_greeks)
    }

    /// Calculates hedge orders to neutralize delta.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying
    ///
    /// # Returns
    ///
    /// A vector of hedge orders to neutralize delta exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the number of shares overflows.
    pub fn calculate_hedge(&self, underlying_price: Decimal) -> MMResult<Vec<HedgeOrder>> {
        let mut hedges = Vec::new();

        let shares_to_hedge = self
            .current_greeks
            .shares_to_hedge(self.contract_multiplier)?;

        if shares_to_hedge.abs() < dec!(1.0) {
            return Ok(hedges);
        }

        let hedge = HedgeOrder::underlying(
            self.underlying_symbol.clone(),
            shares_to_hedge,
            underlying_price,
        );

        hedges.push(hedge);
        Ok(hedges)
    }

    /// Resets the current Greeks to zero.
    pub fn reset(&mut self) {
        self.current_greeks.reset();
    }

    /// Returns the dollar delta exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_delta(&self, underlying_price: Decimal) -> MMResult<Decimal> {
        self.current_greeks
            .dollar_delta(underlying_price, self.contract_multiplier)
    }

    /// Returns the dollar gamma exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_gamma(&self, underlying_price: Decimal) -> MMResult<Decimal> {
        self.current_greeks
            .dollar_gamma(underlying_price, self.contract_multiplier)
    }

    /// Returns the dollar vega exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_vega(&self) -> MMResult<Decimal> {
        self.current_greeks.dollar_vega(self.contract_multiplier)
    }

    /// Returns the dollar theta exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_theta(&self) -> MMResult<Decimal> {
        self.current_greeks.dollar_theta(self.contract_multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::greeks::PositionGreeks;
    use crate::types::error::MMError;

    #[test]
    fn test_chain_risk_limits_default() {
        let limits = ChainRiskLimits::default();
        assert_eq!(limits.max_chain_delta, dec!(500.0));
        assert_eq!(limits.delta_hedge_threshold, dec!(25.0));
    }

    #[test]
    fn test_chain_risk_limits_should_hedge() {
        let limits = ChainRiskLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(20.0);
        assert!(!limits.should_hedge_delta(&greeks));

        greeks.delta = dec!(30.0);
        assert!(limits.should_hedge_delta(&greeks));
    }

    #[test]
    fn test_chain_risk_limits_breached() {
        let limits = ChainRiskLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(600.0); // Exceeds max_chain_delta

        assert!(limits.is_any_limit_breached(&greeks));

        let breached = limits.breached_limits(&greeks);
        assert_eq!(breached.len(), 1);
        assert!(breached[0].contains("Delta"));
    }

    #[test]
    fn test_chain_risk_manager_new() {
        let manager = ChainRiskManager::with_defaults("BTC");
        assert_eq!(manager.current_greeks().delta, Decimal::ZERO);
    }

    #[test]
    fn test_chain_risk_manager_add_position() {
        let mut manager = ChainRiskManager::with_defaults("BTC");

        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        manager.add_position_greeks(&greeks, dec!(10.0)).unwrap();

        assert_eq!(manager.current_greeks().delta, dec!(5.0));
        assert_eq!(manager.current_greeks().gamma, dec!(0.2));
    }

    #[test]
    fn test_chain_risk_manager_calculate_hedge() {
        let mut manager = ChainRiskManager::with_defaults("BTC");

        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        // Add enough delta to trigger hedge
        manager.add_position_greeks(&greeks, dec!(100.0)).unwrap(); // 50 delta

        assert!(manager.should_hedge());

        let hedges = manager.calculate_hedge(dec!(50000.0)).unwrap();
        assert_eq!(hedges.len(), 1);
        assert_eq!(hedges[0].symbol, "BTC");
    }

    #[test]
    fn test_chain_risk_manager_reset() {
        let mut manager = ChainRiskManager::with_defaults("BTC");

        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        manager.add_position_greeks(&greeks, dec!(10.0)).unwrap();
        assert!(manager.current_greeks().delta != Decimal::ZERO);

        manager.reset();
        assert_eq!(manager.current_greeks().delta, Decimal::ZERO);
    }

    fn overflow(what: &str) -> MMError {
        MMError::NumericalError(format!("{what} overflows"))
    }

    #[test]
    fn test_chain_risk_manager_position_overflow_leaves_greeks_unchanged() {
        let mut manager = ChainRiskManager::with_defaults("BTC");
        let greeks = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        let unit = PositionGreeks::new(dec!(1), dec!(0), dec!(0), dec!(0), dec!(0));
        manager.add_position_greeks(&unit, dec!(5)).unwrap();
        let before = *manager.current_greeks();

        assert_eq!(
            manager.add_position_greeks(&greeks, dec!(2)),
            Err(overflow("scaled delta"))
        );
        assert_eq!(
            manager.remove_position_greeks(&greeks, dec!(2)),
            Err(overflow("scaled delta"))
        );
        assert_eq!(*manager.current_greeks(), before);
    }

    #[test]
    fn test_chain_risk_manager_hedge_and_dollar_values_overflow() {
        let mut manager = ChainRiskManager::with_defaults("BTC");
        manager.update_greeks(PortfolioGreeks {
            delta: Decimal::MAX,
            gamma: Decimal::MAX,
            theta: Decimal::MAX,
            vega: Decimal::MAX,
            rho: dec!(0),
        });
        // The default contract multiplier is 100.
        assert_eq!(
            manager.calculate_hedge(dec!(1)).map(|hedges| hedges.len()),
            Err(overflow("shares to hedge"))
        );
        assert_eq!(manager.dollar_delta(dec!(1)), Err(overflow("dollar delta")));
        assert_eq!(
            manager.dollar_gamma(dec!(100)),
            Err(overflow("dollar gamma"))
        );
        assert_eq!(manager.dollar_vega(), Err(overflow("dollar vega")));
        assert_eq!(manager.dollar_theta(), Err(overflow("dollar theta")));
    }
}
