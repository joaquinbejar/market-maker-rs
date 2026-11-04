//! Chain-level risk management.
//!
//! This module provides risk management functionality for option chains,
//! including Greeks aggregation, risk limits, and hedge calculations.

use crate::Decimal;
use crate::options::greeks::PortfolioGreeks;
use crate::options::market_maker::HedgeOrder;
use crate::types::error::{MMError, MMResult};
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
    /// A fill could not be booked, so `current_greeks` misses it: the chain
    /// reports breached and the hedge and dollar readings fail until
    /// [`Self::reset`] or [`Self::reconcile`]. A snapshot without this field
    /// predates it and reads false, which says nothing about reconciliation.
    #[cfg_attr(feature = "serde", serde(default))]
    booking_failed: bool,
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
            booking_failed: false,
        }
    }

    /// Creates a new chain risk manager with default limits.
    #[must_use]
    pub fn with_defaults(underlying_symbol: impl Into<String>) -> Self {
        Self::new(underlying_symbol, ChainRiskLimits::default(), dec!(100.0))
    }

    /// Returns a reference to the current Greeks.
    ///
    /// This is an incomplete snapshot while [`Self::has_unbooked_fill`] is
    /// true: it misses a real fill.
    #[must_use]
    pub fn current_greeks(&self) -> &PortfolioGreeks {
        &self.current_greeks
    }

    /// Returns true while the Greeks miss a fill that could not be booked,
    /// until [`Self::reset`], which zeroes the Greeks, or [`Self::reconcile`],
    /// which installs correct ones. Later fills that book do not clear it, and
    /// neither does a successful retry, because nothing proves which fill it
    /// was.
    #[must_use]
    pub fn has_unbooked_fill(&self) -> bool {
        self.booking_failed
    }

    /// The current Greeks, unless they are known to miss a fill.
    fn booked_greeks(&self) -> MMResult<&PortfolioGreeks> {
        if self.booking_failed {
            Err(MMError::GreeksMissFill)
        } else {
            Ok(&self.current_greeks)
        }
    }

    /// Latches a failed booking.
    fn after_booking(&mut self, booked: MMResult<()>) -> MMResult<()> {
        booked.map_err(|error| {
            self.booking_failed = true;
            MMError::FillNotBooked(error.to_string())
        })
    }

    /// Returns a reference to the risk limits.
    #[must_use]
    pub fn limits(&self) -> &ChainRiskLimits {
        &self.limits
    }

    /// Updates the current Greeks.
    ///
    /// This does not clear a failed booking; [`Self::reconcile`] does.
    pub fn update_greeks(&mut self, greeks: PortfolioGreeks) {
        self.current_greeks = greeks;
    }

    /// Installs Greeks recomputed from the positions, which include any fill
    /// that could not be booked, and clears the failed booking.
    pub fn reconcile(&mut self, greeks: PortfolioGreeks) {
        self.current_greeks = greeks;
        self.booking_failed = false;
    }

    /// Adds Greeks from a position.
    ///
    /// # Errors
    ///
    /// [`MMError::FillNotBooked`] on overflow: this call leaves the Greeks
    /// unchanged, so they miss a real fill, and [`Self::has_unbooked_fill`]
    /// stays true until [`Self::reset`] or [`Self::reconcile`].
    pub fn add_position_greeks(
        &mut self,
        greeks: &crate::options::greeks::PositionGreeks,
        quantity: Decimal,
    ) -> MMResult<()> {
        let booked = self.current_greeks.add(greeks, quantity);
        self.after_booking(booked)
    }

    /// Removes Greeks from a position.
    ///
    /// # Errors
    ///
    /// [`MMError::FillNotBooked`] on overflow: this call leaves the Greeks
    /// unchanged, so they miss a real fill, and [`Self::has_unbooked_fill`]
    /// stays true until [`Self::reset`] or [`Self::reconcile`].
    pub fn remove_position_greeks(
        &mut self,
        greeks: &crate::options::greeks::PositionGreeks,
        quantity: Decimal,
    ) -> MMResult<()> {
        let booked = self.current_greeks.remove(greeks, quantity);
        self.after_booking(booked)
    }

    /// Checks if the chain should hedge delta.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true.
    pub fn should_hedge(&self) -> MMResult<bool> {
        Ok(self.limits.should_hedge_delta(self.booked_greeks()?))
    }

    /// Checks if any risk limit is breached, which is also the case while
    /// [`Self::has_unbooked_fill`] is true.
    #[must_use]
    pub fn is_risk_breached(&self) -> bool {
        self.booking_failed || self.limits.is_any_limit_breached(&self.current_greeks)
    }

    /// Returns which limits are breached, with a first entry for a fill that
    /// could not be booked while [`Self::has_unbooked_fill`] is true.
    #[must_use]
    pub fn breached_limits(&self) -> Vec<String> {
        let mut breached = self.limits.breached_limits(&self.current_greeks);
        if self.booking_failed {
            breached.insert(0, MMError::GreeksMissFill.to_string());
        }
        breached
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
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and [`MMError::NumericalError`] when the number of shares overflows.
    pub fn calculate_hedge(&self, underlying_price: Decimal) -> MMResult<Vec<HedgeOrder>> {
        let mut hedges = Vec::new();

        let shares_to_hedge = self
            .booked_greeks()?
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

    /// Resets the current Greeks to zero and clears a failed booking.
    pub fn reset(&mut self) {
        self.current_greeks.reset();
        self.booking_failed = false;
    }

    /// Returns the dollar delta exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and [`MMError::NumericalError`] when the exposure overflows.
    pub fn dollar_delta(&self, underlying_price: Decimal) -> MMResult<Decimal> {
        self.booked_greeks()?
            .dollar_delta(underlying_price, self.contract_multiplier)
    }

    /// Returns the dollar gamma exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and [`MMError::NumericalError`] when the exposure overflows.
    pub fn dollar_gamma(&self, underlying_price: Decimal) -> MMResult<Decimal> {
        self.booked_greeks()?
            .dollar_gamma(underlying_price, self.contract_multiplier)
    }

    /// Returns the dollar vega exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and [`MMError::NumericalError`] when the exposure overflows.
    pub fn dollar_vega(&self) -> MMResult<Decimal> {
        self.booked_greeks()?.dollar_vega(self.contract_multiplier)
    }

    /// Returns the dollar theta exposure.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and [`MMError::NumericalError`] when the exposure overflows.
    pub fn dollar_theta(&self) -> MMResult<Decimal> {
        self.booked_greeks()?.dollar_theta(self.contract_multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::greeks::PositionGreeks;

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

        assert!(manager.should_hedge().unwrap());

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
            Err(not_booked("scaled delta"))
        );
        assert_eq!(
            manager.remove_position_greeks(&greeks, dec!(2)),
            Err(not_booked("scaled delta"))
        );
        assert_eq!(*manager.current_greeks(), before);
    }

    fn not_booked(what: &str) -> MMError {
        MMError::FillNotBooked(overflow(what).to_string())
    }

    fn assert_fails_closed(manager: &ChainRiskManager) {
        let refused = MMError::GreeksMissFill;
        assert!(manager.has_unbooked_fill());
        assert!(manager.is_risk_breached());
        assert_eq!(
            manager.breached_limits().first(),
            Some(&refused.to_string())
        );
        assert_eq!(manager.should_hedge(), Err(refused.clone()));
        assert_eq!(
            manager
                .calculate_hedge(dec!(100))
                .map(|hedges| hedges.len()),
            Err(refused.clone())
        );
        assert_eq!(manager.dollar_delta(dec!(100)), Err(refused.clone()));
        assert_eq!(manager.dollar_gamma(dec!(100)), Err(refused.clone()));
        assert_eq!(manager.dollar_vega(), Err(refused.clone()));
        assert_eq!(manager.dollar_theta(), Err(refused));
    }

    fn assert_open(manager: &ChainRiskManager) {
        assert!(!manager.has_unbooked_fill());
        assert!(!manager.is_risk_breached());
        assert!(manager.breached_limits().is_empty());
        assert_eq!(manager.should_hedge(), Ok(false));
        assert_eq!(manager.dollar_vega(), Ok(dec!(0)));
    }

    #[test]
    fn test_failed_add_fails_closed_on_a_flat_book_until_reset() {
        let mut manager = ChainRiskManager::with_defaults("BTC");
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        let small = PositionGreeks::new(dec!(0.01), dec!(0), dec!(0), dec!(0), dec!(0));
        assert_open(&manager);

        assert_eq!(
            manager.add_position_greeks(&huge, dec!(2)),
            Err(not_booked("scaled delta"))
        );
        assert_eq!(*manager.current_greeks(), PortfolioGreeks::new());
        assert_fails_closed(&manager);

        // Neither a later fill that books, a second failure nor a plain
        // update clears it.
        manager.add_position_greeks(&small, dec!(1)).unwrap();
        assert_eq!(manager.current_greeks().delta, dec!(0.01));
        assert_fails_closed(&manager);
        assert_eq!(
            manager.add_position_greeks(&huge, dec!(2)),
            Err(not_booked("scaled delta"))
        );
        assert_eq!(manager.current_greeks().delta, dec!(0.01));
        assert_fails_closed(&manager);
        manager.update_greeks(PortfolioGreeks::new());
        assert_fails_closed(&manager);

        manager.reset();
        assert_open(&manager);
    }

    #[test]
    fn test_failed_remove_fails_closed_until_reconciled() {
        let mut manager = ChainRiskManager::with_defaults("BTC");
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));

        assert_eq!(
            manager.remove_position_greeks(&huge, dec!(2)),
            Err(not_booked("scaled delta"))
        );
        assert_fails_closed(&manager);

        let recomputed = PortfolioGreeks {
            delta: dec!(1),
            ..PortfolioGreeks::new()
        };
        manager.reconcile(recomputed);
        assert_eq!(*manager.current_greeks(), recomputed);
        assert_open(&manager);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_failed_booking_survives_a_snapshot() {
        let mut manager = ChainRiskManager::with_defaults("BTC");
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        manager.add_position_greeks(&huge, dec!(2)).unwrap_err();

        let snapshot = serde_json::to_value(&manager).unwrap();
        let restored: ChainRiskManager = serde_json::from_value(snapshot.clone()).unwrap();
        assert_fails_closed(&restored);

        // A snapshot from before the field reads as no failed booking.
        let mut old = snapshot;
        old.as_object_mut().unwrap().remove("booking_failed");
        let restored: ChainRiskManager = serde_json::from_value(old).unwrap();
        assert!(!restored.has_unbooked_fill());
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
