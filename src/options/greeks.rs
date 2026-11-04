//! Portfolio Greeks aggregation and tracking.
//!
//! This module provides types for tracking and aggregating Greeks across
//! multiple options positions in a portfolio.

// TODO(#166): legacy pricing arithmetic, converted in #166
#![allow(clippy::arithmetic_side_effects)]

use crate::Decimal;
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Greeks for a single option position.
///
/// Represents the sensitivity measures for an individual option contract.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::options::PositionGreeks;
/// use rust_decimal_macros::dec;
///
/// let greeks = PositionGreeks {
///     delta: dec!(0.55),
///     gamma: dec!(0.02),
///     theta: dec!(-0.05),
///     vega: dec!(0.15),
///     rho: dec!(0.08),
/// };
///
/// assert!(greeks.delta > dec!(0.0));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PositionGreeks {
    /// Delta: Rate of change of option price with respect to underlying price.
    /// Range: -1.0 to 1.0 for single options.
    pub delta: Decimal,

    /// Gamma: Rate of change of delta with respect to underlying price.
    /// Always positive for long options.
    pub gamma: Decimal,

    /// Theta: Rate of change of option price with respect to time (time decay).
    /// Usually negative for long options (value decreases as time passes).
    pub theta: Decimal,

    /// Vega: Rate of change of option price with respect to volatility.
    /// Always positive for long options.
    pub vega: Decimal,

    /// Rho: Rate of change of option price with respect to interest rate.
    pub rho: Decimal,
}

impl PositionGreeks {
    /// Creates a new `PositionGreeks` with the specified values.
    #[must_use]
    pub fn new(
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        rho: Decimal,
    ) -> Self {
        Self {
            delta,
            gamma,
            theta,
            vega,
            rho,
        }
    }

    /// Creates a zero Greeks instance.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Scales the Greeks by a quantity multiplier.
    ///
    /// # Arguments
    ///
    /// * `quantity` - The position quantity (positive for long, negative for short)
    ///
    /// # Returns
    ///
    /// A new `PositionGreeks` with all values scaled by the quantity.
    #[must_use]
    pub fn scale(&self, quantity: Decimal) -> Self {
        Self {
            delta: self.delta * quantity,
            gamma: self.gamma * quantity,
            theta: self.theta * quantity,
            vega: self.vega * quantity,
            rho: self.rho * quantity,
        }
    }
}

/// Aggregated Greeks for an entire portfolio of options.
///
/// Tracks the total Greek exposure across all positions and provides
/// methods for adding, removing, and analyzing portfolio risk.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::options::{PortfolioGreeks, PositionGreeks};
/// use rust_decimal_macros::dec;
///
/// let mut portfolio = PortfolioGreeks::new();
///
/// // Add a long call position
/// let call_greeks = PositionGreeks::new(
///     dec!(0.55),  // delta
///     dec!(0.02),  // gamma
///     dec!(-0.05), // theta
///     dec!(0.15),  // vega
///     dec!(0.08),  // rho
/// );
/// portfolio.add(&call_greeks, dec!(10.0)); // 10 contracts
///
/// assert_eq!(portfolio.delta, dec!(5.5)); // 0.55 * 10
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PortfolioGreeks {
    /// Total portfolio delta exposure.
    pub delta: Decimal,

    /// Total portfolio gamma exposure.
    pub gamma: Decimal,

    /// Total portfolio theta (daily time decay).
    pub theta: Decimal,

    /// Total portfolio vega exposure.
    pub vega: Decimal,

    /// Total portfolio rho exposure.
    pub rho: Decimal,
}

impl PortfolioGreeks {
    /// Creates a new empty `PortfolioGreeks`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds Greeks from a position to the portfolio.
    ///
    /// # Arguments
    ///
    /// * `greeks` - The Greeks of the position to add
    /// * `quantity` - The position quantity (positive for long, negative for short)
    pub fn add(&mut self, greeks: &PositionGreeks, quantity: Decimal) {
        self.delta += greeks.delta * quantity;
        self.gamma += greeks.gamma * quantity;
        self.theta += greeks.theta * quantity;
        self.vega += greeks.vega * quantity;
        self.rho += greeks.rho * quantity;
    }

    /// Removes Greeks from a position from the portfolio.
    ///
    /// # Arguments
    ///
    /// * `greeks` - The Greeks of the position to remove
    /// * `quantity` - The position quantity that was removed
    pub fn remove(&mut self, greeks: &PositionGreeks, quantity: Decimal) {
        self.delta -= greeks.delta * quantity;
        self.gamma -= greeks.gamma * quantity;
        self.theta -= greeks.theta * quantity;
        self.vega -= greeks.vega * quantity;
        self.rho -= greeks.rho * quantity;
    }

    /// Calculates the dollar delta (delta exposure in currency terms).
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying asset
    /// * `multiplier` - Contract multiplier (e.g., 100 for equity options)
    ///
    /// # Returns
    ///
    /// The dollar value of delta exposure.
    #[must_use]
    pub fn dollar_delta(&self, underlying_price: Decimal, multiplier: Decimal) -> Decimal {
        self.delta * underlying_price * multiplier
    }

    /// Calculates the dollar gamma (gamma exposure in currency terms).
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying asset
    /// * `multiplier` - Contract multiplier
    ///
    /// # Returns
    ///
    /// The dollar value of gamma exposure (P&L change for 1% move squared).
    #[must_use]
    pub fn dollar_gamma(&self, underlying_price: Decimal, multiplier: Decimal) -> Decimal {
        // Dollar gamma = 0.5 * gamma * S^2 * multiplier / 100
        // This gives the P&L impact for a 1% move in the underlying
        let one_pct_move = underlying_price * dec!(0.01);
        dec!(0.5) * self.gamma * one_pct_move * one_pct_move * multiplier
    }

    /// Calculates the dollar vega (vega exposure in currency terms).
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Contract multiplier
    ///
    /// # Returns
    ///
    /// The dollar value of vega exposure (P&L change for 1% vol move).
    #[must_use]
    pub fn dollar_vega(&self, multiplier: Decimal) -> Decimal {
        self.vega * multiplier
    }

    /// Calculates the dollar theta (daily time decay in currency terms).
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Contract multiplier
    ///
    /// # Returns
    ///
    /// The dollar value of daily theta decay.
    #[must_use]
    pub fn dollar_theta(&self, multiplier: Decimal) -> Decimal {
        self.theta * multiplier
    }

    /// Resets all Greeks to zero.
    pub fn reset(&mut self) {
        self.delta = Decimal::ZERO;
        self.gamma = Decimal::ZERO;
        self.theta = Decimal::ZERO;
        self.vega = Decimal::ZERO;
        self.rho = Decimal::ZERO;
    }

    /// Checks if the portfolio is approximately delta neutral.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum absolute delta to be considered neutral
    ///
    /// # Returns
    ///
    /// `true` if absolute delta is within tolerance.
    #[must_use]
    pub fn is_delta_neutral(&self, tolerance: Decimal) -> bool {
        self.delta.abs() <= tolerance
    }

    /// Calculates the number of underlying shares needed to delta hedge.
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Contract multiplier
    ///
    /// # Returns
    ///
    /// Number of shares to buy (positive) or sell (negative) to neutralize delta.
    #[must_use]
    pub fn shares_to_hedge(&self, multiplier: Decimal) -> Decimal {
        // Negative because we need to take opposite position
        -self.delta * multiplier
    }
}

impl std::ops::Add for PortfolioGreeks {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            delta: self.delta + other.delta,
            gamma: self.gamma + other.gamma,
            theta: self.theta + other.theta,
            vega: self.vega + other.vega,
            rho: self.rho + other.rho,
        }
    }
}

impl std::ops::AddAssign for PortfolioGreeks {
    fn add_assign(&mut self, other: Self) {
        self.delta += other.delta;
        self.gamma += other.gamma;
        self.theta += other.theta;
        self.vega += other.vega;
        self.rho += other.rho;
    }
}

impl std::ops::Sub for PortfolioGreeks {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            delta: self.delta - other.delta,
            gamma: self.gamma - other.gamma,
            theta: self.theta - other.theta,
            vega: self.vega - other.vega,
            rho: self.rho - other.rho,
        }
    }
}

impl std::ops::SubAssign for PortfolioGreeks {
    fn sub_assign(&mut self, other: Self) {
        self.delta -= other.delta;
        self.gamma -= other.gamma;
        self.theta -= other.theta;
        self.vega -= other.vega;
        self.rho -= other.rho;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_greeks_new() {
        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        assert_eq!(greeks.delta, dec!(0.5));
        assert_eq!(greeks.gamma, dec!(0.02));
        assert_eq!(greeks.theta, dec!(-0.05));
        assert_eq!(greeks.vega, dec!(0.15));
        assert_eq!(greeks.rho, dec!(0.08));
    }

    #[test]
    fn test_position_greeks_scale() {
        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        let scaled = greeks.scale(dec!(10.0));

        assert_eq!(scaled.delta, dec!(5.0));
        assert_eq!(scaled.gamma, dec!(0.2));
        assert_eq!(scaled.theta, dec!(-0.5));
        assert_eq!(scaled.vega, dec!(1.5));
        assert_eq!(scaled.rho, dec!(0.8));
    }

    #[test]
    fn test_portfolio_greeks_add() {
        let mut portfolio = PortfolioGreeks::new();

        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        portfolio.add(&greeks, dec!(10.0));

        assert_eq!(portfolio.delta, dec!(5.0));
        assert_eq!(portfolio.gamma, dec!(0.2));
        assert_eq!(portfolio.theta, dec!(-0.5));
        assert_eq!(portfolio.vega, dec!(1.5));
        assert_eq!(portfolio.rho, dec!(0.8));
    }

    #[test]
    fn test_portfolio_greeks_remove() {
        let mut portfolio = PortfolioGreeks::new();

        let greeks =
            PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

        portfolio.add(&greeks, dec!(10.0));
        portfolio.remove(&greeks, dec!(5.0));

        assert_eq!(portfolio.delta, dec!(2.5));
        assert_eq!(portfolio.gamma, dec!(0.1));
        assert_eq!(portfolio.theta, dec!(-0.25));
        assert_eq!(portfolio.vega, dec!(0.75));
        assert_eq!(portfolio.rho, dec!(0.4));
    }

    #[test]
    fn test_portfolio_dollar_delta() {
        let mut portfolio = PortfolioGreeks::new();
        portfolio.delta = dec!(10.0);

        let dollar_delta = portfolio.dollar_delta(dec!(100.0), dec!(100.0));

        assert_eq!(dollar_delta, dec!(100000.0)); // 10 * 100 * 100
    }

    #[test]
    fn test_portfolio_is_delta_neutral() {
        let mut portfolio = PortfolioGreeks::new();

        portfolio.delta = dec!(0.05);
        assert!(portfolio.is_delta_neutral(dec!(0.1)));
        assert!(!portfolio.is_delta_neutral(dec!(0.01)));

        portfolio.delta = dec!(-0.05);
        assert!(portfolio.is_delta_neutral(dec!(0.1)));
    }

    #[test]
    fn test_portfolio_shares_to_hedge() {
        let mut portfolio = PortfolioGreeks::new();
        portfolio.delta = dec!(10.0);

        let shares = portfolio.shares_to_hedge(dec!(100.0));

        assert_eq!(shares, dec!(-1000.0)); // Need to sell 1000 shares
    }

    #[test]
    fn test_portfolio_greeks_ops() {
        let p1 = PortfolioGreeks {
            delta: dec!(1.0),
            gamma: dec!(0.1),
            theta: dec!(-0.05),
            vega: dec!(0.2),
            rho: dec!(0.01),
        };

        let p2 = PortfolioGreeks {
            delta: dec!(2.0),
            gamma: dec!(0.2),
            theta: dec!(-0.10),
            vega: dec!(0.4),
            rho: dec!(0.02),
        };

        let sum = p1 + p2;
        assert_eq!(sum.delta, dec!(3.0));
        assert_eq!(sum.gamma, dec!(0.3));

        let diff = p2 - p1;
        assert_eq!(diff.delta, dec!(1.0));
        assert_eq!(diff.gamma, dec!(0.1));
    }

    #[test]
    fn test_portfolio_reset() {
        let mut portfolio = PortfolioGreeks {
            delta: dec!(10.0),
            gamma: dec!(1.0),
            theta: dec!(-0.5),
            vega: dec!(2.0),
            rho: dec!(0.1),
        };

        portfolio.reset();

        assert_eq!(portfolio.delta, Decimal::ZERO);
        assert_eq!(portfolio.gamma, Decimal::ZERO);
        assert_eq!(portfolio.theta, Decimal::ZERO);
        assert_eq!(portfolio.vega, Decimal::ZERO);
        assert_eq!(portfolio.rho, Decimal::ZERO);
    }
}
