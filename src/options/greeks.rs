//! Portfolio Greeks aggregation and tracking.
//!
//! This module provides types for tracking and aggregating Greeks across
//! multiple options positions in a portfolio.
//!
//! Every computed figure is checked: an overflow is
//! [`MMError::NumericalError`](crate::types::error::MMError::NumericalError),
//! and an operation that fails leaves its receiver unchanged.

use crate::Decimal;
use crate::types::decimal::{OrOverflow, negate};
use crate::types::error::MMResult;
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when a scaled value overflows.
    pub fn scale(&self, quantity: Decimal) -> MMResult<Self> {
        Ok(Self {
            delta: self
                .delta
                .checked_mul(quantity)
                .or_overflow("scaled delta")?,
            gamma: self
                .gamma
                .checked_mul(quantity)
                .or_overflow("scaled gamma")?,
            theta: self
                .theta
                .checked_mul(quantity)
                .or_overflow("scaled theta")?,
            vega: self.vega.checked_mul(quantity).or_overflow("scaled vega")?,
            rho: self.rho.checked_mul(quantity).or_overflow("scaled rho")?,
        })
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
/// portfolio.add(&call_greeks, dec!(10.0)).unwrap(); // 10 contracts
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when a scaled value or a total overflows; the portfolio is unchanged.
    pub fn add(&mut self, greeks: &PositionGreeks, quantity: Decimal) -> MMResult<()> {
        *self = self.checked_add(Self::of_position(greeks.scale(quantity)?))?;
        Ok(())
    }

    /// Removes Greeks from a position from the portfolio.
    ///
    /// # Arguments
    ///
    /// * `greeks` - The Greeks of the position to remove
    /// * `quantity` - The position quantity that was removed
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when a scaled value or a total overflows; the portfolio is unchanged.
    pub fn remove(&mut self, greeks: &PositionGreeks, quantity: Decimal) -> MMResult<()> {
        *self = self.checked_sub(Self::of_position(greeks.scale(quantity)?))?;
        Ok(())
    }

    /// The field by field sum of two portfolios.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when a total overflows.
    pub fn checked_add(self, other: Self) -> MMResult<Self> {
        Ok(Self {
            delta: self
                .delta
                .checked_add(other.delta)
                .or_overflow("total delta")?,
            gamma: self
                .gamma
                .checked_add(other.gamma)
                .or_overflow("total gamma")?,
            theta: self
                .theta
                .checked_add(other.theta)
                .or_overflow("total theta")?,
            vega: self
                .vega
                .checked_add(other.vega)
                .or_overflow("total vega")?,
            rho: self.rho.checked_add(other.rho).or_overflow("total rho")?,
        })
    }

    /// The field by field difference of two portfolios.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when a total overflows.
    pub fn checked_sub(self, other: Self) -> MMResult<Self> {
        Ok(Self {
            delta: self
                .delta
                .checked_sub(other.delta)
                .or_overflow("total delta")?,
            gamma: self
                .gamma
                .checked_sub(other.gamma)
                .or_overflow("total gamma")?,
            theta: self
                .theta
                .checked_sub(other.theta)
                .or_overflow("total theta")?,
            vega: self
                .vega
                .checked_sub(other.vega)
                .or_overflow("total vega")?,
            rho: self.rho.checked_sub(other.rho).or_overflow("total rho")?,
        })
    }

    /// The Greeks of one position as a portfolio.
    fn of_position(position: PositionGreeks) -> Self {
        Self {
            delta: position.delta,
            gamma: position.gamma,
            theta: position.theta,
            vega: position.vega,
            rho: position.rho,
        }
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_delta(
        &self,
        underlying_price: Decimal,
        multiplier: Decimal,
    ) -> MMResult<Decimal> {
        self.delta
            .checked_mul(underlying_price)
            .and_then(|x| x.checked_mul(multiplier))
            .or_overflow("dollar delta")
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_gamma(
        &self,
        underlying_price: Decimal,
        multiplier: Decimal,
    ) -> MMResult<Decimal> {
        // Dollar gamma = 0.5 * gamma * S^2 * multiplier / 100
        // This gives the P&L impact for a 1% move in the underlying
        let one_pct_move = underlying_price.checked_mul(dec!(0.01));
        one_pct_move
            .and_then(|m| {
                dec!(0.5)
                    .checked_mul(self.gamma)?
                    .checked_mul(m)?
                    .checked_mul(m)?
                    .checked_mul(multiplier)
            })
            .or_overflow("dollar gamma")
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_vega(&self, multiplier: Decimal) -> MMResult<Decimal> {
        self.vega.checked_mul(multiplier).or_overflow("dollar vega")
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the exposure overflows.
    pub fn dollar_theta(&self, multiplier: Decimal) -> MMResult<Decimal> {
        self.theta
            .checked_mul(multiplier)
            .or_overflow("dollar theta")
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`](crate::types::error::MMError::NumericalError)
    /// when the number of shares overflows.
    pub fn shares_to_hedge(&self, multiplier: Decimal) -> MMResult<Decimal> {
        // Negative because we need to take opposite position
        negate(self.delta)
            .checked_mul(multiplier)
            .or_overflow("shares to hedge")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::error::MMError;

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

        let scaled = greeks.scale(dec!(10.0)).unwrap();

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

        portfolio.add(&greeks, dec!(10.0)).unwrap();

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

        portfolio.add(&greeks, dec!(10.0)).unwrap();
        portfolio.remove(&greeks, dec!(5.0)).unwrap();

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

        let dollar_delta = portfolio.dollar_delta(dec!(100.0), dec!(100.0)).unwrap();

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

        let shares = portfolio.shares_to_hedge(dec!(100.0)).unwrap();

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

        let sum = p1.checked_add(p2).unwrap();
        assert_eq!(sum.delta, dec!(3.0));
        assert_eq!(sum.gamma, dec!(0.3));

        let diff = p2.checked_sub(p1).unwrap();
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

    fn overflow(what: &str) -> MMError {
        MMError::NumericalError(format!("{what} overflows"))
    }

    /// One position per field, with only that field at the bound.
    fn each_greek_at_max() -> [(&'static str, PositionGreeks); 5] {
        let zero = PositionGreeks::zero();
        [
            (
                "delta",
                PositionGreeks {
                    delta: Decimal::MAX,
                    ..zero
                },
            ),
            (
                "gamma",
                PositionGreeks {
                    gamma: Decimal::MAX,
                    ..zero
                },
            ),
            (
                "theta",
                PositionGreeks {
                    theta: Decimal::MAX,
                    ..zero
                },
            ),
            (
                "vega",
                PositionGreeks {
                    vega: Decimal::MAX,
                    ..zero
                },
            ),
            (
                "rho",
                PositionGreeks {
                    rho: Decimal::MAX,
                    ..zero
                },
            ),
        ]
    }

    #[test]
    fn test_position_greeks_scale_overflow_names_the_field() {
        for (name, greeks) in each_greek_at_max() {
            assert_eq!(
                greeks.scale(dec!(2)),
                Err(overflow(&format!("scaled {name}")))
            );
        }
    }

    #[test]
    fn test_portfolio_add_scale_overflow_leaves_portfolio_unchanged() {
        let start = PortfolioGreeks {
            delta: dec!(1),
            ..PortfolioGreeks::new()
        };
        for (name, greeks) in each_greek_at_max() {
            let expected = Err(overflow(&format!("scaled {name}")));
            let mut portfolio = start;
            assert_eq!(portfolio.add(&greeks, dec!(2)), expected);
            assert_eq!(portfolio, start);
            assert_eq!(portfolio.remove(&greeks, dec!(2)), expected);
            assert_eq!(portfolio, start);
        }
    }

    #[test]
    fn test_portfolio_add_total_overflow_leaves_portfolio_unchanged() {
        // Delta fits; the overflow of the last field must not leave it applied.
        let mut portfolio = PortfolioGreeks {
            delta: dec!(1),
            rho: Decimal::MAX,
            ..PortfolioGreeks::new()
        };
        let start = portfolio;
        let greeks = PositionGreeks::new(dec!(1), dec!(0), dec!(0), dec!(0), Decimal::MAX);
        assert_eq!(portfolio.add(&greeks, dec!(1)), Err(overflow("total rho")));
        assert_eq!(portfolio, start);
    }

    #[test]
    fn test_portfolio_remove_total_overflow_leaves_portfolio_unchanged() {
        let mut portfolio = PortfolioGreeks {
            delta: dec!(1),
            rho: Decimal::MIN,
            ..PortfolioGreeks::new()
        };
        let start = portfolio;
        let greeks = PositionGreeks::new(dec!(1), dec!(0), dec!(0), dec!(0), Decimal::MAX);
        assert_eq!(
            portfolio.remove(&greeks, dec!(1)),
            Err(overflow("total rho"))
        );
        assert_eq!(portfolio, start);
    }

    #[test]
    fn test_portfolio_checked_ops_overflow_names_the_field() {
        for (name, greeks) in each_greek_at_max() {
            let high = PortfolioGreeks::of_position(greeks);
            let low = PortfolioGreeks::new().checked_sub(high).unwrap();
            let expected = Err(overflow(&format!("total {name}")));
            assert_eq!(high.checked_add(high), expected);
            assert_eq!(low.checked_add(low), expected);
            assert_eq!(high.checked_sub(low), expected);
            assert_eq!(low.checked_sub(high), expected);
        }
    }

    #[test]
    fn test_portfolio_dollar_values_overflow() {
        let max = PortfolioGreeks {
            delta: Decimal::MAX,
            gamma: Decimal::MAX,
            theta: Decimal::MAX,
            vega: Decimal::MAX,
            rho: dec!(0),
        };
        assert_eq!(
            max.dollar_delta(dec!(2), dec!(1)),
            Err(overflow("dollar delta"))
        );
        assert_eq!(
            max.dollar_delta(dec!(1), dec!(2)),
            Err(overflow("dollar delta"))
        );
        assert_eq!(max.dollar_vega(dec!(2)), Err(overflow("dollar vega")));
        assert_eq!(max.dollar_theta(dec!(2)), Err(overflow("dollar theta")));
    }

    #[test]
    fn test_portfolio_dollar_gamma_overflow_at_each_step() {
        let portfolio = |gamma| PortfolioGreeks {
            gamma,
            ..PortfolioGreeks::new()
        };
        let expected = Err(overflow("dollar gamma"));
        // The first product with the 1% move: 5e19 * 1e10.
        assert_eq!(
            portfolio(dec!(1e20)).dollar_gamma(dec!(1e12), dec!(1)),
            expected
        );
        // The second: 1 * 1e15 * 1e15.
        assert_eq!(
            portfolio(dec!(2)).dollar_gamma(dec!(1e17), dec!(1)),
            expected
        );
        // The product with the multiplier: 1 * 1 * 1 * 2 MAX.
        assert_eq!(
            portfolio(dec!(2)).dollar_gamma(dec!(100), Decimal::MAX),
            Ok(Decimal::MAX)
        );
        assert_eq!(
            portfolio(dec!(4)).dollar_gamma(dec!(100), Decimal::MAX),
            expected
        );
    }

    #[test]
    fn test_portfolio_dollar_gamma_unchanged_on_ordinary_input() {
        let portfolio = PortfolioGreeks {
            gamma: dec!(0.04),
            ..PortfolioGreeks::new()
        };
        // 0.5 * 0.04 * 1 * 1 * 100
        assert_eq!(
            portfolio.dollar_gamma(dec!(100), dec!(100)).unwrap(),
            dec!(2)
        );
    }

    #[test]
    fn test_portfolio_shares_to_hedge_overflow() {
        let portfolio = PortfolioGreeks {
            delta: Decimal::MAX,
            ..PortfolioGreeks::new()
        };
        assert_eq!(
            portfolio.shares_to_hedge(dec!(2)),
            Err(overflow("shares to hedge"))
        );
        // The bound itself negates exactly.
        assert_eq!(portfolio.shares_to_hedge(dec!(1)), Ok(Decimal::MIN));
    }
}
