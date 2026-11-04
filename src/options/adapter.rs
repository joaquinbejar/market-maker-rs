//! Adapter functions for OptionStratLib integration.
//!
//! This module provides adapter functions to convert between `market-maker-rs`
//! types and `optionstratlib` types, as well as utility functions for
//! calculating Greeks and theoretical values.

use crate::Decimal;
use crate::options::greeks::PositionGreeks;
use crate::types::decimal::{OrOverflow, saturating_div};
use crate::types::error::{MMError, MMResult};
use optionstratlib::greeks::{delta, gamma, rho, theta, vega};
use optionstratlib::model::ExpirationDate;
use optionstratlib::model::option::Options;
use optionstratlib::pricing::black_scholes_model::black_scholes;
use rust_decimal_macros::dec;

/// Milliseconds in the year used by the ms helpers of [`OptionsAdapter`]
/// (365.25 days). Option valuation in [`crate::options::pricing`] uses
/// [`crate::options::pricing::PRICING_YEAR_DAYS`] (365 days) instead; the
/// two are different conventions and are not interchangeable.
pub const ADAPTER_MS_PER_YEAR: Decimal = dec!(31557600000);

/// Adapter for OptionStratLib integration.
///
/// Provides static methods for calculating Greeks, theoretical values,
/// and converting between library types.
///
/// # Example
///
/// ```rust,ignore
/// use market_maker_rs::options::OptionsAdapter;
/// use optionstratlib::{Options, ExpirationDate, OptionStyle};
/// use optionstratlib::prelude::pos_or_panic;
///
/// let option = Options::new(/* ... */);
/// let greeks = OptionsAdapter::calculate_greeks(&option)?;
/// let theo = OptionsAdapter::theoretical_value(&option)?;
/// ```
pub struct OptionsAdapter;

impl OptionsAdapter {
    /// Calculates Greeks for an option using OptionStratLib.
    ///
    /// # Arguments
    ///
    /// * `option` - The option contract from OptionStratLib
    ///
    /// # Returns
    ///
    /// `PositionGreeks` containing delta, gamma, theta, vega, and rho.
    ///
    /// # Errors
    ///
    /// Returns `MMError::NumericalError` if Greeks calculation fails.
    pub fn calculate_greeks(option: &Options) -> MMResult<PositionGreeks> {
        let d = delta(option).map_err(|e: optionstratlib::error::GreeksError| {
            MMError::NumericalError(e.to_string())
        })?;

        let g = gamma(option).map_err(|e: optionstratlib::error::GreeksError| {
            MMError::NumericalError(e.to_string())
        })?;

        let t = theta(option).map_err(|e: optionstratlib::error::GreeksError| {
            MMError::NumericalError(e.to_string())
        })?;

        let v = vega(option).map_err(|e: optionstratlib::error::GreeksError| {
            MMError::NumericalError(e.to_string())
        })?;

        let r = rho(option).map_err(|e: optionstratlib::error::GreeksError| {
            MMError::NumericalError(e.to_string())
        })?;

        Ok(PositionGreeks::new(d, g, t, v, r))
    }

    /// Calculates the theoretical value of an option using Black-Scholes.
    ///
    /// # Arguments
    ///
    /// * `option` - The option contract from OptionStratLib
    ///
    /// # Returns
    ///
    /// The theoretical price of the option.
    ///
    /// # Errors
    ///
    /// Returns `MMError::NumericalError` if pricing fails.
    pub fn theoretical_value(option: &Options) -> MMResult<Decimal> {
        black_scholes(option).map_err(|e: optionstratlib::error::PricingError| {
            MMError::NumericalError(e.to_string())
        })
    }

    /// Converts ExpirationDate to time remaining in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `expiration` - The expiration date from OptionStratLib
    ///
    /// # Returns
    ///
    /// Time to expiration in milliseconds.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the expiration cannot be
    /// converted to years: zero would read as already expired.
    pub fn time_to_terminal_ms(expiration: &ExpirationDate) -> MMResult<u64> {
        let years = expiration
            .get_years()
            .map_err(|e| MMError::NumericalError(format!("expiration in years: {e}")))?
            .to_dec();
        // Years to milliseconds at 365.25 days a year; an overflow saturates,
        // like the conversion to u64.
        Ok(match years.checked_mul(ADAPTER_MS_PER_YEAR) {
            Some(ms) => crate::types::decimal::decimal_to_u64_saturating(ms),
            None => u64::MAX,
        })
    }

    /// Converts milliseconds to years for use with OptionStratLib.
    ///
    /// # Arguments
    ///
    /// * `ms` - Time in milliseconds
    ///
    /// # Returns
    ///
    /// Time in years as Decimal.
    #[must_use]
    pub fn ms_to_years(ms: u64) -> Decimal {
        // A u64 divided by a positive constant above 1 cannot overflow.
        saturating_div(Decimal::from(ms), ADAPTER_MS_PER_YEAR)
    }

    /// Calculates the moneyness of an option.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying
    /// * `strike_price` - Strike price of the option
    ///
    /// # Returns
    ///
    /// Moneyness ratio (underlying / strike). Values > 1 indicate ITM for calls.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero strike, which has no moneyness,
    /// and when the ratio overflows.
    pub fn moneyness(underlying_price: Decimal, strike_price: Decimal) -> MMResult<Decimal> {
        if strike_price.is_zero() {
            return Err(MMError::NumericalError(
                "moneyness of a zero strike is undefined".to_string(),
            ));
        }
        underlying_price
            .checked_div(strike_price)
            .or_overflow("moneyness")
    }

    /// Determines if an option is at-the-money within a tolerance.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying
    /// * `strike_price` - Strike price of the option
    /// * `tolerance` - Percentage tolerance (e.g., 0.02 for 2%)
    ///
    /// # Returns
    ///
    /// `true` if the option is within the ATM tolerance.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when [`Self::moneyness`] fails or the
    /// distance from 1 overflows.
    pub fn is_atm(
        underlying_price: Decimal,
        strike_price: Decimal,
        tolerance: Decimal,
    ) -> MMResult<bool> {
        let moneyness = Self::moneyness(underlying_price, strike_price)?;
        let distance = moneyness
            .checked_sub(dec!(1.0))
            .or_overflow("distance from at the money")?;
        Ok(distance.abs() <= tolerance)
    }

    /// Calculates the intrinsic value of an option.
    ///
    /// # Arguments
    ///
    /// * `option` - The option contract
    ///
    /// # Returns
    ///
    /// The intrinsic value (always >= 0).
    #[must_use]
    pub fn intrinsic_value(option: &Options) -> Decimal {
        let underlying = option.underlying_price.to_dec();
        let strike = option.strike_price.to_dec();

        // Both are non negative, so their difference is representable.
        let intrinsic = match option.option_style {
            optionstratlib::OptionStyle::Call => underlying.saturating_sub(strike),
            optionstratlib::OptionStyle::Put => strike.saturating_sub(underlying),
        };

        if intrinsic > Decimal::ZERO {
            intrinsic
        } else {
            Decimal::ZERO
        }
    }

    /// Calculates the time value of an option.
    ///
    /// # Arguments
    ///
    /// * `option` - The option contract
    /// * `market_price` - Current market price of the option
    ///
    /// # Returns
    ///
    /// The time value (market price - intrinsic value).
    #[must_use]
    pub fn time_value(option: &Options, market_price: Decimal) -> Decimal {
        let intrinsic = Self::intrinsic_value(option);
        // The intrinsic value is non negative, so this can only saturate at
        // the negative bound, which the floor below turns into the same zero.
        let time_val = market_price.saturating_sub(intrinsic);

        if time_val > Decimal::ZERO {
            time_val
        } else {
            Decimal::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optionstratlib::model::types::{OptionStyle, OptionType, Side};
    use optionstratlib::prelude::pos_or_panic;

    fn create_test_option() -> Options {
        Options::new(
            OptionType::European,
            Side::Long,
            "TEST".to_string(),
            pos_or_panic!(100.0),                      // strike
            ExpirationDate::Days(pos_or_panic!(30.0)), // 30 days to expiry
            pos_or_panic!(0.2),                        // 20% IV
            pos_or_panic!(1.0),                        // quantity
            pos_or_panic!(100.0),                      // underlying price (ATM)
            dec!(0.05),                                // 5% risk-free rate
            OptionStyle::Call,
            pos_or_panic!(0.0), // no dividend
            None,
        )
    }

    #[test]
    fn test_calculate_greeks() {
        let option = create_test_option();
        let greeks = OptionsAdapter::calculate_greeks(&option).unwrap();

        // ATM call should have delta around 0.5
        assert!(greeks.delta > dec!(0.4) && greeks.delta < dec!(0.6));
        // Gamma should be positive
        assert!(greeks.gamma > Decimal::ZERO);
        // Theta should be negative for long options
        assert!(greeks.theta < Decimal::ZERO);
        // Vega should be positive
        assert!(greeks.vega > Decimal::ZERO);
    }

    #[test]
    fn test_theoretical_value() {
        let option = create_test_option();
        let theo = OptionsAdapter::theoretical_value(&option).unwrap();

        // ATM option should have positive value
        assert!(theo > Decimal::ZERO);
        // ATM 30-day option with 20% vol should be roughly 2-4% of underlying
        assert!(theo > dec!(2.0) && theo < dec!(6.0));
    }

    #[test]
    fn test_time_to_terminal_ms() {
        let expiration = ExpirationDate::Days(pos_or_panic!(365.0));
        let ms = OptionsAdapter::time_to_terminal_ms(&expiration).unwrap();

        // Should be approximately 1 year in milliseconds
        let expected_ms: u64 = 31_557_600_000; // 365.25 days
        let tolerance: u64 = 1_000_000_000; // 1000 seconds tolerance

        assert!((ms as i64 - expected_ms as i64).unsigned_abs() < tolerance);
    }

    #[test]
    fn test_ms_to_years() {
        let ms: u64 = 31_557_600_000; // 1 year in ms
        let years = OptionsAdapter::ms_to_years(ms);

        assert!((years - dec!(1.0)).abs() < dec!(0.01));
    }

    #[test]
    fn test_moneyness() {
        // ATM
        assert_eq!(
            OptionsAdapter::moneyness(dec!(100.0), dec!(100.0)).unwrap(),
            dec!(1.0)
        );

        // ITM call (underlying > strike)
        assert!(OptionsAdapter::moneyness(dec!(110.0), dec!(100.0)).unwrap() > dec!(1.0));

        // OTM call (underlying < strike)
        assert!(OptionsAdapter::moneyness(dec!(90.0), dec!(100.0)).unwrap() < dec!(1.0));
    }

    #[test]
    fn test_moneyness_overflow_is_a_typed_error() {
        assert_eq!(
            OptionsAdapter::moneyness(Decimal::MAX, Decimal::new(1, 28)),
            Err(MMError::NumericalError("moneyness overflows".to_string()))
        );
    }

    #[test]
    fn test_moneyness_zero_strike_is_a_typed_error() {
        assert_eq!(
            OptionsAdapter::moneyness(dec!(100), Decimal::ZERO),
            Err(MMError::NumericalError(
                "moneyness of a zero strike is undefined".to_string()
            ))
        );
    }

    #[test]
    fn test_is_atm_propagates_moneyness_errors() {
        assert!(OptionsAdapter::is_atm(Decimal::MAX, Decimal::new(1, 28), dec!(0.02)).is_err());
        assert!(OptionsAdapter::is_atm(dec!(100), Decimal::ZERO, dec!(0.02)).is_err());
    }

    #[test]
    fn test_is_atm_distance_overflow_is_a_typed_error() {
        // Moneyness at the negative bound: one below it is not representable.
        assert_eq!(
            OptionsAdapter::is_atm(Decimal::MIN, dec!(1), dec!(0.02)),
            Err(MMError::NumericalError(
                "distance from at the money overflows".to_string()
            ))
        );
    }

    #[test]
    fn test_ms_to_years_at_the_bound() {
        let years = OptionsAdapter::ms_to_years(u64::MAX);
        assert!(years > dec!(584000000) && years < dec!(585000000));
    }

    #[test]
    fn test_intrinsic_and_time_value_at_the_bounds() {
        let mut option = create_test_option();
        option.underlying_price = optionstratlib::prelude::Positive::MAX;
        option.strike_price = pos_or_panic!(100.0);
        let intrinsic = OptionsAdapter::intrinsic_value(&option);
        assert!(intrinsic > dec!(1e27));

        // A deeply negative price is floored at zero, not an overflow.
        assert_eq!(
            OptionsAdapter::time_value(&option, Decimal::MIN),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_is_atm() {
        // Exactly ATM
        assert!(OptionsAdapter::is_atm(dec!(100.0), dec!(100.0), dec!(0.02)).unwrap());

        // Within 2% tolerance
        assert!(OptionsAdapter::is_atm(dec!(101.0), dec!(100.0), dec!(0.02)).unwrap());
        assert!(OptionsAdapter::is_atm(dec!(99.0), dec!(100.0), dec!(0.02)).unwrap());

        // Outside tolerance
        assert!(!OptionsAdapter::is_atm(dec!(105.0), dec!(100.0), dec!(0.02)).unwrap());
    }

    #[test]
    fn test_intrinsic_value_call() {
        let mut option = create_test_option();

        // ATM - no intrinsic value
        assert_eq!(OptionsAdapter::intrinsic_value(&option), Decimal::ZERO);

        // ITM call
        option.underlying_price = pos_or_panic!(110.0);
        assert_eq!(OptionsAdapter::intrinsic_value(&option), dec!(10.0));

        // OTM call
        option.underlying_price = pos_or_panic!(90.0);
        assert_eq!(OptionsAdapter::intrinsic_value(&option), Decimal::ZERO);
    }

    #[test]
    fn test_intrinsic_value_put() {
        let mut option = create_test_option();
        option.option_style = OptionStyle::Put;

        // ATM - no intrinsic value
        assert_eq!(OptionsAdapter::intrinsic_value(&option), Decimal::ZERO);

        // ITM put
        option.underlying_price = pos_or_panic!(90.0);
        assert_eq!(OptionsAdapter::intrinsic_value(&option), dec!(10.0));

        // OTM put
        option.underlying_price = pos_or_panic!(110.0);
        assert_eq!(OptionsAdapter::intrinsic_value(&option), Decimal::ZERO);
    }

    #[test]
    fn test_time_value() {
        let option = create_test_option();

        // ATM option - all value is time value
        let market_price = dec!(3.0);
        let time_val = OptionsAdapter::time_value(&option, market_price);
        assert_eq!(time_val, dec!(3.0));
    }
}
