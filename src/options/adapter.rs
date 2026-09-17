//! Adapter functions for OptionStratLib integration.
//!
//! This module provides adapter functions to convert between `market-maker-rs`
//! types and `optionstratlib` types, as well as utility functions for
//! calculating Greeks and theoretical values.

use crate::Decimal;
use crate::options::greeks::PositionGreeks;
use crate::types::error::{MMError, MMResult};
use optionstratlib::greeks::{delta, gamma, rho, theta, vega};
use optionstratlib::model::ExpirationDate;
use optionstratlib::model::option::Options;
use optionstratlib::pricing::black_scholes_model::black_scholes;
use rust_decimal_macros::dec;

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
    /// Time to expiration in milliseconds. Returns 0 if conversion fails.
    #[must_use]
    pub fn time_to_terminal_ms(expiration: &ExpirationDate) -> u64 {
        let years = match expiration.get_years() {
            Ok(y) => y.to_dec(),
            Err(_) => return 0,
        };
        // Convert years to milliseconds: years * 365.25 * 24 * 60 * 60 * 1000
        let ms_per_year = dec!(31557600000.0); // 365.25 * 24 * 60 * 60 * 1000
        let ms = years * ms_per_year;

        // Convert to u64, handling potential overflow
        ms.to_string().parse::<f64>().map(|f| f as u64).unwrap_or(0)
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
        let ms_decimal = Decimal::from(ms);
        let ms_per_year = dec!(31557600000.0); // 365.25 * 24 * 60 * 60 * 1000
        ms_decimal / ms_per_year
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
    #[must_use]
    pub fn moneyness(underlying_price: Decimal, strike_price: Decimal) -> Decimal {
        if strike_price.is_zero() {
            return Decimal::ZERO;
        }
        underlying_price / strike_price
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
    #[must_use]
    pub fn is_atm(underlying_price: Decimal, strike_price: Decimal, tolerance: Decimal) -> bool {
        let moneyness = Self::moneyness(underlying_price, strike_price);
        (moneyness - dec!(1.0)).abs() <= tolerance
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

        let intrinsic = match option.option_style {
            optionstratlib::OptionStyle::Call => underlying - strike,
            optionstratlib::OptionStyle::Put => strike - underlying,
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
        let time_val = market_price - intrinsic;

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
        let ms = OptionsAdapter::time_to_terminal_ms(&expiration);

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
            OptionsAdapter::moneyness(dec!(100.0), dec!(100.0)),
            dec!(1.0)
        );

        // ITM call (underlying > strike)
        assert!(OptionsAdapter::moneyness(dec!(110.0), dec!(100.0)) > dec!(1.0));

        // OTM call (underlying < strike)
        assert!(OptionsAdapter::moneyness(dec!(90.0), dec!(100.0)) < dec!(1.0));
    }

    #[test]
    fn test_is_atm() {
        // Exactly ATM
        assert!(OptionsAdapter::is_atm(dec!(100.0), dec!(100.0), dec!(0.02)));

        // Within 2% tolerance
        assert!(OptionsAdapter::is_atm(dec!(101.0), dec!(100.0), dec!(0.02)));
        assert!(OptionsAdapter::is_atm(dec!(99.0), dec!(100.0), dec!(0.02)));

        // Outside tolerance
        assert!(!OptionsAdapter::is_atm(
            dec!(105.0),
            dec!(100.0),
            dec!(0.02)
        ));
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
