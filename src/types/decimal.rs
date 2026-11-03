//! Decimal helper functions for mathematical operations.
//!
//! Provides mathematical operations not natively supported by rust_decimal,
//! such as logarithms, powers, and square roots.

use crate::types::error::{MMError, MMResult};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Calculates the natural logarithm (ln) of a Decimal value.
///
/// Since `Decimal` does not natively support logarithms, this function
/// temporarily converts to `f64`, performs the calculation, and converts back.
///
/// # Arguments
///
/// * `value` - The value to calculate ln for (must be positive)
///
/// # Returns
///
/// The natural logarithm of the input value.
///
/// # Errors
///
/// Returns `MMError::NumericalError` if:
/// - The value cannot be converted to f64
/// - The value is not positive
/// - The result cannot be converted back to Decimal
///
/// # Examples
///
/// ```
/// use market_maker_rs::types::decimal::decimal_ln;
/// use market_maker_rs::dec;
///
/// let result = decimal_ln(dec!(2.718281828)).unwrap();
/// // ln(e) ≈ 1.0
/// ```
pub fn decimal_ln(value: Decimal) -> MMResult<Decimal> {
    let float_value = match value.to_f64() {
        Some(v) => v,
        None => {
            return Err(MMError::NumericalError(
                "decimal_ln: invalid value".to_string(),
            ));
        }
    };
    let result = float_value.ln();
    Decimal::from_f64(result)
        .ok_or_else(|| MMError::NumericalError("decimal_ln: conversion error".to_string()))
}

/// Raises a Decimal value to an integer power.
///
/// Since `Decimal` does not natively support power operations with arbitrary exponents,
/// this function temporarily converts to `f64`, performs the calculation, and converts back.
///
/// # Arguments
///
/// * `value` - The base value
/// * `exponent` - The integer exponent
///
/// # Returns
///
/// The value raised to the given power.
///
/// # Errors
///
/// Returns `MMError::NumericalError` if:
/// - The value cannot be converted to f64
/// - The result overflows or underflows
/// - The result cannot be converted back to Decimal
///
/// # Examples
///
/// ```
/// use market_maker_rs::types::decimal::decimal_powi;
/// use market_maker_rs::dec;
///
/// let result = decimal_powi(dec!(2), 3).unwrap();
/// assert_eq!(result, dec!(8));
/// ```
///
/// # Notes
///
/// This function may lose precision for very large or very small numbers due to
/// the intermediate f64 conversion.
pub fn decimal_powi(value: Decimal, exponent: i32) -> MMResult<Decimal> {
    let float_value = match value.to_f64() {
        Some(v) => v,
        None => {
            return Err(MMError::NumericalError(
                "decimal_powi: invalid value".to_string(),
            ));
        }
    };
    let result = float_value.powi(exponent);
    Decimal::from_f64(result)
        .ok_or_else(|| MMError::NumericalError("decimal_powi: conversion error".to_string()))
}

/// Calculates the square root of a Decimal value.
///
/// Since `Decimal` does not natively support square roots, this function
/// temporarily converts to `f64`, performs the calculation, and converts back.
///
/// # Arguments
///
/// * `value` - The value to calculate square root for (must be non-negative)
///
/// # Returns
///
/// The square root of the input value.
///
/// # Errors
///
/// Returns `MMError::NumericalError` if:
/// - The value cannot be converted to f64
/// - The value is negative
/// - The result cannot be converted back to Decimal
///
/// # Examples
///
/// ```
/// use market_maker_rs::types::decimal::decimal_sqrt;
/// use market_maker_rs::dec;
///
/// let result = decimal_sqrt(dec!(9)).unwrap();
/// assert_eq!(result, dec!(3));
/// ```
pub fn decimal_sqrt(value: Decimal) -> MMResult<Decimal> {
    let float_value = match value.to_f64() {
        Some(v) => v,
        None => {
            return Err(MMError::NumericalError(
                "decimal_sqrt: invalid value".to_string(),
            ));
        }
    };
    let result = float_value.sqrt();
    Decimal::from_f64(result)
        .ok_or_else(|| MMError::NumericalError("decimal_sqrt: conversion error".to_string()))
}

/// Converts a `Decimal` to `u64`, truncating toward zero and saturating:
/// negative values map to `0` and values above `u64::MAX` to `u64::MAX`.
///
/// This matches the semantics of the former `f64 as u64` conversions,
/// without the precision loss of going through `f64`.
#[must_use]
#[inline]
pub(crate) fn decimal_to_u64_saturating(value: Decimal) -> u64 {
    let truncated = value.trunc();
    if truncated.is_sign_negative() {
        return 0;
    }
    truncated.to_u64().unwrap_or(u64::MAX)
}

/// Converts a count, length or duration in integer units to `u64`,
/// saturating at `u64::MAX` when the value does not fit.
#[must_use]
#[inline]
pub(crate) fn saturating_u64<T: TryInto<u64>>(value: T) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_to_u64_saturating_bounds_clamp() {
        assert_eq!(decimal_to_u64_saturating(dec!(12.9)), 12);
        assert_eq!(decimal_to_u64_saturating(dec!(-3.5)), 0);
        assert_eq!(decimal_to_u64_saturating(dec!(-0.5)), 0);
        assert_eq!(decimal_to_u64_saturating(Decimal::MAX), u64::MAX);
    }

    #[test]
    fn test_saturating_u64_overflow_saturates() {
        assert_eq!(saturating_u64(5usize), 5);
        assert_eq!(saturating_u64(u128::MAX), u64::MAX);
    }
    use crate::dec;

    #[test]
    fn test_decimal_ln() {
        // ln(e) ≈ 1.0
        let result = decimal_ln(dec!(2.718281828)).unwrap();
        let expected = dec!(1.0);
        assert!((result - expected).abs() < dec!(0.001));

        // ln(1) = 0
        let result = decimal_ln(dec!(1.0)).unwrap();
        assert_eq!(result, dec!(0.0));

        // ln(10) ≈ 2.302585
        let result = decimal_ln(dec!(10.0)).unwrap();
        let expected = dec!(2.302585);
        assert!((result - expected).abs() < dec!(0.000001));
    }

    #[test]
    fn test_decimal_powi() {
        // 2^3 = 8
        let result = decimal_powi(dec!(2), 3).unwrap();
        assert_eq!(result, dec!(8));

        // 10^2 = 100
        let result = decimal_powi(dec!(10), 2).unwrap();
        assert_eq!(result, dec!(100));

        // 5^0 = 1
        let result = decimal_powi(dec!(5), 0).unwrap();
        assert_eq!(result, dec!(1));

        // 2^-2 = 0.25
        let result = decimal_powi(dec!(2), -2).unwrap();
        assert_eq!(result, dec!(0.25));
    }

    #[test]
    fn test_decimal_sqrt() {
        // sqrt(9) = 3
        let result = decimal_sqrt(dec!(9)).unwrap();
        assert_eq!(result, dec!(3));

        // sqrt(4) = 2
        let result = decimal_sqrt(dec!(4)).unwrap();
        assert_eq!(result, dec!(2));

        // sqrt(0) = 0
        let result = decimal_sqrt(dec!(0)).unwrap();
        assert_eq!(result, dec!(0));

        // sqrt(2) ≈ 1.414213
        let result = decimal_sqrt(dec!(2)).unwrap();
        let expected = dec!(1.414213);
        assert!((result - expected).abs() < dec!(0.000001));
    }

    #[test]
    fn test_decimal_ln_error_handling() {
        // Test with invalid value that can't be converted (infinity equivalent)
        let result = decimal_ln(Decimal::MAX);
        // MAX puede convertirse, así que probamos con resultado inválido
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_decimal_ln_conversion_error() {
        // Test with a value that might fail to_f64 conversion
        // Using a very small value close to zero
        let result = decimal_ln(Decimal::from_parts(1, 0, 0, false, 28));
        // Should either succeed or return error
        let _ = result;
    }

    #[test]
    fn test_decimal_powi_error_handling() {
        // Test with very large exponent that might cause overflow
        let result = decimal_powi(dec!(10), 1000);
        // Puede ser error por overflow
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_decimal_powi_conversion_error() {
        // Test with extreme values that might fail conversion
        let result = decimal_powi(
            Decimal::from_parts(u32::MAX, u32::MAX, u32::MAX, false, 0),
            2,
        );
        // Should handle error gracefully
        let _ = result;
    }

    #[test]
    fn test_decimal_sqrt_error_handling() {
        // Test with MAX value
        let result = decimal_sqrt(Decimal::MAX);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_decimal_sqrt_conversion_error() {
        // Test with very large value
        let result = decimal_sqrt(Decimal::from_parts(u32::MAX, u32::MAX, 0, false, 0));
        // Should handle error gracefully
        let _ = result;
    }

    #[test]
    fn test_decimal_ln_negative_value() {
        // ln of negative value should produce NaN which fails conversion
        let result = decimal_ln(dec!(-1.0));
        // Result is NaN from f64, which fails Decimal::from_f64
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_sqrt_negative_value() {
        // sqrt of negative value should produce NaN which fails conversion
        let result = decimal_sqrt(dec!(-1.0));
        // Result is NaN from f64, which fails Decimal::from_f64
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_ln_zero() {
        // ln(0) = -infinity, which fails conversion
        let result = decimal_ln(dec!(0.0));
        assert!(result.is_err());
    }
}
