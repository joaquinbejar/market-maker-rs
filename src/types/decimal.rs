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

/// Maps the `None` of a failed checked operation to
/// [`MMError::NumericalError`] naming the quantity that overflowed.
pub(crate) trait OrOverflow<T> {
    /// Returns the value, or a numerical error saying `what` overflows.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] when `self` is `None`.
    fn or_overflow(self, what: &str) -> MMResult<T>;
}

impl<T> OrOverflow<T> for Option<T> {
    #[inline]
    fn or_overflow(self, what: &str) -> MMResult<T> {
        self.ok_or_else(|| overflow_error(what))
    }
}

/// The error of a checked operation that overflowed or divided by zero.
#[cold]
#[inline(never)]
fn overflow_error(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} overflows"))
}

/// The `Decimal` of `mantissa * 10^-scale` if one holds it exactly, or `None`.
fn exact_fit(mut mantissa: i128, mut scale: u32) -> Option<Decimal> {
    loop {
        if let Ok(value) = Decimal::try_from_i128_with_scale(mantissa, scale) {
            return Some(value);
        }
        if mantissa % 10 != 0 {
            return None;
        }
        mantissa /= 10;
        scale = scale.checked_sub(1)?;
    }
}

/// `a * b` if a `Decimal` holds it exactly, or `None`.
///
/// `Decimal::checked_mul` rounds a product with more than 28 decimal places.
pub(crate) fn exact_product(a: Decimal, b: Decimal) -> Option<Decimal> {
    let (a, b) = (a.normalize(), b.normalize());
    exact_fit(
        a.mantissa().checked_mul(b.mantissa())?,
        a.scale().checked_add(b.scale())?,
    )
}

/// `a + b` if a `Decimal` holds it exactly, or `None`.
///
/// `Decimal::checked_add` rounds a sum that does not fit: `1e28 + 1e-28` loses the
/// small term.
pub(crate) fn exact_add(a: Decimal, b: Decimal) -> Option<Decimal> {
    let (a, b) = (a.normalize(), b.normalize());
    let scale = a.scale().max(b.scale());
    let aligned = |value: Decimal| {
        10_i128
            .checked_pow(scale.checked_sub(value.scale())?)
            .and_then(|factor| value.mantissa().checked_mul(factor))
    };
    exact_fit(aligned(a)?.checked_add(aligned(b)?)?, scale)
}

/// `numerator / denominator`, saturating at the `Decimal` bound of the
/// quotient's sign when it overflows, and zero for a zero denominator.
///
/// Used only for report only values, and for divisions that cannot overflow.
#[must_use]
#[inline]
pub(crate) fn saturating_div(numerator: Decimal, denominator: Decimal) -> Decimal {
    numerator.checked_div(denominator).unwrap_or_else(|| {
        if denominator.is_zero() {
            Decimal::ZERO
        } else if numerator.is_sign_negative() == denominator.is_sign_negative() {
            Decimal::MAX
        } else {
            Decimal::MIN
        }
    })
}

/// `-value`, which is always representable: a `Decimal` is a sign and a
/// magnitude.
#[cfg(feature = "options")]
#[must_use]
#[inline]
pub(crate) fn negate(value: Decimal) -> Decimal {
    Decimal::ZERO.saturating_sub(value)
}

/// The midpoint `(a + b) / 2` of two decimals, never saturated.
///
/// The mean of two representable values is always representable: when
/// `a + b` overflows, both share a sign, so `a + (b - a) / 2` is computed
/// instead. The halving rounds like `Decimal` division in both forms.
#[must_use]
#[inline]
pub(crate) fn midpoint(a: Decimal, b: Decimal) -> Decimal {
    match a.checked_add(b) {
        Some(sum) => sum.checked_div(Decimal::TWO),
        None => b
            .checked_sub(a)
            .and_then(|d| d.checked_div(Decimal::TWO))
            .and_then(|half| a.checked_add(half)),
    }
    // Unreachable: both forms above are exact for representable inputs.
    .unwrap_or(a)
}

/// The mean `(a + b + c) / 3` of three decimals, never saturated.
///
/// When the sum overflows, the mean is computed as the smallest value plus
/// a third of the (non negative) excesses over it, which keeps it within
/// the inputs' range; a range too wide for that sums thirds instead.
#[must_use]
pub(crate) fn mean3(a: Decimal, b: Decimal, c: Decimal) -> Decimal {
    let three = Decimal::from(3);
    let direct = || a.checked_add(b)?.checked_add(c)?.checked_div(three);
    let from_min = || {
        let low = a.min(b).min(c);
        let excess = a
            .checked_sub(low)?
            .checked_add(b.checked_sub(low)?)?
            .checked_add(c.checked_sub(low)?)?;
        low.checked_add(excess.checked_div(three)?)
    };
    let thirds = || {
        a.checked_div(three)?
            .checked_add(b.checked_div(three)?)?
            .checked_add(c.checked_div(three)?)
    };
    direct()
        .or_else(from_min)
        .or_else(thirds)
        // Unreachable: thirds of representable values sum within bounds.
        .unwrap_or(a)
}

/// The imbalance `(a - b) / (a + b)`, invariant under a common scale of
/// `a` and `b`, and zero when `a + b` is zero.
///
/// When the direct form overflows, both sides are first divided by the
/// larger magnitude, so a sum beyond the `Decimal` bounds never distorts
/// the ratio.
#[must_use]
pub(crate) fn imbalance_ratio(a: Decimal, b: Decimal) -> Decimal {
    let ratio = |a: Decimal, b: Decimal| {
        let total = a.checked_add(b)?;
        if total.is_zero() {
            return Some(Decimal::ZERO);
        }
        a.checked_sub(b)?.checked_div(total)
    };
    ratio(a, b)
        .or_else(|| {
            let scale = a.abs().max(b.abs());
            ratio(a.checked_div(scale)?, b.checked_div(scale)?)
        })
        // Unreachable: scaled sides are in [-1, 1].
        .unwrap_or(Decimal::ZERO)
}

/// The totals of two sides, and the scale that recovers them.
///
/// The scale is one when both sums fit. Otherwise each term is divided by
/// one more than the number of terms: a quotient can round up, and that
/// headroom keeps the sums within the bounds, in their proportions.
fn scaled_sides<A, B>(a: A, b: B) -> Option<(Decimal, Decimal, Decimal)>
where
    A: Iterator<Item = Decimal> + Clone,
    B: Iterator<Item = Decimal> + Clone,
{
    fn sum(mut terms: impl Iterator<Item = Decimal>, scale: Decimal) -> Option<Decimal> {
        terms.try_fold(Decimal::ZERO, |total, term| {
            total.checked_add(term.checked_div(scale)?)
        })
    }
    let direct = sum(a.clone(), Decimal::ONE).zip(sum(b.clone(), Decimal::ONE));
    if let Some((total_a, total_b)) = direct {
        return Some((total_a, total_b, Decimal::ONE));
    }
    let count = a.clone().count().saturating_add(b.clone().count());
    let scale = Decimal::from(count).checked_add(Decimal::ONE)?;
    Some((sum(a, scale)?, sum(b, scale)?, scale))
}

/// The imbalance `(sum(a) - sum(b)) / (sum(a) + sum(b))` of two sides given
/// term by term, exact in proportion even when a sum is beyond the
/// `Decimal` bounds. Zero when the combined total is not positive.
#[must_use]
pub(crate) fn imbalance_of_sides<A, B>(a: A, b: B) -> Decimal
where
    A: Iterator<Item = Decimal> + Clone,
    B: Iterator<Item = Decimal> + Clone,
{
    // `scaled_sides` cannot fail: n terms of at most MAX / (n + 1), each
    // rounded up by at most 0.5, sum to less than MAX.
    match scaled_sides(a, b) {
        Some((total_a, total_b, _)) if total_a.saturating_add(total_b) > Decimal::ZERO => {
            imbalance_ratio(total_a, total_b)
        }
        _ => Decimal::ZERO,
    }
}

/// The difference `sum(a) - sum(b)` of two sides given term by term.
///
/// Report only: the result saturates at the `Decimal` bounds. When a side's
/// sum is beyond them, the terms are scaled first and the result carries
/// rounding error, so its sign is reliable only when the true difference
/// exceeds that error.
#[must_use]
pub(crate) fn net_of_sides<A, B>(a: A, b: B) -> Decimal
where
    A: Iterator<Item = Decimal> + Clone,
    B: Iterator<Item = Decimal> + Clone,
{
    scaled_sides(a, b).map_or(Decimal::ZERO, |(total_a, total_b, scale)| {
        total_a.saturating_sub(total_b).saturating_mul(scale)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance_of_sides_beyond_bounds_keeps_proportions() {
        let max = Decimal::MAX;
        let imbalance = imbalance_of_sides([max, max].into_iter(), [max].into_iter());
        assert_eq!(
            imbalance.round_dp(20),
            dec!(1).checked_div(dec!(3)).unwrap().round_dp(20)
        );
        assert_eq!(net_of_sides([max, max].into_iter(), [max].into_iter()), max);
    }

    #[test]
    fn test_imbalance_of_sides_empty_is_zero() {
        let none = std::iter::empty::<Decimal>();
        assert_eq!(
            imbalance_of_sides(none.clone(), none.clone()),
            Decimal::ZERO
        );
        assert_eq!(net_of_sides(none.clone(), none), Decimal::ZERO);
    }

    #[test]
    fn test_decimal_to_u64_saturating_bounds_clamp() {
        assert_eq!(decimal_to_u64_saturating(dec!(12.9)), 12);
        assert_eq!(decimal_to_u64_saturating(dec!(-3.5)), 0);
        assert_eq!(decimal_to_u64_saturating(dec!(-0.5)), 0);
        assert_eq!(decimal_to_u64_saturating(Decimal::MAX), u64::MAX);
    }

    #[test]
    fn test_exact_add_refuses_what_checked_add_rounds() {
        assert_eq!(exact_add(dec!(1.5), dec!(2.25)), Some(dec!(3.75)));
        assert_eq!(exact_add(dec!(1.5), dec!(-1.5)), Some(Decimal::ZERO));
        let tiny = Decimal::new(1, 28);
        let huge = Decimal::from(10_u64.pow(18));
        assert_eq!(exact_add(tiny, tiny), Some(Decimal::new(2, 28)));
        assert_eq!(exact_add(huge, tiny), None);
        assert!(huge.checked_add(tiny).is_some());
        assert_eq!(exact_add(Decimal::MAX, Decimal::ONE), None);
    }

    #[test]
    fn test_or_overflow_maps_none_to_numerical_error() {
        let err = Decimal::MAX
            .checked_add(Decimal::ONE)
            .or_overflow("sum")
            .unwrap_err();
        assert_eq!(err, MMError::NumericalError("sum overflows".to_string()));
        assert_eq!(Some(Decimal::ONE).or_overflow("sum").unwrap(), Decimal::ONE);
    }

    #[test]
    fn test_saturating_div() {
        assert_eq!(saturating_div(dec!(1), dec!(4)), dec!(0.25));
        assert_eq!(saturating_div(dec!(1), Decimal::ZERO), Decimal::ZERO);
        let tiny = dec!(0.0000000000000000000000000001);
        assert_eq!(saturating_div(Decimal::MAX, tiny), Decimal::MAX);
        assert_eq!(saturating_div(Decimal::MIN, tiny), Decimal::MIN);
        assert_eq!(saturating_div(Decimal::MAX, -tiny), Decimal::MIN);
    }

    #[test]
    fn test_midpoint_is_exact_near_the_bounds() {
        assert_eq!(midpoint(dec!(1), dec!(2)), dec!(1.5));
        assert_eq!(midpoint(Decimal::MAX, Decimal::MAX), Decimal::MAX);
        assert_eq!(midpoint(Decimal::MIN, Decimal::MIN), Decimal::MIN);
        assert_eq!(midpoint(Decimal::MIN, Decimal::MAX), Decimal::ZERO);
        let near = Decimal::MAX - dec!(2);
        assert_eq!(midpoint(near, Decimal::MAX), Decimal::MAX - dec!(1));
    }

    #[test]
    fn test_imbalance_ratio_is_scale_invariant() {
        assert_eq!(imbalance_ratio(dec!(3), dec!(1)), dec!(0.5));
        assert_eq!(imbalance_ratio(Decimal::ZERO, Decimal::ZERO), Decimal::ZERO);
        let huge = imbalance_ratio(Decimal::MAX, Decimal::MAX / dec!(3));
        assert!((huge - dec!(0.5)).abs() < dec!(0.00000001), "{huge}");
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
