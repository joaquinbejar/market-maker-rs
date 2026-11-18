//! Conversions between `Decimal` and `f64` that give the same bits on every target.
//!
//! A value enters the float domain by Rust's correctly rounded parsing of its decimal
//! form, and leaves it rounded half to even from its exact binary value, read from Rust's
//! exact decimal printing. Neither step does float arithmetic, so neither depends on the
//! target's math library.

use std::cmp::Ordering;

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

/// `value` as the nearest `f64`.
pub(crate) fn decimal_to_f64(value: Decimal) -> MMResult<f64> {
    let parsed: f64 = value
        .normalize()
        .to_string()
        .parse()
        .map_err(|_| numeric("decimal to f64"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(numeric("decimal to f64"))
    }
}

/// `x` rounded half to even to `decimals` places, from its exact binary value.
pub(crate) fn round_half_even_decimals(x: f64, decimals: u32) -> MMResult<Decimal> {
    let (int_part, frac) = exact_digits(x)?;
    let keep = usize::try_from(decimals).map_err(|_| numeric("rounding"))?;
    let digits: Vec<u8> = int_part.bytes().chain(frac.bytes()).collect();
    let cut = int_part.len().saturating_add(keep);
    round_digits(x, &digits, int_part.len(), cut)
}

/// `x` rounded half to even to `significant` significant digits, from its exact binary
/// value. Zero stays zero.
#[allow(dead_code)] // for the strategy float paths
pub(crate) fn round_half_even_significant(x: f64, significant: u32) -> MMResult<Decimal> {
    let (int_part, frac) = exact_digits(x)?;
    let digits: Vec<u8> = int_part.bytes().chain(frac.bytes()).collect();
    let Some(first) = digits.iter().position(|d| *d != b'0') else {
        return Ok(Decimal::ZERO);
    };
    let keep = usize::try_from(significant).map_err(|_| numeric("rounding"))?;
    let cut = first.saturating_add(keep).max(int_part.len());
    round_digits(x, &digits, int_part.len(), cut)
}

/// The exact decimal expansion of `|x|`: an `f64` has at most 1074 fractional digits.
fn exact_digits(x: f64) -> MMResult<(String, String)> {
    if !x.is_finite() {
        return Err(numeric("value is not finite"));
    }
    let exact = format!("{:.1074}", x.abs());
    let (int_part, frac) = exact.split_once('.').ok_or_else(|| numeric("rounding"))?;
    Ok((int_part.to_string(), frac.to_string()))
}

/// Keeps `digits[..cut]` (the point after `point` digits), rounding half to even on the
/// rest, with the sign of `x`.
fn round_digits(x: f64, digits: &[u8], point: usize, cut: usize) -> MMResult<Decimal> {
    let kept = digits.get(..cut).ok_or_else(|| numeric("rounding"))?;
    let next = digits.get(cut).copied().unwrap_or(b'0');
    let rest_nonzero = digits
        .get(cut.saturating_add(1)..)
        .is_some_and(|rest| rest.iter().any(|d| *d != b'0'));
    let last_odd = kept
        .last()
        .is_some_and(|d| d.checked_sub(b'0').is_some_and(|v| v & 1 == 1));
    let up = match next.cmp(&b'5') {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => rest_nonzero || last_odd,
    };
    let scale = u32::try_from(cut.saturating_sub(point)).map_err(|_| numeric("rounding"))?;
    if scale > 28 {
        return Err(numeric("rounding beyond 28 decimals"));
    }
    let text = std::str::from_utf8(kept).map_err(|_| numeric("rounding"))?;
    let mantissa: i128 = if text.is_empty() {
        0
    } else {
        text.parse().map_err(|_| numeric("rounding"))?
    };
    let mantissa = if up {
        mantissa.checked_add(1).ok_or_else(|| numeric("rounding"))?
    } else {
        mantissa
    };
    let signed = if x.is_sign_negative() {
        mantissa.checked_neg().ok_or_else(|| numeric("rounding"))?
    } else {
        mantissa
    };
    Decimal::try_from_i128_with_scale(signed, scale)
        .map(|d| d.normalize())
        .map_err(|_| numeric("rounding"))
}

fn numeric(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} does not convert"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn decimals_round_half_even_on_the_exact_value() {
        // 2^-10 = 0.0009765625 is a tie at 9 places: kept digit 2 is even.
        assert_eq!(
            round_half_even_decimals(0.000_976_562_5, 9).unwrap(),
            dec!(0.000976562)
        );
        // 3 * 2^-10 = 0.0029296875: kept digit 7 is odd, so it rounds up.
        assert_eq!(
            round_half_even_decimals(0.002_929_687_5, 9).unwrap(),
            dec!(0.002929688)
        );
        assert_eq!(round_half_even_decimals(-0.5, 9).unwrap(), dec!(-0.5));
        assert_eq!(
            round_half_even_decimals(0.999_999_999_9, 9).unwrap(),
            dec!(1)
        );
    }

    #[test]
    fn significant_digits_follow_the_first_nonzero_digit() {
        assert_eq!(
            round_half_even_significant(0.000_123_456_789, 3).unwrap(),
            dec!(0.000123)
        );
        assert_eq!(
            round_half_even_significant(1.0 / 3.0, 12).unwrap(),
            dec!(0.333333333333)
        );
        assert_eq!(round_half_even_significant(2.5, 1).unwrap(), dec!(2));
        assert_eq!(round_half_even_significant(0.0, 12).unwrap(), Decimal::ZERO);
    }

    #[test]
    fn parsing_is_correctly_rounded() {
        assert_eq!(decimal_to_f64(dec!(0.1)).unwrap(), 0.1);
        assert_eq!(decimal_to_f64(dec!(-2.5)).unwrap(), -2.5);
    }
}
