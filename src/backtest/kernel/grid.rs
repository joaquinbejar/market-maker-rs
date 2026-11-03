//! Exact conversions between `Decimal` values and integer grid units.

use crate::Decimal;

/// `10^n` as `u128`, `None` when it does not fit.
#[must_use]
pub(super) fn pow10_u128(n: u32) -> Option<u128> {
    10u128.checked_pow(n)
}

/// `10^n` as `i128`, `None` when it does not fit.
#[must_use]
pub(super) fn pow10_i128(n: u32) -> Option<i128> {
    10i128.checked_pow(n)
}

/// A positive unit `m * 10^-s` with its normalized mantissa and scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Unit {
    /// Normalized mantissa (no trailing zeros once scaled).
    pub(super) mantissa: u64,
    /// Scale (decimal places).
    pub(super) scale: u32,
    /// The unit as a `Decimal`.
    pub(super) value: Decimal,
}

impl Unit {
    /// Splits a positive `Decimal` into its normalized mantissa and scale.
    #[must_use]
    pub(super) fn new(value: Decimal) -> Option<Self> {
        if value <= Decimal::ZERO {
            return None;
        }
        let normal = value.normalize();
        let mantissa = u64::try_from(normal.mantissa()).ok()?;
        Some(Self {
            mantissa,
            scale: normal.scale(),
            value: normal,
        })
    }

    /// Exact number of units in `value`, or `None` if `value` is not a whole
    /// multiple of the unit or the count does not fit.
    #[must_use]
    pub(super) fn count(self, value: Decimal) -> Option<i128> {
        let normal = value.normalize();
        let scale = normal.scale();
        if scale > self.scale {
            return None;
        }
        let shift = pow10_i128(self.scale.checked_sub(scale)?)?;
        let scaled = normal.mantissa().checked_mul(shift)?;
        let unit = i128::from(self.mantissa);
        if scaled.checked_rem(unit)? != 0 {
            return None;
        }
        let count = scaled.checked_div(unit)?;
        // Round trip through `Decimal`: the count must rebuild the value.
        let back = Decimal::try_from_i128_with_scale(count, 0)
            .ok()?
            .checked_mul(self.value)?;
        (back == value).then_some(count)
    }

    /// `count` units as a `Decimal`.
    #[must_use]
    pub(super) fn to_decimal(self, count: i128) -> Option<Decimal> {
        Decimal::try_from_i128_with_scale(count, 0)
            .ok()?
            .checked_mul(self.value)
    }
}

/// `value * 10^-scale` as a `Decimal`.
#[must_use]
pub(super) fn scaled_decimal(value: i128, scale: u32) -> Option<Decimal> {
    Decimal::try_from_i128_with_scale(value, scale).ok()
}

/// A half-U value `half * U / 2`, with `U = 10^-scale`, as a `Decimal`.
///
/// An even value is `half / 2` U; an odd one is `5 * half` at scale
/// `scale + 1`.
#[must_use]
pub(super) fn half_units_decimal(half: i128, scale: u32) -> Option<Decimal> {
    if half.checked_rem(2)? == 0 {
        scaled_decimal(half.checked_div(2)?, scale)
    } else {
        scaled_decimal(half.checked_mul(5)?, scale.checked_add(1)?)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_unit_count_on_grid_returns_exact_count() {
        let tick = Unit::new(dec!(0.25)).unwrap();
        assert_eq!((tick.mantissa, tick.scale), (25, 2));
        assert_eq!(tick.count(dec!(100.75)), Some(403));
        assert_eq!(tick.count(dec!(100.10)), None);
        assert_eq!(tick.to_decimal(403), Some(dec!(100.75)));
    }

    #[test]
    fn test_unit_new_non_positive_returns_none() {
        assert!(Unit::new(dec!(0)).is_none());
        assert!(Unit::new(dec!(-1)).is_none());
    }

    #[test]
    fn test_half_units_decimal_odd_and_even_match() {
        assert_eq!(half_units_decimal(250, 3), Some(dec!(0.125)));
        assert_eq!(half_units_decimal(251, 3), Some(dec!(0.1255)));
        assert_eq!(half_units_decimal(-3, 0), Some(dec!(-1.5)));
    }
}
