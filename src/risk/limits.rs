//! Risk limits implementation for position and exposure control.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Risk limits configuration for position and exposure control.
///
/// This struct defines the maximum allowed position size and notional exposure,
/// as well as a scaling factor to reduce order sizes as position approaches limits.
///
/// # Fields
///
/// - `max_position`: Maximum absolute position size in units (e.g., 100 BTC)
/// - `max_notional`: Maximum notional exposure in currency (e.g., $1,000,000)
/// - `scaling_factor`: Factor (0.0 to 1.0) controlling how aggressively to scale orders
///
/// # Scaling Behavior
///
/// When `scaling_factor` is set, order sizes are reduced as position approaches limits:
/// - At 0% of limit: full order size
/// - At 50% of limit with 0.5 scaling: ~75% of order size
/// - At 100% of limit: 0% of order size (orders blocked)
///
/// The scaling formula is: `scaled_size = desired_size * (1 - (position_ratio * scaling_factor))`
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::RiskLimits;
/// use market_maker_rs::dec;
///
/// let limits = RiskLimits::new(
///     dec!(100.0),   // max 100 units
///     dec!(10000.0), // max $10,000 notional
///     dec!(0.5),     // 50% scaling
/// ).unwrap();
///
/// // At 50% position, orders are scaled down
/// let scaled = limits.scale_order_size(dec!(50.0), dec!(10.0));
/// assert!(scaled < dec!(10.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RiskLimits {
    /// Maximum absolute position size in units.
    ///
    /// This limit applies to both long and short positions.
    /// A value of 100 means position must stay within [-100, +100].
    pub max_position: Decimal,

    /// Maximum notional exposure in currency units.
    ///
    /// Notional = |position| × price.
    /// This provides a value-based limit that accounts for price.
    pub max_notional: Decimal,

    /// Scaling factor for order size reduction (0.0 to 1.0).
    ///
    /// Controls how aggressively order sizes are reduced as position
    /// approaches the limit. Higher values = more aggressive scaling.
    /// - 0.0: No scaling (orders at full size until limit hit)
    /// - 0.5: Moderate scaling
    /// - 1.0: Linear scaling (order size proportional to remaining capacity)
    pub scaling_factor: Decimal,
}

impl RiskLimits {
    /// Creates a new `RiskLimits` instance with validation.
    ///
    /// # Arguments
    ///
    /// * `max_position` - Maximum absolute position size (must be positive)
    /// * `max_notional` - Maximum notional exposure (must be positive)
    /// * `scaling_factor` - Order scaling factor (must be in [0.0, 1.0])
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if:
    /// - `max_position` is not positive
    /// - `max_notional` is not positive
    /// - `scaling_factor` is not in [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(
    ///     dec!(100.0),
    ///     dec!(10000.0),
    ///     dec!(0.5),
    /// ).unwrap();
    /// ```
    pub fn new(
        max_position: Decimal,
        max_notional: Decimal,
        scaling_factor: Decimal,
    ) -> MMResult<Self> {
        if max_position <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "max_position must be positive".to_string(),
            ));
        }

        if max_notional <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "max_notional must be positive".to_string(),
            ));
        }

        if scaling_factor < Decimal::ZERO || scaling_factor > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "scaling_factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            max_position,
            max_notional,
            scaling_factor,
        })
    }

    /// Checks if a new order would violate position or notional limits.
    ///
    /// # Arguments
    ///
    /// * `current_position` - Current position size (positive = long, negative = short)
    /// * `order_size` - Size of the proposed order (positive = buy, negative = sell)
    /// * `price` - Current market price for notional calculation
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if the order is allowed
    /// - `Ok(false)` if the order would violate limits
    /// - `Err` if price is invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();
    ///
    /// // Order within limits
    /// assert!(limits.check_order(dec!(50.0), dec!(10.0), dec!(100.0)).unwrap());
    ///
    /// // Order would exceed position limit
    /// assert!(!limits.check_order(dec!(95.0), dec!(10.0), dec!(100.0)).unwrap());
    /// ```
    pub fn check_order(
        &self,
        current_position: Decimal,
        order_size: Decimal,
        price: Decimal,
    ) -> MMResult<bool> {
        if price <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(
                "price must be positive".to_string(),
            ));
        }

        let new_position = current_position + order_size;

        // Check position limit
        if new_position.abs() > self.max_position {
            return Ok(false);
        }

        // Check notional limit
        let new_notional = new_position.abs() * price;
        if new_notional > self.max_notional {
            return Ok(false);
        }

        Ok(true)
    }

    /// Calculates a scaled order size based on current position.
    ///
    /// As position approaches the limit, order sizes are reduced according
    /// to the scaling factor. This helps prevent sudden limit breaches and
    /// provides smoother inventory management.
    ///
    /// # Arguments
    ///
    /// * `current_position` - Current position size
    /// * `desired_size` - Desired order size before scaling
    ///
    /// # Returns
    ///
    /// Scaled order size, which will be:
    /// - Equal to `desired_size` when position is zero
    /// - Reduced as position approaches limit
    /// - Zero when position is at or beyond limit
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(1.0)).unwrap();
    ///
    /// // At zero position, full size
    /// assert_eq!(limits.scale_order_size(dec!(0.0), dec!(10.0)), dec!(10.0));
    ///
    /// // At 50% of limit with scaling_factor=1.0, 50% size
    /// assert_eq!(limits.scale_order_size(dec!(50.0), dec!(10.0)), dec!(5.0));
    ///
    /// // At limit, zero size
    /// assert_eq!(limits.scale_order_size(dec!(100.0), dec!(10.0)), dec!(0.0));
    /// ```
    #[must_use]
    pub fn scale_order_size(&self, current_position: Decimal, desired_size: Decimal) -> Decimal {
        if desired_size <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let position_ratio = current_position.abs() / self.max_position;

        // If already at or beyond limit, return zero
        if position_ratio >= Decimal::ONE {
            return Decimal::ZERO;
        }

        // Calculate scaling multiplier: 1 - (position_ratio * scaling_factor)
        let scale_multiplier = Decimal::ONE - (position_ratio * self.scaling_factor);

        // Ensure multiplier is non-negative
        let scale_multiplier = scale_multiplier.max(Decimal::ZERO);

        desired_size * scale_multiplier
    }

    /// Checks if the current position breaches the position limit.
    ///
    /// # Arguments
    ///
    /// * `position` - Current position size
    ///
    /// # Returns
    ///
    /// `true` if the absolute position exceeds `max_position`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();
    ///
    /// assert!(!limits.is_position_limit_breached(dec!(50.0)));
    /// assert!(!limits.is_position_limit_breached(dec!(100.0)));
    /// assert!(limits.is_position_limit_breached(dec!(101.0)));
    /// assert!(limits.is_position_limit_breached(dec!(-101.0)));
    /// ```
    #[must_use]
    pub fn is_position_limit_breached(&self, position: Decimal) -> bool {
        position.abs() > self.max_position
    }

    /// Checks if the current notional exposure breaches the notional limit.
    ///
    /// # Arguments
    ///
    /// * `position` - Current position size
    /// * `price` - Current market price
    ///
    /// # Returns
    ///
    /// `true` if notional exposure exceeds `max_notional`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();
    ///
    /// // 50 units at $100 = $5,000 notional (within limit)
    /// assert!(!limits.is_notional_limit_breached(dec!(50.0), dec!(100.0)));
    ///
    /// // 50 units at $250 = $12,500 notional (exceeds limit)
    /// assert!(limits.is_notional_limit_breached(dec!(50.0), dec!(250.0)));
    /// ```
    #[must_use]
    pub fn is_notional_limit_breached(&self, position: Decimal, price: Decimal) -> bool {
        let notional = position.abs() * price;
        notional > self.max_notional
    }

    /// Returns the remaining position capacity before hitting the limit.
    ///
    /// # Arguments
    ///
    /// * `current_position` - Current position size
    ///
    /// # Returns
    ///
    /// Remaining capacity in units. Returns zero if limit is already breached.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();
    ///
    /// assert_eq!(limits.remaining_position_capacity(dec!(60.0)), dec!(40.0));
    /// assert_eq!(limits.remaining_position_capacity(dec!(-60.0)), dec!(40.0));
    /// assert_eq!(limits.remaining_position_capacity(dec!(100.0)), dec!(0.0));
    /// ```
    #[must_use]
    pub fn remaining_position_capacity(&self, current_position: Decimal) -> Decimal {
        let remaining = self.max_position - current_position.abs();
        remaining.max(Decimal::ZERO)
    }

    /// Returns the current position utilization as a percentage (0.0 to 1.0+).
    ///
    /// # Arguments
    ///
    /// * `current_position` - Current position size
    ///
    /// # Returns
    ///
    /// Position utilization ratio. Values > 1.0 indicate limit breach.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::RiskLimits;
    /// use market_maker_rs::dec;
    ///
    /// let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();
    ///
    /// assert_eq!(limits.position_utilization(dec!(50.0)), dec!(0.5));
    /// assert_eq!(limits.position_utilization(dec!(100.0)), dec!(1.0));
    /// assert_eq!(limits.position_utilization(dec!(150.0)), dec!(1.5));
    /// ```
    #[must_use]
    pub fn position_utilization(&self, current_position: Decimal) -> Decimal {
        current_position.abs() / self.max_position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_new_valid_limits() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5));
        assert!(limits.is_ok());

        let limits = limits.unwrap();
        assert_eq!(limits.max_position, dec!(100.0));
        assert_eq!(limits.max_notional, dec!(10000.0));
        assert_eq!(limits.scaling_factor, dec!(0.5));
    }

    #[test]
    fn test_new_zero_scaling_factor() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.0));
        assert!(limits.is_ok());
    }

    #[test]
    fn test_new_one_scaling_factor() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(1.0));
        assert!(limits.is_ok());
    }

    #[test]
    fn test_new_invalid_max_position() {
        let limits = RiskLimits::new(dec!(0.0), dec!(10000.0), dec!(0.5));
        assert!(limits.is_err());
        assert!(matches!(
            limits.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));

        let limits = RiskLimits::new(dec!(-100.0), dec!(10000.0), dec!(0.5));
        assert!(limits.is_err());
    }

    #[test]
    fn test_new_invalid_max_notional() {
        let limits = RiskLimits::new(dec!(100.0), dec!(0.0), dec!(0.5));
        assert!(limits.is_err());

        let limits = RiskLimits::new(dec!(100.0), dec!(-10000.0), dec!(0.5));
        assert!(limits.is_err());
    }

    #[test]
    fn test_new_invalid_scaling_factor() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(-0.1));
        assert!(limits.is_err());

        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(1.1));
        assert!(limits.is_err());
    }

    #[test]
    fn test_check_order_within_limits() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // Buy order within limits
        assert!(
            limits
                .check_order(dec!(50.0), dec!(10.0), dec!(100.0))
                .unwrap()
        );

        // Sell order within limits
        assert!(
            limits
                .check_order(dec!(50.0), dec!(-10.0), dec!(100.0))
                .unwrap()
        );

        // Order from flat position
        assert!(
            limits
                .check_order(dec!(0.0), dec!(50.0), dec!(100.0))
                .unwrap()
        );
    }

    #[test]
    fn test_check_order_exceeds_position_limit() {
        let limits = RiskLimits::new(dec!(100.0), dec!(100000.0), dec!(0.5)).unwrap();

        // Would exceed long limit
        assert!(
            !limits
                .check_order(dec!(95.0), dec!(10.0), dec!(100.0))
                .unwrap()
        );

        // Would exceed short limit
        assert!(
            !limits
                .check_order(dec!(-95.0), dec!(-10.0), dec!(100.0))
                .unwrap()
        );
    }

    #[test]
    fn test_check_order_exceeds_notional_limit() {
        let limits = RiskLimits::new(dec!(100.0), dec!(5000.0), dec!(0.5)).unwrap();

        // 60 units at $100 = $6,000 notional (exceeds $5,000 limit)
        assert!(
            !limits
                .check_order(dec!(50.0), dec!(10.0), dec!(100.0))
                .unwrap()
        );
    }

    #[test]
    fn test_check_order_exactly_at_limit() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // Exactly at position limit should be allowed
        assert!(
            limits
                .check_order(dec!(90.0), dec!(10.0), dec!(100.0))
                .unwrap()
        );

        // Exactly at notional limit should be allowed
        assert!(
            limits
                .check_order(dec!(0.0), dec!(100.0), dec!(100.0))
                .unwrap()
        );
    }

    #[test]
    fn test_check_order_invalid_price() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        assert!(
            limits
                .check_order(dec!(50.0), dec!(10.0), dec!(0.0))
                .is_err()
        );
        assert!(
            limits
                .check_order(dec!(50.0), dec!(10.0), dec!(-100.0))
                .is_err()
        );
    }

    #[test]
    fn test_check_order_reducing_position() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // Selling when long should always be allowed (reduces risk)
        assert!(
            limits
                .check_order(dec!(150.0), dec!(-50.0), dec!(100.0))
                .unwrap()
        );

        // Buying when short should always be allowed (reduces risk)
        assert!(
            limits
                .check_order(dec!(-150.0), dec!(50.0), dec!(100.0))
                .unwrap()
        );
    }

    #[test]
    fn test_scale_order_size_at_zero_position() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // At zero position, no scaling
        assert_eq!(limits.scale_order_size(dec!(0.0), dec!(10.0)), dec!(10.0));
    }

    #[test]
    fn test_scale_order_size_at_half_position() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(1.0)).unwrap();

        // At 50% position with scaling_factor=1.0, should get 50% of desired size
        assert_eq!(limits.scale_order_size(dec!(50.0), dec!(10.0)), dec!(5.0));
    }

    #[test]
    fn test_scale_order_size_at_limit() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // At limit, should return zero
        assert_eq!(limits.scale_order_size(dec!(100.0), dec!(10.0)), dec!(0.0));
    }

    #[test]
    fn test_scale_order_size_beyond_limit() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // Beyond limit, should return zero
        assert_eq!(limits.scale_order_size(dec!(150.0), dec!(10.0)), dec!(0.0));
    }

    #[test]
    fn test_scale_order_size_negative_position() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(1.0)).unwrap();

        // Negative position should scale the same as positive
        assert_eq!(limits.scale_order_size(dec!(-50.0), dec!(10.0)), dec!(5.0));
    }

    #[test]
    fn test_scale_order_size_zero_scaling_factor() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.0)).unwrap();

        // With zero scaling factor, no scaling until limit
        assert_eq!(limits.scale_order_size(dec!(50.0), dec!(10.0)), dec!(10.0));
        assert_eq!(limits.scale_order_size(dec!(99.0), dec!(10.0)), dec!(10.0));
        assert_eq!(limits.scale_order_size(dec!(100.0), dec!(10.0)), dec!(0.0));
    }

    #[test]
    fn test_scale_order_size_zero_desired() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        assert_eq!(limits.scale_order_size(dec!(50.0), dec!(0.0)), dec!(0.0));
        assert_eq!(limits.scale_order_size(dec!(50.0), dec!(-10.0)), dec!(0.0));
    }

    #[test]
    fn test_is_position_limit_breached() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        assert!(!limits.is_position_limit_breached(dec!(0.0)));
        assert!(!limits.is_position_limit_breached(dec!(50.0)));
        assert!(!limits.is_position_limit_breached(dec!(100.0)));
        assert!(limits.is_position_limit_breached(dec!(100.1)));
        assert!(limits.is_position_limit_breached(dec!(-100.1)));
    }

    #[test]
    fn test_is_notional_limit_breached() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        // 50 units at $100 = $5,000 (within limit)
        assert!(!limits.is_notional_limit_breached(dec!(50.0), dec!(100.0)));

        // 100 units at $100 = $10,000 (at limit)
        assert!(!limits.is_notional_limit_breached(dec!(100.0), dec!(100.0)));

        // 100 units at $101 = $10,100 (exceeds limit)
        assert!(limits.is_notional_limit_breached(dec!(100.0), dec!(101.0)));

        // Short position also counts
        assert!(limits.is_notional_limit_breached(dec!(-100.0), dec!(101.0)));
    }

    #[test]
    fn test_remaining_position_capacity() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        assert_eq!(limits.remaining_position_capacity(dec!(0.0)), dec!(100.0));
        assert_eq!(limits.remaining_position_capacity(dec!(60.0)), dec!(40.0));
        assert_eq!(limits.remaining_position_capacity(dec!(-60.0)), dec!(40.0));
        assert_eq!(limits.remaining_position_capacity(dec!(100.0)), dec!(0.0));
        assert_eq!(limits.remaining_position_capacity(dec!(150.0)), dec!(0.0));
    }

    #[test]
    fn test_position_utilization() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        assert_eq!(limits.position_utilization(dec!(0.0)), dec!(0.0));
        assert_eq!(limits.position_utilization(dec!(50.0)), dec!(0.5));
        assert_eq!(limits.position_utilization(dec!(-50.0)), dec!(0.5));
        assert_eq!(limits.position_utilization(dec!(100.0)), dec!(1.0));
        assert_eq!(limits.position_utilization(dec!(150.0)), dec!(1.5));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let limits = RiskLimits::new(dec!(100.0), dec!(10000.0), dec!(0.5)).unwrap();

        let json = serde_json::to_string(&limits).unwrap();
        let deserialized: RiskLimits = serde_json::from_str(&json).unwrap();

        assert_eq!(limits, deserialized);
    }
}
