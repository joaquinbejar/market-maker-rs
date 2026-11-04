//! Inventory position tracking.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Represents the market maker's current inventory position.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple)
)]
pub struct InventoryPosition {
    /// Current quantity held.
    ///
    /// Positive = long position, Negative = short position, Zero = flat.
    pub quantity: Decimal,

    /// Average entry price for the current position.
    pub avg_entry_price: Decimal,

    /// Timestamp of last position update, in milliseconds since Unix epoch.
    pub last_update: u64,
}

impl InventoryPosition {
    /// Creates a new flat (zero) position.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantity: Decimal::ZERO,
            avg_entry_price: Decimal::ZERO,
            last_update: 0,
        }
    }

    /// Returns true if the position is flat (zero).
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.quantity == Decimal::ZERO
    }

    /// Returns true if the position is long (positive).
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.quantity > Decimal::ZERO
    }

    /// Returns true if the position is short (negative).
    #[must_use]
    pub fn is_short(&self) -> bool {
        self.quantity < Decimal::ZERO
    }

    /// Updates the position with a new fill (execution).
    ///
    /// This method calculates the new average entry price using weighted averages
    /// and updates the position quantity.
    ///
    /// # Arguments
    ///
    /// * `fill_quantity` - Quantity filled (positive = buy, negative = sell)
    /// * `fill_price` - Price at which the fill occurred
    /// * `timestamp` - Timestamp of the fill in milliseconds
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::position::inventory::InventoryPosition;
    /// use market_maker_rs::dec;
    ///
    /// let mut position = InventoryPosition::new();
    /// position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
    /// assert_eq!(position.quantity, dec!(10.0));
    /// assert_eq!(position.avg_entry_price, dec!(100.0));
    ///
    /// position.update_fill(dec!(5.0), dec!(102.0), 2000).unwrap();
    /// assert_eq!(position.quantity, dec!(15.0));
    /// // Weighted average: (10*100 + 5*102) / 15 = 100.666...
    /// ```
    ///
    /// Any price is accepted, negative ones included (some instruments
    /// trade below zero). The average entry price is the exact weighted
    /// mean, computed in an overflow safe form.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if the position quantity or the
    /// average entry price overflows; the position is then left unchanged.
    pub fn update_fill(
        &mut self,
        fill_quantity: Decimal,
        fill_price: Decimal,
        timestamp: u64,
    ) -> MMResult<()> {
        let new_quantity = self.quantity.checked_add(fill_quantity).ok_or_else(|| {
            MMError::NumericalError(format!(
                "position quantity overflows: {} + {}",
                self.quantity, fill_quantity
            ))
        })?;

        // If crossing from long to short or vice versa, reset avg price
        if (self.quantity > Decimal::ZERO && new_quantity < Decimal::ZERO)
            || (self.quantity < Decimal::ZERO && new_quantity > Decimal::ZERO)
        {
            self.avg_entry_price = fill_price;
        }
        // If increasing position, calculate weighted average
        else if new_quantity.abs() > self.quantity.abs() {
            // avg + (price - avg) * fill / new: no intermediate grows past
            // the inputs, unlike the total cost. `new_quantity` is non zero
            // here (its magnitude grew), so only an overflow can fail.
            let old_avg = self.avg_entry_price;
            let weighted = fill_price
                .checked_sub(old_avg)
                .and_then(|diff| {
                    diff.checked_mul(fill_quantity)
                        .and_then(|v| v.checked_div(new_quantity))
                        // `fill / new` is in (0, 1] here, so this form cannot
                        // overflow where the product above did.
                        .or_else(|| diff.checked_mul(fill_quantity.checked_div(new_quantity)?))
                })
                .and_then(|w| old_avg.checked_add(w));
            // Only prices of opposite sign near the bounds reach here (their
            // difference overflows); no form without that difference keeps
            // the rounding of the exact mean, so this is an error rather
            // than an approximate average.
            self.avg_entry_price = weighted.ok_or_else(|| {
                MMError::NumericalError(format!(
                    "average entry price overflows: {} at {} plus {} at {}",
                    self.quantity, old_avg, fill_quantity, fill_price
                ))
            })?;
        }
        // If reducing position, keep same avg price
        // (realized PnL is calculated separately)

        self.quantity = new_quantity;
        self.last_update = timestamp;

        // Handle precision for flat positions
        if self.quantity == Decimal::ZERO {
            self.avg_entry_price = Decimal::ZERO;
        }
        Ok(())
    }

    /// Calculates the unrealized PnL at a given market price.
    ///
    /// # Arguments
    ///
    /// * `current_price` - Current market price
    ///
    /// # Returns
    ///
    /// Unrealized profit or loss in currency units.
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::position::inventory::InventoryPosition;
    /// use market_maker_rs::dec;
    ///
    /// let mut position = InventoryPosition::new();
    /// position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
    ///
    /// // If price rises to 105, unrealized PnL = 10 * (105 - 100) = 50
    /// assert_eq!(position.unrealized_pnl(dec!(105.0)), dec!(50.0));
    ///
    /// // If price falls to 95, unrealized PnL = 10 * (95 - 100) = -50
    /// assert_eq!(position.unrealized_pnl(dec!(95.0)), dec!(-50.0));
    /// ```
    ///
    /// The arithmetic saturates at the `Decimal` bounds instead of panicking.
    #[must_use]
    pub fn unrealized_pnl(&self, current_price: Decimal) -> Decimal {
        if self.is_flat() {
            return Decimal::ZERO;
        }
        self.quantity
            .saturating_mul(current_price.saturating_sub(self.avg_entry_price))
    }
}

impl Default for InventoryPosition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_fill_unrepresentable_average_returns_error_and_keeps_position() {
        let mut position = InventoryPosition {
            quantity: dec!(1),
            avg_entry_price: Decimal::MIN,
            last_update: 0,
        };
        let before = position.clone();
        let result = position.update_fill(dec!(3), Decimal::MAX, 1);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        assert_eq!(position, before);
    }

    #[test]
    fn test_update_fill_quantity_overflow_returns_error_and_keeps_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(Decimal::MAX, dec!(1), 1).unwrap();
        let before = position.clone();
        let result = position.update_fill(dec!(1), dec!(2), 2);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        assert_eq!(position, before);
    }

    #[test]
    fn test_update_fill_huge_values_average_without_overflow() {
        // quantity * price overflows the total cost form, but the average
        // of two fills at the same huge price is that price.
        let mut position = InventoryPosition::new();
        position
            .update_fill(dec!(1_000_000_000_000), Decimal::MAX / dec!(2), 1)
            .unwrap();
        position
            .update_fill(dec!(1_000_000_000_000), Decimal::MAX / dec!(2), 2)
            .unwrap();
        assert_eq!(position.avg_entry_price, Decimal::MAX / dec!(2));
        assert_eq!(position.quantity, dec!(2_000_000_000_000));
    }
    use crate::dec;

    #[test]
    fn test_new_position_is_flat() {
        let position = InventoryPosition::new();
        assert_eq!(position.quantity, Decimal::ZERO);
        assert_eq!(position.avg_entry_price, Decimal::ZERO);
        assert_eq!(position.last_update, 0);
        assert!(position.is_flat());
    }

    #[test]
    fn test_default_position() {
        let position = InventoryPosition::default();
        assert_eq!(position.quantity, Decimal::ZERO);
        assert!(position.is_flat());
    }

    #[test]
    fn test_is_long() {
        let position = InventoryPosition {
            quantity: dec!(10.0),
            avg_entry_price: dec!(100.0),
            last_update: 1000,
        };
        assert!(position.is_long());
        assert!(!position.is_flat());
        assert!(!position.is_short());
    }

    #[test]
    fn test_is_short() {
        let position = InventoryPosition {
            quantity: dec!(-10.0),
            avg_entry_price: dec!(100.0),
            last_update: 1000,
        };
        assert!(position.is_short());
        assert!(!position.is_flat());
        assert!(!position.is_long());
    }

    #[test]
    fn test_is_flat() {
        let position = InventoryPosition {
            quantity: Decimal::ZERO,
            avg_entry_price: dec!(100.0),
            last_update: 1000,
        };
        assert!(position.is_flat());
        assert!(!position.is_long());
        assert!(!position.is_short());
    }

    #[test]
    fn test_very_small_position_is_flat() {
        let position = InventoryPosition {
            quantity: Decimal::ZERO, // Decimal is exact, so only ZERO is flat
            avg_entry_price: dec!(100.0),
            last_update: 1000,
        };
        assert!(position.is_flat());
    }

    #[test]
    fn test_update_fill_buy() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
        assert_eq!(position.quantity, dec!(10.0));
        assert_eq!(position.avg_entry_price, dec!(100.0));
        assert_eq!(position.last_update, 1000);
    }

    #[test]
    fn test_update_fill_sell() {
        let mut position = InventoryPosition::new();
        position
            .update_fill(dec!(-10.0), dec!(100.0), 1000)
            .unwrap();
        assert_eq!(position.quantity, dec!(-10.0));
        assert_eq!(position.avg_entry_price, dec!(100.0));
    }

    #[test]
    fn test_update_fill_weighted_average() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
        position.update_fill(dec!(5.0), dec!(102.0), 2000).unwrap();

        assert_eq!(position.quantity, dec!(15.0));
        // (10*100 + 5*102) / 15 = 100.666666...
        let expected = dec!(100.666666666666666667);
        assert!((position.avg_entry_price - expected).abs() < dec!(0.000001));
    }

    #[test]
    fn test_update_fill_reduce_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
        position.update_fill(dec!(-5.0), dec!(105.0), 2000).unwrap();

        assert_eq!(position.quantity, dec!(5.0));
        // Avg price should remain at original entry
        assert_eq!(position.avg_entry_price, dec!(100.0));
    }

    #[test]
    fn test_update_fill_flatten_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
        position
            .update_fill(dec!(-10.0), dec!(105.0), 2000)
            .unwrap();

        assert_eq!(position.quantity, Decimal::ZERO);
        assert_eq!(position.avg_entry_price, Decimal::ZERO);
        assert!(position.is_flat());
    }

    #[test]
    fn test_update_fill_flip_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();
        position
            .update_fill(dec!(-15.0), dec!(105.0), 2000)
            .unwrap();

        assert_eq!(position.quantity, dec!(-5.0));
        // When flipping, new avg price is the flip fill price
        assert_eq!(position.avg_entry_price, dec!(105.0));
    }

    #[test]
    fn test_unrealized_pnl_long_profit() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();

        let pnl = position.unrealized_pnl(dec!(105.0));
        assert_eq!(pnl, dec!(50.0)); // 10 * (105 - 100)
    }

    #[test]
    fn test_unrealized_pnl_long_loss() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000).unwrap();

        let pnl = position.unrealized_pnl(dec!(95.0));
        assert_eq!(pnl, dec!(-50.0)); // 10 * (95 - 100)
    }

    #[test]
    fn test_unrealized_pnl_short_profit() {
        let mut position = InventoryPosition::new();
        position
            .update_fill(dec!(-10.0), dec!(100.0), 1000)
            .unwrap();

        let pnl = position.unrealized_pnl(dec!(95.0));
        assert_eq!(pnl, dec!(50.0)); // -10 * (95 - 100)
    }

    #[test]
    fn test_unrealized_pnl_flat() {
        let position = InventoryPosition::new();
        let pnl = position.unrealized_pnl(dec!(100.0));
        assert_eq!(pnl, Decimal::ZERO);
    }
}
