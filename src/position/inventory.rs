//! Inventory position tracking.

use crate::Decimal;

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
    /// position.update_fill(dec!(10.0), dec!(100.0), 1000);
    /// assert_eq!(position.quantity, dec!(10.0));
    /// assert_eq!(position.avg_entry_price, dec!(100.0));
    ///
    /// position.update_fill(dec!(5.0), dec!(102.0), 2000);
    /// assert_eq!(position.quantity, dec!(15.0));
    /// // Weighted average: (10*100 + 5*102) / 15 = 100.666...
    /// ```
    ///
    /// This method cannot return an error, so its arithmetic saturates at
    /// the `Decimal` bounds instead of panicking; values that large are far
    /// outside any real position.
    pub fn update_fill(&mut self, fill_quantity: Decimal, fill_price: Decimal, timestamp: u64) {
        let new_quantity = self.quantity.saturating_add(fill_quantity);

        // If crossing from long to short or vice versa, reset avg price
        if (self.quantity > Decimal::ZERO && new_quantity < Decimal::ZERO)
            || (self.quantity < Decimal::ZERO && new_quantity > Decimal::ZERO)
        {
            self.avg_entry_price = fill_price;
        }
        // If increasing position, calculate weighted average
        else if new_quantity.abs() > self.quantity.abs() {
            let total_cost = self
                .quantity
                .saturating_mul(self.avg_entry_price)
                .saturating_add(fill_quantity.saturating_mul(fill_price));
            // `new_quantity` is non zero here (its magnitude grew), so only
            // an overflow can fail; keep the previous average in that case.
            if let Some(avg) = total_cost.checked_div(new_quantity) {
                self.avg_entry_price = avg;
            }
        }
        // If reducing position, keep same avg price
        // (realized PnL is calculated separately)

        self.quantity = new_quantity;
        self.last_update = timestamp;

        // Handle precision for flat positions
        if self.quantity == Decimal::ZERO {
            self.avg_entry_price = Decimal::ZERO;
        }
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
    /// position.update_fill(dec!(10.0), dec!(100.0), 1000);
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
        position.update_fill(dec!(10.0), dec!(100.0), 1000);
        assert_eq!(position.quantity, dec!(10.0));
        assert_eq!(position.avg_entry_price, dec!(100.0));
        assert_eq!(position.last_update, 1000);
    }

    #[test]
    fn test_update_fill_sell() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(-10.0), dec!(100.0), 1000);
        assert_eq!(position.quantity, dec!(-10.0));
        assert_eq!(position.avg_entry_price, dec!(100.0));
    }

    #[test]
    fn test_update_fill_weighted_average() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);
        position.update_fill(dec!(5.0), dec!(102.0), 2000);

        assert_eq!(position.quantity, dec!(15.0));
        // (10*100 + 5*102) / 15 = 100.666666...
        let expected = dec!(100.666666666666666667);
        assert!((position.avg_entry_price - expected).abs() < dec!(0.000001));
    }

    #[test]
    fn test_update_fill_reduce_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);
        position.update_fill(dec!(-5.0), dec!(105.0), 2000);

        assert_eq!(position.quantity, dec!(5.0));
        // Avg price should remain at original entry
        assert_eq!(position.avg_entry_price, dec!(100.0));
    }

    #[test]
    fn test_update_fill_flatten_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);
        position.update_fill(dec!(-10.0), dec!(105.0), 2000);

        assert_eq!(position.quantity, Decimal::ZERO);
        assert_eq!(position.avg_entry_price, Decimal::ZERO);
        assert!(position.is_flat());
    }

    #[test]
    fn test_update_fill_flip_position() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);
        position.update_fill(dec!(-15.0), dec!(105.0), 2000);

        assert_eq!(position.quantity, dec!(-5.0));
        // When flipping, new avg price is the flip fill price
        assert_eq!(position.avg_entry_price, dec!(105.0));
    }

    #[test]
    fn test_unrealized_pnl_long_profit() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);

        let pnl = position.unrealized_pnl(dec!(105.0));
        assert_eq!(pnl, dec!(50.0)); // 10 * (105 - 100)
    }

    #[test]
    fn test_unrealized_pnl_long_loss() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(10.0), dec!(100.0), 1000);

        let pnl = position.unrealized_pnl(dec!(95.0));
        assert_eq!(pnl, dec!(-50.0)); // 10 * (95 - 100)
    }

    #[test]
    fn test_unrealized_pnl_short_profit() {
        let mut position = InventoryPosition::new();
        position.update_fill(dec!(-10.0), dec!(100.0), 1000);

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
