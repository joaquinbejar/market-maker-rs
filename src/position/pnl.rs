//! PnL (Profit and Loss) calculations.

use crate::Decimal;

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Represents profit and loss information.
///
/// Updates cannot return errors, so sums saturate at the `Decimal` bounds
/// instead of panicking.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple)
)]
pub struct PnL {
    /// Realized PnL from closed positions.
    pub realized: Decimal,

    /// Unrealized PnL from current open position.
    pub unrealized: Decimal,

    /// Total PnL (realized + unrealized).
    pub total: Decimal,
}

impl PnL {
    /// Creates a new PnL with zero values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            realized: Decimal::ZERO,
            unrealized: Decimal::ZERO,
            total: Decimal::ZERO,
        }
    }

    /// Updates the PnL with new realized and unrealized values.
    ///
    /// # Arguments
    ///
    /// * `realized` - Realized PnL from closed positions
    /// * `unrealized` - Unrealized PnL from open positions
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::position::pnl::PnL;
    /// use market_maker_rs::dec;
    ///
    /// let mut pnl = PnL::new();
    /// pnl.update(dec!(100.0), dec!(50.0));
    ///
    /// assert_eq!(pnl.realized, dec!(100.0));
    /// assert_eq!(pnl.unrealized, dec!(50.0));
    /// assert_eq!(pnl.total, dec!(150.0));
    /// ```
    pub fn update(&mut self, realized: Decimal, unrealized: Decimal) {
        self.realized = realized;
        self.unrealized = unrealized;
        self.total = realized.saturating_add(unrealized);
    }

    /// Adds realized PnL (e.g., from a closed trade).
    ///
    /// # Arguments
    ///
    /// * `amount` - Amount to add to realized PnL (can be negative)
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::position::pnl::PnL;
    /// use market_maker_rs::dec;
    ///
    /// let mut pnl = PnL::new();
    /// pnl.add_realized(dec!(100.0));
    /// assert_eq!(pnl.realized, dec!(100.0));
    /// assert_eq!(pnl.total, dec!(100.0));
    ///
    /// pnl.add_realized(dec!(-25.0));
    /// assert_eq!(pnl.realized, dec!(75.0));
    /// assert_eq!(pnl.total, dec!(75.0));
    /// ```
    pub fn add_realized(&mut self, amount: Decimal) {
        self.realized = self.realized.saturating_add(amount);
        self.total = self.realized.saturating_add(self.unrealized);
    }

    /// Updates the unrealized PnL component.
    ///
    /// # Arguments
    ///
    /// * `amount` - New unrealized PnL value
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::position::pnl::PnL;
    /// use market_maker_rs::dec;
    ///
    /// let mut pnl = PnL::new();
    /// pnl.add_realized(dec!(100.0));
    /// pnl.set_unrealized(dec!(50.0));
    ///
    /// assert_eq!(pnl.unrealized, dec!(50.0));
    /// assert_eq!(pnl.total, dec!(150.0));
    /// ```
    pub fn set_unrealized(&mut self, amount: Decimal) {
        self.unrealized = amount;
        self.total = self.realized.saturating_add(self.unrealized);
    }
}

impl Default for PnL {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_new_pnl() {
        let pnl = PnL::new();
        assert_eq!(pnl.realized, Decimal::ZERO);
        assert_eq!(pnl.unrealized, Decimal::ZERO);
        assert_eq!(pnl.total, Decimal::ZERO);
    }

    #[test]
    fn test_default_pnl() {
        let pnl = PnL::default();
        assert_eq!(pnl.realized, Decimal::ZERO);
        assert_eq!(pnl.unrealized, Decimal::ZERO);
        assert_eq!(pnl.total, Decimal::ZERO);
    }

    #[test]
    fn test_pnl_with_values() {
        let pnl = PnL {
            realized: dec!(100.0),
            unrealized: dec!(50.0),
            total: dec!(150.0),
        };
        assert_eq!(pnl.realized, dec!(100.0));
        assert_eq!(pnl.unrealized, dec!(50.0));
        assert_eq!(pnl.total, dec!(150.0));
    }

    #[test]
    fn test_pnl_update() {
        let mut pnl = PnL::new();
        pnl.update(dec!(100.0), dec!(50.0));

        assert_eq!(pnl.realized, dec!(100.0));
        assert_eq!(pnl.unrealized, dec!(50.0));
        assert_eq!(pnl.total, dec!(150.0));
    }

    #[test]
    fn test_pnl_add_realized() {
        let mut pnl = PnL::new();
        pnl.add_realized(dec!(100.0));
        assert_eq!(pnl.realized, dec!(100.0));
        assert_eq!(pnl.total, dec!(100.0));

        pnl.add_realized(dec!(50.0));
        assert_eq!(pnl.realized, dec!(150.0));
        assert_eq!(pnl.total, dec!(150.0));
    }

    #[test]
    fn test_pnl_add_realized_negative() {
        let mut pnl = PnL::new();
        pnl.add_realized(dec!(100.0));
        pnl.add_realized(dec!(-25.0));

        assert_eq!(pnl.realized, dec!(75.0));
        assert_eq!(pnl.total, dec!(75.0));
    }

    #[test]
    fn test_pnl_set_unrealized() {
        let mut pnl = PnL::new();
        pnl.add_realized(dec!(100.0));
        pnl.set_unrealized(dec!(50.0));

        assert_eq!(pnl.realized, dec!(100.0));
        assert_eq!(pnl.unrealized, dec!(50.0));
        assert_eq!(pnl.total, dec!(150.0));
    }

    #[test]
    fn test_pnl_combined_operations() {
        let mut pnl = PnL::new();

        // Realize some profit
        pnl.add_realized(dec!(100.0));
        assert_eq!(pnl.total, dec!(100.0));

        // Update unrealized
        pnl.set_unrealized(dec!(50.0));
        assert_eq!(pnl.total, dec!(150.0));

        // Add more realized
        pnl.add_realized(dec!(25.0));
        assert_eq!(pnl.realized, dec!(125.0));
        assert_eq!(pnl.total, dec!(175.0));

        // Update unrealized to loss
        pnl.set_unrealized(dec!(-30.0));
        assert_eq!(pnl.total, dec!(95.0));
    }
}
