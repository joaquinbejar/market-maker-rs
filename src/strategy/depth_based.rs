//! Depth-based offering strategy.
//!
//! This strategy places orders at a target depth in the order book, adjusting
//! position sizes based on current inventory to maintain a target exposure.
//!
//! # Strategy Logic
//!
//! 1. Find the price at the target depth in the order book
//! 2. Place orders one tick inside that price
//! 3. Adjust order sizes based on current inventory and max exposure
//! 4. Cancel orders that don't match the target price
//!
//! # Examples
//!
//! ```ignore
//! use market_maker_rs::strategy::depth_based::DepthBasedOffering;
//! use market_maker_rs::dec;
//!
//! let strategy = DepthBasedOffering::new(
//!     dec!(100.0),  // max_exposure: maximum position size
//!     dec!(50.0),   // target_depth: depth to place orders at
//! );
//! ```

use crate::Decimal;
use crate::types::decimal::OrOverflow;
use crate::types::error::MMResult;

/// Type alias for price.
pub type Price = Decimal;

/// Type alias for amount/quantity.
pub type Amount = Decimal;

/// Depth-based offering strategy.
///
/// Places orders at a specific depth in the order book to provide liquidity
/// while managing inventory risk through position-based sizing.
///
/// This strategy integrates with `orderbook-rs` crate for order book management.
#[derive(Debug, Clone)]
pub struct DepthBasedOffering {
    max_exposure: Amount,
    target_depth: Amount,
}

impl DepthBasedOffering {
    /// Creates a new depth-based offering strategy.
    ///
    /// # Arguments
    ///
    /// * `max_exposure` - Maximum allowed position size (absolute value)
    /// * `target_depth` - Target cumulative depth in the order book
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::strategy::depth_based::DepthBasedOffering;
    /// use market_maker_rs::dec;
    ///
    /// // Create strategy with 100 units max exposure at 50 units depth
    /// let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
    /// ```
    #[must_use]
    pub fn new(max_exposure: Amount, target_depth: Amount) -> Self {
        Self {
            max_exposure,
            target_depth,
        }
    }

    /// Gets the maximum exposure.
    #[must_use]
    pub fn max_exposure(&self) -> Amount {
        self.max_exposure
    }

    /// Gets the target depth.
    #[must_use]
    pub fn target_depth(&self) -> Amount {
        self.target_depth
    }

    /// Calculates order size for asks based on inventory.
    ///
    /// # Arguments
    ///
    /// * `inventory_position` - Current inventory (positive = long, negative = short)
    ///
    /// # Returns
    ///
    /// Suggested ask (sell) size.
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::strategy::depth_based::DepthBasedOffering;
    /// use market_maker_rs::dec;
    ///
    /// let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
    /// let ask_size = strategy.calculate_ask_size(dec!(20.0)).unwrap(); // Long 20 units
    /// assert_eq!(ask_size, dec!(120.0)); // Sell more to reduce position
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::types::error::MMError::NumericalError`] if the size overflows.
    pub fn calculate_ask_size(&self, inventory_position: Decimal) -> MMResult<Amount> {
        self.max_exposure
            .checked_add(inventory_position)
            .or_overflow("depth-based ask size")
    }

    /// Calculates order size for bids based on inventory.
    ///
    /// # Arguments
    ///
    /// * `inventory_position` - Current inventory (positive = long, negative = short)
    ///
    /// # Returns
    ///
    /// Suggested bid (buy) size.
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::strategy::depth_based::DepthBasedOffering;
    /// use market_maker_rs::dec;
    ///
    /// let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
    /// let bid_size = strategy.calculate_bid_size(dec!(20.0)).unwrap(); // Long 20 units
    /// assert_eq!(bid_size, dec!(80.0)); // Buy less to avoid increasing position
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::types::error::MMError::NumericalError`] if the size overflows.
    pub fn calculate_bid_size(&self, inventory_position: Decimal) -> MMResult<Amount> {
        self.max_exposure
            .checked_sub(inventory_position)
            .or_overflow("depth-based bid size")
    }

    /// Suggests optimal price adjustment based on target depth.
    ///
    /// This method calculates how much to adjust from the current best price
    /// based on the depth target and tick size.
    ///
    /// # Arguments
    ///
    /// * `cumulative_depth_at_level` - Cumulative depth found at a price level
    /// * `tick_size` - Minimum price increment
    /// * `is_ask` - Whether this is for an ask order (true) or bid order (false)
    ///
    /// # Returns
    ///
    /// Price adjustment to apply.
    #[must_use]
    pub fn price_adjustment(
        &self,
        cumulative_depth_at_level: Amount,
        tick_size: Decimal,
        is_ask: bool,
    ) -> Decimal {
        if cumulative_depth_at_level >= self.target_depth {
            if is_ask {
                // Negating a Decimal flips its sign bit: it cannot overflow.
                #[allow(clippy::arithmetic_side_effects)]
                let inside = -tick_size; // Place inside (lower) for asks
                inside
            } else {
                tick_size // Place inside (higher) for bids
            }
        } else {
            Decimal::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_sizes_overflow_returns_numerical_error() {
        let strategy = DepthBasedOffering::new(Decimal::MAX, dec!(50.0));
        let ask = strategy.calculate_ask_size(Decimal::ONE);
        assert!(
            matches!(ask, Err(crate::types::error::MMError::NumericalError(_))),
            "{ask:?}"
        );
        let bid = strategy.calculate_bid_size(Decimal::NEGATIVE_ONE);
        assert!(
            matches!(bid, Err(crate::types::error::MMError::NumericalError(_))),
            "{bid:?}"
        );
    }

    #[test]
    fn test_depth_based_offering_creation() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        assert_eq!(strategy.max_exposure(), dec!(100.0));
        assert_eq!(strategy.target_depth(), dec!(50.0));
    }

    #[test]
    fn test_calculate_ask_size_flat_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let ask_size = strategy.calculate_ask_size(Decimal::ZERO).unwrap();
        assert_eq!(ask_size, dec!(100.0));
    }

    #[test]
    fn test_calculate_ask_size_long_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let ask_size = strategy.calculate_ask_size(dec!(20.0)).unwrap();
        // With long position, ask size increases to reduce exposure
        assert_eq!(ask_size, dec!(120.0));
    }

    #[test]
    fn test_calculate_ask_size_short_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let ask_size = strategy.calculate_ask_size(dec!(-20.0)).unwrap();
        // With short position, ask size decreases
        assert_eq!(ask_size, dec!(80.0));
    }

    #[test]
    fn test_calculate_bid_size_flat_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let bid_size = strategy.calculate_bid_size(Decimal::ZERO).unwrap();
        assert_eq!(bid_size, dec!(100.0));
    }

    #[test]
    fn test_calculate_bid_size_long_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let bid_size = strategy.calculate_bid_size(dec!(20.0)).unwrap();
        // With long position, bid size decreases to avoid increasing exposure
        assert_eq!(bid_size, dec!(80.0));
    }

    #[test]
    fn test_calculate_bid_size_short_inventory() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));
        let bid_size = strategy.calculate_bid_size(dec!(-20.0)).unwrap();
        // With short position, bid size increases to reduce exposure
        assert_eq!(bid_size, dec!(120.0));
    }

    #[test]
    fn test_price_adjustment_for_ask() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));

        // When depth target is met, adjust price inside by one tick
        let adjustment = strategy.price_adjustment(dec!(50.0), dec!(0.01), true);
        assert_eq!(adjustment, dec!(-0.01)); // Lower price for asks

        // When depth target not met, no adjustment
        let adjustment = strategy.price_adjustment(dec!(40.0), dec!(0.01), true);
        assert_eq!(adjustment, Decimal::ZERO);
    }

    #[test]
    fn test_price_adjustment_for_bid() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));

        // When depth target is met, adjust price inside by one tick
        let adjustment = strategy.price_adjustment(dec!(50.0), dec!(0.01), false);
        assert_eq!(adjustment, dec!(0.01)); // Higher price for bids

        // When depth target not met, no adjustment
        let adjustment = strategy.price_adjustment(dec!(40.0), dec!(0.01), false);
        assert_eq!(adjustment, Decimal::ZERO);
    }

    #[test]
    fn test_symmetric_sizing() {
        let strategy = DepthBasedOffering::new(dec!(100.0), dec!(50.0));

        // Ask size + Bid size should equal 2 * max_exposure when flat
        let ask = strategy.calculate_ask_size(Decimal::ZERO).unwrap();
        let bid = strategy.calculate_bid_size(Decimal::ZERO).unwrap();
        assert_eq!(ask + bid, dec!(200.0));

        // Should still hold with inventory
        let ask = strategy.calculate_ask_size(dec!(30.0)).unwrap();
        let bid = strategy.calculate_bid_size(dec!(30.0)).unwrap();
        assert_eq!(ask + bid, dec!(200.0));
    }
}
