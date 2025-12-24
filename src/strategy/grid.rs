//! Grid trading strategy implementation.
//!
//! Grid trading places orders at fixed price intervals around a reference price,
//! profiting from price oscillations within a range.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Order side for grid orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OrderSide {
    /// Buy order (bid).
    Buy,
    /// Sell order (ask).
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => write!(f, "Buy"),
            Self::Sell => write!(f, "Sell"),
        }
    }
}

/// Grid spacing type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GridSpacingType {
    /// Geometric spacing: price * (1 ± level * spacing).
    #[default]
    Geometric,
    /// Arithmetic spacing: price ± level * spacing * price.
    Arithmetic,
}

/// Configuration for the grid trading strategy.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::grid::GridConfig;
/// use market_maker_rs::dec;
///
/// let config = GridConfig::new(
///     5,              // 5 levels per side
///     dec!(0.005),    // 0.5% spacing
///     dec!(1.0),      // 1 unit base size
///     dec!(100.0),    // max 100 units position
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GridConfig {
    /// Number of grid levels on each side (buy and sell).
    pub levels_per_side: u32,

    /// Spacing between levels as decimal (e.g., 0.005 for 0.5%).
    pub grid_spacing: Decimal,

    /// Base order size per level in units.
    pub base_size: Decimal,

    /// Size multiplier for levels further from mid (optional).
    ///
    /// If set, size at level N = base_size * (1 + (N-1) * progression).
    /// For example, with progression=0.2, level 3 has size = base * 1.4.
    pub size_progression: Option<Decimal>,

    /// Maximum total position allowed in units.
    pub max_position: Decimal,

    /// Grid spacing type (geometric or arithmetic).
    pub spacing_type: GridSpacingType,
}

impl GridConfig {
    /// Creates a new `GridConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `levels_per_side` - Number of levels on each side (must be > 0)
    /// * `grid_spacing` - Spacing between levels as decimal (must be > 0)
    /// * `base_size` - Base order size per level (must be > 0)
    /// * `max_position` - Maximum total position (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if any parameter is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::grid::GridConfig;
    /// use market_maker_rs::dec;
    ///
    /// let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0)).unwrap();
    /// assert_eq!(config.levels_per_side, 5);
    /// ```
    pub fn new(
        levels_per_side: u32,
        grid_spacing: Decimal,
        base_size: Decimal,
        max_position: Decimal,
    ) -> MMResult<Self> {
        if levels_per_side == 0 {
            return Err(MMError::InvalidConfiguration(
                "levels_per_side must be greater than 0".to_string(),
            ));
        }

        if grid_spacing <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "grid_spacing must be positive".to_string(),
            ));
        }

        if base_size <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "base_size must be positive".to_string(),
            ));
        }

        if max_position <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "max_position must be positive".to_string(),
            ));
        }

        Ok(Self {
            levels_per_side,
            grid_spacing,
            base_size,
            size_progression: None,
            max_position,
            spacing_type: GridSpacingType::default(),
        })
    }

    /// Sets the size progression factor.
    ///
    /// # Arguments
    ///
    /// * `progression` - Size multiplier per level (must be >= 0)
    #[must_use]
    pub fn with_size_progression(mut self, progression: Decimal) -> Self {
        self.size_progression = Some(progression);
        self
    }

    /// Sets the grid spacing type.
    #[must_use]
    pub fn with_spacing_type(mut self, spacing_type: GridSpacingType) -> Self {
        self.spacing_type = spacing_type;
        self
    }
}

/// A single order in the grid.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GridOrder {
    /// Order price.
    pub price: Decimal,
    /// Order size in units.
    pub size: Decimal,
    /// Order side (buy or sell).
    pub side: OrderSide,
    /// Grid level (negative for bids, positive for asks, 0 is reference).
    pub level: i32,
}

impl GridOrder {
    /// Creates a new grid order.
    #[must_use]
    pub fn new(price: Decimal, size: Decimal, side: OrderSide, level: i32) -> Self {
        Self {
            price,
            size,
            side,
            level,
        }
    }

    /// Returns the notional value of this order.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price * self.size
    }
}

/// Grid trading strategy.
///
/// Places orders at fixed price intervals around a reference price.
/// Buy orders are placed below the reference, sell orders above.
///
/// # Strategy Logic
///
/// 1. Define a grid with N levels above and below a reference price
/// 2. Place buy orders below reference, sell orders above
/// 3. Adjust grid based on inventory to manage risk
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::grid::{GridStrategy, GridConfig, OrderSide};
/// use market_maker_rs::dec;
///
/// let config = GridConfig::new(3, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
/// let strategy = GridStrategy::new(config).unwrap();
///
/// let orders = strategy.generate_grid(dec!(100.0));
///
/// // Should have 3 buy orders below and 3 sell orders above
/// assert_eq!(orders.len(), 6);
///
/// let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
/// let sells: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Sell).collect();
/// assert_eq!(buys.len(), 3);
/// assert_eq!(sells.len(), 3);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GridStrategy {
    /// Strategy configuration.
    config: GridConfig,
    /// Current reference price for the grid.
    reference_price: Decimal,
}

impl GridStrategy {
    /// Creates a new grid strategy with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Grid configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::grid::{GridStrategy, GridConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0)).unwrap();
    /// let strategy = GridStrategy::new(config).unwrap();
    /// ```
    pub fn new(config: GridConfig) -> MMResult<Self> {
        Ok(Self {
            config,
            reference_price: Decimal::ZERO,
        })
    }

    /// Creates a new grid strategy with an initial reference price.
    pub fn with_reference_price(config: GridConfig, reference_price: Decimal) -> MMResult<Self> {
        if reference_price <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "reference_price must be positive".to_string(),
            ));
        }

        Ok(Self {
            config,
            reference_price,
        })
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &GridConfig {
        &self.config
    }

    /// Returns the current reference price.
    #[must_use]
    pub fn reference_price(&self) -> Decimal {
        self.reference_price
    }

    /// Updates the reference price.
    ///
    /// # Arguments
    ///
    /// * `price` - New reference price
    pub fn update_reference_price(&mut self, price: Decimal) {
        self.reference_price = price;
    }

    /// Generates all grid orders around the reference price.
    ///
    /// Creates symmetric buy and sell orders at each grid level.
    ///
    /// # Arguments
    ///
    /// * `reference_price` - Center price for the grid
    ///
    /// # Returns
    ///
    /// Vector of grid orders, sorted by price (lowest to highest).
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::grid::{GridStrategy, GridConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
    /// let strategy = GridStrategy::new(config).unwrap();
    ///
    /// let orders = strategy.generate_grid(dec!(100.0));
    /// assert_eq!(orders.len(), 4); // 2 buys + 2 sells
    /// ```
    #[must_use]
    pub fn generate_grid(&self, reference_price: Decimal) -> Vec<GridOrder> {
        let mut orders = Vec::with_capacity((self.config.levels_per_side * 2) as usize);

        // Generate buy orders (below reference)
        for level in 1..=self.config.levels_per_side {
            let price = self.calculate_price(reference_price, -(level as i32));
            let size = self.calculate_level_size(level as i32);

            orders.push(GridOrder::new(price, size, OrderSide::Buy, -(level as i32)));
        }

        // Generate sell orders (above reference)
        for level in 1..=self.config.levels_per_side {
            let price = self.calculate_price(reference_price, level as i32);
            let size = self.calculate_level_size(level as i32);

            orders.push(GridOrder::new(price, size, OrderSide::Sell, level as i32));
        }

        // Sort by price (lowest to highest)
        orders.sort_by(|a, b| a.price.cmp(&b.price));

        orders
    }

    /// Generates grid orders adjusted for current inventory.
    ///
    /// Reduces order sizes on the side that would increase position risk.
    /// - If long, reduce buy sizes
    /// - If short, reduce sell sizes
    ///
    /// # Arguments
    ///
    /// * `reference_price` - Center price for the grid
    /// * `current_inventory` - Current position (positive = long, negative = short)
    ///
    /// # Returns
    ///
    /// Vector of grid orders with adjusted sizes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::grid::{GridStrategy, GridConfig, OrderSide};
    /// use market_maker_rs::dec;
    ///
    /// let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(10.0)).unwrap();
    /// let strategy = GridStrategy::new(config).unwrap();
    ///
    /// // With long inventory, buy sizes should be reduced
    /// let orders = strategy.generate_grid_with_inventory(dec!(100.0), dec!(5.0));
    ///
    /// let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
    /// let sells: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Sell).collect();
    ///
    /// // Buy sizes reduced due to long position
    /// assert!(buys.iter().all(|o| o.size < dec!(1.0)));
    /// // Sell sizes unchanged
    /// assert!(sells.iter().all(|o| o.size == dec!(1.0)));
    /// ```
    #[must_use]
    pub fn generate_grid_with_inventory(
        &self,
        reference_price: Decimal,
        current_inventory: Decimal,
    ) -> Vec<GridOrder> {
        let mut orders = self.generate_grid(reference_price);

        // Calculate inventory ratio (how close to max position)
        let inventory_ratio = current_inventory.abs() / self.config.max_position;
        let scale_factor = (Decimal::ONE - inventory_ratio).max(Decimal::ZERO);

        for order in &mut orders {
            // If long, reduce buy sizes; if short, reduce sell sizes
            let should_reduce = (current_inventory > Decimal::ZERO && order.side == OrderSide::Buy)
                || (current_inventory < Decimal::ZERO && order.side == OrderSide::Sell);

            if should_reduce {
                order.size *= scale_factor;
            }
        }

        // Filter out orders with zero or near-zero size
        orders.retain(|o| o.size > Decimal::new(1, 8)); // > 0.00000001

        orders
    }

    /// Calculates the price for a specific grid level.
    ///
    /// # Arguments
    ///
    /// * `reference_price` - Center price
    /// * `level` - Grid level (negative = below, positive = above)
    #[must_use]
    pub fn calculate_price(&self, reference_price: Decimal, level: i32) -> Decimal {
        let level_decimal = Decimal::from(level);

        match self.config.spacing_type {
            GridSpacingType::Geometric => {
                // price = reference * (1 + level * spacing)
                reference_price * (Decimal::ONE + level_decimal * self.config.grid_spacing)
            }
            GridSpacingType::Arithmetic => {
                // price = reference + level * spacing * reference
                reference_price + level_decimal * self.config.grid_spacing * reference_price
            }
        }
    }

    /// Calculates the order size for a specific level.
    ///
    /// If size progression is enabled, levels further from the reference
    /// have larger sizes.
    ///
    /// # Arguments
    ///
    /// * `level` - Grid level (absolute value used)
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::grid::{GridStrategy, GridConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0))
    ///     .unwrap()
    ///     .with_size_progression(dec!(0.2));
    /// let strategy = GridStrategy::new(config).unwrap();
    ///
    /// // Level 1: base_size * (1 + 0 * 0.2) = 1.0
    /// assert_eq!(strategy.calculate_level_size(1), dec!(1.0));
    ///
    /// // Level 3: base_size * (1 + 2 * 0.2) = 1.4
    /// assert_eq!(strategy.calculate_level_size(3), dec!(1.4));
    /// ```
    #[must_use]
    pub fn calculate_level_size(&self, level: i32) -> Decimal {
        let abs_level = level.unsigned_abs();

        match self.config.size_progression {
            Some(progression) => {
                // size = base_size * (1 + (level - 1) * progression)
                let multiplier =
                    Decimal::ONE + Decimal::from(abs_level.saturating_sub(1)) * progression;
                self.config.base_size * multiplier
            }
            None => self.config.base_size,
        }
    }

    /// Returns the total number of orders in a full grid.
    #[must_use]
    pub fn total_orders(&self) -> u32 {
        self.config.levels_per_side * 2
    }

    /// Calculates the price range covered by the grid.
    ///
    /// # Arguments
    ///
    /// * `reference_price` - Center price
    ///
    /// # Returns
    ///
    /// Tuple of (lowest_price, highest_price).
    #[must_use]
    pub fn price_range(&self, reference_price: Decimal) -> (Decimal, Decimal) {
        let lowest = self.calculate_price(reference_price, -(self.config.levels_per_side as i32));
        let highest = self.calculate_price(reference_price, self.config.levels_per_side as i32);
        (lowest, highest)
    }

    /// Calculates the maximum notional exposure if all orders fill.
    ///
    /// # Arguments
    ///
    /// * `reference_price` - Center price for calculation
    #[must_use]
    pub fn max_notional_exposure(&self, reference_price: Decimal) -> Decimal {
        let orders = self.generate_grid(reference_price);
        orders.iter().map(|o| o.notional()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_config_valid() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0));
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.levels_per_side, 5);
        assert_eq!(config.grid_spacing, dec!(0.005));
        assert_eq!(config.base_size, dec!(1.0));
        assert_eq!(config.max_position, dec!(100.0));
    }

    #[test]
    fn test_config_invalid_levels() {
        let config = GridConfig::new(0, dec!(0.005), dec!(1.0), dec!(100.0));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_spacing() {
        let config = GridConfig::new(5, dec!(0.0), dec!(1.0), dec!(100.0));
        assert!(config.is_err());

        let config = GridConfig::new(5, dec!(-0.005), dec!(1.0), dec!(100.0));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_base_size() {
        let config = GridConfig::new(5, dec!(0.005), dec!(0.0), dec!(100.0));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_max_position() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(0.0));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_with_progression() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0))
            .unwrap()
            .with_size_progression(dec!(0.2));

        assert_eq!(config.size_progression, Some(dec!(0.2)));
    }

    #[test]
    fn test_strategy_new() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config);
        assert!(strategy.is_ok());
    }

    #[test]
    fn test_strategy_with_reference_price() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::with_reference_price(config, dec!(100.0)).unwrap();
        assert_eq!(strategy.reference_price(), dec!(100.0));
    }

    #[test]
    fn test_strategy_invalid_reference_price() {
        let config = GridConfig::new(5, dec!(0.005), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::with_reference_price(config, dec!(0.0));
        assert!(strategy.is_err());
    }

    #[test]
    fn test_generate_grid_symmetric() {
        let config = GridConfig::new(3, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        let orders = strategy.generate_grid(dec!(100.0));

        assert_eq!(orders.len(), 6); // 3 buys + 3 sells

        let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
        let sells: Vec<_> = orders
            .iter()
            .filter(|o| o.side == OrderSide::Sell)
            .collect();

        assert_eq!(buys.len(), 3);
        assert_eq!(sells.len(), 3);

        // All buys should be below reference
        assert!(buys.iter().all(|o| o.price < dec!(100.0)));
        // All sells should be above reference
        assert!(sells.iter().all(|o| o.price > dec!(100.0)));
    }

    #[test]
    fn test_generate_grid_prices() {
        let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        let orders = strategy.generate_grid(dec!(100.0));

        // Expected prices with 1% spacing:
        // Buy level -2: 100 * (1 - 0.02) = 98
        // Buy level -1: 100 * (1 - 0.01) = 99
        // Sell level 1: 100 * (1 + 0.01) = 101
        // Sell level 2: 100 * (1 + 0.02) = 102

        let prices: Vec<_> = orders.iter().map(|o| o.price).collect();
        assert_eq!(prices, vec![dec!(98), dec!(99), dec!(101), dec!(102)]);
    }

    #[test]
    fn test_generate_grid_with_inventory_long() {
        let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(10.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        // 50% of max position long
        let orders = strategy.generate_grid_with_inventory(dec!(100.0), dec!(5.0));

        let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
        let sells: Vec<_> = orders
            .iter()
            .filter(|o| o.side == OrderSide::Sell)
            .collect();

        // Buy sizes should be reduced by 50%
        assert!(buys.iter().all(|o| o.size == dec!(0.5)));
        // Sell sizes unchanged
        assert!(sells.iter().all(|o| o.size == dec!(1.0)));
    }

    #[test]
    fn test_generate_grid_with_inventory_short() {
        let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(10.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        // 50% of max position short
        let orders = strategy.generate_grid_with_inventory(dec!(100.0), dec!(-5.0));

        let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
        let sells: Vec<_> = orders
            .iter()
            .filter(|o| o.side == OrderSide::Sell)
            .collect();

        // Buy sizes unchanged
        assert!(buys.iter().all(|o| o.size == dec!(1.0)));
        // Sell sizes should be reduced by 50%
        assert!(sells.iter().all(|o| o.size == dec!(0.5)));
    }

    #[test]
    fn test_generate_grid_with_max_inventory() {
        let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(10.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        // At max position, buy orders should be filtered out
        let orders = strategy.generate_grid_with_inventory(dec!(100.0), dec!(10.0));

        let buys: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
        assert!(buys.is_empty());
    }

    #[test]
    fn test_calculate_level_size_no_progression() {
        let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        assert_eq!(strategy.calculate_level_size(1), dec!(1.0));
        assert_eq!(strategy.calculate_level_size(3), dec!(1.0));
        assert_eq!(strategy.calculate_level_size(5), dec!(1.0));
    }

    #[test]
    fn test_calculate_level_size_with_progression() {
        let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0))
            .unwrap()
            .with_size_progression(dec!(0.2));
        let strategy = GridStrategy::new(config).unwrap();

        // Level 1: 1.0 * (1 + 0 * 0.2) = 1.0
        assert_eq!(strategy.calculate_level_size(1), dec!(1.0));

        // Level 2: 1.0 * (1 + 1 * 0.2) = 1.2
        assert_eq!(strategy.calculate_level_size(2), dec!(1.2));

        // Level 3: 1.0 * (1 + 2 * 0.2) = 1.4
        assert_eq!(strategy.calculate_level_size(3), dec!(1.4));
    }

    #[test]
    fn test_price_range() {
        let config = GridConfig::new(3, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        let (low, high) = strategy.price_range(dec!(100.0));

        // Low: 100 * (1 - 3 * 0.01) = 97
        assert_eq!(low, dec!(97));
        // High: 100 * (1 + 3 * 0.01) = 103
        assert_eq!(high, dec!(103));
    }

    #[test]
    fn test_total_orders() {
        let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let strategy = GridStrategy::new(config).unwrap();

        assert_eq!(strategy.total_orders(), 10);
    }

    #[test]
    fn test_arithmetic_spacing() {
        let config = GridConfig::new(2, dec!(0.01), dec!(1.0), dec!(100.0))
            .unwrap()
            .with_spacing_type(GridSpacingType::Arithmetic);
        let strategy = GridStrategy::new(config).unwrap();

        // Arithmetic spacing gives same result as geometric for percentage-based
        let orders = strategy.generate_grid(dec!(100.0));
        let prices: Vec<_> = orders.iter().map(|o| o.price).collect();

        assert_eq!(prices, vec![dec!(98), dec!(99), dec!(101), dec!(102)]);
    }

    #[test]
    fn test_order_side_display() {
        assert_eq!(OrderSide::Buy.to_string(), "Buy");
        assert_eq!(OrderSide::Sell.to_string(), "Sell");
    }

    #[test]
    fn test_grid_order_notional() {
        let order = GridOrder::new(dec!(100.0), dec!(5.0), OrderSide::Buy, -1);
        assert_eq!(order.notional(), dec!(500.0));
    }

    #[test]
    fn test_update_reference_price() {
        let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();
        let mut strategy = GridStrategy::new(config).unwrap();

        strategy.update_reference_price(dec!(150.0));
        assert_eq!(strategy.reference_price(), dec!(150.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config = GridConfig::new(5, dec!(0.01), dec!(1.0), dec!(100.0)).unwrap();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: GridConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
