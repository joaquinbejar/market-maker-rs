# Implement Grid Trading Strategy

## Summary

Add a grid trading strategy that places orders at fixed price intervals around a reference price.

## Motivation

Grid trading is a popular market making approach that profits from price oscillations within a range. It's simpler than Avellaneda-Stoikov and works well in ranging markets.

## Detailed Description

Implement a `GridStrategy` that generates a series of buy and sell orders at predetermined price levels.

### Strategy Logic

1. Define a grid with N levels above and below a reference price
2. Place buy orders below reference, sell orders above
3. When an order fills, place a new order on the opposite side
4. Adjust grid based on inventory to manage risk

### Proposed API

```rust
pub struct GridConfig {
    /// Number of grid levels on each side
    pub levels_per_side: u32,
    
    /// Spacing between levels (as percentage, e.g., 0.005 for 0.5%)
    pub grid_spacing: Decimal,
    
    /// Base order size per level
    pub base_size: Decimal,
    
    /// Size multiplier for levels further from mid (optional)
    pub size_progression: Option<Decimal>,
    
    /// Maximum total position allowed
    pub max_position: Decimal,
}

pub struct GridStrategy {
    config: GridConfig,
    reference_price: Decimal,
}

pub struct GridOrder {
    pub price: Decimal,
    pub size: Decimal,
    pub side: OrderSide,
    pub level: i32,  // Negative for bids, positive for asks
}

impl GridStrategy {
    pub fn new(config: GridConfig) -> MMResult<Self>;
    
    /// Generate all grid orders around reference price
    pub fn generate_grid(&self, reference_price: Decimal) -> Vec<GridOrder>;
    
    /// Generate grid orders adjusted for current inventory
    pub fn generate_grid_with_inventory(
        &self,
        reference_price: Decimal,
        current_inventory: Decimal,
    ) -> Vec<GridOrder>;
    
    /// Update reference price (e.g., based on moving average)
    pub fn update_reference_price(&mut self, price: Decimal);
    
    /// Calculate order size for a specific level
    pub fn calculate_level_size(&self, level: i32) -> Decimal;
}
```

## Acceptance Criteria

- [ ] `GridConfig` struct with all configuration parameters
- [ ] `GridStrategy` struct implementing grid logic
- [ ] `GridOrder` struct representing individual grid orders
- [ ] `generate_grid()` creates symmetric grid around reference price
- [ ] `generate_grid_with_inventory()` adjusts for position risk
- [ ] Support for size progression (larger sizes at better prices)
- [ ] Validation of config parameters
- [ ] Unit tests covering:
  - Symmetric grid generation
  - Inventory-adjusted grid
  - Size progression
  - Edge cases (single level, zero spacing)
- [ ] Example demonstrating grid strategy usage
- [ ] Documentation with strategy explanation

## Technical Notes

- Grid levels should be calculated as: `reference_price * (1 ± level * grid_spacing)`
- Consider adding support for arithmetic vs geometric grid spacing
- Inventory adjustment should reduce size on the side that would increase position

## Labels

`enhancement`, `strategy`, `priority:medium`
