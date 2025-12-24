# Implement Realistic Fill Models

## Summary

Add realistic fill models for more accurate backtesting that accounts for queue position and market impact.

## Motivation

Simple "immediate fill" models overestimate strategy performance because they ignore:
- Queue position (orders ahead of yours)
- Partial fills
- Market impact of your orders
- Adverse selection (fills happen when price moves against you)

## Detailed Description

Implement multiple fill models with increasing realism.

### Fill Models

1. **Immediate Fill**: Fill at quote price when market crosses (baseline)
2. **Queue Position**: Simulate queue priority based on time and size
3. **Probabilistic Fill**: Fill probability based on depth and time
4. **Market Impact**: Price impact proportional to order size

### Proposed API

```rust
/// Fill model trait
pub trait FillModel: Send + Sync {
    /// Determine if and how an order fills given market state
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult;
    
    /// Reset model state (e.g., queue positions)
    fn reset(&mut self);
}

/// Simulated order for fill model
#[derive(Debug, Clone)]
pub struct SimulatedOrder {
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub submitted_at: u64,
}

/// Fill result
#[derive(Debug, Clone)]
pub enum FillResult {
    /// No fill occurred
    NoFill,
    
    /// Partial fill
    PartialFill {
        filled_quantity: Decimal,
        fill_price: Decimal,
    },
    
    /// Complete fill
    FullFill {
        fill_price: Decimal,
    },
}

/// Immediate fill model (simplest)
pub struct ImmediateFillModel;

impl FillModel for ImmediateFillModel {
    fn simulate_fill(&self, order: &SimulatedOrder, tick: &MarketTick, _: u64) -> FillResult;
    fn reset(&mut self) {}
}

/// Queue position fill model
pub struct QueuePositionFillModel {
    /// Estimated queue depth at each price level
    queue_depth: HashMap<Decimal, Decimal>,
    
    /// Fill rate (volume per ms that clears queue)
    fill_rate: Decimal,
}

impl QueuePositionFillModel {
    pub fn new(fill_rate: Decimal) -> Self;
    
    /// Update queue estimate from market data
    pub fn update_queue(&mut self, tick: &MarketTick);
}

/// Probabilistic fill model
pub struct ProbabilisticFillModel {
    /// Base fill probability per tick
    base_probability: Decimal,
    
    /// Depth factor (higher = less likely to fill in deep book)
    depth_factor: Decimal,
    
    /// Time factor (higher = more likely to fill over time)
    time_factor: Decimal,
    
    /// Random number generator
    rng: StdRng,
}

impl ProbabilisticFillModel {
    pub fn new(base_probability: Decimal, depth_factor: Decimal, time_factor: Decimal, seed: u64) -> Self;
    
    /// Calculate fill probability for order
    pub fn calculate_probability(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> Decimal;
}

/// Market impact model
pub struct MarketImpactFillModel {
    /// Impact coefficient (price impact = coeff * sqrt(size / adv))
    impact_coefficient: Decimal,
    
    /// Average daily volume for normalization
    average_daily_volume: Decimal,
    
    /// Underlying fill model
    base_model: Box<dyn FillModel>,
}

impl MarketImpactFillModel {
    pub fn new(
        impact_coefficient: Decimal,
        average_daily_volume: Decimal,
        base_model: Box<dyn FillModel>,
    ) -> Self;
    
    /// Calculate price impact for order size
    pub fn calculate_impact(&self, size: Decimal) -> Decimal;
}
```

## Acceptance Criteria

- [ ] `FillModel` trait defining interface
- [ ] `SimulatedOrder` and `FillResult` types
- [ ] `ImmediateFillModel` implementation (baseline)
- [ ] `QueuePositionFillModel` with queue simulation
- [ ] `ProbabilisticFillModel` with configurable parameters
- [ ] `MarketImpactFillModel` with square-root impact
- [ ] Integration with `BacktestEngine`
- [ ] Unit tests covering:
  - Each fill model independently
  - Fill probability calculations
  - Market impact calculations
  - Partial fill scenarios
- [ ] Comparison example showing different model results
- [ ] Documentation explaining each model's assumptions

## Technical Notes

- Queue position model: estimate position based on time priority
- Probabilistic model: `P(fill) = base * exp(-depth_factor * depth) * (1 - exp(-time_factor * time))`
- Market impact: Square-root model is standard: `impact = coeff * sigma * sqrt(size / ADV)`
- Use seeded RNG for reproducible backtests
- Consider adding adverse selection model (fills correlate with price moves against you)

## Labels

`enhancement`, `backtest`, `priority:medium`
