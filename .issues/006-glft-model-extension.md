# Implement Guéant-Lehalle-Fernandez-Tapia Model Extension

## Summary

Extend the Avellaneda-Stoikov model with terminal inventory penalty from the Guéant-Lehalle-Fernandez-Tapia (GLFT) framework.

## Motivation

The original A-S model assumes the market maker can hold inventory at terminal time. The GLFT extension adds a penalty for terminal inventory, making the model more realistic for strategies that must flatten positions.

## Detailed Description

The GLFT model modifies the reservation price and spread calculations to account for:

1. **Terminal Inventory Penalty**: Cost of liquidating remaining inventory at T
2. **Dynamic Risk Aversion**: Gamma increases as terminal time approaches
3. **Inventory Skew**: More aggressive skew to flatten position near terminal

### Mathematical Extension

Original A-S reservation price:
```
r = s - q * γ * σ² * (T - t)
```

GLFT modification:
```
r = s - q * γ * σ² * (T - t) - q * φ * f(T - t)
```

Where `φ` is the terminal penalty parameter and `f(T-t)` is a function that increases as terminal approaches.

### Proposed API

```rust
pub struct GLFTConfig {
    /// Base risk aversion (gamma)
    pub risk_aversion: Decimal,
    
    /// Order intensity (k)
    pub order_intensity: Decimal,
    
    /// Terminal inventory penalty (phi)
    pub terminal_penalty: Decimal,
    
    /// Terminal time in milliseconds
    pub terminal_time: u64,
    
    /// Minimum spread constraint
    pub min_spread: Decimal,
    
    /// Whether to use dynamic gamma scaling
    pub dynamic_gamma: bool,
}

pub struct GLFTStrategy;

impl GLFTStrategy {
    /// Calculate reservation price with terminal penalty
    pub fn calculate_reservation_price(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<Decimal>;
    
    /// Calculate optimal spread with GLFT adjustments
    pub fn calculate_optimal_spread(
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<Decimal>;
    
    /// Calculate optimal quotes
    pub fn calculate_optimal_quotes(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<(Decimal, Decimal)>;
    
    /// Calculate dynamic gamma based on time to terminal
    pub fn calculate_dynamic_gamma(
        base_gamma: Decimal,
        time_to_terminal_ms: u64,
        total_session_ms: u64,
    ) -> Decimal;
}
```

## Acceptance Criteria

- [ ] `GLFTConfig` struct with terminal penalty parameter
- [ ] `GLFTStrategy` implementing extended model
- [ ] `calculate_reservation_price()` with terminal penalty term
- [ ] `calculate_optimal_spread()` with GLFT adjustments
- [ ] `calculate_dynamic_gamma()` for time-varying risk aversion
- [ ] Comparison tests showing difference from base A-S model
- [ ] Unit tests covering:
  - Terminal penalty effect on reservation price
  - Behavior as time approaches terminal
  - Dynamic gamma scaling
  - Edge cases (at terminal, far from terminal)
- [ ] Documentation with mathematical derivation
- [ ] Reference to GLFT paper in docs

## Technical Notes

- Paper reference: Guéant, Lehalle, Fernandez-Tapia (2012)
- Terminal penalty should increase inventory skew near end of session
- Dynamic gamma: `γ_t = γ_0 * (1 + α * (1 - (T-t)/T))` where α controls scaling
- Consider making penalty function configurable (linear, exponential, etc.)

## Labels

`enhancement`, `strategy`, `priority:low`
