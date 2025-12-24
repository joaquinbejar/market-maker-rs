# Implement Position Limits and Exposure Control

## Summary

Add position limits and exposure control to prevent excessive risk-taking during market making operations.

## Motivation

A market maker without proper position limits can accumulate dangerous inventory levels, leading to significant losses during adverse market movements. This feature is critical for production deployment.

## Detailed Description

Implement a `RiskLimits` struct that enforces:

1. **Maximum Position Size**: Absolute limit on inventory (long or short)
2. **Maximum Notional Exposure**: Limit based on position value (quantity × price)
3. **Position Scaling**: Reduce order sizes as position approaches limits

### Proposed API

```rust
pub struct RiskLimits {
    /// Maximum absolute position size (units)
    pub max_position: Decimal,
    
    /// Maximum notional exposure (currency units)
    pub max_notional: Decimal,
    
    /// Factor to scale orders as position grows (0.0 to 1.0)
    pub scaling_factor: Decimal,
}

impl RiskLimits {
    /// Check if a new order would violate position limits
    pub fn check_order(&self, current_position: Decimal, order_size: Decimal, price: Decimal) -> MMResult<bool>;
    
    /// Calculate scaled order size based on current position
    pub fn scale_order_size(&self, current_position: Decimal, desired_size: Decimal) -> Decimal;
    
    /// Check if position limit is breached
    pub fn is_position_limit_breached(&self, position: Decimal) -> bool;
}
```

## Acceptance Criteria

- [ ] `RiskLimits` struct with `max_position`, `max_notional`, `scaling_factor` fields
- [ ] `check_order()` method returns `MMResult<bool>` indicating if order is allowed
- [ ] `scale_order_size()` reduces order size as position approaches limit
- [ ] `is_position_limit_breached()` returns true when limits exceeded
- [ ] Validation in constructor for positive limits
- [ ] Unit tests covering:
  - Order within limits
  - Order exceeding position limit
  - Order exceeding notional limit
  - Scaling behavior near limits
  - Edge cases (zero position, exactly at limit)
- [ ] Documentation with usage examples
- [ ] Integration example with existing strategies

## Technical Notes

- Use `Decimal` for all calculations to maintain precision
- Return `MMError::InvalidPositionUpdate` for limit violations
- Consider thread-safety for concurrent access

## Labels

`enhancement`, `risk`, `priority:high`
