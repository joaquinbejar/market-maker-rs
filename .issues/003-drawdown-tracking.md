# Implement Drawdown Tracking

## Summary

Track and limit maximum drawdown from peak equity to manage risk over time.

## Motivation

Drawdown is a critical risk metric that measures the decline from a historical peak. Tracking drawdown helps identify strategy degradation and prevents excessive capital loss.

## Detailed Description

Implement a `DrawdownTracker` that maintains peak equity and calculates current drawdown in real-time.

### Proposed API

```rust
pub struct DrawdownTracker {
    /// Historical peak equity value
    peak_equity: Decimal,
    
    /// Current equity value
    current_equity: Decimal,
    
    /// Maximum allowed drawdown (as decimal, e.g., 0.10 for 10%)
    max_allowed_drawdown: Decimal,
    
    /// Timestamp of peak equity
    peak_timestamp: u64,
    
    /// Historical drawdown records
    drawdown_history: Vec<DrawdownRecord>,
}

pub struct DrawdownRecord {
    pub drawdown: Decimal,
    pub timestamp: u64,
    pub peak_equity: Decimal,
    pub trough_equity: Decimal,
}

impl DrawdownTracker {
    pub fn new(initial_equity: Decimal, max_allowed_drawdown: Decimal) -> Self;
    
    /// Update with new equity value
    pub fn update(&mut self, equity: Decimal, timestamp: u64);
    
    /// Get current drawdown as decimal (0.0 to 1.0)
    pub fn current_drawdown(&self) -> Decimal;
    
    /// Get current drawdown as percentage string
    pub fn current_drawdown_pct(&self) -> Decimal;
    
    /// Check if max drawdown has been reached
    pub fn is_max_drawdown_reached(&self) -> bool;
    
    /// Get peak equity value
    pub fn peak_equity(&self) -> Decimal;
    
    /// Get maximum historical drawdown
    pub fn max_historical_drawdown(&self) -> Decimal;
    
    /// Reset tracker (e.g., for new trading period)
    pub fn reset(&mut self, new_equity: Decimal, timestamp: u64);
}
```

## Acceptance Criteria

- [ ] `DrawdownTracker` struct with peak/current equity tracking
- [ ] `DrawdownRecord` for historical drawdown events
- [ ] `update()` correctly updates peak when equity increases
- [ ] `current_drawdown()` returns correct value (peak - current) / peak
- [ ] `is_max_drawdown_reached()` returns true when threshold breached
- [ ] Historical drawdown tracking for analysis
- [ ] `max_historical_drawdown()` returns worst drawdown seen
- [ ] Integration with `PnL` module
- [ ] Unit tests covering:
  - New highs (peak updates)
  - Drawdown calculation accuracy
  - Max drawdown detection
  - Reset functionality
- [ ] Documentation with examples

## Technical Notes

- Drawdown is calculated as: `(peak - current) / peak`
- Store drawdown as positive decimal (0.10 = 10% drawdown)
- Consider adding time-weighted drawdown metrics in future

## Labels

`enhancement`, `risk`, `priority:medium`
