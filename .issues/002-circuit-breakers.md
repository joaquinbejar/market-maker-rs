# Implement Circuit Breakers

## Summary

Add automatic trading halts when adverse conditions are detected to protect against catastrophic losses.

## Motivation

Circuit breakers are essential safety mechanisms that prevent runaway losses during extreme market conditions, system malfunctions, or strategy failures. They provide automatic protection without requiring manual intervention.

## Detailed Description

Implement a `CircuitBreaker` system that monitors trading activity and halts operations when thresholds are breached.

### Trigger Conditions

1. **Max Daily Loss**: Stop trading when cumulative daily loss exceeds threshold
2. **Volatility Spike**: Pause when market volatility exceeds normal range
3. **Consecutive Losses**: Halt after N consecutive losing trades
4. **Rapid Drawdown**: Stop if equity drops X% within Y minutes

### Proposed API

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    /// Normal operation
    Active,
    /// Trading halted due to trigger
    Triggered { reason: TriggerReason, triggered_at: u64 },
    /// Cooling down before resuming
    Cooldown { resume_at: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TriggerReason {
    MaxDailyLoss,
    VolatilitySpike,
    ConsecutiveLosses,
    RapidDrawdown,
    Manual,
}

pub struct CircuitBreakerConfig {
    pub max_daily_loss: Decimal,
    pub max_volatility: Decimal,
    pub max_consecutive_losses: u32,
    pub rapid_drawdown_threshold: Decimal,
    pub rapid_drawdown_window_ms: u64,
    pub cooldown_duration_ms: u64,
}

pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitBreakerState,
    // ... internal tracking fields
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self;
    
    /// Check current state and update if needed
    pub fn check(&mut self, context: &TradingContext) -> CircuitBreakerState;
    
    /// Record a trade result
    pub fn record_trade(&mut self, pnl: Decimal, timestamp: u64);
    
    /// Update with current volatility
    pub fn update_volatility(&mut self, volatility: Decimal);
    
    /// Manually trigger circuit breaker
    pub fn trigger_manual(&mut self, timestamp: u64);
    
    /// Reset circuit breaker (e.g., at start of new day)
    pub fn reset(&mut self);
    
    /// Check if trading is allowed
    pub fn is_trading_allowed(&self) -> bool;
}
```

## Acceptance Criteria

- [ ] `CircuitBreakerState` enum with `Active`, `Triggered`, `Cooldown` variants
- [ ] `TriggerReason` enum for different trigger types
- [ ] `CircuitBreakerConfig` with all configurable thresholds
- [ ] `CircuitBreaker` struct with state management
- [ ] `check()` evaluates all conditions and updates state
- [ ] `record_trade()` tracks PnL for loss-based triggers
- [ ] `update_volatility()` for volatility-based triggers
- [ ] `is_trading_allowed()` returns false when triggered or in cooldown
- [ ] Automatic transition from `Triggered` to `Cooldown` to `Active`
- [ ] Unit tests for each trigger condition
- [ ] Unit tests for state transitions
- [ ] Documentation with configuration examples

## Technical Notes

- Use timestamps in milliseconds (u64) for consistency with rest of codebase
- Consider using a state machine pattern for clean state transitions
- Cooldown should be configurable per trigger type in future iterations

## Labels

`enhancement`, `risk`, `priority:high`
