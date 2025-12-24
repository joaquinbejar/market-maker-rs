# Implement Live Metrics Tracking

## Summary

Track real-time operational metrics for monitoring live trading systems.

## Motivation

Live metrics are essential for:
- Monitoring system health
- Detecting issues early
- Performance dashboards
- Alerting on anomalies
- Post-session analysis

## Detailed Description

Implement a `LiveMetrics` tracker that maintains real-time counters and gauges for trading operations.

### Proposed API

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Atomic counter for thread-safe incrementing
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub fn new() -> Self { Self(AtomicU64::new(0)) }
    pub fn increment(&self) { self.0.fetch_add(1, Ordering::Relaxed); }
    pub fn add(&self, n: u64) { self.0.fetch_add(n, Ordering::Relaxed); }
    pub fn get(&self) -> u64 { self.0.load(Ordering::Relaxed) }
    pub fn reset(&self) { self.0.store(0, Ordering::Relaxed); }
}

/// Gauge for values that go up and down
#[derive(Debug, Default)]
pub struct Gauge(AtomicI64);

impl Gauge {
    pub fn new() -> Self { Self(AtomicI64::new(0)) }
    pub fn set(&self, value: i64) { self.0.store(value, Ordering::Relaxed); }
    pub fn get(&self) -> i64 { self.0.load(Ordering::Relaxed) }
    pub fn increment(&self) { self.0.fetch_add(1, Ordering::Relaxed); }
    pub fn decrement(&self) { self.0.fetch_sub(1, Ordering::Relaxed); }
}

/// Live metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: u64,
    
    // Counters
    pub quotes_generated: u64,
    pub orders_submitted: u64,
    pub orders_filled: u64,
    pub orders_cancelled: u64,
    pub orders_rejected: u64,
    pub partial_fills: u64,
    
    // Gauges
    pub open_orders: i64,
    pub current_position: Decimal,
    pub current_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    
    // Rates (calculated)
    pub fill_rate: Decimal,          // fills / quotes
    pub cancel_rate: Decimal,        // cancels / orders
    pub quotes_per_second: Decimal,
    pub fills_per_second: Decimal,
    
    // Timing
    pub uptime_ms: u64,
    pub last_quote_time: u64,
    pub last_fill_time: u64,
}

/// Live metrics tracker
pub struct LiveMetrics {
    start_time: u64,
    
    // Counters
    quotes_generated: Counter,
    orders_submitted: Counter,
    orders_filled: Counter,
    orders_cancelled: Counter,
    orders_rejected: Counter,
    partial_fills: Counter,
    
    // Gauges
    open_orders: Gauge,
    
    // Values requiring mutex (Decimal not atomic)
    position: Arc<RwLock<Decimal>>,
    realized_pnl: Arc<RwLock<Decimal>>,
    unrealized_pnl: Arc<RwLock<Decimal>>,
    
    // Timestamps
    last_quote_time: AtomicU64,
    last_fill_time: AtomicU64,
}

impl LiveMetrics {
    pub fn new(start_time: u64) -> Self;
    
    // Counter methods
    pub fn record_quote(&self, timestamp: u64);
    pub fn record_order_submitted(&self);
    pub fn record_order_filled(&self, timestamp: u64);
    pub fn record_order_cancelled(&self);
    pub fn record_order_rejected(&self);
    pub fn record_partial_fill(&self);
    
    // Gauge methods
    pub fn set_open_orders(&self, count: i64);
    pub fn increment_open_orders(&self);
    pub fn decrement_open_orders(&self);
    
    // Position/PnL methods
    pub fn update_position(&self, position: Decimal);
    pub fn update_pnl(&self, realized: Decimal, unrealized: Decimal);
    
    // Snapshot
    pub fn snapshot(&self, current_time: u64) -> MetricsSnapshot;
    
    // Reset (e.g., for new trading day)
    pub fn reset(&self, new_start_time: u64);
}

/// Thread-safe wrapper
pub type SharedLiveMetrics = Arc<LiveMetrics>;
```

## Acceptance Criteria

- [ ] `Counter` struct with atomic operations
- [ ] `Gauge` struct for bidirectional values
- [ ] `MetricsSnapshot` with all metric values
- [ ] `LiveMetrics` tracker with thread-safe implementation
- [ ] Counter methods for quotes, orders, fills, cancels
- [ ] Gauge methods for open orders
- [ ] Position and PnL tracking (with RwLock for Decimal)
- [ ] `snapshot()` returns current state
- [ ] Rate calculations (fill rate, quotes/second)
- [ ] Uptime tracking
- [ ] `reset()` for new trading sessions
- [ ] Unit tests covering:
  - Counter increment/reset
  - Gauge set/increment/decrement
  - Snapshot accuracy
  - Thread safety (concurrent access)
- [ ] Documentation with usage examples

## Technical Notes

- Use atomics for counters (no lock contention)
- Use `RwLock` for Decimal values (not atomic-compatible)
- `Ordering::Relaxed` is sufficient for counters
- Consider adding rate limiting for snapshot generation
- Rates calculated as: `count / (current_time - start_time) * 1000` for per-second

## Labels

`enhancement`, `monitoring`, `priority:medium`
