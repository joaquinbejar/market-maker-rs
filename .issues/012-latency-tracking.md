# Implement Latency Tracking

## Summary

Track and report latency metrics for orders and market data to monitor system performance.

## Motivation

Latency is critical for market making performance. Tracking latency helps:
- Identify connectivity issues
- Measure execution quality
- Detect degradation before it impacts PnL
- Compare performance across venues

## Detailed Description

Implement a `LatencyTracker` that measures and reports various latency metrics.

### Key Metrics

1. **Order-to-Ack**: Time from order submission to exchange acknowledgment
2. **Order-to-Fill**: Time from submission to first fill
3. **Market Data Latency**: Delay in receiving market data updates
4. **Round-Trip Time**: Full cycle for order operations

### Proposed API

```rust
use std::time::Duration;

/// Latency measurement
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    pub value_us: u64,  // Microseconds
    pub timestamp: u64,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub count: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub std_dev_us: u64,
}

/// Latency metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatencyMetric {
    OrderToAck,
    OrderToFill,
    OrderToCancel,
    MarketDataDelay,
    RoundTrip,
}

/// Latency tracker configuration
pub struct LatencyTrackerConfig {
    /// Window size for rolling statistics
    pub window_size: usize,
    
    /// Whether to keep full histogram
    pub keep_histogram: bool,
    
    /// Histogram bucket size in microseconds
    pub histogram_bucket_us: u64,
}

/// Latency tracker
pub struct LatencyTracker {
    config: LatencyTrackerConfig,
    measurements: HashMap<LatencyMetric, VecDeque<LatencyMeasurement>>,
    histograms: Option<HashMap<LatencyMetric, Histogram>>,
}

impl LatencyTracker {
    pub fn new(config: LatencyTrackerConfig) -> Self;
    
    /// Record a latency measurement
    pub fn record(&mut self, metric: LatencyMetric, latency_us: u64, timestamp: u64);
    
    /// Record latency from Duration
    pub fn record_duration(&mut self, metric: LatencyMetric, duration: Duration, timestamp: u64);
    
    /// Get statistics for a metric
    pub fn get_stats(&self, metric: LatencyMetric) -> Option<LatencyStats>;
    
    /// Get all statistics
    pub fn get_all_stats(&self) -> HashMap<LatencyMetric, LatencyStats>;
    
    /// Get recent measurements
    pub fn get_recent(&self, metric: LatencyMetric, count: usize) -> Vec<&LatencyMeasurement>;
    
    /// Check if latency exceeds threshold
    pub fn is_degraded(&self, metric: LatencyMetric, threshold_us: u64) -> bool;
    
    /// Get histogram (if enabled)
    pub fn get_histogram(&self, metric: LatencyMetric) -> Option<&Histogram>;
    
    /// Reset all measurements
    pub fn reset(&mut self);
}

/// Simple histogram for latency distribution
pub struct Histogram {
    buckets: Vec<u64>,
    bucket_size_us: u64,
    total_count: u64,
}

impl Histogram {
    pub fn new(bucket_size_us: u64, num_buckets: usize) -> Self;
    pub fn record(&mut self, value_us: u64);
    pub fn percentile(&self, p: f64) -> u64;
    pub fn get_buckets(&self) -> &[u64];
}
```

## Acceptance Criteria

- [ ] `LatencyMeasurement` struct with microsecond precision
- [ ] `LatencyStats` with min, max, avg, percentiles
- [ ] `LatencyMetric` enum for different measurement types
- [ ] `LatencyTracker` with rolling window storage
- [ ] `record()` and `record_duration()` methods
- [ ] `get_stats()` calculates statistics from window
- [ ] Percentile calculation (p50, p95, p99)
- [ ] `is_degraded()` threshold check
- [ ] Optional histogram support
- [ ] Unit tests covering:
  - Recording measurements
  - Statistics calculation
  - Percentile accuracy
  - Window rotation
  - Histogram bucketing
- [ ] Documentation with usage examples

## Technical Notes

- Use microseconds (u64) for precision without floating point
- Rolling window with `VecDeque` for efficient rotation
- Percentile calculation: sort and index, or use histogram approximation
- Consider reservoir sampling for memory-bounded tracking
- Standard deviation: `sqrt(sum((x - mean)²) / n)`

## Labels

`enhancement`, `execution`, `monitoring`, `priority:low`
