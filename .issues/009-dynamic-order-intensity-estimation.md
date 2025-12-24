# Implement Dynamic Order Intensity Estimation

## Summary

Dynamically estimate the order intensity parameter `k` from observed trade data for more accurate Avellaneda-Stoikov calculations.

## Motivation

The order intensity parameter `k` in the A-S model represents how frequently market orders arrive. Using a static value can lead to suboptimal quotes. Dynamic estimation from real trade data improves strategy performance.

## Detailed Description

Implement an `OrderIntensityEstimator` that estimates `k` from:

1. **Trade arrival rate**: Poisson process parameter λ
2. **Fill probability**: Historical fill rates at different spread levels
3. **Time-of-day patterns**: Intensity varies throughout trading session

### Mathematical Background

In the A-S model, order arrival follows a Poisson process with intensity:
```
λ(δ) = A * exp(-k * δ)
```

Where:
- `λ(δ)` = arrival rate at spread δ
- `A` = baseline arrival rate
- `k` = order intensity parameter (to be estimated)

### Proposed API

```rust
pub struct OrderIntensityConfig {
    /// Window for estimation in milliseconds
    pub estimation_window_ms: u64,
    
    /// Minimum trades required for valid estimate
    pub min_trades: usize,
    
    /// Smoothing factor for EWMA updates
    pub smoothing_factor: Decimal,
}

pub struct FillObservation {
    pub spread_at_fill: Decimal,
    pub time_to_fill_ms: u64,
    pub timestamp: u64,
}

pub struct IntensityEstimate {
    /// Estimated k parameter
    pub k: Decimal,
    
    /// Baseline arrival rate A
    pub baseline_rate: Decimal,
    
    /// Confidence/quality of estimate
    pub confidence: Decimal,
    
    /// Number of observations used
    pub sample_size: usize,
    
    /// Timestamp of estimate
    pub timestamp: u64,
}

pub struct OrderIntensityEstimator {
    config: OrderIntensityConfig,
    observations: Vec<FillObservation>,
    current_estimate: Option<IntensityEstimate>,
}

impl OrderIntensityEstimator {
    pub fn new(config: OrderIntensityConfig) -> Self;
    
    /// Record a fill observation
    pub fn record_fill(&mut self, observation: FillObservation);
    
    /// Estimate k from collected observations
    pub fn estimate(&mut self, current_time: u64) -> MMResult<IntensityEstimate>;
    
    /// Get current estimate (if available)
    pub fn get_estimate(&self) -> Option<&IntensityEstimate>;
    
    /// Get k value with fallback to default
    pub fn get_k_or_default(&self, default: Decimal) -> Decimal;
    
    /// Calculate expected fill probability at given spread
    pub fn fill_probability(&self, spread: Decimal, time_horizon_ms: u64) -> Option<Decimal>;
    
    /// Clear old observations
    pub fn cleanup(&mut self, current_time: u64);
}
```

## Acceptance Criteria

- [ ] `OrderIntensityConfig` struct with estimation parameters
- [ ] `FillObservation` struct for recording fills
- [ ] `IntensityEstimate` struct with k, baseline rate, confidence
- [ ] `OrderIntensityEstimator` implementing estimation logic
- [ ] `record_fill()` stores observations with spread and timing
- [ ] `estimate()` calculates k using regression or MLE
- [ ] `fill_probability()` predicts fill likelihood at spread
- [ ] EWMA smoothing for stable estimates
- [ ] Confidence metric based on sample size and variance
- [ ] Integration with `StrategyConfig`
- [ ] Unit tests covering:
  - Estimation with synthetic data
  - Insufficient data handling
  - EWMA smoothing behavior
  - Fill probability calculation
- [ ] Documentation with methodology explanation
- [ ] Example showing dynamic k updates

## Technical Notes

- Estimation method: Log-linear regression of fill rate vs spread
- Alternative: Maximum Likelihood Estimation assuming Poisson arrivals
- Minimum 20-30 observations recommended for stable estimate
- Consider time-of-day adjustments (separate estimates for different periods)
- Handle case where all fills are at same spread (undefined k)

## Labels

`enhancement`, `analytics`, `priority:medium`
