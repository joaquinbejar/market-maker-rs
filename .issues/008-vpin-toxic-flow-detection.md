# Implement VPIN (Volume-Synchronized Probability of Informed Trading)

## Summary

Implement the VPIN metric to detect toxic order flow and informed trading activity.

## Motivation

VPIN (Easley, López de Prado, O'Hara, 2012) is a real-time metric that estimates the probability of informed trading. High VPIN values indicate toxic flow that can lead to adverse selection losses for market makers.

## Detailed Description

VPIN measures order flow toxicity by:

1. Grouping trades into volume buckets (not time buckets)
2. Classifying trades as buy or sell initiated
3. Calculating the absolute imbalance within each bucket
4. Averaging imbalance over a rolling window of buckets

### Algorithm

1. Define bucket size `V` (e.g., 1% of daily volume)
2. Fill buckets with trades until volume reaches `V`
3. For each bucket, calculate: `|V_buy - V_sell| / V`
4. VPIN = average of last N bucket imbalances

### Proposed API

```rust
pub struct VPINConfig {
    /// Volume per bucket (in base currency units)
    pub bucket_volume: Decimal,
    
    /// Number of buckets for rolling average
    pub num_buckets: usize,
    
    /// Toxicity alert threshold (e.g., 0.7)
    pub toxicity_threshold: Decimal,
}

pub struct VolumeBucket {
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub total_volume: Decimal,
    pub imbalance: Decimal,
    pub start_time: u64,
    pub end_time: u64,
    pub trade_count: u64,
}

pub struct VPINCalculator {
    config: VPINConfig,
    completed_buckets: VecDeque<VolumeBucket>,
    current_bucket: VolumeBucket,
}

impl VPINCalculator {
    pub fn new(config: VPINConfig) -> Self;
    
    /// Add a trade and update VPIN
    pub fn add_trade(&mut self, trade: &Trade) -> Option<Decimal>;
    
    /// Get current VPIN value
    pub fn get_vpin(&self) -> Option<Decimal>;
    
    /// Check if current VPIN exceeds toxicity threshold
    pub fn is_toxic(&self) -> bool;
    
    /// Get the last N completed buckets
    pub fn get_buckets(&self) -> &VecDeque<VolumeBucket>;
    
    /// Get current (incomplete) bucket
    pub fn get_current_bucket(&self) -> &VolumeBucket;
    
    /// Reset calculator
    pub fn reset(&mut self);
}
```

## Acceptance Criteria

- [ ] `VPINConfig` struct with bucket size and window parameters
- [ ] `VolumeBucket` struct tracking buy/sell volume per bucket
- [ ] `VPINCalculator` implementing the VPIN algorithm
- [ ] `add_trade()` correctly fills buckets and rotates when full
- [ ] `get_vpin()` returns rolling average of bucket imbalances
- [ ] `is_toxic()` threshold check for alerts
- [ ] Trade classification (buy vs sell initiated)
- [ ] Unit tests covering:
  - Bucket filling and rotation
  - VPIN calculation accuracy
  - Toxicity detection
  - Edge cases (insufficient buckets, empty calculator)
- [ ] Documentation with:
  - Algorithm explanation
  - Reference to original paper
  - Recommended parameter values
- [ ] Example demonstrating VPIN usage

## Technical Notes

- Paper: "Flow Toxicity and Liquidity in a High-Frequency World" (2012)
- Trade classification: Use tick rule (compare to previous price) or quote rule (compare to mid)
- Typical bucket size: 1/50 of average daily volume
- Typical window: 50 buckets
- VPIN range: [0, 1], higher = more toxic
- Consider adding bulk trade classification option

## Labels

`enhancement`, `analytics`, `priority:low`
