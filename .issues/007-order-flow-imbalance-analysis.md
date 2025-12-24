# Implement Order Flow Imbalance Analysis

## Summary

Add tools to analyze order flow and detect directional pressure in the market.

## Motivation

Order flow analysis helps market makers understand the balance between buying and selling pressure. This information can be used to adjust quotes, predict short-term price movements, and avoid adverse selection.

## Detailed Description

Implement an `OrderFlowAnalyzer` that tracks and analyzes trade flow in real-time.

### Key Metrics

1. **Buy/Sell Volume**: Aggregate volume by trade direction
2. **Order Flow Imbalance (OFI)**: Net buying vs selling pressure
3. **Volume-Weighted Average Price (VWAP)**: By side
4. **Trade Intensity**: Trades per unit time

### Proposed API

```rust
pub struct Trade {
    pub price: Decimal,
    pub size: Decimal,
    pub side: TradeSide,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

pub struct OrderFlowStats {
    /// Total buy volume in window
    pub buy_volume: Decimal,
    
    /// Total sell volume in window
    pub sell_volume: Decimal,
    
    /// Number of buy trades
    pub buy_count: u64,
    
    /// Number of sell trades
    pub sell_count: u64,
    
    /// Order flow imbalance: (buy - sell) / (buy + sell)
    pub imbalance: Decimal,
    
    /// Net flow: buy_volume - sell_volume
    pub net_flow: Decimal,
    
    /// Buy VWAP
    pub buy_vwap: Option<Decimal>,
    
    /// Sell VWAP
    pub sell_vwap: Option<Decimal>,
    
    /// Window start timestamp
    pub window_start: u64,
    
    /// Window end timestamp
    pub window_end: u64,
}

pub struct OrderFlowAnalyzer {
    /// Rolling window duration in milliseconds
    window_ms: u64,
    
    /// Trade buffer
    trades: Vec<Trade>,
}

impl OrderFlowAnalyzer {
    pub fn new(window_ms: u64) -> Self;
    
    /// Add a new trade to the analyzer
    pub fn add_trade(&mut self, trade: Trade);
    
    /// Get current order flow statistics
    pub fn get_stats(&self, current_time: u64) -> OrderFlowStats;
    
    /// Get imbalance value directly
    pub fn get_imbalance(&self, current_time: u64) -> Decimal;
    
    /// Check if flow is significantly bullish
    pub fn is_bullish(&self, threshold: Decimal, current_time: u64) -> bool;
    
    /// Check if flow is significantly bearish
    pub fn is_bearish(&self, threshold: Decimal, current_time: u64) -> bool;
    
    /// Clear old trades outside window
    pub fn cleanup(&mut self, current_time: u64);
    
    /// Get trade intensity (trades per second)
    pub fn trade_intensity(&self, current_time: u64) -> Decimal;
}
```

## Acceptance Criteria

- [ ] `Trade` struct with price, size, side, timestamp
- [ ] `TradeSide` enum for buy/sell classification
- [ ] `OrderFlowStats` struct with comprehensive metrics
- [ ] `OrderFlowAnalyzer` with rolling window implementation
- [ ] `add_trade()` efficiently adds trades to buffer
- [ ] `get_stats()` calculates all metrics for current window
- [ ] `get_imbalance()` returns normalized imbalance [-1, 1]
- [ ] `is_bullish()` / `is_bearish()` threshold checks
- [ ] `cleanup()` removes stale trades to manage memory
- [ ] VWAP calculation by side
- [ ] Unit tests covering:
  - Empty analyzer
  - Single trade
  - Balanced flow
  - Imbalanced flow
  - Window expiration
  - VWAP accuracy
- [ ] Documentation with usage examples

## Technical Notes

- Use `VecDeque` for efficient front removal during cleanup
- Imbalance formula: `(buy_vol - sell_vol) / (buy_vol + sell_vol)`
- Handle division by zero when no trades in window
- Consider adding exponential weighting for recent trades

## Labels

`enhancement`, `analytics`, `priority:medium`
