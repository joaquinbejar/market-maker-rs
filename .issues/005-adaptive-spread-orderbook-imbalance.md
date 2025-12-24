# Implement Adaptive Spread Based on Order Book Imbalance

## Summary

Dynamically adjust bid-ask spreads based on order book imbalance and trade flow to improve quote positioning.

## Motivation

Static spreads don't account for directional pressure in the market. By widening spreads on the side facing adverse flow and tightening on the other, we can reduce adverse selection and improve fill quality.

## Detailed Description

Implement an `AdaptiveSpreadCalculator` that adjusts spreads based on:

1. **Order Book Imbalance**: Ratio of bid vs ask depth
2. **Trade Flow Imbalance**: Recent buy vs sell volume
3. **Volatility Regime**: Current vs historical volatility

### Proposed API

```rust
pub struct OrderBookImbalance {
    /// Imbalance ratio: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    /// Range: -1.0 (all asks) to +1.0 (all bids)
    pub imbalance: Decimal,
    
    /// Total bid depth considered
    pub bid_depth: Decimal,
    
    /// Total ask depth considered
    pub ask_depth: Decimal,
    
    /// Number of levels analyzed
    pub levels: u32,
}

pub struct TradeFlowImbalance {
    /// Buy volume in window
    pub buy_volume: Decimal,
    
    /// Sell volume in window
    pub sell_volume: Decimal,
    
    /// Net flow: buy_volume - sell_volume
    pub net_flow: Decimal,
    
    /// Imbalance ratio
    pub imbalance: Decimal,
}

pub struct AdaptiveSpreadConfig {
    /// Base spread from underlying strategy
    pub base_spread: Decimal,
    
    /// Maximum spread adjustment factor (e.g., 2.0 = can double spread)
    pub max_adjustment: Decimal,
    
    /// Sensitivity to order book imbalance (0.0 to 1.0)
    pub orderbook_sensitivity: Decimal,
    
    /// Sensitivity to trade flow (0.0 to 1.0)
    pub tradeflow_sensitivity: Decimal,
}

pub struct AdaptiveSpreadCalculator {
    config: AdaptiveSpreadConfig,
}

pub struct AdaptiveSpread {
    pub bid_spread: Decimal,  // Distance from mid for bid
    pub ask_spread: Decimal,  // Distance from mid for ask
    pub total_spread: Decimal,
}

impl AdaptiveSpreadCalculator {
    pub fn new(config: AdaptiveSpreadConfig) -> Self;
    
    /// Calculate order book imbalance from depth data
    pub fn calculate_orderbook_imbalance(
        bid_depths: &[(Decimal, Decimal)],  // (price, size) pairs
        ask_depths: &[(Decimal, Decimal)],
        levels: u32,
    ) -> OrderBookImbalance;
    
    /// Calculate trade flow imbalance from recent trades
    pub fn calculate_tradeflow_imbalance(
        trades: &[Trade],
        window_ms: u64,
        current_time: u64,
    ) -> TradeFlowImbalance;
    
    /// Calculate adaptive spread based on imbalances
    pub fn calculate_spread(
        &self,
        orderbook_imbalance: &OrderBookImbalance,
        tradeflow_imbalance: Option<&TradeFlowImbalance>,
    ) -> AdaptiveSpread;
}
```

## Acceptance Criteria

- [ ] `OrderBookImbalance` struct with imbalance calculation
- [ ] `TradeFlowImbalance` struct for trade flow analysis
- [ ] `AdaptiveSpreadConfig` with sensitivity parameters
- [ ] `AdaptiveSpreadCalculator` implementing spread adjustment logic
- [ ] `calculate_orderbook_imbalance()` from depth data
- [ ] `calculate_tradeflow_imbalance()` from trade history
- [ ] `calculate_spread()` returns asymmetric bid/ask spreads
- [ ] Spread widens on side facing adverse flow
- [ ] Integration with `orderbook-rs` crate
- [ ] Unit tests covering:
  - Balanced order book (symmetric spread)
  - Bid-heavy imbalance (tighter bid, wider ask)
  - Ask-heavy imbalance (wider bid, tighter ask)
  - Combined orderbook + tradeflow adjustment
- [ ] Example with mock orderbook data
- [ ] Documentation explaining the adjustment logic

## Technical Notes

- Imbalance formula: `(bid - ask) / (bid + ask)` gives range [-1, 1]
- Positive imbalance = more bids = expect price to rise = widen ask spread
- Consider volume-weighted depth for more accurate imbalance
- Integration with existing Avellaneda-Stoikov as spread modifier

## Labels

`enhancement`, `strategy`, `priority:medium`
