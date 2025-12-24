# Implement Backtesting Engine Core

## Summary

Create a backtesting engine to validate strategies on historical data before live deployment.

## Motivation

Backtesting is essential for:
- Validating strategy logic before risking capital
- Optimizing parameters
- Understanding strategy behavior in different market conditions
- Estimating expected performance metrics

## Detailed Description

Implement an event-driven backtesting engine that simulates strategy execution on historical data.

### Architecture

1. **Data Feed**: Provides historical market data (ticks, OHLCV, order book snapshots)
2. **Strategy**: Generates quotes based on market state
3. **Execution Simulator**: Simulates order fills
4. **Position Tracker**: Tracks inventory and PnL
5. **Metrics Collector**: Records performance data

### Proposed API

```rust
/// Market tick data
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub timestamp: u64,
    pub bid_price: Decimal,
    pub bid_size: Decimal,
    pub ask_price: Decimal,
    pub ask_size: Decimal,
    pub last_price: Option<Decimal>,
    pub last_size: Option<Decimal>,
}

/// OHLCV bar data
#[derive(Debug, Clone)]
pub struct OHLCVBar {
    pub timestamp: u64,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

/// Historical data source trait
pub trait HistoricalDataSource {
    fn next_tick(&mut self) -> Option<MarketTick>;
    fn peek_tick(&self) -> Option<&MarketTick>;
    fn reset(&mut self);
    fn len(&self) -> usize;
}

/// Vector-based data source
pub struct VecDataSource {
    ticks: Vec<MarketTick>,
    index: usize,
}

/// Strategy trait for backtesting
pub trait BacktestStrategy {
    /// Called on each market update, returns optional quotes
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote>;
    
    /// Called when an order is filled
    fn on_fill(&mut self, fill: &SimulatedFill);
    
    /// Reset strategy state
    fn reset(&mut self);
}

/// Simulated fill
#[derive(Debug, Clone)]
pub struct SimulatedFill {
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub timestamp: u64,
}

/// Backtest configuration
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: Decimal,
    
    /// Trading fees (as decimal, e.g., 0.001 for 0.1%)
    pub fee_rate: Decimal,
    
    /// Minimum tick size
    pub tick_size: Decimal,
    
    /// Minimum lot size
    pub lot_size: Decimal,
    
    /// Slippage model
    pub slippage: SlippageModel,
}

/// Slippage model
#[derive(Debug, Clone)]
pub enum SlippageModel {
    None,
    Fixed(Decimal),
    Percentage(Decimal),
    VolatilityBased { multiplier: Decimal },
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_pnl: Decimal,
    pub total_fees: Decimal,
    pub net_pnl: Decimal,
    pub num_trades: u64,
    pub num_ticks: u64,
    pub start_time: u64,
    pub end_time: u64,
    pub max_position: Decimal,
    pub final_position: Decimal,
    pub equity_curve: Vec<(u64, Decimal)>,
    pub trades: Vec<SimulatedFill>,
}

/// Backtest engine
pub struct BacktestEngine<S: BacktestStrategy, D: HistoricalDataSource> {
    config: BacktestConfig,
    strategy: S,
    data_source: D,
    position: InventoryPosition,
    pnl: PnL,
    equity_curve: Vec<(u64, Decimal)>,
    trades: Vec<SimulatedFill>,
}

impl<S: BacktestStrategy, D: HistoricalDataSource> BacktestEngine<S, D> {
    pub fn new(config: BacktestConfig, strategy: S, data_source: D) -> Self;
    
    /// Run the backtest
    pub fn run(&mut self) -> BacktestResult;
    
    /// Run with progress callback
    pub fn run_with_progress<F: FnMut(usize, usize)>(&mut self, callback: F) -> BacktestResult;
    
    /// Get current state (for debugging)
    pub fn get_state(&self) -> (&InventoryPosition, &PnL);
    
    /// Reset engine for another run
    pub fn reset(&mut self);
}
```

## Acceptance Criteria

- [ ] `MarketTick` and `OHLCVBar` data structures
- [ ] `HistoricalDataSource` trait for data abstraction
- [ ] `VecDataSource` implementation for in-memory data
- [ ] `BacktestStrategy` trait for strategy integration
- [ ] `SimulatedFill` struct for fill records
- [ ] `BacktestConfig` with fees, slippage, tick/lot sizes
- [ ] `SlippageModel` enum with multiple options
- [ ] `BacktestResult` with comprehensive metrics
- [ ] `BacktestEngine` implementing simulation loop
- [ ] Basic fill model (immediate fill at quote price)
- [ ] Fee calculation
- [ ] Equity curve tracking
- [ ] Position and PnL tracking using existing modules
- [ ] Unit tests covering:
  - Simple strategy backtest
  - Fee calculation
  - Slippage application
  - Position tracking accuracy
- [ ] Example with sample data and strategy
- [ ] Documentation with usage guide

## Technical Notes

- Event loop: iterate ticks, call strategy, simulate fills, update state
- Fill simulation: if market price crosses quote, fill occurs
- Consider adding support for order book replay in future
- Equity curve: record (timestamp, equity) at each tick or fill
- Use existing `InventoryPosition` and `PnL` from position module

## Labels

`enhancement`, `backtest`, `priority:high`
