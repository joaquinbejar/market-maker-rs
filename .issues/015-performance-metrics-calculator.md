# Implement Performance Metrics Calculator

## Summary

Calculate comprehensive performance metrics from backtest or live trading results.

## Motivation

Proper performance evaluation requires more than just total PnL. Risk-adjusted metrics help:
- Compare strategies fairly
- Identify risk/reward tradeoffs
- Detect strategy degradation
- Meet regulatory reporting requirements

## Detailed Description

Implement a `PerformanceMetrics` calculator that computes standard quantitative finance metrics.

### Key Metrics

1. **Return Metrics**: Total return, annualized return, CAGR
2. **Risk Metrics**: Volatility, max drawdown, VaR
3. **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
4. **Trading Metrics**: Win rate, profit factor, average trade

### Proposed API

```rust
/// Input data for metrics calculation
#[derive(Debug, Clone)]
pub struct EquityPoint {
    pub timestamp: u64,
    pub equity: Decimal,
}

/// Trade record for metrics
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub entry_time: u64,
    pub exit_time: u64,
    pub side: Side,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub quantity: Decimal,
    pub pnl: Decimal,
    pub fees: Decimal,
}

/// Performance metrics result
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    // Return metrics
    pub total_return: Decimal,
    pub total_return_pct: Decimal,
    pub annualized_return: Decimal,
    pub cagr: Decimal,
    
    // Risk metrics
    pub volatility: Decimal,           // Annualized
    pub downside_volatility: Decimal,  // Only negative returns
    pub max_drawdown: Decimal,
    pub max_drawdown_duration_ms: u64,
    pub var_95: Decimal,               // 95% Value at Risk
    pub var_99: Decimal,               // 99% Value at Risk
    
    // Risk-adjusted metrics
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub calmar_ratio: Decimal,
    pub information_ratio: Option<Decimal>,  // If benchmark provided
    
    // Trading metrics
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub win_rate: Decimal,
    pub profit_factor: Decimal,        // Gross profit / gross loss
    pub average_trade_pnl: Decimal,
    pub average_winner: Decimal,
    pub average_loser: Decimal,
    pub largest_winner: Decimal,
    pub largest_loser: Decimal,
    pub avg_trade_duration_ms: u64,
    
    // Market making specific
    pub average_spread_captured: Option<Decimal>,
    pub inventory_turnover: Option<Decimal>,
    pub time_in_market_pct: Option<Decimal>,
}

/// Configuration for metrics calculation
pub struct MetricsConfig {
    /// Risk-free rate for Sharpe calculation (annualized)
    pub risk_free_rate: Decimal,
    
    /// Trading days per year for annualization
    pub trading_days_per_year: u32,
    
    /// Benchmark returns (optional, for information ratio)
    pub benchmark_returns: Option<Vec<Decimal>>,
}

/// Performance metrics calculator
pub struct MetricsCalculator {
    config: MetricsConfig,
}

impl MetricsCalculator {
    pub fn new(config: MetricsConfig) -> Self;
    
    /// Calculate all metrics from equity curve and trades
    pub fn calculate(
        &self,
        equity_curve: &[EquityPoint],
        trades: &[TradeRecord],
        initial_capital: Decimal,
    ) -> MMResult<PerformanceMetrics>;
    
    /// Calculate returns from equity curve
    pub fn calculate_returns(&self, equity_curve: &[EquityPoint]) -> Vec<Decimal>;
    
    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(&self, returns: &[Decimal]) -> Decimal;
    
    /// Calculate Sortino ratio
    pub fn sortino_ratio(&self, returns: &[Decimal]) -> Decimal;
    
    /// Calculate maximum drawdown
    pub fn max_drawdown(&self, equity_curve: &[EquityPoint]) -> (Decimal, u64);
    
    /// Calculate Value at Risk
    pub fn var(&self, returns: &[Decimal], confidence: Decimal) -> Decimal;
    
    /// Calculate profit factor
    pub fn profit_factor(&self, trades: &[TradeRecord]) -> Decimal;
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            risk_free_rate: Decimal::ZERO,
            trading_days_per_year: 252,
            benchmark_returns: None,
        }
    }
}
```

## Acceptance Criteria

- [ ] `EquityPoint` and `TradeRecord` input types
- [ ] `PerformanceMetrics` struct with all metrics
- [ ] `MetricsConfig` with risk-free rate and annualization
- [ ] `MetricsCalculator` implementing all calculations
- [ ] Return metrics: total, annualized, CAGR
- [ ] Risk metrics: volatility, max drawdown, VaR
- [ ] Risk-adjusted: Sharpe, Sortino, Calmar ratios
- [ ] Trading metrics: win rate, profit factor, averages
- [ ] Proper annualization of all metrics
- [ ] Handle edge cases (no trades, single point, etc.)
- [ ] Unit tests with known data verifying calculations
- [ ] Documentation with metric definitions and formulas
- [ ] Example showing metrics from backtest results

## Technical Notes

- Sharpe: `(mean_return - rf) / std_dev * sqrt(252)`
- Sortino: `(mean_return - rf) / downside_std_dev * sqrt(252)`
- Calmar: `annualized_return / max_drawdown`
- Profit factor: `sum(winning_trades) / abs(sum(losing_trades))`
- VaR: Historical method - percentile of return distribution
- Use `Decimal` throughout for precision
- Consider adding rolling metrics (e.g., rolling Sharpe)

## Labels

`enhancement`, `backtest`, `analytics`, `priority:medium`
