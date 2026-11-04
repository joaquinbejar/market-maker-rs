//! Performance metrics calculator for backtesting and live trading.
//!
//! This module provides comprehensive performance metrics calculation including
//! return metrics, risk metrics, risk-adjusted metrics, and trading statistics.
//!
//! # Overview
//!
//! The metrics module includes:
//!
//! - **Return metrics**: Total return, annualized return, CAGR
//! - **Risk metrics**: Volatility, max drawdown, VaR
//! - **Risk-adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
//! - **Trading metrics**: Win rate, profit factor, average trade
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{
//!     EquityPoint, TradeRecord, MetricsCalculator, MetricsConfig
//! };
//! use market_maker_rs::execution::Action;
//! use market_maker_rs::dec;
//!
//! // Create equity curve
//! let equity_curve = vec![
//!     EquityPoint::new(0, dec!(10000.0)),
//!     EquityPoint::new(86400000, dec!(10100.0)),
//!     EquityPoint::new(172800000, dec!(10250.0)),
//! ];
//!
//! // Create trade records
//! let trades = vec![
//!     TradeRecord::new(
//!         0, 86400000, Action::Buy,
//!         dec!(100.0), dec!(101.0), dec!(10.0),
//!         dec!(100.0), dec!(1.0)
//!     ),
//! ];
//!
//! let calculator = MetricsCalculator::new(MetricsConfig::default());
//! let metrics = calculator.calculate(&equity_curve, &trades, dec!(10000.0)).unwrap();
//!
//! println!("Total return: {}%", metrics.total_return_pct);
//! println!("Sharpe ratio: {}", metrics.sharpe_ratio);
//! ```

use crate::Decimal;
use crate::execution::Action;
use crate::types::decimal::{saturating_div, saturating_u64};
use crate::types::error::{MMError, MMResult};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A point on the equity curve.
///
/// Represents the portfolio equity at a specific point in time.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::EquityPoint;
/// use market_maker_rs::dec;
///
/// let point = EquityPoint::new(1000, dec!(10000.0));
/// assert_eq!(point.timestamp, 1000);
/// assert_eq!(point.equity, dec!(10000.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EquityPoint {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Portfolio equity value.
    pub equity: Decimal,
}

impl EquityPoint {
    /// Creates a new equity point.
    #[must_use]
    pub fn new(timestamp: u64, equity: Decimal) -> Self {
        Self { timestamp, equity }
    }
}

/// A completed trade record for metrics calculation.
///
/// Contains all information about a round-trip trade including
/// entry/exit prices, PnL, and fees.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::TradeRecord;
/// use market_maker_rs::execution::Action;
/// use market_maker_rs::dec;
///
/// let trade = TradeRecord::new(
///     1000, 2000, Action::Buy,
///     dec!(100.0), dec!(105.0), dec!(10.0),
///     dec!(50.0), dec!(1.0)
/// );
///
/// assert!(trade.is_winner());
/// assert_eq!(trade.net_pnl(), dec!(49.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TradeRecord {
    /// Entry timestamp in milliseconds.
    pub entry_time: u64,
    /// Exit timestamp in milliseconds.
    pub exit_time: u64,
    /// Trade side.
    pub side: Action,
    /// Entry price.
    pub entry_price: Decimal,
    /// Exit price.
    pub exit_price: Decimal,
    /// Trade quantity.
    pub quantity: Decimal,
    /// Gross PnL (before fees).
    pub pnl: Decimal,
    /// Total fees paid.
    pub fees: Decimal,
}

impl TradeRecord {
    /// Creates a new trade record.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        entry_time: u64,
        exit_time: u64,
        side: Action,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        pnl: Decimal,
        fees: Decimal,
    ) -> Self {
        Self {
            entry_time,
            exit_time,
            side,
            entry_price,
            exit_price,
            quantity,
            pnl,
            fees,
        }
    }

    /// Returns the net PnL after fees.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn net_pnl(&self) -> Decimal {
        self.pnl.saturating_sub(self.fees)
    }

    /// Returns true if this trade was profitable.
    #[must_use]
    pub fn is_winner(&self) -> bool {
        self.net_pnl() > Decimal::ZERO
    }

    /// Returns true if this trade was a loss.
    #[must_use]
    pub fn is_loser(&self) -> bool {
        self.net_pnl() < Decimal::ZERO
    }

    /// Returns the trade duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.exit_time.saturating_sub(self.entry_time)
    }

    /// Returns the return percentage for this trade.
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn return_pct(&self) -> Decimal {
        if self.entry_price == Decimal::ZERO {
            return Decimal::ZERO;
        }
        let notional = self.entry_price.saturating_mul(self.quantity);
        if notional == Decimal::ZERO {
            return Decimal::ZERO;
        }
        saturating_div(self.net_pnl(), notional).saturating_mul(Decimal::ONE_HUNDRED)
    }
}

/// Configuration for metrics calculation.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::MetricsConfig;
/// use market_maker_rs::dec;
///
/// let config = MetricsConfig::default()
///     .with_risk_free_rate(dec!(0.05))
///     .with_trading_days(252);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricsConfig {
    /// Risk-free rate for Sharpe calculation (annualized, as decimal e.g. 0.05 for 5%).
    pub risk_free_rate: Decimal,
    /// Trading days per year for annualization.
    pub trading_days_per_year: u32,
    /// Benchmark returns for information ratio calculation (optional).
    pub benchmark_returns: Option<Vec<Decimal>>,
}

impl MetricsConfig {
    /// Creates a new metrics configuration.
    #[must_use]
    pub fn new(risk_free_rate: Decimal, trading_days_per_year: u32) -> Self {
        Self {
            risk_free_rate,
            trading_days_per_year,
            benchmark_returns: None,
        }
    }

    /// Sets the risk-free rate.
    #[must_use]
    pub fn with_risk_free_rate(mut self, rate: Decimal) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Sets the trading days per year.
    #[must_use]
    pub fn with_trading_days(mut self, days: u32) -> Self {
        self.trading_days_per_year = days;
        self
    }

    /// Sets the benchmark returns.
    #[must_use]
    pub fn with_benchmark(mut self, returns: Vec<Decimal>) -> Self {
        self.benchmark_returns = Some(returns);
        self
    }
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

/// Comprehensive performance metrics.
///
/// Contains all calculated performance metrics from an equity curve
/// and trade history.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    // Return metrics
    /// Total return in absolute terms.
    pub total_return: Decimal,
    /// Total return as percentage.
    pub total_return_pct: Decimal,
    /// Annualized return.
    pub annualized_return: Decimal,
    /// Compound annual growth rate.
    pub cagr: Decimal,

    // Risk metrics
    /// Annualized volatility (standard deviation of returns).
    pub volatility: Decimal,
    /// Downside volatility (only negative returns).
    pub downside_volatility: Decimal,
    /// Maximum drawdown as decimal (e.g., 0.15 for 15%).
    pub max_drawdown: Decimal,
    /// Duration of maximum drawdown in milliseconds.
    pub max_drawdown_duration_ms: u64,
    /// 95% Value at Risk.
    pub var_95: Decimal,
    /// 99% Value at Risk.
    pub var_99: Decimal,

    // Risk-adjusted metrics
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: Decimal,
    /// Sortino ratio (annualized).
    pub sortino_ratio: Decimal,
    /// Calmar ratio (annualized return / max drawdown).
    pub calmar_ratio: Decimal,
    /// Information ratio (if benchmark provided).
    pub information_ratio: Option<Decimal>,

    // Trading metrics
    /// Total number of trades.
    pub total_trades: u64,
    /// Number of winning trades.
    pub winning_trades: u64,
    /// Number of losing trades.
    pub losing_trades: u64,
    /// Win rate as decimal (e.g., 0.55 for 55%).
    pub win_rate: Decimal,
    /// Profit factor (gross profit / gross loss).
    pub profit_factor: Decimal,
    /// Average PnL per trade.
    pub average_trade_pnl: Decimal,
    /// Average PnL of winning trades.
    pub average_winner: Decimal,
    /// Average PnL of losing trades (negative value).
    pub average_loser: Decimal,
    /// Largest winning trade.
    pub largest_winner: Decimal,
    /// Largest losing trade (negative value).
    pub largest_loser: Decimal,
    /// Average trade duration in milliseconds.
    pub avg_trade_duration_ms: u64,

    // Market making specific (optional)
    /// Average spread captured per trade.
    pub average_spread_captured: Option<Decimal>,
    /// Inventory turnover ratio.
    pub inventory_turnover: Option<Decimal>,
    /// Percentage of time with active positions.
    pub time_in_market_pct: Option<Decimal>,
}

impl PerformanceMetrics {
    /// Returns true if the strategy was profitable.
    #[must_use]
    pub fn is_profitable(&self) -> bool {
        self.total_return > Decimal::ZERO
    }

    /// Returns the risk-reward ratio (average winner / average loser).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn risk_reward_ratio(&self) -> Decimal {
        if self.average_loser == Decimal::ZERO {
            return Decimal::ZERO;
        }
        saturating_div(self.average_winner, self.average_loser.abs())
    }

    /// Returns the expectancy per trade.
    /// Formula: (win_rate * avg_winner) - ((1 - win_rate) * abs(avg_loser))
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn expectancy(&self) -> Decimal {
        let win_component = self.win_rate.saturating_mul(self.average_winner);
        let loss_component = Decimal::ONE
            .saturating_sub(self.win_rate)
            .saturating_mul(self.average_loser.abs());
        win_component.saturating_sub(loss_component)
    }
}

/// Performance metrics calculator.
///
/// Calculates comprehensive performance metrics from equity curves
/// and trade records.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{MetricsCalculator, MetricsConfig, EquityPoint};
/// use market_maker_rs::dec;
///
/// let calculator = MetricsCalculator::new(MetricsConfig::default());
///
/// let equity = vec![
///     EquityPoint::new(0, dec!(10000.0)),
///     EquityPoint::new(1000, dec!(10100.0)),
/// ];
///
/// let returns = calculator.calculate_returns(&equity);
/// assert_eq!(returns.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct MetricsCalculator {
    config: MetricsConfig,
}

impl MetricsCalculator {
    /// Creates a new metrics calculator.
    #[must_use]
    pub fn new(config: MetricsConfig) -> Self {
        Self { config }
    }

    /// Creates a calculator with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(MetricsConfig::default())
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &MetricsConfig {
        &self.config
    }

    /// Calculates all performance metrics from equity curve and trades.
    ///
    /// # Arguments
    ///
    /// * `equity_curve` - Time series of equity values
    /// * `trades` - List of completed trades
    /// * `initial_capital` - Starting capital
    ///
    /// Every metric is report only: a sum, product or ratio
    /// beyond the `Decimal` bounds saturates instead of failing.
    ///
    /// # Errors
    ///
    /// Returns an error if the equity curve is empty.
    pub fn calculate(
        &self,
        equity_curve: &[EquityPoint],
        trades: &[TradeRecord],
        initial_capital: Decimal,
    ) -> MMResult<PerformanceMetrics> {
        if equity_curve.is_empty() {
            return Err(MMError::InvalidConfiguration(
                "Equity curve cannot be empty".to_string(),
            ));
        }

        let returns = self.calculate_returns(equity_curve);
        let (max_dd, max_dd_duration) = self.max_drawdown(equity_curve);

        // Calculate return metrics
        let final_equity = equity_curve
            .last()
            .map(|p| p.equity)
            .unwrap_or(initial_capital);
        let total_return = final_equity.saturating_sub(initial_capital);
        let total_return_pct = if initial_capital != Decimal::ZERO {
            saturating_div(total_return, initial_capital).saturating_mul(Decimal::ONE_HUNDRED)
        } else {
            Decimal::ZERO
        };

        // Calculate time span for annualization
        let time_span_ms = match (equity_curve.first(), equity_curve.last()) {
            (Some(first), Some(last)) => last.timestamp.saturating_sub(first.timestamp),
            _ => 0,
        };
        let years = self.ms_to_years(time_span_ms);

        let annualized_return = self.annualize_return(
            saturating_div(total_return_pct, Decimal::ONE_HUNDRED),
            years,
        );
        let cagr = self.calculate_cagr(initial_capital, final_equity, years);

        // Calculate risk metrics
        let volatility = self.calculate_volatility(&returns);
        let downside_volatility = self.calculate_downside_volatility(&returns);
        let var_95 = self.var(&returns, dec!(0.95));
        let var_99 = self.var(&returns, dec!(0.99));

        // Calculate risk-adjusted metrics
        let sharpe = self.sharpe_ratio(&returns);
        let sortino = self.sortino_ratio(&returns);
        let calmar = if max_dd != Decimal::ZERO {
            saturating_div(annualized_return, max_dd)
        } else {
            Decimal::ZERO
        };

        // Calculate information ratio if benchmark provided
        let information_ratio = self.calculate_information_ratio(&returns);

        // Calculate trading metrics
        let trading_metrics = self.calculate_trading_metrics(trades);

        Ok(PerformanceMetrics {
            total_return,
            total_return_pct,
            annualized_return,
            cagr,
            volatility,
            downside_volatility,
            max_drawdown: max_dd,
            max_drawdown_duration_ms: max_dd_duration,
            var_95,
            var_99,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            information_ratio,
            ..trading_metrics
        })
    }

    /// Calculates period returns from equity curve.
    ///
    /// Returns are calculated as: (equity\[i\] - equity\[i-1\]) / equity\[i-1\],
    /// saturating at the `Decimal` bounds.
    #[must_use]
    pub fn calculate_returns(&self, equity_curve: &[EquityPoint]) -> Vec<Decimal> {
        if equity_curve.len() < 2 {
            return Vec::new();
        }

        equity_curve
            .windows(2)
            .filter_map(|w| match w {
                [prev, next] if prev.equity != Decimal::ZERO => Some(saturating_div(
                    next.equity.saturating_sub(prev.equity),
                    prev.equity,
                )),
                _ => None,
            })
            .collect()
    }

    /// Calculates the Sharpe ratio.
    ///
    /// Formula: (mean_return - rf) / std_dev * sqrt(trading_days)
    #[must_use]
    pub fn sharpe_ratio(&self, returns: &[Decimal]) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let mean = self.mean(returns);
        let std_dev = self.std_dev(returns);

        if std_dev == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let excess_return = mean.saturating_sub(self.daily_risk_free_rate());
        let annualization_factor = decimal_sqrt(Decimal::from(self.config.trading_days_per_year));

        saturating_div(excess_return, std_dev).saturating_mul(annualization_factor)
    }

    /// Calculates the Sortino ratio.
    ///
    /// Formula: (mean_return - rf) / downside_std_dev * sqrt(trading_days)
    #[must_use]
    pub fn sortino_ratio(&self, returns: &[Decimal]) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let mean = self.mean(returns);
        let downside_dev = self.downside_deviation(returns);

        if downside_dev == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let excess_return = mean.saturating_sub(self.daily_risk_free_rate());
        let annualization_factor = decimal_sqrt(Decimal::from(self.config.trading_days_per_year));

        saturating_div(excess_return, downside_dev).saturating_mul(annualization_factor)
    }

    /// Calculates maximum drawdown and its duration.
    ///
    /// Returns (max_drawdown, duration_ms).
    #[must_use]
    pub fn max_drawdown(&self, equity_curve: &[EquityPoint]) -> (Decimal, u64) {
        let Some(first) = equity_curve.first() else {
            return (Decimal::ZERO, 0);
        };

        let mut peak = first.equity;
        let mut max_dd = Decimal::ZERO;
        let mut max_dd_duration: u64 = 0;
        let mut current_dd_start = first.timestamp;

        for point in equity_curve {
            if point.equity > peak {
                peak = point.equity;
                current_dd_start = point.timestamp;
            }

            if peak > Decimal::ZERO {
                let dd = saturating_div(peak.saturating_sub(point.equity), peak);
                if dd > max_dd {
                    max_dd = dd;
                    max_dd_duration = point.timestamp.saturating_sub(current_dd_start);
                }
            }
        }

        (max_dd, max_dd_duration)
    }

    /// Calculates Value at Risk using historical method.
    ///
    /// # Arguments
    ///
    /// * `returns` - Return series
    /// * `confidence` - Confidence level (e.g., 0.95 for 95%)
    #[must_use]
    pub fn var(&self, returns: &[Decimal], confidence: Decimal) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let mut sorted: Vec<Decimal> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // VaR is the loss at the (1 - confidence) percentile
        let percentile = Decimal::ONE.saturating_sub(confidence);
        let last = sorted.len().saturating_sub(1);
        // Truncates toward zero and clamps to the series, like a
        // saturating float cast would.
        let raw = percentile.saturating_mul(Decimal::from(last)).trunc();
        let fallback = if raw.is_sign_negative() { 0 } else { last };
        let index = raw.to_usize().unwrap_or(fallback).min(last);

        // Return as positive number (loss)
        // Negating a Decimal flips its sign bit: it cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let loss = sorted.get(index).map_or(Decimal::ZERO, |worst| -*worst);
        loss
    }

    /// Calculates profit factor.
    ///
    /// Formula: sum(winning_trades) / abs(sum(losing_trades))
    #[must_use]
    pub fn profit_factor(&self, trades: &[TradeRecord]) -> Decimal {
        let gross_profit = saturating_sum(
            trades
                .iter()
                .filter(|t| t.net_pnl() > Decimal::ZERO)
                .map(|t| t.net_pnl()),
        );

        let gross_loss = saturating_sum(
            trades
                .iter()
                .filter(|t| t.net_pnl() < Decimal::ZERO)
                .map(|t| t.net_pnl().abs()),
        );

        if gross_loss == Decimal::ZERO {
            if gross_profit > Decimal::ZERO {
                return Decimal::from(999); // Infinite profit factor capped
            }
            return Decimal::ZERO;
        }

        saturating_div(gross_profit, gross_loss)
    }

    // Helper methods

    /// The risk free rate per trading day.
    fn daily_risk_free_rate(&self) -> Decimal {
        saturating_div(
            self.config.risk_free_rate,
            Decimal::from(self.config.trading_days_per_year),
        )
    }

    fn calculate_volatility(&self, returns: &[Decimal]) -> Decimal {
        let std_dev = self.std_dev(returns);
        let annualization = decimal_sqrt(Decimal::from(self.config.trading_days_per_year));
        std_dev.saturating_mul(annualization)
    }

    fn calculate_downside_volatility(&self, returns: &[Decimal]) -> Decimal {
        let downside_dev = self.downside_deviation(returns);
        let annualization = decimal_sqrt(Decimal::from(self.config.trading_days_per_year));
        downside_dev.saturating_mul(annualization)
    }

    fn calculate_cagr(&self, initial: Decimal, final_val: Decimal, years: Decimal) -> Decimal {
        if initial <= Decimal::ZERO || years <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let ratio = saturating_div(final_val, initial);
        if ratio <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // CAGR = (final/initial)^(1/years) - 1
        // Approximation using ln: exp(ln(ratio) / years) - 1
        let ln_ratio = decimal_ln(ratio);
        let exponent = saturating_div(ln_ratio, years);
        decimal_exp(exponent).saturating_sub(Decimal::ONE)
    }

    fn annualize_return(&self, total_return: Decimal, years: Decimal) -> Decimal {
        if years <= Decimal::ZERO {
            return Decimal::ZERO;
        }
        // Simple annualization: total_return / years
        // For compound: (1 + total_return)^(1/years) - 1
        saturating_div(total_return, years)
    }

    fn ms_to_years(&self, ms: u64) -> Decimal {
        let ms_per_day = Decimal::from(86_400_000u64);
        let days = saturating_div(Decimal::from(ms), ms_per_day);
        saturating_div(days, Decimal::from(self.config.trading_days_per_year))
    }

    fn calculate_information_ratio(&self, returns: &[Decimal]) -> Option<Decimal> {
        let benchmark = self.config.benchmark_returns.as_ref()?;

        if returns.len() != benchmark.len() || returns.is_empty() {
            return None;
        }

        // Calculate excess returns
        let excess: Vec<Decimal> = returns
            .iter()
            .zip(benchmark.iter())
            .map(|(r, b)| r.saturating_sub(*b))
            .collect();

        let mean_excess = self.mean(&excess);
        let tracking_error = self.std_dev(&excess);

        if tracking_error == Decimal::ZERO {
            return None;
        }

        let annualization = decimal_sqrt(Decimal::from(self.config.trading_days_per_year));
        Some(saturating_div(mean_excess, tracking_error).saturating_mul(annualization))
    }

    fn calculate_trading_metrics(&self, trades: &[TradeRecord]) -> PerformanceMetrics {
        let total_trades = saturating_u64(trades.len());

        if total_trades == 0 {
            return PerformanceMetrics::default();
        }

        let winners: Vec<&TradeRecord> = trades.iter().filter(|t| t.is_winner()).collect();
        let losers: Vec<&TradeRecord> = trades.iter().filter(|t| t.is_loser()).collect();

        let winning_trades = saturating_u64(winners.len());
        let losing_trades = saturating_u64(losers.len());

        let win_rate = saturating_div(Decimal::from(winning_trades), Decimal::from(total_trades));

        let total_pnl = saturating_sum(trades.iter().map(TradeRecord::net_pnl));
        let average_trade_pnl = saturating_div(total_pnl, Decimal::from(total_trades));

        // `saturating_div` is zero for an empty side.
        let average_of = |side: &[&TradeRecord]| {
            saturating_div(
                saturating_sum(side.iter().map(|t| t.net_pnl())),
                Decimal::from(side.len()),
            )
        };
        let average_winner = average_of(&winners);
        let average_loser = average_of(&losers);

        let largest_winner = winners
            .iter()
            .map(|t| t.net_pnl())
            .max()
            .unwrap_or(Decimal::ZERO);

        let largest_loser = losers
            .iter()
            .map(|t| t.net_pnl())
            .min()
            .unwrap_or(Decimal::ZERO);

        let total_duration = trades
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.duration_ms()));
        // `total_trades` is nonzero here.
        let avg_trade_duration_ms = total_duration.checked_div(total_trades).unwrap_or(0);

        let profit_factor = self.profit_factor(trades);

        PerformanceMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            average_trade_pnl,
            average_winner,
            average_loser,
            largest_winner,
            largest_loser,
            avg_trade_duration_ms,
            ..Default::default()
        }
    }

    fn mean(&self, values: &[Decimal]) -> Decimal {
        // Zero for an empty slice.
        saturating_div(
            saturating_sum(values.iter().copied()),
            Decimal::from(values.len()),
        )
    }

    /// The sample standard deviation, zero for fewer than 2 values.
    fn sample_std_dev(&self, values: &[Decimal]) -> Decimal {
        if values.len() < 2 {
            return Decimal::ZERO;
        }

        let mean = self.mean(values);
        let squares = saturating_sum(values.iter().map(|v| {
            let diff = v.saturating_sub(mean);
            diff.saturating_mul(diff)
        }));
        let variance = saturating_div(squares, Decimal::from(values.len().saturating_sub(1)));

        decimal_sqrt(variance)
    }

    fn std_dev(&self, values: &[Decimal]) -> Decimal {
        self.sample_std_dev(values)
    }

    fn downside_deviation(&self, returns: &[Decimal]) -> Decimal {
        let negative_returns: Vec<Decimal> = returns
            .iter()
            .filter(|r| **r < Decimal::ZERO)
            .copied()
            .collect();

        self.sample_std_dev(&negative_returns)
    }
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// A sum of report values, saturating at the `Decimal` bounds.
fn saturating_sum(values: impl Iterator<Item = Decimal>) -> Decimal {
    values.fold(Decimal::ZERO, Decimal::saturating_add)
}

/// Approximate square root using Newton's method, saturating at the
/// `Decimal` bounds (report only).
fn decimal_sqrt(n: Decimal) -> Decimal {
    if n <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut x = n;
    let two = Decimal::TWO;

    for _ in 0..20 {
        let next = saturating_div(x.saturating_add(saturating_div(n, x)), two);
        if next.saturating_sub(x).abs() < dec!(0.0000001) {
            return next;
        }
        x = next;
    }

    x
}

/// Approximate natural logarithm.
// For `n > 0`, `x = (n - 1) / (n + 1)` is in (-1, 1) (the sum saturates
// only at `n = Decimal::MAX`, where x is 1 to 28 digits), so every term and
// the partial sums stay below 40 in magnitude.
#[allow(clippy::arithmetic_side_effects)]
fn decimal_ln(n: Decimal) -> Decimal {
    if n <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    // Use the identity: ln(x) ≈ 2 * sum((x-1)/(x+1))^(2k+1) / (2k+1)
    let x = n.saturating_sub(Decimal::ONE) / n.saturating_add(Decimal::ONE);
    let x2 = x * x;

    let mut result = Decimal::ZERO;
    let mut term = x;

    for k in 0..20 {
        let divisor = Decimal::from(2 * k + 1);
        result += term / divisor;
        term *= x2;
    }

    result * Decimal::TWO
}

/// Approximate exponential function, saturating at the `Decimal` bounds
/// for a large argument (report only).
///
/// A negative argument is computed as `1 / exp(-x)`, so the result stays in
/// (0, 1] and underflows toward zero instead of following an alternating
/// series between the bounds.
fn decimal_exp(x: Decimal) -> Decimal {
    if x.is_sign_negative() {
        // `exp(-x) >= 1`, so the quotient is in (0, 1].
        return saturating_div(Decimal::ONE, decimal_exp(x.abs()));
    }
    // Taylor series: e^x = sum(x^n / n!)
    let mut result = Decimal::ONE;
    let mut term = Decimal::ONE;

    for n in 1..30 {
        term = term.saturating_mul(saturating_div(x, Decimal::from(n)));
        result = result.saturating_add(term);
        if term.abs() < dec!(0.0000001) {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn create_equity_curve(values: &[(u64, Decimal)]) -> Vec<EquityPoint> {
        values
            .iter()
            .map(|(ts, eq)| EquityPoint::new(*ts, *eq))
            .collect()
    }

    // EquityPoint tests
    #[test]
    fn test_calculate_extreme_values_saturate_instead_of_panicking() {
        let tiny = dec!(0.0000000000000000000000000001);
        let curve =
            create_equity_curve(&[(0, tiny), (1, Decimal::MAX), (2, tiny), (3, Decimal::MAX)]);
        let trade = TradeRecord::new(
            0,
            1,
            Action::Buy,
            tiny,
            Decimal::MAX,
            Decimal::MAX,
            Decimal::MAX,
            Decimal::MIN,
        );
        let metrics = MetricsCalculator::with_defaults()
            .calculate(&curve, &[trade.clone(), trade], tiny)
            .unwrap();
        assert_eq!(metrics.total_return, Decimal::MAX);
        assert_eq!(metrics.total_return_pct, Decimal::MAX);
    }

    #[test]
    fn test_short_losing_backtest_cagr_stays_above_minus_one() {
        // A 50% loss over one hour annualizes toward -100%, never below.
        let curve = create_equity_curve(&[(0, dec!(100)), (3_600_000, dec!(50))]);
        let metrics = MetricsCalculator::with_defaults()
            .calculate(&curve, &[], dec!(100))
            .unwrap();
        assert!(metrics.cagr >= Decimal::NEGATIVE_ONE, "{}", metrics.cagr);
        assert!(metrics.cagr < Decimal::ZERO, "{}", metrics.cagr);
        let small = decimal_exp(dec!(-1));
        assert!((small - dec!(0.3678794)).abs() < dec!(0.000001), "{small}");
        assert!(decimal_exp(dec!(-1000)) >= Decimal::ZERO);
    }

    #[test]
    fn test_equity_point_new() {
        let point = EquityPoint::new(1000, dec!(10000.0));
        assert_eq!(point.timestamp, 1000);
        assert_eq!(point.equity, dec!(10000.0));
    }

    // TradeRecord tests
    #[test]
    fn test_trade_record_new() {
        let trade = TradeRecord::new(
            1000,
            2000,
            Action::Buy,
            dec!(100.0),
            dec!(105.0),
            dec!(10.0),
            dec!(50.0),
            dec!(1.0),
        );

        assert_eq!(trade.entry_time, 1000);
        assert_eq!(trade.exit_time, 2000);
        assert_eq!(trade.pnl, dec!(50.0));
        assert_eq!(trade.fees, dec!(1.0));
    }

    #[test]
    fn test_trade_record_net_pnl() {
        let trade = TradeRecord::new(
            0,
            0,
            Action::Buy,
            dec!(100.0),
            dec!(105.0),
            dec!(10.0),
            dec!(50.0),
            dec!(2.0),
        );
        assert_eq!(trade.net_pnl(), dec!(48.0));
    }

    #[test]
    fn test_trade_record_is_winner() {
        let winner = TradeRecord::new(
            0,
            0,
            Action::Buy,
            dec!(100.0),
            dec!(105.0),
            dec!(10.0),
            dec!(50.0),
            dec!(1.0),
        );
        let loser = TradeRecord::new(
            0,
            0,
            Action::Buy,
            dec!(100.0),
            dec!(95.0),
            dec!(10.0),
            dec!(-50.0),
            dec!(1.0),
        );

        assert!(winner.is_winner());
        assert!(!winner.is_loser());
        assert!(!loser.is_winner());
        assert!(loser.is_loser());
    }

    #[test]
    fn test_trade_record_duration() {
        let trade = TradeRecord::new(
            1000,
            5000,
            Action::Buy,
            dec!(100.0),
            dec!(105.0),
            dec!(10.0),
            dec!(50.0),
            dec!(1.0),
        );
        assert_eq!(trade.duration_ms(), 4000);
    }

    // MetricsConfig tests
    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert_eq!(config.risk_free_rate, Decimal::ZERO);
        assert_eq!(config.trading_days_per_year, 252);
        assert!(config.benchmark_returns.is_none());
    }

    #[test]
    fn test_metrics_config_builder() {
        let config = MetricsConfig::default()
            .with_risk_free_rate(dec!(0.05))
            .with_trading_days(365)
            .with_benchmark(vec![dec!(0.01), dec!(0.02)]);

        assert_eq!(config.risk_free_rate, dec!(0.05));
        assert_eq!(config.trading_days_per_year, 365);
        assert!(config.benchmark_returns.is_some());
    }

    // MetricsCalculator tests
    #[test]
    fn test_calculate_returns() {
        let calculator = MetricsCalculator::with_defaults();
        let equity = create_equity_curve(&[
            (0, dec!(10000.0)),
            (1000, dec!(10100.0)),
            (2000, dec!(10201.0)),
        ]);

        let returns = calculator.calculate_returns(&equity);

        assert_eq!(returns.len(), 2);
        assert_eq!(returns[0], dec!(0.01)); // 1% return
        assert_eq!(returns[1], dec!(0.01)); // 1% return
    }

    #[test]
    fn test_calculate_returns_empty() {
        let calculator = MetricsCalculator::with_defaults();
        let returns = calculator.calculate_returns(&[]);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_calculate_returns_single_point() {
        let calculator = MetricsCalculator::with_defaults();
        let equity = create_equity_curve(&[(0, dec!(10000.0))]);
        let returns = calculator.calculate_returns(&equity);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_max_drawdown() {
        let calculator = MetricsCalculator::with_defaults();
        let equity = create_equity_curve(&[
            (0, dec!(10000.0)),
            (1000, dec!(11000.0)), // Peak
            (2000, dec!(9900.0)),  // 10% drawdown
            (3000, dec!(10500.0)), // Recovery
        ]);

        let (max_dd, _duration) = calculator.max_drawdown(&equity);

        // Max drawdown should be 10% (from 11000 to 9900)
        assert_eq!(max_dd, dec!(0.1));
    }

    #[test]
    fn test_max_drawdown_empty() {
        let calculator = MetricsCalculator::with_defaults();
        let (max_dd, duration) = calculator.max_drawdown(&[]);
        assert_eq!(max_dd, Decimal::ZERO);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let calculator = MetricsCalculator::with_defaults();
        // Varying positive returns should have positive Sharpe
        let returns = vec![dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.012)];
        let sharpe = calculator.sharpe_ratio(&returns);
        assert!(sharpe > Decimal::ZERO);
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        let calculator = MetricsCalculator::with_defaults();
        let sharpe = calculator.sharpe_ratio(&[]);
        assert_eq!(sharpe, Decimal::ZERO);
    }

    #[test]
    fn test_sortino_ratio() {
        let calculator = MetricsCalculator::with_defaults();
        // Mix of positive and negative returns
        let returns = vec![dec!(0.02), dec!(-0.01), dec!(0.03), dec!(-0.005)];
        let sortino = calculator.sortino_ratio(&returns);
        // Should be positive with more positive returns
        assert!(sortino > Decimal::ZERO);
    }

    #[test]
    fn test_var() {
        let calculator = MetricsCalculator::with_defaults();
        let returns = vec![
            dec!(-0.05),
            dec!(-0.03),
            dec!(-0.01),
            dec!(0.01),
            dec!(0.02),
            dec!(0.03),
            dec!(0.04),
            dec!(0.05),
        ];

        let var_95 = calculator.var(&returns, dec!(0.95));
        // VaR should be positive (representing loss)
        assert!(var_95 > Decimal::ZERO);
    }

    #[test]
    fn test_var_out_of_range_confidence_clamps_to_series() {
        let calculator = MetricsCalculator::with_defaults();
        let returns = vec![dec!(-0.05), dec!(0.01), dec!(0.05)];

        // Confidence above 1: negative percentile clamps to the worst return.
        assert_eq!(calculator.var(&returns, dec!(1.5)), dec!(0.05));
        // Negative confidence: percentile above 1 clamps to the best return.
        assert_eq!(calculator.var(&returns, dec!(-5)), dec!(-0.05));
    }

    #[test]
    fn test_profit_factor() {
        let calculator = MetricsCalculator::with_defaults();
        let trades = vec![
            TradeRecord::new(
                0,
                0,
                Action::Buy,
                dec!(100.0),
                dec!(110.0),
                dec!(1.0),
                dec!(100.0),
                dec!(0.0),
            ),
            TradeRecord::new(
                0,
                0,
                Action::Buy,
                dec!(100.0),
                dec!(90.0),
                dec!(1.0),
                dec!(-50.0),
                dec!(0.0),
            ),
        ];

        let pf = calculator.profit_factor(&trades);
        // 100 / 50 = 2
        assert_eq!(pf, dec!(2.0));
    }

    #[test]
    fn test_profit_factor_no_losses() {
        let calculator = MetricsCalculator::with_defaults();
        let trades = vec![TradeRecord::new(
            0,
            0,
            Action::Buy,
            dec!(100.0),
            dec!(110.0),
            dec!(1.0),
            dec!(100.0),
            dec!(0.0),
        )];

        let pf = calculator.profit_factor(&trades);
        // Should be capped at 999
        assert_eq!(pf, dec!(999));
    }

    #[test]
    fn test_calculate_full_metrics() {
        let calculator = MetricsCalculator::with_defaults();

        let equity = create_equity_curve(&[
            (0, dec!(10000.0)),
            (86400000, dec!(10100.0)),
            (172800000, dec!(10250.0)),
            (259200000, dec!(10150.0)),
            (345600000, dec!(10300.0)),
        ]);

        let trades = vec![
            TradeRecord::new(
                0,
                86400000,
                Action::Buy,
                dec!(100.0),
                dec!(101.0),
                dec!(10.0),
                dec!(100.0),
                dec!(1.0),
            ),
            TradeRecord::new(
                86400000,
                172800000,
                Action::Buy,
                dec!(101.0),
                dec!(102.5),
                dec!(10.0),
                dec!(150.0),
                dec!(1.0),
            ),
            TradeRecord::new(
                172800000,
                259200000,
                Action::Sell,
                dec!(102.5),
                dec!(101.5),
                dec!(10.0),
                dec!(-100.0),
                dec!(1.0),
            ),
            TradeRecord::new(
                259200000,
                345600000,
                Action::Buy,
                dec!(101.5),
                dec!(103.0),
                dec!(10.0),
                dec!(150.0),
                dec!(1.0),
            ),
        ];

        let metrics = calculator
            .calculate(&equity, &trades, dec!(10000.0))
            .unwrap();

        // Verify basic metrics
        assert_eq!(metrics.total_return, dec!(300.0));
        assert_eq!(metrics.total_return_pct, dec!(3.0));
        assert_eq!(metrics.total_trades, 4);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 1);
        assert!(metrics.win_rate > dec!(0.7)); // 75% win rate
        assert!(metrics.profit_factor > Decimal::ONE);
    }

    #[test]
    fn test_calculate_empty_equity() {
        let calculator = MetricsCalculator::with_defaults();
        let result = calculator.calculate(&[], &[], dec!(10000.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_no_trades() {
        let calculator = MetricsCalculator::with_defaults();
        // Use longer time span to avoid overflow in annualization
        let equity = create_equity_curve(&[
            (0, dec!(10000.0)),
            (86400000, dec!(10100.0)), // 1 day apart
        ]);

        let metrics = calculator.calculate(&equity, &[], dec!(10000.0)).unwrap();

        assert_eq!(metrics.total_trades, 0);
        assert_eq!(metrics.win_rate, Decimal::ZERO);
    }

    // PerformanceMetrics tests
    #[test]
    fn test_performance_metrics_is_profitable() {
        let metrics = PerformanceMetrics {
            total_return: dec!(100.0),
            ..Default::default()
        };
        assert!(metrics.is_profitable());

        let metrics_loss = PerformanceMetrics {
            total_return: dec!(-100.0),
            ..Default::default()
        };
        assert!(!metrics_loss.is_profitable());
    }

    #[test]
    fn test_performance_metrics_risk_reward() {
        let metrics = PerformanceMetrics {
            average_winner: dec!(100.0),
            average_loser: dec!(-50.0),
            ..Default::default()
        };

        assert_eq!(metrics.risk_reward_ratio(), dec!(2.0));
    }

    #[test]
    fn test_performance_metrics_expectancy() {
        let metrics = PerformanceMetrics {
            win_rate: dec!(0.5),
            average_winner: dec!(100.0),
            average_loser: dec!(-50.0),
            ..Default::default()
        };

        // Expectancy = 0.5 * 100 - 0.5 * 50 = 50 - 25 = 25
        assert_eq!(metrics.expectancy(), dec!(25.0));
    }

    // Helper function tests
    #[test]
    fn test_decimal_sqrt() {
        let result = decimal_sqrt(dec!(4.0));
        assert!((result - dec!(2.0)).abs() < dec!(0.0001));

        let result = decimal_sqrt(dec!(9.0));
        assert!((result - dec!(3.0)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_decimal_sqrt_zero() {
        assert_eq!(decimal_sqrt(Decimal::ZERO), Decimal::ZERO);
    }

    #[test]
    fn test_decimal_ln() {
        // ln(e) ≈ 1
        let e = dec!(2.718281828);
        let result = decimal_ln(e);
        assert!((result - Decimal::ONE).abs() < dec!(0.01));
    }

    #[test]
    fn test_decimal_exp() {
        // e^0 = 1
        assert_eq!(decimal_exp(Decimal::ZERO), Decimal::ONE);

        // e^1 ≈ 2.718
        let result = decimal_exp(Decimal::ONE);
        assert!((result - dec!(2.718281828)).abs() < dec!(0.01));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_equity_point_serialization() {
        let point = EquityPoint::new(1000, dec!(10000.0));
        let json = serde_json::to_string(&point).unwrap();
        let deserialized: EquityPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(point, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_trade_record_serialization() {
        let trade = TradeRecord::new(
            1000,
            2000,
            Action::Buy,
            dec!(100.0),
            dec!(105.0),
            dec!(10.0),
            dec!(50.0),
            dec!(1.0),
        );
        let json = serde_json::to_string(&trade).unwrap();
        let deserialized: TradeRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(trade, deserialized);
    }
}
