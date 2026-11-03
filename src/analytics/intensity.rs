//! Dynamic order intensity estimation.
//!
//! This module provides tools to dynamically estimate the order intensity
//! parameter `k` from observed trade data for more accurate Avellaneda-Stoikov
//! calculations.
//!
//! # Mathematical Background
//!
//! In the Avellaneda-Stoikov model, order arrival follows a Poisson process
//! with intensity:
//!
//! ```text
//! λ(δ) = A * exp(-k * δ)
//! ```
//!
//! Where:
//! - `λ(δ)` = arrival rate at spread δ
//! - `A` = baseline arrival rate
//! - `k` = order intensity parameter (to be estimated)
//!
//! # Estimation Method
//!
//! The estimator uses log-linear regression on fill observations:
//!
//! ```text
//! ln(λ) = ln(A) - k * δ
//! ```
//!
//! By collecting fill observations at different spread levels, we can
//! estimate both `A` and `k` through linear regression.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::intensity::{
//!     OrderIntensityEstimator, OrderIntensityConfig, FillObservation
//! };
//! use market_maker_rs::dec;
//!
//! let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
//! let mut estimator = OrderIntensityEstimator::new(config);
//!
//! // Record fill observations
//! estimator.record_fill(FillObservation::new(dec!(0.001), 500, 1000));
//! estimator.record_fill(FillObservation::new(dec!(0.002), 800, 2000));
//! estimator.record_fill(FillObservation::new(dec!(0.001), 400, 3000));
//! estimator.record_fill(FillObservation::new(dec!(0.003), 1200, 4000));
//! estimator.record_fill(FillObservation::new(dec!(0.002), 700, 5000));
//!
//! if let Ok(estimate) = estimator.estimate(6000) {
//!     println!("Estimated k: {}", estimate.k);
//!     println!("Baseline rate: {}", estimate.baseline_rate);
//! }
//! ```

use rust_decimal_macros::dec;
use std::collections::VecDeque;

use crate::Decimal;
use crate::types::decimal::{decimal_to_u64_saturating, saturating_u64};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for order intensity estimation.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::intensity::OrderIntensityConfig;
/// use market_maker_rs::dec;
///
/// let config = OrderIntensityConfig::new(
///     60_000,      // 60 second estimation window
///     10,          // minimum 10 trades for valid estimate
///     dec!(0.1),   // EWMA smoothing factor
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderIntensityConfig {
    /// Window for estimation in milliseconds.
    pub estimation_window_ms: u64,

    /// Minimum trades required for valid estimate.
    pub min_trades: usize,

    /// Smoothing factor for EWMA updates (0 < α ≤ 1).
    /// Higher values give more weight to recent observations.
    pub smoothing_factor: Decimal,

    /// Default k value to use when estimate is unavailable.
    pub default_k: Decimal,

    /// Maximum k value to prevent extreme estimates.
    pub max_k: Decimal,

    /// Minimum k value to prevent extreme estimates.
    pub min_k: Decimal,
}

impl OrderIntensityConfig {
    /// Creates a new `OrderIntensityConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `estimation_window_ms` - Window for estimation in milliseconds
    /// * `min_trades` - Minimum trades required for valid estimate
    /// * `smoothing_factor` - EWMA smoothing factor (0 < α ≤ 1)
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::analytics::intensity::OrderIntensityConfig;
    /// use market_maker_rs::dec;
    ///
    /// let config = OrderIntensityConfig::new(60_000, 10, dec!(0.1)).unwrap();
    /// ```
    pub fn new(
        estimation_window_ms: u64,
        min_trades: usize,
        smoothing_factor: Decimal,
    ) -> MMResult<Self> {
        if estimation_window_ms == 0 {
            return Err(MMError::InvalidConfiguration(
                "estimation_window_ms must be positive".to_string(),
            ));
        }

        if min_trades == 0 {
            return Err(MMError::InvalidConfiguration(
                "min_trades must be positive".to_string(),
            ));
        }

        if smoothing_factor <= Decimal::ZERO || smoothing_factor > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "smoothing_factor must be in (0, 1]".to_string(),
            ));
        }

        Ok(Self {
            estimation_window_ms,
            min_trades,
            smoothing_factor,
            default_k: dec!(1.5),
            max_k: dec!(10.0),
            min_k: dec!(0.1),
        })
    }

    /// Sets the default k value.
    #[must_use]
    pub fn with_default_k(mut self, default_k: Decimal) -> Self {
        self.default_k = default_k;
        self
    }

    /// Sets the k value bounds.
    #[must_use]
    pub fn with_k_bounds(mut self, min_k: Decimal, max_k: Decimal) -> Self {
        self.min_k = min_k;
        self.max_k = max_k;
        self
    }
}

/// A single fill observation for intensity estimation.
///
/// Records the spread at which a fill occurred and the time it took.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::intensity::FillObservation;
/// use market_maker_rs::dec;
///
/// let obs = FillObservation::new(
///     dec!(0.001),  // spread at fill (0.1%)
///     500,          // time to fill: 500ms
///     1234567890,   // timestamp
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FillObservation {
    /// Spread at which the fill occurred (as decimal, e.g., 0.001 = 0.1%).
    pub spread_at_fill: Decimal,

    /// Time from order placement to fill in milliseconds.
    pub time_to_fill_ms: u64,

    /// Timestamp of the fill in milliseconds.
    pub timestamp: u64,

    /// Side of the fill (optional, for more detailed analysis).
    pub side: Option<FillSide>,
}

/// Side of a fill observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FillSide {
    /// Bid side fill.
    Bid,
    /// Ask side fill.
    Ask,
}

impl FillObservation {
    /// Creates a new fill observation.
    ///
    /// # Arguments
    ///
    /// * `spread_at_fill` - Spread at which the fill occurred
    /// * `time_to_fill_ms` - Time from order placement to fill
    /// * `timestamp` - Timestamp of the fill
    #[must_use]
    pub fn new(spread_at_fill: Decimal, time_to_fill_ms: u64, timestamp: u64) -> Self {
        Self {
            spread_at_fill,
            time_to_fill_ms,
            timestamp,
            side: None,
        }
    }

    /// Creates a new fill observation with side information.
    #[must_use]
    pub fn with_side(
        spread_at_fill: Decimal,
        time_to_fill_ms: u64,
        timestamp: u64,
        side: FillSide,
    ) -> Self {
        Self {
            spread_at_fill,
            time_to_fill_ms,
            timestamp,
            side: Some(side),
        }
    }

    /// Returns the implied arrival rate (1 / time_to_fill in seconds).
    #[must_use]
    pub fn implied_rate(&self) -> Decimal {
        if self.time_to_fill_ms > 0 {
            Decimal::from(1000) / Decimal::from(self.time_to_fill_ms)
        } else {
            Decimal::from(1000) // Default to 1000/s for instant fills
        }
    }
}

/// Result of order intensity estimation.
///
/// Contains the estimated parameters and quality metrics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntensityEstimate {
    /// Estimated k parameter (order intensity).
    pub k: Decimal,

    /// Baseline arrival rate A (arrivals per second at zero spread).
    pub baseline_rate: Decimal,

    /// Confidence/quality of estimate (0 to 1).
    /// Based on sample size and fit quality.
    pub confidence: Decimal,

    /// Number of observations used in the estimate.
    pub sample_size: usize,

    /// R-squared value of the regression fit.
    pub r_squared: Decimal,

    /// Standard error of the k estimate.
    pub k_std_error: Decimal,

    /// Timestamp when estimate was computed.
    pub timestamp: u64,
}

impl IntensityEstimate {
    /// Returns true if this is a high-confidence estimate.
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= dec!(0.7)
    }

    /// Returns true if this is a low-confidence estimate.
    #[must_use]
    pub fn is_low_confidence(&self) -> bool {
        self.confidence < dec!(0.3)
    }
}

/// Order intensity estimator for dynamic k parameter estimation.
///
/// Collects fill observations and estimates the order intensity parameter
/// using log-linear regression.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::intensity::{
///     OrderIntensityEstimator, OrderIntensityConfig, FillObservation
/// };
/// use market_maker_rs::{Decimal, dec};
///
/// let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
/// let mut estimator = OrderIntensityEstimator::new(config);
///
/// // Record fills at different spreads
/// for i in 0..10 {
///     let spread = dec!(0.001) + dec!(0.0001) * Decimal::from(i % 3);
///     let time_to_fill = 500 + (i * 100) as u64;
///     estimator.record_fill(FillObservation::new(spread, time_to_fill, i * 1000));
/// }
///
/// let k = estimator.get_k_or_default(dec!(1.5));
/// println!("Using k = {}", k);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderIntensityEstimator {
    /// Configuration parameters.
    config: OrderIntensityConfig,

    /// Collected fill observations.
    observations: VecDeque<FillObservation>,

    /// Current estimate (if available).
    current_estimate: Option<IntensityEstimate>,

    /// EWMA of k for smoothing.
    ewma_k: Option<Decimal>,

    /// Total fills recorded.
    total_fills: u64,
}

impl OrderIntensityEstimator {
    /// Creates a new order intensity estimator.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters
    #[must_use]
    pub fn new(config: OrderIntensityConfig) -> Self {
        Self {
            config,
            observations: VecDeque::new(),
            current_estimate: None,
            ewma_k: None,
            total_fills: 0,
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &OrderIntensityConfig {
        &self.config
    }

    /// Returns the number of observations in the window.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Returns total fills recorded (including expired ones).
    #[must_use]
    pub fn total_fills(&self) -> u64 {
        self.total_fills
    }

    /// Records a fill observation.
    ///
    /// # Arguments
    ///
    /// * `observation` - The fill observation to record
    pub fn record_fill(&mut self, observation: FillObservation) {
        self.observations.push_back(observation);
        self.total_fills += 1;
    }

    /// Records a fill from components.
    ///
    /// Convenience method to record a fill without creating a struct.
    pub fn record_fill_components(
        &mut self,
        spread_at_fill: Decimal,
        time_to_fill_ms: u64,
        timestamp: u64,
    ) {
        self.record_fill(FillObservation::new(
            spread_at_fill,
            time_to_fill_ms,
            timestamp,
        ));
    }

    /// Estimates k from collected observations.
    ///
    /// Uses log-linear regression: ln(λ) = ln(A) - k * δ
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidMarketState` if insufficient observations.
    pub fn estimate(&mut self, current_time: u64) -> MMResult<IntensityEstimate> {
        // Clean up old observations first
        self.cleanup(current_time);

        if self.observations.len() < self.config.min_trades {
            return Err(MMError::InvalidMarketState(format!(
                "insufficient observations: {} < {}",
                self.observations.len(),
                self.config.min_trades
            )));
        }

        // Collect data points for regression
        let mut spreads: Vec<Decimal> = Vec::new();
        let mut log_rates: Vec<Decimal> = Vec::new();

        for obs in &self.observations {
            let rate = obs.implied_rate();
            if rate > Decimal::ZERO {
                spreads.push(obs.spread_at_fill);
                // Use natural log approximation
                log_rates.push(ln_approx(rate));
            }
        }

        if spreads.len() < 2 {
            return Err(MMError::InvalidMarketState(
                "need at least 2 valid observations for regression".to_string(),
            ));
        }

        // Check for spread variance (can't estimate k if all same spread)
        let spread_variance = calculate_variance(&spreads);
        if spread_variance < dec!(0.0000001) {
            return Err(MMError::InvalidMarketState(
                "insufficient spread variance for estimation".to_string(),
            ));
        }

        // Perform linear regression: ln(rate) = ln(A) - k * spread
        let (intercept, slope, r_squared) = linear_regression(&spreads, &log_rates);

        // k = -slope (since ln(rate) = ln(A) - k * spread)
        let mut k = -slope;

        // Clamp k to valid range
        k = k.max(self.config.min_k).min(self.config.max_k);

        // A = exp(intercept)
        let baseline_rate = exp_approx(intercept);

        // Calculate standard error of k
        let k_std_error = calculate_slope_std_error(&spreads, &log_rates, slope, intercept);

        // Calculate confidence based on sample size and R²
        let sample_factor = Decimal::from(self.observations.len().min(100)) / Decimal::from(100);
        let r2_factor = r_squared.max(Decimal::ZERO);
        let confidence = (sample_factor * dec!(0.5) + r2_factor * dec!(0.5)).min(Decimal::ONE);

        // Apply EWMA smoothing
        let smoothed_k = match self.ewma_k {
            Some(prev_k) => {
                let alpha = self.config.smoothing_factor;
                alpha * k + (Decimal::ONE - alpha) * prev_k
            }
            None => k,
        };
        self.ewma_k = Some(smoothed_k);

        let estimate = IntensityEstimate {
            k: smoothed_k,
            baseline_rate,
            confidence,
            sample_size: self.observations.len(),
            r_squared,
            k_std_error,
            timestamp: current_time,
        };

        self.current_estimate = Some(estimate.clone());
        Ok(estimate)
    }

    /// Gets the current estimate if available.
    #[must_use]
    pub fn get_estimate(&self) -> Option<&IntensityEstimate> {
        self.current_estimate.as_ref()
    }

    /// Gets k value with fallback to default.
    ///
    /// # Arguments
    ///
    /// * `default` - Default k value if no estimate available
    #[must_use]
    pub fn get_k_or_default(&self, default: Decimal) -> Decimal {
        self.current_estimate
            .as_ref()
            .map(|e| e.k)
            .or(self.ewma_k)
            .unwrap_or(default)
    }

    /// Gets k value using config default.
    #[must_use]
    pub fn get_k(&self) -> Decimal {
        self.get_k_or_default(self.config.default_k)
    }

    /// Calculates expected fill probability at given spread.
    ///
    /// Uses the formula: P(fill in time t) = 1 - exp(-λ(δ) * t)
    /// where λ(δ) = A * exp(-k * δ)
    ///
    /// # Arguments
    ///
    /// * `spread` - Spread level
    /// * `time_horizon_ms` - Time horizon in milliseconds
    ///
    /// # Returns
    ///
    /// Fill probability in [0, 1], or None if no estimate available.
    #[must_use]
    pub fn fill_probability(&self, spread: Decimal, time_horizon_ms: u64) -> Option<Decimal> {
        let estimate = self.current_estimate.as_ref()?;

        // λ(δ) = A * exp(-k * δ)
        let lambda = estimate.baseline_rate * exp_approx(-estimate.k * spread);

        // Convert time to seconds
        let time_seconds = Decimal::from(time_horizon_ms) / Decimal::from(1000);

        // P(fill) = 1 - exp(-λ * t)
        let prob = Decimal::ONE - exp_approx(-lambda * time_seconds);

        Some(prob.max(Decimal::ZERO).min(Decimal::ONE))
    }

    /// Calculates expected time to fill at given spread.
    ///
    /// Uses the formula: `E\[T\] = 1 / λ(δ)`
    ///
    /// # Arguments
    ///
    /// * `spread` - Spread level
    ///
    /// # Returns
    ///
    /// Expected time to fill in milliseconds, or None if no estimate.
    #[must_use]
    pub fn expected_time_to_fill_ms(&self, spread: Decimal) -> Option<u64> {
        let estimate = self.current_estimate.as_ref()?;

        // λ(δ) = A * exp(-k * δ)
        let lambda = estimate.baseline_rate * exp_approx(-estimate.k * spread);

        if lambda > Decimal::ZERO {
            // E[T] = 1/λ in seconds, convert to ms
            let time_seconds = Decimal::ONE / lambda;
            let time_ms = time_seconds * Decimal::from(1000);
            Some(decimal_to_u64_saturating(time_ms))
        } else {
            None
        }
    }

    /// Clears old observations outside the estimation window.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    pub fn cleanup(&mut self, current_time: u64) {
        let window_start = current_time.saturating_sub(self.config.estimation_window_ms);
        while let Some(obs) = self.observations.front() {
            if obs.timestamp < window_start {
                self.observations.pop_front();
            } else {
                break;
            }
        }
    }

    /// Resets the estimator, clearing all observations and estimates.
    pub fn reset(&mut self) {
        self.observations.clear();
        self.current_estimate = None;
        self.ewma_k = None;
        self.total_fills = 0;
    }

    /// Returns statistics about the observations.
    #[must_use]
    pub fn observation_stats(&self) -> Option<ObservationStats> {
        if self.observations.is_empty() {
            return None;
        }

        let spreads: Vec<Decimal> = self.observations.iter().map(|o| o.spread_at_fill).collect();
        let times: Vec<u64> = self
            .observations
            .iter()
            .map(|o| o.time_to_fill_ms)
            .collect();

        let mean_spread = spreads.iter().copied().sum::<Decimal>() / Decimal::from(spreads.len());
        let mean_time = times
            .iter()
            .fold(0u64, |acc, &t| acc.saturating_add(t))
            .checked_div(saturating_u64(times.len()))
            .unwrap_or(0);

        let min_spread = spreads.iter().copied().min().unwrap_or(Decimal::ZERO);
        let max_spread = spreads.iter().copied().max().unwrap_or(Decimal::ZERO);

        let min_time = times.iter().copied().min().unwrap_or(0);
        let max_time = times.iter().copied().max().unwrap_or(0);

        Some(ObservationStats {
            count: self.observations.len(),
            mean_spread,
            min_spread,
            max_spread,
            mean_time_to_fill_ms: mean_time,
            min_time_to_fill_ms: min_time,
            max_time_to_fill_ms: max_time,
        })
    }
}

/// Statistics about collected observations.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ObservationStats {
    /// Number of observations.
    pub count: usize,

    /// Mean spread at fill.
    pub mean_spread: Decimal,

    /// Minimum spread at fill.
    pub min_spread: Decimal,

    /// Maximum spread at fill.
    pub max_spread: Decimal,

    /// Mean time to fill in milliseconds.
    pub mean_time_to_fill_ms: u64,

    /// Minimum time to fill in milliseconds.
    pub min_time_to_fill_ms: u64,

    /// Maximum time to fill in milliseconds.
    pub max_time_to_fill_ms: u64,
}

/// Approximates natural logarithm using Taylor series.
fn ln_approx(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::from(-1000); // Large negative for log(0)
    }

    // Use ln(x) = ln(a * 10^n) = ln(a) + n * ln(10)
    // where 1 <= a < 10
    let ln_10 = dec!(2.302585);

    // Normalize x to [1, 10)
    let mut normalized = x;
    let mut exponent = 0i32;

    while normalized >= Decimal::from(10) {
        normalized /= Decimal::from(10);
        exponent += 1;
    }
    while normalized < Decimal::ONE {
        normalized *= Decimal::from(10);
        exponent -= 1;
    }

    // For x in [1, 10), use ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - ...
    // Better: use ln(x) = 2 * arctanh((x-1)/(x+1))
    let y = (normalized - Decimal::ONE) / (normalized + Decimal::ONE);
    let y2 = y * y;

    // arctanh(y) ≈ y + y³/3 + y⁵/5 + y⁷/7
    let ln_normalized = Decimal::from(2)
        * (y + y * y2 / Decimal::from(3)
            + y * y2 * y2 / Decimal::from(5)
            + y * y2 * y2 * y2 / Decimal::from(7));

    ln_normalized + Decimal::from(exponent) * ln_10
}

/// Approximates exponential function using Taylor series.
fn exp_approx(x: Decimal) -> Decimal {
    // Clamp to prevent overflow
    let x_clamped = x.max(Decimal::from(-20)).min(Decimal::from(20));

    // exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
    let mut result = Decimal::ONE;
    let mut term = Decimal::ONE;

    for i in 1..20 {
        term = term * x_clamped / Decimal::from(i);
        result += term;
        if term.abs() < dec!(0.0000001) {
            break;
        }
    }

    result.max(Decimal::ZERO)
}

/// Performs simple linear regression.
/// Returns (intercept, slope, r_squared).
fn linear_regression(x: &[Decimal], y: &[Decimal]) -> (Decimal, Decimal, Decimal) {
    let n = Decimal::from(x.len());
    if x.len() < 2 {
        return (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    }

    let sum_x: Decimal = x.iter().copied().sum();
    let sum_y: Decimal = y.iter().copied().sum();
    let sum_xy: Decimal = x.iter().zip(y.iter()).map(|(xi, yi)| *xi * *yi).sum();
    let sum_x2: Decimal = x.iter().map(|xi| *xi * *xi).sum();
    let sum_y2: Decimal = y.iter().map(|yi| *yi * *yi).sum();

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let ss_xy = sum_xy - n * mean_x * mean_y;
    let ss_xx = sum_x2 - n * mean_x * mean_x;
    let ss_yy = sum_y2 - n * mean_y * mean_y;

    if ss_xx.abs() < dec!(0.0000001) {
        return (mean_y, Decimal::ZERO, Decimal::ZERO);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    // R² = (SS_xy)² / (SS_xx * SS_yy)
    let r_squared = if ss_yy.abs() > dec!(0.0000001) {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        Decimal::ONE
    };

    (
        intercept,
        slope,
        r_squared.max(Decimal::ZERO).min(Decimal::ONE),
    )
}

/// Calculates variance of a slice.
fn calculate_variance(values: &[Decimal]) -> Decimal {
    if values.len() < 2 {
        return Decimal::ZERO;
    }

    let n = Decimal::from(values.len());
    let mean = values.iter().copied().sum::<Decimal>() / n;
    let sum_sq_diff: Decimal = values.iter().map(|v| (*v - mean) * (*v - mean)).sum();

    sum_sq_diff / (n - Decimal::ONE)
}

/// Calculates standard error of the slope estimate.
fn calculate_slope_std_error(
    x: &[Decimal],
    y: &[Decimal],
    slope: Decimal,
    intercept: Decimal,
) -> Decimal {
    if x.len() < 3 {
        return Decimal::ZERO;
    }

    let n = Decimal::from(x.len());

    // Calculate residual sum of squares
    let rss: Decimal = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let predicted = intercept + slope * *xi;
            let residual = *yi - predicted;
            residual * residual
        })
        .sum();

    // Calculate SS_xx
    let mean_x = x.iter().copied().sum::<Decimal>() / n;
    let ss_xx: Decimal = x.iter().map(|xi| (*xi - mean_x) * (*xi - mean_x)).sum();

    if ss_xx.abs() < dec!(0.0000001) {
        return Decimal::ZERO;
    }

    // Standard error = sqrt(RSS / (n-2)) / sqrt(SS_xx)
    let mse = rss / (n - Decimal::from(2));
    let se_squared = mse / ss_xx;

    // Approximate sqrt
    decimal_sqrt_approx(se_squared)
}

/// Approximate square root using Newton's method.
fn decimal_sqrt_approx(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut guess = x / Decimal::from(2);
    let epsilon = dec!(0.0000001);

    for _ in 0..20 {
        let new_guess = (guess + x / guess) / Decimal::from(2);
        if (new_guess - guess).abs() < epsilon {
            return new_guess;
        }
        guess = new_guess;
    }

    guess
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_config_valid() {
        let config = OrderIntensityConfig::new(60_000, 10, dec!(0.1));
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.estimation_window_ms, 60_000);
        assert_eq!(config.min_trades, 10);
        assert_eq!(config.smoothing_factor, dec!(0.1));
    }

    #[test]
    fn test_config_invalid_window() {
        let config = OrderIntensityConfig::new(0, 10, dec!(0.1));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_min_trades() {
        let config = OrderIntensityConfig::new(60_000, 0, dec!(0.1));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_smoothing() {
        let config = OrderIntensityConfig::new(60_000, 10, dec!(0.0));
        assert!(config.is_err());

        let config = OrderIntensityConfig::new(60_000, 10, dec!(1.1));
        assert!(config.is_err());
    }

    #[test]
    fn test_fill_observation() {
        let obs = FillObservation::new(dec!(0.001), 500, 1000);

        assert_eq!(obs.spread_at_fill, dec!(0.001));
        assert_eq!(obs.time_to_fill_ms, 500);
        assert_eq!(obs.timestamp, 1000);

        // Implied rate = 1000 / 500 = 2 per second
        assert_eq!(obs.implied_rate(), dec!(2));
    }

    #[test]
    fn test_estimator_new() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let estimator = OrderIntensityEstimator::new(config);

        assert_eq!(estimator.observation_count(), 0);
        assert!(estimator.get_estimate().is_none());
    }

    #[test]
    fn test_estimator_record_fill() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        estimator.record_fill(FillObservation::new(dec!(0.001), 500, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 800, 2000));

        assert_eq!(estimator.observation_count(), 2);
        assert_eq!(estimator.total_fills(), 2);
    }

    #[test]
    fn test_estimator_insufficient_observations() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        estimator.record_fill(FillObservation::new(dec!(0.001), 500, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 800, 2000));

        let result = estimator.estimate(3000);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimator_estimate() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        // Add observations with varying spreads and times
        // Higher spread -> longer time to fill (lower rate)
        estimator.record_fill(FillObservation::new(dec!(0.001), 200, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 400, 2000));
        estimator.record_fill(FillObservation::new(dec!(0.003), 600, 3000));
        estimator.record_fill(FillObservation::new(dec!(0.001), 250, 4000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 450, 5000));

        let result = estimator.estimate(6000);
        assert!(result.is_ok());

        let estimate = result.unwrap();
        assert!(estimate.k > Decimal::ZERO);
        assert!(estimate.baseline_rate > Decimal::ZERO);
        assert!(estimate.confidence >= Decimal::ZERO);
        assert!(estimate.confidence <= Decimal::ONE);
    }

    #[test]
    fn test_estimator_get_k_or_default() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let estimator = OrderIntensityEstimator::new(config);

        // No estimate yet, should return default
        assert_eq!(estimator.get_k_or_default(dec!(1.5)), dec!(1.5));
    }

    #[test]
    fn test_estimator_cleanup() {
        let config = OrderIntensityConfig::new(5000, 2, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        estimator.record_fill(FillObservation::new(dec!(0.001), 500, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 800, 2000));
        estimator.record_fill(FillObservation::new(dec!(0.001), 400, 8000));

        assert_eq!(estimator.observation_count(), 3);

        // Cleanup at t=10000, window is [5000, 10000]
        estimator.cleanup(10_000);

        // Only observation at t=8000 should remain
        assert_eq!(estimator.observation_count(), 1);
    }

    #[test]
    fn test_estimator_reset() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        estimator.record_fill(FillObservation::new(dec!(0.001), 500, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 800, 2000));

        estimator.reset();

        assert_eq!(estimator.observation_count(), 0);
        assert_eq!(estimator.total_fills(), 0);
        assert!(estimator.get_estimate().is_none());
    }

    #[test]
    fn test_fill_probability() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        // Add enough observations
        for i in 0..10 {
            let spread = dec!(0.001) + dec!(0.0005) * Decimal::from(i % 3);
            let time = 300 + (i * 50);
            estimator.record_fill(FillObservation::new(spread, time, i * 1000));
        }

        let _ = estimator.estimate(11_000);

        // Should have fill probability
        let prob = estimator.fill_probability(dec!(0.001), 1000);
        assert!(prob.is_some());

        let p = prob.unwrap();
        assert!(p >= Decimal::ZERO);
        assert!(p <= Decimal::ONE);
    }

    #[test]
    fn test_observation_stats() {
        let config = OrderIntensityConfig::new(60_000, 5, dec!(0.1)).unwrap();
        let mut estimator = OrderIntensityEstimator::new(config);

        estimator.record_fill(FillObservation::new(dec!(0.001), 200, 1000));
        estimator.record_fill(FillObservation::new(dec!(0.002), 400, 2000));
        estimator.record_fill(FillObservation::new(dec!(0.003), 600, 3000));

        let stats = estimator.observation_stats().unwrap();

        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_spread, dec!(0.001));
        assert_eq!(stats.max_spread, dec!(0.003));
        assert_eq!(stats.min_time_to_fill_ms, 200);
        assert_eq!(stats.max_time_to_fill_ms, 600);
    }

    #[test]
    fn test_ln_approx() {
        // ln(1) = 0
        let result = ln_approx(Decimal::ONE);
        assert!((result - Decimal::ZERO).abs() < dec!(0.01));

        // ln(e) ≈ 1
        let e = dec!(2.718281828);
        let result = ln_approx(e);
        assert!((result - Decimal::ONE).abs() < dec!(0.01));

        // ln(10) ≈ 2.303
        let result = ln_approx(Decimal::from(10));
        assert!((result - dec!(2.303)).abs() < dec!(0.01));
    }

    #[test]
    fn test_exp_approx() {
        // exp(0) = 1
        let result = exp_approx(Decimal::ZERO);
        assert!((result - Decimal::ONE).abs() < dec!(0.001));

        // exp(1) ≈ 2.718
        let result = exp_approx(Decimal::ONE);
        assert!((result - dec!(2.718)).abs() < dec!(0.01));

        // exp(-1) ≈ 0.368
        let result = exp_approx(-Decimal::ONE);
        assert!((result - dec!(0.368)).abs() < dec!(0.01));
    }

    #[test]
    fn test_linear_regression() {
        // Perfect linear relationship: y = 2 + 3x
        let x = vec![dec!(1), dec!(2), dec!(3), dec!(4), dec!(5)];
        let y = vec![dec!(5), dec!(8), dec!(11), dec!(14), dec!(17)];

        let (intercept, slope, r_squared) = linear_regression(&x, &y);

        assert!((intercept - dec!(2)).abs() < dec!(0.001));
        assert!((slope - dec!(3)).abs() < dec!(0.001));
        assert!((r_squared - Decimal::ONE).abs() < dec!(0.001));
    }

    #[test]
    fn test_intensity_estimate_confidence() {
        let estimate = IntensityEstimate {
            k: dec!(1.5),
            baseline_rate: dec!(2.0),
            confidence: dec!(0.8),
            sample_size: 50,
            r_squared: dec!(0.9),
            k_std_error: dec!(0.1),
            timestamp: 1000,
        };

        assert!(estimate.is_high_confidence());
        assert!(!estimate.is_low_confidence());
    }

    #[test]
    fn test_fill_side() {
        let obs = FillObservation::with_side(dec!(0.001), 500, 1000, FillSide::Bid);
        assert_eq!(obs.side, Some(FillSide::Bid));

        let obs = FillObservation::with_side(dec!(0.001), 500, 1000, FillSide::Ask);
        assert_eq!(obs.side, Some(FillSide::Ask));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config = OrderIntensityConfig::new(60_000, 10, dec!(0.1)).unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OrderIntensityConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
