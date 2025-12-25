//! Parameter calibration tools for strategy optimization.
//!
//! This module provides tools to calibrate strategy parameters from historical
//! data for optimal performance.
//!
//! # Overview
//!
//! Optimal parameter selection is crucial for strategy performance:
//!
//! - **Risk aversion (γ)**: Affects inventory management aggressiveness
//! - **Order intensity (k)**: Should reflect actual market conditions
//! - **Volatility regime**: Parameters may need adjustment for different regimes
//!
//! # Components
//!
//! - [`CalibrationConfig`]: Configuration for calibration process
//! - [`CalibrationResult`]: Result with value, confidence interval, and quality
//! - [`RiskAversionCalibrator`]: Calibrates γ from half-life or history
//! - [`OrderIntensityCalibrator`]: Calibrates k from fill observations
//! - [`VolatilityRegimeDetector`]: Detects and classifies volatility regimes
//! - [`ParameterOptimizer`]: Combines all calibrators for full optimization
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::strategy::calibration::{
//!     CalibrationConfig, RiskAversionCalibrator, VolatilityRegime,
//!     VolatilityRegimeDetector,
//! };
//! use market_maker_rs::dec;
//!
//! // Calibrate risk aversion from desired half-life
//! let config = CalibrationConfig::default();
//! let calibrator = RiskAversionCalibrator::new(config);
//!
//! let result = calibrator.calibrate_from_halflife(
//!     300_000,      // 5 minute half-life
//!     dec!(0.02),   // 2% volatility
//! );
//!
//! println!("Recommended γ: {} (quality: {})", result.value, result.quality);
//!
//! // Detect volatility regime
//! let detector = VolatilityRegimeDetector::new(dec!(1.5), 3_600_000);
//! let regime = detector.detect_regime(dec!(0.04), dec!(0.02));
//! assert_eq!(regime, VolatilityRegime::High);
//! ```

use crate::Decimal;
use crate::types::decimal::{decimal_ln, decimal_sqrt};
use crate::types::error::{MMError, MMResult};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Natural logarithm of 2, used in half-life calculations.
const LN_2: &str = "0.6931471805599453";

/// Configuration for the calibration process.
///
/// Controls the behavior of all calibrators including minimum sample
/// requirements and estimation methods.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::calibration::CalibrationConfig;
/// use market_maker_rs::dec;
///
/// let config = CalibrationConfig {
///     min_samples: 100,
///     confidence_level: dec!(0.95),
///     robust_estimation: true,
/// };
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationConfig {
    /// Minimum data points required for calibration.
    pub min_samples: usize,

    /// Confidence level for interval estimates (0 to 1).
    pub confidence_level: Decimal,

    /// Whether to use robust estimation (outlier-resistant).
    pub robust_estimation: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            min_samples: 30,
            confidence_level: Decimal::from_str_exact("0.95").unwrap(),
            robust_estimation: true,
        }
    }
}

impl CalibrationConfig {
    /// Creates a new calibration config.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum data points required
    /// * `confidence_level` - Confidence level (0 to 1)
    /// * `robust_estimation` - Use outlier-resistant methods
    #[must_use]
    pub fn new(min_samples: usize, confidence_level: Decimal, robust_estimation: bool) -> Self {
        Self {
            min_samples,
            confidence_level,
            robust_estimation,
        }
    }

    /// Creates a config for quick calibration with fewer samples.
    #[must_use]
    pub fn quick() -> Self {
        Self {
            min_samples: 10,
            confidence_level: Decimal::from_str_exact("0.90").unwrap(),
            robust_estimation: false,
        }
    }

    /// Creates a config for high-precision calibration.
    #[must_use]
    pub fn precise() -> Self {
        Self {
            min_samples: 100,
            confidence_level: Decimal::from_str_exact("0.99").unwrap(),
            robust_estimation: true,
        }
    }
}

/// Result of a calibration process.
///
/// Contains the estimated value along with confidence interval,
/// sample size, quality score, and any notes or warnings.
///
/// # Type Parameter
///
/// * `T` - The type of the calibrated value (usually `Decimal`)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationResult<T> {
    /// Estimated value.
    pub value: T,

    /// Confidence interval (low, high).
    pub confidence_interval: (T, T),

    /// Number of samples used in estimation.
    pub sample_size: usize,

    /// Quality score from 0 (poor) to 1 (excellent).
    pub quality: Decimal,

    /// Warnings or notes about the calibration.
    pub notes: Vec<String>,
}

impl<T: fmt::Display> fmt::Display for CalibrationResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}, {}] (n={}, q={:.2})",
            self.value,
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.sample_size,
            self.quality
        )
    }
}

impl<T: Clone> CalibrationResult<T> {
    /// Creates a new calibration result.
    #[must_use]
    pub fn new(
        value: T,
        confidence_interval: (T, T),
        sample_size: usize,
        quality: Decimal,
    ) -> Self {
        Self {
            value,
            confidence_interval,
            sample_size,
            quality,
            notes: Vec::new(),
        }
    }

    /// Adds a note to the result.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Creates a result with a note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.add_note(note);
        self
    }

    /// Returns true if the calibration quality is acceptable (>= 0.5).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.quality >= Decimal::from_str_exact("0.5").unwrap()
    }

    /// Returns true if the calibration quality is good (>= 0.7).
    #[must_use]
    pub fn is_good(&self) -> bool {
        self.quality >= Decimal::from_str_exact("0.7").unwrap()
    }

    /// Returns true if the calibration quality is excellent (>= 0.9).
    #[must_use]
    pub fn is_excellent(&self) -> bool {
        self.quality >= Decimal::from_str_exact("0.9").unwrap()
    }
}

/// Risk aversion (γ) calibrator.
///
/// Calibrates the risk aversion parameter based on desired inventory
/// behavior or historical data.
///
/// # Mathematical Background
///
/// In the Avellaneda-Stoikov model, γ controls how aggressively the
/// market maker skews quotes to manage inventory. Higher γ means
/// faster inventory mean-reversion but potentially lower profits.
///
/// The half-life formula is derived from the inventory dynamics:
/// ```text
/// γ = ln(2) / (half_life × σ²)
/// ```
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::calibration::{CalibrationConfig, RiskAversionCalibrator};
/// use market_maker_rs::dec;
///
/// let calibrator = RiskAversionCalibrator::new(CalibrationConfig::default());
///
/// // 5-minute half-life with 2% volatility
/// let result = calibrator.calibrate_from_halflife(300_000, dec!(0.02));
/// println!("Recommended γ: {}", result.value);
/// ```
#[derive(Debug, Clone)]
pub struct RiskAversionCalibrator {
    config: CalibrationConfig,
}

impl RiskAversionCalibrator {
    /// Creates a new risk aversion calibrator.
    #[must_use]
    pub fn new(config: CalibrationConfig) -> Self {
        Self { config }
    }

    /// Calibrates γ based on desired inventory half-life.
    ///
    /// Half-life is the time for inventory to decay to 50% through quote skewing.
    ///
    /// # Formula
    ///
    /// ```text
    /// γ = ln(2) / (half_life_seconds × σ²)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `desired_halflife_ms` - Desired half-life in milliseconds
    /// * `volatility` - Annualized volatility (e.g., 0.02 for 2%)
    ///
    /// # Returns
    ///
    /// Calibration result with recommended γ value.
    #[must_use]
    pub fn calibrate_from_halflife(
        &self,
        desired_halflife_ms: u64,
        volatility: Decimal,
    ) -> CalibrationResult<Decimal> {
        let ln2 = Decimal::from_str_exact(LN_2).unwrap_or(Decimal::ONE);

        // Convert ms to seconds
        let halflife_seconds =
            Decimal::from(desired_halflife_ms) / Decimal::from_str_exact("1000").unwrap();

        // γ = ln(2) / (half_life × σ²)
        let sigma_squared = volatility * volatility;

        let gamma = if sigma_squared.is_zero() || halflife_seconds.is_zero() {
            Decimal::ONE
        } else {
            ln2 / (halflife_seconds * sigma_squared)
        };

        // Confidence interval based on volatility uncertainty (±20%)
        let margin = gamma * Decimal::from_str_exact("0.2").unwrap();
        let low = gamma - margin;
        let high = gamma + margin;

        // Quality based on reasonable parameter ranges
        let quality = self.assess_gamma_quality(gamma);

        let mut result = CalibrationResult::new(gamma, (low, high), 1, quality);

        if gamma > Decimal::from(100) {
            result.add_note("Very high γ - consider longer half-life");
        }
        if gamma < Decimal::from_str_exact("0.01").unwrap() {
            result.add_note("Very low γ - consider shorter half-life");
        }

        result
    }

    /// Calibrates γ from historical inventory and PnL data.
    ///
    /// Finds γ that would have minimized inventory variance while
    /// maintaining profitability.
    ///
    /// # Arguments
    ///
    /// * `inventory_history` - Slice of (timestamp_ms, inventory) tuples
    /// * `pnl_history` - Slice of (timestamp_ms, pnl) tuples
    /// * `volatility` - Annualized volatility
    ///
    /// # Returns
    ///
    /// Calibration result or error if insufficient data.
    pub fn calibrate_from_history(
        &self,
        inventory_history: &[(u64, Decimal)],
        pnl_history: &[(u64, Decimal)],
        volatility: Decimal,
    ) -> MMResult<CalibrationResult<Decimal>> {
        if inventory_history.len() < self.config.min_samples {
            return Err(MMError::InvalidConfiguration(format!(
                "Insufficient inventory samples: {} < {}",
                inventory_history.len(),
                self.config.min_samples
            )));
        }

        if pnl_history.len() < self.config.min_samples {
            return Err(MMError::InvalidConfiguration(format!(
                "Insufficient PnL samples: {} < {}",
                pnl_history.len(),
                self.config.min_samples
            )));
        }

        // Calculate inventory statistics
        let inventories: Vec<Decimal> = inventory_history.iter().map(|(_, inv)| *inv).collect();
        let inv_mean = self.mean(&inventories);
        let inv_variance = self.variance(&inventories, inv_mean);

        // Calculate observed half-life from autocorrelation
        let observed_halflife_ms = self.estimate_halflife(inventory_history);

        // Calculate γ from observed half-life
        let ln2 = Decimal::from_str_exact(LN_2).unwrap_or(Decimal::ONE);
        let halflife_seconds =
            Decimal::from(observed_halflife_ms) / Decimal::from_str_exact("1000").unwrap();
        let sigma_squared = volatility * volatility;

        let gamma = if sigma_squared.is_zero() || halflife_seconds.is_zero() {
            Decimal::ONE
        } else {
            ln2 / (halflife_seconds * sigma_squared)
        };

        // Calculate quality based on data consistency
        let quality = self.calculate_history_quality(inv_variance, pnl_history);

        // Confidence interval based on sample size
        let std_error = decimal_sqrt(inv_variance).unwrap_or(Decimal::ONE)
            / decimal_sqrt(Decimal::from(inventories.len())).unwrap_or(Decimal::ONE);
        let z_score = Decimal::from_str_exact("1.96").unwrap(); // 95% CI
        let margin = gamma * std_error * z_score;

        let low = (gamma - margin).max(Decimal::from_str_exact("0.001").unwrap());
        let high = gamma + margin;

        let mut result =
            CalibrationResult::new(gamma, (low, high), inventory_history.len(), quality);

        if inv_variance > Decimal::from(100) {
            result.add_note("High inventory variance detected");
        }

        Ok(result)
    }

    /// Assesses the quality of a γ value.
    fn assess_gamma_quality(&self, gamma: Decimal) -> Decimal {
        // Optimal γ range is typically 0.1 to 10
        let optimal_low = Decimal::from_str_exact("0.1").unwrap();
        let optimal_high = Decimal::from(10);

        if gamma >= optimal_low && gamma <= optimal_high {
            Decimal::from_str_exact("0.9").unwrap()
        } else if gamma >= Decimal::from_str_exact("0.01").unwrap() && gamma <= Decimal::from(100) {
            Decimal::from_str_exact("0.7").unwrap()
        } else {
            Decimal::from_str_exact("0.4").unwrap()
        }
    }

    /// Calculates quality score from historical data.
    fn calculate_history_quality(
        &self,
        inv_variance: Decimal,
        pnl_history: &[(u64, Decimal)],
    ) -> Decimal {
        let pnls: Vec<Decimal> = pnl_history.iter().map(|(_, pnl)| *pnl).collect();
        let pnl_mean = self.mean(&pnls);

        // Quality based on profitability and inventory control
        let mut quality = Decimal::from_str_exact("0.5").unwrap();

        // Bonus for positive PnL
        if pnl_mean > Decimal::ZERO {
            quality += Decimal::from_str_exact("0.2").unwrap();
        }

        // Bonus for low inventory variance
        if inv_variance < Decimal::from(10) {
            quality += Decimal::from_str_exact("0.2").unwrap();
        }

        // Bonus for sufficient samples
        if pnl_history.len() >= 100 {
            quality += Decimal::from_str_exact("0.1").unwrap();
        }

        quality.min(Decimal::ONE)
    }

    /// Estimates half-life from inventory time series.
    fn estimate_halflife(&self, inventory_history: &[(u64, Decimal)]) -> u64 {
        if inventory_history.len() < 2 {
            return 300_000; // Default 5 minutes
        }

        // Simple estimation: time for inventory to cross zero
        let mut crossings = Vec::new();
        let mut last_sign = inventory_history[0].1.is_sign_positive();

        for (ts, inv) in inventory_history.iter().skip(1) {
            let current_sign = inv.is_sign_positive();
            if current_sign != last_sign {
                crossings.push(*ts);
            }
            last_sign = current_sign;
        }

        if crossings.len() < 2 {
            return 300_000; // Default
        }

        // Average time between crossings approximates half-life
        let total_time = crossings.last().unwrap() - crossings.first().unwrap();
        let avg_crossing_time = total_time / (crossings.len() as u64 - 1);

        avg_crossing_time.max(1000) // At least 1 second
    }

    /// Calculates mean of values.
    fn mean(&self, values: &[Decimal]) -> Decimal {
        if values.is_empty() {
            return Decimal::ZERO;
        }
        let sum: Decimal = values.iter().copied().sum();
        sum / Decimal::from(values.len())
    }

    /// Calculates variance of values.
    fn variance(&self, values: &[Decimal], mean: Decimal) -> Decimal {
        if values.len() < 2 {
            return Decimal::ZERO;
        }
        let sum_sq: Decimal = values.iter().map(|v| (*v - mean) * (*v - mean)).sum();
        sum_sq / Decimal::from(values.len() - 1)
    }
}

impl Default for RiskAversionCalibrator {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

/// Fill observation for order intensity calibration.
///
/// Records a single observation of fill rate at a given spread.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FillObservation {
    /// Spread in basis points.
    pub spread_bps: Decimal,

    /// Fill rate (0 to 1).
    pub fill_rate: Decimal,

    /// Number of observations at this spread.
    pub count: usize,

    /// Timestamp in milliseconds.
    pub timestamp: u64,
}

impl FillObservation {
    /// Creates a new fill observation.
    #[must_use]
    pub fn new(spread_bps: Decimal, fill_rate: Decimal, count: usize, timestamp: u64) -> Self {
        Self {
            spread_bps,
            fill_rate,
            count,
            timestamp,
        }
    }
}

/// Order intensity (k) calibrator.
///
/// Calibrates the order intensity parameter from fill observations
/// using log-linear regression.
///
/// # Mathematical Background
///
/// The fill probability follows: P(fill) = A × exp(-k × spread)
///
/// Taking logs: ln(P) = ln(A) - k × spread
///
/// We estimate k using linear regression on this transformed equation.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::calibration::{
///     CalibrationConfig, FillObservation, OrderIntensityCalibrator,
/// };
/// use market_maker_rs::dec;
///
/// let calibrator = OrderIntensityCalibrator::new(CalibrationConfig::default());
///
/// let observations = vec![
///     FillObservation::new(dec!(5.0), dec!(0.8), 100, 1000),
///     FillObservation::new(dec!(10.0), dec!(0.5), 100, 2000),
///     FillObservation::new(dec!(15.0), dec!(0.3), 100, 3000),
/// ];
///
/// // Note: This will fail with default min_samples=30
/// // Use CalibrationConfig::quick() for fewer samples
/// ```
#[derive(Debug, Clone)]
pub struct OrderIntensityCalibrator {
    config: CalibrationConfig,
}

impl OrderIntensityCalibrator {
    /// Creates a new order intensity calibrator.
    #[must_use]
    pub fn new(config: CalibrationConfig) -> Self {
        Self { config }
    }

    /// Calibrates k from fill observations.
    ///
    /// Uses regression on: ln(fill_rate) = ln(A) - k × spread
    ///
    /// # Arguments
    ///
    /// * `fill_observations` - Slice of fill observations at different spreads
    ///
    /// # Returns
    ///
    /// Calibration result with estimated k value.
    pub fn calibrate_from_fills(
        &self,
        fill_observations: &[FillObservation],
    ) -> MMResult<CalibrationResult<Decimal>> {
        // Filter valid observations (fill_rate > 0)
        let valid_obs: Vec<&FillObservation> = fill_observations
            .iter()
            .filter(|o| o.fill_rate > Decimal::ZERO && o.fill_rate <= Decimal::ONE)
            .collect();

        let total_count: usize = valid_obs.iter().map(|o| o.count).sum();

        if total_count < self.config.min_samples {
            return Err(MMError::InvalidConfiguration(format!(
                "Insufficient fill observations: {} < {}",
                total_count, self.config.min_samples
            )));
        }

        if valid_obs.len() < 2 {
            return Err(MMError::InvalidConfiguration(
                "Need at least 2 different spread levels".to_string(),
            ));
        }

        // Prepare data for regression: x = spread, y = ln(fill_rate)
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        let mut weights = Vec::new();

        for obs in &valid_obs {
            x_values.push(obs.spread_bps);
            y_values.push(decimal_ln(obs.fill_rate).unwrap_or(Decimal::ZERO));
            weights.push(Decimal::from(obs.count));
        }

        // Weighted linear regression
        let (slope, _intercept, r_squared) =
            self.weighted_linear_regression(&x_values, &y_values, &weights);

        // k is the negative of the slope
        let k = -slope;

        // Ensure k is positive
        let k = k.max(Decimal::from_str_exact("0.001").unwrap());

        // Confidence interval based on R² and sample size
        let std_error = self.estimate_std_error(&x_values, &y_values, slope, _intercept);
        let z_score = Decimal::from_str_exact("1.96").unwrap();
        let margin = std_error * z_score;

        let low = (k - margin).max(Decimal::from_str_exact("0.001").unwrap());
        let high = k + margin;

        // Quality based on R² and sample size
        let quality = self.calculate_quality(r_squared, total_count);

        let mut result = CalibrationResult::new(k, (low, high), total_count, quality);

        if r_squared < Decimal::from_str_exact("0.5").unwrap() {
            result.add_note("Low R² - relationship may not be log-linear");
        }

        if k > Decimal::ONE {
            result.add_note("High k value - spreads may be too wide");
        }

        Ok(result)
    }

    /// Performs weighted linear regression.
    ///
    /// Returns (slope, intercept, r_squared).
    fn weighted_linear_regression(
        &self,
        x: &[Decimal],
        y: &[Decimal],
        weights: &[Decimal],
    ) -> (Decimal, Decimal, Decimal) {
        let n = x.len();
        if n < 2 {
            return (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
        }

        let total_weight: Decimal = weights.iter().copied().sum();

        // Weighted means
        let x_mean: Decimal = x
            .iter()
            .zip(weights.iter())
            .map(|(xi, wi)| *xi * *wi)
            .sum::<Decimal>()
            / total_weight;

        let y_mean: Decimal = y
            .iter()
            .zip(weights.iter())
            .map(|(yi, wi)| *yi * *wi)
            .sum::<Decimal>()
            / total_weight;

        // Weighted covariance and variance
        let mut cov_xy = Decimal::ZERO;
        let mut var_x = Decimal::ZERO;
        let mut var_y = Decimal::ZERO;

        for i in 0..n {
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            cov_xy += weights[i] * dx * dy;
            var_x += weights[i] * dx * dx;
            var_y += weights[i] * dy * dy;
        }

        if var_x.is_zero() {
            return (Decimal::ZERO, y_mean, Decimal::ZERO);
        }

        let slope = cov_xy / var_x;
        let intercept = y_mean - slope * x_mean;

        // R² calculation
        let r_squared = if var_y.is_zero() {
            Decimal::ONE
        } else {
            (cov_xy * cov_xy) / (var_x * var_y)
        };

        (slope, intercept, r_squared)
    }

    /// Estimates standard error of the slope.
    fn estimate_std_error(
        &self,
        x: &[Decimal],
        y: &[Decimal],
        slope: Decimal,
        intercept: Decimal,
    ) -> Decimal {
        let n = x.len();
        if n < 3 {
            return Decimal::from_str_exact("0.1").unwrap();
        }

        // Calculate residual sum of squares
        let mut rss = Decimal::ZERO;
        for i in 0..n {
            let predicted = intercept + slope * x[i];
            let residual = y[i] - predicted;
            rss += residual * residual;
        }

        // Calculate x variance
        let x_mean: Decimal = x.iter().copied().sum::<Decimal>() / Decimal::from(n);
        let x_var: Decimal = x.iter().map(|xi| (*xi - x_mean) * (*xi - x_mean)).sum();

        if x_var.is_zero() {
            return Decimal::from_str_exact("0.1").unwrap();
        }

        // Standard error of slope
        let mse = rss / Decimal::from(n - 2);
        decimal_sqrt(mse / x_var).unwrap_or(Decimal::from_str_exact("0.1").unwrap())
    }

    /// Calculates quality score.
    fn calculate_quality(&self, r_squared: Decimal, sample_size: usize) -> Decimal {
        let mut quality = r_squared * Decimal::from_str_exact("0.6").unwrap();

        // Bonus for sample size
        if sample_size >= 100 {
            quality += Decimal::from_str_exact("0.3").unwrap();
        } else if sample_size >= 50 {
            quality += Decimal::from_str_exact("0.2").unwrap();
        } else if sample_size >= 30 {
            quality += Decimal::from_str_exact("0.1").unwrap();
        }

        quality.min(Decimal::ONE)
    }
}

impl Default for OrderIntensityCalibrator {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

/// Volatility regime classification.
///
/// Represents different market volatility states that may require
/// parameter adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VolatilityRegime {
    /// Low volatility (< 0.5× normal).
    Low,
    /// Normal volatility (0.5× to 1.5× normal).
    Normal,
    /// High volatility (1.5× to 2.5× normal).
    High,
    /// Extreme volatility (> 2.5× normal).
    Extreme,
}

impl fmt::Display for VolatilityRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Extreme => write!(f, "Extreme"),
        }
    }
}

impl VolatilityRegime {
    /// Returns all regime variants.
    #[must_use]
    pub fn all() -> &'static [VolatilityRegime] {
        &[Self::Low, Self::Normal, Self::High, Self::Extreme]
    }

    /// Returns true if this is a high-risk regime.
    #[must_use]
    pub fn is_high_risk(&self) -> bool {
        matches!(self, Self::High | Self::Extreme)
    }
}

/// Parameter adjustments for a volatility regime.
///
/// Contains multipliers to apply to base parameters when operating
/// in different volatility conditions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeAdjustments {
    /// Multiplier for risk aversion (γ).
    pub gamma_multiplier: Decimal,

    /// Multiplier for minimum spread.
    pub spread_multiplier: Decimal,

    /// Multiplier for position limits.
    pub position_limit_multiplier: Decimal,
}

impl RegimeAdjustments {
    /// Creates new regime adjustments.
    #[must_use]
    pub fn new(
        gamma_multiplier: Decimal,
        spread_multiplier: Decimal,
        position_limit_multiplier: Decimal,
    ) -> Self {
        Self {
            gamma_multiplier,
            spread_multiplier,
            position_limit_multiplier,
        }
    }

    /// Returns adjustments for no change (all multipliers = 1).
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            gamma_multiplier: Decimal::ONE,
            spread_multiplier: Decimal::ONE,
            position_limit_multiplier: Decimal::ONE,
        }
    }
}

impl Default for RegimeAdjustments {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Volatility regime detector.
///
/// Detects and classifies the current volatility regime by comparing
/// current volatility to historical baseline.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::calibration::{VolatilityRegime, VolatilityRegimeDetector};
/// use market_maker_rs::dec;
///
/// let detector = VolatilityRegimeDetector::new(dec!(1.5), 3600_000);
///
/// // Current vol is 1.8× historical
/// let regime = detector.detect_regime(dec!(0.036), dec!(0.02));
/// assert_eq!(regime, VolatilityRegime::High);
///
/// // Get recommended adjustments
/// let adjustments = detector.regime_adjustments(regime);
/// println!("Spread multiplier: {}", adjustments.spread_multiplier);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityRegimeDetector {
    /// Threshold for regime change detection.
    regime_threshold: Decimal,

    /// Lookback window in milliseconds.
    lookback_ms: u64,
}

impl VolatilityRegimeDetector {
    /// Creates a new volatility regime detector.
    ///
    /// # Arguments
    ///
    /// * `regime_threshold` - Multiplier threshold for regime changes (e.g., 1.5)
    /// * `lookback_ms` - Lookback window for historical volatility
    #[must_use]
    pub fn new(regime_threshold: Decimal, lookback_ms: u64) -> Self {
        Self {
            regime_threshold,
            lookback_ms,
        }
    }

    /// Returns the lookback window in milliseconds.
    #[must_use]
    pub fn lookback_ms(&self) -> u64 {
        self.lookback_ms
    }

    /// Detects the current volatility regime.
    ///
    /// # Arguments
    ///
    /// * `current_volatility` - Current observed volatility
    /// * `historical_volatility` - Baseline historical volatility
    ///
    /// # Returns
    ///
    /// The detected volatility regime.
    #[must_use]
    pub fn detect_regime(
        &self,
        current_volatility: Decimal,
        historical_volatility: Decimal,
    ) -> VolatilityRegime {
        if historical_volatility.is_zero() {
            return VolatilityRegime::Normal;
        }

        let ratio = current_volatility / historical_volatility;

        // Thresholds: 0.5, 1.5, 2.5 (relative to regime_threshold)
        let low_threshold = Decimal::ONE / self.regime_threshold;
        let high_threshold = self.regime_threshold;
        let extreme_threshold = self.regime_threshold + (self.regime_threshold - Decimal::ONE);

        if ratio < low_threshold {
            VolatilityRegime::Low
        } else if ratio <= high_threshold {
            VolatilityRegime::Normal
        } else if ratio <= extreme_threshold {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }

    /// Returns recommended parameter adjustments for a regime.
    ///
    /// # Arguments
    ///
    /// * `regime` - The volatility regime
    ///
    /// # Returns
    ///
    /// Recommended parameter multipliers.
    #[must_use]
    pub fn regime_adjustments(&self, regime: VolatilityRegime) -> RegimeAdjustments {
        match regime {
            VolatilityRegime::Low => RegimeAdjustments::new(
                Decimal::from_str_exact("0.7").unwrap(), // Less aggressive
                Decimal::from_str_exact("0.8").unwrap(), // Tighter spreads
                Decimal::from_str_exact("1.2").unwrap(), // Larger positions OK
            ),
            VolatilityRegime::Normal => RegimeAdjustments::neutral(),
            VolatilityRegime::High => RegimeAdjustments::new(
                Decimal::from_str_exact("1.5").unwrap(), // More aggressive inventory control
                Decimal::from_str_exact("1.5").unwrap(), // Wider spreads
                Decimal::from_str_exact("0.7").unwrap(), // Smaller positions
            ),
            VolatilityRegime::Extreme => RegimeAdjustments::new(
                Decimal::from_str_exact("2.0").unwrap(), // Very aggressive
                Decimal::from_str_exact("2.5").unwrap(), // Much wider spreads
                Decimal::from_str_exact("0.3").unwrap(), // Minimal positions
            ),
        }
    }

    /// Detects regime and returns adjustments in one call.
    #[must_use]
    pub fn detect_and_adjust(
        &self,
        current_volatility: Decimal,
        historical_volatility: Decimal,
    ) -> (VolatilityRegime, RegimeAdjustments) {
        let regime = self.detect_regime(current_volatility, historical_volatility);
        let adjustments = self.regime_adjustments(regime);
        (regime, adjustments)
    }
}

impl Default for VolatilityRegimeDetector {
    fn default() -> Self {
        Self::new(Decimal::from_str_exact("1.5").unwrap(), 3_600_000)
    }
}

/// Optimized parameters result.
///
/// Contains the results of a full parameter optimization run.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizedParameters {
    /// Calibrated risk aversion.
    pub risk_aversion: CalibrationResult<Decimal>,

    /// Calibrated order intensity (if available).
    pub order_intensity: Option<CalibrationResult<Decimal>>,

    /// Detected volatility regime.
    pub regime: VolatilityRegime,

    /// Recommended parameter adjustments.
    pub adjustments: RegimeAdjustments,
}

impl OptimizedParameters {
    /// Returns the adjusted risk aversion value.
    #[must_use]
    pub fn adjusted_gamma(&self) -> Decimal {
        self.risk_aversion.value * self.adjustments.gamma_multiplier
    }

    /// Returns the adjusted order intensity value (if available).
    #[must_use]
    pub fn adjusted_k(&self) -> Option<Decimal> {
        self.order_intensity.as_ref().map(|k| k.value)
    }

    /// Returns true if all calibrations are acceptable quality.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        let gamma_ok = self.risk_aversion.is_acceptable();
        let k_ok = self
            .order_intensity
            .as_ref()
            .is_none_or(|k| k.is_acceptable());
        gamma_ok && k_ok
    }
}

/// Combined parameter optimizer.
///
/// Combines all calibrators to perform full parameter optimization.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::calibration::{CalibrationConfig, ParameterOptimizer};
/// use market_maker_rs::dec;
///
/// let optimizer = ParameterOptimizer::new(CalibrationConfig::default());
///
/// // Quick optimization from basic inputs
/// let result = optimizer.optimize_quick(
///     dec!(0.02),    // current volatility
///     dec!(0.015),   // historical volatility
///     300_000,       // desired half-life (5 min)
/// );
///
/// println!("Regime: {}", result.regime);
/// println!("Adjusted γ: {}", result.adjusted_gamma());
/// ```
#[derive(Debug, Clone)]
pub struct ParameterOptimizer {
    risk_aversion_calibrator: RiskAversionCalibrator,
    order_intensity_calibrator: OrderIntensityCalibrator,
    regime_detector: VolatilityRegimeDetector,
}

impl ParameterOptimizer {
    /// Creates a new parameter optimizer.
    #[must_use]
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            risk_aversion_calibrator: RiskAversionCalibrator::new(config.clone()),
            order_intensity_calibrator: OrderIntensityCalibrator::new(config),
            regime_detector: VolatilityRegimeDetector::default(),
        }
    }

    /// Creates an optimizer with custom regime detector.
    #[must_use]
    pub fn with_regime_detector(
        config: CalibrationConfig,
        regime_detector: VolatilityRegimeDetector,
    ) -> Self {
        Self {
            risk_aversion_calibrator: RiskAversionCalibrator::new(config.clone()),
            order_intensity_calibrator: OrderIntensityCalibrator::new(config),
            regime_detector,
        }
    }

    /// Performs quick optimization from basic inputs.
    ///
    /// # Arguments
    ///
    /// * `current_volatility` - Current observed volatility
    /// * `historical_volatility` - Baseline historical volatility
    /// * `desired_halflife_ms` - Desired inventory half-life
    ///
    /// # Returns
    ///
    /// Optimized parameters.
    #[must_use]
    pub fn optimize_quick(
        &self,
        current_volatility: Decimal,
        historical_volatility: Decimal,
        desired_halflife_ms: u64,
    ) -> OptimizedParameters {
        // Detect regime
        let (regime, adjustments) = self
            .regime_detector
            .detect_and_adjust(current_volatility, historical_volatility);

        // Calibrate risk aversion
        let risk_aversion = self
            .risk_aversion_calibrator
            .calibrate_from_halflife(desired_halflife_ms, current_volatility);

        OptimizedParameters {
            risk_aversion,
            order_intensity: None,
            regime,
            adjustments,
        }
    }

    /// Performs full optimization with fill data.
    ///
    /// # Arguments
    ///
    /// * `current_volatility` - Current observed volatility
    /// * `historical_volatility` - Baseline historical volatility
    /// * `desired_halflife_ms` - Desired inventory half-life
    /// * `fill_observations` - Historical fill observations
    ///
    /// # Returns
    ///
    /// Optimized parameters or error.
    pub fn optimize_full(
        &self,
        current_volatility: Decimal,
        historical_volatility: Decimal,
        desired_halflife_ms: u64,
        fill_observations: &[FillObservation],
    ) -> MMResult<OptimizedParameters> {
        // Detect regime
        let (regime, adjustments) = self
            .regime_detector
            .detect_and_adjust(current_volatility, historical_volatility);

        // Calibrate risk aversion
        let risk_aversion = self
            .risk_aversion_calibrator
            .calibrate_from_halflife(desired_halflife_ms, current_volatility);

        // Calibrate order intensity
        let order_intensity = self
            .order_intensity_calibrator
            .calibrate_from_fills(fill_observations)?;

        Ok(OptimizedParameters {
            risk_aversion,
            order_intensity: Some(order_intensity),
            regime,
            adjustments,
        })
    }

    /// Returns a reference to the risk aversion calibrator.
    #[must_use]
    pub fn risk_aversion_calibrator(&self) -> &RiskAversionCalibrator {
        &self.risk_aversion_calibrator
    }

    /// Returns a reference to the order intensity calibrator.
    #[must_use]
    pub fn order_intensity_calibrator(&self) -> &OrderIntensityCalibrator {
        &self.order_intensity_calibrator
    }

    /// Returns a reference to the regime detector.
    #[must_use]
    pub fn regime_detector(&self) -> &VolatilityRegimeDetector {
        &self.regime_detector
    }
}

impl Default for ParameterOptimizer {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    // CalibrationConfig tests
    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.confidence_level, dec!(0.95));
        assert!(config.robust_estimation);
    }

    #[test]
    fn test_calibration_config_quick() {
        let config = CalibrationConfig::quick();
        assert_eq!(config.min_samples, 10);
        assert!(!config.robust_estimation);
    }

    #[test]
    fn test_calibration_config_precise() {
        let config = CalibrationConfig::precise();
        assert_eq!(config.min_samples, 100);
        assert_eq!(config.confidence_level, dec!(0.99));
    }

    // CalibrationResult tests
    #[test]
    fn test_calibration_result_new() {
        let result = CalibrationResult::new(dec!(1.5), (dec!(1.0), dec!(2.0)), 50, dec!(0.8));

        assert_eq!(result.value, dec!(1.5));
        assert_eq!(result.confidence_interval, (dec!(1.0), dec!(2.0)));
        assert_eq!(result.sample_size, 50);
        assert_eq!(result.quality, dec!(0.8));
        assert!(result.notes.is_empty());
    }

    #[test]
    fn test_calibration_result_with_note() {
        let result = CalibrationResult::new(dec!(1.5), (dec!(1.0), dec!(2.0)), 50, dec!(0.8))
            .with_note("Test note");

        assert_eq!(result.notes.len(), 1);
        assert_eq!(result.notes[0], "Test note");
    }

    #[test]
    fn test_calibration_result_quality_checks() {
        let poor = CalibrationResult::new(dec!(1.0), (dec!(0.5), dec!(1.5)), 10, dec!(0.3));
        assert!(!poor.is_acceptable());
        assert!(!poor.is_good());
        assert!(!poor.is_excellent());

        let acceptable = CalibrationResult::new(dec!(1.0), (dec!(0.5), dec!(1.5)), 30, dec!(0.6));
        assert!(acceptable.is_acceptable());
        assert!(!acceptable.is_good());

        let good = CalibrationResult::new(dec!(1.0), (dec!(0.5), dec!(1.5)), 50, dec!(0.75));
        assert!(good.is_acceptable());
        assert!(good.is_good());
        assert!(!good.is_excellent());

        let excellent = CalibrationResult::new(dec!(1.0), (dec!(0.5), dec!(1.5)), 100, dec!(0.95));
        assert!(excellent.is_acceptable());
        assert!(excellent.is_good());
        assert!(excellent.is_excellent());
    }

    // RiskAversionCalibrator tests
    #[test]
    fn test_risk_aversion_from_halflife() {
        let calibrator = RiskAversionCalibrator::default();

        // 5-minute half-life with 2% volatility
        let result = calibrator.calibrate_from_halflife(300_000, dec!(0.02));

        // γ = ln(2) / (300 * 0.02²) = 0.693 / (300 * 0.0004) = 0.693 / 0.12 ≈ 5.78
        assert!(result.value > dec!(5.0));
        assert!(result.value < dec!(7.0));
        assert!(result.is_acceptable());
    }

    #[test]
    fn test_risk_aversion_zero_volatility() {
        let calibrator = RiskAversionCalibrator::default();
        let result = calibrator.calibrate_from_halflife(300_000, dec!(0.0));

        // Should return default value
        assert_eq!(result.value, Decimal::ONE);
    }

    #[test]
    fn test_risk_aversion_from_history() {
        let calibrator = RiskAversionCalibrator::new(CalibrationConfig::quick());

        // Create synthetic inventory history with oscillations
        let mut inventory_history = Vec::new();
        for i in 0..50 {
            let t = i as u64 * 10_000; // 10 second intervals
            let inv = if i % 10 < 5 { dec!(10.0) } else { dec!(-10.0) };
            inventory_history.push((t, inv));
        }

        let pnl_history: Vec<(u64, Decimal)> = (0..50)
            .map(|i| (i as u64 * 10_000, dec!(100.0) + Decimal::from(i)))
            .collect();

        let result = calibrator
            .calibrate_from_history(&inventory_history, &pnl_history, dec!(0.02))
            .unwrap();

        assert!(result.value > Decimal::ZERO);
    }

    #[test]
    fn test_risk_aversion_insufficient_data() {
        let calibrator = RiskAversionCalibrator::default();

        let inventory_history = vec![(1000u64, dec!(10.0))];
        let pnl_history = vec![(1000u64, dec!(100.0))];

        let result =
            calibrator.calibrate_from_history(&inventory_history, &pnl_history, dec!(0.02));

        assert!(result.is_err());
    }

    // OrderIntensityCalibrator tests
    #[test]
    fn test_order_intensity_from_fills() {
        let calibrator = OrderIntensityCalibrator::new(CalibrationConfig::quick());

        // Create observations with exponential decay
        let observations = vec![
            FillObservation::new(dec!(5.0), dec!(0.8), 20, 1000),
            FillObservation::new(dec!(10.0), dec!(0.5), 20, 2000),
            FillObservation::new(dec!(15.0), dec!(0.3), 20, 3000),
            FillObservation::new(dec!(20.0), dec!(0.15), 20, 4000),
        ];

        let result = calibrator.calibrate_from_fills(&observations).unwrap();

        // k should be positive
        assert!(result.value > Decimal::ZERO);
    }

    #[test]
    fn test_order_intensity_insufficient_data() {
        let calibrator = OrderIntensityCalibrator::default();

        let observations = vec![FillObservation::new(dec!(5.0), dec!(0.8), 5, 1000)];

        let result = calibrator.calibrate_from_fills(&observations);
        assert!(result.is_err());
    }

    // VolatilityRegime tests
    #[test]
    fn test_volatility_regime_display() {
        assert_eq!(VolatilityRegime::Low.to_string(), "Low");
        assert_eq!(VolatilityRegime::Normal.to_string(), "Normal");
        assert_eq!(VolatilityRegime::High.to_string(), "High");
        assert_eq!(VolatilityRegime::Extreme.to_string(), "Extreme");
    }

    #[test]
    fn test_volatility_regime_is_high_risk() {
        assert!(!VolatilityRegime::Low.is_high_risk());
        assert!(!VolatilityRegime::Normal.is_high_risk());
        assert!(VolatilityRegime::High.is_high_risk());
        assert!(VolatilityRegime::Extreme.is_high_risk());
    }

    // VolatilityRegimeDetector tests
    #[test]
    fn test_regime_detector_low() {
        let detector = VolatilityRegimeDetector::default();
        let regime = detector.detect_regime(dec!(0.005), dec!(0.02));
        assert_eq!(regime, VolatilityRegime::Low);
    }

    #[test]
    fn test_regime_detector_normal() {
        let detector = VolatilityRegimeDetector::default();
        let regime = detector.detect_regime(dec!(0.02), dec!(0.02));
        assert_eq!(regime, VolatilityRegime::Normal);
    }

    #[test]
    fn test_regime_detector_high() {
        let detector = VolatilityRegimeDetector::default();
        let regime = detector.detect_regime(dec!(0.035), dec!(0.02));
        assert_eq!(regime, VolatilityRegime::High);
    }

    #[test]
    fn test_regime_detector_extreme() {
        let detector = VolatilityRegimeDetector::default();
        let regime = detector.detect_regime(dec!(0.06), dec!(0.02));
        assert_eq!(regime, VolatilityRegime::Extreme);
    }

    #[test]
    fn test_regime_detector_zero_historical() {
        let detector = VolatilityRegimeDetector::default();
        let regime = detector.detect_regime(dec!(0.02), dec!(0.0));
        assert_eq!(regime, VolatilityRegime::Normal);
    }

    #[test]
    fn test_regime_adjustments() {
        let detector = VolatilityRegimeDetector::default();

        let low_adj = detector.regime_adjustments(VolatilityRegime::Low);
        assert!(low_adj.gamma_multiplier < Decimal::ONE);
        assert!(low_adj.spread_multiplier < Decimal::ONE);
        assert!(low_adj.position_limit_multiplier > Decimal::ONE);

        let normal_adj = detector.regime_adjustments(VolatilityRegime::Normal);
        assert_eq!(normal_adj.gamma_multiplier, Decimal::ONE);

        let high_adj = detector.regime_adjustments(VolatilityRegime::High);
        assert!(high_adj.gamma_multiplier > Decimal::ONE);
        assert!(high_adj.spread_multiplier > Decimal::ONE);
        assert!(high_adj.position_limit_multiplier < Decimal::ONE);

        let extreme_adj = detector.regime_adjustments(VolatilityRegime::Extreme);
        assert!(extreme_adj.gamma_multiplier > high_adj.gamma_multiplier);
    }

    // ParameterOptimizer tests
    #[test]
    fn test_optimizer_quick() {
        let optimizer = ParameterOptimizer::default();

        let result = optimizer.optimize_quick(dec!(0.02), dec!(0.015), 300_000);

        assert!(result.risk_aversion.value > Decimal::ZERO);
        assert!(result.order_intensity.is_none());
        assert_eq!(result.regime, VolatilityRegime::Normal);
    }

    #[test]
    fn test_optimizer_quick_high_vol() {
        let optimizer = ParameterOptimizer::default();

        let result = optimizer.optimize_quick(dec!(0.04), dec!(0.02), 300_000);

        assert_eq!(result.regime, VolatilityRegime::High);
        assert!(result.adjustments.gamma_multiplier > Decimal::ONE);
    }

    #[test]
    fn test_optimizer_full() {
        let optimizer = ParameterOptimizer::new(CalibrationConfig::quick());

        let observations = vec![
            FillObservation::new(dec!(5.0), dec!(0.8), 20, 1000),
            FillObservation::new(dec!(10.0), dec!(0.5), 20, 2000),
            FillObservation::new(dec!(15.0), dec!(0.3), 20, 3000),
        ];

        let result = optimizer
            .optimize_full(dec!(0.02), dec!(0.02), 300_000, &observations)
            .unwrap();

        assert!(result.risk_aversion.value > Decimal::ZERO);
        assert!(result.order_intensity.is_some());
        assert!(result.order_intensity.unwrap().value > Decimal::ZERO);
    }

    #[test]
    fn test_optimized_parameters_adjusted_gamma() {
        let optimizer = ParameterOptimizer::default();

        // High volatility regime
        let result = optimizer.optimize_quick(dec!(0.04), dec!(0.02), 300_000);

        let base_gamma = result.risk_aversion.value;
        let adjusted_gamma = result.adjusted_gamma();

        // Adjusted should be higher due to high vol regime
        assert!(adjusted_gamma > base_gamma);
    }

    #[test]
    fn test_optimized_parameters_is_acceptable() {
        let optimizer = ParameterOptimizer::default();
        let result = optimizer.optimize_quick(dec!(0.02), dec!(0.02), 300_000);

        assert!(result.is_acceptable());
    }

    // RegimeAdjustments tests
    #[test]
    fn test_regime_adjustments_neutral() {
        let adj = RegimeAdjustments::neutral();
        assert_eq!(adj.gamma_multiplier, Decimal::ONE);
        assert_eq!(adj.spread_multiplier, Decimal::ONE);
        assert_eq!(adj.position_limit_multiplier, Decimal::ONE);
    }

    #[test]
    fn test_regime_adjustments_default() {
        let adj = RegimeAdjustments::default();
        assert_eq!(adj.gamma_multiplier, Decimal::ONE);
    }

    // FillObservation tests
    #[test]
    fn test_fill_observation_new() {
        let obs = FillObservation::new(dec!(10.0), dec!(0.5), 100, 1000);
        assert_eq!(obs.spread_bps, dec!(10.0));
        assert_eq!(obs.fill_rate, dec!(0.5));
        assert_eq!(obs.count, 100);
        assert_eq!(obs.timestamp, 1000);
    }
}
