//! Volatility estimation methods.
//!
//! This module provides tools for estimating market volatility from historical price data.
//! Volatility is a crucial input for the Avellaneda-Stoikov model as it determines the
//! optimal spread width.
//!
//! # Methods Implemented
//!
//! - **Simple Standard Deviation**: Traditional statistical volatility
//! - **Exponentially Weighted Moving Average (EWMA)**: Gives more weight to recent observations
//! - **Parkinson's Range-Based**: Uses high-low price range (more efficient)
//!
//! # Examples
//!
//! ```
//! use market_maker_rs::market_state::volatility::VolatilityEstimator;
//! use market_maker_rs::dec;
//!
//! let prices = vec![dec!(100.0), dec!(101.0), dec!(99.5), dec!(100.5), dec!(102.0)];
//! let estimator = VolatilityEstimator::new();
//!
//! let volatility = estimator.calculate_simple(&prices).unwrap();
//! assert!(volatility > dec!(0.0));
//! ```

use crate::Decimal;
use crate::types::decimal::{OrOverflow, decimal_ln, decimal_sqrt};
use crate::types::error::{MMError, MMResult};

/// Volatility estimator with multiple calculation methods.
///
/// This struct provides various methods to estimate volatility from price data.
/// All volatility values are annualized by default (assuming 252 trading days per year).
///
/// # Examples
///
/// ```
/// use market_maker_rs::market_state::volatility::VolatilityEstimator;
/// use market_maker_rs::dec;
///
/// let estimator = VolatilityEstimator::new();
/// let prices = vec![dec!(100.0), dec!(101.0), dec!(102.0), dec!(101.5)];
///
/// // Calculate using different methods
/// let simple_vol = estimator.calculate_simple(&prices).unwrap();
/// let ewma_vol = estimator.calculate_ewma(&prices, dec!(0.94)).unwrap();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct VolatilityEstimator {
    annualization_factor: Option<Decimal>,
}

impl VolatilityEstimator {
    /// Creates a new volatility estimator with default settings.
    ///
    /// Uses 252 trading days per year for annualization.
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::market_state::volatility::VolatilityEstimator;
    ///
    /// let estimator = VolatilityEstimator::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            annualization_factor: None, // Will use sqrt(252) by default
        }
    }

    /// Creates a new volatility estimator with custom annualization factor.
    ///
    /// # Arguments
    ///
    /// * `annualization_factor` - Custom factor for annualizing volatility
    ///   (e.g., sqrt(252) for daily returns, sqrt(365*24) for hourly returns)
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::market_state::volatility::VolatilityEstimator;
    /// use market_maker_rs::dec;
    ///
    /// // For hourly data
    /// let estimator = VolatilityEstimator::with_annualization_factor(dec!(87.178));
    /// ```
    #[must_use]
    pub fn with_annualization_factor(factor: Decimal) -> Self {
        Self {
            annualization_factor: Some(factor),
        }
    }

    /// Gets the annualization factor.
    ///
    /// Returns sqrt(252) if no custom factor was set.
    fn get_annualization_factor(&self) -> MMResult<Decimal> {
        if let Some(factor) = self.annualization_factor {
            Ok(factor)
        } else {
            // Default: sqrt(252) for daily returns
            decimal_sqrt(Decimal::from(252))
        }
    }

    /// Calculates simple historical volatility using standard deviation of log returns.
    ///
    /// This is the traditional method: σ = sqrt(Σ(r_i - mean(r))² / (n-1))
    /// where r_i = ln(P_i / P_{i-1}) are log returns.
    ///
    /// # Arguments
    ///
    /// * `prices` - Vector of historical prices (chronologically ordered)
    ///
    /// # Returns
    ///
    /// Annualized volatility as a decimal (e.g., 0.2 for 20% volatility).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Less than 3 prices provided (the sample variance needs 2 returns)
    /// - Any price is zero or negative
    /// - Numerical calculation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::market_state::volatility::VolatilityEstimator;
    /// use market_maker_rs::dec;
    ///
    /// let estimator = VolatilityEstimator::new();
    /// let prices = vec![dec!(100.0), dec!(102.0), dec!(101.0), dec!(103.0)];
    ///
    /// let volatility = estimator.calculate_simple(&prices).unwrap();
    /// assert!(volatility > dec!(0.0));
    /// ```
    pub fn calculate_simple(&self, prices: &[Decimal]) -> MMResult<Decimal> {
        if prices.len() < 2 {
            return Err(MMError::InvalidMarketState(
                "need at least 2 prices to calculate volatility".to_string(),
            ));
        }

        // Calculate log returns
        let log_returns = log_returns(prices)?;

        self.simple_of_returns(&log_returns)
    }

    /// Annualized sample standard deviation of log returns already computed.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] with fewer than 2 returns or on
    /// arithmetic failure.
    pub(crate) fn simple_of_returns(&self, log_returns: &[Decimal]) -> MMResult<Decimal> {
        // Sample variance of the log returns
        let variance = sample_variance(log_returns)?;

        // Calculate standard deviation
        let std_dev = decimal_sqrt(variance)?;

        // Annualize
        let annualization_factor = self.get_annualization_factor()?;
        std_dev
            .checked_mul(annualization_factor)
            .or_overflow("annualized volatility")
    }

    /// Calculates volatility using Exponentially Weighted Moving Average (EWMA).
    ///
    /// This method gives more weight to recent observations:
    /// σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t
    ///
    /// # Arguments
    ///
    /// * `prices` - Vector of historical prices (chronologically ordered)
    /// * `lambda` - Decay factor (typical values: 0.94 for daily, 0.97 for monthly data)
    ///
    /// # Returns
    ///
    /// Annualized volatility.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Less than 3 prices provided (the sample variance needs 2 returns)
    /// - Lambda not in range (0, 1)
    /// - Any price is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::market_state::volatility::VolatilityEstimator;
    /// use market_maker_rs::dec;
    ///
    /// let estimator = VolatilityEstimator::new();
    /// let prices = vec![dec!(100.0), dec!(101.5), dec!(99.5), dec!(102.0)];
    ///
    /// // RiskMetrics standard lambda for daily data
    /// let volatility = estimator.calculate_ewma(&prices, dec!(0.94)).unwrap();
    /// assert!(volatility > dec!(0.0));
    /// ```
    pub fn calculate_ewma(&self, prices: &[Decimal], lambda: Decimal) -> MMResult<Decimal> {
        if prices.len() < 2 {
            return Err(MMError::InvalidMarketState(
                "need at least 2 prices for EWMA".to_string(),
            ));
        }

        if lambda <= Decimal::ZERO || lambda >= Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "lambda must be between 0 and 1".to_string(),
            ));
        }

        // Calculate log returns
        let log_returns = log_returns(prices)?;

        self.ewma_of_returns(&log_returns, lambda)
    }

    /// Annualized EWMA of log returns already computed, in chronological order.
    /// `lambda` is in `(0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] with fewer than 2 returns or on
    /// arithmetic failure.
    pub(crate) fn ewma_of_returns(
        &self,
        log_returns: &[Decimal],
        lambda: Decimal,
    ) -> MMResult<Decimal> {
        // Initialize variance with simple variance of first few returns
        let initial_variance = if log_returns.len() >= 5 {
            sample_variance(log_returns.get(..5).unwrap_or_default())?
        } else {
            sample_variance(log_returns)?
        };

        // EWMA recursion. `0 < lambda < 1`, so `1 - lambda` cannot overflow.
        let one_minus_lambda = Decimal::ONE.saturating_sub(lambda);
        let mut variance = initial_variance;

        for &log_return in log_returns {
            variance = log_return
                .checked_mul(log_return)
                .and_then(|sq| one_minus_lambda.checked_mul(sq))
                .and_then(|new| lambda.checked_mul(variance)?.checked_add(new))
                .or_overflow("ewma variance")?;
        }

        // Calculate standard deviation and annualize
        let std_dev = decimal_sqrt(variance)?;
        let annualization_factor = self.get_annualization_factor()?;
        std_dev
            .checked_mul(annualization_factor)
            .or_overflow("annualized volatility")
    }

    /// Calculates volatility using Parkinson's range-based estimator.
    ///
    /// This method is more efficient than close-to-close estimators as it uses
    /// high-low price range: σ = sqrt(Σ(ln(H_i/L_i))² / (4*n*ln(2)))
    ///
    /// # Arguments
    ///
    /// * `high_prices` - Vector of high prices for each period
    /// * `low_prices` - Vector of low prices for each period
    ///
    /// # Returns
    ///
    /// Annualized volatility.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Vectors have different lengths
    /// - Less than 1 data point
    /// - Any high < low or prices invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::market_state::volatility::VolatilityEstimator;
    /// use market_maker_rs::dec;
    ///
    /// let estimator = VolatilityEstimator::new();
    /// let highs = vec![dec!(102.0), dec!(103.5), dec!(101.0)];
    /// let lows = vec![dec!(99.0), dec!(100.5), dec!(98.0)];
    ///
    /// let volatility = estimator.calculate_parkinson(&highs, &lows).unwrap();
    /// assert!(volatility > dec!(0.0));
    /// ```
    pub fn calculate_parkinson(
        &self,
        high_prices: &[Decimal],
        low_prices: &[Decimal],
    ) -> MMResult<Decimal> {
        if high_prices.len() != low_prices.len() {
            return Err(MMError::InvalidMarketState(
                "high and low price vectors must have same length".to_string(),
            ));
        }

        if high_prices.is_empty() {
            return Err(MMError::InvalidMarketState(
                "need at least 1 price pair for Parkinson estimator".to_string(),
            ));
        }

        let mut sum_squared_ratios = Decimal::ZERO;

        for (&high, &low) in high_prices.iter().zip(low_prices) {
            if high <= Decimal::ZERO || low <= Decimal::ZERO {
                return Err(MMError::InvalidMarketState(
                    "prices must be positive".to_string(),
                ));
            }

            if high < low {
                return Err(MMError::InvalidMarketState(
                    "high price must be >= low price".to_string(),
                ));
            }

            let ratio = high.checked_div(low).or_overflow("high low ratio")?;
            let log_ratio = decimal_ln(ratio)?;
            sum_squared_ratios = log_ratio
                .checked_mul(log_ratio)
                .and_then(|sq| sum_squared_ratios.checked_add(sq))
                .or_overflow("parkinson sum")?;
        }

        // Parkinson formula: σ² = (1/(4n*ln(2))) * Σ(ln(H/L))²
        let ln_2 = decimal_ln(Decimal::from(2))?;
        let n = Decimal::from(high_prices.len());
        let four = Decimal::from(4);

        let variance = four
            .checked_mul(n)
            .and_then(|x| x.checked_mul(ln_2))
            .and_then(|d| sum_squared_ratios.checked_div(d))
            .or_overflow("parkinson variance")?;
        let std_dev = decimal_sqrt(variance)?;

        // Annualize
        let annualization_factor = self.get_annualization_factor()?;
        std_dev
            .checked_mul(annualization_factor)
            .or_overflow("annualized volatility")
    }
}

/// The sample variance `Σ(r - mean)² / (n - 1)` of `returns`.
///
/// # Errors
///
/// Returns [`MMError::NumericalError`] if a sum overflows or there are
/// fewer than 2 returns.
fn sample_variance(returns: &[Decimal]) -> MMResult<Decimal> {
    let n = Decimal::from(returns.len());
    let mean = returns
        .iter()
        .try_fold(Decimal::ZERO, |acc, &r| acc.checked_add(r))
        .and_then(|sum| sum.checked_div(n))
        .or_overflow("mean log return")?;
    let squared_deviations = returns
        .iter()
        .try_fold(Decimal::ZERO, |acc, &r| {
            let dev = r.checked_sub(mean)?;
            acc.checked_add(dev.checked_mul(dev)?)
        })
        .or_overflow("squared deviations")?;
    let dof = returns
        .len()
        .checked_sub(1)
        .map(Decimal::from)
        .or_overflow("degrees of freedom")?;
    squared_deviations
        .checked_div(dof)
        .or_overflow("sample variance")
}

/// Computes the log returns `ln(p[i] / p[i - 1])` of a price series.
///
/// # Errors
///
/// Returns `MMError::InvalidMarketState` if any price is not positive or
/// there are fewer than 2 returns.
fn log_returns(prices: &[Decimal]) -> MMResult<Vec<Decimal>> {
    let log_returns = consecutive_log_returns(prices)?;
    // The sample variance divides by `n - 1`.
    if log_returns.len() < 2 {
        return Err(MMError::InvalidMarketState(format!(
            "need at least 2 log returns (3 prices), got {}",
            log_returns.len()
        )));
    }
    Ok(log_returns)
}

/// Computes the log returns of a price series, however few.
///
/// # Errors
///
/// Returns `MMError::InvalidMarketState` if any price is not positive.
pub(crate) fn consecutive_log_returns(prices: &[Decimal]) -> MMResult<Vec<Decimal>> {
    let mut log_returns = Vec::with_capacity(prices.len().saturating_sub(1));
    for window in prices.windows(2) {
        let [prev, next] = window else {
            continue;
        };
        if *next <= Decimal::ZERO || *prev <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(
                "prices must be positive".to_string(),
            ));
        }
        log_returns.push(decimal_ln(
            next.checked_div(*prev).or_overflow("price ratio")?,
        )?);
    }
    Ok(log_returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_variance_overflow_returns_numerical_error() {
        let result = sample_variance(&[Decimal::MAX, Decimal::MAX]);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        assert_eq!(sample_variance(&[dec!(1), dec!(3)]).unwrap(), dec!(2));
    }

    #[test]
    fn test_calculate_simple_two_prices_returns_err() {
        let estimator = VolatilityEstimator::new();
        let result = estimator.calculate_simple(&[dec!(100), dec!(101)]);
        assert!(matches!(result, Err(MMError::InvalidMarketState(_))));
    }

    #[test]
    fn test_calculate_ewma_two_prices_returns_err() {
        let estimator = VolatilityEstimator::new();
        let result = estimator.calculate_ewma(&[dec!(100), dec!(101)], dec!(0.94));
        assert!(matches!(result, Err(MMError::InvalidMarketState(_))));
    }
    use crate::dec;

    #[test]
    fn test_volatility_estimator_new() {
        let estimator = VolatilityEstimator::new();
        assert!(estimator.annualization_factor.is_none());
    }

    #[test]
    fn test_volatility_estimator_with_custom_factor() {
        let factor = dec!(100);
        let estimator = VolatilityEstimator::with_annualization_factor(factor);
        assert_eq!(estimator.annualization_factor, Some(factor));
    }

    #[test]
    fn test_calculate_simple_valid() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![
            dec!(100.0),
            dec!(101.0),
            dec!(99.5),
            dec!(100.5),
            dec!(102.0),
        ];

        let vol = estimator.calculate_simple(&prices).unwrap();
        assert!(vol > Decimal::ZERO);
        // Volatility should be reasonable (less than 100% annualized)
        assert!(vol < Decimal::ONE);
    }

    #[test]
    fn test_calculate_simple_insufficient_data() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![dec!(100.0)];

        let result = estimator.calculate_simple(&prices);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_calculate_simple_negative_price() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![dec!(100.0), dec!(-50.0), dec!(75.0)];

        let result = estimator.calculate_simple(&prices);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_ewma_valid() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![
            dec!(100.0),
            dec!(101.5),
            dec!(99.0),
            dec!(102.0),
            dec!(101.0),
        ];

        let vol = estimator.calculate_ewma(&prices, dec!(0.94)).unwrap();
        assert!(vol > Decimal::ZERO);
        assert!(vol < Decimal::ONE);
    }

    #[test]
    fn test_calculate_ewma_invalid_lambda() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![dec!(100.0), dec!(101.0), dec!(102.0)];

        // Lambda = 0
        let result = estimator.calculate_ewma(&prices, Decimal::ZERO);
        assert!(result.is_err());

        // Lambda = 1
        let result = estimator.calculate_ewma(&prices, Decimal::ONE);
        assert!(result.is_err());

        // Lambda > 1
        let result = estimator.calculate_ewma(&prices, dec!(1.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_parkinson_valid() {
        let estimator = VolatilityEstimator::new();
        let highs = vec![dec!(102.0), dec!(103.0), dec!(101.5)];
        let lows = vec![dec!(99.0), dec!(100.0), dec!(98.5)];

        let vol = estimator.calculate_parkinson(&highs, &lows).unwrap();
        assert!(vol > Decimal::ZERO);
    }

    #[test]
    fn test_calculate_parkinson_mismatched_lengths() {
        let estimator = VolatilityEstimator::new();
        let highs = vec![dec!(102.0), dec!(103.0)];
        let lows = vec![dec!(99.0)];

        let result = estimator.calculate_parkinson(&highs, &lows);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_parkinson_high_less_than_low() {
        let estimator = VolatilityEstimator::new();
        let highs = vec![dec!(100.0), dec!(99.0)];
        let lows = vec![dec!(101.0), dec!(100.0)];

        let result = estimator.calculate_parkinson(&highs, &lows);
        assert!(result.is_err());
    }

    #[test]
    fn test_ewma_vs_simple() {
        let estimator = VolatilityEstimator::new();
        // Prices with increasing volatility at the end
        let prices = vec![
            dec!(100.0),
            dec!(100.5),
            dec!(101.0),
            dec!(105.0),
            dec!(95.0),
        ];

        let simple_vol = estimator.calculate_simple(&prices).unwrap();
        let ewma_vol = estimator.calculate_ewma(&prices, dec!(0.94)).unwrap();

        // EWMA should give more weight to recent volatility
        // Both should be positive
        assert!(simple_vol > Decimal::ZERO);
        assert!(ewma_vol > Decimal::ZERO);
    }

    #[test]
    fn test_calculate_ewma_insufficient_data() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![dec!(100.0)]; // Solo 1 precio

        let result = estimator.calculate_ewma(&prices, dec!(0.94));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_calculate_ewma_negative_price() {
        let estimator = VolatilityEstimator::new();
        let prices = vec![dec!(100.0), dec!(-50.0), dec!(75.0)];

        let result = estimator.calculate_ewma(&prices, dec!(0.94));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_calculate_ewma_with_many_prices() {
        let estimator = VolatilityEstimator::new();
        // Test with >= 5 prices to cover the first branch of initial_variance
        let prices = vec![
            dec!(100.0),
            dec!(101.0),
            dec!(102.0),
            dec!(101.5),
            dec!(103.0),
            dec!(104.0),
        ];

        let vol = estimator.calculate_ewma(&prices, dec!(0.94)).unwrap();
        assert!(vol > Decimal::ZERO);
    }

    #[test]
    fn test_calculate_parkinson_empty_vectors() {
        let estimator = VolatilityEstimator::new();
        let highs: Vec<Decimal> = vec![];
        let lows: Vec<Decimal> = vec![];

        let result = estimator.calculate_parkinson(&highs, &lows);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_calculate_parkinson_negative_price() {
        let estimator = VolatilityEstimator::new();
        let highs = vec![dec!(102.0), dec!(-103.0)];
        let lows = vec![dec!(99.0), dec!(-105.0)];

        let result = estimator.calculate_parkinson(&highs, &lows);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_calculate_parkinson_zero_price() {
        let estimator = VolatilityEstimator::new();
        let highs = vec![dec!(102.0), Decimal::ZERO];
        let lows = vec![dec!(99.0), dec!(98.0)];

        let result = estimator.calculate_parkinson(&highs, &lows);
        assert!(result.is_err());
    }
}
