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
use crate::types::decimal::{decimal_ln, decimal_sqrt};
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

        // Calculate mean of log returns
        let sum: Decimal = log_returns.iter().sum();
        let mean = sum / Decimal::from(log_returns.len());

        // Calculate variance
        let squared_deviations: Decimal = log_returns
            .iter()
            .map(|&r| {
                let dev = r - mean;
                dev * dev
            })
            .sum();

        let variance = squared_deviations / Decimal::from(log_returns.len() - 1);

        // Calculate standard deviation
        let std_dev = decimal_sqrt(variance)?;

        // Annualize
        let annualization_factor = self.get_annualization_factor()?;
        Ok(std_dev * annualization_factor)
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

        // Initialize variance with simple variance of first few returns
        let initial_variance = if log_returns.len() >= 5 {
            let first_returns = log_returns.get(..5).unwrap_or_default();
            let mean: Decimal = first_returns.iter().sum::<Decimal>() / Decimal::from(5);
            let squared_devs: Decimal = first_returns
                .iter()
                .map(|&r| {
                    let dev = r - mean;
                    dev * dev
                })
                .sum();
            squared_devs / Decimal::from(4)
        } else {
            let mean: Decimal =
                log_returns.iter().sum::<Decimal>() / Decimal::from(log_returns.len());
            let squared_devs: Decimal = log_returns
                .iter()
                .map(|&r| {
                    let dev = r - mean;
                    dev * dev
                })
                .sum();
            squared_devs / Decimal::from(log_returns.len() - 1)
        };

        // EWMA recursion
        let one_minus_lambda = Decimal::ONE - lambda;
        let mut variance = initial_variance;

        for &log_return in &log_returns {
            let squared_return = log_return * log_return;
            variance = lambda * variance + one_minus_lambda * squared_return;
        }

        // Calculate standard deviation and annualize
        let std_dev = decimal_sqrt(variance)?;
        let annualization_factor = self.get_annualization_factor()?;
        Ok(std_dev * annualization_factor)
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

            let ratio = high / low;
            let log_ratio = decimal_ln(ratio)?;
            sum_squared_ratios += log_ratio * log_ratio;
        }

        // Parkinson formula: σ² = (1/(4n*ln(2))) * Σ(ln(H/L))²
        let ln_2 = decimal_ln(Decimal::from(2))?;
        let n = Decimal::from(high_prices.len());
        let four = Decimal::from(4);

        let variance = sum_squared_ratios / (four * n * ln_2);
        let std_dev = decimal_sqrt(variance)?;

        // Annualize
        let annualization_factor = self.get_annualization_factor()?;
        Ok(std_dev * annualization_factor)
    }
}

/// Computes the log returns `ln(p[i] / p[i - 1])` of a price series.
///
/// # Errors
///
/// Returns `MMError::InvalidMarketState` if any price is not positive or
/// there are fewer than 2 returns.
fn log_returns(prices: &[Decimal]) -> MMResult<Vec<Decimal>> {
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
        log_returns.push(decimal_ln(*next / *prev)?);
    }
    // The sample variance divides by `n - 1`.
    if log_returns.len() < 2 {
        return Err(MMError::InvalidMarketState(format!(
            "need at least 2 log returns (3 prices), got {}",
            log_returns.len()
        )));
    }
    Ok(log_returns)
}

#[cfg(test)]
mod tests {
    use super::*;

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
