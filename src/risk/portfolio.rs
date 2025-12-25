//! Portfolio risk management with correlation-aware calculations.
//!
//! This module provides tools for multi-asset portfolio risk management,
//! including correlation tracking and portfolio-level risk calculations.
//!
//! # Overview
//!
//! Market makers often trade multiple correlated assets:
//!
//! - Cross-margining benefits from correlation
//! - Hedging opportunities between correlated pairs
//! - Portfolio-level risk limits more accurate than per-asset limits
//! - Spread trading between correlated assets
//!
//! # Components
//!
//! - [`AssetId`]: Unique identifier for assets
//! - [`CorrelationMatrix`]: Symmetric correlation matrix with validation
//! - [`PortfolioPosition`]: Multi-asset position tracking
//! - [`PortfolioRiskCalculator`]: Variance, volatility, VaR calculations
//! - [`HedgeCalculator`]: Cross-asset hedging ratios
//!
//! # Mathematical Background
//!
//! ## Portfolio Variance
//!
//! ```text
//! σ²_p = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
//! ```
//!
//! ## Hedge Ratio
//!
//! ```text
//! β = ρ × (σ_target / σ_hedge)
//! ```
//!
//! ## Parametric VaR
//!
//! ```text
//! VaR = z × σ_p × √horizon
//! ```
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::risk::portfolio::{
//!     AssetId, CorrelationMatrix, PortfolioPosition, PortfolioRiskCalculator,
//! };
//! use market_maker_rs::dec;
//!
//! // Create assets
//! let btc = AssetId::new("BTC");
//! let eth = AssetId::new("ETH");
//!
//! // Create correlation matrix
//! let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
//! matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
//!
//! // Create portfolio
//! let mut portfolio = PortfolioPosition::new();
//! portfolio.set_position(btc.clone(), dec!(1.0), dec!(0.05)); // 1 BTC, 5% vol
//! portfolio.set_position(eth.clone(), dec!(10.0), dec!(0.08)); // 10 ETH, 8% vol
//!
//! // Calculate portfolio risk
//! let calculator = PortfolioRiskCalculator::new(matrix);
//! let vol = calculator.portfolio_volatility(&portfolio).unwrap();
//! println!("Portfolio volatility: {:.2}%", vol * dec!(100));
//! ```

use crate::Decimal;
use crate::types::decimal::decimal_sqrt;
use crate::types::error::{MMError, MMResult};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for an asset.
///
/// Used to identify assets in correlation matrices and portfolios.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::portfolio::AssetId;
///
/// let btc = AssetId::new("BTC");
/// let eth = AssetId::from("ETH");
///
/// assert_ne!(btc, eth);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssetId(pub String);

impl AssetId {
    /// Creates a new asset ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the asset ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for AssetId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for AssetId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Symmetric correlation matrix for multiple assets.
///
/// Stores correlations between pairs of assets efficiently using
/// only the upper triangle of the matrix.
///
/// # Invariants
///
/// - Diagonal elements are always 1.0 (self-correlation)
/// - Off-diagonal elements are in range \[-1, 1\]
/// - Matrix is symmetric: ρ(A,B) = ρ(B,A)
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::portfolio::{AssetId, CorrelationMatrix};
/// use market_maker_rs::dec;
///
/// let btc = AssetId::new("BTC");
/// let eth = AssetId::new("ETH");
/// let sol = AssetId::new("SOL");
///
/// let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone(), sol.clone()]);
///
/// // Set correlations
/// matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
/// matrix.set_correlation(&btc, &sol, dec!(0.6)).unwrap();
/// matrix.set_correlation(&eth, &sol, dec!(0.7)).unwrap();
///
/// // Get correlations (symmetric)
/// assert_eq!(matrix.get_correlation(&btc, &eth), Some(dec!(0.8)));
/// assert_eq!(matrix.get_correlation(&eth, &btc), Some(dec!(0.8)));
///
/// // Self-correlation is always 1
/// assert_eq!(matrix.get_correlation(&btc, &btc), Some(dec!(1.0)));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CorrelationMatrix {
    /// List of assets in the matrix.
    assets: Vec<AssetId>,
    /// Flat vector storing upper triangle (including diagonal).
    /// Index formula: i * n - i * (i + 1) / 2 + j for i <= j
    correlations: Vec<Decimal>,
    /// Timestamp of last update in milliseconds.
    last_update: u64,
}

impl CorrelationMatrix {
    /// Creates a new correlation matrix initialized to identity.
    ///
    /// All self-correlations are 1.0, all cross-correlations are 0.0.
    #[must_use]
    pub fn new(assets: Vec<AssetId>) -> Self {
        let n = assets.len();
        // Upper triangle size: n * (n + 1) / 2
        let size = n * (n + 1) / 2;
        let mut correlations = vec![Decimal::ZERO; size];

        // Set diagonal to 1.0
        for i in 0..n {
            let idx = Self::index_for(i, i, n);
            correlations[idx] = Decimal::ONE;
        }

        Self {
            assets,
            correlations,
            last_update: 0,
        }
    }

    /// Creates an identity correlation matrix (no correlations).
    #[must_use]
    pub fn identity(assets: Vec<AssetId>) -> Self {
        Self::new(assets)
    }

    /// Returns the number of assets in the matrix.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Returns the list of assets.
    #[must_use]
    pub fn assets(&self) -> &[AssetId] {
        &self.assets
    }

    /// Returns the timestamp of the last update.
    #[must_use]
    pub fn last_update(&self) -> u64 {
        self.last_update
    }

    /// Calculates the flat index for position (i, j) in upper triangle.
    fn index_for(i: usize, j: usize, n: usize) -> usize {
        let (row, col) = if i <= j { (i, j) } else { (j, i) };
        row * n - row * (row + 1) / 2 + col
    }

    /// Gets the index of an asset in the matrix.
    fn asset_index(&self, asset: &AssetId) -> Option<usize> {
        self.assets.iter().position(|a| a == asset)
    }

    /// Gets the correlation between two assets.
    ///
    /// Returns `None` if either asset is not in the matrix.
    #[must_use]
    pub fn get_correlation(&self, asset1: &AssetId, asset2: &AssetId) -> Option<Decimal> {
        let i = self.asset_index(asset1)?;
        let j = self.asset_index(asset2)?;
        let idx = Self::index_for(i, j, self.assets.len());
        Some(self.correlations[idx])
    }

    /// Sets the correlation between two assets.
    ///
    /// # Arguments
    ///
    /// * `asset1` - First asset
    /// * `asset2` - Second asset
    /// * `correlation` - Correlation value in \[-1, 1\]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Either asset is not in the matrix
    /// - Correlation is outside \[-1, 1\]
    /// - Trying to set self-correlation to non-1.0 value
    pub fn set_correlation(
        &mut self,
        asset1: &AssetId,
        asset2: &AssetId,
        correlation: Decimal,
    ) -> MMResult<()> {
        // Validate correlation range
        if correlation < Decimal::NEGATIVE_ONE || correlation > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(format!(
                "Correlation must be in [-1, 1], got {}",
                correlation
            )));
        }

        let i = self.asset_index(asset1).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("Asset {} not in matrix", asset1))
        })?;

        let j = self.asset_index(asset2).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("Asset {} not in matrix", asset2))
        })?;

        // Self-correlation must be 1.0
        if i == j && correlation != Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "Self-correlation must be 1.0".to_string(),
            ));
        }

        let idx = Self::index_for(i, j, self.assets.len());
        self.correlations[idx] = correlation;

        Ok(())
    }

    /// Updates correlations from return data.
    ///
    /// Calculates Pearson correlation coefficients from historical returns.
    ///
    /// # Arguments
    ///
    /// * `returns` - Map of asset ID to vector of returns
    /// * `timestamp` - Current timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// Returns error if return vectors have different lengths or insufficient data.
    pub fn update_from_returns(
        &mut self,
        returns: &HashMap<AssetId, Vec<Decimal>>,
        timestamp: u64,
    ) -> MMResult<()> {
        // Verify all assets have return data
        let mut return_len: Option<usize> = None;

        for asset in &self.assets {
            let asset_returns = returns.get(asset).ok_or_else(|| {
                MMError::InvalidConfiguration(format!("No returns for asset {}", asset))
            })?;

            match return_len {
                None => return_len = Some(asset_returns.len()),
                Some(len) if len != asset_returns.len() => {
                    return Err(MMError::InvalidConfiguration(
                        "Return vectors must have same length".to_string(),
                    ));
                }
                _ => {}
            }
        }

        let n_returns = return_len.unwrap_or(0);
        if n_returns < 2 {
            return Err(MMError::InvalidConfiguration(
                "Need at least 2 return observations".to_string(),
            ));
        }

        // Calculate correlations for each pair
        for i in 0..self.assets.len() {
            for j in i..self.assets.len() {
                if i == j {
                    continue; // Diagonal stays 1.0
                }

                let returns_i = returns.get(&self.assets[i]).unwrap();
                let returns_j = returns.get(&self.assets[j]).unwrap();

                let correlation = Self::calculate_correlation(returns_i, returns_j)?;
                let idx = Self::index_for(i, j, self.assets.len());
                self.correlations[idx] = correlation;
            }
        }

        self.last_update = timestamp;
        Ok(())
    }

    /// Calculates Pearson correlation between two return series.
    fn calculate_correlation(x: &[Decimal], y: &[Decimal]) -> MMResult<Decimal> {
        let n = x.len();
        if n < 2 {
            return Ok(Decimal::ZERO);
        }

        let n_dec = Decimal::from(n);

        // Calculate means
        let mean_x: Decimal = x.iter().copied().sum::<Decimal>() / n_dec;
        let mean_y: Decimal = y.iter().copied().sum::<Decimal>() / n_dec;

        // Calculate covariance and variances
        let mut cov = Decimal::ZERO;
        let mut var_x = Decimal::ZERO;
        let mut var_y = Decimal::ZERO;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x.is_zero() || var_y.is_zero() {
            return Ok(Decimal::ZERO);
        }

        // Correlation = cov / sqrt(var_x * var_y)
        let denominator = decimal_sqrt(var_x * var_y)?;
        if denominator.is_zero() {
            return Ok(Decimal::ZERO);
        }

        let correlation = cov / denominator;

        // Clamp to [-1, 1] to handle numerical errors
        Ok(correlation.max(Decimal::NEGATIVE_ONE).min(Decimal::ONE))
    }

    /// Validates that the matrix is properly formed.
    ///
    /// Checks:
    /// - All diagonal elements are 1.0
    /// - All off-diagonal elements are in \[-1, 1\]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let n = self.assets.len();

        for i in 0..n {
            for j in i..n {
                let idx = Self::index_for(i, j, n);
                let corr = self.correlations[idx];

                if i == j {
                    // Diagonal must be 1.0
                    if corr != Decimal::ONE {
                        return false;
                    }
                } else {
                    // Off-diagonal must be in [-1, 1]
                    if corr < Decimal::NEGATIVE_ONE || corr > Decimal::ONE {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Returns the correlation matrix as a 2D vector.
    #[must_use]
    pub fn to_matrix(&self) -> Vec<Vec<Decimal>> {
        let n = self.assets.len();
        let mut matrix = vec![vec![Decimal::ZERO; n]; n];

        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let idx = Self::index_for(i, j, n);
                *cell = self.correlations[idx];
            }
        }

        matrix
    }
}

/// Multi-asset portfolio position.
///
/// Tracks positions and volatilities for multiple assets.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::portfolio::{AssetId, PortfolioPosition};
/// use market_maker_rs::dec;
///
/// let mut portfolio = PortfolioPosition::new();
///
/// let btc = AssetId::new("BTC");
/// portfolio.set_position(btc.clone(), dec!(1.5), dec!(0.05));
///
/// assert_eq!(portfolio.get_position(&btc), Some(dec!(1.5)));
/// assert_eq!(portfolio.get_volatility(&btc), Some(dec!(0.05)));
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PortfolioPosition {
    /// Position sizes by asset.
    positions: HashMap<AssetId, Decimal>,
    /// Volatilities by asset.
    volatilities: HashMap<AssetId, Decimal>,
}

impl PortfolioPosition {
    /// Creates a new empty portfolio.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the position and volatility for an asset.
    ///
    /// # Arguments
    ///
    /// * `asset` - Asset identifier
    /// * `position` - Position size (can be negative for short)
    /// * `volatility` - Asset volatility (annualized)
    pub fn set_position(&mut self, asset: AssetId, position: Decimal, volatility: Decimal) {
        self.positions.insert(asset.clone(), position);
        self.volatilities.insert(asset, volatility);
    }

    /// Gets the position for an asset.
    #[must_use]
    pub fn get_position(&self, asset: &AssetId) -> Option<Decimal> {
        self.positions.get(asset).copied()
    }

    /// Gets the volatility for an asset.
    #[must_use]
    pub fn get_volatility(&self, asset: &AssetId) -> Option<Decimal> {
        self.volatilities.get(asset).copied()
    }

    /// Removes an asset from the portfolio.
    pub fn remove_asset(&mut self, asset: &AssetId) {
        self.positions.remove(asset);
        self.volatilities.remove(asset);
    }

    /// Returns all assets in the portfolio.
    #[must_use]
    pub fn assets(&self) -> Vec<&AssetId> {
        self.positions.keys().collect()
    }

    /// Returns the number of assets in the portfolio.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if the portfolio is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns the total absolute position (sum of |position|).
    #[must_use]
    pub fn total_absolute_position(&self) -> Decimal {
        self.positions.values().map(|p| p.abs()).sum()
    }

    /// Returns the net position (sum of positions).
    #[must_use]
    pub fn net_position(&self) -> Decimal {
        self.positions.values().copied().sum()
    }
}

/// Portfolio risk calculator.
///
/// Calculates portfolio-level risk metrics using correlation data.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::portfolio::{
///     AssetId, CorrelationMatrix, PortfolioPosition, PortfolioRiskCalculator,
/// };
/// use market_maker_rs::dec;
///
/// let btc = AssetId::new("BTC");
/// let eth = AssetId::new("ETH");
///
/// let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
/// matrix.set_correlation(&btc, &eth, dec!(0.7)).unwrap();
///
/// let mut portfolio = PortfolioPosition::new();
/// portfolio.set_position(btc, dec!(1.0), dec!(0.05));
/// portfolio.set_position(eth, dec!(2.0), dec!(0.08));
///
/// let calculator = PortfolioRiskCalculator::new(matrix);
/// let variance = calculator.portfolio_variance(&portfolio).unwrap();
/// let volatility = calculator.portfolio_volatility(&portfolio).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PortfolioRiskCalculator {
    correlation_matrix: CorrelationMatrix,
}

impl PortfolioRiskCalculator {
    /// Creates a new portfolio risk calculator.
    #[must_use]
    pub fn new(correlation_matrix: CorrelationMatrix) -> Self {
        Self { correlation_matrix }
    }

    /// Returns a reference to the correlation matrix.
    #[must_use]
    pub fn correlation_matrix(&self) -> &CorrelationMatrix {
        &self.correlation_matrix
    }

    /// Calculates portfolio variance.
    ///
    /// Formula: σ²_p = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
    ///
    /// Where:
    /// - wᵢ, wⱼ are position sizes
    /// - σᵢ, σⱼ are volatilities
    /// - ρᵢⱼ is the correlation between assets i and j
    pub fn portfolio_variance(&self, portfolio: &PortfolioPosition) -> MMResult<Decimal> {
        let assets = portfolio.assets();

        if assets.is_empty() {
            return Ok(Decimal::ZERO);
        }

        let mut variance = Decimal::ZERO;

        for asset_i in &assets {
            let pos_i = portfolio.get_position(asset_i).unwrap_or(Decimal::ZERO);
            let vol_i = portfolio.get_volatility(asset_i).unwrap_or(Decimal::ZERO);

            for asset_j in &assets {
                let pos_j = portfolio.get_position(asset_j).unwrap_or(Decimal::ZERO);
                let vol_j = portfolio.get_volatility(asset_j).unwrap_or(Decimal::ZERO);

                let correlation = self
                    .correlation_matrix
                    .get_correlation(asset_i, asset_j)
                    .unwrap_or_else(|| {
                        if asset_i == asset_j {
                            Decimal::ONE
                        } else {
                            Decimal::ZERO
                        }
                    });

                variance += pos_i * pos_j * vol_i * vol_j * correlation;
            }
        }

        Ok(variance)
    }

    /// Calculates portfolio volatility (standard deviation).
    ///
    /// Formula: σ_p = √(σ²_p)
    pub fn portfolio_volatility(&self, portfolio: &PortfolioPosition) -> MMResult<Decimal> {
        let variance = self.portfolio_variance(portfolio)?;
        if variance <= Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }
        decimal_sqrt(variance)
    }

    /// Calculates parametric Value at Risk (VaR).
    ///
    /// Formula: VaR = z × σ_p × √horizon
    ///
    /// # Arguments
    ///
    /// * `portfolio` - Portfolio position
    /// * `confidence` - Confidence level (e.g., 0.95 for 95%)
    /// * `horizon_days` - Time horizon in days
    ///
    /// # Returns
    ///
    /// VaR as a positive value representing potential loss.
    pub fn portfolio_var(
        &self,
        portfolio: &PortfolioPosition,
        confidence: Decimal,
        horizon_days: u32,
    ) -> MMResult<Decimal> {
        let volatility = self.portfolio_volatility(portfolio)?;

        // Z-score for confidence level (approximation)
        let z_score = self.confidence_to_z_score(confidence);

        // Scale by time horizon
        let horizon_factor = decimal_sqrt(Decimal::from(horizon_days))?;

        Ok(z_score * volatility * horizon_factor)
    }

    /// Converts confidence level to z-score.
    fn confidence_to_z_score(&self, confidence: Decimal) -> Decimal {
        // Common z-scores for standard confidence levels
        let conf_90 = Decimal::from_str_exact("0.90").unwrap();
        let conf_95 = Decimal::from_str_exact("0.95").unwrap();
        let conf_99 = Decimal::from_str_exact("0.99").unwrap();

        if confidence >= conf_99 {
            Decimal::from_str_exact("2.326").unwrap() // 99%
        } else if confidence >= conf_95 {
            Decimal::from_str_exact("1.645").unwrap() // 95%
        } else if confidence >= conf_90 {
            Decimal::from_str_exact("1.282").unwrap() // 90%
        } else {
            Decimal::ONE // Default
        }
    }

    /// Calculates marginal risk contribution for each asset.
    ///
    /// Marginal risk contribution shows how much each asset contributes
    /// to the total portfolio risk.
    ///
    /// Formula: MRC_i = (∂σ_p/∂w_i) × w_i
    pub fn marginal_risk_contribution(
        &self,
        portfolio: &PortfolioPosition,
    ) -> MMResult<HashMap<AssetId, Decimal>> {
        let portfolio_vol = self.portfolio_volatility(portfolio)?;

        if portfolio_vol.is_zero() {
            // Return zero contribution for all assets
            return Ok(portfolio
                .assets()
                .into_iter()
                .map(|a| (a.clone(), Decimal::ZERO))
                .collect());
        }

        let mut contributions = HashMap::new();

        for asset_i in portfolio.assets() {
            let pos_i = portfolio.get_position(asset_i).unwrap_or(Decimal::ZERO);
            let vol_i = portfolio.get_volatility(asset_i).unwrap_or(Decimal::ZERO);

            // Calculate covariance contribution
            let mut cov_contribution = Decimal::ZERO;

            for asset_j in portfolio.assets() {
                let pos_j = portfolio.get_position(asset_j).unwrap_or(Decimal::ZERO);
                let vol_j = portfolio.get_volatility(asset_j).unwrap_or(Decimal::ZERO);

                let correlation = self
                    .correlation_matrix
                    .get_correlation(asset_i, asset_j)
                    .unwrap_or_else(|| {
                        if asset_i == asset_j {
                            Decimal::ONE
                        } else {
                            Decimal::ZERO
                        }
                    });

                cov_contribution += pos_j * vol_i * vol_j * correlation;
            }

            // Marginal contribution = (cov_contribution / portfolio_vol) * pos_i
            let marginal = (cov_contribution / portfolio_vol) * pos_i;
            contributions.insert(asset_i.clone(), marginal);
        }

        Ok(contributions)
    }

    /// Calculates the diversification ratio.
    ///
    /// Ratio > 1 indicates diversification benefit.
    ///
    /// Formula: DR = Σ(|w_i| × σ_i) / σ_p
    pub fn diversification_ratio(&self, portfolio: &PortfolioPosition) -> MMResult<Decimal> {
        let portfolio_vol = self.portfolio_volatility(portfolio)?;

        if portfolio_vol.is_zero() {
            return Ok(Decimal::ONE);
        }

        let weighted_vol_sum: Decimal = portfolio
            .assets()
            .iter()
            .map(|asset| {
                let pos = portfolio.get_position(asset).unwrap_or(Decimal::ZERO);
                let vol = portfolio.get_volatility(asset).unwrap_or(Decimal::ZERO);
                pos.abs() * vol
            })
            .sum();

        Ok(weighted_vol_sum / portfolio_vol)
    }
}

/// Cross-asset hedge calculator.
///
/// Calculates optimal hedge ratios between correlated assets.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::portfolio::{AssetId, CorrelationMatrix, HedgeCalculator};
/// use market_maker_rs::dec;
///
/// let btc = AssetId::new("BTC");
/// let eth = AssetId::new("ETH");
///
/// let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
/// matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
///
/// let calculator = HedgeCalculator::new(matrix);
///
/// // Calculate hedge ratio: how much ETH to short to hedge BTC
/// let ratio = calculator.hedge_ratio(&btc, &eth, dec!(0.05), dec!(0.08)).unwrap();
/// println!("Hedge ratio: {}", ratio);
/// ```
#[derive(Debug, Clone)]
pub struct HedgeCalculator {
    correlation_matrix: CorrelationMatrix,
}

impl HedgeCalculator {
    /// Creates a new hedge calculator.
    #[must_use]
    pub fn new(correlation_matrix: CorrelationMatrix) -> Self {
        Self { correlation_matrix }
    }

    /// Calculates the hedge ratio between two assets.
    ///
    /// Formula: β = ρ × (σ_target / σ_hedge)
    ///
    /// # Arguments
    ///
    /// * `target` - Asset to hedge
    /// * `hedge` - Asset to use as hedge
    /// * `target_vol` - Volatility of target asset
    /// * `hedge_vol` - Volatility of hedge asset
    ///
    /// # Returns
    ///
    /// Hedge ratio (negative for opposite position).
    pub fn hedge_ratio(
        &self,
        target: &AssetId,
        hedge: &AssetId,
        target_vol: Decimal,
        hedge_vol: Decimal,
    ) -> MMResult<Decimal> {
        if hedge_vol.is_zero() {
            return Err(MMError::InvalidConfiguration(
                "Hedge asset volatility cannot be zero".to_string(),
            ));
        }

        let correlation = self
            .correlation_matrix
            .get_correlation(target, hedge)
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!(
                    "No correlation between {} and {}",
                    target, hedge
                ))
            })?;

        // β = -ρ × (σ_target / σ_hedge)
        // Negative because hedge should be opposite direction
        let ratio = -correlation * (target_vol / hedge_vol);

        Ok(ratio)
    }

    /// Finds the best hedge asset from available options.
    ///
    /// Returns the asset with highest absolute correlation and the hedge ratio.
    ///
    /// # Arguments
    ///
    /// * `target` - Asset to hedge
    /// * `available` - Available hedge assets
    /// * `volatilities` - Map of asset volatilities
    ///
    /// # Returns
    ///
    /// Best hedge asset and its hedge ratio, or None if no suitable hedge found.
    pub fn find_best_hedge(
        &self,
        target: &AssetId,
        available: &[AssetId],
        volatilities: &HashMap<AssetId, Decimal>,
    ) -> Option<(AssetId, Decimal)> {
        let target_vol = volatilities.get(target)?;

        let mut best: Option<(AssetId, Decimal, Decimal)> = None; // (asset, correlation, ratio)

        for hedge_asset in available {
            if hedge_asset == target {
                continue;
            }

            let correlation = self
                .correlation_matrix
                .get_correlation(target, hedge_asset)?;

            let hedge_vol = volatilities.get(hedge_asset)?;

            if hedge_vol.is_zero() {
                continue;
            }

            let abs_corr = correlation.abs();

            match &best {
                None => {
                    let ratio = -correlation * (*target_vol / *hedge_vol);
                    best = Some((hedge_asset.clone(), abs_corr, ratio));
                }
                Some((_, best_corr, _)) if abs_corr > *best_corr => {
                    let ratio = -correlation * (*target_vol / *hedge_vol);
                    best = Some((hedge_asset.clone(), abs_corr, ratio));
                }
                _ => {}
            }
        }

        best.map(|(asset, _, ratio)| (asset, ratio))
    }

    /// Calculates the residual risk after hedging.
    ///
    /// Formula: σ_residual = σ_target × √(1 - ρ²)
    pub fn residual_risk(
        &self,
        target: &AssetId,
        hedge: &AssetId,
        target_vol: Decimal,
    ) -> MMResult<Decimal> {
        let correlation = self
            .correlation_matrix
            .get_correlation(target, hedge)
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!(
                    "No correlation between {} and {}",
                    target, hedge
                ))
            })?;

        let one_minus_rho_sq = Decimal::ONE - correlation * correlation;

        if one_minus_rho_sq <= Decimal::ZERO {
            return Ok(Decimal::ZERO); // Perfect hedge
        }

        let residual_factor = decimal_sqrt(one_minus_rho_sq)?;
        Ok(target_vol * residual_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    // AssetId tests
    #[test]
    fn test_asset_id_new() {
        let asset = AssetId::new("BTC");
        assert_eq!(asset.as_str(), "BTC");
        assert_eq!(asset.to_string(), "BTC");
    }

    #[test]
    fn test_asset_id_from() {
        let asset1: AssetId = "ETH".into();
        let asset2: AssetId = String::from("SOL").into();

        assert_eq!(asset1.as_str(), "ETH");
        assert_eq!(asset2.as_str(), "SOL");
    }

    #[test]
    fn test_asset_id_equality() {
        let a1 = AssetId::new("BTC");
        let a2 = AssetId::new("BTC");
        let a3 = AssetId::new("ETH");

        assert_eq!(a1, a2);
        assert_ne!(a1, a3);
    }

    // CorrelationMatrix tests
    #[test]
    fn test_correlation_matrix_new() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);

        assert_eq!(matrix.asset_count(), 2);
        assert_eq!(matrix.get_correlation(&btc, &btc), Some(Decimal::ONE));
        assert_eq!(matrix.get_correlation(&eth, &eth), Some(Decimal::ONE));
        assert_eq!(matrix.get_correlation(&btc, &eth), Some(Decimal::ZERO));
    }

    #[test]
    fn test_correlation_matrix_set_get() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();

        assert_eq!(matrix.get_correlation(&btc, &eth), Some(dec!(0.8)));
        assert_eq!(matrix.get_correlation(&eth, &btc), Some(dec!(0.8))); // Symmetric
    }

    #[test]
    fn test_correlation_matrix_invalid_range() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);

        assert!(matrix.set_correlation(&btc, &eth, dec!(1.5)).is_err());
        assert!(matrix.set_correlation(&btc, &eth, dec!(-1.5)).is_err());
    }

    #[test]
    fn test_correlation_matrix_self_correlation() {
        let btc = AssetId::new("BTC");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone()]);

        // Cannot set self-correlation to non-1.0
        assert!(matrix.set_correlation(&btc, &btc, dec!(0.5)).is_err());
    }

    #[test]
    fn test_correlation_matrix_is_valid() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        assert!(matrix.is_valid());

        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
        assert!(matrix.is_valid());
    }

    #[test]
    fn test_correlation_matrix_to_matrix() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.7)).unwrap();

        let m = matrix.to_matrix();
        assert_eq!(m[0][0], Decimal::ONE);
        assert_eq!(m[1][1], Decimal::ONE);
        assert_eq!(m[0][1], dec!(0.7));
        assert_eq!(m[1][0], dec!(0.7));
    }

    #[test]
    fn test_correlation_matrix_update_from_returns() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);

        let mut returns = HashMap::new();
        returns.insert(
            btc.clone(),
            vec![dec!(0.01), dec!(0.02), dec!(-0.01), dec!(0.03)],
        );
        returns.insert(
            eth.clone(),
            vec![dec!(0.015), dec!(0.025), dec!(-0.005), dec!(0.035)],
        );

        matrix.update_from_returns(&returns, 1000).unwrap();

        let corr = matrix.get_correlation(&btc, &eth).unwrap();
        assert!(corr > dec!(0.9)); // Should be highly correlated
    }

    // PortfolioPosition tests
    #[test]
    fn test_portfolio_position_new() {
        let portfolio = PortfolioPosition::new();
        assert!(portfolio.is_empty());
        assert_eq!(portfolio.asset_count(), 0);
    }

    #[test]
    fn test_portfolio_position_set_get() {
        let mut portfolio = PortfolioPosition::new();
        let btc = AssetId::new("BTC");

        portfolio.set_position(btc.clone(), dec!(1.5), dec!(0.05));

        assert_eq!(portfolio.get_position(&btc), Some(dec!(1.5)));
        assert_eq!(portfolio.get_volatility(&btc), Some(dec!(0.05)));
        assert_eq!(portfolio.asset_count(), 1);
    }

    #[test]
    fn test_portfolio_position_remove() {
        let mut portfolio = PortfolioPosition::new();
        let btc = AssetId::new("BTC");

        portfolio.set_position(btc.clone(), dec!(1.0), dec!(0.05));
        assert_eq!(portfolio.asset_count(), 1);

        portfolio.remove_asset(&btc);
        assert_eq!(portfolio.asset_count(), 0);
    }

    #[test]
    fn test_portfolio_position_totals() {
        let mut portfolio = PortfolioPosition::new();

        portfolio.set_position(AssetId::new("BTC"), dec!(1.0), dec!(0.05));
        portfolio.set_position(AssetId::new("ETH"), dec!(-2.0), dec!(0.08));

        assert_eq!(portfolio.total_absolute_position(), dec!(3.0));
        assert_eq!(portfolio.net_position(), dec!(-1.0));
    }

    // PortfolioRiskCalculator tests
    #[test]
    fn test_portfolio_variance_single_asset() {
        let btc = AssetId::new("BTC");

        let matrix = CorrelationMatrix::new(vec![btc.clone()]);
        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc, dec!(1.0), dec!(0.05));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let variance = calculator.portfolio_variance(&portfolio).unwrap();

        // Variance = w² × σ² = 1² × 0.05² = 0.0025
        assert_eq!(variance, dec!(0.0025));
    }

    #[test]
    fn test_portfolio_variance_two_assets() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.5)).unwrap();

        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc, dec!(1.0), dec!(0.10));
        portfolio.set_position(eth, dec!(1.0), dec!(0.10));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let variance = calculator.portfolio_variance(&portfolio).unwrap();

        // Variance = σ₁²w₁² + σ₂²w₂² + 2ρσ₁σ₂w₁w₂
        // = 0.01 + 0.01 + 2×0.5×0.1×0.1×1×1 = 0.02 + 0.01 = 0.03
        assert_eq!(variance, dec!(0.03));
    }

    #[test]
    fn test_portfolio_volatility() {
        let btc = AssetId::new("BTC");

        let matrix = CorrelationMatrix::new(vec![btc.clone()]);
        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc, dec!(1.0), dec!(0.04));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let volatility = calculator.portfolio_volatility(&portfolio).unwrap();

        // Volatility = sqrt(0.04² × 1²) = 0.04
        assert_eq!(volatility, dec!(0.04));
    }

    #[test]
    fn test_portfolio_var() {
        let btc = AssetId::new("BTC");

        let matrix = CorrelationMatrix::new(vec![btc.clone()]);
        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc, dec!(1.0), dec!(0.10));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let var = calculator.portfolio_var(&portfolio, dec!(0.95), 1).unwrap();

        // VaR = z × σ × √horizon = 1.645 × 0.10 × 1 = 0.1645
        assert_eq!(var, dec!(0.1645));
    }

    #[test]
    fn test_diversification_ratio() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.5)).unwrap();

        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc, dec!(1.0), dec!(0.10));
        portfolio.set_position(eth, dec!(1.0), dec!(0.10));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let ratio = calculator.diversification_ratio(&portfolio).unwrap();

        // Ratio > 1 indicates diversification benefit
        assert!(ratio > Decimal::ONE);
    }

    // HedgeCalculator tests
    #[test]
    fn test_hedge_ratio() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();

        let calculator = HedgeCalculator::new(matrix);
        let ratio = calculator
            .hedge_ratio(&btc, &eth, dec!(0.05), dec!(0.10))
            .unwrap();

        // β = -ρ × (σ_target / σ_hedge) = -0.8 × (0.05 / 0.10) = -0.4
        assert_eq!(ratio, dec!(-0.4));
    }

    #[test]
    fn test_hedge_ratio_zero_vol() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();

        let calculator = HedgeCalculator::new(matrix);
        let result = calculator.hedge_ratio(&btc, &eth, dec!(0.05), dec!(0.0));

        assert!(result.is_err());
    }

    #[test]
    fn test_find_best_hedge() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");
        let sol = AssetId::new("SOL");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone(), sol.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();
        matrix.set_correlation(&btc, &sol, dec!(0.5)).unwrap();
        matrix.set_correlation(&eth, &sol, dec!(0.6)).unwrap();

        let mut vols = HashMap::new();
        vols.insert(btc.clone(), dec!(0.05));
        vols.insert(eth.clone(), dec!(0.08));
        vols.insert(sol.clone(), dec!(0.12));

        let calculator = HedgeCalculator::new(matrix);
        let best = calculator.find_best_hedge(&btc, &[eth.clone(), sol.clone()], &vols);

        assert!(best.is_some());
        let (best_asset, _ratio) = best.unwrap();
        assert_eq!(best_asset, eth); // ETH has higher correlation
    }

    #[test]
    fn test_residual_risk() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.8)).unwrap();

        let calculator = HedgeCalculator::new(matrix);
        let residual = calculator.residual_risk(&btc, &eth, dec!(0.10)).unwrap();

        // σ_residual = σ_target × √(1 - ρ²) = 0.10 × √(1 - 0.64) = 0.10 × 0.6 = 0.06
        assert!(residual > dec!(0.05));
        assert!(residual < dec!(0.07));
    }

    #[test]
    fn test_residual_risk_perfect_correlation() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(1.0)).unwrap();

        let calculator = HedgeCalculator::new(matrix);
        let residual = calculator.residual_risk(&btc, &eth, dec!(0.10)).unwrap();

        // Perfect correlation = zero residual risk
        assert_eq!(residual, Decimal::ZERO);
    }

    #[test]
    fn test_marginal_risk_contribution() {
        let btc = AssetId::new("BTC");
        let eth = AssetId::new("ETH");

        let mut matrix = CorrelationMatrix::new(vec![btc.clone(), eth.clone()]);
        matrix.set_correlation(&btc, &eth, dec!(0.5)).unwrap();

        let mut portfolio = PortfolioPosition::new();
        portfolio.set_position(btc.clone(), dec!(1.0), dec!(0.10));
        portfolio.set_position(eth.clone(), dec!(1.0), dec!(0.10));

        let calculator = PortfolioRiskCalculator::new(matrix);
        let contributions = calculator.marginal_risk_contribution(&portfolio).unwrap();

        // Both assets should have positive contributions
        assert!(contributions.get(&btc).unwrap() > &Decimal::ZERO);
        assert!(contributions.get(&eth).unwrap() > &Decimal::ZERO);
    }
}
