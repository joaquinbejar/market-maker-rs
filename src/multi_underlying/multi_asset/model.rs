//! The closed-form joint quotes of Bergault, Evangelista, Gueant and Vieira.

use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

use super::matrix::{Clipped, Matrix};
use crate::Decimal;
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

/// Most assets a model accepts: a bound on the construction cost and on the
/// matrices stored.
pub const MAX_ASSETS: usize = 64;

/// Relative size of a negative eigenvalue that counts as rounding noise, per row.
///
/// About 4500 times the machine epsilon. By Weyl's inequality it covers a few
/// roundings per entry of a matrix whose entries come without cancellation, and
/// the error of the eigendecomposition, about the size of the matrix times the
/// epsilon and the largest entry per sweep, over up to 64 sweeps.
const PSD_TOLERANCE: f64 = 1e-12;

/// The objective of the market maker (section 2.2 of the paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuotingObjective {
    /// Model A: expected CARA utility of the final mark-to-market value.
    ExponentialUtility,
    /// Model B: expected final value minus a running inventory penalty.
    RiskAdjustedExpectation,
}

/// Which covariance the risk matrix is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RiskView {
    /// The full covariance: an inventory in one asset moves the quotes of the
    /// assets correlated with it.
    Correlated,
    /// The covariance with its off diagonal entries removed, so each asset is
    /// its own one asset problem.
    Diagonal,
}

/// What the model needs to know about one asset.
///
/// Units must agree across assets: `volatility` per square root of the time
/// unit, `intensity_scale` per time unit, `intensity_decay` per unit of price,
/// and `trade_size` and `inventory_limit` in units of the asset.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetQuotingParams {
    volatility: Decimal,
    intensity_scale: Decimal,
    intensity_decay: Decimal,
    trade_size: Decimal,
    inventory_limit: Decimal,
}

impl AssetQuotingParams {
    /// Parameters of the asset: `Lambda(delta) = A exp(-k delta)` with
    /// `A = intensity_scale` and `k = intensity_decay`, trades of `trade_size`,
    /// and no side quoted that would take the absolute inventory beyond
    /// `inventory_limit`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if any value is not positive,
    /// or if `trade_size` is above `inventory_limit`, which could never quote.
    pub fn new(
        volatility: Decimal,
        intensity_scale: Decimal,
        intensity_decay: Decimal,
        trade_size: Decimal,
        inventory_limit: Decimal,
    ) -> MMResult<Self> {
        for (name, value) in [
            ("volatility", volatility),
            ("intensity scale", intensity_scale),
            ("intensity decay", intensity_decay),
            ("trade size", trade_size),
            ("inventory limit", inventory_limit),
        ] {
            if value <= Decimal::ZERO {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be positive, got {value}"
                )));
            }
        }
        if trade_size > inventory_limit {
            return Err(MMError::InvalidConfiguration(format!(
                "trade size {trade_size} is above the inventory limit {inventory_limit}"
            )));
        }
        Ok(Self {
            volatility,
            intensity_scale,
            intensity_decay,
            trade_size,
            inventory_limit,
        })
    }

    /// Volatility per square root of the time unit.
    #[must_use]
    #[inline]
    pub fn volatility(&self) -> Decimal {
        self.volatility
    }

    /// Trade size, in units of the asset.
    #[must_use]
    #[inline]
    pub fn trade_size(&self) -> Decimal {
        self.trade_size
    }

    /// Largest absolute inventory, in units of the asset.
    #[must_use]
    #[inline]
    pub fn inventory_limit(&self) -> Decimal {
        self.inventory_limit
    }
}

/// Distances of the two quotes from the reference price, before any rounding,
/// sizing or margin check. They are not orders: `MultiAssetQuoter`, with the
/// `margin` feature, is the way to a quote.
///
/// Either distance can be negative: a large inventory puts one side through
/// the reference price. Their sum never depends on the inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteOffsets {
    /// `delta^{i,b}`: the bid is the reference price minus this.
    pub bid: Decimal,
    /// `delta^{i,a}`: the ask is the reference price plus this.
    pub ask: Decimal,
}

/// The asymptotic closed-form quotes for symmetric exponential intensities.
///
/// With `D+ = diag(2 A^i C^i k^i z^i)` and
/// `Gamma = D+^(-1/2) (D+^(1/2) Sigma D+^(1/2))^(1/2) D+^(-1/2)`, section 3.3.3
/// of arXiv:1810.04383v5 gives, for asset `i` and inventory vector `q`:
///
/// ```text
/// delta^{i,b} =  sqrt(gamma) (q' Gamma e_i + z^i Gamma_ii / 2) + b^i
/// delta^{i,a} = -sqrt(gamma) (q' Gamma e_i - z^i Gamma_ii / 2) + b^i
/// ```
///
/// where `b^i = ln(1 + gamma z^i / k^i) / (gamma z^i)` and
/// `C^i = (1 + gamma z^i / k^i)^-(1 + k^i / (gamma z^i))` in Model A, and
/// `b^i = 1 / k^i` and `C^i = 1/e` in Model B.
///
/// Everything that depends only on the configuration is computed once, here;
/// [`MultiAssetModel::offsets`] is `Decimal` arithmetic on the stored matrices
/// and does not allocate.
///
/// The paper's finite horizon, asymmetric intensities, drift, client tiers and
/// request size distributions are not implemented, and neither is any option
/// logic. The paper also warns that the approximation can be inaccurate for
/// large inventories.
///
/// # Examples
///
/// ```
/// use market_maker_rs::dec;
/// use market_maker_rs::multi_underlying::multi_asset::{
///     AssetQuotingParams, MultiAssetModel, QuotingObjective, RiskView,
/// };
///
/// let asset = |sigma| AssetQuotingParams::new(sigma, dec!(10), dec!(20), dec!(1), dec!(50));
/// let model = MultiAssetModel::new(
///     vec![asset(dec!(1.2)).unwrap(), asset(dec!(0.6)).unwrap()],
///     &[vec![dec!(1), dec!(0.5)], vec![dec!(0.5), dec!(1)]],
///     dec!(0.08),
///     QuotingObjective::RiskAdjustedExpectation,
///     RiskView::Correlated,
/// )
/// .unwrap();
///
/// // The same long position in asset 1 skews it less when asset 2 hedges it.
/// let hedged = model.skew(0, &[dec!(4), dec!(-4)]).unwrap();
/// let concentrated = model.skew(0, &[dec!(4), dec!(4)]).unwrap();
/// assert!(hedged < concentrated);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultiAssetModel {
    assets: Vec<AssetQuotingParams>,
    objective: QuotingObjective,
    risk_view: RiskView,
    /// `Gamma`, by rows.
    risk_matrix: Vec<Decimal>,
    /// `sqrt(gamma) Gamma`, by rows.
    skew_matrix: Vec<Decimal>,
    /// `Sigma` under the chosen view, by rows.
    covariance: Vec<Decimal>,
    /// `sqrt(gamma) z^i Gamma_ii / 2 + b^i`.
    half_spreads: Vec<Decimal>,
}

fn to_f64(value: Decimal, name: &str) -> MMResult<f64> {
    value.to_f64().ok_or_else(|| {
        MMError::NumericalError(format!("{name} {value} has no floating point value"))
    })
}

fn to_decimal(value: f64, name: &str) -> MMResult<Decimal> {
    Decimal::from_f64(value)
        .ok_or_else(|| MMError::NumericalError(format!("{name} {value} is not a decimal")))
}

/// Checks a correlation matrix and returns it in floating point, with the
/// eigenvalue clipped within `band` as [`Matrix::principal_sqrt`] reports it.
fn checked_correlation(
    correlation: &[Vec<Decimal>],
    size: usize,
    band: f64,
) -> MMResult<(Matrix, Option<Clipped>)> {
    if correlation.len() != size || correlation.iter().any(|row| row.len() != size) {
        return Err(MMError::InvalidConfiguration(format!(
            "correlation matrix must be {size} by {size}"
        )));
    }
    let mut matrix = Matrix::zeros(size);
    for (i, row) in correlation.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let mirrored = correlation.get(j).and_then(|other| other.get(i));
            if mirrored != Some(value) {
                return Err(MMError::InvalidConfiguration(format!(
                    "correlation matrix is not symmetric at ({i}, {j})"
                )));
            }
            if i == j && *value != Decimal::ONE {
                return Err(MMError::InvalidConfiguration(format!(
                    "correlation diagonal at {i} must be 1, got {value}"
                )));
            }
            matrix.set(i, j, to_f64(*value, "correlation")?);
        }
    }
    // Not positive semidefinite is refused here, for either view: a diagonal
    // run must not be accepted on a correlation the correlated run refuses. With
    // a unit diagonal this also refuses any entry outside [-1, 1].
    let (_, clipped) = matrix.principal_sqrt(PSD_TOLERANCE, band)?;
    Ok((matrix, clipped))
}

/// The more negative of two clipped eigenvalues, each relative to its matrix.
fn worse(left: Option<Clipped>, right: Option<Clipped>) -> Option<Clipped> {
    match (left, right) {
        (Some(a), Some(b)) => Some(if b.eigenvalue < a.eigenvalue { b } else { a }),
        (a, b) => a.or(b),
    }
}

impl MultiAssetModel {
    /// Builds the model. `risk_aversion` is `gamma`, and `correlation` is the
    /// matrix of `rho^{i,j}`, one row per asset, in the order of `assets`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if there are no assets or more
    /// than [`MAX_ASSETS`], if `risk_aversion` is not positive, or if
    /// `correlation` is not a symmetric positive semidefinite matrix of the
    /// right size with a unit diagonal and entries in `[-1, 1]`. Returns
    /// [`MMError::NumericalError`] if a value does not fit in a `Decimal` or the
    /// eigendecomposition fails.
    pub fn new(
        assets: Vec<AssetQuotingParams>,
        correlation: &[Vec<Decimal>],
        risk_aversion: Decimal,
        objective: QuotingObjective,
        risk_view: RiskView,
    ) -> MMResult<Self> {
        Self::build(
            assets,
            correlation,
            risk_aversion,
            objective,
            risk_view,
            PSD_TOLERANCE,
        )
        .map(|(model, _)| model)
    }

    /// As [`MultiAssetModel::new`], for a correlation the crate derived, which is
    /// positive semidefinite in exact arithmetic and whose entries off the
    /// diagonal are each known to within `entry_error`. By Weyl's inequality that
    /// moves an eigenvalue by at most `entry_error` per row, so a negative
    /// eigenvalue above the floor of `PSD_TOLERANCE + entry_error` per row is
    /// rounding and is set to zero. One below the floor of `PSD_TOLERANCE` alone,
    /// which [`MultiAssetModel::new`] refuses, is returned.
    ///
    /// The check applies to the correlation and again to the scaled covariance
    /// whose root is taken. That one is the correlation under a congruence by a
    /// positive diagonal, which by Ostrowski's theorem keeps a negative eigenvalue
    /// no more negative relative to its largest entry, and four roundings per
    /// entry, well within `PSD_TOLERANCE`.
    ///
    /// # Errors
    ///
    /// As [`MultiAssetModel::new`], with that wider floor.
    #[cfg(all(feature = "options", feature = "margin"))]
    pub(super) fn derived(
        assets: Vec<AssetQuotingParams>,
        correlation: &[Vec<Decimal>],
        risk_aversion: Decimal,
        objective: QuotingObjective,
        risk_view: RiskView,
        entry_error: f64,
    ) -> MMResult<(Self, Option<Clipped>)> {
        Self::build(
            assets,
            correlation,
            risk_aversion,
            objective,
            risk_view,
            PSD_TOLERANCE + entry_error,
        )
    }

    fn build(
        assets: Vec<AssetQuotingParams>,
        correlation: &[Vec<Decimal>],
        risk_aversion: Decimal,
        objective: QuotingObjective,
        risk_view: RiskView,
        band: f64,
    ) -> MMResult<(Self, Option<Clipped>)> {
        let size = assets.len();
        if size == 0 || size > MAX_ASSETS {
            return Err(MMError::InvalidConfiguration(format!(
                "a model takes 1 to {MAX_ASSETS} assets, got {size}"
            )));
        }
        if risk_aversion <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "risk aversion must be positive, got {risk_aversion}"
            )));
        }
        let (rho, clipped) = checked_correlation(correlation, size, band)?;
        let gamma = to_f64(risk_aversion, "risk aversion")?;
        let gamma_root = gamma.sqrt();

        let mut sigmas = Vec::with_capacity(size);
        let mut depths = Vec::with_capacity(size);
        let mut bases = Vec::with_capacity(size);
        for asset in &assets {
            let scale = to_f64(asset.intensity_scale, "intensity scale")?;
            let decay = to_f64(asset.intensity_decay, "intensity decay")?;
            let trade = to_f64(asset.trade_size, "trade size")?;
            let (constant, base) = match objective {
                QuotingObjective::ExponentialUtility => {
                    let ratio = gamma * trade / decay;
                    (
                        // In logarithms: `1 + ratio` rounds to one for a tiny
                        // ratio, where the constant must tend to `1/e`.
                        (-(1.0 + ratio.recip()) * ratio.ln_1p()).exp(),
                        ratio.ln_1p() / (gamma * trade),
                    )
                }
                QuotingObjective::RiskAdjustedExpectation => ((-1.0_f64).exp(), 1.0 / decay),
            };
            sigmas.push(to_f64(asset.volatility, "volatility")?);
            depths.push(2.0 * scale * constant * decay * trade);
            bases.push(base);
        }

        let mut covariance = Matrix::zeros(size);
        let mut scaled = Matrix::zeros(size);
        for (i, (sigma_i, depth_i)) in sigmas.iter().zip(&depths).enumerate() {
            for (j, (sigma_j, depth_j)) in sigmas.iter().zip(&depths).enumerate() {
                let cross = match risk_view {
                    RiskView::Correlated => rho.at(i, j),
                    RiskView::Diagonal if i == j => 1.0,
                    RiskView::Diagonal => 0.0,
                };
                let value = cross * sigma_i * sigma_j;
                covariance.set(i, j, value);
                scaled.set(i, j, depth_i.sqrt() * value * depth_j.sqrt());
            }
        }
        let (root, scaled_clipped) = scaled.principal_sqrt(PSD_TOLERANCE, band)?;
        let clipped = worse(clipped, scaled_clipped);

        let mut gamma_matrix = Matrix::zeros(size);
        for (i, depth_i) in depths.iter().enumerate() {
            for (j, depth_j) in depths.iter().enumerate() {
                // The mean keeps the rounding of the rotations from breaking
                // the symmetry the paper relies on.
                let symmetric = 0.5 * (root.at(i, j) + root.at(j, i));
                gamma_matrix.set(i, j, symmetric / (depth_i * depth_j).sqrt());
            }
        }

        let mut risk_matrix = Vec::with_capacity(size.saturating_mul(size));
        let mut skew_matrix = Vec::with_capacity(size.saturating_mul(size));
        let mut stored_covariance = Vec::with_capacity(size.saturating_mul(size));
        for i in 0..size {
            for j in 0..size {
                let value = gamma_matrix.at(i, j);
                risk_matrix.push(to_decimal(value, "risk matrix entry")?);
                skew_matrix.push(to_decimal(gamma_root * value, "skew coefficient")?);
                stored_covariance.push(to_decimal(covariance.at(i, j), "covariance")?);
            }
        }
        let mut half_spreads = Vec::with_capacity(size);
        for (i, (asset, base)) in assets.iter().zip(&bases).enumerate() {
            let trade = to_f64(asset.trade_size, "trade size")?;
            let half = 0.5 * gamma_root * trade * gamma_matrix.at(i, i) + base;
            half_spreads.push(to_decimal(half, "half spread")?);
        }

        let model = Self {
            assets,
            objective,
            risk_view,
            risk_matrix,
            skew_matrix,
            covariance: stored_covariance,
            half_spreads,
        };
        Ok((model, clipped))
    }

    /// Number of assets.
    #[must_use]
    #[inline]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// The parameters of `asset`.
    #[must_use]
    #[inline]
    pub fn asset(&self, asset: usize) -> Option<&AssetQuotingParams> {
        self.assets.get(asset)
    }

    /// The objective this model was built for.
    #[must_use]
    #[inline]
    pub fn objective(&self) -> QuotingObjective {
        self.objective
    }

    /// The covariance view this model was built from.
    #[must_use]
    #[inline]
    pub fn risk_view(&self) -> RiskView {
        self.risk_view
    }

    fn cell(&self, row: usize, column: usize) -> Option<usize> {
        let size = self.assets.len();
        (row < size && column < size).then(|| row.saturating_mul(size).saturating_add(column))
    }

    /// The entry `Gamma_{row,column}`, or `None` out of range.
    #[must_use]
    pub fn risk_matrix(&self, row: usize, column: usize) -> Option<Decimal> {
        self.cell(row, column)
            .and_then(|cell| self.risk_matrix.get(cell).copied())
    }

    /// The entry of `Sigma` under the chosen view, or `None` out of range.
    #[must_use]
    pub fn covariance(&self, row: usize, column: usize) -> Option<Decimal> {
        self.cell(row, column)
            .and_then(|cell| self.covariance.get(cell).copied())
    }

    fn check_inventory(&self, inventory: &[Decimal]) -> MMResult<()> {
        if inventory.len() == self.assets.len() {
            Ok(())
        } else {
            Err(MMError::InvalidPositionUpdate(format!(
                "inventory has {} entries for {} assets",
                inventory.len(),
                self.assets.len()
            )))
        }
    }

    /// The skew of `asset`: `sqrt(gamma) q' Gamma e_i`. It is what the inventory
    /// subtracts from both quote prices.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidPositionUpdate`] if `inventory` does not have
    /// one entry per asset, [`MMError::InvalidConfiguration`] if `asset` is out
    /// of range, and [`MMError::NumericalError`] on overflow.
    pub fn skew(&self, asset: usize, inventory: &[Decimal]) -> MMResult<Decimal> {
        self.check_inventory(inventory)?;
        let size = self.assets.len();
        let row = self
            .cell(asset, 0)
            .and_then(|start| self.skew_matrix.get(start..start.saturating_add(size)))
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!("asset {asset} is out of range"))
            })?;
        let mut skew = Decimal::ZERO;
        for (coefficient, position) in row.iter().zip(inventory) {
            let term = coefficient
                .checked_mul(*position)
                .or_overflow("inventory skew term")?;
            skew = skew.checked_add(term).or_overflow("inventory skew")?;
        }
        Ok(skew)
    }

    /// The half spread of `asset`, which the inventory does not change.
    #[must_use]
    pub fn half_spread(&self, asset: usize) -> Option<Decimal> {
        self.half_spreads.get(asset).copied()
    }

    /// The quote distances of `asset` for the inventory vector.
    ///
    /// # Errors
    ///
    /// As [`MultiAssetModel::skew`].
    pub fn offsets(&self, asset: usize, inventory: &[Decimal]) -> MMResult<QuoteOffsets> {
        let skew = self.skew(asset, inventory)?;
        let half = self.half_spread(asset).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("asset {asset} is out of range"))
        })?;
        Ok(QuoteOffsets {
            bid: half.checked_add(skew).or_overflow("bid offset")?,
            ask: half.checked_sub(skew).or_overflow("ask offset")?,
        })
    }

    /// `q' Sigma q` under the chosen view.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidPositionUpdate`] if `inventory` does not have
    /// one entry per asset, and [`MMError::NumericalError`] on overflow.
    pub fn portfolio_variance(&self, inventory: &[Decimal]) -> MMResult<Decimal> {
        self.check_inventory(inventory)?;
        let size = self.assets.len().max(1);
        let mut variance = Decimal::ZERO;
        for (row, left) in self.covariance.chunks(size).zip(inventory) {
            for (entry, right) in row.iter().zip(inventory) {
                let term = entry
                    .checked_mul(*left)
                    .and_then(|value| value.checked_mul(*right))
                    .or_overflow("portfolio variance term")?;
                variance = variance
                    .checked_add(term)
                    .or_overflow("portfolio variance")?;
            }
        }
        Ok(variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_worse_clip_of_the_two_matrices_is_the_one_reported() {
        let clip = |eigenvalue: f64| {
            Some(Clipped {
                eigenvalue,
                floor: -2e-12,
                band: -1e-9,
            })
        };
        assert_eq!(worse(clip(-3e-12), clip(-5e-12)), clip(-5e-12));
        assert_eq!(worse(clip(-5e-12), clip(-3e-12)), clip(-5e-12));
        assert_eq!(worse(None, clip(-3e-12)), clip(-3e-12));
        assert_eq!(worse(clip(-3e-12), None), clip(-3e-12));
        assert_eq!(worse(None, None), None);
    }
}
