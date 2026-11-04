//! Walk-forward calibration of volatility and fill intensities.
//!
//! Parameters are estimated only from a half-open training window and stored
//! with that window, the data resolution and a hash of the data used, so that
//! each estimate can be traced and compared out of sample.
//!
//! # Units
//!
//! The units are those of the Avellaneda-Stoikov and GLFT modules, and of the
//! millisecond simulator clock:
//!
//! - timestamps and durations: milliseconds;
//! - `volatility`: absolute price units per √year, with a 365 day year;
//! - `intensity_scale_a`: fills per second, at distance zero;
//! - `order_intensity_k`: 1 / price units, in `λ(δ) = A · exp(-k · δ) · s(t)`;
//! - `s(t)`: dimensionless multiplier per bucket of the UTC day, so timestamps
//!   are milliseconds since the Unix epoch when seasonality is used.
//!
//! # Causality
//!
//! A sample belongs to a window by its timestamp, and an exposure is stamped
//! when it ends, so its fills are known by then. [`calibrate_window`] reads
//! only the samples inside the training window. The final partial grid bucket
//! of a window is dropped rather than completed with later data.
//!
//! # Insufficient data
//!
//! A window that cannot identify the parameters gives a record with
//! [`CalibrationOutcome::Insufficient`]. No default value is substituted, and
//! a later window never inherits parameters: that fold is left unscored.
//!
//! # Intensity fit
//!
//! `A`, `k` and `s(t)` are the joint Poisson maximum likelihood estimates, so
//! exposures without fills count as evidence of a low rate, distances may be
//! continuous, and `k` is not confused with a bucket that quotes further away. The existing log-linear [`OrderIntensityCalibrator`] cannot
//! take a level without fills, and does not return `A`.
//!
//! [`OrderIntensityCalibrator`]: crate::strategy::calibration::OrderIntensityCalibrator
//!
//! # Example
//!
//! ```
//! use market_maker_rs::strategy::walk_forward::{WindowSpec, chronological_splits};
//!
//! let spec = WindowSpec { train_ms: 2_000, test_ms: 1_000, step_ms: 1_000 };
//! let splits = chronological_splits(0, 4_000, &spec).unwrap();
//! assert_eq!(splits.len(), 2);
//! assert!(splits.iter().all(|s| s.train.end_ms == s.test.start_ms));
//! ```

use std::collections::{BTreeMap, BTreeSet};

use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

use crate::Decimal;
use crate::market_state::volatility::VolatilityEstimator;
use crate::types::decimal::{OrOverflow, decimal_sqrt};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Milliseconds in the 365 day year used for the volatility.
const MS_PER_YEAR: u64 = 31_536_000_000;
/// Milliseconds in a UTC day.
const MS_PER_DAY: u64 = 86_400_000;
/// Milliseconds per second.
const MS_PER_SECOND: u64 = 1_000;
/// Largest resampling grid accepted for one window.
const MAX_GRID_POINTS: u64 = 10_000_000;
/// Fewest grid prices that give a sample variance of the returns.
const MIN_GRID_PRICES: usize = 3;
/// Fewest distance levels that identify a slope.
const MIN_DISTANCE_LEVELS: usize = 2;
/// Most intraday buckets accepted: one per minute.
const MAX_SEASONALITY_BUCKETS: u32 = 1_440;
/// Bisection steps of the `k` fit; each halves the bracket.
const K_BISECTION_STEPS: u32 = 200;
/// Upper bound of the `k` bracket, in 1 / price units.
const K_BRACKET_LIMIT: f64 = 1e15;

/// Half-open time interval `[start_ms, end_ms)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeWindow {
    /// First millisecond inside the window.
    pub start_ms: u64,
    /// First millisecond after the window.
    pub end_ms: u64,
}

impl TimeWindow {
    /// Creates a window.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] when `start_ms >= end_ms`.
    pub fn new(start_ms: u64, end_ms: u64) -> MMResult<Self> {
        if start_ms >= end_ms {
            return Err(MMError::InvalidConfiguration(format!(
                "window start {start_ms} must precede end {end_ms}"
            )));
        }
        Ok(Self { start_ms, end_ms })
    }
}

/// A training window and the test window that follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChronologicalSplit {
    /// Window the parameters may be estimated from.
    pub train: TimeWindow,
    /// Window the parameters are evaluated on; it starts at `train.end_ms`.
    pub test: TimeWindow,
}

/// Lengths of the walk-forward windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct WindowSpec {
    /// Training window length. Positive.
    pub train_ms: u64,
    /// Test window length. Positive; one day gives a report per day.
    pub test_ms: u64,
    /// Distance between folds. At least `test_ms`, so tests do not overlap.
    pub step_ms: u64,
}

/// Builds the rolling splits of `[start_ms, end_ms)`.
///
/// Only complete test windows are returned. To protect a holdout, pass its
/// start as `end_ms`: no window then reaches it.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a zero length, for
/// `step_ms < test_ms`, or when not even one split fits, and
/// [`MMError::NumericalError`] on overflow.
pub fn chronological_splits(
    start_ms: u64,
    end_ms: u64,
    spec: &WindowSpec,
) -> MMResult<Vec<ChronologicalSplit>> {
    if spec.train_ms == 0 || spec.test_ms == 0 || spec.step_ms < spec.test_ms {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid windows: train {} test {} step {}",
            spec.train_ms, spec.test_ms, spec.step_ms
        )));
    }
    let mut splits = Vec::new();
    let mut train_start = start_ms;
    loop {
        let train_end = train_start
            .checked_add(spec.train_ms)
            .or_overflow("train window end")?;
        let test_end = train_end
            .checked_add(spec.test_ms)
            .or_overflow("test window end")?;
        if test_end > end_ms {
            break;
        }
        splits.push(ChronologicalSplit {
            train: TimeWindow::new(train_start, train_end)?,
            test: TimeWindow::new(train_end, test_end)?,
        });
        train_start = train_start
            .checked_add(spec.step_ms)
            .or_overflow("next train window")?;
    }
    if splits.is_empty() {
        return Err(MMError::InvalidConfiguration(format!(
            "no split fits in [{start_ms}, {end_ms})"
        )));
    }
    Ok(splits)
}

/// An observed price.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceSample {
    /// Observation time in milliseconds.
    pub timestamp_ms: u64,
    /// Price. Positive.
    pub price: Decimal,
}

/// Time a quote rested at a distance from the reference price, and its fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuoteExposure {
    /// Time the exposure ended, in milliseconds.
    pub timestamp_ms: u64,
    /// Distance from the reference price, in price units. Not negative.
    pub distance: Decimal,
    /// Time at risk in milliseconds. Positive.
    pub exposure_ms: u64,
    /// Fills during the exposure.
    pub fills: u64,
}

/// Validated, chronologically ordered calibration data.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationData<'a> {
    prices: &'a [PriceSample],
    exposures: &'a [QuoteExposure],
}

impl<'a> CalibrationData<'a> {
    /// Validates the samples.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for a non positive price, a
    /// negative distance, a zero exposure, or timestamps that go back.
    pub fn new(prices: &'a [PriceSample], exposures: &'a [QuoteExposure]) -> MMResult<Self> {
        if let Some(bad) = prices.iter().find(|s| s.price <= Decimal::ZERO) {
            return Err(MMError::InvalidMarketState(format!(
                "non positive price {} at {}",
                bad.price, bad.timestamp_ms
            )));
        }
        if let Some(bad) = exposures
            .iter()
            .find(|e| e.distance < Decimal::ZERO || e.exposure_ms == 0)
        {
            return Err(MMError::InvalidMarketState(format!(
                "invalid exposure at {}: distance {}, {} ms",
                bad.timestamp_ms, bad.distance, bad.exposure_ms
            )));
        }
        if !prices.is_sorted_by_key(|s| s.timestamp_ms)
            || !exposures.is_sorted_by_key(|e| e.timestamp_ms)
        {
            return Err(MMError::InvalidMarketState(
                "samples are not in chronological order".to_string(),
            ));
        }
        Ok(Self { prices, exposures })
    }

    fn prices_in(&self, window: &TimeWindow) -> &'a [PriceSample] {
        let from = self
            .prices
            .partition_point(|s| s.timestamp_ms < window.start_ms);
        let to = self
            .prices
            .partition_point(|s| s.timestamp_ms < window.end_ms);
        self.prices.get(from..to).unwrap_or_default()
    }

    fn exposures_in(&self, window: &TimeWindow) -> &'a [QuoteExposure] {
        let from = self
            .exposures
            .partition_point(|e| e.timestamp_ms < window.start_ms);
        let to = self
            .exposures
            .partition_point(|e| e.timestamp_ms < window.end_ms);
        self.exposures.get(from..to).unwrap_or_default()
    }
}

/// Estimator of the relative volatility inside a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum VolatilityRule {
    /// Sample standard deviation of the log returns.
    Simple,
    /// Exponentially weighted variance with this decay, in `(0, 1)`.
    Ewma {
        /// Decay factor.
        lambda: Decimal,
    },
}

/// When the parameters used out of sample are re-estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UpdateRule {
    /// Calibrate on the first training window and never again.
    Frozen,
    /// Calibrate on each fold's own training window.
    Rolling,
}

/// Calibration settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[non_exhaustive]
pub struct WalkForwardConfig {
    /// Resampling grid of the prices, in milliseconds. Positive.
    pub resolution_ms: u64,
    /// Volatility estimator.
    pub volatility_rule: VolatilityRule,
    /// Update rule of [`walk_forward`].
    pub update_rule: UpdateRule,
    /// Fewest grid prices for a volatility. At least 3.
    pub min_grid_prices: usize,
    /// Fewest distance levels with fills for an intensity. At least 2.
    pub min_distance_levels: usize,
    /// Fewest fills for an intensity. Positive.
    pub min_fills: u64,
    /// Least total exposure for an intensity, in milliseconds. Positive.
    pub min_exposure_ms: u64,
    /// Intraday buckets of the seasonality, from 1 to 1440.
    pub seasonality_buckets: u32,
    /// Fewest fills every bucket needs. With a lower count the window is
    /// [`CalibrationFailure::ThinSeasonalBucket`]. Zero accepts a bucket with
    /// exposure and no fills, whose multiplier is then 0. A bucket with no
    /// exposure is always [`CalibrationFailure::UnobservedSeasonalBucket`].
    pub min_bucket_fills: u64,
}

impl WalkForwardConfig {
    /// Creates a configuration with a one second grid, the simple estimator,
    /// rolling updates, one seasonal bucket and minima of 30 grid prices, 2
    /// levels, 30 fills and one minute of exposure.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resolution_ms: MS_PER_SECOND,
            volatility_rule: VolatilityRule::Simple,
            update_rule: UpdateRule::Rolling,
            min_grid_prices: 30,
            min_distance_levels: MIN_DISTANCE_LEVELS,
            min_fills: 30,
            min_exposure_ms: 60_000,
            seasonality_buckets: 1,
            min_bucket_fills: 30,
        }
    }

    /// Sets the resampling grid.
    #[must_use]
    pub fn with_resolution_ms(mut self, resolution_ms: u64) -> Self {
        self.resolution_ms = resolution_ms;
        self
    }

    /// Sets the volatility estimator.
    #[must_use]
    pub fn with_volatility_rule(mut self, rule: VolatilityRule) -> Self {
        self.volatility_rule = rule;
        self
    }

    /// Sets the update rule.
    #[must_use]
    pub fn with_update_rule(mut self, rule: UpdateRule) -> Self {
        self.update_rule = rule;
        self
    }

    /// Sets the intraday seasonality buckets and their minimum fills.
    #[must_use]
    pub fn with_seasonality(mut self, buckets: u32, min_bucket_fills: u64) -> Self {
        self.seasonality_buckets = buckets;
        self.min_bucket_fills = min_bucket_fills;
        self
    }

    /// Sets the data minima.
    #[must_use]
    pub fn with_minima(
        mut self,
        grid_prices: usize,
        distance_levels: usize,
        fills: u64,
        exposure_ms: u64,
    ) -> Self {
        self.min_grid_prices = grid_prices;
        self.min_distance_levels = distance_levels;
        self.min_fills = fills;
        self.min_exposure_ms = exposure_ms;
        self
    }

    fn validate(&self) -> MMResult<()> {
        let lambda_ok = match self.volatility_rule {
            VolatilityRule::Simple => true,
            VolatilityRule::Ewma { lambda } => lambda > Decimal::ZERO && lambda < Decimal::ONE,
        };
        if self.resolution_ms == 0
            || !lambda_ok
            || self.min_grid_prices < MIN_GRID_PRICES
            || self.min_distance_levels < MIN_DISTANCE_LEVELS
            || self.min_fills == 0
            || self.min_exposure_ms == 0
            || self.seasonality_buckets == 0
            || self.seasonality_buckets > MAX_SEASONALITY_BUCKETS
        {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid walk-forward configuration: {self:?}"
            )));
        }
        Ok(())
    }
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters estimated from a training window.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct CalibratedParameters {
    /// Absolute price volatility per √year.
    pub volatility: Decimal,
    /// Last training price, which turns the relative volatility into absolute.
    pub reference_price: Decimal,
    /// `A`, fills per second at distance zero.
    pub intensity_scale_a: Decimal,
    /// `k`, 1 / price units.
    pub order_intensity_k: Decimal,
    /// Intraday multipliers `s(t)`, one per bucket of the UTC day.
    pub seasonality: Vec<Decimal>,
}

impl CalibratedParameters {
    /// Returns `λ(δ) = A · exp(-k · δ) · s(t)` in fills per second. A rate
    /// below the resolution of `Decimal` reads as zero.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] when the rate is not representable.
    pub fn rate_per_second(&self, distance: Decimal, timestamp_ms: u64) -> MMResult<Decimal> {
        let base = intensity(self.intensity_scale_a, self.order_intensity_k, distance)?;
        let buckets = u32::try_from(self.seasonality.len())
            .ok()
            .filter(|b| *b > 0)
            .ok_or_else(|| MMError::NumericalError("no seasonality buckets".to_string()))?;
        let multiplier = self
            .seasonality
            .get(bucket_of(timestamp_ms, buckets)?)
            .copied()
            .or_overflow("seasonality bucket")?;
        base.checked_mul(multiplier)
            .or_overflow("seasonal fill rate")
    }
}

/// Why a window could not identify the parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum CalibrationFailure {
    /// Too few grid prices.
    TooFewPrices {
        /// Grid prices available.
        have: usize,
        /// Grid prices required.
        need: usize,
    },
    /// The prices did not move, so the volatility is zero.
    FlatPrices,
    /// Too little time at risk.
    TooLittleExposure {
        /// Milliseconds available.
        have_ms: u64,
        /// Milliseconds required.
        need_ms: u64,
    },
    /// Too few fills.
    TooFewFills {
        /// Fills available.
        have: u64,
        /// Fills required.
        need: u64,
    },
    /// Too few distinct distances with fills.
    TooFewDistanceLevels {
        /// Levels available.
        have: usize,
        /// Levels required.
        need: usize,
    },
    /// The fitted rate does not fall with distance, so `k` is not positive.
    NonDecayingIntensity,
    /// The likelihood has no usable `k`: each bucket's fills are all at its
    /// nearest distance, so no finite `k` exists, or the estimate is above the
    /// search limit of 1e15 per price unit.
    UnboundedIntensity,
    /// An intraday bucket has no exposure, so nothing is known about its rate.
    UnobservedSeasonalBucket {
        /// Bucket index.
        bucket: usize,
    },
    /// An intraday bucket has too few fills for a multiplier.
    ThinSeasonalBucket {
        /// Bucket index.
        bucket: usize,
        /// Fills available.
        have: u64,
        /// Fills required.
        need: u64,
    },
}

/// Result of calibrating one window.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum CalibrationOutcome {
    /// The window identified the parameters.
    Calibrated(CalibratedParameters),
    /// The window did not; there are no parameters.
    Insufficient(CalibrationFailure),
}

/// Parameters together with what they were estimated from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct CalibrationRecord {
    /// Training interval.
    pub training: TimeWindow,
    /// Resampling grid of the prices, in milliseconds.
    pub resolution_ms: u64,
    /// Caller's name for the dataset.
    pub dataset_id: String,
    /// FNV-1a 128 of the samples inside the training interval.
    pub dataset_hash: u128,
    /// Price samples inside the interval.
    pub price_samples: usize,
    /// Fills inside the interval.
    pub fills: u64,
    /// Exposure inside the interval, in milliseconds.
    pub exposure_ms: u64,
    /// Estimated parameters, or why there are none.
    pub outcome: CalibrationOutcome,
}

impl CalibrationRecord {
    /// Returns the parameters, if the window identified them.
    #[must_use]
    pub fn parameters(&self) -> Option<&CalibratedParameters> {
        match &self.outcome {
            CalibrationOutcome::Calibrated(parameters) => Some(parameters),
            CalibrationOutcome::Insufficient(_) => None,
        }
    }
}

/// Calibrates from the samples inside `window` and nothing else.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid configuration or
/// an oversized grid, and [`MMError::NumericalError`] on arithmetic failure.
/// Lack of data is not an error: see [`CalibrationOutcome::Insufficient`].
pub fn calibrate_window(
    data: &CalibrationData<'_>,
    window: &TimeWindow,
    config: &WalkForwardConfig,
    dataset_id: &str,
) -> MMResult<CalibrationRecord> {
    config.validate()?;
    let prices = data.prices_in(window);
    let exposures = data.exposures_in(window);
    let (fills, exposure_ms) = exposure_totals(exposures)?;
    let outcome = match estimate(prices, exposures, fills, exposure_ms, window, config)? {
        Ok(parameters) => CalibrationOutcome::Calibrated(parameters),
        Err(failure) => CalibrationOutcome::Insufficient(failure),
    };
    Ok(CalibrationRecord {
        training: *window,
        resolution_ms: config.resolution_ms,
        dataset_id: dataset_id.to_string(),
        dataset_hash: window_hash(prices, exposures),
        price_samples: prices.len(),
        fills,
        exposure_ms,
        outcome,
    })
}

/// Constant parameters the calibrated ones are compared against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConstantParameters {
    /// Absolute price volatility per √year.
    pub volatility: Decimal,
    /// `A`, fills per second at distance zero.
    pub intensity_scale_a: Decimal,
    /// `k`, 1 / price units.
    pub order_intensity_k: Decimal,
}

/// Out-of-sample errors of one set of parameters on one test window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct OutOfSampleError {
    /// `|predicted - realized|` volatility; `None` without test prices.
    pub volatility: Option<Decimal>,
    /// Sum over distances of `|fills - expected fills|` per second of
    /// exposure; `None` without test exposure.
    pub intensity: Option<Decimal>,
}

/// One walk-forward fold.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct FoldReport {
    /// Fold windows.
    pub split: ChronologicalSplit,
    /// Record used on the test window. Under [`UpdateRule::Frozen`] it is the
    /// first fold's.
    pub record: CalibrationRecord,
    /// Calibrated errors; `None` when the record has no parameters.
    pub calibrated: Option<OutOfSampleError>,
    /// Errors of the constant parameters.
    pub constant: OutOfSampleError,
    /// `|Δk|` against the previous fold with parameters.
    pub k_change: Option<Decimal>,
    /// `|Δvolatility|` against the previous fold with parameters.
    pub volatility_change: Option<Decimal>,
}

/// Walk-forward result, fold by fold.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct WalkForwardReport {
    /// Folds in chronological order.
    pub folds: Vec<FoldReport>,
}

impl WalkForwardReport {
    /// Returns the folds with at least one calibrated out-of-sample error.
    #[must_use]
    pub fn scored_folds(&self) -> usize {
        self.folds
            .iter()
            .filter(|f| {
                f.calibrated
                    .is_some_and(|e| e.volatility.is_some() || e.intensity.is_some())
            })
            .count()
    }
}

/// Runs `splits` under the configured update rule.
///
/// Calibration sees only each training window; the scoring of a fold reads
/// its test window through the finished record.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid configuration, no
/// splits, a test window that does not start at or after its training end, splits out
/// of chronological order (which would let frozen parameters score the past), or
/// constant parameters that are not positive; and [`MMError::NumericalError`]
/// on arithmetic failure.
pub fn walk_forward(
    data: &CalibrationData<'_>,
    splits: &[ChronologicalSplit],
    config: &WalkForwardConfig,
    dataset_id: &str,
    constant: &ConstantParameters,
) -> MMResult<WalkForwardReport> {
    config.validate()?;
    if constant.volatility <= Decimal::ZERO
        || constant.intensity_scale_a <= Decimal::ZERO
        || constant.order_intensity_k <= Decimal::ZERO
    {
        return Err(MMError::InvalidConfiguration(format!(
            "constant parameters must be positive: {constant:?}"
        )));
    }
    if splits.is_empty() {
        return Err(MMError::InvalidConfiguration("no splits".to_string()));
    }
    if let Some(bad) = splits.iter().find(|s| s.test.start_ms < s.train.end_ms) {
        return Err(MMError::InvalidConfiguration(format!(
            "test window overlaps its training window: {bad:?}"
        )));
    }
    if !splits.is_sorted_by_key(|s| s.test.start_ms) {
        return Err(MMError::InvalidConfiguration(
            "splits are not in chronological order".to_string(),
        ));
    }

    let mut folds: Vec<FoldReport> = Vec::with_capacity(splits.len());
    let mut frozen: Option<CalibrationRecord> = None;
    let mut previous: Option<CalibratedParameters> = None;
    for split in splits {
        let record = match (config.update_rule, &frozen) {
            (UpdateRule::Frozen, Some(record)) => record.clone(),
            (UpdateRule::Frozen, None) | (UpdateRule::Rolling, _) => {
                calibrate_window(data, &split.train, config, dataset_id)?
            }
        };
        if matches!(config.update_rule, UpdateRule::Frozen) && frozen.is_none() {
            frozen = Some(record.clone());
        }

        let prices = data.prices_in(&split.test);
        let exposures = data.exposures_in(&split.test);
        let realized = realized_volatility(prices, &split.test, config)?;
        let calibrated = record
            .parameters()
            .map(|p| {
                score(p.volatility, realized, exposures, |d, t| {
                    p.rate_per_second(d, t)
                })
            })
            .transpose()?;
        let constant_error = score(constant.volatility, realized, exposures, |d, _| {
            intensity(constant.intensity_scale_a, constant.order_intensity_k, d)
        })?;

        let (k_change, volatility_change) = match (&previous, record.parameters()) {
            (Some(before), Some(now)) => (
                Some(abs_diff(now.order_intensity_k, before.order_intensity_k)?),
                Some(abs_diff(now.volatility, before.volatility)?),
            ),
            (None, _) | (_, None) => (None, None),
        };
        if let Some(parameters) = record.parameters() {
            previous = Some(parameters.clone());
        }
        folds.push(FoldReport {
            split: *split,
            record,
            calibrated,
            constant: constant_error,
            k_change,
            volatility_change,
        });
    }
    Ok(WalkForwardReport { folds })
}

/// Fitted intensity parameters.
struct IntensityFit {
    a: Decimal,
    k: Decimal,
    seasonality: Vec<Decimal>,
}

/// Estimates the parameters, or says why the window cannot.
fn estimate(
    prices: &[PriceSample],
    exposures: &[QuoteExposure],
    fills: u64,
    exposure_ms: u64,
    window: &TimeWindow,
    config: &WalkForwardConfig,
) -> MMResult<Result<CalibratedParameters, CalibrationFailure>> {
    let grid = grid_prices(prices, window, config.resolution_ms)?;
    if grid.len() < config.min_grid_prices {
        return Ok(Err(CalibrationFailure::TooFewPrices {
            have: grid.len(),
            need: config.min_grid_prices,
        }));
    }
    let Some(&reference_price) = grid.last() else {
        return Ok(Err(CalibrationFailure::TooFewPrices {
            have: 0,
            need: config.min_grid_prices,
        }));
    };
    let relative = relative_volatility(&grid, config.resolution_ms, config.volatility_rule)?;
    if relative.is_zero() {
        return Ok(Err(CalibrationFailure::FlatPrices));
    }
    let volatility = relative
        .checked_mul(reference_price)
        .or_overflow("absolute volatility")?;

    if exposure_ms < config.min_exposure_ms {
        return Ok(Err(CalibrationFailure::TooLittleExposure {
            have_ms: exposure_ms,
            need_ms: config.min_exposure_ms,
        }));
    }
    if fills < config.min_fills {
        return Ok(Err(CalibrationFailure::TooFewFills {
            have: fills,
            need: config.min_fills,
        }));
    }
    let fit = match fit_intensity(exposures, config)? {
        Ok(fit) => fit,
        Err(failure) => return Ok(Err(failure)),
    };
    Ok(Ok(CalibratedParameters {
        volatility,
        reference_price,
        intensity_scale_a: fit.a,
        order_intensity_k: fit.k,
        seasonality: fit.seasonality,
    }))
}

/// Resamples to the last price at or before each complete grid bucket's end.
fn grid_prices(
    prices: &[PriceSample],
    window: &TimeWindow,
    resolution_ms: u64,
) -> MMResult<Vec<Decimal>> {
    let span = window
        .end_ms
        .checked_sub(window.start_ms)
        .or_overflow("window length")?;
    let points = span.checked_div(resolution_ms).or_overflow("grid points")?;
    if points > MAX_GRID_POINTS {
        return Err(MMError::InvalidConfiguration(format!(
            "grid of {points} points exceeds {MAX_GRID_POINTS}"
        )));
    }
    let mut grid = Vec::new();
    let mut samples = prices.iter().peekable();
    let mut last = None;
    for point in 1..=points {
        let bucket_end = point
            .checked_mul(resolution_ms)
            .and_then(|offset| window.start_ms.checked_add(offset))
            .or_overflow("grid bucket end")?;
        while let Some(sample) = samples.next_if(|s| s.timestamp_ms < bucket_end) {
            last = Some(sample.price);
        }
        if let Some(price) = last {
            grid.push(price);
        }
    }
    Ok(grid)
}

/// Relative volatility per √year of a regular grid.
fn relative_volatility(
    grid: &[Decimal],
    resolution_ms: u64,
    rule: VolatilityRule,
) -> MMResult<Decimal> {
    let steps_per_year = Decimal::from(MS_PER_YEAR)
        .checked_div(Decimal::from(resolution_ms))
        .or_overflow("grid steps per year")?;
    let estimator = VolatilityEstimator::with_annualization_factor(decimal_sqrt(steps_per_year)?);
    match rule {
        VolatilityRule::Simple => estimator.calculate_simple(grid),
        VolatilityRule::Ewma { lambda } => estimator.calculate_ewma(grid, lambda),
    }
}

/// Absolute volatility realized on a test window, by the simple estimator
/// whatever the training rule. `None` when the window has too few grid prices.
fn realized_volatility(
    prices: &[PriceSample],
    window: &TimeWindow,
    config: &WalkForwardConfig,
) -> MMResult<Option<Decimal>> {
    let grid = grid_prices(prices, window, config.resolution_ms)?;
    let Some(&last) = grid.last().filter(|_| grid.len() >= config.min_grid_prices) else {
        return Ok(None);
    };
    relative_volatility(&grid, config.resolution_ms, VolatilityRule::Simple)?
        .checked_mul(last)
        .or_overflow("realized volatility")
        .map(Some)
}

/// Fills and exposure, summed.
fn exposure_totals(exposures: &[QuoteExposure]) -> MMResult<(u64, u64)> {
    exposures.iter().try_fold((0u64, 0u64), |(fills, ms), e| {
        Ok((
            fills.checked_add(e.fills).or_overflow("total fills")?,
            ms.checked_add(e.exposure_ms)
                .or_overflow("total exposure")?,
        ))
    })
}

/// One intraday bucket of the fit.
#[derive(Default)]
struct BucketFit {
    /// Distance and seconds of exposure of each level.
    levels: Vec<(f64, f64)>,
    fills: f64,
}

impl BucketFit {
    /// `Σ T · exp(-k · d)`.
    fn mass(&self, k: f64) -> f64 {
        self.levels.iter().map(|(d, t)| t * (-k * d).exp()).sum()
    }

    /// The bucket's expected fills times its mean distance under `k`, once
    /// its own multiplier is profiled out.
    fn fill_moment(&self, k: f64) -> f64 {
        let mass = self.mass(k);
        if self.fills == 0.0 || mass <= 0.0 {
            return 0.0;
        }
        let moment: f64 = self
            .levels
            .iter()
            .map(|(d, t)| t * (-k * d).exp() * d)
            .sum();
        self.fills * moment / mass
    }
}

/// Poisson maximum likelihood of `λ = c_b · exp(-k · δ)`, with a free `c_b`
/// per intraday bucket.
///
/// Each `c_b` is `n_b / Σ T · exp(-k · d)` given `k`. The likelihood in `k` is
/// then maximal where the observed sum of `n · d` equals the sum of
/// [`BucketFit::fill_moment`], which falls with `k`, so `k` is found by
/// bisection. Finally `A = Σ n / Σ T · exp(-k · d)` and `s_b = c_b / A`.
fn fit_intensity(
    exposures: &[QuoteExposure],
    config: &WalkForwardConfig,
) -> MMResult<Result<IntensityFit, CalibrationFailure>> {
    let mut cells: BTreeMap<(usize, Decimal), (u64, u64)> = BTreeMap::new();
    for e in exposures {
        let bucket = bucket_of(e.timestamp_ms, config.seasonality_buckets)?;
        let cell = cells.entry((bucket, e.distance)).or_default();
        cell.0 = cell.0.checked_add(e.fills).or_overflow("level fills")?;
        cell.1 = cell
            .1
            .checked_add(e.exposure_ms)
            .or_overflow("level exposure")?;
    }
    let distances_with_fills = cells
        .iter()
        .filter(|(_, (fills, _))| *fills > 0)
        .map(|((_, distance), _)| *distance)
        .collect::<BTreeSet<_>>()
        .len();
    if distances_with_fills < config.min_distance_levels {
        return Ok(Err(CalibrationFailure::TooFewDistanceLevels {
            have: distances_with_fills,
            need: config.min_distance_levels,
        }));
    }

    let bucket_count = usize::try_from(config.seasonality_buckets)
        .map_err(|_| MMError::InvalidConfiguration("too many buckets".to_string()))?;
    let mut buckets: Vec<BucketFit> = Vec::with_capacity(bucket_count);
    buckets.resize_with(bucket_count, BucketFit::default);
    let mut bucket_fills = vec![0u64; bucket_count];
    let mut observed = 0.0f64;
    for ((index, distance), (fills, exposure_ms)) in cells {
        let d = to_f64(distance)?;
        let n = to_f64(Decimal::from(fills))?;
        let seconds = to_f64(Decimal::from(exposure_ms))? / 1_000.0;
        let bucket = buckets.get_mut(index).or_overflow("seasonality bucket")?;
        bucket.levels.push((d, seconds));
        bucket.fills += n;
        let total = bucket_fills
            .get_mut(index)
            .or_overflow("seasonality bucket")?;
        *total = total.checked_add(fills).or_overflow("bucket fills")?;
        observed += n * d;
    }
    for (bucket, (fit, fills)) in buckets.iter().zip(&bucket_fills).enumerate() {
        if fit.levels.is_empty() {
            return Ok(Err(CalibrationFailure::UnobservedSeasonalBucket { bucket }));
        }
        if *fills < config.min_bucket_fills {
            return Ok(Err(CalibrationFailure::ThinSeasonalBucket {
                bucket,
                have: *fills,
                need: config.min_bucket_fills,
            }));
        }
    }

    let predicted = |k: f64| buckets.iter().map(|b| b.fill_moment(k)).sum::<f64>();
    if predicted(0.0) <= observed {
        return Ok(Err(CalibrationFailure::NonDecayingIntensity));
    }
    // As `k` grows, each bucket's mean distance falls to its nearest level. At
    // that limit rounding would stop the bisection at an arbitrary finite `k`.
    let limit: f64 = buckets
        .iter()
        .map(|b| b.fills * b.levels.first().map_or(0.0, |(d, _)| *d))
        .sum();
    if observed <= limit {
        return Ok(Err(CalibrationFailure::UnboundedIntensity));
    }
    let (mut low, mut high) = (0.0f64, 1.0f64);
    while predicted(high) > observed {
        low = high;
        high *= 2.0;
        if high > K_BRACKET_LIMIT {
            return Ok(Err(CalibrationFailure::UnboundedIntensity));
        }
    }
    for _ in 0..K_BISECTION_STEPS {
        let middle = 0.5 * (low + high);
        if predicted(middle) > observed {
            low = middle;
        } else {
            high = middle;
        }
    }
    let k = 0.5 * (low + high);
    let total_fills: f64 = buckets.iter().map(|b| b.fills).sum();
    let total_mass: f64 = buckets.iter().map(|b| b.mass(k)).sum();
    let a = total_fills / total_mass;
    let seasonality = buckets
        .iter()
        .map(|b| from_f64(b.fills / b.mass(k) / a))
        .collect::<MMResult<Vec<_>>>()?;
    Ok(Ok(IntensityFit {
        a: from_f64(a)?,
        k: from_f64(k)?,
        seasonality,
    }))
}

/// Scores a volatility and a rate function on a test window.
fn score(
    volatility: Decimal,
    realized: Option<Decimal>,
    exposures: &[QuoteExposure],
    rate: impl Fn(Decimal, u64) -> MMResult<Decimal>,
) -> MMResult<OutOfSampleError> {
    let volatility = realized.map(|r| abs_diff(volatility, r)).transpose()?;

    let mut levels: BTreeMap<Decimal, (u64, Decimal)> = BTreeMap::new();
    let mut total_ms = 0u64;
    for e in exposures {
        let expected = expected_fills(rate(e.distance, e.timestamp_ms)?, e.exposure_ms)?;
        let level = levels.entry(e.distance).or_default();
        level.0 = level.0.checked_add(e.fills).or_overflow("level fills")?;
        level.1 = level
            .1
            .checked_add(expected)
            .or_overflow("level expected")?;
        total_ms = total_ms
            .checked_add(e.exposure_ms)
            .or_overflow("test exposure")?;
    }
    let intensity = if total_ms == 0 {
        None
    } else {
        let mut gap = Decimal::ZERO;
        for (fills, expected) in levels.values() {
            gap = gap
                .checked_add(abs_diff(Decimal::from(*fills), *expected)?)
                .or_overflow("fill gap")?;
        }
        Some(seconds_rate(gap, total_ms)?)
    };
    Ok(OutOfSampleError {
        volatility,
        intensity,
    })
}

/// `count` per second of `exposure_ms`.
fn seconds_rate(count: Decimal, exposure_ms: u64) -> MMResult<Decimal> {
    count
        .checked_mul(Decimal::from(MS_PER_SECOND))
        .and_then(|scaled| scaled.checked_div(Decimal::from(exposure_ms)))
        .or_overflow("rate per second")
}

/// Fills a rate per second predicts over `exposure_ms`.
fn expected_fills(rate: Decimal, exposure_ms: u64) -> MMResult<Decimal> {
    rate.checked_mul(Decimal::from(exposure_ms))
        .and_then(|scaled| scaled.checked_div(Decimal::from(MS_PER_SECOND)))
        .or_overflow("expected fills")
}

/// `A · exp(-k · δ)`.
fn intensity(a: Decimal, k: Decimal, distance: Decimal) -> MMResult<Decimal> {
    let exponent = k
        .checked_mul(distance)
        .and_then(|v| Decimal::ZERO.checked_sub(v))
        .or_overflow("intensity exponent")?;
    a.checked_mul(decimal_exp(exponent)?)
        .or_overflow("fill intensity")
}

fn to_f64(value: Decimal) -> MMResult<f64> {
    value
        .to_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MMError::NumericalError(format!("{value} is not a finite f64")))
}

fn from_f64(value: f64) -> MMResult<Decimal> {
    Some(value)
        .filter(|v| v.is_finite())
        .and_then(Decimal::from_f64)
        .ok_or_else(|| MMError::NumericalError(format!("{value} is not a decimal")))
}

/// `exp(value)` through `f64`, as the other statistical estimators do.
fn decimal_exp(value: Decimal) -> MMResult<Decimal> {
    value
        .to_f64()
        .map(f64::exp)
        .filter(|v| v.is_finite())
        .and_then(Decimal::from_f64)
        .ok_or_else(|| MMError::NumericalError(format!("exp({value}) is not representable")))
}

/// Index of the intraday bucket of a timestamp. `buckets` is positive.
fn bucket_of(timestamp_ms: u64, buckets: u32) -> MMResult<usize> {
    let time_of_day = timestamp_ms
        .checked_rem(MS_PER_DAY)
        .or_overflow("time of day")?;
    let index = time_of_day
        .checked_mul(u64::from(buckets))
        .and_then(|v| v.checked_div(MS_PER_DAY))
        .or_overflow("intraday bucket")?;
    usize::try_from(index).map_err(|_| MMError::NumericalError(format!("bucket {index}")))
}

/// `|a - b|`.
fn abs_diff(a: Decimal, b: Decimal) -> MMResult<Decimal> {
    a.checked_sub(b).map(|d| d.abs()).or_overflow("difference")
}

/// FNV-1a 128 of the samples, independent of the scale of their decimals. It
/// is the digest of the `kernel` feature, which this module does not require.
fn window_hash(prices: &[PriceSample], exposures: &[QuoteExposure]) -> u128 {
    let mut hash = FNV_OFFSET;
    for s in prices {
        hash = fnv(hash, &s.timestamp_ms.to_le_bytes());
        hash = fnv(hash, &s.price.normalize().serialize());
    }
    for e in exposures {
        hash = fnv(hash, &e.timestamp_ms.to_le_bytes());
        hash = fnv(hash, &e.distance.normalize().serialize());
        hash = fnv(hash, &e.exposure_ms.to_le_bytes());
        hash = fnv(hash, &e.fills.to_le_bytes());
    }
    hash
}

const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

fn fnv(hash: u128, bytes: &[u8]) -> u128 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u128::from(*b)).wrapping_mul(FNV_PRIME))
}
