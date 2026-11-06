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
//! - `volatility`: absolute price units per √year, with a 365 day year. With
//!   declared sessions the year is of open time: see [Sessions](#sessions);
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
//! continuous, and `k` is not confused with a bucket that quotes further away.
//! An exposure counts entirely in the intraday bucket where it ends, so keep
//! exposures short against the buckets. [`OrderIntensityCalibrator`] and
//! [`OrderIntensityEstimator`] are not reused: both regress logarithms of
//! observed fills, so neither can use an exposure without fills.
//!
//! [`OrderIntensityCalibrator`]: crate::strategy::calibration::OrderIntensityCalibrator
//! [`OrderIntensityEstimator`]: crate::analytics::intensity::OrderIntensityEstimator
//!
//! # Price gaps
//!
//! A grid bucket without a sample repeats the last price, which reads as a
//! return of zero. So the grid minimum counts only buckets with a sample, and
//! a window with a longer silence than `max_gap_buckets` allows, including at
//! its start or end, is [`CalibrationFailure::PriceGap`].
//!
//! # Sessions
//!
//! By default the market is taken as always open, so a venue that stops
//! printing overnight fails a one day window with `PriceGap`, and raising
//! `max_gap_buckets` to pass it deflates the volatility with zero returns.
//! [`CalibrationData::with_sessions`] declares when the market is open instead.
//! Only the caller's calendar makes a silence a closure, so a feed that fails
//! during a session stays a `PriceGap` however long it lasts.
//!
//! With sessions, each one has its own grid, from its start or the window's,
//! and its own partial final bucket dropped. The gap limit applies inside
//! each session and no return crosses two of them. Therefore:
//!
//! - `volatility` is the diffusion while open, per √year of open time. Use it
//!   with a horizon in open time. It is not comparable to a close to close
//!   volatility, as the move across a closure is in neither the estimate nor
//!   the score, and inventory held through a closure bears that risk unpriced.
//! - The estimators pool the returns in order, so EWMA decays per return and
//!   not per unit of clock time across a closure.
//! - An intraday bucket with no exposure that the window reaches only while
//!   closed gets a multiplier of 0, not a failure. `A` is then per second of
//!   time at risk in the other buckets. Such a bucket predicts no fills, also
//!   if the market is open then later, such as training on a half day and
//!   testing on a full one. The score's floor makes that a large, visible cost.
//!   A fit therefore holds only for the same intraday calendar: a daylight
//!   saving change moves a session in UTC and has the same effect.
//! - An exposure stamped exactly at a session's end counts in the session's
//!   last intraday bucket, in training and in scoring, as a quote resting until
//!   the close is not at risk once it has closed. If the next session starts
//!   at that instant, it counts in that session.
//! - [`CalibrationRecord::open_ms`] says how much of the window was open.
//!
//! # Forecasts out of support
//!
//! A fit extrapolates to distances it never saw, and `A · s(t)` can pass the
//! range of `Decimal`. [`walk_forward`] then reports that fold's intensity as
//! out of support, in [`OutOfSampleError::intensity_out_of_support`], and keeps
//! the other folds. The score is left out rather than saturated, since a
//! capped rate would read as a plausible number.
//!
//! # In the replay
//!
//! [`crate::backtest::calibrated`] quotes the Avellaneda-Stoikov baseline from a
//! [`WalkForwardReport`] in the causal replay, each fold's parameters holding only
//! during its test window.
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
use crate::market_state::volatility::{VolatilityEstimator, consecutive_log_returns};
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
/// Fewest returns that give a sample variance.
const MIN_RETURNS: usize = 2;
/// Most splits accepted from one call.
const MAX_SPLITS: usize = 1_000_000;
/// Relative margin under which the fit cannot tell a difference from rounding.
const FIT_TOLERANCE: f64 = 1e-12;
/// Fills per second assumed where the predicted rate is lower, so that a fill
/// the model calls impossible costs a large but finite score.
const SCORE_RATE_FLOOR: f64 = 1e-12;
/// Smallest rate kept; below it the rate is under `Decimal` resolution.
const RATE_ZERO_BELOW: f64 = 1e-28;
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
/// `step_ms < test_ms`, more than a million splits, or when not even one
/// split fits, and
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
        if splits.len() >= MAX_SPLITS {
            return Err(MMError::InvalidConfiguration(format!(
                "more than {MAX_SPLITS} splits in [{start_ms}, {end_ms})"
            )));
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
    /// Declared sessions, ordered and apart. Empty means always open.
    sessions: &'a [TimeWindow],
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
        Ok(Self {
            prices,
            exposures,
            sessions: &[],
        })
    }

    /// Declares the half-open intervals when the market is open, for a venue
    /// that closes. See [Sessions](self#sessions). Exposures are not checked
    /// against them, as they already say how long the quotes were at risk.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for no sessions, an empty
    /// session, or sessions that overlap or go back; and
    /// [`MMError::InvalidMarketState`] for a price sample outside every session,
    /// which contradicts the declaration, and is not dropped silently.
    ///
    /// # Examples
    ///
    /// ```
    /// use market_maker_rs::dec;
    /// use market_maker_rs::strategy::walk_forward::{CalibrationData, PriceSample, TimeWindow};
    ///
    /// let sessions = [TimeWindow::new(1_000, 2_000).unwrap(), TimeWindow::new(5_000, 6_000).unwrap()];
    /// let prices = [PriceSample { timestamp_ms: 1_000, price: dec!(100) }];
    /// assert!(CalibrationData::new(&prices, &[]).unwrap().with_sessions(&sessions).is_ok());
    ///
    /// let closed = [PriceSample { timestamp_ms: 3_000, price: dec!(100) }];
    /// assert!(CalibrationData::new(&closed, &[]).unwrap().with_sessions(&sessions).is_err());
    /// ```
    pub fn with_sessions(mut self, sessions: &'a [TimeWindow]) -> MMResult<Self> {
        if sessions.is_empty() {
            return Err(MMError::InvalidConfiguration(
                "no sessions declared".to_string(),
            ));
        }
        if let Some(bad) = sessions.iter().find(|s| s.start_ms >= s.end_ms) {
            return Err(MMError::InvalidConfiguration(format!(
                "empty session: {bad:?}"
            )));
        }
        if let Some(bad) = sessions
            .windows(2)
            .find(|pair| matches!(pair, [a, b] if b.start_ms < a.end_ms))
        {
            return Err(MMError::InvalidConfiguration(format!(
                "sessions overlap or go back: {bad:?}"
            )));
        }
        let mut open = sessions.iter().peekable();
        for sample in self.prices {
            while open.next_if(|s| s.end_ms <= sample.timestamp_ms).is_some() {}
            if !open
                .peek()
                .is_some_and(|s| s.start_ms <= sample.timestamp_ms)
            {
                return Err(MMError::InvalidMarketState(format!(
                    "price at {} is outside every session",
                    sample.timestamp_ms
                )));
            }
        }
        self.sessions = sessions;
        Ok(self)
    }

    fn prices_in(&self, window: &TimeWindow) -> &'a [PriceSample] {
        samples_in(self.prices, window)
    }

    /// The parts of `window` when the market is open, in order.
    fn open_parts(&self, window: &TimeWindow) -> Vec<TimeWindow> {
        if self.sessions.is_empty() {
            return vec![*window];
        }
        let from = self
            .sessions
            .partition_point(|s| s.end_ms <= window.start_ms);
        let to = self
            .sessions
            .partition_point(|s| s.start_ms < window.end_ms);
        self.sessions
            .get(from..to)
            .unwrap_or_default()
            .iter()
            .map(|s| TimeWindow {
                start_ms: s.start_ms.max(window.start_ms),
                end_ms: s.end_ms.min(window.end_ms),
            })
            .collect()
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
    /// Fewest grid buckets with a sample for a volatility. At least 3.
    pub min_grid_prices: usize,
    /// Longest silence between samples of a session, in grid buckets. Positive.
    pub max_gap_buckets: u32,
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
    /// exposure is [`CalibrationFailure::UnobservedSeasonalBucket`], unless
    /// declared sessions show it closed.
    pub min_bucket_fills: u64,
}

impl WalkForwardConfig {
    /// Creates a configuration with a one second grid, the simple estimator,
    /// rolling updates, one seasonal bucket, a silence of at most 5 grid
    /// buckets and minima of 30 grid prices, 2 levels, 30 fills and one minute
    /// of exposure.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resolution_ms: MS_PER_SECOND,
            volatility_rule: VolatilityRule::Simple,
            update_rule: UpdateRule::Rolling,
            min_grid_prices: 30,
            max_gap_buckets: 5,
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

    /// Sets the longest silence between samples, in grid buckets.
    #[must_use]
    pub fn with_max_gap_buckets(mut self, buckets: u32) -> Self {
        self.max_gap_buckets = buckets;
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
            || self.max_gap_buckets == 0
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
    /// below 1e-28, the resolution of `Decimal`, reads as zero, and so does a
    /// time in an intraday bucket that was closed in training, whether or not
    /// the market is open at `timestamp_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] when the rate is above the range of
    /// `Decimal`, or on arithmetic failure.
    pub fn rate_per_second(&self, distance: Decimal, timestamp_ms: u64) -> MMResult<Decimal> {
        let rate = self.rate_f64(distance, timestamp_ms)?;
        representable_rate(rate)
            .ok_or_else(|| MMError::NumericalError(format!("{rate} is not a decimal")))
    }

    /// `λ(δ)` before any rounding to `Decimal`.
    fn rate_f64(&self, distance: Decimal, timestamp_ms: u64) -> MMResult<f64> {
        let base = intensity_f64(self.intensity_scale_a, self.order_intensity_k, distance)?;
        let buckets = u32::try_from(self.seasonality.len())
            .ok()
            .filter(|b| *b > 0)
            .ok_or_else(|| MMError::NumericalError("no seasonality buckets".to_string()))?;
        let multiplier = self
            .seasonality
            .get(bucket_of(timestamp_ms, buckets)?)
            .copied()
            .or_overflow("seasonality bucket")?;
        Ok(base * to_f64(multiplier)?)
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
    /// Enough grid prices, but too few returns inside sessions.
    TooFewReturns {
        /// Returns available.
        have: usize,
        /// Returns required.
        need: usize,
    },
    /// A silence in the prices is longer than allowed.
    PriceGap {
        /// Longest silence in the window, in milliseconds.
        gap_ms: u64,
        /// Longest silence allowed, in milliseconds.
        limit_ms: u64,
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
    /// `A` or a multiplier from the fit is not representable as a `Decimal`.
    UnrepresentableIntensity,
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
    /// FNV-1a 128 of the samples inside the training interval and, with
    /// sessions, of the sessions clipped to it and the intraday time each
    /// exposure counts at.
    pub dataset_hash: u128,
    /// Price samples inside the interval.
    pub price_samples: usize,
    /// Fills inside the interval.
    pub fills: u64,
    /// Exposure inside the interval, in milliseconds.
    pub exposure_ms: u64,
    /// Time the market was declared open inside the interval, in milliseconds.
    /// When set, `volatility` is per √year of open time. `None` when no sessions
    /// were declared, as in a record from before they existed: the market is
    /// taken as always open, and the year is of clock time.
    pub open_ms: Option<u64>,
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
    let parts = data.open_parts(window);
    let closed = closed_buckets(window, &parts, config.seasonality_buckets)?;
    let totals = exposure_totals(exposures)?;
    let (fills, exposure_ms) = totals;
    let outcome = match estimate(
        prices,
        exposures,
        totals,
        &parts,
        data.sessions,
        &closed,
        config,
    )? {
        Ok(parameters) => CalibrationOutcome::Calibrated(parameters),
        Err(failure) => CalibrationOutcome::Insufficient(failure),
    };
    Ok(CalibrationRecord {
        training: *window,
        resolution_ms: config.resolution_ms,
        dataset_id: dataset_id.to_string(),
        dataset_hash: window_hash(prices, exposures, &parts, data.sessions),
        price_samples: prices.len(),
        fills,
        exposure_ms,
        open_ms: if data.sessions.is_empty() {
            None
        } else {
            Some(open_time_ms(&parts)?)
        },
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
    /// `|predicted - realized|` volatility; `None` when the test prices would
    /// not pass the training minimum or gap limit.
    pub volatility: Option<Decimal>,
    /// Poisson negative log likelihood of the test fills per second of
    /// exposure, `Σ (μ - n · ln μ) / Σ T`, without the `ln(n!)` term that
    /// depends only on the data. Lower is better, it may be negative, and it
    /// compares models on one window only. A predicted rate below 1e-12 fills
    /// per second counts as 1e-12, so a fill at rate zero costs a large finite
    /// amount. `None` without test exposure, or when
    /// [`Self::intensity_out_of_support`] is set.
    pub intensity: Option<Decimal>,
    /// Set when a predicted rate or their score is above the range of `Decimal`,
    /// which leaves the window without an intensity score. It names the test
    /// exposure with the highest predicted rate, the earliest on a tie.
    pub intensity_out_of_support: Option<OutOfSupportForecast>,
}

/// The test exposure with the highest predicted rate in a window that cannot
/// be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct OutOfSupportForecast {
    /// Time the exposure ended, in milliseconds.
    pub timestamp_ms: u64,
    /// Distance from the reference price, in price units.
    pub distance: Decimal,
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

    /// Returns the folds whose calibrated intensity forecast is out of support.
    #[must_use]
    pub fn out_of_support_folds(&self) -> usize {
        self.folds
            .iter()
            .filter(|f| {
                f.calibrated
                    .is_some_and(|e| e.intensity_out_of_support.is_some())
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
        let realized = realized_volatility(prices, &data.open_parts(&split.test), config)?;
        let calibrated = record
            .parameters()
            .map(|p| {
                score(p.volatility, realized, exposures, |d, t| {
                    p.rate_f64(d, intraday_time(data.sessions, t))
                })
            })
            .transpose()?;
        let constant_error = score(constant.volatility, realized, exposures, |d, _| {
            intensity_f64(constant.intensity_scale_a, constant.order_intensity_k, d)
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
    (fills, exposure_ms): (u64, u64),
    parts: &[TimeWindow],
    sessions: &[TimeWindow],
    closed: &[bool],
    config: &WalkForwardConfig,
) -> MMResult<Result<CalibratedParameters, CalibrationFailure>> {
    let grid = grid_prices(prices, parts, config.resolution_ms)?;
    if grid.observed < config.min_grid_prices {
        return Ok(Err(CalibrationFailure::TooFewPrices {
            have: grid.observed,
            need: config.min_grid_prices,
        }));
    }
    let limit_ms = gap_limit_ms(config)?;
    let gap_ms = longest_gap_ms(prices, parts)?;
    if gap_ms > limit_ms {
        return Ok(Err(CalibrationFailure::PriceGap { gap_ms, limit_ms }));
    }
    let returns = grid.log_returns()?;
    if returns.len() < MIN_RETURNS {
        return Ok(Err(CalibrationFailure::TooFewReturns {
            have: returns.len(),
            need: MIN_RETURNS,
        }));
    }
    let Some(reference_price) = grid.last_price() else {
        return Ok(Err(CalibrationFailure::TooFewPrices {
            have: grid.observed,
            need: config.min_grid_prices,
        }));
    };
    let relative = relative_volatility(&returns, config.resolution_ms, config.volatility_rule)?;
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
    let fit = match fit_intensity(exposures, sessions, closed, config)? {
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

/// Grid prices of a window.
struct PriceGrid {
    /// One run of prices per session with any; no return crosses two runs.
    runs: Vec<Vec<Decimal>>,
    /// Buckets with a sample of their own.
    observed: usize,
}

impl PriceGrid {
    /// Log returns inside each run, in order.
    fn log_returns(&self) -> MMResult<Vec<Decimal>> {
        let mut returns = Vec::new();
        for run in &self.runs {
            returns.extend(consecutive_log_returns(run)?);
        }
        Ok(returns)
    }

    /// The last grid price.
    fn last_price(&self) -> Option<Decimal> {
        self.runs.last().and_then(|run| run.last()).copied()
    }
}

/// Resamples each part to the last price before each complete grid bucket's
/// end, and counts the buckets with a sample of their own.
fn grid_prices(
    prices: &[PriceSample],
    parts: &[TimeWindow],
    resolution_ms: u64,
) -> MMResult<PriceGrid> {
    let mut points = Vec::with_capacity(parts.len());
    let mut total = 0u64;
    for part in parts {
        let span = part
            .end_ms
            .checked_sub(part.start_ms)
            .or_overflow("window length")?;
        let part_points = span.checked_div(resolution_ms).or_overflow("grid points")?;
        total = total.checked_add(part_points).or_overflow("grid points")?;
        points.push(part_points);
    }
    if total > MAX_GRID_POINTS {
        return Err(MMError::InvalidConfiguration(format!(
            "grid of {total} points exceeds {MAX_GRID_POINTS}"
        )));
    }
    let mut grid = PriceGrid {
        runs: Vec::new(),
        observed: 0,
    };
    for (part, part_points) in parts.iter().zip(points) {
        let mut run = Vec::new();
        let mut samples = samples_in(prices, part).iter().peekable();
        let mut last = None;
        for point in 1..=part_points {
            let bucket_end = point
                .checked_mul(resolution_ms)
                .and_then(|offset| part.start_ms.checked_add(offset))
                .or_overflow("grid bucket end")?;
            let mut sampled = false;
            while let Some(sample) = samples.next_if(|s| s.timestamp_ms < bucket_end) {
                last = Some(sample.price);
                sampled = true;
            }
            if sampled {
                grid.observed = grid.observed.saturating_add(1);
            }
            if let Some(price) = last {
                run.push(price);
            }
        }
        if !run.is_empty() {
            grid.runs.push(run);
        }
    }
    Ok(grid)
}

/// The samples inside `window`. `prices` is in chronological order.
fn samples_in<'a>(prices: &'a [PriceSample], window: &TimeWindow) -> &'a [PriceSample] {
    let from = prices.partition_point(|s| s.timestamp_ms < window.start_ms);
    let to = prices.partition_point(|s| s.timestamp_ms < window.end_ms);
    prices.get(from..to).unwrap_or_default()
}

/// Longest silence allowed by the configuration.
fn gap_limit_ms(config: &WalkForwardConfig) -> MMResult<u64> {
    config
        .resolution_ms
        .checked_mul(u64::from(config.max_gap_buckets))
        .or_overflow("price gap limit")
}

/// Longest time without a sample inside a part, counting from each part's
/// start to its end.
fn longest_gap_ms(prices: &[PriceSample], parts: &[TimeWindow]) -> MMResult<u64> {
    let mut longest = 0u64;
    for part in parts {
        let mut previous = part.start_ms;
        let times = samples_in(prices, part)
            .iter()
            .map(|s| s.timestamp_ms)
            .chain(std::iter::once(part.end_ms));
        for time in times {
            let gap = time.checked_sub(previous).or_overflow("price gap")?;
            longest = longest.max(gap);
            previous = time;
        }
    }
    Ok(longest)
}

/// Relative volatility per √year of the returns of a regular grid.
fn relative_volatility(
    returns: &[Decimal],
    resolution_ms: u64,
    rule: VolatilityRule,
) -> MMResult<Decimal> {
    let steps_per_year = Decimal::from(MS_PER_YEAR)
        .checked_div(Decimal::from(resolution_ms))
        .or_overflow("grid steps per year")?;
    let estimator = VolatilityEstimator::with_annualization_factor(decimal_sqrt(steps_per_year)?);
    match rule {
        VolatilityRule::Simple => estimator.simple_of_returns(returns),
        VolatilityRule::Ewma { lambda } => estimator.ewma_of_returns(returns, lambda),
    }
}

/// Absolute volatility realized on a test window, by the simple estimator
/// whatever the training rule. `None` when the window's prices would not pass
/// the training minimum or gap limit, or give too few returns.
fn realized_volatility(
    prices: &[PriceSample],
    parts: &[TimeWindow],
    config: &WalkForwardConfig,
) -> MMResult<Option<Decimal>> {
    let grid = grid_prices(prices, parts, config.resolution_ms)?;
    if grid.observed < config.min_grid_prices
        || longest_gap_ms(prices, parts)? > gap_limit_ms(config)?
    {
        return Ok(None);
    }
    let returns = grid.log_returns()?;
    let Some(last) = grid.last_price().filter(|_| returns.len() >= MIN_RETURNS) else {
        return Ok(None);
    };
    relative_volatility(&returns, config.resolution_ms, VolatilityRule::Simple)?
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
    /// Nearest distance of the bucket; `levels` are in ascending distance.
    fn nearest(&self) -> f64 {
        self.levels.first().map_or(0.0, |(d, _)| *d)
    }

    /// `Σ T · exp(-k · (d - nearest))`, which keeps the nearest level's weight
    /// from underflowing whatever `k` is.
    fn relative_mass(&self, k: f64) -> f64 {
        let nearest = self.nearest();
        self.levels
            .iter()
            .map(|(d, t)| t * (-k * (d - nearest)).exp())
            .sum()
    }

    /// `Σ T · exp(-k · d)`, which may underflow to zero.
    fn mass(&self, k: f64) -> f64 {
        self.relative_mass(k) * (-k * self.nearest()).exp()
    }

    /// The bucket's expected fills times its mean distance under `k`, once
    /// its own multiplier is profiled out.
    fn fill_moment(&self, k: f64) -> f64 {
        let mass = self.relative_mass(k);
        if self.fills == 0.0 || mass <= 0.0 {
            return 0.0;
        }
        let nearest = self.nearest();
        let moment: f64 = self
            .levels
            .iter()
            .map(|(d, t)| t * (-k * (d - nearest)).exp() * d)
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
    sessions: &[TimeWindow],
    closed: &[bool],
    config: &WalkForwardConfig,
) -> MMResult<Result<IntensityFit, CalibrationFailure>> {
    let mut cells: BTreeMap<(usize, Decimal), (u64, u64)> = BTreeMap::new();
    for e in exposures {
        let bucket = bucket_of(
            intraday_time(sessions, e.timestamp_ms),
            config.seasonality_buckets,
        )?;
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
            if closed.get(bucket).copied().unwrap_or(false) {
                continue;
            }
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
    if predicted(0.0) <= observed * (1.0 + FIT_TOLERANCE) {
        return Ok(Err(CalibrationFailure::NonDecayingIntensity));
    }
    // As `k` grows, each bucket's mean distance falls to its nearest level. At
    // that limit rounding would stop the bisection at an arbitrary finite `k`.
    let limit: f64 = buckets.iter().map(|b| b.fills * b.nearest()).sum();
    if observed <= limit * (1.0 + FIT_TOLERANCE) {
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
    let Some(fit_k) = Decimal::from_f64(k) else {
        return Ok(Err(CalibrationFailure::UnrepresentableIntensity));
    };
    if fit_k <= Decimal::ZERO {
        return Ok(Err(CalibrationFailure::NonDecayingIntensity));
    }
    let Some(fit_a) = positive_decimal(a) else {
        return Ok(Err(CalibrationFailure::UnrepresentableIntensity));
    };
    let mut seasonality = Vec::with_capacity(buckets.len());
    for bucket in &buckets {
        let multiplier = if bucket.fills == 0.0 {
            Some(Decimal::ZERO)
        } else {
            positive_decimal(bucket.fills / bucket.mass(k) / a)
        };
        let Some(multiplier) = multiplier else {
            return Ok(Err(CalibrationFailure::UnrepresentableIntensity));
        };
        seasonality.push(multiplier);
    }
    Ok(Ok(IntensityFit {
        a: fit_a,
        k: fit_k,
        seasonality,
    }))
}

/// Scores a volatility and a rate function on a test window.
fn score(
    volatility: Decimal,
    realized: Option<Decimal>,
    exposures: &[QuoteExposure],
    rate: impl Fn(Decimal, u64) -> MMResult<f64>,
) -> MMResult<OutOfSampleError> {
    let volatility = realized.map(|r| abs_diff(volatility, r)).transpose()?;

    let mut likelihood = 0.0f64;
    let mut total_seconds = 0.0f64;
    let mut highest: Option<(f64, &QuoteExposure)> = None;
    let mut in_range = true;
    for e in exposures {
        let seconds = to_f64(Decimal::from(e.exposure_ms))? / 1_000.0;
        let predicted = rate(e.distance, e.timestamp_ms)?;
        if highest.is_none_or(|(rate, _)| predicted > rate) {
            highest = Some((predicted, e));
        }
        let Some(forecast) = representable_rate(predicted) else {
            in_range = false;
            continue;
        };
        let per_second = to_f64(forecast)?.max(SCORE_RATE_FLOOR);
        let expected = per_second * seconds;
        likelihood += expected - to_f64(Decimal::from(e.fills))? * expected.ln();
        total_seconds += seconds;
    }
    let Some((_, highest)) = highest else {
        return Ok(OutOfSampleError {
            volatility,
            intensity: None,
            intensity_out_of_support: None,
        });
    };
    // The score is a mean of terms that are in range, but rounding in `f64`
    // can take it just above the range, so it is checked as well.
    let score = in_range
        .then_some(likelihood / total_seconds)
        .and_then(Decimal::from_f64);
    Ok(OutOfSampleError {
        volatility,
        intensity: score,
        intensity_out_of_support: score.is_none().then_some(OutOfSupportForecast {
            timestamp_ms: highest.timestamp_ms,
            distance: highest.distance,
        }),
    })
}

/// `A · exp(-k · δ)` in `f64`. A huge `A` and a tiny `exp` must meet before
/// any rounding to `Decimal`, whose resolution would lose the `exp`.
fn intensity_f64(a: Decimal, k: Decimal, distance: Decimal) -> MMResult<f64> {
    Ok(to_f64(a)? * (-to_f64(k)? * to_f64(distance)?).exp())
}

fn to_f64(value: Decimal) -> MMResult<f64> {
    value
        .to_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MMError::NumericalError(format!("{value} is not a finite f64")))
}

/// A rate as a `Decimal`. A result below the resolution of `Decimal` is zero,
/// and `None` means it is above the range.
fn representable_rate(rate: f64) -> Option<Decimal> {
    if rate < RATE_ZERO_BELOW {
        return Some(Decimal::ZERO);
    }
    Decimal::from_f64(rate)
}

/// A finite, strictly positive `f64` as a `Decimal`, if it stays positive.
fn positive_decimal(value: f64) -> Option<Decimal> {
    Some(value)
        .filter(|v| v.is_finite())
        .and_then(Decimal::from_f64)
        .filter(|v| *v > Decimal::ZERO)
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

/// Time the market is open in the parts of a window.
fn open_time_ms(parts: &[TimeWindow]) -> MMResult<u64> {
    parts.iter().try_fold(0u64, |total, part| {
        part.end_ms
            .checked_sub(part.start_ms)
            .and_then(|span| total.checked_add(span))
            .or_overflow("open time")
    })
}

/// The time that places an exposure in an intraday bucket. An exposure stamped
/// exactly at a session's end was at risk until then, so it counts in that
/// session's last millisecond, unless the next session starts at that instant.
fn intraday_time(sessions: &[TimeWindow], timestamp_ms: u64) -> u64 {
    let ending = sessions.partition_point(|s| s.end_ms < timestamp_ms);
    let closes_here = sessions
        .get(ending)
        .is_some_and(|s| s.end_ms == timestamp_ms);
    let next_opens_here = ending
        .checked_add(1)
        .and_then(|next| sessions.get(next))
        .is_some_and(|s| s.start_ms == timestamp_ms);
    if closes_here && !next_opens_here {
        timestamp_ms.saturating_sub(1)
    } else {
        timestamp_ms
    }
}

/// Which intraday buckets the window reaches only while closed. A bucket the
/// window never reaches is unknown, not closed, and without sessions the one
/// part is the window, so none is closed.
fn closed_buckets(window: &TimeWindow, parts: &[TimeWindow], buckets: u32) -> MMResult<Vec<bool>> {
    let count = usize::try_from(buckets)
        .map_err(|_| MMError::InvalidConfiguration("too many buckets".to_string()))?;
    let mut reached = vec![false; count];
    mark_buckets(window, buckets, &mut reached)?;
    let mut open = vec![false; count];
    for part in parts {
        mark_buckets(part, buckets, &mut open)?;
    }
    Ok(reached
        .iter()
        .zip(&open)
        .map(|(reached, open)| *reached && !*open)
        .collect())
}

/// Marks the intraday buckets that `interval` overlaps. `flags` has `buckets`
/// positive entries.
fn mark_buckets(interval: &TimeWindow, buckets: u32, flags: &mut [bool]) -> MMResult<()> {
    // One day reaches every bucket.
    let end_ms = interval
        .end_ms
        .min(interval.start_ms.saturating_add(MS_PER_DAY));
    let mut time = interval.start_ms;
    while time < end_ms {
        let index = bucket_of(time, buckets)?;
        *flags.get_mut(index).or_overflow("seasonality bucket")? = true;
        // The first millisecond of the next bucket, or of the next day.
        let next_bucket = u64::try_from(index)
            .ok()
            .and_then(|i| i.checked_add(1))
            .and_then(|i| i.checked_mul(MS_PER_DAY))
            .map(|ms| ms.div_ceil(u64::from(buckets)))
            .or_overflow("next intraday bucket")?;
        let day_start = time
            .checked_div(MS_PER_DAY)
            .and_then(|day| day.checked_mul(MS_PER_DAY))
            .or_overflow("day start")?;
        // The clock may end before the next bucket does.
        let Some(next) = day_start.checked_add(next_bucket) else {
            break;
        };
        time = next;
    }
    Ok(())
}

/// `|a - b|`.
fn abs_diff(a: Decimal, b: Decimal) -> MMResult<Decimal> {
    a.checked_sub(b).map(|d| d.abs()).or_overflow("difference")
}

/// FNV-1a 128 of the samples, independent of the scale of their decimals. It
/// is the digest of the `kernel` feature, which this module does not require.
/// With sessions it covers what they decide inside the window: its open parts,
/// and the intraday time of each exposure, which a session ending at the
/// window's start can move.
fn window_hash(
    prices: &[PriceSample],
    exposures: &[QuoteExposure],
    parts: &[TimeWindow],
    sessions: &[TimeWindow],
) -> u128 {
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
    if sessions.is_empty() {
        return hash;
    }
    for part in parts {
        hash = fnv(hash, &part.start_ms.to_le_bytes());
        hash = fnv(hash, &part.end_ms.to_le_bytes());
    }
    for e in exposures {
        hash = fnv(hash, &intraday_time(sessions, e.timestamp_ms).to_le_bytes());
    }
    hash
}

pub(crate) const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

pub(crate) fn fnv(hash: u128, bytes: &[u8]) -> u128 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u128::from(*b)).wrapping_mul(FNV_PRIME))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    #![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

    use super::*;

    /// Buckets holding some millisecond of the interval, by `bucket_of` itself.
    fn scanned(interval: &TimeWindow, buckets: u32) -> Vec<bool> {
        let mut flags = vec![false; buckets as usize];
        for time in interval.start_ms..interval.end_ms {
            flags[bucket_of(time, buckets).unwrap()] = true;
        }
        flags
    }

    fn marked(interval: &TimeWindow, buckets: u32) -> Vec<bool> {
        let mut flags = vec![false; buckets as usize];
        mark_buckets(interval, buckets, &mut flags).unwrap();
        flags
    }

    #[test]
    fn test_mark_buckets_inexact_width_matches_bucket_of_at_each_edge() {
        // A seventh of a day is not a whole number of milliseconds.
        let edge = MS_PER_DAY.div_ceil(7);
        for (start_ms, end_ms) in [
            (0, 1),
            (0, edge),
            (0, edge + 1),
            (edge - 1, edge),
            (edge - 1, edge + 1),
            (edge, edge + 1),
            (2 * edge - 2, 2 * edge + 2),
        ] {
            let interval = TimeWindow { start_ms, end_ms };
            assert_eq!(marked(&interval, 7), scanned(&interval, 7), "{interval:?}");
        }
    }

    #[test]
    fn test_mark_buckets_crossing_midnight_wraps_to_the_first_buckets() {
        let interval = TimeWindow {
            start_ms: MS_PER_DAY - 1,
            end_ms: MS_PER_DAY + 1,
        };
        let flags = marked(&interval, 24);
        assert!(flags[23] && flags[0]);
        assert_eq!(flags.iter().filter(|f| **f).count(), 2);
    }

    #[test]
    fn test_mark_buckets_a_day_or_more_reaches_every_bucket() {
        for span in [MS_PER_DAY, 400 * MS_PER_DAY] {
            let interval = TimeWindow {
                start_ms: 12_345,
                end_ms: 12_345 + span,
            };
            assert!(marked(&interval, 1_440).iter().all(|f| *f));
        }
        // Just under a day from inside a bucket misses nothing either: the
        // start bucket is reached again at its other edge.
        let interval = TimeWindow {
            start_ms: 3_600_000 + 1,
            end_ms: 3_600_000 + MS_PER_DAY,
        };
        assert!(marked(&interval, 24).iter().all(|f| *f));
    }
}
