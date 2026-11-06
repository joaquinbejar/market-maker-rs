//! OptionStratLib's calibrated surface behind [`VolSurface`], and what is known when.
//!
//! [`UpstreamSurface`] reads an [`ImpliedVolatilitySurface`] and its derivatives, so
//! [`SurfacePricer`](crate::options::pricing::SurfacePricer) and the chain take it unchanged. The
//! interpolation, the wings and the short and long expiries are all upstream's. [`UpstreamStore`]
//! holds calibrations with the time each became available, and [`SurfaceQuality`] reports how a
//! surface sits on its quotes.
//!
//! # What a read refuses
//!
//! Upstream reports these and does not refuse them, so the [`UpstreamPolicy`] decides, and by
//! default it refuses:
//!
//! - **Static arbitrage.** A calendar violation gives a negative `dw/dT` with no error. A surface
//!   whose report did not pass is refused when it is built. If it is allowed, the negative slope
//!   reaches the pricer, whose theta stays well defined.
//! - **An expiry with no slice.** A read between two expiries with an unfitted one inside them says
//!   so, but at a fitted expiry the flag stays false even when its slope spans the gap, so that is
//!   looked up here. Either read is refused.
//! - **Expiries too close to measure.** A gap below the resolution of `T` gives a huge `dw/dT` with
//!   no error. A surface with a gap under the policy's is refused when it is built.
//!
//! A read beyond the [`ExtrapolationPolicy`](optionstratlib::volatility::ExtrapolationPolicy) the
//! surface was calibrated with is an error, and one within it is served: declaring that policy is
//! what allows it.
//!
//! # `dw/dT`
//!
//! As [`VolSurface`] asks, it is at fixed log moneyness, and on a fitted expiry it is the slope of
//! the segment towards shorter expiries. The pricer converts it to a fixed strike. At the first
//! expiry, or with only one, there is no such segment, and upstream gives `w / T`, the slope at
//! constant implied volatility. So this differs from [`VolSurface`]'s convention for slices at
//! the first node, which is the slope of the segment towards longer expiries.
//!
//! # Ageing
//!
//! A surface is calibrated as of one instant, and its expiries are absolute. Served later, a read
//! at `tau` is of the expiry `T = tau + age` out from that instant, rolled down at constant implied
//! volatility per `k`, as [`SurfaceStore`](super::store::SurfaceStore) does:
//! `w'(k, tau) = w(k, T) tau / T`. An old expiry is never read as a new one. This is an explicit
//! policy, not a forecast, and upstream has no read for it, so the chain rule is here.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use optionstratlib::prelude::Positive;
use optionstratlib::volatility::{
    ImpliedVolatilitySurface, SurfaceExpiry, SurfacePointDerivatives, SurfaceResidual,
};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

use super::store::YEAR_MS;
use super::{TotalVariance, VolSurface};
use crate::Decimal;
use crate::types::error::{MMError, MMResult};

/// A second, in years, the default smallest gap between two fitted expiries.
const SECOND_IN_YEARS: f64 = 1_000.0 / YEAR_MS;

/// How close to a fitted expiry a read counts as at it, relative to its `T`: wide enough for what
/// `f64` and a sum with the age leave of it, and well under any gap the policy accepts.
const AT_EXPIRY: f64 = 1e-11;

fn invalid(what: String) -> MMError {
    MMError::InvalidMarketState(what)
}

fn number(value: Decimal, name: &str) -> MMResult<f64> {
    value
        .to_f64()
        .filter(|read| read.is_finite())
        .ok_or_else(|| MMError::NumericalError(format!("{name} {value} has no finite value")))
}

/// `at` in ms, which it must be exactly: truncating a finer time would let data be available in
/// the millisecond before it existed.
fn millis(at: DateTime<Utc>) -> MMResult<u64> {
    if !at.timestamp_subsec_nanos().is_multiple_of(1_000_000) {
        return Err(invalid(format!("{at} is finer than a millisecond")));
    }
    u64::try_from(at.timestamp_millis())
        .map_err(|_| invalid(format!("{at} is before the Unix epoch")))
}

/// What a surface may do and still be read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpstreamPolicy {
    /// Accept a surface whose static arbitrage report did not pass.
    pub allow_static_arbitrage: bool,
    /// Serve a value or a slope that spans an expiry with no slice.
    pub allow_unfitted_spans: bool,
    /// The smallest gap between two fitted expiries, in years.
    pub min_expiry_gap_years: f64,
}

impl Default for UpstreamPolicy {
    fn default() -> Self {
        Self {
            allow_static_arbitrage: false,
            allow_unfitted_spans: false,
            min_expiry_gap_years: SECOND_IN_YEARS,
        }
    }
}

/// A fitted expiry, as a read needs it.
#[derive(Debug, Clone, Copy)]
struct Node {
    years: f64,
    time_to_expiry: Positive,
    /// An expiry with no slice lies between this one and the one before.
    slope_spans_unfitted: bool,
}

/// A checked [`ImpliedVolatilitySurface`], read as a [`VolSurface`]. See the
/// [module documentation](self).
#[derive(Debug, Clone)]
pub struct UpstreamSurface {
    surface: Arc<ImpliedVolatilitySurface>,
    nodes: Arc<[Node]>,
    policy: UpstreamPolicy,
    age_years: f64,
}

impl UpstreamSurface {
    /// Checks `surface` against `policy`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the minimum gap is not finite and non negative,
    /// and [`MMError::InvalidMarketState`] for a static arbitrage report that did not pass, or two
    /// fitted expiries closer than the minimum gap, unless the policy allows it.
    pub fn new(surface: ImpliedVolatilitySurface, policy: UpstreamPolicy) -> MMResult<Self> {
        if !policy.min_expiry_gap_years.is_finite() || policy.min_expiry_gap_years < 0.0 {
            return Err(MMError::InvalidConfiguration(format!(
                "minimum expiry gap must be finite and not negative, got {}",
                policy.min_expiry_gap_years
            )));
        }
        if !policy.allow_static_arbitrage && !surface.arbitrage_report().passed() {
            return Err(invalid(format!(
                "the surface has static arbitrage: {} violations",
                surface.arbitrage_report().violations().len()
            )));
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(surface.expiries().len());
        for expiry in surface.expiries() {
            let time_to_expiry = expiry.time_to_expiry();
            let years = number(time_to_expiry.to_dec(), "time to expiry")?;
            let earlier = nodes.last().map(|node| node.years);
            if let Some(earlier) = earlier
                && years - earlier < policy.min_expiry_gap_years
            {
                return Err(invalid(format!(
                    "fitted expiries {earlier} and {years} years out are closer than {}",
                    policy.min_expiry_gap_years
                )));
            }
            let slope_spans_unfitted = match earlier {
                Some(earlier) => surface.unfitted().iter().any(|unfitted| {
                    unfitted
                        .time_to_expiry
                        .and_then(|years| years.to_dec().to_f64())
                        .is_some_and(|gap| earlier < gap && gap < years)
                }),
                None => false,
            };
            nodes.push(Node {
                years,
                time_to_expiry,
                slope_spans_unfitted,
            });
        }
        Ok(Self {
            surface: Arc::new(surface),
            nodes: nodes.into(),
            policy,
            age_years: 0.0,
        })
    }

    /// The calibration behind it.
    #[must_use]
    pub fn surface(&self) -> &ImpliedVolatilitySurface {
        &self.surface
    }

    /// How old this read of it is, in years.
    #[must_use]
    pub fn age_years(&self) -> f64 {
        self.age_years
    }

    /// The same calibration, read `age_years` after its as-of time.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for an age that is negative or not finite.
    pub fn aged(&self, age_years: f64) -> MMResult<Self> {
        if !age_years.is_finite() || age_years < 0.0 {
            return Err(invalid(format!("age {age_years} is not usable")));
        }
        Ok(Self {
            age_years,
            ..self.clone()
        })
    }

    fn read(&self, k: f64, years: f64) -> MMResult<(SurfacePointDerivatives, f64)> {
        let refused = |error: &dyn std::fmt::Display| {
            invalid(format!("surface read at k {k}, T {years}: {error}"))
        };
        let log_moneyness = Decimal::from_f64(k)
            .ok_or_else(|| MMError::NumericalError(format!("k {k} is not a decimal")))?;
        // Within what the arithmetic leaves of it, this is the expiry itself, whose slope is the
        // one towards shorter expiries. Of two within that, the nearer is the one meant.
        let node = self
            .nodes
            .iter()
            .filter(|node| (node.years - years).abs() <= AT_EXPIRY * node.years)
            .min_by(|a, b| (a.years - years).abs().total_cmp(&(b.years - years).abs()));
        let time_to_expiry = match node {
            Some(node) => node.time_to_expiry,
            None => Decimal::from_f64(years)
                .and_then(|years| Positive::new_decimal(years).ok())
                .ok_or_else(|| {
                    MMError::NumericalError(format!("T {years} is not a positive decimal"))
                })?,
        };
        let read = self
            .surface
            .point_with_derivatives(log_moneyness, time_to_expiry)
            .map_err(|error| refused(&error))?;
        if !self.policy.allow_unfitted_spans {
            if read.point.bridges_unfitted_expiry {
                return Err(refused(&"the value spans an expiry with no slice"));
            }
            if node.is_some_and(|node| node.slope_spans_unfitted) {
                return Err(refused(&"the slope spans an expiry with no slice"));
            }
        }
        let read_years = number(read.point.time_to_expiry.to_dec(), "time to expiry")?;
        Ok((read, read_years))
    }
}

impl VolSurface for UpstreamSurface {
    fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        let years = tau + self.age_years;
        let (read, read_years) = self.read(k, years)?;
        let w = number(read.point.total_variance.to_dec(), "total variance")?;
        let w_k = number(read.total_variance_slope, "dw/dk")?;
        let w_kk = number(read.total_variance_curvature, "d2w/dk2")?;
        let w_t = number(read.total_variance_time_slope, "dw/dT")?;
        if self.age_years == 0.0 {
            return TotalVariance::new(w, w_k, w_kk, w_t);
        }
        // w'(k, tau) = w(k, tau + age) tau / (tau + age).
        let scale = tau / read_years;
        let age_term = self.age_years / (read_years * read_years);
        TotalVariance::new(
            w * scale,
            w_k * scale,
            w_kk * scale,
            w_t * scale + w * age_term,
        )
    }
}

/// One calibration attempt.
#[derive(Debug, Clone)]
pub struct UpstreamAttempt {
    as_of_ms: u64,
    available_at_ms: u64,
    outcome: Result<UpstreamSurface, String>,
}

impl UpstreamAttempt {
    /// A calibration that gave `surface`, available from `available_at_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if it is available before its as-of time, or that
    /// time is before the Unix epoch or finer than a millisecond.
    pub fn calibrated(surface: UpstreamSurface, available_at_ms: u64) -> MMResult<Self> {
        let as_of_ms = millis(surface.surface().as_of())?;
        Self::checked(as_of_ms, available_at_ms, Ok(surface))
    }

    /// A calibration that failed, so that an audit can see it. It is never served.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if it is available before its as-of time.
    pub fn failed(as_of_ms: u64, available_at_ms: u64, reason: String) -> MMResult<Self> {
        Self::checked(as_of_ms, available_at_ms, Err(reason))
    }

    fn checked(
        as_of_ms: u64,
        available_at_ms: u64,
        outcome: Result<UpstreamSurface, String>,
    ) -> MMResult<Self> {
        if available_at_ms < as_of_ms {
            return Err(invalid(format!(
                "a calibration as of {as_of_ms} cannot be available at {available_at_ms}"
            )));
        }
        Ok(Self {
            as_of_ms,
            available_at_ms,
            outcome,
        })
    }

    /// The latest data it read, in ms.
    #[must_use]
    pub fn as_of_ms(&self) -> u64 {
        self.as_of_ms
    }

    /// When it could first be used, in ms.
    #[must_use]
    pub fn available_at_ms(&self) -> u64 {
        self.available_at_ms
    }

    /// Why it failed, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.outcome.as_ref().err().map(String::as_str)
    }
}

/// A surface served at one instant.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ServedUpstream {
    /// The surface, rolled down to the instant. Hand it to a pricer.
    pub surface: UpstreamSurface,
    /// The instant it was asked for, in ms.
    pub valuation_at_ms: u64,
    /// The latest data it read, in ms.
    pub as_of_ms: u64,
    /// When it became available, in ms.
    pub available_at_ms: u64,
    /// `valuation_at_ms - as_of_ms`.
    pub age_ms: u64,
    /// An attempt the rule would have preferred, available at the instant, was passed over.
    pub fallback: bool,
}

/// Calibrations with the time each became available, append only, so a past answer never changes.
///
/// [`UpstreamStore::surface_at`] sees only what was available at the instant, and its age runs from
/// the data and not from when it became available, so a late calculation does not make old data
/// look fresh.
#[derive(Debug, Clone, Default)]
pub struct UpstreamStore {
    attempts: Vec<UpstreamAttempt>,
}

impl UpstreamStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an attempt.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] unless it became available after the last one.
    pub fn insert(&mut self, attempt: UpstreamAttempt) -> MMResult<()> {
        if let Some(last) = self.attempts.last()
            && attempt.available_at_ms <= last.available_at_ms
        {
            return Err(invalid(format!(
                "an attempt available at {} follows one available at {}",
                attempt.available_at_ms, last.available_at_ms
            )));
        }
        self.attempts.push(attempt);
        Ok(())
    }

    /// Every attempt, in the order they became available.
    #[must_use]
    pub fn attempts(&self) -> &[UpstreamAttempt] {
        &self.attempts
    }

    /// The attempt with the newest data available at `at_ms`, no older than `max_age_ms`; between
    /// two on the same data, the later available. A failed one is passed over.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] when nothing is usable at `at_ms`, saying why.
    pub fn surface_at(&self, at_ms: u64, max_age_ms: u64) -> MMResult<ServedUpstream> {
        let mut candidates: Vec<&UpstreamAttempt> = self
            .attempts
            .iter()
            .filter(|attempt| attempt.available_at_ms <= at_ms)
            .collect();
        candidates
            .sort_by_key(|attempt| std::cmp::Reverse((attempt.as_of_ms, attempt.available_at_ms)));
        let mut reasons = Vec::new();
        for (rank, attempt) in candidates.into_iter().enumerate() {
            let age_ms = at_ms.saturating_sub(attempt.as_of_ms);
            if age_ms > max_age_ms {
                // In decreasing order of data: every one left is at least as old.
                reasons.push(format!("as of {}: {age_ms} ms old", attempt.as_of_ms));
                break;
            }
            match &attempt.outcome {
                Ok(surface) => {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "an age in ms is far under 2^53"
                    )]
                    let age_years = age_ms as f64 / YEAR_MS;
                    return Ok(ServedUpstream {
                        surface: surface.aged(age_years)?,
                        valuation_at_ms: at_ms,
                        as_of_ms: attempt.as_of_ms,
                        available_at_ms: attempt.available_at_ms,
                        age_ms,
                        fallback: rank > 0,
                    });
                }
                Err(reason) => reasons.push(format!("as of {}: {reason}", attempt.as_of_ms)),
            }
        }
        Err(invalid(format!(
            "no surface is usable at {at_ms}: {}",
            if reasons.is_empty() {
                "none is available yet".to_string()
            } else {
                reasons.join("; ")
            }
        )))
    }
}

/// How one expiry sits on its quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExpiryQuality {
    /// The expiry.
    pub expiry: DateTime<Utc>,
    /// Its time to expiry, in years.
    pub time_to_expiry: Decimal,
    /// The quotes in its fit.
    pub quotes: usize,
    /// Weighted RMSE of the fit, in volatility.
    pub rmse: Decimal,
    /// The same with every quote at one weight, which is what shows a fit that weights hide.
    pub unweighted_rmse: Decimal,
    /// Quotes whose fitted price falls outside their bid and ask.
    pub quotes_outside_bid_ask: usize,
    /// The RMSE of a flat volatility on the same quotes, less this fit's, in volatility: positive
    /// when the fit is the closer. In sample, as both are.
    pub baseline_improvement: Decimal,
}

/// How a surface sits on the quotes it was built from: residuals by strike and expiry, and what
/// was left out, as counts. Everything comes from upstream's own record of the calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SurfaceQuality {
    /// The calibration's as-of time.
    pub as_of: DateTime<Utc>,
    /// Each expiry with a slice.
    pub expiries: Vec<ExpiryQuality>,
    /// The residual of each quote in a fit, by expiry and strike.
    pub residuals: Vec<SurfaceResidual>,
    /// Quotes rejected before any fit, by reason.
    pub rejected: BTreeMap<String, usize>,
    /// Expiries that got no slice, by reason.
    pub unfitted: BTreeMap<String, usize>,
    /// Static arbitrage violations on the report's grid.
    pub arbitrage_violations: usize,
    /// Points of that grid that could not be evaluated.
    pub arbitrage_unevaluated: usize,
    /// Whether the calendar condition could be checked: it needs two expiries.
    pub calendar_checked: bool,
}

/// The name of a reason, without what it carries.
fn reason_name(reason: &dyn std::fmt::Debug) -> String {
    let text = format!("{reason:?}");
    text.split([' ', '{', '(']).next().unwrap_or("").to_string()
}

fn expiry_quality(expiry: &SurfaceExpiry) -> ExpiryQuality {
    ExpiryQuality {
        expiry: expiry.expiry(),
        time_to_expiry: expiry.time_to_expiry().to_dec(),
        quotes: expiry.residuals().len(),
        rmse: expiry.rmse().to_dec(),
        unweighted_rmse: expiry.unweighted_rmse().to_dec(),
        quotes_outside_bid_ask: expiry.quotes_outside_bid_ask(),
        baseline_improvement: expiry.baseline().improvement(),
    }
}

impl SurfaceQuality {
    /// Reads the report of `surface`.
    #[must_use]
    pub fn of(surface: &ImpliedVolatilitySurface) -> Self {
        let mut rejected = BTreeMap::new();
        for rejection in surface.rejections() {
            let count = rejected
                .entry(reason_name(&rejection.reason))
                .or_insert(0_usize);
            *count = count.saturating_add(1);
        }
        let mut unfitted = BTreeMap::new();
        for expiry in surface.unfitted() {
            let count = unfitted
                .entry(reason_name(&expiry.reason))
                .or_insert(0_usize);
            *count = count.saturating_add(1);
        }
        Self {
            as_of: surface.as_of(),
            expiries: surface.expiries().iter().map(expiry_quality).collect(),
            residuals: surface.residuals().copied().collect(),
            rejected,
            unfitted,
            arbitrage_violations: surface.arbitrage_report().violations().len(),
            arbitrage_unevaluated: surface.arbitrage_report().unevaluated().len(),
            calendar_checked: surface.calendar_checked(),
        }
    }
}
