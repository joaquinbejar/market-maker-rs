//! Out of sample validation of the SVI fit against a simple baseline.
//!
//! The split is fixed before any fit, by a declared rule: per expiry, the
//! out of the money quotes are sorted by `k` and every other one (odd
//! positions) is held out, but only for an expiry with at least
//! [`HoldoutRule::min_quotes`] quotes, so that at least half are left to
//! train on; any other expiry is reported as not applicable with its size,
//! and all its quotes train. SVI and the baseline train on exactly the same
//! quotes, and nothing (grid, weights, tolerances) is chosen with the
//! held out ones.
//!
//! The baseline ([`BaselineSurface`]) is linear in `w` between the quoted
//! `k` of each expiry, flat beyond them, and linear in `w` across expiries.
//!
//! Residuals are observed minus model implied volatility. They are reported
//! by bucket, each with its size and the points the model could not
//! evaluate (never dropped to improve a metric):
//!
//! - **wings**: `|k| / sqrt(w_atm) > z_wing`, the preregistered threshold,
//!   with `w_atm` the trained SVI slice's `w(0)` (never the held out data);
//! - **auxiliary wings**: `|k| > abs_k`, a separate fixed cut;
//! - **shortest expiry**;
//! - **all**.

use std::collections::BTreeMap;

use super::calibrate::{FitConfig, SliceObservation, fit_slice};
use super::iv::IvObservation;
use super::svi::{ShortExpiryPolicy, SviSurface};
use super::{TotalVariance, VolSurface};
use crate::types::error::{MMError, MMResult};

/// One expiry of the baseline: `(k, w)` points by increasing `k`.
#[derive(Debug, Clone, PartialEq)]
struct BaselineSlice {
    tau: f64,
    points: Vec<(f64, f64)>,
}

impl BaselineSlice {
    fn eval(&self, k: f64) -> MMResult<(f64, f64)> {
        let (Some(&(k0, w0)), Some(&(kn, wn))) = (self.points.first(), self.points.last()) else {
            return Err(MMError::InvalidConfiguration(
                "empty baseline slice".to_string(),
            ));
        };
        if k <= k0 {
            return Ok((w0, 0.0));
        }
        if k >= kn {
            return Ok((wn, 0.0));
        }
        for pair in self.points.windows(2) {
            if let [(ka, wa), (kb, wb)] = pair
                && k <= *kb
            {
                let slope = (wb - wa) / (kb - ka);
                return Ok((wa + slope * (k - ka), slope));
            }
        }
        Err(MMError::InvalidConfiguration(
            "baseline bracket not found".to_string(),
        ))
    }
}

/// Linear in `w` between quoted `k`, flat beyond, linear across expiries.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineSurface {
    slices: Vec<BaselineSlice>,
}

impl BaselineSurface {
    /// From `(tau, [(k, w)])` per expiry.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for fewer than two expiries, an
    /// expiry without points, non-increasing `tau` or `k`, or a value that
    /// is not finite.
    pub fn new(mut slices: Vec<(f64, Vec<(f64, f64)>)>) -> MMResult<Self> {
        slices.sort_by(|a, b| a.0.total_cmp(&b.0));
        if slices.len() < 2 || slices.windows(2).any(|p| matches!(p, [a, b] if b.0 <= a.0)) {
            return Err(MMError::InvalidConfiguration(
                "a baseline needs at least two increasing expiries".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(slices.len());
        for (tau, mut points) in slices {
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            let ok = tau.is_finite()
                && tau > 0.0
                && !points.is_empty()
                && points
                    .iter()
                    .all(|(k, w)| k.is_finite() && w.is_finite() && *w > 0.0)
                && points.windows(2).all(|p| matches!(p, [a, b] if b.0 > a.0));
            if !ok {
                return Err(MMError::InvalidConfiguration(format!(
                    "baseline expiry {tau}: points must be finite, positive and strictly increasing in k"
                )));
            }
            out.push(BaselineSlice { tau, points });
        }
        Ok(Self { slices: out })
    }
}

impl VolSurface for BaselineSurface {
    fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        let (Some(first), Some(last)) = (self.slices.first(), self.slices.last()) else {
            return Err(MMError::InvalidConfiguration("empty baseline".to_string()));
        };
        if tau < first.tau || tau > last.tau {
            return Err(MMError::InvalidMarketState(format!(
                "tau {tau} outside the baseline [{}, {}]",
                first.tau, last.tau
            )));
        }
        let j = self
            .slices
            .iter()
            .position(|s| s.tau >= tau)
            .unwrap_or(self.slices.len().saturating_sub(1))
            .max(1);
        let (Some(lo), Some(hi)) = (self.slices.get(j - 1), self.slices.get(j)) else {
            return Err(MMError::InvalidConfiguration(
                "baseline bracket".to_string(),
            ));
        };
        let x = (tau - lo.tau) / (hi.tau - lo.tau);
        let (wl, kl) = lo.eval(k)?;
        let (wh, kh) = hi.eval(k)?;
        TotalVariance::new(
            (1.0 - x) * wl + x * wh,
            (1.0 - x) * kl + x * kh,
            0.0,
            (wh - wl) / (hi.tau - lo.tau),
        )
    }
}

/// The preregistered split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldoutRule {
    /// Fewest out of the money quotes an expiry needs for a holdout.
    pub min_quotes: usize,
}

impl Default for HoldoutRule {
    fn default() -> Self {
        Self { min_quotes: 10 }
    }
}

/// The preregistered buckets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WingRule {
    /// Wing when `|k| / sqrt(w_atm) > z_wing`.
    pub z_wing: f64,
    /// Auxiliary wing when `|k| > abs_k`.
    pub abs_k: f64,
}

impl Default for WingRule {
    fn default() -> Self {
        Self {
            z_wing: 2.0,
            abs_k: 0.2,
        }
    }
}

/// Residual statistics of one model on one bucket.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BucketStats {
    /// Held out points in the bucket.
    pub points: usize,
    /// Points the model could not evaluate.
    pub missing: usize,
    /// Root mean square volatility residual over the evaluated points.
    pub rmse_vol: f64,
    /// Largest absolute volatility residual.
    pub max_abs_vol: f64,
}

/// SVI and baseline on one bucket.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BucketComparison {
    /// The SVI fit.
    pub svi: BucketStats,
    /// The baseline.
    pub baseline: BucketStats,
}

/// The holdout validation.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutReport {
    /// Expiries (ms) without a holdout, with their size.
    pub not_applicable: Vec<(u64, usize)>,
    /// Expiries whose SVI slice failed to fit, with the reason.
    pub fit_failures: Vec<(u64, String)>,
    /// Every held out point.
    pub all: BucketComparison,
    /// `|k| / sqrt(w_atm) > z_wing`.
    pub wings: BucketComparison,
    /// `|k| > abs_k`.
    pub wings_abs: BucketComparison,
    /// The shortest expiry.
    pub shortest: BucketComparison,
}

#[derive(Default)]
struct Acc {
    n: usize,
    missing: usize,
    sum_sq: f64,
    max_abs: f64,
}

impl Acc {
    fn add(&mut self, residual: Option<f64>) {
        self.n = self.n.saturating_add(1);
        match residual {
            Some(r) => {
                self.sum_sq += r * r;
                self.max_abs = self.max_abs.max(r.abs());
            }
            None => self.missing = self.missing.saturating_add(1),
        }
    }

    fn stats(&self) -> BucketStats {
        let evaluated = self.n.saturating_sub(self.missing);
        BucketStats {
            points: self.n,
            missing: self.missing,
            #[expect(clippy::cast_precision_loss, reason = "a count as a divisor")]
            rmse_vol: if evaluated == 0 {
                0.0
            } else {
                (self.sum_sq / evaluated as f64).sqrt()
            },
            max_abs_vol: self.max_abs,
        }
    }
}

fn residual(model: &impl VolSurface, o: &IvObservation) -> Option<f64> {
    let tau = o.quote.tau;
    let w = model.total_variance(o.quote.k, tau).ok()?.w();
    Some((o.w_mid / tau).sqrt() - (w / tau).sqrt())
}

/// Splits `observations` by the rule, trains SVI and the baseline on the
/// same training quotes, and scores both on the held out ones.
///
/// # Errors
///
/// [`MMError::InvalidConfiguration`] for an invalid fit configuration, or
/// when fewer than two expiries can be trained.
pub fn validate_holdout(
    observations: &[IvObservation],
    fit: &FitConfig,
    rule: HoldoutRule,
    wings: WingRule,
) -> MMResult<HoldoutReport> {
    fit.validate()?;
    let mut by_expiry: BTreeMap<u64, Vec<&IvObservation>> = BTreeMap::new();
    for o in observations {
        by_expiry
            .entry(o.quote.quote.expiry_at)
            .or_default()
            .push(o);
    }
    let mut not_applicable = Vec::new();
    let mut fit_failures = Vec::new();
    let mut svi_slices = Vec::new();
    let mut baseline_slices = Vec::new();
    let mut held: Vec<&IvObservation> = Vec::new();
    let mut atm: BTreeMap<u64, f64> = BTreeMap::new();
    for (expiry, mut group) in by_expiry {
        group.sort_by(|a, b| a.quote.k.total_cmp(&b.quote.k));
        let (train, test): (Vec<_>, Vec<_>) = if group.len() >= rule.min_quotes {
            let (t, h): (Vec<_>, Vec<_>) = group.iter().enumerate().partition(|(i, _)| i % 2 == 0);
            (
                t.into_iter().map(|(_, o)| *o).collect(),
                h.into_iter().map(|(_, o)| *o).collect(),
            )
        } else {
            not_applicable.push((expiry, group.len()));
            (group.clone(), Vec::new())
        };
        let tau = train.first().map_or(0.0, |o| o.quote.tau);
        let points: MMResult<Vec<SliceObservation>> = train
            .iter()
            .map(|o| SliceObservation::new(o.quote.k, o.w_mid, o.weight))
            .collect();
        match fit_slice(tau, &points?, fit) {
            Ok(f) => {
                if let Ok((w0, _, _)) = f.slice.eval(0.0) {
                    atm.insert(expiry, w0);
                }
                svi_slices.push(f.slice);
                let base: Vec<(f64, f64)> = train
                    .iter()
                    .filter(|o| o.w_mid > 0.0)
                    .map(|o| (o.quote.k, o.w_mid))
                    .collect();
                baseline_slices.push((tau, base));
                held.extend(test);
            }
            Err(failure) => fit_failures.push((expiry, failure.to_string())),
        }
    }
    let svi = SviSurface::new(svi_slices, ShortExpiryPolicy::Reject)?;
    let baseline = BaselineSurface::new(baseline_slices)?;
    let shortest_expiry = atm.keys().next().copied();
    let mut acc: [[Acc; 2]; 4] = Default::default();
    for o in held {
        let expiry = o.quote.quote.expiry_at;
        let k = o.quote.k;
        let z = atm
            .get(&expiry)
            .map_or(f64::INFINITY, |w| k.abs() / w.sqrt());
        let buckets = [
            true,
            z > wings.z_wing,
            k.abs() > wings.abs_k,
            Some(expiry) == shortest_expiry,
        ];
        let (rs, rb) = (residual(&svi, o), residual(&baseline, o));
        for (bucket, member) in acc.iter_mut().zip(buckets) {
            if member {
                let [s, b] = bucket;
                s.add(rs);
                b.add(rb);
            }
        }
    }
    let cmp = |a: &[Acc; 2]| BucketComparison {
        svi: a[0].stats(),
        baseline: a[1].stats(),
    };
    Ok(HoldoutReport {
        not_applicable,
        fit_failures,
        all: cmp(&acc[0]),
        wings: cmp(&acc[1]),
        wings_abs: cmp(&acc[2]),
        shortest: cmp(&acc[3]),
    })
}
