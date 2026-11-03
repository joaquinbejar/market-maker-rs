//! Out of sample validation of the SVI fit against a simple baseline.
//!
//! Everything about the targets is fixed before any fit, by a declared
//! rule: per expiry, the out of the money quotes are sorted by `k` and
//! every other one (odd positions) is held out, but only for an expiry with
//! at least [`HoldoutRule::min_quotes`] quotes, so that at least half are
//! left to train on; any other expiry is reported as not applicable with
//! its size, and is not fitted. The shortest expiry with a holdout is fixed
//! with the split. Nothing (grid, weights, tolerances) is chosen with the
//! held out quotes.
//!
//! Per expiry, SVI ([`fit_slice`]) and the baseline ([`BaselineSlice`],
//! linear in `w` between the quoted `k`, flat beyond) are each trained on
//! exactly the same quotes, independently: a failure of one leaves the
//! other scored. Each held out quote is scored on its own expiry's model.
//!
//! Residuals are observed minus model implied volatility. They are reported
//! by bucket and by expiry, each with its size and the points a model could
//! not evaluate, its fit having failed included (never dropped to improve a
//! metric):
//!
//! - **wings**: `|k| / sqrt(w_atm) > z_wing`, the preregistered threshold,
//!   with `w_atm` the trained SVI slice's `w(0)` (never the held out data);
//!   where the SVI fit failed there is no `w_atm`, and those points are
//!   counted as unassigned rather than guessed into or out of the bucket;
//! - **auxiliary wings**: `|k| > abs_k`, a separate fixed cut;
//! - **shortest expiry**;
//! - **all**.

use std::collections::BTreeMap;

use super::calibrate::{FitConfig, SliceObservation, fit_slice};
use super::iv::IvObservation;
use super::svi::SviSlice;
use crate::types::error::{MMError, MMResult};

/// One expiry of the baseline: linear in `w` between the quoted `k`, flat
/// beyond them.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineSlice {
    points: Vec<(f64, f64)>,
}

impl BaselineSlice {
    /// From `(k, w)` points.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] without points, or for a value
    /// that is not finite, a `w` that is not positive, or a repeated `k`.
    pub fn new(mut points: Vec<(f64, f64)>) -> MMResult<Self> {
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ok = !points.is_empty()
            && points
                .iter()
                .all(|(k, w)| k.is_finite() && w.is_finite() && *w > 0.0)
            && points.windows(2).all(|p| matches!(p, [a, b] if b.0 > a.0));
        if !ok {
            return Err(MMError::InvalidConfiguration(
                "baseline: points must be finite, w positive and k strictly increasing".to_string(),
            ));
        }
        Ok(Self { points })
    }

    /// The total variance at `k`.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a `k` that is not finite.
    pub fn w(&self, k: f64) -> MMResult<f64> {
        let (Some(&(k0, w0)), Some(&(kn, wn))) = (self.points.first(), self.points.last()) else {
            return Err(MMError::InvalidConfiguration(
                "empty baseline slice".to_string(),
            ));
        };
        if !k.is_finite() {
            return Err(MMError::InvalidConfiguration(format!(
                "baseline: k must be finite, got {k}"
            )));
        }
        if k <= k0 {
            return Ok(w0);
        }
        if k >= kn {
            return Ok(wn);
        }
        for pair in self.points.windows(2) {
            if let [(ka, wa), (kb, wb)] = pair
                && k <= *kb
            {
                return Ok(wa + (wb - wa) * (k - ka) / (kb - ka));
            }
        }
        Err(MMError::InvalidConfiguration(
            "baseline bracket not found".to_string(),
        ))
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

/// One expiry with a holdout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpiryHoldout {
    /// Expiry, ms.
    pub expiry_at: u64,
    /// Out of the money quotes of the expiry.
    pub quotes: usize,
    /// Of which held out.
    pub held: usize,
    /// SVI and baseline on the held out quotes.
    pub comparison: BucketComparison,
}

/// The holdout validation.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutReport {
    /// Expiries (ms) without a holdout, with their size.
    pub not_applicable: Vec<(u64, usize)>,
    /// Expiries whose SVI slice failed to fit, with the reason; their held
    /// out quotes count as missing for SVI.
    pub fit_failures: Vec<(u64, String)>,
    /// Expiries whose baseline could not be built, with the reason; their
    /// held out quotes count as missing for the baseline.
    pub baseline_failures: Vec<(u64, String)>,
    /// Each expiry with a holdout, by increasing expiry.
    pub expiries: Vec<ExpiryHoldout>,
    /// The shortest expiry with a holdout, fixed with the split.
    pub shortest_expiry: Option<u64>,
    /// Held out quotes whose wing membership is undecided: no trained SVI
    /// `w_atm` for their expiry. Left out of the `wings` bucket only.
    pub wings_unassigned: usize,
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

#[derive(Default)]
struct Pair([Acc; 2]);

impl Pair {
    fn add(&mut self, svi: Option<f64>, baseline: Option<f64>) {
        let [s, b] = &mut self.0;
        s.add(svi);
        b.add(baseline);
    }

    fn comparison(&self) -> BucketComparison {
        let [s, b] = &self.0;
        BucketComparison {
            svi: s.stats(),
            baseline: b.stats(),
        }
    }
}

/// Observed minus model implied volatility, for a model `w` that is finite
/// and positive.
fn vol_residual(o: &IvObservation, w: Option<f64>) -> Option<f64> {
    let tau = o.quote.tau;
    let w = w.filter(|w| w.is_finite() && *w > 0.0)?;
    let r = (o.w_mid / tau).sqrt() - (w / tau).sqrt();
    r.is_finite().then_some(r)
}

/// One expiry's split, fixed before any fit.
struct Split<'a> {
    expiry: u64,
    quotes: usize,
    train: Vec<&'a IvObservation>,
    test: Vec<&'a IvObservation>,
}

/// Splits `observations` by the rule, trains SVI and the baseline on the
/// same training quotes of each expiry, independently, and scores both on
/// the held out ones.
///
/// # Errors
///
/// [`MMError::InvalidConfiguration`] for an invalid fit configuration.
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
    // 1. The split and the shortest expiry, before any fit.
    let mut not_applicable = Vec::new();
    let mut splits = Vec::new();
    for (expiry, mut group) in by_expiry {
        if group.len() < rule.min_quotes {
            not_applicable.push((expiry, group.len()));
            continue;
        }
        group.sort_by(|a, b| a.quote.k.total_cmp(&b.quote.k));
        let quotes = group.len();
        let (train, test): (Vec<_>, Vec<_>) =
            group.into_iter().enumerate().partition(|(i, _)| i % 2 == 0);
        splits.push(Split {
            expiry,
            quotes,
            train: train.into_iter().map(|(_, o)| o).collect(),
            test: test.into_iter().map(|(_, o)| o).collect(),
        });
    }
    let shortest_expiry = splits.first().map(|s| s.expiry);
    // 2. Fits on the training quotes, each model on its own; 3. scores.
    let mut fit_failures = Vec::new();
    let mut baseline_failures = Vec::new();
    let mut expiries = Vec::with_capacity(splits.len());
    let mut wings_unassigned = 0_usize;
    let (mut all, mut wing, mut wing_abs, mut shortest) = Default::default();
    for split in &splits {
        let tau = split.train.first().map_or(0.0, |o| o.quote.tau);
        let points: MMResult<Vec<SliceObservation>> = split
            .train
            .iter()
            .map(|o| SliceObservation::new(o.quote.k, o.w_mid, o.weight))
            .collect();
        let svi: Option<SviSlice> = match points
            .map_err(|e| e.to_string())
            .and_then(|p| fit_slice(tau, &p, fit).map_err(|e| e.to_string()))
        {
            Ok(f) => Some(f.slice),
            Err(why) => {
                fit_failures.push((split.expiry, why));
                None
            }
        };
        let baseline =
            match BaselineSlice::new(split.train.iter().map(|o| (o.quote.k, o.w_mid)).collect()) {
                Ok(b) => Some(b),
                Err(error) => {
                    baseline_failures.push((split.expiry, error.to_string()));
                    None
                }
            };
        let w_atm = svi
            .as_ref()
            .and_then(|s| s.eval(0.0).ok())
            .map(|(w, _, _)| w)
            .filter(|w| w.is_finite() && *w > 0.0);
        let mut per_expiry = Pair::default();
        for o in &split.test {
            let k = o.quote.k;
            let rs = vol_residual(o, svi.as_ref().and_then(|s| s.eval(k).ok()).map(|v| v.0));
            let rb = vol_residual(o, baseline.as_ref().and_then(|b| b.w(k).ok()));
            per_expiry.add(rs, rb);
            Pair::add(&mut all, rs, rb);
            match w_atm {
                Some(w) if k.abs() / w.sqrt() > wings.z_wing => Pair::add(&mut wing, rs, rb),
                Some(_) => {}
                None => wings_unassigned = wings_unassigned.saturating_add(1),
            }
            if k.abs() > wings.abs_k {
                Pair::add(&mut wing_abs, rs, rb);
            }
            if Some(split.expiry) == shortest_expiry {
                Pair::add(&mut shortest, rs, rb);
            }
        }
        expiries.push(ExpiryHoldout {
            expiry_at: split.expiry,
            quotes: split.quotes,
            held: split.test.len(),
            comparison: per_expiry.comparison(),
        });
    }
    Ok(HoldoutReport {
        not_applicable,
        fit_failures,
        baseline_failures,
        expiries,
        shortest_expiry,
        wings_unassigned,
        all: Pair::comparison(&all),
        wings: Pair::comparison(&wing),
        wings_abs: Pair::comparison(&wing_abs),
        shortest: Pair::comparison(&shortest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_slice_interpolates_in_w_and_is_flat_beyond() {
        let b = BaselineSlice::new(vec![(0.1, 0.05), (-0.1, 0.03)]).unwrap();
        assert_eq!(b.w(-0.5).unwrap(), 0.03);
        assert_eq!(b.w(0.5).unwrap(), 0.05);
        assert!((b.w(0.0).unwrap() - 0.04).abs() < 1e-15);
        assert!(b.w(f64::NAN).is_err());
    }

    #[test]
    fn test_baseline_slice_invalid_points_return_err() {
        assert!(BaselineSlice::new(Vec::new()).is_err());
        assert!(BaselineSlice::new(vec![(0.0, 0.0)]).is_err());
        assert!(BaselineSlice::new(vec![(0.0, 0.01), (0.0, 0.02)]).is_err());
        assert!(BaselineSlice::new(vec![(f64::NAN, 0.01)]).is_err());
    }
}
