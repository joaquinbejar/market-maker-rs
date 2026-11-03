//! Causal availability of calibrated surfaces.
//!
//! Every calibration attempt is stored with two times, both in
//! milliseconds since the Unix epoch, UTC:
//!
//! - `data_cutoff`: the latest data the calibration read (its `as_of`);
//! - `available_at`: when the calibration finished, so `>= data_cutoff`.
//!
//! [`SurfaceStore::surface_at`] only sees attempts with `available_at <= t`:
//! a calibration that finishes late is invisible until then, and the age of
//! a surface is always `t - data_cutoff`, so a late calculation does not
//! make old data look fresh. The store is append only, with strictly
//! increasing `available_at`, so a past answer never changes. A
//! calibration of older data may finish after a newer one; it is stored,
//! and served only when nothing with newer data is usable.
//!
//! # Which attempt is served
//!
//! Among the attempts available at `t`, the one with the newest data cutoff
//! that the [`UsePolicy`] allows and whose aged surface is usable; between
//! attempts on the same cutoff, the later available one.
//!
//! # Use is a separate decision from quality
//!
//! Each attempt carries a [`SurfaceStatus`] from its calibration. A
//! [`UsePolicy`] decides what may be served: a `Failed` attempt never, a
//! `Degraded` one only when the policy allows it, and nothing older than
//! `max_age_ms`. Storing or serving a surface never changes its status.
//!
//! # Ageing
//!
//! Slices keep their absolute expiry. Served at `t`, each slice is rolled
//! down at constant implied volatility per `k`:
//! `w_i(k; t) = w_i(k) tau_i(t) / tau_i(data_cutoff)`, which scales `a` and
//! `b` and keeps `rho`, `m` and `sigma`. Slices expired by `t` are dropped.
//! For slices with absolute expiries calibrated at the same cutoff and
//! `t >= cutoff`, the factor `tau_i(t) / tau_i(cutoff)` is positive and grows
//! with the expiry, so `w_i <= w_j` at each `k` before ageing stays true
//! after it; nothing more (no butterfly property, nothing for a surface
//! that was not ordered) is claimed. This is an explicit ageing policy,
//! not a guarantee: the aged surface is
//! diagnosed again on a grid holding all its nodes, and under a policy that
//! rejects degraded surfaces it is served only if that diagnosis is clean.
//! An old expiry is never read as a new one.

use super::diagnostics::{DiagnosticsConfig, SurfaceDiagnostics, diagnose};
use super::svi::{SviSlice, SviSurface};
use crate::types::error::{MMError, MMResult};

/// Milliseconds in the 365 day year of the surface conventions.
pub const YEAR_MS: f64 = 365.0 * 86_400_000.0;

/// Relative tolerance between a slice's `tau` and its expiry at the cutoff.
const TAU_CONSISTENCY: f64 = 1e-9;

/// The verdict of a calibration on its own surface.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceStatus {
    /// Every check passed.
    Healthy,
    /// Usable with care; the reasons are listed.
    Degraded(Vec<String>),
    /// Not usable; the reasons are listed.
    Failed(Vec<String>),
}

/// One calibration attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedSurface {
    surface: Option<SviSurface>,
    expiries_at: Vec<u64>,
    data_cutoff: u64,
    available_at: u64,
    status: SurfaceStatus,
}

impl CalibratedSurface {
    /// A calibrated surface whose slice `i` expires at `expiries_at[i]`.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] when `available_at < data_cutoff`,
    /// the expiries do not match the slices one to one, an expiry is not
    /// after the cutoff, or a slice's `tau` differs from
    /// `(expiry - data_cutoff) / YEAR_MS`.
    pub fn new(
        surface: SviSurface,
        expiries_at: Vec<u64>,
        data_cutoff: u64,
        available_at: u64,
        status: SurfaceStatus,
    ) -> MMResult<Self> {
        if available_at < data_cutoff {
            return Err(MMError::InvalidConfiguration(format!(
                "available_at {available_at} is before data_cutoff {data_cutoff}"
            )));
        }
        if expiries_at.len() != surface.slices().len() {
            return Err(MMError::InvalidConfiguration(format!(
                "{} expiries for {} slices",
                expiries_at.len(),
                surface.slices().len()
            )));
        }
        for (slice, &expiry) in surface.slices().iter().zip(expiries_at.iter()) {
            let tau = tau_between(data_cutoff, expiry)?;
            if (tau - slice.tau()).abs() > TAU_CONSISTENCY * slice.tau() {
                return Err(MMError::InvalidConfiguration(format!(
                    "slice tau {} does not match expiry {expiry} at cutoff {data_cutoff} ({tau})",
                    slice.tau()
                )));
            }
        }
        Ok(Self {
            surface: Some(surface),
            expiries_at,
            data_cutoff,
            available_at,
            status,
        })
    }

    /// A failed attempt with no surface, recorded so that its failure is
    /// part of the history.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] when `available_at < data_cutoff`.
    pub fn failed(data_cutoff: u64, available_at: u64, reasons: Vec<String>) -> MMResult<Self> {
        if available_at < data_cutoff {
            return Err(MMError::InvalidConfiguration(format!(
                "available_at {available_at} is before data_cutoff {data_cutoff}"
            )));
        }
        Ok(Self {
            surface: None,
            expiries_at: Vec::new(),
            data_cutoff,
            available_at,
            status: SurfaceStatus::Failed(reasons),
        })
    }

    /// The calibrated surface, at its cutoff; `None` for a failed attempt
    /// without one.
    #[must_use]
    pub fn surface(&self) -> Option<&SviSurface> {
        self.surface.as_ref()
    }

    /// Absolute expiry of each slice, ms.
    #[must_use]
    pub fn expiries_at(&self) -> &[u64] {
        &self.expiries_at
    }

    /// Latest data the calibration read, ms.
    #[must_use]
    pub fn data_cutoff(&self) -> u64 {
        self.data_cutoff
    }

    /// When the calibration finished, ms.
    #[must_use]
    pub fn available_at(&self) -> u64 {
        self.available_at
    }

    /// The calibration's verdict.
    #[must_use]
    pub fn status(&self) -> &SurfaceStatus {
        &self.status
    }

    /// The surface rolled down to `t` (see the [module docs](self)).
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidMarketState`] when `t` is before the cutoff, there
    /// is no surface, or fewer than two slices are alive at `t`.
    pub fn aged(&self, t: u64) -> MMResult<SviSurface> {
        let Some(surface) = self.surface.as_ref() else {
            return Err(MMError::InvalidMarketState(
                "a failed calibration has no surface".to_string(),
            ));
        };
        let slices = self.aged_slices(t)?.into_iter().map(|(_, s)| s).collect();
        SviSurface::new(slices, surface.short_expiry()).map_err(|e| {
            MMError::InvalidMarketState(format!("surface aged to {t} is not usable: {e}"))
        })
    }

    /// The slices alive at `t`, rolled down to `t`, with their expiries.
    /// Each keeps its `k` range; its `tau` becomes `tau(t)`.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidMarketState`] when `t` is before the cutoff or
    /// there is no surface.
    pub fn aged_slices(&self, t: u64) -> MMResult<Vec<(u64, SviSlice)>> {
        let Some(surface) = self.surface.as_ref() else {
            return Err(MMError::InvalidMarketState(
                "a failed calibration has no surface".to_string(),
            ));
        };
        if t < self.data_cutoff {
            return Err(MMError::InvalidMarketState(format!(
                "valuation time {t} is before the data cutoff {}",
                self.data_cutoff
            )));
        }
        let mut slices = Vec::new();
        for (slice, &expiry) in surface.slices().iter().zip(self.expiries_at.iter()) {
            if expiry <= t {
                continue;
            }
            let tau_t = tau_between(t, expiry)?;
            let scale = tau_t / slice.tau();
            slices.push((
                expiry,
                SviSlice::new(
                    tau_t,
                    slice.a() * scale,
                    slice.b() * scale,
                    slice.rho(),
                    slice.m(),
                    slice.sigma(),
                    slice.k_range(),
                )?,
            ));
        }
        Ok(slices)
    }
}

/// Largest millisecond count an `f64` holds exactly, `2^53`.
const EXACT_MS: u64 = 1 << 53;

/// `(to - from) / YEAR_MS` for `to > from`; the difference converts to
/// `f64` exactly, as it is at most `2^53` (about 285 000 years).
pub(crate) fn tau_between(from: u64, to: u64) -> MMResult<f64> {
    let ms = to
        .checked_sub(from)
        .filter(|ms| *ms > 0)
        .ok_or_else(|| MMError::InvalidConfiguration(format!("expiry {to} is not after {from}")))?;
    if ms > EXACT_MS {
        return Err(MMError::InvalidConfiguration(format!(
            "{ms} ms between {from} and {to} is not exact in f64"
        )));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "at most 2^53, checked above: exact"
    )]
    let ms = ms as f64;
    Ok(ms / YEAR_MS)
}

/// What may be served.
#[derive(Debug, Clone, PartialEq)]
pub struct UsePolicy {
    /// Serve `Degraded` surfaces, and aged surfaces whose own diagnosis is
    /// not clean.
    pub allow_degraded: bool,
    /// Largest `t - data_cutoff` served, ms.
    pub max_age_ms: u64,
    /// `k` grid of the aged surface's diagnosis (its `tau` axis holds every
    /// node and segment midpoint).
    pub k_grid: Vec<f64>,
    /// Tolerances of that diagnosis.
    pub diagnostics: DiagnosticsConfig,
}

/// A surface served at `t`.
#[derive(Debug, Clone, PartialEq)]
pub struct ServedSurface {
    /// The surface rolled down to `t`.
    pub surface: SviSurface,
    /// The valuation time, ms.
    pub valuation_at: u64,
    /// `t - data_cutoff`, ms.
    pub age_ms: u64,
    /// The cutoff of the calibration served, ms.
    pub data_cutoff: u64,
    /// When that calibration became available, ms.
    pub available_at: u64,
    /// The stored calibration's status, unchanged.
    pub status: SurfaceStatus,
    /// The status of what is served: the stored one when the aged surface's
    /// diagnosis is clean, `Degraded` otherwise (served only under a policy
    /// that allows degraded surfaces).
    pub served_status: SurfaceStatus,
    /// Whether an attempt preferred by the serving rule (newer data, or the
    /// same data available later), available at `t`, was passed over.
    pub fallback: bool,
    /// Diagnosis of the aged surface.
    pub diagnostics: SurfaceDiagnostics,
}

/// Every calibration attempt, append only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceStore {
    entries: Vec<CalibratedSurface>,
}

impl SurfaceStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an attempt.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidTimestamp`] unless its `available_at` is after the
    /// last one's: the past is never rewritten.
    pub fn insert(&mut self, entry: CalibratedSurface) -> MMResult<()> {
        if let Some(last) = self.entries.last()
            && entry.available_at <= last.available_at
        {
            return Err(MMError::InvalidTimestamp(format!(
                "available_at {} is not after the last one, {}",
                entry.available_at, last.available_at
            )));
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Every attempt, in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[CalibratedSurface] {
        &self.entries
    }

    /// The attempt available at `t` with the newest data cutoff that
    /// `policy` allows (ties go to the later available one), aged to `t` and
    /// diagnosed. See the [module docs](self).
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidMarketState`] when nothing is usable at `t`: no
    /// attempt available yet, every candidate failed, degraded under a
    /// strict policy, too old, or not alive at `t`.
    pub fn surface_at(&self, t: u64, policy: &UsePolicy) -> MMResult<ServedSurface> {
        // Every attempt available at t, by the serving rule: newest data
        // first, then latest available.
        let mut candidates: Vec<&CalibratedSurface> = self
            .entries
            .iter()
            .filter(|e| e.available_at <= t)
            .collect();
        if candidates.is_empty() {
            return Err(MMError::InvalidMarketState(format!(
                "no surface is available at {t}"
            )));
        }
        candidates.sort_by(|a, b| {
            b.data_cutoff
                .cmp(&a.data_cutoff)
                .then(b.available_at.cmp(&a.available_at))
        });
        let mut reasons = Vec::new();
        for (rank, entry) in candidates.into_iter().enumerate() {
            match &entry.status {
                SurfaceStatus::Failed(_) => {
                    reasons.push(format!("{}: failed", entry.available_at));
                    continue;
                }
                SurfaceStatus::Degraded(_) if !policy.allow_degraded => {
                    reasons.push(format!("{}: degraded", entry.available_at));
                    continue;
                }
                _ => {}
            }
            let age_ms = t.checked_sub(entry.data_cutoff).ok_or_else(|| {
                MMError::InvalidMarketState(format!(
                    "data cutoff {} is after {t}",
                    entry.data_cutoff
                ))
            })?;
            if age_ms > policy.max_age_ms {
                // Candidates are in decreasing data cutoff order: every one
                // left is at least as old.
                reasons.push(format!("{}: {age_ms} ms old", entry.available_at));
                break;
            }
            let surface = match entry.aged(t) {
                Ok(surface) => surface,
                Err(error) => {
                    reasons.push(format!("{}: {error}", entry.available_at));
                    continue;
                }
            };
            let grid = surface.node_grid(policy.k_grid.clone(), &[])?;
            let diagnostics = diagnose(&surface, &grid, policy.diagnostics)?;
            if !diagnostics.is_clean() && !policy.allow_degraded {
                reasons.push(format!("{}: aged surface not clean", entry.available_at));
                continue;
            }
            let served_status = if diagnostics.is_clean() {
                entry.status.clone()
            } else {
                let mut reasons = match &entry.status {
                    SurfaceStatus::Degraded(r) => r.clone(),
                    _ => Vec::new(),
                };
                reasons.push(format!("aged to {t}: diagnosis not clean"));
                SurfaceStatus::Degraded(reasons)
            };
            return Ok(ServedSurface {
                surface,
                valuation_at: t,
                age_ms,
                data_cutoff: entry.data_cutoff,
                available_at: entry.available_at,
                status: entry.status.clone(),
                served_status,
                fallback: rank != 0,
                diagnostics,
            });
        }
        Err(MMError::InvalidMarketState(format!(
            "no usable surface at {t}: {}",
            reasons.join("; ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::surface::{KRange, ShortExpiryPolicy, VolSurface};

    const DAY: u64 = 86_400_000;

    /// Slices expiring 30, 60 and 120 days after `cutoff`, with level `a0`.
    fn calibrated(
        cutoff: u64,
        available: u64,
        a0: f64,
        status: SurfaceStatus,
    ) -> CalibratedSurface {
        let range = KRange::new(-0.5, 0.5).unwrap();
        let expiries: Vec<u64> = [30, 60, 120].iter().map(|d| cutoff + d * DAY).collect();
        let slices = expiries
            .iter()
            .map(|&e| {
                let tau = (e - cutoff) as f64 / YEAR_MS;
                SviSlice::new(tau, a0 * tau, 0.1 * tau.sqrt(), -0.4, 0.0, 0.2, range).unwrap()
            })
            .collect();
        let surface = SviSurface::new(slices, ShortExpiryPolicy::Reject).unwrap();
        CalibratedSurface::new(surface, expiries, cutoff, available, status).unwrap()
    }

    fn policy(allow_degraded: bool) -> UsePolicy {
        UsePolicy {
            allow_degraded,
            max_age_ms: 10 * DAY,
            k_grid: vec![-0.5, -0.25, 0.0, 0.25, 0.5],
            diagnostics: DiagnosticsConfig::default(),
        }
    }

    #[test]
    fn test_calibrated_surface_inconsistent_times_return_err() {
        let base = calibrated(0, 0, 0.04, SurfaceStatus::Healthy);
        let surface = base.surface().unwrap().clone();
        let expiries = base.expiries_at().to_vec();
        assert!(
            CalibratedSurface::new(
                surface.clone(),
                expiries.clone(),
                10,
                5,
                SurfaceStatus::Healthy
            )
            .is_err()
        );
        assert!(
            CalibratedSurface::new(
                surface.clone(),
                expiries[..2].to_vec(),
                0,
                0,
                SurfaceStatus::Healthy
            )
            .is_err()
        );
        // Expiries that do not match the slices' tau at the cutoff.
        assert!(
            CalibratedSurface::new(surface, expiries, DAY, DAY, SurfaceStatus::Healthy).is_err()
        );
    }

    #[test]
    fn test_store_is_append_only() {
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(0, 100, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        assert!(
            store
                .insert(calibrated(0, 100, 0.04, SurfaceStatus::Healthy))
                .is_err()
        );
        assert!(
            store
                .insert(calibrated(0, 50, 0.04, SurfaceStatus::Healthy))
                .is_err()
        );
        assert_eq!(store.entries().len(), 1);
    }

    #[test]
    fn test_surface_at_older_data_finishing_later_does_not_hide_newer_data() {
        // Data to 2000 available at 2500, then data to 1000 available at
        // 3000.
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(2000, 2500, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        let p = UsePolicy {
            max_age_ms: 1500,
            ..policy(false)
        };
        let before = store.surface_at(3000, &p).unwrap();
        assert_eq!(before.data_cutoff, 2000);
        store
            .insert(calibrated(1000, 3000, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        // The newer data is still served, and nothing preferred was passed
        // over.
        let after = store.surface_at(3000, &p).unwrap();
        assert_eq!(after, before);
        assert!(!after.fallback);
        // With a looser age limit, still the newer data.
        let loose = UsePolicy {
            max_age_ms: 10 * DAY,
            ..policy(false)
        };
        assert_eq!(store.surface_at(3000, &loose).unwrap().data_cutoff, 2000);
        // The older data is served only when the newer cannot be: a failed
        // retry on the newest cutoff does not change the answer.
        store
            .insert(CalibratedSurface::failed(2000, 3500, vec!["no fit".into()]).unwrap())
            .unwrap();
        let ranked = store.surface_at(3500, &loose).unwrap();
        assert_eq!(
            (ranked.data_cutoff, ranked.available_at, ranked.fallback),
            (2000, 2500, true)
        );
    }

    #[test]
    fn test_surface_at_same_cutoff_prefers_the_later_attempt() {
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(DAY, DAY + 10, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        store
            .insert(calibrated(DAY, DAY + 20, 0.05, SurfaceStatus::Healthy))
            .unwrap();
        let served = store.surface_at(2 * DAY, &policy(false)).unwrap();
        assert_eq!(served.available_at, DAY + 20);
        assert!(!served.fallback);
    }

    #[test]
    fn test_surface_at_sees_only_what_is_available() {
        let mut store = SurfaceStore::new();
        // Data to day 1, finished late on day 2.
        store
            .insert(calibrated(DAY, 2 * DAY, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        store
            .insert(calibrated(
                3 * DAY,
                3 * DAY + 1000,
                0.05,
                SurfaceStatus::Healthy,
            ))
            .unwrap();
        let p = policy(false);
        assert!(store.surface_at(2 * DAY - 1, &p).is_err());
        let first = store.surface_at(2 * DAY, &p).unwrap();
        assert_eq!(first.data_cutoff, DAY);
        // Age counts from the data cutoff, not from availability.
        assert_eq!(first.age_ms, DAY);
        assert!(!first.fallback);
        let still_first = store.surface_at(3 * DAY + 999, &p).unwrap();
        assert_eq!(still_first.data_cutoff, DAY);
        let second = store.surface_at(3 * DAY + 1000, &p).unwrap();
        assert_eq!(second.data_cutoff, 3 * DAY);
    }

    #[test]
    fn test_surface_at_later_inserts_do_not_change_past_answers() {
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(0, DAY, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        let p = policy(false);
        let before = store.surface_at(2 * DAY, &p).unwrap();
        store
            .insert(calibrated(2 * DAY, 5 * DAY, 0.09, SurfaceStatus::Healthy))
            .unwrap();
        assert_eq!(store.surface_at(2 * DAY, &p).unwrap(), before);
    }

    #[test]
    fn test_surface_at_never_serves_failed_and_respects_degraded_policy() {
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(0, DAY, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        store
            .insert(calibrated(
                DAY,
                2 * DAY,
                0.05,
                SurfaceStatus::Degraded(vec!["wide".into()]),
            ))
            .unwrap();
        store
            .insert(CalibratedSurface::failed(2 * DAY, 3 * DAY, vec!["no fit".into()]).unwrap())
            .unwrap();
        let strict = store.surface_at(3 * DAY, &policy(false)).unwrap();
        assert_eq!(strict.data_cutoff, 0);
        assert!(strict.fallback);
        let lenient = store.surface_at(3 * DAY, &policy(true)).unwrap();
        assert_eq!(lenient.data_cutoff, DAY);
        assert_eq!(lenient.status, SurfaceStatus::Degraded(vec!["wide".into()]));
        assert!(lenient.fallback);
    }

    #[test]
    fn test_surface_at_too_old_returns_err() {
        let mut store = SurfaceStore::new();
        store
            .insert(calibrated(0, DAY, 0.04, SurfaceStatus::Healthy))
            .unwrap();
        assert!(store.surface_at(11 * DAY, &policy(false)).is_err());
    }

    #[test]
    fn test_aged_surface_rolls_down_at_constant_vol_and_drops_expired() {
        let entry = calibrated(0, 0, 0.04, SurfaceStatus::Healthy);
        let t = 10 * DAY;
        let aged = entry.aged(t).unwrap();
        assert_eq!(aged.slices().len(), 3);
        let original = entry.surface().unwrap();
        for (old, new) in original.slices().iter().zip(aged.slices()) {
            let expected_tau = old.tau() - 10.0 * DAY as f64 / YEAR_MS;
            assert!((new.tau() - expected_tau).abs() < 1e-15);
            // Same implied volatility at every k.
            for k in [-0.4, 0.0, 0.3] {
                let vol_old = (old.eval(k).unwrap().0 / old.tau()).sqrt();
                let vol_new = (new.eval(k).unwrap().0 / new.tau()).sqrt();
                assert!((vol_old - vol_new).abs() < 1e-12);
            }
        }
        // After 45 days the 30 day slice has expired.
        let later = entry.aged(45 * DAY).unwrap();
        assert_eq!(later.slices().len(), 2);
        // After 90 days only one slice is alive: not a surface.
        assert!(entry.aged(90 * DAY).is_err());
        assert!(
            entry
                .aged(0)
                .unwrap()
                .total_variance(0.0, 30.0 * DAY as f64 / YEAR_MS)
                .is_ok()
        );
    }

    #[test]
    fn test_surface_at_aged_diagnosis_gates_strict_policy() {
        // The status says Healthy, but the surface (Vogt's slice, a known
        // butterfly violation) is not: the aged surface's own diagnosis is
        // what a strict policy serves on, never the stored status alone.
        let range = KRange::new(-1.5, 1.5).unwrap();
        let tau = |d: u64| (d * DAY) as f64 / YEAR_MS;
        let vogt =
            |d| SviSlice::new(tau(d), -0.0410, 0.1331, 0.3060, 0.3586, 0.4153, range).unwrap();
        let surface =
            SviSurface::new(vec![vogt(365), vogt(730)], ShortExpiryPolicy::Reject).unwrap();
        let entry = CalibratedSurface::new(
            surface,
            vec![365 * DAY, 730 * DAY],
            0,
            0,
            SurfaceStatus::Healthy,
        )
        .unwrap();
        let mut store = SurfaceStore::new();
        store.insert(entry).unwrap();
        let mut wide = policy(true);
        wide.k_grid = (0..=60).map(|i| -1.5 + 0.05 * f64::from(i)).collect();
        let lenient = store.surface_at(5 * DAY, &wide).unwrap();
        assert!(!lenient.diagnostics.butterfly.is_clean());
        // Stored as Healthy, served as Degraded: the two are not confused.
        assert_eq!(lenient.status, SurfaceStatus::Healthy);
        assert!(matches!(lenient.served_status, SurfaceStatus::Degraded(_)));
        wide.allow_degraded = false;
        assert!(store.surface_at(5 * DAY, &wide).is_err());
    }

    #[test]
    fn test_served_diagnosis_covers_the_aged_and_shocked_nodes() {
        let mut store = SurfaceStore::new();
        let entry = calibrated(DAY, DAY, 0.04, SurfaceStatus::Healthy);
        let calibrated_grid = entry
            .surface()
            .unwrap()
            .node_grid(policy(false).k_grid, &[])
            .unwrap();
        store.insert(entry).unwrap();
        let served = store.surface_at(6 * DAY, &policy(false)).unwrap();
        assert_eq!(served.served_status, SurfaceStatus::Healthy);
        // The served status was decided on the aged surface's own nodes, not
        // on the calibration's.
        assert!(served.surface.covers_nodes(&served.diagnostics.grid));
        assert!(!served.surface.covers_nodes(&calibrated_grid));
        // Shocks move no node: each shocked surface is diagnosed on the same
        // grid, which holds its nodes.
        let scenarios = crate::options::surface::run_scenarios(
            &served.surface,
            &crate::options::surface::standard_scenarios().unwrap(),
            &served.diagnostics.grid,
            policy(false).diagnostics,
        )
        .unwrap();
        for scenario in &scenarios {
            assert!(served.surface.covers_nodes(&scenario.diagnostics.grid));
        }
    }

    #[test]
    fn test_ageing_preserves_calendar_order() {
        // tau_i(t) / tau_i(cutoff) grows with the expiry, so w_2 >= w_1 at
        // the cutoff stays true after ageing.
        let range = KRange::new(-0.5, 0.5).unwrap();
        let tau = |d: u64| (d * DAY) as f64 / YEAR_MS;
        let s1 = SviSlice::new(tau(30), 0.006, 0.02, -0.4, 0.0, 0.2, range).unwrap();
        let s2 = SviSlice::new(tau(60), 0.0061, 0.02, -0.4, 0.0, 0.2, range).unwrap();
        let surface = SviSurface::new(vec![s1, s2], ShortExpiryPolicy::Reject).unwrap();
        let entry = CalibratedSurface::new(
            surface,
            vec![30 * DAY, 60 * DAY],
            0,
            0,
            SurfaceStatus::Healthy,
        )
        .unwrap();
        let mut store = SurfaceStore::new();
        store.insert(entry).unwrap();
        let served = store.surface_at(9 * DAY, &policy(false)).unwrap();
        assert!(served.diagnostics.calendar_gap.is_clean());
        assert!(served.diagnostics.calendar_slope.is_clean());
    }
}
