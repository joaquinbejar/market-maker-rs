//! Static diagnostics of a surface on a declared grid of `(k, tau)`.
//!
//! The checks work on any [`VolSurface`], so the same report is produced
//! for a calibrated surface, an aged one and every shocked scenario:
//!
//! - **butterfly**: Gatheral and Jacquier's density condition
//!   `g(k) = (1 - k w_k / (2 w))^2 - (w_k^2 / 4)(1 / w + 1 / 4) + w_kk / 2 >= 0`
//!   at every grid point;
//! - **calendar**: `w(k, tau)` does not decrease in `tau`, both as
//!   `dw/dtau >= 0` at each point and as `w(k, tau_{j+1}) >= w(k, tau_j)`
//!   between consecutive grid expiries;
//! - **evaluation**: every grid point has a valid total variance.
//!
//! A clean report describes the grid only. It is not a proof that the
//! surface is free of static arbitrage everywhere, and a limited set of
//! checks passing is never presented as one.

use super::VolSurface;
use crate::types::error::{MMError, MMResult};

/// The points the checks run on: increasing `k` and increasing positive
/// `tau`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticGrid {
    k: Vec<f64>,
    tau: Vec<f64>,
}

impl DiagnosticGrid {
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] unless both axes are non-empty,
    /// finite and strictly increasing, and every `tau` is positive.
    pub fn new(k: Vec<f64>, tau: Vec<f64>) -> MMResult<Self> {
        let increasing = |v: &[f64]| {
            !v.is_empty()
                && v.iter().all(|x| x.is_finite())
                && v.windows(2).all(|p| matches!(p, [a, b] if b > a))
        };
        if !increasing(&k) {
            return Err(MMError::InvalidConfiguration(
                "diagnostic k grid must be non-empty, finite and strictly increasing".to_string(),
            ));
        }
        if !increasing(&tau) || tau.iter().any(|t| *t <= 0.0) {
            return Err(MMError::InvalidConfiguration(
                "diagnostic tau grid must be non-empty, positive, finite and strictly increasing"
                    .to_string(),
            ));
        }
        Ok(Self { k, tau })
    }

    /// `n` evenly spaced points on `[lo, hi]` in `k`, and the given `tau`.
    ///
    /// # Errors
    ///
    /// As [`DiagnosticGrid::new`], or for `n < 2`.
    pub fn uniform_k(lo: f64, hi: f64, n: usize, tau: Vec<f64>) -> MMResult<Self> {
        if n < 2 {
            return Err(MMError::InvalidConfiguration(format!(
                "a uniform k grid needs at least 2 points, got {n}"
            )));
        }
        let last = n.saturating_sub(1);
        let step = (hi - lo) / last as f64;
        let k = (0..n)
            .map(|i| if i == last { hi } else { lo + step * i as f64 })
            .collect();
        Self::new(k, tau)
    }

    /// The `k` axis.
    #[must_use]
    pub fn k(&self) -> &[f64] {
        &self.k
    }

    /// The `tau` axis.
    #[must_use]
    pub fn tau(&self) -> &[f64] {
        &self.tau
    }
}

/// Tolerances of the checks: a value counts as a violation only below
/// `-tolerance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiagnosticsConfig {
    /// Tolerance on `g(k)`.
    pub butterfly_tolerance: f64,
    /// Tolerance on `dw/dtau` and on the gaps between grid expiries.
    pub calendar_tolerance: f64,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            butterfly_tolerance: 1e-10,
            calendar_tolerance: 1e-12,
        }
    }
}

impl DiagnosticsConfig {
    fn validate(&self) -> MMResult<()> {
        for (name, v) in [
            ("butterfly_tolerance", self.butterfly_tolerance),
            ("calendar_tolerance", self.calendar_tolerance),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be finite and non-negative, got {v}"
                )));
            }
        }
        Ok(())
    }
}

/// One grid point and the checked value there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiagnosticPoint {
    /// Log forward moneyness.
    pub k: f64,
    /// Expiry in years (the shorter one for a gap between expiries).
    pub tau: f64,
    /// The checked value: `g`, `dw/dtau` or the gap.
    pub value: f64,
}

/// The result of one check over the grid.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Check {
    /// Points checked.
    pub points: usize,
    /// Points below `-tolerance`.
    pub violations: usize,
    /// The point with the smallest value.
    pub worst: Option<DiagnosticPoint>,
}

impl Check {
    fn record(&mut self, point: DiagnosticPoint, tolerance: f64) {
        self.points = self.points.saturating_add(1);
        if point.value < -tolerance {
            self.violations = self.violations.saturating_add(1);
        }
        if self.worst.is_none_or(|w| point.value < w.value) {
            self.worst = Some(point);
        }
    }

    /// No violation.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.violations == 0
    }
}

/// The diagnostics of one surface on one grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceDiagnostics {
    /// The grid the checks ran on.
    pub grid: DiagnosticGrid,
    /// The tolerances used.
    pub config: DiagnosticsConfig,
    /// `g(k) >= 0` at each point.
    pub butterfly: Check,
    /// `dw/dtau >= 0` at each point.
    pub calendar_slope: Check,
    /// `w(k, tau_{j+1}) - w(k, tau_j) >= 0` between grid expiries.
    pub calendar_gap: Check,
    /// Points where the surface gave no valid total variance, with the
    /// error, in grid order.
    pub evaluation_failures: Vec<(f64, f64, String)>,
}

impl SurfaceDiagnostics {
    /// No violation and no evaluation failure on the grid.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.butterfly.is_clean()
            && self.calendar_slope.is_clean()
            && self.calendar_gap.is_clean()
            && self.evaluation_failures.is_empty()
    }
}

/// Gatheral and Jacquier's `g(k)`.
fn butterfly_g(k: f64, w: f64, w_k: f64, w_kk: f64) -> f64 {
    let first = 1.0 - k * w_k / (2.0 * w);
    first * first - (w_k * w_k / 4.0) * (1.0 / w + 0.25) + w_kk / 2.0
}

/// Runs the checks of the [module docs](self) on `surface` over `grid`.
///
/// A point the surface cannot evaluate, or where `g` is not finite, is
/// recorded as a failure, not skipped; the gap checks next to it are not
/// computed.
///
/// The calendar checks run at the grid's expiries only. For a surface that
/// is piecewise in `tau`, use a grid holding every node (for an
/// [`SviSurface`](super::SviSurface), `node_grid`): the checks at the nodes
/// then cover every segment.
///
/// # Errors
///
/// [`MMError::InvalidConfiguration`] for invalid tolerances.
pub fn diagnose(
    surface: &impl VolSurface,
    grid: &DiagnosticGrid,
    config: DiagnosticsConfig,
) -> MMResult<SurfaceDiagnostics> {
    config.validate()?;
    let mut butterfly = Check::default();
    let mut calendar_slope = Check::default();
    let mut calendar_gap = Check::default();
    let mut evaluation_failures = Vec::new();
    for &k in &grid.k {
        let mut previous: Option<(f64, f64)> = None;
        for &tau in &grid.tau {
            match surface.total_variance(k, tau) {
                Ok(tv) => {
                    let g = butterfly_g(k, tv.w(), tv.w_k(), tv.w_kk());
                    if !g.is_finite() {
                        evaluation_failures.push((
                            k,
                            tau,
                            format!("butterfly g is not finite: {g}"),
                        ));
                        previous = None;
                        continue;
                    }
                    butterfly.record(
                        DiagnosticPoint { k, tau, value: g },
                        config.butterfly_tolerance,
                    );
                    calendar_slope.record(
                        DiagnosticPoint {
                            k,
                            tau,
                            value: tv.w_tau(),
                        },
                        config.calendar_tolerance,
                    );
                    if let Some((tau_prev, w_prev)) = previous {
                        calendar_gap.record(
                            DiagnosticPoint {
                                k,
                                tau: tau_prev,
                                value: tv.w() - w_prev,
                            },
                            config.calendar_tolerance,
                        );
                    }
                    previous = Some((tau, tv.w()));
                }
                Err(error) => {
                    evaluation_failures.push((k, tau, error.to_string()));
                    previous = None;
                }
            }
        }
    }
    Ok(SurfaceDiagnostics {
        grid: grid.clone(),
        config,
        butterfly,
        calendar_slope,
        calendar_gap,
        evaluation_failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::surface::{FlatSurface, KRange, ShortExpiryPolicy, SviSlice, SviSurface};
    use rust_decimal_macros::dec;

    fn grid(taus: Vec<f64>) -> DiagnosticGrid {
        DiagnosticGrid::uniform_k(-1.5, 1.5, 61, taus).unwrap()
    }

    #[test]
    fn test_diagnostic_grid_invalid_axes_return_err() {
        assert!(DiagnosticGrid::new(vec![], vec![0.5]).is_err());
        assert!(DiagnosticGrid::new(vec![0.0, 0.0], vec![0.5]).is_err());
        assert!(DiagnosticGrid::new(vec![0.0], vec![0.0]).is_err());
        assert!(DiagnosticGrid::new(vec![0.0], vec![0.5, 0.25]).is_err());
        assert!(DiagnosticGrid::new(vec![f64::NAN], vec![0.5]).is_err());
        assert!(DiagnosticGrid::uniform_k(-1.0, 1.0, 1, vec![0.5]).is_err());
    }

    #[test]
    fn test_diagnose_flat_surface_is_clean() {
        let report = diagnose(
            &FlatSurface::new(dec!(0.4)).unwrap(),
            &grid(vec![0.1, 0.5, 1.0]),
            DiagnosticsConfig::default(),
        )
        .unwrap();
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.butterfly.points, 183);
        assert_eq!(report.calendar_gap.points, 122);
    }

    #[test]
    fn test_diagnose_vogt_slice_flags_butterfly() {
        // Axel Vogt's raw SVI slice (Gatheral and Jacquier 2014, section 4):
        // in the domain, but with a negative density.
        let range = KRange::new(-1.5, 1.5).unwrap();
        let vogt =
            |tau| SviSlice::new(tau, -0.0410, 0.1331, 0.3060, 0.3586, 0.4153, range).unwrap();
        let surface =
            SviSurface::new(vec![vogt(1.0), vogt(2.0)], ShortExpiryPolicy::Reject).unwrap();
        let report = diagnose(&surface, &grid(vec![1.0]), DiagnosticsConfig::default()).unwrap();
        assert!(!report.butterfly.is_clean());
        assert!(report.butterfly.worst.is_some_and(|p| p.value < 0.0));
    }

    #[test]
    fn test_diagnose_decreasing_variance_flags_calendar() {
        let range = KRange::new(-1.0, 1.0).unwrap();
        let s = |tau, a| SviSlice::new(tau, a, 0.1, -0.5, 0.0, 0.2, range).unwrap();
        let surface =
            SviSurface::new(vec![s(0.5, 0.04), s(1.0, 0.02)], ShortExpiryPolicy::Reject).unwrap();
        let report = diagnose(
            &surface,
            &grid(vec![0.5, 0.75, 1.0]),
            DiagnosticsConfig::default(),
        )
        .unwrap();
        assert!(!report.calendar_slope.is_clean());
        assert!(!report.calendar_gap.is_clean());
        assert!(!report.is_clean());
    }

    #[test]
    fn test_diagnose_inner_decreasing_segment_needs_the_nodes() {
        // 0.2 -> 0.3 decreases; a grid without those nodes cannot see it.
        let range = KRange::new(-1.0, 1.0).unwrap();
        let s = |tau, a| SviSlice::new(tau, a, 0.1, -0.5, 0.0, 0.2, range).unwrap();
        let surface = SviSurface::new(
            vec![s(0.1, 0.01), s(0.2, 0.03), s(0.3, 0.02), s(0.4, 0.05)],
            ShortExpiryPolicy::Reject,
        )
        .unwrap();
        let k = vec![-0.5, 0.0, 0.5];
        let ends = DiagnosticGrid::new(k.clone(), vec![0.1, 0.4]).unwrap();
        assert!(!surface.covers_nodes(&ends));
        let blind = diagnose(&surface, &ends, DiagnosticsConfig::default()).unwrap();
        assert!(
            blind.is_clean(),
            "a grid without the nodes describes only itself"
        );
        let nodes = surface.node_grid(k, &[]).unwrap();
        let report = diagnose(&surface, &nodes, DiagnosticsConfig::default()).unwrap();
        assert!(!report.calendar_slope.is_clean());
        assert!(!report.calendar_gap.is_clean());
    }

    #[test]
    fn test_diagnose_points_outside_surface_are_recorded_as_failures() {
        let range = KRange::new(-1.0, 1.0).unwrap();
        let s = |tau, a| SviSlice::new(tau, a, 0.1, -0.5, 0.0, 0.2, range).unwrap();
        let surface =
            SviSurface::new(vec![s(0.5, 0.02), s(1.0, 0.04)], ShortExpiryPolicy::Reject).unwrap();
        let report = diagnose(
            &surface,
            &grid(vec![0.25, 0.5]),
            DiagnosticsConfig::default(),
        )
        .unwrap();
        assert_eq!(report.evaluation_failures.len(), 61);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_diagnose_negative_tolerance_returns_err() {
        let config = DiagnosticsConfig {
            butterfly_tolerance: -1.0,
            ..DiagnosticsConfig::default()
        };
        assert!(
            diagnose(
                &FlatSurface::new(dec!(0.4)).unwrap(),
                &grid(vec![0.5]),
                config
            )
            .is_err()
        );
    }
}
