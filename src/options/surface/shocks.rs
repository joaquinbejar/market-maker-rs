//! Shock scenarios on a volatility surface.
//!
//! A shock moves the implied volatility `sigma(k, tau) = sqrt(w / tau)`:
//!
//! ```text
//! sigma'(k, tau) = sigma(k, tau) + parallel + skew * k + term * g(tau),
//! g(tau) = sqrt(tau_ref / tau)
//! ```
//!
//! `parallel` moves every point, `skew` tilts each smile around `k = 0`, and
//! `term` moves short expiries more than long ones (`g` is 1 at `tau_ref`).
//! `g` grows without bound as `tau` goes to 0, so the shock is finite only
//! where the base surface is defined; no uniform bound is claimed, and an
//! aged surface whose first expiry is close sees a larger term shock.
//!
//! The shocked surface is computed from the derivatives of `sigma'` in `k`
//! and `tau`, then `w' = sigma'^2 tau` and its derivatives. A point where
//! `sigma' <= 0` is an error.
//!
//! A shocked surface does not inherit any diagnostic of its base: a skew or
//! term shock can open butterfly or calendar violations the base did not
//! have. [`run_scenarios`] therefore returns every scenario with its own
//! [`SurfaceDiagnostics`], and a scenario is not to be used as validated
//! without them.

use super::diagnostics::{DiagnosticGrid, DiagnosticsConfig, SurfaceDiagnostics, diagnose};
use super::{TotalVariance, VolSurface};
use crate::types::error::{MMError, MMResult};

/// A named shock of the implied volatility; see the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceShock {
    name: String,
    parallel: f64,
    skew: f64,
    term: f64,
    tau_ref: f64,
}

impl SurfaceShock {
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for an empty name, a value that is
    /// not finite, or `tau_ref <= 0`.
    pub fn new(name: &str, parallel: f64, skew: f64, term: f64, tau_ref: f64) -> MMResult<Self> {
        if name.trim().is_empty() {
            return Err(MMError::InvalidConfiguration(
                "a surface shock needs a name".to_string(),
            ));
        }
        for (field, v) in [
            ("parallel", parallel),
            ("skew", skew),
            ("term", term),
            ("tau_ref", tau_ref),
        ] {
            if !v.is_finite() {
                return Err(MMError::InvalidConfiguration(format!(
                    "shock {name}: {field} must be finite, got {v}"
                )));
            }
        }
        if tau_ref <= 0.0 {
            return Err(MMError::InvalidConfiguration(format!(
                "shock {name}: tau_ref must be positive, got {tau_ref}"
            )));
        }
        Ok(Self {
            name: name.to_string(),
            parallel,
            skew,
            term,
            tau_ref,
        })
    }

    /// The scenario name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Parallel move of the implied volatility, per 1.00 of volatility.
    #[must_use]
    pub fn parallel(&self) -> f64 {
        self.parallel
    }

    /// Move per unit of `k`.
    #[must_use]
    pub fn skew(&self) -> f64 {
        self.skew
    }

    /// Move at `tau_ref`, scaled by `sqrt(tau_ref / tau)`.
    #[must_use]
    pub fn term(&self) -> f64 {
        self.term
    }

    /// Reference expiry of the term shock, in years.
    #[must_use]
    pub fn tau_ref(&self) -> f64 {
        self.tau_ref
    }
}

/// `base` under `shock`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShockedSurface<S> {
    base: S,
    shock: SurfaceShock,
}

impl<S: VolSurface> ShockedSurface<S> {
    /// Wraps `base`.
    #[must_use]
    pub fn new(base: S, shock: SurfaceShock) -> Self {
        Self { base, shock }
    }

    /// The shock.
    #[must_use]
    pub fn shock(&self) -> &SurfaceShock {
        &self.shock
    }

    /// The base surface.
    #[must_use]
    pub fn base(&self) -> &S {
        &self.base
    }
}

impl<S: VolSurface> VolSurface for ShockedSurface<S> {
    fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        let base = self.base.total_variance(k, tau)?;
        let (w, w_k, w_kk, w_tau) = (base.w(), base.w_k(), base.w_kk(), base.w_tau());
        // sigma = sqrt(w / tau) and its derivatives at fixed tau or k.
        let sigma = (w / tau).sqrt();
        let sigma_k = w_k / (2.0 * tau * sigma);
        let sigma_kk = (w_kk / (2.0 * tau) - sigma_k * sigma_k) / sigma;
        let sigma_tau = (w_tau - sigma * sigma) / (2.0 * tau * sigma);
        let s = &self.shock;
        let g = (s.tau_ref / tau).sqrt();
        let g_tau = -g / (2.0 * tau);
        let shocked = sigma + s.parallel + s.skew * k + s.term * g;
        let shocked_k = sigma_k + s.skew;
        let shocked_tau = sigma_tau + s.term * g_tau;
        if !(shocked.is_finite() && shocked > 0.0) {
            return Err(MMError::NumericalError(format!(
                "shock {}: implied volatility {shocked} at k {k}, tau {tau} is not positive",
                s.name
            )));
        }
        TotalVariance::new(
            shocked * shocked * tau,
            2.0 * tau * shocked * shocked_k,
            2.0 * tau * (shocked_k * shocked_k + shocked * sigma_kk),
            shocked * shocked + 2.0 * tau * shocked * shocked_tau,
        )
    }
}

/// A small named set of scenarios: volatility up and down 5 points, a
/// steeper and a flatter skew, and the short end up 5 points at
/// `tau_ref = 30 / 365`.
///
/// # Errors
///
/// Never in practice; the values are constants checked by
/// [`SurfaceShock::new`].
pub fn standard_scenarios() -> MMResult<Vec<SurfaceShock>> {
    let tau_ref = 30.0 / 365.0;
    Ok(vec![
        SurfaceShock::new("vol-up-5", 0.05, 0.0, 0.0, tau_ref)?,
        SurfaceShock::new("vol-down-5", -0.05, 0.0, 0.0, tau_ref)?,
        SurfaceShock::new("skew-steeper", 0.0, -0.05, 0.0, tau_ref)?,
        SurfaceShock::new("skew-flatter", 0.0, 0.05, 0.0, tau_ref)?,
        SurfaceShock::new("short-end-up-5", 0.0, 0.0, 0.05, tau_ref)?,
    ])
}

/// One scenario and the diagnostics of its shocked surface.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioReport {
    /// The shock.
    pub shock: SurfaceShock,
    /// Diagnostics of the shocked surface on the grid.
    pub diagnostics: SurfaceDiagnostics,
}

/// Each shock applied to `base`, with the diagnostics of the shocked
/// surface. A point where the shocked volatility is not positive is an
/// evaluation failure in that scenario's report.
///
/// # Errors
///
/// As [`diagnose`].
pub fn run_scenarios<S: VolSurface + Clone>(
    base: &S,
    shocks: &[SurfaceShock],
    grid: &DiagnosticGrid,
    config: DiagnosticsConfig,
) -> MMResult<Vec<ScenarioReport>> {
    shocks
        .iter()
        .map(|shock| {
            let shocked = ShockedSurface::new(base.clone(), shock.clone());
            Ok(ScenarioReport {
                shock: shock.clone(),
                diagnostics: diagnose(&shocked, grid, config)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::surface::{FlatSurface, KRange, ShortExpiryPolicy, SviSlice, SviSurface};
    use rust_decimal_macros::dec;

    fn base() -> SviSurface {
        let range = KRange::new(-1.0, 1.0).unwrap();
        let s = |tau, a| SviSlice::new(tau, a, 0.1, -0.5, 0.0, 0.2, range).unwrap();
        SviSurface::new(
            vec![s(0.25, 0.01), s(0.5, 0.02), s(1.0, 0.04)],
            ShortExpiryPolicy::Reject,
        )
        .unwrap()
    }

    #[test]
    fn test_surface_shock_invalid_values_return_err() {
        assert!(SurfaceShock::new("", 0.0, 0.0, 0.0, 0.1).is_err());
        assert!(SurfaceShock::new("x", f64::NAN, 0.0, 0.0, 0.1).is_err());
        assert!(SurfaceShock::new("x", 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_shocked_surface_zero_shock_is_identity() {
        let zero = SurfaceShock::new("zero", 0.0, 0.0, 0.0, 0.1).unwrap();
        let shocked = ShockedSurface::new(base(), zero);
        for (k, tau) in [(-0.4, 0.3), (0.0, 0.5), (0.7, 0.9)] {
            let a = base().total_variance(k, tau).unwrap();
            let b = shocked.total_variance(k, tau).unwrap();
            assert!((a.w() - b.w()).abs() < 1e-15);
            assert!((a.w_k() - b.w_k()).abs() < 1e-13);
            assert!((a.w_kk() - b.w_kk()).abs() < 1e-12);
            assert!((a.w_tau() - b.w_tau()).abs() < 1e-13);
        }
    }

    #[test]
    fn test_shocked_surface_derivatives_match_finite_differences() {
        let shock = SurfaceShock::new("mixed", 0.02, -0.03, 0.01, 0.1).unwrap();
        let shocked = ShockedSurface::new(base(), shock);
        let (k, tau, h) = (0.15, 0.6, 1e-6);
        let tv = shocked.total_variance(k, tau).unwrap();
        let wk_p = shocked.total_variance(k + h, tau).unwrap();
        let wk_m = shocked.total_variance(k - h, tau).unwrap();
        let wt_p = shocked.total_variance(k, tau + h).unwrap();
        let wt_m = shocked.total_variance(k, tau - h).unwrap();
        assert!((tv.w_k() - (wk_p.w() - wk_m.w()) / (2.0 * h)).abs() < 1e-8);
        assert!((tv.w_kk() - (wk_p.w_k() - wk_m.w_k()) / (2.0 * h)).abs() < 1e-6);
        assert!((tv.w_tau() - (wt_p.w() - wt_m.w()) / (2.0 * h)).abs() < 1e-8);
    }

    #[test]
    fn test_shocked_surface_non_positive_vol_returns_err() {
        let crash = SurfaceShock::new("crash", -1.0, 0.0, 0.0, 0.1).unwrap();
        let shocked = ShockedSurface::new(FlatSurface::new(dec!(0.3)).unwrap(), crash);
        assert!(shocked.total_variance(0.0, 0.5).is_err());
    }

    #[test]
    fn test_run_scenarios_reports_diagnostics_per_scenario() {
        let grid = DiagnosticGrid::uniform_k(-1.0, 1.0, 41, vec![0.25, 0.5, 0.75, 1.0]).unwrap();
        let mut shocks = standard_scenarios().unwrap();
        // A violent skew tilt opens violations the base does not have.
        shocks.push(SurfaceShock::new("skew-violent", 0.0, -1.5, 0.0, 0.1).unwrap());
        let reports = run_scenarios(&base(), &shocks, &grid, DiagnosticsConfig::default()).unwrap();
        assert_eq!(reports.len(), shocks.len());
        assert!(
            diagnose(&base(), &grid, DiagnosticsConfig::default())
                .unwrap()
                .is_clean()
        );
        let violent = reports
            .iter()
            .find(|r| r.shock.name() == "skew-violent")
            .unwrap();
        assert!(!violent.diagnostics.is_clean());
    }
}
