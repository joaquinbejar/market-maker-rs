//! Volatility surfaces as total implied variance.
//!
//! A surface answers one question: the total implied variance
//! `w(k, tau) = sigma_iv(k, tau)^2 * tau` and its derivatives at a log forward
//! moneyness `k = ln(K / F)` and a time to expiry `tau` in years. It knows
//! nothing about spot dynamics, discounting or payoffs: the option pricer
//! decides where to read the surface (sticky strike or sticky moneyness) and
//! resolves `tau = 0` (the payoff) before any surface is consulted, so a
//! surface is never asked for `tau <= 0`.
//!
//! # Numerics
//!
//! Model arithmetic in this module is `f64`, an explicit exception to the
//! crate's `Decimal` rule that covers option pricing, Greeks, implied
//! volatility and surface calibration under `options`. Monetary inputs and
//! outputs stay `Decimal` and cross into `f64` through checked conversions;
//! every value a surface returns is checked to be finite (and `w > 0`), and a
//! failure is a typed error, never a saturated or zero value. Results are
//! deterministic on a given platform; they are not promised to be bit
//! identical across architectures, so tests use declared tolerances.
//!
//! # The `w_tau` contract
//!
//! [`TotalVariance::w_tau`] is `dw / dtau` at fixed `k`. A surface that is
//! piecewise in `tau` (slices joined by interpolation) must have at least two
//! slices for it to be meaningful, and returns, on a node, the slope of the
//! segment towards shorter `tau` (the direction calendar time moves), except
//! on the first node, where it returns the slope of the only segment there
//! is. A surface that is smooth in `tau` (such as [`FlatSurface`]) returns
//! its derivative.
//!
//! # No arbitrage checks here
//!
//! A [`TotalVariance`] is only required to be a valid number: `w > 0` and
//! finite derivatives. Static arbitrage is a property of the whole surface,
//! not of one point, and is diagnosed where surfaces are built (calibration
//! and its diagnostics): a negative `w_tau` (calendar arbitrage) or a smile
//! that fails the butterfly condition is accepted here. A consumer must not
//! assume `w_tau >= 0`, for example to read a forward variance; the option
//! pricer uses `w_tau` only in theta, where a negative value is still
//! well defined.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};
use rust_decimal::prelude::ToPrimitive;

/// Calibration of raw SVI slices to observed total variances.
pub mod calibrate;
/// Static diagnostics of a surface on a declared grid.
pub mod diagnostics;
/// Implied total variance of cleaned quotes.
pub mod iv;
/// From a quote set to a calibrated surface and its quality report.
pub mod pipeline;
/// Cleaning of option quotes before calibration.
pub mod quotes;
/// Causal replay of quote snapshots.
pub mod replay;
/// Shock scenarios on a surface, each with its own diagnostics.
pub mod shocks;
/// Causal availability of calibrated surfaces.
pub mod store;
/// Raw SVI slices and surfaces of slices.
pub mod svi;
pub mod upstream;
/// Out of sample validation of the SVI fit against a baseline.
pub mod validation;

pub use calibrate::{
    ActiveConstraint, FitConfig, FitFailure, Residual, SliceFit, SliceObservation, fit_slice,
};
pub use diagnostics::{
    Check, DiagnosticGrid, DiagnosticPoint, DiagnosticsConfig, SurfaceDiagnostics, diagnose,
};
pub use iv::{Extraction, IvConfig, IvFailure, IvObservation, extract, implied_total_variance};
pub use pipeline::{
    QualityReport, QuoteResidual, SliceReport, StabilityConfig, StabilityPoint,
    SurfaceCalibrationConfig, calibrate_surface,
};
pub use quotes::{
    CleanQuote, CleaningConfig, CleaningReport, OptionQuote, QuoteFlag, QuoteVerdict, Rates,
    Rejection, clean_quotes,
};
pub use replay::{ReplayReport, ReplayResidual, ReplayStep, Snapshot, replay};
pub use shocks::{ScenarioReport, ShockedSurface, SurfaceShock, run_scenarios, standard_scenarios};
pub use store::{
    CalibratedSurface, ServedSurface, SurfaceStatus, SurfaceStore, UsePolicy, YEAR_MS,
};
pub use svi::{KRange, Region, ShortExpiryPolicy, SviSlice, SviSurface};
pub use validation::{
    BaselineSlice, BucketComparison, BucketStats, ExpiryHoldout, HoldoutReport, HoldoutRule,
    WingRule, validate_holdout,
};

/// A volatility surface: total implied variance and its derivatives.
///
/// Implementations provide [`VolSurface::total_variance_at`]; callers use
/// [`VolSurface::total_variance`], which runs [`check_point`] before calling
/// it. That is a convention, not a guarantee Rust enforces: an
/// implementation could override the provided method (it should not), and
/// `total_variance_at` is public like every trait method, so code that calls
/// it directly must check the point itself. Consumers such as the option
/// pricer also validate their inputs at their own boundary.
/// Implementations must be thread safe (`Send + Sync`), so one surface can
/// be shared by the pricers of a whole chain.
pub trait VolSurface: Send + Sync {
    /// The surface's own evaluation. It assumes `(k, tau)` passed
    /// [`check_point`] (finite `k`, finite `tau > 0`):
    /// [`VolSurface::total_variance`] checks it before the call, and a direct
    /// caller must check it first.
    ///
    /// # Errors
    ///
    /// - [`MMError::InvalidMarketState`] when `(k, tau)` is outside the domain
    ///   the surface declares.
    /// - [`MMError::NumericalError`] when the result is not a valid
    ///   [`TotalVariance`].
    fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance>;

    /// The total implied variance at log forward moneyness `k` and time to
    /// expiry `tau` (years), with its derivatives, after checking the point.
    /// Implementations should not override it: it is where the point is
    /// checked.
    ///
    /// # Errors
    ///
    /// - [`MMError::InvalidMarketState`] when `k` is not finite or `tau` is not
    ///   finite and strictly positive ([`check_point`]).
    /// - The errors of [`VolSurface::total_variance_at`].
    fn total_variance(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        check_point(k, tau)?;
        self.total_variance_at(k, tau)
    }
}

/// Total implied variance `w` at one point, with `w_k = dw/dk`,
/// `w_kk = d2w/dk2` and `w_tau = dw/dtau` (at fixed `k`; see the module docs
/// for nodes).
///
/// The fields are private: a value can only be built through
/// [`TotalVariance::new`], which validates it, so a pricer never receives an
/// invalid one from any [`VolSurface`] implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TotalVariance {
    w: f64,
    w_k: f64,
    w_kk: f64,
    w_tau: f64,
}

impl TotalVariance {
    /// Builds a total variance after checking it.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when `w` is not finite and strictly
    /// positive, or when any of `w_k`, `w_kk`, `w_tau` is not finite.
    pub fn new(w: f64, w_k: f64, w_kk: f64, w_tau: f64) -> MMResult<Self> {
        if !(w.is_finite() && w > 0.0) {
            return Err(numerical(format!(
                "total variance must be finite and positive, got {w}"
            )));
        }
        for (name, value) in [("w_k", w_k), ("w_kk", w_kk), ("w_tau", w_tau)] {
            if !value.is_finite() {
                return Err(numerical(format!(
                    "total variance derivative {name} must be finite, got {value}"
                )));
            }
        }
        Ok(Self {
            w,
            w_k,
            w_kk,
            w_tau,
        })
    }

    /// Total implied variance, `sigma_iv^2 * tau`.
    #[inline]
    #[must_use]
    pub fn w(&self) -> f64 {
        self.w
    }

    /// `dw / dk` at fixed `tau`.
    #[inline]
    #[must_use]
    pub fn w_k(&self) -> f64 {
        self.w_k
    }

    /// `d2w / dk2` at fixed `tau`.
    #[inline]
    #[must_use]
    pub fn w_kk(&self) -> f64 {
        self.w_kk
    }

    /// `dw / dtau` at fixed `k` (see the module docs for nodes).
    #[inline]
    #[must_use]
    pub fn w_tau(&self) -> f64 {
        self.w_tau
    }
}

/// Checks a surface query point: `k` finite, `tau` finite and strictly
/// positive. [`VolSurface::total_variance`] calls it before the surface's
/// own evaluation; a direct caller of [`VolSurface::total_variance_at`]
/// must call it first.
///
/// # Errors
///
/// [`MMError::InvalidMarketState`] naming the offending value.
pub fn check_point(k: f64, tau: f64) -> MMResult<()> {
    if !k.is_finite() {
        return Err(invalid(format!("log moneyness k must be finite, got {k}")));
    }
    if !(tau.is_finite() && tau > 0.0) {
        return Err(invalid(format!(
            "time to expiry must be finite and positive for a surface, got {tau}"
        )));
    }
    Ok(())
}

/// A constant implied volatility: `w = sigma^2 * tau` for every `k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatSurface {
    sigma: Decimal,
    sigma_f64: f64,
}

impl FlatSurface {
    /// A flat surface at implied volatility `sigma` (annualised, e.g. `0.2`).
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] when `sigma` is not strictly
    /// positive or has no finite `f64` value.
    pub fn new(sigma: Decimal) -> MMResult<Self> {
        if sigma <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "flat surface volatility must be positive, got {sigma}"
            )));
        }
        let sigma_f64 = sigma
            .to_f64()
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!(
                    "flat surface volatility {sigma} has no positive finite f64 value"
                ))
            })?;
        Ok(Self { sigma, sigma_f64 })
    }

    /// The implied volatility of the surface.
    #[inline]
    #[must_use]
    pub fn sigma(&self) -> Decimal {
        self.sigma
    }
}

impl VolSurface for FlatSurface {
    /// `w = sigma^2 tau`, `w_k = w_kk = 0`, `w_tau = sigma^2`.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when `sigma^2 tau` overflows or underflows
    /// to zero.
    fn total_variance_at(&self, _k: f64, tau: f64) -> MMResult<TotalVariance> {
        let var = self.sigma_f64 * self.sigma_f64;
        TotalVariance::new(var * tau, 0.0, 0.0, var)
    }
}

#[cold]
#[inline(never)]
fn invalid(msg: String) -> MMError {
    MMError::InvalidMarketState(msg)
}

#[cold]
#[inline(never)]
fn numerical(msg: String) -> MMError {
    MMError::NumericalError(msg)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::sync::Arc;

    #[test]
    fn test_total_variance_valid_values_round_trip() {
        let tv = TotalVariance::new(0.04, -0.1, 0.5, 0.2).unwrap();
        assert_eq!(
            (tv.w(), tv.w_k(), tv.w_kk(), tv.w_tau()),
            (0.04, -0.1, 0.5, 0.2)
        );
    }

    #[test]
    fn test_total_variance_non_positive_or_non_finite_w_is_error() {
        for w in [0.0, -1e-300, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    TotalVariance::new(w, 0.0, 0.0, 0.0),
                    Err(MMError::NumericalError(_))
                ),
                "w = {w}"
            );
        }
    }

    #[test]
    fn test_total_variance_non_finite_derivative_is_error() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(TotalVariance::new(0.04, bad, 0.0, 0.0).is_err());
            assert!(TotalVariance::new(0.04, 0.0, bad, 0.0).is_err());
            assert!(TotalVariance::new(0.04, 0.0, 0.0, bad).is_err());
        }
    }

    #[test]
    fn test_check_point_invalid_inputs_are_errors() {
        assert!(check_point(0.0, 1.0).is_ok());
        for (k, tau) in [
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (0.0, 0.0),
            (0.0, -1.0),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
        ] {
            assert!(
                matches!(check_point(k, tau), Err(MMError::InvalidMarketState(_))),
                "k = {k}, tau = {tau}"
            );
        }
    }

    #[test]
    fn test_flat_surface_values_are_analytic() {
        let s = FlatSurface::new(dec!(0.2)).unwrap();
        let tv = s.total_variance(0.3, 0.5).unwrap();
        assert!((tv.w() - 0.02).abs() <= 1e-17);
        assert_eq!((tv.w_k(), tv.w_kk()), (0.0, 0.0));
        assert!((tv.w_tau() - 0.04).abs() <= 1e-17);
        assert_eq!(s.sigma(), dec!(0.2));
    }

    #[test]
    fn test_flat_surface_non_positive_sigma_is_error() {
        for sigma in [dec!(0), dec!(-0.2)] {
            assert!(matches!(
                FlatSurface::new(sigma),
                Err(MMError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn test_flat_surface_invalid_point_is_error() {
        let s = FlatSurface::new(dec!(0.2)).unwrap();
        assert!(matches!(
            s.total_variance(0.0, 0.0),
            Err(MMError::InvalidMarketState(_))
        ));
        assert!(matches!(
            s.total_variance(f64::NAN, 0.1),
            Err(MMError::InvalidMarketState(_))
        ));
    }

    #[test]
    fn test_flat_surface_underflow_to_zero_is_error_not_zero() {
        // sigma^2 tau = 1e-20 * 1e-306 is below the smallest subnormal.
        let s = FlatSurface::new(dec!(0.0000000001)).unwrap();
        assert!(matches!(
            s.total_variance(0.0, 1e-306),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_vol_surface_point_is_checked_before_the_implementation() {
        // An implementation that would accept anything: the provided method
        // (when not overridden) refuses an invalid point before calling it.
        struct Permissive;
        impl VolSurface for Permissive {
            fn total_variance_at(&self, _k: f64, _tau: f64) -> MMResult<TotalVariance> {
                TotalVariance::new(1.0, 0.0, 0.0, 0.0)
            }
        }
        assert!(Permissive.total_variance(0.0, 1.0).is_ok());
        for (k, tau) in [
            (f64::NAN, 1.0),
            (0.0, 0.0),
            (0.0, -1.0),
            (0.0, f64::INFINITY),
        ] {
            assert!(matches!(
                Permissive.total_variance(k, tau),
                Err(MMError::InvalidMarketState(_))
            ));
        }
    }

    #[test]
    fn test_total_variance_accepts_a_negative_w_tau() {
        // Calendar arbitrage is a surface diagnostic, not an invalid point.
        assert!(TotalVariance::new(0.04, 0.0, 0.0, -0.01).is_ok());
    }

    #[test]
    fn test_vol_surface_shared_trait_object_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>(_: &T) {}
        let s: Arc<dyn VolSurface> = Arc::new(FlatSurface::new(dec!(0.2)).unwrap());
        assert_send_sync(&*s);
        assert!(s.total_variance(0.0, 1.0).is_ok());
    }
}
