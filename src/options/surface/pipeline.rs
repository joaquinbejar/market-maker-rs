//! From a quote set to a calibrated surface and its quality report.
//!
//! [`calibrate_surface`] runs, in order: [`clean_quotes`], [`extract`] of the
//! implied total variances, one [`fit_slice`] per expiry, the
//! [`SviSurface`] of the fitted slices, and [`diagnose`] on a grid holding
//! every node and segment midpoint ([`SviSurface::node_grid`]). The
//! [`QualityReport`] keeps every step's outcome: the cleaning verdicts, the
//! implied variance failures, each slice's fit or failure with its active
//! constraints, the residuals by strike and expiry, the diagnosis and the
//! stability against the previous surface.
//!
//! The status follows from the report and nothing is smoothed to improve
//! it:
//!
//! - **Failed** when fewer than two slices could be fitted (no surface);
//! - **Degraded** when a slice failed, the diagnosis is not clean, a quote
//!   used the in the money side, or the surface moved more than the
//!   stability thresholds since the previous one;
//! - **Healthy** otherwise.
//!
//! A clean diagnosis describes the grid only; it is not a proof that the
//! surface has no static arbitrage.

use std::collections::BTreeMap;

use rust_decimal::prelude::ToPrimitive;

use super::VolSurface;
use super::calibrate::{
    ActiveConstraint, FitConfig, FitFailure, SliceFit, SliceObservation, fit_slice,
};
use super::diagnostics::{DiagnosticsConfig, SurfaceDiagnostics, diagnose};
use super::iv::{Extraction, IvConfig, IvFailure, extract};
use super::quotes::{CleaningConfig, CleaningReport, OptionQuote, QuoteFlag, Rates, clean_quotes};
use super::store::{CalibratedSurface, SurfaceStatus};
use super::svi::{ShortExpiryPolicy, SviSurface};
use crate::Decimal;
use crate::options::pricing::black_undiscounted;
use crate::types::error::{MMError, MMResult};

/// Thresholds of the stability check against the previous surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityConfig {
    /// Largest relative change of the at the money `w`, per expiry.
    pub max_atm_change: f64,
    /// Largest absolute change of the at the money skew `dw/dk`, per expiry.
    pub max_skew_change: f64,
}

/// Every setting of one calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceCalibrationConfig {
    /// Quote cleaning.
    pub cleaning: CleaningConfig,
    /// Implied variance extraction.
    pub iv: IvConfig,
    /// Slice fit.
    pub fit: FitConfig,
    /// Diagnosis tolerances.
    pub diagnostics: DiagnosticsConfig,
    /// `k` grid of the diagnosis.
    pub k_grid: Vec<f64>,
    /// Short expiry policy of the surface.
    pub short_expiry: ShortExpiryPolicy,
    /// Stability thresholds; `None` skips the check.
    pub stability: Option<StabilityConfig>,
}

/// The residual of one quote that entered the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteResidual {
    /// Index of the quote in the input.
    pub index: usize,
    /// Strike.
    pub strike: Decimal,
    /// Expiry, ms.
    pub expiry_at: u64,
    /// Log forward moneyness.
    pub k: f64,
    /// Observed implied volatility (from the mid).
    pub vol_observed: f64,
    /// Fitted implied volatility.
    pub vol_fitted: f64,
    /// Observed minus fitted volatility.
    pub vol_residual: f64,
    /// Half the bid/ask band in volatility: the observation's uncertainty.
    pub vol_half_band: f64,
    /// Observed mid minus the fitted price, discounted.
    pub price_residual: f64,
}

/// One expiry's fit.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceReport {
    /// Expiry, ms.
    pub expiry_at: u64,
    /// Observations offered to the fit.
    pub observations: usize,
    /// The fit, or why there is none.
    pub fit: Result<SliceFit, FitFailure>,
}

/// The change of one expiry since the previous surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityPoint {
    /// Years to expiry at this calibration.
    pub tau: f64,
    /// Relative change of the at the money `w`.
    pub atm_change: f64,
    /// Absolute change of the at the money skew.
    pub skew_change: f64,
    /// Above a threshold.
    pub unstable: bool,
}

/// Everything one calibration found.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Cleaning verdicts.
    pub cleaning: CleaningReport,
    /// Quotes without an implied variance.
    pub iv_failures: Vec<(usize, &'static str, IvFailure)>,
    /// One report per expiry in the fit set.
    pub slices: Vec<SliceReport>,
    /// Residuals of the fitted quotes, by expiry then `k`.
    pub residuals: Vec<QuoteResidual>,
    /// Diagnosis of the surface, when there is one.
    pub diagnostics: Option<SurfaceDiagnostics>,
    /// Stability against the previous surface, when checked.
    pub stability: Vec<StabilityPoint>,
    /// Why the status is not Healthy.
    pub reasons: Vec<String>,
}

fn group_by_expiry(extraction: &Extraction) -> BTreeMap<u64, Vec<usize>> {
    let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, o) in extraction.observations.iter().enumerate() {
        groups.entry(o.quote.quote.expiry_at).or_default().push(i);
    }
    groups
}

/// Calibrates a surface from `quotes` at `cutoff`, available at
/// `available_at`; see the [module docs](self).
///
/// # Errors
///
/// - [`MMError::InvalidTimestamp`] when the set carries data after
///   `cutoff` (no surface may be built from it).
/// - [`MMError::InvalidConfiguration`] for an invalid configuration.
/// - [`MMError::NumericalError`] when a derived value is not finite.
pub fn calibrate_surface(
    quotes: &[OptionQuote],
    cutoff: u64,
    available_at: u64,
    rates: Rates,
    previous: Option<&SviSurface>,
    config: &SurfaceCalibrationConfig,
) -> MMResult<(CalibratedSurface, QualityReport)> {
    config.fit.validate()?;
    let cleaning = clean_quotes(quotes, cutoff, rates, &config.cleaning)?;
    let extraction = extract(&cleaning, &config.iv)?;
    let mut reasons = Vec::new();
    if cleaning
        .verdicts
        .iter()
        .any(|v| v.used_in_fit && v.flags.contains(&QuoteFlag::UsedItmSide))
    {
        reasons.push("an in the money side was used".to_string());
    }

    let mut slices = Vec::new();
    let mut fitted = Vec::new();
    let mut expiries = Vec::new();
    let mut residuals = Vec::new();
    for (expiry_at, members) in group_by_expiry(&extraction) {
        let obs: Vec<_> = members
            .iter()
            .filter_map(|&i| extraction.observations.get(i))
            .collect();
        let tau = obs.first().map_or(0.0, |o| o.quote.tau);
        let points: MMResult<Vec<SliceObservation>> = obs
            .iter()
            .map(|o| SliceObservation::new(o.quote.k, o.w_mid, o.weight))
            .collect();
        let fit = fit_slice(tau, &points?, &config.fit);
        match &fit {
            Ok(f) => {
                for (o, r) in obs.iter().zip(f.residuals.iter()) {
                    let strike = o.quote.quote.strike.to_f64().unwrap_or(f64::NAN);
                    let fitted_price = black_undiscounted(
                        o.quote.forward,
                        strike,
                        r.w_fitted,
                        o.quote.quote.style,
                    )? * o.quote.discount;
                    let mid = o
                        .quote
                        .quote
                        .bid
                        .checked_add(o.quote.quote.ask)
                        .and_then(|s| s.checked_div(Decimal::TWO))
                        .and_then(|m| m.to_f64())
                        .ok_or_else(|| {
                            MMError::NumericalError("quote mid has no f64 value".to_string())
                        })?;
                    residuals.push(QuoteResidual {
                        index: o.quote.index,
                        strike: o.quote.quote.strike,
                        expiry_at,
                        k: o.quote.k,
                        vol_observed: (o.w_mid / tau).sqrt(),
                        vol_fitted: (r.w_fitted / tau).sqrt(),
                        vol_residual: r.vol_residual,
                        vol_half_band: 0.5 * ((o.w_ask / tau).sqrt() - (o.w_bid / tau).sqrt()),
                        price_residual: mid - fitted_price,
                    });
                }
                if f.active
                    .iter()
                    .any(|c| matches!(c, ActiveConstraint::LeeUpper | ActiveConstraint::LeeLower))
                {
                    reasons.push(format!("expiry {expiry_at}: fit on Lee's bound"));
                }
                fitted.push(f.slice);
                expiries.push(expiry_at);
            }
            Err(failure) => reasons.push(format!("expiry {expiry_at}: {failure}")),
        }
        slices.push(SliceReport {
            expiry_at,
            observations: obs.len(),
            fit,
        });
    }

    let mut report = QualityReport {
        cleaning,
        iv_failures: extraction.failures,
        slices,
        residuals,
        diagnostics: None,
        stability: Vec::new(),
        reasons,
    };
    let surface = match SviSurface::new(fitted, config.short_expiry) {
        Ok(surface) => surface,
        Err(error) => {
            report.reasons.push(format!("no surface: {error}"));
            let entry = CalibratedSurface::failed(cutoff, available_at, report.reasons.clone())?;
            return Ok((entry, report));
        }
    };
    let grid = surface.node_grid(config.k_grid.clone(), &[])?;
    let diagnostics = diagnose(&surface, &grid, config.diagnostics)?;
    if !diagnostics.is_clean() {
        report.reasons.push("diagnosis not clean".to_string());
    }
    report.diagnostics = Some(diagnostics);
    if let (Some(prev), Some(stability)) = (previous, config.stability) {
        for slice in surface.slices() {
            let tau = slice.tau();
            let (Ok(now), Ok(before)) = (
                surface.total_variance(0.0, tau),
                prev.total_variance(0.0, tau),
            ) else {
                continue;
            };
            let atm_change = (now.w() - before.w()).abs() / before.w();
            let skew_change = (now.w_k() - before.w_k()).abs();
            let unstable =
                atm_change > stability.max_atm_change || skew_change > stability.max_skew_change;
            if unstable {
                report
                    .reasons
                    .push(format!("tau {tau}: unstable since the previous surface"));
            }
            report.stability.push(StabilityPoint {
                tau,
                atm_change,
                skew_change,
                unstable,
            });
        }
    }
    let status = if report.reasons.is_empty() {
        SurfaceStatus::Healthy
    } else {
        SurfaceStatus::Degraded(report.reasons.clone())
    };
    let entry = CalibratedSurface::new(surface, expiries, cutoff, available_at, status)?;
    Ok((entry, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::surface::{KRange, SviSlice, YEAR_MS};
    use optionstratlib::OptionStyle;
    use rust_decimal::prelude::FromPrimitive;
    use rust_decimal_macros::dec;

    const DAY: u64 = 86_400_000;
    const CUTOFF: u64 = 1_000 * DAY;

    fn truth_slice(days: u64) -> SviSlice {
        let tau = (days * DAY) as f64 / YEAR_MS;
        let b = 0.12 * tau.sqrt();
        let atm = 0.25 * tau;
        SviSlice::new(
            tau,
            atm - b * 0.25,
            b,
            -0.4,
            0.0,
            0.25,
            KRange::new(-1.0, 1.0).unwrap(),
        )
        .unwrap()
    }

    /// Quotes priced on the truth with a symmetric spread of 1% of the price
    /// (at least 0.01), both sides at every strike.
    fn quotes(days: &[u64], strikes: &[f64]) -> Vec<OptionQuote> {
        quotes_scaled(days, strikes, 1.0)
    }

    /// As [`quotes`], with every implied volatility of the truth times
    /// `vol_scale`.
    fn quotes_scaled(days: &[u64], strikes: &[f64], vol_scale: f64) -> Vec<OptionQuote> {
        let mut out = Vec::new();
        for &d in days {
            let slice = truth_slice(d);
            for &strike in strikes {
                let k = (strike / 100.0_f64).ln();
                let w = slice.eval(k).unwrap().0 * vol_scale * vol_scale;
                for style in [OptionStyle::Call, OptionStyle::Put] {
                    let price = black_undiscounted(100.0, strike, w, style).unwrap();
                    let half = (price * 0.005).max(0.005);
                    out.push(OptionQuote {
                        strike: Decimal::from_f64(strike).unwrap(),
                        expiry_at: CUTOFF + d * DAY,
                        style,
                        bid: Decimal::from_f64(price - half).unwrap().max(Decimal::ZERO),
                        ask: Decimal::from_f64(price + half).unwrap(),
                        observed_at: CUTOFF - 1000,
                        received_at: CUTOFF - 900,
                        underlying: dec!(100),
                        underlying_observed_at: CUTOFF - 1000,
                        underlying_received_at: CUTOFF - 900,
                    });
                }
            }
        }
        out
    }

    fn config() -> SurfaceCalibrationConfig {
        SurfaceCalibrationConfig {
            cleaning: CleaningConfig {
                max_age_ms: 60_000,
                max_spot_lag_ms: 500,
                max_relative_spread: dec!(2),
                parity_tolerance: dec!(0.01),
            },
            iv: IvConfig::default(),
            fit: FitConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            k_grid: (0..=40).map(|i| -1.0 + 0.05 * f64::from(i)).collect(),
            short_expiry: ShortExpiryPolicy::Reject,
            stability: Some(StabilityConfig {
                max_atm_change: 0.2,
                max_skew_change: 0.05,
            }),
        }
    }

    fn rates() -> Rates {
        Rates {
            rate: Decimal::ZERO,
            carry: Decimal::ZERO,
        }
    }

    const STRIKES: [f64; 13] = [
        70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0, 140.0,
    ];

    #[test]
    fn test_calibrate_surface_recovers_a_known_surface() {
        let (entry, report) = calibrate_surface(
            &quotes(&[30, 90, 180], &STRIKES),
            CUTOFF,
            CUTOFF,
            rates(),
            None,
            &config(),
        )
        .unwrap();
        assert_eq!(
            entry.status(),
            &SurfaceStatus::Healthy,
            "{:?}",
            report.reasons
        );
        assert!(
            report
                .diagnostics
                .as_ref()
                .is_some_and(SurfaceDiagnostics::is_clean)
        );
        let surface = entry.surface().unwrap();
        for d in [30, 90, 180] {
            let truth = truth_slice(d);
            for k in [-0.3, -0.1, 0.0, 0.1, 0.3] {
                let tau = truth.tau();
                let got = (surface.total_variance(k, tau).unwrap().w() / tau).sqrt();
                let want = (truth.eval(k).unwrap().0 / tau).sqrt();
                assert!((got - want).abs() < 2e-3, "days {d} k {k}: {got} vs {want}");
            }
        }
        // One residual per strike and expiry: the out of the money side.
        assert_eq!(report.residuals.len(), 3 * STRIKES.len());
        assert!(report.residuals.iter().all(|r| r.vol_residual.abs() < 5e-3));
    }

    #[test]
    fn test_calibrate_surface_too_few_expiries_fails() {
        let (entry, report) = calibrate_surface(
            &quotes(&[30], &STRIKES),
            CUTOFF,
            CUTOFF,
            rates(),
            None,
            &config(),
        )
        .unwrap();
        assert!(matches!(entry.status(), SurfaceStatus::Failed(_)));
        assert!(entry.surface().is_none());
        assert!(!report.reasons.is_empty());
    }

    #[test]
    fn test_calibrate_surface_failed_slice_degrades() {
        // The 90 day expiry has only 3 strikes: too few to fit.
        let mut q = quotes(&[30, 180], &STRIKES);
        q.extend(quotes(&[90], &[95.0, 100.0, 105.0]));
        let (entry, report) =
            calibrate_surface(&q, CUTOFF, CUTOFF, rates(), None, &config()).unwrap();
        assert!(
            matches!(entry.status(), SurfaceStatus::Degraded(_)),
            "{:?}",
            entry.status()
        );
        assert!(
            report
                .slices
                .iter()
                .any(|s| matches!(s.fit, Err(FitFailure::TooFewObservations { .. })))
        );
        assert_eq!(entry.surface().unwrap().slices().len(), 2);
    }

    #[test]
    fn test_calibrate_surface_future_data_returns_err() {
        let mut q = quotes(&[30, 90], &STRIKES);
        if let Some(first) = q.first_mut() {
            first.observed_at = CUTOFF + 1;
        }
        assert!(matches!(
            calibrate_surface(&q, CUTOFF, CUTOFF, rates(), None, &config()),
            Err(MMError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn test_calibrate_surface_large_move_is_unstable() {
        let q = quotes(&[30, 90, 180], &STRIKES);
        let (first, _) = calibrate_surface(&q, CUTOFF, CUTOFF, rates(), None, &config()).unwrap();
        let previous = first.surface().unwrap().clone();
        // Every implied volatility 30% higher, parity intact.
        let bumped = quotes_scaled(&[30, 90, 180], &STRIKES, 1.3);
        let (second, report) =
            calibrate_surface(&bumped, CUTOFF, CUTOFF, rates(), Some(&previous), &config())
                .unwrap();
        assert!(report.stability.iter().any(|p| p.unstable));
        assert!(matches!(second.status(), SurfaceStatus::Degraded(_)));
    }
}
