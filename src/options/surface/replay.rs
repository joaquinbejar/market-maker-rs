//! Causal replay of a sequence of quote snapshots.
//!
//! Each snapshot is calibrated with [`calibrate_surface`] at its cutoff and
//! stored with the time its calibration became available; the previous
//! calibration given to the stability check is the latest one available at
//! the snapshot's cutoff. Then, for each snapshot `i` but the last:
//!
//! 1. at `t_i`, its calibration's `available_at`, the model is frozen as
//!    the calibration [`SurfaceStore::surface_at`] serves at `t_i` (which
//!    may be an earlier one, under the [`UsePolicy`]);
//! 2. that frozen model is aged to the next snapshot's cutoff with the
//!    store's roll-down;
//! 3. it is scored on the next snapshot's cleaned out of the money quotes,
//!    which are labels only: never an input, a selection or a tuning of the
//!    model scored on them. `surface_at` is never asked again at the label
//!    time, which could already hold a calibration of the labels.
//!
//! A step whose labels' cutoff is not after `t_i` (a calibration that
//! finishes after the next snapshot's data) is an error with the reason,
//! never a score: those labels were observed by the time the model existed,
//! so scoring them would not be a forecast.
//!
//! The replay is evidence about the surface and its causality; it makes no
//! claim about option PnL or execution.

use super::VolSurface;
use super::iv::{IvConfig, extract};
use super::pipeline::{QualityReport, SurfaceCalibrationConfig, calibrate_surface};
use super::quotes::{OptionQuote, Rates, clean_quotes};
use super::store::{SurfaceStatus, SurfaceStore, UsePolicy};
use crate::Decimal;
use crate::types::error::{MMError, MMResult};

/// One snapshot of quotes.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// Latest data in the snapshot, ms.
    pub cutoff: u64,
    /// When its calibration is available, ms; `>= cutoff`.
    pub available_at: u64,
    /// The quotes.
    pub quotes: Vec<OptionQuote>,
}

/// The residual of one label quote under the frozen model.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayResidual {
    /// Index of the quote in its snapshot.
    pub index: usize,
    /// Strike.
    pub strike: Decimal,
    /// Expiry, ms.
    pub expiry_at: u64,
    /// Log forward moneyness at the label's cutoff.
    pub k: f64,
    /// Observed minus model implied volatility; `None` where the model
    /// could not be evaluated (the reason is in `missing`).
    pub vol_residual: Option<f64>,
}

/// One step: a frozen model and its score on the next snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayStep {
    /// When the model was frozen, ms.
    pub frozen_at: u64,
    /// Cutoff of the calibration served then, ms.
    pub model_cutoff: u64,
    /// When that calibration became available, ms.
    pub model_available_at: u64,
    /// Whether a later calibration was passed over by the policy.
    pub fallback: bool,
    /// The stored status of the model's calibration.
    pub model_status: SurfaceStatus,
    /// Cutoff of the labels, ms.
    pub label_cutoff: u64,
    /// `label_cutoff - model_cutoff`, ms.
    pub age_at_labels_ms: u64,
    /// One residual per label quote.
    pub residuals: Vec<ReplayResidual>,
    /// Label quotes the model could not evaluate, with the reason.
    pub missing: Vec<(usize, String)>,
}

/// The replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// The quality report of each snapshot's calibration.
    pub calibrations: Vec<QualityReport>,
    /// One step per snapshot but the last; `Err` text when no model was
    /// usable at that time.
    pub steps: Vec<Result<ReplayStep, String>>,
}

/// Replays `snapshots`, by increasing cutoff and availability; see the
/// [module docs](self).
///
/// # Errors
///
/// [`MMError::InvalidTimestamp`] when the snapshots are not in strictly
/// increasing cutoff and availability order or carry future data, and the
/// errors of [`calibrate_surface`].
pub fn replay(
    snapshots: &[Snapshot],
    rates: Rates,
    config: &SurfaceCalibrationConfig,
    policy: &UsePolicy,
    iv: &IvConfig,
) -> MMResult<ReplayReport> {
    if snapshots
        .windows(2)
        .any(|p| matches!(p, [a, b] if b.cutoff <= a.cutoff || b.available_at <= a.available_at))
    {
        return Err(MMError::InvalidTimestamp(
            "snapshots must have strictly increasing cutoffs and availability".to_string(),
        ));
    }
    let mut store = SurfaceStore::new();
    let mut calibrations = Vec::with_capacity(snapshots.len());
    for snap in snapshots {
        let previous = store
            .entries()
            .iter()
            .rev()
            .find(|e| e.available_at() <= snap.cutoff && e.surface().is_some());
        let (entry, report) = calibrate_surface(
            &snap.quotes,
            snap.cutoff,
            snap.available_at,
            rates,
            previous,
            config,
        )?;
        store.insert(entry)?;
        calibrations.push(report);
    }
    let mut steps = Vec::new();
    for pair in snapshots.windows(2) {
        let [model_snap, label_snap] = pair else {
            continue;
        };
        let t = model_snap.available_at;
        // A forecast scores data after its freeze: labels observed by `t`
        // are not a forecast of the frozen model, however it is aged.
        if label_snap.cutoff <= t {
            steps.push(Err(format!(
                "at {t}: the labels' cutoff {} is not after the freeze",
                label_snap.cutoff
            )));
            continue;
        }
        let served = match store.surface_at(t, policy) {
            Ok(served) => served,
            Err(error) => {
                steps.push(Err(format!("at {t}: {error}")));
                continue;
            }
        };
        // Freeze the served calibration and age it to the labels' cutoff.
        let Some(entry) = store
            .entries()
            .iter()
            .find(|e| e.available_at() == served.available_at)
        else {
            steps.push(Err(format!("at {t}: served calibration not found")));
            continue;
        };
        let model = match entry.aged(label_snap.cutoff) {
            Ok(model) => model,
            Err(error) => {
                steps.push(Err(format!(
                    "at {t}: frozen model not usable at the labels: {error}"
                )));
                continue;
            }
        };
        let labels = clean_quotes(
            &label_snap.quotes,
            label_snap.cutoff,
            rates,
            &config.cleaning,
        )?;
        let observed = extract(&labels, iv)?;
        let mut residuals = Vec::with_capacity(observed.observations.len());
        let mut missing = Vec::new();
        for o in &observed.observations {
            let tau = o.quote.tau;
            let vol_residual = match model.total_variance(o.quote.k, tau) {
                Ok(tv) => Some((o.w_mid / tau).sqrt() - (tv.w() / tau).sqrt()),
                Err(error) => {
                    missing.push((o.quote.index, error.to_string()));
                    None
                }
            };
            residuals.push(ReplayResidual {
                index: o.quote.index,
                strike: o.quote.quote.strike,
                expiry_at: o.quote.quote.expiry_at,
                k: o.quote.k,
                vol_residual,
            });
        }
        for (index, which, failure) in observed.failures {
            missing.push((index, format!("{which}: {failure}")));
        }
        steps.push(Ok(ReplayStep {
            frozen_at: t,
            model_cutoff: entry.data_cutoff(),
            model_available_at: entry.available_at(),
            fallback: served.fallback,
            model_status: entry.status().clone(),
            label_cutoff: label_snap.cutoff,
            age_at_labels_ms: label_snap.cutoff.saturating_sub(entry.data_cutoff()),
            residuals,
            missing,
        }));
    }
    Ok(ReplayReport {
        calibrations,
        steps,
    })
}
