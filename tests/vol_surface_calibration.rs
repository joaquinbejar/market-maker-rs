//! Calibration fixtures: dirty quotes leave the clean fit unchanged, the
//! holdout compares SVI with the baseline, and the replay is causal.

#![cfg(feature = "options")]

use market_maker_rs::Decimal;
use market_maker_rs::options::black_undiscounted;
use market_maker_rs::options::surface::{
    CleaningConfig, DiagnosticsConfig, FitConfig, HoldoutRule, IvConfig, KRange, OptionQuote,
    Rates, Rejection, ShortExpiryPolicy, Snapshot, SurfaceCalibrationConfig, SurfaceStatus,
    SviSlice, UsePolicy, VolSurface, WingRule, YEAR_MS, calibrate_surface, clean_quotes, extract,
    replay, validate_holdout,
};
use optionstratlib::OptionStyle;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal_macros::dec;

const DAY: u64 = 86_400_000;
const CUTOFF: u64 = 1_000 * DAY;
const STRIKES: [f64; 15] = [
    60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0, 140.0,
    150.0,
];

/// The truth's implied volatility at `k` for an expiry `days` away.
fn truth_vol2(days: u64, k: f64) -> f64 {
    let tau = (days * DAY) as f64 / YEAR_MS;
    let b = 0.12 * tau.sqrt();
    let slice = SviSlice::new(
        tau,
        0.25 * tau - b * 0.25,
        b,
        -0.4,
        0.0,
        0.25,
        KRange::new(-1.0, 1.0).unwrap(),
    )
    .unwrap();
    slice.eval(k).unwrap().0 / tau
}

fn quote(
    cutoff: u64,
    expiry_days: u64,
    truth_days: u64,
    strike: f64,
    style: OptionStyle,
) -> OptionQuote {
    let tau = (expiry_days * DAY) as f64 / YEAR_MS;
    let k = (strike / 100.0_f64).ln();
    let w = truth_vol2(truth_days, k) * tau;
    let price = black_undiscounted(100.0, strike, w, style).unwrap();
    let half = (price * 0.005).max(0.005);
    OptionQuote {
        strike: Decimal::from_f64(strike).unwrap(),
        expiry_at: cutoff + expiry_days * DAY,
        style,
        bid: Decimal::from_f64(price - half).unwrap().max(Decimal::ZERO),
        ask: Decimal::from_f64(price + half).unwrap(),
        observed_at: cutoff - 1000,
        received_at: cutoff - 900,
        underlying: dec!(100),
        underlying_observed_at: cutoff - 1000,
        underlying_received_at: cutoff - 900,
    }
}

/// Both sides at every strike, expiries `days` from `cutoff`, priced at the
/// truth of `truth_days` (constant implied volatility per k across days).
fn snapshot_quotes(
    cutoff: u64,
    days: &[u64],
    truth_days: &[u64],
    strikes: &[f64],
) -> Vec<OptionQuote> {
    let mut out = Vec::new();
    for (&d, &td) in days.iter().zip(truth_days) {
        for &s in strikes {
            for style in [OptionStyle::Call, OptionStyle::Put] {
                out.push(quote(cutoff, d, td, s, style));
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
        stability: None,
    }
}

fn rates() -> Rates {
    Rates {
        rate: Decimal::ZERO,
        carry: Decimal::ZERO,
    }
}

#[test]
fn test_dirty_quotes_are_rejected_with_reasons_and_leave_the_clean_fit_unchanged() {
    let clean = snapshot_quotes(CUTOFF, &[30, 90, 180], &[30, 90, 180], &STRIKES);
    let mut dirty = clean.clone();
    // Extra strikes that must not enter: crossed, stale, too wide.
    let mut crossed = quote(CUTOFF, 90, 90, 97.5, OptionStyle::Call);
    std::mem::swap(&mut crossed.bid, &mut crossed.ask);
    let mut stale = quote(CUTOFF, 90, 90, 102.5, OptionStyle::Call);
    stale.observed_at = CUTOFF - 3_600_000;
    stale.underlying_observed_at = stale.observed_at;
    let mut wide = quote(CUTOFF, 30, 30, 107.5, OptionStyle::Call);
    wide.bid = dec!(0.01);
    // (ask - bid) / mid = 1.998, above this test's 1.9 (a relative spread
    // cannot exceed 2: that is a zero bid).
    wide.ask = dec!(20);
    // An older copy of a clean quote at another price: superseded.
    let mut older = clean[10].clone();
    older.observed_at -= 500;
    older.underlying_observed_at -= 500;
    older.bid += dec!(0.5);
    older.ask += dec!(0.5);
    // A pair breaking parity at an extra strike.
    let call = quote(CUTOFF, 180, 180, 112.5, OptionStyle::Call);
    let mut put = quote(CUTOFF, 180, 180, 112.5, OptionStyle::Put);
    put.bid += dec!(3);
    put.ask += dec!(3);
    let extra = [crossed, stale, wide, older, call, put];
    let first_extra = dirty.len();
    dirty.extend(extra);

    let mut config = config();
    config.cleaning.max_relative_spread = dec!(1.9);
    let (clean_entry, _) =
        calibrate_surface(&clean, CUTOFF, CUTOFF, rates(), None, &config).unwrap();
    let (dirty_entry, report) =
        calibrate_surface(&dirty, CUTOFF, CUTOFF, rates(), None, &config).unwrap();
    assert_eq!(clean_entry.surface(), dirty_entry.surface());
    let reason = |i: usize| report.cleaning.verdicts[i].rejection.clone();
    assert_eq!(reason(first_extra), Some(Rejection::Crossed));
    assert!(matches!(
        reason(first_extra + 1),
        Some(Rejection::Stale { .. })
    ));
    assert!(matches!(
        reason(first_extra + 2),
        Some(Rejection::TooWide { .. })
    ));
    assert_eq!(reason(first_extra + 3), Some(Rejection::Superseded));
    assert!(matches!(
        reason(first_extra + 4),
        Some(Rejection::ParityInconsistent { .. })
    ));
    assert!(matches!(
        reason(first_extra + 5),
        Some(Rejection::ParityInconsistent { .. })
    ));
}

#[test]
fn test_holdout_compares_svi_with_the_baseline_by_bucket() {
    // Strikes that make sense for a 7 day expiry: within a few of its
    // standard deviations; far wings there have zero bids and mids that are
    // half the ask, which say nothing about the volatility.
    let near: Vec<f64> = (0..15).map(|i| 90.0 + 1.5 * f64::from(i)).collect();
    let mut quotes = snapshot_quotes(CUTOFF, &[7], &[7], &near);
    quotes.extend(snapshot_quotes(
        CUTOFF,
        &[30, 90, 180],
        &[30, 90, 180],
        &STRIKES,
    ));
    // A sparse expiry: no holdout there.
    quotes.extend(snapshot_quotes(
        CUTOFF,
        &[365],
        &[365],
        &[80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
    ));
    let cleaning = clean_quotes(&quotes, CUTOFF, rates(), &config().cleaning).unwrap();
    let observed = extract(&cleaning, &IvConfig::default()).unwrap();
    let report = validate_holdout(
        &observed.observations,
        &FitConfig::default(),
        HoldoutRule::default(),
        WingRule::default(),
    )
    .unwrap();
    assert_eq!(report.not_applicable, vec![(CUTOFF + 365 * DAY, 6)]);
    assert!(report.fit_failures.is_empty(), "{:?}", report.fit_failures);
    assert!(report.baseline_failures.is_empty());
    assert_eq!(report.wings_unassigned, 0);
    assert_eq!(report.shortest_expiry, Some(CUTOFF + 7 * DAY));
    assert_eq!(report.expiries.len(), 4);
    assert!(
        report
            .expiries
            .iter()
            .all(|e| e.quotes == 15 && e.held == 7)
    );
    // 7 of 15 held out on each of the 4 dense expiries.
    assert_eq!(report.all.svi.points, 28);
    assert_eq!(report.all.svi.missing, 0);
    assert!(report.all.svi.rmse_vol < 2e-3, "{:?}", report.all);
    assert!(report.wings.svi.points > 0 && report.shortest.svi.points == 7);
    // Every bucket reports both models on the same points.
    for b in [report.all, report.wings, report.wings_abs, report.shortest] {
        assert_eq!(b.svi.points, b.baseline.points);
    }
}

#[test]
fn test_replay_scores_the_frozen_model_and_is_causal() {
    let days = [30, 90, 180];
    let snaps: Vec<Snapshot> = (0..3_u64)
        .map(|i| {
            let cutoff = CUTOFF + i * DAY;
            let left: Vec<u64> = days.iter().map(|d| d - i).collect();
            Snapshot {
                cutoff,
                // The calibration finishes late: 12 hours after the cutoff.
                available_at: cutoff + DAY / 2,
                quotes: snapshot_quotes(cutoff, &left, &days, &STRIKES),
            }
        })
        .collect();
    let policy = UsePolicy {
        allow_degraded: false,
        max_age_ms: 5 * DAY,
        k_grid: config().k_grid,
        diagnostics: DiagnosticsConfig::default(),
    };
    let report = replay(&snaps, rates(), &config(), &policy, &IvConfig::default()).unwrap();
    assert_eq!(report.steps.len(), 2);
    for (i, step) in report.steps.iter().enumerate() {
        let step = step.as_ref().unwrap();
        // The model frozen at t_i is snapshot i's calibration, scored on
        // snapshot i + 1, one day older.
        assert_eq!(step.model_cutoff, snaps[i].cutoff);
        assert_eq!(step.label_cutoff, snaps[i + 1].cutoff);
        assert_eq!(step.age_at_labels_ms, DAY);
        assert_eq!(step.model_status, SurfaceStatus::Healthy);
        assert!(step.missing.is_empty(), "{:?}", step.missing);
        // Constant vol per k: the aged model predicts the next day well.
        assert!(
            step.residuals
                .iter()
                .all(|r| r.vol_residual.is_some_and(|v| v.abs() < 5e-3))
        );
    }
    // Changing the last snapshot changes nothing before it.
    let mut changed = snaps.clone();
    if let Some(last) = changed.last_mut() {
        for q in &mut last.quotes {
            q.bid *= dec!(1.1);
            q.ask *= dec!(1.1);
        }
    }
    let again = replay(&changed, rates(), &config(), &policy, &IvConfig::default()).unwrap();
    assert_eq!(again.calibrations[0], report.calibrations[0]);
    assert_eq!(again.calibrations[1], report.calibrations[1]);
    let (a, b) = (
        again.steps[0].as_ref().unwrap(),
        report.steps[0].as_ref().unwrap(),
    );
    assert_eq!(a, b);
}

#[test]
fn test_replay_refuses_to_score_labels_observed_before_the_freeze() {
    // Daily cutoffs, each calibration available a day and a half later:
    // the model frozen at t_i exists only after the next snapshot's data.
    let days = [30, 90, 180];
    let snaps: Vec<Snapshot> = (0..3_u64)
        .map(|i| {
            let cutoff = CUTOFF + i * DAY;
            let left: Vec<u64> = days.iter().map(|d| d - i).collect();
            Snapshot {
                cutoff,
                available_at: cutoff + DAY + DAY / 2,
                quotes: snapshot_quotes(cutoff, &left, &days, &STRIKES),
            }
        })
        .collect();
    let policy = UsePolicy {
        allow_degraded: false,
        max_age_ms: 5 * DAY,
        k_grid: config().k_grid,
        diagnostics: DiagnosticsConfig::default(),
    };
    let report = replay(&snaps, rates(), &config(), &policy, &IvConfig::default()).unwrap();
    assert_eq!(report.steps.len(), 2);
    for step in &report.steps {
        let error = step.as_ref().unwrap_err();
        assert!(error.contains("not after the freeze"), "{error}");
    }
}

#[test]
fn test_holdout_keeps_the_targets_of_a_failed_fit() {
    // Three expiries of 15 out of the money quotes: 21 held out. The
    // shortest expiry's SVI fit is forced to fail; its 7 targets stay in
    // every denominator, missing for SVI and scored by the baseline.
    let quotes = snapshot_quotes(CUTOFF, &[30, 90, 180], &[30, 90, 180], &STRIKES);
    let cleaning = clean_quotes(&quotes, CUTOFF, rates(), &config().cleaning).unwrap();
    let mut observed = extract(&cleaning, &IvConfig::default()).unwrap();
    let first = CUTOFF + 30 * DAY;
    for (i, o) in observed
        .observations
        .iter_mut()
        .filter(|o| o.quote.quote.expiry_at == first)
        .enumerate()
    {
        // A zigzag no SVI slice follows within max_rmse_vol.
        o.w_mid *= if i % 4 < 2 { 2.5 } else { 0.4 };
    }
    let report = validate_holdout(
        &observed.observations,
        &FitConfig::default(),
        HoldoutRule::default(),
        WingRule::default(),
    )
    .unwrap();
    assert_eq!(report.fit_failures.len(), 1, "{:?}", report.fit_failures);
    assert_eq!(report.fit_failures[0].0, first);
    assert!(report.baseline_failures.is_empty());
    assert_eq!(report.all.svi.points, 21);
    assert_eq!(report.all.svi.missing, 7);
    assert_eq!(report.all.baseline.points, 21);
    assert_eq!(report.all.baseline.missing, 0);
    // The shortest expiry is the one that failed, not the next one.
    assert_eq!(report.shortest_expiry, Some(first));
    assert_eq!(report.shortest.svi.points, 7);
    assert_eq!(report.shortest.svi.missing, 7);
    assert_eq!(report.shortest.baseline.missing, 0);
    // No trained w_atm there: its wing membership is undecided.
    assert_eq!(report.wings_unassigned, 7);
    let failed = report
        .expiries
        .iter()
        .find(|e| e.expiry_at == first)
        .unwrap();
    assert_eq!((failed.held, failed.comparison.svi.missing), (7, 7));
}

#[test]
fn test_locked_and_near_locked_quotes_cannot_dominate_their_slice() {
    let clean = snapshot_quotes(CUTOFF, &[30, 90, 180], &[30, 90, 180], &STRIKES);
    // A locked quote at an extra strike: rejected, the fit unchanged.
    let mut locked = quote(CUTOFF, 90, 90, 102.5, OptionStyle::Call);
    locked.bid = locked.ask;
    let mut with_locked = clean.clone();
    with_locked.push(locked);
    let (clean_entry, _) =
        calibrate_surface(&clean, CUTOFF, CUTOFF, rates(), None, &config()).unwrap();
    let (locked_entry, report) =
        calibrate_surface(&with_locked, CUTOFF, CUTOFF, rates(), None, &config()).unwrap();
    assert_eq!(
        report.cleaning.verdicts[clean.len()].rejection,
        Some(Rejection::Locked)
    );
    assert_eq!(clean_entry.surface(), locked_entry.surface());

    // A near locked quote (a band of 1e-9 in price), its mid one full
    // spread above the truth: its weight is capped at max_weight_ratio
    // times the slice's median and reported, and it moves the slice less
    // than with no cap.
    let mut tight = quote(CUTOFF, 90, 90, 102.5, OptionStyle::Call);
    let mid = (tight.bid + tight.ask) / dec!(2) + (tight.ask - tight.bid);
    tight.bid = mid - dec!(0.0000000005);
    tight.ask = mid + dec!(0.0000000005);
    let mut with_tight = clean.clone();
    with_tight.push(tight);
    let (capped_entry, report) =
        calibrate_surface(&with_tight, CUTOFF, CUTOFF, rates(), None, &config()).unwrap();
    let flagged: Vec<usize> = report
        .residuals
        .iter()
        .filter(|r| r.weight_capped)
        .map(|r| r.index)
        .collect();
    assert_eq!(flagged, vec![clean.len()]);
    let mut uncapped_config = config();
    uncapped_config.fit.max_weight_ratio = 1e300;
    let (uncapped_entry, _) =
        calibrate_surface(&with_tight, CUTOFF, CUTOFF, rates(), None, &uncapped_config).unwrap();
    let tau = (90 * DAY) as f64 / YEAR_MS;
    let shift = |e: &market_maker_rs::options::surface::CalibratedSurface| {
        (0..=30)
            .map(|i| {
                let k = -0.6 + 0.04 * f64::from(i);
                let v = |s: &market_maker_rs::options::surface::SviSurface| {
                    (s.total_variance(k, tau).unwrap().w() / tau).sqrt()
                };
                (v(e.surface().unwrap()) - v(clean_entry.surface().unwrap())).abs()
            })
            .fold(0.0_f64, f64::max)
    };
    let (capped, uncapped) = (shift(&capped_entry), shift(&uncapped_entry));
    assert!(capped < 0.85 * uncapped, "{capped} vs {uncapped}");
}
