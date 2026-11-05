//! Walk-forward calibration on synthetic processes of known parameters.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::dec;
use market_maker_rs::prelude::Decimal;
use market_maker_rs::strategy::walk_forward::*;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const DAY_MS: u64 = 86_400_000;
const YEAR_S: f64 = 31_536_000.0;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
    fn poisson(&mut self, mean: f64) -> u64 {
        let limit = (-mean).exp();
        let (mut n, mut p) = (0, self.unit());
        while p > limit {
            n += 1;
            p *= self.unit();
        }
        n
    }
}

/// Prices at `step_ms` with relative volatility `sigma` per √year.
fn prices(seed: u64, from: u64, to: u64, step_ms: u64, sigma: f64) -> Vec<PriceSample> {
    let mut rng = Rng(seed);
    let step = sigma * (step_ms as f64 / 1000.0 / YEAR_S).sqrt();
    let mut level = 100.0f64;
    (from..to)
        .step_by(step_ms as usize)
        .map(|timestamp_ms| {
            level *= (step * rng.normal()).exp();
            PriceSample {
                timestamp_ms,
                price: Decimal::from_f64(level).unwrap(),
            }
        })
        .collect()
}

/// Exposures of one minute at distances 0.1, 0.2, ... 0.5, Poisson fills.
fn exposures(
    seed: u64,
    from: u64,
    to: u64,
    a: f64,
    k: f64,
    season: impl Fn(u64) -> f64,
) -> Vec<QuoteExposure> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for t in (from..to).step_by(60_000) {
        for level in 1..=5u32 {
            let d = f64::from(level) / 10.0;
            let timestamp_ms = t + 60_000;
            if timestamp_ms >= to {
                continue;
            }
            out.push(QuoteExposure {
                timestamp_ms,
                distance: Decimal::from(level) / dec!(10),
                exposure_ms: 60_000,
                fills: rng.poisson(a * (-k * d).exp() * 60.0 * season(timestamp_ms)),
            });
        }
    }
    out
}

fn constant() -> ConstantParameters {
    ConstantParameters {
        volatility: dec!(1),
        intensity_scale_a: dec!(0.01),
        order_intensity_k: dec!(1),
    }
}

fn config() -> WalkForwardConfig {
    WalkForwardConfig::new().with_resolution_ms(60_000)
}

#[test]
fn test_calibrate_window_known_process_recovers_parameters() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let data = CalibrationData::new(&p, &e).unwrap();
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let record = calibrate_window(&data, &window, &config(), "synthetic").unwrap();
    let fit = record.parameters().expect("calibrated");

    let relative = (fit.volatility / fit.reference_price).to_f64().unwrap();
    assert!((relative - 0.5).abs() < 0.05, "volatility {relative}");
    let a = fit.intensity_scale_a.to_f64().unwrap();
    let k = fit.order_intensity_k.to_f64().unwrap();
    assert!((a - 2.0).abs() < 0.1, "a {a}");
    assert!((k - 3.0).abs() < 0.15, "k {k}");
}

#[test]
fn test_calibrate_window_units_match_the_quoting_models() {
    // Absolute price per √year: doubling every price doubles the volatility;
    // fills per second: the same fills over half the exposure doubles A.
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let doubled: Vec<_> = p
        .iter()
        .map(|s| PriceSample {
            timestamp_ms: s.timestamp_ms,
            price: s.price * dec!(2),
        })
        .collect();
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let halved: Vec<_> = e
        .iter()
        .map(|x| QuoteExposure {
            exposure_ms: x.exposure_ms / 2,
            ..*x
        })
        .collect();
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let one = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "a",
    )
    .unwrap();
    let two = calibrate_window(
        &CalibrationData::new(&doubled, &halved).unwrap(),
        &window,
        &config(),
        "a",
    )
    .unwrap();
    let (one, two) = (one.parameters().unwrap(), two.parameters().unwrap());
    let vol = (two.volatility / one.volatility).to_f64().unwrap();
    let a = (two.intensity_scale_a / one.intensity_scale_a)
        .to_f64()
        .unwrap();
    assert!((vol - 2.0).abs() < 1e-9, "{vol}");
    assert!((a - 2.0).abs() < 1e-9, "{a}");
    assert_eq!(
        one.order_intensity_k.round_dp(9),
        two.order_intensity_k.round_dp(9)
    );
}

#[test]
fn test_calibrate_window_changing_later_data_leaves_record_unchanged() {
    let train = TimeWindow::new(0, DAY_MS).unwrap();
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, 2 * DAY_MS, 2.0, 3.0, |_| 1.0);
    let before = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &train,
        &config(),
        "d",
    )
    .unwrap();

    let (mut p2, mut e2) = (p.clone(), e.clone());
    p2.iter_mut()
        .filter(|s| s.timestamp_ms >= DAY_MS)
        .for_each(|s| s.price *= dec!(3));
    e2.iter_mut()
        .filter(|x| x.timestamp_ms >= DAY_MS)
        .for_each(|x| x.fills += 1000);
    let after = calibrate_window(
        &CalibrationData::new(&p2, &e2).unwrap(),
        &train,
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(before, after);

    // A change inside the window does move the hash.
    let mut p3 = p.clone();
    p3[10].price *= dec!(1.001);
    let inside = calibrate_window(
        &CalibrationData::new(&p3, &e).unwrap(),
        &train,
        &config(),
        "d",
    )
    .unwrap();
    assert_ne!(before.dataset_hash, inside.dataset_hash);
}

#[test]
fn test_calibrate_window_missing_data_reports_each_failure() {
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let run = |p: &[PriceSample], e: &[QuoteExposure]| {
        calibrate_window(
            &CalibrationData::new(p, e).unwrap(),
            &window,
            &config(),
            "d",
        )
        .unwrap()
        .outcome
    };
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);

    assert!(matches!(
        run(&[], &e),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewPrices { have: 0, .. })
    ));
    let flat: Vec<_> = p
        .iter()
        .map(|s| PriceSample {
            price: dec!(100),
            ..*s
        })
        .collect();
    assert_eq!(
        run(&flat, &e),
        CalibrationOutcome::Insufficient(CalibrationFailure::FlatPrices)
    );
    assert!(matches!(
        run(&p, &[]),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooLittleExposure { have_ms: 0, .. })
    ));
    let none: Vec<_> = e.iter().map(|x| QuoteExposure { fills: 0, ..*x }).collect();
    assert!(matches!(
        run(&p, &none),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewFills { have: 0, .. })
    ));
    let one: Vec<_> = e
        .iter()
        .map(|x| QuoteExposure {
            distance: dec!(0.1),
            ..*x
        })
        .collect();
    assert!(matches!(
        run(&p, &one),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewDistanceLevels { have: 1, .. })
    ));
    let rising: Vec<_> = e
        .iter()
        .map(|x| QuoteExposure {
            distance: dec!(1) - x.distance,
            ..*x
        })
        .collect();
    assert_eq!(
        run(&p, &rising),
        CalibrationOutcome::Insufficient(CalibrationFailure::NonDecayingIntensity)
    );
}

#[test]
fn test_calibrate_window_seasonality_recovers_the_intraday_shape() {
    let season = |t: u64| if t % DAY_MS < DAY_MS / 2 { 2.0 } else { 0.5 };
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, 2 * DAY_MS, 2.0, 3.0, season);
    let window = TimeWindow::new(0, 2 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(2, 30);
    let record =
        calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    let ratio = (fit.seasonality[0] / fit.seasonality[1]).to_f64().unwrap();
    assert!((ratio - 4.0).abs() < 0.2, "ratio {ratio}");
    let morning = fit.rate_per_second(dec!(0.1), 1_000).unwrap();
    let evening = fit.rate_per_second(dec!(0.1), DAY_MS / 2 + 1_000).unwrap();
    assert!(morning > evening * dec!(3));
}

#[test]
fn test_calibration_data_invalid_samples_are_refused() {
    let good = PriceSample {
        timestamp_ms: 1,
        price: dec!(1),
    };
    assert!(
        CalibrationData::new(
            &[PriceSample {
                price: dec!(0),
                ..good
            }],
            &[]
        )
        .is_err()
    );
    assert!(
        CalibrationData::new(
            &[
                good,
                PriceSample {
                    timestamp_ms: 0,
                    ..good
                }
            ],
            &[]
        )
        .is_err()
    );
    let x = QuoteExposure {
        timestamp_ms: 1,
        distance: dec!(0),
        exposure_ms: 1,
        fills: 0,
    };
    assert!(
        CalibrationData::new(
            &[],
            &[QuoteExposure {
                exposure_ms: 0,
                ..x
            }]
        )
        .is_err()
    );
    assert!(
        CalibrationData::new(
            &[],
            &[QuoteExposure {
                distance: dec!(-1),
                ..x
            }]
        )
        .is_err()
    );
    assert!(
        CalibrationData::new(
            &[],
            &[
                x,
                QuoteExposure {
                    timestamp_ms: 0,
                    ..x
                }
            ]
        )
        .is_err()
    );
}

#[test]
fn test_chronological_splits_invalid_or_short_range_is_refused() {
    let spec = WindowSpec {
        train_ms: 10,
        test_ms: 5,
        step_ms: 5,
    };
    let splits = chronological_splits(0, 25, &spec).unwrap();
    assert_eq!(splits.len(), 3);
    assert_eq!(
        splits[2].test,
        TimeWindow {
            start_ms: 20,
            end_ms: 25
        }
    );
    assert!(chronological_splits(0, 14, &spec).is_err());
    assert!(chronological_splits(0, 100, &WindowSpec { step_ms: 4, ..spec }).is_err());
    assert!(
        chronological_splits(
            0,
            100,
            &WindowSpec {
                train_ms: 0,
                ..spec
            }
        )
        .is_err()
    );
    assert!(chronological_splits(u64::MAX - 1, u64::MAX, &spec).is_err());
}

fn regime_change_report(rule: UpdateRule, train_days: u64) -> WalkForwardReport {
    // Volatility and intensity change on day 4 of 7.
    let mut p = prices(7, 0, 4 * DAY_MS, 60_000, 0.3);
    p.extend(prices(9, 4 * DAY_MS, 7 * DAY_MS, 60_000, 1.2));
    let mut e = exposures(11, 0, 4 * DAY_MS, 1.0, 2.0, |_| 1.0);
    e.extend(exposures(13, 4 * DAY_MS, 7 * DAY_MS, 3.0, 5.0, |_| 1.0));
    let data = CalibrationData::new(&p, &e).unwrap();
    let spec = WindowSpec {
        train_ms: train_days * DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let splits = chronological_splits(0, 7 * DAY_MS, &spec).unwrap();
    walk_forward(
        &data,
        &splits,
        &config().with_update_rule(rule),
        "regime",
        &constant(),
    )
    .unwrap()
}

fn mean_intensity_error(report: &WalkForwardReport, last_folds: usize) -> (f64, f64) {
    let folds = &report.folds[report.folds.len() - last_folds..];
    let sum = |pick: &dyn Fn(&FoldReport) -> Decimal| {
        folds.iter().map(|f| pick(f).to_f64().unwrap()).sum::<f64>() / last_folds as f64
    };
    (
        sum(&|f| f.calibrated.unwrap().intensity.unwrap()),
        sum(&|f| f.constant.intensity.unwrap()),
    )
}

#[test]
fn test_walk_forward_rolling_beats_constant_and_frozen_after_regime_change() {
    let rolling = regime_change_report(UpdateRule::Rolling, 1);
    let frozen = regime_change_report(UpdateRule::Frozen, 1);
    assert_eq!(rolling.scored_folds(), 6);

    let (rolling_err, constant_err) = mean_intensity_error(&rolling, 2);
    let (frozen_err, _) = mean_intensity_error(&frozen, 2);
    assert!(rolling_err < constant_err, "{rolling_err} {constant_err}");
    assert!(rolling_err < frozen_err, "{rolling_err} {frozen_err}");

    // Frozen parameters never move; rolling ones react to the change.
    assert!(
        frozen
            .folds
            .iter()
            .skip(1)
            .all(|f| f.k_change == Some(dec!(0)))
    );
    assert!(
        frozen
            .folds
            .iter()
            .all(|f| f.record == frozen.folds[0].record)
    );
    let largest = rolling
        .folds
        .iter()
        .filter_map(|f| f.k_change)
        .max()
        .unwrap();
    assert!(largest > dec!(1), "largest change {largest}");
}

#[test]
fn test_walk_forward_window_sizes_report_sensitivity() {
    for train_days in [1, 2, 3] {
        let report = regime_change_report(UpdateRule::Rolling, train_days);
        let folds = report.folds.len();
        let (calibrated, constant) = mean_intensity_error(&report, folds);
        // Evidence for the PR: run with --nocapture.
        println!(
            "train_days={train_days} folds={folds} calibrated={calibrated:.6} constant={constant:.6}"
        );
        for f in &report.folds {
            println!(
                "  test_start_day={} k={} vol={} calibrated={:?} constant={:?}",
                f.split.test.start_ms / DAY_MS,
                f.record.parameters().unwrap().order_intensity_k.round_dp(3),
                f.record.parameters().unwrap().volatility.round_dp(3),
                f.calibrated.unwrap(),
                f.constant
            );
        }
        assert_eq!(report.scored_folds(), folds);
    }
}

#[test]
fn test_walk_forward_insufficient_fold_is_unscored_not_inherited() {
    // The second training window has no exposures.
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let mut e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    e.extend(exposures(11, 2 * DAY_MS, 3 * DAY_MS, 2.0, 3.0, |_| 1.0));
    let data = CalibrationData::new(&p, &e).unwrap();
    let spec = WindowSpec {
        train_ms: DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let splits = chronological_splits(0, 3 * DAY_MS, &spec).unwrap();
    let report = walk_forward(&data, &splits, &config(), "gap", &constant()).unwrap();
    assert!(report.folds[0].calibrated.is_some());
    assert!(report.folds[1].calibrated.is_none());
    assert!(report.folds[1].record.parameters().is_none());
    assert!(report.folds[1].k_change.is_none());
    assert_eq!(report.scored_folds(), 1);
}

#[test]
fn test_walk_forward_invalid_inputs_are_refused() {
    let data = CalibrationData::new(&[], &[]).unwrap();
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: 10,
        },
        test: TimeWindow {
            start_ms: 5,
            end_ms: 15,
        },
    };
    assert!(walk_forward(&data, &[], &config(), "d", &constant()).is_err());
    assert!(walk_forward(&data, &[split], &config(), "d", &constant()).is_err());
    let good = ChronologicalSplit {
        test: TimeWindow {
            start_ms: 10,
            end_ms: 15,
        },
        ..split
    };
    let bad = ConstantParameters {
        volatility: dec!(0),
        ..constant()
    };
    assert!(walk_forward(&data, &[good], &config(), "d", &bad).is_err());
    assert!(
        walk_forward(
            &data,
            &[good],
            &config().with_resolution_ms(0),
            "d",
            &constant()
        )
        .is_err()
    );
}

#[test]
fn test_calibrate_window_continuous_distances_with_zero_fills_recovers_k() {
    // Every exposure has its own distance, and most have no fills.
    let mut rng = Rng(21);
    let e: Vec<_> = (0..20_000u64)
        .map(|i| {
            let d = 0.05 + 0.5 * rng.unit();
            QuoteExposure {
                timestamp_ms: i * 4_000,
                distance: Decimal::from_f64(d).unwrap(),
                exposure_ms: 4_000,
                fills: rng.poisson(0.5 * (-3.0 * d).exp() * 4.0),
            }
        })
        .collect();
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let record = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    let fit = record.parameters().unwrap();
    let k = fit.order_intensity_k.to_f64().unwrap();
    let a = fit.intensity_scale_a.to_f64().unwrap();
    assert!((k - 3.0).abs() < 0.3, "k {k}");
    assert!((a - 0.5).abs() < 0.06, "a {a}");
}

#[test]
fn test_calibrate_window_resolution_and_ewma_keep_the_annual_unit() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let data = CalibrationData::new(&p, &e).unwrap();
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    for cfg in [
        config().with_resolution_ms(300_000),
        config().with_volatility_rule(VolatilityRule::Ewma {
            lambda: dec!(0.999),
        }),
    ] {
        let record = calibrate_window(&data, &window, &cfg, "d").unwrap();
        let fit = record.parameters().unwrap();
        let relative = (fit.volatility / fit.reference_price).to_f64().unwrap();
        assert!((relative - 0.5).abs() < 0.12, "volatility {relative}");
    }
}

#[test]
fn test_calibrate_window_partial_final_bucket_is_dropped() {
    let p = prices(7, 0, 3_600_000, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let cfg = config()
        .with_resolution_ms(3_600_000)
        .with_minima(3, 2, 1, 1);
    // One complete bucket and a partial one give one grid price.
    let window = TimeWindow::new(0, 5_400_000).unwrap();
    let record =
        calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
    assert_eq!(
        record.outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewPrices { have: 1, need: 3 })
    );
}

#[test]
fn test_calibrate_window_thin_or_unobserved_bucket_is_insufficient() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let morning = exposures(11, 0, DAY_MS / 2, 2.0, 3.0, |_| 1.0);
    let data = CalibrationData::new(&p, &morning).unwrap();
    for min_bucket_fills in [0, 30] {
        let cfg = config().with_seasonality(2, min_bucket_fills);
        assert_eq!(
            calibrate_window(&data, &window, &cfg, "d").unwrap().outcome,
            CalibrationOutcome::Insufficient(CalibrationFailure::UnobservedSeasonalBucket {
                bucket: 1
            })
        );
    }

    // Exposure without fills in the afternoon is evidence of a zero rate.
    let mut quiet = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    quiet
        .iter_mut()
        .filter(|e| e.timestamp_ms >= DAY_MS / 2)
        .for_each(|e| e.fills = 0);
    let data = CalibrationData::new(&p, &quiet).unwrap();
    let thin = calibrate_window(&data, &window, &config().with_seasonality(2, 30), "d").unwrap();
    assert_eq!(
        thin.outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::ThinSeasonalBucket {
            bucket: 1,
            have: 0,
            need: 30
        })
    );
    let closed = calibrate_window(&data, &window, &config().with_seasonality(2, 0), "d").unwrap();
    assert_eq!(closed.parameters().unwrap().seasonality[1], dec!(0));
}

/// Two buckets, with exposures inside the window, each with `(distance, seconds, fills)` cells.
fn two_bucket_outcome(cells: [[(Decimal, u64, u64); 2]; 2]) -> CalibrationOutcome {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let mut e = Vec::new();
    for (bucket, levels) in cells.iter().enumerate() {
        for (distance, seconds, fills) in levels {
            e.push(QuoteExposure {
                timestamp_ms: [1_000_000, 50_000_000][bucket],
                distance: *distance,
                exposure_ms: seconds * 1_000,
                fills: *fills,
            });
        }
    }
    e.sort_by_key(|x| x.timestamp_ms);
    let cfg = config().with_seasonality(2, 1).with_minima(30, 2, 1, 1);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d")
        .unwrap()
        .outcome
}

#[test]
fn test_calibrate_window_exposure_mix_across_buckets_recovers_k() {
    // A 2:1 fall per unit of distance in both buckets, with opposite exposure.
    let outcome = two_bucket_outcome([
        [(dec!(0), 100, 200), (dec!(1), 1, 1)],
        [(dec!(0), 1, 8), (dec!(1), 100, 400)],
    ]);
    let CalibrationOutcome::Calibrated(fit) = outcome else {
        panic!("not calibrated: {outcome:?}");
    };
    let k = fit.order_intensity_k.to_f64().unwrap();
    assert!((k - std::f64::consts::LN_2).abs() < 1e-9, "k {k}");
    // The afternoon's rate at distance zero is 8 against the morning's 2.
    let ratio = (fit.seasonality[1] / fit.seasonality[0]).to_f64().unwrap();
    assert!((ratio - 4.0).abs() < 1e-9, "ratio {ratio}");
    let a = fit.intensity_scale_a.to_f64().unwrap();
    let morning = a * fit.seasonality[0].to_f64().unwrap();
    assert!((morning - 2.0).abs() < 1e-9, "morning {morning}");
}

#[test]
fn test_calibrate_window_fills_only_at_nearest_distance_is_unbounded() {
    assert_eq!(
        two_bucket_outcome([
            [(dec!(0), 100, 50), (dec!(1), 100, 0)],
            [(dec!(1), 100, 50), (dec!(2), 100, 0)],
        ]),
        CalibrationOutcome::Insufficient(CalibrationFailure::UnboundedIntensity)
    );
}

#[test]
fn test_walk_forward_changing_test_data_leaves_records_unchanged() {
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, 3 * DAY_MS, 2.0, 3.0, |_| 1.0);
    let spec = WindowSpec {
        train_ms: DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let split = chronological_splits(0, 2 * DAY_MS, &spec).unwrap();
    let (mut p2, mut e2) = (p.clone(), e.clone());
    p2.iter_mut()
        .filter(|s| s.timestamp_ms >= DAY_MS)
        .for_each(|s| s.price *= dec!(3));
    e2.iter_mut()
        .filter(|x| x.timestamp_ms >= DAY_MS)
        .for_each(|x| x.fills += 1000);
    for rule in [UpdateRule::Rolling, UpdateRule::Frozen] {
        let cfg = config().with_update_rule(rule);
        let one = walk_forward(
            &CalibrationData::new(&p, &e).unwrap(),
            &split,
            &cfg,
            "d",
            &constant(),
        )
        .unwrap();
        let two = walk_forward(
            &CalibrationData::new(&p2, &e2).unwrap(),
            &split,
            &cfg,
            "d",
            &constant(),
        )
        .unwrap();
        assert_eq!(one.folds[0].record, two.folds[0].record);
        assert_ne!(one.folds[0].calibrated, two.folds[0].calibrated);
    }
}

#[test]
fn test_walk_forward_empty_test_window_is_not_scored() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &[split],
        &config(),
        "d",
        &constant(),
    )
    .unwrap();
    assert_eq!(report.scored_folds(), 0);
}

#[test]
fn test_walk_forward_unordered_splits_are_refused() {
    let fold = |train: u64| ChronologicalSplit {
        train: TimeWindow {
            start_ms: train,
            end_ms: train + 10,
        },
        test: TimeWindow {
            start_ms: train + 10,
            end_ms: train + 20,
        },
    };
    let data = CalibrationData::new(&[], &[]).unwrap();
    let cfg = config().with_update_rule(UpdateRule::Frozen);
    assert!(walk_forward(&data, &[fold(100), fold(0)], &cfg, "d", &constant()).is_err());
}

#[test]
fn test_calibrate_window_oversized_config_is_refused() {
    let data = CalibrationData::new(&[], &[]).unwrap();
    let window = TimeWindow::new(0, u64::MAX).unwrap();
    assert!(calibrate_window(&data, &window, &config().with_seasonality(1_441, 0), "d").is_err());
    assert!(calibrate_window(&data, &window, &config().with_resolution_ms(1), "d").is_err());
}

#[test]
fn test_calibrate_window_volatility_matches_hand_computation() {
    // Alternating log returns of +r and -r on a 365 day year.
    let r = 0.001f64;
    let p: Vec<_> = (0..101u64)
        .map(|i| PriceSample {
            timestamp_ms: i * 60_000,
            price: Decimal::from_f64(if i % 2 == 0 { 100.0 } else { 100.0 * r.exp() }).unwrap(),
        })
        .collect();
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, 101 * 60_000).unwrap();
    let cfg = config().with_minima(30, 2, 1, 1);
    let record =
        calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    assert_eq!(fit.reference_price, dec!(100));
    // 100 returns with mean 0: the sample variance is 100 r² / 99.
    let per_step = (100.0 * r * r / 99.0f64).sqrt();
    let expected = 100.0 * per_step * (31_536_000_000.0f64 / 60_000.0).sqrt();
    let got = fit.volatility.to_f64().unwrap();
    assert!((got / expected - 1.0).abs() < 1e-6, "{got} {expected}");
}

#[test]
fn test_calibrate_window_every_hashed_field_changes_the_hash() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let hash = |p: &[PriceSample], e: &[QuoteExposure]| {
        calibrate_window(
            &CalibrationData::new(p, e).unwrap(),
            &window,
            &config(),
            "d",
        )
        .unwrap()
        .dataset_hash
    };
    let base = hash(&p, &e);
    let mut q = p.clone();
    q[5].price += dec!(0.01);
    assert_ne!(hash(&q, &e), base, "price");
    let mut q = p.clone();
    q[5].timestamp_ms += 1;
    assert_ne!(hash(&q, &e), base, "price time");
    let edits: [fn(&mut QuoteExposure); 4] = [
        |x| x.timestamp_ms += 1,
        |x| x.distance += dec!(0.01),
        |x| x.exposure_ms += 1,
        |x| x.fills += 1,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut f = e.clone();
        edit(&mut f[12]);
        f.sort_by_key(|x| x.timestamp_ms);
        assert_ne!(hash(&p, &f), base, "exposure field {i}");
    }
    // The same numbers at another decimal scale are the same data.
    let mut q = p.clone();
    let scale = q[5].price.scale().min(27) + 1;
    q[5].price.rescale(scale);
    assert_eq!(hash(&q, &e), base);
}

#[test]
fn test_calibrate_window_silent_prices_are_a_gap() {
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let run = |p: &[PriceSample], cfg: &WalkForwardConfig| {
        calibrate_window(&CalibrationData::new(p, &e).unwrap(), &window, cfg, "d")
            .unwrap()
            .outcome
    };
    // Two samples in a day: the grid is full, but only two buckets were seen.
    let two = prices(7, 0, 120_000, 60_000, 0.5);
    assert_eq!(
        run(&two, &config()),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewPrices { have: 2, need: 30 })
    );
    // One hour of data, then silence: enough buckets, a trailing gap.
    let hour = prices(7, 0, 3_600_000, 60_000, 0.5);
    assert_eq!(
        run(&hour, &config()),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: DAY_MS - 3_540_000,
            limit_ms: 300_000
        })
    );
    // A gap in the middle, and one exactly at the limit, which passes.
    let mut middle = prices(7, 0, DAY_MS, 60_000, 0.5);
    middle.retain(|s| !(600_000..840_000).contains(&s.timestamp_ms));
    assert!(matches!(
        run(&middle, &config()),
        CalibrationOutcome::Calibrated(_)
    ));
    middle.retain(|s| s.timestamp_ms != 840_000);
    assert!(matches!(
        run(&middle, &config()),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: 360_000,
            ..
        })
    ));
    assert!(matches!(
        run(&middle, &config().with_max_gap_buckets(6)),
        CalibrationOutcome::Calibrated(_)
    ));
}

#[test]
fn test_walk_forward_score_ranks_the_true_model_on_continuous_distances() {
    // True A = 1 and k = 1, 1 s exposures, most without fills.
    let mut rng = Rng(31);
    let mut e = Vec::new();
    for i in 1..(2 * DAY_MS / 10_000) {
        let d = 3.0 * rng.unit();
        e.push(QuoteExposure {
            timestamp_ms: i * 10_000,
            distance: Decimal::from_f64(d).unwrap(),
            exposure_ms: 1_000,
            fills: rng.poisson((-d).exp()),
        });
    }
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let near_zero = ConstantParameters {
        volatility: dec!(1),
        intensity_scale_a: dec!(0.000000001),
        order_intensity_k: dec!(1),
    };
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &[split],
        &config(),
        "d",
        &near_zero,
    )
    .unwrap();
    let fold = &report.folds[0];
    assert!(fold.calibrated.unwrap().intensity < fold.constant.intensity);
}

#[test]
fn test_walk_forward_score_sees_the_seasonality() {
    let season = |t: u64| if t % DAY_MS < DAY_MS / 2 { 0.2 } else { 1.8 };
    let p = prices(7, 0, 4 * DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, 4 * DAY_MS, 2.0, 3.0, season);
    let data = CalibrationData::new(&p, &e).unwrap();
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: 2 * DAY_MS,
        },
        test: TimeWindow {
            start_ms: 2 * DAY_MS,
            end_ms: 4 * DAY_MS,
        },
    };
    let error = |cfg: WalkForwardConfig| {
        walk_forward(&data, &[split], &cfg, "d", &constant())
            .unwrap()
            .folds[0]
            .calibrated
            .unwrap()
            .intensity
            .unwrap()
    };
    assert!(error(config().with_seasonality(2, 30)) < error(config()));
}

#[test]
fn test_walk_forward_k_change_compares_across_an_insufficient_fold() {
    let p = prices(7, 0, 4 * DAY_MS, 60_000, 0.5);
    let mut e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    e.extend(exposures(11, 2 * DAY_MS, 3 * DAY_MS, 2.0, 5.0, |_| 1.0));
    let data = CalibrationData::new(&p, &e).unwrap();
    let spec = WindowSpec {
        train_ms: DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let splits = chronological_splits(0, 4 * DAY_MS, &spec).unwrap();
    let report = walk_forward(&data, &splits, &config(), "gap", &constant()).unwrap();
    assert!(report.folds[1].record.parameters().is_none());
    assert!(report.folds[1].k_change.is_none());
    let change = report.folds[2].k_change.expect("against the first fold");
    assert!((change.to_f64().unwrap() - 2.0).abs() < 0.3, "{change}");
}

/// Two levels of 60 s, one bucket, as `(distance, fills)`.
fn two_levels(cells: [(Decimal, u64); 2]) -> Vec<QuoteExposure> {
    cells
        .iter()
        .map(|(distance, fills)| QuoteExposure {
            timestamp_ms: 100_000,
            distance: *distance,
            exposure_ms: 60_000,
            fills: *fills,
        })
        .collect()
}

#[test]
fn test_walk_forward_unrepresentable_fold_does_not_lose_the_report() {
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_minima(30, 2, 1, 1);
    for cells in [
        [(dec!(10), 30), (dec!(10.01), 20)],
        [(dec!(33), 1000), (dec!(34), 135)],
        [(dec!(1.0), 20), (dec!(1.0000001), 10)],
    ] {
        let e = two_levels(cells);
        let record =
            calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
        assert_eq!(
            record.outcome,
            CalibrationOutcome::Insufficient(CalibrationFailure::UnrepresentableIntensity),
            "{cells:?}"
        );
    }

    let mut e = two_levels([(dec!(10), 30), (dec!(10.01), 20)]);
    e.extend(exposures(11, DAY_MS, 3 * DAY_MS, 2.0, 3.0, |_| 1.0));
    let spec = WindowSpec {
        train_ms: DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let splits = chronological_splits(0, 3 * DAY_MS, &spec).unwrap();
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &splits,
        &cfg,
        "d",
        &constant(),
    )
    .unwrap();
    assert_eq!(report.folds.len(), 2);
    assert!(report.folds[0].calibrated.is_none());
    assert!(report.folds[1].calibrated.is_some());
}

#[test]
fn test_calibrate_window_buckets_far_apart_keep_the_bisection_working() {
    // k = ln 2, at distances where exp(-k · d) underflows only with a large k.
    let outcome = two_bucket_outcome([
        [(dec!(1), 100, 200), (dec!(2), 100, 100)],
        [(dec!(700), 100, 200), (dec!(701), 100, 100)],
    ]);
    assert_eq!(
        outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::UnrepresentableIntensity)
    );
}

#[test]
fn test_calibrate_window_equal_rates_are_not_decaying() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_minima(30, 2, 1, 1);
    for cells in [
        [(dec!(0.003), 100), (dec!(0.006), 100)],
        [(dec!(0.1), 70), (dec!(0.3), 70)],
        [(dec!(7E28), 1000), (dec!(7.9E28), 900)],
    ] {
        let e = two_levels(cells);
        let record =
            calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
        assert_eq!(
            record.outcome,
            CalibrationOutcome::Insufficient(CalibrationFailure::NonDecayingIntensity),
            "{cells:?}"
        );
    }
}

#[test]
fn test_chronological_splits_too_many_are_refused() {
    let spec = WindowSpec {
        train_ms: 1,
        test_ms: 1,
        step_ms: 1,
    };
    assert!(chronological_splits(0, u64::MAX, &spec).is_err());
    assert_eq!(
        chronological_splits(0, 1_000_001, &spec).unwrap().len(),
        1_000_000
    );
}

#[test]
fn test_rate_per_second_huge_a_and_tiny_exp_keeps_the_rate() {
    // A about 5e27 and k about 2: exp(-k · 33) alone rounds away in a Decimal.
    let at = |t: u64, distance, fills| QuoteExposure {
        timestamp_ms: t,
        distance,
        exposure_ms: 6_000_000,
        fills,
    };
    let e = vec![
        at(40_000_000, dec!(33), 600),
        at(40_000_000, dec!(34), 81),
        at(DAY_MS + 40_000_000, dec!(33), 600),
        at(DAY_MS + 40_000_000, dec!(34), 81),
    ];
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let worse = ConstantParameters {
        volatility: dec!(1),
        intensity_scale_a: dec!(1),
        order_intensity_k: dec!(0.1),
    };
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &[split],
        &config(),
        "d",
        &worse,
    )
    .unwrap();
    let fold = &report.folds[0];
    let rate = fold
        .record
        .parameters()
        .unwrap()
        .rate_per_second(dec!(33), 40_000_000)
        .unwrap();
    assert!((rate.to_f64().unwrap() - 0.1).abs() < 1e-6, "rate {rate}");
    assert!(fold.calibrated.unwrap().intensity < fold.constant.intensity);
}

#[test]
fn test_walk_forward_realized_volatility_needs_samples_and_no_gap() {
    let train = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let run = |test: Vec<PriceSample>| {
        let mut p = train.clone();
        p.extend(test);
        let report = walk_forward(
            &CalibrationData::new(&p, &e).unwrap(),
            &[split],
            &config(),
            "d",
            &constant(),
        )
        .unwrap();
        (
            report.folds[0].calibrated.unwrap().volatility,
            report.folds[0].constant.volatility,
        )
    };
    let full = prices(9, DAY_MS, 2 * DAY_MS, 60_000, 0.5);
    let scored = run(full.clone());
    assert!(scored.0.is_some() && scored.1.is_some());

    let mut gap = full.clone();
    gap.retain(|s| !(DAY_MS + 600_000..DAY_MS + 1_200_000).contains(&s.timestamp_ms));
    assert_eq!(run(gap), (None, None));

    // One sample per 5 grid buckets: within the gap limit, so only counting
    // the buckets with a sample can refuse them.
    let cfg_buckets = prices(9, DAY_MS, 2 * DAY_MS, 300_000, 0.5);
    assert_eq!(cfg_buckets.len(), 288);
    let few: Vec<_> = full.iter().take(20).copied().collect();
    assert_eq!(run(few), (None, None));
    let sparse = |min: usize| {
        let cfg = config().with_minima(min, 2, 30, 60_000);
        let mut p = train.clone();
        p.extend(cfg_buckets.clone());
        walk_forward(
            &CalibrationData::new(&p, &e).unwrap(),
            &[split],
            &cfg,
            "d",
            &constant(),
        )
        .unwrap()
        .folds[0]
            .constant
            .volatility
    };
    assert!(sparse(288).is_some());
    assert!(sparse(289).is_none());
}

#[test]
fn test_calibrate_window_late_first_sample_is_a_gap() {
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let p = prices(7, 600_000, DAY_MS, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let record = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(
        record.outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: 600_000,
            limit_ms: 300_000
        })
    );
}

#[test]
fn test_walk_forward_fill_at_rate_zero_scores_the_floor() {
    // The afternoon has exposure and no fills in training, so its multiplier
    // is 0, and the one test exposure there has a fill.
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let mut e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    e.iter_mut()
        .filter(|x| x.timestamp_ms >= DAY_MS / 2)
        .for_each(|x| x.fills = 0);
    e.push(QuoteExposure {
        timestamp_ms: DAY_MS + DAY_MS / 2 + 60_000,
        distance: dec!(0.1),
        exposure_ms: 60_000,
        fills: 1,
    });
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let cfg = config().with_seasonality(2, 0);
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &[split],
        &cfg,
        "d",
        &constant(),
    )
    .unwrap();
    let score = report.folds[0]
        .calibrated
        .unwrap()
        .intensity
        .unwrap()
        .to_f64()
        .unwrap();
    let mu = 1e-12 * 60.0f64;
    let expected = (mu - mu.ln()) / 60.0;
    assert!((score - expected).abs() < 1e-9, "{score} {expected}");
}

#[test]
fn test_walk_forward_score_matches_the_exact_likelihood() {
    let cell = |timestamp_ms, distance, fills| QuoteExposure {
        timestamp_ms,
        distance,
        exposure_ms: 1_000,
        fills,
    };
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let cfg = config().with_seasonality(2, 0).with_minima(30, 2, 1, 1);
    for (afternoon_fills, expected) in [(0, -4.018260123027243), (1, 2.889495155954897)] {
        let mut e = Vec::new();
        for (day, extra) in [(0, 0), (DAY_MS, afternoon_fills)] {
            e.push(cell(day + 1_000_000, dec!(0), 10));
            e.push(cell(day + 1_000_000, dec!(1), 5));
            e.push(cell(day + 50_000_000, dec!(0), extra));
            e.push(cell(day + 50_000_000, dec!(1), 0));
        }
        let data = CalibrationData::new(&p, &e).unwrap();
        let report = walk_forward(&data, &[split], &cfg, "d", &constant()).unwrap();
        let score = report.folds[0]
            .calibrated
            .unwrap()
            .intensity
            .unwrap()
            .to_f64()
            .unwrap();
        assert!((score - expected).abs() < 1e-9, "{score} {expected}");
    }
}

#[test]
fn test_walk_forward_realized_volatility_gap_at_either_end_is_unscored() {
    let p = prices(7, 0, 2 * DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let split = ChronologicalSplit {
        train: TimeWindow {
            start_ms: 0,
            end_ms: DAY_MS,
        },
        test: TimeWindow {
            start_ms: DAY_MS,
            end_ms: 2 * DAY_MS,
        },
    };
    let silent: [std::ops::Range<u64>; 2] =
        [DAY_MS..DAY_MS + 600_000, 2 * DAY_MS - 600_000..2 * DAY_MS];
    for range in silent {
        let mut q = p.clone();
        q.retain(|s| !range.contains(&s.timestamp_ms));
        let report = walk_forward(
            &CalibrationData::new(&q, &e).unwrap(),
            &[split],
            &config(),
            "d",
            &constant(),
        )
        .unwrap();
        assert_eq!(report.folds[0].constant.volatility, None, "{range:?}");
    }
}

#[test]
fn test_rate_per_second_small_rate_is_kept() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let cell = |distance, fills| QuoteExposure {
        timestamp_ms: 1_000_000,
        distance,
        exposure_ms: 1_000,
        fills,
    };
    let e = [cell(dec!(0), 10), cell(dec!(1), 5)];
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_seasonality(1, 1).with_minima(30, 2, 1, 1);
    let record =
        calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d").unwrap();
    let rate = record
        .parameters()
        .unwrap()
        .rate_per_second(dec!(20), 1_000_000)
        .unwrap();
    let expected = dec!(0.0000095367431640625);
    assert!((rate - expected).abs() < dec!(0.000000000000001), "{rate}");
}

/// Trains on the first day, where the morning quotes only at distances 33 and
/// 34 and the afternoon at 1 and 1.5, both with `k` = 2. So `A` stays near 160,
/// while the morning's own scale `A · s` is about 4.6e29: above `Decimal` at
/// distance 0, and about 10 fills per second where it was observed.
fn extrapolating_training_exposures() -> Vec<QuoteExposure> {
    let at = |timestamp_ms, distance, fills| QuoteExposure {
        timestamp_ms,
        distance,
        exposure_ms: 6_000_000,
        fills,
    };
    vec![
        at(20_000_000, dec!(33), 60_000),
        at(20_000_000, dec!(34), 8_120),
        at(60_000_000, dec!(1), 81_000),
        at(60_000_000, dec!(1.5), 29_872),
    ]
}

fn frozen_folds_over_three_days() -> Vec<ChronologicalSplit> {
    let day = |i: u64| TimeWindow {
        start_ms: i * DAY_MS,
        end_ms: (i + 1) * DAY_MS,
    };
    vec![
        ChronologicalSplit {
            train: day(0),
            test: day(1),
        },
        ChronologicalSplit {
            train: day(1),
            test: day(2),
        },
    ]
}

/// One exposure on the morning of day 1, at each given distance in order, and
/// one on day 2 at a distance the training saw.
fn out_of_support_report(p: &[PriceSample], test_distances: &[Decimal]) -> WalkForwardReport {
    let mut e = extrapolating_training_exposures();
    for (i, distance) in test_distances.iter().enumerate() {
        e.push(QuoteExposure {
            timestamp_ms: DAY_MS + 20_000_000 + i as u64,
            distance: *distance,
            exposure_ms: 60_000,
            fills: 1,
        });
    }
    e.push(QuoteExposure {
        timestamp_ms: 2 * DAY_MS + 20_000_000,
        distance: dec!(33),
        exposure_ms: 60_000,
        fills: 600,
    });
    let cfg = config()
        .with_update_rule(UpdateRule::Frozen)
        .with_seasonality(2, 1)
        .with_minima(30, 2, 1, 1);
    walk_forward(
        &CalibrationData::new(p, &e).unwrap(),
        &frozen_folds_over_three_days(),
        &cfg,
        "d",
        &constant(),
    )
    .unwrap()
}

#[test]
fn test_walk_forward_rate_above_decimal_range_marks_only_that_fold() {
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let report = out_of_support_report(&p, &[dec!(0)]);
    assert_eq!(report.folds.len(), 2);

    let extreme = report.folds[0].calibrated.unwrap();
    assert_eq!(extreme.intensity, None, "no capped score");
    let forecast = extreme.intensity_out_of_support.unwrap();
    assert_eq!(forecast.timestamp_ms, DAY_MS + 20_000_000);
    assert_eq!(forecast.distance, dec!(0));
    // The rest of that fold is still scored.
    assert!(extreme.volatility.is_some());
    assert!(report.folds[0].constant.intensity.is_some());
    assert_eq!(report.folds[0].constant.intensity_out_of_support, None);

    // The other fold has the same parameters and a score of its own.
    let normal = report.folds[1].calibrated.unwrap();
    assert!(normal.intensity.is_some());
    assert_eq!(normal.intensity_out_of_support, None);
    assert_eq!(report.out_of_support_folds(), 1);
    assert_eq!(report.scored_folds(), 2);

    // The rate itself keeps its documented error, and is not capped either.
    let fit = report.folds[0].record.parameters().unwrap();
    assert!(matches!(
        fit.rate_per_second(dec!(0), 20_000_000),
        Err(market_maker_rs::types::error::MMError::NumericalError(_))
    ));
    let observed = fit.rate_per_second(dec!(33), 20_000_000).unwrap();
    assert!(
        (observed.to_f64().unwrap() / 10.0 - 1.0).abs() < 1e-3,
        "{observed}"
    );
}

#[test]
fn test_walk_forward_out_of_support_names_the_highest_rate() {
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    // Distance 33 is in range, then 0.5 overflows, and 0 overflows further.
    let report = out_of_support_report(&p, &[dec!(33), dec!(0.5), dec!(0), dec!(0.5)]);
    let forecast = report.folds[0]
        .calibrated
        .unwrap()
        .intensity_out_of_support
        .unwrap();
    assert_eq!(forecast.timestamp_ms, DAY_MS + 20_000_002);
    assert_eq!(forecast.distance, dec!(0));

    // On a tie, the earliest.
    let report = out_of_support_report(&p, &[dec!(0.5), dec!(0.5)]);
    let forecast = report.folds[0]
        .calibrated
        .unwrap()
        .intensity_out_of_support
        .unwrap();
    assert_eq!(forecast.timestamp_ms, DAY_MS + 20_000_000);
}

#[test]
fn test_walk_forward_out_of_support_alone_does_not_count_as_scored() {
    // Day 1 has no prices, so that fold has no volatility score either.
    let mut p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    p.retain(|s| !(DAY_MS..2 * DAY_MS).contains(&s.timestamp_ms));
    let report = out_of_support_report(&p, &[dec!(0)]);
    assert_eq!(report.folds[0].calibrated.unwrap().volatility, None);
    assert_eq!(report.out_of_support_folds(), 1);
    assert_eq!(report.scored_folds(), 1);
}

#[test]
fn test_walk_forward_in_range_rate_is_not_out_of_support() {
    // At distance 1 the morning rate is about 6.2e28, below the range limit.
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let report = out_of_support_report(&p, &[dec!(1)]);
    let score = report.folds[0].calibrated.unwrap();
    assert_eq!(score.intensity_out_of_support, None);
    assert!(score.intensity.unwrap() > dec!(1e28));
    assert_eq!(report.out_of_support_folds(), 0);
}

#[test]
fn test_walk_forward_constant_at_the_range_limit_keeps_the_report() {
    // `A` is the largest Decimal, which rounds above the range as an f64.
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let e = vec![QuoteExposure {
        timestamp_ms: DAY_MS + 20_000_000,
        distance: dec!(0),
        exposure_ms: 60_000,
        fills: 1,
    }];
    let extreme = ConstantParameters {
        volatility: dec!(1),
        intensity_scale_a: Decimal::MAX,
        order_intensity_k: dec!(1),
    };
    let report = walk_forward(
        &CalibrationData::new(&p, &e).unwrap(),
        &frozen_folds_over_three_days(),
        &config(),
        "d",
        &extreme,
    )
    .unwrap();
    let score = report.folds[0].constant;
    assert_eq!(score.intensity, None);
    assert_eq!(score.intensity_out_of_support.unwrap().distance, dec!(0));
    // No exposure on day 2, so nothing to score, and no marker either.
    assert_eq!(report.folds[1].constant.intensity, None);
    assert_eq!(report.folds[1].constant.intensity_out_of_support, None);
    // That counts calibrated forecasts only.
    assert_eq!(report.out_of_support_folds(), 0);
}

#[test]
fn test_walk_forward_score_rounding_above_the_range_marks_only_that_fold() {
    // The rate is the largest f64 under the range, so each exposure is in range
    // and only the rounding of their mean reaches 2^96. These durations do it.
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let top = ConstantParameters {
        volatility: dec!(1),
        intensity_scale_a: Decimal::from_str_exact("79228162514264328797450928128").unwrap(),
        order_intensity_k: dec!(1),
    };
    let durations: [&[u64]; 3] = [
        &[182_409, 116_756],
        &[440_241, 335_367],
        &[114_790, 129_975, 144_929],
    ];
    for exposure_times in durations {
        let at = |timestamp_ms, distance, exposure_ms| QuoteExposure {
            timestamp_ms,
            distance,
            exposure_ms,
            fills: 0,
        };
        let mut e: Vec<_> = exposure_times
            .iter()
            .enumerate()
            .map(|(i, ms)| at(DAY_MS + 1 + i as u64, dec!(0), *ms))
            .collect();
        e.push(at(2 * DAY_MS + 1, dec!(10), 60_000));
        let report = walk_forward(
            &CalibrationData::new(&p, &e).unwrap(),
            &frozen_folds_over_three_days(),
            &config(),
            "d",
            &top,
        )
        .unwrap();
        let marked = report.folds[0].constant;
        assert_eq!(marked.intensity, None, "{exposure_times:?}");
        let forecast = marked.intensity_out_of_support.unwrap();
        assert_eq!(forecast.timestamp_ms, DAY_MS + 1);
        assert_eq!(forecast.distance, dec!(0));
        // The other fold keeps its score.
        assert!(report.folds[1].constant.intensity.is_some());
        assert_eq!(report.folds[1].constant.intensity_out_of_support, None);
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_walk_forward_report_serde_keeps_out_of_support_and_reads_older_reports() {
    let p = prices(7, 0, 3 * DAY_MS, 60_000, 0.5);
    let report = out_of_support_report(&p, &[dec!(0)]);
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(
        serde_json::from_str::<WalkForwardReport>(&json).unwrap(),
        report
    );

    // A report written before the field existed reads as having no marker.
    let in_range = out_of_support_report(&p, &[dec!(1)]);
    let json = serde_json::to_string(&in_range).unwrap();
    let older = json.replace(",\"intensity_out_of_support\":null", "");
    assert_ne!(older, json);
    assert_eq!(
        serde_json::from_str::<WalkForwardReport>(&older).unwrap(),
        in_range
    );
}

const HOUR_MS: u64 = 3_600_000;

/// 09:00 to 17:00 UTC on each of the first `days` days.
fn day_sessions(days: u64) -> Vec<TimeWindow> {
    (0..days)
        .map(|d| TimeWindow {
            start_ms: d * DAY_MS + 9 * HOUR_MS,
            end_ms: d * DAY_MS + 17 * HOUR_MS,
        })
        .collect()
}

/// One path of one minute prices, cut across the sessions, so that the price
/// does not move while the market is closed.
fn session_prices(sessions: &[TimeWindow], sigma: f64) -> Vec<PriceSample> {
    let open_ms: u64 = sessions.iter().map(|s| s.end_ms - s.start_ms).sum();
    let mut path = prices(7, 0, open_ms, 60_000, sigma).into_iter();
    sessions
        .iter()
        .flat_map(|s| {
            let steps = (s.end_ms - s.start_ms) / 60_000;
            (0..steps)
                .map(|i| PriceSample {
                    timestamp_ms: s.start_ms + i * 60_000,
                    price: path.next().unwrap().price,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Exposures of one minute inside each session.
fn session_exposures(sessions: &[TimeWindow]) -> Vec<QuoteExposure> {
    sessions
        .iter()
        .enumerate()
        .flat_map(|(i, s)| exposures(11 + i as u64, s.start_ms, s.end_ms, 2.0, 3.0, |_| 1.0))
        .collect()
}

fn in_sessions<'a>(
    p: &'a [PriceSample],
    e: &'a [QuoteExposure],
    sessions: &'a [TimeWindow],
) -> CalibrationData<'a> {
    CalibrationData::new(p, e)
        .unwrap()
        .with_sessions(sessions)
        .unwrap()
}

/// Four prices of one minute from `from`, with log returns `+r`, `-r`, `+r`.
fn zigzag(from: u64, level: f64, r: f64) -> Vec<PriceSample> {
    (0..4u64)
        .map(|i| PriceSample {
            timestamp_ms: from + i * 60_000,
            price: Decimal::from_f64(if i % 2 == 0 { level } else { level * r.exp() }).unwrap(),
        })
        .collect()
}

#[test]
fn test_calibrate_window_overnight_closure_neither_fails_nor_dilutes() {
    let sigma = 0.5;
    let sessions = day_sessions(3);
    let p = session_prices(&sessions, sigma);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(0, 3 * DAY_MS).unwrap();
    let relative = |outcome: &CalibrationOutcome| match outcome {
        CalibrationOutcome::Calibrated(fit) => {
            (fit.volatility / fit.reference_price).to_f64().unwrap()
        }
        other => panic!("{other:?}"),
    };

    // Continuous by default: the closure fails the window.
    let plain = CalibrationData::new(&p, &e).unwrap();
    let continuous = calibrate_window(&plain, &window, &config(), "d").unwrap();
    assert!(matches!(
        continuous.outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap { gap_ms, .. })
            if gap_ms > 15 * HOUR_MS
    ));
    // And raising the limit to pass it dilutes the volatility with zero returns,
    // by the square root of the open share of the grid, 24 of its 63 hours, as
    // it starts at the first sample.
    let lax = config().with_max_gap_buckets(2_000);
    let diluted = relative(
        &calibrate_window(&plain, &window, &lax, "d")
            .unwrap()
            .outcome,
    );
    let open_share = 24.0f64 / 63.0;
    assert!(
        (diluted / (sigma * open_share.sqrt()) - 1.0).abs() < 0.06,
        "{diluted}"
    );

    // With sessions, the diffusion is recovered at the same limit.
    let data = in_sessions(&p, &e, &sessions);
    let record = calibrate_window(&data, &window, &config(), "d").unwrap();
    let recovered = relative(&record.outcome);
    assert!((recovered / sigma - 1.0).abs() < 0.06, "{recovered}");
    let fit = record.parameters().unwrap();
    assert!((fit.order_intensity_k.to_f64().unwrap() / 3.0 - 1.0).abs() < 0.1);
    assert!((fit.intensity_scale_a.to_f64().unwrap() / 2.0 - 1.0).abs() < 0.2);
    assert_eq!(fit.reference_price, p.last().unwrap().price);
}

#[test]
fn test_calibrate_window_no_return_crosses_a_closure_by_hand() {
    // Log returns +r, -r, +r in each session, with a jump of 50% between them.
    let r = 0.001f64;
    let sessions = [
        TimeWindow::new(0, 240_000).unwrap(),
        TimeWindow::new(HOUR_MS, HOUR_MS + 240_000).unwrap(),
    ];
    let mut p = zigzag(0, 100.0, r);
    p.extend(zigzag(HOUR_MS, 150.0, r));
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let data = in_sessions(&p, &e, &sessions);
    let steps_per_year = 31_536_000_000.0f64 / 60_000.0;
    let reference = p.last().unwrap().price;
    let reference_f64 = reference.to_f64().unwrap();
    let cfg = config().with_minima(3, 2, 1, 1);

    let record = calibrate_window(&data, &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    assert_eq!(fit.reference_price, reference);
    // Six returns of mean r/3: the squared deviations add up to 48 r² / 9.
    let variance = 48.0 * r * r / 9.0 / 5.0;
    let expected = reference_f64 * (variance * steps_per_year).sqrt();
    let got = fit.volatility.to_f64().unwrap();
    assert!((got / expected - 1.0).abs() < 1e-6, "{got} {expected}");

    // The EWMA recursion runs over the same returns, from their variance.
    let ewma = cfg.with_volatility_rule(VolatilityRule::Ewma { lambda: dec!(0.5) });
    let record = calibrate_window(&data, &window, &ewma, "d").unwrap();
    // The first five returns have mean r/5 and squared deviations of 4.8 r².
    let mut variance = 4.8 * r * r / 4.0;
    for step in [r, -r, r, r, -r, r] {
        variance = 0.5 * variance + 0.5 * step * step;
    }
    let expected = reference_f64 * (variance * steps_per_year).sqrt();
    let got = record.parameters().unwrap().volatility.to_f64().unwrap();
    assert!((got / expected - 1.0).abs() < 1e-6, "{got} {expected}");
}

#[test]
fn test_calibrate_window_session_grids_share_no_price() {
    // The first session ends in a partial bucket with a wild price, and the
    // second has no sample in its first bucket. Neither may reach its grid.
    let sessions = [
        TimeWindow::new(0, 240_000 + 30_000).unwrap(),
        TimeWindow::new(HOUR_MS, HOUR_MS + 300_000).unwrap(),
    ];
    let r = 0.001f64;
    let quiet: Vec<_> = zigzag(0, 100.0, r)
        .into_iter()
        .chain(zigzag(HOUR_MS + 60_000, 150.0, r))
        .collect();
    let mut wild = quiet.clone();
    wild.insert(
        4,
        PriceSample {
            timestamp_ms: 250_000,
            price: dec!(1000),
        },
    );
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_minima(3, 2, 1, 1);
    let run = |p: &[PriceSample]| {
        calibrate_window(&in_sessions(p, &e, &sessions), &window, &cfg, "d").unwrap()
    };
    let quiet_fit = run(&quiet);
    let wild_fit = run(&wild);
    assert_eq!(quiet_fit.outcome, wild_fit.outcome);
    // Hand value of the same six returns, so the runs did not just agree.
    let variance = 48.0 * r * r / 9.0 / 5.0;
    let expected = quiet.last().unwrap().price.to_f64().unwrap() * (variance * 525_600.0f64).sqrt();
    let got = quiet_fit.parameters().unwrap().volatility.to_f64().unwrap();
    assert!((got / expected - 1.0).abs() < 1e-6, "{got} {expected}");
}

#[test]
fn test_calibrate_window_silence_inside_a_session_stays_a_gap() {
    let sessions = day_sessions(2);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(0, 2 * DAY_MS).unwrap();
    let run = |p: &[PriceSample]| {
        calibrate_window(&in_sessions(p, &e, &sessions), &window, &config(), "d")
            .unwrap()
            .outcome
    };
    let limit_ms = 300_000;

    // A feed outage of two hours during the second session.
    let mut outage = session_prices(&sessions, 0.5);
    let (from, to) = (DAY_MS + 10 * HOUR_MS, DAY_MS + 12 * HOUR_MS);
    outage.retain(|s| !(from..to).contains(&s.timestamp_ms));
    assert_eq!(
        run(&outage),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: 2 * HOUR_MS + 60_000,
            limit_ms,
        })
    );

    // A declared session with no prices at all.
    let mut silent = session_prices(&sessions, 0.5);
    silent.retain(|s| s.timestamp_ms < DAY_MS);
    assert_eq!(
        run(&silent),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: 8 * HOUR_MS,
            limit_ms,
        })
    );

    // A late open, and a feed that stops before the close, each in one session.
    let mut late = session_prices(&sessions, 0.5);
    late.retain(|s| s.timestamp_ms >= 9 * HOUR_MS + 600_000);
    assert_eq!(
        run(&late),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: 600_000,
            limit_ms,
        })
    );
    let mut early = session_prices(&sessions, 0.5);
    early.retain(|s| s.timestamp_ms < 16 * HOUR_MS || s.timestamp_ms >= DAY_MS);
    assert_eq!(
        run(&early),
        CalibrationOutcome::Insufficient(CalibrationFailure::PriceGap {
            gap_ms: HOUR_MS + 60_000,
            limit_ms,
        })
    );
}

#[test]
fn test_calibrate_window_returns_inside_sessions_are_counted() {
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_minima(3, 2, 1, 1);
    let sample = |timestamp_ms, price| PriceSample {
        timestamp_ms,
        price,
    };
    let session =
        |start: u64, minutes: u64| TimeWindow::new(start, start + minutes * 60_000).unwrap();
    let run = |p: &[PriceSample], s: &[TimeWindow]| {
        calibrate_window(&in_sessions(p, &e, s), &window, &cfg, "d")
            .unwrap()
            .outcome
    };

    // Enough grid prices, but each session has only one.
    let singles = [session(0, 1), session(HOUR_MS, 1), session(2 * HOUR_MS, 1)];
    let p = [
        sample(0, dec!(100)),
        sample(HOUR_MS, dec!(101)),
        sample(2 * HOUR_MS, dec!(102)),
    ];
    assert_eq!(
        run(&p, &singles),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewReturns { have: 0, need: 2 })
    );

    // One return is not enough for a variance, and two are.
    let pair = [session(0, 2), session(HOUR_MS, 1)];
    let p = [
        sample(0, dec!(100)),
        sample(60_000, dec!(101)),
        sample(HOUR_MS, dec!(102)),
    ];
    assert_eq!(
        run(&p, &pair),
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewReturns { have: 1, need: 2 })
    );
    let pairs = [session(0, 2), session(HOUR_MS, 2)];
    let p = [
        sample(0, dec!(100)),
        sample(60_000, dec!(101)),
        sample(HOUR_MS, dec!(102)),
        sample(HOUR_MS + 60_000, dec!(102.5)),
    ];
    assert!(matches!(run(&p, &pairs), CalibrationOutcome::Calibrated(_)));

    // A session shorter than a bucket gives no grid price; the reference price
    // is still the last one on the grid.
    let short = [
        session(0, 2),
        session(HOUR_MS, 2),
        TimeWindow::new(2 * HOUR_MS, 2 * HOUR_MS + 30_000).unwrap(),
    ];
    let mut with_short = p.to_vec();
    with_short.push(sample(2 * HOUR_MS, dec!(500)));
    match run(&with_short, &short) {
        CalibrationOutcome::Calibrated(fit) => assert_eq!(fit.reference_price, dec!(102.5)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_calibration_data_invalid_sessions_are_refused() {
    let e: Vec<QuoteExposure> = Vec::new();
    let window = |start_ms, end_ms| TimeWindow { start_ms, end_ms };
    let sample = |timestamp_ms| PriceSample {
        timestamp_ms,
        price: dec!(100),
    };
    let declare = |p: &[PriceSample], s: &[TimeWindow]| {
        CalibrationData::new(p, &e)
            .unwrap()
            .with_sessions(s)
            .map(|_| ())
    };
    let configuration = |r: Result<(), market_maker_rs::types::error::MMError>| {
        matches!(
            r,
            Err(market_maker_rs::types::error::MMError::InvalidConfiguration(_))
        )
    };
    let market = |r: Result<(), market_maker_rs::types::error::MMError>| {
        matches!(
            r,
            Err(market_maker_rs::types::error::MMError::InvalidMarketState(
                _
            ))
        )
    };

    // No sessions would otherwise read as always open.
    assert!(configuration(declare(&[], &[])));
    assert!(configuration(declare(&[], &[window(5, 5)])));
    assert!(configuration(declare(&[], &[window(6, 5)])));
    assert!(configuration(declare(&[], &[window(0, 10), window(9, 20)])));
    assert!(configuration(declare(&[], &[window(10, 20), window(0, 5)])));
    // Touching sessions are two sessions.
    assert!(declare(&[], &[window(0, 10), window(10, 20)]).is_ok());

    // Each session is half open, and a sample outside them all is refused.
    let sessions = [window(10, 20), window(30, 40)];
    assert!(declare(&[sample(10), sample(19), sample(30), sample(39)], &sessions).is_ok());
    assert!(market(declare(&[sample(9)], &sessions)));
    assert!(market(declare(&[sample(20)], &sessions)));
    assert!(market(declare(&[sample(29)], &sessions)));
    assert!(market(declare(&[sample(40)], &sessions)));
    assert!(market(declare(&[sample(10), sample(25)], &sessions)));

    // Exposures say how long the quotes were at risk, and are not checked.
    let outside = [QuoteExposure {
        timestamp_ms: 25,
        distance: dec!(0.1),
        exposure_ms: 1,
        fills: 0,
    }];
    assert!(
        CalibrationData::new(&[], &outside)
            .unwrap()
            .with_sessions(&sessions)
            .is_ok()
    );
}

#[test]
fn test_calibrate_window_sessions_are_part_of_what_the_record_hashes() {
    let sessions = day_sessions(2);
    let p = session_prices(&sessions[..1], 0.5);
    let e = session_exposures(&sessions[..1]);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let run = |s: &[TimeWindow]| {
        calibrate_window(&in_sessions(&p, &e, s), &window, &config(), "d").unwrap()
    };
    let base = run(&sessions);

    // A session the window does not reach, or the part of one past its end, is
    // later data, so the record does not depend on it.
    let mut later = sessions.clone();
    later[1].end_ms += HOUR_MS;
    assert_eq!(run(&later), base);
    let mut past_the_end = sessions.clone();
    past_the_end[0].end_ms = 17 * HOUR_MS + 60_000;
    let narrow = TimeWindow::new(0, 17 * HOUR_MS).unwrap();
    let clipped = |s: &[TimeWindow]| {
        calibrate_window(&in_sessions(&p, &e, s), &narrow, &config(), "d").unwrap()
    };
    assert_eq!(clipped(&past_the_end), clipped(&sessions));

    // A session the window does reach changes what the data means.
    let mut earlier = sessions.clone();
    earlier[0].start_ms -= 60_000;
    assert_ne!(run(&earlier).dataset_hash, base.dataset_hash);
    let mut later_close = sessions.clone();
    later_close[0].end_ms += 60_000;
    assert_ne!(run(&later_close).dataset_hash, base.dataset_hash);

    // Without sessions the hash is what it was: a plain record is unaffected.
    let plain = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    assert_ne!(plain.dataset_hash, base.dataset_hash);
}

#[test]
fn test_calibrate_window_session_touching_the_window_is_not_part_of_it() {
    // Windows are half open: a session ending where the window starts, or
    // starting where it ends, has no time in common with it.
    let inside = TimeWindow::new(DAY_MS + 9 * HOUR_MS, DAY_MS + 17 * HOUR_MS).unwrap();
    let p = session_prices(&[inside], 0.5);
    let e = session_exposures(&[inside]);
    let window = TimeWindow::new(DAY_MS, 2 * DAY_MS).unwrap();
    let run = |s: &[TimeWindow]| {
        calibrate_window(&in_sessions(&p, &e, s), &window, &config(), "d").unwrap()
    };
    let before = TimeWindow::new(9 * HOUR_MS, DAY_MS).unwrap();
    let after = TimeWindow::new(2 * DAY_MS, 2 * DAY_MS + HOUR_MS).unwrap();
    assert_eq!(run(&[before, inside, after]), run(&[inside]));
}

#[test]
fn test_calibrate_window_session_ending_at_the_start_changes_the_hash_if_it_moves_a_stamp() {
    // An exposure stamped at the window's start counts in the session that ended
    // then, which is otherwise outside the window, so the hash must see it.
    let inside = TimeWindow::new(DAY_MS + 9 * HOUR_MS, DAY_MS + 17 * HOUR_MS).unwrap();
    let before = TimeWindow::new(DAY_MS - 60_000, DAY_MS).unwrap();
    let p = session_prices(&[inside], 0.5);
    let plain = session_exposures(&[inside]);
    let mut stamped = plain.clone();
    stamped.insert(
        0,
        QuoteExposure {
            timestamp_ms: DAY_MS,
            distance: dec!(0.1),
            exposure_ms: 60_000,
            fills: 2,
        },
    );
    let window = TimeWindow::new(DAY_MS, 2 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 1);
    let run = |e: &[QuoteExposure], s: &[TimeWindow]| {
        calibrate_window(&in_sessions(&p, e, s), &window, &cfg, "d").unwrap()
    };
    let alone = run(&stamped, &[inside]);
    let with_before = run(&stamped, &[before, inside]);
    assert!(alone.parameters().unwrap().seasonality[0] > Decimal::ZERO);
    assert_eq!(alone.parameters().unwrap().seasonality[23], Decimal::ZERO);
    assert!(with_before.parameters().unwrap().seasonality[23] > Decimal::ZERO);
    assert_eq!(
        with_before.parameters().unwrap().seasonality[0],
        Decimal::ZERO
    );
    assert_ne!(alone.dataset_hash, with_before.dataset_hash);

    // Without such a stamp that session decides nothing, and neither record moves.
    let cfg_day = config();
    let unstamped = |s: &[TimeWindow]| {
        calibrate_window(&in_sessions(&p, &plain, s), &window, &cfg_day, "d").unwrap()
    };
    assert_eq!(unstamped(&[before, inside]), unstamped(&[inside]));
}

#[test]
fn test_calibrate_window_without_sessions_hashes_only_the_samples() {
    // FNV-1a 128 of the samples alone: records written before sessions existed
    // keep their hash.
    let p = prices(7, 0, 3_600_000, 60_000, 0.5);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let record = calibrate_window(
        &CalibrationData::new(&p, &[]).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    let mut expected = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
    for s in &p {
        let bytes = s.timestamp_ms.to_le_bytes();
        let price = s.price.normalize().serialize();
        for b in bytes.iter().chain(&price) {
            expected =
                (expected ^ u128::from(*b)).wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    }
    assert_eq!(record.dataset_hash, expected);
}

#[test]
fn test_calibrate_window_grid_limit_counts_every_session() {
    // Each session is under the limit of ten million points, but not the sum.
    let sessions = [
        TimeWindow::new(0, 6_000_000).unwrap(),
        TimeWindow::new(7_000_000, 13_000_000).unwrap(),
    ];
    let cfg = config().with_resolution_ms(1);
    let data = CalibrationData::new(&[], &[])
        .unwrap()
        .with_sessions(&sessions)
        .unwrap();
    let whole = TimeWindow::new(0, 13_000_000).unwrap();
    assert!(matches!(
        calibrate_window(&data, &whole, &cfg, "d"),
        Err(market_maker_rs::types::error::MMError::InvalidConfiguration(_))
    ));
    let one = TimeWindow::new(0, 6_000_000).unwrap();
    assert!(calibrate_window(&data, &one, &cfg, "d").is_ok());
}

#[test]
fn test_calibrate_window_session_wider_than_the_window_equals_no_session() {
    // Clipped to the window, the session covers it, so nothing may change.
    let session = [TimeWindow::new(0, DAY_MS).unwrap()];
    let p = session_prices(&session, 0.5);
    let e = session_exposures(&session);
    let window = TimeWindow::new(HOUR_MS, 5 * HOUR_MS + 30_000).unwrap();
    let declared = calibrate_window(&in_sessions(&p, &e, &session), &window, &config(), "d");
    let plain = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    );
    assert_eq!(declared.unwrap().outcome, plain.unwrap().outcome);
}

#[test]
fn test_walk_forward_realized_volatility_uses_the_sessions() {
    let sessions = day_sessions(4);
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let split = ChronologicalSplit {
        train: TimeWindow::new(0, 2 * DAY_MS).unwrap(),
        test: TimeWindow::new(2 * DAY_MS, 4 * DAY_MS).unwrap(),
    };
    let data = in_sessions(&p, &e, &sessions);
    let report = walk_forward(&data, &[split], &config(), "d", &constant()).unwrap();
    let on_test = calibrate_window(&data, &split.test, &config(), "d").unwrap();
    let realized = on_test.parameters().unwrap().volatility;
    assert_eq!(
        report.folds[0].constant.volatility,
        Some((constant().volatility - realized).abs())
    );
    assert!(report.folds[0].calibrated.unwrap().volatility.is_some());

    // The same closures unscore the test window when they are not declared,
    // and the training window of that fold fails as well.
    let plain = CalibrationData::new(&p, &e).unwrap();
    let report = walk_forward(&plain, &[split], &config(), "d", &constant()).unwrap();
    assert_eq!(report.folds[0].constant.volatility, None);
    assert!(report.folds[0].calibrated.is_none());
}

#[test]
fn test_walk_forward_realized_volatility_needs_returns_inside_sessions() {
    // Three test sessions of one bucket each pass the minimum of grid prices
    // and the gap limit, and give no return to measure.
    let day = |i: u64| i * DAY_MS;
    let mut sessions = day_sessions(2);
    let singles = [day(2), day(2) + HOUR_MS, day(2) + 2 * HOUR_MS];
    sessions.extend(
        singles
            .iter()
            .map(|s| TimeWindow::new(*s, s + 60_000).unwrap()),
    );
    let mut p = session_prices(&sessions[..2], 0.5);
    p.extend(singles.iter().map(|s| PriceSample {
        timestamp_ms: *s,
        price: dec!(100),
    }));
    let e = session_exposures(&sessions[..2]);
    let split = ChronologicalSplit {
        train: TimeWindow::new(0, day(2)).unwrap(),
        test: TimeWindow::new(day(2), day(3)).unwrap(),
    };
    let cfg = config().with_minima(3, 2, 1, 1);
    let report = walk_forward(
        &in_sessions(&p, &e, &sessions),
        &[split],
        &cfg,
        "d",
        &constant(),
    )
    .unwrap();
    assert!(report.folds[0].record.parameters().is_some());
    assert_eq!(report.folds[0].constant.volatility, None);
    assert_eq!(report.folds[0].calibrated.unwrap().volatility, None);
}

#[test]
fn test_calibrate_window_closed_seasonal_buckets_are_not_a_failure() {
    let sessions = day_sessions(3);
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(0, 3 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 30);

    let plain = CalibrationData::new(&p, &e).unwrap();
    let lax = cfg.with_max_gap_buckets(2_000);
    assert_eq!(
        calibrate_window(&plain, &window, &lax, "d")
            .unwrap()
            .outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::UnobservedSeasonalBucket {
            bucket: 0
        })
    );

    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    assert_eq!(fit.seasonality.len(), 24);
    for (bucket, multiplier) in fit.seasonality.iter().enumerate() {
        if (9..17).contains(&bucket) {
            assert!(
                (multiplier.to_f64().unwrap() - 1.0).abs() < 0.2,
                "{multiplier}"
            );
        } else {
            assert_eq!(*multiplier, Decimal::ZERO, "bucket {bucket}");
        }
    }
    // `A` is per second at risk: the closed hours do not dilute it.
    assert!((fit.intensity_scale_a.to_f64().unwrap() / 2.0 - 1.0).abs() < 0.2);
    assert_eq!(
        fit.rate_per_second(dec!(0.1), 3 * HOUR_MS).unwrap(),
        Decimal::ZERO
    );
    assert!(fit.rate_per_second(dec!(0.1), 12 * HOUR_MS).unwrap() > Decimal::ZERO);
}

#[test]
fn test_calibrate_window_unobserved_bucket_stays_a_failure_unless_closed() {
    let cfg = config().with_seasonality(24, 1).with_minima(30, 2, 1, 1);
    let unobserved = |bucket| {
        CalibrationOutcome::Insufficient(CalibrationFailure::UnobservedSeasonalBucket { bucket })
    };

    // An open bucket without exposure is unknown, not closed.
    let sessions = day_sessions(2);
    let p = session_prices(&sessions, 0.5);
    let window = TimeWindow::new(0, 2 * DAY_MS).unwrap();
    let mut e = session_exposures(&sessions);
    e.retain(|x| !(12 * HOUR_MS..13 * HOUR_MS).contains(&(x.timestamp_ms % DAY_MS)));
    assert_eq!(
        calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d")
            .unwrap()
            .outcome,
        unobserved(12)
    );

    // So is one open for only part of the hour: the session opens at 09:30.
    let half_hour = [TimeWindow::new(9 * HOUR_MS + 1_800_000, 17 * HOUR_MS).unwrap()];
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let p = session_prices(&half_hour, 0.5);
    let mut e = session_exposures(&half_hour);
    e.retain(|x| x.timestamp_ms >= 10 * HOUR_MS);
    assert_eq!(
        calibrate_window(&in_sessions(&p, &e, &half_hour), &window, &cfg, "d")
            .unwrap()
            .outcome,
        unobserved(9)
    );

    // And a bucket the window never reaches says nothing about the venue.
    let sessions = day_sessions(1);
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(9 * HOUR_MS, 17 * HOUR_MS).unwrap();
    assert_eq!(
        calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d")
            .unwrap()
            .outcome,
        unobserved(0)
    );

    // An exposure stamped in a closed hour makes that bucket observed.
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let mut e = session_exposures(&sessions);
    e.push(QuoteExposure {
        timestamp_ms: 20 * HOUR_MS,
        distance: dec!(0.1),
        exposure_ms: 60_000,
        fills: 0,
    });
    assert_eq!(
        calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d")
            .unwrap()
            .outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::ThinSeasonalBucket {
            bucket: 20,
            have: 0,
            need: 1,
        })
    );
}

#[test]
fn test_calibrate_window_session_across_midnight_opens_both_days_buckets() {
    // 22:00 to 02:00, on two nights.
    let sessions: Vec<_> = (0..2u64)
        .map(|d| {
            TimeWindow::new(d * DAY_MS + 22 * HOUR_MS, (d + 1) * DAY_MS + 2 * HOUR_MS).unwrap()
        })
        .collect();
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(0, 3 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 30);
    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    for (bucket, multiplier) in fit.seasonality.iter().enumerate() {
        let open = !(2..22).contains(&bucket);
        assert_eq!(*multiplier > Decimal::ZERO, open, "bucket {bucket}");
    }
}

/// One exposure of a minute with fills, stamped at each session's end.
fn exposures_at_the_close(sessions: &[TimeWindow], fills: u64) -> Vec<QuoteExposure> {
    sessions
        .iter()
        .map(|s| QuoteExposure {
            timestamp_ms: s.end_ms,
            distance: dec!(0.1),
            exposure_ms: 60_000,
            fills,
        })
        .collect()
}

fn with_closing_exposures(sessions: &[TimeWindow], fills: u64) -> Vec<QuoteExposure> {
    let mut e = session_exposures(sessions);
    e.extend(exposures_at_the_close(sessions, fills));
    e.sort_by_key(|x| x.timestamp_ms);
    e
}

#[test]
fn test_calibrate_window_exposure_at_the_close_keeps_the_closed_bucket_closed() {
    let sessions = day_sessions(3);
    let p = session_prices(&sessions, 0.5);
    let window = TimeWindow::new(0, 3 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 30);

    // A few fills at the close do not make the next hour a thin bucket.
    let e = with_closing_exposures(&sessions, 2);
    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    assert_eq!(fit.seasonality[17], Decimal::ZERO);
    assert!(fit.seasonality[16] > Decimal::ZERO);

    // Nor do many give the closed hour a multiplier, and they count in the last.
    let e = with_closing_exposures(&sessions, 20_000);
    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    let fit = record.parameters().unwrap();
    assert_eq!(fit.seasonality[17], Decimal::ZERO);
    assert!(fit.seasonality[16] > dec!(1.5) * fit.seasonality[15]);
}

#[test]
fn test_calibrate_window_session_ending_inside_a_bucket_keeps_its_stamp_there() {
    // The close at 17:30 is in the hour of 17:00 either way, and it stays open.
    let sessions: Vec<_> = (0..2u64)
        .map(|d| {
            TimeWindow::new(
                d * DAY_MS + 9 * HOUR_MS,
                d * DAY_MS + 17 * HOUR_MS + 1_800_000,
            )
            .unwrap()
        })
        .collect();
    let p = session_prices(&sessions, 0.5);
    let e = with_closing_exposures(&sessions, 2);
    let window = TimeWindow::new(0, 2 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 1);
    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    assert!(record.parameters().unwrap().seasonality[17] > Decimal::ZERO);
}

#[test]
fn test_calibrate_window_stamp_at_adjacent_sessions_stays_in_the_later_one() {
    // 09:00 to 12:00 and 12:00 to 17:00: the market does not close at noon, so
    // an exposure stamped then is in the hour after it. No other exposure is.
    let sessions: Vec<_> = (0..2u64)
        .flat_map(|d| {
            [
                TimeWindow::new(d * DAY_MS + 9 * HOUR_MS, d * DAY_MS + 12 * HOUR_MS).unwrap(),
                TimeWindow::new(d * DAY_MS + 12 * HOUR_MS, d * DAY_MS + 17 * HOUR_MS).unwrap(),
            ]
        })
        .collect();
    let p = session_prices(&sessions, 0.5);
    let mut e = session_exposures(&sessions);
    e.retain(|x| x.timestamp_ms % DAY_MS / HOUR_MS != 12);
    for d in 0..2u64 {
        for (level, fills) in [(dec!(0.1), 30), (dec!(0.3), 15)] {
            e.push(QuoteExposure {
                timestamp_ms: d * DAY_MS + 12 * HOUR_MS,
                distance: level,
                exposure_ms: 60_000,
                fills,
            });
        }
    }
    e.sort_by_key(|x| x.timestamp_ms);
    let window = TimeWindow::new(0, 2 * DAY_MS).unwrap();
    let cfg = config().with_seasonality(24, 1);
    let record = calibrate_window(&in_sessions(&p, &e, &sessions), &window, &cfg, "d").unwrap();
    assert!(record.parameters().unwrap().seasonality[12] > Decimal::ZERO);
}

#[test]
fn test_calibrate_window_without_sessions_a_stamp_stays_in_its_own_bucket() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let mut e = exposures(11, 0, DAY_MS / 2, 2.0, 3.0, |_| 1.0);
    e.push(QuoteExposure {
        timestamp_ms: DAY_MS / 2,
        distance: dec!(0.1),
        exposure_ms: 60_000,
        fills: 0,
    });
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let cfg = config().with_seasonality(2, 1).with_minima(30, 2, 1, 1);
    assert_eq!(
        calibrate_window(&CalibrationData::new(&p, &e).unwrap(), &window, &cfg, "d")
            .unwrap()
            .outcome,
        CalibrationOutcome::Insufficient(CalibrationFailure::ThinSeasonalBucket {
            bucket: 1,
            have: 0,
            need: 1,
        })
    );
}

#[test]
fn test_walk_forward_exposure_at_the_close_scores_at_the_last_bucket() {
    let sessions = day_sessions(2);
    let p = session_prices(&sessions, 0.5);
    let mut e = session_exposures(&sessions[..1]);
    let close = sessions[1].end_ms;
    e.push(QuoteExposure {
        timestamp_ms: close,
        distance: dec!(0.1),
        exposure_ms: 60_000,
        fills: 1,
    });
    let split = ChronologicalSplit {
        train: TimeWindow::new(0, DAY_MS).unwrap(),
        test: TimeWindow::new(DAY_MS, 2 * DAY_MS).unwrap(),
    };
    let cfg = config().with_seasonality(24, 1);
    let report = walk_forward(
        &in_sessions(&p, &e, &sessions),
        &[split],
        &cfg,
        "d",
        &constant(),
    )
    .unwrap();
    let fit = report.folds[0].record.parameters().unwrap();
    let rate = fit
        .rate_per_second(dec!(0.1), close - 1)
        .unwrap()
        .to_f64()
        .unwrap();
    assert!(rate > 0.0);
    let mu = rate * 60.0;
    let expected = (mu - mu.ln()) / 60.0;
    let score = report.folds[0]
        .calibrated
        .unwrap()
        .intensity
        .unwrap()
        .to_f64()
        .unwrap();
    assert!((score - expected).abs() < 1e-6, "{score} {expected}");
}

#[test]
fn test_walk_forward_bucket_closed_in_training_predicts_no_fills_at_a_visible_cost() {
    // A half day in training, and a full day in the test.
    let sessions = [
        TimeWindow::new(9 * HOUR_MS, 13 * HOUR_MS).unwrap(),
        TimeWindow::new(DAY_MS + 9 * HOUR_MS, DAY_MS + 17 * HOUR_MS).unwrap(),
    ];
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let split = ChronologicalSplit {
        train: TimeWindow::new(0, DAY_MS).unwrap(),
        test: TimeWindow::new(DAY_MS, 2 * DAY_MS).unwrap(),
    };
    let cfg = config().with_seasonality(24, 1);
    let report = walk_forward(
        &in_sessions(&p, &e, &sessions),
        &[split],
        &cfg,
        "d",
        &constant(),
    )
    .unwrap();
    let fold = &report.folds[0];
    let fit = fold.record.parameters().unwrap();
    assert_eq!(
        fit.rate_per_second(dec!(0.1), DAY_MS + 15 * HOUR_MS)
            .unwrap(),
        Decimal::ZERO
    );
    // The fills of those open hours cost far more than a constant's error.
    let calibrated = fold.calibrated.unwrap().intensity.unwrap();
    let constant_score = fold.constant.intensity.unwrap();
    assert!(
        calibrated > constant_score + dec!(1),
        "{calibrated} {constant_score}"
    );
}

#[test]
fn test_calibration_record_open_ms_says_how_much_was_open() {
    let sessions = day_sessions(3);
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let data = in_sessions(&p, &e, &sessions);
    let open_ms = |start, end| {
        calibrate_window(&data, &TimeWindow::new(start, end).unwrap(), &config(), "d")
            .unwrap()
            .open_ms
    };
    assert_eq!(open_ms(0, 3 * DAY_MS), Some(24 * HOUR_MS));
    // Clipped to the window at both ends.
    assert_eq!(
        open_ms(10 * HOUR_MS, DAY_MS + 12 * HOUR_MS),
        Some(7 * HOUR_MS + 3 * HOUR_MS)
    );
    // Sessions were declared, and none was open: zero, not unknown.
    assert_eq!(open_ms(18 * HOUR_MS, 20 * HOUR_MS), Some(0));
    let plain = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &TimeWindow::new(0, DAY_MS).unwrap(),
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(plain.open_ms, None);
}

#[test]
fn test_calibrate_window_at_the_end_of_the_clock_is_not_an_error() {
    // The next intraday bucket does not fit in a u64, with or without sessions.
    let window = TimeWindow::new(u64::MAX - 1_000, u64::MAX).unwrap();
    let cfg = config().with_seasonality(24, 1);
    let plain = CalibrationData::new(&[], &[]).unwrap();
    let insufficient =
        CalibrationOutcome::Insufficient(CalibrationFailure::TooFewPrices { have: 0, need: 30 });
    assert_eq!(
        calibrate_window(&plain, &window, &cfg, "d")
            .unwrap()
            .outcome,
        insufficient
    );
    let sessions = [window];
    let declared = plain.with_sessions(&sessions).unwrap();
    assert_eq!(
        calibrate_window(&declared, &window, &cfg, "d")
            .unwrap()
            .outcome,
        insufficient
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_calibration_record_serde_round_trip_is_identical() {
    let p = prices(7, 0, DAY_MS, 60_000, 0.5);
    let e = exposures(11, 0, DAY_MS, 2.0, 3.0, |_| 1.0);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let record = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    let json = serde_json::to_string(&record).unwrap();
    assert_eq!(
        serde_json::from_str::<CalibrationRecord>(&json).unwrap(),
        record
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_calibration_record_serde_keeps_open_ms_and_reads_older_records() {
    let sessions = day_sessions(1);
    let p = session_prices(&sessions, 0.5);
    let e = session_exposures(&sessions);
    let window = TimeWindow::new(0, DAY_MS).unwrap();
    let declared =
        calibrate_window(&in_sessions(&p, &e, &sessions), &window, &config(), "d").unwrap();
    let json = serde_json::to_string(&declared).unwrap();
    assert_eq!(
        serde_json::from_str::<CalibrationRecord>(&json).unwrap(),
        declared
    );

    // A record from before the field existed reads as having no sessions.
    let plain = calibrate_window(
        &CalibrationData::new(&p, &e).unwrap(),
        &window,
        &config(),
        "d",
    )
    .unwrap();
    let json = serde_json::to_string(&plain).unwrap();
    let older = json.replace(",\"open_ms\":null", "");
    assert_ne!(older, json);
    assert_eq!(
        serde_json::from_str::<CalibrationRecord>(&older).unwrap(),
        plain
    );
}
