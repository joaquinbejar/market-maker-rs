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
