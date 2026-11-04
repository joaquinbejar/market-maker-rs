//! The micro-price against fixtures computed independently of the code.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::analytics::micro_price::*;
use market_maker_rs::dec;
use market_maker_rs::prelude::Decimal;
use market_maker_rs::strategy::walk_forward::{TimeWindow, WindowSpec, chronological_splits};
use rust_decimal::prelude::ToPrimitive;

const TICK: Decimal = dec!(0.01);
const GAP_MS: u64 = 1_000;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn config() -> MicroPriceConfig {
    MicroPriceConfig::new(TICK, 2, 1)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS)
}

/// A one tick quote at `bid_ticks`, in the high (bid heavy) or low bucket.
fn quote(timestamp_ms: u64, bid_ticks: i64, high: bool) -> TopOfBook {
    let bid = Decimal::from(bid_ticks) * TICK;
    TopOfBook {
        timestamp_ms,
        bid,
        ask: bid + TICK,
        bid_size: if high { dec!(3) } else { dec!(1) },
        ask_size: if high { dec!(1) } else { dec!(3) },
    }
}

/// Feeds isolated pairs from the high state: each is too far from the next to
/// form a pair with it.
fn pairs_from_high(cells: &[(usize, i64, bool)]) -> Vec<TopOfBook> {
    let mut quotes = Vec::new();
    let mut time = 0;
    for (count, move_ticks, to_high) in cells {
        for _ in 0..*count {
            quotes.push(quote(time, 1000, true));
            quotes.push(quote(time + 1, 1000 + move_ticks, *to_high));
            time += GAP_MS * 2;
        }
    }
    quotes
}

/// End of the training window of the fixtures.
const TRAIN_END: u64 = 1_000_000_000;

fn record_of(quotes: &[TopOfBook], config: &MicroPriceConfig) -> MicroPriceRecord {
    let window = TimeWindow::new(0, TRAIN_END).unwrap();
    calibrate_micro_price(quotes, &window, config, "d").unwrap()
}

fn model_of(quotes: &[TopOfBook], config: &MicroPriceConfig) -> MicroPriceModel {
    let record = record_of(quotes, config);
    record
        .model()
        .cloned()
        .unwrap_or_else(|| panic!("{:?}", record.fit))
}

fn hand_fixture() -> Vec<TopOfBook> {
    pairs_from_high(&[
        (5, 0, true),  // Q(H, H) = 0.5
        (1, 0, false), // Q(H, L) = 0.1
        (3, 1, false), // T(H, L) = 0.3, up one tick
        (1, -1, true), // T(H, H) = 0.1, down one tick
    ])
}

#[test]
fn test_fit_hand_computed_chain_gives_the_closed_form() {
    // In ticks, with a = (1, -1) on (H, L) by the mirror: R K = 0.2 a,
    // Q a = (0.5 - 0.1) a, so G¹ = 0.2 / 0.6 = 1/3 on H.
    // T a = (0.1 - 0.3) a, so B a = -0.2 / 0.6 a = -1/3 a, and
    // G* = (1/3) / (1 + 1/3) = 1/4 tick.
    let model = model_of(&hand_fixture(), &config());
    let (low, high) = (model.states()[0], model.states()[1]);
    assert_eq!((high.imbalance_bucket, high.spread_ticks), (2, 1));
    assert_eq!(high.observations, 10);
    let in_ticks = |v: Decimal| (v / TICK).to_f64().unwrap();
    assert!((in_ticks(high.first_adjustment) - 1.0 / 3.0).abs() < 1e-9);
    assert!((in_ticks(high.adjustment) - 0.25).abs() < 1e-9);
    assert!((in_ticks(low.adjustment) + 0.25).abs() < 1e-9);
    assert_eq!(high.status, StateStatus::Calibrated);

    let signal = model.signal(&quote(77, 1000, true)).unwrap();
    assert_eq!(signal.timestamp_ms, 77);
    assert_eq!(signal.mid, dec!(10.005));
    assert_eq!(signal.source, SignalSource::MicroPrice);
    assert!((((signal.price - signal.mid) / TICK).to_f64().unwrap() - 0.25).abs() < 1e-9);
}

#[test]
fn test_fit_simulated_chain_long_run_mean_matches_the_adjustment() {
    // Checks the fixture's 1/4 tick, and does not call the crate. From first
    // principles: walk the chain of the hand fixture and average
    // the mid move over many mid changes, starting in the high state.
    let mut rng = Rng(5);
    let mut total = 0.0;
    let runs = 200_000;
    for _ in 0..runs {
        let (mut high, mut price, mut moves) = (true, 0.0f64, 0);
        while moves < 12 {
            let u = rng.unit();
            let sign = if high { 1.0 } else { -1.0 };
            if u < 0.5 {
            } else if u < 0.6 {
                high = !high;
            } else if u < 0.9 {
                price += sign;
                high = !high;
                moves += 1;
            } else {
                price -= sign;
                moves += 1;
            }
        }
        total += price;
    }
    let estimate = total / f64::from(runs);
    assert!((estimate - 0.25).abs() < 0.02, "{estimate}");
}

#[test]
fn test_fit_imbalance_independent_of_moves_gives_the_mid() {
    // Appendix A: moves unrelated to imbalance.
    let model = model_of(
        &pairs_from_high(&[(5, 0, true), (2, 1, true), (2, -1, true), (1, 0, false)]),
        &config(),
    );
    for state in model.states() {
        assert!(state.adjustment.abs() < dec!(0.00000001), "{state:?}");
    }
}

#[test]
fn test_fit_rare_state_is_a_mid_fallback() {
    let mut quotes = hand_fixture();
    let time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    // One pair out of a second spread state, below the minimum of 2.
    let wide = TopOfBook {
        ask: dec!(10.02),
        ..quote(time, 1000, true)
    };
    quotes.push(wide);
    quotes.push(TopOfBook {
        timestamp_ms: time + 1,
        ..wide
    });
    let cfg = MicroPriceConfig::new(TICK, 2, 2)
        .with_min_state_observations(3)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    let rare = model.states()[3];
    assert_eq!((rare.imbalance_bucket, rare.spread_ticks), (2, 2));
    assert_eq!(rare.status, StateStatus::Rare);
    assert_eq!(rare.adjustment, dec!(0));
    let signal = model.signal(&wide).unwrap();
    assert_eq!(signal.price, signal.mid);
    assert_eq!(signal.source, SignalSource::MidFallback(StateStatus::Rare));
}

#[test]
fn test_fit_state_without_moves_is_a_mid_fallback_not_singular() {
    let mut quotes = hand_fixture();
    let time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    let wide = TopOfBook {
        ask: dec!(10.02),
        ..quote(time, 1000, true)
    };
    quotes.push(wide);
    quotes.push(TopOfBook {
        timestamp_ms: time + 1,
        ..wide
    });
    let cfg = MicroPriceConfig::new(TICK, 2, 2)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    assert_eq!(model.states()[3].status, StateStatus::NoPriceMove);
    assert_eq!(model.states()[2].status, StateStatus::NoPriceMove);
    assert_eq!(model.states()[1].status, StateStatus::Calibrated);
}

#[test]
fn test_fit_no_data_or_no_moves_is_insufficient() {
    let window = TimeWindow::new(0, u64::MAX).unwrap();
    let fit = |q: &[TopOfBook]| {
        calibrate_micro_price(q, &window, &config(), "d")
            .unwrap()
            .fit
    };
    let none = MicroPriceFit::Insufficient(MicroPriceFailure::NoCalibratedState);
    assert_eq!(fit(&[]), none);
    assert_eq!(fit(&pairs_from_high(&[(5, 0, true)])), none);
}

#[test]
fn test_fit_drifting_closed_class_does_not_converge() {
    // Only up moves that stay in the high state: the mirror drifts down, and
    // neither reaches the other, so theorem 3.1's condition fails.
    let record = calibrate_micro_price(
        &pairs_from_high(&[(10, 1, true)]),
        &TimeWindow::new(0, TRAIN_END).unwrap(),
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(
        record.fit,
        MicroPriceFit::Insufficient(MicroPriceFailure::NotConverged)
    );
}

#[test]
fn test_fit_limit_outside_the_quotes_is_a_mid_fallback() {
    // At one tick, large moves against small ones; at two ticks, the fixture.
    let mut quotes = pairs_from_high(&[(5, 0, true), (1, 5, false), (1, 0, false)]);
    let mut time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    let wide = |t, bid_ticks, high| {
        let q = quote(t, bid_ticks, high);
        TopOfBook {
            ask: q.ask + TICK,
            ..q
        }
    };
    for (count, move_ticks, to_high) in [(5, 0, true), (1, 0, false), (3, 1, false), (1, -1, true)]
    {
        for _ in 0..count {
            quotes.push(wide(time, 1000, true));
            quotes.push(wide(time + 1, 1000 + move_ticks, to_high));
            time += 2 * GAP_MS;
        }
    }
    let cfg = MicroPriceConfig::new(TICK, 2, 2)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    let (narrow, broad) = (model.states()[1], model.states()[3]);
    assert_eq!(narrow.status, StateStatus::OutsideQuotes, "{narrow:?}");
    assert!(narrow.adjustment.abs() > TICK / dec!(2));
    assert_eq!(broad.status, StateStatus::Calibrated, "{broad:?}");
    let signal = model.signal(&quote(0, 1000, true)).unwrap();
    assert_eq!(signal.price, signal.mid);
    assert_eq!(
        signal.source,
        SignalSource::MidFallback(StateStatus::OutsideQuotes)
    );
}

#[test]
fn test_signal_stays_within_the_quotes_and_never_invalid() {
    let model = model_of(&simulated(9, 40_000), &MicroPriceConfig::new(TICK, 5, 2));
    for q in simulated(10, 2_000) {
        let signal = model.signal(&q).unwrap();
        assert!(signal.price >= q.bid && signal.price <= q.ask, "{signal:?}");
    }
}

#[test]
fn test_observe_invalid_quotes_are_refused_and_leave_no_trace() {
    let mut calibrator = MicroPriceCalibrator::new(config()).unwrap();
    calibrator.observe(&quote(10, 1000, true)).unwrap();
    let before = calibrator.clone();
    let good = quote(20, 1000, true);
    for bad in [
        TopOfBook {
            bid: dec!(0),
            ..good
        },
        TopOfBook {
            ask: good.bid,
            ..good
        },
        TopOfBook {
            bid_size: dec!(-1),
            ..good
        },
        TopOfBook {
            bid_size: dec!(0),
            ask_size: dec!(0),
            ..good
        },
        TopOfBook {
            ask: good.ask + dec!(0.001),
            ..good
        },
        quote(9, 1000, true),
    ] {
        assert!(calibrator.observe(&bad).is_err(), "{bad:?}");
        assert_eq!(calibrator, before);
    }
}

#[test]
fn test_config_invalid_values_are_refused() {
    let good = MicroPriceConfig::new(TICK, 5, 2);
    assert!(good.validate().is_ok());
    for bad in [
        MicroPriceConfig::new(dec!(0), 5, 2),
        MicroPriceConfig::new(TICK, 1, 2),
        MicroPriceConfig::new(TICK, 5, 0),
        MicroPriceConfig::new(TICK, 17, 16),
        good.with_min_state_observations(0),
        good.with_max_quote_gap_ms(0),
    ] {
        assert!(MicroPriceCalibrator::new(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn test_imbalance_extremes_have_buckets() {
    let cfg = MicroPriceConfig::new(TICK, 4, 1).with_min_state_observations(1);
    let model = model_of(&simulated(3, 20_000), &cfg);
    let at = |bid_size, ask_size| TopOfBook {
        bid_size,
        ask_size,
        ..quote(1, 1000, true)
    };
    assert!(model.signal(&at(dec!(0), dec!(1))).unwrap().adjustment < dec!(0));
    assert!(model.signal(&at(dec!(1), dec!(0))).unwrap().adjustment > dec!(0));
}

/// Quotes from a chain whose size imbalance predicts the next move.
fn simulated(seed: u64, count: u64) -> Vec<TopOfBook> {
    let mut rng = Rng(seed);
    let (mut bid, mut imbalance) = (1000i64, 0.5f64);
    let mut out = Vec::new();
    for t in 0..count {
        let bid_size = (imbalance * 100.0).round().clamp(1.0, 99.0);
        out.push(TopOfBook {
            timestamp_ms: t * 100,
            bid: Decimal::from(bid) * TICK,
            ask: Decimal::from(bid + 1) * TICK,
            bid_size: Decimal::from(bid_size as u64),
            ask_size: Decimal::from(100 - bid_size as u64),
        });
        if rng.unit() < 0.1 {
            if rng.unit() < imbalance {
                bid += 1;
                imbalance = 0.3 + 0.2 * rng.unit();
            } else {
                bid -= 1;
                imbalance = 0.5 + 0.2 * rng.unit();
            }
        } else {
            imbalance = (imbalance + 0.08 * (rng.unit() - 0.5)).clamp(0.01, 0.99);
        }
    }
    out
}

#[test]
fn test_calibrate_changing_later_quotes_leaves_model_and_signal_unchanged() {
    let quotes = simulated(1, 20_000);
    let window = TimeWindow::new(0, 1_000_000).unwrap();
    let cfg = MicroPriceConfig::new(TICK, 5, 2);
    let before = calibrate_micro_price(&quotes, &window, &cfg, "d").unwrap();
    let mut changed = quotes.clone();
    changed
        .iter_mut()
        .filter(|q| q.timestamp_ms >= window.end_ms)
        .for_each(|q| q.bid_size += dec!(50));
    let after = calibrate_micro_price(&changed, &window, &cfg, "d").unwrap();
    assert_eq!(before, after);
    let (a, b) = (before.model().unwrap(), after.model().unwrap());
    for q in quotes.iter().take(9_000) {
        assert_eq!(a.signal(q).unwrap(), b.signal(q).unwrap());
    }

    let mut inside = quotes.clone();
    inside[5].bid_size += dec!(1);
    let moved = calibrate_micro_price(&inside, &window, &cfg, "d").unwrap();
    assert_ne!(before.dataset_hash, moved.dataset_hash);
}

#[test]
fn test_calibrator_restored_midway_gives_the_same_fit() {
    let quotes = simulated(2, 5_000);
    let cfg = MicroPriceConfig::new(TICK, 5, 2);
    let mut whole = MicroPriceCalibrator::new(cfg).unwrap();
    let mut first = MicroPriceCalibrator::new(cfg).unwrap();
    for q in &quotes {
        whole.observe(q).unwrap();
    }
    for q in &quotes[..2_500] {
        first.observe(q).unwrap();
    }
    let mut restored = first.clone();
    for q in &quotes[2_500..] {
        restored.observe(q).unwrap();
    }
    assert_eq!(whole, restored);
    assert_eq!(whole.fit().unwrap(), restored.fit().unwrap());
}

#[cfg(feature = "serde")]
#[test]
fn test_calibrator_serde_round_trip_continues_identically() {
    let quotes = simulated(2, 1_000);
    let mut calibrator = MicroPriceCalibrator::new(MicroPriceConfig::new(TICK, 5, 2)).unwrap();
    for q in &quotes[..500] {
        calibrator.observe(q).unwrap();
    }
    let json = serde_json::to_string(&calibrator).unwrap();
    let restored: MicroPriceCalibrator = serde_json::from_str(&json).unwrap();
    assert_eq!(calibrator, restored);
}

#[test]
fn test_compare_hand_computed_errors_for_all_estimators() {
    let record = record_of(&hand_fixture(), &config());
    let t = TRAIN_END;
    let sized = |q: TopOfBook, bid_size, ask_size| TopOfBook {
        bid_size,
        ask_size,
        ..q
    };
    let quotes = [
        // Mid 10.005, weighted 10.009, micro 10.0075; high bucket.
        sized(quote(t, 1000, true), dec!(9), dec!(1)),
        // Mid 10.015, weighted and micro 10.0125; low bucket.
        quote(t + 100, 1001, false),
        // I = 1/2 is in the low bucket and has no side. Mid and weighted
        // 10.015, micro 10.0125.
        sized(quote(t + 200, 1001, true), dec!(1), dec!(1)),
        // Two ticks is out of range. Mid and micro 10.03, weighted 10.025.
        TopOfBook {
            ask: dec!(10.04),
            ..quote(t + 300, 1002, false)
        },
        // Mid 10.025. Its own horizon is past the window.
        quote(t + 400, 1002, true),
    ];
    let window = TimeWindow::new(t, t + 401).unwrap();
    let report = compare_estimators(&record, &quotes, &window, &[100]).unwrap();
    let horizon = &report[0];
    assert_eq!((horizon.samples, horizon.unscored_samples), (4, 1));
    assert_eq!((horizon.fallback_samples, horizon.invalid_quotes), (1, 0));

    // future mid - estimator, for samples 0 to 3; sides are +, -, none, -.
    let check = |got: EstimatorError, errors: [f64; 4]| {
        let close = |v: Decimal, want: f64| {
            assert!((v.to_f64().unwrap() - want).abs() < 1e-9, "{v} {want}");
        };
        close(
            got.rmse,
            (errors.iter().map(|e| e * e).sum::<f64>() / 4.0).sqrt(),
        );
        close(
            got.mean_absolute,
            errors.iter().map(|e| e.abs()).sum::<f64>() / 4.0,
        );
        close(
            got.directional_markout.unwrap(),
            (errors[0] - errors[1] - errors[3]) / 3.0,
        );
    };
    let set = horizon.overall.unwrap();
    check(set.mid, [0.01, 0.0, 0.015, -0.005]);
    check(set.weighted_mid, [0.006, 0.0025, 0.015, 0.0]);
    check(set.micro_price, [0.0075, 0.0025, 0.0175, -0.005]);

    // The out of range quote counts only overall.
    let cells: Vec<_> = horizon
        .by_state
        .iter()
        .map(|s| (s.imbalance_bucket, s.spread_ticks, s.samples))
        .collect();
    assert_eq!(cells, [(1, 1, 2), (2, 1, 1)]);
}

#[test]
fn test_compare_window_not_after_training_is_refused() {
    let record = record_of(&hand_fixture(), &config());
    let inside = TimeWindow::new(TRAIN_END - 1, TRAIN_END + 100).unwrap();
    assert!(compare_estimators(&record, &[], &inside, &[1]).is_err());
    let after = TimeWindow::new(TRAIN_END, TRAIN_END + 100).unwrap();
    assert!(compare_estimators(&record, &[], &after, &[1]).is_ok());
    let empty = record_of(&[], &config());
    assert!(compare_estimators(&empty, &[], &after, &[1]).is_err());
}

#[test]
fn test_invalid_quote_is_counted_not_fatal() {
    let mut quotes = hand_fixture();
    let crossed = TopOfBook {
        ask: dec!(9),
        ..quote(1, 1000, true)
    };
    quotes.insert(1, crossed);
    let clean = record_of(&hand_fixture(), &config());
    let record = record_of(&quotes, &config());
    assert_eq!(record.invalid_quotes, 1);
    assert_eq!(record.fit, clean.fit);
    assert_eq!(record.dataset_hash, clean.dataset_hash);

    let t = TRAIN_END;
    let test = [
        quote(t, 1000, true),
        TopOfBook {
            ask: dec!(9),
            ..quote(t + 50, 1000, true)
        },
        quote(t + 100, 1001, true),
    ];
    let window = TimeWindow::new(t, t + 1_000).unwrap();
    let report = compare_estimators(&clean, &test, &window, &[60]).unwrap();
    assert_eq!(report[0].invalid_quotes, 1);
    // The sample at t reads its future from the quote before the bad one.
    let set = report[0].overall.unwrap();
    assert_eq!(report[0].samples, 2);
    assert_eq!(set.mid.rmse, dec!(0));
}

#[test]
fn test_calibrate_quotes_before_the_window_are_ignored() {
    let quotes = hand_fixture();
    let shifted: Vec<_> = quotes
        .iter()
        .map(|q| TopOfBook {
            timestamp_ms: q.timestamp_ms + 5_000,
            ..*q
        })
        .collect();
    let mut with_earlier = vec![quote(0, 500, false), quote(1, 600, true)];
    with_earlier.extend(shifted.iter().copied());
    let window = TimeWindow::new(5_000, u64::MAX).unwrap();
    let plain = calibrate_micro_price(&shifted, &window, &config(), "d").unwrap();
    let earlier = calibrate_micro_price(&with_earlier, &window, &config(), "d").unwrap();
    assert_eq!(plain, earlier);
}

#[test]
fn test_compare_reports_every_horizon_in_the_order_given() {
    let quotes = simulated(4, 20_000);
    let split = chronological_splits(
        0,
        2_000_000,
        &WindowSpec {
            train_ms: 1_000_000,
            test_ms: 1_000_000,
            step_ms: 1_000_000,
        },
    )
    .unwrap()[0];
    let cfg = MicroPriceConfig::new(TICK, 5, 2);
    let record = calibrate_micro_price(&quotes, &split.train, &cfg, "d").unwrap();
    let report = compare_estimators(&record, &quotes, &split.test, &[5_000, 500, 60_000]).unwrap();
    assert_eq!(
        report.iter().map(|h| h.horizon_ms).collect::<Vec<_>>(),
        [5_000, 500, 60_000]
    );
    for horizon in &report {
        let set = horizon.overall.unwrap();
        let (micro, mid) = (set.micro_price.rmse, set.mid.rmse);
        println!(
            "horizon_ms={} samples={} fallback={} unscored={} rmse mid={mid} weighted={} micro={micro} markout mid={:?} weighted={:?} micro={:?}",
            horizon.horizon_ms,
            horizon.samples,
            horizon.fallback_samples,
            horizon.unscored_samples,
            set.weighted_mid.rmse,
            set.mid.directional_markout,
            set.weighted_mid.directional_markout,
            set.micro_price.directional_markout,
        );
        assert_eq!(horizon.samples + horizon.unscored_samples, 10_000);
        let in_states: u64 = horizon.by_state.iter().map(|s| s.samples).sum();
        assert_eq!(in_states, horizon.samples);
    }
    assert!(report[0].unscored_samples > 0);
    // The 500 ms horizon is where this synthetic signal should show; longer
    // ones are dominated by noise and are reported as they come.
    let short = report[1].overall.unwrap();
    assert!(short.micro_price.rmse < short.mid.rmse);
    for state in &report[0].by_state {
        println!(
            "imbalance_bucket={} spread={} samples={} rmse mid={} weighted={} micro={}",
            state.imbalance_bucket,
            state.spread_ticks,
            state.samples,
            state.errors.mid.rmse.round_dp(6),
            state.errors.weighted_mid.rmse.round_dp(6),
            state.errors.micro_price.rmse.round_dp(6),
        );
    }
}

#[test]
fn test_compare_future_quote_past_gap_or_window_is_unscored() {
    let record = record_of(&hand_fixture(), &config());
    let t = TRAIN_END;
    let window = TimeWindow::new(t, t + 100_000).unwrap();
    let quotes = [quote(t, 1000, true), quote(t + 90_000, 1001, true)];
    let report = compare_estimators(&record, &quotes, &window, &[5_000, 200_000]).unwrap();
    assert_eq!((report[0].samples, report[0].unscored_samples), (0, 2));
    assert_eq!((report[1].samples, report[1].unscored_samples), (0, 2));
    assert!(report[0].overall.is_none());
}

#[test]
fn test_compare_invalid_horizons_are_refused() {
    let model = record_of(&hand_fixture(), &config());
    let window = TimeWindow::new(TRAIN_END, TRAIN_END + 100).unwrap();
    assert!(compare_estimators(&model, &[], &window, &[]).is_err());
    assert!(compare_estimators(&model, &[], &window, &[0]).is_err());
    assert!(compare_estimators(&model, &[], &window, &[1; 65]).is_err());
    let q = [
        quote(TRAIN_END + 5, 1000, true),
        quote(TRAIN_END + 1, 1000, true),
    ];
    assert!(compare_estimators(&model, &q, &window, &[1]).is_err());
}

#[test]
fn test_weighted_mid_paper_example() {
    // Section 1: 9 x 10.00 / 10.02 x 27 gives 10.005.
    let q = TopOfBook {
        timestamp_ms: 0,
        bid: dec!(10.00),
        ask: dec!(10.02),
        bid_size: dec!(9),
        ask_size: dec!(27),
    };
    assert_eq!(q.weighted_mid().unwrap(), dec!(10.005));
}
