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
use market_maker_rs::types::error::MMError;
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
    MicroPriceConfig::new(TICK, 3, 1)
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
    let (low, high) = (model.states()[0], model.states()[2]);
    assert_eq!((high.imbalance_bucket, high.spread_ticks), (3, 1));
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
    let cfg = MicroPriceConfig::new(TICK, 3, 2)
        .with_min_state_observations(3)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    let rare = model.states()[5];
    assert_eq!((rare.imbalance_bucket, rare.spread_ticks), (3, 2));
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
    let cfg = MicroPriceConfig::new(TICK, 3, 2)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    assert_eq!(model.states()[5].status, StateStatus::NoPriceMove);
    assert_eq!(model.states()[3].status, StateStatus::NoPriceMove);
    assert_eq!(model.states()[2].status, StateStatus::Calibrated);
}

#[test]
fn test_fit_reaching_only_a_rare_state_is_not_calibrated() {
    // Ten isolated pairs from a one tick to a three tick quote, with the same
    // mid: no mid move anywhere in the data.
    let mut quotes = Vec::new();
    for i in 0..10 {
        let narrow = quote(i * 3_000, 1000, true);
        quotes.push(narrow);
        quotes.push(TopOfBook {
            timestamp_ms: i * 3_000 + 1,
            bid: dec!(9.99),
            ask: dec!(10.02),
            ..narrow
        });
    }
    let cfg = MicroPriceConfig::new(TICK, 3, 3)
        .with_min_state_observations(3)
        .with_max_quote_gap_ms(GAP_MS);
    assert_eq!(
        record_of(&quotes, &cfg).fit,
        MicroPriceFit::Insufficient(MicroPriceFailure::NoCalibratedState)
    );
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
    let cfg = MicroPriceConfig::new(TICK, 3, 2)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS);
    let model = model_of(&quotes, &cfg);
    let (narrow, broad) = (model.states()[2], model.states()[5]);
    assert_eq!(narrow.status, StateStatus::OutsideQuotes, "{narrow:?}");
    // It does not absorb in the matrices, so it is not counted as absorption.
    assert_eq!(narrow.fallback_within_moves, dec!(0));
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
        MicroPriceConfig::new(TICK, 4, 2),
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
    let cfg = MicroPriceConfig::new(TICK, 5, 1).with_min_state_observations(1);
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
        // I = 1/2 is in the middle bucket, with no data and no side. Mid, weighted
        // and micro 10.015.
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
    assert_eq!((horizon.fallback_samples, horizon.invalid_quotes), (2, 0));

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
    check(set.micro_price, [0.0075, 0.0025, 0.015, -0.005]);

    // The out of range quote counts only overall.
    let cells: Vec<_> = horizon
        .by_state
        .iter()
        .map(|s| (s.imbalance_bucket, s.spread_ticks, s.samples))
        .collect();
    assert_eq!(cells, [(1, 1, 1), (2, 1, 1), (3, 1, 1)]);
}

#[test]
fn test_compare_high_price_level_keeps_the_errors() {
    let record = record_of(&hand_fixture(), &config());
    let t = TRAIN_END;
    // At 2e16 an f64 cannot tell these mids apart.
    let high = |q: TopOfBook| TopOfBook {
        bid: q.bid + dec!(20000000000000000),
        ask: q.ask + dec!(20000000000000000),
        ..q
    };
    let quotes = [high(quote(t, 1000, true)), high(quote(t + 100, 1001, true))];
    let window = TimeWindow::new(t, t + 101).unwrap();
    let report = compare_estimators(&record, &quotes, &window, &[100]).unwrap();
    let rmse = report[0].overall.unwrap().mid.rmse.to_f64().unwrap();
    assert!((rmse - 0.01).abs() < 1e-12, "{rmse}");
}

#[test]
fn test_signal_is_antisymmetric_for_all_small_size_pairs() {
    for buckets in [3, 5, 7, 11] {
        let cfg = MicroPriceConfig::new(TICK, buckets, 1).with_min_state_observations(1);
        let model = model_of(&simulated(3, 40_000), &cfg);
        for bid_size in 0..12u32 {
            for ask_size in 0..12u32 {
                if bid_size + ask_size == 0 {
                    continue;
                }
                let at = |bid: u32, ask: u32| TopOfBook {
                    bid_size: Decimal::from(bid),
                    ask_size: Decimal::from(ask),
                    ..quote(1, 1000, true)
                };
                let one = model.signal(&at(bid_size, ask_size)).unwrap();
                let two = model.signal(&at(ask_size, bid_size)).unwrap();
                assert_eq!(one.adjustment, -two.adjustment, "{bid_size}:{ask_size}");
                if bid_size == ask_size {
                    assert_eq!(one.adjustment, dec!(0));
                }
            }
        }
    }
}

#[test]
fn test_bucket_boundaries_follow_the_exact_rule() {
    // n = 3: I <= 1/3 is bucket 1, without rounding 1/3 in a Decimal, and its
    // mirror I >= 2/3 is bucket 3; anything between is bucket 2.
    let cfg = MicroPriceConfig::new(TICK, 3, 1).with_min_state_observations(1);
    let mut quotes = Vec::new();
    for (i, (bid, ask)) in [(1, 2), (2, 1), (2, 3), (3, 2), (1, 1)].iter().enumerate() {
        let t = i as u64 * 200_000;
        for offset in [0, 1] {
            quotes.push(TopOfBook {
                bid_size: Decimal::from(*bid),
                ask_size: Decimal::from(*ask),
                ..quote(t + offset, 1000, true)
            });
        }
    }
    let record = record_of(&quotes, &cfg);
    let MicroPriceFit::Insufficient(failure) = record.fit else {
        panic!("no mid moves in this data");
    };
    assert_eq!(failure, MicroPriceFailure::NoCalibratedState);
    let mut calibrator = MicroPriceCalibrator::new(cfg).unwrap();
    for q in &quotes {
        calibrator.observe(q).unwrap();
    }
    // Add one move out of each bucket, so that the table is reported.
    let mut moving = quotes.clone();
    for (i, (bid, ask)) in [(1, 2), (2, 3), (2, 1)].iter().enumerate() {
        let t = 2_000_000 + i as u64 * 200_000;
        moving.push(TopOfBook {
            bid_size: Decimal::from(*bid),
            ask_size: Decimal::from(*ask),
            ..quote(t, 1000, true)
        });
        moving.push(quote(t + 1, 1001, false));
    }
    let model = model_of(&moving, &cfg);
    let observed: Vec<u64> = model.states().iter().map(|s| s.observations).collect();
    // Unmoved pairs from 1:2 and 2:1, mirrored, give 2 each in buckets 1 and 3;
    // from 2:3, 3:2 and 1:1, mirrored, 6 in bucket 2. The moves add 2, 2 and 2.
    assert_eq!(observed, [4, 8, 4]);
}

#[test]
fn test_fit_large_move_is_kept_and_reported() {
    let clean = simulated(6, 20_000);
    let mut printed = clean.clone();
    printed[10_000].bid += dec!(10);
    printed[10_000].ask += dec!(10);
    let cfg = MicroPriceConfig::new(TICK, 5, 2);
    let (a, b) = (record_of(&clean, &cfg), record_of(&printed, &cfg));
    assert_eq!(a.largest_mid_move_ticks, dec!(1));
    assert_eq!(b.largest_mid_move_ticks, dec!(1000));
    assert_eq!(b.pairs, a.pairs);
    // Not filtered: the print does move the fit.
    assert_ne!(
        a.model().unwrap().states()[0].adjustment,
        b.model().unwrap().states()[0].adjustment
    );
}

#[test]
fn test_calibrate_unrepresentable_quote_is_a_typed_error() {
    let good = quote(1, 1000, true);
    for bad in [
        TopOfBook {
            ask: Decimal::MAX,
            ..good
        },
        TopOfBook {
            bid: dec!(1E28),
            ask: dec!(1.1E28),
            ..good
        },
        TopOfBook {
            bid: dec!(1E17),
            ask: dec!(1E17) + TICK,
            ..good
        },
        TopOfBook {
            bid_size: Decimal::MAX,
            ask_size: Decimal::MAX,
            ..good
        },
    ] {
        let window = TimeWindow::new(0, TRAIN_END).unwrap();
        let result = calibrate_micro_price(&[bad], &window, &config(), "d");
        assert!(matches!(result, Err(MMError::NumericalError(_))), "{bad:?}");
        let record = record_of(&hand_fixture(), &config());
        let test = [TopOfBook {
            timestamp_ms: TRAIN_END,
            ..bad
        }];
        let test_window = TimeWindow::new(TRAIN_END, TRAIN_END + 10).unwrap();
        let result = compare_estimators(&record, &test, &test_window, &[1]);
        assert!(matches!(result, Err(MMError::NumericalError(_))), "{bad:?}");
    }
}

#[test]
fn test_record_reports_coverage() {
    let mut quotes = hand_fixture();
    let time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    let balanced = TopOfBook {
        bid_size: dec!(1),
        ask_size: dec!(1),
        ..quote(time, 1000, true)
    };
    quotes.push(balanced);
    quotes.push(TopOfBook {
        timestamp_ms: time + 1,
        ..balanced
    });
    let record = record_of(&quotes, &config().with_min_state_observations(3));
    // 10 from each outer bucket are calibrated; 2 from the middle one are rare.
    let share = record.model().unwrap().calibrated_share();
    assert_eq!(share.round_dp(12), (dec!(20) / dec!(22)).round_dp(12));
}

#[test]
fn test_fallback_within_moves_reports_absorption() {
    let model = model_of(&hand_fixture(), &config());
    assert_eq!(model.states()[2].fallback_within_moves, dec!(0));
    // The middle bucket has no data, so it is a fallback.
    assert_eq!(model.states()[1].fallback_within_moves, dec!(1));

    // The high state also goes to a rare middle state, one pair in eleven.
    let mut quotes = hand_fixture();
    let time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    quotes.push(quote(time, 1000, true));
    quotes.push(TopOfBook {
        bid_size: dec!(1),
        ask_size: dec!(1),
        ..quote(time + 1, 1000, true)
    });
    let model = model_of(&quotes, &config().with_min_state_observations(3));
    let high = model.states()[2];
    assert_eq!(high.status, StateStatus::Calibrated);
    assert_eq!(model.states()[1].status, StateStatus::Rare);
    assert!(high.fallback_within_moves > dec!(0.1), "{high:?}");
    assert!(high.fallback_within_moves < dec!(1), "{high:?}");
}

#[test]
fn test_half_spread_boundary_stays_calibrated() {
    // In eighths, so that f64 is exact: stay 5, to L unmoved 1, up a tick to L 1
    // and to H 1. T cancels on the antisymmetric vector, so G* = G¹ =
    // (1/4) / (1 - (5/8 - 1/8)) = half a tick, still inside the quote.
    let model = model_of(
        &pairs_from_high(&[(5, 0, true), (1, 0, false), (1, 1, false), (1, 1, true)]),
        &config(),
    );
    let high = model.states()[2];
    assert_eq!(high.adjustment, TICK / dec!(2));
    assert_eq!(high.status, StateStatus::Calibrated);
}

#[test]
fn test_dataset_hash_covers_every_field_of_a_quote() {
    let quotes = hand_fixture();
    let base = record_of(&quotes, &config()).dataset_hash;
    let edits: [fn(&mut TopOfBook); 5] = [
        |q| q.timestamp_ms += 1,
        |q| q.bid -= TICK,
        |q| q.ask += TICK,
        |q| q.bid_size += dec!(1),
        |q| q.ask_size += dec!(1),
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut changed = quotes.clone();
        edit(&mut changed[0]);
        assert_ne!(
            record_of(&changed, &config()).dataset_hash,
            base,
            "field {i}"
        );
    }
}

#[test]
fn test_skipped_pairs_counts_gap_and_spread() {
    let wide = TopOfBook {
        ask: dec!(10.05),
        ..quote(10, 1000, true)
    };
    let quotes = [
        quote(0, 1000, true),
        wide,
        quote(20, 1000, true),
        quote(20 + GAP_MS + 1, 1000, true),
        quote(20 + GAP_MS + 2, 1000, true),
    ];
    let record = record_of(&quotes, &config());
    assert_eq!(record.pairs, 1);
    assert_eq!(record.skipped_spread_pairs, 2);
    assert_eq!(record.skipped_gap_pairs, 1);
}

#[test]
fn test_compare_horizon_at_the_window_end_is_unscored() {
    let record = record_of(&hand_fixture(), &config());
    let t = TRAIN_END;
    let window = TimeWindow::new(t, t + 100).unwrap();
    let report = compare_estimators(&record, &[quote(t, 1000, true)], &window, &[100]).unwrap();
    assert_eq!((report[0].samples, report[0].unscored_samples), (0, 1));
}

#[cfg(feature = "serde")]
#[test]
fn test_decode_invalid_state_is_refused() {
    let calibrator = MicroPriceCalibrator::new(config()).unwrap();
    let json = serde_json::to_string(&calibrator).unwrap();
    assert!(serde_json::from_str::<MicroPriceCalibrator>(&json).is_ok());
    for (from, to) in [
        ("\"states\":3", "\"states\":4000000000000000000"),
        ("\"states\":3", "\"states\":2"),
        ("\"tick_size\":\"0.01\"", "\"tick_size\":\"0\""),
        ("\"tick_size\":\"0.01\"", "\"tick_size\":\"-0.01\""),
        ("\"move_half_ticks\":[0,", "\"move_half_ticks\":["),
    ] {
        assert!(json.contains(from), "{from}");
        let bad = json.replace(from, to);
        assert!(
            serde_json::from_str::<MicroPriceCalibrator>(&bad).is_err(),
            "{to}"
        );
    }

    let model = model_of(&hand_fixture(), &config());
    let json = serde_json::to_string(&model).unwrap();
    assert_eq!(
        serde_json::from_str::<MicroPriceModel>(&json).unwrap(),
        model
    );
    let adjustment = model.states()[2].adjustment.to_string();
    let bad = json.replace(&adjustment, "1000");
    assert!(serde_json::from_str::<MicroPriceModel>(&bad).is_err());
}

#[test]
fn test_fallback_within_moves_matches_the_closed_form() {
    // From H, per pair: stay 1, to a rare middle state 1, up a tick to L 3, and
    // the mirror from L. Before each move the chain is absorbed with chance
    // a = 1/4, so within 10 moves it is 1 - (3/4)^11.
    let mut quotes = pairs_from_high(&[(1, 0, true), (3, 1, false)]);
    let time = quotes.last().unwrap().timestamp_ms + 2 * GAP_MS;
    quotes.push(quote(time, 1000, true));
    quotes.push(TopOfBook {
        bid_size: dec!(1),
        ask_size: dec!(1),
        ..quote(time + 1, 1000, true)
    });
    let model = model_of(&quotes, &config());
    let got = model.states()[2].fallback_within_moves.to_f64().unwrap();
    let want = 1.0 - 0.75f64.powi(11);
    assert!((got - want).abs() < 1e-12, "{got} {want}");
}

#[test]
fn test_first_adjustment_is_exactly_antisymmetric() {
    for buckets in [3, 5, 7, 11] {
        let cfg = MicroPriceConfig::new(TICK, buckets, 2).with_min_state_observations(1);
        let model = model_of(&simulated(3, 40_000), &cfg);
        let n = buckets as usize;
        for row in model.states().chunks(n) {
            for i in 0..n {
                assert_eq!(row[i].first_adjustment, -row[n - 1 - i].first_adjustment);
            }
        }
    }
}

#[test]
fn test_largest_mid_move_counts_downward_moves() {
    let record = record_of(
        &pairs_from_high(&[(5, 0, true), (2, 1, false), (1, -7, false)]),
        &config(),
    );
    assert_eq!(record.largest_mid_move_ticks, dec!(7));
}

#[test]
fn test_calibrated_share_excludes_every_fallback_status() {
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
    let cfg = MicroPriceConfig::new(TICK, 3, 2)
        .with_min_state_observations(1)
        .with_max_quote_gap_ms(GAP_MS);
    // 7 pairs per outer one tick state are OutsideQuotes, 10 per two tick one are not.
    let share = model_of(&quotes, &cfg).calibrated_share();
    assert_eq!(share.round_dp(12), (dec!(20) / dec!(34)).round_dp(12));
}

#[cfg(feature = "serde")]
mod decoding {
    use super::*;
    use serde_json::{Value, json};

    fn model_json() -> Value {
        let text = serde_json::to_string(&model_of(&hand_fixture(), &config())).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn calibrator_json() -> Value {
        let mut calibrator = MicroPriceCalibrator::new(config()).unwrap();
        for q in &hand_fixture() {
            calibrator.observe(q).unwrap();
        }
        let text = serde_json::to_string(&calibrator).unwrap();
        let mut value: Value = serde_json::from_str(&text).unwrap();
        value["hash"] = json!(0);
        value
    }

    fn model_ok(value: &Value) -> bool {
        serde_json::from_value::<MicroPriceModel>(value.clone()).is_ok()
    }

    fn calibrator_ok(value: &Value) -> bool {
        serde_json::from_value::<MicroPriceCalibrator>(value.clone()).is_ok()
    }

    #[test]
    fn test_decode_model_refuses_inconsistent_states() {
        let base = model_json();
        assert!(model_ok(&base));
        let edits: [fn(&mut Value); 12] = [
            |v| {
                v["states"].as_array_mut().unwrap().pop();
            },
            |v| v["states"][0]["imbalance_bucket"] = json!(2),
            |v| v["states"][0]["spread_ticks"] = json!(2),
            |v| v["states"][0]["adjustment"] = json!("0.0025"),
            |v| v["states"][0]["first_adjustment"] = json!("0.0025"),
            |v| v["states"][0]["fallback_within_moves"] = json!("5"),
            |v| v["states"][0]["fallback_within_moves"] = json!("-1"),
            |v| v["states"][1]["fallback_within_moves"] = json!("0.5"),
            |v| v["states"][1]["adjustment"] = json!("0.001"),
            |v| v["states"][0]["status"] = json!("OutsideQuotes"),
            |v| v["states"][0]["observations"] = json!(1),
            // OutsideQuotes inside the quotes, on both mirrors.
            |v| {
                v["states"][0]["status"] = json!("OutsideQuotes");
                v["states"][2]["status"] = json!("OutsideQuotes");
            },
        ];
        for (i, edit) in edits.iter().enumerate() {
            let mut bad = base.clone();
            edit(&mut bad);
            assert!(!model_ok(&bad), "edit {i}");
        }
        // A whole extra row, consistent in itself, for a spread not configured.
        let mut bad = base.clone();
        let states = bad["states"].as_array_mut().unwrap();
        let row: Vec<Value> = states
            .iter()
            .cloned()
            .map(|mut state| {
                state["spread_ticks"] = json!(2);
                state
            })
            .collect();
        states.extend(row);
        assert!(!model_ok(&bad));
    }

    #[test]
    fn test_decode_calibrator_refuses_broken_counts() {
        let base = calibrator_json();
        assert!(calibrator_ok(&base));
        let edits: [fn(&mut Value); 6] = [
            // Moves a count between two cells, so that the sum still holds.
            |v| {
                v["stay"][2 * 3 + 2] = json!(4);
                v["stay"][2 * 3] = json!(2);
            },
            |v| v["moved"][2 * 3 + 2] = json!(50),
            |v| v["stay"][2 * 3 + 2] = json!(50),
            |v| v["move_half_ticks"][2] = json!(50),
            |v| v["pairs"] = json!(1),
            |v| {
                v["previous"] = json!({"timestamp_ms": 1, "bid": "2", "ask": "1", "bid_size": "1", "ask_size": "1"});
            },
        ];
        for (i, edit) in edits.iter().enumerate() {
            let mut bad = base.clone();
            edit(&mut bad);
            assert!(!calibrator_ok(&bad), "edit {i}");
        }
    }

    #[test]
    fn test_decode_config_refuses_invalid_values() {
        let good = json!({
            "tick_size": "0.01",
            "imbalance_buckets": 3,
            "spread_states": 1,
            "min_state_observations": 1,
            "max_quote_gap_ms": 1000
        });
        assert!(serde_json::from_value::<MicroPriceConfig>(good.clone()).is_ok());
        for (field, value) in [
            ("tick_size", json!("0")),
            ("imbalance_buckets", json!(4)),
            ("spread_states", json!(0)),
            ("min_state_observations", json!(0)),
            ("max_quote_gap_ms", json!(0)),
        ] {
            let mut bad = good.clone();
            bad[field] = value;
            assert!(
                serde_json::from_value::<MicroPriceConfig>(bad).is_err(),
                "{field}"
            );
        }
    }
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
