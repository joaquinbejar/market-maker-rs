//! The A-S baseline quoting from walk-forward calibration in the causal replay.
//!
//! Synthetic processes of known parameters only. These check the wiring and its
//! causality, and make no claim about profitability.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::calibrated::{CalibratedQuoter, ParameterSchedule, QuoterStats};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestStrategy, MarketTick, SimLatencyConfig, SimulatedFill,
    VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::strategy::walk_forward::*;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const MINUTE_MS: u64 = 60_000;
const DAY_MS: u64 = 86_400_000;
const YEAR_S: f64 = 31_536_000.0;
const GAMMA: Decimal = dec!(0.1);
const HORIZON_MS: u64 = 3_600_000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.unit()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }
    fn poisson(&mut self, mean: f64) -> u64 {
        let limit = (-mean).exp();
        let (mut count, mut product) = (0, self.unit());
        while product > limit {
            count += 1;
            product *= self.unit();
        }
        count
    }
}

/// One minute prices over `days`, with this relative volatility per √year by day.
fn prices(seed: u64, sigma_of_day: &[f64]) -> Vec<PriceSample> {
    let mut rng = Rng(seed);
    let mut level = 100.0f64;
    let step_years = (MINUTE_MS as f64 / 1000.0 / YEAR_S).sqrt();
    let mut out = Vec::new();
    for (day, sigma) in sigma_of_day.iter().enumerate() {
        for minute in 0..DAY_MS / MINUTE_MS {
            level *= (sigma * step_years * rng.normal()).exp();
            out.push(PriceSample {
                timestamp_ms: day as u64 * DAY_MS + minute * MINUTE_MS,
                price: Decimal::from_f64(level).unwrap().round_dp(6),
            });
        }
    }
    out
}

/// One minute exposures at five distances, by day, for a rate `A exp(-k d)`.
fn exposures(seed: u64, k_of_day: &[f64]) -> Vec<QuoteExposure> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for (day, k) in k_of_day.iter().enumerate() {
        for minute in 0..DAY_MS / MINUTE_MS {
            for level in 1..=5u32 {
                let distance = f64::from(level) / 10.0;
                out.push(QuoteExposure {
                    timestamp_ms: day as u64 * DAY_MS + minute * MINUTE_MS,
                    distance: Decimal::from(level) / dec!(10),
                    exposure_ms: MINUTE_MS,
                    fills: rng.poisson(2.0 * (-k * distance).exp() * 60.0),
                });
            }
        }
    }
    out
}

fn ticks_of(prices: &[PriceSample]) -> Vec<MarketTick> {
    prices
        .iter()
        .map(|s| {
            MarketTick::new(
                s.timestamp_ms,
                s.price - dec!(0.05),
                dec!(5),
                s.price + dec!(0.05),
                dec!(5),
            )
        })
        .collect()
}

fn constant() -> ConstantParameters {
    ConstantParameters {
        volatility: dec!(5),
        intensity_scale_a: dec!(2),
        order_intensity_k: dec!(3),
    }
}

/// Folds of a day each, under an update rule, from a change of regime on day 2.
fn report_of(rule: UpdateRule, p: &[PriceSample], e: &[QuoteExposure]) -> WalkForwardReport {
    let spec = WindowSpec {
        train_ms: DAY_MS,
        test_ms: DAY_MS,
        step_ms: DAY_MS,
    };
    let days = p.last().unwrap().timestamp_ms / DAY_MS + 1;
    let splits = chronological_splits(0, days * DAY_MS, &spec).unwrap();
    let config = WalkForwardConfig::new()
        .with_resolution_ms(MINUTE_MS)
        .with_update_rule(rule);
    walk_forward(
        &CalibrationData::new(p, e).unwrap(),
        &splits,
        &config,
        "synthetic",
        &constant(),
    )
    .unwrap()
}

fn regime_data() -> (Vec<PriceSample>, Vec<QuoteExposure>) {
    (
        prices(7, &[0.5, 0.5, 2.0, 2.0]),
        exposures(11, &[3.0, 3.0, 6.0, 6.0]),
    )
}

fn engine(
    schedule: ParameterSchedule,
    ticks: Vec<MarketTick>,
) -> BacktestEngine<CalibratedQuoter, VecDataSource> {
    let quoter = CalibratedQuoter::new(schedule, GAMMA, HORIZON_MS, dec!(1)).unwrap();
    BacktestEngine::new(
        BacktestConfig::default()
            .with_fee_rate(dec!(0.0005))
            .with_tick_size(dec!(0.01))
            .with_lot_size(dec!(0.001)),
        quoter,
        VecDataSource::new(ticks),
    )
}

fn position(quantity: Decimal) -> InventoryPosition {
    let mut position = InventoryPosition::new();
    position.quantity = quantity;
    position
}

#[test]
fn test_quote_is_the_baseline_with_the_parameters_of_its_fold() {
    let (p, e) = regime_data();
    let report = report_of(UpdateRule::Rolling, &p, &e);
    let schedule = ParameterSchedule::from_report(&report).unwrap();
    let mut quoter = CalibratedQuoter::new(schedule, GAMMA, HORIZON_MS, dec!(2)).unwrap();
    for fold in &report.folds {
        let fit = fold.record.parameters().unwrap();
        let time = fold.split.test.start_ms + 7 * MINUTE_MS;
        let tick = MarketTick::new(time, dec!(99.95), dec!(5), dec!(100.05), dec!(5));
        let inventory = dec!(1.5);
        let quote = quoter.on_tick(&tick, &position(inventory)).unwrap();
        let (bid, ask) = calculate_optimal_quotes(
            dec!(100),
            inventory,
            GAMMA,
            fit.volatility,
            HORIZON_MS,
            fit.order_intensity_k,
        )
        .unwrap();
        assert_eq!(quote.bid_price, bid);
        assert_eq!(quote.ask_price, ask);
        assert_eq!(quote.bid_size, dec!(2));
        assert_eq!(quote.ask_size, dec!(2));
        assert_eq!(quote.timestamp, time);
    }
    assert_eq!(quoter.stats().quotes, report.folds.len() as u64);
}

#[test]
fn test_each_tick_reads_the_fold_it_falls_in_and_its_training_came_first() {
    let (p, e) = regime_data();
    let report = report_of(UpdateRule::Rolling, &p, &e);
    let schedule = ParameterSchedule::from_report(&report).unwrap();
    for fold in &report.folds {
        assert!(fold.record.training.end_ms <= fold.split.test.start_ms);
        let fit = fold.record.parameters().unwrap();
        for time in [fold.split.test.start_ms, fold.split.test.end_ms - 1] {
            let held = schedule.parameters_at(time).unwrap();
            assert_eq!(held.volatility, fit.volatility);
            assert_eq!(held.order_intensity_k, fit.order_intensity_k);
        }
    }
    // The first day only trains, and nothing holds for it.
    assert_eq!(schedule.parameters_at(0), None);
    assert_eq!(schedule.parameters_at(DAY_MS - 1), None);
    // Nor after the last window.
    assert_eq!(schedule.parameters_at(4 * DAY_MS), None);
}

#[test]
fn test_later_data_leaves_earlier_parameters_and_quotes_as_they_were() {
    let (p, e) = regime_data();
    let base = report_of(UpdateRule::Rolling, &p, &e);
    let base_schedule = ParameterSchedule::from_report(&base).unwrap();
    // Rewrite everything from a day on: what holds up to and on that day stays,
    // as it was trained before, and what trains on it moves.
    let rewritten_from = |day: u64| {
        let mut later_p = p.clone();
        for sample in later_p
            .iter_mut()
            .filter(|s| s.timestamp_ms >= day * DAY_MS)
        {
            sample.price *= dec!(3);
        }
        let mut later_e = e.clone();
        for exposure in later_e
            .iter_mut()
            .filter(|x| x.timestamp_ms >= day * DAY_MS)
        {
            exposure.fills += 5;
        }
        (later_p, later_e)
    };
    for day in [1, 2] {
        let (later_p, later_e) = rewritten_from(day);
        let moved = report_of(UpdateRule::Rolling, &later_p, &later_e);
        let schedule = ParameterSchedule::from_report(&moved).unwrap();
        let at = |schedule: &ParameterSchedule, d: u64| schedule.parameters_at(d * DAY_MS);
        for held in 1..=day {
            assert_eq!(
                at(&schedule, held),
                at(&base_schedule, held),
                "{day} {held}"
            );
        }
        assert_ne!(at(&schedule, day + 1), at(&base_schedule, day + 1), "{day}");
    }
    // A replay up to that day also stays, fill by fill.
    let (later_p, later_e) = rewritten_from(2);
    let moved = report_of(UpdateRule::Rolling, &later_p, &later_e);
    let run = |report: &WalkForwardReport, data: &[PriceSample]| {
        let schedule = ParameterSchedule::from_report(report).unwrap();
        let ticks: Vec<_> = ticks_of(data)
            .into_iter()
            .filter(|t| t.timestamp < 2 * DAY_MS)
            .collect();
        let mut engine = engine(schedule, ticks);
        let result = engine.run();
        (
            result.trades.len(),
            result.net_pnl,
            engine.strategy().stats(),
        )
    };
    assert_eq!(run(&base, &p), run(&moved, &later_p));
}

#[test]
fn test_rolling_follows_a_change_of_regime_and_frozen_does_not() {
    let (p, e) = regime_data();
    let rolling = ParameterSchedule::from_report(&report_of(UpdateRule::Rolling, &p, &e)).unwrap();
    let frozen = ParameterSchedule::from_report(&report_of(UpdateRule::Frozen, &p, &e)).unwrap();
    let on_day = |schedule: &ParameterSchedule, day: u64| {
        schedule.parameters_at(day * DAY_MS + MINUTE_MS).unwrap()
    };
    // Day 3 reads day 2, after the change: more volatility and a larger k.
    assert!(on_day(&rolling, 3).volatility > dec!(2) * on_day(&rolling, 2).volatility);
    assert!(on_day(&rolling, 3).order_intensity_k > on_day(&rolling, 2).order_intensity_k);
    // Near their true values: 0.5 and 2 per √year relative, near a price of 100.
    let relative = |p: Decimal| p.to_f64().unwrap() / 100.0;
    assert!((relative(on_day(&rolling, 2).volatility) / 0.5 - 1.0).abs() < 0.3);
    assert!((relative(on_day(&rolling, 3).volatility) / 2.0 - 1.0).abs() < 0.3);
    assert!((on_day(&rolling, 3).order_intensity_k.to_f64().unwrap() / 6.0 - 1.0).abs() < 0.2);
    // Frozen keeps the first training for every day.
    assert_eq!(on_day(&frozen, 1), on_day(&frozen, 3));
    assert_eq!(on_day(&frozen, 1), on_day(&rolling, 1));
}

#[test]
fn test_calibrated_and_constant_parameters_both_reach_the_replay() {
    let (p, e) = regime_data();
    let window = TimeWindow::new(DAY_MS, 4 * DAY_MS).unwrap();
    let ticks = ticks_of(&p);
    let schedule = ParameterSchedule::from_report(&report_of(UpdateRule::Rolling, &p, &e)).unwrap();
    let mut calibrated = engine(schedule, ticks.clone());
    let calibrated_result = calibrated.run();
    let fixed = ParameterSchedule::constant(window, &constant()).unwrap();
    let mut baseline = engine(fixed, ticks.clone());
    let baseline_result = baseline.run();
    // Each reads the whole tape, quotes on the same ticks and is consistent.
    assert_eq!(calibrated_result.num_ticks, ticks.len() as u64);
    assert_eq!(baseline_result.num_ticks, ticks.len() as u64);
    let on_window = 3 * DAY_MS / MINUTE_MS;
    assert_eq!(calibrated.strategy().stats().quotes, on_window);
    assert_eq!(baseline.strategy().stats().quotes, on_window);
    assert!(calibrated_result.accounting.is_consistent());
    assert!(baseline_result.accounting.is_consistent());
    // Neither trades before its first window.
    for result in [&calibrated_result, &baseline_result] {
        assert!(result.trades.iter().all(|t| t.timestamp >= DAY_MS));
    }
    // The parameters do reach the replay.
    assert_ne!(calibrated_result.net_pnl, baseline_result.net_pnl);
}

#[test]
fn test_a_run_reproduces_and_reset_clears_the_stats() {
    let (p, e) = regime_data();
    let run = || {
        let schedule =
            ParameterSchedule::from_report(&report_of(UpdateRule::Rolling, &p, &e)).unwrap();
        let mut engine = engine(schedule, ticks_of(&p));
        let result = engine.run();
        (
            result.trades.len(),
            result.net_pnl,
            result.final_position,
            engine.strategy().stats(),
        )
    };
    assert_eq!(run(), run());
    let window = TimeWindow::new(0, 10).unwrap();
    let mut quoter = CalibratedQuoter::new(
        ParameterSchedule::constant(window, &constant()).unwrap(),
        GAMMA,
        HORIZON_MS,
        dec!(1),
    )
    .unwrap();
    let tick = MarketTick::new(1, dec!(99.95), dec!(5), dec!(100.05), dec!(5));
    assert!(quoter.on_tick(&tick, &position(dec!(0))).is_some());
    assert_eq!(quoter.stats().quotes, 1);
    quoter.reset();
    assert_eq!(quoter.stats(), QuoterStats::default());
}

/// Prices with too few samples on day 1, so its calibration fails for day 2.
fn data_with_a_failed_fold() -> (Vec<PriceSample>, Vec<QuoteExposure>) {
    let mut p = prices(7, &[2.0, 2.0, 2.0, 2.0]);
    p.retain(|s| {
        !(DAY_MS..2 * DAY_MS).contains(&s.timestamp_ms) || s.timestamp_ms % (60 * MINUTE_MS) == 0
    });
    (p, exposures(11, &[3.0, 3.0, 3.0, 3.0]))
}

#[test]
fn test_a_failed_fold_is_silent_and_inherits_nothing() {
    let (p, e) = data_with_a_failed_fold();
    let report = report_of(UpdateRule::Rolling, &p, &e);
    assert!(report.folds[0].record.parameters().is_some());
    assert!(report.folds[1].record.parameters().is_none());
    assert!(report.folds[2].record.parameters().is_some());
    let schedule = ParameterSchedule::from_report(&report).unwrap();
    assert_eq!(schedule.parameters_at(2 * DAY_MS), None);
    let ticks: Vec<_> = ticks_of(&p);
    let on_failed = ticks
        .iter()
        .filter(|t| (2 * DAY_MS..3 * DAY_MS).contains(&t.timestamp))
        .count() as u64;
    let mut engine = engine(schedule, ticks);
    let result = engine.run();
    let stats = engine.strategy().stats();
    assert_eq!(stats.without_parameters, on_failed);
    assert_eq!(stats.quote_errors, 0);
    // Its first tick matches before the quoter hears of it. After that nothing rests.
    assert!(
        result
            .trades
            .iter()
            .all(|t| t.timestamp <= 2 * DAY_MS || t.timestamp >= 3 * DAY_MS),
        "{:?}",
        result.trades
    );
    assert!(result.trades.iter().any(|t| t.timestamp >= 3 * DAY_MS));
}

#[test]
fn test_a_quote_that_cannot_be_computed_is_silent_and_counted() {
    // A mid of zero is refused by the baseline.
    let window = TimeWindow::new(0, 10_000).unwrap();
    let mut quoter = CalibratedQuoter::new(
        ParameterSchedule::constant(window, &constant()).unwrap(),
        GAMMA,
        HORIZON_MS,
        dec!(1),
    )
    .unwrap();
    let tick = MarketTick::new(1, dec!(0), dec!(5), dec!(0), dec!(5));
    assert!(quoter.on_tick(&tick, &position(dec!(0))).is_none());
    assert_eq!(quoter.stats().quote_errors, 1);
    assert_eq!(quoter.stats().quotes, 0);
    assert!(quoter.cancels_when_silent());
}

#[test]
fn test_order_in_flight_at_a_window_end_can_outlive_it_under_latency() {
    // The stated limit. The last tick of the window quotes, and the fall comes after
    // the window ends. With no latency the silent tick cancels first; with latency
    // the quoter only hears of silence after the fall, so the order fills.
    let window = TimeWindow::new(1_000, 2_000).unwrap();
    let ticks = vec![
        MarketTick::new(1_900, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(2_100, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(2_200, dec!(49.95), dec!(5), dec!(50.05), dec!(5)),
    ];
    let run = |latency: Option<SimLatencyConfig>| {
        let fixed = ParameterSchedule::constant(window, &constant()).unwrap();
        let mut engine = engine(fixed, ticks.clone());
        if let Some(latency) = latency {
            engine = engine.with_latency(latency).unwrap();
        }
        engine.run().trades
    };
    assert!(run(None).is_empty());
    let late = run(Some(SimLatencyConfig::new().with_observation_ms(150)));
    assert_eq!(late.len(), 1, "{late:?}");
    assert!(late[0].timestamp >= window.end_ms);
}

#[test]
fn test_quoter_cancels_what_rests_once_it_has_no_parameters() {
    // A quote wide of the book rests on the last tick of the window. Then the mid
    // falls through its bid. With nothing to quote from, it must not fill.
    let window = TimeWindow::new(1_000, 2_000).unwrap();
    let ticks = vec![
        MarketTick::new(1_500, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(2_000, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(3_000, dec!(49.95), dec!(5), dec!(50.05), dec!(5)),
        MarketTick::new(4_000, dec!(149.95), dec!(5), dec!(150.05), dec!(5)),
    ];
    let fixed = ParameterSchedule::constant(window, &constant()).unwrap();
    let mut engine = engine(fixed, ticks);
    let result = engine.run();
    assert_eq!(engine.strategy().stats().quotes, 1);
    assert_eq!(engine.strategy().stats().outside_schedule, 3);
    assert_eq!(result.trades.len(), 0, "{:?}", result.trades);
}

/// Quotes once, wide of the book, then stays silent; a test sets what that means.
struct OnceThenSilent {
    cancels: bool,
    done: bool,
}

impl BacktestStrategy for OnceThenSilent {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        if self.done {
            return None;
        }
        self.done = true;
        Some(Quote {
            bid_price: dec!(90),
            bid_size: dec!(1),
            ask_price: dec!(110),
            ask_size: dec!(1),
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn cancels_when_silent(&self) -> bool {
        self.cancels
    }
    fn reset(&mut self) {
        self.done = false;
    }
}

fn through_the_bid() -> Vec<MarketTick> {
    vec![
        MarketTick::new(1_000, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(2_000, dec!(99.95), dec!(5), dec!(100.05), dec!(5)),
        MarketTick::new(3_000, dec!(49.95), dec!(5), dec!(50.05), dec!(5)),
    ]
}

#[test]
fn test_silence_keeps_orders_by_default_and_cancels_when_the_strategy_says_so() {
    let run = |cancels: bool, latency: Option<SimLatencyConfig>| {
        let mut engine = BacktestEngine::new(
            BacktestConfig::default().with_tick_size(dec!(0.01)),
            OnceThenSilent {
                cancels,
                done: false,
            },
            VecDataSource::new(through_the_bid()),
        );
        if let Some(latency) = latency {
            engine = engine.with_latency(latency).unwrap();
        }
        engine.run().trades.len()
    };
    assert_eq!(run(false, None), 1);
    assert_eq!(run(true, None), 0);
    // A cancel takes its latency: slower than the fall, the order still fills.
    let slow = SimLatencyConfig::new().with_cancel_ms(5_000);
    assert_eq!(run(true, Some(slow)), 1);
}
