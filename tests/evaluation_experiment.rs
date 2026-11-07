//! The rules of an experiment: the same execution for every candidate, no leak from
//! the holdout, failures kept in the report, and reproducible reports.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    BootstrapConfig, BootstrapOutcome, Candidate, CandidateKind, EvaluationError, EvaluationRow,
    Experiment, ExperimentConfig, MetricId, MetricsSpec, Provenance, Reading, RowMetrics,
    RowOutcome, Scenario, Splits, WindowKind, block_bootstrap,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestStrategy, FillModel, FillResult, ImmediateFillModel, MarketTick,
    ProbabilisticFillModel, SimLatencyConfig, SimulatedFill, SimulatedOrder,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::pretrade::{PretradeDecision, PretradePolicy, PretradeRequest};
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};

const DAY: u64 = 86_400_000;
const HOUR: u64 = 3_600_000;

/// Hourly ticks from day 0 to day 12, with moves larger than the quoted spread.
fn ticks() -> Vec<MarketTick> {
    let mut state = 777_u64;
    let mut mid = dec!(1000);
    (0..12 * 24)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            mid += Decimal::from((state >> 33) % 201) / dec!(100) - dec!(1);
            MarketTick::new(
                i * HOUR,
                mid - dec!(0.05),
                dec!(5),
                mid + dec!(0.05),
                dec!(5),
            )
        })
        .collect()
}

fn splits() -> Splits {
    Splits::new(0, 4 * DAY, 8 * DAY, 12 * DAY).unwrap()
}

fn experiment_config() -> ExperimentConfig {
    ExperimentConfig {
        name: "fixture".to_string(),
        execution: BacktestConfig::new()
            .with_initial_capital(dec!(100000))
            .with_fee_rate(dec!(0.001))
            .with_tick_size(dec!(0.01))
            .with_lot_size(dec!(0.01)),
        splits: splits(),
        metrics: MetricsSpec {
            horizons_ms: vec![HOUR],
            max_mark_delay_ms: 0,
        },
        bootstrap: BootstrapConfig {
            replicates: 200,
            mean_block_days: 2,
            confidence_percent: 90,
            seed: 11,
            min_days: 2,
        },
        seed: 42,
        provenance: Provenance {
            commit: "unverified-commit".to_string(),
            toolchain: "unverified-toolchain".to_string(),
        },
    }
}

fn immediate() -> Scenario {
    Scenario::new(
        "immediate",
        SimLatencyConfig::new(),
        BTreeMap::new(),
        |_| ImmediateFillModel,
    )
}

fn immediate_named(name: &str) -> Scenario {
    Scenario::new(name, SimLatencyConfig::new(), BTreeMap::new(), |_| {
        ImmediateFillModel
    })
}

/// A strategy with its own distance from the mid.
struct Quoter(Decimal);

impl BacktestStrategy for Quoter {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        Some(Quote {
            bid_price: tick.mid_price() - self.0,
            bid_size: dec!(0.5),
            ask_price: tick.mid_price() + self.0,
            ask_size: dec!(0.5),
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

fn quoter(name: &str, half_spread: Decimal) -> Candidate {
    Candidate::model(
        name,
        BTreeMap::from([("half_spread".to_string(), half_spread.to_string())]),
        move |_| Ok(Box::new(Quoter(half_spread)) as Box<dyn BacktestStrategy>),
    )
}

fn baselines() -> Vec<Candidate> {
    vec![
        Candidate::no_trade(),
        Candidate::fixed_spread(dec!(0.5), dec!(0.5)),
    ]
}

fn with(models: Vec<Candidate>) -> Vec<Candidate> {
    let mut all = baselines();
    all.extend(models);
    all
}

fn row<'a>(rows: &'a [EvaluationRow], candidate: &str, scenario: &str) -> &'a EvaluationRow {
    rows.iter()
        .find(|r| r.candidate == candidate && r.scenario == scenario)
        .unwrap_or_else(|| panic!("no row {candidate} {scenario}"))
}

fn completed(row: &EvaluationRow) -> &RowMetrics {
    match &row.outcome {
        RowOutcome::Completed(metrics) => metrics,
        other => panic!("{} did not complete: {other:?}", row.candidate),
    }
}

// ------------------------------------------------------------ the same rules

#[test]
fn test_two_candidates_with_the_same_quotes_get_identical_results() {
    let report = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25)), quoter("b", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let rows = &report.report().rows;
    let (a, b) = (
        completed(row(rows, "a", "immediate")),
        completed(row(rows, "b", "immediate")),
    );
    assert!(a.metrics.fills > 10, "only {} fills", a.metrics.fills);
    assert_eq!(a, b);
}

#[test]
fn test_candidate_order_does_not_change_any_row() {
    let seeded = || {
        Scenario::new(
            "probabilistic",
            SimLatencyConfig::new().with_submit_ms(5).with_jitter(3, 9),
            BTreeMap::new(),
            |seed| ProbabilisticFillModel::new(dec!(0.5), dec!(0), dec!(1), seed),
        )
    };
    let run = |models: Vec<Candidate>| {
        Experiment::new(experiment_config(), ticks(), models, vec![seeded()])
            .unwrap()
            .evaluate_validation()
            .report()
            .clone()
    };
    let forward = run(with(vec![quoter("a", dec!(0.25)), quoter("b", dec!(0.75))]));
    assert!(
        completed(row(&forward.rows, "a", "probabilistic"))
            .metrics
            .fills
            > 0
    );
    let mut backward = vec![quoter("b", dec!(0.75)), quoter("a", dec!(0.25))];
    backward.extend(baselines().into_iter().rev());
    assert_eq!(forward, run(backward));
}

/// Counts how many models were built and which seeds they got.
#[derive(Debug)]
struct Counting(ImmediateFillModel);

impl FillModel for Counting {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        self.0.simulate_fill(order, tick, time_in_queue_ms)
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "counting"
    }
}

#[test]
fn test_each_run_gets_a_new_fill_model_with_a_seed_that_ignores_the_candidate() {
    let seeds = Rc::new(RefCell::new(Vec::new()));
    let seen = Rc::clone(&seeds);
    let scenario = Scenario::new("s", SimLatencyConfig::new(), BTreeMap::new(), move |seed| {
        seen.borrow_mut().push(seed);
        Counting(ImmediateFillModel)
    });
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![scenario],
    )
    .unwrap()
    .evaluate_validation();
    let manifest = validated.report().manifest.clone();
    let seed = manifest.scenarios["s"].validation_seed;
    let runs: Vec<u64> = seeds.borrow().iter().copied().filter(|s| *s != 0).collect();
    assert_eq!(
        runs,
        [seed, seed, seed],
        "one model per candidate, same seed"
    );
    assert_ne!(seed, manifest.scenarios["s"].holdout_seed);
}

// ------------------------------------------------------------ windows and the holdout

#[test]
fn test_strategy_factory_sees_only_training_ticks_on_every_window() {
    let seen: Rc<RefCell<Vec<(usize, u64)>>> = Rc::default();
    let log = Rc::clone(&seen);
    let spy = Candidate::model("spy", BTreeMap::new(), move |train| {
        log.borrow_mut()
            .push((train.len(), train.last().unwrap().timestamp));
        Ok(Box::new(Quoter(dec!(0.25))) as Box<dyn BacktestStrategy>)
    });
    let _report = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![spy]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation()
    .select(&["spy"])
    .unwrap()
    .run_holdout();
    // Validation and the holdout both build from days 0 to 3, the tick exactly at 4 *
    // DAY being validation's.
    let last_train = 4 * DAY - HOUR;
    assert_eq!(*seen.borrow(), [(96, last_train), (96, last_train)]);
}

#[test]
fn test_tick_on_a_boundary_belongs_to_the_later_split() {
    let report = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let daily = &completed(row(&report.report().rows, "no-trade", "immediate")).daily;
    assert_eq!(daily.first().unwrap().day, 4);
    assert_eq!(daily.last().unwrap().day, 7);
    assert!(daily.iter().all(|d| d.bucket.marks == 24));
}

#[test]
fn test_changing_the_holdout_data_changes_no_validation_result() {
    let run = |data: Vec<MarketTick>| {
        Experiment::new(
            experiment_config(),
            data,
            with(vec![quoter("a", dec!(0.25))]),
            vec![immediate()],
        )
        .unwrap()
        .evaluate_validation()
        .report()
        .rows
        .clone()
    };
    let original = ticks();
    let mut changed = original.clone();
    for tick in changed.iter_mut().filter(|t| t.timestamp >= 8 * DAY) {
        tick.bid_price *= dec!(3);
        tick.ask_price *= dec!(3);
    }
    assert_eq!(run(original), run(changed));
}

#[test]
fn test_markout_near_the_end_of_validation_is_excluded_rather_than_read_from_the_holdout() {
    let mut config = experiment_config();
    config.metrics.horizons_ms = vec![2 * HOUR];
    let report = Experiment::new(
        config,
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let quality = completed(row(&report.report().rows, "a", "immediate"))
        .metrics
        .fill_quality[&(2 * HOUR)];
    // Fills at the last two hours of validation have no mark two hours on.
    assert!(quality.all.excluded >= 1);
    assert_eq!(
        quality.all.measured + quality.all.excluded,
        quality.all.fills
    );
}

#[test]
fn test_holdout_runs_the_selection_and_the_baselines_under_the_reference_only() {
    let sensitivity = Scenario::new(
        "slow",
        SimLatencyConfig::new().with_submit_ms(HOUR),
        BTreeMap::new(),
        |_| ImmediateFillModel,
    );
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25)), quoter("b", dec!(0.75))]),
        vec![immediate(), sensitivity],
    )
    .unwrap()
    .evaluate_validation();
    // Every candidate under every scenario, on validation.
    assert_eq!(validated.report().rows.len(), 8);
    let selection = validated.select(&["a"]).unwrap();
    assert_eq!(*selection.selected(), BTreeSet::from(["a".to_string()]));
    let report = selection.run_holdout();
    let names: Vec<(&str, &str, WindowKind)> = report
        .holdout
        .iter()
        .map(|r| (r.candidate.as_str(), r.scenario.as_str(), r.window))
        .collect();
    assert_eq!(
        names,
        [
            ("a", "immediate", WindowKind::Holdout),
            ("fixed-spread", "immediate", WindowKind::Holdout),
            ("no-trade", "immediate", WindowKind::Holdout),
        ]
    );
    assert_eq!(report.not_selected, ["b"]);
    assert_eq!(report.validation.len(), 8);
    assert_eq!(report.manifest.selection, ["a"]);
    assert!(report.manifest.validation_digest.is_some());
    assert_eq!(report.manifest.candidates.len(), 4);
    assert_eq!(
        report.manifest.candidates["no-trade"].kind,
        CandidateKind::Baseline
    );
}

#[test]
fn test_holdout_result_comes_from_holdout_ticks() {
    let report = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation()
    .select(&["fixed-spread"])
    .unwrap()
    .run_holdout();
    let daily = &completed(row(&report.holdout, "fixed-spread", "immediate")).daily;
    assert_eq!(daily.first().unwrap().day, 8);
    assert_eq!(daily.last().unwrap().day, 11);
}

// ------------------------------------------------------------ refusals

fn refused(result: Result<Experiment, EvaluationError>) -> String {
    match result {
        Err(EvaluationError::InvalidInput { reason }) => reason,
        Err(other) => panic!("unexpected {other}"),
        Ok(_) => panic!("accepted"),
    }
}

#[test]
fn test_new_refuses_a_run_without_a_simple_baseline() {
    let reason = refused(Experiment::new(
        experiment_config(),
        ticks(),
        vec![quoter("a", dec!(0.25))],
        vec![immediate()],
    ));
    assert!(reason.contains("baseline"), "{reason}");
}

#[test]
fn test_new_refuses_bad_data_and_repeated_names() {
    let mut out_of_order = ticks();
    out_of_order.swap(3, 4);
    let reason = refused(Experiment::new(
        experiment_config(),
        out_of_order,
        baselines(),
        vec![immediate()],
    ));
    assert!(reason.contains("out of order"), "{reason}");

    let mut outside = ticks();
    outside.push(MarketTick::new(
        12 * DAY,
        dec!(1),
        dec!(1),
        dec!(2),
        dec!(1),
    ));
    let reason = refused(Experiment::new(
        experiment_config(),
        outside,
        baselines(),
        vec![immediate()],
    ));
    assert!(reason.contains("outside"), "{reason}");

    let no_holdout: Vec<MarketTick> = ticks()
        .into_iter()
        .filter(|t| t.timestamp < 8 * DAY)
        .collect();
    let reason = refused(Experiment::new(
        experiment_config(),
        no_holdout,
        baselines(),
        vec![immediate()],
    ));
    assert!(reason.contains("every split"), "{reason}");

    let reason = refused(Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(1)), quoter("a", dec!(2))]),
        vec![immediate()],
    ));
    assert!(reason.contains("twice"), "{reason}");

    let reason = refused(Experiment::new(
        experiment_config(),
        ticks(),
        baselines(),
        vec![immediate(), immediate()],
    ));
    assert!(reason.contains("different"), "{reason}");

    let reason = refused(Experiment::new(
        experiment_config(),
        ticks(),
        baselines(),
        vec![],
    ));
    assert!(reason.contains("scenario"), "{reason}");

    let mut no_horizon = experiment_config();
    no_horizon.metrics.horizons_ms = vec![0];
    let reason = refused(Experiment::new(
        no_horizon,
        ticks(),
        baselines(),
        vec![immediate()],
    ));
    assert!(reason.contains("horizon"), "{reason}");
}

#[test]
fn test_splits_refuse_limits_that_do_not_increase() {
    assert!(Splits::new(0, 0, 1, 2).is_err());
    assert!(Splits::new(0, 2, 1, 3).is_err());
    assert!(Splits::new(0, 1, 2, 2).is_err());
}

#[test]
fn test_select_refuses_an_unknown_name_an_empty_list_and_a_failed_row() {
    let validated = || {
        let broken = Candidate::model("broken", BTreeMap::new(), |_| {
            Err(MMError::InvalidConfiguration("cannot build".to_string()))
        });
        Experiment::new(
            experiment_config(),
            ticks(),
            with(vec![broken]),
            vec![immediate()],
        )
        .unwrap()
        .evaluate_validation()
    };
    for (names, expected) in [
        (vec!["nobody"], "unknown"),
        (vec![], "needs a candidate"),
        (vec!["broken"], "failed"),
    ] {
        match validated().select(&names) {
            Err(EvaluationError::InvalidInput { reason }) => {
                assert!(reason.contains(expected), "{reason}");
            }
            other => panic!("{other:?}"),
        }
    }
}

// ------------------------------------------------------------ failures stay visible

struct FailingFill;

impl FillModel for FailingFill {
    fn simulate_fill(
        &self,
        _order: &SimulatedOrder,
        _tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        Err(MMError::InvalidMarketState("no fill today".to_string()))
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "failing"
    }
}

#[test]
fn test_failed_factory_is_a_row_with_its_reason_and_the_others_go_on() {
    let broken = Candidate::model("broken", BTreeMap::new(), |_| {
        Err(MMError::InvalidConfiguration("cannot build".to_string()))
    });
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![broken, quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let rows = &validated.report().rows;
    assert_eq!(rows.len(), 4);
    match &row(rows, "broken", "immediate").outcome {
        RowOutcome::Failed { reason } => assert!(reason.contains("cannot build"), "{reason}"),
        other => panic!("{other:?}"),
    }
    completed(row(rows, "a", "immediate"));
    // The failed candidate is still in the final report.
    let report = validated.select(&["a"]).unwrap().run_holdout();
    assert!(report.validation.iter().any(|r| r.candidate == "broken"));
    assert_eq!(report.not_selected, ["broken"]);
}

#[test]
fn test_engine_fault_fails_the_row_of_every_candidate_that_trades() {
    let scenario = Scenario::new("failing", SimLatencyConfig::new(), BTreeMap::new(), |_| {
        FailingFill
    });
    let validated = Experiment::new(experiment_config(), ticks(), baselines(), vec![scenario])
        .unwrap()
        .evaluate_validation();
    let rows = &validated.report().rows;
    match &row(rows, "fixed-spread", "failing").outcome {
        RowOutcome::Failed { reason } => assert!(reason.contains("no fill today"), "{reason}"),
        other => panic!("{other:?}"),
    }
    // Never quoting, it never asks the model.
    completed(row(rows, "no-trade", "failing"));
}

#[test]
fn test_failed_baseline_is_left_out_of_the_differences_and_visible() {
    let scenario = Scenario::new("failing", SimLatencyConfig::new(), BTreeMap::new(), |_| {
        FailingFill
    });
    let validated = Experiment::new(experiment_config(), ticks(), baselines(), vec![scenario])
        .unwrap()
        .evaluate_validation();
    let no_trade = completed(row(&validated.report().rows, "no-trade", "failing"));
    assert!(no_trade.against_baselines.is_empty());
}

// ------------------------------------------------------------ policies and regimes

#[derive(Debug)]
struct RefuseAll(Arc<AtomicU64>);

impl PretradePolicy for RefuseAll {
    fn check(&mut self, _request: &PretradeRequest) -> PretradeDecision {
        self.0.fetch_add(1, Ordering::Relaxed);
        PretradeDecision::Rejected {
            reason: "no".to_string(),
        }
    }
    fn reset(&mut self) {}
    fn description(&self) -> String {
        "refuse all".to_string()
    }
}

#[test]
fn test_pretrade_policy_applies_to_every_candidate_and_goes_in_the_manifest() {
    let checks = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&checks);
    let validated = Experiment::new(experiment_config(), ticks(), baselines(), vec![immediate()])
        .unwrap()
        .with_pretrade_policy(move || Box::new(RefuseAll(Arc::clone(&counter))))
        .evaluate_validation();
    let report = validated.report();
    assert_eq!(
        report.manifest.pretrade_policy.as_deref(),
        Some("refuse all")
    );
    assert_eq!(
        completed(row(&report.rows, "fixed-spread", "immediate"))
            .metrics
            .fills,
        0
    );
    assert!(checks.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_regimes_sum_to_the_run_with_a_new_labeller_per_run() {
    let labellers = Rc::new(RefCell::new(0));
    let made = Rc::clone(&labellers);
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .with_regimes("moves above 0.5", move || {
        *made.borrow_mut() += 1;
        let mut last: Option<Decimal> = None;
        Box::new(move |mark| {
            let label = match last {
                Some(previous) if (mark.mid - previous).abs() > dec!(0.5) => "moving",
                _ => "quiet",
            };
            last = Some(mark.mid);
            label.to_string()
        })
    })
    .evaluate_validation();
    assert_eq!(*labellers.borrow(), 3);
    let metrics = completed(row(&validated.report().rows, "a", "immediate"));
    assert_eq!(metrics.regimes.len(), 2);
    let sum: Decimal = metrics
        .regimes
        .values()
        .map(|b| b.attribution.net_pnl)
        .sum();
    assert_eq!(sum, metrics.metrics.attribution.net_pnl);
}

// ------------------------------------------------------------ intervals

fn interval(outcome: BootstrapOutcome) -> (f64, f64, f64, u32) {
    match outcome {
        BootstrapOutcome::Interval {
            mean,
            lower,
            upper,
            days,
        } => (mean, lower, upper, days),
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_difference_to_a_baseline_that_never_trades_equals_the_row_itself() {
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let metrics = completed(row(&validated.report().rows, "a", "immediate"));
    // No-trade earns zero every day, so the same resampled days give the same numbers.
    assert_eq!(
        metrics.against_baselines["no-trade"],
        metrics.mean_daily_net
    );
    let (mean, lower, upper, days) = interval(metrics.mean_daily_net);
    assert_eq!(days, 4);
    assert!(lower <= mean && mean <= upper);
    let net = metrics.metrics.attribution.net_pnl;
    assert!((mean - f64::try_from(net).unwrap() / 4.0).abs() < 1e-9);
}

fn daily_net(metrics: &RowMetrics) -> Vec<Decimal> {
    metrics
        .daily
        .iter()
        .filter(|day| day.bucket.marks > 0)
        .map(|day| day.bucket.attribution.net_pnl)
        .collect()
}

#[test]
fn test_block_bootstrap_of_a_rows_daily_results_has_its_mean_and_days() {
    let config = experiment_config();
    let validated = Experiment::new(
        config.clone(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let rows = &validated.report().rows;
    let a = completed(row(rows, "a", "immediate"));
    let fixed = completed(row(rows, "fixed-spread", "immediate"));
    let series = daily_net(a);
    let difference: Vec<Decimal> = series
        .iter()
        .zip(daily_net(fixed))
        .map(|(x, y)| x - y)
        .collect();
    // The harness seeds each window and scenario on its own, so only the parts that do
    // not depend on the seed are compared here: the mean and the days.
    for (values, harness) in [
        (&series, a.mean_daily_net),
        (&difference, a.against_baselines["fixed-spread"]),
    ] {
        let (mean, lower, upper, days) =
            interval(block_bootstrap(values, &config.bootstrap).unwrap());
        let (expected_mean, _, _, expected_days) = interval(harness);
        assert_eq!(days, expected_days);
        assert!((mean - expected_mean).abs() < 1e-9);
        assert!(lower <= mean && mean <= upper);
    }
}

#[test]
fn test_days_without_ticks_stay_in_the_calendar_and_out_of_the_bootstrap() {
    let data: Vec<MarketTick> = ticks()
        .into_iter()
        .filter(|t| t.timestamp / DAY != 5)
        .collect();
    let validated = Experiment::new(
        experiment_config(),
        data,
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let metrics = completed(row(&validated.report().rows, "a", "immediate"));
    assert_eq!(metrics.daily.len(), 4);
    assert_eq!(metrics.daily[1].bucket.marks, 0);
    assert_eq!(interval(metrics.mean_daily_net).3, 3);
}

#[test]
fn test_too_few_days_report_insufficient_instead_of_an_interval() {
    let mut config = experiment_config();
    config.bootstrap.min_days = 5;
    let validated = Experiment::new(config, ticks(), baselines(), vec![immediate()])
        .unwrap()
        .evaluate_validation();
    let metrics = completed(row(&validated.report().rows, "fixed-spread", "immediate"));
    assert_eq!(
        metrics.mean_daily_net,
        BootstrapOutcome::Insufficient {
            days: 4,
            required: 5
        }
    );
}

#[test]
fn test_every_row_carries_its_metric_definitions() {
    let validated = Experiment::new(experiment_config(), ticks(), baselines(), vec![immediate()])
        .unwrap()
        .evaluate_validation();
    let metrics = completed(row(&validated.report().rows, "fixed-spread", "immediate"));
    let net = metrics
        .values
        .iter()
        .find(|v| v.id == MetricId::NetPnl)
        .unwrap();
    assert_eq!(
        net.reading,
        Reading::Value(metrics.metrics.attribution.net_pnl)
    );
    assert_eq!(net.definition, MetricId::NetPnl.definition());
}

// ------------------------------------------------------------ the manifest

#[test]
fn test_manifest_pins_data_and_changes_with_the_data() {
    let manifest = |data| {
        Experiment::new(experiment_config(), data, baselines(), vec![immediate()])
            .unwrap()
            .evaluate_validation()
            .report()
            .manifest
            .clone()
    };
    let original = manifest(ticks());
    assert_eq!(original.provenance.commit, "unverified-commit");
    assert_eq!(original.crate_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(original.reference_scenario, "immediate");
    assert!(
        original.scenarios["immediate"]
            .fill_model
            .starts_with("ImmediateFill")
    );
    assert!(original.selection.is_empty());
    assert!(original.validation_digest.is_none());
    assert_eq!(original.data_hashes.len(), 3);

    // 1.0 and 1.00 are the same data.
    let mut rescaled = ticks();
    rescaled[100].bid_size = dec!(5.00);
    assert_eq!(original.data_hashes, manifest(rescaled).data_hashes);

    let mut changed = ticks();
    changed[100].bid_price += dec!(0.01);
    let other = manifest(changed);
    assert_ne!(
        original.data_hashes["validation"],
        other.data_hashes["validation"]
    );
    assert_eq!(
        original.data_hashes["holdout"],
        other.data_hashes["holdout"]
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_two_runs_serialize_to_the_same_bytes() {
    let run = || {
        let report = Experiment::new(
            experiment_config(),
            ticks(),
            with(vec![quoter("a", dec!(0.25))]),
            vec![immediate()],
        )
        .unwrap()
        .evaluate_validation()
        .select(&["a"])
        .unwrap()
        .run_holdout();
        serde_json::to_vec(&report).unwrap()
    };
    assert_eq!(run(), run());
}

#[test]
fn test_scenario_fee_rate_replaces_the_rate_and_nothing_else() {
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![
            immediate(),
            immediate_named("double").with_fee_rate(dec!(0.002)),
            immediate_named("rebate").with_fee_rate(dec!(-0.001)),
        ],
    )
    .unwrap()
    .evaluate_validation();
    let rows = &validated.report().rows;
    let totals = |scenario: &str| completed(row(rows, "a", scenario)).metrics.attribution;
    let (base, double, rebate) = (totals("immediate"), totals("double"), totals("rebate"));
    assert!(base.quote_fees > dec!(0));
    // The experiment's rate is 0.001: the same fills at twice it and at minus it.
    assert_eq!(double.quote_fees, base.quote_fees * dec!(2));
    assert_eq!(rebate.quote_fees, -base.quote_fees);
    assert_eq!(double.spread_capture, base.spread_capture);
    assert_eq!(double.inventory_pnl, base.inventory_pnl);
    assert_eq!(double.net_pnl, base.net_pnl - base.quote_fees);
    let scenarios = &validated.report().manifest.scenarios;
    assert_eq!(scenarios["immediate"].fee_rate, None);
    assert_eq!(scenarios["double"].fee_rate, Some(dec!(0.002)));
}

#[test]
fn test_holdout_uses_the_fee_rate_of_the_reference_scenario() {
    let report = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![
            immediate().with_fee_rate(dec!(0)),
            immediate_named("charged"),
        ],
    )
    .unwrap()
    .evaluate_validation()
    .select(&["a"])
    .unwrap()
    .run_holdout();
    let row = report.holdout.iter().find(|r| r.candidate == "a").unwrap();
    assert_eq!(completed(row).metrics.attribution.quote_fees, dec!(0));
}

#[test]
fn test_difference_to_an_identical_trading_baseline_is_exactly_zero() {
    let validated = Experiment::new(
        experiment_config(),
        ticks(),
        with(vec![quoter("same", dec!(0.5))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    let metrics = completed(row(&validated.report().rows, "same", "immediate"));
    assert!(metrics.metrics.fills > 0);
    assert_eq!(interval(metrics.against_baselines["fixed-spread"]).0, 0.0);
    let (_, lower, upper, _) = interval(metrics.against_baselines["fixed-spread"]);
    assert_eq!((lower, upper), (0.0, 0.0));
}

#[test]
fn test_data_hash_covers_the_last_trade_fields() {
    let hashes = |data| {
        Experiment::new(experiment_config(), data, baselines(), vec![immediate()])
            .unwrap()
            .evaluate_validation()
            .report()
            .manifest
            .data_hashes
            .clone()
    };
    // Tick 10 is in the training window, and each field alone must move its hash.
    let with_last = |price: Option<Decimal>, size: Option<Decimal>| {
        let mut data = ticks();
        data[10].last_price = price;
        data[10].last_size = size;
        hashes(data)["train"].clone()
    };
    let none = with_last(None, None);
    let price = with_last(Some(dec!(100)), None);
    let size = with_last(None, Some(dec!(100)));
    let other_price = with_last(Some(dec!(101)), None);
    let other_size = with_last(None, Some(dec!(101)));
    let all = [&none, &price, &size, &other_price, &other_size];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // The same value at another scale is the same data here too.
    assert_eq!(price, with_last(Some(dec!(100.00)), None));
}

#[test]
fn test_manifest_records_the_regime_labeller() {
    let manifest = |description: Option<&str>, label: &'static str| {
        let experiment = Experiment::new(
            experiment_config(),
            ticks(),
            with(vec![quoter("a", dec!(0.25))]),
            vec![immediate()],
        )
        .unwrap();
        let experiment = match description {
            Some(text) => experiment.with_regimes(text, move || {
                Box::new(move |_: &market_maker_rs::backtest::evaluation::Mark| label.to_string())
            }),
            None => experiment,
        };
        let validated = experiment.evaluate_validation();
        let regimes: Vec<String> = completed(row(&validated.report().rows, "a", "immediate"))
            .regimes
            .keys()
            .cloned()
            .collect();
        (validated.report().manifest.clone(), regimes)
    };
    let (none, no_regimes) = manifest(None, "");
    let (first, first_regimes) = manifest(Some("always x"), "x");
    let (second, second_regimes) = manifest(Some("always y"), "y");
    assert!(no_regimes.is_empty());
    assert_eq!(none.regime_labeller, None);
    // Different labels in the reports, so the manifests must differ too.
    assert_ne!(first_regimes, second_regimes);
    assert_eq!(first.regime_labeller.as_deref(), Some("always x"));
    assert_ne!(first, second);
    assert_ne!(first, none);
}

/// Quotes on its first tick and is silent afterwards. The fixture moves at most 1 a
/// tick, so 1.5 from the mid cannot fill on the next tick, before the first silence.
struct QuoteOnce {
    cancels: bool,
    quoted: bool,
}

impl BacktestStrategy for QuoteOnce {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        if std::mem::replace(&mut self.quoted, true) {
            return None;
        }
        Some(Quote {
            bid_price: tick.mid_price() - dec!(1.5),
            bid_size: dec!(0.5),
            ask_price: tick.mid_price() + dec!(1.5),
            ask_size: dec!(0.5),
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn cancels_when_silent(&self) -> bool {
        self.cancels
    }
    fn reset(&mut self) {
        self.quoted = false;
    }
}

#[test]
fn test_experiment_runs_a_strategy_as_the_engine_does_when_it_goes_silent() {
    let window: Vec<MarketTick> = ticks()
        .into_iter()
        .filter(|t| (4 * DAY..8 * DAY).contains(&t.timestamp))
        .collect();
    for (cancels, name) in [(true, "cancels"), (false, "rests")] {
        let direct = market_maker_rs::backtest::BacktestEngine::new(
            experiment_config().execution.with_record_trades(true),
            QuoteOnce {
                cancels,
                quoted: false,
            },
            market_maker_rs::backtest::VecDataSource::new(window.clone()),
        )
        .run();
        let candidate = Candidate::model(name, BTreeMap::new(), move |_| {
            Ok(Box::new(QuoteOnce {
                cancels,
                quoted: false,
            }) as Box<dyn BacktestStrategy>)
        });
        let validated = Experiment::new(
            experiment_config(),
            ticks(),
            with(vec![candidate]),
            vec![immediate()],
        )
        .unwrap()
        .evaluate_validation();
        let metrics = completed(row(&validated.report().rows, name, "immediate"));
        assert_eq!(metrics.metrics.fills, direct.num_trades, "{name}");
        assert_eq!(
            metrics.metrics.attribution.net_pnl, direct.net_pnl,
            "{name}"
        );
        assert_eq!(
            direct.num_trades > 0,
            !cancels,
            "the fixture must tell them apart"
        );
    }
}

fn validation_interval(config: ExperimentConfig) -> (f64, f64, f64, u32) {
    let validated = Experiment::new(
        config,
        ticks(),
        with(vec![quoter("a", dec!(0.25))]),
        vec![immediate()],
    )
    .unwrap()
    .evaluate_validation();
    interval(completed(row(&validated.report().rows, "a", "immediate")).mean_daily_net)
}

#[test]
fn test_the_bootstrap_seed_drives_the_resampling_and_the_experiment_seed_does_not() {
    let base = validation_interval(experiment_config());
    let mut other = experiment_config();
    other.bootstrap.seed += 1;
    let moved = validation_interval(other);
    assert_eq!(base.0, moved.0, "the point estimate does not resample");
    assert!((base.1, base.2) != (moved.1, moved.2));
    // The immediate model ignores its seed, so this seed changes nothing here.
    let mut other = experiment_config();
    other.seed += 1;
    assert_eq!(base, validation_interval(other));
}

#[test]
fn test_validation_digest_moves_with_the_rows() {
    let digest = |half_spread: Decimal| {
        Experiment::new(
            experiment_config(),
            ticks(),
            with(vec![quoter("a", half_spread)]),
            vec![immediate()],
        )
        .unwrap()
        .evaluate_validation()
        .select(&["a"])
        .unwrap()
        .run_holdout()
        .manifest
        .validation_digest
        .unwrap()
    };
    let base = digest(dec!(0.25));
    assert!(!base.is_empty());
    assert_eq!(base, digest(dec!(0.25)));
    assert_ne!(base, digest(dec!(0.75)));
}
