//! A small reproducible comparison on seeded synthetic data.
//!
//! The ticks come from the queue-reactive simulator with hand set parameters, one
//! simulated second standing for one hour, so that a run spans days. They are not
//! market data, and nothing here says anything about performance on a market. The
//! example shows the protocol: chronological splits, one set of execution rules,
//! parameters from the training window alone, a selection made on validation, one
//! look at the holdout, and every row kept, including what fails. Hourly samples of
//! a fast book mean most moves jump past a resting quote, so the fills are mostly
//! adverse; that is a property of this setup, not a finding.
//!
//! A last row runs the fixed spread baseline on ticks from the Hawkes flow model.
//! That model holds its reference price fixed, so the steps between its segments
//! are an assumption of this example, not part of the model.
//!
//! Run with `cargo run --release --example baseline_evaluation -- <commit> <toolchain>`.
//! The commit and the toolchain go to the manifest as declared; they are not checked.

use std::collections::BTreeMap;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    BootstrapConfig, BootstrapOutcome, Candidate, EvaluationRow, Experiment, ExperimentConfig,
    MetricsSpec, Provenance, RowOutcome, Scenario, Splits,
};
use market_maker_rs::backtest::hawkes::{HawkesParameters, HawkesSimulator, fill_rate};
use market_maker_rs::backtest::queue_reactive::{
    QrEventKind, QrParameters, QrSimulator, QrState, QrStep, RateTable, UnsupportedPolicy,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestStrategy, ImmediateFillModel, MarketTick, ProbabilisticFillModel,
    QueuePositionFillModel, SimLatencyConfig, SimulatedFill,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy};
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::strategy::signal_quote::{
    FeatureKind, FeaturePolicy, FeatureValue, SignalConfig, UnavailablePolicy, compute_adjustment,
};
use market_maker_rs::types::error::{MMError, MMResult};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const HOUR: u64 = 3_600_000;
const DAY: u64 = 24 * HOUR;
const CAP: usize = 12;
const TICK: Decimal = dec!(0.5);
const YEAR_HOURS: f64 = 8760.0;

/// Thin queues that often empty, so that the reference price moves many times an
/// hour. Arrivals fall and departures rise with the queue.
fn rates(scale: f64) -> RateTable {
    (0..=CAP)
        .map(|n| {
            let size = n as f64;
            let limit = if n == CAP {
                0.0
            } else {
                scale * 1.0 / (1.0 + 0.3 * size)
            };
            let (cancel, market) = if n == 0 {
                (0.0, 0.0)
            } else {
                (scale * 0.3 * size, scale * 0.5)
            };
            (limit, cancel, market)
        })
        .collect()
}

fn best_offset(queues: &[u32]) -> Option<usize> {
    queues.iter().position(|q| *q > 0)
}

/// The top of the book, or nothing when a side is empty within the limits: that
/// sample is skipped rather than given an invented price.
fn tick_from(state: &QrState, timestamp: u64) -> Option<MarketTick> {
    let (bid, ask) = (best_offset(&state.bid)?, best_offset(&state.ask)?);
    let reference = Decimal::from(state.reference_ticks) * TICK;
    let half = TICK / dec!(2);
    Some(MarketTick::new(
        timestamp,
        reference - half - Decimal::from(bid) * TICK,
        Decimal::from(state.bid[bid]),
        reference + half + Decimal::from(ask) * TICK,
        Decimal::from(state.ask[ask]),
    ))
}

fn simulate(days: u64, seed: u64) -> MMResult<Vec<MarketTick>> {
    let params = QrParameters::new(
        TICK,
        0.6,
        0.0,
        vec![dec!(1), dec!(1)],
        vec![rates(40.0), rates(20.0)],
        vec![vec![true; CAP + 1]; 2],
        UnsupportedPolicy::Stop,
    )?;
    let mut simulator = QrSimulator::new(params, seed, 2000)?;
    let mut ticks = Vec::new();
    let mut hour = 0_u64;
    while hour < days * 24 {
        match simulator.step()? {
            QrStep::Event { .. } => {}
            other => {
                return Err(MMError::InvalidMarketState(format!(
                    "the simulator stopped: {other:?}"
                )));
            }
        }
        let state = simulator.state();
        while state.time_seconds >= hour as f64 && hour < days * 24 {
            if let Some(tick) = tick_from(&state, hour * HOUR) {
                ticks.push(tick);
            }
            hour += 1;
        }
    }
    Ok(ticks)
}

/// Absolute volatility per square root of a year, from hourly mids.
fn training_volatility(train: &[MarketTick]) -> MMResult<Decimal> {
    let mids: Vec<f64> = train
        .iter()
        .filter_map(|tick| tick.mid_price().to_f64())
        .collect();
    let changes: Vec<f64> = mids.windows(2).map(|w| w[1] - w[0]).collect();
    if changes.is_empty() {
        return Err(MMError::InvalidMarketState("no training moves".to_string()));
    }
    let n = changes.len() as f64;
    let mean = changes.iter().sum::<f64>() / n;
    let variance = changes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
    Decimal::from_f64((variance * YEAR_HOURS).sqrt())
        .filter(|sigma| *sigma > Decimal::ZERO)
        .ok_or_else(|| MMError::NumericalError("training volatility".to_string()))
}

const SIZE: Decimal = dec!(0.5);
const RISK_AVERSION: Decimal = dec!(0.01);
const ORDER_INTENSITY: Decimal = dec!(1.5);
const HORIZON_MS: u64 = 24 * HOUR;

fn quote(tick: &MarketTick, bid: Decimal, ask: Decimal) -> Quote {
    Quote {
        bid_price: bid,
        bid_size: SIZE,
        ask_price: ask,
        ask_size: SIZE,
        timestamp: tick.timestamp,
    }
}

struct Inventory {
    sigma: Decimal,
    signals: Option<SignalConfig>,
}

impl BacktestStrategy for Inventory {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        let (mut bid, mut ask) = calculate_optimal_quotes(
            tick.mid_price(),
            position.quantity,
            RISK_AVERSION,
            self.sigma,
            HORIZON_MS,
            ORDER_INTENSITY,
        )
        .ok()?;
        if let Some(signals) = &self.signals {
            let depth = tick.bid_size + tick.ask_size;
            let imbalance = (tick.bid_size - tick.ask_size).checked_div(depth)?;
            let features = [FeatureValue {
                kind: FeatureKind::Imbalance,
                value: imbalance,
                as_of_ms: tick.timestamp,
            }];
            let adjustment = compute_adjustment(&features, tick.timestamp, signals).ok()?;
            bid += adjustment.skew - adjustment.widen;
            ask += adjustment.skew + adjustment.widen;
        }
        Some(quote(tick, bid, ask))
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

struct Glft {
    config: GLFTConfig,
    sigma: Decimal,
}

impl BacktestStrategy for Glft {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        let quotes = GLFTStrategy::closed_form_quotes(
            tick.mid_price(),
            position.quantity,
            &self.config,
            self.sigma,
        )
        .ok()?;
        // At the inventory bound GLFT drops a side. The engine takes two sided
        // quotes, so it then sends no new quote.
        Some(quote(tick, quotes.bid?, quotes.ask?))
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

fn described(pairs: &[(&str, Decimal)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.normalize().to_string()))
        .collect()
}

fn avellaneda_stoikov() -> Candidate {
    Candidate::model(
        "avellaneda-stoikov",
        described(&[("risk_aversion", RISK_AVERSION), ("k", ORDER_INTENSITY)]),
        |train| {
            let sigma = training_volatility(train)?;
            Ok(Box::new(Inventory {
                sigma,
                signals: None,
            }) as Box<dyn BacktestStrategy>)
        },
    )
}

fn with_imbalance() -> Candidate {
    let weight = dec!(0.25);
    Candidate::model(
        "avellaneda-stoikov+imbalance",
        described(&[
            ("risk_aversion", RISK_AVERSION),
            ("k", ORDER_INTENSITY),
            ("imbalance_weight", weight),
        ]),
        move |train| {
            let sigma = training_volatility(train)?;
            let signals =
                SignalConfig::baseline(dec!(0.5), dec!(0), dec!(1)).with_imbalance(FeaturePolicy {
                    weight,
                    max_age_ms: 0,
                    when_unavailable: UnavailablePolicy::Neutral,
                });
            Ok(Box::new(Inventory {
                sigma,
                signals: Some(signals),
            }) as Box<dyn BacktestStrategy>)
        },
    )
}

fn glft() -> Candidate {
    Candidate::model(
        "glft",
        described(&[
            ("risk_aversion", RISK_AVERSION),
            ("k", ORDER_INTENSITY),
            ("a", dec!(1)),
        ]),
        |train| {
            let sigma = training_volatility(train)?;
            let config = GLFTConfig::new(
                RISK_AVERSION,
                ORDER_INTENSITY,
                dec!(1),
                20,
                SIZE,
                u64::MAX / 2,
                dec!(0.01),
            )?;
            Ok(Box::new(Glft { config, sigma }) as Box<dyn BacktestStrategy>)
        },
    )
}

/// One independent Hawkes segment per hour, and a seeded step of three ticks between
/// them. Returns the ticks and the coarse fill measure of the flow.
fn hawkes_ticks(days: u64, seed: u64) -> MMResult<(Vec<MarketTick>, Option<f64>)> {
    let weights = [[0.3, 0.05, 0.15], [0.1, 0.25, 0.05], [0.05, 0.05, 0.2]];
    let params = HawkesParameters::new(rates(1.0), vec![2.0], vec![weights])?;
    let half = TICK / dec!(2);
    let mut reference = Decimal::from(2000) * TICK;
    let mut word = seed | 1;
    let (mut ticks, mut segments) = (Vec::new(), Vec::new());
    for hour in 0..days * 24 {
        let (segment, _) = HawkesSimulator::new(params.clone(), seed + hour, 3)?.run(60.0)?;
        let queue = segment
            .events
            .iter()
            .fold(3_i64, |queue, event| match event.kind {
                QrEventKind::Limit => queue + 1,
                _ => queue - 1,
            });
        let size = Decimal::from(queue.max(1));
        ticks.push(MarketTick::new(
            hour * HOUR,
            reference - half,
            size,
            reference + half,
            size,
        ));
        segments.push(segment);
        word ^= word << 13;
        word ^= word >> 7;
        word ^= word << 17;
        let step = TICK * dec!(3);
        reference = if word & 1 == 0 {
            reference + step
        } else {
            reference - step
        };
    }
    Ok((ticks, fill_rate(&segments, CAP as u32, 10.0, 20.0)?))
}

fn print_row(row: &EvaluationRow) {
    match &row.outcome {
        RowOutcome::Completed(metrics) => {
            let m = &metrics.metrics;
            let interval = match metrics.mean_daily_net {
                BootstrapOutcome::Interval { lower, upper, .. } => {
                    format!("[{lower:.4}, {upper:.4}]")
                }
                other => format!("{other:?}"),
            };
            println!(
                "{:<30} {:<16} {:>6} {:>12} {:>10} {:>10} {:>10} {}",
                row.candidate,
                row.scenario,
                m.fills,
                m.attribution.net_pnl.round_dp(4),
                m.attribution.spread_capture.round_dp(4),
                m.attribution.inventory_pnl.round_dp(4),
                m.max_drawdown.round_dp(4),
                interval,
            );
        }
        RowOutcome::Failed { reason } => {
            println!(
                "{:<30} {:<16} FAILED: {reason}",
                row.candidate, row.scenario
            );
        }
        other => println!("{other:?}"),
    }
}

fn print_rows(rows: &[EvaluationRow]) {
    println!(
        "{:<30} {:<16} {:>6} {:>12} {:>10} {:>10} {:>10} mean daily net, 90 %",
        "candidate", "scenario", "fills", "net", "capture", "inventory", "drawdown"
    );
    for row in rows {
        print_row(row);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let provenance = Provenance {
        commit: args.next().unwrap_or_else(|| "undeclared".to_string()),
        toolchain: args.next().unwrap_or_else(|| "undeclared".to_string()),
    };
    let ticks = simulate(60, 20_261_105)?;
    let config = |provenance: Provenance| -> Result<ExperimentConfig, Box<dyn std::error::Error>> {
        Ok(ExperimentConfig {
            name: "baseline-evaluation-synthetic".to_string(),
            execution: BacktestConfig::new()
                .with_initial_capital(dec!(100000))
                .with_fee_rate(dec!(0.0001))
                .with_tick_size(dec!(0.01))
                .with_lot_size(dec!(0.01)),
            splits: Splits::new(0, 20 * DAY, 40 * DAY, 60 * DAY)?,
            metrics: MetricsSpec {
                horizons_ms: vec![HOUR, 4 * HOUR],
                max_mark_delay_ms: 0,
            },
            bootstrap: BootstrapConfig {
                replicates: 2000,
                mean_block_days: 3,
                confidence_percent: 90,
                seed: 1,
                min_days: 10,
            },
            seed: 2,
            provenance,
        })
    };
    let hawkes_config = config(provenance.clone())?;
    let config = config(provenance)?;
    let scenarios = vec![
        Scenario::new(
            "immediate",
            SimLatencyConfig::new(),
            BTreeMap::new(),
            |_| ImmediateFillModel::new(),
        ),
        Scenario::new(
            "double-fee",
            SimLatencyConfig::new(),
            BTreeMap::new(),
            |_| ImmediateFillModel::new(),
        )
        .with_fee_rate(dec!(0.0002)),
        Scenario::new(
            "one-hour-latency",
            SimLatencyConfig::new().with_submit_ms(HOUR),
            BTreeMap::new(),
            |_| ImmediateFillModel::new(),
        ),
        Scenario::new(
            "half-fill-chance",
            SimLatencyConfig::new(),
            described(&[("base_probability", dec!(0.5)), ("time_factor", dec!(1))]),
            |seed| ProbabilisticFillModel::new(dec!(0.5), dec!(0), dec!(1), seed),
        ),
        Scenario::new(
            "queue-position",
            SimLatencyConfig::new(),
            described(&[("fill_rate_per_ms", dec!(0.000001))]),
            |_| QueuePositionFillModel::new(dec!(0.000001)),
        ),
    ];
    let candidates = vec![
        Candidate::no_trade(),
        Candidate::fixed_spread(dec!(1), SIZE),
        avellaneda_stoikov(),
        with_imbalance(),
        glft(),
    ];
    let validated = Experiment::new(config, ticks, candidates, scenarios)?.evaluate_validation();
    println!("Synthetic data: no statement about any market.\n\nValidation:");
    print_rows(&validated.report().rows);

    // The model with the highest validation net result under the reference scenario,
    // a rule fixed here, before the holdout runs.
    let reference = validated.report().manifest.reference_scenario.clone();
    let chosen = validated
        .report()
        .rows
        .iter()
        .filter(|row| row.scenario == reference)
        .filter(|row| row.candidate.starts_with("avellaneda") || row.candidate == "glft")
        .filter_map(|row| match &row.outcome {
            RowOutcome::Completed(m) => {
                Some((m.metrics.attribution.net_pnl, row.candidate.clone()))
            }
            _ => None,
        })
        .max()
        .map(|(_, name)| name)
        .ok_or("no model completed on validation")?;
    println!("\nSelected on validation: {chosen}");
    let report = validated.select(&[chosen.as_str()])?.run_holdout();
    println!("\nHoldout, once:");
    print_rows(&report.holdout);
    println!("\nDid not reach the holdout: {:?}", report.not_selected);
    println!("\nManifest:\n{:#?}", report.manifest);

    let (ticks, measure) = hawkes_ticks(60, 404)?;
    println!("\nHawkes flow, coarse fill measure over 10 s: {measure:?}");
    let scenario = vec![Scenario::new(
        "immediate",
        SimLatencyConfig::new(),
        BTreeMap::new(),
        |_| ImmediateFillModel::new(),
    )];
    let baseline = vec![Candidate::fixed_spread(dec!(1), SIZE)];
    let validated =
        Experiment::new(hawkes_config, ticks, baseline, scenario)?.evaluate_validation();
    print_rows(&validated.report().rows);
    println!(
        "Hawkes manifest:
{:#?}",
        validated.report().manifest
    );
    Ok(())
}
