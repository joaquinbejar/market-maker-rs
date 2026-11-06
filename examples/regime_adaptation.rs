//! Fixed parameters against a simple causal adaptation, across regime changes and across two
//! flow simulators (#112).
//!
//! Everything is SYNTHETIC, and says nothing about a market. Flow comes from a queue-reactive or
//! a Hawkes simulator with made-up parameters; the mid follows that flow's market orders through
//! a linear impact, plus a diffusion, one simulated second standing for one minute. A regime
//! raises the flow, tilts it and scales the diffusion. Training holds no stress, and validation
//! and the holdout each hold one, at different times, so a change is unseen when it comes.
//!
//! No policy here is learned, and the neural simulator does not exist yet, so neither is compared.
//!
//! The signals are causal: a [`Detector`] reads only mids and trades at or before the tick, through
//! `VolatilityEstimator` and `VPINCalculator`, and the adaptive arm prices through
//! `AdaptiveSpreadCalculator`. Only the arm named [`ORACLE`] reads the latent regime, as a reference
//! that no policy could realize, and not a bound: it scales its spread by a declared factor. The report's regime split reads it too, after the run.
//!
//! The [`GoNoGo`] rule is fixed here, before the holdout runs, and is applied per simulator and
//! fill rule: adaptation goes only if it holds on every group. The arm prices from the ratio and
//! the flow, not from the alarm, so delay and false alarms measure a signal beside its decisions.
//! A volatility that fails to compute leaves the last ratio in place.
//!
//! Run with `cargo run --release --example regime_adaptation -- <commit> <toolchain>`.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use market_maker_rs::Decimal;
use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
use market_maker_rs::analytics::vpin::{VPINCalculator, VPINConfig};
use market_maker_rs::backtest::evaluation::{
    BootstrapConfig, Candidate, EvaluationRow, Experiment, ExperimentConfig, HoldoutReport,
    MetricsSpec, Provenance, RowOutcome, Scenario, Splits,
};
use market_maker_rs::backtest::hawkes::{HawkesParameters, HawkesSimulator};
use market_maker_rs::backtest::queue_reactive::{
    QrEventKind, QrParameters, QrSide, QrSimulator, QrStep, RateTable, UnsupportedPolicy,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestStrategy, ImmediateFillModel, MarketTick, ProbabilisticFillModel,
    SimLatencyConfig, SimulatedFill,
};
use market_maker_rs::dec;
use market_maker_rs::market_state::volatility::VolatilityEstimator;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::adaptive_spread::{
    AdaptiveSpreadCalculator, AdaptiveSpreadConfig, OrderBookImbalance, TradeFlowImbalance,
};
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const MINUTE: u64 = 60_000;
const DAY_MINUTES: u64 = 24 * 60;
const DAY: u64 = DAY_MINUTES * MINUTE;
const VALIDATION_DAY: u64 = 2;
const HOLDOUT_DAY: u64 = 4;
const END_DAY: u64 = 6;
/// Stress in validation, then in the holdout, in days: differently placed on purpose.
const STRESSES: [(f64, f64); 2] = [(2.5, 3.0), (5.0, 5.75)];
/// The reported paths, fixed in the commit before their one run. Paths 1 to 10 were seen while
/// this example was developed, and are not evidence.
const PATHS: std::ops::Range<u64> = 101..111;
const CAP: usize = 20;
const IMPACT: f64 = 0.01;
const DIFFUSION: f64 = 0.02;
const STRESS_DIFFUSION: f64 = 3.0;
const STRESS_FLOW: f64 = 2.0;
const STRESS_TILT: f64 = 1.5;
const SIZE: Decimal = dec!(1);
const FEE: Decimal = dec!(0.0001);
/// The half spread, in training's standard deviations of a minute's mid change.
const HALF_SPREAD_DEVS: f64 = 1.5;
const LOOKBACK: usize = 30;
const EWMA_LAMBDA: Decimal = dec!(0.9);
const FLOW_WINDOW_MS: u64 = 10 * MINUTE;
const INTERVAL_DRAWS: usize = 2_000;

pub const FIXED: &str = "fixed";
pub const ADAPTIVE: &str = "adaptive";
pub const ORACLE: &str = "oracle-labelled";
const TRANSFER: &str = "adaptive-other-simulator";

/// Fixed before the holdout: what adaptation must show on a simulator.
#[derive(Debug, Clone, Copy)]
struct GoNoGo {
    /// Most minutes from a change to the first alarm, as a mean over paths.
    max_delay_minutes: f64,
    /// Most alarms a calm day, as a mean over paths.
    max_false_alarms_per_day: f64,
}

const RULE: GoNoGo = GoNoGo {
    max_delay_minutes: 60.0,
    max_false_alarms_per_day: 2.0,
};

/// The detector's alarm: on above the on ratio, off below the off ratio, and so counted once.
const ALARM_ON: Decimal = dec!(1.8);
const ALARM_OFF: Decimal = dec!(1.3);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Flow {
    QueueReactive,
    Hawkes,
}

const FLOWS: [Flow; 2] = [Flow::QueueReactive, Flow::Hawkes];

fn stressed_at(timestamp: u64) -> bool {
    let day = timestamp as f64 / DAY as f64;
    STRESSES.iter().any(|(from, to)| *from <= day && day < *to)
}

fn qr_rates(scale: f64) -> RateTable {
    (0..=CAP)
        .map(|n| {
            let size = n as f64;
            let limit = if n == CAP { 0.0 } else { scale * 1.5 };
            let (cancel, market) = if n == 0 {
                (0.0, 0.0)
            } else {
                (scale * 0.05 * size, scale * 0.3)
            };
            (limit, cancel, market)
        })
        .collect()
}

fn hawkes_params(market: f64) -> MMResult<HawkesParameters> {
    let rates = (0..=CAP)
        .map(|n| {
            let size = n as f64;
            (
                if n == CAP { 0.0 } else { 1.5 },
                if n == 0 { 0.0 } else { 0.05 * size },
                if n == 0 { 0.0 } else { 0.3 * market },
            )
        })
        .collect();
    let weights = [[0.10, 0.05, 0.15], [0.05, 0.10, 0.05], [0.02, 0.02, 0.30]];
    HawkesParameters::new(rates, vec![2.0], vec![weights])
}

/// A market order: seconds into its segment, and whether the aggressor bought.
type Order = (f64, bool);

/// The market orders of one segment under one regime.
fn segment_orders(flow: Flow, stressed: bool, seconds: f64, seed: u64) -> MMResult<Vec<Order>> {
    let (scale, tilt) = if stressed {
        (STRESS_FLOW, STRESS_TILT)
    } else {
        (1.0, 1.0)
    };
    match flow {
        Flow::Hawkes => {
            let side = |market: f64, seed: u64, bought: bool| -> MMResult<Vec<Order>> {
                let mut sim = HawkesSimulator::new(hawkes_params(market)?, seed, 10)?;
                let (segment, stop) = sim.run(seconds)?;
                if let Some(stop) = stop {
                    return Err(MMError::InvalidMarketState(format!("{stop:?}")));
                }
                Ok(segment
                    .events
                    .iter()
                    .filter(|e| e.kind == QrEventKind::Market)
                    .map(|e| (e.time_seconds, bought))
                    .collect())
            };
            let mut orders = side(scale, seed.wrapping_mul(2), false)?;
            orders.extend(side(scale * tilt, seed.wrapping_mul(2) + 1, true)?);
            orders.sort_by(|a, b| a.0.total_cmp(&b.0));
            Ok(orders)
        }
        Flow::QueueReactive => {
            let params = QrParameters::new(
                dec!(0.01),
                0.0,
                0.0,
                vec![dec!(1)],
                vec![qr_rates(scale)],
                vec![vec![true; CAP + 1]],
                UnsupportedPolicy::Stop,
            )?;
            let mut sim = QrSimulator::new(params, seed, 10_000)?;
            let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9) | 1);
            let keep = 1.0 / tilt;
            let mut orders = Vec::new();
            while sim.state().time_seconds < seconds {
                match sim.step()? {
                    QrStep::Event { observation, .. } => {
                        if observation.kind == QrEventKind::Market {
                            // The simulator's rates are symmetric, so the tilt thins one side.
                            let bought = observation.side == QrSide::Ask;
                            if bought || noise.unit() < keep {
                                orders.push((sim.state().time_seconds, bought));
                            }
                        }
                    }
                    other => {
                        return Err(MMError::InvalidMarketState(format!("{other:?}")));
                    }
                }
            }
            Ok(orders)
        }
    }
}

/// A small generator for the diffusion, apart from the flow's.
struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (1.0 - self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// One path: its ticks, and the trades a causal reader may take up to a tick's time.
#[derive(Debug, Clone)]
struct Path {
    ticks: Vec<MarketTick>,
    trades: Vec<Trade>,
}

fn price(value: f64) -> MMResult<Decimal> {
    Decimal::from_f64(value)
        .map(|d| d.round_dp(2))
        .ok_or_else(|| MMError::NumericalError(format!("price {value}")))
}

/// Builds a path a minute at a time. Each run of one regime is a new simulator.
fn simulate(flow: Flow, seed: u64) -> MMResult<Path> {
    let minutes = END_DAY * DAY_MINUTES;
    let mut noise = Noise(seed.wrapping_mul(0x5851_F42D).wrapping_add(3));
    let (mut mid, mut minute, mut ticks, mut trades) = (100.0_f64, 0_u64, Vec::new(), Vec::new());
    let mut segment = 0_u64;
    while minute < minutes {
        let stressed = stressed_at(minute * MINUTE);
        let mut end = minute + 1;
        while end < minutes && stressed_at(end * MINUTE) == stressed {
            end += 1;
        }
        let seconds = (end - minute) as f64;
        let orders = segment_orders(flow, stressed, seconds, seed * 1_000 + segment)?;
        let mut orders = orders.into_iter().peekable();
        for at in minute..end {
            let until = (at + 1 - minute) as f64;
            let sigma = if stressed {
                DIFFUSION * STRESS_DIFFUSION
            } else {
                DIFFUSION
            };
            mid += sigma * noise.normal();
            let stamp = at * MINUTE;
            let half = dec!(0.01);
            let level = price(mid.max(1.0))?;
            ticks.push(MarketTick::new(
                stamp,
                level - half,
                dec!(5),
                level + half,
                dec!(5),
            ));
            while let Some((_, bought)) = orders.next_if(|(t, _)| *t < until) {
                mid += if bought { IMPACT } else { -IMPACT };
                let side = if bought {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                };
                // Stamped at the minute's end: it is known from the next tick on.
                trades.push(Trade::new(
                    price(mid.max(1.0))?,
                    dec!(1),
                    side,
                    stamp + MINUTE,
                ));
            }
        }
        minute = end;
        segment += 1;
    }
    Ok(Path { ticks, trades })
}

/// The standard deviation of a minute's log return, by `VolatilityEstimator`'s EWMA, without
/// annualizing.
fn recent_volatility(mids: &[Decimal]) -> MMResult<Decimal> {
    VolatilityEstimator::with_annualization_factor(Decimal::ONE).calculate_ewma(mids, EWMA_LAMBDA)
}

fn training_volatility(train: &[MarketTick]) -> MMResult<Decimal> {
    let mids: Vec<Decimal> = train.iter().map(MarketTick::mid_price).collect();
    VolatilityEstimator::with_annualization_factor(Decimal::ONE).calculate_simple(&mids)
}

/// What the causal signals say at each tick. It reads nothing past the tick.
struct Detector {
    baseline: Decimal,
    mids: VecDeque<Decimal>,
    trades: Rc<Vec<Trade>>,
    next_trade: usize,
    vpin: VPINCalculator,
    recent: VecDeque<Trade>,
    alarm: bool,
    alarms: Vec<u64>,
    ratio: Decimal,
}

impl Detector {
    fn new(baseline: Decimal, trades: Rc<Vec<Trade>>) -> MMResult<Self> {
        Ok(Self {
            baseline,
            mids: VecDeque::new(),
            trades,
            next_trade: 0,
            vpin: VPINCalculator::new(VPINConfig::new(dec!(50), 20, dec!(0.7))?),
            recent: VecDeque::new(),
            alarm: false,
            alarms: Vec::new(),
            ratio: Decimal::ONE,
        })
    }

    fn observe(&mut self, tick: &MarketTick) {
        self.mids.push_back(tick.mid_price());
        if self.mids.len() > LOOKBACK {
            self.mids.pop_front();
        }
        while let Some(trade) = self
            .trades
            .get(self.next_trade)
            .filter(|t| t.timestamp <= tick.timestamp)
        {
            let _ = self.vpin.add_trade(trade);
            self.recent.push_back(trade.clone());
            self.next_trade += 1;
        }
        while self
            .recent
            .front()
            .is_some_and(|t| t.timestamp + FLOW_WINDOW_MS <= tick.timestamp)
        {
            self.recent.pop_front();
        }
        let mids: Vec<Decimal> = self.mids.iter().copied().collect();
        if mids.len() < LOOKBACK {
            return;
        }
        if let Ok(volatility) = recent_volatility(&mids)
            && let Some(ratio) = volatility.checked_div(self.baseline)
        {
            self.ratio = ratio;
        }
        let toxic = self.vpin.is_toxic();
        if !self.alarm && (self.ratio >= ALARM_ON || toxic) {
            self.alarm = true;
            self.alarms.push(tick.timestamp);
        } else if self.alarm && self.ratio < ALARM_OFF && !toxic {
            self.alarm = false;
        }
    }

    fn flow(&self) -> TradeFlowImbalance {
        let (mut bought, mut sold) = (Decimal::ZERO, Decimal::ZERO);
        for trade in &self.recent {
            match trade.side {
                TradeSide::Buy => bought += trade.size,
                TradeSide::Sell => sold += trade.size,
            }
        }
        TradeFlowImbalance::new(bought, sold)
    }
}

fn two_sided(tick: &MarketTick, half: Decimal, skew: Decimal) -> Quote {
    let mid = tick.mid_price();
    Quote {
        bid_price: (mid - half + skew).round_dp(2),
        bid_size: SIZE,
        ask_price: (mid + half + skew).round_dp(2),
        ask_size: SIZE,
        timestamp: tick.timestamp,
    }
}

#[derive(Debug, Clone, Copy)]
enum Pricing {
    Fixed,
    Adaptive,
    Oracle,
}

struct Quoter {
    pricing: Pricing,
    half: Decimal,
    detector: Detector,
    calculator: AdaptiveSpreadCalculator,
    log: Rc<RefCell<Vec<u64>>>,
}

impl BacktestStrategy for Quoter {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        self.detector.observe(tick);
        *self.log.borrow_mut() = self.detector.alarms.clone();
        match self.pricing {
            Pricing::Fixed => Some(two_sided(tick, self.half, Decimal::ZERO)),
            // The latent regime: only this labelled arm reads it.
            Pricing::Oracle => {
                let scale = if stressed_at(tick.timestamp) {
                    Decimal::from_f64(STRESS_DIFFUSION)?
                } else {
                    Decimal::ONE
                };
                Some(two_sided(tick, self.half * scale, Decimal::ZERO))
            }
            Pricing::Adaptive => {
                let book = OrderBookImbalance::new(tick.bid_size, tick.ask_size, 1);
                let flow = self.detector.flow();
                let spread = self
                    .calculator
                    .calculate_spread_with_volatility(
                        &book,
                        Some(&flow),
                        self.detector.ratio,
                        Decimal::ONE,
                    )
                    .ok()?;
                let mid = tick.mid_price();
                Some(Quote {
                    bid_price: spread.bid_price(mid).ok()?.round_dp(2),
                    bid_size: SIZE,
                    ask_price: spread.ask_price(mid).ok()?.round_dp(2),
                    ask_size: SIZE,
                    timestamp: tick.timestamp,
                })
            }
        }
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn reset(&mut self) {}
}

fn described(pairs: &[(&str, String)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

/// An arm's half spread in price, from the training window's return volatility and last mid.
fn half_of(baseline: Decimal, mid: Decimal) -> MMResult<Decimal> {
    let devs = Decimal::from_f64(HALF_SPREAD_DEVS)
        .ok_or_else(|| MMError::NumericalError("half spread".to_string()))?;
    Ok((baseline * mid * devs).round_dp(2).max(dec!(0.01)))
}

/// What an arm calibrates on: another simulator's training window, or its own.
fn calibration_window<'a>(
    source: Option<&'a [MarketTick]>,
    train: &'a [MarketTick],
) -> &'a [MarketTick] {
    source.unwrap_or(train)
}

/// `source` holds what to calibrate on: `None` is the arm's own training window.
fn arm(
    name: &str,
    pricing: Pricing,
    trades: &Rc<Vec<Trade>>,
    source: Option<Vec<MarketTick>>,
    log: Rc<RefCell<Vec<u64>>>,
) -> Candidate {
    let trades = Rc::clone(trades);
    Candidate::model(
        name,
        described(&[("half_spread_devs", HALF_SPREAD_DEVS.to_string())]),
        move |train| {
            let window = calibration_window(source.as_deref(), train);
            let baseline = training_volatility(window)?;
            let mid = window.last().map_or(dec!(100), MarketTick::mid_price);
            let half = half_of(baseline, mid)?;
            Ok(Box::new(Quoter {
                pricing,
                half,
                detector: Detector::new(baseline, Rc::clone(&trades))?,
                calculator: AdaptiveSpreadCalculator::new(AdaptiveSpreadConfig::new(
                    half * dec!(2) / mid,
                    dec!(2),
                    dec!(0),
                    dec!(0.2),
                )?),
                log: Rc::clone(&log),
            }))
        },
    )
}

fn config(provenance: &Provenance, seed: u64) -> MMResult<ExperimentConfig> {
    let to_error = |e: market_maker_rs::backtest::evaluation::EvaluationError| {
        MMError::InvalidConfiguration(e.to_string())
    };
    Ok(ExperimentConfig {
        name: format!("regime-adaptation-{seed}"),
        execution: BacktestConfig::new()
            .with_initial_capital(dec!(100000))
            .with_fee_rate(FEE)
            .with_tick_size(dec!(0.01))
            .with_lot_size(dec!(1)),
        splits: Splits::new(0, VALIDATION_DAY * DAY, HOLDOUT_DAY * DAY, END_DAY * DAY)
            .map_err(to_error)?,
        metrics: MetricsSpec {
            horizons_ms: vec![5 * MINUTE],
            max_mark_delay_ms: 0,
        },
        bootstrap: BootstrapConfig {
            replicates: 200,
            mean_block_days: 1,
            confidence_percent: 90,
            seed: 11,
            min_days: 2,
        },
        seed,
        provenance: provenance.clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FillRule {
    Immediate,
    HalfChance,
}

const RULES: [FillRule; 2] = [FillRule::Immediate, FillRule::HalfChance];

fn scenario(rule: FillRule) -> Scenario {
    match rule {
        FillRule::Immediate => Scenario::new(
            "immediate",
            SimLatencyConfig::new(),
            BTreeMap::new(),
            |_| ImmediateFillModel::new(),
        ),
        FillRule::HalfChance => Scenario::new(
            "half-fill-chance",
            SimLatencyConfig::new(),
            described(&[("base_probability", "0.5".to_string())]),
            |seed| ProbabilisticFillModel::new(dec!(0.5), dec!(0), dec!(1), seed),
        ),
    }
}

/// The holdout of one path under one fill rule, with the adaptive arm's alarms.
fn run_path(
    flow: Flow,
    seed: u64,
    rule: FillRule,
    provenance: &Provenance,
) -> MMResult<(HoldoutReport, Vec<u64>)> {
    let path = simulate(flow, seed)?;
    let other = match flow {
        Flow::QueueReactive => Flow::Hawkes,
        Flow::Hawkes => Flow::QueueReactive,
    };
    let other_train: Vec<MarketTick> = simulate(other, seed)?
        .ticks
        .into_iter()
        .filter(|t| t.timestamp < VALIDATION_DAY * DAY)
        .collect();
    let trades = Rc::new(path.trades);
    let alarms = Rc::new(RefCell::new(Vec::new()));
    let ignored = || Rc::new(RefCell::new(Vec::new()));
    let candidates = vec![
        Candidate::no_trade(),
        Candidate::fixed_spread(dec!(0.05), SIZE),
        arm(FIXED, Pricing::Fixed, &trades, None, ignored()),
        arm(
            ADAPTIVE,
            Pricing::Adaptive,
            &trades,
            None,
            Rc::clone(&alarms),
        ),
        arm(ORACLE, Pricing::Oracle, &trades, None, ignored()),
        arm(
            TRANSFER,
            Pricing::Adaptive,
            &trades,
            Some(other_train),
            ignored(),
        ),
    ];
    let to_error = |e: market_maker_rs::backtest::evaluation::EvaluationError| {
        MMError::InvalidConfiguration(e.to_string())
    };
    let report = Experiment::new(
        config(provenance, seed)?,
        path.ticks,
        candidates,
        vec![scenario(rule)],
    )
    .map_err(to_error)?
    .with_regimes(
        "the latent regime, read for the report only, after the run",
        || {
            Box::new(|mark| {
                if stressed_at(mark.timestamp_ms) {
                    "stress".to_string()
                } else {
                    "calm".to_string()
                }
            })
        },
    )
    .evaluate_validation()
    .select(&[FIXED, ADAPTIVE, ORACLE, TRANSFER])
    .map_err(to_error)?
    .run_holdout();
    let seen = alarms.borrow().clone();
    Ok((report, seen))
}

/// Minutes from the holdout's change to the first alarm after it, if any.
fn delay_minutes(alarms: &[u64]) -> Option<f64> {
    let start = (STRESSES[1].0 * DAY as f64) as u64;
    let end = (STRESSES[1].1 * DAY as f64) as u64;
    alarms
        .iter()
        .find(|t| **t >= start && **t < end)
        .map(|t| (t - start) as f64 / MINUTE as f64)
}

/// Alarms per calm day of the holdout.
fn false_alarms_per_day(alarms: &[u64]) -> f64 {
    let calm = alarms
        .iter()
        .filter(|t| **t >= HOLDOUT_DAY * DAY && !stressed_at(**t))
        .count();
    let stress_days = STRESSES[1].1 - STRESSES[1].0;
    calm as f64 / ((END_DAY - HOLDOUT_DAY) as f64 - stress_days)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Measure {
    net: f64,
    drawdown: f64,
    stress_net: f64,
    calm_net: f64,
    fills: f64,
}

fn number(value: Decimal) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn measure_of(row: &EvaluationRow) -> Result<Measure, String> {
    match &row.outcome {
        RowOutcome::Completed(metrics) => {
            let regime = |name: &str| {
                metrics
                    .regimes
                    .get(name)
                    .map_or(0.0, |b| number(b.attribution.net_pnl))
            };
            Ok(Measure {
                net: number(metrics.metrics.attribution.net_pnl),
                drawdown: number(metrics.metrics.max_drawdown),
                stress_net: regime("stress"),
                calm_net: regime("calm"),
                fills: metrics.metrics.fills as f64,
            })
        }
        RowOutcome::Failed { reason } => Err(reason.clone()),
        _ => Err("unknown row".to_string()),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The mean and a 90% interval, resampling paths.
fn interval(values: &[f64]) -> (f64, f64, f64) {
    let mut noise = Noise(77);
    let mut means: Vec<f64> = (0..INTERVAL_DRAWS)
        .map(|_| {
            let sample: Vec<f64> = values
                .iter()
                .map(|_| values[(noise.unit() * values.len() as f64) as usize])
                .collect();
            mean(&sample)
        })
        .collect();
    means.sort_by(f64::total_cmp);
    (
        mean(values),
        means[INTERVAL_DRAWS / 20],
        means[INTERVAL_DRAWS - 1 - INTERVAL_DRAWS / 20],
    )
}

/// An arm minus the fixed arm, path by path.
fn paired(arm: &[Measure], fixed: &[Measure], f: fn(&Measure) -> f64) -> Vec<f64> {
    arm.iter().zip(fixed).map(|(a, b)| f(a) - f(b)).collect()
}

fn shown((m, low, high): (f64, f64, f64)) -> String {
    format!("{m:.2} [{low:.2}, {high:.2}]")
}

/// The rule's verdict for one simulator.
fn verdict(
    paired: (f64, f64, f64),
    drawdown: (f64, f64, f64),
    delay: f64,
    missed: u32,
    false_alarms: f64,
) -> bool {
    missed == 0
        && paired.1 > 0.0
        && drawdown.2 <= 0.0
        && delay <= RULE.max_delay_minutes
        && false_alarms <= RULE.max_false_alarms_per_day
}

fn arm_row(flow: Flow, rule: FillRule, name: &str, measures: &[Measure]) -> String {
    let of = |f: fn(&Measure) -> f64| shown(interval(&measures.iter().map(f).collect::<Vec<_>>()));
    format!(
        "| {flow:?} | {rule:?} | {name} | {} | {} | {} | {} | {} |",
        of(|m| m.net),
        of(|m| m.stress_net),
        of(|m| m.calm_net),
        of(|m| m.drawdown),
        of(|m| m.fills)
    )
}

/// `gaps` holds the adaptive net, the adaptive drawdown, the oracle's net and the transfer's.
fn summary_row(
    flow: Flow,
    rule: FillRule,
    gaps: [(f64, f64, f64); 4],
    delay: f64,
    missed: u32,
    alarms_rate: f64,
) -> (String, bool) {
    let [net, drawdown, oracle, transfer] = gaps;
    let go = verdict(net, drawdown, delay, missed, alarms_rate);
    (
        format!(
            "| {flow:?} | {rule:?} | {} | {} | {} | {} | {delay:.1} | {missed} | {alarms_rate:.2} | {} |",
            shown(net),
            shown(drawdown),
            shown(oracle),
            shown(transfer),
            if go { "go" } else { "no-go" },
        ),
        go,
    )
}

fn main() -> MMResult<()> {
    let mut args = std::env::args().skip(1);
    let provenance = Provenance {
        commit: args.next().unwrap_or_else(|| "undeclared".to_string()),
        toolchain: args.next().unwrap_or_else(|| "undeclared".to_string()),
    };
    println!(
        "SYNTHETIC. {} paths per simulator; stresses at days {STRESSES:?}; go needs the paired net's \
         interval above zero, a drawdown no worse, delay <= {} min and <= {} false alarms a calm day.",
        PATHS.end - PATHS.start,
        RULE.max_delay_minutes,
        RULE.max_false_alarms_per_day
    );
    let mut go_all = true;
    println!(
        "\n| flow | fills | arm | net | stress net | calm net | drawdown | fills |\n| --- | --- | --- | --- | --- | --- | --- | --- |"
    );
    let mut summaries = Vec::new();
    for flow in FLOWS {
        for rule in RULES {
            let mut rows: BTreeMap<String, Vec<Measure>> = BTreeMap::new();
            let (mut delays, mut missed, mut false_alarms) = (Vec::new(), 0_u32, Vec::new());
            for seed in PATHS {
                let (report, alarms) = run_path(flow, seed, rule, &provenance)?;
                for row in &report.holdout {
                    let measure = measure_of(row).map_err(MMError::InvalidMarketState)?;
                    rows.entry(row.candidate.clone()).or_default().push(measure);
                }
                println!("alarms {flow:?} {rule:?} path {seed}: {alarms:?}");
                match delay_minutes(&alarms) {
                    Some(delay) => delays.push(delay),
                    None => missed += 1,
                }
                false_alarms.push(false_alarms_per_day(&alarms));
            }
            for (name, measures) in &rows {
                println!("{}", arm_row(flow, rule, name, measures));
            }
            let gap =
                |arm: &str, f: fn(&Measure) -> f64| interval(&paired(&rows[arm], &rows[FIXED], f));
            let (net, drawdown) = (gap(ADAPTIVE, |m| m.net), gap(ADAPTIVE, |m| m.drawdown));
            let delay = if delays.is_empty() {
                f64::INFINITY
            } else {
                mean(&delays)
            };
            let alarms_rate = mean(&false_alarms);
            let (line, go) = summary_row(
                flow,
                rule,
                [
                    net,
                    drawdown,
                    gap(ORACLE, |m| m.net),
                    gap(TRANSFER, |m| m.net),
                ],
                delay,
                missed,
                alarms_rate,
            );
            go_all &= go;
            summaries.push(line);
        }
    }
    println!(
        "\nAgainst the fixed arm, paired by path:\n| flow | fills | adaptive net | adaptive drawdown | oracle net | other simulator net | delay min | missed | false alarms a day | rule |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    );
    for line in summaries {
        println!("{line}");
    }
    println!(
        "\nAdaptation overall: {}",
        if go_all { "GO" } else { "NO-GO" }
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn detector(trades: Vec<Trade>) -> Detector {
        // A return volatility of 2e-4 a minute, as on a mid of 100 moved by 0.02.
        Detector::new(dec!(0.0002), Rc::new(trades)).unwrap()
    }

    fn at(minute: u64, mid: Decimal) -> MarketTick {
        MarketTick::new(
            minute * MINUTE,
            mid - dec!(0.01),
            dec!(5),
            mid + dec!(0.01),
            dec!(5),
        )
    }

    #[test]
    fn test_the_regime_schedule_is_as_declared_and_training_holds_no_stress() {
        assert!(!stressed_at(0));
        assert!(!stressed_at(VALIDATION_DAY * DAY - 1));
        assert!(stressed_at(DAY * 5 / 2));
        assert!(!stressed_at(DAY * 3));
        assert!(stressed_at(DAY * 5));
        assert!(!stressed_at(DAY * 23 / 4));
    }

    #[test]
    fn test_a_path_reproduces_by_seed_and_each_flow_gives_orders() {
        for flow in FLOWS {
            let one = segment_orders(flow, false, 200.0, 7).unwrap();
            assert_eq!(one, segment_orders(flow, false, 200.0, 7).unwrap());
            assert_ne!(one, segment_orders(flow, false, 200.0, 8).unwrap());
            assert!(!one.is_empty());
        }
    }

    #[test]
    fn test_a_stress_raises_the_flow_and_tilts_it_to_buyers() {
        for flow in FLOWS {
            let count = |stressed| {
                let orders: Vec<Order> = (0..10)
                    .flat_map(|s| segment_orders(flow, stressed, 500.0, s + 1).unwrap())
                    .collect();
                (orders.len(), orders.iter().filter(|o| o.1).count())
            };
            let (calm, stress) = (count(false), count(true));
            assert!(stress.0 > calm.0, "{flow:?} {calm:?} {stress:?}");
            assert!(
                stress.1 as f64 / stress.0 as f64 > 0.55,
                "{flow:?} {stress:?}"
            );
            assert!(
                (calm.1 as f64 / calm.0 as f64 - 0.5).abs() < 0.05,
                "{flow:?} {calm:?}"
            );
        }
    }

    #[test]
    fn test_the_detector_reads_no_trade_after_the_tick() {
        let trades = vec![
            Trade::new(dec!(100), dec!(1), TradeSide::Buy, 0),
            Trade::new(dec!(100), dec!(1), TradeSide::Buy, MINUTE),
        ];
        let mut d = detector(trades);
        d.observe(&at(0, dec!(100)));
        assert_eq!(d.flow().buy_volume, dec!(1));
        d.observe(&at(1, dec!(100)));
        assert_eq!(d.flow().buy_volume, dec!(2));
    }

    #[test]
    fn test_the_flow_window_drops_old_trades() {
        let mut d = detector(vec![Trade::new(dec!(100), dec!(1), TradeSide::Sell, 0)]);
        d.observe(&at(0, dec!(100)));
        assert_eq!(d.flow().sell_volume, dec!(1));
        d.observe(&at(10, dec!(100)));
        assert_eq!(d.flow().sell_volume, dec!(0));
    }

    /// Mids moving by `step`, alternately, for `count` minutes from `from`.
    fn feed(d: &mut Detector, from: u64, count: u64, step: Decimal) {
        for minute in from..from + count {
            let mid = if minute % 2 == 0 {
                dec!(100)
            } else {
                dec!(100) + step
            };
            d.observe(&at(minute, mid));
        }
    }

    #[test]
    fn test_an_alarm_comes_on_a_volatility_rise_once_and_goes_with_it() {
        let mut d = detector(vec![]);
        feed(&mut d, 0, 40, dec!(0.02));
        let calm = d.ratio;
        assert!(d.alarms.is_empty(), "{calm}");
        feed(&mut d, 40, 40, dec!(0.5));
        assert!(d.ratio > calm);
        assert_eq!(d.alarms.len(), 1);
        assert!(d.alarm);
        feed(&mut d, 80, 200, dec!(0.0));
        assert!(!d.alarm, "{}", d.ratio);
        assert_eq!(d.alarms.len(), 1);
    }

    #[test]
    fn test_delay_and_false_alarms_by_hand() {
        let start = (STRESSES[1].0 * DAY as f64) as u64;
        assert_eq!(delay_minutes(&[start + 5 * MINUTE]), Some(5.0));
        // One before the change is no detection, nor is one after it ended.
        assert_eq!(delay_minutes(&[start - MINUTE]), None);
        let end = (STRESSES[1].1 * DAY as f64) as u64;
        assert_eq!(delay_minutes(&[end]), None);
        // 2 days, 0.75 of stress: 1.25 calm days.
        let calm = HOLDOUT_DAY * DAY + MINUTE;
        assert_eq!(false_alarms_per_day(&[calm]), 1.0 / 1.25);
        // Validation's alarms and a true detection do not count.
        assert_eq!(false_alarms_per_day(&[DAY, start + MINUTE]), 0.0);
    }

    #[test]
    fn test_the_rule_needs_all_of_its_terms() {
        let good = (1.0, 0.5, 2.0);
        let ok_drawdown = (-1.0, -2.0, -0.1);
        assert!(verdict(good, ok_drawdown, 10.0, 0, 1.0));
        assert!(!verdict(good, ok_drawdown, 10.0, 1, 1.0));
        assert!(!verdict((1.0, -0.1, 2.0), ok_drawdown, 10.0, 0, 1.0));
        assert!(!verdict(good, (0.0, -1.0, 0.5), 10.0, 0, 1.0));
        assert!(!verdict(good, ok_drawdown, 61.0, 0, 1.0));
        assert!(!verdict(good, ok_drawdown, 10.0, 0, 2.5));
    }

    #[test]
    fn test_only_the_oracle_prices_from_the_regime() {
        let trades = Rc::new(vec![]);
        let quoter = |pricing| Quoter {
            pricing,
            half: dec!(0.03),
            detector: Detector::new(dec!(0.02), Rc::clone(&trades)).unwrap(),
            calculator: AdaptiveSpreadCalculator::new(
                AdaptiveSpreadConfig::new(dec!(0.0006), dec!(2), dec!(0), dec!(0.2)).unwrap(),
            ),
            log: Rc::new(RefCell::new(vec![])),
        };
        let position = InventoryPosition::new();
        let width = |mut q: Quoter, day: u64| {
            let tick = at(day * DAY_MINUTES, dec!(100));
            let quote = q.on_tick(&tick, &position).unwrap();
            quote.ask_price - quote.bid_price
        };
        // Day 5 is stress, day 1 is calm, with nothing observed that differs.
        assert_eq!(
            width(quoter(Pricing::Fixed), 5),
            width(quoter(Pricing::Fixed), 1)
        );
        assert_eq!(
            width(quoter(Pricing::Adaptive), 5),
            width(quoter(Pricing::Adaptive), 1)
        );
        assert!(width(quoter(Pricing::Oracle), 5) > width(quoter(Pricing::Oracle), 1));
    }

    #[test]
    fn test_the_half_spread_is_in_price_from_a_return_volatility() {
        // 1.5 * 2e-4 of a mid of 100, and never under a tick.
        assert_eq!(half_of(dec!(0.0002), dec!(100)).unwrap(), dec!(0.03));
        assert_eq!(half_of(dec!(0.00001), dec!(100)).unwrap(), dec!(0.01));
    }

    #[test]
    fn test_a_transfer_arm_calibrates_on_the_other_window() {
        let (own, other) = ([at(0, dec!(100))], [at(1, dec!(200))]);
        assert_eq!(calibration_window(None, &own)[0], own[0]);
        assert_eq!(calibration_window(Some(&other), &own)[0], other[0]);
    }

    #[test]
    fn test_a_pair_is_the_arm_minus_the_fixed_arm() {
        let of = |net| Measure {
            net,
            ..Measure::default()
        };
        assert_eq!(paired(&[of(5.0)], &[of(3.0)], |m| m.net), [2.0]);
    }

    #[test]
    fn test_a_trade_is_known_only_after_its_minute() {
        let path = simulate(Flow::Hawkes, 1).unwrap();
        assert!(path.trades.iter().all(|t| t.timestamp % MINUTE == 0));
        // An order in the first minute is known at the second tick, not the first.
        let mut any_early = false;
        for seed in 1..10 {
            for flow in FLOWS {
                let seen = simulate(flow, seed).unwrap();
                assert!(seen.trades.iter().all(|t| t.timestamp >= MINUTE));
                any_early |= seen.trades.iter().any(|t| t.timestamp == MINUTE);
            }
        }
        assert!(
            any_early,
            "no order came in a first minute: this tested nothing"
        );
        // Stress changes the diffusion: the path's minute moves are larger inside it.
        let moves = |from: u64, to: u64| {
            let changes: Vec<f64> = path.ticks[from as usize..to as usize]
                .windows(2)
                .map(|w| number(w[1].mid_price() - w[0].mid_price()).abs())
                .collect();
            mean(&changes)
        };
        let day = DAY_MINUTES;
        assert!(moves(5 * day, 5 * day + 500) > 2.0 * moves(4 * day, 4 * day + 500));
    }

    #[test]
    fn test_an_alarm_comes_from_a_toxic_flow_and_stays_while_it_lasts() {
        // Only buyers, in full buckets, on a mid that does not move.
        let trades: Vec<Trade> = (0..1_200)
            .map(|i| Trade::new(dec!(100), dec!(1), TradeSide::Buy, i / 10 * MINUTE))
            .collect();
        let mut d = detector(trades);
        for minute in 0..120 {
            d.observe(&at(minute, dec!(100)));
        }
        assert!(d.vpin.is_toxic());
        assert!(d.ratio < ALARM_OFF, "{}", d.ratio);
        assert_eq!(d.alarms.len(), 1);
        assert!(d.alarm);
    }

    #[test]
    fn test_the_adaptive_arm_prices_from_the_flow() {
        let buyers = vec![Trade::new(dec!(100), dec!(50), TradeSide::Buy, 0)];
        let mid_of = |trades: Vec<Trade>| {
            let mut q = Quoter {
                pricing: Pricing::Adaptive,
                half: dec!(0.03),
                detector: detector(trades),
                calculator: AdaptiveSpreadCalculator::new(
                    AdaptiveSpreadConfig::new(dec!(0.01), dec!(2), dec!(0), dec!(0.5)).unwrap(),
                ),
                log: Rc::new(RefCell::new(vec![])),
            };
            let quote = q
                .on_tick(&at(0, dec!(100)), &InventoryPosition::new())
                .unwrap();
            quote.bid_price + quote.ask_price
        };
        assert_ne!(mid_of(buyers), mid_of(vec![]));
    }

    fn quick_measures(flow: Flow, rule: FillRule, seed: u64) -> BTreeMap<String, Measure> {
        let provenance = Provenance {
            commit: "t".to_string(),
            toolchain: "t".to_string(),
        };
        run_path(flow, seed, rule, &provenance)
            .unwrap()
            .0
            .holdout
            .iter()
            .map(|row| (row.candidate.clone(), measure_of(row).unwrap()))
            .collect()
    }

    #[test]
    fn test_the_transfer_arm_is_not_the_adaptive_arm() {
        for flow in FLOWS {
            let rows = quick_measures(flow, FillRule::Immediate, 1);
            assert_ne!(rows[TRANSFER], rows[ADAPTIVE], "{flow:?}");
        }
    }

    #[test]
    fn test_the_half_chance_rule_fills_about_half() {
        let of = |rule| quick_measures(Flow::Hawkes, rule, 1)[FIXED].fills;
        let share = of(FillRule::HalfChance) / of(FillRule::Immediate);
        assert!((share - 0.5).abs() < 0.1, "{share}");
    }

    #[test]
    fn test_an_arm_row_prints_each_figure_under_its_header() {
        let measure = Measure {
            net: 1.0,
            stress_net: 2.0,
            calm_net: 3.0,
            drawdown: 4.0,
            fills: 5.0,
        };
        let row = arm_row(Flow::Hawkes, FillRule::Immediate, "x", &[measure; 3]);
        let cells: Vec<&str> = row.split('|').map(str::trim).collect();
        assert_eq!(&cells[1..4], ["Hawkes", "Immediate", "x"]);
        for (cell, figure) in cells[4..9]
            .iter()
            .zip(["1.00", "2.00", "3.00", "4.00", "5.00"])
        {
            assert!(cell.starts_with(figure), "{cell}");
        }
    }

    #[test]
    fn test_a_summary_row_prints_each_gap_and_its_verdict() {
        let at = |m: f64| (m, m - 0.5, m + 0.5);
        let drawdown = (-3.0, -4.0, -2.0);
        let gaps = [at(1.0), drawdown, at(5.0), at(6.0)];
        let (row, go) = summary_row(Flow::Hawkes, FillRule::Immediate, gaps, 7.0, 0, 0.5);
        assert!(go);
        let cells: Vec<&str> = row.split('|').map(str::trim).collect();
        assert!(cells[3].starts_with("1.00"), "{}", cells[3]);
        assert!(cells[4].starts_with("-3.00"), "{}", cells[4]);
        assert!(cells[5].starts_with("5.00"), "{}", cells[5]);
        assert!(cells[6].starts_with("6.00"), "{}", cells[6]);
        assert_eq!(&cells[7..11], ["7.0", "0", "0.50", "go"]);
        // A missed stress fails the rule whatever else holds.
        let (row, go) = summary_row(Flow::Hawkes, FillRule::Immediate, gaps, 7.0, 1, 0.5);
        assert!(!go);
        assert!(row.contains("no-go"));
    }

    #[test]
    fn test_a_measure_reads_each_figure_of_its_row() {
        let provenance = Provenance {
            commit: "t".to_string(),
            toolchain: "t".to_string(),
        };
        let (report, _) = run_path(Flow::Hawkes, 1, FillRule::Immediate, &provenance).unwrap();
        let row = report
            .holdout
            .iter()
            .find(|r| r.candidate == FIXED)
            .unwrap();
        let RowOutcome::Completed(metrics) = &row.outcome else {
            panic!("not completed");
        };
        let measure = measure_of(row).unwrap();
        let regime = |name: &str| number(metrics.regimes[name].attribution.net_pnl);
        assert_eq!(measure.stress_net, regime("stress"));
        assert_eq!(measure.calm_net, regime("calm"));
        assert_ne!(measure.stress_net, measure.calm_net);
        assert_eq!(measure.net, number(metrics.metrics.attribution.net_pnl));
        assert_eq!(measure.drawdown, number(metrics.metrics.max_drawdown));
        assert_ne!(measure.drawdown, measure.net);
        assert_eq!(measure.fills, metrics.metrics.fills as f64);
    }

    #[test]
    fn test_interval_brackets_the_mean() {
        let values: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect();
        let (m, low, high) = interval(&values);
        assert_eq!(m, 0.5);
        assert!(
            (low - 0.42).abs() < 0.02 && (high - 0.58).abs() < 0.02,
            "{low} {high}"
        );
        assert_eq!(interval(&[2.0; 5]), (2.0, 2.0, 2.0));
    }

    #[test]
    fn test_a_short_run_of_every_arm_reaches_the_holdout() {
        let provenance = Provenance {
            commit: "t".to_string(),
            toolchain: "t".to_string(),
        };
        let (report, _) = run_path(Flow::Hawkes, 1, FillRule::Immediate, &provenance).unwrap();
        for row in &report.holdout {
            assert!(measure_of(row).is_ok(), "{:?}", row.outcome);
        }
        assert_eq!(report.holdout.len(), 6);
        assert!(report.not_selected.is_empty());
    }
}
