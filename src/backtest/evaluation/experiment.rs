//! Runs strategies under the same rules, and keeps the holdout for the end.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{RefUnwindSafe, UnwindSafe};

use rust_decimal::prelude::ToPrimitive;

use crate::Decimal;
use crate::backtest::data::{MarketTick, VecDataSource};
use crate::backtest::engine::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, SimulatedFill,
};
use crate::backtest::fill_models::FillModel;
use crate::backtest::latency::SimLatencyConfig;
use crate::position::inventory::InventoryPosition;
use crate::risk::pretrade::{OrderEvent, PretradeDecision, PretradePolicy, PretradeRequest};
use crate::strategy::quote::Quote;
use crate::strategy::walk_forward::{FNV_OFFSET, TimeWindow, fnv};
use crate::types::error::MMResult;

use super::EvaluationError;
use super::attribution::{
    BucketAttribution, DailyAttribution, EvaluationInput, ExecutedFill, LedgerTotals, Mark,
    MetricValue, MetricsSpec, WindowMetrics, attribute_by, attribute_daily, evaluate_window, sub,
};

#[cfg(feature = "serde")]
use serde::Serialize;

/// Most bootstrap replicates, which bounds the time of a report.
const MAX_REPLICATES: u32 = 1_000_000;

type EvalResult<T> = Result<T, EvaluationError>;

fn invalid<T>(reason: String) -> EvalResult<T> {
    Err(EvaluationError::InvalidInput { reason })
}

/// Contiguous windows: training ends where validation starts, and validation
/// where the holdout starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Splits {
    train: TimeWindow,
    validation: TimeWindow,
    holdout: TimeWindow,
}

impl Splits {
    /// Creates the windows from their limits.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] unless the limits increase.
    pub fn new(
        train_start_ms: u64,
        validation_start_ms: u64,
        holdout_start_ms: u64,
        end_ms: u64,
    ) -> EvalResult<Self> {
        let window = |start, end| {
            TimeWindow::new(start, end).map_err(|error| EvaluationError::InvalidInput {
                reason: error.to_string(),
            })
        };
        Ok(Self {
            train: window(train_start_ms, validation_start_ms)?,
            validation: window(validation_start_ms, holdout_start_ms)?,
            holdout: window(holdout_start_ms, end_ms)?,
        })
    }

    /// The training window.
    #[must_use]
    pub fn train(&self) -> TimeWindow {
        self.train
    }

    /// The validation window.
    #[must_use]
    pub fn validation(&self) -> TimeWindow {
        self.validation
    }

    /// The holdout window.
    #[must_use]
    pub fn holdout(&self) -> TimeWindow {
        self.holdout
    }
}

/// A stationary bootstrap of daily net results (Politis and Romano, 1994).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct BootstrapConfig {
    /// Resamples. Positive, at most a million.
    pub replicates: u32,
    /// Mean block length, in days. Positive.
    pub mean_block_days: u32,
    /// Confidence level of the interval, in percent, from 1 to 99.
    pub confidence_percent: u32,
    /// Seed. The resampled days depend on it, the window and the scenario, never
    /// on a candidate.
    pub seed: u64,
    /// Fewest observed days that give an interval. At least 2.
    pub min_days: u32,
}

impl BootstrapConfig {
    fn validate(&self) -> EvalResult<()> {
        if self.replicates == 0
            || self.replicates > MAX_REPLICATES
            || self.mean_block_days == 0
            || self.confidence_percent == 0
            || self.confidence_percent >= 100
            || self.min_days < 2
        {
            return invalid(format!("invalid bootstrap: {self:?}"));
        }
        Ok(())
    }
}

/// An interval on the mean daily net result, or why there is none.
///
/// It describes the sampling variation of a mean over these days. It is not the
/// probability of beating anything, and it is not corrected for how many candidates
/// were tried, which the manifest records.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub enum BootstrapOutcome {
    /// A percentile interval.
    Interval {
        /// Mean over the observed days.
        mean: f64,
        /// Lower limit.
        lower: f64,
        /// Upper limit.
        upper: f64,
        /// Observed days.
        days: u32,
    },
    /// Too few observed days.
    Insufficient {
        /// Observed days.
        days: u32,
        /// Days needed.
        required: u32,
    },
}

/// A small seeded generator, the same on every platform (SplitMix64).
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`, by the top bits of a 128 bit product.
    fn below(&mut self, n: u64) -> u64 {
        let wide = u128::from(self.next_u64()).wrapping_mul(u128::from(n));
        u64::try_from(wide >> 64).unwrap_or(0)
    }
}

/// The mean of `values` over `config.replicates` stationary resamples. A block
/// restarts with an integer comparison on a `u64`, so that no floating point enters
/// the choice of days. The draws do not depend on the values, so series of the same
/// length get the same days from the same seed.
fn resampled_means(values: &[f64], config: &BootstrapConfig, seed: u64) -> Vec<f64> {
    let Ok(n) = u64::try_from(values.len()) else {
        return Vec::new();
    };
    // `mean_block_days` is positive, checked with the rest of the configuration.
    let restart_below = u64::MAX
        .checked_div(u64::from(config.mean_block_days))
        .unwrap_or(u64::MAX);
    let mut rng = SplitMix(seed);
    let mut means = Vec::with_capacity(config.replicates.try_into().unwrap_or(0));
    let length = f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
    for _ in 0..config.replicates {
        let mut at = rng.below(n);
        let mut sum = 0.0;
        for _ in 0..n {
            let index = usize::try_from(at).unwrap_or(0);
            sum += values.get(index).copied().unwrap_or(0.0);
            at = if rng.next_u64() < restart_below {
                rng.below(n)
            } else {
                let next = at.saturating_add(1);
                if next == n { 0 } else { next }
            };
        }
        means.push(sum / length);
    }
    means
}

/// A stationary block bootstrap of the mean of daily values, the one [`Experiment`]
/// gives on the mean daily net result and on the daily difference to a baseline.
///
/// A block starts on a uniform day and goes on to the next day, round the end of the
/// series, until it restarts, with a mean length of `config.mean_block_days`. The means
/// of `config.replicates` resamples give a percentile interval at
/// `config.confidence_percent`, with the same number of resamples cut from each tail.
/// Fewer than `config.min_days` values give [`BootstrapOutcome::Insufficient`].
///
/// The resampled days depend on `config.seed` and on how many values there are, never
/// on the values, so two series of one length are resampled on the same days. The
/// interval on a paired comparison is therefore this function over the daily
/// differences. [`Experiment`] derives a seed for each window and scenario from its
/// configured one; this function uses `config.seed` as it is.
///
/// # Errors
///
/// Returns [`EvaluationError::InvalidInput`] for a configuration [`Experiment`] would
/// refuse, and for a value with no finite `f64`.
pub fn block_bootstrap(
    series: &[Decimal],
    config: &BootstrapConfig,
) -> EvalResult<BootstrapOutcome> {
    config.validate()?;
    bootstrap(series, config, config.seed)
}

fn bootstrap(
    series: &[Decimal],
    config: &BootstrapConfig,
    seed: u64,
) -> EvalResult<BootstrapOutcome> {
    let days = u32::try_from(series.len()).unwrap_or(u32::MAX);
    if days < config.min_days {
        return Ok(BootstrapOutcome::Insufficient {
            days,
            required: config.min_days,
        });
    }
    let values = finite(series)?;
    let mean = values.iter().sum::<f64>() / f64::from(days);
    interval(mean, resampled_means(&values, config, seed), config, days)
}

/// The values as `f64`, refusing one with no finite `f64`.
fn finite(series: &[Decimal]) -> EvalResult<Vec<f64>> {
    let mut values = Vec::with_capacity(series.len());
    for value in series {
        match value.to_f64() {
            Some(number) if number.is_finite() => values.push(number),
            _ => {
                return invalid(format!("daily result {value} has no finite f64"));
            }
        }
    }
    Ok(values)
}

/// The percentile interval of the resampled means, the same tail cut from each side.
fn interval(
    mean: f64,
    mut means: Vec<f64>,
    config: &BootstrapConfig,
    days: u32,
) -> EvalResult<BootstrapOutcome> {
    means.sort_by(f64::total_cmp);
    // The same tail on each side, from integers: (100 - c) R / 200 values.
    let replicates = u64::from(config.replicates);
    let tail = (u64::from(100_u32.saturating_sub(config.confidence_percent)))
        .saturating_mul(replicates)
        / 200;
    let lower = usize::try_from(tail)
        .ok()
        .and_then(|i| means.get(i))
        .copied();
    let upper = usize::try_from(replicates.saturating_sub(tail).saturating_sub(1))
        .ok()
        .and_then(|i| means.get(i))
        .copied();
    match (lower, upper) {
        (Some(lower), Some(upper)) => Ok(BootstrapOutcome::Interval {
            mean,
            lower,
            upper,
            days,
        }),
        _ => invalid("the bootstrap has no replicates".to_string()),
    }
}

/// A stationary block bootstrap of the equally weighted mean of several groups' means: for
/// example, of regimes, each with its own days.
///
/// Each group is resampled on its own, as [`block_bootstrap`] resamples one series, from its
/// own stream: group `i` (in the order given) is seeded `config.seed + i`, so group 0 keeps
/// `config.seed` and a single group gives exactly [`block_bootstrap`]'s interval. A replicate's
/// value is the unweighted mean of the groups' replicate means, whatever their lengths, and
/// the interval is [`block_bootstrap`]'s percentile cut of those values. Days are never pooled
/// across groups, and a group's days keep their order. A paired comparison is a group of daily
/// differences per group, which the caller builds.
///
/// The streams are distinct seeds of the same generator: they keep the groups' draws apart
/// deterministically, which is not a proof of independence. The interval's `days` is the
/// days of every group together. A group shorter than `config.min_days` gives
/// [`BootstrapOutcome::Insufficient`] with that group's days.
///
/// # Errors
///
/// Returns [`EvaluationError::InvalidInput`] for a configuration [`block_bootstrap`] would
/// refuse, for no group, for a group with no days, for a value with no finite `f64`, and
/// for a seed that `config.seed + i` would overflow.
pub fn stratified_block_bootstrap(
    groups: &[&[Decimal]],
    config: &BootstrapConfig,
) -> EvalResult<BootstrapOutcome> {
    config.validate()?;
    if groups.is_empty() {
        return invalid("a stratified bootstrap needs at least one group".to_string());
    }
    let mut shortest = u32::MAX;
    let mut days = 0_u32;
    for (index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return invalid(format!("group {index} has no days"));
        }
        let length = u32::try_from(group.len()).unwrap_or(u32::MAX);
        shortest = shortest.min(length);
        days = days.saturating_add(length);
    }
    if shortest < config.min_days {
        return Ok(BootstrapOutcome::Insufficient {
            days: shortest,
            required: config.min_days,
        });
    }
    let count = f64::from(u32::try_from(groups.len()).unwrap_or(u32::MAX));
    let mut pooled = vec![0.0; usize::try_from(config.replicates).unwrap_or(0)];
    let mut mean = 0.0;
    for (index, group) in groups.iter().enumerate() {
        let seed = u64::try_from(index)
            .ok()
            .and_then(|i| config.seed.checked_add(i));
        let Some(seed) = seed else {
            return invalid(format!("seed {} plus group {index} overflows", config.seed));
        };
        let values = finite(group)?;
        let length = f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
        mean += values.iter().sum::<f64>() / length / count;
        let means = resampled_means(&values, config, seed);
        for (total, value) in pooled.iter_mut().zip(means) {
            *total += value / count;
        }
    }
    interval(mean, pooled, config, days)
}

/// Who declares a commit and a toolchain. Nothing here verifies them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Provenance {
    /// Commit of the code, as the caller states it.
    pub commit: String,
    /// Toolchain, as the caller states it.
    pub toolchain: String,
}

/// The rules every candidate runs under.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Name.
    pub name: String,
    /// Fees, capital, tick and lot. The experiment records every trade whatever
    /// this says.
    pub execution: BacktestConfig,
    /// Windows.
    pub splits: Splits,
    /// Metrics.
    pub metrics: MetricsSpec,
    /// Bootstrap.
    pub bootstrap: BootstrapConfig,
    /// Seed of the fill models.
    pub seed: u64,
    /// Declared provenance.
    pub provenance: Provenance,
}

/// Whether a candidate is a simple baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub enum CandidateKind {
    /// Runs on the holdout whatever is selected.
    Baseline,
    /// Runs on the holdout only if selected.
    Model,
}

type StrategyFactory = dyn Fn(&[MarketTick]) -> MMResult<Box<dyn BacktestStrategy>>;

/// A strategy to evaluate.
///
/// It brings no fees and no execution rules, only a strategy built from the training
/// ticks. On the holdout it is built again from the same training ticks: nothing is
/// refit on validation data.
pub struct Candidate {
    name: String,
    kind: CandidateKind,
    parameters: BTreeMap<String, String>,
    make: Box<StrategyFactory>,
}

impl fmt::Debug for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Candidate")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl Candidate {
    /// A model. `parameters` go to the manifest, so they should pin the factory.
    #[must_use]
    pub fn model(
        name: impl Into<String>,
        parameters: BTreeMap<String, String>,
        make: impl Fn(&[MarketTick]) -> MMResult<Box<dyn BacktestStrategy>> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            kind: CandidateKind::Model,
            parameters,
            make: Box::new(make),
        }
    }

    /// The baseline that never quotes.
    #[must_use]
    pub fn no_trade() -> Self {
        Self {
            name: "no-trade".to_string(),
            kind: CandidateKind::Baseline,
            parameters: BTreeMap::new(),
            make: Box::new(|_| Ok(Box::new(NoTrade))),
        }
    }

    /// The baseline that quotes a fixed distance either side of the mid.
    #[must_use]
    pub fn fixed_spread(half_spread: Decimal, size: Decimal) -> Self {
        let parameters = BTreeMap::from([
            (
                "half_spread".to_string(),
                half_spread.normalize().to_string(),
            ),
            ("size".to_string(), size.normalize().to_string()),
        ]);
        Self {
            name: "fixed-spread".to_string(),
            kind: CandidateKind::Baseline,
            parameters,
            make: Box::new(move |_| Ok(Box::new(FixedSpread { half_spread, size }))),
        }
    }
}

struct NoTrade;

impl BacktestStrategy for NoTrade {
    fn on_tick(&mut self, _tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        None
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

struct FixedSpread {
    half_spread: Decimal,
    size: Decimal,
}

impl BacktestStrategy for FixedSpread {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        let mid = tick.mid_price();
        Some(Quote {
            bid_price: mid.checked_sub(self.half_spread)?,
            bid_size: self.size,
            ask_price: mid.checked_add(self.half_spread)?,
            ask_size: self.size,
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

/// A strategy behind a box, for the engine's type parameter.
struct BoxedStrategy(Box<dyn BacktestStrategy>);

impl BacktestStrategy for BoxedStrategy {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        self.0.on_tick(tick, position)
    }
    fn on_fill(&mut self, fill: &SimulatedFill) {
        self.0.on_fill(fill);
    }
    fn on_execution_report(&mut self, report: &crate::backtest::lifecycle::ExecutionReport) {
        self.0.on_execution_report(report);
    }
    fn cancels_when_silent(&self) -> bool {
        self.0.cancels_when_silent()
    }
    fn reset(&mut self) {
        self.0.reset();
    }
}

/// What a pretrade policy needs to be for the engine.
pub trait SharedPolicy: PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe {}
impl<T: PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe> SharedPolicy for T {}

/// A policy behind a box, for the engine's type parameter.
#[derive(Debug)]
struct BoxedPolicy(Box<dyn SharedPolicy>);

impl PretradePolicy for BoxedPolicy {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        self.0.check(request)
    }
    fn on_market_data(&mut self, mark: Decimal, timestamp: u64) {
        self.0.on_market_data(mark, timestamp);
    }
    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        self.0.on_order_event(event)
    }
    fn reset(&mut self) {
        self.0.reset();
    }
    fn description(&self) -> String {
        self.0.description()
    }
    fn in_contracts(&self) -> bool {
        self.0.in_contracts()
    }
}

type PolicyFactory = dyn Fn() -> Box<dyn SharedPolicy>;

struct RunRequest {
    execution: BacktestConfig,
    strategy: Box<dyn BacktestStrategy>,
    ticks: Vec<MarketTick>,
    seed: u64,
    policy: Option<Box<dyn SharedPolicy>>,
}

struct RawRun {
    result: BacktestResult,
    fault_count: u64,
    first_fault: Option<String>,
}

type RunFn = dyn Fn(RunRequest) -> MMResult<RawRun>;

/// An execution assumption: a fill model, a latency and optionally a fee rate.
pub struct Scenario {
    name: String,
    latency: SimLatencyConfig,
    fee_rate: Option<Decimal>,
    parameters: BTreeMap<String, String>,
    fill_model_description: Box<dyn Fn() -> String>,
    run: Box<RunFn>,
}

impl fmt::Debug for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scenario")
            .field("name", &self.name)
            .field("latency", &self.latency)
            .field("fee_rate", &self.fee_rate)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl Scenario {
    /// A scenario. `make_fill_model` gets a seed that depends on the experiment's
    /// seed, the window and the scenario, never on a candidate. It is called once per
    /// run, so no state passes between candidates, and with seed 0 to describe the
    /// model in the manifest.
    #[must_use]
    pub fn new<F, M>(
        name: impl Into<String>,
        latency: SimLatencyConfig,
        parameters: BTreeMap<String, String>,
        make_fill_model: M,
    ) -> Self
    where
        F: FillModel + 'static,
        M: Fn(u64) -> F + 'static,
    {
        let make_fill_model = std::rc::Rc::new(make_fill_model);
        let describe = std::rc::Rc::clone(&make_fill_model);
        Self {
            name: name.into(),
            latency,
            fee_rate: None,
            parameters,
            fill_model_description: Box::new(move || describe(0).description()),
            run: Box::new(move |request: RunRequest| {
                let mut engine = BacktestEngine::with_fill_model(
                    request.execution,
                    BoxedStrategy(request.strategy),
                    VecDataSource::new(request.ticks),
                    make_fill_model(request.seed),
                )?
                .with_latency(latency)?;
                if let Some(policy) = request.policy {
                    engine = engine.with_pretrade_policy(BoxedPolicy(policy));
                }
                let result = engine.run();
                Ok(RawRun {
                    result,
                    fault_count: engine.fault_count(),
                    first_fault: engine.faults().first().map(ToString::to_string),
                })
            }),
        }
    }
}

impl Scenario {
    /// Replaces the experiment's fee rate under this scenario, a negative rate being
    /// a rebate. The fee currency stays the experiment's. Any other rule is unchanged.
    #[must_use]
    pub fn with_fee_rate(mut self, fee_rate: Decimal) -> Self {
        self.fee_rate = Some(fee_rate);
        self
    }
}

/// Names the regime of a mark. Created fresh for each run, and called in time
/// order, so it can keep state and sees only the past.
pub type RegimeLabeller = dyn FnMut(&Mark) -> String;

type RegimeFactory = dyn Fn() -> Box<RegimeLabeller>;

/// Which window a row comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub enum WindowKind {
    /// Validation.
    Validation,
    /// Holdout.
    Holdout,
}

impl WindowKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Holdout => "holdout",
        }
    }
}

/// The results of a run that could be measured.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct RowMetrics {
    /// The window.
    pub metrics: WindowMetrics,
    /// Every metric, with its definition.
    pub values: Vec<MetricValue>,
    /// Each UTC day of the window, those without ticks included.
    pub daily: Vec<DailyAttribution>,
    /// Each regime seen; empty without a labeller.
    pub regimes: BTreeMap<String, BucketAttribution>,
    /// The interval on the mean daily net result, over the days with ticks.
    pub mean_daily_net: BootstrapOutcome,
    /// The interval on the mean daily difference to each baseline that completed,
    /// by its name, over the same resampled days.
    pub against_baselines: BTreeMap<String, BootstrapOutcome>,
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub enum RowOutcome {
    /// It ran and reconciled.
    Completed(Box<RowMetrics>),
    /// It did not. The row stays in the report, with the reason.
    Failed {
        /// Why.
        reason: String,
    },
}

/// One candidate under one scenario over one window.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct EvaluationRow {
    /// Candidate name.
    pub candidate: String,
    /// Baseline or model.
    pub kind: CandidateKind,
    /// Scenario name.
    pub scenario: String,
    /// Window.
    pub window: WindowKind,
    /// Result.
    pub outcome: RowOutcome,
}

/// A candidate as the manifest records it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct CandidateRecord {
    /// Baseline or model.
    pub kind: CandidateKind,
    /// Its declared parameters.
    pub parameters: BTreeMap<String, String>,
}

/// A scenario as the manifest records it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct ScenarioRecord {
    /// Debug text of the latency configuration, with its seed.
    pub latency: String,
    /// The fee rate replacing the experiment's, if any.
    pub fee_rate: Option<Decimal>,
    /// The fill model's own description.
    pub fill_model: String,
    /// Declared parameters.
    pub parameters: BTreeMap<String, String>,
    /// Seed given to the fill model on validation.
    pub validation_seed: u64,
    /// Seed given to the fill model on the holdout.
    pub holdout_seed: u64,
}

/// What it would take to reproduce a report. It holds no time of day, so two runs
/// give the same bytes.
///
/// The commit and the toolchain are what the caller declared. The data hashes are
/// FNV-1a 128: they catch an accident, not tampering.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct Manifest {
    /// Experiment name.
    pub name: String,
    /// Version of this crate.
    pub crate_version: String,
    /// Declared provenance.
    pub provenance: Provenance,
    /// Windows.
    pub splits: Splits,
    /// Hash of the ticks of each split, in hexadecimal.
    pub data_hashes: BTreeMap<String, String>,
    /// Debug text of the execution configuration.
    pub execution: String,
    /// The pretrade policy every run uses, if any.
    pub pretrade_policy: Option<String>,
    /// The declared description of the regime labeller, if any.
    pub regime_labeller: Option<String>,
    /// Horizons and the mark rule.
    pub metrics: MetricsSpec,
    /// Bootstrap.
    pub bootstrap: BootstrapConfig,
    /// Seed of the fill models.
    pub seed: u64,
    /// How a strategy is built: always from the training ticks alone, for validation
    /// and for the holdout.
    pub refit: String,
    /// Every candidate tried. Its size is the selection pressure a reader should
    /// allow for.
    pub candidates: BTreeMap<String, CandidateRecord>,
    /// Every scenario. Selection is on the first, the reference.
    pub scenarios: BTreeMap<String, ScenarioRecord>,
    /// The reference scenario.
    pub reference_scenario: String,
    /// What was selected, once it was.
    pub selection: Vec<String>,
    /// FNV-1a 128 of the debug text of the validation rows the selection came from,
    /// once it was made.
    pub validation_digest: Option<String>,
}

/// The validation rows: every candidate under every scenario.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct ValidationReport {
    /// The manifest, before any selection.
    pub manifest: Manifest,
    /// The rows, in the order of the sorted candidate names, per scenario.
    pub rows: Vec<EvaluationRow>,
}

/// The final report.
///
/// The holdout rows are the selected candidates and the baselines, under the
/// reference scenario only. The other candidates keep their validation rows, with no
/// holdout result. `not_selected` names them.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct HoldoutReport {
    /// The manifest, with the selection.
    pub manifest: Manifest,
    /// Every validation row, the failed ones too.
    pub validation: Vec<EvaluationRow>,
    /// The holdout rows.
    pub holdout: Vec<EvaluationRow>,
    /// Candidates that did not reach the holdout.
    pub not_selected: Vec<String>,
}

fn hash_ticks(ticks: &[MarketTick]) -> String {
    let mut hash = FNV_OFFSET;
    for tick in ticks {
        hash = fnv(hash, &tick.timestamp.to_le_bytes());
        for value in [tick.bid_price, tick.ask_price, tick.bid_size, tick.ask_size] {
            hash = fnv(hash, &value.normalize().serialize());
        }
        // A strategy or fill model can read these. The tag tells absent from any value.
        for value in [tick.last_price, tick.last_size] {
            hash = match value {
                Some(value) => fnv(fnv(hash, &[1]), &value.normalize().serialize()),
                None => fnv(hash, &[0]),
            };
        }
    }
    format!("{hash:032x}")
}

fn derived_seed(seed: u64, window: WindowKind, scenario: &str) -> u64 {
    let hash = fnv(
        fnv(
            fnv(FNV_OFFSET, &seed.to_le_bytes()),
            window.label().as_bytes(),
        ),
        scenario.as_bytes(),
    );
    u64::try_from(hash >> 64).unwrap_or(0)
}

/// An experiment before validation. See the [module documentation](super).
///
/// The holdout ticks are private, and they stay in the object as it passes through
/// [`Experiment::evaluate_validation`], [`ValidatedExperiment::select`] and
/// [`Selection::run_holdout`], each of which consumes the last. So one experiment
/// reaches its holdout once, after a selection from its own validation.
///
/// This guards one object. Nothing stops a caller from building another over the
/// same data; only comparing the data hashes of two manifests shows that.
pub struct Experiment {
    config: ExperimentConfig,
    train: Vec<MarketTick>,
    validation: Vec<MarketTick>,
    holdout: Vec<MarketTick>,
    candidates: BTreeMap<String, Candidate>,
    /// The first scenario, the only one the holdout runs under.
    reference: Scenario,
    /// The scenarios after it, for sensitivity on validation.
    sensitivity: Vec<Scenario>,
    policy: Option<Box<PolicyFactory>>,
    /// The declared description of the labeller, and how to make one.
    regimes: Option<(String, Box<RegimeFactory>)>,
}

impl fmt::Debug for Experiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Experiment")
            .field("name", &self.config.name)
            .field("splits", &self.config.splits)
            .field("candidates", &self.candidates.keys())
            .finish_non_exhaustive()
    }
}

impl Experiment {
    /// Checks and splits the data.
    ///
    /// The order of `candidates` does not matter.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] if ticks are out of order, outside
    /// the windows or leave a window empty, if there is no baseline or no scenario,
    /// if a name repeats, or if the bootstrap, a horizon or the capital is invalid.
    pub fn new(
        config: ExperimentConfig,
        ticks: Vec<MarketTick>,
        candidates: Vec<Candidate>,
        scenarios: Vec<Scenario>,
    ) -> EvalResult<Self> {
        config.bootstrap.validate()?;
        if config.metrics.horizons_ms.contains(&0) {
            return invalid("a markout horizon must be positive".to_string());
        }
        if config.execution.initial_capital <= Decimal::ZERO {
            return invalid("the capital must be positive".to_string());
        }
        if !candidates.iter().any(|c| c.kind == CandidateKind::Baseline) {
            return invalid("an experiment needs a simple baseline".to_string());
        }
        let names: BTreeSet<&str> = scenarios.iter().map(|s| s.name.as_str()).collect();
        if names.len() != scenarios.len() {
            return invalid("scenario names must be different".to_string());
        }
        let mut scenarios = scenarios.into_iter();
        let Some(reference) = scenarios.next() else {
            return invalid("an experiment needs a scenario".to_string());
        };
        let sensitivity = scenarios.collect();
        let mut by_name = BTreeMap::new();
        for candidate in candidates {
            let name = candidate.name.clone();
            if by_name.insert(name.clone(), candidate).is_some() {
                return invalid(format!("candidate {name} appears twice"));
            }
        }
        let splits = config.splits;
        let (mut train, mut validation, mut holdout) = (Vec::new(), Vec::new(), Vec::new());
        let mut previous = 0_u64;
        for tick in ticks {
            let t = tick.timestamp;
            if t < previous {
                return invalid(format!("tick at {t} is out of order"));
            }
            previous = t;
            let inside = |w: TimeWindow| w.start_ms <= t && t < w.end_ms;
            if inside(splits.train) {
                train.push(tick);
            } else if inside(splits.validation) {
                validation.push(tick);
            } else if inside(splits.holdout) {
                holdout.push(tick);
            } else {
                return invalid(format!("tick at {t} is outside the splits"));
            }
        }
        if train.is_empty() || validation.is_empty() || holdout.is_empty() {
            return invalid(format!(
                "every split needs ticks: {} train, {} validation, {} holdout",
                train.len(),
                validation.len(),
                holdout.len()
            ));
        }
        Ok(Self {
            config,
            train,
            validation,
            holdout,
            candidates: by_name,
            reference,
            sensitivity,
            policy: None,
            regimes: None,
        })
    }

    /// Uses a pretrade policy in every run, a new one each time.
    #[must_use]
    pub fn with_pretrade_policy(
        mut self,
        make: impl Fn() -> Box<dyn SharedPolicy> + 'static,
    ) -> Self {
        self.policy = Some(Box::new(make));
        self
    }

    /// Reports by regime, with a new labeller for each run. The manifest records
    /// `description`, which must say everything that decides the labels, such as the
    /// rule and its parameters: code cannot be hashed, so a labeller that changes under
    /// the same description is not detected.
    #[must_use]
    pub fn with_regimes(
        mut self,
        description: impl Into<String>,
        make: impl Fn() -> Box<RegimeLabeller> + 'static,
    ) -> Self {
        self.regimes = Some((description.into(), Box::new(make)));
        self
    }

    fn manifest(&self, selection: Vec<String>, validation_digest: Option<String>) -> Manifest {
        let scenarios = self
            .scenarios()
            .map(|s| {
                (
                    s.name.clone(),
                    ScenarioRecord {
                        latency: format!("{:?}", s.latency),
                        fee_rate: s.fee_rate,
                        fill_model: (s.fill_model_description)(),
                        parameters: s.parameters.clone(),
                        validation_seed: derived_seed(
                            self.config.seed,
                            WindowKind::Validation,
                            &s.name,
                        ),
                        holdout_seed: derived_seed(self.config.seed, WindowKind::Holdout, &s.name),
                    },
                )
            })
            .collect();
        Manifest {
            name: self.config.name.clone(),
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            provenance: self.config.provenance.clone(),
            splits: self.config.splits,
            data_hashes: BTreeMap::from([
                ("train".to_string(), hash_ticks(&self.train)),
                ("validation".to_string(), hash_ticks(&self.validation)),
                ("holdout".to_string(), hash_ticks(&self.holdout)),
            ]),
            execution: format!("{:?}", self.config.execution),
            pretrade_policy: self.policy.as_ref().map(|make| make().description()),
            regime_labeller: self
                .regimes
                .as_ref()
                .map(|(description, _)| description.clone()),
            metrics: self.config.metrics.clone(),
            bootstrap: self.config.bootstrap,
            seed: self.config.seed,
            refit: "each strategy is built from the training ticks alone".to_string(),
            candidates: self
                .candidates
                .iter()
                .map(|(name, c)| {
                    (
                        name.clone(),
                        CandidateRecord {
                            kind: c.kind,
                            parameters: c.parameters.clone(),
                        },
                    )
                })
                .collect(),
            scenarios,
            reference_scenario: self.reference.name.clone(),
            selection,
            validation_digest,
        }
    }

    /// Every scenario, the reference first.
    fn scenarios(&self) -> impl Iterator<Item = &Scenario> {
        std::iter::once(&self.reference).chain(&self.sensitivity)
    }

    /// One run, or why not.
    fn measure(
        &self,
        candidate: &Candidate,
        scenario: &Scenario,
        kind: WindowKind,
        window: TimeWindow,
        ticks: &[MarketTick],
    ) -> Result<(RowMetrics, Vec<Decimal>), String> {
        let strategy = (candidate.make)(&self.train).map_err(|e| format!("factory: {e}"))?;
        let mut execution = self.config.execution.clone();
        execution.record_trades = true;
        if let Some(fee_rate) = scenario.fee_rate {
            execution.fee_rate = fee_rate;
        }
        let raw = (scenario.run)(RunRequest {
            execution,
            strategy,
            ticks: ticks.to_vec(),
            seed: derived_seed(self.config.seed, kind, &scenario.name),
            policy: self.policy.as_ref().map(|make| make()),
        })
        .map_err(|e| format!("setup: {e}"))?;
        let result = raw.result;
        if !result.accounting.is_consistent() {
            return Err(format!("accounting: {:?}", result.accounting));
        }
        if raw.fault_count > 0 {
            return Err(format!(
                "{} engine faults, the first: {}",
                raw.fault_count,
                raw.first_fault.unwrap_or_default()
            ));
        }
        if usize::try_from(result.num_ticks).ok() != Some(ticks.len()) {
            return Err(format!(
                "the run stopped after {} of {} ticks",
                result.num_ticks,
                ticks.len()
            ));
        }
        let marks: Vec<Mark> = ticks
            .iter()
            .map(|tick| Mark {
                timestamp_ms: tick.timestamp,
                mid: tick.mid_price(),
            })
            .collect();
        let fills: Vec<ExecutedFill> = result
            .trades
            .iter()
            .map(ExecutedFill::from_simulated)
            .collect();
        let capital = self.config.execution.initial_capital;
        let text = |e: EvaluationError| e.to_string();
        let input = EvaluationInput::new(window, &marks, &fills, capital).map_err(text)?;
        let (split, daily) = attribute_daily(&input).map_err(text)?;
        split
            .reconcile(LedgerTotals {
                net_pnl: result.net_pnl,
                position: result.final_position,
            })
            .map_err(text)?;
        let metrics = evaluate_window(&input, &self.config.metrics).map_err(text)?;
        let values = metrics.values().map_err(text)?;
        let regimes = match &self.regimes {
            Some((_, make)) => {
                let mut label = make();
                attribute_by(&input, |mark| label(mark))
                    .map_err(text)?
                    .buckets
            }
            None => BTreeMap::new(),
        };
        let series = daily
            .iter()
            .filter(|day| day.bucket.marks > 0)
            .map(|day| day.bucket.attribution.net_pnl)
            .collect();
        Ok((
            RowMetrics {
                metrics,
                values,
                daily,
                regimes,
                mean_daily_net: BootstrapOutcome::Insufficient {
                    days: 0,
                    required: self.config.bootstrap.min_days,
                },
                against_baselines: BTreeMap::new(),
            },
            series,
        ))
    }

    /// The rows of one scenario over one window, with their intervals.
    fn rows_for(
        &self,
        scenario: &Scenario,
        kind: WindowKind,
        names: &BTreeSet<&String>,
    ) -> Vec<EvaluationRow> {
        let (window, ticks) = match kind {
            WindowKind::Validation => (self.config.splits.validation, &self.validation),
            WindowKind::Holdout => (self.config.splits.holdout, &self.holdout),
        };
        let mut measured = Vec::new();
        let mut baselines: BTreeMap<String, Vec<Decimal>> = BTreeMap::new();
        for (name, candidate) in &self.candidates {
            if !names.contains(name) {
                continue;
            }
            let run = self.measure(candidate, scenario, kind, window, ticks);
            if let (Ok((_, series)), CandidateKind::Baseline) = (&run, candidate.kind) {
                baselines.insert(name.clone(), series.clone());
            }
            measured.push((name, candidate.kind, run));
        }
        let seed = derived_seed(self.config.bootstrap.seed, kind, &scenario.name);
        let bootstrap_config = &self.config.bootstrap;
        measured
            .into_iter()
            .map(|(name, candidate_kind, run)| {
                let outcome = match run {
                    Err(reason) => RowOutcome::Failed { reason },
                    Ok((mut row, series)) => match with_intervals(
                        &mut row,
                        &series,
                        &baselines,
                        name,
                        bootstrap_config,
                        seed,
                    ) {
                        Ok(()) => RowOutcome::Completed(Box::new(row)),
                        Err(error) => RowOutcome::Failed {
                            reason: error.to_string(),
                        },
                    },
                };
                EvaluationRow {
                    candidate: name.clone(),
                    kind: candidate_kind,
                    scenario: scenario.name.clone(),
                    window: kind,
                    outcome,
                }
            })
            .collect()
    }

    /// Runs every candidate under every scenario over the validation window.
    #[must_use]
    pub fn evaluate_validation(self) -> ValidatedExperiment {
        let names: BTreeSet<&String> = self.candidates.keys().collect();
        let rows = self
            .scenarios()
            .flat_map(|scenario| self.rows_for(scenario, WindowKind::Validation, &names))
            .collect();
        let report = ValidationReport {
            manifest: self.manifest(Vec::new(), None),
            rows,
        };
        ValidatedExperiment {
            experiment: self,
            report,
        }
    }
}

fn with_intervals(
    row: &mut RowMetrics,
    series: &[Decimal],
    baselines: &BTreeMap<String, Vec<Decimal>>,
    name: &str,
    config: &BootstrapConfig,
    seed: u64,
) -> EvalResult<()> {
    row.mean_daily_net = bootstrap(series, config, seed)?;
    for (baseline, other) in baselines {
        if baseline == name {
            continue;
        }
        if other.len() != series.len() {
            return invalid(format!("{baseline} has other days than {name}"));
        }
        let difference = series
            .iter()
            .zip(other)
            .map(|(a, b)| sub(*a, *b, "a daily difference"))
            .collect::<EvalResult<Vec<Decimal>>>()?;
        row.against_baselines
            .insert(baseline.clone(), bootstrap(&difference, config, seed)?);
    }
    Ok(())
}

/// An experiment whose validation has run.
pub struct ValidatedExperiment {
    experiment: Experiment,
    report: ValidationReport,
}

impl fmt::Debug for ValidatedExperiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatedExperiment")
            .field("experiment", &self.experiment)
            .finish_non_exhaustive()
    }
}

impl ValidatedExperiment {
    /// The validation rows, to select from.
    #[must_use]
    pub fn report(&self) -> &ValidationReport {
        &self.report
    }

    /// Chooses what runs on the holdout, by name. The baselines always run.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] for an empty selection, an unknown
    /// name, or a candidate that failed under the reference scenario, and the
    /// experiment is gone: selection is not something to retry until it passes.
    pub fn select(self, names: &[&str]) -> EvalResult<Selection> {
        if names.is_empty() {
            return invalid("a selection needs a candidate".to_string());
        }
        let reference = self.experiment.reference.name.clone();
        for name in names {
            if !self.experiment.candidates.contains_key(*name) {
                return invalid(format!("unknown candidate {name}"));
            }
            let failed = self.report.rows.iter().any(|row| {
                row.candidate == *name
                    && row.scenario == reference
                    && matches!(row.outcome, RowOutcome::Failed { .. })
            });
            if failed {
                return invalid(format!("candidate {name} failed on validation"));
            }
        }
        let selected: BTreeSet<String> = names.iter().map(ToString::to_string).collect();
        let digest = {
            let text = format!("{:?}", self.report.rows);
            format!("{:032x}", fnv(FNV_OFFSET, text.as_bytes()))
        };
        Ok(Selection {
            experiment: self.experiment,
            validation: self.report.rows,
            selected,
            digest,
        })
    }
}

/// A selection, which owns its experiment, so it cannot be used with another.
/// It is neither `Clone` nor constructible from outside.
pub struct Selection {
    experiment: Experiment,
    validation: Vec<EvaluationRow>,
    selected: BTreeSet<String>,
    digest: String,
}

impl fmt::Debug for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Selection")
            .field("selected", &self.selected)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

impl Selection {
    /// What was selected.
    #[must_use]
    pub fn selected(&self) -> &BTreeSet<String> {
        &self.selected
    }

    /// Runs the selection and the baselines over the holdout, under the reference
    /// scenario, and ends the experiment.
    ///
    /// ```
    /// # use market_maker_rs::backtest::evaluation::Selection;
    /// fn once(selection: Selection) {
    ///     let _report = selection.run_holdout();
    /// }
    /// ```
    ///
    /// A second look does not compile:
    ///
    /// ```compile_fail,E0382
    /// # use market_maker_rs::backtest::evaluation::Selection;
    /// fn twice(selection: Selection) {
    ///     let _report = selection.run_holdout();
    ///     let _again = selection.run_holdout();
    /// }
    /// ```
    #[must_use]
    pub fn run_holdout(self) -> HoldoutReport {
        let experiment = &self.experiment;
        let names: BTreeSet<&String> = experiment
            .candidates
            .iter()
            .filter(|(name, c)| c.kind == CandidateKind::Baseline || self.selected.contains(*name))
            .map(|(name, _)| name)
            .collect();
        let holdout = experiment.rows_for(&experiment.reference, WindowKind::Holdout, &names);
        let not_selected = experiment
            .candidates
            .keys()
            .filter(|name| !names.contains(name))
            .cloned()
            .collect();
        HoldoutReport {
            manifest: experiment.manifest(
                self.selected.iter().cloned().collect(),
                Some(self.digest.clone()),
            ),
            validation: self.validation,
            holdout,
            not_selected,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_matches_the_published_splitmix64_sequence() {
        // Seed 0 of the reference implementation by Sebastiano Vigna.
        let mut rng = SplitMix(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    fn config(mean_block_days: u32) -> BootstrapConfig {
        BootstrapConfig {
            replicates: 500,
            mean_block_days,
            confidence_percent: 90,
            seed: 7,
            min_days: 2,
        }
    }

    #[test]
    fn test_resampled_means_of_a_constant_are_that_constant() {
        for block in [1, 3, 1000] {
            let means = resampled_means(&[2.5; 9], &config(block), 11);
            assert_eq!(means.len(), 500);
            assert!(means.iter().all(|m| (*m - 2.5).abs() < 1e-12));
        }
    }

    #[test]
    fn test_resampled_means_long_blocks_keep_the_order_of_the_days() {
        // A block that never restarts walks round the circle, so every resample of
        // 0..n is a whole cycle with the same mean. Mean block 1 does not.
        let values: Vec<f64> = (0..16).map(f64::from).collect();
        let long = resampled_means(&values, &config(u32::MAX), 3);
        assert!(long.iter().filter(|m| (**m - 7.5).abs() < 1e-12).count() > 490);
        let short = resampled_means(&values, &config(1), 3);
        assert!(short.iter().filter(|m| (**m - 7.5).abs() < 1e-12).count() < 100);
    }

    #[test]
    fn test_bootstrap_same_seed_same_days_for_any_values() {
        let a = [dec(1), dec(-2), dec(3), dec(4), dec(-5)];
        let doubled = a.map(|x| x * dec(2));
        let config = config(2);
        let (
            BootstrapOutcome::Interval {
                lower: l1,
                upper: u1,
                mean: m1,
                ..
            },
            BootstrapOutcome::Interval {
                lower: l2,
                upper: u2,
                mean: m2,
                ..
            },
        ) = (
            bootstrap(&a, &config, 5).unwrap(),
            bootstrap(&doubled, &config, 5).unwrap(),
        )
        else {
            panic!("expected intervals");
        };
        assert!((l2 - 2.0 * l1).abs() < 1e-12);
        assert!((u2 - 2.0 * u1).abs() < 1e-12);
        assert!((m2 - 2.0 * m1).abs() < 1e-12);
        assert!(l1 <= m1 && m1 <= u1);
    }

    #[test]
    fn test_block_bootstrap_is_the_harness_bootstrap_under_its_seed() {
        let series: Vec<Decimal> = [3, -1, 4, 1, -5, 9, 2, -6, 5, 3].map(dec).to_vec();
        let seed = derived_seed(11, WindowKind::Validation, "scenario");
        for block in [1, 2, 5] {
            let harness = config(block);
            let given = BootstrapConfig { seed, ..harness };
            assert_eq!(
                block_bootstrap(&series, &given).unwrap(),
                bootstrap(&series, &harness, seed).unwrap()
            );
        }
        // Another seed resamples other days.
        let other = BootstrapConfig {
            seed: seed.wrapping_add(1),
            ..config(2)
        };
        assert_ne!(
            block_bootstrap(&series, &other).unwrap(),
            bootstrap(&series, &config(2), seed).unwrap()
        );
    }

    #[test]
    fn test_block_bootstrap_refuses_what_the_harness_refuses() {
        let series = [dec(1), dec(2), dec(3)];
        for bad in [
            BootstrapConfig {
                replicates: 0,
                ..config(2)
            },
            BootstrapConfig {
                mean_block_days: 0,
                ..config(2)
            },
            BootstrapConfig {
                confidence_percent: 100,
                ..config(2)
            },
            BootstrapConfig {
                min_days: 1,
                ..config(2)
            },
        ] {
            assert!(
                matches!(
                    block_bootstrap(&series, &bad),
                    Err(EvaluationError::InvalidInput { .. })
                ),
                "{bad:?}"
            );
        }
        let mut few = config(2);
        few.min_days = 4;
        assert_eq!(
            block_bootstrap(&series, &few).unwrap(),
            BootstrapOutcome::Insufficient {
                days: 3,
                required: 4
            }
        );
    }

    #[test]
    fn test_block_bootstrap_resamples_series_of_one_length_on_the_same_days() {
        let a: Vec<f64> = (0..12).map(|i| f64::from(i * i % 7)).collect();
        let b: Vec<f64> = (0..12).map(|i| f64::from(3 - i % 5)).collect();
        let difference: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let (ma, mb, md) = (
            resampled_means(&a, &config(3), 9),
            resampled_means(&b, &config(3), 9),
            resampled_means(&difference, &config(3), 9),
        );
        for ((x, y), d) in ma.iter().zip(&mb).zip(&md) {
            assert!((x - y - d).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bootstrap_too_few_days_is_insufficient_not_an_interval() {
        let mut config = config(2);
        config.min_days = 3;
        assert_eq!(
            bootstrap(&[dec(1), dec(2)], &config, 5).unwrap(),
            BootstrapOutcome::Insufficient {
                days: 2,
                required: 3
            }
        );
    }

    #[test]
    fn test_bootstrap_config_refuses_invalid_values() {
        let good = config(2);
        good.validate().unwrap();
        for bad in [
            BootstrapConfig {
                replicates: 0,
                ..good
            },
            BootstrapConfig {
                replicates: MAX_REPLICATES + 1,
                ..good
            },
            BootstrapConfig {
                mean_block_days: 0,
                ..good
            },
            BootstrapConfig {
                confidence_percent: 0,
                ..good
            },
            BootstrapConfig {
                confidence_percent: 100,
                ..good
            },
            BootstrapConfig {
                min_days: 1,
                ..good
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    fn dec(value: i64) -> Decimal {
        Decimal::from(value)
    }

    #[test]
    fn test_interval_bounds_are_the_declared_order_statistics() {
        // 90 % of 500 replicates leaves (100 - 90) * 500 / 200 = 25 on each side.
        let series: Vec<Decimal> = (0..20).map(|i| Decimal::from(i * i % 13)).collect();
        let values: Vec<f64> = series.iter().map(|v| v.to_f64().unwrap()).collect();
        let mut means = resampled_means(&values, &config(2), 5);
        means.sort_by(f64::total_cmp);
        let BootstrapOutcome::Interval { lower, upper, .. } =
            bootstrap(&series, &config(2), 5).unwrap()
        else {
            panic!("no interval");
        };
        assert_eq!(lower, means[25]);
        assert_eq!(upper, means[474]);
        assert!(
            means[25] < means[50],
            "the fixture must tell the ranks apart"
        );
    }

    fn tampered_rows(tamper: fn(&mut BacktestResult)) -> Vec<EvaluationRow> {
        let real = Scenario::new("tampered", SimLatencyConfig::new(), BTreeMap::new(), |_| {
            crate::backtest::ImmediateFillModel::new()
        });
        let run = real.run;
        let scenario = Scenario {
            run: Box::new(move |request| {
                let mut raw = run(request)?;
                tamper(&mut raw.result);
                Ok(raw)
            }),
            name: real.name,
            latency: real.latency,
            fee_rate: real.fee_rate,
            parameters: real.parameters,
            fill_model_description: real.fill_model_description,
        };
        let day = 86_400_000_u64;
        let ticks: Vec<MarketTick> = (0..72_u64)
            .map(|i| {
                let mid = Decimal::from(100 + (i % 5) * 2);
                MarketTick::new(
                    i * 3_600_000,
                    mid - Decimal::ONE,
                    Decimal::ONE,
                    mid + Decimal::ONE,
                    Decimal::ONE,
                )
            })
            .collect();
        let config = ExperimentConfig {
            name: "tampered".to_string(),
            execution: BacktestConfig::new()
                .with_tick_size(Decimal::ONE)
                .with_lot_size(Decimal::ONE),
            splits: Splits::new(0, day, 2 * day, 3 * day).unwrap(),
            metrics: MetricsSpec {
                horizons_ms: vec![3_600_000],
                max_mark_delay_ms: 0,
            },
            bootstrap: config(1),
            seed: 1,
            provenance: Provenance {
                commit: "c".to_string(),
                toolchain: "t".to_string(),
            },
        };
        let candidates = vec![Candidate::fixed_spread(Decimal::ONE, Decimal::ONE)];
        Experiment::new(config, ticks, candidates, vec![scenario])
            .unwrap()
            .evaluate_validation()
            .report()
            .rows
            .clone()
    }

    fn failure(tamper: fn(&mut BacktestResult)) -> String {
        match &tampered_rows(tamper)[0].outcome {
            RowOutcome::Failed { reason } => reason.clone(),
            other => panic!("a bad run was reported as a result: {other:?}"),
        }
    }

    #[test]
    fn test_an_untampered_run_completes() {
        let rows = tampered_rows(|_| {});
        assert!(matches!(rows[0].outcome, RowOutcome::Completed(_)));
    }

    #[test]
    fn test_an_inconsistent_run_is_a_failed_row() {
        let reason = failure(|r| r.accounting.dropped_fills = 1);
        assert!(reason.starts_with("accounting"), "{reason}");
    }

    #[test]
    fn test_a_run_that_stops_early_is_a_failed_row() {
        let reason = failure(|r| r.num_ticks -= 1);
        assert!(reason.contains("the run stopped"), "{reason}");
    }

    #[test]
    fn test_a_run_that_does_not_reconcile_is_a_failed_row() {
        let reason = failure(|r| r.net_pnl += Decimal::new(1, 8));
        assert!(reason.contains("does not reconcile"), "{reason}");
        let reason = failure(|r| r.final_position += Decimal::ONE);
        assert!(reason.contains("does not reconcile"), "{reason}");
    }
}
