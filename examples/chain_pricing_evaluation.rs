//! Measures the chain's replaced pricing with the evaluation harness (#95).
//!
//! The chain used to take its theoretical value from a formula with no expiry and no
//! volatility, the intrinsic value plus `spot / 50`. It now takes it from a surface. Both
//! arms here go through the same `ChainMarketMaker::refresh_all_quotes`, so they share the
//! spread, the size and the rounding, and differ only in that value.
//!
//! Everything is SYNTHETIC, and says nothing about a market. One option contract runs per
//! experiment, on hourly ticks: a seeded underlying, and option quotes from a declared market.
//! In the first world that market is the surface's own model, so the surface doing well there
//! is expected. In the second it has a volatility premium and a skew that a flat surface at the
//! realized volatility misses. Every arm is declared here and reaches the holdout, so nothing
//! is selected. The volatility comes from the training window's underlying alone.
//!
//! Three fill rules set what a wrong quote costs, and each path reaches its holdout under each.
//! At the touch, anything crossed fills at the market's price, the best case for the quoter. At
//! the limit, the harness's own `ImmediateFillModel`, it fills at the quote's price, the worst.
//! On arrival is the nearest to a venue: a quote that crosses the market it arrives in takes that
//! touch, and one that rests until the market moves through it trades at its own limit.
//!
//! Quotes are replaced every hour, and a tick the chain cannot quote leaves the last quote
//! resting; those ticks are counted and printed. Hourly ticks hide any latency under an hour, so
//! none is modelled. The position is never hedged, and is valued at the contract's own mid. The
//! same `LimitsPolicy` runs for every arm, and it counts an unfilled quote until it is replaced,
//! so how near an arm gets to the limit depends on how often it fills. On validation only, the
//! reference rule also runs at twice the fee, and with a one half chance of a fill at the limit.
//!
//! Run with
//! `cargo run --release --features chain --example chain_pricing_evaluation -- <commit> <toolchain>`.
//! The commit and the toolchain go to the manifests as declared; they are not checked.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use financial_types::{Action, OptionStyle};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    BootstrapConfig, Candidate, EvaluationRow, Experiment, ExperimentConfig, HoldoutReport,
    MetricsSpec, Provenance, RowOutcome, Scenario, Splits,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestStrategy, FillModel, FillResult, ImmediateFillModel, MarketTick,
    ProbabilisticFillModel, SimLatencyConfig, SimulatedFill, SimulatedOrder,
};
use market_maker_rs::chain::{
    ChainMarketMaker, ChainMarketMakerConfig, ChainPricing, ChainQuoteOutcome,
};
use market_maker_rs::dec;
use market_maker_rs::options::pricing::value;
use market_maker_rs::options::{
    Dynamics, FlatSurface, OptionInputs, OptionPricer, OptionValuation, PRICING_YEAR_DAYS,
    SurfacePricer,
};
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::LimitsPolicy;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};
use option_chain_orderbook::orderbook::ExpirationOrderBook;
use optionstratlib::model::ExpirationDate;
use optionstratlib::prelude::pos_or_panic;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const HOUR: u64 = 3_600_000;
const DAY: u64 = 24 * HOUR;
const DAYS: u64 = 60;
const EXPIRY_MS: u64 = 180 * DAY;
const VALIDATION_START: u64 = 20 * DAY;
const HOLDOUT_START: u64 = 40 * DAY;
const END: u64 = DAYS * DAY;
const SPOT_START: f64 = 10_000.0;
const TRUE_SIGMA: f64 = 0.4;
const HOURS_PER_YEAR: f64 = 8_760.0;
const PATHS: u64 = 20;
/// Fee per unit of notional, for every arm.
const FEE: Decimal = dec!(0.0001);
/// The reference fill rule at twice [`FEE`], for sensitivity on validation.
const DOUBLE_FEE: &str = "double-fee";
const INTERVAL_DRAWS: usize = 4_000;
/// The arms of the surface, as multiples of the training window's realized volatility.
const VOL_MULTIPLES: [Decimal; 3] = [dec!(0.75), dec!(1), dec!(1.25)];
const PREVIOUS: &str = "previous-formula";
/// The surface at the realized volatility, which the difference is taken against.
const SURFACE: &str = "surface-x1";

/// One option of the study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Contract {
    name: &'static str,
    strike: u64,
    style: OptionStyle,
}

const CONTRACTS: [Contract; 3] = [
    Contract {
        name: "put 8500",
        strike: 8_500,
        style: OptionStyle::Put,
    },
    Contract {
        name: "call 10000",
        strike: 10_000,
        style: OptionStyle::Call,
    },
    Contract {
        name: "call 11500",
        strike: 11_500,
        style: OptionStyle::Call,
    },
];

/// How the market prices a contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum World {
    /// A flat volatility, the underlying's own.
    TheModel,
    /// A premium of a quarter over the underlying's volatility, and a skew by strike.
    PremiumAndSkew,
}

const WORLDS: [World; 2] = [World::TheModel, World::PremiumAndSkew];

impl World {
    /// The market's volatility for a strike: it stays at that strike as the spot moves.
    fn market_sigma(self, strike: u64) -> f64 {
        match self {
            Self::TheModel => TRUE_SIGMA,
            Self::PremiumAndSkew => {
                let log_moneyness = (strike as f64 / SPOT_START).ln();
                TRUE_SIGMA * 1.25 * (1.0 - 0.6 * log_moneyness)
            }
        }
    }
}

/// The formula the chain used before: the intrinsic value plus `spot / 50`, in whole price
/// units. It has no Greeks, which the quoting does not read.
#[derive(Debug, Clone, Copy)]
struct RemovedFormula;

impl OptionPricer for RemovedFormula {
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
        let intrinsic = match inputs.style {
            OptionStyle::Call => inputs.spot - inputs.strike,
            OptionStyle::Put => inputs.strike - inputs.spot,
        }
        .max(Decimal::ZERO);
        Ok(OptionValuation {
            price: intrinsic + (inputs.spot / dec!(50)).trunc(),
            delta: Decimal::ZERO,
            gamma: Decimal::ZERO,
            vega: Decimal::ZERO,
            theta: Decimal::ZERO,
        })
    }
}

/// A crossed quote takes the market's price: the best case for the quoter. See the module docs.
#[derive(Debug, Clone, Copy)]
struct TouchFillModel;

impl FillModel for TouchFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        match order.side {
            Action::Buy if tick.ask_price <= order.price => Ok(FillResult::FullFill {
                fill_price: tick.ask_price,
            }),
            Action::Sell if tick.bid_price >= order.price => Ok(FillResult::FullFill {
                fill_price: tick.bid_price,
            }),
            Action::Buy | Action::Sell => Ok(FillResult::NoFill),
            action => Err(MMError::UnsupportedAction { action }),
        }
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "TouchFill"
    }

    fn description(&self) -> String {
        "TouchFill: a quote at or through the market's touch fills in full at the touch".to_string()
    }
}

/// A quote that crossed the market it arrived in takes that touch, and one that rested until
/// the market moved through it trades at its own limit.
#[derive(Debug, Clone, Default)]
struct ArrivalFillModel {
    arrival: Option<MarketTick>,
    current: Option<MarketTick>,
}

impl FillModel for ArrivalFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        let arrived = self
            .arrival
            .as_ref()
            .filter(|arrival| arrival.timestamp == order.submitted_at);
        Ok(match order.side {
            Action::Buy => match arrived {
                Some(market) if market.ask_price <= order.price => FillResult::FullFill {
                    fill_price: market.ask_price,
                },
                _ if tick.ask_price <= order.price => FillResult::FullFill {
                    fill_price: order.price,
                },
                _ => FillResult::NoFill,
            },
            Action::Sell => match arrived {
                Some(market) if market.bid_price >= order.price => FillResult::FullFill {
                    fill_price: market.bid_price,
                },
                _ if tick.bid_price >= order.price => FillResult::FullFill {
                    fill_price: order.price,
                },
                _ => FillResult::NoFill,
            },
            action => return Err(MMError::UnsupportedAction { action }),
        })
    }

    fn on_tick(&mut self, tick: &MarketTick) {
        self.arrival = self.current.take();
        self.current = Some(tick.clone());
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn name(&self) -> &'static str {
        "ArrivalFill"
    }

    fn description(&self) -> String {
        "ArrivalFill: a quote crossing the market it arrived in takes that touch; one the market \
         later moves through fills at its limit"
            .to_string()
    }
}

/// A small generator of its own, so that a path depends on its seed alone.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

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

/// One step of a driftless lognormal: its expected next level is `level`.
fn next_level(level: f64, step: f64, normal: f64) -> f64 {
    level * (step * normal - 0.5 * step * step).exp()
}

/// The underlying at each hour, in whole price units. It holds the whole path: a quoter reads
/// it at the time of its tick, and `test_a_quote_does_not_depend_on_the_tape_after_its_tick`
/// checks that nothing later reaches it. A surface arm reads the training window only.
#[derive(Debug, Clone)]
struct Spots(Rc<Vec<u64>>);

impl Spots {
    fn simulate(seed: u64) -> MMResult<Self> {
        let step = TRUE_SIGMA / HOURS_PER_YEAR.sqrt();
        let mut noise = Noise::new(seed);
        let mut level = SPOT_START;
        let hours = DAYS * 24;
        let mut spots = Vec::new();
        for _ in 0..hours {
            spots.push(
                level
                    .round()
                    .to_u64()
                    .ok_or_else(|| MMError::NumericalError(format!("spot {level}")))?,
            );
            level = next_level(level, step, noise.normal());
        }
        Ok(Self(Rc::new(spots)))
    }

    fn at(&self, timestamp: u64) -> Option<u64> {
        self.0.get(usize::try_from(timestamp / HOUR).ok()?).copied()
    }

    /// The annualised volatility of the hourly log returns before `end_ms`.
    fn realized_before(&self, end_ms: u64) -> MMResult<Decimal> {
        let returns: Vec<f64> = (0..)
            .map(|hour: u64| hour * HOUR)
            .take_while(|t| *t + HOUR < end_ms)
            .filter_map(|t| {
                let (a, b) = (self.at(t)?, self.at(t + HOUR)?);
                Some((b as f64 / a as f64).ln())
            })
            .collect();
        if returns.len() < 2 {
            return Err(MMError::InvalidMarketState(
                "too few training returns".to_string(),
            ));
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Decimal::from_f64((variance * HOURS_PER_YEAR).sqrt())
            .map(|sigma| sigma.round_dp(6))
            .filter(|sigma| *sigma > Decimal::ZERO)
            .ok_or_else(|| MMError::NumericalError("realized volatility".to_string()))
    }
}

/// Years from `timestamp` to the expiry, in the pricing year.
fn tau_at(timestamp: u64) -> MMResult<Decimal> {
    let left = EXPIRY_MS
        .checked_sub(timestamp)
        .ok_or_else(|| MMError::InvalidTimestamp(format!("{timestamp} is past the expiry")))?;
    Decimal::from(left)
        .checked_div(PRICING_YEAR_DAYS * Decimal::from(DAY))
        .ok_or_else(|| MMError::NumericalError("time to expiry".to_string()))
}

/// The market's quote at one instant: its model value at two decimals, 1% either side and at
/// least one price unit, with a depth of 100.
fn market_tick(
    spot: u64,
    contract: Contract,
    world: World,
    timestamp: u64,
) -> MMResult<MarketTick> {
    let sigma = Decimal::from_f64(world.market_sigma(contract.strike))
        .ok_or_else(|| MMError::NumericalError("market volatility".to_string()))?;
    let inputs = OptionInputs::european(
        Decimal::from(spot),
        Decimal::from(contract.strike),
        tau_at(timestamp)?,
        Decimal::ZERO,
        Decimal::ZERO,
        contract.style,
    );
    let price = value(
        &inputs,
        &FlatSurface::new(sigma)?,
        Dynamics::StickyMoneyness,
    )?
    .price;
    let half = (price * dec!(0.01)).max(Decimal::ONE);
    Ok(MarketTick::new(
        timestamp,
        (price - half).max(dec!(0.01)).round_dp(2),
        dec!(100),
        (price + half).round_dp(2),
        dec!(100),
    ))
}

fn market_ticks(spots: &Spots, contract: Contract, world: World) -> MMResult<Vec<MarketTick>> {
    spots
        .0
        .iter()
        .zip(0_u64..)
        .map(|(spot, hour)| market_tick(*spot, contract, world, hour * HOUR))
        .collect()
}

/// Quotes one contract through the chain, whichever pricer it is given.
struct ChainQuoter {
    maker: ChainMarketMaker,
    pricer: Arc<dyn OptionPricer>,
    spots: Spots,
    contract: Contract,
    unquoted: Rc<Cell<u64>>,
}

impl ChainQuoter {
    /// The chain holds every strike of the study, so which one is at the money follows the spot.
    fn new(
        pricer: Arc<dyn OptionPricer>,
        spots: Spots,
        contract: Contract,
        unquoted: Rc<Cell<u64>>,
    ) -> Self {
        let chain = ExpirationOrderBook::new("SYN", ExpirationDate::Days(pos_or_panic!(180.0)));
        for listed in CONTRACTS {
            chain.get_or_create_strike(listed.strike);
        }
        let config = ChainMarketMakerConfig {
            base_quote_size: 1,
            ..ChainMarketMakerConfig::default()
        };
        Self {
            maker: ChainMarketMaker::new(Arc::new(chain), config),
            pricer,
            spots,
            contract,
            unquoted,
        }
    }

    /// The chain's quote at this instant, or nothing if it does not quote the contract.
    fn chain_quote(&self, timestamp: u64) -> Option<Quote> {
        let spot = self.spots.at(timestamp)?;
        let pricing = ChainPricing::new(
            Arc::clone(&self.pricer),
            tau_at(timestamp).ok()?,
            Decimal::ZERO,
            Decimal::ZERO,
        )
        .ok()?;
        let outcomes = self.maker.refresh_all_quotes(spot, &pricing).ok()?;
        outcomes.into_iter().find_map(|outcome| match outcome {
            ChainQuoteOutcome::Quote(quote)
                if quote.strike == self.contract.strike && quote.style == self.contract.style =>
            {
                Some(Quote {
                    bid_price: Decimal::from(quote.bid_price),
                    bid_size: Decimal::from(quote.bid_size),
                    ask_price: Decimal::from(quote.ask_price),
                    ask_size: Decimal::from(quote.ask_size),
                    timestamp,
                })
            }
            _ => None,
        })
    }
}

impl BacktestStrategy for ChainQuoter {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        let quote = self.chain_quote(tick.timestamp);
        if quote.is_none() {
            self.unquoted.set(self.unquoted.get() + 1);
        }
        quote
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

fn surface_name(multiple: Decimal) -> String {
    format!("surface-x{}", multiple.normalize())
}

fn parameters(pairs: &[(&str, String)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn previous_arm(spots: &Spots, contract: Contract, unquoted: &Rc<Cell<u64>>) -> Candidate {
    let spots = spots.clone();
    let unquoted = Rc::clone(unquoted);
    Candidate::model(
        PREVIOUS,
        parameters(&[("value", "intrinsic + spot / 50".to_string())]),
        move |_train| {
            Ok(Box::new(ChainQuoter::new(
                Arc::new(RemovedFormula),
                spots.clone(),
                contract,
                Rc::clone(&unquoted),
            )) as Box<dyn BacktestStrategy>)
        },
    )
}

/// A surface arm's volatility, from the training window's underlying alone.
fn surface_sigma(spots: &Spots, multiple: Decimal) -> MMResult<Decimal> {
    Ok(spots.realized_before(VALIDATION_START)? * multiple)
}

fn surface_pricer(spots: &Spots, multiple: Decimal) -> MMResult<Arc<dyn OptionPricer>> {
    Ok(Arc::new(SurfacePricer::new(
        Arc::new(FlatSurface::new(surface_sigma(spots, multiple)?)?),
        Dynamics::StickyMoneyness,
    )?))
}

fn surface_arm(
    spots: &Spots,
    contract: Contract,
    multiple: Decimal,
    unquoted: &Rc<Cell<u64>>,
) -> Candidate {
    let spots = spots.clone();
    let unquoted = Rc::clone(unquoted);
    Candidate::model(
        surface_name(multiple),
        parameters(&[
            ("surface", "flat".to_string()),
            ("realized_vol_multiple", multiple.normalize().to_string()),
            ("realized_vol_before_ms", VALIDATION_START.to_string()),
        ]),
        move |_train| {
            Ok(Box::new(ChainQuoter::new(
                surface_pricer(&spots, multiple)?,
                spots.clone(),
                contract,
                Rc::clone(&unquoted),
            )) as Box<dyn BacktestStrategy>)
        },
    )
}

fn arm_names() -> Vec<String> {
    let mut names = vec![PREVIOUS.to_string()];
    names.extend(VOL_MULTIPLES.into_iter().map(surface_name));
    names
}

/// Which fill rule an experiment reaches its holdout under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillRule {
    Arrival,
    Touch,
    Limit,
}

const FILL_RULES: [FillRule; 3] = [FillRule::Arrival, FillRule::Touch, FillRule::Limit];

impl FillRule {
    const fn name(self) -> &'static str {
        match self {
            Self::Arrival => "arrival",
            Self::Touch => "touch",
            Self::Limit => "limit-price",
        }
    }

    fn scenario(self) -> Scenario {
        self.named(self.name())
    }

    fn named(self, name: &str) -> Scenario {
        match self {
            Self::Arrival => Scenario::new(name, SimLatencyConfig::new(), BTreeMap::new(), |_| {
                ArrivalFillModel::default()
            }),
            Self::Touch => Scenario::new(name, SimLatencyConfig::new(), BTreeMap::new(), |_| {
                TouchFillModel
            }),
            Self::Limit => Scenario::new(name, SimLatencyConfig::new(), BTreeMap::new(), |_| {
                ImmediateFillModel::new()
            }),
        }
    }
}

/// The reference first, then the other rules, then a chance of a fill at the limit, and the
/// reference again at twice the fee.
fn scenarios(reference: FillRule) -> Vec<Scenario> {
    let others = FILL_RULES.into_iter().filter(|rule| *rule != reference);
    let mut scenarios = vec![reference.scenario()];
    scenarios.extend(others.map(FillRule::scenario));
    scenarios.push(Scenario::new(
        "half-fill-chance",
        SimLatencyConfig::new(),
        parameters(&[
            ("base_probability", "0.5".to_string()),
            ("time_factor", "1".to_string()),
        ]),
        |seed| ProbabilisticFillModel::new(dec!(0.5), dec!(0), dec!(1), seed),
    ));
    scenarios.push(reference.named(DOUBLE_FEE).with_fee_rate(FEE * dec!(2)));
    scenarios
}

fn config(provenance: &Provenance, seed: u64) -> MMResult<ExperimentConfig> {
    Ok(ExperimentConfig {
        name: "chain-pricing-synthetic".to_string(),
        execution: BacktestConfig::new()
            .with_initial_capital(dec!(10000000))
            .with_fee_rate(FEE)
            .with_tick_size(dec!(0.01))
            .with_lot_size(dec!(1)),
        splits: Splits::new(0, VALIDATION_START, HOLDOUT_START, END)
            .map_err(|error| MMError::InvalidConfiguration(error.to_string()))?,
        metrics: MetricsSpec {
            horizons_ms: vec![HOUR, 4 * HOUR],
            max_mark_delay_ms: 0,
        },
        bootstrap: BootstrapConfig {
            replicates: 500,
            mean_block_days: 3,
            confidence_percent: 90,
            seed: 1,
            min_days: 10,
        },
        seed,
        provenance: provenance.clone(),
    })
}

/// One experiment. Every arm is declared and selected, so what reaches the holdout was chosen
/// before any result. Returns the report and the ticks no arm could quote.
fn run_path(
    world: World,
    contract: Contract,
    seed: u64,
    rule: FillRule,
    provenance: &Provenance,
) -> MMResult<(HoldoutReport, u64)> {
    let spots = Spots::simulate(seed)?;
    let ticks = market_ticks(&spots, contract, world)?;
    let first_mid = ticks
        .first()
        .map(MarketTick::mid_price)
        .ok_or_else(|| MMError::InvalidMarketState("no ticks".to_string()))?;
    let unquoted = Rc::new(Cell::new(0));
    let mut candidates = vec![
        Candidate::no_trade(),
        Candidate::fixed_spread((first_mid * dec!(0.02)).round_dp(2), dec!(1)),
        previous_arm(&spots, contract, &unquoted),
    ];
    candidates.extend(
        VOL_MULTIPLES
            .into_iter()
            .map(|multiple| surface_arm(&spots, contract, multiple, &unquoted)),
    );
    let names = arm_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let to_error = |error: market_maker_rs::backtest::evaluation::EvaluationError| {
        MMError::InvalidConfiguration(error.to_string())
    };
    let report = Experiment::new(
        config(provenance, seed)?,
        ticks,
        candidates,
        scenarios(rule),
    )
    .map_err(to_error)?
    .with_pretrade_policy(|| {
        Box::new(LimitsPolicy::new(
            RiskLimits::new(dec!(20), dec!(1000000000), dec!(1))
                .expect("constant limits are valid"),
        ))
    })
    .evaluate_validation()
    .select(&names)
    .map_err(to_error)?
    .run_holdout();
    Ok((report, unquoted.get()))
}

/// The mean distance of a quote's mid to the market's, as a share of the market's, over the
/// ticks at or after `from_ms`. This reads the pricing directly, before any fill.
fn quote_distance(quoter: &ChainQuoter, ticks: &[MarketTick], from_ms: u64) -> f64 {
    let shares: Vec<f64> = ticks
        .iter()
        .filter(|tick| tick.timestamp >= from_ms)
        .filter_map(|tick| {
            let quote = quoter.chain_quote(tick.timestamp)?;
            let mid = (quote.bid_price + quote.ask_price) / dec!(2);
            let market = tick.mid_price();
            Some(number(((mid - market) / market).abs()))
        })
        .collect();
    mean(&shares)
}

/// That distance on the holdout ticks of one path, per arm.
fn quote_distances(world: World, contract: Contract, seed: u64) -> MMResult<BTreeMap<String, f64>> {
    let spots = Spots::simulate(seed)?;
    let ticks = market_ticks(&spots, contract, world)?;
    let mut pricers: Vec<(String, Arc<dyn OptionPricer>)> =
        vec![(PREVIOUS.to_string(), Arc::new(RemovedFormula))];
    for multiple in VOL_MULTIPLES {
        pricers.push((surface_name(multiple), surface_pricer(&spots, multiple)?));
    }
    Ok(pricers
        .into_iter()
        .map(|(name, pricer)| {
            let quoter = ChainQuoter::new(pricer, spots.clone(), contract, Rc::new(Cell::new(0)));
            (name, quote_distance(&quoter, &ticks, HOLDOUT_START))
        })
        .collect())
}

/// What one row of one path adds to a table.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Measure {
    net: f64,
    capture: f64,
    inventory: f64,
    drawdown: f64,
    fills: f64,
    mean_abs: f64,
    peak: f64,
}

fn number(value: Decimal) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn measure_of(row: &EvaluationRow) -> Result<Measure, String> {
    match &row.outcome {
        RowOutcome::Completed(metrics) => {
            let m = &metrics.metrics;
            Ok(Measure {
                net: number(m.attribution.net_pnl),
                capture: number(m.attribution.spread_capture),
                inventory: number(m.attribution.inventory_pnl),
                drawdown: number(m.max_drawdown),
                fills: m.fills as f64,
                mean_abs: m.inventory.mean_abs.map_or(f64::NAN, number),
                peak: number(m.inventory.max_abs),
            })
        }
        RowOutcome::Failed { reason } => Err(reason.clone()),
        other => Err(format!("{other:?}")),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The mean and a 90% interval, resampling whole paths, which are independent.
fn interval(values: &[f64]) -> (f64, f64, f64) {
    let mut noise = Noise::new(77);
    let mut draws: Vec<f64> = (0..INTERVAL_DRAWS)
        .map(|_| {
            let sample: Vec<f64> = values
                .iter()
                .map(|_| values[(noise.unit() * values.len() as f64) as usize])
                .collect();
            mean(&sample)
        })
        .collect();
    draws.sort_by(f64::total_cmp);
    (
        mean(values),
        draws[INTERVAL_DRAWS / 20],
        draws[INTERVAL_DRAWS - 1 - INTERVAL_DRAWS / 20],
    )
}

/// The rows of one world, contract and fill rule, per arm and path.
#[derive(Debug, Default)]
struct Tally {
    holdout: BTreeMap<String, BTreeMap<u64, Measure>>,
    /// By scenario and arm: the net on validation.
    validation: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    failed: Vec<String>,
    data_hashes: Vec<String>,
    unquoted: u64,
}

impl Tally {
    fn holdout_row(
        &mut self,
        seed: u64,
        arm: &str,
        scenario: &str,
        measured: Result<Measure, String>,
    ) {
        match measured {
            Ok(measure) => {
                self.holdout
                    .entry(arm.to_string())
                    .or_default()
                    .insert(seed, measure);
            }
            Err(reason) => self
                .failed
                .push(format!("path {seed} {arm} {scenario} holdout: {reason}")),
        }
    }

    fn validation_row(
        &mut self,
        seed: u64,
        arm: &str,
        scenario: &str,
        measured: Result<Measure, String>,
    ) {
        match measured {
            Ok(measure) => self
                .validation
                .entry(scenario.to_string())
                .or_default()
                .entry(arm.to_string())
                .or_default()
                .push(measure.net),
            Err(reason) => self
                .failed
                .push(format!("path {seed} {arm} {scenario} validation: {reason}")),
        }
    }

    fn add(&mut self, seed: u64, report: &HoldoutReport, unquoted: u64) {
        for row in &report.holdout {
            self.holdout_row(seed, &row.candidate, &row.scenario, measure_of(row));
        }
        for row in &report.validation {
            self.validation_row(seed, &row.candidate, &row.scenario, measure_of(row));
        }
        self.data_hashes
            .extend(report.manifest.data_hashes.values().cloned());
        self.unquoted += unquoted;
    }

    /// The difference on each path where both arms completed, so that a failed row cannot shift
    /// the pairs.
    fn differences(&self, arm: &str, base: &str, field: fn(&Measure) -> f64) -> Vec<f64> {
        match (self.holdout.get(arm), self.holdout.get(base)) {
            (Some(a), Some(b)) => a
                .iter()
                .filter_map(|(seed, x)| b.get(seed).map(|y| field(x) - field(y)))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn pick(rows: &[Measure], field: fn(&Measure) -> f64) -> Vec<f64> {
    rows.iter().map(field).collect()
}

/// A line of the holdout table.
fn holdout_line(
    world: World,
    contract: Contract,
    rule: FillRule,
    arm: &str,
    rows: &[Measure],
) -> String {
    let (net, low, high) = interval(&pick(rows, |m| m.net));
    let peak = pick(rows, |m| m.peak).into_iter().fold(0.0_f64, f64::max);
    format!(
        "| {world:?} | {} | {} | {arm} | {net:.1} [{low:.1}, {high:.1}] | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {peak} |",
        contract.name,
        rule.name(),
        mean(&pick(rows, |m| m.capture)),
        mean(&pick(rows, |m| m.inventory)),
        mean(&pick(rows, |m| m.drawdown)),
        mean(&pick(rows, |m| m.fills)),
        mean(&pick(rows, |m| m.mean_abs)),
    )
}

/// FNV-1a 64 of the data hashes of a group of paths, for the record.
fn digest(hashes: &[String]) -> u64 {
    hashes
        .iter()
        .flat_map(|hash| hash.bytes())
        .fold(0xCBF2_9CE4_8422_2325_u64, |state, byte| {
            (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let provenance = Provenance {
        commit: args.next().unwrap_or_else(|| "undeclared".to_string()),
        toolchain: args.next().unwrap_or_else(|| "undeclared".to_string()),
    };
    println!(
        "SYNTHETIC: no statement about any market. {PATHS} paths per contract, net over the {} days of each window, 90% intervals resampling paths.\n",
        (END - HOLDOUT_START) / DAY
    );
    println!(
        "| world | contract | fills at | arm | holdout net | capture | inventory | drawdown | fills | mean abs position | peak |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut differences = Vec::new();
    let mut sensitivity = Vec::new();
    let mut records = Vec::new();
    let mut distances = Vec::new();
    for world in WORLDS {
        for contract in CONTRACTS {
            let mut by_arm: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for seed in 1..=PATHS {
                for (arm, distance) in quote_distances(world, contract, seed)? {
                    by_arm.entry(arm).or_default().push(distance * 100.0);
                }
            }
            for (arm, values) in &by_arm {
                let (distance, low, high) = interval(values);
                distances.push(format!(
                    "| {world:?} | {} | {arm} | {distance:.1} [{low:.1}, {high:.1}] |",
                    contract.name
                ));
            }
            for rule in FILL_RULES {
                let mut tally = Tally::default();
                for seed in 1..=PATHS {
                    let (report, unquoted) = run_path(world, contract, seed, rule, &provenance)?;
                    tally.add(seed, &report, unquoted);
                }
                for (arm, rows) in &tally.holdout {
                    let rows: Vec<Measure> = rows.values().copied().collect();
                    println!("{}", holdout_line(world, contract, rule, arm, &rows));
                }
                let gaps = tally.differences(SURFACE, PREVIOUS, |m| m.net);
                let (gap, low, high) = interval(&gaps);
                let (edge, edge_low, edge_high) =
                    interval(&tally.differences(SURFACE, PREVIOUS, |m| m.capture));
                differences.push(format!(
                    "| {world:?} | {} | {} | {gap:.1} [{low:.1}, {high:.1}] | {edge:.1} [{edge_low:.1}, {edge_high:.1}] | {} |",
                    contract.name,
                    rule.name(),
                    gaps.len()
                ));
                // Validation does not depend on which rule is the reference.
                if rule == FillRule::Arrival {
                    for (scenario, arms) in &tally.validation {
                        for (arm, nets) in arms {
                            let (net, low, high) = interval(nets);
                            sensitivity.push(format!(
                                "| {world:?} | {} | {scenario} | {arm} | {net:.1} [{low:.1}, {high:.1}] |",
                                contract.name
                            ));
                        }
                    }
                }
                records.push(format!(
                    "{world:?} {} {}: data hashes digest {:016x}, failed rows {}, ticks not quoted {}",
                    contract.name,
                    rule.name(),
                    digest(&tally.data_hashes),
                    tally.failed.len(),
                    tally.unquoted
                ));
                records.extend(tally.failed);
            }
        }
    }
    println!(
        "\n{SURFACE} minus {PREVIOUS}, holdout, paired by path:\n\n| world | contract | fills at | net | capture | paths |\n| --- | --- | --- | --- | --- | --- |"
    );
    for line in differences {
        println!("{line}");
    }
    println!(
        "\nDistance of the quote's mid to the market's on the holdout ticks, % of the market's:\n\n| world | contract | arm | distance |\n| --- | --- | --- | --- |"
    );
    for line in distances {
        println!("{line}");
    }
    println!(
        "\nValidation net under each scenario:\n\n| world | contract | scenario | arm | net |\n| --- | --- | --- | --- | --- |"
    );
    for line in sensitivity {
        println!("{line}");
    }
    println!("\nRecords:");
    for line in records {
        println!("{line}");
    }
    println!("\nManifest of the first path of the last group:");
    let (last, _) = run_path(
        World::PremiumAndSkew,
        CONTRACTS[2],
        1,
        FillRule::Limit,
        &provenance,
    )?;
    println!("{:#?}", last.manifest);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use market_maker_rs::options::pricing::OptionInputs as Inputs;

    const ATM: Contract = CONTRACTS[1];

    fn provenance() -> Provenance {
        Provenance {
            commit: "test".to_string(),
            toolchain: "test".to_string(),
        }
    }

    fn inputs(spot: Decimal, strike: Decimal, style: OptionStyle) -> Inputs {
        Inputs::european(spot, strike, dec!(0.5), Decimal::ZERO, Decimal::ZERO, style)
    }

    /// A tape the test sets by hand, from one flat level.
    fn flat_spots(level: u64) -> Spots {
        Spots(Rc::new(vec![level; (DAYS * 24) as usize]))
    }

    fn quoter(pricer: Arc<dyn OptionPricer>, spots: Spots, contract: Contract) -> ChainQuoter {
        ChainQuoter::new(pricer, spots, contract, Rc::new(Cell::new(0)))
    }

    fn quote_at(pricer: Arc<dyn OptionPricer>, spots: Spots, contract: Contract, t: u64) -> Quote {
        quoter(pricer, spots, contract)
            .chain_quote(t)
            .expect("a quote")
    }

    #[test]
    fn test_removed_formula_is_the_intrinsic_value_plus_a_fiftieth_of_the_spot() {
        let price = |spot, strike, style| {
            RemovedFormula
                .value(&inputs(spot, strike, style))
                .unwrap()
                .price
        };
        assert_eq!(
            price(dec!(10000), dec!(9000), OptionStyle::Call),
            dec!(1200)
        );
        assert_eq!(
            price(dec!(10000), dec!(11000), OptionStyle::Call),
            dec!(200)
        );
        assert_eq!(
            price(dec!(10000), dec!(11000), OptionStyle::Put),
            dec!(1200)
        );
        assert_eq!(price(dec!(10000), dec!(9000), OptionStyle::Put), dec!(200));
        // A whole number of price units, as the old integer division gave.
        assert_eq!(
            price(dec!(10049), dec!(11000), OptionStyle::Call),
            dec!(200)
        );
    }

    #[test]
    fn test_the_previous_arm_quotes_through_the_chain_with_the_old_spread() {
        // Theo 200 at the money, 200 bps times the ATM multiple of 1.5 is 300 bps, so a half
        // spread of 200 * 300 / 20000 = 3, and the ATM size is twice the base of 1.
        let q = quote_at(Arc::new(RemovedFormula), flat_spots(10_000), ATM, 0);
        assert_eq!((q.bid_price, q.ask_price), (dec!(197), dec!(203)));
        assert_eq!((q.bid_size, q.ask_size), (dec!(2), dec!(2)));
    }

    #[test]
    fn test_the_surface_arm_prices_at_the_spot_and_the_time_of_its_tick() {
        let surface = || -> Arc<dyn OptionPricer> {
            Arc::new(
                SurfacePricer::new(
                    Arc::new(FlatSurface::new(dec!(0.4)).unwrap()),
                    Dynamics::StickyMoneyness,
                )
                .unwrap(),
            )
        };
        let spots = Spots(Rc::new(vec![10_000, 10_500, 10_500]));
        let theo = |t: u64, spot: u64| {
            value(
                &OptionInputs::european(
                    Decimal::from(spot),
                    dec!(10000),
                    tau_at(t).unwrap(),
                    Decimal::ZERO,
                    Decimal::ZERO,
                    OptionStyle::Call,
                ),
                &FlatSurface::new(dec!(0.4)).unwrap(),
                Dynamics::StickyMoneyness,
            )
            .unwrap()
            .price
        };
        let mid = |q: Quote| (q.bid_price + q.ask_price) / dec!(2);
        let first = mid(quote_at(surface(), spots.clone(), ATM, 0));
        let second = mid(quote_at(surface(), spots.clone(), ATM, HOUR));
        let later = mid(quote_at(surface(), spots, ATM, 2 * HOUR));
        // The half spread is whole units, so the mid is the theo, rounded to a unit.
        assert_eq!(first, theo(0, 10_000).round());
        assert_eq!(second, theo(HOUR, 10_500).round());
        assert_eq!(later, theo(2 * HOUR, 10_500).round());
        assert_ne!(first, second);
        // Time passes in the value too, seen on a contract far enough to move by a unit.
        let spots = flat_spots(10_000);
        let early = mid(quote_at(surface(), spots.clone(), ATM, 0));
        let late = mid(quote_at(surface(), spots, ATM, 50 * DAY));
        assert!(late < early, "{late} {early}");
    }

    #[test]
    fn test_a_strategy_cannot_read_a_spot_past_the_tape() {
        let quoter = quoter(Arc::new(RemovedFormula), Spots(Rc::new(vec![10_000])), ATM);
        assert!(quoter.chain_quote(0).is_some());
        assert!(quoter.chain_quote(HOUR).is_none());
    }

    #[test]
    fn test_a_tick_with_no_quote_is_counted_and_a_quoted_one_is_not() {
        let unquoted = Rc::new(Cell::new(0));
        let mut quoter = ChainQuoter::new(
            Arc::new(RemovedFormula),
            Spots(Rc::new(vec![10_000])),
            ATM,
            Rc::clone(&unquoted),
        );
        let position = InventoryPosition::new();
        let tick = |t| MarketTick::new(t, dec!(1), dec!(1), dec!(2), dec!(1));
        assert!(quoter.on_tick(&tick(0), &position).is_some());
        assert_eq!(unquoted.get(), 0);
        assert!(quoter.on_tick(&tick(HOUR), &position).is_none());
        assert_eq!(unquoted.get(), 1);
    }

    #[test]
    fn test_a_quote_does_not_depend_on_the_tape_after_its_tick() {
        let surface = || -> Arc<dyn OptionPricer> {
            Arc::new(
                SurfacePricer::new(
                    Arc::new(FlatSurface::new(dec!(0.4)).unwrap()),
                    Dynamics::StickyMoneyness,
                )
                .unwrap(),
            )
        };
        let tape = Spots::simulate(7).unwrap();
        let t = 100 * HOUR;
        let mut changed: Vec<u64> = tape.0.to_vec();
        changed[101..].iter_mut().for_each(|s| *s *= 2);
        let changed = Spots(Rc::new(changed));
        let at = |spots: &Spots, t| quote_at(surface(), spots.clone(), ATM, t);
        assert_eq!(at(&tape, t), at(&changed, t));
        assert_ne!(at(&tape, t + HOUR), at(&changed, t + HOUR));
    }

    #[test]
    fn test_only_the_strike_nearest_the_spot_is_quoted_as_at_the_money() {
        // Theo 200 away from the money: 200 bps times 1 + 0.5 * 0.15 is 215 bps, a half spread
        // of 2, and the base size of 1.
        let spots = flat_spots(10_000);
        let put = quote_at(Arc::new(RemovedFormula), spots, CONTRACTS[0], 0);
        assert_eq!((put.bid_price, put.ask_price), (dec!(198), dec!(202)));
        assert_eq!(put.bid_size, dec!(1));
        // Nearer to 11500, that strike is the one at the money, with the larger size.
        let high = quote_at(
            Arc::new(RemovedFormula),
            flat_spots(11_000),
            CONTRACTS[2],
            0,
        );
        assert_eq!(high.bid_size, dec!(2));
        let low = quote_at(Arc::new(RemovedFormula), flat_spots(11_000), ATM, 0);
        assert_eq!(low.bid_size, dec!(1));
    }

    #[test]
    fn test_the_quoter_picks_its_own_contract_of_the_chain() {
        // A call and a put share a strike: the intrinsic value tells them apart.
        let spots = flat_spots(11_000);
        let call = quote_at(Arc::new(RemovedFormula), spots.clone(), ATM, 0);
        let put = Contract {
            name: "put",
            strike: 10_000,
            style: OptionStyle::Put,
        };
        let put = quote_at(Arc::new(RemovedFormula), spots, put, 0);
        assert!(call.bid_price > dec!(1000), "{call:?}");
        assert!(put.ask_price < dec!(300), "{put:?}");
    }

    #[test]
    fn test_a_crossed_quote_takes_the_touch_and_a_quote_inside_rests() {
        let tick = MarketTick::new(0, dec!(10), dec!(1), dec!(12), dec!(1));
        let fill = |side, price| {
            TouchFillModel
                .simulate_fill(&SimulatedOrder::new(side, price, dec!(1), 0), &tick, 0)
                .unwrap()
        };
        let at = |fill_price| FillResult::FullFill { fill_price };
        assert_eq!(fill(Action::Buy, dec!(20)), at(dec!(12)));
        assert_eq!(fill(Action::Buy, dec!(12)), at(dec!(12)));
        assert_eq!(fill(Action::Buy, dec!(11)), FillResult::NoFill);
        assert_eq!(fill(Action::Sell, dec!(2)), at(dec!(10)));
        assert_eq!(fill(Action::Sell, dec!(10)), at(dec!(10)));
        assert_eq!(fill(Action::Sell, dec!(11)), FillResult::NoFill);
    }

    #[test]
    fn test_a_quote_takes_the_touch_only_of_the_market_it_arrived_in() {
        let market = |t, bid, ask| MarketTick::new(t, bid, dec!(1), ask, dec!(1));
        let mut model = ArrivalFillModel::default();
        // Placed at 100, in a market of 10 by 12, and evaluated on the next tick.
        model.on_tick(&market(100, dec!(10), dec!(12)));
        let next = market(200, dec!(20), dec!(22));
        model.on_tick(&next);
        let fill = |side, price, at| {
            model
                .simulate_fill(&SimulatedOrder::new(side, price, dec!(1), at), &next, 0)
                .unwrap()
        };
        let at = |fill_price| FillResult::FullFill { fill_price };
        // It crossed as it arrived: that touch, whatever the market did since.
        assert_eq!(fill(Action::Buy, dec!(13), 100), at(dec!(12)));
        assert_eq!(fill(Action::Buy, dec!(12), 100), at(dec!(12)));
        assert_eq!(fill(Action::Sell, dec!(9), 100), at(dec!(10)));
        assert_eq!(fill(Action::Sell, dec!(10), 100), at(dec!(10)));
        // It rested, and the market moved through it: its own limit.
        assert_eq!(fill(Action::Sell, dec!(15), 100), at(dec!(15)));
        // Up through a buy does not fill, and a buy the market came down to needs its own case.
        assert_eq!(fill(Action::Buy, dec!(11), 100), FillResult::NoFill);
        assert_eq!(fill(Action::Sell, dec!(21), 100), FillResult::NoFill);
        let lower = market(200, dec!(5), dec!(7));
        let fill_lower = model
            .simulate_fill(
                &SimulatedOrder::new(Action::Buy, dec!(8), dec!(1), 100),
                &lower,
                0,
            )
            .unwrap();
        assert_eq!(fill_lower, at(dec!(8)));
        // At the arrival touch exactly, it crossed, though the market has since moved away.
        let sold = model
            .simulate_fill(
                &SimulatedOrder::new(Action::Sell, dec!(10), dec!(1), 100),
                &lower,
                0,
            )
            .unwrap();
        assert_eq!(sold, at(dec!(10)));
        // An older order is not treated as just arrived, whatever that market was.
        assert_eq!(fill(Action::Sell, dec!(9), 50), at(dec!(9)));
        assert_eq!(fill(Action::Buy, dec!(23), 50), at(dec!(23)));
        // Nothing to remember after a reset, or before a second tick.
        let mut fresh = model.clone();
        fresh.reset();
        assert_eq!(fresh.arrival, None);
        assert_eq!(fresh.current, None);
    }

    #[test]
    fn test_the_distance_to_the_market_by_hand() {
        // The formula's theo is 200 at the money, with a mid of 200.
        let quoter = quoter(Arc::new(RemovedFormula), flat_spots(10_000), ATM);
        let tick =
            |t, mid: Decimal| MarketTick::new(t, mid - dec!(1), dec!(1), mid + dec!(1), dec!(1));
        let ticks = [
            tick(0, dec!(500)),
            tick(HOUR, dec!(400)),
            tick(2 * HOUR, dec!(100)),
        ];
        // 50% and 100%; the tick before the window is left out.
        assert!((quote_distance(&quoter, &ticks, HOUR) - 0.75).abs() < 1e-12);
        assert!((quote_distance(&quoter, &ticks[..1], 0) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_the_surface_is_nearer_the_market_than_the_formula_when_it_is_the_market() {
        let distances = quote_distances(World::TheModel, ATM, 1).unwrap();
        assert_eq!(distances.len(), 4);
        assert!(distances[SURFACE] < 0.1, "{distances:?}");
        assert!(distances[PREVIOUS] > 0.4, "{distances:?}");
        // Over the holdout ticks, not the whole path.
        let spots = Spots::simulate(1).unwrap();
        let ticks = market_ticks(&spots, ATM, World::TheModel).unwrap();
        let formula = quoter(Arc::new(RemovedFormula), spots, ATM);
        assert_eq!(
            distances[PREVIOUS],
            quote_distance(&formula, &ticks, HOLDOUT_START)
        );
        assert_ne!(distances[PREVIOUS], quote_distance(&formula, &ticks, 0));
        // The holdout ticks of the world asked for, not another's.
        let other = quote_distances(World::PremiumAndSkew, ATM, 1).unwrap();
        assert!(other[SURFACE] > distances[SURFACE], "{other:?}");
    }

    #[test]
    fn test_a_crossed_quote_pays_less_on_arrival_than_at_its_limit() {
        let run = |rule| {
            run_path(World::TheModel, ATM, 1, rule, &provenance())
                .unwrap()
                .0
        };
        // The formula is far under this market, so its fills crossed on arrival.
        let previous = |report: &HoldoutReport| holdout_measure(report, PREVIOUS).capture;
        assert!(previous(&run(FillRule::Arrival)) > previous(&run(FillRule::Limit)));
    }

    #[test]
    fn test_the_market_tick_is_its_model_value_with_a_declared_spread() {
        let tick = market_tick(10_000, ATM, World::TheModel, 0).unwrap();
        let price = value(
            &OptionInputs::european(
                dec!(10000),
                dec!(10000),
                tau_at(0).unwrap(),
                Decimal::ZERO,
                Decimal::ZERO,
                OptionStyle::Call,
            ),
            &FlatSurface::new(Decimal::from_f64(TRUE_SIGMA).unwrap()).unwrap(),
            Dynamics::StickyMoneyness,
        )
        .unwrap()
        .price;
        assert_eq!(tick.bid_price, (price * dec!(0.99)).round_dp(2));
        assert_eq!(tick.ask_price, (price * dec!(1.01)).round_dp(2));
        assert_eq!(tick.timestamp, 0);
        // The spread is one unit at least, and the bid stays positive.
        let far = Contract {
            name: "far",
            strike: 100_000,
            style: OptionStyle::Call,
        };
        let far = market_tick(10_000, far, World::TheModel, 0).unwrap();
        assert_eq!(far.bid_price, dec!(0.01));
        assert!(far.ask_price >= dec!(1));
    }

    #[test]
    fn test_the_worlds_price_a_strike_as_declared() {
        assert_eq!(World::TheModel.market_sigma(8_500), TRUE_SIGMA);
        assert_eq!(World::TheModel.market_sigma(11_500), TRUE_SIGMA);
        let skew = |strike| World::PremiumAndSkew.market_sigma(strike);
        assert!((skew(10_000) - 0.5).abs() < 1e-12);
        assert!(skew(8_500) > skew(10_000) && skew(10_000) > skew(11_500));
        let log = (8_500.0_f64 / 10_000.0).ln();
        assert!((skew(8_500) - 0.5 * (1.0 - 0.6 * log)).abs() < 1e-12);
    }

    #[test]
    fn test_time_to_expiry_is_in_the_pricing_year_and_refuses_the_past() {
        assert_eq!(tau_at(EXPIRY_MS).unwrap(), Decimal::ZERO);
        assert_eq!(
            tau_at(EXPIRY_MS - 365 * DAY / 365).unwrap(),
            Decimal::ONE / dec!(365)
        );
        assert!(tau_at(EXPIRY_MS + 1).is_err());
    }

    #[test]
    fn test_realized_volatility_reads_only_the_training_window() {
        let spots = Spots::simulate(5).unwrap();
        let before = spots.realized_before(VALIDATION_START).unwrap();
        let mut later: Vec<u64> = spots.0.to_vec();
        let at = (VALIDATION_START / HOUR) as usize;
        later[at..].iter_mut().for_each(|s| *s *= 3);
        // The last return it may read ends at the last hour before the window.
        assert_eq!(
            before,
            Spots(Rc::new(later.clone()))
                .realized_before(VALIDATION_START)
                .unwrap()
        );
        later[at - 1] *= 3;
        assert_ne!(
            before,
            Spots(Rc::new(later))
                .realized_before(VALIDATION_START)
                .unwrap()
        );
        // Close to the generating value over 480 hours: the standard error is under 4%.
        let sigma = before.to_f64().unwrap();
        assert!((sigma - TRUE_SIGMA).abs() < 0.1 * TRUE_SIGMA, "{sigma}");
    }

    #[test]
    fn test_realized_volatility_by_hand() {
        // Log returns of +ln 2 and -ln 2: mean 0, variance 2 (ln 2)^2 over n - 1 = 1.
        let spots = Spots(Rc::new(vec![100, 200, 100]));
        let sigma = spots.realized_before(3 * HOUR).unwrap().to_f64().unwrap();
        let expected = (2.0 * 2.0_f64.ln().powi(2) * HOURS_PER_YEAR).sqrt();
        assert!((sigma - expected).abs() < 1e-5, "{sigma} {expected}");
        assert!(
            Spots(Rc::new(vec![100, 200]))
                .realized_before(HOUR)
                .is_err()
        );
    }

    #[test]
    fn test_realized_volatility_subtracts_the_mean_return() {
        let spots = Spots(Rc::new(vec![100, 200, 300]));
        let (a, b) = (2.0_f64.ln(), 1.5_f64.ln());
        let middle = f64::midpoint(a, b);
        let variance = (a - middle).powi(2) + (b - middle).powi(2);
        let sigma = spots.realized_before(3 * HOUR).unwrap().to_f64().unwrap();
        let expected = (variance * HOURS_PER_YEAR).sqrt();
        assert!((sigma - expected).abs() < 1e-5, "{sigma} {expected}");
    }

    #[test]
    fn test_a_surface_arm_takes_its_volatility_from_the_training_window() {
        let spots = Spots::simulate(5).unwrap();
        let realized = spots.realized_before(VALIDATION_START).unwrap();
        assert_eq!(
            surface_sigma(&spots, dec!(1.25)).unwrap(),
            realized * dec!(1.25)
        );
        let mut later: Vec<u64> = spots.0.to_vec();
        later[(VALIDATION_START / HOUR) as usize..]
            .iter_mut()
            .for_each(|s| *s *= 3);
        assert_eq!(
            surface_sigma(&Spots(Rc::new(later)), dec!(1.25)).unwrap(),
            realized * dec!(1.25)
        );
    }

    #[test]
    fn test_a_lognormal_step_by_hand_has_no_drift() {
        let step = 0.1;
        assert_eq!(next_level(100.0, step, 0.0), 100.0 * (-0.005_f64).exp());
        assert_eq!(next_level(100.0, step, 2.0), 100.0 * 0.195_f64.exp());
    }

    #[test]
    fn test_the_market_tick_reads_the_time_left_at_its_timestamp() {
        let at = |t| market_tick(10_000, ATM, World::TheModel, t).unwrap();
        assert!(at(50 * DAY).ask_price < at(0).ask_price);
        assert_eq!(at(50 * DAY).timestamp, 50 * DAY);
    }

    #[test]
    fn test_paths_reproduce_by_seed_and_differ_between_seeds() {
        let one = Spots::simulate(3).unwrap();
        assert_eq!(one.0, Spots::simulate(3).unwrap().0);
        assert_ne!(one.0, Spots::simulate(4).unwrap().0);
        assert_eq!(one.0.len(), (DAYS * 24) as usize);
        assert_eq!(one.at(0), Some(10_000));
    }

    #[test]
    fn test_interval_brackets_the_mean_and_resamples_paths() {
        let (m, low, high) = interval(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m, 2.5);
        assert!(low < m && m < high);
        assert!((1.0..=2.5).contains(&low) && (2.5..=4.0).contains(&high));
        assert_eq!(interval(&[2.0; 5]), (2.0, 2.0, 2.0));
        // Half zeros and half ones: the mean of 100 has a standard error of 0.05, so a 90%
        // interval runs about 0.082 either side.
        let values: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect();
        let (_, low, high) = interval(&values);
        assert!((low - 0.418).abs() <= 0.012, "{low}");
        assert!((high - 0.582).abs() <= 0.012, "{high}");
    }

    #[test]
    fn test_a_study_line_prints_each_figure_under_its_header() {
        let row = Measure {
            net: 10.0,
            capture: 2.0,
            inventory: 8.0,
            drawdown: 3.0,
            fills: 4.0,
            mean_abs: 5.0,
            peak: 6.0,
        };
        // The peak is the largest of the paths, and the others are means.
        let lower = Measure { peak: 1.0, ..row };
        let line = holdout_line(World::TheModel, ATM, FillRule::Limit, "arm", &[lower, row]);
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        assert_eq!(cells[1], "TheModel");
        assert_eq!(cells[2], "call 10000");
        assert_eq!(cells[3], "limit-price");
        assert_eq!(cells[4], "arm");
        assert_eq!(cells[5], "10.0 [10.0, 10.0]");
        assert_eq!(cells[6], "2.0");
        assert_eq!(cells[7], "8.0");
        assert_eq!(cells[8], "3.0");
        assert_eq!(cells[9], "4.0");
        assert_eq!(cells[10], "5.00");
        assert_eq!(cells[11], "6");
    }

    fn report(world: World, contract: Contract, seed: u64) -> HoldoutReport {
        run_path(world, contract, seed, FillRule::Touch, &provenance())
            .unwrap()
            .0
    }

    fn holdout_measure(report: &HoldoutReport, name: &str) -> Measure {
        measure_of(
            report
                .holdout
                .iter()
                .find(|row| row.candidate == name)
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_the_fill_rules_bracket_what_the_wrong_quote_costs() {
        let run = |rule| {
            run_path(World::TheModel, ATM, 1, rule, &provenance())
                .unwrap()
                .0
        };
        let (touch, limit) = (run(FillRule::Touch), run(FillRule::Limit));
        assert_eq!(touch.manifest.reference_scenario, "touch");
        assert_eq!(limit.manifest.reference_scenario, "limit-price");
        assert!(
            holdout_measure(&limit, PREVIOUS).net < holdout_measure(&touch, PREVIOUS).net,
            "{:?} {:?}",
            holdout_measure(&limit, PREVIOUS),
            holdout_measure(&touch, PREVIOUS)
        );
        // Validation under one scenario does not depend on which is the reference.
        let row = |report: &HoldoutReport| {
            report
                .validation
                .iter()
                .find(|row| row.candidate == PREVIOUS && row.scenario == "touch")
                .map(measure_of)
                .unwrap()
                .unwrap()
        };
        assert_eq!(row(&touch), row(&limit));
    }

    #[test]
    fn test_no_tick_of_a_path_goes_unquoted_in_either_world() {
        for world in WORLDS {
            for contract in CONTRACTS {
                let (_, unquoted) =
                    run_path(world, contract, 1, FillRule::Touch, &provenance()).unwrap();
                assert_eq!(unquoted, 0, "{world:?} {contract:?}");
            }
        }
    }

    #[test]
    fn test_every_arm_reaches_the_holdout_and_nothing_fails() {
        let report = report(World::TheModel, ATM, 1);
        assert!(report.not_selected.is_empty(), "{:?}", report.not_selected);
        let mut names: Vec<&str> = report
            .holdout
            .iter()
            .map(|row| row.candidate.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "fixed-spread",
                "no-trade",
                "previous-formula",
                "surface-x0.75",
                "surface-x1",
                "surface-x1.25"
            ]
        );
        for row in report.holdout.iter().chain(&report.validation) {
            measure_of(row).unwrap_or_else(|reason| panic!("{}: {reason}", row.candidate));
        }
        // Every arm runs under every scenario on validation, and the surface arms do trade.
        assert_eq!(report.validation.len(), 6 * 5);
        for multiple in VOL_MULTIPLES {
            let name = surface_name(multiple);
            assert!(holdout_measure(&report, &name).fills > 0.0, "{name}");
        }
        assert_eq!(report.manifest.reference_scenario, "touch");
    }

    #[test]
    fn test_the_arms_differ_only_by_their_value_and_the_limit_binds_them_all() {
        let report = report(World::TheModel, ATM, 1);
        let row = |name: &str| {
            measure_of(
                report
                    .holdout
                    .iter()
                    .find(|row| row.candidate == name)
                    .unwrap(),
            )
            .unwrap()
        };
        // The formula is far under this market, so it sells until the limit stops it.
        let previous = row(PREVIOUS);
        assert_eq!(previous.peak, 20.0, "{previous:?}");
        for name in arm_names() {
            assert!(row(&name).peak <= 20.0, "{name}");
        }
        assert_eq!(row("no-trade").fills, 0.0);
        assert_ne!(row(SURFACE), row("surface-x1.25"));
    }

    #[test]
    fn test_a_run_reproduces_and_a_path_changes_it() {
        let one = report(World::TheModel, ATM, 2);
        assert_eq!(one, report(World::TheModel, ATM, 2));
        assert_ne!(
            one.manifest.data_hashes,
            report(World::TheModel, ATM, 3).manifest.data_hashes
        );
        assert_ne!(
            one.manifest.data_hashes,
            report(World::PremiumAndSkew, ATM, 2).manifest.data_hashes
        );
    }

    #[test]
    fn test_the_manifest_records_the_rules_every_arm_ran_under() {
        let manifest = report(World::TheModel, ATM, 4).manifest;
        assert_eq!(manifest.seed, 4);
        assert!(
            manifest.execution.contains("fee_rate: 0.0001"),
            "{}",
            manifest.execution
        );
        let policy = manifest.pretrade_policy.expect("a policy");
        assert!(policy.contains("max position 20"), "{policy}");
        let scenario = |name: &str| manifest.scenarios.get(name).unwrap();
        assert!(scenario("arrival").fill_model.starts_with("ArrivalFill"));
        assert!(scenario("touch").fill_model.starts_with("TouchFill"));
        assert!(
            scenario("limit-price")
                .fill_model
                .starts_with("ImmediateFill")
        );
        assert!(scenario("half-fill-chance").fill_model.contains("0.5"));
        assert_eq!(scenario("limit-price").fee_rate, None);
        assert_eq!(scenario(DOUBLE_FEE).fee_rate, Some(dec!(0.0002)));
        assert_eq!(
            scenario(DOUBLE_FEE).fill_model,
            scenario(&manifest.reference_scenario).fill_model
        );
    }

    #[test]
    fn test_twice_the_fee_runs_under_the_fill_rule_of_each_reference() {
        for rule in FILL_RULES {
            let (report, _) = run_path(World::TheModel, ATM, 1, rule, &provenance()).unwrap();
            let scenarios = &report.manifest.scenarios;
            assert_eq!(report.manifest.reference_scenario, rule.name());
            assert_eq!(
                scenarios[DOUBLE_FEE].fill_model,
                scenarios[rule.name()].fill_model,
                "{rule:?}"
            );
        }
    }

    #[test]
    fn test_twice_the_fee_on_validation_charges_twice_on_the_same_fills() {
        let report = report(World::TheModel, ATM, 1);
        let fees = |scenario: &str| {
            let row = report
                .validation
                .iter()
                .find(|row| row.candidate == PREVIOUS && row.scenario == scenario)
                .unwrap();
            let RowOutcome::Completed(metrics) = &row.outcome else {
                panic!("{:?}", row.outcome);
            };
            (
                metrics.metrics.attribution.quote_fees,
                metrics.metrics.fills,
            )
        };
        let (reference, double) = (fees(&report.manifest.reference_scenario), fees(DOUBLE_FEE));
        assert!(reference.1 > 0);
        assert_eq!(double.1, reference.1);
        assert_eq!(double.0, reference.0 * dec!(2));
    }

    #[test]
    fn test_a_measure_reads_each_figure_of_its_row() {
        let report = report(World::TheModel, ATM, 1);
        let row = report
            .holdout
            .iter()
            .find(|row| row.candidate == PREVIOUS)
            .unwrap();
        let RowOutcome::Completed(metrics) = &row.outcome else {
            panic!("{:?}", row.outcome);
        };
        let m = &metrics.metrics;
        let found = measure_of(row).unwrap();
        assert_eq!(found.net, m.attribution.net_pnl.to_f64().unwrap());
        assert_eq!(
            found.capture,
            m.attribution.spread_capture.to_f64().unwrap()
        );
        assert_eq!(
            found.inventory,
            m.attribution.inventory_pnl.to_f64().unwrap()
        );
        assert_eq!(found.drawdown, m.max_drawdown.to_f64().unwrap());
        assert_eq!(found.fills, m.fills as f64);
        assert_eq!(
            found.mean_abs,
            m.inventory.mean_abs.unwrap().to_f64().unwrap()
        );
        assert_eq!(found.peak, m.inventory.max_abs.to_f64().unwrap());
        assert_ne!(found.capture, found.inventory);
    }

    #[test]
    fn test_a_failed_row_is_kept_with_its_reason() {
        let mut tally = Tally::default();
        tally.holdout_row(3, "arm", "touch", Err("broke".to_string()));
        tally.validation_row(4, "arm", "touch", Err("too".to_string()));
        assert_eq!(
            tally.failed,
            [
                "path 3 arm touch holdout: broke",
                "path 4 arm touch validation: too"
            ]
        );
        assert!(tally.holdout.is_empty() && tally.validation.is_empty());
    }

    #[test]
    fn test_a_failed_row_does_not_shift_the_pairs() {
        let at = |net| Measure {
            net,
            capture: 0.0,
            inventory: 0.0,
            drawdown: 0.0,
            fills: 0.0,
            mean_abs: 0.0,
            peak: 0.0,
        };
        let mut tally = Tally::default();
        for (seed, a, b) in [(1, 10.0, 1.0), (2, 20.0, 2.0), (3, 30.0, 3.0)] {
            tally.holdout_row(seed, "a", "touch", Ok(at(a)));
            let measured = if seed == 2 {
                Err("failed".to_string())
            } else {
                Ok(at(b))
            };
            tally.holdout_row(seed, "b", "touch", measured);
        }
        assert_eq!(tally.differences("a", "b", |m| m.net), [9.0, 27.0]);
    }

    #[test]
    fn test_the_tally_pairs_arms_by_path_and_keeps_failures() {
        let mut tally = Tally::default();
        tally.add(1, &report(World::TheModel, ATM, 1), 2);
        tally.add(2, &report(World::TheModel, ATM, 2), 3);
        let first = report(World::TheModel, ATM, 1);
        let net = |report: &HoldoutReport, name: &str| {
            measure_of(
                report
                    .holdout
                    .iter()
                    .find(|row| row.candidate == name)
                    .unwrap(),
            )
            .unwrap()
            .net
        };
        let gaps = tally.differences(SURFACE, PREVIOUS, |m| m.net);
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0], net(&first, SURFACE) - net(&first, PREVIOUS));
        assert!(tally.failed.is_empty(), "{:?}", tally.failed);
        assert_eq!(tally.validation.len(), 5);
        assert_eq!(tally.data_hashes.len(), 6);
        assert_eq!(tally.unquoted, 5);
    }
}
