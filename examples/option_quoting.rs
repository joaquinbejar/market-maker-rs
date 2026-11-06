//! Compares joint option quotes under identical execution (#98).
//!
//! Everything is SYNTHETIC, and says nothing about a market. Three calls on one underlying are
//! quoted hourly for ten days, on a flat surface that never moves, so vega earns and costs nothing
//! here and the volatility factor only shapes the skew. The spot follows a seeded random walk, and
//! each quote fills with the probability its declared intensity gives at its distance from the
//! pricer's value. Every arm sees the same spot path and the same uniform draws, by step, contract
//! and side, so what differs between arms is only what their quotes do with those draws.
//!
//! Part one is the adjustment itself, at fixed positions: a concentrated book against a hedged one
//! of the same size. Part two runs the arms:
//!
//! - **correlated, no hedge** and **diagonal, no hedge**: the same limits, with and without the
//!   correlations between contracts.
//! - **correlated, hedged** and **diagonal, hedged**: the same, sending each proposed hedge at the
//!   spot with no cost, which flatters them.
//!
//! Net PnL is the change in cash, positions at the pricer's value and the hedge at the spot.
//! Capital use is `OptionCapital::committed` over the budget, where a short and a sale count at the
//! initial margin of `margin-rs`'s rule `short-option-im/1`, that crate's own rates and not a
//! venue's. No arm is selected: all are printed.
//! Nothing here is a claim about which is better on real flow.
//!
//! `cargo run --release --features options,margin,multi-underlying --example option_quoting`

use std::sync::Arc;

use margin_rs::ledger::{Execution, Ledger};
use margin_rs::{Asset, ExecutionId, InstrumentId, SpotInstrument, TransferId};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::multi_underlying::multi_asset::{
    ContractTerms, FactorRisk, OptionConstraintKind, OptionMarket, OptionQuoter,
    OptionQuoterConfig, QuotingObjective, RiskScenario, RiskView, UnderlyingTerms,
};
use market_maker_rs::options::FlatSurface;
use market_maker_rs::options::market_maker::GreeksLimits;
use market_maker_rs::options::pricing::{Dynamics, OptionInputs, OptionPricer, SurfacePricer};
use market_maker_rs::options::surface::shocks::{ShockedSurface, SurfaceShock};
use market_maker_rs::types::action::Action;
use optionstratlib::OptionStyle;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

type Failure = Box<dyn std::error::Error>;

const HOUR_MS: u64 = 3_600_000;
const STEPS: u64 = 240;
const SEEDS: u64 = 40;
const HOURS_PER_YEAR: f64 = 365.0 * 24.0;
/// Annual volatility of the walk, against the surface's 40%.
const WALK_VOLATILITY: f64 = 0.4;
const START_SPOT: f64 = 100.0;
const INTENSITY_SCALE: f64 = 0.5;
const INTENSITY_DECAY: f64 = 2.0;
const BUDGET: Decimal = dec!(400);
const DEPOSIT: Decimal = dec!(100000);
const STRIKES: [Decimal; 3] = [dec!(90), dec!(100), dec!(110)];

/// One fixed mix of a step, a stream and a seed into a uniform draw.
fn uniform(seed: u64, step: u64, stream: u64) -> f64 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(step.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(stream.wrapping_mul(0x94D0_49BB_1331_11EB));
    for _ in 0..3 {
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
    }
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn normal(seed: u64, step: u64) -> f64 {
    let (a, b) = (uniform(seed, step, 1_000), uniform(seed, step, 1_001));
    (-2.0 * a.max(1e-16).ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

fn decimal(value: f64) -> Result<Decimal, Failure> {
    Decimal::from_f64(value).ok_or_else(|| format!("{value} is not a decimal").into())
}

fn number(value: Decimal) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn spot_id() -> Result<InstrumentId, Failure> {
    Ok(InstrumentId::new("AAA-USD")?)
}

fn new_ledger() -> Result<Ledger, Failure> {
    let (base, usd) = (Asset::new("AAA")?, Asset::new("USD")?);
    let mut ledger = Ledger::default();
    ledger.register_asset(base.clone(), 8)?;
    ledger.register_asset(usd.clone(), 8)?;
    ledger.register_instrument(SpotInstrument::new(spot_id()?, base, usd.clone(), 4, 2)?)?;
    ledger.deposit(TransferId::new("deposit")?, usd, DEPOSIT)?;
    Ok(ledger)
}

fn pricer(sigma: Decimal) -> Result<Arc<dyn OptionPricer>, Failure> {
    let surface = FlatSurface::new(sigma)?;
    Ok(Arc::new(SurfacePricer::new(
        Arc::new(surface),
        Dynamics::StickyMoneyness,
    )?))
}

fn config(view: RiskView, expiry_ms: u64) -> Result<OptionQuoterConfig, Failure> {
    let surface = FlatSurface::new(dec!(0.4))?;
    let shock = SurfaceShock::new("vol up", 0.1, 0.0, 0.0, 1.0)?;
    let shocked = ShockedSurface::new(surface, shock);
    let scenario = |name: &str, factor: Decimal, pricer: Arc<dyn OptionPricer>| RiskScenario {
        name: name.to_string(),
        spot_factor: factor,
        pricer,
    };
    let up: Arc<dyn OptionPricer> = Arc::new(SurfacePricer::new(
        Arc::new(shocked),
        Dynamics::StickyMoneyness,
    )?);
    let contracts = STRIKES
        .iter()
        .map(|strike| {
            Ok(ContractTerms {
                name: format!("C{strike}"),
                strike: *strike,
                expiry_ms,
                style: OptionStyle::Call,
                price_scale: 3,
                quantity_scale: 2,
                intensity_scale: decimal(INTENSITY_SCALE)?,
                intensity_decay: decimal(INTENSITY_DECAY)?,
                trade_size: dec!(1),
                inventory_limit: dec!(20),
            })
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(OptionQuoterConfig {
        underlying: UnderlyingTerms {
            instrument: spot_id()?,
            intensity_scale: dec!(5),
            intensity_decay: dec!(2),
            trade_size: dec!(1),
            inventory_limit: dec!(100),
        },
        contracts,
        cash_asset: Asset::new("USD")?,
        pricer: pricer(dec!(0.4))?,
        factors: FactorRisk {
            spot_volatility: decimal(START_SPOT * WALK_VOLATILITY / HOURS_PER_YEAR.sqrt())?,
            volatility_of_volatility: dec!(0.002),
            correlation: dec!(-0.3),
        },
        risk_aversion: dec!(0.05),
        objective: QuotingObjective::RiskAdjustedExpectation,
        risk_view: view,
        capital_budget: BUDGET,
        scenarios: vec![
            scenario("spot +10%, vol +10 points", dec!(1.1), up.clone()),
            scenario("spot -10%, vol +10 points", dec!(0.9), up),
        ],
    })
}

fn limits() -> GreeksLimits {
    GreeksLimits {
        max_delta: dec!(15),
        max_gamma: dec!(3),
        max_vega: dec!(300),
        max_theta: dec!(-2),
        delta_hedge_threshold: dec!(1),
    }
}

/// What one arm did on one path, as means over its steps where it says so.
#[derive(Debug, Default, Clone, Copy)]
struct Outcome {
    pnl: f64,
    inventory: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    capital: f64,
    peak_capital: f64,
    fills: f64,
    withheld: f64,
    cut: f64,
}

impl Outcome {
    fn read(&self, field: usize) -> f64 {
        [
            self.pnl,
            self.inventory,
            self.delta,
            self.gamma,
            self.vega,
            self.capital,
            self.peak_capital,
            self.fills,
            self.withheld,
            self.cut,
        ][field]
    }
}

const FIELDS: [&str; 10] = [
    "net PnL",
    "mean |contract position|",
    "mean |delta|",
    "mean |gamma|",
    "mean |vega|",
    "mean capital use",
    "peak capital use",
    "fills",
    "sides withheld",
    "sides cut by the budget",
];

fn run(seed: u64, view: RiskView, hedging: bool) -> Result<Outcome, Failure> {
    let start_ms = 1_000_000;
    let expiry_ms = start_ms + 30 * 24 * HOUR_MS;
    let settings = config(view, expiry_ms)?;
    let contracts = settings.contracts.clone();
    let valuation = settings.pricer.clone();
    let mut quoter = OptionQuoter::new(settings)?.with_greek_limits(limits());
    let mut ledger = new_ledger()?;
    let mut outcome = Outcome::default();
    let mut spot = START_SPOT;
    let (usd, id) = (Asset::new("USD")?, spot_id()?);
    let step_vol = WALK_VOLATILITY / HOURS_PER_YEAR.sqrt();

    let value_of = |spot: Decimal, now_ms: u64| -> Result<Vec<_>, Failure> {
        let mut values = Vec::with_capacity(contracts.len());
        for terms in &contracts {
            let days = Decimal::from(terms.expiry_ms - now_ms) / dec!(86400000);
            values.push(valuation.value(&OptionInputs::european(
                spot,
                terms.strike,
                days / dec!(365),
                dec!(0),
                dec!(0),
                terms.style,
            ))?);
        }
        Ok(values)
    };

    for step in 0..STEPS {
        let now_ms = start_ms + step * HOUR_MS;
        let spot_decimal = decimal(spot)?.round_dp(4);
        let market = OptionMarket {
            spot: spot_decimal,
            rate: dec!(0),
            carry: dec!(0),
            now_ms,
        };
        for order in quoter.live_orders() {
            quoter.on_cancel(&mut ledger, &order.order_id)?;
        }
        let round = quoter.quote(&mut ledger, &market)?;
        for constraint in &round.constraints {
            match constraint.kind {
                OptionConstraintKind::CapitalBudget { allowed, .. } if !allowed.is_zero() => {
                    outcome.cut += 1.0;
                }
                OptionConstraintKind::ThroughValue { .. } => {}
                _ => outcome.withheld += 1.0,
            }
        }
        let use_now = number(round.capital.committed) / number(BUDGET);
        outcome.capital += use_now / STEPS as f64;
        outcome.peak_capital = outcome.peak_capital.max(use_now);

        let values = value_of(spot_decimal, now_ms)?;
        let positions = quoter.positions().to_vec();
        for held in &positions {
            outcome.inventory += number(held.abs()) / (STEPS as f64 * contracts.len() as f64);
        }
        let total = |read: fn(&market_maker_rs::options::pricing::OptionValuation) -> Decimal| {
            positions
                .iter()
                .zip(&values)
                .map(|(held, value)| number(*held) * number(read(value)))
                .sum::<f64>()
        };
        outcome.gamma += total(|value| value.gamma).abs() / STEPS as f64;
        outcome.vega += total(|value| value.vega).abs() / STEPS as f64;
        outcome.delta += number(round.net_delta.abs()) / STEPS as f64;

        if hedging && let Some(hedge) = &round.hedge {
            let side = if hedge.quantity.is_sign_positive() {
                Action::Buy
            } else {
                Action::Sell
            };
            ledger.book_execution_allowing_overdraft(
                &Execution::new(
                    ExecutionId::new(format!("hedge-{step}"))?,
                    id.clone(),
                    side,
                    spot_decimal,
                    hedge.quantity.abs(),
                ),
                None,
            )?;
        }

        // The same draw decides a side's fill in every arm, at this arm's distance.
        for quote in &round.quotes {
            let side = u64::from(quote.side == Action::Sell);
            let draw = uniform(seed, step, quote.contract as u64 * 2 + side);
            let value = values
                .get(quote.contract)
                .ok_or("a quote of an unknown contract")?;
            let distance = (number(quote.price) - number(value.price)).abs();
            let intensity = INTENSITY_SCALE * (-INTENSITY_DECAY * distance).exp();
            if draw < 1.0 - (-intensity).exp() {
                quoter.on_fill(&mut ledger, &quote.order_id, quote.quantity)?;
                outcome.fills += 1.0;
            }
        }
        spot *= (step_vol * normal(seed, step) - 0.5 * step_vol * step_vol).exp();
    }

    let last_ms = start_ms + STEPS * HOUR_MS;
    let last_spot = decimal(spot)?.round_dp(4);
    let values = value_of(last_spot, last_ms)?;
    let positions_value: f64 = quoter
        .positions()
        .iter()
        .zip(&values)
        .map(|(held, value)| number(*held) * number(value.price))
        .sum();
    let hedge_value = ledger
        .position(&id)
        .map_or(0.0, |held| number(held.quantity) * spot);
    let cash = ledger.balance(&usd).ok_or("no cash")?.total;
    outcome.pnl = number(cash) - number(DEPOSIT) + positions_value + hedge_value;
    Ok(outcome)
}

fn adjustment(position: [Decimal; 3], hedge: Decimal, view: RiskView) -> Result<Decimal, Failure> {
    let start_ms = 1_000_000;
    let settings = config(view, start_ms + 30 * 24 * HOUR_MS)?;
    let mut ledger = new_ledger()?;
    if !hedge.is_zero() {
        ledger.book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("start")?,
                spot_id()?,
                if hedge.is_sign_positive() {
                    Action::Buy
                } else {
                    Action::Sell
                },
                dec!(100),
                hedge.abs(),
            ),
            None,
        )?;
    }
    let mut quoter = OptionQuoter::new(settings)?.with_restored_state(position.to_vec(), 0)?;
    let round = quoter.quote(
        &mut ledger,
        &OptionMarket {
            spot: dec!(100),
            rate: dec!(0),
            carry: dec!(0),
            now_ms: start_ms,
        },
    )?;
    let price = |side| {
        round
            .quotes
            .iter()
            .find(|quote| quote.contract == 1 && quote.side == side)
            .map(|quote| quote.price)
            .ok_or("no quote at the money")
    };
    Ok((price(Action::Buy)? + price(Action::Sell)?) / dec!(2))
}

fn main() -> Result<(), Failure> {
    println!("SYNTHETIC: seeded spot, declared fills, a flat surface. Nothing here is real data.");
    println!(
        "{STEPS} hourly steps, {SEEDS} seeds, three calls at 90, 100 and 110 expiring in 30 days."
    );
    println!("Capital use: committed over the budget of {BUDGET}.\n");

    println!("Mid of the at the money quote, by book, under each view:");
    for (name, position, hedge) in [
        ("flat", [dec!(0), dec!(0), dec!(0)], dec!(0)),
        ("long 4 at the money", [dec!(0), dec!(4), dec!(0)], dec!(0)),
        (
            "the same, delta hedged",
            [dec!(0), dec!(4), dec!(0)],
            dec!(-2),
        ),
        (
            "the same, spread against 110",
            [dec!(0), dec!(4), dec!(-4)],
            dec!(0),
        ),
    ] {
        println!(
            "  {name:32} correlated {:>8}  diagonal {:>8}",
            adjustment(position, hedge, RiskView::Correlated)?,
            adjustment(position, hedge, RiskView::Diagonal)?,
        );
    }

    let arms = [
        ("correlated, no hedge", RiskView::Correlated, false),
        ("diagonal, no hedge", RiskView::Diagonal, false),
        ("correlated, hedged", RiskView::Correlated, true),
        ("diagonal, hedged", RiskView::Diagonal, true),
    ];
    let mut results = Vec::new();
    for (name, view, hedging) in arms {
        let mut outcomes = Vec::new();
        for seed in 1..=SEEDS {
            outcomes.push(run(seed, view, hedging)?);
        }
        results.push((name, outcomes));
    }
    println!("\nMean per path [lowest, highest]:");
    for (name, outcomes) in &results {
        println!("{name}");
        for (field, label) in FIELDS.iter().enumerate() {
            let read: Vec<f64> = outcomes.iter().map(|outcome| outcome.read(field)).collect();
            let mean = read.iter().sum::<f64>() / read.len() as f64;
            let low = read.iter().copied().fold(f64::INFINITY, f64::min);
            let high = read.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!("  {label:26} {mean:10.3} [{low:.3}, {high:.3}]");
        }
    }
    println!(
        "\nEvery side is scaled to what the budget leaves, so without a hedge capital use passes 1 \
         only when the positions held move past it between rounds: a long's value and a short's \
         margin both follow the spot. A hedge is sized by the delta and sent outside the round, \
         so it can pass the budget, as the hedged arms show, and the next rounds then have less \
         left."
    );
    Ok(())
}
