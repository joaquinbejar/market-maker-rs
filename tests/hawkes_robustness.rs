//! A synthetic robustness study of classical policies under bounded perturbations of
//! Hawkes flow and price impact (#108).
//!
//! Everything here is a simulation with the model's own parameters: it says how these
//! policies behave in this model under these bounds, and nothing about a market. No
//! learned policy exists yet, so only classical ones are compared, and no margin
//! shortfall is measured, as this study has no margin account.
//!
//! The protocol is fixed in the constants below, before the holdout seeds run:
//! the adversary picks one scenario per policy from [`GRID_MARKET`], [`GRID_BRANCHING`]
//! and [`GRID_IMPACT`], maximising the expected shortfall of the PnL on [`SELECTION_SEEDS`]
//! only. It sets parameters before an episode and sees nothing of its path, and it changes
//! neither matching, accounting nor the limits. The fixed stress is [`FIXED_STRESS`].
//! Every arm is reported on [`HOLDOUT_SEEDS`].
//!
//! A policy requotes at each event, from the mid it could see then, so a quote can go
//! stale before the next order, and that order fills at the quote. What is held at the end
//! is charged half the impact of selling it all. Drawdown is sampled at market orders and at
//! the end only.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use std::ops::Range;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::hawkes::{HawkesEvent, HawkesParameters, HawkesSimulator};
use market_maker_rs::backtest::queue_reactive::QrEventKind;
use market_maker_rs::dec;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const GRID_MARKET: [f64; 3] = [0.7, 1.0, 1.3];
const GRID_BRANCHING: [f64; 3] = [0.8, 1.0, 1.2];
const GRID_IMPACT: [f64; 3] = [1.0, 1.5, 2.0];
const FIXED_STRESS: Scenario = Scenario {
    bid_market: 1.3,
    ask_market: 1.3,
    branching: 1.2,
    impact: 2.0,
};
const BASE: Scenario = Scenario {
    bid_market: 1.0,
    ask_market: 1.0,
    branching: 1.0,
    impact: 1.0,
};
const SELECTION_SEEDS: Range<u64> = 1..401;
const HOLDOUT_SEEDS: Range<u64> = 1_001..1_401;
const BOOTSTRAP_SEED: u64 = 77;
const BOOTSTRAP_DRAWS: usize = 2_000;
const TAIL: f64 = 0.05;

const HORIZON_SECONDS: f64 = 300.0;
const CAP: usize = 20;
const DECAY: f64 = 2.0;
const BASE_MARKET: f64 = 0.3;
const BASE_IMPACT: f64 = 0.005;
const SIGMA: f64 = 0.02;
const FILL_DECAY: f64 = 20.0;
const FIXED_HALF_SPREAD: f64 = 0.05;
const SECONDS_PER_YEAR: f64 = 31_536_000.0;

fn fee_rate() -> Decimal {
    dec!(0.0001)
}

fn inventory_limit() -> Decimal {
    dec!(10)
}

/// Bounded multipliers of the base model. Identifiable: each one scales a parameter the
/// estimators of `backtest::hawkes` recover, or the impact per market order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scenario {
    bid_market: f64,
    ask_market: f64,
    branching: f64,
    impact: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Fixed,
    AvellanedaStoikov,
}

fn queue_params(market: f64, branching: f64) -> HawkesParameters {
    let rates = (0..=CAP)
        .map(|n| {
            let size = n as f64;
            (
                if n == CAP { 0.0 } else { 1.5 },
                if n == 0 { 0.0 } else { 0.05 * size },
                if n == 0 { 0.0 } else { BASE_MARKET * market },
            )
        })
        .collect();
    let weights = [[0.10, 0.05, 0.15], [0.05, 0.10, 0.05], [0.02, 0.02, 0.30]]
        .map(|row| row.map(|alpha: f64| alpha * branching));
    HawkesParameters::new(rates, vec![DECAY], vec![weights]).unwrap()
}

/// A small generator for the path's noise and the fill draws, apart from the flow's.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Bid,
    Ask,
}

/// The two queues' events in time order.
fn flow(scenario: Scenario, seed: u64) -> Vec<(HawkesEvent, Side)> {
    let run = |market: f64, seed: u64, side: Side| {
        let params = queue_params(market, scenario.branching);
        let mut sim = HawkesSimulator::new(params, seed, 10).unwrap();
        let (segment, stop) = sim.run(HORIZON_SECONDS).unwrap();
        assert!(stop.is_none(), "{stop:?}");
        segment.events.into_iter().map(move |e| (e, side))
    };
    let mut events: Vec<_> = run(scenario.bid_market, seed.wrapping_mul(2), Side::Bid)
        .chain(run(
            scenario.ask_market,
            seed.wrapping_mul(2).wrapping_add(1),
            Side::Ask,
        ))
        .collect();
    events.sort_by(|a, b| a.0.time_seconds.total_cmp(&b.0.time_seconds));
    events
}

fn price(value: f64) -> Decimal {
    Decimal::from_f64(value).unwrap().round_dp(4)
}

/// Half spreads to the mid given what the policy can see: time, mid and inventory.
fn quotes(policy: Policy, mid: f64, inventory: Decimal, now: f64) -> (Decimal, Decimal) {
    let mid_price = price(mid);
    match policy {
        Policy::Fixed => (
            mid_price - price(FIXED_HALF_SPREAD),
            mid_price + price(FIXED_HALF_SPREAD),
        ),
        Policy::AvellanedaStoikov => {
            let left = ((HORIZON_SECONDS - now).max(1.0) * 1_000.0) as u64;
            calculate_optimal_quotes(
                mid_price,
                inventory,
                dec!(2),
                price(SIGMA * SECONDS_PER_YEAR.sqrt()),
                left,
                Decimal::from_f64(FILL_DECAY).unwrap(),
            )
            .unwrap()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Episode {
    pnl: f64,
    fills: u32,
    /// Fills at a quote the mid had moved through since it was set.
    stale_fills: u32,
    max_abs_inventory: Decimal,
    time_weighted_inventory: f64,
    max_drawdown: f64,
    terminal_inventory: Decimal,
}

/// The accounts of an episode, in `Decimal`.
#[derive(Debug, Clone, Copy, Default)]
struct Book {
    cash: Decimal,
    inventory: Decimal,
}

impl Book {
    /// One unit at `at`, with the fee on the notional. Both legs move in one step.
    fn fill(&mut self, side: Side, at: Decimal) {
        let fee = at * fee_rate();
        match side {
            Side::Bid => {
                self.inventory += Decimal::ONE;
                self.cash -= at + fee;
            }
            Side::Ask => {
                self.inventory -= Decimal::ONE;
                self.cash += at - fee;
            }
        }
    }

    fn equity(&self, mid: f64) -> f64 {
        (self.cash + self.inventory * price(mid)).to_f64().unwrap()
    }
}

/// The chance an order reaches a quote `distance` away from the mid. Through the mid, it does.
fn fill_probability(distance: f64) -> f64 {
    (-FILL_DECAY * distance.max(0.0)).exp()
}

/// A sell into the bid moves the mid down, and a buy from the ask up.
fn impact_move(side: Side, impact: f64) -> f64 {
    match side {
        Side::Bid => -impact,
        Side::Ask => impact,
    }
}

/// Selling `inventory` one order at a time moves the mid by `impact` each, so it pays half
/// of the total move on the whole.
fn liquidation_cost(inventory: Decimal, impact: f64) -> f64 {
    let held = inventory.abs().to_f64().unwrap();
    held * held * impact / 2.0
}

fn run_episode(policy: Policy, scenario: Scenario, seed: u64) -> Episode {
    let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9).wrapping_add(1));
    let mut book = Book::default();
    let mut out = Episode::default();
    let (mut mid, mut now, mut peak) = (100.0f64, 0.0f64, 0.0f64);
    let impact = BASE_IMPACT * scenario.impact;
    let mut advance =
        |mid: &mut f64, now: &mut f64, to: f64, inventory: Decimal, out: &mut Episode| {
            let delta = to - *now;
            *mid += SIGMA * delta.sqrt() * noise.normal();
            out.time_weighted_inventory += inventory.abs().to_f64().unwrap() * delta;
            *now = to;
        };
    let mut draws = Noise(seed.wrapping_mul(0x5851_F42D).wrapping_add(3));
    for (event, side) in flow(scenario, seed) {
        // The quote rests from the last event, before the mid moves on to this one.
        let (bid, ask) = quotes(policy, mid, book.inventory, now);
        advance(
            &mut mid,
            &mut now,
            event.time_seconds,
            book.inventory,
            &mut out,
        );
        if event.kind != QrEventKind::Market {
            continue;
        }
        // A side at the limit is not quoted, and the draw is taken either way, so arms of a
        // seed stay paired.
        let (at, distance, open) = match side {
            Side::Bid => (
                bid,
                mid - bid.to_f64().unwrap(),
                book.inventory < inventory_limit(),
            ),
            Side::Ask => (
                ask,
                ask.to_f64().unwrap() - mid,
                book.inventory > -inventory_limit(),
            ),
        };
        let draw = draws.unit();
        if open && draw < fill_probability(distance) {
            book.fill(side, at);
            out.stale_fills += u32::from(distance < 0.0);
            out.fills += 1;
            out.max_abs_inventory = out.max_abs_inventory.max(book.inventory.abs());
        }
        mid += impact_move(side, impact);
        let equity = book.equity(mid);
        peak = peak.max(equity);
        out.max_drawdown = out.max_drawdown.max(peak - equity);
    }
    advance(
        &mut mid,
        &mut now,
        HORIZON_SECONDS,
        book.inventory,
        &mut out,
    );
    let equity = book.equity(mid) - liquidation_cost(book.inventory, impact);
    peak = peak.max(equity);
    out.max_drawdown = out.max_drawdown.max(peak - equity);
    out.pnl = equity;
    out.terminal_inventory = book.inventory;
    out.time_weighted_inventory /= HORIZON_SECONDS;
    out
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Expected loss in the worst `TAIL` of the outcomes: positive is a loss.
fn cvar(pnls: &[f64]) -> f64 {
    let mut sorted = pnls.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = ((sorted.len() as f64 * TAIL).ceil() as usize).max(1);
    -mean(&sorted[..count])
}

/// A 95% interval by resampling whole episodes, which are independent, so each is its own
/// block.
fn interval(values: &[f64], statistic: fn(&[f64]) -> f64) -> (f64, f64) {
    let mut noise = Noise(BOOTSTRAP_SEED);
    let mut draws: Vec<f64> = (0..BOOTSTRAP_DRAWS)
        .map(|_| {
            let sample: Vec<f64> = values
                .iter()
                .map(|_| values[(noise.unit() * values.len() as f64) as usize])
                .collect();
            statistic(&sample)
        })
        .collect();
    draws.sort_by(f64::total_cmp);
    (
        draws[BOOTSTRAP_DRAWS / 40],
        draws[BOOTSTRAP_DRAWS - 1 - BOOTSTRAP_DRAWS / 40],
    )
}

fn grid() -> Vec<Scenario> {
    let mut out = Vec::new();
    for bid_market in GRID_MARKET {
        for ask_market in GRID_MARKET {
            for branching in GRID_BRANCHING {
                for impact in GRID_IMPACT {
                    out.push(Scenario {
                        bid_market,
                        ask_market,
                        branching,
                        impact,
                    });
                }
            }
        }
    }
    out
}

/// The scenario with the largest expected shortfall on `seeds`; the first on a tie.
fn adversary(policy: Policy, seeds: Range<u64>) -> Scenario {
    let mut worst = (BASE, f64::NEG_INFINITY);
    for scenario in grid() {
        let pnls: Vec<f64> = seeds
            .clone()
            .map(|seed| run_episode(policy, scenario, seed).pnl)
            .collect();
        let loss = cvar(&pnls);
        if loss > worst.1 {
            worst = (scenario, loss);
        }
    }
    worst.0
}

#[test]
fn test_a_fill_moves_cash_and_inventory_together_with_the_fee() {
    let mut book = Book::default();
    book.fill(Side::Bid, dec!(100));
    assert_eq!(book.inventory, dec!(1));
    assert_eq!(book.cash, dec!(-100.01));
    book.fill(Side::Ask, dec!(101));
    assert_eq!(book.inventory, dec!(0));
    assert_eq!(book.cash, dec!(0.9799));
    assert_eq!(book.equity(100.0), 0.9799);
}

#[test]
fn test_liquidation_charges_half_the_impact_of_the_position() {
    assert_eq!(liquidation_cost(dec!(-4), 0.5), 4.0);
    assert_eq!(liquidation_cost(dec!(0), 0.5), 0.0);
}

/// A larger impact costs a position more, and a quote left behind by the mid loses.
#[test]
fn test_stale_quotes_and_impact_cost_a_fixed_policy_on_average() {
    let pnl = |s: Scenario| {
        (1..200)
            .map(|seed| run_episode(Policy::Fixed, s, seed).pnl)
            .sum::<f64>()
            / 199.0
    };
    let high = Scenario {
        impact: 4.0,
        ..BASE
    };
    assert!(pnl(high) < pnl(BASE), "{} {}", pnl(high), pnl(BASE));
}

#[test]
fn test_impact_and_fill_probability_have_their_signs() {
    assert_eq!(impact_move(Side::Bid, 0.5), -0.5);
    assert_eq!(impact_move(Side::Ask, 0.5), 0.5);
    assert_eq!(fill_probability(-1.0), 1.0);
    assert_eq!(fill_probability(0.05), (-1.0f64).exp());
    assert!(queue_params(1.0, 1.2).branching_ratio() > queue_params(1.0, 1.0).branching_ratio());
}

/// The quote comes from before the mid moved, so it can be left behind.
#[test]
fn test_a_quote_can_go_stale_before_an_order() {
    let stale: u32 = (1..200)
        .map(|seed| run_episode(Policy::Fixed, BASE, seed).stale_fills)
        .sum();
    assert!(stale > 0);
}

#[test]
fn test_an_episode_is_a_function_of_its_seed() {
    let one = run_episode(Policy::Fixed, BASE, 5);
    assert_eq!(one, run_episode(Policy::Fixed, BASE, 5));
    assert_ne!(one, run_episode(Policy::Fixed, BASE, 6));
    assert!(one.fills > 0);
}

#[test]
fn test_the_whole_grid_is_a_stable_model() {
    for branching in GRID_BRANCHING {
        for market in GRID_MARKET {
            assert!(queue_params(market, branching).branching_ratio() < 1.0);
        }
    }
    assert_eq!(grid().len(), 81);
    assert!(grid().contains(&FIXED_STRESS));
    assert!(grid().contains(&BASE));
}

#[test]
fn test_the_inventory_limit_holds_in_every_corner() {
    let mut reached = false;
    for scenario in [
        FIXED_STRESS,
        Scenario {
            bid_market: 0.7,
            ..FIXED_STRESS
        },
        Scenario {
            ask_market: 0.7,
            ..FIXED_STRESS
        },
    ] {
        for policy in [Policy::Fixed, Policy::AvellanedaStoikov] {
            for seed in 1..20 {
                let episode = run_episode(policy, scenario, seed);
                assert!(episode.max_abs_inventory <= inventory_limit());
                reached = reached || episode.max_abs_inventory == inventory_limit();
            }
        }
    }
    assert!(reached, "the limit never bound, so this tested nothing");
}

#[test]
fn test_impact_changes_no_flow() {
    let high = Scenario {
        impact: 2.0,
        ..BASE
    };
    assert_eq!(flow(BASE, 3), flow(high, 3));
}

#[test]
fn test_more_market_orders_on_one_side_push_inventory_that_way() {
    let buyers = Scenario {
        bid_market: 0.7,
        ask_market: 1.3,
        ..BASE
    };
    let terminal: f64 = (1..60)
        .map(|seed| {
            run_episode(Policy::Fixed, buyers, seed)
                .terminal_inventory
                .to_f64()
                .unwrap()
        })
        .sum();
    assert!(terminal < 0.0, "{terminal}");
}

#[test]
fn test_cvar_is_the_mean_of_the_worst_tail() {
    let values: Vec<f64> = (1..=40).rev().map(f64::from).collect();
    // The worst 5% of 40 is 2 values: 1 and 2.
    assert_eq!(cvar(&values), -1.5);
    assert_eq!(cvar(&[3.0]), -3.0);
}

#[test]
fn test_avellaneda_stoikov_skews_against_inventory_and_fixed_does_not() {
    let (flat_bid, flat_ask) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(0), 0.0);
    let (long_bid, long_ask) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(5), 0.0);
    assert!(flat_bid < flat_ask);
    assert!(long_bid < flat_bid && long_ask < flat_ask);
    assert_eq!(
        quotes(Policy::Fixed, 100.0, dec!(0), 0.0),
        quotes(Policy::Fixed, 100.0, dec!(5), 0.0)
    );
}

#[test]
fn test_the_adversary_is_the_argmax_on_its_seeds() {
    let pick = adversary(Policy::Fixed, 1..4);
    assert_eq!(pick, adversary(Policy::Fixed, 1..4));
    // It is the argmax on those seeds.
    let shortfall = |s: Scenario| {
        let pnls: Vec<f64> = (1..4)
            .map(|seed| run_episode(Policy::Fixed, s, seed).pnl)
            .collect();
        cvar(&pnls)
    };
    for scenario in grid() {
        assert!(shortfall(scenario) <= shortfall(pick));
    }
}

/// The evidence table. `--ignored --nocapture`, in release.
#[test]
#[ignore = "the study itself: several minutes without --release"]
fn test_robustness_evidence_table() {
    println!(
        "SYNTHETIC. {} holdout episodes of {HORIZON_SECONDS} s; selection on {} other seeds; 95% intervals by resampling episodes.",
        HOLDOUT_SEEDS.count(),
        SELECTION_SEEDS.count()
    );
    println!(
        "| policy | arm | scenario (bid, ask, branching, impact) | mean pnl | cvar 5% | mean drawdown | mean abs inventory | peak abs inventory | at limit | fills |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for policy in [Policy::Fixed, Policy::AvellanedaStoikov] {
        let adversarial = adversary(policy, SELECTION_SEEDS);
        for (arm, scenario) in [
            ("base", BASE),
            ("fixed stress", FIXED_STRESS),
            ("adversary", adversarial),
        ] {
            let episodes: Vec<Episode> = HOLDOUT_SEEDS
                .map(|seed| run_episode(policy, scenario, seed))
                .collect();
            let pnls: Vec<f64> = episodes.iter().map(|e| e.pnl).collect();
            let pick = |f: fn(&Episode) -> f64| {
                let values: Vec<f64> = episodes.iter().map(f).collect();
                let (low, high) = interval(&values, mean);
                format!("{:.3} [{low:.3}, {high:.3}]", mean(&values))
            };
            let (tail_low, tail_high) = interval(&pnls, cvar);
            let peak = episodes.iter().map(|e| e.max_abs_inventory).max().unwrap();
            let at_limit = episodes
                .iter()
                .filter(|e| e.max_abs_inventory == inventory_limit())
                .count();
            println!(
                "| {policy:?} | {arm} | {}, {}, {}, {} | {} | {:.3} [{tail_low:.3}, {tail_high:.3}] | {} | {} | {peak} | {at_limit} | {} |",
                scenario.bid_market,
                scenario.ask_market,
                scenario.branching,
                scenario.impact,
                pick(|e| e.pnl),
                cvar(&pnls),
                pick(|e| e.max_drawdown),
                pick(|e| e.time_weighted_inventory),
                pick(|e| f64::from(e.fills)),
            );
        }
    }
}
