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
//!
//! The fill rule takes a quote as it is. A skew can set a bid above the mid or an ask under
//! it, and such a quote fills with probability 1 at a passive fee, where a venue would have
//! crossed it. See [`Episode::through_when_set`] and `test_fills_through_the_mid_when_set`.

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
/// Scenarios outside every range of the grid on the axes they move, fixed before they run.
/// The adversary never sees them, and they are not bounds of plausibility.
const OFF_GRID: [(&str, Scenario); 4] = [
    (
        "buyers 1.6, sellers 0.4",
        Scenario {
            bid_market: 0.4,
            ask_market: 1.6,
            branching: 1.0,
            impact: 1.0,
        },
    ),
    (
        "sellers 1.6, buyers 0.4",
        Scenario {
            bid_market: 1.6,
            ask_market: 0.4,
            branching: 1.0,
            impact: 1.0,
        },
    ),
    (
        "branching 1.5, impact 3",
        Scenario {
            bid_market: 1.0,
            ask_market: 1.0,
            branching: 1.5,
            impact: 3.0,
        },
    ),
    (
        "all at once",
        Scenario {
            bid_market: 0.4,
            ask_market: 1.6,
            branching: 1.5,
            impact: 3.0,
        },
    ),
];
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

/// Bid and ask prices given what the policy can see: time, mid and inventory.
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
    /// Fills at a quote through the mid when the order came, however it got there.
    stale_fills: u32,
    /// Those whose quote was already through the mid when it was set, which only a skew
    /// can do: a venue would have crossed it, not rested it at a passive fee.
    through_when_set: u32,
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
    let mut draws = Noise(seed.wrapping_mul(0x5851_F42D).wrapping_add(3));
    run_path(
        policy,
        BASE_IMPACT * scenario.impact,
        &flow(scenario, seed),
        || noise.normal(),
        || draws.unit(),
    )
}

/// An episode on given events, with `normal` driving the diffusion and `draw` the fills.
fn run_path(
    policy: Policy,
    impact: f64,
    events: &[(HawkesEvent, Side)],
    mut normal: impl FnMut() -> f64,
    mut draw: impl FnMut() -> f64,
) -> Episode {
    let mut book = Book::default();
    let mut out = Episode::default();
    let (mut mid, mut now, mut peak) = (100.0f64, 0.0f64, 0.0f64);
    let mut advance =
        |mid: &mut f64, now: &mut f64, to: f64, inventory: Decimal, out: &mut Episode| {
            let delta = to - *now;
            *mid += SIGMA * delta.sqrt() * normal();
            out.time_weighted_inventory += inventory.abs().to_f64().unwrap() * delta;
            *now = to;
        };
    for &(event, side) in events {
        // The quote rests from the last event, before the mid moves on to this one.
        let (bid, ask) = quotes(policy, mid, book.inventory, now);
        let mid_when_set = mid;
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
        let drawn = draw();
        if open && drawn < fill_probability(distance) {
            book.fill(side, at);
            out.stale_fills += u32::from(distance < 0.0);
            out.through_when_set += u32::from(match side {
                Side::Bid => at.to_f64().unwrap() > mid_when_set,
                Side::Ask => at.to_f64().unwrap() < mid_when_set,
            });
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

/// A larger impact costs a position more on average.
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
fn test_off_grid_scenarios_are_stable_and_outside_the_grid() {
    let (low, high) = (GRID_MARKET[0], GRID_MARKET[2]);
    for (_, scenario) in OFF_GRID {
        assert!(!grid().contains(&scenario));
        let off = |market: f64| market < low || market > high;
        let outside = off(scenario.bid_market)
            || off(scenario.ask_market)
            || scenario.branching > GRID_BRANCHING[2]
            || scenario.impact > GRID_IMPACT[2];
        assert!(outside, "{scenario:?}");
        assert!(queue_params(scenario.bid_market, scenario.branching).branching_ratio() < 1.0);
        assert!(queue_params(scenario.ask_market, scenario.branching).branching_ratio() < 1.0);
        assert!(run_episode(Policy::Fixed, scenario, 1).max_abs_inventory <= inventory_limit());
    }
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

fn market(time_seconds: f64, side: Side) -> (HawkesEvent, Side) {
    (
        HawkesEvent {
            time_seconds,
            kind: QrEventKind::Market,
        },
        side,
    )
}

/// An episode with no diffusion, whose fills follow `fills` in order.
fn scripted(
    policy: Policy,
    impact: f64,
    events: &[(HawkesEvent, Side)],
    fills: &[bool],
) -> Episode {
    let mut next = fills.iter();
    run_path(
        policy,
        impact,
        events,
        || 0.0,
        || if *next.next().unwrap() { 0.0 } else { 1.0 },
    )
}

fn close(found: f64, expected: f64) {
    assert!((found - expected).abs() < 1e-9, "{found} {expected}");
}

#[test]
fn test_an_episode_by_hand_with_a_position_at_the_end() {
    let events = [market(100.0, Side::Bid), market(200.0, Side::Bid)];
    let e = scripted(Policy::Fixed, 0.01, &events, &[true, true]);
    // Buys at 99.95 and 99.94, fee 1 bp: cash -199.909989. Mid 100, 99.99, then 99.98.
    assert_eq!(e.fills, 2);
    assert_eq!(e.terminal_inventory, dec!(2));
    assert_eq!(e.max_abs_inventory, dec!(2));
    // Equity 0.030005, then 0.050011, less 2 * 2 * 0.01 / 2 at the end.
    close(e.pnl, 0.030011);
    // The fall from the peak comes only from the charge at the end.
    close(e.max_drawdown, 0.02);
    // Held 0, 1, 2 for 100 s each, over 300 s.
    close(e.time_weighted_inventory, 1.0);
}

#[test]
fn test_drawdown_is_sampled_at_each_market_order() {
    let events = [
        market(10.0, Side::Bid),
        market(20.0, Side::Bid),
        market(30.0, Side::Ask),
        market(40.0, Side::Ask),
    ];
    let e = scripted(Policy::Fixed, 0.01, &events, &[true, false, false, false]);
    assert_eq!(e.fills, 1);
    // Equity 0.030005, 0.020005, 0.030005, 0.040005, then 0.035005 after the charge. The
    // deepest fall is the one in the loop, from an order that missed.
    close(e.max_drawdown, 0.01);
    close(e.pnl, 0.035005);
}

#[test]
fn test_a_short_side_order_moves_the_books_and_the_mid_the_other_way() {
    let events = [market(150.0, Side::Ask)];
    let e = scripted(Policy::Fixed, 0.01, &events, &[true]);
    // Sold at 100.05, fee 0.010005, bought back marked at 100.01, charge 0.005.
    assert_eq!(e.terminal_inventory, dec!(-1));
    close(e.pnl, 100.05 - 0.010005 - 100.01 - 0.005);
    close(e.time_weighted_inventory, 0.5);
}

#[test]
fn test_a_side_at_the_limit_takes_no_fill_but_its_order_still_moves_the_mid() {
    let impact = 0.01;
    let events: Vec<_> = (1..=12).map(|i| market(f64::from(i), Side::Bid)).collect();
    let e = scripted(Policy::Fixed, impact, &events, &[true; 12]);
    assert_eq!(e.fills, 10);
    assert_eq!(e.terminal_inventory, inventory_limit());
    // Ten buys, each at the half spread under a mid that falls by the impact of each order.
    let mut book = Book::default();
    for k in 0..10 {
        book.fill(
            Side::Bid,
            price(100.0 - impact * f64::from(k) - FIXED_HALF_SPREAD),
        );
    }
    // All twelve orders moved it, the two at the limit too.
    let expected = book.equity(100.0 - 12.0 * impact) - liquidation_cost(dec!(10), impact);
    close(e.pnl, expected);
}

#[test]
fn test_avellaneda_stoikov_in_an_episode_sees_its_inventory_and_the_time_left() {
    let events = [market(100.0, Side::Bid), market(200.0, Side::Bid)];
    let impact = 0.01;
    let e = scripted(Policy::AvellanedaStoikov, impact, &events, &[true, true]);
    let (first, _) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(0), 0.0);
    // The second quote is set after the first fill, at the mid the order left, 100 s in.
    let (second, _) = quotes(Policy::AvellanedaStoikov, 100.0 - impact, dec!(1), 100.0);
    let mut book = Book::default();
    book.fill(Side::Bid, first);
    book.fill(Side::Bid, second);
    let expected = book.equity(100.0 - 2.0 * impact) - liquidation_cost(dec!(2), impact);
    close(e.pnl, expected);
    // Both inputs change the quote, so an episode that ignored one would differ.
    assert_ne!(
        second,
        quotes(Policy::AvellanedaStoikov, 100.0 - impact, dec!(0), 100.0).0
    );
    assert_ne!(
        second,
        quotes(Policy::AvellanedaStoikov, 100.0 - impact, dec!(1), 0.0).0
    );
}

#[test]
fn test_a_fill_through_the_mid_when_set_is_told_from_a_quote_left_behind() {
    // Long one unit with 299 s left, the skew puts the ask under the mid; short, the bid over.
    let (_, ask) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(1), 1.0);
    let (bid, _) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(-1), 1.0);
    assert!(ask < dec!(100), "{ask}");
    assert!(bid > dec!(100), "{bid}");
    for (open, close) in [(Side::Bid, Side::Ask), (Side::Ask, Side::Bid)] {
        let events = [market(1.0, open), market(2.0, close)];
        let e = scripted(Policy::AvellanedaStoikov, 0.0, &events, &[true, true]);
        assert_eq!((e.fills, e.through_when_set, e.stale_fills), (2, 1, 1));
    }
    // A quote the mid moved through after it was set is stale and was not through then.
    let left_behind = [market(1.0, Side::Bid), market(2.0, Side::Bid)];
    let mut steps = [0.0, -1.0, 0.0].into_iter();
    let mut fills = [0.0, 0.0].into_iter();
    let e = run_path(
        Policy::Fixed,
        0.0,
        &left_behind,
        || steps.next().unwrap(),
        || fills.next().unwrap(),
    );
    assert_eq!(e.through_when_set, 0);
    assert_eq!(e.stale_fills, 0);
    let mut steps = [0.0, -100.0, 0.0].into_iter();
    let mut fills = [0.0, 0.0].into_iter();
    let e = run_path(
        Policy::Fixed,
        0.0,
        &left_behind,
        || steps.next().unwrap(),
        || fills.next().unwrap(),
    );
    assert_eq!(e.through_when_set, 0);
    assert_eq!(e.stale_fills, 1);
}

#[test]
fn test_branching_and_market_multipliers_reach_the_flow() {
    let count = |scenario: Scenario| {
        (1..30)
            .map(|seed| {
                flow(scenario, seed)
                    .iter()
                    .filter(|(e, _)| e.kind == QrEventKind::Market)
                    .count()
            })
            .sum::<usize>()
    };
    let more_branching = Scenario {
        branching: 1.2,
        ..BASE
    };
    let more_market = Scenario {
        bid_market: 1.3,
        ask_market: 1.3,
        ..BASE
    };
    assert_ne!(flow(BASE, 3), flow(more_branching, 3));
    assert!(count(more_branching) > count(BASE));
    assert!(count(more_market) > count(BASE));
}

#[test]
fn test_the_two_queues_do_not_share_a_seed() {
    let events = flow(BASE, 3);
    let times = |side| {
        events
            .iter()
            .filter(|(_, s)| *s == side)
            .map(|(e, _)| e.time_seconds)
            .collect::<Vec<_>>()
    };
    assert!(!times(Side::Bid).is_empty());
    assert_ne!(times(Side::Bid), times(Side::Ask));
}

#[test]
fn test_selection_and_holdout_seeds_do_not_meet_as_flow_doubles_them() {
    // `flow` uses 2 s and 2 s + 1.
    const { assert!((SELECTION_SEEDS.end - 1) * 2 + 1 < HOLDOUT_SEEDS.start * 2) };
}

#[test]
fn test_the_table_selects_on_the_selection_seeds_only() {
    let seen = std::cell::RefCell::new(Vec::new());
    let arms = table_arms(Policy::Fixed, |policy, seeds| {
        seen.borrow_mut().push((policy, seeds));
        FIXED_STRESS
    });
    assert_eq!(*seen.borrow(), [(Policy::Fixed, SELECTION_SEEDS)]);
    assert!(arms.contains(&("adversary", FIXED_STRESS)));
    assert!(arms.contains(&("base", BASE)));
    assert!(OFF_GRID.iter().all(|arm| arms.contains(arm)));
}

#[test]
fn test_interval_resamples_and_keeps_the_central_95_percent() {
    // Half zeros and half ones: the mean of 100 has a standard error of 0.05.
    let values: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect();
    let (low, high) = interval(&values, mean);
    assert!((low - 0.40).abs() <= 0.011, "{low}");
    assert!((high - 0.60).abs() <= 0.011, "{high}");
    assert_eq!(interval(&[2.0; 10], mean), (2.0, 2.0));
}

#[test]
fn test_a_table_row_prints_each_figure_under_its_header() {
    let episode = |pnl: f64, fills, inventory: f64, drawdown: f64, peak: Decimal| Episode {
        pnl,
        fills,
        time_weighted_inventory: inventory,
        max_drawdown: drawdown,
        max_abs_inventory: peak,
        ..Episode::default()
    };
    // Twenty episodes, so the worst 5% is the one at -40.
    let mut episodes = vec![episode(2.0, 3, 5.0, 7.0, dec!(4)); 19];
    episodes.push(episode(-40.0, 3, 5.0, 7.0, inventory_limit()));
    let row = table_row(Policy::Fixed, "base", BASE, &episodes);
    let cells: Vec<&str> = row.split('|').map(str::trim).collect();
    assert_eq!(cells[1], "Fixed");
    assert_eq!(cells[2], "base");
    assert_eq!(cells[3], "1, 1, 1, 1");
    assert!(cells[4].starts_with("-0.100 "), "{}", cells[4]);
    assert!(cells[5].starts_with("40.000 "), "{}", cells[5]);
    assert!(cells[6].starts_with("7.000 "), "{}", cells[6]);
    assert!(cells[7].starts_with("5.000 "), "{}", cells[7]);
    assert_eq!(cells[8], "10");
    assert_eq!(cells[9], "1");
    assert!(cells[10].starts_with("3.000 "), "{}", cells[10]);
}

/// How many fills in the holdout were at a quote through the mid when it was set: the limit
/// stated with the table. `--ignored --nocapture`, in release.
#[test]
#[ignore = "runs the adversary's selection"]
fn test_fills_through_the_mid_when_set() {
    for policy in [Policy::Fixed, Policy::AvellanedaStoikov] {
        for (arm, scenario) in table_arms(policy, adversary) {
            let (mut through, mut fills) = (0, 0);
            for seed in HOLDOUT_SEEDS {
                let e = run_episode(policy, scenario, seed);
                through += e.through_when_set;
                fills += e.fills;
            }
            println!("{policy:?} {arm}: {through} of {fills}");
        }
    }
}

/// The arms of the table, the adversary picked on the selection seeds alone.
fn table_arms(
    policy: Policy,
    pick: impl Fn(Policy, Range<u64>) -> Scenario,
) -> Vec<(&'static str, Scenario)> {
    let mut arms = vec![
        ("base", BASE),
        ("fixed stress", FIXED_STRESS),
        ("adversary", pick(policy, SELECTION_SEEDS)),
    ];
    arms.extend(OFF_GRID);
    arms
}

/// One row of the table.
fn table_row(policy: Policy, arm: &str, scenario: Scenario, episodes: &[Episode]) -> String {
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
    format!(
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
    )
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
        for (arm, scenario) in table_arms(policy, adversary) {
            let episodes: Vec<Episode> = HOLDOUT_SEEDS
                .map(|seed| run_episode(policy, scenario, seed))
                .collect();
            println!("{}", table_row(policy, arm, scenario, &episodes));
        }
    }
}
