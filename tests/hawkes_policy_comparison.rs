//! The same classical policies on Poisson, queue-reactive and Hawkes flow (#91).
//!
//! SYNTHETIC: the data comes from a Hawkes model with made-up parameters, so the Hawkes
//! fit is correctly specified and its doing well is expected. It says nothing about a
//! market. Policy results are reported apart from flow fit, and no policy or model is
//! selected on the holdout: the decay is chosen on validation, and the policies are fixed.
//!
//! Each simulator is one queue per side, independent, as the module models one queue. The
//! fill rule, fees and inventory limit are the same in every arm. The gap of an arm is its
//! policy metric against the same metric on flow from the true model, on other seeds, which
//! here plays the data. What is held at the end is charged half the impact of selling it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use std::ops::Range;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::hawkes::{
    HawkesEvent, HawkesFitConfig, HawkesParameters, HawkesSegment, HawkesSimulator, fit_hawkes,
    fit_poisson, score_flow, select_decay,
};
use market_maker_rs::backtest::queue_reactive::QrEventKind;
use market_maker_rs::dec;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const CAP: usize = 20;
const HORIZON: f64 = 300.0;
const TRAIN: Range<u64> = 1..41;
const VALIDATION: Range<u64> = 41..61;
const DATA_HOLDOUT: Range<u64> = 1_001..1_401;
const SIM_SEEDS: Range<u64> = 2_001..2_401;
const DECAYS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
const IMPACT: f64 = 0.005;
const SIGMA: f64 = 0.02;
const FILL_DECAY: f64 = 20.0;
const HALF_SPREAD: f64 = 0.05;

fn truth() -> HawkesParameters {
    let rates = (0..=CAP)
        .map(|n| {
            (
                if n == CAP { 0.0 } else { 1.5 },
                if n == 0 { 0.0 } else { 0.05 * n as f64 },
                if n == 0 { 0.0 } else { 0.3 },
            )
        })
        .collect();
    let weights = [[0.10, 0.05, 0.15], [0.05, 0.10, 0.05], [0.02, 0.02, 0.30]];
    HawkesParameters::new(rates, vec![2.0], vec![weights]).unwrap()
}

fn segments(params: &HawkesParameters, seeds: Range<u64>) -> Vec<HawkesSegment> {
    seeds
        .map(|seed| {
            let mut sim = HawkesSimulator::new(params.clone(), seed, 10).unwrap();
            let (segment, stop) = sim.run(HORIZON).unwrap();
            assert!(stop.is_none(), "{stop:?}");
            segment
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Fixed,
    AvellanedaStoikov,
}

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

fn price(value: f64) -> Decimal {
    Decimal::from_f64(value).unwrap().round_dp(4)
}

fn quotes(policy: Policy, mid: f64, inventory: Decimal, now: f64) -> (f64, f64) {
    match policy {
        Policy::Fixed => (mid - HALF_SPREAD, mid + HALF_SPREAD),
        Policy::AvellanedaStoikov => {
            let left = ((HORIZON - now).max(1.0) * 1_000.0) as u64;
            let (bid, ask) = calculate_optimal_quotes(
                price(mid),
                inventory,
                dec!(2),
                price(SIGMA * 31_536_000.0f64.sqrt()),
                left,
                price(FILL_DECAY),
            )
            .unwrap();
            (bid.to_f64().unwrap(), ask.to_f64().unwrap())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Outcome {
    pnl: f64,
    fills: f64,
    abs_inventory: f64,
    max_drawdown: f64,
    peak_inventory: Decimal,
}

/// One unit at `at`, with the fee on the notional: both legs move in one step.
fn fill(cash: Decimal, inventory: Decimal, on_bid: bool, at: Decimal) -> (Decimal, Decimal) {
    let fee = at * dec!(0.0001);
    if on_bid {
        (cash - at - fee, inventory + Decimal::ONE)
    } else {
        (cash + at - fee, inventory - Decimal::ONE)
    }
}

/// Equity at every event, so that drawdown is sampled alike at each arm's events.
fn mark(cash: Decimal, inventory: Decimal, mid: f64, peak: &mut f64, drawdown: &mut f64) {
    let equity = (cash + inventory * price(mid)).to_f64().unwrap();
    *peak = peak.max(equity);
    *drawdown = drawdown.max(*peak - equity);
}

/// One episode over a bid queue's and an ask queue's flow. Only market orders trade.
fn episode(policy: Policy, bid: &HawkesSegment, ask: &HawkesSegment, seed: u64) -> Outcome {
    let mut events: Vec<(f64, bool, QrEventKind)> = bid
        .events
        .iter()
        .map(|e| (e.time_seconds, true, e.kind))
        .chain(ask.events.iter().map(|e| (e.time_seconds, false, e.kind)))
        .collect();
    events.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9).wrapping_add(1));
    let (mut mid, mut now) = (100.0f64, 0.0f64);
    let (mut cash, mut inventory) = (Decimal::ZERO, Decimal::ZERO);
    let (mut peak, mut drawdown, mut fills, mut held) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let limit = dec!(10);
    let mut peak_inventory = Decimal::ZERO;
    for (time, on_bid, kind) in events {
        let (quote_bid, quote_ask) = quotes(policy, mid, inventory, now);
        let delta = time - now;
        mid += SIGMA * delta.sqrt() * noise.normal();
        held += inventory.abs().to_f64().unwrap() * delta;
        now = time;
        if kind != QrEventKind::Market {
            mark(cash, inventory, mid, &mut peak, &mut drawdown);
            continue;
        }
        let draw = noise.unit();
        let distance = if on_bid {
            mid - quote_bid
        } else {
            quote_ask - mid
        };
        let open = if on_bid {
            inventory < limit
        } else {
            inventory > -limit
        };
        if open && draw < (-FILL_DECAY * distance.max(0.0)).exp() {
            let at = price(if on_bid { quote_bid } else { quote_ask });
            (cash, inventory) = fill(cash, inventory, on_bid, at);
            fills += 1.0;
            peak_inventory = peak_inventory.max(inventory.abs());
        }
        mid += if on_bid { -IMPACT } else { IMPACT };
        mark(cash, inventory, mid, &mut peak, &mut drawdown);
    }
    held += inventory.abs().to_f64().unwrap() * (HORIZON - now);
    let size = inventory.abs().to_f64().unwrap();
    let pnl = (cash + inventory * price(mid)).to_f64().unwrap() - size * size * IMPACT / 2.0;
    peak = peak.max(pnl);
    Outcome {
        pnl,
        fills,
        abs_inventory: held / HORIZON,
        max_drawdown: drawdown.max(peak - pnl),
        peak_inventory,
    }
}

fn outcomes(policy: Policy, params: &HawkesParameters, seeds: Range<u64>) -> Vec<Outcome> {
    let run = |seed: u64| {
        let mut sim = HawkesSimulator::new(params.clone(), seed, 10).unwrap();
        let (segment, stop) = sim.run(HORIZON).unwrap();
        // A fit whose flow stops early would read as a quiet market.
        assert!(stop.is_none(), "{stop:?}");
        segment
    };
    // The price noise is the same in every arm, by the episode's position.
    seeds
        .enumerate()
        .map(|(i, seed)| episode(policy, &run(seed * 2), &run(seed * 2 + 1), i as u64 + 1))
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_error(values: &[f64]) -> f64 {
    let m = mean(values);
    let variance =
        values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0);
    (variance / values.len() as f64).sqrt()
}

fn fitted() -> (HawkesParameters, HawkesParameters, HawkesParameters, f64) {
    let train = segments(&truth(), TRAIN);
    let validation = segments(&truth(), VALIDATION);
    let config = HawkesFitConfig::new(CAP as u32, vec![]).with_iterations(100);
    let (_, best) = select_decay(&train, &validation, &DECAYS, &config).unwrap();
    let decay = DECAYS[best.unwrap()];
    let poisson = fit_poisson(&train, CAP as u32).unwrap();
    let qr = fit_hawkes(&train, &config).unwrap();
    let hawkes = fit_hawkes(
        &train,
        &HawkesFitConfig::new(CAP as u32, vec![decay]).with_iterations(100),
    )
    .unwrap();
    (
        poisson,
        qr.parameters().unwrap().clone(),
        hawkes.parameters().unwrap().clone(),
        decay,
    )
}

#[test]
fn test_an_episode_is_a_function_of_its_inputs() {
    let params = truth();
    let one = outcomes(Policy::Fixed, &params, 1..4);
    assert_eq!(one, outcomes(Policy::Fixed, &params, 1..4));
    assert!(one.iter().all(|o| o.fills > 0.0));
}

#[test]
fn test_a_fill_moves_cash_and_inventory_with_the_fee() {
    let (cash, inventory) = fill(dec!(0), dec!(0), true, dec!(100));
    assert_eq!((cash, inventory), (dec!(-100.01), dec!(1)));
    let (cash, inventory) = fill(cash, inventory, false, dec!(101));
    assert_eq!((cash, inventory), (dec!(0.9799), dec!(0)));
}

fn market_orders(count: i32) -> HawkesSegment {
    HawkesSegment {
        initial_queue: 20,
        duration_seconds: HORIZON,
        events: (0..count)
            .map(|i| HawkesEvent {
                time_seconds: f64::from(i) + 1.0,
                kind: QrEventKind::Market,
            })
            .collect(),
    }
}

#[test]
fn test_the_inventory_limit_binds_and_the_fill_chance_is_the_distance() {
    let quiet = market_orders(0);
    // Far more orders than the limit needs: it binds, and is not passed.
    let flood = episode(Policy::Fixed, &market_orders(200), &quiet, 1);
    assert_eq!(flood.peak_inventory, dec!(10));
    // At a half spread of 0.05 and a decay of 20, about 37% of orders reach the quote.
    let fills: f64 = (1..300)
        .map(|seed| episode(Policy::Fixed, &market_orders(5), &quiet, seed).fills)
        .sum();
    let share = fills / (299.0 * 5.0);
    assert!((share - (-1.0f64).exp()).abs() < 0.05, "{share}");
}

#[test]
fn test_avellaneda_stoikov_skews_and_fixed_does_not() {
    let (flat, _) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(0), 0.0);
    let (long, _) = quotes(Policy::AvellanedaStoikov, 100.0, dec!(5), 0.0);
    assert!(long < flat);
    assert_eq!(
        quotes(Policy::Fixed, 100.0, dec!(0), 0.0),
        quotes(Policy::Fixed, 100.0, dec!(5), 0.0)
    );
    assert_eq!(quotes(Policy::Fixed, 100.0, dec!(0), 0.0), (99.95, 100.05));
}

#[test]
fn test_the_standard_error_by_hand() {
    assert_eq!(mean(&[1.0, 3.0]), 2.0);
    assert_eq!(standard_error(&[1.0, 3.0]), 1.0);
}

#[test]
fn test_a_policy_on_no_market_orders_does_nothing() {
    let quiet = HawkesSegment {
        initial_queue: 10,
        duration_seconds: HORIZON,
        events: vec![],
    };
    let outcome = episode(Policy::Fixed, &quiet, &quiet, 1);
    assert_eq!(
        outcome,
        Outcome {
            pnl: 0.0,
            fills: 0.0,
            abs_inventory: 0.0,
            max_drawdown: 0.0,
            peak_inventory: Decimal::ZERO,
        }
    );
}

#[test]
fn test_hawkes_flow_clusters_market_orders_more_than_its_poisson_fit() {
    // The arms must differ in what a policy meets, or the comparison would show nothing.
    let train = segments(&truth(), TRAIN);
    let poisson = fit_poisson(&train, CAP as u32).unwrap();
    let dispersion = |data: &[HawkesSegment]| {
        let counts: Vec<f64> = data
            .iter()
            .map(|s| {
                s.events
                    .iter()
                    .filter(|e| e.kind == QrEventKind::Market)
                    .count() as f64
            })
            .collect();
        let m = mean(&counts);
        counts.iter().map(|c| (c - m).powi(2)).sum::<f64>() / (counts.len() as f64 - 1.0) / m
    };
    assert!(dispersion(&train) > dispersion(&segments(&poisson, TRAIN)));
}

/// The evidence table. `--ignored --nocapture`, in release.
#[test]
#[ignore = "the study itself: slow without --release"]
fn test_policy_comparison_evidence_table() {
    let (poisson, qr, hawkes, decay) = fitted();
    let holdout = segments(&truth(), DATA_HOLDOUT);
    println!("SYNTHETIC. decay {decay} chosen on validation. Flow fit, on holdout:");
    println!("| model | params | ll/event | aic |\n| --- | --- | --- | --- |");
    for (name, params) in [("poisson", &poisson), ("qr", &qr), ("hawkes", &hawkes)] {
        let fit = score_flow(&holdout, params).unwrap();
        println!(
            "| {name} | {} | {:.5} | {:.0} |",
            fit.parameter_count, fit.log_likelihood_per_event, fit.aic
        );
    }
    println!(
        "\nPolicy, reported apart: mean [standard error] over {} episodes per arm, and each \
         model arm's gap to data with its standard error. Same price noise by episode.",
        SIM_SEEDS.count()
    );
    println!(
        "| policy | flow | branching | pnl | gap | fills | gap | abs inventory | gap | drawdown | gap |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let picks: [fn(&Outcome) -> f64; 4] = [
        |o| o.pnl,
        |o| o.fills,
        |o| o.abs_inventory,
        |o| o.max_drawdown,
    ];
    for policy in [Policy::Fixed, Policy::AvellanedaStoikov] {
        let data = outcomes(policy, &truth(), DATA_HOLDOUT);
        for (name, params) in [
            ("data", &truth()),
            ("poisson", &poisson),
            ("qr", &qr),
            ("hawkes", &hawkes),
        ] {
            let runs = if name == "data" {
                data.clone()
            } else {
                outcomes(policy, params, SIM_SEEDS)
            };
            let mut row = String::new();
            for pick in picks {
                let values: Vec<f64> = runs.iter().map(pick).collect();
                let base: Vec<f64> = data.iter().map(pick).collect();
                let (m, se) = (mean(&values), standard_error(&values));
                row += &format!(" {m:.3} [{se:.3}] |");
                if name == "data" {
                    row += " |";
                } else {
                    let gap = m - mean(&base);
                    let error = (se * se + standard_error(&base).powi(2)).sqrt();
                    row += &format!(" {gap:.3} [{error:.3}] |");
                }
            }
            println!(
                "| {policy:?} | {name} | {:.3} |{row}",
                params.branching_ratio()
            );
        }
    }
}
