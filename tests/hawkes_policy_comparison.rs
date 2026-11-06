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
//!
//! The fill rule takes a quote as it is. A skew can set a bid above the mid or an ask under
//! it, and such a quote fills with probability 1 at a passive fee, where a venue would have
//! crossed it. See `test_market_order_rates_and_fills_through_the_mid`.

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
    DecayScore, HawkesEvent, HawkesFitConfig, HawkesParameters, HawkesSegment, HawkesSimulator,
    fit_hawkes, fit_poisson, score_flow, select_decay,
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

/// What an episode draws: the diffusion's normals and the fills' uniforms.
trait Draws {
    fn unit(&mut self) -> f64;
    fn normal(&mut self) -> f64;
}

struct Noise(u64);

impl Draws for Noise {
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
    /// Fills at a quote already through the mid when it was set, which only a skew can do. A
    /// venue would have crossed it, not rested it at a passive fee.
    through_when_set: f64,
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
    let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9).wrapping_add(1));
    episode_with(policy, bid, ask, &mut noise)
}

fn episode_with(
    policy: Policy,
    bid: &HawkesSegment,
    ask: &HawkesSegment,
    noise: &mut impl Draws,
) -> Outcome {
    let mut events: Vec<(f64, bool, QrEventKind)> = bid
        .events
        .iter()
        .map(|e| (e.time_seconds, true, e.kind))
        .chain(ask.events.iter().map(|e| (e.time_seconds, false, e.kind)))
        .collect();
    events.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut mid, mut now) = (100.0f64, 0.0f64);
    let (mut cash, mut inventory) = (Decimal::ZERO, Decimal::ZERO);
    let (mut peak, mut drawdown, mut fills, mut held) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let limit = dec!(10);
    let mut peak_inventory = Decimal::ZERO;
    let mut through_when_set = 0.0f64;
    for (time, on_bid, kind) in events {
        let (quote_bid, quote_ask) = quotes(policy, mid, inventory, now);
        let mid_when_set = mid;
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
            let through = if on_bid {
                quote_bid > mid_when_set
            } else {
                quote_ask < mid_when_set
            };
            through_when_set += f64::from(u8::from(through));
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
        through_when_set,
    }
}

/// The two queues of an episode, each on its own seed.
fn queues(params: &HawkesParameters, seed: u64) -> (HawkesSegment, HawkesSegment) {
    let run = |seed: u64| {
        let mut sim = HawkesSimulator::new(params.clone(), seed, 10).unwrap();
        let (segment, stop) = sim.run(HORIZON).unwrap();
        // A fit whose flow stops early would read as a quiet market.
        assert!(stop.is_none(), "{stop:?}");
        segment
    };
    (run(seed * 2), run(seed * 2 + 1))
}

fn outcomes(policy: Policy, params: &HawkesParameters, seeds: Range<u64>) -> Vec<Outcome> {
    // Each episode's generator starts from the same seed in every arm, by its position. The
    // arms draw from it at their own events, so their paths differ, yet are not independent.
    seeds
        .enumerate()
        .map(|(i, seed)| {
            let (bid, ask) = queues(params, seed);
            episode(policy, &bid, &ask, i as u64 + 1)
        })
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

/// The mean of the differences by episode, and its standard error, so that what the pairs
/// share is counted.
fn paired_gap(arm: &[f64], base: &[f64]) -> (f64, f64) {
    let differences: Vec<f64> = arm.iter().zip(base).map(|(a, b)| a - b).collect();
    (mean(&differences), standard_error(&differences))
}

/// The seeds of the fits, and of an arm's episodes.
fn fit_seeds() -> (Range<u64>, Range<u64>) {
    (TRAIN, VALIDATION)
}

fn arm_seeds(name: &str) -> Range<u64> {
    if name == "data" {
        DATA_HOLDOUT
    } else {
        SIM_SEEDS
    }
}

/// The three fits, the decay chosen, and the scores it was chosen on.
fn fitted() -> (
    HawkesParameters,
    HawkesParameters,
    HawkesParameters,
    f64,
    Vec<DecayScore>,
) {
    let (train, validation) = fit_seeds();
    let train = segments(&truth(), train);
    let validation = segments(&truth(), validation);
    let config = HawkesFitConfig::new(CAP as u32, vec![]).with_iterations(100);
    let (scores, best) = select_decay(&train, &validation, &DECAYS, &config).unwrap();
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
        scores,
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
fn test_the_gap_counts_what_the_pairs_share() {
    // Moving together, the differences have no spread, though each arm has.
    let (gap, error) = paired_gap(&[1.0, 3.0, 5.0], &[0.0, 2.0, 4.0]);
    assert_eq!((gap, error), (1.0, 0.0));
    // Moving against each other, they have more than each arm.
    let (_, error) = paired_gap(&[1.0, 3.0], &[3.0, 1.0]);
    assert_eq!(error, 2.0);
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
            through_when_set: 0.0,
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

/// Given draws: normals that run out at zero, and fills that all reach the quote unless told.
#[derive(Default)]
struct Scripted {
    normals: std::collections::VecDeque<f64>,
    units: std::collections::VecDeque<f64>,
}

impl Draws for Scripted {
    fn unit(&mut self) -> f64 {
        self.units.pop_front().unwrap_or(0.0)
    }

    fn normal(&mut self) -> f64 {
        self.normals.pop_front().unwrap_or(0.0)
    }
}

fn at_times(times: &[f64]) -> HawkesSegment {
    HawkesSegment {
        initial_queue: 20,
        duration_seconds: HORIZON,
        events: times
            .iter()
            .map(|&time_seconds| HawkesEvent {
                time_seconds,
                kind: QrEventKind::Market,
            })
            .collect(),
    }
}

fn close(found: f64, expected: f64) {
    assert!((found - expected).abs() < 1e-9, "{found} {expected}");
}

#[test]
fn test_an_episode_by_hand_with_a_position_at_the_end() {
    let quiet = at_times(&[]);
    let o = episode_with(
        Policy::Fixed,
        &at_times(&[100.0, 200.0]),
        &quiet,
        &mut Scripted::default(),
    );
    // Buys at 99.95, then at 99.945 after the first order's impact, with a fee of 1 bp: cash
    // -199.9149895. The mid goes to 99.995, then 99.99, a position of 2 is worth 199.98, and
    // the charge at the end is 2 * 2 * 0.005 / 2.
    assert_eq!(o.fills, 2.0);
    assert_eq!(o.peak_inventory, dec!(2));
    close(o.pnl, 0.0550105);
    // Equity 0.035005, then 0.0650105: the only fall is the charge.
    close(o.max_drawdown, 0.01);
    // Held 0, 1, 2 for 100 s each, over 300 s.
    close(o.abs_inventory, 1.0);
}

#[test]
fn test_drawdown_is_sampled_at_each_market_order_and_an_ask_moves_the_mid_up() {
    // A buy, then two sells that miss: the first order's impact is undone twice.
    let draws = Scripted {
        units: [0.0, 1.0, 1.0, 1.0].into(),
        ..Scripted::default()
    };
    let mut draws = draws;
    let o = episode_with(
        Policy::Fixed,
        &at_times(&[10.0, 20.0]),
        &at_times(&[30.0, 40.0]),
        &mut draws,
    );
    assert_eq!(o.fills, 1.0);
    // Equity 0.035005, 0.030005, 0.035005, 0.040005, then 0.037505 after the charge of 0.0025.
    close(o.max_drawdown, 0.005);
    close(o.pnl, 0.037505);
}

#[test]
fn test_a_quote_rests_while_the_mid_moves_before_the_order() {
    // SIGMA * sqrt(100) = 0.2, so the mid is 100.2 when the order comes, at the quote of 99.95.
    let mut draws = Scripted {
        normals: [1.0].into(),
        ..Scripted::default()
    };
    let o = episode_with(
        Policy::Fixed,
        &at_times(&[100.0]),
        &at_times(&[]),
        &mut draws,
    );
    // Cash -99.959995, one unit at 100.195 after the impact, a charge of 0.0025.
    close(o.pnl, 0.232505);
}

#[test]
fn test_avellaneda_stoikov_in_an_episode_sees_its_inventory_and_the_time_left() {
    let o = episode_with(
        Policy::AvellanedaStoikov,
        &at_times(&[100.0, 200.0]),
        &at_times(&[]),
        &mut Scripted::default(),
    );
    let first = quotes(Policy::AvellanedaStoikov, 100.0, dec!(0), 0.0).0;
    let moved = 100.0 - IMPACT;
    let second = quotes(Policy::AvellanedaStoikov, moved, dec!(1), 100.0).0;
    let (cash, inventory) = fill(Decimal::ZERO, Decimal::ZERO, true, price(first));
    let (cash, inventory) = fill(cash, inventory, true, price(second));
    let end = (cash + inventory * price(moved - IMPACT)).to_f64().unwrap();
    close(o.pnl, end - 2.0 * 2.0 * IMPACT / 2.0);
    assert_ne!(
        second,
        quotes(Policy::AvellanedaStoikov, moved, dec!(0), 100.0).0
    );
    assert_ne!(
        second,
        quotes(Policy::AvellanedaStoikov, moved, dec!(1), 0.0).0
    );
}

#[test]
fn test_a_fill_through_the_mid_when_set_is_counted() {
    let run = |policy| {
        episode_with(
            policy,
            &at_times(&[1.0]),
            &at_times(&[2.0]),
            &mut Scripted::default(),
        )
    };
    // Long one unit with 299 s left, the skew sets the ask under the mid.
    assert!(quotes(Policy::AvellanedaStoikov, 100.0, dec!(1), 1.0).1 < 100.0);
    let skewed = run(Policy::AvellanedaStoikov);
    assert_eq!((skewed.fills, skewed.through_when_set), (2.0, 1.0));
    assert_eq!(run(Policy::Fixed).through_when_set, 0.0);
}

#[test]
fn test_each_queue_has_its_own_seed_and_the_seed_ranges_do_not_meet() {
    let (bid, ask) = queues(&truth(), 3);
    assert_ne!(bid.events, ask.events);
    const {
        // `queues` uses 2 s and 2 s + 1, and the fits use the seeds themselves.
        assert!((TRAIN.end - 1) * 2 + 1 < DATA_HOLDOUT.start * 2);
        assert!(VALIDATION.start >= TRAIN.end);
        assert!(VALIDATION.end <= DATA_HOLDOUT.start);
        assert!((DATA_HOLDOUT.end - 1) * 2 + 1 < SIM_SEEDS.start * 2);
    };
}

#[test]
fn test_the_arms_use_their_own_seeds_and_the_fits_stay_off_the_holdout() {
    assert_eq!(arm_seeds("data"), DATA_HOLDOUT);
    for name in ["poisson", "qr", "hawkes"] {
        assert_eq!(arm_seeds(name), SIM_SEEDS);
    }
    assert_eq!(fit_seeds(), (TRAIN, VALIDATION));
}

#[test]
fn test_each_arm_is_the_model_it_names() {
    let (poisson, qr, hawkes, decay, _) = fitted();
    assert!(DECAYS.contains(&decay));
    assert_eq!(poisson.branching_ratio(), 0.0);
    assert_eq!(qr.branching_ratio(), 0.0);
    assert!(hawkes.branching_ratio() > 0.0);
    // Fits, not the truth, and the queue-reactive one has its rates by size.
    assert_ne!(qr, truth());
    assert_ne!(hawkes, truth());
    assert_ne!(poisson, qr);
}

/// Market orders per queue per episode in each arm, and the fills through the mid when set: the
/// limits stated with the table. `--ignored --nocapture`, in release.
#[test]
#[ignore = "fits the models and runs every arm"]
fn test_market_order_rates_and_fills_through_the_mid() {
    let (poisson, qr, hawkes, _, _) = fitted();
    for (name, params) in [
        ("data", &truth()),
        ("poisson", &poisson),
        ("qr", &qr),
        ("hawkes", &hawkes),
    ] {
        let counts: Vec<f64> = arm_seeds(name)
            .flat_map(|seed| {
                let (bid, ask) = queues(params, seed);
                [bid, ask]
            })
            .map(|segment| {
                segment
                    .events
                    .iter()
                    .filter(|e| e.kind == QrEventKind::Market)
                    .count() as f64
            })
            .collect();
        let runs = outcomes(Policy::AvellanedaStoikov, params, arm_seeds(name));
        let through: f64 = runs.iter().map(|o| o.through_when_set).sum();
        let fills: f64 = runs.iter().map(|o| o.fills).sum();
        println!(
            "{name}: {:.2} [{:.2}] market orders per queue; AS {through} of {fills} through the mid",
            mean(&counts),
            standard_error(&counts)
        );
    }
}

/// One row of the policy table: each metric, and its gap to the data but on the data's own row.
fn policy_row(
    policy: Policy,
    name: &str,
    branching: f64,
    runs: &[Outcome],
    data: &[Outcome],
) -> String {
    let picks: [fn(&Outcome) -> f64; 4] = [
        |o| o.pnl,
        |o| o.fills,
        |o| o.abs_inventory,
        |o| o.max_drawdown,
    ];
    let mut row = String::new();
    for pick in picks {
        let values: Vec<f64> = runs.iter().map(pick).collect();
        let base: Vec<f64> = data.iter().map(pick).collect();
        let (m, se) = (mean(&values), standard_error(&values));
        row += &format!(" {m:.3} [{se:.3}] |");
        if name == "data" {
            row += " |";
        } else {
            let (gap, error) = paired_gap(&values, &base);
            row += &format!(" {gap:.3} [{error:.3}] |");
        }
    }
    format!("| {policy:?} | {name} | {branching:.3} |{row}")
}

#[test]
fn test_a_policy_row_prints_each_metric_and_the_gap_with_its_sign() {
    let outcome = |pnl, fills, abs_inventory, max_drawdown| Outcome {
        pnl,
        fills,
        abs_inventory,
        max_drawdown,
        peak_inventory: Decimal::ZERO,
        through_when_set: 0.0,
    };
    let data = [outcome(1.0, 10.0, 3.0, 5.0), outcome(3.0, 10.0, 3.0, 5.0)];
    // The arm runs above the data on PnL and below on the rest, each by its own amount.
    let arm = [outcome(4.0, 9.0, 2.5, 4.0), outcome(6.0, 9.0, 2.5, 4.0)];
    let row = policy_row(Policy::Fixed, "qr", 0.25, &arm, &data);
    let cells: Vec<&str> = row.split('|').map(str::trim).collect();
    assert_eq!(&cells[1..4], ["Fixed", "qr", "0.250"]);
    assert!(cells[4].starts_with("5.000 "), "{}", cells[4]);
    assert!(cells[5].starts_with("3.000 "), "{}", cells[5]);
    assert!(cells[6].starts_with("9.000 "), "{}", cells[6]);
    assert!(cells[7].starts_with("-1.000 "), "{}", cells[7]);
    assert!(cells[8].starts_with("2.500 "), "{}", cells[8]);
    assert!(cells[9].starts_with("-0.500 "), "{}", cells[9]);
    assert!(cells[10].starts_with("4.000 "), "{}", cells[10]);
    assert!(cells[11].starts_with("-1.000 "), "{}", cells[11]);
    // The data has no gap to itself.
    let own = policy_row(Policy::Fixed, "data", 0.25, &data, &data);
    assert_eq!(own.split('|').map(str::trim).nth(5), Some(""));
}

#[test]
fn test_the_fill_chance_is_read_at_the_mid_when_the_order_comes() {
    // With the mid at 100.2 the bid of 99.95 is 0.25 away: e^-5 is under a draw of 0.1. From
    // the mid when it was set it would be 0.05 away, and e^-1 is over it.
    let mut draws = Scripted {
        normals: [1.0].into(),
        units: [0.1].into(),
    };
    let o = episode_with(
        Policy::Fixed,
        &at_times(&[100.0]),
        &at_times(&[]),
        &mut draws,
    );
    assert_eq!(o.fills, 0.0);
}

#[test]
fn test_a_quote_the_mid_fell_through_after_it_was_set_is_not_through_when_set() {
    // SIGMA * sqrt(100) * -10 = -2, so the mid is 98 under the bid of 99.95 at the order.
    let mut draws = Scripted {
        normals: [-10.0].into(),
        ..Scripted::default()
    };
    let o = episode_with(
        Policy::Fixed,
        &at_times(&[100.0]),
        &at_times(&[]),
        &mut draws,
    );
    assert_eq!((o.fills, o.through_when_set), (1.0, 0.0));
}

#[test]
fn test_fitted_trains_and_validates_on_its_own_seeds_and_uses_the_chosen_decay() {
    let (poisson, qr, hawkes, decay, scores) = fitted();
    let train = segments(&truth(), TRAIN);
    let validation = segments(&truth(), VALIDATION);
    let config = HawkesFitConfig::new(CAP as u32, vec![]).with_iterations(100);
    let (expected, best) = select_decay(&train, &validation, &DECAYS, &config).unwrap();
    // The same decay could win on other data, so the scores tell which data it was.
    assert_eq!(scores, expected);
    assert_eq!(decay, DECAYS[best.unwrap()]);
    assert_eq!(poisson, fit_poisson(&train, CAP as u32).unwrap());
    assert_eq!(
        &qr,
        fit_hawkes(&train, &config).unwrap().parameters().unwrap()
    );
    let with_decay = HawkesFitConfig::new(CAP as u32, vec![decay]).with_iterations(100);
    assert_eq!(
        &hawkes,
        fit_hawkes(&train, &with_decay)
            .unwrap()
            .parameters()
            .unwrap()
    );
    assert_eq!(hawkes.decays(), [decay]);
}

/// The evidence table. `--ignored --nocapture`, in release.
#[test]
#[ignore = "the study itself: slow without --release"]
fn test_policy_comparison_evidence_table() {
    let (poisson, qr, hawkes, decay, _) = fitted();
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
         model arm's gap to data, with the standard error of the differences by episode. Same \
         initial noise seed by episode, with different draws at each arm's events.",
        SIM_SEEDS.count()
    );
    println!(
        "| policy | flow | branching | pnl | gap | fills | gap | abs inventory | gap | drawdown | gap |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for policy in [Policy::Fixed, Policy::AvellanedaStoikov] {
        let data = outcomes(policy, &truth(), arm_seeds("data"));
        for (name, params) in [
            ("data", &truth()),
            ("poisson", &poisson),
            ("qr", &qr),
            ("hawkes", &hawkes),
        ] {
            let runs = if name == "data" {
                data.clone()
            } else {
                outcomes(policy, params, arm_seeds(name))
            };
            println!(
                "{}",
                policy_row(policy, name, params.branching_ratio(), &runs, &data)
            );
        }
    }
}
