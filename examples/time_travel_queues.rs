//! Chronological replay against consistent time travel for a market maker at the touch (#109).
//!
//! #219's example compares the two modes on an abstract state. This one does it on queues:
//! - (a) on #102's queue-reactive days (`synthetic`), where the market reacts to an order;
//! - (b) on the Hyperliquid capture (`capture`).
//!
//! In (a) the queue-reactive simulator continues from a state that holds the agent's order. So
//! the true continuation after an action is known, and each mode is measured against it. In (b)
//! there is no truth, only each mode's own measures.
//!
//! Time travel (Ragel and Challet, <https://arxiv.org/abs/2408.02322v2>) is a separate mode,
//! labelled in every row, and it is not causal evidence. Chronological replay is the baseline:
//! the tape goes on as it happened, and an action has no effect on the market.
//!
//! # State and actions
//!
//! The state on a one second grid is:
//! - the spread in ticks (1, 2, or 3 and more);
//! - each side's touch queue in event sizes, in buckets 0-1, 2, 3, 4, 5-9 and 10 and more.
//!
//! An action forces the next state:
//! - joining a side adds one event size to its queue;
//! - taking a side removes one. Taking the last one opens the spread by a tick, and the
//!   queue there is unknown, so it is put in the lowest bucket.
//!
//! The rules are fixed, and no policy is trained. #102's learned policies are not here: they
//! act on another observation and another scale, and running them on this state would test the
//! mapping, not the mode.
//!
//! # The search
//!
//! Time travel lands on a time `t'` in the same split, with the current state and the forced
//! next state, by equations 1 and 2 of the paper. It never lands within [`SEPARATIONS`] of the
//! current time, the separation being chosen on validation. With no candidate it goes on in
//! order, which is inconsistent and counted. A search never leaves its split.
//!
//! # The protocol
//!
//! Everything below is fixed before any run. The holdout is read once.
//!
//! Run (a) with `cargo run --release --features serde --example time_travel_queues -- synthetic
//! bridges/python/experiments/m4_05/design.json`, and (b) with `... -- capture <capture dir>`.
//! No data is in the repository.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::queue_reactive::{
    MAX_LIMITS, QrParameters, QrSimulator, QrState, QrStep, UnsupportedPolicy,
};
use serde_json::Value;

type Failure = Box<dyn std::error::Error>;

/// The grid of states, in milliseconds.
const GRID_MS: u64 = 1_000;

/// The horizon of a markout and of a move, in grid steps.
const HORIZON_STEPS: usize = 10;

/// Bucket bounds of a touch queue, in event sizes: 0-1, 2, 3, 4, 5-9, 10 and more. #102's
/// regimes cap a queue at 4 event sizes, so (a) uses the first four; the capture's queues are
/// longer.
const QUEUE_BUCKETS: [u32; 5] = [2, 3, 4, 5, 10];

/// The largest spread state, in ticks: 3 stands for 3 and more.
const MAX_SPREAD_STATE: u32 = 3;

/// The separations a landing keeps from the current time, in grid steps. Validation chooses
/// one, by the smallest no-candidate share, the shorter on a tie.
const SEPARATIONS: [usize; 3] = [5, 10, 30];

/// The imbalance above which `TakeImbalance` takes the thinner side.
const TAKE_IMBALANCE: f64 = 0.6;

/// Seeds of the search's draws.
const SEARCH_SEEDS: [u64; 5] = [11, 23, 37, 53, 71];

/// (a): the hours simulated from each day's start, on the grid. A whole day at one second,
/// with a full simulator state per step, would hold 6.2 million states over 72 days, so the
/// first [`SYNTHETIC_HOURS`] hours of each are used.
const SYNTHETIC_HOURS: u64 = 4;

/// (a): the splits, as #102's: days 1 to 14, 15 to 19 and 20 to 24 of each regime.
const SPLITS: [(&str, u64, u64); 3] = [
    ("train", 1, 14),
    ("validation", 15, 19),
    ("holdout", 20, 24),
];

/// (b): the capture's split of #257's longest warm stretch by time: train, validation and
/// holdout shares.
const CAPTURE_SPLIT: [f64; 3] = [0.6, 0.2, 0.2];

/// Intervals: a moving block bootstrap over days, as #102.
const REPLICATES: usize = 2_000;
/// Mean block length, in days.
const MEAN_BLOCK_DAYS: f64 = 2.0;
/// Seed of the bootstrap.
const BOOTSTRAP_SEED: u64 = 109;

/// Stop rule of (b): above this no-candidate share on train, the mode is reported unusable on
/// the capture and its holdout is not read.
const MAX_NO_CANDIDATE_SHARE: f64 = 0.5;

/// Day seeds, as #102's `qr_days`: the regime's number times this stride, plus the day.
const SEED_STRIDE: u64 = 1_000_003;

/// A deterministic generator for the search's draws (splitmix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        // n is small, so the modulo bias is negligible and stated.
        usize::try_from(self.next() % u64::try_from(n).unwrap_or(u64::MAX)).unwrap_or(0)
    }
}

/// The state a rule and the search see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct State {
    spread: u32,
    bid: u8,
    ask: u8,
}

fn bucket(queue: u32) -> u8 {
    let above = QUEUE_BUCKETS
        .iter()
        .filter(|bound| queue >= **bound)
        .count();
    u8::try_from(above).unwrap_or(u8::MAX)
}

/// What a grid time shows: the spread in ticks, each side's touch queue in event sizes and
/// the reference price in ticks, which stands for the mid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sample {
    spread: u32,
    bid: u32,
    ask: u32,
    mid: i64,
}

impl Sample {
    fn state(&self) -> State {
        State {
            spread: self.spread.min(MAX_SPREAD_STATE),
            bid: bucket(self.bid),
            ask: bucket(self.ask),
        }
    }

    fn imbalance(&self) -> Option<f64> {
        let total = f64::from(self.bid) + f64::from(self.ask);
        (total > 0.0).then(|| (f64::from(self.bid) - f64::from(self.ask)) / total)
    }
}

/// The nearest non-empty limit of a side, and its queue; `limits` and 0 when it is empty.
fn touch(queues: &[u32; MAX_LIMITS], limits: usize) -> (usize, u32) {
    match queues.iter().take(limits).position(|q| *q > 0) {
        Some(offset) => (offset, queues[offset]),
        None => (limits, 0),
    }
}

fn sample_of(state: &QrState, limits: usize) -> Sample {
    let (bid_offset, bid) = touch(&state.bid, limits);
    let (ask_offset, ask) = touch(&state.ask, limits);
    let spread = u32::try_from(bid_offset + ask_offset + 1).unwrap_or(u32::MAX);
    Sample {
        spread,
        bid,
        ask,
        mid: state.reference_ticks,
    }
}

/// The simulator's state at each grid time from `start` seconds: the state after every event
/// at or before that time.
fn grid(sim: &mut QrSimulator, steps: usize, start: f64) -> Result<Vec<QrState>, Failure> {
    let mut out = Vec::with_capacity(steps);
    let mut current = sim.state();
    let mut pending: Option<QrState> = None;
    let mut stalled = false;
    let seconds = GRID_MS as f64 / 1000.0;
    for k in 0..steps {
        let at = start + k as f64 * seconds;
        loop {
            if let Some(next) = pending {
                if next.time_seconds <= at {
                    current = next;
                    pending = None;
                } else {
                    break;
                }
            }
            if stalled {
                break;
            }
            match sim.step()? {
                QrStep::Event { .. } => pending = Some(sim.state()),
                QrStep::Stalled | QrStep::EventLimit => stalled = true,
                QrStep::Unsupported { .. } => {
                    return Err("a queue the parameters do not simulate".into());
                }
                _ => return Err("a simulator step this study does not know".into()),
            }
        }
        out.push(current);
    }
    Ok(out)
}

/// What a market maker does at the touch at one decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Action {
    Hold,
    JoinBid,
    JoinAsk,
    JoinBoth,
    TakeBid,
    TakeAsk,
}

/// The state with the action applied: one event size added to a side's touch queue to join
/// it, or removed to take it. `None` when the action cannot apply: a side with no touch, or a
/// queue at the simulator's cap.
fn force(state: &QrState, action: Action, limits: usize, cap: u32) -> Option<QrState> {
    let mut out = *state;
    let join = |queues: &mut [u32; MAX_LIMITS]| -> Option<()> {
        let (offset, queue) = touch(queues, limits);
        if offset >= limits || queue >= cap {
            return None;
        }
        queues[offset] = queue + 1;
        Some(())
    };
    match action {
        Action::Hold => {}
        Action::JoinBid => join(&mut out.bid)?,
        Action::JoinAsk => join(&mut out.ask)?,
        Action::JoinBoth => {
            join(&mut out.bid)?;
            join(&mut out.ask)?;
        }
        Action::TakeBid | Action::TakeAsk => {
            let queues = if action == Action::TakeBid {
                &mut out.bid
            } else {
                &mut out.ask
            };
            let (offset, queue) = touch(queues, limits);
            if offset >= limits {
                return None;
            }
            queues[offset] = queue - 1;
        }
    }
    Some(out)
}

/// A fixed rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Rule {
    /// Never acts: a control, for which every mode is the tape and the truth.
    Hold,
    /// Joins both sides at every decision.
    JoinBoth,
    /// Joins the side with the smaller queue, the bid on a tie.
    JoinSmaller,
    /// Takes the thinner side when the imbalance is beyond [`TAKE_IMBALANCE`].
    TakeImbalance,
}

const RULES: [Rule; 4] = [
    Rule::Hold,
    Rule::JoinBoth,
    Rule::JoinSmaller,
    Rule::TakeImbalance,
];

impl Rule {
    fn act(self, sample: &Sample) -> Action {
        match self {
            Rule::Hold => Action::Hold,
            Rule::JoinBoth => Action::JoinBoth,
            Rule::JoinSmaller => {
                if sample.ask < sample.bid {
                    Action::JoinAsk
                } else {
                    Action::JoinBid
                }
            }
            Rule::TakeImbalance => match sample.imbalance() {
                Some(i) if i > TAKE_IMBALANCE => Action::TakeAsk,
                Some(i) if i < -TAKE_IMBALANCE => Action::TakeBid,
                _ => Action::Hold,
            },
        }
    }
}

/// How the tape is replayed. A row of a report carries its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
    Chronological,
    TimeTravelAny,
    TimeTravelForward,
}

const MODES: [Mode; 3] = [
    Mode::Chronological,
    Mode::TimeTravelAny,
    Mode::TimeTravelForward,
];

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Chronological => "chronological replay",
            Mode::TimeTravelAny => "TIME TRAVEL, any in split, not causal",
            Mode::TimeTravelForward => "TIME TRAVEL, forward only, not causal",
        }
    }
}

/// One day of a regime, on the grid.
struct Day {
    regime: usize,
    day: u64,
    states: Vec<QrState>,
    samples: Vec<Sample>,
}

/// A position on the tapes of one split: a day's index and a grid step.
type Position = (usize, usize);

/// The times of each transition of the state inside a split, where a landing leaves a whole
/// horizon after it.
struct Search {
    by_transition: HashMap<(State, State), Vec<Position>>,
    mean_move: HashMap<(State, State), f64>,
}

impl Search {
    fn new(days: &[&Day]) -> Self {
        let mut by_transition: HashMap<(State, State), Vec<Position>> = HashMap::new();
        let mut sums: HashMap<(State, State), (f64, usize)> = HashMap::new();
        for (index, day) in days.iter().enumerate() {
            let last = day.samples.len().saturating_sub(HORIZON_STEPS + 2);
            for k in 0..last {
                let key = (day.samples[k].state(), day.samples[k + 1].state());
                by_transition.entry(key).or_default().push((index, k));
                let moved = (day.samples[k + HORIZON_STEPS].mid - day.samples[k].mid) as f64;
                let entry = sums.entry(key).or_insert((0.0, 0));
                entry.0 += moved;
                entry.1 += 1;
            }
        }
        let mean_move = sums
            .into_iter()
            .map(|(k, (s, n))| (k, s / n as f64))
            .collect();
        Search {
            by_transition,
            mean_move,
        }
    }

    /// A landing for this transition by the paper's equations 1 and 2: uniform over the
    /// times with it, at least `separation` steps from now on the same day, and later than
    /// now in forward mode.
    fn land(
        &self,
        key: (State, State),
        now: Position,
        separation: usize,
        forward: bool,
        rng: &mut Rng,
    ) -> Option<Position> {
        let times = self.by_transition.get(&key)?;
        let candidates: Vec<Position> = times
            .iter()
            .copied()
            .filter(|p| p.0 != now.0 || p.1.abs_diff(now.1) >= separation)
            .filter(|p| !forward || *p > now)
            .collect();
        (!candidates.is_empty()).then(|| candidates[rng.below(candidates.len())])
    }
}

/// What a decision's truth is in (a): the state one step on and the mid's move over the
/// horizon, from the simulator continued with the agent's order in its state.
type Truth = Option<(State, i64)>;

/// One rule's walk over one day in one mode, measured against the truth.
#[derive(Debug, Default, Clone)]
struct Walk {
    steps: usize,
    jumps: usize,
    cross_day: usize,
    jump_steps: usize,
    reused: usize,
    no_candidate: usize,
    unforceable: usize,
    compared: usize,
    move_error: f64,
    state_mismatch: usize,
    landing_move: f64,
    landing_control: f64,
    takes: usize,
    take_markout_mode: f64,
    take_markout_truth: f64,
    joins: usize,
    join_move_mode: f64,
    join_move_truth: f64,
}

/// The truth of an action at a position, cached per split.
struct Truths<'a> {
    params: &'a QrParameters,
    days: &'a [&'a Day],
    limits: usize,
    cap: u32,
    cache: HashMap<(Position, Action), Truth>,
}

impl Truths<'_> {
    fn get(&mut self, at: Position, action: Action) -> Result<Truth, Failure> {
        if let Some(found) = self.cache.get(&(at, action)) {
            return Ok(*found);
        }
        let day = self.days[at.0];
        let truth = if action == Action::Hold {
            Some((
                day.samples[at.1 + 1].state(),
                day.samples[at.1 + HORIZON_STEPS].mid - day.samples[at.1].mid,
            ))
        } else {
            match force(&day.states[at.1], action, self.limits, self.cap) {
                None => None,
                Some(forced) => {
                    let start = at.1 as f64 * GRID_MS as f64 / 1000.0;
                    let mut sim = QrSimulator::from_state(self.params.clone(), forced)?;
                    let states = grid(&mut sim, HORIZON_STEPS + 1, start)?;
                    let next = sample_of(&states[1], self.limits).state();
                    let moved = states[HORIZON_STEPS].reference_ticks - day.samples[at.1].mid;
                    Some((next, moved))
                }
            }
        };
        self.cache.insert((at, action), truth);
        Ok(truth)
    }
}

/// The state the action forces next, as the search sees it.
fn forced_state(day: &Day, k: usize, action: Action, limits: usize, cap: u32) -> Option<State> {
    if action == Action::Hold {
        return Some(day.samples[k + 1].state());
    }
    force(&day.states[k], action, limits, cap).map(|s| sample_of(&s, limits).state())
}

#[allow(clippy::too_many_arguments)]
fn walk(
    days: &[&Day],
    start: usize,
    rule: Rule,
    mode: Mode,
    separation: usize,
    seed: u64,
    search: &Search,
    truths: &mut Truths<'_>,
) -> Result<Walk, Failure> {
    let limits = truths.limits;
    let cap = truths.cap;
    let mut rng = Rng(seed ^ (start as u64).wrapping_mul(0x9E37_79B9));
    let mut out = Walk::default();
    let mut landed: HashSet<Position> = HashSet::new();
    let mut at: Position = (start, 0);
    let steps = days[start].samples.len().saturating_sub(HORIZON_STEPS + 2);
    for _ in 0..steps {
        let day = days[at.0];
        if at.1 + HORIZON_STEPS + 1 >= day.samples.len() {
            break;
        }
        let sample = day.samples[at.1];
        let action = rule.act(&sample);
        let truth = truths.get(at, action)?;
        let natural = day.samples[at.1 + 1].state();
        let forced = forced_state(day, at.1, action, limits, cap);
        if action != Action::Hold && forced.is_none() {
            out.unforceable += 1;
        }
        // Where the replay goes on from, and the time its move is read from.
        let mut base = at;
        let mut next = (at.0, at.1 + 1);
        if mode != Mode::Chronological
            && let Some(target) = forced.filter(|f| *f != natural)
        {
            let key = (sample.state(), target);
            match search.land(
                key,
                at,
                separation,
                mode == Mode::TimeTravelForward,
                &mut rng,
            ) {
                Some(landing) => {
                    out.jumps += 1;
                    if landing.0 == at.0 {
                        out.jump_steps += landing.1.abs_diff(at.1);
                    } else {
                        out.cross_day += 1;
                    }
                    if !landed.insert(landing) {
                        out.reused += 1;
                    }
                    let tape = days[landing.0];
                    out.landing_move += (tape.samples[landing.1 + HORIZON_STEPS].mid
                        - tape.samples[landing.1].mid)
                        as f64;
                    out.landing_control += search.mean_move.get(&key).copied().unwrap_or(0.0);
                    base = landing;
                    next = (landing.0, landing.1 + 1);
                }
                None => out.no_candidate += 1,
            }
        }
        let tape = days[base.0];
        let mode_move = tape.samples[base.1 + HORIZON_STEPS].mid - tape.samples[base.1].mid;
        let mode_next = days[next.0].samples[next.1].state();
        if let Some((truth_next, truth_move)) = truth {
            out.compared += 1;
            out.move_error += (mode_move - truth_move).abs() as f64;
            out.state_mismatch += usize::from(mode_next != truth_next);
            let half = f64::from(sample.spread) / 2.0;
            match action {
                Action::TakeAsk | Action::TakeBid => {
                    let sign = if action == Action::TakeAsk { 1.0 } else { -1.0 };
                    out.takes += 1;
                    out.take_markout_mode += sign * mode_move as f64 - half;
                    out.take_markout_truth += sign * truth_move as f64 - half;
                }
                Action::JoinBid | Action::JoinAsk => {
                    let sign = if action == Action::JoinBid { 1.0 } else { -1.0 };
                    out.joins += 1;
                    out.join_move_mode += sign * mode_move as f64;
                    out.join_move_truth += sign * truth_move as f64;
                }
                _ => {}
            }
        }
        out.steps += 1;
        at = next;
    }
    Ok(out)
}

/// A moving block bootstrap interval of the mean of per-day values, as #102's.
fn interval(values: &[f64], seed: u64) -> (f64, f64, f64) {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n.max(1) as f64;
    if n < 2 {
        return (mean, mean, mean);
    }
    let mut rng = Rng(seed);
    let p = 1.0 / MEAN_BLOCK_DAYS;
    let mut draws = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let mut i = rng.below(n);
        let mut total = 0.0;
        for k in 0..n {
            if k > 0 {
                let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
                i = if u < p { rng.below(n) } else { (i + 1) % n };
            }
            total += values[i];
        }
        draws.push(total / n as f64);
    }
    draws.sort_by(f64::total_cmp);
    let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
    (mean, at(0.05), at(0.95))
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Per day and rule, each mode's walk, the search seeds averaged inside a day.
#[derive(Debug, Default, Clone)]
struct DayResult {
    day: Walk,
    seeds: usize,
}

fn add(total: &mut Walk, w: &Walk) {
    total.steps += w.steps;
    total.jumps += w.jumps;
    total.cross_day += w.cross_day;
    total.jump_steps += w.jump_steps;
    total.reused += w.reused;
    total.no_candidate += w.no_candidate;
    total.unforceable += w.unforceable;
    total.compared += w.compared;
    total.move_error += w.move_error;
    total.state_mismatch += w.state_mismatch;
    total.landing_move += w.landing_move;
    total.landing_control += w.landing_control;
    total.takes += w.takes;
    total.take_markout_mode += w.take_markout_mode;
    total.take_markout_truth += w.take_markout_truth;
    total.joins += w.joins;
    total.join_move_mode += w.join_move_mode;
    total.join_move_truth += w.join_move_truth;
}

/// Each mode and rule's per-day results, in order.
type Results = BTreeMap<(Mode, Rule), Vec<DayResult>>;

/// Every walk of a split, per regime window: (mode, rule) -> per-day results in order.
fn run_split(
    params: &[QrParameters],
    days: &[Day],
    split: &str,
    separation: usize,
) -> Result<Results, Failure> {
    let (_, first, last) = SPLITS
        .iter()
        .find(|s| s.0 == split)
        .copied()
        .ok_or("split")?;
    let mut out: Results = BTreeMap::new();
    for (regime, params) in params.iter().enumerate() {
        let window: Vec<&Day> = days
            .iter()
            .filter(|d| d.regime == regime && (first..=last).contains(&d.day))
            .collect();
        let search = Search::new(&window);
        let limits = params.limits();
        let cap = u32::try_from(params.max_queue()).unwrap_or(u32::MAX);
        let mut truths = Truths {
            params,
            days: &window,
            limits,
            cap,
            cache: HashMap::new(),
        };
        for start in 0..window.len() {
            for rule in RULES {
                for mode in MODES {
                    let seeds: &[u64] = if mode == Mode::Chronological {
                        &SEARCH_SEEDS[..1]
                    } else {
                        &SEARCH_SEEDS
                    };
                    let mut result = DayResult::default();
                    for seed in seeds {
                        let w = walk(
                            &window,
                            start,
                            rule,
                            mode,
                            separation,
                            *seed,
                            &search,
                            &mut truths,
                        )?;
                        add(&mut result.day, &w);
                        result.seeds += 1;
                    }
                    out.entry((mode, rule)).or_default().push(result);
                }
            }
        }
    }
    Ok(out)
}

fn no_candidate_share(results: &Results, mode: Mode) -> f64 {
    let mut part = 0;
    let mut whole = 0;
    for ((m, rule), days) in results {
        if *m != mode || *rule == Rule::Hold {
            continue;
        }
        for d in days {
            part += d.day.no_candidate;
            whole += d.day.steps;
        }
    }
    share(part, whole)
}

fn report(split: &str, separation: usize, results: &Results) {
    println!("\n== (a) synthetic, {split}, separation {separation} s ==");
    println!(
        "mode | rule | steps | jumps | cross-day | mean jump (s) | reuse | no candidate | unforceable \
         | |move error| (ticks) [90 %] | next state wrong [90 %] | landing move - control | \
         take markout mode / truth | join move mode / truth"
    );
    for ((mode, rule), days) in results {
        let mut total = Walk::default();
        for d in days {
            add(&mut total, &d.day);
        }
        let per_day =
            |f: &dyn Fn(&Walk) -> f64| -> Vec<f64> { days.iter().map(|d| f(&d.day)).collect() };
        let err = interval(
            &per_day(&|w| {
                if w.compared == 0 {
                    0.0
                } else {
                    w.move_error / w.compared as f64
                }
            }),
            BOOTSTRAP_SEED,
        );
        let wrong = interval(
            &per_day(&|w| share(w.state_mismatch, w.compared)),
            BOOTSTRAP_SEED,
        );
        let jumps_within = total.jumps - total.cross_day;
        println!(
            "{} | {:?} | {} | {:.4} | {:.4} | {:.1} | {:.4} | {:.4} | {:.4} | {:.3} [{:.3}, {:.3}] | {:.4} [{:.4}, {:.4}] | {:.3} | {:.3} / {:.3} | {:.3} / {:.3}",
            mode.label(),
            rule,
            total.steps,
            share(total.jumps, total.steps),
            share(total.cross_day, total.jumps),
            if jumps_within == 0 {
                0.0
            } else {
                total.jump_steps as f64 / jumps_within as f64 * GRID_MS as f64 / 1000.0
            },
            share(total.reused, total.jumps),
            share(total.no_candidate, total.steps),
            share(total.unforceable, total.steps),
            err.0,
            err.1,
            err.2,
            wrong.0,
            wrong.1,
            wrong.2,
            if total.jumps == 0 {
                0.0
            } else {
                (total.landing_move - total.landing_control) / total.jumps as f64
            },
            if total.takes == 0 {
                0.0
            } else {
                total.take_markout_mode / total.takes as f64
            },
            if total.takes == 0 {
                0.0
            } else {
                total.take_markout_truth / total.takes as f64
            },
            if total.joins == 0 {
                0.0
            } else {
                total.join_move_mode / total.joins as f64
            },
            if total.joins == 0 {
                0.0
            } else {
                total.join_move_truth / total.joins as f64
            },
        );
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, Failure> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is not text").into())
}

fn float(value: &Value, key: &str) -> Result<f64, Failure> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{key} is not a number").into())
}

/// A regime's parameters from #102's design file, as its `qr_days` reads them.
fn parameters(regime: &Value) -> Result<QrParameters, Failure> {
    let sizes = regime
        .get("event_sizes")
        .and_then(Value::as_array)
        .ok_or("event_sizes")?
        .iter()
        .map(|size| Ok(size.as_str().ok_or("event size")?.parse::<Decimal>()?))
        .collect::<Result<Vec<_>, Failure>>()?;
    let rates = regime
        .get("rates")
        .and_then(Value::as_array)
        .ok_or("rates")?
        .iter()
        .map(|table| {
            table
                .as_array()
                .ok_or("rate table")?
                .iter()
                .map(|row| {
                    let row = row.as_array().ok_or("rate row")?;
                    let at = |i: usize| row.get(i).and_then(Value::as_f64).ok_or("rate");
                    Ok((at(0)?, at(1)?, at(2)?))
                })
                .collect::<Result<Vec<_>, Failure>>()
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    let supported = rates.iter().map(|table| vec![true; table.len()]).collect();
    let policy = match text(regime, "policy")? {
        "stop" => UnsupportedPolicy::Stop,
        "nearest_supported" => UnsupportedPolicy::NearestSupported,
        other => return Err(format!("unknown policy {other}").into()),
    };
    Ok(QrParameters::new(
        text(regime, "tick_size")?.parse::<Decimal>()?,
        float(regime, "theta")?,
        float(regime, "theta_reinit")?,
        sizes,
        rates,
        supported,
        policy,
    )?)
}

fn synthetic(design_path: &str) -> Result<(), Failure> {
    let design: Value = serde_json::from_slice(&fs::read(design_path)?)?;
    let simulator = design.get("simulator").ok_or("simulator")?;
    let reference = simulator
        .get("reference_ticks")
        .and_then(Value::as_i64)
        .ok_or("reference_ticks")?;
    let regimes = simulator
        .get("regimes")
        .and_then(Value::as_array)
        .ok_or("regimes")?;
    let steps = usize::try_from(SYNTHETIC_HOURS * 3_600_000 / GRID_MS)?;
    let mut params = Vec::new();
    let mut days = Vec::new();
    for (index, regime) in regimes.iter().enumerate() {
        let p = parameters(regime)?;
        let number = u64::try_from(index)? + 1;
        for day in 1..=SPLITS[2].2 {
            let seed = SEED_STRIDE * number + day;
            let mut sim = QrSimulator::new(p.clone(), seed, reference)?;
            let states = grid(&mut sim, steps, 0.0)?;
            let samples = states.iter().map(|s| sample_of(s, p.limits())).collect();
            days.push(Day {
                regime: index,
                day,
                states,
                samples,
            });
        }
        params.push(p);
        eprintln!("simulated {}", text(regime, "name")?);
    }
    println!(
        "Synthetic: #102's queue-reactive regimes and day seeds, the first {SYNTHETIC_HOURS} h of each day, made-up parameters."
    );
    let train = run_split(&params, &days, "train", SEPARATIONS[1])?;
    report("train", SEPARATIONS[1], &train);
    // Validation chooses the separation by the smallest no-candidate share of time travel.
    let mut best: Option<(f64, usize, Results)> = None;
    for separation in SEPARATIONS {
        let results = run_split(&params, &days, "validation", separation)?;
        let share = no_candidate_share(&results, Mode::TimeTravelAny);
        println!("validation: separation {separation} s, no-candidate share {share:.4}");
        if best.as_ref().is_none_or(|b| share < b.0) {
            best = Some((share, separation, results));
        }
    }
    let (_, separation, validation) = best.ok_or("no separation")?;
    report("validation", separation, &validation);
    let holdout = run_split(&params, &days, "holdout", separation)?;
    report("holdout", separation, &holdout);
    Ok(())
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("synthetic") => synthetic(&args.next().ok_or("#102's design file")?),
        Some("capture") => {
            let _ = (CAPTURE_SPLIT, MAX_NO_CANDIDATE_SHARE);
            Err("part (b) reads #263's book samples and lands with it".into())
        }
        _ => Err("usage: synthetic <design.json> | capture <dir>".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> QrParameters {
        // #102's calm table: nothing leaves an empty queue and nothing joins one at the cap.
        let table = vec![
            (4.0, 0.0, 0.0),
            (4.0, 1.0, 1.0),
            (3.0, 2.0, 1.0),
            (2.0, 3.0, 1.0),
            (0.0, 4.0, 1.0),
        ];
        QrParameters::new(
            Decimal::new(1, 2),
            0.6,
            0.2,
            vec![Decimal::ONE, Decimal::ONE],
            vec![table.clone(), table],
            vec![vec![true; 5], vec![true; 5]],
            UnsupportedPolicy::NearestSupported,
        )
        .unwrap()
    }

    #[test]
    fn test_queues_fall_in_their_buckets_and_the_spread_caps() {
        assert_eq!(
            (
                bucket(0),
                bucket(1),
                bucket(2),
                bucket(3),
                bucket(4),
                bucket(5),
                bucket(9),
                bucket(10)
            ),
            (0, 0, 1, 2, 3, 4, 4, 5)
        );
        let s = Sample {
            spread: 7,
            bid: 3,
            ask: 12,
            mid: 0,
        };
        assert_eq!(
            s.state(),
            State {
                spread: 3,
                bid: 2,
                ask: 5
            }
        );
    }

    #[test]
    fn test_a_hold_continued_from_the_tape_is_the_tape() {
        let p = small();
        let mut sim = QrSimulator::new(p.clone(), 7, 1_000).unwrap();
        let tape = grid(&mut sim, 200, 0.0).unwrap();
        for k in [0usize, 17, 100] {
            let mut fork = QrSimulator::from_state(p.clone(), tape[k]).unwrap();
            let again = grid(&mut fork, HORIZON_STEPS + 1, k as f64).unwrap();
            for (i, state) in again.iter().enumerate() {
                assert_eq!(
                    sample_of(state, 2),
                    sample_of(&tape[k + i], 2),
                    "step {k} + {i}"
                );
            }
        }
    }

    #[test]
    fn test_joining_adds_and_taking_removes_at_the_touch_and_not_beyond() {
        let p = small();
        let sim = QrSimulator::new(p.clone(), 3, 1_000).unwrap();
        let mut state = sim.state();
        state.bid = [0, 4, 0, 0, 0];
        state.ask = [2, 0, 0, 0, 0];
        let joined = force(&state, Action::JoinBoth, 2, 100).unwrap();
        assert_eq!((joined.bid[1], joined.ask[0]), (5, 3));
        let taken = force(&state, Action::TakeAsk, 2, 100).unwrap();
        assert_eq!(taken.ask[0], 1);
        // A queue at the cap cannot be joined, and an empty side cannot be taken.
        assert!(force(&state, Action::JoinBid, 2, 4).is_none());
        state.ask = [0; MAX_LIMITS];
        assert!(force(&state, Action::TakeAsk, 2, 100).is_none());
        assert_eq!(force(&state, Action::Hold, 2, 100), Some(state));
    }

    #[test]
    fn test_the_rules_act_on_what_they_see() {
        let s = |bid, ask| Sample {
            spread: 1,
            bid,
            ask,
            mid: 0,
        };
        assert_eq!(Rule::JoinSmaller.act(&s(3, 2)), Action::JoinAsk);
        assert_eq!(Rule::JoinSmaller.act(&s(2, 2)), Action::JoinBid);
        assert_eq!(Rule::TakeImbalance.act(&s(9, 1)), Action::TakeAsk);
        assert_eq!(Rule::TakeImbalance.act(&s(1, 9)), Action::TakeBid);
        assert_eq!(Rule::TakeImbalance.act(&s(6, 4)), Action::Hold);
        assert_eq!(Rule::TakeImbalance.act(&s(0, 0)), Action::Hold);
    }

    fn tape_day(regime: usize, day: u64, p: &QrParameters, seed: u64) -> Day {
        let mut sim = QrSimulator::new(p.clone(), seed, 1_000).unwrap();
        let states = grid(&mut sim, 400, 0.0).unwrap();
        let samples = states.iter().map(|s| sample_of(s, 2)).collect();
        Day {
            regime,
            day,
            states,
            samples,
        }
    }

    #[test]
    fn test_a_landing_keeps_its_separation_its_split_and_forward_time() {
        let p = small();
        let days = [tape_day(0, 1, &p, 1), tape_day(0, 2, &p, 2)];
        let window: Vec<&Day> = days.iter().collect();
        let search = Search::new(&window);
        let mut rng = Rng(5);
        for (key, times) in &search.by_transition {
            for now in [(0usize, 50usize), (1, 300)] {
                for _ in 0..5 {
                    if let Some(l) = search.land(*key, now, 30, true, &mut rng) {
                        assert!(times.contains(&l));
                        assert!(l > now);
                        assert!(l.0 != now.0 || l.1.abs_diff(now.1) >= 30);
                    }
                }
            }
        }
    }

    #[test]
    fn test_holding_never_jumps_and_matches_its_truth() {
        let p = small();
        let days = [tape_day(0, 1, &p, 9)];
        let window: Vec<&Day> = days.iter().collect();
        let search = Search::new(&window);
        let mut truths = Truths {
            params: &p,
            days: &window,
            limits: 2,
            cap: u32::try_from(p.max_queue()).unwrap(),
            cache: HashMap::new(),
        };
        for mode in MODES {
            let w = walk(&window, 0, Rule::Hold, mode, 10, 11, &search, &mut truths).unwrap();
            assert_eq!((w.jumps, w.no_candidate, w.state_mismatch), (0, 0, 0));
            assert_eq!(w.move_error, 0.0);
            assert!(w.steps > 300);
        }
    }

    #[test]
    fn test_a_join_in_chronological_replay_differs_from_its_truth_somewhere() {
        // The market reacts to the order in the truth and not in the replay.
        let p = small();
        let days = [tape_day(0, 1, &p, 13)];
        let window: Vec<&Day> = days.iter().collect();
        let search = Search::new(&window);
        let mut truths = Truths {
            params: &p,
            days: &window,
            limits: 2,
            cap: u32::try_from(p.max_queue()).unwrap(),
            cache: HashMap::new(),
        };
        let w = walk(
            &window,
            0,
            Rule::JoinBoth,
            Mode::Chronological,
            10,
            11,
            &search,
            &mut truths,
        )
        .unwrap();
        assert!(w.compared > 0);
        assert!(w.state_mismatch > 0);
        assert_eq!(w.jumps, 0);
    }
}
