//! Chronological replay against consistent time travel, on a known environment.
//!
//! Everything here is synthetic, and it decides nothing about any market. Time
//! travel (Ragel and Challet, <https://arxiv.org/abs/2408.02322v2>) is a separate
//! mode here, labelled in each row, and it is not causal evidence. No policy is
//! trained: fixed rules are read under each mode, against values computed exactly.
//!
//! # Modes
//!
//! - Chronological replay, the baseline: the tape goes on as it happened, so an
//!   action has no effect on the market.
//! - Time travel: an action forces the next state of the system. If the tape's
//!   next state differs, the replay jumps to a time `t'` with the same state and
//!   the forced next state, by equations 1 and 2 of the paper, and goes on from
//!   `t' + 1`. Candidates are drawn uniformly, leaving out those too close to the
//!   current time. As in the paper, if there is no candidate the replay carries
//!   on in order, which is inconsistent, and here it is counted.
//!
//! # Limits on the search
//!
//! A replay only reads the window it was given, so it cannot cross from test
//! into train or back. Inside it, a jump may go to the past: consistency breaks
//! global causality, as the paper says. [`Ordering::ForwardOnly`] keeps time
//! moving forward.
//!
//! A rule sees only its position and the state at the time the replay has
//! reached, never a time beyond that. That does not keep a window's later times
//! out of what it sees: the search reads them to choose a landing, a landing can
//! be later on the tape, and the state there carries the hidden regime.
//!
//! # Environment
//!
//! A system state moves by at most one level per step, and sets the drift of the
//! mid. In the informed case, a hidden persistent regime tilts both the moves of
//! the state and the drift, so a natural move carries information. An action
//! moves the state just the same but carries none, as the rule knows nothing of
//! the regime. The search lands on natural moves, so it selects their regime,
//! which then persists. A control with the same drift and no tilt has no such
//! selection. These results are for this design only.
//!
//! Run with: `cargo run --example consistent_time_travel`

use std::collections::{BTreeMap, BTreeSet};

/// Levels of the system state.
const STATES: usize = 5;
/// Level with no drift.
const CENTER: i32 = 2;
/// Mid drift per level from the center, in ticks per step.
const DRIFT_PER_LEVEL: f64 = 0.2;
/// Standard deviation of the mid's noise, in ticks per step.
const NOISE: f64 = 0.5;
/// Probability of each natural move up or down, with no tilt.
const MOVE: f64 = 0.25;
/// Change in that probability from the regime, in the informed case.
const TILT: f64 = 0.2;
/// Mid drift from the regime, in ticks per step, in the informed case.
const REGIME_DRIFT: f64 = 0.3;
/// Probability the regime stays at each step.
const REGIME_STAYS: f64 = 0.98;
/// Cost per level of an action, in ticks. Every arm pays it.
const ACTION_COST: f64 = 0.1;
/// Steps of an episode.
const HORIZON: usize = 20;

/// Steps of a tape.
const TAPE_STEPS: usize = 200_000;
/// First step of the test window; train comes before it.
const TEST_START: usize = 100_000;
/// Episodes read in a window.
const EPISODES: usize = 1_000;
/// A candidate closer than this to the current time is left out.
const MIN_SEPARATION: usize = 10;
/// Planned seeds, all reported.
const SEEDS: [u64; 5] = [11, 23, 37, 53, 71];
/// Bootstrap replicates.
const REPLICATES: usize = 200;
/// Episodes per bootstrap block.
const BLOCK_EPISODES: usize = 20;

/// A small seeded generator, so that a run reproduces without a dependency.
struct Rng(u64);

impl Rng {
    /// SplitMix64.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize
    }

    /// A standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.unit()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// The world a tape comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
struct World {
    name: &'static str,
    /// Change in the probability of a natural move, from the regime.
    tilt: f64,
    /// Mid drift from the regime, in ticks per step.
    regime_drift: f64,
}

/// The state holds all that matters for what comes next.
const MARKOV: World = World {
    name: "Markov",
    tilt: 0.0,
    regime_drift: 0.0,
};
/// A hidden regime tilts the natural moves and the drift.
const INFORMED: World = World {
    name: "Informed",
    tilt: TILT,
    regime_drift: REGIME_DRIFT,
};

impl World {
    /// Probabilities of a natural move up and down, in a regime of +1 or -1.
    fn move_probabilities(self, regime: i32) -> (f64, f64) {
        let tilt = self.tilt * f64::from(regime);
        (MOVE + tilt, MOVE - tilt)
    }

    /// Expected change of the mid over a step.
    fn drift(self, state: i32, regime: i32) -> f64 {
        DRIFT_PER_LEVEL * f64::from(state - CENTER) + self.regime_drift * f64::from(regime)
    }
}

fn clamp_state(state: i32) -> i32 {
    state.clamp(0, STATES as i32 - 1)
}

/// A history of the system with no agent in it.
#[derive(Debug, Clone)]
struct Tape {
    states: Vec<i32>,
    /// Mid at each time, in ticks.
    mids: Vec<f64>,
    /// Hidden regime at each time. No replay reads it to decide anything: it is
    /// `[truth]`, known only because the environment is synthetic.
    regimes: Vec<i32>,
}

impl Tape {
    fn generate(world: World, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut state = CENTER;
        let mut regime = if rng.unit() < 0.5 { 1 } else { -1 };
        let mut mid = 1_000.0;
        let mut tape = Tape {
            states: Vec::with_capacity(TAPE_STEPS + 1),
            mids: Vec::with_capacity(TAPE_STEPS + 1),
            regimes: Vec::with_capacity(TAPE_STEPS + 1),
        };
        for _ in 0..=TAPE_STEPS {
            tape.states.push(state);
            tape.mids.push(mid);
            tape.regimes.push(regime);
            mid += world.drift(state, regime) + NOISE * rng.normal();
            let (up, down) = world.move_probabilities(regime);
            let draw = rng.unit();
            if draw < up {
                state = clamp_state(state + 1);
            } else if draw < up + down {
                state = clamp_state(state - 1);
            }
            if rng.unit() >= REGIME_STAYS {
                regime = -regime;
            }
        }
        tape
    }

    /// FNV-1a 64, for the manifest. Integrity, not security.
    fn hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for ((state, mid), regime) in self.states.iter().zip(&self.mids).zip(&self.regimes) {
            feed(&state.to_le_bytes());
            feed(&mid.to_bits().to_le_bytes());
            feed(&regime.to_le_bytes());
        }
        hash
    }
}

/// A half open range of tape times. A replay reads nothing outside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    start: usize,
    end: usize,
}

const TRAIN: Window = Window {
    start: 0,
    end: TEST_START,
};
const TEST: Window = Window {
    start: TEST_START,
    end: TAPE_STEPS + 1,
};

/// Which way a jump may go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ordering {
    /// Anywhere in the window, including the past.
    AnyInWindow,
    /// Only later than the current time.
    ForwardOnly,
}

/// How a tape is replayed. A row of a report carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
    Chronological,
    TimeTravelAnyInWindow,
    TimeTravelForwardOnly,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Chronological => "chronological replay",
            Mode::TimeTravelAnyInWindow => "TIME TRAVEL, any in window, not causal",
            Mode::TimeTravelForwardOnly => "TIME TRAVEL, forward only, not causal",
        }
    }

    fn ordering(self) -> Option<Ordering> {
        match self {
            Mode::Chronological => None,
            Mode::TimeTravelAnyInWindow => Some(Ordering::AnyInWindow),
            Mode::TimeTravelForwardOnly => Some(Ordering::ForwardOnly),
        }
    }
}

const MODES: [Mode; 3] = [
    Mode::Chronological,
    Mode::TimeTravelAnyInWindow,
    Mode::TimeTravelForwardOnly,
];

/// Times of each transition of the state, inside one window.
struct Search {
    window: Window,
    by_transition: BTreeMap<(i32, i32), Vec<usize>>,
}

impl Search {
    fn new(tape: &Tape, window: Window) -> Self {
        let mut by_transition: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
        for time in window.start..window.end.saturating_sub(1) {
            by_transition
                .entry((tape.states[time], tape.states[time + 1]))
                .or_default()
                .push(time);
        }
        Search {
            window,
            by_transition,
        }
    }

    /// Draws a time with this transition, by the paper's equations 1 and 2.
    fn jump(
        &self,
        from: i32,
        to: i32,
        now: usize,
        ordering: Ordering,
        rng: &mut Rng,
    ) -> Option<usize> {
        let times = self.by_transition.get(&(from, to))?;
        let candidates: Vec<usize> = times
            .iter()
            .copied()
            .filter(|time| time.abs_diff(now) >= MIN_SEPARATION)
            .filter(|time| ordering == Ordering::AnyInWindow || *time > now)
            .collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[rng.below(candidates.len())])
        }
    }
}

/// What a rule can see. The tape's regime and later times are not in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Observation {
    state: i32,
    position: i32,
}

/// A fixed rule. The action is the levels it moves the state, and it trades one
/// unit in that direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// Never acts: a control, whose value is 0 by construction.
    Hold,
    /// Buys above the center, sells below.
    Follow,
    /// Sells above the center, buys below.
    Fade,
    /// As `Fade`, with an action larger than any natural move of the state, so
    /// the tape has no time that is consistent with it.
    FadeLarge,
}

const RULES: [Rule; 4] = [Rule::Hold, Rule::Follow, Rule::Fade, Rule::FadeLarge];

impl Rule {
    fn action(self, observation: Observation) -> i32 {
        let offset = observation.state - CENTER;
        let (want, size) = match self {
            Rule::Hold => (0, 0),
            Rule::Follow => (offset.signum(), 1),
            Rule::Fade => (-offset.signum(), 1),
            Rule::FadeLarge => (-offset.signum(), 2),
        };
        if want == 0 || (observation.position + want).abs() > 1 {
            0
        } else {
            want * size
        }
    }
}

/// The exact value of `rule`, by backward induction over the state, the regime
/// and the position, from the tape's stationary state with no position.
fn true_value(world: World, rule: Rule, horizon: usize) -> f64 {
    let position_index = |position: i32| (position + 1) as usize;
    let regime_index = |regime: i32| usize::from(regime > 0);
    let mut value = [[[0.0f64; 3]; 2]; STATES];
    for _ in 0..horizon {
        let mut earlier = [[[0.0f64; 3]; 2]; STATES];
        for state in 0..STATES as i32 {
            for regime in [-1, 1] {
                for position in -1..=1 {
                    let action = rule.action(Observation { state, position });
                    let held = position + action.signum();
                    let reward = f64::from(held) * world.drift(state, regime)
                        - ACTION_COST * f64::from(action.abs());
                    let mut future = 0.0;
                    for (next_state, p_state) in next_states(world, state, regime, action) {
                        for (next_regime, p_regime) in
                            [(regime, REGIME_STAYS), (-regime, 1.0 - REGIME_STAYS)]
                        {
                            future += p_state
                                * p_regime
                                * value[next_state as usize][regime_index(next_regime)]
                                    [position_index(held)];
                        }
                    }
                    earlier[state as usize][regime_index(regime)][position_index(position)] =
                        reward + future;
                }
            }
        }
        value = earlier;
    }
    let stationary = stationary_distribution(world);
    let mut total = 0.0;
    for state in 0..STATES {
        for regime in 0..2 {
            total += stationary[state][regime] * value[state][regime][position_index(0)];
        }
    }
    total
}

/// The next states of the true system, with their probabilities. An action
/// forces the move, and so tells nothing of the regime.
fn next_states(world: World, state: i32, regime: i32, action: i32) -> Vec<(i32, f64)> {
    if action != 0 {
        return vec![(clamp_state(state + action), 1.0)];
    }
    let (up, down) = world.move_probabilities(regime);
    vec![
        (clamp_state(state + 1), up),
        (clamp_state(state - 1), down),
        (state, 1.0 - up - down),
    ]
}

/// The stationary law of the state and regime with no agent, by iteration.
fn stationary_distribution(world: World) -> [[f64; 2]; STATES] {
    let mut law = [[1.0 / (2.0 * STATES as f64); 2]; STATES];
    for _ in 0..5_000 {
        let mut next = [[0.0; 2]; STATES];
        for state in 0..STATES as i32 {
            for regime in [-1, 1] {
                let mass = law[state as usize][usize::from(regime > 0)];
                for (next_state, p_state) in next_states(world, state, regime, 0) {
                    for (next_regime, p_regime) in
                        [(regime, REGIME_STAYS), (-regime, 1.0 - REGIME_STAYS)]
                    {
                        next[next_state as usize][usize::from(next_regime > 0)] +=
                            mass * p_state * p_regime;
                    }
                }
            }
        }
        law = next;
    }
    law
}

/// One episode of a replay, with its diagnostics.
#[derive(Debug, Clone, PartialEq)]
struct Episode {
    mode: Mode,
    /// Sum of the rewards, in ticks: `[estimated]`, under the mode's assumptions.
    value: f64,
    steps: usize,
    /// Times read, in order. Each is inside the window.
    visited: Vec<usize>,
    /// What the rule saw at each decision.
    observations: Vec<Observation>,
    /// Steps where an action moved the state away from the tape's next state.
    discrepancies: usize,
    jumps: usize,
    /// Size of each jump, in steps of the tape.
    jump_distances: Vec<usize>,
    /// Change of the mid's level at each jump, in ticks.
    jump_mid_gaps: Vec<f64>,
    /// Discrepancies with no consistent time, where the replay carried on.
    unsupported: usize,
    /// Levels between the forced state and the tape's, summed over those steps.
    state_mismatch: u64,
    /// Jumps landing on a time whose regime agrees with the action: `[truth]`.
    jumps_with_regime: usize,
    /// The episode ran out of tape in its window before its horizon.
    truncated: bool,
}

/// Replays one episode of `rule` from `start`. It reads only `search.window`.
fn replay(
    tape: &Tape,
    search: &Search,
    start: usize,
    rule: Rule,
    mode: Mode,
    rng: &mut Rng,
) -> Episode {
    let mut episode = Episode {
        mode,
        value: 0.0,
        steps: 0,
        visited: Vec::with_capacity(HORIZON),
        observations: Vec::with_capacity(HORIZON),
        discrepancies: 0,
        jumps: 0,
        jump_distances: Vec::new(),
        jump_mid_gaps: Vec::new(),
        unsupported: 0,
        state_mismatch: 0,
        jumps_with_regime: 0,
        truncated: false,
    };
    let mut now = start;
    let mut position = 0;
    for _ in 0..HORIZON {
        if now + 1 >= search.window.end {
            episode.truncated = true;
            break;
        }
        let state = tape.states[now];
        let observation = Observation { state, position };
        let action = rule.action(observation);
        episode.observations.push(observation);
        // The time whose move of the mid this step earns.
        let mut time = now;
        if action != 0 {
            let forced = clamp_state(state + action);
            if tape.states[now + 1] != forced {
                episode.discrepancies += 1;
                if let Some(ordering) = mode.ordering() {
                    match search.jump(state, forced, now, ordering, rng) {
                        Some(found) => {
                            episode.jumps += 1;
                            episode.jump_distances.push(found.abs_diff(now));
                            episode
                                .jump_mid_gaps
                                .push(tape.mids[found] - tape.mids[now]);
                            if tape.regimes[found] == action.signum() {
                                episode.jumps_with_regime += 1;
                            }
                            time = found;
                        }
                        None => {
                            episode.unsupported += 1;
                            episode.state_mismatch +=
                                u64::from(tape.states[now + 1].abs_diff(forced));
                        }
                    }
                }
            }
        }
        position += action.signum();
        episode.visited.push(time);
        episode.value += f64::from(position) * (tape.mids[time + 1] - tape.mids[time])
            - ACTION_COST * f64::from(action.abs());
        episode.steps += 1;
        now = time + 1;
    }
    episode
}

/// Evenly spaced starts of episodes inside a window, in order.
fn starts(window: Window) -> Vec<usize> {
    let span = window.end - window.start - HORIZON - 1;
    (0..EPISODES)
        .map(|i| window.start + i * span / EPISODES)
        .collect()
}

/// Replays every episode of a window under one mode.
fn run_window(tape: &Tape, window: Window, rule: Rule, mode: Mode, seed: u64) -> Vec<Episode> {
    let search = Search::new(tape, window);
    let mut rng = Rng(seed);
    starts(window)
        .into_iter()
        .map(|start| replay(tape, &search, start, rule, mode, &mut rng))
        .collect()
}

/// A row of a report: one rule, one mode.
#[derive(Debug, Clone, PartialEq)]
struct Summary {
    mode: Mode,
    mean: f64,
    low: f64,
    high: f64,
    /// Share of the steps with a discrepancy.
    discrepancy_share: f64,
    /// Share of the steps that jumped.
    jump_share: f64,
    /// Share of the discrepancies with no consistent time.
    unsupported_share: f64,
    /// Mean levels of mismatch where a discrepancy had no consistent time.
    mean_state_mismatch: f64,
    /// Mean size of a jump, in steps of the tape.
    mean_jump_distance: f64,
    /// Mean absolute change of the mid's level at a jump, in ticks.
    mean_jump_mid_gap: f64,
    /// Share of the times read more than once.
    reused_share: f64,
    /// The most times one tape time was read.
    max_reuse: usize,
    /// Share of the jumps landing on a regime that agrees with the action.
    jumps_with_regime_share: f64,
    /// Share of the episodes that ran out of tape.
    truncated_share: f64,
}

/// Summarizes episodes of one mode. `None` if they are empty or mix modes, as a
/// time travel result must never be read as a chronological one.
fn summarize(episodes: &[Episode], seed: u64) -> Option<Summary> {
    let mode = episodes.first()?.mode;
    if episodes.iter().any(|episode| episode.mode != mode) {
        return None;
    }
    let values: Vec<f64> = episodes.iter().map(|episode| episode.value).collect();
    let mean = mean_of(&values);
    let (low, high) = block_interval(&values, seed);
    let total = |field: fn(&Episode) -> usize| episodes.iter().map(field).sum::<usize>();
    let steps = total(|e| e.steps);
    let discrepancies = total(|e| e.discrepancies);
    let jumps = total(|e| e.jumps);
    let unsupported = total(|e| e.unsupported);
    let mismatch: u64 = episodes.iter().map(|e| e.state_mismatch).sum();
    let distances: Vec<f64> = episodes
        .iter()
        .flat_map(|e| e.jump_distances.iter().map(|d| *d as f64))
        .collect();
    let gaps: Vec<f64> = episodes
        .iter()
        .flat_map(|e| e.jump_mid_gaps.iter().map(|g| g.abs()))
        .collect();
    let mut reads: BTreeMap<usize, usize> = BTreeMap::new();
    for time in episodes.iter().flat_map(|e| &e.visited) {
        *reads.entry(*time).or_default() += 1;
    }
    let reused = reads.values().filter(|count| **count > 1).count();
    Some(Summary {
        mode,
        mean,
        low,
        high,
        discrepancy_share: share(discrepancies, steps),
        jump_share: share(jumps, steps),
        unsupported_share: share(unsupported, discrepancies),
        mean_state_mismatch: if unsupported == 0 {
            0.0
        } else {
            mismatch as f64 / unsupported as f64
        },
        mean_jump_distance: mean_of(&distances),
        mean_jump_mid_gap: mean_of(&gaps),
        reused_share: share(reused, reads.len()),
        max_reuse: reads.values().copied().max().unwrap_or(0),
        jumps_with_regime_share: share(total(|e| e.jumps_with_regime), jumps),
        truncated_share: share(
            episodes.iter().filter(|e| e.truncated).count(),
            episodes.len(),
        ),
    })
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// A 95% interval of the mean, by a moving block bootstrap over episodes.
fn block_interval(values: &[f64], seed: u64) -> (f64, f64) {
    if values.len() < BLOCK_EPISODES {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng(seed);
    let last_start = values.len() - BLOCK_EPISODES + 1;
    let mut means = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let mut sum = 0.0;
        let mut count = 0;
        while count < values.len() {
            let start = rng.below(last_start);
            for value in values[start..start + BLOCK_EPISODES]
                .iter()
                .take(values.len() - count)
            {
                sum += value;
                count += 1;
            }
        }
        means.push(sum / count as f64);
    }
    means.sort_by(f64::total_cmp);
    let at = |quantile: f64| means[((REPLICATES - 1) as f64 * quantile).round() as usize];
    (at(0.025), at(0.975))
}

/// The rule with the best mean under `mode` on train. It never reads test.
fn select_on_train(tape: &Tape, mode: Mode, seed: u64) -> Rule {
    let mut best = (Rule::Hold, f64::NEG_INFINITY);
    for rule in RULES {
        let episodes = run_window(tape, TRAIN, rule, mode, seed);
        let values: Vec<f64> = episodes.iter().map(|e| e.value).collect();
        let mean = mean_of(&values);
        if mean > best.1 {
            best = (rule, mean);
        }
    }
    best.0
}

/// The rule with the best exact value.
fn best_true_rule(world: World) -> Rule {
    RULES
        .into_iter()
        .max_by(|a, b| true_value(world, *a, HORIZON).total_cmp(&true_value(world, *b, HORIZON)))
        .unwrap_or(Rule::Hold)
}

fn main() {
    println!("SYNTHETIC ONLY: a known environment, with no market data and no trained policy.");
    println!(
        "Config: {STATES} states, horizon {HORIZON}, tape of {TAPE_STEPS} steps, test from \
         {TEST_START}, {EPISODES} episodes per window, min separation {MIN_SEPARATION}, seeds \
         {SEEDS:?}, {REPLICATES} replicates, blocks of {BLOCK_EPISODES}, action cost \
         {ACTION_COST}."
    );
    for world in [MARKOV, INFORMED] {
        println!("\n===== {} world, test window =====", world.name);
        for rule in RULES {
            println!(
                "\n{rule:?}: [truth] {:.3}",
                true_value(world, rule, HORIZON)
            );
            for seed in SEEDS {
                let tape = Tape::generate(world, seed);
                println!("  seed {seed}, tape hash {:016x}", tape.hash());
                for mode in MODES {
                    let episodes = run_window(&tape, TEST, rule, mode, seed);
                    if let Some(s) = summarize(&episodes, seed) {
                        println!(
                            "    [{}] [estimated] {:.3} [{:.3}, {:.3}]; discrepancy {:.3}, \
                             jump {:.3}, unsupported {:.3}, mismatch {:.2} levels, jump distance \
                             {:.0} steps, mid gap {:.2}, times reused {:.3}, max reuse {}, \
                             truncated {:.3}, [truth] jumps with the regime {:.3}",
                            s.mode.label(),
                            s.mean,
                            s.low,
                            s.high,
                            s.discrepancy_share,
                            s.jump_share,
                            s.unsupported_share,
                            s.mean_state_mismatch,
                            s.mean_jump_distance,
                            s.mean_jump_mid_gap,
                            s.reused_share,
                            s.max_reuse,
                            s.truncated_share,
                            s.jumps_with_regime_share,
                        );
                    }
                }
            }
        }
        println!(
            "\nRule selected on train, by mode; best by [truth] {:?}:",
            best_true_rule(world)
        );
        for seed in SEEDS {
            let tape = Tape::generate(world, seed);
            let selected: BTreeSet<(Mode, String)> = MODES
                .into_iter()
                .map(|mode| (mode, format!("{:?}", select_on_train(&tape, mode, seed))))
                .collect();
            let row: Vec<String> = selected
                .iter()
                .map(|(mode, rule)| format!("{}: {rule}", mode.label()))
                .collect();
            println!("  seed {seed}: {}", row.join("; "));
        }
    }
    println!(
        "\nReading, for this synthetic design only: in order, an action has no effect, so the \
         baseline is biased where actions move the market. Time travel came close to the truth \
         where the state holds all that matters, and overstated a rule where a natural move \
         carries information, as a search on the state cannot tell an action from an informed \
         move. Use: keep chronological replay as the reference, for what it assumes and not as \
         an unbiased value, and read time travel only as a labelled research mode, with its \
         discontinuities, reuse and unsupported actions. No real data was read, and neither \
         mode shows causality."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    const FIRST_SEED: u64 = SEEDS[0];

    fn tape_of(world: World) -> &'static Tape {
        static MARKOV_TAPE: OnceLock<Tape> = OnceLock::new();
        static INFORMED_TAPE: OnceLock<Tape> = OnceLock::new();
        if world == MARKOV {
            MARKOV_TAPE.get_or_init(|| Tape::generate(MARKOV, FIRST_SEED))
        } else {
            INFORMED_TAPE.get_or_init(|| Tape::generate(INFORMED, FIRST_SEED))
        }
    }

    /// A tape from its states, with a mid of 100 times each index and regime 1.
    fn hand_tape(states: &[i32]) -> Tape {
        Tape {
            states: states.to_vec(),
            mids: (0..states.len()).map(|i| 100.0 * i as f64).collect(),
            regimes: vec![1; states.len()],
        }
    }

    fn whole(tape: &Tape) -> Window {
        Window {
            start: 0,
            end: tape.states.len(),
        }
    }

    fn mean_value(episodes: &[Episode]) -> f64 {
        let values: Vec<f64> = episodes.iter().map(|e| e.value).collect();
        mean_of(&values)
    }

    #[test]
    fn test_true_value_one_step_matches_hand_computation() {
        // Uniform stationary law. Follow acts off center and earns drift less cost:
        // 0.1 one level away and 0.3 two levels away, on each side.
        let expected = (0.3 + 0.1 + 0.0 + 0.1 + 0.3) / 5.0;
        let got = true_value(MARKOV, Rule::Follow, 1);
        assert!((got - expected).abs() < 1e-12, "{got}");
        // Fade holds the other way round, so it loses the drift and the cost.
        let expected = (-0.5 - 0.3 + 0.0 - 0.3 - 0.5) / 5.0;
        let got = true_value(MARKOV, Rule::Fade, 1);
        assert!((got - expected).abs() < 1e-12, "{got}");
    }

    #[test]
    fn test_true_value_of_never_acting_is_zero() {
        for world in [MARKOV, INFORMED] {
            assert_eq!(true_value(world, Rule::Hold, HORIZON), 0.0);
        }
    }

    #[test]
    fn test_true_value_counts_the_effect_of_an_action_on_later_drift() {
        // From state 3 Follow buys, which forces state 4, where drift is larger.
        // A second step is worth more than without that push: at 4 the drift is
        // 0.4, and with no push the expected next drift is that of the natural
        // law, 0.2 here, as moves up and down balance.
        let world = MARKOV;
        let one = true_value(world, Rule::Follow, 1);
        let two = true_value(world, Rule::Follow, 2);
        assert!(two > 2.0 * one);
    }

    #[test]
    fn test_stationary_law_matches_the_tape() {
        for world in [MARKOV, INFORMED] {
            let law = stationary_distribution(world);
            let total: f64 = law.iter().flatten().sum();
            assert!((total - 1.0).abs() < 1e-9);
            let tape = tape_of(world);
            for (state, row) in law.iter().enumerate() {
                for (expected, regime) in row.iter().zip([-1, 1]) {
                    let observed = tape
                        .states
                        .iter()
                        .zip(&tape.regimes)
                        .filter(|(s, r)| **s as usize == state && **r == regime)
                        .count();
                    let share = observed as f64 / tape.states.len() as f64;
                    assert!(
                        (share - expected).abs() < 0.03,
                        "{} {state} {regime} {share} {expected}",
                        world.name
                    );
                }
            }
        }
    }

    /// Plays a rule in the true system, with an action forcing the next state.
    fn simulate_true_system(world: World, rule: Rule, episodes: usize, seed: u64) -> (f64, f64) {
        let mut rng = Rng(seed);
        let law = stationary_distribution(world);
        let (mut sum, mut squares) = (0.0, 0.0);
        for _ in 0..episodes {
            // Draws the start from the stationary law.
            let mut draw = rng.unit();
            let (mut state, mut regime) = (0, 1);
            'start: for (s, row) in law.iter().enumerate() {
                for (r, mass) in row.iter().enumerate() {
                    if draw < *mass {
                        state = s as i32;
                        regime = if r == 0 { -1 } else { 1 };
                        break 'start;
                    }
                    draw -= mass;
                }
            }
            let mut position = 0;
            let mut value = 0.0;
            for _ in 0..HORIZON {
                let action = rule.action(Observation { state, position });
                position += action.signum();
                value += f64::from(position) * (world.drift(state, regime) + NOISE * rng.normal())
                    - ACTION_COST * f64::from(action.abs());
                let mut draw = rng.unit();
                for (next, probability) in next_states(world, state, regime, action) {
                    if draw < probability {
                        state = next;
                        break;
                    }
                    draw -= probability;
                }
                if rng.unit() >= REGIME_STAYS {
                    regime = -regime;
                }
            }
            sum += value;
            squares += value * value;
        }
        let mean = sum / episodes as f64;
        let error = ((squares / episodes as f64 - mean * mean) / episodes as f64).sqrt();
        (mean, error)
    }

    #[test]
    fn test_true_value_matches_a_simulation_of_the_true_system() {
        for world in [MARKOV, INFORMED] {
            for rule in [Rule::Follow, Rule::Fade, Rule::FadeLarge] {
                let truth = true_value(world, rule, HORIZON);
                let (mean, error) = simulate_true_system(world, rule, 100_000, 99);
                assert!(
                    (mean - truth).abs() < 4.0 * error,
                    "{} {rule:?} {mean} {truth} {error}",
                    world.name
                );
            }
        }
    }

    #[test]
    fn test_rule_actions_and_their_position_limit() {
        let at = |state, position| Observation { state, position };
        assert_eq!(Rule::Hold.action(at(4, 0)), 0);
        assert_eq!(Rule::Follow.action(at(3, 0)), 1);
        assert_eq!(Rule::Follow.action(at(1, 0)), -1);
        assert_eq!(Rule::Follow.action(at(CENTER, 0)), 0);
        assert_eq!(Rule::Follow.action(at(3, 1)), 0);
        assert_eq!(Rule::Follow.action(at(1, -1)), 0);
        assert_eq!(Rule::Follow.action(at(1, 1)), -1);
        assert_eq!(Rule::Fade.action(at(3, 0)), -1);
        assert_eq!(Rule::FadeLarge.action(at(3, 0)), -2);
        assert_eq!(Rule::FadeLarge.action(at(1, 0)), 2);
    }

    #[test]
    fn test_action_forces_the_next_state_and_natural_moves_follow_the_regime() {
        assert_eq!(next_states(INFORMED, 2, 1, 1), vec![(3, 1.0)]);
        assert_eq!(next_states(INFORMED, 4, 1, 2), vec![(4, 1.0)]);
        let natural = next_states(INFORMED, 2, 1, 0);
        assert_eq!(natural[0].0, 3);
        assert!((natural[0].1 - (MOVE + TILT)).abs() < 1e-12);
        assert!((natural[1].1 - (MOVE - TILT)).abs() < 1e-12);
        let against = next_states(INFORMED, 2, -1, 0);
        assert!((against[0].1 - (MOVE - TILT)).abs() < 1e-12);
        assert_eq!(MARKOV.move_probabilities(-1), (MOVE, MOVE));
        assert_eq!(MARKOV.drift(3, -1), DRIFT_PER_LEVEL);
        assert!((INFORMED.drift(3, -1) - (DRIFT_PER_LEVEL - REGIME_DRIFT)).abs() < 1e-12);
    }

    #[test]
    fn test_time_travel_recovers_the_truth_where_the_state_is_all_that_matters() {
        for seed in SEEDS {
            let tape = Tape::generate(MARKOV, seed);
            for rule in [Rule::Follow, Rule::Fade] {
                let truth = true_value(MARKOV, rule, HORIZON);
                let travel = mean_value(&run_window(
                    &tape,
                    TEST,
                    rule,
                    Mode::TimeTravelAnyInWindow,
                    seed,
                ));
                let chronological =
                    mean_value(&run_window(&tape, TEST, rule, Mode::Chronological, seed));
                assert!(
                    (travel - truth).abs() < 0.3,
                    "{seed} {rule:?} {travel} {truth}"
                );
                // In order, the action has no effect, and the replay is biased.
                assert!(
                    (chronological - truth).abs() > 0.4,
                    "{seed} {rule:?} {chronological} {truth}"
                );
            }
        }
    }

    #[test]
    fn test_time_travel_mistakes_an_action_for_an_informed_move() {
        // The stated negative result, for this design: it flatters this rule far more
        // than replay in order misses it, on every seed.
        let truth = true_value(INFORMED, Rule::Fade, HORIZON);
        for seed in SEEDS {
            let tape = Tape::generate(INFORMED, seed);
            let travel = run_window(&tape, TEST, Rule::Fade, Mode::TimeTravelAnyInWindow, seed);
            let chronological = run_window(&tape, TEST, Rule::Fade, Mode::Chronological, seed);
            let travel_error = mean_value(&travel) - truth;
            let chronological_error = (mean_value(&chronological) - truth).abs();
            assert!(travel_error > 2.0, "{seed} {travel_error}");
            assert!(travel_error > 2.0 * chronological_error, "{seed}");
        }
    }

    #[test]
    fn test_the_forced_move_selects_a_regime_that_the_state_alone_does_not() {
        // The mechanism itself. Times with state 3 are the landings by equation 1
        // alone, and those with the move 3 to 2 add equation 2, for a forced sale.
        // The second set lands far more often where the regime agrees with the sale.
        let tape = tape_of(INFORMED);
        let agrees = |times: &[usize]| {
            let count = times.iter().filter(|t| tape.regimes[**t] == -1).count();
            count as f64 / times.len() as f64
        };
        let same_state: Vec<usize> = (TEST.start..TEST.end - 1)
            .filter(|t| tape.states[*t] == 3)
            .collect();
        let same_move: Vec<usize> = same_state
            .iter()
            .copied()
            .filter(|t| tape.states[t + 1] == 2)
            .collect();
        assert!(
            agrees(&same_move) > agrees(&same_state) + 0.4,
            "{} {}",
            agrees(&same_move),
            agrees(&same_state)
        );
        // Without the tilt the move says nothing of the regime.
        let control = World {
            name: "control",
            tilt: 0.0,
            regime_drift: REGIME_DRIFT,
        };
        let tape = Tape::generate(control, FIRST_SEED);
        let agrees = |times: &[usize]| {
            let count = times.iter().filter(|t| tape.regimes[**t] == -1).count();
            count as f64 / times.len() as f64
        };
        let same_state: Vec<usize> = (TEST.start..TEST.end - 1)
            .filter(|t| tape.states[*t] == 3)
            .collect();
        let same_move: Vec<usize> = same_state
            .iter()
            .copied()
            .filter(|t| tape.states[t + 1] == 2)
            .collect();
        assert!((agrees(&same_move) - agrees(&same_state)).abs() < 0.05);
    }

    #[test]
    fn test_with_no_tilt_time_travel_loses_its_overstatement() {
        // The same drift from the regime and no tilt: the jumps select nothing, and
        // the error is a fraction of the informed world's, on every seed.
        let control = World {
            name: "control",
            tilt: 0.0,
            regime_drift: REGIME_DRIFT,
        };
        let truth = true_value(control, Rule::Fade, HORIZON);
        for seed in SEEDS {
            let tape = Tape::generate(control, seed);
            let travel = run_window(&tape, TEST, Rule::Fade, Mode::TimeTravelAnyInWindow, seed);
            let error = (mean_value(&travel) - truth).abs();
            assert!(error < 0.7, "{seed} {error} {truth}");
        }
    }

    #[test]
    fn test_reindexing_never_leaves_the_window() {
        for world in [MARKOV, INFORMED] {
            let tape = tape_of(world);
            for window in [TRAIN, TEST] {
                for rule in RULES {
                    for mode in MODES {
                        for episode in run_window(tape, window, rule, mode, FIRST_SEED) {
                            assert!(!episode.visited.is_empty());
                            for time in &episode.visited {
                                assert!(*time >= window.start, "{mode:?} {time}");
                                assert!(time + 1 < window.end, "{mode:?} {time}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_train_and_test_windows_share_no_time_and_touch() {
        assert_eq!(TRAIN.end, TEST.start);
        assert_eq!(TRAIN.start, 0);
        assert_eq!(TEST.end, TAPE_STEPS + 1);
        let train = starts(TRAIN);
        let test = starts(TEST);
        assert_eq!(train.len(), EPISODES);
        assert!(train.iter().all(|s| s + HORIZON < TRAIN.end));
        assert!(
            test.iter()
                .all(|s| *s >= TEST.start && s + HORIZON < TEST.end)
        );
        assert!(train.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn test_a_replay_of_one_window_does_not_read_the_other() {
        // Rewrite everything before the test window: states, mids and regimes.
        let tape = tape_of(MARKOV);
        let base = run_window(
            tape,
            TEST,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        let mut changed = tape.clone();
        for time in 0..TEST.start {
            changed.states[time] = 0;
            changed.mids[time] = -1.0;
            changed.regimes[time] = -1;
        }
        assert_eq!(
            run_window(
                &changed,
                TEST,
                Rule::Fade,
                Mode::TimeTravelAnyInWindow,
                FIRST_SEED
            ),
            base
        );
        // And the other way round.
        let base = run_window(
            tape,
            TRAIN,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        let mut changed = tape.clone();
        for time in TEST.start..tape.states.len() {
            changed.states[time] = 0;
            changed.mids[time] = -1.0;
            changed.regimes[time] = -1;
        }
        assert_eq!(
            run_window(
                &changed,
                TRAIN,
                Rule::Fade,
                Mode::TimeTravelAnyInWindow,
                FIRST_SEED
            ),
            base
        );
    }

    #[test]
    fn test_search_ignores_a_transition_across_the_window_edge() {
        // Times 0 to 3 are in the window. The move 2 to 3 is only at its edge.
        let tape = hand_tape(&[1, 1, 1, 2, 3, 3]);
        let window = Window { start: 0, end: 4 };
        let search = Search::new(&tape, window);
        let mut rng = Rng(1);
        assert_eq!(
            search.jump(2, 3, 100, Ordering::AnyInWindow, &mut rng),
            None
        );
        assert_eq!(
            search.jump(1, 2, 100, Ordering::AnyInWindow, &mut rng),
            Some(2)
        );
        // The window can also start past a transition.
        let late = Search::new(&tape, Window { start: 3, end: 6 });
        assert_eq!(late.jump(1, 2, 100, Ordering::AnyInWindow, &mut rng), None);
        assert_eq!(
            late.jump(2, 3, 100, Ordering::AnyInWindow, &mut rng),
            Some(3)
        );
    }

    #[test]
    fn test_search_leaves_out_times_too_close_and_respects_the_ordering() {
        let mut states = vec![0; 60];
        states[20] = 1;
        states[21] = 2;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let mut rng = Rng(1);
        let jump = |now, ordering: Ordering, rng: &mut Rng| search.jump(1, 2, now, ordering, rng);
        assert_eq!(
            jump(20 + MIN_SEPARATION - 1, Ordering::AnyInWindow, &mut rng),
            None
        );
        assert_eq!(
            jump(20 - MIN_SEPARATION + 1, Ordering::AnyInWindow, &mut rng),
            None
        );
        assert_eq!(
            jump(20 + MIN_SEPARATION, Ordering::AnyInWindow, &mut rng),
            Some(20)
        );
        assert_eq!(
            jump(20 - MIN_SEPARATION, Ordering::AnyInWindow, &mut rng),
            Some(20)
        );
        assert_eq!(
            jump(20 + MIN_SEPARATION, Ordering::ForwardOnly, &mut rng),
            None
        );
        assert_eq!(
            jump(20 - MIN_SEPARATION, Ordering::ForwardOnly, &mut rng),
            Some(20)
        );
    }

    #[test]
    fn test_jump_lands_on_a_consistent_time_and_earns_its_move() {
        // Follow at state 3 forces state 4. The tape goes 3 to 3 at time 0, and it
        // has 3 to 4 at time 30 only.
        let mut states = vec![CENTER; 40];
        states[0] = 3;
        states[1] = 3;
        states[30] = 3;
        states[31] = 4;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            0,
            Rule::Follow,
            Mode::TimeTravelAnyInWindow,
            &mut Rng(1),
        );
        assert_eq!(episode.discrepancies, 1);
        assert_eq!(episode.jumps, 1);
        assert_eq!(episode.jump_distances, vec![30]);
        assert_eq!(episode.jump_mid_gaps, vec![3_000.0]);
        assert_eq!(episode.visited[0], 30);
        assert_eq!(episode.visited[1], 31);
        assert_eq!(episode.observations[1].state, 4);
        assert_eq!(episode.unsupported, 0);
        assert_eq!(episode.jumps_with_regime, 1);
        assert_eq!(episode.mode, Mode::TimeTravelAnyInWindow);
        // It earns the mid's moves from where it landed, in steps of 100 ticks at a
        // position of 1, less the action's cost, until the tape ends at time 39.
        assert_eq!(episode.visited.last(), Some(&38));
        assert!(episode.truncated);
        let expected = 100.0 * episode.steps as f64 - ACTION_COST;
        assert!((episode.value - expected).abs() < 1e-9, "{}", episode.value);
    }

    #[test]
    fn test_step_earns_the_move_of_the_time_it_landed_on() {
        // From time 5, Follow at state 3 must land on time 30, whose move of the mid
        // is the only large one. A later position of 1 earns all moves after it.
        let mut states = vec![CENTER; 40];
        states[5] = 3;
        states[6] = 3;
        states[30] = 3;
        states[31] = 4;
        let mut tape = hand_tape(&states);
        tape.mids = vec![0.0; 40];
        for mid in &mut tape.mids[31..] {
            *mid = 1_000.0;
        }
        tape.mids[6] = 7.0;
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            5,
            Rule::Follow,
            Mode::TimeTravelAnyInWindow,
            &mut Rng(1),
        );
        assert_eq!(episode.jump_distances, vec![25]);
        assert_eq!(episode.jump_mid_gaps, vec![0.0]);
        assert_eq!(episode.visited[0], 30);
        assert!(
            (episode.value - (1_000.0 - ACTION_COST)).abs() < 1e-9,
            "{}",
            episode.value
        );
    }

    #[test]
    fn test_jump_mid_gap_is_the_change_of_level_between_the_two_times() {
        let mut states = vec![CENTER; 40];
        states[5] = 3;
        states[6] = 3;
        states[30] = 3;
        states[31] = 4;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            5,
            Rule::Follow,
            Mode::TimeTravelAnyInWindow,
            &mut Rng(1),
        );
        assert_eq!(episode.jump_mid_gaps, vec![2_500.0]);
    }

    #[test]
    fn test_replay_in_order_ignores_the_action_and_counts_the_discrepancy() {
        let mut states = vec![CENTER; 40];
        states[0] = 3;
        states[1] = 3;
        states[30] = 3;
        states[31] = 4;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            0,
            Rule::Follow,
            Mode::Chronological,
            &mut Rng(1),
        );
        assert_eq!(episode.discrepancies, 1);
        assert_eq!(episode.jumps, 0);
        assert_eq!(episode.unsupported, 0);
        let expected: Vec<usize> = (0..HORIZON).collect();
        assert_eq!(episode.visited, expected);
    }

    #[test]
    fn test_no_jump_when_the_tape_already_goes_where_the_action_does() {
        let mut states = vec![CENTER; 40];
        states[0] = 3;
        states[1] = 4;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            0,
            Rule::Follow,
            Mode::TimeTravelAnyInWindow,
            &mut Rng(1),
        );
        assert_eq!(episode.discrepancies, 0);
        assert_eq!(episode.jumps, 0);
        assert_eq!(episode.visited[..2], [0, 1]);
    }

    #[test]
    fn test_every_jump_in_a_run_reaches_the_forced_state() {
        let tape = tape_of(INFORMED);
        let search = Search::new(tape, TEST);
        let mut rng = Rng(3);
        for start in starts(TEST).into_iter().take(200) {
            let episode = replay(
                tape,
                &search,
                start,
                Rule::Fade,
                Mode::TimeTravelAnyInWindow,
                &mut rng,
            );
            for (index, time) in episode.visited.iter().enumerate() {
                let observation = episode.observations[index];
                // The time earned has the state the rule saw.
                assert_eq!(tape.states[*time], observation.state);
                // And what the rule sees next is that time's next state, whose move
                // agrees with the action whenever a jump was possible.
                let action = Rule::Fade.action(observation);
                if action != 0 && episode.unsupported == 0 {
                    assert_eq!(
                        tape.states[time + 1],
                        clamp_state(observation.state + action)
                    );
                }
                if let Some(next) = episode.observations.get(index + 1) {
                    assert_eq!(next.state, tape.states[time + 1]);
                }
            }
        }
    }

    #[test]
    fn test_unsupported_action_is_counted_and_goes_on_in_order() {
        let tape = tape_of(MARKOV);
        let travel = run_window(
            tape,
            TEST,
            Rule::FadeLarge,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        let chronological =
            run_window(tape, TEST, Rule::FadeLarge, Mode::Chronological, FIRST_SEED);
        let summary = summarize(&travel, FIRST_SEED).unwrap();
        assert_eq!(summary.jump_share, 0.0);
        assert!(summary.discrepancy_share > 0.0);
        assert_eq!(summary.unsupported_share, 1.0);
        assert!(summary.mean_state_mismatch >= 1.0);
        // So it reads the tape exactly as replay in order does, and says so.
        for (a, b) in travel.iter().zip(&chronological) {
            assert_eq!(a.visited, b.visited);
            assert_eq!(a.value, b.value);
        }
        // And its interval leaves out the truth, which is why it is counted.
        let truth = true_value(MARKOV, Rule::FadeLarge, HORIZON);
        assert!(
            truth < summary.low || truth > summary.high,
            "{summary:?} {truth}"
        );
    }

    #[test]
    fn test_mismatch_measures_levels_from_the_forced_state() {
        // FadeLarge at state 3 forces state 1, and the tape stays at 3: two levels.
        let mut states = vec![CENTER; 40];
        states[0] = 3;
        states[1] = 3;
        let tape = hand_tape(&states);
        let search = Search::new(&tape, whole(&tape));
        let episode = replay(
            &tape,
            &search,
            0,
            Rule::FadeLarge,
            Mode::TimeTravelAnyInWindow,
            &mut Rng(1),
        );
        assert_eq!(episode.unsupported, 1);
        assert_eq!(episode.state_mismatch, 2);
        assert_eq!(episode.jumps, 0);
    }

    #[test]
    fn test_forward_only_never_goes_back_and_any_in_window_does() {
        let tape = tape_of(MARKOV);
        let forward = run_window(
            tape,
            TEST,
            Rule::Fade,
            Mode::TimeTravelForwardOnly,
            FIRST_SEED,
        );
        for episode in &forward {
            assert!(episode.visited.windows(2).all(|pair| pair[0] < pair[1]));
        }
        let any = run_window(
            tape,
            TEST,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        assert!(
            any.iter()
                .any(|episode| episode.visited.windows(2).any(|pair| pair[1] < pair[0]))
        );
    }

    #[test]
    fn test_forward_only_runs_out_of_tape_and_says_so() {
        let tape = tape_of(MARKOV);
        let forward = summarize(
            &run_window(
                tape,
                TEST,
                Rule::Fade,
                Mode::TimeTravelForwardOnly,
                FIRST_SEED,
            ),
            FIRST_SEED,
        )
        .unwrap();
        let any = summarize(
            &run_window(
                tape,
                TEST,
                Rule::Fade,
                Mode::TimeTravelAnyInWindow,
                FIRST_SEED,
            ),
            FIRST_SEED,
        )
        .unwrap();
        assert!(
            forward.truncated_share > any.truncated_share,
            "{forward:?} {any:?}"
        );
        assert!(forward.max_reuse > any.max_reuse);
    }

    #[test]
    fn test_episode_near_the_window_end_is_truncated_inside_it() {
        let tape = hand_tape(&[CENTER; 10]);
        let window = Window { start: 0, end: 5 };
        let search = Search::new(&tape, window);
        let episode = replay(
            &tape,
            &search,
            0,
            Rule::Hold,
            Mode::Chronological,
            &mut Rng(1),
        );
        assert!(episode.truncated);
        assert_eq!(episode.visited, vec![0, 1, 2, 3]);
        assert_eq!(episode.steps, 4);
        // A window long enough for the horizon is not truncated.
        let long = hand_tape(&[CENTER; HORIZON + 1]);
        let search = Search::new(&long, whole(&long));
        let episode = replay(
            &long,
            &search,
            0,
            Rule::Hold,
            Mode::Chronological,
            &mut Rng(1),
        );
        assert!(!episode.truncated);
        assert_eq!(episode.steps, HORIZON);
    }

    #[test]
    fn test_the_rule_reads_no_mid_and_no_regime() {
        // Decisions rest on the state and the position alone: rewriting the mids and
        // the regimes leaves every time read and every observation as it was.
        let tape = tape_of(INFORMED);
        let base = run_window(
            tape,
            TEST,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        let mut changed = tape.clone();
        for mid in &mut changed.mids {
            *mid = -*mid;
        }
        for regime in &mut changed.regimes {
            *regime = -*regime;
        }
        let moved = run_window(
            &changed,
            TEST,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        for (a, b) in base.iter().zip(&moved) {
            assert_eq!(a.visited, b.visited);
            assert_eq!(a.observations, b.observations);
        }
    }

    #[test]
    fn test_replay_in_order_reads_nothing_after_its_episode() {
        // No future state reaches a rule: without a search, a tape rewritten after
        // an episode leaves that episode as it was.
        let tape = tape_of(MARKOV);
        let start = TEST.start;
        let search = Search::new(tape, TEST);
        let base = replay(
            tape,
            &search,
            start,
            Rule::Fade,
            Mode::Chronological,
            &mut Rng(1),
        );
        let mut changed = tape.clone();
        for time in start + HORIZON + 1..changed.states.len() {
            changed.states[time] = 0;
            changed.mids[time] = 0.0;
        }
        let search = Search::new(&changed, TEST);
        assert_eq!(
            replay(
                &changed,
                &search,
                start,
                Rule::Fade,
                Mode::Chronological,
                &mut Rng(1)
            ),
            base
        );
    }

    #[test]
    fn test_never_acting_reads_the_tape_in_order_in_every_mode() {
        let tape = tape_of(INFORMED);
        for mode in MODES {
            for episode in run_window(tape, TEST, Rule::Hold, mode, FIRST_SEED) {
                assert_eq!(episode.value, 0.0);
                assert_eq!(episode.discrepancies, 0);
                assert!(
                    episode
                        .visited
                        .windows(2)
                        .all(|pair| pair[1] == pair[0] + 1)
                );
            }
        }
    }

    #[test]
    fn test_modes_are_labelled_and_never_mixed() {
        assert_eq!(Mode::Chronological.label(), "chronological replay");
        for mode in [Mode::TimeTravelAnyInWindow, Mode::TimeTravelForwardOnly] {
            assert!(mode.label().contains("TIME TRAVEL"));
            assert!(mode.label().contains("not causal"));
        }
        assert_eq!(Mode::Chronological.ordering(), None);
        assert_eq!(
            Mode::TimeTravelAnyInWindow.ordering(),
            Some(Ordering::AnyInWindow)
        );
        assert_eq!(
            Mode::TimeTravelForwardOnly.ordering(),
            Some(Ordering::ForwardOnly)
        );
        let tape = tape_of(MARKOV);
        let one = run_window(tape, TEST, Rule::Fade, Mode::Chronological, FIRST_SEED);
        let other = run_window(
            tape,
            TEST,
            Rule::Fade,
            Mode::TimeTravelAnyInWindow,
            FIRST_SEED,
        );
        assert_eq!(
            summarize(&one, FIRST_SEED).unwrap().mode,
            Mode::Chronological
        );
        let mixed: Vec<Episode> = one.iter().chain(&other).cloned().collect();
        assert_eq!(summarize(&mixed, FIRST_SEED), None);
        assert_eq!(summarize(&[], FIRST_SEED), None);
    }

    #[test]
    fn test_summary_matches_hand_counted_episodes() {
        let episode = |visited: Vec<usize>, value: f64| Episode {
            mode: Mode::TimeTravelAnyInWindow,
            value,
            steps: visited.len(),
            visited,
            observations: Vec::new(),
            discrepancies: 2,
            jumps: 1,
            jump_distances: vec![30],
            jump_mid_gaps: vec![-4.0],
            unsupported: 1,
            state_mismatch: 3,
            jumps_with_regime: 1,
            truncated: false,
        };
        let mut episodes = vec![episode(vec![1, 2, 3, 4], 1.0); 15];
        episodes.push(Episode {
            jumps_with_regime: 0,
            truncated: true,
            ..episode(vec![4, 5, 6, 7], 3.0)
        });
        let summary = summarize(&episodes, 1).unwrap();
        assert_eq!(summary.mean, (15.0 + 3.0) / 16.0);
        assert_eq!(summary.discrepancy_share, 32.0 / 64.0);
        assert_eq!(summary.jump_share, 16.0 / 64.0);
        assert_eq!(summary.unsupported_share, 16.0 / 32.0);
        assert_eq!(summary.mean_state_mismatch, 3.0);
        assert_eq!(summary.mean_jump_distance, 30.0);
        assert_eq!(summary.mean_jump_mid_gap, 4.0);
        // Times 1 to 4 and 4 to 7: all but 5, 6 and 7 read more than once.
        assert_eq!(summary.reused_share, 4.0 / 7.0);
        assert_eq!(summary.max_reuse, 16);
        assert_eq!(summary.jumps_with_regime_share, 15.0 / 16.0);
        assert_eq!(summary.truncated_share, 1.0 / 16.0);
    }

    #[test]
    fn test_runs_reproduce_by_seed() {
        let tape = Tape::generate(MARKOV, FIRST_SEED);
        assert_eq!(tape.hash(), tape_of(MARKOV).hash());
        assert_ne!(Tape::generate(MARKOV, SEEDS[1]).hash(), tape.hash());
        assert_ne!(Tape::generate(INFORMED, FIRST_SEED).hash(), tape.hash());
        for change in [0, 1, 2] {
            let mut other = tape.clone();
            match change {
                0 => other.states[9] += 1,
                1 => other.mids[9] += 1.0,
                _ => other.regimes[9] = -other.regimes[9],
            }
            assert_ne!(other.hash(), tape.hash(), "{change}");
        }
        let a = run_window(&tape, TEST, Rule::Fade, Mode::TimeTravelAnyInWindow, 5);
        let b = run_window(&tape, TEST, Rule::Fade, Mode::TimeTravelAnyInWindow, 5);
        assert_eq!(a, b);
        assert_eq!(summarize(&a, 5), summarize(&b, 5));
        // The seed of the search is what a second run under time travel changes.
        let c = run_window(&tape, TEST, Rule::Fade, Mode::TimeTravelAnyInWindow, 6);
        assert_ne!(a, c);
    }

    #[test]
    fn test_selection_reads_train_only_and_takes_the_best() {
        let tape = tape_of(MARKOV);
        for mode in MODES {
            let selected = select_on_train(tape, mode, FIRST_SEED);
            assert_eq!(selected, Rule::Follow, "{mode:?}");
            let mut changed = tape.clone();
            for time in TEST.start..changed.mids.len() {
                // Anything reading this would turn to a rule that buys at state 0.
                changed.mids[time] = 1e9 * time as f64;
                changed.states[time] = 0;
            }
            assert_eq!(
                select_on_train(&changed, mode, FIRST_SEED),
                selected,
                "{mode:?}"
            );
        }
        assert_eq!(best_true_rule(MARKOV), Rule::Follow);
    }

    #[test]
    fn test_block_interval_brackets_the_mean_and_needs_enough_values() {
        let values: Vec<f64> = (0..1_000).map(|i| f64::from(i % 7)).collect();
        let (low, high) = block_interval(&values, 3);
        let mean = mean_of(&values);
        assert!(low < mean && mean < high, "{low} {mean} {high}");
        assert!(high - low < 1.0);
        let (low, high) = block_interval(&values[..BLOCK_EPISODES - 1], 3);
        assert!(low.is_nan() && high.is_nan());
    }

    #[test]
    fn test_share_and_mean_of_nothing_are_zero() {
        assert_eq!(share(1, 0), 0.0);
        assert_eq!(share(1, 4), 0.25);
        assert_eq!(mean_of(&[]), 0.0);
        assert_eq!(mean_of(&[1.0, 2.0]), 1.5);
        assert_eq!(clamp_state(-1), 0);
        assert_eq!(clamp_state(STATES as i32), STATES as i32 - 1);
    }
}
