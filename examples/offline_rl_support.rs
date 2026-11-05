//! Support diagnostics for offline evaluation on a small known environment.
//!
//! Everything here is synthetic. It decides nothing about any market: it checks
//! which diagnostics can be trusted before any log is interpreted, on an
//! environment whose true policy values are computed exactly. No policy is
//! trained, and the comparison with interactive simulation stays pending.
//!
//! A log shows what one policy did. It does not show what another would have
//! got, so each value here says what it rests on:
//!
//! - `[observed]`: read from the log;
//! - `[estimated]`: an off policy estimate, which needs its stated assumption;
//! - `[truth]`: exact, and known only because the environment is synthetic.
//!
//! # Environment
//!
//! Each step quotes a half spread on both sides, for a bounded inventory. A
//! hidden regime, drawn afresh each step, sets the fill probabilities and the
//! adverse selection per fill. The reward, in ticks, is the edge of the fills
//! less a quadratic inventory cost. The log records the inventory, not the
//! regime. Each arm has the same costs and limits.
//!
//! # Estimators
//!
//! - Per decision importance sampling, self normalized at each step, needs the
//!   probability the behaviour policy gave each action. If the log does not
//!   record it, it is counted from the log, which assumes no hidden confounder.
//! - A tabular model of outcomes by inventory and action needs each cell the
//!   target reaches to be in the log, and also assumes no hidden confounder.
//!
//! # Protocol
//!
//! Episodes are split in chronological order, and the planned seeds are all
//! reported. The model, the support counts and any counted probability come
//! from train. A policy is selected on validation. Importance sampling and its
//! effective sample read holdout. Intervals come from a moving block bootstrap
//! over episodes, are statistical only, and hold counted probabilities fixed.
//!
//! # Decision
//!
//! [`decide`] takes the thresholds as constants, fixed before any run, and the
//! truth is never an input. A go says only that these checks found nothing
//! against reading the estimators for that target on that log. It does not say
//! that they are right, or that the policy is good. No reward can remove a
//! reason of censoring, unrecorded probabilities or missing support. Rewards do
//! reach it in one way: whether the two estimators' intervals overlap, which is
//! a coarse check. A wide interval overlaps a biased estimate, which happens here
//! on one target of one seed, and rewards that moved in holdout alone could hide
//! a disagreement.
//!
//! Run with: `cargo run --example offline_rl_support`

use std::collections::BTreeSet;

/// Steps in an episode.
const HORIZON: usize = 3;
/// Half spreads a policy can quote.
const ACTIONS: usize = 3;
/// Half spread of each action, in ticks.
const HALF_SPREAD: [f64; ACTIONS] = [1.0, 2.0, 3.0];
/// Largest absolute inventory; a fill past it is not possible.
const MAX_INVENTORY: i32 = 2;
/// Inventories from `-MAX_INVENTORY` to `MAX_INVENTORY`.
const STATES: usize = 5;
/// Probability the hidden regime is toxic at each step.
const TOXIC_PROBABILITY: f64 = 0.3;
/// Probability of a fill per side, by regime (calm, toxic) and action.
const FILL: [[f64; ACTIONS]; 2] = [[0.8, 0.5, 0.3], [0.9, 0.7, 0.5]];
/// Adverse selection per fill, in ticks, by regime.
const ADVERSE: [f64; 2] = [0.0, 2.0];
/// Inventory cost per step, in ticks per unit squared.
const INVENTORY_COST: f64 = 0.1;

/// Episodes in a log.
const EPISODES: usize = 8_000;
/// Episodes of the train split; they come first.
const TRAIN_EPISODES: usize = 4_000;
/// Episodes of the validation split; they follow train, and holdout follows.
const VALIDATION_EPISODES: usize = 2_000;
/// Planned seeds, all reported.
const SEEDS: [u64; 5] = [11, 23, 37, 53, 71];
/// Bootstrap replicates.
const REPLICATES: usize = 100;
/// Episodes per bootstrap block.
const BLOCK_EPISODES: usize = 25;

/// Fewest log entries a cell needs to count as supported.
const MIN_CELL_COUNT: u64 = 30;
/// Fewest effective trajectories at any step of importance sampling.
const MIN_EFFECTIVE_SAMPLE: f64 = 10.0;

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

    fn pick(&mut self, probabilities: &[f64; ACTIONS]) -> usize {
        let mut draw = self.unit();
        for (action, p) in probabilities.iter().enumerate() {
            if draw < *p {
                return action;
            }
            draw -= p;
        }
        ACTIONS - 1
    }
}

fn state_index(inventory: i32) -> usize {
    (inventory + MAX_INVENTORY) as usize
}

/// The policy that wrote a log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Each action equally likely, whatever the regime.
    Randomized,
    /// Never quotes the tightest half spread.
    Narrow,
    /// Sees the hidden regime: mostly tight when calm, mostly wide when toxic.
    RegimeAware,
}

impl Behaviour {
    fn probabilities(self, regime: usize) -> [f64; ACTIONS] {
        match (self, regime) {
            (Behaviour::Randomized, _) => [1.0 / 3.0; ACTIONS],
            (Behaviour::Narrow, _) => [0.0, 0.5, 0.5],
            (Behaviour::RegimeAware, 0) => [0.7, 0.2, 0.1],
            (Behaviour::RegimeAware, _) => [0.1, 0.2, 0.7],
        }
    }
}

/// What a log keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogShape {
    /// Every step, with the probability the behaviour gave its action.
    Full,
    /// Every step, without that probability.
    NoPropensity,
    /// Only the steps with a fill, without that probability, as a log of
    /// aggregated fills would be: a quote that did not fill leaves no trace.
    FillsOnly,
}

/// What happened to a step's quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Execution {
    bid_filled: bool,
    ask_filled: bool,
}

impl Execution {
    fn index(self) -> usize {
        usize::from(self.bid_filled) * 2 + usize::from(self.ask_filled)
    }
}

/// One step of a log.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Record {
    /// Position of the step in the episode.
    step: usize,
    /// Inventory before the step. The regime is not in it.
    observation: i32,
    action: usize,
    execution: Execution,
    /// Reward of the step, in ticks.
    reward: f64,
    /// Whether the episode ends with this step.
    terminal: bool,
    /// Probability the behaviour gave `action`, if the log records it.
    behaviour_probability: Option<f64>,
}

type Episode = Vec<Record>;

/// A log, in chronological order.
#[derive(Debug, Clone)]
struct Log {
    shape: LogShape,
    episodes: Vec<Episode>,
}

impl Log {
    fn train(&self) -> &[Episode] {
        &self.episodes[..TRAIN_EPISODES]
    }

    fn validation(&self) -> &[Episode] {
        &self.episodes[TRAIN_EPISODES..TRAIN_EPISODES + VALIDATION_EPISODES]
    }

    fn holdout(&self) -> &[Episode] {
        &self.episodes[TRAIN_EPISODES + VALIDATION_EPISODES..]
    }

    /// Share of the steps of the horizon that left no record.
    fn censored_share(&self) -> f64 {
        let recorded: usize = self.episodes.iter().map(Vec::len).sum();
        1.0 - recorded as f64 / (self.episodes.len() * HORIZON) as f64
    }

    /// FNV-1a 64 of the records, for the manifest. Integrity, not security.
    fn hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for record in self.episodes.iter().flatten() {
            feed(&record.step.to_le_bytes());
            feed(&record.observation.to_le_bytes());
            feed(&record.action.to_le_bytes());
            feed(&[record.execution.index() as u8, u8::from(record.terminal)]);
            feed(&record.reward.to_bits().to_le_bytes());
            let probability = record.behaviour_probability.map_or(u64::MAX, f64::to_bits);
            feed(&probability.to_le_bytes());
        }
        hash
    }
}

/// The inventory after an execution.
fn next_inventory(inventory: i32, execution: Execution) -> i32 {
    inventory + i32::from(execution.bid_filled) - i32::from(execution.ask_filled)
}

/// The reward of a step: the edge of its fills less the inventory cost.
fn step_reward(inventory: i32, regime: usize, action: usize, execution: Execution) -> f64 {
    let fills = f64::from(u8::from(execution.bid_filled) + u8::from(execution.ask_filled));
    let after = f64::from(next_inventory(inventory, execution));
    fills * (HALF_SPREAD[action] - ADVERSE[regime]) - INVENTORY_COST * after * after
}

/// Fill probabilities of a step's sides. A side past the limit cannot fill.
fn side_probabilities(inventory: i32, regime: usize, action: usize) -> (f64, f64) {
    let fill = FILL[regime][action];
    let bid = if inventory < MAX_INVENTORY { fill } else { 0.0 };
    let ask = if inventory > -MAX_INVENTORY {
        fill
    } else {
        0.0
    };
    (bid, ask)
}

/// Writes a log of `behaviour`, as `shape` keeps it.
fn generate_log(behaviour: Behaviour, shape: LogShape, seed: u64) -> Log {
    let mut rng = Rng(seed);
    let mut episodes = Vec::with_capacity(EPISODES);
    for _ in 0..EPISODES {
        let mut episode = Vec::with_capacity(HORIZON);
        let mut inventory = 0;
        for step in 0..HORIZON {
            let regime = usize::from(rng.unit() < TOXIC_PROBABILITY);
            let probabilities = behaviour.probabilities(regime);
            let action = rng.pick(&probabilities);
            let (bid, ask) = side_probabilities(inventory, regime, action);
            let execution = Execution {
                bid_filled: rng.unit() < bid,
                ask_filled: rng.unit() < ask,
            };
            let filled = execution.bid_filled || execution.ask_filled;
            if shape != LogShape::FillsOnly || filled {
                episode.push(Record {
                    step,
                    observation: inventory,
                    action,
                    execution,
                    reward: step_reward(inventory, regime, action, execution),
                    terminal: step + 1 == HORIZON,
                    behaviour_probability: (shape == LogShape::Full)
                        .then_some(probabilities[action]),
                });
            }
            inventory = next_inventory(inventory, execution);
        }
        episodes.push(episode);
    }
    Log { shape, episodes }
}

/// The exact value of always quoting `target`, from a flat inventory, by
/// backward induction over the whole state, regime included.
fn true_value(target: usize, horizon: usize) -> f64 {
    let mut value = [0.0; STATES];
    for _ in 0..horizon {
        let mut earlier = [0.0; STATES];
        for inventory in -MAX_INVENTORY..=MAX_INVENTORY {
            let mut total = 0.0;
            for (regime, regime_probability) in
                [(0, 1.0 - TOXIC_PROBABILITY), (1, TOXIC_PROBABILITY)]
            {
                let (bid, ask) = side_probabilities(inventory, regime, target);
                for (bid_filled, p_bid) in [(true, bid), (false, 1.0 - bid)] {
                    for (ask_filled, p_ask) in [(true, ask), (false, 1.0 - ask)] {
                        let execution = Execution {
                            bid_filled,
                            ask_filled,
                        };
                        let after = next_inventory(inventory, execution);
                        let probability = regime_probability * p_bid * p_ask;
                        if probability > 0.0 {
                            total += probability
                                * (step_reward(inventory, regime, target, execution)
                                    + value[state_index(after)]);
                        }
                    }
                }
            }
            earlier[state_index(inventory)] = total;
        }
        value = earlier;
    }
    value[state_index(0)]
}

/// Log entries by inventory and action.
fn cell_counts(episodes: &[Episode]) -> [[u64; ACTIONS]; STATES] {
    let mut counts = [[0; ACTIONS]; STATES];
    for record in episodes.iter().flatten() {
        counts[state_index(record.observation)][record.action] += 1;
    }
    counts
}

/// Cells `target` reaches that the log lacks: its action at each inventory.
fn out_of_support(counts: &[[u64; ACTIONS]; STATES], target: usize) -> Vec<(i32, usize)> {
    (-MAX_INVENTORY..=MAX_INVENTORY)
        .filter(|inventory| counts[state_index(*inventory)][target] < MIN_CELL_COUNT)
        .map(|inventory| (inventory, target))
        .collect()
}

/// Share of actions and of inventories with any supported cell.
fn coverage(counts: &[[u64; ACTIONS]; STATES]) -> (f64, f64) {
    let supported = |count: &u64| *count >= MIN_CELL_COUNT;
    let actions = (0..ACTIONS)
        .filter(|action| counts.iter().any(|row| supported(&row[*action])))
        .count();
    let states = counts
        .iter()
        .filter(|row| row.iter().any(supported))
        .count();
    (
        actions as f64 / ACTIONS as f64,
        states as f64 / STATES as f64,
    )
}

/// The value of `target` by a tabular model of outcomes. `None` when the log
/// lacks a cell the target reaches. It assumes no hidden confounder.
fn model_value(fit: &[&Episode], target: usize) -> Option<f64> {
    let mut counts = [[0u64; 4]; STATES];
    let mut rewards = [[0.0f64; 4]; STATES];
    for record in fit.iter().copied().flatten() {
        if record.action == target {
            let state = state_index(record.observation);
            counts[state][record.execution.index()] += 1;
            rewards[state][record.execution.index()] += record.reward;
        }
    }
    let mut value = [0.0; STATES];
    for _ in 0..HORIZON {
        let mut earlier = [0.0; STATES];
        for inventory in -MAX_INVENTORY..=MAX_INVENTORY {
            let state = state_index(inventory);
            let total: u64 = counts[state].iter().sum();
            if total == 0 {
                return None;
            }
            let mut sum = 0.0;
            for (bid_filled, ask_filled) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let execution = Execution {
                    bid_filled,
                    ask_filled,
                };
                let outcome = execution.index();
                if counts[state][outcome] > 0 {
                    let after = state_index(next_inventory(inventory, execution));
                    sum += rewards[state][outcome] + counts[state][outcome] as f64 * value[after];
                }
            }
            earlier[state] = sum / total as f64;
        }
        value = earlier;
    }
    Some(value[state_index(0)])
}

/// Result of importance sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Importance {
    value: f64,
    /// Smallest effective number of trajectories across the steps.
    min_effective: f64,
}

/// Per decision importance sampling of `target`, self normalized at each step.
/// A trajectory counts until it leaves the target. A log without recorded
/// probabilities uses `counted`. `None` when no trajectory is left at some step,
/// or a probability is zero for an action taken. Episodes are whole: [`assess`]
/// refuses a censored log, and a missing step here would read as left out.
fn importance(
    episodes: &[&Episode],
    target: usize,
    counted: &[[u64; ACTIONS]; STATES],
) -> Option<Importance> {
    let mut weights_by_step = [(0.0f64, 0.0f64, 0.0f64); HORIZON];
    for episode in episodes {
        let mut weight = 1.0;
        for record in episode.iter() {
            if record.action != target {
                break;
            }
            let probability = record.behaviour_probability.unwrap_or_else(|| {
                let row = &counted[state_index(record.observation)];
                row[target] as f64 / row.iter().sum::<u64>().max(1) as f64
            });
            if probability <= 0.0 {
                return None;
            }
            weight /= probability;
            let (sum, squares, weighted) = &mut weights_by_step[record.step];
            *sum += weight;
            *squares += weight * weight;
            *weighted += weight * record.reward;
        }
    }
    let mut value = 0.0;
    let mut min_effective = f64::INFINITY;
    for (sum, squares, weighted) in weights_by_step {
        if sum <= 0.0 {
            return None;
        }
        value += weighted / sum;
        min_effective = min_effective.min(sum * sum / squares);
    }
    Some(Importance {
        value,
        min_effective,
    })
}

/// A 95% interval by a moving block bootstrap over episodes. `None` if a
/// replicate has no value.
fn block_interval(
    episodes: &[Episode],
    seed: u64,
    statistic: impl Fn(&[&Episode]) -> Option<f64>,
) -> Option<(f64, f64)> {
    let mut rng = Rng(seed);
    let last_start = episodes.len().checked_sub(BLOCK_EPISODES)? + 1;
    let mut values = Vec::with_capacity(REPLICATES);
    let mut sample: Vec<&Episode> = Vec::with_capacity(episodes.len());
    for _ in 0..REPLICATES {
        sample.clear();
        while sample.len() < episodes.len() {
            let start = rng.below(last_start);
            sample.extend(&episodes[start..start + BLOCK_EPISODES]);
        }
        sample.truncate(episodes.len());
        values.push(statistic(&sample)?);
    }
    values.sort_by(f64::total_cmp);
    let at = |share: f64| values[((REPLICATES - 1) as f64 * share).round() as usize];
    Some((at(0.025), at(0.975)))
}

/// An estimate with its interval.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Estimate {
    value: f64,
    low: f64,
    high: f64,
}

impl Estimate {
    fn overlaps(self, other: Estimate) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    #[cfg(test)]
    fn contains(self, value: f64) -> bool {
        self.low <= value && value <= self.high
    }
}

/// What the decision reads. Only `estimators_overlap` depends on rewards.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Diagnostics {
    /// Some steps left no record, so the log is a selection of what happened.
    censored: bool,
    /// The log records the probability of each action.
    propensity_recorded: bool,
    /// Cells the target reaches that the log lacks.
    out_of_support_cells: usize,
    /// Effective trajectories, if importance sampling has a value.
    min_effective: Option<f64>,
    /// Whether the two estimators' intervals overlap, if both exist.
    estimators_overlap: Option<bool>,
}

/// Why a target cannot be evaluated on a log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reason {
    Censored,
    NoRecordedPropensity,
    OutOfSupport,
    LowEffectiveSample,
    EstimatorsDisagree,
    /// There is no second estimate to check the first against.
    NoCrossCheck,
}

/// The decision for one target on one log.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Decision {
    Go,
    NoGo(BTreeSet<Reason>),
}

/// Decides from diagnostics. What was not measured cannot pass.
fn decide(diagnostics: &Diagnostics) -> Decision {
    let mut reasons = BTreeSet::new();
    if diagnostics.censored {
        reasons.insert(Reason::Censored);
    }
    if !diagnostics.propensity_recorded {
        reasons.insert(Reason::NoRecordedPropensity);
    }
    if diagnostics.out_of_support_cells > 0 {
        reasons.insert(Reason::OutOfSupport);
    }
    if diagnostics
        .min_effective
        .is_some_and(|effective| effective < MIN_EFFECTIVE_SAMPLE)
    {
        reasons.insert(Reason::LowEffectiveSample);
    }
    if diagnostics.estimators_overlap == Some(false) {
        reasons.insert(Reason::EstimatorsDisagree);
    }
    // A sample nobody measured cannot pass, whatever else is in order.
    if diagnostics.min_effective.is_none() && reasons.is_empty() {
        reasons.insert(Reason::LowEffectiveSample);
    }
    if diagnostics.estimators_overlap.is_none() && reasons.is_empty() {
        reasons.insert(Reason::NoCrossCheck);
    }
    if reasons.is_empty() {
        Decision::Go
    } else {
        Decision::NoGo(reasons)
    }
}

/// The shape of a log of best quotes and aggregated fills, declared by hand and
/// not measured: it leaves out a quote that did not fill and the probabilities
/// of the actions, so importance sampling and the model have nothing to run on.
const MARKET_LOG: Diagnostics = Diagnostics {
    censored: true,
    propensity_recorded: false,
    out_of_support_cells: 0,
    min_effective: None,
    estimators_overlap: None,
};

/// A target assessed on a log.
#[derive(Debug, Clone, PartialEq)]
struct Assessment {
    target: usize,
    diagnostics: Diagnostics,
    /// Importance sampling on holdout, `[estimated]`.
    importance: Option<Estimate>,
    /// The model fitted on train, `[estimated]`.
    model: Option<Estimate>,
    decision: Decision,
}

/// Assesses `target` on `log`. A censored log or a target out of support gets
/// no estimate: nothing can be said from such a log without further assumptions.
fn assess(log: &Log, target: usize, seed: u64) -> Assessment {
    let censored = log.censored_share() > 0.0;
    let counts = cell_counts(log.train());
    let out_of_support_cells = out_of_support(&counts, target).len();
    let (importance_estimate, model_estimate, min_effective) =
        if censored || out_of_support_cells > 0 {
            (None, None, None)
        } else {
            estimates(log, target, seed, &counts)
        };
    let diagnostics = Diagnostics {
        censored,
        propensity_recorded: log.shape == LogShape::Full,
        out_of_support_cells,
        min_effective,
        estimators_overlap: importance_estimate
            .zip(model_estimate)
            .map(|(a, b)| a.overlaps(b)),
    };
    Assessment {
        target,
        diagnostics,
        importance: importance_estimate,
        model: model_estimate,
        decision: decide(&diagnostics),
    }
}

fn estimates(
    log: &Log,
    target: usize,
    seed: u64,
    counts: &[[u64; ACTIONS]; STATES],
) -> (Option<Estimate>, Option<Estimate>, Option<f64>) {
    let holdout: Vec<&Episode> = log.holdout().iter().collect();
    let train: Vec<&Episode> = log.train().iter().collect();
    let point = importance(&holdout, target, counts);
    let importance_estimate = point.and_then(|point| {
        block_interval(log.holdout(), seed, |sample| {
            importance(sample, target, counts).map(|i| i.value)
        })
        .map(|(low, high)| Estimate {
            value: point.value,
            low,
            high,
        })
    });
    let model_estimate = model_value(&train, target).and_then(|value| {
        block_interval(log.train(), seed, |sample| model_value(sample, target))
            .map(|(low, high)| Estimate { value, low, high })
    });
    (
        importance_estimate,
        model_estimate,
        point.map(|p| p.min_effective),
    )
}

/// The action with the best importance sampling value on validation, and that
/// value. It reads train only for counted probabilities, and never holdout.
fn select_on_validation(log: &Log) -> Option<(usize, f64)> {
    let counts = cell_counts(log.train());
    let validation: Vec<&Episode> = log.validation().iter().collect();
    (0..ACTIONS)
        .filter_map(|target| {
            importance(&validation, target, &counts).map(|estimate| (target, estimate.value))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

/// Mean reward per recorded step, `[observed]`.
fn mean_reward(log: &Log) -> f64 {
    let (sum, count) = log
        .episodes
        .iter()
        .flatten()
        .fold((0.0, 0usize), |(sum, count), record| {
            (sum + record.reward, count + 1)
        });
    sum / count as f64
}

/// A log the report reads.
struct Scenario {
    name: &'static str,
    behaviour: Behaviour,
    shape: LogShape,
}

const SCENARIOS: [Scenario; 5] = [
    Scenario {
        name: "randomized, probabilities recorded",
        behaviour: Behaviour::Randomized,
        shape: LogShape::Full,
    },
    Scenario {
        name: "narrow, never the tightest action",
        behaviour: Behaviour::Narrow,
        shape: LogShape::Full,
    },
    Scenario {
        name: "sees the hidden regime, probabilities recorded",
        behaviour: Behaviour::RegimeAware,
        shape: LogShape::Full,
    },
    Scenario {
        name: "sees the hidden regime, probabilities not recorded",
        behaviour: Behaviour::RegimeAware,
        shape: LogShape::NoPropensity,
    },
    Scenario {
        name: "sees the hidden regime, fills only",
        behaviour: Behaviour::RegimeAware,
        shape: LogShape::FillsOnly,
    },
];

fn format_estimate(estimate: Option<Estimate>) -> String {
    estimate.map_or("none".to_string(), |e| {
        format!("{:.3} [{:.3}, {:.3}]", e.value, e.low, e.high)
    })
}

fn format_decision(decision: &Decision) -> String {
    match decision {
        Decision::Go => "GO".to_string(),
        Decision::NoGo(reasons) => format!("NO-GO {reasons:?}"),
    }
}

fn main() {
    println!("SYNTHETIC ONLY: a known environment, with no market data and no trained policy.");
    println!(
        "Config: horizon {HORIZON}, actions {ACTIONS}, episodes {EPISODES} \
         ({TRAIN_EPISODES} train, {VALIDATION_EPISODES} validation, the rest holdout), \
         seeds {SEEDS:?}, {REPLICATES} replicates, blocks of {BLOCK_EPISODES}, \
         min cell {MIN_CELL_COUNT}, min effective {MIN_EFFECTIVE_SAMPLE}."
    );
    println!("\n[truth] exact value of always quoting each action:");
    for (target, half_spread) in HALF_SPREAD.iter().enumerate() {
        println!(
            "  half spread {half_spread}: {:.3}",
            true_value(target, HORIZON)
        );
    }

    for scenario in &SCENARIOS {
        println!("\n== {} ==", scenario.name);
        for seed in SEEDS {
            let log = generate_log(scenario.behaviour, scenario.shape, seed);
            let (actions, states) = coverage(&cell_counts(log.train()));
            println!(
                "seed {seed}: log hash {:016x}; [observed] action coverage {actions:.2}, \
                 state coverage {states:.2}, censored steps {:.3}, mean reward {:.3}",
                log.hash(),
                log.censored_share(),
                mean_reward(&log),
            );
            for (target, half_spread) in HALF_SPREAD.iter().enumerate() {
                let assessment = assess(&log, target, seed);
                println!(
                    "  half spread {half_spread}: [estimated] importance {}, model {}; [observed] effective {}, \
                     unsupported cells {}; [truth] {:.3}; {}",
                    format_estimate(assessment.importance),
                    format_estimate(assessment.model),
                    assessment
                        .diagnostics
                        .min_effective
                        .map_or("none".to_string(), |e| format!("{e:.1}")),
                    assessment.diagnostics.out_of_support_cells,
                    true_value(target, HORIZON),
                    format_decision(&assessment.decision),
                );
            }
        }
    }

    println!("\n== a policy selected on validation, read again on holdout ==");
    for seed in SEEDS {
        let log = generate_log(Behaviour::Randomized, LogShape::Full, seed);
        let counts = cell_counts(log.train());
        let holdout: Vec<&Episode> = log.holdout().iter().collect();
        if let Some((target, on_validation)) = select_on_validation(&log) {
            let on_holdout = importance(&holdout, target, &counts).map_or(f64::NAN, |i| i.value);
            println!(
                "seed {seed}: selected half spread {}, [estimated] on validation {on_validation:.3}, \
                 on holdout {on_holdout:.3}; [truth] {:.3}",
                HALF_SPREAD[target],
                true_value(target, HORIZON),
            );
        }
    }

    println!("\n== decision ==");
    println!(
        "Declared shape of a log of best quotes and aggregated fills: {}.",
        format_decision(&decide(&MARKET_LOG))
    );
    println!(
        "Stay with interactive simulation. A log could reopen this only if it records every \
         quote and the probability of its action, from a policy that randomizes, and passes \
         the diagnostics above. No real log was read. The comparison with interactive \
         simulation is still pending."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    const FIRST_SEED: u64 = SEEDS[0];
    /// Largest error accepted of an estimator that should be right.
    const TOLERANCE: f64 = 0.30;

    fn log_of(behaviour: Behaviour, shape: LogShape, seed: u64) -> Log {
        generate_log(behaviour, shape, seed)
    }

    fn randomized() -> &'static Log {
        static LOG: OnceLock<Log> = OnceLock::new();
        LOG.get_or_init(|| log_of(Behaviour::Randomized, LogShape::Full, FIRST_SEED))
    }

    fn reasons(assessment: &Assessment) -> Vec<Reason> {
        match &assessment.decision {
            Decision::Go => Vec::new(),
            Decision::NoGo(reasons) => reasons.iter().copied().collect(),
        }
    }

    #[test]
    fn test_true_value_one_step_matches_hand_computation() {
        // Half spread 1 from flat. Calm: 2 * 0.8 * 1 - 0.1 * 2 * 0.8 * 0.2 = 1.568.
        // Toxic: 2 * 0.9 * (1 - 2) - 0.1 * 2 * 0.9 * 0.1 = -1.818.
        let expected = 0.7 * 1.568 + 0.3 * -1.818;
        assert!((true_value(0, 1) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_true_value_two_steps_adds_the_next_inventory() {
        // Half spread 3, calm or toxic: from one sided inventory, the second
        // step has a side as usual, so two steps are close to twice one, less
        // the cost of the inventory carried, which adds up.
        let one = true_value(2, 1);
        let two = true_value(2, 2);
        assert!(two < 2.0 * one);
        assert!(two > one);
    }

    #[test]
    fn test_inventory_limit_stops_a_side() {
        assert_eq!(side_probabilities(MAX_INVENTORY, 0, 0), (0.0, 0.8));
        assert_eq!(side_probabilities(-MAX_INVENTORY, 0, 0), (0.8, 0.0));
        assert_eq!(side_probabilities(MAX_INVENTORY - 1, 0, 0), (0.8, 0.8));
        for episode in &randomized().episodes {
            for record in episode {
                assert!(record.observation.abs() <= MAX_INVENTORY);
            }
        }
    }

    #[test]
    fn test_log_records_the_declared_shape() {
        let log = randomized();
        assert_eq!(log.episodes.len(), EPISODES);
        for episode in &log.episodes {
            assert_eq!(episode.len(), HORIZON);
            for (step, record) in episode.iter().enumerate() {
                assert_eq!(record.step, step);
                assert_eq!(record.terminal, step + 1 == HORIZON);
                assert_eq!(record.behaviour_probability, Some(1.0 / 3.0));
            }
        }
        // The inventory of each step follows the execution of the one before.
        let episode = &log.episodes[0];
        for pair in episode.windows(2) {
            assert_eq!(
                pair[1].observation,
                next_inventory(pair[0].observation, pair[0].execution)
            );
        }
        let without = log_of(Behaviour::Randomized, LogShape::NoPropensity, FIRST_SEED);
        assert!(
            without
                .episodes
                .iter()
                .flatten()
                .all(|r| r.behaviour_probability.is_none())
        );
    }

    #[test]
    fn test_log_is_reproducible_by_seed() {
        let again = log_of(Behaviour::Randomized, LogShape::Full, FIRST_SEED);
        assert_eq!(again.hash(), randomized().hash());
        let other = log_of(Behaviour::Randomized, LogShape::Full, SEEDS[1]);
        assert_ne!(other.hash(), randomized().hash());
        // The report's numbers reproduce as well, intervals included.
        assert_eq!(
            assess(&again, 1, FIRST_SEED),
            assess(randomized(), 1, FIRST_SEED)
        );
    }

    #[test]
    fn test_splits_are_chronological_and_disjoint() {
        let log = randomized();
        assert_eq!(log.train().len(), TRAIN_EPISODES);
        assert_eq!(log.validation().len(), VALIDATION_EPISODES);
        assert_eq!(
            log.holdout().len(),
            EPISODES - TRAIN_EPISODES - VALIDATION_EPISODES
        );
        assert_eq!(log.train()[0], log.episodes[0]);
        assert_eq!(log.validation()[0], log.episodes[TRAIN_EPISODES]);
        assert_eq!(
            log.holdout()[0],
            log.episodes[TRAIN_EPISODES + VALIDATION_EPISODES]
        );
    }

    #[test]
    fn test_randomized_log_recovers_the_truth_on_every_seed() {
        for seed in SEEDS {
            let log = log_of(Behaviour::Randomized, LogShape::Full, seed);
            for target in 0..ACTIONS {
                let assessment = assess(&log, target, seed);
                assert_eq!(assessment.decision, Decision::Go, "{seed} {target}");
                let truth = true_value(target, HORIZON);
                let importance = assessment.importance.unwrap();
                let model = assessment.model.unwrap();
                assert!(
                    (importance.value - truth).abs() < TOLERANCE,
                    "{seed} {target} {importance:?} {truth}"
                );
                assert!(
                    (model.value - truth).abs() < TOLERANCE,
                    "{seed} {target} {model:?} {truth}"
                );
            }
        }
    }

    #[test]
    fn test_model_value_recovers_a_hand_built_log() {
        // Half spread 1 from flat, one record per outcome at each inventory: the
        // value is the mean reward of the step, then of the next inventories.
        let mut episodes = Vec::new();
        for inventory in -MAX_INVENTORY..=MAX_INVENTORY {
            let record = Record {
                step: 0,
                observation: inventory,
                action: 0,
                execution: Execution {
                    bid_filled: false,
                    ask_filled: false,
                },
                reward: f64::from(inventory + 10),
                terminal: false,
                behaviour_probability: None,
            };
            episodes.push(vec![record]);
        }
        let refs: Vec<&Episode> = episodes.iter().collect();
        // No fill keeps the inventory at 0: each step has a reward of 10.
        assert_eq!(model_value(&refs, 0), Some(10.0 * HORIZON as f64));
        // Another action has no cells.
        assert_eq!(model_value(&refs, 1), None);
    }

    #[test]
    fn test_importance_weights_match_hand_computation() {
        let record = |step, action, reward, probability| Record {
            step,
            observation: 0,
            action,
            execution: Execution {
                bid_filled: false,
                ask_filled: false,
            },
            reward,
            terminal: step + 1 == HORIZON,
            behaviour_probability: Some(probability),
        };
        let full = |action, reward, probability| -> Episode {
            (0..HORIZON)
                .map(|step| record(step, action, reward, probability))
                .collect()
        };
        // Two trajectories of the target, with different weights, and one that
        // leaves it at once and must not count.
        let episodes = [full(0, 1.0, 0.5), full(0, 4.0, 0.25), full(1, 100.0, 0.5)];
        let refs: Vec<&Episode> = episodes.iter().collect();
        let got = importance(&refs, 0, &[[0; ACTIONS]; STATES]).unwrap();
        let mut expected = 0.0;
        let mut min_effective = f64::INFINITY;
        for step in 0..HORIZON as i32 {
            let (a, b) = (2.0f64.powi(step + 1), 4.0f64.powi(step + 1));
            expected += (a * 1.0 + b * 4.0) / (a + b);
            min_effective = min_effective.min((a + b) * (a + b) / (a * a + b * b));
        }
        assert!((got.value - expected).abs() < 1e-12);
        assert!((got.min_effective - min_effective).abs() < 1e-12);
    }

    #[test]
    fn test_importance_counts_probabilities_when_the_log_has_none() {
        let mut episode: Episode = Vec::new();
        for step in 0..HORIZON {
            episode.push(Record {
                step,
                observation: 0,
                action: 0,
                execution: Execution {
                    bid_filled: false,
                    ask_filled: false,
                },
                reward: 1.0,
                terminal: step + 1 == HORIZON,
                behaviour_probability: None,
            });
        }
        let counts = {
            let mut counts = [[0; ACTIONS]; STATES];
            counts[state_index(0)] = [1, 3, 0];
            counts
        };
        // With one trajectory the weights cancel, so the value is the reward.
        let got = importance(&[&episode], 0, &counts).unwrap();
        assert_eq!(got.value, HORIZON as f64);
        assert!((got.min_effective - 1.0).abs() < 1e-12);
    }

    fn quiet_record(step: usize, observation: i32, action: usize, reward: f64) -> Record {
        Record {
            step,
            observation,
            action,
            execution: Execution {
                bid_filled: false,
                ask_filled: false,
            },
            reward,
            terminal: step + 1 == HORIZON,
            behaviour_probability: None,
        }
    }

    fn quiet_episode(observation: i32, action: usize, reward: f64) -> Episode {
        (0..HORIZON)
            .map(|step| quiet_record(step, observation, action, reward))
            .collect()
    }

    #[test]
    fn test_importance_weighs_by_the_counted_probability_of_each_inventory() {
        // At inventory 0 the target has 1 in 4 of the entries, at 1 it has 1 in 2,
        // so at the first step the weights are 4 and 2.
        let mut counts = [[0; ACTIONS]; STATES];
        counts[state_index(0)] = [1, 3, 0];
        counts[state_index(1)] = [1, 1, 0];
        let (a, b) = (quiet_episode(0, 0, 1.0), quiet_episode(1, 0, 7.0));
        let got = importance(&[&a, &b], 0, &counts).unwrap();
        let mut expected = 0.0;
        for step in 1..=HORIZON as i32 {
            let (wa, wb) = (4.0f64.powi(step), 2.0f64.powi(step));
            expected += (wa * 1.0 + wb * 7.0) / (wa + wb);
        }
        assert!((got.value - expected).abs() < 1e-12, "{got:?} {expected}");
    }

    #[test]
    fn test_importance_has_no_value_without_a_trajectory_or_a_probability() {
        let counts = [[1; ACTIONS]; STATES];
        // Every trajectory leaves the target before the last step.
        let mut leaves = quiet_episode(0, 0, 1.0);
        leaves[HORIZON - 1].action = 1;
        assert_eq!(importance(&[&leaves], 0, &counts), None);
        // A recorded probability of zero contradicts the action taken.
        let mut impossible = quiet_episode(0, 0, 1.0);
        impossible[0].behaviour_probability = Some(0.0);
        assert_eq!(importance(&[&impossible], 0, &counts), None);
        // Even beside a sound trajectory: the whole log is then in doubt.
        let sound = quiet_episode(0, 0, 1.0);
        assert_eq!(importance(&[&impossible, &sound], 0, &counts), None);
        // So does a count of zero, for an action train never saw.
        let unseen = [[0, 1, 1]; STATES];
        let episode = quiet_episode(0, 0, 1.0);
        assert_eq!(importance(&[&episode], 0, &unseen), None);
    }

    #[test]
    fn test_log_hash_covers_each_field() {
        let base = randomized().hash();
        let change: [fn(&mut Record); 6] = [
            |r| r.step += 1,
            |r| r.observation += 1,
            |r| r.action = (r.action + 1) % ACTIONS,
            |r| r.execution.bid_filled = !r.execution.bid_filled,
            |r| r.reward += 1.0,
            |r| r.behaviour_probability = None,
        ];
        for (index, apply) in change.iter().enumerate() {
            let mut log = randomized().clone();
            apply(&mut log.episodes[17][1]);
            assert_ne!(log.hash(), base, "field {index}");
        }
        let mut log = randomized().clone();
        log.episodes[17][2].terminal = false;
        assert_ne!(log.hash(), base, "terminal");
    }

    #[test]
    fn test_out_of_support_action_gets_no_estimate() {
        for seed in SEEDS {
            let log = log_of(Behaviour::Narrow, LogShape::Full, seed);
            let tightest = assess(&log, 0, seed);
            assert_eq!(reasons(&tightest), vec![Reason::OutOfSupport]);
            assert_eq!(tightest.diagnostics.out_of_support_cells, STATES);
            assert_eq!(tightest.importance, None);
            assert_eq!(tightest.model, None);
            // The same log can answer for what it has.
            for target in 1..ACTIONS {
                let assessment = assess(&log, target, seed);
                assert_eq!(assessment.decision, Decision::Go, "{seed} {target}");
                let truth = true_value(target, HORIZON);
                assert!((assessment.importance.unwrap().value - truth).abs() < TOLERANCE);
            }
        }
    }

    #[test]
    fn test_out_of_support_cell_threshold_is_exact() {
        let mut counts = [[MIN_CELL_COUNT; ACTIONS]; STATES];
        assert!(out_of_support(&counts, 0).is_empty());
        counts[state_index(2)][0] = MIN_CELL_COUNT - 1;
        assert_eq!(out_of_support(&counts, 0), vec![(2, 0)]);
        // It looks only at the target's own action.
        assert!(out_of_support(&counts, 1).is_empty());
    }

    #[test]
    fn test_coverage_counts_supported_actions_and_states() {
        let mut counts = [[0; ACTIONS]; STATES];
        assert_eq!(coverage(&counts), (0.0, 0.0));
        counts[0][1] = MIN_CELL_COUNT;
        counts[1][1] = MIN_CELL_COUNT - 1;
        assert_eq!(coverage(&counts), (1.0 / 3.0, 1.0 / 5.0));
        let narrow = log_of(Behaviour::Narrow, LogShape::Full, FIRST_SEED);
        assert_eq!(coverage(&cell_counts(narrow.train())), (2.0 / 3.0, 1.0));
        assert_eq!(coverage(&cell_counts(randomized().train())), (1.0, 1.0));
    }

    #[test]
    fn test_recorded_probabilities_expose_a_hidden_confounder() {
        // The recorded probabilities undo the confounding, and the model cannot,
        // so they disagree: visible in the log, with no use of the truth.
        for seed in SEEDS {
            let log = log_of(Behaviour::RegimeAware, LogShape::Full, seed);
            let assessment = assess(&log, 0, seed);
            assert_eq!(reasons(&assessment), vec![Reason::EstimatorsDisagree]);
            let truth = true_value(0, HORIZON);
            let importance_error = (assessment.importance.unwrap().value - truth).abs();
            let model_error = assessment.model.unwrap().value - truth;
            assert!(model_error > 1.0, "{model_error}");
            assert!(importance_error < model_error / 2.0, "{importance_error}");
        }
    }

    #[test]
    fn test_hidden_confounder_passes_support_checks_and_still_misleads() {
        // Without recorded probabilities the estimators share the assumption,
        // so they agree, support looks fine, and both are wrong. Only the truth
        // of this synthetic case shows it, so the log alone cannot justify a go.
        for seed in SEEDS {
            let log = log_of(Behaviour::RegimeAware, LogShape::NoPropensity, seed);
            let assessment = assess(&log, 0, seed);
            assert_eq!(reasons(&assessment), vec![Reason::NoRecordedPropensity]);
            assert_eq!(assessment.diagnostics.out_of_support_cells, 0);
            assert!(assessment.diagnostics.min_effective.unwrap() >= MIN_EFFECTIVE_SAMPLE);
            assert_eq!(assessment.diagnostics.estimators_overlap, Some(true));
            let truth = true_value(0, HORIZON);
            let importance = assessment.importance.unwrap();
            let model = assessment.model.unwrap();
            assert!(importance.value - truth > 1.0, "{importance:?} {truth}");
            assert!(model.value - truth > 1.0, "{model:?} {truth}");
            assert!(!importance.contains(truth));
            assert!(!model.contains(truth));
        }
    }

    #[test]
    fn test_fills_only_log_is_censored_and_flatters_the_reward() {
        for seed in SEEDS {
            let censored = log_of(Behaviour::RegimeAware, LogShape::FillsOnly, seed);
            let full = log_of(Behaviour::RegimeAware, LogShape::NoPropensity, seed);
            assert!(censored.censored_share() > 0.0);
            assert_eq!(full.censored_share(), 0.0);
            for target in 0..ACTIONS {
                let assessment = assess(&censored, target, seed);
                assert_eq!(assessment.importance, None);
                assert_eq!(assessment.model, None);
                let reasons = reasons(&assessment);
                assert!(reasons.contains(&Reason::Censored), "{reasons:?}");
                assert!(reasons.contains(&Reason::NoRecordedPropensity));
            }
            // What is left is a selection: the steps that filled.
            assert!(mean_reward(&censored) > mean_reward(&full));
            assert!(
                censored
                    .episodes
                    .iter()
                    .flatten()
                    .all(|r| r.execution.bid_filled || r.execution.ask_filled)
            );
        }
    }

    #[test]
    fn test_decide_names_each_reason_alone_and_together() {
        let clean = Diagnostics {
            censored: false,
            propensity_recorded: true,
            out_of_support_cells: 0,
            min_effective: Some(MIN_EFFECTIVE_SAMPLE),
            estimators_overlap: Some(true),
        };
        assert_eq!(decide(&clean), Decision::Go);
        let cases = [
            (
                Diagnostics {
                    censored: true,
                    ..clean
                },
                Reason::Censored,
            ),
            (
                Diagnostics {
                    propensity_recorded: false,
                    ..clean
                },
                Reason::NoRecordedPropensity,
            ),
            (
                Diagnostics {
                    out_of_support_cells: 1,
                    ..clean
                },
                Reason::OutOfSupport,
            ),
            (
                Diagnostics {
                    min_effective: Some(MIN_EFFECTIVE_SAMPLE - 0.1),
                    ..clean
                },
                Reason::LowEffectiveSample,
            ),
            (
                Diagnostics {
                    min_effective: None,
                    ..clean
                },
                Reason::LowEffectiveSample,
            ),
            (
                Diagnostics {
                    estimators_overlap: Some(false),
                    ..clean
                },
                Reason::EstimatorsDisagree,
            ),
        ];
        for (diagnostics, reason) in cases {
            assert_eq!(
                decide(&diagnostics),
                Decision::NoGo(BTreeSet::from([reason])),
                "{diagnostics:?}"
            );
        }
        // Not measured adds nothing when another reason already says why.
        assert_eq!(
            decide(&Diagnostics {
                censored: true,
                min_effective: None,
                ..clean
            }),
            Decision::NoGo(BTreeSet::from([Reason::Censored]))
        );
        // Without a second estimate there is no cross check, which is a reason
        // on its own, and adds nothing when another already says why.
        assert_eq!(
            decide(&Diagnostics {
                estimators_overlap: None,
                ..clean
            }),
            Decision::NoGo(BTreeSet::from([Reason::NoCrossCheck]))
        );
        assert_eq!(
            decide(&Diagnostics {
                censored: true,
                estimators_overlap: None,
                ..clean
            }),
            Decision::NoGo(BTreeSet::from([Reason::Censored]))
        );
    }

    #[test]
    fn test_market_logs_of_quotes_and_fills_are_no_go() {
        assert_eq!(
            decide(&MARKET_LOG),
            Decision::NoGo(BTreeSet::from([
                Reason::Censored,
                Reason::NoRecordedPropensity,
            ]))
        );
    }

    #[test]
    fn test_no_reward_removes_a_reason_of_support_or_record() {
        // These reasons read no reward, so raising the rewards of the target's
        // action everywhere in the log leaves them standing.
        let raised = |log: &Log, target: usize| {
            let mut log = log.clone();
            for record in log.episodes.iter_mut().flatten() {
                if record.action == target {
                    record.reward += 1_000.0;
                }
            }
            log
        };
        let narrow = log_of(Behaviour::Narrow, LogShape::Full, FIRST_SEED);
        assert_eq!(
            reasons(&assess(&raised(&narrow, 0), 0, FIRST_SEED)),
            vec![Reason::OutOfSupport]
        );
        let unrecorded = log_of(Behaviour::RegimeAware, LogShape::NoPropensity, FIRST_SEED);
        assert!(
            reasons(&assess(&raised(&unrecorded, 0), 0, FIRST_SEED))
                .contains(&Reason::NoRecordedPropensity)
        );
        let censored = log_of(Behaviour::RegimeAware, LogShape::FillsOnly, FIRST_SEED);
        assert!(reasons(&assess(&raised(&censored, 0), 0, FIRST_SEED)).contains(&Reason::Censored));
    }

    #[test]
    fn test_rewards_moved_in_holdout_alone_can_hide_a_disagreement() {
        // The stated limit of the overlap check: this log has a real disagreement,
        // and rewards raised in holdout alone bring importance sampling to the
        // model, so the decision turns to go. The truth has not moved.
        let log = log_of(Behaviour::RegimeAware, LogShape::Full, FIRST_SEED);
        let base = assess(&log, 0, FIRST_SEED);
        assert_eq!(reasons(&base), vec![Reason::EstimatorsDisagree]);
        let gap = base.model.unwrap().value - base.importance.unwrap().value;
        let mut moved = log.clone();
        for episode in &mut moved.episodes[TRAIN_EPISODES + VALIDATION_EPISODES..] {
            for record in episode {
                record.reward += gap / HORIZON as f64;
            }
        }
        assert_eq!(assess(&moved, 0, FIRST_SEED).decision, Decision::Go);
    }

    #[test]
    fn test_wide_interval_lets_a_biased_model_through() {
        // A stated negative result, on a planned seed: a go whose model is far from
        // the truth, as importance sampling has few effective trajectories here,
        // above the minimum, and so an interval wide enough to overlap it.
        let log = log_of(Behaviour::RegimeAware, LogShape::Full, 71);
        let assessment = assess(&log, 2, 71);
        assert_eq!(assessment.decision, Decision::Go);
        let truth = true_value(2, HORIZON);
        assert!(truth - assessment.model.unwrap().value > 1.0);
        assert!(assessment.importance.unwrap().contains(truth));
        assert!(!assessment.model.unwrap().contains(truth));
    }

    #[test]
    fn test_importance_stops_counting_a_trajectory_that_left_the_target() {
        // Target, other, target: the last step of the first trajectory must not
        // count, whatever its reward, and the second one holds that step up.
        let mut left = quiet_episode(0, 0, 1.0);
        left[HORIZON - 2].action = 1;
        left[HORIZON - 1].reward = 1_000.0;
        for record in &mut left {
            record.behaviour_probability = Some(0.5);
        }
        let mut stayed = quiet_episode(0, 0, 1.0);
        for record in &mut stayed {
            record.behaviour_probability = Some(0.5);
        }
        let got = importance(&[&left, &stayed], 0, &[[0; ACTIONS]; STATES]).unwrap();
        assert_eq!(got.value, HORIZON as f64);
        // The effective number at that step is one.
        assert!((got.min_effective - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_selection_never_reads_holdout() {
        let selected = select_on_validation(randomized()).unwrap();
        let mut changed = randomized().clone();
        for episode in &mut changed.episodes[TRAIN_EPISODES + VALIDATION_EPISODES..] {
            for record in episode {
                record.reward = 1e9;
                record.action = 0;
            }
        }
        assert_eq!(select_on_validation(&changed), Some(selected));
        // And it does read validation.
        let mut changed = randomized().clone();
        for episode in &mut changed.episodes[TRAIN_EPISODES..TRAIN_EPISODES + VALIDATION_EPISODES] {
            for record in episode {
                record.reward = if record.action == 2 { 1e3 } else { -1e3 };
            }
        }
        assert_eq!(select_on_validation(&changed).map(|s| s.0), Some(2));
    }

    #[test]
    fn test_model_is_fitted_on_train_and_importance_reads_holdout() {
        let base = assess(randomized(), 1, FIRST_SEED);
        let mut later = randomized().clone();
        for episode in &mut later.episodes[TRAIN_EPISODES..] {
            for record in episode {
                record.reward += 50.0;
            }
        }
        let moved = assess(&later, 1, FIRST_SEED);
        assert_eq!(moved.model, base.model);
        assert!(moved.importance.unwrap().value > base.importance.unwrap().value + 100.0);
        // Then importance disagrees with the model: its reward moved alone.
        assert_eq!(moved.diagnostics.estimators_overlap, Some(false));
    }

    #[test]
    fn test_block_interval_needs_a_value_in_every_replicate() {
        let episodes = &randomized().episodes[..200];
        let (low, high) = block_interval(episodes, 7, |sample| {
            Some(sample.iter().map(|e| e[0].reward).sum::<f64>() / sample.len() as f64)
        })
        .unwrap();
        assert!(low < high);
        let mean = episodes.iter().map(|e| e[0].reward).sum::<f64>() / episodes.len() as f64;
        assert!(low < mean && mean < high, "{low} {mean} {high}");
        assert_eq!(block_interval(episodes, 7, |_| None), None);
        // Fewer episodes than a block cannot be resampled.
        assert_eq!(
            block_interval(&episodes[..BLOCK_EPISODES - 1], 7, |_| Some(1.0)),
            None
        );
    }

    #[test]
    fn test_estimate_overlap_and_containment_include_their_edges() {
        let a = Estimate {
            value: 1.0,
            low: 0.0,
            high: 2.0,
        };
        let b = Estimate {
            value: 3.0,
            low: 2.0,
            high: 4.0,
        };
        let c = Estimate {
            value: 3.0,
            low: 2.5,
            high: 4.0,
        };
        assert!(a.overlaps(b) && b.overlaps(a));
        assert!(!a.overlaps(c) && !c.overlaps(a));
        assert!(a.contains(0.0) && a.contains(2.0));
        assert!(!a.contains(2.1) && !a.contains(-0.1));
    }

    #[test]
    fn test_pick_follows_the_probabilities() {
        let mut rng = Rng(5);
        let mut counts = [0usize; ACTIONS];
        for _ in 0..30_000 {
            counts[rng.pick(&[0.2, 0.3, 0.5])] += 1;
        }
        assert!((counts[0] as f64 / 30_000.0 - 0.2).abs() < 0.02);
        assert!((counts[1] as f64 / 30_000.0 - 0.3).abs() < 0.02);
        assert!((counts[2] as f64 / 30_000.0 - 0.5).abs() < 0.02);
        for _ in 0..1_000 {
            assert!(rng.below(7) < 7);
        }
    }
}
