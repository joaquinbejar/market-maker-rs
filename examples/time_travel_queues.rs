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
//! - each side's touch queue in event sizes, in buckets 0-1, 2-4, 5-9 and 10 and more.
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

/// The grid of states, in milliseconds.
const GRID_MS: u64 = 1_000;

/// The horizon of a markout and of a move, in grid steps.
const HORIZON_STEPS: usize = 10;

/// Bucket bounds of a touch queue, in event sizes: 0-1, 2-4, 5-9, 10 and more.
const QUEUE_BUCKETS: [u32; 3] = [2, 5, 10];

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

fn main() {
    // The protocol above is committed before any run; the study follows.
    let _ = (
        GRID_MS,
        HORIZON_STEPS,
        QUEUE_BUCKETS,
        MAX_SPREAD_STATE,
        SEPARATIONS,
        TAKE_IMBALANCE,
        SEARCH_SEEDS,
        SYNTHETIC_HOURS,
        SPLITS,
        CAPTURE_SPLIT,
        REPLICATES,
        MEAN_BLOCK_DAYS,
        BOOTSTRAP_SEED,
        MAX_NO_CANDIDATE_SHARE,
    );
}
