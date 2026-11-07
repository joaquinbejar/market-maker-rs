//! A stationary queue-reactive fit from the Hyperliquid capture (#290).
//!
//! #263's comparison found the fitted model transient on the capture: at every queue size its
//! fitted arrivals exceeded its departures, so simulated queues climbed to the cap. Three causes
//! are examined: exits by price move, block-tied times, and events dropped unevenly by type.
//!
//! Estimators, declared before any run:
//! - E0: `estimate_parameters` as it is, which leaves out every event without a positive wait;
//! - E1: every event counted, a zero wait adding no time;
//! - E2: per queue, each block's events collapsed into their net change, then counted as E1;
//! - E3: an explicit exit by price move while the touch queue is not empty, built only under
//!   [`E3_CONDITION`].
//!
//! The protocol below is fixed before any run. #286 already read the holdout windows used in
//! phase 3, as the report says.

/// The condition under which E3 is built, fixed before phase 1 runs: on the month's training
/// part, at least this share of touch-queue lives ends by a mid move while the queue is not
/// empty ...
const E3_MIN_EXIT_BY_MOVE_SHARE: f64 = 0.5;

/// ... and both E1 and E2 are transient: their simulated mean touch queue over a 2 h training
/// window is at least this share of the cap.
const E3_TRANSIENT_SHARE_OF_CAP: f64 = 0.5;

/// The written condition, for the report.
const E3_CONDITION: &str = "E3 is built if, on the month's training part, at least half of the touch-queue lives end by a mid move while the queue is not empty, and both E1 and E2 are transient (their simulated mean touch queue over a 2 h training window reaches half the cap)";

/// The queue cap, as #263's comparison.
const MAX_QUEUE: u32 = 200;

/// Phase 3's choice: among the stationary estimators, the lowest training-window queue size
/// total variation at limit 0. Stationary means the simulated mean touch queue stays under
/// this share of the cap.
const STATIONARY_SHARE_OF_CAP: f64 = 0.5;

/// Seed of every model run, as #263's comparison.
const QR_SEED: u64 = 263;

fn main() {
    let _ = (
        E3_MIN_EXIT_BY_MOVE_SHARE,
        E3_TRANSIENT_SHARE_OF_CAP,
        E3_CONDITION,
        MAX_QUEUE,
        STATIONARY_SHARE_OF_CAP,
        QR_SEED,
    );
}
