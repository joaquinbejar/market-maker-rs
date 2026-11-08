//! #290's protocol, fixed before any run (first committed in 23efdd9 inside
//! `examples/qr_estimation.rs`, values unchanged), shared by `qr_estimation` and
//! `hyperliquid_validation`.

#![allow(dead_code)]

/// The condition under which E3 is built, fixed before phase 1 runs: on the month's training
/// part, at least this share of touch-queue lives ends by a mid move while the queue is not
/// empty ...
pub const E3_MIN_EXIT_BY_MOVE_SHARE: f64 = 0.5;

/// ... and both E1 and E2 are transient: their simulated mean touch queue over a 2 h training
/// window is at least this share of the cap.
pub const E3_TRANSIENT_SHARE_OF_CAP: f64 = 0.5;

/// The written condition, for the report.
pub const E3_CONDITION: &str = "E3 is built if, on the month's training part, at least half of the touch-queue lives end by a mid move while the queue is not empty, and both E1 and E2 are transient (their simulated mean touch queue over a 2 h training window reaches half the cap)";

/// The queue cap, as #263's comparison.
pub const MAX_QUEUE: u32 = 200;

/// Phase 3's choice: among the stationary estimators, the lowest training-window queue size
/// total variation at limit 0. Stationary means the simulated mean touch queue stays under
/// this share of the cap.
pub const STATIONARY_SHARE_OF_CAP: f64 = 0.5;

/// Seed of every model run, as #263's comparison.
pub const QR_SEED: u64 = 263;

/// The length of a simulated window, in seconds: 2 h, as #286's windows.
pub const WINDOW_SECONDS: f64 = 7_200.0;

/// Bounds of the capture's block length histogram, in milliseconds.
pub const BLOCK_GAP_BOUNDS_MS: [u64; 9] = [1, 10, 50, 100, 250, 500, 1_000, 2_000, 5_000];

/// Whether a mean touch queue is transient under `share` of the cap.
pub fn reaches(mean_touch_queue: f64, share: f64) -> bool {
    mean_touch_queue >= share * f64::from(MAX_QUEUE)
}
