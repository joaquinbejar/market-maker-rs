//! Metric definitions and attribution for comparing strategies.
//!
//! This module is plain arithmetic over plain records: [`Mark`]s and
//! [`ExecutedFill`]s inside one [`TimeWindow`]. It knows nothing about the engine
//! that produced them, so another crate can use the same definitions for its own
//! strategies. [`ExecutedFill::from_simulated`] is the only link to
//! [`BacktestEngine`](crate::backtest::BacktestEngine).
//!
//! # Attribution
//!
//! For a spot style position that starts flat, with `s = +1` for a buy and `-1`
//! for a sell, `q` the size, `P` the fill price, `M_f` the mark in force at the
//! fill and `b` a fee in the base asset:
//!
//! ```text
//! spread capture   = Σ s q (M_f - P)
//! inventory        = Σ p_i (M_i - M_{i-1}), p_i the position over that interval
//! quote fees       = Σ fees fixed in the quote asset
//! base fees        = Σ b M_f
//! net              = spread capture + inventory - quote fees - base fees
//! ```
//!
//! The position `p` is net of base fees, since a fee in the base asset is base
//! that is no longer held, and later price moves do not apply to it. `net` is
//! also computed directly, as cash plus the final position at the last mark minus
//! quote fees, and the two must agree. Every sum and product must be exact: a
//! `Decimal` that would round is [`EvaluationError::Inexact`].
//!
//! The formula is linear in the mark, so it does not apply to an inverse contract
//! or to any position whose value is not `position * mark`. Those runs must not be
//! given to this module: records carry no contract, so `check_contract` (feature
//! `instruments`) is how a caller refuses a specification first. Hedge costs are not part of `net`:
//! with one instrument [`HedgeCosts::NotModelled`] says so, rather than a zero.
//!
//! # A hedge in a second instrument
//!
//! [`EvaluationInput::with_hedge`] adds a second instrument's marks, fills and funding
//! payments (#266). The hedge is attributed against its own marks, by the same rule
//! of the mark in force, into [`HedgeAttribution`]: `spread_capture +
//! inventory_pnl - fees + funding`, checked against the hedge's cash, position and
//! funding as the quoted lane is. Each of its terms goes into the bucket of the
//! quoted mark in force at its time, or the first bucket before any quoted mark,
//! so the buckets still sum to the total. It is reported in
//! [`Attribution::hedge_costs`] and not in [`Attribution::net_pnl`], which stays
//! the quoted lane's; [`BucketedAttribution::reconcile_lanes`] checks both lanes.
//!
//! # The mark in force
//!
//! It is the last mark with a timestamp at or before the fill's. Marks at equal
//! timestamps are applied in the order given, and a fill at that timestamp comes
//! after the last of them. A fill before the first mark is an error.
//!
//! # Buckets
//!
//! [`attribute_by`] labels each mark. A bucket gets the price move from the previous
//! mark to each of its marks, and the fills whose mark in force is one of its marks.
//! So the inventory carried into a day is attributed to that day, the move across
//! midnight belongs to the day it arrives in, and a day's inventory result runs to
//! that day's last mark. [`attribute_daily`] keeps the whole UTC calendar of the
//! window: a day without marks and fills has a zero attribution and `marks == 0`,
//! so a reader can tell it from a flat day that was observed. Fills count by the
//! mark in force: a fill on a day with no marks counts on the earlier day that owns
//! that mark.
//!
//! # Experiments
//!
//! [`Experiment`] holds one copy of the execution rules. A [`Candidate`] brings only a
//! strategy built from the training ticks, so two candidates cannot differ in fees,
//! fills, latency or windows. The steps consume each other:
//!
//! ```text
//! Experiment::new -> evaluate_validation -> select -> run_holdout
//! ```
//!
//! Each run gets a new strategy, fill model and policy, with seeds from the
//! experiment, the window and the scenario, so the order of the candidates changes
//! nothing. A run that does not reconcile, has an engine fault or stops early is a
//! failed row with its reason, and the others go on. The experiment builds its own
//! spot engine and takes no instrument, so an inverse contract cannot enter it.
//!
//! Sensitivity is the [`Scenario`]s after the first: every candidate runs under each
//! of them on validation, and none on the holdout. The bootstrap uses the days with
//! ticks only, and the same resampled days for every row, so a difference to a
//! baseline is a paired one.
//!
//! # Limits
//!
//! A window starts flat and nothing carries over from another window. The input
//! holds nothing outside its window, which [`EvaluationInput::new`] checks, so a
//! markout near the end of a window cannot read a later window's marks: that fill
//! is counted as excluded instead.

mod attribution;
mod definitions;
mod experiment;

#[cfg(feature = "instruments")]
pub use attribution::check_contract;
pub use attribution::{
    Attribution, BucketAttribution, BucketedAttribution, DailyAttribution, EvaluationInput,
    ExecutedFill, FillQualityReport, FillQualitySpec, FundingPayment, HedgeAttribution, HedgeCosts,
    HedgeNotModelledReason, InventoryStats, LedgerTotals, Mark, MetricValue, MetricsSpec, Reading,
    SideQuality, WindowMetrics, attribute, attribute_by, attribute_daily, evaluate_window,
    fill_quality, inventory_stats, max_drawdown,
};
pub use definitions::{
    Aggregation, CapitalDenominator, Direction, MetricDefinition, MetricHorizon, MetricId,
    MetricUnit,
};

pub use experiment::{
    BootstrapConfig, BootstrapOutcome, Candidate, CandidateKind, CandidateRecord, EvaluationRow,
    Experiment, ExperimentConfig, HoldoutReport, Manifest, Provenance, RegimeLabeller, RowMetrics,
    RowOutcome, Scenario, ScenarioRecord, Selection, SharedPolicy, Splits, ValidatedExperiment,
    ValidationReport, WindowKind, block_bootstrap, stratified_block_bootstrap,
};

use crate::Decimal;

#[cfg(doc)]
use crate::strategy::walk_forward::TimeWindow;

/// Why an evaluation could not be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EvaluationError {
    /// The records break a precondition.
    #[error("invalid evaluation input: {reason}")]
    InvalidInput {
        /// What is wrong.
        reason: String,
    },
    /// A fill has no mark at or before it, so it has no reference price.
    #[error("the fill at {timestamp_ms} ms has no mark at or before it")]
    FillBeforeFirstMark {
        /// Time of the fill, in milliseconds.
        timestamp_ms: u64,
    },
    /// A sum or product would round or overflow a `Decimal`.
    #[error("{what} is not exact in a Decimal")]
    Inexact {
        /// The quantity.
        what: &'static str,
    },
    /// The contract's value is not its position times its mark.
    #[error("unsupported contract: {reason}")]
    UnsupportedContract {
        /// Why.
        reason: String,
    },
    /// The attribution and the books disagree.
    #[error("{what} does not reconcile: attributed {attributed}, booked {booked}")]
    NotReconciled {
        /// The quantity compared.
        what: &'static str,
        /// The value from the attribution.
        attributed: Decimal,
        /// The value from the books.
        booked: Decimal,
    },
}
