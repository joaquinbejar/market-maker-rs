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
//! [`HedgeCosts::NotModelled`] says so, rather than a zero.
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
//! so a reader can tell it from a flat day that was observed.
//!
//! # Limits
//!
//! A window starts flat and nothing carries over from another window. The input
//! holds nothing outside its window, which [`EvaluationInput::new`] checks, so a
//! markout near the end of a window cannot read a later window's marks: that fill
//! is counted as excluded instead.

mod attribution;
mod definitions;

#[cfg(feature = "instruments")]
pub use attribution::check_contract;
pub use attribution::{
    Attribution, BucketAttribution, BucketedAttribution, DailyAttribution, EvaluationInput,
    ExecutedFill, FillQualityReport, FillQualitySpec, HedgeCosts, HedgeNotModelledReason,
    InventoryStats, LedgerTotals, Mark, MetricValue, MetricsSpec, Reading, SideQuality,
    WindowMetrics, attribute, attribute_by, attribute_daily, evaluate_window, fill_quality,
    inventory_stats, max_drawdown,
};
pub use definitions::{
    Aggregation, CapitalDenominator, Direction, MetricDefinition, MetricHorizon, MetricId,
    MetricUnit,
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
