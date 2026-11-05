//! A perpetual and the funding a venue declares for it.
//!
//! A perpetual has no expiry and pays funding instead. [`PerpetualTerms`] holds the
//! contract, the funding interval, who publishes the rates, and the declared rate and mark
//! of each funding instant. Nothing here is estimated: an instant with no declared rate
//! stays visibly unpaid in a backtest rather than being filled in.
//!
//! A position pays the instants in `(opened, closed]`: one opened exactly on an instant does
//! not pay it, and one closed exactly on an instant does. With a positive rate a long pays
//! and a short receives.

use financial_types::instrument::{ContractKind, InstrumentSpec};
use thiserror::Error;

use super::futures::SettlementSource;
use crate::Decimal;

/// Why funding terms were refused.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PerpetualError {
    /// The contract is not a perpetual.
    #[error("{instrument} is {kind:?}, not a perpetual")]
    NotAPerpetual {
        /// The contract.
        instrument: String,
        /// What it is.
        kind: ContractKind,
    },
    /// The funding interval is zero.
    #[error("the funding interval must not be zero")]
    ZeroInterval,
    /// No funding instant was declared.
    #[error("no funding instant was declared")]
    NoFunding,
    /// An instant is not a multiple of the interval.
    #[error("funding instant {at_ms} is off the grid of {interval_ms} ms")]
    OffGrid {
        /// The instant.
        at_ms: u64,
        /// The interval.
        interval_ms: u64,
    },
    /// An instant does not follow the previous one by exactly one interval.
    #[error("funding instant {at_ms} does not follow {previous_ms} by one interval")]
    NotConsecutive {
        /// The previous instant.
        previous_ms: u64,
        /// The instant.
        at_ms: u64,
    },
    /// A funding mark is not positive.
    #[error("funding mark {mark} at {at_ms} is not positive")]
    NotPositiveMark {
        /// The instant.
        at_ms: u64,
        /// The mark.
        mark: Decimal,
    },
}

/// The rate and mark a venue declared for one funding instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FundingPoint {
    /// The instant, in milliseconds since the epoch.
    pub at_ms: u64,
    /// The mark the notional is taken at.
    pub mark: Decimal,
    /// The rate for the interval: with a positive one, a long pays.
    pub rate: Decimal,
}

/// A perpetual, its funding grid and its declared funding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerpetualTerms {
    spec: InstrumentSpec,
    interval_ms: u64,
    source: SettlementSource,
    points: Vec<FundingPoint>,
}

impl PerpetualTerms {
    /// Terms of the perpetual `spec`, with `points` in order on the grid of `interval_ms`,
    /// each one interval after the last.
    ///
    /// # Errors
    ///
    /// A [`PerpetualError`] for another kind of contract, a zero interval, no points, a
    /// point off the grid or out of sequence, or a mark that is not positive.
    pub fn new(
        spec: InstrumentSpec,
        interval_ms: u64,
        source: SettlementSource,
        points: Vec<FundingPoint>,
    ) -> Result<Self, PerpetualError> {
        if !matches!(spec.kind(), ContractKind::Perpetual) {
            return Err(PerpetualError::NotAPerpetual {
                instrument: spec.id().as_str().to_string(),
                kind: spec.kind(),
            });
        }
        if interval_ms == 0 {
            return Err(PerpetualError::ZeroInterval);
        }
        if points.is_empty() {
            return Err(PerpetualError::NoFunding);
        }
        let mut previous: Option<u64> = None;
        for point in &points {
            if point.at_ms.checked_rem(interval_ms) != Some(0) {
                return Err(PerpetualError::OffGrid {
                    at_ms: point.at_ms,
                    interval_ms,
                });
            }
            if let Some(previous_ms) = previous
                && previous_ms.checked_add(interval_ms) != Some(point.at_ms)
            {
                return Err(PerpetualError::NotConsecutive {
                    previous_ms,
                    at_ms: point.at_ms,
                });
            }
            if point.mark <= Decimal::ZERO {
                return Err(PerpetualError::NotPositiveMark {
                    at_ms: point.at_ms,
                    mark: point.mark,
                });
            }
            previous = Some(point.at_ms);
        }
        Ok(Self {
            spec,
            interval_ms,
            source,
            points,
        })
    }

    /// The contract.
    #[must_use]
    pub fn spec(&self) -> &InstrumentSpec {
        &self.spec
    }

    /// The funding interval, in milliseconds.
    #[must_use]
    pub fn interval_ms(&self) -> u64 {
        self.interval_ms
    }

    /// Who publishes the rates.
    #[must_use]
    pub fn source(&self) -> &SettlementSource {
        &self.source
    }

    /// The declared funding, in order.
    #[must_use]
    pub fn points(&self) -> &[FundingPoint] {
        &self.points
    }

    /// How many instants of the grid in `(after_ms, until_ms]` have no declared rate.
    #[must_use]
    pub fn missing_between(&self, after_ms: u64, until_ms: u64) -> u64 {
        let on_grid = |from: u64, to: u64| {
            let index = |at: u64| at.checked_div(self.interval_ms).unwrap_or(0);
            index(to).saturating_sub(index(from))
        };
        let total = on_grid(after_ms, until_ms);
        let (Some(first), Some(last)) = (self.points.first(), self.points.last()) else {
            return total;
        };
        let from = after_ms.max(first.at_ms.saturating_sub(self.interval_ms));
        let to = until_ms.min(last.at_ms);
        let covered = if to > from { on_grid(from, to) } else { 0 };
        total.saturating_sub(covered)
    }
}
