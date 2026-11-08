//! The agreement of the reward with the evaluation metrics.

use crate::backtest::evaluation::{ExecutedFill, FundingPayment, Mark};
use crate::backtest::kernel::{EventSink, FillRec, HedgeEvent, KernelEvent};
use crate::strategy::walk_forward::TimeWindow;

/// Collects what the evaluation needs from a run: the completed rows and the fills,
/// and with a hedge lane its rows, fills and funding payments (#266).
///
/// Give it to [`RlEnv::reset_with`](super::RlEnv::reset_with) and every
/// [`RlEnv::step_with`](super::RlEnv::step_with) of an episode. It sees venue events,
/// so it belongs to an audit and must not reach a policy.
#[derive(Debug, Clone, Default)]
pub struct EpisodeRecorder {
    pub(super) rows: Vec<u64>,
    pub(super) fills: Vec<FillRec>,
    pub(super) hedge: Vec<HedgeEvent>,
}

impl EventSink for EpisodeRecorder {
    fn event(&mut self, event: KernelEvent) {
        match event {
            KernelEvent::RowEnd { row, .. } => self.rows.push(row),
            KernelEvent::Fill(fill) => self.fills.push(fill),
            KernelEvent::Report(_)
            | KernelEvent::Gate(_)
            | KernelEvent::Delivered(_)
            | KernelEvent::Observed { .. }
            | KernelEvent::EndOfData { .. }
            | KernelEvent::Halted(_)
            | KernelEvent::Decided(_)
            | KernelEvent::DroppedMsg { .. }
            | KernelEvent::Expired { .. }
            | KernelEvent::DroppedExpiry { .. } => {}
        }
    }

    fn hedge(&mut self, event: HedgeEvent) {
        self.hedge.push(event);
    }
}

/// An episode's records in the form of the evaluation, closing fill included.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EpisodeRecords {
    /// From the first mark to just after the last record.
    pub window: TimeWindow,
    /// One per completed row, at its mid.
    pub marks: Vec<Mark>,
    /// The fills in order, then the terminal close if there was one.
    pub fills: Vec<ExecutedFill>,
    /// With a hedge lane, one per hedge row run, at its mid; empty without one.
    pub hedge_marks: Vec<Mark>,
    /// With a hedge lane, its fills in order, then its terminal close if there
    /// was one; empty without one.
    pub hedge_fills: Vec<ExecutedFill>,
    /// With a hedge lane, its funding payments; empty without one.
    pub hedge_funding: Vec<FundingPayment>,
}
