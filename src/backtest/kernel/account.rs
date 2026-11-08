//! Agent-known capacity account (admission and mask).
//!
//! Built only from what the agent knows: its own submissions (order ids are
//! assigned synchronously) and the delivered reports, including their
//! `origin` and `cancel_pending` fields. Each open order reserves one order
//! slot, two venue messages and three deliveries until it is closed; partial
//! fill reports are bounded from the dataset instead, not from what the agent
//! knows ([`super::CapacityNeed::partial_reports_per_order`]).

use super::book::{Arr, Blank, ExecKind, Full, Origin, ReportRec};

/// An order the agent submitted and has not seen closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OpenOrder {
    pub(super) order_id: u64,
    pub(super) awaiting_cancel_outcome: bool,
}

impl Blank for OpenOrder {
    const BLANK: Self = Self {
        order_id: 0,
        awaiting_cancel_outcome: false,
    };
}

/// The account: open orders in ascending id order.
pub(super) type AgentAccount<const ORD: usize> = Arr<OpenOrder, ORD>;

/// Opens `order_id` (ids grow, so appending keeps the order).
pub(super) fn open<const ORD: usize>(
    account: &mut AgentAccount<ORD>,
    order_id: u64,
) -> Result<(), Full> {
    account.push(OpenOrder {
        order_id,
        awaiting_cancel_outcome: false,
    })
}

/// Applies a delivered report, reading only its order id, kind, origin and
/// `cancel_pending`.
pub(super) fn on_delivery<const ORD: usize>(account: &mut AgentAccount<ORD>, report: &ReportRec) {
    let Some(index) = account.position(|o| o.order_id == report.order_id) else {
        return;
    };
    let close = match report.kind {
        // An expiry that beat the agent's cancel leaves the order awaiting
        // that cancel's outcome, as a fill before it does.
        ExecKind::Canceled => !report.cancel_pending,
        ExecKind::CancelRejected => true,
        ExecKind::Rejected => match report.origin {
            Origin::Local => true,
            Origin::Venue => !report.cancel_pending,
        },
        ExecKind::Fill => !report.cancel_pending,
        ExecKind::Accepted | ExecKind::PartialFill => false,
    };
    if close {
        let _ = account.remove(index);
    } else if matches!(
        report.kind,
        ExecKind::Fill | ExecKind::Rejected | ExecKind::Canceled
    ) && let Some(open) = account.get_mut(index)
    {
        open.awaiting_cancel_outcome = true;
    }
}

/// Number of open orders.
pub(super) fn held<const ORD: usize>(account: &AgentAccount<ORD>) -> u64 {
    u64::try_from(account.len()).unwrap_or(u64::MAX)
}
