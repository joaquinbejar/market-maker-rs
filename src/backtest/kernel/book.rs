//! Fixed capacity containers and the venue side records.

use crate::execution::Action;
use crate::types::action::Dir;
use core::num::NonZeroU64;

/// A value used to fill unused container slots.
pub(super) trait Blank: Copy {
    /// The filler.
    const BLANK: Self;
}

/// The container is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Full;

/// An ordered list of at most `N` items; unused slots hold the blank value
/// so equal contents compare equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Arr<T: Blank, const N: usize> {
    items: [T; N],
    len: usize,
}

impl<T: Blank, const N: usize> Arr<T, N> {
    /// An empty list.
    pub(super) const EMPTY: Self = Self {
        items: [T::BLANK; N],
        len: 0,
    };

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn as_slice(&self) -> &[T] {
        self.items.get(..self.len).unwrap_or(&[])
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len;
        self.items.get_mut(..len).unwrap_or(&mut [])
    }

    pub(super) fn first(&self) -> Option<&T> {
        self.as_slice().first()
    }

    pub(super) fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    pub(super) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.as_mut_slice().get_mut(index)
    }

    pub(super) fn push(&mut self, value: T) -> Result<(), Full> {
        let slot = self.items.get_mut(self.len).ok_or(Full)?;
        *slot = value;
        self.len = self.len.checked_add(1).ok_or(Full)?;
        Ok(())
    }

    /// Inserts `value` at `index` (at most `len`), shifting the tail.
    pub(super) fn insert(&mut self, index: usize, value: T) -> Result<(), Full> {
        if self.len >= N || index > self.len {
            return Err(Full);
        }
        let tail = self.items.get_mut(index..=self.len).ok_or(Full)?;
        tail.rotate_right(1);
        let slot = tail.first_mut().ok_or(Full)?;
        *slot = value;
        self.len = self.len.checked_add(1).ok_or(Full)?;
        Ok(())
    }

    /// Removes and returns the item at `index`, shifting the tail.
    pub(super) fn remove(&mut self, index: usize) -> Option<T> {
        let value = *self.get(index)?;
        let len = self.len;
        let tail = self.items.get_mut(index..len)?;
        tail.rotate_left(1);
        let last = tail.last_mut()?;
        *last = T::BLANK;
        self.len = len.checked_sub(1)?;
        Some(value)
    }

    pub(super) fn pop_front(&mut self) -> Option<T> {
        self.remove(0)
    }

    pub(super) fn position(&self, pred: impl Fn(&T) -> bool) -> Option<usize> {
        self.as_slice().iter().position(pred)
    }

    /// Removes every item from `len` on.
    pub(super) fn truncate(&mut self, len: usize) {
        while self.len > len {
            let Some(last) = self.len.checked_sub(1) else {
                return;
            };
            let _ = self.remove(last);
        }
    }

    pub(super) fn clear(&mut self) {
        *self = Self::EMPTY;
    }
}

/// Side, encoded as 1 (buy) and 2 (sell).
pub(super) fn dir_code(dir: Dir) -> u8 {
    match dir {
        Dir::Buy => 1,
        Dir::Sell => 2,
    }
}

/// Inverse of [`dir_code`].
pub(super) fn dir_from_code(code: u8) -> Option<Dir> {
    match code {
        1 => Some(Dir::Buy),
        2 => Some(Dir::Sell),
        _ => None,
    }
}

/// Venue state of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OrderState {
    Submitted = 1,
    Active = 2,
    Filled = 3,
    Canceled = 4,
    Rejected = 5,
}

impl OrderState {
    pub(super) fn code(self) -> u8 {
        match self {
            Self::Submitted => 1,
            Self::Active => 2,
            Self::Filled => 3,
            Self::Canceled => 4,
            Self::Rejected => 5,
        }
    }

    pub(super) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Submitted),
            2 => Some(Self::Active),
            3 => Some(Self::Filled),
            4 => Some(Self::Canceled),
            5 => Some(Self::Rejected),
            _ => None,
        }
    }

    pub(super) fn is_terminal(self) -> bool {
        match self {
            Self::Filled | Self::Canceled | Self::Rejected => true,
            Self::Submitted | Self::Active => false,
        }
    }
}

/// An order at the venue (resting, pending activation or awaiting a
/// cancel outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VenueOrder {
    pub(super) id: u64,
    pub(super) side: Dir,
    pub(super) price: u64,
    pub(super) qty: u64,
    pub(super) filled: u64,
    pub(super) state: OrderState,
    pub(super) active_after_ns: u64,
    pub(super) cancel_inflight: bool,
}

impl VenueOrder {
    pub(super) fn remaining(&self) -> u64 {
        self.qty.saturating_sub(self.filled)
    }
}

impl Blank for VenueOrder {
    const BLANK: Self = Self {
        id: 0,
        side: Dir::Buy,
        price: 0,
        qty: 0,
        filled: 0,
        state: OrderState::Submitted,
        active_after_ns: 0,
        cancel_inflight: false,
    };
}

/// A venue message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MsgKind {
    /// An order reaching the venue; with a life, the venue schedules its
    /// expiry when it activates.
    Activate {
        order: VenueOrder,
        life: Option<NonZeroU64>,
    },
    Cancel {
        id: u64,
    },
    /// The venue's own cancel of a resting order at its expiry.
    Expire {
        id: u64,
    },
}

/// A venue message keyed by `(eff_ns, seq)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VenueMsg {
    pub(super) eff_ns: u64,
    pub(super) seq: u64,
    pub(super) kind: MsgKind,
}

impl VenueMsg {
    pub(super) fn key(&self) -> (u64, u64) {
        (self.eff_ns, self.seq)
    }
}

impl Blank for VenueMsg {
    const BLANK: Self = Self {
        eff_ns: 0,
        seq: 0,
        kind: MsgKind::Cancel { id: 0 },
    };
}

/// Kind of an execution report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecKind {
    /// Order accepted and resting.
    Accepted,
    /// Order rejected (locally or by venue validation).
    Rejected,
    /// Order canceled.
    Canceled,
    /// Cancel arrived after the order ended.
    CancelRejected,
    /// Order partially filled (not produced by the immediate model); it stays
    /// live, and its last report is a `Fill` or a `Canceled`.
    PartialFill,
    /// Order completely filled.
    Fill,
}

impl ExecKind {
    pub(super) fn code(self) -> u8 {
        match self {
            Self::Accepted => 1,
            Self::Rejected => 2,
            Self::Canceled => 3,
            Self::CancelRejected => 4,
            Self::PartialFill => 5,
            Self::Fill => 6,
        }
    }

    pub(super) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Accepted),
            2 => Some(Self::Rejected),
            3 => Some(Self::Canceled),
            4 => Some(Self::CancelRejected),
            5 => Some(Self::PartialFill),
            6 => Some(Self::Fill),
            _ => None,
        }
    }
}

/// Where a report was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// In the action transaction (crossed quote or pretrade refusal).
    Local,
    /// By a venue message or matching.
    Venue,
}

impl Origin {
    pub(super) fn code(self) -> u8 {
        match self {
            Self::Local => 1,
            Self::Venue => 2,
        }
    }

    pub(super) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Local),
            2 => Some(Self::Venue),
            _ => None,
        }
    }
}

/// An execution report as the kernel produces and delivers it.
///
/// The fields mirror the engine's `ExecutionReport` (in ticks, lots and
/// money units), plus two delivered fields the engine does not have:
/// `origin` and `cancel_pending`. `cancel_pending` is set on a fill or a
/// venue `Rejected` when a cancel for the order is still outstanding at the
/// venue, so a `Canceled` or `CancelRejected` follows while the run keeps
/// processing venue messages (pending messages are dropped at the end of
/// the data, and a truncated run stops early).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReportRec {
    /// Venue time of the event, in nanoseconds.
    pub ts_ns: u64,
    /// Sequence number of the event stamp.
    pub seq: u64,
    /// Order id.
    pub order_id: u64,
    /// Order side.
    pub side: Action,
    /// Transition kind.
    pub kind: ExecKind,
    /// Where the report was produced.
    pub origin: Origin,
    /// A cancel for the order is still outstanding.
    pub cancel_pending: bool,
    /// Execution price for fills, limit price otherwise, in ticks.
    pub price: u64,
    /// Order quantity, in lots.
    pub qty: u64,
    /// Quantity executed by this event, in lots.
    pub last: u64,
    /// Cumulative filled quantity, in lots.
    pub cum: u64,
    /// Fee of this event, in money units `10^-s`.
    pub fee_u: i128,
}

impl ReportRec {
    /// Open quantity after the event, in lots.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.qty.saturating_sub(self.cum)
    }
}

impl Blank for ReportRec {
    const BLANK: Self = Self {
        ts_ns: 0,
        seq: 0,
        order_id: 0,
        side: Action::Buy,
        kind: ExecKind::Accepted,
        origin: Origin::Venue,
        cancel_pending: false,
        price: 0,
        qty: 0,
        last: 0,
        cum: 0,
        fee_u: 0,
    };
}

/// A fill, as the engine records it in its trades.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FillRec {
    /// Time of the fill (the matched row), in nanoseconds.
    pub ts_ns: u64,
    /// Execution id.
    pub exec_id: u64,
    /// Order id.
    pub order_id: u64,
    /// Side.
    pub side: Action,
    /// Price, in ticks.
    pub price: u64,
    /// Quantity, in lots.
    pub qty: u64,
    /// Fee, in money units.
    pub fee_u: i128,
}

/// A report waiting in the delivery FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Delivery {
    pub(super) at_ns: u64,
    pub(super) report: ReportRec,
    pub(super) fill: Option<FillRec>,
}

impl Blank for Delivery {
    const BLANK: Self = Self {
        at_ns: 0,
        report: ReportRec::BLANK,
        fill: None,
    };
}

/// An observation waiting in the observation FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ObsEntry {
    pub(super) ready_ns: u64,
    pub(super) row: u64,
}

impl Blank for ObsEntry {
    const BLANK: Self = Self {
        ready_ns: 0,
        row: 0,
    };
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    impl Blank for u64 {
        const BLANK: Self = 0;
    }

    #[test]
    fn test_arr_insert_remove_keeps_order_and_blanks_tail() {
        let mut a: Arr<u64, 4> = Arr::EMPTY;
        a.push(1).unwrap();
        a.push(3).unwrap();
        a.insert(1, 2).unwrap();
        a.insert(0, 0).unwrap();
        assert_eq!(a.as_slice(), &[0, 1, 2, 3]);
        assert_eq!(a.push(9), Err(Full));
        assert_eq!(a.remove(1), Some(1));
        assert_eq!(a.as_slice(), &[0, 2, 3]);
        let mut b: Arr<u64, 4> = Arr::EMPTY;
        for v in [0, 2, 3] {
            b.push(v).unwrap();
        }
        assert_eq!(a, b);
        assert_eq!(a.remove(7), None);
    }
}
