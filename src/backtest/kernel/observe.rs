//! What the agent sees: the observation, the action grid and the mask.

use crate::Decimal;

use super::encode::Writer;
use super::state::{RejectReason, Status};

/// The agent's view at a decision point.
///
/// A pure function of the agent view (delivered fills and reports), the
/// account, the last observed row, the step count, the status, the
/// admission reject counters and the configuration. It never reads venue
/// state, undelivered reports or rows not yet observed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// Last observed row, if any.
    pub row: Option<u64>,
    /// Exchange time of that row, in nanoseconds (0 if none).
    pub ts_ns: u64,
    /// Its bid, in ticks.
    pub bid: u64,
    /// Its bid size, in lots.
    pub bid_lots: u64,
    /// Its ask, in ticks.
    pub ask: u64,
    /// Its ask size, in lots.
    pub ask_lots: u64,
    /// Position known from delivered fills.
    pub position: Decimal,
    /// Average entry of that position.
    pub avg_entry: Decimal,
    /// Delivered fills.
    pub delivered_fills: u64,
    /// Delivered local rejects.
    pub delivered_local_rejects: u64,
    /// Orders open in the agent account.
    pub open_orders: u64,
    /// Open orders waiting for a cancel outcome after a fill or venue
    /// reject.
    pub awaiting_cancel_outcome: u64,
    /// Decisions taken.
    pub step: u64,
    /// Run status.
    pub status: Status,
    /// Admission rejects, by [`RejectReason::index`].
    pub rejected: [u64; 8],
}

fn status_code(status: Status) -> (u8, u16) {
    match status {
        Status::Running => (0, 0),
        Status::Terminal(reason) => (
            1,
            match reason {
                super::state::TermReason::EndOfData => 1,
            },
        ),
        Status::Truncated => (2, 0),
        Status::Halted(reason) => (3, reason.code()),
    }
}

pub(super) fn write_status(w: &mut Writer, status: Status) {
    let (tag, code) = status_code(status);
    w.u8(tag);
    w.u16(code);
}

impl Observation {
    /// Canonical encoding, for bit-exact comparisons.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.option_u64(self.row);
        for v in [self.ts_ns, self.bid, self.bid_lots, self.ask, self.ask_lots] {
            w.u64(v);
        }
        w.decimal(self.position);
        w.decimal(self.avg_entry);
        for v in [
            self.delivered_fills,
            self.delivered_local_rejects,
            self.open_orders,
            self.awaiting_cancel_outcome,
            self.step,
        ] {
            w.u64(v);
        }
        write_status(&mut w, self.status);
        for v in self.rejected {
            w.u64(v);
        }
        w.finish()
    }
}

/// Quote grid relative to the observed mid: `bid = floor(mid2 / 2) - off`,
/// `ask = ceil(mid2 / 2) + off`, with `K` offsets per side and `M` sizes
/// (both at most 64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionGrid<const K: usize, const M: usize> {
    bid_offsets: [u16; K],
    ask_offsets: [u16; K],
    sizes: [u64; M],
}

impl<const K: usize, const M: usize> ActionGrid<K, M> {
    /// Builds a grid.
    ///
    /// # Errors
    ///
    /// Returns [`crate::types::error::MMError::InvalidConfiguration`] if
    /// `K` or `M` exceeds 64.
    pub fn new(
        bid_offsets: [u16; K],
        ask_offsets: [u16; K],
        sizes: [u64; M],
    ) -> crate::types::error::MMResult<Self> {
        if K > 64 || M > 64 {
            return Err(crate::types::error::MMError::InvalidConfiguration(format!(
                "action grid sizes must be at most 64, got {K} offsets and {M} sizes"
            )));
        }
        Ok(Self {
            bid_offsets,
            ask_offsets,
            sizes,
        })
    }

    /// Bid offsets.
    #[must_use]
    pub fn bid_offsets(&self) -> &[u16; K] {
        &self.bid_offsets
    }

    /// Ask offsets.
    #[must_use]
    pub fn ask_offsets(&self) -> &[u16; K] {
        &self.ask_offsets
    }

    /// Sizes, in lots.
    #[must_use]
    pub fn sizes(&self) -> &[u64; M] {
        &self.sizes
    }

    /// Bid price of offset `i` around `mid2` half ticks, if representable.
    #[must_use]
    pub fn bid_price(&self, mid2: u64, i: usize) -> Option<u64> {
        (mid2 / 2).checked_sub(u64::from(*self.bid_offsets.get(i)?))
    }

    /// Ask price of offset `i` around `mid2` half ticks, if representable.
    #[must_use]
    pub fn ask_price(&self, mid2: u64, i: usize) -> Option<u64> {
        mid2.div_ceil(2)
            .checked_add(u64::from(*self.ask_offsets.get(i)?))
    }
}

/// Admission mask over an [`ActionGrid`], factorised exactly: a quote
/// `(bid i, ask j, bid size a, ask size b)` is admitted iff
/// `global_quote_ok` and bits `i` of `bid`, `j` of `ask`, `a` and `b` of
/// `size` are set. `Hold` is always admitted while the run continues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionMask {
    /// The run is at a decision point and the account has room for two
    /// more open orders.
    pub global_quote_ok: bool,
    /// Admissible bid offsets.
    pub bid: u64,
    /// Admissible ask offsets.
    pub ask: u64,
    /// Admissible sizes.
    pub size: u64,
    /// Reason quotes are refused when `global_quote_ok` is false.
    pub refusal: Option<RejectReason>,
}

impl ActionMask {
    /// Returns true if the quote at these grid indices is admitted.
    #[must_use]
    pub fn allows(&self, bid: usize, ask: usize, bid_size: usize, ask_size: usize) -> bool {
        let bit = |mask: u64, i: usize| {
            u32::try_from(i)
                .ok()
                .and_then(|i| mask.checked_shr(i))
                .is_some_and(|v| v & 1 == 1)
        };
        self.global_quote_ok
            && bit(self.bid, bid)
            && bit(self.ask, ask)
            && bit(self.size, bid_size)
            && bit(self.size, ask_size)
    }

    /// Canonical encoding.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bool(self.global_quote_ok);
        w.u64(self.bid);
        w.u64(self.ask);
        w.u64(self.size);
        w.u16(self.refusal.map_or(0, RejectReason::code));
        w.finish()
    }
}
