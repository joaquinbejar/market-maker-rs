//! The venue ledger (updated on fill) and the agent view (updated on
//! delivery).

use crate::Decimal;
use crate::position::inventory::InventoryPosition;

/// Cost basis exactly as the engine's `InventoryPosition` keeps it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct InvDec {
    pub(super) quantity: Decimal,
    pub(super) avg_entry: Decimal,
    pub(super) last_update: u64,
}

impl InvDec {
    pub(super) const FLAT: Self = Self {
        quantity: Decimal::ZERO,
        avg_entry: Decimal::ZERO,
        last_update: 0,
    };

    /// Applies a fill through the engine's own `update_fill`; `None`, with
    /// the position unchanged, if the quantity overflows.
    pub(super) fn update_fill(&mut self, delta: Decimal, price: Decimal, ts: u64) -> Option<()> {
        let mut position = InventoryPosition {
            quantity: self.quantity,
            avg_entry_price: self.avg_entry,
            last_update: self.last_update,
        };
        position.update_fill(delta, price, ts).ok()?;
        self.quantity = position.quantity;
        self.avg_entry = position.avg_entry_price;
        self.last_update = position.last_update;
        Some(())
    }
}

/// Mirror of the engine's venue side bookkeeping (`process_fill`,
/// `mark_to`, equity, peak and drawdown).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VenueLedger {
    /// Sum of the fill cash flows, in U.
    pub(super) cash_u: i128,
    /// Quote fees, in U.
    pub(super) fixed_fees_u: i128,
    /// Base fee units (always zero in v1).
    pub(super) base_fee_units: i128,
    /// The engine's `position`.
    pub(super) pos: InvDec,
    /// The engine's `pnl.realized`.
    pub(super) realized: Decimal,
    /// Last mark, in half ticks.
    pub(super) last_mark2: Option<u64>,
    /// Peak equity, in half-U.
    pub(super) peak_h: i128,
    /// Largest drawdown, in half-U.
    pub(super) max_dd_h: i128,
    /// Largest absolute position, in lots.
    pub(super) max_abs_pos: u64,
}

impl VenueLedger {
    pub(super) fn new(c0_u: i128) -> Option<Self> {
        Some(Self {
            cash_u: 0,
            fixed_fees_u: 0,
            base_fee_units: 0,
            pos: InvDec::FLAT,
            realized: Decimal::ZERO,
            last_mark2: None,
            peak_h: c0_u.checked_mul(2)?,
            max_dd_h: 0,
            max_abs_pos: 0,
        })
    }
}

/// What the agent knows of its position: updated per delivered fill.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AgentView {
    /// The engine's `agent_position`.
    pub(super) pos: InvDec,
    /// Last observed row.
    pub(super) last_obs_row: Option<u64>,
    /// Delivered local rejects.
    pub(super) delivered_local_rejects: u64,
    /// Delivered fills.
    pub(super) delivered_fills: u64,
}

impl AgentView {
    pub(super) const NEW: Self = Self {
        pos: InvDec::FLAT,
        last_obs_row: None,
        delivered_local_rejects: 0,
        delivered_fills: 0,
    };
}
