//! Integer reservations and the shared pretrade capacity.
//!
//! The gate mirrors the engine's `LimitsPolicy`: it decides on the venue
//! position (fills the agent may not have received yet), the remaining
//! quantity of the live orders on the order's side and the new order, through
//! the same `limits_capacity` function the policy uses.

use crate::Decimal;
use crate::risk::RiskLimits;
use crate::risk::pretrade::{CapacityVerdict, GateNumericError, RejectCode, limits_capacity};
use crate::types::action::Dir;

use super::grid::Unit;

/// Venue position and live reservations, in lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GateState {
    pub(super) venue_pos_lots: i64,
    pub(super) live_buy_lots: u64,
    pub(super) live_sell_lots: u64,
    /// Last positive mark, in half ticks.
    pub(super) mark2: Option<u64>,
}

impl GateState {
    pub(super) const NEW: Self = Self {
        venue_pos_lots: 0,
        live_buy_lots: 0,
        live_sell_lots: 0,
        mark2: None,
    };

    pub(super) fn live(&self, dir: Dir) -> u64 {
        match dir {
            Dir::Buy => self.live_buy_lots,
            Dir::Sell => self.live_sell_lots,
        }
    }

    fn live_mut(&mut self, dir: Dir) -> &mut u64 {
        match dir {
            Dir::Buy => &mut self.live_buy_lots,
            Dir::Sell => &mut self.live_sell_lots,
        }
    }

    /// Reserves `lots` for a submitted order.
    pub(super) fn reserve(&mut self, dir: Dir, lots: u64) -> Option<()> {
        let live = self.live_mut(dir);
        *live = live.checked_add(lots)?;
        Some(())
    }

    /// Releases the remaining `lots` of a canceled or rejected order.
    pub(super) fn release(&mut self, dir: Dir, lots: u64) -> Option<()> {
        let live = self.live_mut(dir);
        *live = live.checked_sub(lots)?;
        Some(())
    }

    /// Books a fill of `lots`.
    pub(super) fn fill(&mut self, dir: Dir, lots: u64) -> Option<()> {
        let signed = i64::try_from(lots).ok()?;
        self.venue_pos_lots = match dir {
            Dir::Buy => self.venue_pos_lots.checked_add(signed)?,
            Dir::Sell => self.venue_pos_lots.checked_sub(signed)?,
        };
        self.release(dir, lots)
    }
}

/// Outcome of the gate for one order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// Submitted as requested.
    Accepted,
    /// Submitted with fewer lots.
    Reduced,
    /// Reduced to a quantity that is not a whole number of lots: refused,
    /// as the engine refuses and releases it.
    ReducedOffLot,
    /// Refused by a limit, or by the combined limits of a hedged kernel
    /// ([`super::CombinedLimits`]), which never reduce.
    Rejected,
}

/// Which check of the gate took a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GateCheck {
    /// [`super::KernelConfig::limits`].
    Limits,
    /// The combined limits of a hedged kernel ([`super::CombinedLimits`]).
    Combined,
    /// The balance gate ([`super::KernelMargin`]).
    Margin,
}

/// Gate decision: the verdict and the lots submitted (0 when refused).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GateDecision {
    pub(super) verdict: GateVerdict,
    pub(super) submitted: u64,
}

/// Runs the shared capacity for an order of `qty` lots at `price` ticks.
///
/// # Errors
///
/// Returns [`GateNumericError`] if a conversion or the exposure sum is not
/// representable (unreachable inside the validated bounds).
pub(super) fn check(
    limits: &RiskLimits,
    gate: &GateState,
    tick: &Unit,
    lot: &Unit,
    dir: Dir,
    price: u64,
    qty: u64,
) -> Result<GateDecision, GateNumericError> {
    let position = lot
        .to_decimal(i128::from(gate.venue_pos_lots))
        .ok_or(GateNumericError)?;
    let live = lot
        .to_decimal(i128::from(gate.live(dir)))
        .ok_or(GateNumericError)?;
    let price_d = tick.to_decimal(i128::from(price)).ok_or(GateNumericError)?;
    let qty_d = lot.to_decimal(i128::from(qty)).ok_or(GateNumericError)?;
    let decision = match limits_capacity(limits, position, live, dir, price_d, qty_d)? {
        CapacityVerdict::Accept => GateDecision {
            verdict: GateVerdict::Accepted,
            submitted: qty,
        },
        CapacityVerdict::Reduce(approved) => match whole_lots(lot, approved) {
            Some(lots) => GateDecision {
                verdict: GateVerdict::Reduced,
                submitted: lots,
            },
            None => GateDecision {
                verdict: GateVerdict::ReducedOffLot,
                submitted: 0,
            },
        },
        CapacityVerdict::Reject(RejectCode::LimitReached) => GateDecision {
            verdict: GateVerdict::Rejected,
            submitted: 0,
        },
    };
    Ok(decision)
}

/// The balances the balance gate decides on, in U and lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Balances {
    /// Quote balance less the reservations of the live buys, in U.
    pub(super) quote_free_u: i128,
    /// Base balance less the live sells, in lots.
    pub(super) base_free_lots: i128,
}

/// What the balance gate reserves with: the tick and the lot, the fee scale `F` and the
/// fee rate mantissa reserved on buys.
#[derive(Debug, Clone, Copy)]
pub(super) struct MarginTerms<'a> {
    pub(super) tick: &'a Unit,
    pub(super) lot: &'a Unit,
    pub(super) fee_scale: i128,
    pub(super) fee_num: i128,
}

impl MarginTerms<'_> {
    /// The quote reserved by a buy of `lots` at `price` ticks, in U: its notional plus
    /// `fee_num` of it, i.e. per unit of the lot's grid `price * mt * (F + fee_num)`.
    pub(super) fn buy_reservation_u(&self, price: u64, lots: u64) -> Option<i128> {
        self.per_grid_u(price)?
            .checked_mul(i128::from(lots))?
            .checked_mul(i128::from(self.lot.mantissa))
    }

    /// The quote reserved per unit of the lot's grid at `price` ticks, in U.
    fn per_grid_u(&self, price: u64) -> Option<i128> {
        i128::from(price)
            .checked_mul(i128::from(self.tick.mantissa))?
            .checked_mul(self.fee_scale.checked_add(self.fee_num)?)
    }
}

/// The balance gate for an order of `qty` lots at `price` ticks, as
/// `RiskGate::submit` sizes it in units of the lot's grid (one lot is
/// `lot.mantissa` of them): accepted when the balance covers it, otherwise
/// reduced to the largest whole number of grid units it covers, refused
/// below one, and refused when that is not a whole number of lots.
///
/// `None` if a value is not representable (unreachable inside the validated
/// bounds).
pub(super) fn margin_check(
    terms: &MarginTerms<'_>,
    balances: &Balances,
    dir: Dir,
    price: u64,
    qty: u64,
) -> Option<GateDecision> {
    let ml = i128::from(terms.lot.mantissa);
    let wanted = i128::from(qty).checked_mul(ml)?;
    let covered = match dir {
        Dir::Buy => {
            let per = terms.per_grid_u(price)?;
            if per <= 0 {
                return None;
            }
            balances.quote_free_u.max(0).checked_div(per)?
        }
        Dir::Sell => balances.base_free_lots.max(0).checked_mul(ml)?,
    };
    let decision = if covered >= wanted {
        GateDecision {
            verdict: GateVerdict::Accepted,
            submitted: qty,
        }
    } else if covered < 1 {
        GateDecision {
            verdict: GateVerdict::Rejected,
            submitted: 0,
        }
    } else if covered.checked_rem(ml)? == 0 {
        GateDecision {
            verdict: GateVerdict::Reduced,
            submitted: u64::try_from(covered.checked_div(ml)?).ok()?,
        }
    } else {
        GateDecision {
            verdict: GateVerdict::ReducedOffLot,
            submitted: 0,
        }
    };
    Some(decision)
}

/// `approved` in lots when it is a whole, positive number of lots, as the
/// engine's `round_to_step(approved, lot, false) == approved` check.
fn whole_lots(lot: &Unit, approved: Decimal) -> Option<u64> {
    if crate::backtest::engine::round_to_step(approved, lot.value, false) != approved {
        return None;
    }
    let lots = lot.count(approved)?;
    u64::try_from(lots).ok().filter(|v| *v > 0)
}
