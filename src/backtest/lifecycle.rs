//! Causal order lifecycle for the backtest engine.
//!
//! This module separates the quoting decision from the moment an order can
//! interact with the market. Every event gets an [`EventStamp`] from a
//! [`SimClock`], which gives a deterministic total order even when several
//! events share the same timestamp.
//!
//! # Ordering rule
//!
//! Within one timestamp, market data is processed before any order that was
//! submitted at that timestamp becomes eligible for matching. An order
//! submitted while handling the tick at `t` can only match ticks with a
//! timestamp strictly greater than its activation time. With zero latency this
//! is still a strict causal tie break: a quote is never filled against the
//! tick it was computed from.
//!
//! # Contract status
//!
//! [`ExecutionReport`] is the backtest's provisional execution vocabulary. It
//! will be aligned with the shared multiproduct contracts once those are
//! agreed; field names may change in a later version.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{SimClock, SimOrder, SimOrderState};
//! use market_maker_rs::execution::Action;
//! use market_maker_rs::dec;
//!
//! let mut clock = SimClock::new();
//! clock.advance_to(1_000).unwrap();
//! let stamp = clock.stamp();
//!
//! let mut order = SimOrder::new(1, Action::Buy, dec!(100), dec!(1), stamp, 1_000);
//! order.accept().unwrap();
//! assert!(!order.is_eligible_at(1_000));
//! assert!(order.is_eligible_at(1_001));
//!
//! let remaining = order.apply_fill(dec!(0.4)).unwrap();
//! assert_eq!(remaining, dec!(0.6));
//! assert_eq!(order.state, SimOrderState::PartiallyFilled);
//! assert!(order.apply_fill(dec!(0.7)).is_err());
//! ```

use crate::Decimal;
use crate::execution::Action;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Identifier of a simulated order. Assigned sequentially by the engine.
pub type SimOrderId = u64;

/// Position of an event in the simulation's total order.
///
/// Stamps compare by timestamp first and then by sequence number, so events
/// with equal timestamps are still strictly ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EventStamp {
    /// Simulated time in milliseconds.
    pub timestamp: u64,
    /// Monotonic sequence number, unique within a run.
    pub sequence: u64,
}

/// Simulated clock producing totally ordered [`EventStamp`]s.
#[derive(Debug, Clone, Default)]
pub struct SimClock {
    now: u64,
    next_sequence: u64,
}

impl SimClock {
    /// Creates a clock at time zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current simulated time in milliseconds.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now
    }

    /// Moves the clock to `timestamp`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if `timestamp` is earlier than
    /// the current time; simulated time never goes backwards.
    pub fn advance_to(&mut self, timestamp: u64) -> MMResult<()> {
        if timestamp < self.now {
            return Err(MMError::InvalidMarketState(format!(
                "clock cannot go backwards: now {}, requested {}",
                self.now, timestamp
            )));
        }
        self.now = timestamp;
        Ok(())
    }

    /// Returns a new stamp at the current time with the next sequence number.
    pub fn stamp(&mut self) -> EventStamp {
        let stamp = EventStamp {
            timestamp: self.now,
            sequence: self.next_sequence,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        stamp
    }

    /// Resets the clock to time zero and sequence zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// State of a simulated order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SimOrderState {
    /// Created by the strategy, not yet accepted by the simulated venue.
    Submitted,
    /// Accepted and resting, nothing filled yet.
    Active,
    /// Accepted and partially filled.
    PartiallyFilled,
    /// Completely filled (terminal).
    Filled,
    /// Canceled (terminal). Earlier fills are kept.
    Canceled,
    /// Rejected by validation (terminal).
    Rejected,
}

impl SimOrderState {
    /// Returns true for states that can never change again.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Canceled | Self::Rejected)
    }

    /// Returns true if the order rests in the book and can be filled.
    #[must_use]
    pub fn is_live(self) -> bool {
        matches!(self, Self::Active | Self::PartiallyFilled)
    }
}

/// A simulated limit order that persists across ticks.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimOrder {
    /// Stable order identifier.
    pub id: SimOrderId,
    /// Order side.
    pub side: Action,
    /// Limit price.
    pub price: Decimal,
    /// Original quantity.
    pub quantity: Decimal,
    /// Cumulative filled quantity.
    pub filled: Decimal,
    /// Current state.
    pub state: SimOrderState,
    /// Stamp of the submission event.
    pub submitted_at: EventStamp,
    /// The order may only match market events with a timestamp strictly
    /// greater than this value.
    pub active_after: u64,
}

impl SimOrder {
    /// Creates a submitted order.
    #[must_use]
    pub fn new(
        id: SimOrderId,
        side: Action,
        price: Decimal,
        quantity: Decimal,
        submitted_at: EventStamp,
        active_after: u64,
    ) -> Self {
        Self {
            id,
            side,
            price,
            quantity,
            filled: Decimal::ZERO,
            state: SimOrderState::Submitted,
            submitted_at,
            active_after,
        }
    }

    /// Quantity still open.
    ///
    /// `fill` keeps `filled <= quantity`, so this never saturates for an
    /// order built by this crate.
    #[must_use]
    pub fn remaining(&self) -> Decimal {
        self.quantity.saturating_sub(self.filled)
    }

    /// Returns true if the order is live and may match a market event at
    /// `timestamp`.
    #[must_use]
    pub fn is_eligible_at(&self, timestamp: u64) -> bool {
        self.state.is_live() && timestamp > self.active_after
    }

    /// Validates the order parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the side is neither buy nor
    /// sell (checked first), and [`MMError::InvalidQuoteGeneration`] if
    /// price or quantity is not strictly positive.
    pub fn validate(&self) -> MMResult<()> {
        Dir::try_from(self.side)?;
        if self.price <= Decimal::ZERO {
            return Err(MMError::InvalidQuoteGeneration(format!(
                "order {} has non positive price {}",
                self.id, self.price
            )));
        }
        if self.quantity <= Decimal::ZERO {
            return Err(MMError::InvalidQuoteGeneration(format!(
                "order {} has non positive quantity {}",
                self.id, self.quantity
            )));
        }
        Ok(())
    }

    /// Moves a submitted order to `Active`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the side is neither buy nor
    /// sell, and [`MMError::InvalidMarketState`] unless the order is
    /// `Submitted`. The order is unchanged on error.
    pub fn accept(&mut self) -> MMResult<()> {
        Dir::try_from(self.side)?;
        self.transition_from_submitted(SimOrderState::Active)
    }

    /// Moves a submitted order to `Rejected`.
    ///
    /// Any side is accepted: rejecting is how an invalid order, including
    /// one with an unsupported side, is taken out.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] unless the order is `Submitted`.
    pub fn reject(&mut self) -> MMResult<()> {
        self.transition_from_submitted(SimOrderState::Rejected)
    }

    fn transition_from_submitted(&mut self, to: SimOrderState) -> MMResult<()> {
        if self.state != SimOrderState::Submitted {
            return Err(MMError::InvalidMarketState(format!(
                "order {} cannot move from {:?} to {:?}",
                self.id, self.state, to
            )));
        }
        self.state = to;
        Ok(())
    }

    /// Applies a fill and returns the new remaining quantity.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the side is neither buy nor
    /// sell (checked first), and [`MMError::InvalidMarketState`] if the
    /// order is not live, the quantity is not strictly positive, or it
    /// exceeds the remainder. The order is left unchanged on error.
    pub fn apply_fill(&mut self, quantity: Decimal) -> MMResult<Decimal> {
        Dir::try_from(self.side)?;
        if !self.state.is_live() {
            return Err(MMError::InvalidMarketState(format!(
                "order {} cannot be filled in state {:?}",
                self.id, self.state
            )));
        }
        if quantity <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(format!(
                "order {} fill quantity must be positive, got {}",
                self.id, quantity
            )));
        }
        let remaining = self.remaining();
        if quantity > remaining {
            return Err(MMError::InvalidMarketState(format!(
                "order {} fill {} exceeds remaining {}",
                self.id, quantity, remaining
            )));
        }
        // 0 < quantity <= remaining was checked above, so this cannot fail.
        self.filled = self.filled.checked_add(quantity).ok_or_else(|| {
            MMError::NumericalError(format!("order {} filled quantity overflows", self.id))
        })?;
        self.state = if self.remaining() == Decimal::ZERO {
            SimOrderState::Filled
        } else {
            SimOrderState::PartiallyFilled
        };
        Ok(self.remaining())
    }

    /// Cancels a live or submitted order. Earlier fills are kept.
    ///
    /// Any side is accepted: cancelling only takes the order out.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if the order is terminal.
    pub fn cancel(&mut self) -> MMResult<()> {
        if self.state.is_terminal() {
            return Err(MMError::InvalidMarketState(format!(
                "order {} is already terminal ({:?})",
                self.id, self.state
            )));
        }
        self.state = SimOrderState::Canceled;
        Ok(())
    }
}

/// Kind of an [`ExecutionReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionKind {
    /// Order accepted and resting.
    Accepted,
    /// Order rejected by validation.
    Rejected,
    /// Order partially filled.
    PartialFill,
    /// Order completely filled.
    Fill,
    /// Order canceled; `remaining` is the canceled quantity.
    Canceled,
    /// Cancel arrived too late: the order was already terminal (for
    /// example completely filled). Earlier fills stand.
    CancelRejected,
}

/// Record of one order lifecycle transition.
///
/// Reports are append only: a later report never modifies an earlier one.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionReport {
    /// Position of the event in the simulation order.
    pub stamp: EventStamp,
    /// Order the report refers to.
    pub order_id: SimOrderId,
    /// Order side.
    pub side: Action,
    /// Transition kind.
    pub kind: ExecutionKind,
    /// Execution price for fills, limit price otherwise.
    pub price: Decimal,
    /// Quantity executed by this event (zero for non fill events).
    pub last_quantity: Decimal,
    /// Cumulative filled quantity after this event.
    pub cumulative_quantity: Decimal,
    /// Open quantity after this event.
    pub remaining: Decimal,
    /// Fee charged by this event.
    pub fee: Decimal,
}

impl ExecutionReport {
    /// Builds a non fill report (accepted, rejected, canceled or cancel
    /// rejected) for `order`.
    #[must_use]
    pub fn status(stamp: EventStamp, order: &SimOrder, kind: ExecutionKind) -> Self {
        Self {
            stamp,
            order_id: order.id,
            side: order.side,
            kind,
            price: order.price,
            last_quantity: Decimal::ZERO,
            cumulative_quantity: order.filled,
            remaining: order.remaining(),
            fee: Decimal::ZERO,
        }
    }

    /// Builds a fill report for `order` after the fill was applied.
    #[must_use]
    pub fn fill(
        stamp: EventStamp,
        order: &SimOrder,
        price: Decimal,
        quantity: Decimal,
        fee: Decimal,
    ) -> Self {
        let kind = if order.state == SimOrderState::Filled {
            ExecutionKind::Fill
        } else {
            ExecutionKind::PartialFill
        };
        Self {
            stamp,
            order_id: order.id,
            side: order.side,
            kind,
            price,
            last_quantity: quantity,
            cumulative_quantity: order.filled,
            remaining: order.remaining(),
            fee,
        }
    }

    /// Returns true for fill reports.
    #[must_use]
    pub fn is_fill(&self) -> bool {
        matches!(self.kind, ExecutionKind::Fill | ExecutionKind::PartialFill)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn stamp(timestamp: u64, sequence: u64) -> EventStamp {
        EventStamp {
            timestamp,
            sequence,
        }
    }

    fn live_order(qty: Decimal) -> SimOrder {
        let mut order = SimOrder::new(7, Action::Sell, dec!(101), qty, stamp(10, 0), 10);
        order.accept().unwrap();
        order
    }

    #[test]
    fn test_stamps_order_equal_timestamps_by_sequence() {
        let mut clock = SimClock::new();
        clock.advance_to(5).unwrap();
        let a = clock.stamp();
        let b = clock.stamp();
        assert_eq!(a.timestamp, b.timestamp);
        assert!(a < b);
        clock.advance_to(6).unwrap();
        let c = clock.stamp();
        assert!(b < c);
    }

    #[test]
    fn test_clock_rejects_going_backwards() {
        let mut clock = SimClock::new();
        clock.advance_to(10).unwrap();
        assert!(clock.advance_to(9).is_err());
        assert!(clock.advance_to(10).is_ok());
        assert_eq!(clock.now(), 10);
        clock.reset();
        assert_eq!(clock.now(), 0);
        assert_eq!(clock.stamp().sequence, 0);
    }

    #[test]
    fn test_order_not_eligible_at_submission_time() {
        let order = live_order(dec!(1));
        assert!(!order.is_eligible_at(10));
        assert!(order.is_eligible_at(11));
    }

    #[test]
    fn test_submitted_order_not_eligible() {
        let order = SimOrder::new(1, Action::Buy, dec!(100), dec!(1), stamp(0, 0), 0);
        assert!(!order.is_eligible_at(100));
    }

    #[test]
    fn test_partial_fills_accumulate_without_exceeding_remainder() {
        let mut order = live_order(dec!(1));
        assert_eq!(order.apply_fill(dec!(0.3)).unwrap(), dec!(0.7));
        assert_eq!(order.state, SimOrderState::PartiallyFilled);
        assert!(order.apply_fill(dec!(0.8)).is_err());
        assert_eq!(order.filled, dec!(0.3));
        assert_eq!(order.apply_fill(dec!(0.7)).unwrap(), Decimal::ZERO);
        assert_eq!(order.state, SimOrderState::Filled);
    }

    #[test]
    fn test_terminal_order_is_never_reactivated() {
        let mut order = live_order(dec!(1));
        order.apply_fill(dec!(1)).unwrap();
        assert!(order.apply_fill(dec!(0.1)).is_err());
        assert!(order.cancel().is_err());
        assert!(order.accept().is_err());
        assert_eq!(order.state, SimOrderState::Filled);
    }

    #[test]
    fn test_cancel_keeps_earlier_fills() {
        let mut order = live_order(dec!(1));
        order.apply_fill(dec!(0.25)).unwrap();
        order.cancel().unwrap();
        assert_eq!(order.state, SimOrderState::Canceled);
        assert_eq!(order.filled, dec!(0.25));
        assert_eq!(order.remaining(), dec!(0.75));
        assert!(order.cancel().is_err());
    }

    #[test]
    fn test_validation_rejects_non_positive_values() {
        let bad_price = SimOrder::new(1, Action::Buy, dec!(0), dec!(1), stamp(0, 0), 0);
        assert!(bad_price.validate().is_err());
        let bad_qty = SimOrder::new(2, Action::Buy, dec!(1), dec!(-1), stamp(0, 0), 0);
        assert!(bad_qty.validate().is_err());
        let mut rejected = bad_qty.clone();
        rejected.reject().unwrap();
        assert!(rejected.state.is_terminal());
    }

    #[test]
    fn test_zero_fill_rejected() {
        let mut order = live_order(dec!(1));
        assert!(order.apply_fill(Decimal::ZERO).is_err());
    }

    #[test]
    fn test_fill_report_kind() {
        let mut order = live_order(dec!(1));
        order.apply_fill(dec!(0.5)).unwrap();
        let partial = ExecutionReport::fill(stamp(11, 1), &order, dec!(101), dec!(0.5), dec!(0.01));
        assert_eq!(partial.kind, ExecutionKind::PartialFill);
        assert!(partial.is_fill());
        assert_eq!(partial.remaining, dec!(0.5));
        order.apply_fill(dec!(0.5)).unwrap();
        let full = ExecutionReport::fill(stamp(12, 2), &order, dec!(101), dec!(0.5), dec!(0.01));
        assert_eq!(full.kind, ExecutionKind::Fill);
        assert_eq!(full.cumulative_quantity, dec!(1));
        let canceled = ExecutionReport::status(stamp(12, 3), &order, ExecutionKind::Canceled);
        assert!(!canceled.is_fill());
    }

    #[test]
    fn test_unsupported_action_order_is_refused_before_any_effect() {
        let unsupported = MMError::UnsupportedAction {
            action: Action::Other,
        };
        let order = SimOrder::new(1, Action::Other, dec!(100), dec!(2), stamp(0, 0), 0);

        assert_eq!(order.validate(), Err(unsupported.clone()));

        let mut accepted = order.clone();
        assert_eq!(accepted.accept(), Err(unsupported.clone()));
        assert_eq!(accepted, order);

        // Even when forced live by the caller, it is never filled.
        let mut live = order.clone();
        live.state = SimOrderState::Active;
        let before = live.clone();
        assert_eq!(live.apply_fill(dec!(1)), Err(unsupported));
        assert_eq!(live, before);

        // Rejecting or cancelling takes it out.
        let mut rejected = order.clone();
        rejected.reject().unwrap();
        assert_eq!(rejected.state, SimOrderState::Rejected);
        let mut cancelled = order;
        cancelled.cancel().unwrap();
        assert_eq!(cancelled.state, SimOrderState::Canceled);
    }
}
