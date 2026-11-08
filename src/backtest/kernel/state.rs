//! Kernel state, phases, outcomes, reasons and events.

use crate::Decimal;
use core::num::NonZeroU64;

use super::account::AgentAccount;
use super::book::{Arr, Delivery, FillRec, ObsEntry, ReportRec, VenueMsg, VenueOrder};
use super::data::DatasetId;
use super::gate::{GateCheck, GateState, GateVerdict};
use super::hedge::HedgeEvent;
use super::ledger::{AgentView, VenueLedger};
use super::rng::KernelRng;
use crate::execution::Action;

/// Where the common advance stands.
///
/// Code 2 is reserved and never used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The current row's first event pass, before matching.
    BeforeMatch = 1,
    /// The current row's second event pass, after its observation was
    /// queued.
    ObsQueued = 3,
    /// Paused at a decision point.
    AwaitDecision = 4,
    /// The run is over.
    Terminal = 5,
}

impl Phase {
    pub(super) fn code(self) -> u8 {
        match self {
            Self::BeforeMatch => 1,
            Self::ObsQueued => 3,
            Self::AwaitDecision => 4,
            Self::Terminal => 5,
        }
    }

    pub(super) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::BeforeMatch),
            3 => Some(Self::ObsQueued),
            4 => Some(Self::AwaitDecision),
            5 => Some(Self::Terminal),
            _ => None,
        }
    }
}

/// Why a run ended normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TermReason {
    /// Every row was processed.
    EndOfData = 1,
}

/// Why a run halted. Codes 1 to 4 and 9 are retired and never reused.
///
/// No halt is reachable through [`super::FinancialKernel::new`] except
/// `StateMismatch` (a state from another kernel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HaltReason {
    /// A ledger value left its exact range.
    LedgerOverflow = 5,
    /// A time computation overflowed.
    TimeOverflow = 6,
    /// A sequence, counter or container overflowed.
    CounterOverflow = 7,
    /// The state belongs to another dataset or configuration.
    StateMismatch = 8,
    /// The pretrade exposure was not representable.
    GateNumeric = 10,
}

impl HaltReason {
    pub(super) fn code(self) -> u16 {
        match self {
            Self::LedgerOverflow => 5,
            Self::TimeOverflow => 6,
            Self::CounterOverflow => 7,
            Self::StateMismatch => 8,
            Self::GateNumeric => 10,
        }
    }

    pub(super) fn from_code(code: u16) -> Option<Self> {
        match code {
            5 => Some(Self::LedgerOverflow),
            6 => Some(Self::TimeOverflow),
            7 => Some(Self::CounterOverflow),
            8 => Some(Self::StateMismatch),
            10 => Some(Self::GateNumeric),
            _ => None,
        }
    }
}

/// Why an action was not admitted. Code 7 is retired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RejectReason {
    /// The run is over (or the state is not at a decision point), or the
    /// action's transaction failed and halted the run: it never took effect
    /// and [`StepOutcome::status`] gives the halt reason. When an admitted
    /// transaction halts, the action is still counted as an action, with no
    /// outcome recorded: `step` and `counters.actions` advance.
    Terminal = 1,
    /// A price is outside `min..=max_price` ticks.
    PriceRange = 2,
    /// A size is outside `1..=max_order_size` lots.
    SizeRange = 3,
    /// Not enough order slots for two more open orders.
    CapacityOrders = 4,
    /// Not enough venue message capacity for two more open orders.
    CapacityMsgs = 5,
    /// Not enough delivery capacity for two more open orders.
    CapacityDeliveries = 6,
    /// The strategy failed to produce a quote (recorded as a hold).
    QuoteError = 8,
}

impl RejectReason {
    /// Numeric code (1 to 8).
    #[must_use]
    pub fn code(self) -> u16 {
        match self {
            Self::Terminal => 1,
            Self::PriceRange => 2,
            Self::SizeRange => 3,
            Self::CapacityOrders => 4,
            Self::CapacityMsgs => 5,
            Self::CapacityDeliveries => 6,
            Self::QuoteError => 8,
        }
    }

    pub(super) fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Terminal),
            2 => Some(Self::PriceRange),
            3 => Some(Self::SizeRange),
            4 => Some(Self::CapacityOrders),
            5 => Some(Self::CapacityMsgs),
            6 => Some(Self::CapacityDeliveries),
            8 => Some(Self::QuoteError),
            _ => None,
        }
    }

    /// Index in [`Counters::rejected`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Terminal => 0,
            Self::PriceRange => 1,
            Self::SizeRange => 2,
            Self::CapacityOrders => 3,
            Self::CapacityMsgs => 4,
            Self::CapacityDeliveries => 5,
            Self::QuoteError => 7,
        }
    }
}

/// Run status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The run continues.
    Running,
    /// The run ended normally.
    Terminal(TermReason),
    /// `max_steps` decisions were taken.
    Truncated,
    /// The run halted.
    Halted(HaltReason),
}

impl KernelEvent {
    /// Returns true for the events the agent receives (deliveries and
    /// observations). Every other event is a venue-time diagnostic that an
    /// agent must not observe.
    #[must_use]
    pub fn is_agent_facing(&self) -> bool {
        match self {
            Self::Delivered(_) | Self::Observed { .. } => true,
            Self::Report(_)
            | Self::Fill(_)
            | Self::Gate(_)
            | Self::RowEnd { .. }
            | Self::EndOfData { .. }
            | Self::Halted(_)
            | Self::Decided(_)
            | Self::DroppedMsg { .. }
            | Self::Expired { .. }
            | Self::DroppedExpiry { .. } => false,
        }
    }
}

impl Status {
    /// Returns true while the run continues.
    #[must_use]
    pub fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

/// An error code carried by [`KernelAction::QuoteError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrCode(pub u16);

/// Side of a one-sided quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteSide {
    /// A buy order (bid).
    Bid,
    /// A sell order (ask).
    Ask,
}

/// One decision of the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelAction {
    /// Leave the resting orders in place.
    Hold,
    /// Cancel every resting or pending order and submit a bid and an ask.
    /// A crossed quote (`bid >= ask`) is admitted and rejected locally on
    /// both sides, as the engine does.
    Quote {
        /// Bid, in ticks.
        bid: u64,
        /// Ask, in ticks.
        ask: u64,
        /// Bid size, in lots.
        bid_lots: u64,
        /// Ask size, in lots.
        ask_lots: u64,
    },
    /// The strategy failed; recorded as a hold, with its code kept in the
    /// outcome, the events and the state.
    QuoteError(ErrCode),
    /// Cancel every resting or pending order and submit one order on
    /// `side`, for a strategy that may only trade one way (an inventory
    /// bound). Outside the engine-equivalence subset: the engine's quote is
    /// always two-sided. Admission is the same as for a `Quote`.
    OneSided {
        /// Side of the order.
        side: QuoteSide,
        /// Price, in ticks.
        price: u64,
        /// Size, in lots.
        lots: u64,
    },
    /// Cancel every resting or pending order and submit nothing. Each cancel
    /// takes effect after the cancel latency, as a quote's cancels do: a fill
    /// before it stands, and a cancel that reaches an order already filled is
    /// rejected (`CancelRejected`). Admitted whenever a `Hold` is, since an
    /// open order already holds the venue message its cancel needs. Outside
    /// the engine-equivalence subset: the engine cancels only to replace.
    Cancel,
    /// A [`KernelAction::Quote`] whose legs the venue cancels by itself
    /// `life_ns` of venue time after each leg's own activation, whatever the
    /// latencies and their jitter: an expiry the kernel schedules when the
    /// leg activates, in the venue queue, in `(time, sequence)` order with
    /// every other message. As for any order, a row at the activation time
    /// does not fill it; a row at or after the expiry finds it gone, since
    /// venue messages run before matching at equal times. So it fills only on
    /// rows strictly inside `(activation, activation + life_ns)`. A replacing
    /// quote's or a cancel's message and the expiry share the two venue
    /// messages an open order reserves: the activation's, which the expiry
    /// takes once the order is active, and the cancel's. Admission is a
    /// `Quote`'s. Outside the engine-equivalence subset.
    ExpiringQuote {
        /// Bid, in ticks.
        bid: u64,
        /// Ask, in ticks.
        ask: u64,
        /// Bid size, in lots.
        bid_lots: u64,
        /// Ask size, in lots.
        ask_lots: u64,
        /// Venue time each leg rests from its activation, in nanoseconds.
        life_ns: NonZeroU64,
    },
}

/// What happened to the action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    /// Admitted (a quote may still be rejected locally, through a report).
    Applied,
    /// Not admitted; nothing but the counters changed.
    Rejected(RejectReason),
    /// A [`KernelAction::QuoteError`]: recorded as a hold, counted in
    /// [`RejectReason::QuoteError`], with the strategy's code.
    QuoteError(ErrCode),
}

/// Result of one [`super::FinancialKernel::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepOutcome {
    /// What happened to the action.
    pub applied: ActionStatus,
    /// Status after the step.
    pub status: Status,
    /// Agent-facing events of the step: deliveries and observations. Venue
    /// events (reports at venue time, fills, gate records, row ends) are
    /// diagnostics and are not counted, so this never reveals undelivered
    /// venue progress.
    pub events: u64,
}

/// Run counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Step calls that took a decision (every call while running).
    pub actions: u64,
    /// Not admitted actions, by [`RejectReason::index`].
    pub rejected: [u64; 8],
    /// Halts (at most one).
    pub halts: u64,
    /// Venue messages dropped at the end of the data: activations, cancels
    /// and expiries.
    pub eof_dropped_msgs: u64,
    /// Observations dropped at the end of the data.
    pub eof_dropped_obs: u64,
    /// `QuoteError` actions.
    pub quote_errors: u64,
    /// Venue fills (venue time; never observed directly).
    pub venue_fills: u64,
    /// Local rejects (venue time; never observed directly).
    pub local_rejects: u64,
}

impl Counters {
    pub(super) const ZERO: Self = Self {
        actions: 0,
        rejected: [0; 8],
        halts: 0,
        eof_dropped_msgs: 0,
        eof_dropped_obs: 0,
        quote_errors: 0,
        venue_fills: 0,
        local_rejects: 0,
    };
}

/// A pretrade decision of the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GateRecord {
    /// Order id.
    pub order_id: u64,
    /// Side.
    pub side: Action,
    /// Requested lots.
    pub requested: u64,
    /// Verdict.
    pub verdict: GateVerdict,
    /// Submitted lots (0 when refused).
    pub submitted: u64,
    /// The check that decided.
    pub check: GateCheck,
}

/// An event of the run, in event order.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum KernelEvent {
    /// A report was produced (venue time); the engine records it here.
    Report(ReportRec),
    /// A fill was booked in the venue ledger.
    Fill(FillRec),
    /// The gate decided on an order.
    Gate(GateRecord),
    /// A report reached the agent.
    Delivered(ReportRec),
    /// An observation reached the agent: a decision point.
    Observed {
        /// Row observed.
        row: u64,
        /// Time of the decision, in nanoseconds.
        at_ns: u64,
    },
    /// A row was fully processed and the ledger marked at its mid.
    RowEnd {
        /// Row index.
        row: u64,
        /// Row time, in nanoseconds.
        ts_ns: u64,
        /// Equity at the row's mid, in half money units.
        equity_half_u: i128,
    },
    /// The data ended; pending messages and observations were dropped.
    EndOfData {
        /// Venue messages dropped.
        dropped_msgs: u64,
        /// Observations dropped.
        dropped_obs: u64,
    },
    /// The run halted.
    Halted(HaltReason),
    /// The outcome of a step's action (the agent's own, synchronous).
    Decided(ActionStatus),
    /// A venue message dropped at the end of the data.
    DroppedMsg {
        /// Order the message is for.
        order_id: u64,
        /// True for a cancel, false for an activation.
        cancel: bool,
    },
    /// The venue canceled an order of a [`KernelAction::ExpiringQuote`] at its
    /// expiry; its `Canceled` report follows.
    Expired {
        /// The order.
        order_id: u64,
        /// Venue time of the expiry, in nanoseconds.
        at_ns: u64,
    },
    /// An expiry still pending at the end of the data, dropped with the
    /// order's other venue messages.
    DroppedExpiry {
        /// The order.
        order_id: u64,
    },
}

/// Receiver of [`KernelEvent`]s. Implementations should not allocate per
/// event on hot paths; the kernel itself never allocates for events.
///
/// Only [`KernelEvent::is_agent_facing`] events may reach an agent; the
/// rest, and [`EventSink::queues`], are diagnostics.
pub trait EventSink {
    /// Receives one event.
    fn event(&mut self, event: KernelEvent);

    /// Queue lengths after every applied event (venue message, delivery,
    /// fill, observation push, action, end of data). A diagnostic for
    /// checking the queue bounds; ignored by default.
    fn queues(&mut self, _lengths: QueueLengths) {}

    /// The hedge lane's money events (#266): its rows, fills and funding
    /// payments, in the lane's own order. Ignored by default, so a sink of the
    /// quoted lane alone is unchanged; a wrapping sink must forward it.
    fn hedge(&mut self, _event: HedgeEvent) {}
}

/// A sink that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSink;

impl EventSink for NoSink {
    fn event(&mut self, _event: KernelEvent) {}
}

impl EventSink for Vec<KernelEvent> {
    fn event(&mut self, event: KernelEvent) {
        self.push(event);
    }
}

/// Lengths of the internal queues, for bound checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueLengths {
    /// Resting orders.
    pub orders: u64,
    /// Terminal orders awaiting a cancel outcome.
    pub awaiting: u64,
    /// Venue messages.
    pub msgs: u64,
    /// Undelivered reports.
    pub deliveries: u64,
    /// Observations in flight.
    pub observed: u64,
    /// Open orders in the agent account.
    pub held: u64,
}

/// The complete simulation state: a `Copy` value, so a copy is a snapshot.
///
/// The fields are private; [`super::FinancialKernel::reset`],
/// [`super::FinancialKernel::step`] and [`super::decode`] are the only ways
/// to build one. The in-memory layout is unstable; the canonical encoding
/// ([`super::encode`]) is the stable form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelState<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize> {
    pub(super) data_id: DatasetId,
    pub(super) cfg_fp: u128,
    pub(super) format_version: u16,
    pub(super) phase: Phase,
    /// Pass to resume after a decision (`BeforeMatch` or `ObsQueued`).
    pub(super) resume: Phase,
    /// Next row to read.
    pub(super) cursor: u64,
    /// Row being processed.
    pub(super) cur_row: Option<u64>,
    pub(super) now_ns: u64,
    pub(super) step: u64,
    pub(super) status: Status,
    pub(super) rng_latency: KernelRng,
    pub(super) rng_fill: KernelRng,
    pub(super) stamp_seq: u64,
    pub(super) next_msg_seq: u64,
    pub(super) next_order_id: u64,
    pub(super) next_exec_id: u64,
    pub(super) last_delivery_ns: u64,
    pub(super) last_observed_ns: u64,
    pub(super) orders: Arr<VenueOrder, ORD>,
    pub(super) msgs: Arr<VenueMsg, MSG>,
    pub(super) awaiting: Arr<VenueOrder, ORD>,
    pub(super) venue: VenueLedger,
    pub(super) gate: GateState,
    pub(super) deliveries: Arr<Delivery, DLV>,
    pub(super) observed: Arr<ObsEntry, OBS>,
    pub(super) agent: AgentView,
    pub(super) account: AgentAccount<ORD>,
    pub(super) last: Option<ActionStatus>,
    pub(super) counters: Counters,
}

impl<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    KernelState<ORD, MSG, DLV, OBS>
{
    /// Identity of the dataset the state belongs to.
    #[must_use]
    pub fn data_id(&self) -> DatasetId {
        self.data_id
    }

    /// Whether `other` equals this state in every field but the dataset's identity: the
    /// same run on rows that differ only where it has not read yet (#282).
    #[must_use]
    pub fn same_apart_from_dataset(&self, other: &Self) -> bool {
        let mut moved = *self;
        moved.data_id = other.data_id;
        moved == *other
    }

    /// Fingerprint of the configuration the state belongs to.
    #[must_use]
    pub fn cfg_fingerprint(&self) -> u128 {
        self.cfg_fp
    }

    /// Phase of the advance.
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Run status.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// Rows read so far.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Simulated time, in nanoseconds.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }

    /// Decisions taken.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Counters.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn counters(&self) -> Counters {
        self.counters
    }

    /// Outcome of the last action.
    #[must_use]
    pub fn last(&self) -> Option<ActionStatus> {
        self.last
    }

    /// Next event stamp sequence number.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn stamp_seq(&self) -> u64 {
        self.stamp_seq
    }

    /// Next venue message sequence number.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn next_msg_seq(&self) -> u64 {
        self.next_msg_seq
    }

    /// Next order id.
    #[must_use]
    pub fn next_order_id(&self) -> u64 {
        self.next_order_id
    }

    /// Next execution id.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn next_exec_id(&self) -> u64 {
        self.next_exec_id
    }

    /// Position the agent knows of (delivered fills).
    #[must_use]
    pub fn agent_position(&self) -> Decimal {
        self.agent.pos.quantity
    }

    /// Average entry the agent knows of.
    #[must_use]
    pub fn agent_avg_entry(&self) -> Decimal {
        self.agent.pos.avg_entry
    }

    /// Venue position (every fill), as the engine's `get_state`.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn venue_position(&self) -> Decimal {
        self.venue.pos.quantity
    }

    /// Venue average entry.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn venue_avg_entry(&self) -> Decimal {
        self.venue.pos.avg_entry
    }

    /// Venue realized cash flow, as the engine's `pnl.realized`.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn venue_realized(&self) -> Decimal {
        self.venue.realized
    }

    /// Venue position, in lots.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn venue_position_lots(&self) -> i64 {
        self.gate.venue_pos_lots
    }

    /// Live reserved lots per side `(buy, sell)`.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn live_lots(&self) -> (u64, u64) {
        (self.gate.live_buy_lots, self.gate.live_sell_lots)
    }

    /// Returns true while `order_id` is open in the agent account (submitted
    /// and its closing report not yet delivered).
    #[must_use]
    pub fn is_open(&self, order_id: u64) -> bool {
        self.account
            .as_slice()
            .iter()
            .any(|o| o.order_id == order_id)
    }

    /// Queue lengths.
    ///
    /// Diagnostic: reveals venue progress; an agent must not observe it.
    #[must_use]
    pub fn queue_lengths(&self) -> QueueLengths {
        let n = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
        QueueLengths {
            orders: n(self.orders.len()),
            awaiting: n(self.awaiting.len()),
            msgs: n(self.msgs.len()),
            deliveries: n(self.deliveries.len()),
            observed: n(self.observed.len()),
            held: n(self.account.len()),
        }
    }
}
