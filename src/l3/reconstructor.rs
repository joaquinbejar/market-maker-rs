//! Book reconstruction from a snapshot and a sequenced message stream.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use super::book::{Book, PriorityEvidence, Resting, exact_add, exact_sub};
use super::message::{ChangeReason, DoneReason, L3Event, L3Message, L3Side, L3Snapshot};
use crate::Decimal;

/// What a size decrease does to an order's place in its queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeDecrease {
    /// It keeps its place. A declared assumption (the usual price time
    /// rule), not a fact the feed documents.
    KeepsPriority,
    /// Its place afterwards is unknown: a queue estimate that depends on it
    /// is refused.
    PriorityUnknown,
}

/// The venue rules and limits the reconstruction runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VenueRules {
    /// What a size decrease does to priority.
    pub size_decrease: SizeDecrease,
    /// Most resting orders held.
    pub max_orders: usize,
    /// Most price levels held, both sides together.
    pub max_levels: usize,
}

impl Default for VenueRules {
    fn default() -> Self {
        Self {
            size_decrease: SizeDecrease::KeepsPriority,
            max_orders: 1_000_000,
            max_levels: 100_000,
        }
    }
}

/// A message inconsistent with the book: the book can no longer be trusted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Discrepancy {
    /// A match against a maker that does not rest on the book.
    #[error("match against maker {maker_id}, which is not on the book")]
    UnknownMaker {
        /// The maker.
        maker_id: String,
    },
    /// A match whose taker rests on the book.
    #[error("match whose taker {taker_id} rests on the book")]
    TakerResting {
        /// The taker.
        taker_id: String,
    },
    /// A match whose side or price differs from the maker's.
    #[error("match side or price differs from maker {maker_id}")]
    MatchMismatch {
        /// The maker.
        maker_id: String,
    },
    /// A match larger than the maker's remaining size.
    #[error("match of {size} exceeds maker {maker_id}'s remaining {remaining}")]
    MatchExceedsRemaining {
        /// The maker.
        maker_id: String,
        /// Its remaining size.
        remaining: Decimal,
        /// The match size.
        size: Decimal,
    },
    /// An `open` for an order already resting.
    #[error("open of {order_id}, already on the book")]
    DuplicateOpen {
        /// The order.
        order_id: String,
    },
    /// A `done` filled for an order still resting with size: a match is
    /// missing.
    #[error("done filled for {order_id} with {remaining} still resting")]
    MissingMatch {
        /// The order.
        order_id: String,
        /// Its tracked remaining size.
        remaining: Decimal,
    },
    /// A `done` canceled whose remaining size differs from the tracked one.
    #[error("done canceled for {order_id} reports {reported:?}, tracked {tracked}")]
    RemainingMismatch {
        /// The order.
        order_id: String,
        /// Tracked.
        tracked: Decimal,
        /// Reported.
        reported: Option<Decimal>,
    },
    /// A `done` whose side or price differs from the order's.
    #[error("done side or price differs from order {order_id}")]
    DoneMismatch {
        /// The order.
        order_id: String,
    },
    /// A `change` of a resting order without the fields it needs.
    #[error("malformed change of {order_id}: {reason}")]
    MalformedChange {
        /// The order.
        order_id: String,
        /// What is wrong.
        reason: String,
    },
    /// A `change` whose old values or side differ from the order's.
    #[error("change old values or side differ from order {order_id}")]
    ChangeMismatch {
        /// The order.
        order_id: String,
    },
    /// A modification this reconstruction does not support: a price change,
    /// a size increase, or any `modify_order`. Its effect on priority is not
    /// documented, so none is invented.
    #[error("unsupported modification of {order_id}")]
    UnsupportedModification {
        /// The order.
        order_id: String,
    },
    /// The visible book would cross.
    #[error("book would cross: bid {bid} >= ask {ask}")]
    CrossedAfterApply {
        /// Best bid after.
        bid: Decimal,
        /// Best ask after.
        ask: Decimal,
    },
    /// A price or size that is not positive.
    #[error("invalid value: {what}")]
    InvalidValue {
        /// What.
        what: String,
    },
    /// The book would exceed [`VenueRules`] capacity.
    #[error("capacity exceeded: {what}")]
    CapacityExceeded {
        /// What.
        what: String,
    },
    /// Checked decimal arithmetic failed.
    #[error("checked arithmetic failed: {what}")]
    Arithmetic {
        /// What.
        what: String,
    },
    /// A message type the reconstruction does not handle.
    #[error("unsupported message type {kind}")]
    UnsupportedInput {
        /// The type.
        kind: String,
    },
}

/// Why a segment stopped being valid.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidationKind {
    /// A sequence was skipped.
    #[error("gap: expected {expected}, got {got}")]
    Gap {
        /// The sequence expected.
        expected: u64,
        /// The one received.
        got: u64,
    },
    /// A sequence at or before the last applied one.
    #[error("sequence {got} at or before the last applied {last}")]
    OutOfOrderOrDuplicate {
        /// The last applied.
        last: u64,
        /// The one received.
        got: u64,
    },
    /// The next sequence does not fit in `u64`.
    #[error("the next sequence overflows u64")]
    SequenceOverflow,
    /// A message inconsistent with the book.
    #[error("discrepancy: {0}")]
    Discrepancy(Discrepancy),
}

/// The end of a valid segment.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalidated at sequence {first_offending_seq} (last good {last_good_seq}): {kind}")]
pub struct Invalidation {
    /// The last sequence applied while valid.
    pub last_good_seq: u64,
    /// The sequence of the offending message.
    pub first_offending_seq: u64,
    /// Why.
    pub kind: InvalidationKind,
}

/// Why a snapshot was refused; the reconstructor's state is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Another product's snapshot.
    #[error("snapshot of {got}, expected {expected}")]
    WrongProduct {
        /// Expected.
        expected: String,
        /// Received.
        got: String,
    },
    /// The same id twice.
    #[error("snapshot repeats order id {0}")]
    DuplicateId(String),
    /// A price or size that is not positive.
    #[error("snapshot value not positive: {0}")]
    InvalidValue(String),
    /// Bids at or above asks.
    #[error("snapshot crossed: bid {bid} >= ask {ask}")]
    Crossed {
        /// Best bid.
        bid: Decimal,
        /// Best ask.
        ask: Decimal,
    },
    /// More orders or levels than [`VenueRules`] allows.
    #[error("snapshot capacity exceeded: {0}")]
    CapacityExceeded(String),
    /// Checked decimal arithmetic failed.
    #[error("snapshot checked arithmetic failed: {0}")]
    Arithmetic(String),
}

/// Why a message was not applied.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ApplyError {
    /// Another product's message: refused, the segment untouched.
    #[error("message of {got}, expected {expected}")]
    WrongProduct {
        /// Expected.
        expected: String,
        /// Received.
        got: String,
    },
    /// No snapshot has been applied yet.
    #[error("no snapshot applied yet")]
    NotBootstrapped,
    /// This message ended the segment.
    #[error("{0}")]
    Invalidated(Invalidation),
    /// The segment had already ended; the message was skipped. Recovery is
    /// only through [`BookReconstructor::bootstrap`].
    #[error("awaiting a snapshot after: {0}")]
    AwaitingSnapshot(Invalidation),
}

/// Why a view is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ViewError {
    /// No snapshot has been applied yet.
    #[error("no snapshot applied yet")]
    NotBootstrapped,
    /// The book is invalid until the next snapshot.
    #[error("book invalid until the next snapshot: {0}")]
    Invalidated(Invalidation),
    /// Checked decimal arithmetic failed.
    #[error("checked arithmetic failed: {0}")]
    Arithmetic(String),
}

/// What an applied message did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    /// It changed the visible book.
    BookChanged,
    /// It concerns no resting order (a `received`, or a `done` or `change`
    /// of an order not on the book), or changed nothing.
    NoBookEffect,
}

/// How a segment ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentEnd {
    /// Still valid.
    Open,
    /// Invalidated.
    Invalidated(Invalidation),
    /// Replaced by a new snapshot while still valid.
    Superseded,
}

/// A run of messages applied on one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The snapshot's sequence.
    pub start_seq: u64,
    /// The last sequence applied while valid.
    pub last_good_seq: u64,
    /// How it ended.
    pub end: SegmentEnd,
}

/// Messages counted by what they did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MessageCounts {
    /// `received`.
    pub received: usize,
    /// `open` applied.
    pub open: usize,
    /// `done` that removed a resting order.
    pub done: usize,
    /// `match` applied.
    pub matched: usize,
    /// Size decreases applied (`change` STP).
    pub stp_decrease: usize,
    /// `done` of an order not on the book.
    pub off_book_done: usize,
    /// `change` of an order not on the book.
    pub off_book_change: usize,
    /// `change` that left the size as it was.
    pub noop_change: usize,
    /// Messages dropped by the snapshot handshake (sequence at or before
    /// the snapshot's).
    pub handshake_dropped: usize,
    /// Messages skipped while waiting for a snapshot.
    pub skipped_while_invalid: usize,
    /// Messages of another product, refused.
    pub wrong_product: usize,
    /// Snapshots refused.
    pub rejected_snapshots: usize,
}

/// The reconstruction's quality record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityManifest {
    /// Product.
    pub product_id: String,
    /// The rules in force.
    pub rules: VenueRules,
    /// Every segment, in order.
    pub segments: Vec<Segment>,
    /// Counters.
    pub counts: MessageCounts,
}

/// Total and unknown priority volume of one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelProfile {
    /// Price.
    pub price: Decimal,
    /// Total visible size.
    pub total: Decimal,
    /// Size of the orders with [`PriorityEvidence::Unknown`].
    pub unknown_priority: Decimal,
}

/// One order of a level's queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueEntry {
    /// Order id.
    pub order_id: String,
    /// Remaining size.
    pub remaining: Decimal,
    /// Priority evidence.
    pub evidence: PriorityEvidence,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    Empty,
    Valid {
        book: Book,
        last_seq: u64,
    },
    Invalidated {
        stale: Book,
        invalidation: Invalidation,
    },
}

/// What a valid message will do, computed before anything changes.
enum Plan {
    NoEffect(Counter),
    Insert {
        id: String,
        resting: Resting,
        new_total: Decimal,
    },
    Reduce {
        id: String,
        remaining: Decimal,
        new_total: Decimal,
        decrease: bool,
        counter: Counter,
    },
    Remove {
        id: String,
        new_total: Decimal,
        counter: Counter,
    },
}

#[derive(Clone, Copy)]
enum Counter {
    Received,
    Open,
    Done,
    Matched,
    Stp,
    OffBookDone,
    OffBookChange,
    NoopChange,
}

/// Rebuilds the visible book of one product from a snapshot and its
/// sequenced messages, failing closed on any gap or inconsistency.
///
/// See the [module docs](super) for the contract.
///
/// A clone is a separate reconstructor: queue estimates of one are not
/// valid on the other.
#[derive(Debug)]
pub struct BookReconstructor {
    product_id: String,
    rules: VenueRules,
    state: State,
    manifest: QualityManifest,
    /// Identity of the current segment, unique in the process: a queue
    /// estimate is valid only on the segment that made it.
    segment_id: u64,
}

/// Source of segment identities.
static NEXT_SEGMENT_ID: AtomicU64 = AtomicU64::new(1);

fn next_segment_id() -> u64 {
    NEXT_SEGMENT_ID.fetch_add(1, Ordering::Relaxed)
}

impl Clone for BookReconstructor {
    fn clone(&self) -> Self {
        Self {
            product_id: self.product_id.clone(),
            rules: self.rules,
            state: self.state.clone(),
            manifest: self.manifest.clone(),
            segment_id: next_segment_id(),
        }
    }
}

fn positive(what: &str, v: Decimal) -> Result<(), Discrepancy> {
    if v > Decimal::ZERO {
        Ok(())
    } else {
        Err(Discrepancy::InvalidValue {
            what: format!("{what} {v} is not positive"),
        })
    }
}

fn arithmetic(what: &str) -> Discrepancy {
    Discrepancy::Arithmetic {
        what: what.to_string(),
    }
}

impl BookReconstructor {
    /// A reconstructor for `product_id`, waiting for a snapshot.
    #[must_use]
    pub fn new(product_id: impl Into<String>, rules: VenueRules) -> Self {
        let product_id = product_id.into();
        Self {
            manifest: QualityManifest {
                product_id: product_id.clone(),
                rules,
                segments: Vec::new(),
                counts: MessageCounts::default(),
            },
            product_id,
            rules,
            state: State::Empty,
            segment_id: next_segment_id(),
        }
    }

    /// The quality record.
    #[must_use]
    pub fn manifest(&self) -> &QualityManifest {
        &self.manifest
    }

    /// The invalidation in force, if the book is waiting for a snapshot.
    #[must_use]
    pub fn invalidation(&self) -> Option<&Invalidation> {
        match &self.state {
            State::Invalidated { invalidation, .. } => Some(invalidation),
            _ => None,
        }
    }

    /// Replaces the book with a validated snapshot and starts a segment.
    ///
    /// # Errors
    ///
    /// A [`SnapshotError`]; the state is then unchanged.
    pub fn bootstrap(&mut self, snapshot: &L3Snapshot) -> Result<(), SnapshotError> {
        let book = match self.validate_snapshot(snapshot) {
            Ok(book) => book,
            Err(error) => {
                self.manifest.counts.rejected_snapshots =
                    self.manifest.counts.rejected_snapshots.saturating_add(1);
                return Err(error);
            }
        };
        if matches!(self.state, State::Valid { .. })
            && let Some(segment) = self.manifest.segments.last_mut()
        {
            segment.end = SegmentEnd::Superseded;
        }
        self.manifest.segments.push(Segment {
            start_seq: snapshot.sequence,
            last_good_seq: snapshot.sequence,
            end: SegmentEnd::Open,
        });
        self.state = State::Valid {
            book,
            last_seq: snapshot.sequence,
        };
        self.segment_id = next_segment_id();
        Ok(())
    }

    /// The snapshot handshake: bootstraps on `snapshot`, drops the buffered
    /// messages at or before its sequence, and applies the rest in order.
    /// The first one applied must follow the snapshot's sequence, or the
    /// segment ends in a gap.
    ///
    /// Another product's messages are refused and counted, as by
    /// [`Self::apply`], and the handshake goes on.
    ///
    /// # Errors
    ///
    /// The snapshot's error, as [`HandshakeError::Snapshot`], or the first
    /// message that could not be applied, as [`HandshakeError::Apply`];
    /// the messages after it are not applied.
    pub fn bootstrap_with_buffer<'a>(
        &mut self,
        snapshot: &L3Snapshot,
        buffered: impl IntoIterator<Item = &'a L3Message>,
    ) -> Result<usize, HandshakeError> {
        self.bootstrap(snapshot).map_err(HandshakeError::Snapshot)?;
        let mut applied = 0_usize;
        for message in buffered {
            if message.product_id == self.product_id && message.sequence <= snapshot.sequence {
                self.manifest.counts.handshake_dropped =
                    self.manifest.counts.handshake_dropped.saturating_add(1);
                continue;
            }
            match self.apply(message) {
                Ok(_) => applied = applied.saturating_add(1),
                Err(ApplyError::WrongProduct { .. }) => {}
                Err(error) => return Err(HandshakeError::Apply(error)),
            }
        }
        Ok(applied)
    }

    /// Applies the next message. Nothing changes unless every check passes.
    ///
    /// # Errors
    ///
    /// An [`ApplyError`]: another product's message is refused with the
    /// segment untouched; a gap, an out of order sequence or a discrepancy
    /// ends the segment ([`ApplyError::Invalidated`]), and later messages are
    /// skipped until the next snapshot.
    pub fn apply(&mut self, message: &L3Message) -> Result<Applied, ApplyError> {
        if message.product_id != self.product_id {
            self.manifest.counts.wrong_product =
                self.manifest.counts.wrong_product.saturating_add(1);
            return Err(ApplyError::WrongProduct {
                expected: self.product_id.clone(),
                got: message.product_id.clone(),
            });
        }
        let (book, last_seq) = match &self.state {
            State::Empty => return Err(ApplyError::NotBootstrapped),
            State::Invalidated { invalidation, .. } => {
                self.manifest.counts.skipped_while_invalid =
                    self.manifest.counts.skipped_while_invalid.saturating_add(1);
                return Err(ApplyError::AwaitingSnapshot(invalidation.clone()));
            }
            State::Valid { book, last_seq } => (book, *last_seq),
        };
        let seq = message.sequence;
        let plan = match last_seq.checked_add(1) {
            None => Err(InvalidationKind::SequenceOverflow),
            Some(expected) if seq == expected => self
                .plan(book, seq, &message.event)
                .map_err(InvalidationKind::Discrepancy),
            Some(_) if seq <= last_seq => Err(InvalidationKind::OutOfOrderOrDuplicate {
                last: last_seq,
                got: seq,
            }),
            Some(expected) => Err(InvalidationKind::Gap { expected, got: seq }),
        };
        match plan {
            Ok(plan) => Ok(self.commit(plan, seq)),
            Err(kind) => {
                let invalidation = Invalidation {
                    last_good_seq: last_seq,
                    first_offending_seq: seq,
                    kind,
                };
                self.invalidate(invalidation.clone());
                Err(ApplyError::Invalidated(invalidation))
            }
        }
    }

    fn invalidate(&mut self, invalidation: Invalidation) {
        let state = std::mem::replace(&mut self.state, State::Empty);
        let stale = match state {
            State::Valid { book, .. } | State::Invalidated { stale: book, .. } => book,
            State::Empty => Book::default(),
        };
        if let Some(segment) = self.manifest.segments.last_mut() {
            segment.end = SegmentEnd::Invalidated(invalidation.clone());
        }
        self.state = State::Invalidated {
            stale,
            invalidation,
        };
    }

    fn commit(&mut self, plan: Plan, seq: u64) -> Applied {
        let State::Valid { book, last_seq } = &mut self.state else {
            return Applied::NoBookEffect;
        };
        *last_seq = seq;
        if let Some(segment) = self.manifest.segments.last_mut() {
            segment.last_good_seq = seq;
        }
        let counts = &mut self.manifest.counts;
        let count = |counter: Counter, counts: &mut MessageCounts| {
            let slot = match counter {
                Counter::Received => &mut counts.received,
                Counter::Open => &mut counts.open,
                Counter::Done => &mut counts.done,
                Counter::Matched => &mut counts.matched,
                Counter::Stp => &mut counts.stp_decrease,
                Counter::OffBookDone => &mut counts.off_book_done,
                Counter::OffBookChange => &mut counts.off_book_change,
                Counter::NoopChange => &mut counts.noop_change,
            };
            *slot = slot.saturating_add(1);
        };
        match plan {
            Plan::NoEffect(counter) => {
                count(counter, counts);
                Applied::NoBookEffect
            }
            Plan::Insert {
                id,
                resting,
                new_total,
            } => {
                book.insert(id, resting, new_total);
                count(Counter::Open, counts);
                Applied::BookChanged
            }
            Plan::Reduce {
                id,
                remaining,
                new_total,
                decrease,
                counter,
            } => {
                book.reduce(&id, remaining, new_total, decrease.then_some(seq));
                count(counter, counts);
                Applied::BookChanged
            }
            Plan::Remove {
                id,
                new_total,
                counter,
            } => {
                book.remove(&id, new_total);
                count(counter, counts);
                Applied::BookChanged
            }
        }
    }

    /// Checks a message against the book and computes its effect, changing
    /// nothing.
    fn plan(&self, book: &Book, seq: u64, event: &L3Event) -> Result<Plan, Discrepancy> {
        match event {
            L3Event::Received { .. } => Ok(Plan::NoEffect(Counter::Received)),
            L3Event::Open {
                order_id,
                side,
                price,
                remaining,
            } => self.plan_open(book, seq, order_id, *side, *price, *remaining),
            L3Event::Done {
                order_id,
                side,
                price,
                reason,
                remaining,
            } => Self::plan_done(book, order_id, *side, *price, *reason, *remaining),
            L3Event::Match {
                maker_id,
                taker_id,
                side,
                price,
                size,
            } => Self::plan_match(book, maker_id, taker_id, *side, *price, *size),
            L3Event::Change {
                order_id,
                side,
                reason,
                price,
                old_size,
                new_size,
                old_price,
                new_price,
                new_funds,
            } => Self::plan_change(
                book,
                order_id,
                *side,
                *reason,
                ChangeFields {
                    price: *price,
                    old_size: *old_size,
                    new_size: *new_size,
                    old_price: *old_price,
                    new_price: *new_price,
                    new_funds: *new_funds,
                },
            ),
            L3Event::Unsupported { kind } => {
                Err(Discrepancy::UnsupportedInput { kind: kind.clone() })
            }
        }
    }

    fn plan_open(
        &self,
        book: &Book,
        seq: u64,
        order_id: &str,
        side: L3Side,
        price: Decimal,
        remaining: Decimal,
    ) -> Result<Plan, Discrepancy> {
        if book.orders.contains_key(order_id) {
            return Err(Discrepancy::DuplicateOpen {
                order_id: order_id.to_string(),
            });
        }
        positive("open price", price)?;
        positive("open size", remaining)?;
        let crossed = match side {
            L3Side::Buy => book
                .best_ask()
                .filter(|ask| price >= *ask)
                .map(|ask| (price, ask)),
            L3Side::Sell => book
                .best_bid()
                .filter(|bid| *bid >= price)
                .map(|bid| (bid, price)),
        };
        if let Some((bid, ask)) = crossed {
            return Err(Discrepancy::CrossedAfterApply { bid, ask });
        }
        if book.orders.len() >= self.rules.max_orders {
            return Err(Discrepancy::CapacityExceeded {
                what: format!("{} orders", self.rules.max_orders),
            });
        }
        let level = book.levels(side).get(&price);
        if level.is_none() && book.level_count() >= self.rules.max_levels {
            return Err(Discrepancy::CapacityExceeded {
                what: format!("{} levels", self.rules.max_levels),
            });
        }
        let new_total = exact_add(level.map_or(Decimal::ZERO, |l| l.total), remaining)
            .ok_or_else(|| arithmetic("level total is not exact"))?;
        Ok(Plan::Insert {
            id: order_id.to_string(),
            resting: Resting {
                side,
                price,
                remaining,
                evidence: PriorityEvidence::Observed { sequence: seq },
                entered_at: seq,
                decreased_at: None,
            },
            new_total,
        })
    }

    fn level_total_less(
        book: &Book,
        order: &Resting,
        less: Decimal,
    ) -> Result<Decimal, Discrepancy> {
        let total = book
            .levels(order.side)
            .get(&order.price)
            .map(|l| l.total)
            .ok_or_else(|| arithmetic("resting order without its level"))?;
        let new_total =
            exact_sub(total, less).ok_or_else(|| arithmetic("level total is not exact"))?;
        if new_total < Decimal::ZERO {
            return Err(arithmetic("level total below zero"));
        }
        Ok(new_total)
    }

    fn plan_done(
        book: &Book,
        order_id: &str,
        side: L3Side,
        price: Option<Decimal>,
        reason: DoneReason,
        remaining: Option<Decimal>,
    ) -> Result<Plan, Discrepancy> {
        let Some(order) = book.orders.get(order_id) else {
            // Not on the book (received and never opened, or already fully
            // matched): no visible effect.
            return Ok(Plan::NoEffect(Counter::OffBookDone));
        };
        if order.side != side || price.is_some_and(|p| p != order.price) {
            return Err(Discrepancy::DoneMismatch {
                order_id: order_id.to_string(),
            });
        }
        match reason {
            // A resting order always has size: a fill would have removed it
            // through its matches.
            DoneReason::Filled => Err(Discrepancy::MissingMatch {
                order_id: order_id.to_string(),
                remaining: order.remaining,
            }),
            DoneReason::Canceled if remaining != Some(order.remaining) => {
                Err(Discrepancy::RemainingMismatch {
                    order_id: order_id.to_string(),
                    tracked: order.remaining,
                    reported: remaining,
                })
            }
            DoneReason::Canceled => Ok(Plan::Remove {
                id: order_id.to_string(),
                new_total: Self::level_total_less(book, order, order.remaining)?,
                counter: Counter::Done,
            }),
        }
    }

    fn plan_match(
        book: &Book,
        maker_id: &str,
        taker_id: &str,
        side: L3Side,
        price: Decimal,
        size: Decimal,
    ) -> Result<Plan, Discrepancy> {
        if book.orders.contains_key(taker_id) {
            return Err(Discrepancy::TakerResting {
                taker_id: taker_id.to_string(),
            });
        }
        let Some(maker) = book.orders.get(maker_id) else {
            return Err(Discrepancy::UnknownMaker {
                maker_id: maker_id.to_string(),
            });
        };
        if maker.side != side || maker.price != price {
            return Err(Discrepancy::MatchMismatch {
                maker_id: maker_id.to_string(),
            });
        }
        positive("match size", size)?;
        if size > maker.remaining {
            return Err(Discrepancy::MatchExceedsRemaining {
                maker_id: maker_id.to_string(),
                remaining: maker.remaining,
                size,
            });
        }
        let remaining = exact_sub(maker.remaining, size)
            .ok_or_else(|| arithmetic("maker remaining is not exact"))?;
        let new_total = Self::level_total_less(book, maker, size)?;
        Ok(if remaining == Decimal::ZERO {
            Plan::Remove {
                id: maker_id.to_string(),
                new_total,
                counter: Counter::Matched,
            }
        } else {
            Plan::Reduce {
                id: maker_id.to_string(),
                remaining,
                new_total,
                decrease: false,
                counter: Counter::Matched,
            }
        })
    }

    fn plan_change(
        book: &Book,
        order_id: &str,
        side: L3Side,
        reason: ChangeReason,
        f: ChangeFields,
    ) -> Result<Plan, Discrepancy> {
        let Some(order) = book.orders.get(order_id) else {
            // Not on the book (received and never opened, such as a market
            // order's funds): no visible effect.
            return Ok(Plan::NoEffect(Counter::OffBookChange));
        };
        let id = || order_id.to_string();
        if reason == ChangeReason::ModifyOrder {
            return Err(Discrepancy::UnsupportedModification { order_id: id() });
        }
        if order.side != side
            || f.price.is_some_and(|p| p != order.price)
            || f.old_size.is_some_and(|s| s != order.remaining)
            || f.old_price.is_some_and(|p| p != order.price)
        {
            return Err(Discrepancy::ChangeMismatch { order_id: id() });
        }
        if f.new_price.is_some_and(|p| p != order.price) {
            return Err(Discrepancy::UnsupportedModification { order_id: id() });
        }
        let Some(new_size) = f.new_size else {
            return Err(Discrepancy::MalformedChange {
                order_id: id(),
                reason: if f.new_funds.is_some() {
                    "funds change for a resting limit order".to_string()
                } else {
                    "no new size".to_string()
                },
            });
        };
        if new_size < Decimal::ZERO {
            return Err(Discrepancy::InvalidValue {
                what: format!("new size {new_size} is negative"),
            });
        }
        if new_size > order.remaining {
            return Err(Discrepancy::UnsupportedModification { order_id: id() });
        }
        if new_size == order.remaining {
            return Ok(Plan::NoEffect(Counter::NoopChange));
        }
        let less = exact_sub(order.remaining, new_size)
            .ok_or_else(|| arithmetic("size decrease is not exact"))?;
        let new_total = Self::level_total_less(book, order, less)?;
        Ok(if new_size == Decimal::ZERO {
            Plan::Remove {
                id: id(),
                new_total,
                counter: Counter::Stp,
            }
        } else {
            Plan::Reduce {
                id: id(),
                remaining: new_size,
                new_total,
                decrease: true,
                counter: Counter::Stp,
            }
        })
    }

    fn validate_snapshot(&self, snapshot: &L3Snapshot) -> Result<Book, SnapshotError> {
        if snapshot.product_id != self.product_id {
            return Err(SnapshotError::WrongProduct {
                expected: self.product_id.clone(),
                got: snapshot.product_id.clone(),
            });
        }
        let count = snapshot.bids.len().saturating_add(snapshot.asks.len());
        if count > self.rules.max_orders {
            return Err(SnapshotError::CapacityExceeded(format!(
                "{count} orders above {}",
                self.rules.max_orders
            )));
        }
        let mut ids = HashSet::with_capacity(count);
        let mut book = Book::default();
        for (side, orders) in [
            (L3Side::Buy, &snapshot.bids),
            (L3Side::Sell, &snapshot.asks),
        ] {
            for o in orders {
                if !ids.insert(o.order_id.as_str()) {
                    return Err(SnapshotError::DuplicateId(o.order_id.clone()));
                }
                if o.price <= Decimal::ZERO || o.size <= Decimal::ZERO {
                    return Err(SnapshotError::InvalidValue(format!(
                        "order {}: price {} size {}",
                        o.order_id, o.price, o.size
                    )));
                }
                let level = book.levels(side).get(&o.price);
                if level.is_none() && book.level_count() >= self.rules.max_levels {
                    return Err(SnapshotError::CapacityExceeded(format!(
                        "more than {} levels",
                        self.rules.max_levels
                    )));
                }
                let new_total = exact_add(level.map_or(Decimal::ZERO, |l| l.total), o.size)
                    .ok_or_else(|| {
                        SnapshotError::Arithmetic(format!("level {} total is not exact", o.price))
                    })?;
                book.insert(
                    o.order_id.clone(),
                    Resting {
                        side,
                        price: o.price,
                        remaining: o.size,
                        evidence: PriorityEvidence::Unknown,
                        entered_at: snapshot.sequence,
                        decreased_at: None,
                    },
                    new_total,
                );
            }
        }
        if let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask())
            && bid >= ask
        {
            return Err(SnapshotError::Crossed { bid, ask });
        }
        Ok(book)
    }

    pub(crate) fn valid_book(&self) -> Result<(&Book, u64), ViewError> {
        match &self.state {
            State::Valid { book, last_seq } => Ok((book, *last_seq)),
            State::Empty => Err(ViewError::NotBootstrapped),
            State::Invalidated { invalidation, .. } => {
                Err(ViewError::Invalidated(invalidation.clone()))
            }
        }
    }

    pub(crate) fn rules(&self) -> VenueRules {
        self.rules
    }

    pub(crate) fn segment_id(&self) -> u64 {
        self.segment_id
    }

    /// The last sequence applied.
    ///
    /// # Errors
    ///
    /// A [`ViewError`] unless the book is valid.
    pub fn last_sequence(&self) -> Result<u64, ViewError> {
        self.valid_book().map(|(_, seq)| seq)
    }

    /// Best bid and ask.
    ///
    /// # Errors
    ///
    /// A [`ViewError`] unless the book is valid.
    pub fn best(&self) -> Result<(Option<Decimal>, Option<Decimal>), ViewError> {
        self.valid_book().map(|(b, _)| (b.best_bid(), b.best_ask()))
    }

    /// Aggregated size per price, best first.
    ///
    /// # Errors
    ///
    /// A [`ViewError`] unless the book is valid.
    pub fn l2(&self, side: L3Side) -> Result<Vec<(Decimal, Decimal)>, ViewError> {
        let (book, _) = self.valid_book()?;
        Ok(Self::l2_of(book, side))
    }

    fn l2_of(book: &Book, side: L3Side) -> Vec<(Decimal, Decimal)> {
        let levels = book.levels(side).iter().map(|(p, l)| (*p, l.total));
        match side {
            L3Side::Buy => levels.rev().collect(),
            L3Side::Sell => levels.collect(),
        }
    }

    /// The last good book's aggregated sizes while invalidated: a stale
    /// diagnostic, never a valid view.
    #[must_use]
    pub fn stale_l2(&self, side: L3Side) -> Option<Vec<(Decimal, Decimal)>> {
        match &self.state {
            State::Invalidated { stale, .. } => Some(Self::l2_of(stale, side)),
            _ => None,
        }
    }

    /// The orders of one level in queue order.
    ///
    /// # Errors
    ///
    /// A [`ViewError`] unless the book is valid. An empty level is an empty
    /// queue.
    pub fn level_queue(&self, side: L3Side, price: Decimal) -> Result<Vec<QueueEntry>, ViewError> {
        let (book, _) = self.valid_book()?;
        Ok(book
            .levels(side)
            .get(&price)
            .map(|level| {
                level
                    .queue
                    .iter()
                    .filter_map(|id| {
                        book.orders.get(id).map(|o| QueueEntry {
                            order_id: id.clone(),
                            remaining: o.remaining,
                            evidence: o.evidence,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Per level, best first: total size and the size of unknown priority.
    ///
    /// # Errors
    ///
    /// A [`ViewError`] unless the book is valid, or
    /// [`ViewError::Arithmetic`] if a checked sum fails.
    pub fn priority_profile(&self, side: L3Side) -> Result<Vec<LevelProfile>, ViewError> {
        let (book, _) = self.valid_book()?;
        let profile = |(price, level): (&Decimal, &super::book::Level)| {
            let unknown_priority = level
                .queue
                .iter()
                .filter_map(|id| book.orders.get(id))
                .filter(|o| o.evidence == PriorityEvidence::Unknown)
                .try_fold(Decimal::ZERO, |acc, o| exact_add(acc, o.remaining))
                .ok_or_else(|| ViewError::Arithmetic(format!("unknown volume at {price}")))?;
            Ok(LevelProfile {
                price: *price,
                total: level.total,
                unknown_priority,
            })
        };
        let levels = book.levels(side).iter().map(profile);
        match side {
            L3Side::Buy => levels.rev().collect(),
            L3Side::Sell => levels.collect(),
        }
    }

    /// Recomputes every invariant of the valid book: positive sizes, each
    /// id once, levels equal to the exact sum of their orders, not crossed.
    ///
    /// # Errors
    ///
    /// The broken invariant, or a [`ViewError`] reason, described.
    pub fn audit(&self) -> Result<(), String> {
        let (book, _) = self.valid_book().map_err(|e| format!("{e:?}"))?;
        book.audit()
    }
}

struct ChangeFields {
    price: Option<Decimal>,
    old_size: Option<Decimal>,
    new_size: Option<Decimal>,
    old_price: Option<Decimal>,
    new_price: Option<Decimal>,
    new_funds: Option<Decimal>,
}

/// Why the snapshot handshake stopped.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HandshakeError {
    /// The snapshot was refused.
    #[error("{0}")]
    Snapshot(SnapshotError),
    /// A buffered message could not be applied.
    #[error("{0}")]
    Apply(ApplyError),
}
