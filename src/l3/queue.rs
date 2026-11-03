//! Volume ahead of an own order, under declared hypotheses.
//!
//! This is queue estimation, kept apart from the visible book: it rests on
//! hypotheses, which every result lists, and it validates no fill model.

use thiserror::Error;

use super::book::{PriorityEvidence, exact_add};
use super::message::L3Side;
use super::reconstructor::{BookReconstructor, SizeDecrease, ViewError};
use crate::Decimal;

/// A hypothesis a queue result rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assumption {
    /// A hypothetical order created now joins the back of the visible queue
    /// at its price: every order resting there now is ahead of it, every
    /// later one behind.
    BackOfVisibleQueueAtCreation,
    /// Orders keep their entry order: an order opened before ours at its
    /// price is ahead of it.
    EntrySequenceIsPriority,
    /// An order ahead decreased its size and kept its place
    /// ([`SizeDecrease::KeepsPriority`]).
    SizeDecreaseKeepsPriority,
}

/// Why a queue result is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QueueError {
    /// The book itself is unavailable.
    #[error("{0}")]
    View(ViewError),
    /// The estimate was made on another reconstructor, or on an earlier
    /// segment of this one (a snapshot has replaced the book since).
    #[error("estimate from another reconstructor or an earlier segment")]
    StaleEstimate,
    /// An order's place is unknown: our own order came from a snapshot, or
    /// it or an order ahead decreased its size under
    /// [`SizeDecrease::PriorityUnknown`].
    #[error("priority of {order_id} unknown")]
    PriorityUnknown {
        /// The order.
        order_id: String,
    },
    /// Our own order does not rest on the book.
    #[error("own order {order_id} is not on the book")]
    NotResting {
        /// The order.
        order_id: String,
    },
    /// Checked decimal arithmetic failed.
    #[error("checked arithmetic failed: {0}")]
    Arithmetic(String),
}

impl From<ViewError> for QueueError {
    fn from(error: ViewError) -> Self {
        Self::View(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Anchor {
    /// A hypothetical order created at `created_at`, ahead of it the orders
    /// resting then, each with its generation.
    Hypothetical {
        ahead: Vec<(String, u64)>,
        created_at: u64,
    },
    /// Our own order, seen in the feed, with its generation.
    Own { order_id: String, entered_at: u64 },
}

/// A handle on the volume ahead of one order, read with
/// [`BookReconstructor::volume_ahead`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueEstimate {
    side: L3Side,
    price: Decimal,
    segment_id: u64,
    anchor: Anchor,
}

/// The volume ahead of an order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueAhead {
    /// The sequence it was read at.
    pub at_seq: u64,
    /// Total size ahead.
    pub volume: Decimal,
    /// Of which from orders of unknown priority among themselves (from the
    /// snapshot); informational, as all of them are ahead.
    pub from_snapshot: Decimal,
    /// Of which from orders whose `open` was observed.
    pub observed: Decimal,
    /// Orders ahead.
    pub orders: usize,
    /// The hypotheses the result rests on.
    pub assumptions: Vec<Assumption>,
}

impl BookReconstructor {
    /// A hypothetical order of `side` at `price` created at the current
    /// sequence, behind every order resting there now.
    ///
    /// # Errors
    ///
    /// [`QueueError::View`] unless the book is valid.
    pub fn queue_estimate(
        &self,
        side: L3Side,
        price: Decimal,
    ) -> Result<QueueEstimate, QueueError> {
        let (book, last_seq) = self.valid_book()?;
        let ahead = book
            .levels(side)
            .get(&price)
            .map(|level| {
                level
                    .queue
                    .iter()
                    .filter_map(|id| book.orders.get(id).map(|o| (id.clone(), o.entered_at)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(QueueEstimate {
            side,
            price,
            segment_id: self.segment_id(),
            anchor: Anchor::Hypothetical {
                ahead,
                created_at: last_seq,
            },
        })
    }

    /// Our own resting order, whose `open` was observed in the feed.
    ///
    /// # Errors
    ///
    /// [`QueueError::View`] unless the book is valid,
    /// [`QueueError::NotResting`] if the order is not on the book, and
    /// [`QueueError::PriorityUnknown`] if it came from a snapshot.
    pub fn queue_estimate_own(&self, order_id: &str) -> Result<QueueEstimate, QueueError> {
        let (book, _) = self.valid_book()?;
        let order = book
            .orders
            .get(order_id)
            .ok_or_else(|| QueueError::NotResting {
                order_id: order_id.to_string(),
            })?;
        if order.evidence == PriorityEvidence::Unknown
            || (order.decreased_at.is_some()
                && self.rules().size_decrease == SizeDecrease::PriorityUnknown)
        {
            return Err(QueueError::PriorityUnknown {
                order_id: order_id.to_string(),
            });
        }
        Ok(QueueEstimate {
            side: order.side,
            price: order.price,
            segment_id: self.segment_id(),
            anchor: Anchor::Own {
                order_id: order_id.to_string(),
                entered_at: order.entered_at,
            },
        })
    }

    /// The volume now ahead of `estimate`'s order. Only the orders ahead
    /// count, each with its current remaining size: their matches, cancels
    /// and decreases reduce it, and orders opened later are behind.
    ///
    /// # Errors
    ///
    /// [`QueueError::View`] unless the book is valid,
    /// [`QueueError::StaleEstimate`] after a new snapshot,
    /// [`QueueError::NotResting`] once our own order has left the book (an
    /// order reusing its id later is not ours), and
    /// [`QueueError::PriorityUnknown`] when an order ahead decreased its
    /// size under [`SizeDecrease::PriorityUnknown`].
    pub fn volume_ahead(&self, estimate: &QueueEstimate) -> Result<QueueAhead, QueueError> {
        let (book, at_seq) = self.valid_book()?;
        if estimate.segment_id != self.segment_id() {
            return Err(QueueError::StaleEstimate);
        }
        let (ahead, since, mut assumptions): (Vec<(String, u64)>, u64, Vec<Assumption>) =
            match &estimate.anchor {
                Anchor::Hypothetical { ahead, created_at } => (
                    ahead.clone(),
                    *created_at,
                    vec![Assumption::BackOfVisibleQueueAtCreation],
                ),
                Anchor::Own {
                    order_id,
                    entered_at,
                } => {
                    // Our order, not a later one reusing its id.
                    let own = book
                        .orders
                        .get(order_id)
                        .filter(|o| o.entered_at == *entered_at)
                        .ok_or_else(|| QueueError::NotResting {
                            order_id: order_id.clone(),
                        })?;
                    let PriorityEvidence::Observed { sequence } = own.evidence else {
                        return Err(QueueError::PriorityUnknown {
                            order_id: order_id.clone(),
                        });
                    };
                    // Our own decrease: later orders may now precede it,
                    // unless decreases keep priority by declaration.
                    let mut assumptions = vec![Assumption::EntrySequenceIsPriority];
                    if own.decreased_at.is_some() {
                        match self.rules().size_decrease {
                            SizeDecrease::PriorityUnknown => {
                                return Err(QueueError::PriorityUnknown {
                                    order_id: order_id.clone(),
                                });
                            }
                            SizeDecrease::KeepsPriority => {
                                assumptions.push(Assumption::SizeDecreaseKeepsPriority);
                            }
                        }
                    }
                    let queue = book
                        .levels(estimate.side)
                        .get(&estimate.price)
                        .map(|level| level.queue.as_slice())
                        .unwrap_or_default();
                    let before: Vec<(String, u64)> = queue
                        .iter()
                        .take_while(|id| *id != order_id)
                        .filter_map(|id| book.orders.get(id).map(|o| (id.clone(), o.entered_at)))
                        .collect();
                    (before, sequence, assumptions)
                }
            };
        let (mut volume, mut from_snapshot, mut observed, mut orders) =
            (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO, 0_usize);
        let add = |acc: Decimal, v: Decimal| {
            exact_add(acc, v)
                .ok_or_else(|| QueueError::Arithmetic("volume ahead is not exact".to_string()))
        };
        for (id, generation) in &ahead {
            // Gone (filled or canceled), or its id now names a later order,
            // which is behind: nothing left ahead from it.
            let Some(order) = book.orders.get(id).filter(|o| o.entered_at == *generation) else {
                continue;
            };
            if order.decreased_at.is_some_and(|at| at > since) {
                match self.rules().size_decrease {
                    SizeDecrease::PriorityUnknown => {
                        return Err(QueueError::PriorityUnknown {
                            order_id: id.clone(),
                        });
                    }
                    SizeDecrease::KeepsPriority => {
                        if !assumptions.contains(&Assumption::SizeDecreaseKeepsPriority) {
                            assumptions.push(Assumption::SizeDecreaseKeepsPriority);
                        }
                    }
                }
            }
            volume = add(volume, order.remaining)?;
            match order.evidence {
                PriorityEvidence::Unknown => from_snapshot = add(from_snapshot, order.remaining)?,
                PriorityEvidence::Observed { .. } => observed = add(observed, order.remaining)?,
            }
            orders = orders.saturating_add(1);
        }
        Ok(QueueAhead {
            at_seq,
            volume,
            from_snapshot,
            observed,
            orders,
            assumptions,
        })
    }
}
