//! The visible book: resting orders by id and price levels in queue order.

use std::collections::{BTreeMap, HashMap};

use super::message::L3Side;
use crate::Decimal;

/// `a + b`, only when `Decimal` holds it exactly: a sum that overflows or
/// rounds (`1e28 + 1e-28` loses the small term) is `None`.
pub(crate) fn exact_add(a: Decimal, b: Decimal) -> Option<Decimal> {
    let sum = a.checked_add(b)?;
    (sum.checked_sub(a)? == b && sum.checked_sub(b)? == a).then_some(sum)
}

/// `a - b`, only when `Decimal` holds it exactly.
pub(crate) fn exact_sub(a: Decimal, b: Decimal) -> Option<Decimal> {
    let difference = a.checked_sub(b)?;
    (difference.checked_add(b)? == a && a.checked_sub(difference)? == b).then_some(difference)
}

/// What is known of a resting order's place in its level's queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityEvidence {
    /// From a snapshot: it rests ahead of every order opened later at its
    /// price, but its order among the snapshot's orders is unknown.
    Unknown,
    /// Its `open` was observed at this sequence. This is its entry sequence,
    /// not a verified FIFO priority.
    Observed {
        /// The sequence of its `open`.
        sequence: u64,
    },
}

/// A resting order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Resting {
    pub(crate) side: L3Side,
    pub(crate) price: Decimal,
    pub(crate) remaining: Decimal,
    pub(crate) evidence: PriorityEvidence,
    /// Its generation: the sequence it entered the book at (the
    /// snapshot's, or its `open`'s). An id reused after the order left is
    /// a new generation.
    pub(crate) entered_at: u64,
    /// Sequence of its last size decrease, if any.
    pub(crate) decreased_at: Option<u64>,
}

/// One price level: its orders in queue order and their total size.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Level {
    pub(crate) queue: Vec<String>,
    pub(crate) total: Decimal,
}

/// The book's state.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Book {
    pub(crate) orders: HashMap<String, Resting>,
    pub(crate) bids: BTreeMap<Decimal, Level>,
    pub(crate) asks: BTreeMap<Decimal, Level>,
}

impl Book {
    pub(crate) fn levels(&self, side: L3Side) -> &BTreeMap<Decimal, Level> {
        match side {
            L3Side::Buy => &self.bids,
            L3Side::Sell => &self.asks,
        }
    }

    fn levels_mut(&mut self, side: L3Side) -> &mut BTreeMap<Decimal, Level> {
        match side {
            L3Side::Buy => &mut self.bids,
            L3Side::Sell => &mut self.asks,
        }
    }

    pub(crate) fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().copied()
    }

    pub(crate) fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().copied()
    }

    pub(crate) fn level_count(&self) -> usize {
        self.bids.len().saturating_add(self.asks.len())
    }

    /// Appends a new resting order to the back of its level. The caller has
    /// checked the id is new and computed the level's new total.
    pub(crate) fn insert(&mut self, id: String, resting: Resting, new_total: Decimal) {
        let (side, price) = (resting.side, resting.price);
        let level = self.levels_mut(side).entry(price).or_default();
        level.queue.push(id.clone());
        level.total = new_total;
        self.orders.insert(id, resting);
    }

    /// Sets an order's remaining size and its level's total. The caller has
    /// checked the order rests, the size is positive and the total.
    pub(crate) fn reduce(
        &mut self,
        id: &str,
        remaining: Decimal,
        new_total: Decimal,
        decreased_at: Option<u64>,
    ) {
        if let Some(order) = self.orders.get_mut(id) {
            order.remaining = remaining;
            if decreased_at.is_some() {
                order.decreased_at = decreased_at;
            }
            let (side, price) = (order.side, order.price);
            if let Some(level) = self.levels_mut(side).get_mut(&price) {
                level.total = new_total;
            }
        }
    }

    /// Removes an order, and its level when it was the last one. The caller
    /// has computed the level's new total.
    pub(crate) fn remove(&mut self, id: &str, new_total: Decimal) {
        let Some(order) = self.orders.remove(id) else {
            return;
        };
        let levels = self.levels_mut(order.side);
        let empty = match levels.get_mut(&order.price) {
            Some(level) => {
                level.queue.retain(|q| q != id);
                level.total = new_total;
                level.queue.is_empty()
            }
            None => false,
        };
        if empty {
            levels.remove(&order.price);
        }
    }

    /// Recomputes every invariant from scratch.
    ///
    /// # Errors
    ///
    /// The first invariant broken, described.
    pub(crate) fn audit(&self) -> Result<(), String> {
        let mut seen = 0_usize;
        for side in [L3Side::Buy, L3Side::Sell] {
            for (price, level) in self.levels(side) {
                if level.queue.is_empty() {
                    return Err(format!("empty level {price}"));
                }
                let mut sum = Decimal::ZERO;
                for id in &level.queue {
                    let Some(order) = self.orders.get(id) else {
                        return Err(format!("level {price} lists unknown id {id}"));
                    };
                    if order.side != side || order.price != *price {
                        return Err(format!("order {id} is not at {price}"));
                    }
                    if order.remaining <= Decimal::ZERO {
                        return Err(format!("order {id} has size {}", order.remaining));
                    }
                    sum = exact_add(sum, order.remaining)
                        .ok_or_else(|| format!("level {price} sum is not exact"))?;
                    seen = seen.saturating_add(1);
                }
                if sum != level.total {
                    return Err(format!(
                        "level {price} total {} differs from its orders' {sum}",
                        level.total
                    ));
                }
            }
        }
        if seen != self.orders.len() {
            return Err(format!(
                "{} orders but {seen} in levels: an id is duplicated or missing",
                self.orders.len()
            ));
        }
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask())
            && bid >= ask
        {
            return Err(format!("crossed: bid {bid} >= ask {ask}"));
        }
        Ok(())
    }
}
