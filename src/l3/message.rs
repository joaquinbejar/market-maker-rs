//! Order by order (L3) messages and snapshots, normalised.
//!
//! The types follow the public Coinbase Exchange `full` channel
//! (`received`, `open`, `done`, `match`, `change`) and its level 3 REST
//! snapshot. Only the fields the reconstruction needs are kept; anything
//! else a feed sends is [`L3Event::Unsupported`], which the reconstructor
//! refuses rather than ignores.

use crate::Decimal;

/// The side of an order. On a [`L3Event::Match`] it is the maker's side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum L3Side {
    /// A bid.
    Buy,
    /// An ask.
    Sell,
}

/// Why an order's lifecycle ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoneReason {
    /// Filled: its remaining size is 0.
    Filled,
    /// Canceled with its remaining size.
    Canceled,
}

/// Why an order changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeReason {
    /// Self trade prevention: the size (or the funds of a market order) is
    /// reduced.
    Stp,
    /// An order modification, which may change the price or the size.
    ModifyOrder,
}

/// One event of the feed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum L3Event {
    /// The matching engine received an order. It is not on the book.
    Received {
        /// Order id.
        order_id: String,
        /// Side.
        side: L3Side,
    },
    /// The unfilled part of an order rests on the book.
    Open {
        /// Order id.
        order_id: String,
        /// Side.
        side: L3Side,
        /// Price.
        price: Decimal,
        /// Size resting on the book.
        remaining: Decimal,
    },
    /// An order left the matching engine, on the book or not.
    Done {
        /// Order id.
        order_id: String,
        /// Side.
        side: L3Side,
        /// Price, absent for market orders.
        price: Option<Decimal>,
        /// Why.
        reason: DoneReason,
        /// Remaining size, absent for market orders.
        remaining: Option<Decimal>,
    },
    /// A trade between a taker and a resting maker.
    Match {
        /// The resting order.
        maker_id: String,
        /// The incoming order.
        taker_id: String,
        /// The maker's side.
        side: L3Side,
        /// Price.
        price: Decimal,
        /// Size traded.
        size: Decimal,
    },
    /// An order's size, funds or price changed.
    Change {
        /// Order id.
        order_id: String,
        /// Side.
        side: L3Side,
        /// Why.
        reason: ChangeReason,
        /// Price, when sent.
        price: Option<Decimal>,
        /// Size before, when sent.
        old_size: Option<Decimal>,
        /// Size after, when sent.
        new_size: Option<Decimal>,
        /// Price before, when sent.
        old_price: Option<Decimal>,
        /// Price after, when sent.
        new_price: Option<Decimal>,
        /// Funds after, for market orders.
        new_funds: Option<Decimal>,
    },
    /// Any other message type, by name.
    Unsupported {
        /// The feed's message type.
        kind: String,
    },
}

/// A sequenced message of one product.
#[derive(Debug, Clone, PartialEq)]
pub struct L3Message {
    /// Product, such as `BTC-USD`; sequences are per product.
    pub product_id: String,
    /// The feed's sequence number.
    pub sequence: u64,
    /// The feed's time, microseconds since the Unix epoch. Informational:
    /// order is given by `sequence` only.
    pub time_us: u64,
    /// The event.
    pub event: L3Event,
}

/// One resting order of a snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotOrder {
    /// Order id.
    pub order_id: String,
    /// Price.
    pub price: Decimal,
    /// Size.
    pub size: Decimal,
}

/// A level 3 snapshot: every visible resting order at `sequence`.
///
/// Within a price, orders are kept in the order given, but that order is
/// never taken as evidence of their relative priority.
#[derive(Debug, Clone, PartialEq)]
pub struct L3Snapshot {
    /// Product.
    pub product_id: String,
    /// The sequence the snapshot reflects.
    pub sequence: u64,
    /// Bids.
    pub bids: Vec<SnapshotOrder>,
    /// Asks.
    pub asks: Vec<SnapshotOrder>,
}
