//! Order by order (L3) book reconstruction, with its quality record and the
//! limits of what it says about queue priority.
//!
//! # Contract
//!
//! A [`BookReconstructor`](crate::l3::BookReconstructor) rebuilds the visible book of one product from a
//! [`L3Snapshot`](crate::l3::L3Snapshot) and the product's sequenced [`L3Message`](crate::l3::L3Message)s, modelled on
//! the public Coinbase Exchange `full` channel. It applies the exchange's
//! events; it never matches orders itself.
//!
//! - **Handshake**: messages buffered before the snapshot are dropped up to
//!   and including its sequence; the first one applied must follow it
//!   ([`BookReconstructor::bootstrap_with_buffer`](crate::l3::BookReconstructor::bootstrap_with_buffer)).
//! - **Sequencing**: each message must carry the next sequence. A gap, a
//!   sequence at or before the last one (with no attempt to recognise
//!   duplicates), or an overflow ends the segment.
//! - **Lifecycle**: only resting orders are tracked. A `received`, and a
//!   `done` or `change` of an order not on the book (never opened, or
//!   already fully matched), has no visible effect and is counted.
//! - **Consistency**: a message inconsistent with the book (a
//!   [`Discrepancy`](crate::l3::Discrepancy), such as a match against an unknown maker, a `done`
//!   filled while size rests, a size increase or price change) ends the
//!   segment too. No requeue rule is invented: the venue documents none.
//! - **Fail closed, transactional**: a message is checked in full before
//!   anything changes. Once a segment ends, every view returns an error
//!   until the next snapshot; the last good book is kept only as a stale
//!   diagnostic ([`BookReconstructor::stale_l2`](crate::l3::BookReconstructor::stale_l2)).
//! - **Snapshots** are validated before they replace anything: one product,
//!   unique ids, positive prices and sizes, checked sums, not crossed, and
//!   within [`VenueRules`](crate::l3::VenueRules) capacity.
//! - Another product's message is refused and counted, leaving the segment
//!   untouched (sequences are per product).
//!
//! # What the evidence supports
//!
//! The visible book (prices, sizes, the ids at each price) is reconstructed
//! exactly within a valid segment. Queue priority is not: orders from a
//! snapshot have [`PriorityEvidence::Unknown`](crate::l3::PriorityEvidence::Unknown) (they are ahead of every
//! later order at their price, but their order among themselves is not
//! known), and an observed `open` gives an entry sequence, not a verified
//! FIFO priority. The volume ahead of an own order
//! ([`BookReconstructor::volume_ahead`](crate::l3::BookReconstructor::volume_ahead)) is computed under hypotheses each
//! result lists ([`Assumption`](crate::l3::Assumption)). Aggregated (L2) data is never presented as
//! order by order data, and none of this validates a fill model.
//!
//! Every size sum is exact: an addition or subtraction that `Decimal`
//! cannot hold exactly (it overflows, or rounds, as `1e28 + 1e-28` would) is
//! refused like any other inconsistency. Nothing saturates or panics.

mod book;
mod message;
mod queue;
mod reconstructor;

pub use book::PriorityEvidence;
pub use message::{
    ChangeReason, DoneReason, L3Event, L3Message, L3Side, L3Snapshot, SnapshotOrder,
};
pub use queue::{Assumption, QueueAhead, QueueError, QueueEstimate};
pub use reconstructor::{
    Applied, ApplyError, BookReconstructor, Discrepancy, HandshakeError, Invalidation,
    InvalidationKind, LevelProfile, MessageCounts, QualityManifest, QueueEntry, Segment,
    SegmentEnd, SizeDecrease, SnapshotError, VenueRules, ViewError,
};
