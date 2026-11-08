//! Converts the Hyperliquid "An Open Book" capture into the inputs of the models.
//!
//! The dataset (Zenodo, DOI 10.5281/zenodo.18184441, CC BY 4.0) holds three streams for
//! December 2025: order statuses, book diffs and trades. Nothing here downloads, decompresses
//! or stores it: the decoders take decompressed bytes.
//!
//! # What is observed and what is inferred
//!
//! - **Order statuses and trades carry a timestamp.** It is observed.
//! - **Book diffs carry none.** A diff takes the time of its order's status, which is
//!   [`TimeEvidence::Joined`]; if none matches, or that time would step back, it takes the time
//!   of the diff before it, which is [`TimeEvidence::Inherited`]. Neither is an observed time of
//!   the diff, and every output built from a diff says which it was through [`Inferred`].
//! - **Why an order left** comes from its last status. Without one, the cause is
//!   [`EventCause::Unknown`], and the event reaches no model.
//!
//! # What is not known
//!
//! - **The book at the start.** [`Replay`] starts empty, so orders resting from before the first
//!   diff are unknown until something happens to them. The capture starts on December 1 with
//!   whatever rested from November. So depth is unreliable at first, every event before
//!   [`ReplayConfig::warmup_ns`] is [`Warmth::Cold`], and passing it proves nothing: the counts
//!   in [`Coverage`] are what there is. No continuity is claimed, across hours or at the start.
//! - **Gaps in the capture.** A replay carried over a gap goes wrong: it misses what the gap held,
//!   and shows crossed touches and repeated adds afterwards. So a silence longer than
//!   [`ReplayConfig::reset_gap_ns`], in the diffs or in the trades, empties the book and restarts
//!   the warm-up, and what spans it is cut: see [`BookEvent::epoch`]. The threshold is declared,
//!   and a quiet market can pass it too, which costs a warm-up and nothing else.
//! - **Re-emitted stretches.** The capture can write a stretch of a stream twice, one right after
//!   the other, which steps its times back and adds orders already on the book. The readers leave
//!   out a line exactly equal to an earlier one of its file, and count it in
//!   [`CaptureRead::repeated_lines`]. Nothing is removed for agreeing on some fields: the trade
//!   hash can be null, and distinct records can agree on time, side, price, size and hash.
//!   Consecutive files of a stream overlap by minutes, the later one writing the earlier one's
//!   tail again, and are not aligned to the clock hour: a [`RepeatFilter`] passed to the readers
//!   of one stream's files in order leaves out a line of the file before too, counted in
//!   [`CaptureRead::repeated_from_previous_file`]. Without it, such a stretch steps the inferred
//!   times back and crosses the book. Order status records follow the same rule.
//! - **The tick and the lot.** [`Coverage::price_grid`] is what the prices seen are multiples of,
//!   an observation that does not prove a rule. [`ReplayConfig::tick`] is declared by the caller.
//! - **Fees and funding.** They are in no file here, so they stay inputs of whatever consumes
//!   this.
//!
//! This is order by order data, and it does not show a queue's priority beyond the order in
//! which diffs arrive. A price is given in a diff only as the order's limit.
//!
//! A duration, as a waiting time or an order's life, comes from two times. It takes the weaker
//! evidence of the two.
//!
//! # The book on a time grid
//!
//! [`ReplayConfig::with_book_samples`] asks for the book at each time of a grid, as the queue
//! reactive simulator samples its own: a sample shows the book after every diff before its time.
//! The diffs of a block share one inferred time, so a sample never shows a block half applied. A
//! sample is taken when the first diff at or after its time arrives, so a reset, which a silence
//! shows only when it ends, leaves no sample of the forgotten book across the silence: the grid
//! starts again after the diff that follows it. [`book_samples`] converts them into the model's
//! [`BookSample`] once the event sizes are known.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::BufRead;
use std::str::FromStr;

use serde::Deserialize;

use super::queue_reactive::{
    BookSample, MAX_LIMITS, QrEventKind, QrSide, QueueObservation, ReferenceObservation, SizedEvent,
};
use crate::Decimal;
use crate::analytics::micro_price::TopOfBook;
use crate::analytics::order_flow::{Trade, TradeSide};
use crate::backtest::hawkes::{HawkesEvent, HawkesSegment};
use crate::strategy::signal_quote::MarketEvent;
use crate::strategy::walk_forward::{PriceSample, QuoteExposure};
use crate::types::error::{MMError, MMResult};

/// Size of a record of the order status stream, in bytes.
pub const ORDER_STATUS_RECORD_BYTES: usize = 54;
/// Status of an order accepted onto the book.
pub const STATUS_OPEN: u8 = 1;
/// Status of a fill, whole or partial.
pub const STATUS_FILLED: u8 = 5;

const NANOS_PER_MS: u64 = 1_000_000;
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Nanoseconds as seconds, for an estimator's waiting time. A wait this long is not in a capture,
/// and saturates at about 136 years.
fn to_seconds(nanos: u64) -> f64 {
    let whole = u32::try_from(nanos / NANOS_PER_SECOND).unwrap_or(u32::MAX);
    let part = u32::try_from(nanos % NANOS_PER_SECOND).unwrap_or(0);
    f64::from(whole) + f64::from(part) * 1e-9
}

fn bad(what: String) -> MMError {
    MMError::InvalidMarketState(what)
}

fn checked(value: Option<Decimal>, what: &str) -> MMResult<Decimal> {
    value.ok_or_else(|| MMError::NumericalError(format!("{what} overflows")))
}

/// Decodes a price or a size: the top 3 bits are its decimals, and the rest is its integer.
///
/// # Errors
///
/// Never for a `u32`; the result type follows the other decoders.
pub fn decode_fixed(encoded: u32) -> MMResult<Decimal> {
    let value = i64::from(encoded & 0x1FFF_FFFF);
    Ok(Decimal::new(value, encoded >> 29))
}

/// One record of the order status stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OrderStatusRecord {
    /// Observed event time, in nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
    /// The dataset's identifier of the user.
    pub user_id: u32,
    /// The status, as listed in the dataset's `statuses.csv`.
    pub status_id: u8,
    /// Whether the order is a sell.
    pub is_ask: bool,
    /// Limit price.
    pub limit_price: Decimal,
    /// Remaining size at this event.
    pub size: Decimal,
    /// The order's identifier.
    pub oid: u64,
    /// Size at submission.
    pub original_size: Decimal,
}

fn le_u32(bytes: &[u8], at: usize) -> MMResult<u32> {
    let slice = bytes
        .get(at..at.saturating_add(4))
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .ok_or_else(|| bad(format!("record too short at {at}")))?;
    Ok(u32::from_le_bytes(slice))
}

fn le_u64(bytes: &[u8], at: usize) -> MMResult<u64> {
    let slice = bytes
        .get(at..at.saturating_add(8))
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .ok_or_else(|| bad(format!("record too short at {at}")))?;
    Ok(u64::from_le_bytes(slice))
}

fn byte(bytes: &[u8], at: usize) -> MMResult<u8> {
    bytes
        .get(at)
        .copied()
        .ok_or_else(|| bad(format!("record too short at {at}")))
}

/// Decodes one 54 byte record, by the offsets of the dataset's schema.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for a record of another length.
pub fn decode_order_status(record: &[u8]) -> MMResult<OrderStatusRecord> {
    if record.len() != ORDER_STATUS_RECORD_BYTES {
        return Err(bad(format!("a record of {} bytes", record.len())));
    }
    Ok(OrderStatusRecord {
        timestamp_ns: le_u64(record, 0)?,
        user_id: le_u32(record, 8)?,
        status_id: byte(record, 13)?,
        is_ask: byte(record, 14)? != 0,
        limit_price: decode_fixed(le_u32(record, 15)?)?,
        size: decode_fixed(le_u32(record, 19)?)?,
        oid: le_u64(record, 23)?,
        original_size: decode_fixed(le_u32(record, 50)?)?,
    })
}

/// Decodes a decompressed order status file, in file order, leaving out the records
/// [`CaptureRead::repeated_lines`] counts. A status file holds one coin, so a real event could
/// repeat all of its 54 bytes only if the source wrote it so, which nothing here can tell from a
/// re-emitted stretch.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] if its length is not a whole number of records.
pub fn decode_order_statuses(bytes: &[u8]) -> MMResult<CaptureRead<OrderStatusRecord>> {
    decode_order_statuses_after(bytes, &mut RepeatFilter::new())
}

/// [`decode_order_statuses`], leaving out too the records of the file before that `filter` saw.
///
/// # Errors
///
/// As [`decode_order_statuses`].
pub fn decode_order_statuses_after(
    bytes: &[u8],
    filter: &mut RepeatFilter,
) -> MMResult<CaptureRead<OrderStatusRecord>> {
    let (records, rest) = bytes.as_chunks::<ORDER_STATUS_RECORD_BYTES>();
    if !rest.is_empty() {
        return Err(bad(format!(
            "{} bytes is not a whole number of records",
            bytes.len()
        )));
    }
    let mut read = CaptureRead::default();
    filter.start_file();
    for record in records {
        read.keep(filter, record, || decode_order_status(record))?;
    }
    Ok(read)
}

/// What a book diff changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BookChange {
    /// An order placed on the book.
    New {
        /// Its size.
        size: Decimal,
    },
    /// An order removed, canceled or fully filled.
    Remove,
    /// An order's size changed by a partial fill.
    Update {
        /// Size before.
        previous: Decimal,
        /// Size after.
        remaining: Decimal,
    },
}

/// A change to the visible book. It has no timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BookDiff {
    /// The order's identifier.
    pub oid: u64,
    /// Its side.
    pub side: QrSide,
    /// Its limit price.
    pub price: Decimal,
    /// The change.
    pub change: BookChange,
}

#[derive(Deserialize)]
struct RawNew {
    sz: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawUpdate {
    orig_sz: String,
    new_sz: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawChange {
    New(RawNew),
    Update(RawUpdate),
    Remove,
}

#[derive(Deserialize)]
struct RawDiff {
    oid: u64,
    coin: String,
    side: String,
    px: String,
    raw_book_diff: RawChange,
}

fn decimal(text: &str, what: &str) -> MMResult<Decimal> {
    Decimal::from_str(text).map_err(|error| bad(format!("{what} {text}: {error}")))
}

fn side_of(text: &str) -> MMResult<QrSide> {
    match text {
        "A" => Ok(QrSide::Ask),
        "B" => Ok(QrSide::Bid),
        other => Err(bad(format!("side {other}"))),
    }
}

/// Parses one line of the book diff stream, with its coin.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for a line that does not follow the schema.
pub fn parse_book_diff(line: &str) -> MMResult<(String, BookDiff)> {
    let raw: RawDiff =
        serde_json::from_str(line).map_err(|error| bad(format!("a book diff: {error}")))?;
    let change = match raw.raw_book_diff {
        RawChange::New(new) => BookChange::New {
            size: decimal(&new.sz, "size")?,
        },
        RawChange::Update(update) => BookChange::Update {
            previous: decimal(&update.orig_sz, "size")?,
            remaining: decimal(&update.new_sz, "size")?,
        },
        RawChange::Remove => BookChange::Remove,
    };
    Ok((
        raw.coin,
        BookDiff {
            oid: raw.oid,
            side: side_of(&raw.side)?,
            price: decimal(&raw.px, "price")?,
            change,
        },
    ))
}

/// What a reader kept of one file, and what it left out.
///
/// The check compares 64-bit hashes of the lines, without keeping them: with `n` lines in a file,
/// a line wrongly left out has a chance of about `n^2 / 2^65`, some 1 in 10^7 at the 1.4 million
/// records of a busy hour.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CaptureRead<T> {
    /// The rows, in file order.
    pub rows: Vec<T>,
    /// Lines or records of the coin left out for being exactly equal to an earlier one of the file.
    /// The capture re-emits stretches of a stream, and a JSON line carries users and ids, so a real
    /// event does not repeat one. Nothing is left out for agreeing on some of its fields.
    pub repeated_lines: u64,
    /// Lines or records left out for being exactly equal to one of the file before, when the
    /// reader was given the [`RepeatFilter`] of that file: consecutive files of a stream overlap
    /// by minutes, and the later one writes the earlier one's tail again.
    pub repeated_from_previous_file: u64,
}

impl<T> Default for CaptureRead<T> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            repeated_lines: 0,
            repeated_from_previous_file: 0,
        }
    }
}

impl<T> CaptureRead<T> {
    /// Keeps `row` unless the filter has seen its line, and counts what it leaves out.
    fn keep(
        &mut self,
        filter: &mut RepeatFilter,
        line: &[u8],
        row: impl FnOnce() -> MMResult<T>,
    ) -> MMResult<()> {
        match filter.check(line) {
            Seen::New => self.rows.push(row()?),
            Seen::InFile => self.repeated_lines = self.repeated_lines.saturating_add(1),
            Seen::InPreviousFile => {
                self.repeated_from_previous_file =
                    self.repeated_from_previous_file.saturating_add(1);
            }
        }
        Ok(())
    }
}

/// Where a line was seen before.
enum Seen {
    New,
    InFile,
    InPreviousFile,
}

/// The lines of one stream's current file and of the file before it, as hashes, so that a reader
/// leaves out a line written again: within its file, or from the tail of the file before, which
/// the next file of the capture repeats. Pass one filter per stream to the readers of its files in
/// order; each read starts a file, and what the file before held is forgotten then.
#[derive(Debug, Clone, Default)]
pub struct RepeatFilter {
    previous: HashSet<u64>,
    current: HashSet<u64>,
}

impl RepeatFilter {
    /// A filter that has seen no file.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The next file starts: the current one becomes the one before.
    fn start_file(&mut self) {
        self.previous = std::mem::take(&mut self.current);
    }

    fn check(&mut self, line: &[u8]) -> Seen {
        let mut hasher = DefaultHasher::new();
        line.hash(&mut hasher);
        let hash = hasher.finish();
        if !self.current.insert(hash) {
            Seen::InFile
        } else if self.previous.contains(&hash) {
            Seen::InPreviousFile
        } else {
            Seen::New
        }
    }
}

/// Reads the diffs of one coin from a decompressed stream, in file order, leaving out the lines
/// [`CaptureRead::repeated_lines`] counts. The stream's order is the only sequence a diff has.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for an unreadable stream or a line of any coin that
/// does not follow the schema.
pub fn read_book_diffs(reader: impl BufRead, coin: &str) -> MMResult<CaptureRead<BookDiff>> {
    read_book_diffs_after(reader, coin, &mut RepeatFilter::new())
}

/// [`read_book_diffs`], leaving out too the lines of the file before that `filter` saw.
///
/// # Errors
///
/// As [`read_book_diffs`].
pub fn read_book_diffs_after(
    reader: impl BufRead,
    coin: &str,
    filter: &mut RepeatFilter,
) -> MMResult<CaptureRead<BookDiff>> {
    let mut read = CaptureRead::default();
    filter.start_file();
    for line in reader.lines() {
        let line = line.map_err(|error| bad(format!("reading book diffs: {error}")))?;
        let (found, diff) = parse_book_diff(&line)?;
        if found != coin {
            continue;
        }
        read.keep(filter, line.as_bytes(), || Ok(diff))?;
    }
    Ok(read)
}

/// A trade of the capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CaptureTrade {
    /// Observed block time, in nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
    /// Price.
    pub price: Decimal,
    /// Size, in the base asset.
    pub size: Decimal,
    /// The aggressor's side.
    pub aggressor: TradeSide,
}

#[derive(Deserialize)]
struct RawTrade {
    coin: String,
    side: String,
    time: String,
    px: String,
    sz: String,
}

/// Days in a month of the Gregorian calendar.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year.rem_euclid(4) == 0
            && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0) =>
        {
            29
        }
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days from the Unix epoch to a civil date of the proleptic Gregorian calendar.
// The caller bounds the year to four digits, so nothing here comes near overflowing.
#[allow(clippy::arithmetic_side_effects)]
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let of_era = year.rem_euclid(400);
    let shifted = if month > 2 { month - 3 } else { month + 9 };
    let of_year = (153 * shifted + 2) / 5 + day - 1;
    let of_cycle = of_era * 365 + of_era / 4 - of_era / 100 + of_year;
    era * 146_097 + of_cycle - 719_468
}

/// Parses a capture time, `2025-12-01T10:00:00.093246718`, as UTC nanoseconds. The dataset states
/// its times in UTC, and the string carries no zone.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for another format or a time out of range.
pub fn parse_time_ns(text: &str) -> MMResult<u64> {
    let fail = || bad(format!("time {text}"));
    let (date, clock) = text.split_once('T').ok_or_else(fail)?;
    let number = |part: Option<&str>, max: i64| -> MMResult<i64> {
        let part = part.filter(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
        let value: i64 = part.and_then(|p| p.parse().ok()).ok_or_else(fail)?;
        if value <= max { Ok(value) } else { Err(fail()) }
    };
    let mut parts = date.split('-');
    let (year, month, day) = (
        number(parts.next(), 9999)?,
        number(parts.next(), 12)?,
        number(parts.next(), 31)?,
    );
    if parts.next().is_some() || month == 0 || day == 0 || day > days_in_month(year, month) {
        return Err(fail());
    }
    let (whole, fraction) = clock.split_once('.').unwrap_or((clock, ""));
    if clock.ends_with('.') {
        return Err(fail());
    }
    let mut parts = whole.split(':');
    let (hour, minute, second) = (
        number(parts.next(), 23)?,
        number(parts.next(), 59)?,
        number(parts.next(), 59)?,
    );
    if parts.next().is_some() || fraction.len() > 9 || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(fail());
    }
    let mut nanos: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse().map_err(|_| fail())?
    };
    for _ in fraction.len()..9 {
        nanos = nanos.saturating_mul(10);
    }
    // The parts are bounded above, so nothing here comes near overflowing.
    #[allow(clippy::arithmetic_side_effects)]
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second;
    let seconds = u64::try_from(seconds).map_err(|_| fail())?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(nanos))
        .ok_or_else(fail)
}

/// Reads the trades of one coin from a decompressed stream, in file order.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for an unreadable stream, a line that is not a trade,
/// or a line of this coin whose fields do not follow the schema.
pub fn read_trades(reader: impl BufRead, coin: &str) -> MMResult<CaptureRead<CaptureTrade>> {
    read_trades_after(reader, coin, &mut RepeatFilter::new())
}

/// [`read_trades`], leaving out too the lines of the file before that `filter` saw.
///
/// # Errors
///
/// As [`read_trades`].
pub fn read_trades_after(
    reader: impl BufRead,
    coin: &str,
    filter: &mut RepeatFilter,
) -> MMResult<CaptureRead<CaptureTrade>> {
    let mut read = CaptureRead::default();
    filter.start_file();
    for line in reader.lines() {
        let line = line.map_err(|error| bad(format!("reading trades: {error}")))?;
        let raw: RawTrade =
            serde_json::from_str(&line).map_err(|error| bad(format!("a trade: {error}")))?;
        if raw.coin != coin {
            continue;
        }
        read.keep(filter, line.as_bytes(), || {
            Ok(CaptureTrade {
                timestamp_ns: parse_time_ns(&raw.time)?,
                price: decimal(&raw.px, "price")?,
                size: decimal(&raw.sz, "size")?,
                aggressor: match raw.side.as_str() {
                    "B" => TradeSide::Buy,
                    "A" => TradeSide::Sell,
                    other => return Err(bad(format!("aggressor {other}"))),
                },
            })
        })?;
    }
    Ok(read)
}

/// Where a time comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimeEvidence {
    /// In the record itself: a status or a trade.
    Observed,
    /// From the status of the diff's order, joined by its identifier.
    Joined,
    /// From the diff before it, for want of a usable join.
    Inherited,
}

/// A time with where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferredTime {
    /// Nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
    /// Where it came from.
    pub evidence: TimeEvidence,
}

impl InferredTime {
    fn millis(self) -> u64 {
        self.timestamp_ns / NANOS_PER_MS
    }

    /// This time, with the weaker evidence of it and `other`, for what is built from both.
    fn weakest(self, other: Self) -> Self {
        let evidence = match (self.evidence, other.evidence) {
            (TimeEvidence::Inherited, _) | (_, TimeEvidence::Inherited) => TimeEvidence::Inherited,
            (TimeEvidence::Joined, _) | (_, TimeEvidence::Joined) => TimeEvidence::Joined,
            (TimeEvidence::Observed, TimeEvidence::Observed) => TimeEvidence::Observed,
        };
        Self { evidence, ..self }
    }
}

/// A model input with where its time came from. The model types hold a plain timestamp, so this
/// is where the difference stays visible.
#[derive(Debug, Clone, PartialEq)]
pub struct Inferred<T> {
    /// The input.
    pub value: T,
    /// Its time's origin.
    pub time: InferredTime,
    /// How many resets came before it. A difference across two epochs, such as a return between
    /// two prices, spans a gap and means nothing.
    pub epoch: u64,
}

/// Whether an event comes before the declared warm-up has passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Warmth {
    /// Before the warm-up: the book's depth is flagged as unreliable.
    Cold,
    /// After it. This is no proof that the book is complete.
    Warm,
}

/// Why a diff changed the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventCause {
    /// An order was placed.
    Limit,
    /// An order was canceled, by its last status.
    Cancel,
    /// A fill took from it, by the diff or by its last status.
    Fill,
    /// It left, and no status says why.
    Unknown,
}

/// One change applied to the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BookEvent {
    /// Its inferred time.
    pub time: InferredTime,
    /// Whether the warm-up had passed.
    pub warmth: Warmth,
    /// The order's side.
    pub side: QrSide,
    /// The order's price.
    pub price: Decimal,
    /// The size added or taken. Not negative. Zero for an order of unknown size.
    pub size: Decimal,
    /// Why.
    pub cause: EventCause,
    /// Limit from 0, nearest first, by the touch before the event. `None` without a two sided
    /// touch or beyond the configured limits.
    pub depth: Option<u8>,
    /// Known size at this price before the event. Orders resting from before the replay are
    /// not in it.
    pub level_before: Decimal,
    /// The mid before the event, if the touch was two sided.
    pub mid_before: Option<Decimal>,
    /// How many resets came before it. Nothing spans two epochs: the book, the waiting times and
    /// the segments all start again.
    pub epoch: u64,
}

/// How a trade's price stood against the rebuilt touch just before its block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TouchCheck {
    /// At the touch the aggressor takes.
    pub at_touch: u64,
    /// Beyond that touch, as in a sweep.
    pub through_touch: u64,
    /// Better than that touch for the aggressor: the book held no such price before the block.
    /// An order that rested and matched inside the block does this, and so does an order this
    /// book is missing, and nothing here tells them apart.
    pub better_than_touch: u64,
    /// After a diff that inherited its time, which may belong to this block, so no touch before
    /// the block is known.
    pub after_inherited: u64,
    /// With no two sided touch.
    pub no_touch: u64,
}

/// What a replay saw and could not establish.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct Coverage {
    /// Order status records read.
    pub statuses: u64,
    /// Diffs read.
    pub diffs: u64,
    /// Trades read.
    pub trades: u64,
    /// Diffs that took a joined time.
    pub joined_times: u64,
    /// `new` diffs with no `open` status, which inherited a time.
    pub unjoined_new: u64,
    /// `update` diffs with no fill status of that remaining size, which inherited a time.
    pub unjoined_update: u64,
    /// `remove` diffs with no ending status, which inherited a time.
    pub unjoined_remove: u64,
    /// Joined times that would have stepped back, kept in order and marked inherited.
    pub time_steps_back: u64,
    /// Diffs before any time at all, which reached no output.
    pub diffs_without_time: u64,
    /// `new` diffs of another side than their `open` status.
    pub side_mismatches: u64,
    /// `new` diffs of another price than their `open` status.
    pub price_mismatches: u64,
    /// `new` diffs of another size than their `open` status.
    pub size_mismatches: u64,
    /// `update` diffs whose previous size differs from the order's known size.
    pub update_mismatches: u64,
    /// `remove` or `update` of an order not seen: it rested from before the replay.
    pub unknown_orders: u64,
    /// Those after the warm-up.
    pub unknown_orders_warm: u64,
    /// A level going below zero, which would mean a size known wrongly.
    pub negative_levels: u64,
    /// Events that left the rebuilt book crossed.
    pub crossed_events: u64,
    /// Prices not on the declared tick.
    pub off_tick_prices: u64,
    /// What the prices seen are multiples of. Observed, not a rule of the venue.
    pub price_grid: Option<Decimal>,
    /// Orders still known to rest at the end.
    pub resting_at_end: u64,
    /// Exposures left out: an order with no mid when it opened or no positive time at risk.
    pub exposures_left_out: u64,
    /// Trades against the touch before the warm-up.
    pub touch_cold: TouchCheck,
    /// Trades against the touch after the warm-up.
    pub touch_warm: TouchCheck,
    /// `new` diffs of an order already resting, which replaced it.
    pub duplicate_new: u64,
    /// `update` or `remove` diffs of another side or price than the known order. The order's
    /// are what the book uses.
    pub order_mismatches: u64,
    /// Fills whose last status was a partial one, left with an unknown cause.
    pub partial_last_status: u64,
    /// The longest time between two consecutive diffs, in nanoseconds. A missing hour shows here.
    pub longest_gap_ns: u64,
    /// Trades stamped before the last one, skipped: they were not placed, sorted or checked. The
    /// capture's known cause is a re-emitted stretch, which [`CaptureRead`] removes within a file.
    pub trades_out_of_order: u64,
    /// The longest time between two consecutive trades, in nanoseconds.
    pub longest_trade_gap_ns: u64,
    /// Silences in the diffs that reset the replay, once each, whether a trade or the next diff
    /// showed it.
    pub resets_on_diffs: u64,
    /// Silences in the trades, which reset the replay unless a reset already fell within them.
    pub resets_on_trades: u64,
    /// Orders resting at a reset, whose lives were cut there and reached no exposure.
    pub orders_cut_by_reset: u64,
    /// Grid times of the book's sampling with no two sided touch, so no sample.
    pub book_samples_without_touch: u64,
}

/// Settings of a replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReplayConfig {
    /// The tick that depths count in. Declared, since no file states it.
    pub tick: Decimal,
    /// Limits a depth is counted up to.
    pub limits: u8,
    /// How long after the first time, or after a reset, events stay [`Warmth::Cold`], in
    /// nanoseconds.
    pub warmup_ns: u64,
    /// The silence that resets the replay, in nanoseconds: more than this between two diffs or
    /// two trades. `u64::MAX` never resets, and carries the book over any gap.
    pub reset_gap_ns: u64,
    /// The book's sampling grid, in nanoseconds, from [`ReplayConfig::with_book_samples`]. `None`,
    /// the default, takes no samples.
    pub book_sample_ns: Option<u64>,
}

impl ReplayConfig {
    /// Creates the settings.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a tick that is not positive or no limits.
    pub fn new(tick: Decimal, limits: u8, warmup_ns: u64, reset_gap_ns: u64) -> MMResult<Self> {
        if tick <= Decimal::ZERO || limits == 0 {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid replay settings: tick {tick}, {limits} limits"
            )));
        }
        Ok(Self {
            tick,
            limits,
            warmup_ns,
            reset_gap_ns,
            book_sample_ns: None,
        })
    }

    /// Samples the book every `interval_ms`, at the multiples of it since the Unix epoch, into
    /// [`ReplayOutput::book`]. See the [module documentation](self#the-book-on-a-time-grid).
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for an interval of zero, or more limits than a
    /// [`BookSample`] holds.
    pub fn with_book_samples(self, interval_ms: u64) -> MMResult<Self> {
        if interval_ms == 0 || usize::from(self.limits) > MAX_LIMITS {
            return Err(MMError::InvalidConfiguration(format!(
                "book samples every {interval_ms} ms over {} limits",
                self.limits
            )));
        }
        let interval_ns = interval_ms.checked_mul(NANOS_PER_MS).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("book samples every {interval_ms} ms"))
        })?;
        Ok(Self {
            book_sample_ns: Some(interval_ns),
            ..self
        })
    }
}

/// The visible book at a time of the sampling grid, by limit from the mid, in units of the asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BookLevels {
    /// The grid time, in milliseconds.
    pub timestamp_ms: u64,
    /// The size resting at each limit of the bid, nearest first, by [`BookEvent::depth`]'s rule.
    /// Limits past [`ReplayConfig::limits`] are zero.
    pub bid: [Decimal; MAX_LIMITS],
    /// The same for the ask.
    pub ask: [Decimal; MAX_LIMITS],
    /// The best bid.
    pub best_bid: Decimal,
    /// The best ask.
    pub best_ask: Decimal,
    /// Whether the warm-up had passed at the grid time.
    pub warmth: Warmth,
}

#[derive(Debug, Clone, Copy)]
struct Resting {
    side: QrSide,
    price: Decimal,
    size: Decimal,
    opened: InferredTime,
    distance: Option<Decimal>,
    fills: u64,
}

#[derive(Debug, Default)]
struct OrderJoin {
    open: Option<(u64, bool, Decimal, Decimal)>,
    fills: Vec<(u64, Decimal)>,
    end: Option<(u64, u8)>,
    partial_last: bool,
}

/// Whether a status other than a fill ends an order on the book. Rejections never reach it, and
/// neither an open nor a trigger ends one.
fn ends_an_order(status: u8) -> bool {
    matches!(status, 2 | 4 | 7 | 10 | 11 | 12 | 13 | 14 | 16)
}

fn join_index(statuses: &[OrderStatusRecord]) -> HashMap<u64, OrderJoin> {
    let mut index: HashMap<u64, OrderJoin> = HashMap::with_capacity(statuses.len() / 2);
    for status in statuses {
        let join = index.entry(status.oid).or_default();
        if status.status_id == STATUS_OPEN {
            if join.open.is_none() {
                join.open = Some((
                    status.timestamp_ns,
                    status.is_ask,
                    status.limit_price,
                    status.size,
                ));
            }
        } else if status.status_id == STATUS_FILLED {
            join.fills.push((status.timestamp_ns, status.size));
            // A fill with size left does not end the order.
            join.partial_last = !status.size.is_zero();
            if status.size.is_zero() {
                join.end = Some((status.timestamp_ns, status.status_id));
            }
        } else if ends_an_order(status.status_id) {
            join.end = Some((status.timestamp_ns, status.status_id));
        }
    }
    index
}

/// What one call to [`Replay::ingest`] converted.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ReplayOutput {
    /// Each change to the book, for [`sized_events`], [`queue_observations`] and
    /// [`hawkes_segments`].
    pub events: Vec<BookEvent>,
    /// The touch, each time it changed, two sided and not crossed.
    pub tops: Vec<Inferred<TopOfBook>>,
    /// Quotes and trades in time order. A trade comes before the quotes of its block, and its
    /// time is observed, where a quote's is inferred.
    pub market_events: Vec<Inferred<MarketEvent>>,
    /// The mid, each time it changed. A mid that stays put leaves no entry, also while the book
    /// is live and its sizes change, so a stretch without one is not a silence of the book; the
    /// mids of [`tops`](Self::tops) sample it at each change of the touch.
    pub prices: Vec<Inferred<PriceSample>>,
    /// Each mid move. Only moves are seen here, so `moved` is always true and `reinitialized`
    /// always false: a move the model would call eligible and that did not happen leaves no
    /// trace, so these do not estimate its `θ`.
    pub references: Vec<Inferred<ReferenceObservation>>,
    /// Each order's life, for every order on the book and not a quoter's: its distance from
    /// the mid when it opened, its time at risk and its fills.
    pub exposures: Vec<Inferred<QuoteExposure>>,
    /// The book at each time of the grid [`ReplayConfig::with_book_samples`] asks for, with a
    /// two sided touch. Its time is the grid's; its evidence is the weaker of the diffs on
    /// either side of it.
    pub book: Vec<Inferred<BookLevels>>,
}

/// Rebuilds the visible book of one coin from its streams, hour after hour. See the
/// [module documentation](self) for what is inferred.
#[derive(Debug)]
pub struct Replay {
    config: ReplayConfig,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    orders: HashMap<u64, Resting>,
    first_time: Option<u64>,
    last_time: Option<u64>,
    last_trade_time: Option<u64>,
    silence_noted: bool,
    last_reset: Option<u64>,
    epoch: u64,
    last_mid: Option<Decimal>,
    last_top: Option<(Decimal, Decimal, Decimal, Decimal)>,
    inherited_in_block: bool,
    grid_scaled: u128,
    coverage: Coverage,
    next_sample_ns: Option<u64>,
    sample_epoch: u64,
    last_applied: Option<InferredTime>,
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a.checked_rem(b).unwrap_or(0));
    }
    a
}

const GRID_SCALE: u32 = 8;

impl Replay {
    /// Starts from an empty book.
    #[must_use]
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            first_time: None,
            last_time: None,
            last_trade_time: None,
            silence_noted: false,
            last_reset: None,
            epoch: 0,
            last_mid: None,
            last_top: None,
            inherited_in_block: false,
            grid_scaled: 0,
            coverage: Coverage::default(),
            next_sample_ns: None,
            sample_epoch: 0,
            last_applied: None,
        }
    }

    /// What it saw so far and could not establish.
    #[must_use]
    pub fn coverage(&self) -> &Coverage {
        &self.coverage
    }

    /// The touch, if it is two sided and not crossed. A crossed book means this one is wrong
    /// somewhere, so it gives neither a mid nor a depth.
    fn touch(&self) -> Option<(Decimal, Decimal)> {
        let bid = self.bids.keys().next_back().copied()?;
        let ask = self.asks.keys().next().copied()?;
        (bid < ask).then_some((bid, ask))
    }

    fn mid(&self) -> MMResult<Option<Decimal>> {
        let Some((bid, ask)) = self.touch() else {
            return Ok(None);
        };
        let sum = checked(bid.checked_add(ask), "a mid")?;
        Ok(Some(checked(sum.checked_div(Decimal::TWO), "a mid")?))
    }

    /// Limit `i` sits `i + 1/2` ticks from a mid half a tick off the grid. Anything within the
    /// first tick of the mid is limit 0, so in a touch four or more ticks wide the best prices
    /// are at limit 1 or beyond and limit 0 is empty.
    fn depth(&self, price: Decimal, mid: Decimal) -> MMResult<Option<u8>> {
        let distance = checked(price.checked_sub(mid), "a distance")?.abs();
        let ticks = checked(distance.checked_div(self.config.tick), "a distance")?;
        let limit = ticks.ceil().saturating_sub(Decimal::ONE).max(Decimal::ZERO);
        let limit = u8::try_from(limit).unwrap_or(u8::MAX);
        Ok((limit < self.config.limits).then_some(limit))
    }

    fn book(&mut self, side: QrSide) -> &mut BTreeMap<Decimal, Decimal> {
        match side {
            QrSide::Bid => &mut self.bids,
            QrSide::Ask => &mut self.asks,
        }
    }

    fn level(&self, side: QrSide, price: Decimal) -> Decimal {
        let book = match side {
            QrSide::Bid => &self.bids,
            QrSide::Ask => &self.asks,
        };
        book.get(&price).copied().unwrap_or(Decimal::ZERO)
    }

    fn add_level(&mut self, side: QrSide, price: Decimal, size: Decimal) -> MMResult<()> {
        let level = self.level(side, price);
        let next = checked(level.checked_add(size), "a level")?;
        self.book(side).insert(price, next);
        Ok(())
    }

    fn take_level(&mut self, side: QrSide, price: Decimal, size: Decimal) -> MMResult<()> {
        let level = self.level(side, price);
        let next = checked(level.checked_sub(size), "a level")?;
        if next < Decimal::ZERO {
            self.coverage.negative_levels = self.coverage.negative_levels.saturating_add(1);
        }
        if next <= Decimal::ZERO {
            self.book(side).remove(&price);
        } else {
            self.book(side).insert(price, next);
        }
        Ok(())
    }

    fn observe_price(&mut self, price: Decimal) -> MMResult<()> {
        if !checked(price.checked_rem(self.config.tick), "a price")?.is_zero() {
            self.coverage.off_tick_prices = self.coverage.off_tick_prices.saturating_add(1);
        }
        let scaled = checked(
            price.checked_mul(Decimal::from(10u64.pow(GRID_SCALE))),
            "a price",
        )?
        .trunc()
        .mantissa()
        .unsigned_abs();
        self.grid_scaled = gcd(self.grid_scaled, scaled);
        Ok(())
    }

    fn grid(&self) -> Option<Decimal> {
        let value = i128::try_from(self.grid_scaled).ok()?;
        (value > 0).then(|| {
            Decimal::try_from_i128_with_scale(value, GRID_SCALE)
                .unwrap_or(Decimal::ZERO)
                .normalize()
        })
    }

    /// The time a diff takes: joined if the join gave one that does not step back, or else the
    /// last time.
    fn time_of(&mut self, joined: Option<u64>) -> Option<InferredTime> {
        let time = match (joined, self.last_time) {
            (Some(at), Some(last)) if at < last => {
                self.coverage.time_steps_back = self.coverage.time_steps_back.saturating_add(1);
                InferredTime {
                    timestamp_ns: last,
                    evidence: TimeEvidence::Inherited,
                }
            }
            (Some(at), _) => {
                self.coverage.joined_times = self.coverage.joined_times.saturating_add(1);
                InferredTime {
                    timestamp_ns: at,
                    evidence: TimeEvidence::Joined,
                }
            }
            (None, Some(last)) => InferredTime {
                timestamp_ns: last,
                evidence: TimeEvidence::Inherited,
            },
            (None, None) => return None,
        };
        Some(time)
    }

    /// Marks that a diff arrived at `time`, once the trades before it have been checked, and
    /// resets if it ends a silence. The warm-up runs from here even if a trade already saw the
    /// silence and reset: no book has been rebuilt until the diffs come back.
    fn note_diff(&mut self, time: u64) {
        if let Some(last) = self.last_time {
            let gap = time.saturating_sub(last);
            self.coverage.longest_gap_ns = self.coverage.longest_gap_ns.max(gap);
            if gap > self.config.reset_gap_ns {
                if self.silence_noted {
                    self.first_time = Some(time);
                } else {
                    self.coverage.resets_on_diffs = self.coverage.resets_on_diffs.saturating_add(1);
                    self.reset(time);
                }
            }
        }
        // A diff that inherited its time says nothing of when the diffs came back.
        if self.last_time.is_none_or(|last| time > last) {
            self.silence_noted = false;
        }
        self.last_time = Some(time);
    }

    /// Resets at a trade if it shows a silence in the trades, or one in the diffs: a trade checked
    /// against a book that has stopped hearing would be read against what it missed.
    fn note_trade(&mut self, time: u64) {
        let limit = self.config.reset_gap_ns;
        let mut silent = false;
        if let Some(last) = self.last_trade_time {
            let gap = time.saturating_sub(last);
            self.coverage.longest_trade_gap_ns = self.coverage.longest_trade_gap_ns.max(gap);
            if gap > limit {
                self.coverage.resets_on_trades = self.coverage.resets_on_trades.saturating_add(1);
                // A reset within this silence already forgot the book, which has been rebuilding
                // from what came after it.
                silent = self.last_reset.is_none_or(|reset| reset <= last);
            }
        }
        if !self.silence_noted
            && self
                .last_time
                .is_some_and(|last| time.saturating_sub(last) > limit)
        {
            self.coverage.resets_on_diffs = self.coverage.resets_on_diffs.saturating_add(1);
            self.silence_noted = true;
            silent = true;
        }
        if silent {
            self.reset(time);
        }
        self.last_trade_time = Some(time);
    }

    /// Forgets the book and what follows from it, and starts the warm-up at `time`.
    fn reset(&mut self, time: u64) {
        let cut = u64::try_from(self.orders.len()).unwrap_or(u64::MAX);
        self.coverage.orders_cut_by_reset = self.coverage.orders_cut_by_reset.saturating_add(cut);
        self.bids.clear();
        self.asks.clear();
        self.orders.clear();
        self.last_mid = None;
        self.last_top = None;
        self.first_time = Some(time);
        self.last_reset = Some(time);
        self.epoch = self.epoch.saturating_add(1);
    }

    fn warmth(&self, time: u64) -> Warmth {
        let warm = self
            .first_time
            .is_some_and(|first| time.saturating_sub(first) >= self.config.warmup_ns);
        if warm { Warmth::Warm } else { Warmth::Cold }
    }

    /// Checks every trade up to `until` against the touch as it stands, which is the touch before
    /// the block at `until`, and adds them to the output.
    fn check_trades(
        &mut self,
        trades: &[CaptureTrade],
        next: &mut usize,
        until: Option<u64>,
        output: &mut ReplayOutput,
    ) {
        while let Some(trade) = trades.get(*next) {
            if until.is_some_and(|at| trade.timestamp_ns > at) {
                break;
            }
            *next = next.saturating_add(1);
            if self
                .last_trade_time
                .is_some_and(|last| trade.timestamp_ns < last)
            {
                self.coverage.trades_out_of_order =
                    self.coverage.trades_out_of_order.saturating_add(1);
                continue;
            }
            self.note_trade(trade.timestamp_ns);
            let warmth = self.warmth(trade.timestamp_ns);
            let touch = self.touch();
            let inherited = self.inherited_in_block;
            let check = match warmth {
                Warmth::Cold => &mut self.coverage.touch_cold,
                Warmth::Warm => &mut self.coverage.touch_warm,
            };
            let count = match (touch, trade.aggressor) {
                _ if inherited => &mut check.after_inherited,
                (None, _) => &mut check.no_touch,
                (Some((_, ask)), TradeSide::Buy) => {
                    if trade.price == ask {
                        &mut check.at_touch
                    } else if trade.price > ask {
                        &mut check.through_touch
                    } else {
                        &mut check.better_than_touch
                    }
                }
                (Some((bid, _)), TradeSide::Sell) => {
                    if trade.price == bid {
                        &mut check.at_touch
                    } else if trade.price < bid {
                        &mut check.through_touch
                    } else {
                        &mut check.better_than_touch
                    }
                }
            };
            *count = count.saturating_add(1);
            output.market_events.push(Inferred {
                value: MarketEvent::Trade(Trade {
                    price: trade.price,
                    size: trade.size,
                    side: trade.aggressor,
                    timestamp: trade.timestamp_ns / NANOS_PER_MS,
                }),
                time: InferredTime {
                    timestamp_ns: trade.timestamp_ns,
                    evidence: TimeEvidence::Observed,
                },
                epoch: self.epoch,
            });
        }
    }

    /// Emits what changed at the touch after an event.
    fn after_event(&mut self, time: InferredTime, output: &mut ReplayOutput) -> MMResult<()> {
        let Some((bid, ask)) = self.touch() else {
            if let (Some(bid), Some(ask)) = (self.bids.keys().next_back(), self.asks.keys().next())
                && bid >= ask
            {
                self.coverage.crossed_events = self.coverage.crossed_events.saturating_add(1);
            }
            return Ok(());
        };
        let top = (
            bid,
            ask,
            self.level(QrSide::Bid, bid),
            self.level(QrSide::Ask, ask),
        );
        if self.last_top == Some(top) {
            return Ok(());
        }
        self.last_top = Some(top);
        let quote = TopOfBook {
            timestamp_ms: time.millis(),
            bid: top.0,
            ask: top.1,
            bid_size: top.2,
            ask_size: top.3,
        };
        output.tops.push(Inferred {
            value: quote,
            time,
            epoch: self.epoch,
        });
        output.market_events.push(Inferred {
            value: MarketEvent::Quote(quote),
            time,
            epoch: self.epoch,
        });
        let mid = self.mid()?;
        if let Some(mid) = mid
            && self.last_mid != Some(mid)
        {
            if self.last_mid.is_some() {
                output.references.push(Inferred {
                    value: ReferenceObservation {
                        timestamp_ms: time.millis(),
                        moved: true,
                        reinitialized: false,
                    },
                    time,
                    epoch: self.epoch,
                });
            }
            self.last_mid = Some(mid);
            output.prices.push(Inferred {
                value: PriceSample {
                    timestamp_ms: time.millis(),
                    price: mid,
                },
                time,
                epoch: self.epoch,
            });
        }
        Ok(())
    }

    /// Applies one hour of a coin's streams, each in its file's order, and carries the book to
    /// the next call. A diff whose order's status is in another file inherits its time.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if a size or a price overflows.
    pub fn ingest(
        &mut self,
        statuses: &[OrderStatusRecord],
        diffs: &[BookDiff],
        trades: &[CaptureTrade],
    ) -> MMResult<ReplayOutput> {
        let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        self.coverage.statuses = self.coverage.statuses.saturating_add(count(statuses.len()));
        self.coverage.diffs = self.coverage.diffs.saturating_add(count(diffs.len()));
        self.coverage.trades = self.coverage.trades.saturating_add(count(trades.len()));
        let index = join_index(statuses);
        let mut fills_used: HashMap<u64, usize> = HashMap::new();
        let mut output = ReplayOutput::default();
        let mut next_trade = 0usize;
        for diff in diffs {
            let joined = self.joined_time(diff, &index, &mut fills_used);
            let Some(time) = self.time_of(joined) else {
                self.coverage.diffs_without_time =
                    self.coverage.diffs_without_time.saturating_add(1);
                continue;
            };
            self.first_time.get_or_insert(time.timestamp_ns);
            self.check_trades(
                trades,
                &mut next_trade,
                Some(time.timestamp_ns),
                &mut output,
            );
            self.note_diff(time.timestamp_ns);
            self.inherited_in_block = time.evidence == TimeEvidence::Inherited;
            self.sample_book(time, &mut output)?;
            self.apply(diff, time, &index, &mut output)?;
            self.last_applied = Some(time);
            self.after_event(time, &mut output)?;
        }
        self.check_trades(trades, &mut next_trade, None, &mut output);
        self.coverage.resting_at_end = count(self.orders.len());
        self.coverage.price_grid = self.grid();
        Ok(output)
    }

    /// Samples the book at every grid time up to `time`, the time of the diff about to apply, so
    /// each sample shows the diffs before its time. After a reset, or at the first diff, the grid
    /// starts after `time`.
    fn sample_book(&mut self, time: InferredTime, output: &mut ReplayOutput) -> MMResult<()> {
        let Some(interval) = self.config.book_sample_ns else {
            return Ok(());
        };
        let now = time.timestamp_ns;
        let after = |at: u64| {
            at.checked_div(interval)
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_mul(interval))
        };
        let mut next = match self.next_sample_ns {
            Some(next) if self.sample_epoch == self.epoch => next,
            _ => {
                self.sample_epoch = self.epoch;
                self.next_sample_ns = after(now);
                return Ok(());
            }
        };
        while next <= now {
            let at = InferredTime {
                timestamp_ns: next,
                evidence: time.evidence,
            };
            let at = self.last_applied.map_or(at, |last| at.weakest(last));
            match self.levels(next)? {
                Some(levels) => output.book.push(Inferred {
                    value: levels,
                    time: at,
                    epoch: self.epoch,
                }),
                None => {
                    self.coverage.book_samples_without_touch =
                        self.coverage.book_samples_without_touch.saturating_add(1);
                }
            }
            let Some(following) = next.checked_add(interval) else {
                self.next_sample_ns = None;
                return Ok(());
            };
            next = following;
        }
        self.next_sample_ns = Some(next);
        Ok(())
    }

    /// The book by limit from the mid, if the touch is two sided.
    fn levels(&self, at_ns: u64) -> MMResult<Option<BookLevels>> {
        let (Some((best_bid, best_ask)), Some(mid)) = (self.touch(), self.mid()?) else {
            return Ok(None);
        };
        Ok(Some(BookLevels {
            timestamp_ms: at_ns / NANOS_PER_MS,
            bid: self.side_levels(self.bids.iter().rev(), mid)?,
            ask: self.side_levels(self.asks.iter(), mid)?,
            best_bid,
            best_ask,
            warmth: self.warmth(at_ns),
        }))
    }

    /// One side's sizes by limit, from prices given from the touch outwards.
    fn side_levels<'a>(
        &self,
        prices: impl Iterator<Item = (&'a Decimal, &'a Decimal)>,
        mid: Decimal,
    ) -> MMResult<[Decimal; MAX_LIMITS]> {
        let mut queues = [Decimal::ZERO; MAX_LIMITS];
        for (price, size) in prices {
            // Prices run away from the mid, so the first past the limits ends the side.
            let Some(depth) = self.depth(*price, mid)? else {
                break;
            };
            if let Some(queue) = queues.get_mut(usize::from(depth)) {
                *queue = checked(queue.checked_add(*size), "a queue")?;
            }
        }
        Ok(queues)
    }

    fn joined_time(
        &mut self,
        diff: &BookDiff,
        index: &HashMap<u64, OrderJoin>,
        fills_used: &mut HashMap<u64, usize>,
    ) -> Option<u64> {
        let join = index.get(&diff.oid);
        match diff.change {
            BookChange::New { size } => {
                let Some((at, is_ask, price, opened)) = join.and_then(|j| j.open) else {
                    self.coverage.unjoined_new = self.coverage.unjoined_new.saturating_add(1);
                    return None;
                };
                if is_ask != (diff.side == QrSide::Ask) {
                    self.coverage.side_mismatches = self.coverage.side_mismatches.saturating_add(1);
                }
                if price != diff.price {
                    self.coverage.price_mismatches =
                        self.coverage.price_mismatches.saturating_add(1);
                }
                if opened != size {
                    self.coverage.size_mismatches = self.coverage.size_mismatches.saturating_add(1);
                }
                Some(at)
            }
            BookChange::Update { remaining, .. } => {
                let fills = join.map_or(&[][..], |j| j.fills.as_slice());
                let used = fills_used.entry(diff.oid).or_insert(0);
                let found = fills
                    .iter()
                    .enumerate()
                    .skip(*used)
                    .find(|(_, (_, size))| *size == remaining);
                if let Some((position, (at, _))) = found {
                    *used = position.saturating_add(1);
                    Some(*at)
                } else {
                    self.coverage.unjoined_update = self.coverage.unjoined_update.saturating_add(1);
                    None
                }
            }
            BookChange::Remove => {
                let Some((at, _)) = join.and_then(|j| j.end) else {
                    self.coverage.unjoined_remove = self.coverage.unjoined_remove.saturating_add(1);
                    if join.is_some_and(|j| j.partial_last) {
                        self.coverage.partial_last_status =
                            self.coverage.partial_last_status.saturating_add(1);
                    }
                    return None;
                };
                Some(at)
            }
        }
    }

    fn note_unknown(&mut self, warmth: Warmth) {
        self.coverage.unknown_orders = self.coverage.unknown_orders.saturating_add(1);
        if warmth == Warmth::Warm {
            self.coverage.unknown_orders_warm = self.coverage.unknown_orders_warm.saturating_add(1);
        }
    }

    fn apply(
        &mut self,
        diff: &BookDiff,
        time: InferredTime,
        index: &HashMap<u64, OrderJoin>,
        output: &mut ReplayOutput,
    ) -> MMResult<()> {
        self.observe_price(diff.price)?;
        let warmth = self.warmth(time.timestamp_ns);
        let mid_before = self.mid()?;
        let depth = match mid_before {
            Some(mid) => self.depth(diff.price, mid)?,
            None => None,
        };
        let level_before = self.level(diff.side, diff.price);
        let (size, cause) = match diff.change {
            BookChange::New { size } => {
                let distance = match mid_before {
                    Some(mid) => Some(checked(diff.price.checked_sub(mid), "a distance")?.abs()),
                    None => None,
                };
                let replaced = self.orders.insert(
                    diff.oid,
                    Resting {
                        side: diff.side,
                        price: diff.price,
                        size,
                        opened: time,
                        distance,
                        fills: 0,
                    },
                );
                if let Some(old) = replaced {
                    self.coverage.duplicate_new = self.coverage.duplicate_new.saturating_add(1);
                    self.take_level(old.side, old.price, old.size)?;
                }
                self.add_level(diff.side, diff.price, size)?;
                (size, EventCause::Limit)
            }
            BookChange::Update {
                previous,
                remaining,
            } => {
                let taken = checked(previous.checked_sub(remaining), "a fill")?.abs();
                if let Some(order) = self.orders.get_mut(&diff.oid) {
                    let (known, side, price) = (order.size, order.side, order.price);
                    order.size = remaining;
                    order.fills = order.fills.saturating_add(1);
                    if known != previous {
                        self.coverage.update_mismatches =
                            self.coverage.update_mismatches.saturating_add(1);
                    }
                    if side != diff.side || price != diff.price {
                        self.coverage.order_mismatches =
                            self.coverage.order_mismatches.saturating_add(1);
                    }
                    let change = checked(known.checked_sub(remaining), "a fill")?;
                    self.take_level(side, price, change)?;
                } else {
                    self.note_unknown(warmth);
                }
                (taken, EventCause::Fill)
            }
            BookChange::Remove => {
                let cause = match index.get(&diff.oid).and_then(|j| j.end) {
                    Some((_, STATUS_FILLED)) => EventCause::Fill,
                    Some(_) => EventCause::Cancel,
                    None => EventCause::Unknown,
                };
                if let Some(order) = self.orders.remove(&diff.oid) {
                    if order.side != diff.side || order.price != diff.price {
                        self.coverage.order_mismatches =
                            self.coverage.order_mismatches.saturating_add(1);
                    }
                    self.take_level(order.side, order.price, order.size)?;
                    self.push_exposure(&order, cause, time, output);
                    (order.size, cause)
                } else {
                    self.note_unknown(warmth);
                    (Decimal::ZERO, cause)
                }
            }
        };
        output.events.push(BookEvent {
            time,
            warmth,
            side: diff.side,
            price: diff.price,
            size,
            cause,
            depth,
            level_before,
            mid_before,
            epoch: self.epoch,
        });
        Ok(())
    }

    fn push_exposure(
        &mut self,
        order: &Resting,
        cause: EventCause,
        time: InferredTime,
        output: &mut ReplayOutput,
    ) {
        let exposure_ms =
            time.timestamp_ns.saturating_sub(order.opened.timestamp_ns) / NANOS_PER_MS;
        let Some(distance) = order.distance.filter(|_| exposure_ms > 0) else {
            self.coverage.exposures_left_out = self.coverage.exposures_left_out.saturating_add(1);
            return;
        };
        let final_fill = u64::from(cause == EventCause::Fill);
        output.exposures.push(Inferred {
            value: QuoteExposure {
                timestamp_ms: time.millis(),
                distance,
                exposure_ms,
                fills: order.fills.saturating_add(final_fill),
            },
            time: time.weakest(order.opened),
            epoch: self.epoch,
        });
    }
}

/// The events' sizes for `estimate_event_sizes`: warm ones with a known size at a depth inside
/// the limits, in order.
#[must_use]
pub fn sized_events(events: &[BookEvent]) -> Vec<Inferred<SizedEvent>> {
    events
        .iter()
        .filter(|event| {
            event.warmth == Warmth::Warm
                && event.size > Decimal::ZERO
                && event.cause != EventCause::Unknown
        })
        .filter_map(|event| {
            Some(Inferred {
                value: SizedEvent {
                    timestamp_ms: event.time.millis(),
                    depth: event.depth?,
                    size: event.size,
                },
                time: event.time,
                epoch: event.epoch,
            })
        })
        .collect()
}

/// Queue observations, and what was left out.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct QueueConversion {
    /// The observations, in order.
    pub observations: Vec<Inferred<QueueObservation>>,
    /// Events before the warm-up.
    pub cold: u64,
    /// Events whose cause no status gives.
    pub unknown_cause: u64,
    /// Events with no two sided touch or beyond the limits.
    pub no_depth: u64,
    /// Observations kept with a waiting time of zero, as events of one block share a time. The
    /// estimator leaves these out, so they take from its counts and not its time.
    pub zero_waits: u64,
}

fn side_index(side: QrSide) -> usize {
    match side {
        QrSide::Bid => 0,
        QrSide::Ask => 1,
    }
}

fn kind_of(cause: EventCause) -> Option<QrEventKind> {
    match cause {
        EventCause::Limit => Some(QrEventKind::Limit),
        EventCause::Cancel => Some(QrEventKind::Cancel),
        EventCause::Fill => Some(QrEventKind::Market),
        EventCause::Unknown => None,
    }
}

/// A queue's size in units of its event size, to the nearest.
fn in_event_sizes(level: Decimal, event_size: Decimal) -> MMResult<u32> {
    let units = checked(level.checked_div(event_size), "a queue")?.round();
    u32::try_from(units).map_err(|_| MMError::NumericalError(format!("a queue of {units}")))
}

/// Converts the replay's book samples into the queue reactive model's, warm ones only, for a data
/// [`Trajectory`](super::queue_reactive::Trajectory). Each limit's size is counted in units of
/// its event size from `estimate_event_sizes`, to the nearest, as [`queue_observations`] counts a
/// queue; limits with no event size must be empty. The spread is the touch's width in `tick`s,
/// rounded up. The model counts an empty first limit as a tick of spread, and the replay's limits
/// count from the mid, so a touch two ticks wide has both sides at limit 0 here and a spread of
/// 2, which the model reaches only with an empty first limit.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an event size or a tick that is not positive,
/// and for a limit with size and no event size, and [`MMError::NumericalError`] for a queue or a
/// spread too large to count.
pub fn book_samples(
    book: &[Inferred<BookLevels>],
    event_sizes: &[Decimal],
    tick: Decimal,
) -> MMResult<Vec<Inferred<BookSample>>> {
    if tick <= Decimal::ZERO || event_sizes.iter().any(|size| *size <= Decimal::ZERO) {
        return Err(MMError::InvalidConfiguration(
            "event sizes and the tick must be positive".to_string(),
        ));
    }
    let units = |levels: &[Decimal; MAX_LIMITS]| -> MMResult<[u32; MAX_LIMITS]> {
        let mut queues = [0u32; MAX_LIMITS];
        for (depth, (level, queue)) in levels.iter().zip(&mut queues).enumerate() {
            match event_sizes.get(depth) {
                Some(size) => *queue = in_event_sizes(*level, *size)?,
                None if level.is_zero() => {}
                None => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "no event size for limit {depth}"
                    )));
                }
            }
        }
        Ok(queues)
    };
    book.iter()
        .filter(|sample| sample.value.warmth == Warmth::Warm)
        .map(|sample| {
            let levels = &sample.value;
            let width = checked(levels.best_ask.checked_sub(levels.best_bid), "a spread")?;
            let ticks = checked(width.checked_div(tick), "a spread")?.ceil();
            let spread_ticks = u32::try_from(ticks)
                .map_err(|_| MMError::NumericalError(format!("a spread of {ticks} ticks")))?;
            Ok(Inferred {
                value: BookSample {
                    timestamp_ms: levels.timestamp_ms,
                    bid: units(&levels.bid)?,
                    ask: units(&levels.ask)?,
                    spread_ticks,
                },
                time: sample.time,
                epoch: sample.epoch,
            })
        })
        .collect()
}

/// Converts events into the queue reactive model's observations, with `event_sizes` per limit
/// from `estimate_event_sizes` on the same window. A move of the mid restarts the waiting times,
/// as the model's estimator does. A reset leaves no two sided touch, so no observation waits from
/// before it: the first one after it follows a move.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an event size that is missing or not positive.
pub fn queue_observations(
    events: &[BookEvent],
    event_sizes: &[Decimal],
) -> MMResult<QueueConversion> {
    if event_sizes.iter().any(|size| *size <= Decimal::ZERO) {
        return Err(MMError::InvalidConfiguration(
            "event sizes must be positive".to_string(),
        ));
    }
    let mut conversion = QueueConversion::default();
    let mut last: HashMap<(usize, u8), InferredTime> = HashMap::new();
    let mut mid = None;
    let mut restarted = None;
    let mut before = None;
    for event in events {
        // The move happened on the event before the one that first sees it.
        if event.mid_before != mid {
            mid = event.mid_before;
            restarted = before;
            last.clear();
        }
        before = Some(event.time);
        // Every event moves its queue's clock, whether or not it reaches a model.
        let previous = event
            .depth
            .and_then(|depth| last.insert((side_index(event.side), depth), event.time))
            .or(restarted);
        if event.warmth == Warmth::Cold {
            conversion.cold = conversion.cold.saturating_add(1);
            continue;
        }
        let Some(kind) = kind_of(event.cause) else {
            conversion.unknown_cause = conversion.unknown_cause.saturating_add(1);
            continue;
        };
        let Some(depth) = event.depth else {
            conversion.no_depth = conversion.no_depth.saturating_add(1);
            continue;
        };
        let event_size = event_sizes.get(usize::from(depth)).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("no event size for limit {depth}"))
        })?;
        let previous = previous.unwrap_or(event.time);
        let now = event.time.timestamp_ns;
        if now <= previous.timestamp_ns {
            conversion.zero_waits = conversion.zero_waits.saturating_add(1);
        }
        conversion.observations.push(Inferred {
            value: QueueObservation {
                timestamp_ms: event.time.millis(),
                side: event.side,
                depth,
                kind,
                queue_before: in_event_sizes(event.level_before, *event_size)?,
                waiting_seconds: to_seconds(now.saturating_sub(previous.timestamp_ns)),
            },
            time: event.time.weakest(previous),
            epoch: event.epoch,
        });
    }
    Ok(conversion)
}

/// A segment with how much of it rests on inherited times.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HawkesRecord {
    /// The segment.
    pub segment: HawkesSegment,
    /// Its events whose time was inherited.
    pub inherited_events: u64,
}

/// Hawkes segments, and what was left out.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct HawkesConversion {
    /// One record per period at a constant mid in which the queue had events to keep.
    pub segments: Vec<HawkesRecord>,
    /// Events before the warm-up.
    pub cold: u64,
    /// Events whose cause no status gives.
    pub unknown_cause: u64,
    /// Periods left out for having no length: their kept events shared one time with the end.
    pub zero_length: u64,
    /// Periods left out because, in event sizes, a departure found the queue empty, which the
    /// model refuses. Real events have their own sizes, so this happens.
    pub inconsistent: u64,
    /// Periods left out because a reset cut them: no move ended them.
    pub cut_by_reset: u64,
}

/// Whether a path of unit events from `initial` ever departs from an empty queue.
fn departs_from_empty(segment: &HawkesSegment) -> bool {
    let mut queue = segment.initial_queue;
    for event in &segment.events {
        match event.kind {
            QrEventKind::Limit => queue = queue.saturating_add(1),
            QrEventKind::Cancel | QrEventKind::Market => {
                if queue == 0 {
                    return true;
                }
                queue = queue.saturating_sub(1);
            }
        }
    }
    false
}

/// Converts events at the first limit of a side into one segment per period at a constant mid.
/// A segment starts at the first event kept, with the queue before it counted in `event_size`,
/// rounded, and ends at the move. The last period, with no move to end it, is left out, and so
/// is one a reset cuts.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an event size that is not positive.
pub fn hawkes_segments(
    events: &[BookEvent],
    side: QrSide,
    event_size: Decimal,
) -> MMResult<HawkesConversion> {
    if event_size <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(format!(
            "event size {event_size}"
        )));
    }
    let mut conversion = HawkesConversion::default();
    let mut mid = None;
    let mut before = 0u64;
    let mut open: Option<(u64, HawkesRecord)> = None;
    let mut epoch = events.first().map_or(0, |event| event.epoch);
    for event in events {
        let now = event.time.timestamp_ns;
        if event.epoch != epoch {
            epoch = event.epoch;
            if open.take().is_some() {
                conversion.cut_by_reset = conversion.cut_by_reset.saturating_add(1);
            }
        }
        // The move happened on the event before the one that first sees it.
        if event.mid_before != mid {
            if let Some((start, mut record)) = open.take() {
                let length = before.saturating_sub(start);
                record.segment.duration_seconds = to_seconds(length);
                if length == 0 {
                    conversion.zero_length = conversion.zero_length.saturating_add(1);
                } else if departs_from_empty(&record.segment) {
                    conversion.inconsistent = conversion.inconsistent.saturating_add(1);
                } else {
                    conversion.segments.push(record);
                }
            }
            mid = event.mid_before;
        }
        before = now;
        if event.side != side || event.depth != Some(0) {
            continue;
        }
        if event.warmth == Warmth::Cold {
            conversion.cold = conversion.cold.saturating_add(1);
            continue;
        }
        let Some(kind) = kind_of(event.cause) else {
            conversion.unknown_cause = conversion.unknown_cause.saturating_add(1);
            continue;
        };
        let (start, record) = match &mut open {
            Some(open) => open,
            slot => slot.insert((
                now,
                HawkesRecord {
                    segment: HawkesSegment {
                        initial_queue: in_event_sizes(event.level_before, event_size)?,
                        ..HawkesSegment::default()
                    },
                    inherited_events: 0,
                },
            )),
        };
        if event.time.evidence == TimeEvidence::Inherited {
            record.inherited_events = record.inherited_events.saturating_add(1);
        }
        record.segment.events.push(HawkesEvent {
            time_seconds: to_seconds(now.saturating_sub(*start)),
            kind,
        });
    }
    Ok(conversion)
}
