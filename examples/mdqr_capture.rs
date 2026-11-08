//! Extracts #106's main study inputs from the Hyperliquid capture: the 2 h windows MDQR is
//! trained, validated and held out on, and the fits of its controls (QR, QR with zero waits
//! counted, the constant baseline, Hawkes and Poisson), under
//! `bridges/python/experiments/m5_01/design.json`.
//!
//! Run with
//! `cargo run --release --features serde --example mdqr_capture -- <design.json> <data dir> <out dir>`.
//!
//! `<data dir>` holds the gunzipped hour files `o/<day>/btc_<hh>.data`, `d/<day>/ex<h>` and
//! `t/<day>/<h>` with `checked.json` (written by `experiments/m5_01/extract.py`); nothing is read
//! unless `checked.json` lists every file this reads. The data is real exchange data (Zenodo,
//! DOI 10.5281/zenodo.18184441, CC BY 4.0) and is not in this repository; the label in
//! `checked.json` is carried into every output, and the summary on standard output holds counts
//! and statuses only.
//!
//! # What it does, in the order it does it
//!
//! 1. Replays the hours [`FIRST_HOUR`] to [`LAST_HOUR`] in order with
//!    `backtest::hyperliquid` as `hyperliquid_validation` does: coin BTC, tick 1, [`LIMITS`]
//!    limits, a reset on a silence over [`RESET_GAP_NS`], a warm-up of [`WARMUP_MINUTES`], book
//!    samples every [`BOOK_SAMPLE_MS`], one `RepeatFilter` per stream across the files. Twice, as
//!    there: a first pass finds the epochs, a second one collects, and every file's digest must
//!    agree between them.
//! 2. Refuses to go on unless the longest warm stretch, by `hyperliquid_validation`'s rule, is
//!    `main_study.epoch.window_utc`, and its `Protocol` split is `main_study.split`.
//! 3. Estimates the event sizes on the size window (`estimate_event_sizes`, [`LIMITS`] limits for
//!    the queue models and one for the Hawkes flow).
//! 4. Writes each window of `main_study.windows` to `<out>/<set>/window-<n>.json` (`n` from 0 in
//!    the design's order): every queue observation inside it, with the event's size in event
//!    sizes unrounded and its block (the index of its distinct time, in nanoseconds, from 0), the
//!    warm book samples, the mid moves, the trades, and its history: the last book sample
//!    strictly before its start, and the observations, mid moves and trades of the
//!    [`HISTORY_MS`] strictly before it.
//! 5. Fits the controls on `queue_train` as `hyperliquid_validation` does and writes
//!    `<out>/fits.json`: the estimate with `ZeroWaitPolicy::Drop` (#263, #290's E0), with
//!    `ZeroWaitPolicy::Count` (#290's E1), the Drop fit's constant baseline, Poisson, and Hawkes
//!    with its decay chosen on validation from [`DECAYS`]. A failed arm gives its reason.
//!
//! The thinning of the fits' inputs is `hyperliquid_validation`'s: [`Strides::for_hours`] from
//! the number of hours replayed, one in a stride by count, and at most [`MAX_PART_EVENTS`] Hawkes
//! events a part. The windows are not thinned.
//!
//! As in `hyperliquid_validation`, the queue observations are formed per part of the split and
//! per hour file, so the first observation of each such slice waits from the slice's start.

use std::error::Error;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::analytics::order_flow::TradeSide;
use market_maker_rs::backtest::hawkes::{
    DecayScore, HawkesFitConfig, HawkesSegment, fit_hawkes, fit_poisson, select_decay,
};
use market_maker_rs::backtest::hyperliquid::{
    BookEvent, BookLevels, EventCause, Inferred, RepeatFilter, Replay, ReplayConfig, ReplayOutput,
    Warmth, book_samples, decode_order_statuses_after, hawkes_segments, parse_time_ns,
    queue_observations, read_book_diffs_after, read_trades_after, sized_events,
};
use market_maker_rs::backtest::queue_reactive::{
    BookSample, EventSizeRecord, QrEstimationConfig, QrEventKind, QrFit, QrParameters, QrRecord,
    QrSide, QueueObservation, ReferenceObservation, SizedEvent, UnsupportedPolicy, ZeroWaitPolicy,
    estimate_event_sizes, estimate_parameters,
};
use market_maker_rs::dec;
use market_maker_rs::strategy::signal_quote::MarketEvent;
use market_maker_rs::strategy::walk_forward::TimeWindow;
use rust_decimal::prelude::ToPrimitive;
use serde::Serialize;
use serde_json::{Value, json};

const NS_PER_MS: u64 = 1_000_000;
const SECOND_MS: u64 = 1_000;
const MINUTE_MS: u64 = 60 * SECOND_MS;
const HOUR_MS: u64 = 3_600_000;

/// The coin and its declared tick, as #263.
const COIN: &str = "BTC";
const TICK: Decimal = dec!(1);
/// The replay's limits a side, as `hyperliquid_validation`.
const LIMITS: u8 = 3;
/// A silence over this resets the replay, as `hyperliquid_validation`.
const RESET_GAP_NS: u64 = 60 * 1_000_000_000;
/// The replay's warm-up, as #263 ran it.
const WARMUP_MINUTES: u64 = 10;
/// The book's sampling grid, as `hyperliquid_validation`.
const BOOK_SAMPLE_MS: u64 = SECOND_MS;
/// The first and last hours replayed, both included (design `main_study.data.members`).
const FIRST_HOUR: &str = "2025-12-21T15:00:00";
const LAST_HOUR: &str = "2025-12-30T18:00:00";

/// `hyperliquid_validation`'s split.
const TRAIN_PER_MILLE: u64 = 600;
const VALIDATION_PER_MILLE: u64 = 200;
const SIZE_PER_MILLE: u64 = 100;
const SIZE_WINDOW_MAX_MS: u64 = HOUR_MS;
const MIN_SIZED_EVENTS: u64 = 30;

/// `hyperliquid_validation`'s thinning.
const FULL_RATE_HOURS: u64 = 24;
const QUEUE_STRIDE: u64 = 1;
const SEGMENT_STRIDE: u64 = 1;
const MAX_PART_EVENTS: usize = 4_000_000;

/// `hyperliquid_validation`'s queue-reactive estimate.
const QR_MAX_QUEUE: u32 = 200;
const QR_MIN_EVENTS: u64 = 30;
const QR_MIN_REFERENCE_MOVES: u64 = 30;

/// `hyperliquid_validation`'s Hawkes fits.
const HAWKES_MAX_QUEUE: u32 = 200;
const HAWKES_MIN_EVENTS: u64 = 30;
const HAWKES_ITERATIONS: u32 = 200;
const DECAYS: [f64; 4] = [0.1, 1.0, 10.0, 100.0];

/// The history kept before each window, for the features of the longest imbalance horizon.
const HISTORY_MS: u64 = 15 * MINUTE_MS;

/// The window sets of the design, in order.
const SETS: [&str; 3] = ["mdqr_train", "validation", "holdout"];

type Fallible<T> = Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------------------------------------
// Files and their check.

const FNV: (u64, u64) = (0xcbf2_9ce4_8422_2325, 0x0100_0000_01b3);

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV.1)
    })
}

/// A reader that digests what passes through it.
struct Digesting<R> {
    inner: R,
    hash: u64,
    bytes: u64,
}

impl<R> Digesting<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hash: FNV.0,
            bytes: 0,
        }
    }
}

impl<R: Read> Read for Digesting<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        let seen = buf.get(..read).unwrap_or_default();
        self.hash = fnv(self.hash, seen);
        self.bytes = self.bytes.saturating_add(read as u64);
        Ok(read)
    }
}

/// An input file, as read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Stamp {
    path: String,
    bytes: u64,
    fnv: String,
}

/// A civil UTC date and hour of a time in milliseconds, after H. Hinnant's algorithm.
fn civil(ms: u64) -> (i64, i64, i64, u64) {
    let seconds = ms / 1_000;
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, seconds % 86_400 / 3_600)
}

/// The three files of each hour from `first_ms` to `last_ms`, both included, relative to the
/// data dir: statuses, diffs, trades.
fn hour_files(first_ms: u64, last_ms: u64) -> Vec<[String; 3]> {
    let mut out = Vec::new();
    let mut at = first_ms / HOUR_MS * HOUR_MS;
    while at <= last_ms {
        let (year, month, day, hour) = civil(at);
        let day = format!("{year:04}{month:02}{day:02}");
        out.push([
            format!("o/{day}/btc_{hour:02}.data"),
            format!("d/{day}/ex{hour}"),
            format!("t/{day}/{hour}"),
        ]);
        at += HOUR_MS;
    }
    out
}

/// `checked.json`'s label, once it is found to list every file in `files`.
fn check_listed(data_dir: &Path, files: &[[String; 3]]) -> Result<String, String> {
    let path = data_dir.join("checked.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("refused: {} cannot be read: {e}", path.display()))?;
    let checked: Value = serde_json::from_str(&text)
        .map_err(|e| format!("refused: {} is not JSON: {e}", path.display()))?;
    let listed = checked
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("refused: {} has no files", path.display()))?;
    let missing: Vec<&String> = files
        .iter()
        .flatten()
        .filter(|f| !listed.contains_key(f.as_str()))
        .collect();
    if let Some(first) = missing.first() {
        return Err(format!(
            "refused: checked.json does not list {} of the files read, e.g. {first}",
            missing.len()
        ));
    }
    Ok(checked
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("unlabelled")
        .to_string())
}

// ---------------------------------------------------------------------------------------------
// The replay, as `hyperliquid_validation`.

struct Hour {
    output: ReplayOutput,
    stamps: [Stamp; 3],
}

fn read_hour(
    replay: &mut Replay,
    data_dir: &Path,
    triple: &[String; 3],
    filters: &mut [RepeatFilter; 3],
) -> Fallible<Hour> {
    let [statuses, diffs, trades] = triple;
    let [status_filter, diff_filter, trade_filter] = filters;
    let bytes = fs::read(data_dir.join(statuses))?;
    let status_stamp = Stamp {
        path: statuses.clone(),
        bytes: bytes.len() as u64,
        fnv: format!("{:016x}", fnv(FNV.0, &bytes)),
    };
    let status_rows = decode_order_statuses_after(&bytes, status_filter)?;
    drop(bytes);
    let mut reader = BufReader::new(Digesting::new(fs::File::open(data_dir.join(diffs))?));
    let diff_rows = read_book_diffs_after(&mut reader, COIN, diff_filter)?;
    io::copy(&mut reader, &mut io::sink())?;
    let diff_stamp = Stamp {
        path: diffs.clone(),
        bytes: reader.get_ref().bytes,
        fnv: format!("{:016x}", reader.get_ref().hash),
    };
    let mut reader = BufReader::new(Digesting::new(fs::File::open(data_dir.join(trades))?));
    let trade_rows = read_trades_after(&mut reader, COIN, trade_filter)?;
    io::copy(&mut reader, &mut io::sink())?;
    let trade_stamp = Stamp {
        path: trades.clone(),
        bytes: reader.get_ref().bytes,
        fnv: format!("{:016x}", reader.get_ref().hash),
    };
    let output = replay.ingest(&status_rows.rows, &diff_rows.rows, &trade_rows.rows)?;
    Ok(Hour {
        output,
        stamps: [status_stamp, diff_stamp, trade_stamp],
    })
}

fn new_filters() -> [RepeatFilter; 3] {
    [
        RepeatFilter::new(),
        RepeatFilter::new(),
        RepeatFilter::new(),
    ]
}

/// One epoch's events, from the first pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EpochSpan {
    epoch: u64,
    first_warm_ns: Option<u64>,
    last_ns: u64,
}

impl EpochSpan {
    /// The warm stretch in whole seconds, `[start, end)` in milliseconds, if it is not empty.
    fn stretch_ms(&self) -> Option<(u64, u64)> {
        let start = self.first_warm_ns?.div_ceil(NS_PER_MS).div_ceil(SECOND_MS) * SECOND_MS;
        let end = self.last_ns / NS_PER_MS / SECOND_MS * SECOND_MS;
        (start < end).then_some((start, end))
    }

    fn stretch_len_ms(&self) -> u64 {
        self.stretch_ms().map_or(0, |(start, end)| end - start)
    }
}

fn take_spans(spans: &mut Vec<EpochSpan>, events: &[BookEvent]) {
    for event in events {
        let ns = event.time.timestamp_ns;
        match spans.last_mut() {
            Some(span) if span.epoch == event.epoch => {
                span.last_ns = span.last_ns.max(ns);
                if event.warmth == Warmth::Warm && span.first_warm_ns.is_none() {
                    span.first_warm_ns = Some(ns);
                }
            }
            _ => spans.push(EpochSpan {
                epoch: event.epoch,
                first_warm_ns: (event.warmth == Warmth::Warm).then_some(ns),
                last_ns: ns,
            }),
        }
    }
}

/// The longest warm stretch, the earliest on a tie.
fn longest(spans: &[EpochSpan]) -> Option<EpochSpan> {
    let mut best: Option<EpochSpan> = None;
    for span in spans {
        if span.stretch_len_ms() > best.map_or(0, |b| b.stretch_len_ms()) {
            best = Some(*span);
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------
// The split, as `hyperliquid_validation`'s Protocol.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Train,
    Validation,
    Holdout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Protocol {
    start_ms: u64,
    sizes_end_ms: u64,
    train_end_ms: u64,
    validation_end_ms: u64,
    end_ms: u64,
}

fn whole_seconds(ms: u64) -> u64 {
    ms / SECOND_MS * SECOND_MS
}

impl Protocol {
    fn new(start_ms: u64, end_ms: u64) -> Result<Self, String> {
        let span = end_ms.saturating_sub(start_ms);
        let train_end_ms = start_ms + whole_seconds(span * TRAIN_PER_MILLE / 1_000);
        let validation_end_ms =
            start_ms + whole_seconds(span * (TRAIN_PER_MILLE + VALIDATION_PER_MILLE) / 1_000);
        let sizes = (train_end_ms - start_ms) * SIZE_PER_MILLE / 1_000;
        let sizes_end_ms = start_ms + whole_seconds(sizes.min(SIZE_WINDOW_MAX_MS));
        let protocol = Self {
            start_ms,
            sizes_end_ms,
            train_end_ms,
            validation_end_ms,
            end_ms,
        };
        if !start_ms.is_multiple_of(SECOND_MS) || !end_ms.is_multiple_of(SECOND_MS) {
            return Err(format!("{protocol:?} is not in whole seconds"));
        }
        if start_ms < sizes_end_ms
            && sizes_end_ms < train_end_ms
            && train_end_ms < validation_end_ms
            && validation_end_ms < end_ms
        {
            Ok(protocol)
        } else {
            Err(format!("a stretch of {span} ms leaves a part empty"))
        }
    }

    fn part(&self, ms: u64) -> Option<Part> {
        if ms < self.start_ms || ms >= self.end_ms {
            None
        } else if ms < self.train_end_ms {
            Some(Part::Train)
        } else if ms < self.validation_end_ms {
            Some(Part::Validation)
        } else {
            Some(Part::Holdout)
        }
    }

    fn sizes(&self) -> TimeWindow {
        TimeWindow {
            start_ms: self.start_ms,
            end_ms: self.sizes_end_ms,
        }
    }

    fn queue_train(&self) -> TimeWindow {
        TimeWindow {
            start_ms: self.sizes_end_ms,
            end_ms: self.train_end_ms,
        }
    }

    fn validation(&self) -> TimeWindow {
        TimeWindow {
            start_ms: self.train_end_ms,
            end_ms: self.validation_end_ms,
        }
    }

    fn holdout(&self) -> TimeWindow {
        TimeWindow {
            start_ms: self.validation_end_ms,
            end_ms: self.end_ms,
        }
    }
}

/// `hyperliquid_validation`'s strides, from the number of hours replayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct Strides {
    factor: u64,
    queue: u64,
    segment: u64,
}

impl Strides {
    fn for_hours(hours: u64) -> Self {
        let factor = hours.div_ceil(FULL_RATE_HOURS).max(1);
        Self {
            factor,
            queue: QUEUE_STRIDE * factor,
            segment: SEGMENT_STRIDE * factor,
        }
    }
}

fn keep_one_in(seen: &mut u64, stride: u64) -> bool {
    let keep = seen.is_multiple_of(stride);
    *seen += 1;
    keep
}

fn between(events: &[BookEvent], start_ms: u64, end_ms: u64) -> &[BookEvent] {
    let from = events.partition_point(|e| e.time.timestamp_ns < start_ms * NS_PER_MS);
    let to = events.partition_point(|e| e.time.timestamp_ns < end_ms * NS_PER_MS);
    events.get(from..to).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// The design.

/// A design time, `2025-12-21T15:13:27Z`, in milliseconds.
fn parse_utc(text: &str) -> Result<u64, String> {
    let bare = text
        .strip_suffix('Z')
        .ok_or_else(|| format!("{text} is not UTC"))?;
    Ok(parse_time_ns(bare).map_err(|e| e.to_string())? / NS_PER_MS)
}

fn window_of(value: &Value) -> Result<TimeWindow, String> {
    let pair = value
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| format!("{value} is not a window"))?;
    let time =
        |v: &Value| -> Result<u64, String> { parse_utc(v.as_str().ok_or("a time is not text")?) };
    Ok(TimeWindow {
        start_ms: time(&pair[0])?,
        end_ms: time(&pair[1])?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Design {
    epoch: TimeWindow,
    sizes: TimeWindow,
    queue_train: TimeWindow,
    validation: TimeWindow,
    holdout: TimeWindow,
    /// Per set of [`SETS`], its windows.
    windows: Vec<(&'static str, Vec<TimeWindow>)>,
}

impl Design {
    fn from_json(design: &Value) -> Result<Self, String> {
        let main = design.get("main_study").ok_or("no main_study")?;
        let get = |path: &[&str]| {
            path.iter()
                .try_fold(main, |v, key| v.get(key))
                .ok_or_else(|| format!("no main_study.{}", path.join(".")))
                .and_then(window_of)
        };
        let mut windows = Vec::new();
        for set in SETS {
            let list = main
                .get("windows")
                .and_then(|w| w.get(set))
                .and_then(Value::as_array)
                .ok_or_else(|| format!("no main_study.windows.{set}"))?;
            windows.push((set, list.iter().map(window_of).collect::<Result<_, _>>()?));
        }
        Ok(Self {
            epoch: get(&["epoch", "window_utc"])?,
            sizes: get(&["split", "sizes"])?,
            queue_train: get(&["split", "queue_train"])?,
            validation: get(&["split", "validation"])?,
            holdout: get(&["split", "holdout"])?,
            windows,
        })
    }

    /// The replay's epoch and split, if they are the design's; and every window must lie in the
    /// stretch, with its history after the size window.
    fn check(&self, spans: &[EpochSpan]) -> Result<(EpochSpan, Protocol), String> {
        let span = longest(spans).ok_or("refused: the replay has no warm stretch")?;
        let (start, end) = span.stretch_ms().ok_or("refused: no warm stretch")?;
        if (TimeWindow {
            start_ms: start,
            end_ms: end,
        }) != self.epoch
        {
            return Err(format!(
                "refused: the replay's longest warm stretch is [{start}, {end}) ms, the design's \
                 epoch [{}, {}) ms",
                self.epoch.start_ms, self.epoch.end_ms
            ));
        }
        let protocol = Protocol::new(start, end).map_err(|e| format!("refused: {e}"))?;
        let ours = [
            protocol.sizes(),
            protocol.queue_train(),
            protocol.validation(),
            protocol.holdout(),
        ];
        let theirs = [self.sizes, self.queue_train, self.validation, self.holdout];
        if ours != theirs {
            return Err(format!(
                "refused: Protocol gives {ours:?}, the design's split is {theirs:?}"
            ));
        }
        for (set, list) in &self.windows {
            for w in list {
                if w.start_ms >= w.end_ms
                    || w.end_ms > protocol.end_ms
                    || w.start_ms < protocol.sizes_end_ms + HISTORY_MS
                {
                    return Err(format!(
                        "refused: {set} window {w:?} is not inside the stretch after the size \
                         window and its history"
                    ));
                }
            }
        }
        Ok((span, protocol))
    }
}

// ---------------------------------------------------------------------------------------------
// The windows.

fn side_text(side: QrSide) -> &'static str {
    if side == QrSide::Bid { "bid" } else { "ask" }
}

fn kind_text(kind: QrEventKind) -> &'static str {
    match kind {
        QrEventKind::Limit => "limit",
        QrEventKind::Cancel => "cancel",
        _ => "market",
    }
}

#[derive(Debug, Clone, Serialize)]
struct ObservationRow {
    timestamp_ns: u64,
    timestamp_ms: u64,
    side: &'static str,
    depth: u8,
    kind: &'static str,
    /// In event sizes, rounded to the nearest, as `queue_observations` counts it.
    queue_before: u32,
    /// The event's size in event sizes, unrounded.
    size: f64,
    waiting_seconds: f64,
    /// The index of its distinct time in nanoseconds within its list, from 0.
    block: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TradeRow {
    timestamp_ns: u64,
    timestamp_ms: u64,
    /// The aggressor's side.
    side: &'static str,
    /// In the base asset, as the capture gives it.
    size: String,
    price: String,
}

#[derive(Debug, Default, Serialize)]
struct History {
    start_ms: u64,
    end_ms: u64,
    last_sample: Option<BookSample>,
    observations: Vec<ObservationRow>,
    references: Vec<ReferenceObservation>,
    /// The sign of each mid move, in the order of `references`: 1 up, -1 down.
    reference_directions: Vec<i8>,
    trades: Vec<TradeRow>,
}

#[derive(Debug, Serialize)]
struct WindowFile {
    data: String,
    set: &'static str,
    index: usize,
    window: TimeWindow,
    event_sizes: Vec<String>,
    observations: Vec<ObservationRow>,
    samples: Vec<BookSample>,
    references: Vec<ReferenceObservation>,
    /// The sign of each mid move, in the order of `references`: 1 up, -1 down.
    reference_directions: Vec<i8>,
    trades: Vec<TradeRow>,
    history: History,
}

struct WindowData {
    file: WindowFile,
    /// The last warm sample strictly before the start, not yet in model units.
    last_before: Option<Inferred<BookLevels>>,
    written: bool,
    /// What was written, counted.
    counts: Value,
}

impl WindowData {
    fn new(data: &str, set: &'static str, index: usize, window: TimeWindow) -> Self {
        Self {
            file: WindowFile {
                data: data.to_string(),
                set,
                index,
                window,
                event_sizes: Vec::new(),
                observations: Vec::new(),
                samples: Vec::new(),
                references: Vec::new(),
                reference_directions: Vec::new(),
                trades: Vec::new(),
                history: History {
                    start_ms: window.start_ms - HISTORY_MS,
                    end_ms: window.start_ms,
                    ..History::default()
                },
            },
            last_before: None,
            written: false,
            counts: Value::Null,
        }
    }

    fn inside(&self, ms: u64) -> bool {
        self.file.window.start_ms <= ms && ms < self.file.window.end_ms
    }

    fn before(&self, ms: u64) -> bool {
        self.file.history.start_ms <= ms && ms < self.file.history.end_ms
    }
}

fn number_blocks(rows: &mut [ObservationRow]) {
    let mut block = 0;
    let mut last = None;
    for row in rows {
        if last.is_some_and(|t| t != row.timestamp_ns) {
            block += 1;
        }
        last = Some(row.timestamp_ns);
        row.block = block;
    }
}

// ---------------------------------------------------------------------------------------------
// The second pass.

struct Collect {
    protocol: Protocol,
    epoch: u64,
    strides: Strides,
    sized: Vec<SizedEvent>,
    sizes: Option<(EventSizeRecord, EventSizeRecord)>,
    size_error: Option<String>,
    unsorted_hours: u64,
    /// Fit inputs on `queue_train`: positive waits (E0), every observation (E1), mid moves.
    queue: Vec<QueueObservation>,
    queue_seen: u64,
    all: Vec<QueueObservation>,
    all_seen: u64,
    references: Vec<ReferenceObservation>,
    queue_error: Option<String>,
    hawkes_train: Vec<HawkesSegment>,
    hawkes_validation: Vec<HawkesSegment>,
    hawkes_events: [usize; 2],
    segments_seen: u64,
    hawkes_error: Option<String>,
    windows: Vec<WindowData>,
    /// The last mid seen in the epoch, to sign the next move.
    last_price: Option<Decimal>,
}

impl Collect {
    fn new(protocol: Protocol, epoch: u64, strides: Strides, design: &Design, data: &str) -> Self {
        let windows = design
            .windows
            .iter()
            .flat_map(|(set, list)| {
                list.iter()
                    .enumerate()
                    .map(|(i, w)| WindowData::new(data, set, i, *w))
            })
            .collect();
        Self {
            protocol,
            epoch,
            strides,
            sized: Vec::new(),
            sizes: None,
            size_error: None,
            unsorted_hours: 0,
            queue: Vec::new(),
            queue_seen: 0,
            all: Vec::new(),
            all_seen: 0,
            references: Vec::new(),
            queue_error: None,
            hawkes_train: Vec::new(),
            hawkes_validation: Vec::new(),
            hawkes_events: [0; 2],
            segments_seen: 0,
            hawkes_error: None,
            windows,
            last_price: None,
        }
    }

    fn event_sizes(&self) -> Option<&Vec<Decimal>> {
        self.sizes.as_ref().and_then(|(q, _)| q.sizes.as_ref().ok())
    }

    /// Takes one hour's output; the time read to, in milliseconds, if any.
    fn take(&mut self, output: &ReplayOutput, out_dir: &Path) -> Fallible<Option<u64>> {
        let epoch = self.epoch;
        for price in output.prices.iter().filter(|p| p.epoch == epoch) {
            let ms = price.time.timestamp_ns / NS_PER_MS;
            let previous = self.last_price.replace(price.value.price);
            let Some(previous) = previous else { continue };
            let direction = if price.value.price > previous { 1 } else { -1 };
            for w in &mut self.windows {
                if w.inside(ms) {
                    w.file.reference_directions.push(direction);
                } else if w.before(ms) {
                    w.file.history.reference_directions.push(direction);
                }
            }
        }
        for reference in output.references.iter().filter(|r| r.epoch == epoch) {
            let ms = reference.time.timestamp_ns / NS_PER_MS;
            if self.protocol.part(ms) == Some(Part::Train) {
                self.references.push(reference.value);
            }
            for w in &mut self.windows {
                if w.inside(ms) {
                    w.file.references.push(reference.value);
                } else if w.before(ms) {
                    w.file.history.references.push(reference.value);
                }
            }
        }
        for event in output.market_events.iter().filter(|e| e.epoch == epoch) {
            let MarketEvent::Trade(trade) = &event.value else {
                continue;
            };
            let ns = event.time.timestamp_ns;
            let row = || TradeRow {
                timestamp_ns: ns,
                timestamp_ms: trade.timestamp,
                side: if trade.side == TradeSide::Buy {
                    "buy"
                } else {
                    "sell"
                },
                size: trade.size.to_string(),
                price: trade.price.to_string(),
            };
            for w in &mut self.windows {
                if w.inside(ns / NS_PER_MS) {
                    w.file.trades.push(row());
                } else if w.before(ns / NS_PER_MS) {
                    w.file.history.trades.push(row());
                }
            }
        }
        self.take_events(&output.events)?;
        self.take_book(output)?;
        let read_to_ms = output.events.last().map(|last| {
            if last.epoch > self.epoch {
                u64::MAX
            } else {
                last.time.timestamp_ns / NS_PER_MS
            }
        });
        if let Some(read_to) = read_to_ms {
            self.flush(out_dir, read_to)?;
        }
        Ok(read_to_ms)
    }

    fn take_book(&mut self, output: &ReplayOutput) -> Fallible<()> {
        let epoch = self.epoch;
        let book: Vec<&Inferred<BookLevels>> = output
            .book
            .iter()
            .filter(|s| s.epoch == epoch && s.value.warmth == Warmth::Warm)
            .collect();
        if book.is_empty() {
            return Ok(());
        }
        let sizes = self.event_sizes().cloned();
        for w in &mut self.windows {
            if w.written {
                continue;
            }
            if let Some(last) = book
                .iter()
                .rev()
                .find(|s| s.value.timestamp_ms < w.file.window.start_ms)
            {
                w.last_before = Some((*last).clone());
            }
            let inside: Vec<Inferred<BookLevels>> = book
                .iter()
                .filter(|s| w.inside(s.value.timestamp_ms))
                .map(|s| (*s).clone())
                .collect();
            if inside.is_empty() {
                continue;
            }
            let sizes = sizes
                .as_ref()
                .ok_or("book samples inside a window before the event sizes")?;
            w.file.samples.extend(
                book_samples(&inside, sizes, TICK)?
                    .into_iter()
                    .map(|s| s.value),
            );
        }
        Ok(())
    }

    fn take_events(&mut self, all: &[BookEvent]) -> Fallible<()> {
        let (Some(first), Some(last)) = (
            all.iter().position(|e| e.epoch == self.epoch),
            all.iter().rposition(|e| e.epoch == self.epoch),
        ) else {
            return Ok(());
        };
        let events = all.get(first..=last).unwrap_or_default();
        if !events.is_sorted_by_key(|e| (e.epoch, e.time.timestamp_ns)) {
            self.unsorted_hours += 1;
            return Ok(());
        }
        let p = self.protocol;
        if self.sizes.is_none() && self.size_error.is_none() {
            let window = between(events, p.start_ms, p.sizes_end_ms);
            self.sized
                .extend(sized_events(window).into_iter().map(|e| e.value));
            let ended = all.last().is_some_and(|e| {
                e.epoch > self.epoch || e.time.timestamp_ns / NS_PER_MS >= p.sizes_end_ms
            });
            if ended {
                self.close_sizes();
            }
        }
        if self.sizes.is_none() {
            return Ok(());
        }
        for (part, start, end) in [
            (Part::Train, p.sizes_end_ms, p.train_end_ms),
            (Part::Validation, p.train_end_ms, p.validation_end_ms),
            (Part::Holdout, p.validation_end_ms, p.end_ms),
        ] {
            let slice = between(events, start, end);
            if !slice.is_empty() {
                self.convert(part, slice)?;
            }
        }
        Ok(())
    }

    fn close_sizes(&mut self) {
        let window = self.protocol.sizes();
        let sized = std::mem::take(&mut self.sized);
        let estimate = |limits| estimate_event_sizes(&sized, &window, limits, MIN_SIZED_EVENTS);
        match (estimate(LIMITS), estimate(1)) {
            (Ok(queue), Ok(hawkes)) => self.sizes = Some((queue, hawkes)),
            (Err(error), _) | (_, Err(error)) => {
                self.size_error = Some(format!("estimate_event_sizes failed: {error}"));
            }
        }
    }

    fn convert(&mut self, part: Part, events: &[BookEvent]) -> Fallible<()> {
        let Some((queue_sizes, hawkes_sizes)) = &self.sizes else {
            return Ok(());
        };
        let event_sizes = queue_sizes.sizes.clone().map_err(|e| format!("{e:?}"))?;
        let first = hawkes_sizes
            .sizes
            .as_ref()
            .ok()
            .and_then(|s| s.first())
            .copied();
        let conversion = queue_observations(events, &event_sizes)?;
        // The events `queue_observations` turned into observations, in the same order.
        let observed: Vec<&BookEvent> = events
            .iter()
            .filter(|e| e.warmth != Warmth::Cold && e.cause != EventCause::Unknown)
            .filter(|e| e.depth.is_some())
            .collect();
        if observed.len() != conversion.observations.len() {
            return Err("observations do not match their events".into());
        }
        for (observation, event) in conversion.observations.iter().zip(observed) {
            let o = observation.value;
            let ns = event.time.timestamp_ns;
            let ms = o.timestamp_ms;
            let size = event_sizes
                .get(usize::from(o.depth))
                .and_then(|unit| event.size.checked_div(*unit))
                .and_then(|units| units.to_f64())
                .ok_or("an event size in event sizes")?;
            let row = || ObservationRow {
                timestamp_ns: ns,
                timestamp_ms: ms,
                side: side_text(o.side),
                depth: o.depth,
                kind: kind_text(o.kind),
                queue_before: o.queue_before,
                size,
                waiting_seconds: o.waiting_seconds,
                block: 0,
            };
            for w in &mut self.windows {
                if w.inside(ms) {
                    w.file.observations.push(row());
                } else if w.before(ms) {
                    w.file.history.observations.push(row());
                }
            }
        }
        // The fits' inputs, kept as `hyperliquid_validation` keeps them, from every part in
        // order, and stored for training only, where the fits read them.
        let train = part == Part::Train;
        for o in &conversion.observations {
            if keep_one_in(&mut self.all_seen, self.strides.queue) && train {
                self.all.push(o.value);
            }
        }
        for o in &conversion.observations {
            if o.value.waiting_seconds > 0.0
                && keep_one_in(&mut self.queue_seen, self.strides.queue)
                && train
            {
                self.queue.push(o.value);
            }
        }
        let (Some(size), None, false) = (first, &self.hawkes_error, part == Part::Holdout) else {
            return Ok(());
        };
        for side in [QrSide::Bid, QrSide::Ask] {
            let conversion = match hawkes_segments(events, side, size) {
                Ok(conversion) => conversion,
                Err(error) => {
                    self.hawkes_error = Some(format!("hawkes_segments: {error}"));
                    return Ok(());
                }
            };
            for record in conversion.segments {
                if !within_cap(&record.segment, HAWKES_MAX_QUEUE) {
                    continue;
                }
                if !keep_one_in(&mut self.segments_seen, self.strides.segment) {
                    continue;
                }
                let (kept, count) = if train {
                    (&mut self.hawkes_train, &mut self.hawkes_events[0])
                } else {
                    (&mut self.hawkes_validation, &mut self.hawkes_events[1])
                };
                if *count + record.segment.events.len() > MAX_PART_EVENTS {
                    continue;
                }
                *count += record.segment.events.len();
                kept.push(record.segment);
            }
        }
        Ok(())
    }

    /// Writes every window read past its end, and drops its data.
    fn flush(&mut self, out_dir: &Path, read_to_ms: u64) -> Fallible<()> {
        let sizes: Vec<String> = self
            .event_sizes()
            .map(|s| s.iter().map(ToString::to_string).collect())
            .unwrap_or_default();
        for w in &mut self.windows {
            if w.written || read_to_ms < w.file.window.end_ms {
                continue;
            }
            if let Some(last) = w.last_before.take() {
                let units = self.sizes.as_ref().and_then(|(q, _)| q.sizes.as_ref().ok());
                let units = units.ok_or("a window before the event sizes")?;
                w.file.history.last_sample = book_samples(&[last], units, TICK)?
                    .into_iter()
                    .next()
                    .map(|s| s.value);
            }
            w.file.event_sizes.clone_from(&sizes);
            if w.file.reference_directions.len() != w.file.references.len()
                || w.file.history.reference_directions.len() != w.file.history.references.len()
            {
                return Err("a mid move without its direction".into());
            }
            number_blocks(&mut w.file.observations);
            number_blocks(&mut w.file.history.observations);
            let dir = out_dir.join(w.file.set);
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("window-{}.json", w.file.index));
            let mut writer = BufWriter::new(fs::File::create(path)?);
            serde_json::to_writer(&mut writer, &w.file)?;
            writer.flush()?;
            w.written = true;
            let blocks = |rows: &[ObservationRow]| rows.last().map_or(0, |r| r.block + 1);
            w.counts = json!({
                "observations": w.file.observations.len(),
                "blocks": blocks(&w.file.observations),
                "samples": w.file.samples.len(),
                "references": w.file.references.len(),
                "trades": w.file.trades.len(),
                "history_last_sample": w.file.history.last_sample.is_some(),
                "history_observations": w.file.history.observations.len(),
                "history_references": w.file.history.references.len(),
                "history_trades": w.file.history.trades.len(),
            });
            w.file.observations = Vec::new();
            w.file.samples = Vec::new();
            w.file.references = Vec::new();
            w.file.reference_directions = Vec::new();
            w.file.trades = Vec::new();
            w.file.history.observations = Vec::new();
            w.file.history.references = Vec::new();
            w.file.history.reference_directions = Vec::new();
            w.file.history.trades = Vec::new();
        }
        Ok(())
    }

    fn summary(&self) -> Value {
        Value::Array(
            self.windows
                .iter()
                .map(|w| {
                    json!({
                        "set": w.file.set,
                        "index": w.file.index,
                        "written": w.written,
                        "counts": w.counts,
                    })
                })
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// The fits.

fn within_cap(segment: &HawkesSegment, cap: u32) -> bool {
    let mut queue = segment.initial_queue;
    if queue > cap {
        return false;
    }
    for event in &segment.events {
        match event.kind {
            QrEventKind::Limit => {
                if queue >= cap {
                    return false;
                }
                queue += 1;
            }
            _ => queue = queue.saturating_sub(1),
        }
    }
    true
}

fn qr_config(tick: Decimal) -> QrEstimationConfig {
    QrEstimationConfig::new(
        tick,
        LIMITS,
        QR_MAX_QUEUE,
        QR_MIN_EVENTS,
        QR_MIN_REFERENCE_MOVES,
        UnsupportedPolicy::NearestSupported,
    )
}

fn hawkes_config(decays: Vec<f64>) -> HawkesFitConfig {
    HawkesFitConfig::new(HAWKES_MAX_QUEUE, decays)
        .with_iterations(HAWKES_ITERATIONS)
        .with_min_events(HAWKES_MIN_EVENTS)
}

/// A queue-reactive model's parameters, exported.
fn qr_export(params: &QrParameters) -> Fallible<Value> {
    let depths = 0..usize::from(LIMITS);
    Ok(json!({
        "fingerprint": params.fingerprint().to_string(),
        "theta": params.theta(),
        "theta_reinit": params.theta_reinit(),
        "cap": params.max_queue(),
        "event_sizes": params.event_sizes().iter().map(ToString::to_string).collect::<Vec<_>>(),
        "rates": depths.clone().map(|d| params.rates(d).map(<[_]>::to_vec)).collect::<Vec<_>>(),
        "invariant": depths.map(|d| params.invariant(d).map(<[_]>::to_vec)).collect::<Vec<_>>(),
        "serialized": serde_json::to_value(params)?,
    }))
}

fn qr_arm(record: Result<QrRecord, String>) -> (Value, Option<QrParameters>) {
    let record = match record {
        Ok(record) => record,
        Err(reason) => return (json!({ "status": "failed", "reason": reason }), None),
    };
    let meta = json!({
        "training": record.training,
        "dataset_id": record.dataset_id,
        "dataset_hash": record.dataset_hash.to_string(),
        "events": record.events,
        "excluded_events": record.excluded_events,
    });
    match &record.fit {
        QrFit::Calibrated(params) => match qr_export(params) {
            Ok(export) => (
                json!({ "status": "ok", "record": meta, "parameters": export }),
                Some(params.clone()),
            ),
            Err(e) => (
                json!({ "status": "failed", "record": meta, "reason": e.to_string() }),
                None,
            ),
        },
        other => (
            json!({ "status": "failed", "record": meta, "reason": format!("{other:?}") }),
            None,
        ),
    }
}

fn status_of(arm: &Value) -> Value {
    arm.get("status").cloned().unwrap_or(Value::Null)
}

/// The arms' fits on `queue_train`, as `hyperliquid_validation` makes them.
fn fits(collect: &Collect, dataset: &str) -> Fallible<Value> {
    let window = collect.protocol.queue_train();
    let mut arms = serde_json::Map::new();
    match (&collect.sizes, &collect.size_error) {
        (Some((queue_sizes, _)), _) if queue_sizes.sizes.is_ok() => {
            let units = queue_sizes.sizes.clone().map_err(|e| format!("{e:?}"))?;
            let estimate = |events: &[QueueObservation], zero_waits| {
                estimate_parameters(
                    events,
                    &collect.references,
                    &units,
                    &window,
                    &qr_config(TICK).with_zero_waits(zero_waits),
                    dataset,
                )
                .map_err(|error| format!("estimate_parameters: {error}"))
            };
            let (qr, drop_params) = qr_arm(estimate(&collect.queue, ZeroWaitPolicy::Drop));
            let (qr_count, _) = qr_arm(estimate(&collect.all, ZeroWaitPolicy::Count));
            let constant = match drop_params.map(|p| p.constant_baseline()) {
                Some(Ok(params)) => match qr_export(&params) {
                    Ok(export) => json!({ "status": "ok", "parameters": export }),
                    Err(e) => json!({ "status": "failed", "reason": e.to_string() }),
                },
                Some(Err(e)) => {
                    json!({ "status": "failed", "reason": format!("constant_baseline: {e}") })
                }
                None => json!({ "status": "failed", "reason": "the Drop fit failed" }),
            };
            arms.insert("qr".into(), qr);
            arms.insert("qr_count".into(), qr_count);
            arms.insert("qr_constant".into(), constant);
        }
        (_, error) => {
            let reason = error.clone().unwrap_or_else(|| {
                collect
                    .queue_error
                    .clone()
                    .unwrap_or("no event sizes".to_string())
            });
            for arm in ["qr", "qr_count", "qr_constant"] {
                arms.insert(arm.into(), json!({ "status": "failed", "reason": reason }));
            }
        }
    }
    let (poisson, hawkes) = hawkes_arms(collect)?;
    arms.insert("poisson".into(), poisson);
    arms.insert("hawkes".into(), hawkes);
    let sizes = collect.sizes.as_ref().map(|(q, h)| {
        json!({
            "queue": serde_json::to_value(q).unwrap_or(Value::Null),
            "hawkes": serde_json::to_value(h).unwrap_or(Value::Null),
        })
    });
    Ok(json!({
        "event_sizes": sizes,
        "inputs": {
            "queue_train": window,
            "strides": collect.strides,
            "queue_drop": collect.queue.len(),
            "queue_count": collect.all.len(),
            "references": collect.references.len(),
            "hawkes_segments": [collect.hawkes_train.len(), collect.hawkes_validation.len()],
            "hawkes_events": collect.hawkes_events,
            "unsorted_hours": collect.unsorted_hours,
        },
        "arms": arms,
    }))
}

fn hawkes_arms(collect: &Collect) -> Fallible<(Value, Value)> {
    let failed = |reason: String| json!({ "status": "failed", "reason": reason });
    if let Some(error) = collect
        .size_error
        .as_ref()
        .or(collect.hawkes_error.as_ref())
    {
        return Ok((failed(error.clone()), failed(error.clone())));
    }
    let (train, validation) = (&collect.hawkes_train, &collect.hawkes_validation);
    if train.is_empty() || validation.is_empty() {
        let reason = "training or validation has no segment".to_string();
        return Ok((failed(reason.clone()), failed(reason)));
    }
    let poisson = match fit_poisson(train, HAWKES_MAX_QUEUE) {
        Ok(params) => json!({
            "status": "ok",
            "fingerprint": params.fingerprint().to_string(),
            "cap": params.max_queue(),
            "rates": params.rates(),
            "serialized": serde_json::to_value(&params)?,
        }),
        Err(e) => failed(format!("fit_poisson: {e}")),
    };
    let (scores, best): (Vec<DecayScore>, Option<usize>) =
        match select_decay(train, validation, &DECAYS, &hawkes_config(Vec::new())) {
            Ok(choice) => choice,
            Err(e) => return Ok((poisson, failed(format!("select_decay: {e}")))),
        };
    let scores_json = serde_json::to_value(&scores)?;
    let Some(decay) = best.and_then(|i| scores.get(i)).map(|s| s.decay) else {
        // Why each candidate gave no parameters, as the fit itself records it.
        let causes = DECAYS
            .iter()
            .map(|d| match fit_hawkes(train, &hawkes_config(vec![*d])) {
                Ok(record) => format!("{d}: {:?}", record.fit),
                Err(e) => format!("{d}: {e}"),
            })
            .collect::<Vec<_>>();
        return Ok((
            poisson,
            json!({ "status": "failed", "reason": "no candidate decay fitted", "decay_scores": scores_json, "causes": causes }),
        ));
    };
    let hawkes = match fit_hawkes(train, &hawkes_config(vec![decay])) {
        Ok(record) => match record.parameters() {
            Some(params) => json!({
                "status": "ok",
                "decay": decay,
                "decay_scores": scores_json,
                "fingerprint": params.fingerprint().to_string(),
                "cap": params.max_queue(),
                "branching_ratio": params.branching_ratio(),
                "rates": params.rates(),
                "record": {
                    "dataset_hash": record.dataset_hash.to_string(),
                    "events": record.events,
                    "seconds": record.seconds,
                    "borrowed_sizes": record.borrowed_sizes,
                },
                "serialized": serde_json::to_value(params)?,
            }),
            None => json!({
                "status": "failed",
                "decay": decay,
                "decay_scores": scores_json,
                "reason": format!("{:?}", record.fit),
            }),
        },
        Err(e) => failed(format!("fit_hawkes: {e}")),
    };
    Ok((poisson, hawkes))
}

// ---------------------------------------------------------------------------------------------
// The run.

/// What a run reads and where it writes.
#[derive(Debug, Clone)]
struct Run {
    design: PathBuf,
    data_dir: PathBuf,
    out_dir: PathBuf,
    first_hour_ms: u64,
    last_hour_ms: u64,
    warmup_ns: u64,
}

fn run(spec: &Run) -> Fallible<Value> {
    let files = hour_files(spec.first_hour_ms, spec.last_hour_ms);
    let label = check_listed(&spec.data_dir, &files)?;
    let design: Value = serde_json::from_str(&fs::read_to_string(&spec.design)?)?;
    let design = Design::from_json(&design)?;
    let config = ReplayConfig::new(TICK, LIMITS, spec.warmup_ns, RESET_GAP_NS)?
        .with_book_samples(BOOK_SAMPLE_MS)?;

    // First pass: the epochs.
    let mut replay = Replay::new(config);
    let mut filters = new_filters();
    let mut spans = Vec::new();
    let mut stamps = Vec::new();
    for (index, triple) in files.iter().enumerate() {
        let hour = read_hour(&mut replay, &spec.data_dir, triple, &mut filters)?;
        take_spans(&mut spans, &hour.output.events);
        stamps.push(hour.stamps);
        eprintln!("first pass: {}/{} read", index + 1, files.len());
    }
    let (span, protocol) = design.check(&spans)?;

    // Second pass: the windows and the fits' inputs.
    fs::create_dir_all(&spec.out_dir)?;
    let strides = Strides::for_hours(files.len() as u64);
    let mut collect = Collect::new(protocol, span.epoch, strides, &design, &label);
    let mut replay = Replay::new(config);
    let mut filters = new_filters();
    for (index, triple) in files.iter().enumerate() {
        let hour = read_hour(&mut replay, &spec.data_dir, triple, &mut filters)?;
        if stamps.get(index) != Some(&hour.stamps) {
            return Err(format!("{} changed between the two passes", triple.join(" ")).into());
        }
        let read_to = collect.take(&hour.output, &spec.out_dir)?;
        eprintln!("second pass: {}/{} read", index + 1, files.len());
        if read_to.is_some_and(|ms| ms >= protocol.end_ms) {
            break;
        }
    }
    if collect.sizes.is_none() && collect.size_error.is_none() {
        collect.close_sizes();
    }
    if let Some(error) = &collect.size_error {
        return Err(format!("no event sizes, so no window: {error}").into());
    }
    collect.flush(&spec.out_dir, u64::MAX)?;

    let dataset = format!("hyperliquid {COIN}");
    let mut fits = fits(&collect, &dataset)?;
    let arms = fits.get("arms").cloned().unwrap_or(Value::Null);
    if let Some(map) = fits.as_object_mut() {
        map.insert("data".into(), json!(label));
        map.insert("epoch".into(), json!(design.epoch));
        map.insert(
            "inputs_read".into(),
            serde_json::to_value(stamps.iter().flatten().collect::<Vec<_>>())?,
        );
    }
    let mut writer = BufWriter::new(fs::File::create(spec.out_dir.join("fits.json"))?);
    serde_json::to_writer_pretty(&mut writer, &fits)?;
    writer.flush()?;

    let status = |name: &str| arms.get(name).map_or(Value::Null, status_of);
    Ok(json!({
        "data": label,
        "hours": files.len(),
        "strides": strides,
        "windows": collect.summary(),
        "fits": {
            "qr": status("qr"),
            "qr_count": status("qr_count"),
            "qr_constant": status("qr_constant"),
            "hawkes": status("hawkes"),
            "poisson": status("poisson"),
        },
    }))
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [design, data_dir, out_dir] = args.as_slice() else {
        return Err("usage: <design.json> <data dir> <out dir>".into());
    };
    let spec = Run {
        design: PathBuf::from(design),
        data_dir: PathBuf::from(data_dir),
        out_dir: PathBuf::from(out_dir),
        first_hour_ms: parse_time_ns(FIRST_HOUR)? / NS_PER_MS,
        last_hour_ms: parse_time_ns(LAST_HOUR)? / NS_PER_MS,
        warmup_ns: WARMUP_MINUTES * 60 * 1_000_000_000,
    };
    let summary = run(&spec)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use market_maker_rs::backtest::hyperliquid::{ORDER_STATUS_RECORD_BYTES, STATUS_OPEN};

    const CANCELED: u8 = 2;
    const SECOND_NS: u64 = 1_000_000_000;
    const HOURS: u64 = 2;
    const WARMUP_NS: u64 = 60 * SECOND_NS;

    fn t0_ms() -> u64 {
        parse_time_ns("2025-12-21T15:00:00").unwrap() / NS_PER_MS
    }

    fn utc_text(ms: u64) -> String {
        let (year, month, day, hour) = civil(ms);
        let rest = ms / 1_000 % 3_600;
        format!(
            "{year:04}-{month:02}-{day:02}T{hour:02}:{:02}:{:02}",
            rest / 60,
            rest % 60
        )
    }

    fn record(ns: u64, status: u8, is_ask: bool, cents: u32, size_cents: u32, oid: u64) -> Vec<u8> {
        let fixed = |value: u32| (2u32 << 29) | value;
        let mut bytes = vec![0u8; ORDER_STATUS_RECORD_BYTES];
        bytes[0..8].copy_from_slice(&ns.to_le_bytes());
        bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
        bytes[13] = status;
        bytes[14] = u8::from(is_ask);
        bytes[15..19].copy_from_slice(&fixed(cents).to_le_bytes());
        bytes[19..23].copy_from_slice(&fixed(size_cents).to_le_bytes());
        bytes[23..31].copy_from_slice(&oid.to_le_bytes());
        bytes[50..54].copy_from_slice(&fixed(size_cents).to_le_bytes());
        bytes
    }

    /// One hour's three files, as text and bytes.
    #[derive(Default)]
    struct HourFiles {
        statuses: Vec<u8>,
        diffs: String,
        trades: String,
    }

    impl HourFiles {
        fn open(&mut self, ns: u64, oid: u64, ask: bool, price: u32, size: u32) {
            self.statuses
                .extend(record(ns, STATUS_OPEN, ask, price * 100, size * 100, oid));
            let side = if ask { "A" } else { "B" };
            self.diffs.push_str(&format!(
                r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":{{"new":{{"sz":"{size}"}}}}}}"#
            ));
            self.diffs.push('\n');
        }

        fn cancel(&mut self, ns: u64, oid: u64, ask: bool, price: u32) {
            self.statuses
                .extend(record(ns, CANCELED, ask, price * 100, 100, oid));
            let side = if ask { "A" } else { "B" };
            self.diffs.push_str(&format!(
                r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":"remove"}}"#
            ));
            self.diffs.push('\n');
        }
    }

    /// The order of second `a`: its side and price.
    fn order_of(a: u64) -> (bool, u32) {
        let depth = u32::try_from(a / 2 % 3).unwrap();
        let ask = (a / 6) % 2 == 1;
        (ask, if ask { 101 + depth } else { 100 - depth })
    }

    /// Two synthetic hours from 2025-12-21 15:00 UTC: a resting book (bid 100, ask 101), then
    /// each second an order added (even seconds) or cancelled (odd) at depth 0 to 2, with a
    /// second order at the bid's touch added in the same block every 10 s and cancelled in the
    /// block 5 s later, and a trade every 7 s.
    fn synthetic(dir: &Path) -> Vec<[String; 3]> {
        let t0 = t0_ms() * NS_PER_MS;
        let files = hour_files(t0_ms(), t0_ms() + (HOURS - 1) * HOUR_MS);
        for (h, triple) in files.iter().enumerate() {
            let mut hour = HourFiles::default();
            for a in (h as u64 * 3_600)..(h as u64 + 1) * 3_600 {
                let ns = t0 + a * SECOND_NS;
                if a == 0 {
                    hour.open(ns, 1, false, 100, 5);
                    hour.open(ns, 2, true, 101, 5);
                    continue;
                }
                let (ask, price) = order_of(a);
                if a % 2 == 0 {
                    hour.open(ns, 10_000 + a, ask, price, 1);
                } else {
                    hour.cancel(ns, 10_000 + a - 1, ask, price);
                }
                if a % 10 == 0 {
                    hour.open(ns, 20_000 + a, false, 100, 1);
                } else if a % 10 == 5 {
                    hour.cancel(ns, 20_000 + a - 5, false, 100);
                }
                if a % 7 == 0 {
                    let ms = (ns + SECOND_NS / 2) / NS_PER_MS;
                    hour.trades.push_str(&format!(
                        r#"{{"coin":"BTC","side":"B","time":"{}.500000000","px":"101","sz":"0.5"}}"#,
                        utc_text(ms)
                    ));
                    hour.trades.push('\n');
                }
            }
            let [o, d, t] = triple;
            for (path, bytes) in [
                (o, hour.statuses.as_slice()),
                (d, hour.diffs.as_bytes()),
                (t, hour.trades.as_bytes()),
            ] {
                let path = dir.join(path);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, bytes).unwrap();
            }
        }
        files
    }

    fn write_checked(dir: &Path, files: &[&String]) {
        let listed: serde_json::Map<String, Value> =
            files.iter().map(|f| ((*f).clone(), json!("0"))).collect();
        let record = json!({ "label": "SYNTHETIC", "files": listed });
        fs::write(dir.join("checked.json"), record.to_string()).unwrap();
    }

    /// The synthetic run's stretch: warm from a minute in, to the last second.
    fn stretch() -> (u64, u64) {
        (
            t0_ms() + 60 * SECOND_MS,
            t0_ms() + (HOURS * 3_600 - 1) * SECOND_MS,
        )
    }

    fn window_json(start_min: u64, end_min: u64) -> Value {
        json!([
            format!("{}Z", utc_text(t0_ms() + start_min * MINUTE_MS)),
            format!("{}Z", utc_text(t0_ms() + end_min * MINUTE_MS)),
        ])
    }

    fn design_json(start: u64, end: u64) -> Value {
        let p = Protocol::new(start, end).unwrap();
        let w = |w: TimeWindow| {
            json!([
                format!("{}Z", utc_text(w.start_ms)),
                format!("{}Z", utc_text(w.end_ms)),
            ])
        };
        json!({ "main_study": {
            "epoch": { "window_utc": w(TimeWindow { start_ms: start, end_ms: end }) },
            "split": {
                "sizes": w(p.sizes()),
                "queue_train": w(p.queue_train()),
                "validation": w(p.validation()),
                "holdout": w(p.holdout()),
            },
            "windows": {
                "mdqr_train": [window_json(30, 40)],
                "validation": [window_json(80, 90)],
                "holdout": [window_json(100, 110)],
            },
        }})
    }

    /// A fixture dir with the synthetic hours, `checked.json` listing them all, and a design.
    fn fixture(name: &str) -> (PathBuf, Run, Vec<[String; 3]>) {
        let root = std::env::temp_dir().join(format!("mdqr_capture_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let data = root.join("data");
        fs::create_dir_all(&data).unwrap();
        let files = synthetic(&data);
        write_checked(&data, &files.iter().flatten().collect::<Vec<_>>());
        let (start, end) = stretch();
        let design = root.join("design.json");
        fs::write(&design, design_json(start, end).to_string()).unwrap();
        let run = Run {
            design,
            data_dir: data,
            out_dir: root.join("out"),
            first_hour_ms: t0_ms(),
            last_hour_ms: t0_ms() + (HOURS - 1) * HOUR_MS,
            warmup_ns: WARMUP_NS,
        };
        (root, run, files)
    }

    #[test]
    fn test_hour_files_follow_the_capture_names() {
        let files = hour_files(t0_ms(), t0_ms() + 9 * HOUR_MS);
        assert_eq!(files.len(), 10);
        assert_eq!(
            files[0],
            [
                "o/20251221/btc_15.data".to_string(),
                "d/20251221/ex15".to_string(),
                "t/20251221/15".to_string()
            ]
        );
        assert_eq!(files[9][0], "o/20251222/btc_00.data");
        assert_eq!(files[9][1], "d/20251222/ex0");
        let all = hour_files(
            parse_time_ns(FIRST_HOUR).unwrap() / NS_PER_MS,
            parse_time_ns(LAST_HOUR).unwrap() / NS_PER_MS,
        );
        assert_eq!(all.len(), 220, "the design's 220 hours");
    }

    #[test]
    fn test_a_run_is_refused_unless_checked_json_lists_every_file() {
        let (root, spec, files) = fixture("checked");
        // One file left out.
        let listed: Vec<&String> = files.iter().flatten().skip(1).collect();
        write_checked(&spec.data_dir, &listed);
        let error = run(&spec).unwrap_err().to_string();
        assert!(error.contains("does not list 1"), "{error}");
        assert!(error.contains(&files[0][0]), "{error}");
        assert!(!spec.out_dir.exists(), "nothing is written");
        // No checked.json at all.
        fs::remove_file(spec.data_dir.join("checked.json")).unwrap();
        let error = run(&spec).unwrap_err().to_string();
        assert!(error.contains("cannot be read"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_windows_hold_their_observations_blocks_samples_and_history() {
        let (root, spec, _) = fixture("windows");
        let summary = run(&spec).unwrap();
        assert_eq!(summary["data"], "SYNTHETIC");
        assert_eq!(summary["hours"], 2);
        for (set, start_min, end_min) in [
            ("mdqr_train", 30, 40),
            ("validation", 80, 90),
            ("holdout", 100, 110),
        ] {
            let path = spec.out_dir.join(set).join("window-0.json");
            let file: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
            let start = t0_ms() + start_min * MINUTE_MS;
            let end = t0_ms() + end_min * MINUTE_MS;
            assert_eq!(file["window"], json!({ "start_ms": start, "end_ms": end }));
            assert_eq!(file["event_sizes"], json!(["1", "1", "1"]));
            let obs = file["observations"].as_array().unwrap();
            // A second's order each second, and the touch's second order added in the block of
            // each tenth second and cancelled in the block 5 s later.
            assert_eq!(obs.len(), 600 + 60 + 60, "{set}");
            let mut blocks = 0;
            let mut last: Option<&Value> = None;
            for o in obs {
                let ns = o["timestamp_ns"].as_u64().unwrap();
                let ms = o["timestamp_ms"].as_u64().unwrap();
                assert_eq!(ms, ns / NS_PER_MS);
                assert!(start <= ms && ms < end);
                assert_eq!(o["size"].as_f64(), Some(1.0));
                if let Some(prev) = last {
                    let same = prev["timestamp_ns"] == o["timestamp_ns"];
                    let step = o["block"].as_u64().unwrap() - prev["block"].as_u64().unwrap();
                    assert_eq!(step, u64::from(!same));
                    let same_queue = prev["side"] == o["side"] && prev["depth"] == o["depth"];
                    if same && same_queue {
                        assert_eq!(o["waiting_seconds"].as_f64(), Some(0.0));
                    }
                } else {
                    assert_eq!(o["block"], 0);
                }
                blocks = o["block"].as_u64().unwrap() + 1;
                last = Some(o);
            }
            assert_eq!(blocks, 600, "one block a second");
            // A second's order rests at its depth, both kinds seen at each.
            for depth in 0..3 {
                assert!(
                    obs.iter()
                        .any(|o| o["depth"] == depth && o["kind"] == "limit")
                );
                assert!(
                    obs.iter()
                        .any(|o| o["depth"] == depth && o["kind"] == "cancel")
                );
            }
            let samples = file["samples"].as_array().unwrap();
            assert_eq!(samples.len(), 600, "a sample a second");
            assert!(samples.iter().all(|s| {
                let ms = s["timestamp_ms"].as_u64().unwrap();
                start <= ms && ms < end
            }));
            let trades = file["trades"].as_array().unwrap();
            assert!((85..=86).contains(&trades.len()), "{}", trades.len());
            // History: strictly before the start, within 15 min of it.
            let history = &file["history"];
            assert_eq!(history["end_ms"], start);
            assert_eq!(history["start_ms"], start - HISTORY_MS);
            let before = |v: &Value| {
                let ms = v["timestamp_ms"].as_u64().unwrap();
                start - HISTORY_MS <= ms && ms < start
            };
            let h_obs = history["observations"].as_array().unwrap();
            assert_eq!(h_obs.len(), 900 + 90 + 90);
            assert!(h_obs.iter().all(before));
            let h_trades = history["trades"].as_array().unwrap();
            assert!(!h_trades.is_empty() && h_trades.iter().all(before));
            let last_sample = &history["last_sample"];
            assert_eq!(last_sample["timestamp_ms"], start - SECOND_MS);
        }
        let fits: Value =
            serde_json::from_str(&fs::read_to_string(spec.out_dir.join("fits.json")).unwrap())
                .unwrap();
        assert_eq!(fits["data"], "SYNTHETIC");
        for arm in ["qr", "qr_count", "qr_constant", "hawkes", "poisson"] {
            let status = fits["arms"][arm]["status"].as_str().unwrap();
            assert!(
                status == "ok" || fits["arms"][arm]["reason"].is_string(),
                "{arm}"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_protocol_follows_the_design_rule() {
        // 10 001 s: training 6 000.6 s and validation 8 000.8 s, both rounded down to the
        // second, and the size window a tenth of training, 600 s.
        let p = Protocol::new(1_000_000, 1_000_000 + 10_001_000).unwrap();
        assert_eq!(p.train_end_ms, 1_000_000 + 6_000_000);
        assert_eq!(p.validation_end_ms, 1_000_000 + 8_000_000);
        assert_eq!(p.sizes_end_ms, 1_000_000 + 600_000);
        // 100 000 s: the size window is capped at an hour.
        let p = Protocol::new(0, 100_000_000).unwrap();
        assert_eq!(p.sizes_end_ms, HOUR_MS);
        assert_eq!(p.train_end_ms, 60_000_000);
        assert!(
            Protocol::new(500, 100_000_000).is_err(),
            "whole seconds only"
        );
        // The synthetic run's design is built by the same rule.
        let (start, end) = stretch();
        let design = Design::from_json(&design_json(start, end)).unwrap();
        assert_eq!(
            design.queue_train,
            Protocol::new(start, end).unwrap().queue_train()
        );
        // The study's design: Protocol on its epoch gives its split.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("bridges/python/experiments/m5_01/design.json");
        let value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let design = Design::from_json(&value).unwrap();
        let p = Protocol::new(design.epoch.start_ms, design.epoch.end_ms).unwrap();
        assert_eq!(
            [p.sizes(), p.queue_train(), p.validation(), p.holdout()],
            [
                design.sizes,
                design.queue_train,
                design.validation,
                design.holdout
            ]
        );
        assert_eq!(design.epoch.end_ms - design.epoch.start_ms, 789_615_000);
        assert_eq!(
            design.windows.iter().map(|(_, w)| w.len()).sum::<usize>(),
            10
        );
        let span = EpochSpan {
            epoch: 3,
            first_warm_ns: Some(design.epoch.start_ms * NS_PER_MS - 500),
            last_ns: design.epoch.end_ms * NS_PER_MS + 999_999_999,
        };
        assert!(design.check(&[span]).is_ok());
    }

    #[test]
    fn test_the_epoch_check_refuses_a_mismatch() {
        let (start, end) = stretch();
        let design = Design::from_json(&design_json(start, end)).unwrap();
        let span = |start_ms: u64, end_ms: u64| EpochSpan {
            epoch: 1,
            first_warm_ns: Some(start_ms * NS_PER_MS),
            last_ns: end_ms * NS_PER_MS,
        };
        assert!(design.check(&[span(start, end)]).is_ok());
        let error = design.check(&[span(start + SECOND_MS, end)]).unwrap_err();
        assert!(error.contains("longest warm stretch"), "{error}");
        // A longer stretch elsewhere is the one taken, so it is refused too.
        let error = design
            .check(&[span(start, end), span(end + HOUR_MS, end + 3 * HOUR_MS)])
            .unwrap_err();
        assert!(error.contains("longest warm stretch"), "{error}");
        // A split that is not Protocol's.
        let mut other = design.clone();
        other.validation.start_ms += SECOND_MS;
        let error = other.check(&[span(start, end)]).unwrap_err();
        assert!(error.contains("Protocol gives"), "{error}");
        // And a whole run on a design whose epoch is a second off writes nothing.
        let (root, spec, _) = fixture("epoch");
        fs::write(
            &spec.design,
            design_json(start + SECOND_MS, end).to_string(),
        )
        .unwrap();
        let error = run(&spec).unwrap_err().to_string();
        assert!(error.contains("refused"), "{error}");
        assert!(!spec.out_dir.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
