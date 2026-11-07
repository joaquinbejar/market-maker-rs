//! Evaluates, on the Hyperliquid "An Open Book" capture, the criteria of issue #214 that the
//! crate's public interfaces can evaluate as they are, under a protocol fixed here before any
//! data is read.
//!
//! The dataset is real exchange data (Zenodo, DOI 10.5281/zenodo.18184441, CC BY 4.0), and it
//! is not in this repository. This example decompresses nothing: pass the files decompressed,
//! one triple per hour, in time order, as for `hyperliquid_hour`.
//!
//! Run with
//! `cargo run --release --features serde --example hyperliquid_validation -- <coin> <tick> <warmup minutes> <out dir> <statuses> <diffs> <trades> [<statuses> <diffs> <trades> ...]`.
//!
//! The report goes to standard output and to `<out dir>/report.txt`. It names every input with
//! its size and FNV-1a 64 digest, and every declared parameter. The commit the report comes from
//! is the caller's to record beside it.
//!
//! # Protocol
//!
//! Everything below is a constant of this file, so it is fixed before the data is read.
//!
//! - **The stretch.** A reset empties the book (see `BookEvent::epoch`), so nothing here spans
//!   one. The data used is the longest epoch, from its first warm event, rounded up to a whole
//!   second, to its last event, rounded down; the earliest on a tie. Every epoch's span is
//!   printed, and so is the one taken. Finding it needs the whole capture, so the files are read
//!   twice: a first pass replays them to find the epochs, and a second one evaluates.
//! - **The split.** The stretch is cut by time into training ([`TRAIN_PER_MILLE`]), validation
//!   ([`VALIDATION_PER_MILLE`]) and holdout (the rest), in that order, each boundary rounded
//!   down to a whole second. The first [`SIZE_PER_MILLE`] of the training part, at most
//!   [`SIZE_WINDOW_MAX_MS`], gives the event sizes of the queue models and is left out of their
//!   fits, since their observations are counted in those sizes and the sizes must be known
//!   before the observations are formed.
//! - **The holdout is never used for a choice.** The one choice made from data is the decay of
//!   the Hawkes kernel, by `select_decay` on training and validation only; the test at the end
//!   of this file checks that a holdout it never reads cannot change it. Every horizon, window
//!   length, fee and latency is reported, and none is picked.
//! - **No randomness.** Nothing draws a random number: the thinning below is by count. So there
//!   is no seed to fix.
//! - **Fees and funding** are in no file of the capture. The fees are the declared grid
//!   [`FEE_RATES`], not the venue's schedule. Funding is not applied, as the simulator has no
//!   input for it.
//!
//! # What is kept, and the memory it implies
//!
//! A month is 744 hours, and an hour of BTC holds about two million book events, so only what
//! each evaluation needs is kept, and the rest of an hour is dropped once it is read. The thinning
//! by count is by strides that grow with the number of input hours past [`FULL_RATE_HOURS`],
//! which the command line fixes before anything is read, so what it keeps stays about that of
//! [`FULL_RATE_HOURS`] hours. The figures are for the BTC hour of 2025-12-01 10:00 UTC.
//!
//! - One hour at a time: its decoded rows and the replay's output. The whole run on that hour
//!   peaked at 1.6 GB resident, nearly all of it this.
//! - Tops of book, training and holdout only, the last of each second ([`SAMPLE_MS`]): at most
//!   3,600 an hour of 72 bytes, so at most about 150 MB for a month.
//! - Mids for the walk-forward, the last of each second: at most 3,600 an hour of 24 bytes,
//!   at most about 65 MB for a month. Its grid is of one second, so the grid prices are the same.
//! - Mid moves (`ReferenceObservation`), all: about 4,500 an hour of 16 bytes, about 55 MB for a
//!   month.
//! - Order lives (`QuoteExposure`), one in [`EXPOSURE_STRIDE`] at the base rate: about 79,000 an
//!   hour of 40 bytes kept, so about 80 MB at most.
//! - Queue observations with a positive wait, one in [`QUEUE_STRIDE`] at the base rate: about
//!   70,000 an hour of 24 bytes, so about 40 MB at most.
//! - Hawkes segments of the first limit, one in [`SEGMENT_STRIDE`] at the base rate: about
//!   140,000 events an hour of 16 bytes, so about 55 MB at most, and never more than
//!   [`MAX_PART_EVENTS`] events per part of the split.
//! - The sized events of the size window only, at most an hour of them.
//! - Market events of the holdout, one file at a time: the part of the holdout in each file is
//!   simulated on its own, from a flat inventory, and only the totals are kept.
//!
//! So a month needs about 0.5 GB kept beside the hour being read. The report ends with what was
//! kept, counted. Thinning by count, independently of what happened, leaves a rate per second of
//! exposure or of waiting where it was in expectation, and drops evidence: it widens the error of
//! each estimate.

use std::error::Error;
use std::fs;
use std::io::{self, BufReader, Read};
use std::mem::size_of;
use std::path::Path;

use market_maker_rs::Decimal;
use market_maker_rs::analytics::micro_price::{
    EstimatorError, MicroPriceConfig, StateStatus, TopOfBook, calibrate_micro_price,
    compare_estimators,
};
use market_maker_rs::analytics::vpin::VPINConfig;
use market_maker_rs::backtest::hawkes::{
    DecayScore, HawkesFitConfig, HawkesParameters, HawkesSegment, fit_hawkes, fit_poisson,
    score_flow, select_decay,
};
use market_maker_rs::backtest::hyperliquid::{
    BookEvent, Coverage, RepeatFilter, Replay, ReplayConfig, ReplayOutput, Warmth,
    decode_order_statuses_after, hawkes_segments, queue_observations, read_book_diffs_after,
    read_trades_after, sized_events,
};
use market_maker_rs::backtest::queue_reactive::{
    EventSizeRecord, QrEstimationConfig, QrEventKind, QrFit, QrSide, QueueObservation,
    ReferenceObservation, SizedEvent, UnsupportedPolicy, estimate_event_sizes, estimate_parameters,
};
use market_maker_rs::dec;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::strategy::signal_quote::{
    MarketEvent, SignalConfig, SimulationConfig, simulate,
};
use market_maker_rs::strategy::walk_forward::{
    CalibrationData, CalibrationOutcome, CalibrationRecord, ConstantParameters, FoldReport,
    OutOfSampleError, PriceSample, QuoteExposure, TimeWindow, WalkForwardConfig, WindowSpec,
    calibrate_window, chronological_splits, walk_forward,
};
use market_maker_rs::types::error::MMResult;
use rust_decimal::prelude::FromStr;

const NS_PER_MS: u64 = 1_000_000;
const SECOND_MS: u64 = 1_000;
const HOUR_MS: u64 = 3_600_000;
const DAY_MS: u64 = 86_400_000;

// Replay, as in `hyperliquid_hour`.

/// Limits a depth counts up to, and the limits of the queue-reactive model.
const LIMITS: u8 = 3;
/// The silence that resets the replay. The capture's own gaps are of minutes.
const RESET_GAP_NS: u64 = 60 * 1_000_000_000;

// The split.

/// Training share of the stretch, per mille.
const TRAIN_PER_MILLE: u64 = 600;
/// Validation share of the stretch, per mille. The holdout is the rest.
const VALIDATION_PER_MILLE: u64 = 200;
/// Share of the training part that gives the event sizes, per mille.
const SIZE_PER_MILLE: u64 = 100;
/// Longest window the event sizes come from.
const SIZE_WINDOW_MAX_MS: u64 = HOUR_MS;
/// Fewest events per limit for an event size.
const MIN_SIZED_EVENTS: u64 = 30;

// Thinning, by count or by time, never by outcome.

/// Tops and mids keep the last of each such bucket of the clock.
const SAMPLE_MS: u64 = SECOND_MS;
/// Input hours kept at the base strides below. With more, each stride is multiplied by
/// `ceil(hours / FULL_RATE_HOURS)`, which the number of files fixes before anything is read, so
/// what is kept stays about that of this many hours.
const FULL_RATE_HOURS: u64 = 24;
/// One order life kept in this many, at the base rate.
const EXPOSURE_STRIDE: u64 = 10;
/// One queue observation with a positive wait kept in this many, at the base rate.
const QUEUE_STRIDE: u64 = 1;
/// One Hawkes segment kept in this many, at the base rate.
const SEGMENT_STRIDE: u64 = 1;
/// Most Hawkes events kept per part of the split, a bound the strides should not reach; past it,
/// segments are counted and dropped.
const MAX_PART_EVENTS: usize = 4_000_000;

// #87, walk-forward calibration.

/// Price grid of the walk-forward.
const WF_RESOLUTION_MS: u64 = SECOND_MS;
/// Longest silence of the mid, in grid buckets: the reset gap. A mid that does not move leaves
/// no sample, and a real gap that long would have reset the replay.
const WF_MAX_GAP_BUCKETS: u32 = 60;
/// Training windows of the folds, in days. Each fold tests on the next whole UTC day.
const WF_TRAIN_DAYS: [u64; 3] = [1, 3, 7];

// #88, micro-price.

/// Imbalance buckets of the micro-price, odd.
const MICRO_BUCKETS: u32 = 5;
/// Spreads of 1 to this many ticks have states.
const MICRO_SPREADS: u32 = 20;
/// Fewest pairs for a state to be calibrated.
const MICRO_MIN_STATE: u64 = 100;
/// Longest time between the quotes of a pair.
const MICRO_MAX_GAP_MS: u64 = 60_000;
/// Horizons of the comparison. The tops are sampled per second, so none is shorter.
const HORIZONS_MS: [u64; 4] = [1_000, 5_000, 30_000, 60_000];

// #89, quoting simulation.

/// Fee on the notional of each fill: a rebate, none and a fee. Declared, not the venue's.
const FEE_RATES: [Decimal; 3] = [dec!(-0.00003), dec!(0), dec!(0.00015)];
/// Time from a decision until its quote rests.
const LATENCIES_MS: [u64; 4] = [0, 100, 500, 2_000];
/// Risk aversion of the inventory quotes, per unit of the coin and of the quote currency.
const RISK_AVERSION: Decimal = dec!(0.00001);
/// Size per side, in units of the coin.
const BASE_SIZE: Decimal = dec!(0.01);
/// Largest absolute position, in units of the coin.
const MAX_POSITION: Decimal = dec!(0.1);
/// Largest notional, in the quote currency.
const MAX_NOTIONAL: Decimal = dec!(100000000);
/// Size scaling of the limits.
const LIMIT_SCALING: Decimal = dec!(1);
/// Markout horizon of the simulation.
const SIM_MARKOUT_MS: u64 = 1_000;

// #90, queue-reactive model.

/// Queue cap, in event sizes.
const QR_MAX_QUEUE: u32 = 200;
/// Fewest events for a queue size to be supported.
const QR_MIN_EVENTS: u64 = 30;
/// Fewest mid moves for `θ`.
const QR_MIN_REFERENCE_MOVES: u64 = 30;
/// Queue sizes of the first limit whose rates each day's estimate prints.
const QR_DAY_SIZES: [usize; 5] = [1, 2, 5, 10, 20];

// #91, Poisson, queue-reactive and Hawkes flow on the first limit.

/// Queue cap, in event sizes. A segment that would pass it is left out and counted.
const HAWKES_MAX_QUEUE: u32 = 200;
/// Fewest events for a queue size's baseline.
const HAWKES_MIN_EVENTS: u64 = 30;
/// Expectation maximization steps.
const HAWKES_ITERATIONS: u32 = 200;
/// Candidate decays, per second, for `select_decay` on validation.
const DECAYS: [f64; 4] = [0.1, 1.0, 10.0, 100.0];

/// FNV-1a 64 offset and prime, to tell inputs apart in a report.
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
#[derive(Debug, Clone, PartialEq, Eq)]
struct Stamp {
    path: String,
    bytes: u64,
    fnv: u64,
}

/// The report, written as it goes.
#[derive(Default)]
struct Report {
    text: String,
}

impl Report {
    fn line(&mut self, line: impl AsRef<str>) {
        println!("{}", line.as_ref());
        self.text.push_str(line.as_ref());
        self.text.push('\n');
    }
}

macro_rules! say {
    ($report:expr, $($arg:tt)*) => {
        $report.line(format!($($arg)*))
    };
}

/// One hour, read and replayed.
struct Hour {
    output: ReplayOutput,
    stamps: [Stamp; 3],
}

/// Reads one triple through the streams' filters and replays it.
fn read_hour(
    replay: &mut Replay,
    coin: &str,
    triple: &[String],
    filters: &mut [RepeatFilter; 3],
) -> Result<Hour, Box<dyn Error>> {
    let [statuses, diffs, trades] = triple else {
        return Err("files come in triples of statuses, diffs and trades".into());
    };
    let [status_filter, diff_filter, trade_filter] = filters;
    let bytes = fs::read(statuses)?;
    let status_stamp = Stamp {
        path: statuses.clone(),
        bytes: bytes.len() as u64,
        fnv: fnv(FNV.0, &bytes),
    };
    let status_rows = decode_order_statuses_after(&bytes, status_filter)?;
    drop(bytes);
    let mut reader = BufReader::new(Digesting::new(fs::File::open(diffs)?));
    let diff_rows = read_book_diffs_after(&mut reader, coin, diff_filter)?;
    io::copy(&mut reader, &mut io::sink())?;
    let diff_stamp = Stamp {
        path: diffs.clone(),
        bytes: reader.get_ref().bytes,
        fnv: reader.get_ref().hash,
    };
    let mut reader = BufReader::new(Digesting::new(fs::File::open(trades)?));
    let trade_rows = read_trades_after(&mut reader, coin, trade_filter)?;
    io::copy(&mut reader, &mut io::sink())?;
    let trade_stamp = Stamp {
        path: trades.clone(),
        bytes: reader.get_ref().bytes,
        fnv: reader.get_ref().hash,
    };
    let output = replay.ingest(&status_rows.rows, &diff_rows.rows, &trade_rows.rows)?;
    Ok(Hour {
        output,
        stamps: [status_stamp, diff_stamp, trade_stamp],
    })
}

fn filters() -> [RepeatFilter; 3] {
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
    first_ns: u64,
    first_warm_ns: Option<u64>,
    last_ns: u64,
    events: u64,
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

/// The first pass: digests, epochs and coverage.
struct Scan {
    stamps: Vec<Stamp>,
    spans: Vec<EpochSpan>,
    coverage: Coverage,
}

fn scan(coin: &str, config: ReplayConfig, files: &[String]) -> Result<Scan, Box<dyn Error>> {
    let mut replay = Replay::new(config);
    let mut filters = filters();
    let mut spans: Vec<EpochSpan> = Vec::new();
    let mut stamps = Vec::with_capacity(files.len());
    let hours = files.len() / 3;
    for (index, triple) in files.chunks(3).enumerate() {
        let hour = read_hour(&mut replay, coin, triple, &mut filters)?;
        for event in &hour.output.events {
            let ns = event.time.timestamp_ns;
            match spans.last_mut() {
                Some(span) if span.epoch == event.epoch => {
                    span.last_ns = span.last_ns.max(ns);
                    span.events += 1;
                    if event.warmth == Warmth::Warm && span.first_warm_ns.is_none() {
                        span.first_warm_ns = Some(ns);
                    }
                }
                _ => spans.push(EpochSpan {
                    epoch: event.epoch,
                    first_ns: ns,
                    first_warm_ns: (event.warmth == Warmth::Warm).then_some(ns),
                    last_ns: ns,
                    events: 1,
                }),
            }
        }
        eprintln!("first pass: {}/{hours} read", index + 1);
        stamps.extend(hour.stamps);
    }
    Ok(Scan {
        stamps,
        spans,
        coverage: replay.coverage().clone(),
    })
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

/// A part of the split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Train,
    Validation,
    Holdout,
}

/// The chronological split of the stretch, in milliseconds since the Unix epoch. Every
/// boundary is a whole second.
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
    /// Splits `[start_ms, end_ms)`, or says why it cannot.
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

    /// The part a time falls in, if it is inside the stretch.
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

    fn window(start_ms: u64, end_ms: u64) -> TimeWindow {
        TimeWindow { start_ms, end_ms }
    }

    fn stretch(&self) -> TimeWindow {
        Self::window(self.start_ms, self.end_ms)
    }

    fn sizes(&self) -> TimeWindow {
        Self::window(self.start_ms, self.sizes_end_ms)
    }

    fn train(&self) -> TimeWindow {
        Self::window(self.start_ms, self.train_end_ms)
    }

    /// Training of the queue models: after the size window.
    fn queue_train(&self) -> TimeWindow {
        Self::window(self.sizes_end_ms, self.train_end_ms)
    }

    fn validation(&self) -> TimeWindow {
        Self::window(self.train_end_ms, self.validation_end_ms)
    }

    fn holdout(&self) -> TimeWindow {
        Self::window(self.validation_end_ms, self.end_ms)
    }

    /// Whole UTC days inside the stretch.
    fn days(&self) -> Vec<TimeWindow> {
        let mut days = Vec::new();
        let mut start = self.start_ms.div_ceil(DAY_MS) * DAY_MS;
        while start + DAY_MS <= self.end_ms {
            days.push(Self::window(start, start + DAY_MS));
            start += DAY_MS;
        }
        days
    }
}

/// Hawkes segments by part of the split, with their events counted.
#[derive(Debug, Default)]
struct HawkesParts {
    train: Vec<HawkesSegment>,
    validation: Vec<HawkesSegment>,
    holdout: Vec<HawkesSegment>,
    events: [usize; 3],
}

impl HawkesParts {
    fn get_mut(&mut self, part: Part) -> (&mut Vec<HawkesSegment>, &mut usize) {
        let [train, validation, holdout] = &mut self.events;
        match part {
            Part::Train => (&mut self.train, train),
            Part::Validation => (&mut self.validation, validation),
            Part::Holdout => (&mut self.holdout, holdout),
        }
    }
}

/// The one choice made from data, on training and validation only: the holdout is not passed.
fn choose_decay(
    parts: &HawkesParts,
    config: &HawkesFitConfig,
) -> MMResult<(Vec<DecayScore>, Option<usize>)> {
    select_decay(&parts.train, &parts.validation, &DECAYS, config)
}

fn hawkes_config(decays: Vec<f64>) -> HawkesFitConfig {
    HawkesFitConfig::new(HAWKES_MAX_QUEUE, decays)
        .with_iterations(HAWKES_ITERATIONS)
        .with_min_events(HAWKES_MIN_EVENTS)
}

/// Whether a path stays within the cap, as the fits require.
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

/// Keeps the last item of each bucket of the clock.
fn keep_last<T>(kept: &mut Vec<T>, item: T, time: impl Fn(&T) -> u64) {
    let bucket = time(&item) / SAMPLE_MS;
    match kept.last_mut() {
        Some(last) if time(last) / SAMPLE_MS == bucket => *last = item,
        _ => kept.push(item),
    }
}

/// Keeps one in `stride` by count.
fn keep_one_in(seen: &mut u64, stride: u64) -> bool {
    let keep = seen.is_multiple_of(stride);
    *seen += 1;
    keep
}

/// Events inside `[start_ms, end_ms)`, from events in time order.
fn between(events: &[BookEvent], start_ms: u64, end_ms: u64) -> &[BookEvent] {
    let from = events.partition_point(|e| e.time.timestamp_ns < start_ms * NS_PER_MS);
    let to = events.partition_point(|e| e.time.timestamp_ns < end_ms * NS_PER_MS);
    events.get(from..to).unwrap_or_default()
}

/// The totals of one point of the fee and latency grid, over the holdout's hours.
#[derive(Debug, Clone)]
struct SimCell {
    fee: Decimal,
    latency_ms: u64,
    files: u64,
    decisions: u64,
    crossing_sides: u64,
    fills: u64,
    volume: Decimal,
    spread_capture: Decimal,
    inventory_pnl: Decimal,
    fees: Decimal,
    total_pnl: Decimal,
    max_abs_inventory: Decimal,
    markout_sum: Decimal,
    markout_volume: Decimal,
    error: Option<String>,
}

impl SimCell {
    fn new(fee: Decimal, latency_ms: u64) -> Self {
        Self {
            fee,
            latency_ms,
            files: 0,
            decisions: 0,
            crossing_sides: 0,
            fills: 0,
            volume: Decimal::ZERO,
            spread_capture: Decimal::ZERO,
            inventory_pnl: Decimal::ZERO,
            fees: Decimal::ZERO,
            total_pnl: Decimal::ZERO,
            max_abs_inventory: Decimal::ZERO,
            markout_sum: Decimal::ZERO,
            markout_volume: Decimal::ZERO,
            error: None,
        }
    }
}

/// The strides of the thinning by count, from the number of input hours alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Strides {
    factor: u64,
    exposure: u64,
    queue: u64,
    segment: u64,
}

impl Strides {
    fn for_hours(hours: u64) -> Self {
        let factor = hours.div_ceil(FULL_RATE_HOURS).max(1);
        Self {
            factor,
            exposure: EXPOSURE_STRIDE * factor,
            queue: QUEUE_STRIDE * factor,
            segment: SEGMENT_STRIDE * factor,
        }
    }
}

/// Size records: the queue model's limits, and the first limit alone for the Hawkes flow.
struct Sizes {
    queue: EventSizeRecord,
    hawkes: EventSizeRecord,
}

/// What the second pass keeps.
struct Collect {
    protocol: Protocol,
    epoch: u64,
    tick: Decimal,
    dataset: String,
    train_tops: Vec<TopOfBook>,
    holdout_tops: Vec<TopOfBook>,
    prices: Vec<PriceSample>,
    exposures: Vec<QuoteExposure>,
    exposures_seen: u64,
    references: Vec<ReferenceObservation>,
    sized: Vec<SizedEvent>,
    sizes: Option<Sizes>,
    strides: Strides,
    unsorted_hours: u64,
    queue: Vec<QueueObservation>,
    queue_seen: u64,
    queue_left_out: [u64; 4],
    queue_error: Option<String>,
    hawkes: HawkesParts,
    segments_seen: u64,
    hawkes_left_out: [u64; 7],
    hawkes_inherited: u64,
    hawkes_error: Option<String>,
    train_record: Option<Result<CalibrationRecord, String>>,
    sim: Vec<SimCell>,
    sim_note: Option<String>,
}

impl Collect {
    fn new(
        protocol: Protocol,
        epoch: u64,
        tick: Decimal,
        dataset: String,
        strides: Strides,
    ) -> Self {
        let sim = FEE_RATES
            .iter()
            .flat_map(|fee| {
                LATENCIES_MS
                    .iter()
                    .map(|latency| SimCell::new(*fee, *latency))
            })
            .collect();
        Self {
            protocol,
            epoch,
            tick,
            dataset,
            train_tops: Vec::new(),
            holdout_tops: Vec::new(),
            prices: Vec::new(),
            exposures: Vec::new(),
            exposures_seen: 0,
            references: Vec::new(),
            sized: Vec::new(),
            sizes: None,
            strides,
            unsorted_hours: 0,
            queue: Vec::new(),
            queue_seen: 0,
            queue_left_out: [0; 4],
            queue_error: None,
            hawkes: HawkesParts::default(),
            segments_seen: 0,
            hawkes_left_out: [0; 7],
            hawkes_inherited: 0,
            hawkes_error: None,
            train_record: None,
            sim,
            sim_note: None,
        }
    }

    fn part(&self, epoch: u64, ns: u64) -> Option<Part> {
        if epoch == self.epoch {
            self.protocol.part(ns / NS_PER_MS)
        } else {
            None
        }
    }

    /// Takes what is needed from one hour's output. True once the stretch has ended.
    fn take(&mut self, output: &ReplayOutput) -> bool {
        for top in &output.tops {
            match self.part(top.epoch, top.time.timestamp_ns) {
                Some(Part::Train) => keep_last(&mut self.train_tops, top.value, |t| t.timestamp_ms),
                Some(Part::Holdout) => {
                    keep_last(&mut self.holdout_tops, top.value, |t| t.timestamp_ms);
                }
                _ => {}
            }
        }
        for price in &output.prices {
            if self.part(price.epoch, price.time.timestamp_ns).is_some() {
                keep_last(&mut self.prices, price.value, |p| p.timestamp_ms);
            }
        }
        for exposure in &output.exposures {
            if self
                .part(exposure.epoch, exposure.time.timestamp_ns)
                .is_some()
                && keep_one_in(&mut self.exposures_seen, self.strides.exposure)
            {
                self.exposures.push(exposure.value);
            }
        }
        for reference in &output.references {
            if self
                .part(reference.epoch, reference.time.timestamp_ns)
                .is_some()
            {
                self.references.push(reference.value);
            }
        }
        self.take_events(&output.events);
        let holdout: Vec<MarketEvent> = output
            .market_events
            .iter()
            .filter(|e| self.part(e.epoch, e.time.timestamp_ns) == Some(Part::Holdout))
            .map(|e| e.value.clone())
            .collect();
        if !holdout.is_empty() {
            self.simulate_hour(&holdout);
        }
        output.events.last().is_some_and(|last| {
            last.epoch > self.epoch || last.time.timestamp_ns / NS_PER_MS >= self.protocol.end_ms
        })
    }

    fn take_events(&mut self, all: &[BookEvent]) {
        let (Some(first), Some(last)) = (
            all.iter().position(|e| e.epoch == self.epoch),
            all.iter().rposition(|e| e.epoch == self.epoch),
        ) else {
            return;
        };
        let events = all.get(first..=last).unwrap_or_default();
        if !events.is_sorted_by_key(|e| (e.epoch, e.time.timestamp_ns)) {
            self.unsorted_hours += 1;
            return;
        }
        let p = self.protocol;
        if self.sizes.is_none() {
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
            return;
        }
        for (part, start, end) in [
            (Part::Train, p.sizes_end_ms, p.train_end_ms),
            (Part::Validation, p.train_end_ms, p.validation_end_ms),
            (Part::Holdout, p.validation_end_ms, p.end_ms),
        ] {
            let slice = between(events, start, end);
            if !slice.is_empty() {
                self.convert(part, slice);
            }
        }
    }

    fn close_sizes(&mut self) {
        let window = self.protocol.sizes();
        let sized = std::mem::take(&mut self.sized);
        let estimate = |limits| estimate_event_sizes(&sized, &window, limits, MIN_SIZED_EVENTS);
        match (estimate(LIMITS), estimate(1)) {
            (Ok(queue), Ok(hawkes)) => self.sizes = Some(Sizes { queue, hawkes }),
            (Err(error), _) | (_, Err(error)) => {
                let text = format!("estimate_event_sizes failed: {error}");
                self.queue_error = Some(text.clone());
                self.hawkes_error = Some(text);
            }
        }
        if self.sizes.is_none() && self.queue_error.is_none() {
            self.queue_error = Some("no event sizes".to_string());
        }
    }

    fn convert(&mut self, part: Part, events: &[BookEvent]) {
        let Some(sizes) = &self.sizes else {
            return;
        };
        if let (Ok(event_sizes), None) = (&sizes.queue.sizes, &self.queue_error) {
            match queue_observations(events, event_sizes) {
                Ok(conversion) => {
                    self.queue_left_out[0] += conversion.cold;
                    self.queue_left_out[1] += conversion.unknown_cause;
                    self.queue_left_out[2] += conversion.no_depth;
                    self.queue_left_out[3] += conversion.zero_waits;
                    for observation in conversion.observations {
                        if observation.value.waiting_seconds > 0.0
                            && keep_one_in(&mut self.queue_seen, self.strides.queue)
                        {
                            self.queue.push(observation.value);
                        }
                    }
                }
                Err(error) => self.queue_error = Some(format!("queue_observations: {error}")),
            }
        }
        let first = sizes
            .hawkes
            .sizes
            .as_ref()
            .ok()
            .and_then(|s| s.first())
            .copied();
        let (Some(size), None) = (first, &self.hawkes_error) else {
            return;
        };
        for side in [QrSide::Bid, QrSide::Ask] {
            let conversion = match hawkes_segments(events, side, size) {
                Ok(conversion) => conversion,
                Err(error) => {
                    self.hawkes_error = Some(format!("hawkes_segments: {error}"));
                    return;
                }
            };
            self.hawkes_left_out[0] += conversion.cold;
            self.hawkes_left_out[1] += conversion.unknown_cause;
            self.hawkes_left_out[2] += conversion.zero_length;
            self.hawkes_left_out[3] += conversion.inconsistent;
            self.hawkes_left_out[4] += conversion.cut_by_reset;
            for record in conversion.segments {
                if !within_cap(&record.segment, HAWKES_MAX_QUEUE) {
                    self.hawkes_left_out[5] += 1;
                    continue;
                }
                if !keep_one_in(&mut self.segments_seen, self.strides.segment) {
                    continue;
                }
                let (kept, events) = self.hawkes.get_mut(part);
                if *events + record.segment.events.len() > MAX_PART_EVENTS {
                    self.hawkes_left_out[6] += 1;
                    continue;
                }
                *events += record.segment.events.len();
                self.hawkes_inherited += record.inherited_events;
                kept.push(record.segment);
            }
        }
    }

    /// The calibration on the training part, once all of it has been read.
    fn train_record(&mut self) -> &Result<CalibrationRecord, String> {
        if self.train_record.is_none() {
            let record = CalibrationData::new(&self.prices, &self.exposures)
                .and_then(|data| {
                    calibrate_window(&data, &self.protocol.train(), &wf_config(), &self.dataset)
                })
                .map_err(|error| error.to_string());
            self.train_record = Some(record);
        }
        self.train_record
            .get_or_insert_with(|| Err("not calibrated".to_string()))
    }

    /// Simulates the part of the holdout in one file at every point of the grid.
    fn simulate_hour(&mut self, events: &[MarketEvent]) {
        let tick = self.tick;
        let parameters = match self.train_record() {
            Ok(record) => match &record.outcome {
                CalibrationOutcome::Calibrated(p) => Ok((p.volatility, p.order_intensity_k)),
                other => Err(format!(
                    "not run: the training part gave no volatility and k ({other:?})"
                )),
            },
            Err(error) => Err(format!("not run: calibrate_window failed: {error}")),
        };
        let (volatility, k) = match parameters {
            Ok(parameters) => parameters,
            Err(note) => {
                self.sim_note = Some(note);
                return;
            }
        };
        let terminal_ms = self.protocol.end_ms.min(
            events
                .iter()
                .map(|e| match e {
                    MarketEvent::Quote(q) => q.timestamp_ms,
                    MarketEvent::Trade(t) => t.timestamp,
                    _ => 0,
                })
                .max()
                .unwrap_or(0)
                + 1,
        );
        let limits = RiskLimits::new(MAX_POSITION, MAX_NOTIONAL, LIMIT_SCALING);
        let vpin = VPINConfig::new(dec!(1), 50, dec!(0.7));
        let (Ok(limits), Ok(vpin)) = (limits, vpin) else {
            self.sim_note = Some("not run: invalid declared limits".to_string());
            return;
        };
        let signals = SignalConfig::baseline(Decimal::ZERO, Decimal::ZERO, Decimal::ONE);
        for cell in &mut self.sim {
            if cell.error.is_some() {
                continue;
            }
            let config = SimulationConfig::new(
                RISK_AVERSION,
                volatility,
                k,
                terminal_ms,
                BASE_SIZE,
                limits.clone(),
                vpin.clone(),
            )
            .with_tick_size(tick)
            .with_latency_ms(cell.latency_ms)
            .with_fee_rate(cell.fee)
            .with_markout_horizon_ms(SIM_MARKOUT_MS);
            match simulate(events, &config, &signals, None) {
                Ok(report) => {
                    cell.files += 1;
                    cell.decisions += report.decisions;
                    cell.crossing_sides += report.crossing_sides;
                    cell.fills += report.fills;
                    cell.volume += report.volume;
                    cell.spread_capture += report.spread_capture;
                    cell.inventory_pnl += report.inventory_pnl;
                    cell.fees += report.fees;
                    cell.total_pnl += report.total_pnl;
                    cell.max_abs_inventory = cell.max_abs_inventory.max(report.max_abs_inventory);
                    if let Some(markout) = report.markout {
                        cell.markout_sum += markout * report.volume;
                        cell.markout_volume += report.volume;
                    }
                }
                Err(error) => cell.error = Some(error.to_string()),
            }
        }
    }
}

fn wf_config() -> WalkForwardConfig {
    WalkForwardConfig::new()
        .with_resolution_ms(WF_RESOLUTION_MS)
        .with_max_gap_buckets(WF_MAX_GAP_BUCKETS)
}

/// `YYYY-MM-DD HH:MM:SS` in UTC.
fn utc(ms: u64) -> String {
    let seconds = ms / 1_000;
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let rest = seconds % 86_400;
    // Days to a civil date, after H. Hinnant's algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

fn window_text(window: &TimeWindow) -> String {
    format!(
        "[{}, {}) UTC, {:.2} h",
        utc(window.start_ms),
        utc(window.end_ms),
        (window.end_ms - window.start_ms) as f64 / HOUR_MS as f64
    )
}

fn decimal(value: Option<Decimal>) -> String {
    value.map_or("-".to_string(), |v| v.round_dp(6).to_string())
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [coin, tick, warmup, out_dir, files @ ..] = args.as_slice() else {
        return Err(
            "usage: <coin> <tick> <warmup minutes> <out dir> (<statuses> <diffs> <trades>)+".into(),
        );
    };
    if files.is_empty() || files.len() % 3 != 0 {
        return Err("files come in triples of statuses, diffs and trades".into());
    }
    fs::create_dir_all(out_dir)?;
    let tick = Decimal::from_str(tick)?;
    let minutes: u64 = warmup.parse()?;
    let config = ReplayConfig::new(tick, LIMITS, minutes * 60 * 1_000_000_000, RESET_GAP_NS)?;
    let mut report = Report::default();
    say!(
        report,
        "REAL DATA: the Hyperliquid capture, for {coin}. Validation of the criteria of #214 that \
         existing interfaces evaluate."
    );
    say!(
        report,
        "Times of book diffs, so of tops, mids and order lives, are inferred per block; trade \
         times are observed."
    );
    let strides = Strides::for_hours((files.len() / 3) as u64);
    print_config(&mut report, coin, tick, minutes, strides);

    let scan = scan(coin, config, files)?;
    say!(report, "\n== Inputs ==");
    for stamp in &scan.stamps {
        say!(
            report,
            "{}: {} bytes, fnv {:016x}",
            stamp.path,
            stamp.bytes,
            stamp.fnv
        );
    }
    print_coverage(&mut report, &scan.coverage);

    say!(report, "\n== Stretch and split ==");
    say!(report, "Epochs (a reset starts the next):");
    for span in scan.spans.iter().take(50) {
        say!(
            report,
            "  epoch {}: {} events from {}, warm from {}, to {}; warm stretch {:.2} h",
            span.epoch,
            span.events,
            utc(span.first_ns / NS_PER_MS),
            span.first_warm_ns
                .map_or("never".to_string(), |ns| utc(ns / NS_PER_MS)),
            utc(span.last_ns / NS_PER_MS),
            span.stretch_len_ms() as f64 / HOUR_MS as f64,
        );
    }
    if scan.spans.len() > 50 {
        say!(report, "  ... {} epochs in all", scan.spans.len());
    }
    let chosen = longest(&scan.spans).and_then(|span| {
        let (start, end) = span.stretch_ms()?;
        Some((span, Protocol::new(start, end)))
    });
    let (span, protocol) = match chosen {
        Some((span, Ok(protocol))) => (span, protocol),
        Some((span, Err(reason))) => {
            say!(
                report,
                "Epoch {} is the longest, and cannot be split: {reason}. Nothing evaluated.",
                span.epoch
            );
            return finish(&report, out_dir);
        }
        None => {
            say!(report, "No warm stretch. Nothing evaluated.");
            return finish(&report, out_dir);
        }
    };
    say!(
        report,
        "Taken: epoch {}, the longest warm stretch, {}",
        span.epoch,
        window_text(&protocol.stretch())
    );
    say!(
        report,
        "Training   {}\n  of which event sizes {}\nValidation {}\nHoldout    {}",
        window_text(&protocol.train()),
        window_text(&protocol.sizes()),
        window_text(&protocol.validation()),
        window_text(&protocol.holdout())
    );
    let days = protocol.days();
    say!(report, "Whole UTC days in the stretch: {}", days.len());

    let mut collect = Collect::new(
        protocol,
        span.epoch,
        tick,
        format!("hyperliquid {coin}"),
        strides,
    );
    let mut replay = Replay::new(config);
    let mut filters = filters();
    let hours = files.len() / 3;
    for (index, triple) in files.chunks(3).enumerate() {
        let hour = read_hour(&mut replay, coin, triple, &mut filters)?;
        let before = scan
            .stamps
            .get(index * 3..index * 3 + 3)
            .unwrap_or_default();
        if before != hour.stamps.as_slice() {
            return Err(format!("{} changed between the two passes", triple.join(" ")).into());
        }
        let ended = collect.take(&hour.output);
        eprintln!("second pass: {}/{hours} read", index + 1);
        if ended {
            say!(
                report,
                "Second pass stopped after file triple {} of {hours}: the stretch had ended.",
                index + 1
            );
            break;
        }
    }
    if collect.sizes.is_none() && collect.queue_error.is_none() {
        collect.close_sizes();
    }
    if collect.unsorted_hours > 0 {
        say!(
            report,
            "{} hours had events out of time order and gave no queue or Hawkes inputs.",
            collect.unsorted_hours
        );
    }

    section_87(&mut report, &mut collect, &days);
    section_88(&mut report, &collect);
    section_89(&mut report, &collect);
    section_90(&mut report, &collect, &days);
    section_91(&mut report, &collect);
    section_rest(&mut report);
    section_kept(&mut report, &collect);
    finish(&report, out_dir)
}

fn finish(report: &Report, out_dir: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(out_dir).join("report.txt");
    fs::write(&path, &report.text)?;
    eprintln!("report written to {}", path.display());
    Ok(())
}

fn print_config(report: &mut Report, coin: &str, tick: Decimal, minutes: u64, strides: Strides) {
    say!(
        report,
        "\n== Configuration, fixed before the data is read =="
    );
    say!(
        report,
        "Coin {coin}; tick {tick} (declared); warm-up {minutes} min; limits {LIMITS}; reset gap {} s",
        RESET_GAP_NS / 1_000_000_000
    );
    say!(
        report,
        "Split of the longest warm stretch with no reset, by time: training {} per mille, \
         validation {} per mille, holdout the rest; boundaries in whole seconds. Event sizes \
         from the first {} per mille of training, at most {} min, left out of the queue fits; \
         at least {MIN_SIZED_EVENTS} events per limit.",
        TRAIN_PER_MILLE,
        VALIDATION_PER_MILLE,
        SIZE_PER_MILLE,
        SIZE_WINDOW_MAX_MS / 60_000
    );
    say!(
        report,
        "Randomness: none; thinning is by count or by clock, so there is no seed."
    );
    say!(
        report,
        "Thinning: tops and mids, the last of each {SAMPLE_MS} ms; by count, base strides \
         {EXPOSURE_STRIDE} for order lives, {QUEUE_STRIDE} for queue observations with a \
         positive wait and {SEGMENT_STRIDE} for Hawkes segments, times {} = ceil(input hours / \
         {FULL_RATE_HOURS}): 1 in {}, {} and {}; at most {MAX_PART_EVENTS} Hawkes events per \
         part.",
        strides.factor,
        strides.exposure,
        strides.queue,
        strides.segment
    );
    say!(
        report,
        "#87: grid {WF_RESOLUTION_MS} ms, longest mid silence {WF_MAX_GAP_BUCKETS} buckets, \
         other settings WalkForwardConfig::new(); training windows {WF_TRAIN_DAYS:?} days, \
         test one UTC day, step one day; constant parameters: the training part's calibration."
    );
    say!(
        report,
        "#88: {MICRO_BUCKETS} imbalance buckets, spreads 1 to {MICRO_SPREADS} ticks, at least \
         {MICRO_MIN_STATE} pairs per state, pairs at most {MICRO_MAX_GAP_MS} ms apart; horizons \
         {HORIZONS_MS:?} ms."
    );
    say!(
        report,
        "#89: fee rates {:?} on notional (declared, not the venue's); latencies {LATENCIES_MS:?} \
         ms; funding not applied (no input for it); risk aversion {RISK_AVERSION}; size \
         {BASE_SIZE}; position limit {MAX_POSITION}; notional limit {MAX_NOTIONAL}; scaling \
         {LIMIT_SCALING}; markout {SIM_MARKOUT_MS} ms; no signal features; volatility and k \
         from the training part's calibration.",
        FEE_RATES.map(|f| f.to_string())
    );
    say!(
        report,
        "#90: queue cap {QR_MAX_QUEUE} event sizes, at least {QR_MIN_EVENTS} events per size, \
         {QR_MIN_REFERENCE_MOVES} mid moves; nearest supported size where unsupported; per day \
         the rates of sizes {QR_DAY_SIZES:?} of the first limit."
    );
    say!(
        report,
        "#91: first limit, both sides pooled; queue cap {HAWKES_MAX_QUEUE}; at least \
         {HAWKES_MIN_EVENTS} events per size; {HAWKES_ITERATIONS} steps; decays {DECAYS:?} per \
         second, chosen on validation."
    );
}

fn print_coverage(report: &mut Report, coverage: &Coverage) {
    say!(report, "\n== Replay coverage, all files ==");
    say!(
        report,
        "Read {} statuses, {} diffs, {} trades. Times of diffs: {} joined, {} inherited ({} \
         stepping back). Unknown orders after the warm-up: {}. Crossed events: {}.",
        coverage.statuses,
        coverage.diffs,
        coverage.trades,
        coverage.joined_times,
        coverage.unjoined_new + coverage.unjoined_update + coverage.unjoined_remove,
        coverage.time_steps_back,
        coverage.unknown_orders_warm,
        coverage.crossed_events,
    );
    say!(
        report,
        "Resets: {} on diffs, {} on trades, {} orders cut. Longest gap between diffs {:.3} s, \
         between trades {:.3} s. Prices seen are multiples of {:?}, {} off the declared tick.",
        coverage.resets_on_diffs,
        coverage.resets_on_trades,
        coverage.orders_cut_by_reset,
        coverage.longest_gap_ns as f64 / 1e9,
        coverage.longest_trade_gap_ns as f64 / 1e9,
        coverage.price_grid,
        coverage.off_tick_prices,
    );
}

fn oos(error: &OutOfSampleError) -> String {
    let intensity = match (error.intensity, error.intensity_out_of_support) {
        (Some(score), _) => score.round_dp(6).to_string(),
        (None, Some(_)) => "out of support".to_string(),
        (None, None) => "-".to_string(),
    };
    format!(
        "vol err {}, intensity {intensity}",
        decimal(error.volatility)
    )
}

fn fold_line(fold: &FoldReport) -> String {
    let fitted = match &fold.record.outcome {
        CalibrationOutcome::Calibrated(p) => format!(
            "vol {}, A {}, k {}",
            p.volatility.round_dp(2),
            p.intensity_scale_a.round_dp(6),
            p.order_intensity_k.round_dp(6)
        ),
        CalibrationOutcome::Insufficient(failure) => format!("insufficient: {failure:?}"),
        other => format!("{other:?}"),
    };
    format!(
        "  test {}: {fitted} | calibrated: {} | constant: {} | |dk| {}, |dvol| {}",
        &utc(fold.split.test.start_ms)[..10],
        fold.calibrated.as_ref().map_or("-".to_string(), oos),
        oos(&fold.constant),
        decimal(fold.k_change),
        decimal(fold.volatility_change.map(|v| v.round_dp(2))),
    )
}

fn section_87(report: &mut Report, collect: &mut Collect, days: &[TimeWindow]) {
    say!(
        report,
        "\n== #87 walk-forward calibration: out-of-sample error and stability per day against \
         constant parameters (a), and sensitivity to the window (b) =="
    );
    say!(
        report,
        "API: strategy::walk_forward::{{CalibrationData::new, calibrate_window, \
         chronological_splits, walk_forward}}. Inputs: {} mids (last per second), {} order \
         lives of every order on the book (1 in {}), {} fills among them. Order \
         lives are not a quoter's.",
        collect.prices.len(),
        collect.exposures.len(),
        collect.strides.exposure,
        collect.exposures.iter().map(|e| e.fills).sum::<u64>()
    );
    let protocol = collect.protocol;
    let constant = match collect.train_record().clone() {
        Ok(record) => {
            say!(
                report,
                "Training part: {} mids, {} fills, {} s exposed: {}",
                record.price_samples,
                record.fills,
                record.exposure_ms / 1_000,
                match &record.outcome {
                    CalibrationOutcome::Calibrated(p) => format!(
                        "volatility {} per year^0.5, A {} per s, k {} per price unit",
                        p.volatility.round_dp(4),
                        p.intensity_scale_a.round_dp(6),
                        p.order_intensity_k.round_dp(6)
                    ),
                    other => format!("{other:?}"),
                }
            );
            record.parameters().map(|p| ConstantParameters {
                volatility: p.volatility,
                intensity_scale_a: p.intensity_scale_a,
                order_intensity_k: p.order_intensity_k,
            })
        }
        Err(error) => {
            say!(report, "Training part: calibrate_window failed: {error}");
            None
        }
    };
    if days.len() < 2 {
        say!(
            report,
            "(a), (b): fewer than two days: per-day sections skipped ({} whole UTC days in the \
             stretch).",
            days.len()
        );
        return;
    }
    let Some(constant) = constant else {
        say!(
            report,
            "(a), (b): not evaluable: no constant parameters from the training part."
        );
        return;
    };
    let data = match CalibrationData::new(&collect.prices, &collect.exposures) {
        Ok(data) => data,
        Err(error) => {
            say!(report, "(a), (b): CalibrationData::new failed: {error}");
            return;
        }
    };
    let first_test = protocol.train_end_ms.div_ceil(DAY_MS) * DAY_MS;
    for train_days in WF_TRAIN_DAYS {
        let train_ms = train_days * DAY_MS;
        say!(
            report,
            "Training window {train_days} d, tests from {} (after the training part):",
            &utc(first_test)[..10]
        );
        let Some(start) = first_test
            .checked_sub(train_ms)
            .filter(|s| *s >= protocol.start_ms)
        else {
            say!(
                report,
                "  not run: the stretch has less than {train_days} d before the first test day."
            );
            continue;
        };
        let spec = WindowSpec {
            train_ms,
            test_ms: DAY_MS,
            step_ms: DAY_MS,
        };
        let result = chronological_splits(start, protocol.end_ms, &spec).and_then(|splits| {
            walk_forward(&data, &splits, &wf_config(), &collect.dataset, &constant)
        });
        let walk = match result {
            Ok(walk) => walk,
            Err(error) => {
                say!(report, "  error: {error}");
                continue;
            }
        };
        for fold in &walk.folds {
            say!(report, "{}", fold_line(fold));
        }
        let both: Vec<(f64, f64)> = walk
            .folds
            .iter()
            .filter_map(|f| {
                let calibrated = f.calibrated.and_then(|e| e.intensity)?;
                let fixed = f.constant.intensity?;
                Some((
                    calibrated.to_string().parse().ok()?,
                    fixed.to_string().parse().ok()?,
                ))
            })
            .collect();
        let better = both.iter().filter(|(c, f)| c < f).count();
        let mean = |pick: fn(&(f64, f64)) -> f64| {
            both.iter().map(pick).sum::<f64>() / both.len().max(1) as f64
        };
        say!(
            report,
            "  {} folds, {} scored, {} out of support; intensity score (lower is better) of the \
             rolling calibration below the constant's in {better} of {} folds scored by both; \
             means {:.6} against {:.6}.",
            walk.folds.len(),
            walk.scored_folds(),
            walk.out_of_support_folds(),
            both.len(),
            mean(|p| p.0),
            mean(|p| p.1),
        );
    }
}

fn estimator(error: &EstimatorError) -> String {
    format!(
        "rmse {} mae {} markout {}",
        error.rmse.round_dp(4),
        error.mean_absolute.round_dp(4),
        decimal(error.directional_markout.map(|m| m.round_dp(4)))
    )
}

fn section_88(report: &mut Report, collect: &Collect) {
    say!(
        report,
        "\n== #88 micro-price: out-of-sample error and markouts of mid, weighted mid and \
         micro-price at fixed horizons (a), and tables by spread and imbalance (b) =="
    );
    say!(
        report,
        "API: analytics::micro_price::{{calibrate_micro_price, compare_estimators, \
         HorizonComparison::by_state}}. Calibrated on {} training tops, compared on {} holdout \
         tops, each the last of its second. Times are inferred per block, so a horizon shorter \
         than a block is not resolved, and none here is shorter than the sampling.",
        collect.train_tops.len(),
        collect.holdout_tops.len()
    );
    let config = MicroPriceConfig::new(collect.tick, MICRO_BUCKETS, MICRO_SPREADS)
        .with_min_state_observations(MICRO_MIN_STATE)
        .with_max_quote_gap_ms(MICRO_MAX_GAP_MS);
    let record = match calibrate_micro_price(
        &collect.train_tops,
        &collect.protocol.train(),
        &config,
        &collect.dataset,
    ) {
        Ok(record) => record,
        Err(error) => {
            say!(report, "calibrate_micro_price failed: {error}");
            return;
        }
    };
    say!(
        report,
        "Fit: {} pairs, {} left out for the spread, {} for the gap, {} invalid quotes, largest \
         mid move {} ticks.",
        record.pairs,
        record.skipped_spread_pairs,
        record.skipped_gap_pairs,
        record.invalid_quotes,
        record.largest_mid_move_ticks
    );
    let Some(model) = record.model() else {
        say!(report, "No model: {:?}. Nothing compared.", record.fit);
        return;
    };
    let count = |status: StateStatus| model.states().iter().filter(|s| s.status == status).count();
    say!(
        report,
        "States: {} calibrated, {} rare, {} with no mid move, {} outside the quotes; calibrated \
         share {}.",
        count(StateStatus::Calibrated),
        count(StateStatus::Rare),
        count(StateStatus::NoPriceMove),
        count(StateStatus::OutsideQuotes),
        model.calibrated_share().round_dp(4)
    );
    let comparisons = match compare_estimators(
        &record,
        &collect.holdout_tops,
        &collect.protocol.holdout(),
        &HORIZONS_MS,
    ) {
        Ok(comparisons) => comparisons,
        Err(error) => {
            say!(report, "compare_estimators failed: {error}");
            return;
        }
    };
    for comparison in &comparisons {
        say!(
            report,
            "(a) horizon {} ms: {} samples, {} on a fallback, {} unscored, {} invalid quotes",
            comparison.horizon_ms,
            comparison.samples,
            comparison.fallback_samples,
            comparison.unscored_samples,
            comparison.invalid_quotes
        );
        if let Some(set) = &comparison.overall {
            say!(report, "    mid          {}", estimator(&set.mid));
            say!(report, "    weighted mid {}", estimator(&set.weighted_mid));
            say!(report, "    micro-price  {}", estimator(&set.micro_price));
        }
        say!(
            report,
            "(b) horizon {} ms by state (imbalance bucket, spread ticks, samples: rmse of mid, \
             weighted mid, micro-price):",
            comparison.horizon_ms
        );
        for state in &comparison.by_state {
            say!(
                report,
                "    ({}, {}, {}): {} {} {}",
                state.imbalance_bucket,
                state.spread_ticks,
                state.samples,
                state.errors.mid.rmse.round_dp(4),
                state.errors.weighted_mid.rmse.round_dp(4),
                state.errors.micro_price.rmse.round_dp(4)
            );
        }
    }
}

fn section_89(report: &mut Report, collect: &Collect) {
    say!(
        report,
        "\n== #89 quoting on the holdout: markout, inventory and attributed PnL, with \
         sensitivity to fees and latency =="
    );
    say!(
        report,
        "API: strategy::signal_quote::simulate, Avellaneda-Stoikov quotes with no signal \
         feature. Fills follow the simulator's own rule: a trade reaching a resting quote fills \
         it at its price, with no queue priority of our own, so their level is optimistic. Fees \
         are the declared grid, not the venue's; funding is not applied. The part of the \
         holdout in each input file is simulated alone, from a flat inventory and with its own \
         terminal time, and the totals are summed; the markout is pooled by each file's filled \
         volume, so it is approximate."
    );
    if let Some(note) = &collect.sim_note {
        say!(report, "{note}");
        return;
    }
    for cell in &collect.sim {
        if let Some(error) = &cell.error {
            say!(
                report,
                "fee {} latency {} ms: error: {error}",
                cell.fee,
                cell.latency_ms
            );
            continue;
        }
        let markout = (cell.markout_volume > Decimal::ZERO)
            .then(|| (cell.markout_sum / cell.markout_volume).round_dp(4));
        say!(
            report,
            "fee {:>8} latency {:>4} ms: files {}, {} decisions, {} sides crossing, {} fills, volume \
             {}, spread capture {}, inventory {}, fees {}, total {}, max |inventory| {}, markout {}",
            cell.fee,
            cell.latency_ms,
            cell.files,
            cell.decisions,
            cell.crossing_sides,
            cell.fills,
            cell.volume.round_dp(4),
            cell.spread_capture.round_dp(4),
            cell.inventory_pnl.round_dp(4),
            cell.fees.round_dp(4),
            cell.total_pnl.round_dp(4),
            cell.max_abs_inventory.round_dp(4),
            decimal(markout)
        );
    }
}

fn section_90(report: &mut Report, collect: &Collect, days: &[TimeWindow]) {
    say!(
        report,
        "\n== #90 queue-reactive model: intensities per state, sizes and reference price \
         changes (a), estimation per day (d) =="
    );
    say!(
        report,
        "API: backtest::queue_reactive::{{estimate_event_sizes, estimate_parameters}} over \
         hyperliquid::queue_observations, and the mid moves of the replay. On a block clock most \
         waits are zero, and those are left out, so no rate here is an event rate. The mid moves \
         are moves only, so θ is 1 by construction and is not estimated. Each file's events are \
         converted alone, so a queue's first wait in a file is lost."
    );
    let Some(sizes) = &collect.sizes else {
        say!(
            report,
            "No event sizes: {}",
            collect.queue_error.as_deref().unwrap_or("-")
        );
        return;
    };
    say!(
        report,
        "Event sizes on {}: {:?}, from {:?} events, {} excluded",
        window_text(&sizes.queue.training),
        sizes.queue.sizes,
        sizes.queue.events,
        sizes.queue.excluded_events
    );
    let Ok(event_sizes) = &sizes.queue.sizes else {
        say!(
            report,
            "(a), (d): not evaluable: no event size for every limit."
        );
        return;
    };
    if let Some(error) = &collect.queue_error {
        say!(report, "(a), (d): stopped: {error}");
        return;
    }
    let [cold, unknown, no_depth, zero] = collect.queue_left_out;
    say!(
        report,
        "Observations kept {} (1 in {} of those with a positive wait); left out: {zero} \
         zero waits, {cold} cold, {unknown} of unknown cause, {no_depth} with no depth. Mid moves \
         {}.",
        collect.queue.len(),
        collect.strides.queue,
        collect.references.len()
    );
    let config = QrEstimationConfig::new(
        collect.tick,
        LIMITS,
        QR_MAX_QUEUE,
        QR_MIN_EVENTS,
        QR_MIN_REFERENCE_MOVES,
        UnsupportedPolicy::NearestSupported,
    );
    let estimate = |window: &TimeWindow| {
        estimate_parameters(
            &collect.queue,
            &collect.references,
            event_sizes,
            window,
            &config,
            &collect.dataset,
        )
    };
    match estimate(&collect.protocol.queue_train()) {
        Ok(record) => {
            say!(
                report,
                "(a) training {}: {} events used, {} excluded",
                window_text(&record.training),
                record.events,
                record.excluded_events
            );
            match &record.fit {
                QrFit::Calibrated(params) => {
                    say!(
                        report,
                        "    θ {:.4}, θ reinit {:.4}; rates per second (limit, cancel, market) at \
                         supported sizes:",
                        params.theta(),
                        params.theta_reinit()
                    );
                    for depth in 0..params.limits() {
                        let rows: Vec<String> = params
                            .rates(depth)
                            .unwrap_or_default()
                            .iter()
                            .enumerate()
                            .filter(|(n, _)| params.is_supported(depth, *n))
                            .map(|(n, (l, c, m))| format!("{n}: {l:.3} {c:.3} {m:.3}"))
                            .collect();
                        say!(report, "    limit {}: {}", depth + 1, rows.join("; "));
                    }
                }
                other => say!(report, "    {other:?}"),
            }
        }
        Err(error) => say!(report, "(a) estimate_parameters failed: {error}"),
    }
    if days.len() < 2 {
        say!(
            report,
            "(d): fewer than two days: per-day sections skipped ({} whole UTC days in the \
             stretch).",
            days.len()
        );
        return;
    }
    say!(
        report,
        "(d) per UTC day, first limit (size: limit cancel market per second):"
    );
    for day in days {
        let line = match estimate(day) {
            Ok(record) => match &record.fit {
                QrFit::Calibrated(params) => {
                    let rates = params.rates(0).unwrap_or_default();
                    let cells: Vec<String> = QR_DAY_SIZES
                        .iter()
                        .map(|n| match rates.get(*n) {
                            Some((l, c, m)) if params.is_supported(0, *n) => {
                                format!("{n}: {l:.3} {c:.3} {m:.3}")
                            }
                            _ => format!("{n}: unsupported"),
                        })
                        .collect();
                    format!("{} events; {}", record.events, cells.join("; "))
                }
                other => format!("{} events; {other:?}", record.events),
            },
            Err(error) => format!("error: {error}"),
        };
        say!(report, "    {}: {line}", &utc(day.start_ms)[..10]);
    }
}

fn flow_line(name: &str, params: &HawkesParameters, holdout: &[HawkesSegment]) -> String {
    match score_flow(holdout, params) {
        Ok(fit) => format!(
            "{name}: log likelihood per event {:.4}, AIC {:.1}, {} parameters, {} events, {} at \
             a floored intensity",
            fit.log_likelihood_per_event,
            fit.aic,
            fit.parameter_count,
            fit.events,
            fit.floored_events
        ),
        Err(error) => format!("{name}: score_flow failed: {error}"),
    }
}

fn section_91(report: &mut Report, collect: &Collect) {
    say!(
        report,
        "\n== #91 (a) Poisson, queue-reactive and Hawkes flow out of sample, with sensitivity to \
         the decay =="
    );
    say!(
        report,
        "API: backtest::hawkes::{{fit_poisson, fit_hawkes, select_decay, score_flow}} over \
         hyperliquid::hawkes_segments. Events of one block tie in time, which biases the \
         excitation and voids a residual Kolmogorov-Smirnov test, so none is reported. \
         Zero-length periods are dropped, and the open period at the end of each file too."
    );
    if let Some(error) = &collect.hawkes_error {
        say!(report, "Not evaluable: {error}");
        return;
    }
    if let Some(sizes) = &collect.sizes {
        say!(
            report,
            "Event size of the first limit on {}: {:?}, from {:?} events",
            window_text(&sizes.hawkes.training),
            sizes.hawkes.sizes,
            sizes.hawkes.events
        );
        if sizes.hawkes.sizes.is_err() {
            say!(
                report,
                "Not evaluable: no event size for the first limit, so no segment is formed."
            );
            return;
        }
    }
    let [cold, unknown, zero, inconsistent, cut, over_cap, budget] = collect.hawkes_left_out;
    let parts = &collect.hawkes;
    let events = |s: &[HawkesSegment]| s.iter().map(|x| x.events.len()).sum::<usize>();
    let seconds = |s: &[HawkesSegment]| s.iter().fold(0.0, |sum, x| sum + x.duration_seconds);
    say!(
        report,
        "Segments: training {} ({} events, {:.0} s), validation {} ({} events, {:.0} s), \
         holdout {} ({} events, {:.0} s); {} events on inherited times. Left out: {zero} of no \
         length, {inconsistent} departing from an empty queue, {cut} cut by a reset, {over_cap} \
         over the cap, {budget} over the budget; {cold} cold and {unknown} unknown events.",
        parts.train.len(),
        events(&parts.train),
        seconds(&parts.train),
        parts.validation.len(),
        events(&parts.validation),
        seconds(&parts.validation),
        parts.holdout.len(),
        events(&parts.holdout),
        seconds(&parts.holdout),
        collect.hawkes_inherited,
    );
    if parts.train.is_empty() || parts.validation.is_empty() || parts.holdout.is_empty() {
        say!(report, "Not evaluable: a part of the split has no segment.");
        return;
    }
    match fit_poisson(&parts.train, HAWKES_MAX_QUEUE) {
        Ok(params) => {
            let (l, c, m) = params.rates().get(1).copied().unwrap_or_default();
            say!(
                report,
                "Poisson: rates per second {l:.4} {c:.4} {m:.4} (limit, cancel, market)"
            );
            say!(
                report,
                "  holdout {}",
                flow_line("Poisson", &params, &parts.holdout)
            );
        }
        Err(error) => say!(report, "Poisson: fit_poisson failed: {error}"),
    }
    match fit_hawkes(&parts.train, &hawkes_config(Vec::new())) {
        Ok(record) => match record.parameters() {
            Some(params) => {
                say!(
                    report,
                    "Queue-reactive: {} sizes borrowed a baseline",
                    record.borrowed_sizes
                );
                say!(
                    report,
                    "  holdout {}",
                    flow_line("queue-reactive", params, &parts.holdout)
                );
            }
            None => say!(report, "Queue-reactive: no fit: {:?}", record.fit),
        },
        Err(error) => say!(report, "Queue-reactive: fit_hawkes failed: {error}"),
    }
    let config = hawkes_config(Vec::new());
    let (scores, best) = match choose_decay(parts, &config) {
        Ok(choice) => choice,
        Err(error) => {
            say!(report, "Hawkes: select_decay failed: {error}");
            return;
        }
    };
    say!(
        report,
        "Hawkes decay sensitivity (log likelihood per event, training then validation):"
    );
    for score in &scores {
        say!(
            report,
            "  decay {}: {:.4} {}",
            score.decay,
            score.training,
            score.validation.map_or(
                "no fit (fit_hawkes gave no parameters)".to_string(),
                |v| format!("{v:.4}")
            )
        );
    }
    let Some(decay) = best.and_then(|i| scores.get(i)).map(|s| s.decay) else {
        say!(
            report,
            "Hawkes: no candidate fitted, so none is scored on the holdout."
        );
        return;
    };
    match fit_hawkes(&parts.train, &hawkes_config(vec![decay])) {
        Ok(record) => match record.parameters() {
            Some(params) => {
                say!(
                    report,
                    "Hawkes, decay {decay} chosen on validation: branching ratio {:.4}",
                    params.branching_ratio()
                );
                say!(
                    report,
                    "  holdout {}",
                    flow_line("Hawkes", params, &parts.holdout)
                );
            }
            None => say!(report, "Hawkes: no fit: {:?}", record.fit),
        },
        Err(error) => say!(report, "Hawkes: fit_hawkes failed: {error}"),
    }
}

fn section_rest(report: &mut Report) {
    say!(report, "\n== Not applicable ==");
    say!(
        report,
        "#77 replay and discrepancy classification: l3 needs a snapshot and sequence numbers, \
         which the capture lacks."
    );
    say!(
        report,
        "#180 sessions: a BTC perpetual never closes; its silences are recording gaps, handled \
         as resets."
    );
    say!(
        report,
        "#91 (b) Wu et al. as a replication: a different instrument and different data."
    );
    say!(report, "\n== Not done ==");
    say!(
        report,
        "#90 (b) distribution comparison and (c) out-of-sample discrepancies with \
         compare_trajectories: it needs a data Trajectory of BookSample, which the replay does \
         not expose."
    );
}

fn section_kept(report: &mut Report, collect: &Collect) {
    let segments: usize = [
        &collect.hawkes.train,
        &collect.hawkes.validation,
        &collect.hawkes.holdout,
    ]
    .iter()
    .flat_map(|part| part.iter())
    .map(|s| s.events.len() * 16 + size_of::<HawkesSegment>())
    .sum();
    let rows = [
        (
            "tops",
            collect.train_tops.len() + collect.holdout_tops.len(),
            size_of::<TopOfBook>(),
        ),
        ("mids", collect.prices.len(), size_of::<PriceSample>()),
        (
            "order lives",
            collect.exposures.len(),
            size_of::<QuoteExposure>(),
        ),
        (
            "mid moves",
            collect.references.len(),
            size_of::<ReferenceObservation>(),
        ),
        (
            "queue observations",
            collect.queue.len(),
            size_of::<QueueObservation>(),
        ),
    ];
    say!(report, "\n== Kept for the evaluations ==");
    for (name, count, size) in rows {
        say!(
            report,
            "{name}: {count} of {size} bytes, {:.1} MB",
            mb(count * size)
        );
    }
    say!(report, "Hawkes segments: {:.1} MB", mb(segments));
}

#[cfg(test)]
mod tests {
    use super::*;
    use market_maker_rs::backtest::hawkes::HawkesEvent;

    const START: u64 = 1_764_583_200_000;

    #[test]
    fn test_split_is_chronological_disjoint_and_covers_the_stretch() {
        let end = START + 2 * DAY_MS + 12_345;
        let end = whole_seconds(end);
        let p = Protocol::new(START, end).unwrap();
        assert!(p.start_ms < p.sizes_end_ms);
        assert!(p.sizes_end_ms < p.train_end_ms);
        assert!(p.train_end_ms < p.validation_end_ms);
        assert!(p.validation_end_ms < p.end_ms);
        assert_eq!(p.sizes_end_ms - p.start_ms, SIZE_WINDOW_MAX_MS);
        for boundary in [p.sizes_end_ms, p.train_end_ms, p.validation_end_ms] {
            assert_eq!(boundary % SECOND_MS, 0);
        }
        assert_eq!(p.train().end_ms, p.validation().start_ms);
        assert_eq!(p.validation().end_ms, p.holdout().start_ms);
        assert_eq!(p.queue_train().start_ms, p.sizes().end_ms);
        // A sweep of synthetic times: each falls in one part, in order, never going back.
        let mut last = None;
        let mut seen = Vec::new();
        for ms in (START - 5_000..end + 5_000).step_by(997) {
            let part = p.part(ms);
            if ms < START || ms >= end {
                assert_eq!(part, None);
                continue;
            }
            let rank = match part.unwrap() {
                Part::Train => 0,
                Part::Validation => 1,
                Part::Holdout => 2,
            };
            assert!(last.is_none_or(|l| l <= rank));
            last = Some(rank);
            if seen.last() != Some(&rank) {
                seen.push(rank);
            }
        }
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(p.part(p.train_end_ms - 1), Some(Part::Train));
        assert_eq!(p.part(p.train_end_ms), Some(Part::Validation));
        assert_eq!(p.part(p.validation_end_ms), Some(Part::Holdout));
        assert_eq!(p.part(p.end_ms), None);
        // From 10:00 UTC, two days and a bit hold one whole UTC day; from midnight, two.
        assert_eq!(p.days().len(), 1);
        let midnight = START / DAY_MS * DAY_MS;
        let days = Protocol::new(midnight, midnight + 2 * DAY_MS)
            .unwrap()
            .days();
        assert_eq!(days.len(), 2);
        assert!(days.windows(2).all(|d| d[0].end_ms == d[1].start_ms));
        assert!(Protocol::new(START, START + 3_000).is_err());
    }

    fn segment(kinds: &[(f64, QrEventKind)], duration: f64) -> HawkesSegment {
        HawkesSegment {
            initial_queue: 3,
            duration_seconds: duration,
            events: kinds
                .iter()
                .map(|(t, kind)| HawkesEvent {
                    time_seconds: *t,
                    kind: *kind,
                })
                .collect(),
        }
    }

    fn flow(offset: f64) -> Vec<HawkesSegment> {
        use QrEventKind::{Cancel, Limit, Market};
        (0..60)
            .map(|i| {
                let t = offset + f64::from(i % 7) * 0.01;
                segment(
                    &[
                        (t, Limit),
                        (t + 0.1, Cancel),
                        (t + 0.15, Limit),
                        (t + 0.4, Market),
                        (t + 0.45, Limit),
                        (t + 0.9, Cancel),
                    ],
                    2.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_the_decay_choice_never_reads_the_holdout() {
        let config = hawkes_config(Vec::new()).with_min_events(5);
        let mut parts = HawkesParts {
            train: flow(0.0),
            validation: flow(0.05),
            holdout: flow(0.1),
            events: [0; 3],
        };
        let first = choose_decay(&parts, &config).unwrap();
        // A holdout that any read would refuse: a segment of negative length.
        parts.holdout = vec![segment(&[], -1.0)];
        let params = fit_poisson(&parts.train, HAWKES_MAX_QUEUE).unwrap();
        assert!(score_flow(&parts.holdout, &params).is_err());
        let second = choose_decay(&parts, &config).unwrap();
        // Debug text, as a failed candidate's training score is NaN.
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        assert!(first.1.is_some());
    }

    #[test]
    fn test_thinning_keeps_the_last_of_a_second_and_one_in_a_stride() {
        let mut kept: Vec<u64> = Vec::new();
        for ms in [10, 999, 1_000, 1_500, 4_000] {
            keep_last(&mut kept, ms, |t| *t);
        }
        assert_eq!(kept, vec![999, 1_500, 4_000]);
        let mut seen = 0;
        let picks: Vec<bool> = (0..5).map(|_| keep_one_in(&mut seen, 2)).collect();
        assert_eq!(picks, vec![true, false, true, false, true]);
        let path = segment(&[(0.1, QrEventKind::Limit)], 1.0);
        assert!(within_cap(&path, 4));
        assert!(!within_cap(&path, 3));
    }
}
