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
//! - **One seed.** The thinning below is by count, so it draws nothing. The only random numbers
//!   are the queue-reactive model's runs of #90 (b), (c) and (d), all from [`QR_SEED`].
//! - **The book's windows for #90 (b), (c) and (d).** The data's book, sampled every
//!   [`BOOK_SAMPLE_MS`], and its queue events are compared with the model's on windows of
//!   [`BOOK_WINDOW_MS`] from their start: the holdout's first hours against the training fit,
//!   out of sample, and each whole UTC day's first hours against that day's own fit, in sample, as
//!   a per-day diagnostic. Each model is run for the window's length, and so is its constant
//!   intensity baseline.
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
//! - Mids for the walk-forward, the mid of the last top of each second: at most 3,600 an hour of
//!   24 bytes, at most about 65 MB for a month. Its grid is of one second, so the grid prices are
//!   the same. They come from the tops and not from the replay's `prices`, which hold the mid
//!   only when it changed: a touch that stays put while its sizes change is a live book whose mid
//!   did not move, and must be a sample, not a silence, for the walk-forward's gap check.
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
//! - The book's windows, whole: about 2 million queue events an hour of 24 bytes and a sample a
//!   second, so about 100 MB per window of [`BOOK_WINDOW_MS`], two at most at a time (a day's
//!   until its end, and the holdout's), and as much again for a model run. A window is compared
//!   once what its fit needs has been read, and then dropped.
//!
//! So a month needs about 0.5 GB kept beside the hour being read, and about 0.4 GB more at
//! most for the book's windows. The report ends with what was
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
    BookEvent, Coverage, EventCause, RepeatFilter, Replay, ReplayConfig, ReplayOutput, Warmth,
    book_samples, decode_order_statuses_after, hawkes_segments, queue_observations,
    read_book_diffs_after, read_trades_after, sized_events,
};
use market_maker_rs::backtest::queue_reactive::{
    BookSample, DiscrepancyReport, EventSizeRecord, QrEstimationConfig, QrEventKind, QrFailure,
    QrFit, QrParameters, QrRecord, QrSide, QrSimulator, QueueObservation, ReferenceObservation,
    ResetObservation, SizedEvent, Trajectory, UnsupportedPolicy, ZeroWaitPolicy, collapse_blocks,
    compare_trajectories, estimate_event_sizes, estimate_parameters,
    estimate_parameters_with_resets,
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
use rust_decimal::prelude::{FromStr, ToPrimitive};

#[path = "support/qr_290.rs"]
mod qr_290;

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
/// Longest silence of the mid, in grid buckets: the reset gap. A mid is sampled at each change
/// of the touch, sizes included, so a silence is a touch that did not change at all.
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
/// Length of each window the book is compared on, from its start: the holdout's first hours for
/// (b) and (c), and each whole UTC day's first hours for (d). A window's events are kept whole,
/// about two million an hour, so it is bounded.
const BOOK_WINDOW_MS: u64 = 2 * HOUR_MS;
/// The book's sampling grid, in the data and in the model.
const BOOK_SAMPLE_MS: u64 = SECOND_MS;
/// Interval the flow law counts events in.
const FLOW_INTERVAL_MS: u64 = SECOND_MS;
/// Seed of every model run, the only randomness here.
const QR_SEED: u64 = 263;

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

/// What a comparison window's fit is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowKind {
    /// In sample, for #290's choice: the training fit, against the first hours of the training
    /// part after the size window.
    Train,
    /// Out of sample: the training fit, against the holdout's first hours.
    Holdout,
    /// In sample: that day's own fit, against its first hours.
    Day(TimeWindow),
}

/// A window the book is compared on, for #90 (b), (c) and (d).
#[derive(Debug)]
struct BookWindow {
    kind: WindowKind,
    window: TimeWindow,
    /// When everything its fit needs has been read.
    due_ms: u64,
    samples: Vec<BookSample>,
    events: Vec<QueueObservation>,
    samples_without_sizes: u64,
    error: Option<String>,
    /// Samples and events compared, once done.
    data: (u64, u64),
    result: Option<Vec<String>>,
    /// #290's phase 3, once done: the data's mean touch queue and each estimator's lines, or
    /// why it was not run.
    result_290: Option<Result<(f64, Vec<Line290>), String>>,
}

impl BookWindow {
    fn new(kind: WindowKind, start_ms: u64, limit_ms: u64, due_ms: u64) -> Self {
        Self {
            kind,
            window: TimeWindow {
                start_ms,
                end_ms: (start_ms + BOOK_WINDOW_MS).min(limit_ms),
            },
            due_ms,
            samples: Vec::new(),
            events: Vec::new(),
            samples_without_sizes: 0,
            error: None,
            data: (0, 0),
            result: None,
            result_290: None,
        }
    }

    /// The training part's first hours, the holdout's, then each whole UTC day's.
    fn all(protocol: &Protocol) -> Vec<Self> {
        let train = protocol.queue_train();
        let holdout = protocol.holdout();
        let mut windows = vec![
            Self::new(
                WindowKind::Train,
                train.start_ms,
                train.end_ms,
                train.end_ms,
            ),
            Self::new(
                WindowKind::Holdout,
                holdout.start_ms,
                holdout.end_ms,
                (holdout.start_ms + BOOK_WINDOW_MS).min(holdout.end_ms),
            ),
        ];
        for day in protocol.days() {
            windows.push(Self::new(
                WindowKind::Day(day),
                day.start_ms,
                day.end_ms,
                day.end_ms,
            ));
        }
        windows
    }

    fn holds(&self, ms: u64) -> bool {
        self.window.start_ms <= ms && ms < self.window.end_ms
    }
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

/// The data of one window against `params` and its constant intensity baseline, each run for
/// the window's length from seed [`QR_SEED`] and sampled on the same grid.
fn compare_window(
    samples: Vec<BookSample>,
    events: Vec<QueueObservation>,
    params: &QrParameters,
    window: TimeWindow,
) -> Vec<String> {
    let mut data = Trajectory::default();
    data.samples = samples;
    data.events = events;
    let seconds = (window.end_ms - window.start_ms) as f64 / 1e3;
    let mut lines = Vec::new();
    let baseline = params.constant_baseline();
    for (name, model) in [("fit", Ok(params.clone())), ("constant", baseline)] {
        let line = model
            .and_then(|model| {
                let run = QrSimulator::new(model.clone(), QR_SEED, 0)?
                    .run(seconds, BOOK_SAMPLE_MS as f64 / 1e3)?;
                compare_trajectories(&data, &run, &model, FLOW_INTERVAL_MS)
            })
            .map_or_else(
                |error| format!("{name}: not evaluable: {error}"),
                |report| format!("{name}: {}", discrepancy_text(&report)),
            );
        lines.push(line);
    }
    lines
}

fn discrepancy_text(r: &DiscrepancyReport) -> String {
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "queue size TV by limit [{}]; spread TV {:.3}; mean wait relative error by limit [{}]; \
         wait KS by limit [{}]; flow relative error L/C/M [{}]; flow law TV L/C/M [{}]; state \
         coverage {:.4}; data {} samples, {} events ({} without a positive wait); model {} \
         samples, {} events ({} without), stopped {:?}",
        list(&r.queue_size),
        r.spread,
        list(&r.waiting_time),
        list(&r.waiting_law),
        list(&r.flow),
        list(&r.flow_law),
        r.coverage,
        r.data_samples,
        r.data_events,
        r.data_invalid_waiting,
        r.model_samples,
        r.model_events,
        r.model_invalid_waiting,
        r.model_stopped
    )
}

/// Keeps the last item of each bucket of the clock.
fn keep_last<T>(kept: &mut Vec<T>, item: T, time: impl Fn(&T) -> u64) {
    let bucket = time(&item) / SAMPLE_MS;
    match kept.last_mut() {
        Some(last) if time(last) / SAMPLE_MS == bucket => *last = item,
        _ => kept.push(item),
    }
}

/// The mid of a top as a price sample, as the replay computes it. `None` on an overflow, which a
/// price of the capture cannot reach.
fn mid_of(top: &TopOfBook) -> Option<PriceSample> {
    let price = top.bid.checked_add(top.ask)?.checked_div(Decimal::TWO)?;
    Some(PriceSample {
        timestamp_ms: top.timestamp_ms,
        price,
    })
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

/// What a touch queue's life ended with, at a move of the mid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Touch {
    /// No event at the touch since the last move, or one whose effect is unknown.
    Unobserved,
    /// The size left by the last event at the touch since the last move.
    Level(Decimal),
}

/// A book event as #290's phase 1 reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TouchEvent {
    time_ns: u64,
    mid: Option<Decimal>,
    /// The side, for a warm event at the touch.
    side: Option<usize>,
    /// What it left at its level; `None` for an unknown cause.
    level_after: Option<Decimal>,
    /// What its level held before it.
    level_before: Decimal,
}

impl TouchEvent {
    fn of(event: &BookEvent) -> Self {
        let at_touch = event.warmth == Warmth::Warm && event.depth == Some(0);
        Self {
            time_ns: event.time.timestamp_ns,
            mid: event.mid_before,
            side: at_touch.then(|| side_index(event.side)),
            level_after: match event.cause {
                EventCause::Limit => Some(event.level_before + event.size),
                EventCause::Cancel | EventCause::Fill => Some(event.level_before - event.size),
                _ => None,
            },
            level_before: event.level_before,
        }
    }
}

fn side_index(side: QrSide) -> usize {
    usize::from(side != QrSide::Bid)
}

/// #290's phase 1: what the training part shows about the drift of the queue model's fit. One
/// estimator's input is every observation, the other's every observation with each block
/// collapsed per queue; each is kept one in the queue stride, as the observations of #90 are.
/// A block that straddles two hour files is collapsed as two.
#[derive(Debug, Default)]
struct Phase290 {
    /// Per limit and kind (limit, cancel, market): with a positive wait, with a zero wait.
    counts: [[[u64; 2]; 3]; LIMITS as usize],
    /// Observations with a negative or not finite wait.
    invalid_waits: u64,
    all: Vec<QueueObservation>,
    all_seen: u64,
    collapsed: Vec<QueueObservation>,
    collapsed_seen: u64,
    /// Touch-queue lives at a move: emptied, moved while not empty, unobserved.
    lives: [u64; 3],
    touch: [Option<Touch>; 2],
    mid: Option<Option<Decimal>>,
    /// Block length histogram over [`qr_290::BLOCK_GAP_BOUNDS_MS`], with one bin above.
    gaps: [u64; qr_290::BLOCK_GAP_BOUNDS_MS.len() + 1],
    gap_total_ns: u128,
    blocks: u64,
    block_events: u64,
    multi_event_blocks: u64,
    last_time_ns: Option<u64>,
    events_in_block: u64,
    /// Phase 3's choice on the training window, or why there is none.
    choice: Option<Result<&'static str, String>>,
    /// E3's touch resets, from every part, one in the queue stride; out of time order until
    /// sorted for a fit.
    resets: Vec<ResetObservation>,
    resets_seen: u64,
    /// On the training part: resets made, and sides at a move with no emptied touch that
    /// gave none, for want of a seen touch.
    resets_made: u64,
    resets_skipped: u64,
    /// A reset per side waiting for its new touch's size.
    pending: [Option<ResetObservation>; 2],
    /// Time of each side's last touch event in the current life.
    touch_ns: [Option<u64>; 2],
    /// On the training part, per limit and kind: the summed size change in event sizes and
    /// the events.
    size_change: [[(f64, u64); 3]; LIMITS as usize],
    /// Per UTC day, over every part: resets made, sides left out for want of a seen touch, and
    /// mid moves seen.
    resets_by_day: std::collections::BTreeMap<u64, [u64; 3]>,
}

impl Phase290 {
    fn take_counts(&mut self, observations: &[QueueObservation]) {
        for o in observations {
            let kind = match o.kind {
                QrEventKind::Limit => 0,
                QrEventKind::Cancel => 1,
                _ => 2,
            };
            let zero = if o.waiting_seconds > 0.0 {
                0
            } else if o.waiting_seconds == 0.0 {
                1
            } else {
                self.invalid_waits += 1;
                continue;
            };
            if let Some(cell) = self
                .counts
                .get_mut(usize::from(o.depth))
                .and_then(|c| c.get_mut(kind))
                .and_then(|c| c.get_mut(zero))
            {
                *cell += 1;
            }
        }
    }

    /// Keeps the estimators' inputs, from every part, one in `stride`.
    fn take_lists(&mut self, observations: &[QueueObservation], stride: u64) {
        for o in observations {
            if keep_one_in(&mut self.all_seen, stride) {
                self.all.push(*o);
            }
        }
        for o in collapse_blocks(observations) {
            if keep_one_in(&mut self.collapsed_seen, stride) {
                self.collapsed.push(o);
            }
        }
    }

    /// Reads the book's events: on the training part (`counting`) for phase 1's counts, and on
    /// every part for E3's resets, in units of `touch_size` and kept one in `stride`.
    fn take_book<'a>(
        &mut self,
        events: impl Iterator<Item = &'a BookEvent>,
        counting: bool,
        sizes: &[Decimal],
        stride: u64,
    ) {
        for event in events {
            if counting
                && event.warmth == Warmth::Warm
                && let Some(depth) = event.depth
                && let (Some(size), Some(row)) = (
                    sizes.get(usize::from(depth)),
                    self.size_change.get_mut(usize::from(depth)),
                )
            {
                let kind = match event.cause {
                    EventCause::Limit => Some(0),
                    EventCause::Cancel => Some(1),
                    EventCause::Fill => Some(2),
                    _ => None,
                };
                if let Some(cell) = kind.and_then(|k| row.get_mut(k))
                    && let Some(units) = (event.size / *size).to_f64()
                {
                    cell.0 += units;
                    cell.1 += 1;
                }
            }
            let touch_size = sizes.first().copied().unwrap_or(Decimal::ONE);
            self.take_touch(TouchEvent::of(event), counting, touch_size, stride);
        }
    }

    fn keep_reset(&mut self, reset: ResetObservation, stride: u64) {
        if keep_one_in(&mut self.resets_seen, stride) {
            self.resets.push(reset);
        }
    }

    fn take_touch(&mut self, event: TouchEvent, counting: bool, touch_size: Decimal, stride: u64) {
        let units = |level: Decimal| (level / touch_size).round().to_u32().unwrap_or(u32::MAX);
        if counting {
            if self.last_time_ns != Some(event.time_ns) {
                if let Some(last) = self.last_time_ns {
                    let gap = event.time_ns.saturating_sub(last);
                    let ms = gap / NS_PER_MS;
                    let bin = qr_290::BLOCK_GAP_BOUNDS_MS.partition_point(|&bound| bound <= ms);
                    if let Some(cell) = self.gaps.get_mut(bin) {
                        *cell += 1;
                    }
                    self.gap_total_ns += u128::from(gap);
                    self.blocks += 1;
                    self.multi_event_blocks += u64::from(self.events_in_block > 1);
                }
                self.last_time_ns = Some(event.time_ns);
                self.events_in_block = 0;
            }
            self.events_in_block += 1;
            self.block_events += 1;
        }
        if self.mid != Some(event.mid) {
            let ends = self.touch.map(|touch| match touch {
                Some(Touch::Level(level)) if level <= Decimal::ZERO => 0,
                Some(Touch::Level(_)) => 1,
                _ => 2,
            });
            if counting && self.mid.is_some() {
                for end in ends {
                    self.lives[end] += 1;
                }
            }
            // A reset still waiting for its new size ends with none.
            for side in 0..2 {
                if let Some(reset) = self.pending.get_mut(side).and_then(Option::take) {
                    self.keep_reset(reset, stride);
                }
            }
            let both = matches!(self.mid, Some(Some(_))) && event.mid.is_some();
            let day = self
                .resets_by_day
                .entry(event.time_ns / NS_PER_MS / DAY_MS)
                .or_default();
            day[2] += u64::from(self.mid.is_some());
            if both && !ends.contains(&0) {
                for side in 0..2 {
                    match (self.touch.get(side), self.touch_ns.get(side)) {
                        (Some(Some(Touch::Level(level))), Some(Some(at))) => {
                            day[0] += 1;
                            if counting {
                                self.resets_made += 1;
                            }
                            if let Some(slot) = self.pending.get_mut(side) {
                                *slot = Some(ResetObservation {
                                    timestamp_ms: event.time_ns / NS_PER_MS,
                                    side: if side == 0 { QrSide::Bid } else { QrSide::Ask },
                                    queue_before: units(*level),
                                    waiting_seconds: event.time_ns.saturating_sub(*at) as f64 / 1e9,
                                    queue_after: None,
                                });
                            }
                        }
                        _ => {
                            day[1] += 1;
                            if counting {
                                self.resets_skipped += 1;
                            }
                        }
                    }
                }
            }
            self.mid = Some(event.mid);
            self.touch = [None; 2];
            self.touch_ns = [None; 2];
        }
        if let Some(side) = event.side {
            if let Some(mut reset) = self.pending.get_mut(side).and_then(Option::take) {
                reset.queue_after = Some(units(event.level_before));
                self.keep_reset(reset, stride);
            }
            if let Some(touch) = self.touch.get_mut(side) {
                *touch = Some(event.level_after.map_or(Touch::Unobserved, Touch::Level));
            }
            if let Some(at) = self.touch_ns.get_mut(side) {
                *at = Some(event.time_ns);
            }
        }
    }

    /// E3's resets in time order.
    fn sorted_resets(&self) -> Vec<ResetObservation> {
        let mut resets = self.resets.clone();
        resets.sort_by_key(|r| r.timestamp_ms);
        resets
    }

    /// The share of lives that ended by a move while not empty, among those whose end is known.
    fn exit_by_move_share(&self) -> Option<f64> {
        let known = self.lives[0] + self.lives[1];
        (known > 0).then(|| self.lives[1] as f64 / known as f64)
    }
}

/// An estimator's name and its fit, or why there is none.
type Fit290Input = (&'static str, Result<QrParameters, String>);

/// #290's estimators, in the order a tie is broken, then E3's sensitivity row, which is
/// reported and never chosen.
const ESTIMATORS_290: [&str; 5] = ["E0", "E1", "E2", "E3", "E3 inv"];

/// The estimators phase 3 chooses among.
const CHOSEN_AMONG_290: [&str; 4] = ["E0", "E1", "E2", "E3"];

/// One run of #290's phase 3: an estimator's fit or its constant baseline against a window.
#[derive(Debug)]
struct Line290 {
    estimator: &'static str,
    baseline: bool,
    /// The comparison and the model's mean touch queue.
    outcome: Result<(DiscrepancyReport, f64), String>,
    /// Touch resets in the model's run, and the smallest touch queue its samples show.
    run_resets: usize,
    run_min_touch: Option<u32>,
    /// For a model with touch resets: the sizes with a positive reset rate, and the largest.
    reset_sizes: Option<(usize, usize)>,
}

/// The sizes with a positive reset rate, and the largest, if `params` has resets.
fn reset_sizes(params: &QrParameters) -> Option<(usize, usize)> {
    params.reset_rates().map(|rates| {
        let positive: Vec<usize> = (0..rates.len()).filter(|n| rates[*n] > 0.0).collect();
        (positive.len(), positive.last().copied().unwrap_or(0))
    })
}

/// Each fit and its constant baseline against the window's data, from seed
/// [`qr_290::QR_SEED`], with the data's mean touch queue.
fn compare_290(data: &Trajectory, fits: &[Fit290Input], window: TimeWindow) -> (f64, Vec<Line290>) {
    let seconds = (window.end_ms - window.start_ms) as f64 / 1e3;
    type Run = (DiscrepancyReport, f64, usize, Option<u32>);
    let run = |model: &QrParameters| -> Result<Run, String> {
        let run = QrSimulator::new(model.clone(), qr_290::QR_SEED, 0)
            .and_then(|mut s| s.run(seconds, BOOK_SAMPLE_MS as f64 / 1e3))
            .map_err(|e| e.to_string())?;
        let report =
            compare_trajectories(data, &run, model, FLOW_INTERVAL_MS).map_err(|e| e.to_string())?;
        let min_touch = run.samples.iter().map(|s| s.bid[0].min(s.ask[0])).min();
        Ok((
            report,
            mean_touch(&run.samples),
            run.resets.len(),
            min_touch,
        ))
    };
    let mut lines = Vec::new();
    for (estimator, fit) in fits {
        for baseline in [false, true] {
            let model = match fit {
                Err(error) => Err(format!("no fit: {error}")),
                Ok(params) if baseline => params.constant_baseline().map_err(|e| e.to_string()),
                Ok(params) => Ok(params.clone()),
            };
            let reset_sizes = model.as_ref().ok().and_then(reset_sizes);
            let outcome = model.and_then(|model| run(&model));
            let run_resets = outcome.as_ref().map_or(0, |(_, _, n, _)| *n);
            let run_min_touch = outcome.as_ref().ok().and_then(|(_, _, _, m)| *m);
            lines.push(Line290 {
                estimator,
                baseline,
                outcome: outcome.map(|(report, mean, _, _)| (report, mean)),
                run_resets,
                run_min_touch,
                reset_sizes,
            });
        }
    }
    (mean_touch(&data.samples), lines)
}

/// Whether a model's mean touch queue stays under #290's stationary share of the cap.
fn stationary_290(mean_touch_queue: f64) -> bool {
    !qr_290::reaches(mean_touch_queue, qr_290::STATIONARY_SHARE_OF_CAP)
}

/// Phase 3's choice on the training window: among the stationary fits, the lowest queue size
/// total variation at limit 0, the first on a tie; none when no fit is stationary.
fn choose_290(lines: &[Line290]) -> Result<&'static str, String> {
    let mut best: Option<(&'static str, f64)> = None;
    for line in lines
        .iter()
        .filter(|l| !l.baseline && CHOSEN_AMONG_290.contains(&l.estimator))
    {
        let Ok((report, mean)) = &line.outcome else {
            continue;
        };
        let Some(tv) = report.queue_size.first().copied() else {
            continue;
        };
        if stationary_290(*mean) && best.is_none_or(|(_, b)| tv < b) {
            best = Some((line.estimator, tv));
        }
    }
    best.map(|(name, _)| name)
        .ok_or_else(|| "no estimator is stationary on the training window".to_string())
}

/// Whether `estimator`'s fit beats its constant baseline on queue size at every limit.
fn beats_baseline_290(lines: &[Line290], estimator: &str) -> Option<bool> {
    let tv = |baseline: bool| {
        lines
            .iter()
            .find(|l| l.estimator == estimator && l.baseline == baseline)
            .and_then(|l| l.outcome.as_ref().ok())
            .map(|(report, _)| report.queue_size.clone())
    };
    let (fit, constant) = (tv(false)?, tv(true)?);
    (fit.len() == constant.len() && !fit.is_empty())
        .then(|| fit.iter().zip(&constant).all(|(f, c)| f < c))
}

/// One estimator's training fit for #290: the drift at the first limit and how its run behaves.
struct Fit290 {
    name: &'static str,
    events: u64,
    excluded: u64,
    params: Result<QrParameters, String>,
}

impl Fit290 {
    /// The mean touch queue, both sides, over a 2 h run from seed [`qr_290::QR_SEED`], with
    /// how the run ended.
    fn run_mean_touch(&self) -> Result<(f64, Option<String>), String> {
        let params = self.params.as_ref().map_err(Clone::clone)?;
        let run = QrSimulator::new(params.clone(), qr_290::QR_SEED, 0)
            .and_then(|mut s| s.run(qr_290::WINDOW_SECONDS, BOOK_SAMPLE_MS as f64 / 1e3))
            .map_err(|e| e.to_string())?;
        Ok((
            mean_touch(&run.samples),
            run.stopped.map(|s| format!("{s:?}")),
        ))
    }
}

fn mean_touch(samples: &[BookSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total: f64 = samples
        .iter()
        .map(|s| f64::from(s.bid[0]) + f64::from(s.ask[0]))
        .sum();
    total / (2.0 * samples.len() as f64)
}

fn invariant_mean(law: &[f64]) -> f64 {
    law.iter().enumerate().map(|(n, p)| n as f64 * p).sum()
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
    book_windows: Vec<BookWindow>,
    qr_train: Option<Result<QrParameters, String>>,
    phase_290: Phase290,
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
            book_windows: BookWindow::all(&protocol),
            qr_train: None,
            phase_290: Phase290::default(),
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
        for top in &output.tops {
            if self.part(top.epoch, top.time.timestamp_ns).is_some()
                && let Some(mid) = mid_of(&top.value)
            {
                keep_last(&mut self.prices, mid, |p| p.timestamp_ms);
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
        self.take_book(output);
        let holdout: Vec<MarketEvent> = output
            .market_events
            .iter()
            .filter(|e| self.part(e.epoch, e.time.timestamp_ns) == Some(Part::Holdout))
            .map(|e| e.value.clone())
            .collect();
        if !holdout.is_empty() {
            self.simulate_hour(&holdout);
        }
        let read_to_ms = output.events.last().map(|last| {
            if last.epoch > self.epoch {
                u64::MAX
            } else {
                last.time.timestamp_ns / NS_PER_MS
            }
        });
        if let Some(read_to_ms) = read_to_ms {
            self.compare_due(read_to_ms);
        }
        read_to_ms.is_some_and(|ms| ms >= self.protocol.end_ms)
    }

    /// Keeps the book samples inside a comparison window, in the model's units.
    fn take_book(&mut self, output: &ReplayOutput) {
        let event_sizes = match &self.sizes {
            Some(sizes) => sizes.queue.sizes.as_ref().ok(),
            None => None,
        };
        for window in &mut self.book_windows {
            if window.result.is_some() {
                continue;
            }
            let inside: Vec<_> = output
                .book
                .iter()
                .filter(|s| s.epoch == self.epoch && window.holds(s.value.timestamp_ms))
                .cloned()
                .collect();
            if inside.is_empty() {
                continue;
            }
            let Some(event_sizes) = event_sizes else {
                window.samples_without_sizes += inside.len() as u64;
                continue;
            };
            match book_samples(&inside, event_sizes, self.tick) {
                Ok(samples) => window.samples.extend(samples.into_iter().map(|s| s.value)),
                Err(error) => window.error = Some(format!("book_samples: {error}")),
            }
        }
    }

    /// The queue-reactive fit on training, once training has been read.
    fn qr_train(&mut self) -> Result<QrParameters, String> {
        if self.qr_train.is_none() {
            let fit = self.qr_fit(&self.protocol.queue_train());
            self.qr_train = Some(fit);
        }
        self.qr_train
            .clone()
            .unwrap_or_else(|| Err("not fitted".to_string()))
    }

    /// The queue-reactive fit on the observations kept inside `window`.
    fn qr_fit(&self, window: &TimeWindow) -> Result<QrParameters, String> {
        let Some(Ok(event_sizes)) = self.sizes.as_ref().map(|s| &s.queue.sizes) else {
            return Err("no event sizes".to_string());
        };
        let record = estimate_parameters(
            &self.queue,
            &self.references,
            event_sizes,
            window,
            &qr_config(self.tick),
            &self.dataset,
        )
        .map_err(|error| format!("estimate_parameters: {error}"))?;
        match record.fit {
            QrFit::Calibrated(params) => Ok(params),
            other => Err(format!("{other:?}")),
        }
    }

    /// Compares every window whose fit can be made once the data has been read to `read_to_ms`,
    /// and drops its data.
    fn compare_due(&mut self, read_to_ms: u64) {
        for index in 0..self.book_windows.len() {
            let window = &self.book_windows[index];
            if window.result.is_some() || read_to_ms < window.due_ms {
                continue;
            }
            let kind = window.kind;
            let fit = match kind {
                WindowKind::Train | WindowKind::Holdout => self.qr_train(),
                WindowKind::Day(day) => self.qr_fit(&day),
            };
            let fits_290 = self.fits_290(kind);
            let window = &mut self.book_windows[index];
            let samples = std::mem::take(&mut window.samples);
            let events = std::mem::take(&mut window.events);
            window.data = (samples.len() as u64, events.len() as u64);
            let mut data = Trajectory::default();
            data.samples = samples;
            data.events = events;
            window.result_290 = Some(match (&window.error, fits_290) {
                (Some(error), _) => Err(format!("not evaluable: {error}")),
                (None, Err(reason)) => Err(reason),
                (None, Ok(fits)) => Ok(compare_290(&data, &fits, window.window)),
            });
            window.result = Some(match (&window.error, fit) {
                (Some(error), _) => vec![format!("not evaluable: {error}")],
                (None, Err(error)) => vec![format!("not evaluable: no fit: {error}")],
                (None, Ok(params)) => {
                    compare_window(data.samples, data.events, &params, window.window)
                }
            });
            if kind == WindowKind::Train {
                let choice = match &window.result_290 {
                    Some(Ok((_, lines))) => choose_290(lines),
                    Some(Err(reason)) => Err(reason.clone()),
                    None => Err("not compared".to_string()),
                };
                self.phase_290.choice = Some(choice);
            }
        }
    }

    /// #290's training fit of `estimator` on the observations kept inside `window`.
    fn fit_290(
        &self,
        estimator: &'static str,
        window: &TimeWindow,
    ) -> Result<QrParameters, String> {
        let record = self.record_290(estimator, window)?;
        match record.fit {
            QrFit::Calibrated(params) => Ok(params),
            other => Err(format!("{other:?}")),
        }
    }

    /// #290's record of `estimator` on the observations kept inside `window`. E3 is E1 with
    /// the touch resets; "E3 inv", a sensitivity row only, is E3 redrawing from the first
    /// limit's invariant law instead of the observed new sizes.
    fn record_290(&self, estimator: &'static str, window: &TimeWindow) -> Result<QrRecord, String> {
        let Some(Ok(event_sizes)) = self.sizes.as_ref().map(|s| &s.queue.sizes) else {
            return Err("no event sizes".to_string());
        };
        let config = |zero_waits| qr_config(self.tick).with_zero_waits(zero_waits);
        let plain = |events: &[QueueObservation], zero_waits| {
            estimate_parameters(
                events,
                &self.references,
                event_sizes,
                window,
                &config(zero_waits),
                &self.dataset,
            )
        };
        let record = match estimator {
            "E0" => plain(&self.queue, ZeroWaitPolicy::Drop),
            "E1" => plain(&self.phase_290.all, ZeroWaitPolicy::Count),
            "E2" => plain(&self.phase_290.collapsed, ZeroWaitPolicy::Count),
            _ => estimate_parameters_with_resets(
                &self.phase_290.all,
                &self.phase_290.sorted_resets(),
                &self.references,
                event_sizes,
                window,
                &config(ZeroWaitPolicy::Count),
                &self.dataset,
            ),
        }
        .map_err(|error| format!("estimate_parameters: {error}"))?;
        if estimator != "E3 inv" {
            return Ok(record);
        }
        let mut record = record;
        record.fit = match std::mem::replace(
            &mut record.fit,
            QrFit::Insufficient(QrFailure::InvalidEstimate),
        ) {
            QrFit::Calibrated(params) => {
                let rates = params.reset_rates().unwrap_or_default().to_vec();
                let law = params.invariant(0).unwrap_or_default().to_vec();
                params.with_resets(rates, law).map_or(
                    QrFit::Insufficient(QrFailure::InvalidEstimate),
                    QrFit::Calibrated,
                )
            }
            other => other,
        };
        Ok(record)
    }

    /// The fits #290's phase 3 compares on a window of `kind`, or why it does not: the
    /// holdout is compared only once the training window has chosen an estimator.
    fn fits_290(&self, kind: WindowKind) -> Result<Vec<Fit290Input>, String> {
        let window = match kind {
            WindowKind::Train => self.protocol.queue_train(),
            WindowKind::Holdout => match &self.phase_290.choice {
                Some(Ok(_)) => self.protocol.queue_train(),
                Some(Err(reason)) => return Err(format!("not run, stop rule: {reason}")),
                None => return Err("not run: the training window was not compared".to_string()),
            },
            WindowKind::Day(day) => day,
        };
        Ok(ESTIMATORS_290
            .iter()
            .map(|name| (*name, self.fit_290(name, &window)))
            .collect())
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
                    for observation in &conversion.observations {
                        let at = observation.value.timestamp_ms;
                        for window in &mut self.book_windows {
                            if window.result.is_none() && window.holds(at) {
                                window.events.push(observation.value);
                            }
                        }
                    }
                    let values: Vec<QueueObservation> =
                        conversion.observations.iter().map(|o| o.value).collect();
                    let counting = part == Part::Train;
                    if counting {
                        self.phase_290.take_counts(&values);
                    }
                    self.phase_290.take_book(
                        events.iter().filter(|e| e.epoch == self.epoch),
                        counting,
                        event_sizes,
                        self.strides.queue,
                    );
                    self.phase_290.take_lists(&values, self.strides.queue);
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
    let config = ReplayConfig::new(tick, LIMITS, minutes * 60 * 1_000_000_000, RESET_GAP_NS)?
        .with_book_samples(BOOK_SAMPLE_MS)?;
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
    collect.compare_due(u64::MAX);
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
    section_90_book(&mut report, &collect);
    section_290(&mut report, &collect);
    section_290_book(&mut report, &collect);
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
        "Randomness: thinning is by count or by clock; the queue-reactive model runs of #90 (b), \
         (c) and (d) use seed {QR_SEED}."
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
         the rates of sizes {QR_DAY_SIZES:?} of the first limit. (b), (c), (d): the book sampled \
         every {BOOK_SAMPLE_MS} ms on windows of {} min from their start (the holdout's, against \
         the training fit; each whole UTC day's, against its own fit), the model and its \
         constant intensity baseline run for as long on the same grid, events counted per \
         {FLOW_INTERVAL_MS} ms for the flow law.",
        BOOK_WINDOW_MS / 60_000
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
         chronological_splits, walk_forward}}. Inputs: {} mids (of the last top per second), {} order \
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

/// #290 phase 1: on the training part, the counts by type with and without a zero wait, the
/// capture's block lengths, how touch-queue lives end, and the drift of E0, E1 and E2's fits.
fn section_290(report: &mut Report, collect: &Collect) {
    let p = &collect.phase_290;
    say!(
        report,
        "\n== #290 phase 1: why the queue-reactive fit is transient (training part {}) ==",
        window_text(&collect.protocol.queue_train())
    );
    say!(
        report,
        "API: backtest::queue_reactive::{{ZeroWaitPolicy, collapse_blocks, estimate_parameters}}. \
         E0 is #90's fit (zero waits left out); E1 counts every observation, a zero wait adding \
         no time; E2 collapses each block per queue into its net change, then counts as E1. Each \
         is kept one in {} observations. A touch-queue life is a side's limit 0 between two mid \
         moves; it ended emptied if the last event at it left nothing, moved if it left a size.",
        collect.strides.queue
    );
    say!(
        report,
        "(1) Observations by limit and type: with a positive wait / with a zero wait (zero share); \
         {} with an invalid wait.",
        p.invalid_waits
    );
    for (depth, kinds) in p.counts.iter().enumerate() {
        let cells: Vec<String> = ["limit", "cancel", "market"]
            .iter()
            .zip(kinds)
            .map(|(name, [positive, zero])| {
                let share = *zero as f64 / (*positive + *zero).max(1) as f64;
                format!("{name} {positive} / {zero} ({share:.3})")
            })
            .collect();
        say!(report, "    limit {}: {}", depth + 1, cells.join("; "));
    }
    say!(
        report,
        "    Mean size change per warm event, in event sizes (descriptive): {}",
        p.size_change
            .iter()
            .enumerate()
            .map(|(depth, kinds)| {
                let cells: Vec<String> = ["limit", "cancel", "market"]
                    .iter()
                    .zip(kinds)
                    .map(|(name, (sum, n))| format!("{name} {:.3}", sum / (*n).max(1) as f64))
                    .collect();
                format!("limit {}: {}", depth + 1, cells.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ")
    );
    let mean_gap_ms = p.gap_total_ns as f64 / p.blocks.max(1) as f64 / NS_PER_MS as f64;
    let bins: Vec<String> = p
        .gaps
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let label = match (
                i.checked_sub(1)
                    .and_then(|j| qr_290::BLOCK_GAP_BOUNDS_MS.get(j)),
                qr_290::BLOCK_GAP_BOUNDS_MS.get(i),
            ) {
                (None, Some(hi)) => format!("<{hi}"),
                (Some(lo), Some(hi)) => format!("{lo}-{hi}"),
                (Some(lo), None) => format!(">={lo}"),
                (None, None) => "-".to_string(),
            };
            format!("{label} ms: {n}")
        })
        .collect();
    say!(
        report,
        "(2) Blocks (distinct event times): {} gaps, mean {mean_gap_ms:.1} ms, {:.2} events per \
         block, {:.3} of blocks with more than one event. Gaps: {}",
        p.blocks,
        p.block_events as f64 / (p.blocks + 1) as f64,
        p.multi_event_blocks as f64 / p.blocks.max(1) as f64,
        bins.join("; ")
    );
    let share = p.exit_by_move_share();
    say!(
        report,
        "(3) Touch-queue lives at a mid move, both sides: emptied {}, moved while not empty {}, \
         unobserved {}; share moved among the known {}",
        p.lives[0],
        p.lives[1],
        p.lives[2],
        share.map_or("-".to_string(), |s| format!("{s:.3}"))
    );
    say!(
        report,
        "    E3's touch resets on the training part: {} at a move with no emptied touch, {} \
         sides with no touch seen since the last move left out; {} kept from every part, one \
         in {}, {} of them with the new touch's size.",
        p.resets_made,
        p.resets_skipped,
        p.resets.len(),
        collect.strides.queue,
        p.resets.iter().filter(|r| r.queue_after.is_some()).count()
    );
    say!(
        report,
        "    Per UTC day, every part: mid moves, resets made, sides left out: {}",
        p.resets_by_day
            .iter()
            .map(|(day, [made, skipped, moves])| {
                format!("{} {moves} {made} {skipped}", &utc(day * DAY_MS)[..10])
            })
            .collect::<Vec<_>>()
            .join("; ")
    );
    if !matches!(collect.sizes.as_ref().map(|s| &s.queue.sizes), Some(Ok(_))) {
        say!(report, "(4): no event sizes.");
        return;
    }
    let window = collect.protocol.queue_train();
    let fits: Vec<Fit290> = ESTIMATORS_290
        .iter()
        .map(|name| match collect.record_290(name, &window) {
            Ok(record) => Fit290 {
                name,
                events: record.events,
                excluded: record.excluded_events,
                params: match record.fit {
                    QrFit::Calibrated(params) => Ok(params),
                    other => Err(format!("{other:?}")),
                },
            },
            Err(error) => Fit290 {
                name,
                events: 0,
                excluded: 0,
                params: Err(error),
            },
        })
        .collect();
    say!(
        report,
        "(4) Training fits; drift = limit - cancel - market per second at limit 1, at sizes \
         {QR_DAY_SIZES:?}; mean touch queue under the invariant law at limit 1 and over a {} s \
         run from seed {}:",
        qr_290::WINDOW_SECONDS,
        qr_290::QR_SEED
    );
    let mut transient = Vec::new();
    for f in &fits {
        let params = match &f.params {
            Ok(params) => params,
            Err(error) => {
                say!(
                    report,
                    "    {}: {} events used, {} excluded; no fit: {error}",
                    f.name,
                    f.events,
                    f.excluded
                );
                transient.push((f.name, None));
                continue;
            }
        };
        let rates = params.rates(0).unwrap_or_default();
        let supported: Vec<usize> = (0..rates.len())
            .filter(|n| params.is_supported(0, *n))
            .collect();
        let rising = supported
            .iter()
            .filter(|n| rates.get(**n).is_some_and(|(l, c, m)| l - c - m > 0.0))
            .count();
        let drifts: Vec<String> = QR_DAY_SIZES
            .iter()
            .map(|n| match rates.get(*n) {
                Some((l, c, m)) if params.is_supported(0, *n) => format!("{n}: {:+.3}", l - c - m),
                _ => format!("{n}: -"),
            })
            .collect();
        let invariant = params.invariant(0).map_or(f64::NAN, invariant_mean);
        let run = f.run_mean_touch();
        let run_text = match &run {
            Ok((mean, None)) => format!("{mean:.1}"),
            Ok((mean, Some(stop))) => format!("{mean:.1} (stopped: {stop})"),
            Err(error) => format!("not evaluable: {error}"),
        };
        say!(
            report,
            "    {}: {} events used, {} excluded; θ {:.4}; drift {}; rising at {rising} of {} \
             supported sizes; invariant mean {invariant:.1}; run mean {run_text}",
            f.name,
            f.events,
            f.excluded,
            params.theta(),
            drifts.join(", "),
            supported.len()
        );
        transient.push((
            f.name,
            run.ok()
                .map(|(mean, _)| qr_290::reaches(mean, qr_290::E3_TRANSIENT_SHARE_OF_CAP)),
        ));
    }
    let transient_of = |name| {
        transient
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, t)| *t)
    };
    let moves = share.map(|s| s >= qr_290::E3_MIN_EXIT_BY_MOVE_SHARE);
    let both = transient_of("E1")
        .zip(transient_of("E2"))
        .map(|(a, b)| a && b);
    say!(report, "(5) {}.", qr_290::E3_CONDITION);
    say!(
        report,
        "    Exits by move reach {}: {}; E1 and E2 transient: {}. E3 {}.",
        qr_290::E3_MIN_EXIT_BY_MOVE_SHARE,
        moves.map_or("unknown".to_string(), |m| m.to_string()),
        both.map_or("unknown".to_string(), |b| b.to_string()),
        match moves.zip(both) {
            Some((true, true)) => "is built",
            Some(_) => "is not built",
            None => "cannot be decided",
        }
    );
}

/// #290's phase 3: each estimator and its constant baseline against #286's windows and the
/// training window, the choice on the training window, and the holdout's verdict.
fn section_290_book(report: &mut Report, collect: &Collect) {
    say!(
        report,
        "\n== #290 phase 3: E0 to E3, and E3 inv as a sensitivity row, against the data, each \
         with its constant baseline =="
    );
    say!(
        report,
        "Windows as #90 (b) to (d), plus the training part's first hours. #286 already read the \
         holdout window. Stationary: the model's mean touch queue stays under {} of the cap of \
         {}. Choice, fixed before any run: among the stationary fits on the training window, the \
         lowest queue size TV at limit 1, the first of E0, E1, E2 on a tie. Stop rule: no \
         stationary fit, no holdout. Success: the chosen fit's queue size TV below its constant \
         baseline's at every limit on the holdout. Model runs from seed {}.",
        qr_290::STATIONARY_SHARE_OF_CAP,
        qr_290::MAX_QUEUE,
        qr_290::QR_SEED
    );
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let resets = collect.phase_290.sorted_resets();
    for window in &collect.book_windows {
        let name = match window.kind {
            WindowKind::Train => "training window, training fits, in sample".to_string(),
            WindowKind::Holdout => "holdout, training fits, out of sample".to_string(),
            WindowKind::Day(_) => "day, own fits, in sample".to_string(),
        };
        say!(report, "{name}: {}", window_text(&window.window));
        let fit_window = match window.kind {
            WindowKind::Train | WindowKind::Holdout => collect.protocol.queue_train(),
            WindowKind::Day(day) => day,
        };
        let inside: Vec<&ResetObservation> = resets
            .iter()
            .filter(|r| fit_window.start_ms <= r.timestamp_ms && r.timestamp_ms < fit_window.end_ms)
            .collect();
        say!(
            report,
            "    E3's fit window {} holds {} kept resets, {} at a size from 1 to the cap, {} with \
             a new size",
            window_text(&fit_window),
            inside.len(),
            inside
                .iter()
                .filter(|r| (1..=qr_290::MAX_QUEUE).contains(&r.queue_before))
                .count(),
            inside.iter().filter(|r| r.queue_after.is_some()).count()
        );
        let lines = match &window.result_290 {
            None => {
                say!(report, "    not reached: the stretch ended before it");
                continue;
            }
            Some(Err(reason)) => {
                say!(report, "    {reason}");
                continue;
            }
            Some(Ok((data_mean, lines))) => {
                say!(report, "    data mean touch queue {data_mean:.2}");
                lines
            }
        };
        for line in lines {
            let label = if line.baseline {
                format!("{} constant", line.estimator)
            } else {
                line.estimator.to_string()
            };
            match &line.outcome {
                Ok((r, mean)) => say!(
                    report,
                    "    {label}: queue size TV by limit [{}]; spread TV {:.3}; wait KS by limit \
                     [{}]; flow relative error L/C/M [{}]; mean touch queue {mean:.2} ({}); \
                     model {} events, stopped {:?}{}",
                    list(&r.queue_size),
                    r.spread,
                    list(&r.waiting_law),
                    list(&r.flow),
                    if stationary_290(*mean) {
                        "stationary"
                    } else {
                        "transient"
                    },
                    r.model_events,
                    r.model_stopped,
                    match line.reset_sizes {
                        Some((sizes, largest)) => format!(
                            "; reset rate positive at {sizes} sizes, the largest {largest}; \
                             {} resets in the run, whose smallest touch queue is {}",
                            line.run_resets,
                            line.run_min_touch
                                .map_or("-".to_string(), |m| m.to_string())
                        ),
                        None => String::new(),
                    }
                ),
                Err(error) => say!(report, "    {label}: not evaluable: {error}"),
            }
        }
        match window.kind {
            WindowKind::Train => match &collect.phase_290.choice {
                Some(Ok(chosen)) => say!(report, "    chosen: {chosen}"),
                Some(Err(reason)) => say!(report, "    no choice: {reason}"),
                None => say!(report, "    no choice made"),
            },
            WindowKind::Holdout => {
                if let Some(Ok(chosen)) = &collect.phase_290.choice {
                    let verdict = match beats_baseline_290(lines, chosen) {
                        Some(true) => "success: below its constant baseline at every limit",
                        Some(false) => "no success: not below its constant baseline at every limit",
                        None => "not evaluable",
                    };
                    say!(report, "    {chosen} (chosen): {verdict}");
                }
            }
            WindowKind::Day(_) => {}
        }
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
}

fn section_90_book(report: &mut Report, collect: &Collect) {
    say!(
        report,
        "\n== #90 (b) queue size, spread, waiting time and flow laws against the data and a \
         constant intensity baseline, (c) out-of-sample discrepancies and state coverage, (d) \
         discrepancies per day =="
    );
    say!(
        report,
        "API: hyperliquid::{{ReplayConfig::with_book_samples, book_samples, \
         queue_observations}} for the data, backtest::queue_reactive::{{QrSimulator::run, \
         QrParameters::constant_baseline, compare_trajectories}}. A sample shows the book after \
         every diff before its time, its limits counted from the mid as the depths are, in event \
         sizes; the spread is the touch's width in ticks. Events of one block share a time, so \
         most data waits are zero: compare_trajectories leaves them out of the waiting times and \
         counts them, and they stay in the flow. The model's limits count from a reference half \
         a tick off the grid, so a touch two ticks wide is a spread of 2 here with both first \
         limits filled, which the model shows only with one empty."
    );
    for window in &collect.book_windows {
        let name = match window.kind {
            WindowKind::Train => "(290) training window, training fit, in sample",
            WindowKind::Holdout => "(b), (c) holdout, training fit, out of sample",
            WindowKind::Day(_) => "(d) day, own fit, in sample",
        };
        say!(
            report,
            "{name}: {}; {} samples and {} events compared{}",
            window_text(&window.window),
            window.data.0,
            window.data.1,
            if window.samples_without_sizes > 0 {
                format!(
                    "; {} samples before the event sizes were known, left out",
                    window.samples_without_sizes
                )
            } else {
                String::new()
            }
        );
        match &window.result {
            Some(lines) => {
                for line in lines {
                    say!(report, "    {line}");
                }
            }
            None => say!(report, "    not reached: the stretch ended before it"),
        }
    }
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
    use market_maker_rs::backtest::hyperliquid::{
        BookDiff, ORDER_STATUS_RECORD_BYTES, OrderStatusRecord, STATUS_OPEN, decode_order_status,
        parse_book_diff,
    };

    const START: u64 = 1_764_583_200_000;

    fn touch(time_ms: u64, mid: i64, side: Option<usize>, level: Option<i64>) -> TouchEvent {
        TouchEvent {
            time_ns: time_ms * NS_PER_MS,
            mid: Some(Decimal::from(mid)),
            side,
            level_after: level.map(Decimal::from),
            level_before: Decimal::ZERO,
        }
    }

    #[test]
    fn test_290_touch_lives_end_emptied_moved_or_unobserved() {
        let mut p = Phase290::default();
        p.take_touch(touch(0, 100, Some(0), Some(0)), true, Decimal::ONE, 1);
        p.take_touch(touch(0, 100, Some(1), Some(5)), true, Decimal::ONE, 1);
        assert_eq!(p.lives, [0, 0, 0], "the first mid closes nothing");
        p.take_touch(touch(1, 101, None, Some(3)), true, Decimal::ONE, 1);
        assert_eq!(p.lives, [1, 1, 0]);
        p.take_touch(touch(2, 101, Some(0), None), true, Decimal::ONE, 1);
        p.take_touch(touch(3, 102, None, None), true, Decimal::ONE, 1);
        assert_eq!(
            p.lives,
            [1, 1, 2],
            "an unknown cause and a quiet side are unobserved"
        );
        p.take_touch(touch(4, 102, Some(1), Some(-1)), true, Decimal::ONE, 1);
        p.take_touch(touch(5, 101, None, None), true, Decimal::ONE, 1);
        assert_eq!(p.lives, [2, 1, 3]);
        assert_eq!(p.exit_by_move_share(), Some(1.0 / 3.0));
        assert_eq!(Phase290::default().exit_by_move_share(), None);
    }

    #[test]
    fn test_290_a_move_that_empties_no_touch_resets_each_seen_side() {
        let mut p = Phase290::default();
        let at = |p: &mut Phase290, e: TouchEvent, counting| {
            p.take_touch(e, counting, Decimal::from(2), 1);
        };
        at(&mut p, touch(0, 100, Some(0), Some(10)), true);
        at(&mut p, touch(1, 100, Some(1), Some(6)), true);
        // The move: both touches left a size, so both reset, their sizes in units of 2.
        at(&mut p, touch(3, 101, None, None), true);
        assert_eq!(p.resets_made, 2);
        assert!(p.resets.is_empty(), "each waits for its new size");
        // The bid's next touch event gives its new size, from the level before it.
        let next = TouchEvent {
            level_before: Decimal::from(14),
            ..touch(4, 101, Some(0), Some(16))
        };
        at(&mut p, next, true);
        assert_eq!(
            p.resets,
            vec![ResetObservation {
                timestamp_ms: 3,
                side: QrSide::Bid,
                queue_before: 5,
                waiting_seconds: 0.003,
                queue_after: Some(7),
            }]
        );
        // The next move ends the ask's wait with no new size, and resets the bid again.
        at(&mut p, touch(6, 100, None, None), true);
        assert_eq!(p.resets.len(), 2);
        assert_eq!(
            p.resets[1],
            ResetObservation {
                timestamp_ms: 3,
                side: QrSide::Ask,
                queue_before: 3,
                waiting_seconds: 0.002,
                queue_after: None,
            }
        );
        assert_eq!(
            (p.resets_made, p.resets_skipped),
            (3, 1),
            "the ask was not seen"
        );
        // A move that empties a touch resets nothing.
        at(&mut p, touch(7, 100, Some(1), Some(0)), true);
        at(&mut p, touch(8, 100, Some(0), Some(4)), true);
        at(&mut p, touch(9, 99, None, None), true);
        assert_eq!((p.resets_made, p.resets_skipped), (3, 1));
        assert_eq!(
            p.resets.len(),
            3,
            "the bid's reset took its new size at its next touch event"
        );
        // Off the training part, resets are kept but not counted.
        at(&mut p, touch(10, 99, Some(0), Some(4)), false);
        at(&mut p, touch(10, 99, Some(1), Some(4)), false);
        at(&mut p, touch(11, 100, None, None), false);
        at(&mut p, touch(12, 101, None, None), false);
        assert_eq!(p.resets.len(), 5);
        assert_eq!((p.resets_made, p.resets_skipped), (3, 1));
        let sorted = p.sorted_resets();
        assert!(sorted.is_sorted_by_key(|r| r.timestamp_ms));
        // By day, on every part: 5 moves, 5 resets made, 3 sides unseen.
        assert_eq!(p.resets_by_day.get(&0), Some(&[5, 3, 5]));
        // From a one sided book, no reset.
        let mut q = Phase290::default();
        at(&mut q, touch(0, 100, Some(0), Some(4)), true);
        at(
            &mut q,
            TouchEvent {
                mid: None,
                ..touch(1, 0, None, None)
            },
            true,
        );
        assert_eq!(q.resets_made + q.resets_skipped, 0);
    }

    #[test]
    fn test_290_blocks_are_distinct_times_and_their_gaps_fall_in_bins() {
        let mut p = Phase290::default();
        for ms in [0, 0, 5, 5, 2_000] {
            p.take_touch(touch(ms, 100, None, None), true, Decimal::ONE, 1);
        }
        assert_eq!((p.blocks, p.block_events, p.multi_event_blocks), (2, 5, 2));
        let mut want = [0; qr_290::BLOCK_GAP_BOUNDS_MS.len() + 1];
        want[1] = 1;
        want[7] = 1;
        assert_eq!(p.gaps, want);
        assert_eq!(p.gap_total_ns, 2_000 * u128::from(NS_PER_MS));
    }

    #[test]
    fn test_290_queue_counts_split_zero_waits_and_collapse_per_block() {
        let event = |depth, kind, queue_before, waiting_seconds| QueueObservation {
            timestamp_ms: 1,
            side: QrSide::Bid,
            depth,
            kind,
            queue_before,
            waiting_seconds,
        };
        let events = [
            event(0, QrEventKind::Limit, 1, 1.0),
            event(0, QrEventKind::Cancel, 2, 0.0),
            event(0, QrEventKind::Market, 1, 0.0),
            event(1, QrEventKind::Market, 3, 2.0),
            event(1, QrEventKind::Limit, 2, -1.0),
        ];
        let mut p = Phase290::default();
        p.take_counts(&events);
        p.take_lists(&events, 1);
        assert_eq!(p.counts[0], [[1, 0], [0, 1], [0, 1]]);
        assert_eq!(p.counts[1], [[0, 0], [0, 0], [1, 0]]);
        assert_eq!(p.invalid_waits, 1);
        assert_eq!(p.all, events.to_vec());
        assert_eq!(
            p.collapsed,
            vec![
                event(0, QrEventKind::Market, 1, 1.0),
                event(1, QrEventKind::Market, 3, 2.0),
                event(1, QrEventKind::Limit, 2, -1.0),
            ]
        );
        let mut thin = Phase290::default();
        thin.take_lists(&events, 2);
        assert_eq!(thin.all.len(), 3);
    }

    fn line(estimator: &'static str, baseline: bool, tv: &[f64], mean: f64) -> Line290 {
        let report: DiscrepancyReport = serde_json::from_value(serde_json::json!({
            "queue_size": tv, "spread": 0.0, "waiting_time": [0.0], "waiting_law": [0.0],
            "flow": [0.0, 0.0, 0.0], "flow_law": [0.0, 0.0, 0.0], "coverage": 1.0,
            "data_samples": 1, "data_events": 1, "model_samples": 1, "model_events": 1,
            "model_stopped": null, "data_invalid_waiting": 0, "model_invalid_waiting": 0
        }))
        .unwrap();
        Line290 {
            estimator,
            baseline,
            outcome: Ok((report, mean)),
            run_resets: 0,
            run_min_touch: None,
            reset_sizes: None,
        }
    }

    #[test]
    fn test_290_choice_takes_the_lowest_first_limit_tv_among_stationary_fits() {
        let lines = [
            line("E0", false, &[0.2, 0.1], 150.0),
            line("E0", true, &[0.0, 0.0], 1.0),
            line("E1", false, &[0.4, 0.1], 20.0),
            line("E2", false, &[0.3, 0.9], 99.9),
        ];
        assert_eq!(
            choose_290(&lines),
            Ok("E2"),
            "E0 is transient, baselines never count"
        );
        let tie = [
            line("E1", false, &[0.3], 1.0),
            line("E2", false, &[0.3], 1.0),
        ];
        assert_eq!(choose_290(&tie), Ok("E1"), "the first on a tie");
        let sensitivity = [
            line("E3 inv", false, &[0.0], 1.0),
            line("E3", false, &[0.5], 1.0),
        ];
        assert_eq!(
            choose_290(&sensitivity),
            Ok("E3"),
            "a sensitivity row is never chosen"
        );
        let mut failed = line("E2", false, &[0.0], 1.0);
        failed.outcome = Err("no fit".to_string());
        let none = [line("E0", false, &[0.1], 100.0), failed];
        assert!(
            choose_290(&none).is_err(),
            "no stationary fit, so no holdout"
        );
        assert!(stationary_290(99.99) && !stationary_290(100.0));
    }

    #[test]
    fn test_290_reset_sizes_are_the_positive_rates_and_the_largest() {
        let rates: Vec<f64> = (0..=QR_MAX_QUEUE as usize)
            .map(|n| if n == 3 || n == 40 { 0.5 } else { 0.0 })
            .collect();
        let law: Vec<f64> = (0..rates.len())
            .map(|n| if n == 1 { 1.0 } else { 0.0 })
            .collect();
        let table: Vec<(f64, f64, f64)> = (0..rates.len())
            .map(|n| {
                let last = n + 1 == rates.len();
                (
                    if last { 0.0 } else { 1.0 },
                    if n == 0 { 0.0 } else { 1.0 },
                    0.0,
                )
            })
            .collect();
        let params = QrParameters::new(
            dec!(1),
            1.0,
            0.0,
            vec![dec!(1)],
            vec![table],
            vec![vec![true; rates.len()]],
            UnsupportedPolicy::NearestSupported,
        )
        .unwrap();
        assert_eq!(reset_sizes(&params), None);
        let with = params.with_resets(rates, law).unwrap();
        assert_eq!(reset_sizes(&with), Some((2, 40)));
    }

    #[test]
    fn test_290_success_needs_every_limit_below_its_own_baseline() {
        let lines = [
            line("E1", false, &[0.2, 0.3], 1.0),
            line("E1", true, &[0.25, 0.35], 1.0),
            line("E2", false, &[0.2, 0.4], 1.0),
            line("E2", true, &[0.25, 0.4], 1.0),
        ];
        assert_eq!(beats_baseline_290(&lines, "E1"), Some(true));
        assert_eq!(
            beats_baseline_290(&lines, "E2"),
            Some(false),
            "a tie is not below"
        );
        assert_eq!(beats_baseline_290(&lines, "E0"), None);
        let short = [
            line("E1", false, &[0.1], 1.0),
            line("E1", true, &[0.2, 0.2], 1.0),
        ];
        assert_eq!(beats_baseline_290(&short, "E1"), None);
    }

    #[test]
    fn test_290_protocol_matches_this_example() {
        assert_eq!(qr_290::MAX_QUEUE, QR_MAX_QUEUE);
        assert_eq!(qr_290::QR_SEED, QR_SEED);
        assert_eq!(qr_290::WINDOW_SECONDS, BOOK_WINDOW_MS as f64 / 1e3);
        assert!(qr_290::reaches(100.0, 0.5));
        assert!(!qr_290::reaches(99.9, 0.5));
        let samples = [
            BookSample {
                timestamp_ms: 0,
                bid: [2, 0, 0, 0, 0],
                ask: [4, 0, 0, 0, 0],
                spread_ticks: 1,
            },
            BookSample {
                timestamp_ms: 1,
                bid: [0, 0, 0, 0, 0],
                ask: [2, 0, 0, 0, 0],
                spread_ticks: 1,
            },
        ];
        assert_eq!(mean_touch(&samples), 2.0);
        assert_eq!(mean_touch(&[]), 0.0);
    }

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
    fn test_book_windows_are_declared_and_compared_once_their_fit_can_be_made() {
        // Starts at 20:00 UTC, so the stretch holds whole days and the holdout starts mid-day.
        let start = 1_764_619_200_000;
        let end = start + 5 * DAY_MS;
        let protocol = Protocol::new(start, end).unwrap();
        let mut collect = Collect::new(
            protocol,
            0,
            dec!(1),
            "synthetic".to_string(),
            Strides::for_hours(1),
        );
        let holdout = protocol.holdout();
        let days = protocol.days();
        assert_eq!(collect.book_windows.len(), 2 + days.len());
        let train = &collect.book_windows[0];
        assert_eq!(train.kind, WindowKind::Train);
        assert_eq!(train.window.start_ms, protocol.queue_train().start_ms);
        assert_eq!(
            train.window.end_ms,
            protocol.queue_train().start_ms + BOOK_WINDOW_MS
        );
        assert_eq!(
            train.due_ms, protocol.train_end_ms,
            "the training fit needs training"
        );
        let first = &collect.book_windows[1];
        assert_eq!(first.kind, WindowKind::Holdout);
        assert_eq!(first.window.start_ms, holdout.start_ms);
        assert_eq!(first.window.end_ms, holdout.start_ms + BOOK_WINDOW_MS);
        assert_eq!(first.due_ms, first.window.end_ms);
        for (window, day) in collect.book_windows[2..].iter().zip(&days) {
            assert_eq!(window.kind, WindowKind::Day(*day));
            assert_eq!(window.window.start_ms, day.start_ms);
            assert_eq!(window.window.end_ms, day.start_ms + BOOK_WINDOW_MS);
            assert_eq!(
                window.due_ms, day.end_ms,
                "a day's own fit needs the whole day"
            );
            assert!(window.holds(day.start_ms));
            assert!(!window.holds(day.start_ms + BOOK_WINDOW_MS));
            assert!(!window.holds(day.start_ms - 1));
        }
        // A short holdout bounds its window.
        let short = Protocol::new(START, START + 4 * HOUR_MS).unwrap();
        let windows = BookWindow::all(&short);
        assert_eq!(windows[1].window.end_ms, short.end_ms);
        assert_eq!(windows[1].due_ms, short.end_ms);

        // Nothing is compared before its time; then each once, and its data is dropped.
        let sample = BookSample {
            timestamp_ms: holdout.start_ms,
            bid: [1; 5],
            ask: [1; 5],
            spread_ticks: 1,
        };
        for window in &mut collect.book_windows {
            window.samples.push(sample);
        }
        collect.compare_due(holdout.start_ms + BOOK_WINDOW_MS - 1);
        let day_due = days[0].end_ms;
        assert!(day_due < holdout.start_ms);
        assert!(collect.book_windows[1].result.is_none());
        assert_eq!(collect.book_windows[1].samples.len(), 1);
        assert!(collect.book_windows[0].result.is_some());
        assert!(collect.book_windows[2].result.is_some());
        // With no fit, nothing is stationary on the training window, so #290 stops there.
        assert!(matches!(collect.phase_290.choice, Some(Err(_))));
        collect.compare_due(holdout.start_ms + BOOK_WINDOW_MS);
        let window = &collect.book_windows[1];
        assert!(window.samples.is_empty());
        assert_eq!(window.data, (1, 0));
        assert_eq!(
            window.result,
            Some(vec!["not evaluable: no fit: no event sizes".to_string()])
        );
        assert!(matches!(
            &window.result_290,
            Some(Err(reason)) if reason.starts_with("not run, stop rule")
        ));
        let read = holdout.start_ms + BOOK_WINDOW_MS;
        for window in &collect.book_windows {
            assert_eq!(window.result.is_some(), window.due_ms <= read);
        }
        assert!(collect.book_windows.iter().any(|w| w.result.is_none()));
        collect.compare_due(u64::MAX);
        assert!(collect.book_windows.iter().all(|w| w.result.is_some()));
    }

    /// An order status record, by the schema's offsets, with two decimals.
    fn status(
        ns: u64,
        id: u8,
        is_ask: bool,
        cents: u32,
        size_cents: u32,
        oid: u64,
    ) -> OrderStatusRecord {
        let fixed = |value: u32| (2u32 << 29) | value;
        let mut bytes = vec![0u8; ORDER_STATUS_RECORD_BYTES];
        bytes[0..8].copy_from_slice(&ns.to_le_bytes());
        bytes[13] = id;
        bytes[14] = u8::from(is_ask);
        bytes[15..19].copy_from_slice(&fixed(cents).to_le_bytes());
        bytes[19..23].copy_from_slice(&fixed(size_cents).to_le_bytes());
        bytes[23..31].copy_from_slice(&oid.to_le_bytes());
        decode_order_status(&bytes).unwrap()
    }

    fn diff(oid: u64, side: &str, price: &str, size: &str) -> BookDiff {
        let line = format!(
            r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":{{"new":{{"sz":"{size}"}}}}}}"#
        );
        parse_book_diff(&line).unwrap().1
    }

    #[test]
    fn test_a_book_window_keeps_the_events_and_samples_inside_it_only() {
        let protocol = Protocol::new(START, START + 10 * HOUR_MS).unwrap();
        let mut collect = Collect::new(
            protocol,
            0,
            dec!(1),
            "synthetic".to_string(),
            Strides::for_hours(1),
        );
        let sized: Vec<SizedEvent> = (0..MIN_SIZED_EVENTS)
            .flat_map(|i| {
                (0..3u8).map(move |depth| SizedEvent {
                    timestamp_ms: START + i * SECOND_MS,
                    depth,
                    size: dec!(1),
                })
            })
            .collect();
        let sizes = |limits| {
            estimate_event_sizes(&sized, &protocol.sizes(), limits, MIN_SIZED_EVENTS).unwrap()
        };
        collect.sizes = Some(Sizes {
            queue: sizes(LIMITS),
            hawkes: sizes(1),
        });
        // A two sided book 10 s before the holdout, then one bid a second for 5 s from its start.
        let h = protocol.holdout().start_ms * NS_PER_MS;
        let ns = |seconds: i64| h.checked_add_signed(seconds * 1_000_000_000).unwrap();
        let mut statuses = vec![
            status(ns(-10), STATUS_OPEN, false, 10_000, 200, 1),
            status(ns(-9), STATUS_OPEN, true, 10_100, 300, 2),
            status(ns(-8), STATUS_OPEN, true, 10_200, 100, 3),
        ];
        // The third is an observation, at limit 1, before the window.
        let mut diffs = vec![
            diff(1, "B", "100", "2"),
            diff(2, "A", "101", "3"),
            diff(3, "A", "102", "1"),
        ];
        for (i, at) in (0..5).enumerate() {
            let oid = 10 + i as u64;
            statuses.push(status(ns(at), STATUS_OPEN, false, 10_000, 100, oid));
            diffs.push(diff(oid, "B", "100", "1"));
        }
        let config = ReplayConfig::new(dec!(1), LIMITS, 0, RESET_GAP_NS)
            .unwrap()
            .with_book_samples(BOOK_SAMPLE_MS)
            .unwrap();
        let mut replay = Replay::new(config);
        let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
        assert!(
            output
                .book
                .iter()
                .any(|s| s.value.timestamp_ms < protocol.holdout().start_ms)
        );
        collect.take(&output);
        let window = &collect.book_windows[1];
        assert_eq!(window.kind, WindowKind::Holdout);
        // The five bids at the touch, and none of the opening's diffs before the window.
        assert_eq!(window.events.len(), 5);
        assert!(window.events.iter().all(|e| window.holds(e.timestamp_ms)));
        // The grid times from the holdout's start to the last diff: 0 to 4 s, each showing the
        // bids before it.
        let bids: Vec<u32> = window.samples.iter().map(|s| s.bid[0]).collect();
        assert_eq!(bids, vec![2, 3, 4, 5, 6]);
        assert!(
            window
                .samples
                .iter()
                .all(|s| s.ask[0] == 3 && s.ask[1] == 1)
        );
        assert_eq!(window.samples_without_sizes, 0);
        assert!(window.result.is_none());
        // #290 counts the training part only, and keeps its estimators' inputs from every part.
        let counted = |c: &Collect| c.phase_290.counts.iter().flatten().flatten().sum::<u64>();
        assert_eq!(counted(&collect), 0);
        assert!(!collect.phase_290.all.is_empty());
        let first_hour = output.events.clone();

        // An hour that reads past the window's end compares it and drops its data. Its diff ends
        // a silence that resets the replay, so no sample spans that silence.
        let past = ns(i64::try_from(BOOK_WINDOW_MS / SECOND_MS).unwrap() + 1);
        let output = replay
            .ingest(
                &[status(past, STATUS_OPEN, false, 10_000, 100, 20)],
                &[diff(20, "B", "100", "1")],
                &[],
            )
            .unwrap();
        collect.take(&output);
        let window = &collect.book_windows[1];
        assert!(window.result.is_some());
        assert!(window.samples.is_empty() && window.events.is_empty());
        assert_eq!(window.data, (5, 5));
        collect.convert(Part::Train, &first_hour);
        assert!(counted(&collect) > 0);
        assert!(
            collect
                .phase_290
                .size_change
                .iter()
                .flatten()
                .any(|(_, n)| *n > 0)
        );
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

    #[test]
    fn test_walk_forward_mids_sample_a_touch_that_does_not_move() {
        use market_maker_rs::backtest::hyperliquid::{Inferred, InferredTime, TimeEvidence};
        let end = whole_seconds(START + 2 * DAY_MS);
        let protocol = Protocol::new(START, end).unwrap();
        let mut collect = Collect::new(
            protocol,
            0,
            dec!(1),
            "synthetic".to_string(),
            Strides::for_hours(1),
        );
        let at = |ms: u64| InferredTime {
            timestamp_ns: ms * NS_PER_MS,
            evidence: TimeEvidence::Joined,
        };
        // Two minutes of a live book whose touch stays at 100 / 101 while its sizes change each
        // second, then one move. The replay's prices hold the mid only when it changes.
        let mut output = ReplayOutput::default();
        for second in 0..=120u64 {
            let ms = START + second * SECOND_MS;
            let ask = if second == 120 { dec!(102) } else { dec!(101) };
            output.tops.push(Inferred {
                value: TopOfBook {
                    timestamp_ms: ms,
                    bid: dec!(100),
                    ask,
                    bid_size: Decimal::from(second + 1),
                    ask_size: dec!(1),
                },
                time: at(ms),
                epoch: 0,
            });
        }
        for (ms, price) in [(START, dec!(100.5)), (START + 120 * SECOND_MS, dec!(101))] {
            output.prices.push(Inferred {
                value: PriceSample {
                    timestamp_ms: ms,
                    price,
                },
                time: at(ms),
                epoch: 0,
            });
        }
        collect.take(&output);
        assert_eq!(collect.prices.len(), 121);
        let longest = collect
            .prices
            .windows(2)
            .map(|pair| pair[1].timestamp_ms - pair[0].timestamp_ms)
            .max();
        assert_eq!(longest, Some(SECOND_MS));
        assert_eq!(collect.prices[0].price, dec!(100.5));
        assert_eq!(collect.prices[120].price, dec!(101));
        // A second epoch's tops are not this stretch's.
        let mut other = ReplayOutput::default();
        other.tops.push(Inferred {
            value: TopOfBook {
                timestamp_ms: START + 200 * SECOND_MS,
                bid: dec!(100),
                ask: dec!(101),
                bid_size: dec!(1),
                ask_size: dec!(1),
            },
            time: at(START + 200 * SECOND_MS),
            epoch: 1,
        });
        collect.take(&other);
        assert_eq!(collect.prices.len(), 121);
    }
}
