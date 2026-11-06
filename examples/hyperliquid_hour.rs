//! Converts one coin's hours of the Hyperliquid "An Open Book" capture and reports what could
//! and could not be established.
//!
//! The dataset is real exchange data (Zenodo, DOI 10.5281/zenodo.18184441, CC BY 4.0), and it
//! is not in this repository. This example decompresses nothing: extract the files and pass
//! them decompressed, one triple per hour, in time order, so the book carries from one to the
//! next.
//!
//! Run with
//! `cargo run --release --features serde --example hyperliquid_hour -- <coin> <tick> <warmup minutes> <statuses> <diffs> <trades> [<statuses> <diffs> <trades> ...]`.
//!
//! Up to [`MAX_HOURS_KEPT`] hours keep their events, for the event sizes and the queue and Hawkes
//! conversions. A longer run reports the counts only, as an hour holds millions of events.
//!
//! The tick is declared, and the report shows what the prices seen are multiples of. Fees and
//! funding are in no file here. The book starts empty, so nothing below claims it is complete.
//! A silence over [`RESET_GAP_NS`] empties the book and restarts the warm-up.
//!
//! Each hour gets a line of what it added: diffs, crossed events, resets, orders cut, re-emitted
//! diffs, statuses and diffs repeating the file before, times stepping back and unknown orders
//! after the warm-up, to see where they cluster. Consecutive files overlap by minutes, so each
//! stream's files are read through one `RepeatFilter`.

use std::fs;
use std::io::BufReader;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::hyperliquid::{
    EventCause, RepeatFilter, Replay, ReplayConfig, TimeEvidence, TouchCheck, Warmth,
    decode_order_statuses_after, hawkes_segments, queue_observations, read_book_diffs_after,
    read_trades_after, sized_events,
};
use market_maker_rs::backtest::queue_reactive::{QrSide, estimate_event_sizes};
use market_maker_rs::strategy::walk_forward::TimeWindow;
use rust_decimal::prelude::FromStr;

/// Most hours whose events stay in memory.
const MAX_HOURS_KEPT: usize = 24;
/// Limits a depth counts up to.
const LIMITS: u8 = 3;
/// The silence that resets the replay. Declared here: the capture's own gaps are of minutes.
const RESET_GAP_NS: u64 = 60 * 1_000_000_000;
/// Fewest events a limit's size is estimated from.
const MIN_EVENTS: u64 = 30;
/// FNV-1a 64 offset and prime, to tell inputs apart in a report.
const FNV: (u64, u64) = (0xcbf2_9ce4_8422_2325, 0x0100_0000_01b3);

fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV.0, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV.1)
    })
}

/// The share of a part in a whole, as a percentage. Zero of nothing is zero.
fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

fn touch_line(name: &str, check: &TouchCheck) -> String {
    let total = check.at_touch
        + check.through_touch
        + check.better_than_touch
        + check.no_touch
        + check.after_inherited;
    format!(
        "{name}: {total} trades, at the touch {} ({:.1}%), through it {} ({:.1}%), better than \
         it {} ({:.1}%), no touch {}, after an inherited diff {}",
        check.at_touch,
        percent(check.at_touch, total),
        check.through_touch,
        percent(check.through_touch, total),
        check.better_than_touch,
        percent(check.better_than_touch, total),
        check.no_touch,
        check.after_inherited,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [coin, tick, warmup, files @ ..] = args.as_slice() else {
        return Err("usage: <coin> <tick> <warmup minutes> (<statuses> <diffs> <trades>)+".into());
    };
    if files.is_empty() || files.len() % 3 != 0 {
        return Err("files come in triples of statuses, diffs and trades".into());
    }
    let minutes: u64 = warmup.parse()?;
    let config = ReplayConfig::new(
        Decimal::from_str(tick)?,
        LIMITS,
        minutes * 60 * 1_000_000_000,
        RESET_GAP_NS,
    )?;
    let mut replay = Replay::new(config);
    let (mut events, mut tops, mut market, mut prices, mut references, mut exposures) =
        (Vec::new(), 0usize, 0usize, 0usize, 0usize, 0usize);
    let keep_events = files.len() / 3 <= MAX_HOURS_KEPT;
    let (mut seen, mut inherited, mut cold) = (0usize, 0usize, 0usize);
    let (mut repeated_statuses, mut repeated_diffs, mut repeated_trades) = (0u64, 0u64, 0u64);
    // One filter per stream: consecutive files overlap, and each repeats the tail of the last.
    let (mut status_filter, mut diff_filter, mut trade_filter) = (
        RepeatFilter::new(),
        RepeatFilter::new(),
        RepeatFilter::new(),
    );
    let mut overlap = [0u64; 3];
    let (mut limit, mut cancel, mut fill, mut unknown) = (0usize, 0usize, 0usize, 0usize);
    println!("REAL DATA: the Hyperliquid capture, for {coin}. Times of book diffs are inferred.");
    for triple in files.chunks(3) {
        let [statuses, diffs, trades] = triple else {
            continue;
        };
        let hour = diffs.clone();
        let before = replay.coverage().clone();
        let bytes = fs::read(statuses)?;
        println!(
            "{statuses}: {} bytes, fnv {:016x}",
            bytes.len(),
            digest(&bytes)
        );
        let statuses = decode_order_statuses_after(&bytes, &mut status_filter)?;
        drop(bytes);
        let diffs = read_book_diffs_after(
            BufReader::new(fs::File::open(diffs)?),
            coin,
            &mut diff_filter,
        )?;
        let trades = read_trades_after(
            BufReader::new(fs::File::open(trades)?),
            coin,
            &mut trade_filter,
        )?;
        overlap[0] += statuses.repeated_from_previous_file;
        overlap[1] += diffs.repeated_from_previous_file;
        overlap[2] += trades.repeated_from_previous_file;
        repeated_statuses += statuses.repeated_lines;
        repeated_diffs += diffs.repeated_lines;
        repeated_trades += trades.repeated_lines;
        let output = replay.ingest(&statuses.rows, &diffs.rows, &trades.rows)?;
        tops += output.tops.len();
        market += output.market_events.len();
        prices += output.prices.len();
        references += output.references.len();
        exposures += output.exposures.len();
        seen += output.events.len();
        let after = replay.coverage();
        println!(
            "  {hour}: {} diffs, {} crossed events, resets {} on diffs and {} on trades, {} \
             orders cut, {} diffs re-emitted, {} statuses and {} diffs from the file before, {} \
             times stepping back, {} unknown orders warm",
            after.diffs - before.diffs,
            after.crossed_events - before.crossed_events,
            after.resets_on_diffs - before.resets_on_diffs,
            after.resets_on_trades - before.resets_on_trades,
            after.orders_cut_by_reset - before.orders_cut_by_reset,
            diffs.repeated_lines,
            statuses.repeated_from_previous_file,
            diffs.repeated_from_previous_file,
            after.time_steps_back - before.time_steps_back,
            after.unknown_orders_warm - before.unknown_orders_warm,
        );
        for event in &output.events {
            inherited += usize::from(event.time.evidence == TimeEvidence::Inherited);
            cold += usize::from(event.warmth == Warmth::Cold);
            match event.cause {
                EventCause::Limit => limit += 1,
                EventCause::Cancel => cancel += 1,
                EventCause::Fill => fill += 1,
                _ => unknown += 1,
            }
        }
        if keep_events {
            events.extend(output.events);
        }
    }
    let coverage = replay.coverage();
    println!(
        "\nRead: {} statuses, {} diffs, {} trades",
        coverage.statuses, coverage.diffs, coverage.trades
    );
    println!(
        "Times of diffs: {} joined, {} `new` unjoined, {} `update` unjoined, {} `remove` unjoined, \
         {} stepping back, {} with no time",
        coverage.joined_times,
        coverage.unjoined_new,
        coverage.unjoined_update,
        coverage.unjoined_remove,
        coverage.time_steps_back,
        coverage.diffs_without_time,
    );
    println!("Events: {seen} with an inferred time, {inherited} of them inherited");
    println!(
        "Against the open status: {} side, {} price, {} size mismatches. Update's previous size \
         against the known one: {} mismatches.",
        coverage.side_mismatches,
        coverage.price_mismatches,
        coverage.size_mismatches,
        coverage.update_mismatches,
    );
    println!(
        "Unknown at the start: {} removes or updates of an order not seen, {} after the warm-up. \
         The book started empty; no continuity is claimed.",
        coverage.unknown_orders, coverage.unknown_orders_warm,
    );
    println!(
        "Book: {} crossed events, {} negative levels, {} orders resting at the end, {} duplicate \
         new, {} of another side or price than known, {} removes after a partial fill only",
        coverage.crossed_events,
        coverage.negative_levels,
        coverage.resting_at_end,
        coverage.duplicate_new,
        coverage.order_mismatches,
        coverage.partial_last_status,
    );
    println!(
        "Longest time between diffs: {:.3} s, between trades: {:.3} s.",
        coverage.longest_gap_ns as f64 / 1e9,
        coverage.longest_trade_gap_ns as f64 / 1e9,
    );
    println!(
        "Re-emitted lines left out: {repeated_statuses} statuses, {repeated_diffs} diffs, \
         {repeated_trades} trades; repeating the file before: {} statuses, {} diffs, {} trades. \
         Trades out of order, skipped: {}.",
        overlap[0], overlap[1], overlap[2], coverage.trades_out_of_order,
    );
    println!(
        "A silence over {} s resets the replay: {} on diffs, {} on trades, {} orders cut there.",
        RESET_GAP_NS / 1_000_000_000,
        coverage.resets_on_diffs,
        coverage.resets_on_trades,
        coverage.orders_cut_by_reset,
    );
    println!(
        "Tick: declared {tick}, {} prices off it. Prices seen are multiples of {:?}: observed, \
         not a rule of the venue.",
        coverage.off_tick_prices, coverage.price_grid,
    );
    println!("Fees and funding: in no file here, so declared inputs of whatever consumes this.");
    println!("{}", touch_line("Touch, cold", &coverage.touch_cold));
    println!("{}", touch_line("Touch, warm", &coverage.touch_warm));
    println!("Causes: {limit} limit, {cancel} cancel, {fill} fill, {unknown} unknown");
    println!("Warmth: {cold} cold events of {seen}");
    println!(
        "Outputs: {tops} tops, {market} market events, {prices} prices, {references} mid moves, \
         {exposures} exposures ({} left out)",
        coverage.exposures_left_out,
    );
    if !keep_events {
        println!(
            "More than {MAX_HOURS_KEPT} hours, so no event sizes and no queue or Hawkes conversions."
        );
    }
    let sized: Vec<_> = sized_events(&events).into_iter().map(|e| e.value).collect();
    if let (Some(first), Some(last)) = (sized.first(), sized.last()) {
        let window = TimeWindow::new(first.timestamp_ms, last.timestamp_ms.saturating_add(1))?;
        let record = estimate_event_sizes(&sized, &window, LIMITS, MIN_EVENTS)?;
        println!(
            "Event sizes, on the warm events themselves, not a training split: {:?}, from {:?} \
             events, {} excluded",
            record.sizes, record.events, record.excluded_events,
        );
        if let Ok(sizes) = &record.sizes {
            let queue = queue_observations(&events, sizes)?;
            println!(
                "Queue observations: {}, {} of them with no waiting time, which the estimator \
                 leaves out; left out here {} cold, {} of unknown cause, {} with no depth",
                queue.observations.len(),
                queue.zero_waits,
                queue.cold,
                queue.unknown_cause,
                queue.no_depth,
            );
            if let Some(first) = sizes.first() {
                for side in [QrSide::Bid, QrSide::Ask] {
                    let hawkes = hawkes_segments(&events, side, *first)?;
                    let kept: usize = hawkes.segments.iter().map(|r| r.segment.events.len()).sum();
                    println!(
                        "Hawkes segments, {side:?}: {}, {kept} events; left out {} of no length, \
                         {} departing from an empty queue in event sizes, {} cold events, {} of \
                         unknown cause",
                        hawkes.segments.len(),
                        hawkes.zero_length,
                        hawkes.inconsistent,
                        hawkes.cold,
                        hawkes.unknown_cause,
                    );
                }
            }
        }
    }
    Ok(())
}
