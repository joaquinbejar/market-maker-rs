//! The Hyperliquid capture's conversion, on hand made records in the dataset's own formats.
//!
//! Nothing here reads the dataset: these are small synthetic fixtures. The schema examples come
//! from the dataset's `SCHEMA.md`.

use market_maker_rs::Decimal;
use market_maker_rs::analytics::order_flow::TradeSide;
use market_maker_rs::backtest::hyperliquid::{
    BookChange, BookDiff, CaptureTrade, EventCause, ORDER_STATUS_RECORD_BYTES, OrderStatusRecord,
    Replay, ReplayConfig, STATUS_FILLED, STATUS_OPEN, TimeEvidence, Warmth, book_samples,
    decode_fixed, decode_order_status, decode_order_statuses, hawkes_segments, parse_book_diff,
    parse_time_ns, queue_observations, read_book_diffs, read_trades, sized_events,
};
use market_maker_rs::backtest::hyperliquid::{
    RepeatFilter, decode_order_statuses_after, read_book_diffs_after, read_trades_after,
};
use market_maker_rs::backtest::queue_reactive::{QrEventKind, QrSide};
use market_maker_rs::dec;
use market_maker_rs::strategy::signal_quote::MarketEvent;

const CANCELED: u8 = 2;
const SECOND: u64 = 1_000_000_000;
const START: u64 = 1_000 * SECOND;

/// Encodes with two decimals, in the dataset's encoding.
fn fixed(value: u32) -> u32 {
    (2 << 29) | value
}

/// A record as its bytes, by the schema's offsets.
fn record(ts: u64, status: u8, is_ask: bool, price: u32, size: u32, oid: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; ORDER_STATUS_RECORD_BYTES];
    bytes[0..8].copy_from_slice(&ts.to_le_bytes());
    bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
    bytes[13] = status;
    bytes[14] = u8::from(is_ask);
    bytes[15..19].copy_from_slice(&fixed(price).to_le_bytes());
    bytes[19..23].copy_from_slice(&fixed(size).to_le_bytes());
    bytes[23..31].copy_from_slice(&oid.to_le_bytes());
    bytes[50..54].copy_from_slice(&fixed(size + 1).to_le_bytes());
    bytes
}

fn status(ts: u64, id: u8, is_ask: bool, price: u32, size: u32, oid: u64) -> OrderStatusRecord {
    decode_order_status(&record(ts, id, is_ask, price, size, oid)).unwrap()
}

fn new(oid: u64, side: QrSide, price: Decimal, size: Decimal) -> BookDiff {
    let side = match side {
        QrSide::Bid => "B",
        QrSide::Ask => "A",
        _ => panic!("a side"),
    };
    let line = format!(
        r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":{{"new":{{"sz":"{size}"}}}}}}"#
    );
    parse_book_diff(&line).unwrap().1
}

fn remove(oid: u64, side: &str, price: &str) -> BookDiff {
    let line = format!(
        r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":"remove"}}"#
    );
    parse_book_diff(&line).unwrap().1
}

fn update(oid: u64, side: &str, price: &str, previous: &str, remaining: &str) -> BookDiff {
    let line = format!(
        r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{price}","raw_book_diff":{{"update":{{"origSz":"{previous}","newSz":"{remaining}"}}}}}}"#
    );
    parse_book_diff(&line).unwrap().1
}

fn config(warmup: u64) -> ReplayConfig {
    ReplayConfig::new(dec!(1), 3, warmup, u64::MAX).unwrap()
}

/// Bid 100 of size 2 from oid 1, ask 101 of size 3 from oid 2, at `START` and a second on.
fn opening() -> (Vec<OrderStatusRecord>, Vec<BookDiff>) {
    (
        vec![
            status(START, STATUS_OPEN, false, 10_000, 200, 1),
            status(START + SECOND, STATUS_OPEN, true, 10_100, 300, 2),
        ],
        vec![
            new(1, QrSide::Bid, dec!(100), dec!(2)),
            new(2, QrSide::Ask, dec!(101), dec!(3)),
        ],
    )
}

#[test]
fn test_decode_fixed_follows_the_schemas_example() {
    // SCHEMA.md: 96543.21 is decimals 2 and value 9654321, encoded 0x40935031.
    assert_eq!(decode_fixed(0x4093_5031).unwrap(), dec!(96543.21));
    assert_eq!(decode_fixed(7).unwrap(), dec!(7));
    // Seven decimals, the most three bits hold.
    assert_eq!(decode_fixed((7 << 29) | 12_345).unwrap(), dec!(0.0012345));
}

#[test]
fn test_decode_order_status_reads_each_field_at_its_offset() {
    let decoded = decode_order_status(&record(5, 9, true, 12_345, 67, 89)).unwrap();
    assert_eq!(decoded.timestamp_ns, 5);
    assert_eq!(decoded.user_id, 7);
    assert_eq!(decoded.status_id, 9);
    assert!(decoded.is_ask);
    assert_eq!(decoded.limit_price, dec!(123.45));
    assert_eq!(decoded.size, dec!(0.67));
    assert_eq!(decoded.oid, 89);
    assert_eq!(decoded.original_size, dec!(0.68));
    assert!(
        !decode_order_status(&record(5, 9, false, 1, 1, 1))
            .unwrap()
            .is_ask
    );
}

#[test]
fn test_decoders_refuse_a_broken_length() {
    assert!(decode_order_status(&[0u8; 53]).is_err());
    assert!(decode_order_statuses(&[0u8; 55]).is_err());
    assert!(decode_order_statuses(&[]).unwrap().rows.is_empty());
    let mut two = record(1, 1, false, 1, 1, 1);
    two.extend(record(2, 1, false, 1, 1, 2));
    let decoded = decode_order_statuses(&two).unwrap().rows;
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].timestamp_ns, 2);
}

#[test]
fn test_parse_book_diff_reads_the_schemas_three_types() {
    let (coin, diff) = parse_book_diff(
        r#"{"user":"0x31","oid":273024455731,"coin":"ETH","side":"A","px":"2964.0","raw_book_diff":{"new":{"sz":"0.2274"}}}"#,
    )
    .unwrap();
    assert_eq!(coin, "ETH");
    assert_eq!(diff.oid, 273_024_455_731);
    assert_eq!(diff.side, QrSide::Ask);
    assert_eq!(diff.price, dec!(2964.0));
    assert_eq!(diff.change, BookChange::New { size: dec!(0.2274) });
    let (_, diff) = parse_book_diff(
        r#"{"user":"0x6b","oid":1,"coin":"ETH","side":"B","px":"2962.3","raw_book_diff":"remove"}"#,
    )
    .unwrap();
    assert_eq!(diff.side, QrSide::Bid);
    assert_eq!(diff.change, BookChange::Remove);
    let (_, diff) = parse_book_diff(
        r#"{"user":"0xec","oid":2,"coin":"ETH","side":"B","px":"2962.5","raw_book_diff":{"update":{"origSz":"5.0632","newSz":"4.7287"}}}"#,
    )
    .unwrap();
    assert_eq!(
        diff.change,
        BookChange::Update {
            previous: dec!(5.0632),
            remaining: dec!(4.7287),
        }
    );
    assert!(parse_book_diff("{}").is_err());
    assert!(
        parse_book_diff(r#"{"oid":2,"coin":"ETH","side":"X","px":"1","raw_book_diff":"remove"}"#)
            .is_err()
    );
}

#[test]
fn test_readers_keep_only_their_coin() {
    let diffs = concat!(
        r#"{"oid":1,"coin":"ETH","side":"A","px":"1","raw_book_diff":"remove"}"#,
        "\n",
        r#"{"oid":2,"coin":"BTC","side":"A","px":"1","raw_book_diff":"remove"}"#,
        "\n",
    );
    let read = read_book_diffs(diffs.as_bytes(), "BTC").unwrap().rows;
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].oid, 2);
    let trades = concat!(
        r#"{"coin":"xyz:BTC","side":"A","time":"2025-12-01T10:00:00.5","px":"1","sz":"1"}"#,
        "\n",
        r#"{"coin":"BTC","side":"A","time":"2025-12-01T10:00:00.5","px":"7","sz":"2"}"#,
        "\n",
        r#"{"coin":"BTC","side":"B","time":"2025-12-01T10:00:01","px":"8","sz":"3"}"#,
        "\n",
    );
    let read = read_trades(trades.as_bytes(), "BTC").unwrap().rows;
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].price, dec!(7));
    assert_eq!(read[0].size, dec!(2));
    assert_eq!(read[0].aggressor, TradeSide::Sell);
    assert_eq!(read[1].aggressor, TradeSide::Buy);
    assert_eq!(read[1].timestamp_ns, read[0].timestamp_ns + SECOND / 2);
}

#[test]
fn test_parse_time_ns_by_hand_and_refuses_what_is_no_date() {
    assert_eq!(parse_time_ns("1970-01-01T00:00:00").unwrap(), 0);
    assert_eq!(
        parse_time_ns("1970-01-02T00:00:01.5").unwrap(),
        86_401 * SECOND + SECOND / 2
    );
    // 2025-12-01T10:00:00 UTC is 1764583200 by the order status file's own first record.
    assert_eq!(
        parse_time_ns("2025-12-01T10:00:00.093246718").unwrap(),
        1_764_583_200_093_246_718
    );
    // A leap day exists only in a leap year.
    assert_eq!(
        parse_time_ns("2024-03-01T00:00:00").unwrap()
            - parse_time_ns("2024-02-29T00:00:00").unwrap(),
        86_400 * SECOND
    );
    for bad in [
        "2025-02-29T00:00:00",
        "2025-13-01T00:00:00",
        "2025-00-01T00:00:00",
        "2025-01-00T00:00:00",
        "2025-04-31T00:00:00",
        "2025-01-01T24:00:00",
        "2025-01-01T00:60:00",
        "2025-01-01T00:00:00.1234567890",
        "2025-01-01T00:00:00.x",
        "2025-01-01",
        "x",
    ] {
        assert!(parse_time_ns(bad).is_err(), "{bad}");
    }
}

#[test]
fn test_a_new_diff_takes_the_time_of_its_open_status_marked_as_joined() {
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[1].time.timestamp_ns, START + SECOND);
    assert_eq!(output.events[1].time.evidence, TimeEvidence::Joined);
    assert_eq!(replay.coverage().joined_times, 2);
    assert_eq!(output.events[1].cause, EventCause::Limit);
    assert_eq!(output.events[1].size, dec!(3));
}

#[test]
fn test_with_no_join_a_diff_inherits_a_marked_time_and_is_counted() {
    let (statuses, mut diffs) = opening();
    // No status for these, of each type.
    diffs.push(new(3, QrSide::Bid, dec!(99), dec!(1)));
    diffs.push(update(1, "B", "100", "2", "1"));
    diffs.push(remove(3, "B", "99"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    for event in &output.events[2..] {
        assert_eq!(event.time.timestamp_ns, START + SECOND);
        assert_eq!(event.time.evidence, TimeEvidence::Inherited);
    }
    let coverage = replay.coverage();
    assert_eq!(coverage.unjoined_new, 1);
    assert_eq!(coverage.unjoined_update, 1);
    assert_eq!(coverage.unjoined_remove, 1);
    assert_eq!(coverage.joined_times, 2);
    // A remove with no status has no cause, and an update is a fill by what it is.
    assert_eq!(output.events[3].cause, EventCause::Fill);
    assert_eq!(output.events[4].cause, EventCause::Unknown);
}

#[test]
fn test_a_diff_before_any_time_reaches_no_output_and_is_counted() {
    let mut replay = Replay::new(config(0));
    let output = replay
        .ingest(&[], &[new(1, QrSide::Bid, dec!(100), dec!(1))], &[])
        .unwrap();
    assert!(output.events.is_empty());
    assert_eq!(replay.coverage().diffs_without_time, 1);
}

#[test]
fn test_a_joined_time_that_steps_back_keeps_the_order_and_is_marked() {
    let statuses = vec![
        status(START + SECOND, STATUS_OPEN, false, 10_000, 100, 1),
        status(START, STATUS_OPEN, true, 10_100, 100, 2),
    ];
    let diffs = vec![
        new(1, QrSide::Bid, dec!(100), dec!(1)),
        new(2, QrSide::Ask, dec!(101), dec!(1)),
    ];
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[1].time.timestamp_ns, START + SECOND);
    assert_eq!(output.events[1].time.evidence, TimeEvidence::Inherited);
    assert_eq!(replay.coverage().time_steps_back, 1);
    assert_eq!(replay.coverage().joined_times, 1);
}

#[test]
fn test_mismatches_with_the_open_status_are_counted() {
    let statuses = vec![
        status(START, STATUS_OPEN, true, 10_000, 100, 1),
        status(START, STATUS_OPEN, false, 10_100, 100, 2),
        status(START, STATUS_OPEN, false, 10_000, 500, 3),
    ];
    let diffs = vec![
        new(1, QrSide::Bid, dec!(100), dec!(1)),
        new(2, QrSide::Bid, dec!(100), dec!(1)),
        new(3, QrSide::Bid, dec!(100), dec!(1)),
    ];
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().side_mismatches, 1);
    assert_eq!(replay.coverage().price_mismatches, 1);
    assert_eq!(replay.coverage().size_mismatches, 1);
}

#[test]
fn test_a_remove_takes_its_cause_and_time_from_the_last_status() {
    let (mut statuses, mut diffs) = opening();
    let (canceled, filled) = (START + 2 * SECOND, START + 3 * SECOND);
    statuses.push(status(canceled, CANCELED, false, 10_000, 200, 1));
    statuses.push(status(filled, STATUS_FILLED, true, 10_100, 0, 2));
    diffs.push(remove(1, "B", "100"));
    diffs.push(remove(2, "A", "101"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[2].cause, EventCause::Cancel);
    assert_eq!(output.events[2].time.timestamp_ns, canceled);
    assert_eq!(output.events[2].size, dec!(2));
    assert_eq!(output.events[3].cause, EventCause::Fill);
    assert_eq!(output.events[3].time.timestamp_ns, filled);
    assert_eq!(output.events[3].time.evidence, TimeEvidence::Joined);
}

#[test]
fn test_an_update_joins_the_fill_of_that_remaining_size_in_turn() {
    let (mut statuses, mut diffs) = opening();
    let (first, second) = (START + 2 * SECOND, START + 3 * SECOND);
    statuses.push(status(first, STATUS_FILLED, false, 10_000, 150, 1));
    statuses.push(status(second, STATUS_FILLED, false, 10_000, 100, 1));
    diffs.push(update(1, "B", "100", "2", "1.5"));
    diffs.push(update(1, "B", "100", "1.5", "1"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[2].time.timestamp_ns, first);
    assert_eq!(output.events[2].size, dec!(0.5));
    assert_eq!(output.events[3].time.timestamp_ns, second);
    assert_eq!(output.events[3].cause, EventCause::Fill);
    assert_eq!(output.tops.last().unwrap().value.bid_size, dec!(1));
    assert_eq!(replay.coverage().update_mismatches, 0);
}

#[test]
fn test_an_update_of_another_size_than_known_is_counted_and_the_level_follows_the_order() {
    let (statuses, mut diffs) = opening();
    diffs.push(update(1, "B", "100", "5", "1"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().update_mismatches, 1);
    // The order was known at 2, so the level goes to 1, however the diff states it.
    assert_eq!(output.tops.last().unwrap().value.bid_size, dec!(1));
}

#[test]
fn test_orders_not_seen_are_unknown_and_counted_by_warmth() {
    let (statuses, mut diffs) = opening();
    diffs.insert(1, remove(9, "B", "90"));
    diffs.push(remove(8, "B", "90"));
    diffs.push(update(7, "B", "90", "2", "1"));
    let mut replay = Replay::new(config(SECOND));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().unknown_orders, 3);
    assert_eq!(replay.coverage().unknown_orders_warm, 2);
    // Its size is not known, so nothing is taken from what is known.
    assert_eq!(output.events[1].size, dec!(0));
    assert_eq!(output.tops.last().unwrap().value.bid_size, dec!(2));
}

#[test]
fn test_events_are_cold_until_the_warmup_has_passed() {
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(SECOND));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[0].warmth, Warmth::Cold);
    assert_eq!(output.events[1].warmth, Warmth::Warm);
    let mut later = Replay::new(config(SECOND + 1));
    let output = later.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[1].warmth, Warmth::Cold);
}

#[test]
fn test_the_book_carries_across_hours_and_a_join_in_another_file_inherits() {
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    let output = replay.ingest(&[], &[remove(1, "B", "100")], &[]).unwrap();
    assert_eq!(output.events[0].size, dec!(2));
    assert_eq!(output.events[0].time.evidence, TimeEvidence::Inherited);
    assert_eq!(replay.coverage().diffs, 3);
    assert_eq!(replay.coverage().resting_at_end, 1);
    assert_eq!(replay.coverage().unknown_orders, 0);
}

#[test]
fn test_tops_prices_and_moves_follow_the_touch() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_000,
        100,
        3,
    ));
    statuses.push(status(START + 3 * SECOND, CANCELED, true, 10_100, 300, 2));
    statuses.push(status(
        START + 3 * SECOND,
        STATUS_OPEN,
        true,
        10_300,
        100,
        4,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(4, QrSide::Ask, dec!(103), dec!(1)));
    diffs.push(remove(2, "A", "101"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // One sided at first, then 100 x 101, more size at 100, a deeper ask with no change at the
    // touch, and the ask leaving for 103.
    let tops: Vec<_> = output.tops.iter().map(|t| t.value).collect();
    assert_eq!(tops.len(), 3);
    assert_eq!(tops[0].bid, dec!(100));
    assert_eq!(tops[0].ask, dec!(101));
    assert_eq!(tops[0].bid_size, dec!(2));
    assert_eq!(tops[0].ask_size, dec!(3));
    assert_eq!(tops[0].timestamp_ms, (START + SECOND) / 1_000_000);
    assert_eq!(tops[1].bid_size, dec!(3));
    assert_eq!(tops[2].ask, dec!(103));
    assert_eq!(tops[2].ask_size, dec!(1));
    assert_eq!(output.tops[0].time.evidence, TimeEvidence::Joined);
    let prices: Vec<_> = output.prices.iter().map(|p| p.value.price).collect();
    assert_eq!(prices, vec![dec!(100.5), dec!(101.5)]);
    assert_eq!(output.references.len(), 1);
    assert!(output.references[0].value.moved);
    assert!(!output.references[0].value.reinitialized);
}

#[test]
fn test_a_crossed_book_is_counted_and_gives_no_top() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_200,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(102), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().crossed_events, 1);
    assert_eq!(output.tops.len(), 1);
}

fn trade(ts: u64, price: Decimal, aggressor: TradeSide) -> CaptureTrade {
    let time = (ts / SECOND) as i64;
    let line = format!(
        r#"{{"coin":"BTC","side":"{}","time":"{}","px":"{price}","sz":"1"}}"#,
        match aggressor {
            TradeSide::Buy => "B",
            TradeSide::Sell => "A",
        },
        format_args!("1970-01-01T00:{:02}:{:02}", time / 60, time % 60),
    );
    read_trades(line.as_bytes(), "BTC").unwrap().rows.remove(0)
}

#[test]
fn test_trades_are_checked_against_the_touch_before_their_block() {
    let (mut statuses, mut diffs) = opening();
    // In the block at START + 2 the ask leaves: a buy in that block saw it still there.
    let block = START + 2 * SECOND;
    statuses.push(status(block, STATUS_FILLED, true, 10_100, 0, 2));
    diffs.push(remove(2, "A", "101"));
    let trades = vec![
        trade(START, dec!(100), TradeSide::Sell),
        trade(block, dec!(101), TradeSide::Buy),
        trade(block, dec!(103), TradeSide::Buy),
        trade(block, dec!(100.5), TradeSide::Buy),
        trade(block, dec!(100), TradeSide::Sell),
        trade(block, dec!(99), TradeSide::Sell),
        trade(block, dec!(100.5), TradeSide::Sell),
    ];
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &trades).unwrap();
    let check = replay.coverage().touch_warm;
    assert_eq!(check.no_touch, 1);
    assert_eq!(check.at_touch, 2);
    assert_eq!(check.through_touch, 2);
    assert_eq!(check.better_than_touch, 2);
    assert_eq!(replay.coverage().touch_cold.at_touch, 0);
    let observed = output
        .market_events
        .iter()
        .filter(|e| e.time.evidence == TimeEvidence::Observed)
        .count();
    assert_eq!(observed, trades.len());
    // In time order, with the trades of a block ahead of its later quotes.
    assert!(
        output
            .market_events
            .windows(2)
            .all(|pair| pair[0].time.timestamp_ns <= pair[1].time.timestamp_ns)
    );
    assert!(matches!(
        output.market_events[0].value,
        MarketEvent::Trade(_)
    ));
}

#[test]
fn test_trades_before_the_warmup_are_counted_apart() {
    let (statuses, diffs) = opening();
    let trades = vec![trade(START + SECOND, dec!(101), TradeSide::Buy)];
    let mut replay = Replay::new(config(10 * SECOND));
    replay.ingest(&statuses, &diffs, &trades).unwrap();
    assert_eq!(replay.coverage().touch_cold.no_touch, 1);
    assert_eq!(replay.coverage().touch_warm.no_touch, 0);
}

#[test]
fn test_trades_out_of_order_are_refused() {
    let trades = vec![
        trade(2 * SECOND, dec!(1), TradeSide::Buy),
        trade(SECOND, dec!(1), TradeSide::Buy),
    ];
    let mut replay = Replay::new(config(0));
    replay.ingest(&[], &[], &trades).unwrap();
    assert_eq!(replay.coverage().trades_out_of_order, 1);
}

#[test]
fn test_an_exposure_is_an_orders_life_with_its_fills() {
    let (mut statuses, mut diffs) = opening();
    let (partial, end) = (START + 3 * SECOND, START + 5 * SECOND);
    // Opens two ticks below the mid of 100.5, is partly filled, and its rest is filled.
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        9_800,
        200,
        3,
    ));
    statuses.push(status(partial, STATUS_FILLED, false, 9_800, 100, 3));
    statuses.push(status(end, STATUS_FILLED, false, 9_800, 0, 3));
    diffs.push(new(3, QrSide::Bid, dec!(98), dec!(2)));
    diffs.push(update(3, "B", "98", "2", "1"));
    diffs.push(remove(3, "B", "98"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.exposures.len(), 1);
    let exposure = output.exposures[0].value;
    assert_eq!(exposure.distance, dec!(2.5));
    assert_eq!(exposure.exposure_ms, 3_000);
    assert_eq!(exposure.fills, 2);
    assert_eq!(exposure.timestamp_ms, end / 1_000_000);
}

#[test]
fn test_a_canceled_order_has_no_final_fill_and_a_life_of_no_time_is_left_out() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        9_800,
        100,
        3,
    ));
    statuses.push(status(START + 3 * SECOND, CANCELED, false, 9_800, 100, 3));
    statuses.push(status(
        START + 3 * SECOND,
        STATUS_OPEN,
        false,
        9_700,
        100,
        4,
    ));
    statuses.push(status(START + 3 * SECOND, CANCELED, false, 9_700, 100, 4));
    diffs.push(new(3, QrSide::Bid, dec!(98), dec!(1)));
    diffs.push(remove(3, "B", "98"));
    diffs.push(new(4, QrSide::Bid, dec!(97), dec!(1)));
    diffs.push(remove(4, "B", "97"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.exposures.len(), 1);
    assert_eq!(output.exposures[0].value.fills, 0);
    assert_eq!(replay.coverage().exposures_left_out, 1);
    // The opening's first order has no mid when it opens.
    let remove_first = Replay::new(config(0))
        .ingest(
            &[
                status(START, STATUS_OPEN, false, 10_000, 100, 1),
                status(START + SECOND, CANCELED, false, 10_000, 100, 1),
            ],
            &[
                new(1, QrSide::Bid, dec!(100), dec!(1)),
                remove(1, "B", "100"),
            ],
            &[],
        )
        .unwrap();
    assert!(remove_first.exposures.is_empty());
}

#[test]
fn test_depth_counts_from_the_mid_in_ticks() {
    let (mut statuses, mut diffs) = opening();
    for (oid, price) in [(3u64, 9_900), (4, 9_800), (5, 9_700)] {
        statuses.push(status(
            START + 2 * SECOND,
            STATUS_OPEN,
            false,
            price,
            100,
            oid,
        ));
    }
    diffs.push(new(3, QrSide::Bid, dec!(99), dec!(1)));
    diffs.push(new(4, QrSide::Bid, dec!(98), dec!(1)));
    diffs.push(new(5, QrSide::Bid, dec!(97), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // With no two sided touch there is no depth.
    assert_eq!(output.events[0].depth, None);
    assert_eq!(output.events[0].mid_before, None);
    assert_eq!(output.events[1].depth, None);
    // Mid 100.5: 1.5 ticks is limit 1, 2.5 is limit 2, and 3.5 is past three limits.
    assert_eq!(output.events[2].depth, Some(1));
    assert_eq!(output.events[3].depth, Some(2));
    assert_eq!(output.events[4].depth, None);
    assert_eq!(output.events[2].mid_before, Some(dec!(100.5)));
}

#[test]
fn test_a_wider_touch_has_its_first_tick_at_limit_zero() {
    let statuses = vec![
        status(START, STATUS_OPEN, false, 10_000, 100, 1),
        status(START, STATUS_OPEN, true, 10_400, 100, 2),
        status(START, STATUS_OPEN, false, 10_100, 100, 3),
    ];
    let diffs = vec![
        new(1, QrSide::Bid, dec!(100), dec!(1)),
        new(2, QrSide::Ask, dec!(104), dec!(1)),
        new(3, QrSide::Bid, dec!(101), dec!(1)),
    ];
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // Mid 102, so 101 is one tick away.
    assert_eq!(output.events[2].depth, Some(0));
}

#[test]
fn test_the_price_grid_and_the_declared_tick_are_reported_apart() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        9_950,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(99.5), dec!(1)));
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().off_tick_prices, 1);
    assert_eq!(replay.coverage().price_grid, Some(dec!(0.5)));
}

#[test]
fn test_config_refuses_what_it_cannot_count_in() {
    assert!(ReplayConfig::new(dec!(0), 3, 0, u64::MAX).is_err());
    assert!(ReplayConfig::new(dec!(-1), 3, 0, u64::MAX).is_err());
    assert!(ReplayConfig::new(dec!(1), 0, 0, u64::MAX).is_err());
}

#[test]
fn test_sized_events_take_warm_events_with_a_size_at_a_depth() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        9_900,
        150,
        3,
    ));
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        9_000,
        100,
        4,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(99), dec!(1.5)));
    diffs.push(new(4, QrSide::Bid, dec!(90), dec!(1)));
    diffs.push(remove(9, "B", "90"));
    let mut replay = Replay::new(config(2 * SECOND));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let sized = sized_events(&output.events);
    // Cold, cold, this one, past the limits, and an order of unknown size.
    assert_eq!(sized.len(), 1);
    assert_eq!(sized[0].value.size, dec!(1.5));
    assert_eq!(sized[0].value.depth, 1);
    assert_eq!(
        sized[0].value.timestamp_ms,
        (START + 2 * SECOND) / 1_000_000
    );
    assert_eq!(sized[0].time.evidence, TimeEvidence::Joined);
}

#[test]
fn test_queue_observations_by_hand() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(5), CANCELED, false, 10_000, 100, 3));
    statuses.push(status(t(6), STATUS_FILLED, false, 10_000, 100, 1));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(remove(3, "B", "100"));
    diffs.push(update(1, "B", "100", "2", "1"));
    diffs.push(remove(9, "B", "100"));
    let mut replay = Replay::new(config(SECOND));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = queue_observations(&output.events, &[dec!(0.5), dec!(1), dec!(1)]).unwrap();
    assert_eq!(conversion.cold, 1);
    assert_eq!(conversion.unknown_cause, 1);
    // The ask at START + 1 has no two sided touch when it arrives.
    assert_eq!(conversion.no_depth, 1);
    let got: Vec<_> = conversion.observations.iter().map(|o| o.value).collect();
    assert_eq!(got.len(), 3);
    // Size 2 at 0.5 per event is 4; a limit one second after the mid was first defined, at the
    // ask of START + 1.
    assert_eq!(got[0].kind, QrEventKind::Limit);
    assert_eq!(got[0].queue_before, 4);
    assert_eq!(got[0].side, QrSide::Bid);
    assert_eq!(got[0].depth, 0);
    assert_eq!(got[0].waiting_seconds, 1.0);
    assert_eq!(got[0].timestamp_ms, t(2) / 1_000_000);
    assert_eq!(got[1].kind, QrEventKind::Cancel);
    assert_eq!(got[1].queue_before, 6);
    assert_eq!(got[1].waiting_seconds, 3.0);
    assert_eq!(got[2].kind, QrEventKind::Market);
    assert_eq!(got[2].queue_before, 4);
    assert_eq!(got[2].waiting_seconds, 1.0);
}

#[test]
fn test_a_mid_move_restarts_the_waiting_times() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(4), STATUS_OPEN, true, 10_050, 100, 4));
    statuses.push(status(t(7), STATUS_OPEN, false, 10_000, 100, 5));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(4, QrSide::Ask, dec!(100.5), dec!(1)));
    diffs.push(new(5, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = queue_observations(&output.events, &[dec!(1); 3]).unwrap();
    let last = conversion.observations.last().unwrap().value;
    // The move at 4 restarted the queue's clock, so 7 - 4 and not 7 - 2.
    assert_eq!(last.waiting_seconds, 3.0);
}

#[test]
fn test_queue_observations_refuse_sizes_that_cannot_count() {
    assert!(queue_observations(&[], &[dec!(0)]).is_err());
    assert!(queue_observations(&[], &[dec!(-1)]).is_err());
    let (statuses, diffs) = opening();
    let mut extended = diffs.clone();
    extended.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &extended, &[]).unwrap();
    // No size for limit 0.
    assert!(queue_observations(&output.events, &[]).is_err());
}

#[test]
fn test_hawkes_segments_by_hand() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(3), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(5), CANCELED, false, 10_000, 100, 3));
    statuses.push(status(t(6), STATUS_OPEN, true, 10_050, 100, 4));
    statuses.push(status(t(9), STATUS_OPEN, false, 10_000, 100, 5));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(remove(3, "B", "100"));
    diffs.push(new(4, QrSide::Ask, dec!(100.5), dec!(1)));
    diffs.push(new(5, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = hawkes_segments(&output.events, QrSide::Bid, dec!(1)).unwrap();
    // From the first event kept, at t(3), with 2 before it, to the ask at t(6) that moves the mid.
    assert_eq!(conversion.segments.len(), 1);
    let record = &conversion.segments[0];
    assert_eq!(record.inherited_events, 0);
    assert_eq!(record.segment.initial_queue, 2);
    assert_eq!(record.segment.duration_seconds, 3.0);
    assert_eq!(record.segment.events.len(), 2);
    assert_eq!(record.segment.events[0].time_seconds, 0.0);
    assert_eq!(record.segment.events[0].kind, QrEventKind::Limit);
    assert_eq!(record.segment.events[1].time_seconds, 2.0);
    assert_eq!(record.segment.events[1].kind, QrEventKind::Cancel);
    // The other side's only event at its first limit is the ask that moved the mid, so a
    // period of no length.
    let asks = hawkes_segments(&output.events, QrSide::Ask, dec!(1)).unwrap();
    assert!(asks.segments.is_empty());
    assert_eq!(asks.zero_length, 1);
    // From a queue of 0 in units this large, the limit is what lets the cancel leave.
    let large = hawkes_segments(&output.events, QrSide::Bid, dec!(1000)).unwrap();
    assert_eq!(large.inconsistent, 0);
    assert_eq!(large.segments[0].segment.initial_queue, 0);
    assert!(hawkes_segments(&[], QrSide::Bid, dec!(0)).is_err());
}

#[test]
fn test_hawkes_segments_count_what_they_leave_out_and_what_is_inherited() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(3), STATUS_OPEN, false, 10_000, 100, 8));
    statuses.push(status(t(6), STATUS_OPEN, true, 10_050, 100, 4));
    // A joined limit, then an inherited limit, a remove of no cause, and the move.
    diffs.push(new(8, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(remove(3, "B", "100"));
    diffs.push(new(4, QrSide::Ask, dec!(100.5), dec!(1)));
    diffs.push(new(9, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = hawkes_segments(&output.events, QrSide::Bid, dec!(1)).unwrap();
    assert_eq!(conversion.unknown_cause, 1);
    assert_eq!(conversion.segments.len(), 1);
    assert_eq!(conversion.segments[0].inherited_events, 1);
    let mut cold = Replay::new(config(100 * SECOND));
    let output = cold.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = hawkes_segments(&output.events, QrSide::Bid, dec!(1)).unwrap();
    assert_eq!(conversion.cold, 4);
    assert!(conversion.segments.is_empty());
}

#[test]
fn test_a_segment_departing_from_an_empty_queue_is_counted_and_left_out() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(3), CANCELED, false, 10_000, 100, 3));
    statuses.push(status(t(5), STATUS_OPEN, true, 10_050, 100, 4));
    statuses.push(status(t(6), STATUS_OPEN, false, 10_000, 100, 5));
    diffs.push(remove(3, "B", "100"));
    diffs.insert(2, new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(4, QrSide::Ask, dec!(100.5), dec!(1)));
    diffs.push(new(5, QrSide::Bid, dec!(100), dec!(1)));
    // Warm after t(2), so that the first event kept is the cancel.
    let mut replay = Replay::new(config(2 * SECOND + 1));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // At so large an event size the queue before the first event is 0, and then a cancel.
    let wide = hawkes_segments(&output.events, QrSide::Bid, dec!(1000)).unwrap();
    assert_eq!(wide.inconsistent, 1);
    assert!(wide.segments.is_empty());
    let fine = hawkes_segments(&output.events, QrSide::Bid, dec!(1)).unwrap();
    assert_eq!(fine.inconsistent, 0);
    assert_eq!(fine.segments.len(), 1);
    assert_eq!(fine.segments[0].segment.initial_queue, 3);
}

#[test]
fn test_readers_read_the_coin_field_and_not_the_marker() {
    // The marker in a nested object is not the coin.
    let diffs = concat!(
        r#"{"meta":{"coin":"BTC"},"oid":3,"coin":"ETH","side":"A","px":"1","raw_book_diff":"remove"}"#,
        "\n",
    );
    assert!(
        read_book_diffs(diffs.as_bytes(), "BTC")
            .unwrap()
            .rows
            .is_empty()
    );
    let trades = concat!(
        r#"{"meta":{"coin":"BTC"},"coin":"ETH","side":"A","time":"2025-12-01T10:00:00","px":"1","sz":"1"}"#,
        "\n",
    );
    assert!(
        read_trades(trades.as_bytes(), "BTC")
            .unwrap()
            .rows
            .is_empty()
    );
}

#[test]
fn test_readers_take_a_valid_line_however_it_is_spaced() {
    let diffs = concat!(
        r#"{"oid": 3, "coin": "BTC", "side": "A", "px": "1", "raw_book_diff": "remove"}"#,
        "\n",
    );
    assert_eq!(
        read_book_diffs(diffs.as_bytes(), "BTC").unwrap().rows.len(),
        1
    );
    let trades = concat!(
        r#"{"coin": "BTC", "side": "A", "time": "2025-12-01T10:00:00", "px": "1", "sz": "1"}"#,
        "\n",
    );
    assert_eq!(read_trades(trades.as_bytes(), "BTC").unwrap().rows.len(), 1);
}

#[test]
fn test_readers_refuse_a_broken_line_of_any_coin() {
    let diffs = concat!(
        r#"{"oid":3,"coin":"ETH","side":"X","px":"1","raw_book_diff":"remove"}"#,
        "\n",
    );
    assert!(read_book_diffs(diffs.as_bytes(), "BTC").is_err());
    assert!(read_book_diffs("not json\n".as_bytes(), "BTC").is_err());
    assert!(read_trades("not json\n".as_bytes(), "BTC").is_err());
}

#[test]
fn test_a_century_is_a_leap_year_only_each_400_years() {
    assert!(parse_time_ns("2000-02-29T00:00:00").is_ok());
    assert!(parse_time_ns("1900-02-29T00:00:00").is_err());
}

#[test]
fn test_a_second_open_status_does_not_move_the_join() {
    let statuses = vec![
        status(START, STATUS_OPEN, false, 10_000, 100, 1),
        status(START + SECOND, STATUS_OPEN, false, 10_000, 100, 1),
    ];
    let mut replay = Replay::new(config(0));
    let output = replay
        .ingest(&statuses, &[new(1, QrSide::Bid, dec!(100), dec!(1))], &[])
        .unwrap();
    assert_eq!(output.events[0].time.timestamp_ns, START);
}

#[test]
fn test_two_fills_to_one_size_each_take_their_own_status() {
    let (mut statuses, mut diffs) = opening();
    let (first, second) = (START + 2 * SECOND, START + 3 * SECOND);
    statuses.push(status(first, STATUS_FILLED, false, 10_000, 150, 1));
    statuses.push(status(second, STATUS_FILLED, false, 10_000, 150, 1));
    diffs.push(update(1, "B", "100", "2", "1.5"));
    diffs.push(update(1, "B", "100", "1.5", "1.5"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[2].time.timestamp_ns, first);
    assert_eq!(output.events[3].time.timestamp_ns, second);
}

#[test]
fn test_a_locked_book_is_counted_and_gives_no_top() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_100,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(101), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().crossed_events, 1);
    assert_eq!(output.tops.len(), 1);
}

#[test]
fn test_a_sell_under_the_bid_is_through_it_and_one_over_is_better() {
    let (mut statuses, mut diffs) = opening();
    let later = START + 2 * SECOND;
    statuses.push(status(later, STATUS_OPEN, false, 9_000, 100, 3));
    diffs.push(new(3, QrSide::Bid, dec!(90), dec!(1)));
    let check = |price| {
        let mut replay = Replay::new(config(0));
        replay
            .ingest(&statuses, &diffs, &[trade(later, price, TradeSide::Sell)])
            .unwrap();
        replay.coverage().touch_warm
    };
    assert_eq!(check(dec!(99)).through_touch, 1);
    assert_eq!(check(dec!(99)).better_than_touch, 0);
    assert_eq!(check(dec!(100.5)).better_than_touch, 1);
    assert_eq!(check(dec!(100.5)).through_touch, 0);
}

#[test]
fn test_sized_events_leave_out_a_remove_of_unknown_cause() {
    // An order known by its new diff, whose remove has no status: its size is known, and its
    // cause is not.
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_000,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(remove(3, "B", "100"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[3].cause, EventCause::Unknown);
    assert_eq!(output.events[3].size, dec!(1));
    assert_eq!(sized_events(&output.events).len(), 1);
}

#[test]
fn test_a_queue_rounds_to_the_nearest_event_size() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_000,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // 2 before it, at 0.8 per event, is 2.5, and half goes to even; at 0.75 it is 2.67, so 3.
    let queue = |size| {
        queue_observations(&output.events, &[size, dec!(1), dec!(1)])
            .unwrap()
            .observations[0]
            .value
            .queue_before
    };
    assert_eq!(queue(dec!(0.8)), 2);
    assert_eq!(queue(dec!(0.75)), 3);
}

#[test]
fn test_an_event_of_no_known_cause_still_moves_its_queues_clock() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(5), STATUS_OPEN, false, 10_000, 100, 4));
    statuses.push(status(t(5), STATUS_OPEN, false, 10_000, 100, 5));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    // No status, so an inherited time, and no cause.
    diffs.push(remove(3, "B", "100"));
    diffs.push(new(4, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(5, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = queue_observations(&output.events, &[dec!(1); 3]).unwrap();
    assert_eq!(conversion.unknown_cause, 1);
    let got = &conversion.observations;
    // From the event at t(2), as the remove inherited that time.
    assert_eq!(got[1].value.waiting_seconds, 3.0);
    // Built on an inherited start, so inherited.
    assert_eq!(got[1].time.evidence, TimeEvidence::Inherited);
    // One block, so no waiting time, and it is counted.
    assert_eq!(got[2].value.waiting_seconds, 0.0);
    assert_eq!(got[2].time.evidence, TimeEvidence::Joined);
    assert_eq!(conversion.zero_waits, 1);
}

#[test]
fn test_a_cold_event_still_moves_its_queues_clock() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_000, 100, 3));
    statuses.push(status(t(5), STATUS_OPEN, false, 10_000, 100, 4));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(new(4, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = Replay::new(config(3 * SECOND));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let conversion = queue_observations(&output.events, &[dec!(1); 3]).unwrap();
    assert_eq!(conversion.observations.len(), 1);
    assert_eq!(conversion.observations[0].value.waiting_seconds, 3.0);
}

#[test]
fn test_an_exposure_on_an_inherited_open_is_inherited() {
    let (mut statuses, mut diffs) = opening();
    let end = START + 5 * SECOND;
    statuses.push(status(end, CANCELED, false, 9_800, 100, 3));
    // No open status, so its opening time is inherited.
    diffs.push(new(3, QrSide::Bid, dec!(98), dec!(1)));
    diffs.push(remove(3, "B", "98"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[3].time.evidence, TimeEvidence::Joined);
    assert_eq!(output.exposures[0].time.evidence, TimeEvidence::Inherited);
}

#[test]
fn test_a_trade_after_an_inherited_diff_is_counted_apart() {
    let (mut statuses, mut diffs) = opening();
    let block = START + 2 * SECOND;
    statuses.push(status(block + SECOND, STATUS_OPEN, false, 9_000, 100, 8));
    // A remove with no status inherits an earlier time, and may belong to the block.
    diffs.push(remove(2, "A", "101"));
    diffs.push(new(8, QrSide::Bid, dec!(90), dec!(1)));
    let trades = vec![trade(block, dec!(101), TradeSide::Buy)];
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &trades).unwrap();
    let check = replay.coverage().touch_warm;
    assert_eq!(check.after_inherited, 1);
    assert_eq!(check.no_touch, 0);
    // With a joined diff between, what follows it is checked again.
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(block, STATUS_OPEN, false, 9_000, 100, 8));
    statuses.push(status(block + SECOND, STATUS_OPEN, false, 8_000, 100, 9));
    diffs.push(new(7, QrSide::Bid, dec!(70), dec!(1)));
    diffs.push(new(8, QrSide::Bid, dec!(90), dec!(1)));
    diffs.push(new(9, QrSide::Bid, dec!(80), dec!(1)));
    let trades = vec![trade(block + SECOND, dec!(101), TradeSide::Buy)];
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &trades).unwrap();
    let check = replay.coverage().touch_warm;
    assert_eq!(check.after_inherited, 0);
    assert_eq!(check.at_touch, 1);
}

#[test]
fn test_a_crossed_book_gives_no_mid_and_no_depth_to_later_events() {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(2), STATUS_OPEN, false, 10_200, 100, 3));
    statuses.push(status(t(3), STATUS_OPEN, false, 9_900, 100, 4));
    diffs.push(new(3, QrSide::Bid, dec!(102), dec!(1)));
    diffs.push(new(4, QrSide::Bid, dec!(99), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[3].mid_before, None);
    assert_eq!(output.events[3].depth, None);
    assert_eq!(replay.coverage().crossed_events, 2);
    let trades = vec![trade(t(4), dec!(101), TradeSide::Buy)];
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &trades).unwrap();
    assert_eq!(replay.coverage().touch_warm.no_touch, 1);
}

#[test]
fn test_a_partial_fill_as_the_last_status_is_no_end_of_the_order() {
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_FILLED,
        false,
        10_000,
        100,
        1,
    ));
    // Its cancel is in a file not given.
    diffs.push(remove(1, "B", "100"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[2].cause, EventCause::Unknown);
    assert_eq!(output.events[2].time.evidence, TimeEvidence::Inherited);
    assert_eq!(replay.coverage().partial_last_status, 1);
    assert_eq!(replay.coverage().unjoined_remove, 1);
    // A cancel after a partial fill ends it as a cancel.
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_FILLED,
        false,
        10_000,
        100,
        1,
    ));
    statuses.push(status(START + 3 * SECOND, CANCELED, false, 10_000, 100, 1));
    diffs.push(remove(1, "B", "100"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(output.events[2].cause, EventCause::Cancel);
    assert_eq!(replay.coverage().partial_last_status, 0);
}

#[test]
fn test_a_duplicate_new_replaces_the_order_and_is_counted() {
    let (mut statuses, mut diffs) = opening();
    diffs.push(new(1, QrSide::Bid, dec!(100), dec!(5)));
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        true,
        10_300,
        100,
        3,
    ));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().duplicate_new, 1);
    // 5 and not 7.
    assert_eq!(output.tops.last().unwrap().value.bid_size, dec!(5));
}

#[test]
fn test_a_diff_of_another_side_or_price_uses_the_known_order_and_is_counted() {
    let (statuses, mut diffs) = opening();
    diffs.push(update(1, "B", "99", "2", "1"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().order_mismatches, 1);
    assert_eq!(output.tops.last().unwrap().value.bid_size, dec!(1));
    let (statuses, mut diffs) = opening();
    diffs.push(remove(1, "B", "99"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().order_mismatches, 1);
    assert_eq!(output.events[2].size, dec!(2));
    // The order's own price loses the size, not the diff's: 100 keeps what else rests there.
    let (mut statuses, mut diffs) = opening();
    statuses.push(status(
        START + 2 * SECOND,
        STATUS_OPEN,
        false,
        10_000,
        100,
        3,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    diffs.push(remove(1, "B", "99"));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let last = output.tops.last().unwrap().value;
    assert_eq!(last.bid, dec!(100));
    assert_eq!(last.bid_size, dec!(1));
}

#[test]
fn test_the_longest_gap_shows_a_missing_hour() {
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().longest_gap_ns, SECOND);
    let later = START + 3_600 * SECOND;
    let more = [status(later, CANCELED, false, 10_000, 200, 1)];
    replay.ingest(&more, &[remove(1, "B", "100")], &[]).unwrap();
    assert_eq!(replay.coverage().longest_gap_ns, later - (START + SECOND));
}

#[test]
fn test_parse_time_ns_refuses_extra_parts_and_signs() {
    for bad in [
        "2025-12-01-99T10:00:00",
        "2025-12-01T10:00:00:77",
        "2025-12-01T10:00:00.",
        "2025-12-01T10:00:+0",
        "+2025-12-01T10:00:00",
        "2025-12-01T10:00:60",
    ] {
        assert!(parse_time_ns(bad).is_err(), "{bad}");
    }
}

#[test]
fn test_a_level_going_below_zero_is_counted() {
    // No consistent stream does this: a remaining size below zero is malformed, and this is the
    // count that would show a size known wrongly.
    let (statuses, mut diffs) = opening();
    let negative = update(1, "B", "100", "2", "-1");
    diffs.push(negative);
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().negative_levels, 1);
    // A level that reaches exactly zero is no breach.
    let (statuses, mut diffs) = opening();
    diffs.push(remove(1, "B", "100"));
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(replay.coverage().negative_levels, 0);
}

#[test]
fn test_decode_fixed_keeps_bit_28_as_part_of_the_value() {
    // The sign bit of the dataset's other, signed encoding; here it is a value bit.
    assert_eq!(decode_fixed(1 << 28).unwrap(), Decimal::from(1u32 << 28));
    assert_eq!(
        decode_fixed((1 << 29) | (1 << 28)).unwrap(),
        dec!(26843545.6)
    );
}

const GAP: u64 = 60 * SECOND;

fn resetting(warmup: u64) -> Replay {
    Replay::new(ReplayConfig::new(dec!(1), 3, warmup, GAP).unwrap())
}

#[test]
fn test_a_silence_in_the_diffs_empties_the_book_and_restarts_the_warmup() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(10 * SECOND);
    let first = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert_eq!(first.events.last().unwrap().epoch, 0);

    // Past the warm-up, and after the gap: the book it remembered is gone.
    let later = START + SECOND + GAP + 1;
    let more = [status(later, STATUS_OPEN, false, 9_900, 100, 3)];
    let output = replay
        .ingest(&more, &[new(3, QrSide::Bid, dec!(99), dec!(1))], &[])
        .unwrap();
    let event = output.events[0];
    assert_eq!(event.epoch, 1);
    assert_eq!(event.warmth, Warmth::Cold);
    assert_eq!(event.level_before, dec!(0));
    assert_eq!(event.mid_before, None);
    assert!(output.tops.is_empty(), "one side left");
    let coverage = replay.coverage();
    assert_eq!(coverage.resets_on_diffs, 1);
    assert_eq!(coverage.resets_on_trades, 0);
    assert_eq!(coverage.orders_cut_by_reset, 2);

    // An order from before the gap is now unknown, and reaches no exposure.
    let after = later + SECOND;
    let gone = [status(after, CANCELED, false, 10_000, 200, 1)];
    let output = replay.ingest(&gone, &[remove(1, "B", "100")], &[]).unwrap();
    assert!(output.exposures.is_empty());
    assert_eq!(replay.coverage().unknown_orders, 1);

    // The warm-up counts from the reset, not from the start.
    let warm = later + 10 * SECOND;
    let last = [status(warm, STATUS_OPEN, false, 9_900, 100, 4)];
    let output = replay
        .ingest(&last, &[new(4, QrSide::Bid, dec!(99), dec!(1))], &[])
        .unwrap();
    assert_eq!(output.events[0].warmth, Warmth::Warm);
    assert_eq!(output.events[0].level_before, dec!(1));
}

#[test]
fn test_a_silence_of_exactly_the_limit_does_not_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    let later = START + SECOND + GAP;
    let more = [status(later, STATUS_OPEN, false, 10_000, 100, 3)];
    let output = replay
        .ingest(&more, &[new(3, QrSide::Bid, dec!(100), dec!(1))], &[])
        .unwrap();
    assert_eq!(output.events[0].epoch, 0);
    assert_eq!(output.events[0].level_before, dec!(2));
    assert_eq!(replay.coverage().resets_on_diffs, 0);
    assert_eq!(replay.coverage().longest_gap_ns, GAP);
}

#[test]
fn test_a_silence_in_the_trades_resets_across_files_with_the_diffs_flowing() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let first = [trade(START, dec!(100), TradeSide::Sell)];
    replay.ingest(&statuses, &diffs, &first).unwrap();
    assert_eq!(replay.coverage().resets_on_trades, 0);

    // The diffs never stop, so only the trades can show this one.
    let mut opened = Vec::new();
    let mut steps = Vec::new();
    for step in 1..=3u64 {
        let at = START + SECOND + step * 30 * SECOND;
        opened.push(status(at, STATUS_OPEN, false, 9_900, 100, 10 + step));
        steps.push(new(10 + step, QrSide::Bid, dec!(99), dec!(1)));
    }
    let late = START + SECOND + 90 * SECOND;
    let output = replay
        .ingest(&opened, &steps, &[trade(late, dec!(100), TradeSide::Sell)])
        .unwrap();
    let coverage = replay.coverage();
    assert_eq!(coverage.resets_on_trades, 1);
    assert_eq!(coverage.resets_on_diffs, 0);
    assert_eq!(coverage.longest_trade_gap_ns, late - START);
    assert_eq!(coverage.orders_cut_by_reset, 4);
    assert_eq!(output.events[0].epoch, 0);
    assert_eq!(output.events[2].epoch, 1);
    assert_eq!(output.events[2].level_before, dec!(0));
}

#[test]
fn test_silences_in_both_streams_keep_the_latest_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(10 * SECOND);
    let first = [trade(START, dec!(100), TradeSide::Sell)];
    replay.ingest(&statuses, &diffs, &first).unwrap();

    // The trade at 70 is older than the diff at 101: it comes first, and the diff restarts the
    // warm-up, as no book has been rebuilt until then.
    let at = START + 101 * SECOND;
    let more = [status(at, STATUS_OPEN, false, 10_000, 100, 3)];
    let trades = [trade(START + 70 * SECOND, dec!(100), TradeSide::Sell)];
    let output = replay
        .ingest(&more, &[new(3, QrSide::Bid, dec!(100), dec!(1))], &trades)
        .unwrap();
    assert_eq!(output.events[0].warmth, Warmth::Cold);
    let coverage = replay.coverage();
    assert_eq!(coverage.resets_on_trades, 1);
    assert_eq!(coverage.resets_on_diffs, 1);
    assert_eq!(coverage.orders_cut_by_reset, 2);
    assert_eq!(
        output.events[0].epoch, 1,
        "one silence, one reset of the book"
    );

    let warm = at + 10 * SECOND;
    let last = [status(warm, STATUS_OPEN, false, 10_000, 100, 4)];
    let output = replay
        .ingest(&last, &[new(4, QrSide::Bid, dec!(100), dec!(1))], &[])
        .unwrap();
    assert_eq!(output.events[0].warmth, Warmth::Warm);
    assert_eq!(output.events[0].epoch, 1);
}

#[test]
fn test_a_reset_does_not_reach_the_trades_before_it() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    // Within the limit of the last diff, so the touch still stands for it.
    let trades = [trade(START + 30 * SECOND, dec!(101), TradeSide::Buy)];
    let at = START + 101 * SECOND;
    let more = [status(at, STATUS_OPEN, false, 10_000, 100, 3)];
    replay
        .ingest(&more, &[new(3, QrSide::Bid, dec!(100), dec!(1))], &trades)
        .unwrap();
    let coverage = replay.coverage();
    assert_eq!(coverage.touch_warm.at_touch, 1);
    assert_eq!(coverage.resets_on_diffs, 1);
    assert_eq!(coverage.resets_on_trades, 0);
}

#[test]
fn test_a_trade_after_the_diffs_go_silent_is_not_read_against_the_stale_book() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    // No gap between trades here, but the diffs stopped at 1 and a trade at 70 shows it.
    let trades = [
        trade(START + 30 * SECOND, dec!(101), TradeSide::Buy),
        trade(START + 70 * SECOND, dec!(101), TradeSide::Buy),
        trade(START + 80 * SECOND, dec!(101), TradeSide::Buy),
    ];
    let at = START + 101 * SECOND;
    let more = [status(at, STATUS_OPEN, false, 10_000, 100, 3)];
    let output = replay
        .ingest(&more, &[new(3, QrSide::Bid, dec!(100), dec!(1))], &trades)
        .unwrap();
    let coverage = replay.coverage();
    assert_eq!(coverage.touch_warm.at_touch, 1, "only the trade at 30");
    assert_eq!(coverage.resets_on_diffs, 1, "one silence, shown three ways");
    assert_eq!(coverage.resets_on_trades, 0);
    assert_eq!(output.events[0].level_before, dec!(0));
    assert_eq!(output.events[0].epoch, 1);
}

#[test]
fn test_a_silence_of_exactly_the_limit_in_the_trades_does_not_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let trades = [
        trade(START, dec!(100), TradeSide::Sell),
        trade(START + GAP, dec!(100), TradeSide::Sell),
    ];
    replay.ingest(&statuses, &diffs, &trades).unwrap();
    assert_eq!(replay.coverage().resets_on_trades, 0);
    assert_eq!(replay.coverage().longest_trade_gap_ns, GAP);
}

#[test]
fn test_no_move_of_the_mid_is_reported_across_a_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    let t = |seconds: u64| START + SECOND + GAP + seconds * SECOND;
    let more = [
        status(t(1), STATUS_OPEN, false, 20_000, 100, 3),
        status(t(2), STATUS_OPEN, true, 20_100, 100, 4),
    ];
    let output = replay
        .ingest(
            &more,
            &[
                new(3, QrSide::Bid, dec!(200), dec!(1)),
                new(4, QrSide::Ask, dec!(201), dec!(1)),
            ],
            &[],
        )
        .unwrap();
    assert_eq!(output.tops.len(), 1);
    assert_eq!(output.prices.len(), 1);
    assert!(output.references.is_empty(), "a new book, not a move");
}

/// Events on both sides of a gap, at the first limit of the bid, with a mid on either side.
fn events_across_a_gap() -> Vec<market_maker_rs::backtest::hyperliquid::BookEvent> {
    let (mut statuses, mut diffs) = opening();
    let t = |seconds: u64| START + seconds * SECOND;
    statuses.push(status(t(3), STATUS_OPEN, false, 10_000, 100, 3));
    diffs.push(new(3, QrSide::Bid, dec!(100), dec!(1)));
    let mut replay = resetting(0);
    let mut events = replay.ingest(&statuses, &diffs, &[]).unwrap().events;
    let l = |seconds: u64| t(1_000) + seconds * SECOND;
    let more = [
        status(l(0), STATUS_OPEN, false, 10_000, 100, 5),
        status(l(1), STATUS_OPEN, true, 10_100, 100, 6),
        status(l(4), STATUS_OPEN, false, 10_000, 100, 7),
    ];
    let later = [
        new(5, QrSide::Bid, dec!(100), dec!(1)),
        new(6, QrSide::Ask, dec!(101), dec!(1)),
        new(7, QrSide::Bid, dec!(100), dec!(1)),
    ];
    events.extend(replay.ingest(&more, &later, &[]).unwrap().events);
    events
}

#[test]
fn test_no_waiting_time_spans_a_reset() {
    let conversion = queue_observations(&events_across_a_gap(), &[dec!(1); 3]).unwrap();
    for observation in &conversion.observations {
        assert!(
            observation.value.waiting_seconds < 100.0,
            "{:?}",
            observation.value
        );
    }
}

#[test]
fn test_waiting_times_start_again_after_a_reset() {
    let joined = queue_observations(&events_across_a_gap(), &[dec!(1); 3]).unwrap();
    // The ask at 1 made the new book two sided, so 4 - 1, with nothing from before the gap.
    let last = joined.observations.last().unwrap().value;
    assert_eq!(last.waiting_seconds, 3.0);
}

#[test]
fn test_the_same_touch_after_a_reset_is_a_new_top() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    assert_eq!(replay.ingest(&statuses, &diffs, &[]).unwrap().tops.len(), 1);
    let t = |seconds: u64| START + SECOND + GAP + seconds * SECOND;
    let more = [
        status(t(1), STATUS_OPEN, false, 10_000, 200, 3),
        status(t(2), STATUS_OPEN, true, 10_100, 300, 4),
    ];
    let output = replay
        .ingest(
            &more,
            &[
                new(3, QrSide::Bid, dec!(100), dec!(2)),
                new(4, QrSide::Ask, dec!(101), dec!(3)),
            ],
            &[],
        )
        .unwrap();
    assert_eq!(output.tops.len(), 1);
    assert_eq!(output.prices.len(), 1);
}

#[test]
fn test_a_period_a_reset_cuts_is_left_out_of_the_segments() {
    let conversion = hawkes_segments(&events_across_a_gap(), QrSide::Bid, dec!(1)).unwrap();
    assert_eq!(conversion.cut_by_reset, 1);
    assert!(
        conversion.segments.is_empty(),
        "no move ended either period"
    );
}

#[test]
fn test_a_reset_with_no_open_period_cuts_none() {
    let events: Vec<_> = events_across_a_gap()
        .into_iter()
        .filter(|event| event.epoch == 1)
        .collect();
    let conversion = hawkes_segments(&events, QrSide::Bid, dec!(1)).unwrap();
    assert_eq!(conversion.cut_by_reset, 0);
}

#[test]
fn test_a_re_emitted_stretch_of_diffs_is_left_out_and_counted() {
    let line = |oid: u64, change: &str| {
        format!(
            r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"B","px":"100","raw_book_diff":{change}}}"#
        )
    };
    let added = line(1, r#"{"new":{"sz":"2"}}"#);
    let other = line(2, r#"{"new":{"sz":"3"}}"#);
    let gone = line(1, r#""remove""#);
    let eth = added.replace("BTC", "ETH");
    let stream = [
        added.as_str(),
        other.as_str(),
        eth.as_str(),
        added.as_str(),
        other.as_str(),
        eth.as_str(),
        gone.as_str(),
    ]
    .join("\n");
    let read = read_book_diffs(stream.as_bytes(), "BTC").unwrap();
    assert_eq!(read.rows.len(), 3);
    assert_eq!(read.repeated_lines, 2);
    assert_eq!(read.rows[2].oid, 1);
    // A new reader for a new file starts again.
    assert_eq!(
        read_book_diffs(added.as_bytes(), "BTC")
            .unwrap()
            .repeated_lines,
        0
    );
}

#[test]
fn test_lines_that_agree_on_fields_are_not_taken_for_a_repeat() {
    let line = |hash: &str| {
        format!(
            r#"{{"coin":"BTC","side":"B","time":"2025-12-30T18:00:00.5","px":"100","sz":"1","hash":"{hash}"}}"#
        )
    };
    let stream = [line("0x1"), line("0x2"), line("0x1")].join("\n");
    let read = read_trades(stream.as_bytes(), "BTC").unwrap();
    assert_eq!(read.rows.len(), 2);
    assert_eq!(read.repeated_lines, 1);
}

#[test]
fn test_a_re_emitted_stretch_steps_back_nothing_once_read() {
    // A stretch written twice, with the times stepping back between, as the capture has it.
    let line = |second: u32, hash: &str| {
        format!(
            r#"{{"coin":"BTC","side":"B","time":"2025-12-30T18:00:{second:02}","px":"101","sz":"1","hash":"{hash}"}}"#
        )
    };
    let stream = [
        line(1, "a"),
        line(2, "b"),
        line(3, "c"),
        line(2, "b"),
        line(3, "c"),
        line(4, "d"),
    ]
    .join("\n");
    let read = read_trades(stream.as_bytes(), "BTC").unwrap();
    assert_eq!(read.repeated_lines, 2);
    assert_eq!(read.rows.len(), 4);
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(0));
    replay.ingest(&statuses, &diffs, &read.rows).unwrap();
    assert_eq!(replay.coverage().trades_out_of_order, 0);
    assert_eq!(replay.coverage().trades, 4);
}

#[test]
fn test_a_trade_before_the_last_is_skipped_and_counted_across_files() {
    let (statuses, diffs) = opening();
    let mut replay = Replay::new(config(0));
    let t = |seconds: u64| START + seconds * SECOND;
    let first = [
        trade(t(1), dec!(101), TradeSide::Buy),
        trade(t(5), dec!(101), TradeSide::Buy),
        trade(t(3), dec!(101), TradeSide::Buy),
        trade(t(5), dec!(101), TradeSide::Buy),
    ];
    replay.ingest(&statuses, &diffs, &first).unwrap();
    let coverage = replay.coverage();
    assert_eq!(coverage.trades_out_of_order, 1);
    assert_eq!(
        coverage.touch_warm.at_touch + coverage.touch_warm.no_touch,
        3,
        "a trade at the same time as the last is kept, and the late one reaches no check"
    );
    replay
        .ingest(&[], &[], &[trade(t(4), dec!(101), TradeSide::Buy)])
        .unwrap();
    assert_eq!(replay.coverage().trades_out_of_order, 2);
    assert_eq!(replay.coverage().longest_trade_gap_ns, 4 * SECOND);
}

#[test]
fn test_every_output_says_which_side_of_a_reset_it_is_on() {
    let t = |seconds: u64| START + SECOND + GAP + seconds * SECOND;
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let first = [trade(START + SECOND, dec!(101), TradeSide::Buy)];
    let before = replay.ingest(&statuses, &diffs, &first).unwrap();
    assert!(before.tops.iter().all(|top| top.epoch == 0));
    assert!(before.prices.iter().all(|price| price.epoch == 0));
    assert!(before.market_events.iter().all(|event| event.epoch == 0));
    assert!(!before.tops.is_empty() && !before.prices.is_empty());

    let more = [
        status(t(1), STATUS_OPEN, false, 10_000, 200, 3),
        status(t(2), STATUS_OPEN, true, 10_100, 300, 4),
        status(t(3), STATUS_OPEN, true, 10_050, 100, 5),
        status(t(4), CANCELED, true, 10_050, 100, 5),
    ];
    let after = replay
        .ingest(
            &more,
            &[
                new(3, QrSide::Bid, dec!(100), dec!(2)),
                new(4, QrSide::Ask, dec!(101), dec!(3)),
                new(5, QrSide::Ask, dec!(100.5), dec!(1)),
                remove(5, "A", "100.5"),
            ],
            &[trade(t(3), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_diffs, 1);
    assert!(!after.tops.is_empty());
    assert!(after.tops.iter().all(|top| top.epoch == 1));
    assert!(after.prices.iter().all(|price| price.epoch == 1));
    assert!(after.market_events.iter().all(|event| event.epoch == 1));
    assert!(!after.references.is_empty());
    assert!(after.references.iter().all(|moved| moved.epoch == 1));
    assert!(!after.exposures.is_empty());
    assert!(after.exposures.iter().all(|life| life.epoch == 1));
    let sized = sized_events(&after.events);
    assert!(!sized.is_empty());
    assert!(sized.iter().all(|event| event.epoch == 1));
    let queue = queue_observations(&after.events, &[dec!(1); 3]).unwrap();
    assert!(!queue.observations.is_empty());
    assert!(queue.observations.iter().all(|seen| seen.epoch == 1));
}

#[test]
fn test_a_second_silence_after_a_trade_saw_the_first_is_seen_too() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    let t = |seconds: u64| START + seconds * SECOND;
    // A trade at 70 sees the first silence; the diff at 101 ends it.
    let more = [status(t(101), STATUS_OPEN, false, 10_000, 100, 3)];
    replay
        .ingest(
            &more,
            &[new(3, QrSide::Bid, dec!(100), dec!(1))],
            &[trade(t(70), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_diffs, 1);
    // A trade at 170 sees a second one, and empties the order the diff at 101 added.
    replay
        .ingest(&[], &[], &[trade(t(170), dec!(101), TradeSide::Buy)])
        .unwrap();
    assert_eq!(replay.coverage().resets_on_diffs, 2);
    assert_eq!(replay.coverage().orders_cut_by_reset, 3);
}

#[test]
fn test_a_trade_exactly_the_limit_after_the_last_diff_does_not_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    replay.ingest(&statuses, &diffs, &[]).unwrap();
    replay
        .ingest(
            &[],
            &[],
            &[trade(START + SECOND + GAP, dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_diffs, 0);
    assert_eq!(replay.coverage().touch_warm.at_touch, 1);
}

#[test]
fn test_a_diff_reset_after_a_later_trade_gap_does_not_move_the_warmup_back() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(60 * SECOND);
    replay
        .ingest(
            &statuses,
            &diffs,
            &[trade(START, dec!(100), TradeSide::Sell)],
        )
        .unwrap();
    let t = |seconds: u64| START + seconds * SECOND;
    // The diff resets at 100, the trade before it at 70. At 131 the warm-up from 100 still runs.
    let more = [
        status(t(100), STATUS_OPEN, false, 10_000, 100, 3),
        status(t(131), STATUS_OPEN, false, 10_000, 100, 4),
    ];
    let output = replay
        .ingest(
            &more,
            &[
                new(3, QrSide::Bid, dec!(100), dec!(1)),
                new(4, QrSide::Bid, dec!(100), dec!(1)),
            ],
            &[trade(t(70), dec!(100), TradeSide::Sell)],
        )
        .unwrap();
    assert_eq!(output.events[1].warmth, Warmth::Cold);
}

#[test]
fn test_a_re_emitted_stretch_of_statuses_is_left_out_and_counted() {
    let a = record(START, STATUS_OPEN, false, 10_000, 100, 1);
    let b = record(START + SECOND, STATUS_OPEN, false, 10_000, 100, 2);
    let c = record(START + 2 * SECOND, CANCELED, false, 10_000, 100, 1);
    let stream = [a.clone(), b.clone(), a, b, c].concat();
    let read = decode_order_statuses(&stream).unwrap();
    assert_eq!(read.repeated_lines, 2);
    assert_eq!(read.rows.len(), 3);
    assert_eq!(read.rows[2].status_id, CANCELED);
}

#[test]
fn test_a_trade_after_the_diffs_came_back_does_not_forget_the_book_again() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let first = [trade(START + SECOND, dec!(101), TradeSide::Buy)];
    replay.ingest(&statuses, &diffs, &first).unwrap();
    let t = |seconds: u64| START + SECOND + GAP + seconds * SECOND;
    let more = [
        status(t(1), STATUS_OPEN, false, 10_000, 200, 3),
        status(t(3), STATUS_OPEN, false, 10_000, 100, 4),
    ];
    let output = replay
        .ingest(
            &more,
            &[
                new(3, QrSide::Bid, dec!(100), dec!(2)),
                new(4, QrSide::Bid, dec!(100), dec!(1)),
            ],
            &[trade(t(2), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    let coverage = replay.coverage();
    assert_eq!(coverage.resets_on_diffs, 1);
    assert_eq!(coverage.resets_on_trades, 1, "both streams went silent");
    assert_eq!(
        output.events[1].level_before,
        dec!(2),
        "the book since the diff stands"
    );
    assert_eq!(output.events[1].epoch, 1);
}

#[test]
fn test_a_later_silence_in_the_trades_resets_after_an_earlier_reset() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let t = |seconds: u64| START + seconds * SECOND;
    replay
        .ingest(&statuses, &diffs, &[trade(t(1), dec!(101), TradeSide::Buy)])
        .unwrap();
    // An outage of both, then the trades go on.
    let more = [status(t(100), STATUS_OPEN, false, 10_000, 200, 3)];
    replay
        .ingest(
            &more,
            &[new(3, QrSide::Bid, dec!(100), dec!(2))],
            &[trade(t(100), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_trades, 1);
    assert_eq!(replay.coverage().orders_cut_by_reset, 2);
    // The diffs keep flowing, but the trades stop for over the limit.
    let later = [
        status(t(140), STATUS_OPEN, false, 10_000, 100, 4),
        status(t(180), STATUS_OPEN, false, 10_000, 100, 5),
    ];
    let output = replay
        .ingest(
            &later,
            &[
                new(4, QrSide::Bid, dec!(100), dec!(1)),
                new(5, QrSide::Bid, dec!(100), dec!(1)),
            ],
            &[trade(t(170), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_trades, 2);
    assert_eq!(output.events[1].epoch, 2);
    assert_eq!(output.events[1].level_before, dec!(0));
}

#[test]
fn test_a_diff_with_no_status_does_not_end_a_silence_a_trade_saw() {
    let (statuses, diffs) = opening();
    let mut replay = resetting(0);
    let t = |seconds: u64| START + seconds * SECOND;
    replay
        .ingest(
            &statuses,
            &diffs,
            &[trade(t(70), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_diffs, 1);
    // A diff with no status inherits the old time, so no time has passed for it.
    let output = replay
        .ingest(
            &[],
            &[new(3, QrSide::Bid, dec!(100), dec!(1))],
            &[trade(t(80), dec!(101), TradeSide::Buy)],
        )
        .unwrap();
    assert_eq!(output.events[0].time.evidence, TimeEvidence::Inherited);
    assert_eq!(replay.coverage().resets_on_diffs, 1, "the same silence");
    assert_eq!(replay.coverage().orders_cut_by_reset, 2);
}

/// A diff line of BTC in the dataset's format.
fn diff_line(oid: u64, side: &str, px: &str, change: &str) -> String {
    format!(
        r#"{{"user":"0x1","oid":{oid},"coin":"BTC","side":"{side}","px":"{px}","raw_book_diff":{change}}}"#
    )
}

#[test]
fn test_a_filter_leaves_out_the_tail_of_the_file_before_and_forgets_older_files() {
    let [a, b, c, d] = [1, 2, 3, 4].map(|oid| diff_line(oid, "B", "100", r#"{"new":{"sz":"1"}}"#));
    let mut filter = RepeatFilter::new();
    let first = read_book_diffs_after(
        [&a, &b, &c].map(String::as_str).join("\n").as_bytes(),
        "BTC",
        &mut filter,
    )
    .unwrap();
    assert_eq!(
        (first.rows.len(), first.repeated_from_previous_file),
        (3, 0)
    );
    // The next file starts with the last one's tail, and repeats a line of its own.
    let second = read_book_diffs_after(
        [&b, &c, &d, &d].map(String::as_str).join("\n").as_bytes(),
        "BTC",
        &mut filter,
    )
    .unwrap();
    assert_eq!(second.rows.len(), 1);
    assert_eq!(second.rows[0].oid, 4);
    assert_eq!(second.repeated_from_previous_file, 2);
    assert_eq!(second.repeated_lines, 1);
    // Only the file before is remembered: a line of two files back is kept.
    let third = read_book_diffs_after(a.as_bytes(), "BTC", &mut filter).unwrap();
    assert_eq!(
        (third.rows.len(), third.repeated_from_previous_file),
        (1, 0)
    );
    // Without the filter nothing carries over.
    assert_eq!(
        read_book_diffs([&b, &d].map(String::as_str).join("\n").as_bytes(), "BTC")
            .unwrap()
            .rows
            .len(),
        2
    );
}

#[test]
fn test_status_and_trade_files_leave_out_the_tail_of_the_file_before() {
    let a = record(START, STATUS_OPEN, false, 10_000, 100, 1);
    let b = record(START + SECOND, STATUS_OPEN, false, 10_000, 100, 2);
    let c = record(START + 2 * SECOND, CANCELED, false, 10_000, 100, 1);
    let mut filter = RepeatFilter::new();
    let first = decode_order_statuses_after(&[a.clone(), b.clone()].concat(), &mut filter).unwrap();
    assert_eq!(first.rows.len(), 2);
    let second = decode_order_statuses_after(&[b, c].concat(), &mut filter).unwrap();
    assert_eq!(second.rows.len(), 1);
    assert_eq!(second.rows[0].status_id, CANCELED);
    assert_eq!(second.repeated_from_previous_file, 1);

    let line = |second: u32, hash: &str| {
        format!(
            r#"{{"coin":"BTC","side":"B","time":"2025-12-30T18:00:{second:02}","px":"101","sz":"1","hash":"{hash}"}}"#
        )
    };
    let mut filter = RepeatFilter::new();
    let first = [line(1, "a"), line(2, "b")].join("\n");
    let second = [line(2, "b"), line(3, "c")].join("\n");
    assert_eq!(
        read_trades_after(first.as_bytes(), "BTC", &mut filter)
            .unwrap()
            .rows
            .len(),
        2
    );
    let read = read_trades_after(second.as_bytes(), "BTC", &mut filter).unwrap();
    assert_eq!(read.rows.len(), 1);
    assert_eq!(read.repeated_from_previous_file, 1);
}

/// Two hour files of statuses and diffs, the second starting with the first's tail, through one
/// replay; with filters or without.
fn overlapping_hours(filtered: bool) -> Replay {
    let open = |ts: u64, is_ask: bool, price: u32, oid: u64| {
        record(ts, STATUS_OPEN, is_ask, price, 100, oid)
    };
    let statuses = [
        [
            open(START, false, 10_000, 1),
            open(START + SECOND, true, 10_100, 2),
        ]
        .concat(),
        [
            open(START, false, 10_000, 1),
            open(START + SECOND, true, 10_100, 2),
            open(START + 2 * SECOND, false, 9_900, 3),
        ]
        .concat(),
    ];
    let one = diff_line(1, "B", "100", r#"{"new":{"sz":"1"}}"#);
    let two = diff_line(2, "A", "101", r#"{"new":{"sz":"1"}}"#);
    let three = diff_line(3, "B", "99", r#"{"new":{"sz":"1"}}"#);
    let diffs = [
        [&one, &two].map(String::as_str).join("\n"),
        [&one, &two, &three].map(String::as_str).join("\n"),
    ];
    let mut replay = Replay::new(config(0));
    let (mut status_filter, mut diff_filter) = (RepeatFilter::new(), RepeatFilter::new());
    for (statuses, diffs) in statuses.iter().zip(&diffs) {
        let (statuses, diffs) = if filtered {
            (
                decode_order_statuses_after(statuses, &mut status_filter).unwrap(),
                read_book_diffs_after(diffs.as_bytes(), "BTC", &mut diff_filter).unwrap(),
            )
        } else {
            (
                decode_order_statuses(statuses).unwrap(),
                read_book_diffs(diffs.as_bytes(), "BTC").unwrap(),
            )
        };
        replay.ingest(&statuses.rows, &diffs.rows, &[]).unwrap();
    }
    replay
}

#[test]
fn test_hours_that_overlap_step_back_unless_read_through_one_filter_per_stream() {
    let unfiltered = overlapping_hours(false);
    assert!(unfiltered.coverage().time_steps_back > 0);
    assert!(unfiltered.coverage().duplicate_new > 0);
    let filtered = overlapping_hours(true);
    assert_eq!(filtered.coverage().time_steps_back, 0);
    assert_eq!(filtered.coverage().duplicate_new, 0);
    assert_eq!(filtered.coverage().diffs, 3);
}

// The book on a time grid (#263).

fn sampling(warmup: u64, reset_gap: u64) -> Replay {
    Replay::new(
        ReplayConfig::new(dec!(1), 3, warmup, reset_gap)
            .unwrap()
            .with_book_samples(1_000)
            .unwrap(),
    )
}

fn ms(ns: u64) -> u64 {
    ns / 1_000_000
}

#[test]
fn test_a_book_sample_shows_every_diff_before_its_time_and_none_at_it() {
    let (mut statuses, mut diffs) = opening();
    // A bid at 99.5, one tick from the mid 100.5, so at limit 0 with 100; one at 99, limit 1.
    statuses.push(status(
        START + 2_500_000_000,
        STATUS_OPEN,
        false,
        9_950,
        50,
        3,
    ));
    statuses.push(status(
        START + 2_500_000_000,
        STATUS_OPEN,
        false,
        9_900,
        100,
        4,
    ));
    // Past three limits on either side: in no sample, and the side is read from the touch out.
    statuses.push(status(
        START + 2_500_000_000,
        STATUS_OPEN,
        false,
        9_600,
        100,
        9,
    ));
    statuses.push(status(
        START + 2_500_000_000,
        STATUS_OPEN,
        true,
        10_500,
        100,
        10,
    ));
    // A block of two diffs at exactly START + 5 s: no sample at 5 s shows either.
    statuses.push(status(
        START + 5 * SECOND,
        STATUS_OPEN,
        true,
        10_100,
        400,
        5,
    ));
    statuses.push(status(
        START + 5 * SECOND,
        STATUS_OPEN,
        true,
        10_200,
        100,
        6,
    ));
    diffs.push(new(3, QrSide::Bid, dec!(99.5), dec!(0.5)));
    diffs.push(new(4, QrSide::Bid, dec!(99), dec!(1)));
    diffs.push(new(9, QrSide::Bid, dec!(96), dec!(1)));
    diffs.push(new(10, QrSide::Ask, dec!(105), dec!(1)));
    diffs.push(new(5, QrSide::Ask, dec!(101), dec!(4)));
    diffs.push(new(6, QrSide::Ask, dec!(102), dec!(1)));
    let mut replay = sampling(0, u64::MAX);
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // The first diff at START starts the grid after it. At START + 1 s the ask arrives, and the
    // sample there shows the book before it: one sided, so no sample.
    assert_eq!(replay.coverage().book_samples_without_touch, 1);
    let times: Vec<u64> = output.book.iter().map(|s| s.value.timestamp_ms).collect();
    assert_eq!(
        times,
        vec![
            ms(START + 2 * SECOND),
            ms(START + 3 * SECOND),
            ms(START + 4 * SECOND),
            ms(START + 5 * SECOND),
        ]
    );
    let first = output.book[0].value;
    assert_eq!(first.bid, [dec!(2), dec!(0), dec!(0), dec!(0), dec!(0)]);
    assert_eq!(first.ask, [dec!(3), dec!(0), dec!(0), dec!(0), dec!(0)]);
    assert_eq!((first.best_bid, first.best_ask), (dec!(100), dec!(101)));
    assert_eq!(first.warmth, Warmth::Warm);
    for sample in &output.book[1..] {
        assert_eq!(
            sample.value.bid,
            [dec!(2.5), dec!(1), dec!(0), dec!(0), dec!(0)]
        );
        assert_eq!(
            sample.value.ask,
            [dec!(3), dec!(0), dec!(0), dec!(0), dec!(0)]
        );
    }
    assert!(
        output
            .book
            .iter()
            .all(|s| s.time.evidence == TimeEvidence::Joined)
    );
    assert!(output.book.iter().all(|s| s.epoch == 0));

    // The block applies, and the next diff, in another hour, shows it whole.
    let later = START + 7 * SECOND;
    let output = replay
        .ingest(
            &[status(later, CANCELED, false, 9_900, 100, 4)],
            &[remove(4, "B", "99")],
            &[],
        )
        .unwrap();
    let times: Vec<u64> = output.book.iter().map(|s| s.value.timestamp_ms).collect();
    assert_eq!(times, vec![ms(START + 6 * SECOND), ms(later)]);
    for sample in &output.book {
        assert_eq!(
            sample.value.ask,
            [dec!(7), dec!(1), dec!(0), dec!(0), dec!(0)]
        );
        assert_eq!(
            sample.value.bid,
            [dec!(2.5), dec!(1), dec!(0), dec!(0), dec!(0)]
        );
    }
}

#[test]
fn test_a_book_sample_takes_the_weaker_time_evidence_and_the_warmth_at_its_time() {
    let (mut statuses, mut diffs) = opening();
    // A diff with no status inherits START + 1 s; the next joins at START + 3 s.
    diffs.push(new(7, QrSide::Bid, dec!(99), dec!(1)));
    statuses.push(status(
        START + 3 * SECOND,
        STATUS_OPEN,
        false,
        9_800,
        100,
        8,
    ));
    diffs.push(new(8, QrSide::Bid, dec!(98), dec!(1)));
    let mut replay = sampling(2_500_000_000, u64::MAX);
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    let samples: Vec<_> = output
        .book
        .iter()
        .map(|s| (s.value.timestamp_ms, s.time.evidence, s.value.warmth))
        .collect();
    assert_eq!(
        samples,
        vec![
            (
                ms(START + 2 * SECOND),
                TimeEvidence::Inherited,
                Warmth::Cold
            ),
            (
                ms(START + 3 * SECOND),
                TimeEvidence::Inherited,
                Warmth::Warm
            ),
        ]
    );
}

#[test]
fn test_a_reset_leaves_no_book_sample_across_the_silence() {
    let (statuses, diffs) = opening();
    let mut replay = sampling(0, GAP);
    let first = replay.ingest(&statuses, &diffs, &[]).unwrap();
    assert!(first.book.is_empty());
    let later = START + SECOND + GAP + 1;
    let more = [
        status(later, STATUS_OPEN, false, 9_900, 100, 3),
        status(later + SECOND, STATUS_OPEN, true, 10_000, 100, 4),
        status(later + 3 * SECOND, STATUS_OPEN, true, 10_100, 100, 5),
    ];
    let more_diffs = [
        new(3, QrSide::Bid, dec!(99), dec!(1)),
        new(4, QrSide::Ask, dec!(100), dec!(1)),
        new(5, QrSide::Ask, dec!(101), dec!(1)),
    ];
    let output = replay.ingest(&more, &more_diffs, &[]).unwrap();
    // Nothing of the forgotten book, neither as a sample nor as a grid time without a touch.
    assert_eq!(replay.coverage().resets_on_diffs, 1);
    assert!(output.book.iter().all(|s| s.epoch == 1));
    assert!(output.book.iter().all(|s| s.value.timestamp_ms > ms(later)));
    // After the reset: the grid restarts after the diff that showed it; at its first time the
    // ask has not come, at the next it has. The other grid time without a touch is the
    // opening's, at START + 1 s.
    assert_eq!(replay.coverage().book_samples_without_touch, 2);
    let times: Vec<u64> = output.book.iter().map(|s| s.value.timestamp_ms).collect();
    assert_eq!(
        times,
        vec![ms(later - 1 + 2 * SECOND), ms(later - 1 + 3 * SECOND)]
    );
    assert_eq!(output.book[0].value.best_ask, dec!(100));
}

#[test]
fn test_a_silence_in_the_trades_also_leaves_no_book_sample_across_it() {
    let (statuses, diffs) = opening();
    let mut replay = sampling(0, GAP);
    replay
        .ingest(
            &statuses,
            &diffs,
            &[trade(START, dec!(100), TradeSide::Sell)],
        )
        .unwrap();
    // Diffs keep coming every 30 s, so only the trades are silent, until a trade at 100 s.
    let mut statuses = Vec::new();
    let mut diffs = Vec::new();
    for (i, at) in [30u64, 60, 90, 120].iter().enumerate() {
        let oid = 10 + i as u64;
        statuses.push(status(
            START + at * SECOND,
            STATUS_OPEN,
            false,
            9_900,
            100,
            oid,
        ));
        diffs.push(new(oid, QrSide::Bid, dec!(99), dec!(1)));
    }
    let output = replay
        .ingest(
            &statuses,
            &diffs,
            &[trade(START + 100 * SECOND, dec!(100), TradeSide::Sell)],
        )
        .unwrap();
    assert_eq!(replay.coverage().resets_on_trades, 1);
    let before: Vec<_> = output.book.iter().filter(|s| s.epoch == 0).collect();
    let after: Vec<_> = output.book.iter().filter(|s| s.epoch == 1).collect();
    // Before the reset, samples up to the last diff of the old epoch; none after the trade
    // forgot the book, which has no touch again before the end.
    assert_eq!(
        before.last().unwrap().value.timestamp_ms,
        ms(START + 90 * SECOND)
    );
    assert!(after.is_empty());
    // The only grid time without a touch is the opening's: the forgotten stretch is not one.
    assert_eq!(replay.coverage().book_samples_without_touch, 1);
    assert!(
        output
            .book
            .iter()
            .all(|s| s.value.timestamp_ms <= ms(START + 90 * SECOND))
    );
}

#[test]
fn test_no_book_samples_unless_asked_and_the_grid_is_checked() {
    let (statuses, diffs) = opening();
    let mut more = statuses.clone();
    more.push(status(
        START + 5 * SECOND,
        STATUS_OPEN,
        false,
        9_900,
        100,
        3,
    ));
    let mut more_diffs = diffs.clone();
    more_diffs.push(new(3, QrSide::Bid, dec!(99), dec!(1)));
    let mut replay = Replay::new(config(0));
    let output = replay.ingest(&more, &more_diffs, &[]).unwrap();
    assert!(output.book.is_empty());
    assert_eq!(replay.coverage().book_samples_without_touch, 0);
    assert!(config(0).book_sample_ns.is_none());
    assert_eq!(
        config(0).with_book_samples(250).unwrap().book_sample_ns,
        Some(250_000_000)
    );
    assert!(config(0).with_book_samples(0).is_err());
    assert!(config(0).with_book_samples(u64::MAX).is_err());
    let six = ReplayConfig::new(dec!(1), 6, 0, u64::MAX).unwrap();
    assert!(six.with_book_samples(1_000).is_err());
    let five = ReplayConfig::new(dec!(1), 5, 0, u64::MAX).unwrap();
    assert!(five.with_book_samples(1_000).is_ok());
}

#[test]
fn test_book_samples_count_queues_in_event_sizes_and_the_spread_in_ticks() {
    let statuses = vec![
        status(START, STATUS_OPEN, false, 10_000, 125, 1),
        status(START, STATUS_OPEN, true, 10_200, 300, 2),
        status(START, STATUS_OPEN, false, 9_900, 40, 3),
        status(START + 2 * SECOND, STATUS_OPEN, true, 10_300, 100, 4),
    ];
    let diffs = vec![
        new(1, QrSide::Bid, dec!(100), dec!(1.25)),
        new(2, QrSide::Ask, dec!(102), dec!(3)),
        new(3, QrSide::Bid, dec!(99), dec!(0.4)),
        new(4, QrSide::Ask, dec!(103), dec!(1)),
    ];
    let mut replay = Replay::new(
        ReplayConfig::new(dec!(1), 3, 1_500_000_000, u64::MAX)
            .unwrap()
            .with_book_samples(1_000)
            .unwrap(),
    );
    let output = replay.ingest(&statuses, &diffs, &[]).unwrap();
    // Grid times START + 1 s (cold) and START + 2 s (warm).
    assert_eq!(output.book.len(), 2);
    let sizes = [dec!(0.5), dec!(0.25), dec!(1)];
    let samples = book_samples(&output.book, &sizes, dec!(1)).unwrap();
    assert_eq!(samples.len(), 1, "the cold sample is left out");
    let sample = samples[0].value;
    assert_eq!(sample.timestamp_ms, ms(START + 2 * SECOND));
    // Mid 101: 100 and 102 are one tick away, limit 0; 99 is limit 1.
    // 1.25 / 0.5 = 2.5 rounds to 2 (banker's, as the queue observations); 0.4 / 0.25 = 1.6 to 2.
    assert_eq!(sample.bid, [2, 2, 0, 0, 0]);
    assert_eq!(sample.ask, [6, 0, 0, 0, 0]);
    assert_eq!(sample.spread_ticks, 2);
    assert_eq!(samples[0].epoch, 0);
    // The spread rounds up on a finer declared tick, as a count of ticks.
    let halves = book_samples(&output.book, &sizes, dec!(0.75)).unwrap();
    assert_eq!(halves[0].value.spread_ticks, 3);
    // A limit with size needs an event size; an empty one does not.
    assert!(book_samples(&output.book, &sizes[..1], dec!(1)).is_err());
    assert!(book_samples(&output.book, &sizes[..2], dec!(1)).is_ok());
    assert!(book_samples(&output.book, &[dec!(0.5), dec!(0), dec!(1)], dec!(1)).is_err());
    assert!(book_samples(&output.book, &sizes, dec!(0)).is_err());
}
