//! L3 reconstruction fixtures: every message type, the handshake, gaps,
//! each discrepancy (atomic, fail closed), snapshots, and queue estimates.

#![cfg(feature = "l3")]

use market_maker_rs::Decimal;
use market_maker_rs::l3::{
    Applied, ApplyError, Assumption, BookReconstructor, ChangeReason, Discrepancy, DoneReason,
    HandshakeError, InvalidationKind, L3Event, L3Message, L3Side, L3Snapshot, PriorityEvidence,
    QueueError, SegmentEnd, SizeDecrease, SnapshotError, SnapshotOrder, VenueRules, ViewError,
};
use rust_decimal_macros::dec;

const P: &str = "BTC-USD";
use L3Side::{Buy, Sell};

fn m(seq: u64, event: L3Event) -> L3Message {
    L3Message {
        product_id: P.to_string(),
        sequence: seq,
        time_us: 1_000,
        event,
    }
}

fn received(id: &str, side: L3Side) -> L3Event {
    L3Event::Received {
        order_id: id.into(),
        side,
    }
}

fn open(id: &str, side: L3Side, price: Decimal, size: Decimal) -> L3Event {
    L3Event::Open {
        order_id: id.into(),
        side,
        price,
        remaining: size,
    }
}

fn cancel(id: &str, side: L3Side, price: Decimal, remaining: Decimal) -> L3Event {
    L3Event::Done {
        order_id: id.into(),
        side,
        price: Some(price),
        reason: DoneReason::Canceled,
        remaining: Some(remaining),
    }
}

fn filled(id: &str, side: L3Side, price: Decimal) -> L3Event {
    L3Event::Done {
        order_id: id.into(),
        side,
        price: Some(price),
        reason: DoneReason::Filled,
        remaining: Some(Decimal::ZERO),
    }
}

fn trade(maker: &str, taker: &str, side: L3Side, price: Decimal, size: Decimal) -> L3Event {
    L3Event::Match {
        maker_id: maker.into(),
        taker_id: taker.into(),
        side,
        price,
        size,
    }
}

fn stp(id: &str, side: L3Side, price: Decimal, old: Decimal, new: Decimal) -> L3Event {
    L3Event::Change {
        order_id: id.into(),
        side,
        reason: ChangeReason::Stp,
        price: Some(price),
        old_size: Some(old),
        new_size: Some(new),
        old_price: None,
        new_price: None,
        new_funds: None,
    }
}

fn so(id: &str, price: Decimal, size: Decimal) -> SnapshotOrder {
    SnapshotOrder {
        order_id: id.into(),
        price,
        size,
    }
}

/// Bids: s1 100 x 1, s2 100 x 2, s3 99 x 1. Asks: s4 101 x 1, s5 102 x 3.
fn snapshot(seq: u64) -> L3Snapshot {
    L3Snapshot {
        product_id: P.to_string(),
        sequence: seq,
        bids: vec![
            so("s1", dec!(100), dec!(1)),
            so("s2", dec!(100), dec!(2)),
            so("s3", dec!(99), dec!(1)),
        ],
        asks: vec![so("s4", dec!(101), dec!(1)), so("s5", dec!(102), dec!(3))],
    }
}

fn booted(seq: u64) -> BookReconstructor {
    let mut r = BookReconstructor::new(P, VenueRules::default());
    r.bootstrap(&snapshot(seq)).unwrap();
    r.audit().unwrap();
    r
}

/// Applies in order, auditing every invariant after each message.
fn feed(r: &mut BookReconstructor, messages: &[L3Message]) {
    for message in messages {
        r.apply(message).unwrap();
        r.audit().unwrap();
    }
}

#[test]
fn test_snapshot_adds_changes_matches_and_removals_rebuild_the_book() {
    let mut r = booted(10);
    feed(
        &mut r,
        &[
            m(11, received("a", Buy)),
            m(12, open("a", Buy, dec!(100), dec!(0.5))),
            m(13, stp("s2", Buy, dec!(100), dec!(2), dec!(1.5))),
            // Taker b sells 1.2 into s1 (1) and s2 (0.2), then its rest
            // never rests (a market order).
            m(14, received("b", Sell)),
            m(15, trade("s1", "b", Buy, dec!(100), dec!(1))),
            m(16, trade("s2", "b", Buy, dec!(100), dec!(0.2))),
            m(17, filled("b", Sell, dec!(100))),
            m(18, cancel("s5", Sell, dec!(102), dec!(3))),
            m(19, open("c", Sell, dec!(103), dec!(2))),
        ],
    );
    assert_eq!(
        r.l2(Buy).unwrap(),
        vec![(dec!(100), dec!(1.8)), (dec!(99), dec!(1))]
    );
    assert_eq!(
        r.l2(Sell).unwrap(),
        vec![(dec!(101), dec!(1)), (dec!(103), dec!(2))]
    );
    let queue = r.level_queue(Buy, dec!(100)).unwrap();
    let ids: Vec<&str> = queue.iter().map(|q| q.order_id.as_str()).collect();
    assert_eq!(ids, ["s2", "a"]);
    assert_eq!(queue[0].evidence, PriorityEvidence::Unknown);
    assert_eq!(
        queue[1].evidence,
        PriorityEvidence::Observed { sequence: 12 }
    );
    assert_eq!(r.last_sequence().unwrap(), 19);
    let c = r.manifest().counts;
    assert_eq!(
        (c.received, c.open, c.matched, c.stp_decrease, c.done),
        (2, 2, 2, 1, 1)
    );
    // b never rested: its done has no visible effect.
    assert_eq!(c.off_book_done, 1);
}

#[test]
fn test_off_book_events_have_no_visible_effect() {
    let mut r = booted(10);
    let before = r.l2(Buy).unwrap();
    let off = [
        m(11, received("x", Buy)),
        m(
            12,
            L3Event::Change {
                order_id: "x".into(),
                side: Buy,
                reason: ChangeReason::Stp,
                price: None,
                old_size: None,
                new_size: None,
                old_price: None,
                new_price: None,
                new_funds: Some(dec!(50)),
            },
        ),
        m(13, filled("x", Buy, dec!(100))),
    ];
    for message in &off {
        assert_eq!(r.apply(message), Ok(Applied::NoBookEffect));
    }
    // A full match removes s4; its later done is idempotent.
    feed(
        &mut r,
        &[
            m(14, trade("s4", "t", Sell, dec!(101), dec!(1))),
            m(15, filled("s4", Sell, dec!(101))),
        ],
    );
    assert_eq!(r.l2(Buy).unwrap(), before);
    assert_eq!(r.l2(Sell).unwrap(), vec![(dec!(102), dec!(3))]);
    let c = r.manifest().counts;
    assert_eq!((c.off_book_change, c.off_book_done), (1, 2));
}

#[test]
fn test_noop_change_and_stp_to_zero() {
    let mut r = booted(10);
    feed(
        &mut r,
        &[
            m(11, stp("s3", Buy, dec!(99), dec!(1), dec!(1))),
            m(12, stp("s3", Buy, dec!(99), dec!(1), dec!(0))),
        ],
    );
    assert_eq!(r.manifest().counts.noop_change, 1);
    // A decrease to zero removes the order and its level.
    assert_eq!(r.l2(Buy).unwrap(), vec![(dec!(100), dec!(3))]);
}

#[test]
fn test_snapshot_handshake_drops_up_to_s_and_needs_s_plus_one() {
    let buffer: Vec<L3Message> = vec![
        m(8, received("old", Buy)),
        m(9, open("old", Buy, dec!(98), dec!(1))),
        m(10, received("dup", Buy)),
        m(11, open("a", Buy, dec!(99), dec!(1))),
        m(12, cancel("s4", Sell, dec!(101), dec!(1))),
    ];
    let mut r = BookReconstructor::new(P, VenueRules::default());
    assert_eq!(r.bootstrap_with_buffer(&snapshot(10), &buffer), Ok(2));
    r.audit().unwrap();
    assert_eq!(r.manifest().counts.handshake_dropped, 3);
    assert_eq!(r.l2(Buy).unwrap()[1], (dec!(99), dec!(2)));

    // A buffer that starts beyond S + 1 is a gap.
    let late = vec![m(12, open("a", Buy, dec!(99), dec!(1)))];
    let mut r = BookReconstructor::new(P, VenueRules::default());
    let Err(HandshakeError::Apply(ApplyError::Invalidated(inv))) =
        r.bootstrap_with_buffer(&snapshot(10), &late)
    else {
        panic!("expected a gap");
    };
    assert_eq!(
        inv.kind,
        InvalidationKind::Gap {
            expected: 11,
            got: 12
        }
    );
}

#[test]
fn test_gap_invalidates_until_a_new_snapshot() {
    let mut r = booted(10);
    feed(&mut r, &[m(11, open("a", Buy, dec!(99), dec!(1)))]);
    let good = r.l2(Buy).unwrap();
    let error = r.apply(&m(13, received("b", Buy))).unwrap_err();
    let ApplyError::Invalidated(inv) = error else {
        panic!("{error:?}");
    };
    assert_eq!((inv.last_good_seq, inv.first_offending_seq), (11, 13));
    // No valid view; the last good book only as a stale diagnostic.
    assert!(matches!(r.l2(Buy), Err(ViewError::Invalidated(_))));
    assert!(r.audit().is_err());
    assert_eq!(r.stale_l2(Buy), Some(good));
    assert!(matches!(
        r.apply(&m(14, received("c", Buy))),
        Err(ApplyError::AwaitingSnapshot(_))
    ));
    assert_eq!(r.manifest().counts.skipped_while_invalid, 1);
    // Recovery only through a snapshot: a new segment.
    r.bootstrap(&snapshot(20)).unwrap();
    feed(&mut r, &[m(21, received("d", Buy))]);
    let segments = &r.manifest().segments;
    assert_eq!(segments.len(), 2);
    assert!(matches!(segments[0].end, SegmentEnd::Invalidated(_)));
    assert_eq!(segments[0].last_good_seq, 11);
    assert_eq!((segments[1].start_seq, segments[1].last_good_seq), (20, 21));
    assert_eq!(segments[1].end, SegmentEnd::Open);
    assert!(r.stale_l2(Buy).is_none());
}

#[test]
fn test_out_of_order_or_duplicate_sequence_invalidates() {
    for seq in [11, 5] {
        let mut r = booted(10);
        feed(&mut r, &[m(11, received("a", Buy))]);
        let error = r.apply(&m(seq, received("a", Buy))).unwrap_err();
        assert!(
            matches!(
                error,
                ApplyError::Invalidated(ref inv) if inv.kind
                    == InvalidationKind::OutOfOrderOrDuplicate { last: 11, got: seq }
            ),
            "{error:?}"
        );
    }
}

#[test]
fn test_sequence_overflow_invalidates() {
    let mut r = booted(u64::MAX);
    let error = r.apply(&m(0, received("a", Buy))).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Invalidated(ref inv) if inv.kind == InvalidationKind::SequenceOverflow
    ));
}

#[test]
fn test_another_product_is_refused_and_leaves_the_segment_untouched() {
    let mut r = booted(10);
    let mut other = m(11, open("e", Buy, dec!(99), dec!(5)));
    other.product_id = "ETH-USD".into();
    assert!(matches!(
        r.apply(&other),
        Err(ApplyError::WrongProduct { .. })
    ));
    assert_eq!(r.manifest().counts.wrong_product, 1);
    // Still valid and still expecting 11.
    feed(&mut r, &[m(11, received("a", Buy))]);
    assert_eq!(r.manifest().segments[0].end, SegmentEnd::Open);
}

/// Each case: the message, the expected discrepancy.
type DiscrepancyCase = (L3Event, fn(&Discrepancy) -> bool);
type SnapshotCase = (L3Snapshot, fn(&SnapshotError) -> bool);

fn discrepancy_cases() -> Vec<DiscrepancyCase> {
    vec![
        (trade("ghost", "t", Buy, dec!(100), dec!(1)), |d| {
            matches!(d, Discrepancy::UnknownMaker { .. })
        }),
        (trade("s1", "s4", Buy, dec!(100), dec!(1)), |d| {
            matches!(d, Discrepancy::TakerResting { .. })
        }),
        (trade("s1", "t", Sell, dec!(100), dec!(1)), |d| {
            matches!(d, Discrepancy::MatchMismatch { .. })
        }),
        (trade("s1", "t", Buy, dec!(99), dec!(1)), |d| {
            matches!(d, Discrepancy::MatchMismatch { .. })
        }),
        (trade("s1", "t", Buy, dec!(100), dec!(1.5)), |d| {
            matches!(d, Discrepancy::MatchExceedsRemaining { .. })
        }),
        (trade("s1", "t", Buy, dec!(100), dec!(0)), |d| {
            matches!(d, Discrepancy::InvalidValue { .. })
        }),
        (open("s1", Buy, dec!(98), dec!(1)), |d| {
            matches!(d, Discrepancy::DuplicateOpen { .. })
        }),
        (open("n", Buy, dec!(98), dec!(0)), |d| {
            matches!(d, Discrepancy::InvalidValue { .. })
        }),
        (open("n", Buy, dec!(101), dec!(1)), |d| {
            matches!(d, Discrepancy::CrossedAfterApply { .. })
        }),
        (open("n", Sell, dec!(100), dec!(1)), |d| {
            matches!(d, Discrepancy::CrossedAfterApply { .. })
        }),
        (filled("s1", Buy, dec!(100)), |d| {
            matches!(d, Discrepancy::MissingMatch { .. })
        }),
        (cancel("s1", Buy, dec!(100), dec!(0.5)), |d| {
            matches!(d, Discrepancy::RemainingMismatch { .. })
        }),
        (cancel("s1", Sell, dec!(100), dec!(1)), |d| {
            matches!(d, Discrepancy::DoneMismatch { .. })
        }),
        (cancel("s1", Buy, dec!(99), dec!(1)), |d| {
            matches!(d, Discrepancy::DoneMismatch { .. })
        }),
        (
            L3Event::Change {
                order_id: "s1".into(),
                side: Buy,
                reason: ChangeReason::Stp,
                price: Some(dec!(100)),
                old_size: Some(dec!(1)),
                new_size: None,
                old_price: None,
                new_price: None,
                new_funds: None,
            },
            |d| matches!(d, Discrepancy::MalformedChange { .. }),
        ),
        (
            L3Event::Change {
                order_id: "s1".into(),
                side: Buy,
                reason: ChangeReason::Stp,
                price: None,
                old_size: None,
                new_size: None,
                old_price: None,
                new_price: None,
                new_funds: Some(dec!(10)),
            },
            |d| matches!(d, Discrepancy::MalformedChange { .. }),
        ),
        (stp("s1", Buy, dec!(100), dec!(0.7), dec!(0.5)), |d| {
            matches!(d, Discrepancy::ChangeMismatch { .. })
        }),
        (stp("s1", Buy, dec!(100), dec!(1), dec!(2)), |d| {
            matches!(d, Discrepancy::UnsupportedModification { .. })
        }),
        (
            L3Event::Change {
                order_id: "s1".into(),
                side: Buy,
                reason: ChangeReason::ModifyOrder,
                price: None,
                old_size: Some(dec!(1)),
                new_size: Some(dec!(1)),
                old_price: Some(dec!(100)),
                new_price: Some(dec!(99.5)),
                new_funds: None,
            },
            |d| matches!(d, Discrepancy::UnsupportedModification { .. }),
        ),
        (
            L3Event::Change {
                order_id: "s1".into(),
                side: Buy,
                reason: ChangeReason::Stp,
                price: None,
                old_size: None,
                new_size: Some(dec!(0.5)),
                old_price: Some(dec!(100)),
                new_price: Some(dec!(99.5)),
                new_funds: None,
            },
            |d| matches!(d, Discrepancy::UnsupportedModification { .. }),
        ),
        (
            L3Event::Unsupported {
                kind: "activate".into(),
            },
            |d| matches!(d, Discrepancy::UnsupportedInput { .. }),
        ),
    ]
}

#[test]
fn test_each_discrepancy_invalidates_atomically() {
    for (i, (event, expected)) in discrepancy_cases().into_iter().enumerate() {
        let mut r = booted(10);
        let (bids, asks) = (r.l2(Buy).unwrap(), r.l2(Sell).unwrap());
        let queue = r.level_queue(Buy, dec!(100)).unwrap();
        let error = r.apply(&m(11, event.clone())).unwrap_err();
        let ApplyError::Invalidated(inv) = &error else {
            panic!("case {i}: {error:?}");
        };
        let InvalidationKind::Discrepancy(d) = &inv.kind else {
            panic!("case {i}: {inv:?}");
        };
        assert!(expected(d), "case {i}: {d:?}");
        assert_eq!((inv.last_good_seq, inv.first_offending_seq), (10, 11));
        // Nothing changed: the stale book is the one before the message.
        assert_eq!(r.stale_l2(Buy).unwrap(), bids, "case {i}");
        assert_eq!(r.stale_l2(Sell).unwrap(), asks, "case {i}");
        assert!(r.level_queue(Buy, dec!(100)).is_err());
        // And a new snapshot restores the same book.
        r.bootstrap(&snapshot(20)).unwrap();
        assert_eq!(r.level_queue(Buy, dec!(100)).unwrap(), queue);
    }
}

#[test]
fn test_capacity_and_arithmetic_failures_invalidate() {
    let rules = VenueRules {
        max_orders: 5,
        ..VenueRules::default()
    };
    let mut r = BookReconstructor::new(P, rules);
    r.bootstrap(&snapshot(10)).unwrap();
    let error = r
        .apply(&m(11, open("n", Buy, dec!(98), dec!(1))))
        .unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Invalidated(ref inv)
            if matches!(inv.kind, InvalidationKind::Discrepancy(Discrepancy::CapacityExceeded { .. }))
    ));
    let rules = VenueRules {
        max_levels: 4,
        ..VenueRules::default()
    };
    let mut r = BookReconstructor::new(P, rules);
    r.bootstrap(&snapshot(10)).unwrap();
    // A new order at an existing level fits; a new level does not.
    feed(&mut r, &[m(11, open("n", Buy, dec!(99), dec!(1)))]);
    assert!(r.apply(&m(12, open("o", Buy, dec!(98), dec!(1)))).is_err());

    let mut r = booted(10);
    let huge = Decimal::MAX;
    let error = r.apply(&m(11, open("h", Buy, dec!(99), huge))).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Invalidated(ref inv)
            if matches!(inv.kind, InvalidationKind::Discrepancy(Discrepancy::Arithmetic { .. }))
    ));
}

#[test]
fn test_invalid_snapshots_are_refused_and_change_nothing() {
    let mut r = booted(10);
    feed(&mut r, &[m(11, open("a", Buy, dec!(99), dec!(1)))]);
    let before = (r.l2(Buy).unwrap(), r.last_sequence().unwrap());
    let mut duplicate = snapshot(20);
    duplicate.asks.push(so("s1", dec!(105), dec!(1)));
    let mut zero = snapshot(20);
    zero.bids.push(so("z", dec!(98), dec!(0)));
    let mut crossed = snapshot(20);
    crossed.bids.push(so("x", dec!(101), dec!(1)));
    let mut other = snapshot(20);
    other.product_id = "ETH-USD".into();
    let mut overflow = snapshot(20);
    overflow.bids.push(so("o1", dec!(97), Decimal::MAX));
    overflow.bids.push(so("o2", dec!(97), Decimal::MAX));
    let cases: [SnapshotCase; 5] = [
        (duplicate, |e| matches!(e, SnapshotError::DuplicateId(_))),
        (zero, |e| matches!(e, SnapshotError::InvalidValue(_))),
        (crossed, |e| matches!(e, SnapshotError::Crossed { .. })),
        (other, |e| matches!(e, SnapshotError::WrongProduct { .. })),
        (overflow, |e| matches!(e, SnapshotError::Arithmetic(_))),
    ];
    for (snap, expected) in cases {
        let error = r.bootstrap(&snap).unwrap_err();
        assert!(expected(&error), "{error:?}");
        assert_eq!((r.l2(Buy).unwrap(), r.last_sequence().unwrap()), before);
    }
    let small = VenueRules {
        max_orders: 4,
        ..VenueRules::default()
    };
    let mut r = BookReconstructor::new(P, small);
    assert!(matches!(
        r.bootstrap(&snapshot(10)),
        Err(SnapshotError::CapacityExceeded(_))
    ));
    assert!(matches!(r.l2(Buy), Err(ViewError::NotBootstrapped)));
    assert_eq!(r.manifest().counts.rejected_snapshots, 1);
}

#[test]
fn test_priority_evidence_and_profile() {
    let mut r = booted(10);
    feed(
        &mut r,
        &[
            m(11, open("a", Buy, dec!(100), dec!(4))),
            // A snapshot order stays Unknown after a decrease.
            m(12, stp("s2", Buy, dec!(100), dec!(2), dec!(1))),
        ],
    );
    let queue = r.level_queue(Buy, dec!(100)).unwrap();
    assert_eq!(queue[1].evidence, PriorityEvidence::Unknown);
    let profile = r.priority_profile(Buy).unwrap();
    assert_eq!(profile[0].price, dec!(100));
    assert_eq!(profile[0].total, dec!(6));
    assert_eq!(profile[0].unknown_priority, dec!(2));
    assert_eq!(profile[1].unknown_priority, dec!(1));
}

#[test]
fn test_volume_ahead_of_a_hypothetical_order() {
    let mut r = booted(10);
    let estimate = r.queue_estimate(Buy, dec!(100)).unwrap();
    let ahead = r.volume_ahead(&estimate).unwrap();
    assert_eq!(
        (ahead.volume, ahead.from_snapshot, ahead.orders),
        (dec!(3), dec!(3), 2)
    );
    assert_eq!(
        ahead.assumptions,
        vec![Assumption::BackOfVisibleQueueAtCreation]
    );
    feed(
        &mut r,
        &[
            // Behind: opened later at the same price, same time stamp.
            m(11, open("later", Buy, dec!(100), dec!(5))),
            // Ahead: s1 matched by 0.4, s2 decreased by 0.5.
            m(12, trade("s1", "t", Buy, dec!(100), dec!(0.4))),
            m(13, stp("s2", Buy, dec!(100), dec!(2), dec!(1.5))),
            // Behind order matched: nothing changes ahead.
            m(14, trade("later", "t", Buy, dec!(100), dec!(1))),
        ],
    );
    let ahead = r.volume_ahead(&estimate).unwrap();
    // 0.6 + 1.5, each from its own remaining (no double subtraction).
    assert_eq!(ahead.volume, dec!(2.1));
    assert_eq!(ahead.at_seq, 14);
    assert!(
        ahead
            .assumptions
            .contains(&Assumption::SizeDecreaseKeepsPriority)
    );
    feed(&mut r, &[m(15, cancel("s2", Buy, dec!(100), dec!(1.5)))]);
    assert_eq!(r.volume_ahead(&estimate).unwrap().volume, dec!(0.6));
    // A new snapshot makes the estimate stale.
    r.bootstrap(&snapshot(30)).unwrap();
    assert_eq!(r.volume_ahead(&estimate), Err(QueueError::StaleEstimate));
}

#[test]
fn test_volume_ahead_of_an_own_order() {
    let mut r = booted(10);
    feed(
        &mut r,
        &[
            m(11, open("x", Buy, dec!(100), dec!(1))),
            m(12, open("own", Buy, dec!(100), dec!(2))),
            m(13, open("y", Buy, dec!(100), dec!(7))),
        ],
    );
    let estimate = r.queue_estimate_own("own").unwrap();
    let ahead = r.volume_ahead(&estimate).unwrap();
    // s1, s2 (snapshot) and x (observed) are ahead; y behind.
    assert_eq!(
        (ahead.volume, ahead.from_snapshot, ahead.observed),
        (dec!(4), dec!(3), dec!(1))
    );
    assert_eq!(ahead.assumptions, vec![Assumption::EntrySequenceIsPriority]);
    // An own order from the snapshot has unknown priority.
    assert!(matches!(
        r.queue_estimate_own("s2"),
        Err(QueueError::PriorityUnknown { .. })
    ));
    feed(&mut r, &[m(14, cancel("own", Buy, dec!(100), dec!(2)))]);
    assert!(matches!(
        r.volume_ahead(&estimate),
        Err(QueueError::NotResting { .. })
    ));
}

#[test]
fn test_priority_unknown_rule_refuses_estimates_after_a_decrease() {
    let rules = VenueRules {
        size_decrease: SizeDecrease::PriorityUnknown,
        ..VenueRules::default()
    };
    let mut r = BookReconstructor::new(P, rules);
    r.bootstrap(&snapshot(10)).unwrap();
    let estimate = r.queue_estimate(Buy, dec!(100)).unwrap();
    feed(
        &mut r,
        &[m(11, stp("s1", Buy, dec!(100), dec!(1), dec!(0.5)))],
    );
    assert!(matches!(
        r.volume_ahead(&estimate),
        Err(QueueError::PriorityUnknown { ref order_id }) if order_id == "s1"
    ));
    // Created after the decrease, the estimate stands.
    let later = r.queue_estimate(Buy, dec!(100)).unwrap();
    assert_eq!(r.volume_ahead(&later).unwrap().volume, dec!(2.5));
    // While invalidated, no estimate at all.
    let _ = r.apply(&m(20, received("z", Buy)));
    assert!(matches!(
        r.volume_ahead(&later),
        Err(QueueError::View(ViewError::Invalidated(_)))
    ));
    assert!(r.queue_estimate(Buy, dec!(100)).is_err());
}

/// A fixed linear congruential sequence.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

#[test]
fn test_random_valid_stream_keeps_every_invariant() {
    // 5 000 valid messages generated against an independent model of the
    // book (id -> side, price, size); invariants audited after each, and
    // the final L2 compared with the model's.
    use std::collections::BTreeMap;
    let mut rng = Lcg(77);
    let mut r = booted(0);
    let mut model: BTreeMap<String, (L3Side, Decimal, Decimal)> = BTreeMap::new();
    for o in snapshot(0).bids {
        model.insert(o.order_id, (Buy, o.price, o.size));
    }
    for o in snapshot(0).asks {
        model.insert(o.order_id, (Sell, o.price, o.size));
    }
    let mut next_id = 0_u64;
    for seq in 1..=5_000_u64 {
        let ids: Vec<String> = model.keys().cloned().collect();
        let pick = |rng: &mut Lcg| ids[(rng.next() as usize) % ids.len().max(1)].clone();
        let roll = rng.next() % 10;
        let event = if ids.is_empty() || roll < 4 {
            next_id += 1;
            let id = format!("o{next_id}");
            let side = if rng.next().is_multiple_of(2) {
                Buy
            } else {
                Sell
            };
            // Bids 90 to 100, asks 101 to 111: never crossed.
            let tick = Decimal::from(rng.next() % 11);
            let price = match side {
                Buy => dec!(90) + tick,
                Sell => dec!(101) + tick,
            };
            let size = Decimal::new((rng.next() % 1000 + 1) as i64, 2);
            model.insert(id.clone(), (side, price, size));
            open(&id, side, price, size)
        } else if roll < 7 {
            let id = pick(&mut rng);
            let (side, price, size) = model[&id];
            let take = (Decimal::new((rng.next() % 100 + 1) as i64, 2)).min(size);
            let left = size - take;
            if left == Decimal::ZERO {
                model.remove(&id);
            } else {
                model.insert(id.clone(), (side, price, left));
            }
            trade(&id, "taker", side, price, take)
        } else if roll < 9 {
            let id = pick(&mut rng);
            let (side, price, size) = model[&id];
            model.remove(&id);
            cancel(&id, side, price, size)
        } else {
            let id = pick(&mut rng);
            let (side, price, size) = model[&id];
            let new = size * dec!(0.5);
            if new == Decimal::ZERO {
                model.remove(&id);
            } else {
                model.insert(id.clone(), (side, price, new));
            }
            stp(&id, side, price, size, new)
        };
        r.apply(&m(seq, event)).unwrap();
        r.audit().unwrap();
    }
    let mut bids: BTreeMap<Decimal, Decimal> = BTreeMap::new();
    let mut asks: BTreeMap<Decimal, Decimal> = BTreeMap::new();
    for (side, price, size) in model.values() {
        let book = if *side == Buy { &mut bids } else { &mut asks };
        *book.entry(*price).or_default() += *size;
    }
    let expected_bids: Vec<_> = bids.into_iter().rev().collect();
    let expected_asks: Vec<_> = asks.into_iter().collect();
    assert_eq!(r.l2(Buy).unwrap(), expected_bids);
    assert_eq!(r.l2(Sell).unwrap(), expected_asks);
}

#[test]
fn test_sums_that_decimal_cannot_hold_exactly_are_refused() {
    // 1e28 + 1e-28 rounds in Decimal: never a silently inexact total.
    let big = Decimal::from_i128_with_scale(10_i128.pow(28), 0);
    let tiny = Decimal::new(1, 28);
    let mixed = L3Snapshot {
        product_id: P.to_string(),
        sequence: 1,
        bids: vec![so("a", dec!(100), big), so("t", dec!(100), tiny)],
        asks: vec![],
    };
    let mut r = BookReconstructor::new(P, VenueRules::default());
    assert!(matches!(
        r.bootstrap(&mixed),
        Err(SnapshotError::Arithmetic(_))
    ));
    // The same through messages: the open of the tiny order invalidates,
    // with nothing changed.
    let single = L3Snapshot {
        product_id: P.to_string(),
        sequence: 1,
        bids: vec![so("a", dec!(100), big)],
        asks: vec![],
    };
    r.bootstrap(&single).unwrap();
    let error = r.apply(&m(2, open("t", Buy, dec!(100), tiny))).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Invalidated(ref inv)
            if matches!(inv.kind, InvalidationKind::Discrepancy(Discrepancy::Arithmetic { .. }))
    ));
    assert_eq!(r.stale_l2(Buy).unwrap(), vec![(dec!(100), big)]);
    // A match whose remaining Decimal cannot hold exactly invalidates too.
    r.bootstrap(&single).unwrap();
    assert!(
        r.apply(&m(2, trade("a", "x", Buy, dec!(100), tiny)))
            .is_err()
    );
    // Exact mixed scales are fine.
    r.bootstrap(&snapshot(1)).unwrap();
    feed(&mut r, &[m(2, open("u", Buy, dec!(100), dec!(0.000001)))]);
    assert_eq!(r.l2(Buy).unwrap()[0], (dec!(100), dec!(3.000001)));
}

#[test]
fn test_own_decrease_under_priority_unknown_refuses_own_estimates() {
    let rules = VenueRules {
        size_decrease: SizeDecrease::PriorityUnknown,
        ..VenueRules::default()
    };
    let empty = L3Snapshot {
        product_id: P.to_string(),
        sequence: 0,
        bids: vec![],
        asks: vec![],
    };
    let mut r = BookReconstructor::new(P, rules);
    r.bootstrap(&empty).unwrap();
    feed(
        &mut r,
        &[
            m(1, open("own", Buy, dec!(100), dec!(2))),
            m(2, open("later", Buy, dec!(100), dec!(1))),
        ],
    );
    let estimate = r.queue_estimate_own("own").unwrap();
    assert_eq!(r.volume_ahead(&estimate).unwrap().volume, dec!(0));
    feed(
        &mut r,
        &[m(3, stp("own", Buy, dec!(100), dec!(2), dec!(1)))],
    );
    assert!(matches!(
        r.volume_ahead(&estimate),
        Err(QueueError::PriorityUnknown { ref order_id }) if order_id == "own"
    ));
    assert!(matches!(
        r.queue_estimate_own("own"),
        Err(QueueError::PriorityUnknown { .. })
    ));
    // Under KeepsPriority the same decrease is a declared assumption.
    let mut k = BookReconstructor::new(P, VenueRules::default());
    k.bootstrap(&empty).unwrap();
    feed(
        &mut k,
        &[
            m(1, open("own", Buy, dec!(100), dec!(2))),
            m(2, stp("own", Buy, dec!(100), dec!(2), dec!(1))),
        ],
    );
    let ahead = k
        .volume_ahead(&k.queue_estimate_own("own").unwrap())
        .unwrap();
    assert!(
        ahead
            .assumptions
            .contains(&Assumption::SizeDecreaseKeepsPriority)
    );
}

#[test]
fn test_estimates_belong_to_their_reconstructor_and_segment() {
    // Two reconstructors of the same product, same sequence, same ids.
    let a = booted(0);
    let mut b = BookReconstructor::new(P, VenueRules::default());
    let mut other = snapshot(0);
    other.bids[0].size = dec!(9);
    b.bootstrap(&other).unwrap();
    let hypothetical = a.queue_estimate(Buy, dec!(100)).unwrap();
    assert_eq!(
        b.volume_ahead(&hypothetical),
        Err(QueueError::StaleEstimate)
    );
    let mut a = a;
    feed(&mut a, &[m(1, open("own", Buy, dec!(100), dec!(1)))]);
    let own = a.queue_estimate_own("own").unwrap();
    let mut b2 = BookReconstructor::new(P, VenueRules::default());
    b2.bootstrap(&snapshot(0)).unwrap();
    feed(&mut b2, &[m(1, open("own", Buy, dec!(100), dec!(1)))]);
    assert_eq!(b2.volume_ahead(&own), Err(QueueError::StaleEstimate));
    // A clone is another reconstructor.
    let clone = a.clone();
    assert_eq!(clone.volume_ahead(&own), Err(QueueError::StaleEstimate));
    assert!(a.volume_ahead(&own).is_ok());
}

#[test]
fn test_a_reused_id_is_a_new_order_behind_the_estimate() {
    // s1 (size 1, ahead of the hypothetical order) is canceled, and its id
    // reopens later with size 9: that later order is behind.
    let mut r = booted(10);
    let estimate = r.queue_estimate(Buy, dec!(100)).unwrap();
    feed(
        &mut r,
        &[
            m(11, cancel("s1", Buy, dec!(100), dec!(1))),
            m(12, open("s1", Buy, dec!(100), dec!(9))),
        ],
    );
    let ahead = r.volume_ahead(&estimate).unwrap();
    assert_eq!((ahead.volume, ahead.orders), (dec!(2), 1));
    // Our own order canceled, and its id reused: not ours any more.
    feed(&mut r, &[m(13, open("own", Buy, dec!(100), dec!(1)))]);
    let own = r.queue_estimate_own("own").unwrap();
    feed(
        &mut r,
        &[
            m(14, cancel("own", Buy, dec!(100), dec!(1))),
            m(15, open("own", Buy, dec!(100), dec!(4))),
        ],
    );
    assert!(matches!(
        r.volume_ahead(&own),
        Err(QueueError::NotResting { .. })
    ));
    // A new estimate on the new order sees what is ahead of it now.
    let new = r.queue_estimate_own("own").unwrap();
    assert_eq!(r.volume_ahead(&new).unwrap().volume, dec!(11));
}

#[test]
fn test_rounding_open_match_and_stp_invalidate_instead_of_losing_size() {
    // A resting size whose next digit Decimal cannot hold: adding 1e-9,
    // matching 1e-9, or decreasing it to 1e-9 would round.
    let big = dec!(99999999999999999999.99999999);
    let small = dec!(0.000000001);
    let base = L3Snapshot {
        product_id: P.to_string(),
        sequence: 1,
        bids: vec![so("a", dec!(100), big)],
        asks: vec![],
    };
    let cases = [
        open("n", Buy, dec!(100), small),
        trade("a", "t", Buy, dec!(100), small),
        // The decrease big - 1e-9 is what Decimal cannot hold.
        stp("a", Buy, dec!(100), big, small),
    ];
    for (i, event) in cases.into_iter().enumerate() {
        let mut r = BookReconstructor::new(P, VenueRules::default());
        r.bootstrap(&base).unwrap();
        let error = r.apply(&m(2, event)).unwrap_err();
        assert!(
            matches!(
                error,
                ApplyError::Invalidated(ref inv) if matches!(
                    inv.kind,
                    InvalidationKind::Discrepancy(Discrepancy::Arithmetic { .. })
                )
            ),
            "case {i}: {error:?}"
        );
        assert_eq!(r.stale_l2(Buy).unwrap(), vec![(dec!(100), big)], "case {i}");
    }
}
