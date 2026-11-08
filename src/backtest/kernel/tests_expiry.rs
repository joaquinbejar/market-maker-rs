//! Quotes the kernel expires by itself (#297): exact in venue time, sharing an
//! open order's two venue messages, and kept in the state's encoding.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::book::{self, MsgKind, VenueMsg};
use super::*;
use crate::dec;
use crate::execution::Action;
use core::num::NonZeroU64;

const MS: u64 = 1_000_000;

fn at(ts_ns: u64, bid: u64, ask: u64) -> KernelTick {
    KernelTick {
        ts_ns,
        bid,
        bid_lots: 1,
        ask,
        ask_lots: 1,
    }
}

/// Rows every millisecond from 1000 ms that neither leg of a 10/20 quote crosses.
fn quiet(count: u64) -> Vec<KernelTick> {
    (0..count).map(|i| at((1_000 + i) * MS, 14, 16)).collect()
}

fn config(latency: KernelLatency) -> KernelConfig {
    KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100)
        .with_initial_capital(dec!(10000))
        .with_latency(latency)
}

fn expiring(bid: u64, ask: u64, life_ns: u64) -> KernelAction {
    KernelAction::ExpiringQuote {
        bid,
        ask,
        bid_lots: 1,
        ask_lots: 1,
        life_ns: NonZeroU64::new(life_ns).unwrap(),
    }
}

fn quote(bid: u64, ask: u64) -> KernelAction {
    KernelAction::Quote {
        bid,
        ask,
        bid_lots: 1,
        ask_lots: 1,
    }
}

/// Resets with `seed` and takes `actions`, then holds to the end.
fn run<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    k: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
    seed: u64,
    actions: &[KernelAction],
) -> (KernelState<ORD, MSG, DLV, OBS>, Vec<KernelEvent>) {
    let mut events = Vec::new();
    let mut s = k.reset_with(seed, &mut events);
    let mut decision = 0;
    while s.phase() == Phase::AwaitDecision {
        let action = actions.get(decision).copied().unwrap_or(KernelAction::Hold);
        k.step_with(&mut s, action, &mut events);
        decision += 1;
    }
    (s, events)
}

/// `(kind, ts_ns, cancel_pending)` of order `id`'s venue reports.
fn order_reports(events: &[KernelEvent], id: u64) -> Vec<(ExecKind, u64, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Report(r) if r.order_id == id => Some((r.kind, r.ts_ns, r.cancel_pending)),
            _ => None,
        })
        .collect()
}

fn expired(events: &[KernelEvent]) -> Vec<(u64, u64)> {
    events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Expired { order_id, at_ns } => Some((*order_id, *at_ns)),
            _ => None,
        })
        .collect()
}

fn has_expire(msgs: &[VenueMsg]) -> bool {
    msgs.iter()
        .any(|m| matches!(m.kind, MsgKind::Expire { .. }))
}

#[test]
fn test_a_leg_rests_exactly_its_life_from_its_activation_for_every_latency_and_jitter() {
    let life = 5 * MS + 7;
    let mut checked = 0;
    for (submit, cancel, ack, jitter) in [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (3, 1, 2, 0),
        (1, 4, 0, 2),
        (2, 2, 1, 3),
    ] {
        let latency = KernelLatency::zero(MS)
            .with_delays(0, submit, cancel, ack)
            .with_jitter(jitter);
        let cfg = config(latency);
        let data = KernelDataset::from_kernel_ticks(quiet(40), &cfg).unwrap();
        let k: FinancialKernel<'_> =
            FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
        for seed in 1..=8 {
            let (_, events) = run(&k, seed, &[expiring(10, 20, life)]);
            for id in [1, 2] {
                let reports = order_reports(&events, id);
                let accepted = reports
                    .iter()
                    .find(|r| r.0 == ExecKind::Accepted)
                    .expect("accepted")
                    .1;
                assert_eq!(
                    reports.iter().filter(|r| r.0 == ExecKind::Canceled).count(),
                    1
                );
                let canceled = reports.iter().find(|r| r.0 == ExecKind::Canceled).unwrap();
                assert_eq!(canceled.1, accepted + life, "{latency:?} seed {seed}");
                assert!(!canceled.2, "no cancel of the agent's in flight");
                assert!(expired(&events).contains(&(id, accepted + life)));
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 5 * 8 * 2);
}

#[test]
fn test_an_expiring_quote_draws_no_latency_and_admits_as_a_quote() {
    // With jitter, the expiry draws nothing: the same seed activates both quotes alike.
    let latency = KernelLatency::zero(MS)
        .with_delays(0, 2, 1, 1)
        .with_jitter(3);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(40), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let accepted = |action| {
        let (_, events) = run(&k, 4, &[action]);
        order_reports(&events, 1)
            .iter()
            .find(|r| r.0 == ExecKind::Accepted)
            .unwrap()
            .1
    };
    assert_eq!(accepted(expiring(10, 20, 3 * MS)), accepted(quote(10, 20)));
    for (bid, ask) in [(0, 20), (10, 2_000), (11, 11)] {
        let mut a = k.reset(1);
        let mut b = k.reset(1);
        let life = NonZeroU64::new(MS).unwrap();
        let expiring = KernelAction::ExpiringQuote {
            bid,
            ask,
            bid_lots: 1,
            ask_lots: 1,
            life_ns: life,
        };
        assert_eq!(
            k.step(&mut a, expiring).applied,
            k.step(&mut b, quote(bid, ask)).applied
        );
    }
}

/// A run whose bid of 15 activates at 1001 ms (submission 1 ms) and lives
/// 3 ms, over rows at 1000 ms, 1001 ms, `cross_ns` (an ask of 15 there) and
/// 1010 ms.
fn bid_fills_at(cross_ns: u64) -> bool {
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 0, 0);
    let cfg = config(latency);
    let mut rows = vec![at(1_000 * MS, 14, 16), at(1_001 * MS, 14, 16)];
    if cross_ns == 1_001 * MS {
        rows = vec![at(1_000 * MS, 14, 16), at(1_001 * MS, 13, 15)];
    } else {
        rows.push(at(cross_ns, 13, 15));
    }
    rows.push(at(1_010 * MS, 14, 16));
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let (s, events) = run(&k, 1, &[expiring(15, 30, 3 * MS)]);
    let accepted = order_reports(&events, 1)
        .iter()
        .find(|r| r.0 == ExecKind::Accepted)
        .unwrap()
        .1;
    assert_eq!(accepted, 1_001 * MS);
    s.counters().venue_fills > 0
}

#[test]
fn test_a_leg_fills_only_strictly_between_its_activation_and_its_expiry() {
    // As for any order, the row at the activation does not fill it.
    assert!(!bid_fills_at(1_001 * MS));
    assert!(bid_fills_at(1_001 * MS + 1));
    assert!(bid_fills_at(1_004 * MS - 1));
    // The expiry runs before the row at its time.
    assert!(!bid_fills_at(1_004 * MS));
    assert!(!bid_fills_at(1_005 * MS));
}

#[test]
fn test_an_expiry_before_the_agents_cancel_cancels_once_and_the_cancel_is_rejected() {
    // Activation at 1001 ms, expiry at 1004 ms; the agent's cancel, sent at 1001 ms with a
    // 5 ms delay, arrives at 1006 ms and finds the order gone.
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 5, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(20), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(1, &mut events);
    k.step_with(&mut s, expiring(10, 20, 3 * MS), &mut events);
    k.step_with(&mut s, KernelAction::Cancel, &mut events);
    let mut seen = false;
    while s.phase() == Phase::AwaitDecision {
        // Between the expiries' reports and the cancels' outcomes, the agent sees both
        // orders awaiting the outcome of its cancel, as after a fill that beat it.
        if s.now_ns == 1_005 * MS {
            assert_eq!(k.observe(&s).awaiting_cancel_outcome, 2);
            assert_eq!(s.account.len(), 2);
            seen = true;
        }
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert!(seen);
    assert_eq!(s.account.len(), 0);
    assert_eq!(
        order_reports(&events, 1),
        vec![
            (ExecKind::Accepted, 1_001 * MS, false),
            (ExecKind::Canceled, 1_004 * MS, true),
            (ExecKind::CancelRejected, 1_006 * MS, false),
        ]
    );
}

#[test]
fn test_the_agents_cancel_before_the_expiry_cancels_once_and_removes_the_expiry() {
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 1, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(20), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(1, &mut events);
    k.step_with(&mut s, expiring(10, 20, 9 * MS), &mut events);
    k.step_with(&mut s, KernelAction::Cancel, &mut events);
    // The cancel lands at 1002 ms; the expiries it replaced are gone at once.
    while s.phase() == Phase::AwaitDecision && s.now_ns < 1_003 * MS {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert!(!has_expire(s.msgs.as_slice()));
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert_eq!(
        order_reports(&events, 1),
        vec![
            (ExecKind::Accepted, 1_001 * MS, false),
            (ExecKind::Canceled, 1_002 * MS, false),
        ]
    );
    assert!(expired(&events).is_empty());
}

#[test]
fn test_at_equal_times_the_earlier_sequence_number_cancels() {
    // (1) The cancel is sent after the activation: the expiry, scheduled at the activation,
    // has the smaller sequence number and cancels; the agent's cancel is rejected.
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 3, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(20), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    // Activation at 1001 ms, expiry at 1004 ms; the cancel sent at 1001 ms lands at 1004 ms.
    let (_, events) = run(&k, 1, &[expiring(10, 20, 3 * MS), KernelAction::Cancel]);
    assert_eq!(
        order_reports(&events, 1),
        vec![
            (ExecKind::Accepted, 1_001 * MS, false),
            (ExecKind::Canceled, 1_004 * MS, true),
            (ExecKind::CancelRejected, 1_004 * MS, false),
        ]
    );
    // (2) The cancel is sent while the order is still on its way: it has the smaller
    // sequence number and cancels; the expiry goes with the order.
    let latency = KernelLatency::zero(MS).with_delays(0, 2, 3, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(20), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    // Activation at 1002 ms, expiry at 1004 ms; the cancel sent at 1001 ms lands at 1004 ms.
    let (_, events) = run(&k, 1, &[expiring(10, 20, 2 * MS), KernelAction::Cancel]);
    assert_eq!(
        order_reports(&events, 1),
        vec![
            (ExecKind::Accepted, 1_002 * MS, false),
            (ExecKind::Canceled, 1_004 * MS, false),
        ]
    );
    assert!(expired(&events).is_empty());
}

#[test]
fn test_a_fill_before_the_expiry_removes_it_and_a_partial_one_leaves_the_rest_to_expire() {
    // Immediate fill: the bid of 15 fills on the ask of 15 at 1002 ms, before its expiry.
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 0, 0);
    let cfg = config(latency);
    let rows = vec![
        at(1_000 * MS, 14, 16),
        at(1_001 * MS, 14, 16),
        at(1_002 * MS, 13, 15),
        at(1_020 * MS, 14, 16),
    ];
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(1, &mut events);
    k.step_with(&mut s, expiring(15, 30, 10 * MS), &mut events);
    k.step_with(&mut s, KernelAction::Hold, &mut events);
    // At the decision after the fill only the ask's expiry is left.
    let left: Vec<u64> = s
        .msgs
        .as_slice()
        .iter()
        .filter_map(|m| match m.kind {
            MsgKind::Expire { id } => Some(id),
            _ => None,
        })
        .collect();
    assert_eq!(left, vec![2]);
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert_eq!(expired(&events), vec![(2, 1_011 * MS)]);
    assert_eq!(
        order_reports(&events, 1)
            .iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        vec![ExecKind::Accepted, ExecKind::Fill]
    );
    // Queue position: two of five lots fill, then the expiry cancels the other three and
    // releases exactly their reservation.
    let cfg = config(latency).with_fill_model(KernelFillModel::QueuePosition {
        rate: dec!(1),
        min_queue_ms: 0,
    });
    let rows = vec![
        KernelTick {
            ts_ns: 1_000 * MS,
            bid: 14,
            bid_lots: 1,
            ask: 16,
            ask_lots: 1,
        },
        KernelTick {
            ts_ns: 1_001 * MS,
            bid: 14,
            bid_lots: 1,
            ask: 16,
            ask_lots: 1,
        },
        KernelTick {
            ts_ns: 1_003 * MS,
            bid: 13,
            bid_lots: 1,
            ask: 14,
            ask_lots: 1,
        },
        KernelTick {
            ts_ns: 1_020 * MS,
            bid: 13,
            bid_lots: 1,
            ask: 16,
            ask_lots: 1,
        },
    ];
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 16, 64, 256, 64> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let action = KernelAction::OneSided {
        side: QuoteSide::Bid,
        price: 14,
        lots: 5,
    };
    // A one-sided order has no expiry; an expiring quote's bid of 5 lots does.
    let (_, plain) = run(&k, 1, &[action]);
    assert!(expired(&plain).is_empty());
    let life = NonZeroU64::new(10 * MS).unwrap();
    let (s, events) = run(
        &k,
        1,
        &[KernelAction::ExpiringQuote {
            bid: 14,
            ask: 30,
            bid_lots: 5,
            ask_lots: 1,
            life_ns: life,
        }],
    );
    let bid: Vec<(ExecKind, u64)> = events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Report(r) if r.order_id == 1 && r.kind != ExecKind::Accepted => {
                Some((r.kind, r.cum))
            }
            _ => None,
        })
        .collect();
    let filled = bid
        .iter()
        .filter(|r| r.0 == ExecKind::PartialFill)
        .map(|r| r.1)
        .max()
        .unwrap();
    assert!(filled > 0 && filled < 5, "{bid:?}");
    assert_eq!(bid.last(), Some(&(ExecKind::Canceled, filled)));
    assert_eq!(expired(&events), vec![(1, 1_011 * MS), (2, 1_011 * MS)]);
    assert_eq!(s.gate.live_buy_lots, 0);
    assert_eq!(s.gate.live_sell_lots, 0);
}

#[test]
fn test_expiring_quotes_never_need_more_than_two_venue_messages_an_order() {
    // The smallest message ring for the orders: replacing and cancelling expiring quotes
    // with long lives, fills and cancels in flight never runs out of room.
    let latency = KernelLatency::zero(MS)
        .with_delays(0, 2, 3, 1)
        .with_jitter(2);
    let mut cfg = config(latency);
    cfg.max_steps = 2_000;
    let rows: Vec<KernelTick> = (0..600)
        .map(|i| {
            if i % 7 == 3 {
                at((1_000 + i) * MS, 13, 15)
            } else if i % 11 == 5 {
                at((1_000 + i) * MS, 15, 17)
            } else {
                at((1_000 + i) * MS, 14, 16)
            }
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 6, 12, 24, 64> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    for seed in 1..=6 {
        let actions: Vec<KernelAction> = (0..600u64)
            .map(|i| match i % 5 {
                0 => expiring(15, 16, (1 + i % 9) * MS),
                1 => KernelAction::Cancel,
                2 => expiring(14, 15, 40 * MS),
                3 => KernelAction::Hold,
                _ => quote(15, 17),
            })
            .collect();
        let (s, events) = run(&k, seed, &actions);
        assert_eq!(
            s.status(),
            Status::Terminal(TermReason::EndOfData),
            "seed {seed}"
        );
        assert!(!expired(&events).is_empty());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, KernelEvent::Report(r) if r.kind == ExecKind::Fill))
        );
    }
}

#[test]
fn test_an_order_cancelled_before_its_activation_never_expires() {
    // The cancel reaches the order while it is still on its way: no expiry is scheduled.
    let latency = KernelLatency::zero(MS).with_delays(0, 5, 1, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(30), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(1, &mut events);
    k.step_with(&mut s, expiring(10, 20, MS), &mut events);
    k.step_with(&mut s, KernelAction::Cancel, &mut events);
    while s.phase() == Phase::AwaitDecision {
        assert!(!has_expire(s.msgs.as_slice()));
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert!(expired(&events).is_empty());
    assert_eq!(
        order_reports(&events, 1)
            .iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        vec![ExecKind::Canceled]
    );
}

#[test]
fn test_an_expiry_beyond_the_clock_halts_before_the_order_is_accepted() {
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 0, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(10), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let (s, events) = run(&k, 1, &[expiring(10, 20, u64::MAX)]);
    assert_eq!(s.status(), Status::Halted(HaltReason::TimeOverflow));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, KernelEvent::Report(r) if r.kind == ExecKind::Accepted)),
        "an order accepted without its expiry"
    );
    assert!(s.orders.as_slice().is_empty());
}

#[test]
fn test_an_expiry_pending_at_the_end_of_the_data_is_dropped_as_an_expiry() {
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 0, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(5), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let (s, events) = run(&k, 1, &[expiring(10, 20, 60_000 * MS)]);
    let dropped: Vec<&KernelEvent> = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                KernelEvent::DroppedExpiry { .. } | KernelEvent::DroppedMsg { .. }
            )
        })
        .collect();
    assert_eq!(
        dropped,
        vec![
            &KernelEvent::DroppedExpiry { order_id: 1 },
            &KernelEvent::DroppedExpiry { order_id: 2 },
        ]
    );
    assert_eq!(s.counters().eof_dropped_msgs, 2);
    assert!(expired(&events).is_empty());
}

/// Every decision state of runs with expiring quotes, cancels and fills.
fn expiring_states() -> (KernelConfig, KernelDataset, Vec<KernelAction>) {
    let latency = KernelLatency::zero(MS)
        .with_delays(0, 2, 3, 1)
        .with_jitter(1);
    let cfg = config(latency);
    let rows: Vec<KernelTick> = (0..80)
        .map(|i| {
            if i % 6 == 4 {
                at((1_000 + i) * MS, 13, 15)
            } else {
                at((1_000 + i) * MS, 14, 16)
            }
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    // Quotes, a cancel racing an expiry, and holds long enough for every message to land.
    let actions = (0..80u64)
        .map(|i| match i % 16 {
            0 => expiring(15, 20, (2 + i % 5) * MS),
            1 | 3 => KernelAction::Hold,
            2 => expiring(15, 18, 3 * MS),
            4 => KernelAction::Cancel,
            _ => KernelAction::Hold,
        })
        .collect();
    (cfg, data, actions)
}

#[test]
fn test_a_state_with_expiries_round_trips_and_resumes_as_the_run_it_left() {
    let (cfg, data, actions) = expiring_states();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let (end, _) = run(&k, 2, &actions);
    // The terminal state at the end of the data decodes too.
    assert_eq!(end.status(), Status::Terminal(TermReason::EndOfData));
    assert_eq!(decode(&encode(&end), &k).unwrap(), end);
    let mut s = k.reset(2);
    let mut decision = 0;
    let (mut pending, mut resting, mut empty_after, mut awaiting) = (0, 0, 0, 0);
    while s.phase() == Phase::AwaitDecision {
        let bytes = encode(&s);
        let restored = decode(&bytes, &k).unwrap();
        assert_eq!(restored, s);
        let kinds = s.msgs.as_slice();
        pending += usize::from(
            kinds
                .iter()
                .any(|m| matches!(m.kind, MsgKind::Activate { life: Some(_), .. })),
        );
        resting += usize::from(has_expire(kinds));
        empty_after += usize::from(kinds.is_empty() && decision > 0);
        awaiting += usize::from(
            s.awaiting
                .as_slice()
                .iter()
                .any(|o| o.state == book::OrderState::Canceled),
        );
        // Resume from the decoded state: the rest of the run is the same.
        let mut resumed = restored;
        for action in actions.iter().skip(decision) {
            if resumed.phase() != Phase::AwaitDecision {
                break;
            }
            k.step(&mut resumed, *action);
        }
        while resumed.phase() == Phase::AwaitDecision {
            k.step(&mut resumed, KernelAction::Hold);
        }
        assert_eq!(
            encode(&resumed),
            encode(&end),
            "resumed at decision {decision}"
        );
        let action = actions.get(decision).copied().unwrap_or(KernelAction::Hold);
        k.step(&mut s, action);
        decision += 1;
    }
    assert!(
        pending > 0 && resting > 0 && empty_after > 0 && awaiting > 0,
        "{pending} {resting} {empty_after} {awaiting}"
    );
}

#[test]
fn test_a_state_without_expiries_keeps_its_bytes() {
    // The same run with plain quotes writes no tag 3 or 4: its bytes are the legacy ones,
    // which the golden fixtures pin; here, a state with a pending plain activation encodes
    // its message with tag 1 and decodes to the same state.
    let latency = KernelLatency::zero(MS).with_delays(0, 2, 1, 0);
    let cfg = config(latency);
    let data = KernelDataset::from_kernel_ticks(quiet(10), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut s = k.reset(1);
    k.step(&mut s, quote(10, 20));
    let msgs = s.msgs.as_slice();
    assert!(
        msgs.iter()
            .all(|m| matches!(m.kind, MsgKind::Activate { life: None, .. }))
    );
    assert!(!msgs.is_empty());
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
}

type Tamper = fn(&mut KernelState<16, 64, 128, 64>) -> bool;

#[test]
fn test_decode_refuses_each_inconsistent_expiry() {
    let (cfg, data, actions) = expiring_states();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut states = Vec::new();
    let mut s = k.reset(2);
    let mut decision = 0;
    while s.phase() == Phase::AwaitDecision {
        states.push(s);
        let action = actions.get(decision).copied().unwrap_or(KernelAction::Hold);
        k.step(&mut s, action);
        decision += 1;
    }
    let expiry_of = |s: &KernelState<16, 64, 128, 64>| {
        s.msgs
            .as_slice()
            .iter()
            .position(|m| matches!(m.kind, MsgKind::Expire { .. }))
    };
    let tampers: [(&str, Tamper); 8] = [
        // An activation time no other check bounds: the expiry's time, past the clock, is not
        // after it.
        ("an expiry not after its order's activation", |s| {
            let now = s.now_ns;
            let Some((id, eff)) = s.msgs.as_slice().iter().find_map(|m| match m.kind {
                MsgKind::Expire { id } if m.eff_ns > now => Some((id, m.eff_ns)),
                _ => None,
            }) else {
                return false;
            };
            s.orders.as_mut_slice().iter_mut().any(|o| {
                if o.id == id {
                    o.active_after_ns = eff;
                    true
                } else {
                    false
                }
            })
        }),
        ("a duplicate expiry", |s| {
            let Some(i) = s
                .msgs
                .as_slice()
                .iter()
                .position(|m| matches!(m.kind, MsgKind::Expire { .. }))
            else {
                return false;
            };
            let mut copy = *s.msgs.get(i).unwrap();
            copy.seq = s.next_msg_seq;
            s.next_msg_seq += 1;
            let at = s
                .msgs
                .as_slice()
                .iter()
                .position(|m| m.key() > copy.key())
                .unwrap_or(s.msgs.len());
            s.msgs.insert(at, copy).is_ok()
        }),
        ("an expiry of no order", |s| {
            let next = s.next_order_id;
            s.msgs.as_mut_slice().iter_mut().any(|m| match &mut m.kind {
                MsgKind::Expire { id } => {
                    *id = next + 5;
                    true
                }
                _ => false,
            })
        }),
        ("an expiry of an order on its way", |s| {
            let pending = s.msgs.as_slice().iter().find_map(|m| match m.kind {
                MsgKind::Activate { order, .. } => Some(order.id),
                _ => None,
            });
            let Some(pending) = pending else { return false };
            s.msgs.as_mut_slice().iter_mut().any(|m| match &mut m.kind {
                MsgKind::Expire { id } => {
                    *id = pending;
                    true
                }
                _ => false,
            })
        }),
        ("an expiry of an awaiting order", |s| {
            let Some(waiting) = s.awaiting.as_slice().first().map(|o| o.id) else {
                return false;
            };
            s.msgs.as_mut_slice().iter_mut().any(|m| match &mut m.kind {
                MsgKind::Expire { id } => {
                    *id = waiting;
                    true
                }
                _ => false,
            })
        }),
        ("an expiry at its order's activation", |s| {
            let Some(i) = s
                .msgs
                .as_slice()
                .iter()
                .position(|m| matches!(m.kind, MsgKind::Expire { .. }))
            else {
                return false;
            };
            let msg = s.msgs.remove(i).unwrap();
            let MsgKind::Expire { id } = msg.kind else {
                return false;
            };
            let Some(order) = s.orders.as_slice().iter().find(|o| o.id == id).copied() else {
                return false;
            };
            let moved = VenueMsg {
                eff_ns: order.active_after_ns,
                ..msg
            };
            let at = s
                .msgs
                .as_slice()
                .iter()
                .position(|m| m.key() > moved.key())
                .unwrap_or(s.msgs.len());
            s.msgs.insert(at, moved).is_ok()
        }),
        ("three venue messages for an order", |s| {
            let Some(id) = s.msgs.as_slice().iter().find_map(|m| match m.kind {
                MsgKind::Expire { id } => Some(id),
                _ => None,
            }) else {
                return false;
            };
            let has_cancel = s
                .msgs
                .as_slice()
                .iter()
                .any(|m| matches!(m.kind, MsgKind::Cancel { id: c } if c == id));
            if !has_cancel {
                return false;
            }
            let extra = VenueMsg {
                eff_ns: s.msgs.as_slice().last().unwrap().eff_ns,
                seq: s.next_msg_seq,
                kind: MsgKind::Cancel { id },
            };
            s.next_msg_seq += 1;
            s.msgs.push(extra).is_ok()
        }),
        ("an expired order awaiting no cancel", |s| {
            s.awaiting.as_mut_slice().iter_mut().any(|o| {
                if o.state == book::OrderState::Canceled {
                    o.cancel_inflight = false;
                    true
                } else {
                    false
                }
            })
        }),
    ];
    for (name, tamper) in tampers {
        let mut applied = false;
        for base in &states {
            let mut t = *base;
            if expiry_of(&t).is_none() && !name.contains("awaiting no cancel") {
                continue;
            }
            if tamper(&mut t) {
                assert!(decode(&encode(&t), &k).is_err(), "{name} decoded");
                applied = true;
            }
        }
        assert!(applied, "{name} never applicable");
    }
    // A pending activation with a life of 0, in the bytes: the life is the last field of a
    // tag 4 message, a value written nowhere else.
    let life = 0x0123_4567_89ab_cdef_u64;
    let cfg = config(KernelLatency::zero(MS).with_delays(0, 2, 0, 0));
    let data = KernelDataset::from_kernel_ticks(quiet(10), &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut s = k.reset(1);
    k.step(&mut s, expiring(10, 20, life));
    let mut bytes = encode(&s);
    let needle = life.to_le_bytes();
    let found: Vec<usize> = bytes
        .windows(8)
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(found.len(), 2, "one life per leg");
    assert!(decode(&bytes, &k).is_ok());
    for i in found {
        bytes[i..i + 8].copy_from_slice(&0u64.to_le_bytes());
    }
    let error = decode(&bytes, &k).unwrap_err();
    assert!(error.to_string().contains("life of 0"), "{error}");
    let _ = Action::Buy;
}

#[test]
fn test_under_the_queue_model_an_expiring_order_may_take_four_stamps() {
    // Each leg: its submission, its activation, the expiry's cancel and the agent's rejected
    // cancel, with no fill: four stamps an order, which the decoder accepts.
    let latency = KernelLatency::zero(MS).with_delays(0, 1, 5, 0);
    let cfg = config(latency).with_fill_model(KernelFillModel::QueuePosition {
        rate: dec!(1),
        min_queue_ms: 0,
    });
    let data = KernelDataset::from_kernel_ticks(quiet(30), &cfg).unwrap();
    let k: FinancialKernel<'_, 16, 64, 256, 64> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut s = k.reset(1);
    k.step(&mut s, expiring(10, 20, 3 * MS));
    k.step(&mut s, KernelAction::Cancel);
    while s.phase() == Phase::AwaitDecision && !s.account.is_empty() {
        k.step(&mut s, KernelAction::Hold);
    }
    let created = s.next_order_id() - 1;
    assert_eq!(created, 2);
    assert_eq!(s.counters().venue_fills, 0);
    assert!(s.stamp_seq > 3 * created, "{} stamps", s.stamp_seq);
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
}
