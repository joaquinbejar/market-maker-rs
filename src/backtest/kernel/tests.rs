//! Unit tests that need the test-only relaxed constructor or private
//! internals: halts, exact capacity rejects, the venue reject path, and the
//! gate and fee unit equivalence.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::*;
use crate::backtest::engine::FeeCurrency;
use crate::dec;
use crate::execution::Action;
use crate::risk::RiskLimits;
use crate::risk::pretrade::{
    LimitsPolicy, OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::types::action::Dir;

const MS: u64 = 1_000_000;

fn tick(ts_ms: u64, bid: u64, ask: u64) -> KernelTick {
    KernelTick {
        ts_ns: ts_ms * MS,
        bid,
        bid_lots: 1,
        ask,
        ask_lots: 1,
    }
}

fn config(latency: KernelLatency) -> KernelConfig {
    KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100)
        .with_initial_capital(dec!(10000))
        .with_latency(latency)
}

fn quote(bid: u64, ask: u64) -> KernelAction {
    KernelAction::Quote {
        bid,
        ask,
        bid_lots: 1,
        ask_lots: 1,
    }
}

fn reports(events: &[KernelEvent], delivered: bool) -> Vec<ReportRec> {
    events
        .iter()
        .filter_map(|e| match (e, delivered) {
            (KernelEvent::Report(r), false) | (KernelEvent::Delivered(r), true) => Some(*r),
            _ => None,
        })
        .collect()
}

#[test]
fn test_kernel_ledger_overflow_halts_with_ledger_intact() {
    // Outside the bounds (relaxed constructor): the notional of one fill
    // overflows i128 money units.
    let p: u64 = 9_000_000_000_000_000_000;
    let cfg = KernelConfig::new(dec!(1), dec!(1), Decimal::from(p), Decimal::from(p), 10)
        .with_fee_rate(dec!(0.1));
    let data =
        KernelDataset::from_kernel_ticks(vec![tick(1, p - 2, p - 1), tick(2, p - 2, p - 1)], &cfg)
            .unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, true, false, false).unwrap();
    let mut s = k.reset(1);
    let action = KernelAction::Quote {
        bid: p - 1,
        ask: p,
        bid_lots: p,
        ask_lots: 1,
    };
    let before = s.venue;
    let gate_before = s.gate;
    let mut events = Vec::new();
    k.step_with(&mut s, action, &mut events);
    assert_eq!(s.status(), Status::Halted(HaltReason::LedgerOverflow));
    assert_eq!(s.phase(), Phase::Terminal);
    assert_eq!(s.counters().halts, 1);
    // Row 0 was marked at its mid; the fill on row 1 left no trace.
    let marked = ledger::VenueLedger {
        last_mark2: Some(2 * p - 3),
        ..before
    };
    assert_eq!(s.venue, marked, "the ledger keeps its pre-fill value");
    assert_eq!(s.gate.venue_pos_lots, gate_before.venue_pos_lots);
    assert!(events.contains(&KernelEvent::Halted(HaltReason::LedgerOverflow)));
    // A halted state stays halted and unchanged.
    let snapshot = s;
    assert_eq!(k.step(&mut s, KernelAction::Hold).events, 0);
    assert_eq!(s, snapshot);
}

#[test]
fn test_kernel_new_rejects_ledger_overflow_config() {
    let p: u64 = 9_000_000_000_000_000_000;
    let cfg = KernelConfig::new(dec!(1), dec!(1), Decimal::from(p), Decimal::from(p), 10)
        .with_fee_rate(dec!(0.1));
    let data = KernelDataset::from_kernel_ticks(vec![tick(1, p - 2, p - 1)], &cfg).unwrap();
    let built: MMResult<FinancialKernel<'_>> = FinancialKernel::new(&cfg, &data);
    assert!(built.is_err());
}

/// Rows that never cross a quote at (10, 20); long cancel delay so every
/// replaced order stays open in the account.
fn capacity_run<const ORD: usize, const MSG: usize, const DLV: usize>()
-> (Vec<ActionStatus>, Vec<Vec<u8>>) {
    let latency = KernelLatency::zero(MS).with_delays(0, 50, 100, 0);
    let cfg = config(latency);
    let rows = (0..12).map(|i| tick(1_000 + i, 14, 16)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, ORD, MSG, DLV, 16> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut s = k.reset(3);
    let mut outcomes = Vec::new();
    let mut encodings = Vec::new();
    while s.phase() == Phase::AwaitDecision {
        let before = encode(&s);
        let out = k.step(&mut s, quote(10, 20));
        outcomes.push(out.applied);
        encodings.push(before);
        let q = s.queue_lengths();
        assert!(
            q.orders + q.awaiting <= q.held && q.msgs <= 2 * q.held && q.deliveries <= 3 * q.held
        );
    }
    (outcomes, encodings)
}

#[test]
fn test_kernel_capacity_orders_exact_limit_rejects_next() {
    // ORD = 6: held 0, 2, 4 admit (4 + 2 = 6), held 6 rejects.
    let (outcomes, _) = capacity_run::<6, 64, 64>();
    assert_eq!(outcomes[..3], [ActionStatus::Applied; 3]);
    assert_eq!(
        outcomes[3],
        ActionStatus::Rejected(RejectReason::CapacityOrders)
    );
}

#[test]
fn test_kernel_capacity_msgs_exact_limit_rejects_next() {
    // MSG = 12: 2 * (held + 2) <= 12 admits up to held 4.
    let (outcomes, _) = capacity_run::<32, 12, 128>();
    assert_eq!(outcomes[..3], [ActionStatus::Applied; 3]);
    assert_eq!(
        outcomes[3],
        ActionStatus::Rejected(RejectReason::CapacityMsgs)
    );
}

#[test]
fn test_kernel_capacity_deliveries_exact_limit_rejects_next() {
    // DLV = 15: 3 * (held + 2) <= 15 admits up to held 3, so held 0 and 2.
    let (outcomes, _) = capacity_run::<32, 64, 15>();
    assert_eq!(outcomes[..2], [ActionStatus::Applied; 2]);
    assert_eq!(
        outcomes[2],
        ActionStatus::Rejected(RejectReason::CapacityDeliveries)
    );
}

#[test]
fn test_kernel_capacity_reject_leaves_queues_unchanged() {
    let latency = KernelLatency::zero(MS).with_delays(0, 50, 100, 0);
    let cfg = config(latency);
    let rows = (0..12).map(|i| tick(1_000 + i, 14, 16)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 6, 64, 64, 16> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut s = k.reset(3);
    for _ in 0..3 {
        k.step(&mut s, quote(10, 20));
    }
    // Pending activations and their cancels are both in the ring.
    let pending = s
        .msgs
        .as_slice()
        .iter()
        .filter(|m| matches!(m.kind, book::MsgKind::Activate { order: o, .. } if o.cancel_inflight))
        .count();
    assert!(pending > 0);
    let mut a = s;
    let mut b = s;
    assert_eq!(
        k.step(&mut a, quote(10, 20)).applied,
        ActionStatus::Rejected(RejectReason::CapacityOrders)
    );
    k.step(&mut b, KernelAction::Hold);
    assert_eq!(a.msgs, b.msgs);
    assert_eq!(a.orders, b.orders);
    assert_eq!(a.deliveries, b.deliveries);
    assert_eq!(a.account, b.account);
}

#[test]
fn test_kernel_venue_reject_with_cancel_in_flight_holds_capacity_until_outcome() {
    // The engine fixture: bid 0 is rejected by venue validation at 1003;
    // the cancel decided at 1001 lands at 1006 with CancelRejected.
    let latency = KernelLatency::zero(MS).with_delays(0, 3, 5, 0);
    let cfg = config(latency);
    let rows = (0..9).map(|i| tick(1_000 + i, 98, 102)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 4, 8, 12, 8> =
        FinancialKernel::new_relaxed(&cfg, &data, false, true, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(0, &mut events);
    assert_eq!(
        k.step_with(&mut s, quote(0, 101), &mut events).applied,
        ActionStatus::Applied
    );
    assert_eq!(
        k.step_with(&mut s, quote(99, 101), &mut events).applied,
        ActionStatus::Applied
    );
    let mut refused_while_open = 0;
    let mut admitted_after = None;
    while s.phase() == Phase::AwaitDecision {
        let open_one = s.account.as_slice().iter().any(|o| o.order_id == 1);
        let out = k.step_with(&mut s, quote(97, 103), &mut events);
        let q = s.queue_lengths();
        assert!(
            q.orders + q.awaiting <= q.held && q.msgs <= 2 * q.held && q.deliveries <= 3 * q.held
        );
        if open_one {
            assert_eq!(
                out.applied,
                ActionStatus::Rejected(RejectReason::CapacityOrders)
            );
            refused_while_open += 1;
        } else if admitted_after.is_none() {
            assert_eq!(out.applied, ActionStatus::Applied);
            admitted_after = Some(s.now_ns());
        }
    }
    let one: Vec<_> = reports(&events, true)
        .into_iter()
        .filter(|r| r.order_id == 1)
        .map(|r| (r.ts_ns / MS, r.kind, r.origin, r.cancel_pending))
        .collect();
    assert_eq!(
        one,
        vec![
            (1003, ExecKind::Rejected, Origin::Venue, true),
            (1006, ExecKind::CancelRejected, Origin::Venue, false),
        ]
    );
    let two: Vec<_> = reports(&events, true)
        .into_iter()
        .filter(|r| r.order_id == 2)
        .map(|r| (r.ts_ns / MS, r.kind))
        .collect();
    assert_eq!(
        two,
        vec![(1003, ExecKind::Accepted), (1006, ExecKind::Canceled)]
    );
    assert!(refused_while_open > 0);
    assert!(admitted_after.is_some());
}

#[test]
fn test_kernel_local_reject_closes_on_delivery() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 2);
    let cfg = config(latency);
    let rows = (0..6).map(|i| tick(1_000 + i, 98, 102)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut s = k.reset(0);
    k.step(&mut s, quote(101, 100));
    assert_eq!(s.queue_lengths().held, 2);
    while s.phase() == Phase::AwaitDecision && s.queue_lengths().held > 0 {
        k.step(&mut s, KernelAction::Hold);
    }
    assert_eq!(s.queue_lengths().held, 0);
    assert_eq!(k.observe(&s).delivered_local_rejects, 2);
}

#[test]
fn test_kernel_state_from_another_dataset_halts_with_mismatch() {
    let cfg = config(KernelLatency::zero(MS));
    let a = KernelDataset::from_kernel_ticks((0..3).map(|i| tick(1 + i, 98, 102)).collect(), &cfg)
        .unwrap();
    let b = KernelDataset::from_kernel_ticks((0..3).map(|i| tick(1 + i, 97, 102)).collect(), &cfg)
        .unwrap();
    let ka: FinancialKernel<'_> = FinancialKernel::new(&cfg, &a).unwrap();
    let kb: FinancialKernel<'_> = FinancialKernel::new(&cfg, &b).unwrap();
    let mut s = ka.reset(0);
    let before = s;
    let out = kb.step(&mut s, KernelAction::Hold);
    assert_eq!(out.status, Status::Halted(HaltReason::StateMismatch));
    assert_eq!(s.counters().halts, 1);
    assert_eq!(s.step_count(), before.step_count());
    assert_eq!(s.venue, before.venue);
    assert_eq!(s.deliveries, before.deliveries);
    assert!(decode(&encode(&before), &kb).is_err());
}

#[test]
fn test_kernel_gate_matches_limits_policy_over_seeded_sequences() {
    let mut x: u64 = 0x1234_5678;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for case in 0..400 {
        let lot = [dec!(1), dec!(0.1), dec!(0.00000001)][case % 3];
        let tick_size = [dec!(0.01), dec!(1), dec!(0.00000001)][(case / 3) % 3];
        let lot_u = grid::Unit::new(lot).unwrap();
        let tick_u = grid::Unit::new(tick_size).unwrap();
        let max_position = if case % 7 == 0 {
            Decimal::MAX
        } else {
            lot * Decimal::from(1 + next() % 20)
        };
        let max_notional = match case % 5 {
            0 => Decimal::MAX,
            1 => tick_size * lot * Decimal::from(1 + next() % 5_000) / dec!(3),
            _ => tick_size * lot * Decimal::from(1 + next() % 50_000),
        };
        let limits = RiskLimits::new(max_position, max_notional, dec!(0)).unwrap();
        let mut policy = LimitsPolicy::new(limits.clone());
        let mut gate = gate::GateState::NEW;
        let mut live: Vec<(u64, Dir, u64)> = Vec::new();
        for id in 1..60u64 {
            let dir = if next() % 2 == 0 { Dir::Buy } else { Dir::Sell };
            match next() % 3 {
                // A fill of a live order (execution ids are never replayed).
                0 if !live.is_empty() => {
                    let i = usize::try_from(next() % live.len() as u64).unwrap();
                    let (oid, d, lots) = live.remove(i);
                    let qty = lot_u.to_decimal(i128::from(lots)).unwrap();
                    policy
                        .on_order_event(&OrderEvent::Fill {
                            order_id: oid.to_string(),
                            execution_id: format!("sim-{oid}"),
                            side: d.into(),
                            price: tick_size,
                            quantity: qty,
                            cumulative_quantity: qty,
                            fee: Decimal::ZERO,
                            fee_asset: crate::backtest::FeeAsset::Quote,
                            timestamp: 0,
                        })
                        .unwrap();
                    gate.fill(d, lots).unwrap();
                }
                // A cancel of a live order.
                1 if !live.is_empty() => {
                    let i = usize::try_from(next() % live.len() as u64).unwrap();
                    let (oid, d, lots) = live.remove(i);
                    policy
                        .on_order_event(&OrderEvent::Canceled {
                            order_id: oid.to_string(),
                        })
                        .unwrap();
                    gate.release(d, lots).unwrap();
                }
                _ => {
                    let price = 1 + next() % 100_000;
                    let lots = 1 + next() % 10;
                    let request = PretradeRequest::new(
                        id.to_string(),
                        dir.into(),
                        Some(tick_u.to_decimal(i128::from(price)).unwrap()),
                        lot_u.to_decimal(i128::from(lots)).unwrap(),
                        OrderPurpose::Quote,
                        0,
                    );
                    let decision = policy.check(&request);
                    let ours =
                        gate::check(&limits, &gate, &tick_u, &lot_u, dir, price, lots).unwrap();
                    let approved = decision.approved_quantity(request.quantity);
                    match (&decision, ours.verdict) {
                        (PretradeDecision::Accepted, GateVerdict::Accepted) => {}
                        (PretradeDecision::Reduced { quantity, .. }, GateVerdict::Reduced) => {
                            assert_eq!(
                                lot_u.to_decimal(i128::from(ours.submitted)).unwrap(),
                                *quantity
                            );
                        }
                        (PretradeDecision::Reduced { .. }, GateVerdict::ReducedOffLot) => {
                            policy
                                .on_order_event(&OrderEvent::VenueRejected {
                                    order_id: id.to_string(),
                                })
                                .unwrap();
                        }
                        (PretradeDecision::Rejected { .. }, GateVerdict::Rejected) => {}
                        (d, v) => panic!("case {case}: policy {d:?} vs gate {v:?}"),
                    }
                    if ours.submitted > 0 {
                        assert_eq!(
                            lot_u.to_decimal(i128::from(ours.submitted)).unwrap(),
                            approved
                        );
                        gate.reserve(dir, ours.submitted).unwrap();
                        live.push((id, dir, ours.submitted));
                    }
                }
            }
            for d in [Dir::Buy, Dir::Sell] {
                assert_eq!(
                    policy.live_quantity(d.into()).unwrap(),
                    lot_u.to_decimal(i128::from(gate.live(d))).unwrap(),
                    "case {case}: live {d:?}"
                );
            }
            assert_eq!(
                policy.position(),
                lot_u.to_decimal(i128::from(gate.venue_pos_lots)).unwrap()
            );
        }
    }
}

#[test]
fn test_kernel_limits_capacity_tiny_price_max_notional_is_non_binding() {
    let limits = RiskLimits::new(Decimal::MAX, Decimal::MAX, dec!(0)).unwrap();
    assert!(limits.max_notional.checked_div(dec!(0.00000001)).is_none());
    let verdict = crate::risk::pretrade::limits_capacity(
        &limits,
        dec!(5),
        dec!(1),
        Dir::Buy,
        dec!(0.00000001),
        dec!(1),
    )
    .unwrap();
    assert_eq!(verdict, crate::risk::pretrade::CapacityVerdict::Accept);
    // The position room itself overflows for a short position at MAX.
    let short = Decimal::MIN + dec!(1);
    let verdict =
        crate::risk::pretrade::limits_capacity(&limits, short, dec!(0), Dir::Buy, dec!(1), dec!(1))
            .unwrap();
    assert_eq!(verdict, crate::risk::pretrade::CapacityVerdict::Accept);
}

#[test]
fn test_kernel_fee_units_match_fee_currency_charge() {
    let mut x: u64 = 99;
    for case in 0..500u64 {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tick_size = [dec!(0.01), dec!(0.25), dec!(1)][usize::try_from(case % 3).unwrap()];
        let lot = [dec!(1), dec!(0.001), dec!(0.5)][usize::try_from((case / 3) % 3).unwrap()];
        let rate = [dec!(0.0001), dec!(-0.00005), dec!(0.001), dec!(0)]
            [usize::try_from((case / 9) % 4).unwrap()];
        let cfg = KernelConfig::new(
            tick_size,
            lot,
            Decimal::from(10_000) * tick_size,
            Decimal::from(100) * lot,
            10,
        )
        .with_fee_rate(rate);
        let v = config::validate(&cfg, config::Relax::default()).unwrap();
        let p = 1 + (x >> 20) % 10_000;
        let q = 1 + (x >> 50) % 100;
        let price = Decimal::from(p) * tick_size;
        let qty = Decimal::from(q) * lot;
        let charge = FeeCurrency::Quote.charge(price, qty, rate).unwrap();
        let ours = grid::scaled_decimal(v.fee_u(p, q).unwrap(), v.s).unwrap();
        assert_eq!(ours, charge.quote_value, "case {case}");
        let notional = grid::scaled_decimal(v.notional_u(p, q).unwrap(), v.s).unwrap();
        assert_eq!(notional, price * qty);
    }
}

/// States of a run with resting orders, pending activations, awaiting
/// orders and queued deliveries.
fn busy_states() -> (KernelConfig, KernelDataset) {
    let latency = KernelLatency::zero(MS).with_delays(1, 5, 8, 3);
    let cfg = config(latency);
    let rows = (0..40)
        .map(|i| tick(1_000 + 3 * i, 96 + i % 4, 100 + i % 3))
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    (cfg, data)
}

fn collect_states<'d>(k: &FinancialKernel<'d>) -> Vec<KernelState<16, 64, 128, 64>> {
    let mut s = k.reset(9);
    let mut out = Vec::new();
    let mut i = 0u64;
    while s.phase() == Phase::AwaitDecision {
        out.push(s);
        let action = match i % 4 {
            0 => quote(101, 103),
            1 => quote(95, 99),
            2 => KernelAction::Hold,
            _ => quote(100, 99),
        };
        k.step(&mut s, action);
        i += 1;
    }
    out
}

type Mutation = fn(&mut KernelState<16, 64, 128, 64>, &FinancialKernel<'_>) -> bool;

#[test]
fn test_kernel_decode_rejects_each_unreachable_state() {
    let (cfg, data) = busy_states();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let states = collect_states(&k);
    for s in &states {
        assert_eq!(&decode(&encode(s), &k).unwrap(), s);
    }
    let mutations: [(&str, Mutation); 27] = [
        ("stamp_seq at u64::MAX", |s, _| {
            s.stamp_seq = u64::MAX;
            true
        }),
        ("stamp_seq below its floor", |s, _| {
            if s.stamp_seq == 0 {
                return false;
            }
            s.stamp_seq =
                s.next_order_id - 1 + s.counters.local_rejects + s.counters.venue_fills - 1;
            true
        }),
        ("next_msg_seq beyond three per order", |s, _| {
            s.next_msg_seq = 3 * s.next_order_id + 1;
            true
        }),
        ("next_order_id beyond two per step", |s, _| {
            s.next_order_id = 2 * s.step + 2;
            true
        }),
        ("next_exec_id beyond the fills", |s, _| {
            s.next_exec_id += 1;
            true
        }),
        ("local rejects beyond the orders", |s, _| {
            s.counters.local_rejects = s.next_order_id;
            true
        }),
        ("quote errors without rejects", |s, _| {
            s.counters.quote_errors += 1;
            true
        }),
        ("drops before the end of the data", |s, _| {
            s.counters.eof_dropped_msgs = 1;
            true
        }),
        ("last delivery far in the future", |s, _| {
            s.last_delivery_ns = u64::MAX;
            true
        }),
        ("last observation far in the future", |s, _| {
            s.last_observed_ns = u64::MAX;
            true
        }),
        ("venue message far in the future", |s, _| {
            s.msgs.get_mut(0).map(|m| m.eff_ns = u64::MAX).is_some()
        }),
        ("venue average entry Decimal::MAX", |s, _| {
            if s.venue.pos.quantity.is_zero() {
                return false;
            }
            s.venue.pos.avg_entry = Decimal::MAX;
            true
        }),
        ("agent average entry negative", |s, _| {
            if s.agent.pos.quantity.is_zero() {
                return false;
            }
            s.agent.pos.avg_entry = dec!(-1);
            true
        }),
        ("venue average entry zero on a position", |s, _| {
            if s.venue.pos.quantity.is_zero() {
                return false;
            }
            s.venue.pos.avg_entry = Decimal::ZERO;
            true
        }),
        ("agent average entry above the price range", |s, k| {
            if s.agent.pos.quantity.is_zero() {
                return false;
            }
            s.agent.pos.avg_entry = k.price(k.cfg.p).unwrap() + Decimal::ONE;
            true
        }),
        ("resting order price 2^63", |s, _| {
            s.orders.get_mut(0).map(|o| o.price = 1 << 63).is_some()
        }),
        ("activation quantity 0", |s, _| {
            s.msgs.as_mut_slice().iter_mut().any(|m| match &mut m.kind {
                book::MsgKind::Activate { order: o, .. } => {
                    o.qty = 0;
                    true
                }
                book::MsgKind::Cancel { .. } | book::MsgKind::Expire { .. } => false,
            })
        }),
        ("awaiting price above P", |s, k| {
            let p = k.cfg.p;
            s.awaiting.get_mut(0).map(|o| o.price = p + 1).is_some()
        }),
        ("report price above P", |s, k| {
            let p = k.cfg.p;
            s.deliveries
                .get_mut(0)
                .map(|d| d.report.price = p + 1)
                .is_some()
        }),
        ("fill quantity above Q", |s, k| {
            let q = k.cfg.q;
            s.deliveries
                .as_mut_slice()
                .iter_mut()
                .any(|d| match &mut d.fill {
                    Some(f) => {
                        f.qty = q + 1;
                        true
                    }
                    None => false,
                })
        }),
        ("delivered fills", |s, _| {
            s.agent.delivered_fills += 1;
            true
        }),
        ("delivered local rejects", |s, _| {
            s.agent.delivered_local_rejects += 1;
            true
        }),
        ("agent position", |s, _| {
            s.agent.pos.quantity += Decimal::ONE;
            true
        }),
        ("open order without counterpart", |s, _| {
            let id = (1..s.next_order_id).find(|id| !s.is_open(*id));
            match id {
                Some(id) => {
                    let at = s
                        .account
                        .position(|o| o.order_id > id)
                        .unwrap_or(s.account.len());
                    s.account
                        .insert(
                            at,
                            account::OpenOrder {
                                order_id: id,
                                awaiting_cancel_outcome: false,
                            },
                        )
                        .is_ok()
                }
                None => false,
            }
        }),
        ("running at max_steps", |s, k| {
            s.step = k.cfg.s_max;
            s.counters.actions = k.cfg.s_max;
            true
        }),
        ("maximum position beyond X_eff", |s, k| {
            s.venue.max_abs_pos = k.cfg.x_eff + 1;
            true
        }),
        ("time after the current row", |s, k| {
            match s.cur_row.and_then(|r| k.data.row(r)) {
                Some(row) => {
                    s.now_ns = row.ts_ns + 1;
                    true
                }
                None => false,
            }
        }),
    ];
    for (name, mutate) in mutations {
        let mut applied = false;
        for base in &states {
            let mut s = *base;
            if mutate(&mut s, &k) {
                assert!(decode(&encode(&s), &k).is_err(), "{name} decoded");
                applied = true;
            }
        }
        assert!(applied, "{name} never applicable");
    }
}

#[test]
fn test_kernel_mask_matches_admission_under_capacity_pressure() {
    // Small capacities, no capacity check: the account fills up and the
    // mask's refusal branch must agree with admission on 256 quotes.
    let latency = KernelLatency::zero(MS).with_delays(0, 3, 6, 2);
    let cfg = config(latency);
    let rows = (0..30)
        .map(|i| tick(1_000 + i, 97 + i % 3, 101 + i % 2))
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 6, 12, 18, 8> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let grid = ActionGrid::new([0, 1, 3, 2_000], [0, 2, 4, 3_000], [1, 2, 10, 11]).unwrap();
    let mut s = k.reset(5);
    let (mut refused, mut admitted, mut decisions) = (0u64, 0u64, 0u64);
    while s.phase() == Phase::AwaitDecision {
        let mask = k.mask(&s, &grid);
        let mut probes = 0;
        for b in 0..4 {
            for a in 0..4 {
                for x in 0..4 {
                    for y in 0..4 {
                        let allowed = mask.allows(b, a, x, y);
                        let applied = match k.grid_action(&s, &grid, (b, a, x, y)) {
                            Some(action) => {
                                let mut c = s;
                                let out = k.step(&mut c, action).applied;
                                if let (Some(reason), ActionStatus::Rejected(got)) =
                                    (mask.refusal, out)
                                    && matches!(
                                        got,
                                        RejectReason::CapacityOrders
                                            | RejectReason::CapacityMsgs
                                            | RejectReason::CapacityDeliveries
                                    )
                                {
                                    assert_eq!(got, reason);
                                }
                                out == ActionStatus::Applied
                            }
                            None => false,
                        };
                        assert_eq!(applied, allowed, "mask {mask:?} at {:?}", (b, a, x, y));
                        admitted += u64::from(applied);
                        probes += 1;
                    }
                }
            }
        }
        assert_eq!(probes, 256);
        if mask.refusal.is_some() {
            assert!(!mask.global_quote_ok);
            refused += 1;
        }
        decisions += 1;
        k.step(&mut s, quote(90, 110));
    }
    assert!(refused > 0, "capacity refusal never reached");
    assert!(admitted > 0 && decisions > refused);
}

#[test]
fn test_kernel_config_rejects_trailing_zero_scales() {
    let base = config(KernelLatency::zero(MS));
    let mut tick_zeros = base.clone();
    tick_zeros.tick_size = dec!(1.000);
    let mut lot_zeros = base.clone();
    lot_zeros.lot_size = dec!(0.0010);
    let mut fee_zeros = base.clone();
    fee_zeros.fee_rate = dec!(0.00010);
    let mut capital = base.clone();
    capital.initial_capital = dec!(1000.000);
    for c in [tick_zeros, lot_zeros, fee_zeros, capital] {
        assert!(
            config::validate(&c, config::Relax::default()).is_err(),
            "{c:?}"
        );
    }
    // Dataset prices and sizes written beyond the grid scale.
    let ticks = [crate::backtest::MarketTick::new(
        1,
        dec!(98.0),
        dec!(1),
        dec!(102),
        dec!(1),
    )];
    assert!(KernelDataset::from_ticks(&ticks, &base).is_err());
    let ticks = [crate::backtest::MarketTick::new(
        1,
        dec!(98),
        dec!(1.00),
        dec!(102),
        dec!(1),
    )];
    assert!(KernelDataset::from_ticks(&ticks, &base).is_err());
    let ticks = [crate::backtest::MarketTick::new(
        1,
        dec!(98),
        dec!(1),
        dec!(102),
        dec!(1),
    )];
    assert!(KernelDataset::from_ticks(&ticks, &base).is_ok());
}

#[test]
fn test_kernel_dataset_digest_is_unchanged() {
    // Pinned before the digest stopped allocating a writer per row.
    let cfg = KernelConfig::new(dec!(0.25), dec!(0.1), dec!(1000), dec!(10), 10);
    let rows = (0..5u64)
        .map(|i| KernelTick {
            ts_ns: 1_000 + 7 * i,
            bid: 100 + i,
            bid_lots: 3,
            ask: 102 + i,
            ask_lots: 1 + i,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    assert_eq!(data.id().digest, 0x3dc3_a1ba_9ab0_1876_c94e_012d_ffc5_52e6);
}

#[test]
fn test_kernel_one_sided_cancels_and_submits_one_order() {
    let latency = KernelLatency::zero(MS).with_delays(0, 2, 3, 1);
    let cfg = config(latency);
    let rows = (0..12).map(|i| tick(1_000 + 2 * i, 97, 101)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(0, &mut events);
    k.step_with(&mut s, quote(95, 105), &mut events);
    let before = s;
    let out = k.step_with(
        &mut s,
        KernelAction::OneSided {
            side: QuoteSide::Ask,
            price: 104,
            lots: 2,
        },
        &mut events,
    );
    assert_eq!(out.applied, ActionStatus::Applied);
    // One order id, two cancels for the two-sided quote's orders.
    assert_eq!(s.next_order_id(), before.next_order_id() + 1);
    assert!(s.is_open(3) && !s.is_open(4));
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    let three: Vec<_> = reports(&events, true)
        .into_iter()
        .filter(|r| r.order_id == 3)
        .map(|r| (r.kind, r.side, r.price, r.qty))
        .collect();
    assert_eq!(three, vec![(ExecKind::Accepted, Action::Sell, 104, 2)]);
    for id in [1, 2] {
        assert!(
            reports(&events, true)
                .iter()
                .any(|r| r.order_id == id && r.kind == ExecKind::Canceled)
        );
    }
    // Admission applies the same ranges; out of range is rejected.
    let mut t = before;
    let out = k.step(
        &mut t,
        KernelAction::OneSided {
            side: QuoteSide::Bid,
            price: 0,
            lots: 1,
        },
    );
    assert_eq!(
        out.applied,
        ActionStatus::Rejected(RejectReason::PriceRange)
    );
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
}

#[test]
fn test_kernel_quote_error_code_kept_in_outcome_event_and_state() {
    let cfg = config(KernelLatency::zero(MS));
    let data =
        KernelDataset::from_kernel_ticks((0..4).map(|i| tick(1 + i, 98, 102)).collect(), &cfg)
            .unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let base = k.reset(0);
    let mut a = base;
    let mut b = base;
    let mut events = Vec::new();
    let out = k.step_with(&mut a, KernelAction::QuoteError(ErrCode(6)), &mut events);
    assert_eq!(out.applied, ActionStatus::QuoteError(ErrCode(6)));
    assert!(events.contains(&KernelEvent::Decided(ActionStatus::QuoteError(ErrCode(6)))));
    assert_eq!(a.last(), Some(ActionStatus::QuoteError(ErrCode(6))));
    k.step(&mut b, KernelAction::QuoteError(ErrCode(7)));
    assert_ne!(encode(&a), encode(&b), "the code is in the encoding");
    assert_eq!(decode(&encode(&a), &k).unwrap(), a);
    assert_eq!(a.counters().rejected[RejectReason::QuoteError.index()], 1);
    assert_eq!(a.counters().quote_errors, 1);
}

#[test]
fn test_kernel_foreign_state_is_refused_by_every_interpreting_accessor() {
    let cfg = config(KernelLatency::zero(MS));
    let other = KernelConfig::new(dec!(0.5), dec!(0.1), dec!(1000), dec!(10), 100);
    let a = KernelDataset::from_kernel_ticks((0..3).map(|i| tick(1 + i, 98, 102)).collect(), &cfg)
        .unwrap();
    let b =
        KernelDataset::from_kernel_ticks((0..3).map(|i| tick(1 + i, 98, 102)).collect(), &other)
            .unwrap();
    let ka: FinancialKernel<'_> = FinancialKernel::new(&cfg, &a).unwrap();
    let kb: FinancialKernel<'_> = FinancialKernel::new(&other, &b).unwrap();
    let s = ka.reset(0);
    assert!(ka.owns(&s) && !kb.owns(&s));
    let obs = kb.observe(&s);
    assert_eq!(obs.status, Status::Halted(HaltReason::StateMismatch));
    assert_eq!(obs.row, None);
    assert!(kb.venue_summary(&s).is_err());
    let grid = ActionGrid::new([0], [0], [1]).unwrap();
    let mask = kb.mask(&s, &grid);
    assert!(!mask.global_quote_ok && mask.bid == 0 && mask.ask == 0 && mask.size == 0);
    assert_eq!(kb.grid_action(&s, &grid, (0, 0, 0, 0)), None);
    assert!(ka.venue_summary(&s).is_ok());
}

#[test]
fn test_kernel_transaction_undo_restores_the_state_exactly() {
    let (cfg, data) = busy_states();
    let limited = cfg
        .clone()
        .with_limits(RiskLimits::new(dec!(2), Decimal::MAX, dec!(0)).unwrap());
    for c in [cfg, limited] {
        let k: FinancialKernel<'_> = FinancialKernel::new(&c, &data).unwrap();
        let mut changed = 0;
        for s0 in collect_states(&k) {
            let all_legs = [
                step::Legs {
                    legs: [Some((Dir::Buy, 99, 2)), Some((Dir::Sell, 101, 3))],
                    crossed: false,
                    life: None,
                },
                step::Legs {
                    legs: [Some((Dir::Buy, 101, 1)), Some((Dir::Sell, 99, 1))],
                    crossed: true,
                    life: None,
                },
                step::Legs {
                    legs: [Some((Dir::Sell, 102, 4)), None],
                    crossed: false,
                    life: None,
                },
                // An expiring quote's transaction rolls back as exactly.
                step::Legs {
                    legs: [Some((Dir::Buy, 99, 2)), Some((Dir::Sell, 101, 3))],
                    crossed: false,
                    life: core::num::NonZeroU64::new(7),
                },
            ];
            for legs in all_legs {
                let mut s = s0;
                let undo = step::Undo::save(&s);
                let mut sink = Vec::new();
                step::quote_transaction(&k.cfg, &mut s, legs, None, &mut sink).unwrap();
                if s != s0 {
                    changed += 1;
                }
                undo.restore(&mut s);
                assert_eq!(s, s0, "rollback must be exact");
                assert_eq!(encode(&s), encode(&s0));
            }
        }
        assert!(changed > 0);
    }
}

fn adapter_kernel_data() -> (KernelConfig, KernelDataset) {
    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(5000), dec!(5), 100)
        .with_initial_capital(dec!(100000))
        .with_limits(RiskLimits::new(dec!(30), Decimal::MAX, dec!(0)).unwrap());
    let rows = (0..5).map(|i| tick(10 + i, 999, 1_001)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    (cfg, data)
}

fn glft(size: Decimal, lot: Decimal, start: u64) -> GlftQuoter {
    GlftQuoter {
        config: crate::strategy::glft::GLFTConfig::new(
            dec!(0.01),
            dec!(0.3),
            dec!(0.9),
            30,
            lot,
            600_000,
            dec!(0),
        )
        .unwrap(),
        volatility: dec!(1684.7),
        size,
        session_start_ms: start,
        form: crate::backtest::kernel::GlftForm::Exact,
        counts: crate::backtest::kernel::GlftCounts::default(),
    }
}

fn observation_at(ms: u64, position: Decimal) -> Observation {
    Observation {
        row: Some(0),
        ts_ns: ms * MS,
        bid: 999,
        bid_lots: 1,
        ask: 1_001,
        ask_lots: 1,
        position,
        avg_entry: dec!(1000),
        delivered_fills: 0,
        delivered_local_rejects: 0,
        open_orders: 0,
        awaiting_cancel_outcome: 0,
        step: 0,
        status: Status::Running,
        rejected: [0; 8],
    }
}

#[test]
fn test_kernel_adapter_glft_at_inventory_bound_posts_one_side() {
    let (cfg, data) = adapter_kernel_data();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut adapter = QuoteAdapter::new(glft(dec!(1), dec!(1), 0), &k, dec!(1)).unwrap();
    // At +Q only an ask is posted, at -Q only a bid; inside, both.
    let long = adapter.act(&k, &observation_at(10, dec!(30)));
    assert!(
        matches!(
            long,
            KernelAction::OneSided {
                side: QuoteSide::Ask,
                lots: 1,
                ..
            }
        ),
        "{long:?}"
    );
    let short = adapter.act(&k, &observation_at(10, dec!(-30)));
    assert!(
        matches!(
            short,
            KernelAction::OneSided {
                side: QuoteSide::Bid,
                lots: 1,
                ..
            }
        ),
        "{short:?}"
    );
    assert!(matches!(
        adapter.act(&k, &observation_at(10, dec!(0))),
        KernelAction::Quote { .. }
    ));
}

#[test]
fn test_kernel_adapter_session_start_converts_time_and_rejects_earlier() {
    let (cfg, data) = adapter_kernel_data();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    // The model time is the elapsed time since the session start.
    let mut late = QuoteAdapter::new(glft(dec!(1), dec!(1), 0), &k, dec!(1)).unwrap();
    let mut started = QuoteAdapter::new(glft(dec!(1), dec!(1), 590_000), &k, dec!(1)).unwrap();
    let at_start = late.act(&k, &observation_at(0, dec!(5)));
    let same = started.act(&k, &observation_at(590_000, dec!(5)));
    assert_eq!(at_start, same, "same elapsed time, same quote");
    assert_ne!(late.act(&k, &observation_at(590_000, dec!(5))), same);
    // Before the session start: an error, never a silent zero.
    assert_eq!(
        started.act(&k, &observation_at(10, dec!(5))),
        KernelAction::QuoteError(quote_error_code(&MMError::InvalidTimestamp(String::new())))
    );
    let mut avs = QuoteAdapter::new(
        AvellanedaQuoter {
            risk_aversion: dec!(0.1),
            volatility: dec!(0.2),
            order_intensity: dec!(1.5),
            session_start_ms: 100,
            horizon_ms: 1_000,
            size: dec!(1),
        },
        &k,
        dec!(1),
    )
    .unwrap();
    let invalid =
        KernelAction::QuoteError(quote_error_code(&MMError::InvalidTimestamp(String::new())));
    assert_eq!(
        avs.act(&k, &observation_at(50, dec!(0))),
        invalid,
        "before the start"
    );
    assert_eq!(
        avs.act(&k, &observation_at(1_101, dec!(0))),
        invalid,
        "past the horizon"
    );
    assert!(matches!(
        avs.act(&k, &observation_at(600, dec!(0))),
        KernelAction::Quote { .. }
    ));
}

#[test]
fn test_kernel_adapter_rejects_lot_and_size_mismatch_and_foreign_kernel() {
    let (cfg, data) = adapter_kernel_data();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    assert!(
        QuoteAdapter::new(glft(dec!(2), dec!(2), 0), &k, dec!(1)).is_err(),
        "model lot 2, kernel lot 1"
    );
    assert!(
        QuoteAdapter::new(glft(dec!(2), dec!(1), 0), &k, dec!(1)).is_err(),
        "size is two model lots"
    );
    let mut adapter = QuoteAdapter::new(glft(dec!(1), dec!(1), 0), &k, dec!(1)).unwrap();
    // Bound to its kernel: another grid gives the mismatch code.
    let other_cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(5000), dec!(5), 100);
    let other_data = KernelDataset::from_kernel_ticks(
        (0..3).map(|i| tick(10 + i, 1_998, 2_002)).collect(),
        &other_cfg,
    )
    .unwrap();
    let other: FinancialKernel<'_> = FinancialKernel::new(&other_cfg, &other_data).unwrap();
    assert_eq!(
        adapter.act(&other, &observation_at(10, dec!(0))),
        KernelAction::QuoteError(ERR_KERNEL_MISMATCH)
    );
}

#[test]
fn test_kernel_decode_rejects_counters_out_of_range_on_a_reset_state() {
    let cfg = config(KernelLatency::zero(MS));
    let data =
        KernelDataset::from_kernel_ticks((0..4).map(|i| tick(1 + i, 98, 102)).collect(), &cfg)
            .unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let s = k.reset(0);
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
    let fields: [fn(&mut KernelState<16, 64, 128, 64>); 8] = [
        |s| s.stamp_seq = u64::MAX,
        |s| s.next_msg_seq = u64::MAX,
        |s| s.next_order_id = u64::MAX,
        |s| s.next_exec_id = u64::MAX,
        |s| s.counters.local_rejects = u64::MAX,
        |s| s.counters.venue_fills = u64::MAX,
        |s| s.last_delivery_ns = u64::MAX,
        |s| s.last_observed_ns = u64::MAX,
    ];
    for (i, mutate) in fields.iter().enumerate() {
        let mut t = s;
        mutate(&mut t);
        assert!(
            decode(&encode(&t), &k).is_err(),
            "field {i} out of range decoded"
        );
    }
    // Every valid state steps without a halt.
    let mut next = decode(&encode(&s), &k).unwrap();
    k.step(&mut next, quote(97, 103));
    assert!(!matches!(next.status(), Status::Halted(_)));
}

#[test]
fn test_kernel_transaction_halt_reports_rejected_and_rolls_back() {
    let latency = KernelLatency::zero(MS).with_delays(0, 3, 4, 2);
    let cfg = config(latency);
    let rows = (0..8).map(|i| tick(1_000 + i, 97, 101)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> =
        FinancialKernel::new_relaxed(&cfg, &data, true, false, true).unwrap();
    let mut s = k.reset(0);
    k.step(&mut s, quote(95, 105));
    // The next order id cannot advance: the transaction fails after it
    // already scheduled the cancels of the resting orders.
    s.next_order_id = u64::MAX;
    let before = s;
    let mut events = Vec::new();
    let out = k.step_with(&mut s, quote(96, 104), &mut events);
    assert_eq!(out.applied, ActionStatus::Rejected(RejectReason::Terminal));
    assert_eq!(out.status, Status::Halted(HaltReason::CounterOverflow));
    assert_eq!(s.status(), Status::Halted(HaltReason::CounterOverflow));
    assert_eq!(s.phase(), Phase::Terminal);
    assert_eq!(s.counters().halts, 1);
    // Rolled back: no cancel, order, delivery, draw or reservation stays.
    assert_eq!(s.msgs, before.msgs);
    assert_eq!(s.orders, before.orders);
    assert_eq!(s.deliveries, before.deliveries);
    assert_eq!(s.account, before.account);
    assert_eq!(s.gate, before.gate);
    assert_eq!(s.rng_latency, before.rng_latency);
    assert_eq!(
        (s.stamp_seq, s.next_msg_seq, s.next_order_id),
        (before.stamp_seq, before.next_msg_seq, before.next_order_id)
    );
    assert_eq!(
        s.last, before.last,
        "no outcome recorded for the failed action"
    );
    assert_eq!(s.counters().rejected, before.counters().rejected);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, KernelEvent::Report(_) | KernelEvent::Decided(_)))
    );
    assert!(events.contains(&KernelEvent::Halted(HaltReason::CounterOverflow)));
    // The halted state stays halted and unchanged.
    let halted = s;
    assert_eq!(k.step(&mut s, KernelAction::Hold).events, 0);
    assert_eq!(s, halted);
}

/// Room for the partial fill reports of these tests.
type QueueKernel<'d> = FinancialKernel<'d, 16, 64, 256, 64>;
type QueueState = KernelState<16, 64, 256, 64>;

fn lots_tick(ts_ms: u64, bid: u64, bid_lots: u64, ask: u64, ask_lots: u64) -> KernelTick {
    KernelTick {
        ts_ns: ts_ms * MS,
        bid,
        bid_lots,
        ask,
        ask_lots,
    }
}

fn queue_model(rate: Decimal, min_queue_ms: u64) -> KernelFillModel {
    KernelFillModel::QueuePosition { rate, min_queue_ms }
}

fn queue_config(latency: KernelLatency, rate: Decimal, min_queue_ms: u64) -> KernelConfig {
    config(latency).with_fill_model(queue_model(rate, min_queue_ms))
}

fn bid_at(price: u64, lots: u64) -> KernelAction {
    KernelAction::OneSided {
        side: QuoteSide::Bid,
        price,
        lots,
    }
}

fn through_rows(count: u64) -> Vec<KernelTick> {
    (0..count)
        .map(|i| lots_tick(1000 + i, 13, 1, 14, 1))
        .collect()
}

/// Steps through `actions`, then holds, and returns the state and the events.
fn drive(k: &QueueKernel<'_>, actions: &[KernelAction]) -> (QueueState, Vec<KernelEvent>) {
    let mut events = Vec::new();
    let mut s = k.reset_with(3, &mut events);
    let mut decision = 0;
    while s.phase() == Phase::AwaitDecision {
        let action = actions.get(decision).copied().unwrap_or(KernelAction::Hold);
        k.step_with(&mut s, action, &mut events);
        decision += 1;
    }
    (s, events)
}

/// `(kind, last, cum)` of the reports of order 1 after its acceptance, with
/// `cancel_pending`.
fn first_order(events: &[KernelEvent]) -> Vec<(ExecKind, u64, u64, bool)> {
    reports(events, false)
        .iter()
        .filter(|r| r.order_id == 1 && r.kind != ExecKind::Accepted)
        .map(|r| (r.kind, r.last, r.cum, r.cancel_pending))
        .collect()
}

#[test]
fn test_queue_model_fills_the_cleared_volume_beyond_what_is_ahead() {
    let cfg = queue_config(KernelLatency::zero(MS), dec!(1), 0);
    let rows = vec![
        lots_tick(1000, 13, 1, 17, 1),
        // 1 ms: 1 cleared, 1 ahead at the touch.
        lots_tick(1001, 14, 1, 15, 1),
        // 2 ms: 2 cleared, 1 ahead.
        lots_tick(1002, 14, 1, 15, 1),
        // 3 ms: the touch grew, nothing beyond it.
        lots_tick(1003, 14, 1, 15, 5),
        // 4 ms: it shrank, 2 beyond it.
        lots_tick(1004, 14, 1, 15, 2),
        // 5 ms: traded through, nothing ahead.
        lots_tick(1005, 13, 1, 14, 9),
        // 6 ms: not crossed.
        lots_tick(1006, 14, 1, 16, 1),
        // 7 ms: at the touch again.
        lots_tick(1007, 14, 1, 15, 1),
        // 17 ms: more than the order.
        lots_tick(1017, 13, 1, 14, 1),
        lots_tick(1018, 13, 1, 14, 1),
    ];
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let (s, events) = drive(&k, &[bid_at(15, 10)]);
    assert_eq!(
        first_order(&events),
        vec![
            (ExecKind::PartialFill, 1, 1, false),
            (ExecKind::PartialFill, 1, 2, false),
            (ExecKind::PartialFill, 3, 5, false),
            (ExecKind::PartialFill, 1, 6, false),
            (ExecKind::Fill, 4, 10, false),
        ]
    );
    assert_eq!(s.counters().venue_fills, 5);
    assert_eq!(s.queue_lengths().held, 0);
    assert!(
        reports(&events, false)
            .iter()
            .filter(|r| matches!(r.kind, ExecKind::PartialFill | ExecKind::Fill))
            .all(|r| r.price == 15 && r.qty == 10)
    );
}

#[test]
fn test_queue_model_min_queue_time_is_inclusive() {
    let first_fill = |min_queue_ms: u64| {
        let cfg = queue_config(KernelLatency::zero(MS), dec!(1), min_queue_ms);
        let data = KernelDataset::from_kernel_ticks(through_rows(6), &cfg).unwrap();
        let k = QueueKernel::new(&cfg, &data).unwrap();
        let (_, events) = drive(&k, &[bid_at(15, 10)]);
        reports(&events, false)
            .iter()
            .find(|r| r.kind == ExecKind::PartialFill)
            .map(|r| (r.ts_ns / MS - 1000, r.cum))
    };
    // The cumulative quantity includes the time before the minimum.
    assert_eq!(first_fill(0), Some((1, 1)));
    assert_eq!(first_fill(1), Some((1, 1)));
    assert_eq!(first_fill(2), Some((2, 2)));
    assert_eq!(first_fill(3), Some((3, 3)));
}

#[test]
fn test_queue_model_zero_increment_takes_no_stamp_no_id_and_no_draw() {
    // Same seed and actions. On row 2 one dataset crosses with the touch hiding
    // the volume and the other does not cross: neither fills, so the two runs
    // may not differ in anything the engine consumes.
    let latency = KernelLatency::zero(MS)
        .with_delays(1, 1, 1, 1)
        .with_jitter(3);
    let cfg = queue_config(latency, dec!(1), 0);
    let run = |row2: KernelTick| {
        let rows = vec![
            lots_tick(1000, 13, 1, 17, 1),
            lots_tick(1001, 13, 1, 17, 1),
            row2,
            lots_tick(1040, 13, 1, 17, 1),
        ];
        let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
        let k = QueueKernel::new(&cfg, &data).unwrap();
        let (s, events) = drive(&k, &[bid_at(15, 3)]);
        (
            s.stamp_seq(),
            s.next_exec_id,
            s.rng_latency,
            s.counters().venue_fills,
            reports(&events, false).len(),
        )
    };
    let hidden = run(lots_tick(1020, 14, 1, 15, 1000));
    let uncrossed = run(lots_tick(1020, 14, 1, 16, 1000));
    assert_eq!(hidden, uncrossed);
    assert_eq!(hidden.3, 0);
}

#[test]
fn test_queue_model_cancel_after_a_partial_keeps_the_fill_and_releases_the_rest() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 3, 0);
    let cfg = queue_config(latency, dec!(1), 0);
    let data = KernelDataset::from_kernel_ticks(through_rows(8), &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    // Decision 1 sends the cancel, in flight for 3 ms.
    let (s, events) = drive(&k, &[bid_at(15, 5), bid_at(1, 2)]);
    assert_eq!(
        first_order(&events),
        vec![
            (ExecKind::PartialFill, 1, 1, false),
            (ExecKind::PartialFill, 1, 2, true),
            (ExecKind::PartialFill, 1, 3, true),
            (ExecKind::Canceled, 0, 3, false),
        ]
    );
    // Only the replacement is still reserved; the position is what filled.
    assert_eq!(s.gate.live_buy_lots, 2);
    assert_eq!(s.gate.venue_pos_lots, 3);
}

#[test]
fn test_queue_model_last_fill_before_a_cancel_arrives_gets_cancel_rejected() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 3, 0);
    let cfg = queue_config(latency, dec!(1), 0);
    let data = KernelDataset::from_kernel_ticks(through_rows(8), &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let (s, events) = drive(&k, &[bid_at(15, 3), bid_at(1, 2)]);
    assert_eq!(
        first_order(&events),
        vec![
            (ExecKind::PartialFill, 1, 1, false),
            (ExecKind::PartialFill, 1, 2, true),
            (ExecKind::Fill, 1, 3, true),
            (ExecKind::CancelRejected, 0, 3, false),
        ]
    );
    assert_eq!(s.gate.venue_pos_lots, 3);
}

#[test]
fn test_queue_model_validation_refuses_what_is_not_on_the_lot_grid() {
    let refused = |rate: Decimal| {
        let cfg = queue_config(KernelLatency::zero(MS), rate, 0);
        match KernelDataset::from_kernel_ticks(vec![tick(1000, 14, 16)], &cfg) {
            Ok(data) => QueueKernel::new(&cfg, &data).is_err(),
            Err(_) => true,
        }
    };
    assert!(refused(dec!(0)));
    assert!(refused(dec!(-1)));
    assert!(refused(dec!(0.5)));
    assert!(refused(dec!(1.0)));
    assert!(!refused(dec!(1)));
    assert!(!refused(dec!(2)));
}

#[test]
fn test_queue_model_cleared_volume_is_exact_at_every_elapsed_time() {
    // No zero in the product makes it fit: mantissa 7000000001000000007000000001.
    let lot = dec!(1000000000.000000001);
    let rate = dec!(7000000001000000007.000000001);
    let rate_lots: i128 = 7_000_000_001;
    let cfg = KernelConfig::new(dec!(1), lot, dec!(1000), dec!(10000000000.00000001), 100)
        .with_fill_model(queue_model(rate, 0));
    let rows = |last_ms: u64| vec![lots_tick(last_ms, 14, 1, 16, 1)];
    // The mantissa times 11 is below 2^96, times 12 is not.
    let data = KernelDataset::from_kernel_ticks(rows(11), &cfg).unwrap();
    assert!(QueueKernel::new(&cfg, &data).is_ok());
    let data = KernelDataset::from_kernel_ticks(rows(12), &cfg).unwrap();
    let error = QueueKernel::new(&cfg, &data).unwrap_err();
    assert!(error.to_string().contains("not exact"), "{error}");
    // What the engine's model computes is that integer, at every time.
    let unit = grid::Unit::new(lot).unwrap();
    for ms in 0..=11u64 {
        let cleared = rate.checked_mul(Decimal::from(ms)).unwrap();
        assert_eq!(
            unit.count(cleared),
            Some(rate_lots * i128::from(ms)),
            "{ms} ms"
        );
    }
}

#[test]
fn test_queue_model_refuses_a_touch_size_that_is_not_exact_in_a_decimal() {
    // About 1e18 per lot: 2^96 is between 79 and 80 * 10^9 lots.
    let lot = dec!(1000000000.000000001);
    let cfg =
        KernelConfig::new(dec!(1), lot, dec!(1000), lot, 100).with_fill_model(queue_model(lot, 0));
    let new_with = |bid_lots: u64, ask_lots: u64| {
        let rows = vec![lots_tick(1000, 14, bid_lots, 16, ask_lots)];
        let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
        QueueKernel::new(&cfg, &data).map(|_| ())
    };
    let (below, above) = (79_000_000_000, 80_000_000_000);
    assert_eq!(new_with(below, below), Ok(()));
    for (bid_lots, ask_lots) in [(above, 1), (1, above)] {
        let error = new_with(bid_lots, ask_lots).unwrap_err();
        assert!(error.to_string().contains("not exact"), "{error}");
    }
}

#[test]
fn test_queue_model_refuses_an_order_maximum_that_is_not_exact_in_a_decimal() {
    let lot = dec!(1000000000.000000001);
    let new_with = |max_order_size: Decimal| {
        let cfg = KernelConfig::new(dec!(1), lot, dec!(1000), max_order_size, 100)
            .with_fill_model(queue_model(lot, 0));
        let rows = vec![lots_tick(1000, 14, 1, 16, 1)];
        let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
        // Past the ledger bound, which refuses such an order first: this is
        // the model's own check.
        QueueKernel::new_relaxed(&cfg, &data, true, false, false).map(|_| ())
    };
    // 10^10 lots fit below 2^96; 10^11 lots do not.
    let lots = |count: u64| lot.checked_mul(Decimal::from(count)).unwrap();
    assert!(new_with(lots(10_000_000_000)).is_ok());
    let error = new_with(lots(100_000_000_000)).unwrap_err();
    assert!(error.to_string().contains("not exact"), "{error}");
}

#[test]
fn test_queue_model_partial_reports_bound_counts_ties_and_the_inclusive_edge() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 3);
    let queue = queue_config(latency, dec!(1), 0);
    let rows = vec![
        tick(1000, 14, 16),
        tick(1000, 14, 16),
        tick(1001, 14, 16),
        // 1000 + 3 >= 1003: both rows at 1000 are still in the window.
        tick(1003, 14, 16),
        tick(1004, 14, 16),
    ];
    let data = KernelDataset::from_kernel_ticks(rows, &queue).unwrap();
    let reports_of =
        |cfg: &KernelConfig| capacity_need(cfg, &data).unwrap().partial_reports_per_order;
    assert_eq!(reports_of(&queue), 4);
    assert_eq!(reports_of(&config(latency)), 0);
    // The jitter widens the window: 1000 + 4 >= 1004.
    assert_eq!(
        reports_of(&queue_config(latency.with_jitter(1), dec!(1), 0)),
        5
    );
}

#[test]
fn test_queue_model_capacity_is_required_and_exact() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 3);
    let cfg = queue_config(latency, dec!(1), 0);
    let data = KernelDataset::from_kernel_ticks(through_rows(6), &cfg).unwrap();
    assert_eq!(
        capacity_need(&cfg, &data)
            .unwrap()
            .partial_reports_per_order,
        4
    );
    // (3 + 4) * 16.
    assert!(FinancialKernel::<'_, 16, 64, 112, 64>::new(&cfg, &data).is_ok());
    let error = FinancialKernel::<'_, 16, 64, 111, 64>::new(&cfg, &data).unwrap_err();
    assert!(error.to_string().contains("DLV"), "{error}");
}

/// Tracks the largest delivery queue against its open orders.
#[derive(Default)]
struct Peak {
    deliveries: u64,
    held_at_peak: u64,
    per_order: u64,
}

impl EventSink for Peak {
    fn event(&mut self, _: KernelEvent) {}

    fn queues(&mut self, q: QueueLengths) {
        assert!(q.deliveries <= self.per_order * q.held, "{q:?}");
        if q.deliveries > self.deliveries {
            self.deliveries = q.deliveries;
            self.held_at_peak = q.held;
        }
    }
}

#[test]
fn test_queue_model_worst_case_at_the_exact_capacity_never_halts() {
    // Every order rests until its slow cancel arrives, is traded through and
    // reports on every row, with the reports held back by the largest delay.
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 5, 5);
    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(1000), 100)
        .with_initial_capital(dec!(10000000))
        .with_latency(latency)
        .with_fill_model(queue_model(dec!(1), 0));
    let data = KernelDataset::from_kernel_ticks(through_rows(60), &cfg).unwrap();
    let need = capacity_need(&cfg, &data).unwrap();
    assert_eq!(need.ord, 26);
    assert_eq!(need.partial_reports_per_order, 6);
    let k: FinancialKernel<'_, 26, 52, 234, 64> = FinancialKernel::new(&cfg, &data).unwrap();
    assert!(FinancialKernel::<'_, 26, 52, 233, 64>::new(&cfg, &data).is_err());
    let mut peak = Peak {
        per_order: 3 + need.partial_reports_per_order,
        ..Peak::default()
    };
    let mut s = k.reset_with(3, &mut peak);
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, bid_at(15, 1000), &mut peak);
        assert_eq!(decode(&encode(&s), &k).unwrap(), s);
    }
    assert_eq!(s.status(), Status::Terminal(TermReason::EndOfData));
    let rejected = s.counters().rejected;
    assert_eq!(rejected[RejectReason::CapacityDeliveries.index()], 0);
    assert_eq!(rejected[RejectReason::CapacityOrders.index()], 0);
    // The old bound of three per open order would not have held.
    assert!(
        peak.deliveries > 3 * peak.held_at_peak,
        "{} deliveries, {} held",
        peak.deliveries,
        peak.held_at_peak
    );
}

#[test]
fn test_queue_model_is_in_the_config_fingerprint() {
    let base = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100);
    let print = |model| base.clone().with_fill_model(model).fingerprint();
    let immediate = print(KernelFillModel::Immediate);
    assert_eq!(immediate, base.fingerprint());
    let queue = print(queue_model(dec!(1), 0));
    assert_ne!(immediate, queue);
    assert_ne!(queue, print(queue_model(dec!(2), 0)));
    assert_ne!(queue, print(queue_model(dec!(1), 1)));
}

/// A dataset where reports wait 5 ms under the queue model.
fn queued_partials() -> (KernelConfig, KernelDataset) {
    let cfg = queue_config(KernelLatency::zero(MS).with_delays(0, 0, 0, 5), dec!(1), 0);
    let data = KernelDataset::from_kernel_ticks(through_rows(12), &cfg).unwrap();
    (cfg, data)
}

/// A decision with two partial fills of order 1 queued.
fn state_with_two_partials_queued(k: &QueueKernel<'_>) -> QueueState {
    let mut s = k.reset(3);
    k.step(&mut s, bid_at(15, 10));
    while s.phase() == Phase::AwaitDecision {
        let partials = s
            .deliveries
            .as_slice()
            .iter()
            .filter(|d| d.report.kind == ExecKind::PartialFill)
            .count();
        if partials >= 2 {
            return s;
        }
        k.step(&mut s, KernelAction::Hold);
    }
    panic!("no two partial fills in flight");
}

fn refused(k: &QueueKernel<'_>, s: &QueueState, what: &str) {
    let error = decode(&encode(s), k).unwrap_err();
    assert!(error.to_string().contains(what), "{what}: {error}");
}

#[test]
fn test_decode_accepts_partials_and_refuses_inconsistent_fill_reports() {
    let (cfg, data) = queued_partials();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let s = state_with_two_partials_queued(&k);
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
    let partial = |n: usize| {
        s.deliveries
            .position(|d| d.report.kind == ExecKind::PartialFill && d.report.cum == n as u64)
            .unwrap()
    };
    let (first, second) = (partial(1), partial(2));
    let tampered = |change: &dyn Fn(&mut book::Delivery)| {
        let mut t = s;
        change(t.deliveries.get_mut(second).unwrap());
        t
    };
    let open = "kind differs from its open quantity";
    refused(&k, &tampered(&|d| d.report.cum = d.report.qty), open);
    refused(&k, &tampered(&|d| d.report.kind = ExecKind::Fill), open);
    let zero = tampered(&|d| {
        d.report.last = 0;
        if let Some(fill) = &mut d.fill {
            fill.qty = 0;
        }
    });
    refused(&k, &zero, "without a quantity");
    refused(&k, &tampered(&|d| d.report.cum += 1), "do not add up");
    let mut t = s;
    t.deliveries.get_mut(first).unwrap().report.last = 9;
    refused(&k, &t, "beyond the cumulative");
    let mut t = s;
    t.orders.get_mut(0).unwrap().filled += 1;
    t.gate.live_buy_lots -= 1;
    refused(&k, &t, "differs from its reports");
}

#[test]
fn test_decode_refuses_a_cancel_that_differs_from_the_fills() {
    let (cfg, data) = queued_partials();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let s = state_with_two_partials_queued(&k);
    let index = s
        .deliveries
        .position(|d| d.report.kind == ExecKind::PartialFill)
        .unwrap();
    let mut canceled = s.deliveries.get(index).unwrap().report;
    canceled.kind = ExecKind::Canceled;
    canceled.last = 0;
    canceled.cum += 5;
    let mut t = s;
    t.deliveries
        .push(book::Delivery {
            at_ns: s.deliveries.as_slice().last().unwrap().at_ns,
            report: canceled,
            fill: None,
        })
        .unwrap();
    refused(&k, &t, "differs from the fills");
}

#[test]
fn test_decode_under_the_immediate_model_refuses_what_only_partials_produce() {
    let cfg = config(KernelLatency::zero(MS).with_delays(0, 0, 0, 5));
    let data = KernelDataset::from_kernel_ticks(through_rows(12), &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();

    // A resting order, accepted, that the market does not cross.
    let mut resting = k.reset(3);
    k.step(&mut resting, bid_at(10, 3));
    k.step(&mut resting, KernelAction::Hold);
    assert_eq!(decode(&encode(&resting), &k).unwrap(), resting);
    let mut t = resting;
    t.orders.get_mut(0).unwrap().filled = 1;
    t.gate.live_buy_lots -= 1;
    refused(&k, &t, "partially filled under the immediate model");

    // A fill report relabelled as partial.
    let mut filled = k.reset(3);
    k.step(&mut filled, bid_at(15, 3));
    let at = |s: &QueueState| s.deliveries.position(|d| d.report.kind == ExecKind::Fill);
    while at(&filled).is_none() {
        k.step(&mut filled, KernelAction::Hold);
    }
    assert_eq!(decode(&encode(&filled), &k).unwrap(), filled);
    let mut t = filled;
    let delivery = t.deliveries.get_mut(at(&filled).unwrap()).unwrap();
    delivery.report.kind = ExecKind::PartialFill;
    delivery.report.cum -= 1;
    refused(&k, &t, "under the immediate model");
}

#[test]
fn test_decode_refuses_more_fills_than_the_orders_have_lots() {
    let (cfg, data) = queued_partials();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let s = state_with_two_partials_queued(&k);
    let created = s.next_order_id() - 1;
    // Every related counter agrees, so only this bound can refuse it.
    let extra = created * 10 + 1 - s.counters.venue_fills;
    let mut t = s;
    t.counters.venue_fills += extra;
    t.agent.delivered_fills += extra;
    t.next_exec_id += extra;
    t.stamp_seq += extra;
    refused(&k, &t, "more local rejects and fills than orders");
}

#[test]
fn test_decode_stamp_bound_depends_on_the_fill_model() {
    let (cfg, data) = queued_partials();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let s = state_with_two_partials_queued(&k);
    let created = s.next_order_id() - 1;
    // Under the queue model the fills can exceed the orders.
    assert!(s.counters().venue_fills > created);
    assert_eq!(decode(&encode(&s), &k).unwrap(), s);
    // Beyond four per order and one per fill.
    let mut t = s;
    t.stamp_seq = 4 * created + s.counters().venue_fills + 1;
    refused(&k, &t, "stamp_seq");
}

#[test]
fn test_quote_error_name_inverts_quote_error_code_for_every_variant() {
    let errors = [
        MMError::InvalidConfiguration(String::new()),
        MMError::InvalidMarketState(String::new()),
        MMError::NumericalError(String::new()),
        MMError::InvalidPositionUpdate(String::new()),
        MMError::InvalidQuoteGeneration(String::new()),
        MMError::InvalidTimestamp(String::new()),
        MMError::ConnectionError(String::new()),
        MMError::UnsupportedAction {
            action: Action::Buy,
        },
        MMError::LockPoisoned(String::new()),
        MMError::UnsupportedContract {
            reason: String::new(),
        },
        MMError::PriceUnderflow,
        MMError::FillNotBooked(String::new()),
        MMError::FillBookedCheckFailed(String::new()),
        MMError::GreeksMissFill,
        MMError::QuotingHalted(String::new()),
    ];
    let mut codes = Vec::new();
    for error in &errors {
        let code = quote_error_code(error);
        let debug = format!("{error:?}");
        let variant = debug.split(['(', ' ', '{']).next().unwrap();
        assert_eq!(quote_error_name(code), Some(variant), "{debug}");
        codes.push(code.0);
    }
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), errors.len());
    for code in [0, 16, 100, 101, 102, 200] {
        assert_eq!(quote_error_name(ErrCode(code)), None);
    }
}

/// Venue fills of a run on rows every 10 ms whose ask sits at a bid of 14,
/// with a quote on the first row and then on every row (`requote`) or never
/// again, under the given submit and cancel delays in milliseconds.
fn fills_with_latency(submit_ms: u64, cancel_ms: u64, requote: bool) -> u64 {
    let cfg = config(KernelLatency::zero(MS).with_delays(0, submit_ms, cancel_ms, 0));
    let rows = (0..8).map(|i| tick(1000 + 10 * i, 12, 14)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let actions: Vec<KernelAction> = (0..8)
        .map(|i| {
            if i == 0 || requote {
                quote(14, 20)
            } else {
                KernelAction::Hold
            }
        })
        .collect();
    let (s, _) = drive(&k, &actions);
    s.counters().venue_fills
}

#[test]
fn test_kernel_latency_of_exactly_one_row_never_fills_when_every_row_requotes() {
    // Events due at a row's time run before its matching, and an order is
    // matched only on rows after the one it becomes active on, so the cancel
    // of the next quote always lands first.
    assert_eq!(fills_with_latency(10, 10, true), 0);
    // One millisecond less and every order fills on the row it reaches.
    assert_eq!(fills_with_latency(9, 9, true), 7);
    // Without the requote the order rests and fills on the row after.
    assert_eq!(fills_with_latency(10, 10, false), 1);
}

fn rational_config(lots: u64, per_ms: u64, min_queue_ms: u64) -> KernelConfig {
    config(KernelLatency::zero(MS)).with_fill_model(KernelFillModel::QueuePositionRational {
        lots,
        per_ms,
        min_queue_ms,
    })
}

/// `(ms after the first row, kind, last, cum)` of order 1's fills.
fn timed_fills(events: &[KernelEvent]) -> Vec<(u64, ExecKind, u64, u64)> {
    reports(events, false)
        .iter()
        .filter(|r| r.order_id == 1 && matches!(r.kind, ExecKind::PartialFill | ExecKind::Fill))
        .map(|r| (r.ts_ns / MS - 1000, r.kind, r.last, r.cum))
        .collect()
}

#[test]
fn test_queue_model_rational_rate_below_one_lot_per_row_fills_over_several_rows() {
    // 1 lot every 25 ms on rows every 10 ms that trade through the bid: 0.4
    // of a lot per row. Floors per row would never fill; the floor of the
    // total from the queue entry fills at 30, 50 and 80 ms.
    let cfg = rational_config(1, 25, 0);
    let rows = (0..10)
        .map(|i| lots_tick(1000 + 10 * i, 13, 1, 14, 1))
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let (s, events) = drive(&k, &[bid_at(15, 3)]);
    assert_eq!(
        timed_fills(&events),
        vec![
            (30, ExecKind::PartialFill, 1, 1),
            (50, ExecKind::PartialFill, 1, 2),
            (80, ExecKind::Fill, 1, 3),
        ]
    );
    assert_eq!(s.counters().venue_fills, 3);
    // The same order under immediate fill is whole on the first row.
    let immediate = config(KernelLatency::zero(MS));
    let rows = (0..10)
        .map(|i| lots_tick(1000 + 10 * i, 13, 1, 14, 1))
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &immediate).unwrap();
    let k = QueueKernel::new(&immediate, &data).unwrap();
    let (_, events) = drive(&k, &[bid_at(15, 3)]);
    assert_eq!(timed_fills(&events), vec![(10, ExecKind::Fill, 3, 3)]);
}

#[test]
fn test_queue_model_rational_counts_the_touch_ahead_and_the_minimum_time() {
    // 2 lots every 5 ms at the touch, 3 lots ahead, nothing before 10 ms:
    // floor(2t/5) clears 2 at 5 ms (held by the minimum), 4 at 10 ms (1
    // beyond the touch) and 6 at 15 ms (3 beyond).
    let cfg = rational_config(2, 5, 10);
    let rows = (0..4)
        .map(|i| lots_tick(1000 + 5 * i, 14, 1, 15, 3))
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k = QueueKernel::new(&cfg, &data).unwrap();
    let (_, events) = drive(&k, &[bid_at(15, 4)]);
    assert_eq!(
        timed_fills(&events),
        vec![
            (10, ExecKind::PartialFill, 1, 1),
            (15, ExecKind::PartialFill, 2, 3),
        ]
    );
}

#[test]
fn test_queue_model_rational_per_ms_one_is_the_queue_position_model() {
    let rows = || {
        vec![
            lots_tick(1000, 13, 1, 17, 1),
            lots_tick(1001, 14, 1, 15, 1),
            lots_tick(1002, 14, 1, 15, 1),
            lots_tick(1003, 14, 1, 15, 5),
            lots_tick(1005, 13, 1, 14, 9),
            lots_tick(1017, 13, 1, 14, 1),
        ]
    };
    let run = |cfg: KernelConfig| {
        let data = KernelDataset::from_kernel_ticks(rows(), &cfg).unwrap();
        let k = QueueKernel::new(&cfg, &data).unwrap();
        let (s, events) = drive(&k, &[bid_at(15, 10)]);
        (s.counters().venue_fills, timed_fills(&events))
    };
    let whole = queue_config(KernelLatency::zero(MS), dec!(2), 0);
    let rational = rational_config(2, 1, 0);
    let (fills, history) = run(whole.clone());
    assert!(fills > 1, "partial fills to compare");
    assert_eq!(run(rational.clone()), (fills, history));
    // Different models, different configurations, and every field counts.
    assert_ne!(whole.fingerprint(), rational.fingerprint());
    let print = |lots, per_ms, min| rational_config(lots, per_ms, min).fingerprint();
    assert_ne!(print(1, 25, 0), print(1, 50, 0));
    assert_ne!(print(1, 25, 0), print(2, 25, 0));
    assert_ne!(print(1, 25, 0), print(1, 25, 5));
}

#[test]
fn test_queue_model_rational_validation_refuses_zero_lots_or_period() {
    let refused = |lots: u64, per_ms: u64| {
        let cfg = rational_config(lots, per_ms, 0);
        let data = KernelDataset::from_kernel_ticks(vec![tick(1000, 14, 16)], &cfg).unwrap();
        match QueueKernel::new(&cfg, &data) {
            Ok(_) => false,
            Err(MMError::InvalidConfiguration(message)) => {
                message.contains("at least one lot in at least one ms")
            }
            Err(_) => false,
        }
    };
    assert!(refused(0, 1));
    assert!(refused(1, 0));
    assert!(refused(0, 0));
    assert!(!refused(1, 1));
    assert!(!refused(1, 10_000));
}

#[test]
fn test_a_cancel_on_an_empty_book_changes_nothing_but_the_decision() {
    let cfg = config(KernelLatency::zero(MS));
    let rows = (0..4).map(|i| tick(1_000 + i, 14, 16)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let (mut a, mut b) = (k.reset(1), k.reset(1));
    assert_eq!(
        k.step(&mut a, KernelAction::Cancel).applied,
        ActionStatus::Applied
    );
    k.step(&mut b, KernelAction::Hold);
    assert_eq!(encode(&a), encode(&b));
}

#[test]
fn test_a_cancel_is_admitted_at_full_capacity_and_closes_every_open_order() {
    // ORD = 6: after three quotes the account is full and a fourth quote is refused, but a
    // cancel is not: it needs no slot, and every open order is closed by the end.
    let latency = KernelLatency::zero(MS).with_delays(0, 50, 100, 0);
    let mut cfg = config(latency);
    cfg.max_steps = 1_000;
    let rows = (0..400).map(|i| tick(1_000 + i, 14, 16)).collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_, 6, 64, 64, 16> =
        FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(3, &mut events);
    for _ in 0..3 {
        k.step_with(&mut s, quote(10, 20), &mut events);
    }
    assert_eq!(
        k.step(&mut s.clone(), quote(10, 20)).applied,
        ActionStatus::Rejected(RejectReason::CapacityOrders)
    );
    assert_eq!(
        k.step_with(&mut s, KernelAction::Cancel, &mut events)
            .applied,
        ActionStatus::Applied
    );
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    assert_eq!(s.account.len(), 0);
    let cancelled = reports(&events, false)
        .iter()
        .filter(|r| r.kind == ExecKind::Canceled)
        .count();
    assert_eq!(cancelled, 6);
}

#[test]
fn test_a_fill_before_the_cancel_arrives_stands_and_the_late_cancel_is_rejected() {
    // The cancel is sent on row 1 and arrives 2 ms later, on row 3; row 2's ask of 15
    // fills the bid of 15 before it, so the bid's cancel is rejected and the ask's lands.
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 2, 0);
    let cfg = config(latency);
    let rows = vec![
        tick(1_000, 14, 16),
        tick(1_001, 14, 16),
        tick(1_002, 13, 15),
        tick(1_003, 14, 16),
        tick(1_004, 14, 16),
    ];
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut events = Vec::new();
    let mut s = k.reset_with(1, &mut events);
    k.step_with(&mut s, quote(15, 30), &mut events);
    k.step_with(&mut s, KernelAction::Cancel, &mut events);
    while s.phase() == Phase::AwaitDecision {
        k.step_with(&mut s, KernelAction::Hold, &mut events);
    }
    let kinds: Vec<_> = reports(&events, false)
        .iter()
        .filter(|r| r.kind != ExecKind::Accepted)
        .map(|r| (r.side, r.kind))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (Action::Buy, ExecKind::Fill),
            (Action::Buy, ExecKind::CancelRejected),
            (Action::Sell, ExecKind::Canceled),
        ]
    );
}
