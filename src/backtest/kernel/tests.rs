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
        .filter(|m| matches!(m.kind, book::MsgKind::Activate(o) if o.cancel_inflight))
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
