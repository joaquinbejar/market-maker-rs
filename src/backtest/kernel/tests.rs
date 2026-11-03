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
        ("next_msg_seq beyond two per order", |s, _| {
            s.next_msg_seq = 2 * s.next_order_id + 1;
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
                book::MsgKind::Activate(o) => {
                    o.qty = 0;
                    true
                }
                book::MsgKind::Cancel { .. } => false,
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
                },
                step::Legs {
                    legs: [Some((Dir::Buy, 101, 1)), Some((Dir::Sell, 99, 1))],
                    crossed: true,
                },
                step::Legs {
                    legs: [Some((Dir::Sell, 102, 4)), None],
                    crossed: false,
                },
            ];
            for legs in all_legs {
                let mut s = s0;
                let undo = step::Undo::save(&s);
                let mut sink = Vec::new();
                step::quote_transaction(&k.cfg, &mut s, legs, &mut sink).unwrap();
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
