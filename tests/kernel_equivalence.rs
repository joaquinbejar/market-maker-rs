//! Exact equivalence of the financial kernel with the backtest engine, and
//! the seeded property loops: at least 500 seeds × 20 datasets each;
//! `KERNEL_SEEDS` raises the seed count.
//!
//! Equivalence is defined only for datasets with non-decreasing timestamps
//! (ties allowed); the generator produces only such datasets.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

/// Deliveries capacity of the harness kernels.
const HARNESS_DLV: usize = 192;

#[path = "support/kernel_harness.rs"]
mod harness;

use harness::*;
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionStatus, GateVerdict, KernelAction, KernelEvent, Phase, RejectReason, Status,
    decode, encode,
};

/// Bytes at the end of a state encoding: last outcome (3), actions (8),
/// rejected (64), six counters (48).
const TAIL: usize = 123;

#[test]
fn test_kernel_equivalence_seeded_runs_match_engine_exactly() {
    let mut cover = [0u64; 8];
    let mut paths = Coverage::default();
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, false);
            let er = run_engine(&sc);
            assert_equivalent(&k, &sc, &kr, &er, &ctx);
            let (flagged, dropped) = assert_cancel_pending_agreement(&kr, &er, &ctx);
            harness::cover(&sc, &kr, &mut paths);
            assert!(kr.probes > 0, "{ctx}: no queue-bound probe");
            // With the capacities new() requires, no capacity reject.
            let rejected = kr.last.counters().rejected;
            for reason in [
                RejectReason::CapacityOrders,
                RejectReason::CapacityMsgs,
                RejectReason::CapacityDeliveries,
            ] {
                assert_eq!(rejected[reason.index()], 0, "{ctx}: {reason:?}");
            }
            cover[0] += flagged;
            cover[1] += dropped;
            cover[2] += kr.last.counters().venue_fills;
            cover[3] += kr.last.counters().local_rejects;
            for e in &kr.events {
                if let KernelEvent::Gate(g) = e {
                    match g.verdict {
                        GateVerdict::Reduced => cover[4] += 1,
                        GateVerdict::ReducedOffLot => cover[5] += 1,
                        GateVerdict::Rejected => cover[6] += 1,
                        GateVerdict::Accepted => {}
                    }
                }
            }
            cover[7] += kr.last.counters().eof_dropped_msgs;
            // One execution id per fill, never replayed.
            let ids: Vec<u64> = kr
                .events
                .iter()
                .filter_map(|e| match e {
                    KernelEvent::Fill(f) => Some(f.exec_id),
                    _ => None,
                })
                .collect();
            assert!(
                ids.iter().copied().eq(1..=ids.len() as u64),
                "{ctx}: execution ids {ids:?}"
            );
        }
    }
    // Every path the comparison is meant to cover was exercised.
    for (name, count) in [
        "cancel pending",
        "pending at eof",
        "fills",
        "local rejects",
        "reduced",
        "reduced off lot",
        "gate rejects",
        "eof drops",
    ]
    .iter()
    .zip(cover)
    {
        assert!(count > 0, "no {name} in the seeded runs");
    }
    // Odd mids and fs = 0 runs compare equity and drawdown in half money
    // units; the race and jitter paths are exercised against the engine.
    let p = paths;
    for (name, count) in [
        ("odd bid + ask rows", p.odd_mid2_rows),
        ("fee scale 0 runs with fills", p.fs0_runs_with_fills),
        ("equal-time cancel beating a fill", p.equal_ts_cancel_wins),
        ("cancel before activation", p.cancel_before_activation),
        ("cancel_pending with jitter", p.cancel_pending_with_jitter),
    ] {
        assert!(count > 0, "no {name} in the seeded runs: {p:?}");
    }
}

#[test]
fn test_kernel_properties_replay_decode_and_reject_equals_hold() {
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            let mut rng = SplitMix64(seed ^ (index << 32) ^ 0xABCD);
            // decode(encode(s)) == s at every decision and at the end, and
            // the queue bounds hold.
            for s in kr.states.iter().chain(std::iter::once(&kr.last)) {
                let decoded = decode(&encode(s), &k).unwrap();
                assert_eq!(&decoded, s, "{ctx}: decode round trip");
                // A valid decoded state never halts on its next step.
                for action in [
                    KernelAction::Hold,
                    KernelAction::Quote {
                        bid: 1,
                        ask: 2,
                        bid_lots: 1,
                        ask_lots: 1,
                    },
                ] {
                    let mut next = decoded;
                    k.step(&mut next, action);
                    assert!(
                        !matches!(next.status(), Status::Halted(_)),
                        "{ctx}: halt after decode"
                    );
                }
                let q = s.queue_lengths();
                assert!(q.orders + q.awaiting <= q.held, "{ctx}: slots {q:?}");
                assert!(q.msgs <= 2 * q.held, "{ctx}: messages {q:?}");
                assert!(q.deliveries <= 3 * q.held, "{ctx}: deliveries {q:?}");
            }
            if kr.states.is_empty() {
                continue;
            }
            let j = usize::try_from(rng.below(kr.states.len() as u64)).unwrap();
            // Cloned-state replay reproduces the rest of the run.
            let mut s = kr.states[j];
            let mut events = Vec::new();
            for action in &kr.actions[j..] {
                k.step_with(&mut s, *action, &mut events);
            }
            assert_eq!(s, kr.last, "{ctx}: replay final state");
            assert_eq!(
                events,
                kr.events[kr.marks[j]..].to_vec(),
                "{ctx}: replay events"
            );
            // Repeated terminal steps change nothing.
            let mut done = kr.last;
            for _ in 0..3 {
                let out = k.step(&mut done, KernelAction::Hold);
                assert_eq!(out.applied, ActionStatus::Rejected(RejectReason::Terminal));
                assert_eq!(out.events, 0);
            }
            assert_eq!(done, kr.last, "{ctx}: terminal step idempotent");
            // A rejected action is a hold.
            let mut a = kr.states[j];
            let mut b = kr.states[j];
            let (mut ea, mut eb) = (Vec::new(), Vec::new());
            let bad = KernelAction::Quote {
                bid: 1,
                ask: u64::MAX,
                bid_lots: 1,
                ask_lots: 1,
            };
            assert_eq!(
                k.step_with(&mut a, bad, &mut ea).applied,
                ActionStatus::Rejected(RejectReason::PriceRange)
            );
            k.step_with(&mut b, KernelAction::Hold, &mut eb);
            // The streams differ only in the step's own outcome record.
            let venue = |events: &[KernelEvent]| -> Vec<KernelEvent> {
                events
                    .iter()
                    .filter(|e| !matches!(e, KernelEvent::Decided(_)))
                    .copied()
                    .collect()
            };
            assert_eq!(venue(&ea), venue(&eb), "{ctx}: rejected events");
            let (xa, xb) = (encode(&a), encode(&b));
            assert_eq!(xa.len(), xb.len());
            let n = xa.len();
            assert_eq!(xa[..n - TAIL], xb[..n - TAIL], "{ctx}: rejected state");
            assert_eq!(
                xa[n - TAIL + 3..n - TAIL + 11],
                xb[n - TAIL + 3..n - TAIL + 11],
                "{ctx}: actions"
            );
            assert_eq!(xa[n - 48..], xb[n - 48..], "{ctx}: other counters");
        }
    }
}

#[test]
fn test_kernel_properties_mask_equals_admission() {
    let mut probes = [0u64; 2];
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            let lots = max_lots(&sc);
            let grid =
                ActionGrid::new([0, 1, 3, 9_000], [0, 2, 5, 12_000], [1, lots, lots + 1]).unwrap();
            let mut rng = SplitMix64(seed.wrapping_mul(31) ^ index);
            for s in &kr.states {
                let mask = k.mask(s, &grid);
                assert_eq!(mask.to_bytes(), k.mask(&s.clone(), &grid).to_bytes());
                for _ in 0..3 {
                    let idx = (
                        usize::try_from(rng.below(4)).unwrap(),
                        usize::try_from(rng.below(4)).unwrap(),
                        usize::try_from(rng.below(3)).unwrap(),
                        usize::try_from(rng.below(3)).unwrap(),
                    );
                    let allowed = mask.allows(idx.0, idx.1, idx.2, idx.3);
                    match k.grid_action(s, &grid, idx) {
                        None => assert!(!allowed, "{ctx}: unrepresentable action allowed"),
                        Some(action) => {
                            let mut c = *s;
                            let applied = k.step(&mut c, action).applied == ActionStatus::Applied;
                            assert_eq!(applied, allowed, "{ctx}: mask {mask:?} action {action:?}");
                            probes[usize::from(applied)] += 1;
                        }
                    }
                }
                let mut c = *s;
                assert_eq!(
                    k.step(&mut c, KernelAction::Hold).applied,
                    ActionStatus::Applied
                );
            }
        }
    }
    assert!(probes[0] > 0 && probes[1] > 0, "mask probes {probes:?}");
}

/// Look-ahead check at decision `j`: future rows, not yet observed rows and
/// matched but undelivered rows are replaced, separately and together. With
/// the same enriched delivered history the observation and the mask are
/// bit-identical; for future rows (and undelivered rows without jitter) the
/// history must be the same.
fn check_lookahead(
    sc: &Scenario,
    k: &Kernel<'_>,
    kr: &KernelRun,
    j: usize,
    rng: &mut SplitMix64,
    compared: &mut [u64; 4],
    ctx: &str,
) {
    use market_maker_rs::backtest::MarketTick;
    let t = decision_times(&kr.events)[j];
    let obs_row = usize::try_from(kr.observations[j].row.unwrap()).unwrap();
    let lat = sc.cfg.latency;
    let obs_max = (lat.observation_q + lat.jitter_q) * lat.quantum_ns;
    let ack_min = lat.ack_q * lat.quantum_ns;
    let ts = |tick: &MarketTick| tick.timestamp * MS;
    let future = |_: usize, x: &MarketTick| ts(x) > t;
    let unobserved = |i: usize, x: &MarketTick| i > obs_row && ts(x) <= t && ts(x) + obs_max > t;
    let undelivered = |i: usize, x: &MarketTick| i > obs_row && ts(x) <= t && ts(x) + ack_min > t;
    let all = |i: usize, x: &MarketTick| future(i, x) || unobserved(i, x) || undelivered(i, x);
    let base = &kr.states[j];
    let base_view = (k.observe(base).to_bytes(), k.mask(base, &grid()).to_bytes());
    let (_, base_hist, base_out) = run_prefix(k, sc.latency_seed, &kr.actions, j);
    let cases: [(&str, &RowPick<'_>, bool); 4] = [
        ("future rows", &future, true),
        ("unobserved rows", &unobserved, false),
        ("undelivered rows", &undelivered, lat.jitter_q == 0),
        ("all together", &all, false),
    ];
    for (c, (name, pick, strict)) in cases.iter().enumerate() {
        let (psc, changed) = perturb(sc, rng, pick);
        if changed == 0 {
            continue;
        }
        let pdata = psc.data();
        let pk = kernel(&psc, &pdata);
        let (ps, hist, out) = run_prefix(&pk, sc.latency_seed, &kr.actions, j);
        let same_history = ps.phase() == Phase::AwaitDecision
            && enriched(&hist) == enriched(&base_hist)
            && out == base_out;
        if *strict {
            assert!(same_history, "{ctx}: {name}: delivered history changed");
        }
        if same_history {
            let view = (pk.observe(&ps).to_bytes(), pk.mask(&ps, &grid()).to_bytes());
            assert_eq!(
                view, base_view,
                "{ctx}: {name}: view changed at decision {j}"
            );
            compared[c] += 1;
        }
    }
}

#[test]
fn test_kernel_properties_lookahead_perturbations_leave_view_unchanged() {
    let mut compared = [0u64; 4];
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            if kr.states.is_empty() {
                continue;
            }
            let mut rng = SplitMix64(seed.wrapping_mul(977) ^ index ^ 0x5EED);
            let j = usize::try_from(rng.below(kr.states.len() as u64)).unwrap();
            check_lookahead(
                &sc,
                &k,
                &kr,
                j,
                &mut rng,
                &mut compared,
                &format!("seed {seed} dataset {index}"),
            );
        }
    }
    for (c, count) in compared.iter().enumerate() {
        assert!(*count > 0, "perturbation {c} never compared");
    }
}

#[test]
fn test_kernel_properties_lookahead_at_every_decision() {
    // Every decision of a seed subset (one twentieth of the seeds).
    let mut compared = [0u64; 4];
    for seed in 0..seeds().div_ceil(20) {
        for index in 0..DATASETS {
            let sc = scenario(seed + 10_000, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            let mut rng = SplitMix64(seed ^ index.rotate_left(17));
            for j in 0..kr.states.len() {
                check_lookahead(
                    &sc,
                    &k,
                    &kr,
                    j,
                    &mut rng,
                    &mut compared,
                    &format!("seed {seed} dataset {index} decision {j}"),
                );
            }
        }
    }
    assert!(compared.iter().all(|c| *c > 0), "{compared:?}");
}

/// Selects the rows a perturbation replaces.
type RowPick<'a> = dyn Fn(usize, &market_maker_rs::backtest::MarketTick) -> bool + 'a;

fn grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 4], [0, 2, 3], [1, 2]).unwrap()
}
