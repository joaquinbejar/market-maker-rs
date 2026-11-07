//! Partial fills in the kernel: exact equivalence with the engine's queue
//! position fill model, and the replay, snapshot and batch gate with partial
//! fills in flight and orders partly filled.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

/// Deliveries capacity of the harness kernels: `(3 + R) * ORD` for `R` up to 13.
const HARNESS_DLV: usize = 1024;

#[path = "support/kernel_harness.rs"]
mod harness;

#[path = "support/kernel_gate.rs"]
mod gate;

use gate::*;
use harness::*;
use market_maker_rs::backtest::kernel::{
    ExecKind, KernelAction, KernelEvent, KernelFillModel, KernelLatency, RejectReason, decode,
    encode, fingerprint,
};
use market_maker_rs::dec;
use std::collections::{HashMap, HashSet};

#[test]
fn test_partial_equivalence_seeded_runs_match_engine_exactly() {
    on_large_stack(test_partial_equivalence_seeded_runs_match_engine_exactly_body);
}

fn test_partial_equivalence_seeded_runs_match_engine_exactly_body() {
    let (mut runs, mut skipped) = (0u64, 0u64);
    let (mut partials, mut completed, mut canceled_with_cum) = (0u64, 0u64, 0u64);
    let mut distinct_sizes = HashSet::new();
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let Some(sc) = partial_scenario(seed, index) else {
                skipped += 1;
                continue;
            };
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, false);
            let er = run_engine(&sc);
            assert_equivalent(&k, &sc, &kr, &er, &ctx);
            assert_cancel_pending_agreement(&kr, &er, &ctx);
            assert!(kr.probes > 0, "{ctx}: no queue-bound probe");
            let rejected = kr.last.counters().rejected;
            for reason in [
                RejectReason::CapacityOrders,
                RejectReason::CapacityMsgs,
                RejectReason::CapacityDeliveries,
            ] {
                assert_eq!(rejected[reason.index()], 0, "{ctx}: {reason:?}");
            }
            // Per order, every fill adds its `last` to `cum`.
            let mut cum: HashMap<u64, u64> = HashMap::new();
            for e in &kr.events {
                let KernelEvent::Report(r) = e else { continue };
                match r.kind {
                    ExecKind::PartialFill => {
                        partials += 1;
                        distinct_sizes.insert(r.last);
                        assert!(r.cum < r.qty && r.last > 0, "{ctx}: {r:?}");
                        let before = cum.insert(r.order_id, r.cum).unwrap_or(0);
                        assert_eq!(before + r.last, r.cum, "{ctx}: {r:?}");
                    }
                    ExecKind::Fill => {
                        let before = cum.get(&r.order_id).copied().unwrap_or(0);
                        assert_eq!(before + r.last, r.qty, "{ctx}: {r:?}");
                        assert_eq!(r.cum, r.qty, "{ctx}: {r:?}");
                        completed += u64::from(before > 0);
                    }
                    ExecKind::Canceled => {
                        let before = cum.get(&r.order_id).copied().unwrap_or(0);
                        assert_eq!(before, r.cum, "{ctx}: {r:?}");
                        canceled_with_cum += u64::from(before > 0);
                    }
                    _ => {}
                }
            }
            // `assert_equivalent` already compared every report in order.
            runs += 1;
        }
    }
    assert!(
        runs > skipped,
        "too many scenarios skipped: {runs} ran, {skipped} did not"
    );
    assert!(partials > 0, "partial fills");
    assert!(completed > 0, "orders completed after a partial fill");
    assert!(
        canceled_with_cum > 0,
        "orders canceled after a partial fill"
    );
    assert!(distinct_sizes.len() > 1, "partial fills of different sizes");
}

/// A seeded scenario under the rational rate, with zero latency and its rows
/// moved onto multiples of `per_ms`, so every order enters the queue on one
/// and the cleared volume is always a whole number of lots.
fn rational_scenario(seed: u64, index: u64) -> Option<Scenario> {
    let mut sc = scenario(seed, index);
    let mix = seed.wrapping_mul(31).wrapping_add(index);
    let per_ms = [2u64, 4, 5][usize::try_from(mix % 3).unwrap()];
    let lots = [1u64, 2, 3][usize::try_from((mix / 3) % 3).unwrap()];
    let min_queue_ms = [0u64, 0, per_ms][usize::try_from((mix / 9) % 3).unwrap()];
    sc.cfg.latency = KernelLatency::zero(MS);
    for t in &mut sc.ticks {
        t.timestamp = 1000 + (t.timestamp - 1000) * per_ms;
    }
    sc.cfg.fill_model = KernelFillModel::QueuePositionRational {
        lots,
        per_ms,
        min_queue_ms,
    };
    fits(&sc).then_some(sc)
}

#[test]
fn test_partial_rational_rate_matches_engine_when_cleared_volume_is_whole_lots() {
    on_large_stack(
        test_partial_rational_rate_matches_engine_when_cleared_volume_is_whole_lots_body,
    );
}

fn test_partial_rational_rate_matches_engine_when_cleared_volume_is_whole_lots_body() {
    let (mut runs, mut partials, mut below_one_lot_per_ms) = (0u64, 0u64, 0u64);
    for seed in 0..seeds() {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let Some(sc) = rational_scenario(seed, index) else {
                continue;
            };
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, false);
            let er = run_engine(&sc);
            assert_equivalent(&k, &sc, &kr, &er, &ctx);
            partials += u64::try_from(count_partials(&kr.events)).unwrap();
            if let KernelFillModel::QueuePositionRational { lots, per_ms, .. } = sc.cfg.fill_model {
                below_one_lot_per_ms += u64::from(lots < per_ms);
            }
            runs += 1;
        }
    }
    assert!(runs > 0, "no rational scenario ran");
    assert!(partials > 0, "partial fills under the rational rate");
    assert!(below_one_lot_per_ms > 0, "rates below one lot per ms");
}

/// A golden configuration of the partial gate.
#[derive(Clone, Copy)]
enum Config {
    NoLatency,
    Jitter,
}

const CONFIGS: [Config; 2] = [Config::NoLatency, Config::Jitter];

fn golden_scenario(config: Config, seed: u64) -> Scenario {
    let mut sc = scenario(seed, 7);
    sc.cfg.fee_rate = dec!(0.0001);
    sc.cfg.latency = match config {
        Config::NoLatency => KernelLatency::zero(MS),
        Config::Jitter => KernelLatency::zero(MS)
            .with_delays(1, 2, 2, 1)
            .with_jitter(2),
    };
    sc.cfg.limits = None;
    sc.cfg.fill_model = KernelFillModel::QueuePosition {
        rate: sc.cfg.lot_size,
        min_queue_ms: 0,
    };
    sc
}

/// `(state fingerprint, event stream hash)` per configuration and seed.
const GOLDEN: [[(u128, u128); 3]; 2] = [
    [
        (
            0x43ac689f3117e5e0be5b438eeffe50b6,
            0x381b606d459bbc81472477504abbcca,
        ),
        (
            0xe1f8c119fa2d4ae99c6e2a5b7423b81,
            0xef7fb7c0cfab6355763f298918650830,
        ),
        (
            0xda5324ced899bc45a6a4f2e399b17861,
            0x9cf70fdde2f1e7bd79e2f632955895ff,
        ),
    ],
    [
        (
            0x227f536fd6e1d99b5cf5a617b8442664,
            0x685217a7e573b8af17960f8a9b06ec50,
        ),
        (
            0x2b1db765dbdf60e27e65da77e25e4f02,
            0x1d1646fac7a2effd31e0e835f24eeb76,
        ),
        (
            0x6c094e8bcc2ce6bb09873996730c4cf1,
            0xd0064597ef7210fa6dfd71c476c1dca,
        ),
    ],
];

fn count_partials(events: &[KernelEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, KernelEvent::Report(r) if r.kind == ExecKind::PartialFill))
        .count()
}

#[test]
fn test_partial_golden_trajectories_match_recorded_hashes() {
    on_large_stack(test_partial_golden_trajectories_match_recorded_hashes_body);
}

fn test_partial_golden_trajectories_match_recorded_hashes_body() {
    let print = std::env::var_os("KERNEL_GOLDEN_PRINT").is_some();
    for (c, config) in CONFIGS.iter().enumerate() {
        let mut partials = 0;
        for seed in 0..3u64 {
            let sc = golden_scenario(*config, seed);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            let got = (fingerprint(&t.last), events_hash(&t.events));
            if print {
                println!("config {c} seed {seed}: (0x{:x}, 0x{:x})", got.0, got.1);
                println!(
                    "config {c} seed {seed}: config fingerprint 0x{:x}, dataset digest 0x{:x}",
                    k.config_fingerprint(),
                    data.id().digest
                );
                continue;
            }
            assert_eq!(
                got,
                GOLDEN[c][usize::try_from(seed).unwrap()],
                "config {c} seed {seed}"
            );
            partials += count_partials(&t.events);
        }
        if !print {
            assert!(partials > 0, "config {c}: partial fills");
        }
    }
    assert!(!print, "printed the golden values; nothing was compared");
}

/// An open order with partly filled quantity, read from the reports so far.
fn partly_filled_open(events: &[KernelEvent]) -> bool {
    let mut cum: HashMap<u64, u64> = HashMap::new();
    for e in events {
        let KernelEvent::Report(r) = e else { continue };
        match r.kind {
            ExecKind::PartialFill => {
                cum.insert(r.order_id, r.cum);
            }
            ExecKind::Fill | ExecKind::Canceled => {
                cum.remove(&r.order_id);
            }
            _ => {}
        }
    }
    !cum.is_empty()
}

#[test]
fn test_partial_restore_with_partial_fills_pending_reproduces_the_reference() {
    on_large_stack(test_partial_restore_with_partial_fills_pending_reproduces_the_reference_body);
}

fn test_partial_restore_with_partial_fills_pending_reproduces_the_reference_body() {
    let (mut restored, mut with_partial_cum, mut with_partial_queued, mut with_jitter) =
        (0, 0, 0, 0);
    for seed in 0..seeds().min(40) {
        for index in 0..DATASETS {
            let Some(sc) = partial_scenario(seed, index) else {
                continue;
            };
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            // As a new process would: its own dataset and kernel.
            let fresh_data = sc.data();
            let fresh = kernel(&sc, &fresh_data);
            for (i, snapshot) in t.states.iter().enumerate() {
                let q = snapshot.queue_lengths();
                let before = &t.events[..t.marks[i]];
                let cum = partly_filled_open(before);
                if q.msgs == 0 && q.deliveries == 0 && !cum {
                    continue;
                }
                let ctx = format!("seed {seed} dataset {index} decision {i}");
                let back = decode(&encode(snapshot), &fresh).unwrap();
                assert_eq!(back, *snapshot, "{ctx}: decoded");
                let (last, events, observations) = replay(&fresh, back, &t.actions[i..]);
                assert_eq!(last, t.last, "{ctx}: state");
                assert_eq!(events, t.events[t.marks[i]..], "{ctx}: events");
                assert_eq!(observations, t.observations[i..], "{ctx}: observations");
                restored += 1;
                with_partial_cum += u64::from(cum);
                with_partial_queued += u64::from(
                    before
                        .iter()
                        .rev()
                        .take(usize::try_from(q.deliveries).unwrap().saturating_mul(4))
                        .any(|e| {
                            matches!(e, KernelEvent::Report(r) if r.kind == ExecKind::PartialFill)
                        })
                        && q.deliveries > 0,
                );
                with_jitter += u64::from(sc.cfg.latency.jitter_q > 0 && scheduled(&events));
            }
        }
    }
    assert!(restored > 0, "restores");
    assert!(
        with_partial_cum > 0,
        "restores of a partly filled open order"
    );
    assert!(
        with_partial_queued > 0,
        "restores with a partial fill report queued"
    );
    assert!(
        with_jitter > 0,
        "restores that schedule messages under jitter"
    );
}

#[test]
fn test_partial_forks_with_different_actions_do_not_contaminate() {
    on_large_stack(test_partial_forks_with_different_actions_do_not_contaminate_body);
}

fn test_partial_forks_with_different_actions_do_not_contaminate_body() {
    let (mut forks, mut diverged, mut with_cum) = (0, 0, 0);
    for seed in 0..seeds().min(40) {
        for index in 0..DATASETS {
            let Some(sc) = partial_scenario(seed, index) else {
                continue;
            };
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            let i = t.states.len() / 2;
            let parent = t.states[i];
            let ctx = format!("seed {seed} dataset {index}");

            // A fork with other actions runs first, then the reference's.
            let other = KernelAction::Quote {
                bid: parent_bid(&t.observations[i]),
                ask: t.observations[i].ask + 1,
                bid_lots: 1,
                ask_lots: 1,
            };
            let others = vec![other; t.actions.len() - i];
            let (other_last, other_events, _) = replay(&k, parent, &others);
            let (last, events, _) = replay(&k, parent, &t.actions[i..]);
            assert_eq!(last, t.last, "{ctx}: reference fork");
            assert_eq!(events, t.events[t.marks[i]..], "{ctx}: reference events");

            // The other fork repeats exactly after the reference ran.
            let (again, again_events, _) = replay(&k, parent, &others);
            assert_eq!(again, other_last, "{ctx}: other fork");
            assert_eq!(again_events, other_events, "{ctx}: other events");

            forks += 1;
            diverged += u64::from(fingerprint(&other_last) != fingerprint(&last));
            with_cum += u64::from(partly_filled_open(&t.events[..t.marks[i]]));
        }
    }
    assert!(forks > 0, "forks");
    assert!(diverged > 0, "forks that diverge");
    assert!(with_cum > 0, "forks of a partly filled open order");
}

/// Final fingerprint and event hash per instance.
fn run_instances(indices: &[u64]) -> Vec<(u64, u128, u128)> {
    indices
        .iter()
        .filter_map(|i| {
            let sc = partial_scenario(i / DATASETS, i % DATASETS)?;
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            Some((*i, fingerprint(&t.last), events_hash(&t.events)))
        })
        .collect()
}

#[test]
fn test_partial_batches_do_not_depend_on_worker_assignment() {
    let order: Vec<u64> = (0..120).collect();
    let mut reference = Vec::new();
    on_large_stack(|| reference = run_instances(&order));
    assert!(reference.len() > 60, "instances: {}", reference.len());
    for workers in [1usize, 2, 7] {
        let mut reversed = order.clone();
        reversed.reverse();
        let chunk = reversed.len().div_ceil(workers);
        let mut got: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = reversed
                .chunks(chunk)
                .map(|part| {
                    std::thread::Builder::new()
                        .stack_size(STACK_BYTES)
                        .spawn_scoped(scope, move || run_instances(part))
                        .unwrap()
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        got.sort_unstable();
        assert_eq!(got, reference, "{workers} workers");
    }
}
