//! The replay, snapshot and batch gate of the kernel: golden trajectories,
//! restore with messages in flight, isolated forks, and results independent
//! of how instances are assigned to workers.
//!
//! The causality, gate and queue invariants have their own loops in
//! `kernel_equivalence.rs` and `kernel_bounds.rs`. The kernel fill model
//! fills a resting order completely, so "partial" here means a size reduced
//! by the risk gate; it does not cover `ExecKind::PartialFill`.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

#[path = "support/kernel_harness.rs"]
mod harness;

use harness::*;
use market_maker_rs::backtest::kernel::{
    ExecKind, GateVerdict, KernelAction, KernelEvent, KernelLatency, Observation, Phase, QuoteSide,
    decode, encode, fingerprint, fnv1a128,
};
use market_maker_rs::dec;
use market_maker_rs::risk::RiskLimits;

/// Absolute position, in lots, at which the driver hedges.
const HEDGE_AT: u64 = 2;

/// The script, except that a position of `HEDGE_AT` lots or more is reduced
/// by an order that crosses the touch, one bounded hedge at a time.
fn hedged(sc: &Scenario, decision: u64, obs: &Observation) -> KernelAction {
    let lots = u64::try_from(
        ((obs.position / sc.cfg.lot_size).abs().round())
            .normalize()
            .mantissa(),
    )
    .unwrap();
    if lots < HEDGE_AT {
        return sc.action(decision, obs.bid, obs.ask);
    }
    let (side, price) = if obs.position.is_sign_positive() {
        (QuoteSide::Ask, obs.bid)
    } else {
        (QuoteSide::Bid, obs.ask)
    };
    KernelAction::OneSided {
        side,
        price,
        lots: lots.min(max_lots(sc)),
    }
}

/// One full run of the hedged driver.
struct Trajectory {
    states: Vec<State>,
    observations: Vec<Observation>,
    actions: Vec<KernelAction>,
    /// Events before each decision.
    marks: Vec<usize>,
    events: Vec<KernelEvent>,
    last: State,
}

fn trajectory(k: &Kernel<'_>, sc: &Scenario) -> Trajectory {
    let mut events = Vec::new();
    let mut s = k.reset_with(sc.latency_seed, &mut events);
    let mut t = Trajectory {
        states: Vec::new(),
        observations: Vec::new(),
        actions: Vec::new(),
        marks: Vec::new(),
        events: Vec::new(),
        last: s,
    };
    let mut decision = 0;
    while s.phase() == Phase::AwaitDecision {
        let obs = k.observe(&s);
        let action = hedged(sc, decision, &obs);
        t.states.push(s);
        t.observations.push(obs);
        t.actions.push(action);
        t.marks.push(events.len());
        k.step_with(&mut s, action, &mut events);
        decision += 1;
    }
    t.events = events;
    t.last = s;
    t
}

fn events_hash(events: &[KernelEvent]) -> u128 {
    fnv1a128(format!("{events:?}").as_bytes())
}

/// A golden configuration: scenario base, latency and limits.
#[derive(Clone, Copy)]
enum Config {
    NoLatency,
    Latency,
    Jitter,
    SmallLimits,
}

const CONFIGS: [Config; 4] = [
    Config::NoLatency,
    Config::Latency,
    Config::Jitter,
    Config::SmallLimits,
];

fn golden_scenario(config: Config, seed: u64) -> Scenario {
    let mut sc = scenario(seed, 7);
    sc.cfg.fee_rate = dec!(0.0001);
    sc.cfg.latency = match config {
        Config::NoLatency | Config::SmallLimits => KernelLatency::zero(MS),
        Config::Latency => KernelLatency::zero(MS).with_delays(1, 2, 2, 1),
        Config::Jitter => KernelLatency::zero(MS)
            .with_delays(1, 2, 2, 1)
            .with_jitter(2),
    };
    sc.cfg.limits = match config {
        Config::SmallLimits => Some(
            RiskLimits::new(
                sc.cfg.lot_size * dec!(2),
                market_maker_rs::Decimal::MAX,
                dec!(0),
            )
            .unwrap(),
        ),
        _ => None,
    };
    sc
}

/// `(state fingerprint, event stream hash)` per configuration and seed.
/// A change here is a change in simulated behaviour or in the state or event
/// format, and must be explained in the PR that makes it.
const GOLDEN: [[(u128, u128); 3]; 4] = [
    [
        (
            0x69dcf515a4193b76f0b23a9ebd5a8bee,
            0x1efe26a60070e86485e0a00b57331efe,
        ),
        (
            0xcca6cfe02f86873eac51f088b51bb52,
            0x1ba5840103d98a7f85f142681a05703c,
        ),
        (
            0x986186d96f6637d25c9e517ca7191f33,
            0x8bf13fcfc8a2cdf0dd01960402e8c63,
        ),
    ],
    [
        (
            0x440b0d5a3b9121900b67f8d7c48b1cd9,
            0x860fa19f57b526b2a29872ac602299fe,
        ),
        (
            0x9617231bc5b379ccb0997b74a7c478ce,
            0xdad7c9f96308c003c60bbc69f67032ae,
        ),
        (
            0x9985fc16cc89bd18fea18e3ea2ffb4ea,
            0x430e72b5fdc31eed21ea0fde467326f5,
        ),
    ],
    [
        (
            0xebeaa070f29c677dfeabe465f6106add,
            0x4781e04a9c44b4133311fbbca3963dc1,
        ),
        (
            0x7669a385f15df825b6a5f983fd1b7d31,
            0x4d2aee0da2303c687891caaf032b3100,
        ),
        (
            0x69df26ea940b7224dc46ffa1fb541ea,
            0x170a096b2adf598d8059784814da4a65,
        ),
    ],
    [
        (
            0xfa4ad89fb5fe29ecfa206f02494ea362,
            0x6cb59bc65b0913d8a209a185212ead92,
        ),
        (
            0x83a399d1fcdb50df025958b536806f7b,
            0x96612c23f29e32604bc0b5aebc8f4b90,
        ),
        (
            0x6dc75ccc107ad88f0c83a126d393429d,
            0x9ad07c4c29e7a0c6a6282552839d3dca,
        ),
    ],
];

#[test]
fn test_kernel_golden_trajectories_match_recorded_hashes() {
    let print = std::env::var_os("KERNEL_GOLDEN_PRINT").is_some();
    for (c, config) in CONFIGS.iter().enumerate() {
        let (mut fills, mut cancels, mut reduced, mut in_flight, mut hedged_down) = (0, 0, 0, 0, 0);
        for seed in 0..3u64 {
            let sc = golden_scenario(*config, seed);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            let got = (fingerprint(&t.last), events_hash(&t.events));
            if print {
                println!("config {c} seed {seed}: (0x{:x}, 0x{:x})", got.0, got.1);
                continue;
            }
            assert_eq!(
                got,
                GOLDEN[c][usize::try_from(seed).unwrap()],
                "config {c} seed {seed}"
            );
            for e in &t.events {
                match e {
                    KernelEvent::Fill(_) => fills += 1,
                    KernelEvent::Report(r) if r.kind == ExecKind::Canceled => cancels += 1,
                    KernelEvent::Gate(g) if g.verdict == GateVerdict::Reduced => reduced += 1,
                    _ => {}
                }
            }
            in_flight += t
                .states
                .iter()
                .filter(|s| s.queue_lengths().msgs > 0)
                .count();
            hedged_down += hedges_that_reduced(&k, &t);
        }
        if print {
            continue;
        }
        assert!(fills > 0, "config {c}: fills");
        assert!(cancels > 0, "config {c}: cancels");
        assert!(
            hedged_down > 0,
            "config {c}: hedges that reduced the position"
        );
        if matches!(config, Config::Latency | Config::Jitter) {
            assert!(in_flight > 0, "config {c}: messages in flight");
        }
        if matches!(config, Config::SmallLimits) {
            assert!(reduced > 0, "config {c}: reduced orders");
        }
    }
}

/// Decisions at which the venue position shrank while the latest order
/// submitted was a hedge.
fn hedges_that_reduced(k: &Kernel<'_>, t: &Trajectory) -> u64 {
    let mut count = 0;
    let mut hedge_live = false;
    let abs = |s: &State| k.venue_summary(s).unwrap().position.abs();
    for (i, action) in t.actions.iter().enumerate() {
        match action {
            KernelAction::OneSided { .. } => hedge_live = true,
            KernelAction::Quote { .. } => hedge_live = false,
            KernelAction::Hold | KernelAction::QuoteError(_) => {}
        }
        let next = t.states.get(i + 1).unwrap_or(&t.last);
        if hedge_live && abs(next) < abs(&t.states[i]) {
            count += 1;
        }
    }
    count
}

/// Whether replayed events include a new report, which means venue messages
/// were scheduled and, with jitter, generator draws consumed.
fn scheduled(events: &[KernelEvent]) -> bool {
    events.iter().any(|e| matches!(e, KernelEvent::Report(_)))
}

/// Replays `actions` from `start`, returning the final state, the events and
/// the observation at each decision.
fn replay(
    k: &Kernel<'_>,
    start: State,
    actions: &[KernelAction],
) -> (State, Vec<KernelEvent>, Vec<Observation>) {
    let mut s = start;
    let mut events = Vec::new();
    let mut observations = Vec::new();
    for action in actions {
        observations.push(k.observe(&s));
        k.step_with(&mut s, *action, &mut events);
    }
    (s, events, observations)
}

#[test]
fn test_kernel_restore_with_pending_messages_reproduces_the_reference() {
    let (mut restored, mut with_messages, mut with_deliveries, mut with_jitter) = (0, 0, 0, 0);
    for seed in 0..seeds().min(40) {
        for index in 0..DATASETS {
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            for (i, snapshot) in t.states.iter().enumerate() {
                let q = snapshot.queue_lengths();
                if q.msgs == 0 && q.deliveries == 0 {
                    continue;
                }
                let ctx = format!("seed {seed} dataset {index} decision {i}");
                let back = decode(&encode(snapshot), &k).unwrap();
                let (last, events, observations) = replay(&k, back, &t.actions[i..]);
                assert_eq!(last, t.last, "{ctx}: state");
                assert_eq!(events, t.events[t.marks[i]..], "{ctx}: events");
                assert_eq!(observations, t.observations[i..], "{ctx}: observations");
                restored += 1;
                with_messages += u64::from(q.msgs > 0);
                with_deliveries += u64::from(q.deliveries > 0);
                with_jitter += u64::from(sc.cfg.latency.jitter_q > 0 && scheduled(&events));
            }
        }
    }
    assert!(restored > 0, "restores");
    assert!(with_messages > 0, "restores with venue messages");
    assert!(with_deliveries > 0, "restores with deliveries");
    assert!(
        with_jitter > 0,
        "restores that schedule messages under jitter"
    );
}

#[test]
fn test_kernel_forks_with_different_actions_do_not_contaminate() {
    let (mut forks, mut diverged, mut with_jitter) = (0, 0, 0);
    for seed in 0..seeds().min(40) {
        for index in 0..DATASETS {
            let sc = scenario(seed, index);
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
            // States are values, so this guards the kernel and the sinks against
            // shared mutable state: with jitter the reference only repeats if
            // the other fork's draws left no trace outside its own state.
            with_jitter += u64::from(
                sc.cfg.latency.jitter_q > 0 && scheduled(&other_events) && scheduled(&events),
            );
        }
    }
    assert!(forks > 0, "forks");
    assert!(diverged > 0, "forks that diverge");
    assert!(with_jitter > 0, "forks that schedule messages under jitter");
}

fn parent_bid(obs: &Observation) -> u64 {
    obs.bid.saturating_sub(1).max(1)
}

/// Runs the instances `indices` on one worker.
fn run_instances(indices: &[u64]) -> Vec<(u64, u128, u128)> {
    indices
        .iter()
        .map(|i| {
            let sc = scenario(i / DATASETS, i % DATASETS);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            (*i, fingerprint(&t.last), events_hash(&t.events))
        })
        .collect()
}

/// Runs every instance over `workers` threads, in the given order.
fn run_batched(order: &[u64], workers: usize) -> Vec<(u64, u128, u128)> {
    let chunk = order.len().div_ceil(workers);
    let mut out: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = order
            .chunks(chunk)
            .map(|part| scope.spawn(move || run_instances(part)))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    out.sort_unstable();
    out
}

#[test]
fn test_kernel_batches_do_not_depend_on_worker_assignment() {
    let order: Vec<u64> = (0..200).collect();
    let reference = run_instances(&order);
    let distinct: std::collections::HashSet<u128> = reference.iter().map(|r| r.1).collect();
    assert!(
        distinct.len() > 100,
        "instances must differ: {}",
        distinct.len()
    );
    let mut reversed = order.clone();
    reversed.reverse();
    let mut shuffled = order.clone();
    let mut rng = SplitMix64(84);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, usize::try_from(rng.below(i as u64 + 1)).unwrap());
    }
    for workers in [1, 2, 3, 8] {
        assert_eq!(
            run_batched(&order, workers),
            reference,
            "{workers} in order"
        );
        assert_eq!(
            run_batched(&reversed, workers),
            reference,
            "{workers} reversed"
        );
        assert_eq!(
            run_batched(&shuffled, workers),
            reference,
            "{workers} shuffled"
        );
    }
}

#[test]
#[ignore = "timing; run in release with --ignored --nocapture"]
fn test_kernel_batch_throughput_per_instance_and_batched() {
    let order: Vec<u64> = (0..2_000).collect();
    let start = std::time::Instant::now();
    let reference = run_instances(&order);
    let sequential = start.elapsed();
    for workers in [2, 4, 8] {
        let start = std::time::Instant::now();
        let batched = run_batched(&order, workers);
        println!("{workers} workers: {:?}", start.elapsed());
        assert_eq!(batched, reference);
    }
    println!("sequential: {sequential:?} for {} instances", order.len());
}
