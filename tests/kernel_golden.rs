//! The replay, snapshot and batch gate of the kernel: golden trajectories,
//! restore with messages in flight, isolated forks, and results independent
//! of how instances are assigned to workers.
//!
//! The causality, gate and queue invariants have their own loops in
//! `kernel_equivalence.rs` and `kernel_bounds.rs`. The immediate fill model
//! fills a resting order completely, so "partial" here means a size reduced
//! by the risk gate; `kernel_partial.rs` covers `ExecKind::PartialFill`.

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

#[path = "support/kernel_gate.rs"]
mod gate;

use arbitrage_rs::hedge::{HedgeDecision, NoHedgeReason};
use gate::*;
use harness::*;
use market_maker_rs::backtest::kernel::{
    ActionStatus, ExecKind, GateVerdict, KernelAction, KernelEvent, KernelLatency, Observation,
    QuoteSide, decode, encode, fingerprint,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;

/// The kernel's first order id.
const FIRST_ORDER_ID: u64 = 1;

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
/// A change here is a change in simulated behaviour, in the state encoding,
/// or in the `Debug` output of the events, which `events_hash` hashes; a
/// renamed event field changes it with no change in behaviour. Explain it in
/// the PR that makes it.
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
            hedged_down += hedges_that_reduced(&t);
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
    assert!(!print, "printed the golden values; nothing was compared");
}

/// Fills of hedge orders that shrank the venue position. Order ids follow
/// the kernel: sequential from `FIRST_ORDER_ID`, one per submitted leg, none
/// for a rejected action.
fn hedges_that_reduced(t: &Trajectory) -> u64 {
    let mut next_id = FIRST_ORDER_ID;
    let mut hedge_ids = std::collections::HashSet::new();
    for (action, outcome) in t.actions.iter().zip(&t.outcomes) {
        if matches!(outcome, ActionStatus::Rejected(_)) {
            continue;
        }
        match action {
            KernelAction::OneSided { .. } => {
                hedge_ids.insert(next_id);
                next_id += 1;
            }
            KernelAction::Quote { .. } | KernelAction::ExpiringQuote { .. } => next_id += 2,
            KernelAction::Hold | KernelAction::QuoteError(_) | KernelAction::Cancel => {}
            other => panic!("an action the golden runs do not take: {other:?}"),
        }
    }
    assert_eq!(next_id, t.last.next_order_id(), "order id derivation");
    let mut position = 0i128;
    let mut count = 0;
    for e in &t.events {
        if let KernelEvent::Fill(f) = e {
            let before = position;
            let qty = i128::from(f.qty);
            position += if f.side == Action::Buy { qty } else { -qty };
            if hedge_ids.contains(&f.order_id) && position.abs() < before.abs() {
                count += 1;
            }
        }
    }
    count
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
            // As a new process would: its own dataset and kernel.
            let fresh_data = sc.data();
            let fresh = kernel(&sc, &fresh_data);
            for (i, snapshot) in t.states.iter().enumerate() {
                let q = snapshot.queue_lengths();
                if q.msgs == 0 && q.deliveries == 0 {
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

/// Final fingerprint and event hash per latency seed, on one shared kernel.
fn run_seeds(k: &Kernel<'_>, sc: &Scenario, seeds: &[u64]) -> Vec<(u64, u128, u128)> {
    seeds
        .iter()
        .map(|seed| {
            let mut one = sc.clone();
            one.latency_seed = *seed;
            let t = trajectory(k, &one);
            (*seed, fingerprint(&t.last), events_hash(&t.events))
        })
        .collect()
}

#[test]
fn test_kernel_shared_across_threads_matches_sequential_runs() {
    let seeds: Vec<u64> = (0..40).collect();
    let mut with_jitter = 0;
    for index in 0..DATASETS {
        let sc = scenario(3, index);
        let data = sc.data();
        let k = kernel(&sc, &data);
        let reference = run_seeds(&k, &sc, &seeds);
        let mut shared: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .chunks(7)
                .map(|part| {
                    let (k, sc) = (&k, &sc);
                    scope.spawn(move || run_seeds(k, sc, part))
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        shared.sort_unstable();
        assert_eq!(shared, reference, "dataset {index}");
        if sc.cfg.latency.jitter_q > 0 {
            let distinct: std::collections::HashSet<u128> = reference.iter().map(|r| r.1).collect();
            with_jitter += u64::from(distinct.len() > 1);
        }
    }
    assert!(with_jitter > 0, "latency seeds that change a jittered run");
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

/// A market where one lot clears at the touch, and an observation of it.
fn hedge_fixture(max_order_lots: u64, actions: Vec<KernelAction>) -> Scenario {
    let cfg = market_maker_rs::backtest::kernel::KernelConfig::new(
        dec!(1),
        dec!(1),
        dec!(1000),
        market_maker_rs::Decimal::from(max_order_lots),
        100,
    )
    .with_initial_capital(dec!(10000));
    let ticks = (0..8u64)
        .map(|i| {
            market_maker_rs::backtest::MarketTick::new(
                1000 + i,
                dec!(13),
                dec!(5),
                dec!(14),
                dec!(5),
            )
        })
        .collect();
    Scenario::fixed(cfg, ticks, actions)
}

#[test]
fn test_kernel_hedges_are_what_arbitrage_rs_plans() {
    let sc = hedge_fixture(10, Vec::new());
    let data = sc.data();
    let k = kernel(&sc, &data);
    let obs = k.observe(&k.reset(1));
    let at = |lots: market_maker_rs::Decimal| Observation {
        position: lots,
        ..obs
    };

    // Inside the band, edges included: the script decides, not the hedge.
    for position in [dec!(0), dec!(1), dec!(-1)] {
        assert!(matches!(
            band_plan(&sc, &at(position), None),
            HedgeDecision::NoHedge {
                reason: NoHedgeReason::InsideBand,
                ..
            }
        ));
        assert_eq!(
            hedged(&sc, 0, &at(position)),
            sc.action(0, obs.bid, obs.ask)
        );
    }
    // From two lots: back to zero, selling at the bid or buying at the ask.
    assert_eq!(
        hedged(&sc, 0, &at(dec!(2))),
        KernelAction::OneSided {
            side: QuoteSide::Ask,
            price: 13,
            lots: 2,
        }
    );
    assert_eq!(
        hedged(&sc, 0, &at(dec!(-3))),
        KernelAction::OneSided {
            side: QuoteSide::Bid,
            price: 14,
            lots: 3,
        }
    );

    // The largest order bounds it even with more quoted, and a touch size bounds
    // it when one is given.
    let small = hedge_fixture(2, Vec::new());
    let plan = band_plan(&small, &at(dec!(5)), Some(5));
    let plan = plan.plan().unwrap();
    assert_eq!(plan.leg.quantity.value(), dec!(2));
    assert_eq!(plan.expected_exposure_after, dec!(3));
    let capped = band_plan(&sc, &at(dec!(5)), Some(1));
    let capped = capped.plan().unwrap();
    assert_eq!(capped.leg.quantity.value(), dec!(1));
    assert_eq!(capped.expected_exposure_after, dec!(4));
}

/// The driver the gate had before arbitrage-rs, for the comparisons below.
fn hand_written_hedge(sc: &Scenario, decision: u64, obs: &Observation) -> KernelAction {
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

#[test]
fn test_kernel_hedges_equal_the_hand_written_driver() {
    // Every position from -12 to 12 lots, including beyond the largest order.
    let sc = hedge_fixture(10, Vec::new());
    let data = sc.data();
    let k = kernel(&sc, &data);
    let obs = k.observe(&k.reset(1));
    for lots in -12..=12i64 {
        let at = Observation {
            position: market_maker_rs::Decimal::from(lots),
            ..obs
        };
        assert_eq!(
            hedged(&sc, 0, &at),
            hand_written_hedge(&sc, 0, &at),
            "{lots} lots"
        );
    }
    // Every decision of the seeded trajectories, on their grids and latencies.
    let (mut decisions, mut hedges, mut longs, mut shorts) = (0u64, 0u64, 0u64, 0u64);
    for seed in 0..seeds().min(40) {
        for index in 0..DATASETS {
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            for (i, obs) in t.observations.iter().enumerate() {
                let decision = u64::try_from(i).unwrap();
                assert_eq!(
                    t.actions[i],
                    hand_written_hedge(&sc, decision, obs),
                    "seed {seed} dataset {index} decision {i}"
                );
                decisions += 1;
                if matches!(t.actions[i], KernelAction::OneSided { .. }) {
                    hedges += 1;
                    longs += u64::from(obs.position.is_sign_positive());
                    shorts += u64::from(obs.position.is_sign_negative());
                }
            }
        }
    }
    println!("decisions {decisions}, hedges {hedges}, long {longs}, short {shorts}");
    assert!(longs > 0 && shorts > 0, "hedges at both signs");
}

#[test]
fn test_kernel_hedge_plan_before_any_row_has_no_quote() {
    let sc = hedge_fixture(10, Vec::new());
    let data = sc.data();
    let k = kernel(&sc, &data);
    let obs = k.observe(&k.reset(1));
    let unquoted = |position| Observation {
        position,
        row: None,
        ts_ns: 0,
        bid: 0,
        ask: 0,
        bid_lots: 0,
        ask_lots: 0,
        ..obs
    };
    assert!(matches!(
        band_plan(&sc, &unquoted(dec!(0)), None),
        HedgeDecision::NoHedge {
            reason: NoHedgeReason::InsideBand,
            ..
        }
    ));
    assert!(matches!(
        band_plan(&sc, &unquoted(dec!(2)), None),
        HedgeDecision::NoHedge {
            reason: NoHedgeReason::NoLiquidity { side: Action::Sell },
            ..
        }
    ));
}

/// An observation of the hedge fixture at `position`.
fn observed_at(sc: &Scenario, position: market_maker_rs::Decimal) -> Observation {
    let data = sc.data();
    let k = kernel(sc, &data);
    Observation {
        position,
        ..k.observe(&k.reset(1))
    }
}

#[test]
fn test_kernel_hedge_rounds_down_to_whole_lots() {
    let sc = hedge_fixture(10, Vec::new());
    let decision = band_plan(&sc, &observed_at(&sc, dec!(2.5)), None);
    assert_eq!(decision.plan().unwrap().leg.quantity.value(), dec!(2));
}

#[test]
#[should_panic(expected = "not a multiple")]
fn test_kernel_hedge_refuses_a_position_off_the_lot_grid() {
    let sc = hedge_fixture(10, Vec::new());
    let _ = hedged(&sc, 0, &observed_at(&sc, dec!(1.5)));
}

#[test]
#[should_panic(expected = "no hedge for an observed position")]
fn test_kernel_hedge_refuses_a_position_it_cannot_hedge() {
    let sc = hedge_fixture(10, Vec::new());
    let obs = Observation {
        bid: 0,
        ask: 0,
        ..observed_at(&sc, dec!(2))
    };
    let _ = hedged(&sc, 0, &obs);
}

#[test]
fn test_kernel_hedge_ends_where_the_plan_expects_once_its_fill_is_delivered() {
    let buy = KernelAction::OneSided {
        side: QuoteSide::Bid,
        price: 15,
        lots: 3,
    };
    let mut sc = hedge_fixture(10, vec![buy]);
    sc.cfg.latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 2);
    let data = sc.data();
    let k = kernel(&sc, &data);
    let mut s = k.reset(1);
    k.step(&mut s, buy);
    while k.observe(&s).position != dec!(3) {
        k.step(&mut s, KernelAction::Hold);
    }
    let obs = k.observe(&s);
    let decision = band_plan(&sc, &obs, None);
    let plan = decision.plan().unwrap();
    assert_eq!(plan.exposure_before, dec!(3));
    assert_eq!(plan.leg.side, Action::Sell);
    assert_eq!(plan.expected_exposure_after, dec!(0));
    let hedge = hedge_action(&sc, plan);
    assert_eq!(hedged(&sc, 1, &obs), hedge);

    // Sending it changes nothing the agent sees until the fill arrives.
    let fills = obs.delivered_fills;
    k.step(&mut s, hedge);
    assert_eq!(k.observe(&s).position, dec!(3));
    while k.observe(&s).delivered_fills == fills {
        k.step(&mut s, KernelAction::Hold);
    }
    assert_eq!(k.observe(&s).position, plan.expected_exposure_after);
}
