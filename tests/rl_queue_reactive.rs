//! The queue-reactive simulator as the market of an episode. All data here is simulated.

#![cfg(feature = "rl")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{ActionGrid, KernelConfig, decode, encode};
use market_maker_rs::backtest::queue_reactive::{
    QrParameters, QrSimulator, QrState, QrStep, UnsupportedPolicy,
};
use market_maker_rs::dec;
use market_maker_rs::rl::{
    EpisodeRecorder, MAX_QR_ROWS, MAX_REACTION_MOVES, ObservationScales, QrEpisode,
    ReactiveQrMarket, RlAction, RlConfig, RlEnv, qr_market,
};
use market_maker_rs::types::error::MMError;

fn rates() -> Vec<(f64, f64, f64)> {
    vec![
        (4.0, 0.0, 0.0),
        (4.0, 1.0, 1.0),
        (3.0, 2.0, 1.0),
        (2.0, 3.0, 1.0),
        (0.0, 4.0, 1.0),
    ]
}

fn params() -> QrParameters {
    QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1), dec!(2)],
        vec![rates(), rates()],
        vec![vec![true; 5]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap()
}

fn config() -> KernelConfig {
    KernelConfig::new(dec!(0.005), dec!(1), dec!(1000), dec!(10), 100_000)
}

fn episode(seed: u64) -> QrEpisode {
    QrEpisode {
        seed,
        reference_ticks: 10_000,
        start_ms: 5_000,
        sample_ms: 500,
        rows: 200,
    }
}

/// `(price, size)` of one side, worked out from the state as the model states it.
fn expected(
    queues: &[u32],
    limits: usize,
    sizes: &[Decimal],
    reference: i64,
    above: bool,
) -> (Decimal, Decimal) {
    let (depth, size) = queues[..limits]
        .iter()
        .enumerate()
        .find(|(_, queue)| **queue > 0)
        .map_or((limits, dec!(0)), |(depth, queue)| {
            (depth, Decimal::from(*queue) * sizes[depth])
        });
    let away = (Decimal::from(depth) + dec!(0.5)) * dec!(0.01);
    let base = Decimal::from(reference) * dec!(0.01);
    (if above { base + away } else { base - away }, size)
}

fn check_rows(params: &QrParameters, episode: &QrEpisode, cfg: &KernelConfig) -> usize {
    let market = qr_market(params, episode, cfg).unwrap();
    let mut simulator =
        QrSimulator::new(params.clone(), episode.seed, episode.reference_ticks).unwrap();
    let seconds = f64::from(episode.sample_ms) / 1000.0;
    let mut empty_sides = 0;
    let rows = market.dataset.ticks();
    assert_eq!(rows.len(), episode.rows as usize);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            simulator.run(seconds, seconds).unwrap();
        }
        let state: QrState = simulator.state();
        let limits = params.limits();
        let sizes = params.event_sizes();
        let (bid, bid_size) = expected(&state.bid, limits, sizes, state.reference_ticks, false);
        let (ask, ask_size) = expected(&state.ask, limits, sizes, state.reference_ticks, true);
        assert_eq!(Decimal::from(row.bid) * dec!(0.005), bid, "row {i}");
        assert_eq!(Decimal::from(row.ask) * dec!(0.005), ask, "row {i}");
        assert_eq!(Decimal::from(row.bid_lots), bid_size, "row {i}");
        assert_eq!(Decimal::from(row.ask_lots), ask_size, "row {i}");
        let ms = episode.start_ms + i as u64 * u64::from(episode.sample_ms);
        assert_eq!(row.ts_ns, ms * 1_000_000, "row {i}");
        empty_sides += usize::from(row.bid_lots == 0) + usize::from(row.ask_lots == 0);
    }
    assert_eq!(market.end, simulator.state());
    assert_eq!(market.stopped, None);
    empty_sides
}

#[test]
fn test_market_rows_are_the_simulator_books_on_the_time_grid() {
    check_rows(&params(), &episode(7), &config());
}

#[test]
fn test_market_side_with_no_volume_has_no_size_one_limit_further() {
    // One limit that empties often: its price then sits one limit further out.
    let table = vec![(1.0, 0.0, 0.0), (1.0, 2.0, 2.0), (0.0, 4.0, 2.0)];
    let thin = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1)],
        vec![table],
        vec![vec![true; 3]],
        UnsupportedPolicy::NearestSupported,
    )
    .unwrap();
    let empty = check_rows(&thin, &episode(3), &config());
    assert!(empty > 0, "the fixture must reach an empty side");
}

#[test]
fn test_market_is_reproduced_by_its_seed_and_changes_with_it() {
    let (p, cfg) = (params(), config());
    let a = qr_market(&p, &episode(7), &cfg).unwrap();
    let b = qr_market(&p, &episode(7), &cfg).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.dataset.id(), b.dataset.id());
    let c = qr_market(&p, &episode(8), &cfg).unwrap();
    assert_ne!(a.dataset.ticks(), c.dataset.ticks());
    assert_ne!(a.dataset.id(), c.dataset.id());
}

#[test]
fn test_market_identity_names_its_parameters_seed_and_reference() {
    let (p, cfg) = (params(), config());
    let market = qr_market(&p, &episode(7), &cfg).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(&p.fingerprint().to_le_bytes());
    expected.extend_from_slice(&7_u64.to_le_bytes());
    expected.extend_from_slice(&10_000_i64.to_le_bytes());
    assert_eq!(market.dataset.id().external.as_slice(), expected.as_slice());
}

#[test]
fn test_market_refuses_an_invalid_episode_or_grid() {
    let (p, cfg) = (params(), config());
    let with = |set: fn(&mut QrEpisode)| {
        let mut e = episode(1);
        set(&mut e);
        qr_market(&p, &e, &cfg)
    };
    assert!(matches!(
        with(|e| {
            e.sample_ms = 0;
            e.rows = 1;
        }),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        with(|e| e.rows = 0),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        with(|e| e.rows = MAX_QR_ROWS + 1),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        with(|e| e.start_ms = u64::MAX),
        Err(MMError::InvalidConfiguration(_) | MMError::InvalidMarketState(_))
    ));
    // A price at or below zero.
    assert!(matches!(
        with(|e| e.reference_ticks = 0),
        Err(MMError::InvalidMarketState(_))
    ));
    // The same tick as the simulator cannot hold a half tick.
    let same = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(10), 100);
    assert!(matches!(
        qr_market(&p, &episode(1), &same),
        Err(MMError::InvalidConfiguration(_))
    ));
    // An event size of 2 is not on a lot of 3.
    let lot = KernelConfig::new(dec!(0.005), dec!(3), dec!(1000), dec!(30), 100);
    let outer_only = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(2)],
        vec![rates()],
        vec![vec![true; 5]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    assert!(matches!(
        qr_market(&outer_only, &episode(1), &lot),
        Err(MMError::InvalidMarketState(_))
    ));
}

#[test]
fn test_market_stops_with_the_reason_and_pads_nothing() {
    // A queue of 3 is not simulated under the stop policy, and it is reachable.
    let limited = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1)],
        vec![rates()],
        vec![vec![true, true, true, false, false]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let e = episode(5);
    let market = qr_market(&limited, &e, &config()).unwrap();
    assert!(matches!(market.stopped, Some(QrStep::Unsupported { .. })));
    let rows = market.dataset.ticks();
    assert!(!rows.is_empty() && rows.len() < e.rows as usize);
    let last = rows.last().unwrap();
    let ms = e.start_ms + (rows.len() as u64 - 1) * u64::from(e.sample_ms);
    assert_eq!(last.ts_ns, ms * 1_000_000);
    // The end holds the state where it stopped, past the last row.
    let mut simulator = QrSimulator::new(limited.clone(), e.seed, e.reference_ticks).unwrap();
    let seconds = f64::from(e.sample_ms) / 1000.0;
    for _ in 1..rows.len() {
        simulator.run(seconds, seconds).unwrap();
    }
    let at_last_row = simulator.state();
    assert!(simulator.run(seconds, seconds).unwrap().stopped.is_some());
    assert_eq!(market.end, simulator.state());
    assert_ne!(market.end, at_last_row);
}

fn scales() -> ObservationScales {
    ObservationScales {
        spread_ticks: dec!(10),
        mid_change_ticks: dec!(10),
        row_gap_ms: dec!(1000),
        size_lots: dec!(10),
        position: dec!(10),
        entry_gap: dec!(1),
    }
}

fn action(step: usize) -> RlAction {
    if step.is_multiple_of(3) {
        RlAction::Quote {
            bid: 0,
            ask: 0,
            bid_size: 0,
            ask_size: 0,
        }
    } else {
        RlAction::Hold
    }
}

#[test]
fn test_env_runs_on_the_market_and_reconciles_with_the_evaluation() {
    let cfg = config().with_fee_rate(dec!(0.001));
    let market = qr_market(&params(), &episode(7), &cfg).unwrap();
    let grid = ActionGrid::new([1], [1], [1]).unwrap();
    let env: RlEnv<'_, 1, 1> = RlEnv::new(
        &cfg,
        &market.dataset,
        grid,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    let mut step = 0;
    while !state.is_done() {
        let t = env
            .step_with(&mut state, action(step), &mut recorder)
            .unwrap();
        step += 1;
        assert!(t.observation.values.iter().all(|v| v.abs() <= 1.0));
        env.reconcile(&state, &recorder).unwrap();
    }
    assert_eq!(step, 200);
    assert!(!env.records(&state, &recorder).unwrap().fills.is_empty());
}

#[test]
fn test_env_on_the_market_is_deterministic_and_restores_from_a_snapshot() {
    let cfg = config().with_fee_rate(dec!(0.001));
    let market = qr_market(&params(), &episode(7), &cfg).unwrap();
    let again = qr_market(&params(), &episode(7), &cfg).unwrap();
    let grid = ActionGrid::new([1], [1], [1]).unwrap();
    let rl = RlConfig::new(scales(), dec!(1));
    let env: RlEnv<'_, 1, 1> = RlEnv::new(&cfg, &market.dataset, grid, rl).unwrap();
    let twin: RlEnv<'_, 1, 1> = RlEnv::new(&cfg, &again.dataset, grid, rl).unwrap();
    assert_eq!(env.spec().id(), twin.spec().id());
    let (mut a, mut b) = (env.reset(1).unwrap(), twin.reset(1).unwrap());
    let mut snapshot = None;
    let mut after = Vec::new();
    let mut step = 0;
    while !a.is_done() {
        if step == 100 {
            snapshot = Some((a, step));
        }
        let (ta, tb) = (
            env.step(&mut a, action(step)).unwrap(),
            twin.step(&mut b, action(step)).unwrap(),
        );
        assert_eq!(ta, tb);
        assert_eq!(encode(a.kernel()), encode(b.kernel()));
        if step >= 100 {
            after.push(ta);
        }
        step += 1;
    }
    let (mut restored, from) = snapshot.unwrap();
    // The kernel's state survives its canonical bytes on another build of the same market.
    let bytes = encode(restored.kernel());
    assert_eq!(&decode(&bytes, twin.kernel()).unwrap(), restored.kernel());
    for (offset, expected) in after.iter().enumerate() {
        let t = env.step(&mut restored, action(from + offset)).unwrap();
        assert_eq!(&t, expected);
    }
    assert_eq!(restored, a);
}

#[test]
fn test_a_reactive_market_without_reactions_is_qr_markets() {
    let cfg = config();
    for e in [episode(7), episode(8)] {
        let plain = qr_market(&params(), &e, &cfg).unwrap();
        let mut reactive = ReactiveQrMarket::new(&params(), &e, &cfg).unwrap();
        assert_eq!(reactive.market().unwrap(), plain);
        reactive.react(50, 0).unwrap();
        assert_eq!(reactive.market().unwrap(), plain, "no moves change nothing");
    }
    // Stopped early: the same rows, reason and length; only the end differs.
    let limited = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1)],
        vec![rates()],
        vec![vec![true, true, true, false, false]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let plain = qr_market(&limited, &episode(5), &cfg).unwrap();
    let reactive = ReactiveQrMarket::new(&limited, &episode(5), &cfg)
        .unwrap()
        .market()
        .unwrap();
    assert_eq!(reactive.dataset, plain.dataset);
    assert_eq!(reactive.stopped, plain.stopped);
}

#[test]
fn test_an_imposed_move_empties_the_touch_it_passes_and_shifts_as_theta_does() {
    let mut simulator = QrSimulator::new(params(), 3, 100).unwrap();
    simulator.run(5.0, 5.0).unwrap();
    let mut state = simulator.state();
    state.bid = [2, 1, 0, 0, 0];
    state.ask = [1, 2, 0, 0, 0];
    let mut up = QrSimulator::from_state(params(), state).unwrap();
    up.impose_reference_move(true).unwrap();
    let after = up.state();
    assert_eq!(after.reference_ticks, state.reference_ticks + 1);
    // The emptied ask touch becomes the bid's first limit, the bid moves one limit out
    // (renormalized from size 1 to 2, rounded up: 2 -> 1), and the ask's second limit
    // comes nearer (2 of size 2 -> 4 of size 1).
    assert_eq!((after.bid[0], after.bid[1], after.ask[0]), (0, 1, 4));
    // The move is at the state's clock, its time moves up to it, and every clock restarts.
    let now = state.clock_seconds.max(state.time_seconds);
    assert_eq!(after.time_seconds, now);
    assert_eq!(after.bid_last, [now; 5]);
    assert_eq!(after.ask_last, [now; 5]);
    let mut down = QrSimulator::from_state(params(), state).unwrap();
    down.impose_reference_move(false).unwrap();
    let after = down.state();
    assert_eq!(after.reference_ticks, state.reference_ticks - 1);
    assert_eq!((after.ask[0], after.ask[1], after.bid[0]), (0, 1, 2));
}

#[test]
fn test_a_reaction_keeps_the_rows_up_to_it_and_simulates_the_rest_from_the_moved_state() {
    let cfg = config();
    let e = episode(7);
    let original = ReactiveQrMarket::new(&params(), &e, &cfg).unwrap();
    let mut reacted = original.clone();
    let emptied = reacted.react(60, -2).unwrap();
    assert_eq!(reacted.ticks().len(), original.ticks().len());
    assert_eq!(reacted.ticks()[..=60], original.ticks()[..=60]);
    assert_ne!(reacted.ticks()[61..], original.ticks()[61..]);
    assert_eq!(reacted.states()[..60], original.states()[..60]);
    // The rest is a direct run from the moved state, with the stored generator state.
    let mut direct = QrSimulator::from_state(params(), original.states()[60]).unwrap();
    let first = direct.state().bid[0];
    direct.impose_reference_move(false).unwrap();
    let second = direct.state().bid[0];
    direct.impose_reference_move(false).unwrap();
    // The first limit's event size is 1: the volume emptied is the two touches passed.
    assert_eq!(emptied, Decimal::from(first + second));
    assert_eq!(reacted.states()[60], direct.state());
    let seconds = f64::from(e.sample_ms) / 1000.0;
    for row in 61..reacted.states().len() {
        direct.run(seconds, seconds).unwrap();
        assert_eq!(reacted.states()[row], direct.state(), "row {row}");
    }
    assert_eq!(
        reacted.states()[61].reference_ticks,
        original.states()[60].reference_ticks - 2
            + (reacted.states()[61].reference_ticks - reacted.states()[60].reference_ticks)
    );
    // A second reaction after the same row adds to the first, and the same reactions
    // give the same market.
    let mut twice = original.clone();
    twice.react(60, -1).unwrap();
    twice.react(60, -1).unwrap();
    assert_eq!(twice, reacted);
    let mut again = original.clone();
    again.react(60, -2).unwrap();
    assert_eq!(again.market().unwrap(), reacted.market().unwrap());
    assert!(original.clone().react(200, 1).is_err(), "past the last row");
    let bound = i64::try_from(MAX_REACTION_MOVES).unwrap();
    assert!(original.clone().react(60, bound + 1).is_err());
    assert!(original.clone().react(60, -bound - 1).is_err());
    assert!(original.clone().react(60, i64::MIN).is_err());
}

#[test]
fn test_after_an_imposed_move_waits_run_from_the_move_and_the_generator_is_accounted() {
    let mut simulator = QrSimulator::new(params(), 11, 100).unwrap();
    simulator.run(5.0, 5.0).unwrap();
    let state = simulator.state();
    let mut moved = QrSimulator::from_state(params(), state).unwrap();
    moved.impose_reference_move(true).unwrap();
    let at = moved.state();
    // One outer queue is drawn from the simulator's own generator, the same way each time.
    assert_ne!(at.rng_state, state.rng_state);
    let mut twice = QrSimulator::from_state(params(), state).unwrap();
    twice.impose_reference_move(true).unwrap();
    assert_eq!(twice.state(), at);
    // The first event after the move waits from the move, not from the queue's last event
    // before it: the clocks restarted there, as after a move of θ.
    let move_time = at.clock_seconds;
    match moved.step().unwrap() {
        QrStep::Event { observation, .. } => {
            let now = moved.state().time_seconds;
            assert!((observation.waiting_seconds - (now - move_time)).abs() < 1e-12);
        }
        other => panic!("{other:?}"),
    }
    // No move leaves the state, the generator included, as it was.
    let cfg = config();
    let market = ReactiveQrMarket::new(&params(), &episode(11), &cfg).unwrap();
    let mut same = market.clone();
    same.react(30, 0).unwrap();
    assert_eq!(same.states(), market.states());
}

#[test]
fn test_states_on_rows_that_differ_later_are_the_same_apart_from_the_dataset() {
    let cfg = config().with_fee_rate(dec!(0.001));
    let original = ReactiveQrMarket::new(&params(), &episode(7), &cfg).unwrap();
    let mut moved = original.clone();
    moved.react(120, 2).unwrap();
    let (a, b) = (original.market().unwrap(), moved.market().unwrap());
    assert_ne!(a.dataset.id(), b.dataset.id());
    let grid = ActionGrid::new([1], [1], [1]).unwrap();
    let env = |data| -> RlEnv<'_, 1, 1> {
        RlEnv::new(&cfg, data, grid, RlConfig::new(scales(), dec!(1))).unwrap()
    };
    let (env_a, env_b) = (env(&a.dataset), env(&b.dataset));
    let mut state_a = env_a.reset(1).unwrap();
    let mut state_b = env_b.reset(1).unwrap();
    let mut diverged = false;
    for step in 0..199 {
        let row = env_a.observe(&state_a).unwrap().row.unwrap_or(0);
        if row < 120 {
            assert!(state_a.same_apart_from_dataset(&state_b), "step {step}");
            assert!(state_a.kernel().same_apart_from_dataset(state_b.kernel()));
            assert_ne!(state_a, state_b, "the dataset's identity differs");
        } else if !state_a.same_apart_from_dataset(&state_b) {
            diverged = true;
        }
        env_a.step(&mut state_a, action(step)).unwrap();
        env_b.step(&mut state_b, action(step)).unwrap();
    }
    assert!(diverged, "the later rows differ, so the states do");
    // Different runs on the same rows are not the same.
    let other = env_a.reset(2).unwrap();
    assert!(!env_a.reset(1).unwrap().same_apart_from_dataset(&other));
}
