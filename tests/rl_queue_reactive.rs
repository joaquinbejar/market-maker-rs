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
    EpisodeRecorder, MAX_QR_ROWS, ObservationScales, QrEpisode, RlAction, RlConfig, RlEnv,
    qr_market,
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
