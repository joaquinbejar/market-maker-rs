//! The queue-reactive simulator against hand fixtures and known parameters.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::backtest::queue_reactive::*;
use market_maker_rs::dec;
use market_maker_rs::prelude::Decimal;
use market_maker_rs::strategy::walk_forward::TimeWindow;
use market_maker_rs::types::error::MMError;

const CAP: usize = 12;

/// Queue reactive rates: arrivals fall and departures rise with the queue.
fn table(scale: f64) -> RateTable {
    (0..=CAP)
        .map(|n| {
            let size = n as f64;
            let limit = if n == CAP {
                0.0
            } else {
                scale * 2.0 / (1.0 + 0.3 * size)
            };
            let cancel = if n == 0 {
                0.0
            } else {
                scale * (0.1 + 0.1 * size)
            };
            let market = if n == 0 {
                0.0
            } else {
                scale * 0.3 / (1.0 + 0.5 * size)
            };
            (limit, cancel, market)
        })
        .collect()
}

fn params() -> QrParameters {
    QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1), dec!(1)],
        vec![table(1.0), table(0.5)],
        vec![vec![true; CAP + 1]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap()
}

fn config() -> QrEstimationConfig {
    estimation_config(2, CAP as u32, 30, 10, UnsupportedPolicy::NearestSupported)
}

fn estimation_config(
    limits: u8,
    max_queue: u32,
    min_events: u64,
    min_reference_moves: u64,
    policy: UnsupportedPolicy,
) -> QrEstimationConfig {
    QrEstimationConfig::new(
        dec!(0.01),
        limits,
        max_queue,
        min_events,
        min_reference_moves,
        policy,
    )
}

fn state_of(simulator: &QrSimulator, bid: [u32; MAX_LIMITS], ask: [u32; MAX_LIMITS]) -> QrState {
    QrState {
        bid,
        ask,
        ..simulator.state()
    }
}

fn window(end_ms: u64) -> TimeWindow {
    TimeWindow::new(0, end_ms).unwrap()
}

#[test]
fn test_invariant_law_matches_the_closed_form() {
    // rho(0) = 1 / 1 and rho(1) = 2 / 4: weights 1, 1, 1/2, total 5/2.
    let rates = vec![(1.0, 0.0, 0.0), (2.0, 0.5, 0.5), (0.0, 3.0, 1.0)];
    let p = QrParameters::new(
        dec!(0.01),
        0.5,
        0.0,
        vec![dec!(1)],
        vec![rates],
        vec![vec![true; 3]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let law = p.invariant(0).unwrap();
    for (got, want) in law.iter().zip([0.4, 0.4, 0.2]) {
        assert!((got - want).abs() < 1e-12, "{got} {want}");
    }
    assert!(p.invariant(1).is_none());
}

#[test]
fn test_simulated_queue_law_converges_to_the_invariant() {
    let p = params();
    let mut sim = QrSimulator::new(
        QrParameters::new(
            dec!(0.01),
            0.0,
            0.0,
            vec![dec!(1)],
            vec![table(1.0)],
            vec![vec![true; CAP + 1]],
            UnsupportedPolicy::Stop,
        )
        .unwrap(),
        3,
        100,
    )
    .unwrap();
    // With theta 0 the queues are independent birth and death processes.
    let t = sim.run(200_000.0, 5.0).unwrap();
    let mut counts = [0f64; CAP + 1];
    for s in &t.samples {
        counts[s.bid[0] as usize] += 1.0;
        counts[s.ask[0] as usize] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    let law = p.invariant(0).unwrap();
    let tv: f64 = counts
        .iter()
        .zip(law)
        .map(|(c, l)| (c / total - l).abs())
        .sum::<f64>()
        / 2.0;
    assert!(tv < 0.02, "{tv}");
}

#[test]
fn test_step_book_rules_hold_over_a_long_run() {
    let p = params();
    let mut sim = QrSimulator::new(p, 11, 1_000).unwrap();
    let mut last_time = 0.0;
    let (mut reference_moves, mut limits_seen) = (0, [0u64; 3]);
    for _ in 0..200_000 {
        let before = sim.state();
        let QrStep::Event {
            observation,
            reference,
        } = sim.step().unwrap()
        else {
            panic!("stopped");
        };
        let after = sim.state();
        assert!(after.time_seconds > last_time);
        last_time = after.time_seconds;
        for q in after.bid.iter().chain(&after.ask) {
            assert!(*q as usize <= CAP);
        }
        assert_eq!(after.bid[2..], [0; 3]);
        assert_eq!(after.ask[2..], [0; 3]);
        let queues = match observation.side {
            QrSide::Bid => (before.bid, after.bid),
            QrSide::Ask => (before.ask, after.ask),
            _ => panic!("side"),
        };
        let d = usize::from(observation.depth);
        assert_eq!(observation.queue_before, queues.0[d]);
        let moved = reference.is_some_and(|r| r.moved);
        match observation.kind {
            QrEventKind::Limit => {
                limits_seen[0] += 1;
                if !moved {
                    assert_eq!(queues.1[d], queues.0[d] + 1);
                }
            }
            QrEventKind::Cancel | QrEventKind::Market => {
                limits_seen[1] += 1;
                assert!(queues.0[d] > 0);
                if !moved {
                    assert_eq!(queues.1[d], queues.0[d] - 1);
                }
            }
            _ => panic!("kind"),
        }
        if moved {
            reference_moves += 1;
            assert_eq!((after.reference_ticks - before.reference_ticks).abs(), 1);
        } else {
            assert_eq!(after.reference_ticks, before.reference_ticks);
        }
    }
    assert!(reference_moves > 100, "{reference_moves}");
}

#[test]
fn test_shift_moves_queues_with_the_reference_and_renormalizes() {
    // Sizes 1 and 2. Ask [1, 3], bid [5, 7]. Up when the last at ask 1 leaves:
    // new ask 1 is 3 * 2 / 1 = 6, new bid 1 is the emptied 0, new bid 2 is
    // ceil(5 * 1 / 2) = 3, and new ask 2 comes from the law, here always 4.
    let mut outer: RateTable = vec![(0.0, 0.0, 0.0); 21];
    outer[4] = (0.0, 0.0, 0.0);
    let mut inner: RateTable = vec![(0.0, 0.0, 0.0); 21];
    inner[1] = (0.0, 1.0, 0.0);
    // The second limit's law sits on 0 unless it can grow: make 0..4 pass.
    for n in 0..4 {
        outer[n].0 = 1.0;
        outer[n + 1].1 = 1.0;
    }
    let p = QrParameters::new(
        dec!(0.01),
        1.0,
        0.0,
        vec![dec!(1), dec!(2)],
        vec![inner, outer],
        vec![vec![true; 21]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let start = QrSimulator::new(p.clone(), 5, 100).unwrap();
    let mut state = state_of(&start, [5, 7, 0, 0, 0], [1, 3, 0, 0, 0]);
    state.bid[0] = 5;
    // Only the first ask queue can move in the first event for these seeds; run
    // until it does, on a chain where other first queue rates are zero at 5.
    let mut sim = QrSimulator::from_state(p, state).unwrap();
    loop {
        let before = sim.state();
        let QrStep::Event {
            observation,
            reference,
        } = sim.step().unwrap()
        else {
            panic!("stopped");
        };
        if observation.side == QrSide::Ask && observation.depth == 0 {
            assert_eq!(reference.map(|r| r.moved), Some(true));
            let after = sim.state();
            assert_eq!(after.reference_ticks, before.reference_ticks + 1);
            assert_eq!(after.ask[0], before.ask[1] * 2);
            assert_eq!(after.bid[0], 0);
            assert_eq!(after.bid[1], before.bid[0].div_ceil(2));
            assert!(after.ask[1] <= 4);
            break;
        }
    }
}

/// Limit 1 only loses the last order; limit 2 grows to at most 4. Event sizes 1, 2.
fn shift_params() -> QrParameters {
    let mut inner: RateTable = vec![(0.0, 0.0, 0.0); 21];
    inner[1] = (0.0, 1.0, 0.0);
    let mut outer: RateTable = vec![(0.0, 0.0, 0.0); 21];
    for n in 0..4 {
        outer[n].0 = 1.0;
        outer[n + 1].1 = 1.0;
    }
    QrParameters::new(
        dec!(0.01),
        1.0,
        0.0,
        vec![dec!(1), dec!(2)],
        vec![inner, outer],
        vec![vec![true; 21]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap()
}

#[test]
fn test_shift_down_is_the_mirror_of_up() {
    let p = shift_params();
    let start = QrSimulator::new(p.clone(), 5, 100).unwrap();
    let mut sim =
        QrSimulator::from_state(p, state_of(&start, [1, 3, 0, 0, 0], [5, 7, 0, 0, 0])).unwrap();
    loop {
        let before = sim.state();
        let QrStep::Event {
            observation,
            reference,
        } = sim.step().unwrap()
        else {
            panic!("stopped");
        };
        if observation.side == QrSide::Bid && observation.depth == 0 {
            assert_eq!(reference.map(|r| r.moved), Some(true));
            let after = sim.state();
            assert_eq!(after.reference_ticks, before.reference_ticks - 1);
            assert_eq!(after.bid[0], before.bid[1] * 2);
            assert_eq!(after.ask[0], 0);
            assert_eq!(after.ask[1], before.ask[0].div_ceil(2));
            assert!(after.bid[1] <= 4);
            break;
        }
    }
}

#[test]
fn test_mid_move_with_the_opposite_queue_occupied_is_not_eligible() {
    // Arrivals only at an empty first queue, which moves the mid toward the other
    // side while that side's first queue holds an order.
    let rates: RateTable = vec![(1.0, 0.0, 0.0), (0.0, 1e-9, 0.0), (0.0, 0.0, 0.0)];
    let far: RateTable = vec![(0.0, 0.0, 0.0); 3];
    let p = QrParameters::new(
        dec!(0.01),
        1.0,
        0.0,
        vec![dec!(1), dec!(1)],
        vec![rates, far],
        vec![vec![true; 3]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let start = QrSimulator::new(p.clone(), 5, 100).unwrap();
    for (bid, ask) in [
        ([0, 1, 0, 0, 0], [1, 1, 0, 0, 0]),
        ([1, 1, 0, 0, 0], [0, 1, 0, 0, 0]),
    ] {
        let mut sim = QrSimulator::from_state(p.clone(), state_of(&start, bid, ask)).unwrap();
        let QrStep::Event { reference, .. } = sim.step().unwrap() else {
            panic!("stopped");
        };
        assert_eq!(reference, None, "{bid:?} {ask:?}");
        assert_eq!(sim.state().reference_ticks, 100);
    }
}

#[test]
fn test_reinit_one_redraws_the_book_at_every_move() {
    let p = QrParameters::new(
        dec!(0.01),
        1.0,
        1.0,
        vec![dec!(1)],
        vec![table(1.0)],
        vec![vec![true; CAP + 1]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let mut sim = QrSimulator::new(p, 9, 0).unwrap();
    let (mut moves, mut occupied) = (0, 0);
    for _ in 0..50_000 {
        let QrStep::Event { reference, .. } = sim.step().unwrap() else {
            panic!("stopped");
        };
        if let Some(r) = reference {
            assert!(r.moved && r.reinitialized);
            moves += 1;
            // Without a redraw, the side the price moved toward has an emptied
            // first queue on the other side.
            let s = sim.state();
            if s.bid[0] > 0 && s.ask[0] > 0 {
                occupied += 1;
            }
        }
    }
    assert!(moves > 100 && occupied > moves / 2, "{moves} {occupied}");
}

#[test]
fn test_time_that_cannot_advance_is_stalled() {
    let p = params();
    let start = QrSimulator::new(p.clone(), 5, 100).unwrap();
    let state = QrState {
        time_seconds: 1e18,
        clock_seconds: 1e18,
        ..start.state()
    };
    let mut sim = QrSimulator::from_state(p, state).unwrap();
    assert_eq!(sim.step().unwrap(), QrStep::Stalled);
}

#[test]
fn test_run_stops_at_its_event_limit() {
    let mut sim = QrSimulator::new(params(), 5, 100).unwrap();
    let t = sim.run_limited(1_000.0, 1.0, 10).unwrap();
    assert_eq!(t.events.len(), 10);
    assert_eq!(t.stopped, Some(QrStep::EventLimit));
}

#[test]
fn test_run_reports_no_event_past_the_horizon_and_samples_the_prior_book() {
    let mut sim = QrSimulator::new(params(), 5, 100).unwrap();
    let t = sim.run(100.0, 1.0).unwrap();
    assert!(t.events.iter().all(|e| e.timestamp_ms <= 100_000));
    assert_eq!(t.samples.len(), 100);
    assert_eq!(
        t.events.len(),
        t.events
            .iter()
            .filter(|e| e.timestamp_ms <= 100_000)
            .count()
    );
    // Replay by hand: each sample is the book before the first event after it.
    let mut replay = QrSimulator::new(params(), 5, 100).unwrap();
    let mut samples = t.samples.iter();
    let mut next = samples.next();
    while let Some(sample) = next {
        let before = replay.state();
        replay.step().unwrap();
        if replay.state().time_seconds * 1_000.0 >= sample.timestamp_ms as f64 {
            assert_eq!((sample.bid, sample.ask), (before.bid, before.ask));
            next = samples.next();
        }
    }
}

#[test]
fn test_run_does_not_apply_the_event_past_the_horizon() {
    let mut sim = QrSimulator::new(params(), 7, 100).unwrap();
    let start = sim.state();
    let t = sim.run(1e-6, 5e-7).unwrap();
    assert!(t.events.is_empty());
    assert_eq!(
        sim.state(),
        QrState {
            clock_seconds: 1e-6,
            ..start
        }
    );
    // Stepping on gives the event a run with no horizon would.
    let mut whole = QrSimulator::new(params(), 7, 100).unwrap();
    assert_eq!(sim.step().unwrap(), whole.step().unwrap());
}

#[test]
fn test_reached_size_without_departures_is_refused() {
    let build = |rates: RateTable| {
        let rows = rates.len();
        QrParameters::new(
            dec!(0.01),
            0.5,
            0.0,
            vec![dec!(1)],
            vec![rates],
            vec![vec![true; rows]],
            UnsupportedPolicy::Stop,
        )
    };
    assert!(build(vec![(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]).is_err());
    assert!(build(vec![(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]).is_err());
    let p = build(vec![(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]).unwrap();
    assert_eq!(p.invariant(0).unwrap(), [0.5, 0.5]);
    // Size 2 has no departures, but size 1 is never reached, so neither is it.
    let p = build(vec![(0.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.0, 0.0, 0.0)]).unwrap();
    assert_eq!(p.invariant(0).unwrap(), [1.0, 0.0, 0.0]);
    // No arrival at all: the law sits at empty, and that is stationary.
    let p = build(vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]).unwrap();
    assert_eq!(p.invariant(0).unwrap(), [1.0, 0.0]);
}

#[test]
fn test_consecutive_runs_lose_no_event() {
    let mut pieces = QrSimulator::new(params(), 5, 100).unwrap();
    let mut kept = Vec::new();
    for _ in 0..1_000 {
        kept.extend(pieces.run(1.0, 1.0).unwrap().events);
    }
    assert!(kept.len() > 1_000);
    assert!((pieces.state().clock_seconds - 1_000.0).abs() < 1e-6);
    let long = QrSimulator::new(params(), 5, 100)
        .unwrap()
        .run(1_000.0, 1.0)
        .unwrap();
    assert_eq!(kept.len(), long.events.len());
    let mut whole = QrSimulator::new(params(), 5, 100).unwrap();
    for event in &kept {
        let QrStep::Event { observation, .. } = whole.step().unwrap() else {
            panic!("stopped");
        };
        assert_eq!(observation, *event);
    }
}

#[test]
fn test_second_run_places_its_grid_after_the_first_horizon() {
    let mut sim = QrSimulator::new(params(), 5, 100).unwrap();
    let first = sim.run(10.5, 1.0).unwrap();
    assert_eq!(first.samples.last().unwrap().timestamp_ms, 10_000);
    let second = sim.run(2.0, 1.0).unwrap();
    let times: Vec<_> = second.samples.iter().map(|s| s.timestamp_ms).collect();
    assert_eq!(times, [11_500, 12_500]);
}

#[test]
fn test_shift_caps_a_renormalized_queue() {
    // Sizes 1 and 100: a queue of 3 at the second limit is 300 at the first,
    // above the cap of 20.
    let mut inner: RateTable = vec![(0.0, 0.0, 0.0); 21];
    inner[1] = (0.0, 1.0, 0.0);
    let outer: RateTable = vec![(0.0, 0.0, 0.0); 21];
    let p = QrParameters::new(
        dec!(0.01),
        1.0,
        0.0,
        vec![dec!(1), dec!(100)],
        vec![inner, outer],
        vec![vec![true; 21]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let start = QrSimulator::new(p.clone(), 5, 100).unwrap();
    let mut sim =
        QrSimulator::from_state(p, state_of(&start, [0, 0, 0, 0, 0], [1, 3, 0, 0, 0])).unwrap();
    sim.step().unwrap();
    assert_eq!(sim.state().ask[0], 20);
}

fn flow_event(timestamp_ms: u64, kind: QrEventKind, waiting_seconds: f64) -> QueueObservation {
    QueueObservation {
        timestamp_ms,
        side: QrSide::Bid,
        depth: 0,
        kind,
        queue_before: 1,
        waiting_seconds,
    }
}

fn flow_trajectory(events: Vec<QueueObservation>, start: u64, end: u64) -> Trajectory {
    let sample = |timestamp_ms| BookSample {
        timestamp_ms,
        bid: [1, 0, 0, 0, 0],
        ask: [1, 0, 0, 0, 0],
        spread_ticks: 1,
    };
    let mut t = Trajectory::default();
    t.samples = vec![sample(start), sample(end)];
    t.events = events;
    t
}

#[test]
fn test_compare_flow_is_a_rate_and_a_law_whatever_the_clock() {
    let p = holes(UnsupportedPolicy::Stop);
    // Same counts, one on an epoch clock: 4 events in 4 s, in intervals of 1 s.
    let at = |start: u64| {
        let events = (0..4)
            .map(|i| flow_event(start + i * 1_000, QrEventKind::Limit, 1.0))
            .collect();
        flow_trajectory(events, start, start + 4_000)
    };
    let epoch = 1_700_000_000_000;
    let r = compare_trajectories(&at(epoch), &at(0), &p, 1_000).unwrap();
    assert_eq!(r.flow, [0.0; 3]);
    assert_eq!(r.flow_law, [0.0; 3]);

    // Same mean rate, bursty: 4 in the first interval, against 1 in each.
    let bursty: Vec<_> = (0..4)
        .map(|i| flow_event(i, QrEventKind::Limit, 1.0))
        .collect();
    let r = compare_trajectories(&at(0), &flow_trajectory(bursty, 0, 4_000), &p, 1_000).unwrap();
    assert_eq!(r.flow[0], 0.0);
    assert_eq!(r.flow_law[0], 1.0);

    assert!(compare_trajectories(&at(0), &at(0), &p, 0).is_err());
    assert!(
        compare_trajectories(&at(0), &at(0), &p, 5_000).is_err(),
        "no whole interval"
    );
    assert!(compare_trajectories(&at(0), &at(0), &p, 1).is_ok());
}

#[test]
fn test_compare_waiting_law_tells_constant_from_varied_with_the_same_mean() {
    let p = holes(UnsupportedPolicy::Stop);
    let with = |waits: &[f64]| {
        let events = waits
            .iter()
            .map(|w| flow_event(0, QrEventKind::Limit, *w))
            .collect();
        flow_trajectory(events, 0, 4_000)
    };
    let r = compare_trajectories(&with(&[1.0, 3.0]), &with(&[2.0, 2.0]), &p, 1_000).unwrap();
    assert_eq!(r.waiting_time, [0.0]);
    assert_eq!(r.waiting_law, [0.5]);
    let same = compare_trajectories(&with(&[1.0, 3.0]), &with(&[3.0, 1.0]), &p, 1_000).unwrap();
    assert_eq!(same.waiting_law, [0.0]);
}

#[test]
fn test_compare_invalid_waiting_is_counted_and_left_out() {
    let p = holes(UnsupportedPolicy::Stop);
    let with = |waits: &[f64]| {
        let events = waits
            .iter()
            .map(|w| flow_event(0, QrEventKind::Limit, *w))
            .collect();
        flow_trajectory(events, 0, 4_000)
    };
    let clean = with(&[2.0, 2.0]);
    for bad in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
        let r = compare_trajectories(&with(&[2.0, 2.0, bad]), &clean, &p, 1_000).unwrap();
        assert_eq!(r.data_invalid_waiting, 1, "{bad}");
        assert_eq!(r.model_invalid_waiting, 0);
        assert_eq!(r.waiting_time, [0.0], "{bad}");
        let r = compare_trajectories(&clean, &with(&[2.0, 2.0, bad]), &p, 1_000).unwrap();
        assert_eq!(r.model_invalid_waiting, 1, "{bad}");
        assert_eq!(r.waiting_time, [0.0], "{bad}");
    }
}

#[test]
fn test_compare_reports_a_short_or_stopped_model() {
    let p = holes(UnsupportedPolicy::Stop);
    let data = flow_trajectory(vec![flow_event(0, QrEventKind::Limit, 1.0)], 0, 4_000);
    let mut model = flow_trajectory(vec![flow_event(0, QrEventKind::Limit, 1.0)], 0, 2_000);
    model.events.push(flow_event(1, QrEventKind::Cancel, 1.0));
    model.stopped = Some(QrStep::Stalled);
    let r = compare_trajectories(&data, &model, &p, 1_000).unwrap();
    assert_eq!((r.data_samples, r.data_events), (2, 1));
    assert_eq!((r.model_samples, r.model_events), (2, 2));
    assert_eq!(r.model_stopped, Some(QrStep::Stalled));
}

#[test]
fn test_rate_above_the_bound_is_refused() {
    let mut rates = table(1.0);
    rates[1].1 = MAX_RATE * 2.0;
    assert!(
        QrParameters::new(
            dec!(0.01),
            0.5,
            0.0,
            vec![dec!(1)],
            vec![rates],
            vec![vec![true; CAP + 1]],
            UnsupportedPolicy::Stop
        )
        .is_err()
    );
}

#[test]
fn test_nearest_policy_keeps_an_empty_queue_from_departing() {
    // Size 0 is unsupported, and its nearest row departs.
    let mut flags = vec![true; CAP + 1];
    flags[0] = false;
    let p = QrParameters::new(
        dec!(0.01),
        0.5,
        0.0,
        vec![dec!(1)],
        vec![table(1.0)],
        vec![flags],
        UnsupportedPolicy::NearestSupported,
    )
    .unwrap();
    let t = QrSimulator::new(p, 5, 0)
        .unwrap()
        .run(5_000.0, 1.0)
        .unwrap();
    assert!(t.stopped.is_none());
    assert!(t.events.iter().any(|e| e.queue_before == 0));
    assert!(
        t.events
            .iter()
            .filter(|e| e.queue_before == 0)
            .all(|e| e.kind == QrEventKind::Limit)
    );
}

#[test]
fn test_constant_baseline_applies_the_policy_to_unsupported_sizes() {
    // Rows above 2 are zero in the table and carry mass by the nearest policy.
    let mut rates = table(1.0);
    for row in rates.iter_mut().skip(3) {
        *row = (0.0, 0.0, 0.0);
    }
    let mut flags = vec![true; CAP + 1];
    for flag in flags.iter_mut().skip(3) {
        *flag = false;
    }
    let p = QrParameters::new(
        dec!(0.01),
        0.5,
        0.0,
        vec![dec!(1)],
        vec![rates],
        vec![flags],
        UnsupportedPolicy::NearestSupported,
    )
    .unwrap();
    let mut expected = 0.0;
    let row2 = p.rates(0).unwrap()[2];
    let law = p.invariant(0).unwrap();
    for (n, weight) in law.iter().enumerate() {
        if n < CAP {
            expected += weight
                * if n <= 2 {
                    p.rates(0).unwrap()[n].0
                } else {
                    row2.0
                };
        }
    }
    let got = p.constant_baseline().unwrap().rates(0).unwrap()[1].0;
    assert!((got - expected).abs() < 1e-12, "{got} {expected}");
}

#[test]
fn test_estimate_ignores_events_before_the_window_and_applies_the_exact_minimum() {
    let event = |timestamp_ms, kind, queue_before| QueueObservation {
        timestamp_ms,
        side: QrSide::Bid,
        depth: 0,
        kind,
        queue_before,
        waiting_seconds: 1.0,
    };
    let reference = ReferenceObservation {
        timestamp_ms: 10,
        moved: true,
        reinitialized: false,
    };
    let cfg = estimation_config(1, 3, 2, 1, UnsupportedPolicy::NearestSupported);
    let w = TimeWindow::new(10, 100).unwrap();
    let inside = [
        event(10, QrEventKind::Limit, 1),
        event(11, QrEventKind::Cancel, 1),
        event(12, QrEventKind::Limit, 2),
    ];
    let run = |e: &[QueueObservation]| {
        estimate_parameters(e, &[reference], &[dec!(1)], &w, &cfg, "d").unwrap()
    };
    let plain = run(&inside);
    let mut with_earlier = vec![
        event(1, QrEventKind::Limit, 3),
        event(2, QrEventKind::Market, 0),
    ];
    with_earlier.extend(inside);
    assert_eq!(run(&with_earlier), plain);
    let fit = plain.parameters().unwrap();
    assert!(fit.is_supported(0, 1), "exactly the minimum of 2");
    assert!(!fit.is_supported(0, 2), "one below it");
}

#[test]
fn test_estimate_subnormal_waiting_time_is_a_typed_failure() {
    let event = QueueObservation {
        timestamp_ms: 1,
        side: QrSide::Bid,
        depth: 0,
        kind: QrEventKind::Limit,
        queue_before: 1,
        waiting_seconds: 5e-324,
    };
    let reference = ReferenceObservation {
        timestamp_ms: 1,
        moved: true,
        reinitialized: false,
    };
    let cfg = estimation_config(1, 3, 1, 1, UnsupportedPolicy::NearestSupported);
    let r =
        estimate_parameters(&[event], &[reference], &[dec!(1)], &window(100), &cfg, "d").unwrap();
    assert_eq!(r.fit, QrFit::Insufficient(QrFailure::InvalidEstimate));
}

#[test]
fn test_reference_does_not_move_with_theta_zero_and_always_with_one() {
    let run = |theta: f64| {
        let p = QrParameters::new(
            dec!(0.01),
            theta,
            0.0,
            vec![dec!(1)],
            vec![table(1.0)],
            vec![vec![true; CAP + 1]],
            UnsupportedPolicy::Stop,
        )
        .unwrap();
        QrSimulator::new(p, 9, 0)
            .unwrap()
            .run(20_000.0, 10.0)
            .unwrap()
    };
    let none = run(0.0);
    assert!(!none.references.is_empty());
    assert!(none.references.iter().all(|r| !r.moved && !r.reinitialized));
    let all = run(1.0);
    assert!(all.references.iter().all(|r| r.moved && !r.reinitialized));
}

#[test]
fn test_same_parameters_and_seed_reproduce_exactly_and_another_seed_differs() {
    let run = |seed| {
        QrSimulator::new(params(), seed, 100)
            .unwrap()
            .run(2_000.0, 1.0)
            .unwrap()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5).events, run(6).events);
}

#[test]
fn test_restored_state_continues_identically() {
    let mut whole = QrSimulator::new(params(), 5, 100).unwrap();
    let first = whole.run(500.0, 1.0).unwrap();
    assert!(!first.events.is_empty());
    let stored = whole.state();
    let mut restored = QrSimulator::from_state(params(), stored).unwrap();
    for _ in 0..1_000 {
        assert_eq!(whole.step().unwrap(), restored.step().unwrap());
    }
    assert_eq!(whole.state(), restored.state());
}

#[test]
fn test_from_state_refuses_a_state_that_does_not_fit() {
    let sim = QrSimulator::new(params(), 5, 100).unwrap();
    let good = sim.state();
    assert!(QrSimulator::from_state(params(), good).is_ok());
    let edits: [fn(&mut QrState); 7] = [
        |s| s.bid[0] = CAP as u32 + 1,
        |s| s.ask[3] = 1,
        |s| s.time_seconds = f64::NAN,
        |s| s.bid_last[0] = 1.0,
        |s| s.clock_seconds = -1.0,
        |s| s.rng_state = 0,
        |s| s.parameters_fingerprint ^= 1,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut bad = good;
        edit(&mut bad);
        assert!(QrSimulator::from_state(params(), bad).is_err(), "edit {i}");
    }
}

#[test]
fn test_parameters_refuse_invalid_values() {
    let build = |theta: f64, reinit: f64, rates: RateTable, sizes: Vec<Decimal>, tick: Decimal| {
        let rows = rates.len();
        QrParameters::new(
            tick,
            theta,
            reinit,
            sizes,
            vec![rates],
            vec![vec![true; rows]],
            UnsupportedPolicy::Stop,
        )
    };
    let good = table(1.0);
    assert!(build(0.5, 0.5, good.clone(), vec![dec!(1)], dec!(0.01)).is_ok());
    assert!(build(1.1, 0.5, good.clone(), vec![dec!(1)], dec!(0.01)).is_err());
    assert!(build(-0.1, 0.5, good.clone(), vec![dec!(1)], dec!(0.01)).is_err());
    assert!(build(0.5, 1.1, good.clone(), vec![dec!(1)], dec!(0.01)).is_err());
    assert!(build(f64::NAN, 0.5, good.clone(), vec![dec!(1)], dec!(0.01)).is_err());
    assert!(build(0.5, 0.5, good.clone(), vec![dec!(0)], dec!(0.01)).is_err());
    assert!(build(0.5, 0.5, good.clone(), vec![dec!(1)], dec!(0)).is_err());
    assert!(build(0.5, 0.5, good.clone(), vec![], dec!(0.01)).is_err());
    let mut edit = good.clone();
    edit[0].1 = 1.0;
    assert!(
        build(0.5, 0.5, edit, vec![dec!(1)], dec!(0.01)).is_err(),
        "cancel from empty"
    );
    let mut edit = good.clone();
    edit[0].2 = 1.0;
    assert!(
        build(0.5, 0.5, edit, vec![dec!(1)], dec!(0.01)).is_err(),
        "market from empty"
    );
    let mut edit = good.clone();
    edit[CAP].0 = 1.0;
    assert!(
        build(0.5, 0.5, edit, vec![dec!(1)], dec!(0.01)).is_err(),
        "arrival at the cap"
    );
    let mut edit = good.clone();
    edit[1].1 = -1.0;
    assert!(build(0.5, 0.5, edit, vec![dec!(1)], dec!(0.01)).is_err());
    let mut edit = good;
    edit[1].0 = f64::INFINITY;
    assert!(build(0.5, 0.5, edit, vec![dec!(1)], dec!(0.01)).is_err());
    let rates = vec![table(1.0); 6];
    assert!(
        QrParameters::new(
            dec!(0.01),
            0.5,
            0.5,
            vec![dec!(1); 6],
            rates,
            vec![vec![true; CAP + 1]; 6],
            UnsupportedPolicy::Stop
        )
        .is_err()
    );
    assert!(
        QrParameters::new(
            dec!(0.01),
            0.5,
            0.5,
            vec![dec!(1)],
            vec![table(1.0)],
            vec![vec![false; CAP + 1]],
            UnsupportedPolicy::Stop
        )
        .is_err()
    );
}

fn holes(policy: UnsupportedPolicy) -> QrParameters {
    let mut supported = vec![true; CAP + 1];
    for flag in supported.iter_mut().skip(3) {
        *flag = false;
    }
    QrParameters::new(
        dec!(0.01),
        0.5,
        0.0,
        vec![dec!(1)],
        vec![table(1.0)],
        vec![supported],
        policy,
    )
    .unwrap()
}

#[test]
fn test_unsupported_state_stop_reports_and_leaves_the_state() {
    let p = holes(UnsupportedPolicy::Stop);
    let start = QrSimulator::new(p.clone(), 5, 0).unwrap();
    let state = state_of(&start, [5, 0, 0, 0, 0], [1, 0, 0, 0, 0]);
    let mut sim = QrSimulator::from_state(p, state).unwrap();
    assert_eq!(
        sim.step().unwrap(),
        QrStep::Unsupported {
            side: QrSide::Bid,
            depth: 0,
            queue: 5
        }
    );
    assert_eq!(sim.state(), state);
}

#[test]
fn test_unsupported_state_nearest_uses_the_nearest_supported_rates() {
    // Only limit orders at the nearest supported size 2 of a table whose rows
    // above are different: a queue at 8 departs at the rate of size 2.
    let p = holes(UnsupportedPolicy::NearestSupported);
    let start = QrSimulator::new(p.clone(), 5, 0).unwrap();
    let state = state_of(&start, [8, 0, 0, 0, 0], [8, 0, 0, 0, 0]);
    let mut sim = QrSimulator::from_state(p, state).unwrap();
    assert!(matches!(sim.step().unwrap(), QrStep::Event { .. }));
    // At the law: the nearest row lets a queue climb past the supported range.
    let t = QrSimulator::new(holes(UnsupportedPolicy::NearestSupported), 5, 0)
        .unwrap()
        .run(5_000.0, 1.0)
        .unwrap();
    assert!(t.stopped.is_none());
    assert!(t.events.iter().any(|e| e.queue_before > 2));
}

#[test]
fn test_all_rates_zero_is_stalled() {
    let rates = vec![(0.0, 0.0, 0.0); 3];
    let p = QrParameters::new(
        dec!(0.01),
        0.5,
        0.0,
        vec![dec!(1)],
        vec![rates],
        vec![vec![true; 3]],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    let mut sim = QrSimulator::new(p, 5, 0).unwrap();
    assert_eq!(sim.step().unwrap(), QrStep::Stalled);
    let t = sim.run(10.0, 1.0).unwrap();
    assert_eq!(t.stopped, Some(QrStep::Stalled));
}

fn generate(seed: u64, horizon: f64) -> Trajectory {
    QrSimulator::new(params(), seed, 100)
        .unwrap()
        .run(horizon, 5.0)
        .unwrap()
}

#[test]
fn test_estimate_recovers_known_rates_and_probabilities() {
    let t = generate(21, 300_000.0);
    let end = t.events.last().unwrap().timestamp_ms + 1;
    let record = estimate_parameters(
        &t.events,
        &t.references,
        &[dec!(1), dec!(1)],
        &window(end),
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(record.excluded_events, 0);
    assert_eq!(record.events, t.events.len() as u64);
    let fit = record.parameters().expect("calibrated");
    assert!((fit.theta() - 0.6).abs() < 0.02, "{}", fit.theta());
    assert!(
        (fit.theta_reinit() - 0.2).abs() < 0.03,
        "{}",
        fit.theta_reinit()
    );
    let truth = params();
    for depth in 0..2 {
        for n in 0..=5usize {
            let (a, b) = (fit.rates(depth).unwrap()[n], truth.rates(depth).unwrap()[n]);
            for (got, want) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                assert!(
                    (got - want).abs() < 0.05 * want.max(0.2),
                    "d{depth} n{n} {got} {want}"
                );
            }
        }
    }
}

#[test]
fn test_estimate_changing_later_data_leaves_the_record_unchanged() {
    let t = generate(21, 20_000.0);
    let end = 10_000_000;
    let w = window(end);
    let cfg = config();
    let before =
        estimate_parameters(&t.events, &t.references, &[dec!(1), dec!(1)], &w, &cfg, "d").unwrap();
    let (mut e, mut r) = (t.events.clone(), t.references.clone());
    e.iter_mut()
        .filter(|x| x.timestamp_ms >= end)
        .for_each(|x| x.waiting_seconds *= 5.0);
    r.iter_mut()
        .filter(|x| x.timestamp_ms >= end)
        .for_each(|x| x.moved = !x.moved);
    assert_eq!(
        before,
        estimate_parameters(&e, &r, &[dec!(1), dec!(1)], &w, &cfg, "d").unwrap()
    );
    let inside = |edit: fn(&mut QueueObservation)| {
        let mut c = t.events.clone();
        edit(&mut c[100]);
        estimate_parameters(&c, &t.references, &[dec!(1), dec!(1)], &w, &cfg, "d")
            .unwrap()
            .dataset_hash
    };
    let edits: [fn(&mut QueueObservation); 6] = [
        |x| x.waiting_seconds += 1.0,
        |x| x.queue_before += 1,
        |x| x.timestamp_ms = x.timestamp_ms.saturating_sub(0).max(x.timestamp_ms),
        |x| x.depth = 1 - x.depth,
        |x| {
            x.side = if x.side == QrSide::Bid {
                QrSide::Ask
            } else {
                QrSide::Bid
            }
        },
        |x| {
            x.kind = if x.kind == QrEventKind::Limit {
                QrEventKind::Cancel
            } else {
                QrEventKind::Limit
            }
        },
    ];
    for (i, edit) in edits.iter().enumerate() {
        if i == 2 {
            continue;
        }
        assert_ne!(inside(*edit), before.dataset_hash, "field {i}");
    }
    let mut c = t.references.clone();
    c[0].moved = !c[0].moved;
    let moved = estimate_parameters(&t.events, &c, &[dec!(1), dec!(1)], &w, &cfg, "d").unwrap();
    assert_ne!(moved.dataset_hash, before.dataset_hash);
}

#[test]
fn test_estimate_counts_invalid_events_and_reports_each_failure() {
    let good = QueueObservation {
        timestamp_ms: 1,
        side: QrSide::Bid,
        depth: 0,
        kind: QrEventKind::Limit,
        queue_before: 1,
        waiting_seconds: 1.0,
    };
    let reference = ReferenceObservation {
        timestamp_ms: 1,
        moved: true,
        reinitialized: false,
    };
    let sizes = [dec!(1)];
    let cfg = estimation_config(1, 3, 1, 1, UnsupportedPolicy::NearestSupported);
    let run = |events: &[QueueObservation], refs: &[ReferenceObservation]| {
        estimate_parameters(events, refs, &sizes, &window(100), &cfg, "d").unwrap()
    };
    for bad in [
        QueueObservation {
            waiting_seconds: 0.0,
            ..good
        },
        QueueObservation {
            waiting_seconds: -1.0,
            ..good
        },
        QueueObservation {
            waiting_seconds: f64::NAN,
            ..good
        },
        QueueObservation { depth: 1, ..good },
        QueueObservation {
            queue_before: 4,
            ..good
        },
        QueueObservation {
            queue_before: 3,
            ..good
        },
        QueueObservation {
            queue_before: 0,
            kind: QrEventKind::Cancel,
            ..good
        },
        QueueObservation {
            queue_before: 0,
            kind: QrEventKind::Market,
            ..good
        },
    ] {
        let record = run(&[good, bad], &[reference]);
        assert_eq!((record.events, record.excluded_events), (1, 1), "{bad:?}");
    }
    assert_eq!(
        run(&[good], &[]).fit,
        QrFit::Insufficient(QrFailure::TooFewReferenceMoves { have: 0, need: 1 })
    );
    let not_moved = ReferenceObservation {
        moved: false,
        ..reference
    };
    assert_eq!(
        run(&[good], &[not_moved]).fit,
        QrFit::Insufficient(QrFailure::ReferenceNeverMoved)
    );
    assert_eq!(
        run(&[], &[reference]).fit,
        QrFit::Insufficient(QrFailure::NoSupportedSize { depth: 0 })
    );
    assert!(
        estimate_parameters(
            &[
                good,
                QueueObservation {
                    timestamp_ms: 0,
                    ..good
                }
            ],
            &[],
            &sizes,
            &window(100),
            &cfg,
            "d"
        )
        .is_err()
    );
    assert!(estimate_parameters(&[], &[], &[], &window(100), &cfg, "d").is_err());
}

#[test]
fn test_estimate_marks_sizes_below_min_events_unsupported() {
    let t = generate(21, 20_000.0);
    let end = t.events.last().unwrap().timestamp_ms + 1;
    let record = estimate_parameters(
        &t.events,
        &t.references,
        &[dec!(1), dec!(1)],
        &window(end),
        &config(),
        "d",
    )
    .unwrap();
    let fit = record.parameters().unwrap();
    let mut counted = [0u64; CAP + 1];
    for e in t.events.iter().filter(|e| e.depth == 0) {
        counted[e.queue_before as usize] += 1;
    }
    for (n, count) in counted.iter().enumerate() {
        assert_eq!(fit.is_supported(0, n), *count >= 30, "n {n} count {count}");
    }
}

#[test]
fn test_constant_baseline_has_constant_rates_and_keeps_the_rules() {
    let baseline = params().constant_baseline().unwrap();
    let rates = baseline.rates(0).unwrap();
    assert_eq!(rates[0].1, 0.0);
    assert_eq!(rates[0].2, 0.0);
    assert_eq!(rates[CAP].0, 0.0);
    assert_eq!(rates[1], rates[5]);
    assert!(rates[1].0 > 0.0 && rates[1].1 > 0.0 && rates[1].2 > 0.0);
    assert_eq!(baseline.theta(), params().theta());
}

#[test]
fn test_out_of_sample_report_fits_better_than_constant_intensity() {
    let train = generate(31, 200_000.0);
    let end = train.events.last().unwrap().timestamp_ms + 1;
    let record = estimate_parameters(
        &train.events,
        &train.references,
        &[dec!(1), dec!(1)],
        &window(end),
        &config(),
        "train",
    )
    .unwrap();
    let fit = record.parameters().unwrap();
    let baseline = fit.constant_baseline().unwrap();
    let day = 20_000.0;
    println!(
        "| day | coverage | queue tv 1 fit/const | queue tv 2 fit/const | spread tv fit/const | wait err 1 fit/const | market flow err fit/const |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for (index, seed) in (41..46).enumerate() {
        let data = generate(seed, day);
        let run = |p: &QrParameters, s: u64| {
            QrSimulator::new(p.clone(), s, 100)
                .unwrap()
                .run(day, 5.0)
                .unwrap()
        };
        let a = compare_trajectories(&data, &run(fit, seed + 100), fit, 60_000).unwrap();
        let b = compare_trajectories(&data, &run(&baseline, seed + 200), fit, 60_000).unwrap();
        println!(
            "| {index} | {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} |",
            a.coverage,
            a.queue_size[0],
            b.queue_size[0],
            a.queue_size[1],
            b.queue_size[1],
            a.spread,
            b.spread,
            a.waiting_law[0],
            b.waiting_law[0],
            a.flow_law[2],
            b.flow_law[2],
            a.flow[2],
            b.flow[2],
        );
        assert!(a.queue_size[0] < b.queue_size[0], "{a:?} {b:?}");
        assert!(a.queue_size[1] < b.queue_size[1], "{a:?} {b:?}");
        assert!(a.spread < b.spread, "{a:?} {b:?}");
        assert!(a.waiting_law[0] < b.waiting_law[0], "{a:?} {b:?}");
        assert_eq!(a.model_stopped, None);
        assert!(a.coverage > 0.9);
        assert_eq!(a.data_events, data.events.len() as u64);
        assert_eq!(a.data_samples, data.samples.len() as u64);
    }
}

#[test]
fn test_compare_hand_values_and_empty_trajectory_is_refused() {
    let sample = |bid0, spread| BookSample {
        timestamp_ms: 1_000,
        bid: [bid0, 0, 0, 0, 0],
        ask: [bid0, 0, 0, 0, 0],
        spread_ticks: spread,
    };
    let event = |kind, waiting| QueueObservation {
        timestamp_ms: 1,
        side: QrSide::Bid,
        depth: 0,
        kind,
        queue_before: 1,
        waiting_seconds: waiting,
    };
    let trajectory = |samples, events| {
        let mut t = Trajectory::default();
        t.samples = samples;
        t.events = events;
        t
    };
    let p = holes(UnsupportedPolicy::Stop);
    let data = trajectory(
        vec![sample(1, 1), sample(2, 1)],
        vec![
            event(QrEventKind::Limit, 2.0),
            event(QrEventKind::Limit, 2.0),
        ],
    );
    let model = trajectory(
        vec![sample(1, 1), sample(1, 3)],
        vec![
            event(QrEventKind::Limit, 1.0),
            event(QrEventKind::Cancel, 1.0),
        ],
    );
    let r = compare_trajectories(&data, &model, &p, 500).unwrap();
    assert_eq!(r.queue_size, [0.5]);
    assert_eq!(r.spread, 0.5);
    assert_eq!(r.waiting_time, [0.5]);
    assert_eq!(r.flow[0], 0.5);
    assert_eq!(r.coverage, 1.0);
    let partly = trajectory(
        vec![sample(1, 1)],
        vec![
            event(QrEventKind::Limit, 1.0),
            QueueObservation {
                queue_before: 5,
                ..event(QrEventKind::Limit, 1.0)
            },
        ],
    );
    assert_eq!(
        compare_trajectories(&partly, &model, &p, 500)
            .unwrap()
            .coverage,
        0.5
    );
    let empty = Trajectory::default();
    assert!(compare_trajectories(&empty, &model, &p, 500).is_err());
    assert!(compare_trajectories(&data, &empty, &p, 500).is_err());
}

#[test]
fn test_run_invalid_grid_is_refused() {
    let mut sim = QrSimulator::new(params(), 5, 100).unwrap();
    assert!(sim.run(0.0, 1.0).is_err());
    assert!(sim.run(10.0, 0.0).is_err());
    assert!(sim.run(f64::NAN, 1.0).is_err());
    assert!(sim.run(10_000_000.0, 1.0).is_err());
}

#[test]
fn test_spread_counts_empty_sides_one_limit_further() {
    let p = params();
    let sim = QrSimulator::new(p.clone(), 5, 100).unwrap();
    let at = |bid, ask| {
        QrSimulator::from_state(p.clone(), state_of(&sim, bid, ask))
            .unwrap()
            .spread_ticks()
    };
    assert_eq!(at([1, 0, 0, 0, 0], [1, 0, 0, 0, 0]), 1);
    assert_eq!(at([0, 1, 0, 0, 0], [1, 0, 0, 0, 0]), 2);
    assert_eq!(at([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), 5);
}

#[test]
fn test_fingerprint_covers_the_parameters() {
    let base = params().fingerprint();
    let with = |theta, sizes: Vec<Decimal>| {
        QrParameters::new(
            dec!(0.01),
            theta,
            0.2,
            sizes,
            vec![table(1.0), table(0.5)],
            vec![vec![true; CAP + 1]; 2],
            UnsupportedPolicy::Stop,
        )
        .unwrap()
        .fingerprint()
    };
    assert_eq!(with(0.6, vec![dec!(1), dec!(1)]), base);
    assert_ne!(with(0.5, vec![dec!(1), dec!(1)]), base);
    assert_ne!(with(0.6, vec![dec!(1), dec!(2)]), base);
    let rates = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1), dec!(1)],
        vec![table(1.0), table(0.6)],
        vec![vec![true; CAP + 1]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    assert_ne!(rates.fingerprint(), base);
}

#[cfg(feature = "serde")]
mod decoding {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn test_parameters_export_reproduces_a_trajectory() {
        let text = serde_json::to_string(&params()).unwrap();
        let restored: QrParameters = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, params());
        assert_eq!(restored.fingerprint(), params().fingerprint());
        let run = |p: QrParameters| {
            QrSimulator::new(p, 5, 100)
                .unwrap()
                .run(500.0, 1.0)
                .unwrap()
        };
        assert_eq!(run(restored), run(params()));
    }

    #[test]
    fn test_decode_refuses_invalid_parameters() {
        let base: Value = serde_json::from_str(&serde_json::to_string(&params()).unwrap()).unwrap();
        let edits: [fn(&mut Value); 5] = [
            |v| v["theta"] = json!("2.0"),
            |v| v["rates"][0][0] = json!(["1.0", "1.0", "0.0"]),
            |v| v["theta_reinit"] = json!("not a number"),
            |v| v["tick_size"] = json!("0"),
            |v| v["extra"] = json!(1),
        ];
        for (i, edit) in edits.iter().enumerate() {
            let mut bad = base.clone();
            edit(&mut bad);
            assert!(
                serde_json::from_value::<QrParameters>(bad).is_err(),
                "edit {i}"
            );
        }
    }

    #[test]
    fn test_state_round_trip_continues_identically() {
        let mut sim = QrSimulator::new(params(), 5, 100).unwrap();
        for _ in 0..100 {
            sim.step().unwrap();
        }
        let text = serde_json::to_string(&sim.state()).unwrap();
        let state: QrState = serde_json::from_str(&text).unwrap();
        assert_eq!(state, sim.state());
        let mut restored = QrSimulator::from_state(params(), state).unwrap();
        assert_eq!(sim.step().unwrap(), restored.step().unwrap());
    }
}

fn sized(timestamp_ms: u64, depth: u8, size: Decimal) -> SizedEvent {
    SizedEvent {
        timestamp_ms,
        depth,
        size,
    }
}

fn size_window() -> TimeWindow {
    TimeWindow::new(100, 200).unwrap()
}

#[test]
fn test_event_sizes_are_the_mean_per_limit_inside_the_window() {
    let events = [
        sized(99, 0, dec!(1000)),
        sized(100, 0, dec!(1)),
        sized(110, 1, dec!(10)),
        sized(120, 0, dec!(2)),
        sized(199, 1, dec!(20)),
        sized(200, 0, dec!(1000)),
    ];
    let record = estimate_event_sizes(&events, &size_window(), 2, 2).unwrap();
    assert_eq!(record.sizes, Ok(vec![dec!(1.5), dec!(15)]));
    assert_eq!(record.events, [2, 2]);
    assert_eq!(record.excluded_events, 0);
}

#[test]
fn test_event_sizes_a_later_event_does_not_change_the_record() {
    let mut events = vec![sized(100, 0, dec!(1))];
    let first = estimate_event_sizes(&events, &size_window(), 1, 1).unwrap();
    events.push(sized(200, 0, dec!(5)));
    assert_eq!(
        first,
        estimate_event_sizes(&events, &size_window(), 1, 1).unwrap()
    );
    events.insert(1, sized(150, 0, dec!(5)));
    let other = estimate_event_sizes(&events, &size_window(), 1, 1).unwrap();
    assert_ne!(first.dataset_hash, other.dataset_hash);
    // The scale of a decimal does not name another dataset.
    let scaled = [sized(100, 0, dec!(1.00))];
    assert_eq!(
        first.dataset_hash,
        estimate_event_sizes(&scaled, &size_window(), 1, 1)
            .unwrap()
            .dataset_hash
    );
}

#[test]
fn test_event_sizes_leave_out_what_is_not_a_size_at_a_limit() {
    let events = [
        sized(100, 0, dec!(0)),
        sized(101, 0, dec!(-3)),
        sized(102, 2, dec!(9)),
        sized(103, 0, dec!(4)),
        sized(104, 1, dec!(6)),
    ];
    let record = estimate_event_sizes(&events, &size_window(), 2, 1).unwrap();
    assert_eq!(record.excluded_events, 3);
    assert_eq!(record.sizes, Ok(vec![dec!(4), dec!(6)]));
}

#[test]
fn test_event_sizes_a_thin_limit_names_itself() {
    let events = [
        sized(100, 0, dec!(1)),
        sized(101, 0, dec!(1)),
        sized(102, 1, dec!(1)),
    ];
    let record = estimate_event_sizes(&events, &size_window(), 2, 2).unwrap();
    assert_eq!(
        record.sizes,
        Err(QrFailure::TooFewSizedEvents {
            depth: 1,
            have: 1,
            need: 2,
        })
    );
    // At the minimum exactly, it is enough.
    assert!(
        estimate_event_sizes(&events, &size_window(), 2, 1)
            .unwrap()
            .sizes
            .is_ok()
    );
}

#[test]
fn test_event_sizes_refuse_bad_settings_order_and_overflow() {
    let ok = [sized(100, 0, dec!(1))];
    for (limits, min) in [(0, 1), (MAX_LIMITS as u8 + 1, 1), (1, 0)] {
        assert!(matches!(
            estimate_event_sizes(&ok, &size_window(), limits, min),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    assert!(estimate_event_sizes(&ok, &size_window(), MAX_LIMITS as u8, 1).is_ok());
    let unordered = [sized(101, 0, dec!(1)), sized(100, 0, dec!(1))];
    assert!(matches!(
        estimate_event_sizes(&unordered, &size_window(), 1, 1),
        Err(MMError::InvalidMarketState(_))
    ));
    let huge = [sized(100, 0, Decimal::MAX), sized(101, 0, Decimal::MAX)];
    assert!(matches!(
        estimate_event_sizes(&huge, &size_window(), 1, 1),
        Err(MMError::NumericalError(_))
    ));
}

#[cfg(feature = "serde")]
#[test]
fn test_event_size_record_serde_round_trip_both_outcomes() {
    let events = [sized(100, 0, dec!(1.5))];
    for min in [1, 2] {
        let record = estimate_event_sizes(&events, &size_window(), 1, min).unwrap();
        assert_eq!(record.sizes.is_ok(), min == 1);
        let json = serde_json::to_string(&record).unwrap();
        assert_eq!(
            serde_json::from_str::<EventSizeRecord>(&json).unwrap(),
            record
        );
    }
}

#[test]
fn test_event_sizes_the_hash_covers_the_size() {
    let hash = |size| {
        estimate_event_sizes(&[sized(100, 0, size)], &size_window(), 1, 1)
            .unwrap()
            .dataset_hash
    };
    assert_ne!(hash(dec!(1)), hash(dec!(2)));
}

#[test]
fn test_event_sizes_the_hash_covers_the_time_and_the_limit() {
    let hash = |timestamp_ms, depth| {
        estimate_event_sizes(&[sized(timestamp_ms, depth, dec!(1))], &size_window(), 2, 1)
            .unwrap()
            .dataset_hash
    };
    assert_ne!(hash(100, 0), hash(101, 0));
    assert_ne!(hash(100, 0), hash(100, 1));
}

#[test]
fn test_event_sizes_name_the_first_of_two_thin_limits() {
    let events = [sized(100, 0, dec!(1)), sized(101, 0, dec!(1))];
    let record = estimate_event_sizes(&events, &size_window(), 3, 2).unwrap();
    assert_eq!(
        record.sizes,
        Err(QrFailure::TooFewSizedEvents {
            depth: 1,
            have: 0,
            need: 2,
        })
    );
}

#[test]
fn test_event_sizes_reach_the_parameters_they_are_for() {
    let t = generate(21, 300_000.0);
    let end = t.events.last().unwrap().timestamp_ms + 1;
    let sized_events: Vec<SizedEvent> = t
        .events
        .iter()
        .map(|e| sized(e.timestamp_ms, e.depth, dec!(100) + Decimal::from(e.depth)))
        .collect();
    let record = estimate_event_sizes(&sized_events, &window(end), 2, 1).unwrap();
    let sizes = record.sizes.unwrap();
    assert_eq!(sizes, [dec!(100), dec!(101)]);
    let fit = estimate_parameters(
        &t.events,
        &t.references,
        &sizes,
        &window(end),
        &config(),
        "d",
    )
    .unwrap();
    assert_eq!(fit.parameters().expect("calibrated").event_sizes(), sizes);
}

fn queue_event(
    side: QrSide,
    kind: QrEventKind,
    queue_before: u32,
    waiting_seconds: f64,
) -> QueueObservation {
    QueueObservation {
        timestamp_ms: 7,
        side,
        depth: 0,
        kind,
        queue_before,
        waiting_seconds,
    }
}

#[test]
fn test_collapse_keeps_a_block_net_change_in_place() {
    use QrEventKind::{Cancel, Limit};
    let other = queue_event(QrSide::Ask, Cancel, 5, 0.25);
    let events = [
        queue_event(QrSide::Bid, Limit, 3, 2.0),
        other,
        queue_event(QrSide::Bid, Limit, 4, 0.0),
        queue_event(QrSide::Bid, Cancel, 5, 0.0),
        queue_event(QrSide::Bid, Limit, 4, 0.0),
    ];
    assert_eq!(
        collapse_blocks(&events),
        vec![
            queue_event(QrSide::Bid, Limit, 3, 2.0),
            queue_event(QrSide::Bid, Limit, 4, 0.0),
            other,
        ]
    );
}

#[test]
fn test_collapse_names_a_net_departure_by_whether_it_holds_a_market_order() {
    use QrEventKind::{Cancel, Limit, Market};
    let with_market = [
        queue_event(QrSide::Bid, Cancel, 4, 1.0),
        queue_event(QrSide::Bid, Market, 3, 0.0),
        queue_event(QrSide::Bid, Limit, 2, 0.0),
        queue_event(QrSide::Bid, Cancel, 3, 0.0),
    ];
    assert_eq!(
        collapse_blocks(&with_market),
        vec![
            queue_event(QrSide::Bid, Market, 4, 1.0),
            queue_event(QrSide::Bid, Market, 3, 0.0),
        ]
    );
    let cancels = [
        queue_event(QrSide::Ask, Cancel, 4, 1.0),
        queue_event(QrSide::Ask, Cancel, 3, 0.0),
    ];
    assert_eq!(collapse_blocks(&cancels), cancels.to_vec());
}

#[test]
fn test_collapse_carries_a_zero_net_block_wait_to_its_queue_next_event() {
    use QrEventKind::{Cancel, Limit};
    let events = [
        queue_event(QrSide::Bid, Limit, 2, 1.5),
        queue_event(QrSide::Bid, Cancel, 3, 0.0),
        queue_event(QrSide::Ask, Limit, 1, 0.75),
        queue_event(QrSide::Bid, Cancel, 2, 0.5),
    ];
    assert_eq!(
        collapse_blocks(&events),
        vec![
            queue_event(QrSide::Ask, Limit, 1, 0.75),
            queue_event(QrSide::Bid, Cancel, 2, 2.0),
        ]
    );
}

#[test]
fn test_collapse_starts_a_block_at_a_wait_or_another_queue() {
    use QrEventKind::{Cancel, Limit};
    let waited = [
        queue_event(QrSide::Bid, Limit, 2, 1.0),
        queue_event(QrSide::Bid, Cancel, 3, 0.5),
    ];
    assert_eq!(collapse_blocks(&waited), waited.to_vec());
    let sized = [
        queue_event(QrSide::Bid, Limit, 2, 1.0),
        queue_event(QrSide::Bid, Limit, 7, 0.0),
    ];
    assert_eq!(
        collapse_blocks(&sized),
        vec![
            queue_event(QrSide::Bid, Limit, 2, 1.0),
            queue_event(QrSide::Bid, Limit, 3, 0.0),
        ],
        "one event size each, from the first event's queue"
    );
    let other_side = [
        queue_event(QrSide::Bid, Limit, 2, 1.0),
        queue_event(QrSide::Ask, Cancel, 3, 0.0),
    ];
    assert_eq!(collapse_blocks(&other_side), other_side.to_vec());
    let other_limit = [
        queue_event(QrSide::Bid, Limit, 2, 1.0),
        QueueObservation {
            depth: 1,
            ..queue_event(QrSide::Bid, Cancel, 3, 0.0)
        },
    ];
    assert_eq!(collapse_blocks(&other_limit), other_limit.to_vec());
}

#[test]
fn test_zero_waits_are_dropped_by_default_and_counted_on_request() {
    use QrEventKind::{Cancel, Limit};
    let events = [
        queue_event(QrSide::Bid, Limit, 1, 1.0),
        queue_event(QrSide::Bid, Limit, 2, 0.0),
        queue_event(QrSide::Bid, Cancel, 3, 0.0),
        queue_event(QrSide::Bid, Cancel, 2, -1.0),
    ];
    let reference = ReferenceObservation {
        timestamp_ms: 7,
        moved: true,
        reinitialized: false,
    };
    let base = estimation_config(1, 3, 1, 1, UnsupportedPolicy::NearestSupported);
    assert_eq!(base.zero_waits, ZeroWaitPolicy::Drop);
    let run = |cfg: &QrEstimationConfig| {
        estimate_parameters(&events, &[reference], &[dec!(1)], &window(100), cfg, "d").unwrap()
    };
    let dropped = run(&base);
    assert_eq!((dropped.events, dropped.excluded_events), (1, 3));
    let counted = run(&base.with_zero_waits(ZeroWaitPolicy::Count));
    assert_eq!((counted.events, counted.excluded_events), (3, 1));
}

#[cfg(feature = "serde")]
#[test]
fn test_a_config_without_zero_waits_reads_as_dropping_them() {
    let cfg = estimation_config(1, 3, 1, 1, UnsupportedPolicy::NearestSupported)
        .with_zero_waits(ZeroWaitPolicy::Count);
    let mut value = serde_json::to_value(cfg).unwrap();
    assert_eq!(value["zero_waits"], "Count");
    value.as_object_mut().unwrap().remove("zero_waits");
    let read: QrEstimationConfig = serde_json::from_value(value).unwrap();
    assert_eq!(read, cfg.with_zero_waits(ZeroWaitPolicy::Drop));
}

#[test]
fn test_counted_size_with_no_time_is_unsupported_not_absorbing() {
    use QrEventKind::{Cancel, Limit};
    // Size 2 is only ever passed through inside a block: its events wait zero.
    let mut events = Vec::new();
    for i in 0..40u64 {
        for (k, (kind, queue_before, waiting_seconds)) in [
            (Limit, 1, 1.0),
            (Limit, 2, 0.0),
            (Cancel, 3, 1.0),
            (Cancel, 2, 0.0),
            (Cancel, 1, 1.0),
            (Limit, 0, 1.0),
        ]
        .into_iter()
        .enumerate()
        {
            events.push(QueueObservation {
                timestamp_ms: 10 * i + k as u64,
                kind,
                queue_before,
                waiting_seconds,
                ..queue_event(QrSide::Bid, Limit, 0, 0.0)
            });
        }
    }
    let reference = ReferenceObservation {
        timestamp_ms: 0,
        moved: true,
        reinitialized: false,
    };
    let cfg = estimation_config(1, 4, 30, 1, UnsupportedPolicy::NearestSupported)
        .with_zero_waits(ZeroWaitPolicy::Count);
    let record =
        estimate_parameters(&events, &[reference], &[dec!(1)], &window(1_000), &cfg, "d").unwrap();
    let fit = record
        .parameters()
        .expect("a size with no time has no rate, not a zero one");
    assert!(fit.is_supported(0, 0) && fit.is_supported(0, 1) && fit.is_supported(0, 3));
    assert!(!fit.is_supported(0, 2), "80 events, but no time at size 2");
}

/// A redraw law: a quarter on each of sizes 1 to 4.
fn reset_law() -> Vec<f64> {
    (0..=CAP)
        .map(|n| if (1..=4).contains(&n) { 0.25 } else { 0.0 })
        .collect()
}

fn with_resets(rate: f64) -> QrParameters {
    let rates = (0..=CAP).map(|n| if n == 0 { 0.0 } else { rate }).collect();
    params().with_resets(rates, reset_law()).unwrap()
}

#[test]
fn test_resets_are_validated_and_fingerprinted() {
    let ok = |n: usize| vec![0.0; n];
    let law = reset_law;
    let refused = |rates: Vec<f64>, law: Vec<f64>| params().with_resets(rates, law).is_err();
    assert!(params().reset_rates().is_none() && params().reset_law().is_none());
    assert!(refused(ok(CAP), law()), "one rate per size");
    assert!(refused(ok(CAP + 2), law()));
    for bad in [-1.0, f64::NAN, f64::INFINITY, MAX_RATE * 2.0] {
        let mut rates = ok(CAP + 1);
        rates[3] = bad;
        assert!(refused(rates, law()), "{bad}");
    }
    let mut at_zero = ok(CAP + 1);
    at_zero[0] = 0.5;
    assert!(refused(at_zero, law()), "an empty touch has no life");
    let mut short = law();
    short.pop();
    assert!(refused(ok(CAP + 1), short), "one probability per size");
    let mut negative = law();
    negative[1] = -0.25;
    negative[5] = 0.5;
    assert!(refused(ok(CAP + 1), negative));
    let mut over = law();
    over[5] = 1e-6;
    assert!(refused(ok(CAP + 1), over), "a law sums to 1");
    let mut nan = law();
    nan[6] = f64::NAN;
    assert!(refused(ok(CAP + 1), nan));
    let zero = params().with_resets(ok(CAP + 1), law()).unwrap();
    assert_eq!(zero.reset_rates(), Some(&ok(CAP + 1)[..]));
    assert_eq!(zero.reset_law(), Some(&law()[..]));
    assert_ne!(zero.fingerprint(), params().fingerprint());
    assert_ne!(
        with_resets(0.3).fingerprint(),
        with_resets(0.4).fingerprint()
    );
    let mut other = law();
    other.swap(1, 6);
    let moved = params().with_resets(ok(CAP + 1), other).unwrap();
    assert_ne!(zero.fingerprint(), moved.fingerprint());
}

#[test]
fn test_zero_reset_rates_leave_a_run_unchanged() {
    let run = |p: QrParameters| {
        QrSimulator::new(p, 9, 100)
            .unwrap()
            .run(2_000.0, 1.0)
            .unwrap()
    };
    let plain = run(params());
    let zero = run(params()
        .with_resets(vec![0.0; CAP + 1], reset_law())
        .unwrap());
    assert_eq!(plain.events, zero.events);
    assert_eq!(plain.samples, zero.samples);
    assert!(zero.resets.is_empty());
}

#[test]
fn test_a_reset_redraws_the_touch_queue_only() {
    let mut sim = QrSimulator::new(with_resets(50.0), 3, 100).unwrap();
    let mut seen = 0;
    for _ in 0..2_000 {
        let before = sim.state();
        match sim.step().unwrap() {
            QrStep::Reset {
                reset,
                queue_after,
                reference,
            } => {
                seen += 1;
                let after = sim.state();
                let (touch, other) = match reset.side {
                    QrSide::Bid => ((before.bid, after.bid), (before.ask, after.ask)),
                    _ => ((before.ask, after.ask), (before.bid, after.bid)),
                };
                assert_eq!(reset.queue_before, touch.0[0]);
                assert!(reset.queue_before > 0);
                assert_eq!(reset.queue_after, Some(queue_after));
                assert!((1..=4).contains(&queue_after), "drawn from the reset law");
                if reference.is_none_or(|r| !r.moved) {
                    assert_eq!(touch.1[0], queue_after);
                    assert_eq!(touch.0[1..], touch.1[1..]);
                    assert_eq!(other.0, other.1);
                    assert_eq!(after.reference_ticks, before.reference_ticks);
                }
                let last = match reset.side {
                    QrSide::Bid => before.bid_last[0],
                    _ => before.ask_last[0],
                };
                assert_eq!(reset.waiting_seconds, after.time_seconds - last);
                assert_eq!(after.events, before.events, "a reset is no queue event");
            }
            QrStep::Event { .. } => {}
            other => panic!("{other:?}"),
        }
    }
    assert!(seen > 100, "{seen}");
}

#[test]
fn test_estimate_recovers_resets_with_the_other_rates() {
    let truth = with_resets(0.3);
    let t = QrSimulator::new(truth.clone(), 21, 100)
        .unwrap()
        .run(200_000.0, 5.0)
        .unwrap();
    assert!(!t.resets.is_empty());
    let end = t
        .events
        .last()
        .unwrap()
        .timestamp_ms
        .max(t.resets.last().unwrap().timestamp_ms)
        + 1;
    let fit = |resets: &[ResetObservation]| {
        estimate_parameters_with_resets(
            &t.events,
            resets,
            &t.references,
            &[dec!(1), dec!(1)],
            &window(end),
            &config(),
            "d",
        )
        .unwrap()
    };
    let record = fit(&t.resets);
    assert_eq!(record.excluded_events, 0);
    assert_eq!(record.events, (t.events.len() + t.resets.len()) as u64);
    let got = record.parameters().unwrap();
    let resets = got.reset_rates().unwrap();
    assert_eq!(resets[0], 0.0);
    for (n, rate) in resets.iter().enumerate().take(6).skip(1) {
        assert!((rate - 0.3).abs() < 0.03, "n{n} {rate}");
    }
    for (n, (g, w)) in got.reset_law().unwrap().iter().zip(reset_law()).enumerate() {
        assert!((g - w).abs() < 0.02, "law n{n} {g} {w}");
    }
    // A reset that empties a touch can move the reference, which restarts every queue's
    // clock and loses the time since its last event, as any move does; more moves, so a wider
    // band than the reset-free recovery's 5 %.
    for depth in 0..2 {
        for n in 0..=5usize {
            let (a, b) = (got.rates(depth).unwrap()[n], truth.rates(depth).unwrap()[n]);
            for (g, w) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                assert!((g - w).abs() < 0.1 * w.max(0.2), "d{depth} n{n} {g} {w}");
            }
        }
    }
    // Without their waits, the touch's rates come out too high.
    let plain = estimate_parameters(
        &t.events,
        &t.references,
        &[dec!(1), dec!(1)],
        &window(end),
        &config(),
        "d",
    )
    .unwrap();
    let plain = plain.parameters().unwrap();
    assert!(plain.reset_rates().is_none());
    assert!(plain.rates(0).unwrap()[2].0 > got.rates(0).unwrap()[2].0 * 1.02);
    // No resets: zero rates, the invariant law, and the other rates as without resets.
    let none = fit(&[]);
    let none = none.parameters().unwrap();
    assert_eq!(none.reset_rates(), Some(&vec![0.0; CAP + 1][..]));
    assert_eq!(none.reset_law(), none.invariant(0), "no new size seen");
    assert_eq!(none.rates(0), plain.rates(0));
}

fn ask_reset(timestamp_ms: u64, queue_before: u32, waiting_seconds: f64) -> ResetObservation {
    ResetObservation {
        timestamp_ms,
        side: QrSide::Ask,
        queue_before,
        waiting_seconds,
        queue_after: None,
    }
}

#[test]
fn test_resets_at_zero_or_past_the_cap_or_unordered_are_refused() {
    let t = generate(4, 2_000.0);
    let end = t.events.last().unwrap().timestamp_ms + 1;
    let run = |resets: &[ResetObservation]| {
        estimate_parameters_with_resets(
            &t.events,
            resets,
            &t.references,
            &[dec!(1), dec!(1)],
            &window(end),
            &config(),
            "d",
        )
    };
    let base = run(&[]).unwrap().excluded_events;
    let bad = [
        ask_reset(1, 0, 1.0),
        ask_reset(2, CAP as u32 + 1, 1.0),
        ask_reset(3, 2, 0.0),
        ask_reset(4, 2, f64::NAN),
    ];
    assert_eq!(run(&bad).unwrap().excluded_events, base + 4);
    assert!(run(&[ask_reset(5, 2, 1.0), ask_reset(4, 2, 1.0)]).is_err());
}

#[test]
fn test_reset_rate_is_resets_over_time_and_its_law_the_new_sizes() {
    let event = |timestamp_ms, kind, queue_before, waiting_seconds| QueueObservation {
        timestamp_ms,
        side: QrSide::Bid,
        depth: 0,
        kind,
        queue_before,
        waiting_seconds,
    };
    let events = [
        event(0, QrEventKind::Limit, 0, 0.5),
        event(1, QrEventKind::Limit, 1, 0.25),
        event(2, QrEventKind::Cancel, 2, 0.5),
        event(2, QrEventKind::Cancel, 1, 0.0),
    ];
    let resets = [
        ResetObservation {
            queue_after: Some(3),
            ..ask_reset(3, 1, 0.25)
        },
        ask_reset(4, 3, 0.0),
        ResetObservation {
            queue_after: Some(99),
            ..ask_reset(5, 2, 0.0)
        },
    ];
    let reference = ReferenceObservation {
        timestamp_ms: 1,
        moved: true,
        reinitialized: false,
    };
    let cfg = estimation_config(1, 4, 1, 1, UnsupportedPolicy::NearestSupported)
        .with_zero_waits(ZeroWaitPolicy::Count);
    let record = estimate_parameters_with_resets(
        &events,
        &resets,
        &[reference],
        &[dec!(1)],
        &window(100),
        &cfg,
        "d",
    )
    .unwrap();
    let fit = record.parameters().unwrap();
    let rates = fit.reset_rates().unwrap();
    assert_eq!(rates[1], 2.0, "one reset in half a second at size 1");
    assert_eq!(rates[3], 0.0, "no time at size 3");
    assert_eq!(
        fit.rates(0).unwrap()[1].0,
        2.0,
        "the reset's wait is time at its size"
    );
    assert_eq!(
        fit.reset_law().unwrap(),
        &[0.0, 0.0, 0.0, 0.5, 0.5],
        "new sizes, past the cap at the cap, an unknown one left out"
    );
}

#[test]
fn test_constant_baseline_keeps_the_mean_reset_rate_and_the_law() {
    let mut rates = vec![0.0; CAP + 1];
    for (n, r) in rates.iter_mut().enumerate().skip(1) {
        *r = 0.1 * n as f64;
    }
    let p = params().with_resets(rates.clone(), reset_law()).unwrap();
    let law = p.invariant(0).unwrap();
    let mean: f64 = rates.iter().zip(law).map(|(r, q)| r * q).sum();
    let baseline = p.constant_baseline().unwrap();
    let got = baseline.reset_rates().unwrap();
    assert_eq!(got[0], 0.0);
    assert!(got[1..].iter().all(|r| (r - mean).abs() < 1e-12));
    assert_eq!(baseline.reset_law(), Some(&reset_law()[..]));
    assert!(
        params()
            .constant_baseline()
            .unwrap()
            .reset_rates()
            .is_none()
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_resets_export_and_older_exports_read_without_them() {
    let p = with_resets(0.25);
    let text = serde_json::to_string(&p).unwrap();
    let back: QrParameters = serde_json::from_str(&text).unwrap();
    assert_eq!(back, p);
    assert_eq!(back.fingerprint(), p.fingerprint());
    let old = serde_json::to_string(&params()).unwrap();
    assert!(!old.contains("reset"));
    let back: QrParameters = serde_json::from_str(&old).unwrap();
    assert!(back.reset_rates().is_none());
    for (field, what) in [
        ("reset_law", "rates without their law"),
        ("reset_rates", "a law without its rates"),
    ] {
        let mut half: serde_json::Value = serde_json::from_str(&text).unwrap();
        half.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<QrParameters>(half).is_err(),
            "{what}"
        );
    }
}

#[test]
fn test_resets_at_sizes_a_run_never_visits_leave_it_unchanged() {
    // The capture's case: resets are seen at small touch sizes, while a transient fit's run
    // lives near the cap. Here the cap is out of reach and the resets sit only there.
    let mut table = table(1.0);
    table[CAP - 1].0 = 0.0;
    let reachable = QrParameters::new(
        dec!(0.01),
        0.6,
        0.2,
        vec![dec!(1), dec!(1)],
        vec![table.clone(), table],
        vec![vec![true; CAP + 1]; 2],
        UnsupportedPolicy::Stop,
    )
    .unwrap();
    assert_eq!(reachable.invariant(0).unwrap()[CAP], 0.0);
    let mut rates = vec![0.0; CAP + 1];
    rates[CAP] = 5.0;
    let with = reachable.clone().with_resets(rates, reset_law()).unwrap();
    assert_ne!(with.fingerprint(), reachable.fingerprint());
    let run = |p: QrParameters| {
        QrSimulator::new(p, 17, 100)
            .unwrap()
            .run(5_000.0, 1.0)
            .unwrap()
    };
    let (plain, reset) = (run(reachable), run(with));
    assert!(reset.resets.is_empty());
    assert_eq!(plain.events, reset.events);
    assert_eq!(plain.samples, reset.samples);
}
