//! The Hawkes model against hand values and known parameters.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::backtest::hawkes::*;
use market_maker_rs::backtest::queue_reactive::{QrEventKind, RateTable};
use market_maker_rs::types::error::MMError;

const CAP: usize = 15;
const LIMIT: QrEventKind = QrEventKind::Limit;
const CANCEL: QrEventKind = QrEventKind::Cancel;
const MARKET: QrEventKind = QrEventKind::Market;

fn table(scale: f64) -> RateTable {
    (0..=CAP)
        .map(|n| {
            let size = n as f64;
            (
                if n == CAP {
                    0.0
                } else {
                    scale * 1.5 / (1.0 + 0.2 * size)
                },
                if n == 0 {
                    0.0
                } else {
                    scale * (0.1 + 0.08 * size)
                },
                if n == 0 { 0.0 } else { scale * 0.2 },
            )
        })
        .collect()
}

fn weights() -> BranchingMatrix {
    [[0.3, 0.05, 0.15], [0.1, 0.25, 0.05], [0.05, 0.05, 0.2]]
}

fn truth() -> HawkesParameters {
    HawkesParameters::new(table(1.0), vec![2.0], vec![weights()]).unwrap()
}

fn generate(params: &HawkesParameters, seed: u64, count: usize, length: f64) -> Vec<HawkesSegment> {
    (0..count)
        .map(|i| {
            HawkesSimulator::new(params.clone(), seed + i as u64, 3)
                .unwrap()
                .run(length)
                .unwrap()
                .0
        })
        .collect()
}

#[test]
fn test_parameters_refuse_unstable_and_invalid_values() {
    let build =
        |weights: BranchingMatrix| HawkesParameters::new(table(1.0), vec![2.0], vec![weights]);
    assert!(build(weights()).is_ok());
    // Each row sums to 1: spectral radius exactly 1.
    assert!(build([[0.5, 0.25, 0.25]; 3]).is_err());
    assert!(build([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).is_err());
    // A cycle with radius above 1 and small diagonal.
    assert!(build([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]]).is_err());
    assert!(build([[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).is_err());
    assert!(build([[f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).is_err());
    // A cycle with radius below 1 has larger entries than a row bound would allow
    // to hide nothing: it is accepted, with its exact radius.
    let cycle = build([[0.0, 0.9, 0.0], [0.0, 0.0, 0.9], [0.9, 0.0, 0.0]]).unwrap();
    assert!(
        (cycle.branching_ratio() - 0.9).abs() < 1e-9,
        "{}",
        cycle.branching_ratio()
    );
    let good = weights();
    for decays in [
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![1e10],
        vec![1e-10],
    ] {
        assert!(HawkesParameters::new(table(1.0), decays, vec![good]).is_err());
    }
    assert!(HawkesParameters::new(table(1.0), vec![1.0, 2.0], vec![good]).is_err());
    assert!(HawkesParameters::new(table(1.0), vec![1.0; 5], vec![[[0.0; 3]; 3]; 5]).is_err());
    let mut rates = table(1.0);
    rates[0].1 = 1.0;
    assert!(HawkesParameters::new(rates, vec![], vec![]).is_err());
    let mut rates = table(1.0);
    rates[CAP].0 = 1.0;
    assert!(HawkesParameters::new(rates, vec![], vec![]).is_err());
    let mut rates = table(1.0);
    rates[1].2 = -1.0;
    assert!(HawkesParameters::new(rates, vec![], vec![]).is_err());
    assert!(HawkesParameters::new(vec![(0.0, 0.0, 0.0)], vec![], vec![]).is_err());
    // Two kernels add up against the bound.
    let half = [[0.6, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    assert!(HawkesParameters::new(table(1.0), vec![1.0, 2.0], vec![half, half]).is_err());
}

#[test]
fn test_score_flow_matches_a_hand_likelihood() {
    // mu_L = 2, alpha^LL = 0.5, beta = 1, queue 1 with mu_C = mu_M = 0. Events at 1 and 2.
    // lambda(1) = 2. lambda(2) = 2 + 0.5 exp(-1). On [0, 3], the compensator is
    // 6 + 0.5 (1 - exp(-2)) + 0.5 (1 - exp(-1)).
    let mut rates: RateTable = vec![(0.0, 0.0, 0.0); 6];
    rates[1].0 = 2.0;
    rates[2].0 = 2.0;
    rates[3].0 = 2.0;
    rates[2].1 = 1e-300;
    rates[3].1 = 1e-300;
    rates[1].1 = 1e-300;
    let mut alpha = [[0.0; 3]; 3];
    alpha[0][0] = 0.5;
    let p = HawkesParameters::new(rates, vec![1.0], vec![alpha]).unwrap();
    let segment = HawkesSegment {
        initial_queue: 1,
        duration_seconds: 3.0,
        events: vec![
            HawkesEvent {
                time_seconds: 1.0,
                kind: LIMIT,
            },
            HawkesEvent {
                time_seconds: 2.0,
                kind: LIMIT,
            },
        ],
    };
    let fit = score_flow(&[segment], &p).unwrap();
    let e = |x: f64| x.exp();
    let hand = 2.0f64.ln() + (2.0 + 0.5 * e(-1.0)).ln()
        - (6.0 + 0.5 * (1.0 - e(-2.0)) + 0.5 * (1.0 - e(-1.0)));
    assert!(
        (fit.log_likelihood - hand).abs() < 1e-9,
        "{} {hand}",
        fit.log_likelihood
    );
    assert_eq!(fit.events, 2);
    assert_eq!(fit.floored_events, 0);
}

#[test]
fn test_market_and_cancel_have_no_intensity_at_an_empty_queue_even_excited() {
    // A market order excited at empty would break the queue rules.
    let t = HawkesSimulator::new(truth(), 3, 0)
        .unwrap()
        .run(20_000.0)
        .unwrap()
        .0;
    let mut queue = 0i64;
    let mut seen_empty = 0;
    for event in &t.events {
        if queue == 0 {
            seen_empty += 1;
            assert_eq!(event.kind, LIMIT);
        }
        queue += if event.kind == LIMIT { 1 } else { -1 };
        assert!((0..=CAP as i64).contains(&queue));
    }
    assert!(seen_empty > 10);
}

#[test]
fn test_simulator_mean_rate_matches_the_branching_in_a_pure_process() {
    // One type, constant baseline away from the bounds, branching n: the rate is
    // mu / (1 - n). Queue bounds are far with departures balancing arrivals.
    let n = 0.5;
    let rates: RateTable = (0..=2_000)
        .map(|q| {
            (
                if q == 2_000 { 0.0 } else { 1.0 },
                if q == 0 { 0.0 } else { 1.0 },
                0.0,
            )
        })
        .collect();
    let mut alpha = [[0.0; 3]; 3];
    alpha[0][0] = n;
    alpha[1][1] = n;
    let p = HawkesParameters::new(rates, vec![3.0], vec![alpha]).unwrap();
    let t = HawkesSimulator::new(p, 9, 1_000)
        .unwrap()
        .run(100_000.0)
        .unwrap()
        .0;
    let limits = t.events.iter().filter(|e| e.kind == LIMIT).count() as f64;
    let rate = limits / 100_000.0;
    assert!((rate - 2.0).abs() < 0.08, "{rate}");
}

#[test]
fn test_same_seed_reproduces_and_excitation_is_in_the_state() {
    let run = |seed| {
        HawkesSimulator::new(truth(), seed, 3)
            .unwrap()
            .run(500.0)
            .unwrap()
            .0
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));

    let mut whole = HawkesSimulator::new(truth(), 5, 3).unwrap();
    for _ in 0..200 {
        whole.step().unwrap();
    }
    let state = whole.state();
    assert!(state.excitation[0].iter().any(|r| *r > 0.0));
    let mut restored = HawkesSimulator::from_state(truth(), state).unwrap();
    for _ in 0..500 {
        assert_eq!(whole.step().unwrap(), restored.step().unwrap());
    }
    // Without the excitation, the continuation differs.
    let mut wiped = HawkesSimulator::new(truth(), 5, 3).unwrap();
    for _ in 0..200 {
        wiped.step().unwrap();
    }
    let mut blank = wiped.state();
    blank.excitation = [[0.0; 3]; MAX_KERNELS];
    let mut other = HawkesSimulator::from_state(truth(), blank).unwrap();
    let mut same = true;
    for _ in 0..50 {
        same &= wiped.step().unwrap() == other.step().unwrap();
    }
    assert!(!same);
}

#[test]
fn test_excitation_decays_and_jumps_by_hand() {
    // One kernel: after an event at t1 and one at t2, R = exp(-beta (t2 - t1)) + 1 on
    // the type of the first, when both are limits.
    let mut sim = HawkesSimulator::new(truth(), 11, 3).unwrap();
    let mut times = [[f64::NEG_INFINITY; 2]; 3];
    let mut expected = [0.0f64; 3];
    for _ in 0..200 {
        let HawkesStep::Event(event) = sim.step().unwrap() else {
            panic!("stopped");
        };
        let index = [LIMIT, CANCEL, MARKET]
            .iter()
            .position(|k| *k == event.kind)
            .unwrap();
        let factor = (-2.0 * (event.time_seconds - times[0][1].max(0.0))).exp();
        for value in &mut expected {
            *value *= factor;
        }
        expected[index] += 1.0;
        times[0][1] = event.time_seconds;
        let state = sim.state();
        for (got, want) in state.excitation[0].iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "{got} {want}");
        }
    }
}

#[test]
fn test_from_state_refuses_a_state_that_does_not_fit() {
    let good = HawkesSimulator::new(truth(), 5, 3).unwrap().state();
    assert!(HawkesSimulator::from_state(truth(), good).is_ok());
    let edits: [fn(&mut HawkesState); 7] = [
        |s| s.queue = CAP as u32 + 1,
        |s| s.time_seconds = f64::NAN,
        |s| s.clock_seconds = -1.0,
        |s| s.excitation[0][0] = -1.0,
        |s| s.excitation[1][0] = 1.0,
        |s| s.rng_state = 0,
        |s| s.fingerprint_low ^= 1,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut bad = good;
        edit(&mut bad);
        assert!(
            HawkesSimulator::from_state(truth(), bad).is_err(),
            "edit {i}"
        );
    }
    assert!(HawkesSimulator::new(truth(), 5, CAP as u32 + 1).is_err());
}

#[test]
fn test_consecutive_runs_equal_one_long_run() {
    let mut pieces = HawkesSimulator::new(truth(), 5, 3).unwrap();
    let mut kept = Vec::new();
    for i in 0..100 {
        let (segment, stop) = pieces.run(2.0).unwrap();
        assert_eq!(stop, None);
        kept.extend(
            segment
                .events
                .iter()
                .map(|e| (e.time_seconds + f64::from(i) * 2.0, e.kind)),
        );
    }
    let long = HawkesSimulator::new(truth(), 5, 3)
        .unwrap()
        .run(200.0)
        .unwrap()
        .0;
    assert_eq!(kept.len(), long.events.len());
    for ((time, kind), event) in kept.iter().zip(&long.events) {
        assert_eq!(*kind, event.kind);
        assert!((time - event.time_seconds).abs() < 1e-9);
    }
    let mut limited = HawkesSimulator::new(truth(), 5, 3).unwrap();
    let (segment, stop) = limited.run_limited(1_000.0, 5).unwrap();
    assert_eq!(
        (segment.events.len(), stop),
        (5, Some(HawkesStep::EventLimit))
    );
    assert!(limited.run(0.0).is_err());
    assert!(limited.run(f64::NAN).is_err());
}

#[test]
fn test_all_zero_intensity_and_frozen_time_are_stalled() {
    let p = HawkesParameters::new(vec![(0.0, 0.0, 0.0); 3], vec![], vec![]).unwrap();
    let mut sim = HawkesSimulator::new(p, 5, 1).unwrap();
    assert_eq!(sim.step().unwrap(), HawkesStep::Stalled);
    let start = HawkesSimulator::new(truth(), 5, 3).unwrap().state();
    let frozen = HawkesState {
        time_seconds: 1e18,
        clock_seconds: 1e18,
        ..start
    };
    let mut sim = HawkesSimulator::from_state(truth(), frozen).unwrap();
    assert_eq!(sim.step().unwrap(), HawkesStep::Stalled);
    assert_eq!(sim.state(), frozen);
}

#[test]
fn test_fit_recovers_known_parameters_and_likelihood_rises() {
    let data = generate(&truth(), 100, 15, 2_000.0);
    let record = fit_hawkes(&data, &HawkesFitConfig::new(CAP as u32, vec![2.0])).unwrap();
    let fit = record.parameters().expect("calibrated");
    let (got, want) = (fit.branching_matrix(), weights());
    for l in 0..3 {
        for m in 0..3 {
            assert!(
                (got[l][m] - want[l][m]).abs() < 0.03,
                "{l}{m} {} {}",
                got[l][m],
                want[l][m]
            );
        }
    }
    let rates = fit.rates();
    let truth_rates = table(1.0);
    for n in 1..=6 {
        assert!(
            (rates[n].0 - truth_rates[n].0).abs() < 0.1,
            "{n} {:?}",
            rates[n]
        );
        assert!(
            (rates[n].1 - truth_rates[n].1).abs() < 0.1,
            "{n} {:?}",
            rates[n]
        );
    }
    for pair in record.likelihood_trace.windows(2) {
        assert!(pair[1] >= pair[0] - 1e-6, "{pair:?}");
    }
    assert!(record.events > 5_000);
    assert_eq!(record.seconds, 30_000.0);
}

#[test]
fn test_fit_on_queue_reactive_data_finds_little_excitation() {
    let qr = HawkesParameters::new(table(1.0), vec![], vec![]).unwrap();
    let data = generate(&qr, 300, 20, 2_000.0);
    let record = fit_hawkes(&data, &HawkesFitConfig::new(CAP as u32, vec![2.0])).unwrap();
    let ratio = record.parameters().unwrap().branching_ratio();
    assert!(ratio < 0.05, "{ratio}");
}

#[test]
fn test_fit_reports_each_failure_and_refuses_invalid_input() {
    let cfg = HawkesFitConfig::new(5, vec![1.0]);
    let empty = fit_hawkes(&[], &cfg).unwrap();
    assert_eq!(
        empty.fit,
        HawkesFit::Insufficient(HawkesFailure::NoSupportedSize)
    );
    assert!(fit_hawkes(&[], &HawkesFitConfig::new(0, vec![])).is_err());
    assert!(fit_hawkes(&[], &HawkesFitConfig::new(5, vec![]).with_iterations(0)).is_err());
    assert!(fit_hawkes(&[], &HawkesFitConfig::new(5, vec![]).with_min_events(0)).is_err());
    let seg = |initial, events: Vec<(f64, QrEventKind)>| HawkesSegment {
        initial_queue: initial,
        duration_seconds: 10.0,
        events: events
            .into_iter()
            .map(|(time_seconds, kind)| HawkesEvent { time_seconds, kind })
            .collect(),
    };
    for bad in [
        seg(0, vec![(1.0, MARKET)]),
        seg(0, vec![(1.0, CANCEL)]),
        seg(6, vec![]),
        seg(5, vec![(1.0, LIMIT)]),
        seg(1, vec![(2.0, LIMIT), (1.0, LIMIT)]),
        seg(1, vec![(11.0, LIMIT)]),
        seg(1, vec![(f64::NAN, LIMIT)]),
        HawkesSegment {
            initial_queue: 1,
            duration_seconds: 0.0,
            events: vec![],
        },
    ] {
        assert!(
            fit_hawkes(std::slice::from_ref(&bad), &cfg).is_err(),
            "{bad:?}"
        );
    }
}

#[test]
fn test_fit_borrows_the_nearest_baseline_for_thin_sizes_and_counts_them() {
    let data = generate(&truth(), 100, 3, 500.0);
    let cfg = HawkesFitConfig::new(CAP as u32, vec![2.0])
        .with_iterations(5)
        .with_min_events(1_000_000);
    assert_eq!(
        fit_hawkes(&data, &cfg).unwrap().fit,
        HawkesFit::Insufficient(HawkesFailure::NoSupportedSize)
    );
    let cfg = HawkesFitConfig::new(CAP as u32, vec![2.0])
        .with_iterations(5)
        .with_min_events(200);
    let record = fit_hawkes(&data, &cfg).unwrap();
    let fit = record.parameters().unwrap();
    assert!(record.borrowed_sizes > 0);
    let rates = fit.rates();
    assert_eq!(rates[CAP].1, rates[CAP - 1].1, "both take the same row");
    assert_eq!(rates[CAP].0, 0.0);
    assert_eq!(rates[0].1, 0.0);
}

#[test]
fn test_fit_changing_data_changes_the_hash() {
    let data = generate(&truth(), 100, 2, 200.0);
    let cfg = HawkesFitConfig::new(CAP as u32, vec![2.0])
        .with_iterations(2)
        .with_min_events(1);
    let base = fit_hawkes(&data, &cfg).unwrap().dataset_hash;
    let edits: [fn(&mut HawkesSegment); 2] = [
        |s| s.duration_seconds += 1.0,
        |s| s.events[0].time_seconds *= 0.5,
    ];
    for edit in edits {
        let mut other = data.clone();
        edit(&mut other[0]);
        assert_ne!(fit_hawkes(&other, &cfg).unwrap().dataset_hash, base);
    }
    let mut other = data;
    other[0].initial_queue += 1;
    // The rules may break with a new start, so the hash is checked on a valid one.
    if let Ok(record) = fit_hawkes(&other, &cfg) {
        assert_ne!(record.dataset_hash, base);
    }
}

#[test]
fn test_fit_poisson_is_the_constant_rates() {
    let segment = HawkesSegment {
        initial_queue: 1,
        duration_seconds: 10.0,
        events: vec![
            HawkesEvent {
                time_seconds: 2.0,
                kind: MARKET,
            },
            HawkesEvent {
                time_seconds: 4.0,
                kind: LIMIT,
            },
        ],
    };
    // Queue 1 on [0, 2), 0 on [2, 4), 1 on [4, 10]: departures exposed 8 s, arrivals 10 s.
    let p = fit_poisson(&[segment], 3).unwrap();
    assert_eq!(p.rates()[1], (0.1, 0.0, 0.125));
    assert_eq!(p.rates()[0], (0.1, 0.0, 0.0));
    assert_eq!(p.rates()[3].0, 0.0);
    assert!(p.decays().is_empty());
    assert!(fit_poisson(&[], 3).is_err());
    assert!(fit_poisson(&[], 0).is_err());
}

#[test]
fn test_residuals_are_close_to_exponential_only_under_the_true_model() {
    let data = generate(&truth(), 500, 10, 5_000.0);
    let right = score_flow(&data, &truth()).unwrap();
    for ks in right.residual_ks {
        assert!(ks < 0.015, "{ks}");
    }
    let qr = HawkesParameters::new(table(1.0), vec![], vec![]).unwrap();
    let wrong = score_flow(&data, &qr).unwrap();
    assert!(
        wrong.residual_ks[0] > 3.0 * right.residual_ks[0],
        "{wrong:?}"
    );
    assert!(right.log_likelihood > wrong.log_likelihood);
    assert!(right.aic < wrong.aic);
    assert_eq!(
        right.parameter_count,
        15 + 15 + 1 + 9,
        "distinct limit, cancel and market baselines, and weights"
    );
    let constant = fit_poisson(&data, CAP as u32).unwrap();
    assert_eq!(score_flow(&data, &constant).unwrap().parameter_count, 3);
}

#[test]
fn test_score_flow_floors_an_impossible_event_and_refuses_no_events() {
    let p = HawkesParameters::new(
        vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        vec![],
        vec![],
    )
    .unwrap();
    let segment = HawkesSegment {
        initial_queue: 1,
        duration_seconds: 2.0,
        events: vec![HawkesEvent {
            time_seconds: 1.0,
            kind: MARKET,
        }],
    };
    let fit = score_flow(std::slice::from_ref(&segment), &p).unwrap();
    assert_eq!(fit.floored_events, 1);
    assert!((fit.log_likelihood - 1e-12f64.ln()).abs() < 1e-9);
    let none = HawkesSegment {
        initial_queue: 1,
        duration_seconds: 2.0,
        events: vec![],
    };
    assert!(score_flow(&[none], &p).is_err());
}

#[test]
fn test_select_decay_finds_the_true_scale_on_validation_and_reports_all() {
    let train = generate(&truth(), 600, 5, 2_000.0);
    let validation = generate(&truth(), 700, 3, 2_000.0);
    let cfg = HawkesFitConfig::new(CAP as u32, vec![]).with_iterations(60);
    let candidates = [0.1, 0.5, 2.0, 10.0, 50.0];
    let (scores, best) = select_decay(&train, &validation, &candidates, &cfg).unwrap();
    assert_eq!(scores.len(), 5);
    assert_eq!(best, Some(2), "{scores:?}");
    assert!(select_decay(&train, &validation, &[], &cfg).is_err());
}

#[test]
fn test_events_per_interval_counts_whole_intervals() {
    let segment = HawkesSegment {
        initial_queue: 1,
        duration_seconds: 3.5,
        events: vec![
            HawkesEvent {
                time_seconds: 0.5,
                kind: LIMIT,
            },
            HawkesEvent {
                time_seconds: 0.6,
                kind: LIMIT,
            },
            HawkesEvent {
                time_seconds: 2.5,
                kind: LIMIT,
            },
            HawkesEvent {
                time_seconds: 3.2,
                kind: LIMIT,
            },
            HawkesEvent {
                time_seconds: 2.6,
                kind: MARKET,
            },
        ],
    };
    let mut ordered = segment;
    ordered
        .events
        .sort_by(|a, b| a.time_seconds.total_cmp(&b.time_seconds));
    assert_eq!(
        events_per_interval(&[ordered.clone()], LIMIT, 1.0).unwrap(),
        [2, 0, 1]
    );
    assert_eq!(
        events_per_interval(&[ordered.clone()], MARKET, 1.0).unwrap(),
        [0, 0, 1]
    );
    assert!(events_per_interval(&[ordered], LIMIT, 0.0).is_err());
}

#[test]
fn test_fill_rate_by_hand() {
    let segment = |events: Vec<(f64, QrEventKind)>| HawkesSegment {
        initial_queue: 2,
        duration_seconds: 10.0,
        events: events
            .into_iter()
            .map(|(time_seconds, kind)| HawkesEvent { time_seconds, kind })
            .collect(),
    };
    // Two ahead at the only probe, at 0: two markets clear them, and a third fills us.
    let fills = segment(vec![
        (1.0, MARKET),
        (2.0, MARKET),
        (3.0, LIMIT),
        (4.0, MARKET),
    ]);
    assert_eq!(
        fill_rate(std::slice::from_ref(&fills), CAP as u32, 10.0, 100.0).unwrap(),
        Some(1.0)
    );
    // A horizon that ends before the third.
    assert_eq!(
        fill_rate(std::slice::from_ref(&fills), CAP as u32, 3.5, 100.0).unwrap(),
        Some(0.0)
    );
    // A limit behind us does not delay us; a cancel removes half of the two ahead.
    let cancels = segment(vec![
        (1.0, CANCEL),
        (2.0, MARKET),
        (2.5, LIMIT),
        (2.6, LIMIT),
        (3.0, MARKET),
    ]);
    assert_eq!(
        fill_rate(&[cancels], CAP as u32, 10.0, 100.0).unwrap(),
        Some(1.0)
    );
    // No probe fits.
    assert_eq!(
        fill_rate(std::slice::from_ref(&fills), CAP as u32, 11.0, 1.0).unwrap(),
        None
    );
    assert!(fill_rate(&[], CAP as u32, 0.0, 1.0).is_err());
    assert!(fill_rate(&[], CAP as u32, 1.0, 0.0).is_err());
    // Four probes, 3 s apart, on a horizon of 1 s: only those with a market in reach.
    let spaced = segment(vec![
        (0.5, LIMIT),
        (3.2, LIMIT),
        (3.5, MARKET),
        (3.6, MARKET),
        (3.7, MARKET),
        (3.8, MARKET),
    ]);
    let rate = fill_rate(&[spaced], CAP as u32, 1.0, 3.0).unwrap().unwrap();
    assert_eq!(rate, 0.25);
}

#[test]
fn test_comparison_of_three_models_with_flow_and_fills_apart() {
    compare_models(6, 3, &[0.5, 2.0, 8.0]);
}

/// The table in the pull request, on the full data.
#[test]
#[ignore = "regenerates the evidence table; run with --ignored --nocapture"]
fn test_comparison_evidence_table() {
    compare_models(20, 10, &[0.5, 1.0, 2.0, 4.0, 8.0]);
}

fn compare_models(train_days: usize, other_days: usize, candidates: &[f64]) {
    // Chronological train, validation, holdout, each independent synthetic days of the
    // same process. Decay chosen on validation only, then one look at the holdout.
    let train = generate(&truth(), 1_000, train_days, 2_000.0);
    let validation = generate(&truth(), 2_000, other_days, 2_000.0);
    let holdout = generate(&truth(), 3_000, other_days, 2_000.0);
    let cap = CAP as u32;
    let cfg = HawkesFitConfig::new(cap, vec![]).with_iterations(80);
    let (scores, best) = select_decay(&train, &validation, candidates, &cfg).unwrap();
    println!("| decay | train ll/event | validation ll/event |");
    println!("| --- | --- | --- |");
    for s in &scores {
        println!(
            "| {} | {:.5} | {:.5} |",
            s.decay,
            s.training,
            s.validation.unwrap()
        );
    }
    let decay = candidates[best.unwrap()];
    let hawkes = fit_hawkes(&train, &HawkesFitConfig::new(cap, vec![decay])).unwrap();
    let hawkes = hawkes.parameters().unwrap().clone();
    let qr = fit_hawkes(&train, &HawkesFitConfig::new(cap, vec![])).unwrap();
    let qr = qr.parameters().unwrap().clone();
    let poisson = fit_poisson(&train, cap).unwrap();
    let mut flow = Vec::new();
    println!(
        "| model | params | ll/event | aic | ks limit | ks cancel | ks market | market flow tv | fill rate |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let data_fill = fill_rate(&holdout, CAP as u32, 3.0, 10.0).unwrap().unwrap();
    let counts = |segments: &[HawkesSegment]| events_per_interval(segments, MARKET, 10.0).unwrap();
    let data_counts = counts(&holdout);
    println!("| data | | | | | | | | {data_fill:.4} |");
    for (name, params, seed) in [
        ("poisson", &poisson, 4_000),
        ("qr", &qr, 5_000),
        ("hawkes", &hawkes, 6_000),
    ] {
        let fit = score_flow(&holdout, params).unwrap();
        let simulated = generate(params, seed, other_days, 2_000.0);
        let tv = total_variation(&data_counts, &counts(&simulated));
        let fills = fill_rate(&simulated, CAP as u32, 3.0, 10.0)
            .unwrap()
            .unwrap();
        println!(
            "| {name} | {} | {:.5} | {:.0} | {:.4} | {:.4} | {:.4} | {tv:.4} | {fills:.4} |",
            fit.parameter_count,
            fit.log_likelihood_per_event,
            fit.aic,
            fit.residual_ks[0],
            fit.residual_ks[1],
            fit.residual_ks[2],
        );
        flow.push((fit, tv));
    }
    assert!(flow[2].0.log_likelihood_per_event > flow[1].0.log_likelihood_per_event);
    assert!(flow[1].0.log_likelihood_per_event > flow[0].0.log_likelihood_per_event);
    assert!(flow[2].0.residual_ks[0] < flow[1].0.residual_ks[0]);
    assert!(flow[2].1 < flow[0].1, "market flow counts per interval");
}

fn total_variation(a: &[u32], b: &[u32]) -> f64 {
    let largest = a.iter().chain(b).copied().max().unwrap_or(0) as usize;
    let mut counts = vec![[0f64; 2]; largest + 1];
    for v in a {
        counts[*v as usize][0] += 1.0 / a.len() as f64;
    }
    for v in b {
        counts[*v as usize][1] += 1.0 / b.len() as f64;
    }
    counts.iter().map(|c| (c[0] - c[1]).abs()).sum::<f64>() / 2.0
}

#[test]
#[ignore = "measures update cost; run with --ignored --nocapture"]
fn test_update_cost_per_event() {
    for (kernels, name) in [(0usize, "qr"), (1, "one kernel"), (4, "four kernels")] {
        let share = 0.1 / kernels.max(1) as f64;
        let p = HawkesParameters::new(
            table(1.0),
            vec![2.0; kernels],
            vec![[[share; 3]; 3]; kernels],
        )
        .unwrap();
        let mut sim = HawkesSimulator::new(p, 5, 3).unwrap();
        let start = std::time::Instant::now();
        let events = 1_000_000;
        for _ in 0..events {
            sim.step().unwrap();
        }
        println!(
            "{name}: {:.0} ns per event",
            start.elapsed().as_nanos() as f64 / f64::from(events)
        );
    }
}

#[cfg(feature = "serde")]
mod decoding {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn test_export_is_exact_and_invalid_input_is_refused() {
        let text = serde_json::to_string(&truth()).unwrap();
        let back: HawkesParameters = serde_json::from_str(&text).unwrap();
        assert_eq!(back, truth());
        assert_eq!(back.fingerprint(), truth().fingerprint());
        let base: Value = serde_json::from_str(&text).unwrap();
        let edits: [fn(&mut Value); 4] = [
            |v| v["weights"][0][0] = json!(["2.0", "2.0", "2.0"]),
            |v| v["decays"][0] = json!("-1.0"),
            |v| v["rates"][0] = json!(["1.0", "1.0", "0.0"]),
            |v| v["extra"] = json!(1),
        ];
        for (i, edit) in edits.iter().enumerate() {
            let mut bad = base.clone();
            edit(&mut bad);
            assert!(
                serde_json::from_value::<HawkesParameters>(bad).is_err(),
                "edit {i}"
            );
        }
    }
}

#[test]
fn test_fit_refuses_invalid_decays_as_an_error() {
    let data = generate(&truth(), 100, 1, 200.0);
    for decays in [vec![0.0], vec![f64::NAN], vec![1.0; MAX_KERNELS + 1]] {
        let config = HawkesFitConfig::new(CAP as u32, decays);
        assert!(matches!(
            fit_hawkes(&data, &config),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn test_trace_ends_at_the_likelihood_of_the_fitted_parameters() {
    let data = generate(&truth(), 100, 2, 500.0);
    let config = HawkesFitConfig::new(CAP as u32, vec![2.0]).with_iterations(3);
    let record = fit_hawkes(&data, &config).unwrap();
    assert_eq!(record.likelihood_trace.len(), 4);
    let fit = score_flow(&data, record.parameters().unwrap()).unwrap();
    let last = *record.likelihood_trace.last().unwrap();
    assert!((last - fit.log_likelihood).abs() < 1e-6, "{last} {fit:?}");
}

#[test]
fn test_measures_refuse_an_interval_too_small_for_the_segment() {
    let data = generate(&truth(), 100, 1, 200.0);
    assert!(matches!(
        events_per_interval(&data, MARKET, 1e-9),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        fill_rate(&data, CAP as u32, 1.0, 1e-9),
        Err(MMError::InvalidConfiguration(_))
    ));
}

#[cfg(feature = "serde")]
#[test]
fn test_a_state_stored_as_json_goes_on_with_the_same_stream() {
    let mut sim = HawkesSimulator::new(truth(), 9, 3).unwrap();
    sim.run(50.0).unwrap();
    let state = sim.state();
    let restored: HawkesState =
        serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert_eq!(restored, state);
    let mut other = HawkesSimulator::from_state(truth(), restored).unwrap();
    assert_eq!(sim.run(50.0).unwrap().0, other.run(50.0).unwrap().0);
}

#[test]
fn test_a_stable_nearly_critical_reducible_matrix_is_measured_exactly() {
    let jordan = [[0.9998, 0.1, 0.0], [0.0, 0.9998, 0.0], [0.0, 0.0, 0.0]];
    let params = HawkesParameters::new(table(1.0), vec![2.0], vec![jordan]).unwrap();
    assert!((params.branching_ratio() - 0.9998).abs() < 1e-9);
    let critical = [[1.0, 0.1, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]];
    assert!(HawkesParameters::new(table(1.0), vec![2.0], vec![critical]).is_err());
    let zero = [[0.0; 3]; 3];
    let params = HawkesParameters::new(table(1.0), vec![2.0], vec![zero]).unwrap();
    assert_eq!(params.branching_ratio(), 0.0);
    // A cycle of three, whose eigenvalues all have the same modulus: 0.8.
    let cycle = [[0.0, 0.8, 0.0], [0.0, 0.0, 0.8], [0.8, 0.0, 0.0]];
    let params = HawkesParameters::new(table(1.0), vec![2.0], vec![cycle]).unwrap();
    assert!((params.branching_ratio() - 0.8).abs() < 1e-9);
}

#[test]
fn test_a_segment_stopped_at_its_event_limit_claims_no_unsimulated_time() {
    let mut sim = HawkesSimulator::new(truth(), 5, 3).unwrap();
    let (segment, stop) = sim.run_limited(1_000.0, 1).unwrap();
    assert_eq!(stop, Some(HawkesStep::EventLimit));
    let last = segment.events.last().unwrap().time_seconds;
    assert!(segment.duration_seconds < 1_000.0);
    assert!(segment.duration_seconds >= last);
    // No events at all, and nothing simulated.
    let mut none = HawkesSimulator::new(truth(), 5, 3).unwrap();
    let (empty, _) = none.run_limited(1_000.0, 0).unwrap();
    assert_eq!(empty.duration_seconds, 0.0);
    // The next run goes on from there, and the two lengths add up to the same clock.
    let before = sim.state().clock_seconds;
    let (next, _) = sim.run(10.0).unwrap();
    assert_eq!(before, segment.duration_seconds);
    assert_eq!(sim.state().clock_seconds, before + next.duration_seconds);
}

#[test]
fn test_a_thin_size_borrows_each_baseline_from_where_that_type_is_allowed() {
    // Sizes 0 and 2 have events and size 1 has none, with one event per short segment.
    let segment = |initial_queue, kind| HawkesSegment {
        initial_queue,
        duration_seconds: 1.0,
        events: vec![HawkesEvent {
            time_seconds: 0.5,
            kind,
        }],
    };
    let mut data = vec![segment(0, LIMIT); 40];
    data.extend(vec![segment(2, CANCEL); 40]);
    data.extend(vec![segment(2, MARKET); 40]);
    let config = HawkesFitConfig::new(2, vec![]).with_min_events(5);
    let record = fit_hawkes(&data, &config).unwrap();
    assert_eq!(record.borrowed_sizes, 1);
    let rates = record.parameters().unwrap().rates();
    assert!(rates[1].0 > 0.0, "limit from size 0: {:?}", rates[1]);
    assert!(rates[1].1 > 0.0, "cancel from size 2: {:?}", rates[1]);
    assert!(rates[1].2 > 0.0, "market from size 2: {:?}", rates[1]);
    assert_eq!(rates[0].1, 0.0);
    assert_eq!(rates[2].0, 0.0);
}

#[test]
fn test_fill_rate_refuses_what_score_flow_refuses() {
    let broken = HawkesSegment {
        initial_queue: 0,
        duration_seconds: 10.0,
        events: vec![HawkesEvent {
            time_seconds: 1.0,
            kind: MARKET,
        }],
    };
    assert!(matches!(
        fill_rate(&[broken], CAP as u32, 1.0, 1.0),
        Err(MMError::InvalidMarketState(_))
    ));
    assert!(fill_rate(&[], 0, 1.0, 1.0).is_err());
}

fn busy_profile() -> SeasonalProfile {
    SeasonalProfile::new(100.0, vec![3.0, 1.0, 0.5, 1.5]).unwrap()
}

#[test]
fn test_profile_is_normalized_and_refuses_bad_input() {
    let p = SeasonalProfile::new(10.0, vec![2.0, 6.0]).unwrap();
    assert_eq!(p.multipliers(), [0.5, 1.5]);
    assert_eq!(p.bucket_seconds(), 10.0);
    for (bucket, m) in [
        (0.0, vec![1.0]),
        (f64::NAN, vec![1.0]),
        (1.0, vec![]),
        (1.0, vec![1.0, 0.0]),
        (1.0, vec![1.0, -1.0]),
        (1.0, vec![f64::INFINITY]),
    ] {
        assert!(SeasonalProfile::new(bucket, m).is_err());
    }
}

#[test]
fn test_profile_time_change_by_hand_and_inverse() {
    let p = SeasonalProfile::new(10.0, vec![0.5, 1.5]).unwrap();
    assert_eq!(p.to_model_time(0.0), 0.0);
    assert_eq!(p.to_model_time(4.0), 2.0);
    assert_eq!(p.to_model_time(10.0), 5.0);
    assert_eq!(p.to_model_time(14.0), 11.0);
    assert_eq!(p.to_model_time(20.0), 20.0);
    assert_eq!(p.to_model_time(24.0), 22.0);
    assert_eq!(p.multiplier(14.0), 1.5);
    assert_eq!(p.multiplier(24.0), 0.5);
    for t in [0.0, 3.0, 10.0, 17.5, 20.0, 93.25] {
        assert!(
            (p.to_clock_time(p.to_model_time(t)) - t).abs() < 1e-9,
            "{t}"
        );
    }
}

/// Days simulated in model time and mapped to clock time.
fn seasonal_days(profile: &SeasonalProfile, seed: u64, days: u64) -> Vec<HawkesSegment> {
    let horizon = profile.bucket_seconds() * profile.multipliers().len() as f64;
    (0..days)
        .map(|day| {
            let mut sim = HawkesSimulator::new(truth(), seed + day, 5).unwrap();
            let (segment, _) = sim.run(horizon).unwrap();
            profile.segment_to_clock_time(&segment)
        })
        .collect()
}

#[test]
fn test_fit_seasonality_recovers_a_known_profile_from_training_only() {
    let profile_truth = busy_profile();
    let train = seasonal_days(&profile_truth, 100, 40);
    let fit = fit_seasonality(&train, CAP as u32, 100.0, 4, 30).unwrap();
    for (found, known) in fit.multipliers().iter().zip(profile_truth.multipliers()) {
        assert!((found / known - 1.0).abs() < 0.1, "{found} {known}");
    }
    // It reads the segments it is given: more of them move it.
    let more = [train.clone(), seasonal_days(&profile_truth, 500, 5)].concat();
    assert_ne!(
        fit,
        fit_seasonality(&more, CAP as u32, 100.0, 4, 30).unwrap()
    );
}

#[test]
fn test_fit_seasonality_refuses_a_thin_or_unseen_bucket() {
    let profile_truth = busy_profile();
    let train = seasonal_days(&profile_truth, 100, 2);
    assert!(matches!(
        fit_seasonality(&train, CAP as u32, 100.0, 4, 1_000_000),
        Err(MMError::InvalidMarketState(_))
    ));
    assert!(matches!(
        fit_seasonality(&train, CAP as u32, 400.0, 4, 1),
        Err(MMError::InvalidMarketState(_))
    ));
    assert!(fit_seasonality(&train, CAP as u32, 100.0, 0, 1).is_err());
    assert!(fit_seasonality(&train, 0, 100.0, 4, 1).is_err());
}

#[test]
fn test_seasonal_score_beats_the_flat_one_out_of_sample_on_seasonal_flow() {
    let profile_truth = busy_profile();
    let train = seasonal_days(&profile_truth, 100, 40);
    let holdout = seasonal_days(&profile_truth, 900, 20);
    let config = HawkesFitConfig::new(CAP as u32, vec![2.0]).with_iterations(40);
    let flat = fit_hawkes(&train, &config).unwrap();
    let flat = score_flow(&holdout, flat.parameters().unwrap()).unwrap();
    let profile = fit_seasonality(&train, CAP as u32, 100.0, 4, 30).unwrap();
    let mapped: Vec<_> = train
        .iter()
        .map(|s| profile.segment_to_model_time(s))
        .collect();
    let seasonal = fit_hawkes(&mapped, &config).unwrap();
    let seasonal = score_flow_seasonal(&holdout, seasonal.parameters().unwrap(), &profile).unwrap();
    assert!(seasonal.log_likelihood > flat.log_likelihood);
    assert!(seasonal.aic < flat.aic);
}

#[test]
fn test_a_flat_profile_changes_no_score() {
    let days = seasonal_days(&SeasonalProfile::new(100.0, vec![1.0; 4]).unwrap(), 5, 5);
    let flat = SeasonalProfile::new(100.0, vec![1.0; 4]).unwrap();
    let params = truth();
    let plain = score_flow(&days, &params).unwrap();
    let seasonal = score_flow_seasonal(&days, &params, &flat).unwrap();
    assert!((plain.log_likelihood - seasonal.log_likelihood).abs() < 1e-6);
    assert_eq!(plain.residual_ks, seasonal.residual_ks);
}

#[test]
fn test_seasonal_score_is_the_model_time_score_plus_the_change_of_time() {
    let profile = busy_profile();
    let days = seasonal_days(&profile, 7, 5);
    let mapped: Vec<_> = days
        .iter()
        .map(|s| profile.segment_to_model_time(s))
        .collect();
    let params = truth();
    let plain = score_flow(&mapped, &params).unwrap();
    let seasonal = score_flow_seasonal(&days, &params, &profile).unwrap();
    let jacobian: f64 = days
        .iter()
        .flat_map(|d| &d.events)
        .map(|e| profile.multiplier(e.time_seconds).ln())
        .sum();
    assert!(jacobian != 0.0);
    assert!((seasonal.log_likelihood - plain.log_likelihood - jacobian).abs() < 1e-9);
    assert_eq!(seasonal.residual_ks, plain.residual_ks);
    assert_eq!(seasonal.parameter_count, plain.parameter_count + 3);
    assert_eq!(
        seasonal.aic,
        2.0 * seasonal.parameter_count as f64 - 2.0 * seasonal.log_likelihood
    );
    assert_eq!(
        seasonal.log_likelihood_per_event,
        seasonal.log_likelihood / seasonal.events as f64
    );
}

#[test]
fn test_profile_inverse_at_edges_and_several_periods_out() {
    let p = busy_profile();
    let period = 400.0;
    for clock in [0.0, 100.0, 200.0, 300.0, 400.0, 3.0 * period + 250.0] {
        assert!((p.to_clock_time(p.to_model_time(clock)) - clock).abs() < 1e-9);
    }
    assert_eq!(p.to_clock_time(2.0 * period), 2.0 * period);
    for tau in [5.0, 240.0, 299.0, 1_000.5] {
        assert!((p.to_model_time(p.to_clock_time(tau)) - tau).abs() < 1e-9);
    }
}

#[test]
fn test_fit_seasonality_with_a_fractional_bucket_and_an_event_at_the_end() {
    let event = |t: f64, kind| HawkesEvent {
        time_seconds: t,
        kind,
    };
    // 3 * 0.3 rounds below 0.9, and 0.9 / 0.3 below 3.
    let segment = HawkesSegment {
        initial_queue: 5,
        duration_seconds: 1.2,
        events: vec![
            event(0.1, LIMIT),
            event(0.4, LIMIT),
            event(0.5, LIMIT),
            event(0.7, LIMIT),
            event(1.0, LIMIT),
            event(1.2, CANCEL),
        ],
    };
    let fit = fit_seasonality(&[segment], CAP as u32, 0.3, 4, 1).unwrap();
    // The event at the end counts in no bucket: rates 1, 2, 1, 1 over their mean 1.25.
    for (found, known) in fit.multipliers().iter().zip([0.8, 1.6, 0.8, 0.8]) {
        assert!((found - known).abs() < 1e-9, "{:?}", fit.multipliers());
    }
}

#[test]
fn test_fit_seasonality_divides_by_the_time_each_bucket_was_seen() {
    // Two buckets of 10 s, seen for 10 s and 5 s, with a rate of 1 per second each.
    let events = (0..15)
        .map(|i| HawkesEvent {
            time_seconds: f64::from(i) + 0.5,
            kind: LIMIT,
        })
        .collect();
    let segment = HawkesSegment {
        initial_queue: 0,
        duration_seconds: 15.0,
        events,
    };
    let fit = fit_seasonality(&[segment], 100, 10.0, 2, 1).unwrap();
    assert_eq!(fit.multipliers(), [1.0, 1.0]);
}

#[test]
fn test_profile_refuses_a_range_that_normalizes_to_nothing() {
    assert!(SeasonalProfile::new(1.0, vec![1e-200, 1e200]).is_err());
    assert!(SeasonalProfile::new(1.0, vec![f64::MAX, f64::MAX]).is_err());
}

#[test]
fn test_fit_seasonality_bounds_the_work_of_a_long_segment() {
    let segment = |duration_seconds| HawkesSegment {
        initial_queue: 0,
        duration_seconds,
        events: vec![],
    };
    for (duration, bucket) in [(1e100, 1.0), (86_400.0, 1e-9)] {
        assert!(matches!(
            fit_seasonality(&[segment(duration)], 1, bucket, 1, 1),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    // At the limit it runs, and finds no events.
    assert!(matches!(
        fit_seasonality(&[segment(1_000.0)], 1, 1.0, 1, 1),
        Err(MMError::InvalidMarketState(_))
    ));
}

#[test]
fn test_profile_keeps_each_bucket_a_span_in_model_time() {
    assert!(SeasonalProfile::new(1e-9, vec![1e-320, 1.0]).is_err());
    assert!(SeasonalProfile::new(1.0, vec![1e-12, 1.0]).is_err());
    // Inside the bound, the smallest bucket still inverts.
    let p = SeasonalProfile::new(1e-9, vec![1e-8, 1.0]).unwrap();
    assert_eq!(p.to_clock_time(0.0), 0.0);
    let inside = 0.5e-9;
    let back = p.to_clock_time(p.to_model_time(inside));
    assert!((back - inside).abs() < 1e-15, "{back}");
}

#[test]
fn test_fit_seasonality_refuses_too_many_buckets_before_allocating() {
    for buckets in [MAX_SEASONAL_BUCKETS + 1, usize::MAX] {
        assert!(matches!(
            fit_seasonality(&[], 1, 1.0, buckets, 1),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    // At the bound it gets as far as the data, which has seen no bucket.
    assert!(matches!(
        fit_seasonality(&[], 1, 1.0, MAX_SEASONAL_BUCKETS, 1),
        Err(MMError::InvalidMarketState(_))
    ));
    assert!(matches!(
        SeasonalProfile::new(1.0, vec![1.0; MAX_SEASONAL_BUCKETS + 1]),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(SeasonalProfile::new(1.0, vec![1.0; MAX_SEASONAL_BUCKETS]).is_ok());
}

#[test]
fn test_fit_seasonality_refuses_a_segment_out_of_order() {
    let at = |time_seconds| HawkesEvent {
        time_seconds,
        kind: QrEventKind::Limit,
    };
    let segment = HawkesSegment {
        initial_queue: 0,
        duration_seconds: 10.0,
        events: vec![at(5.0), at(4.0)],
    };
    assert!(matches!(
        fit_seasonality(&[segment], 8, 10.0, 1, 1),
        Err(MMError::InvalidMarketState(_))
    ));
}

#[test]
fn test_profile_bucket_and_multipliers_have_a_range() {
    let one = || vec![1.0];
    assert!(SeasonalProfile::new(MAX_RATE, one()).is_ok());
    assert!(SeasonalProfile::new(1.0 / MAX_RATE, one()).is_ok());
    assert!(SeasonalProfile::new(MAX_RATE * 2.0, one()).is_err());
    assert!(SeasonalProfile::new(0.5 / MAX_RATE, one()).is_err());
    // Positive, and still out of range after the normalization.
    assert!(SeasonalProfile::new(1.0, vec![1.0, 1e-12]).is_err());
}

#[test]
fn test_segment_ending_mid_period_maps_its_duration_both_ways() {
    let p = SeasonalProfile::new(10.0, vec![0.5, 1.5]).unwrap();
    let at = |time_seconds| HawkesEvent {
        time_seconds,
        kind: QrEventKind::Limit,
    };
    let clock = HawkesSegment {
        initial_queue: 0,
        duration_seconds: 14.0,
        events: vec![at(4.0), at(12.0)],
    };
    // 5 + 1.5 * 4 = 11, 0.5 * 4 = 2, 5 + 1.5 * 2 = 8.
    let model = p.segment_to_model_time(&clock);
    assert_eq!(model.duration_seconds, 11.0);
    assert_eq!(model.events[0].time_seconds, 2.0);
    assert_eq!(model.events[1].time_seconds, 8.0);
    assert_eq!(p.segment_to_clock_time(&model), clock);
}

#[test]
fn test_a_mapped_segment_stays_in_order_across_bucket_edges() {
    // Pairs of times as close as a float allows, either side of many edges.
    for (bucket, multipliers) in [(0.3, vec![0.7, 1.3, 1.1]), (33.3, vec![0.9, 1.7, 0.4])] {
        let p = SeasonalProfile::new(bucket, multipliers).unwrap();
        let mut events = Vec::new();
        for edge in 1..200_000u32 {
            let t = f64::from(edge) * bucket;
            for time_seconds in [t.next_down(), t, t.next_up()] {
                events.push(HawkesEvent {
                    time_seconds,
                    kind: QrEventKind::Limit,
                });
            }
        }
        let duration_seconds = 200_000.0 * bucket;
        let model = p.segment_to_model_time(&HawkesSegment {
            initial_queue: 0,
            duration_seconds,
            events,
        });
        assert!(model.duration_seconds >= model.events.last().unwrap().time_seconds);
        assert!(
            model
                .events
                .windows(2)
                .all(|w| w[0].time_seconds <= w[1].time_seconds)
        );
    }
}

#[test]
fn test_score_flow_seasonal_refuses_what_score_flow_refuses() {
    let p = SeasonalProfile::new(10.0, vec![0.5, 1.5]).unwrap();
    let at = |time_seconds| HawkesEvent {
        time_seconds,
        kind: QrEventKind::Limit,
    };
    let segment = |duration_seconds, events| HawkesSegment {
        initial_queue: 0,
        duration_seconds,
        events,
    };
    let params = truth();
    let cases = [
        segment(20.0, vec![at(5.0), at(4.0)]),
        segment(20.0, vec![at(5.0), at(5.0f64.next_down())]),
        segment(20.0, vec![at(25.0)]),
        segment(20.0, vec![at(f64::NAN)]),
        segment(20.0, vec![at(-1.0)]),
        segment(-1.0, vec![]),
        segment(f64::NAN, vec![]),
    ];
    for case in cases {
        assert!(
            matches!(
                score_flow_seasonal(std::slice::from_ref(&case), &params, &p),
                Err(MMError::InvalidMarketState(_))
            ),
            "{case:?}"
        );
        assert!(score_flow(std::slice::from_ref(&case), &params).is_err());
    }
}

#[test]
fn test_a_mapped_segment_stays_out_of_order_if_it_was() {
    let p = SeasonalProfile::new(10.0, vec![0.5, 1.5]).unwrap();
    let at = |time_seconds| HawkesEvent {
        time_seconds,
        kind: QrEventKind::Limit,
    };
    let mapped = p.segment_to_model_time(&HawkesSegment {
        initial_queue: 0,
        duration_seconds: 4.0,
        events: vec![at(14.0), at(4.0)],
    });
    // 11 and 2, and the duration 2 under the event at 11: nothing is smoothed over.
    assert_eq!(mapped.events[0].time_seconds, 11.0);
    assert_eq!(mapped.events[1].time_seconds, 2.0);
    assert_eq!(mapped.duration_seconds, 2.0);
    // An event past the end stays past it.
    let past = p.segment_to_model_time(&HawkesSegment {
        initial_queue: 0,
        duration_seconds: 4.0,
        events: vec![at(14.0)],
    });
    assert_eq!(past.duration_seconds, 2.0);
    assert_eq!(past.events[0].time_seconds, 11.0);
}

#[test]
fn test_an_ordered_segment_stays_ordered_for_many_profiles() {
    let mut state = 11u64;
    let mut unit = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 11) as f64 / 9_007_199_254_740_992.0
    };
    for _ in 0..400 {
        let count = 1 + (unit() * 20.0) as usize;
        let multipliers: Vec<f64> = (0..count).map(|_| 0.01 + unit() * 10.0).collect();
        let bucket = 0.1 + unit() * 100.0;
        let p = SeasonalProfile::new(bucket, multipliers).unwrap();
        let mut times = Vec::new();
        for _ in 0..2000 {
            let edge = (unit() * 1e5).floor() * bucket;
            times.extend([edge.next_down(), edge, edge.next_up()]);
            let inside = unit() * 1e5 * bucket;
            times.extend([inside, inside.next_up(), inside.next_up().next_up()]);
        }
        times.sort_by(f64::total_cmp);
        let events: Vec<HawkesEvent> = times
            .iter()
            .filter(|t| **t >= 0.0)
            .map(|&time_seconds| HawkesEvent {
                time_seconds,
                kind: QrEventKind::Limit,
            })
            .collect();
        let mapped = p.segment_to_model_time(&HawkesSegment {
            initial_queue: 0,
            duration_seconds: 1e5 * bucket,
            events,
        });
        assert!(
            mapped
                .events
                .windows(2)
                .all(|w| w[0].time_seconds <= w[1].time_seconds)
        );
    }
}

/// Two times one float apart whose model times step back by 1.66 epsilon of the time.
#[test]
fn test_a_step_back_above_one_epsilon_is_absorbed() {
    let p = SeasonalProfile::new(
        0.2845512768607465,
        vec![
            239.99824988007933,
            7.9206389936465404,
            0.012312631405157908,
            0.6260097999572359,
            0.735898669898175,
            1.0645153725281677,
            0.17912309764152748,
            0.06620341688644195,
            0.00019949309355531952,
            7204.743159862191,
            0.0006930553994369046,
            0.8782140251135744,
            0.08626805830325544,
            0.0018301430877892511,
            0.00024168680519148954,
            0.22009310820970146,
            47.33219458690915,
        ],
    )
    .unwrap();
    let (a, t) = (8.82108958268314, 8.821089582683141);
    assert!(p.to_model_time(a) > p.to_model_time(t));
    let mapped = p.segment_to_model_time(&HawkesSegment {
        initial_queue: 0,
        duration_seconds: t * 2.0,
        events: [a, t]
            .map(|time_seconds| HawkesEvent {
                time_seconds,
                kind: QrEventKind::Limit,
            })
            .to_vec(),
    });
    assert!(mapped.events[0].time_seconds <= mapped.events[1].time_seconds);
}

#[test]
fn test_a_step_back_larger_than_a_rounding_is_not_absorbed() {
    let p = SeasonalProfile::new(1e6, vec![1.0]).unwrap();
    let later = 1000.0;
    let earlier = later * (1.0 - 64.0 * f64::EPSILON);
    let mapped = p.segment_to_model_time(&HawkesSegment {
        initial_queue: 0,
        duration_seconds: 2000.0,
        events: [later, earlier]
            .map(|time_seconds| HawkesEvent {
                time_seconds,
                kind: QrEventKind::Limit,
            })
            .to_vec(),
    });
    assert!(mapped.events[1].time_seconds < mapped.events[0].time_seconds);
}

#[test]
fn test_a_mapped_duration_stays_above_an_event_that_rounded_past_it() {
    let p = SeasonalProfile::new(0.3, vec![0.7, 1.3, 1.1]).unwrap();
    let mut seen = 0;
    for edge in 1..200_000u32 {
        let end = f64::from(edge) * 0.3;
        let before = end.next_down();
        if p.to_model_time(before) <= p.to_model_time(end) {
            continue;
        }
        seen += 1;
        let mapped = p.segment_to_model_time(&HawkesSegment {
            initial_queue: 0,
            duration_seconds: end,
            events: vec![HawkesEvent {
                time_seconds: before,
                kind: QrEventKind::Limit,
            }],
        });
        assert!(mapped.duration_seconds >= mapped.events[0].time_seconds);
    }
    assert!(seen > 0, "no edge rounded back, so this tested nothing");
}

#[cfg(feature = "serde")]
mod seasonal_serde {
    use super::*;

    fn text(bucket: f64, multipliers: &[f64]) -> String {
        let quoted: Vec<String> = multipliers.iter().map(|m| format!("\"{m:?}\"")).collect();
        format!(
            r#"{{"bucket_seconds":"{bucket:?}","multipliers":[{}]}}"#,
            quoted.join(",")
        )
    }

    #[test]
    fn test_profile_reads_back_equal_for_many_profiles() {
        // A plain generator: the multipliers of a normalized profile are not short decimals.
        let mut state = 7u64;
        let mut unit = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / 9_007_199_254_740_992.0
        };
        let mut moved_by_rounding = 0;
        for _ in 0..2000 {
            let count = 1 + (unit() * 50.0) as usize;
            let multipliers: Vec<f64> = (0..count).map(|_| 0.01 + unit() * 10.0).collect();
            let p = SeasonalProfile::new(0.1 + unit() * 1000.0, multipliers).unwrap();
            let json = serde_json::to_string(&p).unwrap();
            assert_eq!(serde_json::from_str::<SeasonalProfile>(&json).unwrap(), p);
            let mean = p.multipliers().iter().sum::<f64>() / count as f64;
            moved_by_rounding += usize::from(mean != 1.0);
        }
        // So a tolerance of zero on the mean would refuse real profiles.
        assert!(moved_by_rounding > 0);
    }

    #[test]
    fn test_a_large_skewed_profile_from_new_reads_back_equal() {
        // The sum rounds further here than the tolerance for a small profile.
        let mut multipliers = vec![1.0; MAX_SEASONAL_BUCKETS];
        multipliers[0] = 1e8;
        let p = SeasonalProfile::new(1.0, multipliers).unwrap();
        let mean = p.multipliers().iter().sum::<f64>() / MAX_SEASONAL_BUCKETS as f64;
        assert!((mean - 1.0).abs() > 1e-12, "{mean}");
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<SeasonalProfile>(&json).unwrap(), p);
    }

    #[test]
    fn test_a_large_profile_just_over_its_tolerance_is_refused() {
        // n * EPSILON for this size is 2.2e-10; a mean off by 1e-9 is over it.
        let mut multipliers = vec![1.0; MAX_SEASONAL_BUCKETS];
        multipliers[0] = 1.0 + 1e-9 * MAX_SEASONAL_BUCKETS as f64;
        let json = serde_json::json!({
            "bucket_seconds": "1.0",
            "multipliers": multipliers.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>(),
        });
        assert!(serde_json::from_value::<SeasonalProfile>(json).is_err());
    }

    #[test]
    fn test_profile_read_holds_the_invariants_of_new() {
        for bad in [
            text(0.0, &[1.0]),
            text(1.0, &[]),
            text(1.0, &[2.0, 2.0]),
            text(1e9, &[1.0000000005]),
            text(1.0, &[1.0 + 1e-11]),
            text(1.0, &[f64::NAN]),
            r#"{"bucket_seconds":"x","multipliers":["1.0"]}"#.to_string(),
            r#"{"bucket_seconds":"1.0","multipliers":["1.0"],"x":1}"#.to_string(),
        ] {
            assert!(
                serde_json::from_str::<SeasonalProfile>(&bad).is_err(),
                "{bad}"
            );
        }
        assert!(serde_json::from_str::<SeasonalProfile>(&text(1.0, &[1.0])).is_ok());
    }

    #[test]
    fn test_a_profile_read_within_the_tolerance_does_not_step_at_a_period_edge() {
        let read: SeasonalProfile = serde_json::from_str(&text(1e9, &[1.0 + 5e-13])).unwrap();
        let (before, at) = (1e9f64.next_down(), 1e9);
        assert!(read.to_model_time(before) <= read.to_model_time(at) + 1e-6);
        // The two maps agree on the period, so a time past the edge comes back.
        let t = 1.5e9;
        assert!((read.to_clock_time(read.to_model_time(t)) - t).abs() < 1e-6);
    }
}
