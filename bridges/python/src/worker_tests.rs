//! The worker against a direct environment, and its lifetime.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use market_maker_rs::backtest::kernel::{KernelFillModel, quote_error_code};
use market_maker_rs::rl::{
    ACTION_SPACE_VERSION, CancelActions, EPISODE_VERSION, EpisodeRecorder,
    HEDGED_ACTION_SPACE_VERSION, HEDGED_OBSERVATION_SPACE_VERSION, HEDGED_REWARD_VERSION,
    OBSERVATION_SPACE_VERSION, REWARD_VERSION, ResidualBounds, ResidualEnv, ResidualProposal,
    RlAction, RlEnv, SpecId,
};
use market_maker_rs::types::error::MMError;

use super::*;

fn config(terminal: &str, policy: &str, max_steps: u64) -> BridgeConfig {
    BridgeConfig::from_json(&config_text(terminal, policy, max_steps)).unwrap()
}

/// The JSON of [`config`].
fn config_text(terminal: &str, policy: &str, max_steps: u64) -> String {
    format!(
        r#"{{
            "tick_size": "0.5", "lot_size": "1", "max_price": "1000", "max_order_size": "10",
            "max_steps": {max_steps}, "fee_rate": "0.001", "initial_capital": "0",
            "latency": {{"quantum_ns": 1000000, "observation_q": 0, "submit_q": 0, "cancel_q": 0,
                         "ack_q": 0, "jitter_q": 0}},
            "fill_model": {{"kind": "immediate"}},
            "limits": null,
            "bid_offsets": [0, 1, 10], "ask_offsets": [0, 1, 10], "sizes": [1, 2],
            "scales": {{"spread_ticks": "10", "mid_change_ticks": "4", "row_gap_ms": "4000",
                        "size_lots": "10", "position": "4", "entry_gap": "2"}},
            "f32_unit": "1", "invalid_action": "{policy}", "invalid_action_penalty": "0.25",
            "terminal": "{terminal}"
        }}"#
    )
}

/// The hand computed long trajectory of the crate's own tests, on its grid.
fn long_rows() -> Vec<KernelTick> {
    let row = |ms: u64, bid, ask| KernelTick {
        ts_ns: ms * 1_000_000,
        bid,
        bid_lots: 5,
        ask,
        ask_lots: 5,
    };
    vec![
        row(1000, 199, 201),
        row(2000, 197, 199),
        row(3000, 200, 202),
    ]
}

fn quote(bid: usize, ask: usize, bid_size: usize, ask_size: usize) -> RlAction {
    RlAction::Quote {
        bid,
        ask,
        bid_size,
        ask_size,
    }
}

#[test]
fn test_worker_final_transitions_match_the_hand_computation() {
    let mut worker = Worker::spawn(config("close", "mask", 100), long_rows(), None).unwrap();
    assert!(!worker.reset(1).unwrap().done);
    let first = worker.step(quote(1, 2, 1, 0)).unwrap();
    assert_eq!(first.transition.reward.total.to_string(), "0.801");
    let second = worker.step(RlAction::Hold).unwrap();
    assert_eq!(second.transition.reward.total.to_string(), "-2");
    let third = worker.step(RlAction::Hold).unwrap();
    assert_eq!(third.transition.reward.total.normalize().to_string(), "1.8");
    assert!(third.transition.terminated && !third.transition.truncated);
    assert_eq!(third.net.normalize().to_string(), "0.601");
}

fn actions(step: usize) -> RlAction {
    match step % 4 {
        0 => quote(0, 0, 1, 0),
        1 => RlAction::Hold,
        2 => quote(1, 1, 0, 1),
        _ => quote(0, 2, 0, 0),
    }
}

#[test]
fn test_worker_is_equal_to_a_direct_environment_on_every_transition() {
    let rows: Vec<KernelTick> = (0..200_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let mut bridge = config("close", "mask", 1000);
    bridge.latency = LatencyConfig {
        quantum_ns: 1_000_000,
        observation_q: 1,
        submit_q: 1,
        cancel_q: 1,
        ack_q: 2,
        jitter_q: 2,
    };
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let info = worker.info();
    assert_eq!(info.spec, env.spec().id());
    assert_eq!(info.kernel_fingerprint, env.kernel().config_fingerprint());
    assert_eq!(info.dataset_digest, data.id().digest);
    assert_eq!(info.rows, 200);
    assert_eq!((info.features, info.offsets, info.sizes), (11, 3, 2));
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(9, &mut recorder).unwrap();
    let reset = worker.reset(9).unwrap();
    assert_eq!(reset.observation, env.observe(&state).unwrap());
    assert_eq!(reset.mask, env.mask(&state));
    let mut step = 0;
    while !state.is_done() {
        let direct = env
            .step_with(&mut state, actions(step), &mut recorder)
            .unwrap();
        let through = worker.step(actions(step)).unwrap();
        assert_eq!(through.transition, direct, "step {step}");
        assert_eq!(through.net, state.net());
        assert_eq!(through.penalties, state.penalties());
        step += 1;
    }
    assert!(step > 100);
    env.reconcile(&state, &recorder).unwrap();
}

#[test]
fn test_worker_truncates_as_the_environment_does() {
    let mut worker = Worker::spawn(config("close", "mask", 2), long_rows(), None).unwrap();
    worker.reset(1).unwrap();
    worker.step(quote(1, 2, 1, 0)).unwrap();
    let cut = worker.step(RlAction::Hold).unwrap();
    assert!(cut.transition.truncated && !cut.transition.terminated);
    assert_eq!(cut.transition.close, None);
}

#[test]
fn test_worker_refuses_an_unknown_specification_before_any_step() {
    let guard = Arc::new(());
    let known = Worker::spawn(config("mark", "mask", 100), long_rows(), None)
        .unwrap()
        .info()
        .spec;
    Worker::spawn(config("mark", "mask", 100), long_rows(), Some(known)).unwrap();
    let mut other = known;
    other.fingerprint ^= 1;
    let error = Worker::spawn_holding(
        config("mark", "mask", 100),
        long_rows(),
        Some(other),
        Arc::clone(&guard),
    )
    .unwrap_err();
    assert!(matches!(error, BridgeError::Refused(_)));
    // Its thread is gone, and took the data with it.
    assert_eq!(Arc::strong_count(&guard), 1);
    let mut later = known;
    later.reward += 1;
    assert!(Worker::spawn(config("mark", "mask", 100), long_rows(), Some(later)).is_err());
}

#[test]
fn test_worker_refuses_what_the_kernel_or_the_json_refuses() {
    let refused = |result: Result<Worker, BridgeError>| {
        assert!(matches!(result, Err(BridgeError::Refused(_))));
    };
    assert!(matches!(
        BridgeConfig::from_json(r#"{"tick_size": "0.5", "surprise": 1}"#),
        Err(BridgeError::Refused(_))
    ));
    let mut bad = config("mark", "mask", 100);
    bad.tick_size = "half".to_string();
    refused(Worker::spawn(bad, long_rows(), None));
    let mut unequal = config("mark", "mask", 100);
    unequal.ask_offsets = vec![0, 1];
    refused(Worker::spawn(unequal, long_rows(), None));
    let mut unbuilt = config("mark", "mask", 100);
    unbuilt.bid_offsets = vec![0, 1, 2, 3, 4, 5, 6];
    unbuilt.ask_offsets = unbuilt.bid_offsets.clone();
    refused(Worker::spawn(unbuilt, long_rows(), None));
    let mut crossed = long_rows();
    crossed[1].bid = 300;
    refused(Worker::spawn(config("mark", "mask", 100), crossed, None));
    let mut penalty = config("mark", "mask", 100);
    penalty.invalid_action_penalty = "-1".to_string();
    refused(Worker::spawn(penalty, long_rows(), None));
}

#[test]
fn test_worker_stops_after_an_error() {
    let guard = Arc::new(());
    let mut worker = Worker::spawn_holding(
        config("mark", "mask", 100),
        long_rows(),
        None,
        Arc::clone(&guard),
    )
    .unwrap();
    worker.reset(1).unwrap();
    // The second size index is fine, a third is off the grid and outside the mask.
    let error = worker.step(quote(0, 0, 2, 0)).unwrap_err();
    assert!(matches!(error, BridgeError::Failed(_)));
    assert!(worker.is_closed());
    assert_eq!(
        worker.step(RlAction::Hold).unwrap_err(),
        BridgeError::Closed
    );
    assert_eq!(worker.reset(1).unwrap_err(), BridgeError::Closed);
    assert_eq!(Arc::strong_count(&guard), 1);
}

#[test]
fn test_worker_refuses_a_step_before_a_reset_and_after_the_end() {
    let mut early = Worker::spawn(config("mark", "mask", 100), long_rows(), None).unwrap();
    assert!(matches!(
        early.step(RlAction::Hold),
        Err(BridgeError::Failed(_))
    ));
    let mut late = Worker::spawn(config("mark", "mask", 100), long_rows(), None).unwrap();
    late.reset(1).unwrap();
    for _ in 0..3 {
        late.step(RlAction::Hold).unwrap();
    }
    assert!(matches!(
        late.step(RlAction::Hold),
        Err(BridgeError::Failed(_))
    ));
}

#[test]
fn test_worker_a_new_reset_restarts_the_episode() {
    let mut worker = Worker::spawn(config("mark", "mask", 100), long_rows(), None).unwrap();
    let first = worker.reset(1).unwrap();
    worker.step(RlAction::Hold).unwrap();
    assert_eq!(worker.reset(1).unwrap(), first);
    worker.step(quote(1, 2, 1, 0)).unwrap();
    let again = worker.step(RlAction::Hold).unwrap();
    assert_eq!(again.net.normalize().to_string(), "-1.199");
}

#[test]
fn test_worker_panic_ends_it_and_nothing_continues() {
    let guard = Arc::new(());
    let mut worker = Worker::spawn_holding(
        config("mark", "mask", 100),
        long_rows(),
        None,
        Arc::clone(&guard),
    )
    .unwrap();
    worker.reset(1).unwrap();
    let error = worker.trigger_panic().unwrap_err();
    assert!(matches!(&error, BridgeError::Panicked(text) if text.contains("asked for")));
    assert_eq!(
        worker.step(RlAction::Hold).unwrap_err(),
        BridgeError::Closed
    );
    assert_eq!(Arc::strong_count(&guard), 1);
}

#[test]
fn test_worker_close_and_drop_end_the_thread_and_free_the_data() {
    let guard = Arc::new(());
    let mut worker = Worker::spawn_holding(
        config("mark", "mask", 100),
        long_rows(),
        None,
        Arc::clone(&guard),
    )
    .unwrap();
    assert_eq!(Arc::strong_count(&guard), 2);
    assert!(!worker.is_closed());
    worker.close().unwrap();
    assert!(worker.is_closed());
    assert_eq!(Arc::strong_count(&guard), 1);
    worker.close().unwrap();
    assert_eq!(
        worker.step(RlAction::Hold).unwrap_err(),
        BridgeError::Closed
    );

    let dropped = Worker::spawn_holding(
        config("mark", "mask", 100),
        long_rows(),
        None,
        Arc::clone(&guard),
    )
    .unwrap();
    assert_eq!(Arc::strong_count(&guard), 2);
    drop(dropped);
    assert_eq!(Arc::strong_count(&guard), 1);
}

#[test]
fn test_config_builds_the_kernel_and_the_choices_it_states() {
    let mut bridge = config("close", "reject", 7);
    bridge.latency = LatencyConfig {
        quantum_ns: 500,
        observation_q: 1,
        submit_q: 2,
        cancel_q: 3,
        ack_q: 4,
        jitter_q: 5,
    };
    bridge.fill_model = FillConfig::QueuePosition {
        rate: "2".to_string(),
        min_queue_ms: 9,
    };
    bridge.limits = Some(LimitsConfig {
        max_position: "3".to_string(),
        max_notional: "400".to_string(),
        scaling_factor: "0.5".to_string(),
    });
    bridge.initial_capital = "1000".to_string();
    let d = |text: &str| Decimal::from_str(text).unwrap();
    let expected = KernelConfig::new(d("0.5"), d("1"), d("1000"), d("10"), 7)
        .with_fee_rate(d("0.001"))
        .with_initial_capital(d("1000"))
        .with_latency(
            KernelLatency::zero(500)
                .with_delays(1, 2, 3, 4)
                .with_jitter(5),
        )
        .with_fill_model(KernelFillModel::QueuePosition {
            rate: d("2"),
            min_queue_ms: 9,
        })
        .with_limits(RiskLimits::new(d("3"), d("400"), d("0.5")).unwrap());
    assert_eq!(bridge.kernel().unwrap(), expected);
    let scales = ObservationScales {
        spread_ticks: d("10"),
        mid_change_ticks: d("4"),
        row_gap_ms: d("4000"),
        size_lots: d("10"),
        position: d("4"),
        entry_gap: d("2"),
    };
    let choices = RlConfig::new(scales, d("1"))
        .with_invalid_action(InvalidActionPolicy::Reject, d("0.25"))
        .with_terminal(TerminalValuation::CloseAtTouch);
    assert_eq!(bridge.rl().unwrap(), choices);
    let marked =
        RlConfig::new(scales, d("1")).with_invalid_action(InvalidActionPolicy::Mask, d("0.25"));
    assert_eq!(config("mark", "mask", 7).rl().unwrap(), marked);
}

#[test]
fn test_versions_are_the_ones_this_bridge_was_written_for() {
    assert_eq!(
        BUILT_FOR,
        (
            OBSERVATION_SPACE_VERSION,
            ACTION_SPACE_VERSION,
            REWARD_VERSION,
            EPISODE_VERSION
        )
    );
    let id = Worker::spawn(config("mark", "mask", 100), long_rows(), None)
        .unwrap()
        .info()
        .spec;
    assert!(known_versions(&id, false, false));
    for change in 0..4 {
        let mut later = id;
        match change {
            0 => later.observation += 1,
            1 => later.action += 1,
            2 => later.reward += 1,
            _ => later.episode += 1,
        }
        assert!(!known_versions(&later, false, false));
    }
    // An environment with a hedge lane in a second instrument is not one this bridge drives.
    let hedged = SpecId {
        observation: HEDGED_OBSERVATION_SPACE_VERSION,
        action: HEDGED_ACTION_SPACE_VERSION,
        reward: HEDGED_REWARD_VERSION,
        ..id
    };
    assert!(!known_versions(&hedged, false, false));
}

#[test]
fn test_worker_reports_the_running_penalties() {
    let mut worker = Worker::spawn(config("mark", "reject", 100), long_rows(), None).unwrap();
    worker.reset(1).unwrap();
    let off_grid = worker.step(quote(0, 0, 2, 0)).unwrap();
    assert_eq!(off_grid.penalties.to_string(), "0.25");
    assert_eq!(off_grid.transition.reward.penalty.to_string(), "0.25");
    let again = worker.step(quote(0, 3, 0, 0)).unwrap();
    assert_eq!(again.penalties.to_string(), "0.50");
    assert_eq!(again.net.to_string(), "0");
}

fn residual_config(baseline: &str) -> BridgeConfig {
    let mut bridge = config("close", "reject", 100);
    let text = format!(
        r#"{{"baseline": {baseline}, "price_ticks": 4, "size_lots": 1, "default_order_size": "1"}}"#
    );
    bridge.residual = Some(serde_json::from_str(&text).unwrap());
    bridge
}

const AVELLANEDA: &str = r#"{"kind": "avellaneda_stoikov", "risk_aversion": "0.1",
    "volatility": "0.2", "order_intensity": "1.5", "session_start_ms": 0,
    "horizon_ms": 100000, "size": "1"}"#;
const GLFT: &str = r#"{"kind": "glft", "risk_aversion": "0.01", "order_intensity_k": "0.3",
    "intensity_scale_a": "0.9", "max_inventory_lots": 30, "terminal_time": 100000,
    "min_spread": "0", "volatility": "5", "session_start_ms": 0}"#;

#[test]
fn test_worker_residual_logs_and_equals_a_direct_environment() {
    for (baseline, source) in [(AVELLANEDA, true), (GLFT, false)] {
        let bridge = residual_config(baseline);
        let kernel = bridge.kernel().unwrap();
        // Enough rows to hold inventory, which is where the volatility reaches a quote.
        let rows: Vec<KernelTick> = (0..100_u64)
            .map(|i| {
                let bid = 190 + (i * 7) % 13;
                KernelTick {
                    ts_ns: (1000 + i * 25) * 1_000_000,
                    bid,
                    bid_lots: 4,
                    ask: bid + 2,
                    ask_lots: 3,
                }
            })
            .collect();
        let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
        let rl = bridge.rl().unwrap();
        let bounds = ResidualBounds {
            price_ticks: 4,
            size_lots: 1,
        };
        let proposal = ResidualProposal {
            bid_away: 0.5,
            ask_away: -0.25,
            bid_size: 1.0,
            ask_size: 0.0,
        };
        let d = |text: &str| Decimal::from_str(text).unwrap();
        let mut worker = Worker::spawn(bridge, rows, None).unwrap();
        worker.reset(1).unwrap();
        let through: Vec<StepReply> = (0..100)
            .map(|_| worker.step_residual(proposal).unwrap())
            .collect();
        macro_rules! compare {
            ($source:expr) => {{
                let mut env: ResidualEnv<'_, _, ORD, MSG, DLV, OBS> =
                    ResidualEnv::new(&kernel, &data, $source, d("1"), bounds, rl).unwrap();
                let mut state = env.reset(1).unwrap();
                let mut filled = false;
                for reply in &through {
                    let step = env.step(&mut state, proposal).unwrap();
                    assert_eq!(reply.transition, step.transition);
                    assert_eq!(reply.decision, Some(step.decision));
                    assert_eq!(step.decision.delta.bid_away_ticks, 2);
                    filled |= !state.kernel().venue_position().is_zero();
                }
                assert_eq!(through[99].net, state.net());
                assert!(filled, "the fixture never holds inventory");
            }};
        }
        if source {
            compare!(AvellanedaQuoter {
                risk_aversion: d("0.1"),
                volatility: d("0.2"),
                order_intensity: d("1.5"),
                session_start_ms: 0,
                horizon_ms: 100_000,
                size: d("1"),
            });
        } else {
            compare!(GlftQuoter {
                config: GLFTConfig::new(d("0.01"), d("0.3"), d("0.9"), 30, d("1"), 100_000, d("0"))
                    .unwrap(),
                volatility: d("5"),
                size: d("1"),
                session_start_ms: 0,
                form: market_maker_rs::backtest::kernel::GlftForm::Exact,
                counts: market_maker_rs::backtest::kernel::GlftCounts::default(),
            });
        }
    }
}

#[test]
fn test_worker_refuses_the_wrong_kind_of_action_and_ends() {
    let mut residual = Worker::spawn(residual_config(AVELLANEDA), long_rows(), None).unwrap();
    residual.reset(1).unwrap();
    assert!(matches!(
        residual.step(quote(0, 0, 1, 0)),
        Err(BridgeError::Failed(_))
    ));
    assert_eq!(residual.reset(1).unwrap_err(), BridgeError::Closed);
    let mut grid = Worker::spawn(config("mark", "mask", 100), long_rows(), None).unwrap();
    grid.reset(1).unwrap();
    assert!(matches!(
        grid.step_residual(ResidualProposal::ZERO),
        Err(BridgeError::Failed(_))
    ));
}

#[test]
fn test_worker_refuses_an_invalid_baseline() {
    let bad = |json: &str, field: &str, value: &str| {
        let text = json.replace(field, value);
        let result = Worker::spawn(residual_config(&text), long_rows(), None);
        assert!(matches!(result, Err(BridgeError::Refused(_))), "{value}");
    };
    bad(AVELLANEDA, r#""0.2""#, r#""not a number""#);
    bad(
        GLFT,
        r#""max_inventory_lots": 30"#,
        r#""max_inventory_lots": 0"#,
    );
    let mut size = residual_config(AVELLANEDA);
    size.residual.as_mut().unwrap().default_order_size = "x".to_string();
    assert!(matches!(
        Worker::spawn(size, long_rows(), None),
        Err(BridgeError::Refused(_))
    ));
}

#[test]
fn test_worker_attribution_splits_the_hand_computed_trajectory() {
    let mut worker = Worker::spawn(config("close", "mask", 100), long_rows(), None).unwrap();
    assert!(matches!(worker.attribution(), Err(BridgeError::Failed(_))));
    let mut worker = Worker::spawn(config("close", "mask", 100), long_rows(), None).unwrap();
    worker.reset(1).unwrap();
    worker.step(quote(1, 2, 1, 0)).unwrap();
    worker.step(RlAction::Hold).unwrap();
    worker.step(RlAction::Hold).unwrap();
    let reply = worker.attribution().unwrap();
    assert_eq!(reply.fills, 2);
    let a = reply.attribution;
    assert_eq!(a.net_pnl.normalize().to_string(), "0.601");
    assert_eq!(a.quote_fees.normalize().to_string(), "0.399");
    // The same net of fees, split between the fills' edge and the position's moves.
    let gross = a.spread_capture + a.inventory_pnl;
    assert_eq!(gross.normalize().to_string(), "1");
    // Bought two lots, then closed at the touch at the end.
    let path: Vec<(String, String)> = reply
        .positions
        .iter()
        .map(|p| {
            (
                p.price.normalize().to_string(),
                p.position.normalize().to_string(),
            )
        })
        .collect();
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].1, "2");
    assert_eq!(path[1].1, "0");
    assert!(reply.positions[0].timestamp_ms <= reply.positions[1].timestamp_ms);
    assert_eq!(reply.inventory.max_abs.normalize().to_string(), "2");
    assert_eq!(reply.inventory.final_position.normalize().to_string(), "0");
    assert!(reply.window_ms.0 < reply.window_ms.1);
    assert!(reply.max_drawdown >= Decimal::ZERO);
}

#[test]
fn test_baselines_parse_to_what_they_state() {
    let d = |text: &str| Decimal::from_str(text).unwrap();
    let parsed = |json: &str| {
        let baseline: BaselineConfig = serde_json::from_str(json).unwrap();
        parse_baseline(&baseline, "1").unwrap()
    };
    assert_eq!(
        parsed(AVELLANEDA),
        Baseline::Avellaneda(AvellanedaQuoter {
            risk_aversion: d("0.1"),
            volatility: d("0.2"),
            order_intensity: d("1.5"),
            session_start_ms: 0,
            horizon_ms: 100_000,
            size: d("1"),
        })
    );
    assert_eq!(
        parsed(GLFT),
        Baseline::Glft(GlftQuoter {
            config: GLFTConfig::new(d("0.01"), d("0.3"), d("0.9"), 30, d("1"), 100_000, d("0"))
                .unwrap(),
            volatility: d("5"),
            size: d("1"),
            session_start_ms: 0,
            form: market_maker_rs::backtest::kernel::GlftForm::Exact,
            counts: market_maker_rs::backtest::kernel::GlftCounts::default(),
        })
    ); // The form, when stated.
    let fallback = GLFT.replace(
        r#""session_start_ms": 0}"#,
        r#""session_start_ms": 0, "form": "exact_or_closed_form"}"#,
    );
    let Baseline::Glft(quoter) = parsed(&fallback) else {
        panic!("a glft baseline");
    };
    assert_eq!(
        quoter.form,
        market_maker_rs::backtest::kernel::GlftForm::ExactOrClosedForm
    );
    let closed = GLFT.replace(
        r#""session_start_ms": 0}"#,
        r#""session_start_ms": 0, "form": "closed_form"}"#,
    );
    let Baseline::Glft(quoter) = parsed(&closed) else {
        panic!("a glft baseline");
    };
    assert_eq!(
        quoter.form,
        market_maker_rs::backtest::kernel::GlftForm::ClosedForm
    );
    let unknown = GLFT.replace(
        r#""session_start_ms": 0}"#,
        r#""session_start_ms": 0, "form": "fast"}"#,
    );
    assert!(serde_json::from_str::<BaselineConfig>(&unknown).is_err());
}

#[test]
fn test_a_strategy_error_is_reported_as_what_it_is() {
    // GLFT over a day, with a large inventory bound: the exact quote needs more steps of the
    // matrix exponential than it allows, so the strategy fails with a NumericalError.
    let baseline = r#"{"kind": "glft", "risk_aversion": "0.001", "order_intensity_k": "50",
        "intensity_scale_a": "1.5", "max_inventory_lots": 5, "terminal_time": 86400000,
        "min_spread": "0", "volatility": "50", "session_start_ms": 0}"#;
    let mut worker = Worker::spawn(residual_config(baseline), long_rows(), None).unwrap();
    worker.reset(1).unwrap();
    let proposal = ResidualProposal {
        bid_away: 0.0,
        ask_away: 0.0,
        bid_size: 0.0,
        ask_size: 0.0,
    };
    let reply = worker.step_residual(proposal).unwrap();
    let code = quote_error_code(&MMError::NumericalError(String::new()));
    assert_eq!(
        reply.transition.action_status,
        ActionStatus::QuoteError(code)
    );
    assert_eq!(
        refused_because(reply.transition.action_status).as_deref(),
        Some("strategy error: NumericalError (code 3)")
    );
}

#[test]
fn test_refusals_and_quote_errors_name_their_cause() {
    use market_maker_rs::backtest::kernel::{
        ERR_KERNEL_MISMATCH, ERR_PRICE_OUT_OF_RANGE, ERR_SIZE_OUT_OF_RANGE, ErrCode, RejectReason,
    };
    use market_maker_rs::rl::{ERR_ACTION_OFF_GRID, ERR_HEDGE_REFUSED};
    assert_eq!(refused_because(ActionStatus::Applied), None);
    assert_eq!(
        refused_because(ActionStatus::Rejected(RejectReason::SizeRange)).as_deref(),
        Some("SizeRange")
    );
    let quote = |code: ErrCode| refused_because(ActionStatus::QuoteError(code)).unwrap();
    assert_eq!(quote(ERR_ACTION_OFF_GRID), "off grid (code 200)");
    assert_eq!(quote(ERR_HEDGE_REFUSED), "hedge not available (code 201)");
    assert_eq!(
        quote(ERR_PRICE_OUT_OF_RANGE),
        "strategy quote price out of range (code 100)"
    );
    assert_eq!(
        quote(ERR_SIZE_OUT_OF_RANGE),
        "strategy quote size out of range (code 101)"
    );
    assert_eq!(
        quote(ERR_KERNEL_MISMATCH),
        "strategy quote for another kernel (code 102)"
    );
    assert_eq!(
        quote(ErrCode(5)),
        "strategy error: InvalidQuoteGeneration (code 5)"
    );
    assert_eq!(quote(ErrCode(999)), "quote error (code 999)");
    // No strategy error reads as off grid.
    for code in 1..=15 {
        assert!(quote(ErrCode(code)).starts_with("strategy error: "));
    }
}

#[test]
fn test_worker_with_hedge_actions_is_equal_to_a_direct_environment_and_hedges() {
    let rows: Vec<KernelTick> = (0..200_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let mut bridge = config("close", "reject", 1000);
    bridge.hedge_actions = true;
    bridge.latency = LatencyConfig {
        quantum_ns: 1_000_000,
        observation_q: 1,
        submit_q: 1,
        cancel_q: 1,
        ack_q: 2,
        jitter_q: 2,
    };
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    assert_eq!(env.spec().action.hedge, HedgeActions::MarketableAtTouch);
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let info = worker.info();
    assert!(info.hedge_actions);
    assert_eq!(info.spec, env.spec().id());
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(9, &mut recorder).unwrap();
    let reset = worker.reset(9).unwrap();
    assert_eq!(reset.hedge_mask, env.hedge_mask(&state));
    // Quote, and hedge whenever the hedge mask allows the first size: a hedge is taken.
    let (mut step, mut hedges) = (0, 0);
    while !state.is_done() {
        let action = if env.hedge_mask(&state) & 1 == 1 && step % 3 == 0 {
            hedges += 1;
            RlAction::Hedge { size: 0 }
        } else {
            actions(step)
        };
        let direct = env.step_with(&mut state, action, &mut recorder).unwrap();
        let through = worker.step(action).unwrap();
        assert_eq!(through.transition, direct, "step {step}");
        assert_eq!(through.net, state.net());
        step += 1;
    }
    assert!(hedges > 0, "no hedge was taken");
    env.reconcile(&state, &recorder).unwrap();
}

#[test]
fn test_hedge_actions_are_off_by_default_and_change_the_specification() {
    let plain = Worker::spawn(config("close", "mask", 100), long_rows(), None).unwrap();
    assert!(!plain.info().hedge_actions);
    let mut bridge = config("close", "mask", 100);
    assert!(!bridge.hedge_actions);
    bridge.hedge_actions = true;
    let hedged = Worker::spawn(bridge, long_rows(), None).unwrap();
    let (p, h) = (plain.info().spec, hedged.info().spec);
    // The same versions, so the bridge drives both; another identity.
    assert!(known_versions(&p, false, false) && known_versions(&h, false, false));
    assert_ne!(p.fingerprint, h.fingerprint);
    // The flag in the configuration.
    let mut json: serde_json::Value =
        serde_json::from_str(&config_text("close", "mask", 100)).unwrap();
    json["hedge_actions"] = serde_json::Value::Bool(true);
    let parsed = BridgeConfig::from_json(&json.to_string()).unwrap();
    assert!(parsed.hedge_actions);
}

#[test]
fn test_hedge_actions_with_a_residual_policy_are_refused() {
    let mut bridge = residual_config(AVELLANEDA);
    bridge.hedge_actions = true;
    assert!(matches!(
        Worker::spawn(bridge, long_rows(), None),
        Err(BridgeError::Refused(reason)) if reason.contains("hedge_actions")
    ));
}

/// Rows ten seconds apart, the ask resting at 99.5 with one lot from the second row on:
/// the hand computed bid of 2 at 99.5 is at the touch, one lot ahead of it.
fn slow_rows() -> Vec<KernelTick> {
    let row = |ms: u64, bid, ask| KernelTick {
        ts_ns: ms * 1_000_000,
        bid,
        bid_lots: 1,
        ask,
        ask_lots: 1,
    };
    vec![
        row(1_000, 199, 201),
        row(11_000, 197, 199),
        row(21_000, 197, 199),
        row(31_000, 197, 199),
        row(41_000, 197, 199),
    ]
}

/// The delivered position after each step of a bid of 2 at 99.5, then holds.
fn positions(fill_model: FillConfig) -> Vec<Decimal> {
    let mut bridge = config("mark", "mask", 100);
    bridge.fill_model = fill_model;
    let mut worker = Worker::spawn(bridge, slow_rows(), None).unwrap();
    worker.reset(1).unwrap();
    let mut out = Vec::new();
    let mut action = RlAction::Quote {
        bid: 1,
        ask: 2,
        bid_size: 1,
        ask_size: 0,
    };
    loop {
        let reply = worker.step(action).unwrap();
        out.push(reply.transition.observation.raw[6].unwrap());
        if reply.transition.terminated || reply.transition.truncated {
            return out;
        }
        action = RlAction::Hold;
    }
}

#[test]
fn test_a_rational_queue_rate_fills_over_rows_where_immediate_fills_at_once() {
    let d = |text: &str| Decimal::from_str(text).unwrap();
    // Immediate: the whole bid at the first row whose ask reaches it.
    assert_eq!(
        positions(FillConfig::Immediate)[..3],
        [d("2"), d("2"), d("2")]
    );
    // One lot every 9 s, one ahead: after 10 s, 1 cleared and none for us; after 20 s, 2
    // cleared and 1 for us; after 30 s, all of it.
    let slow = FillConfig::QueuePositionRational {
        lots: 1,
        per_ms: 9_000,
        min_queue_ms: 0,
    };
    assert_eq!(positions(slow)[..3], [d("0"), d("1"), d("2")]);
}

#[test]
fn test_a_rational_queue_rate_is_configured_refused_at_zero_and_in_the_fingerprint() {
    let json = |fill: &str| -> FillConfig { serde_json::from_str(fill).unwrap() };
    let rational = json(
        r#"{"kind": "queue_position_rational", "lots": 2, "per_ms": 5000, "min_queue_ms": 7}"#,
    );
    assert_eq!(
        rational,
        FillConfig::QueuePositionRational {
            lots: 2,
            per_ms: 5_000,
            min_queue_ms: 7,
        }
    );
    let mut bridge = config("mark", "mask", 100);
    bridge.fill_model = rational.clone();
    assert_eq!(
        bridge.kernel().unwrap().fill_model,
        KernelFillModel::QueuePositionRational {
            lots: 2,
            per_ms: 5_000,
            min_queue_ms: 7,
        }
    );
    // The kernel's fingerprint carries it, and the worker reports that fingerprint.
    let info = |fill: FillConfig| {
        let mut bridge = config("mark", "mask", 100);
        bridge.fill_model = fill;
        Worker::spawn(bridge, long_rows(), None)
            .unwrap()
            .info()
            .kernel_fingerprint
    };
    let base = info(rational.clone());
    assert_ne!(
        base,
        info(FillConfig::QueuePositionRational {
            lots: 2,
            per_ms: 5_001,
            min_queue_ms: 7,
        })
    );
    assert_ne!(base, info(FillConfig::Immediate));
    // Refused as the kernel refuses it: no environment, nothing ran.
    for (lots, per_ms) in [(0, 5_000), (2, 0)] {
        let mut bridge = config("mark", "mask", 100);
        bridge.fill_model = FillConfig::QueuePositionRational {
            lots,
            per_ms,
            min_queue_ms: 7,
        };
        assert!(matches!(
            Worker::spawn(bridge, long_rows(), None),
            Err(BridgeError::Refused(_))
        ));
    }
    // Unknown fields are refused, as for the other fill models.
    assert!(
        serde_json::from_str::<FillConfig>(
            r#"{"kind": "queue_position_rational", "lots": 1, "per_ms": 2, "min_queue_ms": 0, "rate": "1"}"#
        )
        .is_err()
    );
}

/// Two Avellaneda-Stoikov points that quote different sizes, so the classical action says which
/// one quoted.
fn scheduled(changes: &str) -> String {
    let two = AVELLANEDA.replace(r#""size": "1""#, r#""size": "2""#);
    format!(r#"{{"kind": "scheduled", "points": [{AVELLANEDA}, {two}], "changes": {changes}}}"#)
}

fn scheduled_rows() -> Vec<KernelTick> {
    (0..100_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect()
}

fn classical_size(decision: &ResidualDecision) -> Option<u64> {
    match decision.classical {
        market_maker_rs::backtest::kernel::KernelAction::Quote { bid_lots, .. } => Some(bid_lots),
        _ => None,
    }
}

#[test]
fn test_a_scheduled_change_applies_from_the_first_decision_that_has_observed_its_row() {
    const CHANGE: u64 = 40;
    let mut bridge = residual_config(&scheduled(&format!("[[{CHANGE}, 1]]")));
    // Under an observation delay longer than the gap between rows.
    bridge.latency.observation_q = 60;
    let kernel = bridge.kernel().unwrap();
    let rows = scheduled_rows();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let rl = bridge.rl().unwrap();
    let Baseline::Scheduled(source) = parse_baseline(
        &bridge.residual.as_ref().unwrap().baseline,
        &bridge.lot_size,
    )
    .unwrap() else {
        panic!("a scheduled baseline");
    };
    let bounds = ResidualBounds {
        price_ticks: 4,
        size_lots: 1,
    };
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    worker.reset(3).unwrap();
    let mut env: ResidualEnv<'_, _, ORD, MSG, DLV, OBS> =
        ResidualEnv::new(&kernel, &data, source, Decimal::ONE, bounds, rl).unwrap();
    let mut state = env.reset(3).unwrap();
    let (mut before, mut after) = (0, 0);
    let mut first_after = None;
    for step in 0_u64.. {
        let seen = env.env().kernel().observe(state.kernel()).row;
        let direct = env.step(&mut state, ResidualProposal::ZERO).unwrap();
        let reply = worker.step_residual(ResidualProposal::ZERO).unwrap();
        assert_eq!(reply.decision, Some(direct.decision));
        assert_eq!(reply.transition, direct.transition);
        if let Some(size) = classical_size(&direct.decision) {
            if seen.is_some_and(|row| row >= CHANGE) {
                assert_eq!(size, 2, "step {step} saw row {seen:?}");
                after += 1;
                first_after = first_after.or(seen);
            } else {
                assert_eq!(size, 1, "step {step} saw row {seen:?}");
                before += 1;
            }
        }
        if direct.transition.terminated || direct.transition.truncated {
            break;
        }
    }
    assert!(before > 0 && after > 0, "{before} {after}");
    // The first decision on the new point is the one that observed the change's row.
    assert_eq!(first_after, Some(CHANGE));
}

#[test]
fn test_a_schedule_without_changes_is_its_first_point() {
    let rows = scheduled_rows();
    let mut plain = Worker::spawn(residual_config(AVELLANEDA), rows.clone(), None).unwrap();
    let mut one = Worker::spawn(residual_config(&scheduled("[]")), rows, None).unwrap();
    plain.reset(5).unwrap();
    one.reset(5).unwrap();
    loop {
        let a = plain.step_residual(ResidualProposal::ZERO).unwrap();
        let b = one.step_residual(ResidualProposal::ZERO).unwrap();
        assert_eq!(a, b);
        if a.transition.terminated || a.transition.truncated {
            break;
        }
    }
    assert_eq!(plain.attribution().unwrap(), one.attribution().unwrap());
}

#[test]
fn test_the_point_in_force_is_that_of_the_last_change_at_or_before_the_row() {
    let baseline: BaselineConfig =
        serde_json::from_str(&scheduled("[[10, 1], [20, 0], [30, 1]]")).unwrap();
    let Baseline::Scheduled(quoter) = parse_baseline(&baseline, "1").unwrap() else {
        panic!("a scheduled baseline");
    };
    let bridge = residual_config(AVELLANEDA);
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(long_rows(), &kernel).unwrap();
    let env: RlEnv<'_, 1, 1, ORD, MSG, DLV, OBS> = RlEnv::new(
        &kernel,
        &data,
        ActionGrid::new([0], [0], [1]).unwrap(),
        bridge.rl().unwrap(),
    )
    .unwrap();
    let state = env.reset(1).unwrap();
    let observed = env.kernel().observe(state.kernel());
    let at = |row: Option<u64>| {
        let mut obs = observed;
        obs.row = row;
        quoter.in_force(&obs)
    };
    let seen: Vec<usize> = [
        None,
        Some(0),
        Some(9),
        Some(10),
        Some(19),
        Some(20),
        Some(29),
        Some(30),
        Some(99),
    ]
    .into_iter()
    .map(at)
    .collect();
    assert_eq!(seen, [0, 0, 0, 1, 1, 0, 0, 1, 1]);
}

#[test]
fn test_a_malformed_schedule_is_refused() {
    let refused = |baseline: &str| {
        let result = Worker::spawn(residual_config(baseline), long_rows(), None);
        assert!(matches!(result, Err(BridgeError::Refused(_))), "{baseline}");
    };
    refused(r#"{"kind": "scheduled", "points": [], "changes": []}"#);
    refused(&scheduled("[[5, 2]]"));
    refused(&scheduled("[[5, 0]]"));
    refused(&scheduled("[[5, 1], [5, 0]]"));
    refused(&scheduled("[[5, 1], [4, 0]]"));
    refused(&scheduled("[[5, 1], [6, 1]]"));
    let nested = format!(
        r#"{{"kind": "scheduled", "points": [{}], "changes": []}}"#,
        scheduled("[]")
    );
    refused(&nested);
    refused(&scheduled("[[5, 1]]").replace(r#""0.2""#, r#""not a number""#));
    // Unknown fields are refused as in every baseline.
    let extra = scheduled("[]").replace(r#""changes""#, r#""extra": 1, "changes""#);
    assert!(serde_json::from_str::<BaselineConfig>(&extra).is_err());
    assert!(
        Worker::spawn(
            residual_config(&scheduled("[[5, 1], [9, 0]]")),
            long_rows(),
            None
        )
        .is_ok()
    );
}

#[test]
fn test_worker_counts_the_order_reports_of_each_step() {
    let mut worker = Worker::spawn(config("close", "mask", 100), long_rows(), None).unwrap();
    worker.reset(1).unwrap();
    let counts = |accepted, canceled, fills| OrderCounts {
        accepted,
        rejected: 0,
        canceled,
        cancel_rejected: 0,
        fills,
    };
    // Two orders rest and the bid fills on the next row.
    assert_eq!(
        worker.step(quote(1, 2, 1, 0)).unwrap().orders,
        counts(2, 0, 1)
    );
    // A new quote cancels the ask still resting and places two more.
    assert_eq!(
        worker.step(quote(2, 2, 0, 0)).unwrap().orders,
        counts(2, 1, 0)
    );
    // A hold sends nothing; the close at the end is not an order.
    assert_eq!(worker.step(RlAction::Hold).unwrap().orders, counts(0, 0, 0));
}

#[test]
fn test_worker_residual_hold_keeps_the_orders_and_equals_the_inner_hold() {
    let bridge = residual_config(AVELLANEDA);
    let kernel = bridge.kernel().unwrap();
    let rows: Vec<KernelTick> = (0..40_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let d = |text: &str| Decimal::from_str(text).unwrap();
    let source = AvellanedaQuoter {
        risk_aversion: d("0.1"),
        volatility: d("0.2"),
        order_intensity: d("1.5"),
        session_start_ms: 0,
        horizon_ms: 100_000,
        size: d("1"),
    };
    let bounds = ResidualBounds {
        price_ticks: 4,
        size_lots: 1,
    };
    let zero = ResidualProposal {
        bid_away: 0.0,
        ask_away: 0.0,
        bid_size: 0.0,
        ask_size: 0.0,
    };
    let mut env: ResidualEnv<'_, _, ORD, MSG, DLV, OBS> =
        ResidualEnv::new(&kernel, &data, source, d("1"), bounds, bridge.rl().unwrap()).unwrap();
    let mut tally = Tally::default();
    let mut state = env.reset_with(1, &mut tally).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    worker.reset(1).unwrap();
    let (mut held_while_resting, mut residual_orders) = (0, 0);
    for step in 0..39 {
        tally = Tally::default();
        // Two decisions in a row, so that a decision follows one that placed orders.
        let (through, direct) = if step % 3 != 2 {
            let step = env.step_with(&mut state, zero, &mut tally).unwrap();
            let reply = worker.step_residual(zero).unwrap();
            residual_orders += reply.orders.accepted;
            (reply, step.transition)
        } else {
            let resting = state.kernel().queue_lengths().orders > 0;
            let reply = worker.step(RlAction::Hold).unwrap();
            // A hold sends nothing: no order placed or canceled.
            assert_eq!((reply.orders.accepted, reply.orders.canceled), (0, 0));
            assert!(reply.decision.is_none());
            held_while_resting += u32::from(resting);
            let direct = env
                .env()
                .step_with(&mut state, RlAction::Hold, &mut tally)
                .unwrap();
            (reply, direct)
        };
        assert_eq!(through.transition, direct, "step {step}");
        assert_eq!(through.net, state.net());
        // Each step's own reports, a residual decision's as a hold's.
        assert_eq!(through.orders, tally.counts(), "step {step}");
    }
    assert!(residual_orders > 0, "no residual decision placed an order");
    assert!(held_while_resting > 0, "no hold with orders resting");
}

/// The order reports of a step, by the name of their kind, from the kernel's own events.
#[derive(Default)]
struct Tally(std::collections::BTreeMap<String, u64>);

impl EventSink for Tally {
    fn event(&mut self, event: KernelEvent) {
        if let KernelEvent::Report(report) = event {
            *self.0.entry(format!("{:?}", report.kind)).or_default() += 1;
        }
    }
}

impl Tally {
    fn counts(&self) -> OrderCounts {
        let of = |kind: &str| self.0.get(kind).copied().unwrap_or(0);
        OrderCounts {
            accepted: of("Accepted"),
            rejected: of("Rejected"),
            canceled: of("Canceled"),
            cancel_rejected: of("CancelRejected"),
            fills: of("PartialFill") + of("Fill"),
        }
    }
}

#[test]
fn test_worker_counts_every_kind_of_report_as_the_kernel_reports_it() {
    // Queue fills in parts, a cancel that a fill outruns, and a position limit that rejects.
    let rows: Vec<KernelTick> = (0..200_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let mut bridge = config("close", "reject", 1000);
    bridge.latency = LatencyConfig {
        quantum_ns: 1_000_000,
        observation_q: 1,
        submit_q: 1,
        cancel_q: 40,
        ack_q: 2,
        jitter_q: 2,
    };
    // A lot about every row: a crossed order of 2 lots fills over two rows.
    bridge.fill_model = FillConfig::QueuePositionRational {
        lots: 1,
        per_ms: 20,
        min_queue_ms: 0,
    };
    bridge.limits = Some(LimitsConfig {
        max_position: "3".to_string(),
        max_notional: "100000".to_string(),
        scaling_factor: "1".to_string(),
    });
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let mut tally = Tally::default();
    let mut state = env.reset_with(9, &mut tally).unwrap();
    worker.reset(9).unwrap();
    let action = |step: usize| match step % 3 {
        0 => quote(0, 0, 1, 1),
        1 => quote(1, 0, 0, 1),
        _ => RlAction::Hold,
    };
    let (mut step, mut total) = (0, OrderCounts::default());
    let mut partial_fills = 0;
    while !state.is_done() {
        tally = Tally::default();
        env.step_with(&mut state, action(step), &mut tally).unwrap();
        partial_fills += tally.0.get("PartialFill").copied().unwrap_or(0);
        let through = worker.step(action(step)).unwrap();
        assert_eq!(through.orders, tally.counts(), "step {step}");
        total.accepted += through.orders.accepted;
        total.rejected += through.orders.rejected;
        total.canceled += through.orders.canceled;
        total.cancel_rejected += through.orders.cancel_rejected;
        total.fills += through.orders.fills;
        step += 1;
    }
    // Every kind is reached, a partial fill among the fills.
    assert!(
        total.accepted > 0
            && total.rejected > 0
            && total.canceled > 0
            && total.cancel_rejected > 0
            && partial_fills > 0
            && total.fills > partial_fills,
        "{total:?}, {partial_fills} partial fills"
    );
}

#[test]
fn test_worker_with_the_cancel_action_is_equal_to_a_direct_environment_and_cancels() {
    let rows: Vec<KernelTick> = (0..200_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let mut bridge = config("close", "reject", 1000);
    bridge.cancel_action = true;
    bridge.latency.cancel_q = 1;
    bridge.latency.submit_q = 1;
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    assert_eq!(env.spec().action.cancel, CancelActions::CancelOnly);
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let info = worker.info();
    assert!(info.cancel_action);
    assert_eq!(info.spec, env.spec().id());
    assert_eq!(info.spec.action, BUILT_FOR_CANCEL_ACTION);
    let mut events: Vec<market_maker_rs::backtest::kernel::KernelEvent> = Vec::new();
    let mut state = env.reset_with(9, &mut events).unwrap();
    worker.reset(9).unwrap();
    let mut step = 0;
    while !state.is_done() {
        let action = if step % 3 == 2 {
            RlAction::Cancel
        } else {
            actions(step)
        };
        let direct = env.step_with(&mut state, action, &mut events).unwrap();
        let through = worker.step(action).unwrap();
        assert_eq!(through.transition, direct, "step {step}");
        assert_eq!(through.net, state.net());
        step += 1;
    }
    let cancelled = events
        .iter()
        .filter(|e| {
            matches!(e, market_maker_rs::backtest::kernel::KernelEvent::Report(r)
                if r.kind == market_maker_rs::backtest::kernel::ExecKind::Canceled)
        })
        .count();
    assert!(cancelled > 0, "no cancel took effect");
}

#[test]
fn test_the_cancel_action_is_off_by_default_and_its_space_is_refused_without_it() {
    let plain = Worker::spawn(config("close", "reject", 100), long_rows(), None).unwrap();
    assert!(!plain.info().cancel_action);
    assert_eq!(plain.info().spec.action, ACTION_SPACE_VERSION);
    let mut bridge = config("close", "reject", 100);
    assert!(!bridge.cancel_action);
    bridge.cancel_action = true;
    let cancel = Worker::spawn(bridge.clone(), long_rows(), None).unwrap();
    let (p, c) = (plain.info().spec, cancel.info().spec);
    assert_eq!(c.action, BUILT_FOR_CANCEL_ACTION);
    assert_ne!(p.fingerprint, c.fingerprint);
    // Each space is known only with the configuration that asks for it.
    assert!(known_versions(&p, false, false) && !known_versions(&p, true, false));
    assert!(known_versions(&c, true, false) && !known_versions(&c, false, false));
    let mut e = c;
    e.action = BUILT_FOR_EXPIRY_ACTION;
    assert!(known_versions(&e, false, true) && known_versions(&e, true, true));
    assert!(!known_versions(&e, false, false) && !known_versions(&e, true, false));
    assert!(!known_versions(&c, true, true) && !known_versions(&p, false, true));
    // A checkpoint of one is refused by the other, before any step.
    assert!(matches!(
        Worker::spawn(config("close", "reject", 100), long_rows(), Some(c)),
        Err(BridgeError::Refused(_))
    ));
    assert!(matches!(
        Worker::spawn(bridge, long_rows(), Some(p)),
        Err(BridgeError::Refused(_))
    ));
    // Without it a cancel is a refused action, named as such.
    let mut plain = plain;
    plain.reset(1).unwrap();
    let reply = plain.step(RlAction::Cancel).unwrap();
    assert_eq!(
        refused_because(reply.transition.action_status).as_deref(),
        Some("cancel not available (code 203)")
    );
    // The flag in the configuration.
    let mut json: serde_json::Value =
        serde_json::from_str(&config_text("close", "mask", 100)).unwrap();
    json["cancel_action"] = serde_json::Value::Bool(true);
    assert!(
        BridgeConfig::from_json(&json.to_string())
            .unwrap()
            .cancel_action
    );
}

#[test]
fn test_the_cancel_action_with_a_residual_policy_is_refused() {
    let mut bridge = residual_config(AVELLANEDA);
    bridge.cancel_action = true;
    assert!(matches!(
        Worker::spawn(bridge, long_rows(), None),
        Err(BridgeError::Refused(reason)) if reason.contains("cancel_action")
    ));
}

/// The gate's cuts and refusals of a step, by check, from the kernel's own events.
#[derive(Default)]
struct GateTally(Vec<(String, String)>);

impl EventSink for GateTally {
    fn event(&mut self, event: KernelEvent) {
        if let KernelEvent::Gate(gate) = event {
            self.0
                .push((format!("{:?}", gate.check), format!("{:?}", gate.verdict)));
        }
    }
}

impl GateTally {
    fn counts(&self) -> GateCounts {
        let of = |check: &str, verdicts: &[&str]| {
            self.0
                .iter()
                .filter(|(c, v)| c == check && verdicts.contains(&v.as_str()))
                .count() as u64
        };
        let check = |name: &str| VerdictCounts {
            reduced: of(name, &["Reduced"]),
            rejected: of(name, &["Rejected", "ReducedOffLot"]),
        };
        GateCounts {
            limits: check("Limits"),
            combined: check("Combined"),
            margin: check("Margin"),
        }
    }
}

fn with_margin(base_deposit: Option<&str>) -> BridgeConfig {
    let mut bridge = config("close", "mask", 100);
    bridge.initial_capital = "150".to_string();
    bridge.margin = base_deposit.map(|deposit| MarginConfig {
        base_deposit: deposit.to_string(),
    });
    bridge
}

// A policy that bids and offers 2 lots on every decision, with a quote balance of 150 and a
// base deposit of 1 lot, on the long trajectory (row 1 trades down to 99.5): the balance
// covers one lot. Through the worker, the gate cuts the first bid and ask to 1 lot and
// refuses the later bids, as the kernel does directly, and the replies count it.
#[test]
fn test_worker_margin_gate_cuts_and_refuses_as_the_kernel_and_counts_it() {
    let run = |deposit: Option<&str>| {
        let bridge = with_margin(deposit);
        let kernel = bridge.kernel().unwrap();
        let data = KernelDataset::from_kernel_ticks(long_rows(), &kernel).unwrap();
        let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
        let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
            RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
        let mut worker = Worker::spawn(bridge, long_rows(), None).unwrap();
        let mut tally = GateTally::default();
        let mut state = env.reset_with(1, &mut tally).unwrap();
        worker.reset(1).unwrap();
        let mut total = GateCounts::default();
        while !state.is_done() {
            tally = GateTally::default();
            let direct = env
                .step_with(&mut state, quote(1, 2, 1, 1), &mut tally)
                .unwrap();
            let through = worker.step(quote(1, 2, 1, 1)).unwrap();
            assert_eq!(through.transition, direct);
            assert_eq!(through.gates, tally.counts());
            total.margin.reduced += through.gates.margin.reduced;
            total.margin.rejected += through.gates.margin.rejected;
            total.limits.rejected += through.gates.limits.rejected;
        }
        (state.kernel().venue_position(), total)
    };
    let (position, total) = run(Some("1"));
    assert_eq!(position, Decimal::ONE);
    // Each quote is cut until the bid fills, then every bid is refused.
    assert!(total.margin.reduced >= 2, "{total:?}");
    assert!(total.margin.rejected >= 1, "{total:?}");
    assert_eq!(total.limits, VerdictCounts::default());
    // Without the gate the same policy buys 2, and the gate decides nothing.
    let (position, total) = run(None);
    assert_eq!(position, Decimal::TWO);
    assert_eq!(total, GateCounts::default());
}

#[test]
fn test_the_margin_configuration_builds_the_kernel_gate_and_its_own_identity() {
    let plain = config("close", "mask", 100);
    assert_eq!(plain.margin, None);
    assert_eq!(plain.kernel().unwrap().margin, None);
    let gated = with_margin(Some("1"));
    assert_eq!(
        gated.kernel().unwrap().margin,
        Some(KernelMargin::new(Decimal::ONE))
    );
    // From JSON, and absent by default.
    let mut json: serde_json::Value =
        serde_json::from_str(&config_text("close", "mask", 100)).unwrap();
    assert_eq!(
        BridgeConfig::from_json(&json.to_string()).unwrap().margin,
        None
    );
    json["margin"] = serde_json::json!({"base_deposit": "1"});
    let parsed = BridgeConfig::from_json(&json.to_string()).unwrap();
    assert_eq!(parsed.margin, gated.margin);
    json["margin"] = serde_json::json!({"base_deposit": "1", "leverage": "20"});
    assert!(BridgeConfig::from_json(&json.to_string()).is_err());
    // Another kernel, so another kernel fingerprint; the plain one keeps its own. The
    // spaces are the same, so a specification is too: a checkpoint is told apart by the
    // kernel fingerprint its metadata records.
    let id = |bridge: BridgeConfig| Worker::spawn(bridge, long_rows(), None).unwrap().info();
    let (p, g) = (id(config("close", "mask", 100)), id(with_margin(Some("1"))));
    assert_ne!(p.kernel_fingerprint, g.kernel_fingerprint);
    assert_eq!(p.spec, g.spec);
    assert_eq!(
        p.kernel_fingerprint,
        id(config("close", "mask", 100)).kernel_fingerprint
    );
    // A deposit off the lot grid is refused before any step.
    assert!(Worker::spawn(with_margin(Some("0.5")), long_rows(), None).is_err());
    assert!(Worker::spawn(with_margin(Some("x")), long_rows(), None).is_err());
}

// The limits decide before the margin, and each is counted under its own check: a position
// limit of 1 lot cuts the 2 lot quotes there, and the margin never sees them.
#[test]
fn test_worker_counts_the_limits_apart_from_the_margin() {
    let mut bridge = with_margin(Some("1"));
    bridge.limits = Some(LimitsConfig {
        max_position: "1".to_string(),
        max_notional: "100000".to_string(),
        scaling_factor: "1".to_string(),
    });
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(long_rows(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    let mut worker = Worker::spawn(bridge, long_rows(), None).unwrap();
    let mut tally = GateTally::default();
    let mut state = env.reset_with(1, &mut tally).unwrap();
    worker.reset(1).unwrap();
    let mut limits = 0;
    while !state.is_done() {
        tally = GateTally::default();
        env.step_with(&mut state, quote(1, 2, 1, 1), &mut tally)
            .unwrap();
        let through = worker.step(quote(1, 2, 1, 1)).unwrap();
        assert_eq!(through.gates, tally.counts());
        limits += through.gates.limits.reduced + through.gates.limits.rejected;
    }
    assert!(limits > 0, "the limits cut nothing");
}

// A residual policy over Avellaneda-Stoikov under the balance gate: two decisions in a row,
// then a hold, so that each kind of step follows one in which the gate decided.
#[test]
fn test_worker_counts_the_gate_on_residual_decisions_and_holds() {
    let mut bridge = residual_config(AVELLANEDA);
    bridge.initial_capital = "400".to_string();
    bridge.margin = Some(MarginConfig {
        base_deposit: "2".to_string(),
    });
    let kernel = bridge.kernel().unwrap();
    let rows: Vec<KernelTick> = (0..40_u64)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let d = |text: &str| Decimal::from_str(text).unwrap();
    let source = AvellanedaQuoter {
        risk_aversion: d("0.1"),
        volatility: d("0.2"),
        order_intensity: d("1.5"),
        session_start_ms: 0,
        horizon_ms: 100_000,
        size: d("1"),
    };
    let bounds = ResidualBounds {
        price_ticks: 4,
        size_lots: 1,
    };
    // The largest size the bounds allow, so the gate has something to cut.
    let bigger = ResidualProposal {
        bid_away: 0.0,
        ask_away: 0.0,
        bid_size: 1.0,
        ask_size: 1.0,
    };
    let mut env: ResidualEnv<'_, _, ORD, MSG, DLV, OBS> =
        ResidualEnv::new(&kernel, &data, source, d("1"), bounds, bridge.rl().unwrap()).unwrap();
    let mut tally = GateTally::default();
    let mut state = env.reset_with(1, &mut tally).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    worker.reset(1).unwrap();
    let mut decided = 0;
    for step in 0..39 {
        tally = GateTally::default();
        let through = if step % 3 != 2 {
            env.step_with(&mut state, bigger, &mut tally).unwrap();
            worker.step_residual(bigger).unwrap()
        } else {
            env.env()
                .step_with(&mut state, RlAction::Hold, &mut tally)
                .unwrap();
            worker.step(RlAction::Hold).unwrap()
        };
        assert_eq!(through.gates, tally.counts(), "step {step}");
        decided += through.gates.margin.reduced + through.gates.margin.rejected;
    }
    assert!(decided > 0, "the gate cut nothing");
}

// Lot 0.5 at a price of 100 and 120 of quote: a bid of 4 lots (2.0) is cut to 1.2, not a
// whole number of lots, so the gate refuses it whole, and the reply counts it as refused.
#[test]
fn test_worker_counts_a_reduction_off_the_lot_grid_as_refused() {
    let mut bridge = with_margin(Some("0"));
    bridge.tick_size = "1".to_string();
    bridge.lot_size = "0.5".to_string();
    bridge.max_order_size = "5".to_string();
    bridge.initial_capital = "120".to_string();
    bridge.sizes = vec![1, 4];
    let rows: Vec<KernelTick> = (0..3_u64)
        .map(|i| KernelTick {
            ts_ns: (1000 + i * 1000) * 1_000_000,
            bid: 100,
            bid_lots: 10,
            ask: 101,
            ask_lots: 10,
        })
        .collect();
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 4]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let mut tally = GateTally::default();
    let mut state = env.reset_with(1, &mut tally).unwrap();
    worker.reset(1).unwrap();
    tally = GateTally::default();
    env.step_with(&mut state, quote(0, 2, 1, 0), &mut tally)
        .unwrap();
    let through = worker.step(quote(0, 2, 1, 0)).unwrap();
    assert!(
        tally
            .0
            .contains(&("Margin".to_string(), "ReducedOffLot".to_string())),
        "{:?}",
        tally.0
    );
    assert_eq!(through.gates, tally.counts());
    assert_eq!(through.gates.margin.reduced, 0);
}

/// The lots the kernel books as fills, by side, independently of the reports the worker
/// counts.
#[derive(Default)]
struct FillTally(FilledLots);

impl EventSink for FillTally {
    fn event(&mut self, event: KernelEvent) {
        if let KernelEvent::Fill(fill) = event {
            match fill.side {
                Action::Buy => self.0.buy += fill.qty,
                Action::Sell => self.0.sell += fill.qty,
                _ => {}
            }
        }
    }
}

fn crossing_rows(n: u64) -> Vec<KernelTick> {
    (0..n)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect()
}

#[test]
fn test_worker_reports_the_lots_filled_in_each_step_by_side() {
    let rows = crossing_rows(200);
    let mut bridge = config("close", "reject", 1000);
    bridge.fill_model = FillConfig::QueuePositionRational {
        lots: 1,
        per_ms: 20,
        min_queue_ms: 0,
    };
    let kernel = bridge.kernel().unwrap();
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap();
    let env: RlEnv<'_, 3, 2, ORD, MSG, DLV, OBS> =
        RlEnv::new(&kernel, &data, grid, bridge.rl().unwrap()).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    let mut tally = FillTally::default();
    let mut state = env.reset_with(9, &mut tally).unwrap();
    worker.reset(9).unwrap();
    let action = |step: usize| match step % 3 {
        0 => quote(0, 0, 1, 1),
        1 => quote(1, 0, 0, 1),
        _ => RlAction::Hold,
    };
    let (mut step, mut total) = (0, FilledLots::default());
    while !state.is_done() {
        tally = FillTally::default();
        env.step_with(&mut state, action(step), &mut tally).unwrap();
        let through = worker.step(action(step)).unwrap();
        assert_eq!(through.filled, tally.0, "step {step}");
        total.buy += through.filled.buy;
        total.sell += through.filled.sell;
        step += 1;
    }
    assert!(total.buy > 0 && total.sell > 0, "{total:?}");
}

#[test]
fn test_worker_reports_the_lots_filled_on_residual_decisions_and_holds() {
    let bridge = residual_config(AVELLANEDA);
    let kernel = bridge.kernel().unwrap();
    let rows = crossing_rows(120);
    let data = KernelDataset::from_kernel_ticks(rows.clone(), &kernel).unwrap();
    let d = |text: &str| Decimal::from_str(text).unwrap();
    let source = AvellanedaQuoter {
        risk_aversion: d("0.1"),
        volatility: d("0.2"),
        order_intensity: d("1.5"),
        session_start_ms: 0,
        horizon_ms: 100_000,
        size: d("1"),
    };
    let bounds = ResidualBounds {
        price_ticks: 4,
        size_lots: 1,
    };
    let inside = ResidualProposal {
        bid_away: 2.0,
        ask_away: 2.0,
        bid_size: 1.0,
        ask_size: 1.0,
    };
    let mut env: ResidualEnv<'_, _, ORD, MSG, DLV, OBS> =
        ResidualEnv::new(&kernel, &data, source, d("1"), bounds, bridge.rl().unwrap()).unwrap();
    let mut tally = FillTally::default();
    let mut state = env.reset_with(1, &mut tally).unwrap();
    let mut worker = Worker::spawn(bridge, rows, None).unwrap();
    worker.reset(1).unwrap();
    let mut total = FilledLots::default();
    for step in 0..119 {
        if state.is_done() {
            break;
        }
        tally = FillTally::default();
        let through = if step % 3 != 2 {
            env.step_with(&mut state, inside, &mut tally).unwrap();
            worker.step_residual(inside).unwrap()
        } else {
            env.env()
                .step_with(&mut state, RlAction::Hold, &mut tally)
                .unwrap();
            worker.step(RlAction::Hold).unwrap()
        };
        assert_eq!(through.filled, tally.0, "step {step}");
        total.buy += through.filled.buy;
        total.sell += through.filled.sell;
    }
    assert!(total.buy + total.sell > 0, "{total:?}");
}

fn grid_bridge() -> BridgeConfig {
    let mut bridge = config("close", "reject", 1000);
    bridge.fill_model = FillConfig::QueuePositionRational {
        lots: 1,
        per_ms: 20,
        min_queue_ms: 0,
    };
    bridge
}

fn grid_action(step: usize) -> RlAction {
    match step % 3 {
        0 => quote(0, 0, 1, 1),
        1 => quote(1, 0, 0, 1),
        _ => RlAction::Hold,
    }
}

/// The same rows from `from` on, moved by `shift` ticks.
fn shifted(rows: &[KernelTick], from: usize, shift: u64) -> Vec<KernelTick> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            if i < from {
                *row
            } else {
                KernelTick {
                    bid: row.bid + shift,
                    ask: row.ask + shift,
                    ..*row
                }
            }
        })
        .collect()
}

/// Runs `steps` grid steps from a reset with seed 9 and returns every reply.
fn run_steps(worker: &mut Worker, from: usize, to: usize) -> Vec<StepReply> {
    (from..to)
        .map(|step| worker.step(grid_action(step)).unwrap())
        .collect()
}

#[test]
fn test_replacing_with_the_same_rows_changes_nothing() {
    let rows = crossing_rows(120);
    let mut plain = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    let mut replaced = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    plain.reset(9).unwrap();
    replaced.reset(9).unwrap();
    let a = run_steps(&mut plain, 0, 40);
    let b = run_steps(&mut replaced, 0, 40);
    assert_eq!(a, b);
    let info = replaced.replace_rows(rows).unwrap();
    assert_eq!(info, replaced.info());
    assert_eq!(info.dataset_digest, plain.info().dataset_digest);
    assert_eq!(
        run_steps(&mut plain, 40, 100),
        run_steps(&mut replaced, 40, 100)
    );
    assert_eq!(
        plain.attribution().unwrap(),
        replaced.attribution().unwrap()
    );
}

#[test]
fn test_new_rows_after_the_read_ones_run_as_a_fresh_run_on_them() {
    let rows = crossing_rows(120);
    let new_rows = shifted(&rows, 45, 3);
    let mut replaced = Worker::spawn(grid_bridge(), rows, None).unwrap();
    replaced.reset(9).unwrap();
    let before = run_steps(&mut replaced, 0, 40);
    let read = before.last().unwrap().transition.observation.row.unwrap();
    assert!(read < 45, "row {read} read");
    let old_info = replaced.info();
    let info = replaced.replace_rows(new_rows.clone()).unwrap();
    // Only the dataset's identity changes.
    assert_eq!(info.spec, old_info.spec);
    assert_eq!(info.kernel_fingerprint, old_info.kernel_fingerprint);
    assert_ne!(info.dataset_digest, old_info.dataset_digest);
    let after = run_steps(&mut replaced, 40, 110);
    let mut fresh = Worker::spawn(grid_bridge(), new_rows, None).unwrap();
    fresh.reset(9).unwrap();
    assert_eq!(run_steps(&mut fresh, 0, 40), before);
    assert_eq!(run_steps(&mut fresh, 40, 110), after);
    assert_eq!(
        fresh.attribution().unwrap(),
        replaced.attribution().unwrap()
    );
}

#[test]
fn test_a_refused_replace_leaves_the_episode_as_it_was() {
    let rows = crossing_rows(120);
    let mut plain = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    let mut refused = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    assert!(matches!(
        refused.replace_rows(rows.clone()),
        Err(BridgeError::Refused(_))
    ));
    plain.reset(9).unwrap();
    refused.reset(9).unwrap();
    run_steps(&mut plain, 0, 40);
    run_steps(&mut refused, 0, 40);
    // A row already read changes.
    assert!(matches!(
        refused.replace_rows(shifted(&rows, 5, 1)),
        Err(BridgeError::Refused(_))
    ));
    // Rows the kernel refuses, past the read ones: a price of zero.
    let mut bad = rows.clone();
    bad[100].bid = 0;
    assert!(matches!(
        refused.replace_rows(bad),
        Err(BridgeError::Refused(_))
    ));
    assert_eq!(refused.info(), plain.info());
    assert_eq!(
        run_steps(&mut plain, 40, 100),
        run_steps(&mut refused, 40, 100)
    );
    assert_eq!(plain.attribution().unwrap(), refused.attribution().unwrap());
}

#[test]
fn test_a_replace_after_the_episode_ended_is_refused() {
    let rows = crossing_rows(20);
    let mut worker = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    worker.reset(9).unwrap();
    let mut step = 0;
    loop {
        let reply = worker.step(grid_action(step)).unwrap();
        step += 1;
        if reply.transition.terminated || reply.transition.truncated {
            break;
        }
    }
    assert!(matches!(
        worker.replace_rows(rows),
        Err(BridgeError::Refused(_))
    ));
    assert!(worker.attribution().is_ok(), "the episode is still there");
}

#[test]
fn test_a_residual_episode_replays_its_proposals_and_holds() {
    let bridge = residual_config(AVELLANEDA);
    let rows = crossing_rows(120);
    let inside = ResidualProposal {
        bid_away: 2.0,
        ask_away: 2.0,
        bid_size: 1.0,
        ask_size: 1.0,
    };
    let act = |worker: &mut Worker, step: usize| {
        if step % 3 == 2 {
            worker.step(RlAction::Hold).unwrap()
        } else {
            worker.step_residual(inside).unwrap()
        }
    };
    let mut replaced = Worker::spawn(bridge.clone(), rows.clone(), None).unwrap();
    replaced.reset(1).unwrap();
    let mut before = Vec::new();
    for step in 0..10 {
        before.push(act(&mut replaced, step));
    }
    let read = before.last().unwrap().transition.observation.row.unwrap() as usize;
    let new_rows = shifted(&rows, read + 1, 2);
    let mut fresh = Worker::spawn(bridge, new_rows.clone(), None).unwrap();
    fresh.reset(1).unwrap();
    for (step, reply) in before.iter().enumerate() {
        assert_eq!(&act(&mut fresh, step), reply, "step {step}");
    }
    replaced.replace_rows(new_rows).unwrap();
    let mut steps = 10;
    for step in 10.. {
        let (a, b) = (act(&mut replaced, step), act(&mut fresh, step));
        assert_eq!(a, b, "step {step}");
        steps += 1;
        if a.transition.terminated || a.transition.truncated {
            break;
        }
    }
    assert!(steps > 10, "{steps} steps");
    assert_eq!(
        fresh.attribution().unwrap(),
        replaced.attribution().unwrap()
    );
}

#[test]
fn test_a_replace_replays_only_the_episode_since_the_last_reset() {
    let rows = crossing_rows(120);
    let mut plain = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    let mut replaced = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    for worker in [&mut plain, &mut replaced] {
        worker.reset(9).unwrap();
        run_steps(worker, 0, 30);
        worker.reset(4).unwrap();
        run_steps(worker, 0, 20);
    }
    replaced.replace_rows(rows).unwrap();
    assert_eq!(
        run_steps(&mut plain, 20, 90),
        run_steps(&mut replaced, 20, 90)
    );
    assert_eq!(
        plain.attribution().unwrap(),
        replaced.attribution().unwrap()
    );
}

#[test]
fn test_each_check_of_a_replace_refuses_on_its_own() {
    let rows = crossing_rows(120);
    let refusal = |result: Result<EnvInfo, BridgeError>| match result {
        Err(BridgeError::Refused(reason)) => reason,
        other => panic!("{other:?}"),
    };
    // A row already observed changes: the rows are refused before any replay.
    let mut worker = Worker::spawn(grid_bridge(), rows.clone(), None).unwrap();
    worker.reset(9).unwrap();
    run_steps(&mut worker, 0, 30);
    assert!(refusal(worker.replace_rows(shifted(&rows, 3, 1))).contains("already read"));
    // With observations late by more than a row, the kernel has matched the row after the
    // one observed: changing it keeps the observed rows, and only the replay tells.
    let mut late = grid_bridge();
    late.latency = LatencyConfig {
        quantum_ns: 1_000_000,
        observation_q: 40,
        submit_q: 1,
        cancel_q: 1,
        ack_q: 1,
        jitter_q: 0,
    };
    let mut plain = Worker::spawn(late.clone(), rows.clone(), None).unwrap();
    plain.reset(9).unwrap();
    run_steps(&mut plain, 0, 30);
    let mut worker = Worker::spawn(late, rows.clone(), None).unwrap();
    worker.reset(9).unwrap();
    let observed = run_steps(&mut worker, 0, 30)
        .last()
        .unwrap()
        .transition
        .observation
        .row
        .unwrap() as usize;
    let reason = refusal(worker.replace_rows(shifted(&rows, observed + 1, 5)));
    // The replayed steps tell first: a step's fill or reward moved.
    assert!(reason.contains("replayed step"), "{reason}");
    // The episode goes on, unchanged, on the old rows.
    assert_eq!(worker.info(), plain.info());
    assert_eq!(
        run_steps(&mut plain, 30, 80),
        run_steps(&mut worker, 30, 80)
    );
    assert_eq!(plain.attribution().unwrap(), worker.attribution().unwrap());
}

#[test]
fn test_a_replace_after_a_proposal_with_a_nan_replays_it() {
    let bridge = residual_config(AVELLANEDA);
    let rows = crossing_rows(120);
    let nan = ResidualProposal {
        bid_away: f64::NAN,
        ask_away: 0.0,
        bid_size: 0.0,
        ask_size: 0.0,
    };
    let mut plain = Worker::spawn(bridge.clone(), rows.clone(), None).unwrap();
    let mut replaced = Worker::spawn(bridge, rows.clone(), None).unwrap();
    for worker in [&mut plain, &mut replaced] {
        worker.reset(1).unwrap();
        let reply = worker.step_residual(nan).unwrap();
        assert!(
            reply.decision.is_some(),
            "the NaN falls back, and is kept in the reply"
        );
        worker.step(RlAction::Hold).unwrap();
    }
    // The same rows: the replay reaches the step with the NaN.
    replaced.replace_rows(rows.clone()).unwrap();
    // A refused replace: the episode is rebuilt and replayed on the old rows, NaN included.
    assert!(matches!(
        replaced.replace_rows(shifted(&rows, 0, 1)),
        Err(BridgeError::Refused(_))
    ));
    let mut bad = rows.clone();
    bad[100].bid = 0;
    assert!(matches!(
        replaced.replace_rows(bad),
        Err(BridgeError::Refused(_))
    ));
    for step in 0..20 {
        assert_eq!(
            plain.step(RlAction::Hold).unwrap(),
            replaced.step(RlAction::Hold).unwrap(),
            "step {step}"
        );
    }
    assert_eq!(
        plain.attribution().unwrap(),
        replaced.attribution().unwrap()
    );
}
