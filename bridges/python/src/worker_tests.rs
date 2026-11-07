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
    ACTION_SPACE_VERSION, EPISODE_VERSION, EpisodeRecorder, HEDGED_ACTION_SPACE_VERSION,
    HEDGED_OBSERVATION_SPACE_VERSION, HEDGED_REWARD_VERSION, OBSERVATION_SPACE_VERSION,
    REWARD_VERSION, ResidualBounds, ResidualEnv, ResidualProposal, RlAction, RlEnv, SpecId,
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
    assert!(known_versions(&id));
    for change in 0..4 {
        let mut later = id;
        match change {
            0 => later.observation += 1,
            1 => later.action += 1,
            2 => later.reward += 1,
            _ => later.episode += 1,
        }
        assert!(!known_versions(&later));
    }
    // An environment with a hedge lane in a second instrument is not one this bridge drives.
    let hedged = SpecId {
        observation: HEDGED_OBSERVATION_SPACE_VERSION,
        action: HEDGED_ACTION_SPACE_VERSION,
        reward: HEDGED_REWARD_VERSION,
        ..id
    };
    assert!(!known_versions(&hedged));
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
        residual.step(RlAction::Hold),
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
    assert!(known_versions(&p) && known_versions(&h));
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
