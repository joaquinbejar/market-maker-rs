//! The spaces of the `rl` module: hand computed trajectories, causality, masks and
//! the agreement of the reward with the evaluation metrics. All data here is synthetic.

#![cfg(feature = "rl")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::evaluation::{
    Direction, EvaluationInput, MetricHorizon, MetricId, MetricUnit, attribute,
};
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionStatus, FinancialKernel, GateCheck, GateVerdict, KernelAction, KernelConfig,
    KernelDataset, KernelEvent, KernelFillModel, KernelLatency, KernelMargin, QuoteSide,
    RejectReason, encode,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    ACTION_SPACE_VERSION, AsOf, CANCEL_ACTION_SPACE_VERSION, CancelActions, EPISODE_VERSION,
    ERR_ACTION_OFF_GRID, ERR_CANCEL_REFUSED, ERR_CROSS_HEDGE_REFUSED, ERR_EXPIRY_REFUSED,
    ERR_HEDGE_REFUSED, EXPIRY_ACTION_SPACE_VERSION, EpisodeRecorder, FEATURE_COUNT, FeatureHorizon,
    FeatureId, FeatureUnit, HedgeActions, InvalidActionPolicy, MAX_QUOTE_LIFE_NS, MAX_QUOTE_LIVES,
    OBSERVATION_SPACE_VERSION, ObservationScales, QuoteExpiry, QuoteLives, REWARD_VERSION,
    RlAction, RlConfig, RlEnv, RlSpec, StateMemory, TerminalValuation,
};
use market_maker_rs::types::error::MMError;

fn scales() -> ObservationScales {
    ObservationScales {
        spread_ticks: dec!(10),
        mid_change_ticks: dec!(4),
        row_gap_ms: dec!(4000),
        size_lots: dec!(10),
        position: dec!(4),
        entry_gap: dec!(2),
    }
}

fn config() -> RlConfig {
    RlConfig::new(scales(), dec!(1))
}

fn tick(ts: u64, bid: Decimal, bid_size: Decimal, ask: Decimal, ask_size: Decimal) -> MarketTick {
    MarketTick::new(ts, bid, bid_size, ask, ask_size)
}

fn quote(bid: usize, ask: usize, bid_size: usize, ask_size: usize) -> RlAction {
    RlAction::Quote {
        bid,
        ask,
        bid_size,
        ask_size,
    }
}

// First trajectory: tick 0.5, lot 1, fee 0.1 %, a long position closed at the touch.
//
//   row  time  bid    ask    mid
//   0    1000  99.5   100.5  100
//   1    2000  98.5   99.5   99
//   2    3000  100    101    100.5
//
// Decision on row 0: buy 2 at 99.5 (offset 1), sell 1 at 105 (offset 10). Row 1 crosses
// the bid: cash -199, fee 2 * 99.5 * 0.001 = 0.199, position 2.
//
//   step  completed row  net                         ledger change
//   0     0 (mid 100)    -199 - 0.199 + 2 * 100      =  0.801
//   1     1 (mid 99)     -199 - 0.199 + 2 * 99       = -2.0
//   2     2 (mid 100.5)  -199 - 0.199 + 2 * 100.5    =  3.0
//
// The close sells 2 at the last bid 100: 2 * (100 - 100.5) = -1, fee 2 * 100 * 0.001 =
// 0.2, so -1.2. The whole episode: bought at 99.5, sold at 100, 1 - 0.199 - 0.2 = 0.601.
fn long_config() -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_fee_rate(dec!(0.001))
}

fn long_ticks() -> Vec<MarketTick> {
    vec![
        tick(1000, dec!(99.5), dec!(5), dec!(100.5), dec!(5)),
        tick(2000, dec!(98.5), dec!(5), dec!(99.5), dec!(5)),
        tick(3000, dec!(100), dec!(5), dec!(101), dec!(5)),
    ]
}

fn long_grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap()
}

#[test]
fn test_trajectory_long_with_fee_and_close_matches_hand_computation() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let rl = config().with_terminal(TerminalValuation::CloseAtTouch);
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();

    let first = env
        .step_with(&mut state, quote(1, 2, 1, 0), &mut recorder)
        .unwrap();
    assert_eq!(first.reward.ledger_change, dec!(0.801));
    assert_eq!(first.reward.terminal_close, dec!(0));
    assert_eq!(first.reward.total, dec!(0.801));
    assert_eq!(first.reward.as_f32, 0.801_f32);
    assert!(!first.terminated && !first.truncated);
    assert_eq!(first.close, None);
    env.reconcile(&state, &recorder).unwrap();

    let second = env
        .step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    assert_eq!(second.reward.ledger_change, dec!(-2));
    assert_eq!(second.reward.total, dec!(-2));
    assert!(!second.terminated && !second.truncated);
    env.reconcile(&state, &recorder).unwrap();

    let third = env
        .step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    assert_eq!(third.reward.ledger_change, dec!(3));
    assert_eq!(third.reward.terminal_close, dec!(-1.2));
    assert_eq!(third.reward.total, dec!(1.8));
    assert!(third.terminated && !third.truncated);
    let close = third.close.unwrap();
    assert_eq!(close.side, Action::Sell);
    assert_eq!(close.price, dec!(100));
    assert_eq!(close.quantity, dec!(2));
    assert_eq!(close.fee, dec!(0.2));
    assert_eq!(close.beyond_touch, dec!(0));
    assert_eq!(close.ts_ns, 3_000_000_000);

    assert_eq!(state.net(), dec!(0.601));
    assert_eq!(state.close(), Some(close));
    assert!(state.is_done());
    env.reconcile(&state, &recorder).unwrap();
}

// Second trajectory: tick 0.25, lot 0.5, a rebate of 0.2 %, a short position marked at the
// last mid.
//
//   row  time  bid  ask    mid
//   0    1000  50   51     50.5
//   1    2000  51   52     51.5
//   2    3000  50   50.75  50.375
//
// Decision on row 0: sell 1.5 at 50.75 (offset 1), and a far bid. Row 1 crosses the ask:
// cash +76.125, fee 1.5 * 50.75 * -0.002 = -0.15225, position -1.5. Decision on row 1:
// replace with a bid of 0.5 at 51 (offset 2) and a far ask. Row 2 crosses it: cash -25.5,
// fee 0.5 * 51 * -0.002 = -0.051, position -1. Cash 50.625, fees -0.20325.
//
//   step  completed row    net                                 ledger change
//   0     0 (mid 50.5)     76.125 + 0.15225 - 1.5 * 50.5       =  0.52725
//   1     1 (mid 51.5)     50.625 + 0.20325 - 1 * 51.5         = -1.199
//   2     2 (mid 50.375)   50.625 + 0.20325 - 1 * 50.375       =  1.125
fn short_config() -> KernelConfig {
    KernelConfig::new(dec!(0.25), dec!(0.5), dec!(1000), dec!(10), 100).with_fee_rate(dec!(-0.002))
}

fn short_ticks() -> Vec<MarketTick> {
    vec![
        tick(1000, dec!(50), dec!(5), dec!(51), dec!(5)),
        tick(2000, dec!(51), dec!(5), dec!(52), dec!(5)),
        tick(3000, dec!(50), dec!(5), dec!(50.75), dec!(5)),
    ]
}

#[test]
fn test_trajectory_short_with_rebate_and_mark_matches_hand_computation() {
    let cfg = short_config();
    let data = KernelDataset::from_ticks(&short_ticks(), &cfg).unwrap();
    let grid = ActionGrid::new([2, 8], [1, 8], [1, 3]).unwrap();
    let env: RlEnv<'_, 2, 2> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();

    let first = env
        .step_with(&mut state, quote(1, 0, 0, 1), &mut recorder)
        .unwrap();
    assert_eq!(first.reward.ledger_change, dec!(0.52725));
    env.reconcile(&state, &recorder).unwrap();

    let second = env
        .step_with(&mut state, quote(0, 1, 0, 0), &mut recorder)
        .unwrap();
    assert_eq!(second.reward.ledger_change, dec!(-1.199));
    env.reconcile(&state, &recorder).unwrap();

    let third = env
        .step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    assert_eq!(third.reward.ledger_change, dec!(1.125));
    assert_eq!(third.reward.terminal_close, dec!(0));
    assert!(third.terminated);
    assert_eq!(third.close, None);
    assert_eq!(state.close(), None);
    assert_eq!(state.net(), dec!(0.45325));
    env.reconcile(&state, &recorder).unwrap();
}

#[test]
fn test_close_short_buys_at_the_ask_and_reports_the_size_beyond_the_touch() {
    // Sell 3 at 100.5 on row 1, then close at the ask 101 of row 2, where only 1 shows.
    let cfg = long_config();
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(5), dec!(100.5), dec!(5)),
        tick(2000, dec!(100.5), dec!(5), dec!(101.5), dec!(5)),
        tick(3000, dec!(100), dec!(5), dec!(101), dec!(1)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let grid = ActionGrid::new([1, 10], [1, 10], [1, 3]).unwrap();
    let rl = config().with_terminal(TerminalValuation::CloseAtTouch);
    let env: RlEnv<'_, 2, 2> = RlEnv::new(&cfg, &data, grid, rl).unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    env.step_with(&mut state, quote(1, 0, 0, 1), &mut recorder)
        .unwrap();
    env.step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    let last = env
        .step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    let close = last.close.unwrap();
    assert_eq!(close.side, Action::Buy);
    assert_eq!(close.price, dec!(101));
    assert_eq!(close.quantity, dec!(3));
    assert_eq!(close.fee, dec!(0.303));
    assert_eq!(close.beyond_touch, dec!(2));
    // -3 * (101 - 100.5) - 0.303, against the last mid.
    assert_eq!(last.reward.terminal_close, dec!(-1.803));
    // Sold 3 at 100.5, bought 3 at 101, with fees of 0.3015 and 0.303.
    assert_eq!(state.net(), dec!(-2.1045));
    env.reconcile(&state, &recorder).unwrap();
}

#[test]
fn test_close_flat_inventory_closes_nothing() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let rl = config().with_terminal(TerminalValuation::CloseAtTouch);
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
    let mut state = env.reset(1).unwrap();
    let mut last = env.step(&mut state, RlAction::Hold).unwrap();
    while !state.is_done() {
        last = env.step(&mut state, RlAction::Hold).unwrap();
    }
    assert!(last.terminated);
    assert_eq!(last.close, None);
    assert_eq!(last.reward.terminal_close, dec!(0));
    assert_eq!(state.net(), dec!(0));
}

#[test]
fn test_truncation_closes_nothing_and_leaves_the_position_marked() {
    // The first trajectory cut after two decisions: row 1 has completed, and the position
    // of 2 stays open at its mid of 99, as in that trajectory's net of -1.199.
    let cfg =
        KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 2).with_fee_rate(dec!(0.001));
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let rl = config().with_terminal(TerminalValuation::CloseAtTouch);
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    env.step_with(&mut state, quote(1, 2, 1, 0), &mut recorder)
        .unwrap();
    let cut = env
        .step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    assert!(cut.truncated && !cut.terminated);
    assert_eq!(cut.close, None);
    assert_eq!(cut.reward.terminal_close, dec!(0));
    assert_eq!(cut.reward.ledger_change, dec!(-2));
    assert_eq!(state.net(), dec!(-1.199));
    assert_eq!(
        env.kernel().venue_summary(state.kernel()).unwrap().position,
        dec!(2)
    );
    assert!(state.is_done());
    assert!(env.step(&mut state, RlAction::Hold).is_err());
    env.reconcile(&state, &recorder).unwrap();
}

#[test]
fn test_truncation_with_a_fill_past_the_last_mark_reconciles() {
    // Jitter can stop a run at a decision after a row was matched and before it completed.
    // That fill has no mark of its own, and the kernel's books value it at the fill price:
    // the reward must value it at the last completed row, as the attribution does.
    let ticks: Vec<MarketTick> = (0..40_u64)
        .map(|i| {
            if i % 2 == 0 {
                tick(1000 + i * 8, dec!(100), dec!(4), dec!(101), dec!(4))
            } else {
                tick(1000 + i * 8, dec!(99), dec!(4), dec!(100), dec!(4))
            }
        })
        .collect();
    let grid = ActionGrid::new([1, 10], [10, 11], [1, 2]).unwrap();
    let mut past_the_mark = 0;
    for max_steps in 2..14 {
        for seed in 0..60 {
            let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), max_steps)
                .with_fee_rate(dec!(0.001))
                .with_latency(KernelLatency::zero(1_000_000).with_jitter(3));
            let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
            let env: RlEnv<'_, 2, 2> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
            let mut recorder = EpisodeRecorder::default();
            let mut state = env.reset_with(seed, &mut recorder).unwrap();
            while !state.is_done() {
                env.step_with(&mut state, quote(0, 0, 1, 0), &mut recorder)
                    .unwrap();
                env.reconcile(&state, &recorder).unwrap();
            }
            let records = env.records(&state, &recorder).unwrap();
            let last_mark = records.marks.last().unwrap().timestamp_ms;
            if records
                .fills
                .last()
                .is_some_and(|f| f.timestamp_ms > last_mark)
            {
                past_the_mark += 1;
            }
        }
    }
    assert!(
        past_the_mark > 0,
        "no run stopped between a fill and its mark"
    );
}

#[test]
fn test_last_decision_on_the_last_row_terminates_rather_than_truncates() {
    let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 3);
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, RlAction::Hold).unwrap();
    env.step(&mut state, RlAction::Hold).unwrap();
    let last = env.step(&mut state, RlAction::Hold).unwrap();
    assert!(last.terminated && !last.truncated);
}

#[test]
fn test_reward_matches_net_pnl_of_the_evaluation_at_every_decision() {
    let latencies = [
        KernelLatency::zero(1_000_000),
        KernelLatency::zero(1_000_000).with_delays(3, 2, 2, 4),
        KernelLatency::zero(1_000_000)
            .with_delays(1, 1, 1, 1)
            .with_jitter(2),
    ];
    let ticks: Vec<MarketTick> = (0_u32..200)
        .map(|i| {
            // A deterministic walk on the 0.5 grid.
            let step = (i64::from(i) * 7) % 13 - 6;
            let bid = dec!(100) + Decimal::from(step) * dec!(0.5);
            tick(
                1000 + u64::from(i) * 25,
                bid,
                dec!(5),
                bid + dec!(1),
                dec!(3),
            )
        })
        .collect();
    for latency in latencies {
        for terminal in [
            TerminalValuation::MarkAtLastMid,
            TerminalValuation::CloseAtTouch,
        ] {
            let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000)
                .with_fee_rate(dec!(0.001))
                .with_latency(latency);
            let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
            let rl = config().with_terminal(terminal);
            let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
            let mut recorder = EpisodeRecorder::default();
            let mut state = env.reset_with(9, &mut recorder).unwrap();
            let mut sum = dec!(0);
            let mut step = 0_usize;
            while !state.is_done() {
                let action = match step % 4 {
                    0 => quote(0, 0, 1, 0),
                    1 => RlAction::Hold,
                    2 => quote(1, 1, 0, 1),
                    _ => quote(0, 2, 0, 0),
                };
                let t = env.step_with(&mut state, action, &mut recorder).unwrap();
                sum += t.reward.total;
                step += 1;
                env.reconcile(&state, &recorder).unwrap();
                let records = env.records(&state, &recorder).unwrap();
                let input =
                    EvaluationInput::new(records.window, &records.marks, &records.fills, dec!(1))
                        .unwrap();
                assert_eq!(attribute(&input).unwrap().total.net_pnl, sum);
            }
            assert_eq!(sum, state.net());
            assert!(!env.records(&state, &recorder).unwrap().fills.is_empty());
        }
    }
}

#[test]
fn test_reconcile_catches_a_ledger_that_differs_from_the_records() {
    // A recorder that missed a fill must not pass.
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    let mut blind = EpisodeRecorder::default();
    let _ = env.reset_with(1, &mut blind).unwrap();
    env.step_with(&mut state, quote(1, 2, 1, 0), &mut recorder)
        .unwrap();
    env.step_with(&mut state, RlAction::Hold, &mut recorder)
        .unwrap();
    env.reconcile(&state, &recorder).unwrap();
    assert!(env.reconcile(&state, &blind).is_err());
}

#[test]
fn test_reward_is_stated_as_the_net_pnl_metric() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let reward = env.spec().reward;
    assert_eq!(reward.metric, MetricId::NetPnl);
    let metric = reward.metric.definition();
    assert_eq!(reward.unit, metric.unit);
    assert_eq!(metric.direction, Direction::PositiveIsGain);
    assert_eq!(metric.horizon, MetricHorizon::Window);
    assert!(!reward.policy_input);
}

#[test]
fn test_f32_reward_is_derived_and_the_decimal_sum_stays_exact() {
    // Rewards of 0.801, -2 and 1.8 in a unit of 4.
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let rl = RlConfig::new(scales(), dec!(4)).with_terminal(TerminalValuation::CloseAtTouch);
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
    let mut state = env.reset(1).unwrap();
    let first = env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    assert_eq!(first.reward.as_f32, 0.20025_f32);
    assert_eq!(first.reward.total, dec!(0.801));
    let second = env.step(&mut state, RlAction::Hold).unwrap();
    assert_eq!(second.reward.as_f32, -0.5_f32);
    let third = env.step(&mut state, RlAction::Hold).unwrap();
    assert_eq!(third.reward.as_f32, 0.45_f32);
    // The unit changes the f32 and nothing exact.
    assert_eq!(state.net(), dec!(0.601));
}

#[test]
fn test_features_at_two_decisions_match_hand_computation() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();

    let at = |v: &market_maker_rs::rl::FeatureVector, id: FeatureId| {
        (v.raw[id.index()], v.values[id.index()])
    };
    let first = env.observe(&state).unwrap();
    assert_eq!(first.row, Some(0));
    assert_eq!(first.ts_ns, 1_000_000_000);
    assert_eq!(at(&first, FeatureId::Spread), (Some(dec!(2)), 0.2));
    assert_eq!(at(&first, FeatureId::MidChange), (None, 0.0));
    assert_eq!(at(&first, FeatureId::RowGap), (None, 0.0));
    assert_eq!(at(&first, FeatureId::BidSize), (Some(dec!(5)), 0.5));
    assert_eq!(at(&first, FeatureId::AskSize), (Some(dec!(5)), 0.5));
    assert_eq!(at(&first, FeatureId::Imbalance), (Some(dec!(0)), 0.0));
    assert_eq!(at(&first, FeatureId::Position), (Some(dec!(0)), 0.0));
    assert_eq!(at(&first, FeatureId::EntryGap), (None, 0.0));
    assert_eq!(at(&first, FeatureId::OpenOrders), (Some(dec!(0)), 0.0));
    assert_eq!(at(&first, FeatureId::AwaitingCancel), (Some(dec!(0)), 0.0));
    assert_eq!(at(&first, FeatureId::StepsLeft), (Some(dec!(100)), 1.0));

    let second = env.step(&mut state, quote(1, 2, 1, 0)).unwrap().observation;
    assert_eq!(second, env.observe(&state).unwrap());
    assert_eq!(second.row, Some(1));
    // The mid fell from 100 to 99, which is 2 ticks of 0.5.
    assert_eq!(at(&second, FeatureId::MidChange), (Some(dec!(-2)), -0.5));
    assert_eq!(at(&second, FeatureId::RowGap), (Some(dec!(1000)), 0.25));
    assert_eq!(at(&second, FeatureId::Position), (Some(dec!(2)), 0.5));
    // 99 - 99.5.
    assert_eq!(at(&second, FeatureId::EntryGap), (Some(dec!(-0.5)), -0.25));
    // The ask rests; with 16 as the capacity.
    assert_eq!(at(&second, FeatureId::OpenOrders), (Some(dec!(1)), 0.0625));
    assert_eq!(at(&second, FeatureId::StepsLeft), (Some(dec!(99)), 0.99));
}

#[test]
fn test_features_count_an_order_waiting_for_the_outcome_of_a_cancel() {
    // Cancels take longer than the data. Replacing the first quote leaves its cancel in
    // flight, and when row 2 fills that order, its report says a cancel is still pending.
    let latency = KernelLatency::zero(1_000_000).with_delays(0, 0, 100_000, 0);
    let cfg =
        KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_latency(latency);
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
        tick(1001, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
        tick(1002, dec!(98), dec!(1), dec!(99), dec!(1)),
        tick(1003, dec!(98), dec!(1), dec!(99), dec!(1)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 0, 0)).unwrap();
    env.step(&mut state, quote(2, 2, 0, 0)).unwrap();
    let observed = env.observe(&state).unwrap();
    let raw = |id: FeatureId| observed.raw[id.index()];
    assert_eq!(raw(FeatureId::Position), Some(dec!(1)));
    assert_eq!(raw(FeatureId::AwaitingCancel), Some(dec!(1)));
    assert_eq!(raw(FeatureId::OpenOrders), Some(dec!(4)));
}

#[test]
fn test_features_keep_the_sign_of_a_short_position() {
    // The second trajectory: short 1.5 entered at 50.75, with the mid up from 50.5 to 51.5,
    // which is 4 ticks of 0.25.
    let cfg = short_config();
    let data = KernelDataset::from_ticks(&short_ticks(), &cfg).unwrap();
    let grid = ActionGrid::new([2, 8], [1, 8], [1, 3]).unwrap();
    let env: RlEnv<'_, 2, 2> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let observed = env.step(&mut state, quote(1, 0, 0, 1)).unwrap().observation;
    let at = |id: FeatureId| (observed.raw[id.index()], observed.values[id.index()]);
    assert_eq!(at(FeatureId::Position), (Some(dec!(-1.5)), -0.375));
    // The gap is the mid less the entry whichever way the position runs, and here that is
    // a loss: the sign of the position and of the gap together say so.
    assert_eq!(at(FeatureId::EntryGap), (Some(dec!(0.75)), 0.375));
    assert_eq!(at(FeatureId::MidChange), (Some(dec!(4)), 1.0));
}

#[test]
fn test_features_keep_the_sign_of_each_direction() {
    // A mid that falls, an entry above the mid, and a book heavier on the ask.
    let cfg = long_config();
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(1), dec!(100.5), dec!(3)),
        tick(2000, dec!(98.5), dec!(1), dec!(99.5), dec!(3)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let first = env.observe(&state).unwrap();
    assert_eq!(first.raw[FeatureId::Imbalance.index()], Some(dec!(-0.5)));
    assert_eq!(first.values[FeatureId::Imbalance.index()], -0.5);
    let second = env.step(&mut state, quote(1, 2, 1, 0)).unwrap().observation;
    let at = |id: FeatureId| (second.raw[id.index()], second.values[id.index()]);
    assert_eq!(at(FeatureId::Position), (Some(dec!(2)), 0.5));
    assert_eq!(at(FeatureId::EntryGap), (Some(dec!(-0.5)), -0.25));
    assert_eq!(at(FeatureId::MidChange), (Some(dec!(-2)), -0.5));
}

#[test]
fn test_features_clip_the_normalized_value_and_keep_the_raw_one() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let mut small = scales();
    small.spread_ticks = dec!(1);
    small.mid_change_ticks = dec!(1);
    let env: RlEnv<'_, 3, 2> =
        RlEnv::new(&cfg, &data, long_grid(), RlConfig::new(small, dec!(1))).unwrap();
    let mut state = env.reset(1).unwrap();
    let first = env.observe(&state).unwrap();
    assert_eq!(first.raw[FeatureId::Spread.index()], Some(dec!(2)));
    assert_eq!(first.values[FeatureId::Spread.index()], 1.0);
    let second = env.step(&mut state, RlAction::Hold).unwrap().observation;
    assert_eq!(second.raw[FeatureId::MidChange.index()], Some(dec!(-2)));
    assert_eq!(second.values[FeatureId::MidChange.index()], -1.0);
}

#[test]
fn test_features_imbalance_uses_both_sides_and_has_no_value_with_an_empty_book() {
    let cfg = long_config();
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(3), dec!(100.5), dec!(1)),
        tick(1001, dec!(99.5), dec!(0), dec!(100.5), dec!(0)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let i = FeatureId::Imbalance.index();
    let first = env.observe(&state).unwrap();
    assert_eq!(first.raw[i], Some(dec!(0.5)));
    assert_eq!(first.values[i], 0.5);
    assert_eq!(first.raw[FeatureId::BidSize.index()], Some(dec!(3)));
    assert_eq!(first.raw[FeatureId::AskSize.index()], Some(dec!(1)));
    let second = env.step(&mut state, RlAction::Hold).unwrap().observation;
    assert_eq!(second.raw[i], None);
    assert_eq!(second.values[i], 0.0);
}

#[test]
fn test_missing_data_a_gap_shows_in_the_row_gap_feature() {
    let cfg = long_config();
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
        tick(61_000, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let after = env.step(&mut state, RlAction::Hold).unwrap().observation;
    let i = FeatureId::RowGap.index();
    assert_eq!(after.raw[i], Some(dec!(60000)));
    assert_eq!(after.values[i], 1.0);
}

#[test]
fn test_missing_data_no_rows_ends_at_reset_with_no_row_features() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&[], &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    assert!(state.is_done());
    let observed = env.observe(&state).unwrap();
    assert_eq!(observed.row, None);
    for id in [
        FeatureId::Spread,
        FeatureId::MidChange,
        FeatureId::RowGap,
        FeatureId::BidSize,
        FeatureId::AskSize,
        FeatureId::Imbalance,
        FeatureId::EntryGap,
    ] {
        assert_eq!(observed.raw[id.index()], None, "{id:?}");
    }
    assert_eq!(observed.raw[FeatureId::Position.index()], Some(dec!(0)));
    assert!(!env.mask(&state).global_quote_ok);
    assert!(env.step(&mut state, RlAction::Hold).is_err());
    assert_eq!(state.net(), dec!(0));
    let recorder = EpisodeRecorder::default();
    assert!(env.reconcile(&state, &recorder).is_err());
}

fn walk(rows: u32, spacing: u64) -> Vec<MarketTick> {
    (0..rows)
        .map(|i| {
            let step = (i64::from(i) * 5) % 9 - 4;
            let bid = dec!(100) + Decimal::from(step) * dec!(0.5);
            tick(
                1000 + u64::from(i) * spacing,
                bid,
                dec!(4),
                bid + dec!(1),
                dec!(2),
            )
        })
        .collect()
}

#[test]
fn test_no_future_observation_is_identical_until_the_clock_reaches_a_changed_row() {
    let prefix = 30_usize;
    let base = walk(60, 25);
    // Other prices and sizes from row 30 on, and a different length.
    let mut other = base[..prefix].to_vec();
    other.extend((0..50_u64).map(|i| {
        tick(
            1000 + 25 * (prefix as u64 + i),
            dec!(150),
            dec!(9),
            dec!(153),
            dec!(7),
        )
    }));
    let reach = 1000 + 25 * prefix as u64;
    let latencies = [
        KernelLatency::zero(1_000_000),
        KernelLatency::zero(1_000_000).with_delays(3, 1, 1, 1),
        KernelLatency::zero(1_000_000).with_delays(0, 5, 2, 7),
    ];
    for latency in latencies {
        let cfg =
            KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000).with_latency(latency);
        let a = KernelDataset::from_ticks(&base, &cfg).unwrap();
        let b = KernelDataset::from_ticks(&other, &cfg).unwrap();
        let ea: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &a, long_grid(), config()).unwrap();
        let eb: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &b, long_grid(), config()).unwrap();
        let (mut sa, mut sb) = (ea.reset(3).unwrap(), eb.reset(3).unwrap());
        let mut compared = 0;
        let mut differs_later = false;
        let mut step = 0_usize;
        while !sa.is_done() && !sb.is_done() {
            let (oa, ob) = (ea.observe(&sa).unwrap(), eb.observe(&sb).unwrap());
            let clock = sa.kernel().now_ns().max(sb.kernel().now_ns());
            if clock < reach * 1_000_000 {
                assert_eq!(oa, ob, "step {step}");
                assert_eq!(ea.mask(&sa), eb.mask(&sb), "step {step}");
                compared += 1;
            } else if oa != ob {
                differs_later = true;
            }
            let action = if step.is_multiple_of(3) {
                quote(0, 0, 0, 0)
            } else {
                RlAction::Hold
            };
            ea.step(&mut sa, action).unwrap();
            eb.step(&mut sb, action).unwrap();
            step += 1;
        }
        assert!(compared >= 20, "only {compared} decisions compared");
        assert!(differs_later, "the changed rows must show eventually");
    }
}

#[test]
fn test_no_latent_undelivered_fills_change_the_venue_and_not_the_observation() {
    // Acknowledgements take longer than the data, so nothing is delivered while it runs.
    // The two runs differ only in fills the agent has not been told of.
    let ticks: Vec<MarketTick> = (0..6_u64)
        .map(|i| {
            let (bid, ask) = if i % 2 == 0 {
                (dec!(100), dec!(101))
            } else {
                (dec!(99), dec!(100))
            };
            tick(1000 + i, bid, dec!(4), ask, dec!(4))
        })
        .collect();
    let latency = KernelLatency::zero(1_000_000).with_delays(0, 0, 0, 100_000);
    let immediate =
        KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 3).with_latency(latency);
    let queue = immediate
        .clone()
        .with_fill_model(KernelFillModel::QueuePosition {
            rate: dec!(1),
            min_queue_ms: 100_000,
        });
    let a = KernelDataset::from_ticks(&ticks, &immediate).unwrap();
    let b = KernelDataset::from_ticks(&ticks, &queue).unwrap();
    let ea: RlEnv<'_, 3, 2, 16, 64, 256, 64> =
        RlEnv::new(&immediate, &a, long_grid(), config()).unwrap();
    let eb: RlEnv<'_, 3, 2, 16, 64, 256, 64> =
        RlEnv::new(&queue, &b, long_grid(), config()).unwrap();
    let (mut sa, mut sb) = (ea.reset(3).unwrap(), eb.reset(3).unwrap());
    ea.step(&mut sa, quote(1, 2, 0, 0)).unwrap();
    eb.step(&mut sb, quote(1, 2, 0, 0)).unwrap();
    let ta = ea.step(&mut sa, RlAction::Hold).unwrap();
    let tb = eb.step(&mut sb, RlAction::Hold).unwrap();
    assert_ne!(sa.kernel().venue_position(), sb.kernel().venue_position());
    assert_ne!(ta.reward.total, tb.reward.total);
    assert_eq!(ta.observation, tb.observation);
    assert_eq!(ta.mask, tb.mask);
    assert_eq!(
        ta.observation.raw[FeatureId::Position.index()],
        Some(dec!(0))
    );
}

fn limited(max_position: Decimal) -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000)
        .with_limits(RiskLimits::new(max_position, dec!(1000000), dec!(1)).unwrap())
}

fn sweeping() -> Vec<MarketTick> {
    // The bid at the mid minus one tick is taken on every odd row, and the ask never,
    // so a position grows without a limit.
    (0..20_u64)
        .map(|i| {
            if i % 2 == 0 {
                tick(1000 + i, dec!(100), dec!(4), dec!(101), dec!(4))
            } else {
                tick(1000 + i, dec!(99), dec!(4), dec!(100), dec!(4))
            }
        })
        .collect()
}

#[test]
fn test_mask_reads_no_limit_and_the_gate_binds_under_either_policy() {
    let free = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000);
    let capped = limited(dec!(1));
    let ticks = sweeping();
    let free_data = KernelDataset::from_ticks(&ticks, &free).unwrap();
    let capped_data = KernelDataset::from_ticks(&ticks, &capped).unwrap();
    let grid = ActionGrid::new([1, 10], [10, 11], [1, 2]).unwrap();
    let action = quote(0, 0, 1, 0);

    let free_env: RlEnv<'_, 2, 2> = RlEnv::new(&free, &free_data, grid, config()).unwrap();
    let mut free_state = free_env.reset(1).unwrap();
    let mask_env: RlEnv<'_, 2, 2> = RlEnv::new(&capped, &capped_data, grid, config()).unwrap();
    let reject = config().with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let reject_env: RlEnv<'_, 2, 2> = RlEnv::new(&capped, &capped_data, grid, reject).unwrap();
    let mut mask_state = mask_env.reset(1).unwrap();
    let mut reject_state = reject_env.reset(1).unwrap();

    let mut refused_by_gate = false;
    while !mask_state.is_done() {
        // The same action is inside the mask with and without limits, at a position
        // the gate will not let grow.
        assert_eq!(mask_env.mask(&mask_state), free_env.mask(&free_state));
        assert!(mask_env.mask(&mask_state).allows(0, 0, 1, 0));
        let a = mask_env.step(&mut mask_state, action).unwrap();
        let b = reject_env.step(&mut reject_state, action).unwrap();
        free_env.step(&mut free_state, action).unwrap();
        assert_eq!(a.action_status, ActionStatus::Applied);
        assert_eq!(a.reward, b.reward);
        assert_eq!(
            mask_state.kernel().venue_position(),
            reject_state.kernel().venue_position()
        );
        assert!(mask_state.kernel().venue_position().abs() <= dec!(1));
        refused_by_gate |= free_state.kernel().venue_position().abs() > dec!(1);
    }
    assert!(
        refused_by_gate,
        "without limits the position must pass the cap"
    );
    assert_eq!(mask_state.kernel().venue_position(), dec!(1));
}

#[test]
fn test_mask_policy_refuses_an_oversized_order_and_changes_nothing() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let grid = ActionGrid::new([1, 2, 10], [1, 2, 10], [1, 11]).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let mask = env.mask(&state);
    assert!(mask.global_quote_ok);
    assert_eq!(mask.size, 0b01);
    let before = state;
    let error = env.step(&mut state, quote(2, 2, 1, 0)).unwrap_err();
    assert!(matches!(error, MMError::InvalidQuoteGeneration(_)));
    assert_eq!(state, before);
    // A hold stays inside the mask.
    env.step(&mut state, RlAction::Hold).unwrap();
}

#[test]
fn test_reject_policy_takes_an_oversized_order_as_a_hold_with_the_penalty_apart() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let grid = ActionGrid::new([1, 2, 10], [1, 2, 10], [1, 11]).unwrap();
    let rl = config().with_invalid_action(InvalidActionPolicy::Reject, dec!(0.25));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, rl).unwrap();
    let mut state = env.reset(1).unwrap();
    let t = env.step(&mut state, quote(2, 2, 1, 0)).unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::Rejected(RejectReason::SizeRange)
    );
    assert_eq!(t.reward.penalty, dec!(0.25));
    assert_eq!(t.reward.ledger_change, dec!(0));
    assert_eq!(t.reward.total, dec!(-0.25));
    assert_eq!(t.reward.as_f32, -0.25);
    assert_eq!(state.penalties(), dec!(0.25));
    assert_eq!(state.kernel().venue_position(), dec!(0));
    // A penalty is not accounting.
    assert_eq!(state.net(), dec!(0));
    let held = env.step(&mut state, RlAction::Hold).unwrap();
    assert_eq!(held.reward.penalty, dec!(0));
    assert_eq!(state.penalties(), dec!(0.25));
}

#[test]
fn test_invalid_prices_and_sizes_follow_the_policy() {
    // Row 0 has a mid of 200 ticks. Offset 200 reaches a price of zero, offset 60000 has
    // no price at all, and 11 lots is above the largest order.
    let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100);
    let ticks = walk(10, 1);
    let first_mid2 = KernelDataset::from_ticks(&ticks, &cfg).unwrap().ticks()[0]
        .mid2()
        .unwrap();
    let on_zero = u16::try_from(first_mid2 / 2).unwrap();
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let grid = ActionGrid::new([1, on_zero, 60_000], [1, 2, 3], [1, 11]).unwrap();
    let cases = [
        (
            quote(1, 0, 0, 0),
            ActionStatus::Rejected(RejectReason::PriceRange),
        ),
        (
            quote(2, 0, 0, 0),
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            quote(0, 0, 1, 0),
            ActionStatus::Rejected(RejectReason::SizeRange),
        ),
        (
            quote(0, 0, 0, 1),
            ActionStatus::Rejected(RejectReason::SizeRange),
        ),
        (
            quote(3, 0, 0, 0),
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            quote(0, 3, 0, 0),
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            quote(0, 0, 2, 0),
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Bid,
                offset: 1,
                size: 0,
            },
            ActionStatus::Rejected(RejectReason::PriceRange),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Bid,
                offset: 2,
                size: 0,
            },
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Bid,
                offset: 3,
                size: 0,
            },
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Ask,
                offset: 3,
                size: 0,
            },
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Ask,
                offset: 0,
                size: 1,
            },
            ActionStatus::Rejected(RejectReason::SizeRange),
        ),
        (
            RlAction::OneSided {
                side: QuoteSide::Ask,
                offset: 0,
                size: 2,
            },
            ActionStatus::QuoteError(ERR_ACTION_OFF_GRID),
        ),
    ];
    let mask_env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
    let rl = config().with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let reject_env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, rl).unwrap();
    for (action, status) in cases {
        let mut state = mask_env.reset(1).unwrap();
        let before = state;
        assert!(mask_env.step(&mut state, action).is_err(), "{action:?}");
        assert_eq!(state, before, "{action:?}");
        let mut state = reject_env.reset(1).unwrap();
        let t = reject_env.step(&mut state, action).unwrap();
        assert_eq!(t.action_status, status, "{action:?}");
        assert_eq!(t.reward.penalty, dec!(1), "{action:?}");
    }
    // Inside the mask on each side, with no charge.
    let mut state = reject_env.reset(1).unwrap();
    for (side, offset) in [(QuoteSide::Bid, 0), (QuoteSide::Ask, 2)] {
        let action = RlAction::OneSided {
            side,
            offset,
            size: 0,
        };
        let t = reject_env.step(&mut state, action).unwrap();
        assert_eq!(t.action_status, ActionStatus::Applied, "{action:?}");
        assert_eq!(t.reward.penalty, dec!(0));
    }
}

#[test]
fn test_one_sided_prices_on_its_own_side_of_the_grid() {
    // Offsets differ per side: the bid at 2 ticks below the mid, the ask at 3 above.
    let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100);
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
        tick(1001, dec!(101.5), dec!(1), dec!(102.5), dec!(1)),
        tick(1002, dec!(99), dec!(1), dec!(99.5), dec!(1)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let grid = ActionGrid::new([2], [3], [1]).unwrap();
    let env: RlEnv<'_, 1, 1> = RlEnv::new(&cfg, &data, grid, config()).unwrap();
    let mut state = env.reset(1).unwrap();
    let ask = RlAction::OneSided {
        side: QuoteSide::Ask,
        offset: 0,
        size: 0,
    };
    env.step(&mut state, ask).unwrap();
    // Sold 1 at 100 + 1.5, on the second row's bid.
    assert_eq!(state.kernel().venue_realized(), dec!(101.5));
    let bid = RlAction::OneSided {
        side: QuoteSide::Bid,
        offset: 0,
        size: 0,
    };
    env.step(&mut state, bid).unwrap();
    // Bought 1 at 102 - 1, on the third row's ask.
    assert_eq!(state.kernel().venue_realized(), dec!(0.5));
}

#[test]
fn test_a_reward_that_does_not_convert_is_an_error_and_ends_the_episode() {
    // A step of -10 less a penalty at the last place of a Decimal has too many digits.
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let grid = ActionGrid::new([1, 2, 10], [1, 2, 10], [10, 11]).unwrap();
    let tiny = Decimal::new(1, 28);
    let rl = config().with_invalid_action(InvalidActionPolicy::Reject, tiny);
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, rl).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(0, 2, 0, 0)).unwrap();
    assert!(!state.is_done());
    let error = env.step(&mut state, quote(0, 2, 1, 0)).unwrap_err();
    assert!(matches!(error, MMError::NumericalError(_)));
    assert!(state.is_done());
    assert!(matches!(
        env.step(&mut state, RlAction::Hold),
        Err(MMError::InvalidMarketState(_))
    ));
}

#[test]
fn test_step_refuses_an_ended_episode_and_a_state_of_another_kernel() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    while !state.is_done() {
        env.step(&mut state, RlAction::Hold).unwrap();
    }
    let ended = state;
    assert!(matches!(
        env.step(&mut state, RlAction::Hold),
        Err(MMError::InvalidMarketState(_))
    ));
    assert_eq!(state, ended);

    let shorter = KernelDataset::from_ticks(&long_ticks()[..2], &cfg).unwrap();
    let other: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &shorter, long_grid(), config()).unwrap();
    let mut foreign = other.reset(1).unwrap();
    let copy = foreign;
    assert!(matches!(
        env.step(&mut foreign, RlAction::Hold),
        Err(MMError::InvalidMarketState(_))
    ));
    assert_eq!(foreign, copy);
    assert!(matches!(
        env.observe(&foreign),
        Err(MMError::InvalidMarketState(_))
    ));
}

#[test]
fn test_env_drives_the_kernel_as_the_kernel_would_be_driven() {
    let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000)
        .with_fee_rate(dec!(0.001))
        .with_latency(
            KernelLatency::zero(1_000_000)
                .with_delays(1, 1, 1, 1)
                .with_jitter(1),
        );
    let ticks = walk(80, 25);
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut state = env.reset(5).unwrap();
    let mut direct = kernel.reset(5);
    let grid = long_grid();
    let mut step = 0_usize;
    while !state.is_done() {
        let (action, kernel_action) = if step.is_multiple_of(2) {
            (
                quote(1, 1, 0, 1),
                kernel.grid_action(&direct, &grid, (1, 1, 0, 1)).unwrap(),
            )
        } else {
            (RlAction::Hold, KernelAction::Hold)
        };
        env.step(&mut state, action).unwrap();
        kernel.step(&mut direct, kernel_action);
        assert_eq!(encode(state.kernel()), encode(&direct));
        step += 1;
    }
}

#[test]
fn test_snapshot_replays_the_same_transitions() {
    let cfg = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 1000).with_latency(
        KernelLatency::zero(1_000_000)
            .with_delays(1, 1, 1, 1)
            .with_jitter(2),
    );
    let ticks = walk(40, 25);
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(11).unwrap();
    for _ in 0..5 {
        env.step(&mut state, quote(0, 0, 0, 0)).unwrap();
    }
    let snapshot = state;
    let first: Vec<_> = (0..10)
        .map(|_| env.step(&mut state, quote(1, 1, 0, 0)).unwrap())
        .collect();
    let mut replay = snapshot;
    let second: Vec<_> = (0..10)
        .map(|_| env.step(&mut replay, quote(1, 1, 0, 0)).unwrap())
        .collect();
    assert_eq!(first, second);
    assert_eq!(state, replay);
}

#[test]
fn test_new_refuses_invalid_choices() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let build = |rl: RlConfig| RlEnv::<'_, 3, 2>::new(&cfg, &data, long_grid(), rl).map(|_| ());
    build(config()).unwrap();
    type Set = fn(&mut ObservationScales, Decimal);
    let setters: [Set; 6] = [
        |s, v| s.spread_ticks = v,
        |s, v| s.mid_change_ticks = v,
        |s, v| s.row_gap_ms = v,
        |s, v| s.size_lots = v,
        |s, v| s.position = v,
        |s, v| s.entry_gap = v,
    ];
    for set in setters {
        for bad in [dec!(0), dec!(-1)] {
            let mut s = scales();
            set(&mut s, bad);
            assert!(matches!(
                build(RlConfig::new(s, dec!(1))),
                Err(MMError::InvalidConfiguration(_))
            ));
        }
    }
    for bad in [dec!(0), dec!(-1)] {
        assert!(build(RlConfig::new(scales(), bad)).is_err());
    }
    assert!(build(config().with_invalid_action(InvalidActionPolicy::Reject, dec!(-0.1))).is_err());
    build(config().with_invalid_action(InvalidActionPolicy::Reject, dec!(0))).unwrap();

    let no_steps = KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 0);
    let no_data = KernelDataset::from_ticks(&long_ticks(), &no_steps).unwrap();
    assert!(RlEnv::<'_, 3, 2>::new(&no_steps, &no_data, long_grid(), config()).is_err());
    let no_orders =
        RlEnv::<'_, 3, 2, 0, 64, 128, 64>::new(&cfg, &data, long_grid(), config()).map(|_| ());
    assert!(no_orders.is_err());
}

fn spec_of(
    cfg: &KernelConfig,
    grid: ActionGrid<3, 2>,
    rl: RlConfig,
) -> market_maker_rs::rl::RlSpec {
    let data = KernelDataset::from_ticks(&long_ticks(), cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(cfg, &data, grid, rl).unwrap();
    env.spec().clone()
}

#[test]
fn test_spec_states_the_table_of_features() {
    let spec = spec_of(&long_config(), long_grid(), config());
    assert_eq!(spec.observation.version, OBSERVATION_SPACE_VERSION);
    assert_eq!(spec.action.version, ACTION_SPACE_VERSION);
    assert_eq!(spec.reward.version, REWARD_VERSION);
    assert_eq!(spec.episode.version, EPISODE_VERSION);
    use AsOf as A;
    use FeatureHorizon as H;
    use FeatureId as F;
    use FeatureUnit as U;
    use StateMemory as M;
    let table = [
        (
            F::Spread,
            U::Ticks,
            A::ObservedRow,
            H::Instant,
            M::None,
            dec!(10),
        ),
        (
            F::MidChange,
            U::Ticks,
            A::ObservedRow,
            H::SincePreviousDecision,
            M::PreviousDecision,
            dec!(4),
        ),
        (
            F::RowGap,
            U::Milliseconds,
            A::ObservedRow,
            H::SincePreviousDecision,
            M::PreviousDecision,
            dec!(4000),
        ),
        (
            F::BidSize,
            U::Lots,
            A::ObservedRow,
            H::Instant,
            M::None,
            dec!(10),
        ),
        (
            F::AskSize,
            U::Lots,
            A::ObservedRow,
            H::Instant,
            M::None,
            dec!(10),
        ),
        (
            F::Imbalance,
            U::Ratio,
            A::ObservedRow,
            H::Instant,
            M::None,
            dec!(1),
        ),
        (
            F::Position,
            U::Base,
            A::DeliveredReports,
            H::EpisodeSoFar,
            M::None,
            dec!(4),
        ),
        (
            F::EntryGap,
            U::QuotePerBase,
            A::DeliveredReports,
            H::Instant,
            M::None,
            dec!(2),
        ),
        (
            F::OpenOrders,
            U::Count,
            A::AgentAccount,
            H::Instant,
            M::None,
            dec!(16),
        ),
        (
            F::AwaitingCancel,
            U::Count,
            A::AgentAccount,
            H::Instant,
            M::None,
            dec!(16),
        ),
        (
            F::StepsLeft,
            U::Count,
            A::DecisionCount,
            H::DecisionsAllowed,
            M::None,
            dec!(100),
        ),
    ];
    assert_eq!(table.len(), FEATURE_COUNT);
    for (i, row) in table.iter().enumerate() {
        let f = spec.observation.features[i];
        assert_eq!(FeatureId::ALL[i], row.0);
        assert_eq!(row.0.index(), i);
        assert_eq!(
            (f.id, f.unit, f.as_of, f.horizon, f.memory, f.divisor),
            *row
        );
    }
    let shape = spec.shape();
    assert_eq!(shape.features, FEATURE_COUNT);
    assert_eq!(shape.bid_offsets, 3);
    assert_eq!(shape.ask_offsets, 3);
    assert_eq!(shape.sizes, 2);
}

#[test]
fn test_spec_fingerprint_is_pinned_so_a_change_needs_a_new_version() {
    let spec = spec_of(&long_config(), long_grid(), config());
    let id = spec.id();
    assert_eq!(id.action, 2);
    assert_eq!(id.fingerprint, 165280442375121912034794721707712240181);
    assert_eq!(
        market_maker_rs::backtest::kernel::fnv1a128(&spec.to_bytes()),
        id.fingerprint
    );
    // What a checkpoint trained under action space 1 recorded for the same choices.
    let version_one = market_maker_rs::rl::SpecId {
        action: 1,
        fingerprint: 281796423096009772465355244436229638458,
        ..id
    };
    assert!(matches!(
        spec.ensure(version_one),
        Err(MMError::InvalidConfiguration(_))
    ));
    let hedged = spec_of(&long_config(), long_grid(), config().with_hedge_actions());
    assert!(hedged.ensure(version_one).is_err());
    assert!(hedged.ensure(id).is_err());
}

#[test]
fn test_spec_fingerprint_covers_every_choice() {
    let cfg = long_config();
    let base = spec_of(&cfg, long_grid(), config()).id().fingerprint;
    let mut seen = vec![base];
    let mut distinct = |spec: market_maker_rs::rl::RlSpec, what: &str| {
        let fp = spec.id().fingerprint;
        assert!(!seen.contains(&fp), "{what}");
        seen.push(fp);
    };
    let scale = |set: fn(&mut ObservationScales)| {
        let mut s = scales();
        set(&mut s);
        RlConfig::new(s, dec!(1))
    };
    let grid = |b: [u16; 3], a: [u16; 3], m: [u64; 2]| ActionGrid::new(b, a, m).unwrap();
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.spread_ticks = dec!(11))),
        "spread",
    );
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.mid_change_ticks = dec!(5))),
        "mid change",
    );
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.row_gap_ms = dec!(5))),
        "gap",
    );
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.size_lots = dec!(11))),
        "size",
    );
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.position = dec!(5))),
        "position",
    );
    distinct(
        spec_of(&cfg, long_grid(), scale(|s| s.entry_gap = dec!(5))),
        "entry",
    );
    distinct(
        spec_of(&cfg, grid([0, 1, 11], [0, 1, 10], [1, 2]), config()),
        "bid offsets",
    );
    distinct(
        spec_of(&cfg, grid([0, 1, 10], [0, 1, 11], [1, 2]), config()),
        "ask offsets",
    );
    distinct(
        spec_of(&cfg, grid([0, 1, 10], [0, 1, 10], [1, 3]), config()),
        "sizes",
    );
    distinct(
        spec_of(&cfg, long_grid(), RlConfig::new(scales(), dec!(2))),
        "f32 unit",
    );
    distinct(
        spec_of(
            &cfg,
            long_grid(),
            config().with_invalid_action(InvalidActionPolicy::Reject, dec!(0)),
        ),
        "policy",
    );
    distinct(
        spec_of(
            &cfg,
            long_grid(),
            config().with_invalid_action(InvalidActionPolicy::Mask, dec!(1)),
        ),
        "penalty",
    );
    distinct(
        spec_of(
            &cfg,
            long_grid(),
            config().with_terminal(TerminalValuation::CloseAtTouch),
        ),
        "terminal",
    );
    let other = |c: KernelConfig| spec_of(&c, long_grid(), config());
    distinct(
        other(KernelConfig::new(
            dec!(0.5),
            dec!(1),
            dec!(1000),
            dec!(10),
            101,
        )),
        "steps",
    );
    distinct(
        other(KernelConfig::new(
            dec!(0.5),
            dec!(1),
            dec!(1000),
            dec!(11),
            100,
        )),
        "order size",
    );
    distinct(
        other(KernelConfig::new(
            dec!(0.5),
            dec!(1),
            dec!(1001),
            dec!(10),
            100,
        )),
        "max price",
    );
}

#[test]
fn test_spec_fingerprint_covers_every_public_field() {
    let base = spec_of(&long_config(), long_grid(), config());
    let id = base.id();
    type Change = fn(&mut RlSpec);
    let changes: [(&str, Change); 29] = [
        ("observation version", |s| s.observation.version += 1),
        ("feature id", |s| s.observation.features.swap(0, 1)),
        ("feature unit", |s| {
            s.observation.features[0].unit = FeatureUnit::Count;
        }),
        ("feature time", |s| {
            s.observation.features[0].as_of = AsOf::DecisionCount;
        }),
        ("feature horizon", |s| {
            s.observation.features[0].horizon = FeatureHorizon::EpisodeSoFar;
        }),
        ("feature memory", |s| {
            s.observation.features[0].memory = StateMemory::PreviousDecision;
        }),
        ("feature divisor", |s| {
            s.observation.features[0].divisor = dec!(11);
        }),
        ("action version", |s| s.action.version += 1),
        ("tick", |s| s.action.tick_size = dec!(0.25)),
        ("lot", |s| s.action.lot_size = dec!(2)),
        ("max price", |s| s.action.max_price = dec!(1001)),
        ("max order", |s| s.action.max_order_size = dec!(11)),
        ("bid offsets", |s| s.action.bid_offsets_ticks[0] = 9),
        ("ask offsets", |s| s.action.ask_offsets_ticks[0] = 9),
        ("sizes", |s| s.action.sizes_lots[0] = 9),
        ("invalid action", |s| {
            s.action.invalid_action = InvalidActionPolicy::Reject;
        }),
        ("hedge", |s| {
            s.action.hedge = HedgeActions::MarketableAtTouch
        }),
        ("cancel", |s| s.action.cancel = CancelActions::CancelOnly),
        ("reward version", |s| s.reward.version += 1),
        ("metric", |s| s.reward.metric = MetricId::SpreadCapture),
        ("reward unit", |s| s.reward.unit = MetricUnit::Base),
        ("penalty", |s| s.reward.invalid_action_penalty = dec!(1)),
        ("f32 unit", |s| s.reward.f32_unit = dec!(2)),
        ("policy input", |s| s.reward.policy_input = true),
        ("episode version", |s| s.episode.version += 1),
        ("max steps", |s| s.episode.max_steps += 1),
        ("terminal", |s| {
            s.episode.terminal = TerminalValuation::CloseAtTouch;
        }),
        ("fewer bid offsets", |s| {
            s.action.bid_offsets_ticks.pop();
        }),
        ("more sizes", |s| s.action.sizes_lots.push(3)),
    ];
    let mut seen = vec![id.fingerprint];
    for (what, change) in changes {
        let mut changed = base.clone();
        change(&mut changed);
        let fingerprint = changed.id().fingerprint;
        assert!(!seen.contains(&fingerprint), "{what}");
        seen.push(fingerprint);
        assert!(changed.ensure(id).is_err(), "{what}");
    }
}

#[test]
fn test_spec_ensure_refuses_another_specification() {
    let spec = spec_of(&long_config(), long_grid(), config());
    let id = spec.id();
    spec.ensure(id).unwrap();
    let other = spec_of(
        &long_config(),
        long_grid(),
        RlConfig::new(scales(), dec!(2)),
    );
    assert!(matches!(
        other.ensure(id),
        Err(MMError::InvalidConfiguration(_))
    ));
    let mut later = id;
    later.reward += 1;
    assert!(spec.ensure(later).is_err());
}

#[test]
fn test_the_cancel_action_is_action_space_4_and_its_own_section() {
    let plain = spec_of(&long_config(), long_grid(), config());
    let cancel = spec_of(&long_config(), long_grid(), config().with_cancel_action());
    assert_eq!(plain.action.version, ACTION_SPACE_VERSION);
    assert_eq!(plain.action.cancel, CancelActions::NotModelled);
    assert_eq!(cancel.action.version, CANCEL_ACTION_SPACE_VERSION);
    assert_eq!(cancel.action.cancel, CancelActions::CancelOnly);
    let (p, c) = (plain.to_bytes(), cancel.to_bytes());
    assert!(!p.windows(2).any(|w| w == b"CX"));
    assert!(c.windows(2).any(|w| w == b"CX"));
    assert!(cancel.ensure(plain.id()).is_err() && plain.ensure(cancel.id()).is_err());
}

// Rows at 100 / 101, then an ask of 99.5 on row 2: a bid of 2 at 99.5 (offset 1) sent on
// row 0 fills on row 2 unless it is cancelled at the decision on row 1.
fn late_cross() -> Vec<MarketTick> {
    vec![
        tick(1000, dec!(99.5), dec!(5), dec!(100.5), dec!(5)),
        tick(2000, dec!(100), dec!(5), dec!(101), dec!(5)),
        tick(3000, dec!(98.5), dec!(5), dec!(99.5), dec!(5)),
        tick(4000, dec!(100), dec!(5), dec!(101), dec!(5)),
    ]
}

#[test]
fn test_a_cancel_withdraws_the_resting_quote_and_nothing_fills() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&late_cross(), &cfg).unwrap();
    let position = |second: RlAction| {
        let env: RlEnv<'_, 3, 2> =
            RlEnv::new(&cfg, &data, long_grid(), config().with_cancel_action()).unwrap();
        let mut state = env.reset(1).unwrap();
        env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
        let t = env.step(&mut state, second).unwrap();
        assert_eq!(t.action_status, ActionStatus::Applied);
        while !state.is_done() {
            env.step(&mut state, RlAction::Hold).unwrap();
        }
        state.kernel().venue_position()
    };
    assert_eq!(position(RlAction::Hold), dec!(2));
    assert_eq!(position(RlAction::Cancel), dec!(0));
}

#[test]
fn test_without_the_cancel_action_a_cancel_is_outside_the_space() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&late_cross(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    let before = state;
    assert!(env.step(&mut state, RlAction::Cancel).is_err());
    assert_eq!(state, before);
    let reject = config().with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), reject).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    let t = env.step(&mut state, RlAction::Cancel).unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::QuoteError(ERR_CANCEL_REFUSED)
    );
    assert_eq!(t.reward.penalty, dec!(1));
    while !state.is_done() {
        env.step(&mut state, RlAction::Hold).unwrap();
    }
    // Refused, so the bid rested and filled.
    assert_eq!(state.kernel().venue_position(), dec!(2));
}

// A policy that bids 2 lots at 99.5 on every decision, with a quote balance of 150 and a
// base deposit of 1 lot, on the long trajectory's rows (row 1 trades down to 99.5). The
// balance covers one lot at 99.5: the gate cuts the first bid to 1 lot and, once it has
// filled, refuses every later bid, so the policy never holds more than the balance paid
// for. The far ask is cut to the base deposit. Without the gate the same policy buys 2.
#[test]
fn test_a_policy_cannot_get_an_order_beyond_the_balance_executed() {
    let run = |margin: Option<KernelMargin>| {
        let mut cfg = long_config().with_initial_capital(dec!(150));
        cfg.margin = margin;
        let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
        let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
        let mut events: Vec<KernelEvent> = Vec::new();
        let mut state = env.reset_with(1, &mut events).unwrap();
        while !state.is_done() {
            env.step_with(&mut state, quote(1, 2, 1, 1), &mut events)
                .unwrap();
        }
        let summary = env.kernel().venue_summary(state.kernel()).unwrap();
        (
            events,
            summary.position,
            state.kernel().counters().venue_fills,
        )
    };
    let (events, position, fills) = run(Some(KernelMargin::new(dec!(1))));
    assert_eq!((position, fills), (dec!(1), 1));
    let margin: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Gate(g) => {
                assert_eq!(g.check, GateCheck::Margin);
                Some((g.side, g.verdict, g.submitted))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        margin[..2],
        [
            (Action::Buy, GateVerdict::Reduced, 1),
            (Action::Sell, GateVerdict::Reduced, 1),
        ]
    );
    assert!(
        margin[2..]
            .iter()
            .filter(|g| g.0 == Action::Buy)
            .all(|g| g.1 == GateVerdict::Rejected),
        "{margin:?}"
    );
    let (events, position, _) = run(None);
    assert_eq!(position, dec!(2));
    assert!(!events.iter().any(|e| matches!(e, KernelEvent::Gate(_))));
}

const MS: u64 = 1_000_000;

fn expiring(life: usize) -> RlAction {
    RlAction::ExpiringQuote {
        bid: 1,
        ask: 2,
        bid_size: 1,
        ask_size: 0,
        life,
    }
}

fn lives(values: &[u64]) -> QuoteLives {
    QuoteLives::new(values).unwrap()
}

#[test]
fn test_quote_lives_refuse_no_value() {
    assert!(QuoteLives::new(&[]).is_err());
}

#[test]
fn test_quote_lives_refuse_zero() {
    assert!(QuoteLives::new(&[0, 5]).is_err());
}

#[test]
fn test_quote_lives_refuse_a_life_above_the_longest() {
    assert!(QuoteLives::new(&[MAX_QUOTE_LIFE_NS]).is_ok());
    assert!(QuoteLives::new(&[MAX_QUOTE_LIFE_NS + 1]).is_err());
}

#[test]
fn test_quote_lives_refuse_duplicates() {
    assert!(QuoteLives::new(&[5, 5]).is_err());
}

#[test]
fn test_quote_lives_refuse_a_decrease() {
    assert!(QuoteLives::new(&[5, 3]).is_err());
}

#[test]
fn test_quote_lives_refuse_more_than_the_maximum() {
    let most: Vec<u64> = (1..=MAX_QUOTE_LIVES as u64).collect();
    assert_eq!(QuoteLives::new(&most).unwrap().as_slice(), most);
    let more: Vec<u64> = (1..=MAX_QUOTE_LIVES as u64 + 1).collect();
    assert!(QuoteLives::new(&more).is_err());
}

#[test]
fn test_expiry_error_code_is_204_and_unique() {
    assert_eq!(ERR_EXPIRY_REFUSED.0, 204);
    for other in [
        ERR_ACTION_OFF_GRID,
        ERR_HEDGE_REFUSED,
        ERR_CROSS_HEDGE_REFUSED,
        ERR_CANCEL_REFUSED,
    ] {
        assert_ne!(ERR_EXPIRY_REFUSED, other);
    }
}

#[test]
fn test_quote_lives_are_action_space_5_and_their_own_section() {
    let cfg = long_config();
    let plain = spec_of(&cfg, long_grid(), config());
    let cancel = spec_of(&cfg, long_grid(), config().with_cancel_action());
    let alone = spec_of(&cfg, long_grid(), config().with_quote_lives(lives(&[5, 9])));
    let both = spec_of(
        &cfg,
        long_grid(),
        config()
            .with_cancel_action()
            .with_quote_lives(lives(&[5, 9])),
    );
    let other = spec_of(
        &cfg,
        long_grid(),
        config().with_quote_lives(lives(&[5, 10])),
    );
    assert_eq!(plain.action.version, ACTION_SPACE_VERSION);
    assert_eq!(cancel.action.version, CANCEL_ACTION_SPACE_VERSION);
    assert_eq!(plain.action.expiry, QuoteExpiry::NotModelled);
    assert_eq!(alone.action.version, EXPIRY_ACTION_SPACE_VERSION);
    assert_eq!(both.action.version, EXPIRY_ACTION_SPACE_VERSION);
    assert_eq!(alone.action.expiry, QuoteExpiry::Lives(vec![5, 9]));
    assert_ne!(alone.id(), both.id());
    assert_ne!(alone.id().fingerprint, both.id().fingerprint);
    assert_ne!(alone.id().fingerprint, other.id().fingerprint);
    assert!(!plain.to_bytes().windows(2).any(|w| w == b"EX"));
    assert!(!cancel.to_bytes().windows(2).any(|w| w == b"EX"));
    assert!(alone.to_bytes().windows(2).any(|w| w == b"EX"));
    // Without lives the identity is the one written before lives existed: the same bytes as
    // a specification carrying them, less the section and with the older version.
    let mut expected = alone.to_bytes();
    let at = expected.windows(2).position(|w| w == b"EX").unwrap();
    let section = 2 + 8 + 2 * 8;
    let removed: Vec<u8> = expected.drain(at..at + section).collect();
    assert_eq!(&removed[..2], b"EX");
    let version = EXPIRY_ACTION_SPACE_VERSION.to_le_bytes();
    let older = ACTION_SPACE_VERSION.to_le_bytes();
    let plain_bytes = plain.to_bytes();
    assert_eq!(expected.len(), plain_bytes.len());
    let differing: Vec<usize> = (0..expected.len())
        .filter(|i| expected[*i] != plain_bytes[*i])
        .collect();
    assert_eq!(differing.len(), 1, "only the action version differs");
    assert_eq!(expected[differing[0]], version[0]);
    assert_eq!(plain_bytes[differing[0]], older[0]);
}

fn position_with(lives_ns: &[u64], first: RlAction, second: RlAction) -> Decimal {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&late_cross(), &cfg).unwrap();
    let rl = config().with_quote_lives(lives(lives_ns));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), rl).unwrap();
    let mut state = env.reset(1).unwrap();
    assert_eq!(
        env.step(&mut state, first).unwrap().action_status,
        ActionStatus::Applied
    );
    env.step(&mut state, second).unwrap();
    while !state.is_done() {
        env.step(&mut state, RlAction::Hold).unwrap();
    }
    state.kernel().venue_position()
}

#[test]
fn test_an_expiring_quote_is_gone_exactly_at_its_life() {
    // Sent and active at row 0 (1000 ms); the crossing row is at 3000 ms.
    let at_row = 2000 * MS;
    assert_eq!(
        position_with(&[at_row], expiring(0), RlAction::Hold),
        dec!(0)
    );
    assert_eq!(
        position_with(&[at_row + 1], expiring(0), RlAction::Hold),
        dec!(2)
    );
}

#[test]
fn test_a_replaced_quote_does_not_expire_its_replacement() {
    // The first quote would expire at 2500 ms; its replacement, sent at 2000 ms, still rests
    // at the crossing row of 3000 ms.
    assert_eq!(
        position_with(&[1500 * MS], expiring(0), quote(1, 2, 1, 0)),
        dec!(2)
    );
    assert_eq!(
        position_with(&[1500 * MS, 1500 * MS + 1], expiring(0), expiring(1)),
        dec!(2)
    );
}

#[test]
fn test_without_lives_an_expiring_quote_is_outside_the_space() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&late_cross(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), config()).unwrap();
    let mut state = env.reset(1).unwrap();
    assert!(env.step(&mut state, expiring(0)).is_err());
    let reject = config().with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), reject).unwrap();
    let mut state = env.reset(1).unwrap();
    assert_eq!(
        env.step(&mut state, expiring(0)).unwrap().action_status,
        ActionStatus::QuoteError(ERR_EXPIRY_REFUSED)
    );
}

#[test]
fn test_a_life_out_of_range_is_off_the_grid() {
    let cfg = long_config();
    let data = KernelDataset::from_ticks(&late_cross(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(
        &cfg,
        &data,
        long_grid(),
        config().with_quote_lives(lives(&[MS])),
    )
    .unwrap();
    let mut state = env.reset(1).unwrap();
    assert!(env.step(&mut state, expiring(1)).is_err());
    let reject = config()
        .with_quote_lives(lives(&[MS]))
        .with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, long_grid(), reject).unwrap();
    let mut state = env.reset(1).unwrap();
    let t = env.step(&mut state, expiring(1)).unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::QuoteError(ERR_ACTION_OFF_GRID)
    );
}
