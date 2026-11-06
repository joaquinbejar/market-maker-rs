//! Hedge actions of the `rl` module: a marketable order at the observed touch that reduces
//! the delivered position, its cost through the kernel's ledger, its mask and its policy, and
//! the kernel's limits deciding on it as on any order. All data here is synthetic.

#![cfg(feature = "rl")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionStatus, FinancialKernel, KernelAction, KernelConfig, KernelDataset, QuoteSide,
};
use market_maker_rs::dec;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    ERR_HEDGE_REFUSED, EpisodeRecorder, HedgeActions, InvalidActionPolicy, ObservationScales,
    RlAction, RlConfig, RlEnv, RlState,
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

fn hedging() -> RlConfig {
    RlConfig::new(scales(), dec!(1)).with_hedge_actions()
}

fn tick(ts: u64, bid: Decimal, ask: Decimal) -> MarketTick {
    MarketTick::new(ts, bid, dec!(5), ask, dec!(5))
}

/// Tick 0.5, lot 1, fee 0.1 %, no latency, the immediate fill model.
fn kernel_config() -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_fee_rate(dec!(0.001))
}

/// Offsets 0, 1 and 10 ticks, sizes 1 and 2 lots.
fn grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap()
}

fn quote(bid: usize, ask: usize, bid_size: usize, ask_size: usize) -> RlAction {
    RlAction::Quote {
        bid,
        ask,
        bid_size,
        ask_size,
    }
}

// A long of 2, half of it hedged.
//
//   row  time  bid    ask    mid
//   0    1000  99.5   100.5  100
//   1    2000  98.5   99.5   99
//   2    3000  100    101    100.5
//   3    4000  100    101    100.5
//   4    5000  100.5  101.5  101
//
// On row 0 the agent bids 2 at 99.5 (offset 1) and asks 1 at 105 (offset 10). Row 1's ask
// reaches the bid: bought 2 at 99.5, fee 2 * 99.5 * 0.001 = 0.199. On row 1 the agent sees
// the long of 2 and hedges 1: a sale of 1 at the observed bid, 98.5, which replaces the ask
// at 105. Row 2's bid of 100 reaches it, and it fills at its own price, 98.5, with a fee of
// 98.5 * 0.001 = 0.0985. The last inventory is marked at the last mid, 101:
//
//   hedged:  -199 - 0.199 + 98.5 - 0.0985 + 1 * 101 =  0.2025
//   held:    -199 - 0.199 + 2 * 101               =  2.801
//
// so the hedge costs 1 * (98.5 - 101) - 0.0985 = -2.5985 against holding.
fn long_ticks() -> Vec<MarketTick> {
    vec![
        tick(1000, dec!(99.5), dec!(100.5)),
        tick(2000, dec!(98.5), dec!(99.5)),
        tick(3000, dec!(100), dec!(101)),
        tick(4000, dec!(100), dec!(101)),
        tick(5000, dec!(100.5), dec!(101.5)),
    ]
}

/// Runs `actions` from the first decision, then holds to the end, checking the reward
/// against the ledger at every step. Returns the state and the hedge masks seen.
fn run<'d>(env: &RlEnv<'d, 3, 2>, actions: &[RlAction]) -> (RlState, Vec<u64>, Vec<ActionStatus>) {
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    let mut masks = vec![env.hedge_mask(&state)];
    let mut statuses = Vec::new();
    let mut total = Decimal::ZERO;
    let mut next = actions.iter();
    while !state.is_done() {
        let action = next.next().copied().unwrap_or(RlAction::Hold);
        let t = env.step_with(&mut state, action, &mut recorder).unwrap();
        assert_eq!(t.hedge_mask, env.hedge_mask(&state));
        masks.push(t.hedge_mask);
        statuses.push(t.action_status);
        total += t.reward.ledger_change + t.reward.terminal_close;
        env.reconcile(&state, &recorder).unwrap();
    }
    assert_eq!(total, state.net());
    (state, masks, statuses)
}

#[test]
fn test_hedge_sells_a_long_at_the_observed_bid_and_costs_through_the_ledger() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), hedging()).unwrap();
    assert_eq!(env.spec().action.hedge, HedgeActions::MarketableAtTouch);

    let (hedged, masks, statuses) = run(&env, &[quote(1, 2, 1, 0), RlAction::Hedge { size: 0 }]);
    // Flat at the first decision; long 2 at the second, so both sizes; long 1 after.
    assert_eq!(masks[0], 0);
    assert_eq!(masks[1], 0b11);
    assert_eq!(masks[2], 0b01);
    assert_eq!(statuses[1], ActionStatus::Applied);
    assert_eq!(hedged.kernel().venue_position(), dec!(1));
    assert_eq!(hedged.net(), dec!(0.2025));

    let (held, _, _) = run(&env, &[quote(1, 2, 1, 0)]);
    assert_eq!(held.kernel().venue_position(), dec!(2));
    assert_eq!(held.net(), dec!(2.801));
    assert_eq!(hedged.net() - held.net(), dec!(-2.5985));
}

// A short of 2, half of it hedged.
//
//   row  time  bid    ask    mid
//   0    1000  99.5   100.5  100
//   1    2000  100.5  101.5  101
//   2    3000  99     100    99.5
//   3    4000  99     100    99.5
//   4    5000  99     100    99.5
//
// On row 0 the agent asks 2 at 100.5 (offset 1) and bids 1 at 95 (offset 10). Row 1's bid
// reaches the ask: sold 2 at 100.5, fee 0.201. On row 1 the agent hedges 1: a purchase of 1
// at the observed ask, 101.5, filled on row 2 at that price, fee 0.1015. Marked at 99.5:
//
//   hedged:  201 - 0.201 - 101.5 - 0.1015 - 1 * 99.5 = -0.3025
//   held:    201 - 0.201 - 2 * 99.5                  =  1.799
//
// so the hedge costs 1 * (99.5 - 101.5) - 0.1015 = -2.1015 against holding.
#[test]
fn test_hedge_buys_a_short_at_the_observed_ask_and_costs_through_the_ledger() {
    let cfg = kernel_config();
    let ticks = vec![
        tick(1000, dec!(99.5), dec!(100.5)),
        tick(2000, dec!(100.5), dec!(101.5)),
        tick(3000, dec!(99), dec!(100)),
        tick(4000, dec!(99), dec!(100)),
        tick(5000, dec!(99), dec!(100)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), hedging()).unwrap();
    let (hedged, masks, _) = run(&env, &[quote(2, 1, 0, 1), RlAction::Hedge { size: 0 }]);
    assert_eq!(masks[1], 0b11);
    assert_eq!(hedged.kernel().venue_position(), dec!(-1));
    assert_eq!(hedged.net(), dec!(-0.3025));
    let (held, _, _) = run(&env, &[quote(2, 1, 0, 1)]);
    assert_eq!(held.net(), dec!(1.799));
    assert_eq!(hedged.net() - held.net(), dec!(-2.1015));
}

#[test]
fn test_hedge_sends_the_kernel_one_order_at_the_touch_and_nothing_else() {
    // The environment and a kernel driven directly with that order agree on every state,
    // under limits too: the hedge reaches the reward only through the kernel.
    let cfg =
        kernel_config().with_limits(RiskLimits::new(dec!(2), dec!(1000000), dec!(1)).unwrap());
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), hedging()).unwrap();
    let kernel = FinancialKernel::new(&cfg, &data).unwrap();
    let mut state = env.reset(1).unwrap();
    let mut direct = kernel.reset(1);
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    let opening = env
        .kernel()
        .grid_action(&direct, &grid(), (1, 2, 1, 0))
        .unwrap();
    kernel.step(&mut direct, opening);
    assert_eq!(*state.kernel(), direct);
    let t = env.step(&mut state, RlAction::Hedge { size: 1 }).unwrap();
    let outcome = kernel.step(
        &mut direct,
        KernelAction::OneSided {
            side: QuoteSide::Ask,
            price: 197,
            lots: 2,
        },
    );
    assert_eq!(t.action_status, outcome.applied);
    assert_eq!(*state.kernel(), direct);
    while !state.is_done() {
        env.step(&mut state, RlAction::Hold).unwrap();
        kernel.step(&mut direct, KernelAction::Hold);
        assert_eq!(*state.kernel(), direct);
    }
    assert_eq!(state.kernel().venue_position(), dec!(0));
}

#[test]
fn test_hedge_mask_reads_no_limit() {
    let free = kernel_config();
    let capped =
        kernel_config().with_limits(RiskLimits::new(dec!(2), dec!(1000000), dec!(1)).unwrap());
    let free_data = KernelDataset::from_ticks(&long_ticks(), &free).unwrap();
    let capped_data = KernelDataset::from_ticks(&long_ticks(), &capped).unwrap();
    let a: RlEnv<'_, 3, 2> = RlEnv::new(&free, &free_data, grid(), hedging()).unwrap();
    let b: RlEnv<'_, 3, 2> = RlEnv::new(&capped, &capped_data, grid(), hedging()).unwrap();
    let (_, free_masks, _) = run(&a, &[quote(1, 2, 1, 0)]);
    let (_, capped_masks, _) = run(&b, &[quote(1, 2, 1, 0)]);
    assert_eq!(free_masks, capped_masks);
}

/// Runs `check` at the second decision of the long trajectory, long 2.
fn at_long_two(rl: RlConfig, check: impl FnOnce(&RlEnv<'_, 3, 2>, &mut RlState)) {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), rl).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    check(&env, &mut state);
}

#[test]
fn test_mask_policy_refuses_a_hedge_beyond_the_position_and_changes_nothing() {
    let grid = ActionGrid::new([0, 1, 10], [0, 1, 10], [2, 3]).unwrap();
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, hedging()).unwrap();
    let mut state = env.reset(1).unwrap();
    // Flat: no hedge at all.
    let before = state;
    let error = env
        .step(&mut state, RlAction::Hedge { size: 0 })
        .unwrap_err();
    assert!(matches!(error, MMError::InvalidQuoteGeneration(_)));
    assert_eq!(state, before);
    env.step(&mut state, quote(1, 2, 0, 0)).unwrap();
    // Long 2: 2 lots reduce it, 3 would turn it short.
    assert_eq!(env.hedge_mask(&state), 0b01);
    let before = state;
    let error = env
        .step(&mut state, RlAction::Hedge { size: 1 })
        .unwrap_err();
    assert!(matches!(error, MMError::InvalidQuoteGeneration(_)));
    assert_eq!(state, before);
    // Off the grid.
    let error = env
        .step(&mut state, RlAction::Hedge { size: 2 })
        .unwrap_err();
    assert!(matches!(error, MMError::InvalidQuoteGeneration(_)));
    assert_eq!(state, before);
    env.step(&mut state, RlAction::Hedge { size: 0 }).unwrap();
}

#[test]
fn test_reject_policy_takes_a_hedge_it_cannot_do_as_a_hold_with_the_penalty() {
    let rl = hedging().with_invalid_action(InvalidActionPolicy::Reject, dec!(0.25));
    at_long_two(rl, |env, state| {
        let grid_of = env.spec().action.sizes_lots.clone();
        assert_eq!(grid_of, vec![1, 2]);
        let mut beyond = *state;
        // Sell the whole long with a hedge of 2, then try another hedge while flat.
        env.step(&mut beyond, RlAction::Hedge { size: 1 }).unwrap();
        let t = env.step(&mut beyond, RlAction::Hedge { size: 0 }).unwrap();
        assert_eq!(t.action_status, ActionStatus::QuoteError(ERR_HEDGE_REFUSED));
        assert_eq!(t.reward.penalty, dec!(0.25));
        assert_eq!(beyond.kernel().venue_position(), dec!(0));
        assert_eq!(beyond.penalties(), dec!(0.25));
    });
}

#[test]
fn test_without_hedge_actions_a_hedge_is_outside_the_space() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&long_ticks(), &cfg).unwrap();
    let plain = RlConfig::new(scales(), dec!(1));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), plain).unwrap();
    assert!(matches!(
        env.spec().action.hedge,
        HedgeActions::NotModelled { .. }
    ));
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    assert_eq!(env.hedge_mask(&state), 0);
    let before = state;
    assert!(env.step(&mut state, RlAction::Hedge { size: 0 }).is_err());
    assert_eq!(state, before);

    let reject = plain.with_invalid_action(InvalidActionPolicy::Reject, dec!(1));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), reject).unwrap();
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, quote(1, 2, 1, 0)).unwrap();
    let t = env.step(&mut state, RlAction::Hedge { size: 0 }).unwrap();
    assert_eq!(t.action_status, ActionStatus::QuoteError(ERR_HEDGE_REFUSED));
    assert_eq!(state.kernel().venue_position(), dec!(2));
}

#[test]
fn test_hedge_mask_leaves_out_a_size_the_kernel_would_not_admit() {
    // The ask falls to the mid every other row, so a bid one tick under the mid fills.
    let ticks: Vec<MarketTick> = (0..8_u64)
        .map(|i| {
            if i % 2 == 0 {
                tick(1000 + i, dec!(100), dec!(101))
            } else {
                tick(1000 + i, dec!(99), dec!(100))
            }
        })
        .collect();
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    // 11 lots is above the largest order, 10.
    let grid = ActionGrid::new([1, 2, 10], [10, 11, 12], [10, 11]).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, hedging()).unwrap();
    let mut state = env.reset(1).unwrap();
    for _ in 0..2 {
        env.step(&mut state, quote(0, 2, 0, 0)).unwrap();
        env.step(&mut state, RlAction::Hold).unwrap();
    }
    let observed = env.observe(&state).unwrap();
    assert!(observed.raw[6].unwrap() >= dec!(11));
    assert_eq!(env.mask(&state).size, 0b01);
    assert_eq!(env.hedge_mask(&state), 0b01);
}

#[test]
fn test_the_kernel_refuses_a_hedge_it_would_not_admit_even_under_reject() {
    // The mask's check is the agent's; the kernel's admission decides on its own. With the
    // policy rejecting rather than erroring, a size above the largest order reaches the
    // kernel and is refused there.
    let ticks: Vec<MarketTick> = (0..8_u64)
        .map(|i| {
            if i % 2 == 0 {
                tick(1000 + i, dec!(100), dec!(101))
            } else {
                tick(1000 + i, dec!(99), dec!(100))
            }
        })
        .collect();
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let grid = ActionGrid::new([1, 2, 10], [10, 11, 12], [10, 11]).unwrap();
    let rl = hedging().with_invalid_action(InvalidActionPolicy::Reject, dec!(0.25));
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid, rl).unwrap();
    let mut state = env.reset(1).unwrap();
    for _ in 0..2 {
        env.step(&mut state, quote(0, 2, 0, 0)).unwrap();
        env.step(&mut state, RlAction::Hold).unwrap();
    }
    let held = state.kernel().venue_position();
    assert_eq!(env.hedge_mask(&state), 0b01);
    let t = env.step(&mut state, RlAction::Hedge { size: 1 }).unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::Rejected(market_maker_rs::backtest::kernel::RejectReason::SizeRange)
    );
    assert_eq!(t.reward.penalty, dec!(0.25));
    assert_eq!(state.kernel().venue_position(), held);
}
