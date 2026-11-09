//! Hedges in a second instrument from the `rl` module (#256): action space 3 over a
//! `HedgedKernel`, its mask, its features, its reward through both lanes' ledgers, and the
//! specification of an environment without a lane left as it was. All data here is
//! synthetic.

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
    EvaluationInput, HedgeAttribution, HedgeCosts, attribute, attribute_by,
};
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionStatus, FundingRate, HedgeInstrument, HedgeLaneConfig, KernelAction,
    KernelConfig, KernelDataset,
};
use market_maker_rs::dec;
use market_maker_rs::rl::{
    ERR_CROSS_HEDGE_REFUSED, EpisodeRecorder, HEDGED_ACTION_SPACE_VERSION,
    HEDGED_OBSERVATION_SPACE_VERSION, HEDGED_REWARD_VERSION, HedgeScales, InvalidActionPolicy,
    ObservationScales, QuoteLives, RlAction, RlConfig, RlEnv, RlState, TerminalValuation,
};

const MS: u64 = 1_000_000;

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

fn hedge_scales() -> HedgeScales {
    HedgeScales {
        spread_ticks: dec!(4),
        basis: dec!(2),
        row_age_ms: dec!(1000),
        position: dec!(4),
    }
}

fn tick(ts: u64, bid: Decimal, ask: Decimal) -> MarketTick {
    MarketTick::new(ts, bid, dec!(5), ask, dec!(5))
}

/// Tick 0.5, lot 1, fee 0.1 %, no latency, the immediate fill model.
fn kernel_config() -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_fee_rate(dec!(0.001))
}

/// The hedge instrument: tick 0.5, lot 1, fee 0.05 %.
fn instrument() -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_fee_rate(dec!(0.0005))
}

fn grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap()
}

// The quoted lane, as in `rl_hedge.rs`: on row 0 the agent bids 2 at 99.5 and asks 1 at
// 105; row 1's ask reaches the bid, so it is long 2 at 99.5 from 2 000 ms.
//
//   quoted  time  bid    ask         hedge  time  bid    ask
//   0       1000  99.5   100.5        0      1500  99     100
//   1       2000  98.5   99.5         1      2500  99.5   100.5
//   2       3000  100    101          2      3500  100    101
//   3       4000  100    101          3      4500  100    101
//   4       5000  100.5  101.5
fn quoted_ticks() -> Vec<MarketTick> {
    vec![
        tick(1000, dec!(99.5), dec!(100.5)),
        tick(2000, dec!(98.5), dec!(99.5)),
        tick(3000, dec!(100), dec!(101)),
        tick(4000, dec!(100), dec!(101)),
        tick(5000, dec!(100.5), dec!(101.5)),
    ]
}

fn hedge_ticks() -> Vec<MarketTick> {
    vec![
        tick(1500, dec!(99), dec!(100)),
        tick(2500, dec!(99.5), dec!(100.5)),
        tick(3500, dec!(100), dec!(101)),
        tick(4500, dec!(100), dec!(101)),
    ]
}

struct Data {
    cfg: KernelConfig,
    quoted: KernelDataset,
    hedge: KernelDataset,
    lane: HedgeLaneConfig,
}

fn data() -> Data {
    let cfg = kernel_config();
    let inst = instrument();
    Data {
        quoted: KernelDataset::from_ticks(&quoted_ticks(), &cfg).unwrap(),
        hedge: KernelDataset::from_ticks(&hedge_ticks(), &inst).unwrap(),
        lane: HedgeLaneConfig::new(inst, HedgeInstrument::Spot, dec!(1), 10_000 * MS),
        cfg,
    }
}

fn env<'d>(d: &'d Data, rl: RlConfig) -> RlEnv<'d, 3, 2> {
    RlEnv::new_hedged(
        &d.cfg,
        &d.quoted,
        &d.lane,
        &d.hedge,
        grid(),
        vec![1, 2],
        hedge_scales(),
        rl,
    )
    .unwrap()
}

fn buy_two() -> RlAction {
    RlAction::Quote {
        bid: 1,
        ask: 2,
        bid_size: 1,
        ask_size: 0,
    }
}

/// Runs `actions` from the first decision, then holds to the end, checking the reward
/// against both lanes' attribution at every step.
fn run(env: &RlEnv<'_, 3, 2>, actions: &[RlAction]) -> (RlState, Vec<u64>) {
    let (state, masks, _) = run_recorded(env, actions);
    (state, masks)
}

fn run_recorded(
    env: &RlEnv<'_, 3, 2>,
    actions: &[RlAction],
) -> (RlState, Vec<u64>, EpisodeRecorder) {
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    let mut masks = vec![env.cross_hedge_mask(&state)];
    let mut total = Decimal::ZERO;
    let mut next = actions.iter();
    while !state.is_done() {
        let action = next.next().copied().unwrap_or(RlAction::Hold);
        let t = env.step_with(&mut state, action, &mut recorder).unwrap();
        assert_eq!(t.cross_hedge_mask, env.cross_hedge_mask(&state));
        masks.push(t.cross_hedge_mask);
        total += t.reward.ledger_change + t.reward.terminal_close;
        env.reconcile(&state, &recorder).unwrap();
    }
    assert_eq!(total, state.net());
    (state, masks, recorder)
}

/// The hedge lane's attribution of an episode, and the whole window's split by quoted row.
fn hedge_attribution(
    env: &RlEnv<'_, 3, 2>,
    state: &RlState,
    recorder: &EpisodeRecorder,
) -> (
    HedgeAttribution,
    Option<Decimal>,
    Vec<(u64, HedgeAttribution)>,
) {
    let records = env.records(state, recorder).unwrap();
    let input = EvaluationInput::new(records.window, &records.marks, &records.fills, dec!(1))
        .unwrap()
        .with_hedge(
            &records.hedge_marks,
            &records.hedge_fills,
            &records.hedge_funding,
        )
        .unwrap();
    let of = |costs: HedgeCosts| match costs {
        HedgeCosts::Attributed(hedge) => hedge,
        other => panic!("expected an attributed hedge, got {other:?}"),
    };
    let total = attribute(&input).unwrap();
    let by_row = attribute_by(&input, |m| m.timestamp_ms).unwrap();
    let buckets = by_row
        .buckets
        .iter()
        .map(|(t, b)| (*t, of(b.attribution.hedge_costs)))
        .collect();
    (
        of(total.total.hedge_costs),
        total.hedge_final_position,
        buckets,
    )
}

#[test]
fn test_a_cross_hedge_reduces_the_net_exposure_and_its_ledger_reaches_the_reward() {
    let d = data();
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    // Long 2 at 2 000 ms; a hedge of 1 sells at the hedge bid seen then, 99, which stands at
    // execution: cash 99, fee 0.0495.
    let (state, masks) = run(&env, &[buy_two(), RlAction::CrossHedge { size: 0 }]);
    // Flat: nothing to hedge. Long 2: both sizes. Net 1 after the hedge: one size.
    assert_eq!(masks[0], 0);
    assert_eq!(masks[1], 0b11);
    assert_eq!(masks[2], 0b01);
    let lane = state.lane().unwrap();
    assert_eq!(lane.venue_position_lots(), -1);
    assert_eq!(lane.counters().filled, 1);
    // Quoted: -199 - 0.199 + 2 * 101 = 2.801. Hedge: 99 - 0.0495 - 100.5 = -1.5495, at the
    // mid of the last hedge row it ran, 100.5.
    assert_eq!(state.net(), dec!(2.801) + dec!(-1.5495));
}

#[test]
fn test_a_close_at_the_touch_closes_both_lanes() {
    let d = data();
    let rl = RlConfig::new(scales(), dec!(1)).with_terminal(TerminalValuation::CloseAtTouch);
    let env = env(&d, rl);
    let (state, _) = run(&env, &[buy_two(), RlAction::CrossHedge { size: 0 }]);
    // Quoted: sell 2 at 100.5 against the mid 101, fee 0.201: -1.201. Hedge: buy 1 at the
    // last hedge ask, 101, against its mid 100.5, fee 0.0505: -0.5505.
    let close = state.close().unwrap();
    assert_eq!((close.quantity, close.price), (dec!(2), dec!(100.5)));
    let hedge = state.hedge_close().unwrap();
    assert_eq!(
        (hedge.quantity, hedge.price, hedge.fee),
        (dec!(1), dec!(101), dec!(0.0505))
    );
    assert_eq!(
        state.net(),
        dec!(2.801) + dec!(-1.5495) + dec!(-1.201) + dec!(-0.5505)
    );
}

#[test]
fn test_the_hedge_features_are_what_the_agent_sees_of_the_lane() {
    let d = data();
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    let mut state = env.reset(1).unwrap();
    let first = env.observe_hedge(&state).unwrap().unwrap();
    // At 1 000 ms no hedge row has run: no row values, flat, nothing in flight.
    assert_eq!(first.row, None);
    assert_eq!(first.raw, [None, None, None, Some(dec!(0)), Some(dec!(0))]);
    env.step(&mut state, buy_two()).unwrap();
    let t = env
        .step(&mut state, RlAction::CrossHedge { size: 0 })
        .unwrap();
    assert_eq!(t.cross_hedge, Some(Ok(())));
    // At 3 000 ms: the hedge row of 2 500 ms, 1 wide (2 ticks), mid 100 against the quoted
    // mid 100.5, 500 ms old; short 1 after the report.
    let h = t.hedge_observation.unwrap();
    assert_eq!(h.row, Some(1));
    assert_eq!(
        h.raw,
        [
            Some(dec!(2)),
            Some(dec!(-0.5)),
            Some(dec!(500)),
            Some(dec!(-1)),
            Some(dec!(0))
        ]
    );
    assert_eq!(h.values, [0.5, -0.25, 0.5, -0.25, 0.0]);
}

#[test]
fn test_outside_the_cross_mask_the_policy_decides() {
    let d = data();
    let masked = env(&d, RlConfig::new(scales(), dec!(1)));
    let mut state = masked.reset(1).unwrap();
    let before = state;
    // Flat: nothing to hedge.
    assert!(
        masked
            .step(&mut state, RlAction::CrossHedge { size: 0 })
            .is_err()
    );
    assert_eq!(state, before);
    let rejecting = env(
        &d,
        RlConfig::new(scales(), dec!(1))
            .with_invalid_action(InvalidActionPolicy::Reject, dec!(0.25)),
    );
    let mut state = rejecting.reset(1).unwrap();
    let t = rejecting
        .step(&mut state, RlAction::CrossHedge { size: 0 })
        .unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::QuoteError(ERR_CROSS_HEDGE_REFUSED)
    );
    assert_eq!(t.reward.penalty, dec!(0.25));
    assert_eq!(t.cross_hedge, None);
    // Long 2 from a fresh episode: hedging all of it is inside the mask, and a size off the
    // grid is outside it.
    let mut state = rejecting.reset(1).unwrap();
    rejecting.step(&mut state, buy_two()).unwrap();
    assert_eq!(rejecting.cross_hedge_mask(&state), 0b11);
    let mut long = state;
    let t = rejecting
        .step(&mut long, RlAction::CrossHedge { size: 1 })
        .unwrap();
    assert_eq!(t.action_status, ActionStatus::Applied);
    let mut past = state;
    let t = rejecting
        .step(&mut past, RlAction::CrossHedge { size: 2 })
        .unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::QuoteError(ERR_CROSS_HEDGE_REFUSED)
    );
}

#[test]
fn test_without_a_lane_the_specification_and_the_cross_hedge_are_as_before() {
    let d = data();
    let plain: RlEnv<'_, 3, 2> = RlEnv::new(
        &d.cfg,
        &d.quoted,
        grid(),
        RlConfig::new(scales(), dec!(1)).with_invalid_action(InvalidActionPolicy::Reject, dec!(0)),
    )
    .unwrap();
    let id = plain.spec().id();
    assert_eq!(
        (id.observation, id.action, id.reward, id.episode),
        (1, 2, 1, 1)
    );
    assert!(plain.spec().observation.hedge_features.is_empty());
    assert!(plain.spec().action.cross_hedge.is_none());
    let mut state = plain.reset(1).unwrap();
    assert_eq!(plain.cross_hedge_mask(&state), 0);
    assert_eq!(plain.observe_hedge(&state).unwrap(), None);
    assert!(state.lane().is_none());
    let t = plain
        .step(&mut state, RlAction::CrossHedge { size: 0 })
        .unwrap();
    assert_eq!(
        t.action_status,
        ActionStatus::QuoteError(ERR_CROSS_HEDGE_REFUSED)
    );
    assert_eq!(t.hedge_observation, None);
    // With a lane: the hedged versions, and an identity of its own.
    let hedged = env(
        &d,
        RlConfig::new(scales(), dec!(1)).with_invalid_action(InvalidActionPolicy::Reject, dec!(0)),
    );
    let h = hedged.spec().id();
    assert_eq!(
        (h.observation, h.action, h.reward, h.episode),
        (
            HEDGED_OBSERVATION_SPACE_VERSION,
            HEDGED_ACTION_SPACE_VERSION,
            HEDGED_REWARD_VERSION,
            1
        )
    );
    assert!(hedged.spec().ensure(id).is_err());
    assert!(plain.spec().ensure(h).is_err());
    let shape = hedged.spec().shape();
    assert_eq!((shape.hedge_features, shape.hedge_sizes), (5, 2));
    // A state of one is refused by the other.
    let mut theirs = hedged.reset(1).unwrap();
    assert!(plain.step(&mut theirs, RlAction::Hold).is_err());
    let mut ours = plain.reset(1).unwrap();
    assert!(hedged.step(&mut ours, RlAction::Hold).is_err());
}

#[test]
fn test_the_hedged_identity_covers_the_lane_and_its_scales() {
    let d = data();
    let id = |lane: &HedgeLaneConfig, sizes: Vec<u64>, scales: HedgeScales| {
        RlEnv::<'_, 3, 2>::new_hedged(
            &d.cfg,
            &d.quoted,
            lane,
            &d.hedge,
            grid(),
            sizes,
            scales,
            RlConfig::new(self::scales(), dec!(1)),
        )
        .unwrap()
        .spec()
        .id()
    };
    let base = id(&d.lane, vec![1, 2], hedge_scales());
    let mut lane = d.lane.clone();
    lane.multiplier = dec!(2);
    assert_ne!(id(&lane, vec![1, 2], hedge_scales()), base);
    let mut lane = d.lane.clone();
    lane.max_staleness_ns = 1;
    assert_ne!(id(&lane, vec![1, 2], hedge_scales()), base);
    assert_ne!(id(&d.lane, vec![1, 3], hedge_scales()), base);
    let mut scales = hedge_scales();
    scales.basis = dec!(3);
    assert_ne!(id(&d.lane, vec![1, 2], scales), base);
    // Refused: no sizes, a size of zero, a scale that is not positive.
    let refused = |sizes: Vec<u64>, scales: HedgeScales| {
        RlEnv::<'_, 3, 2>::new_hedged(
            &d.cfg,
            &d.quoted,
            &d.lane,
            &d.hedge,
            grid(),
            sizes,
            scales,
            RlConfig::new(self::scales(), dec!(1)),
        )
        .is_err()
    };
    assert!(refused(vec![], hedge_scales()));
    assert!(refused(vec![1, 0], hedge_scales()));
    assert!(refused(vec![1; 65], hedge_scales()));
    let mut scales = hedge_scales();
    scales.row_age_ms = dec!(0);
    assert!(refused(vec![1], scales));
}

#[test]
fn test_a_snapshot_replays_the_same_transitions() {
    let d = data();
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, buy_two()).unwrap();
    let snapshot = state;
    let plan = [
        RlAction::CrossHedge { size: 0 },
        RlAction::Hold,
        RlAction::Hold,
    ];
    let mut runs = Vec::new();
    for start in [state, snapshot] {
        let mut s = start;
        let mut transitions = Vec::new();
        for action in plan {
            if s.is_done() {
                break;
            }
            transitions.push(env.step(&mut s, action).unwrap());
        }
        runs.push((transitions, s));
    }
    assert_eq!(runs[0], runs[1]);
    let _ = KernelAction::Hold;
}

#[test]
fn test_the_cross_mask_is_empty_while_the_lane_refuses() {
    // Stale after 400 ms: at 2 000 ms the hedge row of 1 500 ms is too old, long 2 or not.
    let mut d = data();
    d.lane.max_staleness_ns = 400 * MS;
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    let mut state = env.reset(1).unwrap();
    env.step(&mut state, buy_two()).unwrap();
    assert_eq!(state.kernel().agent_position(), dec!(2));
    assert_eq!(env.cross_hedge_mask(&state), 0);
    assert!(
        env.step(&mut state, RlAction::CrossHedge { size: 0 })
            .is_err()
    );
}

#[test]
fn test_the_hedge_lane_is_attributed_by_hand_apart_from_the_quoted_lane() {
    let d = data();
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    let (state, _, recorder) = run_recorded(&env, &[buy_two(), RlAction::CrossHedge { size: 0 }]);
    let (hedge, position, buckets) = hedge_attribution(&env, &state, &recorder);
    // Sold 1 at 99 at 2 000 ms against the hedge mark in force, 99.5 (row at 1 500): -0.5.
    // Short 1 through the marks 100, 100.5 and 100.5: -1. Fee 0.0495, no funding.
    assert_eq!(hedge.spread_capture, dec!(-0.5));
    assert_eq!(hedge.inventory_pnl, dec!(-1));
    assert_eq!(hedge.fees, dec!(0.0495));
    assert_eq!(hedge.funding, dec!(0));
    assert_eq!(hedge.net_pnl, dec!(-1.5495));
    assert_eq!(hedge.costs().unwrap(), dec!(0.5495));
    assert_eq!(position, Some(dec!(-1)));
    // By quoted row: the fill and the move at 2 500 in the row at 2 000, the move at 3 500
    // in the row at 3 000, and the flat move at 4 500 in the row at 4 000.
    let net = |t: u64| buckets.iter().find(|(at, _)| *at == t).unwrap().1.net_pnl;
    assert_eq!(net(1000), dec!(0));
    assert_eq!(net(2000), dec!(-1.0495));
    assert_eq!(net(3000), dec!(-0.5));
    assert_eq!(net(4000), dec!(0));
    assert_eq!(net(5000), dec!(0));
}

#[test]
fn test_the_hedge_close_is_a_hedge_fill_of_the_attribution() {
    let d = data();
    let rl = RlConfig::new(scales(), dec!(1)).with_terminal(TerminalValuation::CloseAtTouch);
    let env = env(&d, rl);
    let (state, _, recorder) = run_recorded(&env, &[buy_two(), RlAction::CrossHedge { size: 0 }]);
    let (hedge, position, _) = hedge_attribution(&env, &state, &recorder);
    // Plus the close: bought 1 at 101 against the mark 100.5, -0.5, with a fee of 0.0505.
    assert_eq!(hedge.spread_capture, dec!(-1));
    assert_eq!(hedge.inventory_pnl, dec!(-1));
    assert_eq!(hedge.fees, dec!(0.1));
    assert_eq!(hedge.net_pnl, dec!(-1.5495) + dec!(-0.5505));
    assert_eq!(position, Some(dec!(0)));
}

#[test]
fn test_a_perpetual_hedge_attributes_its_funding() {
    let mut d = data();
    // A rate of 0.0001 at 3 000 ms, valued at the mark of the hedge row run before it,
    // 100 (row at 2 500): the short of 1 receives 1 * 100 * 0.0001 = 0.01.
    d.lane = HedgeLaneConfig::new(
        instrument(),
        HedgeInstrument::LinearPerpetual {
            funding: vec![FundingRate {
                at_ns: 3000 * MS,
                rate: dec!(0.0001),
            }],
        },
        dec!(1),
        10_000 * MS,
    );
    let env = env(&d, RlConfig::new(scales(), dec!(1)));
    let (state, _, recorder) = run_recorded(&env, &[buy_two(), RlAction::CrossHedge { size: 0 }]);
    let (hedge, _, buckets) = hedge_attribution(&env, &state, &recorder);
    assert_eq!(hedge.funding, dec!(0.01));
    assert_eq!(hedge.net_pnl, dec!(-1.5495) + dec!(0.01));
    assert_eq!(hedge.costs().unwrap(), dec!(0.5395));
    assert_eq!(state.lane().unwrap().counters().funding_paid, 1);
    // The payment at 3 000 ms is in the row at 3 000.
    let at = |t: u64| buckets.iter().find(|(a, _)| *a == t).unwrap().1.funding;
    assert_eq!(at(3000), dec!(0.01));
    assert_eq!(state.net(), dec!(2.801) + dec!(-1.5495) + dec!(0.01));
}

#[test]
fn test_a_hedge_lane_refuses_the_cancel_action() {
    let d = data();
    let refused = RlEnv::<'_, 3, 2>::new_hedged(
        &d.cfg,
        &d.quoted,
        &d.lane,
        &d.hedge,
        grid(),
        vec![1, 2],
        hedge_scales(),
        RlConfig::new(scales(), dec!(1)).with_cancel_action(),
    );
    assert!(matches!(refused, Err(e) if e.to_string().contains("cancel action")));
}

#[test]
fn test_a_hedge_lane_refuses_quote_lives_at_construction() {
    let d = data();
    let refused = RlEnv::<'_, 3, 2>::new_hedged(
        &d.cfg,
        &d.quoted,
        &d.lane,
        &d.hedge,
        grid(),
        vec![1, 2],
        hedge_scales(),
        RlConfig::new(scales(), dec!(1)).with_quote_lives(QuoteLives::new(&[5]).unwrap()),
    );
    assert!(matches!(refused, Err(e) if e.to_string().contains("quote lives")));
}
