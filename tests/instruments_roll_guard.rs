//! What may be quoted and reserved during a roll. The contracts of the guard's tests
//! are synthetic, with invented fills.

#![cfg(feature = "arbitrage")]

use financial_types::Action;
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::execution::{OrderManager, OrderRequest, OrderType, TimeInForce};
use market_maker_rs::instruments::futures::{
    FutureTerms, SettlementMethod, SettlementPrice, SettlementSource,
};
use market_maker_rs::instruments::roll::{
    RollError, RollEvent, RollFailurePolicy, RollFill, RollGuard, RollLeg, RollPlan, RollRequest,
    RollState,
};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
    Margining, TickSchedule,
};
use market_maker_rs::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use market_maker_rs::types::fee::FeeAsset;

const NEAR_EXPIRY: u64 = 1_000_000;
const FAR_EXPIRY: u64 = 2_000_000;
const DEADLINE: u64 = 999_000;
const NOW: u64 = 500_000;
const NEAR: &str = "NEAR";
const FAR: &str = "FAR";

fn asset(code: &str) -> AssetCode {
    AssetCode::new(code).unwrap()
}

fn source() -> SettlementSource {
    SettlementSource::new("the settlement source").unwrap()
}

fn future(name: &str, expiry_ms: u64) -> FutureTerms {
    FutureTerms::new(
        InstrumentSpec::new(InstrumentDraft {
            id: InstrumentId::new(name).unwrap(),
            kind: ContractKind::Future,
            margining: Margining::Linear,
            underlying: asset("BTC"),
            price_asset: asset("USDT"),
            settlement_asset: asset("USDT"),
            size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
            price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
            lot: dec!(1),
            expiry_ms: Some(expiry_ms),
            option: None,
        })
        .unwrap(),
        SettlementMethod::Cash,
        source(),
    )
    .unwrap()
}

fn plan_of(near_position: Decimal, far_position: Decimal, policy: RollFailurePolicy) -> RollPlan {
    RollPlan::new(RollRequest {
        near: future(NEAR, NEAR_EXPIRY),
        far: future(FAR, FAR_EXPIRY),
        near_position,
        far_position,
        near_price: dec!(60000),
        far_price: dec!(61000),
        policy,
        deadline_ms: DEADLINE,
        created_at_ms: NOW,
    })
    .unwrap()
}

/// Long 10 near, with room in the exposure limit.
fn long_plan() -> RollPlan {
    plan_of(dec!(10), dec!(0), RollFailurePolicy::Abort)
}

fn guard_of(plan: &RollPlan) -> RollGuard {
    RollGuard::new(plan, dec!(1000)).unwrap()
}

fn ask_at(
    guard: &mut RollGuard,
    id: &str,
    instrument: &str,
    side: Action,
    quantity: Decimal,
    at_ms: u64,
) -> PretradeDecision {
    guard.check(
        &PretradeRequest::new(
            id,
            side,
            Some(dec!(60000)),
            quantity,
            OrderPurpose::Quote,
            at_ms,
        )
        .with_instrument(instrument),
    )
}

fn ask(
    guard: &mut RollGuard,
    id: &str,
    instrument: &str,
    side: Action,
    quantity: Decimal,
) -> PretradeDecision {
    ask_at(guard, id, instrument, side, quantity, NOW)
}

fn reduced(quantity: Decimal) -> impl Fn(&PretradeDecision) -> bool {
    move |d| matches!(d, PretradeDecision::Reduced { quantity: q, .. } if *q == quantity)
}

fn order_fill(order: &str, exec: &str, side: Action, quantity: Decimal, at_ms: u64) -> OrderEvent {
    OrderEvent::Fill {
        order_id: order.to_string(),
        execution_id: exec.to_string(),
        side,
        price: dec!(60000),
        quantity,
        cumulative_quantity: quantity,
        fee: dec!(0),
        fee_asset: FeeAsset::Quote,
        timestamp: at_ms,
    }
}

fn plan_fill(plan: &RollPlan, leg: RollLeg, id: &str, contracts: Decimal) -> RollEvent {
    RollEvent::Fill(RollFill {
        leg,
        attempt: Some(0),
        exec_id: id.to_string(),
        side: plan.side(leg),
        contracts,
        cum_contracts: contracts,
        price: dec!(60000),
        fee: dec!(0),
        fee_asset: asset("USDT"),
        at_ms: NOW + 1,
    })
}

#[test]
fn the_near_contract_closes_only_without_flipping() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(4)),
        PretradeDecision::Accepted
    );
    // Adding is refused, and a larger sale is reduced to what is left.
    assert!(ask(&mut guard, "b", NEAR, Action::Buy, dec!(1)).is_rejected());
    assert!(reduced(dec!(6))(&ask(
        &mut guard,
        "c",
        NEAR,
        Action::Sell,
        dec!(7)
    )));
    assert!(ask(&mut guard, "d", NEAR, Action::Sell, dec!(1)).is_rejected());
    // A short mirrors it.
    let short = plan_of(dec!(-10), dec!(0), RollFailurePolicy::Abort);
    let mut guard = guard_of(&short);
    assert!(ask(&mut guard, "a", NEAR, Action::Sell, dec!(1)).is_rejected());
    assert!(reduced(dec!(10))(&ask(
        &mut guard,
        "b",
        NEAR,
        Action::Buy,
        dec!(11)
    )));
    // What was approved is what is live: a reduced order counts as reduced.
    let mut approved = guard_of(&long_plan());
    assert_eq!(
        ask(&mut approved, "a", NEAR, Action::Sell, dec!(4)),
        PretradeDecision::Accepted
    );
    assert!(reduced(dec!(6))(&ask(
        &mut approved,
        "b",
        NEAR,
        Action::Sell,
        dec!(7)
    )));
    approved
        .on_order_event(&OrderEvent::Canceled {
            order_id: "a".to_string(),
        })
        .unwrap();
    assert_eq!(
        ask(&mut approved, "c", NEAR, Action::Sell, dec!(4)),
        PretradeDecision::Accepted
    );
    // A position off the lot reduces on the lot.
    let off_lot = plan_of(dec!(2.5), dec!(0), RollFailurePolicy::Abort);
    let mut guard = guard_of(&off_lot);
    assert!(reduced(dec!(2))(&ask(
        &mut guard,
        "a",
        NEAR,
        Action::Sell,
        dec!(5)
    )));
    // An identifier that found no room may come back with one that has it.
    let mut again = guard_of(&long_plan());
    assert!(ask(&mut again, "a", NEAR, Action::Buy, dec!(1)).is_rejected());
    assert_eq!(
        ask(&mut again, "a", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
    // Flat, there is nothing to close.
    let mut flat = guard_of(&long_plan());
    assert_eq!(
        ask(&mut flat, "s", NEAR, Action::Sell, dec!(10)),
        PretradeDecision::Accepted
    );
    flat.on_order_event(&order_fill("s", "e", Action::Sell, dec!(10), NOW))
        .unwrap();
    assert!(ask(&mut flat, "x", NEAR, Action::Sell, dec!(1)).is_rejected());
    assert!(ask(&mut flat, "y", NEAR, Action::Buy, dec!(1)).is_rejected());
}

#[test]
fn the_far_contract_is_capped_at_the_plan_from_either_sign() {
    // Short 3 on far and long 10 near: the target is long 7.
    let plan = plan_of(dec!(10), dec!(-3), RollFailurePolicy::Abort);
    let mut guard = guard_of(&plan);
    assert!(reduced(dec!(10))(&ask(
        &mut guard,
        "a",
        FAR,
        Action::Buy,
        dec!(11)
    )));
    assert!(ask(&mut guard, "b", FAR, Action::Buy, dec!(1)).is_rejected());
    // Below its start nothing is opened.
    assert!(ask(&mut guard, "c", FAR, Action::Sell, dec!(1)).is_rejected());
    guard
        .on_order_event(&order_fill("a", "e", Action::Buy, dec!(5), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Far), Ok(dec!(2)));
    // Back towards the start, through zero: 5, and the live buys are no help.
    assert!(reduced(dec!(5))(&ask(
        &mut guard,
        "d",
        FAR,
        Action::Sell,
        dec!(6)
    )));
    // Reducing goes down to zero, past the start of a position already held.
    let held = plan_of(dec!(10), dec!(3), RollFailurePolicy::Abort);
    let mut guard = guard_of(&held);
    assert_eq!(
        ask(&mut guard, "a", FAR, Action::Buy, dec!(2)),
        PretradeDecision::Accepted
    );
    guard
        .on_order_event(&order_fill("a", "e", Action::Buy, dec!(2), NOW))
        .unwrap();
    assert!(reduced(dec!(5))(&ask(
        &mut guard,
        "b",
        FAR,
        Action::Sell,
        dec!(6)
    )));
    // A short plan mirrors the cap.
    let short = plan_of(dec!(-10), dec!(0), RollFailurePolicy::Abort);
    let mut guard = guard_of(&short);
    assert!(ask(&mut guard, "a", FAR, Action::Buy, dec!(1)).is_rejected());
    assert!(reduced(dec!(10))(&ask(
        &mut guard,
        "b",
        FAR,
        Action::Sell,
        dec!(11)
    )));
}

#[test]
fn the_exposure_limit_binds_the_side_that_adds_and_only_it() {
    let plan = long_plan();
    assert_eq!(
        RollGuard::new(&plan, dec!(9.99)),
        Err(RollError::InvalidExposureLimit {
            max_exposure: dec!(9.99),
            exposure: dec!(10),
        })
    );
    // A tight limit: far can only follow what near has filled, as a live sale of near
    // may not fill.
    let mut tight = RollGuard::new(&plan, dec!(10)).unwrap();
    assert_eq!(
        ask(&mut tight, "n", NEAR, Action::Sell, dec!(10)),
        PretradeDecision::Accepted
    );
    assert!(ask(&mut tight, "a", FAR, Action::Buy, dec!(1)).is_rejected());
    tight
        .on_order_event(&order_fill("n", "e", Action::Sell, dec!(3), NOW))
        .unwrap();
    assert!(reduced(dec!(3))(&ask(
        &mut tight,
        "b",
        FAR,
        Action::Buy,
        dec!(4)
    )));
    assert!(ask(&mut tight, "c", FAR, Action::Buy, dec!(1)).is_rejected());
    // A wider one lets far lead by that much, floored to the lot.
    let mut wide = RollGuard::new(&plan, dec!(12.5)).unwrap();
    assert!(reduced(dec!(2))(&ask(
        &mut wide,
        "a",
        FAR,
        Action::Buy,
        dec!(3)
    )));

    // Past the limit, by a far overfill: what reduces passes on both legs.
    let mut past = RollGuard::new(&plan, dec!(15)).unwrap();
    assert_eq!(
        ask(&mut past, "a", FAR, Action::Buy, dec!(5)),
        PretradeDecision::Accepted
    );
    past.on_order_event(&order_fill("a", "e", Action::Buy, dec!(12), NOW))
        .unwrap();
    assert_eq!(past.position(RollLeg::Far), Ok(dec!(12)));
    assert!(ask(&mut past, "b", FAR, Action::Buy, dec!(1)).is_rejected());
    assert_eq!(
        ask(&mut past, "c", FAR, Action::Sell, dec!(12)),
        PretradeDecision::Accepted
    );
    assert_eq!(
        ask(&mut past, "d", NEAR, Action::Sell, dec!(10)),
        PretradeDecision::Accepted
    );

    // The same on the short side.
    let short = plan_of(dec!(-10), dec!(0), RollFailurePolicy::Abort);
    let mut past = RollGuard::new(&short, dec!(15)).unwrap();
    assert_eq!(
        ask(&mut past, "a", FAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    past.on_order_event(&order_fill("a", "e", Action::Sell, dec!(12), NOW))
        .unwrap();
    assert!(ask(&mut past, "b", FAR, Action::Sell, dec!(1)).is_rejected());
    assert_eq!(
        ask(&mut past, "c", FAR, Action::Buy, dec!(12)),
        PretradeDecision::Accepted
    );
    assert_eq!(
        ask(&mut past, "d", NEAR, Action::Buy, dec!(10)),
        PretradeDecision::Accepted
    );
}

#[test]
fn the_window_closes_at_the_deadline_and_never_reopens() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask_at(&mut guard, "a", NEAR, Action::Sell, dec!(1), DEADLINE - 1),
        PretradeDecision::Accepted
    );
    assert!(ask_at(&mut guard, "b", NEAR, Action::Sell, dec!(1), DEADLINE).is_rejected());
    // A stale request does not reopen it, on either leg, though it would reduce.
    assert!(ask_at(&mut guard, "c", NEAR, Action::Sell, dec!(1), NOW).is_rejected());
    assert!(ask_at(&mut guard, "d", FAR, Action::Buy, dec!(1), NOW).is_rejected());
    assert_eq!(guard.latest_ms(), DEADLINE);
    // What the plan saw carries over, at a sync and at the construction, with no fill.
    let mut seen = long_plan();
    seen.apply(&RollEvent::Tick { at_ms: DEADLINE }).unwrap();
    let mut synced = guard_of(&long_plan());
    synced.sync(&seen).unwrap();
    assert_eq!(synced.latest_ms(), DEADLINE);
    assert!(ask(&mut synced, "a", NEAR, Action::Sell, dec!(1)).is_rejected());
    let mut built = guard_of(&seen);
    assert!(ask(&mut built, "a", NEAR, Action::Sell, dec!(1)).is_rejected());
    for event in [
        RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: DEADLINE,
        },
        RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: DEADLINE,
        },
    ] {
        let mut plan = long_plan();
        plan.apply(&event).unwrap();
        let mut guard = guard_of(&plan);
        assert!(ask(&mut guard, "a", NEAR, Action::Sell, dec!(1)).is_rejected());
    }
    // A fill that executed earlier than the time seen does not take it back.
    let mut back = long_plan();
    for (id, at_ms) in [("n1", DEADLINE), ("n2", NOW)] {
        let event = match plan_fill(&back, RollLeg::Near, id, dec!(1)) {
            RollEvent::Fill(fill) => RollEvent::Fill(RollFill {
                at_ms,
                attempt: None,
                ..fill
            }),
            other => other,
        };
        back.apply(&event).unwrap();
    }
    assert_eq!(guard_of(&back).latest_ms(), DEADLINE);
    // Market data and fills move the time too.
    let mut by_data = guard_of(&plan);
    by_data.on_market_data(dec!(60000), DEADLINE);
    assert!(ask(&mut by_data, "a", NEAR, Action::Sell, dec!(1)).is_rejected());
    let mut by_fill = guard_of(&plan);
    assert_eq!(
        ask(&mut by_fill, "a", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
    by_fill
        .on_order_event(&order_fill("a", "e", Action::Sell, dec!(1), DEADLINE))
        .unwrap();
    assert_eq!(by_fill.position(RollLeg::Near), Ok(dec!(9)));
    assert!(ask(&mut by_fill, "b", NEAR, Action::Sell, dec!(1)).is_rejected());
}

#[test]
fn what_cannot_be_judged_is_rejected() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    let anonymous =
        PretradeRequest::new("a", Action::Sell, None, dec!(1), OrderPurpose::Quote, NOW);
    assert!(guard.check(&anonymous).is_rejected());
    assert!(ask(&mut guard, "b", "OTHER", Action::Sell, dec!(1)).is_rejected());
    assert!(ask(&mut guard, "c", NEAR, Action::Other, dec!(1)).is_rejected());
    assert!(ask(&mut guard, "d", NEAR, Action::Sell, dec!(0)).is_rejected());
    assert!(ask(&mut guard, "e", NEAR, Action::Sell, dec!(-1)).is_rejected());
    assert_eq!(
        ask(&mut guard, "f", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
    assert!(ask(&mut guard, "f", NEAR, Action::Sell, dec!(1)).is_rejected());
    // A rejected identifier may come back.
    assert_eq!(
        ask(&mut guard, "b", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
}

#[test]
fn the_lifecycle_releases_and_a_late_fill_still_counts() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(10)),
        PretradeDecision::Accepted
    );
    assert!(ask(&mut guard, "x", NEAR, Action::Sell, dec!(1)).is_rejected());
    // Asked to cancel, it is still live.
    let cancel = |id: &str| OrderEvent::CancelRequested {
        order_id: id.to_string(),
    };
    guard.on_order_event(&cancel("a")).unwrap();
    assert!(ask(&mut guard, "y", NEAR, Action::Sell, dec!(1)).is_rejected());
    guard
        .on_order_event(&OrderEvent::Canceled {
            order_id: "a".to_string(),
        })
        .unwrap();
    assert_eq!(
        ask(&mut guard, "b", NEAR, Action::Sell, dec!(4)),
        PretradeDecision::Accepted
    );
    guard
        .on_order_event(&OrderEvent::VenueRejected {
            order_id: "b".to_string(),
        })
        .unwrap();
    // A fill after the cancel keeps its leg and moves the position; nothing is live.
    guard
        .on_order_event(&order_fill("a", "e1", Action::Sell, dec!(3), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(7)));
    assert_eq!(
        ask(&mut guard, "c", NEAR, Action::Sell, dec!(7)),
        PretradeDecision::Accepted
    );
    // Overfilled, it moves by what it was; live stays at zero, not negative.
    guard
        .on_order_event(&order_fill("c", "e2", Action::Sell, dec!(9), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(-2)));
    assert_eq!(
        ask(&mut guard, "d", NEAR, Action::Buy, dec!(2)),
        PretradeDecision::Accepted
    );
    // An overfilled order does not make room for the others of its direction.
    let mut other = guard_of(&long_plan());
    assert_eq!(
        ask(&mut other, "a", NEAR, Action::Sell, dec!(2)),
        PretradeDecision::Accepted
    );
    assert_eq!(
        ask(&mut other, "b", NEAR, Action::Sell, dec!(3)),
        PretradeDecision::Accepted
    );
    other
        .on_order_event(&order_fill("a", "e", Action::Sell, dec!(4), NOW))
        .unwrap();
    // Position 6, with 3 live: 3 of room, not the 5 a negative remainder would give.
    assert!(reduced(dec!(3))(&ask(
        &mut other,
        "c",
        NEAR,
        Action::Sell,
        dec!(5)
    )));
    // A cancel that came too late, and a venue accept, change nothing.
    for event in [
        OrderEvent::CancelRejected {
            order_id: "d".to_string(),
        },
        OrderEvent::Accepted {
            order_id: "d".to_string(),
        },
    ] {
        guard.on_order_event(&event).unwrap();
    }
    assert!(ask(&mut guard, "z", NEAR, Action::Buy, dec!(1)).is_rejected());
    // An order it never saw is ignored.
    guard
        .on_order_event(&order_fill("never", "e3", Action::Sell, dec!(1), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(-2)));
}

#[test]
fn a_fill_is_one_per_leg_and_execution() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "n", NEAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    assert_eq!(
        ask(&mut guard, "f", FAR, Action::Buy, dec!(5)),
        PretradeDecision::Accepted
    );
    assert_eq!(
        ask(&mut guard, "n2", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
    // The same identifier on both contracts is two trades.
    guard
        .on_order_event(&order_fill("n", "42", Action::Sell, dec!(2), NOW))
        .unwrap();
    guard
        .on_order_event(&order_fill("f", "42", Action::Buy, dec!(2), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    assert_eq!(guard.position(RollLeg::Far), Ok(dec!(2)));
    // A repeat, and a trade reassigned to another order, count once.
    guard
        .on_order_event(&order_fill("n", "42", Action::Sell, dec!(2), NOW))
        .unwrap();
    guard
        .on_order_event(&order_fill("n2", "42", Action::Sell, dec!(2), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    // Live sells are 3 left of n and 1 of n2, whose fill did not count, so 4 can go.
    assert!(reduced(dec!(4))(&ask(
        &mut guard,
        "n3",
        NEAR,
        Action::Sell,
        dec!(5)
    )));
    // Other content is a fault that stops everything until a reset.
    assert!(
        guard
            .on_order_event(&order_fill("n", "42", Action::Sell, dec!(3), NOW))
            .is_err()
    );
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    assert!(ask(&mut guard, "o", FAR, Action::Buy, dec!(1)).is_rejected());
    guard.sync(&plan).unwrap();
    assert!(ask(&mut guard, "p", FAR, Action::Buy, dec!(1)).is_rejected());
    guard.reset();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(10)));
    assert_eq!(
        ask(&mut guard, "n", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
}

#[test]
fn more_ticks_on_an_old_branch_cannot_restore_working() {
    let initial = long_plan();
    let mut plan = initial.clone();
    plan.apply(&RollEvent::Rejected {
        leg: RollLeg::Far,
        attempt: 0,
        at_ms: NOW + 2,
    })
    .unwrap();
    let mut guard = guard_of(&initial);
    guard.sync(&plan).unwrap();
    assert!(ask(&mut guard, "a", FAR, Action::Buy, dec!(1)).is_rejected());
    let mut fork = initial;
    for at_ms in [NOW + 3, NOW + 4] {
        fork.apply(&RollEvent::Tick { at_ms }).unwrap();
    }
    assert_eq!(fork.state(), RollState::Working);
    assert_eq!(guard.sync(&fork), Err(RollError::StalePlan));
    assert!(ask_at(&mut guard, "b", FAR, Action::Buy, dec!(1), NOW + 4).is_rejected());
}

#[test]
fn a_sync_cannot_switch_the_failure_policy_with_the_same_history() {
    // Both see the same events, and only the second may add to near as it unwinds.
    let aborting = in_state(RollState::Aborting, RollFailurePolicy::Abort);
    let unwinding = in_state(RollState::Unwinding, RollFailurePolicy::Unwind);
    let mut guard = guard_of(&long_plan());
    guard.sync(&aborting).unwrap();
    assert!(ask(&mut guard, "a", NEAR, Action::Buy, dec!(1)).is_rejected());
    let before = guard.clone();
    assert_eq!(guard.sync(&unwinding), Err(RollError::OtherPlan));
    assert_eq!(guard, before);
    assert!(ask(&mut guard, "b", NEAR, Action::Buy, dec!(1)).is_rejected());
    // A different time of creation is another plan as well.
    let later = RollPlan::new(RollRequest {
        near: future(NEAR, NEAR_EXPIRY),
        far: future(FAR, FAR_EXPIRY),
        near_position: dec!(10),
        far_position: dec!(0),
        near_price: dec!(60000),
        far_price: dec!(61000),
        policy: RollFailurePolicy::Abort,
        deadline_ms: DEADLINE,
        created_at_ms: NOW + 1,
    })
    .unwrap();
    assert_eq!(guard.sync(&later), Err(RollError::OtherPlan));
    assert_eq!(guard, before);
}

#[test]
fn a_fill_that_does_not_fit_its_order_stops_everything() {
    // Not a quantity at all: not counted.
    for quantity in [dec!(0), dec!(-1)] {
        let mut guard = guard_of(&long_plan());
        assert_eq!(
            ask(&mut guard, "a", NEAR, Action::Sell, dec!(5)),
            PretradeDecision::Accepted
        );
        assert!(
            guard
                .on_order_event(&order_fill("a", "e", Action::Sell, quantity, NOW))
                .is_err()
        );
        assert_eq!(guard.position(RollLeg::Near), Ok(dec!(10)));
        assert!(ask(&mut guard, "b", NEAR, Action::Sell, dec!(1)).is_rejected());
    }
    // On the other side of its order: a real execution, counted, and a fault.
    let mut guard = guard_of(&long_plan());
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    assert!(
        guard
            .on_order_event(&order_fill("a", "e", Action::Buy, dec!(2), NOW))
            .is_err()
    );
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(12)));
    assert!(ask(&mut guard, "b", NEAR, Action::Sell, dec!(1)).is_rejected());
    guard.reset();
    assert_eq!(
        ask(&mut guard, "b", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
}

#[test]
fn a_fill_on_another_action_stops_everything_and_a_sync_does_not_clear_it() {
    let plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
    assert!(
        guard
            .on_order_event(&order_fill("a", "e", Action::Other, dec!(1), NOW))
            .is_err()
    );
    assert!(ask(&mut guard, "b", NEAR, Action::Sell, dec!(1)).is_rejected());
    guard.sync(&plan).unwrap();
    assert!(ask(&mut guard, "c", NEAR, Action::Sell, dec!(1)).is_rejected());
    guard.reset();
    assert_eq!(
        ask(&mut guard, "c", NEAR, Action::Sell, dec!(1)),
        PretradeDecision::Accepted
    );
}

#[test]
fn a_sync_is_a_union_of_facts_in_any_order() {
    // The guard first, a lagging sync, then the plan catches up, in two steps.
    let mut plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    guard
        .on_order_event(&order_fill("a", "n1", Action::Sell, dec!(2), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    // Live is what is left of the order, unchanged by any of that: 3 of the 8 reduce.
    assert!(reduced(dec!(5))(&ask(
        &mut guard,
        "b",
        NEAR,
        Action::Sell,
        dec!(6)
    )));

    // The plan first, then the guard hears: the same position.
    let mut plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    guard
        .on_order_event(&order_fill("a", "n1", Action::Sell, dec!(2), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    assert!(reduced(dec!(5))(&ask(
        &mut guard,
        "b",
        NEAR,
        Action::Sell,
        dec!(6)
    )));

    // A fill the guard never saw arrives with the plan.
    let mut plan = long_plan();
    let mut guard = guard_of(&plan);
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Far, "f1", dec!(4)))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Far), Ok(dec!(0)));
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Far), Ok(dec!(4)));
}

#[test]
fn a_sync_refuses_what_is_older_or_different() {
    let mut plan = long_plan();
    let older = plan.clone();
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    let mut guard = guard_of(&older);
    guard.sync(&plan).unwrap();
    let before = guard.clone();
    assert_eq!(guard.sync(&older), Err(RollError::StalePlan));
    assert_eq!(guard, before);
    let other = RollPlan::new(RollRequest {
        near: future("OTHER", NEAR_EXPIRY),
        far: future(FAR, FAR_EXPIRY),
        near_position: dec!(10),
        far_position: dec!(0),
        near_price: dec!(60000),
        far_price: dec!(61000),
        policy: RollFailurePolicy::Abort,
        deadline_ms: DEADLINE,
        created_at_ms: NOW,
    })
    .unwrap();
    assert_eq!(guard.sync(&other), Err(RollError::OtherPlan));
    assert_eq!(guard, before);
    // The same contracts, but not the same plan.
    for different in [
        plan_of(dec!(9), dec!(0), RollFailurePolicy::Abort),
        plan_of(dec!(10), dec!(1), RollFailurePolicy::Abort),
        RollPlan::new(RollRequest {
            near: future(NEAR, NEAR_EXPIRY),
            far: future(FAR, FAR_EXPIRY),
            near_position: dec!(10),
            far_position: dec!(0),
            near_price: dec!(60000),
            far_price: dec!(61000),
            policy: RollFailurePolicy::Abort,
            deadline_ms: DEADLINE - 1,
            created_at_ms: NOW,
        })
        .unwrap(),
    ] {
        assert_eq!(guard.sync(&different), Err(RollError::OtherPlan));
        assert_eq!(guard, before);
    }

    // An older copy that differs by no fill: a rejection stopped the plan, and the copy
    // before it would start far opening again.
    let working = long_plan();
    let mut stopped = working.clone();
    stopped
        .apply(&RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: NOW + 2,
        })
        .unwrap();
    let mut guard = guard_of(&working);
    guard.sync(&stopped).unwrap();
    assert!(ask(&mut guard, "a", FAR, Action::Buy, dec!(1)).is_rejected());
    assert_eq!(guard.sync(&working), Err(RollError::StalePlan));
    assert!(ask(&mut guard, "b", FAR, Action::Buy, dec!(1)).is_rejected());
    // Repeating the same copy is fine, and a later one, but another copy of the same age
    // is not the same plan.
    guard.sync(&stopped).unwrap();
    let mut fork = working.clone();
    fork.apply(&RollEvent::Tick { at_ms: NOW + 5 }).unwrap();
    assert_eq!(guard.sync(&fork), Err(RollError::StalePlan));
    // A copy from before the rejection with more events since is no descendant either,
    // and would reopen far.
    fork.apply(&RollEvent::Tick { at_ms: NOW + 6 }).unwrap();
    fork.apply(&RollEvent::Tick { at_ms: NOW + 7 }).unwrap();
    assert_eq!(fork.state(), RollState::Working);
    assert_eq!(guard.sync(&fork), Err(RollError::StalePlan));
    assert!(ask(&mut guard, "c", FAR, Action::Buy, dec!(1)).is_rejected());
    let mut later = stopped.clone();
    later
        .apply(&RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: NOW + 3,
        })
        .unwrap();
    guard.sync(&later).unwrap();

    // A repeat does not count as an event, so a copy from before it is the same age.
    let mut repeated = long_plan();
    let fill = plan_fill(&repeated, RollLeg::Near, "n1", dec!(2));
    repeated.apply(&fill).unwrap();
    let copy = repeated.clone();
    repeated.apply(&fill).unwrap();
    assert_eq!(repeated, copy);

    // Another copy further along that lacks a fill or a settlement seen is no later.
    let start = long_plan();
    let mut with_fill = start.clone();
    with_fill
        .apply(&plan_fill(&start, RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    let mut only_ticks = start.clone();
    only_ticks
        .apply(&RollEvent::Tick { at_ms: NOW + 1 })
        .unwrap();
    only_ticks
        .apply(&RollEvent::Tick { at_ms: NOW + 2 })
        .unwrap();
    let mut guard = guard_of(&start);
    guard.sync(&with_fill).unwrap();
    assert_eq!(guard.sync(&only_ticks), Err(RollError::StalePlan));
    let mut with_settlement = start.clone();
    settle(&mut with_settlement, dec!(10));
    let mut guard = guard_of(&start);
    guard.sync(&with_settlement).unwrap();
    assert_eq!(guard.sync(&only_ticks), Err(RollError::StalePlan));

    // A plan without a settlement an earlier sync saw is older too.
    let near = future(NEAR, NEAR_EXPIRY);
    let unsettled = long_plan();
    let mut settled = unsettled.clone();
    settled
        .apply(&RollEvent::NearSettled {
            price: SettlementPrice::new(
                near.spec().id().clone(),
                source(),
                NEAR_EXPIRY,
                dec!(61000),
                NEAR_EXPIRY,
            )
            .unwrap(),
            contracts: dec!(10),
        })
        .unwrap();
    let mut guard = guard_of(&unsettled);
    guard.sync(&settled).unwrap();
    assert_eq!(guard.sync(&unsettled), Err(RollError::StalePlan));

    // The plan and the guard saw one execution with other content.
    let mut plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(5)),
        PretradeDecision::Accepted
    );
    guard
        .on_order_event(&order_fill("a", "n1", Action::Sell, dec!(1), NOW))
        .unwrap();
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    let before = guard.clone();
    assert_eq!(
        guard.sync(&plan),
        Err(RollError::ConflictingFill {
            exec_id: "n1".to_string(),
        })
    );
    assert_eq!(guard, before);
}

/// A settlement of 10 near contracts on `plan`.
fn settle(plan: &mut RollPlan, contracts: Decimal) {
    plan.apply(&RollEvent::NearSettled {
        price: SettlementPrice::new(
            InstrumentId::new(NEAR).unwrap(),
            source(),
            NEAR_EXPIRY,
            dec!(61000),
            NEAR_EXPIRY,
        )
        .unwrap(),
        contracts,
    })
    .unwrap();
}

fn late_near_fill(plan: &RollPlan) -> RollEvent {
    match plan_fill(plan, RollLeg::Near, "n1", dec!(2)) {
        RollEvent::Fill(fill) => RollEvent::Fill(RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill
        }),
        other => other,
    }
}

#[test]
fn a_settlement_closes_the_window_and_a_late_fill_agrees_in_either_order() {
    // The guard approved near sales, the plan settled, and a late fill closed at the
    // settlement price. The guard hears the settlement first, then the fill.
    let plan_before = long_plan();
    let mut guard = guard_of(&plan_before);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(2)),
        PretradeDecision::Accepted
    );
    let mut plan = plan_before.clone();
    settle(&mut plan, dec!(10));
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(0)));
    assert_eq!(guard.latest_ms(), NEAR_EXPIRY);
    // The window is closed, for far and for a stale near request alike.
    assert!(ask(&mut guard, "f", FAR, Action::Buy, dec!(1)).is_rejected());
    assert!(ask(&mut guard, "g", NEAR, Action::Sell, dec!(1)).is_rejected());
    guard
        .on_order_event(&order_fill("a", "n1", Action::Sell, dec!(2), NOW))
        .unwrap();
    // Provisional, as only the plan's record can say that it closed.
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(-2)));
    plan.apply(&late_near_fill(&plan.clone())).unwrap();
    assert_eq!(plan.closed_at_settlement(), dec!(-2));
    guard.sync(&plan).unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(0)));
    assert_eq!(guard.position(RollLeg::Far), Ok(dec!(0)));

    // The other order: the plan hears everything first, and the guard the same after.
    let mut plan = long_plan();
    let mut guard = guard_of(&plan);
    assert_eq!(
        ask(&mut guard, "a", NEAR, Action::Sell, dec!(2)),
        PretradeDecision::Accepted
    );
    settle(&mut plan, dec!(10));
    plan.apply(&late_near_fill(&plan.clone())).unwrap();
    guard.sync(&plan).unwrap();
    guard
        .on_order_event(&order_fill("a", "n1", Action::Sell, dec!(2), NOW))
        .unwrap();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(0)));
    assert!(ask(&mut guard, "f", FAR, Action::Buy, dec!(1)).is_rejected());
}

/// A plan in the given state, by what it observes.
fn in_state(state: RollState, policy: RollFailurePolicy) -> RollPlan {
    let mut plan = plan_of(dec!(10), dec!(0), policy);
    let reject_far = RollEvent::Rejected {
        leg: RollLeg::Far,
        attempt: 0,
        at_ms: NOW + 2,
    };
    let cancel_near = RollEvent::CancelAck {
        leg: RollLeg::Near,
        attempt: 0,
        cum_contracts: dec!(4),
        at_ms: NOW + 3,
    };
    let start = plan_fill(&plan, RollLeg::Near, "n1", dec!(4));
    match state {
        RollState::Working => {}
        RollState::Completed => {
            plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(10)))
                .unwrap();
            plan.apply(&plan_fill(&plan.clone(), RollLeg::Far, "f1", dec!(10)))
                .unwrap();
        }
        RollState::Aborting | RollState::Unwinding => {
            plan.apply(&start).unwrap();
            plan.apply(&reject_far).unwrap();
        }
        RollState::Aborted | RollState::UnwindRequested => {
            plan.apply(&start).unwrap();
            plan.apply(&reject_far).unwrap();
            plan.apply(&cancel_near).unwrap();
        }
        RollState::Faulted => {
            plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(1.5)))
                .unwrap();
        }
        _ => unreachable!(),
    }
    assert_eq!(plan.state(), state);
    plan
}

#[test]
fn once_the_plan_stops_far_opens_nothing_and_near_can_still_reduce() {
    for (state, policy) in [
        (RollState::Aborting, RollFailurePolicy::Abort),
        (RollState::Aborted, RollFailurePolicy::Abort),
        (RollState::Faulted, RollFailurePolicy::Abort),
        (RollState::Unwinding, RollFailurePolicy::Unwind),
        (RollState::UnwindRequested, RollFailurePolicy::Unwind),
    ] {
        let plan = in_state(state, policy);
        let near_now = plan.position(RollLeg::Near);
        let mut guard = guard_of(&plan_of(dec!(10), dec!(0), policy));
        guard.sync(&plan).unwrap();
        assert!(
            ask(&mut guard, "a", FAR, Action::Buy, dec!(1)).is_rejected(),
            "{state:?}"
        );
        assert_eq!(
            ask(&mut guard, "b", NEAR, Action::Sell, near_now.floor()),
            PretradeDecision::Accepted,
            "{state:?}"
        );
        // Near adds only while the plan unwinds, back towards its start of 10.
        let unwinding = matches!(state, RollState::Unwinding | RollState::UnwindRequested);
        let added = ask(&mut guard, "c", NEAR, Action::Buy, dec!(100));
        if unwinding {
            assert!(reduced(dec!(10) - near_now)(&added), "{state:?}: {added:?}");
        } else {
            assert!(added.is_rejected(), "{state:?}");
        }
    }
    // Completed, far has its target and can only come back; near closes what is left.
    let plan = in_state(RollState::Completed, RollFailurePolicy::Abort);
    let mut guard = guard_of(&long_plan());
    guard.sync(&plan).unwrap();
    assert!(ask(&mut guard, "a", FAR, Action::Buy, dec!(1)).is_rejected());
    assert_eq!(
        ask(&mut guard, "b", FAR, Action::Sell, dec!(10)),
        PretradeDecision::Accepted
    );
    assert!(ask(&mut guard, "c", NEAR, Action::Sell, dec!(1)).is_rejected());
    // Working, near does not add.
    let mut working = guard_of(&long_plan());
    assert!(ask(&mut working, "a", NEAR, Action::Buy, dec!(1)).is_rejected());
}

#[test]
fn reset_returns_to_the_construction_undoing_syncs() {
    let mut plan = long_plan();
    let at_start = plan.clone();
    let mut guard = guard_of(&at_start);
    plan.apply(&plan_fill(&plan.clone(), RollLeg::Near, "n1", dec!(2)))
        .unwrap();
    guard.sync(&plan).unwrap();
    guard.on_market_data(dec!(60000), DEADLINE);
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(8)));
    guard.reset();
    assert_eq!(guard.position(RollLeg::Near), Ok(dec!(10)));
    assert_eq!(guard.latest_ms(), NOW);
    assert_eq!(guard, guard_of(&at_start));
}

#[test]
fn the_plans_own_legs_fit_a_guard_on_the_live_path() {
    let plan = long_plan();
    let mut guard = RollGuard::new(&plan, dec!(20)).unwrap();
    let mut manager = OrderManager::with_defaults();
    let order = |symbol: &str, side, quantity| OrderRequest {
        symbol: symbol.to_string(),
        side,
        order_type: OrderType::Limit,
        quantity,
        price: Some(dec!(60000)),
        time_in_force: TimeInForce::GoodTilCancel,
        client_order_id: None,
    };
    for (id, symbol, leg) in [("near", NEAR, RollLeg::Near), ("far", FAR, RollLeg::Far)] {
        let decision = manager
            .register_order_checked(
                &order(symbol, plan.side(leg), plan.contracts(leg)),
                id.to_string(),
                NOW,
                OrderPurpose::Other,
                &mut guard,
            )
            .unwrap();
        assert_eq!(decision, PretradeDecision::Accepted, "{id}");
    }
    // A quote that would stack on top of them does not pass.
    let quote = manager
        .register_order_checked(
            &order(FAR, Action::Buy, dec!(1)),
            "quote".to_string(),
            NOW,
            OrderPurpose::Quote,
            &mut guard,
        )
        .unwrap();
    assert!(quote.is_rejected());
    assert_eq!(guard.description(), "roll guard from NEAR to FAR");
}

#[cfg(feature = "margin")]
mod reserves {
    use super::*;
    use std::collections::BTreeMap;

    use margin_rs::ledger::{Execution, Ledger};
    use margin_rs::margin_reservations::{BookedFill, MarginDecision};
    use margin_rs::risk::RiskGate;
    use margin_rs::rules::{TieredInverseIm, TieredLinearIm};
    use margin_rs::{
        AccountId, Asset, ExecutionId, InstrumentId as LedgerId, MarginError, OrderId,
        ReservationId, TransferId, VenueId,
    };
    use market_maker_rs::instruments::futures::FutureError;
    use market_maker_rs::instruments::margin::{MarginScales, margin_instrument};
    use market_maker_rs::instruments::reference::deribit;
    use market_maker_rs::instruments::roll::RollReserves;

    fn oid(id: &str) -> OrderId {
        OrderId::new(id).unwrap()
    }

    fn lid(id: &str) -> LedgerId {
        LedgerId::new(id).unwrap()
    }

    fn hold_id() -> ReservationId {
        ReservationId::new("roll").unwrap()
    }

    fn scales() -> MarginScales {
        MarginScales {
            price_scale: 2,
            contract_scale: 0,
        }
    }

    fn gate_of(terms: &[&FutureTerms], settlement: &str, cash: Decimal) -> RiskGate {
        let mut ledger = Ledger::default();
        ledger
            .register_asset(Asset::new(settlement).unwrap(), 8)
            .unwrap();
        for one in terms {
            ledger
                .register_instrument(margin_instrument(one.spec(), scales()).unwrap())
                .unwrap();
        }
        let mut gate = RiskGate::new(ledger);
        gate.deposit(
            TransferId::new("seed").unwrap(),
            Asset::new(settlement).unwrap(),
            cash,
        )
        .unwrap();
        gate
    }

    fn book(gate: &mut RiskGate, id: &str, instrument: &str, side: Action, contracts: Decimal) {
        gate.book_derivative_execution(
            &Execution::new(
                ExecutionId::new(id).unwrap(),
                lid(instrument),
                side,
                dec!(60000),
                contracts,
            )
            .with_executed_at(NOW),
        )
        .unwrap();
    }

    fn marks(near: Decimal, far: Decimal) -> BTreeMap<LedgerId, Decimal> {
        BTreeMap::from([(lid(NEAR), near), (lid(FAR), far)])
    }

    fn only_far(far: Decimal) -> BTreeMap<LedgerId, Decimal> {
        BTreeMap::from([(lid(FAR), far)])
    }

    fn reserves() -> RollReserves<TieredLinearIm> {
        RollReserves::new(
            TieredLinearIm::linear_v1(Asset::new("USDT").unwrap(), 8).unwrap(),
            AccountId::new("account").unwrap(),
            VenueId::new("venue").unwrap(),
            hold_id(),
            &future(NEAR, NEAR_EXPIRY),
            &future(FAR, FAR_EXPIRY),
        )
        .unwrap()
    }

    fn price() -> SettlementPrice {
        SettlementPrice::new(
            InstrumentId::new(NEAR).unwrap(),
            source(),
            NEAR_EXPIRY,
            dec!(61000),
            NEAR_EXPIRY,
        )
        .unwrap()
    }

    #[test]
    fn both_legs_are_margined_with_no_offset_and_the_settlement_leaves_the_hold() {
        let (near, far) = (future(NEAR, NEAR_EXPIRY), future(FAR, FAR_EXPIRY));
        let mut gate = gate_of(&[&near, &far], "USDT", dec!(200000));
        book(&mut gate, "open", NEAR, Action::Buy, dec!(10));
        let mut reserves = reserves();
        let at = marks(dec!(60000), dec!(60000));
        // 10% of 11 near at 60000, with its live buy, then of 5 far: the rule nets nothing.
        assert_eq!(
            reserves
                .reserve(
                    RollLeg::Near,
                    oid("n"),
                    Action::Buy,
                    dec!(1),
                    &mut gate,
                    &at
                )
                .unwrap(),
            MarginDecision::Reserved {
                required: dec!(66000),
                reserved: dec!(6000),
            }
        );
        assert_eq!(
            reserves
                .reserve(RollLeg::Far, oid("f"), Action::Buy, dec!(5), &mut gate, &at)
                .unwrap(),
            MarginDecision::Reserved {
                required: dec!(96000),
                reserved: dec!(30000),
            }
        );
        assert_eq!(gate.hold(&hold_id()), dec!(96000));
        // Before the ledger settled, this is refused and changes nothing.
        assert_eq!(
            reserves.near_settled(&mut gate, &at),
            Err(RollError::NearNotSettled)
        );
        assert_eq!(gate.hold(&hold_id()), dec!(96000));

        // Settling moves cash and closes the position, and the hold stands.
        let booked = near.settle_in_gate(&mut gate, &price()).unwrap();
        assert_eq!(booked.cashflow.contracts, dec!(10));
        assert_eq!(booked.cashflow.realized_pnl, dec!(10000));
        assert_eq!(gate.hold(&hold_id()), dec!(96000));
        assert_eq!(
            gate.ledger().position(&lid(NEAR)).unwrap().quantity,
            dec!(0)
        );
        // Now only far's mark is needed, and only far's order is left.
        assert_eq!(
            reserves.near_settled(&mut gate, &only_far(dec!(60000))),
            Ok(dec!(30000))
        );
        assert_eq!(gate.hold(&hold_id()), dec!(30000));
        // Repeating it only syncs, and far's order is still there.
        assert_eq!(
            reserves.near_settled(&mut gate, &only_far(dec!(60000))),
            Ok(dec!(30000))
        );
        // No new near reservations.
        assert_eq!(
            reserves.reserve(
                RollLeg::Near,
                oid("n2"),
                Action::Sell,
                dec!(1),
                &mut gate,
                &at
            ),
            Err(RollError::NearAlreadySettled)
        );
        // Far's order released at last.
        assert_eq!(reserves.release_on_cancel(&oid("f")), dec!(5));
        assert_eq!(
            reserves.sync_hold(&mut gate, &only_far(dec!(60000))),
            Ok(dec!(0))
        );
        gate.ledger().reconcile().unwrap();
    }

    #[test]
    fn a_failed_sync_changes_nothing_and_can_be_retried() {
        let (near, far) = (future(NEAR, NEAR_EXPIRY), future(FAR, FAR_EXPIRY));
        let mut gate = gate_of(&[&near, &far], "USDT", dec!(100000));
        book(&mut gate, "open", NEAR, Action::Buy, dec!(1));
        let mut reserves = reserves();
        let at = marks(dec!(60000), dec!(60000));
        assert!(matches!(
            reserves
                .reserve(
                    RollLeg::Near,
                    oid("n"),
                    Action::Buy,
                    dec!(1),
                    &mut gate,
                    &at
                )
                .unwrap(),
            MarginDecision::Reserved { .. }
        ));
        assert!(matches!(
            reserves
                .reserve(
                    RollLeg::Far,
                    oid("f"),
                    Action::Buy,
                    dec!(10),
                    &mut gate,
                    &at
                )
                .unwrap(),
            MarginDecision::Reserved { .. }
        ));
        assert_eq!(gate.hold(&hold_id()), dec!(72000));
        near.settle_in_gate(&mut gate, &price()).unwrap();
        // A mark far beyond the cash: the sync fails, and neither order nor hold moved.
        let high = only_far(dec!(6000000));
        assert!(matches!(
            reserves.near_settled(&mut gate, &high),
            Err(RollError::Future(FutureError::Ledger(_)))
        ));
        assert_eq!(gate.hold(&hold_id()), dec!(72000));
        // Near's order is still there: with no mark for it, a requirement cannot be made.
        assert!(matches!(
            reserves.sync_hold(&mut gate, &only_far(dec!(60000))),
            Err(RollError::Future(FutureError::Ledger(
                MarginError::MissingMark { .. }
            )))
        ));
        // The retry succeeds, and far is intact.
        assert_eq!(
            reserves.near_settled(&mut gate, &only_far(dec!(60000))),
            Ok(dec!(60000))
        );
        assert_eq!(gate.hold(&hold_id()), dec!(60000));
        assert_eq!(reserves.release_on_cancel(&oid("n")), dec!(0));
        assert_eq!(reserves.release_on_cancel(&oid("f")), dec!(10));
    }

    #[test]
    fn a_fill_after_the_settlement_or_the_cancel_stays_visible() {
        let (near, far) = (future(NEAR, NEAR_EXPIRY), future(FAR, FAR_EXPIRY));
        let mut gate = gate_of(&[&near, &far], "USDT", dec!(200000));
        book(&mut gate, "open", NEAR, Action::Buy, dec!(4));
        let mut reserves = reserves();
        let at = marks(dec!(60000), dec!(60000));
        reserves
            .reserve(
                RollLeg::Near,
                oid("n"),
                Action::Sell,
                dec!(2),
                &mut gate,
                &at,
            )
            .unwrap();
        near.settle_in_gate(&mut gate, &price()).unwrap();
        let empty = BTreeMap::new();
        reserves.near_settled(&mut gate, &empty).unwrap();
        // Executed before the expiry and booked now: the order closed, and the fill is
        // flagged, not refused, while the position stays flat.
        let late = ExecutionId::new("late").unwrap();
        gate.book_derivative_execution(
            &Execution::new(late.clone(), lid(NEAR), Action::Sell, dec!(60000), dec!(2))
                .with_executed_at(NEAR_EXPIRY - 1),
        )
        .unwrap();
        let applied = reserves
            .apply_booked_fill(&oid("n"), &late, &mut gate, &empty)
            .unwrap();
        assert!(
            matches!(
                applied,
                BookedFill::Applied {
                    remaining,
                    overfill: Some(overfill),
                    ..
                } if remaining == dec!(0) && overfill == dec!(2)
            ),
            "{applied:?}"
        );
        assert_eq!(
            gate.ledger().position(&lid(NEAR)).unwrap().quantity,
            dec!(0)
        );
        assert_eq!(
            reserves.apply_booked_fill(&oid("n"), &late, &mut gate, &empty),
            Ok(BookedFill::Duplicate)
        );
        // Far's fill, booked before its order is reported.
        reserves
            .reserve(
                RollLeg::Far,
                oid("f"),
                Action::Buy,
                dec!(2),
                &mut gate,
                &only_far(dec!(60000)),
            )
            .unwrap();
        book(&mut gate, "far", FAR, Action::Buy, dec!(1));
        let applied = reserves
            .apply_booked_fill(
                &oid("f"),
                &ExecutionId::new("far").unwrap(),
                &mut gate,
                &only_far(dec!(60000)),
            )
            .unwrap();
        assert!(matches!(
            applied,
            BookedFill::Applied { overfill: None, .. }
        ));
    }

    #[test]
    fn another_contract_under_the_name_is_refused() {
        let (near, far) = (future(NEAR, NEAR_EXPIRY), future(FAR, FAR_EXPIRY));
        let later_near = future(NEAR, NEAR_EXPIRY + 1);
        let mut gate = gate_of(&[&later_near, &far], "USDT", dec!(200000));
        let mut reserves = reserves();
        let at = marks(dec!(60000), dec!(60000));
        for result in [
            reserves.near_settled(&mut gate, &at).map(|_| ()),
            reserves
                .reserve(
                    RollLeg::Near,
                    oid("n"),
                    Action::Buy,
                    dec!(1),
                    &mut gate,
                    &at,
                )
                .map(|_| ()),
        ] {
            assert!(
                matches!(
                    result,
                    Err(RollError::Future(FutureError::OtherTermsInLedger { .. }))
                ),
                "{result:?}"
            );
        }
        let _ = near;
    }

    #[test]
    fn inverse_contracts_are_margined_in_their_settlement_asset() {
        let near = FutureTerms::new(
            deribit::btc_future(InstrumentId::new(NEAR).unwrap(), NEAR_EXPIRY).unwrap(),
            SettlementMethod::Cash,
            source(),
        )
        .unwrap();
        let far = FutureTerms::new(
            deribit::btc_future(InstrumentId::new(FAR).unwrap(), FAR_EXPIRY).unwrap(),
            SettlementMethod::Cash,
            source(),
        )
        .unwrap();
        let mut gate = gate_of(&[&near, &far], "BTC", dec!(10));
        let mut reserves = RollReserves::new(
            TieredInverseIm::inverse_v1(Asset::new("BTC").unwrap(), 8).unwrap(),
            AccountId::new("account").unwrap(),
            VenueId::new("venue").unwrap(),
            hold_id(),
            &near,
            &far,
        )
        .unwrap();
        let at = marks(dec!(60000), dec!(50000));
        // 1000 contracts of 10 USD at 50000 are 0.2 BTC, and 10% of it is 0.02 BTC.
        assert_eq!(
            reserves
                .reserve(
                    RollLeg::Far,
                    oid("f"),
                    Action::Buy,
                    dec!(1000),
                    &mut gate,
                    &at
                )
                .unwrap(),
            MarginDecision::Reserved {
                required: dec!(0.02),
                reserved: dec!(0.02),
            }
        );
        assert_eq!(gate.hold(&hold_id()), dec!(0.02));
    }
}
