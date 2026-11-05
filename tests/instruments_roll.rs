//! A plan to roll a position between dated futures, and its check against a ledger.
//!
//! The contracts of different sizes are synthetic. The inverse ones are Deribit's
//! `btc_future` terms, with invented expiries, prices and fills.

#![cfg(feature = "arbitrage")]

use financial_types::Action;
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::instruments::futures::{
    FutureError, FutureTerms, SettlementMethod, SettlementPrice, SettlementSource,
};
use market_maker_rs::instruments::reference::deribit;
use market_maker_rs::instruments::roll::{
    RollAction, RollError, RollEvent, RollFailurePolicy, RollFallback, RollFill, RollLeg, RollPlan,
    RollProgress, RollReport, RollRequest, RollState,
};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
    Margining, TickSchedule,
};

const NEAR_EXPIRY: u64 = 1_000_000;
const FAR_EXPIRY: u64 = 2_000_000;
const NOW: u64 = 500_000;
const SOURCE: &str = "the settlement source";

fn asset(code: &str) -> AssetCode {
    AssetCode::new(code).unwrap()
}

fn source() -> SettlementSource {
    SettlementSource::new(SOURCE).unwrap()
}

fn terms_of(spec: InstrumentSpec) -> FutureTerms {
    FutureTerms::new(spec, SettlementMethod::Cash, source()).unwrap()
}

/// A linear future on BTC settled in USDT, with `size` BTC per contract.
fn linear(name: &str, size: Decimal, lot: Decimal, expiry_ms: u64) -> FutureTerms {
    terms_of(
        InstrumentSpec::new(InstrumentDraft {
            id: InstrumentId::new(name).unwrap(),
            kind: ContractKind::Future,
            margining: Margining::Linear,
            underlying: asset("BTC"),
            price_asset: asset("USDT"),
            settlement_asset: asset("USDT"),
            size: ContractSize::base(size, asset("BTC")).unwrap(),
            price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
            lot,
            expiry_ms: Some(expiry_ms),
            option: None,
        })
        .unwrap(),
    )
}

fn inverse(name: &str, expiry_ms: u64) -> FutureTerms {
    terms_of(deribit::btc_future(InstrumentId::new(name).unwrap(), expiry_ms).unwrap())
}

/// Near and far of size one, on the same grid.
fn same_size() -> (FutureTerms, FutureTerms) {
    (
        linear("NEAR", dec!(1), dec!(1), NEAR_EXPIRY),
        linear("FAR", dec!(1), dec!(1), FAR_EXPIRY),
    )
}

fn request(near: FutureTerms, far: FutureTerms, near_position: Decimal) -> RollRequest {
    RollRequest {
        near,
        far,
        near_position,
        far_position: dec!(0),
        near_price: dec!(60000),
        far_price: dec!(61000),
        policy: RollFailurePolicy::Abort,
        deadline_ms: NEAR_EXPIRY - 1000,
        created_at_ms: NOW,
    }
}

/// A plan on the same size contracts, long 10.
fn long_ten() -> RollPlan {
    let (near, far) = same_size();
    RollPlan::new(request(near, far, dec!(10))).unwrap()
}

/// A fill on the plan's side of `leg`, as the first and only of its attempt.
fn fill(plan: &RollPlan, leg: RollLeg, id: &str, contracts: Decimal) -> RollFill {
    RollFill {
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
    }
}

fn apply(plan: &mut RollPlan, event: RollEvent) -> RollReport {
    plan.apply(&event).unwrap()
}

fn apply_fill(plan: &mut RollPlan, event: RollFill) -> RollReport {
    apply(plan, RollEvent::Fill(event))
}

fn settlement(near: &FutureTerms, price: Decimal, published_ms: u64) -> SettlementPrice {
    SettlementPrice::new(
        near.spec().id().clone(),
        source(),
        near.expiry_ms(),
        price,
        published_ms,
    )
    .unwrap()
}

#[test]
fn a_long_roll_preserves_the_underlying_across_contract_sizes() {
    // 1.25 BTC on contracts of 1 BTC with a lot of 0.001, onto contracts of 0.1 BTC
    // with a lot of 1: the grid is 0.1 BTC, so 1.2 BTC roll and 0.05 BTC stay.
    let near = linear("NEAR", dec!(1), dec!(0.001), NEAR_EXPIRY);
    let far = linear("FAR", dec!(0.1), dec!(1), FAR_EXPIRY);
    let mut plan = RollPlan::new(request(near, far, dec!(1.25))).unwrap();
    assert_eq!(plan.grid(), dec!(0.1));
    assert_eq!(plan.package(), dec!(1.2));
    assert_eq!(plan.uncovered(), dec!(0.05));
    assert_eq!(plan.contracts(RollLeg::Near), dec!(1.2));
    assert_eq!(plan.contracts(RollLeg::Far), dec!(12));
    assert_eq!(plan.side(RollLeg::Near), Action::Sell);
    assert_eq!(plan.side(RollLeg::Far), Action::Buy);
    assert_eq!(plan.exposure().unwrap(), dec!(1.25));
    assert_eq!(plan.state(), RollState::Working);

    let near_fill = fill(&plan, RollLeg::Near, "n1", dec!(1.2));
    let report = apply_fill(&mut plan, near_fill);
    // One leg alone: the gap shows at once.
    assert_eq!(report.near_position, dec!(0.05));
    assert_eq!(report.far_position, dec!(0));
    assert_eq!(report.exposure, dec!(0.05));
    assert_eq!(report.exposure_gap, dec!(-1.2));
    assert_eq!(report.state, RollState::Working);
    assert!(report.refused.is_none());

    let far_fill = fill(&plan, RollLeg::Far, "f1", dec!(12));
    let report = apply_fill(&mut plan, far_fill);
    assert_eq!(report.near_position, dec!(0.05));
    assert_eq!(report.far_position, dec!(12));
    assert_eq!(report.exposure, dec!(1.25));
    assert_eq!(report.exposure_gap, dec!(0));
    assert_eq!(report.state, RollState::Completed);
}

#[test]
fn a_short_inverse_roll_preserves_the_contract_value() {
    // Short 1000 contracts of 10 USD: its value in USD stays, whatever the price.
    let mut plan = RollPlan::new(request(
        inverse("BTC-NEAR", NEAR_EXPIRY),
        inverse("BTC-FAR", FAR_EXPIRY),
        dec!(-1000),
    ))
    .unwrap();
    assert_eq!(plan.side(RollLeg::Near), Action::Buy);
    assert_eq!(plan.side(RollLeg::Far), Action::Sell);
    assert_eq!(plan.exposure().unwrap(), dec!(-10000));
    // The primitive's figures are in value on an inverse contract: 10 USD a lot.
    assert_eq!(plan.grid(), dec!(10));
    assert_eq!(plan.package(), dec!(10000));
    assert_eq!(plan.uncovered(), dec!(0));

    let near = RollFill {
        cum_contracts: dec!(500),
        fee_asset: asset("BTC"),
        fee: dec!(0.0001),
        ..fill(&plan, RollLeg::Near, "n1", dec!(500))
    };
    let report = apply_fill(&mut plan, near);
    assert_eq!(report.near_position, dec!(-500));
    assert_eq!(report.exposure, dec!(-5000));
    assert_eq!(report.exposure_gap, dec!(5000));
    assert_eq!(
        plan.progress().unwrap(),
        Some(RollProgress {
            matched: dec!(0),
            near_excess: dec!(500),
            far_excess: dec!(0),
        })
    );
    let far = fill(&plan, RollLeg::Far, "f1", dec!(500));
    let report = apply_fill(&mut plan, far);
    assert_eq!(report.exposure_gap, dec!(0));
    assert_eq!(report.far_position, dec!(-500));
    assert_eq!(
        plan.progress().unwrap(),
        Some(RollProgress {
            matched: dec!(5000),
            near_excess: dec!(0),
            far_excess: dec!(0),
        })
    );
}

#[test]
fn a_far_position_of_the_other_sign_crosses_zero_with_the_right_gap() {
    let (near, far) = same_size();
    let mut plan = RollPlan::new(RollRequest {
        far_position: dec!(-3),
        ..request(near, far, dec!(10))
    })
    .unwrap();
    assert_eq!(plan.exposure().unwrap(), dec!(7));
    assert_eq!(plan.position(RollLeg::Far), dec!(-3));
    let first = fill(&plan, RollLeg::Far, "f1", dec!(4));
    let report = apply_fill(&mut plan, first);
    // Through zero, from short 3 to long 1: 4 more than the start.
    assert_eq!(report.far_position, dec!(1));
    assert_eq!(report.exposure, dec!(11));
    assert_eq!(report.exposure_gap, dec!(4));
    let second = RollFill {
        cum_contracts: dec!(10),
        ..fill(&plan, RollLeg::Far, "f2", dec!(6))
    };
    apply_fill(&mut plan, second);
    let near_fill = fill(&plan, RollLeg::Near, "n1", dec!(10));
    let report = apply_fill(&mut plan, near_fill);
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(report.far_position, dec!(7));
    assert_eq!(report.exposure, dec!(7));
    assert_eq!(report.exposure_gap, dec!(0));
    assert_eq!(report.state, RollState::Completed);
}

#[test]
fn a_roll_needs_two_contracts_of_one_product_in_order_and_in_time() {
    let (near, far) = same_size();
    let eth = terms_of({
        let mut draft = InstrumentDraft::from(far.spec().clone());
        draft.underlying = asset("ETH");
        draft.size = ContractSize::base(dec!(1), asset("ETH")).unwrap();
        InstrumentSpec::new(draft).unwrap()
    });
    assert!(matches!(
        RollPlan::new(request(near.clone(), eth, dec!(10))),
        Err(RollError::Future(FutureError::OtherProduct { .. }))
    ));
    assert!(matches!(
        RollPlan::new(request(near.clone(), near.clone(), dec!(10))),
        Err(RollError::Future(FutureError::DuplicateContract { .. }))
    ));
    let same_expiry = linear("FAR", dec!(1), dec!(1), NEAR_EXPIRY);
    assert!(matches!(
        RollPlan::new(request(near.clone(), same_expiry, dec!(10))),
        Err(RollError::FarNotLater { .. })
    ));
    let earlier = linear("FAR", dec!(1), dec!(1), NEAR_EXPIRY - 1);
    assert!(matches!(
        RollPlan::new(request(near.clone(), earlier, dec!(10))),
        Err(RollError::FarNotLater { .. })
    ));

    // The near contract must be trading: a millisecond before its expiry, and not at it.
    let at = |created_at_ms, deadline_ms| RollRequest {
        created_at_ms,
        deadline_ms,
        ..request(near.clone(), far.clone(), dec!(10))
    };
    assert!(RollPlan::new(at(NEAR_EXPIRY - 1, NEAR_EXPIRY)).is_ok());
    assert!(matches!(
        RollPlan::new(at(NEAR_EXPIRY, NEAR_EXPIRY + 1)),
        Err(RollError::NearExpired { .. })
    ));
    // The deadline: after the plan, and by the near contract's expiry.
    assert!(matches!(
        RollPlan::new(at(NOW, NOW)),
        Err(RollError::InvalidDeadline { .. })
    ));
    assert!(RollPlan::new(at(NOW, NOW + 1)).is_ok());
    assert!(RollPlan::new(at(NOW, NEAR_EXPIRY)).is_ok());
    assert!(matches!(
        RollPlan::new(at(NOW, NEAR_EXPIRY + 1)),
        Err(RollError::InvalidDeadline { .. })
    ));
    for price in [dec!(0), dec!(-1)] {
        assert_eq!(
            RollPlan::new(RollRequest {
                near_price: price,
                ..request(near.clone(), far.clone(), dec!(10))
            }),
            Err(RollError::InvalidPrice { price })
        );
        assert_eq!(
            RollPlan::new(RollRequest {
                far_price: price,
                ..request(near.clone(), far.clone(), dec!(10))
            }),
            Err(RollError::InvalidPrice { price })
        );
    }
    let beyond = i64::MAX as u64 + 1;
    let late_far = linear("FAR", dec!(1), dec!(1), beyond);
    assert_eq!(
        RollPlan::new(request(near.clone(), late_far, dec!(10))),
        Err(RollError::TimeOutOfRange {
            field: "expiry_ms",
            value: beyond,
        })
    );
    // Nothing on the grid, including a flat position.
    for position in [dec!(0), dec!(0.5), dec!(-0.5)] {
        assert!(matches!(
            RollPlan::new(request(near.clone(), far.clone(), position)),
            Err(RollError::NothingToRoll { .. })
        ));
    }
}

#[test]
fn the_grid_is_the_least_common_multiple_of_both_lots() {
    // Lots of 0.4 and 0.6 BTC, neither dividing the other: a grid of 1.2 BTC.
    let near = linear("NEAR", dec!(0.1), dec!(4), NEAR_EXPIRY);
    let far = linear("FAR", dec!(0.2), dec!(3), FAR_EXPIRY);
    let plan = RollPlan::new(request(near, far, dec!(30))).unwrap();
    assert_eq!(plan.grid(), dec!(1.2));
    assert_eq!(plan.package(), dec!(2.4));
    assert_eq!(plan.uncovered(), dec!(0.6));
    assert_eq!(plan.contracts(RollLeg::Near), dec!(24));
    assert_eq!(plan.contracts(RollLeg::Far), dec!(12));
}

#[test]
fn a_position_off_the_lot_is_a_fact_and_what_does_not_fit_stays() {
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(near, far, dec!(-2.5))).unwrap();
    assert_eq!(plan.package(), dec!(2));
    assert_eq!(plan.uncovered(), dec!(0.5));
    assert_eq!(plan.exposure().unwrap(), dec!(-2.5));
    let near_fill = fill(&plan, RollLeg::Near, "n", dec!(2));
    apply_fill(&mut plan, near_fill);
    let far_fill = fill(&plan, RollLeg::Far, "f", dec!(2));
    let report = apply_fill(&mut plan, far_fill);
    assert_eq!(report.near_position, dec!(-0.5));
    assert_eq!(report.far_position, dec!(-2));
    assert_eq!(report.exposure_gap, dec!(0));
    assert_eq!(report.state, RollState::Completed);
}

/// Checks that the fill was refused by the intents and is recorded all the same.
fn assert_recorded_and_faulted(
    plan: &RollPlan,
    report: &RollReport,
    near_position: Decimal,
    far_position: Decimal,
    fills: usize,
) {
    assert!(report.refused.is_some(), "{report:?}");
    assert_eq!(report.state, RollState::Faulted);
    assert_eq!(plan.state(), RollState::Faulted);
    assert_eq!(plan.fault(), report.refused.as_deref());
    assert_eq!(report.near_position, near_position);
    assert_eq!(report.far_position, far_position);
    assert_eq!(plan.fills().len(), fills);
    assert_eq!(plan.progress(), Ok(None));
}

#[test]
fn a_real_fill_off_the_lot_stays_and_stops_the_intents() {
    let mut plan = long_ten();
    let off_lot = fill(&plan, RollLeg::Near, "n1", dec!(1.5));
    let report = apply_fill(&mut plan, off_lot);
    assert_recorded_and_faulted(&plan, &report, dec!(8.5), dec!(0), 1);
    assert_eq!(report.exposure_gap, dec!(-1.5));
    // Every live attempt is cancelled, once.
    assert_eq!(
        report.actions,
        [
            RollAction::Cancel {
                leg: RollLeg::Near,
                attempt: 0,
            },
            RollAction::Cancel {
                leg: RollLeg::Far,
                attempt: 0,
            },
        ]
    );
    // From then on only the record moves.
    let next = RollFill {
        cum_contracts: dec!(2.5),
        ..fill(&plan, RollLeg::Near, "n2", dec!(1))
    };
    let report = apply_fill(&mut plan, next);
    assert!(report.actions.is_empty());
    assert!(report.refused.is_none());
    assert_eq!(report.state, RollState::Faulted);
    assert_eq!(report.near_position, dec!(7.5));
    let tick = apply(&mut plan, RollEvent::Tick { at_ms: NEAR_EXPIRY });
    assert!(tick.actions.is_empty());
    assert_eq!(tick.near_position, dec!(7.5));
}

#[test]
fn an_attempt_already_asked_to_cancel_is_not_asked_again_on_a_fault() {
    let mut plan = long_ten();
    let at_deadline = apply(
        &mut plan,
        RollEvent::Tick {
            at_ms: NEAR_EXPIRY - 1000,
        },
    );
    assert_eq!(at_deadline.actions.len(), 2);
    let off_lot = fill(&plan, RollLeg::Near, "n1", dec!(1.5));
    let report = apply_fill(&mut plan, off_lot);
    assert!(report.refused.is_some());
    assert!(report.actions.is_empty(), "{report:?}");
    assert_eq!(report.near_position, dec!(8.5));
}

#[test]
fn a_real_fill_on_a_rejected_attempt_stays() {
    let mut plan = long_ten();
    apply(
        &mut plan,
        RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: NOW + 1,
        },
    );
    let late = fill(&plan, RollLeg::Far, "f1", dec!(3));
    let report = apply_fill(&mut plan, late);
    assert_recorded_and_faulted(&plan, &report, dec!(10), dec!(3), 1);
    assert_eq!(report.exposure_gap, dec!(3));
}

#[test]
fn a_real_fill_with_a_wrong_cumulative_stays() {
    let mut plan = long_ten();
    let wrong = RollFill {
        cum_contracts: dec!(5),
        ..fill(&plan, RollLeg::Near, "n1", dec!(2))
    };
    let report = apply_fill(&mut plan, wrong);
    assert_recorded_and_faulted(&plan, &report, dec!(8), dec!(0), 1);
    assert_eq!(report.exposure_gap, dec!(-2));
}

#[test]
fn a_real_fill_of_an_unknown_attempt_or_the_wrong_side_or_time_stays() {
    let mut plan = long_ten();
    let unknown = RollFill {
        attempt: Some(7),
        ..fill(&plan, RollLeg::Near, "n1", dec!(2))
    };
    let report = apply_fill(&mut plan, unknown);
    assert_recorded_and_faulted(&plan, &report, dec!(8), dec!(0), 1);

    // The near leg of a long sells, so a buy moves it the other way.
    let mut plan = long_ten();
    let wrong_side = RollFill {
        side: Action::Buy,
        ..fill(&plan, RollLeg::Near, "n1", dec!(2))
    };
    let report = apply_fill(&mut plan, wrong_side);
    assert_recorded_and_faulted(&plan, &report, dec!(12), dec!(0), 1);

    let mut plan = long_ten();
    let beyond = RollFill {
        at_ms: i64::MAX as u64 + 1,
        ..fill(&plan, RollLeg::Far, "f1", dec!(2))
    };
    let report = apply_fill(&mut plan, beyond);
    assert_recorded_and_faulted(&plan, &report, dec!(10), dec!(2), 1);
}

#[test]
fn a_sum_the_primitive_cannot_hold_does_not_lose_the_position() {
    // The far leg keeps the gap in range, while the near leg's fills pass a `Decimal`.
    let huge = dec!(50000000000000000000000000000);
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(near, far, dec!(70000000000000000000000000000))).unwrap();
    for event in [
        RollFill {
            cum_contracts: huge,
            ..fill(&plan, RollLeg::Near, "n1", huge)
        },
        RollFill {
            cum_contracts: huge,
            ..fill(&plan, RollLeg::Far, "f1", huge)
        },
    ] {
        assert!(apply_fill(&mut plan, event).refused.is_none());
    }
    let second = RollFill {
        cum_contracts: huge,
        ..fill(&plan, RollLeg::Near, "n2", huge)
    };
    let report = apply_fill(&mut plan, second);
    assert_recorded_and_faulted(
        &plan,
        &report,
        dec!(-30000000000000000000000000000),
        huge,
        3,
    );
    assert_eq!(report.exposure_gap, -huge);
}

#[test]
fn what_a_decimal_cannot_hold_is_an_error_and_changes_nothing() {
    // The gap would be 1e29, though the position alone would fit.
    let huge = dec!(50000000000000000000000000000);
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(
        near.clone(),
        far,
        dec!(70000000000000000000000000000),
    ))
    .unwrap();
    let first = RollFill {
        cum_contracts: huge,
        ..fill(&plan, RollLeg::Near, "n1", huge)
    };
    apply_fill(&mut plan, first);
    let before = plan.clone();
    let second = RollFill {
        cum_contracts: huge,
        ..fill(&plan, RollLeg::Near, "n2", huge)
    };
    assert!(matches!(
        plan.apply(&RollEvent::Fill(second)),
        Err(RollError::Plan { .. })
    ));
    assert_eq!(plan, before);
    // The same for a settlement that declares too much.
    assert!(matches!(
        plan.apply(&RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY),
            contracts: huge + huge / dec!(2) - dec!(1),
        }),
        Err(RollError::Plan { .. })
    ));
    assert_eq!(plan, before);
}

#[test]
fn only_what_is_no_fill_or_conflicts_is_an_error_and_changes_nothing() {
    let mut plan = long_ten();
    let good = fill(&plan, RollLeg::Near, "n1", dec!(2));
    apply_fill(&mut plan, good.clone());
    let before = plan.clone();
    // A repeat is the same execution.
    let report = apply_fill(&mut plan, good.clone());
    assert!(report.duplicate);
    assert!(report.actions.is_empty());
    assert_eq!(plan, before);
    // The same identifier with other content on its leg.
    for conflict in [
        RollFill {
            price: dec!(60001),
            ..good.clone()
        },
        RollFill {
            contracts: dec!(3),
            ..good.clone()
        },
        RollFill {
            fee: dec!(1),
            ..good.clone()
        },
    ] {
        assert_eq!(
            plan.apply(&RollEvent::Fill(conflict)),
            Err(RollError::ConflictingFill {
                exec_id: "n1".to_string(),
            })
        );
        assert_eq!(plan, before);
    }
    for not_a_fill in [
        RollFill {
            price: dec!(0),
            ..fill(&plan, RollLeg::Near, "n2", dec!(1))
        },
        RollFill {
            price: dec!(-1),
            ..fill(&plan, RollLeg::Near, "n2", dec!(1))
        },
        fill(&plan, RollLeg::Near, "n2", dec!(0)),
        fill(&plan, RollLeg::Near, "n2", dec!(-1)),
        RollFill {
            side: Action::Other,
            ..fill(&plan, RollLeg::Near, "n2", dec!(1))
        },
    ] {
        assert!(matches!(
            plan.apply(&RollEvent::Fill(not_a_fill)),
            Err(RollError::InvalidFill { .. })
        ));
        assert_eq!(plan, before);
    }
    // An overfill is a fact too, and shows as excess.
    let mut over = long_ten();
    let overfill = fill(&over, RollLeg::Near, "n1", dec!(12));
    let report = apply_fill(&mut over, overfill);
    assert!(report.refused.is_none());
    assert_eq!(report.near_position, dec!(-2));
    assert_eq!(over.progress().unwrap().unwrap().near_excess, dec!(12));
}

#[test]
fn execution_identifiers_are_per_contract() {
    let mut plan = long_ten();
    let near = fill(&plan, RollLeg::Near, "42", dec!(10));
    let far = fill(&plan, RollLeg::Far, "42", dec!(10));
    assert!(apply_fill(&mut plan, near.clone()).refused.is_none());
    let report = apply_fill(&mut plan, far.clone());
    assert!(!report.duplicate);
    assert!(report.refused.is_none());
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(report.far_position, dec!(10));
    assert_eq!(report.state, RollState::Completed);
    // Each one's repeat and conflict stay on its own leg.
    assert!(apply_fill(&mut plan, near).duplicate);
    assert!(apply_fill(&mut plan, far.clone()).duplicate);
    assert_eq!(
        plan.apply(&RollEvent::Fill(RollFill {
            price: dec!(60001),
            ..far
        })),
        Err(RollError::ConflictingFill {
            exec_id: "42".to_string(),
        })
    );
}

#[test]
fn fees_are_kept_per_asset() {
    let mut plan = long_ten();
    let first = RollFill {
        fee: dec!(1.5),
        ..fill(&plan, RollLeg::Near, "n1", dec!(2))
    };
    let second = RollFill {
        fee: dec!(0.25),
        fee_asset: asset("BNB"),
        cum_contracts: dec!(4),
        ..fill(&plan, RollLeg::Near, "n2", dec!(2))
    };
    let third = RollFill {
        fee: dec!(-0.5),
        ..fill(&plan, RollLeg::Far, "f1", dec!(2))
    };
    for event in [first, second, third] {
        assert!(apply_fill(&mut plan, event).refused.is_none());
    }
    let fees = plan.fees().unwrap();
    assert_eq!(fees.len(), 2);
    assert_eq!(fees[&asset("USDT")], dec!(1));
    assert_eq!(fees[&asset("BNB")], dec!(0.25));
}

#[test]
fn a_rejection_aborts_with_the_gap_kept() {
    let mut plan = long_ten();
    let near_fill = fill(&plan, RollLeg::Near, "n1", dec!(4));
    apply_fill(&mut plan, near_fill);
    let report = apply(
        &mut plan,
        RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: NOW + 2,
        },
    );
    assert_eq!(report.state, RollState::Aborting);
    assert_eq!(
        report.actions,
        [RollAction::Cancel {
            leg: RollLeg::Near,
            attempt: 0,
        }]
    );
    let report = apply(
        &mut plan,
        RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(4),
            at_ms: NOW + 3,
        },
    );
    assert_eq!(report.state, RollState::Aborted);
    assert_eq!(report.near_position, dec!(6));
    assert_eq!(report.exposure_gap, dec!(-4));
    // A cancel that does not match the fills is refused, and changes nothing.
    let mut other = long_ten();
    let before = other.clone();
    assert!(matches!(
        other.apply(&RollEvent::Tick { at_ms: 1 }),
        Err(RollError::Plan { .. })
    ));
    assert_eq!(other, before);
}

#[test]
fn a_resubmission_carries_the_real_limit_and_stops_at_the_deadline() {
    let (near, far) = same_size();
    let deadline_ms = NEAR_EXPIRY - 1000;
    let complete = RollFailurePolicy::Complete {
        max_resubmits: 2,
        then: RollFallback::Abort,
    };
    let new_plan = || {
        RollPlan::new(RollRequest {
            policy: complete,
            ..request(near.clone(), far.clone(), dec!(10))
        })
        .unwrap()
    };
    let mut plan = new_plan();
    let report = apply(
        &mut plan,
        RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: deadline_ms - 1,
        },
    );
    assert_eq!(
        report.actions,
        [RollAction::Resubmit {
            leg: RollLeg::Far,
            attempt: 1,
            contracts: dec!(10),
            price: dec!(61000),
        }]
    );
    // At the deadline nothing is resubmitted or opened, for either side's failure.
    for at_ms in [deadline_ms, NEAR_EXPIRY, NEAR_EXPIRY + 1] {
        for leg in [RollLeg::Near, RollLeg::Far] {
            let mut plan = new_plan();
            let report = apply(
                &mut plan,
                RollEvent::Rejected {
                    leg,
                    attempt: 0,
                    at_ms,
                },
            );
            assert!(
                !report
                    .actions
                    .iter()
                    .any(|a| matches!(a, RollAction::Resubmit { .. })),
                "{report:?}"
            );
        }
    }
    // A tick at the deadline asks to cancel what is open, and a moment before does not.
    let mut plan = new_plan();
    assert!(
        apply(
            &mut plan,
            RollEvent::Tick {
                at_ms: deadline_ms - 1
            }
        )
        .actions
        .is_empty()
    );
    let report = apply(&mut plan, RollEvent::Tick { at_ms: deadline_ms });
    assert_eq!(
        report.actions,
        [
            RollAction::Cancel {
                leg: RollLeg::Near,
                attempt: 0,
            },
            RollAction::Cancel {
                leg: RollLeg::Far,
                attempt: 0,
            },
        ]
    );
    // A fill after the deadline is still what happened.
    let after = RollFill {
        at_ms: deadline_ms + 1,
        ..fill(&plan, RollLeg::Far, "f1", dec!(2))
    };
    assert_eq!(apply_fill(&mut plan, after).far_position, dec!(2));
}

#[test]
fn an_unwind_offsets_the_excess_and_its_fill_moves_only_the_record() {
    let (near, far) = same_size();
    let mut plan = RollPlan::new(RollRequest {
        policy: RollFailurePolicy::Unwind,
        ..request(near, far, dec!(10))
    })
    .unwrap();
    let near_fill = fill(&plan, RollLeg::Near, "n1", dec!(4));
    apply_fill(&mut plan, near_fill);
    apply(
        &mut plan,
        RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: NOW + 2,
        },
    );
    let report = apply(
        &mut plan,
        RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(4),
            at_ms: NOW + 3,
        },
    );
    assert_eq!(report.state, RollState::UnwindRequested);
    assert_eq!(
        report.actions,
        [RollAction::Unwind {
            leg: RollLeg::Near,
            side: Action::Buy,
            contracts: dec!(4),
        }]
    );
    let unwind = RollFill {
        leg: RollLeg::Near,
        attempt: None,
        exec_id: "u1".to_string(),
        side: Action::Buy,
        contracts: dec!(4),
        cum_contracts: dec!(4),
        price: dec!(60000),
        fee: dec!(0),
        fee_asset: asset("USDT"),
        at_ms: NOW + 4,
    };
    let report = apply_fill(&mut plan, unwind);
    assert!(report.refused.is_none());
    assert_eq!(report.near_position, dec!(10));
    assert_eq!(report.exposure_gap, dec!(0));
    assert_eq!(report.state, RollState::UnwindRequested);
}

#[test]
fn a_settlement_is_declared_and_moves_the_position_by_what_it_closed() {
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(near.clone(), far, dec!(10))).unwrap();
    let near_fill = fill(&plan, RollLeg::Near, "n1", dec!(4));
    apply_fill(&mut plan, near_fill);
    let far_fill = fill(&plan, RollLeg::Far, "f1", dec!(1));
    apply_fill(&mut plan, far_fill);

    // It must be this contract's, from its source, at or after its expiry.
    assert!(matches!(
        plan.apply(&RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY - 1),
            contracts: dec!(6),
        }),
        Err(RollError::Future(FutureError::PublishedBeforeExpiry { .. }))
    ));
    let other_source = SettlementPrice::new(
        near.spec().id().clone(),
        SettlementSource::new("a last price").unwrap(),
        NEAR_EXPIRY,
        dec!(61000),
        NEAR_EXPIRY,
    )
    .unwrap();
    assert!(matches!(
        plan.apply(&RollEvent::NearSettled {
            price: other_source,
            contracts: dec!(6),
        }),
        Err(RollError::Future(FutureError::WrongSource { .. }))
    ));

    let price = settlement(&near, dec!(61000), NEAR_EXPIRY);
    let report = apply(
        &mut plan,
        RollEvent::NearSettled {
            price: price.clone(),
            contracts: dec!(6),
        },
    );
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(report.far_position, dec!(1));
    // The start was 10, and only 1 is left.
    assert_eq!(report.exposure, dec!(1));
    assert_eq!(report.exposure_gap, dec!(-9));
    assert_eq!(plan.near_settled(), Some(dec!(6)));
    // What is open is cancelled, and nothing is resubmitted or opened.
    assert_eq!(
        report.actions,
        [
            RollAction::Cancel {
                leg: RollLeg::Near,
                attempt: 0,
            },
            RollAction::Cancel {
                leg: RollLeg::Far,
                attempt: 0,
            },
        ]
    );
    let before = plan.clone();
    let repeat = apply(
        &mut plan,
        RollEvent::NearSettled {
            price: price.clone(),
            contracts: dec!(6),
        },
    );
    assert!(repeat.duplicate);
    assert_eq!(plan, before);
    for conflict in [
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61001), NEAR_EXPIRY),
            contracts: dec!(6),
        },
        RollEvent::NearSettled {
            price: price.clone(),
            contracts: dec!(5),
        },
    ] {
        assert_eq!(plan.apply(&conflict), Err(RollError::ConflictingSettlement));
        assert_eq!(plan, before);
    }
    // Far fills after it are facts as well.
    let far_late = RollFill {
        cum_contracts: dec!(3),
        at_ms: NEAR_EXPIRY + 5,
        ..fill(&plan, RollLeg::Far, "f2", dec!(2))
    };
    assert_eq!(apply_fill(&mut plan, far_late).far_position, dec!(3));
}

#[test]
fn a_settlement_may_arrive_after_the_time_it_carries() {
    // Publication is not arrival: the plan has already seen a later time.
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(near.clone(), far, dec!(10))).unwrap();
    let tick = apply(
        &mut plan,
        RollEvent::Tick {
            at_ms: NEAR_EXPIRY + 100,
        },
    );
    assert_eq!(tick.actions.len(), 2);
    let report = apply(
        &mut plan,
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY + 5),
            contracts: dec!(10),
        },
    );
    assert!(report.actions.is_empty());
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(plan.near_settled(), Some(dec!(10)));
    // The same when no time has passed yet, on a fresh plan.
    let mut fresh = long_ten();
    let report = apply(
        &mut fresh,
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY + 5),
            contracts: dec!(10),
        },
    );
    assert_eq!(report.actions.len(), 2);
}

#[test]
fn a_near_fill_after_the_settlement_depends_on_when_it_executed() {
    let (near, far) = same_size();
    let mut plan = RollPlan::new(request(near.clone(), far.clone(), dec!(10))).unwrap();
    apply(
        &mut plan,
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY),
            contracts: dec!(10),
        },
    );
    // Executed before the expiry, observed now: it closed at the settlement price.
    let before_expiry = RollFill {
        at_ms: NEAR_EXPIRY - 1,
        ..fill(&plan, RollLeg::Near, "n1", dec!(2))
    };
    let report = apply_fill(&mut plan, before_expiry);
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(plan.closed_at_settlement(), dec!(-2));
    assert_eq!(plan.fills().len(), 1);
    // Executed at the expiry, it moves the position and leaves a residual.
    let at_expiry = RollFill {
        at_ms: NEAR_EXPIRY,
        cum_contracts: dec!(5),
        ..fill(&plan, RollLeg::Near, "n2", dec!(3))
    };
    let report = apply_fill(&mut plan, at_expiry);
    assert_eq!(report.near_position, dec!(-3));
    assert_eq!(report.exposure, dec!(-3));
    assert_eq!(plan.closed_at_settlement(), dec!(-2));

    // The settlement closed only 8, as it was booked after a fill the plan sees later:
    // that fill takes the residual to zero, and what is beyond it closed.
    let mut first_in_ledger = RollPlan::new(request(near.clone(), far, dec!(10))).unwrap();
    apply(
        &mut first_in_ledger,
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY),
            contracts: dec!(8),
        },
    );
    assert_eq!(first_in_ledger.position(RollLeg::Near), dec!(2));
    let seen_later = RollFill {
        at_ms: NEAR_EXPIRY - 1,
        ..fill(&first_in_ledger, RollLeg::Near, "n1", dec!(3))
    };
    let report = apply_fill(&mut first_in_ledger, seen_later);
    assert_eq!(report.near_position, dec!(0));
    assert_eq!(first_in_ledger.closed_at_settlement(), dec!(-1));
    // A fill that adds to the residual closes whole, and one smaller than it moves it.
    let (near, far) = same_size();
    let mut short = RollPlan::new(request(near.clone(), far, dec!(-10))).unwrap();
    apply(
        &mut short,
        RollEvent::NearSettled {
            price: settlement(&near, dec!(61000), NEAR_EXPIRY),
            contracts: dec!(-8),
        },
    );
    assert_eq!(short.position(RollLeg::Near), dec!(-2));
    let smaller = RollFill {
        at_ms: NEAR_EXPIRY - 1,
        ..fill(&short, RollLeg::Near, "n1", dec!(1))
    };
    assert_eq!(apply_fill(&mut short, smaller).near_position, dec!(-1));
    assert_eq!(short.closed_at_settlement(), dec!(0));
    let adds = RollFill {
        at_ms: NEAR_EXPIRY - 1,
        side: Action::Sell,
        attempt: None,
        ..fill(&short, RollLeg::Near, "n2", dec!(1))
    };
    assert_eq!(apply_fill(&mut short, adds).near_position, dec!(-1));
    assert_eq!(short.closed_at_settlement(), dec!(-1));
}

#[cfg(feature = "margin")]
mod ledger {
    use super::*;
    use margin_rs::ledger::{Execution, Ledger, LedgerConfig};
    use margin_rs::{Asset, TransferId};
    use market_maker_rs::execution::{Action as FillSide, Fill, OrderId};
    use market_maker_rs::instruments::margin::{MarginScales, fill_execution, margin_instrument};
    use market_maker_rs::instruments::roll::{RollDiscrepancy, RollSettlement};

    fn ledger_of(terms: &[&FutureTerms]) -> Ledger {
        let mut ledger = Ledger::new(LedgerConfig::default());
        for asset_code in ["BTC", "USDT", "BNB"] {
            ledger
                .register_asset(Asset::new(asset_code).unwrap(), 8)
                .unwrap();
        }
        for one in terms {
            ledger
                .register_instrument(
                    margin_instrument(
                        one.spec(),
                        MarginScales {
                            price_scale: 2,
                            contract_scale: 3,
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        ledger
            .deposit(
                TransferId::new("seed").unwrap(),
                Asset::new("USDT").unwrap(),
                dec!(100000),
            )
            .unwrap();
        ledger
    }

    fn execution_of(terms: &FutureTerms, fill: &RollFill) -> Execution {
        fill_execution(
            terms.spec(),
            &Fill {
                symbol: terms.spec().id().as_str().to_string(),
                order_id: OrderId::new("order"),
                trade_id: fill.exec_id.clone(),
                price: fill.price,
                quantity: fill.contracts,
                side: fill.side,
                timestamp: fill.at_ms,
                fee: fill.fee,
                fee_currency: fill.fee_asset.as_str().to_string(),
            },
        )
        .unwrap()
    }

    fn book(ledger: &mut Ledger, terms: &FutureTerms, fill: &RollFill) {
        ledger
            .book_execution_allowing_overdraft(&execution_of(terms, fill), None)
            .unwrap();
    }

    /// A plan long 10 near with its opening position in the ledger at 60000.
    fn opened() -> (FutureTerms, FutureTerms, Ledger, RollPlan) {
        let (near, far) = same_size();
        let mut ledger = ledger_of(&[&near, &far]);
        let plan = RollPlan::new(request(near.clone(), far.clone(), dec!(10))).unwrap();
        let open = RollFill {
            exec_id: "open".to_string(),
            side: FillSide::Buy,
            attempt: None,
            ..fill(&plan, RollLeg::Near, "open", dec!(10))
        };
        book(&mut ledger, &near, &open);
        let plan = plan.anchored_to(&ledger);
        (near, far, ledger, plan)
    }

    #[test]
    fn a_full_roll_agrees_with_its_ledger_and_its_cash_is_its_own() {
        let (near, far, mut ledger, mut plan) = opened();
        let near_fill = RollFill {
            price: dec!(61000),
            fee: dec!(2),
            ..fill(&plan, RollLeg::Near, "n1", dec!(10))
        };
        let far_fill = RollFill {
            price: dec!(62000),
            fee: dec!(3),
            ..fill(&plan, RollLeg::Far, "f1", dec!(10))
        };
        book(&mut ledger, &near, &near_fill);
        book(&mut ledger, &far, &far_fill);
        apply_fill(&mut plan, near_fill);
        apply_fill(&mut plan, far_fill);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.ledger_position, dec!(0));
        assert_eq!(reconciled.far.ledger_position, dec!(10));
        assert_eq!(reconciled.near.unexplained, dec!(0));
        // Closing 10 from 60000 at 61000, with its fee.
        assert_eq!(reconciled.near.roll_realized, dec!(10000));
        assert_eq!(reconciled.near.roll_fees["USDT"], dec!(2));
        assert_eq!(reconciled.far.roll_realized, dec!(0));
        assert_eq!(reconciled.far.roll_fees["USDT"], dec!(3));
        assert_eq!(reconciled.settlement, None);
    }

    #[test]
    fn realized_before_the_roll_stays_out_of_the_rolls_cash() {
        let (near, far) = same_size();
        let mut ledger = ledger_of(&[&near, &far]);
        let mut plan = RollPlan::new(request(near.clone(), far.clone(), dec!(10))).unwrap();
        // History: bought 20 at 60000, sold 10 at 60500, for 5000 realized.
        let history = |id: &str, side, price, contracts| RollFill {
            exec_id: id.to_string(),
            side,
            price,
            attempt: None,
            ..fill(&plan, RollLeg::Near, id, contracts)
        };
        book(
            &mut ledger,
            &near,
            &history("h1", FillSide::Buy, dec!(60000), dec!(20)),
        );
        book(
            &mut ledger,
            &near,
            &history("h2", FillSide::Sell, dec!(60500), dec!(10)),
        );
        plan = plan.anchored_to(&ledger);
        // The roll closes 4 at 61000, for 4000.
        let roll = RollFill {
            price: dec!(61000),
            ..fill(&plan, RollLeg::Near, "n1", dec!(4))
        };
        book(&mut ledger, &near, &roll);
        apply_fill(&mut plan, roll);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.roll_realized, dec!(4000));
        assert_eq!(reconciled.near.position_realized, dec!(9000));
    }

    #[test]
    fn a_fill_only_the_plan_or_only_the_ledger_knows_is_quantified() {
        let (near, far, mut ledger, mut plan) = opened();
        // One leg alone, that the ledger does not hold.
        let far_fill = fill(&plan, RollLeg::Far, "f1", dec!(3));
        apply_fill(&mut plan, far_fill);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert_eq!(reconciled.far.expected_position, dec!(3));
        assert_eq!(reconciled.far.ledger_position, dec!(0));
        assert_eq!(reconciled.far.unexplained, dec!(-3));
        assert_eq!(
            reconciled.discrepancies,
            [
                RollDiscrepancy::MissingInLedger {
                    leg: RollLeg::Far,
                    exec_id: "f1".to_string(),
                },
                RollDiscrepancy::UnexplainedPosition {
                    leg: RollLeg::Far,
                    quantity: dec!(-3),
                },
            ]
        );
        // A quote's fill outside the plan, in the ledger only, on the near contract.
        let quote = RollFill {
            attempt: None,
            ..fill(&plan, RollLeg::Near, "quote", dec!(2))
        };
        book(&mut ledger, &near, &quote);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert_eq!(reconciled.near.unexplained, dec!(-2));
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::UnexplainedPosition {
                    leg: RollLeg::Near,
                    quantity: dec!(-2),
                })
        );
        assert_eq!(reconciled.near.unexplained_entries, 1);
        let _ = far;
    }

    #[test]
    fn a_round_trip_outside_the_plan_is_not_clean_though_the_position_is_the_same() {
        let (near, far, mut ledger, plan) = opened();
        let (sold, bought) = (
            RollFill {
                price: dec!(61000),
                fee: dec!(1),
                attempt: None,
                ..fill(&plan, RollLeg::Near, "q1", dec!(2))
            },
            RollFill {
                side: FillSide::Buy,
                price: dec!(60500),
                fee: dec!(0.5),
                fee_asset: asset("BNB"),
                attempt: None,
                ..fill(&plan, RollLeg::Near, "q2", dec!(2))
            },
        );
        book(&mut ledger, &near, &sold);
        book(&mut ledger, &near, &bought);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert_eq!(reconciled.near.unexplained, dec!(0));
        assert!(!reconciled.is_clean());
        assert_eq!(
            reconciled.discrepancies,
            [RollDiscrepancy::UnexplainedEntries {
                leg: RollLeg::Near,
                count: 2,
            }]
        );
        assert_eq!(reconciled.near.unexplained_entries, 2);
        // Sold 2 from 60000 at 61000: 2000; buying back opens no realized PnL.
        assert_eq!(reconciled.near.unexplained_realized, dec!(2000));
        assert_eq!(reconciled.near.unexplained_fees["USDT"], dec!(1));
        assert_eq!(reconciled.near.unexplained_fees["BNB"], dec!(0.5));
        assert_eq!(reconciled.far.unexplained_entries, 0);

        // Another instrument's entries do not count, and the opening before the anchor
        // does not either, which `opened` shows by being clean.
        let (_, _, ledger, plan) = opened();
        assert!(plan.reconcile(&ledger).unwrap().is_clean());
        let _ = far;
    }

    #[test]
    fn a_split_of_a_late_fill_is_compared_by_quantity() {
        // A sale outside the plan, a settlement of the other 8, and then a late sale of 5
        // that the ledger closes whole. The plan takes 2 of it for the residual.
        let (near, _far, mut ledger, mut plan) = opened();
        let foreign = RollFill {
            attempt: None,
            ..fill(&plan, RollLeg::Near, "foreign", dec!(2))
        };
        book(&mut ledger, &near, &foreign);
        let price = settlement(&near, dec!(62000), NEAR_EXPIRY + 5);
        near.settle_in_ledger(&mut ledger, &price).unwrap();
        apply(
            &mut plan,
            RollEvent::NearSettled {
                price,
                contracts: dec!(8),
            },
        );
        let late = RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&plan, RollLeg::Near, "late", dec!(5))
        };
        book(&mut ledger, &near, &late);
        let report = apply_fill(&mut plan, late);
        assert_eq!(report.near_position, dec!(0));
        assert_eq!(plan.closed_at_settlement(), dec!(-3));
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(!reconciled.is_clean());
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::FillDiffers {
                    leg: RollLeg::Near,
                    exec_id: "late".to_string(),
                    field: "closed_at_settlement",
                }),
            "{reconciled:?}"
        );
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::UnexplainedEntries {
                    leg: RollLeg::Near,
                    count: 1,
                })
        );
    }

    #[test]
    fn same_execution_id_on_both_legs_stays_apart_when_one_closes_at_settlement() {
        let (near, far, mut ledger, mut plan) = opened();
        let price = settlement(&near, dec!(62000), NEAR_EXPIRY + 5);
        near.settle_in_ledger(&mut ledger, &price).unwrap();
        apply(
            &mut plan,
            RollEvent::NearSettled {
                price,
                contracts: dec!(10),
            },
        );
        let near_late = RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&plan, RollLeg::Near, "42", dec!(2))
        };
        let far_fill = RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&plan, RollLeg::Far, "42", dec!(2))
        };
        book(&mut ledger, &near, &near_late);
        book(&mut ledger, &far, &far_fill);
        apply_fill(&mut plan, near_late);
        apply_fill(&mut plan, far_fill);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
    }

    #[test]
    fn same_execution_id_on_both_legs_reconciles_each_on_its_contract() {
        let (near, far, mut ledger, mut plan) = opened();
        let near_fill = RollFill {
            fee: dec!(1),
            ..fill(&plan, RollLeg::Near, "42", dec!(10))
        };
        let far_fill = RollFill {
            fee: dec!(2),
            ..fill(&plan, RollLeg::Far, "42", dec!(10))
        };
        book(&mut ledger, &near, &near_fill);
        book(&mut ledger, &far, &far_fill);
        apply_fill(&mut plan, near_fill);
        apply_fill(&mut plan, far_fill);
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.roll_fees["USDT"], dec!(1));
        assert_eq!(reconciled.far.roll_fees["USDT"], dec!(2));
    }

    #[test]
    fn an_entry_with_other_terms_is_told_apart_by_field() {
        let differing = |tweak: fn(&mut RollFill), field: &str| {
            let (near, _far, mut ledger, mut plan) = opened();
            let good = RollFill {
                fee: dec!(1),
                ..fill(&plan, RollLeg::Near, "n1", dec!(4))
            };
            let mut booked = good.clone();
            tweak(&mut booked);
            book(&mut ledger, &near, &booked);
            apply_fill(&mut plan, good);
            let reconciled = plan.reconcile(&ledger).unwrap();
            assert!(
                reconciled
                    .discrepancies
                    .contains(&RollDiscrepancy::FillDiffers {
                        leg: RollLeg::Near,
                        exec_id: "n1".to_string(),
                        field: Box::leak(field.to_string().into_boxed_str()),
                    }),
                "{field}: {reconciled:?}"
            );
        };
        differing(|f| f.price = dec!(60001), "price");
        differing(|f| f.contracts = dec!(5), "contracts");
        differing(|f| f.side = FillSide::Buy, "side");
        differing(|f| f.fee = dec!(2), "fee");
        differing(|f| f.fee_asset = asset("BNB"), "fee_asset");
    }

    #[test]
    fn a_leg_with_fees_in_two_assets_keeps_them_apart() {
        let (near, _far, mut ledger, mut plan) = opened();
        let first = RollFill {
            fee: dec!(1.5),
            ..fill(&plan, RollLeg::Near, "n1", dec!(2))
        };
        let second = RollFill {
            fee: dec!(0.25),
            fee_asset: asset("BNB"),
            cum_contracts: dec!(4),
            ..fill(&plan, RollLeg::Near, "n2", dec!(2))
        };
        for event in [first, second] {
            book(&mut ledger, &near, &event);
            apply_fill(&mut plan, event);
        }
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.roll_fees.len(), 2);
        assert_eq!(reconciled.near.roll_fees["USDT"], dec!(1.5));
        assert_eq!(reconciled.near.roll_fees["BNB"], dec!(0.25));
    }

    #[test]
    fn a_settled_remainder_is_explained_and_its_cash_is_apart() {
        let (near, far, mut ledger, mut plan) = opened();
        let near_fill = RollFill {
            price: dec!(61000),
            ..fill(&plan, RollLeg::Near, "n1", dec!(4))
        };
        book(&mut ledger, &near, &near_fill);
        apply_fill(&mut plan, near_fill);
        let price = settlement(&near, dec!(62000), NEAR_EXPIRY + 5);

        // The ledger settled, and the plan was not told.
        let mut told_late = ledger.clone();
        near.settle_in_ledger(&mut told_late, &price).unwrap();
        let reconciled = plan.reconcile(&told_late).unwrap();
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::PlanUnawareOfSettlement)
        );
        assert_eq!(reconciled.near.unexplained, dec!(0));

        // The plan was told, and the ledger did not settle.
        let mut told = plan.clone();
        apply(
            &mut told,
            RollEvent::NearSettled {
                price: price.clone(),
                contracts: dec!(6),
            },
        );
        let reconciled = told.reconcile(&ledger).unwrap();
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::LedgerNotSettled)
        );
        assert_eq!(reconciled.near.unexplained, dec!(6));

        // Both.
        near.settle_in_ledger(&mut ledger, &price).unwrap();
        let reconciled = told.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        // 6 from 60000 at 62000 apart from the roll's 4 from 60000 at 61000.
        assert_eq!(
            reconciled.settlement,
            Some(RollSettlement {
                contracts: dec!(6),
                price: dec!(62000),
                realized_pnl: dec!(12000),
            })
        );
        assert_eq!(reconciled.near.roll_realized, dec!(4000));
        assert_eq!(reconciled.near.position_realized, dec!(16000));

        // Declared differently from what was booked.
        let (_, _, mut other, mut plan) = opened();
        near.settle_in_ledger(&mut other, &price).unwrap();
        apply(
            &mut plan,
            RollEvent::NearSettled {
                price: settlement(&near, dec!(62001), NEAR_EXPIRY + 5),
                contracts: dec!(9),
            },
        );
        let reconciled = plan.reconcile(&other).unwrap();
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::SettledContractsDiffer {
                    plan: dec!(9),
                    ledger: dec!(10),
                })
        );
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::SettledPriceDiffers {
                    plan: dec!(62001),
                    ledger: dec!(62000),
                })
        );
        let _ = far;
    }

    #[test]
    fn a_late_fill_is_booked_closed_or_refused_as_the_plan_says() {
        let (near, _far, mut ledger, mut plan) = opened();
        let price = settlement(&near, dec!(62000), NEAR_EXPIRY + 5);
        near.settle_in_ledger(&mut ledger, &price).unwrap();
        apply(
            &mut plan,
            RollEvent::NearSettled {
                price,
                contracts: dec!(10),
            },
        );
        // Executed before the expiry and delivered after: a real ledger books it flat.
        let before_expiry = RollFill {
            price: dec!(61000),
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&plan, RollLeg::Near, "n1", dec!(2))
        };
        book(&mut ledger, &near, &before_expiry);
        assert_eq!(apply_fill(&mut plan, before_expiry).near_position, dec!(0));
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.ledger_position, dec!(0));
        // A sale at 61000 closed at 62000 at once: -1000 for 2 contracts.
        assert_eq!(reconciled.near.roll_realized, dec!(-2000));

        // Executed at the expiry: a real ledger refuses it.
        let at_expiry = RollFill {
            at_ms: NEAR_EXPIRY,
            cum_contracts: dec!(5),
            ..fill(&plan, RollLeg::Near, "n2", dec!(3))
        };
        assert!(
            ledger
                .book_execution_allowing_overdraft(&execution_of(&near, &at_expiry), None)
                .is_err()
        );
        assert_eq!(apply_fill(&mut plan, at_expiry).near_position, dec!(-3));
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert_eq!(reconciled.near.unexplained, dec!(3));
        assert_eq!(
            reconciled.discrepancies,
            [
                RollDiscrepancy::MissingInLedger {
                    leg: RollLeg::Near,
                    exec_id: "n2".to_string(),
                },
                RollDiscrepancy::UnexplainedPosition {
                    leg: RollLeg::Near,
                    quantity: dec!(3),
                },
            ]
        );
    }

    #[test]
    fn a_fill_the_ledger_booked_before_settling_agrees_in_either_order_of_news() {
        // Booked before the ledger settled, and reported to the plan after: the ledger
        // closed 8, and the identity of the position holds with no flag on the entry.
        let (near, _far, mut ledger, mut plan) = opened();
        let before_expiry = RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&plan, RollLeg::Near, "n1", dec!(2))
        };
        book(&mut ledger, &near, &before_expiry);
        let price = settlement(&near, dec!(62000), NEAR_EXPIRY + 5);
        near.settle_in_ledger(&mut ledger, &price).unwrap();
        apply(
            &mut plan,
            RollEvent::NearSettled {
                price: price.clone(),
                contracts: dec!(8),
            },
        );
        apply_fill(&mut plan, before_expiry.clone());
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        assert_eq!(reconciled.near.expected_position, dec!(0));
        assert_eq!(plan.closed_at_settlement(), dec!(0));

        // The plan saw the fill first, then a settlement of everything, as the ledger
        // booked the fill after it: near goes to -2 until reconciled, which says so.
        let (near_two, _, mut later, mut first_seen) = opened();
        let seen = RollFill {
            at_ms: NEAR_EXPIRY - 1,
            ..fill(&first_seen, RollLeg::Near, "n1", dec!(2))
        };
        apply_fill(&mut first_seen, seen.clone());
        let all = settlement(&near_two, dec!(62000), NEAR_EXPIRY + 5);
        near_two.settle_in_ledger(&mut later, &all).unwrap();
        book(&mut later, &near_two, &seen);
        apply(
            &mut first_seen,
            RollEvent::NearSettled {
                price: all,
                contracts: dec!(10),
            },
        );
        assert_eq!(first_seen.position(RollLeg::Near), dec!(-2));
        let reconciled = first_seen.reconcile(&later).unwrap();
        assert_eq!(reconciled.near.ledger_position, dec!(0));
        assert_eq!(reconciled.near.unexplained, dec!(2));
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::FillDiffers {
                    leg: RollLeg::Near,
                    exec_id: "n1".to_string(),
                    field: "closed_at_settlement",
                })
        );

        // Told it closed 10, the plan takes that fill as closed, and the entry says no.
        let (_, _, _, mut wrong) = opened();
        apply(
            &mut wrong,
            RollEvent::NearSettled {
                price,
                contracts: dec!(10),
            },
        );
        apply_fill(&mut wrong, before_expiry);
        let reconciled = wrong.reconcile(&ledger).unwrap();
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::FillDiffers {
                    leg: RollLeg::Near,
                    exec_id: "n1".to_string(),
                    field: "closed_at_settlement",
                })
        );
        assert!(
            reconciled
                .discrepancies
                .contains(&RollDiscrepancy::SettledContractsDiffer {
                    plan: dec!(10),
                    ledger: dec!(8),
                })
        );
    }

    #[test]
    fn an_inverse_roll_reconciles_in_its_settlement_asset() {
        let near = inverse("BTC-NEAR", NEAR_EXPIRY);
        let far = inverse("BTC-FAR", FAR_EXPIRY);
        let mut ledger = ledger_of(&[&near, &far]);
        ledger
            .deposit(
                TransferId::new("btc").unwrap(),
                Asset::new("BTC").unwrap(),
                dec!(1),
            )
            .unwrap();
        let mut plan = RollPlan::new(request(near.clone(), far.clone(), dec!(1000))).unwrap();
        let open = RollFill {
            side: FillSide::Buy,
            attempt: None,
            price: dec!(60000),
            ..fill(&plan, RollLeg::Near, "open", dec!(1000))
        };
        book(&mut ledger, &near, &open);
        plan = plan.anchored_to(&ledger);
        let near_fill = RollFill {
            price: dec!(62000),
            fee: dec!(0.0001),
            fee_asset: asset("BTC"),
            ..fill(&plan, RollLeg::Near, "n1", dec!(1000))
        };
        let far_fill = RollFill {
            price: dec!(62500),
            fee: dec!(0.0001),
            fee_asset: asset("BTC"),
            ..fill(&plan, RollLeg::Far, "f1", dec!(1000))
        };
        book(&mut ledger, &near, &near_fill);
        book(&mut ledger, &far, &far_fill);
        apply_fill(&mut plan, near_fill);
        let report = apply_fill(&mut plan, far_fill);
        assert_eq!(report.exposure_gap, dec!(0));
        let reconciled = plan.reconcile(&ledger).unwrap();
        assert!(reconciled.is_clean(), "{reconciled:?}");
        // 10000 / 60000 = 0.16666667 and 10000 / 62000 = 0.16129032, in BTC.
        assert_eq!(reconciled.near.roll_realized, dec!(0.00537635));
        assert_eq!(reconciled.near.roll_fees["BTC"], dec!(0.0001));
    }

    #[test]
    fn a_ledger_with_other_terms_is_refused() {
        let (near, far) = same_size();
        let other = linear("NEAR", dec!(2), dec!(1), NEAR_EXPIRY);
        let plan = RollPlan::new(request(near, far.clone(), dec!(10))).unwrap();
        // It must be anchored first.
        assert_eq!(
            plan.reconcile(&ledger_of(&[&other, &far])),
            Err(RollError::NotAnchored)
        );
        let plan = plan.anchored_to(&ledger_of(&[]));
        assert!(matches!(
            plan.reconcile(&ledger_of(&[&other, &far])),
            Err(RollError::Future(FutureError::OtherTermsInLedger { .. }))
        ));
        let other_far = linear("FAR", dec!(2), dec!(1), FAR_EXPIRY);
        let (held_near, _) = same_size();
        assert!(matches!(
            plan.reconcile(&ledger_of(&[&held_near, &other_far])),
            Err(RollError::Future(FutureError::OtherTermsInLedger { .. }))
        ));
        // A ledger without the contracts holds no position in them.
        let empty = plan.reconcile(&ledger_of(&[])).unwrap();
        assert_eq!(empty.near.ledger_position, dec!(0));
        assert_eq!(empty.near.unexplained, dec!(-10));
    }
}
