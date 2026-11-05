//! A roll through the backtest: the same plan booked directly in a ledger, from a script
//! of literal fills and events worked out by hand, must end in the same ledger and the
//! same plan as a run through the engine.

#![cfg(all(feature = "arbitrage", feature = "margin"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use financial_types::instrument::InstrumentId as SpecId;
use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::{Execution, Ledger, LedgerConfig};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::contracts::{
    ContractCommand, ContractConfig, ContractEngine, ContractFill, ContractReport,
    ContractRunResult, ContractStrategy, ContractView,
};
use market_maker_rs::backtest::{
    FillModel, FillResult, ImmediateFillModel, MarketTick, SimLatencyConfig, SimulatedOrder,
    VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::instruments::futures::{
    FutureTerms, SettlementMethod, SettlementPrice, SettlementSource,
};
use market_maker_rs::instruments::reference::deribit;
use market_maker_rs::instruments::roll::guard::RollGuard;
use market_maker_rs::instruments::roll::run::{RollStrategy, SharedRollGuard};
use market_maker_rs::instruments::roll::{
    RollEvent, RollFailurePolicy, RollFallback, RollFill, RollLeg, RollPlan, RollRequest, RollState,
};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentSpec, Margining, TickSchedule,
};
use market_maker_rs::risk::pretrade::{PretradeDecision, PretradePolicy, PretradeRequest};
use market_maker_rs::types::error::{MMError, MMResult};

const NEAR_MS: u64 = 1000;
const FAR_MS: u64 = 2000;
const CREATED_MS: u64 = 5;
const DEADLINE_MS: u64 = 900;

fn spec_of(name: &str, expiry_ms: u64) -> InstrumentSpec {
    let asset = |code| AssetCode::new(code).unwrap();
    InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new(name).unwrap(),
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
    .unwrap()
}

fn source() -> SettlementSource {
    SettlementSource::new("the venue's index").unwrap()
}

fn terms_of(spec: InstrumentSpec) -> FutureTerms {
    FutureTerms::new(spec, SettlementMethod::Cash, source()).unwrap()
}

fn near() -> FutureTerms {
    terms_of(spec_of("NEAR", NEAR_MS))
}

fn far() -> FutureTerms {
    terms_of(spec_of("FAR", FAR_MS))
}

/// A ledger of both contracts, long `position` of the near one at `price`.
fn opened(
    near: &FutureTerms,
    far: &FutureTerms,
    assets: [&str; 2],
    price: Decimal,
    position: Decimal,
) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for asset in assets {
        ledger
            .register_asset(Asset::new(asset).unwrap(), 8)
            .unwrap();
    }
    for terms in [near, far] {
        ledger
            .register_instrument(
                Instrument::from_spec(
                    terms.spec(),
                    InstrumentScales {
                        price_scale: 8,
                        contract_scale: 8,
                    },
                )
                .unwrap(),
            )
            .unwrap();
    }
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("open").unwrap(),
                InstrumentId::new(near.spec().id().as_str()).unwrap(),
                Action::Buy,
                price,
                position,
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
    ledger
}

fn usdt_ledger() -> Ledger {
    opened(&near(), &far(), ["BTC", "USDT"], dec!(60000), dec!(10))
}

fn request(policy: RollFailurePolicy) -> RollRequest {
    RollRequest {
        near: near(),
        far: far(),
        near_position: dec!(10),
        far_position: dec!(0),
        near_price: dec!(61000),
        far_price: dec!(62000),
        policy,
        deadline_ms: DEADLINE_MS,
        created_at_ms: CREATED_MS,
    }
}

fn plan_on(ledger: &Ledger, policy: RollFailurePolicy) -> RollPlan {
    RollPlan::new(request(policy)).unwrap().anchored_to(ledger)
}

fn tick(timestamp: u64, bid: Decimal, ask: Decimal) -> MarketTick {
    MarketTick::new(timestamp, bid, dec!(100000), ask, dec!(100000))
}

fn data(ticks: Vec<MarketTick>) -> VecDataSource {
    VecDataSource::new(ticks)
}

fn settlement(terms: &FutureTerms, price: Decimal, published_ms: u64) -> SettlementPrice {
    SettlementPrice::new(
        terms.spec().id().clone(),
        source(),
        terms.expiry_ms(),
        price,
        published_ms,
    )
    .unwrap()
}

/// One literal step of a plan booked directly.
enum Step {
    /// A fill, booked in the ledger and given to the plan.
    Fill(FutureTerms, RollFill),
    /// An event for the plan alone.
    Event(RollEvent),
    /// A settlement, booked in the ledger and given to the plan.
    Settle(FutureTerms, SettlementPrice, Decimal),
}

#[allow(clippy::too_many_arguments)]
fn fill_of(
    terms: &FutureTerms,
    leg: RollLeg,
    attempt: Option<u32>,
    exec_id: &str,
    side: Action,
    price: Decimal,
    contracts: Decimal,
    fee: Decimal,
    at_ms: u64,
) -> Step {
    Step::Fill(
        terms.clone(),
        RollFill {
            leg,
            attempt,
            exec_id: exec_id.to_string(),
            side,
            contracts,
            cum_contracts: contracts,
            price,
            fee,
            fee_asset: terms.spec().settlement_asset().clone(),
            at_ms,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn roll_fill_of(
    terms: &FutureTerms,
    leg: RollLeg,
    attempt: Option<u32>,
    exec_id: &str,
    side: Action,
    price: Decimal,
    contracts: Decimal,
    at_ms: u64,
) -> RollFill {
    RollFill {
        leg,
        attempt,
        exec_id: exec_id.to_string(),
        side,
        contracts,
        cum_contracts: contracts,
        price,
        fee: dec!(0),
        fee_asset: terms.spec().settlement_asset().clone(),
        at_ms,
    }
}

/// The ledger and the plan after `steps`, with no engine involved.
fn booked_directly(mut ledger: Ledger, mut plan: RollPlan, steps: Vec<Step>) -> (Ledger, RollPlan) {
    for step in steps {
        match step {
            Step::Fill(terms, fill) => {
                let mut execution = Execution::new(
                    ExecutionId::new(fill.exec_id.clone()).unwrap(),
                    InstrumentId::new(terms.spec().id().as_str()).unwrap(),
                    fill.side,
                    fill.price,
                    fill.contracts,
                )
                .with_executed_at(fill.at_ms);
                if !fill.fee.is_zero() {
                    execution =
                        execution.with_fee(Asset::new(fill.fee_asset.as_str()).unwrap(), fill.fee);
                }
                ledger
                    .book_execution_allowing_overdraft(&execution, None)
                    .unwrap();
                plan.apply(&RollEvent::Fill(fill)).unwrap();
            }
            Step::Event(event) => {
                plan.apply(&event).unwrap();
            }
            Step::Settle(terms, price, contracts) => {
                terms.settle_in_ledger(&mut ledger, &price).unwrap();
                plan.apply(&RollEvent::NearSettled { price, contracts })
                    .unwrap();
            }
        }
    }
    (ledger, plan)
}

/// The run and the direct booking end in the same ledger and the same plan.
fn agree(ledger: &Ledger, plan: &RollPlan, direct_ledger: &Ledger, direct_plan: &RollPlan) {
    assert_eq!(ledger.snapshot(), direct_ledger.snapshot());
    assert_eq!(ledger.journal(), direct_ledger.journal());
    assert_eq!(ledger.expiry_journal(), direct_ledger.expiry_journal());
    assert_eq!(plan.state(), direct_plan.state());
    assert_eq!(
        plan.position(RollLeg::Near),
        direct_plan.position(RollLeg::Near)
    );
    assert_eq!(
        plan.position(RollLeg::Far),
        direct_plan.position(RollLeg::Far)
    );
    assert_eq!(plan.fills(), direct_plan.fills());
    assert_eq!(plan.fees(), direct_plan.fees());
    assert_eq!(plan.exposure_gap(), direct_plan.exposure_gap());
    assert_eq!(plan.near_settled(), direct_plan.near_settled());
    let reconciled = plan.reconcile(ledger).unwrap();
    assert!(reconciled.is_clean(), "{reconciled:?}");
    assert_eq!(reconciled, direct_plan.reconcile(direct_ledger).unwrap());
}

type Engine = ContractEngine<RollStrategy, VecDataSource, ImmediateFillModel>;

fn engine_of(
    ledger: Ledger,
    strategy: RollStrategy,
    fee_rate: Decimal,
    near_ticks: Vec<MarketTick>,
    far_ticks: Vec<MarketTick>,
) -> Engine {
    ContractEngine::new(
        ContractConfig::new(dec!(1000), fee_rate, 8),
        ledger,
        strategy,
    )
    .with_contract(near(), data(near_ticks), ImmediateFillModel::new())
    .unwrap()
    .with_contract(far(), data(far_ticks), ImmediateFillModel::new())
    .unwrap()
}

/// Rejects the requests at these positions, counted from zero.
#[derive(Debug, Clone)]
struct Refusing(Vec<usize>, usize);

impl PretradePolicy for Refusing {
    fn check(&mut self, _request: &PretradeRequest) -> PretradeDecision {
        let at = self.1;
        self.1 += 1;
        if self.0.contains(&at) {
            PretradeDecision::Rejected {
                reason: "refused".to_string(),
            }
        } else {
            PretradeDecision::Accepted
        }
    }
    fn reset(&mut self) {
        self.1 = 0;
    }
    fn in_contracts(&self) -> bool {
        true
    }
}

fn full_near_ticks() -> Vec<MarketTick> {
    vec![
        tick(10, dec!(60990), dec!(61010)),
        tick(20, dec!(61000), dec!(61020)),
    ]
}

fn full_far_ticks() -> Vec<MarketTick> {
    vec![tick(30, dec!(61980), dec!(62000))]
}

/// Sells the near at 61000 on tick 20 and buys the far at 62000 on tick 30.
fn full_steps(near_fee: Decimal, far_fee: Decimal) -> Vec<Step> {
    vec![
        fill_of(
            &near(),
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(10),
            near_fee,
            20,
        ),
        fill_of(
            &far(),
            RollLeg::Far,
            Some(0),
            "sim-2",
            Action::Buy,
            dec!(62000),
            dec!(10),
            far_fee,
            30,
        ),
    ]
}

#[test]
fn a_full_roll_agrees_with_the_same_plan_booked_directly() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    let strategy = RollStrategy::new(plan.clone(), 100).with_guard(guard.clone());
    let mut engine = engine_of(
        ledger.clone(),
        strategy,
        dec!(0.0001),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_pretrade_policy(guard.clone())
    .unwrap();
    let result = engine.run();
    // 0.0001 of 610000 and of 620000.
    let (direct_ledger, direct_plan) =
        booked_directly(ledger, plan, full_steps(dec!(61), dec!(62)));
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    assert_eq!(engine.strategy().plan().exposure_gap().unwrap(), dec!(0));
    // Closing 10 from 60000 at 61000, and 10 long from 62000 marked 61990.
    assert_eq!(result.total_pnl, dec!(9900));
    assert_eq!(result.total_fees, dec!(123));
    assert_eq!(result.net_pnl, dec!(9777));
    assert_eq!(result.equity_curve.last(), Some(&(30, dec!(10777))));
    assert!(result.is_consistent(), "{result:?}");
    // The guard saw it all, and the plan's last state.
    assert_eq!(
        guard.with_guard(|g| g.position(RollLeg::Near)).unwrap(),
        Ok(dec!(0))
    );
    assert_eq!(
        guard.with_guard(|g| g.position(RollLeg::Far)).unwrap(),
        Ok(dec!(10))
    );
    assert_eq!(guard.with_guard(RollGuard::latest_ms).unwrap(), 30);
    // Both orders went through the guard, naming their contract.
    let decided: Vec<_> = engine
        .pretrade_decisions()
        .iter()
        .map(|r| (r.request.instrument.clone().unwrap(), r.submitted))
        .collect();
    assert_eq!(
        decided,
        vec![
            ("NEAR".to_string(), dec!(10)),
            ("FAR".to_string(), dec!(10))
        ]
    );
}

#[test]
fn a_roll_with_latency_agrees_when_it_fills_on_the_same_ticks() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let strategy = RollStrategy::new(plan.clone(), 100);
    let mut engine = engine_of(
        ledger.clone(),
        strategy,
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_latency(
        SimLatencyConfig::new()
            .with_observation_ms(1)
            .with_submit_ms(2)
            .with_cancel_ms(3)
            .with_ack_ms(4),
    )
    .unwrap();
    let result = engine.run();
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, full_steps(dec!(0), dec!(0)));
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_leg_rejected_then_resubmitted_agrees() {
    let ledger = usdt_ledger();
    let policy = RollFailurePolicy::Complete {
        max_resubmits: 1,
        then: RollFallback::Abort,
    };
    let plan = plan_on(&ledger, policy);
    let strategy = RollStrategy::new(plan.clone(), 100);
    let mut engine = engine_of(
        ledger.clone(),
        strategy,
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    // The far leg's attempt 0, sent second.
    .with_pretrade_policy(Refusing(vec![1], 0))
    .unwrap();
    let result = engine.run();
    let steps = vec![
        Step::Event(RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: 10,
        }),
        fill_of(
            &near(),
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(10),
            dec!(0),
            20,
        ),
        fill_of(
            &far(),
            RollLeg::Far,
            Some(1),
            "sim-2",
            Action::Buy,
            dec!(62000),
            dec!(10),
            dec!(0),
            30,
        ),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    assert!(result.is_consistent(), "{result:?}");
}

/// The far leg fills, the near one does not cross by the deadline, is cancelled, and what
/// is left settles at the expiry.
#[test]
fn a_residual_left_at_the_deadline_and_settled_at_the_expiry_agrees() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let strategy = RollStrategy::new(plan.clone(), 100);
    let mut engine = engine_of(
        ledger.clone(),
        strategy,
        dec!(0),
        vec![
            tick(10, dec!(59000), dec!(59010)),
            tick(DEADLINE_MS, dec!(59000), dec!(59010)),
        ],
        full_far_ticks(),
    )
    .with_settlement(settlement(&near(), dec!(61500), 1100))
    .unwrap();
    let result = engine.run();
    let steps = vec![
        fill_of(
            &far(),
            RollLeg::Far,
            Some(0),
            "sim-1",
            Action::Buy,
            dec!(62000),
            dec!(10),
            dec!(0),
            30,
        ),
        Step::Event(RollEvent::Tick { at_ms: DEADLINE_MS }),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: DEADLINE_MS,
        }),
        Step::Settle(near(), settlement(&near(), dec!(61500), 1100), dec!(10)),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().position(RollLeg::Near), dec!(0));
    assert_eq!(engine.strategy().plan().near_settled(), Some(dec!(10)));
    // Settled from 60000 at 61500, and 10 long from 62000 marked 61990.
    assert_eq!(result.total_pnl, dec!(14900));
    assert!(result.is_consistent(), "{result:?}");
}

/// With no room for both, the guard rejects the far leg. Near fills before that reaches the
/// plan, and the unwind crosses the near contract's latest book.
fn unwinding(near_ticks: Vec<MarketTick>, max_book_age_ms: u64) -> (Ledger, RollPlan, Engine) {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(10)).unwrap());
    let strategy = RollStrategy::new(plan.clone(), max_book_age_ms).with_guard(guard.clone());
    let engine = engine_of(ledger.clone(), strategy, dec!(0), near_ticks, vec![])
        .with_pretrade_policy(guard)
        .unwrap()
        .with_latency(SimLatencyConfig::new().with_ack_ms(15))
        .unwrap();
    (ledger, plan, engine)
}

#[test]
fn an_unwind_crosses_the_latest_book_of_its_leg_and_agrees() {
    let (ledger, plan, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        100,
    );
    let result = engine.run();
    let steps = vec![
        Step::Event(RollEvent::Rejected {
            leg: RollLeg::Far,
            attempt: 0,
            at_ms: 10,
        }),
        fill_of(
            &near(),
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(10),
            dec!(0),
            20,
        ),
        fill_of(
            &near(),
            RollLeg::Near,
            None,
            "sim-2",
            Action::Buy,
            dec!(61010),
            dec!(10),
            dec!(0),
            50,
        ),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().state(), RollState::UnwindRequested);
    assert_eq!(engine.strategy().plan().position(RollLeg::Near), dec!(10));
    assert!(result.is_consistent(), "{result:?}");
}

fn faulted_with(result: &ContractRunResult, engine: &Engine, reason: &str) {
    assert!(result.fault_count > 0, "{result:?}");
    assert!(!result.is_consistent());
    assert!(
        engine
            .faults()
            .iter()
            .any(|f| f.to_string().contains(reason)),
        "{:?}",
        engine.faults()
    );
}

#[test]
fn an_unwind_needs_a_book_of_its_leg_recent_enough_with_size() {
    // The latest book is 5 ms old at the decision, past 1.
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        1,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "ms old");

    // Exactly as old as allowed, it is used.
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        5,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 2);
    assert!(result.is_consistent(), "{result:?}");

    // No size on the side it needs.
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            MarketTick::new(30, dec!(61000), dec!(5), dec!(61010), dec!(0)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "short of 10");

    // A price that is not one, on the side it needs.
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            MarketTick::new(30, dec!(61000), dec!(5), dec!(0), dec!(5)),
        ],
        100,
    );
    let result = engine.run();
    faulted_with(&result, &engine, "no price");
}

/// The far leg fills before its tick is observed, so the far book is not known yet, and
/// the near book is not used for it.
#[test]
fn an_unwind_never_uses_another_leg_or_a_book_not_observed_yet() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    let strategy = RollStrategy::new(plan, 10_000);
    let mut engine = engine_of(
        ledger,
        strategy,
        dec!(0),
        vec![tick(10, dec!(59000), dec!(59010))],
        vec![
            tick(120, dec!(61980), dec!(62000)),
            tick(300, dec!(61980), dec!(62000)),
        ],
    )
    // The near leg's attempt 0, sent first.
    .with_pretrade_policy(Refusing(vec![0], 0))
    .unwrap()
    .with_latency(
        SimLatencyConfig::new()
            .with_observation_ms(100)
            .with_cancel_ms(50),
    )
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(result.trades[0].contract.as_str(), "FAR");
    faulted_with(&result, &engine, "no book of the Far leg");

    // A reset forgets the books the last run saw.
    engine.reset();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "no book of the Far leg");
}

#[test]
fn an_unwind_that_does_not_happen_is_a_fault() {
    // Rejected: the far leg first, then the unwind.
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        vec![],
    )
    .with_pretrade_policy(Refusing(vec![1, 2], 0))
    .unwrap()
    .with_latency(SimLatencyConfig::new().with_ack_ms(15))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "unwind of the Near leg was rejected");

    // Cancelled by the venue at the expiry, as the book moved away.
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61100), dec!(61110)),
            tick(NEAR_MS, dec!(61100), dec!(61110)),
        ],
        vec![],
    )
    .with_pretrade_policy(Refusing(vec![1], 0))
    .unwrap()
    .with_latency(SimLatencyConfig::new().with_ack_ms(15))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "cancelled with 10 left");
}

/// With no room for both, the plan's policy deals with the rejection: with nothing filled,
/// it aborts and keeps its position.
#[test]
fn a_guard_with_no_room_for_both_legs_rejects_the_far_one() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(10)).unwrap());
    let strategy = RollStrategy::new(plan, 100).with_guard(guard.clone());
    let mut engine = engine_of(
        ledger,
        strategy,
        dec!(0),
        vec![
            tick(10, dec!(59000), dec!(59010)),
            tick(20, dec!(61000), dec!(61020)),
        ],
        full_far_ticks(),
    )
    .with_pretrade_policy(guard)
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert_eq!(engine.strategy().plan().state(), RollState::Aborted);
    assert_eq!(engine.strategy().plan().position(RollLeg::Near), dec!(10));
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_tick_before_the_plan_or_of_another_contract_starts_nothing() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![tick(CREATED_MS - 1, dec!(61000), dec!(61010))],
        vec![],
    );
    let result = engine.run();
    assert!(engine.execution_reports().is_empty());
    assert!(result.is_consistent(), "{result:?}");

    // The first tick at the plan's creation starts it.
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![tick(CREATED_MS, dec!(59000), dec!(59010))],
        vec![],
    );
    engine.run();
    assert_eq!(engine.execution_reports().len(), 2);

    // A third contract in the run.
    let other = terms_of(spec_of("OTHER", FAR_MS));
    let mut ledger = usdt_ledger();
    ledger
        .register_instrument(
            Instrument::from_spec(
                other.spec(),
                InstrumentScales {
                    price_scale: 8,
                    contract_scale: 8,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![],
        vec![],
    )
    .with_contract(
        other,
        data(vec![tick(50, dec!(100), dec!(102))]),
        ImmediateFillModel::new(),
    )
    .unwrap();
    engine.run();
    assert!(engine.execution_reports().is_empty());
}

#[test]
fn a_reset_gives_the_same_run_again_with_guard_and_plan_restored() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    let strategy = RollStrategy::new(plan.clone(), 100).with_guard(guard.clone());
    let mut engine = engine_of(
        ledger,
        strategy,
        dec!(0.0001),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_pretrade_policy(guard.clone())
    .unwrap();
    let first = engine.run();
    let snapshot = engine.ledger().snapshot();
    let fills = engine.strategy().plan().fills().to_vec();
    engine.reset();
    assert_eq!(engine.strategy().plan().fills(), plan.fills());
    assert_eq!(
        guard.with_guard(|g| g.position(RollLeg::Near)).unwrap(),
        Ok(dec!(10))
    );
    let second = engine.run();
    assert_eq!(second.net_pnl, first.net_pnl);
    assert_eq!(second.equity_curve, first.equity_curve);
    assert_eq!(second.trades, first.trades);
    assert_eq!(engine.ledger().snapshot(), snapshot);
    assert_eq!(engine.strategy().plan().fills(), &fills[..]);
    assert!(second.is_consistent(), "{second:?}");
}

/// Panics holding the lock, so that it is poisoned.
fn poison(guard: &SharedRollGuard) {
    let held = guard.clone();
    let _ = std::thread::spawn(move || {
        let _ = held.with_guard(|_| panic!("in the lock"));
    })
    .join();
}

#[test]
fn a_poisoned_guard_closes_new_orders_and_a_reset_opens_them() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    poison(&guard);
    assert!(guard.with_guard(|_| ()).is_err());
    let strategy = RollStrategy::new(plan, 100).with_guard(guard.clone());
    let mut engine = engine_of(
        ledger,
        strategy,
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_pretrade_policy(guard.clone())
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    faulted_with(&result, &engine, "roll guard");
    assert!(!engine.pretrade_decisions().is_empty());
    assert!(
        engine
            .pretrade_decisions()
            .iter()
            .all(|r| matches!(r.decision, PretradeDecision::Rejected { .. }))
    );

    engine.reset();
    assert!(guard.with_guard(|_| ()).is_ok());
    let result = engine.run();
    assert_eq!(result.num_trades, 2);
    assert!(result.is_consistent(), "{result:?}");
}

/// Poisons the guard when it hears of its first fill.
#[derive(Debug)]
struct PoisonAtFill(RollStrategy, SharedRollGuard);

impl ContractStrategy for PoisonAtFill {
    fn on_tick(
        &mut self,
        contract: &SpecId,
        tick: &MarketTick,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.0.on_tick(contract, tick, view)
    }
    fn on_report(&mut self, report: &ContractReport, view: &ContractView) -> Vec<ContractCommand> {
        self.0.on_report(report, view)
    }
    fn on_fill(&mut self, fill: &ContractFill, view: &ContractView) -> Vec<ContractCommand> {
        poison(&self.1);
        self.0.on_fill(fill, view)
    }
    fn take_fault(&mut self) -> Option<MMError> {
        self.0.take_fault()
    }
    fn reset(&mut self) {
        self.0.reset();
    }
}

#[test]
fn a_guard_poisoned_mid_run_loses_no_real_fill() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    let strategy = PoisonAtFill(
        RollStrategy::new(plan, 100).with_guard(guard.clone()),
        guard.clone(),
    );
    let mut engine = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 8),
        ledger,
        strategy,
    )
    .with_contract(near(), data(full_near_ticks()), ImmediateFillModel::new())
    .unwrap()
    .with_contract(far(), data(full_far_ticks()), ImmediateFillModel::new())
    .unwrap()
    .with_pretrade_policy(guard)
    .unwrap();
    let result = engine.run();
    // The far order was resting, and its fill stands, in the ledger and in the plan.
    assert_eq!(result.num_trades, 2);
    assert_eq!(engine.ledger().journal().len(), 3);
    assert_eq!(engine.strategy().0.plan().position(RollLeg::Far), dec!(10));
    assert!(result.fault_count > 0);
    assert!(!result.is_consistent());
    assert!(
        engine
            .faults()
            .iter()
            .any(|f| f.to_string().contains("roll guard")),
        "{:?}",
        engine.faults()
    );
}

/// Thin: 5 at the ask, for an unwind of 10. Off the grid: an ask of 61010.5, which the
/// engine does not round.
#[test]
fn an_unwind_does_not_invent_liquidity_or_round_a_price() {
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            MarketTick::new(30, dec!(61000), dec!(5), dec!(61010), dec!(5)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "short of 10");

    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010.5)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "was rejected");
}

/// A book of tick 20 and a fill of tick 20 delivered at 520: at the decision the book is
/// 500 old, whatever the fill's own time says.
#[test]
fn the_age_of_a_book_is_judged_when_the_decision_is_taken() {
    let run = |max_book_age_ms: u64| {
        let ledger = usdt_ledger();
        let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
        let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(10)).unwrap());
        let strategy = RollStrategy::new(plan, max_book_age_ms).with_guard(guard.clone());
        let mut engine = engine_of(
            ledger,
            strategy,
            dec!(0),
            vec![
                tick(10, dec!(60990), dec!(61010)),
                tick(20, dec!(61000), dec!(61020)),
                tick(600, dec!(61000), dec!(61020)),
            ],
            vec![],
        )
        .with_pretrade_policy(guard)
        .unwrap()
        .with_latency(SimLatencyConfig::new().with_ack_ms(500))
        .unwrap();
        let result = engine.run();
        (result, engine)
    };
    let (result, engine) = run(499);
    faulted_with(&result, &engine, "500 ms old");
    let (result, engine) = run(500);
    assert_eq!(result.fault_count, 0, "{:?}", engine.faults());
}

/// A near order still live at the deadline is cancelled there, with no tick at that time,
/// so a later tick that would fill it does not.
#[test]
fn the_deadline_cancels_with_no_tick_at_that_time() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
        dec!(0),
        vec![
            tick(10, dec!(59000), dec!(59010)),
            tick(DEADLINE_MS + 5, dec!(61000), dec!(61020)),
        ],
        vec![tick(DEADLINE_MS + 6, dec!(61980), dec!(62000))],
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    let steps = vec![
        Step::Event(RollEvent::Tick { at_ms: DEADLINE_MS }),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: DEADLINE_MS,
        }),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Far,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: DEADLINE_MS,
        }),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert!(result.is_consistent(), "{result:?}");
}

/// A report delivered exactly at the deadline does not take the place of its wake, and the
/// orders are cancelled before later ticks that would fill them.
#[test]
fn an_ack_at_the_deadline_cannot_suppress_the_deadline_wake() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![
            tick(10, dec!(59000), dec!(59010)),
            tick(950, dec!(61000), dec!(61020)),
        ],
        vec![tick(951, dec!(61980), dec!(62000))],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(890))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert!(result.is_consistent(), "{result:?}");
    let canceled = engine
        .execution_reports()
        .iter()
        .filter(|r| r.report.kind == market_maker_rs::backtest::ExecutionKind::Canceled)
        .count();
    assert_eq!(canceled, 2);
}

/// A fill delivered exactly at the deadline, with the far leg still live.
#[test]
fn a_fill_delivered_at_the_deadline_cannot_suppress_the_wake() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(400, dec!(61000), dec!(61020)),
        ],
        vec![tick(950, dec!(61980), dec!(62000))],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(500))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(result.trades[0].contract.as_str(), "NEAR");
    assert_eq!(engine.strategy().plan().position(RollLeg::Far), dec!(0));
}

#[test]
fn a_first_tick_past_the_deadline_sends_nothing_and_the_plan_ends() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![tick(DEADLINE_MS, dec!(61000), dec!(61020))],
        vec![tick(DEADLINE_MS + 1, dec!(61980), dec!(62000))],
    );
    let result = engine.run();
    assert!(engine.execution_reports().is_empty());
    assert_eq!(engine.strategy().plan().state(), RollState::Aborted);
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_plan_that_saw_events_already_sends_no_new_attempts() {
    let ledger = usdt_ledger();
    let mut plan = plan_on(&ledger, RollFailurePolicy::Abort);
    plan.apply(&RollEvent::Tick { at_ms: 6 }).unwrap();
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    );
    let result = engine.run();
    assert!(engine.execution_reports().is_empty());
    faulted_with(&result, &engine, "has seen events already");
}

/// A near fill still on its way when the settlement is published reaches the plan first,
/// and the settlement closed nothing, as the ledger had booked the fill.
#[test]
fn a_fill_on_its_way_at_the_settlement_agrees() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
        dec!(0),
        vec![
            tick(10, dec!(59000), dec!(59010)),
            tick(800, dec!(61000), dec!(61020)),
        ],
        vec![tick(1500, dec!(62990), dec!(63000))],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(300))
    .unwrap()
    .with_settlement(settlement(&near(), dec!(61500), NEAR_MS))
    .unwrap();
    let result = engine.run();
    let steps = vec![
        Step::Event(RollEvent::Tick { at_ms: DEADLINE_MS }),
        fill_of(
            &near(),
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(10),
            dec!(0),
            800,
        ),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Far,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: 1200,
        }),
        Step::Settle(near(), settlement(&near(), dec!(61500), NEAR_MS), dec!(0)),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().near_settled(), Some(dec!(0)));
    assert!(result.is_consistent(), "{result:?}");
}

/// Approves less than it was asked for on the request at one position, counted from zero.
#[derive(Debug, Clone)]
struct Reducing {
    at: usize,
    to: Decimal,
    seen: usize,
}

impl PretradePolicy for Reducing {
    fn check(&mut self, _request: &PretradeRequest) -> PretradeDecision {
        let at = self.seen;
        self.seen += 1;
        if at == self.at {
            PretradeDecision::Reduced {
                quantity: self.to,
                reason: "room".to_string(),
            }
        } else {
            PretradeDecision::Accepted
        }
    }
    fn reset(&mut self) {
        self.seen = 0;
    }
    fn in_contracts(&self) -> bool {
        true
    }
}

/// With an exposure limit between the start and the room for both, the guard reduces the
/// far leg. When what it approved fills, the plan hears the attempt ended, and gives up.
#[test]
fn a_far_leg_the_guard_reduces_ends_the_attempt_and_the_plan_aborts() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(12)).unwrap());
    let strategy = RollStrategy::new(plan.clone(), 100).with_guard(guard.clone());
    let mut engine = engine_of(
        ledger.clone(),
        strategy,
        dec!(0),
        vec![tick(10, dec!(59000), dec!(59010))],
        full_far_ticks(),
    )
    .with_pretrade_policy(guard)
    .unwrap();
    let result = engine.run();
    assert_eq!(engine.pretrade_decisions()[1].submitted, dec!(2));
    assert_eq!(result.trades[0].quantity, dec!(2));
    let far_fill = RollFill {
        cum_contracts: dec!(2),
        ..roll_fill_of(
            &far(),
            RollLeg::Far,
            Some(0),
            "sim-1",
            Action::Buy,
            dec!(62000),
            dec!(2),
            30,
        )
    };
    let steps = vec![
        Step::Fill(far(), far_fill),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Far,
            attempt: 0,
            cum_contracts: dec!(2),
            at_ms: 30,
        }),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(0),
            at_ms: 30,
        }),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().state(), RollState::Aborted);
    assert_ne!(engine.strategy().plan().exposure_gap().unwrap(), dec!(0));
    assert!(result.is_consistent(), "{result:?}");
}

/// A reduced order filling in parts ends only with its last part, and the rest is resent.
#[test]
fn a_reduced_near_leg_is_resent_for_the_rest() {
    let ledger = usdt_ledger();
    let policy = RollFailurePolicy::Complete {
        max_resubmits: 1,
        then: RollFallback::Abort,
    };
    let plan = plan_on(&ledger, policy);
    let mut engine = engine_of(
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
        dec!(0),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(25, dec!(61000), dec!(61020)),
        ],
        full_far_ticks(),
    )
    .with_pretrade_policy(Reducing {
        at: 0,
        to: dec!(4),
        seen: 0,
    })
    .unwrap();
    let result = engine.run();
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    let near_quantities: Vec<Decimal> = result
        .trades
        .iter()
        .filter(|t| t.contract.as_str() == "NEAR")
        .map(|t| t.quantity)
        .collect();
    assert_eq!(near_quantities, vec![dec!(4), dec!(6)]);
    let first = RollFill {
        cum_contracts: dec!(4),
        ..roll_fill_of(
            &near(),
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(4),
            20,
        )
    };
    let second = RollFill {
        cum_contracts: dec!(6),
        ..roll_fill_of(
            &near(),
            RollLeg::Near,
            Some(1),
            "sim-2",
            Action::Sell,
            dec!(61000),
            dec!(6),
            25,
        )
    };
    let steps = vec![
        Step::Fill(near(), first),
        Step::Event(RollEvent::CancelAck {
            leg: RollLeg::Near,
            attempt: 0,
            cum_contracts: dec!(4),
            at_ms: 20,
        }),
        Step::Fill(near(), second),
        fill_of(
            &far(),
            RollLeg::Far,
            Some(0),
            "sim-3",
            Action::Buy,
            dec!(62000),
            dec!(10),
            dec!(0),
            30,
        ),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_reduced_unwind_is_a_fault() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    #[derive(Debug, Clone)]
    struct Both;
    impl PretradePolicy for Both {
        fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
            if request.instrument.as_deref() == Some("FAR") {
                return PretradeDecision::Rejected {
                    reason: "refused".to_string(),
                };
            }
            if request.side == Action::Buy {
                return PretradeDecision::Reduced {
                    quantity: dec!(4),
                    reason: "room".to_string(),
                };
            }
            PretradeDecision::Accepted
        }
        fn reset(&mut self) {}
        fn in_contracts(&self) -> bool {
            true
        }
    }
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        vec![],
    )
    .with_pretrade_policy(Both)
    .unwrap()
    .with_latency(SimLatencyConfig::new().with_ack_ms(15))
    .unwrap();
    let result = engine.run();
    faulted_with(&result, &engine, "was reduced to 4");
}

/// The observation of its first tick comes after the settlement of the near contract.
#[test]
fn settlement_before_the_first_observation_does_not_cancel_unsent_orders() {
    let run = |ledger: Ledger| {
        let plan = plan_on(&ledger, RollFailurePolicy::Abort);
        let mut engine = engine_of(
            ledger,
            RollStrategy::new(plan, 100),
            dec!(0),
            vec![tick(10, dec!(61000), dec!(61020))],
            vec![tick(1500, dec!(61980), dec!(62000))],
        )
        .with_latency(SimLatencyConfig::new().with_observation_ms(1000))
        .unwrap()
        .with_settlement(settlement(&near(), dec!(60000), NEAR_MS))
        .unwrap();
        let result = engine.run();
        (engine, result)
    };
    let (engine, result) = run(usdt_ledger());
    assert_eq!(result.num_trades, 0);
    assert!(engine.execution_reports().is_empty());
    assert_eq!(result.fault_count, 0, "{:?}", engine.faults());
    assert!(result.is_consistent(), "{result:?}");
    let plan = engine.strategy().plan();
    assert_eq!(plan.state(), RollState::Aborted);
    assert_eq!(plan.position(RollLeg::Near), dec!(0));
    assert_eq!(plan.near_settled(), Some(dec!(10)));
    assert!(plan.reconcile(engine.ledger()).unwrap().is_clean());

    // The same with less held than the plan says: it is not fit to run.
    let (engine, result) = run(opened(
        &near(),
        &far(),
        ["BTC", "USDT"],
        dec!(60000),
        dec!(5),
    ));
    assert!(engine.execution_reports().is_empty());
    assert_eq!(result.fault_count, 1, "{:?}", engine.faults());
    faulted_with_settled(&result, &engine);
}

fn faulted_with_settled(result: &ContractRunResult, engine: &Engine) {
    assert!(!result.is_consistent());
    assert!(
        engine.faults()[0]
            .to_string()
            .contains("the ledger holds 5"),
        "{:?}",
        engine.faults()
    );
    assert_eq!(engine.strategy().plan().near_settled(), None);
}

/// With no tick of the near contract at all, its settlement comes first, under any policy,
/// and it agrees with the same plan booked directly. A later first tick sends nothing.
#[test]
fn a_settlement_with_no_tick_before_it_agrees_under_each_policy() {
    let policies = [
        (RollFailurePolicy::Abort, RollState::Aborted),
        (RollFailurePolicy::Unwind, RollState::UnwindRequested),
        (
            RollFailurePolicy::Complete {
                max_resubmits: 2,
                then: RollFallback::Abort,
            },
            RollState::Aborted,
        ),
    ];
    for (policy, state) in policies {
        let ledger = usdt_ledger();
        let plan = plan_on(&ledger, policy);
        let price = settlement(&near(), dec!(61500), NEAR_MS);
        let mut engine = engine_of(
            ledger.clone(),
            RollStrategy::new(plan.clone(), 100),
            dec!(0),
            vec![],
            vec![tick(1500, dec!(61980), dec!(62000))],
        )
        .with_settlement(price.clone())
        .unwrap();
        let result = engine.run();
        let steps = vec![
            Step::Settle(near(), price, dec!(10)),
            Step::Event(RollEvent::CancelAck {
                leg: RollLeg::Far,
                attempt: 0,
                cum_contracts: dec!(0),
                at_ms: NEAR_MS,
            }),
            Step::Event(RollEvent::CancelAck {
                leg: RollLeg::Near,
                attempt: 0,
                cum_contracts: dec!(0),
                at_ms: NEAR_MS,
            }),
        ];
        let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
        agree(
            engine.ledger(),
            engine.strategy().plan(),
            &direct_ledger,
            &direct_plan,
        );
        assert_eq!(result.fault_count, 0, "{:?}", engine.faults());
        assert!(result.is_consistent(), "{result:?}");
        assert!(engine.execution_reports().is_empty());
        assert_eq!(engine.strategy().plan().state(), state);
    }
}

#[test]
fn a_settlement_first_still_checks_the_plan() {
    let run = |ledger: Ledger, plan: RollPlan| {
        let mut engine = engine_of(
            ledger,
            RollStrategy::new(plan, 100),
            dec!(0),
            vec![],
            vec![tick(1500, dec!(61980), dec!(62000))],
        )
        .with_settlement(settlement(&near(), dec!(61500), NEAR_MS))
        .unwrap();
        let result = engine.run();
        assert!(engine.execution_reports().is_empty());
        assert_eq!(engine.strategy().plan().near_settled(), None);
        (engine, result)
    };
    // A plan already in progress.
    let ledger = usdt_ledger();
    let mut plan = plan_on(&ledger, RollFailurePolicy::Abort);
    plan.apply(&RollEvent::Tick { at_ms: 6 }).unwrap();
    let (engine, result) = run(ledger, plan);
    assert_eq!(result.fault_count, 1, "{:?}", engine.faults());
    faulted_with(&result, &engine, "has seen events already");

    // Near: the ledger closed 5, and the plan says 10.
    let ledger = opened(&near(), &far(), ["BTC", "USDT"], dec!(60000), dec!(5));
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let (engine, result) = run(ledger, plan);
    assert_eq!(result.fault_count, 1, "{:?}", engine.faults());
    faulted_with(&result, &engine, "the Near leg, and the ledger holds 5");

    // Far: the ledger holds 1, and the plan says 0.
    let mut ledger = usdt_ledger();
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("held").unwrap(),
                InstrumentId::new("FAR").unwrap(),
                Action::Buy,
                dec!(60000),
                dec!(1),
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let (engine, result) = run(ledger, plan);
    assert_eq!(result.fault_count, 1, "{:?}", engine.faults());
    faulted_with(&result, &engine, "the Far leg, and the ledger holds 1");
}

/// A sell unwind crosses the bid: the far leg fills alone while the near one is refused,
/// and the far excess sells at the far bid, whose own size is what counts.
fn far_excess(far_book: MarketTick) -> (Ledger, RollPlan, Engine) {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Unwind);
    let engine = engine_of(
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
        dec!(0),
        vec![tick(10, dec!(59000), dec!(59010))],
        vec![
            far_book,
            tick(30, dec!(61990), dec!(62000)),
            tick(40, dec!(61980), dec!(61990)),
        ],
    )
    // The near leg's attempt 0, sent first.
    .with_pretrade_policy(Refusing(vec![0], 0))
    .unwrap()
    .with_latency(SimLatencyConfig::new().with_cancel_ms(50))
    .unwrap();
    (ledger, plan, engine)
}

#[test]
fn a_sell_unwind_crosses_the_bid_of_its_leg_and_agrees() {
    // A bid of 61980 for 10, with an ask too thin to matter.
    let (ledger, plan, mut engine) = far_excess(MarketTick::new(
        20,
        dec!(61980),
        dec!(10),
        dec!(62010),
        dec!(1),
    ));
    let result = engine.run();
    let steps = vec![
        Step::Event(RollEvent::Rejected {
            leg: RollLeg::Near,
            attempt: 0,
            at_ms: 10,
        }),
        fill_of(
            &far(),
            RollLeg::Far,
            Some(0),
            "sim-1",
            Action::Buy,
            dec!(62000),
            dec!(10),
            dec!(0),
            30,
        ),
        fill_of(
            &far(),
            RollLeg::Far,
            None,
            "sim-2",
            Action::Sell,
            dec!(61980),
            dec!(10),
            dec!(0),
            40,
        ),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(result.trades[1].price, dec!(61980));
    assert!(result.is_consistent(), "{result:?}");

    // A bid too thin, with an ask that would have been enough.
    let (_, _, mut engine) = far_excess(MarketTick::new(
        20,
        dec!(61980),
        dec!(9),
        dec!(62010),
        dec!(100),
    ));
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    faulted_with(&result, &engine, "has 9 to Sell, short of 10");
}

/// Of two ticks at one time, the later is the book.
#[test]
fn of_two_books_at_one_time_an_unwind_reads_the_later() {
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(30, dec!(61000), dec!(61011)),
            tick(50, dec!(61000), dec!(61011)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.trades[1].price, dec!(61011));
}

/// A first run past the deadline must not leave a clock that stops the next one sending.
#[test]
fn a_reset_forgets_the_clock_of_the_last_run() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        vec![
            tick(30, dec!(61980), dec!(62000)),
            tick(DEADLINE_MS + 50, dec!(61980), dec!(62000)),
        ],
    );
    let first = engine.run();
    assert_eq!(first.num_trades, 2);
    engine.reset();
    let second = engine.run();
    assert_eq!(second.trades, first.trades);
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
}

/// The far contract settling before the first observation still counts as held at the start.
#[test]
fn a_far_settlement_before_the_first_observation_is_no_mismatch() {
    let mut ledger = usdt_ledger();
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("held").unwrap(),
                InstrumentId::new("FAR").unwrap(),
                Action::Buy,
                dec!(60000),
                dec!(3),
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
    let plan = RollPlan::new(RollRequest {
        far_position: dec!(3),
        ..request(RollFailurePolicy::Abort)
    })
    .unwrap()
    .anchored_to(&ledger);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        vec![tick(10, dec!(61000), dec!(61020))],
        vec![tick(3100, dec!(61980), dec!(62000))],
    )
    .with_latency(SimLatencyConfig::new().with_observation_ms(3000))
    .unwrap()
    .with_settlement(settlement(&far(), dec!(62000), FAR_MS))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.fault_count, 0, "{:?}", engine.faults());
    assert!(engine.execution_reports().is_empty());
    assert_eq!(engine.strategy().plan().state(), RollState::Aborted);
    assert!(line_settled(&result, "FAR"));
}

fn line_settled(result: &ContractRunResult, name: &str) -> bool {
    result
        .contracts
        .iter()
        .find(|c| c.contract.as_str() == name)
        .unwrap()
        .settled
}

/// A plan for a position the ledger does not hold would flip it.
#[test]
fn a_plan_for_positions_the_ledger_does_not_hold_sends_nothing() {
    // The ledger is long 5 of the near contract, and the plan says 10.
    let ledger = opened(&near(), &far(), ["BTC", "USDT"], dec!(60000), dec!(5));
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    );
    let result = engine.run();
    assert!(engine.execution_reports().is_empty());
    assert_eq!(result.num_trades, 0);
    faulted_with(&result, &engine, "the ledger holds 5");

    // Refused on its first tick, it does not hear of the settlement later either.
    let ledger = opened(&near(), &far(), ["BTC", "USDT"], dec!(60000), dec!(5));
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_settlement(settlement(&near(), dec!(61000), NEAR_MS))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.fault_count, 1, "{:?}", engine.faults());
    assert_eq!(engine.strategy().plan().near_settled(), None);

    // The ledger holds something on the far contract that the plan does not know of.
    let mut ledger = usdt_ledger();
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("held").unwrap(),
                InstrumentId::new("FAR").unwrap(),
                Action::Buy,
                dec!(60000),
                dec!(1),
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    );
    let result = engine.run();
    assert!(engine.execution_reports().is_empty());
    faulted_with(&result, &engine, "the Far leg, and the ledger holds 1");
}

/// A fill's identifier, however it is written, stands only for an entry on its contract.
#[test]
fn reconcile_refuses_a_named_entry_on_another_contract() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let far_fill = roll_fill_of(
        &far(),
        RollLeg::Far,
        Some(0),
        "x",
        Action::Buy,
        dec!(62000),
        dec!(10),
        30,
    );
    let mut ledger = ledger;
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("3:FAR:x").unwrap(),
                InstrumentId::new("NEAR").unwrap(),
                Action::Buy,
                dec!(62000),
                dec!(10),
            )
            .with_executed_at(30),
            None,
        )
        .unwrap();
    let mut plan = plan;
    plan.apply(&RollEvent::Fill(far_fill)).unwrap();
    let reconciled = plan.reconcile(&ledger).unwrap();
    assert!(
        reconciled.discrepancies.iter().any(|d| matches!(
            d,
            market_maker_rs::instruments::roll::RollDiscrepancy::MissingInLedger { .. }
        )),
        "{reconciled:?}"
    );
}

/// A fill booked under the identifier `fill_execution` gives on another contract does not
/// stand for one of this contract under the same text.
#[test]
fn reconcile_reads_an_execution_only_on_its_own_contract() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    // The far fill as the plan saw it, booked in the ledger on the near contract.
    let far_fill = match fill_of(
        &far(),
        RollLeg::Far,
        Some(0),
        "sim-1",
        Action::Buy,
        dec!(62000),
        dec!(10),
        dec!(0),
        30,
    ) {
        Step::Fill(_, fill) => fill,
        _ => unreachable!(),
    };
    let (ledger, plan) = booked_directly(ledger, plan, vec![Step::Fill(near(), far_fill)]);
    let reconciled = plan.reconcile(&ledger).unwrap();
    assert!(!reconciled.is_clean());
    assert!(
        reconciled.discrepancies.iter().any(|d| matches!(
            d,
            market_maker_rs::instruments::roll::RollDiscrepancy::MissingInLedger { .. }
        )),
        "{reconciled:?}"
    );
}

#[test]
fn a_poisoned_guard_alone_rejects_and_reports() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut guard = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    poison(&guard);
    let request = PretradeRequest::new(
        "1",
        Action::Sell,
        Some(dec!(61000)),
        dec!(1),
        market_maker_rs::risk::pretrade::OrderPurpose::Other,
        10,
    );
    assert!(matches!(
        guard.check(&request),
        PretradeDecision::Rejected { ref reason } if reason.contains("roll guard")
    ));
    assert!(matches!(
        guard.on_order_event(&market_maker_rs::risk::pretrade::OrderEvent::Canceled {
            order_id: "1".to_string(),
        }),
        Err(MMError::LockPoisoned(_))
    ));
    guard.on_market_data(dec!(100), 10);

    // A strategy not holding it still sends, and the policy refuses each order.
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let policy = SharedRollGuard::new(RollGuard::new(&plan, dec!(20)).unwrap());
    poison(&policy);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_pretrade_policy(policy)
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert!(!engine.pretrade_decisions().is_empty());
    assert!(
        engine
            .pretrade_decisions()
            .iter()
            .all(|r| matches!(r.decision, PretradeDecision::Rejected { .. }))
    );
}

/// Exactly the size it needs is enough.
#[test]
fn an_unwind_takes_a_book_with_exactly_its_size() {
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            MarketTick::new(30, dec!(61000), dec!(5), dec!(61010), dec!(10)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 2);
    assert!(result.is_consistent(), "{result:?}");
}

/// An out of order tick is not the latest book, even if seen last.
#[test]
fn an_unwind_reads_the_latest_book_and_not_the_last_seen() {
    let (_, _, mut engine) = unwinding(
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(30, dec!(61000), dec!(61010)),
            tick(25, dec!(60000), dec!(60500)),
            tick(50, dec!(61000), dec!(61010)),
        ],
        100,
    );
    let result = engine.run();
    assert_eq!(result.trades[1].price, dec!(61010));
}

/// Partial fills by tick, cumulative per order, of what crosses.
#[derive(Debug, Clone)]
struct InParts;

impl FillModel for InParts {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        let crossed = match order.side {
            Action::Buy => tick.ask_price <= order.price,
            _ => tick.bid_price >= order.price,
        };
        if !crossed {
            return Ok(FillResult::NoFill);
        }
        Ok(if tick.timestamp == 20 {
            FillResult::PartialFill {
                filled_quantity: dec!(2),
                fill_price: order.price,
            }
        } else if tick.timestamp == 40 {
            FillResult::PartialFill {
                filled_quantity: dec!(2.5),
                fill_price: order.price,
            }
        } else {
            FillResult::FullFill {
                fill_price: order.price,
            }
        })
    }
    fn reset(&mut self) {}
    fn name(&self) -> &'static str {
        "in parts"
    }
}

fn in_parts(
    ledger: Ledger,
    strategy: RollStrategy,
    near_ticks: Vec<MarketTick>,
    far_ticks: Vec<MarketTick>,
) -> ContractEngine<RollStrategy, VecDataSource, InParts> {
    ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 8),
        ledger,
        strategy,
    )
    .with_contract(near(), data(near_ticks), InParts)
    .unwrap()
    .with_contract(far(), data(far_ticks), InParts)
    .unwrap()
}

/// A reduced order filling in parts ends only with its last part.
#[test]
fn a_reduced_order_filling_in_parts_ends_with_its_last_part() {
    let ledger = usdt_ledger();
    let policy = RollFailurePolicy::Complete {
        max_resubmits: 1,
        then: RollFallback::Abort,
    };
    let plan = plan_on(&ledger, policy);
    let mut engine = in_parts(
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(20, dec!(61000), dec!(61020)),
            tick(25, dec!(61000), dec!(61020)),
            tick(26, dec!(61000), dec!(61020)),
        ],
        full_far_ticks(),
    )
    .with_pretrade_policy(Reducing {
        at: 0,
        to: dec!(4),
        seen: 0,
    })
    .unwrap();
    let result = engine.run();
    let near_quantities: Vec<Decimal> = result
        .trades
        .iter()
        .filter(|t| t.contract.as_str() == "NEAR")
        .map(|t| t.quantity)
        .collect();
    assert_eq!(near_quantities, vec![dec!(2), dec!(2), dec!(6)]);
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    assert!(result.is_consistent(), "{result:?}");
    let reconciled = engine.strategy().plan().reconcile(engine.ledger()).unwrap();
    assert!(reconciled.is_clean(), "{reconciled:?}");
}

/// A fill off the lot is real, so it stays on the record, and the plan's intents stop.
#[test]
fn a_fill_the_plans_intents_refuse_is_a_fault() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = in_parts(
        ledger,
        RollStrategy::new(plan, 100),
        vec![
            tick(10, dec!(60990), dec!(61010)),
            tick(40, dec!(61000), dec!(61020)),
        ],
        vec![],
    );
    let result = engine.run();
    assert_eq!(result.trades[0].quantity, dec!(2.5));
    assert!(result.fault_count > 0);
    assert!(!result.is_consistent());
    assert!(
        engine
            .faults()
            .iter()
            .any(|f| f.to_string().contains("intents did not take a fill")),
        "{:?}",
        engine.faults()
    );
    assert_eq!(engine.strategy().plan().state(), RollState::Faulted);
    assert_eq!(engine.strategy().plan().position(RollLeg::Near), dec!(7.5));
}

/// The far contract settling later is no event of this plan.
#[test]
fn a_settlement_of_the_far_contract_does_not_reach_the_plan() {
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = engine_of(
        ledger,
        RollStrategy::new(plan, 100),
        dec!(0),
        full_near_ticks(),
        full_far_ticks(),
    )
    .with_settlement(settlement(&far(), dec!(62500), FAR_MS))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.fault_count, 0, "{:?}", engine.faults());
    assert_eq!(engine.strategy().plan().near_settled(), None);
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    assert_eq!(engine.ledger().expiry_journal().len(), 1);
}

/// What the plan cannot take is a fault, not a run that reads clean.
#[test]
fn the_plan_refusing_a_report_is_a_fault() {
    /// Sends one more order on the roll's contract than the plan knows of, with the tag of
    /// an order of the plan, and repeats a report the plan took.
    #[derive(Debug)]
    struct Repeats(RollStrategy, Option<ContractReport>);
    impl ContractStrategy for Repeats {
        fn on_tick(
            &mut self,
            contract: &SpecId,
            tick: &MarketTick,
            view: &ContractView,
        ) -> Vec<ContractCommand> {
            let commands = self.0.on_tick(contract, tick, view);
            // A cancel acknowledged with a quantity that never filled.
            if let Some(mut report) = self.1.take() {
                report.report.cumulative_quantity = dec!(3);
                self.0.on_report(&report, view);
            }
            commands
        }
        fn on_report(
            &mut self,
            report: &ContractReport,
            view: &ContractView,
        ) -> Vec<ContractCommand> {
            self.1.get_or_insert_with(|| {
                let mut report = report.clone();
                report.report.kind = market_maker_rs::backtest::ExecutionKind::Canceled;
                report
            });
            self.0.on_report(report, view)
        }
        fn on_fill(&mut self, fill: &ContractFill, view: &ContractView) -> Vec<ContractCommand> {
            self.0.on_fill(fill, view)
        }
        fn take_fault(&mut self) -> Option<MMError> {
            self.0.take_fault()
        }
        fn reset(&mut self) {
            self.0.reset();
        }
    }
    let ledger = usdt_ledger();
    let plan = plan_on(&ledger, RollFailurePolicy::Abort);
    let mut engine = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 8),
        ledger,
        Repeats(RollStrategy::new(plan, 100), None),
    )
    .with_contract(near(), data(full_near_ticks()), ImmediateFillModel::new())
    .unwrap()
    .with_contract(far(), data(full_far_ticks()), ImmediateFillModel::new())
    .unwrap();
    let result = engine.run();
    assert!(result.fault_count > 0, "{result:?}");
    assert!(!result.is_consistent());
    assert!(
        engine
            .faults()
            .iter()
            .any(|f| f.to_string().contains("the plan refused an event")),
        "{:?}",
        engine.faults()
    );
}

/// On an inverse contract the size is its value, and cash is in BTC.
#[test]
fn an_inverse_roll_agrees() {
    let near = terms_of(
        deribit::btc_future(
            market_maker_rs::instruments::InstrumentId::new("BTC-A").unwrap(),
            NEAR_MS,
        )
        .unwrap(),
    );
    let far = terms_of(
        deribit::btc_future(
            market_maker_rs::instruments::InstrumentId::new("BTC-B").unwrap(),
            FAR_MS,
        )
        .unwrap(),
    );
    let ledger = opened(&near, &far, ["BTC", "USD"], dec!(60000), dec!(100));
    let request = RollRequest {
        near: near.clone(),
        far: far.clone(),
        near_position: dec!(100),
        far_position: dec!(0),
        near_price: dec!(61000),
        far_price: dec!(62000),
        policy: RollFailurePolicy::Abort,
        deadline_ms: DEADLINE_MS,
        created_at_ms: CREATED_MS,
    };
    let plan = RollPlan::new(request).unwrap().anchored_to(&ledger);
    let mut engine = ContractEngine::new(
        ContractConfig::new(dec!(1), dec!(0), 8),
        ledger.clone(),
        RollStrategy::new(plan.clone(), 100),
    )
    .with_contract(
        near.clone(),
        data(full_near_ticks()),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .with_contract(
        far.clone(),
        data(full_far_ticks()),
        ImmediateFillModel::new(),
    )
    .unwrap();
    let result = engine.run();
    let steps = vec![
        fill_of(
            &near,
            RollLeg::Near,
            Some(0),
            "sim-1",
            Action::Sell,
            dec!(61000),
            dec!(100),
            dec!(0),
            20,
        ),
        fill_of(
            &far,
            RollLeg::Far,
            Some(0),
            "sim-2",
            Action::Buy,
            dec!(62000),
            dec!(100),
            dec!(0),
            30,
        ),
    ];
    let (direct_ledger, direct_plan) = booked_directly(ledger, plan, steps);
    agree(
        engine.ledger(),
        engine.strategy().plan(),
        &direct_ledger,
        &direct_plan,
    );
    assert_eq!(engine.strategy().plan().state(), RollState::Completed);
    assert!(result.is_consistent(), "{result:?}");
}
