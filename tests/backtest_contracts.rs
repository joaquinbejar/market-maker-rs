//! Several dated futures in one run. Amounts are worked out by hand from the quotes.
//! Linear contracts of 1 BTC, tick 1 and lot 1, unless a test says otherwise, so a fee
//! of 0.001 on 2 contracts at 105 is 0.21.

#![cfg(all(feature = "margin", feature = "instruments"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use financial_types::instrument::InstrumentId as SpecId;
use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::{DerivativeSettlement, Execution, FundingPayment, Ledger, LedgerConfig};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::contracts::{
    ContractCommand, ContractConfig, ContractEngine, ContractFill, ContractOrder, ContractReport,
    ContractRunResult, ContractStrategy, ContractView, MAX_COMMANDS_PER_INSTANT,
    MAX_STRATEGY_FAULTS_PER_CALLBACK,
};
use market_maker_rs::backtest::engine::{
    BacktestConfig, BacktestStrategy, FeeCurrency, SlippageModel,
};
use market_maker_rs::backtest::{
    BacktestEngine, ExecutionKind, FillModel, FillResult, ImmediateFillModel,
    MarginLedgerAccountant, MarketTick, QueuePositionFillModel, SimLatencyConfig, SimulatedFill,
    SimulatedOrder, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::instruments::futures::{
    FutureTerms, SettlementMethod, SettlementPrice, SettlementSource,
};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentSpec, Margining, TickSchedule,
};
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::{
    LimitsPolicy, OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};

const NEAR_MS: u64 = 1000;
const FAR_MS: u64 = 2000;

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

fn near_id() -> SpecId {
    SpecId::new("NEAR").unwrap()
}

fn far_id() -> SpecId {
    SpecId::new("FAR").unwrap()
}

fn ledger_of(specs: &[InstrumentSpec]) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for asset in ["BTC", "USDT"] {
        ledger
            .register_asset(Asset::new(asset).unwrap(), 8)
            .unwrap();
    }
    for spec in specs {
        ledger
            .register_instrument(
                Instrument::from_spec(
                    spec,
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
}

fn both_ledger() -> Ledger {
    ledger_of(&[near().spec().clone(), far().spec().clone()])
}

fn seed(ledger: &mut Ledger, id: &str, name: &str, side: Action, price: Decimal, qty: Decimal) {
    seed_with_fee(ledger, id, name, side, price, qty, dec!(0));
}

fn seed_with_fee(
    ledger: &mut Ledger,
    id: &str,
    name: &str,
    side: Action,
    price: Decimal,
    qty: Decimal,
    fee: Decimal,
) {
    let mut execution = Execution::new(
        ExecutionId::new(id).unwrap(),
        InstrumentId::new(name).unwrap(),
        side,
        price,
        qty,
    )
    .with_executed_at(1);
    if !fee.is_zero() {
        execution = execution.with_fee(Asset::new("USDT").unwrap(), fee);
    }
    ledger
        .book_execution_allowing_overdraft(&execution, None)
        .unwrap();
}

fn tick(timestamp: u64, bid: Decimal, ask: Decimal) -> MarketTick {
    MarketTick::new(timestamp, bid, dec!(100000), ask, dec!(100000))
}

fn data(ticks: Vec<MarketTick>) -> VecDataSource {
    VecDataSource::new(ticks)
}

fn config(fee_rate: Decimal) -> ContractConfig {
    ContractConfig::new(dec!(1000), fee_rate, 8)
}

fn order(
    tag: u64,
    contract: SpecId,
    side: Action,
    price: Decimal,
    qty: Decimal,
) -> ContractCommand {
    ContractCommand::Submit(ContractOrder {
        tag,
        contract,
        side,
        price,
        quantity: qty,
        purpose: OrderPurpose::Other,
    })
}

/// Commands from a script, one entry per tick seen; also answers given fills, and logs
/// what reached it, in order.
#[derive(Debug, Clone, Default)]
struct Script {
    on_ticks: Vec<Vec<ContractCommand>>,
    seen: usize,
    on_fills: Vec<ContractCommand>,
    log: Vec<String>,
}

impl Script {
    fn of(on_ticks: Vec<Vec<ContractCommand>>) -> Self {
        Self {
            on_ticks,
            ..Self::default()
        }
    }
}

impl ContractStrategy for Script {
    fn on_tick(
        &mut self,
        contract: &SpecId,
        tick: &MarketTick,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.log.push(format!(
            "tick {} {} near {} far {}",
            contract.as_str(),
            tick.timestamp,
            view.position(&near_id()),
            view.position(&far_id()),
        ));
        let step = self.on_ticks.get(self.seen).cloned().unwrap_or_default();
        self.seen += 1;
        step
    }

    fn on_report(&mut self, report: &ContractReport, _view: &ContractView) -> Vec<ContractCommand> {
        self.log.push(format!(
            "{:?} {} tag {}",
            report.report.kind,
            report.contract.as_str(),
            report.tag
        ));
        Vec::new()
    }

    fn on_fill(&mut self, fill: &ContractFill, _view: &ContractView) -> Vec<ContractCommand> {
        self.log.push(format!(
            "fill {} {} {}",
            fill.contract.as_str(),
            fill.execution_id,
            fill.quantity
        ));
        std::mem::take(&mut self.on_fills)
    }

    fn on_settlement(
        &mut self,
        contract: &SpecId,
        price: &SettlementPrice,
        contracts: Decimal,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.log.push(format!(
            "settled {} at {} closing {contracts}",
            contract.as_str(),
            price.price()
        ));
        Vec::new()
    }

    fn reset(&mut self) {
        self.seen = 0;
        self.log.clear();
    }
}

type Engine<F = ImmediateFillModel> = ContractEngine<Script, VecDataSource, F>;

fn two_contracts(
    fee_rate: Decimal,
    ledger: Ledger,
    script: Script,
    near_ticks: Vec<MarketTick>,
    far_ticks: Vec<MarketTick>,
) -> Engine {
    ContractEngine::new(config(fee_rate), ledger, script)
        .with_contract(near(), data(near_ticks), ImmediateFillModel::new())
        .unwrap()
        .with_contract(far(), data(far_ticks), ImmediateFillModel::new())
        .unwrap()
}

fn line<'a>(
    result: &'a ContractRunResult,
    contract: &SpecId,
) -> &'a market_maker_rs::backtest::contracts::ContractSummary {
    result
        .contracts
        .iter()
        .find(|c| &c.contract == contract)
        .unwrap()
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

#[test]
fn two_contracts_book_in_one_ledger_and_value_at_their_own_marks() {
    let script = Script::of(vec![vec![
        order(1, near_id(), Action::Buy, dec!(100), dec!(2)),
        order(2, far_id(), Action::Sell, dec!(105), dec!(2)),
    ]]);
    let mut engine = two_contracts(
        dec!(0.001),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(30, dec!(98), dec!(100))],
        vec![
            tick(20, dec!(105), dec!(107)),
            tick(40, dec!(103), dec!(105)),
        ],
    );
    let result = engine.run();
    assert_eq!(result.num_ticks, 4);
    assert_eq!(result.num_trades, 2);
    assert_eq!((result.start_time, result.end_time), (10, 40));
    // Flat; short 2 at 105 marked 106, less 0.21; long 2 at 100 marked 99, less 0.20;
    // the short now marked 104.
    assert_eq!(
        result.equity_curve,
        vec![
            (10, dec!(1000)),
            (20, dec!(997.79)),
            (30, dec!(995.59)),
            (40, dec!(999.59)),
        ]
    );
    assert_eq!(result.total_pnl, dec!(0));
    assert_eq!(result.unrealized_pnl, dec!(0));
    assert_eq!(result.total_fees, dec!(0.41));
    assert_eq!(result.net_pnl, dec!(-0.41));
    assert_eq!(result.max_drawdown, dec!(4.41));
    let near_line = line(&result, &near_id());
    assert_eq!(near_line.position, dec!(2));
    assert_eq!(near_line.max_position, dec!(2));
    assert_eq!(near_line.fills, 1);
    assert_eq!(near_line.last_mark, Some(dec!(99)));
    assert_eq!(near_line.last_mark_ms, Some(30));
    let far_line = line(&result, &far_id());
    assert_eq!(far_line.position, dec!(-2));
    assert_eq!(far_line.fills, 1);
    assert_eq!(far_line.last_mark, Some(dec!(104)));
    assert_eq!(far_line.last_mark_ms, Some(40));
    assert_eq!(result.trades[0].contract, far_id());
    assert_eq!(result.trades[0].execution_id, "sim-1");
    assert_eq!(result.trades[0].tag, 2);
    assert_eq!(result.trades[1].contract, near_id());
    assert_eq!(result.trades[1].execution_id, "sim-2");
    assert!(result.is_consistent(), "{result:?}");
    // The strategy knows what was delivered, and no other contract's position.
    assert!(
        engine
            .strategy()
            .log
            .contains(&"tick NEAR 30 near 2 far -2".to_string())
    );
    assert_eq!(engine.ledger().journal().len(), 2);
}

/// Equal times go in the order of registration, and an order sent on the first tick
/// waits for a later one on the second.
#[test]
fn a_tie_in_time_goes_by_registration_and_cannot_fill_on_its_own_time() {
    let script = Script::of(vec![
        vec![order(1, far_id(), Action::Sell, dec!(105), dec!(1))],
        vec![],
        vec![],
    ]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102))],
        vec![
            tick(10, dec!(105), dec!(107)),
            tick(11, dec!(105), dec!(107)),
        ],
    );
    let result = engine.run();
    let log = &engine.strategy().log;
    assert!(log[0].starts_with("tick NEAR 10"), "{log:?}");
    assert!(log.iter().any(|l| l.starts_with("tick FAR 10")), "{log:?}");
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].timestamp, 11);
}

/// An out of order tick does not take the clock back, in either contract.
#[test]
fn an_out_of_order_tick_does_not_take_the_clock_back() {
    let script = Script::of(vec![vec![order(
        1,
        far_id(),
        Action::Sell,
        dec!(105),
        dec!(1),
    )]]);
    // A tick of 5 after one of 20 on the far contract, whose own time is before the
    // order became active, so it cannot fill it.
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102))],
        vec![
            tick(20, dec!(90), dec!(92)),
            tick(5, dec!(105), dec!(107)),
            tick(30, dec!(105), dec!(107)),
        ],
    );
    let result = engine.run();
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].timestamp, 30);
    let times: Vec<u64> = result.equity_curve.iter().map(|(t, _)| *t).collect();
    assert_eq!(times, vec![10, 20, 5, 30]);
}

#[test]
fn each_contract_has_its_own_fill_model() {
    // Half of the far book is taken on each tick; the near order fills in full.
    let script = Script::of(vec![vec![
        order(1, near_id(), Action::Buy, dec!(100), dec!(4)),
        order(2, far_id(), Action::Sell, dec!(105), dec!(4)),
    ]]);
    #[derive(Debug, Clone)]
    enum Either {
        Immediate(ImmediateFillModel),
        Queue(QueuePositionFillModel),
    }
    impl FillModel for Either {
        fn simulate_fill(
            &self,
            order: &SimulatedOrder,
            tick: &MarketTick,
            time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            match self {
                Self::Immediate(model) => model.simulate_fill(order, tick, time_in_queue_ms),
                Self::Queue(model) => model.simulate_fill(order, tick, time_in_queue_ms),
            }
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "either"
        }
    }
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(100)),
                tick(20, dec!(98), dec!(100)),
            ]),
            Either::Immediate(ImmediateFillModel::new()),
        )
        .unwrap()
        .with_contract(
            far(),
            data(vec![
                tick(11, dec!(105), dec!(107)),
                tick(21, dec!(105), dec!(107)),
            ]),
            Either::Queue(QueuePositionFillModel::new(dec!(1))),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(line(&result, &near_id()).position, dec!(4));
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn nothing_executes_at_an_expiry_and_ticks_after_it_do_not_move_the_mark() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![
            tick(10, dec!(102), dec!(104)),
            // At the expiry, and after it, each of which would fill the order.
            tick(NEAR_MS, dec!(90), dec!(92)),
            tick(NEAR_MS + 1, dec!(90), dec!(92)),
        ],
        vec![],
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    let near_line = line(&result, &near_id());
    assert!(near_line.expired);
    assert!(!near_line.settled);
    assert_eq!(near_line.ignored_ticks, 2);
    assert_eq!(near_line.last_mark, Some(dec!(103)));
    assert_eq!(near_line.last_mark_ms, Some(10));
    // The venue cancelled the live order.
    let kinds: Vec<_> = engine
        .execution_reports()
        .iter()
        .map(|r| r.report.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![ExecutionKind::Accepted, ExecutionKind::Canceled]
    );
    assert_eq!(engine.fault_count(), 0);
}

/// A tick of the far contract alone moves the clock past the near expiry.
#[test]
fn a_tick_of_another_contract_expires_a_contract_that_stopped_ticking() {
    let script = Script::of(vec![
        vec![order(1, near_id(), Action::Buy, dec!(90), dec!(1))],
        vec![order(2, near_id(), Action::Buy, dec!(90), dec!(1))],
    ]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102))],
        vec![tick(NEAR_MS, dec!(105), dec!(107))],
    );
    let result = engine.run();
    assert!(line(&result, &near_id()).expired);
    assert!(!line(&result, &far_id()).expired);
    let kinds: Vec<_> = engine
        .execution_reports()
        .iter()
        .map(|r| (r.tag, r.report.kind))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (1, ExecutionKind::Accepted),
            (1, ExecutionKind::Canceled),
            // Sent after the expiry, it is rejected locally.
            (2, ExecutionKind::Rejected),
        ]
    );
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn an_order_on_its_way_when_a_contract_expires_is_cancelled_by_the_venue() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![
            tick(NEAR_MS - 1, dec!(98), dec!(102)),
            tick(NEAR_MS + 50, dec!(90), dec!(92)),
        ],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_submit_ms(10))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    let kinds: Vec<_> = engine
        .execution_reports()
        .iter()
        .map(|r| r.report.kind)
        .collect();
    assert_eq!(kinds, vec![ExecutionKind::Canceled]);
}

#[test]
fn a_declared_settlement_books_when_published_even_after_the_last_tick() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(2),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![],
    )
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS + 100))
    .unwrap();
    let result = engine.run();
    // Long 2 from 100 marked 101, then settled at 103.
    assert_eq!(
        result.equity_curve,
        vec![(10, dec!(1002)), (NEAR_MS + 100, dec!(1006))]
    );
    assert_eq!(result.total_pnl, dec!(6));
    assert_eq!(result.unrealized_pnl, dec!(0));
    assert_eq!(result.end_time, 10);
    let near_line = line(&result, &near_id());
    assert_eq!(near_line.position, dec!(0));
    assert!(near_line.expired);
    assert!(near_line.settled);
    assert!(!near_line.expired_unsettled);
    assert_eq!(result.stale_from_ms, None);
    assert!(result.is_consistent(), "{result:?}");
    assert_eq!(
        engine.strategy().log.last().unwrap(),
        "settled NEAR at 103 closing 2"
    );
    assert_eq!(engine.ledger().expiry_journal().len(), 1);
}

/// A fill matched before the expiry and delivered after the publication is in the
/// ledger before the settlement, which closes it, and the strategy hears of it first.
#[test]
fn a_fill_on_its_way_at_the_settlement_booked_when_it_matched() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![
            tick(990, dec!(98), dec!(102)),
            tick(995, dec!(98), dec!(100)),
        ],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(500))
    .unwrap()
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS + 10))
    .unwrap();
    let result = engine.run();
    assert_eq!(engine.ledger().journal().len(), 1);
    assert_eq!(result.total_pnl, dec!(3));
    assert_eq!(line(&result, &near_id()).position, dec!(0));
    let log = &engine.strategy().log;
    let fill = log.iter().position(|l| l.starts_with("fill NEAR")).unwrap();
    let settled = log
        .iter()
        .position(|l| l == "settled NEAR at 103 closing 1")
        .unwrap();
    assert!(fill < settled, "{log:?}");
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn past_its_expiry_with_no_settlement_a_run_is_not_consistent() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![tick(NEAR_MS + 5, dec!(105), dec!(107))],
    );
    let result = engine.run();
    let near_line = line(&result, &near_id());
    assert!(near_line.expired_unsettled);
    assert_eq!(near_line.position, dec!(1));
    assert_eq!(result.stale_from_ms, Some(NEAR_MS + 5));
    assert!(!result.is_consistent());
    // Its PnL stays apart from cash, at the last mark.
    assert_eq!(result.unrealized_pnl, dec!(1));
    assert!(engine.ledger().expiry_journal().is_empty());
    assert_eq!(engine.fault_count(), 0);

    // Flat, an expiry with no settlement leaves nothing resting on a mark.
    let mut flat = two_contracts(
        dec!(0),
        both_ledger(),
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![tick(NEAR_MS + 5, dec!(105), dec!(107))],
    );
    let result = flat.run();
    assert!(line(&result, &near_id()).expired);
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn data_that_ends_before_the_expiry_leaves_a_contract_simply_open() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![],
    );
    let result = engine.run();
    let near_line = line(&result, &near_id());
    assert!(!near_line.expired);
    assert!(!near_line.expired_unsettled);
    assert_eq!(near_line.position, dec!(1));
    assert_eq!(result.stale_from_ms, None);
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_position_with_no_mark_yet_is_no_fault_but_must_get_one() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    // Two far ticks before the near contract's first.
    let mut engine = two_contracts(
        dec!(0),
        ledger.clone(),
        Script::default(),
        vec![tick(30, dec!(100), dec!(102))],
        vec![
            tick(10, dec!(105), dec!(107)),
            tick(20, dec!(105), dec!(107)),
        ],
    );
    let result = engine.run();
    assert_eq!(result.unvalued_ticks, 2);
    assert_eq!(engine.fault_count(), 0);
    assert_eq!(result.equity_curve, vec![(30, dec!(1001))]);
    assert_eq!(result.max_drawdown, dec!(0));
    assert!(result.never_valued.is_empty());
    assert!(result.is_consistent(), "{result:?}");

    // It never ticks, so the run never could value it.
    let mut never = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![],
        vec![tick(10, dec!(105), dec!(107))],
    );
    let result = never.run();
    assert_eq!(result.unvalued_ticks, 1);
    assert_eq!(result.never_valued, vec![near_id()]);
    assert!(result.equity_curve.is_empty());
    assert!(!result.is_consistent());
    assert_eq!(never.fault_count(), 0);
}

#[test]
fn a_mark_the_ledger_cannot_value_is_a_fault_and_keeps_the_figures() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let script = Script::of(vec![
        vec![],
        vec![],
        vec![order(1, near_id(), Action::Buy, dec!(100), dec!(1))],
    ]);
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![
            tick(10, dec!(100), dec!(102)),
            tick(20, dec!(0), dec!(0)),
            tick(30, dec!(100), dec!(100)),
        ],
        vec![],
    );
    let result = engine.run();
    assert_eq!(result.accounting.valuation_failures, 1);
    assert_eq!(engine.fault_count(), 1);
    assert_eq!(result.equity_curve[1], (20, dec!(1001)));
    assert!(!result.is_consistent());
    // A fault stops new orders.
    assert!(
        engine
            .execution_reports()
            .iter()
            .any(|r| r.report.kind == ExecutionKind::Rejected)
    );
}

#[test]
fn what_the_ledger_already_held_is_not_counted_again() {
    let mut ledger = both_ledger();
    // Opened at 100 with a fee of 7, then partly closed at 110 for 10 realized, all
    // before the run.
    seed_with_fee(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(2),
        dec!(7),
    );
    seed(
        &mut ledger,
        "seed-2",
        "NEAR",
        Action::Sell,
        dec!(110),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(104))],
        vec![],
    );
    let result = engine.run();
    // Only the open contract's unrealized: 1 * (102 - 100).
    assert_eq!(result.total_pnl, dec!(2));
    assert_eq!(result.total_fees, dec!(0));
    assert_eq!(result.equity_curve, vec![(10, dec!(1002))]);
}

#[test]
fn settlements_and_funding_the_ledger_already_held_are_not_counted_again() {
    let asset = |code| AssetCode::new(code).unwrap();
    let perpetual = InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new("PERP").unwrap(),
        kind: ContractKind::Perpetual,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: None,
        option: None,
    })
    .unwrap();
    let mut funded = ledger_of(&[near().spec().clone(), perpetual]);
    funded
        .book_funding(&FundingPayment {
            id: ExecutionId::new("funded").unwrap(),
            instrument: InstrumentId::new("PERP").unwrap(),
            amount: dec!(5),
            as_of_ms: 1,
        })
        .unwrap();
    let mut settled = ledger_of(&[near().spec().clone()]);
    settled
        .book_settlement(&DerivativeSettlement {
            id: ExecutionId::new("settled").unwrap(),
            // A settlement is for a derivative held outside the ledger.
            instrument: InstrumentId::new("OUTSIDE").unwrap(),
            asset: Asset::new("USDT").unwrap(),
            realized_pnl: dec!(5),
            fee: dec!(0),
        })
        .unwrap();
    for ledger in [funded, settled] {
        let mut engine = ContractEngine::new(config(dec!(0)), ledger, Script::default())
            .with_contract(
                near(),
                data(vec![tick(10, dec!(98), dec!(102))]),
                ImmediateFillModel::new(),
            )
            .unwrap();
        let result = engine.run();
        assert_eq!(result.total_pnl, dec!(0));
        assert_eq!(result.equity_curve, vec![(10, dec!(1000))]);
        assert!(result.is_consistent(), "{result:?}");
    }
}

/// With no fee, the journal holds none, rather than a fee of zero.
#[test]
fn a_fill_with_no_fee_books_none() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    );
    engine.run();
    assert_eq!(engine.ledger().journal()[0].fee, None);
}

#[test]
fn a_cancel_that_crosses_a_fill_is_answered_and_is_no_fault() {
    let mut script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    script.on_fills = vec![ContractCommand::Cancel { tag: 1 }];
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(engine.fault_count(), 0);
    let kinds: Vec<_> = engine
        .execution_reports()
        .iter()
        .map(|r| r.report.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![
            ExecutionKind::Accepted,
            ExecutionKind::Fill,
            ExecutionKind::CancelRejected
        ]
    );
}

/// Wakes at the times it was given, and logs what it saw.
#[derive(Debug, Default)]
struct Wakes {
    at: Vec<u64>,
    seen: Vec<String>,
    orders: Vec<ContractCommand>,
}

impl ContractStrategy for Wakes {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        tick: &MarketTick,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.seen
            .push(format!("tick {} at {}", tick.timestamp, view.now_ms()));
        Vec::new()
    }
    fn on_report(&mut self, report: &ContractReport, view: &ContractView) -> Vec<ContractCommand> {
        self.seen.push(format!(
            "{:?} of {} at {}",
            report.report.kind,
            report.report.stamp.timestamp,
            view.now_ms()
        ));
        Vec::new()
    }
    fn on_fill(&mut self, fill: &ContractFill, view: &ContractView) -> Vec<ContractCommand> {
        self.seen
            .push(format!("fill of {} at {}", fill.timestamp, view.now_ms()));
        Vec::new()
    }
    fn on_settlement(
        &mut self,
        _contract: &SpecId,
        price: &SettlementPrice,
        _contracts: Decimal,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.seen.push(format!(
            "settled of {} at {}",
            price.published_ms(),
            view.now_ms()
        ));
        Vec::new()
    }
    fn next_wake_ms(&self) -> Option<u64> {
        self.at.first().copied()
    }
    fn on_wake(&mut self, view: &ContractView) -> Vec<ContractCommand> {
        self.seen.push(format!("wake at {}", view.now_ms()));
        // Each entry is asked for until it is its turn, and a repeat is dropped.
        while self.at.first().is_some_and(|at| *at <= view.now_ms()) {
            self.at.remove(0);
        }
        std::mem::take(&mut self.orders)
    }
    fn reset(&mut self) {}
}

#[test]
fn a_strategy_wakes_with_no_tick_and_each_callback_knows_the_time() {
    let strategy = Wakes {
        at: vec![15, 25],
        orders: vec![order(1, near_id(), Action::Buy, dec!(100), dec!(1))],
        ..Wakes::default()
    };
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), strategy)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(30, dec!(98), dec!(100)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap()
        .with_latency(SimLatencyConfig::new().with_ack_ms(100))
        .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(
        engine.strategy().seen,
        vec![
            "tick 10 at 10",
            "wake at 15",
            "wake at 25",
            "tick 30 at 30",
            "Accepted of 15 at 115",
            "Fill of 30 at 130",
            "fill of 30 at 130",
        ]
    );
    assert!(result.is_consistent(), "{result:?}");
}

/// A report delivered at the time asked for does not take the wake's place, and it comes
/// first.
#[test]
fn a_wake_at_the_time_of_a_delivery_runs_after_it() {
    let strategy = Wakes {
        at: vec![5, 110],
        orders: vec![order(1, near_id(), Action::Buy, dec!(90), dec!(1))],
        ..Wakes::default()
    };
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), strategy)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(200, dec!(98), dec!(102)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap()
        .with_latency(SimLatencyConfig::new().with_ack_ms(105))
        .unwrap();
    engine.run();
    assert_eq!(
        engine.strategy().seen,
        vec![
            "wake at 5",
            "tick 10 at 10",
            "Accepted of 5 at 110",
            "wake at 110",
            "tick 200 at 200",
        ]
    );
}

/// Keeps asking for one time, and then for one in the past.
#[derive(Debug, Default)]
struct KeepsAsking(u32);

impl ContractStrategy for KeepsAsking {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        _tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        Vec::new()
    }
    fn next_wake_ms(&self) -> Option<u64> {
        Some(if self.0 == 0 { 5 } else { 1 })
    }
    fn on_wake(&mut self, _view: &ContractView) -> Vec<ContractCommand> {
        self.0 += 1;
        assert!(self.0 < 3, "woken again and again");
        Vec::new()
    }
    fn reset(&mut self) {
        self.0 = 0;
    }
}

/// Keeps asking for the time it was woken at.
#[derive(Debug, Default)]
struct KeepsAskingSame(u32);

impl ContractStrategy for KeepsAskingSame {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        _tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        Vec::new()
    }
    fn next_wake_ms(&self) -> Option<u64> {
        Some(5)
    }
    fn on_wake(&mut self, _view: &ContractView) -> Vec<ContractCommand> {
        self.0 += 1;
        assert!(self.0 < 3, "woken again and again");
        Vec::new()
    }
    fn reset(&mut self) {
        self.0 = 0;
    }
}

#[test]
fn a_time_already_woken_at_or_past_does_not_wake_again_until_a_reset() {
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), KeepsAsking::default())
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    engine.run();
    assert_eq!(engine.strategy().0, 1);

    let mut engine =
        ContractEngine::new(config(dec!(0)), both_ledger(), KeepsAskingSame::default())
            .with_contract(
                near(),
                data(vec![tick(10, dec!(98), dec!(102))]),
                ImmediateFillModel::new(),
            )
            .unwrap();
    engine.run();
    assert_eq!(engine.strategy().0, 1);
    engine.reset();
    engine.run();
    assert_eq!(engine.strategy().0, 1);
}

/// A time already past is ignored rather than repeated without end, and one past the data
/// never comes unless a settlement takes the clock there.
#[test]
fn a_wake_not_after_the_clock_is_ignored_and_the_data_ending_stops_the_clock() {
    // Asking for a time it already woke at, or one in the past, does nothing.
    let strategy = Wakes {
        at: vec![5, 5, 1],
        ..Wakes::default()
    };
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), strategy)
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    engine.run();
    assert_eq!(engine.strategy().seen, vec!["wake at 5", "tick 10 at 10"]);

    let strategy = Wakes {
        at: vec![500],
        ..Wakes::default()
    };
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), strategy)
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    engine.run();
    assert_eq!(engine.strategy().seen, vec!["tick 10 at 10"]);

    let strategy = Wakes {
        at: vec![500],
        ..Wakes::default()
    };
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), strategy)
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap()
        .with_settlement(settlement(&near(), dec!(100), NEAR_MS))
        .unwrap();
    engine.run();
    assert_eq!(
        engine.strategy().seen,
        vec!["tick 10 at 10", "wake at 500", "settled of 1000 at 1000"]
    );
}

/// A strategy with more faults than one callback takes.
#[derive(Debug, Default)]
struct Faulty(u32);

impl ContractStrategy for Faulty {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        _tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        vec![order(1, near_id(), Action::Buy, dec!(90), dec!(1))]
    }
    fn take_fault(&mut self) -> Option<MMError> {
        self.0 += 1;
        Some(MMError::InvalidMarketState("always".to_string()))
    }
    fn reset(&mut self) {}
}

#[test]
fn what_a_strategy_reports_is_a_fault_that_stops_its_own_commands_and_is_bounded() {
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), Faulty::default())
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert!(!result.is_consistent());
    assert_eq!(
        engine.strategy().0,
        MAX_STRATEGY_FAULTS_PER_CALLBACK as u32 + MAX_STRATEGY_FAULTS_PER_CALLBACK as u32
    );
    assert_eq!(
        engine.execution_reports()[0].report.kind,
        ExecutionKind::Rejected
    );
}

#[test]
fn a_cancel_works_with_latency_and_a_late_one_is_rejected() {
    // Cancelled before the next tick, so the tick does not fill it.
    let script = Script::of(vec![
        vec![order(1, near_id(), Action::Buy, dec!(100), dec!(1))],
        vec![ContractCommand::Cancel { tag: 1 }],
    ]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![
            tick(10, dec!(98), dec!(102)),
            tick(20, dec!(98), dec!(102)),
            tick(30, dec!(98), dec!(100)),
        ],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_cancel_ms(5))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    let kinds: Vec<_> = engine
        .execution_reports()
        .iter()
        .map(|r| r.report.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![ExecutionKind::Accepted, ExecutionKind::Canceled]
    );

    // The cancel is on its way when the order fills.
    let script = Script::of(vec![
        vec![order(1, near_id(), Action::Buy, dec!(100), dec!(1))],
        vec![ContractCommand::Cancel { tag: 1 }],
    ]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![
            tick(10, dec!(98), dec!(102)),
            tick(20, dec!(98), dec!(102)),
            tick(21, dec!(98), dec!(100)),
            tick(40, dec!(98), dec!(102)),
        ],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_cancel_ms(5))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(engine.fault_count(), 0);
    assert_eq!(
        engine.execution_reports().last().unwrap().report.kind,
        ExecutionKind::CancelRejected
    );
}

#[test]
fn a_command_that_makes_no_sense_is_a_fault_and_stops_new_orders() {
    let cases: Vec<(&str, Vec<ContractCommand>)> = vec![
        ("never sent", vec![ContractCommand::Cancel { tag: 9 }]),
        (
            "not in this run",
            vec![order(
                1,
                SpecId::new("OTHER").unwrap(),
                Action::Buy,
                dec!(100),
                dec!(1),
            )],
        ),
        (
            "already used",
            vec![
                order(1, near_id(), Action::Buy, dec!(90), dec!(1)),
                order(1, near_id(), Action::Buy, dec!(90), dec!(1)),
            ],
        ),
        (
            "not in this run",
            vec![ContractCommand::CancelAll {
                contract: SpecId::new("OTHER").unwrap(),
            }],
        ),
    ];
    for (reason, commands) in cases {
        let script = Script::of(vec![
            commands,
            vec![order(5, near_id(), Action::Buy, dec!(90), dec!(1))],
        ]);
        let mut engine = two_contracts(
            dec!(0),
            both_ledger(),
            script,
            vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(102))],
            vec![],
        );
        let result = engine.run();
        assert_eq!(engine.fault_count(), 1, "{reason}");
        // A run stopped by a fault does not read clean.
        assert_eq!(result.fault_count, 1, "{reason}");
        assert!(!result.is_consistent(), "{reason}");
        assert!(
            engine.faults()[0].to_string().contains(reason),
            "{:?}",
            engine.faults()
        );
        let last = engine.execution_reports().last().unwrap();
        assert_eq!((last.tag, last.report.kind), (5, ExecutionKind::Rejected));
    }
}

#[test]
fn an_order_off_the_grid_or_of_another_action_is_refused_and_not_rounded() {
    let cases = [
        (Action::Buy, dec!(100.5), dec!(1)),
        (Action::Buy, dec!(100), dec!(1.5)),
        (Action::Buy, dec!(100), dec!(0)),
        (Action::Other, dec!(100), dec!(1)),
    ];
    for (side, price, quantity) in cases {
        let script = Script::of(vec![vec![order(1, near_id(), side, price, quantity)]]);
        let mut engine = two_contracts(
            dec!(0),
            both_ledger(),
            script,
            vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(90), dec!(92))],
            vec![],
        );
        let result = engine.run();
        assert_eq!(result.num_trades, 0);
        assert_eq!(engine.fault_count(), 0);
        let kinds: Vec<_> = engine
            .execution_reports()
            .iter()
            .map(|r| r.report.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![ExecutionKind::Rejected],
            "{side:?} {price} {quantity}"
        );
    }
}

/// A policy that records what it was asked, and takes one decision.
#[derive(Debug, Clone)]
struct Recorded {
    decision: PretradeDecision,
    requests: std::sync::Arc<std::sync::Mutex<Vec<PretradeRequest>>>,
    events: std::sync::Arc<std::sync::Mutex<Vec<OrderEvent>>>,
}

impl Recorded {
    fn of(decision: PretradeDecision) -> Self {
        Self {
            decision,
            requests: std::sync::Arc::default(),
            events: std::sync::Arc::default(),
        }
    }
}

impl PretradePolicy for Recorded {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        self.requests.lock().unwrap().push(request.clone());
        self.decision.clone()
    }
    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        self.events.lock().unwrap().push(event.clone());
        Ok(())
    }
    fn reset(&mut self) {}
    fn in_contracts(&self) -> bool {
        true
    }
}

#[test]
fn a_policy_decides_in_contracts_on_requests_that_name_the_contract() {
    let run = |decision: PretradeDecision| {
        let policy = Recorded::of(decision);
        let script = Script::of(vec![vec![order(
            1,
            far_id(),
            Action::Sell,
            dec!(105),
            dec!(5),
        )]]);
        let mut engine = two_contracts(
            dec!(0),
            both_ledger(),
            script,
            vec![tick(10, dec!(98), dec!(102))],
            vec![tick(20, dec!(105), dec!(107))],
        )
        .with_pretrade_policy(policy.clone())
        .unwrap();
        let result = engine.run();
        let decided = engine.pretrade_decisions().to_vec();
        let reports = engine.execution_reports().to_vec();
        (policy, result, decided, reports)
    };

    let (policy, result, decided, _) = run(PretradeDecision::Accepted);
    let requests = policy.requests.lock().unwrap();
    assert_eq!(requests[0].instrument.as_deref(), Some("FAR"));
    assert_eq!(requests[0].quantity, dec!(5));
    assert_eq!(requests[0].price, Some(dec!(105)));
    assert_eq!(requests[0].side, Action::Sell);
    assert_eq!(result.trades[0].quantity, dec!(5));
    assert_eq!(decided[0].submitted, dec!(5));
    // It sees the fill before the ledger, naming the same execution.
    assert!(policy.events.lock().unwrap().iter().any(|event| matches!(
        event,
        OrderEvent::Fill { execution_id, fee_asset, .. }
            if execution_id == "sim-1"
                && *fee_asset == market_maker_rs::backtest::FeeAsset::Settlement
    )));

    // A reduction flows into the order, and only that executes.
    let (_, result, decided, _) = run(PretradeDecision::Reduced {
        quantity: dec!(2),
        reason: "room".to_string(),
    });
    assert_eq!(result.trades[0].quantity, dec!(2));
    assert_eq!(decided[0].submitted, dec!(2));

    // A rejection is local, and nothing reaches the venue.
    let (_, result, decided, reports) = run(PretradeDecision::Rejected {
        reason: "no".to_string(),
    });
    assert_eq!(result.num_trades, 0);
    assert_eq!(decided[0].submitted, dec!(0));
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].report.kind, ExecutionKind::Rejected);

    // What is refused before it never reaches the policy.
    let policy = Recorded::of(PretradeDecision::Accepted);
    let script = Script::of(vec![vec![
        order(1, far_id(), Action::Other, dec!(105), dec!(1)),
        order(2, far_id(), Action::Sell, dec!(105.5), dec!(1)),
        order(3, far_id(), Action::Sell, dec!(105), dec!(1.5)),
    ]]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102))],
        vec![],
    )
    .with_pretrade_policy(policy.clone())
    .unwrap();
    engine.run();
    assert!(policy.requests.lock().unwrap().is_empty());
    assert_eq!(engine.execution_reports().len(), 3);

    // A reduction off the lot is refused and released in the policy.
    let (policy, result, _, _) = run(PretradeDecision::Reduced {
        quantity: dec!(1.5),
        reason: "room".to_string(),
    });
    assert_eq!(result.num_trades, 0);
    assert!(
        policy
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, OrderEvent::VenueRejected { .. }))
    );
}

#[test]
fn a_policy_that_does_not_decide_in_contracts_is_refused() {
    let policy = LimitsPolicy::new(RiskLimits::new(dec!(10), dec!(1000), dec!(0.5)).unwrap());
    let error = two_contracts(dec!(0), both_ledger(), Script::default(), vec![], vec![])
        .with_pretrade_policy(policy)
        .unwrap_err();
    assert!(
        matches!(error, MMError::InvalidConfiguration(ref reason) if reason.contains("in contracts")),
        "{error:?}"
    );
}

#[test]
fn a_contract_the_ledger_does_not_hold_with_those_terms_is_refused() {
    let refused = |engine: MMResult<Engine>, reason: &str| match engine {
        Err(MMError::InvalidConfiguration(text)) => {
            assert!(text.contains(reason), "{text}");
        }
        other => panic!("{reason}: {other:?}"),
    };
    let add = |ledger: Ledger, terms: FutureTerms| {
        ContractEngine::new(config(dec!(0)), ledger, Script::default()).with_contract(
            terms,
            data(vec![]),
            ImmediateFillModel::new(),
        )
    };
    refused(
        add(both_ledger(), terms_of(spec_of("OTHER", FAR_MS))),
        "not registered",
    );
    // Registered under this name with another expiry.
    refused(
        add(both_ledger(), terms_of(spec_of("NEAR", FAR_MS))),
        "NEAR",
    );
    refused(
        add(both_ledger(), near()).and_then(|engine| {
            engine.with_contract(near(), data(vec![]), ImmediateFillModel::new())
        }),
        "already added",
    );
    let asset = |code| AssetCode::new(code).unwrap();
    let mut usdc = InstrumentDraft::from(spec_of("USDC", FAR_MS));
    usdc.price_asset = asset("USDC");
    usdc.settlement_asset = asset("USDC");
    let usdc = InstrumentSpec::new(usdc).unwrap();
    let mut ledger = ledger_of(&[near().spec().clone()]);
    ledger
        .register_asset(Asset::new("USDC").unwrap(), 8)
        .unwrap();
    ledger
        .register_instrument(
            Instrument::from_spec(
                &usdc,
                InstrumentScales {
                    price_scale: 8,
                    contract_scale: 8,
                },
            )
            .unwrap(),
        )
        .unwrap();
    refused(
        add(ledger, near()).and_then(|engine| {
            engine.with_contract(terms_of(usdc), data(vec![]), ImmediateFillModel::new())
        }),
        "unlike the others",
    );
    let engine = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 9),
        both_ledger(),
        Script::default(),
    )
    .with_contract(near(), data(vec![]), ImmediateFillModel::new());
    refused(engine, "fee scale");
}

#[test]
fn a_settlement_must_be_of_a_contract_added_with_its_terms_and_only_one() {
    let refused = |engine: MMResult<Engine>, reason: &str| match engine {
        Err(MMError::InvalidConfiguration(text)) => {
            assert!(text.contains(reason), "{text}");
        }
        other => panic!("{reason}: {other:?}"),
    };
    let engine = || two_contracts(dec!(0), both_ledger(), Script::default(), vec![], vec![]);
    let other = terms_of(spec_of("OTHER", NEAR_MS));
    refused(
        engine().with_settlement(settlement(&other, dec!(100), NEAR_MS)),
        "was not added",
    );
    // Published before the expiry, which the contract refuses.
    refused(
        engine().with_settlement(settlement(&near(), dec!(100), NEAR_MS - 1)),
        "settlement of NEAR",
    );
    refused(
        engine()
            .with_settlement(settlement(&near(), dec!(100), NEAR_MS))
            .unwrap()
            .with_settlement(settlement(&near(), dec!(101), NEAR_MS)),
        "already declared",
    );
}

#[test]
fn a_run_that_could_not_be_valued_is_refused_before_it_starts() {
    let mut empty: Engine = ContractEngine::new(config(dec!(0)), both_ledger(), Script::default());
    let result = empty.run();
    assert!(
        result
            .accounting
            .setup_refused
            .unwrap()
            .contains("needs a contract")
    );

    // The far contract is open in the ledger, and has no data here.
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "FAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = ContractEngine::new(config(dec!(0)), ledger, Script::default())
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.num_ticks, 0);
    assert!(
        result
            .accounting
            .setup_refused
            .unwrap()
            .contains("FAR is open")
    );
}

#[test]
fn fills_past_the_balance_book_and_are_reported() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(2),
    )]]);
    let mut engine = two_contracts(
        dec!(0.001),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    );
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(
        result.overdrafts.get(&Asset::new("USDT").unwrap()),
        Some(&dec!(0.2))
    );
}

#[test]
fn execution_numbers_continue_past_any_journal_of_the_ledger() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "sim-41",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let script = Script::of(vec![vec![order(
        1,
        far_id(),
        Action::Sell,
        dec!(105),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![tick(10, dec!(98), dec!(102))],
        vec![tick(20, dec!(105), dec!(107))],
    );
    let result = engine.run();
    assert_eq!(result.trades[0].execution_id, "sim-42");
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn commands_after_the_data_ended_are_discarded_and_counted() {
    // The fill is delivered after the data ended, and its answer has nowhere to go.
    let mut script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    // Off the grid, so if it ran it would be rejected, and that would show.
    script.on_fills = vec![order(2, near_id(), Action::Sell, dec!(101.5), dec!(1))];
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(500))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.discarded_commands, 1);
    assert_eq!(result.num_trades, 1);
    assert!(engine.execution_reports().iter().all(|r| r.tag == 1));
}

#[test]
fn a_reset_restores_the_ledger_data_models_and_counters() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(2),
    );
    let script = Script::of(vec![vec![order(
        1,
        far_id(),
        Action::Sell,
        dec!(105),
        dec!(2),
    )]]);
    let mut engine = two_contracts(
        dec!(0.001),
        ledger,
        script,
        vec![tick(10, dec!(100), dec!(102))],
        vec![
            tick(20, dec!(105), dec!(107)),
            tick(NEAR_MS + 1, dec!(105), dec!(107)),
        ],
    )
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS))
    .unwrap();
    let first = engine.run();
    let snapshot = engine.ledger().snapshot();
    let reports = engine.execution_reports().to_vec();
    let log = engine.strategy().log.clone();
    engine.reset();
    assert!(engine.execution_reports().is_empty());
    assert_eq!(engine.ledger().journal().len(), 1);
    let second = engine.run();
    assert_eq!(second.equity_curve, first.equity_curve);
    assert_eq!(second.trades, first.trades);
    assert_eq!(second.net_pnl, first.net_pnl);
    assert_eq!(second.contracts, first.contracts);
    assert_eq!(engine.execution_reports(), &reports[..]);
    assert_eq!(engine.strategy().log, log);
    assert_eq!(engine.ledger().snapshot(), snapshot);
}

#[test]
fn observations_on_their_way_when_the_data_ends_are_dropped() {
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        Script::default(),
        vec![tick(10, dec!(98), dec!(102))],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_observation_ms(50))
    .unwrap()
    .with_settlement(settlement(&near(), dec!(100), NEAR_MS))
    .unwrap();
    engine.run();
    assert!(
        engine.strategy().log.iter().all(|l| !l.starts_with("tick")),
        "{:?}",
        engine.strategy().log
    );
}

#[test]
fn a_price_slippage_takes_below_zero_does_not_execute() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Sell,
        dec!(100),
        dec!(1),
    )]]);
    let mut config = config(dec!(0));
    config.slippage = SlippageModel::Fixed(dec!(100));
    let mut engine = ContractEngine::new(config, both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(20, dec!(100), dec!(102)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn slippage_goes_against_the_order_on_each_side() {
    let script = Script::of(vec![vec![
        order(1, near_id(), Action::Buy, dec!(100), dec!(1)),
        order(2, far_id(), Action::Sell, dec!(105), dec!(1)),
    ]]);
    let mut config = config(dec!(0));
    config.slippage = SlippageModel::Fixed(dec!(1));
    let mut engine = ContractEngine::new(config, both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(20, dec!(98), dec!(100)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap()
        .with_contract(
            far(),
            data(vec![tick(30, dec!(105), dec!(107))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.trades[0].price, dec!(101));
    assert_eq!(result.trades[1].price, dec!(104));
}

/// A model's fill quantities are cumulative, and never pass the order.
#[test]
fn a_partial_fill_counts_what_it_adds_up_to_the_order() {
    #[derive(Debug, Clone)]
    struct Cumulative(Decimal);
    impl FillModel for Cumulative {
        fn simulate_fill(
            &self,
            _order: &SimulatedOrder,
            tick: &MarketTick,
            _time_in_queue_ms: u64,
        ) -> MMResult<FillResult> {
            Ok(FillResult::PartialFill {
                filled_quantity: self.0 * Decimal::from(tick.timestamp),
                fill_price: dec!(100),
            })
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "cumulative"
        }
    }
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(5),
    )]]);
    let mut engine = ContractEngine::new(config(dec!(0)), both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(1, dec!(98), dec!(102)),
                tick(2, dec!(98), dec!(102)),
                tick(3, dec!(98), dec!(102)),
                tick(9, dec!(98), dec!(102)),
            ]),
            Cumulative(dec!(1)),
        )
        .unwrap();
    let result = engine.run();
    let filled: Vec<Decimal> = result.trades.iter().map(|t| t.quantity).collect();
    assert_eq!(filled, vec![dec!(2), dec!(1), dec!(2)]);
    assert_eq!(line(&result, &near_id()).position, dec!(5));
}

#[test]
fn a_fill_the_ledger_refuses_is_recorded_and_the_run_is_not_consistent() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut config = config(dec!(0));
    // More digits than the instrument's price scale of 8.
    config.slippage = SlippageModel::Fixed(dec!(0.000000001));
    let mut engine = ContractEngine::new(config, both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(20, dec!(98), dec!(100)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 1);
    assert_eq!(result.accounting.refused_executions, vec![1]);
    assert_eq!(engine.accounting_errors().len(), 1);
    assert!(engine.ledger().journal().is_empty());
    assert_eq!(line(&result, &near_id()).fills, 0);
    assert!(!result.is_consistent());
}

#[test]
fn cancelling_a_contract_leaves_the_orders_of_another() {
    let script = Script::of(vec![
        vec![
            order(1, near_id(), Action::Buy, dec!(90), dec!(1)),
            order(2, far_id(), Action::Sell, dec!(200), dec!(1)),
        ],
        vec![ContractCommand::CancelAll { contract: far_id() }],
    ]);
    // The second tick comes while both are still on their way.
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(11, dec!(98), dec!(102))],
        vec![],
    )
    .with_latency(SimLatencyConfig::new().with_submit_ms(100))
    .unwrap();
    engine.run();
    let canceled: Vec<u64> = engine
        .execution_reports()
        .iter()
        .filter(|r| r.report.kind == ExecutionKind::Canceled)
        .map(|r| r.tag)
        .collect();
    assert_eq!(canceled, vec![2]);
}

#[test]
fn after_a_settlement_a_strategy_knows_the_position_is_gone() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(2),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![tick(NEAR_MS + 5, dec!(105), dec!(107))],
    )
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS))
    .unwrap();
    engine.run();
    let log = &engine.strategy().log;
    assert_eq!(log[0], "tick NEAR 10 near 2 far 0");
    assert_eq!(log[1], "settled NEAR at 103 closing 2");
    assert_eq!(log[2], "tick FAR 1005 near 0 far 0");
}

#[test]
fn a_settlement_reaches_the_strategy_after_the_report_latency() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(10, dec!(100), dec!(102))],
        vec![tick(NEAR_MS + 5, dec!(105), dec!(107))],
    )
    .with_latency(SimLatencyConfig::new().with_ack_ms(100))
    .unwrap()
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS))
    .unwrap();
    engine.run();
    let log = &engine.strategy().log;
    assert_eq!(log[1], "tick FAR 1005 near 1 far 0");
    assert_eq!(log[2], "settled NEAR at 103 closing 1");
}

#[test]
fn a_reset_forgets_the_marks_of_the_last_run() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        Script::default(),
        vec![tick(30, dec!(100), dec!(102))],
        vec![tick(10, dec!(105), dec!(107))],
    );
    let first = engine.run();
    assert_eq!(first.unvalued_ticks, 1);
    engine.reset();
    let second = engine.run();
    assert_eq!(second.unvalued_ticks, 1);
    assert_eq!(second.equity_curve, first.equity_curve);
}

#[test]
fn equity_overflow_is_visible_in_accounting_status() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let mut config = config(dec!(0));
    config.initial_capital = Decimal::MAX;
    let script = Script::of(vec![
        vec![],
        vec![order(1, near_id(), Action::Buy, dec!(90), dec!(1))],
    ]);
    let mut engine = ContractEngine::new(config, ledger, script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(100), dec!(102)),
                tick(20, dec!(100), dec!(102)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert!(result.accounting.valuation_failures > 0);
    assert!(!result.is_consistent());
    assert!(result.equity_curve.is_empty(), "{:?}", result.equity_curve);
    assert_eq!(result.total_pnl, dec!(0));
    // A fault stops new orders.
    assert_eq!(
        engine.execution_reports().last().unwrap().report.kind,
        ExecutionKind::Rejected
    );
}

#[test]
fn exhausted_execution_number_never_replays_a_seed_fill() {
    let mut ledger = both_ledger();
    let last = format!("sim-{}", u64::MAX);
    seed(&mut ledger, &last, "NEAR", Action::Buy, dec!(100), dec!(1));
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let policy = Recorded::of(PretradeDecision::Accepted);
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    )
    .with_pretrade_policy(policy.clone())
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert_eq!(result.accounting.dropped_fills, 1);
    assert!(!result.is_consistent());
    assert_eq!(line(&result, &near_id()).position, dec!(1));
    assert_eq!(line(&result, &near_id()).fills, 0);
    assert_eq!(engine.ledger().journal().len(), 1);
    assert!(engine.faults()[0].to_string().contains("used up"));
    // Nothing moved: the order is open, and the policy saw no fill.
    assert!(
        policy
            .events
            .lock()
            .unwrap()
            .iter()
            .all(|event| !matches!(event, OrderEvent::Fill { .. }))
    );
    assert!(
        engine
            .execution_reports()
            .iter()
            .all(|r| r.report.kind == ExecutionKind::Accepted)
    );
}

#[test]
fn failed_declared_settlement_is_not_a_consistent_run_even_when_flat() {
    // More digits than the ledger holds, flat all along.
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        Script::default(),
        vec![tick(10, dec!(98), dec!(102))],
        vec![],
    )
    .with_settlement(settlement(&near(), dec!(0.000000001), NEAR_MS))
    .unwrap();
    let result = engine.run();
    let near_line = line(&result, &near_id());
    assert!(near_line.settlement_failed);
    assert!(!near_line.settled);
    assert_eq!(result.fault_count, 1);
    assert!(!result.is_consistent());

    // What only fails on the position it finds, flat when declared.
    let huge = dec!(10000000000000000000);
    let script = Script::of(vec![vec![order(1, near_id(), Action::Buy, dec!(1), huge)]]);
    let mut engine = two_contracts(
        dec!(0),
        both_ledger(),
        script,
        vec![
            MarketTick::new(10, dec!(1), huge, dec!(3), huge),
            MarketTick::new(20, dec!(0), huge, dec!(1), huge),
        ],
        vec![],
    )
    .with_settlement(settlement(&near(), dec!(10000000000), NEAR_MS))
    .unwrap();
    let result = engine.run();
    let near_line = line(&result, &near_id());
    assert_eq!(result.num_trades, 1);
    assert!(near_line.settlement_failed);
    assert!(!near_line.settled);
    assert_eq!(near_line.position, huge);
    assert!(!result.is_consistent());
    assert!(
        engine.faults()[0]
            .to_string()
            .contains("settlement of NEAR")
    );
    assert!(
        engine
            .strategy()
            .log
            .iter()
            .all(|l| !l.starts_with("settled"))
    );
}

/// Whether a settlement can book depends on the position when it is published, not on
/// the one when it is declared.
#[test]
fn a_future_settlement_can_be_declared_before_the_position_is_closed() {
    let mut ledger = both_ledger();
    seed(&mut ledger, "seed-1", "NEAR", Action::Buy, dec!(1), dec!(2));
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Sell,
        dec!(1),
        dec!(2),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![tick(10, dec!(0.5), dec!(1.5)), tick(20, dec!(1), dec!(3))],
        vec![],
    )
    .with_settlement(settlement(&near(), Decimal::MAX, NEAR_MS))
    .unwrap();
    let result = engine.run();
    let near_line = line(&result, &near_id());
    assert_eq!(near_line.position, dec!(0));
    assert!(near_line.settled);
    assert!(result.is_consistent(), "{result:?}");
}

/// A tick observed after its contract settled shows the position gone, and an order sent
/// on it is rejected locally.
#[test]
fn a_late_observation_of_a_contract_already_settled() {
    let mut ledger = both_ledger();
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(1),
    );
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Sell,
        dec!(100),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0),
        ledger,
        script,
        vec![tick(990, dec!(100), dec!(102))],
        vec![tick(1100, dec!(105), dec!(107))],
    )
    .with_latency(SimLatencyConfig::new().with_observation_ms(50))
    .unwrap()
    .with_settlement(settlement(&near(), dec!(103), NEAR_MS))
    .unwrap();
    let result = engine.run();
    let log = &engine.strategy().log;
    assert_eq!(log[0], "settled NEAR at 103 closing 1");
    assert_eq!(log[1], "tick NEAR 990 near 0 far 0");
    assert_eq!(result.num_trades, 0);
    assert_eq!(result.total_pnl, dec!(3));
    assert_eq!(
        engine.execution_reports()[0].report.kind,
        ExecutionKind::Rejected
    );
}

/// A strategy that answers every rejection with two new orders, at zero latency.
#[derive(Debug, Default)]
struct Insists {
    reports: u64,
    tag: u64,
}

impl Insists {
    fn again(&mut self) -> Vec<ContractCommand> {
        self.tag += 2;
        vec![
            order(self.tag - 1, near_id(), Action::Buy, dec!(100.5), dec!(1)),
            order(self.tag, near_id(), Action::Buy, dec!(100.5), dec!(1)),
        ]
    }
}

impl ContractStrategy for Insists {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        _tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.again()
    }
    fn on_report(
        &mut self,
        _report: &ContractReport,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.reports += 1;
        self.again()
    }
    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[test]
fn a_strategy_that_never_stops_answering_is_a_fault_and_the_run_ends() {
    let mut config = config(dec!(0));
    config.record_trades = false;
    let mut engine = ContractEngine::new(config, both_ledger(), Insists::default())
        .with_contract(
            near(),
            data(vec![tick(10, dec!(98), dec!(102))]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.fault_count, 1);
    assert!(result.discarded_commands > 0);
    assert!(!result.is_consistent());
    assert_eq!(engine.strategy().reports, MAX_COMMANDS_PER_INSTANT);
    assert!(engine.faults()[0].to_string().contains("commands at 10"));
}

#[test]
fn up_to_the_budget_commands_run_and_each_time_has_its_own() {
    // Two ticks whose rejections together pass one budget, but not one each.
    let half = usize::try_from(MAX_COMMANDS_PER_INSTANT / 2 + 1).unwrap();
    let many = |from: u64| -> Vec<ContractCommand> {
        (0..half as u64)
            .map(|i| order(from + i, near_id(), Action::Buy, dec!(100.5), dec!(1)))
            .collect()
    };
    let script = Script::of(vec![many(0), many(1_000_000)]);
    let mut config = config(dec!(0));
    config.record_trades = false;
    config.record_equity_curve = false;
    let mut engine = ContractEngine::new(config, both_ledger(), script)
        .with_contract(
            near(),
            data(vec![
                tick(10, dec!(98), dec!(102)),
                tick(20, dec!(98), dec!(102)),
            ]),
            ImmediateFillModel::new(),
        )
        .unwrap();
    let result = engine.run();
    assert_eq!(result.fault_count, 0);
    assert_eq!(result.discarded_commands, 0);
}

#[test]
fn a_second_run_needs_a_reset() {
    let script = Script::of(vec![vec![order(
        1,
        near_id(),
        Action::Buy,
        dec!(100),
        dec!(1),
    )]]);
    let mut engine = two_contracts(
        dec!(0.001),
        both_ledger(),
        script,
        vec![tick(10, dec!(98), dec!(102)), tick(20, dec!(98), dec!(100))],
        vec![],
    );
    let first = engine.run();
    assert!(first.is_consistent(), "{first:?}");
    let again = engine.run();
    assert_eq!(again.num_ticks, 0);
    assert!(
        again
            .accounting
            .setup_refused
            .as_ref()
            .unwrap()
            .contains("needs a reset")
    );
    assert!(!again.is_consistent());
    // The ledger did not move, and a reset opens it again.
    assert_eq!(engine.ledger().journal().len(), 1);
    engine.reset();
    let second = engine.run();
    assert_eq!(second.net_pnl, first.net_pnl);
    assert!(second.is_consistent(), "{second:?}");
}

// The single contract engine and this one, on the same run.

/// `(bid price, bid size, ask price, ask size)` per tick; `None` leaves the orders.
type Quotes = Vec<Option<(Decimal, Decimal, Decimal, Decimal)>>;

#[derive(Debug, Clone)]
struct QuoteScript(Quotes, usize, Vec<String>);

impl BacktestStrategy for QuoteScript {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        self.2.push(format!("tick {}", tick.timestamp));
        let step = self.0.get(self.1).copied().flatten();
        self.1 += 1;
        step.map(|(bid_price, bid_size, ask_price, ask_size)| Quote {
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, fill: &SimulatedFill) {
        self.2
            .push(format!("fill {} {}", fill.price, fill.quantity));
    }
    fn on_execution_report(&mut self, report: &market_maker_rs::backtest::ExecutionReport) {
        self.2
            .push(format!("{} {:?}", report.order_id, report.kind));
    }
    fn reset(&mut self) {
        self.1 = 0;
        self.2.clear();
    }
}

/// What the quote script does, in orders: cancel all, then the bid and the ask.
#[derive(Debug, Clone)]
struct OrderScript(Quotes, usize, u64, Vec<String>);

impl ContractStrategy for OrderScript {
    fn on_tick(
        &mut self,
        contract: &SpecId,
        tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.3.push(format!("tick {}", tick.timestamp));
        let step = self.0.get(self.1).copied().flatten();
        self.1 += 1;
        let Some((bid_price, bid_size, ask_price, ask_size)) = step else {
            return Vec::new();
        };
        self.2 += 2;
        vec![
            ContractCommand::CancelAll {
                contract: contract.clone(),
            },
            order(
                self.2 - 1,
                contract.clone(),
                Action::Buy,
                bid_price,
                bid_size,
            ),
            order(self.2, contract.clone(), Action::Sell, ask_price, ask_size),
        ]
    }
    fn on_report(&mut self, report: &ContractReport, _view: &ContractView) -> Vec<ContractCommand> {
        self.3.push(format!(
            "{} {:?}",
            report.report.order_id, report.report.kind
        ));
        Vec::new()
    }
    fn on_fill(&mut self, fill: &ContractFill, _view: &ContractView) -> Vec<ContractCommand> {
        self.3
            .push(format!("fill {} {}", fill.price, fill.quantity));
        Vec::new()
    }
    fn reset(&mut self) {
        self.1 = 0;
        self.2 = 0;
        self.3.clear();
    }
}

fn backtest_config(fee_rate: Decimal) -> BacktestConfig {
    BacktestConfig {
        initial_capital: dec!(1000),
        fee_rate,
        fee_currency: FeeCurrency::Settlement { scale: 8 },
        tick_size: dec!(1),
        lot_size: dec!(1),
        default_order_size: dec!(1),
        ..BacktestConfig::default()
    }
}

/// A model that fails on one tick, and otherwise fills as another.
#[derive(Debug, Clone)]
struct FailingAt<F>(F, u64);

impl<F: FillModel> FillModel for FailingAt<F> {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> MMResult<FillResult> {
        if tick.timestamp == self.1 {
            return Err(MMError::InvalidMarketState("model failure".to_string()));
        }
        self.0.simulate_fill(order, tick, time_in_queue_ms)
    }
    fn on_tick(&mut self, tick: &MarketTick) {
        self.0.on_tick(tick);
    }
    fn reset(&mut self) {
        self.0.reset();
    }
    fn name(&self) -> &'static str {
        "failing"
    }
}

/// Runs both, twice with a reset between, and holds every figure and report equal.
fn agree<F: FillModel + Clone>(
    fee_rate: Decimal,
    ledger: Ledger,
    latency: SimLatencyConfig,
    model: F,
    quotes: Quotes,
    ticks: Vec<MarketTick>,
) -> ContractRunResult {
    let spec = near().spec().clone();
    let accountant =
        MarginLedgerAccountant::new(ledger.clone(), InstrumentId::new("NEAR").unwrap()).unwrap();
    let mut single = BacktestEngine::with_fill_model(
        backtest_config(fee_rate),
        QuoteScript(quotes.clone(), 0, Vec::new()),
        data(ticks.clone()),
        model.clone(),
    )
    .unwrap()
    .with_latency(latency)
    .unwrap()
    .with_instrument(spec, accountant)
    .unwrap();
    let mut engine = ContractEngine::new(
        config(fee_rate),
        ledger,
        OrderScript(quotes, 0, 0, Vec::new()),
    )
    .with_contract(near(), data(ticks), model)
    .unwrap()
    .with_latency(latency)
    .unwrap();
    let mut last = None;
    for run in 0..2 {
        if run == 1 {
            single.reset();
            engine.reset();
        }
        let theirs = single.run();
        let ours = engine.run();
        assert_eq!(ours.num_ticks, theirs.num_ticks);
        assert_eq!(ours.num_trades, theirs.num_trades);
        assert_eq!(
            (ours.start_time, ours.end_time),
            (theirs.start_time, theirs.end_time)
        );
        assert_eq!(ours.total_pnl, theirs.total_pnl);
        assert_eq!(ours.total_fees, theirs.total_fees);
        assert_eq!(ours.net_pnl, theirs.net_pnl);
        assert_eq!(ours.equity_curve, theirs.equity_curve);
        assert_eq!(ours.max_drawdown, theirs.max_drawdown);
        assert_eq!(ours.sharpe_ratio, theirs.sharpe_ratio);
        let summary = &ours.contracts[0];
        assert_eq!(summary.position, theirs.final_position);
        assert_eq!(summary.max_position, theirs.max_position);
        assert_eq!(
            ours.accounting.refused_executions,
            theirs.accounting.refused_executions
        );
        assert_eq!(
            ours.accounting.valuation_failures,
            theirs.accounting.valuation_failures
        );
        assert_eq!(
            ours.accounting.dropped_fills,
            theirs.accounting.dropped_fills
        );
        assert_eq!(engine.fault_count(), single.fault_count());
        assert_eq!(engine.faults(), single.faults());
        let their_fills: Vec<_> = theirs
            .trades
            .iter()
            .map(|t| {
                (
                    t.side,
                    t.price,
                    t.quantity,
                    t.timestamp,
                    t.fee,
                    format!("sim-{}", t.execution_id.unwrap()),
                )
            })
            .collect();
        let our_fills: Vec<_> = ours
            .trades
            .iter()
            .map(|t| {
                (
                    t.side,
                    t.price,
                    t.quantity,
                    t.timestamp,
                    t.fee,
                    t.execution_id.clone(),
                )
            })
            .collect();
        assert_eq!(our_fills, their_fills);
        let our_reports: Vec<_> = engine
            .execution_reports()
            .iter()
            .map(|r| r.report.clone())
            .collect();
        assert_eq!(our_reports, single.execution_reports());
        // What reached each strategy, in the order it did.
        assert_eq!(engine.strategy().3, single.strategy().2);
        let snapshot = single.accounting_snapshot().unwrap().unwrap();
        assert_eq!(summary.position, snapshot.position);
        if let Some(previous) = &last {
            let previous: &ContractRunResult = previous;
            assert_eq!(previous.equity_curve, ours.equity_curve);
            assert_eq!(previous.trades, ours.trades);
        }
        last = Some(ours);
    }
    last.unwrap()
}

/// Quotes that make and take around 100, on the grid.
fn quoting() -> (Quotes, Vec<MarketTick>) {
    let quote = |bid, ask| Some((bid, dec!(2), ask, dec!(1)));
    let quotes = vec![
        quote(dec!(99), dec!(103)),
        None,
        quote(dec!(100), dec!(104)),
        quote(dec!(100), dec!(104)),
        None,
        quote(dec!(101), dec!(105)),
        None,
        quote(dec!(90), dec!(110)),
    ];
    let ticks = vec![
        tick(10, dec!(99), dec!(101)),
        tick(12, dec!(98), dec!(99)),
        tick(20, dec!(100), dec!(102)),
        tick(25, dec!(104), dec!(106)),
        tick(25, dec!(103), dec!(105)),
        tick(40, dec!(97), dec!(100)),
        tick(41, dec!(105), dec!(107)),
        tick(60, dec!(100), dec!(102)),
    ];
    (quotes, ticks)
}

#[test]
fn one_contract_agrees_with_the_single_contract_engine() {
    let (quotes, ticks) = quoting();
    let result = agree(
        dec!(0.001),
        ledger_of(&[near().spec().clone()]),
        SimLatencyConfig::new(),
        ImmediateFillModel::new(),
        quotes,
        ticks,
    );
    assert!(result.num_trades > 2, "{result:?}");
}

#[test]
fn one_contract_agrees_with_latency_and_jitter_and_a_rebate() {
    let (quotes, ticks) = quoting();
    for seed in 1..=20 {
        let latency = SimLatencyConfig::new()
            .with_observation_ms(1)
            .with_submit_ms(2)
            .with_cancel_ms(3)
            .with_ack_ms(4)
            .with_jitter(9, seed);
        agree(
            dec!(-0.0005),
            ledger_of(&[near().spec().clone()]),
            latency,
            ImmediateFillModel::new(),
            quotes.clone(),
            ticks.clone(),
        );
    }
}

#[test]
fn one_contract_agrees_with_partial_fills_from_a_queue() {
    let (quotes, ticks) = quoting();
    let latency = SimLatencyConfig::new().with_submit_ms(1).with_cancel_ms(2);
    agree(
        dec!(0.001),
        ledger_of(&[near().spec().clone()]),
        latency,
        QueuePositionFillModel::new(dec!(0.5)),
        quotes,
        ticks,
    );
}

#[test]
fn one_contract_agrees_on_a_position_the_ledger_started_with() {
    let (quotes, ticks) = quoting();
    let mut ledger = ledger_of(&[near().spec().clone()]);
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Buy,
        dec!(100),
        dec!(3),
    );
    let result = agree(
        dec!(0.001),
        ledger,
        SimLatencyConfig::new(),
        ImmediateFillModel::new(),
        quotes,
        ticks,
    );
    // 3, buying 2 at 99 on tick 12, selling 1 at 104 on tick 25, buying 2 at 100 on tick
    // 40 for the largest, and selling 1 at 105 on tick 41.
    assert_eq!(result.contracts[0].position, dec!(5));
    assert_eq!(result.contracts[0].max_position, dec!(6));

    // With no fill, the largest position is the one it started with.
    let mut ledger = ledger_of(&[near().spec().clone()]);
    seed(
        &mut ledger,
        "seed-1",
        "NEAR",
        Action::Sell,
        dec!(100),
        dec!(3),
    );
    let result = agree(
        dec!(0),
        ledger,
        SimLatencyConfig::new(),
        ImmediateFillModel::new(),
        vec![],
        vec![tick(10, dec!(99), dec!(101))],
    );
    assert_eq!(result.contracts[0].max_position, dec!(3));
}

#[test]
fn one_contract_agrees_on_a_fault_of_the_fill_model() {
    let (quotes, ticks) = quoting();
    agree(
        dec!(0.001),
        ledger_of(&[near().spec().clone()]),
        SimLatencyConfig::new(),
        FailingAt(ImmediateFillModel::new(), 25),
        quotes,
        ticks,
    );
}
