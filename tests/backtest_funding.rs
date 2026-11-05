//! Perpetual funding in a contract backtest, against the shared funding fixtures in
//! `tests/contract_fixtures.rs`. Amounts are worked out by hand.

#![cfg(all(feature = "margin", feature = "instruments"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use financial_types::instrument::InstrumentId as SpecId;
use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::{Execution, FundingPayment, Ledger, LedgerConfig};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::contracts::{
    ContractCommand, ContractConfig, ContractEngine, ContractOrder, ContractRunResult,
    ContractStrategy, ContractView,
};
use market_maker_rs::backtest::{ImmediateFillModel, MarketTick, VecDataSource};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::instruments::futures::SettlementSource;
use market_maker_rs::instruments::perpetual::{FundingPoint, PerpetualError, PerpetualTerms};
use market_maker_rs::instruments::reference::deribit;
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentSpec, Margining, TickSchedule,
};
use market_maker_rs::risk::pretrade::OrderPurpose;

/// The 8 hour grid in milliseconds.
const STEP: u64 = 28_800_000;
const T0: u64 = 59_027 * STEP;

fn perp(multiplier: Decimal) -> InstrumentSpec {
    perp_named("PERP", multiplier)
}

fn perp_named(name: &str, multiplier: Decimal) -> InstrumentSpec {
    let asset = |code| AssetCode::new(code).unwrap();
    InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new(name).unwrap(),
        kind: ContractKind::Perpetual,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(multiplier, asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: None,
        option: None,
    })
    .unwrap()
}

fn ledger_of(spec: &InstrumentSpec, usdt_scale: u32) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger
        .register_asset(Asset::new("BTC").unwrap(), 8)
        .unwrap();
    ledger
        .register_asset(Asset::new("USDT").unwrap(), usdt_scale)
        .unwrap();
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
    ledger
}

fn seed(ledger: &mut Ledger, name: &str, side: Action, price: Decimal, quantity: Decimal) {
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("seed").unwrap(),
                InstrumentId::new(name).unwrap(),
                side,
                price,
                quantity,
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
}

fn point(at_ms: u64, mark: Decimal, rate: Decimal) -> FundingPoint {
    FundingPoint { at_ms, mark, rate }
}

fn source() -> SettlementSource {
    SettlementSource::new("the venue's funding history").unwrap()
}

fn terms(spec: InstrumentSpec, points: Vec<FundingPoint>) -> PerpetualTerms {
    PerpetualTerms::new(spec, STEP, source(), points).unwrap()
}

/// The fixture's schedule: instants 1, 2 and 3 after `T0`.
fn fixture_points() -> Vec<FundingPoint> {
    vec![
        point(T0 + STEP, dec!(60000), dec!(0.0001)),
        point(T0 + 2 * STEP, dec!(59000), dec!(-0.00005)),
        point(T0 + 3 * STEP, dec!(61000), dec!(0.0002)),
    ]
}

fn tick(timestamp: u64, mid: Decimal) -> MarketTick {
    MarketTick::new(
        timestamp,
        mid - dec!(1),
        dec!(100000),
        mid + dec!(1),
        dec!(100000),
    )
}

fn ticks_at(times: &[u64]) -> VecDataSource {
    VecDataSource::new(times.iter().map(|t| tick(*t, dec!(60000))).collect())
}

/// Sends commands when it sees a tick at given times.
#[derive(Debug, Clone, Default)]
struct Script(Vec<(u64, ContractCommand)>);

impl ContractStrategy for Script {
    fn on_tick(
        &mut self,
        _contract: &SpecId,
        tick: &MarketTick,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.0
            .iter()
            .filter(|(at, _)| *at == tick.timestamp)
            .map(|(_, command)| command.clone())
            .collect()
    }

    fn reset(&mut self) {}
}

fn market(tag: u64, side: Action, price: Decimal, quantity: Decimal) -> ContractCommand {
    ContractCommand::Submit(ContractOrder {
        tag,
        contract: SpecId::new("PERP").unwrap(),
        side,
        price,
        quantity,
        purpose: OrderPurpose::Other,
    })
}

type Engine = ContractEngine<Script, VecDataSource, ImmediateFillModel>;

fn engine(ledger: Ledger, script: Script, terms: PerpetualTerms, times: &[u64]) -> Engine {
    ContractEngine::new(ContractConfig::new(dec!(1000), dec!(0), 2), ledger, script)
        .with_perpetual(terms, ticks_at(times), ImmediateFillModel::new())
        .unwrap()
}

fn seeded_run(side: Action, quantity: Decimal) -> (ContractRunResult, Ledger) {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", side, dec!(60000), quantity);
    let times = [T0, T0 + STEP, T0 + 2 * STEP, T0 + 3 * STEP];
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &times,
    );
    let result = run.run();
    (result, run.ledger().clone())
}

#[test]
fn a_long_pays_the_fixture_schedule_through_the_ledger() {
    let (result, ledger) = seeded_run(Action::Buy, dec!(2));
    assert!(result.is_consistent(), "{result:?}");
    let amounts: Vec<Decimal> = result.fundings.iter().map(|f| f.amount).collect();
    assert_eq!(amounts, vec![dec!(-12), dec!(5.9), dec!(-24.4)]);
    assert_eq!(result.total_funding, dec!(-30.5));
    assert_eq!(result.contracts[0].funding, dec!(-30.5));
    assert_eq!(ledger.funding(&Asset::new("USDT").unwrap()), dec!(-30.5));
    assert_eq!(ledger.funding_journal().len(), 3);
    // Apart from realized PnL and fees: nothing traded in the run.
    assert_eq!(result.realized_pnl, dec!(0));
    assert_eq!(result.total_fees, dec!(0));
    assert_eq!(result.unrealized_pnl, dec!(0));
    assert_eq!(result.total_pnl, dec!(-30.5));
    assert_eq!(result.net_pnl, dec!(-30.5));
    // Each instant is on the curve, before the tick at the same time.
    assert_eq!(result.equity_curve.last().unwrap().1, dec!(969.5));
    assert!(result.equity_curve.contains(&(T0 + STEP, dec!(988))));
}

#[test]
fn a_short_mirrors_the_long_so_funding_is_zero_sum() {
    let (long, _) = seeded_run(Action::Buy, dec!(2));
    let (short, _) = seeded_run(Action::Sell, dec!(2));
    assert_eq!(short.total_funding, dec!(30.5));
    assert_eq!(long.total_funding + short.total_funding, dec!(0));
    // A negative rate reverses the direction.
    assert_eq!(short.fundings[1].amount, dec!(-5.9));
    assert_eq!(short.fundings[1].position, dec!(-2));
}

#[test]
fn paid_instants_are_half_open() {
    let spec = perp(dec!(1));
    let ledger = ledger_of(&spec, 8);
    // Opened by a fill exactly on instant 1, closed by a fill exactly on instant 3.
    let script = Script(vec![
        (T0 + 1, market(1, Action::Buy, dec!(60001), dec!(2))),
        (
            T0 + 2 * STEP + 1,
            market(2, Action::Sell, dec!(59999), dec!(2)),
        ),
    ]);
    let times = [
        T0,
        T0 + 1,
        T0 + STEP,
        T0 + 2 * STEP + 1,
        T0 + 3 * STEP,
        T0 + 4 * STEP,
    ];
    let mut points = fixture_points();
    points.push(point(T0 + 4 * STEP, dec!(61000), dec!(0.0002)));
    let mut run = engine(ledger, script, terms(spec, points), &times);
    let result = run.run();
    assert!(result.is_consistent(), "{result:?}");
    let paid: Vec<u64> = result.fundings.iter().map(|f| f.at_ms).collect();
    // Not instant 1, on which it opened; instant 3, on which it closed; not instant 4.
    assert_eq!(paid, vec![T0 + 2 * STEP, T0 + 3 * STEP]);
    assert_eq!(result.total_funding, dec!(5.9) - dec!(24.4));
    assert_eq!(result.num_trades, 2);
    // Realized is the trades alone: bought at 60001, sold at 59999.
    assert_eq!(result.realized_pnl, dec!(-4));
    assert_eq!(result.total_pnl, dec!(-4) + result.total_funding);
}

fn rounding_run(side: Action, quantity: Decimal) -> ContractRunResult {
    let spec = perp(dec!(0.1));
    let mut ledger = ledger_of(&spec, 2);
    seed(&mut ledger, "PERP", side, dec!(61234), quantity);
    let points = vec![point(T0 + STEP, dec!(61234.57), dec!(0.000123))];
    engine(
        ledger,
        Script::default(),
        terms(spec, points),
        &[T0, T0 + STEP],
    )
    .run()
}

#[test]
fn rounding_leaves_the_fixtures_explicit_residual() {
    // A 0.3 BTC long pays 2.259555633, settled as 2.26. A 0.1 BTC short receives
    // 0.753185211, settled as 0.75, so three of them receive 2.25.
    let long = rounding_run(Action::Buy, dec!(3));
    let short = rounding_run(Action::Sell, dec!(1));
    assert_eq!(long.total_funding, dec!(-2.26));
    assert_eq!(short.total_funding, dec!(0.75));
    assert_eq!(
        -long.total_funding - dec!(3) * short.total_funding,
        dec!(0.01)
    );
}

#[test]
fn an_inverse_perpetual_pays_in_its_settlement_asset() {
    // 100 contracts of 10 USD at 50000 are 0.02 BTC; at 0.0001 that is 0.000002 BTC.
    let spec = deribit::btc_perpetual().unwrap();
    let id = spec.id().as_str().to_string();
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, &id, Action::Buy, dec!(50000), dec!(100));
    let points = vec![point(T0 + STEP, dec!(50000), dec!(0.0001))];
    let result = ContractEngine::new(
        ContractConfig::new(dec!(1), dec!(0), 8),
        ledger,
        Script::default(),
    )
    .with_perpetual(
        terms(spec, points),
        VecDataSource::new(vec![tick(T0, dec!(50000)), tick(T0 + STEP, dec!(50000))]),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .run();
    assert!(result.is_consistent(), "{result:?}");
    assert_eq!(result.total_funding, dec!(-0.000002));
}

#[test]
fn an_instant_with_no_declared_rate_makes_the_run_not_consistent() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    // The ticks reach instant 5, and rates stop at 3.
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &[T0, T0 + 5 * STEP],
    );
    let result = run.run();
    assert_eq!(result.contracts[0].funding_missing, 2);
    assert!(!result.is_consistent());
    assert_eq!(result.total_funding, dec!(-30.5));
}

#[test]
fn only_instants_after_the_first_tick_count() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    // Instants 1 and 2 are at and before the first tick, and instant 3 is after the last.
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &[T0 + 2 * STEP, T0 + 2 * STEP + 1],
    );
    let result = run.run();
    assert!(result.fundings.is_empty(), "{:?}", result.fundings);
    assert!(result.is_consistent(), "{result:?}");
}

#[test]
fn a_flat_position_pays_nothing_and_an_amount_rounded_away_books_nothing() {
    let spec = perp(dec!(1));
    let flat = engine(
        ledger_of(&spec, 8),
        Script::default(),
        terms(spec.clone(), fixture_points()),
        &[T0, T0 + 3 * STEP],
    )
    .run();
    assert!(flat.fundings.is_empty());
    assert!(flat.is_consistent());

    let mut ledger = ledger_of(&spec, 2);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(1));
    let points = vec![point(T0 + STEP, dec!(1), dec!(0.0001))];
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, points),
        &[T0, T0 + STEP],
    );
    let result = run.run();
    assert!(result.is_consistent(), "{result:?}");
    assert_eq!(result.fundings[0].amount, dec!(0));
    assert!(run.ledger().funding_journal().is_empty());
}

#[test]
fn funding_the_balance_does_not_cover_is_an_overdraft() {
    let (result, _) = seeded_run(Action::Buy, dec!(2));
    // Balance 0: -12, then -6.1, then -30.5.
    assert_eq!(result.overdrafts[&Asset::new("USDT").unwrap()], dec!(30.5));
    let (short, _) = seeded_run(Action::Sell, dec!(2));
    // Receives 12 first, so 6.1 stays covered and the balance never goes below zero.
    assert!(short.overdrafts.is_empty(), "{:?}", short.overdrafts);
}

#[test]
fn funding_the_ledger_already_held_is_not_counted_again_and_a_clash_is_a_fault() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let held = |id: &str, amount| FundingPayment {
        id: ExecutionId::new(id).unwrap(),
        instrument: InstrumentId::new("PERP").unwrap(),
        amount,
        as_of_ms: 2,
    };
    ledger.book_funding(&held("before", dec!(-7))).unwrap();
    let times = [T0, T0 + STEP];
    let points = vec![point(T0 + STEP, dec!(60000), dec!(0.0001))];
    let result = engine(
        ledger.clone(),
        Script::default(),
        terms(spec.clone(), points.clone()),
        &times,
    )
    .run();
    assert_eq!(result.total_funding, dec!(-12));
    assert_eq!(result.total_pnl, dec!(-12));
    assert_eq!(result.realized_pnl, dec!(0));

    // Another payment under the identifier this run would use.
    let clash = funding_id_at_instant_1();
    ledger.book_funding(&held(&clash, dec!(-1))).unwrap();
    let result = engine(ledger, Script::default(), terms(spec, points), &times).run();
    assert_eq!(result.fault_count, 1);
    assert!(!result.is_consistent());
    assert_eq!(result.contracts[0].funding, dec!(0));
    assert!(result.fundings.is_empty());
}

#[test]
fn a_reset_runs_the_funding_again() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let times = [T0, T0 + 3 * STEP];
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &times,
    );
    let first = run.run();
    run.reset();
    let second = run.run();
    assert_eq!(first.total_funding, dec!(-30.5));
    assert_eq!(second.total_funding, dec!(-30.5));
    assert_eq!(second.contracts[0].funding, dec!(-30.5));
    assert_eq!(second.fundings, first.fundings);
}

#[test]
fn trades_not_recorded_leave_out_the_journal_only() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let mut config = ContractConfig::new(dec!(1000), dec!(0), 2);
    config.record_trades = false;
    let result = ContractEngine::new(config, ledger, Script::default())
        .with_perpetual(
            terms(spec, fixture_points()),
            ticks_at(&[T0, T0 + 3 * STEP]),
            ImmediateFillModel::new(),
        )
        .unwrap()
        .run();
    assert!(result.fundings.is_empty());
    assert_eq!(result.total_funding, dec!(-30.5));
}

#[test]
fn terms_refuse_what_is_not_a_funding_schedule() {
    let good = || vec![point(STEP, dec!(1), dec!(0))];
    let new = |spec, interval, points| PerpetualTerms::new(spec, interval, source(), points);
    assert!(matches!(
        new(perp(dec!(1)), 0, good()),
        Err(PerpetualError::ZeroInterval)
    ));
    assert!(matches!(
        new(perp(dec!(1)), STEP, vec![]),
        Err(PerpetualError::NoFunding)
    ));
    assert!(matches!(
        new(perp(dec!(1)), STEP, vec![point(STEP + 1, dec!(1), dec!(0))]),
        Err(PerpetualError::OffGrid { .. })
    ));
    assert!(matches!(
        new(
            perp(dec!(1)),
            STEP,
            vec![
                point(STEP, dec!(1), dec!(0)),
                point(3 * STEP, dec!(1), dec!(0))
            ]
        ),
        Err(PerpetualError::NotConsecutive { .. })
    ));
    assert!(matches!(
        new(
            perp(dec!(1)),
            STEP,
            vec![point(STEP, dec!(1), dec!(0)), point(STEP, dec!(1), dec!(0))]
        ),
        Err(PerpetualError::NotConsecutive { .. })
    ));
    assert!(matches!(
        new(perp(dec!(1)), STEP, vec![point(STEP, dec!(0), dec!(0))]),
        Err(PerpetualError::NotPositiveMark { .. })
    ));
    let asset = |code| AssetCode::new(code).unwrap();
    let future = InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new("FUT").unwrap(),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: Some(STEP),
        option: None,
    })
    .unwrap();
    assert!(matches!(
        new(future, STEP, good()),
        Err(PerpetualError::NotAPerpetual { .. })
    ));
}

#[test]
fn missing_counts_the_grid_outside_the_declared_points() {
    let t = terms(perp(dec!(1)), fixture_points());
    let i = |n: u64| T0 + n * STEP;
    assert_eq!(t.missing_between(i(0), i(3)), 0);
    assert_eq!(t.missing_between(i(0), i(5)), 2);
    assert_eq!(t.missing_between(i(0), i(3) - 1), 0);
    assert_eq!(t.missing_between(i(0) - STEP, i(1)), 1);
    assert_eq!(t.missing_between(i(1), i(2)), 0);
    assert_eq!(t.missing_between(i(4), i(6)), 2);
    assert_eq!(t.missing_between(i(3), i(3)), 0);
    assert_eq!(t.missing_between(i(0) - 5 * STEP, i(0) - 3 * STEP), 2);
}

#[test]
fn a_perpetual_has_no_settlement() {
    use market_maker_rs::instruments::futures::SettlementPrice;
    let spec = perp(dec!(1));
    let run = engine(
        ledger_of(&spec, 8),
        Script::default(),
        terms(spec.clone(), fixture_points()),
        &[T0],
    );
    let price = SettlementPrice::new(spec.id().clone(), source(), T0, dec!(1), T0).unwrap();
    let error = run.with_settlement(price).unwrap_err().to_string();
    assert!(error.contains("a perpetual does not settle"), "{error}");
}

#[test]
fn an_instant_with_no_tick_is_on_the_equity_curve() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &[T0, T0 + 3 * STEP + 1],
    );
    let result = run.run();
    assert!(result.equity_curve.contains(&(T0 + STEP, dec!(988))));
    assert!(result.equity_curve.contains(&(T0 + 2 * STEP, dec!(993.9))));
    assert!(result.equity_curve.contains(&(T0 + 3 * STEP, dec!(969.5))));
}

#[test]
fn the_identical_payment_already_in_the_ledger_is_a_fault_not_a_free_instant() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    ledger
        .book_funding(&FundingPayment {
            id: ExecutionId::new(funding_id_at_instant_1()).unwrap(),
            instrument: InstrumentId::new("PERP").unwrap(),
            amount: dec!(-12),
            as_of_ms: T0 + STEP,
        })
        .unwrap();
    let points = vec![point(T0 + STEP, dec!(60000), dec!(0.0001))];
    let result = engine(
        ledger,
        Script::default(),
        terms(spec, points),
        &[T0, T0 + STEP],
    )
    .run();
    assert_eq!(result.fault_count, 1);
    assert!(!result.is_consistent());
    assert_eq!(result.contracts[0].funding, dec!(0));
}

#[test]
fn funding_books_for_the_longest_identifier_a_ledger_takes() {
    let name = "P".repeat(64);
    let spec = perp_named(&name, dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, &name, Action::Buy, dec!(60000), dec!(2));
    let result = engine(
        ledger,
        Script::default(),
        terms(spec, fixture_points()),
        &[T0, T0 + 3 * STEP],
    )
    .run();
    assert!(result.is_consistent(), "{result:?}");
    assert_eq!(result.total_funding, dec!(-30.5));
}

/// The identifier a run gives instant 1 of `PERP`.
fn funding_id_at_instant_1() -> String {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let points = vec![point(T0 + STEP, dec!(60000), dec!(0.0001))];
    let mut run = engine(
        ledger,
        Script::default(),
        terms(spec, points),
        &[T0, T0 + STEP],
    );
    run.run();
    run.ledger().funding_journal()[0].id.as_str().to_string()
}

#[test]
fn funding_at_the_first_tick_is_excluded_at_the_end_of_the_clock() {
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let points = vec![point(u64::MAX, dec!(60000), dec!(0.0001))];
    let result = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 2),
        ledger,
        Script::default(),
    )
    .with_perpetual(
        PerpetualTerms::new(spec, 1, source(), points).unwrap(),
        ticks_at(&[u64::MAX]),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .run();
    assert!(result.fundings.is_empty(), "{:?}", result.fundings);
}

#[test]
fn replaying_an_instant_on_a_funded_ledger_is_a_fault_whatever_the_lane_order() {
    let asset = |code| AssetCode::new(code).unwrap();
    let future = InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new("FUT").unwrap(),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: Some(T0 + 100 * STEP),
        option: None,
    })
    .unwrap();
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    ledger
        .register_instrument(
            Instrument::from_spec(
                &future,
                InstrumentScales {
                    price_scale: 8,
                    contract_scale: 8,
                },
            )
            .unwrap(),
        )
        .unwrap();
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let points = vec![point(T0 + STEP, dec!(60000), dec!(0.0001))];
    let times = [T0, T0 + STEP];
    // The perpetual is the first lane here and the second below.
    let mut first = engine(
        ledger,
        Script::default(),
        terms(spec.clone(), points.clone()),
        &times,
    );
    assert_eq!(first.run().total_funding, dec!(-12));
    let funded = first.ledger().clone();
    let again = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 2),
        funded,
        Script::default(),
    )
    .with_contract(
        market_maker_rs::instruments::futures::FutureTerms::new(
            future,
            market_maker_rs::instruments::futures::SettlementMethod::Cash,
            source(),
        )
        .unwrap(),
        ticks_at(&times),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .with_perpetual(
        terms(spec, points),
        ticks_at(&times),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .run();
    assert_eq!(again.fault_count, 1);
    assert_eq!(again.contracts[1].funding, dec!(0));
}

#[test]
fn a_run_with_no_ticks_pays_no_funding_even_if_a_settlement_runs_the_clock() {
    use market_maker_rs::instruments::futures::{FutureTerms, SettlementMethod, SettlementPrice};
    let asset = |code| AssetCode::new(code).unwrap();
    let expiry = T0 + 3 * STEP;
    let future = InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new("FUT").unwrap(),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: Some(expiry),
        option: None,
    })
    .unwrap();
    let spec = perp(dec!(1));
    let mut ledger = ledger_of(&spec, 8);
    ledger
        .register_instrument(
            Instrument::from_spec(
                &future,
                InstrumentScales {
                    price_scale: 8,
                    contract_scale: 8,
                },
            )
            .unwrap(),
        )
        .unwrap();
    seed(&mut ledger, "PERP", Action::Buy, dec!(60000), dec!(2));
    let id = future.id().clone();
    let mut run = ContractEngine::new(
        ContractConfig::new(dec!(1000), dec!(0), 2),
        ledger,
        Script::default(),
    )
    .with_perpetual(
        terms(spec, fixture_points()),
        ticks_at(&[]),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .with_contract(
        FutureTerms::new(future, SettlementMethod::Cash, source()).unwrap(),
        ticks_at(&[]),
        ImmediateFillModel::new(),
    )
    .unwrap()
    .with_settlement(SettlementPrice::new(id, source(), expiry, dec!(1), expiry).unwrap())
    .unwrap();
    let result = run.run();
    assert!(result.contracts[1].settled);
    assert!(result.fundings.is_empty(), "{:?}", result.fundings);
    assert!(run.ledger().funding_journal().is_empty());
}
