//! Initial margin before a derivative order reaches the book (#230). Expected values are worked
//! out by hand in the comments. `linear-im/1` asks 10% of the notional at the last mark.

#![cfg(all(feature = "margin", feature = "instruments"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::{Ledger, LedgerConfig};
use margin_rs::risk::RiskGate;
use margin_rs::rules::TieredLinearIm;
use margin_rs::{
    AccountId, Asset, Instrument, InstrumentId, ReservationId, SpotInstrument, TransferId, VenueId,
};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::engine::{BacktestConfig, BacktestStrategy, FeeCurrency};
use market_maker_rs::backtest::{
    BacktestEngine, BacktestResult, DerivativeMarginAccount, MarginLedgerAccountant, MarketTick,
    SimLatencyConfig, SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId as SpecId, InstrumentSpec,
    Margining, TickSchedule,
};
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::fee::FeeAsset;

type Account = DerivativeMarginAccount<TieredLinearIm>;

/// A quote per tick, bid price and size then ask price and size.
type Script = Vec<Option<(Decimal, Decimal, Decimal, Decimal)>>;

const FAR_LOW: Decimal = dec!(1);
const FAR_HIGH: Decimal = dec!(10000000);

/// A linear perpetual on BTC in USDT, `multiplier` BTC per contract, tick 1 and lot 1.
fn perpetual(multiplier: Decimal) -> InstrumentSpec {
    let asset = |code| AssetCode::new(code).unwrap();
    InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new("PERP").unwrap(),
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

fn id() -> InstrumentId {
    InstrumentId::new("PERP").unwrap()
}

fn usdt() -> Asset {
    Asset::new("USDT").unwrap()
}

fn ledger_for(spec: &InstrumentSpec, deposit: Decimal) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(usdt(), 8).unwrap();
    ledger
        .register_asset(Asset::new("BTC").unwrap(), 8)
        .unwrap();
    let scales = InstrumentScales {
        price_scale: 2,
        contract_scale: 0,
    };
    ledger
        .register_instrument(Instrument::from_spec(spec, scales).unwrap())
        .unwrap();
    if deposit > Decimal::ZERO {
        ledger
            .deposit(TransferId::new("seed").unwrap(), usdt(), deposit)
            .unwrap();
    }
    ledger
}

fn account_from(ledger: Ledger) -> Account {
    DerivativeMarginAccount::new(
        TieredLinearIm::linear_v1(usdt(), 8).unwrap(),
        RiskGate::new(ledger),
        id(),
        AccountId::new("a").unwrap(),
        VenueId::new("v").unwrap(),
        ReservationId::new("margin").unwrap(),
    )
    .unwrap()
}

fn account(spec: &InstrumentSpec, deposit: Decimal) -> Account {
    account_from(ledger_for(spec, deposit))
}

/// Quotes from a script, one per tick; `None` is silent and cancels what rests.
#[derive(Debug, Clone)]
struct Scripted(Script, usize);

impl BacktestStrategy for Scripted {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
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

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn cancels_when_silent(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.1 = 0;
    }
}

fn config(spec: &InstrumentSpec, fee_rate: Decimal) -> BacktestConfig {
    BacktestConfig {
        initial_capital: dec!(1000),
        fee_rate,
        fee_currency: FeeCurrency::Settlement { scale: 8 },
        tick_size: spec.price_ticks().bands()[0].tick,
        lot_size: spec.lot(),
        default_order_size: spec.lot(),
        ..BacktestConfig::default()
    }
    .with_record_trades(true)
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

/// A buy of `size` from tick 0, which tick 1's ask of 60000 reaches.
fn buy_script(size: Decimal) -> (Script, Vec<MarketTick>) {
    (
        vec![Some((dec!(60000), size, FAR_HIGH, size)), None, None],
        vec![
            tick(1000, dec!(60100)),
            tick(1001, dec!(59999)),
            tick(1002, dec!(60000)),
        ],
    )
}

fn gated(
    spec: &InstrumentSpec,
    account: &Account,
    fee: Decimal,
    script: Script,
    ticks: Vec<MarketTick>,
) -> BacktestResult {
    let mut engine = BacktestEngine::new(
        config(spec, fee),
        Scripted(script, 0),
        VecDataSource::new(ticks),
    )
    .with_instrument(spec.clone(), account.accountant())
    .unwrap()
    .with_pretrade_policy(account.policy());
    engine.run()
}

fn control(
    spec: &InstrumentSpec,
    fee: Decimal,
    script: Script,
    ticks: Vec<MarketTick>,
) -> BacktestResult {
    let ledger = ledger_for(spec, dec!(1000));
    let mut engine = BacktestEngine::new(
        config(spec, fee),
        Scripted(script, 0),
        VecDataSource::new(ticks),
    )
    .with_instrument(
        spec.clone(),
        MarginLedgerAccountant::new(ledger, id()).unwrap(),
    )
    .unwrap();
    engine.run()
}

fn request(order: &str, side: Action, quantity: Decimal) -> PretradeRequest {
    PretradeRequest::new(
        order,
        side,
        Some(dec!(60000)),
        quantity,
        OrderPurpose::Quote,
        0,
    )
}

fn fill(order: &str, execution: &str, quantity: Decimal) -> OrderEvent {
    OrderEvent::Fill {
        order_id: order.to_string(),
        execution_id: execution.to_string(),
        side: Action::Buy,
        price: dec!(60000),
        quantity,
        cumulative_quantity: quantity,
        fee: dec!(0),
        fee_asset: FeeAsset::Settlement,
        timestamp: 1,
    }
}

fn held(account: &Account) -> Decimal {
    account.with_gate(|gate| {
        let balance = gate.ledger().balance(&usdt()).unwrap();
        balance.total - balance.available
    })
}

#[test]
fn test_a_proposal_beyond_initial_margin_does_not_execute_against_a_control() {
    let spec = perpetual(dec!(0.01));
    // At 60100, 17 contracts of 0.01 are 10217 of notional, 1021.7 against 1000.
    let account = account(&spec, dec!(1000));
    let (script, ticks) = buy_script(dec!(17));
    let result = gated(&spec, &account, dec!(0), script.clone(), ticks.clone());
    assert!(result.trades.is_empty(), "{:?}", result.trades);
    assert_eq!(result.final_position, dec!(0));
    assert!(account.errors().is_empty(), "{:?}", account.errors());
    // With no gate the same proposal does execute, so the gate is what stopped it.
    let without = control(&spec, dec!(0), script, ticks);
    assert_eq!(without.final_position, dec!(17));
}

#[test]
fn test_the_notional_decides_not_price_times_quantity() {
    let spec = perpetual(dec!(0.01));
    // 16 contracts: price * quantity is 961600, far above 1000. The notional is 9616, so 961.6.
    let account = account(&spec, dec!(1000));
    let (script, ticks) = buy_script(dec!(16));
    let result = gated(&spec, &account, dec!(0), script, ticks);
    assert_eq!(result.final_position, dec!(16));
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    // The same 16 contracts of a multiplier ten times larger ask 9616, and are refused.
    let larger = perpetual(dec!(0.1));
    let refused = account_from(ledger_for(&larger, dec!(1000)));
    let (script, ticks) = buy_script(dec!(16));
    assert!(
        gated(&larger, &refused, dec!(0), script, ticks)
            .trades
            .is_empty()
    );
}

#[test]
fn test_the_requirement_is_held_so_that_cash_is_not_used_twice() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    // 10 contracts: 6000 of notional, 600 held of 1000.
    assert_eq!(
        policy.check(&request("1", Action::Buy, dec!(10))),
        PretradeDecision::Accepted
    );
    assert_eq!(held(&account), dec!(600));
    // 6 more on that side is 960, and 7 more would be 1020.
    assert!(
        policy
            .check(&request("2", Action::Buy, dec!(7)))
            .is_rejected()
    );
    assert_eq!(held(&account), dec!(600));
    assert_eq!(
        policy.check(&request("3", Action::Buy, dec!(6))),
        PretradeDecision::Accepted
    );
    assert_eq!(held(&account), dec!(960));
    // The other side nets within the worst case, so it asks nothing more up to the same size.
    assert_eq!(
        policy.check(&request("4", Action::Sell, dec!(16))),
        PretradeDecision::Accepted
    );
    assert_eq!(held(&account), dec!(960));
}

#[test]
fn test_a_requested_cancel_keeps_its_margin_and_a_fill_before_the_effect_stands() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert_eq!(
        policy.check(&request("1", Action::Buy, dec!(10))),
        PretradeDecision::Accepted
    );
    let order = "1".to_string();
    policy
        .on_order_event(&OrderEvent::Accepted {
            order_id: order.clone(),
        })
        .unwrap();
    policy
        .on_order_event(&OrderEvent::CancelRequested {
            order_id: order.clone(),
        })
        .unwrap();
    assert_eq!(held(&account), dec!(600));
    // Filled for 4 before the cancel takes effect: the position and the 6 left still ask 600.
    policy.on_order_event(&fill("1", "sim-0", dec!(4))).unwrap();
    assert_eq!(held(&account), dec!(600));
    let position = account.with_gate(|gate| gate.ledger().position(&id()).unwrap().quantity);
    assert_eq!(position, dec!(4));
    // Effective: what is left goes, and the position of 4 still asks 240.
    policy
        .on_order_event(&OrderEvent::Canceled { order_id: order })
        .unwrap();
    assert_eq!(held(&account), dec!(240));
    assert!(account.errors().is_empty(), "{:?}", account.errors());
}

#[test]
fn test_a_venue_rejection_releases_and_a_rejected_cancel_does_not() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(10)))
            .is_rejected()
    );
    policy
        .on_order_event(&OrderEvent::CancelRejected {
            order_id: "1".to_string(),
        })
        .unwrap();
    assert_eq!(held(&account), dec!(600));
    policy
        .on_order_event(&OrderEvent::VenueRejected {
            order_id: "1".to_string(),
        })
        .unwrap();
    assert_eq!(held(&account), dec!(0));
}

#[test]
fn test_a_mark_moves_the_hold_and_no_mark_refuses() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    // With no mark there is no notional.
    assert!(
        policy
            .check(&request("0", Action::Buy, dec!(1)))
            .is_rejected()
    );
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(10)))
            .is_rejected()
    );
    policy.on_market_data(dec!(50000), 1);
    assert_eq!(held(&account), dec!(500));
    assert_eq!(account.hold_shortfalls(), 0);
    // At 200000, 10 contracts ask 2000, beyond the cash: counted, and the hold stays.
    policy.on_market_data(dec!(200000), 2);
    assert_eq!(held(&account), dec!(500));
    assert_eq!(account.hold_shortfalls(), 1);
}

#[test]
fn test_the_books_match_an_unguarded_run_when_nothing_is_refused() {
    let spec = perpetual(dec!(0.01));
    let script = vec![
        Some((dec!(60000), dec!(5), FAR_HIGH, dec!(5))),
        None,
        Some((FAR_LOW, dec!(5), dec!(61000), dec!(5))),
        None,
        None,
    ];
    let ticks = vec![
        tick(1000, dec!(60100)),
        tick(1001, dec!(59999)),
        tick(1002, dec!(60500)),
        tick(1003, dec!(61001)),
        tick(1004, dec!(61000)),
    ];
    let account = account(&spec, dec!(1000));
    let with = gated(&spec, &account, dec!(0.0005), script.clone(), ticks.clone());
    let without = control(&spec, dec!(0.0005), script, ticks);
    assert!(with.accounting.is_consistent(), "{:?}", with.accounting);
    // Bought 5 at 60000, sold 5 at 61000: 50 of PnL, fees 1.5 and 1.525.
    assert_eq!(with.num_trades, 2);
    assert_eq!(with.total_pnl, dec!(50));
    assert_eq!(with.total_fees, dec!(3.025));
    assert_eq!(with.net_pnl, without.net_pnl);
    assert_eq!(with.equity_curve, without.equity_curve);
    assert_eq!(held(&account), dec!(0));
    let total = account.with_gate(|g| g.ledger().balance(&usdt()).unwrap().total);
    assert_eq!(total, dec!(1046.975));
    assert_eq!(
        account.with_gate(|g| g.ledger().journal().len()),
        with.num_trades as usize
    );
}

#[test]
fn test_a_second_run_starts_from_the_same_account() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let (script, ticks) = buy_script(dec!(5));
    let mut engine = BacktestEngine::new(
        config(&spec, dec!(0)),
        Scripted(script, 0),
        VecDataSource::new(ticks),
    )
    .with_instrument(spec.clone(), account.accountant())
    .unwrap()
    .with_pretrade_policy(account.policy());
    let first = engine.run();
    assert_eq!(first.final_position, dec!(5));
    engine.reset();
    assert_eq!(held(&account), dec!(0));
    assert!(account.with_gate(|g| g.ledger().journal().is_empty()));
    assert_eq!(engine.run().final_position, dec!(5));
}

#[test]
fn test_a_fault_stops_new_orders_until_a_reset() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(1)))
            .is_rejected()
    );
    let mut wrong = fill("1", "sim-0", dec!(1));
    if let OrderEvent::Fill { fee, fee_asset, .. } = &mut wrong {
        *fee = dec!(1);
        *fee_asset = FeeAsset::Base;
    }
    assert!(policy.on_order_event(&wrong).is_err());
    assert!(account.fault().is_some());
    assert_eq!(account.errors().len(), 1);
    assert!(
        policy
            .check(&request("2", Action::Buy, dec!(1)))
            .is_rejected()
    );
    policy.reset();
    assert!(account.fault().is_none());
    assert!(account.errors().is_empty());
}

#[test]
fn test_an_other_instrument_and_a_side_that_is_not_one_are_refused() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        policy
            .check(&request("1", Action::Other, dec!(1)))
            .is_rejected()
    );
    let mut other = request("2", Action::Buy, dec!(1));
    other.instrument = Some("OTHER".to_string());
    assert!(policy.check(&other).is_rejected());
    assert_eq!(held(&account), dec!(0));
    // One contract's marks only, so an engine of several contracts must not take it.
    assert!(policy.in_one_contract());
    assert!(!policy.in_contracts());
}

#[test]
fn test_new_refuses_a_spot_instrument_an_unknown_one_and_a_used_ledger() {
    let build = |ledger: Ledger, instrument: InstrumentId| {
        DerivativeMarginAccount::new(
            TieredLinearIm::linear_v1(usdt(), 8).unwrap(),
            RiskGate::new(ledger),
            instrument,
            AccountId::new("a").unwrap(),
            VenueId::new("v").unwrap(),
            ReservationId::new("margin").unwrap(),
        )
    };
    let spec = perpetual(dec!(0.01));
    assert!(
        build(
            ledger_for(&spec, dec!(0)),
            InstrumentId::new("NONE").unwrap()
        )
        .is_err()
    );
    let mut spot = ledger_for(&spec, dec!(0));
    let pair = InstrumentId::new("BTC-USDT").unwrap();
    spot.register_instrument(
        SpotInstrument::new(pair.clone(), Asset::new("BTC").unwrap(), usdt(), 2, 2).unwrap(),
    )
    .unwrap();
    assert!(build(spot, pair).is_err());
    let mut used = ledger_for(&spec, dec!(1000));
    used.book_execution_allowing_overdraft(
        &margin_rs::ledger::Execution::new(
            margin_rs::ExecutionId::new("sim-0").unwrap(),
            id(),
            Action::Buy,
            dec!(60000),
            dec!(1),
        )
        .with_executed_at(1),
        None,
    )
    .unwrap();
    assert!(build(used, id()).is_err());
}

#[test]
fn test_an_accountant_of_other_terms_is_refused() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let (script, ticks) = buy_script(dec!(1));
    let other = perpetual(dec!(0.1));
    assert!(
        BacktestEngine::new(
            config(&other, dec!(0)),
            Scripted(script, 0),
            VecDataSource::new(ticks),
        )
        .with_instrument(other.clone(), account.accountant())
        .is_err()
    );
}

#[test]
fn test_the_accountant_counts_fills_and_refuses_one_that_was_not_booked() {
    use market_maker_rs::backtest::{AccountingFill, FeeAsset as PostedFee, FillAccountant};
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    let mut accountant = account.accountant();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(4)))
            .is_rejected()
    );
    let posted = |execution_id| AccountingFill {
        execution_id,
        order_id: Default::default(),
        side: Action::Buy,
        price: dec!(60000),
        quantity: dec!(2),
        cumulative_quantity: dec!(2),
        fee: dec!(0),
        fee_asset: PostedFee::Settlement,
        timestamp: 1,
    };
    assert!(accountant.post_fill(&posted(0)).is_err());
    policy.on_order_event(&fill("1", "sim-0", dec!(2))).unwrap();
    accountant.post_fill(&posted(0)).unwrap();
    assert_eq!(accountant.snapshot().unwrap().fills_posted, 1);
    // The same execution again changes nothing.
    policy.on_order_event(&fill("1", "sim-0", dec!(2))).unwrap();
    assert_eq!(accountant.snapshot().unwrap().fills_posted, 1);
}

#[test]
fn test_a_fill_past_its_order_stands_and_is_recorded() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(1)))
            .is_rejected()
    );
    policy.on_order_event(&fill("1", "sim-0", dec!(3))).unwrap();
    let position = account.with_gate(|gate| gate.ledger().position(&id()).unwrap().quantity);
    assert_eq!(position, dec!(3));
    assert_eq!(account.errors().len(), 1, "{:?}", account.errors());
}

#[test]
fn test_a_fee_scale_above_the_settlement_asset_is_refused() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let (script, ticks) = buy_script(dec!(1));
    let mut too_fine = config(&spec, dec!(0));
    too_fine.fee_currency = FeeCurrency::Settlement { scale: 9 };
    assert!(
        BacktestEngine::new(too_fine, Scripted(script, 0), VecDataSource::new(ticks))
            .with_instrument(spec.clone(), account.accountant())
            .is_err()
    );
}

#[test]
fn test_an_inverse_perpetual_is_margined_on_its_contract_value() {
    use market_maker_rs::instruments::reference::deribit;
    let spec = deribit::btc_perpetual().unwrap();
    let btc = Asset::new("BTC").unwrap();
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(btc.clone(), 8).unwrap();
    ledger.register_asset(usdt(), 8).unwrap();
    let scales = InstrumentScales {
        price_scale: 2,
        contract_scale: 0,
    };
    ledger
        .register_instrument(Instrument::from_spec(&spec, scales).unwrap())
        .unwrap();
    ledger
        .deposit(TransferId::new("seed").unwrap(), btc.clone(), dec!(0.1))
        .unwrap();
    let account = DerivativeMarginAccount::new(
        margin_rs::rules::TieredInverseIm::inverse_v1(btc.clone(), 8).unwrap(),
        RiskGate::new(ledger),
        InstrumentId::new(spec.id().as_str()).unwrap(),
        AccountId::new("a").unwrap(),
        VenueId::new("v").unwrap(),
        ReservationId::new("margin").unwrap(),
    )
    .unwrap();
    let mut policy = account.policy();
    policy.on_market_data(dec!(50000), 0);
    // 2500 contracts of 10 USD at 50000 are 0.5 BTC of notional, 0.05 of 0.1 held.
    assert!(
        !policy
            .check(&request("1", Action::Buy, dec!(2500)))
            .is_rejected()
    );
    let held = account.with_gate(|gate| {
        let balance = gate.ledger().balance(&btc).unwrap();
        balance.total - balance.available
    });
    assert_eq!(held, dec!(0.05));
    // 5001 in all would ask just over the 0.1.
    assert!(
        policy
            .check(&request("2", Action::Buy, dec!(2501)))
            .is_rejected()
    );
    assert!(
        !policy
            .check(&request("3", Action::Buy, dec!(2500)))
            .is_rejected()
    );
}

#[test]
fn test_in_the_engine_a_fill_before_the_cancel_takes_effect_stands_and_later_none_does() {
    let spec = perpetual(dec!(0.01));
    let run = |cross_at: u64| {
        let account = account(&spec, dec!(1000));
        // Quoted at 1000, silent from 1100, with the cancel taking effect at 1150.
        let script = vec![Some((dec!(60000), dec!(5), FAR_HIGH, dec!(5))), None, None];
        let mut ticks = vec![
            tick(1000, dec!(60100)),
            tick(1100, dec!(60100)),
            tick(cross_at, dec!(59999)),
        ];
        ticks.push(tick(cross_at + 1, dec!(60100)));
        let mut engine = BacktestEngine::new(
            config(&spec, dec!(0)),
            Scripted(script, 0),
            VecDataSource::new(ticks),
        )
        .with_instrument(spec.clone(), account.accountant())
        .unwrap()
        .with_pretrade_policy(account.policy())
        .with_latency(SimLatencyConfig::new().with_cancel_ms(50))
        .unwrap();
        let result = engine.run();
        assert!(account.errors().is_empty(), "{:?}", account.errors());
        (result.final_position, held(&account))
    };
    // The fill stands, and the position of 5 at 60100 still asks 300.5.
    assert_eq!(run(1149), (dec!(5), dec!(300.5)));
    // After the effect there is no order, and nothing is held.
    assert_eq!(run(1150), (dec!(0), dec!(0)));
}

#[test]
fn test_new_refuses_a_model_in_another_currency() {
    let spec = perpetual(dec!(0.01));
    let built = DerivativeMarginAccount::new(
        TieredLinearIm::linear_v1(Asset::new("BTC").unwrap(), 8).unwrap(),
        RiskGate::new(ledger_for(&spec, dec!(1000))),
        id(),
        AccountId::new("a").unwrap(),
        VenueId::new("v").unwrap(),
        ReservationId::new("margin").unwrap(),
    );
    assert!(built.is_err());
}

#[test]
fn test_a_refusal_for_margin_says_so_with_the_figures() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    let PretradeDecision::Rejected { reason } = policy.check(&request("1", Action::Buy, dec!(17)))
    else {
        panic!("not refused");
    };
    assert!(reason.contains("InsufficientMargin"), "{reason}");
    assert!(reason.contains("1020"), "{reason}");
    assert_eq!(account.hold_shortfalls(), 0);
}

#[test]
fn test_no_unit_of_cash_is_held_by_margin_and_a_spot_reservation() {
    let spec = perpetual(dec!(0.01));
    let mut ledger = ledger_for(&spec, dec!(1000));
    // 500 of the cash is reserved for something else before the account starts.
    ledger
        .reserve(
            margin_rs::ReservationId::new("other").unwrap(),
            usdt(),
            dec!(500),
        )
        .unwrap();
    let account = account_from(ledger);
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    // 500 is left: 8 contracts ask 480, and 9 would ask 540.
    assert!(
        policy
            .check(&request("1", Action::Buy, dec!(9)))
            .is_rejected()
    );
    assert!(
        !policy
            .check(&request("2", Action::Buy, dec!(8)))
            .is_rejected()
    );
}

#[test]
fn test_a_hold_error_that_is_not_about_the_cash_stops_new_orders() {
    let spec = perpetual(dec!(0.01));
    let account = account(&spec, dec!(1000));
    let mut policy = account.policy();
    policy.on_market_data(dec!(60000), 0);
    assert!(
        !policy
            .check(&request("0", Action::Buy, dec!(1)))
            .is_rejected()
    );
    // With an order open, a mark the service refuses: no shortfall, and a fault.
    policy.on_market_data(dec!(0), 1);
    assert_eq!(account.hold_shortfalls(), 0);
    assert!(account.fault().is_some());
    assert_eq!(account.errors().len(), 1);
    let PretradeDecision::Rejected { reason } = policy.check(&request("1", Action::Buy, dec!(1)))
    else {
        panic!("not refused");
    };
    assert!(reason.contains("policy fault"), "{reason}");
}
