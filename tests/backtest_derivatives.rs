//! Derivative contracts in the backtest engine and `MarginLedgerAccountant`
//! (#191): hand cases, the shared PnL fixtures, and every refused configuration.
//!
//! The expected values are literals from the contract terms, or exact fractions
//! of the shared fixtures, never recomputed with the code under test.

#![cfg(all(feature = "margin", feature = "instruments"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use financial_types::Side;
use financial_types::instrument::fixtures::{self, Case, Expected, Operation};
use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::{
    DerivativeSettlement, Execution, ExecutionCashflow, ExpirySettlement, FundingPayment, Ledger,
    LedgerConfig,
};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::engine::{BacktestConfig, BacktestStrategy, FeeCurrency};
use market_maker_rs::backtest::{
    AccountingFill, AccountingSnapshot, BacktestEngine, BacktestResult, FeeAsset, FillAccountant,
    MarginLedgerAccountant, MarketTick, SimOrderId, SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::instruments::reference::deribit;
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId as SpecId, InstrumentSpec,
    Margining, TickBand, TickSchedule,
};
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::prelude::OptionStyle;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::LimitsPolicy;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};

const EXPIRY_MS: u64 = 1_795_766_400_000;

fn scales() -> InstrumentScales {
    InstrumentScales {
        price_scale: 8,
        contract_scale: 8,
    }
}

/// A ledger with BTC and USDT at 8 decimals and `spec` at scales of 8.
fn ledger_for(spec: &InstrumentSpec) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for asset in ["BTC", "USDT"] {
        ledger
            .register_asset(Asset::new(asset).unwrap(), 8)
            .unwrap();
    }
    ledger
        .register_instrument(Instrument::from_spec(spec, scales()).unwrap())
        .unwrap();
    ledger
}

fn accountant_for(spec: &InstrumentSpec) -> MarginLedgerAccountant {
    MarginLedgerAccountant::new(
        ledger_for(spec),
        InstrumentId::new(spec.id().as_str()).unwrap(),
    )
    .unwrap()
}

/// A linear future or perpetual on BTC settled in USDT, with `multiplier` BTC per
/// contract, tick 1 and lot 1.
fn linear(id: &str, multiplier: Decimal, expiry_ms: Option<u64>) -> InstrumentSpec {
    let asset = |code| AssetCode::new(code).unwrap();
    InstrumentSpec::new(InstrumentDraft {
        id: SpecId::new(id).unwrap(),
        kind: if expiry_ms.is_some() {
            ContractKind::Future
        } else {
            ContractKind::Perpetual
        },
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(multiplier, asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms,
        option: None,
    })
    .unwrap()
}

/// A fill, as the engine posts it.
fn posted(
    id: u64,
    side: Action,
    price: Decimal,
    quantity: Decimal,
    fee: Decimal,
) -> AccountingFill {
    AccountingFill {
        execution_id: id,
        order_id: SimOrderId::default(),
        side,
        price,
        quantity,
        cumulative_quantity: quantity,
        fee,
        fee_asset: FeeAsset::Settlement,
        timestamp: id,
    }
}

/// Quotes from a script, one per tick; `None` leaves the resting orders.
#[derive(Debug, Clone)]
struct Scripted(Vec<Option<(Decimal, Decimal, Decimal, Decimal)>>, usize);

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

    fn reset(&mut self) {
        self.1 = 0;
    }
}

/// Far enough from any price in these tests never to fill.
const FAR_LOW: Decimal = dec!(1);
const FAR_HIGH: Decimal = dec!(10000000);

fn buy_at(price: Decimal, size: Decimal) -> Option<(Decimal, Decimal, Decimal, Decimal)> {
    Some((price, size, FAR_HIGH, size))
}

fn sell_at(price: Decimal, size: Decimal) -> Option<(Decimal, Decimal, Decimal, Decimal)> {
    Some((FAR_LOW, size, price, size))
}

fn tick(timestamp: u64, bid: Decimal, ask: Decimal) -> MarketTick {
    MarketTick::new(timestamp, bid, dec!(100000), ask, dec!(100000))
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
}

/// A tick whose touch fills an order of `side` at `price`.
fn crossing(timestamp: u64, side: Side, price: Decimal, half: Decimal) -> MarketTick {
    match side {
        Side::Long => tick(timestamp, price - half, price),
        Side::Short => tick(timestamp, price, price + half),
    }
}

/// A run opening on `side` at `entry` from tick 0, filled on tick 1, closing at
/// `exit` from tick 2 when `close` holds, filled on tick 3, ending at the mid
/// `last`. Sizes are in contracts.
#[allow(clippy::too_many_arguments)]
fn run(
    spec: &InstrumentSpec,
    fee_rate: Decimal,
    side: Side,
    size: Decimal,
    entry: Decimal,
    close: bool,
    exit: Decimal,
    last: Decimal,
) -> BacktestResult {
    let half = spec.price_ticks().bands()[0].tick;
    let far = Some((FAR_LOW, spec.lot(), FAR_HIGH, spec.lot()));
    let (open_order, close_order, close_side) = match side {
        Side::Long => (buy_at(entry, size), sell_at(exit, size), Side::Short),
        Side::Short => (sell_at(entry, size), buy_at(exit, size), Side::Long),
    };
    let script = vec![
        open_order,
        far,
        if close { close_order } else { far },
        far,
        far,
    ];
    let ticks = vec![
        tick(1000, dec!(50000), dec!(70000)),
        crossing(1001, side, entry, half),
        tick(1002, entry - half, entry + half),
        if close {
            crossing(1003, close_side, exit, half)
        } else {
            tick(1003, entry - half, entry + half)
        },
        tick(1004, last - half, last + half),
    ];
    let mut engine = BacktestEngine::new(
        config(spec, fee_rate),
        Scripted(script, 0),
        VecDataSource::new(ticks),
    )
    .with_instrument(spec.clone(), accountant_for(spec))
    .unwrap();
    engine.run()
}

#[test]
fn linear_contract_is_in_usdt_with_its_multiplier() {
    // 5 contracts of 0.01 BTC: 3000 USDT at 60000 and 3050 USDT at 61000, so a
    // move of 1000 is 50 USDT; fees of 0.05% are 1.5 and 1.525 USDT.
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let result = run(
        &spec,
        dec!(0.0005),
        Side::Long,
        dec!(5),
        dec!(60000),
        true,
        dec!(61000),
        dec!(62000),
    );
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert_eq!(result.num_trades, 2);
    assert_eq!(result.final_position, dec!(0));
    assert_eq!(result.total_pnl, dec!(50));
    assert_eq!(result.total_fees, dec!(3.025));
    assert_eq!(result.net_pnl, dec!(46.975));
    // Equity: 1000 + 50 - 3.025, whatever the mark, once flat.
    assert_eq!(result.equity_curve.last().unwrap().1, dec!(1046.975));
}

#[test]
fn inverse_contract_is_in_btc() {
    // 1000 contracts of 10 USD: 10000/60000 and 10000/61000 BTC, each leg rounded
    // at 8 decimals, 0.16666667 and 0.16393443. Fees of 0.05%: 1/12000 and
    // 1/12200 BTC, 0.00008333 and 0.00008197.
    let spec = deribit::btc_perpetual().unwrap();
    let result = run(
        &spec,
        dec!(0.0005),
        Side::Long,
        dec!(1000),
        dec!(60000),
        true,
        dec!(61000),
        dec!(62000),
    );
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert_eq!(result.total_pnl, dec!(0.00273224));
    assert_eq!(result.total_fees, dec!(0.0001653));
    assert_eq!(result.net_pnl, dec!(0.00256694));
    assert_eq!(result.equity_curve.last().unwrap().1, dec!(1000.00256694));
}

#[test]
fn an_open_inverse_position_is_marked_in_btc_with_the_sign_of_the_side() {
    let spec = deribit::btc_perpetual().unwrap();
    // Long at 60000, marked at 61000: 0.16666667 - 0.16393443.
    let long = run(
        &spec,
        dec!(0),
        Side::Long,
        dec!(1000),
        dec!(60000),
        false,
        dec!(60000),
        dec!(61000),
    );
    assert!(long.accounting.is_consistent(), "{:?}", long.accounting);
    assert_eq!(long.final_position, dec!(1000));
    assert_eq!(long.total_pnl, dec!(0.00273224));
    // Short at 60000, marked at 61000: the price rose, so it lost.
    let short = run(
        &spec,
        dec!(0),
        Side::Short,
        dec!(1000),
        dec!(60000),
        false,
        dec!(60000),
        dec!(61000),
    );
    assert!(short.accounting.is_consistent(), "{:?}", short.accounting);
    assert_eq!(short.final_position, dec!(-1000));
    assert_eq!(short.total_pnl, dec!(-0.00273224));
}

#[test]
fn a_rebate_keeps_its_sign_and_reduces_the_fees() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let result = run(
        &spec,
        dec!(-0.0001),
        Side::Long,
        dec!(5),
        dec!(60000),
        false,
        dec!(60000),
        dec!(60000),
    );
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    // 3000 USDT x -0.01%.
    assert_eq!(result.total_fees, dec!(-0.3));
}

#[test]
fn adding_a_partial_close_and_a_flip_through_zero_realize_exactly() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut a = accountant_for(&spec);
    let zero = Decimal::ZERO;
    // Buy 5 at 60000, add 5 at 61000: long 10 at an average of 60500.
    a.post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), zero))
        .unwrap();
    a.post_fill(&posted(2, Action::Buy, dec!(61000), dec!(5), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(10));
    assert_eq!(s.average_entry_price, Some(dec!(60500)));
    assert_eq!(s.cash_flow, dec!(0));
    // 10 x 0.01 x (62000 - 60500) = 150 USDT unrealized.
    assert_eq!(a.unrealized_at(dec!(62000)).unwrap().unwrap(), dec!(150));
    // Sell 4 at 62000: 4 x 0.01 x 1500 = 60 realized; 6 left at 60500.
    a.post_fill(&posted(3, Action::Sell, dec!(62000), dec!(4), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(6));
    assert_eq!(s.average_entry_price, Some(dec!(60500)));
    assert_eq!(s.cash_flow, dec!(60));
    assert_eq!(a.unrealized_at(dec!(62000)).unwrap().unwrap(), dec!(90));
    // Sell 10 at 60000: closes 6 for 6 x 0.01 x -500 = -30, and opens 4 short at
    // 60000.
    a.post_fill(&posted(4, Action::Sell, dec!(60000), dec!(10), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(-4));
    assert_eq!(s.average_entry_price, Some(dec!(60000)));
    assert_eq!(s.cash_flow, dec!(30));
    assert_eq!(s.fills_posted, 4);
    // Short 4 from 60000 at 59000: +40.
    assert_eq!(a.unrealized_at(dec!(59000)).unwrap().unwrap(), dec!(40));
}

#[test]
fn an_inverse_contract_adds_closes_and_flips_on_its_recorded_rests() {
    // 10 USD a contract, so 1000 contracts cost 10000 / price BTC, kept at 8
    // decimals: 0.16666667 at 60000 and 0.16393443 at 61000.
    let spec = deribit::btc_perpetual().unwrap();
    let mut a = accountant_for(&spec);
    let zero = Decimal::ZERO;
    a.post_fill(&posted(1, Action::Buy, dec!(60000), dec!(1000), zero))
        .unwrap();
    a.post_fill(&posted(2, Action::Buy, dec!(61000), dec!(1000), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(2000));
    // The harmonic mean: 20000 / 0.3306011.
    assert_eq!(s.average_entry_price.unwrap().round_dp(2), dec!(60495.87));
    assert_eq!(s.cash_flow, dec!(0));
    // A quarter of the cost, 0.082650275, less 5000 / 62000 = 0.08064516.
    a.post_fill(&posted(3, Action::Sell, dec!(62000), dec!(500), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(1500));
    assert_eq!(s.cash_flow, dec!(0.00200512));
    // The remaining cost, 0.24795082, less 15000 / 60000 = 0.25 is -0.00204918;
    // the other 500 contracts open short at 60000, for 0.08333333.
    a.post_fill(&posted(4, Action::Sell, dec!(60000), dec!(2000), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(-500));
    // 5000 / 0.08333333, from the rounded rest.
    assert_eq!(s.average_entry_price.unwrap().round_dp(2), dec!(60000));
    assert_eq!(s.cash_flow, dec!(-0.00004406));
    // 5000 / 59000 = 0.08474576, less the 0.08333333 received short, is 0.00141243.
    assert_eq!(
        a.unrealized_at(dec!(59000)).unwrap().unwrap(),
        dec!(0.00141243)
    );
    a.post_fill(&posted(5, Action::Buy, dec!(59000), dec!(500), zero))
        .unwrap();
    let s = a.snapshot().unwrap();
    assert_eq!(s.position, dec!(0));
    assert_eq!(s.cash_flow, dec!(0.00136837));
    assert_eq!(s.fills_posted, 5);
    assert_eq!(a.unrealized_at(dec!(59000)).unwrap().unwrap(), dec!(0));
}

#[test]
fn marking_at_the_exit_and_then_closing_there_keeps_the_total() {
    for (spec, size) in [
        (linear("FUT", dec!(0.01), Some(EXPIRY_MS)), dec!(5)),
        (deribit::btc_perpetual().unwrap(), dec!(1000)),
    ] {
        let mut a = accountant_for(&spec);
        a.post_fill(&posted(1, Action::Buy, dec!(60000), size, dec!(0)))
            .unwrap();
        let before = a.unrealized_at(dec!(61000)).unwrap().unwrap();
        a.post_fill(&posted(2, Action::Sell, dec!(61000), size, dec!(0)))
            .unwrap();
        let s = a.snapshot().unwrap();
        // Both round the same two values once, so nothing is lost or counted twice.
        assert_eq!(s.cash_flow, before);
        assert_eq!(a.unrealized_at(dec!(61000)).unwrap().unwrap(), dec!(0));
    }
}

#[test]
fn a_ledger_that_starts_open_is_marked_at_its_entry_before_any_tick() {
    // Not at the zero mark a flat start uses, which an open position would refuse.
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut seeded = accountant_for(&spec);
    seeded
        .post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0)))
        .unwrap();
    let engine = BacktestEngine::new(
        config(&spec, dec!(0)),
        Scripted(Vec::new(), 0),
        VecDataSource::new(vec![tick(1000, dec!(60999), dec!(61001))]),
    )
    .with_instrument(spec.clone(), seeded);
    let mut engine = engine.unwrap();
    assert_eq!(engine.fault_count(), 0);
    let (position, pnl) = engine.get_state();
    assert_eq!(position.quantity, dec!(5));
    assert_eq!(position.avg_entry_price, dec!(60000));
    assert_eq!(pnl.unrealized, dec!(0));
    // The tick's mid: 5 x 0.01 x 1000.
    let result = engine.run();
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert_eq!(result.total_pnl, dec!(50));
    assert_eq!(result.net_pnl, dec!(50));
}

#[test]
fn a_run_never_reuses_an_execution_the_ledger_started_with() {
    // Seeded at another price, a reused id would be refused as a conflict. Seeded
    // with the very fill of the run, it would be a silent duplicate.
    for (seed_price, average, total) in [
        (dec!(60000), dec!(60500), dec!(50)),
        (dec!(61000), dec!(61000), dec!(0)),
    ] {
        let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
        let mut seeded = accountant_for(&spec);
        seeded
            .post_fill(&AccountingFill {
                timestamp: 1001,
                ..posted(1, Action::Buy, seed_price, dec!(5), dec!(0))
            })
            .unwrap();
        let far = Some((FAR_LOW, spec.lot(), FAR_HIGH, spec.lot()));
        let mut engine = BacktestEngine::new(
            config(&spec, dec!(0)),
            Scripted(vec![buy_at(dec!(61000), dec!(5)), far, far], 0),
            VecDataSource::new(vec![
                tick(1000, dec!(50000), dec!(70000)),
                crossing(1001, Side::Long, dec!(61000), dec!(1)),
                tick(1002, dec!(60999), dec!(61001)),
            ]),
        )
        .with_instrument(spec.clone(), seeded)
        .unwrap();
        let first = engine.run();
        assert!(first.accounting.is_consistent(), "{:?}", first.accounting);
        assert!(first.accounting.refused_executions.is_empty());
        let (position, _) = engine.get_state();
        assert_eq!(position.quantity, dec!(10));
        assert_eq!(position.avg_entry_price, average);
        assert_eq!(first.total_pnl, total);
        // A run after reset starts from the same handover state.
        engine.reset();
        let (position, pnl) = engine.get_state();
        assert_eq!(position.quantity, dec!(5));
        assert_eq!(position.avg_entry_price, seed_price);
        assert_eq!(pnl.realized, dec!(0));
        let second = engine.run();
        assert_eq!(second.accounting, first.accounting);
        assert_eq!(second.total_pnl, first.total_pnl);
        assert_eq!(second.net_pnl, first.net_pnl);
        assert_eq!(second.total_fees, first.total_fees);
        assert_eq!(second.num_trades, first.num_trades);
        assert_eq!(second.final_position, first.final_position);
        assert_eq!(engine.get_state().0.quantity, dec!(10));
    }
}

/// The id of a first fill posted to `ledger`, on the instrument `traded`, which
/// must book.
fn id_of_the_last_fill(ledger: Ledger, traded: &InstrumentId) -> String {
    let mut a = MarginLedgerAccountant::new(ledger, traded.clone()).unwrap();
    let before = a.snapshot().unwrap().position;
    a.post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0)))
        .unwrap();
    assert_eq!(a.snapshot().unwrap().position, before + dec!(5));
    let journal = a.ledger().journal();
    journal[journal.len() - 1].execution_id.as_str().to_string()
}

#[test]
fn execution_ids_continue_after_every_journal_that_shares_them() {
    // Each journal alone, so that one left out fails its own case.
    let perp = linear("PERP", dec!(0.01), None);
    let perp_id = InstrumentId::new("PERP").unwrap();
    let id = |text: &str| ExecutionId::new(text).unwrap();

    let mut funded = ledger_for(&perp);
    funded
        .book_funding(&FundingPayment {
            id: id("sim-5"),
            instrument: perp_id.clone(),
            amount: dec!(1),
            as_of_ms: 1,
        })
        .unwrap();
    assert_eq!(id_of_the_last_fill(funded, &perp_id), "sim-6");

    let mut settled = ledger_for(&perp);
    settled
        .book_settlement(&DerivativeSettlement {
            id: id("sim-7"),
            // A settlement is for a derivative held outside the ledger.
            instrument: InstrumentId::new("OUTSIDE").unwrap(),
            asset: Asset::new("USDT").unwrap(),
            realized_pnl: dec!(1),
            fee: dec!(0),
        })
        .unwrap();
    assert_eq!(id_of_the_last_fill(settled, &perp_id), "sim-8");

    // A flat future that expired, beside the instrument traded.
    let future = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut expired = ledger_for(&perp);
    expired
        .register_instrument(Instrument::from_spec(&future, scales()).unwrap())
        .unwrap();
    expired
        .settle_expiry(&ExpirySettlement {
            id: id("sim-9"),
            instrument: InstrumentId::new("FUT").unwrap(),
            price: dec!(60000),
            as_of_ms: EXPIRY_MS,
        })
        .unwrap();
    assert_eq!(id_of_the_last_fill(expired, &perp_id), "sim-10");
}

/// A ledger on the perpetual `PERP`, with one identifier in each journal that
/// shares them: an execution, a funding payment, a settlement and an expiry.
fn ledger_with_all_journals(execution: &str, funding: &str, settled: &str, expiry: &str) -> Ledger {
    let id = |text: &str| ExecutionId::new(text).unwrap();
    let perp_id = InstrumentId::new("PERP").unwrap();
    let mut ledger = ledger_for(&linear("PERP", dec!(0.01), None));
    ledger
        .register_instrument(
            Instrument::from_spec(&linear("FUT", dec!(0.01), Some(EXPIRY_MS)), scales()).unwrap(),
        )
        .unwrap();
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                id(execution),
                perp_id.clone(),
                Action::Buy,
                dec!(60000),
                dec!(1),
            )
            .with_executed_at(1),
            None,
        )
        .unwrap();
    ledger
        .book_funding(&FundingPayment {
            id: id(funding),
            instrument: perp_id,
            amount: dec!(1),
            as_of_ms: 1,
        })
        .unwrap();
    ledger
        .book_settlement(&DerivativeSettlement {
            id: id(settled),
            instrument: InstrumentId::new("OUTSIDE").unwrap(),
            asset: Asset::new("USDT").unwrap(),
            realized_pnl: dec!(1),
            fee: dec!(0),
        })
        .unwrap();
    ledger
        .settle_expiry(&ExpirySettlement {
            id: id(expiry),
            instrument: InstrumentId::new("FUT").unwrap(),
            price: dec!(60000),
            as_of_ms: EXPIRY_MS,
        })
        .unwrap();
    ledger
}

#[test]
fn execution_ids_continue_after_the_highest_of_all_the_journals() {
    let perp_id = InstrumentId::new("PERP").unwrap();
    // The highest in each journal in turn, so that no order of reading can pass.
    for ids in [
        ["sim-50", "sim-2", "sim-3", "sim-4"],
        ["sim-2", "sim-50", "sim-3", "sim-4"],
        ["sim-2", "sim-3", "sim-50", "sim-4"],
        ["sim-2", "sim-3", "sim-4", "sim-50"],
    ] {
        let ledger = ledger_with_all_journals(ids[0], ids[1], ids[2], ids[3]);
        let mut a = MarginLedgerAccountant::new(ledger, perp_id.clone()).unwrap();
        // Through the handover as well as through the constructor.
        a.begin_from_current_state();
        let start = a.ledger().snapshot();
        let fill = posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0));
        a.post_fill(&fill).unwrap();
        // The same fill again is the same execution.
        a.post_fill(&fill).unwrap();
        let journal = a.ledger().journal();
        assert_eq!(journal.len(), 2, "{ids:?}");
        assert_eq!(journal[1].execution_id.as_str(), "sim-51", "{ids:?}");
        assert_eq!(a.snapshot().unwrap().position, dec!(6));

        // A reset returns to the same start and the offset does not accumulate.
        a.reset();
        assert_eq!(a.ledger().snapshot(), start);
        a.post_fill(&fill).unwrap();
        assert_eq!(a.ledger().journal()[1].execution_id.as_str(), "sim-51");
        // Handed over again, the numbers continue after the run's own.
        a.begin_from_current_state();
        a.post_fill(&fill).unwrap();
        assert_eq!(a.ledger().journal()[2].execution_id.as_str(), "sim-52");
    }
}

#[test]
fn other_identifiers_do_not_move_the_execution_ids() {
    let perp_id = InstrumentId::new("PERP").unwrap();
    for ids in [
        ["sim-x", "simx-9", "SIM-9", "9"],
        ["sim-", "sim-12x", "xsim-9", "sim-9.5"],
        // Above u64: it does not parse, and the sequence never reaches it.
        [
            "sim-18446744073709551616",
            "other",
            "sim-99999999999999999999999",
            "sim--9",
        ],
    ] {
        let ledger = ledger_with_all_journals(ids[0], ids[1], ids[2], ids[3]);
        assert_eq!(id_of_the_last_fill(ledger, &perp_id), "sim-1", "{ids:?}");
    }
}

#[test]
fn execution_ids_past_the_largest_number_are_an_error_with_the_ledger_intact() {
    let perp_id = InstrumentId::new("PERP").unwrap();
    let last = format!("sim-{}", u64::MAX);
    let ledger = ledger_with_all_journals("sim-1", &last, "sim-2", "sim-3");
    let mut a = MarginLedgerAccountant::new(ledger, perp_id).unwrap();
    let start = a.ledger().snapshot();
    let before = a.snapshot().unwrap();
    let error = a
        .post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0)))
        .unwrap_err();
    assert!(matches!(error, MMError::NumericalError(_)), "{error:?}");
    assert_eq!(a.ledger().snapshot(), start);
    assert_eq!(a.snapshot().unwrap(), before);
}

#[test]
fn a_replay_onto_a_ledger_wrapped_again_books_new_fills() {
    // The execution id is a key for one accountant, not across ledgers.
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let fill = posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0));
    let mut a = accountant_for(&spec);
    a.post_fill(&fill).unwrap();
    a.post_fill(&fill).unwrap();
    assert_eq!(a.snapshot().unwrap().position, dec!(5));
    let mut again =
        MarginLedgerAccountant::new(a.ledger().clone(), InstrumentId::new("FUT").unwrap()).unwrap();
    again.post_fill(&fill).unwrap();
    assert_eq!(again.snapshot().unwrap().position, dec!(10));
}

/// What a snapshot of `traded` must say, from a walk over the journal in the test
/// and not through the accountant's sums.
fn walked(a: &MarginLedgerAccountant, traded: &str, base: &str, quote: &str) -> AccountingSnapshot {
    let traded = InstrumentId::new(traded).unwrap();
    let base = Asset::new(base).unwrap();
    let quote = Asset::new(quote).unwrap();
    let derivative = base == quote;
    let mut walk = AccountingSnapshot::default();
    for entry in a
        .ledger()
        .journal()
        .iter()
        .filter(|entry| entry.instrument == traded)
    {
        if !derivative {
            match entry.side {
                Action::Buy => walk.cash_flow -= entry.notional,
                _ => walk.cash_flow += entry.notional,
            }
        }
        if let Some(fee) = &entry.fee {
            if fee.asset == quote {
                walk.fees += fee.amount;
            } else if fee.asset == base {
                walk.base_fees += fee.amount;
            }
        }
        walk.fills_posted += 1;
    }
    let position = a.ledger().position(&traded).unwrap();
    walk.position = position.quantity - walk.base_fees;
    walk.average_entry_price = position.average_price();
    if derivative {
        walk.cash_flow = position.realized_pnl;
    }
    walk
}

/// A ledger with two spot pairs of BTC against USDT, and BNB, which is neither base
/// nor quote of them.
fn spot_ledger(asset_scale: u32) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for asset in ["BTC", "USDT", "BNB"] {
        ledger
            .register_asset(Asset::new(asset).unwrap(), asset_scale)
            .unwrap();
    }
    for id in ["BTC-USDT", "BTC-OTHER"] {
        ledger
            .register_instrument(
                margin_rs::SpotInstrument::new(
                    InstrumentId::new(id).unwrap(),
                    Asset::new("BTC").unwrap(),
                    Asset::new("USDT").unwrap(),
                    2,
                    2,
                )
                .unwrap(),
            )
            .unwrap();
    }
    ledger
}

/// One execution, booked directly in the ledger.
fn book(
    ledger: &mut Ledger,
    id: &str,
    traded: &str,
    side: Action,
    price: Decimal,
    quantity: Decimal,
    fee: Option<(&str, Decimal)>,
) {
    let mut execution = Execution::new(
        ExecutionId::new(id).unwrap(),
        InstrumentId::new(traded).unwrap(),
        side,
        price,
        quantity,
    )
    .with_executed_at(1);
    if let Some((asset, amount)) = fee {
        execution = execution.with_fee(Asset::new(asset).unwrap(), amount);
    }
    ledger
        .book_execution_allowing_overdraft(&execution, None)
        .unwrap();
}

fn spot_fill(
    id: u64,
    side: Action,
    price: Decimal,
    fee: Decimal,
    asset: FeeAsset,
) -> AccountingFill {
    AccountingFill {
        fee_asset: asset,
        ..posted(id, side, price, dec!(1), fee)
    }
}

#[test]
fn a_spot_snapshot_says_what_a_walk_over_the_journal_says() {
    // Seeded with another instrument's entries, a fee in a third asset and a base
    // fee, none of which the sums may take wrongly.
    let mut ledger = spot_ledger(8);
    book(
        &mut ledger,
        "x-1",
        "BTC-OTHER",
        Action::Buy,
        dec!(7),
        dec!(3),
        Some(("USDT", dec!(9))),
    );
    book(
        &mut ledger,
        "x-2",
        "BTC-OTHER",
        Action::Buy,
        dec!(7),
        dec!(3),
        Some(("BTC", dec!(9))),
    );
    book(
        &mut ledger,
        "x-3",
        "BTC-USDT",
        Action::Buy,
        dec!(100),
        dec!(2),
        Some(("BNB", dec!(4))),
    );
    book(
        &mut ledger,
        "x-4",
        "BTC-USDT",
        Action::Buy,
        dec!(100),
        dec!(1),
        Some(("BTC", dec!(0.25))),
    );
    book(
        &mut ledger,
        "x-5",
        "BTC-USDT",
        Action::Sell,
        dec!(110),
        dec!(1),
        Some(("USDT", dec!(0.5))),
    );
    let mut a =
        MarginLedgerAccountant::new(ledger, InstrumentId::new("BTC-USDT").unwrap()).unwrap();
    let check = |a: &MarginLedgerAccountant| {
        assert_eq!(a.snapshot().unwrap(), walked(a, "BTC-USDT", "BTC", "USDT"));
    };
    check(&a);
    // By hand: -200 - 100 + 110, fees of 0.5 in USDT and 0.25 in BTC, 2 + 1 - 1 - 0.25.
    let seeded = a.snapshot().unwrap();
    assert_eq!(seeded.cash_flow, dec!(-190));
    assert_eq!(seeded.fees, dec!(0.5));
    assert_eq!(seeded.base_fees, dec!(0.25));
    assert_eq!(seeded.position, dec!(1.75));
    assert_eq!(seeded.fills_posted, 3);

    let quote_fee = spot_fill(1, Action::Buy, dec!(120), dec!(0.1), FeeAsset::Quote);
    let base_fee = spot_fill(2, Action::Buy, dec!(90), dec!(0.01), FeeAsset::Base);
    let rebate = spot_fill(3, Action::Sell, dec!(130), dec!(-0.2), FeeAsset::Quote);
    let base_rebate = spot_fill(4, Action::Sell, dec!(130), dec!(-0.02), FeeAsset::Base);
    for fill in [&quote_fee, &base_fee, &rebate, &base_rebate] {
        a.post_fill(fill).unwrap();
        check(&a);
        // Replayed, it is the same execution.
        a.post_fill(fill).unwrap();
        check(&a);
    }
    // -190 - 120 - 90 + 130 + 130 = -140; fees 0.5 + 0.1 - 0.2; base 0.25 + 0.01 - 0.02.
    let run = a.snapshot().unwrap();
    assert_eq!(run.cash_flow, dec!(-140));
    assert_eq!(run.fees, dec!(0.4));
    assert_eq!(run.base_fees, dec!(0.24));
    assert_eq!(run.position, dec!(1.76));
    assert_eq!(run.fills_posted, 7);

    // Refused posts: other terms under the same id, and a price off the scale.
    let other_terms = spot_fill(1, Action::Buy, dec!(121), dec!(0.1), FeeAsset::Quote);
    assert!(a.post_fill(&other_terms).is_err());
    assert_eq!(a.snapshot().unwrap(), run);
    let off_scale = spot_fill(5, Action::Buy, dec!(1.001), dec!(0.1), FeeAsset::Base);
    assert!(a.post_fill(&off_scale).is_err());
    assert_eq!(a.snapshot().unwrap(), run);
    check(&a);

    a.reset();
    assert_eq!(a.snapshot().unwrap(), seeded);
    check(&a);
    a.post_fill(&quote_fee).unwrap();
    a.begin_from_current_state();
    let handed = a.snapshot().unwrap();
    check(&a);
    a.post_fill(&base_fee).unwrap();
    check(&a);
    a.reset();
    assert_eq!(a.snapshot().unwrap(), handed);
    check(&a);
}

#[test]
fn a_derivative_snapshot_says_what_a_walk_over_the_journal_says() {
    for (spec, id, base, size, settlement) in [
        (
            linear("PERP", dec!(0.01), None),
            "PERP",
            "USDT",
            dec!(5),
            "USDT",
        ),
        (
            deribit::btc_perpetual().unwrap(),
            "",
            "BTC",
            dec!(1000),
            "BTC",
        ),
    ] {
        let name = if id.is_empty() {
            spec.id().as_str().to_string()
        } else {
            id.to_string()
        };
        let traded = InstrumentId::new(&name).unwrap();
        // Seeded with its own fill and fee, funding, and an expired future beside it.
        let mut ledger = ledger_for(&spec);
        ledger
            .register_instrument(
                Instrument::from_spec(&linear("FUT", dec!(0.01), Some(EXPIRY_MS)), scales())
                    .unwrap(),
            )
            .unwrap();
        book(
            &mut ledger,
            "x-1",
            &name,
            Action::Buy,
            dec!(60000),
            size,
            Some((settlement, dec!(0.0001))),
        );
        ledger
            .book_funding(&FundingPayment {
                id: ExecutionId::new("x-2").unwrap(),
                instrument: traded.clone(),
                amount: dec!(1),
                as_of_ms: 1,
            })
            .unwrap();
        ledger
            .settle_expiry(&ExpirySettlement {
                id: ExecutionId::new("x-3").unwrap(),
                instrument: InstrumentId::new("FUT").unwrap(),
                price: dec!(60000),
                as_of_ms: EXPIRY_MS,
            })
            .unwrap();
        let mut a = MarginLedgerAccountant::new(ledger, traded).unwrap();
        let check = |a: &MarginLedgerAccountant| {
            let snapshot = a.snapshot().unwrap();
            assert_eq!(snapshot, walked(a, &name, base, base), "{name}");
            // A settlement asset fee is a fee, and never also a base fee.
            assert_eq!(snapshot.base_fees, dec!(0), "{name}");
        };
        check(&a);
        let seeded = a.snapshot().unwrap();
        assert_eq!(seeded.fees, dec!(0.0001));
        assert_eq!(seeded.fills_posted, 1);

        let open = posted(1, Action::Buy, dec!(61000), size, dec!(0.0002));
        let close = posted(2, Action::Sell, dec!(62000), size, dec!(-0.0001));
        for fill in [&open, &close] {
            a.post_fill(fill).unwrap();
            check(&a);
            a.post_fill(fill).unwrap();
            check(&a);
        }
        let run = a.snapshot().unwrap();
        assert_eq!(run.fees, dec!(0.0002));
        assert_eq!(run.fills_posted, 3);
        assert_ne!(run.cash_flow, seeded.cash_flow);

        let other_terms = posted(2, Action::Sell, dec!(62001), size, dec!(-0.0001));
        assert!(a.post_fill(&other_terms).is_err());
        assert_eq!(a.snapshot().unwrap(), run);

        a.reset();
        assert_eq!(a.snapshot().unwrap(), seeded);
        a.post_fill(&open).unwrap();
        a.begin_from_current_state();
        let handed = a.snapshot().unwrap();
        a.post_fill(&close).unwrap();
        check(&a);
        a.reset();
        assert_eq!(a.snapshot().unwrap(), handed);
        check(&a);
    }
}

const HUGE: Decimal = dec!(50000000000000000000000000000);

#[test]
fn a_cash_flow_that_overflows_stays_an_error_until_a_reset() {
    // Another pair's purchase leaves the balance low enough to take it, so that
    // only the sum of this instrument overflows, and the ledger books every fill.
    let mut ledger = spot_ledger(2);
    book(
        &mut ledger,
        "x-1",
        "BTC-OTHER",
        Action::Buy,
        HUGE,
        dec!(1),
        None,
    );
    let wrapped = ledger.clone();
    let mut a =
        MarginLedgerAccountant::new(ledger, InstrumentId::new("BTC-USDT").unwrap()).unwrap();
    let start = a.snapshot().unwrap();
    let post = |a: &mut MarginLedgerAccountant, id, side, price| {
        a.post_fill(&posted(id, side, price, dec!(1), dec!(0)))
    };
    post(&mut a, 1, Action::Sell, HUGE).unwrap();
    post(&mut a, 2, Action::Buy, dec!(1)).unwrap();
    assert_eq!(a.snapshot().unwrap().cash_flow, HUGE - dec!(1));
    // Booked, and past the bound by the sum alone.
    post(&mut a, 3, Action::Sell, HUGE).unwrap();
    let overflowed = |a: &MarginLedgerAccountant, what: &str| {
        assert_eq!(
            a.snapshot().unwrap_err(),
            MMError::NumericalError(format!("{what} overflows"))
        );
    };
    overflowed(&a, "ledger cash flow");
    overflowed(&a, "ledger cash flow");
    assert_eq!(a.ledger().journal().len(), 4);
    // A later fill would bring the sum back in range: it stays an error, as a walk
    // that stops at the first overflow would.
    post(&mut a, 4, Action::Buy, HUGE).unwrap();
    overflowed(&a, "ledger cash flow");

    // Wrapping such a ledger is not refused; its snapshot fails the same way, and
    // a reset returns to that state.
    let mut rewrapped =
        MarginLedgerAccountant::new(a.ledger().clone(), InstrumentId::new("BTC-USDT").unwrap())
            .unwrap();
    overflowed(&rewrapped, "ledger cash flow");
    rewrapped.reset();
    overflowed(&rewrapped, "ledger cash flow");

    // A reset to a clean start clears it.
    a.reset();
    assert_eq!(a.snapshot().unwrap(), start);
    assert_eq!(a.ledger().snapshot(), wrapped.snapshot());
    // A handover while it stands keeps it, on the next reset too.
    post(&mut a, 1, Action::Sell, HUGE).unwrap();
    post(&mut a, 2, Action::Buy, dec!(1)).unwrap();
    post(&mut a, 3, Action::Sell, HUGE).unwrap();
    a.begin_from_current_state();
    overflowed(&a, "ledger cash flow");
    a.reset();
    overflowed(&a, "ledger cash flow");
}

#[test]
fn fees_that_overflow_are_an_error_with_the_fill_booked() {
    // A rebate on another pair keeps the ledger's own total of fees in range.
    let mut ledger = spot_ledger(2);
    book(
        &mut ledger,
        "x-1",
        "BTC-OTHER",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("USDT", -HUGE)),
    );
    let mut a =
        MarginLedgerAccountant::new(ledger, InstrumentId::new("BTC-USDT").unwrap()).unwrap();
    let start = a.snapshot().unwrap();
    let fill = |id| spot_fill(id, Action::Buy, dec!(1), HUGE, FeeAsset::Quote);
    a.post_fill(&fill(1)).unwrap();
    assert_eq!(a.snapshot().unwrap().fees, HUGE);
    a.post_fill(&fill(2)).unwrap();
    assert_eq!(a.ledger().journal().len(), 3);
    assert_eq!(
        a.snapshot().unwrap_err(),
        MMError::NumericalError("ledger fees overflows".to_string())
    );
    // A rebate that would bring it back does not.
    a.post_fill(&spot_fill(3, Action::Buy, dec!(1), -HUGE, FeeAsset::Quote))
        .unwrap();
    assert!(a.snapshot().is_err());
    a.reset();
    assert_eq!(a.snapshot().unwrap(), start);

    // A later overflow of another sum does not replace the first.
    a.post_fill(&fill(1)).unwrap();
    a.post_fill(&fill(2)).unwrap();
    for (id, side, price) in [
        (3, Action::Sell, dec!(1)),
        (4, Action::Sell, dec!(1)),
        (5, Action::Sell, HUGE),
        (6, Action::Buy, dec!(1)),
        (7, Action::Sell, HUGE),
    ] {
        a.post_fill(&posted(id, side, price, dec!(1), dec!(0)))
            .unwrap();
    }
    assert_eq!(
        a.snapshot().unwrap_err(),
        MMError::NumericalError("ledger fees overflows".to_string())
    );
    a.reset();
    assert_eq!(a.snapshot().unwrap(), start);
}

#[test]
fn a_derivative_with_fees_that_overflow_is_an_error_with_the_fill_booked() {
    let spec = linear("PERP", dec!(0.01), None);
    let other = linear("OTHER", dec!(0.01), None);
    let traded = InstrumentId::new("PERP").unwrap();
    let mut seeded = ledger_for(&spec);
    seeded
        .register_instrument(Instrument::from_spec(&other, scales()).unwrap())
        .unwrap();
    // Flat again, and with a rebate that keeps the ledger's total of fees in range.
    book(
        &mut seeded,
        "x-1",
        "OTHER",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("USDT", -HUGE)),
    );
    book(
        &mut seeded,
        "x-2",
        "OTHER",
        Action::Sell,
        dec!(1),
        dec!(1),
        None,
    );
    let wrapped = seeded.clone();
    let mut a = MarginLedgerAccountant::new(seeded, traded.clone()).unwrap();
    let start = a.snapshot().unwrap();
    a.post_fill(&posted(1, Action::Buy, dec!(1), dec!(1), HUGE))
        .unwrap();
    assert_eq!(a.snapshot().unwrap().fees, HUGE);
    a.post_fill(&posted(2, Action::Sell, dec!(1), dec!(1), HUGE))
        .unwrap();
    assert_eq!(a.ledger().journal().len(), 4);
    assert_eq!(
        a.snapshot().unwrap_err(),
        MMError::NumericalError("ledger fees overflows".to_string())
    );
    a.reset();
    assert_eq!(a.snapshot().unwrap(), start);
    assert_eq!(a.ledger().snapshot(), wrapped.snapshot());

    // A ledger already past it is refused when wrapped, as it was.
    let mut past = wrapped;
    book(
        &mut past,
        "x-3",
        "PERP",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("USDT", HUGE)),
    );
    book(
        &mut past,
        "x-4",
        "PERP",
        Action::Sell,
        dec!(1),
        dec!(1),
        Some(("USDT", HUGE)),
    );
    assert!(matches!(
        MarginLedgerAccountant::new(past, traded),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn base_fees_that_would_overflow_are_refused_but_a_replay_is_not() {
    let mut ledger = spot_ledger(2);
    book(
        &mut ledger,
        "x-1",
        "BTC-OTHER",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("BTC", -HUGE)),
    );
    let mut a =
        MarginLedgerAccountant::new(ledger, InstrumentId::new("BTC-USDT").unwrap()).unwrap();
    let fill = |id, fee| spot_fill(id, Action::Buy, dec!(1), fee, FeeAsset::Base);
    a.post_fill(&fill(1, HUGE)).unwrap();
    let booked = a.snapshot().unwrap();
    assert_eq!(booked.base_fees, HUGE);
    let ledger_state = a.ledger().snapshot();

    // Near the bound, the identical fill is a duplicate and not an overflow.
    a.post_fill(&fill(1, HUGE)).unwrap();
    assert_eq!(a.snapshot().unwrap(), booked);
    assert_eq!(a.ledger().snapshot(), ledger_state);
    // The same id with other terms is a conflict, not an overflow.
    let error = a.post_fill(&fill(1, HUGE - dec!(1))).unwrap_err();
    assert!(
        matches!(error, MMError::InvalidPositionUpdate(_)),
        "{error:?}"
    );
    assert_eq!(a.snapshot().unwrap(), booked);
    assert_eq!(a.ledger().snapshot(), ledger_state);
    // A new fill past the bound is refused before anything is booked.
    assert_eq!(
        a.post_fill(&fill(2, HUGE)).unwrap_err(),
        MMError::NumericalError("ledger base fees overflows".to_string())
    );
    assert_eq!(a.snapshot().unwrap(), booked);
    assert_eq!(a.ledger().snapshot(), ledger_state);

    // A ledger already past it is refused when wrapped, as it was.
    let mut past = spot_ledger(2);
    book(
        &mut past,
        "x-1",
        "BTC-OTHER",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("BTC", -HUGE)),
    );
    book(
        &mut past,
        "x-2",
        "BTC-USDT",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("BTC", HUGE)),
    );
    book(
        &mut past,
        "x-3",
        "BTC-USDT",
        Action::Buy,
        dec!(1),
        dec!(1),
        Some(("BTC", HUGE)),
    );
    assert!(matches!(
        MarginLedgerAccountant::new(past, InstrumentId::new("BTC-USDT").unwrap()),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn another_open_position_in_the_ledger_is_refused_when_wrapped() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let other = linear("OTHER", dec!(0.01), Some(EXPIRY_MS));
    let main_id = InstrumentId::new("FUT").unwrap();
    let other_id = InstrumentId::new("OTHER").unwrap();
    let mut ledger = ledger_for(&spec);
    ledger
        .register_instrument(Instrument::from_spec(&other, scales()).unwrap())
        .unwrap();
    let mut on_other = MarginLedgerAccountant::new(ledger, other_id).unwrap();
    on_other
        .post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0)))
        .unwrap();
    let error =
        MarginLedgerAccountant::new(on_other.ledger().clone(), main_id.clone()).unwrap_err();
    assert!(
        matches!(&error, MMError::InvalidConfiguration(reason) if reason.contains("OTHER")),
        "{error:?}"
    );
    // Flat again, it needs no mark and leaks nothing into this instrument.
    on_other
        .post_fill(&posted(2, Action::Sell, dec!(61000), dec!(5), dec!(0)))
        .unwrap();
    let mut on_main = MarginLedgerAccountant::new(on_other.ledger().clone(), main_id).unwrap();
    assert_eq!(on_main.unrealized_at(dec!(61000)), Some(Ok(dec!(0))));
    on_main
        .post_fill(&posted(1, Action::Buy, dec!(60000), dec!(5), dec!(0)))
        .unwrap();
    assert_eq!(on_main.unrealized_at(dec!(61000)), Some(Ok(dec!(50))));
    let snapshot = on_main.snapshot().unwrap();
    assert_eq!(snapshot.position, dec!(5));
    assert_eq!(snapshot.cash_flow, dec!(0));
    assert_eq!(snapshot.fills_posted, 1);
}

#[test]
fn reset_returns_to_the_exact_initial_state() {
    let spec = deribit::btc_perpetual().unwrap();
    let mut engine = BacktestEngine::new(
        config(&spec, dec!(0.0005)),
        Scripted(vec![buy_at(dec!(60000), dec!(1000)); 4], 0),
        VecDataSource::new(vec![
            tick(1000, dec!(50000), dec!(70000)),
            tick(1001, dec!(59999.5), dec!(60000)),
            tick(1002, dec!(59999.5), dec!(60000)),
        ]),
    )
    .with_instrument(spec.clone(), accountant_for(&spec))
    .unwrap();
    let first = engine.run();
    assert_ne!(first.total_fees, dec!(0));
    engine.reset();
    let snapshot = engine.accounting_snapshot().unwrap().unwrap();
    assert_eq!(snapshot, AccountingSnapshot::default());
    let (position, pnl) = engine.get_state();
    assert_eq!(position.quantity, dec!(0));
    assert_eq!(
        (pnl.realized, pnl.unrealized, pnl.total),
        (dec!(0), dec!(0), dec!(0))
    );
}

#[test]
fn a_fill_just_before_the_expiry_books_and_one_at_it_is_refused() {
    let at = |timestamp: u64| {
        let spec = linear("FUT", dec!(0.01), Some(1003));
        let mut engine = BacktestEngine::new(
            config(&spec, dec!(0)),
            Scripted(vec![buy_at(dec!(60000), dec!(5))], 0),
            VecDataSource::new(vec![
                tick(1000, dec!(50000), dec!(70000)),
                tick(timestamp, dec!(59999), dec!(60000)),
            ]),
        )
        .with_instrument(spec.clone(), accountant_for(&spec))
        .unwrap();
        engine.run()
    };
    let before = at(1002);
    assert!(before.accounting.is_consistent(), "{:?}", before.accounting);
    assert_eq!(before.final_position, dec!(5));
    let on = at(1003);
    assert_eq!(on.accounting.refused_executions, vec![1]);
    assert!(!on.accounting.is_consistent());
    assert_eq!(on.final_position, dec!(0));
}

#[test]
fn a_fill_at_a_fee_of_another_asset_is_refused_and_changes_nothing() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut a = accountant_for(&spec);
    let before = a.snapshot().unwrap();
    let mut fill = posted(1, Action::Buy, dec!(60000), dec!(5), dec!(1));
    for asset in [
        FeeAsset::Quote,
        FeeAsset::Base,
        FeeAsset::Other("BNB".to_string()),
    ] {
        fill.fee_asset = asset;
        assert!(a.post_fill(&fill).is_err());
        assert_eq!(a.snapshot().unwrap(), before);
    }
    // A zero fee has no asset to get wrong.
    fill.fee = dec!(0);
    fill.fee_asset = FeeAsset::Quote;
    a.post_fill(&fill).unwrap();
    assert_eq!(a.snapshot().unwrap().position, dec!(5));
}

#[test]
fn a_spot_accountant_refuses_a_settlement_fee_and_is_not_a_derivative() {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for asset in ["BTC", "USDT"] {
        ledger
            .register_asset(Asset::new(asset).unwrap(), 8)
            .unwrap();
    }
    let id = InstrumentId::new("BTC-USDT").unwrap();
    ledger
        .register_instrument(
            margin_rs::SpotInstrument::new(
                id.clone(),
                Asset::new("BTC").unwrap(),
                Asset::new("USDT").unwrap(),
                8,
                8,
            )
            .unwrap(),
        )
        .unwrap();
    let mut spot = MarginLedgerAccountant::new(ledger, id).unwrap();
    assert!(!spot.requires_instrument());
    assert!(spot.unrealized_at(dec!(100)).is_none());
    let fill = posted(1, Action::Buy, dec!(100), dec!(1), dec!(1));
    assert!(spot.post_fill(&fill).is_err());
    assert_eq!(spot.snapshot().unwrap(), AccountingSnapshot::default());
}

/// A run with no with_instrument, or a later change, for the refused setups.
fn engine_without_instrument(config: BacktestConfig) -> BacktestEngine<Scripted, VecDataSource> {
    BacktestEngine::new(
        config,
        Scripted(vec![buy_at(dec!(60000), dec!(5))], 0),
        VecDataSource::new(vec![
            tick(1000, dec!(50000), dec!(70000)),
            tick(1001, dec!(59999), dec!(60000)),
        ]),
    )
}

fn assert_refused(result: &BacktestResult, reason: &str) {
    let refused = result.accounting.setup_refused.as_deref().unwrap();
    assert!(refused.contains(reason), "{refused}");
    assert!(!result.accounting.is_consistent());
    assert_eq!(result.num_ticks, 0);
    assert_eq!(result.num_trades, 0);
    assert_eq!(result.total_pnl, dec!(0));
}

fn with_instrument_error(
    spec: &InstrumentSpec,
    config: BacktestConfig,
    accountant: impl FillAccountant
    + Send
    + Sync
    + std::panic::UnwindSafe
    + std::panic::RefUnwindSafe
    + 'static,
) -> String {
    match engine_without_instrument(config).with_instrument(spec.clone(), accountant) {
        Ok(_) => panic!("accepted"),
        Err(MMError::InvalidConfiguration(message)) => message,
        Err(other) => panic!("{other:?}"),
    }
}

#[test]
fn with_instrument_refuses_what_it_cannot_book() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let good = config(&spec, dec!(0));

    // Fee currency, lot and tick.
    let quote_fee = BacktestConfig {
        fee_currency: FeeCurrency::Quote,
        ..good.clone()
    };
    let e = with_instrument_error(&spec, quote_fee, accountant_for(&spec));
    assert!(e.contains("Settlement"), "{e}");
    let lot = BacktestConfig {
        lot_size: dec!(2),
        ..good.clone()
    };
    let e = with_instrument_error(&spec, lot, accountant_for(&spec));
    assert!(e.contains("lot"), "{e}");
    let tick_size = BacktestConfig {
        tick_size: dec!(2),
        ..good.clone()
    };
    let e = with_instrument_error(&spec, tick_size, accountant_for(&spec));
    assert!(e.contains("tick"), "{e}");
    // A fee scale above the settlement asset's 8.
    let fine = BacktestConfig {
        fee_currency: FeeCurrency::Settlement { scale: 9 },
        ..good.clone()
    };
    let e = with_instrument_error(&spec, fine, accountant_for(&spec));
    assert!(e.contains("fee scale"), "{e}");

    // Spot and options are not derivatives of this run.
    let option = deribit::btc_option(
        SpecId::new("BTC-27NOV26-100000-C").unwrap(),
        EXPIRY_MS,
        dec!(100000),
        OptionStyle::Call,
    )
    .unwrap();
    let e = with_instrument_error(&option, config(&spec, dec!(0)), accountant_for(&spec));
    assert!(e.contains("not a future or a perpetual"), "{e}");
}

#[test]
fn the_accountant_must_book_the_same_terms_not_only_the_same_id() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let good = config(&spec, dec!(0));
    // The same id with another contract size, another expiry, or a perpetual.
    for other in [
        linear("FUT", dec!(0.1), Some(EXPIRY_MS)),
        linear("FUT", dec!(0.01), Some(EXPIRY_MS + 1)),
        linear("FUT", dec!(0.01), None),
    ] {
        let e = with_instrument_error(&spec, good.clone(), accountant_for(&other));
        assert!(e.contains("the terms differ"), "{e}");
    }
    // Another instrument.
    let e = with_instrument_error(
        &spec,
        good.clone(),
        accountant_for(&linear("OTHER", dec!(0.01), Some(EXPIRY_MS))),
    );
    assert!(e.contains("the terms differ"), "{e}");
}

/// A custom accountant, with what it declares chosen by the test.
#[derive(Debug, Default)]
struct Custom {
    snapshot: AccountingSnapshot,
    opts_in: bool,
    valuation: Option<Result<Decimal, MMError>>,
}

impl FillAccountant for Custom {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        self.snapshot.position += fill.signed_quantity()?;
        self.snapshot.fills_posted += 1;
        Ok(())
    }

    fn snapshot(&self) -> MMResult<AccountingSnapshot> {
        Ok(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = AccountingSnapshot::default();
    }

    fn unrealized_at(&self, _mark: Decimal) -> Option<MMResult<Decimal>> {
        self.valuation.clone()
    }

    fn check_instrument(&self, spec: &InstrumentSpec, _fee_scale: u32) -> MMResult<()> {
        if self.opts_in {
            Ok(())
        } else {
            Err(MMError::InvalidConfiguration(format!(
                "cannot book {}",
                spec.id().as_str()
            )))
        }
    }
}

/// A custom accountant that keeps every default.
#[derive(Debug, Default)]
struct Plain(AccountingSnapshot);

impl FillAccountant for Plain {
    fn post_fill(&mut self, _fill: &AccountingFill) -> MMResult<()> {
        Ok(())
    }

    fn snapshot(&self) -> MMResult<AccountingSnapshot> {
        Ok(self.0.clone())
    }

    fn reset(&mut self) {}
}

#[test]
fn a_custom_accountant_must_opt_in() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let e = with_instrument_error(&spec, config(&spec, dec!(0)), Plain::default());
    assert!(e.contains("cannot book instrument FUT"), "{e}");
}

#[test]
fn an_accountant_that_cannot_value_never_gets_position_times_mark() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    // Opted in, but with no valuation, or one that fails from the start.
    for valuation in [
        None,
        Some(Err(MMError::NumericalError("no mark".to_string()))),
    ] {
        let accountant = Custom {
            opts_in: true,
            valuation,
            ..Custom::default()
        };
        let mut engine = engine_without_instrument(config(&spec, dec!(0)))
            .with_instrument(spec.clone(), accountant)
            .unwrap();
        let result = engine.run();
        assert_eq!(result.total_pnl, dec!(0));
        // Every mark is counted, without an invented execution id per tick.
        assert!(result.accounting.valuation_failures >= 2);
        assert!(result.accounting.snapshot_failures.is_empty());
        assert!(result.accounting.refused_executions.is_empty());
        assert!(!result.accounting.is_consistent());
        // A fault stops new orders, as any other does.
        assert_eq!(result.num_trades, 0);
        assert!(engine.fault_count() >= 2);
    }
}

/// Values at 7 for the first marks, then fails.
#[derive(Debug, Default)]
struct FailsLater {
    marks: std::sync::atomic::AtomicU64,
    good_marks: u64,
}

impl FillAccountant for FailsLater {
    fn post_fill(&mut self, _fill: &AccountingFill) -> MMResult<()> {
        Ok(())
    }

    fn snapshot(&self) -> MMResult<AccountingSnapshot> {
        Ok(AccountingSnapshot {
            position: dec!(5),
            ..AccountingSnapshot::default()
        })
    }

    fn reset(&mut self) {
        *self.marks.get_mut() = 0;
    }

    fn unrealized_at(&self, _mark: Decimal) -> Option<MMResult<Decimal>> {
        let mark = self.marks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Some(if mark < self.good_marks {
            Ok(dec!(7))
        } else {
            Err(MMError::NumericalError("no mark".to_string()))
        })
    }

    fn check_instrument(&self, _spec: &InstrumentSpec, _fee_scale: u32) -> MMResult<()> {
        Ok(())
    }
}

#[test]
fn a_valuation_that_fails_keeps_the_previous_numbers_and_says_so() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let accountant = FailsLater {
        good_marks: 2,
        ..FailsLater::default()
    };
    let mut engine = engine_without_instrument(config(&spec, dec!(0)))
        .with_instrument(spec.clone(), accountant)
        .unwrap();
    let result = engine.run();
    // The last good value, not 5 x the mark of the failed ones. The second tick
    // marks twice: after its fill and at its end.
    assert_eq!(result.total_pnl, dec!(7));
    assert_eq!(result.accounting.valuation_failures, 2);
    assert!(!result.accounting.is_consistent());
    // Counted afresh after a reset, not added to the last run's.
    engine.reset();
    let again = engine.run();
    assert_eq!(again.accounting.valuation_failures, 2);
}

#[test]
fn a_refusal_lasts_only_as_long_as_its_reason() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut engine = engine_without_instrument(config(&spec, dec!(0)));
    assert_refused(&engine.run(), "use with_instrument");
    engine.reset();
    assert_refused(&engine.run(), "use with_instrument");
    let mut engine = engine
        .with_instrument(spec.clone(), accountant_for(&spec))
        .unwrap();
    let result = engine.run();
    assert_eq!(result.accounting.setup_refused, None);
    assert_eq!(result.num_ticks, 2);
}

#[test]
fn a_tick_schedule_with_several_bands_is_refused() {
    let flat = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut draft = InstrumentDraft::from(flat.clone());
    draft.price_ticks = TickSchedule::new(vec![
        TickBand {
            up_to: Some(dec!(1000)),
            tick: dec!(1),
        },
        TickBand {
            up_to: None,
            tick: dec!(5),
        },
    ])
    .unwrap();
    let bands = InstrumentSpec::new(draft).unwrap();
    // A contract whose first band matches the configured tick, so that only the
    // number of bands is wrong.
    assert_eq!(bands.price_ticks().bands()[0].tick, dec!(1));
    let error = engine_without_instrument(config(&bands, dec!(0)))
        .with_instrument(bands.clone(), accountant_for(&flat))
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(&error, MMError::InvalidConfiguration(reason) if reason.contains("tick")),
        "{error:?}"
    );
}

#[test]
fn a_fee_that_is_not_exact_drops_the_fill_and_says_so() {
    // 0.01 x 60001 = 600.01, and 600.01 x 1e-28 needs thirty places.
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let mut engine = BacktestEngine::new(
        config(&spec, Decimal::new(1, 28)),
        Scripted(vec![buy_at(dec!(60001), dec!(1))], 0),
        VecDataSource::new(vec![
            tick(1000, dec!(50000), dec!(70000)),
            tick(1001, dec!(60000), dec!(60001)),
        ]),
    )
    .with_instrument(spec.clone(), accountant_for(&spec))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.num_trades, 0);
    assert_eq!(result.final_position, dec!(0));
    assert_eq!(result.accounting.dropped_fills, 1);
    assert!(!result.accounting.is_consistent());
    assert_eq!(engine.fault_count(), 1);
    assert!(matches!(
        engine.faults().first(),
        Some(MMError::NumericalError(reason)) if reason.contains("not exact")
    ));
    engine.reset();
    assert_eq!(engine.run().accounting.dropped_fills, 1);
}

#[test]
fn trades_and_reports_carry_the_fee_in_the_settlement_asset() {
    // 1000 contracts of 10 USD at 60000 are 1/6 BTC; 0.05% of it is 0.00008333.
    let spec = deribit::btc_perpetual().unwrap();
    let half = spec.price_ticks().bands()[0].tick;
    let mut engine = BacktestEngine::new(
        config(&spec, dec!(0.0005)),
        Scripted(vec![buy_at(dec!(60000), dec!(1000))], 0),
        VecDataSource::new(vec![
            tick(1000, dec!(50000), dec!(70000)),
            crossing(1001, Side::Long, dec!(60000), half),
        ]),
    )
    .with_instrument(spec.clone(), accountant_for(&spec))
    .unwrap();
    let result = engine.run();
    assert_eq!(result.total_fees, dec!(0.00008333));
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].fee, dec!(0.00008333));
    let fees: Vec<Decimal> = engine
        .execution_reports()
        .iter()
        .filter(|report| !report.last_quantity.is_zero())
        .map(|report| report.fee)
        .collect();
    assert_eq!(fees, [dec!(0.00008333)]);
}

#[test]
fn a_valuation_uses_what_the_accountant_says() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let accountant = Custom {
        opts_in: true,
        valuation: Some(Ok(dec!(7))),
        ..Custom::default()
    };
    let mut engine = engine_without_instrument(config(&spec, dec!(0)))
        .with_instrument(spec.clone(), accountant)
        .unwrap();
    let result = engine.run();
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert_eq!(result.total_pnl, dec!(7));
}

#[test]
fn a_later_builder_call_gets_the_run_refused_before_anything_happens() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    let built = || {
        engine_without_instrument(config(&spec, dec!(0)))
            .with_instrument(spec.clone(), accountant_for(&spec))
            .unwrap()
    };

    let mut policy = built().with_pretrade_policy(LimitsPolicy::new(
        RiskLimits::new(dec!(100), Decimal::MAX, dec!(0)).unwrap(),
    ));
    assert_refused(&policy.run(), "pretrade policy");

    let other = accountant_for(&linear("FUT", dec!(0.1), Some(EXPIRY_MS)));
    let mut replaced = built().with_accountant(other);
    assert_refused(&replaced.run(), "the terms differ");

    let mut plain = built().with_accountant(Plain::default());
    assert_refused(&plain.run(), "cannot book");
    assert!(plain.fault_count() > 0);
}

#[test]
fn settlement_fees_and_derivative_accountants_need_the_contract() {
    let spec = linear("FUT", dec!(0.01), Some(EXPIRY_MS));
    // A settlement fee, on the built in bookkeeping.
    let mut built_in = engine_without_instrument(config(&spec, dec!(0.001)));
    assert_refused(&built_in.run(), "use with_instrument");

    // A derivative accountant, with a fee currency that would be wrong for it.
    let quote = BacktestConfig {
        fee_currency: FeeCurrency::Quote,
        ..config(&spec, dec!(0.001))
    };
    let mut accountant_only =
        engine_without_instrument(quote).with_accountant(accountant_for(&spec));
    assert_refused(&accountant_only.run(), "use with_instrument");

    assert!(matches!(
        FeeCurrency::Settlement { scale: 8 }.charge(dec!(1), dec!(1), dec!(0.1)),
        Err(MMError::InvalidConfiguration(_))
    ));
}

/// An exact fraction, every operation checked.
#[derive(Debug, Clone, Copy)]
struct Q(i128, i128);

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl Q {
    fn new(n: i128, d: i128) -> Self {
        let g = gcd(n, d).max(1) * d.signum();
        Self(n / g, d / g)
    }

    fn dec(value: Decimal) -> Self {
        Self::new(value.mantissa(), 10_i128.pow(value.scale()))
    }

    fn ratio(num: Decimal, den: Decimal) -> Self {
        let (n, d) = (Self::dec(num), Self::dec(den));
        Self::new(n.0.checked_mul(d.1).unwrap(), n.1.checked_mul(d.0).unwrap())
    }

    fn add(self, o: Self) -> Self {
        let n = self.0.checked_mul(o.1).unwrap() + o.0.checked_mul(self.1).unwrap();
        Self::new(n, self.1.checked_mul(o.1).unwrap())
    }

    fn neg(self) -> Self {
        Self(-self.0, self.1)
    }
}

impl PartialEq for Q {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0 && self.1 == o.1
    }
}

fn rest(fill: &ExecutionCashflow) -> Q {
    let residual = fill.derivative.as_ref().unwrap().residual.as_ref().unwrap();
    Q::ratio(residual.numerator, residual.denominator)
}

#[test]
fn the_shared_pnl_fixtures_agree_through_the_ledger_and_the_whole_engine() {
    let cases: Vec<Case> = fixtures::cases()
        .unwrap()
        .into_iter()
        .filter(|case| matches!(case.operation, Operation::RealizedPnl { .. }))
        .collect();
    assert_eq!(cases.len(), 5, "a new PnL fixture must be run here");
    let (mut linear_cases, mut inverse_cases) = (0, 0);
    for case in &cases {
        let Operation::RealizedPnl {
            side,
            contracts,
            entry,
            exit,
        } = case.operation
        else {
            unreachable!();
        };
        let Expected::Amount { exact, .. } = &case.expected else {
            panic!("{}: expected an amount", case.name);
        };
        let exact = Q::ratio(exact.ratio.num(), exact.ratio.den());
        let (open, close) = match side {
            Side::Long => (Action::Buy, Action::Sell),
            Side::Short => (Action::Sell, Action::Buy),
        };
        // Through the accountant, whose ledger keeps what each leg rounded.
        let mut ledger = ledger_for(&case.spec);
        let id = InstrumentId::new(case.spec.id().as_str()).unwrap();
        let mut accountant = MarginLedgerAccountant::new(ledger.clone(), id.clone()).unwrap();
        accountant
            .post_fill(&posted(1, open, entry, contracts, dec!(0)))
            .unwrap();
        accountant
            .post_fill(&posted(2, close, exit, contracts, dec!(0)))
            .unwrap();
        let realized = accountant.snapshot().unwrap().cash_flow;
        ledger = accountant.ledger().clone();
        let journal = ledger.journal();
        match case.spec.margining() {
            Margining::Linear => {
                assert_eq!(Q::dec(realized), exact, "{}", case.name);
                linear_cases += 1;
            }
            _ => {
                // realized + s·(r_open - r_close) = exact: no tolerance.
                let rests = rest(&journal[0]).add(rest(&journal[1]).neg());
                let rests = match side {
                    Side::Long => rests,
                    Side::Short => rests.neg(),
                };
                assert_eq!(Q::dec(realized).add(rests), exact, "{}", case.name);
                inverse_cases += 1;
            }
        }

        // Through the whole engine, the same realized PnL.
        let result = run(
            &case.spec,
            dec!(0),
            side,
            contracts,
            entry,
            true,
            exit,
            exit,
        );
        assert!(
            result.accounting.is_consistent(),
            "{}: {:?}",
            case.name,
            result.accounting
        );
        assert_eq!(result.num_trades, 2, "{}", case.name);
        assert_eq!(result.final_position, dec!(0), "{}", case.name);
        assert_eq!(result.total_pnl, realized, "{}", case.name);
    }
    assert_eq!((linear_cases, inverse_cases), (2, 3));
}
