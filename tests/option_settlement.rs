//! Option expiry, exercise and assignment booked in a ledger, by hand. Synthetic fixtures.
//!
//! The ledger holds USDC (6 decimals) and BTC (8 decimals) and the spot BTC-USDC. Each option
//! is on BTC in USDC, strike 100, 10 BTC a contract, registered in the ledger as "C1" with its
//! expiry before the terms' cutoff, and held there by a fill at premium zero, unless a test says
//! otherwise.

use chrono::{DateTime, TimeZone, Utc};
use margin_rs::instrument::{
    ExerciseStyle, LinearTerms, OptionContract, OptionDelivery, OptionInstrument, OptionMargining,
    OptionRight,
};
use margin_rs::ledger::{Execution, Ledger, LedgerSnapshot, OptionOutcome};
use margin_rs::{
    Action, Asset, ExecutionId, InstrumentId, ReservationId, SpotInstrument, TransferId,
};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::options::settlement::{OptionBook, OptionPosition};
use market_maker_rs::types::error::MMError;
use optionstratlib::Options;
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::model::{
    CashRounding, ExpirationDate, SettlementEvent, SettlementPrice, SettlementTerms, SettlementType,
};
use optionstratlib::prelude::{Positive, pos_or_panic};

/// The options' expiry in the ledger, before the terms' cutoff of 1_700_000_000 s.
const EXPIRY_MS: u64 = 1_699_000_000_000;
/// When the fills that open the positions were executed, before the expiry.
const FILLED_AT_MS: u64 = 1_698_000_000_000;
const CONTRACT: &str = "C1";

fn usdc() -> Asset {
    Asset::new("USDC").unwrap()
}

fn btc() -> Asset {
    Asset::new("BTC").unwrap()
}

fn iid(id: &str) -> InstrumentId {
    InstrumentId::new(id).unwrap()
}

fn spot() -> InstrumentId {
    iid("BTC-USDC")
}

/// A ledger with these balances of cash and of the underlying, and no option.
fn ledger(cash: Decimal, base: Decimal) -> Ledger {
    let mut ledger = Ledger::default();
    ledger.register_asset(usdc(), 6).unwrap();
    ledger.register_asset(btc(), 8).unwrap();
    ledger
        .register_instrument(SpotInstrument::new(spot(), btc(), usdc(), 2, 8).unwrap())
        .unwrap();
    if cash > Decimal::ZERO {
        ledger
            .deposit(TransferId::new("cash").unwrap(), usdc(), cash)
            .unwrap();
    }
    if base > Decimal::ZERO {
        ledger
            .deposit(TransferId::new("base").unwrap(), btc(), base)
            .unwrap();
    }
    ledger
}

fn option(kind: OptionType, side: Side, style: OptionStyle, quantity: f64) -> Options {
    Options::new(
        kind,
        side,
        "BTC".to_string(),
        pos_or_panic!(100.0),
        ExpirationDate::Days(pos_or_panic!(30.0)),
        pos_or_panic!(0.2),
        pos_or_panic!(quantity),
        pos_or_panic!(100.0),
        dec!(0.0),
        style,
        Positive::ZERO,
        None,
    )
    .with_contract_size(pos_or_panic!(10.0))
}

fn european(side: Side, style: OptionStyle) -> Options {
    option(OptionType::European, side, style, 2.0)
}

fn long_call() -> Options {
    european(Side::Long, OptionStyle::Call)
}

/// The ledger's contract for `option`, delivered as `delivery`.
fn contract_for(option: &Options, delivery: OptionDelivery) -> OptionContract {
    OptionContract {
        underlying: btc(),
        strike: option.strike_price.to_dec(),
        expiry_ms: EXPIRY_MS,
        right: match option.option_style {
            OptionStyle::Call => OptionRight::Call,
            OptionStyle::Put => OptionRight::Put,
        },
        exercise: match option.option_type {
            OptionType::American => ExerciseStyle::American,
            _ => ExerciseStyle::European,
        },
        delivery,
        margining: OptionMargining::Linear,
    }
}

/// Registers a linear option `id` settled in `settlement`.
fn register(
    ledger: &mut Ledger,
    id: &str,
    settlement: Asset,
    multiplier: Decimal,
    contract: OptionContract,
) {
    ledger
        .register_instrument(
            OptionInstrument::new(
                LinearTerms::new(iid(id), settlement, multiplier, 4, 4).unwrap(),
                contract,
            )
            .unwrap(),
        )
        .unwrap();
}

/// Books a fill of `contracts` of `id` at `premium`, before the expiry.
fn fill(ledger: &mut Ledger, id: &str, side: Action, premium: Decimal, contracts: Decimal) {
    let execution_id = ExecutionId::new(format!("fill:{id}:{}", ledger.journal().len())).unwrap();
    let execution = Execution::new(execution_id, iid(id), side, premium, contracts)
        .with_executed_at(FILLED_AT_MS);
    ledger
        .book_execution_allowing_overdraft(&execution, None)
        .unwrap();
}

/// Books the option's side and quantity of `id` at `premium`.
fn hold(ledger: &mut Ledger, id: &str, option: &Options, premium: Decimal) {
    let side = match option.side {
        Side::Long => Action::Buy,
        Side::Short => Action::Sell,
    };
    fill(ledger, id, side, premium, option.quantity.to_dec());
}

/// Registers `id` as `option` settled in USDC, and holds it at premium zero.
fn list(ledger: &mut Ledger, id: &str, option: &Options, delivery: OptionDelivery) {
    register(
        ledger,
        id,
        usdc(),
        option.contract_size.to_dec(),
        contract_for(option, delivery),
    );
    hold(ledger, id, option, dec!(0));
}

fn position_in(option: Options, id: &str) -> OptionPosition {
    OptionPosition::new(option, iid(id)).with_spot(spot())
}

fn position(option: Options) -> OptionPosition {
    position_in(option, CONTRACT)
}

/// A book with `option` open as "C1", listed and held in `ledger`.
fn book_of(ledger: &mut Ledger, option: Options, delivery: OptionDelivery) -> OptionBook {
    list(ledger, CONTRACT, &option, delivery);
    let mut book = OptionBook::new();
    book.open(ledger, position(option)).unwrap();
    book
}

fn cutoff() -> DateTime<Utc> {
    Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
}

fn terms(kind: SettlementType, event: SettlementEvent, price: Option<f64>) -> SettlementTerms {
    SettlementTerms {
        settlement_type: kind,
        event,
        price: price.map(|price| SettlementPrice {
            price: pos_or_panic!(price),
            source: "exchange close".to_string(),
            observed_at: cutoff(),
        }),
        cutoff: cutoff(),
        rounding: CashRounding { scale: 2 },
    }
}

fn expiry(kind: SettlementType, price: f64) -> SettlementTerms {
    terms(kind, SettlementEvent::Expiry, Some(price))
}

fn cash(price: f64) -> SettlementTerms {
    expiry(SettlementType::Cash, price)
}

fn physical(price: f64) -> SettlementTerms {
    expiry(SettlementType::Physical, price)
}

/// The delivery of the instrument that settles as `kind`.
fn delivery_of(kind: SettlementType) -> OptionDelivery {
    match kind {
        SettlementType::Cash => OptionDelivery::Cash,
        SettlementType::Physical => OptionDelivery::Physical,
    }
}

/// Total and reserved of an asset.
fn held(ledger: &Ledger, asset: &Asset) -> (Decimal, Decimal) {
    let balance = ledger.balance(asset).unwrap();
    (balance.total, balance.reserved)
}

/// Contracts the ledger holds of `id`.
fn contracts(ledger: &Ledger, id: &str) -> Decimal {
    ledger
        .position(&iid(id))
        .map_or(Decimal::ZERO, |position| position.quantity)
}

/// Everything a refused call must leave alone.
fn state(ledger: &Ledger) -> (LedgerSnapshot, usize, usize) {
    (
        ledger.snapshot(),
        ledger.journal().len(),
        ledger.option_settlement_journal().len(),
    )
}

#[test]
fn test_a_long_call_in_the_money_settled_in_cash_receives_the_intrinsic_value() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Cash);
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    // (110 - 100) * 2 contracts * 10 a contract.
    assert_eq!(record.obligation.cash, dec!(200));
    assert_eq!(record.obligation.underlying, dec!(0));
    assert_eq!(record.outcome, OptionOutcome::Cash { price: dec!(110) });
    assert_eq!(held(&ledger, &usdc()), (dec!(1200), dec!(0)));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    assert_eq!(record.price_source.as_deref(), Some("exchange close"));
    assert_eq!(record.event, SettlementEvent::Expiry);
    assert_eq!(record.settlement_type, SettlementType::Cash);
    assert!(record.shortfalls.is_empty());
    assert!(ledger.reconcile().unwrap().is_clean());
    // It is the option's realized PnL, and no transfer of capital: only the seed deposit is a
    // transfer.
    assert_eq!(record.realized_pnl, dec!(200));
    let after = ledger.position(&iid(CONTRACT)).unwrap();
    assert_eq!(after.quantity, dec!(0));
    assert_eq!(after.realized_pnl, dec!(200));
    assert_eq!(ledger.transfers().count(), 1);
}

#[test]
fn test_a_short_call_in_the_money_settled_in_cash_pays_the_intrinsic_value() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(
        &mut ledger,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Cash,
    );
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(800));
    assert_eq!(record.realized_pnl, dec!(-200));
    assert_eq!(contracts(&ledger, CONTRACT), dec!(0));
}

#[test]
fn test_at_and_out_of_the_money_a_position_lapses_with_nothing_moved() {
    for (style, price) in [
        (OptionStyle::Call, 100.0),
        (OptionStyle::Call, 90.0),
        (OptionStyle::Put, 100.0),
        (OptionStyle::Put, 110.0),
    ] {
        for kind in [SettlementType::Cash, SettlementType::Physical] {
            let mut ledger = ledger(dec!(1000), dec!(5));
            let mut book = book_of(&mut ledger, european(Side::Short, style), delivery_of(kind));
            let record = book
                .settle(&mut ledger, CONTRACT, &expiry(kind, price))
                .unwrap();
            assert_eq!(record.obligation.cash, dec!(0));
            assert_eq!(record.obligation.underlying, dec!(0));
            assert_eq!(held(&ledger, &usdc()).0, dec!(1000));
            assert_eq!(held(&ledger, &btc()).0, dec!(5));
            // A cash option settles at its price, paying nothing; a physical one lapses.
            let outcome = match kind {
                SettlementType::Cash => OptionOutcome::Cash {
                    price: Decimal::try_from(price).unwrap(),
                },
                SettlementType::Physical => OptionOutcome::Lapsed,
            };
            assert_eq!(record.outcome, outcome);
            // It is gone even so, from the book and the ledger.
            assert!(book.position(CONTRACT).is_none());
            assert_eq!(contracts(&ledger, CONTRACT), dec!(0));
        }
    }
}

#[test]
fn test_a_long_call_settled_physically_pays_the_strike_and_receives_by_the_multiplier() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Physical);
    let record = book
        .settle(&mut ledger, CONTRACT, &physical(110.0))
        .unwrap();
    assert_eq!(record.settlement_type, SettlementType::Physical);
    assert_eq!(record.obligation.extinguished_quantity, pos_or_panic!(2.0));
    assert_eq!(
        record.outcome,
        OptionOutcome::Delivery {
            spot: spot(),
            delivery_id: ExecutionId::new("delivery:C1").unwrap(),
        }
    );
    // 2 contracts of 10: 20 of the underlying, for 100 each.
    assert_eq!(held(&ledger, &usdc()).0, dec!(3000));
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
    assert!(ledger.reconcile().unwrap().is_clean());
    // A spot execution at the strike, so the underlying's cost is the strike, and no PnL yet.
    assert_eq!(ledger.transfers().count(), 1);
    assert_eq!(record.realized_pnl, dec!(0));
    let underlying = ledger.position(&spot()).unwrap();
    assert_eq!(underlying.quantity, dec!(20));
    assert_eq!(underlying.cost_basis, dec!(2000));
    // The opening fill and the delivery.
    assert_eq!(ledger.journal().len(), 2);
    assert_eq!(contracts(&ledger, CONTRACT), dec!(0));
}

#[test]
fn test_a_short_call_settled_physically_delivers_and_receives_the_strike() {
    let mut ledger = ledger(dec!(0), dec!(25));
    let mut book = book_of(
        &mut ledger,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Physical,
    );
    book.settle(&mut ledger, CONTRACT, &physical(110.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(2000));
    assert_eq!(held(&ledger, &btc()).0, dec!(5));
}

#[test]
fn test_a_long_put_settled_physically_delivers_and_receives_the_strike() {
    let mut ledger = ledger(dec!(0), dec!(20));
    let mut book = book_of(
        &mut ledger,
        european(Side::Long, OptionStyle::Put),
        OptionDelivery::Physical,
    );
    book.settle(&mut ledger, CONTRACT, &physical(90.0)).unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(2000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
}

#[test]
fn test_a_settled_position_cannot_be_held_or_settled_again() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Cash);
    let terms = cash(110.0);
    let record = book.settle(&mut ledger, CONTRACT, &terms).unwrap();
    assert!(book.position(CONTRACT).is_none());
    assert_eq!(book.settlement(CONTRACT), Some(&record));
    assert!(book.settle(&mut ledger, CONTRACT, &terms).is_err());
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
    // Nor can the contract be opened again in this book, even with the ledger holding it again.
    hold(&mut ledger, CONTRACT, &long_call(), dec!(0));
    let again = position(long_call());
    assert!(book.open(&ledger, again.clone()).is_err());
    // Nor twice in another.
    let mut open = OptionBook::new();
    open.open(&ledger, again.clone()).unwrap();
    assert!(open.open(&ledger, again).is_err());
}

#[test]
fn test_open_orders_are_released_first_and_can_pay_the_settlement() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let order = ReservationId::new("order-1").unwrap();
    ledger.reserve(order.clone(), usdc(), dec!(950)).unwrap();
    let mut book = book_of(
        &mut ledger,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Cash,
    );
    book.add_reservation(CONTRACT, order.clone()).unwrap();
    // It owes 200, with 50 free until the order's reservation goes.
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    assert_eq!(record.released, vec![(order, usdc(), dec!(950))]);
    assert_eq!(held(&ledger, &usdc()), (dec!(800), dec!(0)));
    assert!(record.shortfalls.is_empty());
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_an_obligation_that_cannot_be_covered_is_booked_with_its_shortfall() {
    // Cash: it owes 200 with 100.
    let mut poor = ledger(dec!(100), dec!(0));
    let mut book = book_of(
        &mut poor,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Cash,
    );
    let record = book.settle(&mut poor, CONTRACT, &cash(110.0)).unwrap();
    assert_eq!(record.shortfalls.get(&usdc()), Some(&dec!(100)));
    assert_eq!(held(&poor, &usdc()).0, dec!(-100));
    assert!(book.position(CONTRACT).is_none());
    assert_eq!(contracts(&poor, CONTRACT), dec!(0));
    // Delivery: it owes 20 of the underlying with 5, and gets paid in cash.
    let mut none = ledger(dec!(0), dec!(5));
    let mut book = book_of(
        &mut none,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Physical,
    );
    let record = book.settle(&mut none, CONTRACT, &physical(110.0)).unwrap();
    assert_eq!(record.shortfalls.get(&btc()), Some(&dec!(15)));
    assert_eq!(held(&none, &btc()).0, dec!(-15));
    assert_eq!(held(&none, &usdc()).0, dec!(2000));
}

/// The terms that fail, on a contract delivered as they say, with what stayed the same.
fn assert_nothing_moved(failing: &SettlementTerms, option: Options) {
    let mut ledger = ledger(dec!(1000), dec!(5));
    let order = ReservationId::new("order-1").unwrap();
    ledger.reserve(order.clone(), usdc(), dec!(10)).unwrap();
    let mut book = book_of(&mut ledger, option, delivery_of(failing.settlement_type));
    book.add_reservation(CONTRACT, order).unwrap();
    let before = state(&ledger);
    assert!(book.settle(&mut ledger, CONTRACT, failing).is_err());
    assert!(book.position(CONTRACT).is_some());
    assert!(book.settlement(CONTRACT).is_none());
    assert_eq!(held(&ledger, &usdc()), (dec!(1000), dec!(10)));
    assert_eq!(held(&ledger, &btc()).0, dec!(5));
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_a_missing_settlement_price_moves_nothing() {
    assert_nothing_moved(
        &terms(SettlementType::Cash, SettlementEvent::Expiry, None),
        long_call(),
    );
}

#[test]
fn test_a_price_after_the_cutoff_moves_nothing() {
    let mut late = cash(110.0);
    late.cutoff = Utc.timestamp_opt(1_699_999_999, 0).single().unwrap();
    assert_nothing_moved(&late, long_call());
}

#[test]
fn test_an_event_of_the_wrong_side_moves_nothing() {
    assert_nothing_moved(
        &terms(
            SettlementType::Cash,
            SettlementEvent::Assignment,
            Some(110.0),
        ),
        long_call(),
    );
    assert_nothing_moved(
        &terms(SettlementType::Cash, SettlementEvent::Exercise, Some(110.0)),
        european(Side::Short, OptionStyle::Call),
    );
}

#[test]
fn test_an_unsupported_option_type_is_refused_and_moves_nothing() {
    // The ledger holds European and American options only, so an Asian one is refused when it
    // is opened, and there is nothing to settle.
    let asian = option(
        OptionType::Asian {
            averaging_type: optionstratlib::model::types::AsianAveragingType::Arithmetic,
        },
        Side::Long,
        OptionStyle::Call,
        2.0,
    );
    let mut ledger = ledger(dec!(1000), dec!(5));
    list(&mut ledger, CONTRACT, &asian, OptionDelivery::Cash);
    let before = state(&ledger);
    let mut book = OptionBook::new();
    assert!(book.open(&ledger, position(asian)).is_err());
    assert!(book.position(CONTRACT).is_none());
    assert!(book.settle(&mut ledger, CONTRACT, &cash(110.0)).is_err());
    assert_eq!(state(&ledger), before);
    assert_eq!(contracts(&ledger, CONTRACT), dec!(2));
}

#[test]
fn test_a_reservation_no_longer_in_the_ledger_is_noted_and_does_not_block() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let live = ReservationId::new("live").unwrap();
    let gone = ReservationId::new("filled").unwrap();
    ledger.reserve(live.clone(), usdc(), dec!(10)).unwrap();
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Cash);
    book.add_reservation(CONTRACT, gone.clone()).unwrap();
    book.add_reservation(CONTRACT, live.clone()).unwrap();
    assert_eq!(
        book.position(CONTRACT).unwrap().reservations(),
        &[gone.clone(), live.clone()]
    );
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    assert_eq!(record.already_gone, vec![gone]);
    assert_eq!(record.released, vec![(live, usdc(), dec!(10))]);
}

#[test]
fn test_a_reservation_is_registered_once_across_contracts() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Cash);
    list(&mut ledger, "C2", &long_call(), OptionDelivery::Cash);
    book.open(&ledger, position_in(long_call(), "C2")).unwrap();
    let order = ReservationId::new("order").unwrap();
    book.add_reservation(CONTRACT, order.clone()).unwrap();
    assert!(book.add_reservation(CONTRACT, order.clone()).is_err());
    assert!(book.add_reservation("C2", order).is_err());
}

/// A ledger with USD at `scale` decimals, holding the long call as "C1" settled in cash in USD.
fn usd_ledger(scale: u32, option: &Options) -> (Ledger, Asset) {
    let usd = Asset::new("USD").unwrap();
    let mut ledger = Ledger::default();
    ledger.register_asset(usd.clone(), scale).unwrap();
    register(
        &mut ledger,
        CONTRACT,
        usd.clone(),
        option.contract_size.to_dec(),
        contract_for(option, OptionDelivery::Cash),
    );
    hold(&mut ledger, CONTRACT, option, dec!(0));
    (ledger, usd)
}

#[test]
fn test_a_failure_after_a_release_undoes_the_release() {
    // A cash amount finer than this asset's scale fails after the order's release.
    let (mut ledger, usd) = usd_ledger(1, &long_call());
    ledger
        .deposit(TransferId::new("cash").unwrap(), usd.clone(), dec!(100))
        .unwrap();
    let order = ReservationId::new("order").unwrap();
    ledger
        .reserve(order.clone(), usd.clone(), dec!(10))
        .unwrap();
    let mut book = OptionBook::new();
    book.open(&ledger, OptionPosition::new(long_call(), iid(CONTRACT)))
        .unwrap();
    book.add_reservation(CONTRACT, order.clone()).unwrap();
    let before = state(&ledger);
    let mut fine = cash(100.0625);
    fine.rounding = CashRounding { scale: 3 };
    assert!(book.settle(&mut ledger, CONTRACT, &fine).is_err());
    assert_eq!(held(&ledger, &usd), (dec!(100), dec!(10)));
    assert!(ledger.reservation(&order).is_some());
    assert!(book.position(CONTRACT).is_some());
    assert!(book.settlement(CONTRACT).is_none());
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_a_physical_settlement_needs_a_usable_spot_instrument() {
    let delivered = physical(110.0);
    let mut ledger = ledger(dec!(5000), dec!(0));
    list(
        &mut ledger,
        CONTRACT,
        &long_call(),
        OptionDelivery::Physical,
    );
    let before = state(&ledger);
    // None declared.
    let mut book = OptionBook::new();
    book.open(&ledger, OptionPosition::new(long_call(), iid(CONTRACT)))
        .unwrap();
    assert!(book.settle(&mut ledger, CONTRACT, &delivered).is_err());
    assert!(book.position(CONTRACT).is_some());
    // One the ledger does not hold.
    let mut book = OptionBook::new();
    let other = iid("ETH-USDC");
    book.open(
        &ledger,
        OptionPosition::new(long_call(), iid(CONTRACT)).with_spot(other),
    )
    .unwrap();
    assert!(book.settle(&mut ledger, CONTRACT, &delivered).is_err());
    // One quoted in another asset than the option's settlement asset.
    let usd = Asset::new("USD").unwrap();
    ledger.register_asset(usd.clone(), 6).unwrap();
    register(
        &mut ledger,
        "C2",
        usd,
        dec!(10),
        contract_for(&long_call(), OptionDelivery::Physical),
    );
    hold(&mut ledger, "C2", &long_call(), dec!(0));
    let mut book = OptionBook::new();
    book.open(&ledger, position_in(long_call(), "C2")).unwrap();
    assert!(book.settle(&mut ledger, "C2", &delivered).is_err());
    assert!(book.position("C2").is_some());
    assert_eq!(ledger.option_settlement_journal().len(), 0);
    assert_eq!(held(&ledger, &usdc()), (dec!(5000), dec!(0)));
    // Nothing but the fill that opened C2 was booked.
    assert_eq!(state(&ledger).1, before.1 + 1);
    // Without a delivery none is needed.
    let mut lapse = OptionBook::new();
    lapse
        .open(&ledger, OptionPosition::new(long_call(), iid(CONTRACT)))
        .unwrap();
    let record = lapse
        .settle(&mut ledger, CONTRACT, &physical(90.0))
        .unwrap();
    assert_eq!(record.outcome, OptionOutcome::Lapsed);
}

#[test]
fn test_a_delivery_that_would_move_other_than_the_obligation_is_refused() {
    // A spot instrument of no decimals of quantity cannot book the fractional contract.
    let mut ledger = Ledger::default();
    ledger.register_asset(usdc(), 6).unwrap();
    ledger.register_asset(btc(), 8).unwrap();
    ledger
        .register_instrument(SpotInstrument::new(spot(), btc(), usdc(), 2, 0).unwrap())
        .unwrap();
    ledger
        .deposit(TransferId::new("cash").unwrap(), usdc(), dec!(5000))
        .unwrap();
    let mut book = book_of(
        &mut ledger,
        option(OptionType::European, Side::Long, OptionStyle::Call, 0.25),
        OptionDelivery::Physical,
    );
    let before = state(&ledger);
    // 0.25 contracts of 10 is 2.5 units, off this instrument's grid of whole units.
    assert!(
        book.settle(&mut ledger, CONTRACT, &physical(110.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert!(book.position(CONTRACT).is_some());
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_a_contract_is_opened_only_as_an_option_of_the_ledger_and_with_a_quantity() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = OptionBook::new();
    // An instrument the ledger does not hold.
    assert!(book.open(&ledger, position_in(long_call(), "C9")).is_err());
    // A name the ledger cannot hold is not an instrument identifier at all.
    assert!(InstrumentId::new("has a space").is_err());
    assert!(InstrumentId::new("x".repeat(65)).is_err());
    // No quantity.
    let none = option(OptionType::European, Side::Long, OptionStyle::Call, 0.0);
    register(
        &mut ledger,
        CONTRACT,
        usdc(),
        dec!(10),
        contract_for(&none, OptionDelivery::Cash),
    );
    assert!(book.open(&ledger, position(none)).is_err());
    hold(&mut ledger, CONTRACT, &long_call(), dec!(0));
    assert!(book.open(&ledger, position(long_call())).is_ok());
}

#[test]
fn test_a_contract_named_as_the_spot_instrument_is_refused() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let before = state(&ledger);
    // The spot is not an option, so it cannot be opened as one.
    let mut book = OptionBook::new();
    let refused = book
        .open(&ledger, position_in(long_call(), "BTC-USDC"))
        .unwrap_err();
    assert!(
        matches!(refused, MMError::InvalidConfiguration(_)),
        "{refused}"
    );
    assert!(book.position("BTC-USDC").is_none());
    assert!(book.settle(&mut ledger, "BTC-USDC", &cash(110.0)).is_err());
    // Nor can an option be registered under the spot's identifier.
    let named_as_spot = OptionInstrument::new(
        LinearTerms::new(spot(), usdc(), dec!(10), 4, 4).unwrap(),
        contract_for(&long_call(), OptionDelivery::Cash),
    )
    .unwrap();
    assert!(ledger.register_instrument(named_as_spot).is_err());
    assert_eq!(held(&ledger, &usdc()).0, dec!(1000));
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_an_american_option_at_expiry_settles_as_the_rules_say() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(
        &mut ledger,
        option(OptionType::American, Side::Long, OptionStyle::Call, 2.0),
        OptionDelivery::Cash,
    );
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    assert_eq!(record.obligation.cash, dec!(200));
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
}

#[test]
fn test_a_cash_amount_finer_than_the_asset_moves_nothing() {
    let (mut ledger, usd) = usd_ledger(1, &long_call());
    let mut book = OptionBook::new();
    book.open(&ledger, OptionPosition::new(long_call(), iid(CONTRACT)))
        .unwrap();
    let before = state(&ledger);
    // 0.0625 over the strike on 20 units is 1.25, finer than this asset's one decimal.
    let mut fine = cash(100.0625);
    fine.rounding = CashRounding { scale: 3 };
    assert!(book.settle(&mut ledger, CONTRACT, &fine).is_err());
    assert_eq!(held(&ledger, &usd).0, dec!(0));
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_cash_rounds_as_the_terms_say() {
    let one = option(OptionType::European, Side::Long, OptionStyle::Call, 1.0);
    let mut terms = cash(100.0026);
    terms.rounding = CashRounding { scale: 2 };
    // 0.0026 on 10 units is 0.026, to two places, in an asset of two places.
    let (mut ledger, usd) = usd_ledger(2, &one);
    let mut book = OptionBook::new();
    book.open(&ledger, OptionPosition::new(one.clone(), iid(CONTRACT)))
        .unwrap();
    let record = book.settle(&mut ledger, CONTRACT, &terms).unwrap();
    assert_eq!(record.obligation.cash, dec!(0.03));
    assert_eq!(held(&ledger, &usd).0, dec!(0.03));
    // A ledger that keeps six places books 0.026, other than the terms: refused.
    let mut ledger = self::ledger(dec!(0), dec!(0));
    let mut book = book_of(&mut ledger, one, OptionDelivery::Cash);
    let before = state(&ledger);
    assert!(book.settle(&mut ledger, CONTRACT, &terms).is_err());
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_exercise_and_assignment_settle_as_the_terms_say() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(
        &mut ledger,
        option(OptionType::American, Side::Long, OptionStyle::Call, 2.0),
        OptionDelivery::Cash,
    );
    let record = book
        .settle(
            &mut ledger,
            CONTRACT,
            &terms(SettlementType::Cash, SettlementEvent::Exercise, Some(110.0)),
        )
        .unwrap();
    assert_eq!(record.event, SettlementEvent::Exercise);
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
    let mut ledger = self::ledger(dec!(1000), dec!(0));
    let mut book = book_of(
        &mut ledger,
        european(Side::Short, OptionStyle::Call),
        OptionDelivery::Cash,
    );
    book.settle(
        &mut ledger,
        CONTRACT,
        &terms(
            SettlementType::Cash,
            SettlementEvent::Assignment,
            Some(110.0),
        ),
    )
    .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(800));
}

#[test]
fn test_an_exercise_can_be_followed_by_hedging_with_what_it_delivered() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Physical);
    book.settle(&mut ledger, CONTRACT, &physical(110.0))
        .unwrap();
    // Sells what the exercise gave it, at the market.
    let sale = Execution::new(
        ExecutionId::new("hedge-1").unwrap(),
        spot(),
        Action::Sell,
        dec!(110),
        dec!(20),
    );
    ledger.book_execution(&sale).unwrap();
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    // Paid 2000 and sold for 2200.
    assert_eq!(held(&ledger, &usdc()).0, dec!(5200));
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_positions_keep_their_own_contracts() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = OptionBook::new();
    for name in ["C1", "C2"] {
        list(&mut ledger, name, &long_call(), OptionDelivery::Cash);
        book.open(&ledger, position_in(long_call(), name)).unwrap();
    }
    book.settle(&mut ledger, "C1", &cash(110.0)).unwrap();
    assert!(book.position("C2").is_some());
    assert_eq!(contracts(&ledger, "C2"), dec!(2));
    book.settle(&mut ledger, "C2", &cash(105.0)).unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(1300));
    assert_eq!(ledger.option_settlement_journal().len(), 2);
}

#[test]
fn test_a_reservation_needs_an_open_contract() {
    let mut book = OptionBook::new();
    assert!(
        book.add_reservation("C9", ReservationId::new("r").unwrap())
            .is_err()
    );
}

/// After `first` settled "C1", a book rebuilt on the same ledger cannot open it, as the ledger
/// is flat; if a fill before the expiry reopens the position, as the ledger allows, the rebuilt
/// book opens it but cannot settle it again under `second`. Nothing moves either way. Returns
/// the second refusal.
fn assert_rebuilt_book_refused(
    option: &Options,
    delivery: OptionDelivery,
    first: &SettlementTerms,
    second: &SettlementTerms,
) -> (Ledger, MMError) {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(&mut ledger, option.clone(), delivery)
        .settle(&mut ledger, CONTRACT, first)
        .unwrap();
    let settled = state(&ledger);
    let mut rebuilt = OptionBook::new();
    let refused = rebuilt.open(&ledger, position(option.clone())).unwrap_err();
    assert!(
        refused.to_string().contains("holds 0 contracts"),
        "{refused}"
    );
    assert!(rebuilt.position(CONTRACT).is_none());
    assert!(rebuilt.settle(&mut ledger, CONTRACT, second).is_err());
    assert_eq!(state(&ledger), settled);

    hold(&mut ledger, CONTRACT, option, dec!(0));
    rebuilt.open(&ledger, position(option.clone())).unwrap();
    let reopened = state(&ledger);
    let refused = rebuilt.settle(&mut ledger, CONTRACT, second).unwrap_err();
    assert!(rebuilt.position(CONTRACT).is_some());
    assert!(rebuilt.settlement(CONTRACT).is_none());
    assert_eq!(state(&ledger), reopened);
    assert_eq!(ledger.option_settlement_journal().len(), 1);
    (ledger, refused)
}

#[test]
fn test_a_ledger_that_already_booked_this_settlement_refuses_it() {
    let terms = cash(110.0);
    let (ledger, refused) =
        assert_rebuilt_book_refused(&long_call(), OptionDelivery::Cash, &terms, &terms);
    assert!(refused.to_string().contains("already booked"), "{refused}");
    assert_eq!(held(&ledger, &usdc()).0, dec!(5200));
}

#[test]
fn test_a_ledger_that_already_delivered_this_settlement_refuses_it() {
    let terms = physical(110.0);
    let (ledger, refused) =
        assert_rebuilt_book_refused(&long_call(), OptionDelivery::Physical, &terms, &terms);
    assert!(refused.to_string().contains("already booked"), "{refused}");
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
    assert_eq!(held(&ledger, &usdc()).0, dec!(3000));
}

#[test]
fn test_a_delivery_rounded_away_from_what_the_ledger_books_is_refused() {
    // 0.3333 contracts of 10 at a strike of 100 is 333.3, which these terms round to 333.
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = book_of(
        &mut ledger,
        option(OptionType::European, Side::Long, OptionStyle::Call, 0.3333),
        OptionDelivery::Physical,
    );
    let before = state(&ledger);
    let mut terms = physical(110.0);
    terms.rounding = CashRounding { scale: 0 };
    assert!(book.settle(&mut ledger, CONTRACT, &terms).is_err());
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    assert!(book.position(CONTRACT).is_some());
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_a_lapse_stays_in_the_ledger_with_nothing_moved() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let record = book_of(&mut ledger, long_call(), OptionDelivery::Physical)
        .settle(&mut ledger, CONTRACT, &physical(90.0))
        .unwrap();
    assert_eq!(record.outcome, OptionOutcome::Lapsed);
    assert_eq!(record.realized_pnl, dec!(0));
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    assert_eq!(contracts(&ledger, CONTRACT), dec!(0));
    let journal = ledger.option_settlement_journal();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0].outcome, OptionOutcome::Lapsed);
    assert_eq!(journal[0].realized_pnl, dec!(0));
    assert_eq!(journal[0].instrument.as_str(), CONTRACT);
    assert_eq!(journal[0].id.as_str(), "settlement:C1");
    assert!(journal[0].delivery.is_none());
}

#[test]
fn test_a_lapsed_contract_cannot_pay_after_the_book_is_rebuilt() {
    // A cash option out of the money settles at its price for nothing; then it cannot pay.
    let (ledger, _) = assert_rebuilt_book_refused(
        &long_call(),
        OptionDelivery::Cash,
        &cash(90.0),
        &cash(110.0),
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
}

#[test]
fn test_a_lapsed_contract_cannot_deliver_after_the_book_is_rebuilt() {
    let (ledger, _) = assert_rebuilt_book_refused(
        &long_call(),
        OptionDelivery::Physical,
        &physical(90.0),
        &physical(110.0),
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
}

#[test]
fn test_a_paid_contract_cannot_lapse_after_the_book_is_rebuilt() {
    let (ledger, _) = assert_rebuilt_book_refused(
        &long_call(),
        OptionDelivery::Cash,
        &cash(110.0),
        &cash(90.0),
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5200));
}

#[test]
fn test_a_delivered_contract_cannot_lapse_after_the_book_is_rebuilt() {
    let (ledger, _) = assert_rebuilt_book_refused(
        &long_call(),
        OptionDelivery::Physical,
        &physical(110.0),
        &physical(90.0),
    );
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
    assert_eq!(held(&ledger, &usdc()).0, dec!(3000));
}

#[test]
fn test_a_second_lapse_after_the_book_is_rebuilt_is_refused() {
    let terms = physical(90.0);
    let (ledger, refused) =
        assert_rebuilt_book_refused(&long_call(), OptionDelivery::Physical, &terms, &terms);
    assert!(refused.to_string().contains("already booked"), "{refused}");
    assert_eq!(ledger.option_settlement_journal().len(), 1);
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
}

// ----- the instrument and the position must be the option's -----------------------------------

/// Opening the long call as "C1", registered with `contract` and `multiplier`, and held at
/// `held` signed contracts, is refused with a message containing `why`.
fn assert_open_refused(contract: OptionContract, multiplier: Decimal, held: Decimal, why: &str) {
    let mut ledger = ledger(dec!(1000), dec!(0));
    register(&mut ledger, CONTRACT, usdc(), multiplier, contract);
    if held > Decimal::ZERO {
        fill(&mut ledger, CONTRACT, Action::Buy, dec!(0), held);
    } else if held < Decimal::ZERO {
        fill(&mut ledger, CONTRACT, Action::Sell, dec!(0), -held);
    }
    let before = state(&ledger);
    let mut book = OptionBook::new();
    let refused = book.open(&ledger, position(long_call())).unwrap_err();
    assert!(
        matches!(refused, MMError::InvalidConfiguration(_)),
        "{refused}"
    );
    assert!(refused.to_string().contains(why), "{refused}");
    assert!(book.position(CONTRACT).is_none());
    assert_eq!(state(&ledger), before);
}

fn matching() -> OptionContract {
    contract_for(&long_call(), OptionDelivery::Cash)
}

#[test]
fn test_open_accepts_the_matching_instrument_and_position() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    register(&mut ledger, CONTRACT, usdc(), dec!(10), matching());
    fill(&mut ledger, CONTRACT, Action::Buy, dec!(0), dec!(2));
    let mut book = OptionBook::new();
    book.open(&ledger, position(long_call())).unwrap();
    assert!(book.position(CONTRACT).is_some());
}

#[test]
fn test_open_refuses_a_strike_other_than_the_options() {
    let contract = OptionContract {
        strike: dec!(105),
        ..matching()
    };
    assert_open_refused(contract, dec!(10), dec!(2), "strike");
}

#[test]
fn test_open_refuses_a_right_other_than_the_options() {
    let contract = OptionContract {
        right: OptionRight::Put,
        ..matching()
    };
    assert_open_refused(contract, dec!(10), dec!(2), "right");
}

#[test]
fn test_open_refuses_an_exercise_style_other_than_the_options() {
    let contract = OptionContract {
        exercise: ExerciseStyle::American,
        ..matching()
    };
    assert_open_refused(contract, dec!(10), dec!(2), "exercise style");
}

#[test]
fn test_open_refuses_a_multiplier_other_than_the_contract_size() {
    assert_open_refused(matching(), dec!(1), dec!(2), "multiplier");
}

#[test]
fn test_open_refuses_a_ledger_position_of_the_wrong_side() {
    assert_open_refused(matching(), dec!(10), dec!(-2), "holds -2 contracts");
}

#[test]
fn test_open_refuses_a_ledger_position_of_the_wrong_quantity() {
    assert_open_refused(matching(), dec!(10), dec!(1), "holds 1 contracts");
}

#[test]
fn test_open_refuses_no_ledger_position() {
    assert_open_refused(matching(), dec!(10), dec!(0), "holds 0 contracts");
}

#[test]
fn test_an_inverse_option_is_refused() {
    let mut ledger = ledger(dec!(1000), dec!(1));
    let inverse = OptionInstrument::new(
        LinearTerms::new(iid(CONTRACT), btc(), dec!(10), 4, 4).unwrap(),
        OptionContract {
            margining: OptionMargining::Inverse,
            ..matching()
        },
    )
    .unwrap();
    ledger.register_instrument(inverse).unwrap();
    fill(&mut ledger, CONTRACT, Action::Buy, dec!(0), dec!(2));
    let before = state(&ledger);
    let mut book = OptionBook::new();
    let refused = book.open(&ledger, position(long_call())).unwrap_err();
    assert!(
        matches!(refused, MMError::InvalidConfiguration(_)),
        "{refused}"
    );
    assert!(refused.to_string().contains("inverse"), "{refused}");
    assert!(book.settle(&mut ledger, CONTRACT, &cash(110.0)).is_err());
    assert_eq!(state(&ledger), before);
}

// ----- the terms must be the instrument's ------------------------------------------------------

/// Settling under terms of the other delivery than the instrument's is refused by this module
/// before the ledger is asked, and moves nothing.
fn assert_delivery_disagrees(delivery: OptionDelivery, terms: &SettlementTerms) {
    let mut ledger = ledger(dec!(5000), dec!(5));
    let mut book = book_of(&mut ledger, long_call(), delivery);
    let before = state(&ledger);
    let refused = book.settle(&mut ledger, CONTRACT, terms).unwrap_err();
    assert!(
        matches!(refused, MMError::InvalidConfiguration(_)),
        "{refused}"
    );
    assert!(refused.to_string().contains("the terms say"), "{refused}");
    assert!(book.position(CONTRACT).is_some());
    assert!(book.settlement(CONTRACT).is_none());
    assert_eq!(state(&ledger), before);
}

#[test]
fn test_settle_refuses_physical_terms_for_a_cash_instrument() {
    assert_delivery_disagrees(OptionDelivery::Cash, &physical(110.0));
    assert_delivery_disagrees(OptionDelivery::Cash, &physical(90.0));
}

#[test]
fn test_settle_refuses_cash_terms_for_a_physical_instrument() {
    assert_delivery_disagrees(OptionDelivery::Physical, &cash(110.0));
    assert_delivery_disagrees(OptionDelivery::Physical, &cash(90.0));
}

// ----- outcomes, premium and reconciliation ----------------------------------------------------

#[test]
fn test_a_cash_option_out_of_the_money_settles_in_cash_for_nothing() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(&mut ledger, long_call(), OptionDelivery::Cash);
    let record = book.settle(&mut ledger, CONTRACT, &cash(90.0)).unwrap();
    assert_eq!(record.outcome, OptionOutcome::Cash { price: dec!(90) });
    assert_eq!(record.obligation.cash, dec!(0));
    assert_eq!(record.realized_pnl, dec!(0));
    assert_eq!(held(&ledger, &usdc()).0, dec!(1000));
    assert_eq!(contracts(&ledger, CONTRACT), dec!(0));
    assert!(book.position(CONTRACT).is_none());
    let journal = ledger.option_settlement_journal();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0].id.as_str(), "settlement:C1");
    assert_eq!(journal[0].instrument.as_str(), CONTRACT);
    assert_eq!(journal[0].outcome, OptionOutcome::Cash { price: dec!(90) });
    assert_eq!(journal[0].value, dec!(0));
    assert_eq!(journal[0].contracts, dec!(2));
    assert_eq!(journal[0].position_after.quantity, dec!(0));
    // The settlement applies at the terms' cutoff.
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_a_premium_paid_shows_in_the_realized_pnl() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    register(&mut ledger, CONTRACT, usdc(), dec!(10), matching());
    // 3 a unit, for 2 contracts of 10: 60 paid.
    hold(&mut ledger, CONTRACT, &long_call(), dec!(3));
    assert_eq!(held(&ledger, &usdc()).0, dec!(940));
    let mut book = OptionBook::new();
    book.open(&ledger, position(long_call())).unwrap();
    let record = book.settle(&mut ledger, CONTRACT, &cash(110.0)).unwrap();
    // It pays 200, less the 60 paid.
    assert_eq!(record.obligation.cash, dec!(200));
    assert_eq!(record.realized_pnl, dec!(140));
    assert_eq!(held(&ledger, &usdc()).0, dec!(1140));
    let after = ledger.position(&iid(CONTRACT)).unwrap();
    assert_eq!(after.realized_pnl, dec!(140));
    assert_eq!(after.cost_basis, dec!(0));
    assert_eq!(
        ledger.option_settlement_journal()[0].realized_pnl,
        dec!(140)
    );
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_a_premium_received_shows_in_the_realized_pnl_of_a_delivery() {
    let mut ledger = ledger(dec!(0), dec!(25));
    let short = european(Side::Short, OptionStyle::Call);
    register(
        &mut ledger,
        CONTRACT,
        usdc(),
        dec!(10),
        contract_for(&short, OptionDelivery::Physical),
    );
    // 3 a unit, for 2 contracts of 10: 60 received.
    hold(&mut ledger, CONTRACT, &short, dec!(3));
    let mut book = OptionBook::new();
    book.open(&ledger, position(short)).unwrap();
    let record = book
        .settle(&mut ledger, CONTRACT, &physical(110.0))
        .unwrap();
    // The premium stays in the option; the underlying goes at the strike.
    assert_eq!(record.realized_pnl, dec!(60));
    assert_eq!(held(&ledger, &usdc()).0, dec!(2060));
    assert_eq!(held(&ledger, &btc()).0, dec!(5));
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_the_ledger_reconciles_after_every_kind_of_settlement() {
    let mut ledger = ledger(dec!(10000), dec!(50));
    let mut book = OptionBook::new();
    let cases = [
        ("CASH-ITM", long_call(), OptionDelivery::Cash, cash(110.0)),
        ("CASH-OTM", long_call(), OptionDelivery::Cash, cash(90.0)),
        (
            "SHORT-CASH",
            european(Side::Short, OptionStyle::Put),
            OptionDelivery::Cash,
            cash(90.0),
        ),
        (
            "DELIVERED",
            long_call(),
            OptionDelivery::Physical,
            physical(110.0),
        ),
        (
            "ASSIGNED",
            european(Side::Short, OptionStyle::Call),
            OptionDelivery::Physical,
            physical(110.0),
        ),
        (
            "LAPSED",
            european(Side::Long, OptionStyle::Put),
            OptionDelivery::Physical,
            physical(110.0),
        ),
    ];
    for (id, option, delivery, _) in &cases {
        register(
            &mut ledger,
            id,
            usdc(),
            dec!(10),
            contract_for(option, *delivery),
        );
        hold(&mut ledger, id, option, dec!(2));
        book.open(&ledger, position_in(option.clone(), id)).unwrap();
    }
    assert!(ledger.reconcile().unwrap().is_clean());
    for (id, _, _, terms) in &cases {
        book.settle(&mut ledger, id, terms).unwrap();
        assert!(ledger.reconcile().unwrap().is_clean(), "after {id}");
        assert_eq!(contracts(&ledger, id), dec!(0));
    }
    assert_eq!(ledger.option_settlement_journal().len(), cases.len());
}

#[test]
fn test_a_contract_opens_only_if_its_settlement_can_be_named_and_then_it_settles() {
    // "delivery:" and "settlement:" with the identifier must fit the ledger's 64 bytes.
    let too_long = "x".repeat(54);
    let mut ledger = ledger(dec!(1000), dec!(0));
    list(&mut ledger, &too_long, &long_call(), OptionDelivery::Cash);
    let mut book = OptionBook::new();
    let refused = book
        .open(&ledger, position_in(long_call(), &too_long))
        .unwrap_err();
    assert!(refused.to_string().contains("64 bytes"), "{refused}");

    let longest = "y".repeat(53);
    list(&mut ledger, &longest, &long_call(), OptionDelivery::Cash);
    book.open(&ledger, position_in(long_call(), &longest))
        .unwrap();
    assert!(book.settle(&mut ledger, &longest, &cash(110.0)).is_ok());
}
