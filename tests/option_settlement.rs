//! Option expiry, exercise and assignment booked in a ledger, by hand. Synthetic fixtures.

use chrono::{DateTime, TimeZone, Utc};
use margin_rs::ledger::{Execution, Ledger};
use margin_rs::{Asset, ExecutionId, InstrumentId, ReservationId, SpotInstrument, TransferId};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::options::settlement::{OptionBook, OptionPosition};
use optionstratlib::Options;
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::model::{
    CashRounding, ExpirationDate, SettlementEvent, SettlementPrice, SettlementTerms, SettlementType,
};
use optionstratlib::prelude::{Positive, pos_or_panic};

fn usdc() -> Asset {
    Asset::new("USDC").unwrap()
}

fn btc() -> Asset {
    Asset::new("BTC").unwrap()
}

/// A ledger with these balances of cash and of the underlying.
fn ledger(cash: Decimal, base: Decimal) -> Ledger {
    let mut ledger = Ledger::default();
    ledger.register_asset(usdc(), 6).unwrap();
    ledger.register_asset(btc(), 8).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(InstrumentId::new("BTC-USDC").unwrap(), btc(), usdc(), 2, 8)
                .unwrap(),
        )
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

fn book_of(option: Options) -> OptionBook {
    let mut book = OptionBook::new();
    book.open("C1", position(option)).unwrap();
    book
}

fn spot() -> InstrumentId {
    InstrumentId::new("BTC-USDC").unwrap()
}

fn position(option: Options) -> OptionPosition {
    OptionPosition::new(option, usdc()).with_spot(spot())
}

/// Total and reserved of an asset.
fn held(ledger: &Ledger, asset: &Asset) -> (Decimal, Decimal) {
    let balance = ledger.balance(asset).unwrap();
    (balance.total, balance.reserved)
}

#[test]
fn test_a_long_call_in_the_money_settled_in_cash_receives_the_intrinsic_value() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    let record = book
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    // (110 - 100) * 2 contracts * 10 a contract.
    assert_eq!(record.obligation.cash, dec!(200));
    assert_eq!(record.obligation.underlying, dec!(0));
    assert_eq!(held(&ledger, &usdc()), (dec!(1200), dec!(0)));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    assert_eq!(record.price_source.as_deref(), Some("exchange close"));
    assert_eq!(record.event, SettlementEvent::Expiry);
    assert_eq!(record.settlement_type, SettlementType::Cash);
    assert!(record.shortfalls.is_empty());
    assert!(ledger.reconcile().unwrap().is_clean());
    // It is settled PnL, and no transfer of capital: only the seed deposit is a transfer.
    assert_eq!(ledger.settled_pnl(&usdc()), dec!(200));
    assert_eq!(ledger.transfers().count(), 1);
}

#[test]
fn test_a_short_call_in_the_money_settled_in_cash_pays_the_intrinsic_value() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    book.settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(800));
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
            let mut book = book_of(european(Side::Short, style));
            let record = book
                .settle(&mut ledger, "C1", &expiry(kind, price))
                .unwrap();
            assert_eq!(record.obligation.cash, dec!(0));
            assert_eq!(record.obligation.underlying, dec!(0));
            assert_eq!(held(&ledger, &usdc()).0, dec!(1000));
            assert_eq!(held(&ledger, &btc()).0, dec!(5));
            // It is gone even so.
            assert!(book.position("C1").is_none());
        }
    }
}

#[test]
fn test_a_long_call_settled_physically_pays_the_strike_and_receives_by_the_multiplier() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    let record = book
        .settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
        .unwrap();
    assert_eq!(record.settlement_type, SettlementType::Physical);
    assert_eq!(record.obligation.extinguished_quantity, pos_or_panic!(2.0));
    // 2 contracts of 10: 20 of the underlying, for 100 each.
    assert_eq!(held(&ledger, &usdc()).0, dec!(3000));
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
    assert!(ledger.reconcile().unwrap().is_clean());
    // A spot execution at the strike, so the underlying's cost is the strike, and no PnL yet.
    assert_eq!(ledger.transfers().count(), 1);
    assert_eq!(ledger.settled_pnl(&usdc()), dec!(0));
    assert_eq!(ledger.journal().len(), 1);
}

#[test]
fn test_a_short_call_settled_physically_delivers_and_receives_the_strike() {
    let mut ledger = ledger(dec!(0), dec!(25));
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    book.settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(2000));
    assert_eq!(held(&ledger, &btc()).0, dec!(5));
}

#[test]
fn test_a_long_put_settled_physically_delivers_and_receives_the_strike() {
    let mut ledger = ledger(dec!(0), dec!(20));
    let mut book = book_of(european(Side::Long, OptionStyle::Put));
    book.settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 90.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(2000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
}

#[test]
fn test_a_settled_position_cannot_be_held_or_settled_again() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    let terms = expiry(SettlementType::Cash, 110.0);
    let record = book.settle(&mut ledger, "C1", &terms).unwrap();
    assert!(book.position("C1").is_none());
    assert_eq!(book.settlement("C1"), Some(&record));
    assert!(book.settle(&mut ledger, "C1", &terms).is_err());
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
    // Nor can the contract be opened again.
    let again = position(european(Side::Long, OptionStyle::Call));
    assert!(book.open("C1", again.clone()).is_err());
    let mut open = book_of(european(Side::Long, OptionStyle::Call));
    assert!(open.open("C1", again).is_err());
}

#[test]
fn test_open_orders_are_released_first_and_can_pay_the_settlement() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let order = ReservationId::new("order-1").unwrap();
    ledger.reserve(order.clone(), usdc(), dec!(950)).unwrap();
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    book.add_reservation("C1", order.clone()).unwrap();
    // It owes 200, with 50 free until the order's reservation goes.
    let record = book
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert_eq!(record.released, vec![(order, usdc(), dec!(950))]);
    assert_eq!(held(&ledger, &usdc()), (dec!(800), dec!(0)));
    assert!(ledger.reconcile().unwrap().is_clean());
}

#[test]
fn test_an_obligation_that_cannot_be_covered_is_booked_with_its_shortfall() {
    // Cash: it owes 200 with 100.
    let mut poor = ledger(dec!(100), dec!(0));
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    let record = book
        .settle(&mut poor, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert_eq!(record.shortfalls.get(&usdc()), Some(&dec!(100)));
    assert_eq!(held(&poor, &usdc()).0, dec!(-100));
    assert!(book.position("C1").is_none());
    // Delivery: it owes 20 of the underlying with 5, and gets paid in cash.
    let mut none = ledger(dec!(0), dec!(5));
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    let record = book
        .settle(&mut none, "C1", &expiry(SettlementType::Physical, 110.0))
        .unwrap();
    assert_eq!(record.shortfalls.get(&btc()), Some(&dec!(15)));
    assert_eq!(held(&none, &btc()).0, dec!(-15));
    assert_eq!(held(&none, &usdc()).0, dec!(2000));
}

/// The terms that fail, with what stayed the same.
fn assert_nothing_moved(failing: &SettlementTerms, option: Options) {
    let mut ledger = ledger(dec!(1000), dec!(5));
    let order = ReservationId::new("order-1").unwrap();
    ledger.reserve(order.clone(), usdc(), dec!(10)).unwrap();
    let mut book = book_of(option);
    book.add_reservation("C1", order).unwrap();
    assert!(book.settle(&mut ledger, "C1", failing).is_err());
    assert!(book.position("C1").is_some());
    assert!(book.settlement("C1").is_none());
    assert_eq!(held(&ledger, &usdc()), (dec!(1000), dec!(10)));
    assert_eq!(held(&ledger, &btc()).0, dec!(5));
}

#[test]
fn test_a_missing_settlement_price_moves_nothing() {
    assert_nothing_moved(
        &terms(SettlementType::Cash, SettlementEvent::Expiry, None),
        european(Side::Long, OptionStyle::Call),
    );
}

#[test]
fn test_a_price_after_the_cutoff_moves_nothing() {
    let mut late = expiry(SettlementType::Cash, 110.0);
    late.cutoff = Utc.timestamp_opt(1_699_999_999, 0).single().unwrap();
    assert_nothing_moved(&late, european(Side::Long, OptionStyle::Call));
}

#[test]
fn test_an_event_of_the_wrong_side_moves_nothing() {
    assert_nothing_moved(
        &terms(
            SettlementType::Cash,
            SettlementEvent::Assignment,
            Some(110.0),
        ),
        european(Side::Long, OptionStyle::Call),
    );
    assert_nothing_moved(
        &terms(SettlementType::Cash, SettlementEvent::Exercise, Some(110.0)),
        european(Side::Short, OptionStyle::Call),
    );
}

#[test]
fn test_an_unsupported_option_type_is_refused_and_moves_nothing() {
    assert_nothing_moved(
        &expiry(SettlementType::Cash, 110.0),
        option(
            OptionType::Asian {
                averaging_type: optionstratlib::model::types::AsianAveragingType::Arithmetic,
            },
            Side::Long,
            OptionStyle::Call,
            2.0,
        ),
    );
}

#[test]
fn test_a_reservation_no_longer_in_the_ledger_is_noted_and_does_not_block() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let live = ReservationId::new("live").unwrap();
    let gone = ReservationId::new("filled").unwrap();
    ledger.reserve(live.clone(), usdc(), dec!(10)).unwrap();
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    book.add_reservation("C1", gone.clone()).unwrap();
    book.add_reservation("C1", live.clone()).unwrap();
    assert_eq!(
        book.position("C1").unwrap().reservations(),
        &[gone.clone(), live.clone()]
    );
    let record = book
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert_eq!(record.already_gone, vec![gone]);
    assert_eq!(record.released, vec![(live, usdc(), dec!(10))]);
}

#[test]
fn test_a_reservation_is_registered_once_across_contracts() {
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    book.open("C2", position(european(Side::Long, OptionStyle::Call)))
        .unwrap();
    let order = ReservationId::new("order").unwrap();
    book.add_reservation("C1", order.clone()).unwrap();
    assert!(book.add_reservation("C1", order.clone()).is_err());
    assert!(book.add_reservation("C2", order).is_err());
}

#[test]
fn test_a_failure_after_a_release_undoes_the_release() {
    // A cash amount finer than this asset's scale fails after the order's release.
    let usd = Asset::new("USD").unwrap();
    let mut ledger = Ledger::default();
    ledger.register_asset(usd.clone(), 1).unwrap();
    ledger
        .deposit(TransferId::new("cash").unwrap(), usd.clone(), dec!(100))
        .unwrap();
    let order = ReservationId::new("order").unwrap();
    ledger
        .reserve(order.clone(), usd.clone(), dec!(10))
        .unwrap();
    let mut book = OptionBook::new();
    book.open(
        "C1",
        OptionPosition::new(european(Side::Long, OptionStyle::Call), usd.clone()),
    )
    .unwrap();
    book.add_reservation("C1", order.clone()).unwrap();
    let mut fine = expiry(SettlementType::Cash, 100.0625);
    fine.rounding = CashRounding { scale: 3 };
    assert!(book.settle(&mut ledger, "C1", &fine).is_err());
    assert_eq!(held(&ledger, &usd), (dec!(100), dec!(10)));
    assert!(ledger.reservation(&order).is_some());
    assert!(book.position("C1").is_some());
    assert!(book.settlement("C1").is_none());
}

#[test]
fn test_a_physical_settlement_needs_a_usable_spot_instrument() {
    let physical = expiry(SettlementType::Physical, 110.0);
    let long = || european(Side::Long, OptionStyle::Call);
    // None declared.
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = OptionBook::new();
    book.open("C1", OptionPosition::new(long(), usdc()))
        .unwrap();
    assert!(book.settle(&mut ledger, "C1", &physical).is_err());
    assert!(book.position("C1").is_some());
    // One the ledger does not hold.
    let mut book = OptionBook::new();
    let other = InstrumentId::new("ETH-USDC").unwrap();
    book.open("C1", OptionPosition::new(long(), usdc()).with_spot(other))
        .unwrap();
    assert!(book.settle(&mut ledger, "C1", &physical).is_err());
    // One quoted in another asset than the cash.
    let usd = Asset::new("USD").unwrap();
    ledger.register_asset(usd.clone(), 6).unwrap();
    let mut book = OptionBook::new();
    book.open("C1", OptionPosition::new(long(), usd).with_spot(spot()))
        .unwrap();
    assert!(book.settle(&mut ledger, "C1", &physical).is_err());
    assert_eq!(ledger.journal().len(), 0);
    // Without a delivery none is needed.
    let mut lapse = OptionBook::new();
    lapse
        .open("C1", OptionPosition::new(long(), usdc()))
        .unwrap();
    assert!(
        lapse
            .settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 90.0))
            .is_ok()
    );
}

#[test]
fn test_a_delivery_that_would_move_other_than_the_obligation_is_refused() {
    // A spot instrument of one decimal of price cannot book a strike of 100.25 exactly, and of
    // no decimals of quantity not the fractional contract.
    let mut ledger = Ledger::default();
    ledger.register_asset(usdc(), 6).unwrap();
    ledger.register_asset(btc(), 8).unwrap();
    ledger
        .register_instrument(SpotInstrument::new(spot(), btc(), usdc(), 2, 0).unwrap())
        .unwrap();
    ledger
        .deposit(TransferId::new("cash").unwrap(), usdc(), dec!(5000))
        .unwrap();
    let mut book = book_of(option(
        OptionType::European,
        Side::Long,
        OptionStyle::Call,
        0.25,
    ));
    // 0.25 contracts of 10 is 2.5 units, off this instrument's grid of whole units.
    assert!(
        book.settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert!(book.position("C1").is_some());
}

#[test]
fn test_a_contract_is_opened_only_with_a_name_the_ledger_can_hold_and_a_quantity() {
    let mut book = OptionBook::new();
    let long = || position(european(Side::Long, OptionStyle::Call));
    assert!(book.open("has a space", long()).is_err());
    assert!(book.open("x".repeat(60), long()).is_err());
    let none = position(option(
        OptionType::European,
        Side::Long,
        OptionStyle::Call,
        0.0,
    ));
    assert!(book.open("C1", none).is_err());
    assert!(book.open("C1", long()).is_ok());
}

#[test]
fn test_a_contract_named_as_the_spot_instrument_is_refused_at_settlement() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = OptionBook::new();
    book.open(
        "BTC-USDC",
        position(european(Side::Long, OptionStyle::Call)),
    )
    .unwrap();
    assert!(
        book.settle(
            &mut ledger,
            "BTC-USDC",
            &expiry(SettlementType::Cash, 110.0)
        )
        .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(1000));
}

#[test]
fn test_an_american_option_at_expiry_settles_as_the_rules_say() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(option(
        OptionType::American,
        Side::Long,
        OptionStyle::Call,
        2.0,
    ));
    let record = book
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert_eq!(record.obligation.cash, dec!(200));
}

#[test]
fn test_a_cash_amount_finer_than_the_asset_moves_nothing() {
    let mut ledger = Ledger::default();
    let usd = Asset::new("USD").unwrap();
    ledger.register_asset(usd.clone(), 1).unwrap();
    ledger.register_asset(btc(), 8).unwrap();
    let mut book = OptionBook::new();
    book.open(
        "C1",
        OptionPosition::new(european(Side::Long, OptionStyle::Call), usd.clone()),
    )
    .unwrap();
    // 0.0625 over the strike on 20 units is 1.25, finer than this asset's one decimal.
    let mut fine = expiry(SettlementType::Cash, 100.0625);
    fine.rounding = CashRounding { scale: 3 };
    assert!(book.settle(&mut ledger, "C1", &fine).is_err());
    assert_eq!(held(&ledger, &usd).0, dec!(0));
}

#[test]
fn test_cash_rounds_as_the_terms_say() {
    let mut ledger = ledger(dec!(0), dec!(0));
    let mut book = book_of(option(
        OptionType::European,
        Side::Long,
        OptionStyle::Call,
        1.0,
    ));
    let mut terms = expiry(SettlementType::Cash, 100.0026);
    terms.rounding = CashRounding { scale: 2 };
    let record = book.settle(&mut ledger, "C1", &terms).unwrap();
    // 0.0026 on 10 units is 0.026, to two places.
    assert_eq!(record.obligation.cash, dec!(0.03));
    assert_eq!(held(&ledger, &usdc()).0, dec!(0.03));
}

#[test]
fn test_exercise_and_assignment_settle_as_the_terms_say() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let mut book = book_of(option(
        OptionType::American,
        Side::Long,
        OptionStyle::Call,
        2.0,
    ));
    let record = book
        .settle(
            &mut ledger,
            "C1",
            &terms(SettlementType::Cash, SettlementEvent::Exercise, Some(110.0)),
        )
        .unwrap();
    assert_eq!(record.event, SettlementEvent::Exercise);
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
    let mut ledger = self::ledger(dec!(1000), dec!(0));
    let mut book = book_of(european(Side::Short, OptionStyle::Call));
    book.settle(
        &mut ledger,
        "C1",
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
    let mut book = book_of(european(Side::Long, OptionStyle::Call));
    book.settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
        .unwrap();
    // Sells what the exercise gave it, at the market.
    let sale = Execution {
        id: ExecutionId::new("hedge-1").unwrap(),
        instrument: InstrumentId::new("BTC-USDC").unwrap(),
        side: Action::Sell,
        price: dec!(110),
        quantity: dec!(20),
        fee: None,
        executed_at_ms: None,
    };
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
        book.open(name, position(european(Side::Long, OptionStyle::Call)))
            .unwrap();
    }
    book.settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    assert!(book.position("C2").is_some());
    book.settle(&mut ledger, "C2", &expiry(SettlementType::Cash, 105.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(1300));
}

#[test]
fn test_a_reservation_needs_an_open_contract() {
    let mut book = OptionBook::new();
    assert!(
        book.add_reservation("C9", ReservationId::new("r").unwrap())
            .is_err()
    );
}

#[test]
fn test_a_ledger_that_already_booked_this_settlement_refuses_it() {
    let mut ledger = ledger(dec!(1000), dec!(0));
    let terms = expiry(SettlementType::Cash, 110.0);
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &terms)
        .unwrap();
    // Another book, as after a restart, on the same ledger and contract.
    let mut restarted = book_of(european(Side::Long, OptionStyle::Call));
    assert!(restarted.settle(&mut ledger, "C1", &terms).is_err());
    assert!(restarted.position("C1").is_some());
    assert_eq!(held(&ledger, &usdc()).0, dec!(1200));
}

#[test]
fn test_a_ledger_that_already_delivered_this_settlement_refuses_it() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let terms = expiry(SettlementType::Physical, 110.0);
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &terms)
        .unwrap();
    let mut restarted = book_of(european(Side::Long, OptionStyle::Call));
    let refused = restarted.settle(&mut ledger, "C1", &terms).unwrap_err();
    assert!(refused.to_string().contains("already booked"), "{refused}");
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
    assert_eq!(held(&ledger, &usdc()).0, dec!(3000));
}

#[test]
fn test_a_delivery_rounded_away_from_what_the_ledger_books_is_refused() {
    // 0.3333 contracts of 10 at a strike of 100 is 333.3, which these terms round to 333.
    let mut ledger = ledger(dec!(5000), dec!(0));
    let mut book = book_of(option(
        OptionType::European,
        Side::Long,
        OptionStyle::Call,
        0.3333,
    ));
    let mut terms = expiry(SettlementType::Physical, 110.0);
    terms.rounding = CashRounding { scale: 0 };
    assert!(book.settle(&mut ledger, "C1", &terms).is_err());
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
    assert!(book.position("C1").is_some());
}

#[test]
fn test_a_lapse_stays_in_the_ledger_with_nothing_moved() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 90.0))
        .unwrap();
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(ledger.settled_pnl(&usdc()), dec!(0));
    let journal = ledger.settlement_journal();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0].realized_pnl, dec!(0));
    assert_eq!(journal[0].instrument.as_str(), "C1");
}

#[test]
fn test_a_lapsed_contract_cannot_pay_after_the_book_is_rebuilt() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 90.0))
        .unwrap();
    let mut rebuilt = book_of(european(Side::Long, OptionStyle::Call));
    assert!(
        rebuilt
            .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert!(rebuilt.position("C1").is_some());
}

#[test]
fn test_a_lapsed_contract_cannot_deliver_after_the_book_is_rebuilt() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 90.0))
        .unwrap();
    let mut rebuilt = book_of(european(Side::Long, OptionStyle::Call));
    assert!(
        rebuilt
            .settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, dec!(5000));
    assert_eq!(held(&ledger, &btc()).0, dec!(0));
}

#[test]
fn test_a_paid_contract_cannot_lapse_after_the_book_is_rebuilt() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 110.0))
        .unwrap();
    let after = held(&ledger, &usdc()).0;
    let mut rebuilt = book_of(european(Side::Long, OptionStyle::Call));
    assert!(
        rebuilt
            .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 90.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &usdc()).0, after);
    assert!(rebuilt.position("C1").is_some());
}

#[test]
fn test_a_delivered_contract_cannot_lapse_after_the_book_is_rebuilt() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &expiry(SettlementType::Physical, 110.0))
        .unwrap();
    let mut rebuilt = book_of(european(Side::Long, OptionStyle::Call));
    assert!(
        rebuilt
            .settle(&mut ledger, "C1", &expiry(SettlementType::Cash, 90.0))
            .is_err()
    );
    assert_eq!(held(&ledger, &btc()).0, dec!(20));
}

#[test]
fn test_a_second_lapse_after_the_book_is_rebuilt_is_refused() {
    let mut ledger = ledger(dec!(5000), dec!(0));
    let terms = expiry(SettlementType::Cash, 90.0);
    book_of(european(Side::Long, OptionStyle::Call))
        .settle(&mut ledger, "C1", &terms)
        .unwrap();
    let mut rebuilt = book_of(european(Side::Long, OptionStyle::Call));
    let refused = rebuilt.settle(&mut ledger, "C1", &terms).unwrap_err();
    assert!(refused.to_string().contains("already booked"), "{refused}");
    assert_eq!(ledger.settlement_journal().len(), 1);
}
