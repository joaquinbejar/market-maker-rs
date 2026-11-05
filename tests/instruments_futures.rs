//! The lifecycle of a dated future: phases, settlement prices, cash settlement in
//! a ledger, and calendars.
//!
//! The real case is Deribit's `BTC-4NOV26`, read on 2026-11-05 at 00:09 UTC from
//! its public API: `public/get_delivery_prices?index_name=btc_usd` has
//! `{"date":"2026-11-04","delivery_price":78044.4}` and
//! `public/get_last_settlements_by_currency?currency=BTC&type=delivery` has its
//! record at `timestamp` 1793779200081, a dated contract of that family expiring at
//! 08:00 UTC, `contract_size` 10 and `tick_size` 2.5. Another delivery there is
//! 77956.81 for `BTC-2NOV26`. How the venue averages the price was not verified.

#![cfg(feature = "instruments")]

use financial_types::Side;
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::instruments::futures::{
    FutureCalendar, FutureError, FuturePhase, FutureTerms, SettlementMethod, SettlementPrice,
    SettlementSource,
};
use market_maker_rs::instruments::reference::{binance_usdm, deribit};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
    Margining, Ratio, RoundingRule, TickSchedule,
};
use market_maker_rs::prelude::OptionStyle;

/// 2026-11-04 08:00 UTC.
const EXPIRY_MS: u64 = 1_793_779_200_000;
/// The settlement record of `BTC-4NOV26`.
const RECORD_MS: u64 = 1_793_779_200_081;
const DELIVERY: Decimal = dec!(78044.4);
const SOURCE: &str = "deribit:public/get_delivery_prices?index_name=btc_usd";

fn source() -> SettlementSource {
    SettlementSource::new(SOURCE).unwrap()
}

fn id(text: &str) -> InstrumentId {
    InstrumentId::new(text).unwrap()
}

fn inverse() -> FutureTerms {
    FutureTerms::new(
        deribit::btc_future(id("BTC-4NOV26"), EXPIRY_MS).unwrap(),
        SettlementMethod::Cash,
        source(),
    )
    .unwrap()
}

fn linear_spec(name: &str, underlying: &str, expiry_ms: u64) -> InstrumentSpec {
    let asset = |code| AssetCode::new(code).unwrap();
    InstrumentSpec::new(InstrumentDraft {
        id: id(name),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset(underlying),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(0.01), asset(underlying)).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
        lot: dec!(1),
        expiry_ms: Some(expiry_ms),
        option: None,
    })
    .unwrap()
}

fn terms_of(spec: InstrumentSpec) -> FutureTerms {
    FutureTerms::new(spec, SettlementMethod::Cash, source()).unwrap()
}

fn linear() -> FutureTerms {
    terms_of(linear_spec("FUT", "BTC", EXPIRY_MS))
}

fn print(terms: &FutureTerms, price: Decimal, published_ms: u64) -> SettlementPrice {
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
fn only_a_dated_future_has_these_terms() {
    assert!(matches!(
        FutureTerms::new(
            deribit::btc_perpetual().unwrap(),
            SettlementMethod::Cash,
            source()
        ),
        Err(FutureError::NotAFuture {
            kind: ContractKind::Perpetual,
            ..
        })
    ));
    assert!(matches!(
        FutureTerms::new(
            binance_usdm::btcusdt_perpetual().unwrap(),
            SettlementMethod::Cash,
            source()
        ),
        Err(FutureError::NotAFuture { .. })
    ));
    // An option has an expiry too.
    assert!(matches!(
        FutureTerms::new(
            deribit::btc_option(
                id("BTC-4NOV26-60000-C"),
                EXPIRY_MS,
                dec!(60000),
                OptionStyle::Call
            )
            .unwrap(),
            SettlementMethod::Cash,
            source()
        ),
        Err(FutureError::NotAFuture {
            kind: ContractKind::Option,
            ..
        })
    ));
    assert_eq!(SettlementSource::new(" "), Err(FutureError::EmptySource));
    let terms = inverse();
    assert_eq!(terms.expiry_ms(), EXPIRY_MS);
    assert_eq!(terms.method(), SettlementMethod::Cash);
    assert_eq!(terms.source().as_str(), SOURCE);
}

#[test]
fn the_expiry_itself_is_not_trading() {
    let terms = inverse();
    assert_eq!(terms.phase(EXPIRY_MS - 1), FuturePhase::Trading);
    assert_eq!(terms.phase(EXPIRY_MS), FuturePhase::AwaitingSettlement);
    assert_eq!(terms.phase(EXPIRY_MS + 1), FuturePhase::AwaitingSettlement);
    assert_eq!(terms.phase(u64::MAX), FuturePhase::AwaitingSettlement);
}

#[test]
fn a_settlement_price_must_declare_this_contract_source_and_expiry() {
    let terms = inverse();
    let make = |instrument: &str, from: &str, expiry_ms, published_ms| {
        SettlementPrice::new(
            id(instrument),
            SettlementSource::new(from).unwrap(),
            expiry_ms,
            DELIVERY,
            published_ms,
        )
        .unwrap()
    };
    let good = make("BTC-4NOV26", SOURCE, EXPIRY_MS, RECORD_MS);
    assert_eq!(terms.check_settlement(&good), Ok(()));
    // Published exactly at the expiry, and a millisecond before it.
    let at = make("BTC-4NOV26", SOURCE, EXPIRY_MS, EXPIRY_MS);
    assert_eq!(terms.check_settlement(&at), Ok(()));
    let before = make("BTC-4NOV26", SOURCE, EXPIRY_MS, EXPIRY_MS - 1);
    assert!(matches!(
        terms.check_settlement(&before),
        Err(FutureError::PublishedBeforeExpiry { .. })
    ));
    let other = make("BTC-5NOV26", SOURCE, EXPIRY_MS, RECORD_MS);
    assert!(matches!(
        terms.check_settlement(&other),
        Err(FutureError::WrongInstrument { .. })
    ));
    let mark = make("BTC-4NOV26", "deribit:mark", EXPIRY_MS, RECORD_MS);
    assert!(matches!(
        terms.check_settlement(&mark),
        Err(FutureError::WrongSource { .. })
    ));
    // Published after this expiry, but another day's price.
    let other_day = make("BTC-4NOV26", SOURCE, EXPIRY_MS - 86_400_000, RECORD_MS);
    assert!(matches!(
        terms.check_settlement(&other_day),
        Err(FutureError::WrongExpiry { .. })
    ));
    for price in [dec!(0), dec!(-1)] {
        assert_eq!(
            SettlementPrice::new(id("BTC-4NOV26"), source(), EXPIRY_MS, price, RECORD_MS),
            Err(FutureError::NotPositivePrice { price })
        );
    }
    assert_eq!(good.instrument(), &id("BTC-4NOV26"));
    assert_eq!(good.source().as_str(), SOURCE);
    assert_eq!(good.expiry_ms(), EXPIRY_MS);
    assert_eq!(good.price(), DELIVERY);
    assert_eq!(good.published_ms(), RECORD_MS);
}

#[test]
fn a_settlement_price_is_not_on_the_order_grid() {
    // 78044.4 is off the 2.5 tick of the contract, and is what the venue delivered.
    let terms = inverse();
    assert!(terms.spec().check_price(DELIVERY).is_err());
    let price = print(&terms, DELIVERY, RECORD_MS);
    assert_eq!(terms.check_settlement(&price), Ok(()));
    assert!(
        terms
            .cash_settlement(Side::Long, dec!(1000), dec!(77000), &price)
            .is_ok()
    );
}

#[test]
fn a_linear_cash_settlement_is_its_pnl_in_the_settlement_asset() {
    // 5 contracts of 0.01 BTC from 60000 to 61000: 50 USDT, with no notional in it.
    let terms = linear();
    let price = print(&terms, dec!(61000), RECORD_MS);
    let long = terms
        .cash_settlement(Side::Long, dec!(5), dec!(60000), &price)
        .unwrap();
    assert!(
        long.pnl
            .ratio
            .equals(&Ratio::from_decimal(dec!(50)))
            .unwrap()
    );
    assert_eq!(long.pnl.asset.as_str(), "USDT");
    assert_eq!(long.price, dec!(61000));
    assert_eq!(long.expiry_ms, EXPIRY_MS);
    assert_eq!(long.published_ms, RECORD_MS);
    let short = terms
        .cash_settlement(Side::Short, dec!(5), dec!(60000), &price)
        .unwrap();
    assert!(
        short
            .pnl
            .ratio
            .equals(&Ratio::from_decimal(dec!(-50)))
            .unwrap()
    );
    assert_eq!(short.pnl.asset.as_str(), "USDT");
}

#[test]
fn an_inverse_cash_settlement_is_exact_and_in_the_underlying() {
    // 1000 contracts of 10 USD from 77000 to 78044.4:
    // 10000 x 1044.4 / (77000 x 78044.4) BTC, which is 3730 / 2146221.
    let terms = inverse();
    let price = print(&terms, DELIVERY, RECORD_MS);
    let long = terms
        .cash_settlement(Side::Long, dec!(1000), dec!(77000), &price)
        .unwrap();
    let exact = Ratio::new(dec!(10444000), dec!(6009418800)).unwrap();
    assert!(long.pnl.ratio.equals(&exact).unwrap());
    assert_eq!(long.pnl.asset.as_str(), "BTC");
    let settled = long.pnl.settle(8, RoundingRule::HalfEven).unwrap();
    assert_eq!(settled.amount.amount, dec!(0.00173794));
    let short = terms
        .cash_settlement(Side::Short, dec!(1000), dec!(77000), &price)
        .unwrap();
    let settled = short.pnl.settle(8, RoundingRule::HalfEven).unwrap();
    assert_eq!(settled.amount.amount, dec!(-0.00173794));
    // A formula's refusal, and a price refused before it.
    assert!(matches!(
        terms.cash_settlement(Side::Long, dec!(0), dec!(77000), &price),
        Err(FutureError::Instrument(_))
    ));
    assert!(matches!(
        terms.cash_settlement(Side::Long, dec!(1000), dec!(0), &price),
        Err(FutureError::Instrument(_))
    ));
    let early = print(&terms, DELIVERY, EXPIRY_MS - 1);
    assert!(matches!(
        terms.cash_settlement(Side::Long, dec!(1000), dec!(77000), &early),
        Err(FutureError::PublishedBeforeExpiry { .. })
    ));
}

#[test]
fn a_calendar_orders_declared_contracts_of_one_product() {
    let at = |name, expiry| terms_of(linear_spec(name, "BTC", expiry));
    let calendar = FutureCalendar::new(vec![
        at("C", 3000),
        at("B", 2000),
        at("A", 2000),
        at("D", 1000),
    ])
    .unwrap();
    let names = |terms: Vec<&FutureTerms>| -> Vec<String> {
        terms
            .iter()
            .map(|c| c.spec().id().as_str().to_string())
            .collect()
    };
    assert_eq!(
        names(calendar.contracts().iter().collect()),
        ["D", "A", "B", "C"]
    );
    assert_eq!(
        names(calendar.trading_at(999).collect()),
        ["D", "A", "B", "C"]
    );
    // At its expiry a contract is no longer trading.
    assert_eq!(names(calendar.trading_at(1000).collect()), ["A", "B", "C"]);
    assert_eq!(names(calendar.trading_at(2000).collect()), ["C"]);
    assert_eq!(
        names(calendar.trading_at(3000).collect()),
        Vec::<String>::new()
    );

    assert_eq!(
        FutureCalendar::new(vec![at("A", 1000), at("A", 2000)]),
        Err(FutureError::DuplicateContract {
            instrument: "A".to_string()
        })
    );
    assert!(matches!(
        FutureCalendar::new(vec![at("A", 1000), terms_of(linear_spec("E", "ETH", 2000))]),
        Err(FutureError::OtherProduct { .. })
    ));
    // The same underlying and price asset, settled in another asset.
    let mut settled_in_btc = InstrumentDraft::from(inverse().spec().clone());
    settled_in_btc.id = id("BTC-USDT-4NOV26");
    settled_in_btc.price_asset = AssetCode::new("USDT").unwrap();
    settled_in_btc.size =
        ContractSize::quote_value(dec!(10), AssetCode::new("USDT").unwrap()).unwrap();
    let settled_in_btc = terms_of(InstrumentSpec::new(settled_in_btc).unwrap());
    assert!(matches!(
        FutureCalendar::new(vec![at("A", 1000), settled_in_btc.clone()]),
        Err(FutureError::OtherProduct { .. })
    ));
    // The same underlying and settlement asset, priced in another asset.
    assert!(matches!(
        FutureCalendar::new(vec![inverse(), settled_in_btc]),
        Err(FutureError::OtherProduct { .. })
    ));
    assert!(
        FutureCalendar::new(Vec::new())
            .unwrap()
            .contracts()
            .is_empty()
    );
}

#[cfg(feature = "margin")]
mod ledger {
    use super::*;
    use margin_rs::instrument::InstrumentScales;
    use margin_rs::ledger::{Execution, FundingPayment, Ledger, LedgerConfig};
    use margin_rs::{Action, Asset, ExecutionId, Instrument, InstrumentId as LedgerId, TransferId};

    fn ledger_with(terms: &FutureTerms, price_scale: u32, cash: Decimal) -> Ledger {
        let settlement = Asset::new(terms.spec().settlement_asset().as_str()).unwrap();
        let mut ledger = Ledger::new(LedgerConfig::default());
        ledger.register_asset(settlement.clone(), 8).unwrap();
        ledger
            .register_instrument(
                Instrument::from_spec(
                    terms.spec(),
                    InstrumentScales {
                        price_scale,
                        contract_scale: 0,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        ledger
            .deposit(TransferId::new("seed").unwrap(), settlement, cash)
            .unwrap();
        ledger
    }

    fn own(ledger: &mut Ledger, terms: &FutureTerms, side: Action, price: Decimal, size: Decimal) {
        ledger
            .book_execution_allowing_overdraft(
                &Execution::new(
                    ExecutionId::new("fill").unwrap(),
                    LedgerId::new(terms.spec().id().as_str()).unwrap(),
                    side,
                    price,
                    size,
                )
                .with_executed_at(EXPIRY_MS - 1),
                None,
            )
            .unwrap();
    }

    fn cash(ledger: &Ledger, terms: &FutureTerms) -> Decimal {
        let settlement = Asset::new(terms.spec().settlement_asset().as_str()).unwrap();
        ledger.balance(&settlement).unwrap().total
    }

    fn position(ledger: &Ledger, terms: &FutureTerms) -> Decimal {
        ledger
            .position(&LedgerId::new(terms.spec().id().as_str()).unwrap())
            .unwrap()
            .quantity
    }

    #[test]
    fn long_and_short_settle_in_their_asset_at_the_real_delivery() {
        // The entry rests at 10000 / 77000 = 0.12987013 BTC, and 10000 / 78044.4 at
        // 0.12813219: 0.00173794, from the ledger's two rounded rests.
        for (side, size, pnl) in [
            (Action::Buy, dec!(1000), dec!(0.00173794)),
            (Action::Sell, dec!(-1000), dec!(-0.00173794)),
        ] {
            let terms = inverse();
            let mut ledger = ledger_with(&terms, 2, dec!(1));
            own(&mut ledger, &terms, side, dec!(77000), size.abs());
            let price = print(&terms, DELIVERY, RECORD_MS);
            assert_eq!(
                terms.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
                FuturePhase::AwaitingSettlement
            );
            let booked = terms.settle_in_ledger(&mut ledger, &price).unwrap();
            assert!(booked.is_new);
            assert_eq!(booked.cashflow.contracts, size);
            assert_eq!(booked.cashflow.realized_pnl, pnl);
            assert_eq!(booked.cashflow.asset.as_str(), "BTC");
            assert_eq!(booked.cashflow.price, DELIVERY);
            assert_eq!(position(&ledger, &terms), dec!(0));
            assert_eq!(cash(&ledger, &terms), dec!(1) + pnl);
            assert_eq!(
                terms.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
                FuturePhase::Settled
            );
        }
        for (side, size, pnl) in [
            (Action::Buy, dec!(5), dec!(50)),
            (Action::Sell, dec!(-5), dec!(-50)),
        ] {
            let terms = linear();
            let mut ledger = ledger_with(&terms, 2, dec!(1000));
            own(&mut ledger, &terms, side, dec!(60000), size.abs());
            let price = print(&terms, dec!(61000), RECORD_MS);
            let booked = terms.settle_in_ledger(&mut ledger, &price).unwrap();
            assert_eq!(booked.cashflow.realized_pnl, pnl);
            assert_eq!(booked.cashflow.asset.as_str(), "USDT");
            assert_eq!(cash(&ledger, &terms), dec!(1000) + pnl);
        }
    }

    #[test]
    fn with_no_settlement_nothing_changes_and_it_is_not_settled() {
        let terms = linear();
        let mut ledger = ledger_with(&terms, 2, dec!(1000));
        own(&mut ledger, &terms, Action::Buy, dec!(60000), dec!(5));
        let before = ledger.snapshot();
        assert_eq!(
            terms.phase_in_ledger(EXPIRY_MS - 1, &ledger).unwrap(),
            FuturePhase::Trading
        );
        assert_eq!(
            terms.phase_in_ledger(EXPIRY_MS, &ledger).unwrap(),
            FuturePhase::AwaitingSettlement
        );
        // Refused prices book nothing and leave it waiting.
        let early = print(&terms, dec!(61000), EXPIRY_MS - 1);
        assert!(matches!(
            terms.settle_in_ledger(&mut ledger, &early),
            Err(FutureError::PublishedBeforeExpiry { .. })
        ));
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(
            terms.phase_in_ledger(EXPIRY_MS, &ledger).unwrap(),
            FuturePhase::AwaitingSettlement
        );
    }

    #[test]
    fn a_price_finer_than_the_explicit_scale_is_refused_with_the_ledger_unchanged() {
        // The real 77956.81 has two digits, and this instrument's scale is one, which
        // is all its 2.5 tick needs. 78044.4 fits the same scale.
        let terms = inverse();
        let mut ledger = ledger_with(&terms, 1, dec!(1));
        own(&mut ledger, &terms, Action::Buy, dec!(77000), dec!(1000));
        let before = ledger.snapshot();
        let finer = print(&terms, dec!(77956.81), RECORD_MS);
        assert!(matches!(
            terms.settle_in_ledger(&mut ledger, &finer),
            Err(FutureError::Ledger(_))
        ));
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(
            terms.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
            FuturePhase::AwaitingSettlement
        );
        let fits = print(&terms, DELIVERY, RECORD_MS);
        assert!(terms.settle_in_ledger(&mut ledger, &fits).unwrap().is_new);
    }

    #[test]
    fn settlement_is_terminal_a_replay_changes_nothing_and_a_new_price_is_refused() {
        let terms = linear();
        let mut ledger = ledger_with(&terms, 2, dec!(1000));
        own(&mut ledger, &terms, Action::Buy, dec!(60000), dec!(5));
        let price = print(&terms, dec!(61000), RECORD_MS);
        let first = terms.settle_in_ledger(&mut ledger, &price).unwrap();
        assert!(first.is_new);
        let after = ledger.snapshot();

        let replay = terms.settle_in_ledger(&mut ledger, &price).unwrap();
        assert!(!replay.is_new);
        assert_eq!(replay.cashflow, first.cashflow);
        assert_eq!(ledger.snapshot(), after);

        for other in [
            print(&terms, dec!(61001), RECORD_MS),
            print(&terms, dec!(61000), RECORD_MS + 1),
        ] {
            assert!(matches!(
                terms.settle_in_ledger(&mut ledger, &other),
                Err(FutureError::Ledger(
                    margin_rs::MarginError::ConflictingEntry { .. }
                ))
            ));
            assert_eq!(ledger.snapshot(), after);
        }
        assert_eq!(cash(&ledger, &terms), dec!(1050));
    }

    #[test]
    fn a_flat_position_settles_with_no_cash() {
        let terms = linear();
        let mut ledger = ledger_with(&terms, 2, dec!(1000));
        let price = print(&terms, dec!(61000), RECORD_MS);
        let booked = terms.settle_in_ledger(&mut ledger, &price).unwrap();
        assert_eq!(booked.cashflow.contracts, dec!(0));
        assert_eq!(booked.cashflow.realized_pnl, dec!(0));
        assert_eq!(cash(&ledger, &terms), dec!(1000));
        assert_eq!(
            terms.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
            FuturePhase::Settled
        );
    }

    #[test]
    fn the_ledger_must_hold_this_contract() {
        let terms = linear();
        let price = print(&terms, dec!(61000), RECORD_MS);
        // Not registered.
        let mut empty = Ledger::new(LedgerConfig::default());
        assert!(matches!(
            terms.settle_in_ledger(&mut empty, &price),
            Err(FutureError::Ledger(
                margin_rs::MarginError::UnknownInstrument { .. }
            ))
        ));
        // The same identifier with another expiry and another multiplier.
        for other in [linear_spec("FUT", "BTC", EXPIRY_MS + 1), {
            let mut draft = InstrumentDraft::from(linear_spec("FUT", "BTC", EXPIRY_MS));
            draft.size = ContractSize::base(dec!(0.1), AssetCode::new("BTC").unwrap()).unwrap();
            InstrumentSpec::new(draft).unwrap()
        }] {
            let mut ledger = ledger_with(&terms_of(other), 2, dec!(1000));
            let before = ledger.snapshot();
            assert_eq!(
                terms.settle_in_ledger(&mut ledger, &price).unwrap_err(),
                FutureError::OtherTermsInLedger {
                    instrument: "FUT".to_string()
                }
            );
            assert_eq!(ledger.snapshot(), before);
        }
    }

    #[test]
    fn a_phase_is_only_read_from_a_ledger_holding_this_contract() {
        let settled = linear();
        let mut ledger = ledger_with(&settled, 2, dec!(1000));
        let price = print(&settled, dec!(61000), RECORD_MS);
        settled.settle_in_ledger(&mut ledger, &price).unwrap();
        assert_eq!(
            settled.phase_in_ledger(EXPIRY_MS - 1, &ledger).unwrap(),
            FuturePhase::Settled
        );
        // The same identifier, a day later: that settlement is not of this contract.
        let later = terms_of(linear_spec("FUT", "BTC", EXPIRY_MS + 86_400_000));
        assert_eq!(
            later.phase_in_ledger(EXPIRY_MS, &ledger).unwrap_err(),
            FutureError::OtherTermsInLedger {
                instrument: "FUT".to_string()
            }
        );
        // A ledger without it leaves the clock.
        let empty = Ledger::new(LedgerConfig::default());
        assert_eq!(
            settled.phase_in_ledger(EXPIRY_MS - 1, &empty).unwrap(),
            FuturePhase::Trading
        );
        assert_eq!(
            settled.phase_in_ledger(EXPIRY_MS, &empty).unwrap(),
            FuturePhase::AwaitingSettlement
        );
    }

    #[test]
    fn two_contracts_in_one_ledger_settle_each_at_its_own_price() {
        let near = terms_of(linear_spec("NEAR", "BTC", EXPIRY_MS));
        let far = terms_of(linear_spec("FAR", "BTC", EXPIRY_MS + 1000));
        let mut ledger = ledger_with(&near, 2, dec!(1000));
        ledger
            .register_instrument(
                Instrument::from_spec(
                    far.spec(),
                    InstrumentScales {
                        price_scale: 2,
                        contract_scale: 0,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let fill = |id: &str, name: &str| {
            Execution::new(
                ExecutionId::new(id).unwrap(),
                LedgerId::new(name).unwrap(),
                Action::Buy,
                dec!(60000),
                dec!(5),
            )
            .with_executed_at(EXPIRY_MS - 1)
        };
        for execution in [fill("a", "NEAR"), fill("b", "FAR")] {
            ledger
                .book_execution_allowing_overdraft(&execution, None)
                .unwrap();
        }
        // 5 x 0.01 x 1000 = 50, then 5 x 0.01 x -1000 = -50 on the other.
        let near_price = print(&near, dec!(61000), RECORD_MS);
        let first = near.settle_in_ledger(&mut ledger, &near_price).unwrap();
        assert!(first.is_new);
        assert_eq!(first.cashflow.realized_pnl, dec!(50));
        assert_eq!(cash(&ledger, &near), dec!(1050));
        assert_eq!(position(&ledger, &near), dec!(0));
        assert_eq!(
            near.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
            FuturePhase::Settled
        );
        // Settling one says nothing about the other.
        assert_eq!(
            far.phase_in_ledger(RECORD_MS, &ledger).unwrap(),
            FuturePhase::Trading
        );
        assert_eq!(position(&ledger, &far), dec!(5));
        let far_price = print(&far, dec!(59000), RECORD_MS + 1000);
        let second = far.settle_in_ledger(&mut ledger, &far_price).unwrap();
        assert!(second.is_new);
        assert_eq!(second.cashflow.realized_pnl, dec!(-50));
        assert_eq!(cash(&ledger, &near), dec!(1000));
        assert_eq!(position(&ledger, &far), dec!(0));
        assert_eq!(
            far.phase_in_ledger(RECORD_MS + 1000, &ledger).unwrap(),
            FuturePhase::Settled
        );
        // Each replay is its own duplicate, and changes nothing.
        let after = ledger.snapshot();
        assert!(
            !near
                .settle_in_ledger(&mut ledger, &near_price)
                .unwrap()
                .is_new
        );
        assert!(
            !far.settle_in_ledger(&mut ledger, &far_price)
                .unwrap()
                .is_new
        );
        assert_eq!(ledger.snapshot(), after);
    }

    #[test]
    fn a_future_never_receives_funding() {
        let terms = linear();
        let mut ledger = ledger_with(&terms, 2, dec!(1000));
        own(&mut ledger, &terms, Action::Buy, dec!(60000), dec!(5));
        let before = ledger.snapshot();
        let paid = ledger.book_funding(&FundingPayment {
            id: ExecutionId::new("funding").unwrap(),
            instrument: LedgerId::new("FUT").unwrap(),
            amount: dec!(1),
            as_of_ms: EXPIRY_MS - 1,
        });
        assert!(paid.is_err());
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(ledger.funding(&Asset::new("USDT").unwrap()), dec!(0));
        let price = print(&terms, dec!(61000), RECORD_MS);
        let booked = terms.settle_in_ledger(&mut ledger, &price).unwrap();
        assert_eq!(booked.cashflow.realized_pnl, dec!(50));
        assert_eq!(ledger.funding(&Asset::new("USDT").unwrap()), dec!(0));
    }
}
