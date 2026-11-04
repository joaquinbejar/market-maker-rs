//! Orders and fills against a contract specification (#93): hand cases, and
//! the shared `financial_types` fixtures through this crate's own paths.
//!
//! Every shared case has an explicit disposition, keyed by its name and
//! operation; an unknown or missing case fails the test, so a new upstream case
//! must be classified here. The expected values of the hand cases are literals
//! from the contract terms, never recomputed with the code under test.

#![cfg(feature = "instruments")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::execution::{Action, Fill, OrderId, OrderRequest, OrderType, TimeInForce};
use market_maker_rs::instruments::MoneyRatio;
use market_maker_rs::instruments::orders::{OrderSpecError, check_order, fill_notional};
use market_maker_rs::instruments::reference::{binance_usdm, deribit};
use market_maker_rs::instruments::{InstrumentError, InstrumentId, InstrumentSpec, Ratio};
use market_maker_rs::prelude::OptionStyle;

const EXPIRY_MS: u64 = 1_795_766_400_000;

fn binance() -> InstrumentSpec {
    binance_usdm::btcusdt_perpetual().unwrap()
}

fn deribit_perpetual() -> InstrumentSpec {
    deribit::btc_perpetual().unwrap()
}

fn deribit_option() -> InstrumentSpec {
    deribit::btc_option(
        InstrumentId::new("BTC-27NOV26-100000-C").unwrap(),
        EXPIRY_MS,
        dec!(100000),
        OptionStyle::Call,
    )
    .unwrap()
}

fn fill(side: Action, price: Decimal, quantity: Decimal) -> Fill {
    Fill {
        symbol: String::new(),
        order_id: OrderId::new("o-1"),
        trade_id: "t-1".to_string(),
        price,
        quantity,
        side,
        timestamp: 1_000,
        fee: Decimal::ZERO,
        fee_currency: "USDT".to_string(),
    }
}

/// `fill_notional` of `f` named for `spec`, for the cases where the instrument
/// is not what is under test.
fn notional_of(spec: &InstrumentSpec, f: &Fill) -> Result<MoneyRatio, OrderSpecError> {
    let named = Fill {
        symbol: spec.id().as_str().to_string(),
        ..f.clone()
    };
    fill_notional(spec, &named)
}

fn is_spec_error(result: Result<(), OrderSpecError>, want: fn(&InstrumentError) -> bool) -> bool {
    matches!(result, Err(OrderSpecError::Spec(ref e)) if want(e))
}

#[test]
fn check_order_accepts_what_is_on_the_grid() {
    let spec = binance();
    let limit = OrderRequest::limit_buy("BTCUSDT", dec!(60000.1), dec!(0.001));
    assert_eq!(check_order(&spec, &limit), Ok(()));
    let sell = OrderRequest::limit_sell("BTCUSDT", dec!(60000.1), dec!(0.5));
    assert_eq!(check_order(&spec, &sell), Ok(()));
    let market = OrderRequest::market_sell("BTCUSDT", dec!(0.001));
    assert_eq!(check_order(&spec, &market), Ok(()));
    let mut post_only = limit;
    post_only.order_type = OrderType::PostOnly;
    post_only.time_in_force = TimeInForce::GoodTilCancel;
    assert_eq!(check_order(&spec, &post_only), Ok(()));
    // The size is in contracts: whole ones, on Deribit.
    let contracts = OrderRequest::limit_buy("BTC-PERPETUAL", dec!(60000.5), dec!(3));
    assert_eq!(check_order(&deribit_perpetual(), &contracts), Ok(()));
}

#[test]
fn check_order_refuses_another_instrument() {
    let order = OrderRequest::limit_buy("BTC-PERPETUAL", dec!(60000.1), dec!(0.001));
    assert_eq!(
        check_order(&binance(), &order),
        Err(OrderSpecError::SymbolMismatch {
            symbol: "BTC-PERPETUAL".to_string(),
            instrument: "BTCUSDT".to_string(),
        })
    );
    // Compared exactly.
    let lower = OrderRequest::limit_buy("btcusdt", dec!(60000.1), dec!(0.001));
    assert!(matches!(
        check_order(&binance(), &lower),
        Err(OrderSpecError::SymbolMismatch { .. })
    ));
}

#[test]
fn check_order_refuses_a_side_other_than_buy_or_sell() {
    let mut order = OrderRequest::limit_buy("BTCUSDT", dec!(60000.1), dec!(0.001));
    order.side = Action::Other;
    assert_eq!(
        check_order(&binance(), &order),
        Err(OrderSpecError::UnsupportedSide {
            side: Action::Other
        })
    );
}

#[test]
fn check_order_refuses_a_price_that_does_not_fit_the_order_type() {
    let spec = binance();
    for order_type in [OrderType::Limit, OrderType::PostOnly] {
        let mut order = OrderRequest::limit_buy("BTCUSDT", dec!(60000.1), dec!(0.001));
        order.order_type = order_type;
        order.price = None;
        assert_eq!(
            check_order(&spec, &order),
            Err(OrderSpecError::MissingPrice { order_type })
        );
    }
    let mut market = OrderRequest::market_buy("BTCUSDT", dec!(0.001));
    market.price = Some(dec!(60000.1));
    assert_eq!(
        check_order(&spec, &market),
        Err(OrderSpecError::UnexpectedPrice {
            price: dec!(60000.1)
        })
    );
}

#[test]
fn check_order_refuses_what_is_off_the_grid_without_rounding() {
    let spec = binance();
    let not_on_grid: fn(&InstrumentError) -> bool =
        |e| matches!(e, InstrumentError::NotOnGrid { .. });
    let not_positive: fn(&InstrumentError) -> bool =
        |e| matches!(e, InstrumentError::NotPositive { .. });
    let off_tick = OrderRequest::limit_buy("BTCUSDT", dec!(60000.15), dec!(0.001));
    assert!(is_spec_error(check_order(&spec, &off_tick), not_on_grid));
    let off_lot = OrderRequest::limit_buy("BTCUSDT", dec!(60000.1), dec!(0.0015));
    assert!(is_spec_error(check_order(&spec, &off_lot), not_on_grid));
    // A market order has no price but still has a lot.
    let market_off_lot = OrderRequest::market_buy("BTCUSDT", dec!(0.0015));
    assert!(is_spec_error(
        check_order(&spec, &market_off_lot),
        not_on_grid
    ));
    let zero = OrderRequest::limit_buy("BTCUSDT", dec!(60000.1), dec!(0));
    assert!(is_spec_error(check_order(&spec, &zero), not_positive));
    let negative = OrderRequest::limit_buy("BTCUSDT", dec!(-1), dec!(0.001));
    assert!(is_spec_error(check_order(&spec, &negative), not_positive));
}

#[test]
fn fill_notional_is_in_the_asset_of_the_contract() {
    // Linear: 0.5 BTC x 60000.1 USDT, where `Fill::notional` happens to agree.
    let linear = fill(Action::Buy, dec!(60000.1), dec!(0.5));
    let notional = notional_of(&binance(), &linear).unwrap();
    assert_eq!(notional.asset.as_str(), "USDT");
    assert!(
        notional
            .ratio
            .equals(&Ratio::from_decimal(dec!(30000.05)))
            .unwrap()
    );
    assert_eq!(linear.notional(), dec!(30000.05));

    // Inverse: 1000 contracts of 10 USD at 60000 is 1/6 BTC, not the 60 million
    // that `Fill::notional` gives.
    let inverse = fill(Action::Sell, dec!(60000), dec!(1000));
    let notional = notional_of(&deribit_perpetual(), &inverse).unwrap();
    assert_eq!(notional.asset.as_str(), "BTC");
    assert!(
        notional
            .ratio
            .equals(&Ratio::new(dec!(1), dec!(6)).unwrap())
            .unwrap()
    );
    assert_eq!(inverse.notional(), dec!(60000000));
}

#[test]
fn fill_notional_refuses_an_option_and_what_is_not_positive() {
    let premium = fill(Action::Buy, dec!(0.05), dec!(1));
    assert_eq!(
        notional_of(&deribit_option(), &premium).unwrap_err(),
        OrderSpecError::OptionNotional {
            instrument: "BTC-27NOV26-100000-C".to_string(),
        }
    );
    for bad in [
        fill(Action::Buy, dec!(0), dec!(1)),
        fill(Action::Buy, dec!(60000), dec!(0)),
    ] {
        assert!(matches!(
            notional_of(&binance(), &bad),
            Err(OrderSpecError::Spec(InstrumentError::NotPositive { .. }))
        ));
    }
}

#[test]
fn fill_notional_does_not_apply_the_grid_to_a_fill() {
    // Off the tick and the lot of the snapshot: it already happened.
    let off = fill(Action::Buy, dec!(60000.15), dec!(0.0015));
    let notional = notional_of(&binance(), &off).unwrap();
    assert!(
        notional
            .ratio
            .equals(&Ratio::from_decimal(dec!(90.000225)))
            .unwrap()
    );
}

#[test]
fn fill_notional_refuses_a_fill_of_another_instrument() {
    let mut other = fill(Action::Buy, dec!(60000), dec!(1000));
    other.symbol = "BTC-PERPETUAL".to_string();
    // The right amount, for the instrument it names.
    assert!(fill_notional(&deribit_perpetual(), &other).is_ok());
    // On another specification it would be in the wrong asset.
    assert_eq!(
        fill_notional(&binance(), &other).unwrap_err(),
        OrderSpecError::SymbolMismatch {
            symbol: "BTC-PERPETUAL".to_string(),
            instrument: "BTCUSDT".to_string(),
        }
    );
    // Compared exactly, and no symbol is not a match.
    other.symbol = "btcusdt".to_string();
    assert!(matches!(
        fill_notional(&binance(), &other),
        Err(OrderSpecError::SymbolMismatch { .. })
    ));
    other.symbol = String::new();
    assert!(matches!(
        fill_notional(&binance(), &other),
        Err(OrderSpecError::SymbolMismatch { .. })
    ));
}

/// A record written before `Fill::symbol`.
#[cfg(feature = "serde")]
fn fill_recorded_before_the_symbol() -> Fill {
    let old = r#"{"order_id":"o-1","trade_id":"t-1","price":"60000","quantity":"1",
        "side":"Buy","timestamp":1000,"fee":"0","fee_currency":"USDT"}"#;
    let decoded: Fill = serde_json::from_str(old).unwrap();
    assert_eq!(decoded.symbol, "");
    decoded
}

#[cfg(feature = "serde")]
#[test]
fn a_fill_recorded_before_the_symbol_has_no_notional() {
    assert!(matches!(
        fill_notional(&binance(), &fill_recorded_before_the_symbol()),
        Err(OrderSpecError::SymbolMismatch { .. })
    ));
}

#[cfg(feature = "margin")]
mod ledger {
    use super::*;
    use financial_types::Side;
    use financial_types::instrument::fixtures::{self, Case, ErrorKind, Expected, Operation};
    use margin_rs::ledger::{Execution, ExecutionCashflow, Ledger, LedgerConfig};
    use margin_rs::{Asset, MarginError};
    use market_maker_rs::instruments::margin::{
        AdapterError, MarginScales, fill_execution, margin_instrument,
    };
    use market_maker_rs::instruments::{
        AssetCode, ContractKind, ContractSize, InstrumentDraft, Margining, MoneyRatio, TickSchedule,
    };
    use std::collections::{BTreeMap, BTreeSet};

    /// `fill_execution` of `f` named for `spec`, as [`notional_of`].
    fn execution_of(spec: &InstrumentSpec, f: &Fill) -> Result<Execution, AdapterError> {
        let named = Fill {
            symbol: spec.id().as_str().to_string(),
            ..f.clone()
        };
        fill_execution(spec, &named)
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

    fn btc() -> Asset {
        Asset::new("BTC").unwrap()
    }

    fn usdt() -> Asset {
        Asset::new("USDT").unwrap()
    }

    /// A ledger with BTC and USDT at 8 decimals and `spec` at scales of 8.
    fn ledger_for(spec: &InstrumentSpec) -> Ledger {
        let mut ledger = Ledger::new(LedgerConfig::default());
        ledger.register_asset(btc(), 8).unwrap();
        ledger.register_asset(usdt(), 8).unwrap();
        let scales = MarginScales {
            price_scale: 8,
            contract_scale: 8,
        };
        ledger
            .register_instrument(margin_instrument(spec, scales).unwrap())
            .unwrap();
        ledger
    }

    fn book(ledger: &mut Ledger, spec: &InstrumentSpec, f: &Fill) -> ExecutionCashflow {
        let execution = execution_of(spec, f).unwrap();
        let booked = ledger
            .book_execution_allowing_overdraft(&execution, None)
            .unwrap();
        booked.booking.cashflow().clone()
    }

    fn at(id: &str, timestamp: u64, side: Action, price: Decimal, quantity: Decimal) -> Fill {
        Fill {
            trade_id: id.to_string(),
            timestamp,
            ..fill(side, price, quantity)
        }
    }

    fn total(ledger: &Ledger, asset: &Asset) -> Decimal {
        ledger.balance(asset).unwrap().total
    }

    #[test]
    fn fill_execution_carries_the_fill_exactly() {
        let mut f = fill(Action::Sell, dec!(60000.1), dec!(0.5));
        f.fee = dec!(1.25);
        let e = execution_of(&binance(), &f).unwrap();
        assert_eq!(e.id.as_str(), "7:BTCUSDT:t-1");
        assert_eq!(e.instrument.as_str(), "BTCUSDT");
        assert_eq!(e.side, Action::Sell);
        assert_eq!((e.price, e.quantity), (dec!(60000.1), dec!(0.5)));
        assert_eq!(e.executed_at_ms, Some(1_000));
        let fee = e.fee.unwrap();
        assert_eq!((fee.asset, fee.amount), (usdt(), dec!(1.25)));

        // No fee, whatever the currency says.
        let mut free = fill(Action::Buy, dec!(60000.1), dec!(0.5));
        free.fee_currency = String::new();
        assert!(execution_of(&binance(), &free).unwrap().fee.is_none());

        // A rebate keeps its sign.
        let mut rebate = fill(Action::Buy, dec!(60000.1), dec!(0.5));
        rebate.fee = dec!(-0.1);
        let fee = execution_of(&binance(), &rebate).unwrap().fee.unwrap();
        assert_eq!(fee.amount, dec!(-0.1));

        // Off the grid of the snapshot: it already happened.
        let off = fill(Action::Buy, dec!(60000.15), dec!(0.0015));
        assert_eq!(
            execution_of(&binance(), &off).unwrap().price,
            dec!(60000.15)
        );
    }

    #[test]
    fn fill_execution_refuses_a_fill_of_another_instrument() {
        let mut other = fill(Action::Buy, dec!(60000), dec!(1000));
        other.symbol = "BTC-PERPETUAL".to_string();
        assert_eq!(
            fill_execution(&deribit_perpetual(), &other)
                .unwrap()
                .instrument
                .as_str(),
            "BTC-PERPETUAL"
        );
        assert_eq!(
            fill_execution(&binance(), &other).unwrap_err(),
            AdapterError::SymbolMismatch {
                symbol: "BTC-PERPETUAL".to_string(),
                instrument: "BTCUSDT".to_string(),
            }
        );
        // Before anything else about the fill.
        let mut option_named_wrong = fill(Action::Other, dec!(0), dec!(0));
        option_named_wrong.symbol = "X".to_string();
        assert!(matches!(
            fill_execution(&deribit_option(), &option_named_wrong),
            Err(AdapterError::SymbolMismatch { .. })
        ));
        other.symbol = String::new();
        assert!(matches!(
            fill_execution(&binance(), &other),
            Err(AdapterError::SymbolMismatch { .. })
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_fill_recorded_before_the_symbol_has_no_execution() {
        assert!(matches!(
            fill_execution(&binance(), &fill_recorded_before_the_symbol()),
            Err(AdapterError::SymbolMismatch { .. })
        ));
    }

    #[test]
    fn fill_execution_refuses_what_cannot_be_booked() {
        let spec = binance();
        assert_eq!(
            execution_of(&deribit_option(), &fill(Action::Buy, dec!(0.05), dec!(1))).unwrap_err(),
            AdapterError::Unsupported {
                kind: ContractKind::Option,
                margining: Margining::Inverse,
            }
        );
        assert_eq!(
            execution_of(&spec, &fill(Action::Other, dec!(60000), dec!(1))).unwrap_err(),
            AdapterError::UnsupportedSide {
                side: Action::Other
            }
        );
        assert_eq!(
            execution_of(&spec, &fill(Action::Buy, dec!(0), dec!(1))).unwrap_err(),
            AdapterError::NotPositive {
                field: "price",
                value: dec!(0),
            }
        );
        assert_eq!(
            execution_of(&spec, &fill(Action::Buy, dec!(60000), dec!(-1))).unwrap_err(),
            AdapterError::NotPositive {
                field: "quantity",
                value: dec!(-1),
            }
        );
        let mut no_id = fill(Action::Buy, dec!(60000), dec!(1));
        no_id.trade_id = String::new();
        assert_eq!(
            execution_of(&spec, &no_id).unwrap_err(),
            AdapterError::EmptyTradeId
        );
        // Refused, never shortened or cleaned up.
        for trade_id in ["with space".to_string(), "x".repeat(64)] {
            let mut bad = fill(Action::Buy, dec!(60000), dec!(1));
            bad.trade_id = trade_id;
            assert!(matches!(
                execution_of(&spec, &bad),
                Err(AdapterError::Margin(_))
            ));
        }
        let mut bad_asset = fill(Action::Buy, dec!(60000), dec!(1));
        bad_asset.fee = dec!(1);
        bad_asset.fee_currency = "not an asset".to_string();
        assert!(matches!(
            execution_of(&spec, &bad_asset),
            Err(AdapterError::Margin(_))
        ));
    }

    #[test]
    fn linear_pnl_long_and_short_in_usdt() {
        let spec = binance();
        // 0.5 x (61000.5 - 60000) = 500.25 USDT.
        for (open, close, pnl) in [
            (Action::Buy, Action::Sell, dec!(500.25)),
            (Action::Sell, Action::Buy, dec!(-500.25)),
        ] {
            let mut ledger = ledger_for(&spec);
            book(
                &mut ledger,
                &spec,
                &at("a", 1, open, dec!(60000), dec!(0.5)),
            );
            let closed = book(
                &mut ledger,
                &spec,
                &at("b", 2, close, dec!(61000.5), dec!(0.5)),
            );
            assert_eq!(closed.realized_pnl, pnl);
            assert_eq!(total(&ledger, &usdt()), pnl);
            assert_eq!(total(&ledger, &btc()), dec!(0));
        }
    }

    #[test]
    fn inverse_pnl_long_and_short_in_btc() {
        let spec = deribit_perpetual();
        // 1000 contracts of 10 USD: 10000/60000 - 10000/61000 = 1/366 BTC; each
        // leg rounds at 8 decimals, 0.16666667 and 0.16393443.
        for (open, close, pnl) in [
            (Action::Buy, Action::Sell, dec!(0.00273224)),
            (Action::Sell, Action::Buy, dec!(-0.00273224)),
        ] {
            let mut ledger = ledger_for(&spec);
            book(
                &mut ledger,
                &spec,
                &at("a", 1, open, dec!(60000), dec!(1000)),
            );
            let closed = book(
                &mut ledger,
                &spec,
                &at("b", 2, close, dec!(61000), dec!(1000)),
            );
            assert_eq!(closed.realized_pnl, pnl);
            assert_eq!(total(&ledger, &btc()), pnl);
            assert_eq!(total(&ledger, &usdt()), dec!(0));
        }
    }

    /// A linear future of `multiplier` BTC per contract, or spot for `None`.
    fn linear_spec(multiplier: Option<Decimal>) -> InstrumentSpec {
        let asset = |code| AssetCode::new(code).unwrap();
        InstrumentSpec::new(InstrumentDraft {
            id: InstrumentId::new(if multiplier.is_some() { "FUT" } else { "SPOT" }).unwrap(),
            kind: if multiplier.is_some() {
                ContractKind::Future
            } else {
                ContractKind::Spot
            },
            margining: Margining::Linear,
            underlying: asset("BTC"),
            price_asset: asset("USDT"),
            settlement_asset: asset("USDT"),
            size: ContractSize::base(multiplier.unwrap_or(Decimal::ONE), asset("BTC")).unwrap(),
            price_ticks: TickSchedule::fixed(dec!(1)).unwrap(),
            lot: dec!(1),
            expiry_ms: multiplier.map(|_| EXPIRY_MS),
            option: None,
        })
        .unwrap()
    }

    #[test]
    fn a_contract_is_not_a_unit_of_the_underlying() {
        // 0.01 BTC per contract: 5 contracts at 60000 is 3000 USDT, and a move of
        // 1000 on them is 50 USDT, not 5000.
        let spec = linear_spec(Some(dec!(0.01)));
        let notional = notional_of(&spec, &fill(Action::Buy, dec!(60000), dec!(5))).unwrap();
        assert_eq!(notional.asset.as_str(), "USDT");
        assert!(
            notional
                .ratio
                .equals(&Ratio::from_decimal(dec!(3000)))
                .unwrap()
        );

        let execution = execution_of(&spec, &fill(Action::Buy, dec!(60000), dec!(5))).unwrap();
        assert_eq!(execution.quantity, dec!(5), "contracts, not BTC");

        let mut ledger = ledger_for(&spec);
        book(
            &mut ledger,
            &spec,
            &at("a", 1, Action::Buy, dec!(60000), dec!(5)),
        );
        let closed = book(
            &mut ledger,
            &spec,
            &at("b", 2, Action::Sell, dec!(61000), dec!(5)),
        );
        assert_eq!(closed.realized_pnl, dec!(50));
        assert_eq!(total(&ledger, &usdt()), dec!(50));
        assert_eq!(total(&ledger, &btc()), dec!(0));
    }

    #[test]
    fn spot_moves_both_assets() {
        let spec = linear_spec(None);
        let mut ledger = ledger_for(&spec);
        book(
            &mut ledger,
            &spec,
            &at("a", 1, Action::Buy, dec!(100), dec!(2)),
        );
        assert_eq!(total(&ledger, &btc()), dec!(2));
        assert_eq!(total(&ledger, &usdt()), dec!(-200));
    }

    #[test]
    fn a_fill_at_or_after_the_expiry_is_refused_by_the_ledger() {
        let spec = linear_spec(Some(dec!(0.01)));
        let mut ledger = ledger_for(&spec);
        let before = ledger.snapshot();
        let late = at("a", EXPIRY_MS, Action::Buy, dec!(60000), dec!(5));
        let execution = execution_of(&spec, &late).unwrap();
        let refused = ledger.book_execution_allowing_overdraft(&execution, None);
        assert!(
            matches!(refused, Err(MarginError::ExecutedAfterExpiry { .. })),
            "{refused:?}"
        );
        assert_eq!(ledger.snapshot(), before);
    }

    #[test]
    fn fees_and_rebates_reach_the_ledger_in_their_asset() {
        let spec = binance();
        let mut ledger = ledger_for(&spec);
        let mut paid = at("a", 1, Action::Buy, dec!(60000), dec!(0.5));
        paid.fee = dec!(1.25);
        book(&mut ledger, &spec, &paid);
        assert_eq!(total(&ledger, &usdt()), dec!(-1.25));
        let mut rebate = at("b", 2, Action::Sell, dec!(60000), dec!(0.5));
        rebate.fee = dec!(-0.25);
        book(&mut ledger, &spec, &rebate);
        assert_eq!(total(&ledger, &usdt()), dec!(-1));
    }

    #[test]
    fn a_fee_in_an_unregistered_asset_is_refused_not_lost() {
        let spec = binance();
        let mut ledger = ledger_for(&spec);
        let before = ledger.snapshot();
        let mut f = fill(Action::Buy, dec!(60000), dec!(0.5));
        f.fee = dec!(1);
        f.fee_currency = "ETH".to_string();
        let execution = execution_of(&spec, &f).unwrap();
        let refused = ledger.book_execution_allowing_overdraft(&execution, None);
        assert!(
            matches!(refused, Err(MarginError::UnknownAsset { .. })),
            "{refused:?}"
        );
        assert_eq!(ledger.snapshot(), before);
    }

    #[test]
    fn the_same_trade_is_booked_once() {
        let spec = binance();
        let mut ledger = ledger_for(&spec);
        let f = at("a", 1, Action::Buy, dec!(60000), dec!(0.5));
        book(&mut ledger, &spec, &f);
        let once = ledger.snapshot();
        book(&mut ledger, &spec, &f);
        assert_eq!(ledger.snapshot(), once);
        assert_eq!(ledger.journal().len(), 1);
    }

    #[test]
    fn the_same_trade_id_on_two_instruments_is_booked_twice() {
        // A venue may number trades per symbol, and a ledger keys by id alone.
        let (linear, inverse) = (binance(), deribit_perpetual());
        let mut ledger = ledger_for(&linear);
        let scales = MarginScales {
            price_scale: 8,
            contract_scale: 8,
        };
        ledger
            .register_instrument(margin_instrument(&inverse, scales).unwrap())
            .unwrap();
        let on_linear = at("42", 1, Action::Buy, dec!(60000), dec!(0.5));
        let on_inverse = at("42", 1, Action::Buy, dec!(60000), dec!(1000));
        book(&mut ledger, &linear, &on_linear);
        book(&mut ledger, &inverse, &on_inverse);
        assert_eq!(ledger.journal().len(), 2);

        // Replaying each changes nothing.
        let once = ledger.snapshot();
        book(&mut ledger, &linear, &on_linear);
        book(&mut ledger, &inverse, &on_inverse);
        assert_eq!(ledger.snapshot(), once);
        assert_eq!(ledger.journal().len(), 2);
    }

    #[test]
    fn execution_ids_of_different_pairs_never_coincide() {
        // Without the length, instrument `A` with trade `B:C` and instrument
        // `A:B` with trade `C` would read the same.
        let spec_named = |id: &str| {
            let draft = binance().draft().clone();
            InstrumentSpec::new(InstrumentDraft {
                id: InstrumentId::new(id).unwrap(),
                ..draft
            })
            .unwrap()
        };
        let trade = |id: &str| at(id, 1, Action::Buy, dec!(60000), dec!(0.5));
        let first = execution_of(&spec_named("A"), &trade("B:C")).unwrap();
        let second = execution_of(&spec_named("A:B"), &trade("C")).unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(first.id.as_str(), "1:A:B:C");
        assert_eq!(second.id.as_str(), "3:A:B:C");
    }

    #[test]
    fn a_real_fill_can_still_be_refused_and_says_so() {
        let spec = binance();
        // A fee asset is written exactly.
        let mut lower = fill(Action::Buy, dec!(60000), dec!(0.5));
        lower.fee = dec!(1);
        lower.fee_currency = "usdt".to_string();
        assert!(matches!(
            execution_of(&spec, &lower),
            Err(AdapterError::Margin(_))
        ));
        // Digits above the instrument's scales: the ledger refuses, unchanged.
        let mut ledger = ledger_for(&spec);
        let before = ledger.snapshot();
        let fine = fill(Action::Buy, dec!(60000.123456789), dec!(0.5));
        let execution = execution_of(&spec, &fine).unwrap();
        let refused = ledger.book_execution_allowing_overdraft(&execution, None);
        assert!(
            matches!(refused, Err(MarginError::PrecisionExceeded { .. })),
            "{refused:?}"
        );
        assert_eq!(ledger.snapshot(), before);
    }

    /// What this crate does with a shared case.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Disposition {
        /// A grid check: `check_order` gives the same result.
        OrderCheck,
        /// A notional: `fill_notional` gives the same amount or error.
        FillNotional,
        /// A linear PnL: booked open and close through `fill_execution`, and the
        /// ledger's realized PnL is the exact amount.
        LinearBooked,
        /// An inverse PnL, the same way: the ledger rounds each leg once, so the
        /// realized PnL plus the recorded rests of the two legs is the exact
        /// amount. No tolerance.
        InverseBooked,
        /// An option's premium or expiry: this crate has no path for it, and both
        /// `fill_execution` and `fill_notional` refuse the option.
        OptionOutside,
        /// The expiry of a perpetual: only `financial_types` computes expiries.
        ExpiryFormulaOnly,
        /// Rounding of a given ratio: only `financial_types` does it.
        SettleFormulaOnly,
    }

    use Disposition::*;

    /// Every case, by `name | operation`, and its disposition.
    const TABLE: &[(&str, Disposition)] = &[
        (
            "linear notional: 0.5 BTC x 60000.1 USDT = 30000.05 USDT | Notional { contracts: 0.5, price: 60000.1 }",
            FillNotional,
        ),
        (
            "linear pnl long: 0.5 x (61000.5 - 60000) = +500.25 USDT | RealizedPnl { side: Side::Long, contracts: 0.5, entry: 60000, exit: 61000.5 }",
            LinearBooked,
        ),
        (
            "linear pnl short: the same move is -500.25 USDT | RealizedPnl { side: Side::Short, contracts: 0.5, entry: 60000, exit: 61000.5 }",
            LinearBooked,
        ),
        (
            "inverse notional: 1000 x 10 USD / 60000 = 1/6 BTC, 0.16666667 at satoshi | Notional { contracts: 1000, price: 60000 }",
            FillNotional,
        ),
        (
            "inverse pnl long: 10000 x (1/60000 - 1/61000) = 1/366 BTC, 0.00273224 | RealizedPnl { side: Side::Long, contracts: 1000, entry: 60000, exit: 61000 }",
            InverseBooked,
        ),
        (
            "inverse pnl short: -1/366 BTC, -0.00273224 | RealizedPnl { side: Side::Short, contracts: 1000, entry: 60000, exit: 61000 }",
            InverseBooked,
        ),
        (
            "inverse future pnl long: 1000 USD from 60000 to 59000 = -1/3540 BTC, -0.00028249 | RealizedPnl { side: Side::Long, contracts: 100, entry: 60000, exit: 59000 }",
            InverseBooked,
        ),
        (
            "option premium long: 1.5 x 1 BTC x 0.0125 = -0.01875 BTC paid | Premium { side: Side::Long, contracts: 1.5, premium: 0.0125 }",
            OptionOutside,
        ),
        (
            "option premium short: +0.01875 BTC received | Premium { side: Side::Short, contracts: 1.5, premium: 0.0125 }",
            OptionOutside,
        ),
        (
            "inverse call long, S 110000, K 100000: 2 x 10000 / 110000 = 2/11 BTC | Expiry { side: Side::Long, contracts: 2, settlement_price: 110000 }",
            OptionOutside,
        ),
        (
            "inverse call short: -2/11 BTC | Expiry { side: Side::Short, contracts: 2, settlement_price: 110000 }",
            OptionOutside,
        ),
        (
            "inverse call at the money: 0 | Expiry { side: Side::Long, contracts: 2, settlement_price: 100000 }",
            OptionOutside,
        ),
        (
            "inverse call out of the money: 0 | Expiry { side: Side::Long, contracts: 2, settlement_price: 90000 }",
            OptionOutside,
        ),
        (
            "inverse put long, S 90000: 2 x 10000 / 90000 = 2/9 BTC | Expiry { side: Side::Long, contracts: 2, settlement_price: 90000 }",
            OptionOutside,
        ),
        (
            "inverse put out of the money: 0 | Expiry { side: Side::Short, contracts: 2, settlement_price: 110000 }",
            OptionOutside,
        ),
        (
            "illustrative linear cash call long, S 120, K 100: 3 x 20 = +60 USDT | Expiry { side: Side::Long, contracts: 3, settlement_price: 120 }",
            OptionOutside,
        ),
        (
            "illustrative physical call long, S 120: +3 BTC, -300 USDT | Expiry { side: Side::Long, contracts: 3, settlement_price: 120 }",
            OptionOutside,
        ),
        (
            "illustrative physical call short, S 120: -3 BTC, +300 USDT | Expiry { side: Side::Short, contracts: 3, settlement_price: 120 }",
            OptionOutside,
        ),
        (
            "illustrative physical put long, S 80: -3 BTC, +300 USDT | Expiry { side: Side::Long, contracts: 3, settlement_price: 80 }",
            OptionOutside,
        ),
        (
            "illustrative physical put short, S 80: +3 BTC, -300 USDT | Expiry { side: Side::Short, contracts: 3, settlement_price: 80 }",
            OptionOutside,
        ),
        (
            "illustrative physical call out of the money: abandoned | Expiry { side: Side::Long, contracts: 3, settlement_price: 90 }",
            OptionOutside,
        ),
        (
            "illustrative physical call at the money: abandoned | Expiry { side: Side::Short, contracts: 3, settlement_price: 100 }",
            OptionOutside,
        ),
        (
            "illustrative physical put out of the money: abandoned | Expiry { side: Side::Long, contracts: 3, settlement_price: 120 }",
            OptionOutside,
        ),
        (
            "Deribit option premium 0.005: on the 0.0001 band (inclusive) | CheckPrice { price: 0.005 }",
            OrderCheck,
        ),
        (
            "Deribit option premium 0.0049: on 0.0001 | CheckPrice { price: 0.0049 }",
            OrderCheck,
        ),
        (
            "Deribit option premium 0.00495: off 0.0001 | CheckPrice { price: 0.00495 }",
            OrderCheck,
        ),
        (
            "Deribit option premium 0.0051: above 0.005, off 0.0005 | CheckPrice { price: 0.0051 }",
            OrderCheck,
        ),
        (
            "Deribit option premium 0.0055: on 0.0005 | CheckPrice { price: 0.0055 }",
            OrderCheck,
        ),
        (
            "Deribit future price 60002.5: on 2.5 | CheckPrice { price: 60002.5 }",
            OrderCheck,
        ),
        (
            "Deribit future price 60001: off 2.5 | CheckPrice { price: 60001 }",
            OrderCheck,
        ),
        (
            "Binance price 60000.15: off 0.1 | CheckPrice { price: 60000.15 }",
            OrderCheck,
        ),
        (
            "Binance size 0.0015 BTC: off 0.001 | CheckLot { contracts: 0.0015 }",
            OrderCheck,
        ),
        (
            "Deribit option size 0.15: off 0.1 | CheckLot { contracts: 0.15 }",
            OrderCheck,
        ),
        (
            "Deribit option size 0.3: on 0.1 | CheckLot { contracts: 0.3 }",
            OrderCheck,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: Floor }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: Floor }",
            SettleFormulaOnly,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: Ceil }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: Ceil }",
            SettleFormulaOnly,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: TowardZero }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: TowardZero }",
            SettleFormulaOnly,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: AwayFromZero }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: AwayFromZero }",
            SettleFormulaOnly,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: HalfEven }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: HalfEven }",
            SettleFormulaOnly,
        ),
        (
            "tie +1/8 at 2 places | Settle { value: Ratio { num: 1, den: 8 }, scale: 2, rule: HalfAwayFromZero }",
            SettleFormulaOnly,
        ),
        (
            "tie -1/8 at 2 places | Settle { value: Ratio { num: -1, den: 8 }, scale: 2, rule: HalfAwayFromZero }",
            SettleFormulaOnly,
        ),
        (
            "tie 3/8 half even: the odd floor goes up, 0.38 | Settle { value: Ratio { num: 3, den: 8 }, scale: 2, rule: HalfEven }",
            SettleFormulaOnly,
        ),
        (
            "2/3 at scale 28 | Settle { value: Ratio { num: 2, den: 3 }, scale: 28, rule: HalfEven }",
            SettleFormulaOnly,
        ),
        (
            "scale 29 is refused | Settle { value: Ratio { num: 1, den: 3 }, scale: 29, rule: HalfEven }",
            SettleFormulaOnly,
        ),
        (
            "Decimal::MAX contracts x price 2: overflow | Notional { contracts: 79228162514264337593543950335, price: 2 }",
            FillNotional,
        ),
        (
            "1.000000000000001 x 1.00000000000001: 31 digits, inexact | Notional { contracts: 1.000000000000001, price: 1.00000000000001 }",
            FillNotional,
        ),
        (
            "1e-28 x 1e-28: below scale 28, inexact | Notional { contracts: 0.0000000000000000000000000001, price: 0.0000000000000000000000000001 }",
            FillNotional,
        ),
        (
            "negative premium is refused | Premium { side: Side::Long, contracts: 1, premium: -0.0001 }",
            OptionOutside,
        ),
        (
            "zero contracts are refused | Notional { contracts: 0, price: 60000 }",
            FillNotional,
        ),
        (
            "expiry of a perpetual is unsupported | Expiry { side: Side::Long, contracts: 1, settlement_price: 60000 }",
            ExpiryFormulaOnly,
        ),
    ];

    fn key(case: &Case) -> String {
        format!("{} | {:?}", case.name, case.operation)
    }

    fn exact(expected: &Expected) -> &MoneyRatio {
        match expected {
            Expected::Amount { exact, .. } => exact,
            other => panic!("expected an amount, got {other:?}"),
        }
    }

    fn q(ratio: &MoneyRatio) -> Q {
        Q::ratio(ratio.ratio.num(), ratio.ratio.den())
    }

    fn run_order_check(case: &Case, name: &str) {
        let symbol = case.spec.id().as_str();
        let order = match &case.operation {
            Operation::CheckPrice { price } => {
                OrderRequest::limit_buy(symbol, *price, case.spec.lot())
            }
            Operation::CheckLot { contracts } => OrderRequest::market_buy(symbol, *contracts),
            other => panic!("{name}: unexpected operation {other:?}"),
        };
        match (&case.expected, check_order(&case.spec, &order)) {
            (Expected::Valid, Ok(())) => {}
            (Expected::Error(kind), Err(OrderSpecError::Spec(e))) if ErrorKind::of(&e) == *kind => {
            }
            (expected, got) => panic!("{name}: expected {expected:?}, got {got:?}"),
        }
    }

    fn run_fill_notional(case: &Case, name: &str) {
        let Operation::Notional { contracts, price } = &case.operation else {
            panic!("{name}: not a notional");
        };
        let got = notional_of(&case.spec, &fill(Action::Buy, *price, *contracts));
        match (&case.expected, got) {
            (Expected::Amount { exact, .. }, Ok(amount)) => {
                assert!(amount.equals(exact).unwrap(), "{name}: {amount:?}");
            }
            (Expected::Error(kind), Err(OrderSpecError::Spec(e))) if ErrorKind::of(&e) == *kind => {
            }
            (expected, got) => panic!("{name}: expected {expected:?}, got {got:?}"),
        }
    }

    /// Books the open and the close of a PnL case.
    fn book_pnl(case: &Case, name: &str) -> (ExecutionCashflow, ExecutionCashflow, Side) {
        let Operation::RealizedPnl {
            side,
            contracts,
            entry,
            exit,
        } = &case.operation
        else {
            panic!("{name}: not a PnL");
        };
        let (open, close) = match side {
            Side::Long => (Action::Buy, Action::Sell),
            Side::Short => (Action::Sell, Action::Buy),
        };
        let mut ledger = ledger_for(&case.spec);
        let opened = book(
            &mut ledger,
            &case.spec,
            &at("a", 1, open, *entry, *contracts),
        );
        let closed = book(
            &mut ledger,
            &case.spec,
            &at("b", 2, close, *exit, *contracts),
        );
        (opened, closed, *side)
    }

    fn run_linear(case: &Case, name: &str) {
        let expected = exact(&case.expected);
        assert_eq!(expected.asset.as_str(), "USDT", "{name}");
        let (_, closed, _) = book_pnl(case, name);
        assert_eq!(Q::dec(closed.realized_pnl), q(expected), "{name}");
    }

    fn rest(fill: &ExecutionCashflow) -> Q {
        let residual = fill.derivative.as_ref().unwrap().residual.as_ref().unwrap();
        Q::ratio(residual.numerator, residual.denominator)
    }

    fn run_inverse(case: &Case, name: &str) {
        let expected = exact(&case.expected);
        assert_eq!(expected.asset.as_str(), "BTC", "{name}");
        let (opened, closed, side) = book_pnl(case, name);
        // realized + s·(r_open - r_close) = exact, s = +1 long, -1 short.
        let rests = rest(&opened).add(rest(&closed).neg());
        let rests = match side {
            Side::Long => rests,
            Side::Short => rests.neg(),
        };
        assert_eq!(
            Q::dec(closed.realized_pnl).add(rests),
            q(expected),
            "{name}"
        );
    }

    fn run_option(case: &Case, name: &str) {
        assert_eq!(case.spec.kind(), ContractKind::Option, "{name}");
        let f = fill(Action::Buy, dec!(1), dec!(1));
        assert!(
            matches!(
                execution_of(&case.spec, &f),
                Err(AdapterError::Unsupported { .. })
            ),
            "{name}"
        );
        assert!(
            matches!(
                notional_of(&case.spec, &f),
                Err(OrderSpecError::OptionNotional { .. })
            ),
            "{name}"
        );
    }

    #[test]
    fn shared_fixtures_every_case_has_a_checked_disposition() {
        let table: BTreeMap<&str, Disposition> = TABLE.iter().copied().collect();
        assert_eq!(table.len(), TABLE.len(), "duplicate key in the table");
        let cases = fixtures::cases().unwrap();
        let mut seen = BTreeSet::new();
        let mut counts: BTreeMap<Disposition, usize> = BTreeMap::new();
        for case in &cases {
            // Canary: the case still passes against financial_types.
            fixtures::check(case).unwrap();
            let name = key(case);
            let disposition = *table
                .get(name.as_str())
                .unwrap_or_else(|| panic!("unclassified fixture: {name}"));
            assert!(seen.insert(name.clone()), "duplicate case: {name}");
            *counts.entry(disposition).or_default() += 1;
            match disposition {
                OrderCheck => run_order_check(case, &name),
                FillNotional => run_fill_notional(case, &name),
                LinearBooked => run_linear(case, &name),
                InverseBooked => run_inverse(case, &name),
                OptionOutside => run_option(case, &name),
                ExpiryFormulaOnly => {
                    assert!(matches!(case.operation, Operation::Expiry { .. }), "{name}");
                    assert_ne!(case.spec.kind(), ContractKind::Option, "{name}");
                }
                SettleFormulaOnly => {
                    assert!(matches!(case.operation, Operation::Settle { .. }), "{name}");
                }
            }
        }
        for (name, _) in TABLE {
            assert!(seen.contains(*name), "fixture no longer present: {name}");
        }
        let expected: BTreeMap<Disposition, usize> = [
            (OrderCheck, 11),
            (FillNotional, 6),
            (LinearBooked, 2),
            (InverseBooked, 3),
            (OptionOutside, 17),
            (ExpiryFormulaOnly, 1),
            (SettleFormulaOnly, 15),
        ]
        .into_iter()
        .collect();
        assert_eq!(counts, expected);
    }
}
