//! Conversion of `financial_types` specifications into `margin-rs`
//! instruments.
//!
//! The expected values are literals read from the specifications (and from
//! the dated venue snapshots of `financial_types::instrument::reference`),
//! never recomputed with the adapter's own helpers.

#![cfg(all(feature = "margin", feature = "instruments"))]

use financial_types::instrument::fixtures;
use margin_rs::ledger::{ExpirySettlement, Ledger, LedgerConfig};
use margin_rs::{ExecutionId, Instrument, MarginError, PositionKind};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::instruments::margin::{AdapterError, MarginScales, margin_instrument};
use market_maker_rs::instruments::reference::{binance_usdm, deribit};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentError, InstrumentId,
    InstrumentSpec, Margining, TickBand, TickSchedule,
};
use market_maker_rs::prelude::OptionStyle;

const EXPIRY_MS: u64 = 1_795_766_400_000;

fn scales(price_scale: u32, contract_scale: u32) -> MarginScales {
    MarginScales {
        price_scale,
        contract_scale,
    }
}

fn asset(code: &str) -> AssetCode {
    AssetCode::new(code).unwrap()
}

fn id(value: &str) -> InstrumentId {
    InstrumentId::new(value).unwrap()
}

/// A linear contract on BTC priced and settled in USDT.
fn linear(
    kind: ContractKind,
    multiplier: Decimal,
    ticks: TickSchedule,
    lot: Decimal,
    expiry_ms: Option<u64>,
) -> InstrumentSpec {
    InstrumentSpec::new(InstrumentDraft {
        id: id("TEST-LINEAR"),
        kind,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(multiplier, asset("BTC")).unwrap(),
        price_ticks: ticks,
        lot,
        expiry_ms,
        option: None,
    })
    .unwrap()
}

/// An inverse contract on BTC priced in USD and settled in BTC.
fn inverse(
    kind: ContractKind,
    contract_value: Decimal,
    ticks: TickSchedule,
    lot: Decimal,
    expiry_ms: Option<u64>,
) -> InstrumentSpec {
    InstrumentSpec::new(InstrumentDraft {
        id: id("TEST-INVERSE"),
        kind,
        margining: Margining::Inverse,
        underlying: asset("BTC"),
        price_asset: asset("USD"),
        settlement_asset: asset("BTC"),
        size: ContractSize::quote_value(contract_value, asset("USD")).unwrap(),
        price_ticks: ticks,
        lot,
        expiry_ms,
        option: None,
    })
    .unwrap()
}

fn fixed(tick: Decimal) -> TickSchedule {
    TickSchedule::fixed(tick).unwrap()
}

fn spot(tick: Decimal, lot: Decimal) -> InstrumentSpec {
    InstrumentSpec::new(InstrumentDraft {
        id: id("BTC/USDT"),
        kind: ContractKind::Spot,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(Decimal::ONE, asset("BTC")).unwrap(),
        price_ticks: fixed(tick),
        lot,
        expiry_ms: None,
        option: None,
    })
    .unwrap()
}

fn below_grid(field: &'static str, grid: &'static str, given: u32, required: u32) -> AdapterError {
    AdapterError::ScaleBelowGrid {
        field,
        grid,
        given,
        required,
    }
}

fn ledger(assets: &[(&str, u32)]) -> Ledger {
    let mut ledger = Ledger::new(LedgerConfig::default());
    for (code, scale) in assets {
        ledger
            .register_asset(margin_rs::Asset::new(code).unwrap(), *scale)
            .unwrap();
    }
    ledger
}

// ------------------------------------------------- reference snapshots

/// Binance `BTCUSDT`: linear perpetual, multiplier 1 BTC, settled in USDT,
/// tick 0.1 (1 digit), step 0.001 (3 digits).
#[test]
fn binance_btcusdt_perpetual_maps_exactly() {
    let spec = binance_usdm::btcusdt_perpetual().unwrap();
    let instrument = margin_instrument(&spec, scales(1, 3)).unwrap();
    let Instrument::LinearPerpetual(perpetual) = &instrument else {
        panic!("expected a linear perpetual, got {instrument:?}");
    };
    let terms = perpetual.terms();
    assert_eq!(terms.id().as_str(), "BTCUSDT");
    assert_eq!(terms.settlement().as_str(), "USDT");
    assert_eq!(terms.multiplier(), dec!(1));
    assert_eq!((terms.price_scale(), terms.contract_scale()), (1, 3));
    assert_eq!(instrument.kind(), PositionKind::LinearPerpetual);
    assert_eq!(instrument.expiry_ms(), None);

    assert_eq!(
        margin_instrument(&spec, scales(0, 3)).unwrap_err(),
        below_grid("price_scale", "price ticks", 0, 1)
    );
    assert_eq!(
        margin_instrument(&spec, scales(1, 2)).unwrap_err(),
        below_grid("contract_scale", "lot", 2, 3)
    );
}

/// Deribit `BTC-PERPETUAL`: inverse perpetual, 10 USD per contract, settled
/// in BTC, tick 0.5 (1 digit), whole contracts (0 digits).
#[test]
fn deribit_btc_perpetual_maps_exactly() {
    let spec = deribit::btc_perpetual().unwrap();
    let instrument = margin_instrument(&spec, scales(1, 0)).unwrap();
    let Instrument::InversePerpetual(perpetual) = &instrument else {
        panic!("expected an inverse perpetual, got {instrument:?}");
    };
    let terms = perpetual.terms();
    assert_eq!(terms.id().as_str(), "BTC-PERPETUAL");
    assert_eq!(terms.settlement().as_str(), "BTC");
    assert_eq!(terms.contract_value(), dec!(10));
    assert_eq!((terms.price_scale(), terms.contract_scale()), (1, 0));
    assert_eq!(instrument.expiry_ms(), None);
    assert_eq!(
        margin_instrument(&spec, scales(0, 0)).unwrap_err(),
        below_grid("price_scale", "price ticks", 0, 1)
    );
}

/// Deribit BTC future: inverse future, 10 USD per contract, tick 2.5, the
/// expiry passed unchanged.
#[test]
fn deribit_btc_future_maps_exactly() {
    let spec = deribit::btc_future(id("BTC-27NOV26"), EXPIRY_MS).unwrap();
    let instrument = margin_instrument(&spec, scales(2, 0)).unwrap();
    let Instrument::InverseFuture(future) = &instrument else {
        panic!("expected an inverse future, got {instrument:?}");
    };
    assert_eq!(future.terms().id().as_str(), "BTC-27NOV26");
    assert_eq!(future.terms().settlement().as_str(), "BTC");
    assert_eq!(future.terms().contract_value(), dec!(10));
    assert_eq!(future.terms().price_scale(), 2);
    assert_eq!(future.expiry_ms(), EXPIRY_MS);
    assert_eq!(instrument.kind(), PositionKind::InverseFuture);
}

/// Deribit BTC option: `margin-rs` 0.2 converts it, as an inverse option, and its scales are
/// checked as for any other instrument.
#[test]
fn options_convert_through_margin_rs() {
    let spec = deribit::btc_option(
        id("BTC-27NOV26-100000-C"),
        EXPIRY_MS,
        dec!(100000),
        OptionStyle::Call,
    )
    .unwrap();
    let converted = margin_instrument(&spec, scales(4, 1)).unwrap();
    let option = converted.option_terms().unwrap();
    assert_eq!(option.contract().strike, dec!(100000));
    assert_eq!(option.contract().expiry_ms, EXPIRY_MS);
    assert_eq!(
        option.contract().margining,
        margin_rs::instrument::OptionMargining::Inverse
    );
    assert!(matches!(
        margin_instrument(&spec, scales(0, 0)),
        Err(AdapterError::ScaleBelowGrid { .. })
    ));
}

// ------------------------------------------------------ other forms

/// Spot: base the underlying, quote the price asset.
#[test]
fn spot_maps_base_and_quote() {
    let spec = spot(dec!(0.01), dec!(0.00001));
    let instrument = margin_instrument(&spec, scales(2, 5)).unwrap();
    let Instrument::Spot(spot) = &instrument else {
        panic!("expected spot, got {instrument:?}");
    };
    assert_eq!(spot.id().as_str(), "BTC/USDT");
    assert_eq!(spot.base().as_str(), "BTC");
    assert_eq!(spot.quote().as_str(), "USDT");
    assert_eq!((spot.price_scale(), spot.quantity_scale()), (2, 5));
}

/// Linear future with a fractional multiplier, a tick that is not a power
/// of ten and a lot of 0.25; the largest expiry is passed unchanged.
#[test]
fn linear_future_fractional_multiplier_tick_and_lot() {
    let spec = linear(
        ContractKind::Future,
        dec!(0.001),
        fixed(dec!(2.5)),
        dec!(0.25),
        Some(u64::MAX),
    );
    let instrument = margin_instrument(&spec, scales(1, 2)).unwrap();
    let Instrument::LinearFuture(future) = &instrument else {
        panic!("expected a linear future, got {instrument:?}");
    };
    assert_eq!(future.terms().multiplier(), dec!(0.001));
    assert_eq!(future.terms().settlement().as_str(), "USDT");
    assert_eq!(future.expiry_ms(), u64::MAX);
    assert_eq!(
        margin_instrument(&spec, scales(1, 1)).unwrap_err(),
        below_grid("contract_scale", "lot", 1, 2)
    );
}

/// Inverse perpetual with a fractional contract value.
#[test]
fn inverse_perpetual_fractional_contract_value() {
    let spec = inverse(
        ContractKind::Perpetual,
        dec!(0.5),
        fixed(dec!(0.05)),
        dec!(1),
        None,
    );
    let instrument = margin_instrument(&spec, scales(2, 0)).unwrap();
    assert_eq!(
        instrument.inverse_terms().unwrap().contract_value(),
        dec!(0.5)
    );
    assert_eq!(instrument.kind(), PositionKind::InversePerpetual);
}

/// The digits of every band count, not only the last one's.
#[test]
fn multi_band_schedule_requires_the_finest_tick() {
    let finest_first = TickSchedule::new(vec![
        TickBand {
            up_to: Some(dec!(10)),
            tick: dec!(0.001),
        },
        TickBand {
            up_to: None,
            tick: dec!(0.5),
        },
    ])
    .unwrap();
    let spec = linear(
        ContractKind::Perpetual,
        dec!(1),
        finest_first,
        dec!(1),
        None,
    );
    assert!(margin_instrument(&spec, scales(3, 0)).is_ok());
    assert_eq!(
        margin_instrument(&spec, scales(2, 0)).unwrap_err(),
        below_grid("price_scale", "price ticks", 2, 3)
    );

    let finest_last = TickSchedule::new(vec![
        TickBand {
            up_to: Some(dec!(10)),
            tick: dec!(0.5),
        },
        TickBand {
            up_to: None,
            tick: dec!(0.0001),
        },
    ])
    .unwrap();
    let spec = linear(ContractKind::Perpetual, dec!(1), finest_last, dec!(1), None);
    assert_eq!(
        margin_instrument(&spec, scales(3, 0)).unwrap_err(),
        below_grid("price_scale", "price ticks", 3, 4)
    );
}

/// Trailing zeros do not count: tick 0.50 has 1 digit, lot 1.000 has 0, and
/// the multiplier is passed as the same value.
#[test]
fn trailing_zeros_are_ignored() {
    let spec = linear(
        ContractKind::Perpetual,
        dec!(1.000),
        fixed(dec!(0.50)),
        dec!(1.000),
        None,
    );
    let instrument = margin_instrument(&spec, scales(1, 0)).unwrap();
    assert_eq!(instrument.linear_terms().unwrap().multiplier(), dec!(1));
}

/// A tick at scale 28 needs 28 digits; `margin-rs` then refuses the scale
/// (its maximum is 18), as a typed error.
#[test]
fn scale_28_tick() {
    let tick = Decimal::new(1, 28);
    let spec = linear(ContractKind::Perpetual, dec!(1), fixed(tick), dec!(1), None);
    assert_eq!(
        margin_instrument(&spec, scales(27, 0)).unwrap_err(),
        below_grid("price_scale", "price ticks", 27, 28)
    );
    assert!(matches!(
        margin_instrument(&spec, scales(28, 0)),
        Err(AdapterError::Margin(MarginError::InvalidInput { .. }))
    ));
}

/// `margin-rs` refuses a contract value whose digits plus the contract
/// scale exceed 28.
#[test]
fn contract_value_digits_plus_contract_scale_above_28() {
    let contract_value = Decimal::new(1, 11);
    let spec = inverse(
        ContractKind::Perpetual,
        contract_value,
        fixed(dec!(0.5)),
        dec!(1),
        None,
    );
    assert!(margin_instrument(&spec, scales(1, 17)).is_ok());
    assert!(matches!(
        margin_instrument(&spec, scales(1, 18)),
        Err(AdapterError::Margin(MarginError::InvalidInput { .. }))
    ));
}

// ----------------------------------------------- scales and the grid

/// Tick 0.5 against a delivery price with 2 decimals: the price scale must
/// cover the settlement price, not only the tick.
#[test]
fn settlement_price_finer_than_the_tick_needs_a_larger_price_scale() {
    let spec = inverse(
        ContractKind::Future,
        dec!(10),
        fixed(dec!(0.5)),
        dec!(1),
        Some(EXPIRY_MS),
    );
    let settle = |price_scale: u32| {
        let mut ledger = ledger(&[("BTC", 8)]);
        ledger
            .register_instrument(margin_instrument(&spec, scales(price_scale, 0)).unwrap())
            .unwrap();
        ledger.settle_expiry(&ExpirySettlement {
            id: ExecutionId::new("settle-1").unwrap(),
            instrument: margin_rs::InstrumentId::new("TEST-INVERSE").unwrap(),
            price: dec!(61234.57),
            as_of_ms: EXPIRY_MS,
        })
    };
    assert!(matches!(
        settle(1),
        Err(MarginError::PrecisionExceeded { .. })
    ));
    assert!(settle(2).is_ok());
}

/// The lot is not the risk minimum: with lot 0.25 and contract scale 2,
/// a size of 0.01 fits the scale but is off the lot.
#[test]
fn lot_is_not_the_contract_scale_unit() {
    let spec = linear(
        ContractKind::Perpetual,
        dec!(1),
        fixed(dec!(0.5)),
        dec!(0.25),
        None,
    );
    let instrument = margin_instrument(&spec, scales(1, 2)).unwrap();
    assert_eq!(instrument.linear_terms().unwrap().contract_scale(), 2);
    let unit = dec!(0.01);
    assert!(matches!(
        spec.check_lot(unit),
        Err(InstrumentError::NotOnGrid { .. })
    ));
    assert!(spec.check_lot(dec!(0.25)).is_ok());
}

// --------------------------------------------- checks left to the ledger

/// A converted instrument registers in a ledger; the checks that need the
/// settlement asset scale come from `register_instrument`.
#[test]
fn ledger_registration_checks() {
    let binance =
        margin_instrument(&binance_usdm::btcusdt_perpetual().unwrap(), scales(1, 3)).unwrap();
    let deribit = margin_instrument(&deribit::btc_perpetual().unwrap(), scales(1, 0)).unwrap();
    let mut ok = ledger(&[("BTC", 8), ("USDT", 8)]);
    ok.register_instrument(binance.clone()).unwrap();
    ok.register_instrument(deribit).unwrap();

    // Unregistered settlement asset.
    assert!(matches!(
        ledger(&[("BTC", 8)]).register_instrument(binance),
        Err(MarginError::UnknownAsset { .. })
    ));

    // Spot quantity scale above the base asset scale.
    let spot = margin_instrument(&spot(dec!(0.01), dec!(0.00001)), scales(2, 9)).unwrap();
    assert!(matches!(
        ledger(&[("BTC", 8), ("USDT", 8)]).register_instrument(spot),
        Err(MarginError::InvalidInput { .. })
    ));

    // Inverse: settlement asset scale plus price scale above 28.
    let wide = margin_instrument(&deribit::btc_perpetual().unwrap(), scales(11, 0)).unwrap();
    assert!(matches!(
        ledger(&[("BTC", 18)]).register_instrument(wide.clone()),
        Err(MarginError::InvalidInput { .. })
    ));
    assert!(ledger(&[("BTC", 17)]).register_instrument(wide).is_ok());
}

// ------------------------------------------------------------ fixtures

/// Canary: the shared fixtures still pass against the published formulas,
/// and every specification they use converts, options included since `margin-rs` 0.2.
#[test]
fn shared_fixtures_canary() {
    let cases = fixtures::cases().unwrap();
    assert!(!cases.is_empty());
    for case in &cases {
        fixtures::check(case).unwrap();
        let result = margin_instrument(&case.spec, scales(18, 18));
        assert!(result.is_ok(), "{}: {result:?}", case.name);
    }
}
