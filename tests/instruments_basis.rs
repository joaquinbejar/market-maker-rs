//! The net edge of a basis trade between a spot and a linear dated future.
//!
//! The real quotes were read on 2026-11-05 at 00:49 UTC, from Binance's public API:
//! `fapi/v1/ticker/bookTicker?symbol=BTCUSDT_261225` has `bidPrice` 81389.7,
//! `bidQty` 0.003, `askPrice` 81392.8 and `askQty` 0.005 at `time` 1793839761620, and
//! `api/v3/ticker/bookTicker?symbol=BTCUSDT` has `bidPrice` 80794.01, `bidQty`
//! 0.21626, `askPrice` 80794.02 and `askQty` 12.83672, with no time.
//! `fapi/v1/exchangeInfo` has `BTCUSDT_261225` with `deliveryDate` 1798185600000,
//! base BTC, quote and margin USDT, tick 0.1 and step 0.001, and
//! `api/v3/exchangeInfo` has `BTCUSDT` with tick 0.01 and step 0.00001.
//!
//! Two things are not read from the venue. A contract of 1 BTC follows from the
//! quantity being in BTC, as there is no `contractSize` on USD-M, as in
//! `binance_usdm::btcusdt_perpetual`. The fees and carry below are assumed.

#![cfg(feature = "arbitrage")]

use financial_types::Action;
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::instruments::basis::{
    BasisCarry, BasisCarryCost, BasisError, BasisEvaluation, BasisInput, BasisLegPlan, BasisLevel,
    BasisOpportunity, BasisQuote, BasisRejection, CarryKind, evaluate_basis,
};
use market_maker_rs::instruments::reference::{binance_usdm, deribit};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
    Margining, TickSchedule,
};
use market_maker_rs::prelude::OptionStyle;

const DELIVERY_MS: u64 = 1_798_185_600_000;
const AS_OF_MS: u64 = 1_793_839_762_000;

fn asset(code: &str) -> AssetCode {
    AssetCode::new(code).unwrap()
}

fn id(text: &str) -> InstrumentId {
    InstrumentId::new(text).unwrap()
}

fn spot_spec() -> InstrumentSpec {
    InstrumentSpec::new(InstrumentDraft {
        id: id("BTCUSDT"),
        kind: ContractKind::Spot,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(0.01)).unwrap(),
        lot: dec!(0.00001),
        expiry_ms: None,
        option: None,
    })
    .unwrap()
}

fn future_draft() -> InstrumentDraft {
    InstrumentDraft {
        id: id("BTCUSDT_261225"),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(dec!(1), asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(0.1)).unwrap(),
        lot: dec!(0.001),
        expiry_ms: Some(DELIVERY_MS),
        option: None,
    }
}

fn future_spec() -> InstrumentSpec {
    InstrumentSpec::new(future_draft()).unwrap()
}

fn quote(spec: InstrumentSpec, price: Decimal, contracts: Decimal, fee_bps: Decimal) -> BasisQuote {
    BasisQuote {
        spec,
        level: Some(BasisLevel { price, contracts }),
        quoted_at_ms: AS_OF_MS,
        fee_bps,
        exit_bps: dec!(0),
    }
}

/// Buy spot at its real ask and sell the future at its real bid, with an assumed
/// taker fee of 10 and 5 basis points.
fn cash_and_carry() -> BasisInput {
    BasisInput {
        as_of_ms: AS_OF_MS,
        buy: quote(spot_spec(), dec!(80794.02), dec!(12.83672), dec!(10)),
        sell: quote(future_spec(), dec!(81389.7), dec!(0.003), dec!(5)),
        target_quantity: dec!(1),
        min_quantity: dec!(0.001),
        max_quote_age_ms: 1000,
        future_margin_rate: dec!(0.05),
        carry: Vec::new(),
        min_net_edge: dec!(0),
    }
}

fn accepted(input: &BasisInput) -> BasisOpportunity {
    match evaluate_basis(input).unwrap() {
        BasisEvaluation::Accepted(opportunity) => opportunity,
        BasisEvaluation::Rejected(rejection) => panic!("rejected: {rejection:?}"),
    }
}

fn rejected(input: &BasisInput) -> BasisRejection {
    match evaluate_basis(input).unwrap() {
        BasisEvaluation::Rejected(rejection) => rejection,
        BasisEvaluation::Accepted(opportunity) => panic!("accepted: {opportunity:?}"),
    }
}

#[test]
fn the_real_quotes_net_of_every_cost() {
    let mut input = cash_and_carry();
    input.buy.exit_bps = dec!(10);
    input.sell.exit_bps = dec!(2);
    input.carry = vec![BasisCarry {
        kind: CarryKind::Financing,
        on: Action::Buy,
        period_bps: dec!(20),
    }];
    let o = accepted(&input);
    assert_eq!(o.asset.as_str(), "USDT");
    // Limited by the future's 0.003 at the bid.
    assert_eq!(o.quantity, dec!(0.003));
    assert_eq!(o.converges_at_ms, DELIVERY_MS);
    assert_eq!(
        o.buy,
        BasisLegPlan {
            instrument: "BTCUSDT".to_string(),
            price: dec!(80794.02),
            contracts: dec!(0.003),
        }
    );
    assert_eq!(
        o.sell,
        BasisLegPlan {
            instrument: "BTCUSDT_261225".to_string(),
            price: dec!(81389.7),
            contracts: dec!(0.003),
        }
    );
    // 0.003 x (81389.7 - 80794.02).
    assert_eq!(o.gross, dec!(1.78704));
    // The notionals are 242.38206 and 244.1691.
    assert_eq!(o.costs.buy_fee, dec!(0.24238206));
    assert_eq!(o.costs.sell_fee, dec!(0.12208455));
    assert_eq!(o.costs.buy_exit, dec!(0.24238206));
    assert_eq!(o.costs.sell_exit, dec!(0.04883382));
    assert_eq!(
        o.costs.carry,
        [BasisCarryCost {
            kind: CarryKind::Financing,
            on: Action::Buy,
            amount: dec!(0.48476412),
        }]
    );
    assert_eq!(o.costs.total, dec!(1.14044661));
    assert_eq!(o.net, dec!(0.64659339));
    // 242.38206 + 5% of 244.1691.
    assert_eq!(o.capital, dec!(254.590515));
}

#[test]
fn selling_spot_needs_a_declared_borrow_and_charges_it() {
    // Sell spot at its bid and buy the future at its real ask: no gross edge there,
    // so the future is offered below spot here.
    let mut input = BasisInput {
        buy: quote(future_spec(), dec!(80000), dec!(0.005), dec!(5)),
        sell: quote(spot_spec(), dec!(80794.01), dec!(0.21626), dec!(10)),
        ..cash_and_carry()
    };
    assert_eq!(evaluate_basis(&input), Err(BasisError::MissingBorrow));
    // A borrow on the other leg is not one on the spot sold.
    input.carry = vec![BasisCarry {
        kind: CarryKind::Borrow,
        on: Action::Buy,
        period_bps: dec!(0),
    }];
    assert_eq!(evaluate_basis(&input), Err(BasisError::MissingBorrow));
    // Other carry on that leg is not a borrow either.
    input.carry = vec![BasisCarry {
        kind: CarryKind::Other,
        on: Action::Sell,
        period_bps: dec!(0),
    }];
    assert_eq!(evaluate_basis(&input), Err(BasisError::MissingBorrow));
    // Zero may be declared, and absence may not.
    input.carry = vec![BasisCarry {
        kind: CarryKind::Borrow,
        on: Action::Sell,
        period_bps: dec!(0),
    }];
    let free = accepted(&input);
    assert_eq!(free.quantity, dec!(0.005));
    assert_eq!(free.gross, dec!(3.97005));
    input.carry[0].period_bps = dec!(30);
    let charged = accepted(&input);
    // 30 basis points of 0.005 x 80794.01 = 403.97005.
    assert_eq!(charged.costs.carry[0].amount, dec!(1.21191015));
    assert_eq!(charged.net, free.net - dec!(1.21191015));
    // Capital: 100% of the spot sold, and the rate on the future.
    assert_eq!(charged.capital, dec!(403.97005) + dec!(400) * dec!(0.05));
}

#[test]
fn rebates_and_negative_carry_reduce_the_costs() {
    let mut input = cash_and_carry();
    input.sell.fee_bps = dec!(-2);
    input.carry = vec![BasisCarry {
        kind: CarryKind::Other,
        on: Action::Sell,
        period_bps: dec!(-1),
    }];
    let o = accepted(&input);
    assert_eq!(o.costs.sell_fee, dec!(-0.04883382));
    assert_eq!(o.costs.carry[0].amount, dec!(-0.02441691));
    assert_eq!(
        o.costs.total,
        dec!(0.24238206) - dec!(0.04883382) - dec!(0.02441691)
    );
    assert_eq!(o.net, o.gross - o.costs.total);
}

#[test]
fn a_contract_of_another_multiplier_is_counted_in_the_underlying() {
    // Contracts of 0.01 BTC with a lot of 2, so a lot exposure of 0.02 BTC.
    let mut draft = future_draft();
    draft.size = ContractSize::base(dec!(0.01), asset("BTC")).unwrap();
    draft.lot = dec!(2);
    let mut input = cash_and_carry();
    input.sell = quote(
        InstrumentSpec::new(draft).unwrap(),
        dec!(81389.7),
        dec!(7),
        dec!(0),
    );
    input.buy.fee_bps = dec!(0);
    let o = accepted(&input);
    // 7 contracts are 0.07 BTC, floored to 0.06 BTC on the 0.02 grid.
    assert_eq!(o.quantity, dec!(0.06));
    assert_eq!(o.sell.contracts, dec!(6));
    assert_eq!(o.buy.contracts, dec!(0.06));
    assert_eq!(o.gross, dec!(35.7408));
    assert_eq!(o.capital, dec!(0.06) * dec!(80794.02) + dec!(244.1691));
}

#[test]
fn liquidity_is_floored_to_the_coarser_lot_and_the_minimum_checked_after() {
    // 0.0035 BTC quoted on the spot leg: 0.003 on the future's 0.001 grid.
    let mut input = cash_and_carry();
    input.sell.level = Some(BasisLevel {
        price: dec!(81389.7),
        contracts: dec!(5),
    });
    input.buy.level = Some(BasisLevel {
        price: dec!(80794.02),
        contracts: dec!(0.0035),
    });
    assert_eq!(accepted(&input).quantity, dec!(0.003));
    // The target too.
    input.buy.level = Some(BasisLevel {
        price: dec!(80794.02),
        contracts: dec!(5),
    });
    input.target_quantity = dec!(0.0049);
    assert_eq!(accepted(&input).quantity, dec!(0.004));
    // The minimum after flooring: 0.0049 is above it, 0.004 is not.
    input.min_quantity = dec!(0.0045);
    assert_eq!(
        rejected(&input),
        BasisRejection::InsufficientLiquidity {
            available: dec!(0.004),
            required: dec!(0.0045),
        }
    );
    // Exactly the minimum is enough.
    input.min_quantity = dec!(0.004);
    assert_eq!(accepted(&input).quantity, dec!(0.004));
    // Below one lot it floors to nothing, and zero is never an opportunity.
    input.target_quantity = dec!(0.0009);
    input.min_quantity = dec!(0.0001);
    assert_eq!(
        rejected(&input),
        BasisRejection::InsufficientLiquidity {
            available: dec!(0),
            required: dec!(0.0001),
        }
    );
    // Nothing quoted at the level.
    let mut empty = cash_and_carry();
    empty.buy.level = Some(BasisLevel {
        price: dec!(80794.02),
        contracts: dec!(0),
    });
    assert!(matches!(
        rejected(&empty),
        BasisRejection::InsufficientLiquidity { .. }
    ));
}

#[test]
fn lots_that_are_not_nested_are_refused() {
    let mut draft = future_draft();
    draft.lot = dec!(0.004);
    let mut spot = InstrumentDraft::from(spot_spec());
    spot.lot = dec!(0.003);
    let mut input = cash_and_carry();
    input.buy.spec = InstrumentSpec::new(spot).unwrap();
    input.sell.spec = InstrumentSpec::new(draft).unwrap();
    assert_eq!(
        evaluate_basis(&input),
        Err(BasisError::LotsNotNested {
            buy: dec!(0.003),
            sell: dec!(0.004),
        })
    );
}

#[test]
fn only_a_spot_and_a_linear_dated_future_are_a_basis() {
    let unsupported = |input: &BasisInput| {
        assert!(
            matches!(evaluate_basis(input), Err(BasisError::Unsupported { .. })),
            "{:?}",
            evaluate_basis(input)
        );
    };
    let mut near_far = cash_and_carry();
    let mut far = future_draft();
    far.id = id("BTCUSDT_270326");
    far.expiry_ms = Some(1_806_048_000_000);
    near_far.buy.spec = InstrumentSpec::new(far).unwrap();
    unsupported(&near_far);

    let mut two_spots = cash_and_carry();
    let mut other = InstrumentDraft::from(spot_spec());
    other.id = id("BTCUSDT2");
    two_spots.sell.spec = InstrumentSpec::new(other).unwrap();
    unsupported(&two_spots);

    let mut perpetual = cash_and_carry();
    perpetual.sell.spec = binance_usdm::btcusdt_perpetual().unwrap();
    unsupported(&perpetual);

    let mut inverse = cash_and_carry();
    inverse.sell.spec = deribit::btc_future(id("BTC-25DEC26"), DELIVERY_MS).unwrap();
    unsupported(&inverse);
    // Priced in the same asset as the spot, only its margining is wrong.
    let mut same_price = InstrumentDraft::from(inverse.sell.spec.clone());
    same_price.price_asset = asset("USDT");
    same_price.size = ContractSize::quote_value(dec!(10), asset("USDT")).unwrap();
    inverse.sell.spec = InstrumentSpec::new(same_price).unwrap();
    unsupported(&inverse);

    let mut option = cash_and_carry();
    option.sell.spec = deribit::btc_option(
        id("BTC-25DEC26-80000-C"),
        DELIVERY_MS,
        dec!(80000),
        OptionStyle::Call,
    )
    .unwrap();
    unsupported(&option);

    let mut underlying = cash_and_carry();
    let mut eth = future_draft();
    eth.underlying = asset("ETH");
    eth.size = ContractSize::base(dec!(1), asset("ETH")).unwrap();
    underlying.sell.spec = InstrumentSpec::new(eth).unwrap();
    unsupported(&underlying);

    let mut priced = cash_and_carry();
    let mut usdc = future_draft();
    usdc.price_asset = asset("USDC");
    usdc.settlement_asset = asset("USDC");
    priced.sell.spec = InstrumentSpec::new(usdc).unwrap();
    unsupported(&priced);
}

#[test]
fn time_limits_are_exact_on_both_sides() {
    // A quote exactly at the maximum age is fresh; a millisecond older is stale.
    let mut input = cash_and_carry();
    input.sell.quoted_at_ms = AS_OF_MS - 1000;
    assert_eq!(accepted(&input).quantity, dec!(0.003));
    input.sell.quoted_at_ms = AS_OF_MS - 1001;
    assert_eq!(
        rejected(&input),
        BasisRejection::StaleQuote {
            instrument: "BTCUSDT_261225".to_string(),
            age_ms: 1001,
        }
    );
    let mut input = cash_and_carry();
    input.buy.quoted_at_ms = AS_OF_MS - 1001;
    assert!(matches!(
        rejected(&input),
        BasisRejection::StaleQuote { .. }
    ));

    // A quote from after the decision is a fault, for either leg.
    let mut input = cash_and_carry();
    input.buy.quoted_at_ms = AS_OF_MS + 1;
    assert!(matches!(
        evaluate_basis(&input),
        Err(BasisError::QuoteFromTheFuture { .. })
    ));
    let mut input = cash_and_carry();
    input.sell.quoted_at_ms = AS_OF_MS + 1;
    assert!(matches!(
        evaluate_basis(&input),
        Err(BasisError::QuoteFromTheFuture { .. })
    ));

    // The future: a millisecond before its expiry, at it and after.
    let at = |as_of_ms: u64| {
        let mut input = cash_and_carry();
        input.as_of_ms = as_of_ms;
        input.buy.quoted_at_ms = as_of_ms;
        input.sell.quoted_at_ms = as_of_ms;
        input
    };
    assert_eq!(accepted(&at(DELIVERY_MS - 1)).quantity, dec!(0.003));
    for as_of_ms in [DELIVERY_MS, DELIVERY_MS + 1] {
        assert_eq!(
            rejected(&at(as_of_ms)),
            BasisRejection::Expired {
                instrument: "BTCUSDT_261225".to_string(),
                expiry_ms: DELIVERY_MS,
            }
        );
    }
    // Beyond what the evaluation can represent.
    let beyond = i64::MAX as u64 + 1;
    assert_eq!(
        evaluate_basis(&at(beyond)),
        Err(BasisError::TimeOutOfRange {
            field: "as_of_ms",
            value: beyond,
        })
    );
    let mut input = cash_and_carry();
    input.buy.quoted_at_ms = beyond;
    assert_eq!(
        evaluate_basis(&input),
        Err(BasisError::TimeOutOfRange {
            field: "quoted_at_ms",
            value: beyond,
        })
    );
    let mut draft = future_draft();
    draft.expiry_ms = Some(beyond);
    let mut input = cash_and_carry();
    input.sell.spec = InstrumentSpec::new(draft).unwrap();
    assert_eq!(
        evaluate_basis(&input),
        Err(BasisError::TimeOutOfRange {
            field: "expiry_ms",
            value: beyond,
        })
    );
}

#[test]
fn a_leg_that_cannot_be_executed_is_discarded() {
    let mut input = cash_and_carry();
    input.buy.level = None;
    assert_eq!(
        rejected(&input),
        BasisRejection::MissingQuote {
            instrument: "BTCUSDT".to_string(),
        }
    );
    let mut input = cash_and_carry();
    input.sell.level = None;
    assert_eq!(
        rejected(&input),
        BasisRejection::MissingQuote {
            instrument: "BTCUSDT_261225".to_string(),
        }
    );
    // 81389.75 is off the future's 0.1 tick, and 80794.015 off the spot's 0.01.
    let mut input = cash_and_carry();
    input.sell.level = Some(BasisLevel {
        price: dec!(81389.75),
        contracts: dec!(0.003),
    });
    assert_eq!(
        rejected(&input),
        BasisRejection::OffTick {
            instrument: "BTCUSDT_261225".to_string(),
            price: dec!(81389.75),
        }
    );
    let mut input = cash_and_carry();
    input.buy.level = Some(BasisLevel {
        price: dec!(80794.015),
        contracts: dec!(1),
    });
    assert!(matches!(rejected(&input), BasisRejection::OffTick { .. }));
}

#[test]
fn an_edge_must_be_positive_and_reach_the_threshold() {
    // A bid above the ask, at it and below it.
    let spread = |bid: Decimal, ask: Decimal| {
        let mut input = cash_and_carry();
        input.buy = quote(spot_spec(), ask, dec!(1), dec!(0));
        input.sell = quote(future_spec(), bid, dec!(0.003), dec!(0));
        input
    };
    assert_eq!(
        accepted(&spread(dec!(80794.1), dec!(80794))).gross,
        dec!(0.0003)
    );
    assert_eq!(
        rejected(&spread(dec!(80794), dec!(80794))),
        BasisRejection::NoGrossEdge { gross: dec!(0) }
    );
    assert_eq!(
        rejected(&spread(dec!(80794), dec!(80794.1))),
        BasisRejection::NoGrossEdge {
            gross: dec!(-0.0003)
        }
    );
    let mut input = cash_and_carry();
    let net = accepted(&input).net;
    assert_eq!(net, dec!(1.42257339));
    // The threshold is money, and inclusive.
    input.min_net_edge = net;
    assert_eq!(accepted(&input).net, net);
    input.min_net_edge = net + dec!(0.00000001);
    match rejected(&input) {
        BasisRejection::NetEdgeBelowThreshold {
            gross,
            costs,
            net: reported,
            min_net_edge,
        } => {
            assert_eq!(gross, dec!(1.78704));
            assert_eq!(costs.buy_fee, dec!(0.24238206));
            assert_eq!(costs.total, dec!(0.36446661));
            assert_eq!(reported, net);
            assert_eq!(min_net_edge, net + dec!(0.00000001));
        }
        other => panic!("{other:?}"),
    }
    // Costs that leave exactly nothing are rejected, even with no threshold: a gross
    // of 1 on a notional of 100 and a fee of 100 basis points.
    let mut even = cash_and_carry();
    even.target_quantity = dec!(1);
    even.buy = quote(spot_spec(), dec!(100), dec!(1), dec!(100));
    even.sell = quote(future_spec(), dec!(101), dec!(1), dec!(0));
    match rejected(&even) {
        BasisRejection::NetEdgeBelowThreshold { gross, net, .. } => {
            assert_eq!(gross, dec!(1));
            assert_eq!(net, dec!(0));
        }
        other => panic!("{other:?}"),
    }
    even.buy.fee_bps = dec!(99);
    assert_eq!(accepted(&even).net, dec!(0.01));
    // Exit costs alone can eat the edge.
    let mut input = cash_and_carry();
    input.buy.exit_bps = dec!(100);
    assert!(matches!(
        rejected(&input),
        BasisRejection::NetEdgeBelowThreshold { .. }
    ));
}

#[test]
fn a_value_out_of_range_is_refused() {
    let invalid = |input: &BasisInput, expected: &str| match evaluate_basis(input) {
        Err(BasisError::InvalidValue { field, .. }) => assert_eq!(field, expected),
        other => panic!("{expected}: {other:?}"),
    };
    let mut input = cash_and_carry();
    input.buy.fee_bps = dec!(10001);
    invalid(&input, "fee_bps");
    let mut input = cash_and_carry();
    input.sell.fee_bps = dec!(-10001);
    invalid(&input, "fee_bps");
    let mut input = cash_and_carry();
    input.buy.exit_bps = dec!(10001);
    invalid(&input, "exit_bps");
    let mut input = cash_and_carry();
    input.sell.exit_bps = dec!(10001);
    invalid(&input, "exit_bps");
    let mut input = cash_and_carry();
    input.carry = vec![BasisCarry {
        kind: CarryKind::Other,
        on: Action::Buy,
        period_bps: dec!(10001),
    }];
    invalid(&input, "period_bps");
    for rate in [dec!(-0.01), dec!(1.01)] {
        let mut input = cash_and_carry();
        input.future_margin_rate = rate;
        invalid(&input, "future_margin_rate");
    }
    // Both ends of the rate are valid.
    for (rate, capital) in [
        (dec!(0), dec!(242.38206)),
        (dec!(1), dec!(242.38206) + dec!(244.1691)),
    ] {
        let mut input = cash_and_carry();
        input.future_margin_rate = rate;
        assert_eq!(accepted(&input).capital, capital);
    }
    let mut input = cash_and_carry();
    input.min_net_edge = dec!(-1);
    invalid(&input, "min_net_edge");
    let mut input = cash_and_carry();
    input.target_quantity = dec!(0);
    invalid(&input, "target_quantity");
    let mut input = cash_and_carry();
    input.min_quantity = dec!(0);
    invalid(&input, "min_quantity");
    let mut input = cash_and_carry();
    input.min_quantity = dec!(2);
    invalid(&input, "min_quantity");
    let mut input = cash_and_carry();
    input.sell.level = Some(BasisLevel {
        price: dec!(0),
        contracts: dec!(1),
    });
    invalid(&input, "price");
    let mut input = cash_and_carry();
    input.buy.level = Some(BasisLevel {
        price: dec!(80794.02),
        contracts: dec!(-1),
    });
    invalid(&input, "contracts");
    // A carry on neither leg.
    let mut input = cash_and_carry();
    input.carry = vec![BasisCarry {
        kind: CarryKind::Other,
        on: Action::Other,
        period_bps: dec!(1),
    }];
    assert_eq!(
        evaluate_basis(&input),
        Err(BasisError::CarryOnNeitherLeg { on: Action::Other })
    );
}

#[test]
fn a_request_that_cannot_be_answered_does_not_depend_on_the_market() {
    // Each of these would be discarded by the market, and each is still refused.
    let markets: [fn(&mut BasisInput); 5] = [
        |input| input.buy.quoted_at_ms = AS_OF_MS - 1001,
        |input| input.buy.level = None,
        |input| {
            input.sell.level = Some(BasisLevel {
                price: dec!(81389.75),
                contracts: dec!(0.003),
            });
        },
        |input| {
            input.buy.level = Some(BasisLevel {
                price: dec!(80794.02),
                contracts: dec!(0),
            });
        },
        |input| {
            input.as_of_ms = DELIVERY_MS;
            input.buy.quoted_at_ms = DELIVERY_MS;
            input.sell.quoted_at_ms = DELIVERY_MS;
        },
    ];
    for market in markets {
        let mut discarded = cash_and_carry();
        market(&mut discarded);
        assert!(matches!(
            evaluate_basis(&discarded),
            Ok(BasisEvaluation::Rejected(_))
        ));

        let mut carry_on_neither = discarded.clone();
        carry_on_neither.carry = vec![BasisCarry {
            kind: CarryKind::Other,
            on: Action::Other,
            period_bps: dec!(1),
        }];
        assert_eq!(
            evaluate_basis(&carry_on_neither),
            Err(BasisError::CarryOnNeitherLeg { on: Action::Other })
        );
        let mut bad_rate = discarded.clone();
        bad_rate.sell.fee_bps = dec!(10001);
        assert!(matches!(
            evaluate_basis(&bad_rate),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut bad_margin = discarded.clone();
        bad_margin.future_margin_rate = dec!(2);
        assert!(matches!(
            evaluate_basis(&bad_margin),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut bad_edge = discarded.clone();
        bad_edge.min_net_edge = dec!(-1);
        assert!(matches!(
            evaluate_basis(&bad_edge),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut future_dated = discarded;
        future_dated.sell.quoted_at_ms = future_dated.as_of_ms + 1;
        assert!(matches!(
            evaluate_basis(&future_dated),
            Err(BasisError::QuoteFromTheFuture { .. })
        ));
    }
}
