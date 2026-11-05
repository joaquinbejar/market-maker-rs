//! The net result of a calendar spread between two linear dated futures, under an
//! assumed exit. Every expected amount below is worked out by hand from its quotes.

#![cfg(feature = "arbitrage")]

use financial_types::Action;
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::instruments::basis::{
    BasisCarry, BasisCarryCost, BasisError, BasisLegPlan, BasisLevel, BasisQuote, BasisRejection,
    CarryKind,
};
use market_maker_rs::instruments::calendar::{
    CalendarEvaluation, CalendarInput, CalendarOpportunity, CalendarSide, evaluate_calendar,
};
use market_maker_rs::instruments::reference::{binance_usdm, deribit};
use market_maker_rs::instruments::{
    AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
    Margining, TickSchedule,
};
use market_maker_rs::prelude::OptionStyle;

const NEAR_MS: u64 = 1_798_185_600_000;
const FAR_MS: u64 = 1_806_048_000_000;
const AS_OF_MS: u64 = 1_793_839_762_000;
const EXIT_MS: u64 = 1_798_000_000_000;

fn asset(code: &str) -> AssetCode {
    AssetCode::new(code).unwrap()
}

fn draft(name: &str, expiry_ms: u64, multiplier: Decimal, lot: Decimal) -> InstrumentDraft {
    InstrumentDraft {
        id: InstrumentId::new(name).unwrap(),
        kind: ContractKind::Future,
        margining: Margining::Linear,
        underlying: asset("BTC"),
        price_asset: asset("USDT"),
        settlement_asset: asset("USDT"),
        size: ContractSize::base(multiplier, asset("BTC")).unwrap(),
        price_ticks: TickSchedule::fixed(dec!(0.1)).unwrap(),
        lot,
        expiry_ms: Some(expiry_ms),
        option: None,
    }
}

fn spec(name: &str, expiry_ms: u64, multiplier: Decimal, lot: Decimal) -> InstrumentSpec {
    InstrumentSpec::new(draft(name, expiry_ms, multiplier, lot)).unwrap()
}

fn near() -> InstrumentSpec {
    spec("NEAR", NEAR_MS, dec!(1), dec!(0.001))
}

fn far() -> InstrumentSpec {
    spec("FAR", FAR_MS, dec!(1), dec!(0.001))
}

fn quote(
    spec: InstrumentSpec,
    price: Decimal,
    contracts: Decimal,
    fee_bps: Decimal,
    exit_bps: Decimal,
) -> BasisQuote {
    BasisQuote {
        spec,
        level: Some(BasisLevel { price, contracts }),
        quoted_at_ms: AS_OF_MS,
        fee_bps,
        exit_bps,
    }
}

fn request(buy: BasisQuote, sell: BasisQuote, exit_spread: Decimal) -> CalendarInput {
    CalendarInput {
        as_of_ms: AS_OF_MS,
        buy,
        sell,
        expected_exit_spread: exit_spread,
        exit_at_ms: EXIT_MS,
        target_quantity: dec!(2),
        min_quantity: dec!(0.001),
        max_quote_age_ms: 1000,
        future_margin_rate: dec!(0.1),
        carry: Vec::new(),
        min_net_edge: dec!(0),
    }
}

/// Buys the near at 100 and sells the far at 103, assuming the spread ends at 1.
fn buy_near() -> CalendarInput {
    request(
        quote(near(), dec!(100), dec!(5), dec!(5), dec!(2)),
        quote(far(), dec!(103), dec!(5), dec!(5), dec!(5)),
        dec!(1),
    )
}

/// Sells the near at 100 and buys the far at 99, assuming the spread ends at 2.
fn sell_near() -> CalendarInput {
    request(
        quote(far(), dec!(99), dec!(5), dec!(4), dec!(5)),
        quote(near(), dec!(100), dec!(5), dec!(5), dec!(2)),
        dec!(2),
    )
}

fn accepted(input: &CalendarInput) -> CalendarOpportunity {
    match evaluate_calendar(input).unwrap() {
        CalendarEvaluation::Accepted(opportunity) => opportunity,
        CalendarEvaluation::Rejected(rejection) => panic!("rejected: {rejection:?}"),
    }
}

fn rejected(input: &CalendarInput) -> BasisRejection {
    match evaluate_calendar(input).unwrap() {
        CalendarEvaluation::Rejected(rejection) => rejection,
        CalendarEvaluation::Accepted(opportunity) => panic!("accepted: {opportunity:?}"),
    }
}

#[test]
fn buying_the_near_earns_what_the_spread_falls() {
    let mut input = buy_near();
    input.carry = vec![BasisCarry {
        kind: CarryKind::Financing,
        on: Action::Buy,
        period_bps: dec!(10),
    }];
    let opportunity = accepted(&input);
    assert_eq!(opportunity.asset, asset("USDT"));
    assert_eq!(opportunity.quantity, dec!(2));
    assert_eq!(opportunity.side, CalendarSide::BuyNearSellFar);
    assert_eq!(
        opportunity.buy,
        BasisLegPlan {
            instrument: "NEAR".to_string(),
            price: dec!(100),
            contracts: dec!(2),
        }
    );
    assert_eq!(
        opportunity.sell,
        BasisLegPlan {
            instrument: "FAR".to_string(),
            price: dec!(103),
            contracts: dec!(2),
        }
    );
    assert_eq!(opportunity.entry_spread, dec!(3));
    assert_eq!(opportunity.expected_exit_spread, dec!(1));
    assert_eq!(opportunity.exit_at_ms, EXIT_MS);
    // 2 * (3 - 1).
    assert_eq!(opportunity.gross, dec!(4));
    // On 200 and 206: 5 bps and 5 bps in, 2 bps and 5 bps out, and 10 bps of carry.
    assert_eq!(opportunity.costs.buy_fee, dec!(0.10));
    assert_eq!(opportunity.costs.sell_fee, dec!(0.103));
    assert_eq!(opportunity.costs.buy_exit, dec!(0.04));
    assert_eq!(opportunity.costs.sell_exit, dec!(0.103));
    assert_eq!(
        opportunity.costs.carry,
        vec![BasisCarryCost {
            kind: CarryKind::Financing,
            on: Action::Buy,
            amount: dec!(0.20),
        }]
    );
    assert_eq!(opportunity.costs.total, dec!(0.546));
    assert_eq!(opportunity.net, dec!(3.454));
    // (200 + 206) * 0.1.
    assert_eq!(opportunity.capital, dec!(40.6));
}

#[test]
fn selling_the_near_earns_what_the_spread_rises_from_a_negative_entry() {
    let mut input = sell_near();
    input.carry = vec![
        BasisCarry {
            kind: CarryKind::Other,
            on: Action::Sell,
            period_bps: dec!(-10),
        },
        BasisCarry {
            kind: CarryKind::Financing,
            on: Action::Buy,
            period_bps: dec!(3),
        },
    ];
    let opportunity = accepted(&input);
    assert_eq!(opportunity.side, CalendarSide::SellNearBuyFar);
    assert_eq!(opportunity.buy.instrument, "FAR");
    assert_eq!(opportunity.sell.instrument, "NEAR");
    // Far minus near whichever leg is bought: 99 - 100.
    assert_eq!(opportunity.entry_spread, dec!(-1));
    assert_eq!(opportunity.expected_exit_spread, dec!(2));
    // 2 * (2 - (-1)).
    assert_eq!(opportunity.gross, dec!(6));
    // The far leg, on 198: 4 bps in, 5 bps out, 3 bps of carry.
    assert_eq!(opportunity.costs.buy_fee, dec!(0.0792));
    assert_eq!(opportunity.costs.buy_exit, dec!(0.099));
    // The near leg, on 200: 5 bps in, 2 bps out, 10 bps of benefit.
    assert_eq!(opportunity.costs.sell_fee, dec!(0.10));
    assert_eq!(opportunity.costs.sell_exit, dec!(0.04));
    assert_eq!(
        opportunity.costs.carry,
        vec![
            BasisCarryCost {
                kind: CarryKind::Other,
                on: Action::Sell,
                amount: dec!(-0.20),
            },
            BasisCarryCost {
                kind: CarryKind::Financing,
                on: Action::Buy,
                amount: dec!(0.0594),
            },
        ]
    );
    assert_eq!(opportunity.costs.total, dec!(0.1776));
    assert_eq!(opportunity.net, dec!(5.8224));
    // (198 + 200) * 0.1.
    assert_eq!(opportunity.capital, dec!(39.8));
}

/// The upstream evaluation rejects a sell price below the buy price, and this passes.
#[test]
fn a_non_positive_entry_with_rebates_can_pass() {
    let mut input = request(
        quote(near(), dec!(100), dec!(1), dec!(-1), dec!(0)),
        quote(far(), dec!(99), dec!(1), dec!(-1), dec!(0)),
        dec!(-3),
    );
    input.target_quantity = dec!(1);
    let opportunity = accepted(&input);
    assert_eq!(opportunity.entry_spread, dec!(-1));
    assert_eq!(opportunity.expected_exit_spread, dec!(-3));
    assert_eq!(opportunity.gross, dec!(2));
    assert_eq!(opportunity.costs.buy_fee, dec!(-0.01));
    assert_eq!(opportunity.costs.sell_fee, dec!(-0.0099));
    assert_eq!(opportunity.costs.total, dec!(-0.0199));
    // A rebate raises the result above the gross.
    assert_eq!(opportunity.net, dec!(2.0199));
}

#[test]
fn a_unit_of_exit_spread_moves_the_result_by_the_size_with_a_sign() {
    let net_at = |mut input: CalendarInput, exit_spread| {
        input.expected_exit_spread = exit_spread;
        accepted(&input).net
    };
    assert_eq!(
        net_at(buy_near(), dec!(1)) - net_at(buy_near(), dec!(2)),
        dec!(2)
    );
    assert_eq!(
        net_at(sell_near(), dec!(3)) - net_at(sell_near(), dec!(2)),
        dec!(2)
    );
}

#[test]
fn no_edge_under_the_assumption_is_discarded_on_either_side() {
    let cases = [
        // At the entry spread nothing is left.
        (buy_near(), dec!(3), dec!(0)),
        (sell_near(), dec!(-1), dec!(0)),
        // Past it, the assumption is a loss.
        (buy_near(), dec!(4), dec!(-2)),
        (sell_near(), dec!(-2), dec!(-2)),
    ];
    for (mut input, exit_spread, gross) in cases {
        input.expected_exit_spread = exit_spread;
        assert_eq!(rejected(&input), BasisRejection::NoGrossEdge { gross });
    }
}

#[test]
fn the_threshold_is_exact_and_a_zero_result_never_passes() {
    // q = 1, gross 1, costs 0.10 + 0.103 on 100 and 103.
    let mut input = request(
        quote(near(), dec!(100), dec!(1), dec!(10), dec!(0)),
        quote(far(), dec!(103), dec!(1), dec!(10), dec!(0)),
        dec!(2),
    );
    input.target_quantity = dec!(1);
    input.min_net_edge = dec!(0.797);
    assert_eq!(accepted(&input).net, dec!(0.797));
    input.min_net_edge = dec!(0.7971);
    match rejected(&input) {
        BasisRejection::NetEdgeBelowThreshold {
            gross,
            costs,
            net,
            min_net_edge,
        } => {
            assert_eq!(gross, dec!(1));
            assert_eq!(costs.total, dec!(0.203));
            assert_eq!(net, dec!(0.797));
            assert_eq!(min_net_edge, dec!(0.7971));
        }
        other => panic!("{other:?}"),
    }

    // The other side, from an entry of zero with rebates: q = 1, gross 1, and -1 bps on
    // each 100 raises it to exactly 1.02.
    let mut other = request(
        quote(far(), dec!(100), dec!(1), dec!(-1), dec!(0)),
        quote(near(), dec!(100), dec!(1), dec!(-1), dec!(0)),
        dec!(1),
    );
    other.target_quantity = dec!(1);
    other.min_net_edge = dec!(1.02);
    let opportunity = accepted(&other);
    assert_eq!(opportunity.side, CalendarSide::SellNearBuyFar);
    assert_eq!(opportunity.entry_spread, dec!(0));
    assert_eq!(opportunity.gross, dec!(1));
    assert_eq!(opportunity.costs.total, dec!(-0.02));
    assert_eq!(opportunity.net, dec!(1.02));
    other.min_net_edge = dec!(1.0201);
    assert!(matches!(
        rejected(&other),
        BasisRejection::NetEdgeBelowThreshold { .. }
    ));

    // A gross of 0.2 on 10 bps of each 100: exactly nothing is left.
    let mut zero = request(
        quote(near(), dec!(100), dec!(1), dec!(10), dec!(0)),
        quote(far(), dec!(100), dec!(1), dec!(10), dec!(0)),
        dec!(-0.2),
    );
    zero.target_quantity = dec!(1);
    match rejected(&zero) {
        BasisRejection::NetEdgeBelowThreshold { net, .. } => assert_eq!(net, dec!(0)),
        other => panic!("{other:?}"),
    }
}

/// The size is in the underlying and each leg has its own contracts, whichever leg is
/// bought, whichever has the finer contract and whichever limits the size.
#[test]
fn different_multipliers_are_counted_in_the_underlying() {
    let coarse = |name, expiry| spec(name, expiry, dec!(0.01), dec!(1));
    let fine = |name, expiry| spec(name, expiry, dec!(1), dec!(0.001));
    // (near spec, far spec, near is the coarse one).
    let contracts = [
        (fine("NEAR", NEAR_MS), coarse("FAR", FAR_MS), false),
        (coarse("NEAR", NEAR_MS), fine("FAR", FAR_MS), true),
    ];
    // (the coarse leg's contracts, the fine leg's contracts, the size in the underlying,
    // the coarse leg's contracts and the fine leg's contracts to trade).
    let limits = [
        // The coarse leg limits: 30 * 0.01.
        (dec!(30), dec!(5), dec!(0.3), dec!(30), dec!(0.3)),
        // The fine leg limits: 0.255 is floored to the coarser lot exposure of 0.01.
        (dec!(500), dec!(0.255), dec!(0.25), dec!(25), dec!(0.25)),
    ];
    for (near_spec, far_spec, near_is_coarse) in contracts {
        for (coarse_size, fine_size, q, coarse_out, fine_out) in limits {
            let (near_size, far_size, near_out, far_out) = if near_is_coarse {
                (coarse_size, fine_size, coarse_out, fine_out)
            } else {
                (fine_size, coarse_size, fine_out, coarse_out)
            };
            let near_quote = quote(near_spec.clone(), dec!(100), near_size, dec!(10), dec!(0));
            let far_quote = quote(far_spec.clone(), dec!(103), far_size, dec!(20), dec!(0));

            // Buys the near: 3 - 1 per unit.
            let mut input = request(near_quote.clone(), far_quote.clone(), dec!(1));
            input.target_quantity = dec!(10);
            let bought = accepted(&input);
            assert_eq!(bought.side, CalendarSide::BuyNearSellFar);
            assert_eq!(bought.quantity, q);
            assert_eq!(bought.buy.contracts, near_out);
            assert_eq!(bought.sell.contracts, far_out);
            assert_eq!(bought.gross, dec!(2) * q);
            assert_eq!(bought.costs.buy_fee, dec!(0.1) * q);
            assert_eq!(bought.costs.sell_fee, dec!(0.206) * q);

            // Sells the near: 5 - 3 per unit.
            let mut input = request(far_quote, near_quote, dec!(5));
            input.target_quantity = dec!(10);
            let sold = accepted(&input);
            assert_eq!(sold.side, CalendarSide::SellNearBuyFar);
            assert_eq!(sold.quantity, q);
            assert_eq!(sold.buy.contracts, far_out);
            assert_eq!(sold.sell.contracts, near_out);
            assert_eq!(sold.gross, dec!(2) * q);
            assert_eq!(sold.costs.buy_fee, dec!(0.206) * q);
            assert_eq!(sold.costs.sell_fee, dec!(0.1) * q);

            for opportunity in [bought, sold] {
                assert_eq!(opportunity.entry_spread, dec!(3));
                assert_eq!(opportunity.costs.total, dec!(0.306) * q);
                assert_eq!(opportunity.net, dec!(1.694) * q);
                assert_eq!(opportunity.capital, dec!(20.3) * q);
            }
        }
    }
}

#[test]
fn lots_that_are_not_nested_are_refused() {
    let mut input = buy_near();
    input.buy.spec = spec("NEAR", NEAR_MS, dec!(1), dec!(0.002));
    input.sell.spec = spec("FAR", FAR_MS, dec!(1), dec!(0.003));
    assert_eq!(
        evaluate_calendar(&input),
        Err(BasisError::LotsNotNested {
            buy: dec!(0.002),
            sell: dec!(0.003),
        })
    );
}

#[test]
fn the_exit_is_after_the_decision_and_before_the_near_expiry() {
    let outside = |input: &CalendarInput| {
        assert_eq!(
            evaluate_calendar(input),
            Err(BasisError::ExitOutsideWindow {
                exit_at_ms: input.exit_at_ms,
                as_of_ms: input.as_of_ms,
                near_expiry_ms: NEAR_MS,
            })
        );
    };
    for mut input in [buy_near(), sell_near()] {
        for exit_at_ms in [AS_OF_MS + 1, NEAR_MS - 1] {
            input.exit_at_ms = exit_at_ms;
            assert_eq!(accepted(&input).exit_at_ms, exit_at_ms);
        }
        // At the near expiry the near leg settles, which is not modelled.
        for exit_at_ms in [NEAR_MS, NEAR_MS + 1, AS_OF_MS, AS_OF_MS - 1] {
            input.exit_at_ms = exit_at_ms;
            outside(&input);
        }
    }
}

#[test]
fn an_exit_at_or_before_the_decision_is_a_refusal_only_while_the_near_is_alive() {
    let expired = BasisRejection::Expired {
        instrument: "NEAR".to_string(),
        expiry_ms: NEAR_MS,
    };
    for mut input in [buy_near(), sell_near()] {
        input.buy.level = None;
        // Alive, and not depending on the missing quote.
        input.exit_at_ms = AS_OF_MS;
        assert!(matches!(
            evaluate_calendar(&input),
            Err(BasisError::ExitOutsideWindow { .. })
        ));

        // The near expired at the decision, and after it.
        input.as_of_ms = NEAR_MS;
        input.buy.quoted_at_ms = NEAR_MS;
        input.sell.quoted_at_ms = NEAR_MS;
        input.exit_at_ms = NEAR_MS - 1;
        assert_eq!(rejected(&input), expired);
        input.as_of_ms = NEAR_MS + 1;
        assert_eq!(rejected(&input), expired);

        // An exit at or after the expiry cannot be answered at any time.
        for exit_at_ms in [NEAR_MS, NEAR_MS + 1] {
            input.exit_at_ms = exit_at_ms;
            assert!(matches!(
                evaluate_calendar(&input),
                Err(BasisError::ExitOutsideWindow { .. })
            ));
        }
    }
}

#[test]
fn only_two_linear_dated_futures_of_one_product_are_a_calendar() {
    let unsupported = |input: &CalendarInput| {
        assert!(
            matches!(
                evaluate_calendar(input),
                Err(BasisError::Unsupported { .. })
            ),
            "{:?}",
            evaluate_calendar(input)
        );
    };
    let mut spot_draft = draft("SPOT", NEAR_MS, dec!(1), dec!(0.001));
    spot_draft.kind = ContractKind::Spot;
    spot_draft.expiry_ms = None;
    let others = [
        InstrumentSpec::new(spot_draft).unwrap(),
        binance_usdm::btcusdt_perpetual().unwrap(),
        deribit::btc_future(InstrumentId::new("BTC-25DEC26").unwrap(), NEAR_MS).unwrap(),
        deribit::btc_option(
            InstrumentId::new("BTC-25DEC26-80000-C").unwrap(),
            NEAR_MS,
            dec!(80000),
            OptionStyle::Call,
        )
        .unwrap(),
    ];
    for other in others {
        let mut input = buy_near();
        input.buy.spec = other.clone();
        unsupported(&input);
        let mut input = buy_near();
        input.sell.spec = other;
        unsupported(&input);
    }

    let mut underlying = buy_near();
    let mut eth = draft("ETH", FAR_MS, dec!(1), dec!(0.001));
    eth.underlying = asset("ETH");
    eth.size = ContractSize::base(dec!(1), asset("ETH")).unwrap();
    underlying.sell.spec = InstrumentSpec::new(eth).unwrap();
    unsupported(&underlying);

    let mut priced = buy_near();
    let mut usdc = draft("USDC", FAR_MS, dec!(1), dec!(0.001));
    usdc.price_asset = asset("USDC");
    usdc.settlement_asset = asset("USDC");
    priced.sell.spec = InstrumentSpec::new(usdc).unwrap();
    unsupported(&priced);

    let mut same_contract = buy_near();
    same_contract.sell.spec = near();
    unsupported(&same_contract);

    // One identifier with two expiries is not two contracts.
    let mut same_id = buy_near();
    same_id.sell.spec = spec("NEAR", FAR_MS, dec!(1), dec!(0.001));
    unsupported(&same_id);

    let mut same_expiry = buy_near();
    same_expiry.sell.spec = spec("FAR", NEAR_MS, dec!(1), dec!(0.001));
    unsupported(&same_expiry);
}

#[test]
fn a_leg_that_cannot_be_executed_is_discarded() {
    for near_is_bought in [true, false] {
        let base = || {
            if near_is_bought {
                buy_near()
            } else {
                sell_near()
            }
        };
        let mut input = base();
        input.sell.level = None;
        assert_eq!(
            rejected(&input),
            BasisRejection::MissingQuote {
                instrument: input.sell.spec.id().as_str().to_string(),
            }
        );
        let mut input = base();
        input.buy.level = None;
        assert_eq!(
            rejected(&input),
            BasisRejection::MissingQuote {
                instrument: input.buy.spec.id().as_str().to_string(),
            }
        );

        // Exactly at the maximum age is fresh, and a millisecond older is stale.
        let mut input = base();
        input.buy.quoted_at_ms = AS_OF_MS - 1000;
        input.sell.quoted_at_ms = AS_OF_MS - 1000;
        accepted(&input);
        input.sell.quoted_at_ms = AS_OF_MS - 1001;
        assert_eq!(
            rejected(&input),
            BasisRejection::StaleQuote {
                instrument: input.sell.spec.id().as_str().to_string(),
                age_ms: 1001,
            }
        );

        let mut input = base();
        let contracts = dec!(5);
        input.buy.level = Some(BasisLevel {
            price: dec!(99.05),
            contracts,
        });
        assert_eq!(
            rejected(&input),
            BasisRejection::OffTick {
                instrument: input.buy.spec.id().as_str().to_string(),
                price: dec!(99.05),
            }
        );

        let mut input = base();
        input.min_quantity = dec!(1);
        input.sell.level = Some(BasisLevel {
            price: input.sell.level.unwrap().price,
            contracts: dec!(0.9995),
        });
        assert_eq!(
            rejected(&input),
            BasisRejection::InsufficientLiquidity {
                available: dec!(0.999),
                required: dec!(1),
            }
        );
    }
    // No size at all.
    let mut input = buy_near();
    input.buy.level = Some(BasisLevel {
        price: dec!(100),
        contracts: dec!(0),
    });
    assert_eq!(
        rejected(&input),
        BasisRejection::InsufficientLiquidity {
            available: dec!(0),
            required: dec!(0.001),
        }
    );
}

#[test]
fn a_value_out_of_range_is_refused() {
    let invalid = |change: fn(&mut CalendarInput), expected: &str| {
        let mut input = buy_near();
        change(&mut input);
        match evaluate_calendar(&input) {
            Err(BasisError::InvalidValue { field, .. }) => assert_eq!(field, expected),
            other => panic!("{expected}: {other:?}"),
        }
    };
    invalid(|i| i.buy.fee_bps = dec!(10001), "fee_bps");
    invalid(|i| i.sell.fee_bps = dec!(-10001), "fee_bps");
    invalid(|i| i.buy.exit_bps = dec!(10001), "exit_bps");
    invalid(|i| i.sell.exit_bps = dec!(-10001), "exit_bps");
    invalid(|i| i.future_margin_rate = dec!(-0.01), "future_margin_rate");
    invalid(|i| i.future_margin_rate = dec!(1.01), "future_margin_rate");
    invalid(|i| i.min_net_edge = dec!(-1), "min_net_edge");
    invalid(|i| i.target_quantity = dec!(0), "target_quantity");
    invalid(|i| i.min_quantity = dec!(0), "min_quantity");
    invalid(|i| i.min_quantity = dec!(3), "min_quantity");
    invalid(
        |i| {
            i.carry = vec![BasisCarry {
                kind: CarryKind::Other,
                on: Action::Buy,
                period_bps: dec!(10001),
            }];
        },
        "period_bps",
    );
    invalid(
        |i| {
            i.sell.level = Some(BasisLevel {
                price: dec!(0),
                contracts: dec!(1),
            });
        },
        "price",
    );
    invalid(
        |i| {
            i.buy.level = Some(BasisLevel {
                price: dec!(100),
                contracts: dec!(-1),
            });
        },
        "contracts",
    );

    let mut input = buy_near();
    input.as_of_ms = u64::MAX;
    assert_eq!(
        evaluate_calendar(&input),
        Err(BasisError::TimeOutOfRange {
            field: "as_of_ms",
            value: u64::MAX,
        })
    );

    // Both ends of the margin rate are valid.
    for (rate, capital) in [(dec!(0), dec!(0)), (dec!(1), dec!(406))] {
        let mut input = buy_near();
        input.future_margin_rate = rate;
        assert_eq!(accepted(&input).capital, capital);
    }
}

#[test]
fn a_request_that_cannot_be_answered_does_not_depend_on_the_market() {
    // Each of these would be discarded by the market, and each is still refused.
    let markets: [fn(&mut CalendarInput); 6] = [
        |input| input.buy.quoted_at_ms = AS_OF_MS - 1001,
        |input| input.buy.level = None,
        |input| {
            input.sell.level = Some(BasisLevel {
                price: dec!(103.05),
                contracts: dec!(5),
            });
        },
        |input| {
            input.buy.level = Some(BasisLevel {
                price: dec!(100),
                contracts: dec!(0),
            });
        },
        |input| input.expected_exit_spread = dec!(3),
        |input| {
            input.as_of_ms = NEAR_MS;
            input.buy.quoted_at_ms = NEAR_MS;
            input.sell.quoted_at_ms = NEAR_MS;
            input.exit_at_ms = NEAR_MS - 1;
        },
    ];
    for market in markets {
        let mut discarded = buy_near();
        market(&mut discarded);
        assert!(matches!(
            evaluate_calendar(&discarded),
            Ok(CalendarEvaluation::Rejected(_))
        ));

        let mut carry_on_neither = discarded.clone();
        carry_on_neither.carry = vec![BasisCarry {
            kind: CarryKind::Other,
            on: Action::Other,
            period_bps: dec!(1),
        }];
        assert_eq!(
            evaluate_calendar(&carry_on_neither),
            Err(BasisError::CarryOnNeitherLeg { on: Action::Other })
        );
        let mut bad_rate = discarded.clone();
        bad_rate.sell.exit_bps = dec!(10001);
        assert!(matches!(
            evaluate_calendar(&bad_rate),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut bad_margin = discarded.clone();
        bad_margin.future_margin_rate = dec!(2);
        assert!(matches!(
            evaluate_calendar(&bad_margin),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut bad_edge = discarded.clone();
        bad_edge.min_net_edge = dec!(-1);
        assert!(matches!(
            evaluate_calendar(&bad_edge),
            Err(BasisError::InvalidValue { .. })
        ));
        let mut late_exit = discarded.clone();
        late_exit.exit_at_ms = NEAR_MS;
        assert!(matches!(
            evaluate_calendar(&late_exit),
            Err(BasisError::ExitOutsideWindow { .. })
        ));
        let mut same_contract = discarded.clone();
        same_contract.sell.spec = same_contract.buy.spec.clone();
        assert!(matches!(
            evaluate_calendar(&same_contract),
            Err(BasisError::Unsupported { .. })
        ));
        let mut future_dated = discarded;
        future_dated.sell.quoted_at_ms = future_dated.as_of_ms + 1;
        assert!(matches!(
            evaluate_calendar(&future_dated),
            Err(BasisError::QuoteFromTheFuture { .. })
        ));
    }
}

#[test]
fn what_does_not_fit_exactly_is_an_error_and_is_never_rounded() {
    let refused = |input: &CalendarInput, why: &str| {
        assert!(
            matches!(evaluate_calendar(input), Err(BasisError::Evaluation { .. })),
            "{why}: {:?}",
            evaluate_calendar(input)
        );
    };
    // The spreads themselves, on either side.
    let mut input = buy_near();
    input.expected_exit_spread = Decimal::MIN;
    refused(&input, "entry - X");
    let mut input = sell_near();
    input.expected_exit_spread = Decimal::MAX;
    refused(&input, "X - entry");
    // The gross fits per unit but not times the size.
    let mut input = buy_near();
    input.expected_exit_spread = dec!(-70000000000000000000000000000);
    refused(&input, "gross");
    // A gross with more decimals than a Decimal holds.
    let mut input = buy_near();
    input.expected_exit_spread = dec!(0.0000000000000000000000000001);
    input.target_quantity = dec!(0.001);
    refused(&input, "not exact");
    // A notional that does not fit, with a small gross: not a rejection or a zero.
    let mut input = request(
        quote(
            near(),
            dec!(7000000000000000000000000000),
            dec!(1000),
            dec!(0),
            dec!(0),
        ),
        quote(
            far(),
            dec!(7000000000000000000000000001),
            dec!(1000),
            dec!(0),
            dec!(0),
        ),
        dec!(0),
    );
    input.target_quantity = dec!(1000);
    refused(&input, "notional");
}
