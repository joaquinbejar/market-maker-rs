//! Joint option quotes on hand made values. Everything here is synthetic.

use std::sync::Arc;

use margin_rs::ledger::{DerivativeSettlement, Execution, Ledger};
use margin_rs::{
    Asset, ExecutionId, InstrumentId, OrderId, ReservationId, SpotInstrument, TransferId,
};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::multi_underlying::multi_asset::{
    AssetQuotingParams, ContractTerms, FactorRisk, LimitedGreek, MultiAssetModel, OptionConstraint,
    OptionConstraintKind, OptionMarket, OptionQuoter, OptionQuoterConfig, QuotingObjective,
    RiskScenario, RiskView, UnderlyingTerms,
};
use market_maker_rs::options::market_maker::GreeksLimits;
use market_maker_rs::options::pricing::{
    Dynamics, OptionInputs, OptionPricer, OptionValuation, SurfacePricer,
};
use market_maker_rs::options::surface::shocks::{ShockedSurface, SurfaceShock};
use market_maker_rs::options::{FlatSurface, VolSurface};
use market_maker_rs::types::action::Action;
use market_maker_rs::types::error::{MMError, MMResult};
use optionstratlib::OptionStyle;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const NOW: u64 = 1_000_000;
const EXPIRY: u64 = NOW + 30 * 86_400_000;

fn usd() -> Asset {
    Asset::new("USD").unwrap()
}

fn base() -> Asset {
    Asset::new("AAA").unwrap()
}

fn spot_id() -> InstrumentId {
    InstrumentId::new("AAA-USD").unwrap()
}

fn ledger(cash: Decimal) -> Ledger {
    let mut ledger = Ledger::default();
    ledger.register_asset(usd(), 8).unwrap();
    ledger.register_asset(base(), 8).unwrap();
    ledger
        .register_instrument(SpotInstrument::new(spot_id(), base(), usd(), 2, 2).unwrap())
        .unwrap();
    if cash > Decimal::ZERO {
        ledger
            .deposit(TransferId::new("cash").unwrap(), usd(), cash)
            .unwrap();
    }
    ledger
}

/// Takes `quantity` of the underlying, positive for a buy, at 100.
fn hold_spot(ledger: &mut Ledger, quantity: Decimal) {
    let side = if quantity > Decimal::ZERO {
        Action::Buy
    } else {
        Action::Sell
    };
    ledger
        .book_execution_allowing_overdraft(
            &Execution::new(
                ExecutionId::new("hedge").unwrap(),
                spot_id(),
                side,
                dec!(100),
                quantity.abs(),
            ),
            None,
        )
        .unwrap();
}

fn valuation(price: Decimal, delta: Decimal, vega: Decimal) -> OptionValuation {
    OptionValuation {
        price,
        delta,
        gamma: dec!(0.02),
        vega,
        theta: dec!(-36.5),
    }
}

/// Fixed values by strike, moving with the spot by the delta, around a spot of 100.
#[derive(Clone)]
struct Values(Vec<(Decimal, OptionValuation)>);

impl OptionPricer for Values {
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
        let (_, found) = self
            .0
            .iter()
            .find(|(strike, _)| *strike == inputs.strike)
            .ok_or_else(|| MMError::NumericalError(format!("no value at {}", inputs.strike)))?;
        let mut value = *found;
        value.price += value.delta * (inputs.spot - dec!(100));
        Ok(value)
    }
}

fn values() -> Values {
    Values(vec![
        (dec!(100), valuation(dec!(10), dec!(0.5), dec!(20))),
        (dec!(110), valuation(dec!(5), dec!(0.3), dec!(15))),
    ])
}

/// The same values, higher by `by`.
fn raised(by: Decimal) -> Values {
    let mut raised = values();
    for (_, value) in &mut raised.0 {
        value.price += by;
    }
    raised
}

fn contract(name: &str, strike: Decimal) -> ContractTerms {
    ContractTerms {
        name: name.to_string(),
        strike,
        expiry_ms: EXPIRY,
        style: OptionStyle::Call,
        price_scale: 2,
        quantity_scale: 2,
        intensity_scale: dec!(10),
        intensity_decay: dec!(20),
        trade_size: dec!(1),
        inventory_limit: dec!(50),
    }
}

fn underlying() -> UnderlyingTerms {
    UnderlyingTerms {
        instrument: spot_id(),
        intensity_scale: dec!(10),
        intensity_decay: dec!(20),
        trade_size: dec!(1),
        inventory_limit: dec!(100),
    }
}

fn scenario(name: &str, spot_factor: Decimal, pricer: impl OptionPricer + 'static) -> RiskScenario {
    RiskScenario {
        name: name.to_string(),
        spot_factor,
        pricer: Arc::new(pricer),
    }
}

fn config(contracts: Vec<ContractTerms>) -> OptionQuoterConfig {
    OptionQuoterConfig {
        underlying: underlying(),
        contracts,
        cash_asset: usd(),
        pricer: Arc::new(values()),
        factors: FactorRisk {
            spot_volatility: dec!(2),
            volatility_of_volatility: dec!(0.05),
            correlation: dec!(-0.5),
        },
        risk_aversion: dec!(0.08),
        objective: QuotingObjective::RiskAdjustedExpectation,
        risk_view: RiskView::Correlated,
        capital_budget: dec!(10000),
        scenarios: vec![scenario("up", dec!(1), raised(dec!(4)))],
    }
}

fn two_contracts() -> Vec<ContractTerms> {
    vec![contract("C100", dec!(100)), contract("C110", dec!(110))]
}

fn market() -> OptionMarket {
    OptionMarket {
        spot: dec!(100),
        rate: dec!(0),
        carry: dec!(0),
        now_ms: NOW,
    }
}

fn quoter(positions: Vec<Decimal>) -> OptionQuoter {
    OptionQuoter::new(config(two_contracts()))
        .unwrap()
        .with_restored_state(positions, 0)
        .unwrap()
}

fn bid_ask(
    quotes: &[market_maker_rs::multi_underlying::multi_asset::AdmittedOptionQuote],
    contract: usize,
) -> (Decimal, Decimal) {
    let price = |side| {
        quotes
            .iter()
            .find(|quote| quote.contract == contract && quote.side == side)
            .unwrap()
            .price
    };
    (price(Action::Buy), price(Action::Sell))
}

fn kinds(
    constraints: &[OptionConstraint],
    contract: usize,
    side: Action,
) -> Vec<OptionConstraintKind> {
    constraints
        .iter()
        .filter(|constraint| constraint.contract == contract && constraint.side == side)
        .map(|constraint| constraint.kind.clone())
        .collect()
}

fn reserved(ledger: &Ledger) -> Decimal {
    ledger.balance(&usd()).unwrap().reserved
}

fn cash(ledger: &Ledger) -> Decimal {
    ledger.balance(&usd()).unwrap().total
}

/// The model the quoter should build, by hand from the fixture's sensitivities.
fn model_by_hand(view: RiskView) -> MultiAssetModel {
    let (s, v, rho) = (2.0_f64, 0.05_f64, -0.5_f64);
    let cov = |a: (f64, f64), b: (f64, f64)| {
        a.0 * b.0 * s * s + a.1 * b.1 * v * v + (a.0 * b.1 + a.1 * b.0) * rho * s * v
    };
    let loads = [(1.0, 0.0), (0.5, 20.0), (0.3, 15.0)];
    let params: Vec<_> = loads
        .iter()
        .enumerate()
        .map(|(i, load)| {
            let limit = if i == 0 { dec!(100) } else { dec!(50) };
            AssetQuotingParams::new(
                Decimal::try_from(cov(*load, *load).sqrt()).unwrap(),
                dec!(10),
                dec!(20),
                dec!(1),
                limit,
            )
            .unwrap()
        })
        .collect();
    let mut correlation = Vec::new();
    for (i, a) in loads.iter().enumerate() {
        let mut row = Vec::new();
        for (j, b) in loads.iter().enumerate() {
            if i == j {
                row.push(dec!(1));
            } else {
                let (p, q) = if i < j { (a, b) } else { (b, a) };
                let value = cov(*p, *q) / (cov(*p, *p).sqrt() * cov(*q, *q).sqrt());
                row.push(Decimal::from_f64(value.clamp(-1.0, 1.0)).unwrap());
            }
        }
        correlation.push(row);
    }
    MultiAssetModel::new(
        params,
        &correlation,
        dec!(0.08),
        QuotingObjective::RiskAdjustedExpectation,
        view,
    )
    .unwrap()
}

#[test]
fn test_quotes_are_the_value_and_the_model_of_the_sensitivities_by_hand() {
    let mut book = ledger(dec!(100000));
    hold_spot(&mut book, dec!(-1));
    let mut quoter = quoter(vec![dec!(2), dec!(-3)]);
    let round = quoter.quote(&mut book, &market()).unwrap();

    let model = model_by_hand(RiskView::Correlated);
    let inventory = [dec!(-1), dec!(2), dec!(-3)];
    for (contract, asset, value) in [(0, 1, dec!(10)), (1, 2, dec!(5))] {
        let offsets = model.offsets(asset, &inventory).unwrap();
        let (bid, ask) = bid_ask(&round.quotes, contract);
        assert_eq!(
            bid,
            (value - offsets.bid)
                .round_dp_with_strategy(2, rust_decimal::RoundingStrategy::ToNegativeInfinity)
        );
        assert_eq!(
            ask,
            (value + offsets.ask)
                .round_dp_with_strategy(2, rust_decimal::RoundingStrategy::ToPositiveInfinity)
        );
    }
}

#[test]
fn test_a_hedged_position_is_skewed_less_than_a_concentrated_one() {
    let skew = |hedge: Option<Decimal>| {
        let mut book = ledger(dec!(100000));
        if let Some(hedge) = hedge {
            hold_spot(&mut book, hedge);
        }
        let mut quoter = quoter(vec![dec!(4), dec!(0)]);
        let round = quoter.quote(&mut book, &market()).unwrap();
        let (bid, ask) = bid_ask(&round.quotes, 0);
        dec!(10) - (bid + ask) / dec!(2)
    };
    let (concentrated, hedged) = (skew(None), skew(Some(dec!(-2))));
    assert!(concentrated > Decimal::ZERO);
    assert!(
        hedged.abs() < concentrated,
        "{hedged} against {concentrated}"
    );
}

#[test]
fn test_an_offsetting_contract_reduces_the_skew_only_under_the_correlated_view() {
    let skew = |view: RiskView, other: Decimal| {
        let mut settings = config(two_contracts());
        settings.risk_view = view;
        let mut book = ledger(dec!(100000));
        let mut quoter = OptionQuoter::new(settings)
            .unwrap()
            .with_restored_state(vec![dec!(4), other], 0)
            .unwrap();
        let round = quoter.quote(&mut book, &market()).unwrap();
        let (bid, ask) = bid_ask(&round.quotes, 0);
        dec!(10) - (bid + ask) / dec!(2)
    };
    let alone = skew(RiskView::Correlated, dec!(0));
    let offset = skew(RiskView::Correlated, dec!(-5));
    assert!(offset.abs() < alone, "{offset} against {alone}");
    assert_eq!(
        skew(RiskView::Diagonal, dec!(0)),
        skew(RiskView::Diagonal, dec!(-5))
    );
}

#[test]
fn test_a_buy_reserves_its_premium_and_a_sell_its_worst_scenario_from_its_price() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (bid, ask) = bid_ask(&round.quotes, 0);
    let held = |side| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == side)
            .unwrap()
            .clone()
    };
    assert_eq!(held(Action::Buy).reserved, bid);
    assert_eq!(held(Action::Sell).reserved, dec!(14) - ask);
    assert_eq!(reserved(&book), bid + dec!(14) - ask);
    assert_eq!(
        book.reservation(&ReservationId::new(held(Action::Buy).order_id.as_str()).unwrap())
            .map(|(_, amount)| amount),
        Some(bid)
    );
    assert_eq!(round.capital.reserved, reserved(&book));
}

#[test]
fn test_a_sale_prices_its_capital_from_the_worst_of_every_scenario() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![
        scenario("small", dec!(1), raised(dec!(2))),
        scenario("spot", dec!(1.2), values()),
        scenario("down", dec!(0.5), raised(dec!(1))),
    ];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let (_, ask) = bid_ask(&round.quotes, 0);
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    // A 20% higher spot at a delta of 0.5 is worth 10 more, the worst of the three.
    assert_eq!(sell.reserved, dec!(20) - ask);
}

#[test]
fn test_a_sale_above_every_scenario_holds_nothing() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("down", dec!(0.5), values())];
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(settings).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap()
        .clone();
    assert_eq!(sell.reserved, dec!(0));
    // With no reservation behind it, it still fills and cancels.
    let id = sell.order_id;
    quoter.on_fill(&mut book, &id, dec!(0.5)).unwrap();
    quoter.on_cancel(&mut book, &id).unwrap();
    assert!(
        quoter
            .live_orders()
            .iter()
            .all(|quote| quote.side != Action::Sell)
    );
}

#[test]
fn test_a_reservation_rounds_up_to_the_cash_asset() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario(
        "third",
        dec!(1),
        Values(vec![(
            dec!(100),
            valuation(dec!(13.3333333333333), dec!(0.5), dec!(20)),
        )]),
    )];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let (_, ask) = bid_ask(&round.quotes, 0);
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    assert_eq!(sell.reserved, dec!(13.33333334) - ask);
}

#[test]
fn test_a_sale_that_unwinds_a_long_charges_only_what_it_adds() {
    let held_by = |position: Decimal| {
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_restored_state(vec![position], 0)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        let (_, ask) = bid_ask(&round.quotes, 0);
        let sell = round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .unwrap()
            .reserved;
        (sell, dec!(14) - ask)
    };
    assert_eq!(held_by(dec!(3)).0, dec!(0));
    let (part, unit) = held_by(dec!(0.4));
    assert_eq!(part, (dec!(0.6) * unit).round_dp(8));
    let (whole, unit) = held_by(dec!(0));
    assert_eq!(whole, unit);
}

#[test]
fn test_a_second_sale_cannot_unwind_what_a_live_sale_already_does() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let first = quoter.quote(&mut book, &market()).unwrap();
    let sell = |round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .unwrap()
            .reserved
    };
    assert_eq!(sell(&first), dec!(0));
    let second = quoter.quote(&mut book, &market()).unwrap();
    assert!(sell(&second) > dec!(0));
}

#[test]
fn test_a_buy_that_covers_a_short_only_counts_once_in_the_capital() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(-1)], 0)
        .unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (bid, ask) = bid_ask(&round.quotes, 0);
    // The short at its scenario loss of 4, the bid and the ask's capital held, less the bid.
    assert_eq!(round.capital.deployed, dec!(4));
    assert_eq!(round.capital.reserved, bid + dec!(14) - ask);
    assert_eq!(round.capital.committed, dec!(4) + dec!(14) - ask);
}

#[test]
fn test_a_second_buy_covers_only_what_the_first_leaves_short() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(-1)], 0)
        .unwrap();
    quoter.quote(&mut book, &market()).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (bid, _) = bid_ask(&round.quotes, 0);
    let held: Decimal = quoter
        .live_orders()
        .iter()
        .map(|quote| quote.reserved)
        .sum();
    assert_eq!(round.capital.reserved, held);
    assert_eq!(round.capital.committed, dec!(4) + held - bid);
}

#[test]
fn test_capital_counts_longs_at_value_shorts_at_scenario_loss_and_the_hedge_at_spot() {
    let mut book = ledger(dec!(100000));
    hold_spot(&mut book, dec!(-2));
    let round = quoter(vec![dec!(3), dec!(-2)])
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.capital.deployed, dec!(30) + dec!(8) + dec!(200));
}

#[test]
fn test_the_budget_scales_every_side_by_one_factor_and_logs_it() {
    let mut settings = config(two_contracts());
    settings.scenarios = vec![scenario("up", dec!(1), raised(dec!(20)))];
    let mut unbudgeted = ledger(dec!(100000));
    let free = OptionQuoter::new(settings.clone())
        .unwrap()
        .quote(&mut unbudgeted, &market())
        .unwrap();
    let asked: Decimal = free.quotes.iter().map(|quote| quote.reserved).sum();
    settings.capital_budget = asked / dec!(2);

    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings.clone())
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 4);
    for quote in &round.quotes {
        // Half of one, rounded down to a lot: the factor is a little above a half or at it.
        assert_eq!(quote.quantity, dec!(0.5), "{quote:?}");
        assert_eq!(
            kinds(&round.constraints, quote.contract, quote.side),
            vec![OptionConstraintKind::CapitalBudget {
                requested: dec!(1),
                allowed: dec!(0.5),
                budget: settings.capital_budget,
            }]
        );
    }
    assert!(round.capital.committed <= settings.capital_budget);
}

#[test]
fn test_adding_instruments_adds_no_capital() {
    let committed = |contracts: Vec<ContractTerms>| {
        let mut settings = config(contracts);
        settings.capital_budget = dec!(5);
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(settings)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        assert!(!round.quotes.is_empty());
        round.capital.committed
    };
    let one = committed(vec![contract("C100", dec!(100))]);
    let two = committed(two_contracts());
    assert!(one <= dec!(5));
    assert!(two <= dec!(5));
}

#[test]
fn test_a_spent_budget_withholds_new_risk_but_lets_a_position_unwind() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.capital_budget = dec!(1);
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(2)], 0)
        .unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
    assert_eq!(round.quotes[0].quantity, dec!(1));
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::CapitalBudget {
            requested: dec!(1),
            allowed: dec!(0),
            budget: dec!(1),
        }]
    );
}

#[test]
fn test_a_side_whose_scenario_gives_no_capital_is_not_cut_by_the_budget() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("down", dec!(0.5), values())];
    settings.capital_budget = dec!(1);
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    assert_eq!(sell.quantity, dec!(1));
    assert_eq!(kinds(&round.constraints, 0, Action::Sell), vec![]);
}

#[test]
fn test_an_empty_ledger_refuses_the_reservation_and_nothing_stands() {
    let mut book = ledger(dec!(0));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert!(round.quotes.is_empty());
    assert!(quoter.live_orders().is_empty());
    assert_eq!(reserved(&book), dec!(0));
    let refused = kinds(&round.constraints, 0, Action::Buy);
    assert!(matches!(
        refused[..],
        [OptionConstraintKind::ReservationRefused { available, .. }] if available == dec!(0)
    ));
}

#[test]
fn test_the_inventory_limit_counts_the_live_orders_of_the_side() {
    let mut terms = contract("C100", dec!(100));
    terms.inventory_limit = dec!(2);
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![terms])).unwrap();
    assert_eq!(quoter.quote(&mut book, &market()).unwrap().quotes.len(), 2);
    assert_eq!(quoter.quote(&mut book, &market()).unwrap().quotes.len(), 2);
    let third = quoter.quote(&mut book, &market()).unwrap();
    assert!(third.quotes.is_empty());
    assert_eq!(
        kinds(&third.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::InventoryLimit {
            resulting: dec!(3),
            limit: dec!(2),
        }]
    );
    assert_eq!(
        kinds(&third.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::InventoryLimit {
            resulting: dec!(-3),
            limit: dec!(2),
        }]
    );
}

fn loose() -> GreeksLimits {
    GreeksLimits {
        max_delta: dec!(1000),
        max_gamma: dec!(1000),
        max_vega: dec!(100000),
        max_theta: dec!(-1000),
        delta_hedge_threshold: dec!(1000),
    }
}

#[test]
fn test_the_delta_limit_withholds_what_raises_it_and_counts_the_hedge() {
    let limits = GreeksLimits {
        max_delta: dec!(1),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    hold_spot(&mut book, dec!(1));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_greek_limits(limits);
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::GreekLimit {
            greek: LimitedGreek::Delta,
            value: dec!(1.5),
            limit: dec!(1),
        }]
    );
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
}

#[test]
fn test_a_delta_exactly_at_the_limit_stands() {
    let limits = GreeksLimits {
        max_delta: dec!(0.5),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 2);
}

#[test]
fn test_each_greek_limit_is_its_own() {
    for (limits, greek, value) in [
        (
            GreeksLimits {
                max_gamma: dec!(0.01),
                ..loose()
            },
            LimitedGreek::Gamma,
            dec!(0.02),
        ),
        (
            GreeksLimits {
                max_vega: dec!(19),
                ..loose()
            },
            LimitedGreek::Vega,
            dec!(20),
        ),
    ] {
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_greek_limits(limits)
            .quote(&mut book, &market())
            .unwrap();
        assert!(round.quotes.is_empty());
        assert!(matches!(
            kinds(&round.constraints, 0, Action::Buy)[..],
            [OptionConstraintKind::GreekLimit { greek: g, value: v, .. }] if g == greek && v == value
        ));
    }
}

#[test]
fn test_the_theta_limit_reads_per_day_and_only_from_below() {
    let limits = GreeksLimits {
        max_theta: dec!(-0.05),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    // A long pays -36.5 a year, -0.1 a day: over the limit. A short earns it, and stands.
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::GreekLimit {
            greek: LimitedGreek::Theta,
            value: dec!(-0.1),
            limit: dec!(-0.05),
        }]
    );
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
}

#[test]
fn test_a_theta_exactly_at_the_limit_stands() {
    let limits = GreeksLimits {
        max_theta: dec!(-0.1),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 2);
}

#[test]
fn test_a_portfolio_beyond_a_limit_can_still_reduce() {
    let limits = GreeksLimits {
        max_delta: dec!(1),
        max_theta: dec!(-0.5),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(10)], 0)
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
}

#[test]
fn test_a_portfolio_beyond_the_theta_limit_can_still_reduce() {
    let limits = GreeksLimits {
        max_theta: dec!(-0.5),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(10)], 0)
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
}

#[test]
fn test_the_limits_count_a_live_order_filling_as_a_worst_case() {
    let limits = GreeksLimits {
        max_delta: dec!(1),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_greek_limits(limits);
    quoter.quote(&mut book, &market()).unwrap();
    quoter.quote(&mut book, &market()).unwrap();
    let third = quoter.quote(&mut book, &market()).unwrap();
    // Two live of each side: a third of either would reach 1.5 if all of its side fill.
    assert!(third.quotes.is_empty());
    assert_eq!(third.constraints.len(), 2);
}

#[test]
fn test_the_scenario_limit_withholds_what_could_lose_too_much() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_max_scenario_loss(dec!(3))
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    // A short loses the 4 by which the scenario raises the value; a long gains it.
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::ScenarioLoss {
            scenario: "up".to_string(),
            loss: dec!(4),
            limit: dec!(3),
        }]
    );
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Buy);
}

#[test]
fn test_a_loss_exactly_at_the_scenario_limit_stands() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_max_scenario_loss(dec!(4))
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 2);
}

#[test]
fn test_the_scenario_limit_counts_the_hedge_and_lets_a_side_reduce() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("up", dec!(1.1), values())];
    let limited = |hedge: Decimal, position: Decimal| {
        let mut book = ledger(dec!(100000));
        hold_spot(&mut book, hedge);
        OptionQuoter::new(settings.clone())
            .unwrap()
            .with_restored_state(vec![position], 0)
            .unwrap()
            .with_max_scenario_loss(dec!(10))
            .unwrap()
            .quote(&mut book, &market())
            .unwrap()
    };
    // At 110, a short of the spot of 3 loses 30, and a buy of the contract gains 5 at best.
    let round = limited(dec!(-3), dec!(0));
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Buy);
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::ScenarioLoss {
            scenario: "up".to_string(),
            loss: dec!(35),
            limit: dec!(10),
        }]
    );
    // A long of the spot gains in the scenario, so a short of the contract loses little.
    let round = limited(dec!(3), dec!(0));
    assert_eq!(round.quotes.len(), 2);
}

#[test]
fn test_a_delta_beyond_the_threshold_sizes_a_hedge_that_is_not_sent() {
    let limits = GreeksLimits {
        delta_hedge_threshold: dec!(1),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(3.33)], 0)
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.net_delta, dec!(1.665));
    let hedge = round.hedge.unwrap();
    // Rounded to the instrument's lot towards zero.
    assert_eq!(hedge.quantity, dec!(-1.66));
    assert_eq!(hedge.price, dec!(100));
    assert_eq!(hedge.symbol, "AAA-USD");
    assert_eq!(
        book.position(&spot_id())
            .map_or(dec!(0), |held| held.quantity),
        dec!(0)
    );
}

#[test]
fn test_no_hedge_at_the_threshold_or_without_limits_or_below_a_lot() {
    let run = |threshold: Option<Decimal>, position: Decimal| {
        let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_restored_state(vec![position], 0)
            .unwrap();
        if let Some(threshold) = threshold {
            quoter = quoter.with_greek_limits(GreeksLimits {
                delta_hedge_threshold: threshold,
                ..loose()
            });
        }
        let mut book = ledger(dec!(100000));
        quoter.quote(&mut book, &market()).unwrap().hedge.is_some()
    };
    assert!(run(Some(dec!(1)), dec!(2.02)));
    assert!(!run(Some(dec!(1)), dec!(2)));
    assert!(!run(None, dec!(40)));
    assert!(!run(Some(dec!(0)), dec!(0.01)));
}

#[test]
fn test_the_net_delta_nets_the_hedge_and_every_contract() {
    let mut book = ledger(dec!(100000));
    hold_spot(&mut book, dec!(-1));
    let round = quoter(vec![dec!(2), dec!(5)])
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.net_delta, dec!(1.5));
}

#[test]
fn test_a_buy_fill_pays_its_premium_and_moves_the_position_and_the_reservation() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (bid, ask) = bid_ask(&round.quotes, 0);
    let buy = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Buy)
        .unwrap();
    let fill = quoter
        .on_fill(&mut book, &buy.order_id, dec!(0.25))
        .unwrap();
    assert_eq!(fill.premium, -(bid * dec!(0.25)));
    assert_eq!(fill.position, dec!(0.25));
    assert_eq!(fill.shortfall, None);
    assert_eq!(quoter.positions(), &[dec!(0.25)]);
    assert_eq!(cash(&book), dec!(100000) - bid * dec!(0.25));
    assert_eq!(book.settled_pnl(&usd()), -(bid * dec!(0.25)));
    // The long it bought now covers a quarter of the live sale.
    let unit = dec!(14) - ask;
    assert_eq!(reserved(&book), bid * dec!(0.75) + unit * dec!(0.75));
    let live = quoter.live_orders();
    assert_eq!(live.len(), 2);
    assert_eq!(live[0].quantity, dec!(0.75));
    assert_eq!(live[0].reserved, bid * dec!(0.75));
    assert_eq!(live[1].reserved, unit * dec!(0.75));
    assert_eq!(book.settlement_journal()[0].instrument.as_str(), "C100");

    quoter
        .on_fill(&mut book, &buy.order_id, dec!(0.75))
        .unwrap();
    assert_eq!(quoter.live_orders().len(), 1);
    assert_eq!(reserved(&book), dec!(0));
    assert!(quoter.on_cancel(&mut book, &buy.order_id).is_err());
}

#[test]
fn test_a_sell_fill_receives_its_premium() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (_, ask) = bid_ask(&round.quotes, 0);
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    let fill = quoter.on_fill(&mut book, &sell.order_id, dec!(1)).unwrap();
    assert_eq!(fill.premium, ask);
    assert_eq!(fill.position, dec!(-1));
    assert_eq!(cash(&book), dec!(100000) + ask);
    assert_eq!(quoter.live_orders().len(), 1);
}

#[test]
fn test_a_partly_filled_sale_holds_what_the_position_now_leaves_it() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(0.4)], 0)
        .unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let (_, ask) = bid_ask(&round.quotes, 0);
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    let unit = dec!(14) - ask;
    assert_eq!(sell.reserved, (dec!(0.6) * unit).round_dp(8));
    let held = |quoter: &OptionQuoter| {
        quoter
            .live_orders()
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .map(|quote| quote.reserved)
    };

    // Still long after this one, so the 0.6 left is 0.3 of cover and 0.3 of new short.
    quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.1))
        .unwrap();
    assert_eq!(held(&quoter), Some((dec!(0.6) * unit).round_dp(8)));

    // Past zero: the 0.5 left is all new short, at the position's -0.2.
    let fill = quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.4))
        .unwrap();
    assert_eq!(fill.position, dec!(-0.1));
    assert_eq!(fill.unreserved, dec!(0));
    assert_eq!(held(&quoter), Some((dec!(0.5) * unit).round_dp(8)));
    let bid = bid_ask(&round.quotes, 0).0;
    assert_eq!(reserved(&book), bid + (dec!(0.5) * unit).round_dp(8));

    // A sale in parts holds its own remainder throughout: 0.25 left at a position of -0.35, so the
    // short and the sale still add up to the 0.6 of new short the whole sale could make.
    let fill = quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.25))
        .unwrap();
    assert_eq!(fill.position, dec!(-0.35));
    assert_eq!(held(&quoter), Some((dec!(0.25) * unit).round_dp(8)));

    quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.25))
        .unwrap();
    assert_eq!(held(&quoter), None);
    assert_eq!(reserved(&book), bid);
}

#[test]
fn test_filling_one_sale_can_take_the_cover_of_another() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let first = quoter.quote(&mut book, &market()).unwrap();
    let second = quoter.quote(&mut book, &market()).unwrap();
    let side = |round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .unwrap()
            .clone()
    };
    let (older, newer) = (side(&first), side(&second));
    assert_eq!(older.reserved, dec!(0));
    let unit = dec!(14) - older.price;
    assert!(newer.reserved > dec!(0));

    // The newer sale fills and the long is gone, so the older one is new short now.
    quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    let live = quoter.live_orders();
    let left = live
        .iter()
        .find(|quote| quote.order_id == older.order_id)
        .unwrap();
    assert_eq!(left.reserved, unit);
    assert!(
        book.reservation(&ReservationId::new(older.order_id.as_str()).unwrap())
            .is_some()
    );
}

#[test]
fn test_a_buy_fill_gives_a_live_sale_its_cover() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let id = |side| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == side)
            .unwrap()
            .order_id
            .clone()
    };
    quoter
        .on_fill(&mut book, &id(Action::Buy), dec!(0.5))
        .unwrap();
    let unit = dec!(14) - bid_ask(&round.quotes, 0).1;
    let sell = quoter
        .live_orders()
        .into_iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    assert_eq!(sell.reserved, (dec!(0.5) * unit).round_dp(8));
}

#[test]
fn test_cancelling_a_sale_passes_its_cover_to_the_next() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let first = quoter.quote(&mut book, &market()).unwrap();
    let second = quoter.quote(&mut book, &market()).unwrap();
    let sell = |round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .unwrap()
            .clone()
    };
    let (older, newer) = (sell(&first), sell(&second));
    assert!(newer.reserved > dec!(0));
    quoter.on_cancel(&mut book, &older.order_id).unwrap();
    let left = quoter
        .live_orders()
        .into_iter()
        .find(|quote| quote.order_id == newer.order_id)
        .unwrap();
    assert_eq!(left.reserved, dec!(0));
}

#[test]
fn test_a_cover_lost_with_too_little_free_cash_is_reported_and_the_fill_stands() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let first = quoter.quote(&mut book, &market()).unwrap();
    let second = quoter.quote(&mut book, &market()).unwrap();
    let sell = |round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .unwrap()
            .clone()
    };
    let (older, newer) = (sell(&first), sell(&second));
    let unit = dec!(14) - older.price;

    // Something else takes the cash, so after the fill frees what it frees, 1 is left over.
    let held = reserved(&book);
    let wanted = held - newer.reserved - newer.price + dec!(1);
    book.book_settlement(&DerivativeSettlement {
        id: ExecutionId::new("elsewhere").unwrap(),
        instrument: InstrumentId::new("OTHER").unwrap(),
        asset: usd(),
        realized_pnl: wanted - dec!(100000),
        fee: dec!(0),
    })
    .unwrap();

    let fill = quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    assert_eq!(fill.position, dec!(0));
    let left = quoter
        .live_orders()
        .into_iter()
        .find(|quote| quote.order_id == older.order_id)
        .unwrap();
    assert_eq!(left.reserved, dec!(1));
    assert_eq!(fill.unreserved, unit - dec!(1));
    assert_eq!(left.quantity, dec!(1));
    assert_eq!(quoter.positions(), &[dec!(0)]);
}

#[test]
fn test_a_fill_that_is_refused_changes_nothing() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    let id = round.quotes[0].order_id.clone();
    let before = (cash(&book), reserved(&book));
    for quantity in [dec!(0), dec!(-1), dec!(1.01)] {
        assert!(quoter.on_fill(&mut book, &id, quantity).is_err());
    }
    let unknown = OrderId::new("nope").unwrap();
    assert!(quoter.on_fill(&mut book, &unknown, dec!(1)).is_err());
    assert_eq!((cash(&book), reserved(&book)), before);
    assert_eq!(quoter.positions(), &[dec!(0)]);
    assert_eq!(quoter.live_orders()[0].quantity, dec!(1));
}

#[test]
fn test_a_restarted_quoter_on_old_rounds_cannot_book_a_fill_twice() {
    let mut book = ledger(dec!(100000));
    let mut first = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = first.quote(&mut book, &market()).unwrap();
    for quote in &round.quotes {
        first.on_fill(&mut book, &quote.order_id, dec!(1)).unwrap();
    }
    let after = (cash(&book), book.settled_pnl(&usd()));

    let mut second = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let again = second.quote(&mut book, &market()).unwrap();
    assert_eq!(again.quotes[0].order_id, round.quotes[0].order_id);
    let reserved_before = reserved(&book);
    assert!(
        second
            .on_fill(&mut book, &again.quotes[0].order_id, dec!(1))
            .is_err()
    );
    assert_eq!((cash(&book), book.settled_pnl(&usd())), after);
    assert_eq!(reserved(&book), reserved_before);
    assert_eq!(second.positions(), &[dec!(0)]);
}

#[test]
fn test_a_round_skips_ids_that_hold_a_reservation() {
    let mut book = ledger(dec!(100000));
    let mut first = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = first.quote(&mut book, &market()).unwrap();
    let mut second = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let next = second.quote(&mut book, &market()).unwrap();
    assert_eq!(next.quotes.len(), 2);
    assert_ne!(next.quotes[0].order_id, round.quotes[0].order_id);
    assert_eq!(second.next_round(), 2);
}

#[test]
fn test_a_cancel_releases_what_the_order_holds() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = quoter.quote(&mut book, &market()).unwrap();
    for quote in &round.quotes {
        quoter.on_cancel(&mut book, &quote.order_id).unwrap();
    }
    assert_eq!(reserved(&book), dec!(0));
    assert!(quoter.live_orders().is_empty());
    assert!(
        quoter
            .on_cancel(&mut book, &round.quotes[0].order_id)
            .is_err()
    );
    // Nothing is live, so the next round starts from a full budget.
    assert_eq!(
        quoter.quote(&mut book, &market()).unwrap().capital.deployed,
        dec!(0)
    );
}

#[test]
fn test_an_expired_contract_is_withheld_unless_it_has_exposure() {
    let mut at_expiry = market();
    at_expiry.now_ms = EXPIRY;
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .quote(&mut book, &at_expiry)
        .unwrap();
    assert!(round.quotes.is_empty());
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::Expired]
    );
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::Expired]
    );

    let mut held = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    assert!(held.quote(&mut book, &at_expiry).is_err());

    let mut live = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    live.quote(&mut book, &market()).unwrap();
    assert!(live.quote(&mut book, &at_expiry).is_err());
}

#[test]
fn test_a_contract_just_before_expiry_is_quoted_with_its_time_left() {
    let mut late = market();
    late.now_ms = EXPIRY - 1;
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .quote(&mut book, &late)
        .unwrap();
    assert_eq!(round.quotes.len(), 2);
}

/// Records the time to expiry and the spot it is asked for.
struct Recording(std::sync::Mutex<Vec<(Decimal, Decimal)>>);

impl OptionPricer for Recording {
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
        self.0.lock().unwrap().push((inputs.tau_years, inputs.spot));
        values().value(inputs)
    }
}

#[test]
fn test_the_pricer_gets_the_contract_and_the_time_left_in_years() {
    let recording = Arc::new(Recording(std::sync::Mutex::new(Vec::new())));
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.pricer = recording.clone();
    settings.scenarios = vec![RiskScenario {
        name: "down".to_string(),
        spot_factor: dec!(0.5),
        pricer: recording.clone(),
    }];
    let mut book = ledger(dec!(100000));
    OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let asked = recording.0.lock().unwrap().clone();
    let tau = dec!(30) / dec!(365);
    assert_eq!(asked, vec![(tau, dec!(100)), (tau, dec!(50))]);
}

#[test]
fn test_a_contract_the_pricer_refuses_is_withheld_unless_it_has_exposure() {
    let mut contracts = two_contracts();
    contracts.push(contract("C999", dec!(999)));
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(contracts.clone()))
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 4);
    assert!(matches!(
        kinds(&round.constraints, 2, Action::Buy)[..],
        [OptionConstraintKind::NotValued { .. }]
    ));
    assert!(matches!(
        kinds(&round.constraints, 2, Action::Sell)[..],
        [OptionConstraintKind::NotValued { .. }]
    ));
    let mut held = OptionQuoter::new(config(contracts))
        .unwrap()
        .with_restored_state(vec![dec!(0), dec!(0), dec!(1)], 0)
        .unwrap();
    assert!(held.quote(&mut book, &market()).is_err());
}

#[test]
fn test_a_scenario_the_pricer_refuses_is_treated_as_a_refusal() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("none", dec!(1), Values(vec![]))];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings.clone())
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert!(round.quotes.is_empty());
    let mut held = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    assert!(held.quote(&mut book, &market()).is_err());
}

#[test]
fn test_a_contract_with_no_sensitivity_is_withheld() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.pricer = Arc::new(Values(vec![(
        dec!(100),
        valuation(dec!(1), dec!(0), dec!(0)),
    )]));
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert!(round.quotes.is_empty());
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::NoSensitivity]
    );
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::NoSensitivity]
    );
}

#[test]
fn test_a_withheld_contract_leaves_the_others_their_own_assets() {
    let contracts = vec![contract("C999", dec!(999)), contract("C100", dec!(100))];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(contracts))
        .unwrap()
        .with_restored_state(vec![dec!(0), dec!(2)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let mut alone = ledger(dec!(100000));
    let single = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(2)], 0)
        .unwrap()
        .quote(&mut alone, &market())
        .unwrap();
    assert_eq!(bid_ask(&round.quotes, 1), bid_ask(&single.quotes, 0));
}

#[test]
fn test_a_price_below_the_grid_is_withheld_and_a_quote_through_the_value_is_reported() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.pricer = Arc::new(Values(vec![(
        dec!(100),
        valuation(dec!(0.01), dec!(0.5), dec!(20)),
    )]));
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert!(matches!(
        kinds(&round.constraints, 0, Action::Buy)[..],
        [OptionConstraintKind::NonPositivePrice { price }] if price <= dec!(0)
    ));

    // Long enough that the ask falls through the value.
    let mut through = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(50)], 0)
        .unwrap()
        .quote(&mut through, &market())
        .unwrap();
    let ask = round.quotes[0].price;
    assert_eq!(round.quotes[0].side, Action::Sell);
    assert!(ask <= dec!(10));
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::ThroughValue {
            price: ask,
            value: dec!(10),
        }]
    );

    // And short enough for the bid.
    let mut through = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(-50)], 0)
        .unwrap()
        .quote(&mut through, &market())
        .unwrap();
    let bid = round.quotes[0].price;
    assert_eq!(round.quotes[0].side, Action::Buy);
    assert!(bid >= dec!(10));
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::ThroughValue {
            price: bid,
            value: dec!(10),
        }]
    );
}

#[test]
fn test_a_position_beyond_the_inventory_limit_can_still_reduce() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(52)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
    assert_eq!(
        kinds(&round.constraints, 0, Action::Buy),
        vec![OptionConstraintKind::InventoryLimit {
            resulting: dec!(53),
            limit: dec!(50),
        }]
    );

    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(-52)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Buy);
}

#[test]
fn test_a_position_exactly_at_the_inventory_limit_only_quotes_back() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(50)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.quotes.len(), 1);
    assert_eq!(round.quotes[0].side, Action::Sell);
}

#[test]
fn test_a_quoter_refuses_what_could_not_work() {
    type Change = Box<dyn Fn(&mut OptionQuoterConfig)>;
    let bad: Vec<(&str, Change)> = vec![
        ("no contract", Box::new(|c| c.contracts.clear())),
        ("no scenario", Box::new(|c| c.scenarios.clear())),
        (
            "too many",
            Box::new(|c| {
                c.contracts = (0..64)
                    .map(|i| contract(&format!("C{i}"), dec!(100)))
                    .collect();
            }),
        ),
        ("budget", Box::new(|c| c.capital_budget = dec!(0))),
        ("aversion", Box::new(|c| c.risk_aversion = dec!(0))),
        (
            "spot vol",
            Box::new(|c| c.factors.spot_volatility = dec!(0)),
        ),
        (
            "vol of vol",
            Box::new(|c| c.factors.volatility_of_volatility = dec!(0)),
        ),
        (
            "correlation",
            Box::new(|c| c.factors.correlation = dec!(1.01)),
        ),
        (
            "negative correlation",
            Box::new(|c| c.factors.correlation = dec!(-1.01)),
        ),
        (
            "spot factor",
            Box::new(|c| c.scenarios[0].spot_factor = dec!(0)),
        ),
        (
            "underlying flow",
            Box::new(|c| c.underlying.intensity_scale = dec!(0)),
        ),
        (
            "underlying limit",
            Box::new(|c| c.underlying.inventory_limit = dec!(0.5)),
        ),
        (
            "repeat",
            Box::new(|c| c.contracts[1].name = "C100".to_string()),
        ),
        ("name", Box::new(|c| c.contracts[0].name = String::new())),
        (
            "named as the underlying",
            Box::new(|c| c.contracts[0].name = "AAA-USD".to_string()),
        ),
        ("strike", Box::new(|c| c.contracts[0].strike = dec!(0))),
        (
            "scale",
            Box::new(|c| c.contracts[0].intensity_scale = dec!(0)),
        ),
        (
            "decay",
            Box::new(|c| c.contracts[0].intensity_decay = dec!(0)),
        ),
        ("size", Box::new(|c| c.contracts[0].trade_size = dec!(0))),
        (
            "limit",
            Box::new(|c| c.contracts[0].inventory_limit = dec!(0.5)),
        ),
        ("lot", Box::new(|c| c.contracts[0].trade_size = dec!(0.001))),
    ];
    for (name, change) in bad {
        let mut settings = config(two_contracts());
        change(&mut settings);
        assert!(OptionQuoter::new(settings).is_err(), "{name}");
    }
    assert!(
        OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_max_scenario_loss(dec!(0))
            .is_err()
    );
    assert!(
        OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_restored_state(vec![], 0)
            .is_err()
    );
    assert!(OptionQuoter::new(config(vec![contract("C", dec!(100)); 1])).is_ok());
}

#[test]
fn test_a_round_refuses_a_market_or_ledger_it_cannot_use() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let mut bad = market();
    bad.spot = dec!(0);
    assert!(quoter.quote(&mut book, &bad).is_err());

    let mut empty = Ledger::default();
    assert!(quoter.quote(&mut empty, &market()).is_err());

    // Quoted in another asset than the cash.
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.cash_asset = base();
    assert!(
        OptionQuoter::new(settings)
            .unwrap()
            .quote(&mut book, &market())
            .is_err()
    );

    // A premium the cash asset could not hold, which no fill could then book.
    let mut terms = contract("C100", dec!(100));
    terms.price_scale = 7;
    assert!(
        OptionQuoter::new(config(vec![terms]))
            .unwrap()
            .quote(&mut book, &market())
            .is_err()
    );
    let mut fits = contract("C100", dec!(100));
    fits.price_scale = 6;
    assert!(
        OptionQuoter::new(config(vec![fits]))
            .unwrap()
            .quote(&mut book, &market())
            .is_ok()
    );
}

#[test]
fn test_on_the_validated_pricer_and_shocked_surfaces_a_book_of_many_contracts_quotes() {
    let flat_surface = FlatSurface::new(dec!(0.4)).unwrap();
    let pricer = |surface: Arc<dyn VolSurface>| {
        Arc::new(SurfacePricer::new(surface, Dynamics::StickyMoneyness).unwrap())
            as Arc<dyn OptionPricer>
    };
    let up = ShockedSurface::new(
        flat_surface,
        SurfaceShock::new("up", 0.1, 0.0, 0.0, 1.0).unwrap(),
    );
    let mut contracts = Vec::new();
    for (i, strike) in [
        dec!(80),
        dec!(90),
        dec!(100),
        dec!(110),
        dec!(120),
        dec!(130),
    ]
    .into_iter()
    .enumerate()
    {
        let mut terms = contract(&format!("C{i}"), strike);
        terms.price_scale = 4;
        terms.intensity_decay = dec!(5);
        contracts.push(terms);
    }
    let mut settings = config(contracts);
    settings.pricer = pricer(Arc::new(flat_surface));
    settings.scenarios = vec![RiskScenario {
        name: "up".to_string(),
        spot_factor: dec!(1.1),
        pricer: pricer(Arc::new(up)),
    }];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(
            vec![dec!(2), dec!(-1), dec!(0), dec!(1), dec!(-2), dec!(0)],
            0,
        )
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    // The cheapest contract's value is below its half spread, so it has no bid.
    assert_eq!(round.quotes.len(), 11, "{:?}", round.constraints);
    assert!(matches!(
        kinds(&round.constraints, 5, Action::Buy)[..],
        [OptionConstraintKind::NonPositivePrice { .. }]
    ));
    assert!(round.net_delta != dec!(0));
    for contract in 0..5 {
        let (bid, ask) = bid_ask(&round.quotes, contract);
        assert!(bid < ask);
    }
    // The higher spot and volatility cost a short more than its premium.
    let sale = round
        .quotes
        .iter()
        .find(|quote| quote.contract == 2 && quote.side == Action::Sell)
        .unwrap();
    assert!(sale.reserved.to_f64().unwrap() > 1.0);
    assert_eq!(round.capital.reserved, reserved(&book));
}

#[test]
fn test_a_short_with_every_scenario_below_its_value_deploys_nothing() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("down", dec!(0.5), values())];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(-3)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.capital.deployed, dec!(0));
}

#[test]
fn test_a_live_sale_does_not_count_as_covering_a_short() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(-2)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let (bid, ask) = bid_ask(&round.quotes, 0);
    assert_eq!(round.capital.deployed, dec!(8));
    assert_eq!(round.capital.reserved, bid + dec!(14) - ask);
    assert_eq!(round.capital.committed, dec!(8) + dec!(14) - ask);
}

#[test]
fn test_the_limits_count_the_sides_already_admitted_this_round() {
    let limits = GreeksLimits {
        max_delta: dec!(0.6),
        ..loose()
    };
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(two_contracts()))
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    // A buy of the first reaches 0.5, and with it a buy of the second would reach 0.8.
    assert_eq!(
        kinds(&round.constraints, 1, Action::Buy),
        vec![OptionConstraintKind::GreekLimit {
            greek: LimitedGreek::Delta,
            value: dec!(0.8),
            limit: dec!(0.6),
        }]
    );
    assert_eq!(round.quotes.len(), 2);
}

#[test]
fn test_a_quote_exactly_at_the_value_is_reported_as_through_it() {
    let (mut bid_at_value, mut ask_at_value) = (false, false);
    for hundredth in -3000..=3000 {
        let position = Decimal::new(hundredth, 2);
        let mut book = ledger(dec!(1000000));
        let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
            .unwrap()
            .with_restored_state(vec![position], 0)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        for quote in &round.quotes {
            let through = kinds(&round.constraints, 0, quote.side)
                .iter()
                .any(|kind| matches!(kind, OptionConstraintKind::ThroughValue { .. }));
            let expected = match quote.side {
                Action::Buy => {
                    bid_at_value |= quote.price == dec!(10);
                    quote.price >= dec!(10)
                }
                _ => {
                    ask_at_value |= quote.price == dec!(10);
                    quote.price <= dec!(10)
                }
            };
            assert_eq!(through, expected, "{position} {quote:?}");
        }
    }
    assert!(bid_at_value && ask_at_value);
}

#[test]
fn test_a_price_of_exactly_zero_is_withheld() {
    let mut zero_seen = false;
    for thousandth in 1..300 {
        let mut settings = config(vec![contract("C100", dec!(100))]);
        settings.pricer = Arc::new(Values(vec![(
            dec!(100),
            valuation(Decimal::new(thousandth, 3), dec!(0.5), dec!(20)),
        )]));
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(settings)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        assert!(round.quotes.iter().all(|quote| quote.price > dec!(0)));
        zero_seen |= kinds(&round.constraints, 0, Action::Buy)
            .contains(&OptionConstraintKind::NonPositivePrice { price: dec!(0) });
    }
    assert!(zero_seen);
}

#[test]
fn test_a_reservation_rounded_past_the_budget_is_released_and_logged() {
    let settings_with = |budget: Decimal| {
        let mut settings = config(vec![contract("C100", dec!(100))]);
        settings.capital_budget = budget;
        settings.scenarios = vec![scenario(
            "third",
            dec!(1),
            Values(vec![(
                dec!(100),
                valuation(dec!(13.3333333333333), dec!(0.5), dec!(20)),
            )]),
        )];
        settings
    };
    // Only the sale stands: the theta limit holds a long back.
    let limits = GreeksLimits {
        max_theta: dec!(-0.05),
        ..loose()
    };
    let mut wide = ledger(dec!(100000));
    let free = OptionQuoter::new(settings_with(dec!(1000)))
        .unwrap()
        .with_greek_limits(limits.clone())
        .quote(&mut wide, &market())
        .unwrap();
    assert_eq!(free.quotes.len(), 1);
    let unit = dec!(13.3333333333333) - free.quotes[0].price;

    // Exactly what the sale needs, which is finer than the cash asset, so it rounds past it.
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(settings_with(unit))
        .unwrap()
        .with_greek_limits(limits);
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert!(round.quotes.is_empty());
    assert!(quoter.live_orders().is_empty());
    assert_eq!(reserved(&book), dec!(0));
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::CapitalBudget {
            requested: dec!(1),
            allowed: dec!(0),
            budget: unit,
        }]
    );
    assert_eq!(round.capital.committed, dec!(0));
}

#[test]
fn test_a_quoter_told_where_the_old_one_stopped_takes_new_ids() {
    let mut book = ledger(dec!(100000));
    let mut first = OptionQuoter::new(config(vec![contract("C100", dec!(100))])).unwrap();
    let round = first.quote(&mut book, &market()).unwrap();
    for quote in &round.quotes {
        first.on_fill(&mut book, &quote.order_id, dec!(1)).unwrap();
    }
    assert_eq!(first.next_round(), 1);
    let mut restored = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(0)], first.next_round())
        .unwrap();
    let fresh = restored.quote(&mut book, &market()).unwrap();
    assert_ne!(fresh.quotes[0].order_id, round.quotes[0].order_id);
    restored
        .on_fill(&mut book, &fresh.quotes[0].order_id, dec!(1))
        .unwrap();
}

#[test]
fn test_a_sale_with_a_negative_delta_raises_the_delta_it_is_limited_on() {
    let limits = GreeksLimits {
        max_delta: dec!(0.4),
        ..loose()
    };
    let mut settings = config(vec![contract("P100", dec!(100))]);
    settings.pricer = Arc::new(Values(vec![(
        dec!(100),
        valuation(dec!(10), dec!(-0.5), dec!(20)),
    )]));
    settings.scenarios = vec![scenario("down", dec!(0.5), values())];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::GreekLimit {
            greek: LimitedGreek::Delta,
            value: dec!(0.5),
            limit: dec!(0.4),
        }]
    );
}

fn sale_of(
    round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound,
) -> market_maker_rs::multi_underlying::multi_asset::AdmittedOptionQuote {
    round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap()
        .clone()
}

fn live_sale(
    quoter: &OptionQuoter,
    id: &OrderId,
) -> market_maker_rs::multi_underlying::multi_asset::AdmittedOptionQuote {
    quoter
        .live_orders()
        .into_iter()
        .find(|quote| quote.order_id == *id)
        .unwrap()
}

/// Takes cash out of the ledger until `free` would be left once the fill adds `gained`.
fn leave_free(book: &mut Ledger, gained: Decimal, free: Decimal) {
    let total = cash(book);
    let target = reserved(book) - gained + free;
    book.book_settlement(&DerivativeSettlement {
        id: ExecutionId::new("elsewhere").unwrap(),
        instrument: InstrumentId::new("OTHER").unwrap(),
        asset: usd(),
        realized_pnl: target - total,
        fee: dec!(0),
    })
    .unwrap();
}

#[test]
fn test_a_long_covers_the_oldest_sales_in_turn() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(3)], 0)
        .unwrap();
    let rounds: Vec<_> = (0..4)
        .map(|_| quoter.quote(&mut book, &market()).unwrap())
        .collect();
    let sales: Vec<_> = rounds.iter().map(sale_of).collect();
    // The fourth is the one with no cover, and cancelling it changes none of the others.
    assert!(sales[3].reserved > dec!(0));
    quoter.on_cancel(&mut book, &sales[3].order_id).unwrap();
    for sale in &sales[..3] {
        assert_eq!(live_sale(&quoter, &sale.order_id).reserved, dec!(0));
    }
}

#[test]
fn test_a_long_does_not_cover_a_sale_beyond_it() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let rounds: Vec<_> = (0..3)
        .map(|_| quoter.quote(&mut book, &market()).unwrap())
        .collect();
    let sales: Vec<_> = rounds.iter().map(sale_of).collect();
    quoter.on_cancel(&mut book, &sales[2].order_id).unwrap();
    assert_eq!(live_sale(&quoter, &sales[0].order_id).reserved, dec!(0));
    assert_eq!(
        live_sale(&quoter, &sales[1].order_id).reserved,
        dec!(14) - sales[1].price
    );
}

#[test]
fn test_a_fill_on_one_contract_leaves_the_sales_of_another() {
    let mut book = ledger(dec!(100000));
    let mut quoter = quoter(vec![dec!(0), dec!(0)]);
    let round = quoter.quote(&mut book, &market()).unwrap();
    let on = |contract, side| {
        round
            .quotes
            .iter()
            .find(|quote| quote.contract == contract && quote.side == side)
            .unwrap()
            .clone()
    };
    let other = on(1, Action::Sell);
    // With the first contract's own sale gone, the long the buy brings has no sale of its own.
    quoter
        .on_cancel(&mut book, &on(0, Action::Sell).order_id)
        .unwrap();
    quoter
        .on_fill(&mut book, &on(0, Action::Buy).order_id, dec!(1))
        .unwrap();
    assert_eq!(live_sale(&quoter, &other.order_id).reserved, other.reserved);
}

#[test]
fn test_what_a_fill_shrinks_pays_for_what_it_raises() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let older = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let newer = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let unit = dec!(14) - older.price;
    // Half of the newer fills: half its capital frees, and half the older one's cover goes. Only
    // the premium is free otherwise, and it is taken here.
    leave_free(&mut book, newer.price * dec!(0.5), dec!(0));
    let fill = quoter
        .on_fill(&mut book, &newer.order_id, dec!(0.5))
        .unwrap();
    assert_eq!(fill.unreserved, dec!(0));
    assert_eq!(
        live_sale(&quoter, &older.order_id).reserved,
        unit * dec!(0.5)
    );
    assert_eq!(
        live_sale(&quoter, &newer.order_id).reserved,
        unit * dec!(0.5)
    );
}

#[test]
fn test_a_sale_that_rises_keeps_what_it_held_and_adds_what_is_free() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(0.5)], 0)
        .unwrap();
    let older = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let newer = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let unit = dec!(14) - older.price;
    assert_eq!(older.reserved, unit * dec!(0.5));
    // The newer fills in full, so what it frees and its premium are both free.
    leave_free(&mut book, newer.reserved + newer.price, dec!(0.1));
    let fill = quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    let left = live_sale(&quoter, &older.order_id);
    assert_eq!(left.reserved, unit * dec!(0.5) + dec!(0.1));
    assert_eq!(fill.unreserved, unit * dec!(0.5) - dec!(0.1));
}

#[test]
fn test_an_overdrawn_ledger_leaves_a_sale_what_it_held() {
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
    let older = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let newer = sale_of(&quoter.quote(&mut book, &market()).unwrap());
    let unit = dec!(14) - older.price;
    leave_free(&mut book, newer.reserved + newer.price, dec!(-5));
    let fill = quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    assert_eq!(live_sale(&quoter, &older.order_id).reserved, dec!(0));
    assert_eq!(fill.unreserved, unit);
}

#[test]
fn test_a_reduction_stops_at_the_limit_on_the_other_side() {
    let mut terms = contract("C100", dec!(100));
    terms.trade_size = dec!(30);
    let mut book = ledger(dec!(1000000));
    let mut quoter = OptionQuoter::new(config(vec![terms]))
        .unwrap()
        .with_restored_state(vec![dec!(60)], 0)
        .unwrap();
    let sale = |round: &market_maker_rs::multi_underlying::multi_asset::OptionQuoteRound| {
        round
            .quotes
            .iter()
            .find(|quote| quote.side == Action::Sell)
            .map(|quote| quote.quantity)
    };
    // 60 to 30, to 0, to -30: all whole.
    for _ in 0..3 {
        assert_eq!(
            sale(&quoter.quote(&mut book, &market()).unwrap()),
            Some(dec!(30))
        );
    }
    // Then only the 20 left to -50, and it says so.
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert_eq!(sale(&round), Some(dec!(20)));
    assert!(kinds(&round.constraints, 0, Action::Sell).contains(
        &OptionConstraintKind::InventoryCapped {
            requested: dec!(30),
            allowed: dec!(20),
            limit: dec!(50),
        }
    ));
    // And nothing further.
    let round = quoter.quote(&mut book, &market()).unwrap();
    assert_eq!(sale(&round), None);
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::InventoryLimit {
            resulting: dec!(-80),
            limit: dec!(50),
        }]
    );
    let total: Decimal = quoter
        .live_orders()
        .iter()
        .filter(|quote| quote.side == Action::Sell)
        .map(|quote| quote.quantity)
        .sum();
    assert_eq!(total, dec!(110), "from 60 to -50");
}

#[test]
fn test_a_buy_stops_at_the_limit_too() {
    let mut terms = contract("C100", dec!(100));
    terms.trade_size = dec!(30);
    let mut book = ledger(dec!(1000000));
    let round = OptionQuoter::new(config(vec![terms]))
        .unwrap()
        .with_restored_state(vec![dec!(45)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let buy = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Buy)
        .unwrap();
    assert_eq!(buy.quantity, dec!(5));
}

#[test]
fn test_a_cap_that_rounds_to_no_lot_withholds_the_side() {
    let mut terms = contract("C100", dec!(100));
    terms.quantity_scale = 1;
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![terms]))
        .unwrap()
        .with_restored_state(vec![dec!(49.96)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert!(matches!(
        kinds(&round.constraints, 0, Action::Buy)[..],
        [OptionConstraintKind::InventoryLimit { .. }]
    ));
}

/// Seeded values for books of many contracts, of every delta and vega the review tried.
struct Scattered(u64);

impl OptionPricer for Scattered {
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
        let strike = inputs.strike.to_u64().unwrap();
        let mix = |salt: u64| {
            let mut x = self
                .0
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(strike.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                .wrapping_add(salt);
            for _ in 0..3 {
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
            }
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        Ok(valuation(
            dec!(10),
            Decimal::from_f64(0.05 + 0.9 * mix(1)).unwrap(),
            Decimal::from_f64(1.0 + 29.0 * mix(2)).unwrap(),
        ))
    }
}

#[test]
fn test_a_book_of_scattered_sensitivities_quotes_at_any_supported_size_and_correlation() {
    for size in [16usize, 32, 63] {
        for correlation in [dec!(-1), dec!(-0.5), dec!(0), dec!(1)] {
            for seed in 0..10u64 {
                let contracts: Vec<ContractTerms> = (0..size)
                    .map(|i| contract(&format!("C{i}"), Decimal::from(50 + i as u64)))
                    .collect();
                let mut settings = config(contracts);
                settings.pricer = Arc::new(Scattered(seed));
                settings.scenarios = vec![RiskScenario {
                    name: "up".to_string(),
                    spot_factor: dec!(1.1),
                    pricer: Arc::new(Scattered(seed)),
                }];
                settings.factors.correlation = correlation;
                let mut book = ledger(dec!(100000));
                OptionQuoter::new(settings)
                    .unwrap()
                    .quote(&mut book, &market())
                    .unwrap_or_else(|error| {
                        panic!("{size} contracts, {correlation}, seed {seed}: {error}")
                    });
            }
        }
    }
}

#[test]
fn test_the_limits_read_what_the_cap_left_and_not_the_size_asked() {
    let mut terms = contract("C100", dec!(100));
    terms.trade_size = dec!(30);
    // 45 long at a delta of 0.5 is 22.5, and the 5 the cap leaves take it to 25.
    let limits = GreeksLimits {
        max_delta: dec!(25),
        ..loose()
    };
    let mut book = ledger(dec!(1000000));
    let round = OptionQuoter::new(config(vec![terms]))
        .unwrap()
        .with_restored_state(vec![dec!(45)], 0)
        .unwrap()
        .with_greek_limits(limits)
        .quote(&mut book, &market())
        .unwrap();
    let buy = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Buy)
        .unwrap();
    assert_eq!(buy.quantity, dec!(5));
}
