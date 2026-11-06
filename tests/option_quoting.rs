//! Joint option quotes on hand made values. Everything here is synthetic.

use std::sync::Arc;

use margin_rs::initial_margin::RuleId;
use margin_rs::instrument::{
    ExerciseStyle, OptionContract, OptionDelivery, OptionInstrument, OptionMargining, OptionRight,
};
use margin_rs::ledger::{DerivativeSettlement, Execution, Ledger};
use margin_rs::rules::{OptionExposure, OptionExposureView, ShortOptionIm, ShortOptionParameters};
use margin_rs::{
    AccountId, Asset, ExecutionId, InstrumentId, LinearTerms, OrderId, ReservationId,
    SpotInstrument, TransferId, VenueId,
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

// The capital of a short is the initial margin of margin-rs's `short-option-im/1` at the spot of
// 100: `q * max(0.15 * 100 - OTM, 0.10 * 100)`, `OTM = max(0, K - 100)` for a call. So one short
// of the call at 100 holds max(15 - 0, 10) = 15 and one of the call at 110 max(15 - 10, 10) = 10.

/// What margin-rs's `rule` requires of `contracts` of the contract `terms` at `spot`, asked of
/// margin-rs directly: the figure the quoter must hold.
fn rule_margin(
    rule: &ShortOptionIm,
    terms: &ContractTerms,
    contracts: Decimal,
    spot: Decimal,
) -> Decimal {
    let option = OptionInstrument::new(
        LinearTerms::new(
            InstrumentId::new(&terms.name).unwrap(),
            usd(),
            dec!(1),
            terms.price_scale,
            terms.quantity_scale,
        )
        .unwrap(),
        OptionContract {
            underlying: base(),
            strike: terms.strike,
            expiry_ms: terms.expiry_ms,
            right: match terms.style {
                OptionStyle::Call => OptionRight::Call,
                OptionStyle::Put => OptionRight::Put,
            },
            exercise: ExerciseStyle::European,
            delivery: OptionDelivery::Cash,
            margining: OptionMargining::Linear,
        },
    )
    .unwrap();
    rule.requirement(&OptionExposureView {
        account: AccountId::new("test").unwrap(),
        venue: VenueId::new("test").unwrap(),
        positions: vec![OptionExposure {
            option,
            contracts,
            underlying_price: spot,
        }],
    })
    .unwrap()
    .initial
}

fn default_rule() -> ShortOptionIm {
    ShortOptionIm::short_option_v1(usd(), 8).unwrap()
}

/// One call at a strike of 101.000000001, so that its margin is finer than the cash asset under a
/// rule at 18 places: max(0.15 * 100 - 1.000000001, 0.10 * 100) = 13.999999999.
fn fine_strike() -> (OptionQuoterConfig, ShortOptionIm) {
    let strike = dec!(101.000000001);
    let pricer = Values(vec![(strike, valuation(dec!(9.5), dec!(0.5), dec!(20)))]);
    let mut settings = config(vec![contract("C101", strike)]);
    settings.pricer = Arc::new(pricer.clone());
    settings.scenarios = vec![scenario("same", dec!(1), pricer)];
    (settings, ShortOptionIm::short_option_v1(usd(), 18).unwrap())
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
fn test_a_buy_reserves_its_premium_and_a_sell_its_initial_margin() {
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
    // One short of the call at 100: max(0.15 * 100 - 0, 0.10 * 100) = 15, whatever the ask.
    assert!(ask > dec!(0));
    assert_eq!(held(Action::Sell).reserved, dec!(15));
    assert_eq!(reserved(&book), bid + dec!(15));
    assert_eq!(
        book.reservation(&ReservationId::new(held(Action::Buy).order_id.as_str()).unwrap())
            .map(|(_, amount)| amount),
        Some(bid)
    );
    assert_eq!(round.capital.reserved, reserved(&book));
}

#[test]
fn test_a_sale_holds_the_same_margin_whatever_the_scenarios() {
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
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    // A 20% higher spot would raise the value by 10, but the margin reads only the round's spot:
    // max(0.15 * 100 - 0, 0.10 * 100) = 15, as under the single scenario of the fixture.
    assert_eq!(sell.reserved, dec!(15));
}

#[test]
fn test_a_sale_that_holds_nothing_still_fills_and_cancels() {
    let mut book = ledger(dec!(100000));
    // A long of 1 covers the whole sale of 1, so it holds nothing.
    let mut quoter = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap();
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
    let (settings, rule) = fine_strike();
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_short_margin(rule)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    // 13.999999999 at the rule's 18 places, up to the cash asset's 8: 14.00000000.
    assert_eq!(sell.reserved, dec!(14.00000000));
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
        assert!(ask > dec!(0));
        sell
    };
    // A long of 3 covers the whole sale of 1.
    assert_eq!(held_by(dec!(3)), dec!(0));
    // A long of 0.4 covers 0.4: the 0.6 left holds 0.6 * 15 = 9.
    assert_eq!(held_by(dec!(0.4)), dec!(9));
    // Flat, the whole sale holds 1 * 15 = 15.
    assert_eq!(held_by(dec!(0)), dec!(15));
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
    let (bid, _) = bid_ask(&round.quotes, 0);
    // The short of 1 at its margin of 1 * max(15 - 0, 10) = 15, the bid and the sale's 15 held,
    // less the bid that covers the short: 15 + bid + 15 - bid = 30.
    assert_eq!(round.capital.deployed, dec!(15));
    assert_eq!(round.capital.reserved, bid + dec!(15));
    assert_eq!(round.capital.committed, dec!(30));
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
    // The short of 1 at its margin of 15.
    assert_eq!(round.capital.committed, dec!(15) + held - bid);
}

#[test]
fn test_capital_counts_longs_at_value_shorts_at_initial_margin_and_the_hedge_at_spot() {
    let mut book = ledger(dec!(100000));
    hold_spot(&mut book, dec!(-2));
    let round = quoter(vec![dec!(3), dec!(-2)])
        .quote(&mut book, &market())
        .unwrap();
    // 3 long of the call at 100 at its value of 10; 2 short of the call at 110 at
    // 2 * max(0.15 * 100 - 10, 0.10 * 100) = 2 * 10 = 20; 2 short of the spot at 100.
    assert_eq!(round.capital.deployed, dec!(30) + dec!(20) + dec!(200));
}

#[test]
fn test_the_budget_scales_every_side_by_one_factor_and_logs_it() {
    let mut settings = config(two_contracts());
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
fn test_the_part_of_a_sale_that_unwinds_a_long_is_not_cut_by_the_budget() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.capital_budget = dec!(1);
    let mut book = ledger(dec!(100000));
    // The long of 0.4 at its value of 10 deploys 4, past the budget of 1: nothing is left for
    // new risk, but the 0.4 of the sale that unwinds the long commits nothing.
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(0.4)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    assert_eq!(sell.quantity, dec!(0.4));
    assert_eq!(sell.reserved, dec!(0));
    assert_eq!(
        kinds(&round.constraints, 0, Action::Sell),
        vec![OptionConstraintKind::CapitalBudget {
            requested: dec!(1),
            allowed: dec!(0.4),
            budget: dec!(1),
        }]
    );
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
    let (bid, _) = bid_ask(&round.quotes, 0);
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
    // The long it bought now covers a quarter of the live sale: 0.75 * 15 = 11.25 is left.
    assert_eq!(reserved(&book), bid * dec!(0.75) + dec!(11.25));
    let live = quoter.live_orders();
    assert_eq!(live.len(), 2);
    assert_eq!(live[0].quantity, dec!(0.75));
    assert_eq!(live[0].reserved, bid * dec!(0.75));
    assert_eq!(live[1].reserved, dec!(11.25));
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
    let sell = round
        .quotes
        .iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    // 15 per contract of new short: the 0.6 beyond the long of 0.4 holds 0.6 * 15 = 9.
    assert_eq!(sell.reserved, dec!(9));
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
    assert_eq!(held(&quoter), Some(dec!(9)));

    // Past zero: the 0.5 left is all new short, at the position's -0.2.
    let fill = quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.4))
        .unwrap();
    assert_eq!(fill.position, dec!(-0.1));
    assert_eq!(fill.unreserved, dec!(0));
    // 0.5 * 15 = 7.5.
    assert_eq!(held(&quoter), Some(dec!(7.5)));
    let bid = bid_ask(&round.quotes, 0).0;
    assert_eq!(reserved(&book), bid + dec!(7.5));

    // A sale in parts holds its own remainder throughout: 0.25 left at a position of -0.35, so the
    // short and the sale still add up to the 0.6 of new short the whole sale could make.
    let fill = quoter
        .on_fill(&mut book, &sell.order_id, dec!(0.25))
        .unwrap();
    assert_eq!(fill.position, dec!(-0.35));
    // 0.25 * 15 = 3.75.
    assert_eq!(held(&quoter), Some(dec!(3.75)));

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
    assert!(newer.reserved > dec!(0));

    // The newer sale fills and the long is gone, so the older one is new short now.
    quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    let live = quoter.live_orders();
    let left = live
        .iter()
        .find(|quote| quote.order_id == older.order_id)
        .unwrap();
    // One contract of new short: 15.
    assert_eq!(left.reserved, dec!(15));
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
    let sell = quoter
        .live_orders()
        .into_iter()
        .find(|quote| quote.side == Action::Sell)
        .unwrap();
    // The long of 0.5 covers half: 0.5 * 15 = 7.5.
    assert_eq!(sell.reserved, dec!(7.5));
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
    // The older sale is new short of 1 now, 15 to hold, and 1 is free: 15 - 1 = 14 unreserved.
    assert_eq!(fill.unreserved, dec!(14));
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
    // The call at 100, flat: max(0.15 * 100 - 0, 0.10 * 100) = 15, whatever the surface.
    let sale = round
        .quotes
        .iter()
        .find(|quote| quote.contract == 2 && quote.side == Action::Sell)
        .unwrap();
    assert_eq!(sale.reserved, dec!(15));
    assert_eq!(round.capital.reserved, reserved(&book));
}

#[test]
fn test_a_short_deploys_its_margin_even_with_every_scenario_below_its_value() {
    let mut settings = config(vec![contract("C100", dec!(100))]);
    settings.scenarios = vec![scenario("down", dec!(0.5), values())];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(-3)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    // The scenario only lowers the value, but the margin reads the spot: 3 * 15 = 45.
    assert_eq!(round.capital.deployed, dec!(45));
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
    let (bid, _) = bid_ask(&round.quotes, 0);
    // The short of 2 at 2 * 15 = 30, the sale's 15 held in full, and the bid less the bid that
    // covers the short: 30 + bid + 15 - bid = 45.
    assert_eq!(round.capital.deployed, dec!(30));
    assert_eq!(round.capital.reserved, bid + dec!(15));
    assert_eq!(round.capital.committed, dec!(45));
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
    // The sale's margin is 13.999999999 at the rule's 18 places (see `fine_strike`), finer than
    // the cash asset, so its reservation rounds up to 14.00000000.
    let unit = dec!(13.999999999);
    let (mut settings, rule) = fine_strike();
    settings.capital_budget = unit;
    // Only the sale stands: the theta limit holds a long back.
    let limits = GreeksLimits {
        max_theta: dec!(-0.05),
        ..loose()
    };

    // Exactly what the sale needs, which is finer than the cash asset, so it rounds past it.
    let mut book = ledger(dec!(100000));
    let mut quoter = OptionQuoter::new(settings)
        .unwrap()
        .with_short_margin(rule)
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
    // One contract of new short: 15.
    assert_eq!(live_sale(&quoter, &sales[1].order_id).reserved, dec!(15));
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
    // Half of the newer fills: half its capital frees, and half the older one's cover goes. Only
    // the premium is free otherwise, and it is taken here.
    leave_free(&mut book, newer.price * dec!(0.5), dec!(0));
    let fill = quoter
        .on_fill(&mut book, &newer.order_id, dec!(0.5))
        .unwrap();
    assert_eq!(fill.unreserved, dec!(0));
    // Each now has 0.5 of new short: 0.5 * 15 = 7.5.
    assert_eq!(live_sale(&quoter, &older.order_id).reserved, dec!(7.5));
    assert_eq!(live_sale(&quoter, &newer.order_id).reserved, dec!(7.5));
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
    // The long of 0.5 covers half of the older: 0.5 * 15 = 7.5.
    assert_eq!(older.reserved, dec!(7.5));
    // The newer fills in full, so what it frees and its premium are both free.
    leave_free(&mut book, newer.reserved + newer.price, dec!(0.1));
    let fill = quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    let left = live_sale(&quoter, &older.order_id);
    // It should hold 15, keeps its 7.5 and adds the 0.1 free: 7.6, and 15 - 7.6 = 7.4 is unreserved.
    assert_eq!(left.reserved, dec!(7.6));
    assert_eq!(fill.unreserved, dec!(7.4));
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
    leave_free(&mut book, newer.reserved + newer.price, dec!(-5));
    let fill = quoter.on_fill(&mut book, &newer.order_id, dec!(1)).unwrap();
    assert_eq!(live_sale(&quoter, &older.order_id).reserved, dec!(0));
    // The older sale should hold one contract of new short, 15, and none of it is free.
    assert_eq!(fill.unreserved, dec!(15));
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

/// A single contract of `style` at `strike`, with a value the pricer and the scenario both give.
fn single(style: OptionStyle, strike: Decimal) -> (OptionQuoterConfig, ContractTerms) {
    let delta = match style {
        OptionStyle::Call => dec!(0.5),
        OptionStyle::Put => dec!(-0.4),
    };
    let pricer = Values(vec![(strike, valuation(dec!(6), delta, dec!(20)))]);
    let mut terms = contract("OPT", strike);
    terms.style = style;
    let mut settings = config(vec![terms.clone()]);
    settings.pricer = Arc::new(pricer.clone());
    settings.scenarios = vec![scenario("same", dec!(1), pricer)];
    (settings, terms)
}

#[test]
fn test_a_sale_holds_the_rules_margin_for_one_contract_in_and_out_of_the_money() {
    // At a spot of 100, a = 15 % and b = 10 %: max(15 - OTM, 10).
    let cases = [
        // Call in the money, OTM = max(0, 90 - 100) = 0: max(15, 10) = 15.
        (OptionStyle::Call, dec!(90), dec!(15)),
        // Call a little out, OTM = 104 - 100 = 4: max(11, 10) = 11.
        (OptionStyle::Call, dec!(104), dec!(11)),
        // Call far out, OTM = 110 - 100 = 10: max(5, 10) = 10, the floor.
        (OptionStyle::Call, dec!(110), dec!(10)),
        // Put in the money, OTM = max(0, 100 - 110) = 0: max(15, 10) = 15.
        (OptionStyle::Put, dec!(110), dec!(15)),
        // Put a little out, OTM = 100 - 97 = 3: max(12, 10) = 12.
        (OptionStyle::Put, dec!(97), dec!(12)),
        // Put far out, OTM = 100 - 90 = 10: max(5, 10) = 10, the floor.
        (OptionStyle::Put, dec!(90), dec!(10)),
    ];
    for (style, strike, expected) in cases {
        let (settings, terms) = single(style, strike);
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(settings)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        let sale = sale_of(&round);
        assert_eq!(sale.quantity, dec!(1), "{style:?} {strike}");
        assert_eq!(sale.reserved, expected, "{style:?} {strike}");
        assert_eq!(
            sale.reserved,
            rule_margin(&default_rule(), &terms, dec!(-1), dec!(100)),
            "{style:?} {strike}"
        );
    }
}

#[test]
fn test_a_held_short_deploys_the_rules_margin() {
    for (style, strike, position, expected) in [
        // 2.5 short of the put at 97: 2.5 * max(15 - 3, 10) = 30.
        (OptionStyle::Put, dec!(97), dec!(-2.5), dec!(30)),
        // 0.5 short of the call at 110: 0.5 * max(15 - 10, 10) = 5.
        (OptionStyle::Call, dec!(110), dec!(-0.5), dec!(5)),
        // 3 short of the call at 90: 3 * max(15 - 0, 10) = 45.
        (OptionStyle::Call, dec!(90), dec!(-3), dec!(45)),
    ] {
        let (settings, terms) = single(style, strike);
        let mut book = ledger(dec!(100000));
        let round = OptionQuoter::new(settings)
            .unwrap()
            .with_restored_state(vec![position], 0)
            .unwrap()
            .quote(&mut book, &market())
            .unwrap();
        assert_eq!(round.capital.deployed, expected, "{style:?} {strike}");
        assert_eq!(
            round.capital.deployed,
            rule_margin(&default_rule(), &terms, position, dec!(100)),
            "{style:?} {strike}"
        );
    }
    // Two shorts, margined each alone and summed: 30 + 5 = 35 at a spot of 100.
    let put = {
        let mut terms = contract("P97", dec!(97));
        terms.style = OptionStyle::Put;
        terms
    };
    let mut settings = config(vec![put, contract("C110", dec!(110))]);
    let pricer = Values(vec![
        (dec!(97), valuation(dec!(6), dec!(-0.4), dec!(20))),
        (dec!(110), valuation(dec!(5), dec!(0.3), dec!(15))),
    ]);
    settings.pricer = Arc::new(pricer.clone());
    settings.scenarios = vec![scenario("same", dec!(1), pricer)];
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(settings)
        .unwrap()
        .with_restored_state(vec![dec!(-2.5), dec!(-0.5)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.capital.deployed, dec!(35));
}

#[test]
fn test_a_custom_rule_changes_the_capital_accordingly() {
    // a = 20 %, b = 12 %, m = 10 %, at a spot of 100: the call at 100 max(20 - 0, 12) = 20 and
    // the call at 110 max(20 - 10, 12) = 12, against 15 and 10 under `short-option-im/1`.
    let rule = ShortOptionIm::new(
        RuleId::new("desk-short-option", 1),
        usd(),
        8,
        ShortOptionParameters {
            initial_rate: dec!(0.20),
            floor_rate: dec!(0.12),
            maintenance_rate: dec!(0.10),
        },
    )
    .unwrap();
    let run = |rule: Option<ShortOptionIm>| {
        let mut quoter = quoter(vec![dec!(0), dec!(-1)]);
        if let Some(rule) = rule {
            quoter = quoter.with_short_margin(rule).unwrap();
        }
        let mut book = ledger(dec!(100000));
        let round = quoter.quote(&mut book, &market()).unwrap();
        let sale = |contract| {
            round
                .quotes
                .iter()
                .find(|quote| quote.contract == contract && quote.side == Action::Sell)
                .unwrap()
                .reserved
        };
        (sale(0), sale(1), round.capital.deployed)
    };
    // Sales of 1 of each, and the short of 1 of the call at 110.
    assert_eq!(run(None), (dec!(15), dec!(10), dec!(10)));
    assert_eq!(run(Some(rule.clone())), (dec!(20), dec!(12), dec!(12)));
    let terms = contract("C100", dec!(100));
    assert_eq!(rule_margin(&rule, &terms, dec!(-1), dec!(100)), dec!(20));
}

#[test]
fn test_a_rule_in_another_asset_than_the_cash_is_refused() {
    let quoter = OptionQuoter::new(config(two_contracts())).unwrap();
    let in_base = ShortOptionIm::short_option_v1(base(), 8).unwrap();
    assert!(matches!(
        quoter.with_short_margin(in_base),
        Err(MMError::InvalidConfiguration(_))
    ));
}

#[test]
fn test_a_sale_that_only_unwinds_a_long_holds_no_margin() {
    let mut book = ledger(dec!(100000));
    let round = OptionQuoter::new(config(vec![contract("C100", dec!(100))]))
        .unwrap()
        .with_restored_state(vec![dec!(1)], 0)
        .unwrap()
        .quote(&mut book, &market())
        .unwrap();
    let (bid, _) = bid_ask(&round.quotes, 0);
    let sale = sale_of(&round);
    assert_eq!(sale.quantity, dec!(1));
    assert_eq!(sale.reserved, dec!(0));
    // Only the bid is held, and the long counts at its value of 10.
    assert_eq!(reserved(&book), bid);
    assert_eq!(round.capital.deployed, dec!(10));
    assert_eq!(round.capital.committed, dec!(10) + bid);
}

/// A book of `size` contracts valued by `Scattered(seed)`, at the factor correlation given.
fn scattered_quoter(size: usize, seed: u64, correlation: Decimal) -> OptionQuoter {
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
    OptionQuoter::new(settings).unwrap()
}

#[test]
fn test_a_derived_correlation_rounded_past_the_floor_is_clipped_and_reported() {
    // At a factor correlation of -1 some contracts' delta and vega nearly offset, and the
    // rounding of this book, positive semidefinite in exact arithmetic, leaves an eigenvalue of
    // about -1.94e-11: past the floor of -1.7e-11 for 17 assets, so a supplied correlation like
    // it is refused, and well within what that cancellation can reach.
    let mut book = ledger(dec!(100000));
    let round = scattered_quoter(16, 19317, dec!(-1))
        .quote(&mut book, &market())
        .unwrap();
    assert!(!round.quotes.is_empty());
    let clipped = round.clipped_eigenvalue.unwrap();
    assert_eq!(clipped.floor, dec!(-0.000000000017));
    assert!(clipped.eigenvalue < clipped.floor, "{clipped:?}");
    assert!(clipped.eigenvalue > dec!(-0.00000000002), "{clipped:?}");
    assert!(clipped.band < clipped.eigenvalue, "{clipped:?}");
}

#[test]
fn test_a_correlation_with_no_eigenvalue_past_the_floor_reports_no_clip() {
    let mut book = ledger(dec!(100000));
    let round = quoter(vec![dec!(2), dec!(-3)])
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.clipped_eigenvalue, None);
    // The same contracts as the clipped book, at a correlation that cancels nothing.
    let mut book = ledger(dec!(100000));
    let round = scattered_quoter(16, 19317, dec!(-0.5))
        .quote(&mut book, &market())
        .unwrap();
    assert_eq!(round.clipped_eigenvalue, None);
}
