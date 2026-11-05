//! Every joint quote through a `margin-rs` gate, the composition of the stages,
//! the capital budget, and a comparison of the diagonal and correlated views.
//!
//! The evidence is synthetic: prices and fills are simulated from the model's
//! own assumptions (correlated Brownian motion, Poisson fills at the assumed
//! exponential intensity). It shows how the quoting rules behave under those
//! assumptions, and says nothing about any real market.

#![cfg(all(feature = "multi-underlying", feature = "margin"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use margin_rs::ledger::{Execution, Ledger, LedgerConfig};
use margin_rs::risk::{FillReport, HaltReason, InstrumentLimits, RiskGate, RiskReason};
use margin_rs::{Asset, ExecutionId, InstrumentId, SpotInstrument, TransferId};
use market_maker_rs::execution::Action;
use market_maker_rs::multi_underlying::multi_asset::{
    ActiveConstraint, AssetQuotingParams, ConstraintKind, MultiAssetModel, MultiAssetQuoter,
    QuoteRound, QuotingObjective, RiskView,
};
use market_maker_rs::types::error::MMError;
use market_maker_rs::{Decimal, dec};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const BASES: [&str; 4] = ["AAA", "BBB", "CCC", "DDD"];
const QUOTE: &str = "USD";
/// A budget no scenario below reaches, for the tests about something else.
const AMPLE: Decimal = dec!(100000);

fn asset(code: &str) -> Asset {
    Asset::new(code).unwrap()
}

fn instrument_id(index: usize) -> InstrumentId {
    InstrumentId::new(format!("{}-{QUOTE}", BASES[index])).unwrap()
}

fn ids(count: usize) -> Vec<InstrumentId> {
    (0..count).map(instrument_id).collect()
}

/// A gate over `count` spot instruments at four decimals of price and quantity.
fn gate_with(count: usize, quote: Decimal, base: Decimal, quote_scale: u32) -> RiskGate {
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(asset(QUOTE), quote_scale).unwrap();
    for (index, code) in BASES.iter().take(count).enumerate() {
        ledger.register_asset(asset(code), 8).unwrap();
        ledger
            .register_instrument(
                SpotInstrument::new(instrument_id(index), asset(code), asset(QUOTE), 4, 4).unwrap(),
            )
            .unwrap();
    }
    let mut gate = RiskGate::new(ledger);
    gate.deposit(TransferId::new("q").unwrap(), asset(QUOTE), quote)
        .unwrap();
    if base > Decimal::ZERO {
        for code in BASES.iter().take(count) {
            gate.deposit(
                TransferId::new(format!("b{code}")).unwrap(),
                asset(code),
                base,
            )
            .unwrap();
        }
    }
    gate
}

fn gate(count: usize) -> RiskGate {
    gate_with(count, dec!(1000000), dec!(1000), 8)
}

fn params(sigma: Decimal, limit: Decimal) -> AssetQuotingParams {
    AssetQuotingParams::new(sigma, dec!(10), dec!(20), dec!(1), limit).unwrap()
}

fn uniform_correlation(count: usize, rho: Decimal) -> Vec<Vec<Decimal>> {
    (0..count)
        .map(|i| {
            (0..count)
                .map(|j| if i == j { dec!(1) } else { rho })
                .collect()
        })
        .collect()
}

fn model(count: usize, rho: Decimal, view: RiskView, limit: Decimal) -> MultiAssetModel {
    let assets = (0..count).map(|_| params(dec!(1), limit)).collect();
    MultiAssetModel::new(
        assets,
        &uniform_correlation(count, rho),
        dec!(0.08),
        QuotingObjective::RiskAdjustedExpectation,
        view,
    )
    .unwrap()
}

fn quoter(count: usize, rho: Decimal, budget: Decimal) -> MultiAssetQuoter {
    MultiAssetQuoter::new(
        model(count, rho, RiskView::Correlated, dec!(100)),
        ids(count),
        budget,
    )
    .unwrap()
}

fn marks(count: usize) -> Vec<Decimal> {
    vec![dec!(100); count]
}

/// Books a fill of an imaginary order through the ledger, to set an inventory.
fn hold(gate: &mut RiskGate, index: usize, quantity: Decimal) {
    let (side, amount) = if quantity >= Decimal::ZERO {
        (Action::Buy, quantity)
    } else {
        (Action::Sell, -quantity)
    };
    let id = format!("seed-{index}-{quantity}");
    let order = margin_rs::OrderId::new(&id).unwrap();
    let request = margin_rs::risk::OrderRequest::new(
        order.clone(),
        instrument_id(index),
        side,
        dec!(100),
        amount,
    );
    assert!(!gate.submit(&request, 0).is_rejected());
    gate.on_venue_accepted(&order).unwrap();
    let execution = Execution::new(
        ExecutionId::new(&id).unwrap(),
        instrument_id(index),
        side,
        dec!(100),
        amount,
    );
    gate.on_fill(&order, &FillReport::new(execution, amount))
        .unwrap();
}

fn find(round: &QuoteRound, asset: usize, side: Action) -> Vec<&ActiveConstraint> {
    round
        .constraints
        .iter()
        .filter(|c| c.asset == asset && c.side == side)
        .collect()
}

fn quote_of(round: &QuoteRound, asset: usize, side: Action) -> Option<(Decimal, Decimal)> {
    round
        .quotes
        .iter()
        .find(|q| q.asset == asset && q.side == side)
        .map(|q| (q.price, q.quantity))
}

fn reserved_in_ledger(gate: &RiskGate, count: usize, mark: Decimal) -> Decimal {
    let ledger = gate.ledger();
    let mut total = ledger.balance(&asset(QUOTE)).unwrap().reserved;
    for code in BASES.iter().take(count) {
        total += ledger.balance(&asset(code)).unwrap().reserved * mark;
    }
    total
}

#[test]
fn test_quote_every_admitted_quote_is_an_order_the_gate_holds() {
    let mut gate = gate(2);
    let mut quoter = quoter(2, dec!(0.5), AMPLE);
    let round = quoter.quote(&mut gate, &marks(2), 7).unwrap();
    assert_eq!(round.quotes.len(), 4);
    assert!(round.constraints.is_empty());
    let mut expected = Decimal::ZERO;
    for quote in &round.quotes {
        let order = gate.order(&quote.order_id).expect("the gate has the order");
        assert_eq!(order.price, quote.price);
        assert_eq!(order.quantity, quote.quantity);
        assert_eq!(order.side, quote.side);
        assert_eq!(order.instrument, instrument_id(quote.asset));
        expected += match quote.side {
            Action::Buy => quote.price * quote.quantity,
            _ => dec!(100) * quote.quantity,
        };
    }
    assert_eq!(reserved_in_ledger(&gate, 2, dec!(100)), expected);
    assert_eq!(round.capital.reserved, expected);
}

#[test]
fn test_quote_rejected_by_a_halted_gate_produces_no_quote_and_logs_each_side() {
    let mut gate = gate(2);
    gate.trip(HaltReason::Manual);
    let mut quoter = quoter(2, dec!(0.5), AMPLE);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert!(round.quotes.is_empty());
    assert_eq!(round.constraints.len(), 4);
    for constraint in &round.constraints {
        match &constraint.kind {
            ConstraintKind::MarginRejected { reasons } => {
                assert!(matches!(reasons[0], RiskReason::Halted { .. }));
            }
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(round.capital.reserved, Decimal::ZERO);
}

#[test]
fn test_quote_reduced_by_the_gate_stands_smaller_and_logs_the_cause() {
    let mut gate = gate(1);
    gate.set_limits(
        &instrument_id(0),
        InstrumentLimits {
            max_order_quantity: Some(dec!(0.25)),
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(quote_of(&round, 0, Action::Buy).unwrap().1, dec!(0.25));
    assert_eq!(quote_of(&round, 0, Action::Sell).unwrap().1, dec!(0.25));
    assert_eq!(round.constraints.len(), 2);
    match &round.constraints[0].kind {
        ConstraintKind::MarginReduced {
            requested,
            approved,
            reasons,
        } => {
            assert_eq!(*requested, dec!(1));
            assert_eq!(*approved, dec!(0.25));
            assert!(matches!(reasons[0], RiskReason::MaxOrderQuantity { .. }));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_quote_without_base_to_sell_withholds_only_the_ask() {
    let mut gate = gate_with(1, dec!(1000000), Decimal::ZERO, 8);
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_some());
    assert!(quote_of(&round, 0, Action::Sell).is_none());
    assert!(matches!(
        &find(&round, 0, Action::Sell)[0].kind,
        ConstraintKind::MarginRejected { .. }
    ));
}

#[test]
fn test_quote_inventory_comes_from_the_ledger_and_skews_the_prices() {
    let mut gate = gate(2);
    hold(&mut gate, 0, dec!(4));
    hold(&mut gate, 1, dec!(-3));
    let mut quoter = quoter(2, dec!(0.5), AMPLE);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    let inventory = [dec!(4), dec!(-3)];
    for asset in 0..2 {
        let offsets = quoter.model().offsets(asset, &inventory).unwrap();
        let bid = (dec!(100) - offsets.bid)
            .round_dp_with_strategy(4, rust_decimal::RoundingStrategy::ToNegativeInfinity);
        let ask = (dec!(100) + offsets.ask)
            .round_dp_with_strategy(4, rust_decimal::RoundingStrategy::ToPositiveInfinity);
        assert_eq!(quote_of(&round, asset, Action::Buy).unwrap().0, bid);
        assert_eq!(quote_of(&round, asset, Action::Sell).unwrap().0, ask);
    }
    assert_ne!(
        quote_of(&round, 0, Action::Buy).unwrap().0,
        quote_of(&round, 1, Action::Buy).unwrap().0
    );
}

#[test]
fn test_quote_prices_round_away_from_the_reference_on_the_grid() {
    let mut gate = gate(1);
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let half = quoter.model().half_spread(0).unwrap();
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    let (bid, _) = quote_of(&round, 0, Action::Buy).unwrap();
    let (ask, _) = quote_of(&round, 0, Action::Sell).unwrap();
    assert!(bid <= dec!(100) - half && dec!(100) - half - bid < dec!(0.0001));
    assert!(ask >= dec!(100) + half && ask - dec!(100) - half < dec!(0.0001));
    assert_eq!(bid.scale().max(ask.scale()), 4);
}

#[test]
fn test_quote_price_not_positive_withholds_the_side_and_logs_it() {
    let mut gate = gate(1);
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let round = quoter.quote(&mut gate, &[dec!(0.0001)], 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    assert!(quote_of(&round, 0, Action::Sell).is_some());
    assert!(matches!(
        &find(&round, 0, Action::Buy)[0].kind,
        ConstraintKind::NonPositivePrice { price } if *price <= Decimal::ZERO
    ));
}

#[test]
fn test_quote_inventory_limit_withholds_the_side_that_breaches_it() {
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(5));
    let mut quoter = MultiAssetQuoter::new(
        model(1, dec!(0), RiskView::Correlated, dec!(5)),
        ids(1),
        AMPLE,
    )
    .unwrap();
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    assert!(quote_of(&round, 0, Action::Sell).is_some());
    assert_eq!(
        find(&round, 0, Action::Buy)[0].kind,
        ConstraintKind::InventoryLimit {
            resulting: dec!(6),
            limit: dec!(5),
        }
    );
    // Short at the limit it is the ask that goes, and exactly at the limit a
    // trade is allowed.
    let mut short = self::gate(1);
    hold(&mut short, 0, dec!(-5));
    let round = quoter.quote(&mut short, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_some());
    assert!(quote_of(&round, 0, Action::Sell).is_none());
    let mut inside = self::gate(1);
    hold(&mut inside, 0, dec!(4));
    let round = quoter.quote(&mut inside, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_some());
}

#[test]
fn test_quote_inventory_beyond_the_limit_can_still_be_reduced() {
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(8));
    let mut quoter = MultiAssetQuoter::new(
        model(1, dec!(0), RiskView::Correlated, dec!(5)),
        ids(1),
        AMPLE,
    )
    .unwrap();
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Sell).is_some());
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    let mut short = self::gate(1);
    hold(&mut short, 0, dec!(-8));
    let round = quoter.quote(&mut short, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_some());
    assert!(quote_of(&round, 0, Action::Sell).is_none());
}

#[test]
fn test_quote_inventory_limit_counts_orders_still_live() {
    // Flat, limit 1: a bid fits, and a second one must not, on either side.
    let mut gate = gate(1);
    let mut quoter = MultiAssetQuoter::new(
        model(1, dec!(0), RiskView::Correlated, dec!(1)),
        ids(1),
        AMPLE,
    )
    .unwrap();
    let first = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(first.quotes.len(), 2);
    let second = quoter.quote(&mut gate, &marks(1), 2).unwrap();
    assert!(second.quotes.is_empty(), "{:?}", second.quotes);
    assert_eq!(
        find(&second, 0, Action::Buy)[0].kind,
        ConstraintKind::InventoryLimit {
            resulting: dec!(2),
            limit: dec!(1),
        }
    );
    assert_eq!(
        find(&second, 0, Action::Sell)[0].kind,
        ConstraintKind::InventoryLimit {
            resulting: dec!(-2),
            limit: dec!(1),
        }
    );
    // A cancelled order no longer counts, while one with a cancel pending does.
    let buy = &first.quotes[0].order_id;
    gate.request_cancel(buy).unwrap();
    let pending = quoter.quote(&mut gate, &marks(1), 3).unwrap();
    assert!(quote_of(&pending, 0, Action::Buy).is_none());
    gate.on_cancel_confirmed(buy).unwrap();
    let replaced = quoter.quote(&mut gate, &marks(1), 4).unwrap();
    assert!(quote_of(&replaced, 0, Action::Buy).is_some());
}

/// The live orders of `gate` in the range they can move each inventory.
fn reachable_box(gate: &RiskGate, count: usize) -> Vec<(Decimal, Decimal)> {
    (0..count)
        .map(|index| {
            let id = instrument_id(index);
            let held = gate
                .ledger()
                .position(&id)
                .map_or(Decimal::ZERO, |p| p.quantity);
            let (mut low, mut high) = (held, held);
            for order in gate.live_orders().filter(|order| order.instrument == id) {
                let remaining = order.remaining().unwrap();
                match order.side {
                    Action::Buy => high += remaining,
                    _ => low -= remaining,
                }
            }
            (low, high)
        })
        .collect()
}

/// The true largest `q' Sigma q` over the box: a convex function peaks at a vertex.
fn largest_variance(model: &MultiAssetModel, reach: &[(Decimal, Decimal)]) -> Decimal {
    let mut largest = Decimal::MIN;
    for corner in 0..(1_u32 << reach.len()) {
        let q: Vec<Decimal> = reach
            .iter()
            .enumerate()
            .map(|(i, (low, high))| if corner >> i & 1 == 1 { *high } else { *low })
            .collect();
        largest = largest.max(model.portfolio_variance(&q).unwrap());
    }
    largest
}

#[test]
fn test_portfolio_limit_two_bids_that_can_fill_together_are_not_both_admitted() {
    let mut gate = gate(2);
    let mut quoter = MultiAssetQuoter::new(
        model(2, dec!(0), RiskView::Correlated, dec!(100)),
        ids(2),
        AMPLE,
    )
    .unwrap()
    .with_max_portfolio_variance(dec!(1.1))
    .unwrap();
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_some());
    assert!(quote_of(&round, 1, Action::Buy).is_none());
    assert!(matches!(
        find(&round, 1, Action::Buy)[0].kind,
        ConstraintKind::PortfolioRisk { .. }
    ));
    let reach = reachable_box(&gate, 2);
    let worst = largest_variance(quoter.model(), &reach);
    assert!(worst <= dec!(1.1), "{worst} over {reach:?}");
}

#[test]
fn test_portfolio_limit_counts_orders_still_live_across_rounds() {
    let mut gate = gate(1);
    let mut quoter = quoter(1, dec!(0), AMPLE)
        .with_max_portfolio_variance(dec!(4))
        .unwrap();
    // Variance 1 then 4 are allowed; a third live bid could reach 9.
    for round in 0..2 {
        let done = quoter.quote(&mut gate, &marks(1), round).unwrap();
        assert!(quote_of(&done, 0, Action::Buy).is_some(), "round {round}");
    }
    let third = quoter.quote(&mut gate, &marks(1), 2).unwrap();
    assert!(quote_of(&third, 0, Action::Buy).is_none());
    assert_eq!(
        find(&third, 0, Action::Buy)[0].kind,
        ConstraintKind::PortfolioRisk {
            variance: dec!(9),
            limit: dec!(4),
        }
    );
}

#[test]
fn test_portfolio_limit_withheld_side_does_not_count_against_the_next_side() {
    // Long 2 of asset 0, independent assets. Its bid could reach 9 and goes. The bid
    // of asset 1 is measured without it: the bound is 5.5, not the 8.75 and more it
    // would be if the bid withheld still counted.
    let mut gate = gate(2);
    hold(&mut gate, 0, dec!(2));
    let mut quoter = MultiAssetQuoter::new(
        model(2, dec!(0), RiskView::Correlated, dec!(100)),
        ids(2),
        AMPLE,
    )
    .unwrap()
    .with_max_portfolio_variance(dec!(6))
    .unwrap();
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    assert!(quote_of(&round, 1, Action::Buy).is_some());
}

#[test]
fn test_portfolio_limit_holds_or_never_rises_over_every_reachable_state() {
    // Inventories on both sides of the limit, for every view and correlation, with
    // orders left live across rounds: the true largest variance, by enumeration,
    // ends under the limit or no higher than it started.
    let limit = dec!(12);
    let start = [
        [dec!(0), dec!(0), dec!(0)],
        [dec!(1), dec!(-1), dec!(1)],
        [dec!(2), dec!(2), dec!(0)],
        [dec!(4), dec!(4), dec!(4)],
        [dec!(-5), dec!(3), dec!(-2)],
    ];
    for view in [RiskView::Correlated, RiskView::Diagonal] {
        for rho in [dec!(-0.4), dec!(0), dec!(0.5), dec!(0.9)] {
            for position in &start {
                let mut gate = gate(3);
                for (index, quantity) in position.iter().enumerate() {
                    if !quantity.is_zero() {
                        hold(&mut gate, index, *quantity);
                    }
                }
                let mut quoter =
                    MultiAssetQuoter::new(model(3, rho, view, dec!(100)), ids(3), AMPLE)
                        .unwrap()
                        .with_max_portfolio_variance(limit)
                        .unwrap();
                let mut before = largest_variance(quoter.model(), &reachable_box(&gate, 3));
                for round in 0..4 {
                    quoter.quote(&mut gate, &marks(3), round).unwrap();
                    let after = largest_variance(quoter.model(), &reachable_box(&gate, 3));
                    assert!(
                        after <= limit.max(before),
                        "{view:?} rho {rho} from {position:?}: {before} to {after}"
                    );
                    before = after;
                }
            }
        }
    }
}

/// A portfolio of the same size under a variance limit between its hedged and
/// concentrated variances. The hedged one reaches at most 8.8 when all four sides
/// fill in any combination, and the bound on that is 11.
fn risk_round(view: RiskView, position: [Decimal; 2]) -> QuoteRound {
    let mut gate = gate(2);
    hold(&mut gate, 0, position[0]);
    hold(&mut gate, 1, position[1]);
    let mut quoter = MultiAssetQuoter::new(model(2, dec!(0.9), view, dec!(100)), ids(2), AMPLE)
        .unwrap()
        .with_max_portfolio_variance(dec!(12))
        .unwrap();
    quoter.quote(&mut gate, &marks(2), 1).unwrap()
}

#[test]
fn test_portfolio_limit_concentrated_withholds_the_risk_raising_sides_only() {
    let concentrated = risk_round(RiskView::Correlated, [dec!(5), dec!(5)]);
    for asset in 0..2 {
        assert!(quote_of(&concentrated, asset, Action::Buy).is_none());
        assert!(
            matches!(
                &find(&concentrated, asset, Action::Buy)[0].kind,
                ConstraintKind::PortfolioRisk { variance, limit }
                    if *variance > *limit && *limit == dec!(12)
            ),
            "{:?}",
            concentrated.constraints
        );
        // Selling lowers the variance, so it stays even beyond the limit.
        assert!(quote_of(&concentrated, asset, Action::Sell).is_some());
    }
}

#[test]
fn test_portfolio_limit_hedged_portfolio_keeps_quoting_both_sides() {
    let hedged = risk_round(RiskView::Correlated, [dec!(5), dec!(-5)]);
    assert_eq!(hedged.quotes.len(), 4, "{:?}", hedged.constraints);
    assert!(hedged.constraints.is_empty());
}

#[test]
fn test_portfolio_limit_diagonal_view_cannot_tell_hedged_from_concentrated() {
    let hedged = risk_round(RiskView::Diagonal, [dec!(5), dec!(-5)]);
    let concentrated = risk_round(RiskView::Diagonal, [dec!(5), dec!(5)]);
    assert_eq!(hedged.constraints.len(), concentrated.constraints.len());
    assert!(!hedged.constraints.is_empty());
}

#[test]
fn test_capital_budget_adding_instruments_does_not_consume_more_capital() {
    let budget = dec!(500);
    let mut sizes = Vec::new();
    for count in 1..=4 {
        let mut gate = gate(count);
        let mut quoter = quoter(count, dec!(0.3), budget);
        let round = quoter.quote(&mut gate, &marks(count), 1).unwrap();
        // Read from the ledger, not from what the quoter reports.
        let held = reserved_in_ledger(&gate, count, dec!(100));
        assert!(held <= budget, "{count} instruments hold {held}");
        assert_eq!(round.capital.reserved, held);
        assert_eq!(round.quotes.len(), count * 2);
        sizes.push(quote_of(&round, 0, Action::Buy).unwrap().1);
    }
    // One instrument fits; the others share the same budget, ever thinner.
    assert_eq!(sizes[0], dec!(1));
    assert!(sizes.windows(2).all(|pair| pair[1] <= pair[0]), "{sizes:?}");
    assert!(sizes[3] < sizes[1]);
}

#[test]
fn test_capital_budget_scales_every_side_by_the_same_factor_and_logs_it() {
    let budget = dec!(200);
    let mut gate = gate(2);
    let mut quoter = quoter(2, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert_eq!(round.quotes.len(), 4);
    let first = round.quotes[0].quantity;
    assert!(round.quotes.iter().all(|q| q.quantity == first));
    // A bid holds its price and an ask holds the reference price, per unit.
    let bid = round.quotes[0].price;
    let demand = dec!(2) * (bid + dec!(100));
    let expected =
        (budget / demand).round_dp_with_strategy(4, rust_decimal::RoundingStrategy::ToZero);
    assert_eq!(first, expected);
    assert_eq!(round.constraints.len(), 4);
    assert_eq!(
        round.constraints[0].kind,
        ConstraintKind::CapitalBudget {
            requested: dec!(1),
            allowed: first,
            budget,
        }
    );
}

#[test]
fn test_capital_budget_too_small_for_a_lot_withholds_everything() {
    let mut gate = gate(1);
    let mut quoter = quoter(1, dec!(0), dec!(0.001));
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(round.quotes.is_empty());
    assert_eq!(round.capital.committed, Decimal::ZERO);
    assert_eq!(round.constraints.len(), 2);
    assert!(round.constraints.iter().all(|c| matches!(
        c.kind,
        ConstraintKind::CapitalBudget { allowed, .. } if allowed.is_zero()
    )));
}

#[test]
fn test_capital_budget_counts_orders_still_live_from_a_previous_round() {
    let budget = dec!(300);
    let mut gate = gate(1);
    let mut quoter = quoter(1, dec!(0), budget);
    let first = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(first.quotes.len(), 2);
    let second = quoter.quote(&mut gate, &marks(1), 2).unwrap();
    let held = reserved_in_ledger(&gate, 1, dec!(100));
    assert!(held <= budget, "{held}");
    assert!(second.quotes[0].quantity < dec!(1));
    // Spending it all, a third round has nothing left to place.
    let third = quoter.quote(&mut gate, &marks(1), 3).unwrap();
    assert!(third.quotes.is_empty());
    assert!(reserved_in_ledger(&gate, 1, dec!(100)) <= budget);
}

/// Four sides on a fine grid with a fee of 1 %, against a budget that binds.
fn fee_round(fee_asset: margin_rs::risk::FeeAsset) -> (QuoteRound, Decimal) {
    let budget = dec!(200);
    let mut gate = gate(2);
    for index in 0..2 {
        gate.set_limits(
            &instrument_id(index),
            InstrumentLimits {
                max_fee_rate: dec!(0.01),
                fee_asset,
                ..InstrumentLimits::default()
            },
        )
        .unwrap();
    }
    let mut quoter = quoter(2, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert!(reserved_in_ledger(&gate, 2, dec!(100)) <= budget);
    (round, budget)
}

#[test]
fn test_capital_budget_sizing_counts_a_declared_fee_on_the_side_that_reserves_it() {
    // The sizing stage sees the fee, so all four sides fit and none is released.
    for fee_asset in [
        margin_rs::risk::FeeAsset::Quote,
        margin_rs::risk::FeeAsset::Base,
    ] {
        let (round, budget) = fee_round(fee_asset);
        assert_eq!(
            round.quotes.len(),
            4,
            "{fee_asset:?}: {:?}",
            round.constraints
        );
        assert!(
            round.capital.reserved > dec!(0.99) * budget,
            "{fee_asset:?}"
        );
    }
}

#[test]
fn test_quote_round_capital_matches_a_fresh_read_with_mixed_outcomes() {
    // Long inventory, so some of it unwinds, with one side reduced by the gate and
    // one released over budget: the copy kept during the round must still agree.
    let budget = dec!(250);
    let mut gate = gate_with(2, dec!(1000000), dec!(1000), 2);
    hold(&mut gate, 1, dec!(1));
    for (index, limit) in [(0, Some(dec!(0.25))), (1, None)] {
        gate.set_limits(
            &instrument_id(index),
            InstrumentLimits {
                max_fee_rate: dec!(0.001),
                max_order_quantity: limit,
                ..InstrumentLimits::default()
            },
        )
        .unwrap();
    }
    let mut quoter = quoter(2, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    assert_eq!(quoter.capital_use(&gate, &marks(2)).unwrap(), round.capital);
    assert!(round.capital.committed <= budget, "{:?}", round.capital);
    assert!(
        round
            .constraints
            .iter()
            .any(|c| matches!(c.kind, ConstraintKind::MarginReduced { .. })),
        "{:?}",
        round.constraints
    );
    assert!(
        round.constraints.iter().any(|c| matches!(
            c.kind,
            ConstraintKind::CapitalBudget { allowed, .. } if allowed.is_zero()
        )),
        "{:?}",
        round.constraints
    );
}

#[test]
fn test_capital_budget_order_over_budget_after_reservation_rounding_is_released() {
    // A coarse quote grid and a fee: the gate reserves about 10 % more per lot
    // than the notional the sizing stage can see.
    let budget = dec!(150);
    let mut gate = gate_with(1, dec!(1000000), dec!(1000), 2);
    gate.set_limits(
        &instrument_id(0),
        InstrumentLimits {
            max_fee_rate: dec!(0.001),
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    let mut quoter = quoter(1, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    let held = reserved_in_ledger(&gate, 1, dec!(100));
    assert!(held <= budget, "{held}");
    assert_eq!(round.capital.reserved, held);
    let released: Vec<_> = round
        .constraints
        .iter()
        .filter(|c| {
            matches!(
                c.kind,
                ConstraintKind::CapitalBudget { allowed, .. } if allowed.is_zero()
            )
        })
        .collect();
    assert_eq!(released.len(), 1, "{:?}", round.constraints);
    // The released side left no order standing and no quote.
    assert!(quote_of(&round, 0, released[0].side).is_none());
}

#[test]
fn test_capital_budget_counts_inventory_already_deployed() {
    // Long 50 of each at 100 is 10000 deployed, far beyond the budget: no bid may
    // add to it, however little the open orders hold.
    let budget = dec!(150);
    let mut gate = gate(2);
    hold(&mut gate, 0, dec!(50));
    hold(&mut gate, 1, dec!(50));
    let mut quoter = quoter(2, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(2), 1).unwrap();
    for asset in 0..2 {
        assert!(quote_of(&round, asset, Action::Buy).is_none());
        assert!(matches!(
            find(&round, asset, Action::Buy)[0].kind,
            ConstraintKind::CapitalBudget { allowed, .. } if allowed.is_zero()
        ));
        // Selling what is held commits nothing new, and it stays at full size.
        assert_eq!(quote_of(&round, asset, Action::Sell).unwrap().1, dec!(1));
    }
    assert_eq!(round.capital.deployed, dec!(10000));
    assert_eq!(round.capital.reserved, dec!(200));
    assert_eq!(round.capital.committed, dec!(10000));
}

#[test]
fn test_capital_budget_short_inventory_counts_and_its_buy_back_stays() {
    let budget = dec!(150);
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(-50));
    let mut quoter = quoter(1, dec!(0), budget);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(round.capital.deployed, dec!(5000));
    assert!(quote_of(&round, 0, Action::Sell).is_none());
    assert!(matches!(
        find(&round, 0, Action::Sell)[0].kind,
        ConstraintKind::CapitalBudget { allowed, .. } if allowed.is_zero()
    ));
    let (bid, size) = quote_of(&round, 0, Action::Buy).unwrap();
    assert_eq!(size, dec!(1));
    // The bid's cash is counted as covering the short, not added to it.
    assert_eq!(round.capital.reserved, bid);
    assert_eq!(round.capital.committed, dec!(5000));
}

#[test]
fn test_capital_budget_side_partly_unwinding_keeps_that_part_when_scaled() {
    // Long 0.4 with a budget of exactly what is deployed: the ask keeps the 0.4 it
    // unwinds and loses the 0.6 that would go short; the bid has nothing left.
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(0.4));
    let mut quoter = quoter(1, dec!(0), dec!(40));
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    assert_eq!(quote_of(&round, 0, Action::Sell).unwrap().1, dec!(0.4));
    assert_eq!(
        find(&round, 0, Action::Sell)[0].kind,
        ConstraintKind::CapitalBudget {
            requested: dec!(1),
            allowed: dec!(0.4),
            budget: dec!(40),
        }
    );
    assert_eq!(round.capital.committed, dec!(40));
}

#[test]
fn test_capital_budget_unwinding_order_over_budget_stands_and_adds_only_its_fee() {
    let mut gate = gate(1);
    gate.set_limits(
        &instrument_id(0),
        InstrumentLimits {
            max_fee_rate: dec!(0.01),
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    hold(&mut gate, 0, dec!(-50));
    let mut quoter = quoter(1, dec!(0), dec!(150));
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    let (bid, size) = quote_of(&round, 0, Action::Buy).unwrap();
    assert_eq!(size, dec!(1));
    let committed = round.capital.committed - dec!(5000);
    assert!(
        committed > Decimal::ZERO && committed <= (bid * dec!(0.01)).ceil(),
        "{committed}"
    );
}

#[test]
fn test_capital_budget_unwinding_part_leaves_the_rest_of_the_budget_alone() {
    // Long 2 at 100, budget 350: 150 left. The ask only unwinds, so the bid has all
    // of it and neither side is scaled.
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(2));
    let mut quoter = quoter(1, dec!(0), dec!(350));
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(quote_of(&round, 0, Action::Buy).unwrap().1, dec!(1));
    assert_eq!(quote_of(&round, 0, Action::Sell).unwrap().1, dec!(1));
    assert!(round.constraints.is_empty(), "{:?}", round.constraints);
}

#[test]
fn test_capital_budget_inventory_unwinds_only_once_across_live_orders() {
    // Long 1, budget all deployed: the first ask unwinds it and stands. A second
    // ask left beside it would go short, so it needs capital there is not.
    let mut long = gate(1);
    hold(&mut long, 0, dec!(1));
    let mut quoter_long = quoter(1, dec!(0), dec!(100));
    let first = quoter_long.quote(&mut long, &marks(1), 1).unwrap();
    assert_eq!(quote_of(&first, 0, Action::Sell).unwrap().1, dec!(1));
    let second = quoter_long.quote(&mut long, &marks(1), 2).unwrap();
    assert!(quote_of(&second, 0, Action::Sell).is_none());
    // The same for a bid that covers a short.
    let mut short = gate(1);
    hold(&mut short, 0, dec!(-1));
    let mut quoter_short = quoter(1, dec!(0), dec!(100));
    let first = quoter_short.quote(&mut short, &marks(1), 1).unwrap();
    assert_eq!(quote_of(&first, 0, Action::Buy).unwrap().1, dec!(1));
    let second = quoter_short.quote(&mut short, &marks(1), 2).unwrap();
    assert!(quote_of(&second, 0, Action::Buy).is_none());
}

#[test]
fn test_capital_use_covers_a_short_at_the_lowest_live_bid() {
    // Short 3 at a mark of 100, with bids of 1 at 90 and 1 at 80: 300 deployed and
    // 170 reserved. Valuing the 2 covered at 80 leaves 310 committed; at the higher
    // bid it would read 290, and the budget would pass more than it should.
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(-3));
    for (id, price) in [("bid-90", dec!(90)), ("bid-80", dec!(80))] {
        let request = margin_rs::risk::OrderRequest::new(
            margin_rs::OrderId::new(id).unwrap(),
            instrument_id(0),
            Action::Buy,
            price,
            dec!(1),
        );
        assert!(!gate.submit(&request, 0).is_rejected());
    }
    let use_of = quoter(1, dec!(0), AMPLE)
        .capital_use(&gate, &marks(1))
        .unwrap();
    assert_eq!(use_of.deployed, dec!(300));
    assert_eq!(use_of.reserved, dec!(170));
    assert_eq!(use_of.committed, dec!(310));
}

#[test]
fn test_capital_use_does_not_double_count_a_sell_of_inventory_held() {
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(5));
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    let (bid, _) = quote_of(&round, 0, Action::Buy).unwrap();
    assert_eq!(round.capital.deployed, dec!(500));
    assert_eq!(round.capital.reserved, bid + dec!(100));
    // The ask's 100 is part of the 500 already deployed; the bid's cash is new.
    assert_eq!(round.capital.committed, dec!(500) + bid);
    assert_eq!(quoter.capital_use(&gate, &marks(1)).unwrap(), round.capital);
}

#[test]
fn test_capital_budget_marks_can_pass_the_budget_and_rounds_then_only_reduce() {
    let budget = dec!(250);
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(1));
    let mut quoter = quoter(1, dec!(0), budget);
    let calm = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert!(quote_of(&calm, 0, Action::Buy).is_some());
    for quote in &calm.quotes {
        gate.on_cancel_confirmed(&quote.order_id).unwrap();
    }
    // The same long at a price of 300 is over budget by itself.
    let round = quoter.quote(&mut gate, &[dec!(300)], 2).unwrap();
    assert!(quote_of(&round, 0, Action::Buy).is_none());
    assert_eq!(quote_of(&round, 0, Action::Sell).unwrap().1, dec!(1));
    assert_eq!(round.capital.committed, dec!(300));
}

#[test]
fn test_quote_through_the_reference_is_reported_and_still_stands() {
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(50));
    let mut quoter = quoter(1, dec!(0), AMPLE);
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    let (ask, _) = quote_of(&round, 0, Action::Sell).unwrap();
    assert!(ask <= dec!(100), "{ask}");
    assert_eq!(
        find(&round, 0, Action::Sell)[0].kind,
        ConstraintKind::ThroughReference {
            price: ask,
            reference: dec!(100),
        }
    );
    assert!(find(&round, 0, Action::Buy).is_empty());
    let mut short = self::gate(1);
    hold(&mut short, 0, dec!(-50));
    let round = quoter.quote(&mut short, &marks(1), 1).unwrap();
    let (bid, _) = quote_of(&round, 0, Action::Buy).unwrap();
    assert!(bid >= dec!(100), "{bid}");
    assert!(matches!(
        find(&round, 0, Action::Buy)[0].kind,
        ConstraintKind::ThroughReference { .. }
    ));
}

#[test]
fn test_quote_new_quoter_on_a_used_gate_does_not_collide_with_the_earlier_orders() {
    let mut gate = gate(1);
    let mut first = quoter(1, dec!(0), AMPLE);
    let earlier = first.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(earlier.quotes.len(), 2);
    let mut second = quoter(1, dec!(0), AMPLE);
    let round = second.quote(&mut gate, &marks(1), 2).unwrap();
    assert_eq!(round.quotes.len(), 2, "{:?}", round.constraints);
    assert!(round.constraints.is_empty());
    for (old, new) in earlier.quotes.iter().zip(&round.quotes) {
        assert_ne!(old.order_id, new.order_id);
    }
    assert_eq!(
        gate.live_orders().count(),
        4,
        "the earlier quotes still stand"
    );
    // The first quoter resumes past the ids the second one used.
    let again = first.quote(&mut gate, &marks(1), 3).unwrap();
    assert_eq!(again.quotes.len(), 2, "{:?}", again.constraints);
}

#[test]
fn test_constraints_display_names_the_cause() {
    let mut gate = gate(1);
    hold(&mut gate, 0, dec!(5));
    let mut quoter = MultiAssetQuoter::new(
        model(1, dec!(0), RiskView::Correlated, dec!(5)),
        ids(1),
        AMPLE,
    )
    .unwrap();
    let round = quoter.quote(&mut gate, &marks(1), 1).unwrap();
    assert_eq!(
        round.constraints[0].to_string(),
        "asset 0 Buy: inventory 6 would exceed the limit 5"
    );
}

#[test]
fn test_quoter_invalid_configuration_is_refused() {
    let two = || model(2, dec!(0), RiskView::Correlated, dec!(10));
    for budget in [dec!(0), dec!(-1)] {
        assert!(matches!(
            MultiAssetQuoter::new(two(), ids(2), budget),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    assert!(matches!(
        MultiAssetQuoter::new(two(), ids(1), AMPLE),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        MultiAssetQuoter::new(two(), vec![instrument_id(0), instrument_id(0)], AMPLE),
        Err(MMError::InvalidConfiguration(_))
    ));
    let quoter = MultiAssetQuoter::new(two(), ids(2), AMPLE).unwrap();
    assert!(matches!(
        quoter.with_max_portfolio_variance(dec!(0)),
        Err(MMError::InvalidConfiguration(_))
    ));
}

#[test]
fn test_quote_bad_marks_are_errors_and_leave_the_gate_untouched() {
    let mut gate = gate(2);
    let mut quoter = quoter(2, dec!(0), AMPLE);
    for bad in [
        vec![dec!(100)],
        vec![dec!(100), dec!(0)],
        vec![dec!(100), dec!(-1)],
    ] {
        assert!(matches!(
            quoter.quote(&mut gate, &bad, 1),
            Err(MMError::InvalidMarketState(_))
        ));
        assert!(matches!(
            quoter.capital_use(&gate, &bad),
            Err(MMError::InvalidMarketState(_))
        ));
    }
    assert_eq!(gate.live_orders().count(), 0);
}

#[test]
fn test_quote_instruments_the_budget_cannot_cover_are_errors() {
    // Unregistered.
    let mut empty = RiskGate::new(Ledger::new(LedgerConfig::default()));
    let mut one = quoter(1, dec!(0), AMPLE);
    assert!(matches!(
        one.quote(&mut empty, &marks(1), 1),
        Err(MMError::InvalidConfiguration(_))
    ));
    // Two quote assets.
    let mut ledger = Ledger::new(LedgerConfig::default());
    for code in ["AAA", "BBB", "USD", "EUR"] {
        ledger.register_asset(asset(code), 8).unwrap();
    }
    ledger
        .register_instrument(
            SpotInstrument::new(instrument_id(0), asset("AAA"), asset("USD"), 4, 4).unwrap(),
        )
        .unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(instrument_id(1), asset("BBB"), asset("EUR"), 4, 4).unwrap(),
        )
        .unwrap();
    let mut mixed = RiskGate::new(ledger);
    let mut two = quoter(2, dec!(0), AMPLE);
    assert!(matches!(
        two.quote(&mut mixed, &marks(2), 1),
        Err(MMError::InvalidConfiguration(_))
    ));
    // One base asset twice.
    let mut ledger = Ledger::new(LedgerConfig::default());
    for code in ["AAA", "USD"] {
        ledger.register_asset(asset(code), 8).unwrap();
    }
    for index in 0..2 {
        ledger
            .register_instrument(
                SpotInstrument::new(instrument_id(index), asset("AAA"), asset("USD"), 4, 4)
                    .unwrap(),
            )
            .unwrap();
    }
    let mut same = RiskGate::new(ledger);
    assert!(matches!(
        two.quote(&mut same, &marks(2), 1),
        Err(MMError::InvalidConfiguration(_))
    ));
    // A derivative is not a spot instrument.
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(asset("USD"), 8).unwrap();
    ledger
        .register_instrument(margin_rs::LinearPerpetual::new(
            margin_rs::LinearTerms::new(instrument_id(0), asset("USD"), dec!(1), 4, 4).unwrap(),
        ))
        .unwrap();
    let mut derivative = RiskGate::new(ledger);
    assert!(matches!(
        one.quote(&mut derivative, &marks(1), 1),
        Err(MMError::InvalidConfiguration(_))
    ));
}

#[test]
fn test_quote_trade_size_finer_than_the_lot_is_an_error() {
    let fine = MultiAssetModel::new(
        vec![AssetQuotingParams::new(dec!(1), dec!(10), dec!(20), dec!(0.00001), dec!(1)).unwrap()],
        &[vec![dec!(1)]],
        dec!(0.08),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap();
    let mut quoter = MultiAssetQuoter::new(fine, ids(1), AMPLE).unwrap();
    let mut gate = gate(1);
    assert!(matches!(
        quoter.quote(&mut gate, &marks(1), 1),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert_eq!(gate.live_orders().count(), 0);
}

/// The comparison: both strategies meet the same prices and the same fill draws.
mod simulation {
    use super::*;

    pub const FEE_RATE: Decimal = dec!(0.00005);
    pub const SIGMA: [f64; 2] = [1.2, 0.6];
    pub const SCALE: f64 = 10.0;
    pub const DECAY: f64 = 20.0;
    pub const GAMMA: f64 = 0.08;
    pub const DT: f64 = 0.01;

    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
        }
    }

    /// One path: the execution conditions, fixed before any strategy runs.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Tape {
        pub marks: Vec<[Decimal; 2]>,
        /// One uniform draw per asset and side at each step.
        pub draws: Vec<[f64; 4]>,
        pub rho: f64,
    }

    impl Tape {
        pub fn new(seed: u64, steps: usize, rho: f64) -> Self {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            for _ in 0..16 {
                rng.unit();
            }
            let mut price = [100.0_f64, 100.0];
            let mut marks = Vec::with_capacity(steps);
            let mut draws = Vec::with_capacity(steps);
            for _ in 0..steps {
                marks.push([
                    Decimal::from_f64(price[0]).unwrap().round_dp(4),
                    Decimal::from_f64(price[1]).unwrap().round_dp(4),
                ]);
                draws.push([rng.unit(), rng.unit(), rng.unit(), rng.unit()]);
                let (a, b) = (rng.normal(), rng.normal());
                let root = DT.sqrt();
                price[0] += SIGMA[0] * root * a;
                price[1] += SIGMA[1] * root * (rho * a + (1.0 - rho * rho).sqrt() * b);
            }
            Self { marks, draws, rho }
        }

        fn variance(&self, q: [f64; 2]) -> f64 {
            let (a, b) = (q[0] * SIGMA[0], q[1] * SIGMA[1]);
            a * a + 2.0 * self.rho * a * b + b * b
        }
    }

    /// One strategy: its assumed correlation, its view and its capital budget.
    #[derive(Debug, Clone, Copy)]
    pub struct Setup {
        pub assumed_rho: f64,
        pub view: RiskView,
        pub budget: Decimal,
    }

    /// The result of a strategy on a tape.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Outcome {
        /// Change of cash, fees included, plus the final inventory at the final mark.
        pub net_pnl: f64,
        /// `net_pnl` minus the running penalty of Model B under the true covariance.
        pub objective: f64,
        /// Mean of `sqrt(q' Sigma q)` under the true covariance.
        pub mean_risk: f64,
        /// Mean absolute inventory per asset: with linear instruments, the delta.
        pub mean_abs_inventory: [f64; 2],
        /// Mean of committed capital over the capital budget.
        pub mean_utilization: f64,
        /// Largest of committed capital over the capital budget. Marks can take it
        /// past one between rounds; a round cannot.
        pub peak_utilization: f64,
        /// Mean of reserved capital alone over the capital budget.
        pub mean_reserved: f64,
        pub fills: usize,
        pub constraints: usize,
    }

    fn f64_of(value: Decimal) -> f64 {
        value.to_f64().unwrap()
    }

    pub fn run(tape: &Tape, setup: Setup) -> Outcome {
        let rho = Decimal::from_f64(setup.assumed_rho).unwrap();
        let assets = SIGMA
            .iter()
            .map(|sigma| {
                AssetQuotingParams::new(
                    Decimal::from_f64(*sigma).unwrap(),
                    Decimal::from_f64(SCALE).unwrap(),
                    Decimal::from_f64(DECAY).unwrap(),
                    dec!(1),
                    dec!(12),
                )
                .unwrap()
            })
            .collect();
        let model = MultiAssetModel::new(
            assets,
            &uniform_correlation(2, rho),
            Decimal::from_f64(GAMMA).unwrap(),
            QuotingObjective::RiskAdjustedExpectation,
            setup.view,
        )
        .unwrap();
        let mut quoter = MultiAssetQuoter::new(model, ids(2), setup.budget).unwrap();
        let mut gate = gate_with(2, dec!(1000000), dec!(1000), 8);
        let limits = InstrumentLimits {
            max_fee_rate: FEE_RATE,
            ..InstrumentLimits::default()
        };
        for index in 0..2 {
            gate.set_limits(&instrument_id(index), limits.clone())
                .unwrap();
        }
        let cash_start = gate.ledger().balance(&asset(QUOTE)).unwrap().total;

        let inventory = |gate: &RiskGate| -> [Decimal; 2] {
            [0, 1].map(|index| {
                gate.ledger()
                    .position(&instrument_id(index))
                    .map_or(Decimal::ZERO, |p| p.quantity)
            })
        };

        let steps = tape.marks.len();
        let (mut risk_sum, mut penalty, mut utilization_sum, mut peak) = (0.0, 0.0, 0.0, 0.0_f64);
        let mut reserved_sum = 0.0;
        let mut absolute = [0.0_f64; 2];
        let (mut fills, mut constraints) = (0, 0);
        for (step, marks) in tape.marks.iter().enumerate() {
            let before = quoter.capital_use(&gate, marks).unwrap().committed;
            let round = quoter.quote(&mut gate, marks, step as u64).unwrap();
            // The only allowance is the fee an unwinding order reserves.
            let fees: Decimal = round
                .quotes
                .iter()
                .map(|quote| quote.price * quote.quantity * simulation::FEE_RATE + dec!(0.00000001))
                .sum();
            assert!(
                round.capital.committed <= setup.budget.max(before) + fees,
                "a round raised committed capital past the budget: {before} to {:?}",
                round.capital
            );
            // What the round tracked as it went is what a fresh read finds.
            assert_eq!(quoter.capital_use(&gate, marks).unwrap(), round.capital);
            constraints += round.constraints.len();
            let utilization = f64_of(round.capital.committed) / f64_of(setup.budget);
            utilization_sum += utilization;
            reserved_sum += f64_of(round.capital.reserved) / f64_of(setup.budget);
            peak = peak.max(utilization);
            for quote in &round.quotes {
                gate.on_venue_accepted(&quote.order_id).unwrap();
                let side_index = usize::from(quote.side == Action::Sell);
                let draw = tape.draws[step][quote.asset * 2 + side_index];
                let mark = marks[quote.asset];
                let distance = match quote.side {
                    Action::Buy => mark - quote.price,
                    _ => quote.price - mark,
                };
                let intensity = SCALE * (-DECAY * f64_of(distance)).exp();
                if draw < 1.0 - (-intensity * DT).exp() {
                    let fee = (quote.price * quote.quantity * FEE_RATE).round_dp_with_strategy(
                        8,
                        rust_decimal::RoundingStrategy::ToPositiveInfinity,
                    );
                    let execution = Execution::new(
                        ExecutionId::new(format!("x-{}", quote.order_id)).unwrap(),
                        instrument_id(quote.asset),
                        quote.side,
                        quote.price,
                        quote.quantity,
                    )
                    .with_fee(asset(QUOTE), fee);
                    gate.on_fill(&quote.order_id, &FillReport::new(execution, quote.quantity))
                        .unwrap();
                    fills += 1;
                } else {
                    gate.request_cancel(&quote.order_id).unwrap();
                    gate.on_cancel_confirmed(&quote.order_id).unwrap();
                }
            }
            let q = inventory(&gate).map(f64_of);
            let variance = tape.variance(q);
            risk_sum += variance.sqrt();
            penalty += 0.5 * GAMMA * variance * DT;
            absolute[0] += q[0].abs();
            absolute[1] += q[1].abs();
        }
        let last = tape.marks.last().unwrap();
        let cash = gate.ledger().balance(&asset(QUOTE)).unwrap().total - cash_start;
        let q = inventory(&gate);
        let net = f64_of(cash + q[0] * last[0] + q[1] * last[1]);
        assert!(
            gate.halted().is_none(),
            "the gate never trips in these runs"
        );
        Outcome {
            net_pnl: net,
            objective: net - penalty,
            mean_risk: risk_sum / steps as f64,
            mean_abs_inventory: absolute.map(|sum| sum / steps as f64),
            mean_utilization: utilization_sum / steps as f64,
            peak_utilization: peak,
            mean_reserved: reserved_sum / steps as f64,
            fills,
            constraints,
        }
    }

    /// Mean of `metric` over paths and the standard error of its paired gap.
    pub fn paired(
        left: &[Outcome],
        right: &[Outcome],
        metric: impl Fn(&Outcome) -> f64,
    ) -> (f64, f64, f64, f64) {
        let n = left.len() as f64;
        let mean = |side: &[Outcome]| side.iter().map(&metric).sum::<f64>() / n;
        let gaps: Vec<f64> = left
            .iter()
            .zip(right)
            .map(|(l, r)| metric(l) - metric(r))
            .collect();
        let gap = gaps.iter().sum::<f64>() / n;
        let spread = (gaps.iter().map(|g| (g - gap).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        (mean(left), mean(right), gap, spread / n.sqrt())
    }
}

use simulation::{Outcome, Setup, Tape, paired, run};

/// A named column of the evidence table.
type Metric = (&'static str, fn(&Outcome) -> f64);

fn runs(
    rho: f64,
    assumed: f64,
    view: RiskView,
    budget: Decimal,
    paths: u64,
    steps: usize,
) -> Vec<Outcome> {
    (0..paths)
        .map(|seed| {
            run(
                &Tape::new(seed + 1, steps, rho),
                Setup {
                    assumed_rho: assumed,
                    view,
                    budget,
                },
            )
        })
        .collect()
}

#[test]
fn test_comparison_tape_replays_identically_and_differs_by_seed() {
    let tape = Tape::new(3, 40, 0.5);
    assert_eq!(tape, Tape::new(3, 40, 0.5));
    assert_ne!(tape, Tape::new(4, 40, 0.5));
    let setup = Setup {
        assumed_rho: 0.5,
        view: RiskView::Correlated,
        budget: AMPLE,
    };
    assert_eq!(run(&tape, setup), run(&tape, setup));
}

#[test]
fn test_comparison_views_with_the_same_risk_get_identical_results() {
    // With no correlation the views quote the same prices, so any gap between
    // them would come from the execution conditions, which must be identical.
    for seed in 1..=3 {
        let tape = Tape::new(seed, 120, 0.0);
        let joint = run(
            &tape,
            Setup {
                assumed_rho: 0.0,
                view: RiskView::Correlated,
                budget: AMPLE,
            },
        );
        let separate = run(
            &tape,
            Setup {
                assumed_rho: 0.0,
                view: RiskView::Diagonal,
                budget: AMPLE,
            },
        );
        assert_eq!(joint, separate);
        assert!(joint.fills > 0, "the runs trade");
    }
}

#[test]
fn test_comparison_wrong_correlation_holds_more_risk_than_the_diagonal_view() {
    // Assuming 0.9 when the assets are independent: it hedges with an asset that
    // does not hedge, and the risk shows it, on the same prices and draws.
    let (rho, assumed, paths, steps) = (0.0, 0.9, 10, 200);
    let joint = runs(rho, assumed, RiskView::Correlated, AMPLE, paths, steps);
    let separate = runs(rho, assumed, RiskView::Diagonal, AMPLE, paths, steps);
    let (a, b, gap, error) = paired(&joint, &separate, |o| o.mean_risk);
    assert!(gap > 3.0 * error && gap > 0.0, "{a} {b} {gap} {error}");
}

#[test]
#[ignore = "needs many paths: cargo test --release --all-features --test multi_asset_quoting -- --ignored"]
fn test_comparison_correlated_view_holds_less_risk_under_strong_correlation() {
    // Measured on 200 paths: 1.056 against 1.095, a paired gap of -0.039 with a
    // standard error of 0.010. Fewer paths do not separate them.
    let (rho, paths, steps) = (0.9, 200, 500);
    let joint = runs(rho, rho, RiskView::Correlated, AMPLE, paths, steps);
    let separate = runs(rho, rho, RiskView::Diagonal, AMPLE, paths, steps);
    let (a, b, gap, error) = paired(&joint, &separate, |o| o.mean_risk);
    assert!(
        gap < 0.0 && gap.abs() > 2.0 * error,
        "{a} {b} {gap} {error}"
    );
}

#[test]
fn test_comparison_capital_budget_holds_in_every_step_of_every_run() {
    let budget = dec!(150);
    for view in [RiskView::Correlated, RiskView::Diagonal] {
        for outcome in runs(0.5, 0.5, view, budget, 3, 150) {
            assert!(outcome.peak_utilization <= 1.0, "{outcome:?}");
            assert!(
                outcome.peak_utilization > 0.9,
                "the budget binds: {outcome:?}"
            );
            assert!(outcome.constraints > 0, "and is logged: {outcome:?}");
        }
    }
    let ample = &runs(0.5, 0.5, RiskView::Correlated, AMPLE, 1, 150)[0];
    assert_eq!(ample.constraints, 0);
}

#[test]
#[ignore = "evidence table: cargo test --release --all-features --test multi_asset_quoting -- --ignored --nocapture"]
fn test_comparison_correlation_scenarios_print_evidence_table() {
    let (paths, steps) = (200, 500);
    println!(
        "SYNTHETIC: {paths} paths of {steps} steps of dt {} on identical tapes per row; sigma {:?}, A {}, k {}, gamma {}, fee rate {}.",
        simulation::DT,
        simulation::SIGMA,
        simulation::SCALE,
        simulation::DECAY,
        simulation::GAMMA,
        simulation::FEE_RATE,
    );
    println!(
        "Utilization: committed capital (deployed inventory plus reservations, without counting an unwinding order twice) / capital budget, read from the ledger after each round; mean reserved over budget is the same with reservations alone. Marks can pass the budget between rounds. The Greek of linear instruments is delta = inventory."
    );
    println!("| true rho | assumed | budget | metric | correlated | diagonal | gap | std err |");
    println!("|---|---|---|---|---|---|---|---|");
    for (rho, assumed, budget) in [
        (-0.5, -0.5, dec!(5000)),
        (0.0, 0.0, dec!(5000)),
        (0.5, 0.5, dec!(5000)),
        (0.9, 0.9, dec!(5000)),
        (-0.5, 0.5, dec!(5000)),
        (0.0, 0.9, dec!(5000)),
        (0.9, 0.9, dec!(500)),
    ] {
        let joint = runs(rho, assumed, RiskView::Correlated, budget, paths, steps);
        let separate = runs(rho, assumed, RiskView::Diagonal, budget, paths, steps);
        let metrics: [Metric; 9] = [
            ("net PnL", |o| o.net_pnl),
            ("objective", |o| o.objective),
            ("mean risk", |o| o.mean_risk),
            ("mean abs inventory 1", |o| o.mean_abs_inventory[0]),
            ("mean abs inventory 2", |o| o.mean_abs_inventory[1]),
            ("mean utilization", |o| o.mean_utilization),
            ("peak utilization", |o| o.peak_utilization),
            ("mean reserved over budget", |o| o.mean_reserved),
            ("fills", |o| o.fills as f64),
        ];
        for (name, metric) in metrics {
            let (a, b, gap, error) = paired(&joint, &separate, metric);
            println!(
                "| {rho} | {assumed} | {budget} | {name} | {a:.4} | {b:.4} | {gap:.4} | {error:.4} |"
            );
        }
    }
}
