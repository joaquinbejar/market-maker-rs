//! Regressions for the checked arithmetic semantics of derived signals and
//! running state near the `Decimal` bounds.

use market_maker_rs::backtest::MarketTick;
use market_maker_rs::execution::{BookLevel, OrderBookSnapshot};
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::adaptive_spread::{OrderBookImbalance, TradeFlowImbalance};
use market_maker_rs::{Decimal, dec};

#[test]
fn midpoint_of_equal_large_prices_stays_at_price() {
    let tick = MarketTick::new(0, Decimal::MAX, dec!(1), Decimal::MAX, dec!(1));
    assert_eq!(tick.mid_price(), Decimal::MAX);
    let tick = MarketTick::new(0, Decimal::MIN, dec!(1), Decimal::MIN, dec!(1));
    assert_eq!(tick.mid_price(), Decimal::MIN);
}

#[test]
fn book_imbalance_is_invariant_under_common_scale() {
    let huge = OrderBookImbalance::new(Decimal::MAX, Decimal::MAX / dec!(2), 1);
    let small = OrderBookImbalance::new(dec!(2), dec!(1), 1);
    assert!(
        (huge.imbalance - small.imbalance).abs() < dec!(0.00000001),
        "huge={} small={}",
        huge.imbalance,
        small.imbalance
    );
}

#[test]
fn trade_flow_imbalance_is_invariant_under_common_scale() {
    let huge = TradeFlowImbalance::new(Decimal::MAX, Decimal::MAX / dec!(2));
    let small = TradeFlowImbalance::new(dec!(2), dec!(1));
    assert!(
        (huge.imbalance - small.imbalance).abs() < dec!(0.00000001),
        "huge={} small={}",
        huge.imbalance,
        small.imbalance
    );
}

#[test]
fn inventory_never_saturates_cost_basis_into_wrong_average() {
    let mut position = InventoryPosition {
        quantity: dec!(1),
        avg_entry_price: Decimal::MIN,
        last_update: 0,
    };
    let before = position.clone();
    match position.update_fill(dec!(3), Decimal::MAX, 1) {
        Ok(()) => assert_eq!(position.avg_entry_price, Decimal::MAX / dec!(2)),
        Err(_) => assert_eq!(position, before),
    }
}

#[test]
fn snapshot_imbalance_is_invariant_when_depth_overflows() {
    let mut huge = OrderBookSnapshot::new("X", 0);
    huge.bids.push(BookLevel::new(dec!(1), Decimal::MAX));
    huge.bids.push(BookLevel::new(dec!(1), Decimal::MAX));
    huge.asks.push(BookLevel::new(dec!(2), Decimal::MAX));
    let mut small = OrderBookSnapshot::new("X", 0);
    small.bids.push(BookLevel::new(dec!(1), dec!(2)));
    small.bids.push(BookLevel::new(dec!(1), dec!(2)));
    small.asks.push(BookLevel::new(dec!(2), dec!(2)));
    assert!(
        (huge.imbalance() - small.imbalance()).abs() < dec!(0.00000001),
        "huge={} small={}",
        huge.imbalance(),
        small.imbalance()
    );
}
