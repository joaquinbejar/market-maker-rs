//! Regressions for the checked arithmetic semantics of derived signals and
//! running state near the `Decimal` bounds.

use market_maker_rs::analytics::order_flow::{OrderFlowAnalyzer, Trade, TradeSide};
use market_maker_rs::analytics::vpin::{VPINCalculator, VPINConfig};
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
    let tick = MarketTick::new(0, dec!(5e28), dec!(1), dec!(6e28), dec!(1));
    assert_eq!(tick.mid_price(), dec!(5.5e28));
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

#[test]
fn test_order_flow_stats_sides_beyond_bounds_keep_the_signal() {
    let mut analyzer = OrderFlowAnalyzer::new(1000);
    for side in [TradeSide::Buy, TradeSide::Buy, TradeSide::Sell] {
        analyzer.add_trade(Trade::new(dec!(1), Decimal::MAX, side, 1));
    }
    let stats = analyzer.get_stats(1);
    assert_eq!(
        stats.imbalance.round_dp(20),
        (dec!(1) / dec!(3)).round_dp(20)
    );
    assert!(stats.is_bullish());
    assert_eq!(stats.net_flow, Decimal::MAX);
    assert_eq!(stats.buy_vwap, None);
    assert_eq!(stats.sell_vwap, Some(dec!(1)));
    assert_eq!(stats.volume_ratio(), None);
}

#[test]
fn test_vpin_bucket_saturated_total_keeps_the_imbalance() {
    let config = VPINConfig::new(Decimal::MAX, 1, dec!(0.5)).unwrap();
    let mut calculator = VPINCalculator::new(config);
    for (size, side) in [
        (dec!(40000000000000000000000000000), TradeSide::Buy),
        (dec!(30000000000000000000000000000), TradeSide::Sell),
        (dec!(40000000000000000000000000000), TradeSide::Buy),
    ] {
        calculator.add_trade(&Trade::new(dec!(1), size, side, 1));
    }
    // True imbalance is (8e28 - 3e28) / 11e28 = 5/11.
    let expected = (dec!(5) / dec!(11)).round_dp(20);
    assert_eq!(calculator.get_vpin().unwrap().round_dp(20), expected);
    let bucket = calculator.get_buckets().front().unwrap();
    assert_eq!(bucket.signed_imbalance().round_dp(20), expected);
}

fn order_flow_stats_at_max(sides: &[TradeSide]) -> (Decimal, Decimal) {
    let mut analyzer = OrderFlowAnalyzer::new(1000);
    for side in sides {
        analyzer.add_trade(Trade::new(dec!(1), Decimal::MAX, *side, 1));
    }
    let stats = analyzer.get_stats(1);
    (stats.imbalance, stats.net_flow)
}

#[test]
fn test_order_flow_stats_one_sided_beyond_bounds_keeps_the_signal() {
    for count in 2..=7 {
        let buys = vec![TradeSide::Buy; count];
        assert_eq!(
            order_flow_stats_at_max(&buys),
            (Decimal::ONE, Decimal::MAX),
            "{count} buys"
        );
        let sells = vec![TradeSide::Sell; count];
        assert_eq!(
            order_flow_stats_at_max(&sells),
            (Decimal::NEGATIVE_ONE, Decimal::MIN),
            "{count} sells"
        );
    }
}

#[test]
fn test_order_flow_stats_even_and_odd_counts_beyond_bounds_keep_proportions() {
    use TradeSide::{Buy, Sell};
    for (sides, expected) in [
        (vec![Buy, Buy, Sell], dec!(1) / dec!(3)),
        (vec![Buy, Buy, Buy, Sell], dec!(0.5)),
        (vec![Buy, Sell, Sell, Sell], dec!(-0.5)),
        (vec![Buy, Buy, Sell, Sell], dec!(0)),
    ] {
        let (imbalance, _) = order_flow_stats_at_max(&sides);
        assert_eq!(imbalance.round_dp(20), expected.round_dp(20), "{sides:?}");
    }
}

#[test]
fn test_order_flow_stats_max_against_tiny_other_side_stays_near_one() {
    let mut analyzer = OrderFlowAnalyzer::new(1000);
    for (size, side) in [
        (Decimal::MAX, TradeSide::Buy),
        (Decimal::MAX, TradeSide::Buy),
        (dec!(0.000001), TradeSide::Sell),
    ] {
        analyzer.add_trade(Trade::new(dec!(1), size, side, 1));
    }
    assert_eq!(analyzer.get_stats(1).imbalance, Decimal::ONE);
}

#[test]
fn test_vpin_bucket_one_sided_saturated_total_is_fully_imbalanced() {
    for side in [TradeSide::Buy, TradeSide::Sell] {
        let config = VPINConfig::new(Decimal::MAX, 1, dec!(0.5)).unwrap();
        let mut calculator = VPINCalculator::new(config);
        let size = dec!(50000000000000000000000000000);
        calculator.add_trade(&Trade::new(dec!(1), size, side, 1));
        calculator.add_trade(&Trade::new(dec!(1), size, side, 1));
        assert_eq!(calculator.get_vpin(), Some(Decimal::ONE), "{side:?}");
    }
}

#[test]
fn test_market_tick_spread_bps_overflow_is_extreme_not_zero() {
    let tick = MarketTick::new(
        0,
        dec!(-5),
        dec!(1),
        dec!(5.0000000000000000000000000002),
        dec!(1),
    );
    assert!(tick.mid_price() > Decimal::ZERO);
    assert_eq!(tick.spread_bps(), Decimal::MAX);
    let crossed = MarketTick::new(
        0,
        dec!(5.0000000000000000000000000002),
        dec!(1),
        dec!(-5),
        dec!(1),
    );
    assert!(crossed.mid_price() > Decimal::ZERO);
    assert_eq!(crossed.spread_bps(), Decimal::MIN);
}
