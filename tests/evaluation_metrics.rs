//! Metric definitions and attribution: hand computed fixtures, then the books of
//! real engine runs. Every expected value is worked out in a comment, so a reader
//! can check it without running anything.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    Attribution, EvaluationError, EvaluationInput, ExecutedFill, FillQualitySpec, HedgeCosts,
    HedgeNotModelledReason, LedgerTotals, Mark, MetricHorizon, MetricId, MetricUnit, MetricsSpec,
    Reading, attribute, attribute_by, attribute_daily, evaluate_window, fill_quality,
    inventory_stats, max_drawdown,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestStrategy, MarketTick, SimLatencyConfig, SimulatedFill,
    VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::strategy::walk_forward::TimeWindow;

const DAY: u64 = 86_400_000;

fn mark(timestamp_ms: u64, mid: Decimal) -> Mark {
    Mark { timestamp_ms, mid }
}

fn fill(timestamp_ms: u64, side: Action, price: Decimal, quantity: Decimal) -> ExecutedFill {
    ExecutedFill {
        timestamp_ms,
        side,
        price,
        quantity,
        quote_fee: Decimal::ZERO,
        base_fee: Decimal::ZERO,
    }
}

fn window(end_ms: u64) -> TimeWindow {
    TimeWindow::new(0, end_ms).unwrap()
}

fn input<'a>(end_ms: u64, marks: &'a [Mark], fills: &'a [ExecutedFill]) -> EvaluationInput<'a> {
    EvaluationInput::new(window(end_ms), marks, fills, dec!(1000)).unwrap()
}

fn whole(data: &EvaluationInput<'_>) -> (Attribution, Decimal) {
    let split = attribute(data).unwrap();
    (split.total, split.final_position)
}

/// Marks 100, 102, 101. A buy of 2 at 99.5 at the first, fee 0.1, and a sell of 1 at
/// 101.5 at the last, fee 0.2.
fn round_trip() -> (Vec<Mark>, Vec<ExecutedFill>) {
    let marks = vec![
        mark(0, dec!(100)),
        mark(1000, dec!(102)),
        mark(2000, dec!(101)),
    ];
    let mut buy = fill(0, Action::Buy, dec!(99.5), dec!(2));
    buy.quote_fee = dec!(0.1);
    let mut sell = fill(2000, Action::Sell, dec!(101.5), dec!(1));
    sell.quote_fee = dec!(0.2);
    (marks, vec![buy, sell])
}

#[test]
fn test_attribute_round_trip_matches_the_hand_computed_components() {
    let (marks, fills) = round_trip();
    let (attribution, position) = whole(&input(3000, &marks, &fills));
    // Capture: 2 (100 - 99.5) + (-1)(101 - 101.5) = 1 + 0.5.
    assert_eq!(attribution.spread_capture, dec!(1.5));
    // Inventory: 2 (102 - 100) + 2 (101 - 102) = 4 - 2.
    assert_eq!(attribution.inventory_pnl, dec!(2));
    assert_eq!(attribution.quote_fees, dec!(0.3));
    assert_eq!(attribution.base_fees, dec!(0));
    // Directly: cash -199 + 101.5, position 1 at 101, fees 0.3.
    assert_eq!(attribution.net_pnl, dec!(3.2));
    assert_eq!(position, dec!(1));
}

#[test]
fn test_attribute_base_fee_leaves_the_position_before_later_moves() {
    // A buy of 1 at 99 with 0.01 base paid as a fee, then the mid moves 100 to 110.
    let marks = [mark(0, dec!(100)), mark(1000, dec!(110))];
    let mut buy = fill(0, Action::Buy, dec!(99), dec!(1));
    buy.base_fee = dec!(0.01);
    let fills = [buy];
    let (attribution, position) = whole(&input(2000, &marks, &fills));
    assert_eq!(position, dec!(0.99));
    // 0.99 * 10, not 1 * 10: the fee no longer moves with the price.
    assert_eq!(attribution.inventory_pnl, dec!(9.9));
    // The fee at the mark in force: 0.01 * 100.
    assert_eq!(attribution.base_fees, dec!(1));
    assert_eq!(attribution.spread_capture, dec!(1));
    // Directly: cash -99, position 0.99 at 110 = 108.9.
    assert_eq!(attribution.net_pnl, dec!(9.9));
}

#[test]
fn test_attribute_short_side_earns_when_the_price_falls() {
    let marks = [mark(0, dec!(100)), mark(1000, dec!(90))];
    let fills = [fill(0, Action::Sell, dec!(100.5), dec!(3))];
    let (attribution, position) = whole(&input(2000, &marks, &fills));
    assert_eq!(position, dec!(-3));
    // Capture: -3 (100 - 100.5). Inventory: -3 (90 - 100).
    assert_eq!(attribution.spread_capture, dec!(1.5));
    assert_eq!(attribution.inventory_pnl, dec!(30));
    assert_eq!(attribution.net_pnl, dec!(31.5));
}

#[test]
fn test_attribute_fill_between_marks_uses_the_last_mark_before_it() {
    let marks = [mark(0, dec!(100)), mark(1000, dec!(104))];
    let fills = [fill(500, Action::Buy, dec!(99), dec!(1))];
    let (attribution, _) = whole(&input(2000, &marks, &fills));
    // The mark in force at 500 is 100, and the position sees the whole move to 104.
    assert_eq!(attribution.spread_capture, dec!(1));
    assert_eq!(attribution.inventory_pnl, dec!(4));
}

#[test]
fn test_attribute_fill_at_tied_marks_uses_the_last_of_them() {
    let marks = [mark(1000, dec!(100)), mark(1000, dec!(102))];
    let fills = [fill(1000, Action::Buy, dec!(101), dec!(1))];
    let (attribution, _) = whole(&input(2000, &marks, &fills));
    assert_eq!(attribution.spread_capture, dec!(1));
    assert_eq!(attribution.inventory_pnl, dec!(0));
}

#[test]
fn test_attribute_prices_with_many_decimals_stay_exact() {
    let marks = [
        mark(0, dec!(65123.12345678)),
        mark(1000, dec!(65124.98765432)),
    ];
    let mut buy = fill(0, Action::Buy, dec!(65122.87654321), dec!(0.00012345));
    buy.quote_fee = dec!(0.000000081234);
    let fills = [buy];
    let (attribution, _) = whole(&input(2000, &marks, &fills));
    // 0.00012345 * 0.24691357 and 0.00012345 * 1.86419754, in full.
    assert_eq!(attribution.spread_capture, dec!(0.0000304814802165));
    assert_eq!(attribution.inventory_pnl, dec!(0.000230135186313));
    assert_eq!(attribution.quote_fees, dec!(0.000000081234));
}

#[test]
fn test_attribute_product_past_28_places_is_inexact_not_rounded() {
    let marks = [mark(0, dec!(1.000000000000000001))];
    let fills = [fill(0, Action::Buy, dec!(1), dec!(0.000000000000000000001))];
    let error = attribute(&input(1000, &marks, &fills)).unwrap_err();
    assert!(matches!(error, EvaluationError::Inexact { .. }), "{error}");
}

#[test]
fn test_attribute_sum_that_would_lose_a_small_term_is_inexact() {
    let marks = [mark(0, dec!(1))];
    let mut small = fill(0, Action::Buy, dec!(1), dec!(1));
    small.quote_fee = Decimal::new(1, 28);
    let mut large = fill(0, Action::Sell, dec!(1), dec!(1));
    large.quote_fee = dec!(10000000000);
    let fills = [small, large];
    let error = attribute(&input(1000, &marks, &fills)).unwrap_err();
    assert!(matches!(error, EvaluationError::Inexact { .. }), "{error}");
}

#[test]
fn test_attribute_no_fills_is_zero_and_hedge_costs_are_not_a_zero() {
    let marks = [mark(0, dec!(100)), mark(1000, dec!(90))];
    let (attribution, position) = whole(&input(2000, &marks, &[]));
    assert_eq!(attribution.net_pnl, dec!(0));
    assert_eq!(position, dec!(0));
    assert_eq!(
        attribution.hedge_costs,
        HedgeCosts::NotModelled {
            reason: HedgeNotModelledReason::SingleInstrument
        }
    );
}

#[test]
fn test_attribution_reconcile_refuses_the_smallest_difference() {
    let (marks, fills) = round_trip();
    let split = attribute(&input(3000, &marks, &fills)).unwrap();
    let books = LedgerTotals {
        net_pnl: dec!(3.2),
        position: dec!(1),
    };
    split.reconcile(books).unwrap();
    let off = LedgerTotals {
        net_pnl: dec!(3.2) + Decimal::new(1, 28),
        position: dec!(1),
    };
    assert!(matches!(
        split.reconcile(off),
        Err(EvaluationError::NotReconciled {
            what: "net pnl",
            ..
        })
    ));
    let moved = LedgerTotals {
        net_pnl: dec!(3.2),
        position: dec!(1.5),
    };
    assert!(matches!(
        split.reconcile(moved),
        Err(EvaluationError::NotReconciled {
            what: "final position",
            ..
        })
    ));
}

#[test]
fn test_input_window_is_half_open_and_refuses_what_is_outside() {
    let w = TimeWindow::new(1000, 2000).unwrap();
    let cap = dec!(1000);
    let at = |t| [mark(t, dec!(100))];
    EvaluationInput::new(w, &at(1000), &[], cap).unwrap();
    EvaluationInput::new(w, &at(1999), &[], cap).unwrap();
    for t in [999, 2000] {
        assert!(matches!(
            EvaluationInput::new(w, &at(t), &[], cap),
            Err(EvaluationError::InvalidInput { .. })
        ));
    }
    let marks = at(1000);
    let late = [fill(2000, Action::Buy, dec!(1), dec!(1))];
    assert!(matches!(
        EvaluationInput::new(w, &marks, &late, cap),
        Err(EvaluationError::InvalidInput { .. })
    ));
}

#[test]
fn test_input_refuses_out_of_order_records_and_invalid_values() {
    let w = window(10_000);
    let cap = dec!(1000);
    let ok = [mark(0, dec!(100))];
    let refused = |marks: &[Mark], fills: &[ExecutedFill], capital: Decimal| {
        matches!(
            EvaluationInput::new(w, marks, fills, capital),
            Err(EvaluationError::InvalidInput { .. })
        )
    };
    assert!(refused(&[], &[], cap));
    assert!(refused(&[mark(10, dec!(1)), mark(5, dec!(1))], &[], cap));
    assert!(refused(&[mark(0, dec!(0))], &[], cap));
    assert!(refused(&ok, &[], dec!(0)));
    assert!(refused(
        &ok,
        &[
            fill(10, Action::Buy, dec!(1), dec!(1)),
            fill(5, Action::Buy, dec!(1), dec!(1))
        ],
        cap
    ));
    assert!(refused(&ok, &[fill(0, Action::Buy, dec!(0), dec!(1))], cap));
    assert!(refused(&ok, &[fill(0, Action::Buy, dec!(1), dec!(0))], cap));
    assert!(refused(
        &ok,
        &[fill(0, Action::Other, dec!(1), dec!(1))],
        cap
    ));
    let late_marks = [mark(100, dec!(100))];
    assert_eq!(
        EvaluationInput::new(
            w,
            &late_marks,
            &[fill(50, Action::Buy, dec!(1), dec!(1))],
            cap
        )
        .unwrap_err(),
        EvaluationError::FillBeforeFirstMark { timestamp_ms: 50 }
    );
}

#[test]
fn test_attribute_daily_carries_inventory_into_each_day_and_keeps_the_calendar() {
    // Long 1 from day 0 to day 3. Day 1 has marks and no fill; day 2 has nothing.
    let marks = [
        mark(0, dec!(100)),
        mark(1000, dec!(101)),
        mark(DAY + 5, dec!(103)),
        mark(3 * DAY + 5, dec!(104)),
        mark(3 * DAY + 10, dec!(103)),
    ];
    let fills = [
        fill(0, Action::Buy, dec!(99), dec!(1)),
        fill(3 * DAY + 10, Action::Sell, dec!(103.5), dec!(1)),
    ];
    let data = input(4 * DAY, &marks, &fills);
    let (split, calendar) = attribute_daily(&data).unwrap();
    assert_eq!(calendar.len(), 4);
    assert_eq!(
        calendar.iter().map(|d| d.day).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    // Day 0: capture 1, inventory 101 - 100.
    assert_eq!(calendar[0].bucket.attribution.net_pnl, dec!(2));
    assert_eq!(calendar[0].bucket.fills, 1);
    // Day 1: no fill, and the move across midnight, 103 - 101, belongs to it.
    assert_eq!(calendar[1].bucket.attribution.inventory_pnl, dec!(2));
    assert_eq!(calendar[1].bucket.attribution.spread_capture, dec!(0));
    assert_eq!(calendar[1].bucket.fills, 0);
    assert_eq!(calendar[1].bucket.marks, 1);
    // Day 2: listed, zero, and marked as not observed.
    assert_eq!(calendar[2].bucket.attribution.net_pnl, dec!(0));
    assert_eq!(calendar[2].bucket.marks, 0);
    // Day 3: inventory (104 - 103) + (103 - 104) = 0, capture -1 (103 - 103.5).
    assert_eq!(calendar[3].bucket.attribution.inventory_pnl, dec!(0));
    assert_eq!(calendar[3].bucket.attribution.spread_capture, dec!(0.5));
    // The days sum exactly to the run: -99 + 103.5.
    assert_eq!(split.total.net_pnl, dec!(4.5));
    assert_eq!(split.total, whole(&data).0);
    let sum: Decimal = calendar.iter().map(|d| d.bucket.attribution.net_pnl).sum();
    assert_eq!(sum, split.total.net_pnl);
}

#[test]
fn test_attribute_by_label_is_called_in_time_order_and_buckets_sum_to_the_total() {
    let (marks, fills) = round_trip();
    let mut seen = Vec::new();
    let split = attribute_by(&input(3000, &marks, &fills), |m| {
        seen.push(m.timestamp_ms);
        m.mid > dec!(100)
    })
    .unwrap();
    assert_eq!(seen, [0, 1000, 2000]);
    // Calm: the first mark, with the buy. Stressed: the two moves and the sell.
    let calm = split.buckets[&false].attribution;
    let stressed = split.buckets[&true].attribution;
    assert_eq!(calm.net_pnl, dec!(0.9));
    assert_eq!(stressed.net_pnl, dec!(2.3));
    assert_eq!(calm.net_pnl + stressed.net_pnl, split.total.net_pnl);
}

#[test]
fn test_max_drawdown_matches_the_hand_computed_equity_path() {
    let (marks, fills) = round_trip();
    // Equity after the fills at each mark: 0.9, 4.9, 3.2.
    assert_eq!(
        max_drawdown(&input(3000, &marks, &fills)).unwrap(),
        dec!(1.7)
    );
    // A fall from the flat start counts: the first peak is zero.
    let marks = [mark(0, dec!(100)), mark(1000, dec!(98))];
    let fills = [fill(0, Action::Buy, dec!(100), dec!(1))];
    assert_eq!(max_drawdown(&input(3000, &marks, &fills)).unwrap(), dec!(2));
}

#[test]
fn test_inventory_mean_is_weighted_by_time_not_per_tick() {
    // Long 4 for 250 ms out of 1000, with many marks in the flat part.
    let marks = [
        mark(0, dec!(100)),
        mark(500, dec!(100)),
        mark(600, dec!(100)),
        mark(700, dec!(100)),
        mark(1000, dec!(100)),
    ];
    let fills = [
        fill(0, Action::Buy, dec!(100), dec!(4)),
        fill(250, Action::Sell, dec!(100), dec!(4)),
    ];
    let stats = inventory_stats(&input(2000, &marks, &fills)).unwrap();
    assert_eq!(stats.mean_abs, Some(dec!(1)));
    assert_eq!(stats.max_abs, dec!(4));
    assert_eq!(stats.final_position, dec!(0));
}

#[test]
fn test_inventory_mean_counts_a_position_held_to_the_last_mark() {
    // Flat to 500, then long 2 up to the last mark at 1000: 2 * 500 / 1000 = 1.
    let marks = [mark(0, dec!(100)), mark(1000, dec!(100))];
    let fills = [fill(500, Action::Buy, dec!(100), dec!(2))];
    let stats = inventory_stats(&input(2000, &marks, &fills)).unwrap();
    assert_eq!(stats.mean_abs, Some(dec!(1)));
    assert_eq!(stats.max_abs, dec!(2));
    assert_eq!(stats.final_position, dec!(2));
}

#[test]
fn test_inventory_mean_is_undefined_at_a_single_instant() {
    let marks = [mark(5, dec!(100))];
    let stats = inventory_stats(&input(10, &marks, &[])).unwrap();
    assert_eq!(stats.mean_abs, None);
}

#[test]
fn test_fill_quality_matches_the_hand_computed_formulas_per_side() {
    let marks = [
        mark(0, dec!(100)),
        mark(1000, dec!(102)),
        mark(2000, dec!(101)),
        mark(3000, dec!(99)),
    ];
    let fills = [
        fill(0, Action::Buy, dec!(99.5), dec!(2)),
        fill(2000, Action::Sell, dec!(101.5), dec!(1)),
    ];
    let spec = FillQualitySpec {
        horizon_ms: 1000,
        max_mark_delay_ms: 0,
    };
    let report = fill_quality(&input(4000, &marks, &fills), spec).unwrap();
    // Buy: M_f 100, M_H 102, P 99.5.
    assert_eq!(report.buys.signed_markout, Some(dec!(2.5)));
    assert_eq!(report.buys.realized_spread, Some(dec!(5)));
    assert_eq!(report.buys.effective_spread, Some(dec!(1)));
    assert_eq!(report.buys.price_impact, Some(dec!(-4)));
    // Sell: M_f 101, M_H 99, P 101.5. The mid fell after we sold, which is good.
    assert_eq!(report.sells.signed_markout, Some(dec!(2.5)));
    assert_eq!(report.sells.realized_spread, Some(dec!(5)));
    assert_eq!(report.sells.effective_spread, Some(dec!(1)));
    assert_eq!(report.sells.price_impact, Some(dec!(-4)));
    // Both, by size: (2 * 2.5 + 1 * 2.5) / 3.
    assert_eq!(report.all.signed_markout, Some(dec!(2.5)));
    assert_eq!(report.all.measured_volume, dec!(3));
}

#[test]
fn test_fill_quality_adverse_move_gives_a_negative_markout_and_positive_impact() {
    let marks = [mark(0, dec!(100)), mark(1000, dec!(97))];
    let fills = [fill(0, Action::Buy, dec!(99.5), dec!(1))];
    let spec = FillQualitySpec {
        horizon_ms: 1000,
        max_mark_delay_ms: 0,
    };
    let report = fill_quality(&input(2000, &marks, &fills), spec).unwrap();
    assert_eq!(report.all.signed_markout, Some(dec!(-2.5)));
    assert_eq!(report.all.realized_spread, Some(dec!(-5)));
    assert_eq!(report.all.effective_spread, Some(dec!(1)));
    assert_eq!(report.all.price_impact, Some(dec!(6)));
}

#[test]
fn test_fill_quality_uses_the_first_mark_at_or_after_the_horizon_within_the_delay() {
    let marks = [
        mark(0, dec!(100)),
        mark(900, dec!(150)),
        mark(1100, dec!(101)),
        mark(1200, dec!(160)),
    ];
    let fills = [fill(0, Action::Buy, dec!(100), dec!(1))];
    let data = input(2000, &marks, &fills);
    let measure = |max_mark_delay_ms| {
        fill_quality(
            &data,
            FillQualitySpec {
                horizon_ms: 1000,
                max_mark_delay_ms,
            },
        )
        .unwrap()
        .all
    };
    // Not the earlier 150, and not the later 160.
    assert_eq!(measure(100).signed_markout, Some(dec!(1)));
    // The mark at 1100 is too late, and an earlier one never stands in for it.
    let late = measure(99);
    assert_eq!(late.signed_markout, None);
    assert_eq!((late.fills, late.measured, late.excluded), (1, 0, 1));
}

#[test]
fn test_fill_quality_takes_the_last_of_marks_tied_at_the_horizon() {
    let marks = [
        mark(0, dec!(100)),
        mark(1000, dec!(100)),
        mark(1000, dec!(102)),
        mark(1500, dec!(90)),
    ];
    let fills = [fill(0, Action::Buy, dec!(100), dec!(1))];
    let spec = FillQualitySpec {
        horizon_ms: 1000,
        max_mark_delay_ms: 0,
    };
    let quality = fill_quality(&input(2000, &marks, &fills), spec).unwrap();
    // 102 - 100, not the 100 before it at the same time, nor the later 90.
    assert_eq!(quality.all.signed_markout, Some(dec!(2)));
}

#[test]
fn test_attribute_daily_counts_a_fill_on_a_day_without_marks_on_the_owner_of_its_mark() {
    let marks = [mark(0, dec!(100)), mark(2 * DAY, dec!(100))];
    let fills = [fill(DAY + 1, Action::Buy, dec!(100), dec!(1))];
    let (_, calendar) = attribute_daily(&input(3 * DAY, &marks, &fills)).unwrap();
    let count: Vec<(u64, u64)> = calendar
        .iter()
        .map(|d| (d.bucket.marks, d.bucket.fills))
        .collect();
    assert_eq!(count, [(1, 1), (0, 0), (1, 0)]);
}

#[test]
fn test_fill_quality_at_the_window_end_excludes_rather_than_reads_the_next_window() {
    // The validation window ends at 1000, and the holdout starts there with a
    // value that would change the answer. It cannot be passed in, and without it the
    // fill is excluded.
    let validation = TimeWindow::new(0, 1000).unwrap();
    let all_marks = [
        mark(0, dec!(100)),
        mark(900, dec!(100)),
        mark(1000, dec!(500)),
    ];
    let fills = [fill(900, Action::Buy, dec!(99), dec!(1))];
    assert!(matches!(
        EvaluationInput::new(validation, &all_marks, &fills, dec!(1000)),
        Err(EvaluationError::InvalidInput { .. })
    ));
    let data = EvaluationInput::new(validation, &all_marks[..2], &fills, dec!(1000)).unwrap();
    let spec = FillQualitySpec {
        horizon_ms: 100,
        max_mark_delay_ms: 10_000,
    };
    let report = fill_quality(&data, spec).unwrap();
    assert_eq!(report.all.excluded, 1);
    assert_eq!(report.all.signed_markout, None);
}

#[test]
fn test_fill_quality_refuses_a_zero_or_overflowing_horizon() {
    let marks = [mark(0, dec!(100))];
    let fills = [fill(1, Action::Buy, dec!(100), dec!(1))];
    let data = input(1000, &marks, &fills);
    for (horizon_ms, max_mark_delay_ms) in [(0, 0), (u64::MAX, 0), (u64::MAX - 2, 5)] {
        let spec = FillQualitySpec {
            horizon_ms,
            max_mark_delay_ms,
        };
        assert!(
            matches!(
                fill_quality(&data, spec),
                Err(EvaluationError::InvalidInput { .. })
            ),
            "{horizon_ms} {max_mark_delay_ms}"
        );
    }
}

#[test]
fn test_metric_table_declares_each_metric_and_values_carry_their_definition() {
    let (marks, fills) = round_trip();
    let spec = MetricsSpec {
        horizons_ms: vec![1000, 500],
        max_mark_delay_ms: 500,
    };
    let metrics = evaluate_window(&input(3000, &marks, &fills), &spec).unwrap();
    let values = metrics.values().unwrap();
    for id in MetricId::ALL {
        assert!(values.iter().any(|v| v.id == id), "{id:?} is not reported");
    }
    for value in &values {
        assert_eq!(value.definition, value.id.definition());
        assert_eq!(
            value.horizon_ms.is_some(),
            value.definition.horizon == MetricHorizon::AfterFill
                || value.id == MetricId::EffectiveSpread,
            "{:?}",
            value.id
        );
    }
    let read = |id, horizon| {
        values
            .iter()
            .find(|v| v.id == id && v.horizon_ms == horizon)
            .unwrap()
            .reading
    };
    assert_eq!(read(MetricId::NetPnl, None), Reading::Value(dec!(3.2)));
    assert_eq!(
        read(MetricId::NetReturnOnCapital, None),
        Reading::Value(dec!(0.0032))
    );
    assert_eq!(read(MetricId::Fills, None), Reading::Value(dec!(2)));
    assert_eq!(read(MetricId::Volume, None), Reading::Value(dec!(3)));
    assert_eq!(
        read(MetricId::HedgeCosts, None),
        Reading::NotModelled(HedgeNotModelledReason::SingleInstrument)
    );
    assert_eq!(
        read(MetricId::SignedMarkout, Some(1000)),
        Reading::Value(dec!(2.5))
    );
    // Each horizon is its own row.
    assert_eq!(
        values
            .iter()
            .filter(|v| v.id == MetricId::SignedMarkout)
            .count(),
        2
    );
}

#[test]
fn test_metric_values_with_no_fill_are_undefined_not_zero() {
    let marks = [mark(0, dec!(100)), mark(1000, dec!(100))];
    let spec = MetricsSpec {
        horizons_ms: vec![1000],
        max_mark_delay_ms: 0,
    };
    let values = evaluate_window(&input(2000, &marks, &[]), &spec)
        .unwrap()
        .values()
        .unwrap();
    let markout = values
        .iter()
        .find(|v| v.id == MetricId::SignedMarkout)
        .unwrap();
    assert_eq!(markout.reading, Reading::Undefined);
}

#[test]
fn test_metric_definitions_pin_units_and_signs() {
    let def = MetricId::SignedMarkout.definition();
    assert_eq!(def.unit, MetricUnit::QuotePerBase);
    assert_eq!(def.horizon, MetricHorizon::AfterFill);
    assert_eq!(
        MetricId::PriceImpact.definition().direction,
        market_maker_rs::backtest::evaluation::Direction::PositiveIsCost
    );
    assert_eq!(
        MetricId::NetReturnOnCapital
            .definition()
            .capital_denominator,
        market_maker_rs::backtest::evaluation::CapitalDenominator::DeclaredCapital
    );
}

#[test]
fn test_every_metric_declares_the_literal_definition() {
    use MetricId as M;
    use market_maker_rs::backtest::evaluation::{
        Aggregation as A, CapitalDenominator as C, Direction as D, MetricDefinition,
        MetricHorizon as H, MetricUnit as U,
    };
    let table = [
        (
            M::NetPnl,
            U::Quote,
            D::PositiveIsGain,
            H::Window,
            C::None,
            A::Sum,
        ),
        (
            M::SpreadCapture,
            U::Quote,
            D::PositiveIsGain,
            H::AtFill,
            C::None,
            A::Sum,
        ),
        (
            M::InventoryPnl,
            U::Quote,
            D::PositiveIsGain,
            H::Window,
            C::None,
            A::Sum,
        ),
        (
            M::QuoteFees,
            U::Quote,
            D::PositiveIsCost,
            H::Window,
            C::None,
            A::Sum,
        ),
        (
            M::BaseFees,
            U::Quote,
            D::PositiveIsCost,
            H::Window,
            C::None,
            A::Sum,
        ),
        (
            M::HedgeCosts,
            U::Quote,
            D::PositiveIsCost,
            H::Window,
            C::None,
            A::Sum,
        ),
        (
            M::SignedMarkout,
            U::QuotePerBase,
            D::PositiveIsGain,
            H::AfterFill,
            C::None,
            A::VolumeWeightedMean,
        ),
        (
            M::RealizedSpread,
            U::QuotePerBase,
            D::PositiveIsGain,
            H::AfterFill,
            C::None,
            A::VolumeWeightedMean,
        ),
        (
            M::EffectiveSpread,
            U::QuotePerBase,
            D::PositiveIsGain,
            H::AtFill,
            C::None,
            A::VolumeWeightedMean,
        ),
        (
            M::PriceImpact,
            U::QuotePerBase,
            D::PositiveIsCost,
            H::AfterFill,
            C::None,
            A::VolumeWeightedMean,
        ),
        (
            M::Fills,
            U::Count,
            D::Magnitude,
            H::Window,
            C::None,
            A::Count,
        ),
        (M::Volume, U::Base, D::Magnitude, H::Window, C::None, A::Sum),
        (
            M::MeanAbsInventory,
            U::Base,
            D::Magnitude,
            H::Window,
            C::None,
            A::TimeWeightedMean,
        ),
        (
            M::MaxAbsInventory,
            U::Base,
            D::Magnitude,
            H::Window,
            C::None,
            A::Maximum,
        ),
        (
            M::MaxDrawdown,
            U::Quote,
            D::MagnitudeIsCost,
            H::Window,
            C::None,
            A::Maximum,
        ),
        (
            M::NetReturnOnCapital,
            U::Ratio,
            D::PositiveIsGain,
            H::Window,
            C::DeclaredCapital,
            A::Quotient,
        ),
        (
            M::MaxDrawdownOnCapital,
            U::Ratio,
            D::MagnitudeIsCost,
            H::Window,
            C::DeclaredCapital,
            A::Quotient,
        ),
    ];
    assert_eq!(table.len(), M::ALL.len());
    for (row, id) in table.iter().zip(M::ALL) {
        assert_eq!(row.0, id, "the table follows MetricId::ALL");
        assert_eq!(
            id.definition(),
            MetricDefinition {
                unit: row.1,
                direction: row.2,
                horizon: row.3,
                capital_denominator: row.4,
                aggregation: row.5,
            },
            "{id:?}"
        );
    }
}

// ------------------------------------------------------------ the engine's books

/// Quotes one tick either side of the mid, always.
struct TwoSided {
    half_spread: Decimal,
    size: Decimal,
}

impl BacktestStrategy for TwoSided {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        let mid = tick.mid_price();
        Some(Quote {
            bid_price: mid - self.half_spread,
            bid_size: self.size,
            ask_price: mid + self.half_spread,
            ask_size: self.size,
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

/// A deterministic walk with moves larger than the quoted spread, over three days.
fn walk(n: u64) -> Vec<MarketTick> {
    let mut state = 12345_u64;
    let mut mid = dec!(1000);
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let step = Decimal::from((state >> 33) % 201) / dec!(100) - dec!(1);
            mid += step;
            let half = dec!(0.05);
            MarketTick::new(i * (3 * DAY / n), mid - half, dec!(5), mid + half, dec!(5))
        })
        .collect()
}

fn records(ticks: &[MarketTick], trades: &[SimulatedFill]) -> (Vec<Mark>, Vec<ExecutedFill>) {
    (
        ticks
            .iter()
            .map(|t| mark(t.timestamp, t.mid_price()))
            .collect(),
        trades.iter().map(ExecutedFill::from_simulated).collect(),
    )
}

fn check_against_books(config: BacktestConfig, latency: Option<SimLatencyConfig>) {
    let ticks = walk(2000);
    let strategy = TwoSided {
        half_spread: dec!(0.25),
        size: dec!(0.5),
    };
    let mut engine = BacktestEngine::new(
        config.with_record_trades(true),
        strategy,
        VecDataSource::new(ticks.clone()),
    );
    if let Some(latency) = latency {
        engine = engine.with_latency(latency).unwrap();
    }
    let result = engine.run();
    assert!(result.accounting.is_consistent());
    assert!(result.num_trades > 100, "only {} fills", result.num_trades);
    assert_ne!(result.final_position, dec!(0));

    let (marks, fills) = records(&ticks, &result.trades);
    let data = input(4 * DAY, &marks, &fills);
    let (split, calendar) = attribute_daily(&data).unwrap();
    split
        .reconcile(LedgerTotals {
            net_pnl: result.net_pnl,
            position: result.final_position,
        })
        .unwrap();
    assert_eq!(max_drawdown(&data).unwrap(), result.max_drawdown);
    let sum: Decimal = calendar.iter().map(|d| d.bucket.attribution.net_pnl).sum();
    assert_eq!(sum, result.net_pnl);
    let fills_by_day: u64 = calendar.iter().map(|d| d.bucket.fills).sum();
    assert_eq!(fills_by_day, result.num_trades);
    assert_eq!(calendar.iter().filter(|d| d.bucket.marks > 0).count(), 3);
}

fn base_config() -> BacktestConfig {
    BacktestConfig::new()
        .with_initial_capital(dec!(100000))
        .with_tick_size(dec!(0.01))
        .with_lot_size(dec!(0.01))
}

#[test]
fn test_attribution_reconciles_with_the_engine_with_quote_fees() {
    check_against_books(base_config().with_fee_rate(dec!(0.001)), None);
}

#[test]
fn test_attribution_reconciles_with_the_engine_with_a_rebate() {
    check_against_books(base_config().with_fee_rate(dec!(-0.0001)), None);
}

#[test]
fn test_attribution_reconciles_with_the_engine_with_base_fees() {
    check_against_books(
        base_config()
            .with_fee_rate(dec!(0.001))
            .with_fee_currency(market_maker_rs::backtest::engine::FeeCurrency::Base { scale: 8 }),
        None,
    );
}

#[test]
fn test_attribution_reconciles_with_the_engine_under_latency() {
    check_against_books(
        base_config().with_fee_rate(dec!(0.001)),
        Some(
            SimLatencyConfig::new()
                .with_observation_ms(50)
                .with_submit_ms(70)
                .with_cancel_ms(30)
                .with_ack_ms(20)
                .with_jitter(40, 7),
        ),
    );
}

#[cfg(feature = "margin")]
#[test]
fn test_attribution_reconciles_with_the_margin_ledger() {
    use margin_rs::ledger::{Ledger, LedgerConfig};
    use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};
    use market_maker_rs::backtest::MarginLedgerAccountant;

    let (base, quote) = (Asset::new("BASE").unwrap(), Asset::new("QUOTE").unwrap());
    let instrument = InstrumentId::new("BASE-QUOTE").unwrap();
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(base.clone(), 2).unwrap();
    ledger.register_asset(quote.clone(), 7).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(instrument.clone(), base.clone(), quote.clone(), 2, 2).unwrap(),
        )
        .unwrap();
    for (id, asset) in [("base", &base), ("quote", &quote)] {
        ledger
            .deposit(TransferId::new(id).unwrap(), asset.clone(), dec!(10000000))
            .unwrap();
    }
    let ticks = walk(2000);
    let mut engine = BacktestEngine::new(
        base_config()
            .with_fee_rate(dec!(0.001))
            .with_record_trades(true),
        TwoSided {
            half_spread: dec!(0.25),
            size: dec!(0.5),
        },
        VecDataSource::new(ticks.clone()),
    )
    .with_accountant(MarginLedgerAccountant::new(ledger, instrument).unwrap());
    let result = engine.run();
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert!(result.num_trades > 100, "only {} fills", result.num_trades);
    let (marks, fills) = records(&ticks, &result.trades);
    attribute(&input(4 * DAY, &marks, &fills))
        .unwrap()
        .reconcile(LedgerTotals {
            net_pnl: result.net_pnl,
            position: result.final_position,
        })
        .unwrap();
}

#[test]
fn test_attribution_reconciles_with_the_engine_with_slippage() {
    check_against_books(
        base_config().with_slippage(market_maker_rs::backtest::SlippageModel::Fixed(dec!(0.01))),
        None,
    );
}

#[test]
fn test_attribution_reconciles_with_the_engine_with_a_third_asset_fee() {
    check_against_books(
        base_config().with_fee_rate(dec!(0.001)).with_fee_currency(
            market_maker_rs::backtest::engine::FeeCurrency::Other {
                asset: "FEE".to_string(),
                quote_per_unit: dec!(4),
                scale: 6,
            },
        ),
        None,
    );
}

#[cfg(feature = "margin")]
#[test]
fn test_attribution_reconciles_with_base_fees_on_the_margin_ledger() {
    use margin_rs::ledger::{Ledger, LedgerConfig};
    use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};
    use market_maker_rs::backtest::MarginLedgerAccountant;

    let (base, quote) = (Asset::new("BASE").unwrap(), Asset::new("QUOTE").unwrap());
    let instrument = InstrumentId::new("BASE-QUOTE").unwrap();
    let mut ledger = Ledger::new(LedgerConfig::default());
    ledger.register_asset(base.clone(), 8).unwrap();
    ledger.register_asset(quote.clone(), 8).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(instrument.clone(), base.clone(), quote.clone(), 2, 2).unwrap(),
        )
        .unwrap();
    ledger
        .deposit(TransferId::new("quote").unwrap(), quote, dec!(10000000))
        .unwrap();
    let ticks = walk(2000);
    let result = BacktestEngine::new(
        base_config()
            .with_fee_rate(dec!(0.001))
            .with_fee_currency(market_maker_rs::backtest::engine::FeeCurrency::Base { scale: 8 }),
        TwoSided {
            half_spread: dec!(0.25),
            size: dec!(0.5),
        },
        VecDataSource::new(ticks.clone()),
    )
    .with_accountant(MarginLedgerAccountant::new(ledger, instrument).unwrap())
    .run();
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert!(result.trades.iter().any(|t| !t.base_fee.is_zero()));
    let (marks, fills) = records(&ticks, &result.trades);
    attribute(&input(4 * DAY, &marks, &fills))
        .unwrap()
        .reconcile(LedgerTotals {
            net_pnl: result.net_pnl,
            position: result.final_position,
        })
        .unwrap();
}

#[cfg(feature = "instruments")]
#[test]
fn test_check_contract_refuses_what_position_times_mark_cannot_value() {
    use market_maker_rs::backtest::evaluation::check_contract;
    use market_maker_rs::instruments::{
        AssetCode, ContractKind, ContractSize, InstrumentDraft, InstrumentId, InstrumentSpec,
        Margining, TickSchedule,
    };
    let asset = |code| AssetCode::new(code).unwrap();
    let spec = |margining, settlement, size| {
        InstrumentSpec::new(InstrumentDraft {
            id: InstrumentId::new("BTC-PERP").unwrap(),
            kind: ContractKind::Perpetual,
            margining,
            underlying: asset("BTC"),
            price_asset: asset("USDT"),
            settlement_asset: asset(settlement),
            size,
            price_ticks: TickSchedule::fixed(dec!(0.25)).unwrap(),
            lot: dec!(1),
            expiry_ms: None,
            option: None,
        })
        .unwrap()
    };
    let base = |amount| ContractSize::base(amount, asset("BTC")).unwrap();
    assert!(check_contract(&spec(Margining::Linear, "USDT", base(dec!(1)))).is_ok());
    let refused = [
        spec(Margining::Linear, "USDT", base(dec!(0.5))),
        spec(
            Margining::Inverse,
            "BTC",
            ContractSize::quote_value(dec!(10), asset("USDT")).unwrap(),
        ),
    ];
    for contract in &refused {
        assert!(matches!(
            check_contract(contract),
            Err(EvaluationError::UnsupportedContract { .. })
        ));
    }
}
