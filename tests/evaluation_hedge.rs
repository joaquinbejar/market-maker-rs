//! The attribution of a hedge in a second instrument (#266): its terms against its own
//! marks, its buckets, its direct check and the reconciliation of both lanes, on
//! hand-computed records. All data here is synthetic.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    EvaluationError, EvaluationInput, ExecutedFill, FundingPayment, HedgeAttribution, HedgeCosts,
    HedgeNotModelledReason, LedgerTotals, Mark, MetricId, MetricsSpec, Reading, attribute,
    attribute_by, attribute_daily, evaluate_window,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::strategy::walk_forward::TimeWindow;

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

const WINDOW: TimeWindow = TimeWindow {
    start_ms: 0,
    end_ms: 2000,
};

// Quoted: marks at 0 (100) and 1000 (101), long 1 bought at 99.9 at 100 ms.
fn quoted() -> (Vec<Mark>, Vec<ExecutedFill>) {
    (
        vec![mark(0, dec!(100)), mark(1000, dec!(101))],
        vec![fill(100, Action::Buy, dec!(99.9), dec!(1))],
    )
}

// Hedge: marks at 500 (50) and 1500 (51); sells 2 at 49.9 at 600 ms with a fee of 0.01;
// pays 0.05 of funding at 1200 ms.
//   spread    -2 * (50 - 49.9) = -0.2
//   inventory -2 * (51 - 50)   = -2
//   net       -0.2 - 2 - 0.01 - 0.05 = -2.26, costs 0.01 + 0.2 + 0.05 = 0.26
fn hedge() -> (Vec<Mark>, Vec<ExecutedFill>, Vec<FundingPayment>) {
    let mut sell = fill(600, Action::Sell, dec!(49.9), dec!(2));
    sell.quote_fee = dec!(0.01);
    (
        vec![mark(500, dec!(50)), mark(1500, dec!(51))],
        vec![sell],
        vec![FundingPayment {
            timestamp_ms: 1200,
            amount: dec!(-0.05),
        }],
    )
}

fn attributed(costs: HedgeCosts) -> HedgeAttribution {
    match costs {
        HedgeCosts::Attributed(hedge) => hedge,
        other => panic!("expected an attributed hedge, got {other:?}"),
    }
}

#[test]
fn test_the_hedge_is_attributed_against_its_own_marks_and_apart_from_the_net() {
    let (marks, fills) = quoted();
    let (h_marks, h_fills, funding) = hedge();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &funding)
        .unwrap();
    let split = attribute(&input).unwrap();
    // The quoted lane alone: 0.1 of spread and 1 of inventory.
    assert_eq!(split.total.net_pnl, dec!(1.1));
    let hedge = attributed(split.total.hedge_costs);
    assert_eq!(hedge.spread_capture, dec!(-0.2));
    assert_eq!(hedge.inventory_pnl, dec!(-2));
    assert_eq!(hedge.fees, dec!(0.01));
    assert_eq!(hedge.funding, dec!(-0.05));
    assert_eq!(hedge.net_pnl, dec!(-2.26));
    assert_eq!(hedge.costs().unwrap(), dec!(0.26));
    assert_eq!(split.hedge_final_position, Some(dec!(-2)));
    split
        .reconcile_lanes(
            LedgerTotals {
                net_pnl: dec!(1.1),
                position: dec!(1),
            },
            LedgerTotals {
                net_pnl: dec!(-2.26),
                position: dec!(-2),
            },
        )
        .unwrap();
}

#[test]
fn test_each_hedge_term_falls_in_the_quoted_bucket_in_force_and_the_buckets_sum() {
    let (marks, fills) = quoted();
    let (h_marks, h_fills, funding) = hedge();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &funding)
        .unwrap();
    let split = attribute_by(&input, |m| m.timestamp_ms).unwrap();
    // The fill at 600 is in the bucket of the quoted mark at 0; the funding at 1200 and
    // the move at 1500 in that of the mark at 1000.
    let first = attributed(split.buckets[&0].attribution.hedge_costs);
    assert_eq!(
        (
            first.spread_capture,
            first.inventory_pnl,
            first.fees,
            first.funding
        ),
        (dec!(-0.2), dec!(0), dec!(0.01), dec!(0))
    );
    assert_eq!(first.net_pnl, dec!(-0.21));
    let second = attributed(split.buckets[&1000].attribution.hedge_costs);
    assert_eq!(
        (
            second.spread_capture,
            second.inventory_pnl,
            second.fees,
            second.funding
        ),
        (dec!(0), dec!(-2), dec!(0), dec!(-0.05))
    );
    assert_eq!(second.net_pnl, dec!(-2.05));
    assert_eq!(attributed(split.total.hedge_costs).net_pnl, dec!(-2.26));
}

#[test]
fn test_a_hedge_record_before_the_first_quoted_mark_is_in_the_first_bucket() {
    let marks = vec![mark(100, dec!(100)), mark(1000, dec!(100))];
    let h_marks = vec![mark(10, dec!(50)), mark(50, dec!(52))];
    let mut buy = fill(20, Action::Buy, dec!(50.5), dec!(1));
    buy.quote_fee = dec!(0.02);
    let h_fills = vec![buy];
    let funding = vec![FundingPayment {
        timestamp_ms: 30,
        amount: dec!(0.03),
    }];
    let input = EvaluationInput::new(WINDOW, &marks, &[], dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &funding)
        .unwrap();
    let split = attribute_by(&input, |m| m.timestamp_ms).unwrap();
    // spread 1 * (50 - 50.5) = -0.5, inventory 1 * (52 - 50) = 2, fee 0.02, funding 0.03.
    let first = attributed(split.buckets[&100].attribution.hedge_costs);
    assert_eq!(first.net_pnl, dec!(1.51));
    assert_eq!(
        attributed(split.buckets[&1000].attribution.hedge_costs).net_pnl,
        dec!(0)
    );
}

#[test]
fn test_every_day_holds_a_hedge_even_with_no_mark() {
    let marks = vec![mark(0, dec!(100)), mark(2 * 86_400_000, dec!(100))];
    let h_marks = vec![mark(5, dec!(50))];
    let window = TimeWindow {
        start_ms: 0,
        end_ms: 2 * 86_400_000 + 1,
    };
    let input = EvaluationInput::new(window, &marks, &[], dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &[], &[])
        .unwrap();
    let (split, days) = attribute_daily(&input).unwrap();
    assert_eq!(days.len(), 3);
    assert_eq!(days[1].bucket.marks, 0);
    for day in &days {
        assert_eq!(
            attributed(day.bucket.attribution.hedge_costs).net_pnl,
            dec!(0)
        );
    }
    assert_eq!(attributed(split.total.hedge_costs).net_pnl, dec!(0));
}

#[test]
fn test_one_instrument_reports_no_hedge_and_reconciles_as_before() {
    let (marks, fills) = quoted();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1)).unwrap();
    let split = attribute(&input).unwrap();
    assert_eq!(
        split.total.hedge_costs,
        HedgeCosts::NotModelled {
            reason: HedgeNotModelledReason::SingleInstrument
        }
    );
    assert_eq!(split.hedge_final_position, None);
    let books = LedgerTotals {
        net_pnl: dec!(1.1),
        position: dec!(1),
    };
    split.reconcile(books).unwrap();
    assert!(matches!(
        split.reconcile_lanes(books, books),
        Err(EvaluationError::InvalidInput { .. })
    ));
}

#[test]
fn test_reconcile_lanes_catches_either_lane_by_any_amount() {
    let (marks, fills) = quoted();
    let (h_marks, h_fills, funding) = hedge();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &funding)
        .unwrap();
    let split = attribute(&input).unwrap();
    let quoted = LedgerTotals {
        net_pnl: dec!(1.1),
        position: dec!(1),
    };
    let hedge = LedgerTotals {
        net_pnl: dec!(-2.26),
        position: dec!(-2),
    };
    let what = |result: Result<(), EvaluationError>| match result {
        Err(EvaluationError::NotReconciled { what, .. }) => what,
        other => panic!("expected a mismatch, got {other:?}"),
    };
    let off = LedgerTotals {
        net_pnl: dec!(-2.2600000001),
        ..hedge
    };
    assert_eq!(what(split.reconcile_lanes(quoted, off)), "hedge net pnl");
    let moved = LedgerTotals {
        position: dec!(-1),
        ..hedge
    };
    assert_eq!(
        what(split.reconcile_lanes(quoted, moved)),
        "hedge final position"
    );
    let quoted_off = LedgerTotals {
        net_pnl: dec!(1.1000000001),
        ..quoted
    };
    assert_eq!(what(split.reconcile_lanes(quoted_off, hedge)), "net pnl");
}

#[test]
fn test_the_hedge_costs_metric_reads_the_attributed_costs() {
    let (marks, fills) = quoted();
    let (h_marks, h_fills, funding) = hedge();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &funding)
        .unwrap();
    let spec = MetricsSpec {
        horizons_ms: vec![],
        max_mark_delay_ms: 0,
    };
    let values = evaluate_window(&input, &spec).unwrap().values().unwrap();
    let costs = values
        .iter()
        .find(|v| v.id == MetricId::HedgeCosts)
        .unwrap();
    assert_eq!(costs.reading, Reading::Value(dec!(0.26)));
}

#[test]
fn test_hedge_records_are_checked() {
    let (marks, fills) = quoted();
    let input = EvaluationInput::new(WINDOW, &marks, &fills, dec!(1)).unwrap();
    let h_marks = vec![mark(500, dec!(50))];
    // A fill before the hedge's first mark.
    let early = vec![fill(400, Action::Buy, dec!(50), dec!(1))];
    assert!(matches!(
        input.with_hedge(&h_marks, &early, &[]),
        Err(EvaluationError::FillBeforeFirstMark { timestamp_ms: 400 })
    ));
    // A base fee.
    let mut based = fill(600, Action::Buy, dec!(50), dec!(1));
    based.base_fee = dec!(0.001);
    assert!(matches!(
        input.with_hedge(&h_marks, &[based], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
    // Funding out of order, and outside the window.
    let unordered = vec![
        FundingPayment {
            timestamp_ms: 900,
            amount: dec!(1),
        },
        FundingPayment {
            timestamp_ms: 800,
            amount: dec!(1),
        },
    ];
    assert!(matches!(
        input.with_hedge(&h_marks, &[], &unordered),
        Err(EvaluationError::InvalidInput { .. })
    ));
    let outside = vec![FundingPayment {
        timestamp_ms: 2000,
        amount: dec!(1),
    }];
    assert!(matches!(
        input.with_hedge(&h_marks, &[], &outside),
        Err(EvaluationError::InvalidInput { .. })
    ));
    // A mark that is not positive, and one outside the window.
    let zero = vec![mark(500, dec!(0))];
    assert!(matches!(
        input.with_hedge(&zero, &[], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
    let late = vec![mark(2500, dec!(50))];
    assert!(matches!(
        input.with_hedge(&late, &[], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
}
