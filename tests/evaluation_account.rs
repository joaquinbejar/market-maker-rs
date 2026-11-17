//! Accounts of many lanes: hand computed fixtures for each new metric. Every
//! expected value is worked out in a comment.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    AccountBooks, AccountFlow, AccountInput, EvaluationError, EvaluationInput, ExecutedFill,
    FlowKind, FundingPayment, LaneFill, LaneFunding, LaneKind, LaneMark, LaneQualitySpec,
    LaneRecords, LaneRole, LaneSequence, LaneTransfer, LedgerTotals, Liquidity, Mark, MetricId,
    Opening, OrderEnd, OrderRecord, Reading, Settlement, TimeUnit, Window, attribute,
    attribute_account, lane_fill_quality,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::strategy::walk_forward::TimeWindow;

fn mark(time: u64, price: Decimal) -> LaneMark {
    LaneMark { time, price }
}

fn fill(time: u64, side: Action, price: Decimal, quantity: Decimal) -> LaneFill {
    LaneFill {
        time,
        side,
        price,
        quantity,
        quote_fee: Decimal::ZERO,
        base_fee: Decimal::ZERO,
        liquidity: Liquidity::Maker,
        excluded_from_quality: false,
        order_id: None,
    }
}

fn ns(start: u64, end: u64) -> Window {
    Window::new(start, end, TimeUnit::Nanoseconds).unwrap()
}

fn value(values: &[market_maker_rs::backtest::evaluation::MetricValue], id: MetricId) -> Reading {
    values.iter().find(|v| v.id == id).unwrap().reading
}

/// A perpetual of multiplier 10 on a nanosecond clock. Marks 50 at 0 ns, 51 at
/// 1 ns, 49 at 5 ns. A maker buy of 3 at 49.8 at 0 ns, fee 0.5, and a taker sell
/// of 1 at 51.2 at 1 ns, fee 0.3, after the mark at 1 ns. Funding of -1.5 at 3 ns.
fn perpetual() -> (Vec<LaneMark>, Vec<LaneFill>, Vec<LaneFunding>) {
    let marks = vec![mark(0, dec!(50)), mark(1, dec!(51)), mark(5, dec!(49))];
    let mut buy = fill(0, Action::Buy, dec!(49.8), dec!(3));
    buy.quote_fee = dec!(0.5);
    let mut sell = fill(1, Action::Sell, dec!(51.2), dec!(1));
    sell.quote_fee = dec!(0.3);
    sell.liquidity = Liquidity::Taker;
    let funding = vec![LaneFunding {
        time: 3,
        amount: dec!(-1.5),
    }];
    (marks, vec![buy, sell], funding)
}

#[test]
fn test_a_quoted_perpetual_with_a_multiplier_and_funding() {
    let (marks, fills, funding) = perpetual();
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(10))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_funding(&funding);
    let input = AccountInput::new(ns(0, 10_000), vec![lane], &[]).unwrap();
    let account = attribute_account(&input).unwrap();
    let a = account.lanes[0].attribution;
    // Capture: 3 * 10 (50 - 49.8) + (-1) * 10 (51 - 51.2) = 6 + 2. The sell is at
    // 1 ns, after the mark of 51 at 1 ns; a millisecond clock would merge both
    // fills and both first marks into one instant.
    assert_eq!(a.spread_capture, dec!(8));
    // Inventory: 3 * 10 (51 - 50) + 2 * 10 (49 - 51) = 30 - 40.
    assert_eq!(a.inventory_pnl, dec!(-10));
    assert_eq!(a.quote_fees, dec!(0.8));
    assert_eq!(a.funding, dec!(-1.5));
    // 8 - 10 - 0.8 - 1.5. Directly: cash -1494 + 512, 2 * 10 * 49 = 980.
    assert_eq!(a.net_pnl, dec!(-4.3));
    assert_eq!(account.lanes[0].final_position, dec!(2));
    assert_eq!(account.quoted.attribution, a);
    assert_eq!(account.hedge.lanes, 0);
    assert_eq!(account.net_pnl, dec!(-4.3));
    let values = account.values(None).unwrap();
    assert_eq!(
        value(&values, MetricId::Funding),
        Reading::Value(dec!(-1.5))
    );
    assert_eq!(
        value(&values, MetricId::ReconciliationResidual),
        Reading::Undefined
    );
}

#[test]
fn test_consecutive_windows_with_an_opening_add_up_to_the_whole() {
    let (marks, fills, funding) = perpetual();
    // First window [0, 3): marks 50 and 51, both fills.
    let first = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(10))
        .with_marks(&marks[..2])
        .with_fills(&fills);
    let first = attribute_account(&AccountInput::new(ns(0, 3), vec![first], &[]).unwrap()).unwrap();
    // 8 + 30 - 0.8.
    assert_eq!(first.net_pnl, dec!(37.2));
    // Second window [3, 10_000): opening 2 at 51, the funding and the mark of 49.
    let second = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(10))
        .with_opening(Opening {
            position: first.lanes[0].final_position,
            price: dec!(51),
        })
        .with_marks(&marks[2..])
        .with_funding(&funding);
    let second =
        attribute_account(&AccountInput::new(ns(3, 10_000), vec![second], &[]).unwrap()).unwrap();
    // 2 * 10 (49 - 51) - 1.5.
    assert_eq!(second.net_pnl, dec!(-41.5));
    assert_eq!(second.lanes[0].opening_position, dec!(2));
    assert_eq!(first.net_pnl + second.net_pnl, dec!(-4.3));
}

#[test]
fn test_an_option_held_to_expiry_settles_with_its_rounding() {
    // An option of multiplier 100: a buy of 2 at a premium of 1.5 under a mark of
    // 1.6, a mark of exactly zero, then a payoff of 2.345 paid as 469.01.
    let marks = [mark(0, dec!(1.6)), mark(10, dec!(0))];
    let fills = [fill(0, Action::Buy, dec!(1.5), dec!(2))];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Option, dec!(100))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_settlement(Settlement {
            time: 30,
            price: dec!(2.345),
            cash: dec!(469.01),
        });
    let input = AccountInput::new(ns(0, 100), vec![lane], &[]).unwrap();
    let account = attribute_account(&input).unwrap();
    let outcome = account.lanes[0];
    let a = outcome.attribution;
    // 2 * 100 (1.6 - 1.5).
    assert_eq!(a.spread_capture, dec!(20));
    // 200 (0 - 1.6) + 200 (2.345 - 0) = -320 + 469.
    assert_eq!(a.inventory_pnl, dec!(149));
    // 469.01 - 200 * 2.345.
    assert_eq!(a.settlement_rounding, dec!(0.01));
    // Directly: -300 + 469.01, flat after the settlement.
    assert_eq!(a.net_pnl, dec!(169.01));
    assert_eq!(outcome.final_position, dec!(0));
    assert!(outcome.settled);
    let values = account.values(None).unwrap();
    assert_eq!(
        value(&values, MetricId::SettlementRounding),
        Reading::Value(dec!(0.01))
    );
}

#[test]
fn test_an_auto_deleveraging_close_and_account_flows() {
    // Opening short of 4 at 100, no mark until 5 ns. A liquidation buy of 1 at
    // 100.2 at 2 ns, against the opening price; mark 102 at 5 ns; an auto
    // deleveraging close, a buy of 3 at 101 at 7 ns; mark 105 at 9 ns.
    let marks = [mark(5, dec!(102)), mark(9, dec!(105))];
    let mut liquidation = fill(2, Action::Buy, dec!(100.2), dec!(1));
    liquidation.liquidity = Liquidity::Taker;
    liquidation.excluded_from_quality = true;
    let fills = [liquidation];
    let transfers = [LaneTransfer {
        time: 7,
        side: Action::Buy,
        quantity: dec!(3),
        cash: dec!(-303),
    }];
    let flows = [
        AccountFlow {
            time: 8,
            kind: FlowKind::AutoDeleveragingTopUp,
            amount: dec!(2),
        },
        AccountFlow {
            time: 8,
            kind: FlowKind::InsuranceFundDraw,
            amount: dec!(1.25),
        },
    ];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
        .with_opening(Opening {
            position: dec!(-4),
            price: dec!(100),
        })
        .with_marks(&marks)
        .with_fills(&fills)
        .with_transfers(&transfers);
    let input = AccountInput::new(ns(0, 10), vec![lane], &flows).unwrap();
    let account = attribute_account(&input).unwrap();
    let a = account.lanes[0].attribution;
    // 1 (100 - 100.2).
    assert_eq!(a.spread_capture, dec!(-0.2));
    // -3 (102 - 100), then flat.
    assert_eq!(a.inventory_pnl, dec!(-6));
    // 3 (102 - 101).
    assert_eq!(a.transfer_edge, dec!(3));
    // Directly: cash -100.2 - 303, less the opening value -400.
    assert_eq!(a.net_pnl, dec!(-3.2));
    assert_eq!(account.external_flows, dec!(3.25));
    assert_eq!(account.net_pnl, dec!(0.05));
    let values = account.values(None).unwrap();
    assert_eq!(
        value(&values, MetricId::TransferEdge),
        Reading::Value(dec!(3))
    );
    assert_eq!(
        value(&values, MetricId::ExternalFlows),
        Reading::Value(dec!(3.25))
    );
    // The liquidation is left out of fill quality.
    let spec = LaneQualitySpec {
        horizon: 1,
        max_mark_delay: 10,
    };
    let quality = lane_fill_quality(&input, 0, spec).unwrap();
    assert_eq!(quality.flagged, 1);
    assert_eq!(quality.all.fills, 0);
}

/// Multiplier 2, marks 10 at 0, 11 at 100, 9 at 200. A maker buy of 3 at 9.9 at
/// 0 for order 1, a taker sell of 2 at 11.1 at 100 for order 2.
fn ordered() -> (Vec<LaneMark>, Vec<LaneFill>, Vec<OrderRecord>) {
    let marks = vec![mark(0, dec!(10)), mark(100, dec!(11)), mark(200, dec!(9))];
    let mut buy = fill(0, Action::Buy, dec!(9.9), dec!(3));
    buy.order_id = Some(1);
    let mut sell = fill(100, Action::Sell, dec!(11.1), dec!(2));
    sell.order_id = Some(2);
    sell.liquidity = Liquidity::Taker;
    let order = |id, accepted_quantity, accepted_replaces, end_state, liquidation| OrderRecord {
        id,
        accepted_quantity,
        replace_increase: Decimal::ZERO,
        accepted_replaces,
        end_state,
        liquidation,
    };
    let orders = vec![
        order(1, dec!(5), 2, OrderEnd::Cancelled, false),
        order(2, dec!(4), 1, OrderEnd::Cancelled, false),
        order(3, dec!(0.5), 0, OrderEnd::Expired, false),
        order(4, dec!(0.5), 0, OrderEnd::Open, false),
        order(5, dec!(1), 0, OrderEnd::Cancelled, true),
    ];
    (marks, vec![buy, sell], orders)
}

#[test]
fn test_fill_ratios_by_count_and_by_quantity() {
    let (marks, fills, orders) = ordered();
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(2))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_orders(&orders);
    let input = AccountInput::new(ns(0, 1000), vec![lane], &[]).unwrap();
    let account = attribute_account(&input).unwrap();
    let stats = account.lanes[0].orders;
    // Orders 1 to 4; order 5 is a liquidation.
    assert_eq!(stats.orders, 4);
    assert_eq!(stats.filled_orders, 2);
    // In units of the underlying, multiplier 2: 10 and 5 contracts.
    assert_eq!(stats.accepted_quantity, dec!(20));
    assert_eq!(stats.filled_quantity, dec!(10));
    assert_eq!(stats.accepted_replaces, 3);
    assert_eq!(stats.liquidation_orders, 1);
    // End states of orders 1 to 4, the open one included.
    assert_eq!(stats.ended_filled, 0);
    assert_eq!(stats.ended_cancelled, 2);
    assert_eq!(stats.ended_expired, 1);
    assert_eq!(stats.ended_open, 1);
    // Volume: (3 + 2) contracts of 2 units.
    assert_eq!(account.lanes[0].volume, dec!(10));
    let values = account.lanes[0].values();
    // 2 / 4 and 10 / 20.
    assert_eq!(
        value(&values, MetricId::FillRatioByCount),
        Reading::Value(dec!(0.5))
    );
    assert_eq!(
        value(&values, MetricId::FillRatioByQuantity),
        Reading::Value(dec!(0.5))
    );
    assert_eq!(value(&values, MetricId::Orders), Reading::Value(dec!(4)));
    assert_eq!(
        value(&values, MetricId::FilledOrders),
        Reading::Value(dec!(2))
    );
    assert_eq!(
        value(&values, MetricId::AcceptedQuantity),
        Reading::Value(dec!(20))
    );
    assert_eq!(
        value(&values, MetricId::FilledQuantity),
        Reading::Value(dec!(10))
    );
    // No order, no ratio.
    let bare = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1));
    let bare = attribute_account(&AccountInput::new(ns(0, 10), vec![bare], &[]).unwrap()).unwrap();
    assert_eq!(
        value(&bare.lanes[0].values(), MetricId::FillRatioByCount),
        Reading::Undefined
    );
}

#[test]
fn test_fill_quality_is_split_by_maker_and_taker() {
    let (marks, fills, orders) = ordered();
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(2))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_orders(&orders);
    let input = AccountInput::new(ns(0, 1000), vec![lane], &[]).unwrap();
    let spec = LaneQualitySpec {
        horizon: 100,
        max_mark_delay: 0,
    };
    let q = lane_fill_quality(&input, 0, spec).unwrap();
    // Maker buy: 6 units, markout 11 - 9.9, effective 2 (10 - 9.9).
    assert_eq!(q.maker.measured_volume, dec!(6));
    assert_eq!(q.maker.signed_markout, Some(dec!(1.1)));
    assert_eq!(q.maker.effective_spread, Some(dec!(0.2)));
    // Taker sell: 4 units, markout -(9 - 11.1), effective 2 (11.1 - 11).
    assert_eq!(q.taker.measured_volume, dec!(4));
    assert_eq!(q.taker.signed_markout, Some(dec!(2.1)));
    assert_eq!(q.taker.effective_spread, Some(dec!(0.2)));
    // All: (6 * 1.1 + 4 * 2.1) / 10.
    assert_eq!(q.all.signed_markout, Some(dec!(1.5)));
    assert_eq!(q.buys, q.maker);
    assert_eq!(q.sells, q.taker);
    assert_eq!(q.flagged, 0);
}

#[test]
fn test_many_lanes_total_per_role_and_reconcile_per_lane() {
    let (p_marks, p_fills, p_funding) = perpetual();
    let o_marks = [mark(0, dec!(1.6)), mark(10, dec!(0))];
    let o_fills = [fill(0, Action::Buy, dec!(1.5), dec!(2))];
    let h_marks = [mark(0, dec!(100)), mark(5, dec!(98))];
    let mut h_fill = fill(0, Action::Sell, dec!(99.9), dec!(1));
    h_fill.quote_fee = dec!(0.05);
    let h_fills = [h_fill];
    let lanes = vec![
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(10))
            .with_marks(&p_marks)
            .with_fills(&p_fills)
            .with_funding(&p_funding),
        LaneRecords::new(LaneRole::Quoted, LaneKind::Option, dec!(100))
            .with_marks(&o_marks)
            .with_fills(&o_fills),
        LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, dec!(1))
            .with_marks(&h_marks)
            .with_fills(&h_fills),
    ];
    let flows = [AccountFlow {
        time: 9,
        kind: FlowKind::Other,
        amount: dec!(-1),
    }];
    let input = AccountInput::new(ns(0, 100), lanes, &flows).unwrap();
    let account = attribute_account(&input).unwrap();
    // Perpetual -4.3. Option 20 + 200 (0 - 1.6) = -300. Hedge: capture
    // -1 (100 - 99.9) = -0.1, inventory -1 (98 - 100) = 2, fee 0.05: 1.85.
    let nets: Vec<Decimal> = account
        .lanes
        .iter()
        .map(|l| l.attribution.net_pnl)
        .collect();
    assert_eq!(nets, [dec!(-4.3), dec!(-300), dec!(1.85)]);
    assert_eq!(account.quoted.lanes, 2);
    assert_eq!(account.quoted.attribution.net_pnl, dec!(-304.3));
    assert_eq!(account.hedge.attribution.net_pnl, dec!(1.85));
    assert_eq!(account.all_lanes.attribution.net_pnl, dec!(-302.45));
    assert_eq!(account.net_pnl, dec!(-303.45));
    let books = AccountBooks {
        lanes: vec![
            LedgerTotals {
                net_pnl: dec!(-4.3),
                position: dec!(2),
            },
            LedgerTotals {
                net_pnl: dec!(-300),
                position: dec!(2),
            },
            LedgerTotals {
                net_pnl: dec!(1.85),
                position: dec!(-1),
            },
        ],
        net_pnl: dec!(-303.45),
    };
    account.reconcile(&books).unwrap();
    assert!(account.residuals(&books).unwrap().is_zero());
    let values = account.values(Some(&books)).unwrap();
    assert_eq!(
        value(&values, MetricId::ReconciliationResidual),
        Reading::Value(dec!(0))
    );
    // The option lane's books off by 0.02: caught on that lane.
    let mut off = books.clone();
    off.lanes[1].net_pnl = dec!(-299.98);
    assert!(matches!(
        account.reconcile(&off),
        Err(EvaluationError::NotReconciled {
            what: "lane net pnl",
            ..
        })
    ));
    let residuals = account.residuals(&off).unwrap();
    assert_eq!(residuals.lanes[1], (dec!(-0.02), dec!(0)));
    // Attributed minus booked: the quoted role reports 0.02 less than its books.
    assert_eq!(residuals.quoted, dec!(-0.02));
    assert_eq!(residuals.hedge, dec!(0));
    assert_eq!(residuals.largest(), dec!(0.02));
    let values = account.values(Some(&off)).unwrap();
    assert_eq!(
        value(&values, MetricId::ReconciliationResidual),
        Reading::Value(dec!(0.02))
    );
    off.lanes.pop();
    assert!(account.residuals(&off).is_err());
}

#[test]
fn test_a_single_lane_account_matches_the_single_lane_attribution() {
    let marks = [
        Mark {
            timestamp_ms: 0,
            mid: dec!(100),
        },
        Mark {
            timestamp_ms: 1000,
            mid: dec!(102),
        },
        Mark {
            timestamp_ms: 2000,
            mid: dec!(101),
        },
    ];
    let mut buy = ExecutedFill {
        timestamp_ms: 0,
        side: Action::Buy,
        price: dec!(99.5),
        quantity: dec!(2),
        quote_fee: dec!(0.1),
        base_fee: dec!(0.01),
    };
    let sell = ExecutedFill {
        timestamp_ms: 2000,
        side: Action::Sell,
        price: dec!(101.5),
        quantity: dec!(1),
        quote_fee: dec!(0.2),
        base_fee: Decimal::ZERO,
    };
    let fills = [buy, sell];
    let h_marks = [
        Mark {
            timestamp_ms: 500,
            mid: dec!(50),
        },
        Mark {
            timestamp_ms: 1500,
            mid: dec!(49),
        },
    ];
    buy.timestamp_ms = 600;
    buy.base_fee = Decimal::ZERO;
    buy.price = dec!(50.1);
    let h_fills = [ExecutedFill {
        side: Action::Sell,
        ..buy
    }];
    let h_funding = [FundingPayment {
        timestamp_ms: 900,
        amount: dec!(0.3),
    }];
    let window = TimeWindow::new(0, 3000).unwrap();
    let single = EvaluationInput::new(window, &marks, &fills, dec!(1000))
        .unwrap()
        .with_hedge(&h_marks, &h_fills, &h_funding)
        .unwrap();
    let expected = attribute(&single).unwrap();

    let lane_marks: Vec<LaneMark> = marks.iter().copied().map(LaneMark::from).collect();
    let lane_fills: Vec<LaneFill> = fills.iter().map(LaneFill::from_executed).collect();
    let hedge_marks: Vec<LaneMark> = h_marks.iter().copied().map(LaneMark::from).collect();
    let hedge_fills: Vec<LaneFill> = h_fills.iter().map(LaneFill::from_executed).collect();
    let hedge_funding = [LaneFunding {
        time: 900,
        amount: dec!(0.3),
    }];
    let lanes = vec![
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, Decimal::ONE)
            .with_marks(&lane_marks)
            .with_fills(&lane_fills),
        LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, Decimal::ONE)
            .with_marks(&hedge_marks)
            .with_fills(&hedge_fills)
            .with_funding(&hedge_funding),
    ];
    let account =
        attribute_account(&AccountInput::new(Window::from(window), lanes, &[]).unwrap()).unwrap();
    let quoted = account.lanes[0].attribution;
    assert_eq!(quoted.spread_capture, expected.total.spread_capture);
    assert_eq!(quoted.inventory_pnl, expected.total.inventory_pnl);
    assert_eq!(quoted.quote_fees, expected.total.quote_fees);
    assert_eq!(quoted.base_fees, expected.total.base_fees);
    assert_eq!(quoted.net_pnl, expected.total.net_pnl);
    assert_eq!(account.lanes[0].final_position, expected.final_position);
    let market_maker_rs::backtest::evaluation::HedgeCosts::Attributed(hedge) =
        expected.total.hedge_costs
    else {
        panic!("a hedge is attributed");
    };
    let lane = account.lanes[1].attribution;
    assert_eq!(lane.spread_capture, hedge.spread_capture);
    assert_eq!(lane.inventory_pnl, hedge.inventory_pnl);
    assert_eq!(lane.quote_fees, hedge.fees);
    assert_eq!(lane.funding, hedge.funding);
    assert_eq!(lane.net_pnl, hedge.net_pnl);
    assert_eq!(
        Some(account.lanes[1].final_position),
        expected.hedge_final_position
    );
}

#[test]
fn test_records_are_checked() {
    let refused = |lane: LaneRecords<'_>| {
        matches!(
            AccountInput::new(ns(0, 100), vec![lane], &[]),
            Err(EvaluationError::InvalidInput { .. })
        )
    };
    let base = || LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1));
    // A zero mark is admitted; a negative one is not.
    let zero = [mark(0, dec!(0))];
    assert!(AccountInput::new(ns(0, 100), vec![base().with_marks(&zero)], &[]).is_ok());
    let negative = [mark(0, dec!(-0.01))];
    assert!(refused(base().with_marks(&negative)));
    // The window is half open: a record at its end is outside.
    let at_end = [mark(100, dec!(1))];
    assert!(refused(base().with_marks(&at_end)));
    assert!(refused(LaneRecords::new(
        LaneRole::Quoted,
        LaneKind::Linear,
        dec!(0)
    )));
    // An option fill has no base fee.
    let marks = [mark(0, dec!(1))];
    let mut with_fee = fill(0, Action::Buy, dec!(1), dec!(1));
    with_fee.base_fee = dec!(0.1);
    let fee_fills = [with_fee];
    assert!(refused(
        LaneRecords::new(LaneRole::Quoted, LaneKind::Option, dec!(1))
            .with_marks(&marks)
            .with_fills(&fee_fills)
    ));
    // Nothing but funding after a settlement.
    let late = [mark(0, dec!(1)), mark(60, dec!(1))];
    let settlement = Settlement {
        time: 50,
        price: dec!(1),
        cash: dec!(0),
    };
    assert!(refused(
        base().with_marks(&late).with_settlement(settlement)
    ));
    // An order filled beyond its size, or one no record names.
    let mut named = fill(0, Action::Buy, dec!(1), dec!(2));
    named.order_id = Some(7);
    let named = [named];
    let small = [OrderRecord {
        id: 7,
        accepted_quantity: dec!(1),
        replace_increase: Decimal::ZERO,
        accepted_replaces: 0,
        end_state: OrderEnd::Filled,
        liquidation: false,
    }];
    assert!(refused(
        base()
            .with_marks(&marks)
            .with_fills(&named)
            .with_orders(&small)
    ));
    assert!(refused(base().with_marks(&marks).with_fills(&named)));
    // A fill of exactly the offered size must end filled.
    let mut exact = fill(0, Action::Buy, dec!(1), dec!(2));
    exact.order_id = Some(8);
    let exact = [exact];
    let full = OrderRecord {
        id: 8,
        accepted_quantity: dec!(2),
        replace_increase: Decimal::ZERO,
        accepted_replaces: 0,
        end_state: OrderEnd::Filled,
        liquidation: false,
    };
    for end_state in [OrderEnd::Cancelled, OrderEnd::Expired, OrderEnd::Open] {
        let orders = [OrderRecord { end_state, ..full }];
        assert!(refused(
            base()
                .with_marks(&marks)
                .with_fills(&exact)
                .with_orders(&orders)
        ));
    }
    // A replace increase needs an accepted replace.
    let unreplaced = [OrderRecord {
        accepted_quantity: dec!(1),
        replace_increase: dec!(1),
        ..full
    }];
    assert!(refused(
        base()
            .with_marks(&marks)
            .with_fills(&exact)
            .with_orders(&unreplaced)
    ));
    let replaced = [OrderRecord {
        accepted_replaces: 1,
        ..unreplaced[0]
    }];
    assert!(!refused(
        base()
            .with_marks(&marks)
            .with_fills(&exact)
            .with_orders(&replaced)
    ));
    // A fill with no mark and no opening price is refused by the attribution.
    let early = [fill(0, Action::Buy, dec!(1), dec!(1))];
    let input = AccountInput::new(ns(0, 100), vec![base().with_fills(&early)], &[]).unwrap();
    assert!(matches!(
        attribute_account(&input),
        Err(EvaluationError::InvalidInput { .. })
    ));
    assert!(Window::new(5, 5, TimeUnit::Microseconds).is_err());
}

#[cfg(feature = "instruments")]
#[test]
fn test_lane_contract_reads_the_multiplier_and_refuses_inverse() {
    use market_maker_rs::backtest::evaluation::{check_contract, lane_contract};
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
    let ten = spec(
        Margining::Linear,
        "USDT",
        ContractSize::base(dec!(10), asset("BTC")).unwrap(),
    );
    let lane = lane_contract(&ten).unwrap();
    assert_eq!(lane.kind, LaneKind::Linear);
    assert_eq!(lane.multiplier, dec!(10));
    // The single lane check still refuses it.
    assert!(check_contract(&ten).is_err());
    let inverse = spec(
        Margining::Inverse,
        "BTC",
        ContractSize::quote_value(dec!(10), asset("USDT")).unwrap(),
    );
    assert!(matches!(
        lane_contract(&inverse),
        Err(EvaluationError::UnsupportedContract { .. })
    ));
}

#[test]
fn test_a_transfer_takes_its_cash_as_given() {
    // A long of 3 at 100, multiplier 1, closed by a transfer that pays 100.01 in
    // all, not 3 * price; mark 100.5 at 0.
    let marks = [mark(0, dec!(100.5))];
    let transfers = [LaneTransfer {
        time: 1,
        side: Action::Sell,
        quantity: dec!(3),
        cash: dec!(100.01),
    }];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
        .with_opening(Opening {
            position: dec!(3),
            price: dec!(100),
        })
        .with_marks(&marks)
        .with_transfers(&transfers);
    let input = AccountInput::new(ns(0, 10), vec![lane], &[]).unwrap();
    let account = attribute_account(&input).unwrap();
    let a = account.lanes[0].attribution;
    // Edge: 100.01 + (-1) 3 * 1 * 100.5 = 100.01 - 301.5.
    assert_eq!(a.transfer_edge, dec!(-201.49));
    // Inventory: 3 (100.5 - 100).
    assert_eq!(a.inventory_pnl, dec!(1.5));
    // Directly: cash 100.01, flat, less the opening value 300.
    assert_eq!(a.net_pnl, dec!(-199.99));
    assert_eq!(account.lanes[0].final_position, dec!(0));

    // A sell side transfer at a price: short 2 contracts of multiplier 10 at a
    // mark of 50 for 1001 received, so an edge of 1001 - 2 * 10 * 50.
    let transfers = [LaneTransfer {
        time: 1,
        side: Action::Sell,
        quantity: dec!(2),
        cash: dec!(1001),
    }];
    let marks = [mark(0, dec!(50)), mark(5, dec!(49))];
    let lane = LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, dec!(10))
        .with_marks(&marks)
        .with_transfers(&transfers);
    let account =
        attribute_account(&AccountInput::new(ns(0, 10), vec![lane], &[]).unwrap()).unwrap();
    let a = account.lanes[0].attribution;
    assert_eq!(a.transfer_edge, dec!(1));
    // Inventory: -20 (49 - 50).
    assert_eq!(a.inventory_pnl, dec!(20));
    // Directly: 1001 - 20 * 49.
    assert_eq!(a.net_pnl, dec!(21));
    assert_eq!(account.lanes[0].final_position, dec!(-2));
}

#[test]
fn test_order_quantities_add_up_in_units_across_multipliers() {
    // Lane A, multiplier 1: an order of 4, filled 4. Lane B, multiplier 100: an
    // order of 1, filled 0.5 then cancelled.
    let a_marks = [mark(0, dec!(10))];
    let mut a_fill = fill(0, Action::Buy, dec!(10), dec!(4));
    a_fill.order_id = Some(1);
    let a_fills = [a_fill];
    let order = |id, accepted_quantity, end_state| OrderRecord {
        id,
        accepted_quantity,
        replace_increase: Decimal::ZERO,
        accepted_replaces: 0,
        end_state,
        liquidation: false,
    };
    let a_orders = [order(1, dec!(4), OrderEnd::Filled)];
    let b_marks = [mark(0, dec!(10))];
    let mut b_fill = fill(0, Action::Sell, dec!(10), dec!(0.5));
    b_fill.order_id = Some(1);
    let b_fills = [b_fill];
    let b_orders = [order(1, dec!(1), OrderEnd::Cancelled)];
    let lanes = vec![
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_marks(&a_marks)
            .with_fills(&a_fills)
            .with_orders(&a_orders),
        LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, dec!(100))
            .with_marks(&b_marks)
            .with_fills(&b_fills)
            .with_orders(&b_orders),
    ];
    let account = attribute_account(&AccountInput::new(ns(0, 10), lanes, &[]).unwrap()).unwrap();
    let all = account.all_lanes.orders;
    // Offered 4 + 100, filled 4 + 50: a ratio of 54 / 104, not of 4.5 / 5.
    assert_eq!(all.accepted_quantity, dec!(104));
    assert_eq!(all.filled_quantity, dec!(54));
    assert_eq!(all.ended_filled, 1);
    assert_eq!(all.ended_cancelled, 1);
    assert_eq!(account.lanes[1].volume, dec!(50));
    let values = account.values(None).unwrap();
    assert_eq!(
        value(&values, MetricId::FillRatioByQuantity),
        Reading::Value(dec!(54) / dec!(104))
    );
}

#[test]
fn test_a_replace_that_raises_the_open_size_adds_to_the_offer() {
    // Accepted 10, filled 8, replaced to an open size of 10 (an increase of 8),
    // filled 10 more: offered 18, filled 18, ended filled.
    let marks = [mark(0, dec!(10))];
    let mut first = fill(1, Action::Buy, dec!(10), dec!(8));
    first.order_id = Some(1);
    let mut second = fill(3, Action::Buy, dec!(10), dec!(10));
    second.order_id = Some(1);
    let fills = [first, second];
    let record = OrderRecord {
        id: 1,
        accepted_quantity: dec!(10),
        replace_increase: dec!(8),
        accepted_replaces: 1,
        end_state: OrderEnd::Filled,
        liquidation: false,
    };
    assert_eq!(record.offered_quantity().unwrap(), dec!(18));
    let orders = [record];
    let lane = || {
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_marks(&marks)
            .with_fills(&fills)
    };
    let input = AccountInput::new(ns(0, 10), vec![lane().with_orders(&orders)], &[]).unwrap();
    let stats = attribute_account(&input).unwrap().lanes[0].orders;
    assert_eq!(stats.accepted_quantity, dec!(18));
    assert_eq!(stats.filled_quantity, dec!(18));
    assert_eq!(stats.ended_filled, 1);
    // Without the increase the fills exceed the offer.
    let short = [OrderRecord {
        replace_increase: Decimal::ZERO,
        ..record
    }];
    assert!(AccountInput::new(ns(0, 10), vec![lane().with_orders(&short)], &[]).is_err());
    // Ended filled with 18 of 20 filled is admitted for a replaced order, as a
    // replace may have lowered the open size; without a replace it is refused.
    let more = [OrderRecord {
        replace_increase: dec!(10),
        ..record
    }];
    assert!(AccountInput::new(ns(0, 10), vec![lane().with_orders(&more)], &[]).is_ok());
    let unreplaced = [OrderRecord {
        accepted_quantity: dec!(20),
        replace_increase: Decimal::ZERO,
        accepted_replaces: 0,
        ..record
    }];
    assert!(AccountInput::new(ns(0, 10), vec![lane().with_orders(&unreplaced)], &[]).is_err());
    // The same order still open is admitted.
    let open = [OrderRecord {
        replace_increase: dec!(10),
        end_state: OrderEnd::Open,
        ..record
    }];
    let input = AccountInput::new(ns(0, 10), vec![lane().with_orders(&open)], &[]).unwrap();
    let stats = attribute_account(&input).unwrap().lanes[0].orders;
    assert_eq!(stats.ended_open, 1);
    assert_eq!(stats.ended_filled, 0);
}

#[test]
fn test_settlement_rounding_within_half_a_quantum() {
    // A long of 1 of multiplier 1, settled at 100.005 and paid 100.00, the cash
    // rounded to the cent.
    let marks = [mark(0, dec!(100))];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Option, dec!(1))
        .with_opening(Opening {
            position: dec!(1),
            price: dec!(100),
        })
        .with_marks(&marks)
        .with_settlement(Settlement {
            time: 5,
            price: dec!(100.005),
            cash: dec!(100.00),
        });
    let account =
        attribute_account(&AccountInput::new(ns(0, 10), vec![lane], &[]).unwrap()).unwrap();
    let a = account.lanes[0].attribution;
    assert_eq!(a.inventory_pnl, dec!(0.005));
    assert_eq!(a.settlement_rounding, dec!(-0.005));
    assert_eq!(a.net_pnl, dec!(0));
}

#[test]
fn test_a_future_settles_at_expiry() {
    // A short of 2 futures of multiplier 5, sold at 200.5 under a mark of 200,
    // mark 198 at 10, settled at 197: the account pays 1970 for the short.
    let marks = [mark(0, dec!(200)), mark(10, dec!(198))];
    let fills = [fill(0, Action::Sell, dec!(200.5), dec!(2))];
    let lane = LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, dec!(5))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_settlement(Settlement {
            time: 20,
            price: dec!(197),
            cash: dec!(-1970),
        });
    let account =
        attribute_account(&AccountInput::new(ns(0, 100), vec![lane], &[]).unwrap()).unwrap();
    let outcome = account.lanes[0];
    let a = outcome.attribution;
    // -2 * 5 (200 - 200.5).
    assert_eq!(a.spread_capture, dec!(5));
    // -10 (198 - 200) - 10 (197 - 198).
    assert_eq!(a.inventory_pnl, dec!(30));
    assert_eq!(a.settlement_rounding, dec!(0));
    // Directly: 2005 - 1970.
    assert_eq!(a.net_pnl, dec!(35));
    assert_eq!(outcome.final_position, dec!(0));
    assert!(outcome.settled);
    // A mark at the settlement instant serves as the horizon mark.
    let marks = [mark(0, dec!(200)), mark(10, dec!(198)), mark(20, dec!(197))];
    let lane = LaneRecords::new(LaneRole::Hedge, LaneKind::Linear, dec!(5))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_settlement(Settlement {
            time: 20,
            price: dec!(197),
            cash: dec!(-1970),
        });
    let input = AccountInput::new(ns(0, 100), vec![lane], &[]).unwrap();
    assert_eq!(attribute_account(&input).unwrap().lanes[0].attribution, a);
    let spec = LaneQualitySpec {
        horizon: 20,
        max_mark_delay: 0,
    };
    let q = lane_fill_quality(&input, 0, spec).unwrap();
    // Sell markout: -(197 - 200.5).
    assert_eq!(q.all.signed_markout, Some(dec!(3.5)));
    // A horizon of 15 falls before the settlement: the mark at 20 measures it
    // within a delay of 5, and leaves it unmeasured within a delay of 4.
    let within = LaneQualitySpec {
        horizon: 15,
        max_mark_delay: 5,
    };
    let q = lane_fill_quality(&input, 0, within).unwrap();
    assert_eq!(q.all.signed_markout, Some(dec!(3.5)));
    let short = LaneQualitySpec {
        horizon: 15,
        max_mark_delay: 4,
    };
    let q = lane_fill_quality(&input, 0, short).unwrap();
    assert_eq!(q.all.signed_markout, None);
}

#[test]
fn test_a_base_fee_with_a_multiplier() {
    // Multiplier 10: a buy of 2 at 49 under a mark of 50, fee 0.1 contracts;
    // mark 51 later.
    let marks = [mark(0, dec!(50)), mark(5, dec!(51))];
    let mut buy = fill(0, Action::Buy, dec!(49), dec!(2));
    buy.base_fee = dec!(0.1);
    let fills = [buy];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(10))
        .with_marks(&marks)
        .with_fills(&fills);
    let account =
        attribute_account(&AccountInput::new(ns(0, 10), vec![lane], &[]).unwrap()).unwrap();
    let outcome = account.lanes[0];
    let a = outcome.attribution;
    // 2 * 10 (50 - 49).
    assert_eq!(a.spread_capture, dec!(20));
    // 0.1 * 10 * 50.
    assert_eq!(a.base_fees, dec!(50));
    // 1.9 * 10 (51 - 50).
    assert_eq!(a.inventory_pnl, dec!(19));
    // Directly: -980 + 1.9 * 10 * 51 = -980 + 969.
    assert_eq!(a.net_pnl, dec!(-11));
    assert_eq!(outcome.final_position, dec!(1.9));
    // Volume gross of the fee: 2 * 10.
    assert_eq!(outcome.volume, dec!(20));
}

#[test]
fn test_an_order_accepted_before_the_window_records_its_open_size_at_the_start() {
    // Accepted before the window starting at 10 and open for 6 at its start;
    // filled for 6 inside it: accepted quantity 6, ended filled.
    let marks = [mark(10, dec!(10))];
    let mut inside = fill(12, Action::Sell, dec!(10), dec!(6));
    inside.order_id = Some(3);
    let fills = [inside];
    let orders = [OrderRecord {
        id: 3,
        accepted_quantity: dec!(6),
        replace_increase: Decimal::ZERO,
        accepted_replaces: 0,
        end_state: OrderEnd::Filled,
        liquidation: false,
    }];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
        .with_marks(&marks)
        .with_fills(&fills)
        .with_orders(&orders);
    let input = AccountInput::new(ns(10, 20), vec![lane], &[]).unwrap();
    let stats = attribute_account(&input).unwrap().lanes[0].orders;
    assert_eq!(stats.accepted_quantity, dec!(6));
    assert_eq!(stats.filled_quantity, dec!(6));
    assert_eq!(stats.ended_filled, 1);
}

#[test]
fn test_an_order_reduced_by_a_replace_then_filled_ends_filled() {
    // Accepted 5, replaced down to 2 (no increase), filled 2: offered 5,
    // filled 2, ended filled. Fill ratio by quantity 2 / 5.
    let marks = [mark(0, dec!(10))];
    let mut only = fill(2, Action::Buy, dec!(10), dec!(2));
    only.order_id = Some(7);
    let fills = [only];
    let record = OrderRecord {
        id: 7,
        accepted_quantity: dec!(5),
        replace_increase: Decimal::ZERO,
        accepted_replaces: 1,
        end_state: OrderEnd::Filled,
        liquidation: false,
    };
    let orders = [record];
    let lane = |orders| {
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_marks(&marks)
            .with_fills(&fills)
            .with_orders(orders)
    };
    let input = AccountInput::new(ns(0, 10), vec![lane(&orders)], &[]).unwrap();
    let stats = attribute_account(&input).unwrap().lanes[0].orders;
    assert_eq!(stats.accepted_quantity, dec!(5));
    assert_eq!(stats.filled_quantity, dec!(2));
    assert_eq!(stats.ended_filled, 1);
    // The same record with no accepted replace is refused.
    let plain = [OrderRecord {
        accepted_replaces: 0,
        ..record
    }];
    assert!(matches!(
        AccountInput::new(ns(0, 10), vec![lane(&plain)], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
    // A replaced order that ended filled with no fill is refused.
    let unfilled = [OrderRecord { id: 8, ..record }];
    assert!(matches!(
        AccountInput::new(ns(0, 10), vec![lane(&unfilled)], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
}

#[test]
fn test_a_sequence_orders_a_fill_before_a_mark_of_its_instant() {
    // Marks 10 at 0 and 12 at 5; a buy of 1 at 11 at 5. By default the mark
    // at 5 comes first: spread capture 12 - 11 = 1, inventory 0. Sequenced
    // fill (1) before mark (2): the fill sees 10, spread capture 10 - 11 = -1,
    // and the move to 12 on a position of 1 earns inventory 2. Net 1 either way.
    // A mark of 12 at 7 measures the fill after a horizon of 1; it moves
    // nothing. For this buy at 11 the effective spread is 2 (m - 11) and the
    // realized spread 2 (12 - 11) = 2, so the price impact is 2 (m - 12): with
    // m = 12 by default, effective 2 and impact 0; with m = 10 when sequenced,
    // effective -2 and impact -4.
    let marks = [mark(0, dec!(10)), mark(5, dec!(12)), mark(7, dec!(12))];
    let fills = [fill(5, Action::Buy, dec!(11), dec!(1))];
    let base = || {
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_marks(&marks)
            .with_fills(&fills)
    };
    let default = AccountInput::new(ns(0, 10), vec![base()], &[]).unwrap();
    let row = attribute_account(&default).unwrap().lanes[0].attribution;
    assert_eq!(row.spread_capture, dec!(1));
    assert_eq!(row.inventory_pnl, dec!(0));
    assert_eq!(row.net_pnl, dec!(1));
    let spec = LaneQualitySpec {
        horizon: 1,
        max_mark_delay: 10,
    };
    let quality = lane_fill_quality(&default, 0, spec).unwrap();

    let sequenced = AccountInput::new(
        ns(0, 10),
        vec![base().with_sequence(LaneSequence::new(&[0, 2, 0], &[1], &[]))],
        &[],
    )
    .unwrap();
    let row = attribute_account(&sequenced).unwrap().lanes[0].attribution;
    assert_eq!(row.spread_capture, dec!(-1));
    assert_eq!(row.inventory_pnl, dec!(2));
    assert_eq!(row.net_pnl, dec!(1));
    let seq_quality = lane_fill_quality(&sequenced, 0, spec).unwrap();
    assert_eq!(quality.all.measured, 1);
    assert_eq!(seq_quality.all.measured, 1);
    assert_eq!(quality.all.price_impact, Some(dec!(0)));
    assert_eq!(seq_quality.all.price_impact, Some(dec!(-4)));
    assert_eq!(quality.all.effective_spread, Some(dec!(2)));
    assert_eq!(seq_quality.all.effective_spread, Some(dec!(-2)));

    // Sequence numbers that tie keep the default order.
    let tied = AccountInput::new(
        ns(0, 10),
        vec![base().with_sequence(LaneSequence::new(&[0, 1, 0], &[1], &[]))],
        &[],
    )
    .unwrap();
    let row = attribute_account(&tied).unwrap().lanes[0].attribution;
    assert_eq!(row.spread_capture, dec!(1));

    // One number per record is required.
    for sequence in [
        LaneSequence::new(&[0, 2], &[1], &[]),
        LaneSequence::new(&[0, 2, 0], &[], &[]),
        LaneSequence::new(&[0, 2, 0], &[1, 1], &[]),
        LaneSequence::new(&[0, 2, 0], &[1], &[3]),
    ] {
        assert!(matches!(
            AccountInput::new(ns(0, 10), vec![base().with_sequence(sequence)], &[]),
            Err(EvaluationError::InvalidInput { .. })
        ));
    }
}

#[test]
fn test_a_sequence_needs_non_decreasing_marks_at_one_instant() {
    let marks = [mark(5, dec!(10)), mark(5, dec!(12))];
    let lane = |sequence: &'static [u64]| {
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_marks(&marks)
            .with_sequence(LaneSequence::new(sequence, &[], &[]))
    };
    assert!(AccountInput::new(ns(0, 10), vec![lane(&[1, 1])], &[]).is_ok());
    assert!(AccountInput::new(ns(0, 10), vec![lane(&[1, 2])], &[]).is_ok());
    assert!(matches!(
        AccountInput::new(ns(0, 10), vec![lane(&[2, 1])], &[]),
        Err(EvaluationError::InvalidInput { .. })
    ));
    // Marks at different instants may decrease.
    let apart = [mark(4, dec!(10)), mark(5, dec!(12))];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
        .with_marks(&apart)
        .with_sequence(LaneSequence::new(&[2, 1], &[], &[]));
    assert!(AccountInput::new(ns(0, 10), vec![lane], &[]).is_ok());
}

#[test]
fn test_the_settlement_stays_last_against_the_highest_sequence() {
    // Long 1 from the opening at 10; a mark of 12 and a settlement at 14, both
    // at 5. The mark numbered u64::MAX still comes before the settlement: the
    // inventory earns 2 at the mark and the settlement rounding is zero.
    let marks = [mark(5, dec!(12))];
    let lane = LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
        .with_opening(Opening {
            position: dec!(1),
            price: dec!(10),
        })
        .with_marks(&marks)
        .with_settlement(Settlement {
            time: 5,
            price: dec!(14),
            cash: dec!(14),
        });
    let plain = AccountInput::new(ns(0, 10), vec![lane], &[]).unwrap();
    let expected = attribute_account(&plain).unwrap().lanes[0].attribution;
    let sequenced = AccountInput::new(
        ns(0, 10),
        vec![lane.with_sequence(LaneSequence::new(&[u64::MAX], &[], &[]))],
        &[],
    )
    .unwrap();
    let row = attribute_account(&sequenced).unwrap().lanes[0].attribution;
    assert_eq!(row, expected);
    assert_eq!(row.net_pnl, dec!(4));
}

#[test]
fn test_a_sequence_orders_a_transfer_before_a_mark_of_its_instant() {
    // Long 1 from the opening at 10; marks 12 at 5; a transfer sells 1 at 5
    // for cash 11. Default: edge against 12 is 11 - 12 = -1, inventory 2.
    // Sequenced before the mark: edge against 10 is 1, inventory 0.
    let marks = [mark(5, dec!(12))];
    let transfers = [LaneTransfer {
        time: 5,
        side: Action::Sell,
        quantity: dec!(1),
        cash: dec!(11),
    }];
    let base = || {
        LaneRecords::new(LaneRole::Quoted, LaneKind::Linear, dec!(1))
            .with_opening(Opening {
                position: dec!(1),
                price: dec!(10),
            })
            .with_marks(&marks)
            .with_transfers(&transfers)
    };
    let default = AccountInput::new(ns(0, 10), vec![base()], &[]).unwrap();
    let row = attribute_account(&default).unwrap().lanes[0].attribution;
    assert_eq!(row.transfer_edge, dec!(-1));
    assert_eq!(row.inventory_pnl, dec!(2));
    let sequenced = AccountInput::new(
        ns(0, 10),
        vec![base().with_sequence(LaneSequence::new(&[2], &[], &[1]))],
        &[],
    )
    .unwrap();
    let row = attribute_account(&sequenced).unwrap().lanes[0].attribution;
    assert_eq!(row.transfer_edge, dec!(1));
    assert_eq!(row.inventory_pnl, dec!(0));
    assert_eq!(row.net_pnl, dec!(1));
}
