//! Integration tests for the market making library.
//!
//! These tests drive complete workflows across modules with exact, hand
//! computed expectations. They cover order lifecycle bookkeeping
//! (`OrderManager`), position accounting (`InventoryPosition`) and cash
//! reconciliation from fills with explicit fees.
//!
//! Scope: this is a minimal deterministic harness. It does not validate
//! causality, latency or queue position in the backtest engine; those
//! guarantees belong to the backtest lifecycle work (M1).

use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::execution::{
    Action, Fill, OrderId, OrderManager, OrderRequest, OrderResponse, OrderStatus,
};
use market_maker_rs::position::inventory::InventoryPosition;

const SYMBOL: &str = "BTC-USD";

/// Cash and fee ledger rebuilt from fills.
#[derive(Debug, Default)]
struct Ledger {
    cash: Decimal,
    fees: Decimal,
}

impl Ledger {
    /// Rebuilds a ledger from the fills stored by the order manager.
    fn from_fills<'a>(fills: impl IntoIterator<Item = &'a Fill>) -> Self {
        let mut ledger = Self::default();
        for fill in fills {
            ledger.apply(fill);
        }
        ledger
    }

    fn apply(&mut self, fill: &Fill) {
        let notional = fill.price * fill.quantity;
        match fill.side {
            Action::Buy => self.cash -= notional,
            Action::Sell => self.cash += notional,
            other => panic!("unsupported action {other}"),
        }
        self.cash -= fill.fee;
        self.fees += fill.fee;
    }
}

fn fill(
    exchange_id: &str,
    trade_id: &str,
    side: Action,
    price: Decimal,
    qty: Decimal,
    fee: Decimal,
    ts: u64,
) -> Fill {
    Fill {
        symbol: "BTC-USD".to_string(),
        order_id: OrderId::new(exchange_id),
        trade_id: trade_id.to_string(),
        price,
        quantity: qty,
        side,
        timestamp: ts,
        fee,
        fee_currency: "USD".to_string(),
    }
}

fn signed(side: Action, qty: Decimal) -> Decimal {
    match side {
        Action::Buy => qty,
        Action::Sell => -qty,
        other => panic!("unsupported action {other}"),
    }
}

/// Registers a limit order and acknowledges it under `exchange_id`.
fn place(
    manager: &mut OrderManager,
    request: &OrderRequest,
    client_id: &str,
    exchange_id: &str,
    ts: u64,
) {
    manager
        .register_order(request, client_id.to_string(), ts)
        .expect("register order");
    let ack = OrderResponse::new(
        OrderId::new(exchange_id),
        OrderStatus::Open {
            filled_qty: Decimal::ZERO,
        },
        ts + 1,
    );
    manager
        .update_order(client_id, &ack, ts + 1)
        .expect("acknowledge order");
}

#[test]
fn partial_fill_then_cancel_reconciles_exactly() {
    let mut manager = OrderManager::with_defaults();
    let mut position = InventoryPosition::new();
    let mut ledger = Ledger::default();

    let request = OrderRequest::limit_buy(SYMBOL, dec!(100), dec!(1.0));
    place(&mut manager, &request, "c-1", "ex-1", 1_000);
    assert_eq!(
        manager.get_open_quantity(SYMBOL, Action::Buy).unwrap(),
        dec!(1.0)
    );

    // Partial fill of 0.4 at 100 with a 0.04 fee.
    let partial = fill(
        "ex-1",
        "t-1",
        Action::Buy,
        dec!(100),
        dec!(0.4),
        dec!(0.04),
        1_010,
    );
    manager.record_fill(&partial, 1_010).expect("record fill");
    position
        .update_fill(
            signed(partial.side, partial.quantity),
            partial.price,
            partial.timestamp,
        )
        .unwrap();
    ledger.apply(&partial);

    let order = manager.get_order_by_client_id("c-1").expect("order exists");
    assert_eq!(order.filled_quantity, dec!(0.4));
    assert_eq!(order.remaining_quantity, dec!(0.6));
    assert_eq!(order.average_fill_price, dec!(100));
    assert!(matches!(order.status, OrderStatus::PartiallyFilled { .. }));

    // Cancel the remainder.
    manager.mark_cancelled("c-1", 1_020).expect("cancel");
    let order = manager.get_order_by_client_id("c-1").expect("order exists");
    assert!(order.is_terminal());
    assert_eq!(order.filled_quantity, dec!(0.4));
    assert_eq!(order.remaining_quantity, dec!(0.6));
    assert_eq!(
        manager.get_open_quantity(SYMBOL, Action::Buy).unwrap(),
        Decimal::ZERO
    );
    assert_eq!(manager.open_order_count(), 0);

    // A cancel never erases the fill that happened before it.
    assert_eq!(position.quantity, dec!(0.4));
    assert_eq!(position.avg_entry_price, dec!(100));
    assert_eq!(ledger.cash, dec!(-40.04));
    assert_eq!(ledger.fees, dec!(0.04));

    // Reconciliation: the fills stored by the order manager, after the
    // cancel, rebuild the same position, cash and fees.
    let recorded: Decimal = order.fills.iter().map(|f| f.quantity).sum();
    assert_eq!(recorded, position.quantity);
    let stored = Ledger::from_fills(&order.fills);
    assert_eq!(stored.cash, dec!(-40.04));
    assert_eq!(stored.fees, dec!(0.04));
}

#[test]
fn round_trip_with_two_fills_reconciles_exactly() {
    let mut manager = OrderManager::with_defaults();
    let mut position = InventoryPosition::new();
    let mut ledger = Ledger::default();

    // Buy 2.0 at 50 in two fills, then sell 2.0 at 51 in one fill.
    place(
        &mut manager,
        &OrderRequest::limit_buy(SYMBOL, dec!(50), dec!(2.0)),
        "b-1",
        "ex-b1",
        2_000,
    );
    let buys = [
        fill(
            "ex-b1",
            "t-1",
            Action::Buy,
            dec!(50),
            dec!(1.5),
            dec!(0.075),
            2_010,
        ),
        fill(
            "ex-b1",
            "t-2",
            Action::Buy,
            dec!(50),
            dec!(0.5),
            dec!(0.025),
            2_011,
        ),
    ];
    for f in &buys {
        manager
            .record_fill(f, f.timestamp)
            .expect("record buy fill");
        position
            .update_fill(signed(f.side, f.quantity), f.price, f.timestamp)
            .unwrap();
        ledger.apply(f);
    }
    let bid = manager.get_order_by_client_id("b-1").expect("bid exists");
    assert!(matches!(bid.status, OrderStatus::Filled { .. }));
    assert_eq!(bid.remaining_quantity, Decimal::ZERO);

    place(
        &mut manager,
        &OrderRequest::limit_sell(SYMBOL, dec!(51), dec!(2.0)),
        "a-1",
        "ex-a1",
        2_100,
    );
    let sell = fill(
        "ex-a1",
        "t-3",
        Action::Sell,
        dec!(51),
        dec!(2.0),
        dec!(0.102),
        2_110,
    );
    manager
        .record_fill(&sell, sell.timestamp)
        .expect("record sell fill");
    position
        .update_fill(signed(sell.side, sell.quantity), sell.price, sell.timestamp)
        .unwrap();
    ledger.apply(&sell);

    assert!(position.is_flat());
    assert_eq!(position.avg_entry_price, Decimal::ZERO);
    assert_eq!(manager.open_order_count(), 0);
    // Gross: -100 + 102 = 2; fees: 0.075 + 0.025 + 0.102 = 0.202.
    assert_eq!(ledger.fees, dec!(0.202));
    assert_eq!(ledger.cash, dec!(1.798));

    // Reconciliation from the order manager's stored fills: signed quantity
    // ties to the position and the rebuilt cash and fees match exactly.
    let bid = manager.get_order_by_client_id("b-1").expect("bid exists");
    let ask = manager.get_order_by_client_id("a-1").expect("ask exists");
    let stored_fills: Vec<&Fill> = bid.fills.iter().chain(ask.fills.iter()).collect();
    let net: Decimal = stored_fills
        .iter()
        .map(|f| signed(f.side, f.quantity))
        .sum();
    assert_eq!(net, position.quantity);
    let stored = Ledger::from_fills(stored_fills);
    assert_eq!(stored.fees, dec!(0.202));
    assert_eq!(stored.cash, dec!(1.798));
}

#[test]
fn fill_for_unknown_order_is_rejected() {
    let mut manager = OrderManager::with_defaults();
    let orphan = fill(
        "ex-missing",
        "t-x",
        Action::Buy,
        dec!(10),
        dec!(1),
        Decimal::ZERO,
        1,
    );
    assert!(manager.record_fill(&orphan, 1).is_err());
}
