//! OrderBook-rs connector example.
//!
//! Demonstrates using the OrderBookConnector to submit, cancel, and manage
//! orders through the ExchangeConnector trait.
//!
//! Run with: `cargo run --example orderbook_connector`

use market_maker_rs::execution::{
    ExchangeConnector, OrderBookConnector, OrderBookConnectorConfig, OrderRequest, OrderType, Side,
    TimeInForce,
};
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() {
    println!("=== OrderBook-rs Connector Example ===\n");

    // Create connector with custom configuration
    let config = OrderBookConnectorConfig {
        price_precision: 2,
        quantity_precision: 8,
        fee_rate: dec!(0.001), // 0.1% fee
        fee_currency: "USD".to_string(),
    };

    let connector = OrderBookConnector::with_config("BTC-USD", config);

    println!("Connector Configuration:");
    println!("  Symbol: BTC-USD");
    println!("  Price Precision: 2 decimals");
    println!("  Quantity Precision: 8 decimals");
    println!("  Fee Rate: 0.1%");
    println!();

    // Set initial balance
    connector.set_balance("USD", dec!(100000.0));
    connector.set_balance("BTC", dec!(10.0));

    println!("Initial Balances:");
    let usd_balance = connector
        .get_balance("USD")
        .await
        .expect("Failed to get balance");
    let btc_balance = connector
        .get_balance("BTC")
        .await
        .expect("Failed to get balance");
    println!("  USD: ${:.2}", usd_balance);
    println!("  BTC: {:.8}", btc_balance);
    println!();

    // Submit a limit buy order
    println!("=== Order Submission ===\n");

    let buy_order = OrderRequest {
        symbol: "BTC-USD".to_string(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        quantity: dec!(0.5),
        price: Some(dec!(49000.0)),
        time_in_force: TimeInForce::GoodTilCancel,
        client_order_id: Some("buy-001".to_string()),
    };

    println!("Submitting limit buy order:");
    println!("  Side: {:?}", buy_order.side);
    println!("  Quantity: {} BTC", buy_order.quantity);
    println!("  Price: ${:?}", buy_order.price);

    let buy_response = connector
        .submit_order(buy_order)
        .await
        .expect("Failed to submit buy order");

    println!("  Order ID: {}", buy_response.order_id);
    println!("  Status: {:?}", buy_response.status);
    println!();

    // Submit a limit sell order
    let sell_order = OrderRequest {
        symbol: "BTC-USD".to_string(),
        side: Side::Sell,
        order_type: OrderType::Limit,
        quantity: dec!(0.3),
        price: Some(dec!(51000.0)),
        time_in_force: TimeInForce::GoodTilCancel,
        client_order_id: Some("sell-001".to_string()),
    };

    println!("Submitting limit sell order:");
    println!("  Side: {:?}", sell_order.side);
    println!("  Quantity: {} BTC", sell_order.quantity);
    println!("  Price: ${:?}", sell_order.price);

    let sell_response = connector
        .submit_order(sell_order)
        .await
        .expect("Failed to submit sell order");

    println!("  Order ID: {}", sell_response.order_id);
    println!("  Status: {:?}", sell_response.status);
    println!();

    // Get order book snapshot
    println!("=== Order Book Snapshot ===\n");

    let snapshot = connector
        .get_orderbook("BTC-USD", 5)
        .await
        .expect("Failed to get orderbook");

    println!("Order Book (top 5 levels):");
    println!("  Bids:");
    for level in &snapshot.bids {
        println!("    ${:.2} x {:.8}", level.price, level.quantity);
    }
    println!("  Asks:");
    for level in &snapshot.asks {
        println!("    ${:.2} x {:.8}", level.price, level.quantity);
    }
    println!();

    // Get open orders
    println!("=== Open Orders ===\n");

    let open_orders = connector
        .get_open_orders("BTC-USD")
        .await
        .expect("Failed to get open orders");

    println!("Open Orders: {}", open_orders.len());
    for order in &open_orders {
        println!("  {} - {:?}", order.order_id, order.status);
    }
    println!();

    // Get order status
    println!("=== Order Status ===\n");

    let status = connector
        .get_order_status(&buy_response.order_id)
        .await
        .expect("Failed to get order status");

    println!("Buy Order Status:");
    println!("  Order ID: {}", status.order_id);
    println!("  Status: {:?}", status.status);
    println!();

    // Modify an order
    println!("=== Order Modification ===\n");

    println!("Modifying buy order price from $49,000 to $49,500...");

    let modify_result = connector
        .modify_order(&buy_response.order_id, Some(dec!(49500.0)), None)
        .await
        .expect("Failed to modify order");

    println!("  New Order ID: {}", modify_result.order_id);
    println!("  Status: {:?}", modify_result.status);
    println!();

    // Cancel an order
    println!("=== Order Cancellation ===\n");

    println!("Cancelling sell order...");

    let cancel_result = connector
        .cancel_order(&sell_response.order_id)
        .await
        .expect("Failed to cancel order");

    println!("  Status: {:?}", cancel_result.status);
    println!();

    // Check open orders after cancellation
    let open_orders = connector
        .get_open_orders("BTC-USD")
        .await
        .expect("Failed to get open orders");

    println!("Open Orders after cancellation: {}", open_orders.len());
    println!();

    // Cancel all remaining orders
    println!("=== Cancel All Orders ===\n");

    let cancelled = connector
        .cancel_all_orders("BTC-USD")
        .await
        .expect("Failed to cancel all orders");

    println!("Cancelled {} orders", cancelled.len());

    let open_orders = connector
        .get_open_orders("BTC-USD")
        .await
        .expect("Failed to get open orders");

    println!("Open Orders after cancel all: {}", open_orders.len());
}
