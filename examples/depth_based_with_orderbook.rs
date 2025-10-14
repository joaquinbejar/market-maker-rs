//! Depth-based strategy integration with orderbook-rs.
//!
//! This example demonstrates how to integrate the `DepthBasedOffering` strategy
//! with the `orderbook-rs` crate for real order book management.
//!
//! Note: This example shows the conceptual integration with orderbook-rs v0.4.5.
//! In production, you would adapt this to your specific orderbook implementation.
//!
//! It demonstrates:
//! 1. Creating an OrderBook from orderbook-rs
//! 2. Simulating order book data
//! 3. Analyzing depth from orderbook snapshots
//! 4. Applying the depth-based strategy
//! 5. Calculating optimal order placement
//!
//! Run with: `cargo run --example depth_based_with_orderbook`

use market_maker_rs::dec;
use market_maker_rs::strategy::depth_based::DepthBasedOffering;
use orderbook_rs::prelude::*;

fn main() {
    println!("=== Depth-Based Strategy Integration with orderbook-rs ===\n");

    // === Setup ===
    println!("=== Setup ===\n");

    // Create orderbook for BTC/USD
    // Note: OrderBook requires type parameters for the price/quantity types
    let orderbook: OrderBook<OrderId> = OrderBook::new("BTC/USD");
    println!("Created OrderBook instance from orderbook-rs");
    println!("  Symbol: {}", orderbook.symbol());
    println!();

    // Initialize strategy
    let max_exposure = dec!(100.0);
    let target_depth = dec!(50.0);
    let strategy = DepthBasedOffering::new(max_exposure, target_depth);

    println!("Strategy Configuration:");
    println!("  Max Exposure: {} units", strategy.max_exposure());
    println!("  Target Depth: {} units", strategy.target_depth());
    println!();

    // === Simulated Order Book Data ===
    println!("=== Simulated Order Book Data ===\n");

    println!("NOTE: In production, you would:");
    println!("  1. Receive market data from exchange");
    println!("  2. Populate orderbook using add_limit_order() or similar");
    println!("  3. Get snapshot using orderbook.create_snapshot(depth)");
    println!();

    println!("For this example, we simulate orderbook data:");
    println!();

    // Simulate order book data as it would come from a snapshot
    println!("Bid Orders (buy side):");
    let bid_levels = vec![
        (dec!(99.95), dec!(10.0)),
        (dec!(99.90), dec!(15.0)),
        (dec!(99.85), dec!(20.0)),
        (dec!(99.80), dec!(25.0)),
        (dec!(99.75), dec!(30.0)),
    ];

    for (price, qty) in &bid_levels {
        println!("  {} @ {}: {} units", price, price, qty);
    }
    println!();

    println!("Ask Orders (sell side):");
    let ask_levels = vec![
        (dec!(100.05), dec!(10.0)),
        (dec!(100.10), dec!(15.0)),
        (dec!(100.15), dec!(20.0)),
        (dec!(100.20), dec!(25.0)),
        (dec!(100.25), dec!(30.0)),
    ];

    for (price, qty) in &ask_levels {
        println!("  {} @ {}: {} units", price, price, qty);
    }
    println!();

    // === Analyze Order Book Depth ===
    println!("=== Analyzing Order Book Depth ===\n");

    println!("Best Bid: {}", bid_levels[0].0);
    println!("Best Ask: {}", ask_levels[0].0);
    println!("Spread: {}", ask_levels[0].0 - bid_levels[0].0);
    println!();

    // Calculate cumulative depth on bid side
    println!("Bid Side Depth Analysis:");
    let mut bid_cumulative = dec!(0.0);
    let mut target_bid_price = dec!(0.0);
    let mut target_bid_found = false;

    for (price, qty) in &bid_levels {
        bid_cumulative += qty;
        let depth_indicator = if bid_cumulative >= target_depth && !target_bid_found {
            target_bid_price = *price;
            target_bid_found = true;
            " ← TARGET DEPTH REACHED"
        } else {
            ""
        };
        println!(
            "  {} @ {}: {} units (cumulative: {}){}",
            price, price, qty, bid_cumulative, depth_indicator
        );
    }
    println!();

    // Calculate cumulative depth on ask side
    println!("Ask Side Depth Analysis:");
    let mut ask_cumulative = dec!(0.0);
    let mut target_ask_price = dec!(0.0);
    let mut target_ask_found = false;

    for (price, qty) in &ask_levels {
        ask_cumulative += qty;
        let depth_indicator = if ask_cumulative >= target_depth && !target_ask_found {
            target_ask_price = *price;
            target_ask_found = true;
            " ← TARGET DEPTH REACHED"
        } else {
            ""
        };
        println!(
            "  {} @ {}: {} units (cumulative: {}){}",
            price, price, qty, ask_cumulative, depth_indicator
        );
    }
    println!();

    // === Apply Strategy ===
    println!("=== Applying Depth-Based Strategy ===\n");

    // Current inventory position
    let current_inventory = dec!(15.0);
    println!("Current Inventory: +{} units (LONG)", current_inventory);
    println!();

    // Calculate order sizes
    let ask_size = strategy.calculate_ask_size(current_inventory);
    let bid_size = strategy.calculate_bid_size(current_inventory);

    println!("Calculated Order Sizes:");
    println!(
        "  Ask Size: {} units (increased to reduce long position)",
        ask_size
    );
    println!(
        "  Bid Size: {} units (decreased to avoid more long exposure)",
        bid_size
    );
    println!();

    // Calculate price placements
    let tick_size = dec!(0.01);

    println!("Price Placement Calculation:");
    println!();

    // Bid price calculation
    if target_bid_found {
        let bid_adjustment = strategy.price_adjustment(bid_cumulative, tick_size, false);
        let final_bid_price = target_bid_price + bid_adjustment;

        println!("  Bid Side:");
        println!("    Target depth reached at: {}", target_bid_price);
        println!("    Price adjustment: +{}", bid_adjustment);
        println!("    Final bid price: {}", final_bid_price);
        println!("    Reasoning: Place one tick inside to be first in queue");
    } else {
        println!("  Bid Side: Target depth not reached");
    }
    println!();

    // Ask price calculation
    if target_ask_found {
        let ask_adjustment = strategy.price_adjustment(ask_cumulative, tick_size, true);
        let final_ask_price = target_ask_price + ask_adjustment;

        println!("  Ask Side:");
        println!("    Target depth reached at: {}", target_ask_price);
        println!("    Price adjustment: {}", ask_adjustment);
        println!("    Final ask price: {}", final_ask_price);
        println!("    Reasoning: Place one tick inside to be first in queue");
    } else {
        println!("  Ask Side: Target depth not reached");
    }
    println!();

    // === Final Strategy Recommendations ===
    println!("=== Strategy Recommendations ===\n");

    if target_bid_found {
        let bid_adjustment = strategy.price_adjustment(bid_cumulative, tick_size, false);
        let final_bid_price = target_bid_price + bid_adjustment;

        println!("Recommended BUY Order:");
        println!("  Side: BUY");
        println!("  Price: {}", final_bid_price);
        println!("  Size: {} units", bid_size);
        println!(
            "  Rationale: Inventory (+{}) is long, reduce bid size",
            current_inventory
        );
    }
    println!();

    if target_ask_found {
        let ask_adjustment = strategy.price_adjustment(ask_cumulative, tick_size, true);
        let final_ask_price = target_ask_price + ask_adjustment;

        println!("Recommended SELL Order:");
        println!("  Side: SELL");
        println!("  Price: {}", final_ask_price);
        println!("  Size: {} units", ask_size);
        println!(
            "  Rationale: Inventory (+{}) is long, increase ask size to reduce",
            current_inventory
        );
    }
    println!();

    println!("To place these orders in production:");
    println!("```rust");
    println!("// Using orderbook-rs API:");
    println!("orderbook.add_limit_order(");
    println!("    order_id,");
    println!("    price,      // Convert Decimal to appropriate type");
    println!("    quantity,   // Convert Decimal to appropriate type");
    println!("    Side::Buy,  // or Side::Sell");
    println!("    TimeInForce::GTC,");
    println!("    Default::default(),");
    println!(");");
    println!("```");
    println!();

    // === Integration Summary ===
    println!("=== Integration Summary ===\n");

    println!("This example demonstrated:");
    println!("  1. ✓ Creating OrderBook instance from orderbook-rs");
    println!("  2. ✓ Simulating order book depth data");
    println!("  3. ✓ Analyzing cumulative depth at price levels");
    println!("  4. ✓ Initializing DepthBasedOffering strategy");
    println!("  5. ✓ Calculating order sizes based on inventory");
    println!("  6. ✓ Finding target depth price levels");
    println!("  7. ✓ Determining optimal price placements");
    println!("  8. ✓ Generating strategy recommendations");
    println!();

    println!("Key Components:");
    println!("  - orderbook-rs::OrderBook - Order book management");
    println!("  - DepthBasedOffering - Strategy calculator");
    println!("  - calculate_ask_size() - Size based on inventory");
    println!("  - calculate_bid_size() - Size based on inventory");
    println!("  - price_adjustment() - Depth-based pricing");
    println!();

    println!("Production Integration Notes:");
    println!("  • Receive market data from exchange WebSocket");
    println!("  • Update orderbook with live data");
    println!("  • Create snapshots periodically");
    println!("  • Extract price levels from snapshot");
    println!("  • Apply strategy calculations");
    println!("  • Convert Decimal types as needed for your orderbook API");
    println!("  • Place orders via exchange REST API");
    println!();

    println!("✓ Depth-based strategy integration example completed successfully!");
}
