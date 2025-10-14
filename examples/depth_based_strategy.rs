//! Depth-based offering strategy example.
//!
//! This example demonstrates the `DepthBasedOffering` strategy which places orders
//! at a specific depth in the order book, adjusting sizes based on current inventory.
//!
//! The strategy:
//! 1. Analyzes order book depth data
//! 2. Calculates order sizes based on max exposure and current position
//! 3. Adjusts sizes to incentivize reducing inventory risk
//! 4. Provides price adjustment recommendations based on order book depth
//!
//! Run with: `cargo run --example depth_based_strategy`

use market_maker_rs::dec;
use market_maker_rs::strategy::depth_based::DepthBasedOffering;

fn main() {
    println!("=== Depth-Based Offering Strategy Example ===\n");

    // Strategy parameters
    let max_exposure = dec!(100.0); // Maximum position size
    let target_depth = dec!(50.0); // Target depth in order book

    println!("Strategy Configuration:");
    println!("  Max Exposure: {} units", max_exposure);
    println!("  Target Depth: {} units", target_depth);
    println!();

    // Create strategy
    let strategy = DepthBasedOffering::new(max_exposure, target_depth);

    println!("Strategy Properties:");
    println!("  Max Exposure: {}", strategy.max_exposure());
    println!("  Target Depth: {}", strategy.target_depth());
    println!();

    // === Scenario 1: Flat Inventory (No Position) ===
    println!("=== Scenario 1: Flat Inventory (No Position) ===\n");

    let inventory = dec!(0.0);
    println!("Current Position: {} units", inventory);
    println!();

    let ask_size = strategy.calculate_ask_size(inventory);
    let bid_size = strategy.calculate_bid_size(inventory);

    println!("Order Sizes:");
    println!("  Ask (Sell) Size: {} units", ask_size);
    println!("  Bid (Buy) Size:  {} units", bid_size);
    println!();

    println!("Analysis:");
    println!("  - Flat position means equal exposure on both sides");
    println!("  - Ask size = Bid size = Max Exposure");
    println!("  - Strategy is balanced, ready to provide liquidity");
    println!();

    // === Scenario 2: Long Position ===
    println!("=== Scenario 2: Long Position (+30 units) ===\n");

    let inventory_long = dec!(30.0);
    println!("Current Position: +{} units (LONG)", inventory_long);
    println!();

    let ask_size_long = strategy.calculate_ask_size(inventory_long);
    let bid_size_long = strategy.calculate_bid_size(inventory_long);

    println!("Order Sizes:");
    println!("  Ask (Sell) Size: {} units", ask_size_long);
    println!("  Bid (Buy) Size:  {} units", bid_size_long);
    println!();

    println!("Analysis:");
    println!("  - Long position increases risk");
    println!(
        "  - Ask size = max_exposure + position = {} + {} = {}",
        max_exposure, inventory_long, ask_size_long
    );
    println!(
        "  - Bid size = max_exposure - position = {} - {} = {}",
        max_exposure, inventory_long, bid_size_long
    );
    println!("  - Larger ask size incentivizes selling to reduce position");
    println!("  - Smaller bid size prevents increasing long exposure");
    println!();

    // === Scenario 3: Short Position ===
    println!("=== Scenario 3: Short Position (-40 units) ===\n");

    let inventory_short = dec!(-40.0);
    println!("Current Position: {} units (SHORT)", inventory_short);
    println!();

    let ask_size_short = strategy.calculate_ask_size(inventory_short);
    let bid_size_short = strategy.calculate_bid_size(inventory_short);

    println!("Order Sizes:");
    println!("  Ask (Sell) Size: {} units", ask_size_short);
    println!("  Bid (Buy) Size:  {} units", bid_size_short);
    println!();

    println!("Analysis:");
    println!("  - Short position needs covering");
    println!(
        "  - Ask size = max_exposure + position = {} + ({}) = {}",
        max_exposure, inventory_short, ask_size_short
    );
    println!(
        "  - Bid size = max_exposure - position = {} - ({}) = {}",
        max_exposure, inventory_short, bid_size_short
    );
    println!("  - Smaller ask size prevents increasing short exposure");
    println!("  - Larger bid size incentivizes buying to cover short");
    println!();

    // === Scenario 4: Extreme Long Position ===
    println!("=== Scenario 4: Extreme Long Position (+80 units) ===\n");

    let inventory_extreme = dec!(80.0);
    println!(
        "Current Position: +{} units (EXTREME LONG)",
        inventory_extreme
    );
    println!();

    let ask_size_extreme = strategy.calculate_ask_size(inventory_extreme);
    let bid_size_extreme = strategy.calculate_bid_size(inventory_extreme);

    println!("Order Sizes:");
    println!("  Ask (Sell) Size: {} units", ask_size_extreme);
    println!("  Bid (Buy) Size:  {} units", bid_size_extreme);
    println!();

    println!("Analysis:");
    println!("  - Position at 80% of max exposure");
    println!("  - Ask size significantly larger: {}", ask_size_extreme);
    println!("  - Bid size significantly smaller: {}", bid_size_extreme);
    println!("  - Strategy heavily incentivizes position reduction");
    println!();

    // === Price Adjustment Examples ===
    println!("=== Price Adjustment Based on Order Book Depth ===\n");

    let tick_size = dec!(0.01);
    println!("Tick Size: {}", tick_size);
    println!("Target Depth: {} units", target_depth);
    println!();

    // Scenario: Depth target reached
    println!("Case 1: Target Depth Reached (50 units)");
    let depth_reached = dec!(50.0);
    let ask_adjustment = strategy.price_adjustment(depth_reached, tick_size, true);
    let bid_adjustment = strategy.price_adjustment(depth_reached, tick_size, false);

    println!("  Cumulative Depth: {} units", depth_reached);
    println!(
        "  Ask Price Adjustment: {} (place inside by one tick)",
        ask_adjustment
    );
    println!(
        "  Bid Price Adjustment: +{} (place inside by one tick)",
        bid_adjustment
    );
    println!("  → Place orders one tick better than the depth level");
    println!();

    // Scenario: Depth target not reached
    println!("Case 2: Target Depth Not Reached (30 units)");
    let depth_not_reached = dec!(30.0);
    let ask_adjustment_none = strategy.price_adjustment(depth_not_reached, tick_size, true);
    let bid_adjustment_none = strategy.price_adjustment(depth_not_reached, tick_size, false);

    println!("  Cumulative Depth: {} units", depth_not_reached);
    println!("  Ask Price Adjustment: {}", ask_adjustment_none);
    println!("  Bid Price Adjustment: {}", bid_adjustment_none);
    println!("  → No adjustment needed, target depth not yet reached");
    println!();

    // === OrderBook Depth Analysis Example ===
    println!("=== OrderBook Depth Analysis Example ===\n");

    // Simulated order book data
    println!("Simulated Order Book (BTC/USD):");
    println!();

    let bid_levels_data = vec![
        (dec!(99.95), dec!(10.0)),
        (dec!(99.90), dec!(15.0)),
        (dec!(99.85), dec!(20.0)),
        (dec!(99.80), dec!(25.0)),
    ];

    let ask_levels_data = vec![
        (dec!(100.05), dec!(10.0)),
        (dec!(100.10), dec!(15.0)),
        (dec!(100.15), dec!(20.0)),
        (dec!(100.20), dec!(25.0)),
    ];

    // Calculate cumulative depth
    let mut bid_cumulative = dec!(0.0);
    println!("Bid Side Depth:");
    for (price, qty) in &bid_levels_data {
        bid_cumulative += qty;
        println!(
            "  @ {}: {} units (cumulative: {})",
            price, qty, bid_cumulative
        );
    }
    println!();

    let mut ask_cumulative = dec!(0.0);
    println!("Ask Side Depth:");
    for (price, qty) in &ask_levels_data {
        ask_cumulative += qty;
        println!(
            "  @ {}: {} units (cumulative: {})",
            price, qty, ask_cumulative
        );
    }
    println!();

    // Apply strategy with the order book
    let inventory_for_ob = dec!(15.0);
    println!(
        "Applying Strategy with Inventory: +{} units",
        inventory_for_ob
    );
    println!();

    let ask_size_ob = strategy.calculate_ask_size(inventory_for_ob);
    let bid_size_ob = strategy.calculate_bid_size(inventory_for_ob);

    println!("Calculated Order Sizes:");
    println!("  Ask Size: {} units", ask_size_ob);
    println!("  Bid Size: {} units", bid_size_ob);
    println!();

    // Find where to place orders based on depth
    let tick_size_ob = dec!(0.01);

    println!("Price Placement Analysis:");
    println!("  Target Depth: {} units", target_depth);
    println!();

    if ask_cumulative >= target_depth {
        let adjustment = strategy.price_adjustment(ask_cumulative, tick_size_ob, true);
        println!("  Ask side reached target depth at level 100.15");
        println!("    Cumulative depth: {}", ask_cumulative);
        println!("    Price adjustment: {}", adjustment);
        println!("    Suggested ask price: 100.14 (one tick inside)");
    }
    println!();

    if bid_cumulative >= target_depth {
        let adjustment = strategy.price_adjustment(bid_cumulative, tick_size_ob, false);
        println!("  Bid side reached target depth at level 99.85");
        println!("    Cumulative depth: {}", bid_cumulative);
        println!("    Price adjustment: +{}", adjustment);
        println!("    Suggested bid price: 99.86 (one tick inside)");
    }
    println!();

    println!("Strategy Recommendation:");
    println!("  Place BUY  order: {} units @ 99.86", bid_size_ob);
    println!("  Place SELL order: {} units @ 100.14", ask_size_ob);
    println!("  → Orders placed at target depth, one tick inside for best execution");
    println!();

    // === Complete Trading Scenario ===
    println!("=== Complete Trading Scenario ===\n");

    let scenarios = vec![
        ("Initial", dec!(0.0)),
        ("After Buy Fill +25", dec!(25.0)),
        ("After Buy Fill +50", dec!(50.0)),
        ("After Sell Fill +30", dec!(30.0)),
        ("After Sell Fill 0", dec!(0.0)),
        ("After Sell Fill -20", dec!(-20.0)),
    ];

    println!("Tracking order sizes through a trading session:\n");
    println!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Scenario", "Position", "Ask Size", "Bid Size"
    );
    println!("{}", "-".repeat(60));

    for (label, position) in scenarios {
        let ask = strategy.calculate_ask_size(position);
        let bid = strategy.calculate_bid_size(position);
        println!("{:<20} {:>12} {:>12} {:>12}", label, position, ask, bid);
    }
    println!();

    // === Strategy Characteristics ===
    println!("=== Strategy Characteristics ===\n");

    println!("1. Inventory Risk Management:");
    println!("   - Automatically adjusts order sizes based on position");
    println!("   - Incentivizes mean reversion to flat position");
    println!("   - Prevents excessive inventory buildup");
    println!();

    println!("2. Symmetric Exposure:");
    let test_positions = vec![dec!(0.0), dec!(20.0), dec!(-30.0), dec!(50.0)];
    println!("   Ask Size + Bid Size = 2 × Max Exposure (always):");
    for pos in test_positions {
        let ask = strategy.calculate_ask_size(pos);
        let bid = strategy.calculate_bid_size(pos);
        println!(
            "     Position {:>6}: {} + {} = {}",
            pos,
            ask,
            bid,
            ask + bid
        );
    }
    println!();

    println!("3. Depth-Based Pricing:");
    println!("   - Places orders at meaningful liquidity levels");
    println!("   - Adjusts price by one tick when target depth is reached");
    println!("   - Maximizes execution probability while staying competitive");
    println!();

    println!("4. Risk Limits:");
    println!("   - Max exposure caps total position size");
    println!("   - Prevents runaway positions in volatile markets");
    println!("   - Configurable per asset and market conditions");
    println!();

    // === Use Cases ===
    println!("=== Practical Use Cases ===\n");

    println!("✓ Market Making:");
    println!("  - Provide liquidity at specific depth levels");
    println!("  - Manage inventory risk automatically");
    println!("  - Adapt to changing market conditions");
    println!();

    println!("✓ Liquidity Provision:");
    println!("  - Place orders where they're most likely to execute");
    println!("  - Maintain presence at key price levels");
    println!("  - Balance profitability with execution");
    println!();

    println!("✓ Risk Management:");
    println!("  - Automatic position sizing");
    println!("  - Built-in exposure limits");
    println!("  - Gradual position reduction");
    println!();

    // === Integration Example ===
    println!("=== Integration Example ===\n");

    println!("Example workflow:");
    println!("1. Initialize strategy with max_exposure and target_depth");
    println!("2. Get orderbook snapshot from orderbook-rs");
    println!("3. Analyze cumulative depth at each price level");
    println!("4. Calculate order sizes based on current inventory");
    println!("5. Apply price adjustment based on depth reached");
    println!("6. Place orders with calculated size and price");
    println!("7. Update inventory after fills");
    println!("8. Repeat on orderbook updates");
    println!();

    println!("Integration code example:");
    println!("```rust");
    println!("use market_maker_rs::strategy::depth_based::DepthBasedOffering;");
    println!("use market_maker_rs::dec;");
    println!();
    println!("// Initialize strategy");
    println!("let strategy = DepthBasedOffering::new(");
    println!("    dec!(100.0),  // max_exposure");
    println!("    dec!(50.0),   // target_depth");
    println!(");");
    println!();
    println!("// Get order book data from your exchange/source");
    println!("let orderbook_bids = get_orderbook_bids();  // Vec<(price, qty)>");
    println!("let orderbook_asks = get_orderbook_asks();");
    println!("let inventory = get_current_position();");
    println!();
    println!("// Calculate order sizes based on inventory");
    println!("let ask_size = strategy.calculate_ask_size(inventory);");
    println!("let bid_size = strategy.calculate_bid_size(inventory);");
    println!();
    println!("// Analyze bid side depth to find price level");
    println!("let mut bid_cumulative = dec!(0.0);");
    println!("let mut target_bid_price = None;");
    println!();
    println!("for (price, qty) in orderbook_bids.iter() {{");
    println!("    bid_cumulative += qty;");
    println!("    if bid_cumulative >= strategy.target_depth() {{");
    println!("        target_bid_price = Some(*price);");
    println!("        break;");
    println!("    }}");
    println!("}}");
    println!();
    println!("// Apply price adjustment");
    println!("if let Some(price) = target_bid_price {{");
    println!("    let adjustment = strategy.price_adjustment(");
    println!("        bid_cumulative,");
    println!("        dec!(0.01),  // tick_size from exchange");
    println!("        false,       // is_ask=false for bids");
    println!("    );");
    println!("    let final_bid_price = price + adjustment;");
    println!();
    println!("    // Place order via your exchange API");
    println!("    exchange.place_limit_order(");
    println!("        \"BUY\",");
    println!("        final_bid_price,");
    println!("        bid_size,");
    println!("    );");
    println!("}}");
    println!();
    println!("// Repeat for ask side...");
    println!("```");
    println!();

    println!("✓ Depth-based strategy example completed successfully!");
}
