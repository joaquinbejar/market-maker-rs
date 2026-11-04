//! Complete example of using the market-maker-rs library.
//!
//! This example demonstrates:
//! - Creating a strategy configuration
//! - Tracking inventory and PnL
//! - Calculating optimal quotes based on market conditions
//! - Simulating fills and position updates
//!
//! Run with: `cargo run --example full_strategy`

use market_maker_rs::prelude::*;
use market_maker_rs::strategy::avellaneda_stoikov::*;

fn main() {
    println!("=== Market Making Strategy Demo ===\n");

    // Step 1: Configure strategy parameters
    let config = StrategyConfig::new(
        dec!(0.1),  // risk_aversion (gamma)
        dec!(1.5),  // order_intensity (k)
        3600000,    // terminal_time: 1 hour from now
        dec!(0.01), // min_spread: $0.01
    )
    .expect("Failed to create strategy config");

    println!("Strategy Configuration:");
    println!("  Risk Aversion (γ): {}", config.risk_aversion);
    println!("  Order Intensity (k): {}", config.order_intensity);
    println!("  Terminal Time: {} ms", config.terminal_time);
    println!("  Min Spread: ${}\n", config.min_spread);

    // Step 2: Initialize market state
    let mut market_state = MarketState::new(
        dec!(100.0), // mid_price: $100
        dec!(0.2),   // volatility: 20% annualized
        0,           // current timestamp
    );

    println!("Initial Market State:");
    println!("  Mid Price: ${}", market_state.mid_price);
    println!("  Volatility: {}%\n", market_state.volatility * dec!(100.0));

    // Step 3: Initialize position tracking
    let mut inventory = InventoryPosition::new();
    let mut pnl = PnL::new();

    println!("Initial Position: Flat (0 units)\n");

    // Step 4: Calculate initial quotes
    let time_remaining = config.terminal_time;

    let (bid, ask) = calculate_optimal_quotes(
        market_state.mid_price,
        inventory.quantity,
        config.risk_aversion,
        market_state.volatility,
        time_remaining,
        config.order_intensity,
    )
    .expect("Failed to calculate quotes");

    println!("Initial Quotes:");
    println!("  Bid: ${:.2}", bid);
    println!("  Ask: ${:.2}", ask);
    println!("  Spread: ${:.2}\n", ask - bid);

    // Step 5: Simulate market activity - Buy at ask
    println!("=== Trade 1: Buy 10 units at ${:.2} ===", ask);
    inventory.update_fill(dec!(10.0), ask, 1000).unwrap();
    pnl.set_unrealized(inventory.unrealized_pnl(market_state.mid_price));

    println!("Position after trade:");
    println!("  Quantity: {}", inventory.quantity);
    println!("  Avg Entry: ${:.2}", inventory.avg_entry_price);
    println!("  Unrealized PnL: ${:.2}\n", pnl.unrealized);

    // Step 6: Recalculate quotes with new inventory
    let (bid_long, ask_long) = calculate_optimal_quotes(
        market_state.mid_price,
        inventory.quantity,
        config.risk_aversion,
        market_state.volatility,
        time_remaining - 1000,
        config.order_intensity,
    )
    .expect("Failed to calculate quotes");

    println!("New Quotes (with long position):");
    println!(
        "  Bid: ${:.2} (moved {} from ${:.2})",
        bid_long,
        if bid_long < bid { "down" } else { "up" },
        bid
    );
    println!(
        "  Ask: ${:.2} (moved {} from ${:.2})",
        ask_long,
        if ask_long < ask { "down" } else { "up" },
        ask
    );
    println!("  Note: With positive inventory, both quotes shift down to incentivize selling\n");

    // Step 7: Market moves up
    market_state.mid_price = dec!(102.0);
    println!(
        "=== Market Update: Mid Price moved to ${} ===",
        market_state.mid_price
    );

    pnl.set_unrealized(inventory.unrealized_pnl(market_state.mid_price));
    println!("  Unrealized PnL: ${:.2}\n", pnl.unrealized);

    // Step 8: Sell some inventory at bid
    let (bid_new, ask_new) = calculate_optimal_quotes(
        market_state.mid_price,
        inventory.quantity,
        config.risk_aversion,
        market_state.volatility,
        time_remaining - 2000,
        config.order_intensity,
    )
    .expect("Failed to calculate quotes");

    println!("Current Quotes:");
    println!("  Bid: ${:.2}", bid_new);
    println!("  Ask: ${:.2}\n", ask_new);

    println!("=== Trade 2: Sell 5 units at ${:.2} ===", bid_new);

    // Calculate realized PnL for this trade
    let realized_pnl_trade = dec!(-5.0) * (bid_new - inventory.avg_entry_price);
    inventory.update_fill(dec!(-5.0), bid_new, 2000).unwrap();
    pnl.add_realized(realized_pnl_trade);
    pnl.set_unrealized(inventory.unrealized_pnl(market_state.mid_price));

    println!("Position after trade:");
    println!("  Quantity: {}", inventory.quantity);
    println!("  Avg Entry: ${:.2}", inventory.avg_entry_price);
    println!("  Realized PnL: ${:.2}", pnl.realized);
    println!("  Unrealized PnL: ${:.2}", pnl.unrealized);
    println!("  Total PnL: ${:.2}\n", pnl.total);

    // Step 9: Close remaining position
    let (bid_final, _) = calculate_optimal_quotes(
        market_state.mid_price,
        inventory.quantity,
        config.risk_aversion,
        market_state.volatility,
        time_remaining - 3000,
        config.order_intensity,
    )
    .expect("Failed to calculate quotes");

    println!(
        "=== Trade 3: Close position - Sell {} units at ${:.2} ===",
        inventory.quantity, bid_final
    );

    let final_realized = inventory.quantity * (bid_final - inventory.avg_entry_price);
    inventory
        .update_fill(-inventory.quantity, bid_final, 3000)
        .unwrap();
    pnl.add_realized(final_realized);
    pnl.set_unrealized(Decimal::ZERO);

    println!("\n=== Final Results ===");
    println!(
        "Position: {} (Flat)",
        if inventory.is_flat() {
            "FLAT"
        } else {
            "NOT FLAT"
        }
    );
    println!("Total Realized PnL: ${:.2}", pnl.realized);
    println!("Total Unrealized PnL: ${:.2}", pnl.unrealized);
    println!("Total PnL: ${:.2}", pnl.total);

    // Step 10: Demonstrate inventory influence
    println!("\n=== Inventory Influence on Quotes ===");
    let scenarios = vec![
        ("Flat (0)", dec!(0.0)),
        ("Long (+20)", dec!(20.0)),
        ("Short (-20)", dec!(-20.0)),
    ];

    for (label, qty) in scenarios {
        let (b, a) = calculate_optimal_quotes(
            dec!(100.0),
            qty,
            config.risk_aversion,
            dec!(0.2),
            3600000,
            config.order_intensity,
        )
        .unwrap();
        let mid = (b + a) / Decimal::from(2);
        println!(
            "{:12} => Bid: ${:.2}, Ask: ${:.2}, Mid: ${:.2}",
            label, b, a, mid
        );
    }

    println!("\nNote: Long positions push quotes down (incentivize selling)");
    println!("      Short positions push quotes up (incentivize buying)");
}
