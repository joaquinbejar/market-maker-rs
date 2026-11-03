//! Inventory management and position tracking example.
//!
//! Demonstrates how to:
//! - Track inventory positions
//! - Update positions with fills
//! - Calculate PnL
//! - Handle position flips
//!
//! Run with: `cargo run --example inventory_management`

use market_maker_rs::prelude::*;

fn main() {
    println!("=== Inventory Management Demo ===\n");

    let mut inventory = InventoryPosition::new();
    let mut pnl = PnL::new();

    println!("Initial State:");
    println!(
        "  Position: {} ({})",
        inventory.quantity,
        if inventory.is_flat() {
            "FLAT"
        } else {
            "NOT FLAT"
        }
    );
    println!();

    // === Scenario 1: Build Long Position ===
    println!("=== Scenario 1: Building Long Position ===\n");

    println!("Trade 1: Buy 10 units at $100.00");
    inventory
        .update_fill(dec!(10.0), dec!(100.0), 1000)
        .unwrap();
    println!("  Position: {:.1} units", inventory.quantity);
    println!("  Avg Entry: ${:.2}", inventory.avg_entry_price);
    println!(
        "  Status: {}",
        if inventory.is_long() { "LONG" } else { "?" }
    );
    println!();

    println!("Trade 2: Buy 5 units at $101.00");
    inventory.update_fill(dec!(5.0), dec!(101.0), 2000).unwrap();
    println!("  Position: {:.1} units", inventory.quantity);
    println!("  Avg Entry: ${:.2}", inventory.avg_entry_price);
    println!(
        "  Explanation: (10×100 + 5×101) / 15 = ${:.2}",
        (dec!(10.0) * dec!(100.0) + dec!(5.0) * dec!(101.0)) / dec!(15.0)
    );
    println!();

    // Update market price and check PnL
    let current_price = dec!(102.0);
    let unrealized = inventory.unrealized_pnl(current_price);
    pnl.set_unrealized(unrealized);

    println!("Market moves to ${:.2}", current_price);
    println!("  Unrealized PnL: ${:.2}", pnl.unrealized);
    println!(
        "  Calculation: {:.1} × (${:.2} - ${:.2}) = ${:.2}",
        inventory.quantity, current_price, inventory.avg_entry_price, unrealized
    );
    println!();

    // === Scenario 2: Reduce Position ===
    println!("=== Scenario 2: Reducing Position ===\n");

    println!("Trade 3: Sell 8 units at $103.00");
    let sell_qty = dec!(8.0);
    let sell_price = dec!(103.0);

    // Calculate realized PnL for this trade
    let realized_this_trade = sell_qty * (sell_price - inventory.avg_entry_price);
    inventory.update_fill(-sell_qty, sell_price, 3000).unwrap();
    pnl.add_realized(realized_this_trade);
    pnl.set_unrealized(inventory.unrealized_pnl(current_price));

    println!("  Position: {:.1} units", inventory.quantity);
    println!("  Avg Entry: ${:.2} (unchanged)", inventory.avg_entry_price);
    println!("  Realized PnL: ${:.2}", realized_this_trade);
    println!("  Total Realized: ${:.2}", pnl.realized);
    println!("  Unrealized: ${:.2}", pnl.unrealized);
    println!("  Total PnL: ${:.2}", pnl.total);
    println!();

    // === Scenario 3: Flip Position ===
    println!("=== Scenario 3: Flipping Position (Long → Short) ===\n");

    println!("Trade 4: Sell 15 units at $104.00");
    let flip_qty = dec!(-15.0);
    let flip_price = dec!(104.0);

    // Close long and go short
    let closing_qty = inventory.quantity;
    let realized_close = closing_qty * (flip_price - inventory.avg_entry_price);

    inventory.update_fill(flip_qty, flip_price, 4000).unwrap();
    pnl.add_realized(realized_close);
    pnl.set_unrealized(inventory.unrealized_pnl(current_price));

    println!("  Position: {:.1} units", inventory.quantity);
    println!(
        "  Avg Entry: ${:.2} (reset to fill price)",
        inventory.avg_entry_price
    );
    println!(
        "  Status: {}",
        if inventory.is_short() { "SHORT" } else { "?" }
    );
    println!("  Realized from close: ${:.2}", realized_close);
    println!("  Total Realized: ${:.2}", pnl.realized);
    println!("  Unrealized: ${:.2}", pnl.unrealized);
    println!();

    // === Scenario 4: Flatten Position ===
    println!("=== Scenario 4: Closing Short Position ===\n");

    let current_price = dec!(103.0);
    println!("Market moves to ${:.2}", current_price);
    println!(
        "  Unrealized PnL: ${:.2}",
        inventory.unrealized_pnl(current_price)
    );
    println!();

    println!(
        "Trade 5: Buy {} units at $103.00 (flatten)",
        inventory.quantity.abs()
    );
    let close_qty = -inventory.quantity;
    let close_price = dec!(103.0);
    let realized_final = -inventory.quantity * (close_price - inventory.avg_entry_price);

    inventory.update_fill(close_qty, close_price, 5000).unwrap();
    pnl.add_realized(realized_final);
    pnl.set_unrealized(Decimal::ZERO);

    println!(
        "  Position: {:.1} units ({})",
        inventory.quantity,
        if inventory.is_flat() {
            "FLAT"
        } else {
            "NOT FLAT"
        }
    );
    println!("  Realized from close: ${:.2}", realized_final);
    println!();

    // === Final Summary ===
    println!("=== Final Summary ===");
    println!("  Total Trades: 5");
    println!("  Final Position: {} (FLAT)", inventory.quantity);
    println!("  Total Realized PnL: ${:.2}", pnl.realized);
    println!("  Total Unrealized PnL: ${:.2}", pnl.unrealized);
    println!("  Total PnL: ${:.2}", pnl.total);
}
