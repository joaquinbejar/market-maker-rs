//! Real-time market making simulation.
//!
//! Simulates a market making session with:
//! - Market price movements
//! - Order fills
//! - Position management
//! - PnL tracking
//!
//! Run with: `cargo run --example real_time_simulation`

use market_maker_rs::prelude::*;
use market_maker_rs::strategy::avellaneda_stoikov::*;

struct MarketMaker {
    config: StrategyConfig,
    inventory: InventoryPosition,
    pnl: PnL,
    market_state: MarketState,
}

impl MarketMaker {
    fn new(config: StrategyConfig, initial_mid: Decimal, volatility: Decimal) -> Self {
        Self {
            config,
            inventory: InventoryPosition::new(),
            pnl: PnL::new(),
            market_state: MarketState::new(initial_mid, volatility, 0),
        }
    }

    fn generate_quotes(&self, time_elapsed: u64) -> (Decimal, Decimal) {
        let time_remaining = self.config.terminal_time.saturating_sub(time_elapsed);

        calculate_optimal_quotes(
            self.market_state.mid_price,
            self.inventory.quantity,
            self.config.risk_aversion,
            self.market_state.volatility,
            time_remaining,
            self.config.order_intensity,
        )
        .unwrap_or((
            self.market_state.mid_price * dec!(0.99),
            self.market_state.mid_price * dec!(1.01),
        ))
    }

    fn handle_fill(&mut self, quantity: Decimal, price: Decimal, timestamp: u64) {
        let old_qty = self.inventory.quantity;

        // Calculate realized PnL if reducing position
        if (old_qty > Decimal::ZERO && quantity < Decimal::ZERO)
            || (old_qty < Decimal::ZERO && quantity > Decimal::ZERO)
        {
            let closing_qty = old_qty.abs().min(quantity.abs());
            let direction = if old_qty > Decimal::ZERO {
                Decimal::ONE
            } else {
                -Decimal::ONE
            };
            let realized = closing_qty * direction * (price - self.inventory.avg_entry_price);
            self.pnl.add_realized(realized);
        }

        self.inventory
            .update_fill(quantity, price, timestamp)
            .unwrap();
        self.pnl
            .set_unrealized(self.inventory.unrealized_pnl(self.market_state.mid_price));
    }

    fn update_market(&mut self, new_mid: Decimal, timestamp: u64) {
        self.market_state.mid_price = new_mid;
        self.market_state.timestamp = timestamp;
        self.pnl
            .set_unrealized(self.inventory.unrealized_pnl(new_mid));
    }
}

fn main() {
    println!("=== Market Making Simulation ===\n");

    let config = StrategyConfig::new(dec!(0.1), dec!(1.5), 3600000, dec!(0.01))
        .expect("Failed to create config");

    let mut mm = MarketMaker::new(config, dec!(100.0), dec!(0.2));

    println!("Initial Setup:");
    println!("  Mid Price: ${:.2}", mm.market_state.mid_price);
    println!(
        "  Volatility: {:.1}%",
        mm.market_state.volatility * dec!(100.0)
    );
    println!("  Terminal Time: {} ms", mm.config.terminal_time);
    println!();

    // Simulation events
    let events = vec![
        (0, "Quote", Decimal::ZERO, Decimal::ZERO),
        (1000, "Fill", dec!(-5.0), dec!(100.65)), // Sell at ask
        (2000, "Market", Decimal::ZERO, dec!(101.0)),
        (3000, "Quote", Decimal::ZERO, Decimal::ZERO),
        (4000, "Fill", dec!(3.0), dec!(100.85)), // Buy at bid
        (5000, "Market", Decimal::ZERO, dec!(102.0)),
        (6000, "Fill", dec!(4.0), dec!(101.35)), // Buy at bid
        (8000, "Market", Decimal::ZERO, dec!(101.5)),
        (9000, "Fill", dec!(-2.0), dec!(102.15)), // Sell at ask
    ];

    for (time_ms, event_type, qty, price) in events {
        match event_type {
            "Quote" => {
                let (bid, ask) = mm.generate_quotes(time_ms);
                println!(
                    "[{}ms] Quotes: Bid ${:.2} / Ask ${:.2} | Pos: {:.0} | PnL: ${:.2}",
                    time_ms, bid, ask, mm.inventory.quantity, mm.pnl.total
                );
            }
            "Fill" => {
                let side = if qty > Decimal::ZERO { "BUY" } else { "SELL" };
                mm.handle_fill(qty, price, time_ms);
                println!(
                    "[{}ms] Fill: {} {:.0} @ ${:.2} | Pos: {:.0} | PnL: ${:.2}",
                    time_ms,
                    side,
                    qty.abs(),
                    price,
                    mm.inventory.quantity,
                    mm.pnl.total
                );
            }
            "Market" => {
                mm.update_market(price, time_ms);
                println!(
                    "[{}ms] Market: Mid → ${:.2} | Unrealized PnL: ${:.2}",
                    time_ms, price, mm.pnl.unrealized
                );
            }
            _ => {}
        }
    }

    println!("\n=== Final Summary ===");
    println!("Position: {:.0} units", mm.inventory.quantity);
    println!("Realized PnL: ${:.2}", mm.pnl.realized);
    println!("Unrealized PnL: ${:.2}", mm.pnl.unrealized);
    println!("Total PnL: ${:.2}", mm.pnl.total);
}
