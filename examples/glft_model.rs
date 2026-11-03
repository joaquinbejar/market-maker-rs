//! GLFT Model Example
//!
//! Quotes of Guéant, Lehalle and Fernandez-Tapia (arXiv:1105.3115) with the
//! parameters of the paper's Figure 1: σ = 0.3 tick/√s, A = 0.9 /s,
//! k = 0.3 /tick, γ = 0.01 /tick, T = 600 s, Q = 30. Prices are in ticks
//! around a mid of 1000 ticks.
//!
//! Run with: `cargo run --example glft_model`

use market_maker_rs::prelude::*;
use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy};
use rust_decimal::prelude::FromPrimitive;

const SECONDS_PER_YEAR: f64 = 31_536_000.0;

fn show(side: Option<Decimal>, mid: Decimal, bid: bool) -> String {
    match side {
        Some(price) => {
            let distance = if bid { mid - price } else { price - mid };
            format!("{:>9.4} (δ {:.4})", price, distance)
        }
        None => format!("{:>20}", "none"),
    }
}

fn main() -> MMResult<()> {
    println!("=== GLFT Model Example ===\n");

    let config = GLFTConfig::new(
        dec!(0.01), // risk_aversion γ, per tick
        dec!(0.3),  // order_intensity_k, per tick
        dec!(0.9),  // intensity_scale_a, fills per second
        30,         // max_inventory_lots Q
        dec!(1),    // lot_size
        600_000,    // terminal_time: 600 s
        dec!(0),    // min_spread
    )?;
    // σ = 0.3 tick/√s, expressed per √year as the crate expects.
    let sigma = Decimal::from_f64(0.3 * SECONDS_PER_YEAR.sqrt())
        .ok_or_else(|| MMError::NumericalError("sigma".to_string()))?;
    let mid = dec!(1000);

    println!(
        "γ = {}, k = {}, A = {}, Q = {}, T = {} ms, σ = {sigma:.2} /√year\n",
        config.risk_aversion,
        config.order_intensity_k,
        config.intensity_scale_a,
        config.max_inventory_lots,
        config.terminal_time,
    );

    println!("--- Quotes at t = 0 (T - t = 600 s) ---\n");
    println!(
        "{:>4} | {:>20} | {:>20} | closed form δ_b / δ_a",
        "q", "bid", "ask"
    );
    for q in [-30, -20, -10, -5, -1, 0, 1, 5, 10, 20, 30] {
        let inventory = Decimal::from(q);
        let exact = GLFTStrategy::calculate_optimal_quotes(mid, inventory, &config, sigma, 0)?;
        let closed = GLFTStrategy::closed_form_quotes(mid, inventory, &config, sigma)?;
        let cb = closed
            .bid
            .map_or("none".to_string(), |b| format!("{:.4}", mid - b));
        let ca = closed
            .ask
            .map_or("none".to_string(), |a| format!("{:.4}", a - mid));
        println!(
            "{q:>4} | {} | {} | {cb} / {ca}",
            show(exact.bid, mid, true),
            show(exact.ask, mid, false)
        );
    }

    println!("\n--- Bid distance over time, q = 5 (Figure 1 shape) ---\n");
    for t_ms in [0_u64, 300_000, 500_000, 590_000, 599_000, 600_000] {
        let quotes = GLFTStrategy::calculate_optimal_quotes(mid, dec!(5), &config, sigma, t_ms)?;
        println!(
            "  T - t = {:>5.1} s: bid {}",
            (600_000 - t_ms) as f64 / 1000.0,
            show(quotes.bid, mid, true)
        );
    }

    println!("\nFar from T the quotes are flat in time (the spectral limit) and");
    println!("close to the closed form; near T they collapse to (1/γ) ln(1 + γ/k).");
    Ok(())
}
