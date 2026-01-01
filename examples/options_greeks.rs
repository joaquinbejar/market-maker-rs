//! Options Greeks calculation example.
//!
//! Demonstrates how to calculate and track Greeks for options positions
//! using the OptionStratLib integration.
//!
//! Run with: `cargo run --example options_greeks --features options`

use market_maker_rs::options::{OptionsAdapter, PortfolioGreeks};
use optionstratlib::model::ExpirationDate;
use optionstratlib::model::option::Options;
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::pos;
use rust_decimal_macros::dec;

fn main() {
    println!("=== Options Greeks Calculation ===\n");

    // Create a call option
    let call_option = Options::new(
        OptionType::European,
        Side::Long,
        "BTC".to_string(),
        pos!(50000.0),                    // strike price
        ExpirationDate::Days(pos!(30.0)), // 30 days to expiry
        pos!(0.6),                        // 60% implied volatility
        pos!(1.0),                        // quantity
        pos!(48000.0),                    // underlying price
        dec!(0.05),                       // 5% risk-free rate
        OptionStyle::Call,
        pos!(0.0), // no dividend
        None,
    );

    println!("Call Option:");
    println!("  Underlying: BTC @ $48,000");
    println!("  Strike: $50,000");
    println!("  Expiration: 30 days");
    println!("  IV: 60%");
    println!();

    // Calculate Greeks
    let greeks =
        OptionsAdapter::calculate_greeks(&call_option).expect("Failed to calculate Greeks");

    println!("Greeks:");
    println!("  Delta: {:.4}", greeks.delta);
    println!("  Gamma: {:.6}", greeks.gamma);
    println!("  Theta: {:.4} (daily decay)", greeks.theta);
    println!("  Vega: {:.4} (per 1% vol)", greeks.vega);
    println!("  Rho: {:.4}", greeks.rho);
    println!();

    // Calculate theoretical value
    let theo = OptionsAdapter::theoretical_value(&call_option).expect("Failed to calculate theo");
    println!("Theoretical Value: ${:.2}", theo);
    println!();

    // Check moneyness
    let moneyness = OptionsAdapter::moneyness(dec!(48000.0), dec!(50000.0));
    let is_atm = OptionsAdapter::is_atm(dec!(48000.0), dec!(50000.0), dec!(0.05));
    println!("Moneyness: {:.4} (OTM call)", moneyness);
    println!("Is ATM (5% tolerance): {}", is_atm);
    println!();

    // Create a put option
    let put_option = Options::new(
        OptionType::European,
        Side::Long,
        "BTC".to_string(),
        pos!(50000.0),
        ExpirationDate::Days(pos!(30.0)),
        pos!(0.6),
        pos!(1.0),
        pos!(48000.0),
        dec!(0.05),
        OptionStyle::Put,
        pos!(0.0),
        None,
    );

    let put_greeks =
        OptionsAdapter::calculate_greeks(&put_option).expect("Failed to calculate put Greeks");

    println!("Put Option Greeks:");
    println!("  Delta: {:.4} (negative for puts)", put_greeks.delta);
    println!("  Gamma: {:.6}", put_greeks.gamma);
    println!("  Theta: {:.4}", put_greeks.theta);
    println!("  Vega: {:.4}", put_greeks.vega);
    println!();

    // Portfolio Greeks aggregation
    println!("=== Portfolio Greeks Aggregation ===\n");

    let mut portfolio = PortfolioGreeks::new();

    // Add 10 call contracts
    portfolio.add(&greeks, dec!(10.0));
    println!("After adding 10 calls:");
    println!("  Portfolio Delta: {:.2}", portfolio.delta);
    println!("  Portfolio Gamma: {:.4}", portfolio.gamma);
    println!("  Portfolio Theta: {:.2}", portfolio.theta);
    println!("  Portfolio Vega: {:.2}", portfolio.vega);
    println!();

    // Add 5 put contracts (delta hedge)
    portfolio.add(&put_greeks, dec!(5.0));
    println!("After adding 5 puts (partial delta hedge):");
    println!("  Portfolio Delta: {:.2}", portfolio.delta);
    println!("  Portfolio Gamma: {:.4}", portfolio.gamma);
    println!();

    // Calculate dollar Greeks
    let underlying_price = dec!(48000.0);
    let multiplier = dec!(1.0); // BTC options typically 1x

    println!("Dollar Greeks (at ${} underlying):", underlying_price);
    println!(
        "  Dollar Delta: ${:.2}",
        portfolio.dollar_delta(underlying_price, multiplier)
    );
    println!(
        "  Dollar Gamma: ${:.2} (per 1% move)",
        portfolio.dollar_gamma(underlying_price, multiplier)
    );
    println!("  Dollar Vega: ${:.2}", portfolio.dollar_vega(multiplier));
    println!(
        "  Dollar Theta: ${:.2}/day",
        portfolio.dollar_theta(multiplier)
    );
    println!();

    // Check delta neutrality
    let tolerance = dec!(1.0);
    println!(
        "Is delta neutral (±{} tolerance): {}",
        tolerance,
        portfolio.is_delta_neutral(tolerance)
    );

    // Calculate shares to hedge
    let shares_to_hedge = portfolio.shares_to_hedge(multiplier);
    println!("Shares needed to delta hedge: {:.4} BTC", shares_to_hedge);
}
