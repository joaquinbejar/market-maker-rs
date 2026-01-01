//! Options market making example.
//!
//! Demonstrates Greeks-aware quoting and spread adjustment for options
//! using the OptionsMarketMaker trait.
//!
//! Run with: `cargo run --example options_market_making --features options`

use market_maker_rs::options::{
    GreeksLimits, OptionsAdapter, OptionsMarketMaker, OptionsMarketMakerConfig,
    OptionsMarketMakerImpl, PortfolioGreeks,
};
use optionstratlib::model::ExpirationDate;
use optionstratlib::model::option::Options;
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::pos;
use rust_decimal_macros::dec;

fn main() {
    println!("=== Options Market Making ===\n");

    // Create market maker with custom configuration
    let config = OptionsMarketMakerConfig {
        base_spread_pct: dec!(0.02),        // 2% base spread
        min_spread_pct: dec!(0.005),        // 0.5% minimum
        max_spread_pct: dec!(0.10),         // 10% maximum
        gamma_spread_multiplier: dec!(2.0), // ATM options get 2x spread
        vega_spread_multiplier: dec!(0.5),
        theta_spread_multiplier: dec!(0.1),
        atm_tolerance: dec!(0.02),       // 2% ATM tolerance
        contract_multiplier: dec!(1.0),  // BTC options
        put_call_skew_factor: dec!(1.0), // No put/call skew
    };

    let market_maker = OptionsMarketMakerImpl::new(config);

    println!("Market Maker Configuration:");
    println!(
        "  Base Spread: {}%",
        market_maker.config().base_spread_pct * dec!(100.0)
    );
    println!(
        "  Min Spread: {}%",
        market_maker.config().min_spread_pct * dec!(100.0)
    );
    println!(
        "  Max Spread: {}%",
        market_maker.config().max_spread_pct * dec!(100.0)
    );
    println!(
        "  Gamma Multiplier: {}x",
        market_maker.config().gamma_spread_multiplier
    );
    println!();

    // Create risk limits
    let risk_limits = GreeksLimits::new(
        dec!(100.0),  // max delta
        dec!(50.0),   // max gamma
        dec!(1000.0), // max vega
        dec!(-500.0), // max theta (negative)
        dec!(10.0),   // delta hedge threshold
    );

    println!("Risk Limits:");
    println!("  Max Delta: {}", risk_limits.max_delta);
    println!("  Max Gamma: {}", risk_limits.max_gamma);
    println!("  Max Vega: {}", risk_limits.max_vega);
    println!("  Max Theta: {}", risk_limits.max_theta);
    println!("  Hedge Threshold: {}", risk_limits.delta_hedge_threshold);
    println!();

    // Create ATM call option
    let atm_call = Options::new(
        OptionType::European,
        Side::Long,
        "BTC".to_string(),
        pos!(50000.0), // strike = underlying (ATM)
        ExpirationDate::Days(pos!(30.0)),
        pos!(0.6),
        pos!(1.0),
        pos!(50000.0), // underlying price
        dec!(0.05),
        OptionStyle::Call,
        pos!(0.0),
        None,
    );

    // Create OTM call option
    let otm_call = Options::new(
        OptionType::European,
        Side::Long,
        "BTC".to_string(),
        pos!(55000.0), // 10% OTM
        ExpirationDate::Days(pos!(30.0)),
        pos!(0.6),
        pos!(1.0),
        pos!(50000.0),
        dec!(0.05),
        OptionStyle::Call,
        pos!(0.0),
        None,
    );

    // Empty portfolio (no existing positions)
    let portfolio = PortfolioGreeks::new();

    println!("=== ATM Call Option ===");
    println!("Strike: $50,000 (ATM)");

    let atm_greeks =
        OptionsAdapter::calculate_greeks(&atm_call).expect("Failed to calculate Greeks");
    println!(
        "Greeks: Delta={:.3}, Gamma={:.5}",
        atm_greeks.delta, atm_greeks.gamma
    );

    // Calculate spread for ATM option
    let atm_spread = market_maker
        .calculate_options_spread(&atm_call, &atm_greeks)
        .expect("Failed to calculate spread");
    println!("Calculated Spread: {:.2}%", atm_spread * dec!(100.0));

    // Calculate Greeks-adjusted quotes
    let (atm_bid, atm_ask) = market_maker
        .calculate_greeks_adjusted_quotes(&atm_call, &portfolio, &risk_limits)
        .expect("Failed to calculate quotes");

    println!("Quotes: Bid=${:.2}, Ask=${:.2}", atm_bid, atm_ask);
    println!(
        "Spread: ${:.2} ({:.2}%)",
        atm_ask - atm_bid,
        (atm_ask - atm_bid) / atm_bid * dec!(100.0)
    );
    println!();

    println!("=== OTM Call Option ===");
    println!("Strike: $55,000 (10% OTM)");

    let otm_greeks =
        OptionsAdapter::calculate_greeks(&otm_call).expect("Failed to calculate Greeks");
    println!(
        "Greeks: Delta={:.3}, Gamma={:.5}",
        otm_greeks.delta, otm_greeks.gamma
    );

    let otm_spread = market_maker
        .calculate_options_spread(&otm_call, &otm_greeks)
        .expect("Failed to calculate spread");
    println!("Calculated Spread: {:.2}%", otm_spread * dec!(100.0));

    let (otm_bid, otm_ask) = market_maker
        .calculate_greeks_adjusted_quotes(&otm_call, &portfolio, &risk_limits)
        .expect("Failed to calculate quotes");

    println!("Quotes: Bid=${:.2}, Ask=${:.2}", otm_bid, otm_ask);
    println!();

    // Demonstrate spread widening with portfolio exposure
    println!("=== Impact of Portfolio Exposure ===\n");

    let mut exposed_portfolio = PortfolioGreeks::new();
    exposed_portfolio.delta = dec!(80.0); // 80% of max delta limit
    exposed_portfolio.gamma = dec!(40.0); // 80% of max gamma limit

    println!("Portfolio with high exposure:");
    println!(
        "  Delta: {} ({}% of limit)",
        exposed_portfolio.delta,
        exposed_portfolio.delta / risk_limits.max_delta * dec!(100.0)
    );
    println!(
        "  Gamma: {} ({}% of limit)",
        exposed_portfolio.gamma,
        exposed_portfolio.gamma / risk_limits.max_gamma * dec!(100.0)
    );
    println!();

    let (exposed_bid, exposed_ask) = market_maker
        .calculate_greeks_adjusted_quotes(&atm_call, &exposed_portfolio, &risk_limits)
        .expect("Failed to calculate quotes");

    println!("ATM Call quotes with high exposure:");
    println!("  Bid: ${:.2} (was ${:.2})", exposed_bid, atm_bid);
    println!("  Ask: ${:.2} (was ${:.2})", exposed_ask, atm_ask);
    println!("  Spread widened due to risk exposure");
    println!();

    // Delta hedging
    println!("=== Delta Hedging ===\n");

    let hedge_orders = market_maker
        .calculate_delta_hedge(&exposed_portfolio, dec!(50000.0), "BTC")
        .expect("Failed to calculate hedge");

    if hedge_orders.is_empty() {
        println!("No hedge needed");
    } else {
        for hedge in &hedge_orders {
            println!("Hedge Order:");
            println!("  Symbol: {}", hedge.symbol);
            println!(
                "  Quantity: {:.4} ({})",
                hedge.quantity.abs(),
                if hedge.quantity > dec!(0.0) {
                    "BUY"
                } else {
                    "SELL"
                }
            );
            println!("  Price: ${:.2}", hedge.price);
            println!("  Delta Impact: {:.2}", hedge.delta_impact);
        }
    }
}
