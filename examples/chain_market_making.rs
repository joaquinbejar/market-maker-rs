//! Option chain market making example.
//!
//! Demonstrates multi-strike quoting across an entire option chain
//! using the Option-Chain-OrderBook integration.
//!
//! Run with: `cargo run --example chain_market_making --features chain`

use market_maker_rs::chain::{
    ChainMarketMaker, ChainMarketMakerConfig, ChainPricing, ChainQuoteOutcome, ChainRiskLimits,
    ChainRiskManager,
};
use market_maker_rs::options::{
    Dynamics, FlatSurface, PositionGreeks, SurfacePricer, tau_from_days,
};
use option_chain_orderbook::orderbook::ExpirationOrderBook;
use optionstratlib::model::ExpirationDate;
use optionstratlib::prelude::pos_or_panic;
use rust_decimal_macros::dec;
use std::sync::Arc;

fn main() {
    println!("=== Option Chain Market Making ===\n");

    // Create an expiration order book for BTC options
    let expiration = ExpirationDate::Days(pos_or_panic!(30.0));
    let chain = ExpirationOrderBook::new("BTC", expiration);

    // Add strikes to the chain
    let strikes = vec![45000, 47500, 50000, 52500, 55000];
    for strike in &strikes {
        chain.get_or_create_strike(*strike);
    }

    println!("Option Chain:");
    println!("  Underlying: BTC");
    println!("  Expiration: 30 days");
    println!("  Strikes: {:?}", strikes);
    println!();

    // Create chain market maker configuration
    let config = ChainMarketMakerConfig {
        base_spread_bps: 200,             // 2% base spread
        atm_spread_multiplier: dec!(1.5), // ATM gets 1.5x spread
        otm_spread_multiplier: dec!(1.0), // OTM gets 1x spread
        max_delta_per_strike: dec!(50.0),
        max_chain_delta: dec!(500.0),
        base_quote_size: 10,
        contract_multiplier: dec!(1.0),
        atm_tolerance_pct: dec!(0.02),
        min_spread_bps: 50,
        max_spread_bps: 1000,
    };

    let chain_arc = Arc::new(chain);
    let market_maker = ChainMarketMaker::new(chain_arc, config);

    println!("Chain Market Maker Configuration:");
    println!(
        "  Base Spread: {} bps",
        market_maker.config().base_spread_bps
    );
    println!(
        "  ATM Multiplier: {}x",
        market_maker.config().atm_spread_multiplier
    );
    println!(
        "  Max Chain Delta: {}",
        market_maker.config().max_chain_delta
    );
    println!(
        "  Base Quote Size: {}",
        market_maker.config().base_quote_size
    );
    println!();

    // Get ATM strike
    let underlying_price: u64 = 50000;
    let atm_strike = market_maker
        .get_atm_strike(underlying_price)
        .expect("Failed to get ATM strike");

    println!("Market State:");
    println!("  Underlying Price: ${}", underlying_price);
    println!("  ATM Strike: ${}", atm_strike);
    println!("  Strike Count: {}", market_maker.strike_count());
    println!();

    // Valuation inputs for this expiration: a flat 60 % implied volatility,
    // sticky moneyness, no rates, 30 days to expiry.
    let surface = FlatSurface::new(dec!(0.6)).expect("valid volatility");
    let pricer =
        SurfacePricer::new(Arc::new(surface), Dynamics::StickyMoneyness).expect("valid dynamics");
    let tau = tau_from_days(dec!(30)).expect("valid time to expiry");
    let pricing = ChainPricing::new(Arc::new(pricer), tau, dec!(0.0), dec!(0.0))
        .expect("valid pricing inputs");

    // Refresh all quotes
    println!("=== Quote Generation ===\n");

    let outcomes = market_maker
        .refresh_all_quotes(underlying_price, &pricing)
        .expect("Failed to refresh quotes");

    println!("Generated {} outcomes:", outcomes.len());
    println!();

    for outcome in &outcomes {
        match outcome {
            ChainQuoteOutcome::Quote(quote) => {
                let is_atm = quote.strike == atm_strike;
                let atm_marker = if is_atm { " (ATM)" } else { "" };

                println!("  ${} {:?}{}:", quote.strike, quote.style, atm_marker);
                println!(
                    "    Bid: ${} x {}  |  Ask: ${} x {}",
                    quote.bid_price, quote.bid_size, quote.ask_price, quote.ask_size
                );
                println!(
                    "    Theo: ${} (model {:.4})  |  Spread: {} bps  |  Delta: {:.4}",
                    quote.theo, quote.valuation.price, quote.spread_bps, quote.valuation.delta
                );
            }
            ChainQuoteOutcome::NotQuoted {
                strike,
                style,
                reason,
            } => println!("  ${strike} {style:?}: not quoted ({reason})"),
            _ => println!("  unknown outcome"),
        }
    }
    println!();

    // Check spread multipliers
    println!("=== Spread Multipliers ===\n");

    for strike in &strikes {
        let is_atm = *strike == atm_strike;
        let multiplier = market_maker
            .get_spread_multiplier(*strike, underlying_price, is_atm)
            .expect("valid spread multiplier");
        println!(
            "  ${}: {:.2}x {}",
            strike,
            multiplier,
            if is_atm { "(ATM)" } else { "" }
        );
    }
    println!();

    // Risk status
    println!("=== Chain Risk Status ===\n");

    let risk_status = market_maker
        .check_chain_risk()
        .expect("Failed to check chain risk");
    println!("Risk Status: {:?}", risk_status);
    println!("  Can Quote: {}", risk_status.can_quote());
    println!("  Requires Action: {}", risk_status.requires_action());
    println!();

    // Chain Greeks (initially zero)
    let chain_greeks = market_maker.get_chain_greeks();
    println!("Chain Greeks:");
    println!("  Delta: {}", chain_greeks.delta);
    println!("  Gamma: {}", chain_greeks.gamma);
    println!("  Vega: {}", chain_greeks.vega);
    println!("  Theta: {}", chain_greeks.theta);
    println!();

    // Demonstrate chain risk manager
    println!("=== Chain Risk Manager ===\n");

    let risk_limits = ChainRiskLimits {
        max_chain_delta: dec!(500.0),
        max_chain_gamma: dec!(100.0),
        max_chain_vega: dec!(5000.0),
        max_chain_theta: dec!(-2000.0),
        max_delta_per_strike: dec!(50.0),
        delta_hedge_threshold: dec!(25.0),
        max_notional: dec!(1000000.0),
    };

    let mut risk_manager = ChainRiskManager::new("BTC", risk_limits, dec!(1.0));

    // Simulate adding positions
    let call_greeks =
        PositionGreeks::new(dec!(0.5), dec!(0.02), dec!(-0.05), dec!(0.15), dec!(0.08));

    // Add 100 contracts across the chain
    risk_manager
        .add_position_greeks(&call_greeks, dec!(100.0))
        .expect("Failed to add position");

    println!("After adding 100 call contracts:");
    println!("  Chain Delta: {}", risk_manager.current_greeks().delta);
    println!("  Chain Gamma: {}", risk_manager.current_greeks().gamma);
    let should_hedge = risk_manager
        .should_hedge()
        .expect("Failed to check the hedge");
    println!("  Should Hedge: {should_hedge}");
    println!();

    // Calculate hedge
    if should_hedge {
        let hedges = risk_manager
            .calculate_hedge(dec!(50000.0))
            .expect("Failed to calculate hedge");
        if !hedges.is_empty() {
            println!("Hedge Orders:");
            for hedge in &hedges {
                println!(
                    "  {} {} @ ${}",
                    if hedge.quantity > dec!(0.0) {
                        "BUY"
                    } else {
                        "SELL"
                    },
                    hedge.quantity.abs(),
                    hedge.price
                );
            }
        }
    }
    println!();

    // Dollar Greeks
    println!("Dollar Greeks (at ${}):", underlying_price);
    println!(
        "  Dollar Delta: ${:.2}",
        risk_manager
            .dollar_delta(dec!(50000.0))
            .expect("Failed to calculate dollar delta")
    );
    println!(
        "  Dollar Gamma: ${:.2}",
        risk_manager
            .dollar_gamma(dec!(50000.0))
            .expect("Failed to calculate dollar gamma")
    );
    println!(
        "  Dollar Vega: ${:.2}",
        risk_manager
            .dollar_vega()
            .expect("Failed to calculate dollar vega")
    );
    println!(
        "  Dollar Theta: ${:.2}/day",
        risk_manager
            .dollar_theta()
            .expect("Failed to calculate dollar theta")
    );
}
