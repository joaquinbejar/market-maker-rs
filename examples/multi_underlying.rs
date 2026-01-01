//! Multi-Underlying Market Making Example
//!
//! This example demonstrates how to use the multi-underlying module to manage
//! market making across multiple assets simultaneously.
//!
//! Features demonstrated:
//! - Creating and configuring a multi-underlying manager
//! - Adding underlyings with different configurations
//! - Setting cross-asset correlations
//! - Updating prices and Greeks
//! - Getting unified risk views
//! - Capital allocation strategies
//! - Cross-asset hedging suggestions
//!
//! Run with: `cargo run --example multi_underlying --features multi-underlying`

use market_maker_rs::dec;
use market_maker_rs::multi_underlying::{
    CapitalAllocationStrategy, CrossAssetHedge, MultiUnderlyingManager, UnderlyingConfig,
    UnderlyingStatus, UnifiedGreeks, UnifiedRisk,
};

fn main() {
    println!("=== Multi-Underlying Market Making Example ===\n");

    // Create a manager with $1M total capital
    let mut manager = MultiUnderlyingManager::new(dec!(1_000_000))
        .with_allocation_strategy(CapitalAllocationStrategy::TargetWeight)
        .with_max_underlyings(5)
        .with_max_total_delta(dec!(50000))
        .with_max_total_vega(dec!(25000));

    println!("Created manager with $1,000,000 capital");
    println!("Allocation strategy: {:?}\n", manager.allocation_strategy());

    // Add underlyings with target weights
    add_underlyings(&mut manager);

    // Set correlations between assets
    set_correlations(&mut manager);

    // Simulate market data updates
    simulate_market_updates(&mut manager);

    // Display unified risk view
    display_unified_risk(&manager);

    // Get cross-asset hedge suggestions
    display_hedge_suggestions(&manager);

    // Demonstrate capital reallocation
    demonstrate_reallocation(&mut manager);

    println!("\n=== Example Complete ===");
}

/// Adds underlyings to the manager with different configurations.
fn add_underlyings(manager: &mut MultiUnderlyingManager) {
    println!("--- Adding Underlyings ---\n");

    // BTC with 40% target weight
    let btc_config = UnderlyingConfig::new("BTC", dec!(0.40))
        .with_max_delta(dec!(10000))
        .with_max_gamma(dec!(500))
        .with_max_vega(dec!(5000))
        .with_max_position_value(dec!(200000));

    if let Err(e) = manager.add_underlying(btc_config) {
        eprintln!("Failed to add BTC: {}", e);
    } else {
        println!("Added BTC with 40% target weight");
    }

    // ETH with 30% target weight
    let eth_config = UnderlyingConfig::new("ETH", dec!(0.30))
        .with_max_delta(dec!(8000))
        .with_max_gamma(dec!(400))
        .with_max_vega(dec!(4000))
        .with_max_position_value(dec!(150000));

    if let Err(e) = manager.add_underlying(eth_config) {
        eprintln!("Failed to add ETH: {}", e);
    } else {
        println!("Added ETH with 30% target weight");
    }

    // SOL with 20% target weight
    let sol_config = UnderlyingConfig::new("SOL", dec!(0.20))
        .with_max_delta(dec!(5000))
        .with_max_gamma(dec!(250))
        .with_max_vega(dec!(2500))
        .with_max_position_value(dec!(100000));

    if let Err(e) = manager.add_underlying(sol_config) {
        eprintln!("Failed to add SOL: {}", e);
    } else {
        println!("Added SOL with 20% target weight");
    }

    // AVAX with 10% target weight
    let avax_config = UnderlyingConfig::new("AVAX", dec!(0.10))
        .with_max_delta(dec!(3000))
        .with_max_gamma(dec!(150))
        .with_max_vega(dec!(1500))
        .with_max_position_value(dec!(50000));

    if let Err(e) = manager.add_underlying(avax_config) {
        eprintln!("Failed to add AVAX: {}", e);
    } else {
        println!("Added AVAX with 10% target weight");
    }

    println!("\nTotal underlyings: {}", manager.underlying_count());
    println!("Symbols: {:?}", manager.symbols());
}

/// Sets correlations between assets.
fn set_correlations(manager: &mut MultiUnderlyingManager) {
    println!("\n--- Setting Correlations ---\n");

    // BTC-ETH: High correlation (0.85)
    manager.set_correlation("BTC", "ETH", dec!(0.85));
    println!("BTC-ETH correlation: 0.85");

    // BTC-SOL: Moderate correlation (0.70)
    manager.set_correlation("BTC", "SOL", dec!(0.70));
    println!("BTC-SOL correlation: 0.70");

    // BTC-AVAX: Moderate correlation (0.65)
    manager.set_correlation("BTC", "AVAX", dec!(0.65));
    println!("BTC-AVAX correlation: 0.65");

    // ETH-SOL: Moderate-high correlation (0.75)
    manager.set_correlation("ETH", "SOL", dec!(0.75));
    println!("ETH-SOL correlation: 0.75");

    // ETH-AVAX: Moderate correlation (0.60)
    manager.set_correlation("ETH", "AVAX", dec!(0.60));
    println!("ETH-AVAX correlation: 0.60");

    // SOL-AVAX: Moderate correlation (0.55)
    manager.set_correlation("SOL", "AVAX", dec!(0.55));
    println!("SOL-AVAX correlation: 0.55");

    // Verify correlation retrieval
    if let Some(corr) = manager.get_correlation("BTC", "ETH") {
        println!("\nVerified BTC-ETH correlation: {}", corr);
    }
}

/// Simulates market data updates.
fn simulate_market_updates(manager: &mut MultiUnderlyingManager) {
    println!("\n--- Simulating Market Updates ---\n");

    // Update prices
    manager.update_price("BTC", dec!(45000));
    manager.update_price("ETH", dec!(2500));
    manager.update_price("SOL", dec!(100));
    manager.update_price("AVAX", dec!(35));
    println!("Updated prices: BTC=$45,000, ETH=$2,500, SOL=$100, AVAX=$35");

    // Update Greeks (simulating options positions)
    // BTC: Long delta position
    manager.update_greeks("BTC", dec!(5.5), dec!(0.02), dec!(1500));
    println!("BTC Greeks: delta=5.5, gamma=0.02, vega=1500");

    // ETH: Short delta position
    manager.update_greeks("ETH", dec!(-12.0), dec!(0.05), dec!(800));
    println!("ETH Greeks: delta=-12.0, gamma=0.05, vega=800");

    // SOL: Neutral delta
    manager.update_greeks("SOL", dec!(0.5), dec!(0.01), dec!(200));
    println!("SOL Greeks: delta=0.5, gamma=0.01, vega=200");

    // AVAX: Small long delta
    manager.update_greeks("AVAX", dec!(2.0), dec!(0.008), dec!(100));
    println!("AVAX Greeks: delta=2.0, gamma=0.008, vega=100");

    // Update P&L
    manager.update_pnl("BTC", dec!(5000), dec!(2000));
    manager.update_pnl("ETH", dec!(-1500), dec!(3000));
    manager.update_pnl("SOL", dec!(500), dec!(200));
    manager.update_pnl("AVAX", dec!(100), dec!(50));
    println!("\nUpdated P&L for all underlyings");

    // Update position values
    manager.update_position_value("BTC", dec!(180000));
    manager.update_position_value("ETH", dec!(120000));
    manager.update_position_value("SOL", dec!(45000));
    manager.update_position_value("AVAX", dec!(15000));
    println!("Updated position values");
}

/// Displays the unified risk view.
fn display_unified_risk(manager: &MultiUnderlyingManager) {
    println!("\n--- Unified Risk View ---\n");

    let risk: UnifiedRisk = manager.get_unified_risk();

    println!("Capital:");
    println!("  Total Capital: ${}", risk.total_capital);
    println!("  Total Position Value: ${}", risk.total_position_value);

    println!("\nP&L:");
    println!("  Unrealized P&L: ${}", risk.total_unrealized_pnl);
    println!("  Realized P&L: ${}", risk.total_realized_pnl);
    println!(
        "  Total P&L: ${}",
        risk.total_unrealized_pnl + risk.total_realized_pnl
    );

    println!("\nGreeks:");
    let greeks: &UnifiedGreeks = &risk.greeks;
    println!("  Total Dollar Delta: ${}", greeks.total_dollar_delta);
    println!("  Total Dollar Gamma: ${}", greeks.total_dollar_gamma);
    println!("  Total Dollar Vega: ${}", greeks.total_dollar_vega);
    println!("  Portfolio Volatility: {:.4}", greeks.portfolio_volatility);
    println!("  Active Underlyings: {}", greeks.underlying_count);

    println!("\nRisk Utilization:");
    println!("  Delta Utilization: {:.2}%", risk.delta_utilization);
    println!("  Vega Utilization: {:.2}%", risk.vega_utilization);

    println!("\nUnderlying Status:");
    println!("  Active: {}", risk.active_underlyings);
    println!("  Halted: {}", risk.halted_underlyings);
}

/// Displays cross-asset hedge suggestions.
fn display_hedge_suggestions(manager: &MultiUnderlyingManager) {
    println!("\n--- Cross-Asset Hedge Suggestions ---\n");

    let hedges: Vec<CrossAssetHedge> = manager.get_cross_asset_hedges();

    if hedges.is_empty() {
        println!("No hedge suggestions (positions may be too small or uncorrelated)");
        return;
    }

    for (i, hedge) in hedges.iter().enumerate() {
        println!("Hedge {}:", i + 1);
        println!(
            "  Source: {} -> Hedge with: {}",
            hedge.source_underlying, hedge.hedge_underlying
        );
        println!("  Hedge Ratio: {:.4}", hedge.hedge_ratio);
        println!("  Correlation: {:.2}", hedge.correlation);
        println!("  Est. Risk Reduction: {:.1}%", hedge.risk_reduction);
        println!("  Type: {:?}", hedge.hedge_type);
        println!();
    }
}

/// Demonstrates capital reallocation with different strategies.
fn demonstrate_reallocation(manager: &mut MultiUnderlyingManager) {
    println!("\n--- Capital Reallocation Demo ---\n");

    // Show current allocations
    println!("Current allocations (TargetWeight strategy):");
    for symbol in manager.symbols() {
        if let Some(state) = manager.get_state(&symbol) {
            println!("  {}: ${} allocated", symbol, state.allocated_capital);
        }
    }

    // Switch to Equal allocation
    println!("\nSwitching to Equal allocation strategy...");
    manager.set_allocation_strategy(CapitalAllocationStrategy::Equal);
    manager.reallocate_capital();

    println!("New allocations (Equal strategy):");
    for symbol in manager.symbols() {
        if let Some(state) = manager.get_state(&symbol) {
            println!("  {}: ${} allocated", symbol, state.allocated_capital);
        }
    }

    // Switch to RiskParity allocation
    println!("\nSwitching to RiskParity allocation strategy...");
    manager.set_allocation_strategy(CapitalAllocationStrategy::RiskParity);
    manager.reallocate_capital();

    println!("New allocations (RiskParity strategy):");
    for symbol in manager.symbols() {
        if let Some(state) = manager.get_state(&symbol) {
            println!("  {}: ${} allocated", symbol, state.allocated_capital);
        }
    }

    // Demonstrate halting an underlying
    println!("\nHalting SOL trading...");
    manager.set_status("SOL", UnderlyingStatus::Halted);

    let risk = manager.get_unified_risk();
    println!("Active underlyings after halt: {}", risk.active_underlyings);
    println!("Halted underlyings: {}", risk.halted_underlyings);
}
