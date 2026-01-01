//! Options risk management example.
//!
//! Demonstrates Greeks-based risk management with circuit breakers,
//! order validation, and auto-hedging.
//!
//! Run with: `cargo run --example options_risk_management --features options`

use market_maker_rs::options::{
    AutoHedgerConfig, GreeksCircuitBreakerState, GreeksLimits, GreeksRiskManager, OrderDecision,
    PositionGreeks,
};
use rust_decimal_macros::dec;

fn main() {
    println!("=== Options Risk Management ===\n");

    // Configure Greeks limits
    let limits = GreeksLimits::new(
        dec!(100.0),  // max delta: 100 contracts equivalent
        dec!(50.0),   // max gamma
        dec!(1000.0), // max vega
        dec!(-500.0), // max theta
        dec!(10.0),   // delta hedge threshold
    );

    // Configure auto-hedger
    let hedger_config = AutoHedgerConfig {
        enabled: true,
        trigger_threshold_pct: dec!(0.8), // hedge at 80% of limit
        target_delta_pct: dec!(0.5),      // target 50% of limit after hedge
        min_hedge_size: dec!(1.0),
        max_hedge_size: dec!(100.0),
        contract_multiplier: dec!(1.0), // BTC options
    };

    // Create risk manager
    let mut risk_manager = GreeksRiskManager::new("BTC", limits, hedger_config);

    println!("Risk Manager Configuration:");
    println!("  Underlying: {}", risk_manager.underlying_symbol());
    println!("  Max Delta: {}", risk_manager.limits().max_delta);
    println!("  Max Gamma: {}", risk_manager.limits().max_gamma);
    println!("  Max Vega: {}", risk_manager.limits().max_vega);
    println!(
        "  Hedge Threshold: {}",
        risk_manager.limits().delta_hedge_threshold
    );
    println!();

    // Simulate option Greeks (typical ATM call)
    let call_greeks = PositionGreeks::new(
        dec!(0.5),   // delta
        dec!(0.02),  // gamma
        dec!(-0.05), // theta
        dec!(0.15),  // vega
        dec!(0.08),  // rho
    );

    println!("Option Greeks (per contract):");
    println!("  Delta: {}", call_greeks.delta);
    println!("  Gamma: {}", call_greeks.gamma);
    println!("  Theta: {}", call_greeks.theta);
    println!("  Vega: {}", call_greeks.vega);
    println!();

    // Check order - should be allowed
    println!("=== Order Validation ===\n");

    let decision = risk_manager.check_order(&call_greeks, dec!(10.0));
    println!("Order: Buy 10 contracts");
    match &decision {
        OrderDecision::Allowed => println!("  Decision: ALLOWED"),
        OrderDecision::Scaled {
            original_size,
            new_size,
            reason,
        } => {
            println!("  Decision: SCALED from {} to {}", original_size, new_size);
            println!("  Reason: {}", reason);
        }
        OrderDecision::Rejected { reason } => {
            println!("  Decision: REJECTED");
            println!("  Reason: {}", reason);
        }
    }
    println!();

    // Execute the order (update Greeks)
    risk_manager.update_on_fill(&call_greeks, dec!(10.0), 1000);

    println!("After fill:");
    println!("  Portfolio Delta: {}", risk_manager.current_greeks().delta);
    println!("  Portfolio Gamma: {}", risk_manager.current_greeks().gamma);
    println!();

    // Check limit utilization
    let utilization = risk_manager.limit_utilization();
    println!("Limit Utilization:");
    println!("  Delta: {:.1}%", utilization.delta * dec!(100.0));
    println!("  Gamma: {:.1}%", utilization.gamma * dec!(100.0));
    println!("  Vega: {:.1}%", utilization.vega * dec!(100.0));
    println!(
        "  Max Utilization: {:.1}%",
        utilization.max_utilization * dec!(100.0)
    );
    println!("  Is Warning: {}", utilization.is_warning());
    println!("  Is Breached: {}", utilization.is_breached());
    println!();

    // Add more positions to approach limits
    println!("=== Approaching Limits ===\n");

    // Add 150 more contracts (total 160 contracts = 80 delta)
    risk_manager.update_on_fill(&call_greeks, dec!(150.0), 2000);

    println!("After adding 150 more contracts:");
    println!("  Portfolio Delta: {}", risk_manager.current_greeks().delta);

    let utilization = risk_manager.limit_utilization();
    println!(
        "  Delta Utilization: {:.1}%",
        utilization.delta * dec!(100.0)
    );
    println!("  Is Warning: {}", utilization.is_warning());
    println!();

    // Check if hedging is needed
    println!("Hedging Status:");
    println!("  Needs Hedge: {}", risk_manager.needs_hedge());
    println!("  Hedge Urgency: {:?}", risk_manager.hedge_urgency());

    if risk_manager.needs_hedge()
        && let Some(hedge) = risk_manager.calculate_hedge_order(dec!(50000.0))
    {
        println!("\nHedge Order:");
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
    println!();

    // Try to add more - should be scaled or rejected
    println!("=== Order at High Utilization ===\n");

    let decision = risk_manager.check_order(&call_greeks, dec!(50.0));
    println!("Order: Buy 50 more contracts");
    match &decision {
        OrderDecision::Allowed => println!("  Decision: ALLOWED"),
        OrderDecision::Scaled {
            original_size,
            new_size,
            reason,
        } => {
            println!("  Decision: SCALED from {} to {}", original_size, new_size);
            println!("  Reason: {}", reason);
        }
        OrderDecision::Rejected { reason } => {
            println!("  Decision: REJECTED");
            println!("  Reason: {}", reason);
        }
    }
    println!();

    // Circuit breaker demonstration
    println!("=== Circuit Breaker ===\n");

    println!(
        "Current State: {:?}",
        risk_manager.circuit_breaker_status().state
    );

    // Add more to trip circuit breaker
    risk_manager.update_on_fill(&call_greeks, dec!(30.0), 3000);

    println!("After adding 30 more contracts:");
    println!("  Portfolio Delta: {}", risk_manager.current_greeks().delta);
    println!(
        "  Circuit Breaker State: {:?}",
        risk_manager.circuit_breaker_status().state
    );

    if risk_manager.circuit_breaker_status().state == GreeksCircuitBreakerState::Open {
        println!(
            "  Reason: {:?}",
            risk_manager.circuit_breaker_status().reason
        );
        println!("\n  ⚠️  Trading halted - circuit breaker tripped!");
    }

    // Try order with circuit breaker open
    let decision = risk_manager.check_order(&call_greeks, dec!(1.0));
    println!("\nOrder attempt with circuit breaker open:");
    match &decision {
        OrderDecision::Rejected { reason } => {
            println!("  Decision: REJECTED");
            println!("  Reason: {}", reason);
        }
        _ => println!("  Unexpected decision"),
    }
    println!();

    // Reset and demonstrate recovery
    println!("=== Recovery ===\n");

    risk_manager.reset();
    println!("After reset:");
    println!("  Portfolio Delta: {}", risk_manager.current_greeks().delta);
    println!(
        "  Circuit Breaker State: {:?}",
        risk_manager.circuit_breaker_status().state
    );

    let decision = risk_manager.check_order(&call_greeks, dec!(10.0));
    println!("\nOrder after reset:");
    match &decision {
        OrderDecision::Allowed => println!("  Decision: ALLOWED ✓"),
        _ => println!("  Unexpected decision"),
    }
}
