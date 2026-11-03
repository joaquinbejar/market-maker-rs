//! Alert System Example
//!
//! This example demonstrates the alert system functionality:
//! - Creating alerts with different severity levels
//! - Alert handlers (log, collecting)
//! - Alert manager with history and deduplication
//!
//! Run with: `cargo run --example alert_system`

use market_maker_rs::prelude::*;
use std::sync::Arc;

fn main() {
    println!("=== Alert System Example ===\n");

    // Create alert manager
    let mut manager = AlertManager::new(
        100,    // max history size
        60_000, // deduplication window: 1 minute
    );

    println!("Alert Manager Configuration:");
    println!("  Max History: 100 alerts");
    println!("  Deduplication Window: 1 minute\n");

    // Create and add a collecting handler
    let collector = Arc::new(CollectingAlertHandler::new(AlertSeverity::Info));

    // Wrapper to use Arc with the manager
    struct ArcHandler(Arc<CollectingAlertHandler>);
    impl AlertHandler for ArcHandler {
        fn handle(&self, alert: &Alert) -> MMResult<()> {
            self.0.handle(alert)
        }
        fn accepts_severity(&self, severity: AlertSeverity) -> bool {
            self.0.accepts_severity(severity)
        }
    }

    manager.add_handler(Box::new(ArcHandler(Arc::clone(&collector))));

    // Generate various alerts
    println!("--- Generating Alerts ---\n");

    let alerts = [
        (
            AlertType::LargeLoss {
                amount: dec!(500.0),
                threshold: dec!(100.0),
            },
            AlertSeverity::Warning,
        ),
        (
            AlertType::PositionLimit {
                current: dec!(95.0),
                limit: dec!(100.0),
                pct: dec!(0.95),
            },
            AlertSeverity::Warning,
        ),
        (
            AlertType::CircuitBreakerTriggered {
                reason: "Max consecutive losses".to_string(),
            },
            AlertSeverity::Critical,
        ),
        (
            AlertType::HighLatency {
                metric: "order_submission".to_string(),
                latency_ms: 500,
                threshold_ms: 100,
            },
            AlertSeverity::Error,
        ),
        (
            AlertType::MarketCondition {
                condition: "high_volatility".to_string(),
                details: "Volatility 3x normal".to_string(),
            },
            AlertSeverity::Info,
        ),
    ];

    let mut timestamp = 1000u64;
    for (alert_type, severity) in alerts {
        manager
            .alert(alert_type.clone(), severity, timestamp)
            .expect("alert delivery");
        println!("Alert: {:?}", severity);
        println!("  Type: {}", alert_type.type_key());
        println!("  Message: {}", alert_type.default_message());
        println!();
        timestamp += 1000;
    }

    println!("--- Alert Statistics ---\n");
    println!("  Total in History: {}", manager.history_count());
    println!("  Unacknowledged: {}", manager.unacknowledged_count());
    println!("  Collected by Handler: {}", collector.count().unwrap());

    // Get alerts by severity
    let critical = manager.get_alerts_by_severity(AlertSeverity::Critical);
    let warnings = manager.get_alerts_by_severity(AlertSeverity::Warning);
    println!("  Critical Alerts: {}", critical.len());
    println!("  Warning Alerts: {}", warnings.len());
}
