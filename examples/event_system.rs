//! Event System Example
//!
//! This example demonstrates how to use the event system module for broadcasting
//! market maker events to multiple consumers with filtering and history support.
//!
//! Features demonstrated:
//! - Creating an event broadcaster
//! - Subscribing to events (all and filtered)
//! - Broadcasting different event types
//! - Event filtering by type, symbol, and alert level
//! - Event history for reconnection
//! - Event aggregation/batching
//!
//! Run with: `cargo run --example event_system --features events`

use std::sync::Arc;

use market_maker_rs::events::{
    AlertCategory, AlertLevel, CancelReason, CircuitBreakerState, EventAggregator,
    EventBroadcaster, EventBroadcasterConfig, EventFilter, EventType, HedgeReason,
    MarketMakerEvent, OptionStyle, Side, SystemStatus,
};

#[tokio::main]
async fn main() {
    println!("=== Event System Example ===\n");

    // Create broadcaster with custom configuration
    let config = EventBroadcasterConfig::default()
        .with_channel_capacity(256)
        .with_max_history_size(1000)
        .with_history_retention_secs(3600)
        .with_heartbeat_interval_secs(30);

    let broadcaster = Arc::new(EventBroadcaster::new(config));
    println!("Created event broadcaster");
    println!("  Channel capacity: 256");
    println!("  Max history size: 1000");
    println!("  History retention: 3600s");

    // Demonstrate different subscription types
    demonstrate_subscriptions(&broadcaster).await;

    // Demonstrate event broadcasting
    demonstrate_broadcasting(&broadcaster).await;

    // Demonstrate event filtering
    demonstrate_filtering(&broadcaster).await;

    // Demonstrate event history
    demonstrate_history(&broadcaster).await;

    // Demonstrate event aggregation
    demonstrate_aggregation().await;

    println!("\n=== Example Complete ===");
}

/// Demonstrates different subscription types.
async fn demonstrate_subscriptions(broadcaster: &Arc<EventBroadcaster>) {
    println!("\n--- Subscription Types ---\n");

    // Subscribe to all events
    let _all_rx = broadcaster.subscribe();
    println!("Created subscription for ALL events");

    // Subscribe with filter for specific event types
    let fill_filter =
        EventFilter::new().with_event_types([EventType::OrderFilled, EventType::OrderCancelled]);
    let _fill_rx = broadcaster.subscribe_filtered(fill_filter);
    println!("Created filtered subscription for fills and cancels");

    // Subscribe with symbol filter
    let btc_filter = EventFilter::new().with_symbols(["BTC".to_string()]);
    let _btc_rx = broadcaster.subscribe_filtered(btc_filter);
    println!("Created filtered subscription for BTC only");

    // Subscribe excluding heartbeats
    let no_heartbeat_filter = EventFilter::new().exclude_heartbeats(true);
    let _no_hb_rx = broadcaster.subscribe_filtered(no_heartbeat_filter);
    println!("Created subscription excluding heartbeats");

    // Subscribe for critical alerts only
    let alert_filter = EventFilter::new()
        .with_event_types([EventType::AlertTriggered])
        .with_alert_levels([AlertLevel::Error, AlertLevel::Critical]);
    let _alert_rx = broadcaster.subscribe_filtered(alert_filter);
    println!("Created subscription for error/critical alerts only");

    println!("\nTotal subscribers: {}", broadcaster.subscriber_count());
}

/// Demonstrates broadcasting different event types.
async fn demonstrate_broadcasting(broadcaster: &Arc<EventBroadcaster>) {
    println!("\n--- Broadcasting Events ---\n");

    // Quote updated event
    let quote_event = MarketMakerEvent::QuoteUpdated {
        symbol: "BTC".to_string(),
        expiration: "20240329".to_string(),
        strike: 50000,
        style: OptionStyle::Call,
        bid_price: 500,
        ask_price: 520,
        bid_size: 10,
        ask_size: 10,
        theo: 510,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(quote_event).await;
    println!("Broadcast QuoteUpdated event to {} subscribers", count);

    // Order filled event
    let fill_event = MarketMakerEvent::OrderFilled {
        order_id: "ORD-001".to_string(),
        symbol: "BTC".to_string(),
        instrument: "BTC-20240329-50000-C".to_string(),
        side: Side::Buy,
        quantity: 5,
        price: 505,
        fee: 2,
        edge: 25,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(fill_event).await;
    println!("Broadcast OrderFilled event to {} subscribers", count);

    // Order cancelled event
    let cancel_event = MarketMakerEvent::OrderCancelled {
        order_id: "ORD-002".to_string(),
        symbol: "ETH".to_string(),
        instrument: "ETH-20240329-3000-P".to_string(),
        reason: CancelReason::PriceChange,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(cancel_event).await;
    println!("Broadcast OrderCancelled event to {} subscribers", count);

    // Greeks updated event
    let greeks_event = MarketMakerEvent::GreeksUpdated {
        symbol: Some("BTC".to_string()),
        delta: 0.55,
        gamma: 0.02,
        vega: 150.0,
        theta: -25.0,
        rho: 10.0,
        dollar_delta: 27500.0,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(greeks_event).await;
    println!("Broadcast GreeksUpdated event to {} subscribers", count);

    // Position changed event
    let position_event = MarketMakerEvent::PositionChanged {
        symbol: "BTC".to_string(),
        instrument: "BTC-20240329-50000-C".to_string(),
        old_quantity: 0,
        new_quantity: 5,
        avg_price: 505,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(position_event).await;
    println!("Broadcast PositionChanged event to {} subscribers", count);

    // P&L updated event
    let pnl_event = MarketMakerEvent::PnLUpdated {
        symbol: None, // Portfolio level
        realized_pnl: 1500,
        unrealized_pnl: 2500,
        total_pnl: 4000,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(pnl_event).await;
    println!("Broadcast PnLUpdated event to {} subscribers", count);

    // Alert triggered event
    let alert_event = MarketMakerEvent::AlertTriggered {
        level: AlertLevel::Warning,
        category: AlertCategory::Risk,
        message: "Delta exposure approaching limit".to_string(),
        details: Some(serde_json::json!({
            "current_delta": 45000,
            "limit": 50000,
            "utilization": 0.90
        })),
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(alert_event).await;
    println!("Broadcast AlertTriggered event to {} subscribers", count);

    // Circuit breaker changed event
    let cb_event = MarketMakerEvent::CircuitBreakerChanged {
        previous_state: CircuitBreakerState::Normal,
        new_state: CircuitBreakerState::Warning,
        reason: "High volatility detected".to_string(),
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(cb_event).await;
    println!(
        "Broadcast CircuitBreakerChanged event to {} subscribers",
        count
    );

    // Config changed event
    let config_event = MarketMakerEvent::ConfigChanged {
        key: "spread_multiplier".to_string(),
        old_value: Some(serde_json::json!(1.0)),
        new_value: serde_json::json!(1.5),
        changed_by: "admin".to_string(),
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(config_event).await;
    println!("Broadcast ConfigChanged event to {} subscribers", count);

    // Underlying price updated event
    let price_event = MarketMakerEvent::UnderlyingPriceUpdated {
        symbol: "BTC".to_string(),
        price: 50000,
        change_pct: 2.5,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(price_event).await;
    println!(
        "Broadcast UnderlyingPriceUpdated event to {} subscribers",
        count
    );

    // Hedge executed event
    let hedge_event = MarketMakerEvent::HedgeExecuted {
        symbol: "BTC".to_string(),
        instrument: "BTC-PERP".to_string(),
        side: Side::Sell,
        quantity: 10,
        price: 50000,
        reason: HedgeReason::DeltaLimit,
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(hedge_event).await;
    println!("Broadcast HedgeExecuted event to {} subscribers", count);

    // System status changed event
    let status_event = MarketMakerEvent::SystemStatusChanged {
        component: "quoter".to_string(),
        old_status: SystemStatus::Starting,
        new_status: SystemStatus::Running,
        message: Some("Quoter initialized successfully".to_string()),
        timestamp: current_timestamp(),
    };
    let count = broadcaster.broadcast(status_event).await;
    println!(
        "Broadcast SystemStatusChanged event to {} subscribers",
        count
    );

    // Heartbeat event
    let heartbeat = MarketMakerEvent::Heartbeat {
        timestamp: current_timestamp(),
        sequence: broadcaster.current_sequence(),
    };
    let count = broadcaster.broadcast(heartbeat).await;
    println!("Broadcast Heartbeat event to {} subscribers", count);

    println!("\nCurrent sequence: {}", broadcaster.current_sequence());
}

/// Demonstrates event filtering.
async fn demonstrate_filtering(broadcaster: &Arc<EventBroadcaster>) {
    println!("\n--- Event Filtering ---\n");

    // Create a filter for BTC fills only
    let filter = EventFilter::new()
        .with_event_types([EventType::OrderFilled])
        .with_symbols(["BTC".to_string()]);

    println!("Filter configuration:");
    println!("  Event types: [OrderFilled]");
    println!("  Symbols: [BTC]");

    // Test filter matching
    let btc_fill = MarketMakerEvent::OrderFilled {
        order_id: "TEST-001".to_string(),
        symbol: "BTC".to_string(),
        instrument: "BTC-PERP".to_string(),
        side: Side::Buy,
        quantity: 1,
        price: 50000,
        fee: 1,
        edge: 10,
        timestamp: current_timestamp(),
    };

    let eth_fill = MarketMakerEvent::OrderFilled {
        order_id: "TEST-002".to_string(),
        symbol: "ETH".to_string(),
        instrument: "ETH-PERP".to_string(),
        side: Side::Buy,
        quantity: 1,
        price: 3000,
        fee: 1,
        edge: 5,
        timestamp: current_timestamp(),
    };

    let heartbeat = MarketMakerEvent::Heartbeat {
        timestamp: current_timestamp(),
        sequence: 999,
    };

    println!("\nFilter matching results:");
    println!("  BTC OrderFilled: {}", filter.matches(&btc_fill));
    println!("  ETH OrderFilled: {}", filter.matches(&eth_fill));
    println!("  Heartbeat: {}", filter.matches(&heartbeat));

    // Demonstrate filtered receiver
    let filtered_rx = broadcaster.subscribe_filtered(filter);
    println!("\nCreated filtered receiver");

    // Broadcast events
    broadcaster.broadcast(btc_fill.clone()).await;
    broadcaster.broadcast(eth_fill).await;
    broadcaster.broadcast(heartbeat).await;

    // The filtered receiver would only receive the BTC fill
    println!("Filtered receiver will only receive BTC fills");
    println!("Filter reference: {:?}", filtered_rx.filter().event_types);
}

/// Demonstrates event history for reconnection.
async fn demonstrate_history(broadcaster: &Arc<EventBroadcaster>) {
    println!("\n--- Event History ---\n");

    // Get all history
    let history = broadcaster.get_history(None).await;
    println!("Total events in history: {}", history.len());

    // Get history since a specific sequence
    let last_known_sequence = 5;
    let missed_events = broadcaster
        .get_reconnection_history(last_known_sequence)
        .await;
    println!(
        "Events since sequence {}: {}",
        last_known_sequence,
        missed_events.len()
    );

    // Display event types in history
    println!("\nEvent types in history:");
    let mut type_counts: std::collections::HashMap<EventType, usize> =
        std::collections::HashMap::new();
    for event in &history {
        *type_counts.entry(event.event_type()).or_insert(0) += 1;
    }
    for (event_type, count) in type_counts {
        println!("  {:?}: {}", event_type, count);
    }

    // Prune history (normally done automatically)
    broadcaster.prune_history().await;
    println!("\nPruned old events from history");
}

/// Demonstrates event aggregation/batching.
async fn demonstrate_aggregation() {
    println!("\n--- Event Aggregation ---\n");

    // Create an aggregator with 100ms batch interval
    let aggregator = Arc::new(EventAggregator::new(100));
    println!("Created event aggregator with 100ms interval");

    // Add events to the batch
    for i in 0..5 {
        let event = MarketMakerEvent::QuoteUpdated {
            symbol: "BTC".to_string(),
            expiration: "20240329".to_string(),
            strike: 50000,
            style: OptionStyle::Call,
            bid_price: 500 + i,
            ask_price: 520 + i,
            bid_size: 10,
            ask_size: 10,
            theo: 510 + i,
            timestamp: current_timestamp(),
        };
        aggregator.add(event).await;
    }

    println!("Added 5 events to batch");
    println!("Pending events: {}", aggregator.pending_count().await);

    // Flush the batch
    let batched_events = aggregator.flush().await;
    println!("Flushed {} events", batched_events.len());
    println!("Pending after flush: {}", aggregator.pending_count().await);

    // In production, you would use start_auto_flush:
    // let broadcaster = Arc::new(EventBroadcaster::new(config));
    // let handle = aggregator.start_auto_flush(broadcaster);
    println!("\nNote: Use start_auto_flush() for automatic batching in production");
}

/// Returns the current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
