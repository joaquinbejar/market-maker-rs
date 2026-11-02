//! Event system for broadcasting market maker events.
//!
//! This module provides a structured event handling system for broadcasting
//! market maker events to the frontend and other consumers, with support for
//! event history and reconnection.
//!
//! # Features
//!
//! - **Event Broadcasting**: Distribute real-time events to multiple consumers
//! - **Event Filtering**: Subscribe to specific event types or symbols
//! - **Event History**: Buffer for reconnection scenarios
//! - **Event Batching**: Aggregate high-frequency updates
//! - **Heartbeat**: Connection keep-alive mechanism
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::events::{EventBroadcaster, EventBroadcasterConfig, MarketMakerEvent};
//!
//! // Create broadcaster
//! let config = EventBroadcasterConfig::default();
//! let broadcaster = EventBroadcaster::new(config);
//!
//! // Subscribe to events
//! let mut rx = broadcaster.subscribe();
//!
//! // Broadcast an event
//! broadcaster.broadcast(MarketMakerEvent::Heartbeat {
//!     timestamp: 1234567890,
//!     sequence: 1,
//! });
//! ```

mod broadcaster;
mod types;

pub use broadcaster::{
    EventAggregator, EventBroadcaster, EventBroadcasterConfig, EventHistory, FilteredEventReceiver,
    TimestampedEvent,
};
pub use types::{
    Action, AlertCategory, AlertLevel, CancelReason, CircuitBreakerState, EventFilter, EventType,
    HedgeReason, MarketMakerEvent, OptionStyle, SystemStatus,
};
