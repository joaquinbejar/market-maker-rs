//! Event types for the market maker event system.
//!
//! This module defines all event types, enums, and filters used by the
//! event broadcasting system.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub use crate::types::action::{Action, OptionStyle};

/// Market maker event types.
///
/// All events that can be broadcast by the market maker system.
/// Events are tagged with their type for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MarketMakerEvent {
    /// Quote updated for an option.
    QuoteUpdated {
        /// Underlying symbol.
        symbol: String,
        /// Expiration date string (e.g., "20240329").
        expiration: String,
        /// Strike price in cents/smallest unit.
        strike: u64,
        /// Option style, serialized as `"call"`/`"put"`.
        #[serde(with = "crate::types::action::serde_lowercase::option_style")]
        style: OptionStyle,
        /// Bid price in cents/smallest unit.
        bid_price: u64,
        /// Ask price in cents/smallest unit.
        ask_price: u64,
        /// Bid size in contracts.
        bid_size: u64,
        /// Ask size in contracts.
        ask_size: u64,
        /// Theoretical value in cents/smallest unit.
        theo: u64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Order was filled.
    OrderFilled {
        /// Unique order identifier.
        order_id: String,
        /// Underlying symbol.
        symbol: String,
        /// Instrument identifier (e.g., "BTC-20240329-50000-C").
        instrument: String,
        /// Order side, serialized as `"buy"`/`"sell"`; any other action fails
        /// to serialize.
        #[serde(with = "crate::types::action::serde_lowercase::action")]
        side: Action,
        /// Filled quantity in contracts.
        quantity: u64,
        /// Fill price in cents/smallest unit.
        price: u64,
        /// Fee paid in cents/smallest unit.
        fee: u64,
        /// Edge captured in cents/smallest unit (can be negative).
        edge: i64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Order was cancelled.
    OrderCancelled {
        /// Unique order identifier.
        order_id: String,
        /// Underlying symbol.
        symbol: String,
        /// Instrument identifier.
        instrument: String,
        /// Reason for cancellation.
        reason: CancelReason,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Portfolio Greeks updated.
    GreeksUpdated {
        /// Symbol (None for portfolio-level).
        symbol: Option<String>,
        /// Delta exposure.
        delta: f64,
        /// Gamma exposure.
        gamma: f64,
        /// Vega exposure.
        vega: f64,
        /// Theta decay.
        theta: f64,
        /// Rho exposure.
        rho: f64,
        /// Dollar delta (delta * underlying price * multiplier).
        dollar_delta: f64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Position changed.
    PositionChanged {
        /// Underlying symbol.
        symbol: String,
        /// Instrument identifier.
        instrument: String,
        /// Previous quantity (signed, negative for short).
        old_quantity: i64,
        /// New quantity (signed, negative for short).
        new_quantity: i64,
        /// Average entry price in cents/smallest unit.
        avg_price: u64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// P&L updated.
    PnLUpdated {
        /// Symbol (None for portfolio-level).
        symbol: Option<String>,
        /// Realized P&L in cents/smallest unit.
        realized_pnl: i64,
        /// Unrealized P&L in cents/smallest unit.
        unrealized_pnl: i64,
        /// Total P&L in cents/smallest unit.
        total_pnl: i64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Alert triggered.
    AlertTriggered {
        /// Alert severity level.
        level: AlertLevel,
        /// Alert category.
        category: AlertCategory,
        /// Human-readable message.
        message: String,
        /// Additional details as JSON.
        details: Option<serde_json::Value>,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Circuit breaker state changed.
    CircuitBreakerChanged {
        /// Previous state.
        previous_state: CircuitBreakerState,
        /// New state.
        new_state: CircuitBreakerState,
        /// Reason for state change.
        reason: String,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Configuration changed.
    ConfigChanged {
        /// Configuration key that changed.
        key: String,
        /// Previous value (None if new key).
        old_value: Option<serde_json::Value>,
        /// New value.
        new_value: serde_json::Value,
        /// Who made the change.
        changed_by: String,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Underlying price updated.
    UnderlyingPriceUpdated {
        /// Underlying symbol.
        symbol: String,
        /// Current price in cents/smallest unit.
        price: u64,
        /// Percentage change from previous price.
        change_pct: f64,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Hedge executed.
    HedgeExecuted {
        /// Underlying symbol.
        symbol: String,
        /// Instrument identifier.
        instrument: String,
        /// Order side, serialized as `"buy"`/`"sell"`; any other action fails
        /// to serialize.
        #[serde(with = "crate::types::action::serde_lowercase::action")]
        side: Action,
        /// Quantity hedged.
        quantity: u64,
        /// Execution price in cents/smallest unit.
        price: u64,
        /// Reason for hedge.
        reason: HedgeReason,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// System status changed.
    SystemStatusChanged {
        /// Component name.
        component: String,
        /// Previous status.
        old_status: SystemStatus,
        /// New status.
        new_status: SystemStatus,
        /// Optional message with details.
        message: Option<String>,
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
    },

    /// Heartbeat for connection keep-alive.
    Heartbeat {
        /// Event timestamp in milliseconds since epoch.
        timestamp: u64,
        /// Sequence number for ordering.
        sequence: u64,
    },
}

impl MarketMakerEvent {
    /// Returns the event type for filtering purposes.
    #[must_use]
    pub fn event_type(&self) -> EventType {
        match self {
            Self::QuoteUpdated { .. } => EventType::QuoteUpdated,
            Self::OrderFilled { .. } => EventType::OrderFilled,
            Self::OrderCancelled { .. } => EventType::OrderCancelled,
            Self::GreeksUpdated { .. } => EventType::GreeksUpdated,
            Self::PositionChanged { .. } => EventType::PositionChanged,
            Self::PnLUpdated { .. } => EventType::PnLUpdated,
            Self::AlertTriggered { .. } => EventType::AlertTriggered,
            Self::CircuitBreakerChanged { .. } => EventType::CircuitBreakerChanged,
            Self::ConfigChanged { .. } => EventType::ConfigChanged,
            Self::UnderlyingPriceUpdated { .. } => EventType::UnderlyingPriceUpdated,
            Self::HedgeExecuted { .. } => EventType::HedgeExecuted,
            Self::SystemStatusChanged { .. } => EventType::SystemStatusChanged,
            Self::Heartbeat { .. } => EventType::Heartbeat,
        }
    }

    /// Returns the symbol associated with this event, if any.
    #[must_use]
    pub fn symbol(&self) -> Option<&str> {
        match self {
            Self::QuoteUpdated { symbol, .. }
            | Self::OrderFilled { symbol, .. }
            | Self::OrderCancelled { symbol, .. }
            | Self::PositionChanged { symbol, .. }
            | Self::UnderlyingPriceUpdated { symbol, .. }
            | Self::HedgeExecuted { symbol, .. } => Some(symbol),
            Self::GreeksUpdated { symbol, .. } | Self::PnLUpdated { symbol, .. } => {
                symbol.as_deref()
            }
            Self::AlertTriggered { .. }
            | Self::CircuitBreakerChanged { .. }
            | Self::ConfigChanged { .. }
            | Self::SystemStatusChanged { .. }
            | Self::Heartbeat { .. } => None,
        }
    }

    /// Returns the timestamp of this event in milliseconds since epoch.
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::QuoteUpdated { timestamp, .. }
            | Self::OrderFilled { timestamp, .. }
            | Self::OrderCancelled { timestamp, .. }
            | Self::GreeksUpdated { timestamp, .. }
            | Self::PositionChanged { timestamp, .. }
            | Self::PnLUpdated { timestamp, .. }
            | Self::AlertTriggered { timestamp, .. }
            | Self::CircuitBreakerChanged { timestamp, .. }
            | Self::ConfigChanged { timestamp, .. }
            | Self::UnderlyingPriceUpdated { timestamp, .. }
            | Self::HedgeExecuted { timestamp, .. }
            | Self::SystemStatusChanged { timestamp, .. }
            | Self::Heartbeat { timestamp, .. } => *timestamp,
        }
    }

    /// Returns true if this is a heartbeat event.
    #[must_use]
    pub fn is_heartbeat(&self) -> bool {
        matches!(self, Self::Heartbeat { .. })
    }

    /// Returns the alert level if this is an alert event.
    #[must_use]
    pub fn alert_level(&self) -> Option<&AlertLevel> {
        match self {
            Self::AlertTriggered { level, .. } => Some(level),
            _ => None,
        }
    }
}

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    /// Informational alert.
    Info,
    /// Warning alert.
    Warning,
    /// Error alert.
    Error,
    /// Critical alert requiring immediate attention.
    Critical,
}

/// Alert category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertCategory {
    /// Risk-related alert.
    Risk,
    /// Position-related alert.
    Position,
    /// Execution-related alert.
    Execution,
    /// System-related alert.
    System,
    /// Configuration-related alert.
    Configuration,
}

/// Cancel reason for order cancellation events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    /// User requested cancellation.
    UserRequested,
    /// Risk limit triggered.
    RiskLimit,
    /// Circuit breaker triggered.
    CircuitBreaker,
    /// Price changed significantly.
    PriceChange,
    /// Order timed out.
    Timeout,
    /// System shutdown.
    SystemShutdown,
}

/// Hedge reason for hedge execution events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HedgeReason {
    /// Delta limit exceeded.
    DeltaLimit,
    /// Automatic hedge trigger.
    AutoHedge,
    /// Manual hedge request.
    ManualRequest,
    /// Risk reduction hedge.
    RiskReduction,
}

/// System status for status change events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemStatus {
    /// System is starting up.
    Starting,
    /// System is running normally.
    Running,
    /// System is paused.
    Paused,
    /// System encountered an error.
    Error,
    /// System is stopping.
    Stopping,
    /// System is stopped.
    Stopped,
}

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitBreakerState {
    /// Normal operation.
    Normal,
    /// Warning state, approaching limits.
    Warning,
    /// Circuit breaker tripped, trading halted.
    Tripped,
    /// Cooldown period after trip.
    Cooldown,
}

/// Event type enum for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// Quote updated event.
    QuoteUpdated,
    /// Order filled event.
    OrderFilled,
    /// Order cancelled event.
    OrderCancelled,
    /// Greeks updated event.
    GreeksUpdated,
    /// Position changed event.
    PositionChanged,
    /// P&L updated event.
    PnLUpdated,
    /// Alert triggered event.
    AlertTriggered,
    /// Circuit breaker changed event.
    CircuitBreakerChanged,
    /// Configuration changed event.
    ConfigChanged,
    /// Underlying price updated event.
    UnderlyingPriceUpdated,
    /// Hedge executed event.
    HedgeExecuted,
    /// System status changed event.
    SystemStatusChanged,
    /// Heartbeat event.
    Heartbeat,
}

/// Event filter for subscriptions.
///
/// Allows filtering events by type, symbol, or alert level.
/// All filter criteria are optional; if not set, all events pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventFilter {
    /// Filter by event types (None = all types).
    pub event_types: Option<HashSet<EventType>>,
    /// Filter by symbols (None = all symbols).
    pub symbols: Option<HashSet<String>>,
    /// Filter by alert levels for alert events (None = all levels).
    pub alert_levels: Option<HashSet<AlertLevel>>,
    /// Exclude heartbeat events.
    pub exclude_heartbeats: bool,
}

impl EventFilter {
    /// Creates a new empty filter that passes all events.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a filter for specific event types.
    #[must_use]
    pub fn with_event_types(mut self, types: impl IntoIterator<Item = EventType>) -> Self {
        self.event_types = Some(types.into_iter().collect());
        self
    }

    /// Creates a filter for specific symbols.
    #[must_use]
    pub fn with_symbols(mut self, symbols: impl IntoIterator<Item = String>) -> Self {
        self.symbols = Some(symbols.into_iter().collect());
        self
    }

    /// Creates a filter for specific alert levels.
    #[must_use]
    pub fn with_alert_levels(mut self, levels: impl IntoIterator<Item = AlertLevel>) -> Self {
        self.alert_levels = Some(levels.into_iter().collect());
        self
    }

    /// Sets whether to exclude heartbeat events.
    #[must_use]
    pub fn exclude_heartbeats(mut self, exclude: bool) -> Self {
        self.exclude_heartbeats = exclude;
        self
    }

    /// Checks if an event matches this filter.
    #[must_use]
    pub fn matches(&self, event: &MarketMakerEvent) -> bool {
        // Check heartbeat exclusion
        if self.exclude_heartbeats && event.is_heartbeat() {
            return false;
        }

        // Check event type filter
        if let Some(ref types) = self.event_types
            && !types.contains(&event.event_type())
        {
            return false;
        }

        // Check symbol filter
        if let Some(ref symbols) = self.symbols
            && let Some(event_symbol) = event.symbol()
            && !symbols.contains(event_symbol)
        {
            return false;
        }
        // Events without symbols pass the symbol filter

        // Check alert level filter (only for alert events)
        if let Some(ref levels) = self.alert_levels
            && let Some(event_level) = event.alert_level()
            && !levels.contains(event_level)
        {
            return false;
        }
        // Non-alert events pass the alert level filter

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_extraction() {
        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert_eq!(event.event_type(), EventType::Heartbeat);
        assert!(event.is_heartbeat());

        let event = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert_eq!(event.event_type(), EventType::OrderFilled);
        assert!(!event.is_heartbeat());
    }

    #[test]
    fn test_event_symbol_extraction() {
        let event = MarketMakerEvent::QuoteUpdated {
            symbol: "ETH".to_string(),
            expiration: "20240329".to_string(),
            strike: 3000,
            style: OptionStyle::Call,
            bid_price: 100,
            ask_price: 110,
            bid_size: 10,
            ask_size: 10,
            theo: 105,
            timestamp: 1000,
        };
        assert_eq!(event.symbol(), Some("ETH"));

        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert_eq!(event.symbol(), None);
    }

    #[test]
    fn test_event_timestamp() {
        let event = MarketMakerEvent::Heartbeat {
            timestamp: 12345,
            sequence: 1,
        };
        assert_eq!(event.timestamp(), 12345);
    }

    #[test]
    fn test_filter_matches_all() {
        let filter = EventFilter::new();
        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert!(filter.matches(&event));
    }

    #[test]
    fn test_filter_exclude_heartbeats() {
        let filter = EventFilter::new().exclude_heartbeats(true);
        let heartbeat = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert!(!filter.matches(&heartbeat));

        let fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert!(filter.matches(&fill));
    }

    #[test]
    fn test_filter_by_event_type() {
        let filter = EventFilter::new()
            .with_event_types([EventType::OrderFilled, EventType::OrderCancelled]);

        let fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert!(filter.matches(&fill));

        let heartbeat = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert!(!filter.matches(&heartbeat));
    }

    #[test]
    fn test_filter_by_symbol() {
        let filter = EventFilter::new().with_symbols(["BTC".to_string(), "ETH".to_string()]);

        let btc_fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert!(filter.matches(&btc_fill));

        let sol_fill = MarketMakerEvent::OrderFilled {
            order_id: "124".to_string(),
            symbol: "SOL".to_string(),
            instrument: "SOL-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 100,
            fee: 1,
            edge: 10,
            timestamp: 1000,
        };
        assert!(!filter.matches(&sol_fill));

        // Events without symbols pass the filter
        let heartbeat = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 1,
        };
        assert!(filter.matches(&heartbeat));
    }

    #[test]
    fn test_filter_by_alert_level() {
        let filter =
            EventFilter::new().with_alert_levels([AlertLevel::Error, AlertLevel::Critical]);

        let error_alert = MarketMakerEvent::AlertTriggered {
            level: AlertLevel::Error,
            category: AlertCategory::Risk,
            message: "Risk limit exceeded".to_string(),
            details: None,
            timestamp: 1000,
        };
        assert!(filter.matches(&error_alert));

        let info_alert = MarketMakerEvent::AlertTriggered {
            level: AlertLevel::Info,
            category: AlertCategory::System,
            message: "System started".to_string(),
            details: None,
            timestamp: 1000,
        };
        assert!(!filter.matches(&info_alert));

        // Non-alert events pass the alert level filter
        let fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert!(filter.matches(&fill));
    }

    #[test]
    fn test_combined_filters() {
        let filter = EventFilter::new()
            .with_event_types([EventType::OrderFilled])
            .with_symbols(["BTC".to_string()])
            .exclude_heartbeats(true);

        let btc_fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };
        assert!(filter.matches(&btc_fill));

        let eth_fill = MarketMakerEvent::OrderFilled {
            order_id: "124".to_string(),
            symbol: "ETH".to_string(),
            instrument: "ETH-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 3000,
            fee: 3,
            edge: 50,
            timestamp: 1000,
        };
        assert!(!filter.matches(&eth_fill));

        let btc_cancel = MarketMakerEvent::OrderCancelled {
            order_id: "125".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            reason: CancelReason::UserRequested,
            timestamp: 1000,
        };
        assert!(!filter.matches(&btc_cancel));
    }

    #[test]
    fn test_event_serialization() {
        let event = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1000,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"order_filled"#));
        assert!(json.contains(r#""symbol":"BTC"#));

        let deserialized: MarketMakerEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, deserialized);
    }

    #[test]
    fn test_all_event_variants_serialization() {
        let events = vec![
            MarketMakerEvent::QuoteUpdated {
                symbol: "BTC".to_string(),
                expiration: "20240329".to_string(),
                strike: 50000,
                style: OptionStyle::Call,
                bid_price: 500,
                ask_price: 520,
                bid_size: 10,
                ask_size: 10,
                theo: 510,
                timestamp: 1000,
            },
            MarketMakerEvent::OrderFilled {
                order_id: "123".to_string(),
                symbol: "BTC".to_string(),
                instrument: "BTC-PERP".to_string(),
                side: Action::Buy,
                quantity: 10,
                price: 50000,
                fee: 5,
                edge: 100,
                timestamp: 1000,
            },
            MarketMakerEvent::OrderCancelled {
                order_id: "123".to_string(),
                symbol: "BTC".to_string(),
                instrument: "BTC-PERP".to_string(),
                reason: CancelReason::UserRequested,
                timestamp: 1000,
            },
            MarketMakerEvent::GreeksUpdated {
                symbol: Some("BTC".to_string()),
                delta: 0.5,
                gamma: 0.01,
                vega: 0.1,
                theta: -0.05,
                rho: 0.02,
                dollar_delta: 25000.0,
                timestamp: 1000,
            },
            MarketMakerEvent::PositionChanged {
                symbol: "BTC".to_string(),
                instrument: "BTC-PERP".to_string(),
                old_quantity: 0,
                new_quantity: 10,
                avg_price: 50000,
                timestamp: 1000,
            },
            MarketMakerEvent::PnLUpdated {
                symbol: None,
                realized_pnl: 1000,
                unrealized_pnl: 500,
                total_pnl: 1500,
                timestamp: 1000,
            },
            MarketMakerEvent::AlertTriggered {
                level: AlertLevel::Warning,
                category: AlertCategory::Risk,
                message: "Delta limit approaching".to_string(),
                details: Some(serde_json::json!({"delta": 0.8})),
                timestamp: 1000,
            },
            MarketMakerEvent::CircuitBreakerChanged {
                previous_state: CircuitBreakerState::Normal,
                new_state: CircuitBreakerState::Warning,
                reason: "High volatility".to_string(),
                timestamp: 1000,
            },
            MarketMakerEvent::ConfigChanged {
                key: "spread_multiplier".to_string(),
                old_value: Some(serde_json::json!(1.0)),
                new_value: serde_json::json!(1.5),
                changed_by: "admin".to_string(),
                timestamp: 1000,
            },
            MarketMakerEvent::UnderlyingPriceUpdated {
                symbol: "BTC".to_string(),
                price: 50000,
                change_pct: 2.5,
                timestamp: 1000,
            },
            MarketMakerEvent::HedgeExecuted {
                symbol: "BTC".to_string(),
                instrument: "BTC-PERP".to_string(),
                side: Action::Sell,
                quantity: 5,
                price: 50000,
                reason: HedgeReason::DeltaLimit,
                timestamp: 1000,
            },
            MarketMakerEvent::SystemStatusChanged {
                component: "quoter".to_string(),
                old_status: SystemStatus::Starting,
                new_status: SystemStatus::Running,
                message: Some("Quoter started successfully".to_string()),
                timestamp: 1000,
            },
            MarketMakerEvent::Heartbeat {
                timestamp: 1000,
                sequence: 1,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: MarketMakerEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    #[test]
    fn test_enum_serialization() {
        // `OptionStyle` and `Action` keep their lowercase form inside the
        // events; see the golden tests below.

        // Test AlertLevel
        assert_eq!(
            serde_json::to_string(&AlertLevel::Critical).unwrap(),
            r#""critical""#
        );

        // Test CancelReason
        assert_eq!(
            serde_json::to_string(&CancelReason::RiskLimit).unwrap(),
            r#""risk_limit""#
        );

        // Test HedgeReason
        assert_eq!(
            serde_json::to_string(&HedgeReason::DeltaLimit).unwrap(),
            r#""delta_limit""#
        );

        // Test SystemStatus
        assert_eq!(
            serde_json::to_string(&SystemStatus::Running).unwrap(),
            r#""running""#
        );

        // Test CircuitBreakerState
        assert_eq!(
            serde_json::to_string(&CircuitBreakerState::Tripped).unwrap(),
            r#""tripped""#
        );
    }

    fn quote_updated(style: OptionStyle) -> MarketMakerEvent {
        MarketMakerEvent::QuoteUpdated {
            symbol: "BTC".to_string(),
            expiration: "20240329".to_string(),
            strike: 50000,
            style,
            bid_price: 100,
            ask_price: 110,
            bid_size: 1,
            ask_size: 2,
            theo: 105,
            timestamp: 1000,
        }
    }

    fn order_filled(side: Action) -> MarketMakerEvent {
        MarketMakerEvent::OrderFilled {
            order_id: "o-1".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-20240329-50000-C".to_string(),
            side,
            quantity: 3,
            price: 120,
            fee: 1,
            edge: -2,
            timestamp: 1000,
        }
    }

    fn hedge_executed(side: Action) -> MarketMakerEvent {
        MarketMakerEvent::HedgeExecuted {
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side,
            quantity: 4,
            price: 50000,
            reason: HedgeReason::DeltaLimit,
            timestamp: 1000,
        }
    }

    /// Golden JSON of every event field that carries a side or an option
    /// style; identical to the output of the former local enums.
    #[test]
    fn test_event_json_golden_side_and_style_fields() {
        let cases = [
            (
                quote_updated(OptionStyle::Call),
                r#"{"type":"quote_updated","symbol":"BTC","expiration":"20240329","strike":50000,"style":"call","bid_price":100,"ask_price":110,"bid_size":1,"ask_size":2,"theo":105,"timestamp":1000}"#,
            ),
            (
                quote_updated(OptionStyle::Put),
                r#"{"type":"quote_updated","symbol":"BTC","expiration":"20240329","strike":50000,"style":"put","bid_price":100,"ask_price":110,"bid_size":1,"ask_size":2,"theo":105,"timestamp":1000}"#,
            ),
            (
                order_filled(Action::Buy),
                r#"{"type":"order_filled","order_id":"o-1","symbol":"BTC","instrument":"BTC-20240329-50000-C","side":"buy","quantity":3,"price":120,"fee":1,"edge":-2,"timestamp":1000}"#,
            ),
            (
                order_filled(Action::Sell),
                r#"{"type":"order_filled","order_id":"o-1","symbol":"BTC","instrument":"BTC-20240329-50000-C","side":"sell","quantity":3,"price":120,"fee":1,"edge":-2,"timestamp":1000}"#,
            ),
            (
                hedge_executed(Action::Buy),
                r#"{"type":"hedge_executed","symbol":"BTC","instrument":"BTC-PERP","side":"buy","quantity":4,"price":50000,"reason":"delta_limit","timestamp":1000}"#,
            ),
            (
                hedge_executed(Action::Sell),
                r#"{"type":"hedge_executed","symbol":"BTC","instrument":"BTC-PERP","side":"sell","quantity":4,"price":50000,"reason":"delta_limit","timestamp":1000}"#,
            ),
        ];
        for (event, golden) in cases {
            assert_eq!(serde_json::to_string(&event).unwrap(), golden);
            let back: MarketMakerEvent = serde_json::from_str(golden).unwrap();
            assert_eq!(back, event);
            // Also through `serde_json::Value`, which hands owned strings to
            // the field deserializers.
            let value: serde_json::Value = serde_json::from_str(golden).unwrap();
            assert_eq!(
                serde_json::from_value::<MarketMakerEvent>(value).unwrap(),
                event
            );
        }
    }

    /// The events accept only the lowercase spellings, as before.
    #[test]
    fn test_event_json_rejects_other_spellings() {
        for (field, value) in [
            ("side", "other"),
            ("side", "Buy"),
            ("side", "Other"),
            ("side", "BUY"),
        ] {
            let json = format!(
                r#"{{"type":"order_filled","order_id":"o-1","symbol":"BTC","instrument":"I","{field}":"{value}","quantity":3,"price":120,"fee":1,"edge":-2,"timestamp":1000}}"#
            );
            assert!(
                serde_json::from_str::<MarketMakerEvent>(&json).is_err(),
                "{value}"
            );
        }
        for value in ["Call", "american", "CALL"] {
            let json = format!(
                r#"{{"type":"quote_updated","symbol":"BTC","expiration":"E","strike":1,"style":"{value}","bid_price":1,"ask_price":2,"bid_size":1,"ask_size":1,"theo":1,"timestamp":1}}"#
            );
            assert!(
                serde_json::from_str::<MarketMakerEvent>(&json).is_err(),
                "{value}"
            );
        }
    }

    /// An event never serializes a side other than buy or sell.
    #[test]
    fn test_event_json_unsupported_action_fails_to_serialize() {
        for event in [order_filled(Action::Other), hedge_executed(Action::Other)] {
            let error = serde_json::to_string(&event).unwrap_err();
            assert!(
                error.to_string().contains("unsupported action Other"),
                "{error}"
            );
        }
    }

    /// Serializing the shared enums directly (outside an event) uses their
    /// own representation, unlike the former local enums.
    #[test]
    fn test_shared_enums_direct_serialization() {
        assert_eq!(serde_json::to_string(&Action::Buy).unwrap(), r#""Buy""#);
        assert_eq!(
            serde_json::to_string(&OptionStyle::Call).unwrap(),
            r#""Call""#
        );
    }
}
