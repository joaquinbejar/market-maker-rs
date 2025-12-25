//! Alert system for critical event notification.
//!
//! This module provides configurable alerts for critical trading events,
//! enabling rapid response to issues and integration with existing
//! alerting infrastructure.
//!
//! # Overview
//!
//! The alert system includes:
//!
//! - **AlertSeverity**: Ordered severity levels (Info, Warning, Error, Critical)
//! - **AlertType**: Common alert scenarios (loss limits, position limits, etc.)
//! - **Alert**: Individual alert instance with metadata
//! - **AlertHandler**: Trait for custom alert handling
//! - **AlertManager**: Coordinates handlers and manages alert history
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::risk::alerts::{
//!     AlertManager, AlertSeverity, AlertType, LogAlertHandler,
//! };
//! use market_maker_rs::dec;
//!
//! // Create alert manager
//! let mut manager = AlertManager::new(100, 60000); // 100 history, 60s dedup
//!
//! // Add log handler for warnings and above
//! manager.add_handler(Box::new(LogAlertHandler::new(AlertSeverity::Warning)));
//!
//! // Raise an alert
//! manager.alert(
//!     AlertType::LargeLoss {
//!         amount: dec!(1000.0),
//!         threshold: dec!(500.0),
//!     },
//!     AlertSeverity::Error,
//!     1000,
//! );
//!
//! // Check unacknowledged alerts
//! println!("Unacknowledged: {}", manager.unacknowledged_count());
//! ```

use crate::Decimal;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Global counter for unique alert IDs.
static ALERT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Alert severity levels, ordered from least to most severe.
///
/// Severity levels can be compared using standard comparison operators.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::alerts::AlertSeverity;
///
/// assert!(AlertSeverity::Critical > AlertSeverity::Error);
/// assert!(AlertSeverity::Warning > AlertSeverity::Info);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertSeverity {
    /// Informational alert, no action required.
    Info,
    /// Warning, should be monitored.
    Warning,
    /// Error, requires attention.
    Error,
    /// Critical, immediate action required.
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl AlertSeverity {
    /// Returns all severity levels in order.
    #[must_use]
    pub fn all() -> &'static [AlertSeverity] {
        &[Self::Info, Self::Warning, Self::Error, Self::Critical]
    }
}

/// Types of alerts that can be raised.
///
/// Each variant contains context-specific data about the alert condition.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertType {
    /// Large single-trade loss detected.
    LargeLoss {
        /// The loss amount.
        amount: Decimal,
        /// The configured threshold.
        threshold: Decimal,
    },

    /// Daily loss limit approaching or reached.
    DailyLossLimit {
        /// Current daily loss.
        current: Decimal,
        /// Configured limit.
        limit: Decimal,
        /// Percentage of limit used.
        pct: Decimal,
    },

    /// Position limit approaching or reached.
    PositionLimit {
        /// Current position size.
        current: Decimal,
        /// Configured limit.
        limit: Decimal,
        /// Percentage of limit used.
        pct: Decimal,
    },

    /// Maximum drawdown threshold reached.
    MaxDrawdown {
        /// Current drawdown.
        drawdown: Decimal,
        /// Configured threshold.
        threshold: Decimal,
    },

    /// Connectivity issue with exchange.
    ConnectivityIssue {
        /// Exchange name.
        exchange: String,
        /// Error description.
        error: String,
    },

    /// High latency detected.
    HighLatency {
        /// Metric name (e.g., "order_submission").
        metric: String,
        /// Observed latency in milliseconds.
        latency_ms: u64,
        /// Configured threshold in milliseconds.
        threshold_ms: u64,
    },

    /// Strategy error occurred.
    StrategyError {
        /// Error message.
        message: String,
    },

    /// Circuit breaker was triggered.
    CircuitBreakerTriggered {
        /// Reason for trigger.
        reason: String,
    },

    /// Order was rejected.
    OrderRejected {
        /// Rejection reason.
        reason: String,
        /// Order details.
        order_details: String,
    },

    /// Unusual market conditions detected.
    MarketCondition {
        /// Condition type.
        condition: String,
        /// Additional details.
        details: String,
    },

    /// Custom alert type.
    Custom {
        /// Alert name.
        name: String,
        /// Alert message.
        message: String,
    },
}

impl AlertType {
    /// Returns a string identifier for this alert type (used for deduplication).
    #[must_use]
    pub fn type_key(&self) -> String {
        match self {
            Self::LargeLoss { .. } => "large_loss".to_string(),
            Self::DailyLossLimit { .. } => "daily_loss_limit".to_string(),
            Self::PositionLimit { .. } => "position_limit".to_string(),
            Self::MaxDrawdown { .. } => "max_drawdown".to_string(),
            Self::ConnectivityIssue { exchange, .. } => format!("connectivity_{}", exchange),
            Self::HighLatency { metric, .. } => format!("high_latency_{}", metric),
            Self::StrategyError { .. } => "strategy_error".to_string(),
            Self::CircuitBreakerTriggered { .. } => "circuit_breaker".to_string(),
            Self::OrderRejected { .. } => "order_rejected".to_string(),
            Self::MarketCondition { condition, .. } => format!("market_{}", condition),
            Self::Custom { name, .. } => format!("custom_{}", name),
        }
    }

    /// Generates a default message for this alert type.
    #[must_use]
    pub fn default_message(&self) -> String {
        match self {
            Self::LargeLoss { amount, threshold } => {
                format!("Large loss detected: {} (threshold: {})", amount, threshold)
            }
            Self::DailyLossLimit {
                current,
                limit,
                pct,
            } => {
                format!(
                    "Daily loss limit: {} / {} ({:.1}%)",
                    current,
                    limit,
                    pct * Decimal::from(100)
                )
            }
            Self::PositionLimit {
                current,
                limit,
                pct,
            } => {
                format!(
                    "Position limit: {} / {} ({:.1}%)",
                    current,
                    limit,
                    pct * Decimal::from(100)
                )
            }
            Self::MaxDrawdown {
                drawdown,
                threshold,
            } => {
                format!(
                    "Max drawdown reached: {:.2}% (threshold: {:.2}%)",
                    drawdown * Decimal::from(100),
                    threshold * Decimal::from(100)
                )
            }
            Self::ConnectivityIssue { exchange, error } => {
                format!("Connectivity issue with {}: {}", exchange, error)
            }
            Self::HighLatency {
                metric,
                latency_ms,
                threshold_ms,
            } => {
                format!(
                    "High latency on {}: {}ms (threshold: {}ms)",
                    metric, latency_ms, threshold_ms
                )
            }
            Self::StrategyError { message } => {
                format!("Strategy error: {}", message)
            }
            Self::CircuitBreakerTriggered { reason } => {
                format!("Circuit breaker triggered: {}", reason)
            }
            Self::OrderRejected {
                reason,
                order_details,
            } => {
                format!("Order rejected: {} ({})", reason, order_details)
            }
            Self::MarketCondition { condition, details } => {
                format!("Market condition {}: {}", condition, details)
            }
            Self::Custom { name, message } => {
                format!("{}: {}", name, message)
            }
        }
    }
}

impl fmt::Display for AlertType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.default_message())
    }
}

/// An individual alert instance.
///
/// Contains all metadata about an alert including its type, severity,
/// message, and acknowledgment status.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Alert {
    /// Unique alert identifier.
    pub id: String,
    /// Type of alert.
    pub alert_type: AlertType,
    /// Severity level.
    pub severity: AlertSeverity,
    /// Human-readable message.
    pub message: String,
    /// Timestamp when alert was raised (milliseconds).
    pub timestamp: u64,
    /// Whether the alert has been acknowledged.
    pub acknowledged: bool,
}

impl Alert {
    /// Creates a new alert.
    ///
    /// # Arguments
    ///
    /// * `alert_type` - Type of alert
    /// * `severity` - Severity level
    /// * `message` - Human-readable message
    /// * `timestamp` - Timestamp in milliseconds
    #[must_use]
    pub fn new(
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        timestamp: u64,
    ) -> Self {
        let counter = ALERT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = format!("{}-{}-{}", alert_type.type_key(), timestamp, counter);

        Self {
            id,
            alert_type,
            severity,
            message,
            timestamp,
            acknowledged: false,
        }
    }

    /// Creates a new alert with a default message.
    #[must_use]
    pub fn with_default_message(
        alert_type: AlertType,
        severity: AlertSeverity,
        timestamp: u64,
    ) -> Self {
        let message = alert_type.default_message();
        Self::new(alert_type, severity, message, timestamp)
    }

    /// Acknowledges this alert.
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Returns true if this alert is critical.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.severity == AlertSeverity::Critical
    }

    /// Returns true if this alert is an error or higher.
    #[must_use]
    pub fn is_error_or_higher(&self) -> bool {
        self.severity >= AlertSeverity::Error
    }
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} - {}",
            self.severity,
            self.alert_type.type_key(),
            self.message
        )
    }
}

/// Trait for handling alerts.
///
/// Implement this trait to create custom alert handlers (e.g., email,
/// Slack, PagerDuty, etc.).
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::alerts::{Alert, AlertHandler, AlertSeverity};
///
/// struct MyHandler;
///
/// impl AlertHandler for MyHandler {
///     fn handle(&self, alert: &Alert) {
///         println!("Custom handler: {}", alert);
///     }
///
///     fn accepts_severity(&self, severity: AlertSeverity) -> bool {
///         severity >= AlertSeverity::Error
///     }
/// }
/// ```
pub trait AlertHandler: Send + Sync {
    /// Handle an alert.
    fn handle(&self, alert: &Alert);

    /// Check if this handler accepts the given severity level.
    ///
    /// Default implementation accepts all severities.
    fn accepts_severity(&self, _severity: AlertSeverity) -> bool {
        true
    }

    /// Returns the handler name for debugging.
    fn name(&self) -> &str {
        "AlertHandler"
    }
}

/// Log-based alert handler.
///
/// Outputs alerts to the standard logging system using appropriate
/// log levels based on severity.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::alerts::{LogAlertHandler, AlertSeverity};
///
/// // Only log warnings and above
/// let handler = LogAlertHandler::new(AlertSeverity::Warning);
/// ```
#[derive(Debug)]
pub struct LogAlertHandler {
    min_severity: AlertSeverity,
}

impl LogAlertHandler {
    /// Creates a new log alert handler.
    ///
    /// # Arguments
    ///
    /// * `min_severity` - Minimum severity level to handle
    #[must_use]
    pub fn new(min_severity: AlertSeverity) -> Self {
        Self { min_severity }
    }

    /// Creates a handler that logs all severities.
    #[must_use]
    pub fn all() -> Self {
        Self::new(AlertSeverity::Info)
    }
}

impl Default for LogAlertHandler {
    fn default() -> Self {
        Self::new(AlertSeverity::Info)
    }
}

impl AlertHandler for LogAlertHandler {
    fn handle(&self, alert: &Alert) {
        match alert.severity {
            AlertSeverity::Info => {
                // Using eprintln for now since we don't have a logging dependency
                eprintln!("[INFO] Alert {}: {}", alert.id, alert.message);
            }
            AlertSeverity::Warning => {
                eprintln!("[WARN] Alert {}: {}", alert.id, alert.message);
            }
            AlertSeverity::Error => {
                eprintln!("[ERROR] Alert {}: {}", alert.id, alert.message);
            }
            AlertSeverity::Critical => {
                eprintln!("[CRITICAL] Alert {}: {}", alert.id, alert.message);
            }
        }
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    fn name(&self) -> &str {
        "LogAlertHandler"
    }
}

/// Callback-based alert handler.
///
/// Allows using a closure as an alert handler.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::alerts::{CallbackAlertHandler, AlertSeverity};
///
/// let handler = CallbackAlertHandler::new(
///     AlertSeverity::Error,
///     |alert| println!("Got alert: {}", alert),
/// );
/// ```
pub struct CallbackAlertHandler<F>
where
    F: Fn(&Alert) + Send + Sync,
{
    min_severity: AlertSeverity,
    callback: F,
}

impl<F> CallbackAlertHandler<F>
where
    F: Fn(&Alert) + Send + Sync,
{
    /// Creates a new callback alert handler.
    ///
    /// # Arguments
    ///
    /// * `min_severity` - Minimum severity level to handle
    /// * `callback` - Function to call for each alert
    pub fn new(min_severity: AlertSeverity, callback: F) -> Self {
        Self {
            min_severity,
            callback,
        }
    }
}

impl<F> AlertHandler for CallbackAlertHandler<F>
where
    F: Fn(&Alert) + Send + Sync,
{
    fn handle(&self, alert: &Alert) {
        (self.callback)(alert);
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    fn name(&self) -> &str {
        "CallbackAlertHandler"
    }
}

impl<F> fmt::Debug for CallbackAlertHandler<F>
where
    F: Fn(&Alert) + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackAlertHandler")
            .field("min_severity", &self.min_severity)
            .finish()
    }
}

/// Collects alerts into a vector for testing.
///
/// This handler is useful for unit tests to verify alerts are raised correctly.
#[derive(Debug)]
pub struct CollectingAlertHandler {
    min_severity: AlertSeverity,
    alerts: std::sync::Mutex<Vec<Alert>>,
}

impl CollectingAlertHandler {
    /// Creates a new collecting handler.
    #[must_use]
    pub fn new(min_severity: AlertSeverity) -> Self {
        Self {
            min_severity,
            alerts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Returns collected alerts.
    #[must_use]
    pub fn alerts(&self) -> Vec<Alert> {
        self.alerts.lock().unwrap().clone()
    }

    /// Returns the number of collected alerts.
    #[must_use]
    pub fn count(&self) -> usize {
        self.alerts.lock().unwrap().len()
    }

    /// Clears collected alerts.
    pub fn clear(&self) {
        self.alerts.lock().unwrap().clear();
    }
}

impl AlertHandler for CollectingAlertHandler {
    fn handle(&self, alert: &Alert) {
        self.alerts.lock().unwrap().push(alert.clone());
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    fn name(&self) -> &str {
        "CollectingAlertHandler"
    }
}

/// Alert manager that coordinates handlers and manages alert history.
///
/// The manager supports:
/// - Multiple handlers with severity filtering
/// - Alert deduplication within a time window
/// - Alert history with configurable size
/// - Acknowledgment tracking
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::alerts::{AlertManager, AlertSeverity, AlertType, LogAlertHandler};
/// use market_maker_rs::dec;
///
/// let mut manager = AlertManager::new(100, 60000);
/// manager.add_handler(Box::new(LogAlertHandler::new(AlertSeverity::Warning)));
///
/// // Raise alerts
/// manager.alert(
///     AlertType::StrategyError { message: "Test error".to_string() },
///     AlertSeverity::Error,
///     1000,
/// );
///
/// // Check status
/// assert_eq!(manager.unacknowledged_count(), 1);
/// ```
pub struct AlertManager {
    handlers: Vec<Box<dyn AlertHandler>>,
    alert_history: VecDeque<Alert>,
    max_history: usize,
    dedup_window_ms: u64,
    recent_alerts: HashMap<String, u64>,
}

impl AlertManager {
    /// Creates a new alert manager.
    ///
    /// # Arguments
    ///
    /// * `max_history` - Maximum number of alerts to keep in history
    /// * `dedup_window_ms` - Time window for deduplication in milliseconds
    #[must_use]
    pub fn new(max_history: usize, dedup_window_ms: u64) -> Self {
        Self {
            handlers: Vec::new(),
            alert_history: VecDeque::with_capacity(max_history),
            max_history,
            dedup_window_ms,
            recent_alerts: HashMap::new(),
        }
    }

    /// Creates a new alert manager with default settings.
    ///
    /// Default: 1000 history, 60 second dedup window.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(1000, 60_000)
    }

    /// Adds an alert handler.
    pub fn add_handler(&mut self, handler: Box<dyn AlertHandler>) {
        self.handlers.push(handler);
    }

    /// Raises an alert with a default message.
    ///
    /// # Arguments
    ///
    /// * `alert_type` - Type of alert
    /// * `severity` - Severity level
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn alert(&mut self, alert_type: AlertType, severity: AlertSeverity, timestamp: u64) {
        let message = alert_type.default_message();
        self.alert_with_message(alert_type, severity, message, timestamp);
    }

    /// Raises an alert with a custom message.
    ///
    /// # Arguments
    ///
    /// * `alert_type` - Type of alert
    /// * `severity` - Severity level
    /// * `message` - Custom message
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn alert_with_message(
        &mut self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        timestamp: u64,
    ) {
        let type_key = alert_type.type_key();

        // Check for deduplication
        if let Some(&last_time) = self.recent_alerts.get(&type_key)
            && timestamp.saturating_sub(last_time) < self.dedup_window_ms
        {
            // Duplicate within window, skip
            return;
        }

        // Create alert
        let alert = Alert::new(alert_type, severity, message, timestamp);

        // Dispatch to handlers
        for handler in &self.handlers {
            if handler.accepts_severity(severity) {
                handler.handle(&alert);
            }
        }

        // Update dedup tracking
        self.recent_alerts.insert(type_key, timestamp);

        // Add to history
        self.alert_history.push_back(alert);

        // Trim history if needed
        while self.alert_history.len() > self.max_history {
            self.alert_history.pop_front();
        }
    }

    /// Returns recent alerts.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of alerts to return
    #[must_use]
    pub fn get_recent_alerts(&self, count: usize) -> Vec<&Alert> {
        self.alert_history.iter().rev().take(count).collect()
    }

    /// Returns alerts filtered by severity.
    #[must_use]
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.alert_history
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Returns alerts at or above the given severity.
    #[must_use]
    pub fn get_alerts_at_or_above(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.alert_history
            .iter()
            .filter(|a| a.severity >= severity)
            .collect()
    }

    /// Acknowledges an alert by ID.
    ///
    /// Returns true if the alert was found and acknowledged.
    pub fn acknowledge(&mut self, alert_id: &str) -> bool {
        for alert in &mut self.alert_history {
            if alert.id == alert_id {
                alert.acknowledged = true;
                return true;
            }
        }
        false
    }

    /// Acknowledges all alerts.
    pub fn acknowledge_all(&mut self) {
        for alert in &mut self.alert_history {
            alert.acknowledged = true;
        }
    }

    /// Returns the count of unacknowledged alerts.
    #[must_use]
    pub fn unacknowledged_count(&self) -> usize {
        self.alert_history
            .iter()
            .filter(|a| !a.acknowledged)
            .count()
    }

    /// Returns unacknowledged alerts.
    #[must_use]
    pub fn get_unacknowledged(&self) -> Vec<&Alert> {
        self.alert_history
            .iter()
            .filter(|a| !a.acknowledged)
            .collect()
    }

    /// Clears old alerts from history.
    ///
    /// # Arguments
    ///
    /// * `max_age_ms` - Maximum age of alerts to keep
    /// * `current_time` - Current timestamp in milliseconds
    pub fn cleanup(&mut self, max_age_ms: u64, current_time: u64) {
        let cutoff = current_time.saturating_sub(max_age_ms);

        // Remove old alerts from history
        self.alert_history.retain(|a| a.timestamp >= cutoff);

        // Clean up dedup tracking
        self.recent_alerts.retain(|_, &mut ts| ts >= cutoff);
    }

    /// Returns the total number of alerts in history.
    #[must_use]
    pub fn history_count(&self) -> usize {
        self.alert_history.len()
    }

    /// Returns the number of registered handlers.
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Clears all alert history.
    pub fn clear_history(&mut self) {
        self.alert_history.clear();
        self.recent_alerts.clear();
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl fmt::Debug for AlertManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlertManager")
            .field("handler_count", &self.handlers.len())
            .field("history_count", &self.alert_history.len())
            .field("max_history", &self.max_history)
            .field("dedup_window_ms", &self.dedup_window_ms)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;
    use std::sync::Arc;

    // AlertSeverity tests
    #[test]
    fn test_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Error);
        assert!(AlertSeverity::Error < AlertSeverity::Critical);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
        assert_eq!(AlertSeverity::Warning.to_string(), "WARNING");
        assert_eq!(AlertSeverity::Error.to_string(), "ERROR");
        assert_eq!(AlertSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_severity_all() {
        let all = AlertSeverity::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], AlertSeverity::Info);
        assert_eq!(all[3], AlertSeverity::Critical);
    }

    // AlertType tests
    #[test]
    fn test_alert_type_key() {
        let alert = AlertType::LargeLoss {
            amount: dec!(100.0),
            threshold: dec!(50.0),
        };
        assert_eq!(alert.type_key(), "large_loss");

        let alert = AlertType::ConnectivityIssue {
            exchange: "binance".to_string(),
            error: "timeout".to_string(),
        };
        assert_eq!(alert.type_key(), "connectivity_binance");
    }

    #[test]
    fn test_alert_type_default_message() {
        let alert = AlertType::LargeLoss {
            amount: dec!(100.0),
            threshold: dec!(50.0),
        };
        let msg = alert.default_message();
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_alert_type_display() {
        let alert = AlertType::StrategyError {
            message: "test error".to_string(),
        };
        let display = format!("{}", alert);
        assert!(display.contains("test error"));
    }

    // Alert tests
    #[test]
    fn test_alert_new() {
        let alert = Alert::new(
            AlertType::StrategyError {
                message: "test".to_string(),
            },
            AlertSeverity::Error,
            "Test message".to_string(),
            1000,
        );

        assert!(alert.id.contains("strategy_error"));
        assert_eq!(alert.severity, AlertSeverity::Error);
        assert_eq!(alert.message, "Test message");
        assert_eq!(alert.timestamp, 1000);
        assert!(!alert.acknowledged);
    }

    #[test]
    fn test_alert_with_default_message() {
        let alert = Alert::with_default_message(
            AlertType::CircuitBreakerTriggered {
                reason: "max loss".to_string(),
            },
            AlertSeverity::Critical,
            2000,
        );

        assert!(alert.message.contains("max loss"));
    }

    #[test]
    fn test_alert_acknowledge() {
        let mut alert = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            "Test".to_string(),
            1000,
        );

        assert!(!alert.acknowledged);
        alert.acknowledge();
        assert!(alert.acknowledged);
    }

    #[test]
    fn test_alert_is_critical() {
        let critical = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Critical,
            "Test".to_string(),
            1000,
        );
        assert!(critical.is_critical());

        let error = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Error,
            "Test".to_string(),
            1000,
        );
        assert!(!error.is_critical());
    }

    #[test]
    fn test_alert_is_error_or_higher() {
        let error = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Error,
            "Test".to_string(),
            1000,
        );
        assert!(error.is_error_or_higher());

        let warning = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Warning,
            "Test".to_string(),
            1000,
        );
        assert!(!warning.is_error_or_higher());
    }

    // LogAlertHandler tests
    #[test]
    fn test_log_handler_accepts_severity() {
        let handler = LogAlertHandler::new(AlertSeverity::Warning);

        assert!(!handler.accepts_severity(AlertSeverity::Info));
        assert!(handler.accepts_severity(AlertSeverity::Warning));
        assert!(handler.accepts_severity(AlertSeverity::Error));
        assert!(handler.accepts_severity(AlertSeverity::Critical));
    }

    #[test]
    fn test_log_handler_all() {
        let handler = LogAlertHandler::all();
        assert!(handler.accepts_severity(AlertSeverity::Info));
    }

    // CollectingAlertHandler tests
    #[test]
    fn test_collecting_handler() {
        let handler = CollectingAlertHandler::new(AlertSeverity::Info);

        let alert = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            "Test".to_string(),
            1000,
        );

        handler.handle(&alert);
        handler.handle(&alert);

        assert_eq!(handler.count(), 2);

        let alerts = handler.alerts();
        assert_eq!(alerts.len(), 2);

        handler.clear();
        assert_eq!(handler.count(), 0);
    }

    // AlertManager tests
    #[test]
    fn test_manager_new() {
        let manager = AlertManager::new(100, 60000);
        assert_eq!(manager.history_count(), 0);
        assert_eq!(manager.handler_count(), 0);
    }

    #[test]
    fn test_manager_add_handler() {
        let mut manager = AlertManager::new(100, 60000);
        manager.add_handler(Box::new(LogAlertHandler::all()));
        assert_eq!(manager.handler_count(), 1);
    }

    #[test]
    fn test_manager_alert() {
        let collector = Arc::new(CollectingAlertHandler::new(AlertSeverity::Info));
        let mut manager = AlertManager::new(100, 60000);

        // We need to clone the Arc for the handler
        struct ArcHandler(Arc<CollectingAlertHandler>);
        impl AlertHandler for ArcHandler {
            fn handle(&self, alert: &Alert) {
                self.0.handle(alert);
            }
        }

        manager.add_handler(Box::new(ArcHandler(Arc::clone(&collector))));

        manager.alert(
            AlertType::StrategyError {
                message: "test".to_string(),
            },
            AlertSeverity::Error,
            1000,
        );

        assert_eq!(manager.history_count(), 1);
        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn test_manager_deduplication() {
        let mut manager = AlertManager::new(100, 60000);

        // First alert
        manager.alert(
            AlertType::StrategyError {
                message: "test".to_string(),
            },
            AlertSeverity::Error,
            1000,
        );

        // Duplicate within window (should be skipped)
        manager.alert(
            AlertType::StrategyError {
                message: "test2".to_string(),
            },
            AlertSeverity::Error,
            2000,
        );

        assert_eq!(manager.history_count(), 1);

        // After window (should be added)
        manager.alert(
            AlertType::StrategyError {
                message: "test3".to_string(),
            },
            AlertSeverity::Error,
            70000,
        );

        assert_eq!(manager.history_count(), 2);
    }

    #[test]
    fn test_manager_different_types_not_deduplicated() {
        let mut manager = AlertManager::new(100, 60000);

        manager.alert(
            AlertType::StrategyError {
                message: "test".to_string(),
            },
            AlertSeverity::Error,
            1000,
        );

        manager.alert(
            AlertType::CircuitBreakerTriggered {
                reason: "test".to_string(),
            },
            AlertSeverity::Error,
            1001,
        );

        assert_eq!(manager.history_count(), 2);
    }

    #[test]
    fn test_manager_history_limit() {
        let mut manager = AlertManager::new(3, 0); // No dedup, max 3 history

        for i in 0..5 {
            manager.alert(
                AlertType::Custom {
                    name: format!("test{}", i),
                    message: "msg".to_string(),
                },
                AlertSeverity::Info,
                i as u64,
            );
        }

        assert_eq!(manager.history_count(), 3);
    }

    #[test]
    fn test_manager_get_recent_alerts() {
        let mut manager = AlertManager::new(100, 0);

        for i in 0..5 {
            manager.alert(
                AlertType::Custom {
                    name: format!("test{}", i),
                    message: "msg".to_string(),
                },
                AlertSeverity::Info,
                i as u64,
            );
        }

        let recent = manager.get_recent_alerts(2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_manager_get_alerts_by_severity() {
        let mut manager = AlertManager::new(100, 0);

        manager.alert(
            AlertType::Custom {
                name: "info".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            1,
        );
        manager.alert(
            AlertType::Custom {
                name: "error".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Error,
            2,
        );
        manager.alert(
            AlertType::Custom {
                name: "error2".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Error,
            3,
        );

        let errors = manager.get_alerts_by_severity(AlertSeverity::Error);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_manager_acknowledge() {
        let mut manager = AlertManager::new(100, 0);

        manager.alert(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            1,
        );

        assert_eq!(manager.unacknowledged_count(), 1);

        let alert_id = manager.get_recent_alerts(1)[0].id.clone();
        assert!(manager.acknowledge(&alert_id));

        assert_eq!(manager.unacknowledged_count(), 0);
    }

    #[test]
    fn test_manager_acknowledge_all() {
        let mut manager = AlertManager::new(100, 0);

        for i in 0..3 {
            manager.alert(
                AlertType::Custom {
                    name: format!("test{}", i),
                    message: "msg".to_string(),
                },
                AlertSeverity::Info,
                i as u64,
            );
        }

        assert_eq!(manager.unacknowledged_count(), 3);
        manager.acknowledge_all();
        assert_eq!(manager.unacknowledged_count(), 0);
    }

    #[test]
    fn test_manager_cleanup() {
        let mut manager = AlertManager::new(100, 0);

        manager.alert(
            AlertType::Custom {
                name: "old".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            1000,
        );
        manager.alert(
            AlertType::Custom {
                name: "new".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            5000,
        );

        assert_eq!(manager.history_count(), 2);

        // Cleanup alerts older than 3000ms at time 6000
        manager.cleanup(3000, 6000);

        assert_eq!(manager.history_count(), 1);
    }

    #[test]
    fn test_manager_clear_history() {
        let mut manager = AlertManager::new(100, 0);

        manager.alert(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            1,
        );

        assert_eq!(manager.history_count(), 1);
        manager.clear_history();
        assert_eq!(manager.history_count(), 0);
    }

    #[test]
    fn test_manager_severity_filtering() {
        let collector = Arc::new(CollectingAlertHandler::new(AlertSeverity::Error));
        let mut manager = AlertManager::new(100, 0);

        struct ArcHandler(Arc<CollectingAlertHandler>);
        impl AlertHandler for ArcHandler {
            fn handle(&self, alert: &Alert) {
                self.0.handle(alert);
            }
            fn accepts_severity(&self, severity: AlertSeverity) -> bool {
                self.0.accepts_severity(severity)
            }
        }

        manager.add_handler(Box::new(ArcHandler(Arc::clone(&collector))));

        // Info alert - should not be collected
        manager.alert(
            AlertType::Custom {
                name: "info".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            1,
        );

        // Error alert - should be collected
        manager.alert(
            AlertType::Custom {
                name: "error".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Error,
            2,
        );

        // Both in history
        assert_eq!(manager.history_count(), 2);
        // Only error collected by handler
        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn test_callback_handler() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&count);

        let handler = CallbackAlertHandler::new(AlertSeverity::Info, move |_alert| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let alert = Alert::new(
            AlertType::Custom {
                name: "test".to_string(),
                message: "msg".to_string(),
            },
            AlertSeverity::Info,
            "Test".to_string(),
            1000,
        );

        handler.handle(&alert);
        handler.handle(&alert);

        assert_eq!(count.load(Ordering::Relaxed), 2);
    }
}
