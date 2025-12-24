# Implement Alert System

## Summary

Add configurable alerts for critical events to enable rapid response to issues.

## Motivation

Automated alerts help:
- Detect problems before they cause significant losses
- Enable unattended operation with safety nets
- Provide audit trail of significant events
- Integrate with existing alerting infrastructure

## Detailed Description

Implement an alert system with configurable thresholds and multiple output handlers.

### Proposed API

```rust
/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    /// Large single-trade loss
    LargeLoss { amount: Decimal, threshold: Decimal },
    
    /// Daily loss limit approaching/reached
    DailyLossLimit { current: Decimal, limit: Decimal, pct: Decimal },
    
    /// Position limit approaching/reached
    PositionLimit { current: Decimal, limit: Decimal, pct: Decimal },
    
    /// Max drawdown reached
    MaxDrawdown { drawdown: Decimal, threshold: Decimal },
    
    /// Connectivity issue
    ConnectivityIssue { exchange: String, error: String },
    
    /// High latency detected
    HighLatency { metric: String, latency_ms: u64, threshold_ms: u64 },
    
    /// Strategy error
    StrategyError { message: String },
    
    /// Circuit breaker triggered
    CircuitBreakerTriggered { reason: String },
    
    /// Order rejected
    OrderRejected { reason: String, order_details: String },
    
    /// Unusual market conditions
    MarketCondition { condition: String, details: String },
    
    /// Custom alert
    Custom { name: String, message: String },
}

/// Alert instance
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: u64,
    pub acknowledged: bool,
}

/// Alert handler trait
pub trait AlertHandler: Send + Sync {
    /// Handle an alert
    fn handle(&self, alert: &Alert);
    
    /// Check if handler accepts this severity level
    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        true  // Default: accept all
    }
}

/// Log-based alert handler (default)
pub struct LogAlertHandler {
    min_severity: AlertSeverity,
}

impl LogAlertHandler {
    pub fn new(min_severity: AlertSeverity) -> Self;
}

impl AlertHandler for LogAlertHandler {
    fn handle(&self, alert: &Alert);
    fn accepts_severity(&self, severity: AlertSeverity) -> bool;
}

/// Webhook alert handler (optional feature)
#[cfg(feature = "webhook-alerts")]
pub struct WebhookAlertHandler {
    url: String,
    min_severity: AlertSeverity,
    client: reqwest::Client,
}

#[cfg(feature = "webhook-alerts")]
impl WebhookAlertHandler {
    pub fn new(url: &str, min_severity: AlertSeverity) -> Self;
}

/// Alert manager
pub struct AlertManager {
    handlers: Vec<Box<dyn AlertHandler>>,
    alert_history: VecDeque<Alert>,
    max_history: usize,
    dedup_window_ms: u64,
    recent_alerts: HashMap<String, u64>,  // For deduplication
}

impl AlertManager {
    pub fn new(max_history: usize, dedup_window_ms: u64) -> Self;
    
    /// Add an alert handler
    pub fn add_handler(&mut self, handler: Box<dyn AlertHandler>);
    
    /// Raise an alert
    pub fn alert(&mut self, alert_type: AlertType, severity: AlertSeverity, timestamp: u64);
    
    /// Raise alert with custom message
    pub fn alert_with_message(
        &mut self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        timestamp: u64,
    );
    
    /// Get recent alerts
    pub fn get_recent_alerts(&self, count: usize) -> Vec<&Alert>;
    
    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<&Alert>;
    
    /// Acknowledge an alert
    pub fn acknowledge(&mut self, alert_id: &str) -> bool;
    
    /// Get unacknowledged alert count
    pub fn unacknowledged_count(&self) -> usize;
    
    /// Clear old alerts
    pub fn cleanup(&mut self, max_age_ms: u64, current_time: u64);
}
```

## Acceptance Criteria

- [ ] `AlertSeverity` enum with ordered levels
- [ ] `AlertType` enum covering common scenarios
- [ ] `Alert` struct with all metadata
- [ ] `AlertHandler` trait for extensibility
- [ ] `LogAlertHandler` default implementation
- [ ] `WebhookAlertHandler` (behind feature flag)
- [ ] `AlertManager` coordinating handlers
- [ ] `alert()` method dispatches to all handlers
- [ ] Alert deduplication (same alert type within window)
- [ ] Alert history with configurable size
- [ ] Acknowledgment tracking
- [ ] Severity filtering per handler
- [ ] Unit tests covering:
  - Alert creation and dispatch
  - Handler filtering by severity
  - Deduplication logic
  - History management
- [ ] Documentation with configuration examples

## Technical Notes

- Generate unique alert ID: `format!("{}-{}", alert_type_name, timestamp)`
- Deduplication key: hash of alert type (not including dynamic values)
- Webhook payload: JSON with alert details
- Consider adding rate limiting per alert type
- Log handler should use appropriate log levels (info!, warn!, error!)

## Labels

`enhancement`, `monitoring`, `priority:low`
