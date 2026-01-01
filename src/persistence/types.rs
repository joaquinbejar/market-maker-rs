//! Persistence types and data structures.

use crate::Decimal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trade fill record.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fill {
    /// Unique fill identifier.
    pub id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Fill price.
    pub price: Decimal,
    /// Fill quantity.
    pub quantity: Decimal,
    /// Fill side.
    pub side: FillSide,
    /// Associated order ID.
    pub order_id: String,
    /// Fill timestamp in milliseconds.
    pub timestamp: u64,
    /// Fee amount.
    pub fee: Decimal,
    /// Fee currency.
    pub fee_currency: String,
}

impl Fill {
    /// Creates a new fill record.
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        price: Decimal,
        quantity: Decimal,
        side: FillSide,
        order_id: impl Into<String>,
    ) -> Self {
        Self {
            id: generate_id(),
            symbol: symbol.into(),
            price,
            quantity,
            side,
            order_id: order_id.into(),
            timestamp: current_timestamp(),
            fee: Decimal::ZERO,
            fee_currency: "USD".to_string(),
        }
    }

    /// Sets the fill ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets the fee.
    #[must_use]
    pub fn with_fee(mut self, fee: Decimal, currency: impl Into<String>) -> Self {
        self.fee = fee;
        self.fee_currency = currency.into();
        self
    }

    /// Sets the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Returns the notional value.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }

    /// Returns the net value after fees.
    #[must_use]
    pub fn net_value(&self) -> Decimal {
        match self.side {
            FillSide::Buy => self.notional() + self.fee,
            FillSide::Sell => self.notional() - self.fee,
        }
    }
}

/// Fill side indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FillSide {
    /// Buy side.
    Buy,
    /// Sell side.
    Sell,
}

/// Position snapshot record.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PositionSnapshot {
    /// Unique snapshot identifier.
    pub id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Position quantity.
    pub quantity: Decimal,
    /// Average entry price.
    pub avg_price: Decimal,
    /// Market price at snapshot time.
    pub market_price: Decimal,
    /// Unrealized P&L.
    pub unrealized_pnl: Decimal,
    /// Realized P&L.
    pub realized_pnl: Decimal,
    /// Snapshot timestamp in milliseconds.
    pub timestamp: u64,
}

impl PositionSnapshot {
    /// Creates a new position snapshot.
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        quantity: Decimal,
        avg_price: Decimal,
        market_price: Decimal,
    ) -> Self {
        let unrealized_pnl = (market_price - avg_price) * quantity;
        Self {
            id: generate_id(),
            symbol: symbol.into(),
            quantity,
            avg_price,
            market_price,
            unrealized_pnl,
            realized_pnl: Decimal::ZERO,
            timestamp: current_timestamp(),
        }
    }

    /// Sets the realized P&L.
    #[must_use]
    pub fn with_realized_pnl(mut self, pnl: Decimal) -> Self {
        self.realized_pnl = pnl;
        self
    }

    /// Sets the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Returns the total P&L.
    #[must_use]
    pub fn total_pnl(&self) -> Decimal {
        self.unrealized_pnl + self.realized_pnl
    }
}

/// Daily P&L record.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DailyPnL {
    /// Unique record identifier.
    pub id: String,
    /// Date in YYYY-MM-DD format.
    pub date: String,
    /// Trading symbol.
    pub symbol: String,
    /// Realized P&L for the day.
    pub realized_pnl: Decimal,
    /// Unrealized P&L at end of day.
    pub unrealized_pnl: Decimal,
    /// Total P&L.
    pub total_pnl: Decimal,
    /// Number of trades.
    pub trade_count: u32,
    /// Total volume traded.
    pub volume: Decimal,
    /// Record timestamp in milliseconds.
    pub timestamp: u64,
}

impl DailyPnL {
    /// Creates a new daily P&L record.
    #[must_use]
    pub fn new(
        date: impl Into<String>,
        symbol: impl Into<String>,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
    ) -> Self {
        Self {
            id: generate_id(),
            date: date.into(),
            symbol: symbol.into(),
            realized_pnl,
            unrealized_pnl,
            total_pnl: realized_pnl + unrealized_pnl,
            trade_count: 0,
            volume: Decimal::ZERO,
            timestamp: current_timestamp(),
        }
    }

    /// Sets the trade count and volume.
    #[must_use]
    pub fn with_trading_stats(mut self, trade_count: u32, volume: Decimal) -> Self {
        self.trade_count = trade_count;
        self.volume = volume;
        self
    }
}

/// Configuration entry.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfigEntry {
    /// Configuration key.
    pub key: String,
    /// Configuration value as JSON string.
    pub value: String,
    /// Last updated timestamp in milliseconds.
    pub updated_at: u64,
}

impl ConfigEntry {
    /// Creates a new configuration entry.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            updated_at: current_timestamp(),
        }
    }
}

/// Event log entry.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EventLog {
    /// Unique event identifier.
    pub id: String,
    /// Event type/category.
    pub event_type: String,
    /// Event severity.
    pub severity: EventSeverity,
    /// Event message.
    pub message: String,
    /// Additional data as JSON string.
    pub data: Option<String>,
    /// Event timestamp in milliseconds.
    pub timestamp: u64,
}

impl EventLog {
    /// Creates a new event log entry.
    #[must_use]
    pub fn new(
        event_type: impl Into<String>,
        severity: EventSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: generate_id(),
            event_type: event_type.into(),
            severity,
            message: message.into(),
            data: None,
            timestamp: current_timestamp(),
        }
    }

    /// Sets additional data.
    #[must_use]
    pub fn with_data(mut self, data: impl Into<String>) -> Self {
        self.data = Some(data.into());
        self
    }

    /// Creates an info event.
    #[must_use]
    pub fn info(event_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(event_type, EventSeverity::Info, message)
    }

    /// Creates a warning event.
    #[must_use]
    pub fn warning(event_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(event_type, EventSeverity::Warning, message)
    }

    /// Creates an error event.
    #[must_use]
    pub fn error(event_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(event_type, EventSeverity::Error, message)
    }

    /// Creates a critical event.
    #[must_use]
    pub fn critical(event_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(event_type, EventSeverity::Critical, message)
    }
}

/// Event severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EventSeverity {
    /// Informational event.
    Info,
    /// Warning event.
    Warning,
    /// Error event.
    Error,
    /// Critical event.
    Critical,
}

impl std::fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSeverity::Info => write!(f, "INFO"),
            EventSeverity::Warning => write!(f, "WARNING"),
            EventSeverity::Error => write!(f, "ERROR"),
            EventSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Generates a unique identifier.
fn generate_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = current_timestamp();
    format!("{}-{}", timestamp, count)
}

/// Returns current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_fill_new() {
        let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
        assert_eq!(fill.symbol, "BTC");
        assert_eq!(fill.price, dec!(50000.0));
        assert_eq!(fill.quantity, dec!(1.0));
        assert_eq!(fill.notional(), dec!(50000.0));
    }

    #[test]
    fn test_fill_with_fee() {
        let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1")
            .with_fee(dec!(50.0), "USD");
        assert_eq!(fill.fee, dec!(50.0));
        assert_eq!(fill.net_value(), dec!(50050.0)); // Buy: notional + fee
    }

    #[test]
    fn test_position_snapshot() {
        let snapshot = PositionSnapshot::new("BTC", dec!(10.0), dec!(48000.0), dec!(50000.0));
        assert_eq!(snapshot.unrealized_pnl, dec!(20000.0)); // (50000 - 48000) * 10
        assert_eq!(snapshot.total_pnl(), dec!(20000.0));
    }

    #[test]
    fn test_daily_pnl() {
        let pnl = DailyPnL::new("2024-01-01", "BTC", dec!(1000.0), dec!(500.0))
            .with_trading_stats(50, dec!(100.0));
        assert_eq!(pnl.total_pnl, dec!(1500.0));
        assert_eq!(pnl.trade_count, 50);
    }

    #[test]
    fn test_event_log() {
        let event = EventLog::warning("RISK", "Delta limit approaching");
        assert_eq!(event.severity, EventSeverity::Warning);
        assert_eq!(event.event_type, "RISK");
    }

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Info < EventSeverity::Warning);
        assert!(EventSeverity::Warning < EventSeverity::Error);
        assert!(EventSeverity::Error < EventSeverity::Critical);
    }
}
