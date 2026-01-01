//! Repository trait for data persistence.

use async_trait::async_trait;

use crate::persistence::types::{ConfigEntry, DailyPnL, EventLog, Fill, PositionSnapshot};
use crate::types::error::MMResult;

/// Abstract repository trait for data persistence.
///
/// This trait defines the interface for persisting and retrieving market maker data.
/// Implementations can use different backends (in-memory, PostgreSQL, etc.).
#[async_trait]
pub trait Repository: Send + Sync {
    // Fill operations

    /// Saves a fill record.
    async fn save_fill(&self, fill: &Fill) -> MMResult<()>;

    /// Gets a fill by ID.
    async fn get_fill(&self, id: &str) -> MMResult<Option<Fill>>;

    /// Gets fills within a time range.
    async fn get_fills(&self, start_time: u64, end_time: u64) -> MMResult<Vec<Fill>>;

    /// Gets fills for a specific symbol.
    async fn get_fills_by_symbol(&self, symbol: &str) -> MMResult<Vec<Fill>>;

    /// Deletes a fill by ID.
    async fn delete_fill(&self, id: &str) -> MMResult<bool>;

    // Position snapshot operations

    /// Saves a position snapshot.
    async fn save_position_snapshot(&self, snapshot: &PositionSnapshot) -> MMResult<()>;

    /// Gets the latest position snapshot for a symbol.
    async fn get_latest_position(&self, symbol: &str) -> MMResult<Option<PositionSnapshot>>;

    /// Gets position snapshots within a time range.
    async fn get_position_history(
        &self,
        symbol: &str,
        start_time: u64,
        end_time: u64,
    ) -> MMResult<Vec<PositionSnapshot>>;

    // Daily P&L operations

    /// Saves a daily P&L record.
    async fn save_daily_pnl(&self, pnl: &DailyPnL) -> MMResult<()>;

    /// Gets daily P&L for a specific date.
    async fn get_daily_pnl(&self, date: &str, symbol: &str) -> MMResult<Option<DailyPnL>>;

    /// Gets daily P&L records within a date range.
    async fn get_pnl_history(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
    ) -> MMResult<Vec<DailyPnL>>;

    // Configuration operations

    /// Saves a configuration entry.
    async fn save_config(&self, key: &str, value: &str) -> MMResult<()>;

    /// Gets a configuration entry.
    async fn get_config(&self, key: &str) -> MMResult<Option<ConfigEntry>>;

    /// Gets all configuration entries.
    async fn get_all_configs(&self) -> MMResult<Vec<ConfigEntry>>;

    /// Deletes a configuration entry.
    async fn delete_config(&self, key: &str) -> MMResult<bool>;

    // Event log operations

    /// Saves an event log entry.
    async fn save_event(&self, event: &EventLog) -> MMResult<()>;

    /// Gets events within a time range.
    async fn get_events(&self, start_time: u64, end_time: u64) -> MMResult<Vec<EventLog>>;

    /// Gets events by type.
    async fn get_events_by_type(&self, event_type: &str) -> MMResult<Vec<EventLog>>;

    /// Gets events by severity (and above).
    async fn get_events_by_severity(
        &self,
        min_severity: crate::persistence::types::EventSeverity,
    ) -> MMResult<Vec<EventLog>>;

    // Utility operations

    /// Clears all data (for testing).
    async fn clear_all(&self) -> MMResult<()>;

    /// Returns the total number of fills.
    async fn fill_count(&self) -> MMResult<usize>;

    /// Returns the total number of events.
    async fn event_count(&self) -> MMResult<usize>;
}
