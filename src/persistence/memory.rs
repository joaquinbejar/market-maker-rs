//! In-memory repository implementation.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::persistence::repository::Repository;
use crate::persistence::types::{
    ConfigEntry, DailyPnL, EventLog, EventSeverity, Fill, PositionSnapshot,
};
use crate::types::error::MMResult;

// Type aliases for cleaner code
type FillMap = HashMap<String, Fill>;
type PositionMap = HashMap<String, Vec<PositionSnapshot>>;
type PnLMap = HashMap<String, DailyPnL>;
type ConfigMap = HashMap<String, ConfigEntry>;
type EventList = Vec<EventLog>;

/// In-memory repository implementation for testing.
pub struct InMemoryRepository {
    fills: Arc<RwLock<FillMap>>,
    positions: Arc<RwLock<PositionMap>>,
    daily_pnl: Arc<RwLock<PnLMap>>,
    configs: Arc<RwLock<ConfigMap>>,
    events: Arc<RwLock<EventList>>,
}

impl Default for InMemoryRepository {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryRepository {
    /// Creates a new in-memory repository.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fills: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            daily_pnl: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Repository for InMemoryRepository {
    async fn save_fill(&self, fill: &Fill) -> MMResult<()> {
        let mut fills: tokio::sync::RwLockWriteGuard<'_, FillMap> = self.fills.write().await;
        fills.insert(fill.id.clone(), fill.clone());
        Ok(())
    }

    async fn get_fill(&self, id: &str) -> MMResult<Option<Fill>> {
        let fills: tokio::sync::RwLockReadGuard<'_, FillMap> = self.fills.read().await;
        Ok(fills.get(id).cloned())
    }

    async fn get_fills(&self, start_time: u64, end_time: u64) -> MMResult<Vec<Fill>> {
        let fills: tokio::sync::RwLockReadGuard<'_, FillMap> = self.fills.read().await;
        let result: Vec<Fill> = fills
            .values()
            .filter(|f| f.timestamp >= start_time && f.timestamp <= end_time)
            .cloned()
            .collect();
        Ok(result)
    }

    async fn get_fills_by_symbol(&self, symbol: &str) -> MMResult<Vec<Fill>> {
        let fills: tokio::sync::RwLockReadGuard<'_, FillMap> = self.fills.read().await;
        let result: Vec<Fill> = fills
            .values()
            .filter(|f| f.symbol == symbol)
            .cloned()
            .collect();
        Ok(result)
    }

    async fn delete_fill(&self, id: &str) -> MMResult<bool> {
        let mut fills: tokio::sync::RwLockWriteGuard<'_, FillMap> = self.fills.write().await;
        Ok(fills.remove(id).is_some())
    }

    async fn save_position_snapshot(&self, snapshot: &PositionSnapshot) -> MMResult<()> {
        let mut positions: tokio::sync::RwLockWriteGuard<'_, PositionMap> =
            self.positions.write().await;
        positions
            .entry(snapshot.symbol.clone())
            .or_default()
            .push(snapshot.clone());
        Ok(())
    }

    async fn get_latest_position(&self, symbol: &str) -> MMResult<Option<PositionSnapshot>> {
        let positions: tokio::sync::RwLockReadGuard<'_, PositionMap> = self.positions.read().await;
        Ok(positions
            .get(symbol)
            .and_then(|snapshots| snapshots.last().cloned()))
    }

    async fn get_position_history(
        &self,
        symbol: &str,
        start_time: u64,
        end_time: u64,
    ) -> MMResult<Vec<PositionSnapshot>> {
        let positions: tokio::sync::RwLockReadGuard<'_, PositionMap> = self.positions.read().await;
        let result: Vec<PositionSnapshot> = positions
            .get(symbol)
            .map(|snapshots| {
                snapshots
                    .iter()
                    .filter(|s| s.timestamp >= start_time && s.timestamp <= end_time)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        Ok(result)
    }

    async fn save_daily_pnl(&self, pnl: &DailyPnL) -> MMResult<()> {
        let mut daily_pnl: tokio::sync::RwLockWriteGuard<'_, PnLMap> = self.daily_pnl.write().await;
        let key = format!("{}:{}", pnl.date, pnl.symbol);
        daily_pnl.insert(key, pnl.clone());
        Ok(())
    }

    async fn get_daily_pnl(&self, date: &str, symbol: &str) -> MMResult<Option<DailyPnL>> {
        let daily_pnl: tokio::sync::RwLockReadGuard<'_, PnLMap> = self.daily_pnl.read().await;
        let key = format!("{}:{}", date, symbol);
        Ok(daily_pnl.get(&key).cloned())
    }

    async fn get_pnl_history(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
    ) -> MMResult<Vec<DailyPnL>> {
        let daily_pnl: tokio::sync::RwLockReadGuard<'_, PnLMap> = self.daily_pnl.read().await;
        let result: Vec<DailyPnL> = daily_pnl
            .values()
            .filter(|p| {
                p.symbol == symbol && p.date.as_str() >= start_date && p.date.as_str() <= end_date
            })
            .cloned()
            .collect();
        Ok(result)
    }

    async fn save_config(&self, key: &str, value: &str) -> MMResult<()> {
        let mut configs: tokio::sync::RwLockWriteGuard<'_, ConfigMap> = self.configs.write().await;
        configs.insert(key.to_string(), ConfigEntry::new(key, value));
        Ok(())
    }

    async fn get_config(&self, key: &str) -> MMResult<Option<ConfigEntry>> {
        let configs: tokio::sync::RwLockReadGuard<'_, ConfigMap> = self.configs.read().await;
        Ok(configs.get(key).cloned())
    }

    async fn get_all_configs(&self) -> MMResult<Vec<ConfigEntry>> {
        let configs: tokio::sync::RwLockReadGuard<'_, ConfigMap> = self.configs.read().await;
        Ok(configs.values().cloned().collect())
    }

    async fn delete_config(&self, key: &str) -> MMResult<bool> {
        let mut configs: tokio::sync::RwLockWriteGuard<'_, ConfigMap> = self.configs.write().await;
        Ok(configs.remove(key).is_some())
    }

    async fn save_event(&self, event: &EventLog) -> MMResult<()> {
        let mut events: tokio::sync::RwLockWriteGuard<'_, EventList> = self.events.write().await;
        events.push(event.clone());
        Ok(())
    }

    async fn get_events(&self, start_time: u64, end_time: u64) -> MMResult<Vec<EventLog>> {
        let events: tokio::sync::RwLockReadGuard<'_, EventList> = self.events.read().await;
        let result: Vec<EventLog> = events
            .iter()
            .filter(|e| e.timestamp >= start_time && e.timestamp <= end_time)
            .cloned()
            .collect();
        Ok(result)
    }

    async fn get_events_by_type(&self, event_type: &str) -> MMResult<Vec<EventLog>> {
        let events: tokio::sync::RwLockReadGuard<'_, EventList> = self.events.read().await;
        let result: Vec<EventLog> = events
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect();
        Ok(result)
    }

    async fn get_events_by_severity(&self, min_severity: EventSeverity) -> MMResult<Vec<EventLog>> {
        let events: tokio::sync::RwLockReadGuard<'_, EventList> = self.events.read().await;
        let result: Vec<EventLog> = events
            .iter()
            .filter(|e| e.severity >= min_severity)
            .cloned()
            .collect();
        Ok(result)
    }

    async fn clear_all(&self) -> MMResult<()> {
        {
            let mut fills: tokio::sync::RwLockWriteGuard<'_, FillMap> = self.fills.write().await;
            fills.clear();
        }
        {
            let mut positions: tokio::sync::RwLockWriteGuard<'_, PositionMap> =
                self.positions.write().await;
            positions.clear();
        }
        {
            let mut daily_pnl: tokio::sync::RwLockWriteGuard<'_, PnLMap> =
                self.daily_pnl.write().await;
            daily_pnl.clear();
        }
        {
            let mut configs: tokio::sync::RwLockWriteGuard<'_, ConfigMap> =
                self.configs.write().await;
            configs.clear();
        }
        {
            let mut events: tokio::sync::RwLockWriteGuard<'_, EventList> =
                self.events.write().await;
            events.clear();
        }
        Ok(())
    }

    async fn fill_count(&self) -> MMResult<usize> {
        let fills: tokio::sync::RwLockReadGuard<'_, FillMap> = self.fills.read().await;
        Ok(fills.len())
    }

    async fn event_count(&self) -> MMResult<usize> {
        let events: tokio::sync::RwLockReadGuard<'_, EventList> = self.events.read().await;
        Ok(events.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::types::FillSide;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_save_and_get_fill() {
        let repo = InMemoryRepository::new();
        let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
        let fill_id = fill.id.clone();

        repo.save_fill(&fill).await.unwrap();

        let retrieved = repo.get_fill(&fill_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().symbol, "BTC");
    }

    #[tokio::test]
    async fn test_get_fills_by_symbol() {
        let repo = InMemoryRepository::new();

        let fill1 = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
        let fill2 = Fill::new("ETH", dec!(3000.0), dec!(10.0), FillSide::Buy, "order-2");
        let fill3 = Fill::new("BTC", dec!(51000.0), dec!(0.5), FillSide::Sell, "order-3");

        repo.save_fill(&fill1).await.unwrap();
        repo.save_fill(&fill2).await.unwrap();
        repo.save_fill(&fill3).await.unwrap();

        let btc_fills = repo.get_fills_by_symbol("BTC").await.unwrap();
        assert_eq!(btc_fills.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_fill() {
        let repo = InMemoryRepository::new();
        let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
        let fill_id = fill.id.clone();

        repo.save_fill(&fill).await.unwrap();
        assert_eq!(repo.fill_count().await.unwrap(), 1);

        let deleted = repo.delete_fill(&fill_id).await.unwrap();
        assert!(deleted);
        assert_eq!(repo.fill_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_position_snapshots() {
        let repo = InMemoryRepository::new();

        let snapshot1 = PositionSnapshot::new("BTC", dec!(10.0), dec!(48000.0), dec!(50000.0));
        let snapshot2 = PositionSnapshot::new("BTC", dec!(15.0), dec!(49000.0), dec!(51000.0));

        repo.save_position_snapshot(&snapshot1).await.unwrap();
        repo.save_position_snapshot(&snapshot2).await.unwrap();

        let latest = repo.get_latest_position("BTC").await.unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().quantity, dec!(15.0));
    }

    #[tokio::test]
    async fn test_daily_pnl() {
        let repo = InMemoryRepository::new();

        let pnl = DailyPnL::new("2024-01-01", "BTC", dec!(1000.0), dec!(500.0));
        repo.save_daily_pnl(&pnl).await.unwrap();

        let retrieved = repo.get_daily_pnl("2024-01-01", "BTC").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().total_pnl, dec!(1500.0));
    }

    #[tokio::test]
    async fn test_config() {
        let repo = InMemoryRepository::new();

        repo.save_config("max_delta", "100.0").await.unwrap();
        repo.save_config("max_gamma", "50.0").await.unwrap();

        let config = repo.get_config("max_delta").await.unwrap();
        assert!(config.is_some());
        assert_eq!(config.unwrap().value, "100.0");

        let all_configs = repo.get_all_configs().await.unwrap();
        assert_eq!(all_configs.len(), 2);
    }

    #[tokio::test]
    async fn test_events() {
        let repo = InMemoryRepository::new();

        let event1 = EventLog::info("TRADE", "Trade executed");
        let event2 = EventLog::warning("RISK", "Delta limit approaching");
        let event3 = EventLog::error("SYSTEM", "Connection lost");

        repo.save_event(&event1).await.unwrap();
        repo.save_event(&event2).await.unwrap();
        repo.save_event(&event3).await.unwrap();

        assert_eq!(repo.event_count().await.unwrap(), 3);

        let risk_events = repo.get_events_by_type("RISK").await.unwrap();
        assert_eq!(risk_events.len(), 1);

        let warnings_and_above = repo
            .get_events_by_severity(EventSeverity::Warning)
            .await
            .unwrap();
        assert_eq!(warnings_and_above.len(), 2);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let repo = InMemoryRepository::new();

        let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
        let event = EventLog::info("TEST", "Test event");

        repo.save_fill(&fill).await.unwrap();
        repo.save_event(&event).await.unwrap();

        assert_eq!(repo.fill_count().await.unwrap(), 1);
        assert_eq!(repo.event_count().await.unwrap(), 1);

        repo.clear_all().await.unwrap();

        assert_eq!(repo.fill_count().await.unwrap(), 0);
        assert_eq!(repo.event_count().await.unwrap(), 0);
    }
}
