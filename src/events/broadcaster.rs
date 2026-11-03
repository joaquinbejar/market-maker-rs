//! Event broadcaster for distributing events to subscribers.
//!
//! This module provides the core broadcasting infrastructure including:
//! - Broadcast channel for event distribution
//! - Event history for reconnection
//! - Filtered subscriptions
//! - Event batching/aggregation

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, broadcast};
use tokio::task::JoinHandle;

use super::types::{EventFilter, MarketMakerEvent};

/// Event broadcaster configuration.
#[derive(Debug, Clone)]
pub struct EventBroadcasterConfig {
    /// Channel capacity for broadcast channel.
    pub channel_capacity: usize,
    /// Maximum number of events to keep in history.
    pub max_history_size: usize,
    /// History retention duration in seconds.
    pub history_retention_secs: u64,
    /// Enable event batching.
    pub enable_batching: bool,
    /// Batch interval in milliseconds.
    pub batch_interval_ms: u64,
    /// Heartbeat interval in seconds.
    pub heartbeat_interval_secs: u64,
}

impl Default for EventBroadcasterConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1024,
            max_history_size: 10000,
            history_retention_secs: 3600,
            enable_batching: false,
            batch_interval_ms: 100,
            heartbeat_interval_secs: 30,
        }
    }
}

impl EventBroadcasterConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the channel capacity.
    #[must_use]
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Sets the maximum history size.
    #[must_use]
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Sets the history retention duration in seconds.
    #[must_use]
    pub fn with_history_retention_secs(mut self, secs: u64) -> Self {
        self.history_retention_secs = secs;
        self
    }

    /// Enables or disables event batching.
    #[must_use]
    pub fn with_batching(mut self, enable: bool) -> Self {
        self.enable_batching = enable;
        self
    }

    /// Sets the batch interval in milliseconds.
    #[must_use]
    pub fn with_batch_interval_ms(mut self, ms: u64) -> Self {
        self.batch_interval_ms = ms;
        self
    }

    /// Sets the heartbeat interval in seconds.
    #[must_use]
    pub fn with_heartbeat_interval_secs(mut self, secs: u64) -> Self {
        self.heartbeat_interval_secs = secs;
        self
    }
}

/// Timestamped event for history tracking.
#[derive(Debug, Clone)]
pub struct TimestampedEvent {
    /// The event.
    pub event: MarketMakerEvent,
    /// Sequence number for ordering.
    pub sequence: u64,
    /// Time when event was received (milliseconds since epoch).
    pub received_at: u64,
}

/// Event history buffer for reconnection support.
pub struct EventHistory {
    events: VecDeque<TimestampedEvent>,
    max_size: usize,
    retention_secs: u64,
}

impl EventHistory {
    /// Creates a new event history buffer.
    #[must_use]
    pub fn new(max_size: usize, retention_secs: u64) -> Self {
        Self {
            events: VecDeque::with_capacity(max_size.min(1000)),
            max_size,
            retention_secs,
        }
    }

    /// Adds an event to the history.
    pub fn push(&mut self, event: MarketMakerEvent, sequence: u64) {
        let received_at = current_timestamp_ms();
        self.events.push_back(TimestampedEvent {
            event,
            sequence,
            received_at,
        });

        // Prune if over capacity
        while self.events.len() > self.max_size {
            self.events.pop_front();
        }
    }

    /// Gets events since a given timestamp.
    #[must_use]
    pub fn since(&self, timestamp: u64) -> Vec<MarketMakerEvent> {
        self.events
            .iter()
            .filter(|e| e.received_at >= timestamp)
            .map(|e| e.event.clone())
            .collect()
    }

    /// Gets events since a given sequence number.
    #[must_use]
    pub fn since_sequence(&self, sequence: u64) -> Vec<MarketMakerEvent> {
        self.events
            .iter()
            .filter(|e| e.sequence > sequence)
            .map(|e| e.event.clone())
            .collect()
    }

    /// Prunes old events based on retention time.
    pub fn prune(&mut self) {
        let retention_ms = self.retention_secs.saturating_mul(1000);
        let cutoff = current_timestamp_ms().saturating_sub(retention_ms);
        while let Some(front) = self.events.front() {
            if front.received_at < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Returns the number of events in history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the latest sequence number, or 0 if empty.
    #[must_use]
    pub fn latest_sequence(&self) -> u64 {
        self.events.back().map(|e| e.sequence).unwrap_or(0)
    }
}

/// Event broadcaster for distributing events to subscribers.
pub struct EventBroadcaster {
    sender: broadcast::Sender<MarketMakerEvent>,
    history: RwLock<EventHistory>,
    sequence: AtomicU64,
    config: EventBroadcasterConfig,
}

impl EventBroadcaster {
    /// Creates a new event broadcaster with the given configuration.
    #[must_use]
    pub fn new(config: EventBroadcasterConfig) -> Self {
        let (sender, _) = broadcast::channel(config.channel_capacity);
        let history = EventHistory::new(config.max_history_size, config.history_retention_secs);

        Self {
            sender,
            history: RwLock::new(history),
            sequence: AtomicU64::new(0),
            config,
        }
    }

    /// Subscribes to all events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<MarketMakerEvent> {
        self.sender.subscribe()
    }

    /// Subscribes with a filter.
    #[must_use]
    pub fn subscribe_filtered(&self, filter: EventFilter) -> FilteredEventReceiver {
        FilteredEventReceiver {
            receiver: self.sender.subscribe(),
            filter,
        }
    }

    /// Broadcasts an event to all subscribers.
    ///
    /// Returns the number of subscribers that received the event.
    pub async fn broadcast(&self, event: MarketMakerEvent) -> usize {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push(event.clone(), sequence);
        }

        // Send to subscribers (ignore errors if no subscribers)
        self.sender.send(event).unwrap_or(0)
    }

    /// Gets event history since a given timestamp.
    pub async fn get_history(&self, since: Option<u64>) -> Vec<MarketMakerEvent> {
        let history = self.history.read().await;
        match since {
            Some(ts) => history.since(ts),
            None => history.events.iter().map(|e| e.event.clone()).collect(),
        }
    }

    /// Gets event history for reconnection (events after a sequence number).
    pub async fn get_reconnection_history(&self, last_sequence: u64) -> Vec<MarketMakerEvent> {
        let history = self.history.read().await;
        history.since_sequence(last_sequence)
    }

    /// Returns the current number of subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Returns the current sequence number.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Prunes old events from history.
    pub async fn prune_history(&self) {
        let mut history = self.history.write().await;
        history.prune();
    }

    /// Returns the broadcaster configuration.
    #[must_use]
    pub fn config(&self) -> &EventBroadcasterConfig {
        &self.config
    }

    /// Starts a heartbeat task that sends periodic heartbeat events.
    ///
    /// Returns a join handle for the spawned task.
    pub fn start_heartbeat(self: &Arc<Self>) -> JoinHandle<()> {
        let broadcaster = Arc::clone(self);
        let interval_secs = self.config.heartbeat_interval_secs;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
            loop {
                interval.tick().await;
                let sequence = broadcaster.current_sequence();
                let event = MarketMakerEvent::Heartbeat {
                    timestamp: current_timestamp_ms(),
                    sequence,
                };
                broadcaster.broadcast(event).await;
            }
        })
    }

    /// Starts a history pruning task that periodically removes old events.
    ///
    /// Returns a join handle for the spawned task.
    pub fn start_history_pruning(self: &Arc<Self>) -> JoinHandle<()> {
        let broadcaster = Arc::clone(self);
        // Prune every 1/10th of retention time, minimum 60 seconds
        let prune_interval = (self.config.history_retention_secs / 10).max(60);

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(prune_interval));
            loop {
                interval.tick().await;
                broadcaster.prune_history().await;
            }
        })
    }
}

/// Filtered event receiver that only yields events matching the filter.
pub struct FilteredEventReceiver {
    receiver: broadcast::Receiver<MarketMakerEvent>,
    filter: EventFilter,
}

impl FilteredEventReceiver {
    /// Receives the next event that matches the filter.
    ///
    /// Returns an error if the channel is closed or lagged.
    pub async fn recv(&mut self) -> Result<MarketMakerEvent, broadcast::error::RecvError> {
        loop {
            let event = self.receiver.recv().await?;
            if self.filter.matches(&event) {
                return Ok(event);
            }
        }
    }

    /// Returns a reference to the filter.
    #[must_use]
    pub fn filter(&self) -> &EventFilter {
        &self.filter
    }

    /// Updates the filter.
    pub fn set_filter(&mut self, filter: EventFilter) {
        self.filter = filter;
    }
}

/// Event aggregator for batching high-frequency events.
pub struct EventAggregator {
    pending: RwLock<Vec<MarketMakerEvent>>,
    interval_ms: u64,
}

impl EventAggregator {
    /// Creates a new event aggregator with the given batch interval.
    #[must_use]
    pub fn new(interval_ms: u64) -> Self {
        Self {
            pending: RwLock::new(Vec::new()),
            interval_ms,
        }
    }

    /// Adds an event to the pending batch.
    pub async fn add(&self, event: MarketMakerEvent) {
        let mut pending = self.pending.write().await;
        pending.push(event);
    }

    /// Flushes and returns all pending events.
    pub async fn flush(&self) -> Vec<MarketMakerEvent> {
        let mut pending = self.pending.write().await;
        std::mem::take(&mut *pending)
    }

    /// Returns the number of pending events.
    pub async fn pending_count(&self) -> usize {
        let pending = self.pending.read().await;
        pending.len()
    }

    /// Returns the batch interval in milliseconds.
    #[must_use]
    pub fn interval_ms(&self) -> u64 {
        self.interval_ms
    }

    /// Starts automatic flushing to a broadcaster.
    ///
    /// Returns a join handle for the spawned task.
    pub fn start_auto_flush(self: Arc<Self>, broadcaster: Arc<EventBroadcaster>) -> JoinHandle<()> {
        let interval_ms = self.interval_ms;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));
            loop {
                interval.tick().await;
                let events = self.flush().await;
                for event in events {
                    broadcaster.broadcast(event).await;
                }
            }
        })
    }
}

/// Returns the current timestamp in milliseconds since epoch.
#[must_use]
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::types::{Action, EventType};

    #[test]
    fn test_config_builder() {
        let config = EventBroadcasterConfig::new()
            .with_channel_capacity(2048)
            .with_max_history_size(5000)
            .with_history_retention_secs(7200)
            .with_batching(true)
            .with_batch_interval_ms(50)
            .with_heartbeat_interval_secs(15);

        assert_eq!(config.channel_capacity, 2048);
        assert_eq!(config.max_history_size, 5000);
        assert_eq!(config.history_retention_secs, 7200);
        assert!(config.enable_batching);
        assert_eq!(config.batch_interval_ms, 50);
        assert_eq!(config.heartbeat_interval_secs, 15);
    }

    #[test]
    fn test_event_history() {
        let mut history = EventHistory::new(5, 3600);

        // Add events
        for i in 0..3 {
            let event = MarketMakerEvent::Heartbeat {
                timestamp: 1000 + i,
                sequence: i,
            };
            history.push(event, i);
        }

        assert_eq!(history.len(), 3);
        assert!(!history.is_empty());
        assert_eq!(history.latest_sequence(), 2);

        // Test since_sequence
        let events = history.since_sequence(0);
        assert_eq!(events.len(), 2); // sequences 1 and 2
    }

    #[test]
    fn test_event_history_capacity() {
        let mut history = EventHistory::new(3, 3600);

        // Add more events than capacity
        for i in 0..5 {
            let event = MarketMakerEvent::Heartbeat {
                timestamp: 1000 + i,
                sequence: i,
            };
            history.push(event, i);
        }

        // Should only keep last 3
        assert_eq!(history.len(), 3);
        assert_eq!(history.latest_sequence(), 4);

        // First event should be sequence 2
        let events = history.since_sequence(0);
        assert_eq!(events.len(), 3);
    }

    #[tokio::test]
    async fn test_broadcaster_subscribe() {
        let config = EventBroadcasterConfig::default();
        let broadcaster = EventBroadcaster::new(config);

        let mut rx = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 1);

        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 0,
        };
        broadcaster.broadcast(event.clone()).await;

        let received = rx.recv().await.unwrap();
        assert_eq!(received, event);
    }

    #[tokio::test]
    async fn test_broadcaster_multiple_subscribers() {
        let config = EventBroadcasterConfig::default();
        let broadcaster = EventBroadcaster::new(config);

        let mut rx1 = broadcaster.subscribe();
        let mut rx2 = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 2);

        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 0,
        };
        let count = broadcaster.broadcast(event.clone()).await;
        assert_eq!(count, 2);

        let received1 = rx1.recv().await.unwrap();
        let received2 = rx2.recv().await.unwrap();
        assert_eq!(received1, event);
        assert_eq!(received2, event);
    }

    #[tokio::test]
    async fn test_broadcaster_filtered_subscription() {
        let config = EventBroadcasterConfig::default();
        let broadcaster = EventBroadcaster::new(config);

        let filter = EventFilter::new().with_event_types([EventType::OrderFilled]);
        let mut filtered_rx = broadcaster.subscribe_filtered(filter);

        // Broadcast a heartbeat (should be filtered out)
        let heartbeat = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 0,
        };
        broadcaster.broadcast(heartbeat).await;

        // Broadcast an order fill (should pass filter)
        let fill = MarketMakerEvent::OrderFilled {
            order_id: "123".to_string(),
            symbol: "BTC".to_string(),
            instrument: "BTC-PERP".to_string(),
            side: Action::Buy,
            quantity: 10,
            price: 50000,
            fee: 5,
            edge: 100,
            timestamp: 1001,
        };
        broadcaster.broadcast(fill.clone()).await;

        // Should receive the fill, not the heartbeat
        let received = filtered_rx.recv().await.unwrap();
        assert_eq!(received, fill);
    }

    #[tokio::test]
    async fn test_broadcaster_history() {
        let config = EventBroadcasterConfig::new().with_max_history_size(10);
        let broadcaster = EventBroadcaster::new(config);

        // Broadcast some events
        for i in 0..5 {
            let event = MarketMakerEvent::Heartbeat {
                timestamp: 1000 + i,
                sequence: i,
            };
            broadcaster.broadcast(event).await;
        }

        // Get all history
        let history = broadcaster.get_history(None).await;
        assert_eq!(history.len(), 5);

        // Get reconnection history
        let reconnection = broadcaster.get_reconnection_history(2).await;
        assert_eq!(reconnection.len(), 2); // sequences 3 and 4
    }

    #[tokio::test]
    async fn test_broadcaster_sequence() {
        let config = EventBroadcasterConfig::default();
        let broadcaster = EventBroadcaster::new(config);

        assert_eq!(broadcaster.current_sequence(), 0);

        broadcaster
            .broadcast(MarketMakerEvent::Heartbeat {
                timestamp: 1000,
                sequence: 0,
            })
            .await;
        assert_eq!(broadcaster.current_sequence(), 1);

        broadcaster
            .broadcast(MarketMakerEvent::Heartbeat {
                timestamp: 1001,
                sequence: 1,
            })
            .await;
        assert_eq!(broadcaster.current_sequence(), 2);
    }

    #[tokio::test]
    async fn test_event_aggregator() {
        let aggregator = EventAggregator::new(100);

        // Add events
        for i in 0..3 {
            let event = MarketMakerEvent::Heartbeat {
                timestamp: 1000 + i,
                sequence: i,
            };
            aggregator.add(event).await;
        }

        assert_eq!(aggregator.pending_count().await, 3);

        // Flush
        let events = aggregator.flush().await;
        assert_eq!(events.len(), 3);
        assert_eq!(aggregator.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_filtered_receiver_update_filter() {
        let config = EventBroadcasterConfig::default();
        let broadcaster = EventBroadcaster::new(config);

        let filter = EventFilter::new().exclude_heartbeats(true);
        let mut filtered_rx = broadcaster.subscribe_filtered(filter);

        // Update filter to allow heartbeats
        filtered_rx.set_filter(EventFilter::new().exclude_heartbeats(false));

        let heartbeat = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 0,
        };
        broadcaster.broadcast(heartbeat.clone()).await;

        let received = filtered_rx.recv().await.unwrap();
        assert_eq!(received, heartbeat);
    }

    #[test]
    fn test_timestamped_event() {
        let event = MarketMakerEvent::Heartbeat {
            timestamp: 1000,
            sequence: 42,
        };
        let timestamped = TimestampedEvent {
            event: event.clone(),
            sequence: 42,
            received_at: 1000,
        };

        assert_eq!(timestamped.sequence, 42);
        assert_eq!(timestamped.received_at, 1000);
        assert_eq!(timestamped.event, event);
    }
}
