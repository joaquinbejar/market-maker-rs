//! Live metrics tracking for real-time trading system monitoring.
//!
//! This module provides thread-safe counters and gauges for tracking
//! operational metrics during live trading.
//!
//! # Overview
//!
//! The live metrics module includes:
//!
//! - **Counter**: Atomic counter for monotonically increasing values
//! - **Gauge**: Atomic gauge for values that go up and down
//! - **LiveMetrics**: Main tracker with all operational metrics
//! - **MetricsSnapshot**: Point-in-time snapshot of all metrics
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::{LiveMetrics, SharedLiveMetrics};
//! use market_maker_rs::dec;
//! use std::sync::Arc;
//!
//! // Create shared metrics tracker
//! let metrics: SharedLiveMetrics = Arc::new(LiveMetrics::new(1000));
//!
//! // Record trading activity
//! metrics.record_quote(1001);
//! metrics.record_order_submitted();
//! metrics.record_order_filled(1002);
//!
//! // Update position and PnL
//! metrics.update_position(dec!(10.0));
//! metrics.update_pnl(dec!(100.0), dec!(50.0));
//!
//! // Get snapshot
//! let snapshot = metrics.snapshot(1100);
//! println!("Quotes: {}, Fills: {}", snapshot.quotes_generated, snapshot.orders_filled);
//! ```

use crate::Decimal;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Atomic counter for thread-safe incrementing.
///
/// Counters are monotonically increasing values used for tracking
/// events like quotes generated, orders submitted, etc.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::Counter;
///
/// let counter = Counter::new();
/// counter.increment();
/// counter.add(5);
/// assert_eq!(counter.get(), 6);
///
/// counter.reset();
/// assert_eq!(counter.get(), 0);
/// ```
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// Creates a new counter initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Creates a new counter with an initial value.
    #[must_use]
    pub fn with_value(value: u64) -> Self {
        Self(AtomicU64::new(value))
    }

    /// Increments the counter by one.
    pub fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds a value to the counter.
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Returns the current counter value.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Resets the counter to zero.
    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

/// Atomic gauge for values that can increase or decrease.
///
/// Gauges are used for tracking values like open orders count,
/// which can go up and down during trading.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::Gauge;
///
/// let gauge = Gauge::new();
/// gauge.set(10);
/// assert_eq!(gauge.get(), 10);
///
/// gauge.increment();
/// assert_eq!(gauge.get(), 11);
///
/// gauge.decrement();
/// assert_eq!(gauge.get(), 10);
/// ```
#[derive(Debug, Default)]
pub struct Gauge(AtomicI64);

impl Gauge {
    /// Creates a new gauge initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        Self(AtomicI64::new(0))
    }

    /// Creates a new gauge with an initial value.
    #[must_use]
    pub fn with_value(value: i64) -> Self {
        Self(AtomicI64::new(value))
    }

    /// Sets the gauge to a specific value.
    pub fn set(&self, value: i64) {
        self.0.store(value, Ordering::Relaxed);
    }

    /// Returns the current gauge value.
    #[must_use]
    pub fn get(&self) -> i64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Increments the gauge by one.
    pub fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the gauge by one.
    pub fn decrement(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }

    /// Adds a value to the gauge.
    pub fn add(&self, n: i64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Subtracts a value from the gauge.
    pub fn sub(&self, n: i64) {
        self.0.fetch_sub(n, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of all live metrics.
///
/// This struct captures the current state of all metrics at a specific
/// timestamp, including calculated rates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricsSnapshot {
    /// Snapshot timestamp in milliseconds.
    pub timestamp: u64,

    // Counters
    /// Total quotes generated since start.
    pub quotes_generated: u64,
    /// Total orders submitted since start.
    pub orders_submitted: u64,
    /// Total orders filled since start.
    pub orders_filled: u64,
    /// Total orders cancelled since start.
    pub orders_cancelled: u64,
    /// Total orders rejected since start.
    pub orders_rejected: u64,
    /// Total partial fills since start.
    pub partial_fills: u64,

    // Gauges
    /// Current number of open orders.
    pub open_orders: i64,
    /// Current position size.
    pub current_position: Decimal,
    /// Current total PnL (realized + unrealized).
    pub current_pnl: Decimal,
    /// Unrealized PnL from open positions.
    pub unrealized_pnl: Decimal,
    /// Realized PnL from closed positions.
    pub realized_pnl: Decimal,

    // Rates (calculated)
    /// Fill rate: fills / quotes.
    pub fill_rate: Decimal,
    /// Cancel rate: cancels / orders submitted.
    pub cancel_rate: Decimal,
    /// Quotes generated per second.
    pub quotes_per_second: Decimal,
    /// Orders filled per second.
    pub fills_per_second: Decimal,

    // Timing
    /// Uptime in milliseconds since start.
    pub uptime_ms: u64,
    /// Timestamp of last quote generated.
    pub last_quote_time: u64,
    /// Timestamp of last fill received.
    pub last_fill_time: u64,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            timestamp: 0,
            quotes_generated: 0,
            orders_submitted: 0,
            orders_filled: 0,
            orders_cancelled: 0,
            orders_rejected: 0,
            partial_fills: 0,
            open_orders: 0,
            current_position: Decimal::ZERO,
            current_pnl: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            fill_rate: Decimal::ZERO,
            cancel_rate: Decimal::ZERO,
            quotes_per_second: Decimal::ZERO,
            fills_per_second: Decimal::ZERO,
            uptime_ms: 0,
            last_quote_time: 0,
            last_fill_time: 0,
        }
    }
}

impl MetricsSnapshot {
    /// Returns true if there have been any quotes generated.
    #[must_use]
    pub fn has_activity(&self) -> bool {
        self.quotes_generated > 0 || self.orders_submitted > 0
    }

    /// Returns the rejection rate (rejected / submitted).
    #[must_use]
    pub fn rejection_rate(&self) -> Decimal {
        if self.orders_submitted == 0 {
            return Decimal::ZERO;
        }
        Decimal::from(self.orders_rejected) / Decimal::from(self.orders_submitted)
    }

    /// Returns the partial fill rate (partial fills / total fills).
    #[must_use]
    pub fn partial_fill_rate(&self) -> Decimal {
        if self.orders_filled == 0 {
            return Decimal::ZERO;
        }
        Decimal::from(self.partial_fills) / Decimal::from(self.orders_filled)
    }
}

/// Live metrics tracker for real-time trading monitoring.
///
/// Provides thread-safe tracking of operational metrics using atomic
/// operations for counters and RwLock for Decimal values.
///
/// # Thread Safety
///
/// All operations are thread-safe. Counters use atomic operations with
/// relaxed ordering (sufficient for counters). Decimal values use RwLock
/// since Decimal is not atomic-compatible.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::LiveMetrics;
/// use market_maker_rs::dec;
///
/// let metrics = LiveMetrics::new(1000);
///
/// // Record activity
/// metrics.record_quote(1001);
/// metrics.record_order_submitted();
/// metrics.increment_open_orders();
/// metrics.record_order_filled(1002);
/// metrics.decrement_open_orders();
///
/// // Get snapshot
/// let snapshot = metrics.snapshot(1100);
/// assert_eq!(snapshot.quotes_generated, 1);
/// assert_eq!(snapshot.orders_filled, 1);
/// ```
#[derive(Debug)]
pub struct LiveMetrics {
    /// Start time in milliseconds.
    start_time: AtomicU64,

    // Counters
    quotes_generated: Counter,
    orders_submitted: Counter,
    orders_filled: Counter,
    orders_cancelled: Counter,
    orders_rejected: Counter,
    partial_fills: Counter,

    // Gauges
    open_orders: Gauge,

    // Values requiring mutex (Decimal not atomic)
    position: Arc<RwLock<Decimal>>,
    realized_pnl: Arc<RwLock<Decimal>>,
    unrealized_pnl: Arc<RwLock<Decimal>>,

    // Timestamps
    last_quote_time: AtomicU64,
    last_fill_time: AtomicU64,
}

impl LiveMetrics {
    /// Creates a new live metrics tracker.
    ///
    /// # Arguments
    ///
    /// * `start_time` - Start timestamp in milliseconds
    #[must_use]
    pub fn new(start_time: u64) -> Self {
        Self {
            start_time: AtomicU64::new(start_time),
            quotes_generated: Counter::new(),
            orders_submitted: Counter::new(),
            orders_filled: Counter::new(),
            orders_cancelled: Counter::new(),
            orders_rejected: Counter::new(),
            partial_fills: Counter::new(),
            open_orders: Gauge::new(),
            position: Arc::new(RwLock::new(Decimal::ZERO)),
            realized_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            unrealized_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            last_quote_time: AtomicU64::new(0),
            last_fill_time: AtomicU64::new(0),
        }
    }

    /// Returns the start time in milliseconds.
    #[must_use]
    pub fn start_time(&self) -> u64 {
        self.start_time.load(Ordering::Relaxed)
    }

    // Counter methods

    /// Records a quote generation event.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Timestamp of the quote in milliseconds
    pub fn record_quote(&self, timestamp: u64) {
        self.quotes_generated.increment();
        self.last_quote_time.store(timestamp, Ordering::Relaxed);
    }

    /// Records an order submission event.
    pub fn record_order_submitted(&self) {
        self.orders_submitted.increment();
    }

    /// Records an order fill event.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Timestamp of the fill in milliseconds
    pub fn record_order_filled(&self, timestamp: u64) {
        self.orders_filled.increment();
        self.last_fill_time.store(timestamp, Ordering::Relaxed);
    }

    /// Records an order cancellation event.
    pub fn record_order_cancelled(&self) {
        self.orders_cancelled.increment();
    }

    /// Records an order rejection event.
    pub fn record_order_rejected(&self) {
        self.orders_rejected.increment();
    }

    /// Records a partial fill event.
    pub fn record_partial_fill(&self) {
        self.partial_fills.increment();
    }

    /// Records multiple quotes at once.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of quotes to record
    /// * `timestamp` - Timestamp of the last quote
    pub fn record_quotes(&self, count: u64, timestamp: u64) {
        self.quotes_generated.add(count);
        self.last_quote_time.store(timestamp, Ordering::Relaxed);
    }

    // Gauge methods

    /// Sets the number of open orders.
    pub fn set_open_orders(&self, count: i64) {
        self.open_orders.set(count);
    }

    /// Increments the open orders count by one.
    pub fn increment_open_orders(&self) {
        self.open_orders.increment();
    }

    /// Decrements the open orders count by one.
    pub fn decrement_open_orders(&self) {
        self.open_orders.decrement();
    }

    /// Returns the current open orders count.
    #[must_use]
    pub fn get_open_orders(&self) -> i64 {
        self.open_orders.get()
    }

    // Position/PnL methods

    /// Updates the current position.
    ///
    /// # Arguments
    ///
    /// * `position` - New position value
    pub fn update_position(&self, position: Decimal) {
        if let Ok(mut pos) = self.position.write() {
            *pos = position;
        }
    }

    /// Updates the PnL values.
    ///
    /// # Arguments
    ///
    /// * `realized` - Realized PnL
    /// * `unrealized` - Unrealized PnL
    pub fn update_pnl(&self, realized: Decimal, unrealized: Decimal) {
        if let Ok(mut rpnl) = self.realized_pnl.write() {
            *rpnl = realized;
        }
        if let Ok(mut upnl) = self.unrealized_pnl.write() {
            *upnl = unrealized;
        }
    }

    /// Updates only the realized PnL.
    pub fn update_realized_pnl(&self, realized: Decimal) {
        if let Ok(mut rpnl) = self.realized_pnl.write() {
            *rpnl = realized;
        }
    }

    /// Updates only the unrealized PnL.
    pub fn update_unrealized_pnl(&self, unrealized: Decimal) {
        if let Ok(mut upnl) = self.unrealized_pnl.write() {
            *upnl = unrealized;
        }
    }

    /// Adds to the realized PnL.
    pub fn add_realized_pnl(&self, amount: Decimal) {
        if let Ok(mut rpnl) = self.realized_pnl.write() {
            *rpnl += amount;
        }
    }

    /// Returns the current position.
    #[must_use]
    pub fn get_position(&self) -> Decimal {
        self.position.read().map(|p| *p).unwrap_or(Decimal::ZERO)
    }

    /// Returns the current realized PnL.
    #[must_use]
    pub fn get_realized_pnl(&self) -> Decimal {
        self.realized_pnl
            .read()
            .map(|p| *p)
            .unwrap_or(Decimal::ZERO)
    }

    /// Returns the current unrealized PnL.
    #[must_use]
    pub fn get_unrealized_pnl(&self) -> Decimal {
        self.unrealized_pnl
            .read()
            .map(|p| *p)
            .unwrap_or(Decimal::ZERO)
    }

    /// Creates a snapshot of current metrics.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    #[must_use]
    pub fn snapshot(&self, current_time: u64) -> MetricsSnapshot {
        let start = self.start_time.load(Ordering::Relaxed);
        let uptime_ms = current_time.saturating_sub(start);

        let quotes_generated = self.quotes_generated.get();
        let orders_submitted = self.orders_submitted.get();
        let orders_filled = self.orders_filled.get();
        let orders_cancelled = self.orders_cancelled.get();

        // Calculate rates
        let fill_rate = if quotes_generated > 0 {
            Decimal::from(orders_filled) / Decimal::from(quotes_generated)
        } else {
            Decimal::ZERO
        };

        let cancel_rate = if orders_submitted > 0 {
            Decimal::from(orders_cancelled) / Decimal::from(orders_submitted)
        } else {
            Decimal::ZERO
        };

        // Per-second rates (uptime in ms, so multiply by 1000)
        let (quotes_per_second, fills_per_second) = if uptime_ms > 0 {
            let seconds = Decimal::from(uptime_ms) / Decimal::from(1000u64);
            (
                Decimal::from(quotes_generated) / seconds,
                Decimal::from(orders_filled) / seconds,
            )
        } else {
            (Decimal::ZERO, Decimal::ZERO)
        };

        // Read Decimal values
        let position = self.get_position();
        let realized = self.get_realized_pnl();
        let unrealized = self.get_unrealized_pnl();

        MetricsSnapshot {
            timestamp: current_time,
            quotes_generated,
            orders_submitted,
            orders_filled,
            orders_cancelled,
            orders_rejected: self.orders_rejected.get(),
            partial_fills: self.partial_fills.get(),
            open_orders: self.open_orders.get(),
            current_position: position,
            current_pnl: realized + unrealized,
            unrealized_pnl: unrealized,
            realized_pnl: realized,
            fill_rate,
            cancel_rate,
            quotes_per_second,
            fills_per_second,
            uptime_ms,
            last_quote_time: self.last_quote_time.load(Ordering::Relaxed),
            last_fill_time: self.last_fill_time.load(Ordering::Relaxed),
        }
    }

    /// Resets all metrics for a new trading session.
    ///
    /// # Arguments
    ///
    /// * `new_start_time` - New start timestamp in milliseconds
    pub fn reset(&self, new_start_time: u64) {
        self.start_time.store(new_start_time, Ordering::Relaxed);

        // Reset counters
        self.quotes_generated.reset();
        self.orders_submitted.reset();
        self.orders_filled.reset();
        self.orders_cancelled.reset();
        self.orders_rejected.reset();
        self.partial_fills.reset();

        // Reset gauges
        self.open_orders.set(0);

        // Reset Decimal values
        if let Ok(mut pos) = self.position.write() {
            *pos = Decimal::ZERO;
        }
        if let Ok(mut rpnl) = self.realized_pnl.write() {
            *rpnl = Decimal::ZERO;
        }
        if let Ok(mut upnl) = self.unrealized_pnl.write() {
            *upnl = Decimal::ZERO;
        }

        // Reset timestamps
        self.last_quote_time.store(0, Ordering::Relaxed);
        self.last_fill_time.store(0, Ordering::Relaxed);
    }

    // Direct counter access for advanced use cases

    /// Returns the total quotes generated.
    #[must_use]
    pub fn total_quotes(&self) -> u64 {
        self.quotes_generated.get()
    }

    /// Returns the total orders submitted.
    #[must_use]
    pub fn total_orders_submitted(&self) -> u64 {
        self.orders_submitted.get()
    }

    /// Returns the total orders filled.
    #[must_use]
    pub fn total_orders_filled(&self) -> u64 {
        self.orders_filled.get()
    }

    /// Returns the total orders cancelled.
    #[must_use]
    pub fn total_orders_cancelled(&self) -> u64 {
        self.orders_cancelled.get()
    }

    /// Returns the total orders rejected.
    #[must_use]
    pub fn total_orders_rejected(&self) -> u64 {
        self.orders_rejected.get()
    }

    /// Returns the total partial fills.
    #[must_use]
    pub fn total_partial_fills(&self) -> u64 {
        self.partial_fills.get()
    }
}

impl Default for LiveMetrics {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Thread-safe shared live metrics.
///
/// Use this type alias when sharing metrics across threads.
pub type SharedLiveMetrics = Arc<LiveMetrics>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;
    use std::thread;

    // Counter tests
    #[test]
    fn test_counter_new() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_counter_with_value() {
        let counter = Counter::with_value(10);
        assert_eq!(counter.get(), 10);
    }

    #[test]
    fn test_counter_increment() {
        let counter = Counter::new();
        counter.increment();
        counter.increment();
        assert_eq!(counter.get(), 2);
    }

    #[test]
    fn test_counter_add() {
        let counter = Counter::new();
        counter.add(5);
        counter.add(3);
        assert_eq!(counter.get(), 8);
    }

    #[test]
    fn test_counter_reset() {
        let counter = Counter::new();
        counter.add(10);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    // Gauge tests
    #[test]
    fn test_gauge_new() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0);
    }

    #[test]
    fn test_gauge_with_value() {
        let gauge = Gauge::with_value(10);
        assert_eq!(gauge.get(), 10);
    }

    #[test]
    fn test_gauge_set() {
        let gauge = Gauge::new();
        gauge.set(42);
        assert_eq!(gauge.get(), 42);
    }

    #[test]
    fn test_gauge_increment_decrement() {
        let gauge = Gauge::new();
        gauge.increment();
        gauge.increment();
        assert_eq!(gauge.get(), 2);

        gauge.decrement();
        assert_eq!(gauge.get(), 1);
    }

    #[test]
    fn test_gauge_add_sub() {
        let gauge = Gauge::new();
        gauge.add(10);
        assert_eq!(gauge.get(), 10);

        gauge.sub(3);
        assert_eq!(gauge.get(), 7);
    }

    #[test]
    fn test_gauge_negative() {
        let gauge = Gauge::new();
        gauge.decrement();
        assert_eq!(gauge.get(), -1);
    }

    // MetricsSnapshot tests
    #[test]
    fn test_metrics_snapshot_default() {
        let snapshot = MetricsSnapshot::default();
        assert_eq!(snapshot.timestamp, 0);
        assert_eq!(snapshot.quotes_generated, 0);
        assert!(!snapshot.has_activity());
    }

    #[test]
    fn test_metrics_snapshot_has_activity() {
        let snapshot = MetricsSnapshot {
            quotes_generated: 1,
            ..Default::default()
        };
        assert!(snapshot.has_activity());
    }

    #[test]
    fn test_metrics_snapshot_rejection_rate() {
        let snapshot = MetricsSnapshot {
            orders_submitted: 100,
            orders_rejected: 5,
            ..Default::default()
        };
        assert_eq!(snapshot.rejection_rate(), dec!(0.05));
    }

    #[test]
    fn test_metrics_snapshot_partial_fill_rate() {
        let snapshot = MetricsSnapshot {
            orders_filled: 100,
            partial_fills: 20,
            ..Default::default()
        };
        assert_eq!(snapshot.partial_fill_rate(), dec!(0.2));
    }

    // LiveMetrics tests
    #[test]
    fn test_live_metrics_new() {
        let metrics = LiveMetrics::new(1000);
        assert_eq!(metrics.start_time(), 1000);
        assert_eq!(metrics.total_quotes(), 0);
    }

    #[test]
    fn test_live_metrics_record_quote() {
        let metrics = LiveMetrics::new(1000);
        metrics.record_quote(1001);
        metrics.record_quote(1002);
        assert_eq!(metrics.total_quotes(), 2);
    }

    #[test]
    fn test_live_metrics_record_orders() {
        let metrics = LiveMetrics::new(1000);
        metrics.record_order_submitted();
        metrics.record_order_submitted();
        metrics.record_order_filled(1001);
        metrics.record_order_cancelled();
        metrics.record_order_rejected();

        assert_eq!(metrics.total_orders_submitted(), 2);
        assert_eq!(metrics.total_orders_filled(), 1);
        assert_eq!(metrics.total_orders_cancelled(), 1);
        assert_eq!(metrics.total_orders_rejected(), 1);
    }

    #[test]
    fn test_live_metrics_open_orders() {
        let metrics = LiveMetrics::new(1000);
        metrics.increment_open_orders();
        metrics.increment_open_orders();
        assert_eq!(metrics.get_open_orders(), 2);

        metrics.decrement_open_orders();
        assert_eq!(metrics.get_open_orders(), 1);

        metrics.set_open_orders(10);
        assert_eq!(metrics.get_open_orders(), 10);
    }

    #[test]
    fn test_live_metrics_position() {
        let metrics = LiveMetrics::new(1000);
        metrics.update_position(dec!(100.5));
        assert_eq!(metrics.get_position(), dec!(100.5));
    }

    #[test]
    fn test_live_metrics_pnl() {
        let metrics = LiveMetrics::new(1000);
        metrics.update_pnl(dec!(100.0), dec!(50.0));

        assert_eq!(metrics.get_realized_pnl(), dec!(100.0));
        assert_eq!(metrics.get_unrealized_pnl(), dec!(50.0));
    }

    #[test]
    fn test_live_metrics_add_realized_pnl() {
        let metrics = LiveMetrics::new(1000);
        metrics.add_realized_pnl(dec!(50.0));
        metrics.add_realized_pnl(dec!(30.0));
        assert_eq!(metrics.get_realized_pnl(), dec!(80.0));
    }

    #[test]
    fn test_live_metrics_snapshot() {
        let metrics = LiveMetrics::new(1000);

        metrics.record_quote(1001);
        metrics.record_quote(1002);
        metrics.record_order_submitted();
        metrics.record_order_filled(1003);
        metrics.update_position(dec!(10.0));
        metrics.update_pnl(dec!(100.0), dec!(50.0));

        let snapshot = metrics.snapshot(2000);

        assert_eq!(snapshot.timestamp, 2000);
        assert_eq!(snapshot.quotes_generated, 2);
        assert_eq!(snapshot.orders_submitted, 1);
        assert_eq!(snapshot.orders_filled, 1);
        assert_eq!(snapshot.current_position, dec!(10.0));
        assert_eq!(snapshot.realized_pnl, dec!(100.0));
        assert_eq!(snapshot.unrealized_pnl, dec!(50.0));
        assert_eq!(snapshot.current_pnl, dec!(150.0));
        assert_eq!(snapshot.uptime_ms, 1000);
        assert_eq!(snapshot.last_quote_time, 1002);
        assert_eq!(snapshot.last_fill_time, 1003);
    }

    #[test]
    fn test_live_metrics_snapshot_rates() {
        let metrics = LiveMetrics::new(0);

        // 10 quotes, 5 fills over 10 seconds
        metrics.record_quotes(10, 10000);
        for _ in 0..5 {
            metrics.record_order_filled(10000);
        }

        let snapshot = metrics.snapshot(10000);

        // Fill rate = 5/10 = 0.5
        assert_eq!(snapshot.fill_rate, dec!(0.5));

        // Quotes per second = 10 / 10 = 1
        assert_eq!(snapshot.quotes_per_second, dec!(1.0));

        // Fills per second = 5 / 10 = 0.5
        assert_eq!(snapshot.fills_per_second, dec!(0.5));
    }

    #[test]
    fn test_live_metrics_reset() {
        let metrics = LiveMetrics::new(1000);

        metrics.record_quote(1001);
        metrics.record_order_submitted();
        metrics.update_position(dec!(100.0));

        metrics.reset(2000);

        assert_eq!(metrics.start_time(), 2000);
        assert_eq!(metrics.total_quotes(), 0);
        assert_eq!(metrics.total_orders_submitted(), 0);
        assert_eq!(metrics.get_position(), Decimal::ZERO);
    }

    #[test]
    fn test_live_metrics_thread_safety() {
        let metrics = Arc::new(LiveMetrics::new(0));
        let mut handles = vec![];

        // Spawn multiple threads that increment counters
        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    m.record_quote(1);
                    m.record_order_submitted();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 10 threads * 100 iterations = 1000
        assert_eq!(metrics.total_quotes(), 1000);
        assert_eq!(metrics.total_orders_submitted(), 1000);
    }

    #[test]
    fn test_live_metrics_concurrent_pnl_updates() {
        let metrics = Arc::new(LiveMetrics::new(0));
        let mut handles = vec![];

        // Spawn threads that update PnL
        for i in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                m.add_realized_pnl(Decimal::from(i + 1));
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Sum of 1..=10 = 55
        assert_eq!(metrics.get_realized_pnl(), dec!(55));
    }

    #[test]
    fn test_shared_live_metrics() {
        let metrics: SharedLiveMetrics = Arc::new(LiveMetrics::new(0));
        metrics.record_quote(1);
        assert_eq!(metrics.total_quotes(), 1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_metrics_snapshot_serialization() {
        let snapshot = MetricsSnapshot {
            timestamp: 1000,
            quotes_generated: 100,
            orders_filled: 50,
            current_pnl: dec!(1000.0),
            ..Default::default()
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let deserialized: MetricsSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.timestamp, 1000);
        assert_eq!(deserialized.quotes_generated, 100);
        assert_eq!(deserialized.orders_filled, 50);
    }
}
