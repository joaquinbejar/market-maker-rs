//! Latency tracking and metrics for order execution and market data.
//!
//! This module provides tools for measuring and reporting latency metrics
//! critical for market making performance.
//!
//! # Overview
//!
//! The latency tracker measures:
//!
//! - **Order-to-Ack**: Time from order submission to exchange acknowledgment
//! - **Order-to-Fill**: Time from submission to first fill
//! - **Order-to-Cancel**: Time from cancel request to confirmation
//! - **Market Data Delay**: Delay in receiving market data updates
//! - **Round-Trip Time**: Full cycle for order operations
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     LatencyTracker, LatencyTrackerConfig, LatencyMetric
//! };
//!
//! let config = LatencyTrackerConfig::default();
//! let mut tracker = LatencyTracker::new(config);
//!
//! // Record order-to-ack latency (500 microseconds)
//! tracker.record(LatencyMetric::OrderToAck, 500, 1000);
//!
//! // Get statistics
//! if let Some(stats) = tracker.get_stats(LatencyMetric::OrderToAck) {
//!     println!("Avg latency: {} us", stats.avg_us);
//!     println!("P99 latency: {} us", stats.p99_us);
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use crate::types::decimal::saturating_u64;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single latency measurement.
///
/// Stores the latency value in microseconds along with the timestamp
/// when the measurement was taken.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::LatencyMeasurement;
///
/// let measurement = LatencyMeasurement::new(500, 1000);
/// assert_eq!(measurement.value_us, 500);
/// assert_eq!(measurement.as_millis(), 0); // 500us < 1ms
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatencyMeasurement {
    /// Latency value in microseconds.
    pub value_us: u64,
    /// Timestamp when measurement was taken, in milliseconds.
    pub timestamp: u64,
}

impl LatencyMeasurement {
    /// Creates a new latency measurement.
    #[must_use]
    pub fn new(value_us: u64, timestamp: u64) -> Self {
        Self {
            value_us,
            timestamp,
        }
    }

    /// Creates a measurement from a Duration.
    #[must_use]
    pub fn from_duration(duration: Duration, timestamp: u64) -> Self {
        Self {
            value_us: saturating_u64(duration.as_micros()),
            timestamp,
        }
    }

    /// Returns the latency as milliseconds.
    #[must_use]
    pub fn as_millis(&self) -> u64 {
        self.value_us / 1000
    }

    /// Returns the latency as a Duration.
    #[must_use]
    pub fn as_duration(&self) -> Duration {
        Duration::from_micros(self.value_us)
    }
}

/// Latency statistics computed from measurements.
///
/// Provides min, max, average, standard deviation, and percentiles.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::LatencyStats;
///
/// let stats = LatencyStats::default();
/// assert_eq!(stats.count, 0);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatencyStats {
    /// Number of measurements.
    pub count: u64,
    /// Minimum latency in microseconds.
    pub min_us: u64,
    /// Maximum latency in microseconds.
    pub max_us: u64,
    /// Average latency in microseconds.
    pub avg_us: u64,
    /// 50th percentile (median) in microseconds.
    pub p50_us: u64,
    /// 95th percentile in microseconds.
    pub p95_us: u64,
    /// 99th percentile in microseconds.
    pub p99_us: u64,
    /// Standard deviation in microseconds.
    pub std_dev_us: u64,
}

impl LatencyStats {
    /// Computes statistics from a slice of measurements.
    #[must_use]
    pub fn from_measurements(measurements: &[LatencyMeasurement]) -> Self {
        if measurements.is_empty() {
            return Self::default();
        }

        let mut values: Vec<u64> = measurements.iter().map(|m| m.value_us).collect();
        values.sort_unstable();

        let (Some(&min_us), Some(&max_us)) = (values.first(), values.last()) else {
            return Self::default();
        };
        let count = saturating_u64(values.len());
        // Non zero: `values` is not empty.
        let n = u128::from(count).max(1);
        // A sum of at most 2^64 values below 2^64 always fits in a u128,
        // and the mean is at most the largest value, so it narrows back.
        let sum: u128 = values.iter().map(|&v| u128::from(v)).sum();
        // `n >= 1`, so the division cannot fail.
        #[allow(clippy::arithmetic_side_effects)]
        let avg_us = u64::try_from(sum / n).unwrap_or(u64::MAX);

        // Population variance, floor(sum(d^2) / n), without overflowing:
        // each squared deviation fits in a u128, and summing the per term
        // quotients and remainders separately keeps both sums in range.
        // `n >= 1`, so the divisions and remainders cannot fail.
        #[allow(clippy::arithmetic_side_effects)]
        let (quotients, remainders) = values.iter().fold((0u128, 0u128), |(q, r), &v| {
            let diff = u128::from(v.abs_diff(avg_us));
            let square = diff.saturating_mul(diff);
            (q.saturating_add(square / n), r.saturating_add(square % n))
        });
        // `n >= 1`, so the division cannot fail.
        #[allow(clippy::arithmetic_side_effects)]
        let variance = quotients.saturating_add(remainders / n);
        let std_dev_us = u64::try_from(variance.isqrt()).unwrap_or(u64::MAX);

        // Calculate percentiles
        let p50_us = percentile(&values, 50.0);
        let p95_us = percentile(&values, 95.0);
        let p99_us = percentile(&values, 99.0);

        Self {
            count,
            min_us,
            max_us,
            avg_us,
            p50_us,
            p95_us,
            p99_us,
            std_dev_us,
        }
    }

    /// Returns true if no measurements have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Latency metric type.
///
/// Identifies the type of latency being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LatencyMetric {
    /// Time from order submission to exchange acknowledgment.
    OrderToAck,
    /// Time from order submission to first fill.
    OrderToFill,
    /// Time from cancel request to confirmation.
    OrderToCancel,
    /// Delay in receiving market data updates.
    MarketDataDelay,
    /// Full round-trip time for order operations.
    RoundTrip,
}

impl LatencyMetric {
    /// Returns all metric variants.
    #[must_use]
    pub fn all() -> &'static [LatencyMetric] {
        &[
            LatencyMetric::OrderToAck,
            LatencyMetric::OrderToFill,
            LatencyMetric::OrderToCancel,
            LatencyMetric::MarketDataDelay,
            LatencyMetric::RoundTrip,
        ]
    }

    /// Returns the metric name as a string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            LatencyMetric::OrderToAck => "order_to_ack",
            LatencyMetric::OrderToFill => "order_to_fill",
            LatencyMetric::OrderToCancel => "order_to_cancel",
            LatencyMetric::MarketDataDelay => "market_data_delay",
            LatencyMetric::RoundTrip => "round_trip",
        }
    }
}

/// Configuration for the latency tracker.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::LatencyTrackerConfig;
///
/// let config = LatencyTrackerConfig::default()
///     .with_window_size(500)
///     .with_histogram(100, 100);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatencyTrackerConfig {
    /// Window size for rolling statistics.
    pub window_size: usize,
    /// Whether to keep full histogram.
    pub keep_histogram: bool,
    /// Histogram bucket size in microseconds.
    pub histogram_bucket_us: u64,
    /// Number of histogram buckets.
    pub histogram_num_buckets: usize,
}

impl Default for LatencyTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            keep_histogram: false,
            histogram_bucket_us: 100,
            histogram_num_buckets: 100,
        }
    }
}

impl LatencyTrackerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the window size for rolling statistics.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Enables histogram tracking with specified bucket size and count.
    #[must_use]
    pub fn with_histogram(mut self, bucket_us: u64, num_buckets: usize) -> Self {
        self.keep_histogram = true;
        self.histogram_bucket_us = bucket_us;
        self.histogram_num_buckets = num_buckets;
        self
    }

    /// Disables histogram tracking.
    #[must_use]
    pub fn without_histogram(mut self) -> Self {
        self.keep_histogram = false;
        self
    }
}

/// Simple histogram for latency distribution.
///
/// Provides efficient percentile approximation using fixed-size buckets.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::Histogram;
///
/// let mut hist = Histogram::new(100, 100); // 100us buckets, 100 buckets (0-10ms)
/// hist.record(500);  // 500us
/// hist.record(1500); // 1.5ms
///
/// let p50 = hist.percentile(50.0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Histogram {
    buckets: Vec<u64>,
    bucket_size_us: u64,
    total_count: u64,
    overflow_count: u64,
}

impl Histogram {
    /// Creates a new histogram.
    ///
    /// # Arguments
    ///
    /// * `bucket_size_us` - Size of each bucket in microseconds
    /// * `num_buckets` - Number of buckets (values above max go to overflow)
    #[must_use]
    pub fn new(bucket_size_us: u64, num_buckets: usize) -> Self {
        Self {
            buckets: vec![0; num_buckets],
            bucket_size_us,
            total_count: 0,
            overflow_count: 0,
        }
    }

    /// Records a value in the histogram.
    pub fn record(&mut self, value_us: u64) {
        let bucket = value_us
            .checked_div(self.bucket_size_us)
            .and_then(|idx| usize::try_from(idx).ok())
            .and_then(|idx| self.buckets.get_mut(idx));
        match bucket {
            Some(count) => *count = count.saturating_add(1),
            None => self.overflow_count = self.overflow_count.saturating_add(1),
        }
        self.total_count = self.total_count.saturating_add(1);
    }

    /// Calculates the approximate percentile value.
    ///
    /// # Arguments
    ///
    /// * `p` - Percentile to calculate (0.0 to 100.0)
    #[must_use]
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total_count == 0 {
            return 0;
        }

        // Stats boundary, not money: a count above 2^53 loses precision
        // only in the target rank. The float to int `as` cast saturates
        // (NaN maps to 0), which is the clamping wanted here.
        let target = ((p / 100.0) * self.total_count as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative = cumulative.saturating_add(count);
            if cumulative >= target {
                // Return the upper bound of this bucket
                return saturating_u64(i)
                    .saturating_add(1)
                    .saturating_mul(self.bucket_size_us);
            }
        }

        // Value is in overflow bucket
        self.max_trackable_us()
    }

    /// Returns the bucket counts.
    #[must_use]
    pub fn get_buckets(&self) -> &[u64] {
        &self.buckets
    }

    /// Returns the total count of recorded values.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Returns the overflow count (values above max bucket).
    #[must_use]
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Returns the bucket size in microseconds.
    #[must_use]
    pub fn bucket_size_us(&self) -> u64 {
        self.bucket_size_us
    }

    /// Returns the maximum trackable value (before overflow).
    #[must_use]
    pub fn max_trackable_us(&self) -> u64 {
        self.bucket_size_us
            .saturating_mul(saturating_u64(self.buckets.len()))
    }

    /// Resets the histogram.
    pub fn reset(&mut self) {
        self.buckets.fill(0);
        self.total_count = 0;
        self.overflow_count = 0;
    }
}

/// Latency tracker for measuring and reporting latency metrics.
///
/// Maintains rolling windows of measurements for each metric type
/// and optionally tracks histograms for distribution analysis.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{
///     LatencyTracker, LatencyTrackerConfig, LatencyMetric
/// };
/// use std::time::Duration;
///
/// let mut tracker = LatencyTracker::new(LatencyTrackerConfig::default());
///
/// // Record measurements
/// tracker.record(LatencyMetric::OrderToAck, 500, 1000);
/// tracker.record(LatencyMetric::OrderToAck, 600, 1001);
/// tracker.record_duration(LatencyMetric::OrderToFill, Duration::from_millis(5), 1002);
///
/// // Get statistics
/// let stats = tracker.get_stats(LatencyMetric::OrderToAck).unwrap();
/// assert_eq!(stats.count, 2);
///
/// // Check for degradation
/// let degraded = tracker.is_degraded(LatencyMetric::OrderToAck, 1000);
/// assert!(!degraded); // avg is 550us, below 1000us threshold
/// ```
#[derive(Debug)]
pub struct LatencyTracker {
    config: LatencyTrackerConfig,
    measurements: HashMap<LatencyMetric, VecDeque<LatencyMeasurement>>,
    histograms: Option<HashMap<LatencyMetric, Histogram>>,
}

impl LatencyTracker {
    /// Creates a new latency tracker.
    #[must_use]
    pub fn new(config: LatencyTrackerConfig) -> Self {
        let histograms = if config.keep_histogram {
            let mut h = HashMap::new();
            for metric in LatencyMetric::all() {
                h.insert(
                    *metric,
                    Histogram::new(config.histogram_bucket_us, config.histogram_num_buckets),
                );
            }
            Some(h)
        } else {
            None
        };

        Self {
            config,
            measurements: HashMap::new(),
            histograms,
        }
    }

    /// Creates a tracker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(LatencyTrackerConfig::default())
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &LatencyTrackerConfig {
        &self.config
    }

    /// Records a latency measurement in microseconds.
    ///
    /// # Arguments
    ///
    /// * `metric` - The type of latency being measured
    /// * `latency_us` - Latency value in microseconds
    /// * `timestamp` - Timestamp when measurement was taken (milliseconds)
    pub fn record(&mut self, metric: LatencyMetric, latency_us: u64, timestamp: u64) {
        let measurement = LatencyMeasurement::new(latency_us, timestamp);

        let measurements = self.measurements.entry(metric).or_default();
        measurements.push_back(measurement);

        // Trim to window size
        while measurements.len() > self.config.window_size {
            measurements.pop_front();
        }

        // Update histogram if enabled
        if let Some(ref mut histograms) = self.histograms
            && let Some(hist) = histograms.get_mut(&metric)
        {
            hist.record(latency_us);
        }
    }

    /// Records latency from a Duration.
    ///
    /// # Arguments
    ///
    /// * `metric` - The type of latency being measured
    /// * `duration` - Latency as a Duration
    /// * `timestamp` - Timestamp when measurement was taken (milliseconds)
    pub fn record_duration(&mut self, metric: LatencyMetric, duration: Duration, timestamp: u64) {
        self.record(metric, saturating_u64(duration.as_micros()), timestamp);
    }

    /// Gets statistics for a specific metric.
    #[must_use]
    pub fn get_stats(&self, metric: LatencyMetric) -> Option<LatencyStats> {
        self.measurements.get(&metric).map(|m| {
            let vec: Vec<LatencyMeasurement> = m.iter().copied().collect();
            LatencyStats::from_measurements(&vec)
        })
    }

    /// Gets statistics for all metrics.
    #[must_use]
    pub fn get_all_stats(&self) -> HashMap<LatencyMetric, LatencyStats> {
        self.measurements
            .iter()
            .map(|(metric, m)| {
                let vec: Vec<LatencyMeasurement> = m.iter().copied().collect();
                (*metric, LatencyStats::from_measurements(&vec))
            })
            .collect()
    }

    /// Gets recent measurements for a metric.
    #[must_use]
    pub fn get_recent(&self, metric: LatencyMetric, count: usize) -> Vec<&LatencyMeasurement> {
        self.measurements
            .get(&metric)
            .map(|m| m.iter().rev().take(count).collect())
            .unwrap_or_default()
    }

    /// Checks if the average latency exceeds a threshold.
    ///
    /// Returns true if the average latency for the metric exceeds
    /// the specified threshold in microseconds.
    #[must_use]
    pub fn is_degraded(&self, metric: LatencyMetric, threshold_us: u64) -> bool {
        self.get_stats(metric)
            .map(|stats| stats.avg_us > threshold_us)
            .unwrap_or(false)
    }

    /// Checks if the P99 latency exceeds a threshold.
    #[must_use]
    pub fn is_p99_degraded(&self, metric: LatencyMetric, threshold_us: u64) -> bool {
        self.get_stats(metric)
            .map(|stats| stats.p99_us > threshold_us)
            .unwrap_or(false)
    }

    /// Gets the histogram for a metric (if enabled).
    #[must_use]
    pub fn get_histogram(&self, metric: LatencyMetric) -> Option<&Histogram> {
        self.histograms.as_ref().and_then(|h| h.get(&metric))
    }

    /// Returns the number of measurements for a metric.
    #[must_use]
    pub fn measurement_count(&self, metric: LatencyMetric) -> usize {
        self.measurements.get(&metric).map(|m| m.len()).unwrap_or(0)
    }

    /// Returns the total number of measurements across all metrics.
    #[must_use]
    pub fn total_measurement_count(&self) -> usize {
        self.measurements.values().map(|m| m.len()).sum()
    }

    /// Resets all measurements and histograms.
    pub fn reset(&mut self) {
        self.measurements.clear();
        if let Some(ref mut histograms) = self.histograms {
            for hist in histograms.values_mut() {
                hist.reset();
            }
        }
    }

    /// Resets measurements for a specific metric.
    pub fn reset_metric(&mut self, metric: LatencyMetric) {
        self.measurements.remove(&metric);
        if let Some(ref mut histograms) = self.histograms
            && let Some(hist) = histograms.get_mut(&metric)
        {
            hist.reset();
        }
    }
}

/// Calculates the percentile value from a sorted slice.
fn percentile(sorted_values: &[u64], p: f64) -> u64 {
    let last = sorted_values.len().saturating_sub(1);
    // Stats boundary, not money: lossless below 2^53 elements, and the
    // float to int `as` cast saturates (NaN maps to 0) before clamping.
    let idx = ((p / 100.0) * last as f64).round() as usize;
    sorted_values.get(idx.min(last)).copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_measurement_new() {
        let m = LatencyMeasurement::new(500, 1000);
        assert_eq!(m.value_us, 500);
        assert_eq!(m.timestamp, 1000);
        assert_eq!(m.as_millis(), 0);
    }

    #[test]
    fn test_latency_measurement_from_duration() {
        let m = LatencyMeasurement::from_duration(Duration::from_millis(5), 1000);
        assert_eq!(m.value_us, 5000);
        assert_eq!(m.as_millis(), 5);
    }

    #[test]
    fn test_latency_measurement_as_duration() {
        let m = LatencyMeasurement::new(5000, 1000);
        assert_eq!(m.as_duration(), Duration::from_micros(5000));
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyStats::from_measurements(&[]);
        assert!(stats.is_empty());
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_latency_stats_single() {
        let measurements = vec![LatencyMeasurement::new(500, 1000)];
        let stats = LatencyStats::from_measurements(&measurements);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.min_us, 500);
        assert_eq!(stats.max_us, 500);
        assert_eq!(stats.avg_us, 500);
        assert_eq!(stats.p50_us, 500);
        assert_eq!(stats.std_dev_us, 0);
    }

    #[test]
    fn test_latency_stats_multiple() {
        let measurements: Vec<LatencyMeasurement> = (1..=100)
            .map(|i| LatencyMeasurement::new(i * 100, i))
            .collect();

        let stats = LatencyStats::from_measurements(&measurements);

        assert_eq!(stats.count, 100);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 10000);
        assert_eq!(stats.avg_us, 5050); // (1+100)/2 * 100
        // Percentiles use nearest-rank method
        assert!(stats.p50_us >= 4900 && stats.p50_us <= 5100);
        assert!(stats.p95_us >= 9400 && stats.p95_us <= 9600);
        assert!(stats.p99_us >= 9800 && stats.p99_us <= 10000);
    }

    #[test]
    fn test_latency_metric_all() {
        let all = LatencyMetric::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_latency_metric_as_str() {
        assert_eq!(LatencyMetric::OrderToAck.as_str(), "order_to_ack");
        assert_eq!(LatencyMetric::OrderToFill.as_str(), "order_to_fill");
        assert_eq!(LatencyMetric::MarketDataDelay.as_str(), "market_data_delay");
    }

    #[test]
    fn test_latency_tracker_config_default() {
        let config = LatencyTrackerConfig::default();
        assert_eq!(config.window_size, 1000);
        assert!(!config.keep_histogram);
    }

    #[test]
    fn test_latency_tracker_config_builder() {
        let config = LatencyTrackerConfig::new()
            .with_window_size(500)
            .with_histogram(50, 200);

        assert_eq!(config.window_size, 500);
        assert!(config.keep_histogram);
        assert_eq!(config.histogram_bucket_us, 50);
        assert_eq!(config.histogram_num_buckets, 200);
    }

    #[test]
    fn test_histogram_new() {
        let hist = Histogram::new(100, 100);
        assert_eq!(hist.total_count(), 0);
        assert_eq!(hist.bucket_size_us(), 100);
        assert_eq!(hist.max_trackable_us(), 10000);
    }

    #[test]
    fn test_histogram_record() {
        let mut hist = Histogram::new(100, 100);
        hist.record(50); // bucket 0
        hist.record(150); // bucket 1
        hist.record(250); // bucket 2

        assert_eq!(hist.total_count(), 3);
        assert_eq!(hist.get_buckets()[0], 1);
        assert_eq!(hist.get_buckets()[1], 1);
        assert_eq!(hist.get_buckets()[2], 1);
    }

    #[test]
    fn test_histogram_overflow() {
        let mut hist = Histogram::new(100, 10); // max 1000us
        hist.record(500);
        hist.record(1500); // overflow

        assert_eq!(hist.total_count(), 2);
        assert_eq!(hist.overflow_count(), 1);
    }

    #[test]
    fn test_histogram_record_zero_bucket_size_counts_overflow() {
        let mut hist = Histogram::new(0, 10);
        hist.record(500);

        assert_eq!(hist.total_count(), 1);
        assert_eq!(hist.overflow_count(), 1);
        assert_eq!(hist.max_trackable_us(), 0);
    }

    #[test]
    fn test_measurement_from_duration_huge_saturates() {
        let m = LatencyMeasurement::from_duration(Duration::MAX, 7);
        assert_eq!(m.value_us, u64::MAX);
    }

    #[test]
    fn test_latency_stats_huge_values_saturate() {
        let measurements = [
            LatencyMeasurement::new(u64::MAX, 1),
            LatencyMeasurement::new(u64::MAX, 2),
        ];
        let stats = LatencyStats::from_measurements(&measurements);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.max_us, u64::MAX);
        assert_eq!(stats.avg_us, u64::MAX);
        assert_eq!(stats.std_dev_us, 0);
    }

    #[test]
    fn test_latency_stats_extreme_spread_mean_and_std_are_exact() {
        let measurements = [
            LatencyMeasurement::new(0, 1),
            LatencyMeasurement::new(u64::MAX, 2),
        ];
        let stats = LatencyStats::from_measurements(&measurements);
        assert_eq!(stats.avg_us, u64::MAX / 2);
        // Both deviations are about 2^63, so the population std is too.
        assert_eq!(stats.std_dev_us, u64::MAX / 2);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut hist = Histogram::new(100, 100);
        for i in 1..=100 {
            hist.record(i * 100 - 50); // 50, 150, 250, ..., 9950
        }

        // P50 should be around 5000us
        let p50 = hist.percentile(50.0);
        assert!((4900..=5100).contains(&p50));

        // P99 should be around 9900us
        let p99 = hist.percentile(99.0);
        assert!((9800..=10000).contains(&p99));
    }

    #[test]
    fn test_histogram_reset() {
        let mut hist = Histogram::new(100, 100);
        hist.record(500);
        hist.record(600);
        hist.reset();

        assert_eq!(hist.total_count(), 0);
        assert_eq!(hist.overflow_count(), 0);
    }

    #[test]
    fn test_latency_tracker_record() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToAck, 600, 1001);

        assert_eq!(tracker.measurement_count(LatencyMetric::OrderToAck), 2);
    }

    #[test]
    fn test_latency_tracker_record_duration() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record_duration(LatencyMetric::OrderToFill, Duration::from_millis(5), 1000);

        let stats = tracker.get_stats(LatencyMetric::OrderToFill).unwrap();
        assert_eq!(stats.avg_us, 5000);
    }

    #[test]
    fn test_latency_tracker_get_stats() {
        let mut tracker = LatencyTracker::with_defaults();

        for i in 1..=10 {
            tracker.record(LatencyMetric::OrderToAck, i * 100, i);
        }

        let stats = tracker.get_stats(LatencyMetric::OrderToAck).unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 1000);
        assert_eq!(stats.avg_us, 550);
    }

    #[test]
    fn test_latency_tracker_get_all_stats() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToFill, 5000, 1001);

        let all_stats = tracker.get_all_stats();
        assert_eq!(all_stats.len(), 2);
        assert!(all_stats.contains_key(&LatencyMetric::OrderToAck));
        assert!(all_stats.contains_key(&LatencyMetric::OrderToFill));
    }

    #[test]
    fn test_latency_tracker_get_recent() {
        let mut tracker = LatencyTracker::with_defaults();

        for i in 1..=10 {
            tracker.record(LatencyMetric::OrderToAck, i * 100, i);
        }

        let recent = tracker.get_recent(LatencyMetric::OrderToAck, 3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].value_us, 1000); // Most recent first
        assert_eq!(recent[1].value_us, 900);
        assert_eq!(recent[2].value_us, 800);
    }

    #[test]
    fn test_latency_tracker_is_degraded() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToAck, 600, 1001);

        // Avg is 550us
        assert!(!tracker.is_degraded(LatencyMetric::OrderToAck, 1000));
        assert!(tracker.is_degraded(LatencyMetric::OrderToAck, 500));
    }

    #[test]
    fn test_latency_tracker_is_p99_degraded() {
        let mut tracker = LatencyTracker::with_defaults();

        for i in 1..=100 {
            tracker.record(LatencyMetric::OrderToAck, i * 10, i);
        }

        // P99 is around 990us
        assert!(!tracker.is_p99_degraded(LatencyMetric::OrderToAck, 1000));
        assert!(tracker.is_p99_degraded(LatencyMetric::OrderToAck, 900));
    }

    #[test]
    fn test_latency_tracker_window_rotation() {
        let config = LatencyTrackerConfig::default().with_window_size(5);
        let mut tracker = LatencyTracker::new(config);

        for i in 1..=10 {
            tracker.record(LatencyMetric::OrderToAck, i * 100, i);
        }

        // Only last 5 should remain
        assert_eq!(tracker.measurement_count(LatencyMetric::OrderToAck), 5);

        let stats = tracker.get_stats(LatencyMetric::OrderToAck).unwrap();
        assert_eq!(stats.min_us, 600); // 6th measurement
        assert_eq!(stats.max_us, 1000); // 10th measurement
    }

    #[test]
    fn test_latency_tracker_with_histogram() {
        let config = LatencyTrackerConfig::default().with_histogram(100, 100);
        let mut tracker = LatencyTracker::new(config);

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToAck, 600, 1001);

        let hist = tracker.get_histogram(LatencyMetric::OrderToAck).unwrap();
        assert_eq!(hist.total_count(), 2);
    }

    #[test]
    fn test_latency_tracker_reset() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToFill, 5000, 1001);

        tracker.reset();

        assert_eq!(tracker.total_measurement_count(), 0);
    }

    #[test]
    fn test_latency_tracker_reset_metric() {
        let mut tracker = LatencyTracker::with_defaults();

        tracker.record(LatencyMetric::OrderToAck, 500, 1000);
        tracker.record(LatencyMetric::OrderToFill, 5000, 1001);

        tracker.reset_metric(LatencyMetric::OrderToAck);

        assert_eq!(tracker.measurement_count(LatencyMetric::OrderToAck), 0);
        assert_eq!(tracker.measurement_count(LatencyMetric::OrderToFill), 1);
    }

    #[test]
    fn test_percentile_function() {
        let values = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        assert_eq!(percentile(&values, 0.0), 100);
        // P50 of 10 values: index = round(0.5 * 9) = 5, value = 600
        assert_eq!(percentile(&values, 50.0), 600);
        assert_eq!(percentile(&values, 100.0), 1000);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let measurement = LatencyMeasurement::new(500, 1000);
        let json = serde_json::to_string(&measurement).unwrap();
        let deserialized: LatencyMeasurement = serde_json::from_str(&json).unwrap();
        assert_eq!(measurement, deserialized);
    }
}
