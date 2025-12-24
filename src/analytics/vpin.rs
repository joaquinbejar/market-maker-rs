//! VPIN (Volume-Synchronized Probability of Informed Trading) calculation.
//!
//! This module implements the VPIN metric to detect toxic order flow and
//! informed trading activity. VPIN was introduced by Easley, López de Prado,
//! and O'Hara in their 2012 paper "Flow Toxicity and Liquidity in a
//! High-Frequency World".
//!
//! # Algorithm
//!
//! VPIN measures order flow toxicity by:
//!
//! 1. Grouping trades into volume buckets (not time buckets)
//! 2. Classifying trades as buy or sell initiated
//! 3. Calculating the absolute imbalance within each bucket
//! 4. Averaging imbalance over a rolling window of buckets
//!
//! # Formula
//!
//! For each bucket `i`:
//! ```text
//! imbalance_i = |V_buy - V_sell| / V_total
//! ```
//!
//! VPIN is the average of the last N bucket imbalances:
//! ```text
//! VPIN = (1/N) * Σ imbalance_i
//! ```
//!
//! # Interpretation
//!
//! - VPIN range: [0, 1]
//! - Higher values indicate more toxic (informed) flow
//! - Typical alert threshold: 0.7
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::vpin::{VPINCalculator, VPINConfig};
//! use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
//! use market_maker_rs::dec;
//!
//! let config = VPINConfig::new(dec!(100.0), 5, dec!(0.7)).unwrap();
//! let mut calculator = VPINCalculator::new(config);
//!
//! // Add trades to fill buckets
//! for i in 0..10 {
//!     calculator.add_trade(&Trade::new(dec!(100.0), dec!(25.0), TradeSide::Buy, i * 1000));
//! }
//!
//! if let Some(vpin) = calculator.get_vpin() {
//!     println!("Current VPIN: {}", vpin);
//! }
//! ```
//!
//! # References
//!
//! Easley, D., López de Prado, M. M., & O'Hara, M. (2012).
//! "Flow Toxicity and Liquidity in a High-Frequency World."
//! The Review of Financial Studies, 25(5), 1457-1493.

use std::collections::VecDeque;

use crate::Decimal;
use crate::analytics::order_flow::{Trade, TradeSide};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for VPIN calculation.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::vpin::VPINConfig;
/// use market_maker_rs::dec;
///
/// let config = VPINConfig::new(
///     dec!(1000.0),  // bucket_volume: 1000 units per bucket
///     50,            // num_buckets: 50 buckets for rolling average
///     dec!(0.7),     // toxicity_threshold: alert at 70%
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VPINConfig {
    /// Volume per bucket in base currency units.
    pub bucket_volume: Decimal,

    /// Number of buckets for rolling average.
    pub num_buckets: usize,

    /// Toxicity alert threshold (e.g., 0.7).
    pub toxicity_threshold: Decimal,
}

impl VPINConfig {
    /// Creates a new `VPINConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `bucket_volume` - Volume per bucket, must be positive
    /// * `num_buckets` - Number of buckets for rolling average, must be > 0
    /// * `toxicity_threshold` - Alert threshold, must be in [0, 1]
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::analytics::vpin::VPINConfig;
    /// use market_maker_rs::dec;
    ///
    /// let config = VPINConfig::new(dec!(1000.0), 50, dec!(0.7)).unwrap();
    /// assert_eq!(config.bucket_volume, dec!(1000.0));
    /// ```
    pub fn new(
        bucket_volume: Decimal,
        num_buckets: usize,
        toxicity_threshold: Decimal,
    ) -> MMResult<Self> {
        if bucket_volume <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "bucket_volume must be positive".to_string(),
            ));
        }

        if num_buckets == 0 {
            return Err(MMError::InvalidConfiguration(
                "num_buckets must be greater than 0".to_string(),
            ));
        }

        if toxicity_threshold < Decimal::ZERO || toxicity_threshold > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "toxicity_threshold must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            bucket_volume,
            num_buckets,
            toxicity_threshold,
        })
    }

    /// Creates a config with typical parameters.
    ///
    /// Uses 50 buckets and 0.7 toxicity threshold as recommended in the paper.
    ///
    /// # Arguments
    ///
    /// * `bucket_volume` - Volume per bucket
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if bucket_volume is not positive.
    pub fn with_defaults(bucket_volume: Decimal) -> MMResult<Self> {
        Self::new(bucket_volume, 50, Decimal::from_str_exact("0.7").unwrap())
    }
}

/// A completed volume bucket for VPIN calculation.
///
/// Each bucket represents a fixed volume of trading activity,
/// with buy and sell volumes tracked separately.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VolumeBucket {
    /// Total buy volume in this bucket.
    pub buy_volume: Decimal,

    /// Total sell volume in this bucket.
    pub sell_volume: Decimal,

    /// Total volume in this bucket (buy + sell).
    pub total_volume: Decimal,

    /// Absolute imbalance: |buy - sell| / total.
    pub imbalance: Decimal,

    /// Timestamp of first trade in bucket (milliseconds).
    pub start_time: u64,

    /// Timestamp of last trade in bucket (milliseconds).
    pub end_time: u64,

    /// Number of trades in this bucket.
    pub trade_count: u64,
}

impl VolumeBucket {
    /// Creates a new empty volume bucket.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buy_volume: Decimal::ZERO,
            sell_volume: Decimal::ZERO,
            total_volume: Decimal::ZERO,
            imbalance: Decimal::ZERO,
            start_time: 0,
            end_time: 0,
            trade_count: 0,
        }
    }

    /// Returns the signed imbalance (positive = buy pressure).
    #[must_use]
    pub fn signed_imbalance(&self) -> Decimal {
        if self.total_volume > Decimal::ZERO {
            (self.buy_volume - self.sell_volume) / self.total_volume
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the duration of this bucket in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Returns true if this bucket has any trades.
    #[must_use]
    pub fn has_trades(&self) -> bool {
        self.trade_count > 0
    }

    /// Returns the buy/sell ratio, or None if no sells.
    #[must_use]
    pub fn buy_sell_ratio(&self) -> Option<Decimal> {
        if self.sell_volume > Decimal::ZERO {
            Some(self.buy_volume / self.sell_volume)
        } else {
            None
        }
    }
}

impl Default for VolumeBucket {
    fn default() -> Self {
        Self::new()
    }
}

/// VPIN calculator implementing the volume-synchronized probability
/// of informed trading metric.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::analytics::vpin::{VPINCalculator, VPINConfig};
/// use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
/// use market_maker_rs::dec;
///
/// let config = VPINConfig::new(dec!(100.0), 5, dec!(0.7)).unwrap();
/// let mut calculator = VPINCalculator::new(config);
///
/// // Add trades - each bucket needs 100 volume
/// calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Buy, 1000));
/// calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Sell, 2000));
///
/// // First bucket complete (balanced), VPIN not available yet (need 5 buckets)
/// assert!(calculator.get_vpin().is_none());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VPINCalculator {
    /// Configuration parameters.
    config: VPINConfig,

    /// Completed volume buckets.
    completed_buckets: VecDeque<VolumeBucket>,

    /// Current (incomplete) bucket being filled.
    current_bucket: VolumeBucket,

    /// Total trades processed.
    total_trades: u64,

    /// Total volume processed.
    total_volume: Decimal,
}

impl VPINCalculator {
    /// Creates a new VPIN calculator with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - VPIN configuration parameters
    #[must_use]
    pub fn new(config: VPINConfig) -> Self {
        Self {
            config,
            completed_buckets: VecDeque::new(),
            current_bucket: VolumeBucket::new(),
            total_trades: 0,
            total_volume: Decimal::ZERO,
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &VPINConfig {
        &self.config
    }

    /// Returns the number of completed buckets.
    #[must_use]
    pub fn completed_bucket_count(&self) -> usize {
        self.completed_buckets.len()
    }

    /// Returns true if enough buckets are available for VPIN calculation.
    #[must_use]
    pub fn has_sufficient_buckets(&self) -> bool {
        self.completed_buckets.len() >= self.config.num_buckets
    }

    /// Returns total trades processed.
    #[must_use]
    pub fn total_trades(&self) -> u64 {
        self.total_trades
    }

    /// Returns total volume processed.
    #[must_use]
    pub fn total_volume(&self) -> Decimal {
        self.total_volume
    }

    /// Adds a trade and updates VPIN calculation.
    ///
    /// Returns the new VPIN value if enough buckets are available.
    ///
    /// # Arguments
    ///
    /// * `trade` - The trade to add
    ///
    /// # Returns
    ///
    /// `Some(vpin)` if enough buckets are available, `None` otherwise.
    pub fn add_trade(&mut self, trade: &Trade) -> Option<Decimal> {
        self.total_trades += 1;
        self.total_volume += trade.size;

        // Update current bucket timestamps
        if self.current_bucket.trade_count == 0 {
            self.current_bucket.start_time = trade.timestamp;
        }
        self.current_bucket.end_time = trade.timestamp;
        self.current_bucket.trade_count += 1;

        // Add volume to appropriate side
        match trade.side {
            TradeSide::Buy => {
                self.current_bucket.buy_volume += trade.size;
            }
            TradeSide::Sell => {
                self.current_bucket.sell_volume += trade.size;
            }
        }
        self.current_bucket.total_volume += trade.size;

        // Check if bucket is complete
        if self.current_bucket.total_volume >= self.config.bucket_volume {
            self.complete_current_bucket();
        }

        self.get_vpin()
    }

    /// Completes the current bucket and starts a new one.
    fn complete_current_bucket(&mut self) {
        // Calculate imbalance for the bucket
        let imbalance = if self.current_bucket.total_volume > Decimal::ZERO {
            let diff = self.current_bucket.buy_volume - self.current_bucket.sell_volume;
            diff.abs() / self.current_bucket.total_volume
        } else {
            Decimal::ZERO
        };
        self.current_bucket.imbalance = imbalance;

        // Handle overflow: if bucket has more than target volume,
        // we could split, but for simplicity we just complete it
        self.completed_buckets
            .push_back(self.current_bucket.clone());

        // Keep only the required number of buckets
        while self.completed_buckets.len() > self.config.num_buckets {
            self.completed_buckets.pop_front();
        }

        // Start new bucket
        self.current_bucket = VolumeBucket::new();
    }

    /// Gets the current VPIN value.
    ///
    /// Returns `None` if not enough buckets are available.
    ///
    /// # Returns
    ///
    /// VPIN value in range [0, 1], or `None` if insufficient data.
    #[must_use]
    pub fn get_vpin(&self) -> Option<Decimal> {
        if self.completed_buckets.len() < self.config.num_buckets {
            return None;
        }

        let sum: Decimal = self.completed_buckets.iter().map(|b| b.imbalance).sum();
        Some(sum / Decimal::from(self.config.num_buckets))
    }

    /// Checks if current VPIN exceeds the toxicity threshold.
    ///
    /// Returns `false` if VPIN is not available.
    #[must_use]
    pub fn is_toxic(&self) -> bool {
        self.get_vpin()
            .map(|vpin| vpin >= self.config.toxicity_threshold)
            .unwrap_or(false)
    }

    /// Gets the toxicity level as a descriptive string.
    #[must_use]
    pub fn toxicity_level(&self) -> &'static str {
        match self.get_vpin() {
            None => "unknown",
            Some(vpin) if vpin < Decimal::from_str_exact("0.3").unwrap() => "low",
            Some(vpin) if vpin < Decimal::from_str_exact("0.5").unwrap() => "moderate",
            Some(vpin) if vpin < Decimal::from_str_exact("0.7").unwrap() => "elevated",
            Some(_) => "high",
        }
    }

    /// Gets the completed buckets.
    #[must_use]
    pub fn get_buckets(&self) -> &VecDeque<VolumeBucket> {
        &self.completed_buckets
    }

    /// Gets the current (incomplete) bucket.
    #[must_use]
    pub fn get_current_bucket(&self) -> &VolumeBucket {
        &self.current_bucket
    }

    /// Gets the fill percentage of the current bucket.
    #[must_use]
    pub fn current_bucket_fill_pct(&self) -> Decimal {
        if self.config.bucket_volume > Decimal::ZERO {
            (self.current_bucket.total_volume / self.config.bucket_volume) * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Resets the calculator, clearing all buckets.
    pub fn reset(&mut self) {
        self.completed_buckets.clear();
        self.current_bucket = VolumeBucket::new();
        self.total_trades = 0;
        self.total_volume = Decimal::ZERO;
    }

    /// Gets statistics about the completed buckets.
    #[must_use]
    pub fn bucket_stats(&self) -> Option<BucketStats> {
        if self.completed_buckets.is_empty() {
            return None;
        }

        let imbalances: Vec<Decimal> = self.completed_buckets.iter().map(|b| b.imbalance).collect();
        let n = Decimal::from(imbalances.len());

        let mean = imbalances.iter().copied().sum::<Decimal>() / n;

        let min = imbalances.iter().copied().min().unwrap_or(Decimal::ZERO);
        let max = imbalances.iter().copied().max().unwrap_or(Decimal::ZERO);

        // Calculate standard deviation
        let variance = imbalances
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<Decimal>()
            / n;

        // Approximate sqrt using Newton's method
        let std_dev = decimal_sqrt_approx(variance);

        Some(BucketStats {
            count: self.completed_buckets.len(),
            mean_imbalance: mean,
            min_imbalance: min,
            max_imbalance: max,
            std_dev_imbalance: std_dev,
        })
    }
}

/// Statistics about completed VPIN buckets.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BucketStats {
    /// Number of completed buckets.
    pub count: usize,

    /// Mean imbalance across buckets.
    pub mean_imbalance: Decimal,

    /// Minimum imbalance.
    pub min_imbalance: Decimal,

    /// Maximum imbalance.
    pub max_imbalance: Decimal,

    /// Standard deviation of imbalance.
    pub std_dev_imbalance: Decimal,
}

/// Approximate square root using Newton's method.
fn decimal_sqrt_approx(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut guess = x / Decimal::from(2);
    let epsilon = Decimal::from_str_exact("0.0000001").unwrap();

    for _ in 0..20 {
        let new_guess = (guess + x / guess) / Decimal::from(2);
        if (new_guess - guess).abs() < epsilon {
            return new_guess;
        }
        guess = new_guess;
    }

    guess
}

/// Trade classifier for determining trade direction.
///
/// Provides methods to classify trades as buy or sell initiated
/// using various rules (tick rule, quote rule).
#[derive(Debug, Clone)]
pub struct TradeClassifier {
    last_price: Option<Decimal>,
}

impl TradeClassifier {
    /// Creates a new trade classifier.
    #[must_use]
    pub fn new() -> Self {
        Self { last_price: None }
    }

    /// Classifies a trade using the tick rule.
    ///
    /// - If price > last price: Buy
    /// - If price < last price: Sell
    /// - If price == last price: Use previous classification (defaults to Buy)
    ///
    /// # Arguments
    ///
    /// * `price` - Current trade price
    ///
    /// # Returns
    ///
    /// The classified trade side.
    pub fn classify_tick_rule(&mut self, price: Decimal) -> TradeSide {
        let side = match self.last_price {
            Some(last) if price > last => TradeSide::Buy,
            Some(last) if price < last => TradeSide::Sell,
            _ => TradeSide::Buy, // Default for first trade or unchanged price
        };
        self.last_price = Some(price);
        side
    }

    /// Classifies a trade using the quote rule (Lee-Ready algorithm).
    ///
    /// - If price > mid: Buy (trade at ask)
    /// - If price < mid: Sell (trade at bid)
    /// - If price == mid: Use tick rule
    ///
    /// # Arguments
    ///
    /// * `price` - Trade price
    /// * `bid` - Best bid price
    /// * `ask` - Best ask price
    ///
    /// # Returns
    ///
    /// The classified trade side.
    pub fn classify_quote_rule(&mut self, price: Decimal, bid: Decimal, ask: Decimal) -> TradeSide {
        let mid = (bid + ask) / Decimal::from(2);

        let side = if price > mid {
            TradeSide::Buy
        } else if price < mid {
            TradeSide::Sell
        } else {
            // At mid, use tick rule
            self.classify_tick_rule(price)
        };

        self.last_price = Some(price);
        side
    }

    /// Resets the classifier state.
    pub fn reset(&mut self) {
        self.last_price = None;
    }
}

impl Default for TradeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_vpin_config_valid() {
        let config = VPINConfig::new(dec!(1000.0), 50, dec!(0.7));
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.bucket_volume, dec!(1000.0));
        assert_eq!(config.num_buckets, 50);
        assert_eq!(config.toxicity_threshold, dec!(0.7));
    }

    #[test]
    fn test_vpin_config_invalid_bucket_volume() {
        let config = VPINConfig::new(dec!(0.0), 50, dec!(0.7));
        assert!(config.is_err());

        let config = VPINConfig::new(dec!(-100.0), 50, dec!(0.7));
        assert!(config.is_err());
    }

    #[test]
    fn test_vpin_config_invalid_num_buckets() {
        let config = VPINConfig::new(dec!(1000.0), 0, dec!(0.7));
        assert!(config.is_err());
    }

    #[test]
    fn test_vpin_config_invalid_threshold() {
        let config = VPINConfig::new(dec!(1000.0), 50, dec!(-0.1));
        assert!(config.is_err());

        let config = VPINConfig::new(dec!(1000.0), 50, dec!(1.1));
        assert!(config.is_err());
    }

    #[test]
    fn test_vpin_config_with_defaults() {
        let config = VPINConfig::with_defaults(dec!(1000.0)).unwrap();
        assert_eq!(config.num_buckets, 50);
        assert_eq!(config.toxicity_threshold, dec!(0.7));
    }

    #[test]
    fn test_volume_bucket_new() {
        let bucket = VolumeBucket::new();
        assert_eq!(bucket.buy_volume, Decimal::ZERO);
        assert_eq!(bucket.sell_volume, Decimal::ZERO);
        assert_eq!(bucket.total_volume, Decimal::ZERO);
        assert_eq!(bucket.trade_count, 0);
    }

    #[test]
    fn test_volume_bucket_signed_imbalance() {
        let mut bucket = VolumeBucket::new();
        bucket.buy_volume = dec!(70.0);
        bucket.sell_volume = dec!(30.0);
        bucket.total_volume = dec!(100.0);

        // (70 - 30) / 100 = 0.4
        assert_eq!(bucket.signed_imbalance(), dec!(0.4));
    }

    #[test]
    fn test_vpin_calculator_new() {
        let config = VPINConfig::new(dec!(100.0), 5, dec!(0.7)).unwrap();
        let calculator = VPINCalculator::new(config);

        assert_eq!(calculator.completed_bucket_count(), 0);
        assert!(!calculator.has_sufficient_buckets());
        assert!(calculator.get_vpin().is_none());
    }

    #[test]
    fn test_vpin_calculator_single_bucket() {
        let config = VPINConfig::new(dec!(100.0), 5, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Add trades to fill one bucket (100 volume)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Sell, 2000));

        // One bucket complete, but need 5 for VPIN
        assert_eq!(calculator.completed_bucket_count(), 1);
        assert!(calculator.get_vpin().is_none());
    }

    #[test]
    fn test_vpin_calculator_sufficient_buckets() {
        let config = VPINConfig::new(dec!(100.0), 3, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Fill 3 buckets with balanced trades (imbalance = 0)
        for i in 0..3 {
            calculator.add_trade(&Trade::new(
                dec!(100.0),
                dec!(50.0),
                TradeSide::Buy,
                i * 1000,
            ));
            calculator.add_trade(&Trade::new(
                dec!(100.0),
                dec!(50.0),
                TradeSide::Sell,
                i * 1000 + 500,
            ));
        }

        assert_eq!(calculator.completed_bucket_count(), 3);
        assert!(calculator.has_sufficient_buckets());

        let vpin = calculator.get_vpin().unwrap();
        assert_eq!(vpin, Decimal::ZERO); // All balanced buckets
    }

    #[test]
    fn test_vpin_calculator_imbalanced() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Bucket 1: All buys (imbalance = 1.0)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Buy, 1000));

        // Bucket 2: All sells (imbalance = 1.0)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Sell, 2000));

        let vpin = calculator.get_vpin().unwrap();
        assert_eq!(vpin, Decimal::ONE); // Average of 1.0 and 1.0
    }

    #[test]
    fn test_vpin_calculator_mixed_imbalance() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Bucket 1: 80 buy, 20 sell (imbalance = |80-20|/100 = 0.6)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(80.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(20.0), TradeSide::Sell, 1500));

        // Bucket 2: 30 buy, 70 sell (imbalance = |30-70|/100 = 0.4)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(30.0), TradeSide::Buy, 2000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(70.0), TradeSide::Sell, 2500));

        let vpin = calculator.get_vpin().unwrap();
        // Average of 0.6 and 0.4 = 0.5
        assert_eq!(vpin, dec!(0.5));
    }

    #[test]
    fn test_vpin_is_toxic() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Fill with highly imbalanced buckets
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Buy, 2000));

        assert!(calculator.is_toxic()); // VPIN = 1.0 > 0.7
    }

    #[test]
    fn test_vpin_not_toxic() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Fill with balanced buckets
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Sell, 1500));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Buy, 2000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Sell, 2500));

        assert!(!calculator.is_toxic()); // VPIN = 0.0 < 0.7
    }

    #[test]
    fn test_vpin_bucket_rotation() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Fill 3 buckets (only last 2 should be kept)
        for i in 0..3 {
            calculator.add_trade(&Trade::new(
                dec!(100.0),
                dec!(100.0),
                TradeSide::Buy,
                i * 1000,
            ));
        }

        assert_eq!(calculator.completed_bucket_count(), 2);
    }

    #[test]
    fn test_vpin_reset() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(100.0), TradeSide::Buy, 2000));

        assert_eq!(calculator.completed_bucket_count(), 2);

        calculator.reset();

        assert_eq!(calculator.completed_bucket_count(), 0);
        assert_eq!(calculator.total_trades(), 0);
        assert_eq!(calculator.total_volume(), Decimal::ZERO);
    }

    #[test]
    fn test_vpin_current_bucket_fill() {
        let config = VPINConfig::new(dec!(100.0), 2, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        calculator.add_trade(&Trade::new(dec!(100.0), dec!(25.0), TradeSide::Buy, 1000));

        assert_eq!(calculator.current_bucket_fill_pct(), dec!(25.0));
    }

    #[test]
    fn test_vpin_toxicity_level() {
        let config = VPINConfig::new(dec!(100.0), 1, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        assert_eq!(calculator.toxicity_level(), "unknown");

        // Low toxicity (balanced)
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(50.0), TradeSide::Sell, 1500));
        assert_eq!(calculator.toxicity_level(), "low");
    }

    #[test]
    fn test_vpin_bucket_stats() {
        let config = VPINConfig::new(dec!(100.0), 3, dec!(0.7)).unwrap();
        let mut calculator = VPINCalculator::new(config);

        // Create buckets with different imbalances
        // Bucket 1: imbalance = 0.2
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(60.0), TradeSide::Buy, 1000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(40.0), TradeSide::Sell, 1500));

        // Bucket 2: imbalance = 0.4
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(70.0), TradeSide::Buy, 2000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(30.0), TradeSide::Sell, 2500));

        // Bucket 3: imbalance = 0.6
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(80.0), TradeSide::Buy, 3000));
        calculator.add_trade(&Trade::new(dec!(100.0), dec!(20.0), TradeSide::Sell, 3500));

        let stats = calculator.bucket_stats().unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_imbalance, dec!(0.2));
        assert_eq!(stats.max_imbalance, dec!(0.6));
    }

    #[test]
    fn test_trade_classifier_tick_rule() {
        let mut classifier = TradeClassifier::new();

        // First trade defaults to Buy
        assert_eq!(classifier.classify_tick_rule(dec!(100.0)), TradeSide::Buy);

        // Price up = Buy
        assert_eq!(classifier.classify_tick_rule(dec!(101.0)), TradeSide::Buy);

        // Price down = Sell
        assert_eq!(classifier.classify_tick_rule(dec!(100.5)), TradeSide::Sell);

        // Price unchanged = previous (Buy by default)
        assert_eq!(classifier.classify_tick_rule(dec!(100.5)), TradeSide::Buy);
    }

    #[test]
    fn test_trade_classifier_quote_rule() {
        let mut classifier = TradeClassifier::new();

        // Trade above mid = Buy
        let side = classifier.classify_quote_rule(dec!(100.6), dec!(100.0), dec!(101.0));
        assert_eq!(side, TradeSide::Buy);

        // Trade below mid = Sell
        let side = classifier.classify_quote_rule(dec!(100.4), dec!(100.0), dec!(101.0));
        assert_eq!(side, TradeSide::Sell);

        // Trade at mid = use tick rule
        let side = classifier.classify_quote_rule(dec!(100.5), dec!(100.0), dec!(101.0));
        // Price went up from 100.4, so Buy
        assert_eq!(side, TradeSide::Buy);
    }

    #[test]
    fn test_trade_classifier_reset() {
        let mut classifier = TradeClassifier::new();
        classifier.classify_tick_rule(dec!(100.0));

        classifier.reset();

        // After reset, first trade defaults to Buy again
        assert_eq!(classifier.classify_tick_rule(dec!(99.0)), TradeSide::Buy);
    }

    #[test]
    fn test_volume_bucket_helpers() {
        let mut bucket = VolumeBucket::new();
        bucket.buy_volume = dec!(60.0);
        bucket.sell_volume = dec!(40.0);
        bucket.total_volume = dec!(100.0);
        bucket.start_time = 1000;
        bucket.end_time = 5000;
        bucket.trade_count = 10;

        assert!(bucket.has_trades());
        assert_eq!(bucket.duration_ms(), 4000);
        assert_eq!(bucket.buy_sell_ratio(), Some(dec!(1.5)));
    }

    #[test]
    fn test_decimal_sqrt_approx() {
        let result = decimal_sqrt_approx(dec!(4.0));
        assert!((result - dec!(2.0)).abs() < dec!(0.0001));

        let result = decimal_sqrt_approx(dec!(9.0));
        assert!((result - dec!(3.0)).abs() < dec!(0.0001));

        let result = decimal_sqrt_approx(dec!(0.0));
        assert_eq!(result, Decimal::ZERO);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_vpin_config_serialization() {
        let config = VPINConfig::new(dec!(1000.0), 50, dec!(0.7)).unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VPINConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
