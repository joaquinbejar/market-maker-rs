//! Drawdown tracking for risk management.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Record of a drawdown event.
///
/// Captures the details of a drawdown period from peak to trough.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DrawdownRecord {
    /// Drawdown as a decimal (e.g., 0.10 for 10% drawdown).
    pub drawdown: Decimal,
    /// Timestamp when this drawdown was recorded, in milliseconds.
    pub timestamp: u64,
    /// Peak equity value before the drawdown.
    pub peak_equity: Decimal,
    /// Trough equity value (lowest point).
    pub trough_equity: Decimal,
}

/// Tracks drawdown from peak equity for risk management.
///
/// Drawdown is the decline from a historical peak in equity. This tracker
/// maintains the peak equity value and calculates current drawdown in real-time.
///
/// # Drawdown Calculation
///
/// Drawdown is calculated as: `(peak - current) / peak`
///
/// For example, if peak equity was $10,000 and current equity is $9,000,
/// the drawdown is (10000 - 9000) / 10000 = 0.10 (10%).
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::DrawdownTracker;
/// use market_maker_rs::dec;
///
/// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
///
/// // Equity increases - new peak
/// tracker.update(dec!(11000.0), 1000);
/// assert_eq!(tracker.peak_equity(), dec!(11000.0));
/// assert_eq!(tracker.current_drawdown(), dec!(0.0));
///
/// // Equity decreases - drawdown begins
/// tracker.update(dec!(9900.0), 2000);
/// assert_eq!(tracker.current_drawdown(), dec!(0.1)); // 10% drawdown
/// assert!(!tracker.is_max_drawdown_reached()); // Below 20% limit
///
/// // Further decline
/// tracker.update(dec!(8800.0), 3000);
/// assert_eq!(tracker.current_drawdown(), dec!(0.2)); // 20% drawdown
/// assert!(tracker.is_max_drawdown_reached()); // At 20% limit
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DrawdownTracker {
    /// Historical peak equity value.
    peak_equity: Decimal,
    /// Current equity value.
    current_equity: Decimal,
    /// Maximum allowed drawdown as decimal (e.g., 0.10 for 10%).
    max_allowed_drawdown: Decimal,
    /// Timestamp when peak equity was reached, in milliseconds.
    peak_timestamp: u64,
    /// Maximum historical drawdown observed.
    max_historical_drawdown: Decimal,
    /// Historical drawdown records.
    drawdown_history: Vec<DrawdownRecord>,
    /// Maximum number of records to keep in history.
    max_history_size: usize,
}

impl DrawdownTracker {
    /// Creates a new `DrawdownTracker` with initial equity.
    ///
    /// # Arguments
    ///
    /// * `initial_equity` - Starting equity value (must be positive)
    /// * `max_allowed_drawdown` - Maximum allowed drawdown as decimal (must be in (0, 1])
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if:
    /// - `initial_equity` is not positive
    /// - `max_allowed_drawdown` is not in (0, 1]
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// assert_eq!(tracker.peak_equity(), dec!(10000.0));
    /// ```
    pub fn new(initial_equity: Decimal, max_allowed_drawdown: Decimal) -> MMResult<Self> {
        if initial_equity <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "initial_equity must be positive".to_string(),
            ));
        }

        if max_allowed_drawdown <= Decimal::ZERO || max_allowed_drawdown > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "max_allowed_drawdown must be between 0 (exclusive) and 1 (inclusive)".to_string(),
            ));
        }

        Ok(Self {
            peak_equity: initial_equity,
            current_equity: initial_equity,
            max_allowed_drawdown,
            peak_timestamp: 0,
            max_historical_drawdown: Decimal::ZERO,
            drawdown_history: Vec::new(),
            max_history_size: 1000,
        })
    }

    /// Creates a new `DrawdownTracker` with initial equity and timestamp.
    ///
    /// # Arguments
    ///
    /// * `initial_equity` - Starting equity value (must be positive)
    /// * `max_allowed_drawdown` - Maximum allowed drawdown as decimal
    /// * `timestamp` - Initial timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    pub fn with_timestamp(
        initial_equity: Decimal,
        max_allowed_drawdown: Decimal,
        timestamp: u64,
    ) -> MMResult<Self> {
        let mut tracker = Self::new(initial_equity, max_allowed_drawdown)?;
        tracker.peak_timestamp = timestamp;
        Ok(tracker)
    }

    /// Sets the maximum history size for drawdown records.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum number of records to keep
    #[must_use]
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Updates the tracker with a new equity value.
    ///
    /// If equity exceeds the current peak, the peak is updated.
    /// If equity is below the peak, drawdown is calculated and recorded.
    ///
    /// # Arguments
    ///
    /// * `equity` - New equity value
    /// * `timestamp` - Timestamp of the update in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    ///
    /// // New high
    /// tracker.update(dec!(12000.0), 1000);
    /// assert_eq!(tracker.peak_equity(), dec!(12000.0));
    ///
    /// // Drawdown
    /// tracker.update(dec!(10800.0), 2000);
    /// assert_eq!(tracker.current_drawdown(), dec!(0.1)); // 10%
    /// ```
    pub fn update(&mut self, equity: Decimal, timestamp: u64) {
        self.current_equity = equity;

        if equity > self.peak_equity {
            // New peak
            self.peak_equity = equity;
            self.peak_timestamp = timestamp;
        } else if self.peak_equity > Decimal::ZERO {
            // Calculate current drawdown
            let drawdown = (self.peak_equity - equity) / self.peak_equity;

            // Update max historical drawdown
            if drawdown > self.max_historical_drawdown {
                self.max_historical_drawdown = drawdown;
            }

            // Record significant drawdowns (> 1%)
            if drawdown > Decimal::new(1, 2) {
                self.record_drawdown(drawdown, timestamp, equity);
            }
        }
    }

    /// Records a drawdown event in history.
    fn record_drawdown(&mut self, drawdown: Decimal, timestamp: u64, trough_equity: Decimal) {
        let record = DrawdownRecord {
            drawdown,
            timestamp,
            peak_equity: self.peak_equity,
            trough_equity,
        };

        self.drawdown_history.push(record);

        // Prune history if needed
        if self.drawdown_history.len() > self.max_history_size {
            self.drawdown_history.remove(0);
        }
    }

    /// Returns the current drawdown as a decimal (0.0 to 1.0).
    ///
    /// A value of 0.10 means 10% drawdown from peak.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// tracker.update(dec!(9000.0), 1000);
    /// assert_eq!(tracker.current_drawdown(), dec!(0.1)); // 10%
    /// ```
    #[must_use]
    pub fn current_drawdown(&self) -> Decimal {
        if self.peak_equity <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let drawdown = (self.peak_equity - self.current_equity) / self.peak_equity;
        drawdown.max(Decimal::ZERO)
    }

    /// Returns the current drawdown as a percentage (0.0 to 100.0).
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// tracker.update(dec!(9000.0), 1000);
    /// assert_eq!(tracker.current_drawdown_pct(), dec!(10.0)); // 10%
    /// ```
    #[must_use]
    pub fn current_drawdown_pct(&self) -> Decimal {
        self.current_drawdown() * Decimal::ONE_HUNDRED
    }

    /// Checks if the maximum allowed drawdown has been reached or exceeded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.10)).unwrap();
    ///
    /// tracker.update(dec!(9500.0), 1000);
    /// assert!(!tracker.is_max_drawdown_reached()); // 5% < 10%
    ///
    /// tracker.update(dec!(9000.0), 2000);
    /// assert!(tracker.is_max_drawdown_reached()); // 10% >= 10%
    /// ```
    #[must_use]
    pub fn is_max_drawdown_reached(&self) -> bool {
        self.current_drawdown() >= self.max_allowed_drawdown
    }

    /// Returns the current peak equity value.
    #[must_use]
    pub fn peak_equity(&self) -> Decimal {
        self.peak_equity
    }

    /// Returns the current equity value.
    #[must_use]
    pub fn current_equity(&self) -> Decimal {
        self.current_equity
    }

    /// Returns the timestamp when peak equity was reached.
    #[must_use]
    pub fn peak_timestamp(&self) -> u64 {
        self.peak_timestamp
    }

    /// Returns the maximum allowed drawdown threshold.
    #[must_use]
    pub fn max_allowed_drawdown(&self) -> Decimal {
        self.max_allowed_drawdown
    }

    /// Returns the maximum historical drawdown observed.
    ///
    /// This is the worst drawdown seen since the tracker was created or last reset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.50)).unwrap();
    ///
    /// tracker.update(dec!(8000.0), 1000); // 20% drawdown
    /// tracker.update(dec!(9000.0), 2000); // Recovery to 10% drawdown
    /// tracker.update(dec!(10000.0), 3000); // Full recovery, new peak
    /// tracker.update(dec!(9500.0), 4000); // 5% drawdown
    ///
    /// // Max historical is still 20% from the first drawdown
    /// assert_eq!(tracker.max_historical_drawdown(), dec!(0.2));
    /// ```
    #[must_use]
    pub fn max_historical_drawdown(&self) -> Decimal {
        self.max_historical_drawdown
    }

    /// Returns the drawdown history.
    #[must_use]
    pub fn drawdown_history(&self) -> &[DrawdownRecord] {
        &self.drawdown_history
    }

    /// Returns the distance to max drawdown as a decimal.
    ///
    /// This is how much more drawdown can occur before hitting the limit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// tracker.update(dec!(9000.0), 1000); // 10% drawdown
    ///
    /// // 10% remaining before hitting 20% limit
    /// assert_eq!(tracker.distance_to_max_drawdown(), dec!(0.10));
    /// ```
    #[must_use]
    pub fn distance_to_max_drawdown(&self) -> Decimal {
        let distance = self.max_allowed_drawdown - self.current_drawdown();
        distance.max(Decimal::ZERO)
    }

    /// Returns the equity value at which max drawdown would be reached.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// tracker.update(dec!(12000.0), 1000); // New peak
    ///
    /// // 20% drawdown from 12000 = 9600
    /// assert_eq!(tracker.equity_at_max_drawdown(), dec!(9600.0));
    /// ```
    #[must_use]
    pub fn equity_at_max_drawdown(&self) -> Decimal {
        self.peak_equity * (Decimal::ONE - self.max_allowed_drawdown)
    }

    /// Resets the tracker with new equity.
    ///
    /// Clears history and sets new peak/current equity.
    ///
    /// # Arguments
    ///
    /// * `new_equity` - New starting equity value
    /// * `timestamp` - Timestamp of the reset in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::DrawdownTracker;
    /// use market_maker_rs::dec;
    ///
    /// let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
    /// tracker.update(dec!(8000.0), 1000);
    ///
    /// tracker.reset(dec!(15000.0), 100000);
    /// assert_eq!(tracker.peak_equity(), dec!(15000.0));
    /// assert_eq!(tracker.current_drawdown(), dec!(0.0));
    /// assert_eq!(tracker.max_historical_drawdown(), dec!(0.0));
    /// ```
    pub fn reset(&mut self, new_equity: Decimal, timestamp: u64) {
        self.peak_equity = new_equity;
        self.current_equity = new_equity;
        self.peak_timestamp = timestamp;
        self.max_historical_drawdown = Decimal::ZERO;
        self.drawdown_history.clear();
    }

    /// Resets only the peak (soft reset).
    ///
    /// Sets current equity as new peak without clearing history.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Timestamp of the reset in milliseconds
    pub fn reset_peak(&mut self, timestamp: u64) {
        self.peak_equity = self.current_equity;
        self.peak_timestamp = timestamp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_new_valid() {
        let tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20));
        assert!(tracker.is_ok());

        let tracker = tracker.unwrap();
        assert_eq!(tracker.peak_equity(), dec!(10000.0));
        assert_eq!(tracker.current_equity(), dec!(10000.0));
        assert_eq!(tracker.max_allowed_drawdown(), dec!(0.20));
        assert_eq!(tracker.current_drawdown(), dec!(0.0));
    }

    #[test]
    fn test_new_invalid_equity() {
        let result = DrawdownTracker::new(dec!(0.0), dec!(0.20));
        assert!(result.is_err());

        let result = DrawdownTracker::new(dec!(-1000.0), dec!(0.20));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_max_drawdown() {
        let result = DrawdownTracker::new(dec!(10000.0), dec!(0.0));
        assert!(result.is_err());

        let result = DrawdownTracker::new(dec!(10000.0), dec!(-0.1));
        assert!(result.is_err());

        let result = DrawdownTracker::new(dec!(10000.0), dec!(1.1));
        assert!(result.is_err());

        // 1.0 (100%) should be valid
        let result = DrawdownTracker::new(dec!(10000.0), dec!(1.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_with_timestamp() {
        let tracker = DrawdownTracker::with_timestamp(dec!(10000.0), dec!(0.20), 12345).unwrap();
        assert_eq!(tracker.peak_timestamp(), 12345);
    }

    #[test]
    fn test_update_new_peak() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        tracker.update(dec!(11000.0), 1000);
        assert_eq!(tracker.peak_equity(), dec!(11000.0));
        assert_eq!(tracker.current_equity(), dec!(11000.0));
        assert_eq!(tracker.peak_timestamp(), 1000);
        assert_eq!(tracker.current_drawdown(), dec!(0.0));
    }

    #[test]
    fn test_update_drawdown() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        tracker.update(dec!(9000.0), 1000);
        assert_eq!(tracker.peak_equity(), dec!(10000.0)); // Peak unchanged
        assert_eq!(tracker.current_equity(), dec!(9000.0));
        assert_eq!(tracker.current_drawdown(), dec!(0.1)); // 10%
        assert_eq!(tracker.current_drawdown_pct(), dec!(10.0));
    }

    #[test]
    fn test_max_drawdown_reached() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.10)).unwrap();

        tracker.update(dec!(9500.0), 1000);
        assert!(!tracker.is_max_drawdown_reached()); // 5% < 10%

        tracker.update(dec!(9000.0), 2000);
        assert!(tracker.is_max_drawdown_reached()); // 10% >= 10%

        tracker.update(dec!(8000.0), 3000);
        assert!(tracker.is_max_drawdown_reached()); // 20% > 10%
    }

    #[test]
    fn test_max_historical_drawdown() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.50)).unwrap();

        tracker.update(dec!(8000.0), 1000); // 20% drawdown
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.2));

        tracker.update(dec!(9000.0), 2000); // Recovery to 10%
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.2)); // Still 20%

        tracker.update(dec!(10000.0), 3000); // Full recovery
        tracker.update(dec!(9500.0), 4000); // 5% drawdown
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.2)); // Still 20%

        tracker.update(dec!(7500.0), 5000); // 25% drawdown
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.25)); // Updated to 25%
    }

    #[test]
    fn test_distance_to_max_drawdown() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        assert_eq!(tracker.distance_to_max_drawdown(), dec!(0.20));

        tracker.update(dec!(9000.0), 1000); // 10% drawdown
        assert_eq!(tracker.distance_to_max_drawdown(), dec!(0.10));

        tracker.update(dec!(8000.0), 2000); // 20% drawdown
        assert_eq!(tracker.distance_to_max_drawdown(), dec!(0.0));

        tracker.update(dec!(7000.0), 3000); // 30% drawdown
        assert_eq!(tracker.distance_to_max_drawdown(), dec!(0.0)); // Can't go negative
    }

    #[test]
    fn test_equity_at_max_drawdown() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();
        assert_eq!(tracker.equity_at_max_drawdown(), dec!(8000.0));

        tracker.update(dec!(12000.0), 1000); // New peak
        assert_eq!(tracker.equity_at_max_drawdown(), dec!(9600.0));
    }

    #[test]
    fn test_reset() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        tracker.update(dec!(8000.0), 1000);
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.2));

        tracker.reset(dec!(15000.0), 100000);
        assert_eq!(tracker.peak_equity(), dec!(15000.0));
        assert_eq!(tracker.current_equity(), dec!(15000.0));
        assert_eq!(tracker.peak_timestamp(), 100000);
        assert_eq!(tracker.current_drawdown(), dec!(0.0));
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.0));
    }

    #[test]
    fn test_reset_peak() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        tracker.update(dec!(8000.0), 1000); // 20% drawdown
        assert_eq!(tracker.current_drawdown(), dec!(0.2));

        tracker.reset_peak(2000);
        assert_eq!(tracker.peak_equity(), dec!(8000.0));
        assert_eq!(tracker.current_drawdown(), dec!(0.0));
        // History preserved
        assert_eq!(tracker.max_historical_drawdown(), dec!(0.2));
    }

    #[test]
    fn test_drawdown_history() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.50)).unwrap();

        // Small drawdown (< 1%) - not recorded
        tracker.update(dec!(9950.0), 1000);
        assert!(tracker.drawdown_history().is_empty());

        // Significant drawdown (> 1%) - recorded
        tracker.update(dec!(9000.0), 2000);
        assert_eq!(tracker.drawdown_history().len(), 1);

        let record = &tracker.drawdown_history()[0];
        assert_eq!(record.drawdown, dec!(0.1));
        assert_eq!(record.peak_equity, dec!(10000.0));
        assert_eq!(record.trough_equity, dec!(9000.0));
        assert_eq!(record.timestamp, 2000);
    }

    #[test]
    fn test_history_pruning() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.50))
            .unwrap()
            .with_max_history_size(3);

        // Add 5 records
        for i in 1..=5 {
            let equity = dec!(10000.0) - Decimal::from(i) * dec!(500.0);
            tracker.update(equity, i as u64 * 1000);
        }

        // Should only keep last 3
        assert_eq!(tracker.drawdown_history().len(), 3);
    }

    #[test]
    fn test_recovery_and_new_peak() {
        let mut tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        // Drawdown
        tracker.update(dec!(9000.0), 1000);
        assert_eq!(tracker.current_drawdown(), dec!(0.1));

        // Partial recovery
        tracker.update(dec!(9500.0), 2000);
        assert_eq!(tracker.current_drawdown(), dec!(0.05));
        assert_eq!(tracker.peak_equity(), dec!(10000.0)); // Peak unchanged

        // Full recovery
        tracker.update(dec!(10000.0), 3000);
        assert_eq!(tracker.current_drawdown(), dec!(0.0));

        // New peak
        tracker.update(dec!(11000.0), 4000);
        assert_eq!(tracker.peak_equity(), dec!(11000.0));
        assert_eq!(tracker.peak_timestamp(), 4000);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let tracker = DrawdownTracker::new(dec!(10000.0), dec!(0.20)).unwrap();

        let json = serde_json::to_string(&tracker).unwrap();
        let deserialized: DrawdownTracker = serde_json::from_str(&json).unwrap();

        assert_eq!(tracker.peak_equity(), deserialized.peak_equity());
        assert_eq!(
            tracker.max_allowed_drawdown(),
            deserialized.max_allowed_drawdown()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_record_serialization() {
        let record = DrawdownRecord {
            drawdown: dec!(0.15),
            timestamp: 12345,
            peak_equity: dec!(10000.0),
            trough_equity: dec!(8500.0),
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: DrawdownRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record, deserialized);
    }
}
