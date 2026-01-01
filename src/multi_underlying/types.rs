//! Multi-underlying types and data structures.

use crate::Decimal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Correlation entry between two underlyings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CorrelationEntry {
    /// First underlying symbol.
    pub underlying_a: String,
    /// Second underlying symbol.
    pub underlying_b: String,
    /// Correlation coefficient (-1 to 1).
    pub correlation: Decimal,
    /// Last update timestamp in milliseconds.
    pub updated_at: u64,
}

impl CorrelationEntry {
    /// Creates a new correlation entry.
    #[must_use]
    pub fn new(
        underlying_a: impl Into<String>,
        underlying_b: impl Into<String>,
        correlation: Decimal,
    ) -> Self {
        Self {
            underlying_a: underlying_a.into(),
            underlying_b: underlying_b.into(),
            correlation,
            updated_at: current_timestamp(),
        }
    }

    /// Returns true if correlation is positive.
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.correlation > Decimal::ZERO
    }

    /// Returns true if correlation is negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.correlation < Decimal::ZERO
    }

    /// Returns the absolute correlation value.
    #[must_use]
    pub fn abs_correlation(&self) -> Decimal {
        if self.correlation < Decimal::ZERO {
            -self.correlation
        } else {
            self.correlation
        }
    }
}

/// State of an underlying in the multi-underlying manager.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnderlyingState {
    /// Underlying symbol.
    pub symbol: String,
    /// Current status.
    pub status: UnderlyingStatus,
    /// Current price.
    pub price: Decimal,
    /// Allocated capital.
    pub allocated_capital: Decimal,
    /// Current position value.
    pub position_value: Decimal,
    /// Unrealized P&L.
    pub unrealized_pnl: Decimal,
    /// Realized P&L.
    pub realized_pnl: Decimal,
    /// Portfolio delta.
    pub delta: Decimal,
    /// Portfolio gamma.
    pub gamma: Decimal,
    /// Portfolio vega.
    pub vega: Decimal,
    /// Last update timestamp in milliseconds.
    pub updated_at: u64,
}

impl UnderlyingState {
    /// Creates a new underlying state.
    #[must_use]
    pub fn new(symbol: impl Into<String>, allocated_capital: Decimal) -> Self {
        Self {
            symbol: symbol.into(),
            status: UnderlyingStatus::Active,
            price: Decimal::ZERO,
            allocated_capital,
            position_value: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            delta: Decimal::ZERO,
            gamma: Decimal::ZERO,
            vega: Decimal::ZERO,
            updated_at: current_timestamp(),
        }
    }

    /// Returns the total P&L.
    #[must_use]
    pub fn total_pnl(&self) -> Decimal {
        self.unrealized_pnl + self.realized_pnl
    }

    /// Returns the capital utilization percentage.
    #[must_use]
    pub fn capital_utilization(&self) -> Decimal {
        if self.allocated_capital > Decimal::ZERO {
            self.position_value / self.allocated_capital * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Returns the dollar delta.
    #[must_use]
    pub fn dollar_delta(&self) -> Decimal {
        self.delta * self.price
    }

    /// Updates the price.
    pub fn update_price(&mut self, price: Decimal) {
        self.price = price;
        self.updated_at = current_timestamp();
    }

    /// Updates the Greeks.
    pub fn update_greeks(&mut self, delta: Decimal, gamma: Decimal, vega: Decimal) {
        self.delta = delta;
        self.gamma = gamma;
        self.vega = vega;
        self.updated_at = current_timestamp();
    }

    /// Updates the P&L.
    pub fn update_pnl(&mut self, unrealized: Decimal, realized: Decimal) {
        self.unrealized_pnl = unrealized;
        self.realized_pnl = realized;
        self.updated_at = current_timestamp();
    }
}

/// Status of an underlying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnderlyingStatus {
    /// Actively trading.
    Active,
    /// Paused (not quoting but tracking).
    Paused,
    /// Halted (circuit breaker triggered).
    Halted,
    /// Disabled (not tracking).
    Disabled,
}

impl Default for UnderlyingStatus {
    fn default() -> Self {
        Self::Active
    }
}

impl std::fmt::Display for UnderlyingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnderlyingStatus::Active => write!(f, "Active"),
            UnderlyingStatus::Paused => write!(f, "Paused"),
            UnderlyingStatus::Halted => write!(f, "Halted"),
            UnderlyingStatus::Disabled => write!(f, "Disabled"),
        }
    }
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
    fn test_correlation_entry() {
        let entry = CorrelationEntry::new("BTC", "ETH", dec!(0.85));
        assert!(entry.is_positive());
        assert!(!entry.is_negative());
        assert_eq!(entry.abs_correlation(), dec!(0.85));
    }

    #[test]
    fn test_correlation_negative() {
        let entry = CorrelationEntry::new("BTC", "GOLD", dec!(-0.3));
        assert!(!entry.is_positive());
        assert!(entry.is_negative());
        assert_eq!(entry.abs_correlation(), dec!(0.3));
    }

    #[test]
    fn test_underlying_state() {
        let mut state = UnderlyingState::new("BTC", dec!(100000.0));
        assert_eq!(state.status, UnderlyingStatus::Active);
        assert_eq!(state.allocated_capital, dec!(100000.0));

        state.update_price(dec!(50000.0));
        state.update_greeks(dec!(10.0), dec!(0.5), dec!(100.0));
        state.update_pnl(dec!(5000.0), dec!(2000.0));

        assert_eq!(state.total_pnl(), dec!(7000.0));
        assert_eq!(state.dollar_delta(), dec!(500000.0));
    }

    #[test]
    fn test_capital_utilization() {
        let mut state = UnderlyingState::new("BTC", dec!(100000.0));
        state.position_value = dec!(50000.0);
        assert_eq!(state.capital_utilization(), dec!(50.0));
    }

    #[test]
    fn test_underlying_status_display() {
        assert_eq!(UnderlyingStatus::Active.to_string(), "Active");
        assert_eq!(UnderlyingStatus::Paused.to_string(), "Paused");
        assert_eq!(UnderlyingStatus::Halted.to_string(), "Halted");
        assert_eq!(UnderlyingStatus::Disabled.to_string(), "Disabled");
    }
}
