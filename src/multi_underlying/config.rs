//! Multi-underlying configuration.

use crate::Decimal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for a single underlying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnderlyingConfig {
    /// Underlying symbol.
    pub symbol: String,
    /// Target capital allocation weight (0.0 to 1.0).
    pub target_weight: Decimal,
    /// Maximum delta limit.
    pub max_delta: Decimal,
    /// Maximum gamma limit.
    pub max_gamma: Decimal,
    /// Maximum vega limit.
    pub max_vega: Decimal,
    /// Maximum position value.
    pub max_position_value: Decimal,
    /// Whether quoting is enabled.
    pub quoting_enabled: bool,
    /// Base spread in basis points.
    pub base_spread_bps: u32,
}

impl UnderlyingConfig {
    /// Creates a new underlying configuration with default limits.
    #[must_use]
    pub fn new(symbol: impl Into<String>, target_weight: Decimal) -> Self {
        Self {
            symbol: symbol.into(),
            target_weight,
            max_delta: Decimal::from(100),
            max_gamma: Decimal::from(50),
            max_vega: Decimal::from(1000),
            max_position_value: Decimal::from(100000),
            quoting_enabled: true,
            base_spread_bps: 100,
        }
    }

    /// Sets the delta limit.
    #[must_use]
    pub fn with_max_delta(mut self, max_delta: Decimal) -> Self {
        self.max_delta = max_delta;
        self
    }

    /// Sets the gamma limit.
    #[must_use]
    pub fn with_max_gamma(mut self, max_gamma: Decimal) -> Self {
        self.max_gamma = max_gamma;
        self
    }

    /// Sets the vega limit.
    #[must_use]
    pub fn with_max_vega(mut self, max_vega: Decimal) -> Self {
        self.max_vega = max_vega;
        self
    }

    /// Sets the maximum position value.
    #[must_use]
    pub fn with_max_position_value(mut self, max_position_value: Decimal) -> Self {
        self.max_position_value = max_position_value;
        self
    }

    /// Sets the base spread.
    #[must_use]
    pub fn with_base_spread_bps(mut self, base_spread_bps: u32) -> Self {
        self.base_spread_bps = base_spread_bps;
        self
    }

    /// Enables or disables quoting.
    #[must_use]
    pub fn with_quoting_enabled(mut self, enabled: bool) -> Self {
        self.quoting_enabled = enabled;
        self
    }
}

/// Capital allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CapitalAllocationStrategy {
    /// Equal allocation across all underlyings.
    Equal,
    /// Allocation based on target weights.
    TargetWeight,
    /// Risk parity (equal risk contribution).
    RiskParity,
    /// Volatility-weighted allocation.
    VolatilityWeighted,
    /// Performance-based (allocate more to winners).
    PerformanceBased,
}

impl Default for CapitalAllocationStrategy {
    fn default() -> Self {
        Self::TargetWeight
    }
}

impl std::fmt::Display for CapitalAllocationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapitalAllocationStrategy::Equal => write!(f, "Equal"),
            CapitalAllocationStrategy::TargetWeight => write!(f, "TargetWeight"),
            CapitalAllocationStrategy::RiskParity => write!(f, "RiskParity"),
            CapitalAllocationStrategy::VolatilityWeighted => write!(f, "VolatilityWeighted"),
            CapitalAllocationStrategy::PerformanceBased => write!(f, "PerformanceBased"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_underlying_config_new() {
        let config = UnderlyingConfig::new("BTC", dec!(0.4));
        assert_eq!(config.symbol, "BTC");
        assert_eq!(config.target_weight, dec!(0.4));
        assert!(config.quoting_enabled);
    }

    #[test]
    fn test_underlying_config_builder() {
        let config = UnderlyingConfig::new("ETH", dec!(0.3))
            .with_max_delta(dec!(50.0))
            .with_max_gamma(dec!(25.0))
            .with_base_spread_bps(150)
            .with_quoting_enabled(false);

        assert_eq!(config.max_delta, dec!(50.0));
        assert_eq!(config.max_gamma, dec!(25.0));
        assert_eq!(config.base_spread_bps, 150);
        assert!(!config.quoting_enabled);
    }

    #[test]
    fn test_capital_allocation_strategy_display() {
        assert_eq!(CapitalAllocationStrategy::Equal.to_string(), "Equal");
        assert_eq!(CapitalAllocationStrategy::RiskParity.to_string(), "RiskParity");
    }
}
