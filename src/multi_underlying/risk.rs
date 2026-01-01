//! Multi-underlying risk types.

use crate::Decimal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unified Greeks across all underlyings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnifiedGreeks {
    /// Total dollar delta across all underlyings.
    pub total_dollar_delta: Decimal,
    /// Total dollar gamma across all underlyings.
    pub total_dollar_gamma: Decimal,
    /// Total dollar vega across all underlyings.
    pub total_dollar_vega: Decimal,
    /// Total dollar theta across all underlyings.
    pub total_dollar_theta: Decimal,
    /// Portfolio volatility (correlation-adjusted).
    pub portfolio_volatility: Decimal,
    /// Number of underlyings.
    pub underlying_count: usize,
}

impl Default for UnifiedGreeks {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedGreeks {
    /// Creates a new unified Greeks instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_dollar_delta: Decimal::ZERO,
            total_dollar_gamma: Decimal::ZERO,
            total_dollar_vega: Decimal::ZERO,
            total_dollar_theta: Decimal::ZERO,
            portfolio_volatility: Decimal::ZERO,
            underlying_count: 0,
        }
    }

    /// Returns the absolute dollar delta.
    #[must_use]
    pub fn abs_dollar_delta(&self) -> Decimal {
        if self.total_dollar_delta < Decimal::ZERO {
            -self.total_dollar_delta
        } else {
            self.total_dollar_delta
        }
    }
}

/// Unified risk view across all underlyings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnifiedRisk {
    /// Total capital allocated.
    pub total_capital: Decimal,
    /// Total position value.
    pub total_position_value: Decimal,
    /// Total unrealized P&L.
    pub total_unrealized_pnl: Decimal,
    /// Total realized P&L.
    pub total_realized_pnl: Decimal,
    /// Portfolio-level delta utilization.
    pub delta_utilization: Decimal,
    /// Portfolio-level gamma utilization.
    pub gamma_utilization: Decimal,
    /// Portfolio-level vega utilization.
    pub vega_utilization: Decimal,
    /// Number of active underlyings.
    pub active_underlyings: usize,
    /// Number of halted underlyings.
    pub halted_underlyings: usize,
    /// Unified Greeks.
    pub greeks: UnifiedGreeks,
}

impl Default for UnifiedRisk {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedRisk {
    /// Creates a new unified risk instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_capital: Decimal::ZERO,
            total_position_value: Decimal::ZERO,
            total_unrealized_pnl: Decimal::ZERO,
            total_realized_pnl: Decimal::ZERO,
            delta_utilization: Decimal::ZERO,
            gamma_utilization: Decimal::ZERO,
            vega_utilization: Decimal::ZERO,
            active_underlyings: 0,
            halted_underlyings: 0,
            greeks: UnifiedGreeks::new(),
        }
    }

    /// Returns the total P&L.
    #[must_use]
    pub fn total_pnl(&self) -> Decimal {
        self.total_unrealized_pnl + self.total_realized_pnl
    }

    /// Returns the capital utilization percentage.
    #[must_use]
    pub fn capital_utilization(&self) -> Decimal {
        if self.total_capital > Decimal::ZERO {
            self.total_position_value / self.total_capital * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Returns true if any risk limit is breached.
    #[must_use]
    pub fn is_risk_breached(&self) -> bool {
        self.delta_utilization > Decimal::from(100)
            || self.gamma_utilization > Decimal::from(100)
            || self.vega_utilization > Decimal::from(100)
    }
}

/// Cross-asset hedge suggestion.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CrossAssetHedge {
    /// Source underlying to hedge.
    pub source_underlying: String,
    /// Underlying to use for hedging.
    pub hedge_underlying: String,
    /// Hedge ratio (units of hedge per unit of source).
    pub hedge_ratio: Decimal,
    /// Expected risk reduction percentage.
    pub risk_reduction: Decimal,
    /// Correlation between the two underlyings.
    pub correlation: Decimal,
    /// Hedge type.
    pub hedge_type: HedgeType,
}

impl CrossAssetHedge {
    /// Creates a new cross-asset hedge suggestion.
    #[must_use]
    pub fn new(
        source_underlying: impl Into<String>,
        hedge_underlying: impl Into<String>,
        hedge_ratio: Decimal,
        risk_reduction: Decimal,
        correlation: Decimal,
    ) -> Self {
        let hedge_type = if correlation > Decimal::ZERO {
            HedgeType::Opposite
        } else {
            HedgeType::Same
        };

        Self {
            source_underlying: source_underlying.into(),
            hedge_underlying: hedge_underlying.into(),
            hedge_ratio,
            risk_reduction,
            correlation,
            hedge_type,
        }
    }

    /// Returns true if this is an effective hedge (>10% risk reduction).
    #[must_use]
    pub fn is_effective(&self) -> bool {
        self.risk_reduction > Decimal::from(10)
    }
}

/// Type of cross-asset hedge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HedgeType {
    /// Take opposite position (for positive correlation).
    Opposite,
    /// Take same direction position (for negative correlation).
    Same,
}

impl std::fmt::Display for HedgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HedgeType::Opposite => write!(f, "Opposite"),
            HedgeType::Same => write!(f, "Same"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_unified_greeks() {
        let mut greeks = UnifiedGreeks::new();
        greeks.total_dollar_delta = dec!(-50000.0);
        greeks.total_dollar_gamma = dec!(10000.0);

        assert_eq!(greeks.abs_dollar_delta(), dec!(50000.0));
    }

    #[test]
    fn test_unified_risk() {
        let mut risk = UnifiedRisk::new();
        risk.total_capital = dec!(1000000.0);
        risk.total_position_value = dec!(500000.0);
        risk.total_unrealized_pnl = dec!(10000.0);
        risk.total_realized_pnl = dec!(5000.0);

        assert_eq!(risk.total_pnl(), dec!(15000.0));
        assert_eq!(risk.capital_utilization(), dec!(50.0));
    }

    #[test]
    fn test_unified_risk_breached() {
        let mut risk = UnifiedRisk::new();
        risk.delta_utilization = dec!(110.0);
        assert!(risk.is_risk_breached());

        risk.delta_utilization = dec!(90.0);
        assert!(!risk.is_risk_breached());
    }

    #[test]
    fn test_cross_asset_hedge() {
        let hedge = CrossAssetHedge::new("BTC", "ETH", dec!(0.5), dec!(25.0), dec!(0.85));
        assert_eq!(hedge.hedge_type, HedgeType::Opposite);
        assert!(hedge.is_effective());
    }

    #[test]
    fn test_cross_asset_hedge_negative_correlation() {
        let hedge = CrossAssetHedge::new("BTC", "GOLD", dec!(0.3), dec!(15.0), dec!(-0.3));
        assert_eq!(hedge.hedge_type, HedgeType::Same);
    }
}
