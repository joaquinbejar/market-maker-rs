//! Multi-underlying manager implementation.

use std::collections::HashMap;

use crate::Decimal;
use crate::multi_underlying::config::{CapitalAllocationStrategy, UnderlyingConfig};
use crate::multi_underlying::risk::{CrossAssetHedge, UnifiedGreeks, UnifiedRisk};
use crate::multi_underlying::types::{CorrelationEntry, UnderlyingState, UnderlyingStatus};
use crate::types::error::{MMError, MMResult};

/// Manager for multiple underlying assets.
///
/// Coordinates market making across multiple underlyings with:
/// - Cross-asset correlation tracking
/// - Capital allocation strategies
/// - Unified risk view
/// - Per-underlying configuration
#[derive(Debug)]
pub struct MultiUnderlyingManager {
    /// Total capital available.
    total_capital: Decimal,
    /// Allocation strategy.
    allocation_strategy: CapitalAllocationStrategy,
    /// Per-underlying configurations.
    configs: HashMap<String, UnderlyingConfig>,
    /// Per-underlying states.
    states: HashMap<String, UnderlyingState>,
    /// Correlation matrix entries.
    correlations: Vec<CorrelationEntry>,
    /// Maximum number of underlyings.
    max_underlyings: usize,
    /// Unified risk limits.
    max_total_delta: Decimal,
    /// Maximum total vega.
    max_total_vega: Decimal,
    /// Maximum total position value.
    max_total_position_value: Decimal,
}

impl MultiUnderlyingManager {
    /// Creates a new multi-underlying manager.
    #[must_use]
    pub fn new(total_capital: Decimal) -> Self {
        Self {
            total_capital,
            allocation_strategy: CapitalAllocationStrategy::default(),
            configs: HashMap::new(),
            states: HashMap::new(),
            correlations: Vec::new(),
            max_underlyings: 10,
            max_total_delta: Decimal::from(100000),
            max_total_vega: Decimal::from(50000),
            max_total_position_value: Decimal::from(5000000),
        }
    }

    /// Sets the allocation strategy.
    #[must_use]
    pub fn with_allocation_strategy(mut self, strategy: CapitalAllocationStrategy) -> Self {
        self.allocation_strategy = strategy;
        self
    }

    /// Sets the maximum number of underlyings.
    #[must_use]
    pub fn with_max_underlyings(mut self, max: usize) -> Self {
        self.max_underlyings = max;
        self
    }

    /// Sets the maximum total delta.
    #[must_use]
    pub fn with_max_total_delta(mut self, max_delta: Decimal) -> Self {
        self.max_total_delta = max_delta;
        self
    }

    /// Sets the maximum total vega.
    #[must_use]
    pub fn with_max_total_vega(mut self, max_vega: Decimal) -> Self {
        self.max_total_vega = max_vega;
        self
    }

    /// Sets the maximum total position value.
    #[must_use]
    pub fn with_max_total_position_value(mut self, max_value: Decimal) -> Self {
        self.max_total_position_value = max_value;
        self
    }

    /// Adds an underlying to the manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the maximum number of underlyings is reached
    /// or if the underlying already exists.
    pub fn add_underlying(&mut self, config: UnderlyingConfig) -> MMResult<()> {
        if self.configs.len() >= self.max_underlyings {
            return Err(MMError::InvalidConfiguration(format!(
                "maximum underlyings ({}) reached",
                self.max_underlyings
            )));
        }

        if self.configs.contains_key(&config.symbol) {
            return Err(MMError::InvalidConfiguration(format!(
                "underlying {} already exists",
                config.symbol
            )));
        }

        let allocated_capital = self.calculate_allocation(&config);
        let state = UnderlyingState::new(&config.symbol, allocated_capital);

        self.states.insert(config.symbol.clone(), state);
        self.configs.insert(config.symbol.clone(), config);

        Ok(())
    }

    /// Removes an underlying from the manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying does not exist.
    pub fn remove_underlying(&mut self, symbol: &str) -> MMResult<()> {
        if !self.configs.contains_key(symbol) {
            return Err(MMError::InvalidConfiguration(format!(
                "underlying {} not found",
                symbol
            )));
        }

        self.configs.remove(symbol);
        self.states.remove(symbol);
        self.correlations
            .retain(|c| c.underlying_a != symbol && c.underlying_b != symbol);

        Ok(())
    }

    /// Gets the configuration for an underlying.
    #[must_use]
    pub fn get_config(&self, symbol: &str) -> Option<&UnderlyingConfig> {
        self.configs.get(symbol)
    }

    /// Gets the state for an underlying.
    #[must_use]
    pub fn get_state(&self, symbol: &str) -> Option<&UnderlyingState> {
        self.states.get(symbol)
    }

    /// Gets mutable state for an underlying.
    pub fn get_state_mut(&mut self, symbol: &str) -> Option<&mut UnderlyingState> {
        self.states.get_mut(symbol)
    }

    /// Returns all underlying symbols.
    #[must_use]
    pub fn symbols(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    /// Returns the number of underlyings.
    #[must_use]
    pub fn underlying_count(&self) -> usize {
        self.configs.len()
    }

    /// Sets a correlation between two underlyings.
    pub fn set_correlation(
        &mut self,
        underlying_a: &str,
        underlying_b: &str,
        correlation: Decimal,
    ) {
        // Remove existing correlation if present
        self.correlations.retain(|c| {
            !((c.underlying_a == underlying_a && c.underlying_b == underlying_b)
                || (c.underlying_a == underlying_b && c.underlying_b == underlying_a))
        });

        self.correlations.push(CorrelationEntry::new(
            underlying_a,
            underlying_b,
            correlation,
        ));
    }

    /// Gets the correlation between two underlyings.
    #[must_use]
    pub fn get_correlation(&self, underlying_a: &str, underlying_b: &str) -> Option<Decimal> {
        self.correlations
            .iter()
            .find(|c| {
                (c.underlying_a == underlying_a && c.underlying_b == underlying_b)
                    || (c.underlying_a == underlying_b && c.underlying_b == underlying_a)
            })
            .map(|c| c.correlation)
    }

    /// Updates the price for an underlying.
    pub fn update_price(&mut self, symbol: &str, price: Decimal) {
        if let Some(state) = self.states.get_mut(symbol) {
            state.update_price(price);
        }
    }

    /// Updates the Greeks for an underlying.
    pub fn update_greeks(&mut self, symbol: &str, delta: Decimal, gamma: Decimal, vega: Decimal) {
        if let Some(state) = self.states.get_mut(symbol) {
            state.update_greeks(delta, gamma, vega);
        }
    }

    /// Updates the P&L for an underlying.
    pub fn update_pnl(&mut self, symbol: &str, unrealized: Decimal, realized: Decimal) {
        if let Some(state) = self.states.get_mut(symbol) {
            state.update_pnl(unrealized, realized);
        }
    }

    /// Updates the position value for an underlying.
    pub fn update_position_value(&mut self, symbol: &str, position_value: Decimal) {
        if let Some(state) = self.states.get_mut(symbol) {
            state.position_value = position_value;
        }
    }

    /// Sets the status for an underlying.
    pub fn set_status(&mut self, symbol: &str, status: UnderlyingStatus) {
        if let Some(state) = self.states.get_mut(symbol) {
            state.status = status;
        }
    }

    /// Gets the unified Greeks across all underlyings.
    #[must_use]
    pub fn get_unified_greeks(&self) -> UnifiedGreeks {
        let mut greeks = UnifiedGreeks::new();

        for state in self.states.values() {
            if state.status == UnderlyingStatus::Active {
                greeks.total_dollar_delta += state.dollar_delta();
                greeks.total_dollar_gamma += state.gamma * state.price;
                greeks.total_dollar_vega += state.vega;
                greeks.underlying_count += 1;
            }
        }

        // Calculate portfolio volatility using correlation
        greeks.portfolio_volatility = self.calculate_portfolio_volatility();

        greeks
    }

    /// Gets the unified risk view across all underlyings.
    #[must_use]
    pub fn get_unified_risk(&self) -> UnifiedRisk {
        let mut risk = UnifiedRisk::new();
        risk.total_capital = self.total_capital;
        risk.greeks = self.get_unified_greeks();

        for state in self.states.values() {
            risk.total_position_value += state.position_value;
            risk.total_unrealized_pnl += state.unrealized_pnl;
            risk.total_realized_pnl += state.realized_pnl;

            match state.status {
                UnderlyingStatus::Active => risk.active_underlyings += 1,
                UnderlyingStatus::Halted => risk.halted_underlyings += 1,
                _ => {}
            }
        }

        // Calculate utilization percentages
        if self.max_total_delta > Decimal::ZERO {
            risk.delta_utilization =
                risk.greeks.abs_dollar_delta() / self.max_total_delta * Decimal::from(100);
        }

        if self.max_total_vega > Decimal::ZERO {
            let abs_vega = if risk.greeks.total_dollar_vega < Decimal::ZERO {
                -risk.greeks.total_dollar_vega
            } else {
                risk.greeks.total_dollar_vega
            };
            risk.vega_utilization = abs_vega / self.max_total_vega * Decimal::from(100);
        }

        risk
    }

    /// Gets cross-asset hedge suggestions.
    #[must_use]
    pub fn get_cross_asset_hedges(&self) -> Vec<CrossAssetHedge> {
        let mut hedges = Vec::new();
        let states: Vec<_> = self.states.iter().collect();

        let mut rest = states.as_slice();
        while let Some((&(symbol_a, state_a), tail)) = rest.split_first() {
            rest = tail;
            for &(symbol_b, state_b) in tail {
                if let Some(correlation) = self.get_correlation(symbol_a, symbol_b) {
                    // Only suggest hedges for active underlyings with significant delta
                    if state_a.status != UnderlyingStatus::Active
                        || state_b.status != UnderlyingStatus::Active
                    {
                        continue;
                    }

                    let delta_a = state_a.dollar_delta();
                    let _delta_b = state_b.dollar_delta();

                    // Skip if delta is too small
                    if delta_a.abs() < Decimal::from(1000) {
                        continue;
                    }

                    // Calculate hedge ratio based on correlation and prices
                    let hedge_ratio = if state_b.price > Decimal::ZERO {
                        (state_a.price / state_b.price) * correlation.abs()
                    } else {
                        Decimal::ZERO
                    };

                    // Estimate risk reduction
                    let risk_reduction = correlation.abs() * Decimal::from(100);

                    if risk_reduction > Decimal::from(10) {
                        hedges.push(CrossAssetHedge::new(
                            symbol_a.clone(),
                            symbol_b.clone(),
                            hedge_ratio,
                            risk_reduction,
                            correlation,
                        ));
                    }
                }
            }
        }

        hedges
    }

    /// Reallocates capital based on the current strategy.
    pub fn reallocate_capital(&mut self) {
        match self.allocation_strategy {
            CapitalAllocationStrategy::Equal => self.allocate_equal(),
            CapitalAllocationStrategy::TargetWeight => self.allocate_by_weight(),
            CapitalAllocationStrategy::RiskParity => self.allocate_risk_parity(),
            CapitalAllocationStrategy::VolatilityWeighted => self.allocate_volatility_weighted(),
            CapitalAllocationStrategy::PerformanceBased => self.allocate_performance_based(),
        }
    }

    /// Returns the total capital.
    #[must_use]
    pub fn total_capital(&self) -> Decimal {
        self.total_capital
    }

    /// Sets the total capital.
    pub fn set_total_capital(&mut self, capital: Decimal) {
        self.total_capital = capital;
    }

    /// Returns the allocation strategy.
    #[must_use]
    pub fn allocation_strategy(&self) -> CapitalAllocationStrategy {
        self.allocation_strategy
    }

    /// Sets the allocation strategy.
    pub fn set_allocation_strategy(&mut self, strategy: CapitalAllocationStrategy) {
        self.allocation_strategy = strategy;
    }

    // Private helper methods

    fn calculate_allocation(&self, config: &UnderlyingConfig) -> Decimal {
        match self.allocation_strategy {
            CapitalAllocationStrategy::Equal => {
                let count = self.configs.len() + 1;
                self.total_capital / Decimal::from(count)
            }
            CapitalAllocationStrategy::TargetWeight => self.total_capital * config.target_weight,
            _ => self.total_capital * config.target_weight,
        }
    }

    fn allocate_equal(&mut self) {
        let count = self.states.len();
        if count == 0 {
            return;
        }

        let allocation = self.total_capital / Decimal::from(count);
        for state in self.states.values_mut() {
            state.allocated_capital = allocation;
        }
    }

    fn allocate_by_weight(&mut self) {
        for (symbol, config) in &self.configs {
            if let Some(state) = self.states.get_mut(symbol) {
                state.allocated_capital = self.total_capital * config.target_weight;
            }
        }
    }

    fn allocate_risk_parity(&mut self) {
        // Simplified risk parity: allocate inversely proportional to volatility
        // In a full implementation, this would use the correlation matrix
        let total_inverse_vol: Decimal = self
            .states
            .values()
            .filter(|s| s.status == UnderlyingStatus::Active)
            .map(|s| {
                if s.vega > Decimal::ZERO {
                    Decimal::ONE / s.vega
                } else {
                    Decimal::ONE
                }
            })
            .sum();

        if total_inverse_vol > Decimal::ZERO {
            for state in self.states.values_mut() {
                if state.status == UnderlyingStatus::Active {
                    let inverse_vol = if state.vega > Decimal::ZERO {
                        Decimal::ONE / state.vega
                    } else {
                        Decimal::ONE
                    };
                    state.allocated_capital = self.total_capital * inverse_vol / total_inverse_vol;
                }
            }
        }
    }

    fn allocate_volatility_weighted(&mut self) {
        // Allocate proportionally to volatility (vega as proxy)
        let total_vol: Decimal = self
            .states
            .values()
            .filter(|s| s.status == UnderlyingStatus::Active)
            .map(|s| s.vega.abs())
            .sum();

        if total_vol > Decimal::ZERO {
            for state in self.states.values_mut() {
                if state.status == UnderlyingStatus::Active {
                    state.allocated_capital = self.total_capital * state.vega.abs() / total_vol;
                }
            }
        }
    }

    fn allocate_performance_based(&mut self) {
        // Allocate more to underlyings with positive P&L
        let total_pnl: Decimal = self
            .states
            .values()
            .filter(|s| s.status == UnderlyingStatus::Active && s.total_pnl() > Decimal::ZERO)
            .map(|s| s.total_pnl())
            .sum();

        if total_pnl > Decimal::ZERO {
            for state in self.states.values_mut() {
                if state.status == UnderlyingStatus::Active && state.total_pnl() > Decimal::ZERO {
                    state.allocated_capital = self.total_capital * state.total_pnl() / total_pnl;
                }
            }
        } else {
            // Fall back to equal allocation if no positive P&L
            self.allocate_equal();
        }
    }

    fn calculate_portfolio_volatility(&self) -> Decimal {
        // Simplified portfolio volatility calculation
        // In a full implementation, this would use the full correlation matrix
        let mut total_variance = Decimal::ZERO;

        for state in self.states.values() {
            if state.status == UnderlyingStatus::Active {
                // Add individual variance (vega squared as proxy)
                total_variance += state.vega * state.vega;
            }
        }

        // Add covariance terms
        let states: Vec<_> = self.states.iter().collect();
        let mut rest = states.as_slice();
        while let Some((&(symbol_a, state_a), tail)) = rest.split_first() {
            rest = tail;
            for &(symbol_b, state_b) in tail {
                if let Some(correlation) = self.get_correlation(symbol_a, symbol_b)
                    && state_a.status == UnderlyingStatus::Active
                    && state_b.status == UnderlyingStatus::Active
                {
                    // 2 * correlation * vega_a * vega_b
                    total_variance += Decimal::from(2) * correlation * state_a.vega * state_b.vega;
                }
            }
        }

        // Return sqrt approximation (simplified)
        if total_variance > Decimal::ZERO {
            // Newton-Raphson approximation for sqrt
            let mut x = total_variance / Decimal::from(2);
            for _ in 0..10 {
                if x > Decimal::ZERO {
                    x = (x + total_variance / x) / Decimal::from(2);
                }
            }
            x
        } else {
            Decimal::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_manager_new() {
        let manager = MultiUnderlyingManager::new(dec!(1000000.0));
        assert_eq!(manager.total_capital(), dec!(1000000.0));
        assert_eq!(manager.underlying_count(), 0);
    }

    #[test]
    fn test_add_underlying() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        let config = UnderlyingConfig::new("BTC", dec!(0.4));

        manager.add_underlying(config).unwrap();
        assert_eq!(manager.underlying_count(), 1);
        assert!(manager.get_config("BTC").is_some());
        assert!(manager.get_state("BTC").is_some());
    }

    #[test]
    fn test_add_duplicate_underlying() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        let config1 = UnderlyingConfig::new("BTC", dec!(0.4));
        let config2 = UnderlyingConfig::new("BTC", dec!(0.3));

        manager.add_underlying(config1).unwrap();
        let result = manager.add_underlying(config2);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_underlying() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();
        manager
            .add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)))
            .unwrap();

        manager.remove_underlying("BTC").unwrap();
        assert_eq!(manager.underlying_count(), 1);
        assert!(manager.get_config("BTC").is_none());
    }

    #[test]
    fn test_correlation() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();
        manager
            .add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)))
            .unwrap();

        manager.set_correlation("BTC", "ETH", dec!(0.85));
        assert_eq!(manager.get_correlation("BTC", "ETH"), Some(dec!(0.85)));
        assert_eq!(manager.get_correlation("ETH", "BTC"), Some(dec!(0.85)));
    }

    #[test]
    fn test_update_price() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();

        manager.update_price("BTC", dec!(50000.0));
        let state = manager.get_state("BTC").unwrap();
        assert_eq!(state.price, dec!(50000.0));
    }

    #[test]
    fn test_update_greeks() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();

        manager.update_greeks("BTC", dec!(10.0), dec!(0.5), dec!(100.0));
        let state = manager.get_state("BTC").unwrap();
        assert_eq!(state.delta, dec!(10.0));
        assert_eq!(state.gamma, dec!(0.5));
        assert_eq!(state.vega, dec!(100.0));
    }

    #[test]
    fn test_unified_greeks() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();
        manager
            .add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)))
            .unwrap();

        manager.update_price("BTC", dec!(50000.0));
        manager.update_price("ETH", dec!(3000.0));
        manager.update_greeks("BTC", dec!(10.0), dec!(0.5), dec!(100.0));
        manager.update_greeks("ETH", dec!(20.0), dec!(1.0), dec!(200.0));

        let greeks = manager.get_unified_greeks();
        assert_eq!(greeks.underlying_count, 2);
        // BTC: 10 * 50000 = 500000, ETH: 20 * 3000 = 60000
        assert_eq!(greeks.total_dollar_delta, dec!(560000.0));
    }

    #[test]
    fn test_unified_risk() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();

        manager.update_pnl("BTC", dec!(5000.0), dec!(2000.0));
        manager.update_position_value("BTC", dec!(100000.0));

        let risk = manager.get_unified_risk();
        assert_eq!(risk.total_capital, dec!(1000000.0));
        assert_eq!(risk.total_unrealized_pnl, dec!(5000.0));
        assert_eq!(risk.total_realized_pnl, dec!(2000.0));
        assert_eq!(risk.total_pnl(), dec!(7000.0));
    }

    #[test]
    fn test_reallocate_equal() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0))
            .with_allocation_strategy(CapitalAllocationStrategy::Equal);

        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();
        manager
            .add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)))
            .unwrap();

        manager.reallocate_capital();

        let btc_state = manager.get_state("BTC").unwrap();
        let eth_state = manager.get_state("ETH").unwrap();
        assert_eq!(btc_state.allocated_capital, dec!(500000.0));
        assert_eq!(eth_state.allocated_capital, dec!(500000.0));
    }

    #[test]
    fn test_cross_asset_hedges() {
        let mut manager = MultiUnderlyingManager::new(dec!(1000000.0));
        manager
            .add_underlying(UnderlyingConfig::new("BTC", dec!(0.4)))
            .unwrap();
        manager
            .add_underlying(UnderlyingConfig::new("ETH", dec!(0.3)))
            .unwrap();

        manager.set_correlation("BTC", "ETH", dec!(0.85));
        manager.update_price("BTC", dec!(50000.0));
        manager.update_price("ETH", dec!(3000.0));
        manager.update_greeks("BTC", dec!(10.0), dec!(0.5), dec!(100.0));
        manager.update_greeks("ETH", dec!(5.0), dec!(0.3), dec!(50.0));

        let hedges = manager.get_cross_asset_hedges();
        assert!(!hedges.is_empty());
    }
}
