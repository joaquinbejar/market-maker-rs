//! Greeks-based risk management for options market making.
//!
//! This module provides comprehensive risk management functionality including:
//! - Greeks-based order validation
//! - Circuit breakers for limit breaches
//! - Auto-hedging with configurable triggers
//! - Limit utilization tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::options::{GreeksRiskManager, AutoHedgerConfig};
//! use market_maker_rs::options::GreeksLimits;
//!
//! let limits = GreeksLimits::default();
//! let hedger_config = AutoHedgerConfig::default();
//! let mut risk_manager = GreeksRiskManager::new(limits, hedger_config);
//!
//! // Check order
//! let decision = risk_manager.check_order(&option_greeks, quantity);
//! ```

// TODO(#166): legacy pricing arithmetic, converted in #166
#![allow(clippy::arithmetic_side_effects)]

use crate::Decimal;
use crate::options::greeks::{PortfolioGreeks, PositionGreeks};
use crate::options::market_maker::{GreeksLimits, HedgeOrder, HedgeType};
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Order decision from risk check.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OrderDecision {
    /// Order is allowed.
    Allowed,
    /// Order is allowed but scaled down.
    Scaled {
        /// Original requested size.
        original_size: Decimal,
        /// New scaled size.
        new_size: Decimal,
        /// Reason for scaling.
        reason: String,
    },
    /// Order is rejected.
    Rejected {
        /// Reason for rejection.
        reason: String,
    },
}

impl OrderDecision {
    /// Returns true if the order is allowed (possibly scaled).
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        !matches!(self, OrderDecision::Rejected { .. })
    }

    /// Returns the effective size (0 if rejected).
    #[must_use]
    pub fn effective_size(&self, original: Decimal) -> Decimal {
        match self {
            OrderDecision::Allowed => original,
            OrderDecision::Scaled { new_size, .. } => *new_size,
            OrderDecision::Rejected { .. } => Decimal::ZERO,
        }
    }
}

/// Circuit breaker state for Greeks limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GreeksCircuitBreakerState {
    /// Circuit breaker is closed (normal operation).
    Closed,
    /// Circuit breaker is open (trading halted).
    Open,
    /// Circuit breaker is in cooldown.
    Cooldown,
}

impl GreeksCircuitBreakerState {
    /// Returns true if trading is allowed.
    #[must_use]
    pub fn allows_trading(&self) -> bool {
        matches!(self, GreeksCircuitBreakerState::Closed)
    }
}

/// Circuit breaker status with details.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GreeksCircuitBreakerStatus {
    /// Current state.
    pub state: GreeksCircuitBreakerState,
    /// Reason for current state.
    pub reason: Option<String>,
    /// Timestamp when state changed (milliseconds).
    pub state_changed_at: u64,
    /// Timestamp when cooldown ends (if in cooldown).
    pub cooldown_ends_at: Option<u64>,
}

impl GreeksCircuitBreakerStatus {
    /// Creates a new closed status.
    #[must_use]
    pub fn closed() -> Self {
        Self {
            state: GreeksCircuitBreakerState::Closed,
            reason: None,
            state_changed_at: 0,
            cooldown_ends_at: None,
        }
    }

    /// Creates a new open status.
    #[must_use]
    pub fn open(reason: String, timestamp: u64) -> Self {
        Self {
            state: GreeksCircuitBreakerState::Open,
            reason: Some(reason),
            state_changed_at: timestamp,
            cooldown_ends_at: None,
        }
    }
}

/// Configuration for Greeks circuit breaker.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GreeksCircuitBreakerConfig {
    /// Trip when delta exceeds this percentage of limit.
    pub delta_trip_pct: Decimal,
    /// Trip when gamma exceeds this percentage of limit.
    pub gamma_trip_pct: Decimal,
    /// Trip when vega exceeds this percentage of limit.
    pub vega_trip_pct: Decimal,
    /// Cooldown period in milliseconds.
    pub cooldown_ms: u64,
    /// Auto-reset when Greeks return below this percentage.
    pub reset_pct: Decimal,
}

impl Default for GreeksCircuitBreakerConfig {
    fn default() -> Self {
        Self {
            delta_trip_pct: dec!(0.95),
            gamma_trip_pct: dec!(0.95),
            vega_trip_pct: dec!(0.95),
            cooldown_ms: 60_000, // 1 minute
            reset_pct: dec!(0.80),
        }
    }
}

/// Greeks circuit breaker.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GreeksCircuitBreaker {
    /// Configuration.
    config: GreeksCircuitBreakerConfig,
    /// Current status.
    status: GreeksCircuitBreakerStatus,
}

impl GreeksCircuitBreaker {
    /// Creates a new circuit breaker.
    #[must_use]
    pub fn new(config: GreeksCircuitBreakerConfig) -> Self {
        Self {
            config,
            status: GreeksCircuitBreakerStatus::closed(),
        }
    }

    /// Creates a new circuit breaker with default config.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(GreeksCircuitBreakerConfig::default())
    }

    /// Returns the current status.
    #[must_use]
    pub fn status(&self) -> &GreeksCircuitBreakerStatus {
        &self.status
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &GreeksCircuitBreakerConfig {
        &self.config
    }

    /// Checks if the circuit breaker should trip.
    pub fn check(&mut self, greeks: &PortfolioGreeks, limits: &GreeksLimits, timestamp: u64) {
        // Check cooldown
        if self.status.state == GreeksCircuitBreakerState::Cooldown
            && let Some(ends_at) = self.status.cooldown_ends_at
            && timestamp >= ends_at
        {
            self.status = GreeksCircuitBreakerStatus::closed();
        }

        // Check if should trip
        if self.status.state == GreeksCircuitBreakerState::Closed {
            let delta_util = greeks.delta.abs() / limits.max_delta;
            let gamma_util = greeks.gamma.abs() / limits.max_gamma;
            let vega_util = greeks.vega.abs() / limits.max_vega;

            if delta_util >= self.config.delta_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Delta utilization {:.1}% exceeds trip threshold",
                        delta_util * dec!(100.0)
                    ),
                    timestamp,
                );
            } else if gamma_util >= self.config.gamma_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Gamma utilization {:.1}% exceeds trip threshold",
                        gamma_util * dec!(100.0)
                    ),
                    timestamp,
                );
            } else if vega_util >= self.config.vega_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Vega utilization {:.1}% exceeds trip threshold",
                        vega_util * dec!(100.0)
                    ),
                    timestamp,
                );
            }
        }

        // Check if should reset
        if self.status.state == GreeksCircuitBreakerState::Open {
            let delta_util = greeks.delta.abs() / limits.max_delta;
            let gamma_util = greeks.gamma.abs() / limits.max_gamma;
            let vega_util = greeks.vega.abs() / limits.max_vega;

            if delta_util < self.config.reset_pct
                && gamma_util < self.config.reset_pct
                && vega_util < self.config.reset_pct
            {
                self.status = GreeksCircuitBreakerStatus {
                    state: GreeksCircuitBreakerState::Cooldown,
                    reason: Some("Greeks returned to safe levels".to_string()),
                    state_changed_at: timestamp,
                    cooldown_ends_at: Some(timestamp + self.config.cooldown_ms),
                };
            }
        }
    }

    /// Manually resets the circuit breaker.
    pub fn reset(&mut self) {
        self.status = GreeksCircuitBreakerStatus::closed();
    }

    /// Returns true if trading is allowed.
    #[must_use]
    pub fn allows_trading(&self) -> bool {
        self.status.state.allows_trading()
    }
}

/// Auto-hedging configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AutoHedgerConfig {
    /// Enable auto-hedging.
    pub enabled: bool,
    /// Delta threshold to trigger hedge (as percentage of limit).
    pub trigger_threshold_pct: Decimal,
    /// Target delta after hedge (as percentage of limit).
    pub target_delta_pct: Decimal,
    /// Minimum hedge size.
    pub min_hedge_size: Decimal,
    /// Maximum hedge size per order.
    pub max_hedge_size: Decimal,
    /// Contract multiplier.
    pub contract_multiplier: Decimal,
}

impl Default for AutoHedgerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_threshold_pct: dec!(0.8),
            target_delta_pct: dec!(0.5),
            min_hedge_size: dec!(1.0),
            max_hedge_size: dec!(100.0),
            contract_multiplier: dec!(100.0),
        }
    }
}

/// Hedge urgency level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HedgeUrgency {
    /// Normal hedging - can wait for good execution.
    Normal,
    /// Urgent hedging - should execute soon.
    Urgent,
    /// Emergency hedging - execute immediately.
    Emergency,
}

/// Auto-hedger for delta neutralization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AutoHedger {
    /// Configuration.
    config: AutoHedgerConfig,
    /// Underlying symbol.
    underlying_symbol: String,
}

impl AutoHedger {
    /// Creates a new auto-hedger.
    #[must_use]
    pub fn new(underlying_symbol: impl Into<String>, config: AutoHedgerConfig) -> Self {
        Self {
            config,
            underlying_symbol: underlying_symbol.into(),
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &AutoHedgerConfig {
        &self.config
    }

    /// Checks if hedging is needed.
    #[must_use]
    pub fn needs_hedge(&self, greeks: &PortfolioGreeks, limits: &GreeksLimits) -> bool {
        if !self.config.enabled {
            return false;
        }

        let delta_util = greeks.delta.abs() / limits.max_delta;
        delta_util >= self.config.trigger_threshold_pct
    }

    /// Calculates the hedge urgency.
    #[must_use]
    pub fn hedge_urgency(&self, greeks: &PortfolioGreeks, limits: &GreeksLimits) -> HedgeUrgency {
        let delta_util = greeks.delta.abs() / limits.max_delta;

        if delta_util >= dec!(0.95) {
            HedgeUrgency::Emergency
        } else if delta_util >= dec!(0.9) {
            HedgeUrgency::Urgent
        } else {
            HedgeUrgency::Normal
        }
    }

    /// Calculates the hedge order.
    #[must_use]
    pub fn calculate_hedge(
        &self,
        greeks: &PortfolioGreeks,
        limits: &GreeksLimits,
        underlying_price: Decimal,
    ) -> Option<HedgeOrder> {
        if !self.needs_hedge(greeks, limits) {
            return None;
        }

        // Calculate target delta
        let target_delta = if greeks.delta > Decimal::ZERO {
            limits.max_delta * self.config.target_delta_pct
        } else {
            -limits.max_delta * self.config.target_delta_pct
        };

        // Calculate delta to hedge
        let delta_to_hedge = greeks.delta - target_delta;

        // Convert to shares
        let shares = -delta_to_hedge * self.config.contract_multiplier;

        // Clamp to min/max
        let clamped_shares = if shares.abs() < self.config.min_hedge_size {
            return None; // Too small to hedge
        } else if shares.abs() > self.config.max_hedge_size {
            if shares > Decimal::ZERO {
                self.config.max_hedge_size
            } else {
                -self.config.max_hedge_size
            }
        } else {
            shares
        };

        Some(HedgeOrder::new(
            self.underlying_symbol.clone(),
            clamped_shares,
            underlying_price,
            HedgeType::Underlying,
            -clamped_shares / self.config.contract_multiplier,
        ))
    }
}

/// Limit utilization tracking.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LimitUtilization {
    /// Delta utilization (0.0 to 1.0+).
    pub delta: Decimal,
    /// Gamma utilization (0.0 to 1.0+).
    pub gamma: Decimal,
    /// Vega utilization (0.0 to 1.0+).
    pub vega: Decimal,
    /// Theta utilization (0.0 to 1.0+).
    pub theta: Decimal,
    /// Maximum utilization across all Greeks.
    pub max_utilization: Decimal,
}

impl LimitUtilization {
    /// Calculates utilization from Greeks and limits.
    #[must_use]
    pub fn calculate(greeks: &PortfolioGreeks, limits: &GreeksLimits) -> Self {
        let delta = greeks.delta.abs() / limits.max_delta;
        let gamma = greeks.gamma.abs() / limits.max_gamma;
        let vega = greeks.vega.abs() / limits.max_vega;
        let theta = if limits.max_theta < Decimal::ZERO {
            greeks.theta / limits.max_theta
        } else {
            Decimal::ZERO
        };

        let max_utilization = delta.max(gamma).max(vega).max(theta);

        Self {
            delta,
            gamma,
            vega,
            theta,
            max_utilization,
        }
    }

    /// Returns true if any limit is breached.
    #[must_use]
    pub fn is_breached(&self) -> bool {
        self.max_utilization > dec!(1.0)
    }

    /// Returns true if approaching limits (>80%).
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.max_utilization > dec!(0.8)
    }

    /// Returns the scaling factor for order sizing.
    #[must_use]
    pub fn scaling_factor(&self) -> Decimal {
        if self.max_utilization >= dec!(1.0) {
            Decimal::ZERO
        } else if self.max_utilization >= dec!(0.8) {
            // Linear scaling from 1.0 at 80% to 0.0 at 100%
            (dec!(1.0) - self.max_utilization) / dec!(0.2)
        } else {
            dec!(1.0)
        }
    }
}

/// Greeks-based risk manager.
///
/// Provides comprehensive risk management for options market making.
#[derive(Debug, Clone)]
pub struct GreeksRiskManager {
    /// Greeks limits.
    limits: GreeksLimits,
    /// Current portfolio Greeks.
    current_greeks: PortfolioGreeks,
    /// Circuit breaker.
    circuit_breaker: GreeksCircuitBreaker,
    /// Auto-hedger.
    hedger: AutoHedger,
    /// Underlying symbol.
    underlying_symbol: String,
}

impl GreeksRiskManager {
    /// Creates a new risk manager.
    #[must_use]
    pub fn new(
        underlying_symbol: impl Into<String>,
        limits: GreeksLimits,
        hedger_config: AutoHedgerConfig,
    ) -> Self {
        let underlying = underlying_symbol.into();
        Self {
            limits,
            current_greeks: PortfolioGreeks::new(),
            circuit_breaker: GreeksCircuitBreaker::with_defaults(),
            hedger: AutoHedger::new(&underlying, hedger_config),
            underlying_symbol: underlying,
        }
    }

    /// Creates a new risk manager with default configuration.
    #[must_use]
    pub fn with_defaults(underlying_symbol: impl Into<String>) -> Self {
        Self::new(
            underlying_symbol,
            GreeksLimits::default(),
            AutoHedgerConfig::default(),
        )
    }

    /// Returns a reference to the limits.
    #[must_use]
    pub fn limits(&self) -> &GreeksLimits {
        &self.limits
    }

    /// Returns a reference to the current Greeks.
    #[must_use]
    pub fn current_greeks(&self) -> &PortfolioGreeks {
        &self.current_greeks
    }

    /// Returns the underlying symbol.
    #[must_use]
    pub fn underlying_symbol(&self) -> &str {
        &self.underlying_symbol
    }

    /// Returns the circuit breaker status.
    #[must_use]
    pub fn circuit_breaker_status(&self) -> &GreeksCircuitBreakerStatus {
        self.circuit_breaker.status()
    }

    /// Returns the current limit utilization.
    #[must_use]
    pub fn limit_utilization(&self) -> LimitUtilization {
        LimitUtilization::calculate(&self.current_greeks, &self.limits)
    }

    /// Checks if an order is allowed given current Greeks.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the option being traded
    /// * `quantity` - Quantity to trade (positive for buy, negative for sell)
    #[must_use]
    pub fn check_order(&self, option_greeks: &PositionGreeks, quantity: Decimal) -> OrderDecision {
        // Check circuit breaker
        if !self.circuit_breaker.allows_trading() {
            return OrderDecision::Rejected {
                reason: "Circuit breaker is open".to_string(),
            };
        }

        // Calculate post-trade Greeks
        let mut post_trade = self.current_greeks;
        post_trade.add(option_greeks, quantity);

        // Check if would breach limits
        if self.limits.is_any_limit_breached(&post_trade) {
            // Try to scale the order
            let utilization = self.limit_utilization();
            let scale = utilization.scaling_factor();

            if scale <= Decimal::ZERO {
                return OrderDecision::Rejected {
                    reason: "Order would breach Greeks limits".to_string(),
                };
            }

            let scaled_qty = quantity * scale;
            if scaled_qty.abs() < dec!(0.01) {
                return OrderDecision::Rejected {
                    reason: "Scaled order too small".to_string(),
                };
            }

            return OrderDecision::Scaled {
                original_size: quantity.abs(),
                new_size: scaled_qty.abs(),
                reason: format!(
                    "Scaled to {:.0}% due to limit utilization",
                    scale * dec!(100.0)
                ),
            };
        }

        OrderDecision::Allowed
    }

    /// Updates Greeks after a fill.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the filled option
    /// * `quantity` - Filled quantity (positive for buy, negative for sell)
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn update_on_fill(
        &mut self,
        option_greeks: &PositionGreeks,
        quantity: Decimal,
        timestamp: u64,
    ) {
        self.current_greeks.add(option_greeks, quantity);
        self.circuit_breaker
            .check(&self.current_greeks, &self.limits, timestamp);
    }

    /// Removes Greeks when a position is closed.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the closed option
    /// * `quantity` - Closed quantity
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn update_on_close(
        &mut self,
        option_greeks: &PositionGreeks,
        quantity: Decimal,
        timestamp: u64,
    ) {
        self.current_greeks.remove(option_greeks, quantity);
        self.circuit_breaker
            .check(&self.current_greeks, &self.limits, timestamp);
    }

    /// Checks if hedging is needed.
    #[must_use]
    pub fn needs_hedge(&self) -> bool {
        self.hedger.needs_hedge(&self.current_greeks, &self.limits)
    }

    /// Calculates the hedge order.
    #[must_use]
    pub fn calculate_hedge_order(&self, underlying_price: Decimal) -> Option<HedgeOrder> {
        self.hedger
            .calculate_hedge(&self.current_greeks, &self.limits, underlying_price)
    }

    /// Returns the hedge urgency.
    #[must_use]
    pub fn hedge_urgency(&self) -> HedgeUrgency {
        self.hedger
            .hedge_urgency(&self.current_greeks, &self.limits)
    }

    /// Resets the risk manager state.
    pub fn reset(&mut self) {
        self.current_greeks.reset();
        self.circuit_breaker.reset();
    }

    /// Manually resets the circuit breaker.
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker.reset();
    }

    /// Updates the Greeks limits.
    pub fn update_limits(&mut self, limits: GreeksLimits) {
        self.limits = limits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_greeks() -> PositionGreeks {
        PositionGreeks::new(
            dec!(0.5),   // delta
            dec!(0.02),  // gamma
            dec!(-0.05), // theta
            dec!(0.15),  // vega
            dec!(0.08),  // rho
        )
    }

    #[test]
    fn test_order_decision_allowed() {
        let decision = OrderDecision::Allowed;
        assert!(decision.is_allowed());
        assert_eq!(decision.effective_size(dec!(10.0)), dec!(10.0));
    }

    #[test]
    fn test_order_decision_scaled() {
        let decision = OrderDecision::Scaled {
            original_size: dec!(10.0),
            new_size: dec!(5.0),
            reason: "test".to_string(),
        };
        assert!(decision.is_allowed());
        assert_eq!(decision.effective_size(dec!(10.0)), dec!(5.0));
    }

    #[test]
    fn test_order_decision_rejected() {
        let decision = OrderDecision::Rejected {
            reason: "test".to_string(),
        };
        assert!(!decision.is_allowed());
        assert_eq!(decision.effective_size(dec!(10.0)), Decimal::ZERO);
    }

    #[test]
    fn test_circuit_breaker_default() {
        let cb = GreeksCircuitBreaker::with_defaults();
        assert!(cb.allows_trading());
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Closed);
    }

    #[test]
    fn test_circuit_breaker_trip() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(98.0); // 98% of 100 limit

        cb.check(&greeks, &limits, 1000);
        assert!(!cb.allows_trading());
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits::default();

        // Trip it
        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(98.0);
        cb.check(&greeks, &limits, 1000);
        assert!(!cb.allows_trading());

        // Reduce Greeks
        greeks.delta = dec!(50.0);
        cb.check(&greeks, &limits, 2000);
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Cooldown);

        // Wait for cooldown
        cb.check(&greeks, &limits, 100_000);
        assert!(cb.allows_trading());
    }

    #[test]
    fn test_auto_hedger_needs_hedge() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let limits = GreeksLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(50.0); // 50% of limit
        assert!(!hedger.needs_hedge(&greeks, &limits));

        greeks.delta = dec!(85.0); // 85% of limit
        assert!(hedger.needs_hedge(&greeks, &limits));
    }

    #[test]
    fn test_auto_hedger_calculate_hedge() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let limits = GreeksLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(85.0);

        let hedge = hedger.calculate_hedge(&greeks, &limits, dec!(50000.0));
        assert!(hedge.is_some());

        let hedge = hedge.unwrap();
        assert_eq!(hedge.symbol, "BTC");
        assert!(hedge.quantity < Decimal::ZERO); // Selling to reduce long delta
    }

    #[test]
    fn test_limit_utilization() {
        let limits = GreeksLimits::default();
        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(50.0);
        greeks.gamma = dec!(25.0);

        let util = LimitUtilization::calculate(&greeks, &limits);
        assert_eq!(util.delta, dec!(0.5));
        assert_eq!(util.gamma, dec!(0.5));
        assert!(!util.is_breached());
        assert!(!util.is_warning());
    }

    #[test]
    fn test_limit_utilization_warning() {
        let limits = GreeksLimits::default();
        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(85.0);

        let util = LimitUtilization::calculate(&greeks, &limits);
        assert!(util.is_warning());
        assert!(!util.is_breached());
    }

    #[test]
    fn test_limit_utilization_breached() {
        let limits = GreeksLimits::default();
        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(110.0);

        let util = LimitUtilization::calculate(&greeks, &limits);
        assert!(util.is_breached());
    }

    #[test]
    fn test_greeks_risk_manager_check_order_allowed() {
        let manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        let decision = manager.check_order(&greeks, dec!(10.0));
        assert!(matches!(decision, OrderDecision::Allowed));
    }

    #[test]
    fn test_greeks_risk_manager_check_order_rejected() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");

        // Fill up to near limit
        let greeks = create_test_greeks();
        manager.update_on_fill(&greeks, dec!(190.0), 1000); // 95 delta

        // Try to add more
        let decision = manager.check_order(&greeks, dec!(20.0));
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_greeks_risk_manager_update_on_fill() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(10.0), 1000);

        assert_eq!(manager.current_greeks().delta, dec!(5.0));
        assert_eq!(manager.current_greeks().gamma, dec!(0.2));
    }

    #[test]
    fn test_greeks_risk_manager_needs_hedge() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        // Below threshold
        manager.update_on_fill(&greeks, dec!(100.0), 1000); // 50 delta
        assert!(!manager.needs_hedge());

        // Above threshold
        manager.update_on_fill(&greeks, dec!(80.0), 2000); // 90 delta total
        assert!(manager.needs_hedge());
    }

    #[test]
    fn test_greeks_risk_manager_calculate_hedge() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(180.0), 1000); // 90 delta

        let hedge = manager.calculate_hedge_order(dec!(50000.0));
        assert!(hedge.is_some());
    }

    #[test]
    fn test_greeks_risk_manager_reset() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(10.0), 1000);
        assert!(manager.current_greeks().delta != Decimal::ZERO);

        manager.reset();
        assert_eq!(manager.current_greeks().delta, Decimal::ZERO);
    }

    #[test]
    fn test_hedge_urgency() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        // Normal
        manager.update_on_fill(&greeks, dec!(160.0), 1000); // 80 delta
        assert_eq!(manager.hedge_urgency(), HedgeUrgency::Normal);

        // Urgent
        manager.update_on_fill(&greeks, dec!(20.0), 2000); // 90 delta
        assert_eq!(manager.hedge_urgency(), HedgeUrgency::Urgent);

        // Emergency
        manager.update_on_fill(&greeks, dec!(10.0), 3000); // 95 delta
        assert_eq!(manager.hedge_urgency(), HedgeUrgency::Emergency);
    }
}
