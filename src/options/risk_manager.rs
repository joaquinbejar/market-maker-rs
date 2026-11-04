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
//! let decision = risk_manager.check_order(&option_greeks, quantity)?;
//! ```
//!
//! # Failures
//!
//! A utilization is an exposure divided by its limit, so it fails for a zero
//! limit as well as on overflow, with
//! [`MMError::NumericalError`](crate::types::error::MMError::NumericalError).
//! A risk check that cannot be computed never allows trading: the circuit
//! breaker opens, and when a fill cannot be booked it stays open until a manual
//! reset. Meanwhile, what would answer from the Greeks is
//! [`MMError::GreeksMissFill`](crate::types::error::MMError::GreeksMissFill).
//! The errors of a fill say whether it was booked:
//! [`MMError::FillNotBooked`](crate::types::error::MMError::FillNotBooked) or
//! [`MMError::FillBookedCheckFailed`](crate::types::error::MMError::FillBookedCheckFailed).

use crate::Decimal;
use crate::options::greeks::{PortfolioGreeks, PositionGreeks};
use crate::options::market_maker::{GreeksLimits, HedgeOrder, HedgeType};
use crate::types::decimal::{OrOverflow, negate, saturating_div};
use crate::types::error::{MMError, MMResult};
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The share of `limit` that `exposure` uses, in absolute value.
fn utilization(exposure: Decimal, limit: Decimal, what: &str) -> MMResult<Decimal> {
    exposure.abs().checked_div(limit).or_overflow(what)
}

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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero delta, gamma or vega limit, when
    /// a utilization overflows, and when the end of the cooldown is beyond
    /// `u64`. A breaker that was closed opens, and one that was open stays
    /// open: a check that cannot be computed never allows trading.
    pub fn check(
        &mut self,
        greeks: &PortfolioGreeks,
        limits: &GreeksLimits,
        timestamp: u64,
    ) -> MMResult<()> {
        let result = self.evaluate(greeks, limits, timestamp);
        if let Err(error) = &result {
            self.fault(error, timestamp);
        }
        result
    }

    /// Opens the breaker because of `error`, unless it is open already.
    fn fault(&mut self, error: &MMError, timestamp: u64) {
        if self.status.state != GreeksCircuitBreakerState::Open {
            self.status = GreeksCircuitBreakerStatus::open(format!("{error}"), timestamp);
        }
    }

    /// The transitions of [`Self::check`].
    fn evaluate(
        &mut self,
        greeks: &PortfolioGreeks,
        limits: &GreeksLimits,
        timestamp: u64,
    ) -> MMResult<()> {
        // Check cooldown
        if self.status.state == GreeksCircuitBreakerState::Cooldown
            && let Some(ends_at) = self.status.cooldown_ends_at
            && timestamp >= ends_at
        {
            self.status = GreeksCircuitBreakerStatus::closed();
        }
        if self.status.state == GreeksCircuitBreakerState::Cooldown {
            return Ok(());
        }

        let delta_util = utilization(greeks.delta, limits.max_delta, "delta utilization")?;
        let gamma_util = utilization(greeks.gamma, limits.max_gamma, "gamma utilization")?;
        let vega_util = utilization(greeks.vega, limits.max_vega, "vega utilization")?;

        // Check if should trip. The percentages are report only values.
        if self.status.state == GreeksCircuitBreakerState::Closed {
            if delta_util >= self.config.delta_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Delta utilization {:.1}% exceeds trip threshold",
                        delta_util.saturating_mul(dec!(100.0))
                    ),
                    timestamp,
                );
            } else if gamma_util >= self.config.gamma_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Gamma utilization {:.1}% exceeds trip threshold",
                        gamma_util.saturating_mul(dec!(100.0))
                    ),
                    timestamp,
                );
            } else if vega_util >= self.config.vega_trip_pct {
                self.status = GreeksCircuitBreakerStatus::open(
                    format!(
                        "Vega utilization {:.1}% exceeds trip threshold",
                        vega_util.saturating_mul(dec!(100.0))
                    ),
                    timestamp,
                );
            }
        }

        // Check if should reset
        if self.status.state == GreeksCircuitBreakerState::Open
            && delta_util < self.config.reset_pct
            && gamma_util < self.config.reset_pct
            && vega_util < self.config.reset_pct
        {
            let cooldown_ends_at = timestamp
                .checked_add(self.config.cooldown_ms)
                .or_overflow("cooldown end")?;
            self.status = GreeksCircuitBreakerStatus {
                state: GreeksCircuitBreakerState::Cooldown,
                reason: Some("Greeks returned to safe levels".to_string()),
                state_changed_at: timestamp,
                cooldown_ends_at: Some(cooldown_ends_at),
            };
        }
        Ok(())
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero delta limit and when the delta
    /// utilization overflows, if hedging is enabled.
    pub fn needs_hedge(&self, greeks: &PortfolioGreeks, limits: &GreeksLimits) -> MMResult<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let delta_util = utilization(greeks.delta, limits.max_delta, "delta utilization")?;
        Ok(delta_util >= self.config.trigger_threshold_pct)
    }

    /// Calculates the hedge urgency.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero delta limit and when the delta
    /// utilization overflows.
    pub fn hedge_urgency(
        &self,
        greeks: &PortfolioGreeks,
        limits: &GreeksLimits,
    ) -> MMResult<HedgeUrgency> {
        let delta_util = utilization(greeks.delta, limits.max_delta, "delta utilization")?;

        Ok(if delta_util >= dec!(0.95) {
            HedgeUrgency::Emergency
        } else if delta_util >= dec!(0.9) {
            HedgeUrgency::Urgent
        } else {
            HedgeUrgency::Normal
        })
    }

    /// Calculates the hedge order.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when [`Self::needs_hedge`] fails, when the
    /// target delta, the delta to hedge or the number of shares overflows, and
    /// for a zero contract multiplier with a hedge to report.
    pub fn calculate_hedge(
        &self,
        greeks: &PortfolioGreeks,
        limits: &GreeksLimits,
        underlying_price: Decimal,
    ) -> MMResult<Option<HedgeOrder>> {
        if !self.needs_hedge(greeks, limits)? {
            return Ok(None);
        }

        // Calculate target delta
        let target = limits
            .max_delta
            .checked_mul(self.config.target_delta_pct)
            .or_overflow("target delta")?;
        let target_delta = if greeks.delta > Decimal::ZERO {
            target
        } else {
            negate(target)
        };

        // Calculate delta to hedge
        let delta_to_hedge = greeks
            .delta
            .checked_sub(target_delta)
            .or_overflow("delta to hedge")?;

        // Convert to shares
        let shares = negate(delta_to_hedge)
            .checked_mul(self.config.contract_multiplier)
            .or_overflow("hedge shares")?;

        // Clamp to min/max
        let clamped_shares = if shares.abs() < self.config.min_hedge_size {
            return Ok(None); // Too small to hedge
        } else if shares.abs() > self.config.max_hedge_size {
            if shares > Decimal::ZERO {
                self.config.max_hedge_size
            } else {
                negate(self.config.max_hedge_size)
            }
        } else {
            shares
        };

        let delta_impact = negate(clamped_shares)
            .checked_div(self.config.contract_multiplier)
            .or_overflow("hedge delta impact")?;

        Ok(Some(HedgeOrder::new(
            self.underlying_symbol.clone(),
            clamped_shares,
            underlying_price,
            HedgeType::Underlying,
            delta_impact,
        )))
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
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero delta, gamma or vega limit and
    /// when a utilization overflows.
    pub fn calculate(greeks: &PortfolioGreeks, limits: &GreeksLimits) -> MMResult<Self> {
        let delta = utilization(greeks.delta, limits.max_delta, "delta utilization")?;
        let gamma = utilization(greeks.gamma, limits.max_gamma, "gamma utilization")?;
        let vega = utilization(greeks.vega, limits.max_vega, "vega utilization")?;
        let theta = if limits.max_theta < Decimal::ZERO {
            greeks
                .theta
                .checked_div(limits.max_theta)
                .or_overflow("theta utilization")?
        } else {
            Decimal::ZERO
        };

        let max_utilization = delta.max(gamma).max(vega).max(theta);

        Ok(Self {
            delta,
            gamma,
            vega,
            theta,
            max_utilization,
        })
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
            // Linear scaling from 1.0 at 80% to 0.0 at 100%. The utilization
            // is in [0.8, 1) here, so neither operation saturates.
            saturating_div(dec!(1.0).saturating_sub(self.max_utilization), dec!(0.2))
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
    /// A fill could not be booked, so `current_greeks` misses it: the circuit
    /// breaker stays open and the hedge and limit readings fail until a manual
    /// reset.
    booking_failed: bool,
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
            booking_failed: false,
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
    ///
    /// This is an incomplete snapshot while [`Self::has_unbooked_fill`] is
    /// true: it misses a real fill.
    #[must_use]
    pub fn current_greeks(&self) -> &PortfolioGreeks {
        &self.current_greeks
    }

    /// Returns true while the Greeks miss a fill that could not be booked,
    /// until [`Self::reconcile`], [`Self::reset`] or
    /// [`Self::reset_circuit_breaker`]. Later fills that book do not clear it.
    #[must_use]
    pub fn has_unbooked_fill(&self) -> bool {
        self.booking_failed
    }

    /// The current Greeks, unless they are known to miss a fill.
    fn booked_greeks(&self) -> MMResult<&PortfolioGreeks> {
        if self.booking_failed {
            Err(MMError::GreeksMissFill)
        } else {
            Ok(&self.current_greeks)
        }
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
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and as [`LimitUtilization::calculate`].
    pub fn limit_utilization(&self) -> MMResult<LimitUtilization> {
        LimitUtilization::calculate(self.booked_greeks()?, &self.limits)
    }

    /// Checks if an order is allowed given current Greeks.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the option being traded
    /// * `quantity` - Quantity to trade (positive for buy, negative for sell)
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when the post trade Greeks overflow, and,
    /// for an order that would breach a limit, when
    /// [`Self::limit_utilization`] fails.
    pub fn check_order(
        &self,
        option_greeks: &PositionGreeks,
        quantity: Decimal,
    ) -> MMResult<OrderDecision> {
        // Check circuit breaker
        if !self.circuit_breaker.allows_trading() {
            return Ok(OrderDecision::Rejected {
                reason: "Circuit breaker is open".to_string(),
            });
        }

        // Calculate post-trade Greeks
        let mut post_trade = self.current_greeks;
        post_trade.add(option_greeks, quantity)?;

        // Check if would breach limits
        if self.limits.is_any_limit_breached(&post_trade) {
            // Try to scale the order
            let utilization = self.limit_utilization()?;
            let scale = utilization.scaling_factor();

            if scale <= Decimal::ZERO {
                return Ok(OrderDecision::Rejected {
                    reason: "Order would breach Greeks limits".to_string(),
                });
            }

            // The scale is in (0, 1], so this does not saturate.
            let scaled_qty = quantity.saturating_mul(scale);
            if scaled_qty.abs() < dec!(0.01) {
                return Ok(OrderDecision::Rejected {
                    reason: "Scaled order too small".to_string(),
                });
            }

            return Ok(OrderDecision::Scaled {
                original_size: quantity.abs(),
                new_size: scaled_qty.abs(),
                reason: format!(
                    "Scaled to {:.0}% due to limit utilization",
                    scale.saturating_mul(dec!(100.0))
                ),
            });
        }

        Ok(OrderDecision::Allowed)
    }

    /// Updates Greeks after a fill.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the filled option
    /// * `quantity` - Filled quantity (positive for buy, negative for sell)
    /// * `timestamp` - Current timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// - [`MMError::FillNotBooked`] when the fill cannot be booked: this call
    ///   leaves the Greeks unchanged, so they miss a real fill. The circuit
    ///   breaker stays open, and [`Self::has_unbooked_fill`] true, until
    ///   [`Self::reconcile`], [`Self::reset`] or
    ///   [`Self::reset_circuit_breaker`].
    /// - [`MMError::FillBookedCheckFailed`] when the fill is booked but
    ///   [`GreeksCircuitBreaker::check`] fails. The Greeks include the fill, so
    ///   do not book it again.
    pub fn update_on_fill(
        &mut self,
        option_greeks: &PositionGreeks,
        quantity: Decimal,
        timestamp: u64,
    ) -> MMResult<()> {
        let booked = self.current_greeks.add(option_greeks, quantity);
        self.after_booking(booked, timestamp)
    }

    /// Removes Greeks when a position is closed.
    ///
    /// # Arguments
    ///
    /// * `option_greeks` - Greeks of the closed option
    /// * `quantity` - Closed quantity
    /// * `timestamp` - Current timestamp in milliseconds
    ///
    /// # Errors
    ///
    /// As [`Self::update_on_fill`].
    pub fn update_on_close(
        &mut self,
        option_greeks: &PositionGreeks,
        quantity: Decimal,
        timestamp: u64,
    ) -> MMResult<()> {
        let booked = self.current_greeks.remove(option_greeks, quantity);
        self.after_booking(booked, timestamp)
    }

    /// Runs the circuit breaker after a booking, or holds it open if the
    /// Greeks miss a fill.
    fn after_booking(&mut self, booked: MMResult<()>, timestamp: u64) -> MMResult<()> {
        if let Err(error) = booked {
            self.booking_failed = true;
            self.circuit_breaker.fault(&error, timestamp);
            return Err(MMError::FillNotBooked(error.to_string()));
        }
        if self.booking_failed {
            return Ok(());
        }
        self.circuit_breaker
            .check(&self.current_greeks, &self.limits, timestamp)
            .map_err(|error| MMError::FillBookedCheckFailed(error.to_string()))
    }

    /// Checks if hedging is needed.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and as [`AutoHedger::needs_hedge`].
    pub fn needs_hedge(&self) -> MMResult<bool> {
        self.hedger.needs_hedge(self.booked_greeks()?, &self.limits)
    }

    /// Calculates the hedge order.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and as [`AutoHedger::calculate_hedge`].
    pub fn calculate_hedge_order(&self, underlying_price: Decimal) -> MMResult<Option<HedgeOrder>> {
        self.hedger
            .calculate_hedge(self.booked_greeks()?, &self.limits, underlying_price)
    }

    /// Returns the hedge urgency.
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while [`Self::has_unbooked_fill`] is true,
    /// and as [`AutoHedger::hedge_urgency`].
    pub fn hedge_urgency(&self) -> MMResult<HedgeUrgency> {
        self.hedger
            .hedge_urgency(self.booked_greeks()?, &self.limits)
    }

    /// Installs Greeks recomputed from the positions, which include any fill
    /// that could not be booked, clears the failed booking and runs the
    /// circuit breaker from closed against them, so a breach reopens it.
    ///
    /// # Errors
    ///
    /// As [`GreeksCircuitBreaker::check`]: the Greeks are installed, and a
    /// check that cannot be computed opens the breaker.
    pub fn reconcile(&mut self, greeks: PortfolioGreeks, timestamp: u64) -> MMResult<()> {
        self.current_greeks = greeks;
        self.booking_failed = false;
        self.circuit_breaker.reset();
        self.circuit_breaker
            .check(&self.current_greeks, &self.limits, timestamp)
    }

    /// Resets the risk manager state: the Greeks go to zero, and a failed
    /// booking is cleared.
    pub fn reset(&mut self) {
        self.current_greeks.reset();
        self.circuit_breaker.reset();
        self.booking_failed = false;
    }

    /// Manually resets the circuit breaker, including after a fill that could
    /// not be booked: the Greeks are kept, and the caller accepts that they
    /// miss that fill. [`Self::reconcile`] installs correct ones instead.
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker.reset();
        self.booking_failed = false;
    }

    /// Updates the Greeks limits. This does not clear a failed booking.
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

        cb.check(&greeks, &limits, 1000).unwrap();
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
        cb.check(&greeks, &limits, 1000).unwrap();
        assert!(!cb.allows_trading());

        // Reduce Greeks
        greeks.delta = dec!(50.0);
        cb.check(&greeks, &limits, 2000).unwrap();
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Cooldown);

        // Wait for cooldown
        cb.check(&greeks, &limits, 100_000).unwrap();
        assert!(cb.allows_trading());
    }

    #[test]
    fn test_auto_hedger_needs_hedge() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let limits = GreeksLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(50.0); // 50% of limit
        assert!(!hedger.needs_hedge(&greeks, &limits).unwrap());

        greeks.delta = dec!(85.0); // 85% of limit
        assert!(hedger.needs_hedge(&greeks, &limits).unwrap());
    }

    #[test]
    fn test_auto_hedger_calculate_hedge() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let limits = GreeksLimits::default();

        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(85.0);

        let hedge = hedger
            .calculate_hedge(&greeks, &limits, dec!(50000.0))
            .unwrap();
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

        let util = LimitUtilization::calculate(&greeks, &limits).unwrap();
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

        let util = LimitUtilization::calculate(&greeks, &limits).unwrap();
        assert!(util.is_warning());
        assert!(!util.is_breached());
    }

    #[test]
    fn test_limit_utilization_breached() {
        let limits = GreeksLimits::default();
        let mut greeks = PortfolioGreeks::new();
        greeks.delta = dec!(110.0);

        let util = LimitUtilization::calculate(&greeks, &limits).unwrap();
        assert!(util.is_breached());
    }

    #[test]
    fn test_greeks_risk_manager_check_order_allowed() {
        let manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        let decision = manager.check_order(&greeks, dec!(10.0)).unwrap();
        assert!(matches!(decision, OrderDecision::Allowed));
    }

    #[test]
    fn test_greeks_risk_manager_check_order_rejected() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");

        // Fill up to near limit
        let greeks = create_test_greeks();
        manager.update_on_fill(&greeks, dec!(190.0), 1000).unwrap(); // 95 delta

        // Try to add more
        let decision = manager.check_order(&greeks, dec!(20.0)).unwrap();
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_greeks_risk_manager_update_on_fill() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(10.0), 1000).unwrap();

        assert_eq!(manager.current_greeks().delta, dec!(5.0));
        assert_eq!(manager.current_greeks().gamma, dec!(0.2));
    }

    #[test]
    fn test_greeks_risk_manager_needs_hedge() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        // Below threshold
        manager.update_on_fill(&greeks, dec!(100.0), 1000).unwrap(); // 50 delta
        assert!(!manager.needs_hedge().unwrap());

        // Above threshold
        manager.update_on_fill(&greeks, dec!(80.0), 2000).unwrap(); // 90 delta total
        assert!(manager.needs_hedge().unwrap());
    }

    #[test]
    fn test_greeks_risk_manager_calculate_hedge() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(180.0), 1000).unwrap(); // 90 delta

        let hedge = manager.calculate_hedge_order(dec!(50000.0)).unwrap();
        assert!(hedge.is_some());
    }

    #[test]
    fn test_greeks_risk_manager_reset() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        manager.update_on_fill(&greeks, dec!(10.0), 1000).unwrap();
        assert!(manager.current_greeks().delta != Decimal::ZERO);

        manager.reset();
        assert_eq!(manager.current_greeks().delta, Decimal::ZERO);
    }

    #[test]
    fn test_hedge_urgency() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = create_test_greeks();

        // Normal
        manager.update_on_fill(&greeks, dec!(160.0), 1000).unwrap(); // 80 delta
        assert_eq!(manager.hedge_urgency().unwrap(), HedgeUrgency::Normal);

        // Urgent
        manager.update_on_fill(&greeks, dec!(20.0), 2000).unwrap(); // 90 delta
        assert_eq!(manager.hedge_urgency().unwrap(), HedgeUrgency::Urgent);

        // Emergency
        manager.update_on_fill(&greeks, dec!(10.0), 3000).unwrap(); // 95 delta
        assert_eq!(manager.hedge_urgency().unwrap(), HedgeUrgency::Emergency);
    }

    fn overflow(what: &str) -> MMError {
        MMError::NumericalError(format!("{what} overflows"))
    }

    fn delta_of(delta: Decimal) -> PortfolioGreeks {
        PortfolioGreeks {
            delta,
            ..PortfolioGreeks::new()
        }
    }

    fn zero_limit(field: &str) -> GreeksLimits {
        let mut limits = GreeksLimits::default();
        match field {
            "delta" => limits.max_delta = Decimal::ZERO,
            "gamma" => limits.max_gamma = Decimal::ZERO,
            _ => limits.max_vega = Decimal::ZERO,
        }
        limits
    }

    #[test]
    fn test_circuit_breaker_zero_limit_fails_closed() {
        for field in ["delta", "gamma", "vega"] {
            let mut cb = GreeksCircuitBreaker::with_defaults();
            assert_eq!(
                cb.check(&PortfolioGreeks::new(), &zero_limit(field), 1000),
                Err(overflow(&format!("{field} utilization")))
            );
            assert_eq!(cb.status().state, GreeksCircuitBreakerState::Open);
            assert_eq!(cb.status().state_changed_at, 1000);
            assert!(!cb.allows_trading());
        }
    }

    #[test]
    fn test_circuit_breaker_utilization_overflow_fails_closed() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits {
            max_delta: Decimal::new(1, 28),
            ..Default::default()
        };
        assert_eq!(
            cb.check(&delta_of(Decimal::MAX), &limits, 1000),
            Err(overflow("delta utilization"))
        );
        assert!(!cb.allows_trading());
    }

    #[test]
    fn test_circuit_breaker_open_stays_open_when_the_check_fails() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        cb.check(&delta_of(dec!(98)), &GreeksLimits::default(), 1000)
            .unwrap();
        let reason = cb.status().reason.clone();

        assert!(
            cb.check(&PortfolioGreeks::new(), &zero_limit("delta"), 2000)
                .is_err()
        );
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Open);
        assert_eq!(cb.status().state_changed_at, 1000);
        assert_eq!(cb.status().reason, reason);
    }

    #[test]
    fn test_circuit_breaker_cooldown_end_overflow_stays_open() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits::default();
        cb.check(&delta_of(dec!(98)), &limits, 1000).unwrap();

        // Below the reset level, at a timestamp the cooldown cannot follow.
        assert_eq!(
            cb.check(&PortfolioGreeks::new(), &limits, u64::MAX),
            Err(overflow("cooldown end"))
        );
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Open);
    }

    #[test]
    fn test_circuit_breaker_running_cooldown_does_not_read_the_limits() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits::default();
        cb.check(&delta_of(dec!(98)), &limits, 1000).unwrap();
        cb.check(&PortfolioGreeks::new(), &limits, 2000).unwrap();
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Cooldown);

        cb.check(&PortfolioGreeks::new(), &zero_limit("delta"), 3000)
            .unwrap();
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Cooldown);
    }

    #[test]
    fn test_circuit_breaker_failed_check_after_cooldown_fails_closed() {
        let mut cb = GreeksCircuitBreaker::with_defaults();
        let limits = GreeksLimits::default();
        cb.check(&delta_of(dec!(98)), &limits, 1000).unwrap();
        cb.check(&PortfolioGreeks::new(), &limits, 2000).unwrap();

        assert!(
            cb.check(&PortfolioGreeks::new(), &zero_limit("delta"), 100_000)
                .is_err()
        );
        assert_eq!(cb.status().state, GreeksCircuitBreakerState::Open);
    }

    #[test]
    fn test_auto_hedger_zero_limit_is_an_error() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let limits = zero_limit("delta");
        let greeks = PortfolioGreeks::new();
        let expected = overflow("delta utilization");
        assert_eq!(hedger.needs_hedge(&greeks, &limits), Err(expected.clone()));
        assert_eq!(
            hedger.hedge_urgency(&greeks, &limits),
            Err(expected.clone())
        );
        assert_eq!(
            hedger
                .calculate_hedge(&greeks, &limits, dec!(1))
                .map(|hedge| hedge.is_some()),
            Err(expected)
        );
    }

    #[test]
    fn test_auto_hedger_disabled_does_not_read_the_limit() {
        let hedger = AutoHedger::new(
            "BTC",
            AutoHedgerConfig {
                enabled: false,
                ..Default::default()
            },
        );
        assert_eq!(
            hedger.needs_hedge(&PortfolioGreeks::new(), &zero_limit("delta")),
            Ok(false)
        );
    }

    fn hedge_error(
        config: AutoHedgerConfig,
        limits: &GreeksLimits,
        delta: Decimal,
    ) -> Option<MMError> {
        AutoHedger::new("BTC", config)
            .calculate_hedge(&delta_of(delta), limits, dec!(1))
            .err()
    }

    #[test]
    fn test_auto_hedger_target_delta_overflow() {
        let config = AutoHedgerConfig {
            target_delta_pct: dec!(2),
            ..Default::default()
        };
        let limits = GreeksLimits {
            max_delta: Decimal::MAX,
            ..Default::default()
        };
        assert_eq!(
            hedge_error(config, &limits, Decimal::MAX),
            Some(overflow("target delta"))
        );
    }

    #[test]
    fn test_auto_hedger_delta_to_hedge_overflow() {
        // A negative target percentage puts the target on the other side.
        let config = AutoHedgerConfig {
            target_delta_pct: dec!(-1),
            ..Default::default()
        };
        let limits = GreeksLimits {
            max_delta: Decimal::MAX,
            ..Default::default()
        };
        assert_eq!(
            hedge_error(config, &limits, Decimal::MAX),
            Some(overflow("delta to hedge"))
        );
    }

    #[test]
    fn test_auto_hedger_shares_overflow() {
        let config = AutoHedgerConfig {
            contract_multiplier: Decimal::MAX,
            ..Default::default()
        };
        assert_eq!(
            hedge_error(config, &GreeksLimits::default(), dec!(85)),
            Some(overflow("hedge shares"))
        );
    }

    #[test]
    fn test_auto_hedger_zero_multiplier_is_an_error() {
        // No minimum, so the zero shares are reported, with a zero divisor.
        let config = AutoHedgerConfig {
            contract_multiplier: Decimal::ZERO,
            min_hedge_size: Decimal::ZERO,
            ..Default::default()
        };
        assert_eq!(
            hedge_error(config, &GreeksLimits::default(), dec!(85)),
            Some(overflow("hedge delta impact"))
        );
    }

    #[test]
    fn test_auto_hedger_clamps_a_short_hedge_to_the_negated_maximum() {
        let hedger = AutoHedger::new("BTC", AutoHedgerConfig::default());
        let hedge = hedger
            .calculate_hedge(&delta_of(dec!(100)), &GreeksLimits::default(), dec!(1))
            .unwrap()
            .unwrap();
        assert_eq!(hedge.quantity, dec!(-100));
        assert_eq!(hedge.delta_impact, dec!(1));
    }

    #[test]
    fn test_limit_utilization_each_limit_is_checked() {
        for field in ["delta", "gamma", "vega"] {
            assert_eq!(
                LimitUtilization::calculate(&PortfolioGreeks::new(), &zero_limit(field))
                    .map(|util| util.max_utilization),
                Err(overflow(&format!("{field} utilization")))
            );
        }
    }

    #[test]
    fn test_limit_utilization_theta_overflow() {
        let limits = GreeksLimits {
            max_theta: Decimal::new(-1, 28),
            ..Default::default()
        };
        let greeks = PortfolioGreeks {
            theta: Decimal::MIN,
            ..PortfolioGreeks::new()
        };
        assert_eq!(
            LimitUtilization::calculate(&greeks, &limits).map(|util| util.max_utilization),
            Err(overflow("theta utilization"))
        );
    }

    #[test]
    fn test_scaling_factor_is_unchanged() {
        let scaling = |max_utilization| {
            LimitUtilization {
                max_utilization,
                ..Default::default()
            }
            .scaling_factor()
        };
        assert_eq!(scaling(dec!(0.5)), dec!(1.0));
        assert_eq!(scaling(dec!(0.9)), dec!(0.5));
        assert_eq!(scaling(dec!(1.0)), Decimal::ZERO);
    }

    fn manager_with(limits: GreeksLimits) -> GreeksRiskManager {
        GreeksRiskManager::new("BTC", limits, AutoHedgerConfig::default())
    }

    #[test]
    fn test_check_order_post_trade_overflow() {
        let manager = GreeksRiskManager::with_defaults("BTC");
        let greeks = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        assert_eq!(
            manager.check_order(&greeks, dec!(2)),
            Err(overflow("scaled delta"))
        );
    }

    #[test]
    fn test_check_order_utilization_failure() {
        let manager = manager_with(zero_limit("delta"));
        assert_eq!(
            manager.check_order(&create_test_greeks(), dec!(1)),
            Err(overflow("delta utilization"))
        );
    }

    fn not_booked(what: &str) -> MMError {
        MMError::FillNotBooked(overflow(what).to_string())
    }

    #[test]
    fn test_update_on_fill_booking_failure_holds_the_breaker_open() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        let small = create_test_greeks();

        assert_eq!(
            manager.update_on_fill(&huge, dec!(2), 1000),
            Err(not_booked("scaled delta"))
        );
        assert_eq!(*manager.current_greeks(), PortfolioGreeks::new());
        assert!(manager.has_unbooked_fill());
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Open
        );

        // Low utilization would close an ordinary breaker: this one misses a
        // fill, so it stays open.
        manager.update_on_fill(&small, dec!(1), 2000).unwrap();
        manager.update_on_close(&small, dec!(1), 200_000).unwrap();
        manager.update_on_fill(&small, dec!(1), 400_000).unwrap();
        assert!(matches!(
            manager.check_order(&small, dec!(1)),
            Ok(OrderDecision::Rejected { .. })
        ));

        // A manual reset accepts it.
        manager.reset_circuit_breaker();
        assert!(manager.check_order(&small, dec!(1)).unwrap().is_allowed());
        manager.update_on_fill(&small, dec!(1), 500_000).unwrap();
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Closed
        );
    }

    #[test]
    fn test_update_on_close_booking_failure_holds_the_breaker_open() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));

        assert_eq!(
            manager.update_on_close(&huge, dec!(2), 1000),
            Err(not_booked("scaled delta"))
        );
        assert!(manager.has_unbooked_fill());
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Open
        );

        manager.reset();
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Closed
        );
        manager.update_on_fill(&huge, dec!(0), 2000).unwrap();
        assert!(
            manager
                .check_order(&create_test_greeks(), dec!(1))
                .unwrap()
                .is_allowed()
        );
    }

    #[test]
    fn test_update_on_fill_check_failure_books_the_fill() {
        let mut manager = manager_with(zero_limit("gamma"));
        let greeks = create_test_greeks();
        assert_eq!(
            manager.update_on_fill(&greeks, dec!(10), 1000),
            Err(MMError::FillBookedCheckFailed(
                overflow("gamma utilization").to_string()
            ))
        );
        // Booked: replaying it would count it twice.
        assert_eq!(manager.current_greeks().delta, dec!(5));
        assert!(!manager.has_unbooked_fill());
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Open
        );
    }

    #[test]
    fn test_update_on_close_check_failure_books_the_close() {
        let mut manager = manager_with(zero_limit("gamma"));
        let greeks = create_test_greeks();
        assert_eq!(
            manager.update_on_close(&greeks, dec!(10), 1000),
            Err(MMError::FillBookedCheckFailed(
                overflow("gamma utilization").to_string()
            ))
        );
        assert_eq!(manager.current_greeks().delta, dec!(-5));
        assert!(!manager.has_unbooked_fill());
    }

    fn assert_readings_refuse(manager: &GreeksRiskManager) {
        let refused = MMError::GreeksMissFill;
        assert_eq!(manager.needs_hedge(), Err(refused.clone()));
        assert_eq!(
            manager
                .calculate_hedge_order(dec!(100))
                .map(|hedge| hedge.is_some()),
            Err(refused.clone())
        );
        assert_eq!(manager.hedge_urgency(), Err(refused.clone()));
        assert_eq!(
            manager.limit_utilization().map(|util| util.delta),
            Err(refused)
        );
        assert!(matches!(
            manager.check_order(&create_test_greeks(), dec!(1)),
            Ok(OrderDecision::Rejected { .. })
        ));
    }

    #[test]
    fn test_readings_refuse_after_a_failed_booking_on_a_flat_book() {
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        let small = create_test_greeks();

        let mut manager = GreeksRiskManager::with_defaults("BTC");
        assert!(!manager.needs_hedge().unwrap());
        assert_eq!(
            manager.update_on_fill(&huge, dec!(2), 1000),
            Err(not_booked("scaled delta"))
        );
        assert_readings_refuse(&manager);

        // A later fill that books does not clear it, a second failure leaves
        // it set, and neither do new limits.
        manager.update_on_fill(&small, dec!(1), 2000).unwrap();
        assert_eq!(manager.current_greeks().delta, dec!(0.5));
        assert!(manager.has_unbooked_fill());
        assert_readings_refuse(&manager);
        assert_eq!(
            manager.update_on_fill(&huge, dec!(2), 3000),
            Err(not_booked("scaled delta"))
        );
        assert_eq!(manager.current_greeks().delta, dec!(0.5));
        assert_readings_refuse(&manager);
        manager.update_limits(GreeksLimits::default());
        assert_readings_refuse(&manager);

        // Each explicit reset does.
        manager.reset_circuit_breaker();
        assert!(!manager.has_unbooked_fill());
        assert!(!manager.needs_hedge().unwrap());
        assert_eq!(manager.current_greeks().delta, dec!(0.5));

        let mut manager = GreeksRiskManager::with_defaults("BTC");
        assert_eq!(
            manager.update_on_close(&huge, dec!(2), 1000),
            Err(not_booked("scaled delta"))
        );
        assert_readings_refuse(&manager);
        manager.reset();
        assert!(!manager.has_unbooked_fill());
        assert!(!manager.needs_hedge().unwrap());
    }

    #[test]
    fn test_replay_after_fill_not_booked_counts_it_once() {
        let mut manager = GreeksRiskManager::with_defaults("BTC");
        let big = PositionGreeks::new(
            dec!(40000000000000000000000000000),
            dec!(0),
            dec!(0),
            dec!(0),
            dec!(0),
        );
        manager.update_on_fill(&big, dec!(1), 1000).unwrap();
        assert!(matches!(
            manager.update_on_fill(&big, dec!(1), 2000),
            Err(MMError::FillNotBooked(_))
        ));
        assert_eq!(manager.current_greeks().delta, big.delta);

        // This call applied nothing, so with room it is counted once. That is
        // no recovery: nothing proves which fill it was.
        manager.update_on_close(&big, dec!(1), 3000).unwrap();
        manager.update_on_fill(&big, dec!(1), 4000).unwrap();
        assert_eq!(manager.current_greeks().delta, big.delta);
        assert!(manager.has_unbooked_fill());
        assert_readings_refuse(&manager);

        // Only an explicit clear ends it.
        manager.reset_circuit_breaker();
        assert!(!manager.has_unbooked_fill());
        assert_eq!(manager.current_greeks().delta, big.delta);
    }

    #[test]
    fn test_reconcile_installs_greeks_and_reruns_the_breaker() {
        let huge = PositionGreeks::new(Decimal::MAX, dec!(0), dec!(0), dec!(0), dec!(0));
        let failed = || {
            let mut manager = GreeksRiskManager::with_defaults("BTC");
            manager.update_on_fill(&huge, dec!(2), 1000).unwrap_err();
            assert_readings_refuse(&manager);
            manager
        };

        // Within the limits: trading again, from the recomputed Greeks.
        let within = PortfolioGreeks {
            delta: dec!(1),
            ..PortfolioGreeks::new()
        };
        let mut manager = failed();
        manager.reconcile(within, 2000).unwrap();
        assert!(!manager.has_unbooked_fill());
        assert_eq!(*manager.current_greeks(), within);
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Closed
        );
        assert!(!manager.needs_hedge().unwrap());

        // Far beyond them: readable, but the breaker reopens.
        let beyond = PortfolioGreeks {
            delta: dec!(1000),
            ..PortfolioGreeks::new()
        };
        let mut manager = failed();
        manager.reconcile(beyond, 2000).unwrap();
        assert!(!manager.has_unbooked_fill());
        assert_eq!(
            manager.circuit_breaker_status().state,
            GreeksCircuitBreakerState::Open
        );
        assert!(manager.needs_hedge().unwrap());
    }

    #[test]
    fn test_manager_readings_propagate_a_zero_limit() {
        let manager = manager_with(zero_limit("delta"));
        let expected = overflow("delta utilization");
        assert_eq!(
            manager.limit_utilization().map(|util| util.delta),
            Err(expected.clone())
        );
        assert_eq!(manager.needs_hedge(), Err(expected.clone()));
        assert_eq!(manager.hedge_urgency(), Err(expected.clone()));
        assert_eq!(
            manager
                .calculate_hedge_order(dec!(1))
                .map(|hedge| hedge.is_some()),
            Err(expected)
        );
    }
}
