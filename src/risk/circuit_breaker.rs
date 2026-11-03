//! Circuit breaker implementation for automatic trading halts.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};
use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reason why the circuit breaker was triggered.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TriggerReason {
    /// Daily loss limit exceeded.
    MaxDailyLoss,
    /// Market volatility exceeded threshold.
    VolatilitySpike,
    /// Too many consecutive losing trades.
    ConsecutiveLosses,
    /// Equity dropped too fast within time window.
    RapidDrawdown,
    /// Manually triggered by operator.
    Manual,
}

impl std::fmt::Display for TriggerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxDailyLoss => write!(f, "max daily loss exceeded"),
            Self::VolatilitySpike => write!(f, "volatility spike detected"),
            Self::ConsecutiveLosses => write!(f, "consecutive losses limit reached"),
            Self::RapidDrawdown => write!(f, "rapid drawdown detected"),
            Self::Manual => write!(f, "manually triggered"),
        }
    }
}

/// Current state of the circuit breaker.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CircuitBreakerState {
    /// Normal operation, trading allowed.
    Active,
    /// Trading halted due to trigger condition.
    Triggered {
        /// Reason for the trigger.
        reason: TriggerReason,
        /// Timestamp when triggered, in milliseconds since Unix epoch.
        triggered_at: u64,
    },
    /// Cooling down before resuming trading.
    Cooldown {
        /// Timestamp when trading can resume, in milliseconds since Unix epoch.
        resume_at: u64,
    },
}

impl CircuitBreakerState {
    /// Returns true if the state is `Active`.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns true if the state is `Triggered`.
    #[must_use]
    pub fn is_triggered(&self) -> bool {
        matches!(self, Self::Triggered { .. })
    }

    /// Returns true if the state is `Cooldown`.
    #[must_use]
    pub fn is_cooldown(&self) -> bool {
        matches!(self, Self::Cooldown { .. })
    }
}

/// Configuration for the circuit breaker.
///
/// All thresholds are configurable to match different risk tolerances
/// and trading strategies.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::CircuitBreakerConfig;
/// use market_maker_rs::dec;
///
/// let config = CircuitBreakerConfig::new(
///     dec!(1000.0),  // max $1,000 daily loss
///     dec!(0.05),    // max 5% volatility
///     5,             // max 5 consecutive losses
///     dec!(0.10),    // 10% rapid drawdown threshold
///     300_000,       // 5 minute drawdown window (ms)
///     60_000,        // 1 minute cooldown (ms)
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CircuitBreakerConfig {
    /// Maximum allowed daily loss in currency units.
    ///
    /// When cumulative daily losses exceed this threshold, trading halts.
    pub max_daily_loss: Decimal,

    /// Maximum allowed volatility (as decimal, e.g., 0.05 for 5%).
    ///
    /// When market volatility exceeds this, trading halts.
    pub max_volatility: Decimal,

    /// Maximum consecutive losing trades before halt.
    ///
    /// After this many losses in a row, trading halts.
    pub max_consecutive_losses: u32,

    /// Rapid drawdown threshold (as decimal, e.g., 0.10 for 10%).
    ///
    /// If equity drops by this percentage within the drawdown window, trading halts.
    pub rapid_drawdown_threshold: Decimal,

    /// Time window for rapid drawdown detection, in milliseconds.
    ///
    /// Drawdown is measured over this rolling window.
    pub rapid_drawdown_window_ms: u64,

    /// Cooldown duration after trigger, in milliseconds.
    ///
    /// After being triggered, the breaker enters cooldown for this duration.
    pub cooldown_duration_ms: u64,
}

impl CircuitBreakerConfig {
    /// Creates a new `CircuitBreakerConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `max_daily_loss` - Maximum daily loss (must be positive)
    /// * `max_volatility` - Maximum volatility threshold (must be positive)
    /// * `max_consecutive_losses` - Maximum consecutive losses (must be > 0)
    /// * `rapid_drawdown_threshold` - Drawdown threshold (must be in (0, 1])
    /// * `rapid_drawdown_window_ms` - Drawdown window in milliseconds (must be > 0)
    /// * `cooldown_duration_ms` - Cooldown duration in milliseconds
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if any parameter is invalid.
    pub fn new(
        max_daily_loss: Decimal,
        max_volatility: Decimal,
        max_consecutive_losses: u32,
        rapid_drawdown_threshold: Decimal,
        rapid_drawdown_window_ms: u64,
        cooldown_duration_ms: u64,
    ) -> MMResult<Self> {
        if max_daily_loss <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "max_daily_loss must be positive".to_string(),
            ));
        }

        if max_volatility <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "max_volatility must be positive".to_string(),
            ));
        }

        if max_consecutive_losses == 0 {
            return Err(MMError::InvalidConfiguration(
                "max_consecutive_losses must be greater than 0".to_string(),
            ));
        }

        if rapid_drawdown_threshold <= Decimal::ZERO || rapid_drawdown_threshold > Decimal::ONE {
            return Err(MMError::InvalidConfiguration(
                "rapid_drawdown_threshold must be between 0 and 1 (exclusive of 0)".to_string(),
            ));
        }

        if rapid_drawdown_window_ms == 0 {
            return Err(MMError::InvalidConfiguration(
                "rapid_drawdown_window_ms must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            max_daily_loss,
            max_volatility,
            max_consecutive_losses,
            rapid_drawdown_threshold,
            rapid_drawdown_window_ms,
            cooldown_duration_ms,
        })
    }
}

/// Equity snapshot for drawdown tracking.
#[derive(Debug, Clone)]
struct EquitySnapshot {
    /// Equity value at this point.
    equity: Decimal,
    /// Timestamp in milliseconds.
    timestamp: u64,
}

/// Circuit breaker for automatic trading halts.
///
/// Monitors trading activity and halts operations when adverse conditions
/// are detected. Supports multiple trigger conditions and automatic cooldown.
///
/// # State Machine
///
/// ```text
/// Active -> Triggered (on breach) -> Cooldown (after acknowledgment) -> Active
/// ```
///
/// # Example
///
/// ```rust
/// use market_maker_rs::risk::{CircuitBreaker, CircuitBreakerConfig};
/// use market_maker_rs::dec;
///
/// let config = CircuitBreakerConfig::new(
///     dec!(1000.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000
/// ).unwrap();
///
/// let mut breaker = CircuitBreaker::new(config);
///
/// // Record some losing trades
/// breaker.record_trade(dec!(-100.0), 1000);
/// breaker.record_trade(dec!(-100.0), 2000);
///
/// assert!(breaker.is_trading_allowed());
///
/// // After too many losses, trading halts
/// for i in 0..5 {
///     breaker.record_trade(dec!(-100.0), 3000 + i as u64 * 1000);
/// }
/// // Daily loss of $700 exceeds nothing yet, but 5 consecutive losses triggers
/// ```
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Configuration parameters.
    config: CircuitBreakerConfig,
    /// Current state.
    state: CircuitBreakerState,
    /// Cumulative daily loss (positive value = loss).
    daily_loss: Decimal,
    /// Current consecutive losing trades count.
    consecutive_losses: u32,
    /// Current market volatility.
    current_volatility: Decimal,
    /// Equity history for drawdown calculation.
    equity_history: VecDeque<EquitySnapshot>,
    /// Peak equity for drawdown calculation.
    peak_equity: Decimal,
    /// Current equity.
    current_equity: Decimal,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Circuit breaker configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::{CircuitBreaker, CircuitBreakerConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = CircuitBreakerConfig::new(
    ///     dec!(1000.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000
    /// ).unwrap();
    ///
    /// let breaker = CircuitBreaker::new(config);
    /// assert!(breaker.is_trading_allowed());
    /// ```
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitBreakerState::Active,
            daily_loss: Decimal::ZERO,
            consecutive_losses: 0,
            current_volatility: Decimal::ZERO,
            equity_history: VecDeque::new(),
            peak_equity: Decimal::ZERO,
            current_equity: Decimal::ZERO,
        }
    }

    /// Creates a new circuit breaker with initial equity.
    ///
    /// # Arguments
    ///
    /// * `config` - Circuit breaker configuration
    /// * `initial_equity` - Starting equity for drawdown calculations
    /// * `timestamp` - Initial timestamp in milliseconds
    #[must_use]
    pub fn with_initial_equity(
        config: CircuitBreakerConfig,
        initial_equity: Decimal,
        timestamp: u64,
    ) -> Self {
        let mut breaker = Self::new(config);
        breaker.current_equity = initial_equity;
        breaker.peak_equity = initial_equity;
        breaker.equity_history.push_back(EquitySnapshot {
            equity: initial_equity,
            timestamp,
        });
        breaker
    }

    /// Returns the current state of the circuit breaker.
    #[must_use]
    pub fn state(&self) -> &CircuitBreakerState {
        &self.state
    }

    /// Returns the current daily loss.
    #[must_use]
    pub fn daily_loss(&self) -> Decimal {
        self.daily_loss
    }

    /// Returns the current consecutive losses count.
    #[must_use]
    pub fn consecutive_losses(&self) -> u32 {
        self.consecutive_losses
    }

    /// Checks if trading is currently allowed.
    ///
    /// Trading is only allowed when the circuit breaker is in `Active` state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::{CircuitBreaker, CircuitBreakerConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = CircuitBreakerConfig::new(
    ///     dec!(1000.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000
    /// ).unwrap();
    ///
    /// let breaker = CircuitBreaker::new(config);
    /// assert!(breaker.is_trading_allowed());
    /// ```
    #[must_use]
    pub fn is_trading_allowed(&self) -> bool {
        self.state.is_active()
    }

    /// Records a trade result and checks for trigger conditions.
    ///
    /// # Arguments
    ///
    /// * `pnl` - Profit/loss from the trade (negative = loss)
    /// * `timestamp` - Timestamp of the trade in milliseconds
    ///
    /// # Returns
    ///
    /// The current state after recording the trade.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::risk::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState};
    /// use market_maker_rs::dec;
    ///
    /// let config = CircuitBreakerConfig::new(
    ///     dec!(100.0), dec!(0.05), 3, dec!(0.10), 300_000, 60_000
    /// ).unwrap();
    ///
    /// let mut breaker = CircuitBreaker::new(config);
    ///
    /// // Three consecutive losses trigger the breaker
    /// breaker.record_trade(dec!(-10.0), 1000);
    /// breaker.record_trade(dec!(-10.0), 2000);
    /// let state = breaker.record_trade(dec!(-10.0), 3000);
    ///
    /// assert!(state.is_triggered());
    /// ```
    pub fn record_trade(&mut self, pnl: Decimal, timestamp: u64) -> CircuitBreakerState {
        // Don't record if already triggered
        if !self.state.is_active() {
            return self.state.clone();
        }

        // Update daily loss. This method cannot fail, so the sums saturate
        // at the `Decimal` bounds; a saturated loss trips the limits.
        if pnl < Decimal::ZERO {
            self.daily_loss = self.daily_loss.saturating_add(pnl.abs());
            self.consecutive_losses = self.consecutive_losses.saturating_add(1);
        } else {
            self.consecutive_losses = 0;
        }

        // Update equity
        self.current_equity = self.current_equity.saturating_add(pnl);
        if self.current_equity > self.peak_equity {
            self.peak_equity = self.current_equity;
        }

        // Add to equity history
        self.equity_history.push_back(EquitySnapshot {
            equity: self.current_equity,
            timestamp,
        });

        // Prune old equity history
        self.prune_equity_history(timestamp);

        // Check triggers
        self.check_triggers(timestamp);

        self.state.clone()
    }

    /// Updates the current market volatility.
    ///
    /// # Arguments
    ///
    /// * `volatility` - Current market volatility (as decimal)
    /// * `timestamp` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// The current state after updating volatility.
    pub fn update_volatility(
        &mut self,
        volatility: Decimal,
        timestamp: u64,
    ) -> CircuitBreakerState {
        self.current_volatility = volatility;

        if self.state.is_active() {
            self.check_triggers(timestamp);
        }

        self.state.clone()
    }

    /// Manually triggers the circuit breaker.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn trigger_manual(&mut self, timestamp: u64) {
        self.state = CircuitBreakerState::Triggered {
            reason: TriggerReason::Manual,
            triggered_at: timestamp,
        };
    }

    /// Starts the cooldown period after a trigger.
    ///
    /// Call this to transition from `Triggered` to `Cooldown` state.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn start_cooldown(&mut self, timestamp: u64) {
        if self.state.is_triggered() {
            self.state = CircuitBreakerState::Cooldown {
                // Saturates: an overflowing cooldown never resumes (fails closed).
                resume_at: timestamp.saturating_add(self.config.cooldown_duration_ms),
            };
        }
    }

    /// Checks if cooldown has expired and transitions to Active if so.
    ///
    /// # Arguments
    ///
    /// * `current_time` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// The current state after checking cooldown.
    pub fn check_cooldown(&mut self, current_time: u64) -> CircuitBreakerState {
        if let CircuitBreakerState::Cooldown { resume_at } = self.state
            && current_time >= resume_at
        {
            self.state = CircuitBreakerState::Active;
        }
        self.state.clone()
    }

    /// Resets the circuit breaker to initial state.
    ///
    /// Typically called at the start of a new trading day.
    ///
    /// # Arguments
    ///
    /// * `initial_equity` - Starting equity for the new period
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn reset(&mut self, initial_equity: Decimal, timestamp: u64) {
        self.state = CircuitBreakerState::Active;
        self.daily_loss = Decimal::ZERO;
        self.consecutive_losses = 0;
        self.current_volatility = Decimal::ZERO;
        self.equity_history.clear();
        self.peak_equity = initial_equity;
        self.current_equity = initial_equity;
        self.equity_history.push_back(EquitySnapshot {
            equity: initial_equity,
            timestamp,
        });
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Checks all trigger conditions and updates state if breached.
    fn check_triggers(&mut self, timestamp: u64) {
        // Check max daily loss
        if self.daily_loss >= self.config.max_daily_loss {
            self.state = CircuitBreakerState::Triggered {
                reason: TriggerReason::MaxDailyLoss,
                triggered_at: timestamp,
            };
            return;
        }

        // Check volatility spike
        if self.current_volatility > self.config.max_volatility {
            self.state = CircuitBreakerState::Triggered {
                reason: TriggerReason::VolatilitySpike,
                triggered_at: timestamp,
            };
            return;
        }

        // Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses {
            self.state = CircuitBreakerState::Triggered {
                reason: TriggerReason::ConsecutiveLosses,
                triggered_at: timestamp,
            };
            return;
        }

        // Check rapid drawdown
        if self.check_rapid_drawdown() {
            self.state = CircuitBreakerState::Triggered {
                reason: TriggerReason::RapidDrawdown,
                triggered_at: timestamp,
            };
        }
    }

    /// Checks if rapid drawdown threshold has been breached.
    fn check_rapid_drawdown(&self) -> bool {
        if self.equity_history.is_empty() {
            return false;
        }

        // Find the peak equity in the window
        let window_peak = self
            .equity_history
            .iter()
            .map(|s| s.equity)
            .max()
            .unwrap_or(self.current_equity);

        if window_peak <= Decimal::ZERO {
            return false;
        }

        // This check cannot fail, so the arithmetic saturates; a drawdown
        // beyond the `Decimal` range trips the breaker (fail closed).
        // Computed as `1 - equity / peak` so no intermediate is clamped.
        let drawdown = self
            .current_equity
            .checked_div(window_peak)
            .and_then(|ratio| Decimal::ONE.checked_sub(ratio))
            .unwrap_or(if self.current_equity.is_sign_negative() {
                Decimal::MAX
            } else {
                Decimal::MIN
            });
        drawdown >= self.config.rapid_drawdown_threshold
    }

    /// Removes equity snapshots older than the drawdown window.
    fn prune_equity_history(&mut self, current_time: u64) {
        let cutoff = current_time.saturating_sub(self.config.rapid_drawdown_window_ms);
        while let Some(front) = self.equity_history.front() {
            if front.timestamp < cutoff {
                self.equity_history.pop_front();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn default_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig::new(
            dec!(1000.0), // max daily loss
            dec!(0.05),   // max volatility
            5,            // max consecutive losses
            dec!(0.10),   // rapid drawdown threshold
            300_000,      // 5 minute window
            60_000,       // 1 minute cooldown
        )
        .unwrap()
    }

    #[test]
    fn test_config_valid() {
        let config = default_config();
        assert_eq!(config.max_daily_loss, dec!(1000.0));
        assert_eq!(config.max_consecutive_losses, 5);
    }

    #[test]
    fn test_config_invalid_max_daily_loss() {
        let result =
            CircuitBreakerConfig::new(dec!(0.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000);
        assert!(result.is_err());

        let result =
            CircuitBreakerConfig::new(dec!(-100.0), dec!(0.05), 5, dec!(0.10), 300_000, 60_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_max_volatility() {
        let result =
            CircuitBreakerConfig::new(dec!(1000.0), dec!(0.0), 5, dec!(0.10), 300_000, 60_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_max_consecutive_losses() {
        let result =
            CircuitBreakerConfig::new(dec!(1000.0), dec!(0.05), 0, dec!(0.10), 300_000, 60_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_rapid_drawdown_threshold() {
        let result =
            CircuitBreakerConfig::new(dec!(1000.0), dec!(0.05), 5, dec!(0.0), 300_000, 60_000);
        assert!(result.is_err());

        let result =
            CircuitBreakerConfig::new(dec!(1000.0), dec!(0.05), 5, dec!(1.1), 300_000, 60_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_rapid_drawdown_window() {
        let result = CircuitBreakerConfig::new(dec!(1000.0), dec!(0.05), 5, dec!(0.10), 0, 60_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_breaker_is_active() {
        let breaker = CircuitBreaker::new(default_config());
        assert!(breaker.is_trading_allowed());
        assert!(breaker.state().is_active());
    }

    #[test]
    fn test_with_initial_equity() {
        let breaker = CircuitBreaker::with_initial_equity(default_config(), dec!(10000.0), 0);
        assert!(breaker.is_trading_allowed());
        assert_eq!(breaker.current_equity, dec!(10000.0));
        assert_eq!(breaker.peak_equity, dec!(10000.0));
    }

    #[test]
    fn test_record_winning_trade() {
        let mut breaker = CircuitBreaker::new(default_config());
        breaker.record_trade(dec!(100.0), 1000);

        assert!(breaker.is_trading_allowed());
        assert_eq!(breaker.daily_loss(), dec!(0.0));
        assert_eq!(breaker.consecutive_losses(), 0);
    }

    #[test]
    fn test_record_losing_trade() {
        let mut breaker = CircuitBreaker::new(default_config());
        breaker.record_trade(dec!(-100.0), 1000);

        assert!(breaker.is_trading_allowed());
        assert_eq!(breaker.daily_loss(), dec!(100.0));
        assert_eq!(breaker.consecutive_losses(), 1);
    }

    #[test]
    fn test_consecutive_losses_reset_on_win() {
        let mut breaker = CircuitBreaker::new(default_config());

        breaker.record_trade(dec!(-100.0), 1000);
        breaker.record_trade(dec!(-100.0), 2000);
        assert_eq!(breaker.consecutive_losses(), 2);

        breaker.record_trade(dec!(50.0), 3000);
        assert_eq!(breaker.consecutive_losses(), 0);
    }

    #[test]
    fn test_trigger_max_daily_loss() {
        let config =
            CircuitBreakerConfig::new(dec!(100.0), dec!(0.05), 10, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut breaker = CircuitBreaker::new(config);

        // Loss of $100 should trigger
        let state = breaker.record_trade(dec!(-100.0), 1000);

        assert!(state.is_triggered());
        assert!(!breaker.is_trading_allowed());

        if let CircuitBreakerState::Triggered { reason, .. } = state {
            assert_eq!(reason, TriggerReason::MaxDailyLoss);
        } else {
            panic!("Expected Triggered state");
        }
    }

    #[test]
    fn test_trigger_consecutive_losses() {
        let config =
            CircuitBreakerConfig::new(dec!(10000.0), dec!(0.05), 3, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_trade(dec!(-10.0), 1000);
        breaker.record_trade(dec!(-10.0), 2000);
        assert!(breaker.is_trading_allowed());

        let state = breaker.record_trade(dec!(-10.0), 3000);
        assert!(state.is_triggered());

        if let CircuitBreakerState::Triggered { reason, .. } = state {
            assert_eq!(reason, TriggerReason::ConsecutiveLosses);
        }
    }

    #[test]
    fn test_trigger_volatility_spike() {
        let mut breaker = CircuitBreaker::new(default_config());

        // Volatility below threshold
        breaker.update_volatility(dec!(0.04), 1000);
        assert!(breaker.is_trading_allowed());

        // Volatility above threshold
        let state = breaker.update_volatility(dec!(0.06), 2000);
        assert!(state.is_triggered());

        if let CircuitBreakerState::Triggered { reason, .. } = state {
            assert_eq!(reason, TriggerReason::VolatilitySpike);
        }
    }

    #[test]
    fn test_trigger_rapid_drawdown() {
        let config =
            CircuitBreakerConfig::new(dec!(10000.0), dec!(1.0), 100, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut breaker = CircuitBreaker::with_initial_equity(config, dec!(1000.0), 0);

        // 10% drawdown should trigger
        let state = breaker.record_trade(dec!(-100.0), 1000);

        assert!(state.is_triggered());
        if let CircuitBreakerState::Triggered { reason, .. } = state {
            assert_eq!(reason, TriggerReason::RapidDrawdown);
        }
    }

    #[test]
    fn test_record_trade_extreme_loss_saturates_and_triggers() {
        let config =
            CircuitBreakerConfig::new(Decimal::MAX, dec!(1.0), 100, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut breaker =
            CircuitBreaker::with_initial_equity(config, dec!(0.0000000000000000000000000001), 0);

        breaker.record_trade(Decimal::MIN, 1000);
        let state = breaker.record_trade(Decimal::MIN, 2000);

        assert!(state.is_triggered());
    }

    #[test]
    fn test_manual_trigger() {
        let mut breaker = CircuitBreaker::new(default_config());

        breaker.trigger_manual(1000);

        assert!(!breaker.is_trading_allowed());
        if let CircuitBreakerState::Triggered {
            reason,
            triggered_at,
        } = breaker.state()
        {
            assert_eq!(*reason, TriggerReason::Manual);
            assert_eq!(*triggered_at, 1000);
        } else {
            panic!("Expected Triggered state");
        }
    }

    #[test]
    fn test_cooldown_transition() {
        let mut breaker = CircuitBreaker::new(default_config());

        breaker.trigger_manual(1000);
        assert!(breaker.state().is_triggered());

        breaker.start_cooldown(2000);
        assert!(breaker.state().is_cooldown());

        if let CircuitBreakerState::Cooldown { resume_at } = breaker.state() {
            assert_eq!(*resume_at, 2000 + 60_000); // cooldown is 60_000ms
        }
    }

    #[test]
    fn test_start_cooldown_max_duration_saturates_and_stays_closed() {
        let mut config = default_config();
        config.cooldown_duration_ms = u64::MAX;
        let mut breaker = CircuitBreaker::new(config);
        breaker.trigger_manual(1000);
        breaker.start_cooldown(2000);
        assert_eq!(
            breaker.state(),
            &CircuitBreakerState::Cooldown {
                resume_at: u64::MAX
            }
        );
        breaker.check_cooldown(u64::MAX - 1);
        assert!(!breaker.is_trading_allowed());
    }

    #[test]
    fn test_cooldown_expiry() {
        let mut breaker = CircuitBreaker::new(default_config());

        breaker.trigger_manual(1000);
        breaker.start_cooldown(2000);

        // Before cooldown expires
        breaker.check_cooldown(50_000);
        assert!(breaker.state().is_cooldown());

        // After cooldown expires
        breaker.check_cooldown(70_000);
        assert!(breaker.state().is_active());
        assert!(breaker.is_trading_allowed());
    }

    #[test]
    fn test_reset() {
        let mut breaker = CircuitBreaker::new(default_config());

        breaker.record_trade(dec!(-500.0), 1000);
        breaker.record_trade(dec!(-500.0), 2000);
        breaker.trigger_manual(3000);

        assert!(!breaker.is_trading_allowed());
        assert_eq!(breaker.daily_loss(), dec!(1000.0));

        breaker.reset(dec!(10000.0), 100_000);

        assert!(breaker.is_trading_allowed());
        assert_eq!(breaker.daily_loss(), dec!(0.0));
        assert_eq!(breaker.consecutive_losses(), 0);
    }

    #[test]
    fn test_no_recording_when_triggered() {
        let config =
            CircuitBreakerConfig::new(dec!(100.0), dec!(0.05), 10, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_trade(dec!(-100.0), 1000); // Triggers
        assert!(!breaker.is_trading_allowed());

        let loss_before = breaker.daily_loss();
        breaker.record_trade(dec!(-100.0), 2000); // Should be ignored

        assert_eq!(breaker.daily_loss(), loss_before);
    }

    #[test]
    fn test_state_helper_methods() {
        let active = CircuitBreakerState::Active;
        assert!(active.is_active());
        assert!(!active.is_triggered());
        assert!(!active.is_cooldown());

        let triggered = CircuitBreakerState::Triggered {
            reason: TriggerReason::Manual,
            triggered_at: 0,
        };
        assert!(!triggered.is_active());
        assert!(triggered.is_triggered());
        assert!(!triggered.is_cooldown());

        let cooldown = CircuitBreakerState::Cooldown { resume_at: 1000 };
        assert!(!cooldown.is_active());
        assert!(!cooldown.is_triggered());
        assert!(cooldown.is_cooldown());
    }

    #[test]
    fn test_trigger_reason_display() {
        assert_eq!(
            TriggerReason::MaxDailyLoss.to_string(),
            "max daily loss exceeded"
        );
        assert_eq!(
            TriggerReason::VolatilitySpike.to_string(),
            "volatility spike detected"
        );
        assert_eq!(
            TriggerReason::ConsecutiveLosses.to_string(),
            "consecutive losses limit reached"
        );
        assert_eq!(
            TriggerReason::RapidDrawdown.to_string(),
            "rapid drawdown detected"
        );
        assert_eq!(TriggerReason::Manual.to_string(), "manually triggered");
    }

    #[test]
    fn test_equity_history_pruning() {
        let config = CircuitBreakerConfig::new(
            dec!(10000.0),
            dec!(1.0),
            100,
            dec!(0.50),
            100,
            60_000, // 100ms window
        )
        .unwrap();
        let mut breaker = CircuitBreaker::with_initial_equity(config, dec!(1000.0), 0);

        breaker.record_trade(dec!(-10.0), 50);
        breaker.record_trade(dec!(-10.0), 100);
        breaker.record_trade(dec!(-10.0), 200); // Should prune entries before 100

        // Old entries should be pruned
        assert!(breaker.equity_history.len() <= 3);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config = default_config();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CircuitBreakerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);

        let state = CircuitBreakerState::Triggered {
            reason: TriggerReason::MaxDailyLoss,
            triggered_at: 1000,
        };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: CircuitBreakerState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }
}
