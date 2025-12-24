//! Guéant-Lehalle-Fernandez-Tapia (GLFT) model extension.
//!
//! This module extends the Avellaneda-Stoikov model with terminal inventory penalty
//! from the GLFT framework, making it more realistic for strategies that must
//! flatten positions at session end.
//!
//! # Reference
//!
//! Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2012).
//! "Dealing with the inventory risk: a solution to the market making problem."
//! Mathematics and Financial Economics, 7(4), 477-507.
//!
//! # Mathematical Model
//!
//! The GLFT model modifies the Avellaneda-Stoikov reservation price:
//!
//! ## Original A-S Reservation Price
//! ```text
//! r = s - q * γ * σ² * (T - t)
//! ```
//!
//! ## GLFT Modification
//! ```text
//! r = s - q * γ * σ² * (T - t) - q * φ * f(T - t)
//! ```
//!
//! Where:
//! - `φ` (phi): Terminal inventory penalty parameter
//! - `f(T-t)`: Penalty function that increases as terminal approaches
//!
//! # Dynamic Risk Aversion
//!
//! The model also supports dynamic gamma scaling:
//! ```text
//! γ_t = γ_0 * (1 + α * (1 - (T-t)/T))
//! ```
//!
//! Where `α` controls how much gamma increases near terminal time.

use crate::Decimal;
use crate::types::decimal::{decimal_ln, decimal_powi};
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const SECONDS_PER_MILLISECOND: Decimal = Decimal::from_parts(1, 0, 0, false, 3); // 0.001
const SECONDS_PER_YEAR: Decimal = Decimal::from_parts(31_536_000, 0, 0, false, 0);

/// Penalty function type for terminal inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PenaltyFunction {
    /// Linear penalty: f(τ) = 1 - τ/T (increases linearly as terminal approaches).
    #[default]
    Linear,
    /// Exponential penalty: f(τ) = exp(-τ/T) (faster increase near terminal).
    Exponential,
    /// Quadratic penalty: f(τ) = (1 - τ/T)² (slower start, faster end).
    Quadratic,
}

/// Configuration for the GLFT strategy.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::glft::GLFTConfig;
/// use market_maker_rs::dec;
///
/// let config = GLFTConfig::new(
///     dec!(0.1),    // risk_aversion (gamma)
///     dec!(1.5),    // order_intensity (k)
///     dec!(0.05),   // terminal_penalty (phi)
///     3_600_000,     // terminal_time (1 hour in ms)
///     dec!(0.0001), // min_spread
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GLFTConfig {
    /// Base risk aversion parameter (gamma).
    pub risk_aversion: Decimal,

    /// Order intensity parameter (k).
    pub order_intensity: Decimal,

    /// Terminal inventory penalty parameter (phi).
    pub terminal_penalty: Decimal,

    /// Terminal time in milliseconds from session start.
    pub terminal_time: u64,

    /// Minimum spread constraint.
    pub min_spread: Decimal,

    /// Whether to use dynamic gamma scaling.
    pub dynamic_gamma: bool,

    /// Dynamic gamma scaling factor (alpha).
    /// Only used if `dynamic_gamma` is true.
    pub gamma_scaling_factor: Decimal,

    /// Penalty function type.
    pub penalty_function: PenaltyFunction,
}

impl GLFTConfig {
    /// Creates a new `GLFTConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `risk_aversion` - Base risk aversion (gamma), must be positive
    /// * `order_intensity` - Order intensity (k), must be positive
    /// * `terminal_penalty` - Terminal inventory penalty (phi), must be non-negative
    /// * `terminal_time` - Terminal time in milliseconds, must be positive
    /// * `min_spread` - Minimum spread constraint, must be non-negative
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::glft::GLFTConfig;
    /// use market_maker_rs::dec;
    ///
    /// let config = GLFTConfig::new(
    ///     dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)
    /// ).unwrap();
    /// ```
    pub fn new(
        risk_aversion: Decimal,
        order_intensity: Decimal,
        terminal_penalty: Decimal,
        terminal_time: u64,
        min_spread: Decimal,
    ) -> MMResult<Self> {
        if risk_aversion <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "risk_aversion must be positive".to_string(),
            ));
        }

        if order_intensity <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "order_intensity must be positive".to_string(),
            ));
        }

        if terminal_penalty < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "terminal_penalty must be non-negative".to_string(),
            ));
        }

        if terminal_time == 0 {
            return Err(MMError::InvalidConfiguration(
                "terminal_time must be positive".to_string(),
            ));
        }

        if min_spread < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(
                "min_spread must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            risk_aversion,
            order_intensity,
            terminal_penalty,
            terminal_time,
            min_spread,
            dynamic_gamma: false,
            gamma_scaling_factor: Decimal::ONE,
            penalty_function: PenaltyFunction::default(),
        })
    }

    /// Enables dynamic gamma scaling.
    ///
    /// # Arguments
    ///
    /// * `scaling_factor` - Alpha parameter for gamma scaling (typically 0.5 to 2.0)
    #[must_use]
    pub fn with_dynamic_gamma(mut self, scaling_factor: Decimal) -> Self {
        self.dynamic_gamma = true;
        self.gamma_scaling_factor = scaling_factor;
        self
    }

    /// Sets the penalty function type.
    #[must_use]
    pub fn with_penalty_function(mut self, penalty_function: PenaltyFunction) -> Self {
        self.penalty_function = penalty_function;
        self
    }
}

/// GLFT strategy implementation.
///
/// Extends the Avellaneda-Stoikov model with terminal inventory penalty.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::glft::{GLFTStrategy, GLFTConfig};
/// use market_maker_rs::dec;
///
/// let config = GLFTConfig::new(
///     dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)
/// ).unwrap();
///
/// let reservation = GLFTStrategy::calculate_reservation_price(
///     dec!(100.0),  // mid_price
///     dec!(10.0),   // inventory
///     &config,
///     dec!(0.2),    // volatility
///     1_800_000,     // current_time (halfway through session)
/// ).unwrap();
///
/// // With positive inventory and terminal penalty, reservation < mid_price
/// assert!(reservation < dec!(100.0));
/// ```
pub struct GLFTStrategy;

impl GLFTStrategy {
    /// Calculates the reservation price with terminal inventory penalty.
    ///
    /// # Formula
    ///
    /// ```text
    /// r = s - q * γ_t * σ² * τ - q * φ * f(τ)
    /// ```
    ///
    /// Where:
    /// - `s`: mid price
    /// - `q`: inventory
    /// - `γ_t`: effective gamma (possibly dynamic)
    /// - `σ`: volatility
    /// - `τ`: time to terminal (T - t)
    /// - `φ`: terminal penalty
    /// - `f(τ)`: penalty function
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid price
    /// * `inventory` - Current inventory position
    /// * `config` - GLFT configuration
    /// * `volatility` - Annualized volatility
    /// * `current_time` - Current time in milliseconds from session start
    ///
    /// # Errors
    ///
    /// Returns error if inputs are invalid.
    pub fn calculate_reservation_price(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<Decimal> {
        if mid_price <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(
                "mid_price must be positive".to_string(),
            ));
        }

        if volatility <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(
                "volatility must be positive".to_string(),
            ));
        }

        let time_to_terminal_ms = config.terminal_time.saturating_sub(current_time);

        // Get effective gamma (possibly dynamic)
        let effective_gamma = Self::calculate_dynamic_gamma(
            config.risk_aversion,
            time_to_terminal_ms,
            config.terminal_time,
            config.dynamic_gamma,
            config.gamma_scaling_factor,
        );

        // Convert time to years
        let time_to_terminal_years = Self::ms_to_years(time_to_terminal_ms);

        // Standard A-S term: q * γ * σ² * τ
        let volatility_squared = decimal_powi(volatility, 2)?;
        let as_adjustment =
            inventory * effective_gamma * volatility_squared * time_to_terminal_years;

        // Terminal penalty term: q * φ * f(τ)
        let penalty_value = Self::calculate_penalty_function(
            time_to_terminal_ms,
            config.terminal_time,
            config.penalty_function,
        );
        let terminal_adjustment = inventory * config.terminal_penalty * penalty_value;

        // r = s - as_adjustment - terminal_adjustment
        let reservation_price = mid_price - as_adjustment - terminal_adjustment;

        Ok(reservation_price)
    }

    /// Calculates the optimal spread with GLFT adjustments.
    ///
    /// # Formula
    ///
    /// ```text
    /// δ = max(min_spread, γ_t * σ² * τ + (2/γ_t) * ln(1 + γ_t/k))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `config` - GLFT configuration
    /// * `volatility` - Annualized volatility
    /// * `current_time` - Current time in milliseconds from session start
    ///
    /// # Errors
    ///
    /// Returns error if inputs are invalid.
    pub fn calculate_optimal_spread(
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<Decimal> {
        if volatility <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(
                "volatility must be positive".to_string(),
            ));
        }

        let time_to_terminal_ms = config.terminal_time.saturating_sub(current_time);

        // Get effective gamma
        let effective_gamma = Self::calculate_dynamic_gamma(
            config.risk_aversion,
            time_to_terminal_ms,
            config.terminal_time,
            config.dynamic_gamma,
            config.gamma_scaling_factor,
        );

        // Convert time to years
        let time_to_terminal_years = Self::ms_to_years(time_to_terminal_ms);

        // Inventory risk term: γ * σ² * τ
        let volatility_squared = decimal_powi(volatility, 2)?;
        let inventory_risk_term = effective_gamma * volatility_squared * time_to_terminal_years;

        // Adverse selection term: (2/γ) * ln(1 + γ/k)
        let adverse_selection_inner = Decimal::ONE + effective_gamma / config.order_intensity;
        let adverse_selection_ln = decimal_ln(adverse_selection_inner)?;
        let adverse_selection_term = (Decimal::TWO / effective_gamma) * adverse_selection_ln;

        let spread = inventory_risk_term + adverse_selection_term;

        // Apply minimum spread constraint
        Ok(spread.max(config.min_spread))
    }

    /// Calculates optimal bid and ask quotes.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid price
    /// * `inventory` - Current inventory position
    /// * `config` - GLFT configuration
    /// * `volatility` - Annualized volatility
    /// * `current_time` - Current time in milliseconds from session start
    ///
    /// # Returns
    ///
    /// Tuple of (bid_price, ask_price).
    ///
    /// # Errors
    ///
    /// Returns error if inputs are invalid or quotes would be invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::glft::{GLFTStrategy, GLFTConfig};
    /// use market_maker_rs::dec;
    ///
    /// let config = GLFTConfig::new(
    ///     dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)
    /// ).unwrap();
    ///
    /// let (bid, ask) = GLFTStrategy::calculate_optimal_quotes(
    ///     dec!(100.0), dec!(0.0), &config, dec!(0.2), 0
    /// ).unwrap();
    ///
    /// assert!(bid < ask);
    /// ```
    pub fn calculate_optimal_quotes(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<(Decimal, Decimal)> {
        let reservation_price = Self::calculate_reservation_price(
            mid_price,
            inventory,
            config,
            volatility,
            current_time,
        )?;

        let spread = Self::calculate_optimal_spread(config, volatility, current_time)?;

        let half_spread = spread / Decimal::TWO;
        let bid_price = reservation_price - half_spread;
        let ask_price = reservation_price + half_spread;

        // Validate quotes
        if bid_price >= ask_price {
            return Err(MMError::InvalidQuoteGeneration(
                "bid price must be less than ask price".to_string(),
            ));
        }

        if bid_price <= Decimal::ZERO {
            return Err(MMError::InvalidQuoteGeneration(
                "bid price must be positive".to_string(),
            ));
        }

        Ok((bid_price, ask_price))
    }

    /// Calculates dynamic gamma based on time to terminal.
    ///
    /// # Formula
    ///
    /// ```text
    /// γ_t = γ_0 * (1 + α * (1 - τ/T))
    /// ```
    ///
    /// Where:
    /// - `γ_0`: base gamma
    /// - `α`: scaling factor
    /// - `τ`: time to terminal
    /// - `T`: total session time
    ///
    /// # Arguments
    ///
    /// * `base_gamma` - Base risk aversion parameter
    /// * `time_to_terminal_ms` - Time remaining until terminal
    /// * `total_session_ms` - Total session duration
    ///
    /// # Example
    ///
    /// ```rust
    /// use market_maker_rs::strategy::glft::GLFTStrategy;
    /// use market_maker_rs::dec;
    ///
    /// // At session start (full time remaining), gamma = base
    /// let gamma_start = GLFTStrategy::calculate_dynamic_gamma(
    ///     dec!(0.1), 3_600_000, 3_600_000, true, dec!(1.0)
    /// );
    /// assert_eq!(gamma_start, dec!(0.1));
    ///
    /// // At session end (no time remaining), gamma = base * (1 + alpha)
    /// let gamma_end = GLFTStrategy::calculate_dynamic_gamma(
    ///     dec!(0.1), 0, 3_600_000, true, dec!(1.0)
    /// );
    /// assert_eq!(gamma_end, dec!(0.2)); // 0.1 * (1 + 1.0)
    /// ```
    #[must_use]
    pub fn calculate_dynamic_gamma(
        base_gamma: Decimal,
        time_to_terminal_ms: u64,
        total_session_ms: u64,
        dynamic_enabled: bool,
        scaling_factor: Decimal,
    ) -> Decimal {
        if !dynamic_enabled || total_session_ms == 0 {
            return base_gamma;
        }

        let time_ratio = Decimal::from(time_to_terminal_ms) / Decimal::from(total_session_ms);
        let time_factor = Decimal::ONE - time_ratio;

        // γ_t = γ_0 * (1 + α * (1 - τ/T))
        base_gamma * (Decimal::ONE + scaling_factor * time_factor)
    }

    /// Calculates the penalty function value.
    ///
    /// # Arguments
    ///
    /// * `time_to_terminal_ms` - Time remaining until terminal
    /// * `total_session_ms` - Total session duration
    /// * `penalty_type` - Type of penalty function
    fn calculate_penalty_function(
        time_to_terminal_ms: u64,
        total_session_ms: u64,
        penalty_type: PenaltyFunction,
    ) -> Decimal {
        if total_session_ms == 0 {
            return Decimal::ONE;
        }

        let time_ratio = Decimal::from(time_to_terminal_ms) / Decimal::from(total_session_ms);

        match penalty_type {
            PenaltyFunction::Linear => {
                // f(τ) = 1 - τ/T
                Decimal::ONE - time_ratio
            }
            PenaltyFunction::Exponential => {
                // f(τ) = exp(-τ/T) ≈ 1 - τ/T + (τ/T)²/2 for small values
                // Using approximation to avoid exp function
                let neg_ratio = -time_ratio;
                // Simple approximation: e^x ≈ 1 + x + x²/2 for small x
                Decimal::ONE + neg_ratio + (neg_ratio * neg_ratio) / Decimal::TWO
            }
            PenaltyFunction::Quadratic => {
                // f(τ) = (1 - τ/T)²
                let factor = Decimal::ONE - time_ratio;
                factor * factor
            }
        }
    }

    /// Converts milliseconds to years.
    fn ms_to_years(ms: u64) -> Decimal {
        let ms_dec = Decimal::from(ms);
        (ms_dec * SECONDS_PER_MILLISECOND) / SECONDS_PER_YEAR
    }

    /// Compares GLFT quotes with standard A-S quotes.
    ///
    /// Useful for understanding the impact of terminal penalty.
    ///
    /// # Returns
    ///
    /// Tuple of ((glft_bid, glft_ask), (as_bid, as_ask)).
    pub fn compare_with_avellaneda_stoikov(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<((Decimal, Decimal), (Decimal, Decimal))> {
        // GLFT quotes
        let glft_quotes =
            Self::calculate_optimal_quotes(mid_price, inventory, config, volatility, current_time)?;

        // A-S quotes (using GLFT config but without terminal penalty)
        let as_config = GLFTConfig {
            terminal_penalty: Decimal::ZERO,
            dynamic_gamma: false,
            ..config.clone()
        };
        let as_quotes = Self::calculate_optimal_quotes(
            mid_price,
            inventory,
            &as_config,
            volatility,
            current_time,
        )?;

        Ok((glft_quotes, as_quotes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_config_valid() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001));
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_invalid_risk_aversion() {
        let config = GLFTConfig::new(dec!(0.0), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001));
        assert!(config.is_err());

        let config = GLFTConfig::new(dec!(-0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_order_intensity() {
        let config = GLFTConfig::new(dec!(0.1), dec!(0.0), dec!(0.05), 3_600_000, dec!(0.0001));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_terminal_penalty() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(-0.05), 3_600_000, dec!(0.0001));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_invalid_terminal_time() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 0, dec!(0.0001));
        assert!(config.is_err());
    }

    #[test]
    fn test_config_with_dynamic_gamma() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001))
            .unwrap()
            .with_dynamic_gamma(dec!(1.5));

        assert!(config.dynamic_gamma);
        assert_eq!(config.gamma_scaling_factor, dec!(1.5));
    }

    #[test]
    fn test_config_with_penalty_function() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001))
            .unwrap()
            .with_penalty_function(PenaltyFunction::Exponential);

        assert_eq!(config.penalty_function, PenaltyFunction::Exponential);
    }

    #[test]
    fn test_reservation_price_flat_inventory() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let reservation = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            Decimal::ZERO,
            &config,
            dec!(0.2),
            0,
        )
        .unwrap();

        // With flat inventory, reservation should equal mid_price
        assert!((reservation - dec!(100.0)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_reservation_price_long_inventory() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let reservation = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(10.0),
            &config,
            dec!(0.2),
            0,
        )
        .unwrap();

        // With positive inventory, reservation < mid_price
        assert!(reservation < dec!(100.0));
    }

    #[test]
    fn test_reservation_price_short_inventory() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let reservation = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(-10.0),
            &config,
            dec!(0.2),
            0,
        )
        .unwrap();

        // With negative inventory, reservation > mid_price
        assert!(reservation > dec!(100.0));
    }

    #[test]
    fn test_terminal_penalty_effect() {
        // Config with terminal penalty
        let config_with_penalty =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.1), 3_600_000, dec!(0.0001)).unwrap();

        // Config without terminal penalty
        let config_no_penalty =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.0), 3_600_000, dec!(0.0001)).unwrap();

        let reservation_with = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(10.0),
            &config_with_penalty,
            dec!(0.2),
            1_800_000, // Halfway through session
        )
        .unwrap();

        let reservation_without = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(10.0),
            &config_no_penalty,
            dec!(0.2),
            1_800_000,
        )
        .unwrap();

        // With penalty, reservation should be lower (more aggressive to sell)
        assert!(reservation_with < reservation_without);
    }

    #[test]
    fn test_terminal_penalty_increases_near_end() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.1), 3_600_000, dec!(0.0001)).unwrap();

        // Early in session
        let reservation_early = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(10.0),
            &config,
            dec!(0.2),
            0, // Start of session
        )
        .unwrap();

        // Late in session
        let reservation_late = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            dec!(10.0),
            &config,
            dec!(0.2),
            3_500_000, // Near end of session
        )
        .unwrap();

        // Near terminal, penalty is higher, so reservation should be lower
        assert!(reservation_late < reservation_early);
    }

    #[test]
    fn test_dynamic_gamma_at_start() {
        let gamma = GLFTStrategy::calculate_dynamic_gamma(
            dec!(0.1),
            3_600_000, // Full time remaining
            3_600_000,
            true,
            dec!(1.0),
        );

        // At start, gamma = base
        assert_eq!(gamma, dec!(0.1));
    }

    #[test]
    fn test_dynamic_gamma_at_end() {
        let gamma = GLFTStrategy::calculate_dynamic_gamma(
            dec!(0.1),
            0, // No time remaining
            3_600_000,
            true,
            dec!(1.0),
        );

        // At end, gamma = base * (1 + alpha) = 0.1 * 2 = 0.2
        assert_eq!(gamma, dec!(0.2));
    }

    #[test]
    fn test_dynamic_gamma_halfway() {
        let gamma = GLFTStrategy::calculate_dynamic_gamma(
            dec!(0.1),
            1_800_000, // Half time remaining
            3_600_000,
            true,
            dec!(1.0),
        );

        // At halfway, gamma = base * (1 + alpha * 0.5) = 0.1 * 1.5 = 0.15
        assert_eq!(gamma, dec!(0.15));
    }

    #[test]
    fn test_dynamic_gamma_disabled() {
        let gamma = GLFTStrategy::calculate_dynamic_gamma(
            dec!(0.1),
            0, // No time remaining
            3_600_000,
            false, // Disabled
            dec!(1.0),
        );

        // When disabled, gamma = base
        assert_eq!(gamma, dec!(0.1));
    }

    #[test]
    fn test_optimal_spread_positive() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let spread = GLFTStrategy::calculate_optimal_spread(&config, dec!(0.2), 0).unwrap();

        assert!(spread > Decimal::ZERO);
    }

    #[test]
    fn test_optimal_spread_min_constraint() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(1.0)).unwrap();

        let spread =
            GLFTStrategy::calculate_optimal_spread(&config, dec!(0.001), 3_599_999).unwrap();

        // Should be at least min_spread
        assert!(spread >= dec!(1.0));
    }

    #[test]
    fn test_optimal_quotes_valid() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let (bid, ask) = GLFTStrategy::calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            &config,
            dec!(0.2),
            0,
        )
        .unwrap();

        assert!(bid < ask);
        assert!(bid > Decimal::ZERO);
    }

    #[test]
    fn test_optimal_quotes_with_inventory() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let (bid_flat, ask_flat) = GLFTStrategy::calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            &config,
            dec!(0.2),
            0,
        )
        .unwrap();

        let (bid_long, ask_long) =
            GLFTStrategy::calculate_optimal_quotes(dec!(100.0), dec!(10.0), &config, dec!(0.2), 0)
                .unwrap();

        // With long inventory, quotes should be lower
        assert!(bid_long < bid_flat);
        assert!(ask_long < ask_flat);
    }

    #[test]
    fn test_compare_with_as() {
        let config = GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.1), 3_600_000, dec!(0.0001))
            .unwrap()
            .with_dynamic_gamma(dec!(1.0));

        let ((glft_bid, glft_ask), (as_bid, as_ask)) =
            GLFTStrategy::compare_with_avellaneda_stoikov(
                dec!(100.0),
                dec!(10.0),
                &config,
                dec!(0.2),
                1_800_000,
            )
            .unwrap();

        // GLFT should be more aggressive (lower quotes with long inventory)
        assert!(glft_bid < as_bid);
        assert!(glft_ask < as_ask);
    }

    #[test]
    fn test_penalty_function_linear() {
        // At start (full time), penalty = 0
        let penalty_start =
            GLFTStrategy::calculate_penalty_function(3_600_000, 3_600_000, PenaltyFunction::Linear);
        assert_eq!(penalty_start, dec!(0.0));

        // At end (no time), penalty = 1
        let penalty_end =
            GLFTStrategy::calculate_penalty_function(0, 3_600_000, PenaltyFunction::Linear);
        assert_eq!(penalty_end, dec!(1.0));

        // Halfway, penalty = 0.5
        let penalty_half =
            GLFTStrategy::calculate_penalty_function(1_800_000, 3_600_000, PenaltyFunction::Linear);
        assert_eq!(penalty_half, dec!(0.5));
    }

    #[test]
    fn test_penalty_function_quadratic() {
        // At start, penalty = 0
        let penalty_start = GLFTStrategy::calculate_penalty_function(
            3_600_000,
            3_600_000,
            PenaltyFunction::Quadratic,
        );
        assert_eq!(penalty_start, dec!(0.0));

        // At end, penalty = 1
        let penalty_end =
            GLFTStrategy::calculate_penalty_function(0, 3_600_000, PenaltyFunction::Quadratic);
        assert_eq!(penalty_end, dec!(1.0));

        // Halfway, penalty = 0.25 (0.5²)
        let penalty_half = GLFTStrategy::calculate_penalty_function(
            1_800_000,
            3_600_000,
            PenaltyFunction::Quadratic,
        );
        assert_eq!(penalty_half, dec!(0.25));
    }

    #[test]
    fn test_invalid_mid_price() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let result = GLFTStrategy::calculate_reservation_price(
            dec!(0.0),
            Decimal::ZERO,
            &config,
            dec!(0.2),
            0,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_volatility() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let result = GLFTStrategy::calculate_reservation_price(
            dec!(100.0),
            Decimal::ZERO,
            &config,
            dec!(0.0),
            0,
        );

        assert!(result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let config =
            GLFTConfig::new(dec!(0.1), dec!(1.5), dec!(0.05), 3_600_000, dec!(0.0001)).unwrap();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: GLFTConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}
