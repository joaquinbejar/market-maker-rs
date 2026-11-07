//! Simulated latency for the backtest engine.
//!
//! Real markets separate the moment a strategy decides from the moment the
//! venue acts on it, and the moment the venue acts from the moment the
//! strategy hears about it. [`SimLatencyConfig`] models four delays:
//!
//! - **observation**: a tick with exchange time `t` reaches the strategy at
//!   `t + observation_ms`;
//! - **submission**: an order decided at `d` becomes active at the venue at
//!   `d + submit_ms`;
//! - **cancel**: a cancel decided at `d` takes effect at `d + cancel_ms`;
//!   fills that happen before that time stand;
//! - **acknowledgement**: a report produced by the venue at `e` is delivered
//!   to the strategy at `e + ack_ms`.
//!
//! Each delay can receive a bounded random jitter drawn from a seeded
//! generator, so runs with the same seed and configuration reproduce the same
//! messages. This module measures nothing; for wall clock measurement see
//! [`crate::execution::LatencyTracker`].
//!
//! # Delays of a whole number of tick intervals
//!
//! Events due at a tick's time run before that tick is matched, and an order
//! is eligible only on ticks after the one it becomes active on. With ticks
//! every `Δ`, a strategy that quotes on every tick, and `submit_ms` and
//! `cancel_ms` both exactly `Δ` with no jitter, an order sent on one tick
//! becomes active on the next, where it cannot match, and the cancel sent by
//! that tick's quote takes effect on the tick after, before matching. No
//! order ever fills. With a delay one millisecond shorter every order fills
//! on the tick it reaches, and an order left resting fills on the tick after
//! it becomes active. The backtest kernel follows the same rules.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::SimLatencyConfig;
//!
//! let latency = SimLatencyConfig::new()
//!     .with_submit_ms(5)
//!     .with_cancel_ms(5)
//!     .with_ack_ms(2)
//!     .with_jitter(3, 42);
//! assert!(latency.validate().is_ok());
//! assert!(!latency.is_zero());
//! ```

use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Upper bound for any configured delay or jitter, in milliseconds (one day).
pub const MAX_SIM_LATENCY_MS: u64 = 86_400_000;

/// Delays applied by the backtest engine, in milliseconds.
///
/// The default is zero for every delay, which keeps the causal zero latency
/// behaviour: an order decided on a tick can only fill on a later tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimLatencyConfig {
    /// Delay between a tick's exchange time and its delivery to the strategy.
    pub observation_ms: u64,
    /// Delay between a quote decision and the orders becoming active.
    pub submit_ms: u64,
    /// Delay between a cancel decision and the cancel taking effect.
    pub cancel_ms: u64,
    /// Delay between a venue event and its report reaching the strategy.
    pub ack_ms: u64,
    /// Maximum extra delay added to each message, drawn uniformly in
    /// `0..=jitter_ms`.
    pub jitter_ms: u64,
    /// Seed of the jitter generator.
    pub seed: u64,
}

impl SimLatencyConfig {
    /// Creates a zero latency configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the observation delay.
    #[must_use]
    pub fn with_observation_ms(mut self, ms: u64) -> Self {
        self.observation_ms = ms;
        self
    }

    /// Sets the submission delay.
    #[must_use]
    pub fn with_submit_ms(mut self, ms: u64) -> Self {
        self.submit_ms = ms;
        self
    }

    /// Sets the cancel delay.
    #[must_use]
    pub fn with_cancel_ms(mut self, ms: u64) -> Self {
        self.cancel_ms = ms;
        self
    }

    /// Sets the acknowledgement delay.
    #[must_use]
    pub fn with_ack_ms(mut self, ms: u64) -> Self {
        self.ack_ms = ms;
        self
    }

    /// Sets the maximum jitter and its seed.
    #[must_use]
    pub fn with_jitter(mut self, jitter_ms: u64, seed: u64) -> Self {
        self.jitter_ms = jitter_ms;
        self.seed = seed;
        self
    }

    /// Returns true if every delay and the jitter are zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.observation_ms == 0
            && self.submit_ms == 0
            && self.cancel_ms == 0
            && self.ack_ms == 0
            && self.jitter_ms == 0
    }

    /// Checks that every delay is within [`MAX_SIM_LATENCY_MS`].
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a delay is too large,
    /// which also rules out timestamp overflow.
    pub fn validate(&self) -> MMResult<()> {
        let delays = [
            ("observation_ms", self.observation_ms),
            ("submit_ms", self.submit_ms),
            ("cancel_ms", self.cancel_ms),
            ("ack_ms", self.ack_ms),
            ("jitter_ms", self.jitter_ms),
        ];
        for (name, value) in delays {
            if value > MAX_SIM_LATENCY_MS {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be at most {MAX_SIM_LATENCY_MS} ms, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// One xorshift64* step: returns `(next_state, output)`.
///
/// Shared by [`JitterRng`] and the financial kernel's generator so both
/// produce bit-identical streams.
#[must_use]
pub(crate) const fn xorshift_step(state: u64) -> (u64, u64) {
    let mut x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    (x, x.wrapping_mul(0x2545_F491_4F6C_DD1D))
}

/// Deterministic jitter source (xorshift64*), seeded per run.
#[derive(Debug, Clone)]
pub(crate) struct JitterRng {
    seed: u64,
    state: u64,
}

impl JitterRng {
    pub(crate) fn new(seed: u64) -> Self {
        let state = Self::nonzero(seed);
        Self { seed, state }
    }

    fn nonzero(seed: u64) -> u64 {
        if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        }
    }

    pub(crate) fn reset(&mut self) {
        self.state = Self::nonzero(self.seed);
    }

    /// Returns a value in `0..=max`.
    pub(crate) fn draw(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let (state, value) = xorshift_step(self.state);
        self.state = state;
        // `max + 1` overflows only for `u64::MAX`, where every value is in range.
        match max.checked_add(1) {
            // `bound >= 1`, so the remainder cannot fail.
            #[allow(clippy::arithmetic_side_effects)]
            Some(bound) => value % bound,
            None => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitter_rng_draw_max_bound_does_not_overflow() {
        let mut rng = JitterRng::new(7);
        for _ in 0..16 {
            let _ = rng.draw(u64::MAX);
        }
        assert!(rng.draw(3) <= 3);
    }

    #[test]
    fn test_default_is_zero() {
        let config = SimLatencyConfig::default();
        assert!(config.is_zero());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_sets_fields() {
        let config = SimLatencyConfig::new()
            .with_observation_ms(1)
            .with_submit_ms(2)
            .with_cancel_ms(3)
            .with_ack_ms(4)
            .with_jitter(5, 6);
        assert_eq!(config.observation_ms, 1);
        assert_eq!(config.submit_ms, 2);
        assert_eq!(config.cancel_ms, 3);
        assert_eq!(config.ack_ms, 4);
        assert_eq!(config.jitter_ms, 5);
        assert_eq!(config.seed, 6);
        assert!(!config.is_zero());
    }

    #[test]
    fn test_validate_rejects_huge_delays() {
        let config = SimLatencyConfig::new().with_ack_ms(MAX_SIM_LATENCY_MS + 1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_jitter_is_bounded_and_reproducible() {
        let mut a = JitterRng::new(7);
        let first: Vec<u64> = (0..100).map(|_| a.draw(3)).collect();
        assert!(first.iter().all(|v| *v <= 3));
        assert!(first.iter().any(|v| *v > 0));
        a.reset();
        let second: Vec<u64> = (0..100).map(|_| a.draw(3)).collect();
        assert_eq!(first, second);
        let mut zero = JitterRng::new(0);
        assert_eq!(zero.draw(0), 0);
        assert!(zero.draw(10) <= 10);
    }
}
