//! Tagged, versioned random stream of the kernel.

use crate::backtest::latency::xorshift_step;

/// Algorithm tag of [`KernelRng`] version 1: xorshift64* with zero seed
/// substitution, bit-identical to the engine's jitter generator.
pub const RNG_ALGO_XORSHIFT64STAR: u8 = 1;

/// Seed used in place of zero, shared with the engine's jitter generator.
const ZERO_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic random stream held inside the kernel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelRng {
    pub(super) algo: u8,
    pub(super) seed: u64,
    pub(super) state: u64,
}

impl KernelRng {
    /// A stream seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { ZERO_SEED } else { seed };
        Self {
            algo: RNG_ALGO_XORSHIFT64STAR,
            seed,
            state,
        }
    }

    /// Algorithm tag.
    #[must_use]
    pub fn algo(&self) -> u8 {
        self.algo
    }

    /// Seed the stream started from.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Current generator state.
    #[must_use]
    pub fn state(&self) -> u64 {
        self.state
    }

    /// A value in `0..=max`; `max == 0` returns 0 without advancing.
    pub fn draw(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let (state, value) = xorshift_step(self.state);
        self.state = state;
        match max.checked_add(1) {
            Some(bound) => value.checked_rem(bound).unwrap_or(value),
            None => value,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::backtest::latency::JitterRng;

    #[test]
    fn test_kernel_rng_draw_matches_jitter_rng_bit_for_bit() {
        for seed in [0u64, 1, 7, 42, u64::MAX] {
            let mut jitter = JitterRng::new(seed);
            let mut kernel = KernelRng::new(seed);
            for i in 0..2_000u64 {
                let max = [0, 1, 3, 17, 1_000, u64::MAX][usize::try_from(i % 6).unwrap()];
                assert_eq!(kernel.draw(max), jitter.draw(max));
            }
        }
    }
}
