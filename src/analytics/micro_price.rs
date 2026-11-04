//! Stoikov's micro-price: a fair value from imbalance and spread transitions.
//!
//! Follows S. Stoikov, *The Micro-Price: A High Frequency Estimator of Future
//! Prices*, manuscript dated 18 April 2018, section 3 and the estimation
//! procedure of section 4. It is not the weighted mid. The author's code is not
//! used.
//!
//! With the state `x = (imbalance bucket, spread in ticks)`:
//!
//! ```text
//! R·K (x)  = E[M(t+1) - M(t) | x]                    expected mid move
//! Q (x, y) = P(mid unchanged and next state y | x)
//! T (x, y) = P(mid changed and next state y | x)
//! G¹ = (1 - Q)⁻¹ R K
//! B  = (1 - Q)⁻¹ T
//! G* = G¹ + Σ_{i≥1} Bⁱ G¹,    P_micro = M + G*
//! ```
//!
//! # Estimation
//!
//! Each observed pair of quotes also counts as its mirror image, with buckets
//! `i` and `n + 1 - i` swapped and the mid move negated, so `B* G¹ = 0` as in
//! the paper's theorem 3.1. The paper's mirror `I -> 1 - I` only maps buckets
//! onto buckets if the discretization is itself symmetric, and its rule
//! `(j - 1) / n < I <= j / n` is not, which biases the adjustments in one
//! direction. So here `n` is odd, the middle bucket holds `I = 1/2`, a book
//! with `I < 1/2` follows the paper's rule (`I = 0` is in bucket 1, which the
//! paper leaves out), and a book with `I > 1/2` takes the mirror of its swapped
//! book. The rule compares integer multiples of sizes, without rounding a
//! division, and `G*` is antisymmetric exactly.
//!
//! The paper's moves are four values of up to one tick. Here any mid move on
//! the half tick grid is accepted and none is filtered, so one bad print does
//! move the fit, in proportion to its size: 1000 ticks in 100,000 quotes can
//! halve the adjustments and more can flip their sign. The record reports
//! `largest_mid_move_ticks` so that this can be seen; cleaning the data is the
//! caller's.
//!
//! # Fallbacks
//!
//! The manuscript defines no treatment of rare states, so these are decisions
//! of this crate. The fit never fails silently into a number: each state has a
//! [`StateStatus`], and a state that is not [`StateStatus::Calibrated`] gives
//! the mid, with the reason in the signal.
//!
//! - A state below `min_state_observations`, or one from which the data never
//!   leads to a mid move, keeps adjustment 0 and absorbs the chains that reach it.
//!   A state that reaches an observed move is calibrated even if some of its
//!   paths end in such a state, which then pulls its adjustment toward 0.
//! - The paper warns the limit may lie outside the quotes. A state where it
//!   does is [`StateStatus::OutsideQuotes`].
//! - That switch makes the signal non-monotonic in the imbalance: a strong
//!   imbalance can give the mid while a weaker one gives a large adjustment.
//! - A pair with a spread above `spread_states` or further apart than
//!   `max_quote_gap_ms` is not counted, and a spread above `spread_states` has
//!   no adjustment. Leaving pairs out cuts the chains and changes the
//!   adjustments, by an amount that grows with the share left out and that can
//!   be large. The record counts each reason, and
//!   [`MicroPriceModel::calibrated_share`] gives the coverage of the fit.
//! - [`StateEstimate::fallback_within_moves`] reports how much a state's
//!   adjustment depends on absorption by rare or no move states.
//!
//! # Invalid quotes
//!
//! [`calibrate_micro_price`] and [`compare_estimators`] leave out a crossed,
//! empty or off grid quote and count it. The quotes on either side of it form
//! a pair if they pass the gap rule. [`MicroPriceCalibrator::observe`] and
//! [`MicroPriceModel::signal`] refuse such a quote. A quote whose values overflow
//! the arithmetic, such as a price too large in ticks for `i64`, is a typed
//! [`MMError::NumericalError`] that ends the whole call in all four.
//!
//! # Causality
//!
//! [`calibrate_micro_price`] reads only the quotes inside its window. The
//! signal is a function of one quote and a fitted model. This module does not
//! feed quoting: any such use is separate from [`compare_estimators`].
//!
//! # Example
//!
//! ```
//! use market_maker_rs::analytics::micro_price::{MicroPriceConfig, TopOfBook};
//! use market_maker_rs::dec;
//!
//! let config = MicroPriceConfig::new(dec!(0.01), 5, 2);
//! assert!(config.validate().is_ok());
//! let quote = TopOfBook {
//!     timestamp_ms: 1,
//!     bid: dec!(10.00),
//!     ask: dec!(10.01),
//!     bid_size: dec!(9),
//!     ask_size: dec!(1),
//! };
//! assert_eq!(quote.weighted_mid().unwrap(), dec!(10.009));
//! ```

use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

use crate::Decimal;
use crate::strategy::walk_forward::{FNV_OFFSET, TimeWindow, fnv};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Most states `imbalance_buckets * spread_states` accepted.
const MAX_STATES: usize = 256;
/// Mid moves over which [`StateEstimate::fallback_within_moves`] is measured.
const FALLBACK_HORIZON_MOVES: usize = 10;
/// Most terms of the series for `G*`.
const MAX_SERIES_TERMS: usize = 10_000;
/// Size of a series term, in half ticks, below which the series has converged.
const SERIES_TOLERANCE: f64 = 1e-12;
/// Pivot under which `1 - Q` counts as singular.
const PIVOT_TOLERANCE: f64 = 1e-12;
/// Most horizons accepted by one comparison.
const MAX_HORIZONS: usize = 64;

/// Best bid and ask with their sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TopOfBook {
    /// Quote time in milliseconds.
    pub timestamp_ms: u64,
    /// Best bid. Positive.
    pub bid: Decimal,
    /// Best ask. Above the bid.
    pub ask: Decimal,
    /// Size at the best bid. Not negative.
    pub bid_size: Decimal,
    /// Size at the best ask. Not negative; both sizes are not zero together.
    pub ask_size: Decimal,
}

impl TopOfBook {
    fn validate(&self) -> MMResult<()> {
        if self.bid <= Decimal::ZERO
            || self.ask <= self.bid
            || self.bid_size < Decimal::ZERO
            || self.ask_size < Decimal::ZERO
            || (self.bid_size.is_zero() && self.ask_size.is_zero())
        {
            return Err(MMError::InvalidMarketState(format!(
                "invalid quote at {}: {} x {} / {} x {}",
                self.timestamp_ms, self.bid_size, self.bid, self.ask, self.ask_size
            )));
        }
        Ok(())
    }

    /// Returns `(bid + ask) / 2`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on overflow.
    pub fn mid(&self) -> MMResult<Decimal> {
        self.bid
            .checked_add(self.ask)
            .and_then(|sum| sum.checked_div(Decimal::TWO))
            .or_overflow("mid price")
    }

    /// Returns the imbalance `bid size / (bid size + ask size)`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for an invalid quote and
    /// [`MMError::NumericalError`] on overflow.
    pub fn imbalance(&self) -> MMResult<Decimal> {
        self.validate()?;
        self.bid_size
            .checked_add(self.ask_size)
            .and_then(|total| self.bid_size.checked_div(total))
            .or_overflow("imbalance")
    }

    /// Returns the weighted mid `I · ask + (1 - I) · bid`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for an invalid quote and
    /// [`MMError::NumericalError`] on overflow.
    pub fn weighted_mid(&self) -> MMResult<Decimal> {
        let imbalance = self.imbalance()?;
        let on_ask = imbalance.checked_mul(self.ask);
        let on_bid = Decimal::ONE
            .checked_sub(imbalance)
            .and_then(|rest| rest.checked_mul(self.bid));
        on_ask
            .zip(on_bid)
            .and_then(|(a, b)| a.checked_add(b))
            .or_overflow("weighted mid")
    }
}

/// Discretization and data rules of the estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "RawConfig"))]
#[non_exhaustive]
pub struct MicroPriceConfig {
    /// Price grid. Positive; quotes off this grid are refused.
    pub tick_size: Decimal,
    /// Imbalance buckets `n`. Odd and at least 3.
    pub imbalance_buckets: u32,
    /// Spreads `m`, of 1 to `m` ticks. Positive.
    pub spread_states: u32,
    /// Fewest pairs, mirrors included, for a state to be calibrated. Positive.
    pub min_state_observations: u64,
    /// Longest time between two quotes of a pair, in milliseconds. Positive.
    pub max_quote_gap_ms: u64,
}

impl MicroPriceConfig {
    /// Creates a configuration with a minimum of 100 observations per state and
    /// a longest gap of one minute.
    #[must_use]
    pub fn new(tick_size: Decimal, imbalance_buckets: u32, spread_states: u32) -> Self {
        Self {
            tick_size,
            imbalance_buckets,
            spread_states,
            min_state_observations: 100,
            max_quote_gap_ms: 60_000,
        }
    }

    /// Sets the fewest observations for a state to be calibrated.
    #[must_use]
    pub fn with_min_state_observations(mut self, observations: u64) -> Self {
        self.min_state_observations = observations;
        self
    }

    /// Sets the longest time between two quotes of a pair.
    #[must_use]
    pub fn with_max_quote_gap_ms(mut self, gap_ms: u64) -> Self {
        self.max_quote_gap_ms = gap_ms;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a value out of its range
    /// an even number of buckets or more than 256 states.
    pub fn validate(&self) -> MMResult<()> {
        let states =
            u64::from(self.imbalance_buckets).saturating_mul(u64::from(self.spread_states));
        if self.tick_size <= Decimal::ZERO
            || self.imbalance_buckets < 3
            || self.imbalance_buckets.is_multiple_of(2)
            || self.spread_states == 0
            || self.min_state_observations == 0
            || self.max_quote_gap_ms == 0
            || states > MAX_STATES as u64
        {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid micro-price configuration: {self:?}"
            )));
        }
        Ok(())
    }

    fn state_count(&self) -> MMResult<usize> {
        self.validate()?;
        let buckets = usize::try_from(self.imbalance_buckets).ok();
        let spreads = usize::try_from(self.spread_states).ok();
        buckets
            .zip(spreads)
            .and_then(|(b, s)| b.checked_mul(s))
            .or_overflow("state count")
    }

    /// `price / tick` as an integer, or an error for a price off the grid.
    fn in_ticks(&self, price: Decimal, what: &str) -> MMResult<i64> {
        let ticks = price.checked_div(self.tick_size).or_overflow(what)?;
        if !ticks.fract().is_zero() {
            return Err(MMError::InvalidMarketState(format!(
                "{what} {price} is off the tick {}",
                self.tick_size
            )));
        }
        ticks.to_i64().or_overflow(what)
    }

    /// The bucket from 0 of a book with `lighter <= heavier` on the side `I`
    /// measures: the smallest `j` with `n · lighter <= j · total`.
    fn lower_bucket(&self, lighter: Decimal, heavier: Decimal) -> MMResult<u32> {
        let total = lighter.checked_add(heavier).or_overflow("total size")?;
        let scaled = lighter
            .checked_mul(Decimal::from(self.imbalance_buckets))
            .or_overflow("imbalance bucket")?;
        for j in 1..self.imbalance_buckets {
            let bound = total
                .checked_mul(Decimal::from(j))
                .or_overflow("imbalance bucket")?;
            if scaled <= bound {
                return j.checked_sub(1).or_overflow("imbalance bucket");
            }
        }
        self.imbalance_buckets
            .checked_sub(1)
            .or_overflow("imbalance bucket")
    }

    /// Returns the state index, if the spread has one.
    fn state_of(&self, quote: &TopOfBook) -> MMResult<Option<usize>> {
        quote.validate()?;
        let last = self
            .imbalance_buckets
            .checked_sub(1)
            .or_overflow("imbalance bucket")?;
        let bucket = if quote.bid_size <= quote.ask_size {
            self.lower_bucket(quote.bid_size, quote.ask_size)?
        } else {
            last.checked_sub(self.lower_bucket(quote.ask_size, quote.bid_size)?)
                .or_overflow("imbalance bucket")?
        };
        let spread = self
            .in_ticks(quote.ask, "ask")?
            .checked_sub(self.in_ticks(quote.bid, "bid")?)
            .or_overflow("spread")?;
        if spread > i64::from(self.spread_states) {
            return Ok(None);
        }
        let spread_index = u32::try_from(spread)
            .ok()
            .and_then(|s| s.checked_sub(1))
            .or_overflow("spread")?;
        let state = spread_index
            .checked_mul(self.imbalance_buckets)
            .and_then(|v| v.checked_add(bucket))
            .and_then(|v| usize::try_from(v).ok())
            .or_overflow("state index")?;
        Ok(Some(state))
    }

    /// Checks everything the estimator computes from a quote.
    fn usable_state(&self, quote: &TopOfBook) -> MMResult<Option<usize>> {
        let state = self.state_of(quote)?;
        quote.mid()?;
        quote.weighted_mid()?;
        Ok(state)
    }

    /// The state with the imbalance bucket mirrored.
    fn mirror(&self, state: usize) -> MMResult<usize> {
        let n = usize::try_from(self.imbalance_buckets)
            .ok()
            .filter(|n| *n > 0)
            .or_overflow("imbalance buckets")?;
        let bucket = state.checked_rem(n).or_overflow("state bucket")?;
        let spread_part = state.checked_sub(bucket).or_overflow("state spread")?;
        n.checked_sub(1)
            .and_then(|last| last.checked_sub(bucket))
            .and_then(|mirrored| spread_part.checked_add(mirrored))
            .or_overflow("mirrored state")
    }
}

/// Counts of the transitions seen so far: the estimator's restorable state.
///
/// Cloning or serializing it between two quotes and continuing from the copy
/// gives the same fit as an uninterrupted run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "RawCalibrator"))]
pub struct MicroPriceCalibrator {
    config: MicroPriceConfig,
    states: usize,
    /// Pairs from `x` to `y` with the mid unchanged, `states²` entries.
    stay: Vec<u64>,
    /// Pairs from `x` to `y` with the mid changed, `states²` entries.
    moved: Vec<u64>,
    /// Sum of the mid moves out of `x`, in half ticks.
    move_half_ticks: Vec<i64>,
    previous: Option<TopOfBook>,
    pairs: u64,
    skipped_spread_pairs: u64,
    skipped_gap_pairs: u64,
    largest_move_half_ticks: u64,
    hash: u128,
}

impl MicroPriceCalibrator {
    /// Creates an empty calibrator.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for an invalid configuration.
    pub fn new(config: MicroPriceConfig) -> MMResult<Self> {
        let states = config.state_count()?;
        let cells = states.checked_mul(states).or_overflow("transition cells")?;
        Ok(Self {
            config,
            states,
            stay: vec![0; cells],
            moved: vec![0; cells],
            move_half_ticks: vec![0; states],
            previous: None,
            pairs: 0,
            skipped_spread_pairs: 0,
            skipped_gap_pairs: 0,
            largest_move_half_ticks: 0,
            hash: FNV_OFFSET,
        })
    }

    /// Adds the next quote.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for an invalid or off grid
    /// quote, or a timestamp that goes back, leaving the calibrator unchanged;
    /// and [`MMError::NumericalError`] on overflow.
    pub fn observe(&mut self, quote: &TopOfBook) -> MMResult<()> {
        let to = self.config.usable_state(quote)?;
        let next_half_ticks = self.half_ticks(quote)?;
        if let Some(previous) = self.previous {
            if quote.timestamp_ms < previous.timestamp_ms {
                return Err(MMError::InvalidMarketState(format!(
                    "quote at {} follows one at {}",
                    quote.timestamp_ms, previous.timestamp_ms
                )));
            }
            let from = self.config.state_of(&previous)?;
            let gap = quote.timestamp_ms.saturating_sub(previous.timestamp_ms);
            let change = next_half_ticks
                .checked_sub(self.half_ticks(&previous)?)
                .or_overflow("mid move")?;
            let transition = from.zip(to);
            if gap > self.config.max_quote_gap_ms {
                self.skipped_gap_pairs =
                    self.skipped_gap_pairs.checked_add(1).or_overflow("pairs")?;
            } else if let Some((from, to)) = transition {
                self.largest_move_half_ticks =
                    self.largest_move_half_ticks.max(change.unsigned_abs());
                self.count(from, to, change)?;
                let negated = change.checked_neg().or_overflow("mid move")?;
                self.count(self.config.mirror(from)?, self.config.mirror(to)?, negated)?;
                self.pairs = self.pairs.checked_add(1).or_overflow("pairs")?;
            } else {
                self.skipped_spread_pairs = self
                    .skipped_spread_pairs
                    .checked_add(1)
                    .or_overflow("pairs")?;
            }
        }
        self.previous = Some(*quote);
        self.hash = hash_quote(self.hash, quote);
        Ok(())
    }

    /// Fits the model on the counts so far.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on arithmetic failure. A lack of
    /// data is not an error: see [`MicroPriceFit::Insufficient`].
    pub fn fit(&self) -> MMResult<MicroPriceFit> {
        fit_counts(self)
    }

    /// Returns the pairs counted, mirrors not included.
    #[must_use]
    pub fn pairs(&self) -> u64 {
        self.pairs
    }

    /// Returns the pairs left out for a spread out of range.
    #[must_use]
    pub fn skipped_spread_pairs(&self) -> u64 {
        self.skipped_spread_pairs
    }

    /// Returns the pairs left out for a gap above `max_quote_gap_ms`.
    #[must_use]
    pub fn skipped_gap_pairs(&self) -> u64 {
        self.skipped_gap_pairs
    }

    /// Returns the largest absolute mid move counted, in ticks.
    #[must_use]
    pub fn largest_mid_move_ticks(&self) -> Decimal {
        Decimal::from(self.largest_move_half_ticks)
            .checked_div(Decimal::TWO)
            .unwrap_or(Decimal::MAX)
    }

    /// Returns the FNV-1a 128 of the quotes observed.
    #[must_use]
    pub fn dataset_hash(&self) -> u128 {
        self.hash
    }

    fn half_ticks(&self, quote: &TopOfBook) -> MMResult<i64> {
        self.config
            .in_ticks(quote.bid, "bid")?
            .checked_add(self.config.in_ticks(quote.ask, "ask")?)
            .or_overflow("mid in half ticks")
    }

    fn count(&mut self, from: usize, to: usize, change: i64) -> MMResult<()> {
        let cell = from
            .checked_mul(self.states)
            .and_then(|v| v.checked_add(to))
            .or_overflow("transition cell")?;
        let table = if change == 0 {
            &mut self.stay
        } else {
            &mut self.moved
        };
        let entry = table.get_mut(cell).or_overflow("transition cell")?;
        *entry = entry.checked_add(1).or_overflow("transition count")?;
        let total = self
            .move_half_ticks
            .get_mut(from)
            .or_overflow("transition state")?;
        *total = total.checked_add(change).or_overflow("mid moves")?;
        Ok(())
    }
}

/// Whether a state's adjustment is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum StateStatus {
    /// The adjustment is estimated and used.
    Calibrated,
    /// Too few observations; the signal is the mid.
    Rare,
    /// The data never leads from this state to a mid move; the signal is the mid.
    NoPriceMove,
    /// The limit lies outside the quotes; the signal is the mid.
    OutsideQuotes,
}

/// A state of the table by spread and imbalance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct StateEstimate {
    /// Imbalance bucket, from 1.
    pub imbalance_bucket: u32,
    /// Spread in ticks, from 1.
    pub spread_ticks: u32,
    /// Pairs out of this state, mirrors included.
    pub observations: u64,
    /// `G¹`, in price units.
    pub first_adjustment: Decimal,
    /// `G*`, in price units. Zero for [`StateStatus::Rare`] and
    /// [`StateStatus::NoPriceMove`]; for [`StateStatus::OutsideQuotes`], the
    /// value that was refused.
    pub adjustment: Decimal,
    /// Probability that the chain from this state is absorbed by a
    /// [`StateStatus::Rare`] or [`StateStatus::NoPriceMove`] state within 10 mid
    /// moves. Those absorb with adjustment 0, so the higher this is, the more
    /// the adjustment is shrunk by them and not by evidence. It is 1 on those
    /// states. [`StateStatus::OutsideQuotes`] is decided after the fit and does
    /// not absorb in the matrices, so it does not count here, even though its
    /// signal is the mid too.
    pub fallback_within_moves: Decimal,
    /// Whether the signal uses the adjustment.
    pub status: StateStatus,
}

/// Fitted adjustments.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "RawModel"))]
pub struct MicroPriceModel {
    config: MicroPriceConfig,
    states: Vec<StateEstimate>,
}

/// Why a signal is what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum SignalSource {
    /// Mid plus the calibrated adjustment.
    MicroPrice,
    /// The mid, for this state's status.
    MidFallback(StateStatus),
    /// The mid, for a spread above `spread_states`.
    SpreadOutOfRange,
}

/// A fair value at the time of a quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct MicroPriceSignal {
    /// Time of the quote it was computed from.
    pub timestamp_ms: u64,
    /// The fair value; within the quote's bid and ask.
    pub price: Decimal,
    /// The quote's mid.
    pub mid: Decimal,
    /// `price - mid`.
    pub adjustment: Decimal,
    /// Whether `price` is the micro-price or a fallback.
    pub source: SignalSource,
}

impl MicroPriceModel {
    /// Returns the table by spread and imbalance.
    #[must_use]
    pub fn states(&self) -> &[StateEstimate] {
        &self.states
    }

    /// Returns the share of the training observations, mirrors included, that
    /// are in [`StateStatus::Calibrated`] states.
    #[must_use]
    pub fn calibrated_share(&self) -> Decimal {
        let total = self
            .states
            .iter()
            .fold(0u64, |sum, s| sum.saturating_add(s.observations));
        let calibrated = self
            .states
            .iter()
            .filter(|s| s.status == StateStatus::Calibrated)
            .fold(0u64, |sum, s| sum.saturating_add(s.observations));
        Decimal::from(calibrated)
            .checked_div(Decimal::from(total))
            .unwrap_or(Decimal::ZERO)
    }

    /// Computes the signal of a quote.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for an invalid or off grid
    /// quote and [`MMError::NumericalError`] on overflow.
    pub fn signal(&self, quote: &TopOfBook) -> MMResult<MicroPriceSignal> {
        let mid = quote.mid()?;
        let state = self.config.usable_state(quote)?;
        let (adjustment, source) = match state.and_then(|s| self.states.get(s)) {
            None => (Decimal::ZERO, SignalSource::SpreadOutOfRange),
            Some(estimate) => match estimate.status {
                StateStatus::Calibrated => (estimate.adjustment, SignalSource::MicroPrice),
                status @ (StateStatus::Rare
                | StateStatus::NoPriceMove
                | StateStatus::OutsideQuotes) => (Decimal::ZERO, SignalSource::MidFallback(status)),
            },
        };
        let price = mid.checked_add(adjustment).or_overflow("micro-price")?;
        Ok(MicroPriceSignal {
            timestamp_ms: quote.timestamp_ms,
            price,
            mid,
            adjustment,
            source,
        })
    }
}

/// Why no model could be fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum MicroPriceFailure {
    /// No state reached the minimum of observations with a mid move in reach.
    NoCalibratedState,
    /// `1 - Q` could not be inverted numerically.
    SingularTransitions,
    /// The series for `G*` did not converge.
    NotConverged,
    /// An adjustment is not representable as a `Decimal`.
    Unrepresentable,
}

/// Result of a fit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum MicroPriceFit {
    /// A model with at least one calibrated state.
    Calibrated(MicroPriceModel),
    /// No model.
    Insufficient(MicroPriceFailure),
}

/// A fit together with what it was estimated from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct MicroPriceRecord {
    /// Training interval.
    pub training: TimeWindow,
    /// Caller's name for the dataset.
    pub dataset_id: String,
    /// FNV-1a 128 of the quotes inside the training interval.
    pub dataset_hash: u128,
    /// Pairs counted, mirrors not included.
    pub pairs: u64,
    /// Pairs left out for a spread out of range.
    pub skipped_spread_pairs: u64,
    /// Pairs left out for a gap above `max_quote_gap_ms`.
    pub skipped_gap_pairs: u64,
    /// Largest absolute mid move counted, in ticks. A large value next to the
    /// tick grid points to a bad print: see the module docs.
    pub largest_mid_move_ticks: Decimal,
    /// Quotes left out as invalid or off the grid. They are not in the hash.
    pub invalid_quotes: u64,
    /// The model, or why there is none.
    pub fit: MicroPriceFit,
}

impl MicroPriceRecord {
    /// Returns the model, if there is one.
    #[must_use]
    pub fn model(&self) -> Option<&MicroPriceModel> {
        match &self.fit {
            MicroPriceFit::Calibrated(model) => Some(model),
            MicroPriceFit::Insufficient(_) => None,
        }
    }
}

/// Fits on the quotes inside `window` and nothing else.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid configuration,
/// [`MMError::InvalidMarketState`] for out of order quotes, and [`MMError::NumericalError`] on arithmetic failure.
pub fn calibrate_micro_price(
    quotes: &[TopOfBook],
    window: &TimeWindow,
    config: &MicroPriceConfig,
    dataset_id: &str,
) -> MMResult<MicroPriceRecord> {
    let inside = quotes_in(quotes, window)?;
    let mut calibrator = MicroPriceCalibrator::new(*config)?;
    let mut invalid_quotes = 0u64;
    for quote in inside {
        match calibrator.observe(quote) {
            Ok(()) => {}
            Err(MMError::InvalidMarketState(_)) => {
                invalid_quotes = invalid_quotes.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(MicroPriceRecord {
        training: *window,
        dataset_id: dataset_id.to_string(),
        dataset_hash: calibrator.dataset_hash(),
        pairs: calibrator.pairs(),
        skipped_spread_pairs: calibrator.skipped_spread_pairs(),
        skipped_gap_pairs: calibrator.skipped_gap_pairs(),
        largest_mid_move_ticks: calibrator.largest_mid_move_ticks(),
        invalid_quotes,
        fit: calibrator.fit()?,
    })
}

/// Out-of-sample errors of one estimator at one horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct EstimatorError {
    /// Root mean square of `future mid - estimator`.
    pub rmse: Decimal,
    /// Mean absolute `future mid - estimator`.
    pub mean_absolute: Decimal,
    /// Mean of `side · (future mid - estimator)`, where `side` is +1 when the
    /// imbalance is above 1/2, -1 below, and the sample is left out at exactly
    /// 1/2: the markout of trading at the estimator in the imbalance's
    /// direction. Positive means the move was underestimated, negative
    /// overestimated. `None` when every sample is at 1/2.
    pub directional_markout: Option<Decimal>,
}

/// The three estimators on the same samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct EstimatorSet {
    /// The mid.
    pub mid: EstimatorError,
    /// The weighted mid.
    pub weighted_mid: EstimatorError,
    /// The signal, fallbacks included.
    pub micro_price: EstimatorError,
}

/// Errors in one cell of the table by spread and imbalance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct StateComparison {
    /// Imbalance bucket, from 1.
    pub imbalance_bucket: u32,
    /// Spread in ticks, from 1.
    pub spread_ticks: u32,
    /// Samples in this state.
    pub samples: u64,
    /// Errors of the estimators.
    pub errors: EstimatorSet,
}

/// The comparison at one horizon.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct HorizonComparison {
    /// Horizon in milliseconds, as given by the caller.
    pub horizon_ms: u64,
    /// Samples compared.
    pub samples: u64,
    /// Of those, samples where the signal was a fallback.
    pub fallback_samples: u64,
    /// Quotes left out as invalid or off the grid.
    pub invalid_quotes: u64,
    /// Samples left out: horizon past the window, or the quote at the horizon
    /// staler than `max_quote_gap_ms`.
    pub unscored_samples: u64,
    /// Errors over all samples; `None` without samples.
    pub overall: Option<EstimatorSet>,
    /// Errors of the calibrated or fallback states with samples. Quotes with
    /// a spread out of range count only in `overall`.
    pub by_state: Vec<StateComparison>,
}

/// Compares the estimators of `record` on `window` at every horizon of
/// `horizons_ms`.
///
/// All horizons are reported, in the order given, and none is selected. The
/// mid at a horizon is that of the last valid quote at or before it, inside
/// the window. Taking the record lets this refuse a window that is not after
/// the training interval.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a record without a model, a
/// window that starts before the training interval ends, no horizons, more
/// than 64, or a zero horizon; [`MMError::InvalidMarketState`] for out of
/// order quotes; and [`MMError::NumericalError`] on arithmetic failure.
pub fn compare_estimators(
    record: &MicroPriceRecord,
    quotes: &[TopOfBook],
    window: &TimeWindow,
    horizons_ms: &[u64],
) -> MMResult<Vec<HorizonComparison>> {
    let Some(model) = record.model() else {
        return Err(MMError::InvalidConfiguration(format!(
            "no model to compare: {:?}",
            record.fit
        )));
    };
    if window.start_ms < record.training.end_ms {
        return Err(MMError::InvalidConfiguration(format!(
            "test window {window:?} is not after training {:?}",
            record.training
        )));
    }
    if horizons_ms.is_empty() || horizons_ms.len() > MAX_HORIZONS || horizons_ms.contains(&0) {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid horizons: {horizons_ms:?}"
        )));
    }
    let inside = quotes_in(quotes, window)?;
    let mut valid = Vec::with_capacity(inside.len());
    let mut invalid_quotes = 0u64;
    for quote in inside {
        match model.config.usable_state(quote) {
            Ok(_) => valid.push(*quote),
            Err(MMError::InvalidMarketState(_)) => {
                invalid_quotes = invalid_quotes.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
    horizons_ms
        .iter()
        .map(|horizon| compare_at(model, &valid, window, *horizon, invalid_quotes))
        .collect()
}

/// Running sums of one estimator's errors.
#[derive(Clone, Copy, Default)]
struct ErrorSums {
    squared: f64,
    absolute: f64,
    directional: f64,
    directional_samples: u64,
}

impl ErrorSums {
    /// Adds `future - estimate`, subtracted in `Decimal` so that a high price
    /// level does not round the error away in `f64`.
    fn add(&mut self, future: Decimal, estimate: Decimal, side: f64) -> MMResult<()> {
        let error = to_f64(
            future
                .checked_sub(estimate)
                .or_overflow("estimator error")?,
        )?;
        self.squared += error * error;
        self.absolute += error.abs();
        if side != 0.0 {
            self.directional += side * error;
            self.directional_samples = self.directional_samples.saturating_add(1);
        }
        Ok(())
    }

    fn finish(&self, samples: u64) -> MMResult<EstimatorError> {
        let count = to_f64(Decimal::from(samples))?;
        let directional_markout = if self.directional_samples == 0 {
            None
        } else {
            Some(from_f64(
                self.directional / to_f64(Decimal::from(self.directional_samples))?,
            )?)
        };
        Ok(EstimatorError {
            rmse: from_f64((self.squared / count).sqrt())?,
            mean_absolute: from_f64(self.absolute / count)?,
            directional_markout,
        })
    }
}

/// Running sums of the three estimators.
#[derive(Clone, Copy, Default)]
struct SetSums {
    samples: u64,
    mid: ErrorSums,
    weighted: ErrorSums,
    micro: ErrorSums,
}

impl SetSums {
    fn add(
        &mut self,
        future: Decimal,
        mid: Decimal,
        weighted: Decimal,
        micro: Decimal,
        side: f64,
    ) -> MMResult<()> {
        self.samples = self.samples.saturating_add(1);
        self.mid.add(future, mid, side)?;
        self.weighted.add(future, weighted, side)?;
        self.micro.add(future, micro, side)
    }

    fn finish(&self) -> MMResult<Option<EstimatorSet>> {
        if self.samples == 0 {
            return Ok(None);
        }
        Ok(Some(EstimatorSet {
            mid: self.mid.finish(self.samples)?,
            weighted_mid: self.weighted.finish(self.samples)?,
            micro_price: self.micro.finish(self.samples)?,
        }))
    }
}

fn compare_at(
    model: &MicroPriceModel,
    quotes: &[TopOfBook],
    window: &TimeWindow,
    horizon_ms: u64,
    invalid_quotes: u64,
) -> MMResult<HorizonComparison> {
    let mut overall = SetSums::default();
    let mut cells = vec![SetSums::default(); model.states.len()];
    let mut fallback_samples = 0u64;
    let mut unscored_samples = 0u64;
    for quote in quotes {
        let target = quote
            .timestamp_ms
            .checked_add(horizon_ms)
            .or_overflow("horizon time")?;
        let future = if target < window.end_ms {
            let after = quotes.partition_point(|q| q.timestamp_ms <= target);
            after
                .checked_sub(1)
                .and_then(|last| quotes.get(last))
                .filter(|q| target.saturating_sub(q.timestamp_ms) <= model.config.max_quote_gap_ms)
        } else {
            None
        };
        let Some(future) = future else {
            unscored_samples = unscored_samples.saturating_add(1);
            continue;
        };
        let signal = model.signal(quote)?;
        let future_mid = future.mid()?;
        let mid = signal.mid;
        let weighted = quote.weighted_mid()?;
        let micro = signal.price;
        let side = match quote.bid_size.cmp(&quote.ask_size) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };
        overall.add(future_mid, mid, weighted, micro, side)?;
        if !matches!(signal.source, SignalSource::MicroPrice) {
            fallback_samples = fallback_samples.saturating_add(1);
        }
        let state = model.config.state_of(quote)?;
        if let Some(cell) = state.and_then(|s| cells.get_mut(s)) {
            cell.add(future_mid, mid, weighted, micro, side)?;
        }
    }
    let mut by_state = Vec::new();
    for (estimate, cell) in model.states.iter().zip(&cells) {
        if let Some(errors) = cell.finish()? {
            by_state.push(StateComparison {
                imbalance_bucket: estimate.imbalance_bucket,
                spread_ticks: estimate.spread_ticks,
                samples: cell.samples,
                errors,
            });
        }
    }
    Ok(HorizonComparison {
        horizon_ms,
        samples: overall.samples,
        fallback_samples,
        invalid_quotes,
        unscored_samples,
        overall: overall.finish()?,
        by_state,
    })
}

fn fit_counts(counts: &MicroPriceCalibrator) -> MMResult<MicroPriceFit> {
    let states = counts.states;
    let tick = to_f64(counts.config.tick_size)?;
    let row_of = |table: &[u64], x: usize| -> MMResult<Vec<f64>> {
        let start = x.checked_mul(states).or_overflow("transition row")?;
        let end = start.checked_add(states).or_overflow("transition row")?;
        table
            .get(start..end)
            .or_overflow("transition row")?
            .iter()
            .map(|c| to_f64(Decimal::from(*c)))
            .collect()
    };

    // Probabilities; a state left out keeps empty rows, so it is absorbing.
    let mut observations = vec![0u64; states];
    let mut q = vec![vec![0.0f64; states]; states];
    let mut t = vec![vec![0.0f64; states]; states];
    let mut expected_move = vec![0.0f64; states];
    let mut status = vec![StateStatus::Calibrated; states];
    for x in 0..states {
        let stay = row_of(&counts.stay, x)?;
        let moved = row_of(&counts.moved, x)?;
        let total: f64 = stay.iter().chain(&moved).sum();
        let count = total.to_u64().or_overflow("state observations")?;
        *observations.get_mut(x).or_overflow("state")? = count;
        if count < counts.config.min_state_observations {
            *status.get_mut(x).or_overflow("state")? = StateStatus::Rare;
            continue;
        }
        let half_ticks = counts.move_half_ticks.get(x).or_overflow("state")?;
        *expected_move.get_mut(x).or_overflow("state")? =
            to_f64(Decimal::from(*half_ticks))? / total;
        *q.get_mut(x).or_overflow("state")? = stay.iter().map(|c| c / total).collect();
        *t.get_mut(x).or_overflow("state")? = moved.iter().map(|c| c / total).collect();
    }
    mark_no_price_move(&mut q, &mut t, &mut expected_move, &mut status)?;
    if !status.contains(&StateStatus::Calibrated) {
        return Ok(MicroPriceFit::Insufficient(
            MicroPriceFailure::NoCalibratedState,
        ));
    }

    // Solve (1 - Q) [G¹ | F | B] = [R K | fallback indicator | T], where F is the
    // probability of reaching a fallback state before the next mid move.
    let mut system = Vec::with_capacity(states);
    for (x, (q_row, t_row)) in q.iter().zip(&t).enumerate() {
        let mut row: Vec<f64> = q_row
            .iter()
            .enumerate()
            .map(|(y, v)| if x == y { 1.0 - v } else { -v })
            .collect();
        row.push(*expected_move.get(x).or_overflow("state")?);
        let fallback = *status.get(x).or_overflow("state")? != StateStatus::Calibrated;
        row.push(if fallback { 1.0 } else { 0.0 });
        row.extend_from_slice(t_row);
        system.push(row);
    }
    if !gauss_jordan(&mut system, states) {
        return Ok(MicroPriceFit::Insufficient(
            MicroPriceFailure::SingularTransitions,
        ));
    }
    let first_column = states;
    let fallback_column = states.checked_add(1).or_overflow("system width")?;
    let matrix_start = states.checked_add(2).or_overflow("system width")?;
    let g1: Vec<f64> = system
        .iter()
        .map(|row| row.get(first_column).copied().or_overflow("solution"))
        .collect::<MMResult<_>>()?;
    let before_move: Vec<f64> = system
        .iter()
        .map(|row| row.get(fallback_column).copied().or_overflow("solution"))
        .collect::<MMResult<_>>()?;
    let b: Vec<&[f64]> = system
        .iter()
        .map(|row| row.get(matrix_start..).or_overflow("solution"))
        .collect::<MMResult<_>>()?;

    let Some(g_star) = sum_series(&g1, &b) else {
        return Ok(MicroPriceFit::Insufficient(MicroPriceFailure::NotConverged));
    };

    let fallback_within = fallback_within_moves(&before_move, &b);
    let n = counts.config.imbalance_buckets;
    let half_tick = tick / 2.0;
    let mut estimates = Vec::with_capacity(states);
    for x in 0..states {
        let index = u32::try_from(x).map_err(|_| overflow("state index"))?;
        let bucket = index.checked_rem(n).or_overflow("state bucket")?;
        let spread = index.checked_div(n).or_overflow("state spread")?;
        let spread_ticks = spread.checked_add(1).or_overflow("state spread")?;
        // Rounding must not break the antisymmetry the mirrored data has.
        let other = counts.config.mirror(x)?;
        let antisymmetric = |values: &[f64]| -> MMResult<f64> {
            let own = values.get(x).copied().or_overflow("state")?;
            let mirrored = values.get(other).copied().or_overflow("state")?;
            Ok((own - mirrored) / 2.0 * half_tick)
        };
        let first = antisymmetric(&g1)?;
        let star = antisymmetric(&g_star)?;
        let (Some(first_adjustment), Some(adjustment)) =
            (decimal_or_none(first), decimal_or_none(star))
        else {
            return Ok(MicroPriceFit::Insufficient(
                MicroPriceFailure::Unrepresentable,
            ));
        };
        let absorbing = *status.get(x).or_overflow("state")? != StateStatus::Calibrated;
        let share = if absorbing {
            1.0
        } else {
            fallback_within.get(x).copied().or_overflow("state")?
        };
        let Some(fallback_within_moves) = decimal_or_none(share.clamp(0.0, 1.0)) else {
            return Ok(MicroPriceFit::Insufficient(
                MicroPriceFailure::Unrepresentable,
            ));
        };
        // In `Decimal`, the type of the price, so that the signal stays inside
        // the quotes exactly.
        let half_spread = half_spread_of(&counts.config, spread_ticks)?;
        let state_status = *status.get(x).or_overflow("state")?;
        let status = match state_status {
            StateStatus::Calibrated if adjustment.abs() > half_spread => StateStatus::OutsideQuotes,
            other => other,
        };
        estimates.push(StateEstimate {
            imbalance_bucket: bucket.checked_add(1).or_overflow("state bucket")?,
            spread_ticks,
            observations: *observations.get(x).or_overflow("state")?,
            first_adjustment,
            adjustment,
            fallback_within_moves,
            status,
        });
    }
    if !estimates
        .iter()
        .any(|e| e.status == StateStatus::Calibrated)
    {
        return Ok(MicroPriceFit::Insufficient(
            MicroPriceFailure::NoCalibratedState,
        ));
    }
    Ok(MicroPriceFit::Calibrated(MicroPriceModel {
        config: counts.config,
        states: estimates,
    }))
}

/// Half the spread of a state, in price units.
fn half_spread_of(config: &MicroPriceConfig, spread_ticks: u32) -> MMResult<Decimal> {
    config
        .tick_size
        .checked_mul(Decimal::from(spread_ticks))
        .and_then(|spread| spread.checked_div(Decimal::TWO))
        .or_overflow("half spread")
}

/// `F + B F + ... ` over 10 mid moves, which stays at 1 on fallback states.
fn fallback_within_moves(before_move: &[f64], b: &[&[f64]]) -> Vec<f64> {
    let mut within = before_move.to_vec();
    for _ in 0..FALLBACK_HORIZON_MOVES {
        within = b
            .iter()
            .zip(before_move)
            .map(|(row, first)| first + row.iter().zip(&within).map(|(p, v)| p * v).sum::<f64>())
            .collect();
    }
    within
}

/// Marks and empties the calibrated states with no path to an observed mid
/// move. Reaching a rare state does not count: it would keep `1 - Q` invertible
/// with an adjustment of 0 that no observed move supports.
fn mark_no_price_move(
    q: &mut [Vec<f64>],
    t: &mut [Vec<f64>],
    expected_move: &mut [f64],
    status: &mut [StateStatus],
) -> MMResult<()> {
    let mut exits: Vec<bool> = t.iter().map(|row| row.iter().any(|v| *v > 0.0)).collect();
    loop {
        let mut changed = false;
        for x in 0..exits.len() {
            if exits.get(x).copied().unwrap_or(true) {
                continue;
            }
            let reaches = q
                .get(x)
                .or_overflow("state")?
                .iter()
                .zip(&exits)
                .any(|(v, exit)| *v > 0.0 && *exit);
            if reaches {
                *exits.get_mut(x).or_overflow("state")? = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (x, exit) in exits.iter().enumerate() {
        let state = status.get_mut(x).or_overflow("state")?;
        if !exit && *state == StateStatus::Calibrated {
            *state = StateStatus::NoPriceMove;
            q.get_mut(x).or_overflow("state")?.fill(0.0);
            t.get_mut(x).or_overflow("state")?.fill(0.0);
            *expected_move.get_mut(x).or_overflow("state")? = 0.0;
        }
    }
    Ok(())
}

/// Reduces the first `size` columns to the identity, in place. Returns false
/// when a pivot is too small.
fn gauss_jordan(rows: &mut [Vec<f64>], size: usize) -> bool {
    for column in 0..size {
        let best = rows
            .iter()
            .enumerate()
            .skip(column)
            .map(|(r, row)| (r, row.get(column).copied().unwrap_or(0.0).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((best, magnitude)) = best else {
            return false;
        };
        if magnitude < PIVOT_TOLERANCE {
            return false;
        }
        rows.swap(column, best);
        let Some(pivot_row) = rows.get(column).cloned() else {
            return false;
        };
        let pivot = pivot_row.get(column).copied().unwrap_or(0.0);
        let normalized: Vec<f64> = pivot_row.iter().map(|v| v / pivot).collect();
        for (r, row) in rows.iter_mut().enumerate() {
            if r == column {
                row.clone_from(&normalized);
                continue;
            }
            let factor = row.get(column).copied().unwrap_or(0.0);
            if factor != 0.0 {
                for (value, top) in row.iter_mut().zip(&normalized) {
                    *value -= factor * top;
                }
            }
        }
    }
    true
}

/// `G¹ + Σ Bⁱ G¹`, or `None` if it does not converge.
fn sum_series(g1: &[f64], b: &[&[f64]]) -> Option<Vec<f64>> {
    let mut term = g1.to_vec();
    let mut sum = g1.to_vec();
    for _ in 0..MAX_SERIES_TERMS {
        term = b
            .iter()
            .map(|row| row.iter().zip(&term).map(|(p, v)| p * v).sum())
            .collect();
        for (total, value) in sum.iter_mut().zip(&term) {
            *total += value;
        }
        let largest = term.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !largest.is_finite() {
            return None;
        }
        if largest < SERIES_TOLERANCE {
            return Some(sum);
        }
    }
    None
}

fn quotes_in<'a>(quotes: &'a [TopOfBook], window: &TimeWindow) -> MMResult<&'a [TopOfBook]> {
    if !quotes.is_sorted_by_key(|q| q.timestamp_ms) {
        return Err(MMError::InvalidMarketState(
            "quotes are not in chronological order".to_string(),
        ));
    }
    let from = quotes.partition_point(|q| q.timestamp_ms < window.start_ms);
    let to = quotes.partition_point(|q| q.timestamp_ms < window.end_ms);
    quotes.get(from..to).or_overflow("quote window")
}

/// Adds a quote to a hash, whatever the scale of its decimals.
fn hash_quote(hash: u128, quote: &TopOfBook) -> u128 {
    let hash = fnv(hash, &quote.timestamp_ms.to_le_bytes());
    let hash = fnv(hash, &quote.bid.normalize().serialize());
    let hash = fnv(hash, &quote.ask.normalize().serialize());
    let hash = fnv(hash, &quote.bid_size.normalize().serialize());
    fnv(hash, &quote.ask_size.normalize().serialize())
}

fn overflow(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} overflows"))
}

fn to_f64(value: Decimal) -> MMResult<f64> {
    value
        .to_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MMError::NumericalError(format!("{value} is not a finite f64")))
}

fn from_f64(value: f64) -> MMResult<Decimal> {
    decimal_or_none(value)
        .ok_or_else(|| MMError::NumericalError(format!("{value} is not a decimal")))
}

fn decimal_or_none(value: f64) -> Option<Decimal> {
    Some(value)
        .filter(|v| v.is_finite())
        .and_then(Decimal::from_f64)
}

/// Decoded form of [`MicroPriceConfig`], validated before use.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    tick_size: Decimal,
    imbalance_buckets: u32,
    spread_states: u32,
    min_state_observations: u64,
    max_quote_gap_ms: u64,
}

#[cfg(feature = "serde")]
impl TryFrom<RawConfig> for MicroPriceConfig {
    type Error = MMError;

    fn try_from(raw: RawConfig) -> MMResult<Self> {
        let config = Self {
            tick_size: raw.tick_size,
            imbalance_buckets: raw.imbalance_buckets,
            spread_states: raw.spread_states,
            min_state_observations: raw.min_state_observations,
            max_quote_gap_ms: raw.max_quote_gap_ms,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Decoded form of [`MicroPriceCalibrator`], validated before use.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCalibrator {
    config: MicroPriceConfig,
    states: usize,
    stay: Vec<u64>,
    moved: Vec<u64>,
    move_half_ticks: Vec<i64>,
    previous: Option<TopOfBook>,
    pairs: u64,
    skipped_spread_pairs: u64,
    skipped_gap_pairs: u64,
    largest_move_half_ticks: u64,
    hash: u128,
}

#[cfg(feature = "serde")]
impl TryFrom<RawCalibrator> for MicroPriceCalibrator {
    type Error = MMError;

    fn try_from(raw: RawCalibrator) -> MMResult<Self> {
        let states = raw.config.state_count()?;
        let cells = states.checked_mul(states).or_overflow("transition cells")?;
        if raw.states != states
            || raw.stay.len() != cells
            || raw.moved.len() != cells
            || raw.move_half_ticks.len() != states
        {
            return Err(MMError::InvalidConfiguration(format!(
                "calibrator state does not match {states} states"
            )));
        }
        if let Some(previous) = &raw.previous {
            raw.config.usable_state(previous)?;
        }
        check_mirrored_counts(&raw, states)?;
        Ok(Self {
            config: raw.config,
            states,
            stay: raw.stay,
            moved: raw.moved,
            move_half_ticks: raw.move_half_ticks,
            previous: raw.previous,
            pairs: raw.pairs,
            skipped_spread_pairs: raw.skipped_spread_pairs,
            skipped_gap_pairs: raw.skipped_gap_pairs,
            largest_move_half_ticks: raw.largest_move_half_ticks,
            hash: raw.hash,
        })
    }
}

/// Every pair counts twice, once as its mirror, so the tables must show it.
#[cfg(feature = "serde")]
fn check_mirrored_counts(raw: &RawCalibrator, states: usize) -> MMResult<()> {
    let broken = |what: &str| {
        Err(MMError::InvalidConfiguration(format!(
            "calibrator {what} do not match their mirrors"
        )))
    };
    let mut total = 0u64;
    for x in 0..states {
        let mirror_x = raw.config.mirror(x)?;
        let moves = raw.move_half_ticks.get(x).or_overflow("state")?;
        let mirror_moves = raw.move_half_ticks.get(mirror_x).or_overflow("state")?;
        if moves.checked_neg() != Some(*mirror_moves) {
            return broken("mid moves");
        }
        for y in 0..states {
            let cell = x.checked_mul(states).and_then(|v| v.checked_add(y));
            let mirror_y = raw.config.mirror(y)?;
            let mirror_cell = mirror_x
                .checked_mul(states)
                .and_then(|v| v.checked_add(mirror_y));
            let (cell, mirror_cell) = cell.zip(mirror_cell).or_overflow("transition cell")?;
            for table in [&raw.stay, &raw.moved] {
                let value = table.get(cell).or_overflow("transition cell")?;
                if Some(value) != table.get(mirror_cell) {
                    return broken("transition counts");
                }
                total = total.checked_add(*value).or_overflow("transition counts")?;
            }
        }
    }
    if raw.pairs.checked_mul(2) != Some(total) {
        return broken("pair counts");
    }
    Ok(())
}

/// Decoded form of [`MicroPriceModel`], validated before use.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    config: MicroPriceConfig,
    states: Vec<StateEstimate>,
}

#[cfg(feature = "serde")]
impl TryFrom<RawModel> for MicroPriceModel {
    type Error = MMError;

    fn try_from(raw: RawModel) -> MMResult<Self> {
        if raw.states.len() != raw.config.state_count()? {
            return Err(MMError::InvalidConfiguration(format!(
                "{} states do not match the configuration",
                raw.states.len()
            )));
        }
        let n = raw.config.imbalance_buckets;
        for (x, state) in raw.states.iter().enumerate() {
            let index = u32::try_from(x).map_err(|_| overflow("state index"))?;
            let bucket = index.checked_rem(n).and_then(|b| b.checked_add(1));
            let spread = index.checked_div(n).and_then(|s| s.checked_add(1));
            let half_spread = half_spread_of(&raw.config, state.spread_ticks)?;
            if bucket != Some(state.imbalance_bucket)
                || spread != Some(state.spread_ticks)
                || (state.status == StateStatus::Calibrated && state.adjustment.abs() > half_spread)
                || (state.status == StateStatus::OutsideQuotes
                    && state.adjustment.abs() <= half_spread)
                || !state_values_are_consistent(state)
                || !state_mirrors(state, raw.states.get(raw.config.mirror(x)?))
            {
                return Err(MMError::InvalidConfiguration(format!(
                    "invalid state {x}: {state:?}"
                )));
            }
        }
        Ok(Self {
            config: raw.config,
            states: raw.states,
        })
    }
}

/// The probability is in `[0, 1]`, and an absorbing state has 1 and adjustment 0.
#[cfg(feature = "serde")]
fn state_values_are_consistent(state: &StateEstimate) -> bool {
    let share = state.fallback_within_moves;
    let in_range = share >= Decimal::ZERO && share <= Decimal::ONE;
    let consistent = match state.status {
        StateStatus::Calibrated | StateStatus::OutsideQuotes => true,
        StateStatus::Rare | StateStatus::NoPriceMove => {
            share == Decimal::ONE && state.adjustment.is_zero()
        }
    };
    in_range && consistent
}

/// A state's adjustments are the negatives of its mirror's, with the same status
/// and observations, so the signal stays antisymmetric.
#[cfg(feature = "serde")]
fn state_mirrors(state: &StateEstimate, mirror: Option<&StateEstimate>) -> bool {
    mirror.is_some_and(|m| {
        state.status == m.status
            && state.observations == m.observations
            && state.adjustment.checked_add(m.adjustment) == Some(Decimal::ZERO)
            && state.first_adjustment.checked_add(m.first_adjustment) == Some(Decimal::ZERO)
    })
}
