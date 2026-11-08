//! A queue-reactive order flow simulator of aggregated queues.
//!
//! After Huang, Lehalle and Rosenbaum, *Simulating and analyzing order book data:
//! The queue-reactive model* (arXiv:1312.0563v2): Model III, with the queues of
//! Model I between reference price changes. It simulates market flow in
//! aggregated queues. It is not a replica of an exchange, models no individual
//! queue priority, and does not model our own orders.
//!
//! # Model
//!
//! Around a reference price, the bid and the ask have `K` limits, with limit `i`
//! at `i - 1/2` ticks from it. A queue of size `n`, in units of its average event
//! size, changes by one with intensities `λL(n)`, `λC(n)` and `λM(n)`, for limit
//! orders, cancellations and market orders, equal on both sides. When an event
//! moves the mid up and the first ask queue is empty, the reference price rises
//! one tick with probability `θ`, and the mirror holds for a move down. Then the
//! queues shift to their neighbours, renormalized by the event sizes, the
//! new outer queue is drawn from its invariant law, and with probability
//! `θ_reinit` the whole book is redrawn from the invariant laws.
//!
//! # Departures from the paper
//!
//! - The intensities are estimated as in section 2.3.2: per queue size, events
//!   of a type over total waiting time. `θ` and `θ_reinit` are estimated from
//!   observed outcomes, not from the paper's volatility and mean reversion surface.
//! - Queues are capped at `max_queue`, so that the state is bounded: a limit order
//!   at the cap does not exist. That is a modelling restriction of this crate.
//! - The estimator follows the paper's restart of the recording at a reference
//!   move, so a stay cut by one enters neither the events nor the waiting time.
//!   With a total queue rate `r` and a reset hazard `h`, the estimate tends to
//!   `r + h` rather than `r`: an upward bias of `h / r` that matters for queues
//!   with slow events when the reference moves often.
//! - A side with no volume within `K` limits has its best price taken one limit
//!   further, `K + 1/2` ticks from the reference price, to define a mid.
//!
//! # Unsupported states
//!
//! A queue size observed fewer than `min_events` times is unsupported. Under
//! [`UnsupportedPolicy::Stop`] the simulator stops there with a typed outcome.
//! Under [`UnsupportedPolicy::NearestSupported`] it uses the rates of the nearest
//! supported size, without an impossible departure from an empty queue. The
//! report states the coverage either way. Under `Stop`, an unsupported size 0
//! leaves an invariant law with no mass above it, so the simulator stops at once
//! if a queue starts empty.
//!
//! # Determinism
//!
//! The same parameters, seed and build reproduce a trajectory exactly. The
//! sampling goes through `f64`, so bit equality across platforms is not claimed.
//! A state can be stored, restored and continued with the same result.
//!
//! # Example
//!
//! ```
//! use market_maker_rs::backtest::queue_reactive::{
//!     QrParameters, QrSimulator, UnsupportedPolicy,
//! };
//! use market_maker_rs::dec;
//!
//! let rates = vec![(1.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.0, 1.0, 0.5)];
//! let params = QrParameters::new(
//!     dec!(0.01), 0.5, 0.1, vec![dec!(1)], vec![rates], vec![vec![true; 3]],
//!     UnsupportedPolicy::Stop,
//! ).unwrap();
//! let mut simulator = QrSimulator::new(params, 7, 100).unwrap();
//! assert!(simulator.step().is_ok());
//! ```

use crate::Decimal;
use crate::strategy::walk_forward::{FNV_OFFSET, TimeWindow, fnv};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Most limits per side the fixed size state holds.
pub const MAX_LIMITS: usize = 5;
/// Largest queue size accepted, in event sizes.
pub const MAX_QUEUE_BOUND: u32 = 4_096;
/// Largest rate accepted, in events per second.
pub const MAX_RATE: f64 = 1e9;
/// Most events one [`QrSimulator::run`] keeps.
pub const MAX_RUN_EVENTS: usize = 50_000_000;
const RNG_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;
const ZERO_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Scale from a 53 bit integer to `[0, 1)`.
const UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Type of a queue event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum QrEventKind {
    /// A limit order adds to the queue.
    Limit,
    /// A cancellation takes from the queue.
    Cancel,
    /// A market order takes from the queue.
    Market,
}

/// Side of a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum QrSide {
    /// Below the reference price.
    Bid,
    /// Above the reference price.
    Ask,
}

/// What an estimation does with an event whose waiting time is zero, as the events of one
/// block of a capture whose times are inferred per block are (#290).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ZeroWaitPolicy {
    /// Leaves it out and counts it as excluded, as before.
    #[default]
    Drop,
    /// Counts it at its queue size, adding no time there: the queue spent no measured time at
    /// a size it passed through inside a block. A size whose counted events all waited zero has
    /// no rate, so it is unsupported whatever its count.
    Count,
}

/// What to do at a queue size with too few observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnsupportedPolicy {
    /// Stop with [`QrStep::Unsupported`].
    Stop,
    /// Use the rates of the nearest supported size.
    NearestSupported,
}

/// Rates per queue size, `(λL, λC, λM)`, for one limit.
pub type RateTable = Vec<(f64, f64, f64)>;

/// Exportable parameters of the simulator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(into = "RawParameters", try_from = "RawParameters")
)]
pub struct QrParameters {
    tick_size: Decimal,
    theta: f64,
    theta_reinit: f64,
    event_sizes: Vec<Decimal>,
    rates: Vec<RateTable>,
    supported: Vec<Vec<bool>>,
    policy: UnsupportedPolicy,
    invariant: Vec<Vec<f64>>,
    /// Touch resets, if any (#290).
    resets: Option<Resets>,
}

/// Rates at which the first limit's queue is redrawn, per size, and the law it is redrawn
/// from.
#[derive(Debug, Clone, PartialEq)]
struct Resets {
    rates: Vec<f64>,
    law: Vec<f64>,
}

/// A float as text, which `serde_json` round trips exactly by default where a JSON
/// number would not. Parameters and states must reproduce a trajectory bit for bit.
#[cfg(feature = "serde")]
fn float_text(value: f64) -> String {
    format!("{value:?}")
}

#[cfg(feature = "serde")]
fn text_float(text: &str) -> MMResult<f64> {
    text.parse::<f64>()
        .map_err(|error| MMError::InvalidConfiguration(format!("float {text}: {error}")))
}

/// Exported form of [`QrParameters`], validated on decode. The invariant laws are
/// not part of it, since they follow from the rates.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawParameters {
    tick_size: Decimal,
    theta: String,
    theta_reinit: String,
    event_sizes: Vec<Decimal>,
    rates: Vec<Vec<[String; 3]>>,
    supported: Vec<Vec<bool>>,
    policy: UnsupportedPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reset_rates: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reset_law: Option<Vec<String>>,
}

#[cfg(feature = "serde")]
impl From<QrParameters> for RawParameters {
    fn from(params: QrParameters) -> Self {
        Self {
            tick_size: params.tick_size,
            theta: float_text(params.theta),
            theta_reinit: float_text(params.theta_reinit),
            event_sizes: params.event_sizes,
            rates: params
                .rates
                .iter()
                .map(|table| {
                    table
                        .iter()
                        .map(|(l, c, m)| [float_text(*l), float_text(*c), float_text(*m)])
                        .collect()
                })
                .collect(),
            supported: params.supported,
            policy: params.policy,
            reset_rates: params
                .resets
                .as_ref()
                .map(|r| r.rates.iter().copied().map(float_text).collect()),
            reset_law: params
                .resets
                .map(|r| r.law.into_iter().map(float_text).collect()),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<RawParameters> for QrParameters {
    type Error = MMError;

    fn try_from(raw: RawParameters) -> MMResult<Self> {
        let rates = raw
            .rates
            .iter()
            .map(|table| {
                table
                    .iter()
                    .map(|[l, c, m]| Ok((text_float(l)?, text_float(c)?, text_float(m)?)))
                    .collect::<MMResult<RateTable>>()
            })
            .collect::<MMResult<Vec<_>>>()?;
        let params = Self::new(
            raw.tick_size,
            text_float(&raw.theta)?,
            text_float(&raw.theta_reinit)?,
            raw.event_sizes,
            rates,
            raw.supported,
            raw.policy,
        )?;
        let floats = |texts: Vec<String>| {
            texts
                .iter()
                .map(|t| text_float(t))
                .collect::<MMResult<Vec<_>>>()
        };
        match (raw.reset_rates, raw.reset_law) {
            (None, None) => Ok(params),
            (Some(rates), Some(law)) => params.with_resets(floats(rates)?, floats(law)?),
            _ => Err(MMError::InvalidConfiguration(
                "reset rates and their law come together".to_string(),
            )),
        }
    }
}

impl QrParameters {
    /// Validates parameters and computes the invariant laws of section 2.3.3.
    ///
    /// `rates[d]` and `supported[d]` hold limit `d + 1`, one row per queue size
    /// from 0 to the same `max_queue`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for no limits or more than
    /// [`MAX_LIMITS`]; a tick or event size that is not positive; a probability
    /// out of `[0, 1]`; tables of different shapes or above [`MAX_QUEUE_BOUND`]; a
    /// rate that is negative, not finite or above [`MAX_RATE`]; a departure from an empty queue; an
    /// arrival at the cap; a limit with no supported size; or a size that is reached
    /// and has no departures, which is absorbing and has no law in the supported
    /// product form.
    pub fn new(
        tick_size: Decimal,
        theta: f64,
        theta_reinit: f64,
        event_sizes: Vec<Decimal>,
        rates: Vec<RateTable>,
        supported: Vec<Vec<bool>>,
        policy: UnsupportedPolicy,
    ) -> MMResult<Self> {
        let invalid = |what: String| Err(MMError::InvalidConfiguration(what));
        if rates.is_empty()
            || rates.len() > MAX_LIMITS
            || event_sizes.len() != rates.len()
            || supported.len() != rates.len()
        {
            return invalid(format!("{} limits with mismatched tables", rates.len()));
        }
        if tick_size <= Decimal::ZERO || event_sizes.iter().any(|s| *s <= Decimal::ZERO) {
            return invalid(format!(
                "sizes must be positive: {tick_size} {event_sizes:?}"
            ));
        }
        if !(0.0..=1.0).contains(&theta) || !(0.0..=1.0).contains(&theta_reinit) {
            return invalid(format!(
                "probabilities out of range: {theta} {theta_reinit}"
            ));
        }
        let rows = rates.first().map_or(0, Vec::len);
        let bound = usize::try_from(MAX_QUEUE_BOUND).unwrap_or(usize::MAX);
        if rows < 2 || rows > bound.saturating_add(1) {
            return invalid(format!("{rows} rows per limit"));
        }
        for (table, flags) in rates.iter().zip(&supported) {
            if table.len() != rows || flags.len() != rows || !flags.contains(&true) {
                return invalid("a limit has the wrong shape or no support".to_string());
            }
            let valid = |v: f64| v.is_finite() && (0.0..=MAX_RATE).contains(&v);
            if table
                .iter()
                .any(|(l, c, m)| !valid(*l) || !valid(*c) || !valid(*m))
            {
                return invalid("rates must be finite, not negative and bounded".to_string());
            }
            let empty_departs = table.first().is_some_and(|(_, c, m)| *c > 0.0 || *m > 0.0);
            let cap_arrives = table.last().is_some_and(|(l, _, _)| *l > 0.0);
            if empty_departs || cap_arrives {
                return invalid("departure from empty or arrival at the cap".to_string());
            }
        }
        let invariant = rates
            .iter()
            .zip(&supported)
            .map(|(table, flags)| invariant_law(table, flags, policy))
            .collect::<MMResult<Vec<_>>>()?;
        Ok(Self {
            tick_size,
            theta,
            theta_reinit,
            event_sizes,
            rates,
            supported,
            policy,
            invariant,
            resets: None,
        })
    }

    /// Returns the size of a tick.
    #[must_use]
    pub fn tick_size(&self) -> Decimal {
        self.tick_size
    }

    /// Returns the number of limits per side, `K`.
    #[must_use]
    pub fn limits(&self) -> usize {
        self.rates.len()
    }

    /// Returns the largest queue size.
    #[must_use]
    pub fn max_queue(&self) -> usize {
        self.rates.first().map_or(0, Vec::len).saturating_sub(1)
    }

    /// Returns the probability a reference price move follows an eligible mid move.
    #[must_use]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Returns the probability a reference price move redraws the book.
    #[must_use]
    pub fn theta_reinit(&self) -> f64 {
        self.theta_reinit
    }

    /// Returns the rates of limit `depth + 1`.
    #[must_use]
    pub fn rates(&self, depth: usize) -> Option<&[(f64, f64, f64)]> {
        self.rates.get(depth).map(Vec::as_slice)
    }

    /// Returns the invariant law of limit `depth + 1`.
    #[must_use]
    pub fn invariant(&self, depth: usize) -> Option<&[f64]> {
        self.invariant.get(depth).map(Vec::as_slice)
    }

    /// Returns whether queue size `n` of limit `depth + 1` is supported.
    #[must_use]
    pub fn is_supported(&self, depth: usize, n: usize) -> bool {
        self.supported
            .get(depth)
            .and_then(|flags| flags.get(n))
            .copied()
            .unwrap_or(false)
    }

    /// Returns the event sizes, per limit.
    #[must_use]
    pub fn event_sizes(&self) -> &[Decimal] {
        &self.event_sizes
    }

    /// Adds touch resets (#290): with rate `rates[n]`, a first-limit queue of size `n` is
    /// redrawn from `law`, a probability per queue size, so its life ends without emptying, as
    /// a touch's does when the price moves away from it. The queue's clock restarts; no other
    /// queue, nor the reference, changes, and the reset is no queue event, so it is reported as
    /// [`QrStep::Reset`]. Without resets the simulator runs as before. The invariant laws
    /// ignore the resets.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] unless `rates` and `law` each hold one value
    /// per queue size, the rates finite, not negative and at most [`MAX_RATE`] with none at
    /// size 0, and the law finite, not negative and summing to 1 within `1e-9`.
    pub fn with_resets(mut self, rates: Vec<f64>, law: Vec<f64>) -> MMResult<Self> {
        let rows = self.max_queue().saturating_add(1);
        let valid = |v: &f64| v.is_finite() && (0.0..=MAX_RATE).contains(v);
        let probability = |v: &f64| v.is_finite() && *v >= 0.0;
        if rates.len() != rows
            || law.len() != rows
            || !rates.iter().all(valid)
            || rates.first().is_some_and(|r| *r > 0.0)
            || !law.iter().all(probability)
            || (law.iter().sum::<f64>() - 1.0).abs() > 1e-9
        {
            return Err(MMError::InvalidConfiguration(format!(
                "resets: {} rates and a law of {} for {rows} sizes; rates finite, bounded and \
                 none at 0; the law a probability",
                rates.len(),
                law.len()
            )));
        }
        self.resets = Some(Resets { rates, law });
        Ok(self)
    }

    /// Returns the touch reset rates per queue size, if any.
    #[must_use]
    pub fn reset_rates(&self) -> Option<&[f64]> {
        self.resets.as_ref().map(|r| r.rates.as_slice())
    }

    /// Returns the law a reset redraws the touch queue from, if any.
    #[must_use]
    pub fn reset_law(&self) -> Option<&[f64]> {
        self.resets.as_ref().map(|r| r.law.as_slice())
    }

    /// Returns the FNV-1a 128 of the parameters, to tie a trajectory to them.
    #[must_use]
    pub fn fingerprint(&self) -> u128 {
        let mut hash = fnv(FNV_OFFSET, &self.tick_size.normalize().serialize());
        hash = fnv(hash, &self.theta.to_bits().to_le_bytes());
        hash = fnv(hash, &self.theta_reinit.to_bits().to_le_bytes());
        let policy: u8 = match self.policy {
            UnsupportedPolicy::Stop => 0,
            UnsupportedPolicy::NearestSupported => 1,
        };
        hash = fnv(hash, &[policy]);
        for size in &self.event_sizes {
            hash = fnv(hash, &size.normalize().serialize());
        }
        for (table, flags) in self.rates.iter().zip(&self.supported) {
            for ((l, c, m), flag) in table.iter().zip(flags) {
                hash = fnv(hash, &l.to_bits().to_le_bytes());
                hash = fnv(hash, &c.to_bits().to_le_bytes());
                hash = fnv(hash, &m.to_bits().to_le_bytes());
                hash = fnv(hash, &[u8::from(*flag)]);
            }
        }
        if let Some(resets) = &self.resets {
            hash = fnv(hash, b"resets");
            for value in resets.rates.iter().chain(&resets.law) {
                hash = fnv(hash, &value.to_bits().to_le_bytes());
            }
        }
        hash
    }

    /// Returns the constant intensity baseline: each rate is its average under the
    /// invariant law, kept at zero where a queue rule requires it, with the same
    /// `θ`, sizes and cap. The law is then a different one, as it should be. Touch reset
    /// rates, if any, become their average under the first limit's law; their redraw law stays.
    ///
    /// # Errors
    ///
    /// As [`QrParameters::new`].
    pub fn constant_baseline(&self) -> MMResult<Self> {
        let rates = self
            .rates
            .iter()
            .zip(&self.supported)
            .zip(&self.invariant)
            .map(|((table, flags), law)| {
                // Under the same rates that produced the law, so that the policy
                // applies to both.
                let mean = |pick: &dyn Fn(&(f64, f64, f64)) -> f64| -> f64 {
                    law.iter()
                        .enumerate()
                        .filter_map(|(n, p)| {
                            resolve(table, flags, self.policy, n).map(|row| pick(&row) * p)
                        })
                        .sum()
                };
                let (limit, cancel, market) = (mean(&|r| r.0), mean(&|r| r.1), mean(&|r| r.2));
                let last = table.len().saturating_sub(1);
                (0..table.len())
                    .map(|n| {
                        (
                            if n == last { 0.0 } else { limit },
                            if n == 0 { 0.0 } else { cancel },
                            if n == 0 { 0.0 } else { market },
                        )
                    })
                    .collect::<RateTable>()
            })
            .collect();
        let supported = self
            .supported
            .iter()
            .map(|flags| vec![true; flags.len()])
            .collect();
        let baseline = Self::new(
            self.tick_size,
            self.theta,
            self.theta_reinit,
            self.event_sizes.clone(),
            rates,
            supported,
            self.policy,
        )?;
        match (&self.resets, self.invariant.first()) {
            (Some(resets), Some(law)) => {
                let mean: f64 = resets.rates.iter().zip(law).map(|(r, p)| r * p).sum();
                let rates = (0..resets.rates.len())
                    .map(|n| if n == 0 { 0.0 } else { mean })
                    .collect();
                baseline.with_resets(rates, resets.law.clone())
            }
            _ => Ok(baseline),
        }
    }

    /// The rates at queue size `n`, or `None` where the policy stops.
    fn rates_at(&self, depth: usize, n: usize) -> Option<(f64, f64, f64)> {
        let table = self.rates.get(depth)?;
        let flags = self.supported.get(depth)?;
        resolve(table, flags, self.policy, n)
    }
}

/// The rates at `n` under the policy.
fn resolve(
    table: &[(f64, f64, f64)],
    flags: &[bool],
    policy: UnsupportedPolicy,
    n: usize,
) -> Option<(f64, f64, f64)> {
    if flags.get(n).copied().unwrap_or(false) {
        return table.get(n).copied();
    }
    match policy {
        UnsupportedPolicy::Stop => None,
        UnsupportedPolicy::NearestSupported => {
            let nearest = flags
                .iter()
                .enumerate()
                .filter(|(_, flag)| **flag)
                .map(|(i, _)| i)
                .min_by_key(|i| i.abs_diff(n))?;
            let (limit, cancel, market) = table.get(nearest).copied()?;
            let last = table.len().saturating_sub(1);
            Some((
                if n == last { 0.0 } else { limit },
                if n == 0 { 0.0 } else { cancel },
                if n == 0 { 0.0 } else { market },
            ))
        }
    }
}

/// `π(n) ∝ Π ρ(j - 1)`, with `ρ(n) = λL(n) / (λC(n + 1) + λM(n + 1))`, cut where
/// the chain cannot arrive. A chain that arrives and cannot leave is absorbing, and
/// is refused as outside this product form.
fn invariant_law(
    table: &[(f64, f64, f64)],
    flags: &[bool],
    policy: UnsupportedPolicy,
) -> MMResult<Vec<f64>> {
    let mut law = Vec::with_capacity(table.len());
    let mut weight = 1.0f64;
    law.push(weight);
    for n in 1..table.len() {
        let below = resolve(table, flags, policy, n.saturating_sub(1));
        let here = resolve(table, flags, policy, n);
        let ratio = match (below, here) {
            (Some((limit, _, _)), Some((_, cancel, market))) => {
                let departures = cancel + market;
                if departures > 0.0 {
                    limit / departures
                } else if limit > 0.0 && weight > 0.0 {
                    return Err(MMError::InvalidConfiguration(format!(
                        "queue size {n} is reached but has no departures: absorbing, with no law in the supported product form"
                    )));
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        weight *= ratio;
        law.push(weight);
    }
    let total: f64 = law.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(MMError::NumericalError(format!(
            "invariant law normalizes to {total}"
        )));
    }
    for value in &mut law {
        *value /= total;
    }
    Ok(law)
}

/// Restorable state of a simulator: fixed size and `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "RawState", try_from = "RawState"))]
pub struct QrState {
    /// Bid queues; entry `d` is limit `d + 1`.
    pub bid: [u32; MAX_LIMITS],
    /// Ask queues; entry `d` is limit `d + 1`.
    pub ask: [u32; MAX_LIMITS],
    /// Reference price, in ticks.
    pub reference_ticks: i64,
    /// Time of the last event, in seconds.
    pub time_seconds: f64,
    /// How far runs have reached, in seconds; at least `time_seconds`. Between
    /// events the book does not change, so this only places the next run's grid.
    pub clock_seconds: f64,
    /// Time of the last event per bid queue, restarted when the reference moves.
    pub bid_last: [f64; MAX_LIMITS],
    /// Time of the last event per ask queue, restarted when the reference moves.
    pub ask_last: [f64; MAX_LIMITS],
    /// Random generator state.
    pub rng_state: u64,
    /// Events so far.
    pub events: u64,
    /// Fingerprint of the parameters this state belongs to.
    pub parameters_fingerprint: u128,
}

/// Exported form of [`QrState`], with exact floats.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawState {
    bid: [u32; MAX_LIMITS],
    ask: [u32; MAX_LIMITS],
    reference_ticks: i64,
    time_seconds: String,
    clock_seconds: String,
    bid_last: [String; MAX_LIMITS],
    ask_last: [String; MAX_LIMITS],
    rng_state: u64,
    events: u64,
    parameters_fingerprint: String,
}

#[cfg(feature = "serde")]
impl From<QrState> for RawState {
    fn from(state: QrState) -> Self {
        Self {
            bid: state.bid,
            ask: state.ask,
            reference_ticks: state.reference_ticks,
            time_seconds: float_text(state.time_seconds),
            clock_seconds: float_text(state.clock_seconds),
            bid_last: state.bid_last.map(float_text),
            ask_last: state.ask_last.map(float_text),
            rng_state: state.rng_state,
            events: state.events,
            parameters_fingerprint: state.parameters_fingerprint.to_string(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<RawState> for QrState {
    type Error = MMError;

    fn try_from(raw: RawState) -> MMResult<Self> {
        let times = |texts: &[String; MAX_LIMITS]| -> MMResult<[f64; MAX_LIMITS]> {
            let mut out = [0.0; MAX_LIMITS];
            for (slot, text) in out.iter_mut().zip(texts) {
                *slot = text_float(text)?;
            }
            Ok(out)
        };
        Ok(Self {
            bid: raw.bid,
            ask: raw.ask,
            reference_ticks: raw.reference_ticks,
            time_seconds: text_float(&raw.time_seconds)?,
            clock_seconds: text_float(&raw.clock_seconds)?,
            bid_last: times(&raw.bid_last)?,
            ask_last: times(&raw.ask_last)?,
            rng_state: raw.rng_state,
            events: raw.events,
            parameters_fingerprint: raw.parameters_fingerprint.parse().map_err(|_| {
                MMError::InvalidConfiguration(format!("fingerprint {}", raw.parameters_fingerprint))
            })?,
        })
    }
}

/// A queue event, as an estimation input: a day of them, in order, from data or
/// from the simulator.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QueueObservation {
    /// Time of the event, in milliseconds.
    pub timestamp_ms: u64,
    /// Side of the queue. The rates are symmetric, so both sides pool.
    pub side: QrSide,
    /// Limit from 0, nearest first.
    pub depth: u8,
    /// Type of the event.
    pub kind: QrEventKind,
    /// Queue size before the event, in event sizes.
    pub queue_before: u32,
    /// Time since the previous event at this queue, in seconds.
    pub waiting_seconds: f64,
}

/// The size of one queue event in the data, in units of the asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SizedEvent {
    /// Time of the event, in milliseconds.
    pub timestamp_ms: u64,
    /// Limit from 0, nearest first. Both sides pool, as the rates do.
    pub depth: u8,
    /// Its size.
    pub size: Decimal,
}

/// Each block's events at each queue collapsed into their net change (#290).
///
/// A block at a queue (side and limit) is a run of that queue's events, other queues' events
/// aside, each after the first with a zero wait, as the events of one block of a capture whose
/// times are inferred per block are. Each event counts as one event size, as the estimator reads
/// it, whatever the queue sizes it reports. The block's net change becomes that many events of
/// one kind, from the first event's queue size by one at a time: limit orders for a net arrival;
/// for a net departure, market orders if the block holds one, cancellations otherwise. They sit
/// where the block's first event was; the first carries its time and wait, the others wait zero.
/// A zero wait after a move of the reference within one block time joins the queue's open block,
/// since an event does not say that its queue's clock restarted. A block with no net change leaves no event, and its wait
/// is carried to the next event its queue keeps, so the time the queue spent at its size is not
/// lost. Events otherwise keep their order.
#[must_use]
pub fn collapse_blocks(events: &[QueueObservation]) -> Vec<QueueObservation> {
    struct Block {
        first: QueueObservation,
        net: i64,
        market: bool,
    }
    let mut blocks: Vec<Block> = Vec::new();
    let mut open: std::collections::HashMap<(bool, u8), usize> = std::collections::HashMap::new();
    for event in events {
        let key = (matches!(event.side, QrSide::Bid), event.depth);
        let step = match event.kind {
            QrEventKind::Limit => 1,
            QrEventKind::Cancel | QrEventKind::Market => -1,
        };
        let joined = open
            .get(&key)
            .and_then(|&b| blocks.get_mut(b))
            .filter(|_| event.waiting_seconds == 0.0);
        if let Some(block) = joined {
            block.net = block.net.saturating_add(step);
            block.market |= event.kind == QrEventKind::Market;
        } else {
            open.insert(key, blocks.len());
            blocks.push(Block {
                first: *event,
                net: step,
                market: event.kind == QrEventKind::Market,
            });
        }
    }
    let mut carried: std::collections::HashMap<(bool, u8), f64> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(events.len());
    for block in blocks {
        let first = block.first;
        let key = (matches!(first.side, QrSide::Bid), first.depth);
        let wait = first.waiting_seconds + carried.remove(&key).unwrap_or(0.0);
        if block.net == 0 {
            carried.insert(key, wait);
            continue;
        }
        let kind = if block.net > 0 {
            QrEventKind::Limit
        } else if block.market {
            QrEventKind::Market
        } else {
            QrEventKind::Cancel
        };
        let mut queue = i64::from(first.queue_before);
        for k in 0..block.net.unsigned_abs() {
            out.push(QueueObservation {
                kind,
                queue_before: u32::try_from(queue.max(0)).unwrap_or(u32::MAX),
                waiting_seconds: if k == 0 { wait } else { 0.0 },
                ..first
            });
            queue = queue.saturating_add(block.net.signum());
        }
    }
    out
}

/// Event sizes together with what they were estimated from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct EventSizeRecord {
    /// The interval estimated on.
    pub training: TimeWindow,
    /// FNV-1a 128 of the events inside it.
    pub dataset_hash: u128,
    /// Events used, per limit.
    pub events: Vec<u64>,
    /// Events left out for a size that is not positive or a limit out of range.
    pub excluded_events: u64,
    /// The mean size per limit, for [`estimate_parameters`], or why there are none.
    pub sizes: Result<Vec<Decimal>, QrFailure>,
}

/// A mid move while the opposite first queue was empty, and what followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReferenceObservation {
    /// Time of the move, in milliseconds.
    pub timestamp_ms: u64,
    /// Whether the reference price moved.
    pub moved: bool,
    /// Whether the book was redrawn. Only with `moved`.
    pub reinitialized: bool,
}

/// A touch reset (#290): a first-limit queue whose life ended without emptying, at its size
/// then. From data, a side's touch at a move of the mid that emptied no touch.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResetObservation {
    /// Time of the reset, in milliseconds.
    pub timestamp_ms: u64,
    /// Side of the queue.
    pub side: QrSide,
    /// Queue size before the reset, in event sizes.
    pub queue_before: u32,
    /// Time since the previous event at this queue, in seconds: the time it spent at
    /// `queue_before` before the reset.
    pub waiting_seconds: f64,
    /// The new touch's size, in event sizes, if known.
    pub queue_after: Option<u32>,
}

/// The result of a simulator step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum QrStep {
    /// An event happened.
    Event {
        /// The event.
        observation: QueueObservation,
        /// The reference price outcome, if the event made one eligible.
        reference: Option<ReferenceObservation>,
    },
    /// The state has a queue size the policy does not simulate. The state stays.
    Unsupported {
        /// Side of the queue.
        side: QrSide,
        /// Limit from 0.
        depth: usize,
        /// Queue size.
        queue: u32,
    },
    /// Nothing can happen again: all rates are zero, or the next event is too
    /// close to the current time for it to advance.
    Stalled,
    /// A run reached its limit of events before its horizon.
    EventLimit,
    /// A touch reset under [`QrParameters::with_resets`]: the first limit's queue on
    /// `reset.side` was redrawn to `queue_after`.
    Reset {
        /// The reset.
        reset: ResetObservation,
        /// The queue's new size.
        queue_after: u32,
        /// The reference price outcome, if the new size made one eligible.
        reference: Option<ReferenceObservation>,
    },
}

/// A simulator of queues under [`QrParameters`].
#[derive(Debug, Clone, PartialEq)]
pub struct QrSimulator {
    params: QrParameters,
    state: QrState,
}

impl QrSimulator {
    /// Starts from a book drawn from the invariant laws.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on a failed conversion.
    pub fn new(params: QrParameters, seed: u64, reference_ticks: i64) -> MMResult<Self> {
        let parameters_fingerprint = params.fingerprint();
        let mut simulator = Self {
            params,
            state: QrState {
                bid: [0; MAX_LIMITS],
                ask: [0; MAX_LIMITS],
                reference_ticks,
                time_seconds: 0.0,
                clock_seconds: 0.0,
                bid_last: [0.0; MAX_LIMITS],
                ask_last: [0.0; MAX_LIMITS],
                rng_state: if seed == 0 { ZERO_SEED } else { seed },
                events: 0,
                parameters_fingerprint,
            },
        };
        simulator.redraw()?;
        Ok(simulator)
    }

    /// Continues from a stored state.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a state of other parameters,
    /// or with a queue above the cap or outside the limits, a time that is not
    /// finite and not negative, or a zero generator state.
    pub fn from_state(params: QrParameters, state: QrState) -> MMResult<Self> {
        let limits = params.limits();
        let max = u32::try_from(params.max_queue()).unwrap_or(u32::MAX);
        let queues_ok = |queues: &[u32; MAX_LIMITS]| {
            queues
                .iter()
                .enumerate()
                .all(|(d, q)| if d < limits { *q <= max } else { *q == 0 })
        };
        let time_ok = |t: f64| t.is_finite() && t >= 0.0 && t <= state.time_seconds;
        if state.parameters_fingerprint != params.fingerprint()
            || !queues_ok(&state.bid)
            || !queues_ok(&state.ask)
            || !state.time_seconds.is_finite()
            || state.time_seconds < 0.0
            || !state.clock_seconds.is_finite()
            || state.clock_seconds < state.time_seconds
            || !state
                .bid_last
                .iter()
                .chain(&state.ask_last)
                .all(|t| time_ok(*t))
            || state.rng_state == 0
        {
            return Err(MMError::InvalidConfiguration(format!(
                "state does not fit the parameters: {state:?}"
            )));
        }
        Ok(Self { params, state })
    }

    /// Returns the state, to store it.
    #[must_use]
    pub fn state(&self) -> QrState {
        self.state
    }

    /// Returns the parameters, to export them.
    #[must_use]
    pub fn parameters(&self) -> &QrParameters {
        &self.params
    }

    /// Returns the spread in ticks, counting an empty side as one limit further.
    #[must_use]
    pub fn spread_ticks(&self) -> u32 {
        self.best_offset(&self.state.bid)
            .saturating_add(self.best_offset(&self.state.ask))
            .saturating_add(1)
    }

    /// Advances to the next event.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on a failed conversion.
    pub fn step(&mut self) -> MMResult<QrStep> {
        let limits = self.params.limits();
        let mut total = 0.0f64;
        // A cell without a kind is a touch reset.
        let mut cells: Vec<(QrSide, usize, Option<QrEventKind>, f64)> =
            Vec::with_capacity(MAX_LIMITS.saturating_mul(6).saturating_add(2));
        for side in [QrSide::Bid, QrSide::Ask] {
            for depth in 0..limits {
                let queue = self.queue(side, depth);
                let n = usize::try_from(queue).map_err(|_| overflow("queue"))?;
                let Some((limit, cancel, market)) = self.params.rates_at(depth, n) else {
                    return Ok(QrStep::Unsupported { side, depth, queue });
                };
                for (kind, rate) in [
                    (QrEventKind::Limit, limit),
                    (QrEventKind::Cancel, cancel),
                    (QrEventKind::Market, market),
                ] {
                    if rate > 0.0 {
                        total += rate;
                        cells.push((side, depth, Some(kind), rate));
                    }
                }
            }
            let touch = self.queue(side, 0);
            let reset = self.params.resets.as_ref().and_then(|resets| {
                usize::try_from(touch)
                    .ok()
                    .and_then(|n| resets.rates.get(n).copied())
            });
            if let Some(rate) = reset.filter(|r| *r > 0.0) {
                total += rate;
                cells.push((side, 0, None, rate));
            }
        }
        if total <= 0.0 {
            return Ok(QrStep::Stalled);
        }
        let waited = -(1.0 - self.unit()).ln() / total;
        let mut pick = self.unit() * total;
        let mut chosen = cells.last().copied();
        for cell in &cells {
            pick -= cell.3;
            if pick < 0.0 {
                chosen = Some(*cell);
                break;
            }
        }
        let (side, depth, kind, _) = chosen.or_overflow("event choice")?;
        let now = self.state.time_seconds + waited;
        if now <= self.state.time_seconds {
            return Ok(QrStep::Stalled);
        }
        self.state.time_seconds = now;
        self.state.clock_seconds = self.state.clock_seconds.max(now);
        let Some(kind) = kind else {
            return self.reset(side, now);
        };
        let before = self.queue(side, depth);
        let last = self.last_event(side, depth);
        let observation = QueueObservation {
            timestamp_ms: milliseconds(now)?,
            side,
            depth: u8::try_from(depth).map_err(|_| overflow("depth"))?,
            kind,
            queue_before: before,
            waiting_seconds: now - last,
        };
        let mid_before = self.mid_half_ticks();
        let after = match kind {
            QrEventKind::Limit => before.checked_add(1),
            QrEventKind::Cancel | QrEventKind::Market => before.checked_sub(1),
        }
        .or_overflow("queue after event")?;
        self.set_queue(side, depth, after)?;
        self.set_last_event(side, depth, now)?;
        self.state.events = self.state.events.saturating_add(1);
        let mid_after = self.mid_half_ticks();
        let reference = self.reference_move(mid_before, mid_after, observation.timestamp_ms)?;
        Ok(QrStep::Event {
            observation,
            reference,
        })
    }

    /// Moves the reference one tick up or down from outside the simulated flow (#282), as a
    /// trade's permanent impact. The first limit's queue on the side the price moves through
    /// is emptied first, then the queues shift as when `θ` moves the reference after that
    /// queue empties, with no `θ` coin and no redraw. The move is imposed at the state's
    /// clock: the state's time moves up to it, which leaves the law of the next event
    /// unchanged since waits are exponential, and every queue's clock restarts there. The
    /// shift draws the outermost queue from the simulator's own generator.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on a failed conversion or a reference price
    /// that overflows.
    pub fn impose_reference_move(&mut self, up: bool) -> MMResult<()> {
        let through = if up { QrSide::Ask } else { QrSide::Bid };
        self.set_queue(through, 0, 0)?;
        self.shift(up)?;
        self.state.reference_ticks = if up {
            self.state.reference_ticks.checked_add(1)
        } else {
            self.state.reference_ticks.checked_sub(1)
        }
        .or_overflow("reference price")?;
        let now = self.state.clock_seconds.max(self.state.time_seconds);
        self.state.time_seconds = now;
        self.state.bid_last = [now; MAX_LIMITS];
        self.state.ask_last = [now; MAX_LIMITS];
        Ok(())
    }

    /// Whether a mid move moves the reference, and its effects.
    /// Redraws the first limit's queue on `side` from its invariant law at `now`.
    fn reset(&mut self, side: QrSide, now: f64) -> MMResult<QrStep> {
        let before = self.queue(side, 0);
        let waiting_seconds = now - self.last_event(side, 0);
        let mid_before = self.mid_half_ticks();
        let law = self
            .params
            .resets
            .as_ref()
            .map(|r| r.law.clone())
            .or_overflow("reset law")?;
        let after = self.draw(&law)?;
        self.set_queue(side, 0, after)?;
        self.set_last_event(side, 0, now)?;
        let timestamp_ms = milliseconds(now)?;
        let mid_after = self.mid_half_ticks();
        let reference = self.reference_move(mid_before, mid_after, timestamp_ms)?;
        Ok(QrStep::Reset {
            reset: ResetObservation {
                timestamp_ms,
                side,
                queue_before: before,
                waiting_seconds,
                queue_after: Some(after),
            },
            queue_after: after,
            reference,
        })
    }

    fn reference_move(
        &mut self,
        before: i64,
        after: i64,
        timestamp_ms: u64,
    ) -> MMResult<Option<ReferenceObservation>> {
        let up = after > before && self.queue(QrSide::Ask, 0) == 0;
        let down = after < before && self.queue(QrSide::Bid, 0) == 0;
        if !up && !down {
            return Ok(None);
        }
        let moved = self.unit() < self.params.theta;
        let mut reinitialized = false;
        if moved {
            self.shift(up)?;
            self.state.reference_ticks = if up {
                self.state.reference_ticks.checked_add(1)
            } else {
                self.state.reference_ticks.checked_sub(1)
            }
            .or_overflow("reference price")?;
            if self.unit() < self.params.theta_reinit {
                self.redraw()?;
                reinitialized = true;
            }
            let now = self.state.time_seconds;
            self.state.bid_last = [now; MAX_LIMITS];
            self.state.ask_last = [now; MAX_LIMITS];
        }
        Ok(Some(ReferenceObservation {
            timestamp_ms,
            moved,
            reinitialized,
        }))
    }

    /// Moves the queues one limit, section 3.1.1, renormalizing by the event sizes.
    fn shift(&mut self, up: bool) -> MMResult<()> {
        let limits = self.params.limits();
        let (from, to) = if up {
            (self.state.ask, self.state.bid)
        } else {
            (self.state.bid, self.state.ask)
        };
        let mut new_from = [0u32; MAX_LIMITS];
        let mut new_to = [0u32; MAX_LIMITS];
        // The side prices move toward: each queue takes the one further out, and
        // the outermost is drawn from its law.
        for depth in 0..limits {
            let next = depth.checked_add(1).or_overflow("depth")?;
            let value = if next < limits {
                self.renormalized(*from.get(next).or_overflow("queue")?, next, depth)?
            } else {
                self.draw_invariant(depth)?
            };
            *new_from.get_mut(depth).or_overflow("queue")? = value;
        }
        // The other side: the nearest takes the emptied opposite queue, the rest
        // the one nearer, and the outermost leaves the record.
        *new_to.first_mut().or_overflow("queue")? =
            self.renormalized(*from.first().or_overflow("queue")?, 0, 0)?;
        for depth in 1..limits {
            let nearer = depth.checked_sub(1).or_overflow("depth")?;
            *new_to.get_mut(depth).or_overflow("queue")? =
                self.renormalized(*to.get(nearer).or_overflow("queue")?, nearer, depth)?;
        }
        if up {
            self.state.ask = new_from;
            self.state.bid = new_to;
        } else {
            self.state.bid = new_from;
            self.state.ask = new_to;
        }
        Ok(())
    }

    /// `q · size(from) / size(to)`, rounded up as in section 2.3.2 and capped.
    fn renormalized(&self, queue: u32, from: usize, to: usize) -> MMResult<u32> {
        let sizes = &self.params.event_sizes;
        let from_size = sizes.get(from).or_overflow("event size")?;
        let to_size = sizes.get(to).or_overflow("event size")?;
        let value = Decimal::from(queue)
            .checked_mul(*from_size)
            .and_then(|v| v.checked_div(*to_size))
            .or_overflow("renormalized queue")?
            .ceil();
        let cap = Decimal::from(u32::try_from(self.params.max_queue()).unwrap_or(u32::MAX));
        rust_decimal::prelude::ToPrimitive::to_u32(&value.min(cap))
            .or_overflow("renormalized queue")
    }

    fn redraw(&mut self) -> MMResult<()> {
        for depth in 0..self.params.limits() {
            let bid = self.draw_invariant(depth)?;
            let ask = self.draw_invariant(depth)?;
            self.set_queue(QrSide::Bid, depth, bid)?;
            self.set_queue(QrSide::Ask, depth, ask)?;
        }
        Ok(())
    }

    fn draw_invariant(&mut self, depth: usize) -> MMResult<u32> {
        let law = self
            .params
            .invariant
            .get(depth)
            .or_overflow("invariant law")?
            .clone();
        self.draw(&law)
    }

    /// A queue size from `law`.
    fn draw(&mut self, law: &[f64]) -> MMResult<u32> {
        let mut pick = self.unit();
        let mut last = 0usize;
        for (n, p) in law.iter().enumerate() {
            if *p > 0.0 {
                last = n;
                pick -= p;
                if pick < 0.0 {
                    break;
                }
            }
        }
        u32::try_from(last).map_err(|_| overflow("drawn queue"))
    }

    /// A value in `[0, 1)`, by xorshift64*.
    fn unit(&mut self) -> f64 {
        let mut x = self.state.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state.rng_state = x;
        let value = x.wrapping_mul(RNG_MULTIPLIER) >> 11;
        #[allow(clippy::cast_precision_loss)] // A 53 bit integer is exact in f64.
        let as_float = value as f64;
        as_float * UNIT_SCALE
    }

    fn queue(&self, side: QrSide, depth: usize) -> u32 {
        let queues = match side {
            QrSide::Bid => &self.state.bid,
            QrSide::Ask => &self.state.ask,
        };
        queues.get(depth).copied().unwrap_or(0)
    }

    fn set_queue(&mut self, side: QrSide, depth: usize, value: u32) -> MMResult<()> {
        let queues = match side {
            QrSide::Bid => &mut self.state.bid,
            QrSide::Ask => &mut self.state.ask,
        };
        *queues.get_mut(depth).or_overflow("queue")? = value;
        Ok(())
    }

    fn last_event(&self, side: QrSide, depth: usize) -> f64 {
        let times = match side {
            QrSide::Bid => &self.state.bid_last,
            QrSide::Ask => &self.state.ask_last,
        };
        times.get(depth).copied().unwrap_or(0.0)
    }

    fn set_last_event(&mut self, side: QrSide, depth: usize, value: f64) -> MMResult<()> {
        let times = match side {
            QrSide::Bid => &mut self.state.bid_last,
            QrSide::Ask => &mut self.state.ask_last,
        };
        *times.get_mut(depth).or_overflow("queue time")? = value;
        Ok(())
    }

    /// Limits from the reference to the best price, from 0, or `K` if empty.
    fn best_offset(&self, queues: &[u32; MAX_LIMITS]) -> u32 {
        let limits = self.params.limits();
        let found = queues
            .iter()
            .take(limits)
            .position(|q| *q > 0)
            .unwrap_or(limits);
        u32::try_from(found).unwrap_or(u32::MAX)
    }

    /// The mid, relative to the reference, in half ticks.
    fn mid_half_ticks(&self) -> i64 {
        i64::from(self.best_offset(&self.state.ask))
            .saturating_sub(i64::from(self.best_offset(&self.state.bid)))
    }
}

fn overflow(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} overflows"))
}

fn milliseconds(seconds: f64) -> MMResult<u64> {
    let scaled = (seconds * 1_000.0).floor();
    rust_decimal::prelude::ToPrimitive::to_u64(&scaled).or_overflow("event time")
}

/// Estimation settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[non_exhaustive]
pub struct QrEstimationConfig {
    /// Price grid. Positive.
    pub tick_size: Decimal,
    /// Limits per side, from 1 to [`MAX_LIMITS`].
    pub limits: u8,
    /// Queue cap. Positive and at most [`MAX_QUEUE_BOUND`].
    pub max_queue: u32,
    /// Fewest events for a queue size to be supported. Positive.
    pub min_events: u64,
    /// Fewest eligible mid moves for `θ`. Positive.
    pub min_reference_moves: u64,
    /// Policy at an unsupported size.
    pub policy: UnsupportedPolicy,
    /// What to do with an event whose waiting time is zero; [`ZeroWaitPolicy::Drop`] unless
    /// [`QrEstimationConfig::with_zero_waits`] says otherwise.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zero_waits: ZeroWaitPolicy,
}

impl QrEstimationConfig {
    /// Creates estimation settings.
    #[must_use]
    pub fn new(
        tick_size: Decimal,
        limits: u8,
        max_queue: u32,
        min_events: u64,
        min_reference_moves: u64,
        policy: UnsupportedPolicy,
    ) -> Self {
        Self {
            tick_size,
            limits,
            max_queue,
            min_events,
            min_reference_moves,
            policy,
            zero_waits: ZeroWaitPolicy::Drop,
        }
    }

    /// The same settings with another policy for events whose waiting time is zero.
    #[must_use]
    pub fn with_zero_waits(mut self, zero_waits: ZeroWaitPolicy) -> Self {
        self.zero_waits = zero_waits;
        self
    }
}

/// Why no parameters could be estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum QrFailure {
    /// A limit has no supported queue size.
    NoSupportedSize {
        /// Limit from 0.
        depth: u8,
    },
    /// A limit has too few events with a size for the mean of [`estimate_event_sizes`].
    TooFewSizedEvents {
        /// Limit from 0.
        depth: u8,
        /// Events available.
        have: u64,
        /// Events required.
        need: u64,
    },
    /// Enough eligible mid moves, but the reference never followed one.
    ReferenceNeverMoved,
    /// Too few eligible mid moves for `θ`.
    TooFewReferenceMoves {
        /// Moves available.
        have: u64,
        /// Moves required.
        need: u64,
    },
    /// The estimates do not form valid parameters: an invariant law without mass,
    /// or a rate above [`MAX_RATE`].
    InvalidEstimate,
}

/// The result of an estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum QrFit {
    /// Parameters.
    Calibrated(QrParameters),
    /// None.
    Insufficient(QrFailure),
}

/// Parameters together with what they were estimated from.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct QrRecord {
    /// Training interval.
    pub training: TimeWindow,
    /// Caller's name for the dataset.
    pub dataset_id: String,
    /// FNV-1a 128 of the observations inside the interval.
    pub dataset_hash: u128,
    /// Queue events used.
    pub events: u64,
    /// Queue events left out: invalid, or a size above the cap.
    pub excluded_events: u64,
    /// The parameters, or why there are none.
    pub fit: QrFit,
}

impl QrRecord {
    /// Returns the parameters, if any.
    #[must_use]
    pub fn parameters(&self) -> Option<&QrParameters> {
        match &self.fit {
            QrFit::Calibrated(params) => Some(params),
            QrFit::Insufficient(_) => None,
        }
    }
}

/// Estimates the event size of each limit as the mean size of its events inside `window`
/// and nothing else, the average event size in which the model counts its queues.
///
/// [`QueueObservation::queue_before`] is in these units, so build those observations from
/// this result on the same window.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for limits out of range or a `min_events` of
/// zero, [`MMError::InvalidMarketState`] for events out of order, and
/// [`MMError::NumericalError`] if a sum or a mean overflows. A limit with too few events
/// gives [`QrFailure::TooFewSizedEvents`].
pub fn estimate_event_sizes(
    events: &[SizedEvent],
    window: &TimeWindow,
    limits: u8,
    min_events: u64,
) -> MMResult<EventSizeRecord> {
    if limits == 0 || usize::from(limits) > MAX_LIMITS || min_events == 0 {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid event size settings: {limits} limits, {min_events} events"
        )));
    }
    if !events.is_sorted_by_key(|e| e.timestamp_ms) {
        return Err(MMError::InvalidMarketState(
            "events are not in chronological order".to_string(),
        ));
    }
    let mut totals = vec![Decimal::ZERO; usize::from(limits)];
    let mut counts = vec![0u64; usize::from(limits)];
    let mut excluded = 0u64;
    let mut hash = FNV_OFFSET;
    for event in inside(events, window, |e| e.timestamp_ms) {
        hash = fnv(hash, &event.timestamp_ms.to_le_bytes());
        hash = fnv(hash, &[event.depth]);
        hash = fnv(hash, &event.size.normalize().serialize());
        let depth = usize::from(event.depth);
        let (Some(total), Some(count), true) = (
            totals.get_mut(depth),
            counts.get_mut(depth),
            event.size > Decimal::ZERO,
        ) else {
            excluded = excluded.saturating_add(1);
            continue;
        };
        *total = total.checked_add(event.size).or_overflow("event sizes")?;
        *count = count.saturating_add(1);
    }
    let mut sizes = Vec::with_capacity(totals.len());
    let mut failure = None;
    for (depth, (total, count)) in totals.iter().zip(&counts).enumerate() {
        if *count < min_events {
            failure = Some(QrFailure::TooFewSizedEvents {
                depth: u8::try_from(depth).unwrap_or(u8::MAX),
                have: *count,
                need: min_events,
            });
            break;
        }
        sizes.push(
            total
                .checked_div(Decimal::from(*count))
                .or_overflow("mean event size")?,
        );
    }
    Ok(EventSizeRecord {
        training: *window,
        dataset_hash: hash,
        events: counts,
        excluded_events: excluded,
        sizes: failure.map_or(Ok(sizes), Err),
    })
}

/// Estimates on the observations inside `window` and nothing else.
///
/// An event is left out and counted if its waiting time is not finite and
/// positive (or, under [`ZeroWaitPolicy::Count`], not finite and not negative), its
/// depth is outside the limits, it departs from an empty queue, or
/// its size is above the cap. An arrival at the cap is left out too.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for invalid settings, and
/// [`MMError::InvalidMarketState`] for observations out of order. Event sizes that
/// are not positive give [`QrFailure::InvalidEstimate`].
pub fn estimate_parameters(
    events: &[QueueObservation],
    references: &[ReferenceObservation],
    event_sizes: &[Decimal],
    window: &TimeWindow,
    config: &QrEstimationConfig,
    dataset_id: &str,
) -> MMResult<QrRecord> {
    estimate(
        events,
        None,
        references,
        event_sizes,
        window,
        config,
        dataset_id,
    )
}

/// Estimates as [`estimate_parameters`], and touch resets (#290) from `resets`: per
/// first-limit queue size `n > 0`, the resets at `n` over the time the first limit's queues
/// spent at `n`, and the law of the new touch's size over the counted resets that report it,
/// a size past the cap counted at the cap. A reset's wait counts as time at its size for
/// every rate, as an event's does, and is counted under the same rules; a reset at size 0 or
/// beyond the cap is left out and counted as excluded. Both sides pool. A size with no time
/// gets reset rate 0. With no new size reported, the law is the first limit's invariant law.
/// The fit carries both through [`QrParameters::with_resets`].
///
/// # Errors
///
/// As [`estimate_parameters`], with `resets` out of order a
/// [`MMError::InvalidMarketState`].
pub fn estimate_parameters_with_resets(
    events: &[QueueObservation],
    resets: &[ResetObservation],
    references: &[ReferenceObservation],
    event_sizes: &[Decimal],
    window: &TimeWindow,
    config: &QrEstimationConfig,
    dataset_id: &str,
) -> MMResult<QrRecord> {
    estimate(
        events,
        Some(resets),
        references,
        event_sizes,
        window,
        config,
        dataset_id,
    )
}

fn estimate(
    events: &[QueueObservation],
    resets: Option<&[ResetObservation]>,
    references: &[ReferenceObservation],
    event_sizes: &[Decimal],
    window: &TimeWindow,
    config: &QrEstimationConfig,
    dataset_id: &str,
) -> MMResult<QrRecord> {
    if config.limits == 0
        || usize::from(config.limits) > MAX_LIMITS
        || config.max_queue == 0
        || config.max_queue > MAX_QUEUE_BOUND
        || config.min_events == 0
        || config.min_reference_moves == 0
        || event_sizes.len() != usize::from(config.limits)
    {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid estimation settings: {config:?}"
        )));
    }
    if !events.is_sorted_by_key(|e| e.timestamp_ms)
        || !references.is_sorted_by_key(|r| r.timestamp_ms)
        || !resets
            .unwrap_or_default()
            .is_sorted_by_key(|r| r.timestamp_ms)
    {
        return Err(MMError::InvalidMarketState(
            "observations are not in chronological order".to_string(),
        ));
    }
    let limits = usize::from(config.limits);
    let rows = usize::try_from(config.max_queue)
        .ok()
        .and_then(|m| m.checked_add(1))
        .or_overflow("rows")?;
    let mut counts = vec![vec![(0u64, 0u64, 0u64); rows]; limits];
    let mut reset_counts = vec![0u64; rows];
    let mut after_counts = vec![0u64; rows];
    let mut waiting = vec![vec![0.0f64; rows]; limits];
    let mut used = 0u64;
    let mut excluded = 0u64;
    let mut hash = FNV_OFFSET;
    for event in inside(events, window, |e| e.timestamp_ms) {
        hash = hash_event(hash, event);
        let depth = usize::from(event.depth);
        let n = usize::try_from(event.queue_before).unwrap_or(usize::MAX);
        if !counted(event, config, limits, rows) {
            excluded = excluded.saturating_add(1);
            continue;
        }
        used = used.saturating_add(1);
        let row = counts
            .get_mut(depth)
            .and_then(|table| table.get_mut(n))
            .or_overflow("count row")?;
        match event.kind {
            QrEventKind::Limit => row.0 = row.0.saturating_add(1),
            QrEventKind::Cancel => row.1 = row.1.saturating_add(1),
            QrEventKind::Market => row.2 = row.2.saturating_add(1),
        }
        let total = waiting
            .get_mut(depth)
            .and_then(|table| table.get_mut(n))
            .or_overflow("waiting row")?;
        *total += event.waiting_seconds;
    }
    for reset in inside(resets.unwrap_or_default(), window, |r| r.timestamp_ms) {
        hash = fnv(hash, &reset.timestamp_ms.to_le_bytes());
        hash = fnv(hash, &[u8::from(reset.side == QrSide::Bid)]);
        hash = fnv(hash, &reset.queue_before.to_le_bytes());
        hash = fnv(hash, &reset.waiting_seconds.to_bits().to_le_bytes());
        hash = fnv(
            hash,
            &reset.queue_after.map_or(u64::MAX, u64::from).to_le_bytes(),
        );
        let n = usize::try_from(reset.queue_before).unwrap_or(usize::MAX);
        let waited = match config.zero_waits {
            ZeroWaitPolicy::Drop => reset.waiting_seconds > 0.0,
            ZeroWaitPolicy::Count => reset.waiting_seconds >= 0.0,
        };
        let (Some(count), Some(total), true) = (
            reset_counts.get_mut(n).filter(|_| n > 0),
            waiting.first_mut().and_then(|times| times.get_mut(n)),
            reset.waiting_seconds.is_finite() && waited,
        ) else {
            excluded = excluded.saturating_add(1);
            continue;
        };
        used = used.saturating_add(1);
        *count = count.saturating_add(1);
        *total += reset.waiting_seconds;
        if let Some(after) = reset.queue_after {
            let at = usize::try_from(after)
                .unwrap_or(usize::MAX)
                .min(rows.saturating_sub(1));
            if let Some(cell) = after_counts.get_mut(at) {
                *cell = cell.saturating_add(1);
            }
        }
    }
    let mut eligible = 0u64;
    let mut moved = 0u64;
    let mut reinitialized = 0u64;
    for reference in inside(references, window, |r| r.timestamp_ms) {
        hash = fnv(hash, &reference.timestamp_ms.to_le_bytes());
        hash = fnv(
            hash,
            &[u8::from(reference.moved), u8::from(reference.reinitialized)],
        );
        eligible = eligible.saturating_add(1);
        moved = moved.saturating_add(u64::from(reference.moved));
        reinitialized = reinitialized.saturating_add(u64::from(reference.reinitialized));
    }

    let finish = |fit: QrFit| QrRecord {
        training: *window,
        dataset_id: dataset_id.to_string(),
        dataset_hash: hash,
        events: used,
        excluded_events: excluded,
        fit,
    };
    if eligible >= config.min_reference_moves && moved == 0 {
        return Ok(finish(QrFit::Insufficient(QrFailure::ReferenceNeverMoved)));
    }
    if eligible < config.min_reference_moves {
        return Ok(finish(QrFit::Insufficient(
            QrFailure::TooFewReferenceMoves {
                have: eligible,
                need: config.min_reference_moves,
            },
        )));
    }
    let mut rates = Vec::with_capacity(limits);
    let mut supported = Vec::with_capacity(limits);
    for (depth, (table, times)) in counts.iter().zip(&waiting).enumerate() {
        let mut rate_rows = Vec::with_capacity(rows);
        let mut flags = Vec::with_capacity(rows);
        for ((limit, cancel, market), time) in table.iter().zip(times) {
            let count = limit.saturating_add(*cancel).saturating_add(*market);
            // A size whose events all waited zero has counts and no time, so no rate: it is
            // unsupported, not a size whose rates are zero. Only counting zero waits makes one.
            flags.push(count >= config.min_events && *time > 0.0);
            let rate = |c: u64| as_float(c) / *time;
            rate_rows.push(if *time > 0.0 {
                (rate(*limit), rate(*cancel), rate(*market))
            } else {
                (0.0, 0.0, 0.0)
            });
        }
        if !flags.contains(&true) {
            let depth = u8::try_from(depth).unwrap_or(u8::MAX);
            return Ok(finish(QrFit::Insufficient(QrFailure::NoSupportedSize {
                depth,
            })));
        }
        rates.push(rate_rows);
        supported.push(flags);
    }
    let theta = as_float(moved) / as_float(eligible);
    let theta_reinit = as_float(reinitialized) / as_float(moved);
    let reset_rates = resets.map(|_| {
        reset_counts
            .iter()
            .zip(waiting.first().map(Vec::as_slice).unwrap_or_default())
            .map(|(c, t)| if *t > 0.0 { as_float(*c) / *t } else { 0.0 })
            .collect::<Vec<_>>()
    });
    Ok(finish(
        match QrParameters::new(
            config.tick_size,
            theta,
            theta_reinit,
            event_sizes.to_vec(),
            rates,
            supported,
            config.policy,
        )
        .and_then(|params| match reset_rates {
            Some(rates) => {
                let known: u64 = after_counts.iter().sum();
                let law = if known > 0 {
                    after_counts
                        .iter()
                        .map(|c| as_float(*c) / as_float(known))
                        .collect()
                } else {
                    params.invariant(0).unwrap_or_default().to_vec()
                };
                params.with_resets(rates, law)
            }
            None => Ok(params),
        }) {
            Ok(params) => QrFit::Calibrated(params),
            Err(MMError::NumericalError(_) | MMError::InvalidConfiguration(_)) => {
                QrFit::Insufficient(QrFailure::InvalidEstimate)
            }
            Err(error) => return Err(error),
        },
    ))
}

#[allow(clippy::cast_precision_loss)] // Report values: counts far below 2^53.
fn as_float(value: u64) -> f64 {
    value as f64
}

/// Whether the estimator counts `event`, under `config`'s zero waits, with `limits` limits
/// and `rows` queue sizes.
fn counted(
    event: &QueueObservation,
    config: &QrEstimationConfig,
    limits: usize,
    rows: usize,
) -> bool {
    let depth = usize::from(event.depth);
    let n = usize::try_from(event.queue_before).unwrap_or(usize::MAX);
    let departs = matches!(event.kind, QrEventKind::Cancel | QrEventKind::Market);
    let arrives = matches!(event.kind, QrEventKind::Limit);
    let waited = match config.zero_waits {
        ZeroWaitPolicy::Drop => event.waiting_seconds > 0.0,
        ZeroWaitPolicy::Count => event.waiting_seconds >= 0.0,
    };
    event.waiting_seconds.is_finite()
        && waited
        && depth < limits
        && n < rows
        && !(departs && n == 0)
        && !(arrives && n.saturating_add(1) == rows)
}

fn inside<'a, T>(items: &'a [T], window: &TimeWindow, time: impl Fn(&T) -> u64) -> &'a [T] {
    let from = items.partition_point(|item| time(item) < window.start_ms);
    let to = items.partition_point(|item| time(item) < window.end_ms);
    items.get(from..to).unwrap_or_default()
}

fn hash_event(hash: u128, event: &QueueObservation) -> u128 {
    let hash = fnv(hash, &event.timestamp_ms.to_le_bytes());
    let side: u8 = match event.side {
        QrSide::Bid => 0,
        QrSide::Ask => 1,
    };
    let kind: u8 = match event.kind {
        QrEventKind::Limit => 0,
        QrEventKind::Cancel => 1,
        QrEventKind::Market => 2,
    };
    let hash = fnv(hash, &[side, event.depth, kind]);
    let hash = fnv(hash, &event.queue_before.to_le_bytes());
    fnv(hash, &event.waiting_seconds.to_bits().to_le_bytes())
}

/// A book observed at a time, for the distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BookSample {
    /// Time of the sample, in milliseconds.
    pub timestamp_ms: u64,
    /// Bid queues.
    pub bid: [u32; MAX_LIMITS],
    /// Ask queues.
    pub ask: [u32; MAX_LIMITS],
    /// Spread in ticks.
    pub spread_ticks: u32,
}

/// A trajectory on a time grid, with its events.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Trajectory {
    /// The book at each grid time.
    pub samples: Vec<BookSample>,
    /// Queue events.
    pub events: Vec<QueueObservation>,
    /// Reference price outcomes.
    pub references: Vec<ReferenceObservation>,
    /// Touch resets, under [`QrParameters::with_resets`].
    pub resets: Vec<ResetObservation>,
    /// Why it ended early, if it did.
    pub stopped: Option<QrStep>,
}

impl QrSimulator {
    /// Runs until `horizon_seconds`, sampling every `sample_seconds`, keeping at
    /// most [`MAX_RUN_EVENTS`] events. A sample shows the book after every event
    /// before its time. The event that would pass the horizon is not applied: the
    /// state keeps the generator from before it, with `clock_seconds` at the horizon,
    /// so a later `run` or step produces it. Consecutive runs equal one long run.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a grid that is not finite and
    /// positive, or more than a million samples, and
    /// [`MMError::NumericalError`] on a failed conversion.
    pub fn run(&mut self, horizon_seconds: f64, sample_seconds: f64) -> MMResult<Trajectory> {
        self.run_limited(horizon_seconds, sample_seconds, MAX_RUN_EVENTS)
    }

    /// Runs like [`QrSimulator::run`], ending with [`QrStep::EventLimit`] at
    /// `max_events` events.
    ///
    /// # Errors
    ///
    /// As [`QrSimulator::run`].
    pub fn run_limited(
        &mut self,
        horizon_seconds: f64,
        sample_seconds: f64,
        max_events: usize,
    ) -> MMResult<Trajectory> {
        let valid = |v: f64| v.is_finite() && v > 0.0;
        if !valid(horizon_seconds)
            || !valid(sample_seconds)
            || horizon_seconds / sample_seconds > 1_000_000.0
        {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid grid: {horizon_seconds} by {sample_seconds}"
            )));
        }
        let mut out = Trajectory::default();
        let mut next_sample = self.state.clock_seconds + sample_seconds;
        let end = self.state.clock_seconds + horizon_seconds;
        loop {
            if out.events.len() >= max_events {
                out.stopped = Some(QrStep::EventLimit);
                break;
            }
            let before = self.state;
            let step = self.step()?;
            match step {
                QrStep::Event {
                    observation,
                    reference,
                } => {
                    while next_sample <= self.state.time_seconds && next_sample <= end {
                        out.samples.push(BookSample {
                            timestamp_ms: milliseconds(next_sample)?,
                            bid: before.bid,
                            ask: before.ask,
                            spread_ticks: spread_of(&before, self.params.limits()),
                        });
                        next_sample += sample_seconds;
                    }
                    if self.state.time_seconds > end {
                        self.state = before;
                        self.state.clock_seconds = end;
                        break;
                    }
                    out.events.push(observation);
                    if let Some(reference) = reference {
                        out.references.push(reference);
                    }
                }
                QrStep::Reset {
                    reset, reference, ..
                } => {
                    while next_sample <= self.state.time_seconds && next_sample <= end {
                        out.samples.push(BookSample {
                            timestamp_ms: milliseconds(next_sample)?,
                            bid: before.bid,
                            ask: before.ask,
                            spread_ticks: spread_of(&before, self.params.limits()),
                        });
                        next_sample += sample_seconds;
                    }
                    if self.state.time_seconds > end {
                        self.state = before;
                        self.state.clock_seconds = end;
                        break;
                    }
                    out.resets.push(reset);
                    if let Some(reference) = reference {
                        out.references.push(reference);
                    }
                }
                QrStep::Unsupported { .. } | QrStep::Stalled | QrStep::EventLimit => {
                    out.stopped = Some(step);
                    break;
                }
            }
        }
        Ok(out)
    }
}

fn spread_of(state: &QrState, limits: usize) -> u32 {
    let offset = |queues: &[u32; MAX_LIMITS]| {
        queues
            .iter()
            .take(limits)
            .position(|q| *q > 0)
            .unwrap_or(limits)
    };
    let total = offset(&state.bid)
        .saturating_add(offset(&state.ask))
        .saturating_add(1);
    u32::try_from(total).unwrap_or(u32::MAX)
}

/// Discrepancies of a model trajectory from observed data, on one window.
///
/// A short or stopped model is not refused, so its counts and reason are here, to
/// be read with the distances.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct DiscrepancyReport {
    /// Total variation of the queue size law, per limit, both sides pooled.
    pub queue_size: Vec<f64>,
    /// Total variation of the spread law.
    pub spread: f64,
    /// Relative error of the mean waiting time, per limit.
    pub waiting_time: Vec<f64>,
    /// Kolmogorov-Smirnov distance of the waiting times, per limit.
    pub waiting_law: Vec<f64>,
    /// Relative error of the events per second, for limit, cancel and market.
    pub flow: [f64; 3],
    /// Total variation of the events per interval, for limit, cancel and market.
    pub flow_law: [f64; 3],
    /// Share of observed events at a size `params` supports.
    pub coverage: f64,
    /// Samples of the data.
    pub data_samples: u64,
    /// Events of the data.
    pub data_events: u64,
    /// Samples of the model.
    pub model_samples: u64,
    /// Events of the model.
    pub model_events: u64,
    /// Why the model ended early, if it did.
    pub model_stopped: Option<QrStep>,
    /// Data events left out of the waiting times, for a time not positive and finite.
    pub data_invalid_waiting: u64,
    /// Model events left out of the waiting times, for a time not positive and finite.
    pub model_invalid_waiting: u64,
}

/// Compares `model` with `data`, using `params` only for the coverage, and counting
/// events per `flow_interval_ms` for the flow law.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] when either has no samples or no events,
/// or spans no time, and [`MMError::InvalidConfiguration`] for an interval of zero
/// or over a million intervals.
pub fn compare_trajectories(
    data: &Trajectory,
    model: &Trajectory,
    params: &QrParameters,
    flow_interval_ms: u64,
) -> MMResult<DiscrepancyReport> {
    if data.samples.is_empty()
        || model.samples.is_empty()
        || data.events.is_empty()
        || model.events.is_empty()
    {
        return Err(MMError::InvalidMarketState(
            "a trajectory has no samples or no events".to_string(),
        ));
    }
    let limits = params.limits();
    let mut queue_size = Vec::with_capacity(limits);
    let mut waiting_time = Vec::with_capacity(limits);
    let mut waiting_law = Vec::with_capacity(limits);
    for depth in 0..limits {
        let sizes = |t: &Trajectory| -> Vec<u32> {
            t.samples
                .iter()
                .flat_map(|s| {
                    [
                        s.bid.get(depth).copied().unwrap_or(0),
                        s.ask.get(depth).copied().unwrap_or(0),
                    ]
                })
                .collect()
        };
        queue_size.push(total_variation(&sizes(data), &sizes(model)));
        let waits = |t: &Trajectory| -> Vec<f64> {
            t.events
                .iter()
                .filter(|e| usize::from(e.depth) == depth && valid_waiting(e))
                .map(|e| e.waiting_seconds)
                .collect()
        };
        let (first, second) = (waits(data), waits(model));
        let mean = |values: &[f64]| -> f64 {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / as_float(values.len() as u64)
            }
        };
        waiting_time.push(relative_error(mean(&first), mean(&second)));
        waiting_law.push(kolmogorov_smirnov(&first, &second));
    }
    let invalid = |t: &Trajectory| t.events.iter().filter(|e| !valid_waiting(e)).count() as u64;
    let spreads =
        |t: &Trajectory| -> Vec<u32> { t.samples.iter().map(|s| s.spread_ticks).collect() };
    let kinds = [QrEventKind::Limit, QrEventKind::Cancel, QrEventKind::Market];
    let mut flow = [0.0; 3];
    let mut flow_law = [0.0; 3];
    for ((kind, mean_error), law_distance) in kinds.iter().zip(&mut flow).zip(&mut flow_law) {
        let (data_counts, data_seconds) = interval_counts(data, *kind, flow_interval_ms)?;
        let (model_counts, model_seconds) = interval_counts(model, *kind, flow_interval_ms)?;
        let per_second = |counts: &[u32], seconds: f64| -> f64 {
            counts.iter().map(|c| f64::from(*c)).sum::<f64>() / seconds
        };
        *mean_error = relative_error(
            per_second(&data_counts, data_seconds),
            per_second(&model_counts, model_seconds),
        );
        *law_distance = total_variation(&data_counts, &model_counts);
    }
    let covered = data
        .events
        .iter()
        .filter(|e| {
            params.is_supported(
                usize::from(e.depth),
                usize::try_from(e.queue_before).unwrap_or(usize::MAX),
            )
        })
        .count() as u64;
    Ok(DiscrepancyReport {
        queue_size,
        spread: total_variation(&spreads(data), &spreads(model)),
        waiting_time,
        waiting_law,
        flow,
        flow_law,
        coverage: as_float(covered) / as_float(data.events.len() as u64),
        data_samples: data.samples.len() as u64,
        data_events: data.events.len() as u64,
        model_samples: model.samples.len() as u64,
        model_events: model.events.len() as u64,
        model_stopped: model.stopped,
        data_invalid_waiting: invalid(data),
        model_invalid_waiting: invalid(model),
    })
}

fn valid_waiting(event: &QueueObservation) -> bool {
    event.waiting_seconds.is_finite() && event.waiting_seconds > 0.0
}

/// Events of a kind per whole interval from the first time seen, and the seconds
/// those intervals cover. A partial last interval is dropped, or it would read as
/// a quiet one.
fn interval_counts(
    trajectory: &Trajectory,
    kind: QrEventKind,
    interval_ms: u64,
) -> MMResult<(Vec<u32>, f64)> {
    let times = || {
        trajectory
            .samples
            .iter()
            .map(|s| s.timestamp_ms)
            .chain(trajectory.events.iter().map(|e| e.timestamp_ms))
    };
    let start = times().min().or_overflow("trajectory start")?;
    let end = times().max().or_overflow("trajectory end")?;
    let span = end.saturating_sub(start);
    if interval_ms == 0 {
        return Err(MMError::InvalidConfiguration(
            "the flow interval must be positive".to_string(),
        ));
    }
    let bins = span.checked_div(interval_ms).or_overflow("flow bins")?;
    if bins > 1_000_000 {
        return Err(MMError::InvalidConfiguration(format!(
            "{bins} flow intervals"
        )));
    }
    if bins == 0 {
        return Err(MMError::InvalidMarketState(format!(
            "a trajectory spans {span} ms, less than the interval of {interval_ms} ms"
        )));
    }
    let mut counts = vec![0u32; usize::try_from(bins).map_err(|_| overflow("flow bins"))?];
    for event in trajectory.events.iter().filter(|e| e.kind == kind) {
        let bin = event
            .timestamp_ms
            .saturating_sub(start)
            .checked_div(interval_ms)
            .and_then(|b| usize::try_from(b).ok());
        if let Some(count) = bin.and_then(|b| counts.get_mut(b)) {
            *count = count.saturating_add(1);
        }
    }
    let seconds = as_float(bins) * as_float(interval_ms) / 1_000.0;
    Ok((counts, seconds))
}

/// The largest gap between two empirical distribution functions; 1 if only one
/// side has values, and 0 if neither has.
fn kolmogorov_smirnov(first: &[f64], second: &[f64]) -> f64 {
    if first.is_empty() || second.is_empty() {
        return if first.is_empty() && second.is_empty() {
            0.0
        } else {
            1.0
        };
    }
    let mut a = first.to_vec();
    let mut b = second.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (na, nb) = (as_float(a.len() as u64), as_float(b.len() as u64));
    let (mut i, mut j) = (0usize, 0usize);
    let mut largest = 0.0f64;
    while let (Some(x), Some(y)) = (a.get(i), b.get(j)) {
        let value = x.min(*y);
        while a.get(i).is_some_and(|v| *v <= value) {
            i = i.saturating_add(1);
        }
        while b.get(j).is_some_and(|v| *v <= value) {
            j = j.saturating_add(1);
        }
        largest = largest.max((as_float(i as u64) / na - as_float(j as u64) / nb).abs());
    }
    largest.max((as_float(i as u64) / na - as_float(j as u64) / nb).abs())
}

/// `|a - b| / a`, or `|b|` where `a` is zero.
fn relative_error(a: f64, b: f64) -> f64 {
    if a > 0.0 { (a - b).abs() / a } else { b.abs() }
}

/// Half the sum of the absolute differences of two empirical laws.
fn total_variation(a: &[u32], b: &[u32]) -> f64 {
    let mut first = a.to_vec();
    let mut second = b.to_vec();
    first.sort_unstable();
    second.sort_unstable();
    let (na, nb) = (as_float(first.len() as u64), as_float(second.len() as u64));
    let (mut i, mut j) = (0usize, 0usize);
    let mut sum = 0.0f64;
    loop {
        let value = match (first.get(i), second.get(j)) {
            (Some(x), Some(y)) => *x.min(y),
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => break,
        };
        let (mut ca, mut cb) = (0u64, 0u64);
        while first.get(i) == Some(&value) {
            ca = ca.saturating_add(1);
            i = i.saturating_add(1);
        }
        while second.get(j) == Some(&value) {
            cb = cb.saturating_add(1);
            j = j.saturating_add(1);
        }
        sum += (as_float(ca) / na - as_float(cb) / nb).abs();
    }
    sum / 2.0
}
