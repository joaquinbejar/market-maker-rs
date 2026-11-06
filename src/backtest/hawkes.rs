//! A queue-reactive Hawkes model of one queue, and its comparison with simpler ones.
//!
//! After Wu, Rambaldi, Muzy and Bacry, *Queue-reactive Hawkes models for the order
//! flow* (arXiv:1901.08938v1), section 2, the QRH-I model. For limit, cancel and
//! market orders `l`, on one queue of size `q`, with a fixed reference price:
//!
//! ```text
//! λ^l(t) = μ^l(q(t-)) + Σ_m Σ_u α_u^{lm} β_u ∫ exp(-β_u (t - s)) dN^m(s)
//! ```
//!
//! with `λ^C = λ^M = 0` at an empty queue, as in the paper. As in the paper too, the
//! model holds while the reference price is constant, and each such period is an
//! independent [`HawkesSegment`], so memory restarts with each. This model does not
//! move the reference price, and is not the QRH-II model of section 3.
//!
//! With `α = 0` it is the queue-reactive model on one queue, and with constant `μ`
//! a Hawkes or Poisson one, so one type carries all the comparisons.
//!
//! # Units
//!
//! Time is in seconds, `μ` in events per second, `β_u` in 1 / seconds, and `α` is
//! dimensionless: `Σ_u α_u^{lm}` is the mean number of `l` events one `m` event
//! causes. Stability requires the spectral radius of that matrix, the branching
//! ratio, to be below 1; [`HawkesParameters::new`] refuses anything else, and
//! negative `α`, so inhibition is out of scope. This is stronger than the paper's
//! need, since the masks at the queue bounds only remove intensity.
//!
//! # Departures from the paper
//!
//! - Estimation is expectation maximization with `β` fixed, which keeps `α` and `μ`
//!   non-negative and raises the likelihood at each step; the paper optimizes the
//!   same convex likelihood numerically.
//! - The queue is capped at `max_queue`, with `λ^L = 0` there.
//! - A queue size with fewer than `min_events` training events takes each baseline
//!   from the nearest supported size where that type is allowed, and the record counts
//!   them.
//! - A fill measure, [`fill_rate`], is added. It is a coarse, deterministic
//!   measure from aggregated queues, not individual queue priority.
//!
//! # Evidence
//!
//! Likelihood, residuals and flow say how well flow is described. Fills say
//! something different, and no result here says a strategy earns more. Choose `β`
//! on a validation set with [`select_decay`], and keep the holdout out of it.

use crate::backtest::queue_reactive::{QrEventKind, RateTable};
use crate::strategy::walk_forward::{FNV_OFFSET, fnv};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Most exponential kernels.
pub const MAX_KERNELS: usize = 4;
/// Largest queue accepted.
pub const MAX_QUEUE: u32 = 4_096;
/// Largest baseline rate, `β` or `1 / β`.
pub const MAX_RATE: f64 = 1e9;
/// Most candidates per event of the thinning.
pub const MAX_THINNING: u32 = 1_000_000;
/// Most events one run keeps.
pub const MAX_RUN_EVENTS: usize = 50_000_000;
/// The most intervals or probes one segment is measured over.
pub const MAX_INTERVALS: f64 = 100_000_000.0;
/// Most buckets in a [`SeasonalProfile`].
pub const MAX_SEASONAL_BUCKETS: usize = 1_000_000;
const EVENT_TYPES: usize = 3;
const KINDS: [QrEventKind; EVENT_TYPES] =
    [QrEventKind::Limit, QrEventKind::Cancel, QrEventKind::Market];
const RNG_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;
const ZERO_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Scale from a 53 bit integer to `[0, 1)`.
const UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
/// Intensity used where the model says zero and an event happened, so that the
/// likelihood stays finite and such an event costs a large amount.
const INTENSITY_FLOOR: f64 = 1e-12;
const BISECTION_STEPS: usize = 2_000;

fn index_of(kind: QrEventKind) -> Option<usize> {
    KINDS.iter().position(|k| *k == kind)
}

fn invalid<T>(what: String) -> MMResult<T> {
    Err(MMError::InvalidConfiguration(what))
}

/// A matrix of branching weights, `[l][m]`.
pub type BranchingMatrix = [[f64; EVENT_TYPES]; EVENT_TYPES];

/// The model's parameters, validated.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(into = "RawParameters", try_from = "RawParameters")
)]
pub struct HawkesParameters {
    rates: RateTable,
    decays: Vec<f64>,
    weights: Vec<BranchingMatrix>,
}

/// Exported form of [`HawkesParameters`], with exact floats.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawParameters {
    rates: Vec<[String; 3]>,
    decays: Vec<String>,
    weights: Vec<Vec<[String; 3]>>,
}

#[cfg(feature = "serde")]
fn float_text(value: f64) -> String {
    format!("{value:?}")
}

#[cfg(feature = "serde")]
fn text_float(text: &str) -> MMResult<f64> {
    text.parse::<f64>()
        .map_err(|error| MMError::InvalidConfiguration(format!("float {text}: {error}")))
}

#[cfg(feature = "serde")]
impl From<HawkesParameters> for RawParameters {
    fn from(params: HawkesParameters) -> Self {
        Self {
            rates: params
                .rates
                .iter()
                .map(|(l, c, m)| [float_text(*l), float_text(*c), float_text(*m)])
                .collect(),
            decays: params.decays.iter().map(|d| float_text(*d)).collect(),
            weights: params
                .weights
                .iter()
                .map(|matrix| matrix.iter().map(|row| row.map(float_text)).collect())
                .collect(),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<RawParameters> for HawkesParameters {
    type Error = MMError;

    fn try_from(raw: RawParameters) -> MMResult<Self> {
        let rates = raw
            .rates
            .iter()
            .map(|[l, c, m]| Ok((text_float(l)?, text_float(c)?, text_float(m)?)))
            .collect::<MMResult<RateTable>>()?;
        let decays = raw
            .decays
            .iter()
            .map(|d| text_float(d))
            .collect::<MMResult<Vec<_>>>()?;
        let mut weights = Vec::with_capacity(raw.weights.len());
        for matrix in &raw.weights {
            let mut out = [[0.0; EVENT_TYPES]; EVENT_TYPES];
            if matrix.len() != EVENT_TYPES {
                return invalid(format!("a weight matrix has {} rows", matrix.len()));
            }
            for (row, texts) in out.iter_mut().zip(matrix) {
                for (cell, text) in row.iter_mut().zip(texts) {
                    *cell = text_float(text)?;
                }
            }
            weights.push(out);
        }
        Self::new(rates, decays, weights)
    }
}

/// Whether the spectral radius of a matrix with non-negative entries is below `t`: for
/// such a matrix, exactly when every leading principal minor of `t I - A` is positive.
fn radius_is_below(matrix: &BranchingMatrix, t: f64) -> bool {
    let [[a, b, c], [d, e, f], [g, h, i]] = *matrix;
    let (a, e, i) = (t - a, t - e, t - i);
    let second = a * e - b * d;
    let third = a * (e * i - f * h) - b * (d * i + f * g) - c * (d * h + e * g);
    a > 0.0 && second > 0.0 && third > 0.0
}

/// The spectral radius of a matrix with non-negative entries, by bisection on
/// [`radius_is_below`], which needs no convergence of the matrix's powers, so a reducible
/// or nearly critical matrix is measured like any other. The result is not below the radius,
/// but for rounding.
fn spectral_radius(matrix: &BranchingMatrix) -> f64 {
    let mut high = matrix
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .fold(0.0f64, f64::max);
    if !high.is_finite() {
        return f64::INFINITY;
    }
    let mut low = 0.0f64;
    for _ in 0..BISECTION_STEPS {
        let middle = low + (high - low) / 2.0;
        if middle <= low || middle >= high {
            break;
        }
        if radius_is_below(matrix, middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

impl HawkesParameters {
    /// Validates parameters.
    ///
    /// `rates[n]` is `(μ^L, μ^C, μ^M)` at queue size `n`, from 0 to the cap.
    /// `weights[u][l][m]` is `α_u^{lm}` for kernel `u` with decay `decays[u]`, in
    /// limit, cancel, market order.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a table with fewer than two
    /// rows or above [`MAX_QUEUE`]; a rate that is negative, not finite or above
    /// [`MAX_RATE`]; a departure from empty or an arrival at the cap; more than
    /// [`MAX_KERNELS`] kernels or mismatched lengths; a decay out of
    /// `[1 / MAX_RATE, MAX_RATE]`; a weight that is negative or not finite; or a
    /// branching ratio at or above 1.
    pub fn new(
        rates: RateTable,
        decays: Vec<f64>,
        weights: Vec<BranchingMatrix>,
    ) -> MMResult<Self> {
        let bound = usize::try_from(MAX_QUEUE).unwrap_or(usize::MAX);
        if rates.len() < 2 || rates.len() > bound.saturating_add(1) {
            return invalid(format!("{} rows of rates", rates.len()));
        }
        let valid_rate = |v: f64| v.is_finite() && (0.0..=MAX_RATE).contains(&v);
        if rates
            .iter()
            .any(|(l, c, m)| !valid_rate(*l) || !valid_rate(*c) || !valid_rate(*m))
        {
            return invalid("rates must be finite, not negative and bounded".to_string());
        }
        let empty_departs = rates.first().is_some_and(|(_, c, m)| *c > 0.0 || *m > 0.0);
        let cap_arrives = rates.last().is_some_and(|(l, _, _)| *l > 0.0);
        if empty_departs || cap_arrives {
            return invalid("departure from empty or arrival at the cap".to_string());
        }
        if decays.len() > MAX_KERNELS || decays.len() != weights.len() {
            return invalid(format!(
                "{} decays and {} weight matrices",
                decays.len(),
                weights.len()
            ));
        }
        if decays
            .iter()
            .any(|d| !d.is_finite() || *d < 1.0 / MAX_RATE || *d > MAX_RATE)
        {
            return invalid(format!("decays out of range: {decays:?}"));
        }
        if weights
            .iter()
            .flatten()
            .flatten()
            .any(|w| !w.is_finite() || *w < 0.0)
        {
            return invalid("weights must be finite and not negative".to_string());
        }
        let params = Self {
            rates,
            decays,
            weights,
        };
        let ratio = params.branching_ratio();
        if ratio >= 1.0 || !ratio.is_finite() {
            return invalid(format!("branching ratio {ratio} is not below 1"));
        }
        Ok(params)
    }

    /// Returns the spectral radius of `Σ_u α_u`.
    #[must_use]
    pub fn branching_ratio(&self) -> f64 {
        spectral_radius(&self.branching_matrix())
    }

    /// Returns `Σ_u α_u`, `[l][m]`.
    #[must_use]
    pub fn branching_matrix(&self) -> BranchingMatrix {
        let mut total = [[0.0; EVENT_TYPES]; EVENT_TYPES];
        for matrix in &self.weights {
            for (out, row) in total.iter_mut().zip(matrix) {
                for (cell, weight) in out.iter_mut().zip(row) {
                    *cell += weight;
                }
            }
        }
        total
    }

    /// Returns the baseline rates per queue size.
    #[must_use]
    pub fn rates(&self) -> &[(f64, f64, f64)] {
        &self.rates
    }

    /// Returns the kernel decays.
    #[must_use]
    pub fn decays(&self) -> &[f64] {
        &self.decays
    }

    /// Returns the largest queue size.
    #[must_use]
    pub fn max_queue(&self) -> usize {
        self.rates.len().saturating_sub(1)
    }

    /// Returns the FNV-1a 128 of the parameters.
    #[must_use]
    pub fn fingerprint(&self) -> u128 {
        let mut hash = FNV_OFFSET;
        for (l, c, m) in &self.rates {
            for value in [l, c, m] {
                hash = fnv(hash, &value.to_bits().to_le_bytes());
            }
        }
        for (decay, matrix) in self.decays.iter().zip(&self.weights) {
            hash = fnv(hash, &decay.to_bits().to_le_bytes());
            for weight in matrix.iter().flatten() {
                hash = fnv(hash, &weight.to_bits().to_le_bytes());
            }
        }
        hash
    }

    /// The baseline of type `l` at queue `n`.
    fn baseline(&self, l: usize, n: usize) -> f64 {
        self.rates
            .get(n)
            .map_or(0.0, |(limit, cancel, market)| match l {
                0 => *limit,
                1 => *cancel,
                _ => *market,
            })
    }

    /// Whether type `l` can happen at queue `n`.
    fn allowed(&self, l: usize, n: usize) -> bool {
        if l == 0 { n < self.max_queue() } else { n > 0 }
    }

    /// `λ^l` given the excitation `R_u^m = Σ exp(-β_u (t - s))`.
    fn intensity(&self, l: usize, n: usize, excitation: &Excitation) -> f64 {
        if !self.allowed(l, n) {
            return 0.0;
        }
        let mut total = self.baseline(l, n);
        for ((decay, matrix), memory) in self.decays.iter().zip(&self.weights).zip(excitation) {
            if let Some(row) = matrix.get(l) {
                total += decay
                    * row
                        .iter()
                        .zip(memory)
                        .map(|(alpha, r)| alpha * r)
                        .sum::<f64>();
            }
        }
        total
    }

    /// `∫ λ^l` from now over `delta` seconds, with the queue fixed.
    fn compensator(&self, l: usize, n: usize, excitation: &Excitation, delta: f64) -> f64 {
        if !self.allowed(l, n) {
            return 0.0;
        }
        let mut total = self.baseline(l, n) * delta;
        for ((decay, matrix), memory) in self.decays.iter().zip(&self.weights).zip(excitation) {
            let share = -(-decay * delta).exp_m1();
            if let Some(row) = matrix.get(l) {
                total += share
                    * row
                        .iter()
                        .zip(memory)
                        .map(|(alpha, r)| alpha * r)
                        .sum::<f64>();
            }
        }
        total
    }
}

/// `R_u^m = Σ exp(-β_u (t - s))` over past events of type `m`, `[u][m]`.
type Excitation = [[f64; EVENT_TYPES]; MAX_KERNELS];

fn decay_excitation(excitation: &mut Excitation, decays: &[f64], delta: f64) {
    for (memory, decay) in excitation.iter_mut().zip(decays) {
        let factor = (-decay * delta).exp();
        for value in memory.iter_mut() {
            *value *= factor;
        }
    }
}

/// Restorable state: fixed size and `Copy`, with the excitation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "RawState", try_from = "RawState"))]
pub struct HawkesState {
    /// Queue size.
    pub queue: u32,
    /// Time of the last accepted event or rejected candidate, in seconds.
    pub time_seconds: f64,
    /// How far runs have reached, in seconds.
    pub clock_seconds: f64,
    /// Excitation at `time_seconds`, `[u][m]`.
    pub excitation: [[f64; EVENT_TYPES]; MAX_KERNELS],
    /// Random generator state.
    pub rng_state: u64,
    /// Events so far.
    pub events: u64,
    /// Low half of the parameters' fingerprint.
    pub fingerprint_low: u64,
    /// High half of the parameters' fingerprint.
    pub fingerprint_high: u64,
}

/// Exported form of [`HawkesState`], with exact floats: a stored state goes on with the
/// same stream.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawState {
    queue: u32,
    time_seconds: String,
    clock_seconds: String,
    excitation: [[String; EVENT_TYPES]; MAX_KERNELS],
    rng_state: u64,
    events: u64,
    fingerprint_low: u64,
    fingerprint_high: u64,
}

#[cfg(feature = "serde")]
impl From<HawkesState> for RawState {
    fn from(state: HawkesState) -> Self {
        Self {
            queue: state.queue,
            time_seconds: float_text(state.time_seconds),
            clock_seconds: float_text(state.clock_seconds),
            excitation: state.excitation.map(|row| row.map(float_text)),
            rng_state: state.rng_state,
            events: state.events,
            fingerprint_low: state.fingerprint_low,
            fingerprint_high: state.fingerprint_high,
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<RawState> for HawkesState {
    type Error = MMError;

    fn try_from(raw: RawState) -> MMResult<Self> {
        let mut excitation = [[0.0; EVENT_TYPES]; MAX_KERNELS];
        for (row, texts) in excitation.iter_mut().zip(&raw.excitation) {
            for (cell, text) in row.iter_mut().zip(texts) {
                *cell = text_float(text)?;
            }
        }
        Ok(Self {
            queue: raw.queue,
            time_seconds: text_float(&raw.time_seconds)?,
            clock_seconds: text_float(&raw.clock_seconds)?,
            excitation,
            rng_state: raw.rng_state,
            events: raw.events,
            fingerprint_low: raw.fingerprint_low,
            fingerprint_high: raw.fingerprint_high,
        })
    }
}

fn fingerprint_parts(fingerprint: u128) -> (u64, u64) {
    let bytes = fingerprint.to_le_bytes();
    let mut low = [0u8; 8];
    let mut high = [0u8; 8];
    low.copy_from_slice(bytes.get(..8).unwrap_or(&[0; 8]));
    high.copy_from_slice(bytes.get(8..).unwrap_or(&[0; 8]));
    (u64::from_le_bytes(low), u64::from_le_bytes(high))
}

/// One simulated or observed event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HawkesEvent {
    /// Time from the start of the segment, in seconds.
    pub time_seconds: f64,
    /// Type of the event.
    pub kind: QrEventKind,
}

/// A period at a constant reference price, an independent realization.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HawkesSegment {
    /// Queue size at the start.
    pub initial_queue: u32,
    /// Length in seconds.
    pub duration_seconds: f64,
    /// Events, in time order.
    pub events: Vec<HawkesEvent>,
}

impl HawkesSegment {
    /// Checks the segment against a cap.
    fn validate(&self, max_queue: usize) -> MMResult<()> {
        let bad = |what: String| Err(MMError::InvalidMarketState(what));
        if !self.duration_seconds.is_finite() || self.duration_seconds <= 0.0 {
            return bad(format!("segment duration {}", self.duration_seconds));
        }
        let mut queue = usize::try_from(self.initial_queue).unwrap_or(usize::MAX);
        if queue > max_queue {
            return bad(format!(
                "initial queue {} above the cap",
                self.initial_queue
            ));
        }
        let mut last = 0.0f64;
        for event in &self.events {
            if !event.time_seconds.is_finite()
                || event.time_seconds < last
                || event.time_seconds > self.duration_seconds
            {
                return bad(format!("event at {} out of order", event.time_seconds));
            }
            last = event.time_seconds;
            queue = match event.kind {
                QrEventKind::Limit if queue < max_queue => queue.saturating_add(1),
                QrEventKind::Cancel | QrEventKind::Market if queue > 0 => queue.saturating_sub(1),
                kind => {
                    return bad(format!(
                        "{kind:?} at queue {queue} at {}",
                        event.time_seconds
                    ));
                }
            };
        }
        Ok(())
    }
}

/// The result of a simulator step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum HawkesStep {
    /// An event, at the state's time.
    Event(HawkesEvent),
    /// Nothing can happen: all intensities are zero, or time cannot advance.
    Stalled,
    /// More than [`MAX_THINNING`] candidates were rejected for one event.
    ThinningLimit,
    /// A run reached its limit of events.
    EventLimit,
}

/// A simulator by Ogata's thinning.
#[derive(Debug, Clone, PartialEq)]
pub struct HawkesSimulator {
    params: HawkesParameters,
    state: HawkesState,
}

impl HawkesSimulator {
    /// Starts with no excitation.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a queue above the cap.
    pub fn new(params: HawkesParameters, seed: u64, queue: u32) -> MMResult<Self> {
        if usize::try_from(queue).unwrap_or(usize::MAX) > params.max_queue() {
            return invalid(format!("queue {queue} above the cap"));
        }
        let (fingerprint_low, fingerprint_high) = fingerprint_parts(params.fingerprint());
        Ok(Self {
            params,
            state: HawkesState {
                queue,
                time_seconds: 0.0,
                clock_seconds: 0.0,
                excitation: [[0.0; EVENT_TYPES]; MAX_KERNELS],
                rng_state: if seed == 0 { ZERO_SEED } else { seed },
                events: 0,
                fingerprint_low,
                fingerprint_high,
            },
        })
    }

    /// Continues from a stored state.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a state of other parameters, a
    /// queue above the cap, a time that is not finite and not negative, a clock
    /// behind it, excitation that is negative, not finite, or on an unused kernel,
    /// or a zero generator state.
    pub fn from_state(params: HawkesParameters, state: HawkesState) -> MMResult<Self> {
        let parts = fingerprint_parts(params.fingerprint());
        let kernels = params.decays.len();
        let excitation_ok = state.excitation.iter().enumerate().all(|(u, memory)| {
            memory
                .iter()
                .all(|r| r.is_finite() && *r >= 0.0 && (u < kernels || *r == 0.0))
        });
        if parts != (state.fingerprint_low, state.fingerprint_high)
            || usize::try_from(state.queue).unwrap_or(usize::MAX) > params.max_queue()
            || !state.time_seconds.is_finite()
            || state.time_seconds < 0.0
            || !state.clock_seconds.is_finite()
            || state.clock_seconds < state.time_seconds
            || !excitation_ok
            || state.rng_state == 0
        {
            return invalid(format!("state does not fit the parameters: {state:?}"));
        }
        Ok(Self { params, state })
    }

    /// Returns the state, to store it.
    #[must_use]
    pub fn state(&self) -> HawkesState {
        self.state
    }

    /// Advances to the next event.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] on a failed conversion.
    pub fn step(&mut self) -> MMResult<HawkesStep> {
        let n = usize::try_from(self.state.queue).map_err(|_| overflow("queue"))?;
        let saved = self.state;
        for _ in 0..MAX_THINNING {
            // Excitation only decays until the next event, and the queue is fixed
            // until then, so the intensity now bounds it.
            let bound = self.total_intensity(n);
            if bound <= 0.0 {
                self.state = saved;
                return Ok(HawkesStep::Stalled);
            }
            let wait = -(1.0 - self.unit()).ln() / bound;
            let next = self.state.time_seconds + wait;
            if next <= self.state.time_seconds {
                self.state = saved;
                return Ok(HawkesStep::Stalled);
            }
            decay_excitation(&mut self.state.excitation, &self.params.decays, wait);
            self.state.time_seconds = next;
            let rates = self.intensities(n);
            let total: f64 = rates.iter().sum();
            if self.unit() * bound < total {
                let mut pick = self.unit() * total;
                let mut chosen = 0usize;
                for (index, rate) in rates.iter().enumerate() {
                    if *rate > 0.0 {
                        chosen = index;
                        pick -= rate;
                        if pick < 0.0 {
                            break;
                        }
                    }
                }
                return self.apply(chosen);
            }
        }
        self.state = saved;
        Ok(HawkesStep::ThinningLimit)
    }

    fn apply(&mut self, l: usize) -> MMResult<HawkesStep> {
        let kind = *KINDS.get(l).or_overflow("event type")?;
        self.state.queue = match kind {
            QrEventKind::Limit => self.state.queue.checked_add(1),
            QrEventKind::Cancel | QrEventKind::Market => self.state.queue.checked_sub(1),
        }
        .or_overflow("queue after event")?;
        for memory in self
            .state
            .excitation
            .iter_mut()
            .take(self.params.decays.len())
        {
            *memory.get_mut(l).or_overflow("excitation")? += 1.0;
        }
        self.state.events = self.state.events.saturating_add(1);
        self.state.clock_seconds = self.state.clock_seconds.max(self.state.time_seconds);
        Ok(HawkesStep::Event(HawkesEvent {
            time_seconds: self.state.time_seconds,
            kind,
        }))
    }

    fn intensities(&self, n: usize) -> [f64; EVENT_TYPES] {
        let mut rates = [0.0; EVENT_TYPES];
        for (l, rate) in rates.iter_mut().enumerate() {
            *rate = self.params.intensity(l, n, &self.state.excitation);
        }
        rates
    }

    fn total_intensity(&self, n: usize) -> f64 {
        self.intensities(n).iter().sum()
    }

    /// Runs for `horizon_seconds` and returns the events as a segment. The event that
    /// would pass the horizon is not applied, so consecutive runs equal one long run.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a horizon that is not finite and
    /// positive, and [`MMError::NumericalError`] on a failed conversion.
    pub fn run(&mut self, horizon_seconds: f64) -> MMResult<(HawkesSegment, Option<HawkesStep>)> {
        self.run_limited(horizon_seconds, MAX_RUN_EVENTS)
    }

    /// Runs like [`HawkesSimulator::run`], with a limit of events.
    /// A stop before the horizon, for any reason, leaves a segment that lasts up to where the
    /// simulation reached, which may be zero.
    ///
    /// # Errors
    ///
    /// As [`HawkesSimulator::run`].
    pub fn run_limited(
        &mut self,
        horizon_seconds: f64,
        max_events: usize,
    ) -> MMResult<(HawkesSegment, Option<HawkesStep>)> {
        if !horizon_seconds.is_finite() || horizon_seconds <= 0.0 {
            return invalid(format!("horizon {horizon_seconds}"));
        }
        let start = self.state.clock_seconds;
        let end = start + horizon_seconds;
        let mut segment = HawkesSegment {
            initial_queue: self.state.queue,
            duration_seconds: horizon_seconds,
            events: Vec::new(),
        };
        loop {
            if segment.events.len() >= max_events {
                self.stop_early(&mut segment, start);
                return Ok((segment, Some(HawkesStep::EventLimit)));
            }
            let before = self.state;
            match self.step()? {
                HawkesStep::Event(event) => {
                    if event.time_seconds > end {
                        self.state = before;
                        self.state.clock_seconds = end;
                        return Ok((segment, None));
                    }
                    segment.events.push(HawkesEvent {
                        time_seconds: event.time_seconds - start,
                        kind: event.kind,
                    });
                }
                stop @ (HawkesStep::Stalled
                | HawkesStep::ThinningLimit
                | HawkesStep::EventLimit) => {
                    self.stop_early(&mut segment, start);
                    return Ok((segment, Some(stop)));
                }
            }
        }
    }

    /// A segment that stopped early lasts only as long as was simulated, so its exposure
    /// claims no time nothing happened in, and the next run starts from there.
    fn stop_early(&mut self, segment: &mut HawkesSegment, start: f64) {
        let reached = self.state.time_seconds.max(start);
        self.state.clock_seconds = reached;
        segment.duration_seconds = reached - start;
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
}

fn overflow(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} overflows"))
}

#[allow(clippy::cast_precision_loss)] // Counts far below 2^53.
fn count_as_float(value: u64) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)] // Report values: counts far below 2^53.
fn as_float(value: usize) -> f64 {
    value as f64
}

/// Estimation settings.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HawkesFitConfig {
    /// Queue cap. Positive and at most [`MAX_QUEUE`].
    pub max_queue: u32,
    /// Fixed decays; empty fits the queue-reactive model.
    pub decays: Vec<f64>,
    /// Expectation maximization steps. Positive.
    pub iterations: u32,
    /// Fewest events for a queue size's baseline. Positive.
    pub min_events: u64,
}

impl HawkesFitConfig {
    /// Creates settings with 200 steps and a minimum of 30 events per size.
    #[must_use]
    pub fn new(max_queue: u32, decays: Vec<f64>) -> Self {
        Self {
            max_queue,
            decays,
            iterations: 200,
            min_events: 30,
        }
    }

    /// Sets the expectation maximization steps.
    #[must_use]
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the fewest events for a queue size's baseline.
    #[must_use]
    pub fn with_min_events(mut self, min_events: u64) -> Self {
        self.min_events = min_events;
        self
    }
}

/// Why no parameters could be estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum HawkesFailure {
    /// No queue size reached `min_events`.
    NoSupportedSize,
    /// The fitted branching ratio is not below 1.
    Unstable,
    /// The fitted parameters are not valid, for example a rate above the bound.
    InvalidEstimate,
}

/// The result of an estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum HawkesFit {
    /// Parameters.
    Calibrated(HawkesParameters),
    /// None.
    Insufficient(HawkesFailure),
}

/// Parameters with what they were estimated from.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct HawkesRecord {
    /// FNV-1a 128 of the segments.
    pub dataset_hash: u128,
    /// Events used.
    pub events: u64,
    /// Seconds observed.
    pub seconds: f64,
    /// Queue sizes that took their baselines from the nearest supported sizes.
    pub borrowed_sizes: u32,
    /// Log likelihood before each step, then at the fitted parameters.
    pub likelihood_trace: Vec<f64>,
    /// The parameters, or why there are none.
    pub fit: HawkesFit,
}

impl HawkesRecord {
    /// Returns the parameters, if any.
    #[must_use]
    pub fn parameters(&self) -> Option<&HawkesParameters> {
        match &self.fit {
            HawkesFit::Calibrated(params) => Some(params),
            HawkesFit::Insufficient(_) => None,
        }
    }
}

/// Sufficient statistics of one expectation step.
struct Expected {
    background: Vec<[f64; EVENT_TYPES]>,
    exposure: Vec<f64>,
    offspring: Vec<BranchingMatrix>,
    weight_exposure: Vec<BranchingMatrix>,
    log_likelihood: f64,
}

/// One pass over the data: the expected split of each event between baseline and
/// kernels, exposures, and the log likelihood under `params`, with `rows` sizes.
fn expectation(segments: &[HawkesSegment], params: &HawkesParameters) -> MMResult<Expected> {
    let rows = params.rates.len();
    let kernels = params.decays.len();
    let mut out = Expected {
        background: vec![[0.0; EVENT_TYPES]; rows],
        exposure: vec![0.0; rows],
        offspring: vec![[[0.0; EVENT_TYPES]; EVENT_TYPES]; kernels],
        weight_exposure: vec![[[0.0; EVENT_TYPES]; EVENT_TYPES]; kernels],
        log_likelihood: 0.0,
    };
    for segment in segments {
        let mut excitation: Excitation = [[0.0; EVENT_TYPES]; MAX_KERNELS];
        let mut queue = usize::try_from(segment.initial_queue).map_err(|_| overflow("queue"))?;
        let mut now = 0.0f64;
        let interval = |excitation: &mut Excitation,
                        out: &mut Expected,
                        n: usize,
                        delta: f64|
         -> MMResult<()> {
            *out.exposure.get_mut(n).or_overflow("exposure")? += delta;
            for l in 0..EVENT_TYPES {
                out.log_likelihood -= params.compensator(l, n, excitation, delta);
                if !params.allowed(l, n) {
                    continue;
                }
                for (u, decay) in params.decays.iter().enumerate() {
                    let share = -(-decay * delta).exp_m1();
                    let memory = excitation.get(u).or_overflow("excitation")?;
                    let row = out
                        .weight_exposure
                        .get_mut(u)
                        .and_then(|matrix| matrix.get_mut(l))
                        .or_overflow("exposure")?;
                    for (cell, r) in row.iter_mut().zip(memory) {
                        *cell += share * r;
                    }
                }
            }
            decay_excitation(excitation, &params.decays, delta);
            Ok(())
        };
        for event in &segment.events {
            interval(&mut excitation, &mut out, queue, event.time_seconds - now)?;
            now = event.time_seconds;
            let l = index_of(event.kind).or_overflow("event type")?;
            let total = params.intensity(l, queue, &excitation).max(INTENSITY_FLOOR);
            out.log_likelihood += total.ln();
            let share = params.baseline(l, queue) / total;
            *out.background
                .get_mut(queue)
                .and_then(|row| row.get_mut(l))
                .or_overflow("background")? += share;
            for (u, (decay, matrix)) in params.decays.iter().zip(&params.weights).enumerate() {
                let row = matrix.get(l).or_overflow("weights")?;
                let memory = excitation.get(u).or_overflow("excitation")?;
                let target = out
                    .offspring
                    .get_mut(u)
                    .and_then(|m| m.get_mut(l))
                    .or_overflow("offspring")?;
                for ((cell, alpha), r) in target.iter_mut().zip(row).zip(memory) {
                    *cell += decay * alpha * r / total;
                }
            }
            queue = match event.kind {
                QrEventKind::Limit => queue.saturating_add(1),
                QrEventKind::Cancel | QrEventKind::Market => queue.saturating_sub(1),
            };
            for memory in excitation.iter_mut().take(kernels) {
                *memory.get_mut(l).or_overflow("excitation")? += 1.0;
            }
        }
        interval(
            &mut excitation,
            &mut out,
            queue,
            segment.duration_seconds - now,
        )?;
    }
    Ok(out)
}

/// Estimates by expectation maximization. Each step cannot lower the likelihood, apart
/// from the floor, rounding and a size that borrows its baseline from another, and the trace
/// in the record shows it.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for invalid settings, decays included, and
/// [`MMError::InvalidMarketState`] for a segment that breaks the queue rules.
pub fn fit_hawkes(segments: &[HawkesSegment], config: &HawkesFitConfig) -> MMResult<HawkesRecord> {
    if config.decays.len() > MAX_KERNELS
        || config
            .decays
            .iter()
            .any(|decay| !(decay.is_finite() && *decay >= 1.0 / MAX_RATE && *decay <= MAX_RATE))
    {
        return invalid(format!("decays {:?}", config.decays));
    }
    if config.max_queue == 0
        || config.max_queue > MAX_QUEUE
        || config.iterations == 0
        || config.min_events == 0
    {
        return invalid(format!("invalid fit settings: {config:?}"));
    }
    let max_queue = usize::try_from(config.max_queue).map_err(|_| overflow("cap"))?;
    let rows = max_queue.checked_add(1).or_overflow("rows")?;
    let mut counts = vec![[0u64; EVENT_TYPES]; rows];
    let mut exposure = vec![0.0f64; rows];
    let mut hash = FNV_OFFSET;
    let mut events = 0u64;
    let mut seconds = 0.0f64;
    for segment in segments {
        segment.validate(max_queue)?;
        hash = fnv(hash, &segment.initial_queue.to_le_bytes());
        hash = fnv(hash, &segment.duration_seconds.to_bits().to_le_bytes());
        seconds += segment.duration_seconds;
        let mut queue = usize::try_from(segment.initial_queue).map_err(|_| overflow("queue"))?;
        let mut now = 0.0f64;
        for event in &segment.events {
            hash = fnv(hash, &event.time_seconds.to_bits().to_le_bytes());
            let l = index_of(event.kind).or_overflow("event type")?;
            hash = fnv(hash, &[u8::try_from(l).unwrap_or(u8::MAX)]);
            *exposure.get_mut(queue).or_overflow("exposure")? += event.time_seconds - now;
            now = event.time_seconds;
            let row = counts.get_mut(queue).and_then(|r| r.get_mut(l));
            let cell = row.or_overflow("counts")?;
            *cell = cell.saturating_add(1);
            events = events.saturating_add(1);
            queue = match event.kind {
                QrEventKind::Limit => queue.saturating_add(1),
                QrEventKind::Cancel | QrEventKind::Market => queue.saturating_sub(1),
            };
        }
        *exposure.get_mut(queue).or_overflow("exposure")? += segment.duration_seconds - now;
    }

    let supported: Vec<bool> = counts
        .iter()
        .map(|row| row.iter().sum::<u64>() >= config.min_events)
        .collect();
    let finish = |fit: HawkesFit, borrowed_sizes: u32, likelihood_trace: Vec<f64>| HawkesRecord {
        dataset_hash: hash,
        events,
        seconds,
        borrowed_sizes,
        likelihood_trace,
        fit,
    };
    if !supported.contains(&true) {
        return Ok(finish(
            HawkesFit::Insufficient(HawkesFailure::NoSupportedSize),
            0,
            Vec::new(),
        ));
    }
    let kernels = config.decays.len();
    // Each pair starts at the same weight, well inside the stable region.
    let start = 0.5 / (as_float(EVENT_TYPES) * as_float(kernels.max(1)));
    let mut weights = vec![[[start; EVENT_TYPES]; EVENT_TYPES]; kernels];
    // The starting baselines are the plain rates.
    let mut background: Vec<[f64; EVENT_TYPES]> =
        counts.iter().map(|row| row.map(count_as_float)).collect();
    let mut rates = baseline_table(&background, &exposure, &supported);
    let mut trace = Vec::new();
    let usize_iterations = usize::try_from(config.iterations).map_err(|_| overflow("steps"))?;
    let mut params =
        match HawkesParameters::new(rates.clone(), config.decays.clone(), weights.clone()) {
            Ok(params) => params,
            Err(MMError::InvalidConfiguration(_)) => {
                return Ok(finish(
                    HawkesFit::Insufficient(HawkesFailure::InvalidEstimate),
                    0,
                    trace,
                ));
            }
            Err(error) => return Err(error),
        };
    for _ in 0..usize_iterations {
        let expected = expectation(segments, &params)?;
        trace.push(expected.log_likelihood);
        background = expected.background;
        rates = baseline_table(&background, &expected.exposure, &supported);
        for ((target, offspring), exposures) in weights
            .iter_mut()
            .zip(&expected.offspring)
            .zip(&expected.weight_exposure)
        {
            for ((target_row, offspring_row), exposure_row) in
                target.iter_mut().zip(offspring).zip(exposures)
            {
                for ((cell, count), denominator) in
                    target_row.iter_mut().zip(offspring_row).zip(exposure_row)
                {
                    *cell = if *denominator > 0.0 {
                        count / denominator
                    } else {
                        0.0
                    };
                }
            }
        }
        match HawkesParameters::new(rates.clone(), config.decays.clone(), weights.clone()) {
            Ok(next) => params = next,
            Err(MMError::InvalidConfiguration(message)) => {
                let failure = if message.contains("branching") {
                    HawkesFailure::Unstable
                } else {
                    HawkesFailure::InvalidEstimate
                };
                return Ok(finish(HawkesFit::Insufficient(failure), 0, trace));
            }
            Err(error) => return Err(error),
        }
    }
    trace.push(expectation(segments, &params)?.log_likelihood);
    let borrowed = supported.iter().filter(|s| !**s).count();
    let borrowed_sizes = u32::try_from(borrowed).unwrap_or(u32::MAX);
    Ok(finish(HawkesFit::Calibrated(params), borrowed_sizes, trace))
}

/// Rates from expected counts and exposures; an unsupported size takes the nearest
/// supported row, with the queue rules kept.
fn baseline_table(
    background: &[[f64; EVENT_TYPES]],
    exposure: &[f64],
    supported: &[bool],
) -> RateTable {
    let last = background.len().saturating_sub(1);
    let raw = |n: usize| -> (f64, f64, f64) {
        match (background.get(n), exposure.get(n)) {
            (Some([l, c, m]), Some(time)) if *time > 0.0 => (l / time, c / time, m / time),
            _ => (0.0, 0.0, 0.0),
        }
    };
    // Per type: the row it comes from must be one where that type is allowed.
    let rate = |n: usize, kind: usize, value: fn((f64, f64, f64)) -> f64| -> f64 {
        let allowed = |i: usize| if kind == 0 { i < last } else { i > 0 };
        if !allowed(n) {
            return 0.0;
        }
        let source = if supported.get(n).copied().unwrap_or(false) {
            Some(n)
        } else {
            supported
                .iter()
                .enumerate()
                .filter(|(i, s)| **s && allowed(*i))
                .map(|(i, _)| i)
                .min_by_key(|i| i.abs_diff(n))
        };
        source.map_or(0.0, |source| value(raw(source)))
    };
    (0..background.len())
        .map(|n| {
            (
                rate(n, 0, |row| row.0),
                rate(n, 1, |row| row.1),
                rate(n, 2, |row| row.2),
            )
        })
        .collect()
}

/// Fits the Poisson baseline: a constant rate per type over the time the type is
/// allowed, with the queue rules, and no kernels.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a cap out of range or no time at
/// which a type is allowed, and [`MMError::InvalidMarketState`] for a segment that
/// breaks the queue rules.
pub fn fit_poisson(segments: &[HawkesSegment], max_queue: u32) -> MMResult<HawkesParameters> {
    if max_queue == 0 || max_queue > MAX_QUEUE {
        return invalid(format!("cap {max_queue}"));
    }
    let cap = usize::try_from(max_queue).map_err(|_| overflow("cap"))?;
    let (mut count, mut exposed) = ([0u64; EVENT_TYPES], [0.0f64; EVENT_TYPES]);
    for segment in segments {
        segment.validate(cap)?;
        let mut queue = usize::try_from(segment.initial_queue).map_err(|_| overflow("queue"))?;
        let mut now = 0.0f64;
        let mut expose = |queue: usize, delta: f64| {
            for (l, time) in exposed.iter_mut().enumerate() {
                if (l == 0 && queue < cap) || (l > 0 && queue > 0) {
                    *time += delta;
                }
            }
        };
        for event in &segment.events {
            expose(queue, event.time_seconds - now);
            now = event.time_seconds;
            let l = index_of(event.kind).or_overflow("event type")?;
            let cell = count.get_mut(l).or_overflow("counts")?;
            *cell = cell.saturating_add(1);
            queue = match event.kind {
                QrEventKind::Limit => queue.saturating_add(1),
                QrEventKind::Cancel | QrEventKind::Market => queue.saturating_sub(1),
            };
        }
        expose(queue, segment.duration_seconds - now);
    }
    if exposed.iter().any(|t| *t <= 0.0) {
        return invalid("a type has no exposure".to_string());
    }
    let [limit, cancel, market] = [0usize, 1, 2].map(|l| {
        count_as_float(count.get(l).copied().unwrap_or(0)) / exposed.get(l).copied().unwrap_or(1.0)
    });
    let rates = (0..=cap)
        .map(|n| {
            (
                if n == cap { 0.0 } else { limit },
                if n == 0 { 0.0 } else { cancel },
                if n == 0 { 0.0 } else { market },
            )
        })
        .collect();
    HawkesParameters::new(rates, Vec::new(), Vec::new())
}

/// An intraday profile of activity: a time change of the whole flow.
///
/// With `s(t)` piecewise constant, of mean 1 over the period, and `τ(t) = ∫ s`, a seasonal
/// flow is a stationary model in `τ`: `λ(t) = s(t) λ̃(τ(t))`. So busy hours speed up the
/// baseline and the kernels' decay alike, which is a departure from the paper, whose
/// kernels stay in clock time. Calibrate it on training only with [`fit_seasonality`], fit
/// and simulate in model time, and score with [`score_flow_seasonal`]. A segment's time 0 is
/// the start of the period, so segments must be aligned to it.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "RawProfile", try_from = "RawProfile"))]
pub struct SeasonalProfile {
    bucket_seconds: f64,
    multipliers: Vec<f64>,
}

/// Exported form of a [`SeasonalProfile`], with exact floats, checked when read back.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    bucket_seconds: String,
    multipliers: Vec<String>,
}

#[cfg(feature = "serde")]
impl From<SeasonalProfile> for RawProfile {
    fn from(profile: SeasonalProfile) -> Self {
        Self {
            bucket_seconds: float_text(profile.bucket_seconds),
            multipliers: profile.multipliers.iter().map(|m| float_text(*m)).collect(),
        }
    }
}

/// How far from 1 the mean of `count` stored multipliers may be: the rounding of a
/// normalization, which grows with the sum.
#[cfg(feature = "serde")]
fn stored_mean_tolerance(count: usize) -> f64 {
    (as_float(count) * f64::EPSILON).max(1e-12)
}

#[cfg(feature = "serde")]
impl TryFrom<RawProfile> for SeasonalProfile {
    type Error = MMError;

    /// Holds a profile to what [`SeasonalProfile::new`] gives, and keeps its multipliers as
    /// stored, so that it reads back equal: a second normalization could move them.
    fn try_from(raw: RawProfile) -> MMResult<Self> {
        let bucket_seconds = text_float(&raw.bucket_seconds)?;
        let multipliers = raw
            .multipliers
            .iter()
            .map(|m| text_float(m))
            .collect::<MMResult<Vec<f64>>>()?;
        check_profile(bucket_seconds, &multipliers)?;
        let mean = multipliers.iter().sum::<f64>() / as_float(multipliers.len());
        if (mean - 1.0).abs() > stored_mean_tolerance(multipliers.len()) {
            return invalid(format!("stored profile of mean {mean}"));
        }
        Ok(Self {
            bucket_seconds,
            multipliers,
        })
    }
}

/// Bounded, so that each bucket keeps a span in model time and the change inverts.
fn check_profile(bucket_seconds: f64, multipliers: &[f64]) -> MMResult<()> {
    let bounded = 1.0 / MAX_RATE..=MAX_RATE;
    if !bounded.contains(&bucket_seconds)
        || multipliers.is_empty()
        || multipliers.len() > MAX_SEASONAL_BUCKETS
        || multipliers.iter().any(|m| !bounded.contains(m))
    {
        return invalid(format!(
            "profile of {} buckets of {bucket_seconds}",
            multipliers.len()
        ));
    }
    Ok(())
}

impl SeasonalProfile {
    /// A profile of equal buckets, repeating. The multipliers are normalized to mean 1.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a bucket out of
    /// `[1 / MAX_RATE, MAX_RATE]`, no multipliers or more than [`MAX_SEASONAL_BUCKETS`], or
    /// one that is not finite and positive, or out of `[1 / MAX_RATE, MAX_RATE]` after the
    /// normalization.
    pub fn new(bucket_seconds: f64, multipliers: Vec<f64>) -> MMResult<Self> {
        if multipliers.is_empty() || multipliers.iter().any(|m| !m.is_finite() || *m <= 0.0) {
            return invalid(format!(
                "profile of {} buckets of {bucket_seconds}",
                multipliers.len()
            ));
        }
        let mean = multipliers.iter().sum::<f64>() / as_float(multipliers.len());
        if !mean.is_finite() || mean <= 0.0 {
            return invalid(format!("profile mean {mean}"));
        }
        let multipliers: Vec<f64> = multipliers.iter().map(|m| m / mean).collect();
        check_profile(bucket_seconds, &multipliers)?;
        Ok(Self {
            bucket_seconds,
            multipliers,
        })
    }

    /// Returns the length of a bucket, in seconds.
    #[must_use]
    pub fn bucket_seconds(&self) -> f64 {
        self.bucket_seconds
    }

    /// Returns the multipliers, of mean 1.
    #[must_use]
    pub fn multipliers(&self) -> &[f64] {
        &self.multipliers
    }

    fn period(&self) -> f64 {
        self.bucket_seconds * as_float(self.multipliers.len())
    }

    /// A period in model time. The period itself, but for the rounding of the normalization,
    /// which the maps must share or they would step at a period's edge.
    fn model_period(&self) -> f64 {
        self.multipliers.iter().sum::<f64>() * self.bucket_seconds
    }

    /// Whole buckets before `t`, and the bucket holding it. Time cannot be negative here.
    fn bucket_of(&self, t: f64) -> (f64, usize) {
        let whole = (t / self.bucket_seconds).floor().max(0.0);
        (whole, self.index_of_bucket(whole))
    }

    fn index_of_bucket(&self, whole: f64) -> usize {
        let count = as_float(self.multipliers.len());
        let index = whole - (whole / count).floor() * count;
        let index = rust_decimal::prelude::ToPrimitive::to_usize(&index).unwrap_or(0);
        index.min(self.multipliers.len().saturating_sub(1))
    }

    /// `s(t)`.
    #[must_use]
    pub fn multiplier(&self, t: f64) -> f64 {
        let (_, index) = self.bucket_of(t);
        self.multipliers.get(index).copied().unwrap_or(1.0)
    }

    /// `τ(t)`, of clock time `t`.
    ///
    /// Monotonic only within rounding at a bucket's edge, where two close times can swap by
    /// about `1e-16` of the time. [`Self::segment_to_model_time`] keeps a segment in order.
    #[must_use]
    pub fn to_model_time(&self, t: f64) -> f64 {
        let (whole, index) = self.bucket_of(t);
        let periods = (whole / as_float(self.multipliers.len())).floor();
        let before: f64 = self.multipliers.iter().take(index).sum();
        let start = whole * self.bucket_seconds;
        periods * self.model_period()
            + before * self.bucket_seconds
            + self.multiplier(t) * (t - start)
    }

    /// The clock time of model time `tau`, the inverse of [`Self::to_model_time`].
    #[must_use]
    pub fn to_clock_time(&self, tau: f64) -> f64 {
        let model_period = self.model_period();
        let periods = (tau / model_period).floor().max(0.0);
        let mut left = tau - periods * model_period;
        let mut clock = periods * self.period();
        for multiplier in &self.multipliers {
            let span = multiplier * self.bucket_seconds;
            if left < span {
                return clock + left / multiplier;
            }
            left -= span;
            clock += self.bucket_seconds;
        }
        clock
    }

    /// A segment in model time, to fit or to score. Its order survives the rounding of the
    /// map: an event that would step back by a rounding takes the time of the one before it.
    /// A segment out of order stays out of order, for its validation to refuse.
    #[must_use]
    pub fn segment_to_model_time(&self, segment: &HawkesSegment) -> HawkesSegment {
        self.map_segment(segment, |t| self.to_model_time(t))
    }

    /// A segment simulated in model time, in clock time, its order kept in the same way.
    #[must_use]
    pub fn segment_to_clock_time(&self, segment: &HawkesSegment) -> HawkesSegment {
        self.map_segment(segment, |t| self.to_clock_time(t))
    }

    fn map_segment(&self, segment: &HawkesSegment, map: impl Fn(f64) -> f64) -> HawkesSegment {
        let mut last = 0.0f64;
        let events: Vec<HawkesEvent> = segment
            .events
            .iter()
            .map(|event| {
                last = after_rounding(map(event.time_seconds), last);
                HawkesEvent {
                    time_seconds: last,
                    kind: event.kind,
                }
            })
            .collect();
        HawkesSegment {
            initial_queue: segment.initial_queue,
            duration_seconds: after_rounding(map(segment.duration_seconds), last),
            events,
        }
    }
}

/// `mapped`, or `last` if it stepped back from it by no more than the rounding of a time map.
fn after_rounding(mapped: f64, last: f64) -> f64 {
    let rounding = 16.0 * f64::EPSILON * last.abs().max(1.0);
    if mapped < last && last - mapped <= rounding {
        last
    } else {
        mapped
    }
}

/// Calibrates a profile as the event rate per bucket over the mean, from `segments` alone,
/// so pass training only. This is a moment estimate, not a joint fit, and it is confounded
/// with the flow's own state: excitation inflates a bucket's count, each segment starts
/// with no memory, which deflates the first bucket, and departures stop at an empty queue.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid bucket, no buckets or more than
/// [`MAX_SEASONAL_BUCKETS`], refused before allocating, a segment of more than
/// [`MAX_INTERVALS`] buckets, and
/// [`MMError::InvalidMarketState`] for a segment out of order, or a bucket with no time
/// observed or too few events, which is refused rather than filled in.
pub fn fit_seasonality(
    segments: &[HawkesSegment],
    max_queue: u32,
    bucket_seconds: f64,
    buckets: usize,
    min_events: u64,
) -> MMResult<SeasonalProfile> {
    if buckets > MAX_SEASONAL_BUCKETS || max_queue == 0 || max_queue > MAX_QUEUE || min_events == 0
    {
        return invalid(format!(
            "{buckets} buckets, cap {max_queue}, {min_events} events"
        ));
    }
    let flat = SeasonalProfile::new(bucket_seconds, vec![1.0; buckets])?;
    let cap = usize::try_from(max_queue).map_err(|_| overflow("cap"))?;
    let mut counts = vec![0u64; buckets];
    let mut exposure = vec![0.0f64; buckets];
    for segment in segments {
        segment.validate(cap)?;
        // An event at the very end has no exposure in the bucket after it.
        for event in segment
            .events
            .iter()
            .filter(|event| event.time_seconds < segment.duration_seconds)
        {
            let (_, index) = flat.bucket_of(event.time_seconds);
            let count = counts.get_mut(index).or_overflow("bucket")?;
            *count = count.saturating_add(1);
        }
        if segment.duration_seconds / bucket_seconds > MAX_INTERVALS {
            return invalid(format!(
                "{} seconds in buckets of {bucket_seconds}",
                segment.duration_seconds
            ));
        }
        // By whole buckets, as a time at an edge can round into the bucket before it.
        let mut whole = 0.0f64;
        let mut start = 0.0f64;
        while start < segment.duration_seconds {
            let end = ((whole + 1.0) * bucket_seconds).min(segment.duration_seconds);
            let index = flat.index_of_bucket(whole);
            *exposure.get_mut(index).or_overflow("bucket")? += end - start;
            whole += 1.0;
            start = end;
        }
    }
    let mut rates = Vec::with_capacity(buckets);
    for (index, (count, time)) in counts.iter().zip(&exposure).enumerate() {
        if *count < min_events {
            return Err(MMError::InvalidMarketState(format!(
                "bucket {index}: {count} events in {time} seconds"
            )));
        }
        rates.push(count_as_float(*count) / time);
    }
    SeasonalProfile::new(bucket_seconds, rates)
}

/// Scores `params`, fitted in model time, on clock time `segments` under `profile`: the
/// likelihood gains `Σ ln s(t_i)` from the change of time, and the residuals are unchanged.
/// So it is comparable with [`score_flow`] on the same segments, and it refuses what that
/// refuses. The parameter count adds the
/// profile's free multipliers.
///
/// # Errors
///
/// As [`score_flow`].
pub fn score_flow_seasonal(
    segments: &[HawkesSegment],
    params: &HawkesParameters,
    profile: &SeasonalProfile,
) -> MMResult<FlowFit> {
    // Before the map, whose rounding allowance must not read as an order.
    for segment in segments {
        segment.validate(params.max_queue())?;
    }
    let mapped: Vec<HawkesSegment> = segments
        .iter()
        .map(|segment| profile.segment_to_model_time(segment))
        .collect();
    let mut fit = score_flow(&mapped, params)?;
    let jacobian: f64 = segments
        .iter()
        .flat_map(|segment| &segment.events)
        .map(|event| profile.multiplier(event.time_seconds).ln())
        .sum();
    let extra = profile.multipliers().len().saturating_sub(1);
    fit.log_likelihood += jacobian;
    fit.log_likelihood_per_event = fit.log_likelihood / count_as_float(fit.events);
    fit.parameter_count = fit.parameter_count.saturating_add(extra);
    fit.aic = 2.0 * as_float(fit.parameter_count) - 2.0 * fit.log_likelihood;
    Ok(fit)
}

/// Goodness of fit of parameters on some segments.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct FlowFit {
    /// Log likelihood.
    pub log_likelihood: f64,
    /// Log likelihood per event, to compare across sets.
    pub log_likelihood_per_event: f64,
    /// Akaike information criterion with `parameter_count`, comparable only on the
    /// same segments.
    pub aic: f64,
    /// Free parameters counted for the criterion: distinct positive baselines per
    /// type, and positive weights.
    pub parameter_count: usize,
    /// Kolmogorov-Smirnov distance from the unit exponential of the time-rescaled
    /// residuals, for limit, cancel and market orders.
    pub residual_ks: [f64; EVENT_TYPES],
    /// Events seen.
    pub events: u64,
    /// Events at a queue size where the model gives no intensity, which took the floor.
    pub floored_events: u64,
}

/// Scores `params` on `segments`.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for no events or a segment that breaks the
/// queue rules, and [`MMError::NumericalError`] on a failed conversion.
pub fn score_flow(segments: &[HawkesSegment], params: &HawkesParameters) -> MMResult<FlowFit> {
    let mut log_likelihood = 0.0f64;
    let mut events = 0u64;
    let mut floored = 0u64;
    let mut residuals: [Vec<f64>; EVENT_TYPES] = [Vec::new(), Vec::new(), Vec::new()];
    for segment in segments {
        segment.validate(params.max_queue())?;
        let mut excitation: Excitation = [[0.0; EVENT_TYPES]; MAX_KERNELS];
        let mut queue = usize::try_from(segment.initial_queue).map_err(|_| overflow("queue"))?;
        let mut now = 0.0f64;
        let mut accumulated = [0.0f64; EVENT_TYPES];
        for event in &segment.events {
            let delta = event.time_seconds - now;
            for (l, sum) in accumulated.iter_mut().enumerate() {
                let piece = params.compensator(l, queue, &excitation, delta);
                *sum += piece;
                log_likelihood -= piece;
            }
            decay_excitation(&mut excitation, &params.decays, delta);
            now = event.time_seconds;
            let l = index_of(event.kind).or_overflow("event type")?;
            let intensity = params.intensity(l, queue, &excitation);
            if intensity <= INTENSITY_FLOOR {
                floored = floored.saturating_add(1);
            }
            log_likelihood += intensity.max(INTENSITY_FLOOR).ln();
            let sum = accumulated.get_mut(l).or_overflow("residual")?;
            residuals.get_mut(l).or_overflow("residual")?.push(*sum);
            *sum = 0.0;
            events = events.saturating_add(1);
            queue = match event.kind {
                QrEventKind::Limit => queue.saturating_add(1),
                QrEventKind::Cancel | QrEventKind::Market => queue.saturating_sub(1),
            };
            for memory in excitation.iter_mut().take(params.decays.len()) {
                *memory.get_mut(l).or_overflow("excitation")? += 1.0;
            }
        }
        let delta = segment.duration_seconds - now;
        for l in 0..EVENT_TYPES {
            log_likelihood -= params.compensator(l, queue, &excitation, delta);
        }
    }
    if events == 0 {
        return Err(MMError::InvalidMarketState(
            "no events to score".to_string(),
        ));
    }
    // Distinct baseline values per type, so a constant rate counts once.
    let mut rate_count = 0usize;
    for pick in [
        (|r: &(f64, f64, f64)| r.0) as fn(&(f64, f64, f64)) -> f64,
        |r| r.1,
        |r| r.2,
    ] {
        let mut values: Vec<u64> = params
            .rates
            .iter()
            .map(pick)
            .filter(|v| *v > 0.0)
            .map(f64::to_bits)
            .collect();
        values.sort_unstable();
        values.dedup();
        rate_count = rate_count.saturating_add(values.len());
    }
    let weight_count = params
        .weights
        .iter()
        .flatten()
        .flatten()
        .filter(|w| **w > 0.0)
        .count();
    let parameter_count = rate_count.saturating_add(weight_count);
    let [limit, cancel, market] = &residuals;
    let per_event = log_likelihood / count_as_float(events);
    Ok(FlowFit {
        log_likelihood,
        log_likelihood_per_event: per_event,
        aic: 2.0 * as_float(parameter_count) - 2.0 * log_likelihood,
        parameter_count,
        residual_ks: [
            exponential_ks(limit),
            exponential_ks(cancel),
            exponential_ks(market),
        ],
        events,
        floored_events: floored,
    })
}

/// Kolmogorov-Smirnov distance from the unit exponential; 1 without values.
fn exponential_ks(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = as_float(sorted.len());
    let mut largest = 0.0f64;
    for (i, value) in sorted.iter().enumerate() {
        let cdf = -(-value).exp_m1();
        let below = as_float(i) / n;
        let above = as_float(i.saturating_add(1)) / n;
        largest = largest.max((cdf - below).abs()).max((above - cdf).abs());
    }
    largest
}

/// Validation scores of one candidate decay.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct DecayScore {
    /// The candidate.
    pub decay: f64,
    /// Log likelihood per event on the training set.
    pub training: f64,
    /// Log likelihood per event on the validation set; `None` if the fit failed.
    pub validation: Option<f64>,
}

/// Fits each candidate decay on `train` and scores it on `validation`, and returns
/// all the scores, which are the sensitivity, and the index of the best. It never
/// sees a holdout. A candidate that fails to fit stays in the list without a score.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for no candidates or invalid settings,
/// [`MMError::InvalidMarketState`] for an invalid segment, and
/// [`MMError::NumericalError`] on a failed conversion.
pub fn select_decay(
    train: &[HawkesSegment],
    validation: &[HawkesSegment],
    candidates: &[f64],
    config: &HawkesFitConfig,
) -> MMResult<(Vec<DecayScore>, Option<usize>)> {
    if candidates.is_empty() {
        return invalid("no candidate decays".to_string());
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for decay in candidates {
        let mut candidate = config.clone();
        candidate.decays = vec![*decay];
        let record = fit_hawkes(train, &candidate)?;
        let (training, validated) = match record.parameters() {
            Some(params) => (
                score_flow(train, params)?.log_likelihood_per_event,
                Some(score_flow(validation, params)?.log_likelihood_per_event),
            ),
            None => (f64::NAN, None),
        };
        scores.push(DecayScore {
            decay: *decay,
            training,
            validation: validated,
        });
    }
    let best = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.validation.map(|v| (i, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    Ok((scores, best))
}

/// Events per interval of one type, in whole intervals of the segments.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an interval that is not positive.
pub fn events_per_interval(
    segments: &[HawkesSegment],
    kind: QrEventKind,
    interval_seconds: f64,
) -> MMResult<Vec<u32>> {
    if !interval_seconds.is_finite() || interval_seconds <= 0.0 {
        return invalid(format!("interval {interval_seconds}"));
    }
    let mut out = Vec::new();
    for segment in segments {
        let whole = (segment.duration_seconds / interval_seconds).floor();
        if whole > MAX_INTERVALS {
            return invalid(format!("{whole} intervals in a segment"));
        }
        let bins = rust_decimal::prelude::ToPrimitive::to_usize(&whole).or_overflow("intervals")?;
        let mut counts = vec![0u32; bins];
        for event in segment.events.iter().filter(|e| e.kind == kind) {
            let index = (event.time_seconds / interval_seconds).floor();
            if let Some(cell) =
                rust_decimal::prelude::ToPrimitive::to_usize(&index).and_then(|i| counts.get_mut(i))
            {
                *cell = cell.saturating_add(1);
            }
        }
        out.extend(counts);
    }
    Ok(out)
}

/// A fill measure for aggregated queues.
///
/// At each probe time, a unit of ours joins the back of the queue. Market orders take
/// the queue from the front, and a cancellation removes the share of it ahead of us.
/// The unit fills when a market order finds nothing ahead of it. Returns the share of
/// probes filled within `horizon_seconds` that have that time left. It is coarse and
/// deterministic, for comparing models on one footing, and says nothing about a
/// strategy's profit. `None` without a probe.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a cap out of range, or a horizon or
/// probe spacing that is not positive, [`MMError::InvalidMarketState`] for a segment that
/// breaks the queue rules, and [`MMError::NumericalError`] on a failed conversion.
pub fn fill_rate(
    segments: &[HawkesSegment],
    max_queue: u32,
    horizon_seconds: f64,
    probe_seconds: f64,
) -> MMResult<Option<f64>> {
    let valid = |v: f64| v.is_finite() && v > 0.0;
    if !valid(horizon_seconds) || !valid(probe_seconds) {
        return invalid(format!("horizon {horizon_seconds}, probes {probe_seconds}"));
    }
    if max_queue == 0 || max_queue > MAX_QUEUE {
        return invalid(format!("cap {max_queue}"));
    }
    let cap = usize::try_from(max_queue).map_err(|_| overflow("cap"))?;
    let (mut probes, mut filled) = (0usize, 0usize);
    for segment in segments {
        segment.validate(cap)?;
        if segment.duration_seconds / probe_seconds > MAX_INTERVALS {
            return invalid(format!("probes every {probe_seconds} in a segment"));
        }
        let mut queues = Vec::with_capacity(segment.events.len().saturating_add(1));
        let mut queue = f64::from(segment.initial_queue);
        queues.push(queue);
        for event in &segment.events {
            queue += match event.kind {
                QrEventKind::Limit => 1.0,
                QrEventKind::Cancel | QrEventKind::Market => -1.0,
            };
            queues.push(queue);
        }
        let mut probe = 0.0f64;
        while probe + horizon_seconds <= segment.duration_seconds {
            let start = segment.events.partition_point(|e| e.time_seconds <= probe);
            let mut ahead = queues.get(start).copied().unwrap_or(0.0);
            probes = probes.saturating_add(1);
            for (offset, event) in segment.events.iter().enumerate().skip(start) {
                if event.time_seconds > probe + horizon_seconds {
                    break;
                }
                let size = queues.get(offset).copied().unwrap_or(0.0);
                match event.kind {
                    QrEventKind::Limit => {}
                    QrEventKind::Cancel => {
                        if size > 0.0 {
                            ahead -= ahead / size;
                        }
                    }
                    QrEventKind::Market => {
                        if ahead < 0.5 {
                            filled = filled.saturating_add(1);
                            break;
                        }
                        ahead -= 1.0;
                    }
                }
            }
            probe += probe_seconds;
        }
    }
    if probes == 0 {
        return Ok(None);
    }
    Ok(Some(as_float(filled) / as_float(probes)))
}
