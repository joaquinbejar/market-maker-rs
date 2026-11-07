//! Guéant, Lehalle and Fernandez-Tapia (GLFT) optimal market making.
//!
//! # Reference
//!
//! Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). "Dealing with
//! the inventory risk: a solution to the market making problem."
//! Mathematics and Financial Economics, 7(4), 477-507. arXiv:1105.3115.
//!
//! # Model
//!
//! The reference price follows `dS = σ dW`. Each side is filled at intensity
//! `A exp(-k δ)` where `δ` is the distance of the quote from `S`. The market
//! maker has CARA utility with risk aversion `γ` on the P&L at the terminal
//! time `T`, and the inventory is bounded to `-Q..=Q` lots: at `q = Q` no bid
//! is posted and at `q = -Q` no ask is posted.
//!
//! Proposition 2 and Theorem 1 of the paper give the exact solution:
//!
//! ```text
//! α = k γ σ² / 2
//! η = A (1 + γ/k)^(-(1 + k/γ))
//! M = tridiagonal on q = -Q..Q, M_qq = α q², M_q,q±1 = -η
//! v(t) = exp(-M (T - t)) · (1, ..., 1)'
//! δ_b(t, q) = (1/k) ln(v_q / v_{q+1}) + (1/γ) ln(1 + γ/k)     q ≠  Q
//! δ_a(t, q) = (1/k) ln(v_q / v_{q-1}) + (1/γ) ln(1 + γ/k)     q ≠ -Q
//! bid = S - δ_b, ask = S + δ_a
//! ```
//!
//! [`GLFTStrategy::closed_form_quotes`] gives the asymptotic approximation
//! after Proposition 3, valid when `T - t` is large:
//!
//! ```text
//! C   = sqrt(σ² γ / (2 k A) · (1 + γ/k)^(1 + k/γ))
//! δ_b ≈ (1/γ) ln(1 + γ/k) + (2q + 1)/2 · C
//! δ_a ≈ (1/γ) ln(1 + γ/k) - (2q - 1)/2 · C
//! ```
//!
//! # Units
//!
//! - Prices, `δ` and `min_spread` are in the quote currency; `γ` and `k` are
//!   per unit of price.
//! - `volatility` is the absolute volatility of the price per √year, with a
//!   365 day year (`Y = 31 536 000 s`), as in the Avellaneda-Stoikov module.
//! - `A` is in fills per second; `terminal_time` and `current_time` are in
//!   milliseconds.
//! - The model trades one lot per fill. Inventory is given in base units and
//!   divided by `lot_size` to obtain the lot count `q`. Rescaling prices to
//!   one lot (`S' = L S`, `σ' = L σ`, `k' = k / L`) turns every formula above
//!   into the same formula with `γ` replaced by `γ_L = γ · L`, where `L` is
//!   `lot_size`. That substitution is applied before computing anything.
//!
//! Dimensional check, with `τ` the time to `T` in seconds:
//!
//! ```text
//! α [1/s] = k [1/price] · γ_L [1/price] · σ² [price²/year] / (2 · Y [s/year])
//! η [1/s] = A [1/s] · (dimensionless)
//! M τ     = dimensionless
//! ```
//!
//! # Numerics
//!
//! Only ratios of `v` enter the quotes, so the computation evaluates
//! `w = exp(-(M - c I) τ) · 1` with the Gershgorin bound `c = -2η ≤ λ_min(M)`.
//! `M - c I` is positive semidefinite, so `w` never grows and nothing
//! overflows at long horizons; the factor `exp(c τ)` cancels in every ratio.
//! The exponential is applied to the vector directly: `τ` is split into `N`
//! steps so that each step has infinity norm at most 2, each step is a Taylor
//! series of the tridiagonal operator applied to the vector, and the vector
//! is renormalized to a maximum of 1 after every step. With a large `Q` and
//! a stiff `α` the entries near `±Q` can fall below the `f64` range relative
//! to the centre; when an entry drops below `1e-200` of the largest one,
//! every entry is moved into a per entry log offset (`v_i ∝ w_i exp(l_i)`,
//! `w_i = 1`) and the remaining steps use the similar operator `D⁻¹ B D`,
//! `D = diag(exp(l))`, whose exponential is the same up to that scaling.
//! When no entry reaches the floor the offsets stay zero and the
//! computation is the plain one. This is scaling and
//! squaring where the squaring is replaced by repeated application to the
//! vector: it never forms the dense exponential (`(2Q + 1)²` entries, `O(n³)`
//! per squaring), it costs `O(N · n)` and it does not allocate.
//!
//! # f64 boundary
//!
//! `exp` and `ln` are evaluated in `f64` in the private helpers below. The
//! inputs `γ`, `k`, `A`, `σ` and `lot_size` are converted from `Decimal`
//! through a checked path; every intermediate that a quote depends on is
//! checked finite and every `v` entry used is checked positive. Any failure
//! is a [`MMError::NumericalError`], never a saturated or substituted value.
//! The two distances `δ_b` and `δ_a` are converted back to `Decimal` through
//! a checked conversion and the quotes are built with checked `Decimal`
//! arithmetic. The validation against an independent reference solution
//! (eigendecomposition and dense scaling and squaring) is in
//! `tests/glft_reference.rs`.

use crate::Decimal;
use crate::types::error::{MMError, MMResult};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Largest inventory bound `Q` accepted by [`GLFTConfig::new`], in lots.
///
/// It bounds the size of the generator to `2 Q + 1 = 401` rows, and
/// therefore the cost of one quote.
pub const MAX_INVENTORY_LOTS: u32 = 200;

/// Size of the stack buffers: `2 * MAX_INVENTORY_LOTS + 1`.
const MAX_DIM: usize = 401;

/// Seconds in the 365 day year used for the volatility.
const SECONDS_PER_YEAR: f64 = 31_536_000.0;

/// Milliseconds per second, as a `Decimal` divisor.
const MILLISECONDS_PER_SECOND: Decimal = Decimal::from_parts(1000, 0, 0, false, 0);

/// Largest infinity norm of one step of the matrix exponential.
const STEP_NORM: f64 = 2.0;

/// Largest number of steps of the matrix exponential in one quote.
/// `N ≈ τ (α Q² + 4η) / 2`, and up to about twice that after a rebalance;
/// beyond the cap the horizon is too stiff for the inventory bound and the
/// quote is a [`MMError::NumericalError`] that points to
/// [`GLFTStrategy::closed_form_quotes`]. See [`GLFTStrategy`] for the cost.
const MAX_EXPM_STEPS: u32 = 11_000;

/// An entry of the normalized value vector below this fraction of the
/// largest one triggers a rebalance into the per entry log offsets.
const REBALANCE_FLOOR: f64 = 1e-200;

/// Largest order of the Taylor series of one step.
const MAX_TAYLOR_ORDER: u32 = 100;

/// Size of the last Taylor term, relative to each entry, at which the series stops.
const TAYLOR_TOL: f64 = 1e-16;

/// Configuration of the GLFT strategy.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::glft::GLFTConfig;
/// use market_maker_rs::dec;
///
/// let config = GLFTConfig::new(
///     dec!(0.01),  // risk_aversion γ, per unit of price
///     dec!(0.3),   // order_intensity_k, per unit of price
///     dec!(0.9),   // intensity_scale_a, fills per second
///     30,          // max_inventory_lots Q
///     dec!(1),     // lot_size, base quantity per fill
///     600_000,     // terminal_time, ms
///     dec!(0),     // min_spread
/// ).unwrap();
/// assert_eq!(config.max_inventory_lots, 30);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct GLFTConfig {
    /// Risk aversion `γ`, per unit of price. Positive.
    pub risk_aversion: Decimal,

    /// Decay `k` of the fill intensity `A exp(-k δ)`, per unit of price. Positive.
    pub order_intensity_k: Decimal,

    /// Scale `A` of the fill intensity, in fills per second. Positive.
    pub intensity_scale_a: Decimal,

    /// Inventory bound `Q` in lots, `1..=MAX_INVENTORY_LOTS`.
    pub max_inventory_lots: u32,

    /// Base quantity of one lot (one fill in the model). Positive.
    pub lot_size: Decimal,

    /// Terminal time `T` in milliseconds from session start. Positive.
    pub terminal_time: u64,

    /// Smallest spread `ask - bid` quoted when both sides are posted. Not negative.
    pub min_spread: Decimal,
}

impl GLFTConfig {
    /// Creates a validated `GLFTConfig`.
    ///
    /// # Arguments
    ///
    /// * `risk_aversion` - `γ`, per unit of price, positive
    /// * `order_intensity_k` - `k`, per unit of price, positive
    /// * `intensity_scale_a` - `A`, fills per second, positive
    /// * `max_inventory_lots` - `Q`, in `1..=MAX_INVENTORY_LOTS`
    /// * `lot_size` - base quantity of one lot, positive
    /// * `terminal_time` - `T` in milliseconds, positive
    /// * `min_spread` - smallest quoted spread, not negative
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] naming the first invalid
    /// parameter and its value.
    pub fn new(
        risk_aversion: Decimal,
        order_intensity_k: Decimal,
        intensity_scale_a: Decimal,
        max_inventory_lots: u32,
        lot_size: Decimal,
        terminal_time: u64,
        min_spread: Decimal,
    ) -> MMResult<Self> {
        let config = Self {
            risk_aversion,
            order_intensity_k,
            intensity_scale_a,
            max_inventory_lots,
            lot_size,
            terminal_time,
            min_spread,
        };
        config.validate()?;
        Ok(config)
    }

    /// Checks the invariants of [`GLFTConfig::new`]. The fields are public,
    /// so every quote calls this again before using them.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] naming the first invalid
    /// parameter and its value.
    pub fn validate(&self) -> MMResult<()> {
        let positive = [
            ("risk_aversion", self.risk_aversion),
            ("order_intensity_k", self.order_intensity_k),
            ("intensity_scale_a", self.intensity_scale_a),
            ("lot_size", self.lot_size),
        ];
        for (name, value) in positive {
            if value <= Decimal::ZERO {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be positive, got {value}"
                )));
            }
        }
        if self.max_inventory_lots == 0 || self.max_inventory_lots > MAX_INVENTORY_LOTS {
            return Err(MMError::InvalidConfiguration(format!(
                "max_inventory_lots must be in 1..={MAX_INVENTORY_LOTS}, got {}",
                self.max_inventory_lots
            )));
        }
        if self.terminal_time == 0 {
            return Err(MMError::InvalidConfiguration(
                "terminal_time must be positive, got 0".to_string(),
            ));
        }
        if self.min_spread < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "min_spread must be non-negative, got {}",
                self.min_spread
            )));
        }
        Ok(())
    }
}

/// GLFT quotes. A side is `None` when the inventory bound forbids it: no bid
/// at `q = Q` and no ask at `q = -Q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GlftQuotes {
    /// Bid price, `None` at the upper inventory bound.
    pub bid: Option<Decimal>,
    /// Ask price, `None` at the lower inventory bound.
    pub ask: Option<Decimal>,
}

/// What [`GLFTStrategy::exact_quotes`] gave.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ExactQuotes {
    /// The quotes of Theorem 1.
    Quotes(GlftQuotes),
    /// The rest of the horizon needs more steps of the matrix exponential than the
    /// 11 000 it allows, so there is no exact quote; [`GLFTStrategy::closed_form_quotes`]
    /// is the approximation the model has for it.
    OverStepCap {
        /// Steps the rest of the horizon needed.
        steps_needed: f64,
        /// Steps that were left.
        budget: u32,
    },
}

/// A failure of the matrix exponential: over its step cap, or another error.
enum Expm {
    OverCap { steps_needed: f64, budget: u32 },
    Failed(MMError),
}

impl From<MMError> for Expm {
    fn from(error: MMError) -> Self {
        Self::Failed(error)
    }
}

/// The error of an exact quote over the step cap.
pub(crate) fn over_cap_error(steps_needed: f64, budget: u32) -> MMError {
    numerical(&format!(
        "horizon too stiff: more than {MAX_EXPM_STEPS} steps of the matrix exponential \
         ({steps_needed} needed for the rest of the horizon, {budget} left); \
         use GLFTStrategy::closed_form_quotes, or reduce max_inventory_lots or terminal_time"
    ))
}

/// GLFT strategy: quotes of Theorem 1 of Guéant, Lehalle and Fernandez-Tapia.
///
/// # Cost
///
/// One step of the matrix exponential on the largest grid (`Q = 200`,
/// 401 entries) costs about 4.4 µs (release build, Apple silicon, measured
/// over full quotes). A quote takes `N ≈ τ (α Q² + 4η) / 2` steps, up to
/// about twice that after a rebalance of a stiff tail, and is capped at
/// 11 000 steps, about 50 ms; beyond the cap it returns
/// [`MMError::NumericalError`] and [`GLFTStrategy::closed_form_quotes`]
/// should be used. With the paper parameters a quote takes about 11 ms at
/// `Q = 200` and `T - t = 600 s`, and well under 1 ms at `Q = 30`.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy};
/// use market_maker_rs::dec;
///
/// let config = GLFTConfig::new(
///     dec!(0.01), dec!(0.3), dec!(0.9), 30, dec!(1), 600_000, dec!(0)
/// ).unwrap();
/// // sigma = 0.3 per sqrt(second), expressed per sqrt(year).
/// let sigma = dec!(1684.7);
///
/// let long = GLFTStrategy::calculate_optimal_quotes(
///     dec!(1000), dec!(5), &config, sigma, 0
/// ).unwrap();
/// let flat = GLFTStrategy::calculate_optimal_quotes(
///     dec!(1000), dec!(0), &config, sigma, 0
/// ).unwrap();
/// // A long inventory lowers both quotes.
/// assert!(long.bid.unwrap() < flat.bid.unwrap());
/// assert!(long.ask.unwrap() < flat.ask.unwrap());
/// ```
pub struct GLFTStrategy;

impl GLFTStrategy {
    /// Optimal quotes of Theorem 1 at `current_time`.
    ///
    /// `inventory` is in base units and must be an integer number of lots
    /// within `±Q` lots. If both sides are posted and `ask - bid` is below
    /// `min_spread`, both quotes move away from the mid by half the
    /// difference. After `terminal_time` the time to `T` is zero.
    ///
    /// # Errors
    ///
    /// - [`MMError::InvalidConfiguration`] if `config` is invalid.
    /// - [`MMError::InvalidMarketState`] if `mid_price` or `volatility` is not
    ///   positive.
    /// - [`MMError::InvalidPositionUpdate`] if `inventory` is not a multiple of
    ///   `lot_size` or is outside `±Q` lots.
    /// - [`MMError::NumericalError`] if a parameter does not convert to a
    ///   finite `f64`, the value vector underflows or is not finite, the
    ///   horizon needs more than 11 000 steps (use
    ///   [`GLFTStrategy::closed_form_quotes`] there), or a result does not
    ///   convert back to `Decimal`.
    /// - [`MMError::InvalidQuoteGeneration`] if the bid is not positive or
    ///   the quotes cross.
    #[must_use = "the quotes are the result of the computation"]
    pub fn calculate_optimal_quotes(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<GlftQuotes> {
        match Self::exact_quotes(mid_price, inventory, config, volatility, current_time)? {
            ExactQuotes::Quotes(quotes) => Ok(quotes),
            ExactQuotes::OverStepCap {
                steps_needed,
                budget,
            } => Err(over_cap_error(steps_needed, budget)),
        }
    }

    /// The quotes of [`GLFTStrategy::calculate_optimal_quotes`], with a horizon over the
    /// matrix exponential's step cap told apart from a failure: it is
    /// [`ExactQuotes::OverStepCap`], so a caller can choose what to do there, such as
    /// [`GLFTStrategy::closed_form_quotes`].
    ///
    /// # Errors
    ///
    /// The errors of [`GLFTStrategy::calculate_optimal_quotes`], except the step cap.
    pub fn exact_quotes(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<ExactQuotes> {
        let (model, q) = Model::new(mid_price, inventory, config, volatility)?;
        let tau_s = seconds_to_terminal(config.terminal_time, current_time)?;
        let mut vv = ValueVector::new();
        match model.value_vector(tau_s, &mut vv) {
            Ok(()) => {}
            Err(Expm::OverCap {
                steps_needed,
                budget,
            }) => {
                return Ok(ExactQuotes::OverStepCap {
                    steps_needed,
                    budget,
                });
            }
            Err(Expm::Failed(error)) => return Err(error),
        }
        let (bid, ask) = model.exact_distances(&vv, q)?;
        build_quotes(mid_price, bid, ask, config.min_spread).map(ExactQuotes::Quotes)
    }

    /// Closed form approximation after Proposition 3, valid for large
    /// `T - t`; it does not depend on the time.
    ///
    /// The same inventory rules, bounds and `min_spread` as
    /// [`GLFTStrategy::calculate_optimal_quotes`] apply.
    ///
    /// # Errors
    ///
    /// The same as [`GLFTStrategy::calculate_optimal_quotes`], except that
    /// no value vector is computed.
    #[must_use = "the quotes are the result of the computation"]
    pub fn closed_form_quotes(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
    ) -> MMResult<GlftQuotes> {
        let (model, q) = Model::new(mid_price, inventory, config, volatility)?;
        let (bid, ask) = model.closed_form_distances(q)?;
        build_quotes(mid_price, bid, ask, config.min_spread)
    }

    /// GLFT quotes next to the Avellaneda-Stoikov quotes for the same
    /// `γ`, `k`, volatility and time to `T`.
    ///
    /// The A-S quotes come from
    /// [`avellaneda_stoikov::calculate_optimal_quotes`](crate::strategy::avellaneda_stoikov::calculate_optimal_quotes)
    /// in the same one lot model: inventory `inventory / lot_size` lots and
    /// risk aversion `γ_L = γ · lot_size`; the paper (footnote 8) obtains them as
    /// the first order expansion of the GLFT quotes near `T`.
    ///
    /// # Returns
    ///
    /// `(glft, (as_bid, as_ask))`.
    ///
    /// # Errors
    ///
    /// Any error of [`GLFTStrategy::calculate_optimal_quotes`] or of the A-S
    /// computation.
    #[must_use = "the quotes are the result of the computation"]
    pub fn compare_with_avellaneda_stoikov(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
        current_time: u64,
    ) -> MMResult<(GlftQuotes, (Decimal, Decimal))> {
        let glft =
            Self::calculate_optimal_quotes(mid_price, inventory, config, volatility, current_time)?;
        // The A-S side sees the same one lot model: q in lots and γ_L = γ L.
        let lots = inventory
            .checked_div(config.lot_size)
            .ok_or_else(|| numerical("inventory / lot_size"))?;
        let gamma_lot = config
            .risk_aversion
            .checked_mul(config.lot_size)
            .ok_or_else(|| numerical("risk_aversion * lot_size"))?;
        let as_quotes = crate::strategy::avellaneda_stoikov::calculate_optimal_quotes(
            mid_price,
            lots,
            gamma_lot,
            volatility,
            config.terminal_time.saturating_sub(current_time),
            config.order_intensity_k,
        )?;
        Ok((glft, as_quotes))
    }
}

#[cold]
#[inline(never)]
fn numerical(what: &str) -> MMError {
    MMError::NumericalError(format!("glft: {what}"))
}

/// Checked `Decimal` to finite `f64`.
fn to_f64(name: &str, value: Decimal) -> MMResult<f64> {
    value
        .to_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| numerical(&format!("{name} {value} does not convert to f64")))
}

/// Checked finite `f64` to `Decimal`.
fn to_decimal(name: &str, value: f64) -> MMResult<Decimal> {
    if !value.is_finite() {
        return Err(numerical(&format!("{name} is not finite: {value}")));
    }
    Decimal::from_f64(value)
        .ok_or_else(|| numerical(&format!("{name} {value} does not convert to Decimal")))
}

/// Checks that an `f64` intermediate is finite and positive.
fn positive_finite(name: &str, value: f64) -> MMResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(numerical(&format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

/// Time to `T` in seconds, zero after `T`.
fn seconds_to_terminal(terminal_time: u64, current_time: u64) -> MMResult<f64> {
    let tau_ms = Decimal::from(terminal_time.saturating_sub(current_time));
    let tau_s = tau_ms
        .checked_div(MILLISECONDS_PER_SECOND)
        .ok_or_else(|| numerical("time to terminal"))?;
    to_f64("time to terminal", tau_s)
}

/// Model parameters in `f64`, per second and per lot.
#[derive(Debug, Clone, Copy)]
struct Model {
    /// `k`, per unit of price.
    k: f64,
    /// `γ_L = γ · lot_size`, per unit of price.
    gamma: f64,
    /// `A`, per second.
    a: f64,
    /// `σ²` per second.
    sigma2: f64,
    /// `α`, per second.
    alpha: f64,
    /// `η`, per second.
    eta: f64,
    /// `(1/γ_L) ln(1 + γ_L/k)`.
    base: f64,
    /// `Q`.
    q_max: u32,
}

impl Model {
    /// Validates the inputs and returns the model and the inventory in lots.
    fn new(
        mid_price: Decimal,
        inventory: Decimal,
        config: &GLFTConfig,
        volatility: Decimal,
    ) -> MMResult<(Self, i64)> {
        config.validate()?;
        if mid_price <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(format!(
                "mid_price must be positive, got {mid_price}"
            )));
        }
        if volatility <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(format!(
                "volatility must be positive, got {volatility}"
            )));
        }
        let q = inventory_lots(inventory, config)?;

        let k = positive_finite("k", to_f64("k", config.order_intensity_k)?)?;
        let gamma_raw = to_f64("risk_aversion", config.risk_aversion)?;
        let lot = to_f64("lot_size", config.lot_size)?;
        let gamma = positive_finite("risk_aversion * lot_size", gamma_raw * lot)?;
        let a = positive_finite("A", to_f64("A", config.intensity_scale_a)?)?;
        let sigma = to_f64("volatility", volatility)?;
        let sigma2 = positive_finite("sigma^2 per second", sigma * sigma / SECONDS_PER_YEAR)?;

        let ratio = gamma / k;
        let log1p_ratio = positive_finite("ln(1 + gamma/k)", ratio.ln_1p())?;
        let alpha = positive_finite("alpha", k * gamma * sigma2 / 2.0)?;
        // η = A (1 + γ/k)^(-(1 + k/γ)) = A exp(-(1 + k/γ) ln(1 + γ/k)).
        let eta = positive_finite("eta", a * (-(1.0 + k / gamma) * log1p_ratio).exp())?;
        let base = positive_finite("base distance", log1p_ratio / gamma)?;
        Ok((
            Self {
                k,
                gamma,
                a,
                sigma2,
                alpha,
                eta,
                base,
                q_max: config.max_inventory_lots,
            },
            q,
        ))
    }

    /// Number of entries `2 Q + 1`.
    fn dim(&self) -> MMResult<usize> {
        let n = self
            .q_max
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| numerical("dimension"))?;
        let n = usize::try_from(n).map_err(|_| numerical("dimension"))?;
        if n > MAX_DIM {
            return Err(numerical("dimension above MAX_DIM"));
        }
        Ok(n)
    }

    /// Diagonal of `M - c I = M + 2η I`: `α q² + 2η` for `q = -Q..=Q`.
    fn fill_shifted_diagonal(&self, diag: &mut [f64]) {
        let mut qf = -f64::from(self.q_max);
        for d in diag.iter_mut() {
            *d = self.alpha * qf * qf + 2.0 * self.eta;
            qf += 1.0;
        }
    }

    /// Per second diagonal of `M - c I` and the step plan for `tau_s`:
    /// `(steps, dt)` with `dt (largest row sum of the scaled operator) <= STEP_NORM`.
    fn plan(
        tau_s: f64,
        diag_s: &[f64],
        eta: f64,
        factors: &Factors<'_>,
        budget: u32,
    ) -> Result<(u32, f64), Expm> {
        let mut norm = 0.0_f64;
        for (i, d) in diag_s.iter().enumerate() {
            let left = factors.left.get(i).copied().unwrap_or(0.0);
            let right = factors.right.get(i).copied().unwrap_or(0.0);
            norm = norm.max(d + eta * (left + right));
        }
        let steps_f = positive_finite("expm norm", tau_s * norm)?;
        let steps_f = (steps_f / STEP_NORM).ceil().max(1.0);
        let steps = Decimal::from_f64(steps_f)
            .and_then(|s| s.to_u32())
            .filter(|s| *s <= budget)
            .ok_or(Expm::OverCap {
                steps_needed: steps_f,
                budget,
            })?;
        Ok((steps, tau_s / f64::from(steps)))
    }

    /// Computes `v = exp(-(M - c I) τ) · 1` up to a positive factor, as
    /// `v_i ∝ w_i exp(l_i)`. The offsets `l` stay zero, and the computation
    /// is the plain normalized vector, unless an entry falls below
    /// `REBALANCE_FLOOR` of the largest one; then every entry is moved into
    /// its offset (`l_i += ln w_i`, `w_i = 1`) and the operator becomes the
    /// diagonal similarity `D⁻¹ B D`, which has the same exponential.
    fn value_vector(&self, tau_s: f64, out: &mut ValueVector) -> Result<(), Expm> {
        let n = self.dim()?;
        out.n = n;
        let (w_all, l_all) = (&mut out.w, &mut out.log_offset);
        let w = w_all
            .get_mut(..n)
            .ok_or_else(|| numerical("value vector size"))?;
        let l = l_all.get_mut(..n).ok_or_else(|| numerical("offset size"))?;
        w.fill(1.0);
        l.fill(0.0);
        if tau_s <= 0.0 {
            return Ok(());
        }

        let mut diag_s_buf = [0.0_f64; MAX_DIM];
        let mut diag_buf = [0.0_f64; MAX_DIM];
        let mut left_buf = [0.0_f64; MAX_DIM];
        let mut right_buf = [0.0_f64; MAX_DIM];
        let mut term_buf = [0.0_f64; MAX_DIM];
        let mut next_buf = [0.0_f64; MAX_DIM];
        let size = || numerical("buffer size");
        let diag_s = diag_s_buf.get_mut(..n).ok_or_else(size)?;
        let diag = diag_buf.get_mut(..n).ok_or_else(size)?;
        let left = left_buf.get_mut(..n).ok_or_else(size)?;
        let right = right_buf.get_mut(..n).ok_or_else(size)?;
        let term = term_buf.get_mut(..n).ok_or_else(size)?;
        let next = next_buf.get_mut(..n).ok_or_else(size)?;
        self.fill_shifted_diagonal(diag_s);
        set_factors(l, left, right)?;

        let mut budget = MAX_EXPM_STEPS;
        let mut remaining = tau_s;
        let (mut steps, mut dt) = Self::plan(
            remaining,
            diag_s,
            self.eta,
            &Factors { left, right },
            budget,
        )?;
        scale_diagonal(diag_s, dt, diag);
        while steps > 0 {
            taylor_step(w, diag, self.eta * dt, left, right, term, next)?;
            // The loop runs only while `steps > 0`.
            steps = steps.saturating_sub(1);
            budget = budget.saturating_sub(1);
            remaining -= dt;
            let max = w.iter().copied().fold(0.0_f64, f64::max);
            let max = positive_finite("value vector maximum", max)?;
            let mut min = f64::INFINITY;
            for x in w.iter_mut() {
                *x /= max;
                min = min.min(*x);
            }
            if steps > 0 && min < REBALANCE_FLOOR {
                for (x, off) in w.iter_mut().zip(l.iter_mut()) {
                    let x_pos = positive_finite("value vector entry", *x)?;
                    *off += x_pos.ln();
                    *x = 1.0;
                }
                set_factors(l, left, right)?;
                let remaining_s = positive_finite("remaining horizon", remaining)?;
                (steps, dt) = Self::plan(
                    remaining_s,
                    diag_s,
                    self.eta,
                    &Factors { left, right },
                    budget,
                )?;
                scale_diagonal(diag_s, dt, diag);
            }
        }
        Ok(())
    }

    /// `(δ_b, δ_a)` of Theorem 1 from the value vector over `-Q..=Q`.
    fn exact_distances(&self, vv: &ValueVector, q: i64) -> MMResult<(Option<f64>, Option<f64>)> {
        let q_max = i64::from(self.q_max);
        // ln v_q up to a common constant: ln w_q, and the offset separately
        // so that a zero offset leaves the plain computation unchanged.
        let at = |q: i64| -> MMResult<(f64, f64)> {
            let i = q
                .checked_add(q_max)
                .and_then(|i| usize::try_from(i).ok())
                .filter(|i| *i < vv.n)
                .ok_or_else(|| numerical("inventory index"))?;
            let x =
                vv.w.get(i)
                    .copied()
                    .ok_or_else(|| numerical("inventory index"))?;
            let off = vv
                .log_offset
                .get(i)
                .copied()
                .ok_or_else(|| numerical("inventory index"))?;
            Ok((positive_finite("value vector entry", x)?.ln(), off))
        };
        let (lw_q, off_q) = at(q)?;
        let distance = |(lw_n, off_n): (f64, f64)| -> MMResult<f64> {
            let d = ((lw_q - lw_n) + (off_q - off_n)) / self.k + self.base;
            if d.is_finite() {
                Ok(d)
            } else {
                Err(numerical(&format!("quote distance is not finite: {d}")))
            }
        };
        let bid = if q < q_max {
            let up = q
                .checked_add(1)
                .ok_or_else(|| numerical("inventory index"))?;
            Some(distance(at(up)?)?)
        } else {
            None
        };
        // `q_max` comes from a `u32`, so its negation fits in an `i64`.
        #[allow(clippy::arithmetic_side_effects)]
        let min_q = -q_max;
        let ask = if q > min_q {
            let down = q
                .checked_sub(1)
                .ok_or_else(|| numerical("inventory index"))?;
            Some(distance(at(down)?)?)
        } else {
            None
        };
        Ok((bid, ask))
    }

    /// `(δ_b, δ_a)` of the closed form after Proposition 3.
    fn closed_form_distances(&self, q: i64) -> MMResult<(Option<f64>, Option<f64>)> {
        let ratio = self.gamma / self.k;
        let growth = ((1.0 + self.k / self.gamma) * ratio.ln_1p()).exp();
        let c = (self.sigma2 * self.gamma / (2.0 * self.k * self.a) * growth).sqrt();
        let c = positive_finite("closed form slope", c)?;
        let q_max = i64::from(self.q_max);
        let qf = to_f64("inventory", Decimal::from(q))?;
        let bid = (q < q_max).then(|| self.base + (2.0 * qf + 1.0) / 2.0 * c);
        // `q_max` comes from a `u32`, so its negation fits in an `i64`.
        #[allow(clippy::arithmetic_side_effects)]
        let min_q = -q_max;
        let ask = (q > min_q).then(|| self.base - (2.0 * qf - 1.0) / 2.0 * c);
        for d in [bid, ask].into_iter().flatten() {
            if !d.is_finite() {
                return Err(numerical(&format!("quote distance is not finite: {d}")));
            }
        }
        Ok((bid, ask))
    }
}

/// Value vector `v_i ∝ w_i exp(log_offset_i)` over the first `n` entries.
struct ValueVector {
    w: [f64; MAX_DIM],
    log_offset: [f64; MAX_DIM],
    n: usize,
}

impl ValueVector {
    fn new() -> Self {
        Self {
            w: [0.0; MAX_DIM],
            log_offset: [0.0; MAX_DIM],
            n: 0,
        }
    }
}

/// Off diagonal factors of the scaled operator `D⁻¹ B D`:
/// `left_i = exp(l_{i-1} - l_i)` and `right_i = exp(l_{i+1} - l_i)`, zero
/// past the ends.
struct Factors<'a> {
    left: &'a [f64],
    right: &'a [f64],
}

fn set_factors(l: &[f64], left: &mut [f64], right: &mut [f64]) -> MMResult<()> {
    for (i, (lf, rf)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let li = l.get(i).copied().unwrap_or(0.0);
        let factor = |j: Option<usize>| -> MMResult<f64> {
            match j.and_then(|j| l.get(j)) {
                Some(lj) => {
                    let f = (lj - li).exp();
                    if f.is_finite() {
                        Ok(f)
                    } else {
                        Err(numerical("scaled operator factor is not finite"))
                    }
                }
                None => Ok(0.0),
            }
        };
        *lf = factor(i.checked_sub(1))?;
        *rf = factor(i.checked_add(1))?;
    }
    Ok(())
}

fn scale_diagonal(diag_s: &[f64], dt: f64, diag: &mut [f64]) {
    for (d, ds) in diag.iter_mut().zip(diag_s) {
        *d = ds * dt;
    }
}

/// One Taylor step `w ← exp(B) w` with
/// `(B x)_i = eta_dt (left_i x_{i-1} + right_i x_{i+1}) - diag_i x_i`.
fn taylor_step(
    v: &mut [f64],
    diag: &[f64],
    eta_dt: f64,
    left: &[f64],
    right: &[f64],
    term: &mut [f64],
    next: &mut [f64],
) -> MMResult<()> {
    term.copy_from_slice(v);
    for order in 1..=MAX_TAYLOR_ORDER {
        let inv = 1.0 / f64::from(order);
        apply_operator(term, diag, eta_dt, left, right, next);
        // Converged when every entry's last term is small relative to that
        // entry, so small tail entries are resolved as well as the largest.
        let mut converged = true;
        let mut finite = true;
        for ((t, x), s) in term.iter_mut().zip(next.iter()).zip(v.iter_mut()) {
            *t = x * inv;
            *s += *t;
            finite &= t.is_finite() && s.is_finite();
            converged &= t.abs() <= TAYLOR_TOL * s.abs();
        }
        if !finite {
            return Err(numerical("taylor series is not finite"));
        }
        if converged {
            return if v.iter().all(|x| *x > 0.0) {
                Ok(())
            } else {
                Err(numerical("taylor step produced a non positive entry"))
            };
        }
    }
    Err(numerical(&format!(
        "taylor series did not converge in {MAX_TAYLOR_ORDER} terms"
    )))
}

/// `out = B x` for the tridiagonal scaled operator of [`taylor_step`].
fn apply_operator(
    x: &[f64],
    diag: &[f64],
    eta_dt: f64,
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
) {
    for (i, ((o, d), (lf, rf))) in out
        .iter_mut()
        .zip(diag)
        .zip(left.iter().zip(right))
        .enumerate()
    {
        let xl = i
            .checked_sub(1)
            .and_then(|j| x.get(j))
            .copied()
            .unwrap_or(0.0);
        let xr = i
            .checked_add(1)
            .and_then(|j| x.get(j))
            .copied()
            .unwrap_or(0.0);
        let centre = x.get(i).copied().unwrap_or(0.0);
        *o = eta_dt * (lf * xl + rf * xr) - d * centre;
    }
}

/// Inventory in lots, checked on the lot grid and within `±Q`.
fn inventory_lots(inventory: Decimal, config: &GLFTConfig) -> MMResult<i64> {
    let off_grid = inventory
        .checked_rem(config.lot_size)
        .is_none_or(|r| !r.is_zero());
    let lots = inventory
        .checked_div(config.lot_size)
        .ok_or_else(|| numerical("inventory / lot_size"))?;
    // The quotient is rounded to 28 digits, so check that it multiplies
    // back to the inventory exactly.
    let exact = lots.fract().is_zero()
        && lots
            .checked_mul(config.lot_size)
            .is_some_and(|back| back == inventory);
    if off_grid || !exact {
        return Err(MMError::InvalidPositionUpdate(format!(
            "inventory {inventory} is not a multiple of lot_size {}",
            config.lot_size
        )));
    }
    let q_max = i64::from(config.max_inventory_lots);
    match lots.to_i64() {
        Some(q) if q.unsigned_abs() <= u64::from(config.max_inventory_lots) => Ok(q),
        Some(_) | None => Err(MMError::InvalidPositionUpdate(format!(
            "inventory {inventory} is {lots} lots, outside the bound of {q_max} lots"
        ))),
    }
}

/// Builds the quotes from the distances, applies `min_spread` and checks them.
fn build_quotes(
    mid_price: Decimal,
    bid_distance: Option<f64>,
    ask_distance: Option<f64>,
    min_spread: Decimal,
) -> MMResult<GlftQuotes> {
    let overflow = || numerical("quote overflows Decimal");
    let mut bid = match bid_distance {
        Some(d) => Some(
            mid_price
                .checked_sub(to_decimal("bid distance", d)?)
                .ok_or_else(overflow)?,
        ),
        None => None,
    };
    let mut ask = match ask_distance {
        Some(d) => Some(
            mid_price
                .checked_add(to_decimal("ask distance", d)?)
                .ok_or_else(overflow)?,
        ),
        None => None,
    };
    if let (Some(b), Some(a)) = (bid, ask) {
        let spread = a.checked_sub(b).ok_or_else(overflow)?;
        if spread < min_spread {
            let half = min_spread
                .checked_sub(spread)
                .and_then(|x| x.checked_div(Decimal::TWO))
                .ok_or_else(overflow)?;
            bid = Some(b.checked_sub(half).ok_or_else(overflow)?);
            ask = Some(a.checked_add(half).ok_or_else(overflow)?);
        }
    }
    if let Some(b) = bid
        && b <= Decimal::ZERO
    {
        return Err(MMError::InvalidQuoteGeneration(format!(
            "bid price must be positive, got {b}"
        )));
    }
    if let Some(a) = ask
        && a <= Decimal::ZERO
    {
        return Err(MMError::InvalidQuoteGeneration(format!(
            "ask price must be positive, got {a}"
        )));
    }
    if let (Some(b), Some(a)) = (bid, ask)
        && b >= a
    {
        return Err(MMError::InvalidQuoteGeneration(format!(
            "bid {b} must be below ask {a}"
        )));
    }
    Ok(GlftQuotes { bid, ask })
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
    use crate::dec;

    /// σ = 0.3 per √s expressed per √year: 0.3 · √31 536 000.
    fn paper_sigma() -> Decimal {
        Decimal::from_f64(0.3 * SECONDS_PER_YEAR.sqrt()).unwrap()
    }

    fn paper_config(q_max: u32) -> GLFTConfig {
        GLFTConfig::new(
            dec!(0.01),
            dec!(0.3),
            dec!(0.9),
            q_max,
            dec!(1),
            600_000,
            dec!(0),
        )
        .unwrap()
    }

    fn distances(quotes: GlftQuotes, mid: Decimal) -> (Decimal, Decimal) {
        (mid - quotes.bid.unwrap(), quotes.ask.unwrap() - mid)
    }

    #[test]
    fn test_max_dim_matches_max_inventory_lots() {
        assert_eq!(
            MAX_DIM,
            usize::try_from(2 * MAX_INVENTORY_LOTS + 1).unwrap()
        );
    }

    #[test]
    fn test_config_new_paper_parameters_ok() {
        let config = paper_config(30);
        assert_eq!(config.max_inventory_lots, 30);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_new_invalid_parameters_return_invalid_configuration() {
        let ok = (
            dec!(0.01),
            dec!(0.3),
            dec!(0.9),
            30,
            dec!(1),
            600_000,
            dec!(0),
        );
        let cases = [
            (dec!(0), ok.1, ok.2, ok.3, ok.4, ok.5, ok.6),
            (dec!(-1), ok.1, ok.2, ok.3, ok.4, ok.5, ok.6),
            (ok.0, dec!(0), ok.2, ok.3, ok.4, ok.5, ok.6),
            (ok.0, ok.1, dec!(0), ok.3, ok.4, ok.5, ok.6),
            (ok.0, ok.1, ok.2, 0, ok.4, ok.5, ok.6),
            (ok.0, ok.1, ok.2, MAX_INVENTORY_LOTS + 1, ok.4, ok.5, ok.6),
            (ok.0, ok.1, ok.2, ok.3, dec!(0), ok.5, ok.6),
            (ok.0, ok.1, ok.2, ok.3, ok.4, 0, ok.6),
            (ok.0, ok.1, ok.2, ok.3, ok.4, ok.5, dec!(-0.1)),
        ];
        for (g, k, a, q, lot, t, ms) in cases {
            let result = GLFTConfig::new(g, k, a, q, lot, t, ms);
            assert!(
                matches!(result, Err(MMError::InvalidConfiguration(_))),
                "{result:?}"
            );
        }
        assert!(GLFTConfig::new(ok.0, ok.1, ok.2, MAX_INVENTORY_LOTS, ok.4, ok.5, ok.6).is_ok());
    }

    #[test]
    fn test_calculate_optimal_quotes_tampered_config_returns_invalid_configuration() {
        let mut config = paper_config(30);
        config.max_inventory_lots = MAX_INVENTORY_LOTS + 1;
        let result =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(0), &config, paper_sigma(), 0);
        assert!(matches!(result, Err(MMError::InvalidConfiguration(_))));
    }

    #[test]
    fn test_calculate_optimal_quotes_invalid_market_returns_invalid_market_state() {
        let config = paper_config(30);
        for (mid, sigma) in [(dec!(0), paper_sigma()), (dec!(1000), dec!(0))] {
            let result = GLFTStrategy::calculate_optimal_quotes(mid, dec!(0), &config, sigma, 0);
            assert!(matches!(result, Err(MMError::InvalidMarketState(_))));
        }
    }

    #[test]
    fn test_calculate_optimal_quotes_inventory_off_grid_or_out_of_bound_returns_error() {
        let mut config = paper_config(30);
        config.lot_size = dec!(0.5);
        for inventory in [dec!(0.25), dec!(15.5), dec!(-15.5), dec!(1.1)] {
            let result = GLFTStrategy::calculate_optimal_quotes(
                dec!(1000),
                inventory,
                &config,
                paper_sigma(),
                0,
            );
            assert!(
                matches!(result, Err(MMError::InvalidPositionUpdate(_))),
                "{inventory}: {result:?}"
            );
        }
        // 15 base units = 30 lots of 0.5: at the bound, valid.
        assert!(
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(15), &config, paper_sigma(), 0)
                .is_ok()
        );
    }

    #[test]
    fn test_calculate_optimal_quotes_at_bounds_omit_one_side() {
        let config = paper_config(8);
        let top =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(8), &config, paper_sigma(), 0)
                .unwrap();
        assert!(top.bid.is_none() && top.ask.is_some());
        let bottom =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(-8), &config, paper_sigma(), 0)
                .unwrap();
        assert!(bottom.ask.is_none() && bottom.bid.is_some());
        let closed =
            GLFTStrategy::closed_form_quotes(dec!(1000), dec!(8), &config, paper_sigma()).unwrap();
        assert!(closed.bid.is_none() && closed.ask.is_some());
    }

    #[test]
    fn test_calculate_optimal_quotes_zero_inventory_symmetric() {
        let config = paper_config(30);
        let quotes =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(0), &config, paper_sigma(), 0)
                .unwrap();
        let (b, a) = distances(quotes, dec!(1000));
        assert!((b - a).abs() < dec!(0.000000001), "{b} vs {a}");
    }

    #[test]
    fn test_calculate_optimal_quotes_skew_grows_with_inventory() {
        let config = paper_config(30);
        let mut previous: Option<(Decimal, Decimal)> = None;
        for q in -5..=5 {
            let quotes = GLFTStrategy::calculate_optimal_quotes(
                dec!(1000),
                Decimal::from(q),
                &config,
                paper_sigma(),
                0,
            )
            .unwrap();
            let (bid, ask) = (quotes.bid.unwrap(), quotes.ask.unwrap());
            if let Some((pb, pa)) = previous {
                assert!(bid < pb && ask < pa, "q={q}");
            }
            previous = Some((bid, ask));
        }
    }

    #[test]
    fn test_calculate_optimal_quotes_terminal_time_returns_base_distance() {
        let config = paper_config(16);
        let base = Decimal::from_f64((1.0_f64 + 0.01 / 0.3).ln() / 0.01).unwrap();
        for q in [-5, 0, 5] {
            let quotes = GLFTStrategy::calculate_optimal_quotes(
                dec!(1000),
                Decimal::from(q),
                &config,
                paper_sigma(),
                600_000,
            )
            .unwrap();
            let (b, a) = distances(quotes, dec!(1000));
            assert!((b - base).abs() < dec!(0.000000001), "{b}");
            assert!((a - base).abs() < dec!(0.000000001), "{a}");
        }
    }

    #[test]
    fn test_closed_form_quotes_large_horizon_match_exact() {
        let mut config = paper_config(30);
        config.terminal_time = 3_600_000;
        for q in [-3, 0, 2] {
            let inventory = Decimal::from(q);
            let exact = GLFTStrategy::calculate_optimal_quotes(
                dec!(1000),
                inventory,
                &config,
                paper_sigma(),
                0,
            )
            .unwrap();
            let closed =
                GLFTStrategy::closed_form_quotes(dec!(1000), inventory, &config, paper_sigma())
                    .unwrap();
            let (eb, ea) = distances(exact, dec!(1000));
            let (cb, ca) = distances(closed, dec!(1000));
            assert!((eb - cb).abs() < dec!(0.05), "q={q}: {eb} vs {cb}");
            assert!((ea - ca).abs() < dec!(0.05), "q={q}: {ea} vs {ca}");
        }
    }

    #[test]
    fn test_calculate_optimal_quotes_min_spread_widens_symmetrically() {
        let mut config = paper_config(30);
        let raw =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(2), &config, paper_sigma(), 0)
                .unwrap();
        let raw_spread = raw.ask.unwrap() - raw.bid.unwrap();
        config.min_spread = raw_spread + dec!(2);
        let wide =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(2), &config, paper_sigma(), 0)
                .unwrap();
        assert_eq!(wide.bid.unwrap(), raw.bid.unwrap() - dec!(1));
        assert_eq!(wide.ask.unwrap(), raw.ask.unwrap() + dec!(1));
    }

    #[test]
    fn test_calculate_optimal_quotes_extreme_parameters_return_numerical_error() {
        // Volatility so large that the horizon is far too stiff.
        let config = paper_config(MAX_INVENTORY_LOTS);
        let result = GLFTStrategy::calculate_optimal_quotes(
            dec!(1000),
            dec!(0),
            &config,
            dec!(10000000000),
            0,
        );
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        // gamma * lot_size overflows f64.
        let mut config = paper_config(1);
        config.risk_aversion = Decimal::MAX;
        config.lot_size = Decimal::MAX;
        let result =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(0), &config, paper_sigma(), 0);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        // A tiny k with a large gamma makes eta underflow to zero.
        let mut config = paper_config(1);
        config.order_intensity_k = dec!(0.0000000001);
        config.risk_aversion = dec!(1000);
        let result =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(0), &config, paper_sigma(), 0);
        assert!(result.is_err(), "{result:?}");
    }

    #[test]
    fn test_compare_with_avellaneda_stoikov_near_terminal_agree() {
        let config = paper_config(16);
        let (glft, (as_bid, as_ask)) = GLFTStrategy::compare_with_avellaneda_stoikov(
            dec!(1000),
            dec!(2),
            &config,
            paper_sigma(),
            599_990,
        )
        .unwrap();
        assert!((glft.bid.unwrap() - as_bid).abs() < dec!(0.001));
        assert!((glft.ask.unwrap() - as_ask).abs() < dec!(0.001));
    }

    #[test]
    fn test_calculate_optimal_quotes_lot_size_rescales_gamma() {
        // gamma_L = gamma * lot_size enters the model: gamma 0.02 with lots of
        // 0.5 is gamma 0.01 with lots of 1, at the same number of lots.
        let a = paper_config(30);
        let mut b = paper_config(30);
        b.risk_aversion = dec!(0.02);
        b.lot_size = dec!(0.5);
        let qa = GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(3), &a, paper_sigma(), 0)
            .unwrap();
        let qb =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(1.5), &b, paper_sigma(), 0)
                .unwrap();
        assert_eq!(qa, qb);
    }

    #[test]
    fn test_quotes_inventory_i64_min_returns_invalid_position_update() {
        let config = paper_config(30);
        let inventory = Decimal::from(i64::MIN);
        let exact = GLFTStrategy::calculate_optimal_quotes(
            dec!(1000),
            inventory,
            &config,
            paper_sigma(),
            0,
        );
        assert!(
            matches!(exact, Err(MMError::InvalidPositionUpdate(_))),
            "{exact:?}"
        );
        let closed =
            GLFTStrategy::closed_form_quotes(dec!(1000), inventory, &config, paper_sigma());
        assert!(
            matches!(closed, Err(MMError::InvalidPositionUpdate(_))),
            "{closed:?}"
        );
    }

    #[test]
    fn test_calculate_optimal_quotes_tiny_off_grid_inventory_returns_invalid_position_update() {
        // 1e-28 / 1e28 rounds to 0 lots, but 1e-28 is off the lot grid.
        let config = GLFTConfig::new(
            dec!(0.0000000000000000000000000001),
            dec!(0.3),
            dec!(0.9),
            2,
            dec!(10000000000000000000000000000),
            1,
            dec!(0),
        )
        .unwrap();
        let inventory = dec!(0.0000000000000000000000000001);
        let exact =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), inventory, &config, dec!(100), 1);
        assert!(
            matches!(exact, Err(MMError::InvalidPositionUpdate(_))),
            "{exact:?}"
        );
        let closed = GLFTStrategy::closed_form_quotes(dec!(1000), inventory, &config, dec!(100));
        assert!(
            matches!(closed, Err(MMError::InvalidPositionUpdate(_))),
            "{closed:?}"
        );
    }

    #[test]
    fn test_compare_with_avellaneda_stoikov_overflow_returns_numerical_error() {
        // q gamma sigma^2 overflows Decimal on the A-S side.
        let config = GLFTConfig::new(
            dec!(1),
            dec!(0.3),
            dec!(0.9),
            MAX_INVENTORY_LOTS,
            dec!(1),
            1,
            dec!(0),
        )
        .unwrap();
        let result = GLFTStrategy::compare_with_avellaneda_stoikov(
            dec!(1000),
            dec!(200),
            &config,
            dec!(100000000000000),
            1,
        );
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_compare_with_avellaneda_stoikov_huge_gamma_at_terminal_returns_numerical_error() {
        // gamma = 1e28, sigma = 10: gamma sigma^2 overflows Decimal on the
        // A-S side even with a zero horizon; it must be an error, not a panic.
        let config = GLFTConfig::new(
            dec!(10000000000000000000000000000),
            dec!(1),
            dec!(1),
            1,
            dec!(1),
            1,
            dec!(1),
        )
        .unwrap();
        let result = GLFTStrategy::compare_with_avellaneda_stoikov(
            dec!(1000),
            dec!(0),
            &config,
            dec!(10),
            1,
        );
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
    }

    #[test]
    fn test_calculate_optimal_quotes_tiny_coupling_tail_matches_two_state_solution() {
        // Q = 1, alpha = 0.5 /s, eta = 2.5e-29 /s, tau = 100 s: v_{+-1} is
        // about e^-50 of v_0, far below the largest Taylor term. With
        // v_0 = 1 + O(eta / alpha) and
        // v_{+-1} = e^{-alpha tau} + (eta / alpha)(1 - e^{-alpha tau}) + O(eta^2 tau^2),
        // the quotes are known to far better than 1e-9.
        let config = GLFTConfig::new(
            dec!(1),
            dec!(1),
            dec!(0.0000000000000000000000000001),
            1,
            dec!(1),
            100_000,
            dec!(0),
        )
        .unwrap();
        let sigma = Decimal::from_f64(SECONDS_PER_YEAR.sqrt()).unwrap();
        let (alpha, tau) = (0.5_f64, 100.0_f64);
        let eta = 1e-28 * 0.25;
        let decay = (-alpha * tau).exp();
        let tail = decay + eta / alpha * (1.0 - decay);
        let base = 2.0_f64.ln();
        let mid = dec!(1000);
        let flat = GLFTStrategy::calculate_optimal_quotes(mid, dec!(0), &config, sigma, 0).unwrap();
        let (b, a) = distances(flat, mid);
        let expected = -tail.ln() + base;
        for d in [b, a] {
            let d = d.to_f64().unwrap();
            assert!((d - expected).abs() < 1e-9, "{d} vs {expected}");
        }
        let top = GLFTStrategy::calculate_optimal_quotes(mid, dec!(1), &config, sigma, 0).unwrap();
        assert!(top.bid.is_none());
        let ask = (top.ask.unwrap() - mid).to_f64().unwrap();
        assert!((ask - (tail.ln() + base)).abs() < 1e-9, "{ask}");
    }

    #[test]
    fn test_compare_with_avellaneda_stoikov_half_lots_near_terminal_agree() {
        // gamma 0.02 per unit with lots of 0.5: gamma_L = 0.01, 1.5 = 3 lots.
        let mut config = paper_config(16);
        config.risk_aversion = dec!(0.02);
        config.lot_size = dec!(0.5);
        let (glft, (as_bid, as_ask)) = GLFTStrategy::compare_with_avellaneda_stoikov(
            dec!(1000),
            dec!(1.5),
            &config,
            paper_sigma(),
            599_990,
        )
        .unwrap();
        assert!(
            (glft.bid.unwrap() - as_bid).abs() < dec!(0.001),
            "{glft:?} {as_bid}"
        );
        assert!(
            (glft.ask.unwrap() - as_ask).abs() < dec!(0.001),
            "{glft:?} {as_ask}"
        );
    }

    /// σ = 2 tick/√s with Q = 200: without rebalancing `v_{±Q}` is about
    /// `e^-930` of `v_0` and underflows.
    fn stiff_config() -> (GLFTConfig, Decimal) {
        let mut config = paper_config(MAX_INVENTORY_LOTS);
        config.terminal_time = 40_000;
        (
            config,
            Decimal::from_f64(2.0 * SECONDS_PER_YEAR.sqrt()).unwrap(),
        )
    }

    #[test]
    fn test_calculate_optimal_quotes_stiff_tail_near_bound_is_finite_and_symmetric() {
        let (config, sigma) = stiff_config();
        let q_max = i64::from(MAX_INVENTORY_LOTS);
        let quote = |q: i64| {
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), Decimal::from(q), &config, sigma, 0)
                .unwrap()
        };
        let top = quote(q_max);
        let below_top = quote(q_max - 1);
        let bottom = quote(-q_max);
        let above_bottom = quote(-(q_max - 1));
        assert!(top.bid.is_none() && bottom.ask.is_none());
        let mid = dec!(1000);
        // Symmetry q -> -q swaps the sides.
        assert_eq!(top.ask.unwrap() - mid, mid - bottom.bid.unwrap());
        assert_eq!(
            below_top.ask.unwrap() - mid,
            mid - above_bottom.bid.unwrap()
        );
        assert_eq!(
            mid - below_top.bid.unwrap(),
            above_bottom.ask.unwrap() - mid
        );
        // Near +Q the ask is far inside the mid (eager to sell), the bid far out.
        assert!(top.ask.unwrap() < mid);
        assert!(below_top.bid.unwrap() < below_top.ask.unwrap());
        // ln(v_{Q-1} / v_Q) is about ln(d_Q / eta) in the quasi steady tail.
        let model = Model::new(mid, dec!(0), &config, sigma).unwrap().0;
        let qf = f64::from(MAX_INVENTORY_LOTS);
        let expected = ((model.alpha * qf * qf + 2.0 * model.eta) / model.eta).ln() / model.k;
        let ask_distance = (top.ask.unwrap() - mid).to_f64().unwrap() - model.base;
        assert!(
            (ask_distance + expected).abs() < 0.05 * expected,
            "{ask_distance} vs -{expected}"
        );
    }

    #[test]
    fn test_calculate_optimal_quotes_stiff_tail_interior_matches_smaller_bound() {
        // Far from the bound the stiff tail does not change the quotes: the
        // rebalanced Q = 200 quotes match Q = 40, which never rebalances.
        let (config, sigma) = stiff_config();
        let mut small = config.clone();
        small.max_inventory_lots = 40;
        for q in [-3, 0, 5] {
            let inventory = Decimal::from(q);
            let a =
                GLFTStrategy::calculate_optimal_quotes(dec!(1000), inventory, &config, sigma, 0)
                    .unwrap();
            let b = GLFTStrategy::calculate_optimal_quotes(dec!(1000), inventory, &small, sigma, 0)
                .unwrap();
            assert!(
                (a.bid.unwrap() - b.bid.unwrap()).abs() < dec!(0.000000001),
                "q={q}"
            );
            assert!(
                (a.ask.unwrap() - b.ask.unwrap()).abs() < dec!(0.000000001),
                "q={q}"
            );
        }
    }

    #[test]
    fn test_calculate_optimal_quotes_beyond_step_cap_returns_numerical_error_naming_closed_form() {
        let mut config = paper_config(MAX_INVENTORY_LOTS);
        config.terminal_time = 36_000_000;
        let result =
            GLFTStrategy::calculate_optimal_quotes(dec!(1000), dec!(0), &config, paper_sigma(), 0);
        match result {
            Err(MMError::NumericalError(msg)) => {
                assert!(msg.contains("closed_form_quotes"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        assert!(
            GLFTStrategy::closed_form_quotes(dec!(1000), dec!(0), &config, paper_sigma()).is_ok()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_config_serde_round_trip_equal() {
        let config = paper_config(30);
        let json = serde_json::to_string(&config).unwrap();
        let back: GLFTConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }
}
