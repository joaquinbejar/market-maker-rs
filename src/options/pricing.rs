//! European option valuation on a volatility surface.
//!
//! One price function, Black-Scholes-Merton read on a total variance
//! surface,
//!
//! ```text
//! V(S, t) = e^{-r tau} * Black(F, K, w(k, tau)),   F = S e^{(r - q) tau},
//! k = ln(K / F_read),   tau = T - t,
//! ```
//!
//! gives the price and every Greek (Black and Scholes, 1973; Merton, 1973,
//! for the continuous carry `q`; Black, 1976, for the forward form). The
//! surface ([`VolSurface`]) only answers `w` and its derivatives; where it
//! is read is the [`Dynamics`]:
//!
//! - [`Dynamics::StickyMoneyness`]: at the current forward,
//!   `F_read = F`, so the smile moves with the spot and
//!   `dk/dS = -1/S`;
//! - [`Dynamics::StickyStrike`]: at the forward of a fixed reference spot,
//!   `F_read = s_ref e^{(r - q) tau}`, so a strike keeps its volatility when
//!   the spot moves.
//!
//! # Units and conventions
//!
//! - Prices, the strike and the spot are in the quote currency per unit of
//!   underlying (contract multiplier 1; a multiplier belongs to the risk
//!   layer). `tau` is in years of [`PRICING_YEAR_DAYS`] days; `rate` and
//!   `carry` are continuously compounded annual rates.
//! - `delta = dV/dS` and `gamma = d2V/dS2` (per unit of spot squared) are
//!   total derivatives under the dynamics: gamma includes the vanna and
//!   volga terms and the curvature `w_kk` of the smile.
//! - `vega = dV/ds` for a parallel shift `s` of the implied volatility of the
//!   whole surface, per 1.00 of volatility (not per point).
//! - `theta = dV/dt` in calendar time, per year, with the surface held as a
//!   function of `(k, tau)` and the rates constant (so `dk/dtau = -(r - q)`
//!   and `w_tau` comes from the surface; see the node convention in
//!   [`crate::options::surface`]).
//! - At `tau = 0` the surface is not consulted: the price is the payoff,
//!   computed exactly in `Decimal`; delta is the payoff's slope with
//!   `1{S > K}` for a call and `1{S > K} - 1` for a put (so 0 and -1 at
//!   `S = K`); gamma, vega and theta are 0. No classical derivative exists
//!   at the kink and none is claimed.
//!
//! # Numerics
//!
//! The model arithmetic is `f64` (the `options` model exception documented
//! in [`crate::options::surface`]): inputs cross from `Decimal` through
//! checked conversions, every intermediate is checked to be finite, puts use
//! `N(-d)` directly (no `1 - N(d)`), and the normal distribution is the
//! fdlibm `erfc` port in this crate. A value that cannot be computed is a
//! typed error, never a saturated, zero or intrinsic substitute.
//!
//! # Why this is not optionstratlib
//!
//! [`crate::options::OptionsAdapter`] prices with optionstratlib at one
//! implied volatility per option. Total variance surfaces, sticky dynamics
//! and the total Greeks they imply do not exist there, so the pricer is
//! written here; optionstratlib serves as the independent cross-check of
//! the flat-surface case in the tests.

use std::sync::Arc;

use crate::Decimal;
use crate::options::special::{norm_cdf, norm_pdf};
use crate::options::surface::VolSurface;
use crate::types::error::{MMError, MMResult};
use optionstratlib::{OptionStyle, OptionType};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal_macros::dec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Days in the pricing year: `tau = days / PRICING_YEAR_DAYS`.
///
/// The ms helpers of [`crate::options::OptionsAdapter`] use
/// [`crate::options::adapter::ADAPTER_MS_PER_YEAR`] (365.25 days) instead;
/// the two are different conventions for different purposes.
pub const PRICING_YEAR_DAYS: Decimal = dec!(365);

/// Time to expiry in years from days, `days / PRICING_YEAR_DAYS`.
///
/// # Errors
///
/// [`MMError::InvalidTimestamp`] when `days` is negative.
pub fn tau_from_days(days: Decimal) -> MMResult<Decimal> {
    if days < Decimal::ZERO {
        return Err(MMError::InvalidTimestamp(format!(
            "days to expiry must be non-negative, got {days}"
        )));
    }
    days.checked_div(PRICING_YEAR_DAYS).ok_or_else(|| {
        numerical(format!(
            "days to expiry {days} cannot be converted to years"
        ))
    })
}

/// Where the surface is read when the spot moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Dynamics {
    /// The smile moves with the forward: `k = ln(K / F)`.
    StickyMoneyness,
    /// A strike keeps its volatility: `k = ln(K / (s_ref e^{(r - q) tau}))`.
    StickyStrike {
        /// The reference spot the surface is anchored to (positive).
        s_ref: Decimal,
    },
}

impl Dynamics {
    /// Checks the parameters of the dynamics.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] when `s_ref` is not positive.
    pub fn validate(&self) -> MMResult<()> {
        match self {
            Self::StickyMoneyness => Ok(()),
            Self::StickyStrike { s_ref } => {
                if *s_ref <= Decimal::ZERO {
                    Err(MMError::InvalidConfiguration(format!(
                        "sticky strike reference spot must be positive, got {s_ref}"
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// What is needed to value one option, all in the conventions of the
/// module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionInputs {
    /// Spot of the underlying, quote currency per unit (positive).
    pub spot: Decimal,
    /// Strike, quote currency per unit (positive).
    pub strike: Decimal,
    /// Time to expiry in years of [`PRICING_YEAR_DAYS`] days (non-negative).
    pub tau_years: Decimal,
    /// Continuously compounded risk-free rate, per year.
    pub rate: Decimal,
    /// Continuously compounded carry (dividend or foreign rate), per year.
    pub carry: Decimal,
    /// Call or put.
    pub style: OptionStyle,
    /// Exercise and payoff type; only [`OptionType::European`] is priced.
    pub exercise: OptionType,
}

impl OptionInputs {
    /// A European option.
    #[must_use]
    pub fn european(
        spot: Decimal,
        strike: Decimal,
        tau_years: Decimal,
        rate: Decimal,
        carry: Decimal,
        style: OptionStyle,
    ) -> Self {
        Self {
            spot,
            strike,
            tau_years,
            rate,
            carry,
            style,
            exercise: OptionType::European,
        }
    }
}

/// The value and Greeks of one option: a data contract, in the units of the
/// module docs (per unit of underlying; theta per year; vega per 1.00 of
/// volatility; gamma per unit of spot squared). Values are expected model
/// values, not executable prices.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptionValuation {
    /// Model price.
    pub price: Decimal,
    /// `dV/dS`, total under the dynamics.
    pub delta: Decimal,
    /// `d2V/dS2`, total under the dynamics.
    pub gamma: Decimal,
    /// `dV/ds` for a parallel implied volatility shift `s`.
    pub vega: Decimal,
    /// `dV/dt`, calendar time, per year.
    pub theta: Decimal,
}

/// A valuation model for options. It is internal to this crate: other
/// crates consume [`OptionValuation`] as data.
pub trait OptionPricer: Send + Sync {
    /// Values one option.
    ///
    /// # Errors
    ///
    /// The errors of [`value`].
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation>;
}

/// [`value`] on a shared surface with fixed dynamics.
#[derive(Clone)]
pub struct SurfacePricer {
    surface: Arc<dyn VolSurface>,
    dynamics: Dynamics,
}

impl std::fmt::Debug for SurfacePricer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfacePricer")
            .field("dynamics", &self.dynamics)
            .finish_non_exhaustive()
    }
}

impl SurfacePricer {
    /// A pricer on `surface` under `dynamics`.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] when the dynamics are invalid.
    pub fn new(surface: Arc<dyn VolSurface>, dynamics: Dynamics) -> MMResult<Self> {
        dynamics.validate()?;
        Ok(Self { surface, dynamics })
    }

    /// The dynamics of the pricer.
    #[must_use]
    pub fn dynamics(&self) -> Dynamics {
        self.dynamics
    }
}

impl OptionPricer for SurfacePricer {
    fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
        value(inputs, self.surface.as_ref(), self.dynamics)
    }
}

/// Values a European option on `surface` under `dynamics`.
///
/// # Errors
///
/// - [`MMError::UnsupportedContract`] for any exercise other than European.
/// - [`MMError::InvalidMarketState`] for a non-positive spot or strike or a
///   negative time to expiry, and the surface's own errors for its domain.
/// - [`MMError::InvalidConfiguration`] for invalid dynamics.
/// - [`MMError::NumericalError`] when an intermediate or a result is not
///   finite or has no `Decimal` value.
pub fn value(
    inputs: &OptionInputs,
    surface: &dyn VolSurface,
    dynamics: Dynamics,
) -> MMResult<OptionValuation> {
    if !matches!(inputs.exercise, OptionType::European) {
        return Err(MMError::UnsupportedContract {
            reason: format!(
                "the Black-Scholes-Merton pricer values European options only, got {:?}",
                inputs.exercise
            ),
        });
    }
    dynamics.validate()?;
    for (name, v) in [("spot", inputs.spot), ("strike", inputs.strike)] {
        if v <= Decimal::ZERO {
            return Err(invalid(format!("{name} must be positive, got {v}")));
        }
    }
    if inputs.tau_years < Decimal::ZERO {
        return Err(invalid(format!(
            "time to expiry must be non-negative, got {}",
            inputs.tau_years
        )));
    }
    let call = matches!(inputs.style, OptionStyle::Call);
    if inputs.tau_years.is_zero() {
        return at_expiry(inputs.spot, inputs.strike, call);
    }
    let m = Model::new(inputs, surface, dynamics)?;
    m.valuation(call)
}

/// The payoff and its conventions at `tau = 0`, exactly, without the surface.
fn at_expiry(spot: Decimal, strike: Decimal, call: bool) -> MMResult<OptionValuation> {
    let itm_call = spot > strike;
    let price = match (call, itm_call) {
        (true, true) => spot.checked_sub(strike),
        (false, false) => strike.checked_sub(spot),
        _ => Some(Decimal::ZERO),
    }
    .ok_or_else(|| {
        numerical(format!(
            "payoff of spot {spot} and strike {strike} overflows"
        ))
    })?;
    let delta = match (call, itm_call) {
        (true, true) => Decimal::ONE,
        (true, false) | (false, true) => Decimal::ZERO,
        (false, false) => Decimal::NEGATIVE_ONE,
    };
    Ok(OptionValuation {
        price,
        delta,
        gamma: Decimal::ZERO,
        vega: Decimal::ZERO,
        theta: Decimal::ZERO,
    })
}

/// The f64 state of one valuation at `tau > 0`.
struct Model {
    s: f64,
    k_strike: f64,
    r: f64,
    q: f64,
    tau: f64,
    fwd: f64,
    df_r: f64,
    df_q: f64,
    sticky_moneyness: bool,
    sig: f64,
    s1: f64,
    s2: f64,
    w_k: f64,
    w_tau: f64,
    vol: f64,
    d1: f64,
    d2: f64,
}

impl Model {
    fn new(inputs: &OptionInputs, surface: &dyn VolSurface, dynamics: Dynamics) -> MMResult<Self> {
        let s = to_f64("spot", inputs.spot)?;
        let k_strike = to_f64("strike", inputs.strike)?;
        let tau = to_f64("time to expiry", inputs.tau_years)?;
        let r = to_f64("rate", inputs.rate)?;
        let q = to_f64("carry", inputs.carry)?;
        if !(s > 0.0 && k_strike > 0.0 && tau > 0.0) {
            return Err(numerical(
                "spot, strike or time to expiry underflows to zero as f64".to_string(),
            ));
        }
        let growth = finite("(rate - carry) * tau", (r - q) * tau)?;
        let df_r = positive("discount factor e^(-r tau)", (-r * tau).exp())?;
        let df_q = positive("carry factor e^(-q tau)", (-q * tau).exp())?;
        let fwd = positive("forward", s * growth.exp())?;
        // k = ln K - ln F, without forming K / F (which could overflow).
        let ln_k = k_strike.ln();
        let k_current = finite("log moneyness", ln_k - s.ln() - growth)?;
        let (sticky_moneyness, k_read) = match dynamics {
            Dynamics::StickyMoneyness => (true, k_current),
            Dynamics::StickyStrike { s_ref } => {
                let s_ref = to_f64("reference spot", s_ref)?;
                if s_ref <= 0.0 {
                    return Err(numerical(
                        "reference spot underflows to zero as f64".to_string(),
                    ));
                }
                (false, finite("log moneyness", ln_k - s_ref.ln() - growth)?)
            }
        };
        let tv = surface.total_variance(k_read, tau)?;
        let w = tv.w();
        let vol = positive("total volatility sqrt(w)", w.sqrt())?;
        let sig = positive("implied volatility", (w / tau).sqrt())?;
        // sigma = sqrt(w / tau): dsigma/dk = w_k / (2 tau sigma) and
        // d2sigma/dk2 = (w_kk / (2 tau) - (dsigma/dk)^2) / sigma.
        let s1 = finite("dsigma/dk", tv.w_k() / (2.0 * tau * sig))?;
        let s2 = finite("d2sigma/dk2", (tv.w_kk() / (2.0 * tau) - s1 * s1) / sig)?;
        let d1 = finite("d1", (-k_current + 0.5 * w) / vol)?;
        let d2 = finite("d2", d1 - vol)?;
        Ok(Self {
            s,
            k_strike,
            r,
            q,
            tau,
            fwd,
            df_r,
            df_q,
            sticky_moneyness,
            sig,
            s1,
            s2,
            w_k: tv.w_k(),
            w_tau: tv.w_tau(),
            vol,
            d1,
            d2,
        })
    }

    fn valuation(&self, call: bool) -> MMResult<OptionValuation> {
        let (s, k, fwd) = (self.s, self.k_strike, self.fwd);
        let (d1, d2) = (self.d1, self.d2);
        let n1 = norm_pdf(d1);
        let n2 = norm_pdf(d2);
        let (bs_delta, black_f) = if call {
            (self.df_q * norm_cdf(d1), norm_cdf(d1))
        } else {
            (-self.df_q * norm_cdf(-d1), -norm_cdf(-d1))
        };
        let price = finite("price", self.df_r * black_from_d(fwd, k, d1, d2, call))?;
        let sqrt_tau = self.tau.sqrt();
        // Partial derivatives at fixed implied volatility.
        let vega_s = finite("vega", self.df_q * s * n1 * sqrt_tau)?;
        let gamma_s = finite("gamma", self.df_q * n1 / (s * self.vol))?;
        let vanna_s = finite("vanna", -self.df_q * n1 * d2 / self.sig)?;
        let volga_s = finite("volga", vega_s * d1 * d2 / self.sig)?;
        // dsigma/dS and d2sigma/dS2 under the dynamics.
        let (ds, dss) = if self.sticky_moneyness {
            // k = ln K - ln S - (r - q) tau: dk/dS = -1/S, d2k/dS2 = 1/S^2.
            (-self.s1 / s, (self.s2 + self.s1) / (s * s))
        } else {
            (0.0, 0.0)
        };
        let delta = finite("delta", bs_delta + vega_s * ds)?;
        let gamma = finite(
            "gamma",
            gamma_s + 2.0 * vanna_s * ds + volga_s * ds * ds + vega_s * dss,
        )?;
        // theta = -dV/dtau, V = e^{-r tau} Black(F(tau), K, w(k(tau), tau)),
        // dk/dtau = -(r - q), dBlack/dw = K n(d2) / (2 sqrt w).
        let black_w = k * n2 / (2.0 * self.vol);
        let dw_dtau = self.w_k * (-(self.r - self.q)) + self.w_tau;
        let dv_dtau =
            -self.r * price + self.df_r * (black_f * fwd * (self.r - self.q) + black_w * dw_dtau);
        let theta = finite("theta", -dv_dtau)?;
        Ok(OptionValuation {
            price: to_decimal("price", price)?,
            delta: to_decimal("delta", delta)?,
            gamma: to_decimal("gamma", gamma)?,
            vega: to_decimal("vega", vega_s)?,
            theta: to_decimal("theta", theta)?,
        })
    }
}

/// The undiscounted Black price of a forward `F`, strike `K` and total
/// variance `w` (Black, 1976). Increasing in `w`. At `w = 0` it is the
/// intrinsic value of the forward, `max(F - K, 0)` or `max(K - F, 0)`.
///
/// # Errors
///
/// [`MMError::InvalidMarketState`] when `F` or `K` is not finite and
/// positive, or `w` is negative or not finite; [`MMError::NumericalError`]
/// for a non-finite result.
pub fn black_undiscounted(fwd: f64, strike: f64, w: f64, style: OptionStyle) -> MMResult<f64> {
    check_black(fwd, strike, w)?;
    let call = matches!(style, OptionStyle::Call);
    if w == 0.0 {
        let intrinsic = if call { fwd - strike } else { strike - fwd };
        return Ok(intrinsic.max(0.0));
    }
    let vol = w.sqrt();
    let d1 = finite("d1", ((fwd.ln() - strike.ln()) + 0.5 * w) / vol)?;
    let d2 = d1 - vol;
    finite("Black price", black_from_d(fwd, strike, d1, d2, call))
}

/// The undiscounted Black price from `d1` and `d2`. The out-of-the-money
/// option is computed directly and the in-the-money one by parity,
/// `intrinsic + out-of-the-money`, so a deep in-the-money price does not
/// carry the cancellation of `F N(d1) - K N(d2)` and stays increasing in `w`.
fn black_from_d(fwd: f64, strike: f64, d1: f64, d2: f64, call: bool) -> f64 {
    let otm_call = || fwd * norm_cdf(d1) - strike * norm_cdf(d2);
    let otm_put = || strike * norm_cdf(-d2) - fwd * norm_cdf(-d1);
    match (call, fwd > strike) {
        (true, true) => (fwd - strike) + otm_put(),
        (true, false) => otm_call(),
        (false, true) => otm_put(),
        (false, false) => (strike - fwd) + otm_call(),
    }
}

/// `d Black / dw = K n(d2) / (2 sqrt w)`, the same for calls and puts.
///
/// # Errors
///
/// The input errors of [`black_undiscounted`], and
/// [`MMError::NumericalError`] at `w = 0`, where the derivative is singular
/// (infinite at the money); a solver must bracket there instead.
pub fn black_w(fwd: f64, strike: f64, w: f64) -> MMResult<f64> {
    check_black(fwd, strike, w)?;
    if w == 0.0 {
        return Err(numerical("d Black / dw is singular at w = 0".to_string()));
    }
    let vol = w.sqrt();
    let d2 = finite("d2", ((fwd.ln() - strike.ln()) - 0.5 * w) / vol)?;
    finite("d Black / dw", strike * norm_pdf(d2) / (2.0 * vol))
}

fn check_black(fwd: f64, strike: f64, w: f64) -> MMResult<()> {
    for (name, v) in [("forward", fwd), ("strike", strike)] {
        if !(v.is_finite() && v > 0.0) {
            return Err(invalid(format!(
                "{name} must be finite and positive, got {v}"
            )));
        }
    }
    if !(w.is_finite() && w >= 0.0) {
        return Err(invalid(format!(
            "total variance must be finite and non-negative, got {w}"
        )));
    }
    Ok(())
}

fn to_f64(name: &str, v: Decimal) -> MMResult<f64> {
    v.to_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| numerical(format!("{name} {v} has no finite f64 value")))
}

fn to_decimal(name: &str, v: f64) -> MMResult<Decimal> {
    Decimal::from_f64(v).ok_or_else(|| numerical(format!("{name} {v} has no Decimal value")))
}

fn finite(name: &str, v: f64) -> MMResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(numerical(format!("{name} is not finite: {v}")))
    }
}

fn positive(name: &str, v: f64) -> MMResult<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(numerical(format!(
            "{name} must be finite and positive, got {v}"
        )))
    }
}

#[cold]
#[inline(never)]
fn invalid(msg: String) -> MMError {
    MMError::InvalidMarketState(msg)
}

#[cold]
#[inline(never)]
fn numerical(msg: String) -> MMError {
    MMError::NumericalError(msg)
}
