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
use crate::options::special::{erfcx, norm_cdf, norm_pdf};
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

    /// Checks the inputs every contract of one expiration shares, without
    /// valuing any contract: a failure here fails for all of them. The
    /// default accepts anything; a strike-specific failure is never reported
    /// here.
    ///
    /// # Errors
    ///
    /// Whatever the pricer considers a shared failure.
    fn check_market(
        &self,
        _spot: Decimal,
        _tau_years: Decimal,
        _rate: Decimal,
        _carry: Decimal,
    ) -> MMResult<()> {
        Ok(())
    }
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

    /// A positive spot, a non-negative time to expiry, rates with a finite
    /// f64 value, and a surface that answers at this expiry at the money
    /// forward (`k = 0`). No contract is valued.
    fn check_market(
        &self,
        spot: Decimal,
        tau_years: Decimal,
        rate: Decimal,
        carry: Decimal,
    ) -> MMResult<()> {
        if spot <= Decimal::ZERO {
            return Err(invalid(format!("spot must be positive, got {spot}")));
        }
        if tau_years < Decimal::ZERO {
            return Err(invalid(format!(
                "time to expiry must be non-negative, got {tau_years}"
            )));
        }
        to_f64("spot", spot)?;
        to_f64("rate", rate)?;
        to_f64("carry", carry)?;
        let tau = to_f64("time to expiry", tau_years)?;
        if tau > 0.0 {
            self.surface.total_variance(0.0, tau)?;
        }
        Ok(())
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
    k_current: f64,
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
        // k = ln(K / S) - (r - q) tau, with ln(K / S) taken near the money
        // as log1p of the exact Decimal (K - S) / S (ln K - ln S would lose
        // the digits that set d1 and d2 at small w).
        let k_current = finite(
            "log moneyness",
            ln_ratio(inputs.strike, inputs.spot)? - growth,
        )?;
        let (sticky_moneyness, k_read) = match dynamics {
            Dynamics::StickyMoneyness => (true, k_current),
            Dynamics::StickyStrike { s_ref } => (
                false,
                finite("log moneyness", ln_ratio(inputs.strike, s_ref)? - growth)?,
            ),
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
            k_current,
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
        let bs_delta = if call {
            self.df_q * norm_cdf(d1)
        } else {
            -self.df_q * norm_cdf(-d1)
        };
        let black = black_price(fwd, k, self.k_current, self.vol, call)?;
        let price = finite("price", self.df_r * black)?;
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
        // dk/dtau = -(r - q), dBlack/dw = K n(d2) / (2 sqrt w). With
        // e^{-r tau} F = e^{-q tau} S the F N(d1) terms of -r V and of the
        // forward's drift cancel analytically, which leaves
        //   call: q S e^{-q tau} N(d1) - r K e^{-r tau} N(d2) - e^{-r tau} Black_w dw/dtau,
        //   put:  r K e^{-r tau} N(-d2) - q S e^{-q tau} N(-d1) - e^{-r tau} Black_w dw/dtau,
        // with no difference of terms of the order of S.
        let black_w = k * n2 / (2.0 * self.vol);
        let dw_dtau = self.w_tau - (self.r - self.q) * self.w_k;
        let carry_part = if call {
            self.q * s * self.df_q * norm_cdf(d1) - self.r * k * self.df_r * norm_cdf(d2)
        } else {
            self.r * k * self.df_r * norm_cdf(-d2) - self.q * s * self.df_q * norm_cdf(-d1)
        };
        let theta = finite("theta", carry_part - self.df_r * black_w * dw_dtau)?;
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
    let vol = positive("total volatility sqrt(w)", w.sqrt())?;
    let k = finite("log moneyness", ln_ratio_f64(strike, fwd))?;
    black_price(fwd, strike, k, vol, call)
}

/// Below this `t = sqrt(w) / 2` the out-of-the-money price is the Taylor
/// series in `t` (Jaeckel, 2015, "Let's be rational", section 3).
const SMALL_T: f64 = 0.21;
/// Below this `h = x / sqrt(w)` the density `n(h)` is below the smallest
/// subnormal, so the normalised price underflows to zero (its true value).
const UNDERFLOW_H: f64 = -40.0;
/// Highest odd order of the small-`t` series.
const SERIES_ORDER: usize = 17;

/// The undiscounted Black price with `k = ln(K / F)` and `vol = sqrt(w) > 0`:
/// the intrinsic value (`K expm1(-k)` near the money, `F - K` away from it, or
/// their negatives for a put) plus the time value `sqrt(F K) b(-|k|, vol)`,
/// where `b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2)` is the
/// normalised out-of-the-money call (Jaeckel, 2015, "Let's be rational"; by
/// put-call symmetry the out-of-the-money put of the pair has the same
/// normalised value). The time value is computed directly and stably, so no
/// price carries the cancellation of `F N(d1) - K N(d2)`, which near the
/// money at small `w` loses every digit.
fn black_price(fwd: f64, strike: f64, k: f64, vol: f64, call: bool) -> MMResult<f64> {
    let otm = finite("normalised Black price", normalised_otm_call(-k.abs(), vol))?;
    let scale = fwd.sqrt() * strike.sqrt();
    let time_value = finite("Black time value", scale * otm)?;
    // Near the money F - K is K (e^{x} - 1) = K expm1(x), x = -k, which keeps
    // the precision of the log ratio (F - K in f64 would carry the rounding
    // of F and K); away from it the direct difference has no cancellation.
    let forward_minus_strike = if k.abs() <= std::f64::consts::LN_2 {
        strike * (-k).exp_m1()
    } else {
        fwd - strike
    };
    let intrinsic = if call {
        forward_minus_strike
    } else {
        -forward_minus_strike
    };
    finite("Black price", intrinsic.max(0.0) + time_value)
}

/// `b(x, s)` for `x <= 0`, with `h = x / s` and `t = s / 2`:
/// - `t < SMALL_T`: `2 n(h) sum_{odd j} G_j(h) t^j / j!`, the Taylor series
///   of `e^{ht} N(h + t) - e^{-ht} N(h - t)` in `t` (see `small_t_sum`);
/// - otherwise, with `d1 = h + t <= 0`:
///   `e^{-(h^2 + t^2)/2} (erfcx(-d1/sqrt 2) - erfcx(-d2/sqrt 2)) / 2`, which
///   has no overflow and no tail cancellation;
/// - otherwise `e^{ht} N(h + t) - e^{-ht} N(h - t)`, where `t >= SMALL_T`
///   bounds the cancellation.
fn normalised_otm_call(x: f64, s: f64) -> f64 {
    let h = x / s;
    let t = 0.5 * s;
    if h < UNDERFLOW_H {
        return 0.0;
    }
    if t < SMALL_T {
        return 2.0 * norm_pdf(h) * small_t_sum(h, t);
    }
    if h + t <= 0.0 {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let e = (-0.5 * (h * h + t * t)).exp();
        return 0.5 * e * (erfcx(-(h + t) * r) - erfcx(-(h - t) * r));
    }
    (h * t).exp() * norm_cdf(h + t) - (-h * t).exp() * norm_cdf(h - t)
}

/// `sum_{odd j <= SERIES_ORDER} G_j(h) t^j / j!`, where `G_j` is the j-th
/// derivative at `t = 0` of `g(t) = e^{ht} N(h + t)` divided by `n(h)`.
///
/// Since `g' = h g + e^{ht} n(h + t)` and `e^{ht} n(h + t) = n(h) e^{-t^2/2}`,
/// `G_0 = Y(h) = N(h) / n(h)` and `G_j = h G_{j-1} + U_{j-1}`, with `U_m`
/// the derivatives of `e^{-t^2/2}` at 0: `U_0 = 1`, `U_{m+2} = -(m + 1) U_m`,
/// zero for odd `m`. The odd part of `g(t) - g(-t)` gives the factor 2 in
/// `normalised_otm_call`.
fn small_t_sum(h: f64, t: f64) -> f64 {
    // Y(h) = sqrt(pi/2) erfcx(-h / sqrt 2), for h <= 0.
    let mut g = (0.5 * std::f64::consts::PI).sqrt() * erfcx(-h * std::f64::consts::FRAC_1_SQRT_2);
    // U_{j-1} for the current j, and the index m = j - 1 as f64.
    let mut u_even = 1.0; // U_0, U_2, ... in turn
    let mut m = 0.0_f64;
    let mut t_pow = 1.0;
    let mut factorial = 1.0;
    let mut total = 0.0;
    for j in 1..=SERIES_ORDER {
        let u = if j % 2 == 1 { u_even } else { 0.0 };
        g = h * g + u;
        t_pow *= t;
        factorial *= m + 1.0;
        if j % 2 == 1 {
            total += g * t_pow / factorial;
            // Next even index: U_{m+2} = -(m + 1) U_m.
            u_even *= -(m + 1.0);
        }
        m += 1.0;
    }
    total
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
    let d2 = finite("d2", (-ln_ratio_f64(strike, fwd) - 0.5 * w) / vol)?;
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

/// `ln(num / den)` for positive Decimals: near 1 (`|num/den - 1| <= 1/2`) as
/// `log1p` of the exact Decimal `(num - den) / den`, elsewhere as
/// `ln num - ln den` (which never forms a ratio that could overflow).
fn ln_ratio(num: Decimal, den: Decimal) -> MMResult<f64> {
    let rel = num
        .checked_sub(den)
        .and_then(|d| d.checked_div(den))
        .and_then(|r| r.to_f64());
    match rel {
        Some(r) if r.is_finite() && r.abs() <= 0.5 => Ok(r.ln_1p()),
        _ => {
            let (n, d) = (
                to_f64("ratio numerator", num)?,
                to_f64("ratio denominator", den)?,
            );
            if !(n > 0.0 && d > 0.0) {
                return Err(numerical(format!(
                    "{num} / {den} has no positive f64 terms"
                )));
            }
            finite("log ratio", n.ln() - d.ln())
        }
    }
}

/// `ln(num / den)` for positive finite f64: near 1 as `log1p((num - den) /
/// den)`, where `num - den` is exact (Sterbenz), elsewhere `ln num - ln den`.
fn ln_ratio_f64(num: f64, den: f64) -> f64 {
    if num >= 0.5 * den && num <= 2.0 * den {
        ((num - den) / den).ln_1p()
    } else {
        num.ln() - den.ln()
    }
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    // The normalised out-of-the-money Black call b(x, s) in 400-digit
    // decimal arithmetic (`scripts/black_reference.py`), on a grid that
    // crosses every branch: the small-t series (s < 0.42), the erfcx form
    // and the direct form, from the money (x = 0) to h = x / s = -40, and
    // s down to 1e-28. Tolerance 2e-11 relative (the series loses digits as
    // h grows negative, where the value itself is below 1e-25).
    const B_TABLE: &[(f64, f64, f64)] = &[
        (0.0, 1e-28, 3.989422804014327e-29),
        (-1e-30, 1e-28, 3.9396222734922846e-29),
        (0.0, 1e-14, 3.9894228040143265e-15),
        (-1e-30, 1e-14, 3.9894228040143265e-15),
        (0.0, 1e-08, 3.989422804014327e-09),
        (-1e-30, 1e-08, 3.989422804014327e-09),
        (-1e-10, 1e-08, 3.939622273492285e-09),
        (0.0, 0.0001, 3.989422802352067e-05),
        (-1e-30, 0.0001, 3.989422802352067e-05),
        (-1e-10, 0.0001, 3.989417802354062e-05),
        (-0.001, 0.0001, 7.474560245421083e-29),
        (0.0, 0.01, 0.003989406181481645),
        (-1e-30, 0.01, 0.003989406181481645),
        (-1e-10, 0.01, 0.003989406131481645),
        (-0.001, 0.01, 0.003509336918642542),
        (-0.1, 0.01, 7.474468572496729e-27),
        (0.0, 0.1, 0.039877611676744924),
        (-1e-30, 0.1, 0.039877611676744924),
        (-1e-10, 0.1, 0.03987761162674493),
        (-0.001, 0.1, 0.03937960884362347),
        (-0.1, 0.1, 0.008324939376702142),
        (-1.0, 0.1, 7.465397611568617e-26),
        (0.0, 0.41, 0.1624278749748963),
        (-1e-30, 0.41, 0.1624278749748963),
        (-1e-10, 0.41, 0.1624278749248963),
        (-0.001, 0.41, 0.16192837165605764),
        (-0.1, 0.41, 0.1173516170807252),
        (-1.0, 0.41, 0.0009739269112510348),
        (-5.0, 0.41, 5.3628497718814674e-36),
        (0.0, 0.43, 0.1702326814403915),
        (-1e-30, 0.43, 0.1702326814403915),
        (-1e-10, 0.43, 0.1702326813903915),
        (-0.001, 0.43, 0.16973315598619984),
        (-0.1, 0.43, 0.12493818147677391),
        (-1.0, 0.43, 0.001433091524446669),
        (-5.0, 0.43, 5.296500261826388e-33),
        (0.0, 1.0, 0.3829249225480262),
        (-1e-30, 1.0, 0.3829249225480262),
        (-1e-10, 1.0, 0.38292492249802623),
        (-0.001, 1.0, 0.38242514642546155),
        (-0.1, 1.0, 0.3351420702955817),
        (-1.0, 1.0, 0.07699102314167414),
        (-5.0, 1.0, 4.755776371084643e-08),
        (0.0, 3.0, 0.8663855974622838),
        (-1e-30, 3.0, 0.8663855974622838),
        (-1e-10, 3.0, 0.8663855974122838),
        (-0.001, 3.0, 0.8658857273259186),
        (-0.1, 3.0, 0.8176638568093587),
        (-1.0, 3.0, 0.4777039972474732),
        (-5.0, 3.0, 0.02621728202255927),
        (0.0, 10.0, 0.9999994266968563),
        (-1e-30, 10.0, 0.9999994266968563),
        (-1e-10, 10.0, 0.9999994266468563),
        (-0.001, 10.0, 0.9994995516760282),
        (-0.1, 10.0, 0.9512288512243005),
        (-1.0, 10.0, 0.6065300890763248),
        (-5.0, 10.0, 0.08208448838567498),
    ];

    #[test]
    fn test_normalised_otm_call_matches_high_precision_on_every_branch() {
        for &(x, s, want) in B_TABLE {
            let got = normalised_otm_call(x, s);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 2e-11,
                "b({x}, {s}) = {got:e}, want {want:e} (rel {rel:e})"
            );
        }
    }

    #[test]
    fn test_normalised_otm_call_below_the_density_underflow_is_zero() {
        assert_eq!(normalised_otm_call(-1.0, 0.01), 0.0); // h = -100
    }
}
