//! Implied total variance of cleaned quotes.
//!
//! Each price is turned into an undiscounted Black price (`price / DF`) and
//! inverted for the total variance `w` on its out of the money price: the
//! price itself for an out of the money option, and the price less the
//! intrinsic value for an in the money one (undiscounted put-call parity,
//! `C - P = F - K`). The inversion is a safeguarded Newton on
//! [`black_undiscounted`](crate::options::black_undiscounted) and
//! [`black_w`](crate::options::black_w), inside the bracket `[0, w_max]`
//! (`w_max = max_vol^2 tau`), with bisection whenever the Newton step leaves
//! the bracket or `black_w` is unavailable (it is singular at `w = 0`). It
//! stops when the price matches to [`IvConfig::price_tolerance`] relative to
//! the out of the money price, or when the bracket is narrower than
//! [`IvConfig::w_tolerance`] relative to its upper end. Every loop is
//! bounded. Where the Black price at a trial `w` is positive but below the
//! smallest positive `f64` ([`MMError::PriceUnderflow`], never a 0),
//! it is below every positive price, the target included, so the root lies
//! above that `w` and the bracket moves there.
//!
//! Undiscounted bounds: `max(F - K, 0) <= C <= F` and
//! `max(K - F, 0) <= P <= K`. Out of the money, a price of 0 has `w = 0`:
//! a valid extreme, which is what a zero bid gives; any positive price is
//! inverted, however small. In the money, a time
//! value that the intrinsic value's rounding leaves unknown to the price
//! tolerance ([`IvFailure::BelowResolution`]) is not inverted. That, a price
//! below the lower bound, and a price at or above the upper bound are each
//! an [`IvFailure`], never a `w`.
//!
//! For a cleaned quote, the mid gives the observed `w`, and the bid and ask
//! give the band `[w_bid, w_ask]` whose width sets the weight,
//! `1 / max(w_ask - w_bid, eps)^2`, which must be finite and positive (a
//! quote whose weight is not is left out with its failure). A bid at or
//! below the lower bound, or
//! within the intrinsic value's rounding, is `w_bid = 0`, flagged
//! [`IvObservation::bid_at_bound`]. A quote whose ask or mid has no implied
//! variance is left out of the fit with its failure.

use optionstratlib::OptionStyle;
use rust_decimal::prelude::ToPrimitive;

use super::quotes::{CleanQuote, CleaningReport};
use crate::options::pricing::{black_undiscounted, black_w};
use crate::types::error::{MMError, MMResult};

/// Settings of the inversion and of the weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvConfig {
    /// Largest implied volatility searched; `w_max = max_vol^2 tau`.
    pub max_vol: f64,
    /// Price tolerance relative to the out of the money price solved for.
    pub price_tolerance: f64,
    /// Relative width of the bracket at which the search stops.
    pub w_tolerance: f64,
    /// Iteration cap.
    pub max_iterations: usize,
    /// Smallest band width in the weight `1 / max(w_ask - w_bid, eps)^2`.
    /// A slice's weights are also capped relative to its median by
    /// [`FitConfig::max_weight_ratio`](super::FitConfig::max_weight_ratio).
    pub weight_eps: f64,
}

impl Default for IvConfig {
    fn default() -> Self {
        Self {
            max_vol: 5.0,
            price_tolerance: 1e-12,
            w_tolerance: 1e-14,
            max_iterations: 200,
            weight_eps: 1e-8,
        }
    }
}

impl IvConfig {
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] unless every value is finite and
    /// positive, the largest weight `1 / weight_eps^2` is finite, and the
    /// iteration cap is at least 1.
    pub fn validate(&self) -> MMResult<()> {
        for (name, v) in [
            ("max_vol", self.max_vol),
            ("price_tolerance", self.price_tolerance),
            ("w_tolerance", self.w_tolerance),
            ("weight_eps", self.weight_eps),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(MMError::InvalidConfiguration(format!(
                    "iv: {name} must be finite and positive, got {v}"
                )));
            }
        }
        if !(1.0 / (self.weight_eps * self.weight_eps)).is_finite() {
            return Err(MMError::InvalidConfiguration(format!(
                "iv: weight_eps {} gives an infinite largest weight",
                self.weight_eps
            )));
        }
        if self.max_iterations == 0 {
            return Err(MMError::InvalidConfiguration(
                "iv: max_iterations must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// `1 / max(w_ask - w_bid, eps)^2`, finite and positive.
///
/// # Errors
///
/// [`IvFailure::Numerical`] when the weight is not finite and positive: a
/// band whose square overflows, or one that is not finite.
fn band_weight(w_bid: f64, w_ask: f64, eps: f64) -> Result<f64, IvFailure> {
    let band = (w_ask - w_bid).max(eps);
    let weight = 1.0 / (band * band);
    if weight.is_finite() && weight > 0.0 {
        Ok(weight)
    } else {
        Err(IvFailure::Numerical(format!(
            "weight {weight} of the band [{w_bid}, {w_ask}] is not finite and positive"
        )))
    }
}

/// Why a price has no implied total variance.
#[derive(Debug, Clone, PartialEq)]
pub enum IvFailure {
    /// The price is below the undiscounted lower bound.
    BelowLowerBound {
        /// Undiscounted price.
        price: f64,
        /// The bound.
        bound: f64,
    },
    /// The price is at or above the undiscounted upper bound.
    AtOrAboveUpperBound {
        /// Undiscounted price.
        price: f64,
        /// The bound.
        bound: f64,
    },
    /// In the money, the time value, `price - intrinsic`, is not known to
    /// [`IvConfig::price_tolerance`] relative accuracy: the intrinsic
    /// value's rounding, about `eps * intrinsic`, is above that share of
    /// it.
    BelowResolution {
        /// `price - intrinsic`.
        time_value: f64,
        /// `eps * intrinsic / price_tolerance`.
        resolution: f64,
    },
    /// `black(w_max)` is below the price: the volatility is above `max_vol`.
    BracketFailed {
        /// The search's upper end.
        w_max: f64,
    },
    /// The iteration cap was reached.
    NoConvergence {
        /// Iterations run.
        iterations: usize,
    },
    /// A pricing or conversion error.
    Numerical(String),
}

impl std::fmt::Display for IvFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BelowLowerBound { price, bound } => {
                write!(f, "price {price} below the lower bound {bound}")
            }
            Self::AtOrAboveUpperBound { price, bound } => {
                write!(f, "price {price} at or above the upper bound {bound}")
            }
            Self::BelowResolution {
                time_value,
                resolution,
            } => write!(
                f,
                "time value {time_value} below the resolution {resolution} of the intrinsic value"
            ),
            Self::BracketFailed { w_max } => write!(f, "no root below w_max {w_max}"),
            Self::NoConvergence { iterations } => {
                write!(f, "no convergence in {iterations} iterations")
            }
            Self::Numerical(why) => write!(f, "{why}"),
        }
    }
}

/// The implied total variance of the undiscounted Black price `price`.
///
/// # Errors
///
/// An [`IvFailure`] with the reason.
pub fn implied_total_variance(
    price: f64,
    fwd: f64,
    strike: f64,
    tau: f64,
    style: OptionStyle,
    config: &IvConfig,
) -> Result<f64, IvFailure> {
    config
        .validate()
        .map_err(|e| IvFailure::Numerical(e.to_string()))?;
    if !(price.is_finite() && fwd.is_finite() && fwd > 0.0 && strike.is_finite() && strike > 0.0)
        || !(tau.is_finite() && tau > 0.0)
    {
        return Err(IvFailure::Numerical(format!(
            "invalid inputs: price {price}, fwd {fwd}, strike {strike}, tau {tau}"
        )));
    }
    let call = matches!(style, OptionStyle::Call);
    let (lower, upper) = if call {
        ((fwd - strike).max(0.0), fwd)
    } else {
        ((strike - fwd).max(0.0), strike)
    };
    // Below the intrinsic value by more than its rounding.
    if price < lower - 2.0 * f64::EPSILON * lower {
        return Err(IvFailure::BelowLowerBound {
            price,
            bound: lower,
        });
    }
    if price >= upper {
        return Err(IvFailure::AtOrAboveUpperBound {
            price,
            bound: upper,
        });
    }
    if price == 0.0 {
        // Out of the money at 0 (a zero bid): w = 0, the declared extreme.
        // In the money, a price at the intrinsic value is below the
        // resolution below.
        return Ok(0.0);
    }
    // Solve on the out of the money price: the price itself, or, in the
    // money, the price less the intrinsic value (undiscounted put-call
    // parity, C - P = F - K), which is the other style's price.
    let target = price - lower;
    // The intrinsic value is rounded to about eps * intrinsic, which the
    // subtraction carries into the target: the target is known to
    // price_tolerance relative accuracy only above eps * intrinsic /
    // price_tolerance. Out of the money the intrinsic value is 0 and every
    // positive price is known exactly.
    let resolution = f64::EPSILON * lower / config.price_tolerance;
    if target <= resolution {
        return Err(IvFailure::BelowResolution {
            time_value: target,
            resolution,
        });
    }
    let otm_style = if strike >= fwd {
        OptionStyle::Call
    } else {
        OptionStyle::Put
    };
    let tol = config.price_tolerance * target;
    // `None`: positive but below the smallest positive f64, so below the
    // target, which is a positive f64.
    let black = |w: f64| {
        black_or_underflow(fwd, strike, w, otm_style)
            .map_err(|e| IvFailure::Numerical(e.to_string()))
    };
    let w_max = config.max_vol * config.max_vol * tau;
    if black(w_max)?.is_none_or(|p| p < target - tol) {
        return Err(IvFailure::BracketFailed { w_max });
    }
    let (mut lo, mut hi) = (0.0_f64, w_max);
    // Brenner and Subrahmanyam's at the money guess, w = 2 pi (C / F)^2,
    // kept inside the bracket.
    let time_value = target / fwd;
    let mut w = (2.0 * std::f64::consts::PI * time_value * time_value)
        .max(0.25 * hi * f64::EPSILON)
        .min(0.5 * hi);
    for _ in 0..config.max_iterations {
        let diff = black(w)?.map(|p| p - target);
        if diff.is_some_and(|d| d.abs() <= tol) {
            return Ok(w);
        }
        if diff.is_none_or(|d| d < 0.0) {
            lo = w;
        } else {
            hi = w;
        }
        if hi - lo <= config.w_tolerance * hi {
            return Ok(0.5 * (lo + hi));
        }
        let newton = diff.and_then(|d| {
            black_w(fwd, strike, w)
                .ok()
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| w - d / v)
        });
        w = match newton {
            Some(next) if next > lo && next < hi => next,
            _ => 0.5 * (lo + hi),
        };
    }
    Err(IvFailure::NoConvergence {
        iterations: config.max_iterations,
    })
}

/// [`black_undiscounted`], with `None` for [`MMError::PriceUnderflow`], a
/// price that is positive but below the smallest positive `f64`: smaller
/// than every positive `f64`, so it still orders against one.
///
/// # Errors
///
/// Every other error of [`black_undiscounted`], unchanged.
pub(crate) fn black_or_underflow(
    fwd: f64,
    strike: f64,
    w: f64,
    style: OptionStyle,
) -> MMResult<Option<f64>> {
    match black_undiscounted(fwd, strike, w, style) {
        Ok(price) => Ok(Some(price)),
        Err(MMError::PriceUnderflow) => Ok(None),
        Err(error) => Err(error),
    }
}

/// A cleaned quote with its implied total variances.
#[derive(Debug, Clone, PartialEq)]
pub struct IvObservation {
    /// The cleaned quote.
    pub quote: CleanQuote,
    /// From the mid.
    pub w_mid: f64,
    /// From the bid (0 for a zero bid or one at the lower bound).
    pub w_bid: f64,
    /// The bid is at or below its lower bound: `w_bid = 0` is a declared
    /// extreme of the band, not an implied variance read from the bid.
    pub bid_at_bound: bool,
    /// From the ask.
    pub w_ask: f64,
    /// `1 / max(w_ask - w_bid, eps)^2`.
    pub weight: f64,
}

/// The implied variances of a cleaning report's fit set.
#[derive(Debug, Clone, PartialEq)]
pub struct Extraction {
    /// Quotes with an implied variance, in fit set order.
    pub observations: Vec<IvObservation>,
    /// Quotes left out, by input index, with the failure of the price that
    /// failed (`"ask"` or `"mid"`).
    pub failures: Vec<(usize, &'static str, IvFailure)>,
}

fn price_f64(v: rust_decimal::Decimal) -> Result<f64, IvFailure> {
    v.to_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| IvFailure::Numerical(format!("price {v} has no finite f64 value")))
}

/// Implied total variances of every quote in `report.fit_set`.
///
/// # Errors
///
/// [`MMError::InvalidConfiguration`] for an invalid `config`.
pub fn extract(report: &CleaningReport, config: &IvConfig) -> MMResult<Extraction> {
    config.validate()?;
    let mut observations = Vec::new();
    let mut failures = Vec::new();
    for c in &report.fit_set {
        let strike = match price_f64(c.quote.strike) {
            Ok(v) => v,
            Err(e) => {
                failures.push((c.index, "strike", e));
                continue;
            }
        };
        let solve = |p: rust_decimal::Decimal| -> Result<f64, IvFailure> {
            let undiscounted = price_f64(p)? / c.discount;
            implied_total_variance(
                undiscounted,
                c.forward,
                strike,
                c.tau,
                c.quote.style,
                config,
            )
        };
        let w_ask = match solve(c.quote.ask) {
            Ok(w) => w,
            Err(e) => {
                failures.push((c.index, "ask", e));
                continue;
            }
        };
        let Some(mid) = c
            .quote
            .bid
            .checked_add(c.quote.ask)
            .and_then(|sum| sum.checked_div(rust_decimal::Decimal::TWO))
        else {
            failures.push((
                c.index,
                "mid",
                IvFailure::Numerical("mid overflowed".to_string()),
            ));
            continue;
        };
        let w_mid = match solve(mid) {
            Ok(w) => w,
            Err(e) => {
                failures.push((c.index, "mid", e));
                continue;
            }
        };
        // A bid at or below the lower bound (a zero bid), or within the
        // rounding of the intrinsic value, is the band's declared extreme
        // w = 0, flagged.
        let (w_bid, bid_at_bound) = match solve(c.quote.bid) {
            Ok(w) => (w, w == 0.0),
            Err(IvFailure::BelowLowerBound { .. } | IvFailure::BelowResolution { .. }) => {
                (0.0, true)
            }
            Err(e) => {
                failures.push((c.index, "bid", e));
                continue;
            }
        };
        let weight = match band_weight(w_bid, w_ask, config.weight_eps) {
            Ok(weight) => weight,
            Err(e) => {
                failures.push((c.index, "weight", e));
                continue;
            }
        };
        observations.push(IvObservation {
            quote: c.clone(),
            w_mid,
            w_bid,
            bid_at_bound,
            w_ask,
            weight,
        });
    }
    Ok(Extraction {
        observations,
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> IvConfig {
        IvConfig::default()
    }

    #[test]
    fn test_implied_total_variance_round_trips_across_strikes_and_levels() {
        let fwd = 100.0;
        for &strike in &[50.0, 80.0, 99.0, 100.0, 101.0, 120.0, 200.0] {
            for &w in &[1e-4, 0.01, 0.09, 0.5, 2.0] {
                for style in [OptionStyle::Call, OptionStyle::Put] {
                    // A price below the smallest positive f64 has no
                    // inverse to check.
                    let Some(price) = black_or_underflow(fwd, strike, w, style).unwrap() else {
                        continue;
                    };
                    let lower = if matches!(style, OptionStyle::Call) {
                        (fwd - strike).max(0.0)
                    } else {
                        (strike - fwd).max(0.0)
                    };
                    // Prices indistinguishable from intrinsic carry no
                    // information about w.
                    if price - lower <= 1e-9 * fwd {
                        continue;
                    }
                    let got =
                        implied_total_variance(price, fwd, strike, 1.0, style, &config()).unwrap();
                    assert!(
                        (got - w).abs() <= 1e-8 * w.max(1e-3),
                        "K {strike} w {w} {style:?}: {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_implied_total_variance_round_trips_small_w_near_the_money() {
        // Short expiries near the money: w in [1e-8, 1e-4], strikes within
        // two standard deviations, tau of one day.
        let (fwd, tau) = (100.0, 1.0 / 365.0);
        let c = config();
        for &w in &[1e-8, 3e-8, 1e-7, 1e-6, 1e-5, 1e-4] {
            let sd: f64 = f64::sqrt(w);
            for &z in &[-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0] {
                let strike = fwd * f64::exp(z * sd);
                for style in [OptionStyle::Call, OptionStyle::Put] {
                    let price = black_undiscounted(fwd, strike, w, style).unwrap();
                    let got = implied_total_variance(price, fwd, strike, tau, style, &c).unwrap();
                    // The solver stops when the price matches to
                    // price_tolerance * fwd; the matching w error is that
                    // over the vega in w, plus round-off.
                    let vega = black_w(fwd, strike, w).unwrap();
                    let allowed = 2.0 * c.price_tolerance * fwd / vega + 1e-12 * w;
                    assert!(
                        (got - w).abs() <= allowed,
                        "w {w} z {z} {style:?}: got {got}, allowed {allowed}"
                    );
                    assert!((got - w).abs() <= 1e-7 * w, "w {w} z {z} {style:?}: {got}");
                }
            }
        }
    }

    /// A fixed linear congruential sequence in [0, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    #[test]
    fn test_implied_total_variance_small_premiums_are_inverted_not_zero() {
        // Out of the money premiums far below price_tolerance * F, which an
        // absolute tolerance would read as w = 0 or stop on early.
        let c = config();
        let fwd = 100.0;
        for (z, vol, tau) in [
            (-7.30, 1.811, 0.6535),
            (7.55, 0.985, 0.26),
            (-5.79, 1.847, 1.497),
        ] {
            let w: f64 = vol * vol * tau;
            let strike = fwd * f64::exp(z * w.sqrt());
            let style = if strike >= fwd {
                OptionStyle::Call
            } else {
                OptionStyle::Put
            };
            let price = black_undiscounted(fwd, strike, w, style).unwrap();
            let got = implied_total_variance(price, fwd, strike, tau, style, &c).unwrap();
            assert!(
                (got - w).abs() <= 1e-9 * w,
                "z {z}: price {price:e}, {got} vs {w}"
            );
        }
    }

    #[test]
    fn test_implied_total_variance_round_trips_to_eight_standard_deviations() {
        // z uniform in [-8, 8] standard deviations, vol in [0.05, 2], tau in
        // [0.02, 2]; both styles. Out of the money every representable
        // price is inverted; in the money, a time value lost in the
        // intrinsic value's rounding is BelowResolution, never a w.
        let c = config();
        let fwd = 100.0;
        let mut rng = Lcg(7);
        let (mut otm, mut tiny, mut itm, mut unresolved) = (0, 0, 0, 0);
        let mut worst = 0.0_f64;
        for _ in 0..20_000 {
            let z = 16.0 * rng.next() - 8.0;
            let vol = 0.05 + 1.95 * rng.next();
            let tau = 0.02 + 1.98 * rng.next();
            let w: f64 = vol * vol * tau;
            let strike = fwd * f64::exp(z * w.sqrt());
            for style in [OptionStyle::Call, OptionStyle::Put] {
                let Some(price) = black_or_underflow(fwd, strike, w, style).unwrap() else {
                    continue;
                };
                let in_the_money = match style {
                    OptionStyle::Call => strike < fwd,
                    OptionStyle::Put => strike > fwd,
                };
                match implied_total_variance(price, fwd, strike, tau, style, &c) {
                    Ok(got) => {
                        let rel = (got - w).abs() / w;
                        if in_the_money {
                            itm += 1;
                        } else {
                            otm += 1;
                            worst = worst.max(rel);
                            if (1e-15 * fwd..=1e-9 * fwd).contains(&price) {
                                tiny += 1;
                            }
                        }
                        assert!(
                            rel <= 1e-9,
                            "z {z} vol {vol} tau {tau} {style:?}: price {price:e}, {got} vs {w}"
                        );
                    }
                    Err(IvFailure::BelowResolution { .. }) if in_the_money => unresolved += 1,
                    Err(e) => panic!("z {z} vol {vol} tau {tau} {style:?}: price {price:e}: {e}"),
                }
            }
        }
        // Observed: every representable out of the money price inverted
        // (worst relative error 2.7e-12), 4757 of them in [1e-15 F, 1e-9 F];
        // in the money, 6690 inverted and 13310 below the resolution.
        assert!(tiny > 1000, "{tiny} premiums in [1e-15 F, 1e-9 F]");
        assert!(
            otm > 19_000 && itm > 1000 && unresolved > 0,
            "{otm} {itm} {unresolved}"
        );
        assert!(worst <= 1e-9, "{worst}");
    }

    #[test]
    fn test_black_or_underflow_reads_the_pricing_underflow() {
        // Far out of the money at small w: positive, below every f64.
        assert_eq!(
            black_undiscounted(100.0, 200.0, 1e-4, OptionStyle::Call),
            Err(MMError::PriceUnderflow)
        );
        assert_eq!(
            black_or_underflow(100.0, 200.0, 1e-4, OptionStyle::Call),
            Ok(None)
        );
        assert!(
            black_or_underflow(100.0, 100.0, 0.04, OptionStyle::Call)
                .unwrap()
                .is_some_and(|p| p > 0.0)
        );
        // Other errors stay errors, unchanged.
        assert!(matches!(
            black_or_underflow(-1.0, 100.0, 0.04, OptionStyle::Call),
            Err(MMError::InvalidMarketState(_))
        ));
    }

    #[test]
    fn test_implied_total_variance_underflowing_trial_moves_the_bracket() {
        // Out of the money, a price well above the tolerance whose first
        // trial, the at the money guess 2 pi (C / F)^2, is a w where the
        // Black price underflows.
        let (fwd, strike, w) = (100.0, 130.0, 0.004);
        let c = config();
        let price = black_undiscounted(fwd, strike, w, OptionStyle::Call).unwrap();
        assert!(price > 1e3 * c.price_tolerance * fwd);
        let guess = 2.0 * std::f64::consts::PI * (price / fwd).powi(2);
        assert!(
            black_or_underflow(fwd, strike, guess, OptionStyle::Call)
                .unwrap()
                .is_none()
        );
        let got = implied_total_variance(price, fwd, strike, 1.0, OptionStyle::Call, &c).unwrap();
        let allowed = 2.0 * c.price_tolerance * fwd / black_w(fwd, strike, w).unwrap() + 1e-12 * w;
        assert!((got - w).abs() <= allowed, "{got}");
    }

    #[test]
    fn test_implied_total_variance_bounds_and_bracket() {
        let c = config();
        // Out of the money, 0 is w = 0; in the money, the intrinsic value
        // and a time value lost in its rounding are below the resolution.
        assert!(matches!(
            implied_total_variance(10.0, 100.0, 90.0, 1.0, OptionStyle::Call, &c),
            Err(IvFailure::BelowResolution { .. })
        ));
        assert!(matches!(
            implied_total_variance(10.0 + 1e-10, 100.0, 90.0, 1.0, OptionStyle::Call, &c),
            Err(IvFailure::BelowResolution { .. })
        ));
        assert_eq!(
            implied_total_variance(0.0, 100.0, 110.0, 1.0, OptionStyle::Call, &c),
            Ok(0.0)
        );
        assert!(matches!(
            implied_total_variance(9.0, 100.0, 90.0, 1.0, OptionStyle::Call, &c),
            Err(IvFailure::BelowLowerBound { .. })
        ));
        assert!(matches!(
            implied_total_variance(100.0, 100.0, 90.0, 1.0, OptionStyle::Call, &c),
            Err(IvFailure::AtOrAboveUpperBound { .. })
        ));
        let tight = IvConfig { max_vol: 0.1, ..c };
        let price = black_undiscounted(100.0, 100.0, 0.25, OptionStyle::Call).unwrap();
        assert!(matches!(
            implied_total_variance(price, 100.0, 100.0, 1.0, OptionStyle::Call, &tight),
            Err(IvFailure::BracketFailed { .. })
        ));
        assert!(matches!(
            implied_total_variance(f64::NAN, 100.0, 100.0, 1.0, OptionStyle::Call, &c),
            Err(IvFailure::Numerical(_))
        ));
    }

    #[test]
    fn test_iv_config_invalid_values_return_err() {
        assert!(
            IvConfig {
                max_vol: 0.0,
                ..config()
            }
            .validate()
            .is_err()
        );
        assert!(
            IvConfig {
                max_iterations: 0,
                ..config()
            }
            .validate()
            .is_err()
        );
        assert!(
            IvConfig {
                weight_eps: f64::NAN,
                ..config()
            }
            .validate()
            .is_err()
        );
        // 1 / eps^2 overflows: a bid equal to the ask would weigh infinity.
        assert!(
            IvConfig {
                weight_eps: 1e-200,
                ..config()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_band_weight_is_finite_and_positive_or_a_failure() {
        assert!(
            band_weight(0.01, 0.03, 1e-8).is_ok_and(|w| (w - 2500.0).abs() < 1e-9),
            "{:?}",
            band_weight(0.01, 0.03, 1e-8)
        );
        // A zero band takes eps.
        assert!(band_weight(0.02, 0.02, 1e-8).is_ok_and(|w| (w / 1e16 - 1.0).abs() < 1e-15));
        // A band whose square overflows gives weight 0: a failure.
        assert!(matches!(
            band_weight(0.0, 1e200, 1e-8),
            Err(IvFailure::Numerical(_))
        ));
        assert!(matches!(
            band_weight(0.0, f64::INFINITY, 1e-8),
            Err(IvFailure::Numerical(_))
        ));
        // eps below the validated range: an infinite weight is a failure.
        assert!(matches!(
            band_weight(0.02, 0.02, 1e-200),
            Err(IvFailure::Numerical(_))
        ));
    }
}
