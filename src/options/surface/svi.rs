//! Raw SVI slices and a surface of slices, linear in total variance between
//! expiries.
//!
//! A slice is the total implied variance of one expiry as a function of the
//! log forward moneyness `k = ln(K / F)`:
//!
//! ```text
//! w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))
//! ```
//!
//! (Gatheral 2004; Gatheral and Jacquier, *Arbitrage-free SVI volatility
//! surfaces*, <https://arxiv.org/abs/1204.0646>).
//!
//! # Domain
//!
//! A slice accepts the raw SVI domain with a strictly positive variance:
//! `b >= 0`, `|rho| <= 1` (the boundary `rho = ±1` included), `sigma > 0`,
//! Lee's moment bound `b (1 + |rho|) <= 2`, and minimum variance
//! `a + b sigma sqrt(1 - rho^2) > 0`. With `|rho| = 1` the minimum is `a`,
//! which `w` approaches in one wing, so `a > 0` is required there. Nothing
//! is clamped: a parameter outside the domain is an error.
//!
//! # Interpolation and extrapolation
//!
//! - Between expiries the surface is linear in `w` at fixed `k`, and
//!   `dw/dtau` is the slope of the segment. On a node it is the slope of the
//!   segment towards shorter expiries (the direction calendar time moves
//!   in); on the first node, the only segment it has.
//! - Beyond the last expiry there is no extrapolation: an error.
//! - Below the first expiry the default is an error too
//!   ([`ShortExpiryPolicy::Reject`]). [`ShortExpiryPolicy::ConstantVol`]
//!   opts into `w(k, tau) = w_1(k) tau / tau_1`, the implied volatility of
//!   the first slice at each `k`, marked [`Region::Extrapolated`].
//! - In `k`, each slice follows its own SVI wings, which tend to straight
//!   lines of slope at most 2 (Lee) as `|k|` grows; they are not exactly
//!   linear pieces. A query outside a slice's declared calibrated range is
//!   [`Region::Extrapolated`].
//!
//! # Numbers
//!
//! The model is evaluated in `f64`, an explicit exception for option
//! models: every input and result is checked finite, and a non-finite value
//! is an error. Results are deterministic on one platform and math library
//! within the stated tolerances; bit identity across architectures is not
//! promised.

use super::{TotalVariance, VolSurface, check_point};
use crate::types::error::{MMError, MMResult};

/// Lee's bound on the asymptotic slope of `w(k)`: `b (1 + |rho|) <= 2`.
pub const LEE_SLOPE_BOUND: f64 = 2.0;

/// Relative tolerance for matching `tau` to a slice node.
const NODE_TOLERANCE: f64 = 1e-12;

/// A declared range of `k`, for example the calibrated range of a slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KRange {
    lo: f64,
    hi: f64,
}

impl KRange {
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] unless both ends are finite and
    /// `lo <= hi`.
    pub fn new(lo: f64, hi: f64) -> MMResult<Self> {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(MMError::InvalidConfiguration(format!(
                "k range must be finite with lo <= hi, got [{lo}, {hi}]"
            )));
        }
        Ok(Self { lo, hi })
    }

    /// Lower end.
    #[must_use]
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Upper end.
    #[must_use]
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// Whether `k` is inside the range, ends included.
    #[must_use]
    pub fn contains(&self, k: f64) -> bool {
        self.lo <= k && k <= self.hi
    }
}

/// Raw SVI parameters of one expiry, with the `k` range it was calibrated
/// on. `tau` is in years.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviSlice {
    tau: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
    k_range: KRange,
}

impl SviSlice {
    /// A slice in the domain described in the [module docs](self).
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a parameter outside the domain
    /// or not finite.
    pub fn new(
        tau: f64,
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
        k_range: KRange,
    ) -> MMResult<Self> {
        let invalid =
            |what: String| Err(MMError::InvalidConfiguration(format!("svi slice: {what}")));
        for (name, v) in [
            ("tau", tau),
            ("a", a),
            ("b", b),
            ("rho", rho),
            ("m", m),
            ("sigma", sigma),
        ] {
            if !v.is_finite() {
                return invalid(format!("{name} must be finite, got {v}"));
            }
        }
        if tau <= 0.0 {
            return invalid(format!("tau must be positive, got {tau}"));
        }
        if b < 0.0 {
            return invalid(format!("b must be non-negative, got {b}"));
        }
        if rho.abs() > 1.0 {
            return invalid(format!("|rho| must be at most 1, got {rho}"));
        }
        if sigma <= 0.0 {
            return invalid(format!("sigma must be positive, got {sigma}"));
        }
        let slope = b * (1.0 + rho.abs());
        if slope > LEE_SLOPE_BOUND {
            return invalid(format!(
                "b (1 + |rho|) = {slope} exceeds Lee's bound {LEE_SLOPE_BOUND}"
            ));
        }
        let slice = Self {
            tau,
            a,
            b,
            rho,
            m,
            sigma,
            k_range,
        };
        let minimum = slice.minimum();
        if !(minimum.is_finite() && minimum > 0.0) {
            return invalid(format!(
                "minimum variance a + b sigma sqrt(1 - rho^2) = {minimum} must be positive"
            ));
        }
        Ok(slice)
    }

    /// Expiry in years.
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// `a`.
    #[must_use]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// `b`.
    #[must_use]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// `rho`.
    #[must_use]
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// `m`.
    #[must_use]
    pub fn m(&self) -> f64 {
        self.m
    }

    /// `sigma`.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The calibrated range of `k`.
    #[must_use]
    pub fn k_range(&self) -> KRange {
        self.k_range
    }

    /// The smallest total variance of the slice, `a + b sigma sqrt(1 - rho^2)`.
    #[must_use]
    pub fn minimum(&self) -> f64 {
        let root = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        self.a + self.b * self.sigma * root
    }

    /// `b (1 + |rho|)`, the asymptotic slope of the steeper wing.
    #[must_use]
    pub fn lee_slope(&self) -> f64 {
        self.b * (1.0 + self.rho.abs())
    }

    /// `w`, `dw/dk` and `d2w/dk2` at `k`.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] if `k` or a result is not finite.
    pub fn eval(&self, k: f64) -> MMResult<(f64, f64, f64)> {
        if !k.is_finite() {
            return Err(MMError::NumericalError(format!(
                "k must be finite, got {k}"
            )));
        }
        let x = k - self.m;
        // sqrt(x^2 + sigma^2) without overflow for a large |x|.
        let r = x.hypot(self.sigma);
        let w = self.a + self.b * (self.rho * x + r);
        let w_k = self.b * (self.rho + x / r);
        let w_kk = self.b * self.sigma * self.sigma / (r * r * r);
        if !(w.is_finite() && w_k.is_finite() && w_kk.is_finite()) {
            return Err(MMError::NumericalError(format!(
                "svi slice at k {k} is not finite: w {w}, w_k {w_k}, w_kk {w_kk}"
            )));
        }
        Ok((w, w_k, w_kk))
    }

    fn params(&self) -> String {
        format!(
            "tau={:.17e};a={:.17e};b={:.17e};rho={:.17e};m={:.17e};sigma={:.17e};k_lo={:.17e};k_hi={:.17e}",
            self.tau,
            self.a,
            self.b,
            self.rho,
            self.m,
            self.sigma,
            self.k_range.lo,
            self.k_range.hi
        )
    }
}

/// What the surface does below its first expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShortExpiryPolicy {
    /// `tau` below the first slice is an error (the default).
    #[default]
    Reject,
    /// `w(k, tau) = w_1(k) tau / tau_1`: the first slice's implied
    /// volatility at each `k`. Every such point is
    /// [`Region::Extrapolated`].
    ConstantVol,
}

/// Whether a point is backed by calibrated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Inside the calibrated `k` range of every slice used, and inside the
    /// expiry range.
    Calibrated,
    /// Outside a slice's calibrated `k` range, or below the first expiry
    /// under [`ShortExpiryPolicy::ConstantVol`].
    Extrapolated,
}

/// Raw SVI slices on increasing expiries, linear in `w` between them. See
/// the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct SviSurface {
    slices: Vec<SviSlice>,
    short_expiry: ShortExpiryPolicy,
}

/// The slices that answer a query at `tau`.
enum Bracket<'a> {
    /// Below the first slice, under [`ShortExpiryPolicy::ConstantVol`].
    Short(&'a SviSlice),
    /// Between (or on) `lo` and `hi`, `x` the weight of `hi`.
    Segment {
        lo: &'a SviSlice,
        hi: &'a SviSlice,
        x: f64,
    },
}

impl SviSurface {
    /// At least two slices with strictly increasing `tau`.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] otherwise.
    pub fn new(slices: Vec<SviSlice>, short_expiry: ShortExpiryPolicy) -> MMResult<Self> {
        if slices.len() < 2 {
            return Err(MMError::InvalidConfiguration(format!(
                "an svi surface needs at least two slices, got {}",
                slices.len()
            )));
        }
        if slices.windows(2).any(|p| match p {
            [lo, hi] => hi.tau <= lo.tau,
            _ => false,
        }) {
            return Err(MMError::InvalidConfiguration(
                "svi slices must have strictly increasing tau".to_string(),
            ));
        }
        Ok(Self {
            slices,
            short_expiry,
        })
    }

    /// The slices, by increasing expiry.
    #[must_use]
    pub fn slices(&self) -> &[SviSlice] {
        &self.slices
    }

    /// The short expiry policy.
    #[must_use]
    pub fn short_expiry(&self) -> ShortExpiryPolicy {
        self.short_expiry
    }

    /// The parameters as canonical text (`{:.17e}` per value), stable for
    /// hashing on one platform.
    #[must_use]
    pub fn params(&self) -> String {
        let policy = match self.short_expiry {
            ShortExpiryPolicy::Reject => "reject",
            ShortExpiryPolicy::ConstantVol => "constant-vol",
        };
        let slices: Vec<String> = self.slices.iter().map(SviSlice::params).collect();
        format!("svi/1;short_expiry={policy};slices=[{}]", slices.join("|"))
    }

    /// Whether `(k, tau)` is backed by calibrated data.
    ///
    /// # Errors
    ///
    /// As [`VolSurface::total_variance`] for a `tau` outside the surface.
    pub fn region(&self, k: f64, tau: f64) -> MMResult<Region> {
        let calibrated = match self.bracket(k, tau)? {
            Bracket::Short(_) => false,
            Bracket::Segment { lo, hi, x } => {
                let lo_ok = x >= 1.0 || lo.k_range.contains(k);
                let hi_ok = x <= 0.0 || hi.k_range.contains(k);
                lo_ok && hi_ok
            }
        };
        Ok(if calibrated {
            Region::Calibrated
        } else {
            Region::Extrapolated
        })
    }

    fn bracket(&self, k: f64, tau: f64) -> MMResult<Bracket<'_>> {
        check_point(k, tau)?;
        let (Some(first), Some(last)) = (self.slices.first(), self.slices.last()) else {
            return Err(MMError::InvalidConfiguration(
                "svi surface without slices".to_string(),
            ));
        };
        let tol = NODE_TOLERANCE * last.tau;
        if tau > last.tau + tol {
            return Err(MMError::InvalidMarketState(format!(
                "tau {tau} is beyond the last expiry {}: no extrapolation",
                last.tau
            )));
        }
        if tau < first.tau - tol {
            return match self.short_expiry {
                ShortExpiryPolicy::Reject => Err(MMError::InvalidMarketState(format!(
                    "tau {tau} is below the first expiry {}",
                    first.tau
                ))),
                ShortExpiryPolicy::ConstantVol => Ok(Bracket::Short(first)),
            };
        }
        // The first slice at or beyond tau (within the tolerance).
        let j = self
            .slices
            .iter()
            .position(|s| s.tau >= tau - tol)
            .unwrap_or(self.slices.len().saturating_sub(1));
        let node = self
            .slices
            .get(j)
            .is_some_and(|s| (s.tau - tau).abs() <= tol);
        let (i_lo, i_hi) = if j == 0 { (0, 1) } else { (j - 1, j) };
        let (Some(lo), Some(hi)) = (self.slices.get(i_lo), self.slices.get(i_hi)) else {
            return Err(MMError::InvalidConfiguration(
                "svi surface bracket out of range".to_string(),
            ));
        };
        let x = if node {
            // On a node: the value is the node's; the segment is towards
            // shorter expiries (or the only one, on the first node).
            if j == 0 { 0.0 } else { 1.0 }
        } else {
            (tau - lo.tau) / (hi.tau - lo.tau)
        };
        Ok(Bracket::Segment { lo, hi, x })
    }
}

impl VolSurface for SviSurface {
    fn total_variance(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        match self.bracket(k, tau)? {
            Bracket::Short(first) => {
                let (w1, w1_k, w1_kk) = first.eval(k)?;
                let scale = tau / first.tau;
                TotalVariance::new(w1 * scale, w1_k * scale, w1_kk * scale, w1 / first.tau)
            }
            Bracket::Segment { lo, hi, x } => {
                let (wl, wl_k, wl_kk) = lo.eval(k)?;
                let (wh, wh_k, wh_kk) = hi.eval(k)?;
                let w = (1.0 - x) * wl + x * wh;
                let w_k = (1.0 - x) * wl_k + x * wh_k;
                let w_kk = (1.0 - x) * wl_kk + x * wh_kk;
                let w_tau = (wh - wl) / (hi.tau - lo.tau);
                TotalVariance::new(w, w_k, w_kk, w_tau)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range() -> KRange {
        KRange::new(-0.5, 0.5).unwrap()
    }

    fn slice(tau: f64, a: f64) -> SviSlice {
        SviSlice::new(tau, a, 0.1, -0.5, 0.0, 0.2, range()).unwrap()
    }

    fn surface(policy: ShortExpiryPolicy) -> SviSurface {
        SviSurface::new(
            vec![slice(0.25, 0.01), slice(0.5, 0.02), slice(1.0, 0.04)],
            policy,
        )
        .unwrap()
    }

    #[test]
    fn test_svi_slice_out_of_domain_returns_err() {
        let r = range();
        assert!(SviSlice::new(0.0, 0.01, 0.1, -0.5, 0.0, 0.2, r).is_err());
        assert!(SviSlice::new(0.5, 0.01, -0.1, -0.5, 0.0, 0.2, r).is_err());
        assert!(SviSlice::new(0.5, 0.01, 0.1, -1.01, 0.0, 0.2, r).is_err());
        assert!(SviSlice::new(0.5, 0.01, 0.1, -0.5, 0.0, 0.0, r).is_err());
        assert!(SviSlice::new(0.5, 0.01, 1.5, -0.5, 0.0, 0.2, r).is_err());
        assert!(SviSlice::new(0.5, -0.2, 0.1, -0.5, 0.0, 0.2, r).is_err());
        assert!(SviSlice::new(0.5, f64::NAN, 0.1, -0.5, 0.0, 0.2, r).is_err());
        // |rho| = 1 needs a > 0: w tends to a in one wing.
        assert!(SviSlice::new(0.5, 0.0, 0.1, -1.0, 0.0, 0.2, r).is_err());
    }

    #[test]
    fn test_svi_slice_rho_boundary_with_positive_a_is_accepted() {
        let s = SviSlice::new(0.02, 0.001, 0.05, -1.0, 0.0, 0.1, range()).unwrap();
        assert_eq!(s.minimum(), 0.001);
        // With rho = -1 the right wing tends to a and the left wing grows
        // with slope 2b.
        let (w_right, _, _) = s.eval(50.0).unwrap();
        assert!((w_right - 0.001).abs() < 1e-5);
        let (_, w_k_left, _) = s.eval(-50.0).unwrap();
        assert!((w_k_left + 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_svi_slice_derivatives_match_finite_differences() {
        let s = slice(0.5, 0.02);
        let h = 1e-5;
        for k in [-1.0, -0.3, 0.0, 0.2, 1.5] {
            let (_, w_k, w_kk) = s.eval(k).unwrap();
            let (wp, wp_k, _) = s.eval(k + h).unwrap();
            let (wm, wm_k, _) = s.eval(k - h).unwrap();
            assert!((w_k - (wp - wm) / (2.0 * h)).abs() < 1e-8, "w_k at {k}");
            assert!(
                (w_kk - (wp_k - wm_k) / (2.0 * h)).abs() < 1e-7,
                "w_kk at {k}"
            );
        }
    }

    #[test]
    fn test_svi_surface_needs_two_increasing_slices() {
        let p = ShortExpiryPolicy::Reject;
        assert!(SviSurface::new(vec![slice(0.5, 0.02)], p).is_err());
        assert!(SviSurface::new(vec![slice(0.5, 0.02), slice(0.5, 0.03)], p).is_err());
        assert!(SviSurface::new(vec![slice(1.0, 0.04), slice(0.5, 0.02)], p).is_err());
    }

    #[test]
    fn test_svi_surface_nodes_and_segments_follow_the_convention() {
        let s = surface(ShortExpiryPolicy::Reject);
        let k = 0.1;
        let (w1, _, _) = slice(0.25, 0.01).eval(k).unwrap();
        let (w2, _, _) = slice(0.5, 0.02).eval(k).unwrap();
        let (w3, _, _) = slice(1.0, 0.04).eval(k).unwrap();
        // First node: its value, the only segment's slope.
        let first = s.total_variance(k, 0.25).unwrap();
        assert!((first.w() - w1).abs() < 1e-15);
        assert!((first.w_tau() - (w2 - w1) / 0.25).abs() < 1e-12);
        // Interior node: its value, the segment towards shorter expiries.
        let mid = s.total_variance(k, 0.5).unwrap();
        assert!((mid.w() - w2).abs() < 1e-15);
        assert!((mid.w_tau() - (w2 - w1) / 0.25).abs() < 1e-12);
        // Last node: its value, the segment towards shorter expiries.
        let last = s.total_variance(k, 1.0).unwrap();
        assert!((last.w() - w3).abs() < 1e-15);
        assert!((last.w_tau() - (w3 - w2) / 0.5).abs() < 1e-12);
        // Between nodes: linear in w.
        let between = s.total_variance(k, 0.75).unwrap();
        assert!((between.w() - 0.5 * (w2 + w3)).abs() < 1e-15);
    }

    #[test]
    fn test_svi_surface_outside_expiries_returns_err_unless_opted_in() {
        let reject = surface(ShortExpiryPolicy::Reject);
        assert!(reject.total_variance(0.0, 1.5).is_err());
        assert!(reject.total_variance(0.0, 0.1).is_err());
        assert!(reject.total_variance(0.0, 0.0).is_err());
        assert!(reject.total_variance(f64::NAN, 0.5).is_err());

        let flat = surface(ShortExpiryPolicy::ConstantVol);
        assert!(flat.total_variance(0.0, 1.5).is_err());
        let short = flat.total_variance(0.1, 0.1).unwrap();
        let (w1, _, _) = slice(0.25, 0.01).eval(0.1).unwrap();
        assert!((short.w() - w1 * 0.1 / 0.25).abs() < 1e-15);
        assert!((short.w_tau() - w1 / 0.25).abs() < 1e-15);
        assert_eq!(flat.region(0.1, 0.1).unwrap(), Region::Extrapolated);
    }

    #[test]
    fn test_svi_surface_region_marks_k_outside_the_calibrated_range() {
        let s = surface(ShortExpiryPolicy::Reject);
        assert_eq!(s.region(0.0, 0.75).unwrap(), Region::Calibrated);
        assert_eq!(s.region(0.6, 0.75).unwrap(), Region::Extrapolated);
        assert_eq!(s.region(-0.5, 0.25).unwrap(), Region::Calibrated);
    }

    #[test]
    fn test_svi_surface_params_are_canonical() {
        let s = surface(ShortExpiryPolicy::Reject);
        let text = s.params();
        assert!(text.starts_with("svi/1;short_expiry=reject;slices=[tau=2.50000000000000000e-1;"));
        assert_eq!(text, surface(ShortExpiryPolicy::Reject).params());
    }
}
