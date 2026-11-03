//! Calibration of one raw SVI slice to observed total variances.
//!
//! The fit is the quasi-explicit method of Zeliade (De Marco and Martini,
//! 2009), as in the arbitrage-rs reference (`experiments/svi.py`). With
//! `y = (k - m) / sigma` a slice is linear in `(a, d, c) = (a, rho b sigma,
//! b sigma)`:
//!
//! ```text
//! w = a + d y + c sqrt(y^2 + 1)
//! ```
//!
//! so for each `(m, sigma)` of a deterministic grid the inner problem is a
//! weighted least squares over the polytope
//!
//! ```text
//! c >= 0,  |d| <= c (|rho| <= 1),  |d| <= 2 sigma - c (Lee),  a_min <= a <= a_max
//! ```
//!
//! with `a_min > 0` declared in [`FitConfig`] and `a_max` the largest
//! observed `w` (or `a_min`, if larger). That box is a declared subset of
//! the raw SVI domain of [`SviSlice`]: every fitted slice has a positive
//! minimum variance, also with `rho = ±1`.
//!
//! The inner minimum is found by trying every face of the polytope (every
//! set of at most three active constraints): the equality constrained least
//! squares on the face is solved through its KKT system by Gaussian
//! elimination with partial pivoting, a face whose system is singular
//! within the pivot tolerance is skipped, and the best point that satisfies
//! every constraint within the feasibility tolerance wins. This is the
//! solution of each face within those tolerances, not exact arithmetic.
//!
//! The search domain is `m` in `[k_min - R, k_max + R]` and `sigma` in
//! `[sigma_lo R, sigma_hi R]` (log spaced), `R = k_max - k_min`, and no
//! window leaves it. The outer grid starts on the whole domain. When its
//! best point is on an edge of the window inside the domain, the window is
//! recentred there and doubled along that axis (up to the domain), so a
//! minimum along a long `(m, sigma)` valley is reached in a few moves; once
//! the best point is interior, the window is refined around it, a
//! refinement counting only when it reaches a window smaller than any
//! before. A best point on an edge at the end, on the domain's boundary
//! or with no recentre left, is [`FitFailure::GridEdge`]: the minimum may
//! lie outside the search. Every loop is bounded and nothing is random.
//!
//! The parameters at an active bound are set from the constraint itself
//! (`rho = ±1`, `b (1 + |rho|) = 2`, `a = a_min` or `a_max`), and the
//! [`SliceFit`] lists the active constraints. A result outside the domain is
//! a [`FitFailure`], never clamped.

use super::svi::{KRange, LEE_SLOPE_BOUND, SviSlice};
use crate::types::error::{MMError, MMResult};

/// One observed point of a slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceObservation {
    k: f64,
    w: f64,
    weight: f64,
}

impl SliceObservation {
    /// `w >= 0` (a quote at its lower bound has `w = 0`) and `weight > 0`,
    /// all finite.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidMarketState`] otherwise.
    pub fn new(k: f64, w: f64, weight: f64) -> MMResult<Self> {
        if !(k.is_finite() && w.is_finite() && w >= 0.0 && weight.is_finite() && weight > 0.0) {
            return Err(MMError::InvalidMarketState(format!(
                "slice observation needs finite k, w >= 0 and weight > 0, got k {k}, w {w}, weight {weight}"
            )));
        }
        Ok(Self { k, w, weight })
    }

    /// Log forward moneyness.
    #[must_use]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// Observed total variance.
    #[must_use]
    pub fn w(&self) -> f64 {
        self.w
    }

    /// Weight in the least squares.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// Settings of the slice fit.
///
/// # When to expect [`FitFailure::GridEdge`]
///
/// The search stays inside `sigma` in `[sigma_lo R, sigma_hi R]`
/// (`R = k_max - k_min`). With noisy data, a slice whose true `sigma` is
/// near `sigma_lo R`, or a smile so flat over the quoted strikes that its
/// best `sigma` is near `sigma_hi R`, can have its least squares optimum on
/// that boundary: the fit then fails with `GridEdge`, by design, rather
/// than return a boundary point as a fit. In one such independent sweep
/// (400 slices, relative noise of `1e-3` on `w`), 9 failed that way and
/// none succeeded wrongly.
///
/// # Cost
///
/// Each pass evaluates `grid^2` points, each solving at most 64 small KKT
/// systems, over at most `refinements + 2 max_recentres + 1` passes. With
/// the defaults, an exact slice of 15 to 17 quotes takes about 0.02 s on
/// average and under 0.1 s in a release build; noisy slices, which use
/// more recentres, up to about 0.3 s. Unoptimised builds are about 20
/// times slower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitConfig {
    /// Points per axis of the outer `(m, sigma)` grid, at least 5.
    pub grid: usize,
    /// Times the grid is refined around its best point to a window smaller
    /// than any before (a refinement that only wins back the size a
    /// recentre doubled does not count).
    pub refinements: usize,
    /// Times the grid may be moved when its best point is on the edge.
    pub max_recentres: usize,
    /// Lower end of `sigma`, as a fraction of the `k` span.
    pub sigma_lo: f64,
    /// Upper end of `sigma`, as a fraction of the `k` span.
    pub sigma_hi: f64,
    /// Lower bound of `a`, positive.
    pub a_min: f64,
    /// Fewest observations a slice is fitted with, at least 5.
    pub min_observations: usize,
    /// Largest root mean square residual, in implied volatility, a fit is
    /// accepted with.
    pub max_rmse_vol: f64,
    /// Relative pivot tolerance of the KKT solves.
    pub pivot_tolerance: f64,
    /// Relative feasibility tolerance of a face's solution.
    pub feasibility_tolerance: f64,
    /// Largest weight a quote may have, as a multiple of its slice's median
    /// weight; a larger one is capped there and reported
    /// ([`SliceFit::capped`]), so one very tight quote cannot decide the
    /// slice alone. A cap beyond the largest `f64` binds no weight.
    pub max_weight_ratio: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            grid: 21,
            refinements: 12,
            max_recentres: 40,
            sigma_lo: 0.005,
            sigma_hi: 10.0,
            a_min: 1e-8,
            min_observations: 5,
            max_rmse_vol: 0.02,
            pivot_tolerance: 1e-12,
            feasibility_tolerance: 1e-12,
            max_weight_ratio: 10.0,
        }
    }
}

impl FitConfig {
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a value outside its range.
    pub fn validate(&self) -> MMResult<()> {
        let bad = |what: String| Err(MMError::InvalidConfiguration(format!("svi fit: {what}")));
        if self.grid < 5 {
            return bad(format!("grid must be at least 5, got {}", self.grid));
        }
        if self.min_observations < 5 {
            return bad(format!(
                "min_observations must be at least 5, got {}",
                self.min_observations
            ));
        }
        for (name, v) in [
            ("sigma_lo", self.sigma_lo),
            ("sigma_hi", self.sigma_hi),
            ("a_min", self.a_min),
            ("max_rmse_vol", self.max_rmse_vol),
            ("pivot_tolerance", self.pivot_tolerance),
            ("feasibility_tolerance", self.feasibility_tolerance),
            ("max_weight_ratio", self.max_weight_ratio),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return bad(format!("{name} must be finite and positive, got {v}"));
            }
        }
        if self.sigma_lo >= self.sigma_hi {
            return bad(format!(
                "sigma_lo {} must be below sigma_hi {}",
                self.sigma_lo, self.sigma_hi
            ));
        }
        Ok(())
    }
}

/// A constraint of the inner problem that holds with equality at the fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActiveConstraint {
    /// `c = 0`: `b = 0`, a flat slice.
    FlatSlice,
    /// `d = c`: `rho = 1`.
    RhoUpper,
    /// `d = -c`: `rho = -1`.
    RhoLower,
    /// `d + c = 2 sigma`: Lee's bound with `rho >= 0`.
    LeeUpper,
    /// `c - d = 2 sigma`: Lee's bound with `rho <= 0`.
    LeeLower,
    /// `a = a_min`.
    AMin,
    /// `a = a_max`.
    AMax,
}

const CONSTRAINTS: [ActiveConstraint; 7] = [
    ActiveConstraint::FlatSlice,
    ActiveConstraint::RhoUpper,
    ActiveConstraint::RhoLower,
    ActiveConstraint::LeeUpper,
    ActiveConstraint::LeeLower,
    ActiveConstraint::AMin,
    ActiveConstraint::AMax,
];

/// The residual of one observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Residual {
    /// Log forward moneyness.
    pub k: f64,
    /// Observed total variance.
    pub w_observed: f64,
    /// Fitted total variance.
    pub w_fitted: f64,
    /// Observed minus fitted implied volatility.
    pub vol_residual: f64,
}

/// A fitted slice and how it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceFit {
    /// The slice, with the observed `k` range as its calibrated range.
    pub slice: SviSlice,
    /// Weighted sum of squared residuals in `w` of the KKT solution, with
    /// the weights scaled so the largest is 1.
    pub objective: f64,
    /// The same sum for the slice's parameters, which come from the active
    /// constraints where one is active: within the tolerances of
    /// `objective`, or the fit fails.
    pub objective_params: f64,
    /// `2 - b (1 + |rho|)`: the distance to Lee's bound.
    pub lee_margin: f64,
    /// `a - a_min`.
    pub a_margin: f64,
    /// Root mean square residual in implied volatility.
    pub rmse_vol: f64,
    /// Constraints active at the fit.
    pub active: Vec<ActiveConstraint>,
    /// Residual of each observation, in input order.
    pub residuals: Vec<Residual>,
    /// Positions, in the observations given, of the quotes whose weight was
    /// capped at [`FitConfig::max_weight_ratio`] times the median.
    pub capped: Vec<usize>,
}

/// Why a slice could not be fitted.
#[derive(Debug, Clone, PartialEq)]
pub enum FitFailure {
    /// The configuration or the input is invalid.
    Invalid(String),
    /// Fewer observations than [`FitConfig::min_observations`].
    TooFewObservations {
        /// Observations given.
        observed: usize,
        /// Minimum required.
        required: usize,
    },
    /// The observations span no range of `k`.
    NoSpan,
    /// No face of the polytope gave a feasible point.
    NoFeasibleFace,
    /// The best point stayed on the grid's edge after every recentre.
    GridEdge,
    /// The fitted parameters are outside the slice domain.
    OutOfDomain(String),
    /// The residual is above [`FitConfig::max_rmse_vol`].
    RmseAboveMax {
        /// The fit's root mean square residual in volatility.
        rmse_vol: f64,
        /// The configured maximum.
        max: f64,
    },
}

impl std::fmt::Display for FitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid fit input: {why}"),
            Self::TooFewObservations { observed, required } => {
                write!(f, "{observed} observations, at least {required} required")
            }
            Self::NoSpan => write!(f, "observations span no range of k"),
            Self::NoFeasibleFace => write!(f, "no feasible face"),
            Self::GridEdge => write!(f, "best point on the grid edge after every recentre"),
            Self::OutOfDomain(why) => write!(f, "fitted slice outside the domain: {why}"),
            Self::RmseAboveMax { rmse_vol, max } => {
                write!(f, "rmse in vol {rmse_vol} above the maximum {max}")
            }
        }
    }
}

/// The inner solution at one `(m, sigma)`.
#[derive(Debug, Clone)]
struct Inner {
    objective: f64,
    x: [f64; 3],
    active: Vec<usize>,
}

/// Solves `m x = rhs` in place by Gaussian elimination with partial
/// pivoting. `None` when a pivot is below `tol` times the largest entry.
fn solve_linear(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>, tol: f64) -> Option<Vec<f64>> {
    let n = rhs.len();
    let scale = m
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&i, &j| {
            let a = m.get(i).and_then(|r| r.get(col)).map_or(0.0, |v| v.abs());
            let b = m.get(j).and_then(|r| r.get(col)).map_or(0.0, |v| v.abs());
            a.total_cmp(&b)
        })?;
        let pivot = *m.get(pivot_row)?.get(col)?;
        if pivot.abs() <= tol * scale {
            return None;
        }
        m.swap(col, pivot_row);
        rhs.swap(col, pivot_row);
        let pivot_values = m.get(col)?.clone();
        let pivot_rhs = *rhs.get(col)?;
        for row in col.saturating_add(1)..n {
            let factor = *m.get(row)?.get(col)? / pivot;
            if factor == 0.0 {
                continue;
            }
            let target = m.get_mut(row)?;
            for (t, p) in target.iter_mut().zip(pivot_values.iter()).skip(col) {
                *t -= factor * p;
            }
            *rhs.get_mut(row)? -= factor * pivot_rhs;
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let r = m.get(row)?;
        let tail: f64 = r
            .iter()
            .zip(x.iter())
            .skip(row.saturating_add(1))
            .map(|(a, b)| a * b)
            .sum();
        let value = (*rhs.get(row)? - tail) / *r.get(row)?;
        if !value.is_finite() {
            return None;
        }
        *x.get_mut(row)? = value;
    }
    Some(x)
}

/// `G x <= h` of the polytope, one row per [`CONSTRAINTS`] entry.
fn constraints(sigma: f64, a_min: f64, a_max: f64) -> [([f64; 3], f64); 7] {
    let lee = LEE_SLOPE_BOUND * sigma;
    [
        ([0.0, 0.0, -1.0], 0.0),
        ([0.0, 1.0, -1.0], 0.0),
        ([0.0, -1.0, -1.0], 0.0),
        ([0.0, 1.0, 1.0], lee),
        ([0.0, -1.0, 1.0], lee),
        ([-1.0, 0.0, 0.0], -a_min),
        ([1.0, 0.0, 0.0], a_max),
    ]
}

/// The exact (within tolerances) minimiser at `(m, sigma)`.
fn inner(
    obs: &[SliceObservation],
    m: f64,
    sigma: f64,
    a_max: f64,
    config: &FitConfig,
) -> Option<Inner> {
    // Normal equations of the weighted least squares.
    let mut q = [[0.0_f64; 3]; 3];
    let mut lin = [0.0_f64; 3];
    for o in obs {
        let y = (o.k - m) / sigma;
        let row = [1.0, y, y.hypot(1.0)];
        for (i, ri) in row.iter().enumerate() {
            if let Some(li) = lin.get_mut(i) {
                *li += o.weight * ri * o.w;
            }
            if let Some(qi) = q.get_mut(i) {
                for (qij, rj) in qi.iter_mut().zip(row.iter()) {
                    *qij += o.weight * ri * rj;
                }
            }
        }
    }
    let rows = constraints(sigma, config.a_min, a_max);
    let h_scale = rows.iter().fold(1.0_f64, |acc, (_, h)| acc.max(h.abs()));
    let mut best: Option<Inner> = None;
    for size in 0..=3_usize {
        for active in combinations(rows.len(), size) {
            let n = 3_usize.saturating_add(size);
            let mut kkt = vec![vec![0.0; n]; n];
            let mut rhs = vec![0.0; n];
            for i in 0..3 {
                for j in 0..3 {
                    if let Some(cell) = kkt.get_mut(i).and_then(|r| r.get_mut(j)) {
                        *cell = 2.0 * q.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    }
                }
                if let Some(cell) = rhs.get_mut(i) {
                    *cell = 2.0 * lin.get(i).copied().unwrap_or(0.0);
                }
            }
            for (slot, &c) in active.iter().enumerate() {
                let Some((g, h)) = rows.get(c) else { continue };
                let r = 3_usize.saturating_add(slot);
                for (j, gj) in g.iter().enumerate() {
                    if let Some(cell) = kkt.get_mut(r).and_then(|row| row.get_mut(j)) {
                        *cell = *gj;
                    }
                    if let Some(cell) = kkt.get_mut(j).and_then(|row| row.get_mut(r)) {
                        *cell = *gj;
                    }
                }
                if let Some(cell) = rhs.get_mut(r) {
                    *cell = *h;
                }
            }
            let Some(solution) = solve_linear(kkt, rhs, config.pivot_tolerance) else {
                continue;
            };
            let x = [
                solution.first().copied().unwrap_or(f64::NAN),
                solution.get(1).copied().unwrap_or(f64::NAN),
                solution.get(2).copied().unwrap_or(f64::NAN),
            ];
            let dot = |g: &[f64; 3]| g.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>();
            let feasible = rows
                .iter()
                .all(|(g, h)| dot(g) - h <= config.feasibility_tolerance * h_scale);
            if !feasible {
                continue;
            }
            // sum weight (row . x - w)^2, summed directly: the expanded
            // x' Q x - 2 lin . x + sum weight w^2 cancels when the fit is
            // nearly exact, which would turn the choice between grid points
            // into rounding noise.
            let objective: f64 = obs
                .iter()
                .map(|o| {
                    let y = (o.k - m) / sigma;
                    let r = x[0] + x[1] * y + x[2] * y.hypot(1.0) - o.w;
                    o.weight * r * r
                })
                .sum();
            if !objective.is_finite() {
                continue;
            }
            if best.as_ref().is_none_or(|b| objective < b.objective) {
                best = Some(Inner {
                    objective,
                    x,
                    active: active.clone(),
                });
            }
        }
        if size == 0 && best.is_some() {
            // The unconstrained minimiser is feasible: for a convex objective
            // it is the constrained one too.
            break;
        }
    }
    best
}

/// Every subset of `0..n` of size `size`, in lexicographic order.
fn combinations(n: usize, size: usize) -> Vec<Vec<usize>> {
    fn go(
        start: usize,
        n: usize,
        size: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            go(i.saturating_add(1), n, size, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    go(0, n, size, &mut Vec::new(), &mut out);
    out
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let last = n.saturating_sub(1).max(1);
    (0..n)
        .map(|i| {
            if i == last {
                hi
            } else {
                lo + (hi - lo) * i as f64 / last as f64
            }
        })
        .collect()
}

/// Fits one slice at expiry `tau` (years) to `observations`.
///
/// # Errors
///
/// A [`FitFailure`] with the reason; nothing is clamped or smoothed.
pub fn fit_slice(
    tau: f64,
    observations: &[SliceObservation],
    config: &FitConfig,
) -> Result<SliceFit, FitFailure> {
    config
        .validate()
        .map_err(|e| FitFailure::Invalid(e.to_string()))?;
    if !(tau.is_finite() && tau > 0.0) {
        return Err(FitFailure::Invalid(format!(
            "tau must be finite and positive, got {tau}"
        )));
    }
    if observations.len() < config.min_observations {
        return Err(FitFailure::TooFewObservations {
            observed: observations.len(),
            required: config.min_observations,
        });
    }
    let k_min = observations
        .iter()
        .map(|o| o.k)
        .fold(f64::INFINITY, f64::min);
    let k_max = observations
        .iter()
        .map(|o| o.k)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = k_max - k_min;
    if !(span.is_finite() && span > 0.0) {
        return Err(FitFailure::NoSpan);
    }
    let a_max = observations
        .iter()
        .map(|o| o.w)
        .fold(config.a_min, f64::max);
    // Weights capped at max_weight_ratio times the median, then scaled so
    // the largest is 1: the normal equations stay far from overflow
    // whatever the bid/ask widths were.
    let mut sorted: Vec<f64> = observations.iter().map(|o| o.weight).collect();
    sorted.sort_by(f64::total_cmp);
    let half = sorted.len() / 2;
    let median = match (sorted.get(half.wrapping_sub(1)), sorted.get(half)) {
        // lo + (hi - lo) / 2: no overflow for weights near f64::MAX.
        (Some(lo), Some(hi)) if sorted.len().is_multiple_of(2) => lo + 0.5 * (hi - lo),
        (_, Some(mid)) => *mid,
        _ => f64::NAN,
    };
    // A cap that overflows to infinity is above every representable weight,
    // so it binds none, as in exact arithmetic.
    let cap = config.max_weight_ratio * median;
    let capped: Vec<usize> = observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.weight > cap)
        .map(|(i, _)| i)
        .collect();
    let max_weight = observations
        .iter()
        .map(|o| o.weight.min(cap))
        .fold(0.0_f64, f64::max);
    let scaled: Vec<SliceObservation> = observations
        .iter()
        .map(|o| SliceObservation {
            k: o.k,
            w: o.w,
            weight: o.weight.min(cap) / max_weight,
        })
        .collect();
    if scaled
        .iter()
        .any(|o| !(o.weight.is_finite() && o.weight > 0.0))
    {
        return Err(FitFailure::Invalid(
            "weights do not scale to finite positive values".to_string(),
        ));
    }
    let observations = scaled.as_slice();
    let n = config.grid;
    let (mut m_lo, mut m_hi) = (k_min - span, k_max + span);
    let (mut s_lo, mut s_hi) = ((config.sigma_lo * span).ln(), (config.sigma_hi * span).ln());
    // The declared search domain: every window stays inside it, and a
    // recentred window never grows beyond it.
    let (dm_lo, dm_hi, ds_lo, ds_hi) = (m_lo, m_hi, s_lo, s_hi);
    let (hm_max, hs_max) = (0.5 * (m_hi - m_lo), 0.5 * (s_hi - s_lo));
    let mut best: Option<(Inner, f64, f64)> = None;
    let mut best_on_edge = true;
    let mut shrinks = 0_usize;
    let mut recentres = 0_usize;
    // The smallest window in m so far: a shrink counts as a refinement only
    // below it.
    let mut smallest = m_hi - m_lo;
    // Bounded: each pass recentres (at most max_recentres times, each at
    // most doubling the window), shrinks to a new smallest window (at most
    // refinements times), or shrinks back what recentres doubled (a shrink
    // divides the window by (grid - 1) / 4 >= 5 at the default grid, so at
    // most one such shrink per recentre there); the pass count caps it
    // whatever the grid.
    let passes = config
        .refinements
        .saturating_add(config.max_recentres.saturating_mul(2))
        .saturating_add(1);
    for _ in 0..passes {
        let ms = linspace(m_lo, m_hi, n);
        let ss = linspace(s_lo, s_hi, n);
        let mut pass_best: Option<(Inner, usize, usize)> = None;
        for (i, &m) in ms.iter().enumerate() {
            for (j, &ls) in ss.iter().enumerate() {
                if let Some(res) = inner(observations, m, ls.exp(), a_max, config)
                    && pass_best
                        .as_ref()
                        .is_none_or(|(b, _, _)| res.objective < b.objective)
                {
                    pass_best = Some((res, i, j));
                }
            }
        }
        let Some((res, i, j)) = pass_best else {
            return Err(FitFailure::NoFeasibleFace);
        };
        let last = n.saturating_sub(1);
        // On an edge of the window: open when the domain goes on beyond
        // it (a recentre can look there), closed on the domain's boundary.
        let (m_low, m_high) = (i == 0, i == last);
        let (s_low, s_high) = (j == 0, j == last);
        let m_edge = (m_low && m_lo > dm_lo) || (m_high && m_hi < dm_hi);
        let s_edge = (s_low && s_lo > ds_lo) || (s_high && s_hi < ds_hi);
        let on_edge = m_edge || s_edge;
        let on_any_edge = m_low || m_high || s_low || s_high;
        let (m_best, ls_best) = (
            ms.get(i).copied().unwrap_or(f64::NAN),
            ss.get(j).copied().unwrap_or(f64::NAN),
        );
        let improves = best
            .as_ref()
            .is_none_or(|(b, _, _)| res.objective <= b.objective);
        if improves {
            best = Some((res, m_best, ls_best));
            best_on_edge = on_any_edge;
        }
        let dm = ms.get(1).copied().unwrap_or(m_hi) - m_lo;
        let ds = ss.get(1).copied().unwrap_or(s_hi) - s_lo;
        if on_edge && improves && recentres < config.max_recentres {
            // Centre the window on the edge point, and double it (up to the
            // first window) along each axis whose edge the point is on:
            // along a narrow valley of (m, sigma) the minimum can lie many
            // shrunk windows away, and a window that only moved by half its
            // width would crawl there one step per recentre. Doubling
            // covers that distance in a few; the other axis keeps its
            // resolution, and the shrinks that follow recover the rest.
            recentres = recentres.saturating_add(1);
            let grow = |edge: bool, lo: f64, hi: f64, max: f64| {
                if edge {
                    (hi - lo).min(max)
                } else {
                    0.5 * (hi - lo)
                }
            };
            let (hm, hs) = (
                grow(m_edge, m_lo, m_hi, hm_max),
                grow(s_edge, s_lo, s_hi, hs_max),
            );
            (m_lo, m_hi) = ((m_best - hm).max(dm_lo), (m_best + hm).min(dm_hi));
            (s_lo, s_hi) = ((ls_best - hs).max(ds_lo), (ls_best + hs).min(ds_hi));
            continue;
        }
        if shrinks >= config.refinements {
            break;
        }
        let (Some((_, mb, lb)), true) = (best.as_ref(), dm > 0.0 && ds > 0.0) else {
            break;
        };
        (m_lo, m_hi) = ((mb - 2.0 * dm).max(dm_lo), (mb + 2.0 * dm).min(dm_hi));
        (s_lo, s_hi) = ((lb - 2.0 * ds).max(ds_lo), (lb + 2.0 * ds).min(ds_hi));
        if m_hi - m_lo < smallest {
            smallest = m_hi - m_lo;
            shrinks = shrinks.saturating_add(1);
        }
    }
    let Some((res, m, ls)) = best else {
        return Err(FitFailure::NoFeasibleFace);
    };
    // The best point on the edge of its window: on the domain's boundary,
    // or with no recentre left to look beyond it. The minimum may lie
    // outside the search.
    if best_on_edge {
        return Err(FitFailure::GridEdge);
    }
    let sigma = ls.exp();
    // Active: the face's constraints, and any other at its bound within
    // the feasibility tolerance (a face may meet a bound it does not hold
    // to rounding); each gives its parameter below.
    let rows = constraints(sigma, config.a_min, a_max);
    let h_scale = rows.iter().fold(1.0_f64, |acc, (_, h)| acc.max(h.abs()));
    let active: Vec<ActiveConstraint> = rows
        .iter()
        .enumerate()
        .filter(|(i, (g, h))| {
            res.active.contains(i) || {
                let gx: f64 = g.iter().zip(res.x.iter()).map(|(a, b)| a * b).sum();
                gx - h >= -config.feasibility_tolerance * h_scale
            }
        })
        .filter_map(|(i, _)| CONSTRAINTS.get(i).copied())
        .collect();
    let [a_raw, d, c] = res.x;
    let has = |c: ActiveConstraint| active.contains(&c);
    // Parameters at an active bound come from the constraint itself.
    let rho = if has(ActiveConstraint::RhoUpper) {
        1.0
    } else if has(ActiveConstraint::RhoLower) {
        -1.0
    } else if c > 0.0 {
        d / c
    } else {
        0.0
    };
    let b = if has(ActiveConstraint::FlatSlice) || c <= 0.0 {
        0.0
    } else if has(ActiveConstraint::LeeUpper) || has(ActiveConstraint::LeeLower) {
        LEE_SLOPE_BOUND / (1.0 + rho.abs())
    } else {
        c / sigma
    };
    let a = if has(ActiveConstraint::AMin) {
        config.a_min
    } else if has(ActiveConstraint::AMax) {
        a_max
    } else {
        a_raw
    };
    let range = KRange::new(k_min, k_max).map_err(|e| FitFailure::Invalid(e.to_string()))?;
    let slice = SviSlice::new(tau, a, b, rho, m, sigma, range)
        .map_err(|e| FitFailure::OutOfDomain(e.to_string()))?;
    let mut residuals = Vec::with_capacity(observations.len());
    let mut sum_sq = 0.0_f64;
    for o in observations {
        let (w_fit, _, _) = slice
            .eval(o.k)
            .map_err(|e| FitFailure::OutOfDomain(e.to_string()))?;
        let vol_residual = (o.w / tau).sqrt() - (w_fit / tau).sqrt();
        sum_sq += vol_residual * vol_residual;
        residuals.push(Residual {
            k: o.k,
            w_observed: o.w,
            w_fitted: w_fit,
            vol_residual,
        });
    }
    // The parameters taken from the active constraints must reproduce the
    // KKT solution of the face; a face is never projected into a success.
    let objective_params: f64 = observations
        .iter()
        .zip(residuals.iter())
        .map(|(o, r)| o.weight * (r.w_fitted - o.w) * (r.w_fitted - o.w))
        .sum();
    let total_weight: f64 = observations.iter().map(|o| o.weight).sum();
    let allowed =
        1e-9 * res.objective + config.feasibility_tolerance * total_weight * a_max * a_max;
    if !(objective_params.is_finite() && objective_params <= res.objective + allowed) {
        return Err(FitFailure::OutOfDomain(format!(
            "parameters from the active constraints give {objective_params}, the face solution {}",
            res.objective
        )));
    }
    let rmse_vol = (sum_sq / observations.len() as f64).sqrt();
    if !(rmse_vol.is_finite() && rmse_vol <= config.max_rmse_vol) {
        return Err(FitFailure::RmseAboveMax {
            rmse_vol,
            max: config.max_rmse_vol,
        });
    }
    Ok(SliceFit {
        lee_margin: LEE_SLOPE_BOUND - slice.lee_slope(),
        a_margin: slice.a() - config.a_min,
        slice,
        objective: res.objective,
        objective_params,
        rmse_vol,
        active,
        residuals,
        capped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observe(slice: &SviSlice, ks: &[f64]) -> Vec<SliceObservation> {
        ks.iter()
            .map(|&k| SliceObservation::new(k, slice.eval(k).unwrap().0, 1.0).unwrap())
            .collect()
    }

    fn ks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        linspace(lo, hi, n)
    }

    #[test]
    fn test_solve_linear_singular_returns_none() {
        assert!(
            solve_linear(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0], 1e-12).is_none()
        );
        let x = solve_linear(vec![vec![0.0, 1.0], vec![2.0, 0.0]], vec![3.0, 4.0], 1e-12).unwrap();
        assert_eq!(x, vec![2.0, 3.0]);
    }

    #[test]
    fn test_combinations_count_faces() {
        let total: usize = (0..=3).map(|s| combinations(7, s).len()).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_fit_config_invalid_values_return_err() {
        let base = FitConfig::default();
        assert!(base.validate().is_ok());
        assert!(FitConfig { grid: 4, ..base }.validate().is_err());
        assert!(FitConfig { a_min: 0.0, ..base }.validate().is_err());
        assert!(
            FitConfig {
                max_weight_ratio: f64::NAN,
                ..base
            }
            .validate()
            .is_err()
        );
        assert!(
            FitConfig {
                sigma_lo: 11.0,
                ..base
            }
            .validate()
            .is_err()
        );
        assert!(
            FitConfig {
                pivot_tolerance: f64::NAN,
                ..base
            }
            .validate()
            .is_err()
        );
    }

    /// 15 exact points of `truth` evenly over `[-r, r]`.
    fn exact_points(truth: &SviSlice, r: f64) -> Vec<SliceObservation> {
        (0..15)
            .map(|i| {
                let k = -r + 2.0 * r * f64::from(i) / 14.0;
                SliceObservation::new(k, truth.eval(k).unwrap().0, 1.0).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_fit_slice_small_sigma_valleys_converge() {
        // sigma / r of 0.03 and 0.018: a long, curved (m, sigma) valley
        // that a window moving half its width per recentre crawled along
        // until the recentres ran out (GridEdge).
        for (tau, a, b, rho, m, sigma, r) in [
            (
                0.245_503_128_527_049,
                0.138_105_455_658_577_74,
                0.127_101_243_666_189_2,
                -0.480_524_832_248_707_7,
                0.056_011_479_397_551_65,
                0.020_706_953_356_863_105,
                0.675_269_550_618_257_9,
            ),
            (
                1.558_819_531_357_679_1,
                1.301_394_575_596_217_3,
                0.622_997_868_208_636_5,
                -0.999,
                -0.095_519_970_621_644,
                0.036_904_243_318_094_71,
                2.071_475_349_420_415,
            ),
        ] {
            let range = KRange::new(-r - 1.0, r + 1.0).unwrap();
            let truth = SviSlice::new(tau, a, b, rho, m, sigma, range).unwrap();
            let fit = fit_slice(tau, &exact_points(&truth, r), &FitConfig::default()).unwrap();
            assert!(fit.rmse_vol < 1e-8, "tau {tau}: {}", fit.rmse_vol);
            assert!(
                (fit.slice.m() - m).abs() < 1e-6,
                "tau {tau}: m {}",
                fit.slice.m()
            );
            assert!(
                (fit.slice.sigma() / sigma - 1.0).abs() < 1e-5,
                "tau {tau}: sigma {}",
                fit.slice.sigma()
            );
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

    /// Fits `cases` random slices and returns the worst rmse_vol: tau in
    /// [0.02, 2], rho in [-0.999, 0.999], sigma / r log uniform in
    /// [0.01, 1], m within 0.3 r, b up to 95% of the Lee bound, a positive
    /// (inside the fit's a >= a_min; a < 0 has its own test); 15 exact
    /// points over [-r, r].
    fn recovery_sweep(cases: usize, seed: u64) -> f64 {
        let mut rng = Lcg(seed);
        let mut worst = 0.0_f64;
        for case in 0..cases {
            let tau = 0.02 + 1.98 * rng.next();
            let r = 0.3 + 2.2 * rng.next();
            let rho = 1.998 * rng.next() - 0.999;
            let sigma = r * f64::exp(f64::ln(0.01) * rng.next());
            let m = 0.3 * r * (2.0 * rng.next() - 1.0);
            let b = (0.05 + 0.9 * rng.next()) * LEE_SLOPE_BOUND / (1.0 + rho.abs());
            let a = tau * (0.005 + 0.2 * rng.next());
            let range = KRange::new(-r - 1.0, r + 1.0).unwrap();
            let truth = SviSlice::new(tau, a, b, rho, m, sigma, range).unwrap();
            let fit = fit_slice(tau, &exact_points(&truth, r), &FitConfig::default())
                .unwrap_or_else(|e| {
                    panic!(
                        "case {case}: tau {tau} a {a} b {b} rho {rho} m {m} sigma {sigma} r {r}: {e}"
                    )
                });
            worst = worst.max(fit.rmse_vol);
        }
        worst
    }

    #[test]
    fn test_fit_slice_recovers_random_slices_down_to_small_sigma() {
        let worst = recovery_sweep(25, 11);
        assert!(worst < 1e-7, "{worst}");
    }

    /// The full sweep; slow unoptimised, so run it with
    /// `cargo test --release --features options -- --ignored recovery_sweep_300`.
    #[test]
    #[ignore = "300 fits: run in release"]
    fn test_fit_slice_recovery_sweep_300() {
        // Observed: all 300 recovered, worst rmse_vol 2.2e-9.
        let worst = recovery_sweep(300, 11);
        assert!(worst < 1e-7, "{worst}");
    }

    #[test]
    fn test_fit_slice_recovers_a_known_slice() {
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.5, 0.02, 0.1, -0.4, 0.03, 0.25, range).unwrap();
        let obs = observe(&truth, &ks(-0.8, 0.8, 17));
        let fit = fit_slice(0.5, &obs, &FitConfig::default()).unwrap();
        assert!(fit.rmse_vol < 1e-6, "rmse {}", fit.rmse_vol);
        for k in ks(-1.2, 1.2, 25) {
            let (w_t, _, _) = truth.eval(k).unwrap();
            let (w_f, _, _) = fit.slice.eval(k).unwrap();
            assert!(
                ((w_f / 0.5).sqrt() - (w_t / 0.5).sqrt()).abs() < 1e-5,
                "k {k}"
            );
        }
        assert_eq!(fit.slice.k_range(), KRange::new(-0.8, 0.8).unwrap());
    }

    #[test]
    fn test_fit_slice_boundary_rho_minus_one_is_reported_active() {
        // A short-dated smile flat on the right and steep on the left: the
        // best fit sits on rho = -1.
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.02, 0.0004, 0.05, -1.0, 0.0, 0.05, range).unwrap();
        let obs = observe(&truth, &ks(-0.4, 0.4, 15));
        let fit = fit_slice(0.02, &obs, &FitConfig::default()).unwrap();
        assert!(
            fit.active.contains(&ActiveConstraint::RhoLower),
            "{:?}",
            fit.active
        );
        assert_eq!(fit.slice.rho(), -1.0);
        assert!(fit.slice.minimum() > 0.0);
    }

    #[test]
    fn test_fit_slice_lee_bound_is_reported_active() {
        // Wings of slope 2.2 ask for more than Lee's bound allows (slope 2),
        // around a smooth vertex (sigma 0.5). Steeper wings pull the
        // constrained optimum to sigma -> 0, below the search: GridEdge.
        let obs: Vec<SliceObservation> = ks(-1.0, 1.0, 21)
            .iter()
            .map(|&k| SliceObservation::new(k, 0.01 + 2.2 * k.hypot(0.5), 1.0).unwrap())
            .collect();
        let config = FitConfig {
            max_rmse_vol: 10.0,
            ..FitConfig::default()
        };
        let fit = fit_slice(1.0, &obs, &config).unwrap();
        assert!(
            fit.active
                .iter()
                .any(|c| matches!(c, ActiveConstraint::LeeUpper | ActiveConstraint::LeeLower)),
            "{:?}",
            fit.active
        );
        assert!(fit.lee_margin.abs() < 1e-12, "{}", fit.lee_margin);
        assert!(fit.objective_params <= fit.objective * (1.0 + 1e-9) + 1e-12);
        // A sharp V of slope 3 wants sigma -> 0, below the search: a
        // GridEdge failure, never a sigma outside the declared range.
        let v: Vec<SliceObservation> = ks(-1.0, 1.0, 21)
            .iter()
            .map(|&k| SliceObservation::new(k, 0.01 + 3.0 * k.abs(), 1.0).unwrap())
            .collect();
        assert_eq!(
            fit_slice(1.0, &v, &config).map(|f| f.slice.sigma()),
            Err(FitFailure::GridEdge)
        );
    }

    #[test]
    fn test_fit_slice_a_min_is_reported_active() {
        // A truth with a < 0 (valid raw SVI) wants a below the fit's a_min.
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.5, -0.02, 0.2, -0.3, 0.0, 0.3, range).unwrap();
        let obs = observe(&truth, &ks(-0.8, 0.8, 17));
        let config = FitConfig {
            max_rmse_vol: 1.0,
            ..FitConfig::default()
        };
        let fit = fit_slice(0.5, &obs, &config).unwrap();
        assert!(
            fit.active.contains(&ActiveConstraint::AMin),
            "{:?}",
            fit.active
        );
        assert_eq!(fit.slice.a(), config.a_min);
        assert_eq!(fit.a_margin, 0.0);
    }

    #[test]
    fn test_fit_slice_caps_a_dominant_weight_and_reports_it() {
        // An exact slice and one point 1% off with a weight 1e10 times the
        // others': capped at max_weight_ratio times the median (10 here),
        // reported, and fitted exactly as if it had been given that weight.
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.5, 0.02, 0.1, -0.4, 0.0, 0.2, range).unwrap();
        let exact = observe(&truth, &ks(-0.6, 0.6, 15));
        let off = 7;
        let point = exact.get(off).copied().unwrap();
        let with_weight = |weight: f64| {
            let mut o = exact.clone();
            *o.get_mut(off).unwrap() =
                SliceObservation::new(point.k(), point.w() * 1.01, weight).unwrap();
            o
        };
        let obs = with_weight(1e10);
        let fit = fit_slice(0.5, &obs, &FitConfig::default()).unwrap();
        assert_eq!(fit.capped, vec![off]);
        let at_cap = fit_slice(0.5, &with_weight(10.0), &FitConfig::default()).unwrap();
        assert!(at_cap.capped.is_empty());
        assert_eq!(fit.slice, at_cap.slice);
        // Without the cap the slice passes through the 1% error; with it,
        // it stays closer to the truth.
        let uncapped = fit_slice(
            0.5,
            &obs,
            &FitConfig {
                max_weight_ratio: 1e300,
                ..FitConfig::default()
            },
        )
        .unwrap();
        let miss = |f: &SliceFit| (f.slice.eval(point.k()).unwrap().0 / point.w() - 1.0).abs();
        assert!(miss(&uncapped) > 0.0099, "{}", miss(&uncapped));
        assert!(
            miss(&fit) < miss(&uncapped),
            "{} {}",
            miss(&fit),
            miss(&uncapped)
        );
    }

    #[test]
    fn test_fit_slice_weight_cap_survives_weights_near_f64_max() {
        // An even count of weights near f64::MAX: the median (1e308) must
        // not overflow, so the larger weight is still capped at ratio 1.
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.5, 0.02, 0.1, -0.4, 0.0, 0.2, range).unwrap();
        let obs: Vec<SliceObservation> = ks(-0.6, 0.6, 16)
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let weight = if i == 7 { 1.7e308 } else { 1e308 };
                SliceObservation::new(k, truth.eval(k).unwrap().0, weight).unwrap()
            })
            .collect();
        let config = FitConfig {
            max_weight_ratio: 1.0,
            ..FitConfig::default()
        };
        assert_eq!(fit_slice(0.5, &obs, &config).unwrap().capped, vec![7]);
        // A cap that overflows binds nothing.
        let wide = FitConfig {
            max_weight_ratio: 10.0,
            ..FitConfig::default()
        };
        assert!(fit_slice(0.5, &obs, &wide).unwrap().capped.is_empty());
    }

    #[test]
    fn test_fit_slice_failures_are_typed() {
        let range = KRange::new(-1.0, 1.0).unwrap();
        let truth = SviSlice::new(0.5, 0.02, 0.1, -0.4, 0.0, 0.25, range).unwrap();
        let config = FitConfig::default();
        let few = observe(&truth, &ks(-0.5, 0.5, 4));
        assert!(matches!(
            fit_slice(0.5, &few, &config),
            Err(FitFailure::TooFewObservations {
                observed: 4,
                required: 5
            })
        ));
        let same_k: Vec<SliceObservation> = (0..6)
            .map(|_| SliceObservation::new(0.1, 0.02, 1.0).unwrap())
            .collect();
        assert_eq!(fit_slice(0.5, &same_k, &config), Err(FitFailure::NoSpan));
        // A zig-zag no smile follows: the residual is too large.
        let noisy: Vec<SliceObservation> = ks(-0.5, 0.5, 11)
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                SliceObservation::new(k, if i % 2 == 0 { 0.02 } else { 0.08 }, 1.0).unwrap()
            })
            .collect();
        assert!(matches!(
            fit_slice(0.5, &noisy, &config),
            Err(FitFailure::RmseAboveMax { .. })
        ));
        assert!(matches!(
            fit_slice(0.0, &few, &config),
            Err(FitFailure::Invalid(_))
        ));
    }

    #[test]
    fn test_slice_observation_invalid_values_return_err() {
        assert!(SliceObservation::new(f64::NAN, 0.1, 1.0).is_err());
        assert!(SliceObservation::new(0.0, -0.1, 1.0).is_err());
        assert!(SliceObservation::new(0.0, 0.1, 0.0).is_err());
        assert!(SliceObservation::new(0.0, 0.0, 1.0).is_ok());
    }
}
