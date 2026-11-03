//! Validation of `GLFTStrategy` against a reference solution of the
//! Gueant, Lehalle and Fernandez-Tapia model. The reference was written for
//! #86 to document the gap; since #134 `GLFTStrategy` implements the model
//! and these tests assert that it matches.
//!
//! # Declared before comparing
//!
//! **Published model** (Gueant, Lehalle and Fernandez-Tapia, "Dealing with
//! the Inventory Risk", arXiv:1105.3115v5): reference price `dS = sigma dW`,
//! fill intensity `A exp(-k delta)` on each side, CARA utility with risk
//! aversion `gamma` on the terminal P&L, horizon `T`, and an inventory bound
//! `Q`: at `q = Q` no bid is posted and at `q = -Q` no ask is posted.
//!
//! - Proposition 2: `v(t) = exp(-M (T - t)) * (1, ..., 1)'`, where `M` is the
//!   symmetric tridiagonal matrix on `q = -Q..Q` with `M_qq = alpha q^2`,
//!   off diagonals `-eta`, `alpha = k gamma sigma^2 / 2` and
//!   `eta = A (1 + gamma / k)^(-(1 + k / gamma))`.
//! - Theorem 1: `delta_b(t, q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k)`
//!   for `q != Q`, and `delta_a(t, q) = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k)`
//!   for `q != -Q`. Distances are measured from the reference price.
//! - Theorem 2: as `T - t` grows the quotes tend to the same formulas with
//!   `v` replaced by `f0`, the positive eigenvector of the smallest
//!   eigenvalue of `M` (the "spectral limit").
//! - Proposition 3 and the closed form after it:
//!   `delta_b ~ (1/gamma) ln(1 + gamma/k) + (2q + 1)/2 * C`,
//!   `delta_a ~ (1/gamma) ln(1 + gamma/k) - (2q - 1)/2 * C`, with
//!   `C = sqrt(sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma))`.
//! - Footnote 8: the Avellaneda-Stoikov quotes
//!   `delta_b ~ (1/gamma) ln(1 + gamma/k) + (1 + 2q)/2 * gamma sigma^2 (T - t)`
//!   are the first order expansion of Theorem 1 for `t` close to `T`.
//!
//! **What `GLFTStrategy` computes** (`src/strategy/glft.rs`, #134): the
//! quotes of Theorem 1, with `exp(-(M - c I) tau) * 1` applied to the vector
//! by Taylor steps, and the closed form after Proposition 3. These tests
//! check it against this reference, which it does not import.
//!
//! **Parameters**: the paper's example (Figure 1): `sigma = 0.3 tick/sqrt(s)`,
//! `A = 0.9 /s`, `k = 0.3 /tick`, `gamma = 0.01 /tick`, `T = 600 s`.
//!
//! **Units**: the reference works in ticks and seconds. `GLFTStrategy`
//! takes an absolute volatility per sqrt(year) and times in milliseconds,
//! with a 365 day year (31 536 000 s), and `A` per second, so it is called
//! with `sigma_year = 0.3 * sqrt(31 536 000)` ticks per sqrt(year), so that
//! its `alpha = k gamma sigma_year^2 / (2 * 31 536 000)` per second equals the
//! paper's `k gamma sigma^2 / 2` per second, `A = 0.9` per second (so
//! `eta tau` is dimensionless with `tau` in seconds), `gamma` and `k` per
//! tick, `lot_size = 1`, prices in ticks around a mid of 1000 ticks and
//! `min_spread = 0`.
//!
//! **Numerics** (f64, test code only, not a public type):
//! - `exp(-M tau)` is evaluated by two independent methods: symmetric
//!   eigendecomposition (cyclic Jacobi) and scaling and squaring with a
//!   Taylor series. They must agree within [`EXPM_REL_TOL`] on `v` and
//!   [`EXPM_QUOTE_TOL`] ticks on the quotes.
//! - Only ratios of `v` enter the quotes, so both methods compute
//!   `exp(-(M - c I) tau)` with the Gershgorin bound `c = -2 eta <= lambda_min(M)`.
//!   That matrix has spectral norm at most 1, so nothing overflows at long
//!   horizons; the factor `exp(-c tau)` cancels in every ratio.
//! - Increasing `Q` changes the model: it truncates the inventory domain
//!   less. It is not a discretization being refined (inventory is integer
//!   in the model). The tests check that quotes at small `|q|` converge as
//!   `Q` grows.

use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy, GlftQuotes};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Agreement of the two matrix exponentials on the normalized `v`
/// (relative, per component).
const EXPM_REL_TOL: f64 = 1e-10;
/// Agreement of the two matrix exponentials on the quotes, in ticks.
const EXPM_QUOTE_TOL: f64 = 1e-9;
/// Semigroup identity `exp(M(a+b)) = exp(Ma) exp(Mb)`, relative Frobenius
/// error. Jacobi reaches about 5e-15; scaling and squaring accumulates
/// rounding over its squarings (15 at 3600 s for Q = 16) and reaches about
/// 1e-12 there.
const SEMIGROUP_TOL: f64 = 1e-11;
/// `GLFTStrategy` goes through `Decimal` and `f64` `ln`/`powi`, so its
/// quote distances are compared with the f64 formula within this many ticks.
const DECIMAL_TOL: f64 = 1e-9;
/// At `T - t = 600 s` the finite horizon quotes must be this close (ticks)
/// to the spectral limit of Theorem 2, for `Q = 32`.
const SPECTRAL_LIMIT_TOL: f64 = 1e-5;
/// Quotes at `|q| <= 3` for `Q = 32` and `Q = 64` must agree within this many
/// ticks (truncation of the inventory domain).
const TRUNCATION_TOL: f64 = 1e-6;

const SECONDS_PER_YEAR: f64 = 31_536_000.0;
const MID_TICKS: Decimal = dec!(1000);

/// Paper example parameters (Figure 1).
const PAPER: Params = Params {
    sigma: 0.3,
    a: 0.9,
    k: 0.3,
    gamma: 0.01,
};
const PAPER_T: f64 = 600.0;

mod reference {
    //! Private reference solver: dense f64 matrices, sizes up to 129.

    pub type Mat = Vec<Vec<f64>>;

    #[derive(Clone, Copy, Debug)]
    pub struct Params {
        pub sigma: f64,
        pub a: f64,
        pub k: f64,
        pub gamma: f64,
    }

    impl Params {
        pub fn alpha(&self) -> f64 {
            self.k * self.gamma * self.sigma * self.sigma / 2.0
        }
        pub fn eta(&self) -> f64 {
            self.a * (1.0 + self.gamma / self.k).powf(-(1.0 + self.k / self.gamma))
        }
        /// `(1/gamma) ln(1 + gamma/k)`, the quote at zero inventory risk.
        pub fn base(&self) -> f64 {
            (1.0 + self.gamma / self.k).ln() / self.gamma
        }
        /// `C` in the closed form approximation after Proposition 3.
        pub fn closed_form_slope(&self) -> f64 {
            (self.sigma * self.sigma * self.gamma / (2.0 * self.k * self.a)
                * (1.0 + self.gamma / self.k).powf(1.0 + self.k / self.gamma))
            .sqrt()
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Method {
        Jacobi,
        Taylor,
    }

    /// Index of inventory `q` in a vector over `-q_max..=q_max`.
    pub fn idx(q: i64, q_max: i64) -> usize {
        usize::try_from(q + q_max).expect("inventory inside the domain")
    }

    /// The matrix `M` of Proposition 2.
    pub fn generator(p: &Params, q_max: i64) -> Mat {
        let n = idx(q_max, q_max) + 1;
        let (alpha, eta) = (p.alpha(), p.eta());
        let mut m = vec![vec![0.0; n]; n];
        for q in -q_max..=q_max {
            let i = idx(q, q_max);
            let qf = q as f64;
            m[i][i] = alpha * qf * qf;
            if q > -q_max {
                m[i][i - 1] = -eta;
            }
            if q < q_max {
                m[i][i + 1] = -eta;
            }
        }
        m
    }

    /// Gershgorin lower bound of the spectrum of `M`: every row has
    /// `M_ii >= 0` and off diagonal mass at most `2 eta`.
    pub fn shift(p: &Params) -> f64 {
        -2.0 * p.eta()
    }

    pub fn identity(n: usize) -> Mat {
        let mut m = vec![vec![0.0; n]; n];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        m
    }

    pub fn matmul(a: &Mat, b: &Mat) -> Mat {
        let n = a.len();
        let mut c = vec![vec![0.0; n]; n];
        for i in 0..n {
            for l in 0..n {
                let ail = a[i][l];
                if ail == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i][j] += ail * b[l][j];
                }
            }
        }
        c
    }

    pub fn frobenius(a: &Mat) -> f64 {
        a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn sub(a: &Mat, b: &Mat) -> Mat {
        a.iter()
            .zip(b)
            .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x - y).collect())
            .collect()
    }

    /// Cyclic Jacobi eigendecomposition of a symmetric matrix. Returns the
    /// eigenvalues and the eigenvectors as columns of the second matrix.
    pub fn jacobi_eigen(m: &Mat) -> (Vec<f64>, Mat) {
        let n = m.len();
        let mut a = m.clone();
        let mut v = identity(n);
        let off_diagonal = |a: &Mat| -> (f64, f64) {
            let off: f64 = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|(i, j)| a[i][j] * a[i][j])
                .sum();
            let scale: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum::<f64>().max(1e-300);
            (off, scale)
        };
        let mut converged = false;
        for _sweep in 0..100 {
            let (off, scale) = off_diagonal(&a);
            if off <= 1e-30 * scale {
                converged = true;
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p][q];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for row in a.iter_mut() {
                        let akp = row[p];
                        let akq = row[q];
                        row[p] = c * akp - s * akq;
                        row[q] = s * akp + c * akq;
                    }
                    let (low, high) = a.split_at_mut(q);
                    for (apk, aqk) in low[p].iter_mut().zip(high[0].iter_mut()) {
                        let (x, y) = (*apk, *aqk);
                        *apk = c * x - s * y;
                        *aqk = s * x + c * y;
                    }
                    for row in v.iter_mut() {
                        let vkp = row[p];
                        let vkq = row[q];
                        row[p] = c * vkp - s * vkq;
                        row[q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        if !converged {
            let (off, scale) = off_diagonal(&a);
            assert!(
                off <= 1e-30 * scale,
                "Jacobi did not converge in 100 sweeps: off diagonal {off:e}, diagonal {scale:e}"
            );
        }
        ((0..n).map(|i| a[i][i]).collect(), v)
    }

    /// `exp(-(M - c I) tau)` through the eigendecomposition of `M`.
    pub fn expm_jacobi(m: &Mat, c: f64, tau: f64) -> Mat {
        let n = m.len();
        let (lambda, vecs) = jacobi_eigen(m);
        let w: Vec<f64> = lambda.iter().map(|l| (-(l - c) * tau).exp()).collect();
        let mut e = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                e[i][j] = (0..n).map(|l| vecs[i][l] * w[l] * vecs[j][l]).sum();
            }
        }
        e
    }

    /// `exp(-(M - c I) tau)` by scaling and squaring with a Taylor series.
    pub fn expm_taylor(m: &Mat, c: f64, tau: f64) -> Mat {
        let n = m.len();
        let mut b: Mat = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let shift = if i == j { c } else { 0.0 };
                        -(m[i][j] - shift) * tau
                    })
                    .collect()
            })
            .collect();
        let norm1 = (0..n)
            .map(|j| (0..n).map(|i| b[i][j].abs()).sum::<f64>())
            .fold(0.0, f64::max);
        let mut squarings = 0;
        let mut scaled = norm1;
        while scaled > 0.25 {
            scaled /= 2.0;
            squarings += 1;
        }
        let factor = 2f64.powi(squarings);
        for row in b.iter_mut() {
            for x in row.iter_mut() {
                *x /= factor;
            }
        }
        let mut sum = identity(n);
        let mut term = identity(n);
        for order in 1..=40 {
            term = matmul(&term, &b);
            let inv = 1.0 / f64::from(order);
            for row in term.iter_mut() {
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
            for (rs, rt) in sum.iter_mut().zip(&term) {
                for (s, t) in rs.iter_mut().zip(rt) {
                    *s += t;
                }
            }
            if frobenius(&term) < 1e-20 * frobenius(&sum) {
                break;
            }
        }
        for _ in 0..squarings {
            sum = matmul(&sum, &sum);
        }
        sum
    }

    pub fn expm(method: Method, m: &Mat, c: f64, tau: f64) -> Mat {
        match method {
            Method::Jacobi => expm_jacobi(m, c, tau),
            Method::Taylor => expm_taylor(m, c, tau),
        }
    }

    /// `v(T - tau)` up to the positive factor `exp(c tau)`, which cancels in
    /// every quote.
    pub fn v(method: Method, p: &Params, q_max: i64, tau: f64) -> Vec<f64> {
        let m = generator(p, q_max);
        let e = expm(method, &m, shift(p), tau);
        e.iter().map(|row| row.iter().sum()).collect()
    }

    /// `(delta_b, delta_a)` of Theorem 1 from any positive vector `w` over
    /// `-q_max..=q_max` (`v` or `f0`). `None` where the paper posts no quote.
    pub fn quotes_from(p: &Params, w: &[f64], q_max: i64, q: i64) -> (Option<f64>, Option<f64>) {
        let at = |q: i64| w[idx(q, q_max)];
        let bid = (q < q_max).then(|| (at(q) / at(q + 1)).ln() / p.k + p.base());
        let ask = (q > -q_max).then(|| (at(q) / at(q - 1)).ln() / p.k + p.base());
        (bid, ask)
    }

    pub fn quotes(
        method: Method,
        p: &Params,
        q_max: i64,
        tau: f64,
        q: i64,
    ) -> (Option<f64>, Option<f64>) {
        quotes_from(p, &v(method, p, q_max, tau), q_max, q)
    }

    /// Positive eigenvector of the smallest eigenvalue of `M` (Theorem 2).
    pub fn ground_state(p: &Params, q_max: i64) -> Vec<f64> {
        let (lambda, vecs) = jacobi_eigen(&generator(p, q_max));
        let i0 = (0..lambda.len())
            .min_by(|&i, &j| lambda[i].total_cmp(&lambda[j]))
            .expect("non empty spectrum");
        vecs.iter().map(|row| row[i0].abs()).collect()
    }

    /// Closed form approximation after Proposition 3.
    pub fn closed_form(p: &Params, q: i64) -> (f64, f64) {
        let (c, qf) = (p.closed_form_slope(), q as f64);
        (
            p.base() + (2.0 * qf + 1.0) / 2.0 * c,
            p.base() - (2.0 * qf - 1.0) / 2.0 * c,
        )
    }
}

use reference::{Method, Params};

fn config(p: &Params, q_max: i64, lot_size: Decimal, horizon_ms: u64) -> GLFTConfig {
    GLFTConfig::new(
        Decimal::from_f64(p.gamma).expect("finite") / lot_size,
        Decimal::from_f64(p.k).expect("finite"),
        Decimal::from_f64(p.a).expect("finite"),
        u32::try_from(q_max).expect("small bound"),
        lot_size,
        horizon_ms,
        Decimal::ZERO,
    )
    .expect("valid config")
}

fn sigma_year(p: &Params) -> Decimal {
    Decimal::from_f64(p.sigma * SECONDS_PER_YEAR.sqrt()).expect("finite")
}

fn distances(quotes: GlftQuotes) -> (Option<f64>, Option<f64>) {
    (
        quotes
            .bid
            .map(|b| (MID_TICKS - b).to_f64().expect("finite")),
        quotes
            .ask
            .map(|a| (a - MID_TICKS).to_f64().expect("finite")),
    )
}

/// Quote distances `(mid - bid, ask - mid)` of `GLFTStrategy` with the paper
/// parameters converted to its units, at `tau` seconds to `T`, inventory
/// bound `q_max` and `q` lots of size 1. `None` where it posts no quote.
fn current_type(p: &Params, q_max: i64, tau_s: f64, q: i64) -> (Option<f64>, Option<f64>) {
    let horizon_ms: u64 = 36_000_000;
    let tau_ms = (tau_s * 1000.0).round() as u64;
    let quotes = GLFTStrategy::calculate_optimal_quotes(
        MID_TICKS,
        Decimal::from(q),
        &config(p, q_max, Decimal::ONE, horizon_ms),
        sigma_year(p),
        horizon_ms - tau_ms,
    )
    .expect("quotes inside the domain");
    distances(quotes)
}

fn assert_close(label: &str, x: Option<f64>, y: Option<f64>, tol: f64) {
    match (x, y) {
        (Some(x), Some(y)) => assert!((x - y).abs() <= tol, "{label}: {x} vs {y}"),
        (None, None) => {}
        _ => panic!("{label}: quote presence differs: {x:?} vs {y:?}"),
    }
}

/// The footnote 8 expansion in f64, independent of `GLFTStrategy`.
fn as_expansion(p: &Params, tau_s: f64, q: i64) -> (f64, f64) {
    let (g, s2, qf) = (p.gamma, p.sigma * p.sigma, q as f64);
    (
        p.base() + (1.0 + 2.0 * qf) / 2.0 * g * s2 * tau_s,
        p.base() + (1.0 - 2.0 * qf) / 2.0 * g * s2 * tau_s,
    )
}

#[test]
fn test_glft_generator_paper_parameters_match_proposition_two() {
    let p = PAPER;
    let q_max = 8;
    let m = reference::generator(&p, q_max);
    let n = m.len();
    assert_eq!(n, 17);
    for (i, row) in m.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            assert_eq!(*x, m[j][i], "M is symmetric");
        }
    }
    let q = 5;
    let i = reference::idx(q, q_max);
    assert!((m[i][i] - p.alpha() * 25.0).abs() < 1e-18);
    assert_eq!(m[i][i + 1], -p.eta());
    assert_eq!(m[i][i - 1], -p.eta());
    assert_eq!(m[i][i + 2], 0.0);
    // alpha and eta for the paper parameters, computed by hand.
    assert!((p.alpha() - 1.35e-4).abs() < 1e-15);
    let eta = 0.9 * (1.0f64 + 0.01 / 0.3).powf(-(1.0 + 30.0));
    assert!((p.eta() - eta).abs() < 1e-15);
    // The Gershgorin shift is a true lower bound of the spectrum.
    let (lambda, _) = reference::jacobi_eigen(&m);
    let lambda_min = lambda.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(lambda_min >= reference::shift(&p));
    assert!(lambda_min < 0.0, "M is not positive definite here");
}

#[test]
fn test_glft_expm_jacobi_and_taylor_agree_within_tolerance() {
    let p = PAPER;
    for q_max in [8, 16, 32] {
        for tau in [0.0, 1.0, 10.0, 60.0, 600.0, 3600.0] {
            let vj = reference::v(Method::Jacobi, &p, q_max, tau);
            let vt = reference::v(Method::Taylor, &p, q_max, tau);
            for (a, b) in vj.iter().zip(&vt) {
                let rel = (a - b).abs() / b.abs();
                assert!(
                    rel <= EXPM_REL_TOL,
                    "Q={q_max} tau={tau}: {a} vs {b} rel {rel}"
                );
            }
            for q in -q_max..=q_max {
                let (bj, aj) = reference::quotes_from(&p, &vj, q_max, q);
                let (bt, at) = reference::quotes_from(&p, &vt, q_max, q);
                for (x, y) in [(bj, bt), (aj, at)] {
                    match (x, y) {
                        (Some(x), Some(y)) => assert!(
                            (x - y).abs() <= EXPM_QUOTE_TOL,
                            "Q={q_max} tau={tau} q={q}: {x} vs {y}"
                        ),
                        (None, None) => {}
                        _ => panic!("quote presence differs at q={q}"),
                    }
                }
            }
        }
    }
}

#[test]
fn test_glft_expm_split_horizon_satisfies_semigroup() {
    let p = PAPER;
    let q_max = 16;
    let m = reference::generator(&p, q_max);
    let c = reference::shift(&p);
    for method in [Method::Jacobi, Method::Taylor] {
        for (a, b) in [(10.0, 50.0), (300.0, 300.0), (1000.0, 2600.0)] {
            let whole = reference::expm(method, &m, c, a + b);
            let product = reference::matmul(
                &reference::expm(method, &m, c, a),
                &reference::expm(method, &m, c, b),
            );
            let rel = reference::frobenius(&reference::sub(&whole, &product))
                / reference::frobenius(&whole);
            assert!(rel <= SEMIGROUP_TOL, "{method:?} a={a} b={b}: rel {rel}");
        }
    }
}

#[test]
fn test_glft_value_vector_positive_above_paper_lower_bound() {
    // Appendix, proof of Proposition 2: v_q(t) >= exp(-(alpha Q^2 - eta)(T - t)).
    // With the shift c the computed vector is exp(c tau) v.
    let p = PAPER;
    for q_max in [8, 16, 32] {
        let bound_rate = p.alpha() * (q_max * q_max) as f64 - p.eta();
        for tau in [1.0, 60.0, 600.0, 3600.0] {
            for method in [Method::Jacobi, Method::Taylor] {
                let v = reference::v(method, &p, q_max, tau);
                let log_bound = (reference::shift(&p) - bound_rate) * tau;
                for (i, x) in v.iter().enumerate() {
                    assert!(*x > 0.0, "{method:?} Q={q_max} tau={tau} i={i}: {x}");
                    assert!(
                        x.ln() >= log_bound - 1e-9,
                        "{method:?} Q={q_max} tau={tau} i={i}: ln v {} < {log_bound}",
                        x.ln()
                    );
                }
            }
        }
    }
}

#[test]
fn test_glft_expm_long_horizon_normalized_without_overflow() {
    let p = PAPER;
    let q_max = 32;
    let tau = 36_000.0;
    // Without the shift, v grows like exp(-lambda_min tau) and lambda_min < 0.
    let (lambda, _) = reference::jacobi_eigen(&reference::generator(&p, q_max));
    let lambda_min = lambda.iter().copied().fold(f64::INFINITY, f64::min);
    assert!((-lambda_min * tau).exp().is_infinite());
    for method in [Method::Jacobi, Method::Taylor] {
        let v = reference::v(method, &p, q_max, tau);
        assert!(v.iter().all(|x| x.is_finite() && *x > 0.0), "{method:?}");
        for q in -3..=3 {
            let (b, a) = reference::quotes_from(&p, &v, q_max, q);
            assert!(b.is_some_and(f64::is_finite) && a.is_some_and(f64::is_finite));
        }
    }
}

#[test]
fn test_glft_quotes_at_terminal_time_match_current_type() {
    // At t = T, v = 1 and both quotes are (1/gamma) ln(1 + gamma/k) at every
    // inventory; the current type gives the same distances.
    let p = PAPER;
    let q_max = 16;
    for q in [-5, -1, 0, 1, 5] {
        let (b, a) = reference::quotes(Method::Taylor, &p, q_max, 0.0, q);
        let (cb, ca) = current_type(&p, q_max, 0.0, q);
        assert!((b.expect("bid") - p.base()).abs() < 1e-15);
        assert!((a.expect("ask") - p.base()).abs() < 1e-15);
        assert_close(&format!("bid q={q}"), cb, Some(p.base()), DECIMAL_TOL);
        assert_close(&format!("ask q={q}"), ca, Some(p.base()), DECIMAL_TOL);
    }
}

#[test]
fn test_glft_quotes_near_terminal_reduce_to_avellaneda_stoikov() {
    // Footnote 8: the A-S quotes are the first order expansion of Theorem 1
    // for t close to T, away from the inventory bounds. The error must be
    // second order: dividing tau by 10 divides it by about 100.
    let p = PAPER;
    let q_max = 16;
    for q in [-3, 0, 2] {
        let error = |tau: f64| {
            let (b, a) = reference::quotes(Method::Taylor, &p, q_max, tau, q);
            let (eb, ea) = as_expansion(&p, tau, q);
            (b.expect("bid") - eb)
                .abs()
                .max((a.expect("ask") - ea).abs())
        };
        let (e1, e01, e001) = (error(1.0), error(0.1), error(0.01));
        assert!(e1 < 1e-3, "q={q}: error at 1 s {e1}");
        assert!(
            e1 / e01 > 50.0 && e01 / e001 > 50.0,
            "q={q}: {e1} {e01} {e001}"
        );
        // The current type follows the reference, not the expansion.
        for tau in [0.01, 1.0, 600.0] {
            let (cb, ca) = current_type(&p, q_max, tau, q);
            let (rb, ra) = reference::quotes(Method::Jacobi, &p, q_max, tau, q);
            assert_close(&format!("bid q={q} tau={tau}"), cb, rb, DECIMAL_TOL);
            assert_close(&format!("ask q={q} tau={tau}"), ca, ra, DECIMAL_TOL);
        }
    }
}

#[test]
fn test_glft_quotes_long_horizon_converge_to_spectral_limit() {
    let p = PAPER;
    let q_max = 32;
    let f0 = reference::ground_state(&p, q_max);
    assert!(f0.iter().all(|x| *x > 0.0), "f0 is positive (Theorem 2)");
    for q in [-5, -2, 0, 3, 10] {
        let (b, a) = reference::quotes(Method::Taylor, &p, q_max, PAPER_T, q);
        let (lb, la) = reference::quotes_from(&p, &f0, q_max, q);
        let (b, a, lb, la) = (b.unwrap(), a.unwrap(), lb.unwrap(), la.unwrap());
        assert!((b - lb).abs() < SPECTRAL_LIMIT_TOL, "q={q}: {b} vs {lb}");
        assert!((a - la).abs() < SPECTRAL_LIMIT_TOL, "q={q}: {a} vs {la}");
        let (cb, ca) = current_type(&p, q_max, PAPER_T, q);
        assert_close(&format!("bid q={q}"), cb, Some(lb), SPECTRAL_LIMIT_TOL);
        assert_close(&format!("ask q={q}"), ca, Some(la), SPECTRAL_LIMIT_TOL);
        // Ten times longer, the current type is still finite and at the limit.
        let (cb, ca) = current_type(&p, q_max, 10.0 * PAPER_T, q);
        assert_close(
            &format!("bid q={q} 6000 s"),
            cb,
            Some(lb),
            SPECTRAL_LIMIT_TOL,
        );
        assert_close(
            &format!("ask q={q} 6000 s"),
            ca,
            Some(la),
            SPECTRAL_LIMIT_TOL,
        );
    }
}

#[test]
fn test_glft_quotes_larger_inventory_bound_converge() {
    let p = PAPER;
    let qs = [8, 16, 32, 64];
    let quotes: Vec<Vec<(f64, f64)>> = qs
        .iter()
        .map(|&q_max| {
            let v = reference::v(Method::Taylor, &p, q_max, PAPER_T);
            (-3..=3)
                .map(|q| {
                    let (b, a) = reference::quotes_from(&p, &v, q_max, q);
                    (b.unwrap(), a.unwrap())
                })
                .collect()
        })
        .collect();
    let gap = |i: usize, j: usize| {
        quotes[i]
            .iter()
            .zip(&quotes[j])
            .map(|(x, y)| (x.0 - y.0).abs().max((x.1 - y.1).abs()))
            .fold(0.0, f64::max)
    };
    let (g8, g16, g32) = (gap(0, 3), gap(1, 3), gap(2, 3));
    assert!(g8 > g16 && g16 > g32, "{g8} {g16} {g32}");
    assert!(g32 < TRUNCATION_TOL, "Q=32 vs Q=64 at |q|<=3: {g32}");
    // The current type converges the same way, and matches each truncation.
    for (&q_max, row) in qs.iter().zip(&quotes) {
        for (q, (rb, ra)) in (-3..=3).zip(row) {
            let (cb, ca) = current_type(&p, q_max, PAPER_T, q);
            assert_close(&format!("Q={q_max} bid q={q}"), cb, Some(*rb), DECIMAL_TOL);
            assert_close(&format!("Q={q_max} ask q={q}"), ca, Some(*ra), DECIMAL_TOL);
        }
    }
}

#[test]
fn test_glft_quotes_at_inventory_bound_omit_one_side() {
    let p = PAPER;
    let q_max = 8;
    let (bid_at_top, ask_at_top) = reference::quotes(Method::Taylor, &p, q_max, PAPER_T, q_max);
    let (bid_at_bottom, ask_at_bottom) =
        reference::quotes(Method::Taylor, &p, q_max, PAPER_T, -q_max);
    assert!(bid_at_top.is_none() && ask_at_top.is_some());
    assert!(ask_at_bottom.is_none() && bid_at_bottom.is_some());
    // The current type omits the same sides and matches the posted ones.
    for (q, rb, ra) in [
        (q_max, bid_at_top, ask_at_top),
        (-q_max, bid_at_bottom, ask_at_bottom),
    ] {
        let (cb, ca) = current_type(&p, q_max, PAPER_T, q);
        assert_close(&format!("bid q={q}"), cb, rb, DECIMAL_TOL);
        assert_close(&format!("ask q={q}"), ca, ra, DECIMAL_TOL);
    }
}

#[test]
fn test_glft_current_type_matches_reference_both_methods() {
    let p = PAPER;
    for q_max in [8, 16, 32] {
        for tau in [0.0, 1.0, 10.0, 60.0, 600.0, 3600.0] {
            let vj = reference::v(Method::Jacobi, &p, q_max, tau);
            let vt = reference::v(Method::Taylor, &p, q_max, tau);
            for q in -q_max..=q_max {
                let (cb, ca) = current_type(&p, q_max, tau, q);
                for (name, v) in [("Jacobi", &vj), ("Taylor", &vt)] {
                    let (rb, ra) = reference::quotes_from(&p, v, q_max, q);
                    let label = format!("{name} Q={q_max} tau={tau} q={q}");
                    assert_close(&format!("{label} bid"), cb, rb, DECIMAL_TOL);
                    assert_close(&format!("{label} ask"), ca, ra, DECIMAL_TOL);
                }
            }
        }
    }
}

#[test]
fn test_glft_current_type_closed_form_matches_reference() {
    // Proposition 3: the closed form, and the exact skew per unit of
    // inventory at the paper horizon is close to its slope C.
    let p = PAPER;
    let q_max = 32;
    let horizon_ms = 600_000;
    let config = config(&p, q_max, Decimal::ONE, horizon_ms);
    for q in [-5, -1, 0, 1, 5] {
        let closed =
            GLFTStrategy::closed_form_quotes(MID_TICKS, Decimal::from(q), &config, sigma_year(&p))
                .expect("closed form");
        let (cb, ca) = distances(closed);
        let (rb, ra) = reference::closed_form(&p, q);
        assert_close(&format!("bid q={q}"), cb, Some(rb), DECIMAL_TOL);
        assert_close(&format!("ask q={q}"), ca, Some(ra), DECIMAL_TOL);
    }
    let (c0, _) = current_type(&p, q_max, PAPER_T, 0);
    let (c1, _) = current_type(&p, q_max, PAPER_T, 1);
    let slope = c1.expect("bid") - c0.expect("bid");
    assert!((slope - p.closed_form_slope()).abs() < 0.01, "{slope}");
}

#[test]
fn test_glft_current_type_lot_size_matches_rescaled_reference() {
    // With lots of 0.5, gamma per unit of price is 2 gamma_L, and inventory
    // 1.5 is 3 lots: the quotes are those of the reference with gamma_L.
    let p = PAPER;
    let q_max = 16;
    let horizon_ms = 600_000;
    let lot = dec!(0.5);
    let config = config(&p, q_max, lot, horizon_ms);
    for q in [-3_i64, 0, 3] {
        let inventory = Decimal::from(q) * lot;
        let quotes = GLFTStrategy::calculate_optimal_quotes(
            MID_TICKS,
            inventory,
            &config,
            sigma_year(&p),
            0,
        )
        .expect("quotes");
        let (cb, ca) = distances(quotes);
        let (rb, ra) = reference::quotes(Method::Jacobi, &p, q_max, PAPER_T, q);
        assert_close(&format!("bid q={q}"), cb, rb, DECIMAL_TOL);
        assert_close(&format!("ask q={q}"), ca, ra, DECIMAL_TOL);
    }
}

fn fmt_opt(x: Option<f64>) -> String {
    x.map_or_else(|| "none".to_string(), |x| format!("{x:.6}"))
}

/// Prints the evidence tables attached to the PR. Run with
/// `cargo test --test glft_reference -- --ignored --nocapture`.
#[test]
#[ignore = "evidence table, run on demand"]
fn test_glft_reference_paper_parameters_print_evidence_table() {
    let p = PAPER;
    let q_max = 32;
    println!(
        "Paper parameters: sigma={} tick/sqrt(s), A={} /s, k={} /tick, gamma={} /tick; Q={q_max}",
        p.sigma, p.a, p.k, p.gamma
    );
    println!(
        "alpha={:.6e}, eta={:.9}, (1/gamma)ln(1+gamma/k)={:.9}, closed form slope C={:.9}",
        p.alpha(),
        p.eta(),
        p.base(),
        p.closed_form_slope()
    );
    println!();
    println!("Table 1: quote distances in ticks (bid = s - s_b, ask = s_a - s).");
    println!(
        "| tau (s) | q | reference bid | reference ask | GLFTStrategy bid | GLFTStrategy ask | closed form bid | closed form ask | spectral limit bid | spectral limit ask |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let f0 = reference::ground_state(&p, q_max);
    for tau in [0.0, 1.0, 10.0, 60.0, 600.0, 3600.0] {
        let v = reference::v(Method::Taylor, &p, q_max, tau);
        for q in [-5, -2, -1, 0, 1, 2, 5] {
            let (rb, ra) = reference::quotes_from(&p, &v, q_max, q);
            let (cb, ca) = current_type(&p, q_max, tau, q);
            let (ab, aa) = reference::closed_form(&p, q);
            let (lb, la) = reference::quotes_from(&p, &f0, q_max, q);
            println!(
                "| {tau} | {q} | {} | {} | {} | {} | {ab:.6} | {aa:.6} | {} | {} |",
                fmt_opt(rb),
                fmt_opt(ra),
                fmt_opt(cb),
                fmt_opt(ca),
                fmt_opt(lb),
                fmt_opt(la)
            );
        }
    }
    println!();
    println!("Table 2: inventory bound at tau = 600 s, Q = 8.");
    println!("| q | reference bid | reference ask | GLFTStrategy bid | GLFTStrategy ask |");
    println!("|---|---|---|---|---|");
    let v8 = reference::v(Method::Taylor, &p, 8, PAPER_T);
    for q in [-8, -7, 7, 8] {
        let (rb, ra) = reference::quotes_from(&p, &v8, 8, q);
        let (cb, ca) = current_type(&p, 8, PAPER_T, q);
        println!(
            "| {q} | {} | {} | {} | {} |",
            fmt_opt(rb),
            fmt_opt(ra),
            fmt_opt(cb),
            fmt_opt(ca)
        );
    }
    println!();
    println!("Table 3: truncation of the inventory domain, tau = 600 s, bid at q.");
    println!("| q | Q=8 | Q=16 | Q=32 | Q=64 | abs(Q - Q64) for Q = 8 / 16 / 32 |");
    println!("|---|---|---|---|---|---|");
    let vs: Vec<(i64, Vec<f64>)> = [8, 16, 32, 64]
        .iter()
        .map(|&qm| (qm, reference::v(Method::Taylor, &p, qm, PAPER_T)))
        .collect();
    for q in [-3, 0, 3] {
        let row: Vec<f64> = vs
            .iter()
            .map(|(qm, v)| reference::quotes_from(&p, v, *qm, q).0.unwrap())
            .collect();
        println!(
            "| {q} | {:.9} | {:.9} | {:.9} | {:.9} | {:.3e} / {:.3e} / {:.3e} |",
            row[0],
            row[1],
            row[2],
            row[3],
            (row[0] - row[3]).abs(),
            (row[1] - row[3]).abs(),
            (row[2] - row[3]).abs()
        );
    }
    println!();
    println!("Table 4: numerical checks of the matrix exponential (Q = 32).");
    println!("| tau (s) | max rel diff v (Jacobi vs Taylor) | max abs diff quotes (ticks) |");
    println!("|---|---|---|");
    for tau in [1.0, 60.0, 600.0, 3600.0, 36_000.0] {
        let vj = reference::v(Method::Jacobi, &p, q_max, tau);
        let vt = reference::v(Method::Taylor, &p, q_max, tau);
        let rel = vj
            .iter()
            .zip(&vt)
            .map(|(a, b)| (a - b).abs() / b.abs())
            .fold(0.0, f64::max);
        let quote_diff = (-q_max..=q_max)
            .flat_map(|q| {
                let (bj, aj) = reference::quotes_from(&p, &vj, q_max, q);
                let (bt, at) = reference::quotes_from(&p, &vt, q_max, q);
                [(bj, bt), (aj, at)]
            })
            .filter_map(|(x, y)| Some((x? - y?).abs()))
            .fold(0.0, f64::max);
        println!("| {tau} | {rel:.3e} | {quote_diff:.3e} |");
    }
    let m = reference::generator(&p, 16);
    let c = reference::shift(&p);
    for method in [Method::Jacobi, Method::Taylor] {
        let whole = reference::expm(method, &m, c, 3600.0);
        let product = reference::matmul(
            &reference::expm(method, &m, c, 1000.0),
            &reference::expm(method, &m, c, 2600.0),
        );
        println!(
            "semigroup {method:?} (Q=16, 1000 s + 2600 s): rel Frobenius {:.3e}",
            reference::frobenius(&reference::sub(&whole, &product)) / reference::frobenius(&whole)
        );
    }
}
