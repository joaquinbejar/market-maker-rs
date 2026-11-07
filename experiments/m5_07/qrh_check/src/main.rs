//! Step 1 of #111: OptionStratLib's quadratic rough Heston model against the values its papers
//! state, as `../criteria.json` fixes them before any run.
//!
//! `cargo run --release -- <criteria.json>` prints one JSON report: each check, its numbers and
//! whether it passes. With `../criteria_1b.json` it runs step 1b instead, the checks fixed after
//! step 1 failed. Everything is synthetic: Monte Carlo paths of the model at the papers'
//! SPX parameters, calibrated to nothing here.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::str::FromStr;

use optionstratlib::OptionStyle;
use optionstratlib::prelude::{Positive, pos_or_panic};
use optionstratlib::simulation::{
    QuadraticRoughHestonGrid, QuadraticRoughHestonKernel, QuadraticRoughHestonParams,
    price_vanilla_quadratic_rough_heston, simulate_quadratic_rough_heston_path,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde_json::{Value, json};

fn dec(text: &str) -> Decimal {
    Decimal::from_str(text).unwrap()
}

fn f(d: Decimal) -> f64 {
    d.to_f64().unwrap()
}

/// Lanczos approximation of the gamma function (g = 7, n = 9), for positive arguments.
fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

/// The regularized lower incomplete gamma function P(a, x), by its series or its continued
/// fraction (Numerical Recipes, gser and gcf).
fn lower_regularized(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_gamma = gamma(a).ln();
    if x < a + 1.0 {
        let (mut sum, mut term, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma).exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma).exp() * h
    }
}

/// The geometric partition of 2107.01611 Appendix A, in closed form: (c_i, gamma_i).
fn closed_form(alpha: f64, n: usize, x: f64) -> Vec<(f64, f64)> {
    let half = n as f64 / 2.0;
    let norm = (1.0 - alpha) * gamma(alpha) * gamma(1.0 - alpha);
    (1..=n)
        .map(|i| {
            let i = i as f64;
            let c = x.powf((1.0 - alpha) * (i - half)) * (1.0 - x.powf(alpha - 1.0)) / norm;
            let g = (1.0 - alpha) * x.powf(i - 1.0 - half) * (x.powf(2.0 - alpha) - 1.0)
                / ((2.0 - alpha) * (x.powf(1.0 - alpha) - 1.0));
            (c, g)
        })
        .collect()
}

/// The squared L2(0, T) distance between the multi-factor kernel and t^(alpha-1) / Gamma(alpha),
/// in closed form.
fn l2_distance(alpha: f64, n: usize, x: f64, horizon: f64) -> f64 {
    let kernel = closed_form(alpha, n, x);
    let ga = gamma(alpha);
    let own = horizon.powf(2.0 * alpha - 1.0) / ((2.0 * alpha - 1.0) * ga * ga);
    let cross: f64 = kernel
        .iter()
        .map(|(c, g)| c * g.powf(-alpha) * lower_regularized(alpha, g * horizon))
        .sum();
    let mut multi = 0.0;
    for (ci, gi) in &kernel {
        for (cj, gj) in &kernel {
            let s = gi + gj;
            multi += ci * cj * (1.0 - (-s * horizon).exp()) / s;
        }
    }
    own - 2.0 * cross + multi
}

/// The x > 1 that minimizes the distance: a grid, then golden section around its best point.
fn optimal_ratio(alpha: f64, n: usize, horizon: f64) -> f64 {
    let grid: Vec<f64> = (0..2000).map(|k| 1.05 + k as f64 * 0.01).collect();
    let best = grid
        .iter()
        .copied()
        .min_by(|a, b| {
            l2_distance(alpha, n, *a, horizon)
                .partial_cmp(&l2_distance(alpha, n, *b, horizon))
                .unwrap()
        })
        .unwrap();
    let (mut lo, mut hi) = (best - 0.01, best + 0.01);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..100 {
        let a = hi - phi * (hi - lo);
        let b = lo + phi * (hi - lo);
        if l2_distance(alpha, n, a, horizon) < l2_distance(alpha, n, b, horizon) {
            hi = b;
        } else {
            lo = a;
        }
    }
    (lo + hi) / 2.0
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// erfc by its Chebyshev fit (Numerical Recipes), relative error under 1.2e-7.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

fn bs_call(s: f64, k: f64, t: f64, vol: f64) -> f64 {
    let d1 = ((s / k).ln() + 0.5 * vol * vol * t) / (vol * t.sqrt());
    s * norm_cdf(d1) - k * norm_cdf(d1 - vol * t.sqrt())
}

/// The Black-Scholes implied volatility of a call (no rates), by bisection; None outside the
/// no-arbitrage range.
fn implied_vol(price: f64, s: f64, k: f64, t: f64) -> Option<f64> {
    let (low, high) = ((s - k).max(0.0), s);
    if price <= low || price >= high {
        return None;
    }
    let (mut lo, mut hi) = (1e-6, 5.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if bs_call(s, k, t, mid) < price {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

fn params(criteria: &Value, kernel: QuadraticRoughHestonKernel) -> QuadraticRoughHestonParams {
    let p = &criteria["model"]["parameters_2212_10164"];
    let d = |k: &str| dec(p[k].as_str().unwrap());
    let z0 = p["z0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| dec(v.as_str().unwrap()))
        .collect();
    QuadraticRoughHestonParams::try_new(
        d("a"),
        d("b"),
        d("c"),
        d("lambda"),
        d("eta"),
        kernel,
        z0,
        pos_or_panic!(f(d("spot"))),
    )
    .unwrap()
}

fn price(
    p: &QuadraticRoughHestonParams,
    t: f64,
    steps: usize,
    style: OptionStyle,
    strike: f64,
    paths: usize,
    seed: u64,
) -> (f64, f64) {
    let grid =
        QuadraticRoughHestonGrid::try_new(Decimal::from_f64_retain(t).unwrap(), steps).unwrap();
    let e = price_vanilla_quadratic_rough_heston(
        p,
        &grid,
        style,
        Positive::new(strike).unwrap(),
        NonZeroUsize::new(paths).unwrap(),
        seed,
    )
    .unwrap();
    (f(e.price()), f(e.standard_error()))
}

fn main() {
    let criteria: Value =
        serde_json::from_str(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap())
            .unwrap();
    if criteria["step"].as_str() == Some("1b") {
        step_1b(&criteria);
        return;
    }
    let model = &criteria["model"];
    let alpha = dec(model["alpha"].as_str().unwrap());
    let factors = model["factors"].as_u64().unwrap() as usize;
    let ratio = dec(model["ratio_x_n"].as_str().unwrap());
    let checks = &criteria["checks"];
    let mut report = serde_json::Map::new();

    // 1. Kernel weights: OptionStratLib against the closed form.
    let kernel = QuadraticRoughHestonKernel::try_new(alpha, factors, ratio).unwrap();
    let expected = closed_form(f(alpha), factors, f(ratio));
    let mut worst = 0.0_f64;
    for (i, (c, g)) in expected.iter().enumerate() {
        let ours_c = f(kernel.weights()[i]);
        let ours_g = f(kernel.discounts()[i]);
        worst = worst
            .max(((ours_c - c) / c).abs())
            .max(((ours_g - g) / g).abs());
    }
    report.insert(
        "kernel_weights".into(),
        json!({
        "weights": kernel.weights(), "discounts": kernel.discounts(),
        "closed_form": expected, "worst_relative_difference": worst, "pass": worst <= 1e-10}),
    );

    // 2. Optimal ratio x*_10 against the printed values of Figure A.1.
    let mut ratios = serde_json::Map::new();
    let mut all = true;
    for (a, printed) in checks["optimal_ratio"]["values"].as_object().unwrap() {
        let ours = optimal_ratio(a.parse().unwrap(), factors, 0.1);
        let printed: f64 = printed.as_str().unwrap().parse().unwrap();
        let pass = (ours - printed).abs() <= 0.02;
        all &= pass;
        ratios.insert(
            a.clone(),
            json!({"ours": ours, "printed": printed, "pass": pass}),
        );
    }
    report.insert(
        "optimal_ratio".into(),
        json!({"by_alpha": ratios, "pass": all}),
    );

    // 3. Initial variance.
    let p = params(&criteria, kernel.clone());
    let v0 = f(p.initial_variance());
    let printed_v0: f64 = checks["initial_variance"]["value"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    report.insert(
        "initial_variance".into(),
        json!({
        "z0": p.initial_z(), "v0": v0, "printed": printed_v0,
        "pass": (v0 * 100.0).round() / 100.0 == printed_v0}),
    );

    // 4. Monte Carlo prices.
    let spot = f(p.spot().to_dec());
    let maturities: [f64; 4] = [0.03, 0.05, 0.07, 0.09];
    let moneyness: [f64; 5] = [-0.15, -0.1, -0.05, 0.0, 0.05];
    let (paths, seed, dt) = (20_000_usize, 7_u64, 0.0005_f64);
    let mut rows = vec![];
    let (mut parity_ok, mut bounds_ok, mut shape_ok, mut skew_ok, mut convergence_ok) =
        (true, true, true, true, true);
    for &t in &maturities {
        let steps = (t / dt).round() as usize;
        let mut calls = vec![];
        let mut vols = vec![];
        for &k in &moneyness {
            let strike = spot * f64::exp(k);
            let (call, call_se) = price(&p, t, steps, OptionStyle::Call, strike, paths, seed);
            let (put, put_se) = price(&p, t, steps, OptionStyle::Put, strike, paths, seed);
            let (half, half_se) = price(&p, t, 2 * steps, OptionStyle::Call, strike, paths, seed);
            let parity = call - put - (spot - strike);
            let parity_se = (call_se * call_se + put_se * put_se).sqrt();
            parity_ok &= parity.abs() <= 3.0 * parity_se;
            bounds_ok &= call >= (spot - strike).max(0.0) && call <= spot;
            let moved = (half - call).abs() <= 3.0 * (call_se * call_se + half_se * half_se).sqrt();
            convergence_ok &= moved;
            let vol = implied_vol(call, spot, strike, t);
            vols.push(vol);
            calls.push((strike, call, call_se));
            rows.push(
                json!({"maturity": t, "log_moneyness": k, "strike": strike, "steps": steps,
                "call": call, "call_se": call_se, "put": put, "put_se": put_se,
                "parity_residual": parity, "parity_se": parity_se,
                "call_half_dt": half, "call_half_dt_se": half_se, "implied_vol": vol}),
            );
        }
        // Decreasing and convex in strike, up to Monte Carlo error (3 standard errors).
        for w in calls.windows(2) {
            let ((_, c0, s0), (_, c1, s1)) = (w[0], w[1]);
            shape_ok &= c1 <= c0 + 3.0 * (s0 * s0 + s1 * s1).sqrt();
        }
        // Convex in the strike: the slope between neighbours does not fall, up to 3 standard
        // errors of the slope difference.
        for w in calls.windows(3) {
            let ((k0, c0, s0), (k1, c1, s1), (k2, c2, s2)) = (w[0], w[1], w[2]);
            let (h0, h1) = (k1 - k0, k2 - k1);
            let change = (c2 - c1) / h1 - (c1 - c0) / h0;
            let error =
                ((s2 / h1).powi(2) + (s1 * (1.0 / h1 + 1.0 / h0)).powi(2) + (s0 / h0).powi(2))
                    .sqrt();
            shape_ok &= change >= -3.0 * error;
        }
        let v: Vec<f64> = vols.iter().flatten().copied().collect();
        skew_ok &= v.len() == vols.len() && v.windows(2).all(|w| w[1] < w[0]);
    }
    // Standard error against paths: four times the paths, about half the error.
    let (_, se1) = price(&p, 0.05, 100, OptionStyle::Call, spot, paths, seed);
    let (_, se4) = price(&p, 0.05, 100, OptionStyle::Call, spot, 4 * paths, seed);
    let se_ratio = se4 / se1;
    // Paths: no NaN and positive variance.
    let grid = QuadraticRoughHestonGrid::try_new(dec("0.09"), 180).unwrap();
    let mut path_ok = true;
    for s in 0..200 {
        let path = simulate_quadratic_rough_heston_path(&p, &grid, 1000 + s).unwrap();
        path_ok &= path.variance().iter().all(|v| *v > Decimal::ZERO);
        path_ok &= path.spot().iter().all(|x| *x > Decimal::ZERO);
    }
    let atm: Vec<f64> = rows
        .iter()
        .filter(|r| r["log_moneyness"].as_f64() == Some(0.0))
        .filter_map(|r| r["implied_vol"].as_f64())
        .collect();
    report.insert(
        "monte_carlo".into(),
        json!({
        "paths": paths, "seed": seed, "dt": dt, "rows": rows,
        "parity_within_3se": parity_ok, "within_bounds": bounds_ok,
        "decreasing_and_convex": shape_ok, "negative_skew": skew_ok,
        "half_dt_within_3se": convergence_ok,
        "se_ratio_4x_paths": se_ratio, "se_halves": (0.4..=0.6).contains(&se_ratio),
        "atm_implied_vols": atm, "sqrt_v0": v0.sqrt(),
        "paths_positive_variance_and_spot": path_ok,
        "pass": parity_ok && bounds_ok && shape_ok && skew_ok && convergence_ok
            && (0.4..=0.6).contains(&se_ratio) && path_ok}),
    );
    let pass = report.values().all(|v| v["pass"].as_bool() == Some(true));
    report.insert("pass".into(), json!(pass));
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(report)).unwrap()
    );
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The Black-Scholes vega of a call (no rates).
fn bs_vega(s: f64, k: f64, t: f64, vol: f64) -> f64 {
    let d1 = ((s / k).ln() + 0.5 * vol * vol * t) / (vol * t.sqrt());
    s * norm_pdf(d1) * t.sqrt()
}

/// V0 = a (Z0 - b)^2 + c for Z0 = sum w_i z_i.
fn variance_of(w: &[f64], z: &[f64], a: f64, b: f64, c: f64) -> (f64, f64) {
    let z0: f64 = w.iter().zip(z).map(|(w, z)| w * z).sum();
    (z0, a * (z0 - b).powi(2) + c)
}

/// The noise-free path (dW = 0) of the variance over [0, T] at step `dt`: the factors decayed by
/// the scheme's implicit step, or by their exact exponential decay. Returns (V_T, the mean of the
/// left-point V_k over the steps, which is what the spot step integrates).
fn noise_free(
    kernel: &[(f64, f64)],
    z0: &[f64],
    (a, b, c, lambda): (f64, f64, f64, f64),
    t: f64,
    dt: f64,
    exact: bool,
) -> (f64, f64) {
    let weights: Vec<f64> = kernel.iter().map(|(w, _)| *w).collect();
    let steps = (t / dt).round() as usize;
    let mut z = z0.to_vec();
    let mut integral = 0.0;
    for _ in 0..steps {
        let (zz, v) = variance_of(&weights, &z, a, b, c);
        integral += v * dt;
        for (zi, (_, g)) in z.iter_mut().zip(kernel) {
            *zi = if exact {
                let decay = (-g * dt).exp();
                *zi * decay - lambda * zz * (1.0 - decay) / g
            } else {
                (*zi - lambda * zz * dt) / (1.0 + g * dt)
            };
        }
    }
    (
        variance_of(&weights, &z, a, b, c).1,
        integral / (steps as f64 * dt),
    )
}

/// The range of V0 = a (Z0 - b)^2 + c when Z0 lies in `z` and each of (a, b, c) lies within
/// `half_widths` of its printed value; `a` stays positive over its range.
fn v0_interval(
    z: (f64, f64),
    (a, b, c): (f64, f64, f64),
    (da, db, dc): (f64, f64, f64),
) -> (f64, f64) {
    let (bl, bh) = (b - db, b + db);
    let nearest = (z.0 - bh).max(bl - z.1).max(0.0);
    let farthest = (z.1 - bl).abs().max((z.0 - bh).abs());
    (
        (a - da) * nearest * nearest + (c - dc),
        (a + da) * farthest * farthest + (c + dc),
    )
}

/// Whether some value in `range` rounds to `printed` at two decimals.
fn meets_two_decimals(range: (f64, f64), printed: f64) -> bool {
    range.0 < printed + 0.005 && range.1 >= printed - 0.005
}

/// The time step criterion: the prices at the last two steps differ by less than 2 standard
/// errors, the larger of the two.
fn converged_pair((c3, s3): (f64, f64), (c4, s4): (f64, f64)) -> bool {
    (c4 - c3).abs() < 2.0 * s3.max(s4)
}

/// The skew criterion: the lower strike's implied volatility is above the at-the-money one by
/// more than 3 combined standard errors.
fn skew_significant((low, low_se): (f64, f64), (atm, atm_se): (f64, f64)) -> bool {
    low - atm > 3.0 * (low_se * low_se + atm_se * atm_se).sqrt()
}

fn step_1b(criteria: &Value) {
    let base: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(&std::env::args().nth(1).unwrap()).with_file_name("criteria.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let model = &base["model"];
    let (alpha, factors, ratio) = (
        dec(model["alpha"].as_str().unwrap()),
        model["factors"].as_u64().unwrap() as usize,
        dec(model["ratio_x_n"].as_str().unwrap()),
    );
    let kernel = QuadraticRoughHestonKernel::try_new(alpha, factors, ratio).unwrap();
    let p = params(&base, kernel.clone());
    let raw = &model["parameters_2212_10164"];
    let num = |k: &str| -> f64 { raw[k].as_str().unwrap().parse().unwrap() };
    let (a, b, c, lambda) = (num("a"), num("b"), num("c"), num("lambda"));
    let z0: Vec<f64> = raw["z0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().parse().unwrap())
        .collect();
    let checks = &criteria["checks"];
    let mut report = serde_json::Map::new();

    // 1. Initial variance: every reading, none chosen by its value.
    let closed = closed_form(f(alpha), factors, f(ratio));
    let weights: Vec<f64> = closed.iter().map(|(w, _)| *w).collect();
    let ga = gamma(f(alpha));
    let unscaled: Vec<f64> = weights.iter().map(|w| w * ga).collect();
    let reversed: Vec<f64> = z0.iter().rev().copied().collect();
    let ones = vec![1.0; factors];
    let (r1_z, r1_v) = variance_of(&weights, &z0, a, b, c);
    let (r2_z, r2_v) = variance_of(&unscaled, &z0, a, b, c);
    let (r3_z, r3_v) = variance_of(&weights, &reversed, a, b, c);
    let (r4_z, r4_v) = variance_of(&ones, &z0, a, b, c);
    // R1 over the rounding of the printed values: Z0 moves by at most 0.0005 sum c_i.
    let spread = 0.0005 * weights.iter().sum::<f64>();
    let (lo, hi) = v0_interval(
        (r1_z - spread, r1_z + spread),
        (a, b, c),
        (0.0005, 0.0005, 0.00005),
    );
    let printed: f64 = 0.18;
    let r1_rounds = (r1_v * 100.0).round() / 100.0 == printed;
    let interval_meets = meets_two_decimals((lo, hi), printed);
    let shifted = ((printed - c) / a).sqrt();
    report.insert(
        "initial_variance".into(),
        json!({
        "readings": {
            "R1": {"z0": r1_z, "v0": r1_v, "optionstratlib_v0": p.initial_variance(),
                   "supported_by_the_text": true},
            "R1_rounding": {"z0_spread": spread, "v0_low": lo, "v0_high": hi,
                            "supported_by_the_text": true},
            "R2": {"z0": r2_z, "v0": r2_v, "supported_by_the_text": false},
            "R3": {"z0": r3_z, "v0": r3_v, "supported_by_the_text": false},
            "R4": {"z0": r4_z, "v0": r4_v, "supported_by_the_text": false}},
        "printed": printed,
        "z0_giving_printed_v0": [b - shifted, b + shifted],
        "r1_rounds_to_printed": r1_rounds, "r1_rounding_interval_meets_printed": interval_meets,
        "pass": r1_rounds || interval_meets}),
    );

    // 2. The time step sweep, and the noise-free diagnosis.
    let grid = &criteria["grid"];
    let floats = |key: &str| -> Vec<f64> {
        grid[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let (maturities, moneyness) = (floats("maturities"), floats("log_moneyness"));
    let (paths, seed) = (
        grid["paths"].as_u64().unwrap() as usize,
        grid["seed"].as_u64().unwrap(),
    );
    let dts: Vec<f64> = checks["dt_convergence"]["dts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let finest = *dts.last().unwrap();
    let spot = f(p.spot().to_dec());
    let mut rows = vec![];
    let (mut converged, mut parity_ok, mut bounds_ok, mut skew_ok) = (true, true, true, true);
    let mut skews = vec![];
    let mut atm = vec![];
    for &t in &maturities {
        let mut vols = vec![];
        for &k in &moneyness {
            let strike = spot * f64::exp(k);
            let sweep: Vec<(f64, f64)> = dts
                .iter()
                .map(|dt| {
                    price(
                        &p,
                        t,
                        (t / dt).round() as usize,
                        OptionStyle::Call,
                        strike,
                        paths,
                        seed,
                    )
                })
                .collect();
            let n = sweep.len();
            let ((c3, s3), (c4, s4)) = (sweep[n - 2], sweep[n - 1]);
            let row_converged = converged_pair((c3, s3), (c4, s4));
            converged &= row_converged;
            let (call, call_se) = (c4, s4);
            let steps = (t / finest).round() as usize;
            let (put, put_se) = price(&p, t, steps, OptionStyle::Put, strike, paths, seed);
            let parity = call - put - (spot - strike);
            let parity_se = (call_se * call_se + put_se * put_se).sqrt();
            parity_ok &= parity.abs() <= 3.0 * parity_se;
            let intrinsic = (spot - strike).max(0.0);
            bounds_ok &= call >= intrinsic - 3.0 * call_se && call <= spot + 3.0 * call_se;
            let vol = implied_vol(call, spot, strike, t);
            let vol_se = vol.map(|v| call_se / bs_vega(spot, strike, t, v));
            vols.push((k, vol, vol_se));
            if k == 0.0 {
                atm.push(vol);
            }
            rows.push(json!({"maturity": t, "log_moneyness": k, "strike": strike,
                "calls_by_dt": sweep.iter().zip(&dts).map(|((c, s), dt)|
                    json!({"dt": dt, "call": c, "se": s})).collect::<Vec<_>>(),
                "last_two_differ_by": c4 - c3, "tolerance_2se": 2.0 * s3.max(s4),
                "converged": row_converged, "put": put, "put_se": put_se,
                "parity_residual": parity, "parity_se": parity_se, "intrinsic": intrinsic,
                "implied_vol": vol, "implied_vol_se": vol_se}));
        }
        let at = |m: f64| {
            vols.iter()
                .find(|(k, _, _)| (*k - m).abs() < 1e-12)
                .unwrap()
        };
        let ((_, low, low_se), (_, mid, mid_se)) = (at(-0.1), at(0.0));
        let skew = match (low, low_se, mid, mid_se) {
            (Some(l), Some(ls), Some(m), Some(ms)) => {
                let combined = (ls * ls + ms * ms).sqrt();
                json!({"maturity": t, "iv_minus_010": l, "iv_atm": m, "difference": l - m,
                       "combined_se": combined, "pass": skew_significant((*l, *ls), (*m, *ms))})
            }
            _ => json!({"maturity": t, "pass": false, "why": "no implied volatility"}),
        };
        skew_ok &= skew["pass"].as_bool() == Some(true);
        skews.push(skew);
    }
    let noise: Vec<Value> = maturities
        .iter()
        .flat_map(|&t| {
            let closed = &closed;
            let z0 = &z0;
            dts.iter().map(move |&dt| {
                let (vi, mi) = noise_free(closed, z0, (a, b, c, lambda), t, dt, false);
                let (ve, me) = noise_free(closed, z0, (a, b, c, lambda), t, dt, true);
                json!({"maturity": t, "dt": dt, "implicit": {"v_t": vi, "mean_left_point_v": mi},
                       "exact_decay": {"v_t": ve, "mean_left_point_v": me}})
            })
        })
        .collect();
    report.insert(
        "dt_convergence".into(),
        json!({
        "paths": paths, "seed": seed, "dts": dts, "rows": rows,
        "noise_free_diagnosis": {"max_gamma_dt_at_coarsest":
            closed.last().unwrap().1 * dts[0], "paths": noise},
        "pass": converged}),
    );
    report.insert(
        "no_arbitrage_bounds".into(),
        json!({"dt": finest, "pass": bounds_ok}),
    );
    report.insert(
        "negative_skew".into(),
        json!({"dt": finest, "by_maturity": skews,
        "pass": skew_ok}),
    );
    report.insert(
        "put_call_parity".into(),
        json!({"dt": finest, "pass": parity_ok}),
    );
    report.insert(
        "reported_only".into(),
        json!({
        "atm_implied_vols_at_finest_dt": atm, "sqrt_v0_r1": r1_v.sqrt()}),
    );
    let pass = report
        .iter()
        .filter(|(k, _)| k.as_str() != "reported_only")
        .all(|(_, v)| v["pass"].as_bool() == Some(true));
    report.insert("pass".into(), json!(pass));
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(report)).unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_closed_form_partition_is_optionstratlibs() {
        let kernel = QuadraticRoughHestonKernel::try_new(dec("0.51"), 10, dec("3.92")).unwrap();
        for (i, (c, g)) in closed_form(0.51, 10, 3.92).iter().enumerate() {
            assert!((f(kernel.weights()[i]) / c - 1.0).abs() < 1e-10);
            assert!((f(kernel.discounts()[i]) / g - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn the_variance_is_the_quadratic_of_the_weighted_factors() {
        // Z0 = 2 * 0.5 + 1 * (-3) = -2; V0 = 0.5 * (-2 - 1)^2 + 0.1 = 4.6.
        let (z, v) = variance_of(&[2.0, 1.0], &[0.5, -3.0], 0.5, 1.0, 0.1);
        assert_eq!(z, -2.0);
        assert!((v - 4.6).abs() < 1e-12);
    }

    #[test]
    fn the_interval_reaches_c_when_z0_can_equal_b() {
        // Z0 in [0.9, 1.1] meets b in [0.95, 1.05] inside both ranges, at no corner of either.
        let (lo, hi) = v0_interval((0.9, 1.1), (2.0, 1.0, 0.1), (0.1, 0.05, 0.01));
        assert!((lo - 0.09).abs() < 1e-12);
        // Farthest |Z0 - b| is 1.1 - 0.95 = 0.15: 2.1 * 0.0225 + 0.11.
        assert!((hi - (2.1 * 0.0225 + 0.11)).abs() < 1e-12);
    }

    #[test]
    fn the_interval_is_the_corners_when_z0_is_away_from_b() {
        let (lo, hi) = v0_interval(
            (-0.74, -0.72),
            (0.265, 0.246, 0.0001),
            (0.0005, 0.0005, 0.00005),
        );
        assert!((lo - (0.2645 * (0.9655_f64).powi(2) + 0.00005)).abs() < 1e-12);
        assert!((hi - (0.2655 * (0.9865_f64).powi(2) + 0.00015)).abs() < 1e-12);
    }

    #[test]
    fn a_range_meets_a_printed_value_only_through_its_rounding_interval() {
        assert!(meets_two_decimals((0.17, 0.176), 0.18));
        assert!(meets_two_decimals((0.1849, 0.19), 0.18));
        assert!(!meets_two_decimals((0.185, 0.19), 0.18));
        assert!(!meets_two_decimals((0.16, 0.1749), 0.18));
        // The lower end of the rounding interval is in it: 0.175 rounds to 0.18.
        assert!(meets_two_decimals((0.1, 0.18 - 0.005), 0.18));
    }

    #[test]
    fn convergence_needs_the_last_two_within_two_standard_errors() {
        assert!(converged_pair((10.0, 0.2), (10.39, 0.1)));
        assert!(!converged_pair((10.0, 0.2), (10.4, 0.1)));
        assert!(!converged_pair((10.4, 0.1), (10.0, 0.2)));
    }

    #[test]
    fn the_skew_must_clear_three_combined_standard_errors() {
        // Combined standard error 0.05: the difference must exceed 0.15.
        assert!(skew_significant((0.36, 0.03), (0.2, 0.04)));
        assert!(!skew_significant((0.35, 0.03), (0.2, 0.04)));
        assert!(!skew_significant((0.2, 0.03), (0.36, 0.04)));
    }

    #[test]
    fn the_noise_free_factor_decays_by_each_rule() {
        // One factor, lambda 0: Z^1_n = z (1 + g dt)^-n by the implicit step, z exp(-g t) exactly.
        let kernel = [(1.0, 100.0)];
        let (a, b, c) = (1.0, 0.0, 0.0);
        let (implicit, _) = noise_free(&kernel, &[1.0], (a, b, c, 0.0), 0.01, 0.001, false);
        let (exact, _) = noise_free(&kernel, &[1.0], (a, b, c, 0.0), 0.01, 0.001, true);
        assert!((implicit - 1.1_f64.powi(-20)).abs() < 1e-12);
        assert!((exact - (-2.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn the_noise_free_mean_uses_the_left_point() {
        // One step: the mean is V_0 alone.
        let (_, mean) = noise_free(
            &[(1.0, 100.0)],
            &[2.0],
            (1.0, 0.0, 0.5, 0.0),
            0.01,
            0.01,
            false,
        );
        assert!((mean - 4.5).abs() < 1e-12);
    }

    #[test]
    fn the_noise_free_forcing_pulls_by_lambda() {
        // One factor with weight 1, gamma 1, lambda 2, one step of 0.1 from z 1:
        // implicit (1 - 2 * 1 * 0.1) / 1.1; exact 1 e^-0.1 - 2 (1 - e^-0.1).
        let kernel = [(1.0, 1.0)];
        let (vi, _) = noise_free(&kernel, &[1.0], (1.0, 0.0, 0.0, 2.0), 0.1, 0.1, false);
        let (ve, _) = noise_free(&kernel, &[1.0], (1.0, 0.0, 0.0, 2.0), 0.1, 0.1, true);
        let e = (-0.1_f64).exp();
        assert!((vi - (0.8_f64 / 1.1).powi(2)).abs() < 1e-12);
        assert!((ve - (e - 2.0 * (1.0 - e)).powi(2)).abs() < 1e-12);
    }

    #[test]
    fn the_implied_volatility_inverts_black_scholes_and_vega_is_its_slope() {
        let (s, k, t, vol) = (3000.0, 2900.0, 0.05, 0.23);
        let price = bs_call(s, k, t, vol);
        assert!((implied_vol(price, s, k, t).unwrap() - vol).abs() < 1e-6);
        assert!(implied_vol(s - k, s, k, t).is_none());
        let h = 1e-5;
        let slope = (bs_call(s, k, t, vol + h) - bs_call(s, k, t, vol - h)) / (2.0 * h);
        assert!((bs_vega(s, k, t, vol) / slope - 1.0).abs() < 1e-4);
    }
}
