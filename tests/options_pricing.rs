//! Validation of `options::pricing` (#95): a reference price table, put-call
//! parity, the payoff at expiry, Greeks against finite differences of the
//! same price function under both dynamics, a cross-check against
//! optionstratlib on a flat surface, the Black helpers and explicit errors.
//!
//! Every step and tolerance is declared below, before the checks.
#![cfg(feature = "options")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use market_maker_rs::Decimal;
use market_maker_rs::options::pricing::{
    Dynamics, OptionInputs, OptionPricer, SurfacePricer, black_undiscounted, black_w, value,
};
use market_maker_rs::options::surface::{FlatSurface, TotalVariance, VolSurface};
use market_maker_rs::types::error::MMError;
use optionstratlib::{OptionStyle, OptionType};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal_macros::dec;

// ------------------------------------------------------------ tolerances

/// Relative tolerance against the 60-digit reference prices.
const REFERENCE_REL_TOL: f64 = 1e-12;
/// Put-call parity, absolute, at prices of order 1 to 100.
const PARITY_TOL: f64 = 1e-11;
/// Spot step for delta and gamma, relative to the one-standard-deviation
/// move `S sigma sqrt(tau)` (sigma taken as 0.25), so the truncation error
/// is the same at every expiry.
const FD_REL_STEP_S: f64 = 1e-3;
/// Volatility step for vega.
const FD_STEP_VOL: f64 = 1e-5;
/// Time step for theta, in years.
const FD_STEP_TAU: f64 = 1e-6;
const DELTA_TOL: f64 = 1e-7;
const GAMMA_REL_TOL: f64 = 1e-4;
const GAMMA_ABS_TOL: f64 = 1e-7;
const VEGA_REL_TOL: f64 = 1e-6;
const VEGA_ABS_TOL: f64 = 1e-7;
const THETA_REL_TOL: f64 = 1e-5;
const THETA_ABS_TOL: f64 = 1e-5;

fn f(d: Decimal) -> f64 {
    d.to_f64().unwrap()
}

fn d(x: f64) -> Decimal {
    Decimal::from_f64(x).unwrap()
}

fn close(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    (got - want).abs() <= abs + rel * want.abs()
}

/// A smooth skewed surface for the finite-difference checks:
/// `sigma(k) = sqrt(a + b k + c k^2) + shift`, `w = sigma^2 tau`. It has no
/// nodes, so theta can use central differences everywhere.
struct SkewSurface {
    a: f64,
    b: f64,
    c: f64,
    shift: f64,
}

impl SkewSurface {
    fn new(shift: f64) -> Self {
        Self {
            a: 0.04,
            b: -0.05,
            c: 0.1,
            shift,
        }
    }
}

impl VolSurface for SkewSurface {
    fn total_variance_at(
        &self,
        k: f64,
        tau: f64,
    ) -> market_maker_rs::types::error::MMResult<TotalVariance> {
        let p = self.a + self.b * k + self.c * k * k;
        let root = p.sqrt();
        let sig = root + self.shift;
        let p1 = self.b + 2.0 * self.c * k;
        let sig_k = p1 / (2.0 * root);
        let sig_kk = 2.0 * self.c / (2.0 * root) - p1 * p1 / (4.0 * p * root);
        TotalVariance::new(
            sig * sig * tau,
            2.0 * sig * sig_k * tau,
            2.0 * (sig_k * sig_k + sig * sig_kk) * tau,
            sig * sig,
        )
    }
}

/// A flat surface at `sigma + shift`, for vega on the flat case.
struct ShiftedFlat(f64);

impl VolSurface for ShiftedFlat {
    fn total_variance_at(
        &self,
        _k: f64,
        tau: f64,
    ) -> market_maker_rs::types::error::MMResult<TotalVariance> {
        let s = self.0;
        TotalVariance::new(s * s * tau, 0.0, 0.0, s * s)
    }
}

fn inputs(
    spot: f64,
    strike: f64,
    tau: f64,
    rate: f64,
    carry: f64,
    style: OptionStyle,
) -> OptionInputs {
    OptionInputs::european(d(spot), d(strike), d(tau), d(rate), d(carry), style)
}

fn price(i: &OptionInputs, s: &dyn VolSurface, dynamics: Dynamics) -> f64 {
    f(value(i, s, dynamics).unwrap().price)
}

// ------------------------------------------------------------ reference table

/// One row: spot, strike, tau, rate, carry, sigma, call, 60-digit
/// reference, published value, half a unit of the published precision.
type ReferenceRow = (f64, f64, f64, f64, f64, f64, bool, f64, f64, f64);

/// (spot, strike, tau, rate, carry, sigma, call, 60-digit reference, published).
///
/// The reference column is `scripts/erfc_reference.py` (60-digit decimal
/// arithmetic). The published column is the value printed in the source, at
/// its precision:
/// - Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
///   chapter 1: the Black-Scholes call example (2.1334), the Merton (1973)
///   put on a dividend-paying stock (2.4648) and the generalised
///   Black-Scholes-Merton put with cost of carry b = 0.05 (4.0870);
/// - Hull (2018), *Options, Futures, and Other Derivatives*, 10th ed., the
///   Black-Scholes-Merton example with S0 = 42, K = 40 (c = 4.76, p = 0.81).
const REFERENCE: &[ReferenceRow] = &[
    (
        60.0,
        65.0,
        0.25,
        0.08,
        0.0,
        0.3,
        true,
        2.133_368_444_916_2,
        2.1334,
        5e-5,
    ),
    (
        100.0,
        95.0,
        0.5,
        0.10,
        0.05,
        0.2,
        false,
        2.464_787_646_755_821,
        2.4648,
        5e-5,
    ),
    (
        75.0,
        70.0,
        0.5,
        0.10,
        0.05,
        0.35,
        false,
        4.086_953_828_635_353_5,
        4.0870,
        5e-5,
    ),
    (
        42.0,
        40.0,
        0.5,
        0.10,
        0.0,
        0.2,
        true,
        4.759_422_392_871_533,
        4.76,
        5e-3,
    ),
    (
        42.0,
        40.0,
        0.5,
        0.10,
        0.0,
        0.2,
        false,
        0.808_599_372_900_093_5,
        0.81,
        5e-3,
    ),
];

#[test]
fn test_value_reference_table_matches_high_precision_and_published_prices() {
    for &(s, k, t, r, q, sig, call, reference, published, half_unit) in REFERENCE {
        let style = if call {
            OptionStyle::Call
        } else {
            OptionStyle::Put
        };
        let surface = FlatSurface::new(d(sig)).unwrap();
        let got = price(
            &inputs(s, k, t, r, q, style),
            &surface,
            Dynamics::StickyMoneyness,
        );
        assert!(
            close(got, reference, REFERENCE_REL_TOL, 0.0),
            "{s} {k} {t} {r} {q} {sig} {call}: {got} vs {reference}"
        );
        assert!(
            (got - published).abs() <= half_unit,
            "{got} vs published {published}"
        );
    }
}

// ------------------------------------------------------------ parity and expiry

#[test]
fn test_value_put_call_parity_holds_on_both_surfaces_and_dynamics() {
    let skew = SkewSurface::new(0.0);
    let flat = FlatSurface::new(dec!(0.25)).unwrap();
    let surfaces: [&dyn VolSurface; 2] = [&flat, &skew];
    for surface in surfaces {
        for dynamics in [
            Dynamics::StickyMoneyness,
            Dynamics::StickyStrike { s_ref: dec!(100) },
        ] {
            for spot in [50.0, 90.0, 100.0, 110.0, 200.0] {
                for tau in [1.0 / 365.0, 0.25, 2.0] {
                    let (r, q) = (0.03, 0.01);
                    let c = price(
                        &inputs(spot, 100.0, tau, r, q, OptionStyle::Call),
                        surface,
                        dynamics,
                    );
                    let p = price(
                        &inputs(spot, 100.0, tau, r, q, OptionStyle::Put),
                        surface,
                        dynamics,
                    );
                    let want = spot * (-q * tau).exp() - 100.0 * (-r * tau).exp();
                    assert!(
                        (c - p - want).abs() <= PARITY_TOL,
                        "{spot} {tau}: {}",
                        c - p - want
                    );
                }
            }
        }
    }
}

#[test]
fn test_value_at_expiry_is_the_payoff_without_the_surface() {
    // A surface that always fails: at tau = 0 it must not be consulted.
    struct Failing;
    impl VolSurface for Failing {
        fn total_variance_at(
            &self,
            _: f64,
            _: f64,
        ) -> market_maker_rs::types::error::MMResult<TotalVariance> {
            Err(MMError::NumericalError("consulted".to_string()))
        }
    }
    for (spot, call, price_want, delta_want) in [
        (dec!(110), true, dec!(10), dec!(1)),
        (dec!(100), true, dec!(0), dec!(0)),
        (dec!(90), true, dec!(0), dec!(0)),
        (dec!(110), false, dec!(0), dec!(0)),
        (dec!(100), false, dec!(0), dec!(-1)),
        (dec!(90), false, dec!(10), dec!(-1)),
    ] {
        let style = if call {
            OptionStyle::Call
        } else {
            OptionStyle::Put
        };
        let i = OptionInputs::european(spot, dec!(100), dec!(0), dec!(0.05), dec!(0), style);
        let v = value(&i, &Failing, Dynamics::StickyMoneyness).unwrap();
        assert_eq!(
            (v.price, v.delta),
            (price_want, delta_want),
            "{spot} {call}"
        );
        assert_eq!((v.gamma, v.vega, v.theta), (dec!(0), dec!(0), dec!(0)));
    }
}

#[test]
fn test_value_one_second_before_expiry_is_continuous_with_the_payoff() {
    let flat = FlatSurface::new(dec!(0.2)).unwrap();
    let tau = 1.0 / (365.0 * 86_400.0);
    for (spot, call) in [
        (101.0, true),
        (99.0, true),
        (100.0, true),
        (99.0, false),
        (101.0, false),
    ] {
        let style = if call {
            OptionStyle::Call
        } else {
            OptionStyle::Put
        };
        let v = value(
            &inputs(spot, 100.0, tau, 0.05, 0.0, style),
            &flat,
            Dynamics::StickyMoneyness,
        )
        .unwrap();
        let payoff = if call {
            (spot - 100.0_f64).max(0.0)
        } else {
            (100.0_f64 - spot).max(0.0)
        };
        // The time value is at most S sigma sqrt(tau) (the ATM value).
        assert!(
            (f(v.price) - payoff).abs() <= spot * 0.2 * tau.sqrt(),
            "{spot} {call}"
        );
        let want_delta = match (call, spot > 100.0) {
            (true, true) => 1.0,
            (false, false) => -1.0,
            _ => 0.0,
        };
        if spot != 100.0 {
            assert!(
                (f(v.delta) - want_delta).abs() <= 1e-9,
                "{spot} {call}: {}",
                v.delta
            );
        }
    }
}

// ------------------------------------------------------------ Greeks

fn check_greeks(label: &str, make: &dyn Fn(f64) -> Box<dyn VolSurface>, dynamics: Dynamics) {
    let surface = make(0.0);
    let (up, down) = (make(FD_STEP_VOL), make(-FD_STEP_VOL));
    for style in [OptionStyle::Call, OptionStyle::Put] {
        for spot in [70.0, 95.0, 100.0, 105.0, 140.0] {
            for tau in [1.0 / 365.0, 0.1, 1.0] {
                let (r, q, k) = (0.03, 0.01, 100.0);
                let tag = format!("{label} {style:?} S={spot} tau={tau}");
                let g = value(
                    &inputs(spot, k, tau, r, q, style),
                    surface.as_ref(),
                    dynamics,
                )
                .unwrap();
                let h = FD_REL_STEP_S * spot * 0.25 * tau.sqrt();
                let p = |s: f64, t: f64, surf: &dyn VolSurface| {
                    price(&inputs(s, k, t, r, q, style), surf, dynamics)
                };
                let (pu, pm, pd) = (
                    p(spot + h, tau, surface.as_ref()),
                    p(spot, tau, surface.as_ref()),
                    p(spot - h, tau, surface.as_ref()),
                );
                let fd_delta = (pu - pd) / (2.0 * h);
                let fd_gamma = (pu - 2.0 * pm + pd) / (h * h);
                let fd_vega =
                    (p(spot, tau, up.as_ref()) - p(spot, tau, down.as_ref())) / (2.0 * FD_STEP_VOL);
                // theta = dV/dt = -dV/dtau.
                let fd_theta = (p(spot, tau - FD_STEP_TAU, surface.as_ref())
                    - p(spot, tau + FD_STEP_TAU, surface.as_ref()))
                    / (2.0 * FD_STEP_TAU);
                assert!(
                    close(f(g.delta), fd_delta, 0.0, DELTA_TOL),
                    "{tag} delta {} vs {fd_delta}",
                    g.delta
                );
                assert!(
                    close(f(g.gamma), fd_gamma, GAMMA_REL_TOL, GAMMA_ABS_TOL),
                    "{tag} gamma {} vs {fd_gamma}",
                    g.gamma
                );
                assert!(
                    close(f(g.vega), fd_vega, VEGA_REL_TOL, VEGA_ABS_TOL),
                    "{tag} vega {} vs {fd_vega}",
                    g.vega
                );
                assert!(
                    close(f(g.theta), fd_theta, THETA_REL_TOL, THETA_ABS_TOL),
                    "{tag} theta {} vs {fd_theta}",
                    g.theta
                );
            }
        }
    }
}

#[test]
fn test_value_greeks_match_finite_differences_on_a_flat_surface() {
    let make = |shift: f64| -> Box<dyn VolSurface> { Box::new(ShiftedFlat(0.25 + shift)) };
    check_greeks("flat", &make, Dynamics::StickyMoneyness);
    check_greeks("flat", &make, Dynamics::StickyStrike { s_ref: dec!(100) });
}

#[test]
fn test_value_greeks_match_finite_differences_on_a_skewed_surface_under_both_dynamics() {
    let make = |shift: f64| -> Box<dyn VolSurface> { Box::new(SkewSurface::new(shift)) };
    check_greeks("skew sticky-moneyness", &make, Dynamics::StickyMoneyness);
    check_greeks(
        "skew sticky-strike",
        &make,
        Dynamics::StickyStrike { s_ref: dec!(100) },
    );
}

#[test]
fn test_value_dynamics_differ_only_through_the_smile() {
    // On a flat surface both dynamics give the same valuation; on a skew they
    // differ in delta and agree in price at the reference spot.
    let flat = FlatSurface::new(dec!(0.2)).unwrap();
    let skew = SkewSurface::new(0.0);
    let i = inputs(100.0, 105.0, 0.5, 0.02, 0.0, OptionStyle::Call);
    let sticky = Dynamics::StickyStrike { s_ref: dec!(100) };
    assert_eq!(
        value(&i, &flat, Dynamics::StickyMoneyness).unwrap(),
        value(&i, &flat, sticky).unwrap()
    );
    // With r != q the two read k at the same forward only when s_ref = S.
    let a = value(&i, &skew, Dynamics::StickyMoneyness).unwrap();
    let b = value(&i, &skew, sticky).unwrap();
    assert_eq!(a.price, b.price);
    assert_ne!(a.delta, b.delta);
}

// ------------------------------------------------------------ optionstratlib

#[test]
fn test_value_flat_surface_agrees_with_optionstratlib() {
    use market_maker_rs::options::OptionsAdapter;
    use optionstratlib::model::ExpirationDate;
    use optionstratlib::model::option::Options;
    use optionstratlib::model::types::Side;
    use optionstratlib::prelude::Positive;

    for style in [OptionStyle::Call, OptionStyle::Put] {
        for (spot, strike, days, sigma, r, q) in [
            (100.0, 100.0, 30.0, 0.2, 0.05, 0.0),
            (100.0, 90.0, 90.0, 0.35, 0.02, 0.01),
            (100.0, 120.0, 365.0, 0.5, 0.0, 0.0),
            (50.0, 55.0, 7.0, 0.8, 0.04, 0.02),
        ] {
            let option = Options::new(
                OptionType::European,
                Side::Long,
                "X".to_string(),
                Positive::new(strike).unwrap(),
                ExpirationDate::Days(Positive::new(days).unwrap()),
                Positive::new(sigma).unwrap(),
                Positive::ONE,
                Positive::new(spot).unwrap(),
                d(r),
                style,
                Positive::new(q).unwrap_or(Positive::ZERO),
                None,
            );
            let theirs = OptionsAdapter::theoretical_value(&option).unwrap();
            let greeks = OptionsAdapter::calculate_greeks(&option).unwrap();
            let surface = FlatSurface::new(d(sigma)).unwrap();
            let i = inputs(spot, strike, days / 365.0, r, q, style);
            let ours = value(&i, &surface, Dynamics::StickyMoneyness).unwrap();
            let tag = format!("{style:?} {spot} {strike} {days} {sigma}");
            assert!(
                close(f(ours.price), f(theirs), 1e-9, 1e-12),
                "{tag} price {} vs {theirs}",
                ours.price
            );
            assert!(
                close(f(ours.delta), f(greeks.delta), 1e-9, 1e-12),
                "{tag} delta {} vs {}",
                ours.delta,
                greeks.delta
            );
            assert!(
                close(f(ours.gamma), f(greeks.gamma), 1e-9, 1e-12),
                "{tag} gamma {} vs {}",
                ours.gamma,
                greeks.gamma
            );
            // optionstratlib reports vega per volatility point and theta per
            // day; ours are per 1.00 of volatility and per year.
            let their_vega = f(greeks.vega) * 100.0;
            let their_theta = f(greeks.theta) * 365.0;
            assert!(
                close(f(ours.vega), their_vega, 1e-9, 1e-10),
                "{tag} vega {} vs {their_vega}",
                ours.vega
            );
            assert!(
                close(f(ours.theta), their_theta, 1e-9, 1e-10),
                "{tag} theta {} vs {their_theta}",
                ours.theta
            );
        }
    }
}

// ------------------------------------------------------------ Black helpers

#[test]
fn test_black_is_increasing_in_w_from_the_intrinsic_value() {
    for &(fwd, strike) in &[
        (1.0_f64, 1000.0_f64),
        (90.0, 100.0),
        (100.0, 100.0),
        (110.0, 100.0),
        (1000.0, 1.0),
    ] {
        for style in [OptionStyle::Call, OptionStyle::Put] {
            let intrinsic = if matches!(style, OptionStyle::Call) {
                (fwd - strike).max(0.0)
            } else {
                (strike - fwd).max(0.0)
            };
            assert_eq!(
                black_undiscounted(fwd, strike, 0.0, style).unwrap(),
                intrinsic
            );
            let mut last = intrinsic;
            let mut representable = false;
            for i in 1..=200 {
                let w = 1e-6 * 1.1_f64.powi(i);
                // Out of the money at the smallest w the true price is below
                // the smallest f64: an error, never 0, and never again once
                // the price is representable.
                let p = match black_undiscounted(fwd, strike, w, style) {
                    Err(MMError::PriceUnderflow) if intrinsic == 0.0 && !representable => {
                        continue;
                    }
                    other => other.unwrap(),
                };
                representable = true;
                assert!(
                    p > 0.0 && p >= last,
                    "{fwd} {strike} {style:?} w={w}: {p} < {last}"
                );
                // Bounds: intrinsic <= C <= F, intrinsic <= P <= K.
                let cap = if matches!(style, OptionStyle::Call) {
                    fwd
                } else {
                    strike
                };
                assert!(
                    p >= intrinsic && p <= cap,
                    "{fwd} {strike} {style:?} w={w}: {p}"
                );
                last = p;
            }
            // Undiscounted parity C - P = F - K.
            let w = 0.3;
            let c = black_undiscounted(fwd, strike, w, OptionStyle::Call).unwrap();
            let p = black_undiscounted(fwd, strike, w, OptionStyle::Put).unwrap();
            assert!((c - p - (fwd - strike)).abs() <= 1e-12 * fwd.max(strike));
        }
    }
}

#[test]
fn test_black_w_matches_a_finite_difference_and_is_singular_at_zero() {
    for &(fwd, strike, w) in &[
        (100.0, 100.0, 0.04),
        (90.0, 100.0, 0.01),
        (130.0, 100.0, 0.2),
    ] {
        let h = 1e-7;
        let fd = (black_undiscounted(fwd, strike, w + h, OptionStyle::Call).unwrap()
            - black_undiscounted(fwd, strike, w - h, OptionStyle::Call).unwrap())
            / (2.0 * h);
        assert!(close(black_w(fwd, strike, w).unwrap(), fd, 1e-6, 1e-9));
    }
    assert!(matches!(
        black_w(100.0, 100.0, 0.0),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn test_black_invalid_inputs_are_errors() {
    for (fwd, strike, w) in [
        (0.0, 1.0, 0.1),
        (1.0, -1.0, 0.1),
        (1.0, 1.0, -1e-300),
        (f64::NAN, 1.0, 0.1),
        (1.0, 1.0, f64::INFINITY),
    ] {
        assert!(matches!(
            black_undiscounted(fwd, strike, w, OptionStyle::Call),
            Err(MMError::InvalidMarketState(_))
        ));
        assert!(black_w(fwd, strike, w).is_err());
    }
}

// ------------------------------------------------------------ errors

#[test]
fn test_value_invalid_or_unsupported_inputs_are_explicit_errors() {
    let flat = FlatSurface::new(dec!(0.2)).unwrap();
    let ok = OptionInputs::european(
        dec!(100),
        dec!(100),
        dec!(0.5),
        dec!(0.01),
        dec!(0),
        OptionStyle::Call,
    );
    for bad in [
        OptionInputs {
            spot: dec!(0),
            ..ok.clone()
        },
        OptionInputs {
            spot: dec!(-1),
            ..ok.clone()
        },
        OptionInputs {
            strike: dec!(0),
            ..ok.clone()
        },
        OptionInputs {
            tau_years: dec!(-0.1),
            ..ok.clone()
        },
    ] {
        assert!(matches!(
            value(&bad, &flat, Dynamics::StickyMoneyness),
            Err(MMError::InvalidMarketState(_))
        ));
    }
    let american = OptionInputs {
        exercise: OptionType::American,
        ..ok.clone()
    };
    assert!(matches!(
        value(&american, &flat, Dynamics::StickyMoneyness),
        Err(MMError::UnsupportedContract { .. })
    ));
    assert!(matches!(
        value(&ok, &flat, Dynamics::StickyStrike { s_ref: dec!(0) }),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(
        SurfacePricer::new(Arc::new(flat), Dynamics::StickyStrike { s_ref: dec!(-1) }).is_err()
    );
    // A rate that makes the discount factor underflow is a numerical error,
    // not a zero price.
    let extreme = OptionInputs {
        rate: dec!(1000),
        tau_years: dec!(10),
        ..ok
    };
    assert!(matches!(
        value(&extreme, &flat, Dynamics::StickyMoneyness),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn test_surface_pricer_is_the_value_function() {
    let flat = Arc::new(FlatSurface::new(dec!(0.3)).unwrap());
    let pricer = SurfacePricer::new(flat.clone(), Dynamics::StickyMoneyness).unwrap();
    let i = inputs(100.0, 95.0, 0.25, 0.01, 0.0, OptionStyle::Put);
    assert_eq!(
        pricer.value(&i).unwrap(),
        value(&i, flat.as_ref(), Dynamics::StickyMoneyness).unwrap()
    );
}

#[test]
fn test_value_prices_are_never_negative_nor_below_the_discounted_intrinsic_on_a_wide_grid() {
    let skew = SkewSurface::new(0.0);
    for spot in [1.0, 20.0, 60.0, 99.0, 100.0, 101.0, 150.0, 500.0, 10_000.0] {
        for tau in [1.0 / 86_400.0 / 365.0, 1.0 / 365.0, 0.1, 1.0, 10.0] {
            for style in [OptionStyle::Call, OptionStyle::Put] {
                for dynamics in [
                    Dynamics::StickyMoneyness,
                    Dynamics::StickyStrike { s_ref: dec!(100) },
                ] {
                    let (r, q) = (0.03, 0.01);
                    let Ok(v) = value(&inputs(spot, 100.0, tau, r, q, style), &skew, dynamics)
                    else {
                        continue; // outside the test surface's domain (p < 0 far in the wings)
                    };
                    let fwd = spot * ((r - q) * tau).exp();
                    let intrinsic = if matches!(style, OptionStyle::Call) {
                        (fwd - 100.0).max(0.0)
                    } else {
                        (100.0 - fwd).max(0.0)
                    } * (-r * tau).exp();
                    let p = f(v.price);
                    assert!(p >= 0.0, "{spot} {tau} {style:?}: {p}");
                    assert!(
                        p >= intrinsic - 1e-12 * intrinsic.max(1.0),
                        "{spot} {tau} {style:?}: {p} < {intrinsic}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_tau_from_days_uses_the_365_day_pricing_year() {
    use market_maker_rs::options::pricing::{PRICING_YEAR_DAYS, tau_from_days};
    assert_eq!(PRICING_YEAR_DAYS, dec!(365));
    assert_eq!(tau_from_days(dec!(73)).unwrap(), dec!(0.2));
    assert_eq!(tau_from_days(dec!(0)).unwrap(), dec!(0));
    assert!(matches!(
        tau_from_days(dec!(-1)),
        Err(MMError::InvalidTimestamp(_))
    ));
}

#[test]
fn test_black_near_the_money_at_tiny_variance_keeps_its_digits() {
    // C = F b(0, s) with b(0, 1e-28) = 3.98942280401432666e-29 (400-digit
    // reference, scripts/black_reference.py): F N(d1) - K N(d2) would give 0.
    let c = black_undiscounted(100.0, 100.0, 1e-56, OptionStyle::Call).unwrap();
    assert!(close(c, 3.989_422_804_014_327e-27, 1e-14, 0.0), "{c:e}");
    let p = black_undiscounted(100.0, 100.0, 1e-56, OptionStyle::Put).unwrap();
    assert_eq!(c, p);
    // The pricer keeps it, to the resolution of Decimal (28 places).
    let flat = FlatSurface::new(Decimal::new(1, 28)).unwrap();
    let i = OptionInputs::european(
        dec!(100),
        dec!(100),
        dec!(1),
        dec!(0),
        dec!(0),
        OptionStyle::Call,
    );
    let v = value(&i, &flat, Dynamics::StickyMoneyness).unwrap();
    assert!(v.price > Decimal::ZERO);
    assert!(
        (f(v.price) - 3.989_422_804_014_327e-27).abs() <= 1e-28,
        "{}",
        v.price
    );
}

#[test]
fn test_black_time_value_is_continuous_across_the_series_boundary() {
    // sqrt(w) = 0.42 is where the small-t series hands over (t = 0.21).
    for &(fwd, strike) in &[(100.0_f64, 100.0_f64), (95.0, 100.0), (100.0, 80.0)] {
        let below =
            black_undiscounted(fwd, strike, (0.42_f64 - 1e-12).powi(2), OptionStyle::Call).unwrap();
        let above =
            black_undiscounted(fwd, strike, (0.42_f64 + 1e-12).powi(2), OptionStyle::Call).unwrap();
        assert!(
            (above - below).abs() <= 1e-9 * above,
            "{fwd} {strike}: {below} {above}"
        );
    }
}

#[test]
fn test_black_near_the_money_at_small_variance_uses_the_exact_log_ratio() {
    // F = 100, w = 1e-28 (s = 1e-14), calls; references in 400-digit
    // arithmetic at the exact f64 inputs (scripts/black_reference.py):
    // ln F - ln K would lose the digits that set h = ln(F/K) / s.
    for (strike, want) in [
        (100.000_000_000_000_01, 3.918_771_352_418_849_3e-13),
        (99.999_999_999_999_99, 4.060_879_899_570_869e-13),
        (100.0, 3.989_422_804_014_326_7e-13),
    ] {
        let got = black_undiscounted(100.0, strike, 1e-28, OptionStyle::Call).unwrap();
        assert!(
            close(got, want, 1e-13, 0.0),
            "K = {strike}: {got:e} vs {want:e}"
        );
    }
    // The pricer at the same inputs given exactly as Decimals.
    let flat = FlatSurface::new(dec!(0.00000000000001)).unwrap();
    for (strike, want) in [
        (dec!(100.00000000000001), 3.939_622_273_492_284_7e-13),
        (dec!(99.99999999999999), 4.039_622_273_492_284_3e-13),
    ] {
        let i = OptionInputs::european(
            dec!(100),
            strike,
            dec!(1),
            dec!(0),
            dec!(0),
            OptionStyle::Call,
        );
        let got = f(value(&i, &flat, Dynamics::StickyMoneyness).unwrap().price);
        assert!(
            close(got, want, 1e-13, 0.0),
            "K = {strike}: {got:e} vs {want:e}"
        );
    }
}

#[test]
fn test_value_theta_of_a_deep_in_the_money_call_has_no_cancellation() {
    // S = 1e24, K = 1, tau = 1, r = 5 %, q = 0, sigma = 0.2: theta is
    // -r K e^{-r tau} N(d2) = -0.05 e^{-0.05} to double precision.
    let flat = FlatSurface::new(dec!(0.2)).unwrap();
    let i = OptionInputs::european(
        dec!(1000000000000000000000000),
        dec!(1),
        dec!(1),
        dec!(0.05),
        dec!(0),
        OptionStyle::Call,
    );
    let v = value(&i, &flat, Dynamics::StickyMoneyness).unwrap();
    let want = -0.05 * (-0.05_f64).exp();
    assert!(close(f(v.theta), want, 1e-12, 0.0), "{} vs {want}", v.theta);
}

#[test]
fn test_surface_pricer_check_market_needs_no_contract() {
    let flat = Arc::new(FlatSurface::new(dec!(0.2)).unwrap());
    let pricer = SurfacePricer::new(flat, Dynamics::StickyMoneyness).unwrap();
    assert!(
        pricer
            .check_market(dec!(100), dec!(0.1), dec!(0.02), dec!(0))
            .is_ok()
    );
    assert!(
        pricer
            .check_market(dec!(100), dec!(0), dec!(0.02), dec!(0))
            .is_ok()
    );
    assert!(matches!(
        pricer.check_market(dec!(0), dec!(0.1), dec!(0), dec!(0)),
        Err(MMError::InvalidMarketState(_))
    ));
    assert!(matches!(
        pricer.check_market(dec!(100), dec!(-0.1), dec!(0), dec!(0)),
        Err(MMError::InvalidMarketState(_))
    ));
    // The shared factors: a discount factor or a forward that is not
    // finite and positive fails for every contract.
    assert!(matches!(
        pricer.check_market(dec!(100), dec!(1), dec!(1000), dec!(0)),
        Err(MMError::NumericalError(_))
    ));
    assert!(matches!(
        pricer.check_market(dec!(100), dec!(1), dec!(0), dec!(-1000)),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn test_surface_pricer_check_market_does_not_evaluate_the_surface() {
    struct Nowhere;
    impl VolSurface for Nowhere {
        fn total_variance_at(
            &self,
            _: f64,
            _: f64,
        ) -> market_maker_rs::types::error::MMResult<TotalVariance> {
            Err(MMError::InvalidMarketState("no answer".to_string()))
        }
    }
    let pricer = SurfacePricer::new(Arc::new(Nowhere), Dynamics::StickyMoneyness).unwrap();
    assert!(
        pricer
            .check_market(dec!(100), dec!(0.25), dec!(0.02), dec!(0))
            .is_ok()
    );
}

#[test]
fn test_value_theta_at_equal_rate_and_carry_and_tiny_vol_has_no_cancellation() {
    // r = q (futures options): the carry part is exactly r V, so theta keeps
    // its digits as sigma falls. References: scripts/black_reference.py
    // ("theta ..."), 400 digits, at the exact decimal inputs.
    let cases = [
        (
            dec!(0.00000000000001),
            dec!(100.000000000001),
            OptionStyle::Put,
            -6.356_075_893_319_161e-14,
        ),
        (
            dec!(0.00000000000001),
            dec!(100),
            OptionStyle::Call,
            -1.707_685_361_078_657_9e-13,
        ),
        (
            dec!(0.00000000000001),
            dec!(100),
            OptionStyle::Put,
            -1.707_685_361_078_657_9e-13,
        ),
        (
            dec!(0.000000000001),
            dec!(100.000000000001),
            OptionStyle::Call,
            -1.709_959_078_540_878_7e-11,
        ),
    ];
    for (sigma, strike, style, want) in cases {
        let flat = FlatSurface::new(sigma).unwrap();
        let i = OptionInputs::european(dec!(100), strike, dec!(1), dec!(0.05), dec!(0.05), style);
        let v = value(&i, &flat, Dynamics::StickyMoneyness).unwrap();
        let rel = ((f(v.theta) - want) / want).abs();
        assert!(
            rel <= 1e-11,
            "sigma {sigma} K {strike} {style:?}: {} vs {want:e} (rel {rel:e})",
            v.theta
        );
    }
}

#[test]
fn test_black_keeps_a_representable_price_whose_normalised_value_underflows() {
    // F = 1e-50, K = 1e306, w = 400: h = -41, b underflows, sqrt(F K) b does
    // not. Reference: scripts/black_reference.py ("black ..."), 400 digits.
    let call = black_undiscounted(1e-50, 1e306, 400.0, OptionStyle::Call).unwrap();
    let want = 1.629_959_562_806_11e-261;
    assert!(close(call, want, 1e-11, 0.0), "{call:e} vs {want:e}");
    // The small-t series branch: t = 0.2, h = -45.
    let call = black_undiscounted(1e150, 6.5e157, 0.16, OptionStyle::Call).unwrap();
    let want = 3.668_096_755_640_983_4e-290;
    assert!(close(call, want, 1e-11, 0.0), "{call:e} vs {want:e}");
    let put = black_undiscounted(1e-50, 1e306, 400.0, OptionStyle::Put).unwrap();
    assert!(close(put, 1e306, 1e-15, 0.0), "{put:e}");
}

#[test]
fn test_black_at_the_edges_of_the_f64_range_has_no_overflow() {
    // F = 1e308, K = 5e-324, w = 1e4: e^{-ht} = e^{|x|/2} overflows while
    // N(h - t) underflows; the price is the intrinsic value for the call and
    // K (the smallest subnormal) for the put.
    let call = black_undiscounted(1e308, 5e-324, 1e4, OptionStyle::Call).unwrap();
    assert_eq!(call, 1e308);
    let put = black_undiscounted(1e308, 5e-324, 1e4, OptionStyle::Put).unwrap();
    assert_eq!(put, f64::from_bits(1), "{put:e}");
}

#[test]
fn test_black_price_below_the_smallest_f64_is_an_error_not_zero() {
    // h = ln(1e-300) / 0.01 = -69078: the true price is positive and far below
    // the smallest subnormal.
    assert_eq!(
        black_undiscounted(1.0, 1e300, 1e-4, OptionStyle::Call),
        Err(MMError::PriceUnderflow)
    );
    assert_eq!(
        black_undiscounted(1e300, 1.0, 1e-4, OptionStyle::Put),
        Err(MMError::PriceUnderflow)
    );
    // w = 0 out of the money is its intrinsic value, exactly 0, not an error.
    assert_eq!(
        black_undiscounted(1.0, 1e300, 0.0, OptionStyle::Call),
        Ok(0.0)
    );
    // Invalid inputs keep their class.
    assert!(matches!(
        black_undiscounted(f64::NAN, 1.0, 1e-4, OptionStyle::Call),
        Err(MMError::InvalidMarketState(_))
    ));
    // The in-the-money side of the same pair is its intrinsic value.
    assert_eq!(
        black_undiscounted(1e300, 1.0, 1e-4, OptionStyle::Call).unwrap(),
        1e300 - 1.0
    );
}

#[test]
fn test_value_theta_of_a_deep_in_the_money_put_has_no_cancellation() {
    // The mirror of the deep in-the-money call: S = 100, K = 1e24, r = 0,
    // q = 5 %: theta is -q S e^{-q tau} N(-d1). Reference:
    // scripts/black_reference.py ("theta ...").
    let flat = FlatSurface::new(dec!(0.2)).unwrap();
    for (spot, want) in [
        (dec!(100), -4.756_147_122_503_57),
        (dec!(1), -4.756_147_122_503_57e-2),
    ] {
        let i = OptionInputs::european(
            spot,
            dec!(1000000000000000000000000),
            dec!(1),
            dec!(0),
            dec!(0.05),
            OptionStyle::Put,
        );
        let v = value(&i, &flat, Dynamics::StickyMoneyness).unwrap();
        assert!(
            close(f(v.theta), want, 1e-12, 0.0),
            "S {spot}: {} vs {want}",
            v.theta
        );
    }
}

#[test]
fn test_black_scaling_never_underflows_an_intermediate() {
    // The mantissa times e^{e} underflows although sqrt(F K) brings the
    // price back: F = 1e300, K one ulp above, w = 1.58e-35 (h about -37,
    // series branch).
    let got = black_undiscounted(
        1e300,
        1.0000000000000002e300,
        1.580842522018662e-35,
        OptionStyle::Call,
    )
    .unwrap();
    let want = 2.073_992_952_514_446e-25;
    assert!(close(got, want, 1e-11, 0.0), "{got:e} vs {want:e}");
    // The direct branch, h = -19 and t = 20: N(h - t) = N(-39) underflows
    // while e^{-ht} N(h - t) is 0.74 % of the price.
    let got = black_undiscounted(1e300, 1e-30, 1600.0, OptionStyle::Put).unwrap();
    let want = 8.360_537_541_944_882e-31;
    assert!(close(got, want, 1e-11, 0.0), "{got:e} vs {want:e}");
}

#[test]
fn test_surface_point_is_where_value_reads_the_surface() {
    use market_maker_rs::options::pricing::surface_point;
    use std::sync::Mutex;
    // A surface that records every point it is asked for.
    struct Recording(Mutex<Vec<(f64, f64)>>);
    impl VolSurface for Recording {
        fn total_variance_at(
            &self,
            k: f64,
            tau: f64,
        ) -> market_maker_rs::types::error::MMResult<TotalVariance> {
            self.0.lock().unwrap().push((k, tau));
            TotalVariance::new(0.04 * tau + 0.01 * k * k, 0.02 * k, 0.02, 0.04)
        }
    }
    for dynamics in [
        Dynamics::StickyMoneyness,
        Dynamics::StickyStrike { s_ref: dec!(99.5) },
    ] {
        for strike in [dec!(100.0000000000000001), dec!(80), dec!(125)] {
            let surface = Recording(Mutex::new(Vec::new()));
            let i = OptionInputs::european(
                dec!(100),
                strike,
                dec!(0.25),
                dec!(0.03),
                dec!(0.01),
                OptionStyle::Call,
            );
            value(&i, &surface, dynamics).unwrap();
            let read = surface.0.lock().unwrap().clone();
            assert_eq!(
                read,
                vec![surface_point(&i, dynamics).unwrap()],
                "{dynamics:?} K {strike}"
            );
        }
    }
    let at_expiry = OptionInputs::european(
        dec!(100),
        dec!(100),
        dec!(0),
        dec!(0),
        dec!(0),
        OptionStyle::Call,
    );
    assert!(matches!(
        surface_point(&at_expiry, Dynamics::StickyMoneyness),
        Err(MMError::InvalidMarketState(_))
    ));
}
