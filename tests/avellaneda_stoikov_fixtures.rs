//! Mathematical validation of the Avellaneda-Stoikov baseline (#85).
//!
//! # Declared before comparing
//!
//! **Model** (Avellaneda and Stoikov 2008, eqs. for the finite horizon
//! approximation): mid price `ds = sigma dW`, exponential utility with risk
//! aversion `gamma`, fill intensity `lambda(delta) = A exp(-k delta)`.
//!
//! - reservation price `r = s - q * gamma * sigma^2 * (T - t)`
//! - optimal total spread `delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)`
//! - quotes `bid = r - delta / 2`, `ask = r + delta / 2`
//!
//! **Units** (the convention the implementation follows):
//! - `s`, `r`, `delta`, quotes: price units;
//! - `q`: base units, signed (long positive);
//! - `sigma`: **absolute** price volatility, price units per sqrt(year); it is
//!   not a relative (percentage) volatility;
//! - `T - t`: milliseconds in the API, converted to years of 365 days
//!   (31 536 000 s);
//! - `gamma`: 1 / price units; `k`: 1 / price units; `A` does not enter the
//!   quotes.
//!
//! **Reference values** were computed outside the quoting code with Python's
//! `decimal` module at 50 significant digits:
//!
//! ```text
//! from decimal import Decimal as D, getcontext; getcontext().prec = 50
//! t = D(tau_ms) / 1000 / D(31536000)
//! r = s - q*g*sig*sig*t
//! d = g*sig*sig*t + (D(2)/g) * (1 + g/k).ln()
//! ```
//!
//! **Tolerance**: the implementation evaluates `ln` and `powi` through `f64`
//! (`types::decimal`), so values are compared with a relative tolerance of
//! `1e-12` (absolute `1e-12` near zero). The continuous solution is compared
//! before any tick rounding; rounding to tick belongs to the execution layer
//! and is checked in `avellaneda_stoikov_simulation.rs`.

use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::strategy::avellaneda_stoikov::{
    calculate_optimal_quotes, calculate_optimal_spread, calculate_reservation_price,
};
use market_maker_rs::types::error::MMError;
use std::str::FromStr;

const REL_TOL: Decimal = dec!(0.000000000001);
/// Tolerance for identities evaluated in exact Decimal arithmetic, where
/// only the 28th significant digit can differ.
const EXACT_TOL: Decimal = dec!(0.00000000000000000001);

fn d(value: &str) -> Decimal {
    Decimal::from_str(value).expect("fixture literal")
}

fn assert_close(label: &str, expected: Decimal, actual: Decimal) {
    let scale = expected.abs().max(Decimal::ONE);
    let error = (expected - actual).abs();
    assert!(
        error <= REL_TOL * scale,
        "{label}: expected {expected}, actual {actual}, error {error}"
    );
}

/// (s, q, gamma, sigma, tau_ms, k, r, delta, bid, ask)
type Row = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    u64,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
);

const FIXTURES: &[Row] = &[
    (
        "100",
        "0",
        "0.1",
        "2",
        3_600_000,
        "1.5",
        "100",
        "1.29081608485188005446",
        "99.35459195757405997277",
        "100.64540804242594002723",
    ),
    (
        "100",
        "5",
        "0.1",
        "2",
        3_600_000,
        "1.5",
        "99.99977168949771689498",
        "1.29081608485188005446",
        "99.35436364707177686775",
        "100.64517973192365692221",
    ),
    (
        "100",
        "-5",
        "0.1",
        "2",
        3_600_000,
        "1.5",
        "100.00022831050228310502",
        "1.29081608485188005446",
        "99.35482026807634307779",
        "100.64563635292822313225",
    ),
    (
        "60000",
        "0.5",
        "0.01",
        "12000",
        86_400_000,
        "0.05",
        "58027.39726027397260273973",
        "3981.66979081084571976289",
        "56036.56236486854974285828",
        "60018.23215567939546262117",
    ),
    (
        "60000",
        "-2",
        "0.01",
        "12000",
        600_000,
        "0.05",
        "60054.79452054794520547945",
        "63.86157163276352798207",
        "60022.86373473156344148842",
        "60086.72530636432696947049",
    ),
    (
        "50",
        "3",
        "0.5",
        "10",
        0,
        "2",
        "50",
        "0.89257420525683902307",
        "49.55371289737158048847",
        "50.44628710262841951153",
    ),
];

#[test]
fn test_as_quotes_reference_fixtures_match_within_tolerance() {
    for (s, q, g, sig, tau, k, r, delta, bid, ask) in FIXTURES {
        let (s, q, g, sig, k) = (d(s), d(q), d(g), d(sig), d(k));
        let label = format!("s={s} q={q} gamma={g} sigma={sig} tau={tau} k={k}");
        let got_r = calculate_reservation_price(s, q, g, sig, *tau).unwrap();
        let got_delta = calculate_optimal_spread(g, sig, *tau, k).unwrap();
        let (got_bid, got_ask) = calculate_optimal_quotes(s, q, g, sig, *tau, k).unwrap();
        assert_close(&format!("r {label}"), d(r), got_r);
        assert_close(&format!("delta {label}"), d(delta), got_delta);
        assert_close(&format!("bid {label}"), d(bid), got_bid);
        assert_close(&format!("ask {label}"), d(ask), got_ask);
    }
}

#[test]
fn test_as_quotes_zero_inventory_symmetric_around_mid() {
    let (bid, ask) = calculate_optimal_quotes(
        dec!(100),
        Decimal::ZERO,
        dec!(0.1),
        dec!(2),
        3_600_000,
        dec!(1.5),
    )
    .unwrap();
    assert!((dec!(100) - bid - (ask - dec!(100))).abs() <= EXACT_TOL);
}

#[test]
fn test_as_quotes_inventory_sign_flips_skew_keeps_spread() {
    let quotes = |q: Decimal| {
        calculate_optimal_quotes(dec!(100), q, dec!(0.1), dec!(2), 3_600_000, dec!(1.5)).unwrap()
    };
    let (flat_bid, flat_ask) = quotes(Decimal::ZERO);
    let (long_bid, long_ask) = quotes(dec!(5));
    let (short_bid, short_ask) = quotes(dec!(-5));
    // Long inventory lowers both quotes, short raises them, by equal amounts.
    assert!(long_bid < flat_bid && long_ask < flat_ask);
    assert!(short_bid > flat_bid && short_ask > flat_ask);
    // Equal up to Decimal's last digit (28 significant digits).
    assert!(((flat_bid - long_bid) - (short_bid - flat_bid)).abs() <= EXACT_TOL);
    // The spread is the same at every inventory.
    assert_close("long spread", flat_ask - flat_bid, long_ask - long_bid);
    assert_close("short spread", flat_ask - flat_bid, short_ask - short_bid);
}

#[test]
fn test_as_spread_zero_horizon_leaves_only_intensity_term() {
    // At T - t = 0 the reservation price is the mid and the spread is
    // (2 / gamma) ln(1 + gamma / k), whatever the inventory.
    let r = calculate_reservation_price(dec!(50), dec!(3), dec!(0.5), dec!(10), 0).unwrap();
    assert_eq!(r, dec!(50));
    let spread = calculate_optimal_spread(dec!(0.5), dec!(10), 0, dec!(2)).unwrap();
    assert_close(
        "spread at zero horizon",
        d("0.89257420525683902307"),
        spread,
    );
}

#[test]
fn test_as_quotes_invalid_parameters_return_typed_errors() {
    let s = dec!(100);
    let q = Decimal::ONE;
    let tau = 3_600_000;
    for gamma in [Decimal::ZERO, dec!(-0.1)] {
        assert!(matches!(
            calculate_reservation_price(s, q, gamma, dec!(2), tau),
            Err(MMError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            calculate_optimal_spread(gamma, dec!(2), tau, dec!(1.5)),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    for sigma in [Decimal::ZERO, dec!(-2)] {
        assert!(matches!(
            calculate_reservation_price(s, q, dec!(0.1), sigma, tau),
            Err(MMError::InvalidMarketState(_))
        ));
        assert!(matches!(
            calculate_optimal_spread(dec!(0.1), sigma, tau, dec!(1.5)),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    for k in [Decimal::ZERO, dec!(-1.5)] {
        assert!(matches!(
            calculate_optimal_spread(dec!(0.1), dec!(2), tau, k),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    for mid in [Decimal::ZERO, dec!(-1)] {
        assert!(matches!(
            calculate_reservation_price(mid, q, dec!(0.1), dec!(2), tau),
            Err(MMError::InvalidMarketState(_))
        ));
    }
    // A negative horizon cannot be expressed: the API takes an unsigned
    // number of milliseconds.
}

#[test]
fn test_as_quotes_negative_reservation_price_refused() {
    // Large inventory and horizon push the reservation price below zero
    // (reference r = -43.7655); the model leaves its domain and the quote
    // function refuses instead of returning a negative bid.
    let result = calculate_optimal_quotes(
        d("1.2345"),
        dec!(1000),
        dec!(2),
        d("0.15"),
        31_536_000_000,
        dec!(50),
    );
    assert!(matches!(result, Err(MMError::InvalidQuoteGeneration(_))));
}

/// Prints the discrepancy table (fixture, expected, actual, error,
/// tolerance) attached to the validation report. Run with
/// `cargo test --test avellaneda_stoikov_fixtures -- --ignored --nocapture`.
#[test]
#[ignore = "evidence table, run on demand"]
fn test_as_quotes_fixtures_print_discrepancy_table() {
    println!("| fixture | quantity | expected | actual | abs error | tolerance |");
    println!("|---|---|---|---|---|---|");
    for (i, (s, q, g, sig, tau, k, r, delta, bid, ask)) in FIXTURES.iter().enumerate() {
        let (s, q, g, sig, k) = (d(s), d(q), d(g), d(sig), d(k));
        let got_r = calculate_reservation_price(s, q, g, sig, *tau).unwrap();
        let got_delta = calculate_optimal_spread(g, sig, *tau, k).unwrap();
        let (got_bid, got_ask) = calculate_optimal_quotes(s, q, g, sig, *tau, k).unwrap();
        for (name, expected, actual) in [
            ("r", d(r), got_r),
            ("delta", d(delta), got_delta),
            ("bid", d(bid), got_bid),
            ("ask", d(ask), got_ask),
        ] {
            let tol = REL_TOL * expected.abs().max(Decimal::ONE);
            println!(
                "| {} | {name} | {expected} | {actual} | {} | {tol} |",
                i + 1,
                (expected - actual).abs()
            );
        }
    }
}
