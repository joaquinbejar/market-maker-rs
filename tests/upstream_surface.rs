//! OptionStratLib's surface read as a `VolSurface`, on quotes made from known slices. Synthetic.

use std::sync::Arc;

use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::options::pricing::{Dynamics, OptionInputs, OptionPricer, SurfacePricer};
use market_maker_rs::options::surface::VolSurface;
use market_maker_rs::options::surface::upstream::{
    SurfaceQuality, UpstreamAttempt, UpstreamPolicy, UpstreamStore, UpstreamSurface,
};
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::model::{ExpirationDate, Options};
use optionstratlib::prelude::Positive;
use optionstratlib::pricing::black_scholes_model::black_scholes;
use optionstratlib::volatility::{
    ExtrapolationPolicy, ImpliedVolatilitySurface, ImpliedVolatilitySurfaceConfig, OptionQuote,
    SviParameters, SviSlice,
};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

const STRIKES: [u32; 9] = [70, 80, 90, 95, 100, 105, 110, 120, 135];

fn as_of() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2030, 1, 1, 12, 0, 0).single().unwrap()
}

fn as_of_ms() -> u64 {
    as_of().timestamp_millis() as u64
}

fn positive(value: Decimal) -> Positive {
    Positive::new_decimal(value).unwrap()
}

fn spot() -> Positive {
    positive(dec!(100))
}

fn f(value: Decimal) -> f64 {
    value.to_f64().unwrap()
}

fn svi(a: Decimal) -> SviParameters {
    SviParameters::try_new(a, dec!(0.1), dec!(-0.4), dec!(0.05), dec!(0.2)).unwrap()
}

/// One expiry's fixture: its distance, and the parameters behind its quotes.
#[derive(Clone, Copy)]
struct Expiry {
    days: i64,
    extra_nanos: i64,
    parameters: SviParameters,
    strikes: &'static [u32],
}

fn expiry(days: i64, a: Decimal) -> Expiry {
    Expiry {
        days,
        extra_nanos: 0,
        parameters: svi(a),
        strikes: &STRIKES,
    }
}

/// Exact quotes of each expiry under its parameters, at this rate and yield.
fn quotes(expiries: &[Expiry], rate: Decimal, yield_q: Decimal) -> Vec<OptionQuote> {
    let mut quotes = Vec::new();
    for expiry in expiries {
        let years = Decimal::from(expiry.days) / dec!(365);
        let forward = f(dec!(100)) * (f(rate - yield_q) * f(years)).exp();
        let slice = SviSlice::try_new(
            expiry.parameters,
            positive(Decimal::from_f64(forward).unwrap()),
            positive(years),
        )
        .unwrap();
        for strike in expiry.strikes {
            let strike = positive(Decimal::from(*strike));
            let option = Options::new(
                OptionType::European,
                Side::Long,
                String::new(),
                strike,
                ExpirationDate::Days(positive(Decimal::from(expiry.days))),
                slice.implied_volatility_at_strike(strike).unwrap(),
                Positive::ONE,
                spot(),
                rate,
                OptionStyle::Call,
                positive(yield_q),
                None,
            );
            let price = positive(black_scholes(&option).unwrap());
            quotes.push(OptionQuote {
                expiry: as_of()
                    + TimeDelta::try_days(expiry.days).unwrap()
                    + TimeDelta::nanoseconds(expiry.extra_nanos),
                strike,
                style: OptionStyle::Call,
                bid: price,
                ask: price,
                quoted_at: as_of(),
            });
        }
    }
    quotes
}

fn calibrated_at(
    expiries: &[Expiry],
    rate: Decimal,
    yield_q: Decimal,
    config: &ImpliedVolatilitySurfaceConfig,
    at: DateTime<Utc>,
) -> ImpliedVolatilitySurface {
    ImpliedVolatilitySurface::calibrate(
        &quotes(expiries, rate, yield_q)
            .into_iter()
            .map(|mut quote| {
                quote.quoted_at = at;
                quote
            })
            .collect::<Vec<_>>(),
        spot(),
        rate,
        positive(yield_q),
        at,
        config,
    )
    .unwrap()
}

fn calibrated(expiries: &[Expiry]) -> ImpliedVolatilitySurface {
    calibrated_at(
        expiries,
        dec!(0),
        dec!(0),
        &ImpliedVolatilitySurfaceConfig::default(),
        as_of(),
    )
}

fn three() -> [Expiry; 3] {
    [
        expiry(30, dec!(0.01)),
        expiry(91, dec!(0.02)),
        expiry(182, dec!(0.04)),
    ]
}

fn adapter(expiries: &[Expiry]) -> UpstreamSurface {
    UpstreamSurface::new(calibrated(expiries), UpstreamPolicy::default()).unwrap()
}

/// What the fitted slice `index` of `surface` says at `k`: w, dw/dk, d2w/dk2, and its T.
fn slice_at(surface: &ImpliedVolatilitySurface, index: usize, k: f64) -> (f64, f64, f64, f64) {
    let expiry = &surface.expiries()[index];
    let slice = expiry.slice();
    let k = Decimal::from_f64(k).unwrap();
    (
        f(slice.total_variance(k).unwrap().to_dec()),
        f(slice.total_variance_slope(k).unwrap()),
        f(slice.total_variance_curvature(k).unwrap()),
        f(expiry.time_to_expiry().to_dec()),
    )
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} against {b}");
}

#[test]
fn test_at_a_fitted_expiry_a_read_is_its_slice_and_the_slope_is_the_shorter_segment() {
    let surface = adapter(&three());
    for k in [-0.1, 0.0, 0.08] {
        let (w1, _, _, t1) = slice_at(surface.surface(), 0, k);
        let (w2, w2_k, w2_kk, t2) = slice_at(surface.surface(), 1, k);
        let read = surface.total_variance(k, t2).unwrap();
        close(read.w(), w2, 1e-12);
        close(read.w_k(), w2_k, 1e-12);
        close(read.w_kk(), w2_kk, 1e-12);
        close(read.w_tau(), (w2 - w1) / (t2 - t1), 1e-10);
    }
}

#[test]
fn test_between_two_expiries_a_read_is_linear_in_time_and_its_slope_is_the_segments() {
    let surface = adapter(&three());
    let k = 0.03;
    let (w1, w1_k, w1_kk, t1) = slice_at(surface.surface(), 0, k);
    let (w2, w2_k, w2_kk, t2) = slice_at(surface.surface(), 1, k);
    let theta = 0.25;
    let read = surface.total_variance(k, t1 + theta * (t2 - t1)).unwrap();
    close(read.w(), w1 + theta * (w2 - w1), 1e-10);
    close(read.w_k(), w1_k + theta * (w2_k - w1_k), 1e-10);
    close(read.w_kk(), w1_kk + theta * (w2_kk - w1_kk), 1e-10);
    close(read.w_tau(), (w2 - w1) / (t2 - t1), 1e-10);
}

#[test]
fn test_at_the_first_or_only_expiry_the_time_slope_is_w_over_t() {
    let surface = adapter(&three());
    let (w, _, _, t) = slice_at(surface.surface(), 0, 0.02);
    close(
        surface.total_variance(0.02, t).unwrap().w_tau(),
        w / t,
        1e-12,
    );

    let single = adapter(&[expiry(91, dec!(0.02))]);
    assert!(!single.surface().calendar_checked());
    let (w, _, _, t) = slice_at(single.surface(), 0, 0.02);
    close(
        single.total_variance(0.02, t).unwrap().w_tau(),
        w / t,
        1e-12,
    );
}

#[test]
fn test_a_read_a_hair_from_an_expiry_is_at_it_and_a_step_from_it_is_not() {
    let surface = adapter(&three());
    let k = 0.02;
    let (w1, _, _, t1) = slice_at(surface.surface(), 0, k);
    let (w2, _, _, t2) = slice_at(surface.surface(), 1, k);
    let (w3, _, _, t3) = slice_at(surface.surface(), 2, k);
    let left = (w2 - w1) / (t2 - t1);
    let right = (w3 - w2) / (t3 - t2);
    close(surface.total_variance(k, t2).unwrap().w_tau(), left, 1e-10);
    let hair = surface.total_variance(k, t2 * (1.0 + 1e-12)).unwrap();
    close(hair.w_tau(), left, 1e-10);
    let step = surface.total_variance(k, t2 * (1.0 + 1e-9)).unwrap();
    close(step.w_tau(), right, 1e-10);
}

#[test]
fn test_the_pricer_reads_it_with_a_rate_and_its_greeks_match_differences() {
    let (rate, yield_q) = (dec!(0.03), dec!(0.01));
    let surface = UpstreamSurface::new(
        calibrated_at(
            &three(),
            rate,
            yield_q,
            &ImpliedVolatilitySurfaceConfig::default(),
            as_of(),
        ),
        UpstreamPolicy::default(),
    )
    .unwrap();
    let pricer = SurfacePricer::new(Arc::new(surface), Dynamics::StickyMoneyness).unwrap();
    let inputs = |spot: Decimal, strike: Decimal, tau: Decimal| {
        OptionInputs::european(
            spot,
            strike,
            tau,
            rate,
            yield_q,
            financial_types::OptionStyle::Call,
        )
    };
    for strike in [dec!(92), dec!(100), dec!(108)] {
        let tau = dec!(0.2);
        let at = pricer.value(&inputs(dec!(100), strike, tau)).unwrap();
        let price = |spot, tau| f(pricer.value(&inputs(spot, strike, tau)).unwrap().price);
        let dt = dec!(0.0005);
        let theta = (price(dec!(100), tau - dt) - price(dec!(100), tau + dt)) / f(dt * dec!(2));
        close(f(at.theta), theta, 1e-3);
        let ds = dec!(0.01);
        let delta = (price(dec!(100) + ds, tau) - price(dec!(100) - ds, tau)) / f(ds * dec!(2));
        close(f(at.delta), delta, 1e-6);
        let gamma = (price(dec!(100) + ds, tau) - 2.0 * price(dec!(100), tau)
            + price(dec!(100) - ds, tau))
            / f(ds * ds);
        close(f(at.gamma), gamma, 1e-3);
    }
}

fn inverted() -> [Expiry; 2] {
    [expiry(30, dec!(0.04)), expiry(91, dec!(0.002))]
}

#[test]
fn test_a_calendar_violation_is_refused_unless_allowed_and_then_its_slope_is_negative() {
    let surface = calibrated(&inverted());
    assert!(!surface.arbitrage_report().passed());
    assert!(UpstreamSurface::new(surface.clone(), UpstreamPolicy::default()).is_err());

    let allowed = UpstreamSurface::new(
        surface,
        UpstreamPolicy {
            allow_static_arbitrage: true,
            ..UpstreamPolicy::default()
        },
    )
    .unwrap();
    let (w1, _, _, t1) = slice_at(allowed.surface(), 0, 0.0);
    let (w2, _, _, t2) = slice_at(allowed.surface(), 1, 0.0);
    let read = allowed.total_variance(0.0, t2).unwrap();
    assert!(read.w_tau() < 0.0);
    close(read.w_tau(), (w2 - w1) / (t2 - t1), 1e-10);

    let pricer = SurfacePricer::new(Arc::new(allowed), Dynamics::StickyMoneyness).unwrap();
    let value = pricer
        .value(&OptionInputs::european(
            dec!(100),
            dec!(100),
            Decimal::from_f64(t2).unwrap(),
            dec!(0),
            dec!(0),
            financial_types::OptionStyle::Call,
        ))
        .unwrap();
    assert!(value.theta > dec!(0), "a negative slope shows in theta");
}

/// Four expiries, the third with too few quotes to fit.
fn with_a_hole() -> [Expiry; 4] {
    let mut expiries = [
        expiry(30, dec!(0.01)),
        expiry(60, dec!(0.015)),
        expiry(91, dec!(0.02)),
        expiry(182, dec!(0.04)),
    ];
    expiries[2].strikes = &[100];
    expiries
}

#[test]
fn test_what_spans_an_expiry_with_no_slice_is_refused_unless_allowed() {
    let surface = calibrated(&with_a_hole());
    assert_eq!(surface.expiries().len(), 3);
    assert_eq!(surface.unfitted().len(), 1);
    let (_, _, _, t1) = slice_at(&surface, 0, 0.0);
    let (_, _, _, t2) = slice_at(&surface, 1, 0.0);
    let (_, _, _, t3) = slice_at(&surface, 2, 0.0);
    let strict = UpstreamSurface::new(surface.clone(), UpstreamPolicy::default()).unwrap();

    // Between the first two nothing is missing.
    assert!(strict.total_variance(0.0, t1).is_ok());
    assert!(strict.total_variance(0.0, (t1 + t2) / 2.0).is_ok());
    assert!(strict.total_variance(0.0, t2).is_ok());
    // The value between the two around the hole.
    let bridged = strict.total_variance(0.0, (t2 + t3) / 2.0).unwrap_err();
    assert!(bridged.to_string().contains("value spans"), "{bridged}");
    // At the expiry after the hole the value is its own slice's, and the slope still spans it.
    let point = surface
        .point_with_derivatives(dec!(0), surface.expiries()[2].time_to_expiry())
        .unwrap();
    assert!(!point.point.bridges_unfitted_expiry);
    let spanned = strict.total_variance(0.0, t3).unwrap_err();
    assert!(spanned.to_string().contains("slope spans"), "{spanned}");

    let open = UpstreamSurface::new(
        surface,
        UpstreamPolicy {
            allow_unfitted_spans: true,
            ..UpstreamPolicy::default()
        },
    )
    .unwrap();
    assert!(open.total_variance(0.0, (t2 + t3) / 2.0).is_ok());
    assert!(open.total_variance(0.0, t3).is_ok());
}

#[test]
fn test_an_unfitted_expiry_before_the_first_or_after_the_last_spans_nothing() {
    let mut expiries = [
        expiry(10, dec!(0.01)),
        expiry(30, dec!(0.015)),
        expiry(91, dec!(0.02)),
        expiry(182, dec!(0.04)),
    ];
    expiries[0].strikes = &[100];
    expiries[3].strikes = &[100];
    let surface = UpstreamSurface::new(calibrated(&expiries), UpstreamPolicy::default()).unwrap();
    let (_, _, _, t1) = slice_at(surface.surface(), 0, 0.0);
    let (_, _, _, t2) = slice_at(surface.surface(), 1, 0.0);
    assert!(surface.total_variance(0.0, t1).is_ok());
    assert!(surface.total_variance(0.0, t2).is_ok());
}

fn too_close() -> [Expiry; 2] {
    let mut expiries = [expiry(91, dec!(0.02)), expiry(91, dec!(0.08))];
    expiries[1].extra_nanos = 5;
    expiries
}

#[test]
fn test_expiries_too_close_to_measure_are_refused_when_built() {
    let surface = calibrated(&too_close());
    assert_eq!(surface.expiries().len(), 2);
    let refused = UpstreamSurface::new(surface.clone(), UpstreamPolicy::default()).unwrap_err();
    assert!(refused.to_string().contains("closer than"), "{refused}");

    // Told to accept them, the slope is the huge one upstream warns of.
    let accepted = UpstreamSurface::new(
        surface,
        UpstreamPolicy {
            min_expiry_gap_years: 0.0,
            ..UpstreamPolicy::default()
        },
    )
    .unwrap();
    let (w1, _, _, t1) = slice_at(accepted.surface(), 0, 0.0);
    let (w2, _, _, t2) = slice_at(accepted.surface(), 1, 0.0);
    // Each read is its own expiry's, even this close.
    close(accepted.total_variance(0.0, t1).unwrap().w(), w1, 1e-12);
    let read = accepted.total_variance(0.0, t2).unwrap();
    close(read.w(), w2, 1e-12);
    assert!(read.w_tau().abs() > 1e6);
}

#[test]
fn test_a_gap_exactly_at_the_minimum_stands() {
    let surface = calibrated(&three());
    let (_, _, _, t1) = slice_at(&surface, 0, 0.0);
    let (_, _, _, t2) = slice_at(&surface, 1, 0.0);
    let policy = |gap| UpstreamPolicy {
        min_expiry_gap_years: gap,
        ..UpstreamPolicy::default()
    };
    assert!(UpstreamSurface::new(surface.clone(), policy(t2 - t1)).is_ok());
    assert!(UpstreamSurface::new(surface, policy((t2 - t1) * 1.001)).is_err());
}

#[test]
fn test_a_policy_gap_that_is_not_a_gap_is_refused() {
    for gap in [-1.0, f64::NAN, f64::INFINITY] {
        let policy = UpstreamPolicy {
            min_expiry_gap_years: gap,
            ..UpstreamPolicy::default()
        };
        assert!(UpstreamSurface::new(calibrated(&three()), policy).is_err());
    }
}

#[test]
fn test_a_read_beyond_the_declared_extrapolation_is_refused_and_within_it_served() {
    let (_, _, _, t3) = slice_at(&calibrated(&three()), 2, 0.0);
    let strict = adapter(&three());
    assert!(strict.total_variance(2.0, t3).is_err());
    assert!(strict.total_variance(0.0, t3 * 2.0).is_err());
    assert!(strict.total_variance(0.0, 0.001).is_err());
    assert!(strict.total_variance(f64::NAN, t3).is_err());
    assert!(strict.total_variance(0.0, -1.0).is_err());

    let config = ImpliedVolatilitySurfaceConfig {
        extrapolation: ExtrapolationPolicy {
            max_wing_extension: positive(dec!(3)),
            max_short_extension: positive(dec!(1)),
            max_long_extension: positive(dec!(1)),
        },
        ..ImpliedVolatilitySurfaceConfig::default()
    };
    let open = UpstreamSurface::new(
        calibrated_at(&three(), dec!(0), dec!(0), &config, as_of()),
        UpstreamPolicy::default(),
    )
    .unwrap();
    assert!(open.total_variance(2.0, t3).is_ok());
    assert!(open.total_variance(0.0, t3 * 2.0).is_ok());
    assert!(open.total_variance(0.0, 0.001).is_ok());
}

#[test]
fn test_an_aged_read_keeps_the_expiry_and_its_implied_volatility() {
    let surface = adapter(&three());
    let age = 10.0 / 365.0;
    let aged = surface.aged(age).unwrap();
    assert_eq!(aged.age_years(), age);
    let (w, w_k, w_kk, t2) = slice_at(surface.surface(), 1, 0.02);
    let tau = t2 - age;
    let read = aged.total_variance(0.02, tau).unwrap();
    close(read.w(), w * tau / t2, 1e-10);
    close(read.w_k(), w_k * tau / t2, 1e-10);
    close(read.w_kk(), w_kk * tau / t2, 1e-10);
    // The volatility of that expiry at that k has not moved.
    close((read.w() / tau).sqrt(), (w / t2).sqrt(), 1e-10);
}

#[test]
fn test_an_aged_time_slope_matches_differences() {
    let aged = adapter(&three()).aged(5.0 / 365.0).unwrap();
    // Inside the second segment, where nothing changes side.
    for (k, tau) in [(0.0, 0.3), (0.05, 0.35), (-0.05, 0.28)] {
        let h = 1e-5;
        let read = aged.total_variance(k, tau).unwrap();
        let up = aged.total_variance(k, tau + h).unwrap().w();
        let down = aged.total_variance(k, tau - h).unwrap().w();
        close(read.w_tau(), (up - down) / (2.0 * h), 1e-7);
    }
}

#[test]
fn test_a_read_of_a_new_age_does_not_touch_the_one_it_came_from() {
    let surface = adapter(&three());
    let before = surface.total_variance(0.0, 0.2).unwrap();
    let _aged = surface.aged(0.01).unwrap();
    assert_eq!(surface.age_years(), 0.0);
    assert_eq!(surface.total_variance(0.0, 0.2).unwrap(), before);
    for age in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(surface.aged(age).is_err());
    }
}

const MINUTE_MS: u64 = 60_000;

/// A surface as of `minutes` after the start.
fn surface_after(minutes: i64, a: Decimal) -> UpstreamSurface {
    let at = as_of() + TimeDelta::try_minutes(minutes).unwrap();
    let mut expiries = three();
    expiries[1].parameters = svi(a);
    UpstreamSurface::new(
        calibrated_at(
            &expiries,
            dec!(0),
            dec!(0),
            &ImpliedVolatilitySurfaceConfig::default(),
            at,
        ),
        UpstreamPolicy::default(),
    )
    .unwrap()
}

fn start_plus(minutes: u64) -> u64 {
    as_of_ms() + minutes * MINUTE_MS
}

const DAY_MINUTES: u64 = 24 * 60;

#[test]
fn test_a_surface_serves_only_from_when_it_became_available() {
    let mut store = UpstreamStore::new();
    let first = surface_after(0, dec!(0.02));
    store
        .insert(UpstreamAttempt::calibrated(first, start_plus(2)).unwrap())
        .unwrap();
    assert!(
        store
            .surface_at(start_plus(1), DAY_MINUTES * MINUTE_MS)
            .is_err()
    );
    assert!(
        store
            .surface_at(start_plus(2) - 1, DAY_MINUTES * MINUTE_MS)
            .is_err()
    );
    let served = store
        .surface_at(start_plus(2), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(served.as_of_ms, start_plus(0));
    assert_eq!(served.available_at_ms, start_plus(2));
    assert_eq!(served.age_ms, 2 * MINUTE_MS);
    assert_eq!(served.valuation_at_ms, start_plus(2));
    assert!(!served.fallback);
    close(
        served.surface.age_years(),
        2.0 / (365.0 * 24.0 * 60.0),
        1e-12,
    );
}

#[test]
fn test_a_later_calibration_changes_nothing_before_it_was_available() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(1)).unwrap())
        .unwrap();
    let t2 = 91.0 / 365.0;
    let before = store
        .surface_at(start_plus(5), DAY_MINUTES * MINUTE_MS)
        .unwrap()
        .surface
        .total_variance(0.0, t2 - 5.0 / (365.0 * 24.0 * 60.0))
        .unwrap();

    store
        .insert(UpstreamAttempt::calibrated(surface_after(10, dec!(0.03)), start_plus(11)).unwrap())
        .unwrap();
    let served = store
        .surface_at(start_plus(5), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(served.as_of_ms, start_plus(0));
    let again = served
        .surface
        .total_variance(0.0, t2 - 5.0 / (365.0 * 24.0 * 60.0))
        .unwrap();
    assert_eq!(again, before);

    // From when it is available, the new one is what is read.
    let after = store
        .surface_at(start_plus(11), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(after.as_of_ms, start_plus(10));
}

#[test]
fn test_the_newest_data_is_served_even_when_older_data_arrived_later() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(10, dec!(0.02)), start_plus(11)).unwrap())
        .unwrap();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(12)).unwrap())
        .unwrap();
    let served = store
        .surface_at(start_plus(13), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(served.as_of_ms, start_plus(10));
    assert!(!served.fallback);
}

#[test]
fn test_between_two_on_the_same_data_the_later_available_is_served() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(1)).unwrap())
        .unwrap();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.03)), start_plus(2)).unwrap())
        .unwrap();
    let served = store
        .surface_at(start_plus(3), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(served.available_at_ms, start_plus(2));
}

#[test]
fn test_a_failure_is_kept_passed_over_and_said() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(1)).unwrap())
        .unwrap();
    store
        .insert(
            UpstreamAttempt::failed(start_plus(10), start_plus(11), "no fit".to_string()).unwrap(),
        )
        .unwrap();
    assert_eq!(store.attempts()[1].failure(), Some("no fit"));
    assert_eq!(store.attempts()[0].failure(), None);
    assert_eq!(store.attempts()[1].as_of_ms(), start_plus(10));
    assert_eq!(store.attempts()[1].available_at_ms(), start_plus(11));
    let served = store
        .surface_at(start_plus(12), DAY_MINUTES * MINUTE_MS)
        .unwrap();
    assert_eq!(served.as_of_ms, start_plus(0));
    assert!(served.fallback);

    let mut only = UpstreamStore::new();
    only.insert(
        UpstreamAttempt::failed(start_plus(10), start_plus(11), "no fit".to_string()).unwrap(),
    )
    .unwrap();
    let error = only
        .surface_at(start_plus(12), DAY_MINUTES * MINUTE_MS)
        .unwrap_err();
    assert!(error.to_string().contains("no fit"), "{error}");
}

#[test]
fn test_age_runs_from_the_data_and_a_limit_applies_to_it() {
    let mut store = UpstreamStore::new();
    // Available long after its data: being late does not make it fresh.
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(30)).unwrap())
        .unwrap();
    assert!(store.surface_at(start_plus(30), 30 * MINUTE_MS).is_ok());
    let old = store
        .surface_at(start_plus(30), 30 * MINUTE_MS - 1)
        .unwrap_err();
    assert!(old.to_string().contains("ms old"), "{old}");
}

#[test]
fn test_an_old_attempt_past_a_newer_too_old_one_is_not_tried() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(1)).unwrap())
        .unwrap();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(5, dec!(0.02)), start_plus(6)).unwrap())
        .unwrap();
    let error = store
        .surface_at(start_plus(20), 10 * MINUTE_MS)
        .unwrap_err()
        .to_string();
    // Only the newest is said to be too old: every one after it is at least as old.
    assert_eq!(error.matches("ms old").count(), 1, "{error}");
    assert!(store.surface_at(start_plus(20), 15 * MINUTE_MS).is_ok());
}

#[test]
fn test_the_store_refuses_what_would_rewrite_the_past() {
    let mut store = UpstreamStore::new();
    store
        .insert(UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), start_plus(5)).unwrap())
        .unwrap();
    for available in [start_plus(4), start_plus(5)] {
        let attempt = UpstreamAttempt::calibrated(surface_after(0, dec!(0.02)), available).unwrap();
        assert!(store.insert(attempt).is_err());
    }
    assert_eq!(store.attempts().len(), 1);
    // Available before its own data.
    assert!(UpstreamAttempt::calibrated(surface_after(10, dec!(0.02)), start_plus(9)).is_err());
    assert!(UpstreamAttempt::failed(10, 9, String::new()).is_err());
    assert!(UpstreamAttempt::failed(10, 10, String::new()).is_ok());
}

#[test]
fn test_a_new_store_says_nothing_is_available() {
    let error = UpstreamStore::new()
        .surface_at(start_plus(0), MINUTE_MS)
        .unwrap_err();
    assert!(error.to_string().contains("none is available"), "{error}");
}

#[test]
fn test_the_report_shows_residuals_by_strike_and_expiry_and_counts_what_was_left_out() {
    // Two expiries with too few quotes to fit, the second past the last fitted one.
    let mut expiries = with_a_hole().to_vec();
    let mut late = expiry(273, dec!(0.06));
    late.strikes = &[100];
    expiries.push(late);
    let mut all = quotes(&expiries, dec!(0), dec!(0));
    // Two crossed quotes and a stale one, on an expiry that still fits.
    for index in [0, 2] {
        let mut crossed = all[index];
        crossed.bid = crossed.ask + positive(dec!(1));
        all.push(crossed);
    }
    let mut stale = all[1];
    stale.quoted_at = as_of() - TimeDelta::try_hours(1).unwrap();
    all.push(stale);
    let surface = ImpliedVolatilitySurface::calibrate(
        &all,
        spot(),
        dec!(0),
        positive(dec!(0)),
        as_of(),
        &ImpliedVolatilitySurfaceConfig::default(),
    )
    .unwrap();
    let quality = SurfaceQuality::of(&surface);
    assert_eq!(quality.as_of, as_of());
    assert_eq!(quality.expiries.len(), 3);
    let per_expiry: usize = quality.expiries.iter().map(|expiry| expiry.quotes).sum();
    assert_eq!(per_expiry, STRIKES.len() * 3);
    assert_eq!(quality.residuals.len(), per_expiry);
    let first = &quality.expiries[0];
    assert_eq!(first.expiry, surface.expiries()[0].expiry());
    assert_eq!(
        first.time_to_expiry,
        surface.expiries()[0].time_to_expiry().to_dec()
    );
    assert_eq!(first.rmse, surface.expiries()[0].rmse().to_dec());
    assert_eq!(
        first.unweighted_rmse,
        surface.expiries()[0].unweighted_rmse().to_dec()
    );
    assert_eq!(
        first.quotes_outside_bid_ask,
        surface.expiries()[0].quotes_outside_bid_ask()
    );
    assert_eq!(
        first.baseline_improvement,
        surface.expiries()[0].baseline().improvement()
    );
    assert!(
        first.baseline_improvement > dec!(0),
        "a smile beats a flat line"
    );

    let at_strike = quality
        .residuals
        .iter()
        .find(|row| row.expiry == first.expiry && row.strike == positive(dec!(100)))
        .unwrap();
    assert!(at_strike.residual.abs() < dec!(0.01));

    assert_eq!(quality.rejected.get("Crossed"), Some(&2));
    assert_eq!(quality.rejected.get("Stale"), Some(&1));
    assert_eq!(quality.rejected.values().sum::<usize>(), 3);
    assert_eq!(quality.unfitted.len(), 1);
    let (reason, count) = quality.unfitted.iter().next().unwrap();
    assert!(reason.starts_with("TooFew"), "{reason}");
    assert_eq!(*count, 2);
    assert_eq!(quality.arbitrage_violations, 0);
    assert_eq!(quality.arbitrage_unevaluated, 0);
    assert!(quality.calendar_checked);
}

#[test]
fn test_the_report_counts_a_violation_and_says_when_the_calendar_could_not_be_checked() {
    let quality = SurfaceQuality::of(&calibrated(&inverted()));
    assert!(quality.arbitrage_violations > 0);
    let single = SurfaceQuality::of(&calibrated(&[expiry(91, dec!(0.02))]));
    assert!(!single.calendar_checked);
}

#[test]
fn test_data_finer_than_a_millisecond_is_refused_and_whole_milliseconds_are_not() {
    let fine = as_of() + TimeDelta::microseconds(500);
    let surface = UpstreamSurface::new(
        calibrated_at(
            &three(),
            dec!(0),
            dec!(0),
            &ImpliedVolatilitySurfaceConfig::default(),
            fine,
        ),
        UpstreamPolicy::default(),
    )
    .unwrap();
    // Truncated, it would be available in the millisecond before its data.
    let refused = UpstreamAttempt::calibrated(surface, as_of_ms()).unwrap_err();
    assert!(
        refused.to_string().contains("finer than a millisecond"),
        "{refused}"
    );

    let whole = UpstreamSurface::new(
        calibrated_at(
            &three(),
            dec!(0),
            dec!(0),
            &ImpliedVolatilitySurfaceConfig::default(),
            as_of() + TimeDelta::milliseconds(1),
        ),
        UpstreamPolicy::default(),
    )
    .unwrap();
    let attempt = UpstreamAttempt::calibrated(whole, as_of_ms() + 1).unwrap();
    assert_eq!(attempt.as_of_ms(), as_of_ms() + 1);
}
