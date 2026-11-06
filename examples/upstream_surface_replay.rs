//! OptionStratLib's calibrated surface, served causally to the pricer. Synthetic data.
//!
//! Five daily snapshots of call quotes on four fixed expiries come from a known SVI surface whose
//! skew moves each day, with mids perturbed by a fixed linear congruential sequence inside half
//! the spread. Each snapshot is calibrated upstream, and the calibration becomes available two
//! minutes after its as-of time. On the fourth day a third of the quotes arrive crossed.
//!
//! Each day it values that day's quotes one minute after the snapshot, when only the day before's
//! surface is available, and five minutes after, when that day's is. It prints, per calibration,
//! the quality report, and per valuation the surface served, its age, and the error of its prices
//! on the quotes against a flat volatility per expiry taken from the same surface, in half
//! spreads. The one minute valuation is out of sample.
//!
//! Run with `cargo run --example upstream_surface_replay --features options`.

use std::sync::Arc;

use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use market_maker_rs::options::pricing::{Dynamics, OptionInputs, OptionPricer, SurfacePricer};
use market_maker_rs::options::surface::VolSurface;
use market_maker_rs::options::surface::YEAR_MS;
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

type Failure = Box<dyn std::error::Error>;

const DAY_MS: u64 = 86_400_000;
const MINUTE_MS: u64 = 60_000;
const DAYS: u64 = 5;
const EXPIRY_DAYS: [i64; 4] = [14, 45, 90, 180];
/// How long a calibration takes to become available after its as-of time.
const LATENCY_MS: u64 = 2 * MINUTE_MS;
/// The oldest data a valuation accepts.
const MAX_AGE_MS: u64 = 2 * DAY_MS;
/// The day whose quotes arrive partly crossed.
const CROSSED_DAY: u64 = 3;
/// How far beyond its quotes, in log moneyness, a slice is read. The forward drifts after the
/// snapshot and the spot moves the next day, so the outermost strikes leave the quoted range by
/// that much; with none declared, those reads are refused.
const WING_EXTENSION: f64 = 0.02;
const RATE: Decimal = dec!(0.03);
const CARRY: Decimal = dec!(0.01);

/// A fixed linear congruential sequence in [-1, 1].
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2030, 1, 1, 12, 0, 0)
        .single()
        .expect("a valid instant")
}

fn ms(at: DateTime<Utc>) -> u64 {
    at.timestamp_millis() as u64
}

fn at_ms(value: u64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(value as i64).expect("a valid instant")
}

fn positive(value: f64) -> Result<Positive, Failure> {
    let value = Decimal::from_f64(value).ok_or("not a decimal")?;
    Ok(Positive::new_decimal(value)?)
}

fn expiries() -> Vec<DateTime<Utc>> {
    EXPIRY_DAYS
        .iter()
        .map(|days| start() + TimeDelta::days(*days))
        .collect()
}

fn years_between(from_ms: u64, to_ms: u64) -> f64 {
    (to_ms - from_ms) as f64 / YEAR_MS
}

/// One quote as generated, with its mid to score prices against.
struct Quote {
    expiry_ms: u64,
    strike: f64,
    mid: f64,
    half_spread: f64,
    crossed: bool,
}

/// One day's chain.
struct Snapshot {
    as_of_ms: u64,
    spot: f64,
    quotes: Vec<Quote>,
}

fn snapshot(day: u64, rng: &mut Lcg) -> Result<Snapshot, Failure> {
    let as_of_ms = ms(start()) + day * DAY_MS;
    let spot = 100.0 * (1.0 + 0.01 * rng.next());
    let rho = Decimal::from_f64(-0.4 + 0.03 * day as f64).ok_or("rho")?;
    let mut quotes = Vec::new();
    for expiry in expiries() {
        let expiry_ms = ms(expiry);
        let years = years_between(as_of_ms, expiry_ms);
        let forward = spot * ((f(RATE) - f(CARRY)) * years).exp();
        let a = Decimal::from_f64(0.24 * years).ok_or("a")?;
        let parameters = SviParameters::try_new(a, dec!(0.1), rho, dec!(0.0), dec!(0.2))?;
        let slice = SviSlice::try_new(parameters, positive(forward)?, positive(years)?)?;
        let sd = 0.45 * years.sqrt();
        for i in 0..9 {
            let strike = (spot * (sd * (f64::from(i) - 4.0) / 2.0).exp()).round();
            let option = Options::new(
                OptionType::European,
                Side::Long,
                String::new(),
                positive(strike)?,
                // In days from the snapshot: a date would be read from the wall clock.
                ExpirationDate::Days(positive(years * 365.0)?),
                slice.implied_volatility_at_strike(positive(strike)?)?,
                Positive::ONE,
                positive(spot)?,
                RATE,
                OptionStyle::Call,
                Positive::new_decimal(CARRY)?,
                None,
            );
            let fair = f(black_scholes(&option)?);
            let half_spread = 0.01 + 0.02 * fair;
            let mid = fair + 0.5 * half_spread * rng.next();
            let crossed = day == CROSSED_DAY && i % 3 == 1;
            quotes.push(Quote {
                expiry_ms,
                strike,
                mid,
                half_spread,
                crossed,
            });
        }
    }
    Ok(Snapshot {
        as_of_ms,
        spot,
        quotes,
    })
}

fn f(value: Decimal) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn calibrate(snapshot: &Snapshot) -> Result<ImpliedVolatilitySurface, Failure> {
    let as_of = at_ms(snapshot.as_of_ms);
    let mut quotes = Vec::new();
    for quote in &snapshot.quotes {
        let (mut bid, mut ask) = (quote.mid - quote.half_spread, quote.mid + quote.half_spread);
        if quote.crossed {
            (bid, ask) = (ask, bid);
        }
        quotes.push(OptionQuote {
            expiry: at_ms(quote.expiry_ms),
            strike: positive(quote.strike)?,
            style: OptionStyle::Call,
            bid: positive(bid.max(1e-6))?,
            ask: positive(ask.max(1e-6))?,
            quoted_at: as_of,
        });
    }
    Ok(ImpliedVolatilitySurface::calibrate(
        &quotes,
        positive(snapshot.spot)?,
        RATE,
        Positive::new_decimal(CARRY)?,
        as_of,
        &ImpliedVolatilitySurfaceConfig {
            extrapolation: ExtrapolationPolicy {
                max_wing_extension: positive(WING_EXTENSION)?,
                ..ExtrapolationPolicy::default()
            },
            ..ImpliedVolatilitySurfaceConfig::default()
        },
    )?)
}

/// A volatility flat in strike per expiry: the surface's at the money, the baseline.
struct Flat(UpstreamSurface);

impl VolSurface for Flat {
    fn total_variance_at(
        &self,
        _k: f64,
        tau: f64,
    ) -> market_maker_rs::types::error::MMResult<market_maker_rs::options::surface::TotalVariance>
    {
        let at_the_money = self.0.total_variance_at(0.0, tau)?;
        market_maker_rs::options::surface::TotalVariance::new(
            at_the_money.w(),
            0.0,
            0.0,
            at_the_money.w_tau(),
        )
    }
}

/// One valuation: what was served, and how its prices sit on the quotes.
#[derive(Debug, Clone, PartialEq)]
struct Valuation {
    day: u64,
    at_ms: u64,
    /// As of, available at, age and fallback of the surface served, or why there was none.
    served: Result<(u64, u64, u64, bool), String>,
    /// Mean absolute error in half spreads: the surface, then the flat baseline.
    error: Option<(f64, f64)>,
    /// Quotes the surface would not price.
    refused: usize,
}

fn mean_error(
    pricer: &SurfacePricer,
    snapshot: &Snapshot,
    at_ms: u64,
) -> Result<(f64, usize), Failure> {
    let (mut total, mut priced, mut refused) = (0.0, 0_usize, 0_usize);
    for quote in snapshot.quotes.iter().filter(|quote| !quote.crossed) {
        let inputs = OptionInputs::european(
            Decimal::from_f64(snapshot.spot).ok_or("spot")?,
            Decimal::from_f64(quote.strike).ok_or("strike")?,
            Decimal::from_f64(years_between(at_ms, quote.expiry_ms)).ok_or("tau")?,
            RATE,
            CARRY,
            financial_types::OptionStyle::Call,
        );
        match pricer.value(&inputs) {
            Ok(value) => {
                total += (f(value.price) - quote.mid).abs() / quote.half_spread;
                priced += 1;
            }
            Err(_) => refused += 1,
        }
    }
    Ok((total / priced.max(1) as f64, refused))
}

/// Runs the replay over the first `calibrated` days' calibrations.
fn replay(calibrated: u64) -> Result<(Vec<SurfaceQuality>, Vec<Valuation>), Failure> {
    let mut rng = Lcg(42);
    let snapshots = (0..DAYS)
        .map(|day| snapshot(day, &mut rng))
        .collect::<Result<Vec<_>, _>>()?;
    let mut store = UpstreamStore::new();
    let mut qualities = Vec::new();
    for snapshot in snapshots.iter().take(calibrated as usize) {
        let available = snapshot.as_of_ms + LATENCY_MS;
        let attempt = match calibrate(snapshot)
            .map_err(|error| error.to_string())
            .and_then(|surface| {
                qualities.push(SurfaceQuality::of(&surface));
                UpstreamSurface::new(surface, UpstreamPolicy::default())
                    .map_err(|error| error.to_string())
            }) {
            Ok(surface) => UpstreamAttempt::calibrated(surface, available)?,
            Err(reason) => UpstreamAttempt::failed(snapshot.as_of_ms, available, reason)?,
        };
        store.insert(attempt)?;
    }
    let mut valuations = Vec::new();
    for (day, snapshot) in (0..DAYS).zip(&snapshots) {
        for offset in [MINUTE_MS, 5 * MINUTE_MS] {
            let at = snapshot.as_of_ms + offset;
            let mut valuation = Valuation {
                day,
                at_ms: at,
                served: Err(String::new()),
                error: None,
                refused: 0,
            };
            match store.surface_at(at, MAX_AGE_MS) {
                Ok(served) => {
                    valuation.served = Ok((
                        served.as_of_ms,
                        served.available_at_ms,
                        served.age_ms,
                        served.fallback,
                    ));
                    let surface = Arc::new(served.surface.clone());
                    let smile = SurfacePricer::new(surface, Dynamics::StickyMoneyness)?;
                    let flat = SurfacePricer::new(
                        Arc::new(Flat(served.surface)),
                        Dynamics::StickyMoneyness,
                    )?;
                    let (smile_error, refused) = mean_error(&smile, snapshot, at)?;
                    let (flat_error, _) = mean_error(&flat, snapshot, at)?;
                    valuation.error = Some((smile_error, flat_error));
                    valuation.refused = refused;
                }
                Err(error) => valuation.served = Err(error.to_string()),
            }
            valuations.push(valuation);
        }
    }
    Ok((qualities, valuations))
}

fn main() -> Result<(), Failure> {
    let (qualities, valuations) = replay(DAYS)?;
    println!("calibrations");
    for quality in &qualities {
        println!(
            "  as of {}: {} residuals, rejected {:?}, unfitted {:?}, {} arbitrage violations",
            quality.as_of,
            quality.residuals.len(),
            quality.rejected,
            quality.unfitted,
            quality.arbitrage_violations
        );
        for expiry in &quality.expiries {
            println!(
                "    T {:.4}: {} quotes, rmse {:.5}, {} outside bid/ask, flat rmse less this {:.5}",
                expiry.time_to_expiry,
                expiry.quotes,
                expiry.rmse,
                expiry.quotes_outside_bid_ask,
                expiry.baseline_improvement
            );
        }
    }
    println!("valuations (error in half spreads: surface, flat)");
    let start_ms = ms(start());
    for valuation in &valuations {
        let minutes = (valuation.at_ms - start_ms - valuation.day * DAY_MS) / MINUTE_MS;
        match (&valuation.served, valuation.error) {
            (Ok((as_of, _, age, fallback)), Some((smile, flat))) => println!(
                "  day {} +{minutes} min: as of day {}, age {} min{}, error {smile:.3} against {flat:.3}, {} refused",
                valuation.day,
                (as_of - start_ms) / DAY_MS,
                age / MINUTE_MS,
                if *fallback { ", fallback" } else { "" },
                valuation.refused
            ),
            (served, _) => println!(
                "  day {} +{minutes} min: none: {}",
                valuation.day,
                served.as_ref().err().map_or("", String::as_str)
            ),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A calibration that is not available yet changes nothing.
    #[test]
    fn test_a_later_calibration_changes_no_earlier_valuation() {
        let (_, all) = replay(DAYS).unwrap();
        let (_, fewer) = replay(DAYS - 1).unwrap();
        let cut = ms(start()) + (DAYS - 1) * DAY_MS + LATENCY_MS;
        let before = |valuations: &[Valuation]| {
            valuations
                .iter()
                .filter(|valuation| valuation.at_ms < cut)
                .cloned()
                .collect::<Vec<_>>()
        };
        assert!(!before(&all).is_empty());
        assert_eq!(before(&all), before(&fewer));
        assert_ne!(all, fewer);
    }

    #[test]
    fn test_every_valuation_uses_only_what_was_available_and_the_first_has_none() {
        let (_, valuations) = replay(DAYS).unwrap();
        let first = valuations[0].served.clone().unwrap_err();
        assert!(first.contains("none is available yet"), "{first}");
        let mut served = 0;
        for valuation in &valuations {
            if let Ok((as_of, available, age, _)) = valuation.served {
                assert!(available <= valuation.at_ms);
                assert!(as_of <= valuation.at_ms);
                assert_eq!(age, valuation.at_ms - as_of);
                served += 1;
            }
        }
        assert!(served > valuations.len() / 2);
    }
}
