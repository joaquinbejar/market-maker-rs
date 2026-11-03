//! Volatility surface validation on a deterministic synthetic dataset.
//!
//! Writes, into the directory given as the first argument (default
//! `target/vol_surface_validation`):
//!
//! - `data.csv`: every quote of every snapshot;
//! - `config.txt`: the calibration, holdout and replay settings;
//! - `report.json`: the first calibration's status and residual summary,
//!   the holdout comparison of SVI against the baseline by bucket, and the
//!   replay of each frozen model on the next snapshot.
//!
//! Record the run with the commit, the toolchain, `shasum -a 256` of
//! `data.csv` and `config.txt`, and `git hash-object src/options/surface/*.rs`.
//!
//! The quotes come from a known SVI surface with a deterministic
//! perturbation of the mids (a fixed linear congruential sequence, half a
//! spread at most) and a skew that moves a little each day, so that neither
//! SVI nor the baseline fits them exactly.
//!
//! Run with `cargo run --example vol_surface_validation --features options`.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use market_maker_rs::Decimal;
use market_maker_rs::options::black_undiscounted;
use market_maker_rs::options::surface::{
    BucketComparison, CleaningConfig, DiagnosticsConfig, FitConfig, HoldoutRule, IvConfig, KRange,
    OptionQuote, Rates, ShortExpiryPolicy, Snapshot, SurfaceCalibrationConfig, SviSlice, UsePolicy,
    WingRule, YEAR_MS, calibrate_surface, clean_quotes, extract, replay, validate_holdout,
};
use optionstratlib::OptionStyle;
use rust_decimal::prelude::FromPrimitive;

const DAY: u64 = 86_400_000;
const START: u64 = 20_000 * DAY;
const EXPIRY_DAYS: [u64; 4] = [14, 45, 90, 180];
const SNAPSHOTS: u64 = 5;

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

fn truth_w(day: u64, tau: f64, k: f64) -> Result<f64, Box<dyn std::error::Error>> {
    let b = 0.12 * tau.sqrt();
    let rho = -0.4 + 0.01 * day as f64;
    let slice = SviSlice::new(
        tau,
        0.25 * tau - b * 0.25,
        b,
        rho,
        0.0,
        0.25,
        KRange::new(-1.0, 1.0)?,
    )?;
    Ok(slice.eval(k)?.0)
}

fn strikes(tau: f64) -> Vec<f64> {
    // 21 strikes over about +-2.5 standard deviations of the expiry.
    let sd = 0.5 * tau.sqrt();
    (0..21)
        .map(|i| (100.0 * (sd * (f64::from(i) - 10.0) / 4.0).exp()).round())
        .collect()
}

fn snapshot(day: u64, rng: &mut Lcg) -> Result<Snapshot, Box<dyn std::error::Error>> {
    let cutoff = START + day * DAY;
    let mut quotes = Vec::new();
    for &d in &EXPIRY_DAYS {
        let left = d - day;
        let tau = (left * DAY) as f64 / YEAR_MS;
        let mut ks = strikes(tau);
        ks.dedup();
        for strike in ks {
            let k = (strike / 100.0_f64).ln();
            let w = truth_w(day, tau, k)?;
            for style in [OptionStyle::Call, OptionStyle::Put] {
                let price = black_undiscounted(100.0, strike, w, style)?;
                let half = (price * 0.01).max(0.01);
                let mid = price + 0.5 * half * rng.next();
                quotes.push(OptionQuote {
                    strike: Decimal::from_f64(strike).ok_or("strike")?,
                    expiry_at: START + d * DAY,
                    style,
                    bid: Decimal::from_f64((mid - half).max(0.0))
                        .ok_or("bid")?
                        .round_dp(4),
                    ask: Decimal::from_f64(mid + half).ok_or("ask")?.round_dp(4),
                    observed_at: cutoff - 1_000,
                    received_at: cutoff - 500,
                    underlying: Decimal::ONE_HUNDRED,
                    underlying_observed_at: cutoff - 1_000,
                    underlying_received_at: cutoff - 500,
                });
            }
        }
    }
    Ok(Snapshot {
        cutoff,
        available_at: cutoff + 60_000,
        quotes,
    })
}

fn bucket(out: &mut String, name: &str, b: &BucketComparison) -> std::fmt::Result {
    write!(
        out,
        "\"{name}\":{{\"points\":{},\"svi\":{{\"missing\":{},\"rmse_vol\":{:.6e},\"max_abs_vol\":{:.6e}}},\"baseline\":{{\"missing\":{},\"rmse_vol\":{:.6e},\"max_abs_vol\":{:.6e}}}}}",
        b.svi.points,
        b.svi.missing,
        b.svi.rmse_vol,
        b.svi.max_abs_vol,
        b.baseline.missing,
        b.baseline.rmse_vol,
        b.baseline.max_abs_vol
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).map_or_else(
        || PathBuf::from("target/vol_surface_validation"),
        PathBuf::from,
    );
    fs::create_dir_all(&dir)?;
    let mut rng = Lcg(42);
    let snaps: Vec<Snapshot> = (0..SNAPSHOTS)
        .map(|d| snapshot(d, &mut rng))
        .collect::<Result<_, _>>()?;
    let rates = Rates {
        rate: Decimal::ZERO,
        carry: Decimal::ZERO,
    };
    let config = SurfaceCalibrationConfig {
        cleaning: CleaningConfig {
            max_age_ms: 60_000,
            max_spot_lag_ms: 1_000,
            max_relative_spread: Decimal::new(19, 1),
            parity_tolerance: Decimal::new(1, 2),
        },
        iv: IvConfig::default(),
        fit: FitConfig::default(),
        diagnostics: DiagnosticsConfig::default(),
        k_grid: (0..=60).map(|i| -1.5 + 0.05 * f64::from(i)).collect(),
        short_expiry: ShortExpiryPolicy::Reject,
        stability: None,
    };
    let policy = UsePolicy {
        allow_degraded: false,
        max_age_ms: 3 * DAY,
        k_grid: config.k_grid.clone(),
        diagnostics: config.diagnostics,
    };
    let (holdout, wings) = (HoldoutRule::default(), WingRule::default());

    let mut data = String::from("snapshot,cutoff,strike,expiry_at,style,bid,ask,underlying\n");
    for (i, s) in snaps.iter().enumerate() {
        for q in &s.quotes {
            writeln!(
                data,
                "{i},{},{},{},{:?},{},{},{}",
                s.cutoff, q.strike, q.expiry_at, q.style, q.bid, q.ask, q.underlying
            )?;
        }
    }
    fs::write(dir.join("data.csv"), &data)?;
    let config_text = format!(
        "cleaning={:?}\niv={:?}\nfit={:?}\ndiagnostics={:?}\nk_grid=[-1.5:0.05:1.5]\nshort_expiry={:?}\nuse_policy=allow_degraded:{},max_age_ms:{}\nholdout={holdout:?}\nwings={wings:?}\nrates=r:{},q:{}\nseed=42\n",
        config.cleaning,
        config.iv,
        config.fit,
        config.diagnostics,
        config.short_expiry,
        policy.allow_degraded,
        policy.max_age_ms,
        rates.rate,
        rates.carry
    );
    fs::write(dir.join("config.txt"), &config_text)?;

    let first = snaps.first().ok_or("no snapshot")?;
    let (entry, quality) = calibrate_surface(
        &first.quotes,
        first.cutoff,
        first.available_at,
        rates,
        None,
        &config,
    )?;
    let cleaning = clean_quotes(&first.quotes, first.cutoff, rates, &config.cleaning)?;
    let observed = extract(&cleaning, &config.iv)?;
    let validation = validate_holdout(&observed.observations, &config.fit, holdout, wings)?;
    let replayed = replay(&snaps, rates, &config, &policy, &config.iv)?;

    let mut report = String::from("{");
    write!(
        report,
        "\"calibration\":{{\"status\":\"{:?}\",\"slices\":{},\"residuals\":{},",
        entry.status(),
        entry.surface().map_or(0, |s| s.slices().len()),
        quality.residuals.len()
    )?;
    let max_res = quality
        .residuals
        .iter()
        .map(|r| r.vol_residual.abs())
        .fold(0.0_f64, f64::max);
    write!(
        report,
        "\"max_abs_vol_residual\":{max_res:.6e},\"diagnosis_clean\":{}}},",
        quality.diagnostics.as_ref().is_some_and(|d| d.is_clean())
    )?;
    report.push_str("\"holdout\":{");
    bucket(&mut report, "all", &validation.all)?;
    report.push(',');
    bucket(&mut report, "wings_z2", &validation.wings)?;
    report.push(',');
    bucket(&mut report, "wings_abs_k_0.2", &validation.wings_abs)?;
    report.push(',');
    bucket(&mut report, "shortest_expiry", &validation.shortest)?;
    report.push_str(",\"expiries\":[");
    for (i, e) in validation.expiries.iter().enumerate() {
        if i > 0 {
            report.push(',');
        }
        write!(
            report,
            "{{\"expiry_at\":{},\"quotes\":{},",
            e.expiry_at, e.quotes
        )?;
        bucket(&mut report, "held", &e.comparison)?;
        report.push('}');
    }
    write!(
        report,
        "],\"shortest_expiry_at\":{},\"wings_unassigned\":{},\"not_applicable\":{:?},\"fit_failures\":{},\"baseline_failures\":{}}},",
        validation
            .shortest_expiry
            .map_or_else(|| "null".to_string(), |e| e.to_string()),
        validation.wings_unassigned,
        validation.not_applicable,
        validation.fit_failures.len(),
        validation.baseline_failures.len()
    )?;
    report.push_str("\"replay\":[");
    for (i, step) in replayed.steps.iter().enumerate() {
        if i > 0 {
            report.push(',');
        }
        match step {
            Ok(s) => {
                let rs: Vec<f64> = s.residuals.iter().filter_map(|r| r.vol_residual).collect();
                let rmse = (rs.iter().map(|r| r * r).sum::<f64>() / rs.len().max(1) as f64).sqrt();
                write!(
                    report,
                    "{{\"frozen_at\":{},\"model_cutoff\":{},\"label_cutoff\":{},\"fallback\":{},\"labels\":{},\"missing\":{},\"rmse_vol\":{rmse:.6e}}}",
                    s.frozen_at,
                    s.model_cutoff,
                    s.label_cutoff,
                    s.fallback,
                    s.residuals.len(),
                    s.missing.len()
                )?;
            }
            Err(e) => write!(report, "{{\"error\":{e:?}}}")?,
        }
    }
    report.push_str("]}\n");
    fs::write(dir.join("report.json"), &report)?;
    println!("{report}");
    Ok(())
}
