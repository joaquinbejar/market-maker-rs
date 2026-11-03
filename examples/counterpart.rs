//! A batch counterpart: calibrates a surface from option quotes and values
//! European options on it, through files, so that an external harness can
//! compare its results with its own reference implementation.
//!
//! ```text
//! counterpart identity
//! counterpart calibrate --quotes quotes.csv --config config.json --out surface.json
//! counterpart price --surface surface.json --dynamics sticky-moneyness \
//!     --requests requests.csv --responses responses.csv
//! counterpart price --surface surface.json --dynamics sticky-strike --s-ref S \
//!     --requests requests.csv --responses responses.csv
//! (either form of price takes an optional --vol-shift X)
//! ```
//!
//! - `identity` prints, as JSON, the protocol, the crate version and the
//!   commit it was built from: the build sets `COUNTERPART_GIT_SHA` and
//!   `COUNTERPART_GIT_DIRTY` (`true` or `false`); both are `null` otherwise.
//!   The binary does not hash itself: the harness hashes the file it runs
//!   and compares it with the hash it expects.
//! - `calibrate` runs `calibrate_surface` on the quotes with the cleaning
//!   settings, rates, cutoff, `k` grid and short expiry policy of
//!   `config.json`, and the crate's defaults for the implied variance, the
//!   fit and the diagnosis (all written to the output). The output holds the
//!   status, the slices and their canonical `params`, and the report.
//! - `price` values every request row on the surface and writes one
//!   response row per request: `ok` with the price, the total Greeks, the
//!   total variance read and the dynamics, or `error` with a class and a
//!   reason and no number. `--vol-shift X` prices on the surface with its
//!   implied volatility moved by `X` everywhere (`ShockedSurface`, a
//!   parallel shock), for finite differences in volatility. The classes
//!   are:
//!   - `InvalidInput`: the row itself (a field that is not a number, an
//!     unknown kind or style, a non-positive spot or strike, `as_of_days`
//!     after `expiry_days`);
//!   - `OutOfDomain`: a valid row the surface refuses, for example a time
//!     to expiry outside its slices (a numerical failure inside the surface
//!     is `Numerical`, not this);
//!   - `Unsupported`: a contract the pricer does not value (American);
//!   - `Numerical`: a numerical limit (a factor that under- or overflows, a
//!     non-finite intermediate).
//!
//! Numbers in the CSV files are read from their text as `Decimal`, never
//! through `f64`, and exactly: a decimal with more than 28 fractional digits
//! or 29 significant ones, which `Decimal` would round, is refused (an
//! `InvalidInput` row, or a protocol failure in the quotes and the config).
//! They are written as `Decimal` without exponent notation; `f64` values
//! (the total variance, the slice parameters) as their shortest round-trip
//! text.
//!
//! `price` does not check the surface file's status, availability or data
//! cutoff: whoever runs it decides whether that surface may be used at the
//! requests' time.
//! Errors in a row become error rows; the exit code is non-zero (2) only on
//! a protocol failure: arguments, I/O, a header or a row with the wrong
//! number of fields, or JSON that does not parse.
//!
//! Run with `cargo run --example counterpart --features options -- <command>`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use market_maker_rs::Decimal;
use market_maker_rs::options::pricing::{
    Dynamics, OptionInputs, OptionPricer, SurfacePricer, surface_point, tau_from_days,
};
use market_maker_rs::options::surface::{
    CleaningConfig, DiagnosticsConfig, FitConfig, IvConfig, KRange, OptionQuote, QualityReport,
    Rates, ShockedSurface, ShortExpiryPolicy, SurfaceCalibrationConfig, SurfaceShock,
    SurfaceStatus, SviSlice, SviSurface, VolSurface, calibrate_surface,
};
use market_maker_rs::types::error::MMError;
use optionstratlib::{OptionStyle, OptionType};
use serde_json::{Value, json};

/// The protocol of the files and of `identity`.
const PROTOCOL: &str = "market-maker-rs-counterpart/1";
/// The format of the surface file.
const SURFACE_FORMAT: &str = "market-maker-rs-counterpart-surface/1";

const QUOTE_COLUMNS: &[&str] = &[
    "strike",
    "expiry_at",
    "style",
    "bid",
    "ask",
    "observed_at",
    "received_at",
    "underlying",
    "underlying_observed_at",
    "underlying_received_at",
];
const REQUEST_COLUMNS: &[&str] = &[
    "id",
    "strike",
    "expiry_days",
    "kind",
    "style",
    "spot",
    "as_of_days",
    "rate",
    "carry",
];
const RESPONSE_COLUMNS: &[&str] = &[
    "id",
    "status",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "total_variance",
    "dynamics",
    "error",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("counterpart: {e}");
            ExitCode::from(2)
        }
    }
}

/// A protocol failure: the run stops and nothing is written.
type Protocol<T> = Result<T, String>;

fn run(args: &[String]) -> Protocol<()> {
    let (command, rest) = args
        .split_first()
        .ok_or_else(|| "a command is required: identity, calibrate or price".to_string())?;
    match command.as_str() {
        "identity" | "--identity" => {
            options(rest, &[], &[])?;
            println!("{}", identity());
            Ok(())
        }
        "calibrate" => {
            let o = options(rest, &["quotes", "config", "out"], &[])?;
            let quotes = read(&o, "quotes")?;
            let config = read(&o, "config")?;
            let out = calibrate(&config, &quotes)?;
            write(&o, "out", &out)
        }
        "price" => {
            let o = options(
                rest,
                &["surface", "dynamics", "requests", "responses"],
                &["s-ref", "vol-shift"],
            )?;
            let base = surface_from_json(&read(&o, "surface")?)?;
            let surface = shifted(base, o.get("vol-shift").map(String::as_str))?;
            let dynamics = dynamics(get(&o, "dynamics")?, o.get("s-ref").map(String::as_str))?;
            let out = price(surface, dynamics, &read(&o, "requests")?)?;
            write(&o, "responses", &out)
        }
        other => Err(format!("unknown command {other:?}")),
    }
}

/// `--key value` pairs: every `required` key, any of the `optional` ones,
/// and no other.
fn options(
    args: &[String],
    required: &[&str],
    optional: &[&str],
) -> Protocol<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let key = flag
            .strip_prefix("--")
            .filter(|k| required.contains(k) || optional.contains(k))
            .ok_or_else(|| format!("unexpected argument {flag:?}"))?;
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        if out.insert(key.to_string(), value.clone()).is_some() {
            return Err(format!("{flag} given twice"));
        }
    }
    for key in required {
        if !out.contains_key(*key) {
            return Err(format!("--{key} is required"));
        }
    }
    Ok(out)
}

fn get<'a>(o: &'a BTreeMap<String, String>, key: &str) -> Protocol<&'a str> {
    o.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("--{key} is required"))
}

fn read(o: &BTreeMap<String, String>, key: &str) -> Protocol<String> {
    let path = get(o, key)?;
    fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))
}

fn write(o: &BTreeMap<String, String>, key: &str, text: &str) -> Protocol<()> {
    let path = get(o, key)?;
    fs::write(path, text).map_err(|e| format!("cannot write {path}: {e}"))
}

fn identity() -> Value {
    json!({
        "protocol": PROTOCOL,
        "crate": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": option_env!("COUNTERPART_GIT_SHA"),
        "git_dirty": option_env!("COUNTERPART_GIT_DIRTY"),
        "year_days": 365,
        "commands": ["identity", "calibrate", "price"],
    })
}

// ------------------------------------------------------------------ CSV

/// The rows of a CSV text with exactly `columns` as its header. Fields are
/// not quoted; a row with another number of fields is a protocol failure.
fn csv_rows<'a>(text: &'a str, columns: &[&str]) -> Protocol<Vec<(usize, Vec<&'a str>)>> {
    let mut lines = text.lines().enumerate().filter(|(_, l)| !l.is_empty());
    let header = lines
        .next()
        .ok_or_else(|| "empty file: a header is required".to_string())?;
    let got: Vec<&str> = header.1.split(',').collect();
    if got != columns {
        return Err(format!(
            "header {:?} is not {:?}",
            header.1,
            columns.join(",")
        ));
    }
    lines
        .map(|(n, line)| {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() == columns.len() {
                Ok((n + 1, fields))
            } else {
                Err(format!(
                    "line {}: {} fields, expected {}",
                    n + 1,
                    fields.len(),
                    columns.len()
                ))
            }
        })
        .collect()
}

/// A CSV field never holds a separator, a quote or a line break.
fn csv_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            if matches!(c, ',' | '"' | '\n' | '\r') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// A plain decimal (`-?digits[.digits]`) that `Decimal` holds exactly.
fn decimal(name: &str, s: &str) -> Result<Decimal, String> {
    let digits = s.strip_prefix('-').unwrap_or(s);
    let plain = match digits.split_once('.') {
        Some((int, frac)) => {
            !int.is_empty()
                && !frac.is_empty()
                && int.bytes().all(|b| b.is_ascii_digit())
                && frac.bytes().all(|b| b.is_ascii_digit())
        }
        None => !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
    };
    if !plain {
        return Err(format!("{name} {s:?} is not a plain decimal number"));
    }
    Decimal::from_str_exact(s).map_err(|e| format!("{name} {s:?} has no exact Decimal value: {e}"))
}

fn integer(name: &str, s: &str) -> Result<u64, String> {
    s.parse()
        .map_err(|e| format!("{name} {s:?} is not a non-negative integer: {e}"))
}

// ------------------------------------------------------------ calibrate

fn calibrate(config_text: &str, quotes_text: &str) -> Protocol<String> {
    let config: Value =
        serde_json::from_str(config_text).map_err(|e| format!("config is not JSON: {e}"))?;
    let c = Config::parse(&config)?;
    let mut quotes = Vec::new();
    for (line, f) in csv_rows(quotes_text, QUOTE_COLUMNS)? {
        quotes.push(quote(&f).map_err(|e| format!("quotes line {line}: {e}"))?);
    }
    let (calibrated, report) = calibrate_surface(
        &quotes,
        c.cutoff,
        c.available_at,
        c.rates,
        None,
        &c.calibration,
    )
    .map_err(|e| format!("calibration refused the input: {e}"))?;
    let (status, reasons) = match calibrated.status() {
        SurfaceStatus::Healthy => ("healthy", Vec::new()),
        SurfaceStatus::Degraded(r) => ("degraded", r.clone()),
        SurfaceStatus::Failed(r) => ("failed", r.clone()),
    };
    let (slices, params, short_expiry) = match calibrated.surface() {
        Some(s) => (
            s.slices().iter().map(slice_json).collect::<Vec<_>>(),
            Value::from(s.params()),
            Value::from(policy_name(s.short_expiry())),
        ),
        None => (Vec::new(), Value::Null, Value::Null),
    };
    let out = json!({
        "format": SURFACE_FORMAT,
        "identity": identity(),
        "status": status,
        "reasons": reasons,
        "data_cutoff": calibrated.data_cutoff(),
        "available_at": calibrated.available_at(),
        "expiries_at": calibrated.expiries_at(),
        "short_expiry": short_expiry,
        "slices": slices,
        "params": params,
        "config": {
            "input": config,
            "iv": format!("{:?}", c.calibration.iv),
            "fit": format!("{:?}", c.calibration.fit),
            "diagnostics": format!("{:?}", c.calibration.diagnostics),
        },
        "report": report_json(&report),
    });
    serde_json::to_string_pretty(&out).map_err(|e| format!("cannot write the surface: {e}"))
}

/// The settings `config.json` gives; the others are the crate's defaults.
struct Config {
    cutoff: u64,
    available_at: u64,
    rates: Rates,
    calibration: SurfaceCalibrationConfig,
}

impl Config {
    fn parse(v: &Value) -> Protocol<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| "config must be a JSON object".to_string())?;
        let allowed = [
            "cutoff",
            "available_at",
            "rate",
            "carry",
            "cleaning",
            "k_grid",
            "short_expiry",
        ];
        if let Some(k) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(format!("config: unknown field {k:?}"));
        }
        let cleaning = field(v, "cleaning")?;
        let known = [
            "max_age_ms",
            "max_spot_lag_ms",
            "max_relative_spread",
            "parity_tolerance",
        ];
        match cleaning.as_object() {
            Some(c) => {
                if let Some(k) = c.keys().find(|k| !known.contains(&k.as_str())) {
                    return Err(format!("config: unknown field cleaning.{k}"));
                }
            }
            None => return Err("config: cleaning must be a JSON object".to_string()),
        }
        let calibration = SurfaceCalibrationConfig {
            cleaning: CleaningConfig {
                max_age_ms: json_u64(cleaning, "max_age_ms")?,
                max_spot_lag_ms: json_u64(cleaning, "max_spot_lag_ms")?,
                max_relative_spread: json_decimal(cleaning, "max_relative_spread")?,
                parity_tolerance: json_decimal(cleaning, "parity_tolerance")?,
            },
            iv: IvConfig::default(),
            fit: FitConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            k_grid: field(v, "k_grid")?
                .as_array()
                .ok_or_else(|| "config: k_grid must be an array".to_string())?
                .iter()
                .map(|x| float_text(Some(x), "config: each k_grid point"))
                .collect::<Protocol<_>>()?,
            short_expiry: policy(
                field(v, "short_expiry")?
                    .as_str()
                    .ok_or_else(|| "config: short_expiry must be a string".to_string())?,
            )?,
            stability: None,
        };
        Ok(Self {
            cutoff: json_u64(v, "cutoff")?,
            available_at: json_u64(v, "available_at")?,
            rates: Rates {
                rate: json_decimal(v, "rate")?,
                carry: json_decimal(v, "carry")?,
            },
            calibration,
        })
    }
}

fn field<'a>(v: &'a Value, name: &str) -> Protocol<&'a Value> {
    v.get(name)
        .ok_or_else(|| format!("config: missing field {name:?}"))
}

fn json_u64(v: &Value, name: &str) -> Protocol<u64> {
    field(v, name)?
        .as_u64()
        .ok_or_else(|| format!("config: {name} must be a non-negative integer"))
}

/// A decimal given as a JSON string, so that it is never a binary float.
fn json_decimal(v: &Value, name: &str) -> Protocol<Decimal> {
    let s = field(v, name)?
        .as_str()
        .ok_or_else(|| format!("config: {name} must be a decimal string"))?;
    decimal(name, s).map_err(|e| format!("config: {e}"))
}

fn policy(s: &str) -> Protocol<ShortExpiryPolicy> {
    match s {
        "reject" => Ok(ShortExpiryPolicy::Reject),
        "constant-vol" => Ok(ShortExpiryPolicy::ConstantVol),
        other => Err(format!(
            "short_expiry {other:?} is not reject or constant-vol"
        )),
    }
}

fn policy_name(p: ShortExpiryPolicy) -> &'static str {
    match p {
        ShortExpiryPolicy::Reject => "reject",
        ShortExpiryPolicy::ConstantVol => "constant-vol",
    }
}

fn quote(f: &[&str]) -> Result<OptionQuote, String> {
    let at = |i: usize| f.get(i).copied().unwrap_or_default();
    Ok(OptionQuote {
        strike: decimal("strike", at(0))?,
        expiry_at: integer("expiry_at", at(1))?,
        style: kind(at(2))?,
        bid: decimal("bid", at(3))?,
        ask: decimal("ask", at(4))?,
        observed_at: integer("observed_at", at(5))?,
        received_at: integer("received_at", at(6))?,
        underlying: decimal("underlying", at(7))?,
        underlying_observed_at: integer("underlying_observed_at", at(8))?,
        underlying_received_at: integer("underlying_received_at", at(9))?,
    })
}

fn kind(s: &str) -> Result<OptionStyle, String> {
    match s {
        "call" => Ok(OptionStyle::Call),
        "put" => Ok(OptionStyle::Put),
        other => Err(format!("kind {other:?} is not call or put")),
    }
}

/// Slice parameters as strings in Rust's shortest round-trip form (`{:e}`),
/// so that they read back bit for bit here and in any other language (a
/// JSON number need not round-trip a binary float exactly).
fn slice_json(s: &SviSlice) -> Value {
    let x = |v: f64| Value::from(format!("{v:e}"));
    json!({
        "tau": x(s.tau()),
        "a": x(s.a()),
        "b": x(s.b()),
        "rho": x(s.rho()),
        "m": x(s.m()),
        "sigma": x(s.sigma()),
        "k_lo": x(s.k_range().lo()),
        "k_hi": x(s.k_range().hi()),
    })
}

/// An `f64` written as a string, read exactly.
fn float_text(v: Option<&Value>, name: &str) -> Protocol<f64> {
    v.and_then(Value::as_str)
        .and_then(|t| f64::from_str(t).ok())
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("{name} must be a finite number written as a string"))
}

fn report_json(r: &QualityReport) -> Value {
    let rejected: Vec<Value> = r
        .cleaning
        .verdicts
        .iter()
        .filter_map(|v| {
            v.rejection
                .as_ref()
                .map(|why| json!({"index": v.index, "rejection": format!("{why:?}")}))
        })
        .collect();
    let slices: Vec<Value> = r
        .slices
        .iter()
        .map(|s| match &s.fit {
            Ok(fit) => json!({
                "expiry_at": s.expiry_at,
                "observations": s.observations,
                "rmse_vol": fit.rmse_vol,
                "lee_margin": fit.lee_margin,
                "a_margin": fit.a_margin,
                "active": fit.active.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
                "capped": fit.capped.len(),
            }),
            Err(why) => json!({
                "expiry_at": s.expiry_at,
                "observations": s.observations,
                "failure": format!("{why:?}"),
            }),
        })
        .collect();
    let residuals: Vec<Value> = r
        .residuals
        .iter()
        .map(|q| {
            json!({
                "index": q.index,
                "expiry_at": q.expiry_at,
                "k": q.k,
                "vol_residual": q.vol_residual,
                "vol_half_band": q.vol_half_band,
                "price_residual": q.price_residual,
                "weight_capped": q.weight_capped,
            })
        })
        .collect();
    json!({
        "quotes": r.cleaning.verdicts.len(),
        "used_in_fit": r.cleaning.verdicts.iter().filter(|v| v.used_in_fit).count(),
        "rejected": rejected,
        "iv_failures": r
            .iv_failures
            .iter()
            .map(|(i, side, why)| json!({"index": i, "side": side, "failure": format!("{why:?}")}))
            .collect::<Vec<_>>(),
        "slices": slices,
        "residuals": residuals,
        "diagnostics_clean": r.diagnostics.as_ref().map(|d| d.is_clean()),
        "reasons": r.reasons,
    })
}

// --------------------------------------------------------------- price

fn surface_from_json(text: &str) -> Protocol<SviSurface> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("surface is not JSON: {e}"))?;
    if v.get("format").and_then(Value::as_str) != Some(SURFACE_FORMAT) {
        return Err(format!("surface: format is not {SURFACE_FORMAT}"));
    }
    let num =
        |s: &Value, name: &str| float_text(s.get(name), &format!("surface: slice field {name}"));
    let mut slices = Vec::new();
    for s in v
        .get("slices")
        .and_then(Value::as_array)
        .ok_or_else(|| "surface: slices must be an array".to_string())?
    {
        let range =
            KRange::new(num(s, "k_lo")?, num(s, "k_hi")?).map_err(|e| format!("surface: {e}"))?;
        slices.push(
            SviSlice::new(
                num(s, "tau")?,
                num(s, "a")?,
                num(s, "b")?,
                num(s, "rho")?,
                num(s, "m")?,
                num(s, "sigma")?,
                range,
            )
            .map_err(|e| format!("surface: {e}"))?,
        );
    }
    let short_expiry = policy(
        v.get("short_expiry")
            .and_then(Value::as_str)
            .ok_or_else(|| "surface: short_expiry must be a string".to_string())?,
    )?;
    SviSurface::new(slices, short_expiry).map_err(|e| format!("surface: {e}"))
}

/// The dynamics: `--s-ref` is required for sticky strike, where it anchors
/// the surface, and refused for sticky moneyness, which has no use for it.
fn dynamics(name: &str, s_ref: Option<&str>) -> Protocol<Dynamics> {
    match (name, s_ref) {
        ("sticky-moneyness", None) => Ok(Dynamics::StickyMoneyness),
        ("sticky-moneyness", Some(_)) => Err("--s-ref applies to sticky-strike only".to_string()),
        ("sticky-strike", Some(text)) => {
            let s_ref = decimal("s-ref", text)?;
            if s_ref <= Decimal::ZERO {
                return Err(format!("--s-ref must be positive, got {s_ref}"));
            }
            Ok(Dynamics::StickyStrike { s_ref })
        }
        ("sticky-strike", None) => Err("--s-ref is required for sticky-strike".to_string()),
        (other, _) => Err(format!(
            "--dynamics {other:?} is not sticky-moneyness or sticky-strike"
        )),
    }
}

fn dynamics_name(d: Dynamics) -> &'static str {
    match d {
        Dynamics::StickyStrike { .. } => "sticky-strike",
        _ => "sticky-moneyness",
    }
}

/// The class of an error of the shared checks or of the valuation, for the
/// harness to map onto its own; surface errors are classed before
/// ([`value_row`] reads the surface first, as `OutOfDomain`).
fn class(e: &MMError) -> &'static str {
    match e {
        MMError::UnsupportedContract { .. } => "Unsupported",
        MMError::NumericalError(_) | MMError::PriceUnderflow => "Numerical",
        _ => "InvalidInput",
    }
}

/// The surface, or the surface with its implied volatility moved by a
/// parallel shift (a finite `f64` written as text).
fn shifted(base: SviSurface, shift: Option<&str>) -> Protocol<Arc<dyn VolSurface>> {
    let Some(text) = shift else {
        return Ok(Arc::new(base));
    };
    let x = f64::from_str(text)
        .ok()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("--vol-shift {text:?} is not a finite number"))?;
    let shock = SurfaceShock::new("vol-shift", x, 0.0, 0.0, 1.0)
        .map_err(|e| format!("--vol-shift: {e}"))?;
    Ok(Arc::new(ShockedSurface::new(base, shock)))
}

fn price(surface: Arc<dyn VolSurface>, dynamics: Dynamics, requests: &str) -> Protocol<String> {
    let pricer = SurfacePricer::new(Arc::clone(&surface), dynamics)
        .map_err(|e| format!("invalid dynamics: {e}"))?;
    let mut out = RESPONSE_COLUMNS.join(",");
    out.push('\n');
    for (_, f) in csv_rows(requests, REQUEST_COLUMNS)? {
        let id = csv_text(f.first().copied().unwrap_or_default());
        let row = match value_row(&f, surface.as_ref(), &pricer, dynamics) {
            Ok(cells) => format!("{id},ok,{cells},{},", dynamics_name(dynamics)),
            Err((cls, why)) => format!(
                "{id},error,,,,,,,{},{cls}: {}",
                dynamics_name(dynamics),
                csv_text(&why)
            ),
        };
        let _ = writeln!(out, "{row}");
    }
    Ok(out)
}

/// The numeric cells `price,delta,gamma,vega,theta,total_variance`, or the
/// class and reason of the row's error.
fn value_row(
    f: &[&str],
    surface: &dyn VolSurface,
    pricer: &SurfacePricer,
    dynamics: Dynamics,
) -> Result<String, (&'static str, String)> {
    let invalid = |why: String| ("InvalidInput", why);
    let at = |i: usize| f.get(i).copied().unwrap_or_default();
    let strike = decimal("strike", at(1)).map_err(invalid)?;
    let expiry = decimal("expiry_days", at(2)).map_err(invalid)?;
    let style = kind(at(3)).map_err(invalid)?;
    let exercise = match at(4) {
        "european" => OptionType::European,
        "american" => OptionType::American,
        other => {
            return Err(invalid(format!(
                "style {other:?} is not european or american"
            )));
        }
    };
    let spot = decimal("spot", at(5)).map_err(invalid)?;
    let as_of = decimal("as_of_days", at(6)).map_err(invalid)?;
    let rate = decimal("rate", at(7)).map_err(invalid)?;
    let carry = decimal("carry", at(8)).map_err(invalid)?;
    let days = expiry
        .checked_sub(as_of)
        .ok_or_else(|| invalid("expiry_days - as_of_days overflows".to_string()))?;
    let model = |e: MMError| (class(&e), e.to_string());
    let tau = tau_from_days(days).map_err(|e| invalid(e.to_string()))?;
    if strike <= Decimal::ZERO {
        return Err(invalid(format!("strike must be positive, got {strike}")));
    }
    // The request first (its contract, spot, time to expiry, rates and the
    // factors derived from them), then the surface, then the value.
    if !matches!(exercise, OptionType::European) {
        return Err((
            "Unsupported",
            format!("only European options are valued, got {exercise:?}"),
        ));
    }
    pricer.check_market(spot, tau, rate, carry).map_err(model)?;
    let mut inputs = OptionInputs::european(spot, strike, tau, rate, carry, style);
    inputs.exercise = exercise;
    let w = total_variance(surface, dynamics, &inputs).map_err(|e| match e {
        Read::Surface(e) => (surface_class(&e), e.to_string()),
        Read::Inputs(e) => model(e),
    })?;
    let v = pricer.value(&inputs).map_err(model)?;
    Ok(format!(
        "{},{},{},{},{},{w}",
        v.price, v.delta, v.gamma, v.vega, v.theta
    ))
}

/// The class of a surface error: a numerical failure inside the surface
/// stays `Numerical`; any other refusal is the surface's domain.
fn surface_class(e: &MMError) -> &'static str {
    match e {
        MMError::NumericalError(_) | MMError::PriceUnderflow => "Numerical",
        _ => "OutOfDomain",
    }
}

/// Why the total variance could not be read.
enum Read {
    /// The inputs have no `f64` value.
    Inputs(MMError),
    /// The surface does not answer at the point.
    Surface(MMError),
}

/// The total variance the valuation reads, at the point
/// [`surface_point`] gives (the pricer's own `k` and `tau`), written as
/// `f64` text (shortest round trip, no exponent); `0` at expiry, where no
/// surface is read.
fn total_variance(
    surface: &dyn VolSurface,
    dynamics: Dynamics,
    i: &OptionInputs,
) -> Result<String, Read> {
    if i.tau_years <= Decimal::ZERO {
        return Ok("0".to_string());
    }
    let (k, tau) = surface_point(i, dynamics).map_err(Read::Inputs)?;
    let w = surface.total_variance(k, tau).map_err(Read::Surface)?.w();
    Ok(format!("{w}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rust_decimal::prelude::FromPrimitive;

    /// A skewed two-slice surface (30 and 60 days).
    fn surface_json() -> String {
        let slice = |tau: f64| {
            let x = |v: f64| format!("{v:e}");
            json!({"tau": x(tau), "a": x(0.002 * tau * 12.0), "b": "0.08", "rho": "-0.6",
                   "m": "0", "sigma": "0.1", "k_lo": "-0.5", "k_hi": "0.5"})
        };
        json!({
            "format": SURFACE_FORMAT,
            "short_expiry": "reject",
            "slices": [slice(30.0 / 365.0), slice(60.0 / 365.0)],
        })
        .to_string()
    }

    fn requests(rows: &[&str]) -> String {
        let mut s = REQUEST_COLUMNS.join(",");
        for r in rows {
            s.push('\n');
            s.push_str(r);
        }
        s
    }

    fn rows(out: &str) -> Vec<Vec<String>> {
        out.lines()
            .skip(1)
            .map(|l| l.split(',').map(str::to_string).collect())
            .collect()
    }

    fn priced(dynamics: Dynamics, req: &[&str]) -> Vec<Vec<String>> {
        let surface = surface_from_json(&surface_json()).unwrap();
        rows(&price(Arc::new(surface), dynamics, &requests(req)).unwrap())
    }

    #[test]
    fn test_an_ok_row_has_every_number_and_no_exponent() {
        let r = priced(
            Dynamics::StickyMoneyness,
            &["a,100,45,call,european,100,0,0.01,0"],
        );
        assert_eq!(r.len(), 1);
        let row = &r[0];
        assert_eq!(row.len(), RESPONSE_COLUMNS.len());
        assert_eq!((row[0].as_str(), row[1].as_str()), ("a", "ok"));
        for cell in &row[2..8] {
            assert!(Decimal::from_str(cell).is_ok(), "{cell}");
            assert!(!cell.contains(['e', 'E']), "{cell}");
        }
        assert_eq!(row[8], "sticky-moneyness");
        assert_eq!(row[9], "");
    }

    #[test]
    fn test_each_error_class_is_an_error_row_without_numbers() {
        let r = priced(
            Dynamics::StickyMoneyness,
            &[
                "spot,100,45,call,european,0,0,0,0",
                "american,100,45,call,american,100,0,0,0",
                "underflow,100,45,call,european,100,0,10000,0",
                "text,100,45,call,european,1e2x,0,0,0",
                "tau,100,90,call,european,100,0,0,0",
                "strike,0,45,call,european,100,0,0,0",
                "late,100,45,call,european,100,46,0,0",
            ],
        );
        let classes: Vec<&str> = r
            .iter()
            .map(|row| {
                assert_eq!(row[1], "error", "{row:?}");
                assert!(row[2..8].iter().all(String::is_empty), "{row:?}");
                row[9].split(':').next().unwrap()
            })
            .collect();
        assert_eq!(
            classes,
            [
                "InvalidInput",
                "Unsupported",
                "Numerical",
                "InvalidInput",
                "OutOfDomain",
                "InvalidInput",
                "InvalidInput"
            ]
        );
    }

    #[test]
    fn test_surface_errors_are_classed_by_kind() {
        // w_tau overflows on the first segment: a numerical failure, not a
        // point outside the surface. And an American option past the last
        // slice is Unsupported: the contract is checked first.
        let overflow = json!({
            "format": SURFACE_FORMAT, "short_expiry": "reject",
            "slices": [
                {"tau": "0.1", "a": "1", "b": "0", "rho": "0", "m": "0", "sigma": "0.1",
                 "k_lo": "-0.1", "k_hi": "0.1"},
                {"tau": "0.2", "a": "1e308", "b": "0", "rho": "0", "m": "0", "sigma": "0.1",
                 "k_lo": "-0.1", "k_hi": "0.1"},
            ],
        })
        .to_string();
        let surface = surface_from_json(&overflow).unwrap();
        let r = rows(
            &price(
                Arc::new(surface),
                Dynamics::StickyMoneyness,
                &requests(&["inf,100,36.5,call,european,100,0,0,0"]),
            )
            .unwrap(),
        );
        assert!(r[0][9].starts_with("Numerical:"), "{}", r[0][9]);
        assert!(r[0][2..8].iter().all(String::is_empty));
        let r = priced(
            Dynamics::StickyMoneyness,
            &["far,100,90,call,american,100,0,0,0"],
        );
        assert!(r[0][9].starts_with("Unsupported:"), "{}", r[0][9]);
    }

    #[test]
    fn test_vol_shift_moves_the_implied_volatility_in_parallel() {
        let req = requests(&["a,100,45,call,european,100,0,0,0"]);
        let base = rows(
            &price(
                shifted(surface_from_json(&surface_json()).unwrap(), None).unwrap(),
                Dynamics::StickyMoneyness,
                &req,
            )
            .unwrap(),
        );
        let up = rows(
            &price(
                shifted(surface_from_json(&surface_json()).unwrap(), Some("0.01")).unwrap(),
                Dynamics::StickyMoneyness,
                &req,
            )
            .unwrap(),
        );
        let (w0, w1): (f64, f64) = (base[0][7].parse().unwrap(), up[0][7].parse().unwrap());
        let tau = 45.0 / 365.0;
        let shifted_w = ((w0 / tau).sqrt() + 0.01).powi(2) * tau;
        assert!((w1 - shifted_w).abs() <= 1e-15, "{w1} vs {shifted_w}");
        let (p0, p1): (f64, f64) = (base[0][2].parse().unwrap(), up[0][2].parse().unwrap());
        assert!(p1 > p0);
        for bad in ["x", "inf", "NaN", ""] {
            assert!(shifted(surface_from_json(&surface_json()).unwrap(), Some(bad)).is_err());
        }
    }

    #[test]
    fn test_sticky_strike_keeps_s_ref_while_sticky_moneyness_follows_the_spot() {
        // Same spots and surface: the two dynamics read the skew at
        // different k, so the deltas differ; sticky strike gives the same
        // total variance at every spot (k fixed by s_ref).
        let req = [
            "lo,100,45,put,european,95,0,0,0",
            "hi,100,45,put,european,105,0,0,0",
        ];
        let sm = priced(Dynamics::StickyMoneyness, &req);
        let ss = priced(
            Dynamics::StickyStrike {
                s_ref: Decimal::from(100),
            },
            &req,
        );
        assert_eq!(ss[0][7], ss[1][7], "sticky strike reads one k");
        assert_ne!(sm[0][7], sm[1][7], "sticky moneyness moves with the spot");
        assert_ne!(sm[0][3], ss[0][3], "the deltas differ by the skew term");
        assert_eq!(ss[0][8], "sticky-strike");
    }

    #[test]
    fn test_decimal_inputs_are_read_from_their_text() {
        // A strike that f64 cannot hold: at expiry the payoff is exact.
        let r = priced(
            Dynamics::StickyMoneyness,
            &["x,100.000000000000000001,45,put,european,100,45,0,0"],
        );
        assert_eq!(r[0][2], "0.000000000000000001");
    }

    #[test]
    fn test_decimals_that_would_round_are_refused() {
        for text in [
            // Rounds to 0 (29 fractional digits).
            "0.00000000000000000000000000001",
            // Rounds at a nonzero magnitude (30 significant digits).
            "12345678901234567890.1234567890",
            "1.00000000000000000000000000001",
            "1e2",
            "1_000",
            ".5",
            "",
        ] {
            assert!(decimal("x", text).is_err(), "{text:?}");
        }
        // 29 significant digits that fit are kept exactly.
        assert_eq!(
            decimal("x", "1.0000000000000000000000000001")
                .unwrap()
                .to_string(),
            "1.0000000000000000000000000001"
        );
        assert_eq!(
            decimal("x", "-0.0000000000000000000000000001").unwrap(),
            Decimal::from_str("-0.0000000000000000000000000001").unwrap()
        );
        let r = priced(
            Dynamics::StickyMoneyness,
            &["tiny,100,45,call,european,0.00000000000000000000000000001,0,0,0"],
        );
        assert_eq!(r[0][1], "error");
        assert!(r[0][9].starts_with("InvalidInput:"), "{}", r[0][9]);
    }

    #[test]
    fn test_reported_total_variance_is_the_one_the_pricer_reads() {
        // Near the money with a tiny smile: k = log1p(1e-18) must come from
        // the exact Decimal ratio, as in the pricer.
        let tiny = json!({
            "format": SURFACE_FORMAT, "short_expiry": "reject",
            "slices": [
                {"tau": format!("{:e}", 30.0 / 365.0), "a": "0", "b": "0.1", "rho": "0",
                 "m": "1e-18", "sigma": "1e-18", "k_lo": "-0.1", "k_hi": "0.1"},
                {"tau": format!("{:e}", 60.0 / 365.0), "a": "0", "b": "0.1", "rho": "0",
                 "m": "1e-18", "sigma": "1e-18", "k_lo": "-0.1", "k_hi": "0.1"},
            ],
        })
        .to_string();
        for d in [
            Dynamics::StickyMoneyness,
            Dynamics::StickyStrike {
                s_ref: Decimal::from(100),
            },
        ] {
            let surface = surface_from_json(&tiny).unwrap();
            let r = rows(
                &price(
                    Arc::new(surface),
                    d,
                    &requests(&["n,100.0000000000000001,30,call,european,100,0,0,0"]),
                )
                .unwrap(),
            );
            let w: f64 = r[0][7].parse().unwrap();
            assert!((w - 1.0000000000000001e-19).abs() <= 1e-34, "{d:?}: {w:e}");
        }
    }

    #[test]
    fn test_protocol_failures_stop_the_run() {
        let surface = || surface_from_json(&surface_json()).unwrap();
        assert!(
            price(
                Arc::new(surface()),
                Dynamics::StickyMoneyness,
                "id,strike\n1,2"
            )
            .is_err()
        );
        assert!(
            price(
                Arc::new(surface()),
                Dynamics::StickyMoneyness,
                &requests(&["a,100,45,call,european,100,0,0"])
            )
            .is_err()
        );
        assert!(surface_from_json("{}").is_err());
        assert!(dynamics("sticky-strike", Some("0")).is_err());
        assert!(dynamics("sticky-strike", None).is_err());
        assert!(dynamics("sticky-moneyness", Some("100")).is_err());
        assert!(dynamics("sticky-delta", None).is_err());
        assert_eq!(
            dynamics("sticky-moneyness", None).unwrap(),
            Dynamics::StickyMoneyness
        );
        assert_eq!(
            dynamics("sticky-strike", Some("100")).unwrap(),
            Dynamics::StickyStrike {
                s_ref: Decimal::from(100)
            }
        );
        assert!(options(&["--x".to_string(), "1".to_string()], &["y"], &[]).is_err());
        assert!(options(&[], &[], &["y"]).is_ok());
    }

    /// Quotes of a known two-slice SVI surface, bid and ask a relative
    /// half spread of 1e-4 around the model price, out of the money side.
    fn synthetic_quotes() -> String {
        let day = 86_400_000_u64;
        let cutoff = 1_000_000_u64;
        let mut s = QUOTE_COLUMNS.join(",");
        let truth = surface_from_json(&surface_json()).unwrap();
        for (days, tau) in [(30_u64, 30.0 / 365.0), (60, 60.0 / 365.0)] {
            for i in -6..=6 {
                let k = f64::from(i) * 0.05;
                let strike = 100.0 * k.exp();
                let w = truth.total_variance(k, tau).unwrap().w();
                let style = if k < 0.0 {
                    OptionStyle::Put
                } else {
                    OptionStyle::Call
                };
                let mid =
                    market_maker_rs::options::pricing::black_undiscounted(100.0, strike, w, style)
                        .unwrap();
                let d = |x: f64| Decimal::from_f64(x).unwrap();
                let name = if k < 0.0 { "put" } else { "call" };
                let _ = write!(
                    s,
                    "\n{},{},{name},{},{},{cutoff},{cutoff},100,{cutoff},{cutoff}",
                    d(strike),
                    cutoff + days * day,
                    d(mid * (1.0 - 1e-4)),
                    d(mid * (1.0 + 1e-4)),
                );
            }
        }
        s
    }

    fn config() -> String {
        json!({
            "cutoff": 1_000_000, "available_at": 1_000_000, "rate": "0", "carry": "0",
            "cleaning": {"max_age_ms": 0, "max_spot_lag_ms": 0,
                         "max_relative_spread": "0.01", "parity_tolerance": "0"},
            "k_grid": ["-0.3", "-0.15", "0", "0.15", "0.3"],
            "short_expiry": "reject",
        })
        .to_string()
    }

    #[test]
    fn test_calibrate_recovers_a_known_surface_and_round_trips_its_file() {
        let out = calibrate(&config(), &synthetic_quotes()).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["format"], SURFACE_FORMAT);
        assert_ne!(v["status"], "failed", "{}", v["reasons"]);
        assert_eq!(v["slices"].as_array().unwrap().len(), 2);
        // The file prices: the calibrated surface reproduces the truth's
        // total variance near the money.
        let fitted = surface_from_json(&out).unwrap();
        let truth = surface_from_json(&surface_json()).unwrap();
        for k in [-0.2, 0.0, 0.2] {
            let tau = 45.0 / 365.0;
            let (a, b) = (
                fitted.total_variance(k, tau).unwrap().w(),
                truth.total_variance(k, tau).unwrap().w(),
            );
            assert!((a - b).abs() <= 1e-4 * b, "k {k}: {a} vs {b}");
        }
        // Bit for bit: the file's slices are the calibrated ones.
        assert_eq!(v["params"], fitted.params());
    }

    #[test]
    fn test_calibration_config_is_strict() {
        let mut c: Value = serde_json::from_str(&config()).unwrap();
        c["extra"] = json!(1);
        assert!(calibrate(&c.to_string(), &synthetic_quotes()).is_err());
        let mut c: Value = serde_json::from_str(&config()).unwrap();
        c["rate"] = json!(0.0);
        assert!(calibrate(&c.to_string(), &synthetic_quotes()).is_err());
        let mut c: Value = serde_json::from_str(&config()).unwrap();
        c["cleaning"]["max_gae_ms"] = json!(0);
        assert!(calibrate(&c.to_string(), &synthetic_quotes()).is_err());
        let mut c: Value = serde_json::from_str(&config()).unwrap();
        c["k_grid"] = json!([0.0]);
        assert!(calibrate(&c.to_string(), &synthetic_quotes()).is_err());
    }

    #[test]
    fn test_identity_names_the_protocol() {
        let id = identity();
        assert_eq!(id["protocol"], PROTOCOL);
        assert_eq!(id["year_days"], 365);
    }
}
