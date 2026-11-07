//! Intervals for the out-of-sample comparison (#102), with the evaluation's block bootstrap.
//!
//! `cargo run --release --example m4_05_intervals -- <series.json>` reads
//! `{"bootstrap": {...}, "series": {"<name>": ["<daily value>", ...]}}`, with the bootstrap as
//! the design states it, and prints, for each name, the interval on the mean of its daily
//! values: `{"mean", "lower", "upper", "days"}`, or the days it has and needs. A paired
//! comparison is a series of daily differences, which the caller builds.

use std::fs;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{BootstrapConfig, BootstrapOutcome, block_bootstrap};
use serde_json::{Map, Value, json};

type Failure = Box<dyn std::error::Error>;

fn whole(value: &Value, key: &str) -> Result<u64, Failure> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("bootstrap.{key} is not a whole number").into())
}

fn main() -> Result<(), Failure> {
    let path = std::env::args().nth(1).ok_or("the series file")?;
    let input: Value = serde_json::from_slice(&fs::read(path)?)?;
    let b = input.get("bootstrap").ok_or("bootstrap")?;
    let config = BootstrapConfig {
        replicates: u32::try_from(whole(b, "replicates")?)?,
        mean_block_days: u32::try_from(whole(b, "mean_block_days")?)?,
        confidence_percent: u32::try_from(whole(b, "confidence_percent")?)?,
        seed: whole(b, "seed")?,
        min_days: u32::try_from(whole(b, "min_days")?)?,
    };
    let mut out = Map::new();
    let series = input
        .get("series")
        .and_then(Value::as_object)
        .ok_or("series")?;
    for (name, values) in series {
        let values = values
            .as_array()
            .ok_or("a series is a list")?
            .iter()
            .map(|v| {
                Ok(v.as_str()
                    .ok_or("a daily value is text")?
                    .parse::<Decimal>()?)
            })
            .collect::<Result<Vec<_>, Failure>>()?;
        let result = match block_bootstrap(&values, &config)? {
            BootstrapOutcome::Interval {
                mean,
                lower,
                upper,
                days,
            } => json!({"mean": mean, "lower": lower, "upper": upper, "days": days}),
            BootstrapOutcome::Insufficient { days, required } => {
                json!({"insufficient": true, "days": days, "required": required})
            }
            other => json!({"unknown": format!("{other:?}")}),
        };
        out.insert(name.clone(), result);
    }
    println!("{}", serde_json::to_string_pretty(&Value::Object(out))?);
    Ok(())
}
