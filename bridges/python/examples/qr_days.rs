//! The synthetic days of the out-of-sample comparison (#102), from its design file.
//!
//! `cargo run --release --example qr_days -- <design.json> <out dir>`. It simulates every
//! regime and day of the design with the queue-reactive simulator, writes each day's rows
//! under `<out dir>/<split>/<regime>/day-<n>.json` as the bridge reads them, and prints the
//! digests the design's manifest records: the dataset digest the environment reports for
//! those rows, and the external digest of the parameters, the seed and the reference.
//! A design may state a `seed_offset` (#108), added to every day's seed, for days that are
//! none of #102's; without one the seeds are #102's.
//! Everything here is synthetic, from made-up parameters.

use std::fs;
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{KernelConfig, KernelDataset};
use market_maker_rs::backtest::queue_reactive::{QrParameters, UnsupportedPolicy};
use market_maker_rs::rl::{QrEpisode, qr_market};
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const DAY_MS: u64 = 86_400_000;
const SEED_STRIDE: u64 = 1_000_003;

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, Failure> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is not text").into())
}

fn number(value: &Value, key: &str) -> Result<u64, Failure> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} is not a whole number").into())
}

fn decimal(value: &Value, key: &str) -> Result<Decimal, Failure> {
    Ok(text(value, key)?.parse::<Decimal>()?)
}

fn float(value: &Value, key: &str) -> Result<f64, Failure> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{key} is not a number").into())
}

fn parameters(regime: &Value) -> Result<QrParameters, Failure> {
    let sizes = regime
        .get("event_sizes")
        .and_then(Value::as_array)
        .ok_or("event_sizes")?
        .iter()
        .map(|size| Ok(size.as_str().ok_or("event size")?.parse::<Decimal>()?))
        .collect::<Result<Vec<_>, Failure>>()?;
    let rates = regime
        .get("rates")
        .and_then(Value::as_array)
        .ok_or("rates")?
        .iter()
        .map(|table| {
            table
                .as_array()
                .ok_or("rate table")?
                .iter()
                .map(|row| {
                    let row = row.as_array().ok_or("rate row")?;
                    let at = |i: usize| row.get(i).and_then(Value::as_f64).ok_or("rate");
                    Ok((at(0)?, at(1)?, at(2)?))
                })
                .collect::<Result<Vec<_>, Failure>>()
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    let supported = rates.iter().map(|table| vec![true; table.len()]).collect();
    let policy = match text(regime, "policy")? {
        "stop" => UnsupportedPolicy::Stop,
        "nearest_supported" => UnsupportedPolicy::NearestSupported,
        other => return Err(format!("unknown policy {other}").into()),
    };
    Ok(QrParameters::new(
        decimal(regime, "tick_size")?,
        float(regime, "theta")?,
        float(regime, "theta_reinit")?,
        sizes,
        rates,
        supported,
        policy,
    )?)
}

fn split_of(splits: &Value, day: u64) -> Result<&'static str, Failure> {
    for name in ["train", "validation", "holdout"] {
        let range = splits.get(name).and_then(Value::as_array).ok_or(name)?;
        let bound = |i: usize| range.get(i).and_then(Value::as_u64).ok_or("split bound");
        if (bound(0)?..=bound(1)?).contains(&day) {
            return Ok(name);
        }
    }
    Err(format!("day {day} is in no split").into())
}

fn write_rows(path: &Path, dataset: &KernelDataset) -> Result<(), Failure> {
    let rows: Vec<[u64; 5]> = dataset
        .ticks()
        .iter()
        .map(|t| [t.ts_ns, t.bid, t.bid_lots, t.ask, t.ask_lots])
        .collect();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec(&rows)?)?;
    Ok(())
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let design_path = PathBuf::from(args.next().ok_or("the design file")?);
    let out = PathBuf::from(args.next().ok_or("the output directory")?);
    let design: Value = serde_json::from_slice(&fs::read(&design_path)?)?;
    let simulator = design.get("simulator").ok_or("simulator")?;
    let kernel = design.get("kernel").ok_or("kernel")?;
    let config = KernelConfig::new(
        decimal(kernel, "tick_size")?,
        decimal(kernel, "lot_size")?,
        decimal(kernel, "max_price")?,
        decimal(kernel, "max_order_size")?,
        number(simulator, "rows_per_day")?,
    );
    let splits = design.get("splits").ok_or("splits")?;
    let days = number(&design, "days")?;
    // A study that needs days other than #102's states an offset; without one, #102's seeds.
    let seed_offset = design
        .get("seed_offset")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let sample_ms = u32::try_from(number(simulator, "sample_ms")?)?;
    let rows = u32::try_from(number(simulator, "rows_per_day")?)?;
    let reference_ticks = i64::try_from(number(simulator, "reference_ticks")?)?;
    let mut records = Vec::new();
    let regimes = simulator
        .get("regimes")
        .and_then(Value::as_array)
        .ok_or("regimes")?;
    for (index, regime) in regimes.iter().enumerate() {
        let name = text(regime, "name")?;
        let params = parameters(regime)?;
        let number_of_regime = u64::try_from(index)?.checked_add(1).ok_or("regime")?;
        for day in 1..=days {
            let seed = SEED_STRIDE
                .checked_mul(number_of_regime)
                .and_then(|base| base.checked_add(day))
                .and_then(|seed| seed.checked_add(seed_offset))
                .ok_or("seed")?;
            let episode = QrEpisode {
                seed,
                reference_ticks,
                start_ms: day
                    .checked_sub(1)
                    .ok_or("day")?
                    .checked_mul(DAY_MS)
                    .ok_or("day")?,
                sample_ms,
                rows,
            };
            let market = qr_market(&params, &episode, &config)?;
            if let Some(reason) = market.stopped {
                return Err(format!("{name} day {day} stopped early: {reason:?}").into());
            }
            // The digest the bridge reports for these rows, which carry no external digest.
            let seen = KernelDataset::from_kernel_ticks(market.dataset.ticks().to_vec(), &config)?;
            let split = split_of(splits, day)?;
            write_rows(
                &out.join(split).join(name).join(format!("day-{day}.json")),
                &seen,
            )?;
            records.push(json!({
                "regime": name,
                "day": day,
                "split": split,
                "seed": seed,
                "rows": seen.id().len,
                "dataset_digest": format!("{:032x}", seen.id().digest),
                "external_digest": market.dataset.id().external.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "parameters_fingerprint": format!("{:032x}", params.fingerprint()),
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({ "design_version": design["design_version"], "days": records })
        )?
    );
    Ok(())
}
