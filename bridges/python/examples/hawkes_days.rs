//! The synthetic days of part C of #107: Hawkes order flow that moves the mid.
//!
//! `cargo run --release --example hawkes_days -- <design.json> <out dir>`. For every impact
//! setting and day of the design's `causal` section, it simulates the buy and the sell
//! market orders with two queue-reactive Hawkes simulators (`backtest::hawkes`, #91), moves the
//! mid by `impact_ticks` per market order in its direction plus a Gaussian diffusion of
//! `diffusion_ticks` per row, and writes the rows as the bridge reads them, under
//! `<out dir>/<split>/<setting>/day-<n>.json`. It prints each day's digests for the design's
//! manifest.
//!
//! Everything here is SYNTHETIC, from made-up parameters, and says nothing about a market.
//!
//! - One simulated second is one row; rows are `sample_ms` apart on the clock.
//! - The ask queue is the buy side's simulator and the bid queue the sell side's; a market
//!   order of the buy side lifts the ask and moves the mid up. Their queues are the touch
//!   sizes (at least 1 lot) and are not tied to price levels, unlike the paper's QRH-I, which
//!   restarts memory at each price change: here memory runs across the day.
//! - The touch is `floor(mid)` for the bid and `spread_ticks` above it for the ask.
//! - A row shows what happened before its time: the events of simulated second `r` act after
//!   row `r` is written.
//! - The diffusion draws from its own generator, seeded apart from the flow's, so the flow of
//!   a day is the same in every impact setting and only the impact changes.
//! - Next to each day's rows, `day-<n>.flow.json` holds each row's buy and sell market orders.
//!   No policy reads it; only the report's probe of what the HSN states track does.

use std::fs;
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::hawkes::{HawkesParameters, HawkesSimulator};
use market_maker_rs::backtest::kernel::{KernelConfig, KernelDataset};
use market_maker_rs::backtest::queue_reactive::QrEventKind;
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const DAY_MS: u64 = 86_400_000;
const SEED_STRIDE: u64 = 1_000_003;

fn get<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Failure> {
    value
        .get(key)
        .ok_or_else(|| format!("{key} is missing").into())
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, Failure> {
    get(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text").into())
}

fn number(value: &Value, key: &str) -> Result<u64, Failure> {
    get(value, key)?
        .as_u64()
        .ok_or_else(|| format!("{key} is not a whole number").into())
}

fn float(value: &Value, key: &str) -> Result<f64, Failure> {
    get(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not a number").into())
}

fn floats(value: &Value) -> Result<Vec<f64>, Failure> {
    value
        .as_array()
        .ok_or("not a list")?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "not a number".into()))
        .collect()
}

/// The flow's parameters: rates by queue size and the kernels.
fn parameters(flow: &Value) -> Result<HawkesParameters, Failure> {
    let cap = usize::try_from(number(flow, "queue_cap")?)?;
    let rates = get(flow, "rates")?;
    let (limit, cancel, market) = (
        float(rates, "limit")?,
        float(rates, "cancel_per_lot")?,
        float(rates, "market")?,
    );
    let table = (0..=cap)
        .map(|n| {
            let size = n as f64;
            (
                if n == cap { 0.0 } else { limit },
                if n == 0 { 0.0 } else { cancel * size },
                if n == 0 { 0.0 } else { market },
            )
        })
        .collect();
    let decays = floats(get(flow, "decays")?)?;
    let weights = get(flow, "weights")?
        .as_array()
        .ok_or("weights")?
        .iter()
        .map(|matrix| {
            let rows = matrix.as_array().ok_or("weight matrix")?;
            let mut out = [[0.0; 3]; 3];
            for (l, row) in rows.iter().enumerate() {
                let row = floats(row)?;
                if row.len() != 3 || l >= 3 {
                    return Err::<_, Failure>("a weight matrix is 3 x 3".into());
                }
                out[l].copy_from_slice(&row);
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(HawkesParameters::new(table, decays, weights)?)
}

/// A small generator for the diffusion, apart from the flow's (xorshift64*).
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (1.0 - self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// One side's day: for each row, its queue before the row and its market orders after it.
fn side_day(
    params: &HawkesParameters,
    seed: u64,
    initial: u32,
    rows: usize,
) -> Result<(Vec<u32>, Vec<u32>), Failure> {
    let mut sim = HawkesSimulator::new(params.clone(), seed, initial)?;
    let (segment, stop) = sim.run(rows as f64)?;
    if let Some(stop) = stop {
        return Err(format!("the flow stopped: {stop:?}").into());
    }
    let mut queues = Vec::with_capacity(rows);
    let mut markets = vec![0_u32; rows];
    let mut queue = initial;
    let mut events = segment.events.iter().peekable();
    for (row, count) in markets.iter_mut().enumerate() {
        queues.push(queue);
        let until = (row + 1) as f64;
        while let Some(event) = events.next_if(|e| e.time_seconds < until) {
            match event.kind {
                QrEventKind::Limit => queue += 1,
                QrEventKind::Cancel => queue -= 1,
                QrEventKind::Market => {
                    queue -= 1;
                    *count += 1;
                }
                other => return Err(format!("unknown event {other:?}").into()),
            }
        }
    }
    Ok((queues, markets))
}

fn split_of(splits: &Value, day: u64) -> Result<&'static str, Failure> {
    for name in ["train", "validation", "holdout"] {
        let range = get(splits, name)?.as_array().ok_or(name)?;
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
    let causal = get(&design, "causal")?;
    let flow = get(causal, "flow")?;
    let kernel = get(causal, "kernel")?;
    let tick: Decimal = text(kernel, "tick_size")?.parse()?;
    let rows_per_day = number(flow, "rows_per_day")?;
    let config = KernelConfig::new(
        tick,
        text(kernel, "lot_size")?.parse()?,
        text(kernel, "max_price")?.parse()?,
        text(kernel, "max_order_size")?.parse()?,
        rows_per_day,
    );
    let params = parameters(flow)?;
    let rows = usize::try_from(rows_per_day)?;
    let sample_ms = number(flow, "sample_ms")?;
    let initial = u32::try_from(number(flow, "initial_queue")?)?;
    let reference = i64::try_from(number(flow, "reference_ticks")?)?;
    let spread = i64::try_from(number(flow, "spread_ticks")?)?;
    if spread < 1 {
        return Err("spread_ticks is at least 1".into());
    }
    let diffusion = float(flow, "diffusion_ticks")?;
    let splits = get(causal, "splits")?;
    let days = number(causal, "days")?;
    let mut records = Vec::new();
    let settings = get(causal, "settings")?.as_array().ok_or("settings")?;
    for day in 1..=days {
        // The flow and the diffusion of a day are the same in every setting.
        // #107's days use stride + day; a design with `seed_offset` uses stride x day + offset.
        let seed = match causal.get("seed_offset").and_then(Value::as_u64) {
            Some(offset) => SEED_STRIDE
                .checked_mul(day)
                .and_then(|s| s.checked_add(offset))
                .ok_or("seed")?,
            None => SEED_STRIDE.checked_add(day).ok_or("seed")?,
        };
        let (ask_queues, buys) = side_day(&params, seed.wrapping_mul(2), initial, rows)?;
        let (bid_queues, sells) = side_day(&params, seed.wrapping_mul(2) + 1, initial, rows)?;
        for setting in settings {
            let name = text(setting, "name")?;
            let impact = float(setting, "impact_ticks")?;
            let mut noise = Noise::new(seed);
            let mut mid = reference as f64;
            let start_ms = (day - 1).checked_mul(DAY_MS).ok_or("day")?;
            let mut ticks = Vec::with_capacity(rows);
            for row in 0..rows {
                if row > 0 {
                    let net = f64::from(buys[row - 1]) - f64::from(sells[row - 1]);
                    mid += impact * net + diffusion * noise.normal();
                }
                let bid = mid.floor() as i64;
                let ask = bid + spread;
                if bid < 1 {
                    return Err(format!("{name} day {day}: the bid left the grid").into());
                }
                let lots = |q: u32| Decimal::from(q.max(1));
                ticks.push(MarketTick::new(
                    start_ms + row as u64 * sample_ms,
                    Decimal::from(bid) * tick,
                    lots(bid_queues[row]),
                    Decimal::from(ask) * tick,
                    lots(ask_queues[row]),
                ));
            }
            let seen = KernelDataset::from_ticks(&ticks, &config)?;
            let split = split_of(splits, day)?;
            let dir = out.join(split).join(name);
            write_rows(&dir.join(format!("day-{day}.json")), &seen)?;
            let flow: Vec<[u32; 2]> = buys.iter().zip(&sells).map(|(b, s)| [*b, *s]).collect();
            fs::write(
                dir.join(format!("day-{day}.flow.json")),
                serde_json::to_vec(&flow)?,
            )?;
            // A design with `seed_offset` also keeps the true queues (rows show at least 1 lot).
            if causal.get("seed_offset").is_some() {
                let queues: Vec<[u32; 2]> = bid_queues
                    .iter()
                    .zip(&ask_queues)
                    .map(|(b, a)| [*b, *a])
                    .collect();
                fs::write(
                    dir.join(format!("day-{day}.queues.json")),
                    serde_json::to_vec(&queues)?,
                )?;
            }
            records.push(json!({
                "setting": name,
                "day": day,
                "split": split,
                "seed": seed,
                "rows": seen.id().len,
                "dataset_digest": format!("{:032x}", seen.id().digest),
                "parameters_fingerprint": format!("{:032x}", params.fingerprint()),
                "market_orders": buys.iter().chain(&sells).map(|c| u64::from(*c)).sum::<u64>(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn flow() -> Value {
        json!({
            "queue_cap": 20,
            "rates": {"limit": 1.5, "cancel_per_lot": 0.05, "market": 0.3},
            "decays": [1.0, 0.1],
            "weights": [
                [[0.10, 0.05, 0.10], [0.05, 0.10, 0.05], [0.02, 0.02, 0.30]],
                [[0.05, 0.02, 0.05], [0.02, 0.05, 0.02], [0.01, 0.01, 0.25]]
            ]
        })
    }

    #[test]
    fn test_a_side_day_replays_its_queue_and_counts_its_market_orders() {
        let params = parameters(&flow()).unwrap();
        let (queues, markets) = side_day(&params, 7, 10, 500).unwrap();
        assert_eq!(queues.len(), 500);
        assert_eq!(queues[0], 10);
        assert!(queues.iter().all(|q| *q <= 20));
        let mut sim = HawkesSimulator::new(params, 7, 10).unwrap();
        let (segment, _) = sim.run(500.0).unwrap();
        let expected = segment
            .events
            .iter()
            .filter(|e| e.kind == QrEventKind::Market)
            .count();
        assert_eq!(markets.iter().map(|m| *m as usize).sum::<usize>(), expected);
        assert!(expected > 0);
    }

    #[test]
    fn test_a_side_day_reproduces_by_seed() {
        let params = parameters(&flow()).unwrap();
        assert_eq!(
            side_day(&params, 3, 10, 200).unwrap(),
            side_day(&params, 3, 10, 200).unwrap()
        );
        assert_ne!(
            side_day(&params, 3, 10, 200).unwrap(),
            side_day(&params, 4, 10, 200).unwrap()
        );
    }

    #[test]
    fn test_the_weights_must_be_three_by_three() {
        let mut bad = flow();
        bad["weights"] = json!([[[0.1, 0.1], [0.1, 0.1]]]);
        assert!(parameters(&bad).is_err());
    }

    #[test]
    fn test_the_noise_is_standard() {
        let mut noise = Noise::new(11);
        let draws: Vec<f64> = (0..200_000).map(|_| noise.normal()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
    }
}
