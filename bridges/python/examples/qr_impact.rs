//! #282's impact scale κ0, the regimes' own impact of one exogenous market order, from #102's
//! design file.
//!
//! `cargo run --release --example qr_impact -- <design.json>`. For each regime it simulates
//! the training days with #102's seeds and, for every market order the simulator draws, the
//! reference's change over the next 10 s in the order's direction (a buy takes the ask, +1; a
//! sell takes the bid, -1), in the simulator's ticks per lot. Orders within 10 s of a day's
//! end are left out. Each side is averaged over its own orders, so they cannot cancel; κ0 is
//! the mean of the two. It prints JSON. Everything here is synthetic.

use std::fs;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::queue_reactive::{
    QrEventKind, QrParameters, QrSide, QrSimulator, QrStep, UnsupportedPolicy,
};
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const SEED_STRIDE: u64 = 1_000_003;
const HORIZON_SECONDS: f64 = 10.0;

fn parameters(regime: &Value) -> Result<QrParameters, Failure> {
    let text = |key: &str| -> Result<&str, Failure> {
        regime
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| key.into())
    };
    let float = |key: &str| -> Result<f64, Failure> {
        regime
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| key.into())
    };
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
    let policy = match text("policy")? {
        "stop" => UnsupportedPolicy::Stop,
        "nearest_supported" => UnsupportedPolicy::NearestSupported,
        other => return Err(format!("unknown policy {other}").into()),
    };
    Ok(QrParameters::new(
        text("tick_size")?.parse::<Decimal>()?,
        float("theta")?,
        float("theta_reinit")?,
        sizes,
        rates,
        supported,
        policy,
    )?)
}

/// Per side, the sum of responses per lot, their squares and the count.
#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    sum: f64,
    squares: f64,
    count: u64,
}

impl Tally {
    fn add(&mut self, value: f64) {
        self.sum += value;
        self.squares += value * value;
        self.count += 1;
    }

    fn json(&self) -> Value {
        let n = self.count as f64;
        let mean = self.sum / n;
        let variance = (self.squares / n - mean * mean).max(0.0);
        json!({ "count": self.count, "mean": mean, "standard_error": (variance / n).sqrt() })
    }
}

/// One day's market orders as `(time, sign, lots, reference before)`, and the reference
/// after each event as `(time, reference)`.
type Day = (Vec<(f64, f64, f64, i64)>, Vec<(f64, i64)>);

fn simulate(
    params: &QrParameters,
    seed: u64,
    reference: i64,
    seconds: f64,
) -> Result<Day, Failure> {
    let mut simulator = QrSimulator::new(params.clone(), seed, reference)?;
    let lot = |depth: u8| -> Result<f64, Failure> {
        let size = params
            .event_sizes()
            .get(usize::from(depth))
            .copied()
            .ok_or("event size")?;
        Ok(size.to_string().parse::<f64>()?)
    };
    let (mut orders, mut path) = (Vec::new(), Vec::new());
    loop {
        let before = simulator.state().reference_ticks;
        match simulator.step()? {
            QrStep::Event { observation, .. } => {
                let now = simulator.state().time_seconds;
                if now > seconds {
                    return Ok((orders, path));
                }
                if observation.kind == QrEventKind::Market {
                    let sign = if observation.side == QrSide::Ask {
                        1.0
                    } else {
                        -1.0
                    };
                    orders.push((now, sign, lot(observation.depth)?, before));
                }
                path.push((now, simulator.state().reference_ticks));
            }
            other => return Err(format!("seed {seed} stopped: {other:?}").into()),
        }
    }
}

fn main() -> Result<(), Failure> {
    let path = std::env::args().nth(1).ok_or("the design file")?;
    let design: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let simulator = design.get("simulator").ok_or("simulator")?;
    let rows = simulator
        .get("rows_per_day")
        .and_then(Value::as_u64)
        .ok_or("rows_per_day")?;
    let sample_ms = simulator
        .get("sample_ms")
        .and_then(Value::as_u64)
        .ok_or("sample_ms")?;
    let reference = simulator
        .get("reference_ticks")
        .and_then(Value::as_i64)
        .ok_or("reference")?;
    let train = design
        .pointer("/splits/train")
        .and_then(Value::as_array)
        .ok_or("splits.train")?;
    let first = train.first().and_then(Value::as_u64).ok_or("train start")?;
    let last = train.get(1).and_then(Value::as_u64).ok_or("train end")?;
    let seconds = (rows - 1) as f64 * sample_ms as f64 / 1000.0;
    let mut out = Vec::new();
    for (index, regime) in simulator
        .get("regimes")
        .and_then(Value::as_array)
        .ok_or("regimes")?
        .iter()
        .enumerate()
    {
        let params = parameters(regime)?;
        let (mut buy, mut sell) = (Tally::default(), Tally::default());
        for day in first..=last {
            let seed = SEED_STRIDE * (index as u64 + 1) + day;
            let (orders, path) = simulate(&params, seed, reference, seconds)?;
            for (time, sign, lots, before) in orders {
                let until = time + HORIZON_SECONDS;
                if until > seconds {
                    continue;
                }
                let at = path.partition_point(|(t, _)| *t <= until);
                let after = path.get(at.wrapping_sub(1)).map_or(reference, |(_, r)| *r);
                let response = sign * (after - before) as f64 / lots;
                if sign > 0.0 {
                    buy.add(response)
                } else {
                    sell.add(response)
                }
            }
        }
        let kappa0 = (buy.sum / buy.count as f64 + sell.sum / sell.count as f64) / 2.0;
        out.push(json!({
            "regime": regime.get("name"),
            "buy": buy.json(),
            "sell": sell.json(),
            "kappa0": kappa0,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "units": "simulator ticks per lot, over the next 10 s",
            "days": [first, last],
            "regimes": out,
        }))?
    );
    Ok(())
}
