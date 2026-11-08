//! The synthetic days of #112: one path a day that changes regime at the design's times.
//!
//! `cargo run --release --example regime_days -- <m5_08 design> <m4_06 design> <m5_02 design>
//! <out dir>`. For each simulator and day it writes the rows as the bridge reads them under
//! `<out dir>/<split>/<simulator>/day-<n>.json`, next to them `day-<n>.trades.json` with the
//! market orders known at each row, and prints the digests for the manifest.
//!
//! Everything here is SYNTHETIC, from made-up parameters, and says nothing about a market.
//!
//! - A day is calm except for the design's segments, in seconds of the day, start inclusive and
//!   end exclusive. The events between rows `s` and `s + 1` come from the regime of second `s`.
//! - `s1` is the queue-reactive simulator with #103's rates per regime. At a change the book is
//!   kept, its queues clipped to the new regime's cap and limits, before the row of that second
//!   is written; its generator and its memory of the last events carry on.
//! - `s2` is #107 C's Hawkes flow at impact 1.0 (two simulators, the buy side lifting the ask).
//!   Its regimes follow #237: stressed doubles the market order baseline of both sides, takes
//!   the buy side's a further 1.5 times, and triples the diffusion; thin halves the limit order
//!   rate. At a change the queues and the excitation carry on. A row shows the events before
//!   its time, as in #107 C.
//! - `trades.json` holds, per row, the buy and the sell volume of the market orders since the
//!   previous row, in lots: what a causal detector knows at that row. No policy reads it.
//! - The day's seed is `1000003 * 4 + day`, a regime number after #103's three, so no day here
//!   is one of #103's or #107's.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::hawkes::{HawkesParameters, HawkesSimulator};
use market_maker_rs::backtest::kernel::{KernelConfig, KernelDataset};
use market_maker_rs::backtest::queue_reactive::{
    MAX_LIMITS, QrEventKind, QrParameters, QrSide, QrSimulator, QrState, UnsupportedPolicy,
};
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const DAY_MS: u64 = 86_400_000;
const SEED_STRIDE: u64 = 1_000_003;
const SEED_REGIME: u64 = 4;
const REGIMES: [&str; 3] = ["calm", "stressed", "thin"];
/// #237's stress: market order baseline, the buy side's tilt and the diffusion.
const STRESS_FLOW: f64 = 2.0;
const STRESS_TILT: f64 = 1.5;
const STRESS_DIFFUSION: f64 = 3.0;
/// The thin regime's limit order rate, against calm's.
const THIN_LIMIT: f64 = 0.5;
/// #107 C's impact setting that is the calm regime of `s2`.
const S2_SETTING: &str = "impact_high";

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

fn decimal(value: &Value, key: &str) -> Result<Decimal, Failure> {
    Ok(text(value, key)?.parse::<Decimal>()?)
}

fn floats(value: &Value) -> Result<Vec<f64>, Failure> {
    value
        .as_array()
        .ok_or("not a list")?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "not a number".into()))
        .collect()
}

/// A regime's index in [`REGIMES`].
fn regime_index(name: &str) -> Result<usize, Failure> {
    REGIMES
        .iter()
        .position(|r| *r == name)
        .ok_or_else(|| format!("unknown regime {name}").into())
}

/// The regime of each second of a day, from the design's segments.
fn schedule(segments: &Value, day: u64, rows: usize) -> Result<Vec<usize>, Failure> {
    let mut regime = vec![0; rows];
    let Some(list) = segments.get(day.to_string()) else {
        return Ok(regime);
    };
    for segment in list.as_array().ok_or("segments")? {
        let parts = segment.as_array().ok_or("segment")?;
        let name = parts
            .first()
            .and_then(Value::as_str)
            .ok_or("segment regime")?;
        let bound = |i: usize| -> Result<usize, Failure> {
            Ok(usize::try_from(
                parts
                    .get(i)
                    .and_then(Value::as_u64)
                    .ok_or("segment bound")?,
            )?)
        };
        let (start, end) = (bound(1)?, bound(2)?);
        if start >= end || end > rows {
            return Err(format!("day {day}: segment {start}..{end} is out of the day").into());
        }
        let index = regime_index(name)?;
        for slot in regime.get_mut(start..end).ok_or("segment")? {
            if *slot != 0 {
                return Err(format!("day {day}: segments overlap").into());
            }
            *slot = index;
        }
    }
    Ok(regime)
}

fn qr_parameters(regime: &Value) -> Result<QrParameters, Failure> {
    let sizes = get(regime, "event_sizes")?
        .as_array()
        .ok_or("event_sizes")?
        .iter()
        .map(|size| Ok(size.as_str().ok_or("event size")?.parse::<Decimal>()?))
        .collect::<Result<Vec<_>, Failure>>()?;
    let rates = get(regime, "rates")?
        .as_array()
        .ok_or("rates")?
        .iter()
        .map(|table| {
            table
                .as_array()
                .ok_or("rate table")?
                .iter()
                .map(|row| {
                    let row = floats(row)?;
                    match row.as_slice() {
                        [l, c, m] => Ok((*l, *c, *m)),
                        _ => Err::<_, Failure>("a rate row has three rates".into()),
                    }
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

/// #103's parameters, in the order of [`REGIMES`].
fn s1_parameters(m4_06: &Value) -> Result<Vec<QrParameters>, Failure> {
    let regimes = get(get(m4_06, "simulator")?, "regimes")?
        .as_array()
        .ok_or("regimes")?;
    REGIMES
        .iter()
        .map(|name| {
            let regime = regimes
                .iter()
                .find(|r| r.get("name").and_then(Value::as_str) == Some(name))
                .ok_or_else(|| format!("#103 has no {name} regime"))?;
            qr_parameters(regime)
        })
        .collect()
}

/// A state carried into other parameters: the book clipped to their cap and limits.
fn splice_qr(state: QrState, params: &QrParameters) -> QrState {
    let limits = params.limits();
    let cap = u32::try_from(params.max_queue()).unwrap_or(u32::MAX);
    let clip = |queues: [u32; MAX_LIMITS]| {
        let mut out = [0; MAX_LIMITS];
        for (d, (slot, queue)) in out.iter_mut().zip(queues).enumerate() {
            *slot = if d < limits { queue.min(cap) } else { 0 };
        }
        out
    };
    QrState {
        bid: clip(state.bid),
        ask: clip(state.ask),
        parameters_fingerprint: params.fingerprint(),
        ..state
    }
}

/// The row of a queue-reactive state, as `rl::qr_market` writes it: limit `d` at `d + 1/2`
/// of the simulator's ticks from the reference, an empty side one limit further and no size.
fn qr_row(
    state: &QrState,
    params: &QrParameters,
    half_tick: Decimal,
    timestamp_ms: u64,
) -> Result<MarketTick, Failure> {
    let limits = params.limits();
    let side = |queues: &[u32; MAX_LIMITS], sign: i64| -> Result<(Decimal, Decimal), Failure> {
        let offset = queues.iter().position(|q| *q > 0).unwrap_or(limits);
        let size = match queues.get(offset).filter(|_| offset < limits) {
            Some(queue) => params
                .event_sizes()
                .get(offset)
                .and_then(|unit| unit.checked_mul(Decimal::from(*queue)))
                .ok_or("size")?,
            None => Decimal::ZERO,
        };
        let away = i64::try_from(offset)?
            .checked_mul(2)
            .and_then(|o| o.checked_add(1))
            .ok_or("offset")?;
        let halves = state
            .reference_ticks
            .checked_mul(2)
            .and_then(|r| r.checked_add(sign.checked_mul(away)?))
            .ok_or("price")?;
        let price = Decimal::from(halves)
            .checked_mul(half_tick)
            .ok_or("price")?;
        Ok((price, size))
    };
    let (bid, bid_size) = side(&state.bid, -1)?;
    let (ask, ask_size) = side(&state.ask, 1)?;
    Ok(MarketTick::new(timestamp_ms, bid, bid_size, ask, ask_size))
}

/// A day's rows and, per row, the buy and the sell volume known at it, in lots.
type Day = (Vec<MarketTick>, Vec<[u64; 2]>);

/// A lot count from a size in the asset, which the kernel's lot of 1 makes whole.
fn lots(size: Decimal) -> Result<u64, Failure> {
    if size.fract() != Decimal::ZERO || size < Decimal::ZERO {
        return Err(format!("size {size} is not whole lots").into());
    }
    Ok(u64::try_from(size.mantissa() / 10_i128.pow(size.scale()))?)
}

/// One `s1` day under a schedule of regimes per second.
fn s1_day(
    params: &[QrParameters],
    regimes: &[usize],
    seed: u64,
    reference_ticks: i64,
    start_ms: u64,
    sample_ms: u64,
    half_tick: Decimal,
) -> Result<Day, Failure> {
    let first = params
        .get(*regimes.first().ok_or("no rows")?)
        .ok_or("regime")?;
    let mut sim = QrSimulator::new(first.clone(), seed, reference_ticks)?;
    let mut current = *regimes.first().ok_or("no rows")?;
    let seconds = sample_ms as f64 / 1000.0;
    let mut ticks = Vec::with_capacity(regimes.len());
    let mut trades = Vec::with_capacity(regimes.len());
    let mut known = [0_u64; 2];
    for (row, regime) in regimes.iter().enumerate() {
        let here = params.get(*regime).ok_or("regime")?;
        if *regime != current {
            sim = QrSimulator::from_state(here.clone(), splice_qr(sim.state(), here))?;
            current = *regime;
        }
        let at = start_ms + u64::try_from(row)? * sample_ms;
        ticks.push(qr_row(&sim.state(), here, half_tick, at)?);
        trades.push(known);
        if row + 1 == regimes.len() {
            break;
        }
        let run = sim.run(seconds, seconds)?;
        if let Some(stop) = run.stopped {
            return Err(format!("the simulator stopped at row {row}: {stop:?}").into());
        }
        known = [0, 0];
        for event in run.events.iter().filter(|e| e.kind == QrEventKind::Market) {
            let size = here
                .event_sizes()
                .get(usize::from(event.depth))
                .copied()
                .ok_or("event size")?;
            // A market order at the ask is a buy; at the bid, a sell.
            let slot = usize::from(matches!(event.side, QrSide::Bid));
            known[slot] += lots(size)?;
        }
    }
    Ok((ticks, trades))
}

/// The Hawkes flow of one side in one regime.
fn hawkes_parameters(
    flow: &Value,
    limit_scale: f64,
    market_scale: f64,
) -> Result<HawkesParameters, Failure> {
    let cap = usize::try_from(number(flow, "queue_cap")?)?;
    let rates = get(flow, "rates")?;
    let (limit, cancel, market) = (
        float(rates, "limit")? * limit_scale,
        float(rates, "cancel_per_lot")?,
        float(rates, "market")? * market_scale,
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
            if rows.len() != 3 {
                return Err::<_, Failure>("a weight matrix is 3 x 3".into());
            }
            for (slot, row) in out.iter_mut().zip(rows) {
                let row = floats(row)?;
                if row.len() != 3 {
                    return Err("a weight matrix is 3 x 3".into());
                }
                slot.copy_from_slice(&row);
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(HawkesParameters::new(table, decays, weights)?)
}

/// `s2`'s regimes: per regime, the buy side's flow, the sell side's and the diffusion scale.
fn s2_regimes(flow: &Value) -> Result<Vec<(HawkesParameters, HawkesParameters, f64)>, Failure> {
    REGIMES
        .iter()
        .map(|name| {
            let (limit, market, tilt, diffusion) = match *name {
                "stressed" => (1.0, STRESS_FLOW, STRESS_TILT, STRESS_DIFFUSION),
                "thin" => (THIN_LIMIT, 1.0, 1.0, 1.0),
                _ => (1.0, 1.0, 1.0, 1.0),
            };
            Ok((
                hawkes_parameters(flow, limit, market * tilt)?,
                hawkes_parameters(flow, limit, market)?,
                diffusion,
            ))
        })
        .collect()
}

/// A Hawkes state carried into other parameters with the same kernels and cap.
fn splice_hawkes(
    sim: &HawkesSimulator,
    params: &HawkesParameters,
) -> Result<HawkesSimulator, Failure> {
    let mut state = sim.state();
    let bytes = params.fingerprint().to_le_bytes();
    let (low, high) = bytes.split_at(8);
    state.fingerprint_low = u64::from_le_bytes(low.try_into()?);
    state.fingerprint_high = u64::from_le_bytes(high.try_into()?);
    Ok(HawkesSimulator::from_state(params.clone(), state)?)
}

/// A small generator for the diffusion, apart from the flow's (xorshift64*), as #107 C's.
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

/// What `s2` takes from #107 C's flow and kernel.
struct S2 {
    regimes: Vec<(HawkesParameters, HawkesParameters, f64)>,
    initial: u32,
    reference: f64,
    spread: i64,
    diffusion: f64,
    impact: f64,
    tick: Decimal,
}

/// One `s2` day under a schedule of regimes per second.
fn s2_day(
    s2: &S2,
    regimes: &[usize],
    seed: u64,
    start_ms: u64,
    sample_ms: u64,
) -> Result<Day, Failure> {
    let first = *regimes.first().ok_or("no rows")?;
    let (buy0, sell0, _) = s2.regimes.get(first).ok_or("regime")?;
    let mut buy = HawkesSimulator::new(buy0.clone(), seed.wrapping_mul(2), s2.initial)?;
    let mut sell = HawkesSimulator::new(sell0.clone(), seed.wrapping_mul(2) + 1, s2.initial)?;
    let mut noise = Noise::new(seed);
    let mut current = first;
    let mut mid = s2.reference;
    let mut ticks = Vec::with_capacity(regimes.len());
    let mut trades = Vec::with_capacity(regimes.len());
    let mut known = [0_u64; 2];
    for (row, regime) in regimes.iter().enumerate() {
        let (buy_params, sell_params, scale) = s2.regimes.get(*regime).ok_or("regime")?;
        if *regime != current {
            buy = splice_hawkes(&buy, buy_params)?;
            sell = splice_hawkes(&sell, sell_params)?;
            current = *regime;
        }
        let bid = mid.floor() as i64;
        if bid < 1 {
            return Err(format!("row {row}: the bid left the grid").into());
        }
        let size = |q: u32| Decimal::from(q.max(1));
        ticks.push(MarketTick::new(
            start_ms + u64::try_from(row)? * sample_ms,
            Decimal::from(bid) * s2.tick,
            size(sell.state().queue),
            Decimal::from(bid + s2.spread) * s2.tick,
            size(buy.state().queue),
        ));
        trades.push(known);
        if row + 1 == regimes.len() {
            break;
        }
        let count = |sim: &mut HawkesSimulator| -> Result<u64, Failure> {
            let (segment, stop) = sim.run(1.0)?;
            if let Some(stop) = stop {
                return Err(format!("the flow stopped at row {row}: {stop:?}").into());
            }
            Ok(segment
                .events
                .iter()
                .filter(|e| e.kind == QrEventKind::Market)
                .count() as u64)
        };
        known = [count(&mut buy)?, count(&mut sell)?];
        let net = known[0] as f64 - known[1] as f64;
        mid += s2.impact * net + s2.diffusion * scale * noise.normal();
    }
    Ok((ticks, trades))
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

fn write_day(
    dir: &Path,
    day: u64,
    dataset: &KernelDataset,
    trades: &[[u64; 2]],
) -> Result<(), Failure> {
    let rows: Vec<[u64; 5]> = dataset
        .ticks()
        .iter()
        .map(|t| [t.ts_ns, t.bid, t.bid_lots, t.ask, t.ask_lots])
        .collect();
    fs::create_dir_all(dir)?;
    fs::write(
        dir.join(format!("day-{day}.json")),
        serde_json::to_vec(&rows)?,
    )?;
    fs::write(
        dir.join(format!("day-{day}.trades.json")),
        serde_json::to_vec(trades)?,
    )?;
    Ok(())
}

fn kernel_config(kernel: &Value, rows: u64) -> Result<KernelConfig, Failure> {
    Ok(KernelConfig::new(
        decimal(kernel, "tick_size")?,
        decimal(kernel, "lot_size")?,
        decimal(kernel, "max_price")?,
        decimal(kernel, "max_order_size")?,
        rows,
    ))
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let mut read = || -> Result<Value, Failure> {
        let path = PathBuf::from(
            args.next()
                .ok_or("three design files and an output directory")?,
        );
        Ok(serde_json::from_slice(&fs::read(&path)?)?)
    };
    let (design, m4_06, m5_02) = (read()?, read()?, read()?);
    let out = PathBuf::from(args.next().ok_or("the output directory")?);
    let rows_per_day = number(&design, "rows_per_day")?;
    let rows = usize::try_from(rows_per_day)?;
    let sample_ms = number(&design, "sample_ms")?;
    let days = number(&design, "days")?;
    let splits = get(&design, "splits")?;
    let segments = get(get(&design, "schedule")?, "segments")?;

    let s1 = s1_parameters(&m4_06)?;
    let s1_config = kernel_config(get(&m4_06, "kernel")?, rows_per_day)?;
    let reference_ticks = i64::try_from(number(get(&m4_06, "simulator")?, "reference_ticks")?)?;

    let causal = get(&m5_02, "causal")?;
    let flow = get(causal, "flow")?;
    let s2_config = kernel_config(get(causal, "kernel")?, rows_per_day)?;
    let impact = get(causal, "settings")?
        .as_array()
        .ok_or("settings")?
        .iter()
        .find(|s| s.get("name").and_then(Value::as_str) == Some(S2_SETTING))
        .ok_or("the impact setting")?;
    let s2 = S2 {
        regimes: s2_regimes(flow)?,
        initial: u32::try_from(number(flow, "initial_queue")?)?,
        reference: number(flow, "reference_ticks")? as f64,
        spread: i64::try_from(number(flow, "spread_ticks")?)?,
        diffusion: float(flow, "diffusion_ticks")?,
        impact: float(impact, "impact_ticks")?,
        tick: decimal(get(causal, "kernel")?, "tick_size")?,
    };

    let mut records = Vec::new();
    for day in 1..=days {
        let regimes = schedule(segments, day, rows)?;
        let seed = SEED_STRIDE
            .checked_mul(SEED_REGIME)
            .and_then(|base| base.checked_add(day))
            .ok_or("seed")?;
        let start_ms = (day - 1).checked_mul(DAY_MS).ok_or("day")?;
        let split = split_of(splits, day)?;
        let mut seconds = BTreeMap::new();
        for regime in &regimes {
            *seconds.entry(REGIMES[*regime]).or_insert(0_u64) += 1;
        }
        for name in ["s1", "s2"] {
            let ((ticks, trades), config) = if name == "s1" {
                (
                    s1_day(
                        &s1,
                        &regimes,
                        seed,
                        reference_ticks,
                        start_ms,
                        sample_ms,
                        s1_config.tick_size,
                    )?,
                    &s1_config,
                )
            } else {
                (
                    s2_day(&s2, &regimes, seed, start_ms, sample_ms)?,
                    &s2_config,
                )
            };
            let seen = KernelDataset::from_ticks(&ticks, config)?;
            write_day(&out.join(split).join(name), day, &seen, &trades)?;
            records.push(json!({
                "simulator": name,
                "day": day,
                "split": split,
                "seed": seed,
                "rows": seen.id().len,
                "dataset_digest": format!("{:032x}", seen.id().digest),
                "seconds_per_regime": seconds,
                "market_order_lots": trades.iter().map(|t| [t[0], t[1]]).fold([0_u64; 2], |a, t| [a[0] + t[0], a[1] + t[1]]),
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "issue": design["issue"],
            "design_version": design["design_version"],
            "seed": "1000003 * 4 + day, for both simulators",
            "days": records,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use market_maker_rs::rl::{QrEpisode, qr_market};

    fn m4_06() -> Value {
        serde_json::from_slice(&fs::read("experiments/m4_06/design.json").unwrap()).unwrap()
    }

    fn m5_02() -> Value {
        serde_json::from_slice(&fs::read("experiments/m5_02/design.json").unwrap()).unwrap()
    }

    fn s2_of(design: &Value) -> S2 {
        let causal = &design["causal"];
        S2 {
            regimes: s2_regimes(&causal["flow"]).unwrap(),
            initial: 10,
            reference: 10_000.0,
            spread: 1,
            diffusion: 1.0,
            impact: 1.0,
            tick: Decimal::new(5, 3),
        }
    }

    #[test]
    fn test_the_schedule_marks_each_second_once() {
        let segments = json!({"3": [["thin", 10, 20]], "5": [["stressed", 0, 5], ["thin", 5, 8]]});
        assert_eq!(schedule(&segments, 1, 30).unwrap(), vec![0; 30]);
        let day = schedule(&segments, 3, 30).unwrap();
        assert_eq!(day.iter().filter(|r| **r == 2).count(), 10);
        assert_eq!((day[9], day[10], day[19], day[20]), (0, 2, 2, 0));
        let day = schedule(&segments, 5, 10).unwrap();
        assert_eq!(day, vec![1, 1, 1, 1, 1, 2, 2, 2, 0, 0]);
        assert!(schedule(&json!({"1": [["thin", 0, 5], ["stressed", 4, 6]]}), 1, 10).is_err());
        assert!(schedule(&json!({"1": [["thin", 5, 50]]}), 1, 10).is_err());
        assert!(schedule(&json!({"1": [["windy", 0, 5]]}), 1, 10).is_err());
    }

    #[test]
    fn test_a_calm_s1_day_is_the_simulator_s_own() {
        let params = s1_parameters(&m4_06()).unwrap();
        let config = kernel_config(&m4_06()["kernel"], 600).unwrap();
        let (ticks, trades) = s1_day(
            &params,
            &[0; 600],
            77,
            10_000,
            3_000,
            1000,
            config.tick_size,
        )
        .unwrap();
        let ours = KernelDataset::from_ticks(&ticks, &config).unwrap();
        let episode = QrEpisode {
            seed: 77,
            reference_ticks: 10_000,
            start_ms: 3_000,
            sample_ms: 1000,
            rows: 600,
        };
        let theirs = qr_market(&params[0], &episode, &config).unwrap();
        assert_eq!(ours.ticks(), theirs.dataset.ticks());
        assert_eq!(trades[0], [0, 0]);
        assert!(trades.iter().map(|t| t[0] + t[1]).sum::<u64>() > 0);
    }

    #[test]
    fn test_an_s1_change_clips_the_book_and_carries_the_generator() {
        let params = s1_parameters(&m4_06()).unwrap();
        let mut regimes = vec![0; 400];
        regimes[200..300].fill(2);
        let config = kernel_config(&m4_06()["kernel"], 400).unwrap();
        let (changed, _) = s1_day(&params, &regimes, 5, 10_000, 0, 1000, config.tick_size).unwrap();
        let (calm, _) = s1_day(&params, &[0; 400], 5, 10_000, 0, 1000, config.tick_size).unwrap();
        // The same path until the change, a different one after.
        assert_eq!(changed[..200], calm[..200]);
        assert_ne!(changed[200..], calm[200..]);
        let cap = Decimal::from(params[2].max_queue() as u64);
        for tick in &changed[200..300] {
            // Every size at the touch fits thin's cap, in event sizes of 1 or 2.
            assert!(tick.bid_size <= cap * Decimal::TWO && tick.ask_size <= cap * Decimal::TWO);
        }
        let state = QrSimulator::new(params[0].clone(), 9, 10_000)
            .unwrap()
            .state();
        let mut full = state;
        full.bid = [4, 4, 0, 0, 0];
        let clipped = splice_qr(full, &params[2]);
        assert_eq!(clipped.bid[..2], [3, 3]);
        assert_eq!(clipped.rng_state, state.rng_state);
        assert!(QrSimulator::from_state(params[2].clone(), clipped).is_ok());
    }

    #[test]
    fn test_s2_reproduces_by_seed_and_runs_on_across_a_change() {
        let s2 = s2_of(&m5_02());
        let mut regimes = vec![0; 600];
        regimes[300..450].fill(1);
        let (a, ta) = s2_day(&s2, &regimes, 3, 0, 1000).unwrap();
        let (b, tb) = s2_day(&s2, &regimes, 3, 0, 1000).unwrap();
        assert_eq!((a.clone(), ta.clone()), (b, tb));
        let (calm, tc) = s2_day(&s2, &[0; 600], 3, 0, 1000).unwrap();
        assert_eq!(a[..301], calm[..301]);
        assert_eq!(ta[..301], tc[..301]);
        assert_ne!(ta[301..], tc[301..]);
        assert_eq!(ta[0], [0, 0]);
    }

    #[test]
    fn test_per_row_hawkes_runs_equal_one_long_run() {
        let flow = &m5_02()["causal"]["flow"];
        let params = hawkes_parameters(flow, 1.0, 1.0).unwrap();
        let mut long = HawkesSimulator::new(params.clone(), 11, 10).unwrap();
        let (segment, _) = long.run(300.0).unwrap();
        let mut short = HawkesSimulator::new(params.clone(), 11, 10).unwrap();
        let mut markets = 0;
        for _ in 0..300 {
            short = splice_hawkes(&short, &params).unwrap();
            let (piece, _) = short.run(1.0).unwrap();
            markets += piece
                .events
                .iter()
                .filter(|e| e.kind == QrEventKind::Market)
                .count();
        }
        let expected = segment
            .events
            .iter()
            .filter(|e| e.kind == QrEventKind::Market)
            .count();
        assert_eq!(markets, expected);
        assert_eq!(short.state().queue, long.state().queue);
    }

    #[test]
    fn test_s2_stress_raises_the_buy_flow_more() {
        let s2 = s2_of(&m5_02());
        let (_, calm) = s2_day(&s2, &[0; 3000], 21, 0, 1000).unwrap();
        let (_, stressed) = s2_day(&s2, &[1; 3000], 21, 0, 1000).unwrap();
        let sum = |t: &[[u64; 2]], i: usize| t.iter().map(|r| r[i]).sum::<u64>();
        assert!(sum(&stressed, 0) > sum(&calm, 0));
        assert!(sum(&stressed, 0) > sum(&stressed, 1));
        let (buy, sell, scale) = &s2.regimes[1];
        assert_eq!(buy.rates()[1].2, 0.3 * 2.0 * 1.5);
        assert_eq!(sell.rates()[1].2, 0.3 * 2.0);
        assert_eq!(*scale, 3.0);
        assert_eq!(s2.regimes[2].0.rates()[0].0, 0.75);
    }

    #[test]
    fn test_lots_are_whole() {
        assert_eq!(lots(Decimal::TWO).unwrap(), 2);
        assert_eq!(lots(Decimal::new(20, 1)).unwrap(), 2);
        assert!(lots(Decimal::new(15, 1)).is_err());
    }
}
