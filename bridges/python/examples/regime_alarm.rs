//! The causal regime detector of #112 over the days `regime_days` writes.
//!
//! `cargo run --release --example regime_alarm -- <m5_08 design> <data dir> [threads]`. For each
//! simulator it takes the baseline volatility from the training days. Then, for each simulator
//! `<fit>` the detector is fitted on and every day of either simulator, it writes
//! `day-<n>.alarm-<fit>.json` next to its rows: per row, the alarm (0 or 1), the volatility ratio
//! and VPIN (null until it has its buckets), each from that row and the rows and market orders
//! before it only. A detector fitted on one simulator and run on the other's days is the
//! comparison of simulator error. It prints the baselines and, for the training and validation
//! days only, the alarms; the holdout's are left for the report.
//!
//! `[lambda] [tag]` after the threads is a post hoc sensitivity run, outside the design's rule:
//! the same detector with another EWMA lambda, its files named `day-<n>.alarm-<fit>-<tag>.json`.
//!
//! Everything here is SYNTHETIC.
//!
//! - The volatility is `VolatilityEstimator`'s EWMA of the mid's log returns over the last
//!   `lookback_rows` rows, not annualized, and the ratio divides it by the baseline: the square
//!   root of the mean over training days of each day's simple variance (`calculate_simple`), so
//!   no return spans two days.
//! - VPIN is `VPINCalculator` over the market orders known at each row, one trade per side and
//!   row at the row's mid, the buy first.
//! - The alarm goes on when the ratio reaches `alarm_on_ratio` or VPIN is toxic, and off when the
//!   ratio is below `alarm_off_ratio` and VPIN is not toxic. A ratio that fails to compute keeps
//!   the last one, which starts at 1; before `lookback_rows` rows there is no ratio and the alarm
//!   stays off.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
use market_maker_rs::analytics::vpin::{VPINCalculator, VPINConfig};
use market_maker_rs::market_state::volatility::VolatilityEstimator;
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error + Send + Sync>;

const SIMULATORS: [&str; 2] = ["s1", "s2"];

fn get<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Failure> {
    value
        .get(key)
        .ok_or_else(|| format!("{key} is missing").into())
}

fn decimal(value: &Value, key: &str) -> Result<Decimal, Failure> {
    Ok(get(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text"))?
        .parse::<Decimal>()?)
}

fn number(value: &Value, key: &str) -> Result<u64, Failure> {
    get(value, key)?
        .as_u64()
        .ok_or_else(|| format!("{key} is not a whole number").into())
}

/// The detector's settings, from the design.
#[derive(Debug, Clone, Copy)]
struct Settings {
    lookback: usize,
    lambda: Decimal,
    bucket_volume: Decimal,
    buckets: usize,
    toxicity: Decimal,
    on: Decimal,
    off: Decimal,
}

impl Settings {
    fn of(design: &Value) -> Result<Self, Failure> {
        let detector = get(design, "detector")?;
        let volatility = get(detector, "volatility")?;
        let vpin = get(detector, "vpin")?;
        let lambda = get(volatility, "estimator")?
            .as_str()
            .and_then(|text| text.split("lambda ").nth(1))
            .and_then(|rest| rest.split(',').next())
            .ok_or("the estimator's lambda")?
            .parse()?;
        Ok(Self {
            lookback: usize::try_from(number(volatility, "lookback_rows")?)?,
            lambda,
            bucket_volume: decimal(vpin, "bucket_volume")?,
            buckets: usize::try_from(number(vpin, "buckets")?)?,
            toxicity: decimal(vpin, "toxicity_threshold")?,
            on: decimal(detector, "alarm_on_ratio")?,
            off: decimal(detector, "alarm_off_ratio")?,
        })
    }
}

/// A day: per row its time in milliseconds and its mid in ticks, and the lots bought and sold
/// by market orders since the previous row.
struct Day {
    times: Vec<u64>,
    mids: Vec<Decimal>,
    trades: Vec<[u64; 2]>,
}

fn read_day(dir: &Path, day: u64) -> Result<Day, Failure> {
    let rows: Vec<[u64; 5]> =
        serde_json::from_slice(&fs::read(dir.join(format!("day-{day}.json")))?)?;
    let trades: Vec<[u64; 2]> =
        serde_json::from_slice(&fs::read(dir.join(format!("day-{day}.trades.json")))?)?;
    if trades.len() != rows.len() {
        return Err(format!(
            "{}: day {day} has {} trade rows for {} rows",
            dir.display(),
            trades.len(),
            rows.len()
        )
        .into());
    }
    let mids = rows
        .iter()
        .map(|r| Decimal::from(r[1] + r[3]) / Decimal::TWO)
        .collect();
    let times = rows.iter().map(|r| r[0] / 1_000_000).collect();
    Ok(Day {
        times,
        mids,
        trades,
    })
}

/// The training days' baseline: the root of the mean of each day's simple variance.
fn baseline(days: &[Day]) -> Result<Decimal, Failure> {
    let estimator = VolatilityEstimator::with_annualization_factor(Decimal::ONE);
    let mut total = Decimal::ZERO;
    for day in days {
        let sd = estimator.calculate_simple(&day.mids)?;
        total = total
            .checked_add(sd.checked_mul(sd).ok_or("variance")?)
            .ok_or("variance")?;
    }
    let count = Decimal::from(u64::try_from(days.len())?);
    let mean = total.checked_div(count).ok_or("no training days")?;
    root(mean)
}

/// The square root of a non-negative decimal, by Newton's method in `Decimal`.
fn root(value: Decimal) -> Result<Decimal, Failure> {
    if value < Decimal::ZERO {
        return Err("a negative variance".into());
    }
    if value == Decimal::ZERO {
        return Ok(Decimal::ZERO);
    }
    let mut x = if value > Decimal::ONE {
        value
    } else {
        Decimal::ONE
    };
    for _ in 0..200 {
        let next = (x + value / x) / Decimal::TWO;
        if next == x {
            break;
        }
        x = next;
    }
    Ok(x)
}

/// One row of the detector's output: alarm, ratio, VPIN.
type AlarmRow = (bool, Option<Decimal>, Option<Decimal>);

/// The detector over one day. It reads nothing past each row.
fn detect(day: &Day, baseline: Decimal, settings: &Settings) -> Result<Vec<AlarmRow>, Failure> {
    let estimator = VolatilityEstimator::with_annualization_factor(Decimal::ONE);
    let mut vpin = VPINCalculator::new(VPINConfig::new(
        settings.bucket_volume,
        settings.buckets,
        settings.toxicity,
    )?);
    let mut window: VecDeque<Decimal> = VecDeque::with_capacity(settings.lookback + 1);
    let mut ratio = Decimal::ONE;
    let mut alarm = false;
    let mut out = Vec::with_capacity(day.mids.len());
    for ((time, mid), [bought, sold]) in day.times.iter().zip(&day.mids).zip(&day.trades) {
        for (lots, side) in [(*bought, TradeSide::Buy), (*sold, TradeSide::Sell)] {
            if lots > 0 {
                let _ = vpin.add_trade(&Trade::new(*mid, Decimal::from(lots), side, *time));
            }
        }
        window.push_back(*mid);
        if window.len() > settings.lookback {
            window.pop_front();
        }
        if window.len() < settings.lookback {
            out.push((false, None, vpin.get_vpin()));
            continue;
        }
        let mids: Vec<Decimal> = window.iter().copied().collect();
        if let Ok(volatility) = estimator.calculate_ewma(&mids, settings.lambda)
            && let Some(next) = volatility.checked_div(baseline)
        {
            ratio = next;
        }
        let toxic = vpin.is_toxic();
        if !alarm && (ratio >= settings.on || toxic) {
            alarm = true;
        } else if alarm && ratio < settings.off && !toxic {
            alarm = false;
        }
        out.push((alarm, Some(ratio), vpin.get_vpin()));
    }
    Ok(out)
}

fn days_of(splits: &Value, name: &str) -> Result<Vec<u64>, Failure> {
    let range = get(splits, name)?.as_array().ok_or(name)?;
    let bound = |i: usize| range.get(i).and_then(Value::as_u64).ok_or("split bound");
    Ok((bound(0)?..=bound(1)?).collect())
}

/// Alarm onsets and seconds on, from a day's output.
fn summary(rows: &[AlarmRow], times: &[u64]) -> (Vec<u64>, u64) {
    let mut onsets = Vec::new();
    let mut on = 0;
    let mut before = false;
    for ((alarm, _, _), time) in rows.iter().zip(times) {
        if *alarm && !before {
            onsets.push(time % 86_400_000 / 1000);
        }
        on += u64::from(*alarm);
        before = *alarm;
    }
    (onsets, on)
}

/// The rows of an alarm file: alarm, ratio and VPIN, decimals as text rounded to 6 places.
fn written(out: &[AlarmRow]) -> Vec<Value> {
    out.iter()
        .map(|(alarm, ratio, vpin)| {
            json!([
                u8::from(*alarm),
                ratio.map(|r| r.round_dp(6).to_string()),
                vpin.map(|v| v.round_dp(6).to_string())
            ])
        })
        .collect()
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let design: Value = serde_json::from_slice(&fs::read(PathBuf::from(
        args.next().ok_or("the design file")?,
    ))?)?;
    let data = PathBuf::from(args.next().ok_or("the data directory")?);
    let threads: usize = args.next().map_or(Ok(4), |t| t.parse())?;
    let mut settings = Settings::of(&design)?;
    // A post hoc sensitivity run: another lambda, written under a tag beside the frozen files.
    let tag = match args.next() {
        Some(lambda) => {
            settings.lambda = lambda.parse()?;
            format!("-{}", args.next().ok_or("a sensitivity run needs a tag")?)
        }
        None => String::new(),
    };
    let splits = get(&design, "splits")?;
    let mut baselines = Vec::new();
    for simulator in SIMULATORS {
        let train: Vec<Day> = days_of(splits, "train")?
            .into_iter()
            .map(|day| read_day(&data.join("train").join(simulator), day))
            .collect::<Result<_, _>>()?;
        baselines.push((simulator, baseline(&train)?));
    }
    let mut report = Vec::new();
    for (fitted, base) in &baselines {
        for simulator in SIMULATORS {
            let mut jobs = Vec::new();
            for split in ["train", "validation", "holdout"] {
                for day in days_of(splits, split)? {
                    jobs.push((split, day));
                }
            }
            let chunks: Vec<Vec<(&str, u64)>> = (0..threads)
                .map(|t| jobs.iter().copied().skip(t).step_by(threads).collect())
                .collect();
            let results: Vec<Result<Vec<Value>, Failure>> = std::thread::scope(|scope| {
                let handles: Vec<_> = chunks
                    .iter()
                    .map(|chunk| {
                        let data = &data;
                        let base = *base;
                        let tag = &tag;
                        scope.spawn(move || -> Result<Vec<Value>, Failure> {
                            let mut records = Vec::new();
                            for (split, day) in chunk {
                                let dir = data.join(split).join(simulator);
                                let rows = read_day(&dir, *day)?;
                                let out = detect(&rows, base, &settings)?;
                                fs::write(
                                    dir.join(format!("day-{day}.alarm-{fitted}{tag}.json")),
                                    serde_json::to_vec(&written(&out))?,
                                )?;
                                if *split != "holdout" {
                                    let (onsets, on) = summary(&out, &rows.times);
                                    records.push(json!({
                                        "split": split, "day": day,
                                        "alarm_onsets_s": onsets, "seconds_on": on,
                                    }));
                                }
                            }
                            Ok(records)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| Err("a thread panicked".into())))
                    .collect()
            });
            let mut records = Vec::new();
            for result in results {
                records.extend(result?);
            }
            records.sort_by_key(|r| r["day"].as_u64());
            report.push(json!({
                "fitted_on": fitted,
                "simulator": simulator,
                "baseline": base.round_dp(12).to_string(),
                "days": records,
            }));
        }
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings {
            lookback: 10,
            lambda: Decimal::new(9, 1),
            bucket_volume: Decimal::from(50),
            buckets: 20,
            toxicity: Decimal::new(7, 1),
            on: Decimal::new(18, 1),
            off: Decimal::new(13, 1),
        }
    }

    fn day(mids: Vec<i64>, trades: Vec<[u64; 2]>) -> Day {
        Day {
            times: (0..mids.len() as u64).map(|i| i * 1000).collect(),
            mids: mids.into_iter().map(Decimal::from).collect(),
            trades,
        }
    }

    #[test]
    fn test_the_settings_come_from_the_design() {
        let design: Value =
            serde_json::from_slice(&fs::read("experiments/m5_08/design.json").unwrap()).unwrap();
        let s = Settings::of(&design).unwrap();
        assert_eq!(s.lookback, 300);
        assert_eq!(s.lambda, Decimal::new(9, 1));
        assert_eq!((s.on, s.off), (Decimal::new(18, 1), Decimal::new(13, 1)));
        assert_eq!((s.buckets, s.toxicity), (20, Decimal::new(7, 1)));
    }

    #[test]
    fn test_the_root_is_exact_enough() {
        assert_eq!(root(Decimal::from(4)).unwrap(), Decimal::TWO);
        let r = root(Decimal::new(2, 6)).unwrap();
        assert!((r * r - Decimal::new(2, 6)).abs() < Decimal::new(1, 24));
    }

    #[test]
    fn test_the_alarm_follows_the_ratio_with_hysteresis_and_reads_no_later_row() {
        // Calm wiggles of 1 tick, then 6 tick swings, then calm again.
        let mut mids = Vec::new();
        for i in 0..200 {
            let swing = if (80..120).contains(&i) { 6 } else { 1 };
            mids.push(10_000 + if i % 2 == 0 { swing } else { -swing });
        }
        let rows = mids.len();
        let calm = day(mids[..60].to_vec(), vec![[0, 0]; 60]);
        let base = baseline(&[calm]).unwrap();
        let full = day(mids.clone(), vec![[0, 0]; rows]);
        let out = detect(&full, base, &settings()).unwrap();
        assert!(out[..9].iter().all(|r| !r.0 && r.1.is_none()));
        assert!(out[20..80].iter().all(|r| !r.0));
        let onset = out.iter().position(|r| r.0).unwrap();
        assert!((80..90).contains(&onset), "{onset}");
        assert!(out[100..120].iter().all(|r| r.0));
        assert!(!out.last().unwrap().0);
        // A prefix gives the same rows: nothing after a row changes it.
        let prefix = day(mids[..100].to_vec(), vec![[0, 0]; 100]);
        assert_eq!(detect(&prefix, base, &settings()).unwrap(), out[..100]);
    }

    #[test]
    fn test_one_sided_flow_is_toxic() {
        let mids = vec![10_000; 400];
        let out = detect(&day(mids, vec![[5, 0]; 400]), Decimal::ONE, &settings()).unwrap();
        // 20 buckets of 50 lots need 200 rows; the ratio is 0 on a flat mid.
        assert!(out[150].2.is_none() && !out[150].0);
        assert_eq!(out[250].2, Some(Decimal::ONE));
        assert!(out[250].0);
    }
}
