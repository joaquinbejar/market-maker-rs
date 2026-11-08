//! The queue-reactive side of the MDQR study (#106): the synthetic truth of its stage S0
//! and the discrepancy between two trajectories.
//!
//! `cargo run --release --example mdqr_qr -- truth <design.json> <out dir>` simulates every
//! window of the design's stage S0 with the queue-reactive simulator at the design's
//! parameters, and writes two files per window under `<out dir>/<set>/window-<n>`:
//! `.trajectory.json`, the sampled book with its events and reference outcomes, as
//! [`compare_trajectories`] reads them; and `.events.json`, every event with its exact time
//! and the book before it, for fitting. It prints the digests the design records.
//!
//! `cargo run --release --example mdqr_qr -- compare <design.json> <data> <model>` prints
//! the [`DiscrepancyReport`] of a model trajectory against a data trajectory, both in the
//! `.trajectory.json` format, with the design's parameters for the coverage.
//!
//! `cargo run --release --example mdqr_qr -- parameters <design.json>` prints the design's
//! parameters as the simulator holds them: rates, invariant laws, theta and the cap.
//!
//! `cargo run --release --example mdqr_qr -- arms <fits.json> <window.json> <out dir> <w>`
//! simulates the capture's fitted queue-reactive arms for #106's main study: 10 draws per arm
//! of the window's 2 h, from the data's book before it, draw `j` at seed `106000 + 100 w + j`.
//! Its inputs and outputs are REAL DATA fits and simulations of them.
//!
//! The other commands are synthetic, from made-up parameters.

use std::fs;
use std::path::{Path, PathBuf};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::queue_reactive::{
    BookSample, DiscrepancyReport, MAX_LIMITS, QrEventKind, QrParameters, QrSide, QrSimulator,
    QrState, QrStep, QueueObservation, ReferenceObservation, Trajectory, UnsupportedPolicy,
    compare_trajectories,
};
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const ARM_LIMITS: usize = 3;
const WINDOW_SECONDS: f64 = 7200.0;
const DRAWS: u64 = 10;

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

fn float(value: &Value, key: &str) -> Result<f64, Failure> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{key} is not a number").into())
}

/// The truth's parameters, from the design's `s0.truth`.
fn parameters(truth: &Value) -> Result<QrParameters, Failure> {
    let sizes = truth
        .get("event_sizes")
        .and_then(Value::as_array)
        .ok_or("event_sizes")?
        .iter()
        .map(|size| Ok(size.as_str().ok_or("event size")?.parse::<Decimal>()?))
        .collect::<Result<Vec<_>, Failure>>()?;
    let rates = truth
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
    let policy = match text(truth, "policy")? {
        "stop" => UnsupportedPolicy::Stop,
        "nearest_supported" => UnsupportedPolicy::NearestSupported,
        other => return Err(format!("unknown policy {other}").into()),
    };
    Ok(QrParameters::new(
        text(truth, "tick_size")?.parse::<Decimal>()?,
        float(truth, "theta")?,
        float(truth, "theta_reinit")?,
        sizes,
        rates,
        supported,
        policy,
    )?)
}

fn side_code(side: QrSide) -> Result<u8, Failure> {
    match side {
        QrSide::Bid => Ok(0),
        QrSide::Ask => Ok(1),
        other => Err(format!("unknown side {other:?}").into()),
    }
}

fn kind_code(kind: QrEventKind) -> Result<u8, Failure> {
    match kind {
        QrEventKind::Limit => Ok(0),
        QrEventKind::Cancel => Ok(1),
        QrEventKind::Market => Ok(2),
        other => Err(format!("unknown kind {other:?}").into()),
    }
}

fn side_of(code: u64) -> Result<QrSide, Failure> {
    match code {
        0 => Ok(QrSide::Bid),
        1 => Ok(QrSide::Ask),
        other => Err(format!("side code {other}").into()),
    }
}

fn kind_of(code: u64) -> Result<QrEventKind, Failure> {
    match code {
        0 => Ok(QrEventKind::Limit),
        1 => Ok(QrEventKind::Cancel),
        2 => Ok(QrEventKind::Market),
        other => Err(format!("kind code {other}").into()),
    }
}

/// A trajectory as JSON: `samples` `[ms, [bid], [ask], spread]`, `events`
/// `[ms, side, depth, kind, queue before, waiting seconds]`, `references`
/// `[ms, moved, reinitialized]`, with sides 0 bid and 1 ask, kinds 0 limit, 1 cancel and
/// 2 market, and `limits` queues a side.
fn trajectory_json(trajectory: &Trajectory, limits: usize) -> Result<Value, Failure> {
    let samples = trajectory
        .samples
        .iter()
        .map(|s| {
            json!([
                s.timestamp_ms,
                s.bid.iter().take(limits).collect::<Vec<_>>(),
                s.ask.iter().take(limits).collect::<Vec<_>>(),
                s.spread_ticks
            ])
        })
        .collect::<Vec<_>>();
    let events = trajectory
        .events
        .iter()
        .map(|e| {
            Ok(json!([
                e.timestamp_ms,
                side_code(e.side)?,
                e.depth,
                kind_code(e.kind)?,
                e.queue_before,
                e.waiting_seconds
            ]))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    let references = trajectory
        .references
        .iter()
        .map(|r| json!([r.timestamp_ms, r.moved, r.reinitialized]))
        .collect::<Vec<_>>();
    Ok(json!({
        "limits": limits,
        "samples": samples,
        "events": events,
        "references": references,
        "stopped": trajectory.stopped.map(|s| format!("{s:?}")),
    }))
}

/// Reads a trajectory written as [`trajectory_json`] writes it, by this study's model too.
fn read_trajectory(path: &Path) -> Result<Trajectory, Failure> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let array = |v: &Value, what: &str| -> Result<Vec<Value>, Failure> {
        Ok(v.as_array().ok_or(format!("{what} is not a list"))?.clone())
    };
    let whole = |v: &Value| {
        v.as_u64()
            .ok_or_else(|| Failure::from("not a whole number"))
    };
    let queues = |v: &Value| -> Result<[u32; MAX_LIMITS], Failure> {
        let mut out = [0u32; MAX_LIMITS];
        for (slot, q) in out.iter_mut().zip(array(v, "queues")?) {
            *slot = u32::try_from(whole(&q)?)?;
        }
        Ok(out)
    };
    let mut trajectory = Trajectory::default();
    for sample in array(value.get("samples").ok_or("samples")?, "samples")? {
        let s = array(&sample, "sample")?;
        let at = |i: usize| s.get(i).ok_or_else(|| Failure::from("short sample"));
        trajectory.samples.push(BookSample {
            timestamp_ms: whole(at(0)?)?,
            bid: queues(at(1)?)?,
            ask: queues(at(2)?)?,
            spread_ticks: u32::try_from(whole(at(3)?)?)?,
        });
    }
    for event in array(value.get("events").ok_or("events")?, "events")? {
        let e = array(&event, "event")?;
        let at = |i: usize| e.get(i).ok_or_else(|| Failure::from("short event"));
        trajectory.events.push(QueueObservation {
            timestamp_ms: whole(at(0)?)?,
            side: side_of(whole(at(1)?)?)?,
            depth: u8::try_from(whole(at(2)?)?)?,
            kind: kind_of(whole(at(3)?)?)?,
            queue_before: u32::try_from(whole(at(4)?)?)?,
            waiting_seconds: at(5)?.as_f64().ok_or("waiting time")?,
        });
    }
    for reference in array(value.get("references").ok_or("references")?, "references")? {
        let r = array(&reference, "reference")?;
        let flag = |i: usize| r.get(i).and_then(Value::as_bool).ok_or("reference flag");
        trajectory.references.push(ReferenceObservation {
            timestamp_ms: whole(r.first().ok_or("short reference")?)?,
            moved: flag(1)?,
            reinitialized: flag(2)?,
        });
    }
    Ok(trajectory)
}

/// Every event of a run with its exact time and the book before it:
/// `[seconds, side, depth, kind, [bid before], [ask before]]`, and the reference outcomes
/// `[seconds, moved, reinitialized]`, until `horizon` seconds or `max_events`.
fn event_record(
    params: &QrParameters,
    seed: u64,
    reference_ticks: i64,
    horizon: f64,
    max_events: usize,
) -> Result<Value, Failure> {
    let limits = params.limits();
    let mut simulator = QrSimulator::new(params.clone(), seed, reference_ticks)?;
    let start = simulator.state();
    let mut events = Vec::new();
    let mut references = Vec::new();
    let mut stopped = None;
    loop {
        if events.len() >= max_events {
            stopped = Some("EventLimit".to_string());
            break;
        }
        let before = simulator.state();
        let step = simulator.step()?;
        let after = simulator.state();
        match step {
            QrStep::Event {
                observation,
                reference,
            } => {
                if after.time_seconds >= horizon {
                    break;
                }
                events.push(json!([
                    after.time_seconds,
                    side_code(observation.side)?,
                    observation.depth,
                    kind_code(observation.kind)?,
                    before.bid.iter().take(limits).collect::<Vec<_>>(),
                    before.ask.iter().take(limits).collect::<Vec<_>>()
                ]));
                if let Some(r) = reference {
                    references.push(json!([after.time_seconds, r.moved, r.reinitialized]));
                }
            }
            other => {
                stopped = Some(format!("{other:?}"));
                break;
            }
        }
    }
    Ok(json!({
        "limits": limits,
        "seed": seed,
        "horizon_seconds": horizon,
        "start": {
            "bid": start.bid.iter().take(limits).collect::<Vec<_>>(),
            "ask": start.ask.iter().take(limits).collect::<Vec<_>>(),
        },
        "events": events,
        "references": references,
        "stopped": stopped,
    }))
}

fn write(path: &Path, value: &Value) -> Result<usize, Failure> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec(value)?;
    fs::write(path, &bytes)?;
    Ok(bytes.len())
}

fn truth(design: &Value, out: &Path) -> Result<(), Failure> {
    let s0 = design.get("s0").ok_or("s0")?;
    let params = parameters(s0.get("truth").ok_or("truth")?)?;
    let horizon = float(s0, "window_seconds")?;
    let sample_ms = number(s0, "sample_ms")?;
    #[allow(clippy::cast_precision_loss)] // A sample interval in milliseconds is small.
    let sample_s = sample_ms as f64 / 1_000.0;
    let reference_ticks = i64::try_from(number(s0, "reference_ticks")?)?;
    let max_events = usize::try_from(number(s0, "max_events_per_window")?)?;
    let mut records = Vec::new();
    for set in ["train", "evaluation", "truth_again"] {
        let seeds = s0
            .get("seeds")
            .and_then(|s| s.get(set))
            .and_then(Value::as_array)
            .ok_or(format!("seeds of {set}"))?;
        for (index, seed) in seeds.iter().enumerate() {
            let seed = seed.as_u64().ok_or("seed")?;
            let mut simulator = QrSimulator::new(params.clone(), seed, reference_ticks)?;
            let trajectory = simulator.run_limited(horizon, sample_s, max_events)?;
            if let Some(reason) = trajectory.stopped {
                return Err(format!("{set} window {index} stopped early: {reason:?}").into());
            }
            let base = out.join(set).join(format!("window-{}", index + 1));
            let traj_bytes = write(
                &base.with_extension("trajectory.json"),
                &trajectory_json(&trajectory, params.limits())?,
            )?;
            let record = event_record(&params, seed, reference_ticks, horizon, max_events)?;
            let recorded = record
                .get("events")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if recorded != trajectory.events.len() {
                return Err(format!(
                    "{set} window {index}: {recorded} recorded events against {} in the run",
                    trajectory.events.len()
                )
                .into());
            }
            let events_bytes = write(&base.with_extension("events.json"), &record)?;
            records.push(json!({
                "set": set,
                "window": index + 1,
                "seed": seed,
                "events": trajectory.events.len(),
                "samples": trajectory.samples.len(),
                "trajectory_bytes": traj_bytes,
                "event_record_bytes": events_bytes,
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "design_version": design["design_version"],
            "parameters_fingerprint": format!("{:032x}", params.fingerprint()),
            "windows": records,
        }))?
    );
    Ok(())
}

fn report_json(report: &DiscrepancyReport) -> Value {
    json!({
        "queue_size": report.queue_size,
        "spread": report.spread,
        "waiting_time": report.waiting_time,
        "waiting_law": report.waiting_law,
        "flow": report.flow,
        "flow_law": report.flow_law,
        "coverage": report.coverage,
        "data_samples": report.data_samples,
        "data_events": report.data_events,
        "model_samples": report.model_samples,
        "model_events": report.model_events,
        "model_stopped": report.model_stopped.map(|s| format!("{s:?}")),
        "data_invalid_waiting": report.data_invalid_waiting,
        "model_invalid_waiting": report.model_invalid_waiting,
    })
}

fn compare(design: &Value, data: &Path, model: &Path) -> Result<(), Failure> {
    let s0 = design.get("s0").ok_or("s0")?;
    let params = parameters(s0.get("truth").ok_or("truth")?)?;
    let interval = number(s0, "flow_interval_ms")?;
    let report = compare_trajectories(
        &read_trajectory(data)?,
        &read_trajectory(model)?,
        &params,
        interval,
    )?;
    println!("{}", serde_json::to_string(&report_json(&report))?);
    Ok(())
}

fn print_parameters(design: &Value) -> Result<(), Failure> {
    let s0 = design.get("s0").ok_or("s0")?;
    let params = parameters(s0.get("truth").ok_or("truth")?)?;
    let limits = params.limits();
    let rates = (0..limits)
        .map(|d| params.rates(d).map(<[_]>::to_vec).unwrap_or_default())
        .collect::<Vec<_>>();
    let invariant = (0..limits)
        .map(|d| params.invariant(d).map(<[_]>::to_vec).unwrap_or_default())
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string(&json!({
            "limits": limits,
            "max_queue": params.max_queue(),
            "theta": params.theta(),
            "theta_reinit": params.theta_reinit(),
            "rates": rates,
            "invariant": invariant,
            "fingerprint": format!("{:032x}", params.fingerprint()),
        }))?
    );
    Ok(())
}

/// The queues of a capture window's last book sample, in the simulator's form, each held to
/// `cap`, with how many were above it.
fn start_queues(side: &Value, cap: u32) -> Result<([u32; MAX_LIMITS], u32), Failure> {
    let mut out = [0; MAX_LIMITS];
    let mut clamped = 0;
    let values = side.as_array().ok_or("book side")?;
    for (slot, value) in out.iter_mut().zip(values.iter().take(ARM_LIMITS)) {
        let queue = u32::try_from(value.as_u64().ok_or("queue")?)?;
        clamped += u32::from(queue > cap);
        *slot = queue.min(cap);
    }
    Ok((out, clamped))
}

/// One draw of a fitted arm from the window's starting book.
fn draw(
    params: &QrParameters,
    sample: &Value,
    seed: u64,
) -> Result<(Trajectory, Vec<i64>, u32), Failure> {
    let cap = u32::try_from(params.max_queue())?;
    let (bid, low) = start_queues(sample.get("bid").ok_or("bid")?, cap)?;
    let (ask, high) = start_queues(sample.get("ask").ok_or("ask")?, cap)?;
    let mut state = QrSimulator::new(params.clone(), seed, 0)?.state();
    state.bid = bid;
    state.ask = ask;
    let mut simulator = QrSimulator::from_state(params.clone(), state)?;
    let trajectory = simulator.run(WINDOW_SECONDS, 1.0)?;
    let directions = directions_of(params, state, trajectory.references.len())?;
    Ok((trajectory, directions, low + high))
}

/// The sign of each of a run's reference outcomes, 0 where it did not move: the run again from
/// its starting state, which is deterministic, reading the reference it keeps.
fn directions_of(
    params: &QrParameters,
    state: QrState,
    references: usize,
) -> Result<Vec<i64>, Failure> {
    let mut simulator = QrSimulator::from_state(params.clone(), state)?;
    let mut out = Vec::with_capacity(references);
    while out.len() < references {
        let before = simulator.state().reference_ticks;
        match simulator.step()? {
            QrStep::Event { reference, .. } | QrStep::Reset { reference, .. } => {
                if reference.is_some() {
                    out.push((simulator.state().reference_ticks - before).signum());
                }
            }
            other => return Err(format!("a run ended early on replay: {other:?}").into()),
        }
    }
    Ok(out)
}

fn arms(fits: &Value, window: &Path, out: &Path, w: u64) -> Result<(), Failure> {
    let in_holdout = window
        .canonicalize()?
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "holdout");
    let claimed = out
        .parent()
        .is_some_and(|p| p.join("holdout.started").exists());
    if in_holdout && !claimed {
        return Err("the holdout is not claimed".into());
    }
    let window: Value = serde_json::from_slice(&fs::read(window)?)?;
    let sample = window
        .pointer("/history/last_sample")
        .ok_or("the window has no starting book")?;
    fs::create_dir_all(out)?;
    for arm in ["qr", "qr_count", "qr_constant"] {
        let serialized = fits
            .pointer(&format!("/arms/{arm}/parameters/serialized"))
            .ok_or("no such arm")?;
        let params: QrParameters = serde_json::from_value(serialized.clone())?;
        for j in 1..=DRAWS {
            let (trajectory, directions, clamped) = draw(&params, sample, 106_000 + 100 * w + j)?;
            let mut value = trajectory_json(&trajectory, ARM_LIMITS)?;
            value["reference_directions"] = json!(directions);
            value["clamped_start_queues"] = json!(clamped);
            write(&out.join(format!("{arm}-w{w}-d{j}.json")), &value)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or("a command: truth, compare or parameters")?;
    let design_path = PathBuf::from(args.next().ok_or("the design file")?);
    let design: Value = serde_json::from_slice(&fs::read(&design_path)?)?;
    match command.as_str() {
        "truth" => truth(
            &design,
            &PathBuf::from(args.next().ok_or("the output directory")?),
        ),
        "compare" => {
            let data = PathBuf::from(args.next().ok_or("the data trajectory")?);
            let model = PathBuf::from(args.next().ok_or("the model trajectory")?);
            compare(&design, &data, &model)
        }
        "parameters" => print_parameters(&design),
        "arms" => {
            let window = PathBuf::from(args.next().ok_or("the window")?);
            let out = PathBuf::from(args.next().ok_or("the output directory")?);
            let w = args.next().ok_or("the window number")?.parse()?;
            arms(&design, &window, &out, w)
        }
        other => Err(format!("unknown command {other}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design() -> Value {
        json!({
            "design_version": 1,
            "s0": {
                "truth": {
                    "tick_size": "0.01", "theta": 0.6, "theta_reinit": 0.2,
                    "event_sizes": ["1", "2"], "policy": "nearest_supported",
                    "rates": [
                        [[4.0, 0.0, 0.0], [4.0, 1.0, 1.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0], [0.0, 4.0, 1.0]],
                        [[4.0, 0.0, 0.0], [4.0, 1.0, 1.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0], [0.0, 4.0, 1.0]]
                    ]
                },
                "window_seconds": 600.0, "sample_ms": 1000, "reference_ticks": 10000,
                "max_events_per_window": 100000, "flow_interval_ms": 1000,
                "seeds": {"train": [11], "evaluation": [12], "truth_again": [13]}
            }
        })
    }

    #[test]
    fn test_a_starting_book_is_held_to_the_cap_and_counted() {
        let (queues, clamped) = start_queues(&json!([3, 250, 7, 9]), 200).unwrap();
        assert_eq!((&queues[..4], clamped), (&[3, 200, 7, 0][..], 1));
    }

    #[test]
    fn test_a_trajectory_round_trips_and_compares_as_zero_with_itself() {
        let params = parameters(&design()["s0"]["truth"]).unwrap();
        let trajectory = QrSimulator::new(params.clone(), 7, 10_000)
            .unwrap()
            .run(600.0, 1.0)
            .unwrap();
        let dir = std::env::temp_dir().join(format!("mdqr-qr-{}", std::process::id()));
        let path = dir.join("t.trajectory.json");
        write(
            &path,
            &trajectory_json(&trajectory, params.limits()).unwrap(),
        )
        .unwrap();
        let read = read_trajectory(&path).unwrap();
        assert_eq!(read.samples, trajectory.samples);
        for (i, (a, b)) in read.events.iter().zip(&trajectory.events).enumerate() {
            assert_eq!(a, b, "event {i}");
        }
        assert_eq!(read.events.len(), trajectory.events.len());
        assert_eq!(read.references, trajectory.references);
        let report = compare_trajectories(&read, &trajectory, &params, 1000).unwrap();
        assert!(report.queue_size.iter().all(|tv| *tv == 0.0));
        assert!(report.flow_law.iter().all(|tv| *tv == 0.0));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_the_event_record_is_the_run_event_for_event() {
        let params = parameters(&design()["s0"]["truth"]).unwrap();
        let trajectory = QrSimulator::new(params.clone(), 7, 10_000)
            .unwrap()
            .run(600.0, 1.0)
            .unwrap();
        let record = event_record(&params, 7, 10_000, 600.0, 100_000).unwrap();
        let events = record["events"].as_array().unwrap();
        assert_eq!(events.len(), trajectory.events.len());
        for (recorded, event) in events.iter().zip(&trajectory.events) {
            let seconds = recorded[0].as_f64().unwrap();
            assert_eq!((seconds * 1000.0).floor() as u64, event.timestamp_ms);
            assert_eq!(
                recorded[1].as_u64().unwrap(),
                u64::from(side_code(event.side).unwrap())
            );
            assert_eq!(recorded[2].as_u64().unwrap(), u64::from(event.depth));
            assert_eq!(
                recorded[3].as_u64().unwrap(),
                u64::from(kind_code(event.kind).unwrap())
            );
            let side = if event.side == QrSide::Bid { 4 } else { 5 };
            let depth = usize::from(event.depth);
            assert_eq!(
                recorded[side][depth].as_u64().unwrap(),
                u64::from(event.queue_before)
            );
        }
        assert_eq!(
            record["references"].as_array().unwrap().len(),
            trajectory.references.len()
        );
    }
}
