//! A stationary queue-reactive fit from the Hyperliquid capture (#290).
//!
//! #263's comparison found the fitted model transient on the capture: at every queue size its
//! fitted arrivals exceeded its departures, so simulated queues climbed to the cap. Three causes
//! are examined: exits by price move, block-tied times, and events dropped unevenly by type.
//!
//! Estimators, declared before any run:
//! - E0: `estimate_parameters` as it is, which leaves out every event without a positive wait;
//! - E1: every event counted, a zero wait adding no time;
//! - E2: per queue, each block's events collapsed into their net change, then counted as E1;
//! - E3: an explicit exit by price move while the touch queue is not empty, built only under
//!   the condition in `support/qr_290.rs`.
//!
//! The protocol, in `support/qr_290.rs`, is fixed before any run. #286 already read the holdout windows used in
//! phase 3, as the report says.
//!
//! The capture's phases run in `hyperliquid_validation`, which holds the split and #286's
//! windows. Modes here:
//! - `synthetic`: the recovery check, run first. #102's calm regime is simulated with known
//!   parameters, its events are tied into blocks whose lengths are exponential with each mean in
//!   [`BLOCK_MEANS_SECONDS`], and each estimator is fitted on them. It reports the error in the
//!   rates and the fitted chain's mean touch queue against the truth's.
//! - `synthetic-e3`: E3's recovery check. The calm regime with touch resets of known rate
//!   [`E3_SYNTHETIC_RESET_RATE`] and redraw law (a quarter on each size from 1 to 4), untied
//!   and tied at the capture's mean block length, fitted as E3. It reports the error in the
//!   reset rates, their law and the other rates, and the mean touch queue of a 2 h run.

use market_maker_rs::backtest::queue_reactive::{
    QrEstimationConfig, QrParameters, QrRecord, QrSimulator, QrStep, QueueObservation,
    ReferenceObservation, ResetObservation, UnsupportedPolicy, ZeroWaitPolicy, collapse_blocks,
    estimate_parameters, estimate_parameters_with_resets,
};
use market_maker_rs::prelude::Decimal;
use market_maker_rs::strategy::walk_forward::TimeWindow;

#[path = "support/qr_290.rs"]
mod qr_290;
use qr_290::QR_SEED;

/// The synthetic check's seeds and length: five runs of 4 hours.
const SYNTHETIC_SEEDS: [u64; 5] = [QR_SEED, QR_SEED + 1, QR_SEED + 2, QR_SEED + 3, QR_SEED + 4];
const SYNTHETIC_HOURS: f64 = 4.0;

/// Mean block lengths, in seconds; 0 leaves the events untied. Phase 1 reports the capture's
/// block lengths, and the check is rerun with them.
const BLOCK_MEANS_SECONDS: [f64; 5] = [0.0, 0.05, CAPTURE_BLOCK_MEAN_SECONDS, 0.25, 1.0];

/// The capture's mean block length on the training part, from phase 1 at 7249e46 (95.7 ms), as
/// the design said the check would be rerun with. Added after phase 1 read the training part.
const CAPTURE_BLOCK_MEAN_SECONDS: f64 = 0.0957;

/// The estimation settings of the synthetic check: #102's cap of 4, both limits.
const SYNTHETIC_MIN_EVENTS: u64 = 30;
const SYNTHETIC_MIN_REFERENCE_MOVES: u64 = 10;

/// Reference price of the simulated book, in ticks, as #102.
const REFERENCE_TICKS: i64 = 10_000;

type Failure = Box<dyn std::error::Error>;

/// #102's calm regime, as its design file states it.
fn calm() -> Result<QrParameters, Failure> {
    let table = vec![
        (4.0, 0.0, 0.0),
        (4.0, 1.0, 1.0),
        (3.0, 2.0, 1.0),
        (2.0, 3.0, 1.0),
        (0.0, 4.0, 1.0),
    ];
    Ok(QrParameters::new(
        Decimal::new(1, 2),
        0.6,
        0.2,
        vec![Decimal::ONE, Decimal::TWO],
        vec![table.clone(), table],
        vec![vec![true; 5], vec![true; 5]],
        UnsupportedPolicy::NearestSupported,
    )?)
}

/// The estimators, each a way to read one list of events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Estimator {
    E0,
    E1,
    E2,
}

impl Estimator {
    const ALL: [Self; 3] = [Self::E0, Self::E1, Self::E2];

    fn name(self) -> &'static str {
        match self {
            Self::E0 => "E0",
            Self::E1 => "E1",
            Self::E2 => "E2",
        }
    }

    fn fit(
        self,
        events: &[QueueObservation],
        references: &[ReferenceObservation],
        event_sizes: &[Decimal],
        window: &TimeWindow,
        config: QrEstimationConfig,
    ) -> Result<QrRecord, Failure> {
        let (events, zero_waits) = match self {
            Self::E0 => (events.to_vec(), ZeroWaitPolicy::Drop),
            Self::E1 => (events.to_vec(), ZeroWaitPolicy::Count),
            Self::E2 => (collapse_blocks(events), ZeroWaitPolicy::Count),
        };
        Ok(estimate_parameters(
            &events,
            references,
            event_sizes,
            window,
            &config.with_zero_waits(zero_waits),
            "synthetic",
        )?)
    }
}

/// A simulated event with its exact time.
#[derive(Debug, Clone, Copy)]
struct Timed {
    now: f64,
    observation: QueueObservation,
    reference: Option<ReferenceObservation>,
}

fn simulate(params: &QrParameters, seed: u64, horizon_seconds: f64) -> Result<Vec<Timed>, Failure> {
    let mut simulator = QrSimulator::new(params.clone(), seed, REFERENCE_TICKS)?;
    let mut out = Vec::new();
    loop {
        match simulator.step()? {
            QrStep::Event {
                observation,
                reference,
            } => {
                let now = simulator.state().time_seconds;
                if now > horizon_seconds {
                    return Ok(out);
                }
                out.push(Timed {
                    now,
                    observation,
                    reference,
                });
            }
            other => return Err(format!("the simulation ended early: {other:?}").into()),
        }
    }
}

/// E3's synthetic reset rate, per second, at every touch size from 1.
const E3_SYNTHETIC_RESET_RATE: f64 = 0.5;

/// The calm regime with E3's synthetic resets.
fn calm_with_resets() -> Result<QrParameters, Failure> {
    let calm = calm()?;
    let rows = calm.max_queue() + 1;
    let rates = (0..rows)
        .map(|n| if n == 0 { 0.0 } else { E3_SYNTHETIC_RESET_RATE })
        .collect();
    let law = (0..rows)
        .map(|n| if (1..=4).contains(&n) { 0.25 } else { 0.0 })
        .collect();
    Ok(calm.with_resets(rates, law)?)
}

/// A simulated reset with its exact time, and the move it made, if any.
#[derive(Debug, Clone, Copy)]
struct TimedReset {
    now: f64,
    reset: ResetObservation,
    reference: Option<ReferenceObservation>,
}

/// Events and resets of a run with resets.
fn simulate_with_resets(
    params: &QrParameters,
    seed: u64,
    horizon_seconds: f64,
) -> Result<(Vec<Timed>, Vec<TimedReset>), Failure> {
    let mut simulator = QrSimulator::new(params.clone(), seed, REFERENCE_TICKS)?;
    let (mut events, mut resets) = (Vec::new(), Vec::new());
    loop {
        let step = simulator.step()?;
        let now = simulator.state().time_seconds;
        if now > horizon_seconds {
            return Ok((events, resets));
        }
        match step {
            QrStep::Event {
                observation,
                reference,
            } => events.push(Timed {
                now,
                observation,
                reference,
            }),
            QrStep::Reset {
                reset, reference, ..
            } => resets.push(TimedReset {
                now,
                reset,
                reference,
            }),
            other => return Err(format!("the simulation ended early: {other:?}").into()),
        }
    }
}

/// Resets tied as [`tied`] ties events, with the moves they made.
fn tied_resets(
    resets: &[TimedReset],
    ends: &[f64],
) -> (Vec<ResetObservation>, Vec<ReferenceObservation>) {
    let at = |t: f64| if ends.is_empty() { t } else { tie(ends, t) };
    let mut out = Vec::with_capacity(resets.len());
    let mut moves = Vec::new();
    for r in resets {
        let now = at(r.now);
        let last = at(r.now - r.reset.waiting_seconds);
        out.push(ResetObservation {
            timestamp_ms: milliseconds(now),
            waiting_seconds: if ends.is_empty() {
                r.reset.waiting_seconds
            } else {
                now - last
            },
            ..r.reset
        });
        if let Some(reference) = r.reference {
            moves.push(ReferenceObservation {
                timestamp_ms: milliseconds(now),
                ..reference
            });
        }
    }
    (out, moves)
}

/// SplitMix64, for the block boundaries.
struct Mix(u64);

impl Mix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Block ends from 0 past `horizon_seconds`, with exponential lengths of mean `mean_seconds`.
fn block_ends(mean_seconds: f64, horizon_seconds: f64, seed: u64) -> Vec<f64> {
    let mut mix = Mix(seed ^ 0x2900_0000_0000_0290);
    let mut ends = vec![0.0];
    let mut t = 0.0;
    while t <= horizon_seconds {
        t += -(1.0 - mix.unit()).ln() * mean_seconds;
        ends.push(t);
    }
    ends
}

/// The end of the block holding `t`: the first end at or after it.
fn tie(ends: &[f64], t: f64) -> f64 {
    let i = ends.partition_point(|&end| end < t);
    ends.get(i).copied().unwrap_or(t)
}

fn milliseconds(seconds: f64) -> u64 {
    (seconds * 1000.0).floor() as u64
}

/// The events and moves as a capture with block times would show them: every time is its
/// block's end, and a wait runs between the ends of the blocks of the event and of the queue's
/// previous event (or the move that restarted its clock). Untied when `ends` is empty.
fn tied(events: &[Timed], ends: &[f64]) -> (Vec<QueueObservation>, Vec<ReferenceObservation>) {
    let at = |t: f64| if ends.is_empty() { t } else { tie(ends, t) };
    let mut observations = Vec::with_capacity(events.len());
    let mut references = Vec::new();
    for event in events {
        let now = at(event.now);
        let last = at(event.now - event.observation.waiting_seconds);
        observations.push(QueueObservation {
            timestamp_ms: milliseconds(now),
            waiting_seconds: if ends.is_empty() {
                event.observation.waiting_seconds
            } else {
                now - last
            },
            ..event.observation
        });
        if let Some(reference) = event.reference {
            references.push(ReferenceObservation {
                timestamp_ms: milliseconds(now),
                ..reference
            });
        }
    }
    (observations, references)
}

/// Errors of a fit against the truth at one limit.
#[derive(Debug, Clone, Copy, Default)]
struct Errors {
    /// Mean absolute error of the rates the truth has positive, relative to them.
    relative: f64,
    /// Largest absolute error of the net drift, arrivals less departures, over the sizes.
    drift: f64,
    /// Mean queue under the fitted chain's invariant law.
    mean_queue: f64,
    /// Rates the fit leaves unsupported.
    unsupported: usize,
}

fn errors(fit: &QrParameters, truth: &QrParameters, depth: usize) -> Result<Errors, Failure> {
    let got = fit.rates(depth).ok_or("no fitted rates")?;
    let want = truth.rates(depth).ok_or("no true rates")?;
    let mut out = Errors::default();
    let mut cells = 0usize;
    for (n, (g, w)) in got.iter().zip(want).enumerate() {
        if !fit.is_supported(depth, n) {
            out.unsupported += 1;
            continue;
        }
        for (gr, wr) in [(g.0, w.0), (g.1, w.1), (g.2, w.2)] {
            if wr > 0.0 {
                out.relative += (gr - wr).abs() / wr;
                cells += 1;
            }
        }
        let drift = (g.0 - g.1 - g.2) - (w.0 - w.1 - w.2);
        out.drift = out.drift.max(drift.abs());
    }
    out.relative /= cells.max(1) as f64;
    out.mean_queue = mean(fit.invariant(depth).ok_or("no invariant law")?);
    Ok(out)
}

fn mean(law: &[f64]) -> f64 {
    law.iter().enumerate().map(|(n, p)| n as f64 * p).sum()
}

/// One row of the synthetic check: a block mean, an estimator and its errors at each limit,
/// averaged over the seeds, with the fitted theta.
struct Row {
    block_mean: f64,
    estimator: &'static str,
    errors: Vec<Errors>,
    theta: f64,
    used: f64,
    excluded: f64,
}

fn synthetic_rows(hours: f64, seeds: &[u64], block_means: &[f64]) -> Result<Vec<Row>, Failure> {
    let truth = calm()?;
    let horizon = hours * 3600.0;
    let config = QrEstimationConfig::new(
        truth.tick_size(),
        u8::try_from(truth.limits())?,
        u32::try_from(truth.max_queue())?,
        SYNTHETIC_MIN_EVENTS,
        SYNTHETIC_MIN_REFERENCE_MOVES,
        UnsupportedPolicy::NearestSupported,
    );
    let runs = seeds
        .iter()
        .map(|&seed| Ok((seed, simulate(&truth, seed, horizon)?)))
        .collect::<Result<Vec<_>, Failure>>()?;
    let mut rows = Vec::new();
    for &block_mean in block_means {
        for estimator in Estimator::ALL {
            let mut sum = vec![Errors::default(); truth.limits()];
            let (mut theta, mut used, mut excluded) = (0.0, 0.0, 0.0);
            for (seed, events) in &runs {
                let ends = if block_mean > 0.0 {
                    block_ends(block_mean, horizon, *seed)
                } else {
                    Vec::new()
                };
                let (observations, references) = tied(events, &ends);
                let end = observations.last().map_or(1, |o| o.timestamp_ms + 1);
                let window = TimeWindow::new(0, end)?;
                let record = estimator.fit(
                    &observations,
                    &references,
                    truth.event_sizes(),
                    &window,
                    config,
                )?;
                let fit = record
                    .parameters()
                    .ok_or_else(|| format!("{} did not fit: {:?}", estimator.name(), record.fit))?;
                for (depth, total) in sum.iter_mut().enumerate() {
                    let e = errors(fit, &truth, depth)?;
                    total.relative += e.relative;
                    total.drift += e.drift;
                    total.mean_queue += e.mean_queue;
                    total.unsupported += e.unsupported;
                }
                theta += fit.theta();
                used += record.events as f64;
                excluded += record.excluded_events as f64;
            }
            let k = runs.len() as f64;
            for e in &mut sum {
                e.relative /= k;
                e.drift /= k;
                e.mean_queue /= k;
            }
            rows.push(Row {
                block_mean,
                estimator: estimator.name(),
                errors: sum,
                theta: theta / k,
                used: used / k,
                excluded: excluded / k,
            });
        }
    }
    Ok(rows)
}

fn synthetic() -> Result<(), Failure> {
    let truth = calm()?;
    let rows = synthetic_rows(SYNTHETIC_HOURS, &SYNTHETIC_SEEDS, &BLOCK_MEANS_SECONDS)?;
    println!(
        "## #290 synthetic recovery (#102 calm regime, cap {})\n",
        truth.max_queue()
    );
    println!(
        "Seeds {SYNTHETIC_SEEDS:?}, {SYNTHETIC_HOURS} h each; means over the seeds. True theta {}; \
         true invariant mean queue {:.3} at limit 0 and {:.3} at limit 1.\n",
        truth.theta(),
        mean(truth.invariant(0).ok_or("law")?),
        mean(truth.invariant(1).ok_or("law")?),
    );
    println!(
        "| block mean (s) | estimator | events used | excluded | rate error L0 | rate error L1 \
         | max drift error L0 | max drift error L1 | mean queue L0 | mean queue L1 | unsupported \
         | theta |"
    );
    println!("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for row in &rows {
        let e0 = row.errors.first().copied().unwrap_or_default();
        let e1 = row.errors.get(1).copied().unwrap_or_default();
        println!(
            "| {} | {} | {:.0} | {:.0} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.3} |",
            row.block_mean,
            row.estimator,
            row.used,
            row.excluded,
            e0.relative,
            e1.relative,
            e0.drift,
            e1.drift,
            e0.mean_queue,
            e1.mean_queue,
            e0.unsupported + e1.unsupported,
            row.theta,
        );
    }
    Ok(())
}

/// The mean touch queue, both sides, of a 2 h run of `params` from seed [`QR_SEED`].
fn run_mean_touch(params: &QrParameters) -> Result<f64, Failure> {
    let run = QrSimulator::new(params.clone(), QR_SEED, REFERENCE_TICKS)?.run(7_200.0, 1.0)?;
    let samples = run.samples.len().max(1) as f64;
    Ok(run
        .samples
        .iter()
        .map(|s| f64::from(s.bid[0]) + f64::from(s.ask[0]))
        .sum::<f64>()
        / (2.0 * samples))
}

/// One row of E3's check: a block mean and E3's errors, averaged over the seeds.
struct E3Row {
    block_mean: f64,
    /// Mean absolute error of the reset rates at sizes 1 to the cap.
    reset_rate: f64,
    /// Total variation between the fitted and the true redraw law.
    reset_law: f64,
    /// The other rates' errors at each limit.
    errors: Vec<Errors>,
    /// The fit's mean touch queue over a 2 h run.
    run_mean: f64,
}

fn e3_rows(hours: f64, seeds: &[u64], block_means: &[f64]) -> Result<Vec<E3Row>, Failure> {
    let truth = calm_with_resets()?;
    let horizon = hours * 3600.0;
    let config = QrEstimationConfig::new(
        truth.tick_size(),
        u8::try_from(truth.limits())?,
        u32::try_from(truth.max_queue())?,
        SYNTHETIC_MIN_EVENTS,
        SYNTHETIC_MIN_REFERENCE_MOVES,
        UnsupportedPolicy::NearestSupported,
    )
    .with_zero_waits(ZeroWaitPolicy::Count);
    let runs = seeds
        .iter()
        .map(|&seed| Ok((seed, simulate_with_resets(&truth, seed, horizon)?)))
        .collect::<Result<Vec<_>, Failure>>()?;
    let true_rates = truth.reset_rates().ok_or("rates")?;
    let true_law = truth.reset_law().ok_or("law")?;
    let mut rows = Vec::new();
    for &block_mean in block_means {
        let mut row = E3Row {
            block_mean,
            reset_rate: 0.0,
            reset_law: 0.0,
            errors: vec![Errors::default(); truth.limits()],
            run_mean: 0.0,
        };
        for (seed, (events, resets)) in &runs {
            let ends = if block_mean > 0.0 {
                block_ends(block_mean, horizon, *seed)
            } else {
                Vec::new()
            };
            let (observations, mut references) = tied(events, &ends);
            let (resets, moves) = tied_resets(resets, &ends);
            references.extend(moves);
            references.sort_by_key(|r| r.timestamp_ms);
            let end = observations
                .last()
                .map_or(0, |o| o.timestamp_ms)
                .max(resets.last().map_or(0, |r| r.timestamp_ms))
                + 1;
            let record = estimate_parameters_with_resets(
                &observations,
                &resets,
                &references,
                truth.event_sizes(),
                &TimeWindow::new(0, end)?,
                &config,
                "synthetic",
            )?;
            let fit = record
                .parameters()
                .ok_or_else(|| format!("E3 did not fit: {:?}", record.fit))?;
            let rates = fit.reset_rates().ok_or("no reset rates")?;
            let law = fit.reset_law().ok_or("no reset law")?;
            let sizes = (rates.len() - 1) as f64;
            row.reset_rate += rates
                .iter()
                .zip(true_rates)
                .skip(1)
                .map(|(g, w)| (g - w).abs())
                .sum::<f64>()
                / sizes;
            row.reset_law += law
                .iter()
                .zip(true_law)
                .map(|(g, w)| (g - w).abs())
                .sum::<f64>()
                / 2.0;
            for (depth, total) in row.errors.iter_mut().enumerate() {
                let e = errors(fit, &truth, depth)?;
                total.relative += e.relative;
                total.drift += e.drift;
                total.unsupported += e.unsupported;
            }
            row.run_mean += run_mean_touch(fit)?;
        }
        let k = runs.len() as f64;
        row.reset_rate /= k;
        row.reset_law /= k;
        row.run_mean /= k;
        for e in &mut row.errors {
            e.relative /= k;
            e.drift /= k;
        }
        rows.push(row);
    }
    Ok(rows)
}

fn synthetic_e3() -> Result<(), Failure> {
    let truth = calm_with_resets()?;
    let block_means = [0.0, CAPTURE_BLOCK_MEAN_SECONDS];
    let rows = e3_rows(SYNTHETIC_HOURS, &SYNTHETIC_SEEDS, &block_means)?;
    println!("## #290 E3 synthetic recovery (#102 calm regime with touch resets)\n");
    println!(
        "Seeds {SYNTHETIC_SEEDS:?}, {SYNTHETIC_HOURS} h each; means over the seeds. True reset \
         rate {E3_SYNTHETIC_RESET_RATE} at every size from 1, redrawn uniformly on 1 to 4; the \
         truth's mean touch queue over a 2 h run {:.3}. E3 counts zero waits.\n",
        run_mean_touch(&truth)?
    );
    println!(
        "| block mean (s) | reset rate error | reset law TV | rate error L0 | rate error L1 \
         | max drift error L0 | unsupported | run mean touch |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    for row in &rows {
        let e0 = row.errors.first().copied().unwrap_or_default();
        let e1 = row.errors.get(1).copied().unwrap_or_default();
        println!(
            "| {} | {:.4} | {:.4} | {:.3} | {:.3} | {:.3} | {} | {:.3} |",
            row.block_mean,
            row.reset_rate,
            row.reset_law,
            e0.relative,
            e1.relative,
            e0.drift,
            e0.unsupported + e1.unsupported,
            row.run_mean,
        );
    }
    Ok(())
}

fn main() -> Result<(), Failure> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("synthetic") => synthetic(),
        Some("synthetic-e3") => synthetic_e3(),
        _ => Err("usage: qr_estimation synthetic | synthetic-e3".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use market_maker_rs::backtest::queue_reactive::{QrEventKind, QrSide};

    #[test]
    fn test_a_tie_is_the_end_of_the_block_and_waits_run_between_block_ends() {
        let ends = [0.0, 1.0, 2.5, 4.0];
        assert_eq!(tie(&ends, 0.0), 0.0);
        assert_eq!(tie(&ends, 0.2), 1.0);
        assert_eq!(tie(&ends, 1.0), 1.0);
        assert_eq!(tie(&ends, 2.6), 4.0);
        let event = |now: f64, waiting_seconds: f64| Timed {
            now,
            observation: QueueObservation {
                timestamp_ms: milliseconds(now),
                side: QrSide::Bid,
                depth: 0,
                kind: QrEventKind::Limit,
                queue_before: 1,
                waiting_seconds,
            },
            reference: None,
        };
        let (same, _) = tied(&[event(0.9, 0.5)], &ends);
        assert_eq!((same[0].timestamp_ms, same[0].waiting_seconds), (1000, 0.0));
        let (across, _) = tied(&[event(2.0, 1.5)], &ends);
        assert_eq!(
            (across[0].timestamp_ms, across[0].waiting_seconds),
            (2500, 1.5)
        );
        let (untied, _) = tied(&[event(2.0, 1.5)], &[]);
        assert_eq!(
            (untied[0].timestamp_ms, untied[0].waiting_seconds),
            (2000, 1.5)
        );
    }

    #[test]
    fn test_block_lengths_have_their_mean() {
        let ends = block_ends(0.25, 10_000.0, QR_SEED);
        let n = (ends.len() - 1) as f64;
        let got = ends.last().unwrap() / n;
        assert!((got - 0.25).abs() < 0.01, "{got}");
    }

    #[test]
    fn test_untied_events_give_every_estimator_the_same_fit() {
        let rows = synthetic_rows(0.25, &[QR_SEED], &[0.0]).unwrap();
        let first = &rows[0];
        assert!(first.excluded < first.used / 100.0);
        for row in &rows[1..] {
            assert_eq!((row.used, row.excluded), (first.used, first.excluded));
            assert_eq!(row.theta, first.theta);
            for (a, b) in row.errors.iter().zip(&first.errors) {
                assert_eq!((a.relative, a.mean_queue), (b.relative, b.mean_queue));
            }
        }
    }

    #[test]
    fn test_ties_drop_events_from_e0_and_e1_keeps_them() {
        let rows = synthetic_rows(0.25, &[QR_SEED], &[0.0, 1.0]).unwrap();
        let get = |mean: f64, name: &str| {
            rows.iter()
                .find(|r| r.block_mean == mean && r.estimator == name)
                .unwrap()
        };
        let untied = get(0.0, "E0");
        let (e0, e1, e2) = (get(1.0, "E0"), get(1.0, "E1"), get(1.0, "E2"));
        assert_eq!(e1.used + e1.excluded, untied.used + untied.excluded);
        assert_eq!(e1.excluded, untied.excluded);
        assert!(e0.used < e1.used);
        assert!(e2.used + e2.excluded < e1.used + e1.excluded);
    }

    #[test]
    fn test_e3_recovers_its_resets_untied() {
        let rows = e3_rows(1.0, &[QR_SEED], &[0.0]).unwrap();
        let row = &rows[0];
        assert!(row.reset_rate < 0.1, "{}", row.reset_rate);
        assert!(row.reset_law < 0.05, "{}", row.reset_law);
    }

    #[test]
    fn test_tied_resets_keep_their_moves_and_tie_their_waits() {
        let reset = ResetObservation {
            timestamp_ms: 900,
            side: QrSide::Ask,
            queue_before: 2,
            waiting_seconds: 0.5,
            queue_after: Some(3),
        };
        let reference = ReferenceObservation {
            timestamp_ms: 900,
            moved: true,
            reinitialized: false,
        };
        let timed = [TimedReset {
            now: 0.9,
            reset,
            reference: Some(reference),
        }];
        let (resets, moves) = tied_resets(&timed, &[0.0, 1.0, 2.5]);
        assert_eq!(
            (resets[0].timestamp_ms, resets[0].waiting_seconds),
            (1000, 0.0)
        );
        assert_eq!(resets[0].queue_after, Some(3));
        assert_eq!(moves[0].timestamp_ms, 1000);
        let (untied, _) = tied_resets(&timed, &[]);
        assert_eq!(untied[0], reset);
    }

    #[test]
    fn test_the_calm_regime_is_102s() {
        let p = calm().unwrap();
        assert_eq!((p.theta(), p.theta_reinit(), p.max_queue()), (0.6, 0.2, 4));
        assert_eq!(p.event_sizes(), &[Decimal::ONE, Decimal::TWO]);
    }
}
