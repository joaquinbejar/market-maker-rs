//! An episode's market from the queue-reactive simulator.

use crate::Decimal;
use crate::backtest::MarketTick;
use crate::backtest::kernel::{KernelConfig, KernelDataset};
use crate::backtest::queue_reactive::{MAX_LIMITS, QrParameters, QrSimulator, QrState, QrStep};
use crate::types::error::{MMError, MMResult};

/// Most rows of one episode, the simulator's own limit on a grid.
pub const MAX_QR_ROWS: u32 = 1_000_000;

/// What to simulate for one episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QrEpisode {
    /// Seed of the simulator. It decides the market, and the kernel's seed decides latency.
    pub seed: u64,
    /// Reference price at the start, in the simulator's ticks.
    pub reference_ticks: i64,
    /// Time of the first row, in milliseconds.
    pub start_ms: u64,
    /// Time between rows, in milliseconds; positive.
    pub sample_ms: u32,
    /// Rows asked for, at most [`MAX_QR_ROWS`]; positive.
    pub rows: u32,
}

/// The market of one episode.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct QrMarket {
    /// The rows. Their identity's `external` digest holds the fingerprint of the
    /// parameters, the seed and the reference price.
    pub dataset: KernelDataset,
    /// Why the simulator stopped before the last row asked for, if it did. The rows
    /// end there and are not padded.
    pub stopped: Option<QrStep>,
    /// The simulator's state at the end: at the last row's time or, when it stopped early,
    /// where it stopped, which holds the events after that row. It can be stored and continued.
    pub end: QrState,
}

/// Offset of the first queue with volume, or one limit past the last if none has any.
fn best_offset(queues: &[u32; MAX_LIMITS], limits: usize) -> usize {
    queues.iter().position(|queue| *queue > 0).unwrap_or(limits)
}

/// The row of a state: limit `d` is at `d + 1/2` of the simulator's ticks from the
/// reference, and a side with no volume has its price one limit further and no size.
fn row_of(
    state: &QrState,
    params: &QrParameters,
    half_tick: Decimal,
    timestamp_ms: u64,
) -> MMResult<MarketTick> {
    let fail = || MMError::NumericalError("a simulated row does not convert".to_string());
    let limits = params.limits();
    let side = |queues: &[u32; MAX_LIMITS], sign: i128| -> MMResult<(Decimal, Decimal)> {
        let offset = best_offset(queues, limits);
        let queue = queues.get(offset).copied().filter(|_| offset < limits);
        let size = match queue {
            Some(queue) => params
                .event_sizes()
                .get(offset)
                .and_then(|unit| unit.checked_mul(Decimal::from(queue)))
                .ok_or_else(fail)?,
            None => Decimal::ZERO,
        };
        let halves = i128::from(state.reference_ticks)
            .checked_mul(2)
            .and_then(|reference| {
                let away = i128::try_from(offset)
                    .ok()?
                    .checked_mul(2)?
                    .checked_add(1)?;
                reference.checked_add(sign.checked_mul(away)?)
            })
            .ok_or_else(fail)?;
        let price = Decimal::try_from_i128_with_scale(halves, 0)
            .ok()
            .and_then(|halves| halves.checked_mul(half_tick))
            .ok_or_else(fail)?;
        Ok((price, size))
    };
    let (bid, bid_size) = side(&state.bid, -1)?;
    let (ask, ask_size) = side(&state.ask, 1)?;
    Ok(MarketTick::new(timestamp_ms, bid, bid_size, ask, ask_size))
}

/// Simulates one episode's market, for [`RlEnv::new`](super::RlEnv::new).
///
/// The first row is the book at the start, and each later row is the book a sample
/// later. The result depends only on the parameters, the episode and the build, as
/// the simulator's own trajectories do.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] when `sample_ms` or `rows` is out of range,
/// or when the kernel's tick is not half the simulator's, which is what puts limits at
/// half ticks on its grid. Returns the simulator's errors, and
/// [`MMError::InvalidMarketState`] for a row the kernel refuses: a price that is not
/// positive or above `max_price`, or a size off the lot grid.
pub fn qr_market(
    params: &QrParameters,
    episode: &QrEpisode,
    config: &KernelConfig,
) -> MMResult<QrMarket> {
    if episode.sample_ms == 0 || episode.rows == 0 || episode.rows > MAX_QR_ROWS {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid episode: {} rows, {} ms apart",
            episode.rows, episode.sample_ms
        )));
    }
    let half_tick = config.tick_size;
    let doubled = half_tick
        .checked_mul(Decimal::TWO)
        .filter(|doubled| *doubled == params.tick_size());
    if doubled.is_none() {
        return Err(MMError::InvalidConfiguration(format!(
            "the kernel's tick {} must be half the simulator's {}",
            half_tick,
            params.tick_size()
        )));
    }
    let fingerprint = params.fingerprint();
    let mut simulator = QrSimulator::new(params.clone(), episode.seed, episode.reference_ticks)?;
    let seconds = f64::from(episode.sample_ms) / 1000.0;
    let mut ticks = Vec::with_capacity(usize::try_from(episode.rows).unwrap_or(0));
    let mut stopped = None;
    let mut timestamp_ms = episode.start_ms;
    let overflow = || MMError::InvalidConfiguration("the episode's time overflows u64".to_string());
    ticks.push(row_of(&simulator.state(), params, half_tick, timestamp_ms)?);
    for _ in 1..episode.rows {
        let run = simulator.run(seconds, seconds)?;
        if let Some(reason) = run.stopped {
            stopped = Some(reason);
            break;
        }
        timestamp_ms = timestamp_ms
            .checked_add(u64::from(episode.sample_ms))
            .ok_or_else(overflow)?;
        ticks.push(row_of(&simulator.state(), params, half_tick, timestamp_ms)?);
    }
    let mut external = [0_u8; 32];
    let parts = fingerprint
        .to_le_bytes()
        .into_iter()
        .chain(episode.seed.to_le_bytes())
        .chain(episode.reference_ticks.to_le_bytes());
    for (slot, byte) in external.iter_mut().zip(parts) {
        *slot = byte;
    }
    Ok(QrMarket {
        dataset: KernelDataset::from_ticks(&ticks, config)?.with_external_digest(external),
        stopped,
        end: simulator.state(),
    })
}
