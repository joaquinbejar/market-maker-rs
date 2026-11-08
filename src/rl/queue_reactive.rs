//! An episode's market from the queue-reactive simulator.

use crate::Decimal;
use crate::backtest::MarketTick;
use crate::backtest::kernel::{KernelConfig, KernelDataset};
use crate::backtest::queue_reactive::{MAX_LIMITS, QrParameters, QrSimulator, QrState, QrStep};
use crate::types::error::{MMError, MMResult};

/// Most rows of one episode, the simulator's own limit on a grid.
pub const MAX_QR_ROWS: u32 = 1_000_000;

/// Most reference moves one [`ReactiveQrMarket::react`] imposes.
pub const MAX_REACTION_MOVES: u64 = 10_000;

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

/// An episode's market that reacts to the agent from outside the simulated flow (#282).
///
/// It holds the rows [`qr_market`] makes and, for each row, the simulator's state the next
/// row is simulated from: the one the row shows, unless a reaction changed it.
/// [`ReactiveQrMarket::react`] imposes reference moves after a row, as a trade's permanent
/// impact, and simulates every later row again with the simulator's own generator; that row
/// and the rows before it do not change. With no reaction its market is [`qr_market`]'s.
/// What causes a reaction, and how many moves, is the caller's.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveQrMarket {
    params: QrParameters,
    episode: QrEpisode,
    config: KernelConfig,
    ticks: Vec<MarketTick>,
    states: Vec<QrState>,
    stopped: Option<QrStep>,
}

impl ReactiveQrMarket {
    /// Simulates the episode as [`qr_market`] does, keeping each row's state.
    ///
    /// # Errors
    ///
    /// As [`qr_market`].
    pub fn new(
        params: &QrParameters,
        episode: &QrEpisode,
        config: &KernelConfig,
    ) -> MMResult<Self> {
        if episode.sample_ms == 0 || episode.rows == 0 || episode.rows > MAX_QR_ROWS {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid episode: {} rows, {} ms apart",
                episode.rows, episode.sample_ms
            )));
        }
        let doubled = config
            .tick_size
            .checked_mul(Decimal::TWO)
            .filter(|doubled| *doubled == params.tick_size());
        if doubled.is_none() {
            return Err(MMError::InvalidConfiguration(format!(
                "the kernel's tick {} must be half the simulator's {}",
                config.tick_size,
                params.tick_size()
            )));
        }
        let simulator = QrSimulator::new(params.clone(), episode.seed, episode.reference_ticks)?;
        let capacity = usize::try_from(episode.rows).unwrap_or(0);
        let mut market = Self {
            params: params.clone(),
            episode: *episode,
            config: config.clone(),
            ticks: Vec::with_capacity(capacity),
            states: Vec::with_capacity(capacity),
            stopped: None,
        };
        market.push(&simulator.state(), episode.start_ms)?;
        market.extend(simulator, episode.start_ms)?;
        Ok(market)
    }

    fn push(&mut self, state: &QrState, timestamp_ms: u64) -> MMResult<()> {
        self.ticks.push(row_of(
            state,
            &self.params,
            self.config.tick_size,
            timestamp_ms,
        )?);
        self.states.push(*state);
        Ok(())
    }

    /// Simulates the rows after the last one, from `simulator` at that row's time
    /// `timestamp_ms`.
    fn extend(&mut self, mut simulator: QrSimulator, mut timestamp_ms: u64) -> MMResult<()> {
        let seconds = f64::from(self.episode.sample_ms) / 1000.0;
        let rows = usize::try_from(self.episode.rows).unwrap_or(usize::MAX);
        while self.ticks.len() < rows {
            let run = simulator.run(seconds, seconds)?;
            if let Some(reason) = run.stopped {
                self.stopped = Some(reason);
                return Ok(());
            }
            timestamp_ms = timestamp_ms
                .checked_add(u64::from(self.episode.sample_ms))
                .ok_or_else(|| {
                    MMError::InvalidConfiguration("the episode's time overflows u64".to_string())
                })?;
            self.push(&simulator.state(), timestamp_ms)?;
        }
        Ok(())
    }

    /// Imposes `moves` reference moves after row `row`, up for a positive count and down
    /// for a negative one, each through [`QrSimulator::impose_reference_move`] on the state
    /// the next row is simulated from, then simulates every later row again from it. Row
    /// `row` itself and the rows before it do not change; a second reaction after the same
    /// row adds to the first. No moves change nothing. Returns the volume the moves emptied
    /// from the first limit's queues they passed through, in the event sizes' unit: modelled
    /// consumption, not a trade of the agent.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a row past the market's last or more
    /// than [`MAX_REACTION_MOVES`] moves, and the simulator's errors.
    pub fn react(&mut self, row: usize, moves: i64) -> MMResult<Decimal> {
        if moves.unsigned_abs() > MAX_REACTION_MOVES {
            return Err(MMError::InvalidConfiguration(format!(
                "{moves} moves, above the {MAX_REACTION_MOVES} a reaction imposes"
            )));
        }
        let state = *self.states.get(row).ok_or_else(|| {
            MMError::InvalidConfiguration(format!(
                "row {row} is past the market's {} rows",
                self.states.len()
            ))
        })?;
        if moves == 0 {
            return Ok(Decimal::ZERO);
        }
        let timestamp_ms = self
            .ticks
            .get(row)
            .map(|tick| tick.timestamp)
            .ok_or_else(|| MMError::InvalidConfiguration(format!("row {row} has no time")))?;
        let mut simulator = QrSimulator::from_state(self.params.clone(), state)?;
        let size = self
            .params
            .event_sizes()
            .first()
            .copied()
            .ok_or_else(|| MMError::InvalidConfiguration("no event size".to_string()))?;
        let mut emptied = Decimal::ZERO;
        for _ in 0..moves.unsigned_abs() {
            let touch = simulator.state();
            let queue = if moves > 0 {
                touch.ask[0]
            } else {
                touch.bid[0]
            };
            emptied = Decimal::from(queue)
                .checked_mul(size)
                .and_then(|volume| emptied.checked_add(volume))
                .ok_or_else(|| MMError::NumericalError("emptied volume".to_string()))?;
            simulator.impose_reference_move(moves > 0)?;
        }
        let next = row.saturating_add(1);
        self.ticks.truncate(next);
        self.states.truncate(next);
        if let Some(slot) = self.states.get_mut(row) {
            *slot = simulator.state();
        }
        self.stopped = None;
        self.extend(simulator, timestamp_ms)?;
        Ok(emptied)
    }

    /// The rows, as [`MarketTick`]s.
    #[must_use]
    pub fn ticks(&self) -> &[MarketTick] {
        &self.ticks
    }

    /// For each row, the simulator's state the next row is simulated from.
    #[must_use]
    pub fn states(&self) -> &[QrState] {
        &self.states
    }

    /// Why the simulator stopped before the last row asked for, if it did.
    #[must_use]
    pub fn stopped(&self) -> Option<QrStep> {
        self.stopped
    }

    /// The market as [`qr_market`] returns it: the same dataset and digest for the same
    /// rows, and as its end the state after the last row; where the simulator stopped, if
    /// it stopped early, is not kept.
    ///
    /// # Errors
    ///
    /// As [`qr_market`], for a row the kernel refuses.
    pub fn market(&self) -> MMResult<QrMarket> {
        let mut external = [0_u8; 32];
        let parts = self
            .params
            .fingerprint()
            .to_le_bytes()
            .into_iter()
            .chain(self.episode.seed.to_le_bytes())
            .chain(self.episode.reference_ticks.to_le_bytes());
        for (slot, byte) in external.iter_mut().zip(parts) {
            *slot = byte;
        }
        let end = self
            .states
            .last()
            .copied()
            .ok_or_else(|| MMError::InvalidMarketState("a market with no rows".to_string()))?;
        Ok(QrMarket {
            dataset: KernelDataset::from_ticks(&self.ticks, &self.config)?
                .with_external_digest(external),
            stopped: self.stopped,
            end,
        })
    }
}
