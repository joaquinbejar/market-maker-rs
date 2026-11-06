//! One environment on its own thread.

use std::str::FromStr;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionMask, KernelConfig, KernelDataset, KernelFillModel, KernelLatency, KernelTick,
};
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    FeatureVector, InvalidActionPolicy, ObservationScales, RlAction, RlConfig, RlEnv, SpecId,
    TerminalValuation, Transition,
};
use serde::Deserialize;

/// Capacities of the kernel in the bridge. [`Worker::spawn`] refuses a dataset and a
/// latency that need more.
pub const ORD: usize = 64;
/// See [`ORD`].
pub const MSG: usize = 256;
/// See [`ORD`].
pub const DLV: usize = 1024;
/// See [`ORD`].
pub const OBS: usize = 128;

/// Stack of the worker thread: the kernel's state is a large `Copy` value.
const STACK_BYTES: usize = 16 * 1024 * 1024;

/// The versions of the spaces this bridge was written for: observation, action, reward and
/// episode. They are literals, so that a new version in the library is refused here until
/// the bridge has been checked against it.
pub const BUILT_FOR: (u16, u16, u16, u16) = (1, 1, 1, 1);

fn known_versions(id: &SpecId) -> bool {
    (id.observation, id.action, id.reward, id.episode) == BUILT_FOR
}

/// Why the bridge refused or stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// The configuration, the data or the specification was refused. Nothing ran.
    Refused(String),
    /// A step failed. The environment takes no more requests.
    Failed(String),
    /// The environment was closed, or failed earlier.
    Closed,
    /// The worker panicked. The environment takes no more requests.
    Panicked(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(reason) => write!(f, "refused: {reason}"),
            Self::Failed(reason) => write!(f, "failed: {reason}"),
            Self::Closed => write!(f, "the environment is closed"),
            Self::Panicked(reason) => write!(f, "the worker panicked: {reason}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Latency in the configuration, in quanta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyConfig {
    /// Length of a quantum, in nanoseconds.
    pub quantum_ns: u64,
    /// Observation delay.
    pub observation_q: u64,
    /// Submission delay.
    pub submit_q: u64,
    /// Cancel delay.
    pub cancel_q: u64,
    /// Acknowledgement delay.
    pub ack_q: u64,
    /// Largest jitter per delay.
    pub jitter_q: u64,
}

/// The fill model in the configuration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FillConfig {
    /// A crossing order fills whole.
    Immediate,
    /// The kernel's queue position model.
    QueuePosition {
        /// Lots cleared per millisecond, as a decimal string.
        rate: String,
        /// Milliseconds in the queue before any fill.
        min_queue_ms: u64,
    },
}

/// Hard limits in the configuration, as decimal strings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitsConfig {
    /// Largest absolute position.
    pub max_position: String,
    /// Largest notional.
    pub max_notional: String,
    /// The limits' scaling factor.
    pub scaling_factor: String,
}

/// The normalizing constants, as decimal strings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalesConfig {
    /// Divisor of the spread, in ticks.
    pub spread_ticks: String,
    /// Divisor of the mid change, in ticks.
    pub mid_change_ticks: String,
    /// Divisor of the row gap, in milliseconds.
    pub row_gap_ms: String,
    /// Divisor of the touch sizes, in lots.
    pub size_lots: String,
    /// Divisor of the position, in base units.
    pub position: String,
    /// Divisor of the entry gap, in quote per base.
    pub entry_gap: String,
}

/// What an action outside the mask does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyConfig {
    /// An error.
    Mask,
    /// A charged hold.
    Reject,
}

/// What values the last inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalConfig {
    /// At the last mid.
    Mark,
    /// Closed at the touch.
    Close,
}

/// Everything an environment is built from. Money is in decimal strings, so no float
/// crosses the boundary on its way to the ledger.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeConfig {
    /// Price grid step.
    pub tick_size: String,
    /// Quantity grid step.
    pub lot_size: String,
    /// Largest price.
    pub max_price: String,
    /// Largest order size.
    pub max_order_size: String,
    /// Decisions before truncation.
    pub max_steps: u64,
    /// Fee rate in the quote asset; any sign.
    pub fee_rate: String,
    /// Capital at the start.
    pub initial_capital: String,
    /// Latency.
    pub latency: LatencyConfig,
    /// Fill model.
    pub fill_model: FillConfig,
    /// Hard limits, if any.
    pub limits: Option<LimitsConfig>,
    /// Bid offsets, in ticks.
    pub bid_offsets: Vec<u16>,
    /// Ask offsets, in ticks; as many as the bid offsets.
    pub ask_offsets: Vec<u16>,
    /// Sizes, in lots.
    pub sizes: Vec<u64>,
    /// The normalizing constants.
    pub scales: ScalesConfig,
    /// Unit of the `f32` reward.
    pub f32_unit: String,
    /// What an action outside the mask does.
    pub invalid_action: PolicyConfig,
    /// Its charge.
    pub invalid_action_penalty: String,
    /// What values the last inventory.
    pub terminal: TerminalConfig,
}

fn decimal(what: &str, text: &str) -> Result<Decimal, BridgeError> {
    Decimal::from_str(text)
        .map_err(|error| BridgeError::Refused(format!("{what} {text:?}: {error}")))
}

impl BridgeConfig {
    /// Reads a configuration from JSON.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::Refused`] for invalid JSON or an unknown field.
    pub fn from_json(text: &str) -> Result<Self, BridgeError> {
        serde_json::from_str(text).map_err(|error| BridgeError::Refused(error.to_string()))
    }

    fn kernel(&self) -> Result<KernelConfig, BridgeError> {
        let l = self.latency;
        let latency = KernelLatency::zero(l.quantum_ns)
            .with_delays(l.observation_q, l.submit_q, l.cancel_q, l.ack_q)
            .with_jitter(l.jitter_q);
        let fill = match &self.fill_model {
            FillConfig::Immediate => KernelFillModel::Immediate,
            FillConfig::QueuePosition { rate, min_queue_ms } => KernelFillModel::QueuePosition {
                rate: decimal("fill rate", rate)?,
                min_queue_ms: *min_queue_ms,
            },
        };
        let mut kernel = KernelConfig::new(
            decimal("tick_size", &self.tick_size)?,
            decimal("lot_size", &self.lot_size)?,
            decimal("max_price", &self.max_price)?,
            decimal("max_order_size", &self.max_order_size)?,
            self.max_steps,
        )
        .with_fee_rate(decimal("fee_rate", &self.fee_rate)?)
        .with_initial_capital(decimal("initial_capital", &self.initial_capital)?)
        .with_latency(latency)
        .with_fill_model(fill);
        if let Some(limits) = &self.limits {
            let limits = RiskLimits::new(
                decimal("max_position", &limits.max_position)?,
                decimal("max_notional", &limits.max_notional)?,
                decimal("scaling_factor", &limits.scaling_factor)?,
            )
            .map_err(|error| BridgeError::Refused(error.to_string()))?;
            kernel = kernel.with_limits(limits);
        }
        Ok(kernel)
    }

    fn rl(&self) -> Result<RlConfig, BridgeError> {
        let s = &self.scales;
        let scales = ObservationScales {
            spread_ticks: decimal("spread_ticks", &s.spread_ticks)?,
            mid_change_ticks: decimal("mid_change_ticks", &s.mid_change_ticks)?,
            row_gap_ms: decimal("row_gap_ms", &s.row_gap_ms)?,
            size_lots: decimal("size_lots", &s.size_lots)?,
            position: decimal("position", &s.position)?,
            entry_gap: decimal("entry_gap", &s.entry_gap)?,
        };
        let policy = match self.invalid_action {
            PolicyConfig::Mask => InvalidActionPolicy::Mask,
            PolicyConfig::Reject => InvalidActionPolicy::Reject,
        };
        let terminal = match self.terminal {
            TerminalConfig::Mark => TerminalValuation::MarkAtLastMid,
            TerminalConfig::Close => TerminalValuation::CloseAtTouch,
        };
        Ok(RlConfig::new(scales, decimal("f32_unit", &self.f32_unit)?)
            .with_invalid_action(
                policy,
                decimal("invalid_action_penalty", &self.invalid_action_penalty)?,
            )
            .with_terminal(terminal))
    }
}

/// What a running environment is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvInfo {
    /// The specification's identity.
    pub spec: SpecId,
    /// Length of the feature vector.
    pub features: usize,
    /// Offsets per side.
    pub offsets: usize,
    /// Sizes.
    pub sizes: usize,
    /// Fingerprint of the kernel's configuration.
    pub kernel_fingerprint: u128,
    /// Digest of the dataset.
    pub dataset_digest: u128,
    /// Rows of the dataset.
    pub rows: u64,
}

/// The observation at a reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResetReply {
    /// What the agent sees.
    pub observation: FeatureVector,
    /// What it may do.
    pub mask: ActionMask,
    /// The data has no decision at all.
    pub done: bool,
}

/// The result of a step, with the running totals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReply {
    /// The environment's transition.
    pub transition: Transition,
    /// [`market_maker_rs::rl::RlState::net`] after the step.
    pub net: Decimal,
    /// [`market_maker_rs::rl::RlState::penalties`] after the step.
    pub penalties: Decimal,
}

enum Request {
    Reset(u64),
    Step(RlAction),
    #[cfg(test)]
    Panic,
}

enum Reply {
    Reset(Result<ResetReply, String>),
    Step(Result<StepReply, String>),
}

/// What a worker thread serves.
struct Job<'a> {
    kernel: &'a KernelConfig,
    data: &'a KernelDataset,
    config: &'a BridgeConfig,
    expected: Option<SpecId>,
    ready: mpsc::Sender<Result<EnvInfo, BridgeError>>,
    requests: mpsc::Receiver<Request>,
    replies: mpsc::Sender<Reply>,
}

/// Builds the environment and answers requests. The environment borrows the data here,
/// and goes before the data does.
fn serve<const K: usize, const M: usize>(job: Job<'_>) {
    let Job {
        kernel: kernel_config,
        data,
        config,
        expected,
        ready,
        requests,
        replies,
    } = job;
    let built = (|| {
        let array = |what: &str, len: usize| {
            BridgeError::Refused(format!("{what} has {len} entries, not the grid's"))
        };
        let bid: [u16; K] = config
            .bid_offsets
            .clone()
            .try_into()
            .map_err(|_| array("bid_offsets", config.bid_offsets.len()))?;
        let ask: [u16; K] = config
            .ask_offsets
            .clone()
            .try_into()
            .map_err(|_| array("ask_offsets", config.ask_offsets.len()))?;
        let sizes: [u64; M] = config
            .sizes
            .clone()
            .try_into()
            .map_err(|_| array("sizes", config.sizes.len()))?;
        let refuse =
            |error: market_maker_rs::types::error::MMError| BridgeError::Refused(error.to_string());
        let grid = ActionGrid::new(bid, ask, sizes).map_err(refuse)?;
        let env: RlEnv<'_, K, M, ORD, MSG, DLV, OBS> =
            RlEnv::new(kernel_config, data, grid, config.rl()?).map_err(refuse)?;
        let spec = env.spec();
        let id = spec.id();
        if !known_versions(&id) {
            return Err(BridgeError::Refused(format!("unknown versions in {id:?}")));
        }
        if let Some(expected) = expected {
            spec.ensure(expected).map_err(refuse)?;
        }
        let shape = spec.shape();
        let info = EnvInfo {
            spec: id,
            features: shape.features,
            offsets: shape.bid_offsets,
            sizes: shape.sizes,
            kernel_fingerprint: env.kernel().config_fingerprint(),
            dataset_digest: data.id().digest,
            rows: data.id().len,
        };
        Ok((env, info))
    })();
    let env = match built {
        Ok((env, info)) => {
            if ready.send(Ok(info)).is_err() {
                return;
            }
            env
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut state = None;
    for request in requests {
        let reply = match request {
            #[cfg(test)]
            Request::Panic => panic!("a panic asked for by a test"),
            Request::Reset(seed) => Reply::Reset(
                env.reset(seed)
                    .map_err(|e| e.to_string())
                    .and_then(|fresh| {
                        let observation = env.observe(&fresh).map_err(|e| e.to_string())?;
                        let reply = ResetReply {
                            observation,
                            mask: env.mask(&fresh),
                            done: fresh.is_done(),
                        };
                        state = Some(fresh);
                        Ok(reply)
                    }),
            ),
            Request::Step(action) => Reply::Step(match state.as_mut() {
                None => Err("a step before any reset".to_string()),
                Some(current) => env
                    .step(current, action)
                    .map(|transition| StepReply {
                        transition,
                        net: current.net(),
                        penalties: current.penalties(),
                    })
                    .map_err(|e| e.to_string()),
            }),
        };
        if replies.send(reply).is_err() {
            return;
        }
    }
}

macro_rules! dispatch {
    ($job:ident; $(($k:literal, $($m:literal),*)),* $(,)?) => {
        match ($job.config.bid_offsets.len(), $job.config.sizes.len()) {
            $($(($k, $m) => serve::<$k, $m>($job),)*)*
            (offsets, sizes) => {
                let _ = $job.ready.send(Err(BridgeError::Refused(format!(
                    "a grid of {offsets} offsets and {sizes} sizes is not built in"
                ))));
            }
        }
    };
}

/// One environment on its own thread. Dropping it closes it.
#[derive(Debug)]
pub struct Worker {
    requests: Option<mpsc::Sender<Request>>,
    replies: mpsc::Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
    info: EnvInfo,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Request")
    }
}

impl std::fmt::Debug for Reply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Reply")
    }
}

impl Worker {
    /// Starts an environment. Its thread owns the rows and the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::Refused`] for anything the kernel or the environment refuses,
    /// or when `expected` is not this specification, before any step can be taken, and with
    /// the thread gone.
    pub fn spawn(
        config: BridgeConfig,
        rows: Vec<KernelTick>,
        expected: Option<SpecId>,
    ) -> Result<Self, BridgeError> {
        Self::spawn_holding(config, rows, expected, ())
    }

    /// As [`Worker::spawn`], with `guard` living exactly as long as the thread.
    fn spawn_holding<G: Send + 'static>(
        config: BridgeConfig,
        rows: Vec<KernelTick>,
        expected: Option<SpecId>,
        guard: G,
    ) -> Result<Self, BridgeError> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (request_tx, request_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("mm-rl-worker".to_string())
            .stack_size(STACK_BYTES)
            .spawn(move || {
                let _guard = guard;
                let prepared = config.kernel().and_then(|kernel| {
                    KernelDataset::from_kernel_ticks(rows, &kernel)
                        .map(|data| (kernel, data))
                        .map_err(|error| BridgeError::Refused(error.to_string()))
                });
                let (kernel, data) = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let job = Job {
                    kernel: &kernel,
                    data: &data,
                    config: &config,
                    expected,
                    ready: ready_tx,
                    requests: request_rx,
                    replies: reply_tx,
                };
                dispatch!(
                    job;
                    (1, 1, 2, 3, 4),
                    (2, 1, 2, 3, 4),
                    (3, 1, 2, 3, 4),
                    (4, 1, 2, 3, 4),
                    (5, 1, 2, 3, 4),
                    (8, 1, 2, 3, 4),
                );
            })
            .map_err(|error| BridgeError::Refused(format!("no worker thread: {error}")))?;
        let ready = ready_rx.recv();
        match ready {
            Ok(Ok(info)) => Ok(Self {
                requests: Some(request_tx),
                replies: reply_rx,
                handle: Some(handle),
                info,
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(error)
            }
            Err(_) => Err(match handle.join() {
                Err(payload) => BridgeError::Panicked(panic_text(payload.as_ref())),
                Ok(()) => BridgeError::Closed,
            }),
        }
    }

    /// What this environment is.
    #[must_use]
    pub fn info(&self) -> EnvInfo {
        self.info
    }

    /// Starts an episode, with `seed` for the latency stream.
    ///
    /// # Errors
    ///
    /// As [`Worker::step`].
    pub fn reset(&mut self, seed: u64) -> Result<ResetReply, BridgeError> {
        match self.call(Request::Reset(seed))? {
            Reply::Reset(Ok(reply)) => Ok(reply),
            Reply::Reset(Err(reason)) | Reply::Step(Err(reason)) => Err(self.fail(reason)),
            Reply::Step(Ok(_)) => Err(self.fail("a step's reply to a reset".to_string())),
        }
    }

    /// Takes one decision.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::Failed`] for any error of the environment, an action outside
    /// the mask under the mask policy included, [`BridgeError::Panicked`] after a panic and
    /// [`BridgeError::Closed`] afterwards, as nothing continues after an error.
    pub fn step(&mut self, action: RlAction) -> Result<StepReply, BridgeError> {
        match self.call(Request::Step(action))? {
            Reply::Step(Ok(reply)) => Ok(reply),
            Reply::Reset(Err(reason)) | Reply::Step(Err(reason)) => Err(self.fail(reason)),
            Reply::Reset(Ok(_)) => Err(self.fail("a reset's reply to a step".to_string())),
        }
    }

    /// Ends the thread and frees the dataset. Calling it again changes nothing.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::Panicked`] if the worker had panicked.
    pub fn close(&mut self) -> Result<(), BridgeError> {
        // With no sender left, the worker's loop ends.
        self.requests = None;
        match self.handle.take() {
            Some(handle) => handle
                .join()
                .map_err(|payload| BridgeError::Panicked(panic_text(payload.as_ref()))),
            None => Ok(()),
        }
    }

    /// Returns true once the worker has ended.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Makes the worker panic, for the tests of what follows a panic.
    ///
    /// # Errors
    ///
    /// As [`Worker::step`].
    #[cfg(test)]
    pub fn trigger_panic(&mut self) -> Result<(), BridgeError> {
        self.call(Request::Panic).map(|_| ())
    }

    fn call(&mut self, request: Request) -> Result<Reply, BridgeError> {
        let sent = self
            .requests
            .as_ref()
            .ok_or(BridgeError::Closed)?
            .send(request);
        if sent.is_ok()
            && let Ok(reply) = self.replies.recv()
        {
            return Ok(reply);
        }
        match self.handle.take().map(JoinHandle::join) {
            Some(Err(payload)) => Err(BridgeError::Panicked(panic_text(payload.as_ref()))),
            Some(Ok(())) | None => Err(BridgeError::Closed),
        }
    }

    /// No request follows an error: the thread ends and the data is freed.
    fn fail(&mut self, reason: String) -> BridgeError {
        let _ = self.close();
        BridgeError::Failed(reason)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(ToString::to_string)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string())
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
