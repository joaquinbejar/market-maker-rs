//! One environment on its own thread.

use std::str::FromStr;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{Attribution, EvaluationInput, attribute};
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionMask, AvellanedaQuoter, GlftQuoter, KernelConfig, KernelDataset,
    KernelFillModel, KernelLatency, KernelTick, QuoteSource,
};
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    EpisodeRecorder, FeatureVector, HedgeActions, InvalidActionPolicy, ObservationScales,
    ResidualBounds, ResidualDecision, ResidualEnv, ResidualProposal, RlAction, RlConfig, RlEnv,
    RlState, SpecId, TerminalValuation, Transition,
};
use market_maker_rs::strategy::glft::GLFTConfig;
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
pub const BUILT_FOR: (u16, u16, u16, u16) = (1, 2, 1, 1);

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

/// The classical strategy a residual corrects, with numbers as decimal strings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BaselineConfig {
    /// Avellaneda-Stoikov.
    AvellanedaStoikov {
        /// Risk aversion.
        risk_aversion: String,
        /// Volatility.
        volatility: String,
        /// Order intensity.
        order_intensity: String,
        /// Session start, in milliseconds of the rows' clock.
        session_start_ms: u64,
        /// Horizon from the session start, in milliseconds.
        horizon_ms: u64,
        /// Size of both sides.
        size: String,
    },
    /// GLFT.
    Glft {
        /// Risk aversion.
        risk_aversion: String,
        /// Decay of the fill intensity.
        order_intensity_k: String,
        /// Scale of the fill intensity.
        intensity_scale_a: String,
        /// Inventory bound, in lots.
        max_inventory_lots: u32,
        /// Terminal time from the session start, in milliseconds.
        terminal_time: u64,
        /// Smallest spread.
        min_spread: String,
        /// Volatility.
        volatility: String,
        /// Session start, in milliseconds of the rows' clock.
        session_start_ms: u64,
    },
}

/// A residual policy over a classical strategy, in place of the action grid.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualConfig {
    /// The classical strategy.
    pub baseline: BaselineConfig,
    /// Largest move of a price, in ticks.
    pub price_ticks: u32,
    /// Largest change of a size, in lots.
    pub size_lots: u32,
    /// What replaces a classical size that is not positive.
    pub default_order_size: String,
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
    /// A residual policy, which then takes no action on the grid.
    #[serde(default)]
    pub residual: Option<ResidualConfig>,
    /// Whether hedge actions exist (`RlAction::Hedge`, `HedgeActions::MarketableAtTouch`).
    /// Off by default, which keeps a configuration's specification and fingerprint.
    #[serde(default)]
    pub hedge_actions: bool,
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
        if self.hedge_actions && self.residual.is_some() {
            return Err(BridgeError::Refused(
                "hedge_actions with a residual policy: a residual policy takes no hedge action"
                    .to_string(),
            ));
        }
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
        let config = RlConfig::new(scales, decimal("f32_unit", &self.f32_unit)?)
            .with_invalid_action(
                policy,
                decimal("invalid_action_penalty", &self.invalid_action_penalty)?,
            )
            .with_terminal(terminal);
        Ok(if self.hedge_actions {
            config.with_hedge_actions()
        } else {
            config
        })
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
    /// Whether hedge actions exist, as the specification states.
    pub hedge_actions: bool,
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
    /// The hedge sizes it may take, as bits over the sizes; 0 without hedge actions.
    pub hedge_mask: u64,
    /// The data has no decision at all.
    pub done: bool,
}

/// The attribution of the episode so far, reconciled with the reward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributionReply {
    /// The components, in the evaluation's definitions.
    pub attribution: Attribution,
    /// Fills, the terminal close included.
    pub fills: usize,
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
    /// How the action came to be, under a residual policy.
    pub decision: Option<ResidualDecision>,
}

enum Request {
    Reset(u64),
    Step(RlAction),
    StepResidual(ResidualProposal),
    Attribution,
    #[cfg(test)]
    Panic,
}

// Sent by value on every step: a box would allocate for each one.
#[allow(clippy::large_enum_variant)]
enum Reply {
    Reset(Result<ResetReply, String>),
    Step(Result<StepReply, String>),
    Attribution(Result<AttributionReply, String>),
}

type State = RlState<ORD, MSG, DLV, OBS>;

/// What the request loop needs from an environment.
trait Driver {
    fn reset(&mut self, seed: u64, recorder: &mut EpisodeRecorder) -> Result<State, String>;
    fn observe(&self, state: &State) -> Result<FeatureVector, String>;
    fn mask(&self, state: &State) -> ActionMask;
    fn hedge_mask(&self, state: &State) -> u64;
    fn step(
        &mut self,
        state: &mut State,
        action: RlAction,
        recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String>;
    fn step_residual(
        &mut self,
        state: &mut State,
        proposal: ResidualProposal,
        recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String>;
    fn inner_attribution(
        &self,
        state: &State,
        recorder: &EpisodeRecorder,
    ) -> Result<AttributionReply, String>;
}

fn attribution_of<const K: usize, const M: usize>(
    env: &RlEnv<'_, K, M, ORD, MSG, DLV, OBS>,
    state: &State,
    recorder: &EpisodeRecorder,
) -> Result<AttributionReply, String> {
    env.reconcile(state, recorder).map_err(|e| e.to_string())?;
    let records = env.records(state, recorder).map_err(|e| e.to_string())?;
    let input = EvaluationInput::new(records.window, &records.marks, &records.fills, Decimal::ONE)
        .map_err(|e| e.to_string())?;
    let split = attribute(&input).map_err(|e| e.to_string())?;
    Ok(AttributionReply {
        attribution: split.total,
        fills: records.fills.len(),
    })
}

impl<const K: usize, const M: usize> Driver for RlEnv<'_, K, M, ORD, MSG, DLV, OBS> {
    fn reset(&mut self, seed: u64, recorder: &mut EpisodeRecorder) -> Result<State, String> {
        self.reset_with(seed, recorder).map_err(|e| e.to_string())
    }

    fn observe(&self, state: &State) -> Result<FeatureVector, String> {
        RlEnv::observe(self, state).map_err(|e| e.to_string())
    }

    fn mask(&self, state: &State) -> ActionMask {
        RlEnv::mask(self, state)
    }

    fn hedge_mask(&self, state: &State) -> u64 {
        RlEnv::hedge_mask(self, state)
    }

    fn step(
        &mut self,
        state: &mut State,
        action: RlAction,
        recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String> {
        let transition = self
            .step_with(state, action, recorder)
            .map_err(|e| e.to_string())?;
        Ok(StepReply {
            transition,
            net: state.net(),
            penalties: state.penalties(),
            decision: None,
        })
    }

    fn step_residual(
        &mut self,
        _state: &mut State,
        _proposal: ResidualProposal,
        _recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String> {
        Err("a residual step on an environment with a grid".to_string())
    }

    fn inner_attribution(
        &self,
        state: &State,
        recorder: &EpisodeRecorder,
    ) -> Result<AttributionReply, String> {
        attribution_of(self, state, recorder)
    }
}

impl<S: QuoteSource + Clone> Driver for ResidualEnv<'_, S, ORD, MSG, DLV, OBS> {
    fn reset(&mut self, seed: u64, recorder: &mut EpisodeRecorder) -> Result<State, String> {
        self.reset_with(seed, recorder).map_err(|e| e.to_string())
    }

    fn observe(&self, state: &State) -> Result<FeatureVector, String> {
        ResidualEnv::observe(self, state).map_err(|e| e.to_string())
    }

    fn mask(&self, state: &State) -> ActionMask {
        self.env().mask(state)
    }

    fn hedge_mask(&self, state: &State) -> u64 {
        self.env().hedge_mask(state)
    }

    fn step(
        &mut self,
        _state: &mut State,
        _action: RlAction,
        _recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String> {
        Err("a grid action on a residual environment".to_string())
    }

    fn step_residual(
        &mut self,
        state: &mut State,
        proposal: ResidualProposal,
        recorder: &mut EpisodeRecorder,
    ) -> Result<StepReply, String> {
        let step = self
            .step_with(state, proposal, recorder)
            .map_err(|e| e.to_string())?;
        Ok(StepReply {
            transition: step.transition,
            net: state.net(),
            penalties: state.penalties(),
            decision: Some(step.decision),
        })
    }

    fn inner_attribution(
        &self,
        state: &State,
        recorder: &EpisodeRecorder,
    ) -> Result<AttributionReply, String> {
        attribution_of(self.env(), state, recorder)
    }
}

/// Answers requests until the sender goes.
fn answer<D: Driver>(
    mut env: D,
    requests: &mpsc::Receiver<Request>,
    replies: &mpsc::Sender<Reply>,
) {
    let mut episode: Option<(State, EpisodeRecorder)> = None;
    let none = || "a step before any reset".to_string();
    for request in requests {
        let reply = match request {
            #[cfg(test)]
            Request::Panic => panic!("a panic asked for by a test"),
            Request::Reset(seed) => {
                let mut recorder = EpisodeRecorder::default();
                Reply::Reset(env.reset(seed, &mut recorder).and_then(|fresh| {
                    let reply = ResetReply {
                        observation: env.observe(&fresh)?,
                        mask: env.mask(&fresh),
                        hedge_mask: env.hedge_mask(&fresh),
                        done: fresh.is_done(),
                    };
                    episode = Some((fresh, recorder));
                    Ok(reply)
                }))
            }
            Request::Step(action) => Reply::Step(match episode.as_mut() {
                None => Err(none()),
                Some((state, recorder)) => env.step(state, action, recorder),
            }),
            Request::StepResidual(proposal) => Reply::Step(match episode.as_mut() {
                None => Err(none()),
                Some((state, recorder)) => env.step_residual(state, proposal, recorder),
            }),
            Request::Attribution => Reply::Attribution(match episode.as_ref() {
                None => Err(none()),
                Some((state, recorder)) => env.inner_attribution(state, recorder),
            }),
        };
        if replies.send(reply).is_err() {
            return;
        }
    }
}

/// Builds a residual environment over `source` and answers requests.
fn serve_residual<S: QuoteSource + Clone>(
    job: Job<'_>,
    source: S,
    default_order_size: Decimal,
    bounds: ResidualBounds,
) {
    let Job {
        kernel,
        data,
        config,
        expected,
        ready,
        requests,
        replies,
    } = job;
    let refuse =
        |error: market_maker_rs::types::error::MMError| BridgeError::Refused(error.to_string());
    let built = config.rl().and_then(|rl| {
        let env: ResidualEnv<'_, S, ORD, MSG, DLV, OBS> =
            ResidualEnv::new(kernel, data, source, default_order_size, bounds, rl)
                .map_err(refuse)?;
        let spec = env.env().spec();
        let id = spec.id();
        check_spec(spec, id, expected)?;
        let shape = spec.shape();
        let info = EnvInfo {
            spec: id,
            features: shape.features,
            offsets: shape.bid_offsets,
            sizes: shape.sizes,
            hedge_actions: spec.action.hedge == HedgeActions::MarketableAtTouch,
            kernel_fingerprint: env.env().kernel().config_fingerprint(),
            dataset_digest: data.id().digest,
            rows: data.id().len,
        };
        Ok((env, info))
    });
    match built {
        Ok((env, info)) => {
            if ready.send(Ok(info)).is_ok() {
                answer(env, &requests, &replies);
            }
        }
        Err(error) => {
            let _ = ready.send(Err(error));
        }
    }
}

fn check_spec(
    spec: &market_maker_rs::rl::RlSpec,
    id: SpecId,
    expected: Option<SpecId>,
) -> Result<(), BridgeError> {
    if !known_versions(&id) {
        return Err(BridgeError::Refused(format!("unknown versions in {id:?}")));
    }
    if let Some(expected) = expected {
        spec.ensure(expected)
            .map_err(|error| BridgeError::Refused(error.to_string()))?;
    }
    Ok(())
}

/// Parses the baseline and serves it.
fn start_residual(job: Job<'_>, residual: &ResidualConfig) {
    let bounds = ResidualBounds {
        price_ticks: residual.price_ticks,
        size_lots: residual.size_lots,
    };
    let parsed = decimal("default_order_size", &residual.default_order_size).and_then(|size| {
        Ok((
            size,
            parse_baseline(&residual.baseline, &job.config.lot_size)?,
        ))
    });
    match parsed {
        Ok((size, Baseline::Avellaneda(source))) => serve_residual(job, source, size, bounds),
        Ok((size, Baseline::Glft(source))) => serve_residual(job, source, size, bounds),
        Err(error) => {
            let _ = job.ready.send(Err(error));
        }
    }
}

fn parse_baseline(baseline: &BaselineConfig, lot_size: &str) -> Result<Baseline, BridgeError> {
    match baseline {
        BaselineConfig::AvellanedaStoikov {
            risk_aversion,
            volatility,
            order_intensity,
            session_start_ms,
            horizon_ms,
            size,
        } => Ok(Baseline::Avellaneda(AvellanedaQuoter {
            risk_aversion: decimal("risk_aversion", risk_aversion)?,
            volatility: decimal("volatility", volatility)?,
            order_intensity: decimal("order_intensity", order_intensity)?,
            session_start_ms: *session_start_ms,
            horizon_ms: *horizon_ms,
            size: decimal("size", size)?,
        })),
        BaselineConfig::Glft {
            risk_aversion,
            order_intensity_k,
            intensity_scale_a,
            max_inventory_lots,
            terminal_time,
            min_spread,
            volatility,
            session_start_ms,
        } => {
            let lot = decimal("lot_size", lot_size)?;
            let model = GLFTConfig::new(
                decimal("risk_aversion", risk_aversion)?,
                decimal("order_intensity_k", order_intensity_k)?,
                decimal("intensity_scale_a", intensity_scale_a)?,
                *max_inventory_lots,
                lot,
                *terminal_time,
                decimal("min_spread", min_spread)?,
            )
            .map_err(|error| BridgeError::Refused(error.to_string()))?;
            Ok(Baseline::Glft(GlftQuoter {
                config: model,
                volatility: decimal("volatility", volatility)?,
                size: lot,
                session_start_ms: *session_start_ms,
            }))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Baseline {
    Avellaneda(AvellanedaQuoter),
    Glft(GlftQuoter),
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
        check_spec(spec, id, expected)?;
        let shape = spec.shape();
        let info = EnvInfo {
            spec: id,
            features: shape.features,
            offsets: shape.bid_offsets,
            sizes: shape.sizes,
            hedge_actions: spec.action.hedge == HedgeActions::MarketableAtTouch,
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
    answer(env, &requests, &replies);
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
                if let Some(residual) = &config.residual {
                    start_residual(job, residual);
                    return;
                }
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
            Reply::Reset(Err(reason))
            | Reply::Step(Err(reason))
            | Reply::Attribution(Err(reason)) => Err(self.fail(reason)),
            Reply::Step(Ok(_)) | Reply::Attribution(Ok(_)) => {
                Err(self.fail("the wrong reply to a reset".to_string()))
            }
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
        self.step_request(Request::Step(action))
    }

    /// Takes one decision of a residual policy.
    ///
    /// # Errors
    ///
    /// As [`Worker::step`]. A proposal is never an error by itself, and calling this on an
    /// environment with a grid is.
    pub fn step_residual(&mut self, proposal: ResidualProposal) -> Result<StepReply, BridgeError> {
        self.step_request(Request::StepResidual(proposal))
    }

    /// The attribution of the episode so far. It fails if it does not reconcile with the
    /// reward, which also ends the environment.
    ///
    /// # Errors
    ///
    /// As [`Worker::step`].
    pub fn attribution(&mut self) -> Result<AttributionReply, BridgeError> {
        match self.call(Request::Attribution)? {
            Reply::Attribution(Ok(reply)) => Ok(reply),
            Reply::Reset(Err(reason))
            | Reply::Step(Err(reason))
            | Reply::Attribution(Err(reason)) => Err(self.fail(reason)),
            Reply::Reset(Ok(_)) | Reply::Step(Ok(_)) => {
                Err(self.fail("the wrong reply to an attribution".to_string()))
            }
        }
    }

    fn step_request(&mut self, request: Request) -> Result<StepReply, BridgeError> {
        match self.call(request)? {
            Reply::Step(Ok(reply)) => Ok(reply),
            Reply::Reset(Err(reason))
            | Reply::Step(Err(reason))
            | Reply::Attribution(Err(reason)) => Err(self.fail(reason)),
            Reply::Reset(Ok(_)) | Reply::Attribution(Ok(_)) => {
                Err(self.fail("the wrong reply to a step".to_string()))
            }
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
