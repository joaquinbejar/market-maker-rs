//! One environment on its own thread.

use std::str::FromStr;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    Attribution, EvaluationInput, InventoryStats, attribute, inventory_stats, max_drawdown,
};
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionMask, ActionStatus, AvellanedaQuoter, ERR_KERNEL_MISMATCH,
    ERR_PRICE_OUT_OF_RANGE, ERR_SIZE_OUT_OF_RANGE, EventSink, ExecKind, GateCheck, GateVerdict,
    GlftCounts, GlftForm, GlftQuoter, KernelConfig, KernelDataset, KernelEvent, KernelFillModel,
    KernelLatency, KernelMargin, KernelTick, Observation, QuoteSource, SourceQuote,
    quote_error_name,
};
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    CancelActions, ERR_ACTION_OFF_GRID, ERR_CANCEL_REFUSED, ERR_EXPIRY_REFUSED, ERR_HEDGE_REFUSED,
    EpisodeRecorder, FeatureVector, HedgeActions, InvalidActionPolicy, ObservationScales,
    QuoteExpiry, QuoteLives, ResidualBounds, ResidualDecision, ResidualEnv, ResidualProposal,
    RlAction, RlConfig, RlEnv, RlState, SpecId, TerminalValuation, Transition,
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

/// The action space version this bridge was written for with the cancel action
/// (`cancel_action`), a literal for the same reason.
pub const BUILT_FOR_CANCEL_ACTION: u16 = 4;

/// The action space version this bridge was written for with quote lives (`quote_lives_ns`),
/// with or without the cancel action, a literal for the same reason.
pub const BUILT_FOR_EXPIRY_ACTION: u16 = 5;

/// The versions are the ones this bridge was written for, and the action space is the one
/// the configuration asks for: version [`BUILT_FOR_CANCEL_ACTION`] with `cancel_action`
/// and no lives, [`BUILT_FOR_EXPIRY_ACTION`] with `quote_lives_ns`.
fn known_versions(id: &SpecId, cancel_action: bool, quote_lives: bool) -> bool {
    let (observation, action, reward, episode) = BUILT_FOR;
    let action = if quote_lives {
        BUILT_FOR_EXPIRY_ACTION
    } else if cancel_action {
        BUILT_FOR_CANCEL_ACTION
    } else {
        action
    };
    (id.observation, id.action, id.reward, id.episode) == (observation, action, reward, episode)
}

/// Why an action was not applied, as `info["refused_because"]` reports it; `None` when it
/// was. A kernel refusal is named by its reason. A quote error says what its code stands
/// for: a strategy's error by its `MMError` variant, a strategy quote the kernel cannot
/// take, an action off the grid, or a hedge not available at this decision; none of them
/// reads as another.
pub fn refused_because(status: ActionStatus) -> Option<String> {
    match status {
        ActionStatus::Applied => None,
        ActionStatus::Rejected(reason) => Some(format!("{reason:?}")),
        ActionStatus::QuoteError(code) => Some(match code {
            ERR_ACTION_OFF_GRID => format!("off grid (code {})", code.0),
            ERR_HEDGE_REFUSED => format!("hedge not available (code {})", code.0),
            ERR_CANCEL_REFUSED => format!("cancel not available (code {})", code.0),
            ERR_EXPIRY_REFUSED => format!("expiring quote not available (code {})", code.0),
            ERR_PRICE_OUT_OF_RANGE => {
                format!("strategy quote price out of range (code {})", code.0)
            }
            ERR_SIZE_OUT_OF_RANGE => {
                format!("strategy quote size out of range (code {})", code.0)
            }
            ERR_KERNEL_MISMATCH => format!("strategy quote for another kernel (code {})", code.0),
            _ => match quote_error_name(code) {
                Some(name) => format!("strategy error: {name} (code {})", code.0),
                None => format!("quote error (code {})", code.0),
            },
        }),
    }
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
    /// The kernel's queue position model at `lots` lots every `per_ms` milliseconds,
    /// which can be below one lot per millisecond. The kernel refuses either at 0.
    QueuePositionRational {
        /// Lots cleared every `per_ms` milliseconds.
        lots: u64,
        /// Milliseconds in which `lots` clear.
        per_ms: u64,
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

/// The kernel's balance gate in the configuration ([`KernelMargin`]): the spot, fully
/// collateralised rule, with the initial capital as the quote balance.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginConfig {
    /// Base deposit, on the lot grid; zero or more. Sells draw on it.
    pub base_deposit: String,
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
        /// Which quotes: `exact` (the default), `closed_form`, or `exact_or_closed_form`,
        /// which takes the closed form only where the exact quote is over its step cap.
        #[serde(default)]
        form: GlftFormConfig,
    },
    /// Classical strategies in turn: `points[0]` from the start, and from each change on the
    /// point it names. A change at row `r` applies from the first decision whose observation
    /// has row `r` or later, which latency may delay, so it never reads a row early.
    Scheduled {
        /// The strategies, none of them scheduled.
        points: Vec<BaselineConfig>,
        /// `[row, index]` pairs: rows strictly increasing, each index a point other than the
        /// one in force.
        changes: Vec<(u64, usize)>,
    },
}

/// [`GlftForm`] in the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlftFormConfig {
    /// [`GlftForm::Exact`].
    #[default]
    Exact,
    /// [`GlftForm::ClosedForm`].
    ClosedForm,
    /// [`GlftForm::ExactOrClosedForm`].
    ExactOrClosedForm,
}

impl From<GlftFormConfig> for GlftForm {
    fn from(form: GlftFormConfig) -> Self {
        match form {
            GlftFormConfig::Exact => Self::Exact,
            GlftFormConfig::ClosedForm => Self::ClosedForm,
            GlftFormConfig::ExactOrClosedForm => Self::ExactOrClosedForm,
        }
    }
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
    /// The kernel's balance gate, if any; it decides after the limits. Absent by default,
    /// which keeps a configuration's specification and fingerprint.
    #[serde(default)]
    pub margin: Option<MarginConfig>,
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
    /// Whether the cancel action exists (`RlAction::Cancel`, `CancelActions::CancelOnly`), of
    /// action space version [`BUILT_FOR_CANCEL_ACTION`]. Off by default, which keeps a
    /// configuration's specification and fingerprint; a specification with it is refused
    /// by a configuration without it.
    #[serde(default)]
    pub cancel_action: bool,
    /// The lives of an expiring quote, in nanoseconds, of action space version
    /// [`BUILT_FOR_EXPIRY_ACTION`]. Absent by default, which keeps a configuration's
    /// specification and fingerprint.
    #[serde(default)]
    pub quote_lives_ns: Option<Vec<u64>>,
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
            FillConfig::QueuePositionRational {
                lots,
                per_ms,
                min_queue_ms,
            } => KernelFillModel::QueuePositionRational {
                lots: *lots,
                per_ms: *per_ms,
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
        if let Some(margin) = &self.margin {
            kernel = kernel.with_margin(KernelMargin::new(decimal(
                "base_deposit",
                &margin.base_deposit,
            )?));
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
        if self.cancel_action && self.residual.is_some() {
            return Err(BridgeError::Refused(
                "cancel_action with a residual policy: a residual policy takes no cancel action"
                    .to_string(),
            ));
        }
        if self.quote_lives_ns.is_some() && self.residual.is_some() {
            return Err(BridgeError::Refused(
                "quote_lives_ns with a residual policy: a residual policy takes no quote life"
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
        let config = if self.hedge_actions {
            config.with_hedge_actions()
        } else {
            config
        };
        let config = if self.cancel_action {
            config.with_cancel_action()
        } else {
            config
        };
        Ok(match &self.quote_lives_ns {
            Some(lives) => config.with_quote_lives(
                QuoteLives::new(lives).map_err(|error| BridgeError::Refused(error.to_string()))?,
            ),
            None => config,
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
    /// Whether the cancel action exists, as the specification states.
    pub cancel_action: bool,
    /// The lives of an expiring quote, as the specification states.
    pub quote_lives: Option<QuoteLives>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributionReply {
    /// The components, in the evaluation's definitions.
    pub attribution: Attribution,
    /// Fills, the terminal close included.
    pub fills: usize,
    /// The position over the window, in the evaluation's definitions.
    pub inventory: InventoryStats,
    /// The largest fall of equity from a peak, valued at the marks.
    pub max_drawdown: Decimal,
    /// From the first mark to just after the last record, in milliseconds.
    pub window_ms: (u64, u64),
    /// The position after each fill, net of base fees, with the fill's time and price.
    pub positions: Vec<PositionChange>,
}

/// The position after one fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionChange {
    /// The fill's time, in milliseconds.
    pub timestamp_ms: u64,
    /// The fill's price.
    pub price: Decimal,
    /// The position after it, net of base fees.
    pub position: Decimal,
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
    /// The venue's order reports during the step, at venue time: an audit, never seen
    /// by a policy.
    pub orders: OrderCounts,
    /// The gate's cuts and refusals during the step, by check: an audit, never seen by a
    /// policy.
    pub gates: GateCounts,
    /// The lots filled during the step, by side (#282): an audit, never seen by a policy.
    pub filled: FilledLots,
}

/// Lots filled, by the side of the agent's order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilledLots {
    /// Lots the agent bought.
    pub buy: u64,
    /// Lots the agent sold.
    pub sell: u64,
}

/// Counts of the venue's order reports, by kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OrderCounts {
    /// Orders accepted and resting.
    pub accepted: u64,
    /// Orders rejected, locally or by the venue.
    pub rejected: u64,
    /// Orders canceled.
    pub canceled: u64,
    /// Cancels that arrived after the order ended.
    pub cancel_rejected: u64,
    /// Fills, partial ones included.
    pub fills: u64,
}

/// Orders the gate cut or refused, for one of its checks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerdictCounts {
    /// Submitted with fewer lots.
    pub reduced: u64,
    /// Refused, an off-lot reduction included.
    pub rejected: u64,
}

/// The gate's decisions during a step that did not submit an order as requested, by the
/// check that decided: an audit, never seen by a policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateCounts {
    /// [`KernelConfig::limits`].
    pub limits: VerdictCounts,
    /// The combined limits of a hedged kernel.
    pub combined: VerdictCounts,
    /// The balance gate, [`KernelMargin`].
    pub margin: VerdictCounts,
}

/// The episode's records, and the counts of the current step's order reports.
#[derive(Debug, Default)]
struct Audit {
    recorder: EpisodeRecorder,
    orders: OrderCounts,
    gates: GateCounts,
    filled: FilledLots,
}

impl EventSink for Audit {
    fn event(&mut self, event: KernelEvent) {
        if let KernelEvent::Report(report) = &event {
            if matches!(report.kind, ExecKind::PartialFill | ExecKind::Fill) {
                // A side this bridge does not know is not counted.
                let lots = match report.side {
                    Action::Buy => Some(&mut self.filled.buy),
                    Action::Sell => Some(&mut self.filled.sell),
                    _ => None,
                };
                if let Some(lots) = lots {
                    *lots = lots.saturating_add(report.last);
                }
            }
            let count = match report.kind {
                ExecKind::Accepted => &mut self.orders.accepted,
                ExecKind::Rejected => &mut self.orders.rejected,
                ExecKind::Canceled => &mut self.orders.canceled,
                ExecKind::CancelRejected => &mut self.orders.cancel_rejected,
                ExecKind::PartialFill | ExecKind::Fill => &mut self.orders.fills,
            };
            *count = count.saturating_add(1);
        }
        if let KernelEvent::Gate(gate) = &event {
            // A check or a verdict this bridge does not know is not counted.
            let check = match gate.check {
                GateCheck::Limits => Some(&mut self.gates.limits),
                GateCheck::Combined => Some(&mut self.gates.combined),
                GateCheck::Margin => Some(&mut self.gates.margin),
                _ => None,
            };
            let count = check.and_then(|check| match gate.verdict {
                GateVerdict::Reduced => Some(&mut check.reduced),
                GateVerdict::ReducedOffLot | GateVerdict::Rejected => Some(&mut check.rejected),
                _ => None,
            });
            if let Some(count) = count {
                *count = count.saturating_add(1);
            }
        }
        self.recorder.event(event);
    }
}

enum Request {
    Reset(u64),
    Step(RlAction),
    StepResidual(ResidualProposal),
    Attribution,
    ReplaceRows(Vec<KernelTick>),
    #[cfg(test)]
    Panic,
}

// Sent by value on every step: a box would allocate for each one.
#[allow(clippy::large_enum_variant)]
enum Reply {
    Reset(Result<ResetReply, String>),
    Step(Result<StepReply, String>),
    Attribution(Result<AttributionReply, String>),
    Replaced(Result<EnvInfo, String>),
}

/// A step taken since the last reset and its reply, to replay on other rows (#282): the
/// replayed step must give the same reply.
#[derive(Debug, Clone, Copy)]
enum Logged {
    Step(RlAction, StepReply),
    Residual(ResidualProposal, StepReply),
}

/// What a replay must reach: the episode before the rows were replaced.
#[derive(Debug, Clone, Copy)]
struct Expect {
    state: State,
    observation: FeatureVector,
    mask: ActionMask,
    hedge_mask: u64,
}

/// An episode to rebuild on other rows: its seed and steps, what the replay must reach (none
/// when the rows are the ones it ran on), and the rows to fall back to if it does not.
#[derive(Debug, Clone)]
struct Resume {
    seed: u64,
    log: Vec<Logged>,
    expect: Option<Box<Expect>>,
    fallback: Option<Vec<KernelTick>>,
}

/// How a request loop ended.
enum Answered {
    /// The requests ended.
    Closed,
    /// Rebuild on `rows`, then resume.
    Rebuild(Vec<KernelTick>, Resume),
}

type State = RlState<ORD, MSG, DLV, OBS>;

/// What the request loop needs from an environment.
trait Driver {
    fn reset(&mut self, seed: u64, audit: &mut Audit) -> Result<State, String>;
    fn observe(&self, state: &State) -> Result<FeatureVector, String>;
    fn mask(&self, state: &State) -> ActionMask;
    fn hedge_mask(&self, state: &State) -> u64;
    fn step(
        &mut self,
        state: &mut State,
        action: RlAction,
        audit: &mut Audit,
    ) -> Result<StepReply, String>;
    fn step_residual(
        &mut self,
        state: &mut State,
        proposal: ResidualProposal,
        audit: &mut Audit,
    ) -> Result<StepReply, String>;
    fn inner_attribution(&self, state: &State, audit: &Audit) -> Result<AttributionReply, String>;
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
    let mut position = Decimal::ZERO;
    let mut positions = Vec::with_capacity(records.fills.len());
    for fill in &records.fills {
        let signed = match fill.side {
            Action::Buy => fill.quantity,
            Action::Sell => -fill.quantity,
            // `Action` is another crate's non exhaustive enum: a side added there is refused.
            _ => return Err(format!("a fill with an unknown side {:?}", fill.side)),
        };
        let change = signed
            .checked_sub(fill.base_fee)
            .and_then(|delta| position.checked_add(delta))
            .ok_or_else(|| "a position does not convert".to_string())?;
        position = change;
        positions.push(PositionChange {
            timestamp_ms: fill.timestamp_ms,
            price: fill.price,
            position,
        });
    }
    Ok(AttributionReply {
        attribution: split.total,
        fills: records.fills.len(),
        inventory: inventory_stats(&input).map_err(|e| e.to_string())?,
        max_drawdown: max_drawdown(&input).map_err(|e| e.to_string())?,
        window_ms: (records.window.start_ms, records.window.end_ms),
        positions,
    })
}

impl<const K: usize, const M: usize> Driver for RlEnv<'_, K, M, ORD, MSG, DLV, OBS> {
    fn reset(&mut self, seed: u64, audit: &mut Audit) -> Result<State, String> {
        self.reset_with(seed, audit).map_err(|e| e.to_string())
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
        audit: &mut Audit,
    ) -> Result<StepReply, String> {
        audit.orders = OrderCounts::default();
        audit.gates = GateCounts::default();
        audit.filled = FilledLots::default();
        let transition = self
            .step_with(state, action, audit)
            .map_err(|e| e.to_string())?;
        Ok(StepReply {
            transition,
            net: state.net(),
            penalties: state.penalties(),
            decision: None,
            orders: audit.orders,
            gates: audit.gates,
            filled: audit.filled,
        })
    }

    fn step_residual(
        &mut self,
        _state: &mut State,
        _proposal: ResidualProposal,
        _audit: &mut Audit,
    ) -> Result<StepReply, String> {
        Err("a residual step on an environment with a grid".to_string())
    }

    fn inner_attribution(&self, state: &State, audit: &Audit) -> Result<AttributionReply, String> {
        attribution_of(self, state, &audit.recorder)
    }
}

impl<S: QuoteSource + Clone> Driver for ResidualEnv<'_, S, ORD, MSG, DLV, OBS> {
    fn reset(&mut self, seed: u64, audit: &mut Audit) -> Result<State, String> {
        self.reset_with(seed, audit).map_err(|e| e.to_string())
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

    /// Only a hold: the resting orders stay, and the classical strategy is not asked, so
    /// a decision interval can hold a residual policy's orders between its decisions.
    fn step(
        &mut self,
        state: &mut State,
        action: RlAction,
        audit: &mut Audit,
    ) -> Result<StepReply, String> {
        if action != RlAction::Hold {
            return Err("a grid action on a residual environment".to_string());
        }
        audit.orders = OrderCounts::default();
        audit.gates = GateCounts::default();
        audit.filled = FilledLots::default();
        let transition = self
            .env()
            .step_with(state, RlAction::Hold, audit)
            .map_err(|e| e.to_string())?;
        Ok(StepReply {
            transition,
            net: state.net(),
            penalties: state.penalties(),
            decision: None,
            orders: audit.orders,
            gates: audit.gates,
            filled: audit.filled,
        })
    }

    fn step_residual(
        &mut self,
        state: &mut State,
        proposal: ResidualProposal,
        audit: &mut Audit,
    ) -> Result<StepReply, String> {
        audit.orders = OrderCounts::default();
        audit.gates = GateCounts::default();
        audit.filled = FilledLots::default();
        let step = self
            .step_with(state, proposal, audit)
            .map_err(|e| e.to_string())?;
        Ok(StepReply {
            transition: step.transition,
            net: state.net(),
            penalties: state.penalties(),
            decision: Some(step.decision),
            orders: audit.orders,
            gates: audit.gates,
            filled: audit.filled,
        })
    }

    fn inner_attribution(&self, state: &State, audit: &Audit) -> Result<AttributionReply, String> {
        attribution_of(self.env(), state, &audit.recorder)
    }
}

/// Equal, or equal down to the bits a `Debug` print shows: a NaN a policy proposed equals
/// itself, which `PartialEq` on floats does not grant. The print is only made when `==` fails.
fn same<T: PartialEq + std::fmt::Debug>(a: &T, b: &T) -> bool {
    a == b || format!("{a:?}") == format!("{b:?}")
}

/// Replays `resume` on `env`, checks it against what it must reach, and answers the replace
/// that asked for it. `None` when it reached it (or had nothing to reach), with the episode;
/// otherwise the rebuild that falls back.
fn replay<D: Driver>(
    env: &mut D,
    info: EnvInfo,
    replies: &mpsc::Sender<Reply>,
    resume: Resume,
) -> Result<(State, Audit, Vec<Logged>, u64), Answered> {
    let mut audit = Audit::default();
    let replayed = (|| -> Result<State, String> {
        let mut state = env.reset(resume.seed, &mut audit)?;
        for (index, step) in resume.log.iter().enumerate() {
            let (reply, logged) = match step {
                Logged::Step(action, logged) => {
                    (env.step(&mut state, *action, &mut audit)?, logged)
                }
                Logged::Residual(proposal, logged) => (
                    env.step_residual(&mut state, *proposal, &mut audit)?,
                    logged,
                ),
            };
            if !same(&reply, logged) {
                return Err(format!(
                    "the replayed step {} differs from the episode's",
                    index + 1
                ));
            }
        }
        Ok(state)
    })();
    // The replayed steps' counts are not reported.
    audit.orders = OrderCounts::default();
    audit.gates = GateCounts::default();
    audit.filled = FilledLots::default();
    let reached = match (&replayed, &resume.expect) {
        (Err(error), _) => Err(format!("the replay failed: {error}")),
        (Ok(_), None) => Ok(()),
        (Ok(state), Some(expect)) => {
            let observation = env.observe(state);
            if !state.same_apart_from_dataset(&expect.state) {
                Err(format!(
                    "the replayed state differs after {} steps",
                    resume.log.len()
                ))
            } else if !observation
                .as_ref()
                .is_ok_and(|observation| same(observation, &expect.observation))
                || env.mask(state) != expect.mask
                || env.hedge_mask(state) != expect.hedge_mask
            {
                Err(format!(
                    "the replayed observation differs after {} steps",
                    resume.log.len()
                ))
            } else {
                Ok(())
            }
        }
    };
    if resume.expect.is_some() {
        let reply = reached.as_ref().map(|()| info).map_err(Clone::clone);
        if replies.send(Reply::Replaced(reply)).is_err() {
            return Err(Answered::Closed);
        }
    }
    match (reached, replayed) {
        (Ok(()), Ok(state)) => Ok((state, audit, resume.log, resume.seed)),
        (_, _) => match resume.fallback {
            // Back to the rows it ran on, where the replay reaches the same episode.
            Some(rows) => Err(Answered::Rebuild(
                rows,
                Resume {
                    expect: None,
                    fallback: None,
                    ..resume
                },
            )),
            None => Err(Answered::Closed),
        },
    }
}

/// Answers requests until the sender goes or the rows are replaced. `resume` rebuilds an
/// episode first, on rows that keep what it read.
fn answer<D: Driver>(
    mut env: D,
    info: EnvInfo,
    rows: &[KernelTick],
    requests: &mpsc::Receiver<Request>,
    replies: &mpsc::Sender<Reply>,
    resume: Option<Resume>,
) -> Answered {
    let mut episode: Option<(State, Audit)> = None;
    let mut log: Vec<Logged> = Vec::new();
    let mut seed = 0;
    if let Some(resume) = resume {
        match replay(&mut env, info, replies, resume) {
            Ok((state, audit, steps, from)) => {
                episode = Some((state, audit));
                log = steps;
                seed = from;
            }
            Err(answered) => return answered,
        }
    }
    let none = || "a step before any reset".to_string();
    for request in requests {
        let reply = match request {
            #[cfg(test)]
            Request::Panic => panic!("a panic asked for by a test"),
            Request::Reset(from) => {
                let mut audit = Audit::default();
                Reply::Reset(env.reset(from, &mut audit).and_then(|fresh| {
                    let reply = ResetReply {
                        observation: env.observe(&fresh)?,
                        mask: env.mask(&fresh),
                        hedge_mask: env.hedge_mask(&fresh),
                        done: fresh.is_done(),
                    };
                    episode = Some((fresh, audit));
                    log.clear();
                    seed = from;
                    Ok(reply)
                }))
            }
            Request::Step(action) => Reply::Step(match episode.as_mut() {
                None => Err(none()),
                Some((state, audit)) => env.step(state, action, audit).inspect(|reply| {
                    log.push(Logged::Step(action, *reply));
                }),
            }),
            Request::StepResidual(proposal) => Reply::Step(match episode.as_mut() {
                None => Err(none()),
                Some((state, audit)) => {
                    env.step_residual(state, proposal, audit).inspect(|reply| {
                        log.push(Logged::Residual(proposal, *reply));
                    })
                }
            }),
            Request::Attribution => Reply::Attribution(match episode.as_ref() {
                None => Err(none()),
                Some((state, audit)) => env.inner_attribution(state, audit),
            }),
            Request::ReplaceRows(new_rows) => match episode.as_ref() {
                None => Reply::Replaced(Err("a replace before any reset".to_string())),
                Some((state, _)) if state.is_done() => {
                    Reply::Replaced(Err("a replace after the episode ended".to_string()))
                }
                Some((state, _)) => match env.observe(state) {
                    Err(error) => Reply::Replaced(Err(error)),
                    Ok(observation) => {
                        // The rows it has read, through the row it observed, must not change.
                        let read = observation
                            .row
                            .and_then(|row| usize::try_from(row).ok())
                            .map_or(0, |row| row.saturating_add(1));
                        if new_rows.get(..read) != rows.get(..read) {
                            Reply::Replaced(Err(format!(
                                "the new rows differ within the {read} rows already read"
                            )))
                        } else {
                            return Answered::Rebuild(
                                new_rows,
                                Resume {
                                    seed,
                                    log,
                                    expect: Some(Box::new(Expect {
                                        state: *state,
                                        observation,
                                        mask: env.mask(state),
                                        hedge_mask: env.hedge_mask(state),
                                    })),
                                    fallback: Some(rows.to_vec()),
                                },
                            );
                        }
                    }
                },
            },
        };
        if replies.send(reply).is_err() {
            return Answered::Closed;
        }
    }
    Answered::Closed
}

/// Builds a residual environment over `source` and answers requests.
fn serve_residual<S: QuoteSource + Clone>(
    job: Job<'_>,
    source: S,
    default_order_size: Decimal,
    bounds: ResidualBounds,
) -> Option<Answered> {
    let Job {
        kernel,
        data,
        config,
        expected,
        ready,
        requests,
        replies,
        resume,
    } = job;
    let refuse =
        |error: market_maker_rs::types::error::MMError| BridgeError::Refused(error.to_string());
    let built = config.rl().and_then(|rl| {
        let env: ResidualEnv<'_, S, ORD, MSG, DLV, OBS> =
            ResidualEnv::new(kernel, data, source, default_order_size, bounds, rl)
                .map_err(refuse)?;
        let spec = env.env().spec();
        let id = spec.id();
        check_spec(spec, id, expected, false, false)?;
        let shape = spec.shape();
        let info = EnvInfo {
            spec: id,
            features: shape.features,
            offsets: shape.bid_offsets,
            sizes: shape.sizes,
            hedge_actions: spec.action.hedge == HedgeActions::MarketableAtTouch,
            cancel_action: spec.action.cancel == CancelActions::CancelOnly,
            quote_lives: quote_lives_of(&spec.action.expiry),
            kernel_fingerprint: env.env().kernel().config_fingerprint(),
            dataset_digest: data.id().digest,
            rows: data.id().len,
        };
        Ok((env, info))
    });
    match built {
        Ok((env, info)) => ready
            .send(Ok(info))
            .is_ok()
            .then(|| answer(env, info, data.ticks(), requests, replies, resume)),
        Err(error) => {
            let _ = ready.send(Err(error));
            None
        }
    }
}

/// The lives a specification states, which it checked when it was built.
fn quote_lives_of(expiry: &QuoteExpiry) -> Option<QuoteLives> {
    match expiry {
        QuoteExpiry::Lives(lives) => QuoteLives::new(lives).ok(),
        _ => None,
    }
}

fn check_spec(
    spec: &market_maker_rs::rl::RlSpec,
    id: SpecId,
    expected: Option<SpecId>,
    cancel_action: bool,
    quote_lives: bool,
) -> Result<(), BridgeError> {
    if !known_versions(&id, cancel_action, quote_lives) {
        return Err(BridgeError::Refused(format!("unknown versions in {id:?}")));
    }
    if let Some(expected) = expected {
        spec.ensure(expected)
            .map_err(|error| BridgeError::Refused(error.to_string()))?;
    }
    Ok(())
}

/// Parses the baseline and serves it.
fn start_residual(job: Job<'_>, residual: &ResidualConfig) -> Option<Answered> {
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
        Ok((size, Baseline::Scheduled(source))) => serve_residual(job, source, size, bounds),
        Err(error) => {
            let _ = job.ready.send(Err(error));
            None
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
            form,
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
                form: (*form).into(),
                counts: GlftCounts::default(),
            }))
        }
        BaselineConfig::Scheduled { points, changes } => Ok(Baseline::Scheduled(parse_schedule(
            points, changes, lot_size,
        )?)),
    }
}

/// Parses a schedule, refusing one that is empty, nested, or whose changes are out of order,
/// name no point, or name the point already in force.
fn parse_schedule(
    points: &[BaselineConfig],
    changes: &[(u64, usize)],
    lot_size: &str,
) -> Result<ScheduledQuoter, BridgeError> {
    let refuse = |why: String| Err(BridgeError::Refused(format!("scheduled baseline: {why}")));
    if points.is_empty() {
        return refuse("no points".to_string());
    }
    let mut parsed = Vec::with_capacity(points.len());
    for point in points {
        match parse_baseline(point, lot_size)? {
            Baseline::Scheduled(_) => return refuse("a point is itself scheduled".to_string()),
            plain => parsed.push(plain),
        }
    }
    let mut in_force = 0;
    let mut last_row = None;
    for &(row, index) in changes {
        if last_row.is_some_and(|last| row <= last) {
            return refuse(format!("change rows are not increasing at row {row}"));
        }
        if index >= parsed.len() {
            return refuse(format!("row {row} names point {index} of {}", parsed.len()));
        }
        if index == in_force {
            return refuse(format!("row {row} names point {index}, already in force"));
        }
        last_row = Some(row);
        in_force = index;
    }
    Ok(ScheduledQuoter {
        points: parsed,
        changes: changes.to_vec(),
    })
}

#[derive(Debug, Clone, PartialEq)]
enum Baseline {
    Avellaneda(AvellanedaQuoter),
    Glft(GlftQuoter),
    Scheduled(ScheduledQuoter),
}

impl QuoteSource for Baseline {
    fn quote(
        &mut self,
        obs: &Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> market_maker_rs::types::error::MMResult<SourceQuote> {
        match self {
            Self::Avellaneda(source) => source.quote(obs, mid, inventory),
            Self::Glft(source) => source.quote(obs, mid, inventory),
            Self::Scheduled(source) => source.quote(obs, mid, inventory),
        }
    }

    fn validate(&self, tick: Decimal, lot: Decimal) -> market_maker_rs::types::error::MMResult<()> {
        match self {
            Self::Avellaneda(source) => source.validate(tick, lot),
            Self::Glft(source) => source.validate(tick, lot),
            Self::Scheduled(source) => source.validate(tick, lot),
        }
    }
}

/// Classical strategies in turn, by the row the agent last observed. See
/// [`BaselineConfig::Scheduled`].
#[derive(Debug, Clone, PartialEq)]
struct ScheduledQuoter {
    /// The strategies, none of them scheduled.
    points: Vec<Baseline>,
    /// `(row, index)`, rows strictly increasing.
    changes: Vec<(u64, usize)>,
}

impl ScheduledQuoter {
    /// The point in force at an observation: that of the last change at or before its row,
    /// or the first point before any row or change.
    fn in_force(&self, obs: &Observation) -> usize {
        let Some(row) = obs.row else {
            return 0;
        };
        let passed = self.changes.partition_point(|(at, _)| *at <= row);
        passed
            .checked_sub(1)
            .and_then(|last| self.changes.get(last))
            .map_or(0, |(_, index)| *index)
    }
}

impl QuoteSource for ScheduledQuoter {
    fn quote(
        &mut self,
        obs: &Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> market_maker_rs::types::error::MMResult<SourceQuote> {
        let index = self.in_force(obs);
        // Parsing checked every index against the points.
        self.points
            .get_mut(index)
            .ok_or_else(|| {
                market_maker_rs::types::error::MMError::InvalidConfiguration(format!(
                    "no point {index}"
                ))
            })?
            .quote(obs, mid, inventory)
    }

    fn validate(&self, tick: Decimal, lot: Decimal) -> market_maker_rs::types::error::MMResult<()> {
        self.points
            .iter()
            .try_for_each(|point| point.validate(tick, lot))
    }
}

/// What a worker thread serves.
struct Job<'a> {
    kernel: &'a KernelConfig,
    data: &'a KernelDataset,
    config: &'a BridgeConfig,
    expected: Option<SpecId>,
    ready: mpsc::Sender<Result<EnvInfo, BridgeError>>,
    requests: &'a mpsc::Receiver<Request>,
    replies: &'a mpsc::Sender<Reply>,
    /// An episode to rebuild on these rows first.
    resume: Option<Resume>,
}

/// Builds the environment and answers requests. The environment borrows the data here,
/// and goes before the data does.
fn serve<const K: usize, const M: usize>(job: Job<'_>) -> Option<Answered> {
    let Job {
        kernel: kernel_config,
        data,
        config,
        expected,
        ready,
        requests,
        replies,
        resume,
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
        check_spec(
            spec,
            id,
            expected,
            config.cancel_action,
            config.quote_lives_ns.is_some(),
        )?;
        let shape = spec.shape();
        let info = EnvInfo {
            spec: id,
            features: shape.features,
            offsets: shape.bid_offsets,
            sizes: shape.sizes,
            hedge_actions: spec.action.hedge == HedgeActions::MarketableAtTouch,
            cancel_action: spec.action.cancel == CancelActions::CancelOnly,
            quote_lives: quote_lives_of(&spec.action.expiry),
            kernel_fingerprint: env.kernel().config_fingerprint(),
            dataset_digest: data.id().digest,
            rows: data.id().len,
        };
        Ok((env, info))
    })();
    let (env, info) = match built {
        Ok((env, info)) => {
            if ready.send(Ok(info)).is_err() {
                return None;
            }
            (env, info)
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return None;
        }
    };
    Some(answer(env, info, data.ticks(), requests, replies, resume))
}

macro_rules! dispatch {
    ($job:ident; $(($k:literal, $($m:literal),*)),* $(,)?) => {
        match ($job.config.bid_offsets.len(), $job.config.sizes.len()) {
            $($(($k, $m) => serve::<$k, $m>($job),)*)*
            (offsets, sizes) => {
                let _ = $job.ready.send(Err(BridgeError::Refused(format!(
                    "a grid of {offsets} offsets and {sizes} sizes is not built in"
                ))));
                None
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
                // The first build answers the caller of spawn; a rebuild after a replace
                // answers through the request loop, so its readiness goes nowhere.
                let mut ready = Some(ready_tx);
                let mut rows = rows;
                let mut resume: Option<Resume> = None;
                loop {
                    let (sender, rebuilt) = match ready.take() {
                        Some(sender) => (sender, None),
                        None => {
                            let (sender, receiver) = mpsc::channel();
                            (sender, Some(receiver))
                        }
                    };
                    let attempt = resume.take();
                    let prepared = config.kernel().and_then(|kernel| {
                        KernelDataset::from_kernel_ticks(rows.clone(), &kernel)
                            .map(|data| (kernel, data))
                            .map_err(|error| BridgeError::Refused(error.to_string()))
                    });
                    let answered = match prepared {
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            None
                        }
                        Ok((kernel, data)) => {
                            let job = Job {
                                kernel: &kernel,
                                data: &data,
                                config: &config,
                                expected,
                                ready: sender,
                                requests: &request_rx,
                                replies: &reply_tx,
                                resume: attempt.clone(),
                            };
                            if let Some(residual) = &config.residual {
                                start_residual(job, residual)
                            } else {
                                dispatch!(
                                    job;
                                    (1, 1, 2, 3, 4),
                                    (2, 1, 2, 3, 4),
                                    (3, 1, 2, 3, 4),
                                    (4, 1, 2, 3, 4),
                                    (5, 1, 2, 3, 4),
                                    (8, 1, 2, 3, 4),
                                )
                            }
                        }
                    };
                    match (answered, rebuilt) {
                        (Some(Answered::Rebuild(next, next_resume)), _) => {
                            rows = next;
                            resume = Some(next_resume);
                        }
                        // The first build failed: spawn reports it.
                        (None, None) | (Some(Answered::Closed), _) => return,
                        // A rebuild on the new rows failed before any replay: answer the
                        // replace, and go back to the rows the episode ran on.
                        (None, Some(receiver)) => {
                            let error = match receiver.try_recv() {
                                Ok(Err(error)) => error.to_string(),
                                _ => "the environment could not be rebuilt".to_string(),
                            };
                            let Some(Resume {
                                seed,
                                log,
                                expect: Some(_),
                                fallback: Some(previous),
                            }) = attempt
                            else {
                                // Rebuilding the rows the episode ran on cannot fail.
                                return;
                            };
                            if reply_tx.send(Reply::Replaced(Err(error))).is_err() {
                                return;
                            }
                            rows = previous;
                            resume = Some(Resume {
                                seed,
                                log,
                                expect: None,
                                fallback: None,
                            });
                        }
                    }
                }
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
            Reply::Step(Ok(_)) | Reply::Attribution(Ok(_)) | Reply::Replaced(_) => {
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
            Reply::Reset(Ok(_)) | Reply::Step(Ok(_)) | Reply::Replaced(_) => {
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
            Reply::Reset(Ok(_)) | Reply::Attribution(Ok(_)) | Reply::Replaced(_) => {
                Err(self.fail("the wrong reply to a step".to_string()))
            }
        }
    }

    /// Replaces the rows under the running episode (#282). The worker rebuilds the
    /// environment on `rows`, resets it with the episode's seed and replays the episode's
    /// steps: every replayed step must give the reply the episode got, and the replay must
    /// reach the same state (apart from the dataset's identity), observation and masks. The
    /// new rows must also keep every row up to the one last observed. With an observation
    /// latency longer than a row, the kernel has matched rows past that one; changing them is
    /// refused by the replay, so a caller that reacts after the observed row is refused then.
    /// The digest of the new rows is the info's; the specification and the kernel's
    /// fingerprint stay. The replayed steps' counts are not reported again, and the
    /// attribution covers them.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::Refused`] before any reset, after the episode ended, for rows
    /// that change what was observed, rows the kernel refuses, or a replay that differs; the
    /// episode then goes on in the same state on the old rows (after a failed rebuild or
    /// replay, in an environment rebuilt on them). Otherwise as [`Worker::step`].
    pub fn replace_rows(&mut self, rows: Vec<KernelTick>) -> Result<EnvInfo, BridgeError> {
        match self.call(Request::ReplaceRows(rows))? {
            Reply::Replaced(Ok(info)) => {
                self.info = info;
                Ok(info)
            }
            Reply::Replaced(Err(reason)) => Err(BridgeError::Refused(reason)),
            Reply::Reset(Err(reason))
            | Reply::Step(Err(reason))
            | Reply::Attribution(Err(reason)) => Err(self.fail(reason)),
            Reply::Reset(Ok(_)) | Reply::Step(Ok(_)) | Reply::Attribution(Ok(_)) => {
                Err(self.fail("the wrong reply to a replace".to_string()))
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
