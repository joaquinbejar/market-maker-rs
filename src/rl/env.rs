//! The environment over one kernel.

use std::num::NonZeroU64;

use rust_decimal::prelude::ToPrimitive;

use crate::Decimal;
use crate::backtest::evaluation::{
    EvaluationError, EvaluationInput, ExecutedFill, FundingPayment, HedgeNotModelledReason,
    LedgerTotals, Mark, MetricId, MetricUnit, attribute,
};
use crate::backtest::kernel::{
    ActionGrid, ActionMask, ActionStatus, ErrCode, EventSink, FinancialKernel, HedgeEvent,
    HedgeInstrument, HedgeLane, HedgeLaneConfig, HedgeRefusal, HedgeView, HedgedAction,
    HedgedKernel, KernelAction, KernelConfig, KernelDataset, KernelEvent, KernelState, KernelTick,
    NoSink, Observation, QueueLengths, QuoteSide, Status, StepOutcome, TermReason,
};
use crate::execution::Action;
use crate::strategy::walk_forward::TimeWindow;
use crate::types::decimal::{exact_add, exact_product};
use crate::types::error::{MMError, MMResult};

use super::audit::{EpisodeRecorder, EpisodeRecords};
use super::features::{FeatureVector, HedgeFeatureVector, PreviousRow, feature_vector};
use super::spec::{
    ACTION_SPACE_VERSION, ActionSpace, CANCEL_ACTION_SPACE_VERSION, CancelActions, CrossHedgeSpace,
    EPISODE_VERSION, EXPIRY_ACTION_SPACE_VERSION, EpisodeSpec, HEDGED_ACTION_SPACE_VERSION,
    HEDGED_OBSERVATION_SPACE_VERSION, HEDGED_REWARD_VERSION, HedgeActions, HedgeScales,
    InvalidActionPolicy, ObservationScales, ObservationSpace, QuoteExpiry, QuoteLives,
    REWARD_VERSION, RewardKind, RewardMark, RewardSpec, RlSpec, TerminalValuation,
};

/// Code of an action whose indices are not on the grid, or whose price is not
/// representable, under [`InvalidActionPolicy::Reject`].
pub const ERR_ACTION_OFF_GRID: ErrCode = ErrCode(200);

/// Code of a hedge that is not available at this decision, under
/// [`InvalidActionPolicy::Reject`]: hedge actions are off, nothing has been observed, the
/// delivered position is flat, or the size is larger than that position.
pub const ERR_HEDGE_REFUSED: ErrCode = ErrCode(201);

/// Code of a hedge in the second instrument that is not available at this decision, under
/// [`InvalidActionPolicy::Reject`]: there is no hedge lane, the lane refuses one now (one
/// in flight, no row or a stale one), the net exposure is flat, or the size would take it
/// past zero.
pub const ERR_CROSS_HEDGE_REFUSED: ErrCode = ErrCode(202);

/// Code of a cancel when the action space has none ([`CancelActions::NotModelled`]),
/// under [`InvalidActionPolicy::Reject`].
pub const ERR_CANCEL_REFUSED: ErrCode = ErrCode(203);

/// Code of an expiring quote when the action space has no lives
/// ([`QuoteExpiry::NotModelled`]), under [`InvalidActionPolicy::Reject`].
pub const ERR_EXPIRY_REFUSED: ErrCode = ErrCode(204);

const NS_PER_MS: u64 = 1_000_000;

/// The choices of an environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RlConfig {
    /// The constants that normalize the features.
    pub scales: ObservationScales,
    /// What the `f32` reward is measured in; positive.
    pub f32_unit: Decimal,
    /// What happens outside the mask.
    pub invalid_action: InvalidActionPolicy,
    /// The charge for an action outside the mask; not negative.
    pub invalid_action_penalty: Decimal,
    /// What values the inventory on termination.
    pub terminal: TerminalValuation,
    /// Whether [`RlAction::Hedge`] exists, as [`HedgeActions::MarketableAtTouch`].
    pub hedge_actions: bool,
    /// Whether [`RlAction::Cancel`] exists, as [`CancelActions::CancelOnly`].
    pub cancel_action: bool,
    /// The lives of [`RlAction::ExpiringQuote`], as [`QuoteExpiry::Lives`].
    pub quote_lives: Option<QuoteLives>,
}

impl RlConfig {
    /// A configuration that treats an action outside the mask as an error, charges
    /// nothing and marks the last inventory at the last mid.
    #[must_use]
    pub fn new(scales: ObservationScales, f32_unit: Decimal) -> Self {
        Self {
            scales,
            f32_unit,
            invalid_action: InvalidActionPolicy::Mask,
            invalid_action_penalty: Decimal::ZERO,
            terminal: TerminalValuation::MarkAtLastMid,
            hedge_actions: false,
            cancel_action: false,
            quote_lives: None,
        }
    }

    /// Sets the policy for an action outside the mask, and its charge.
    #[must_use]
    pub fn with_invalid_action(mut self, policy: InvalidActionPolicy, penalty: Decimal) -> Self {
        self.invalid_action = policy;
        self.invalid_action_penalty = penalty;
        self
    }

    /// Sets what values the inventory on termination.
    #[must_use]
    pub fn with_terminal(mut self, terminal: TerminalValuation) -> Self {
        self.terminal = terminal;
        self
    }

    /// Adds [`RlAction::Hedge`] to the action space, as [`HedgeActions::MarketableAtTouch`].
    #[must_use]
    pub fn with_hedge_actions(mut self) -> Self {
        self.hedge_actions = true;
        self
    }

    /// Adds [`RlAction::Cancel`] to the action space, as [`CancelActions::CancelOnly`], of
    /// version [`CANCEL_ACTION_SPACE_VERSION`].
    #[must_use]
    pub fn with_cancel_action(mut self) -> Self {
        self.cancel_action = true;
        self
    }

    /// Adds [`RlAction::ExpiringQuote`] with these lives, of action space version
    /// [`EXPIRY_ACTION_SPACE_VERSION`], with or without the cancel action.
    #[must_use]
    pub fn with_quote_lives(mut self, lives: QuoteLives) -> Self {
        self.quote_lives = Some(lives);
        self
    }
}

/// One decision, as indices into the [`ActionGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RlAction {
    /// Leave the resting orders in place.
    Hold,
    /// Replace the resting orders with a bid and an ask.
    Quote {
        /// Index of the bid offset.
        bid: usize,
        /// Index of the ask offset.
        ask: usize,
        /// Index of the bid size.
        bid_size: usize,
        /// Index of the ask size.
        ask_size: usize,
    },
    /// Replace the resting orders with one order on `side`.
    OneSided {
        /// The side.
        side: QuoteSide,
        /// Index of the offset on that side.
        offset: usize,
        /// Index of the size.
        size: usize,
    },
    /// Reduce the delivered position with a marketable order at the observed touch, as
    /// [`HedgeActions::MarketableAtTouch`] states. It replaces the resting orders.
    Hedge {
        /// Index of the size in the grid's sizes, at most the delivered position at the decision.
        size: usize,
    },
    /// Reduce the net exposure of both lanes' delivered positions with an
    /// immediate-or-cancel order in the hedge instrument, as [`CrossHedgeSpace`] states. The
    /// quoted lane holds.
    CrossHedge {
        /// Index of the size in the hedge sizes, never taking the net exposure past zero.
        size: usize,
    },
    /// Cancel every resting or pending order and send nothing, as [`CancelActions::CancelOnly`]
    /// states. Outside the mask without it.
    Cancel,
    /// Replace the resting orders with a bid and an ask that expire by themselves, as
    /// [`QuoteExpiry::Lives`] states. Outside the mask without lives, or with `life` out of
    /// range.
    ExpiringQuote {
        /// Index of the bid offset.
        bid: usize,
        /// Index of the ask offset.
        ask: usize,
        /// Index of the bid size.
        bid_size: usize,
        /// Index of the ask size.
        ask_size: usize,
        /// Index of the life in [`RlConfig::quote_lives`].
        life: usize,
    },
}

/// The inventory closed on termination, under [`TerminalValuation::CloseAtTouch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TerminalClose {
    /// [`Action::Sell`] for a long position, [`Action::Buy`] for a short one.
    pub side: Action,
    /// The last row's bid or ask.
    pub price: Decimal,
    /// The whole position, positive.
    pub quantity: Decimal,
    /// `quantity * price * fee rate`; negative for a rebate.
    pub fee: Decimal,
    /// The size above what that touch displayed. Priced at the touch all the same.
    pub beyond_touch: Decimal,
    /// Time of the close, in nanoseconds: the last row's.
    pub ts_ns: u64,
}

/// The reward of one step, in the quote asset.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StepReward {
    /// The change of the net result in the kernel's ledger.
    pub ledger_change: Decimal,
    /// What closing the inventory changed; zero unless it was closed.
    pub terminal_close: Decimal,
    /// The charge for an action outside the mask. Not accounting.
    pub penalty: Decimal,
    /// `ledger_change + terminal_close - penalty`.
    pub total: Decimal,
    /// `total / f32_unit`. Derived: nothing reads it back.
    pub as_f32: f32,
}

/// What one step gave.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Transition {
    /// The observation after the step. On a termination it is from before the close,
    /// which is an assumption of the valuation and not something the agent was told of.
    pub observation: FeatureVector,
    /// The mask after the step.
    pub mask: ActionMask,
    /// [`RlEnv::hedge_mask`] after the step.
    pub hedge_mask: u64,
    /// The reward of the step.
    pub reward: StepReward,
    /// What the kernel did with the action.
    pub action_status: ActionStatus,
    /// The data ended.
    pub terminated: bool,
    /// `max_steps` decisions were taken before the data ended.
    pub truncated: bool,
    /// The close, on a termination that closed inventory.
    pub close: Option<TerminalClose>,
    /// [`RlEnv::cross_hedge_mask`] after the step; zero with no hedge lane.
    pub cross_hedge_mask: u64,
    /// The hedge features after the step, with a hedge lane.
    pub hedge_observation: Option<HedgeFeatureVector>,
    /// For a hedge in the second instrument that reached the lane, whether it was
    /// submitted.
    pub cross_hedge: Option<Result<(), HedgeRefusal>>,
    /// The hedge lane's close, on a termination that closed its inventory.
    pub hedge_close: Option<TerminalClose>,
}

/// The state of an episode: a `Copy` value, so a copy is a snapshot.
///
/// Only [`RlEnv::reset`] builds one, and only the environment changes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RlState<
    const ORD: usize = 16,
    const MSG: usize = 64,
    const DLV: usize = 128,
    const OBS: usize = 64,
> {
    kernel: KernelState<ORD, MSG, DLV, OBS>,
    previous: Option<PreviousRow>,
    mark_row: Option<u64>,
    net: Decimal,
    penalties: Decimal,
    close: Option<TerminalClose>,
    fault: bool,
    lane: Option<HedgeLane>,
    hedge_close: Option<(TerminalClose, Decimal)>,
}

impl<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    RlState<ORD, MSG, DLV, OBS>
{
    /// The kernel's state.
    ///
    /// Diagnostic: it holds venue progress, and a policy must not read it.
    #[must_use]
    pub fn kernel(&self) -> &KernelState<ORD, MSG, DLV, OBS> {
        &self.kernel
    }

    /// Whether `other` equals this state in every field but its kernel's dataset identity,
    /// as [`KernelState::same_apart_from_dataset`] (#282).
    #[must_use]
    pub fn same_apart_from_dataset(&self, other: &Self) -> bool {
        let mut moved = *self;
        moved.kernel = other.kernel;
        moved == *other && self.kernel.same_apart_from_dataset(&other.kernel)
    }

    /// Returns true once no more steps can be taken.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.fault || !self.kernel.status().is_running()
    }

    /// The net result so far, close included and penalties excluded. It is the sum of
    /// [`StepReward::ledger_change`] and [`StepReward::terminal_close`].
    #[must_use]
    pub fn net(&self) -> Decimal {
        self.net
    }

    /// The charges for actions outside the mask so far.
    #[must_use]
    pub fn penalties(&self) -> Decimal {
        self.penalties
    }

    /// The terminal close, if there was one.
    #[must_use]
    pub fn close(&self) -> Option<TerminalClose> {
        self.close
    }

    /// The hedge lane's state, with a hedge lane.
    ///
    /// Diagnostic: it holds venue progress, and a policy must not read it.
    #[must_use]
    pub fn lane(&self) -> Option<&HedgeLane> {
        self.lane.as_ref()
    }

    /// The hedge lane's terminal close, if there was one.
    #[must_use]
    pub fn hedge_close(&self) -> Option<TerminalClose> {
        self.hedge_close.map(|(close, _)| close)
    }
}

/// Forwards every event, and notes the last completed row.
struct Tap<'a, E> {
    inner: &'a mut E,
    row: Option<u64>,
}

impl<E: EventSink> EventSink for Tap<'_, E> {
    fn event(&mut self, event: KernelEvent) {
        if let KernelEvent::RowEnd { row, .. } = event {
            self.row = Some(row);
        }
        self.inner.event(event);
    }

    fn queues(&mut self, lengths: QueueLengths) {
        self.inner.queues(lengths);
    }

    fn hedge(&mut self, event: HedgeEvent) {
        self.inner.hedge(event);
    }
}

fn inexact(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} is not exact in a Decimal"))
}

fn add(a: Decimal, b: Decimal, what: &str) -> MMResult<Decimal> {
    exact_add(a, b).ok_or_else(|| inexact(what))
}

fn negated(mut value: Decimal) -> Decimal {
    value.set_sign_negative(!value.is_sign_negative());
    value
}

fn sub(a: Decimal, b: Decimal, what: &str) -> MMResult<Decimal> {
    add(a, negated(b), what)
}

fn mul(a: Decimal, b: Decimal, what: &str) -> MMResult<Decimal> {
    exact_product(a, b).ok_or_else(|| inexact(what))
}

fn bit(mask: u64, index: usize) -> bool {
    u32::try_from(index)
        .ok()
        .and_then(|i| mask.checked_shr(i))
        .is_some_and(|v| v & 1 == 1)
}

/// Returns true if `action` is inside `mask`, `hedge`, `cross`, `cancel` and the `lives`
/// available, while the run continues.
fn allowed(
    mask: &ActionMask,
    hedge: u64,
    cross: u64,
    cancel: bool,
    lives: usize,
    action: RlAction,
) -> bool {
    match action {
        RlAction::Hold => true,
        RlAction::Cancel => cancel,
        RlAction::Hedge { size } => bit(hedge, size),
        RlAction::CrossHedge { size } => bit(cross, size),
        RlAction::ExpiringQuote {
            bid,
            ask,
            bid_size,
            ask_size,
            life,
        } => life < lives && mask.allows(bid, ask, bid_size, ask_size),
        RlAction::Quote {
            bid,
            ask,
            bid_size,
            ask_size,
        } => mask.allows(bid, ask, bid_size, ask_size),
        RlAction::OneSided { side, offset, size } => {
            let offsets = match side {
                QuoteSide::Bid => mask.bid,
                QuoteSide::Ask => mask.ask,
            };
            mask.global_quote_ok && bit(offsets, offset) && bit(mask.size, size)
        }
    }
}

/// The environment. See the [module documentation](super).
///
/// `K` and `M` are the offsets per side and the sizes of the grid.
#[derive(Debug, Clone)]
pub struct RlEnv<
    'd,
    const K: usize,
    const M: usize,
    const ORD: usize = 16,
    const MSG: usize = 64,
    const DLV: usize = 128,
    const OBS: usize = 64,
> {
    kernel: FinancialKernel<'d, ORD, MSG, DLV, OBS>,
    grid: ActionGrid<K, M>,
    config: RlConfig,
    fee_rate: Decimal,
    spec: RlSpec,
    cross: Option<Cross<'d, ORD, MSG, DLV, OBS>>,
}

/// The hedge lane of an environment.
#[derive(Debug, Clone)]
struct Cross<'d, const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize> {
    kernel: HedgedKernel<'d, ORD, MSG, DLV, OBS>,
    sizes: Vec<u64>,
    multiplier: Decimal,
    tick: Decimal,
    lot: Decimal,
    fee_rate: Decimal,
}

/// What an action becomes.
#[derive(Debug, Clone, Copy)]
enum Resolved {
    Kernel(KernelAction),
    Cross(QuoteSide, u64),
}

impl<
    'd,
    const K: usize,
    const M: usize,
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
> RlEnv<'d, K, M, ORD, MSG, DLV, OBS>
{
    /// Builds an environment over a new kernel.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a scale or `f32_unit` that is not
    /// positive, a negative penalty, zero `max_steps`, and for whatever
    /// [`FinancialKernel::new`] refuses.
    pub fn new(
        kernel_config: &KernelConfig,
        data: &'d KernelDataset,
        grid: ActionGrid<K, M>,
        config: RlConfig,
    ) -> MMResult<Self> {
        config.scales.validate()?;
        if config.f32_unit <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "f32_unit must be positive, got {}",
                config.f32_unit
            )));
        }
        if config.invalid_action_penalty < Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid_action_penalty must not be negative, got {}",
                config.invalid_action_penalty
            )));
        }
        if kernel_config.max_steps == 0 {
            return Err(MMError::InvalidConfiguration(
                "max_steps must be positive, got 0".to_string(),
            ));
        }
        let kernel = FinancialKernel::new(kernel_config, data)?;
        Self::assemble(kernel, kernel_config, grid, config)
    }

    /// The environment over a kernel already built from `kernel_config`.
    fn assemble(
        kernel: FinancialKernel<'d, ORD, MSG, DLV, OBS>,
        kernel_config: &KernelConfig,
        grid: ActionGrid<K, M>,
        config: RlConfig,
    ) -> MMResult<Self> {
        let orders = Decimal::from(u64::try_from(ORD).unwrap_or(u64::MAX));
        let spec = RlSpec {
            observation: ObservationSpace::new(
                &config.scales,
                orders,
                Decimal::from(kernel_config.max_steps),
            ),
            action: ActionSpace {
                version: if config.quote_lives.is_some() {
                    EXPIRY_ACTION_SPACE_VERSION
                } else if config.cancel_action {
                    CANCEL_ACTION_SPACE_VERSION
                } else {
                    ACTION_SPACE_VERSION
                },
                tick_size: kernel_config.tick_size,
                lot_size: kernel_config.lot_size,
                max_price: kernel_config.max_price,
                max_order_size: kernel_config.max_order_size,
                bid_offsets_ticks: grid.bid_offsets().to_vec(),
                ask_offsets_ticks: grid.ask_offsets().to_vec(),
                sizes_lots: grid.sizes().to_vec(),
                invalid_action: config.invalid_action,
                hedge: if config.hedge_actions {
                    HedgeActions::MarketableAtTouch
                } else {
                    HedgeActions::NotModelled {
                        reason: HedgeNotModelledReason::SingleInstrument,
                    }
                },
                cross_hedge: None,
                cancel: if config.cancel_action {
                    CancelActions::CancelOnly
                } else {
                    CancelActions::NotModelled
                },
                expiry: match &config.quote_lives {
                    Some(lives) => QuoteExpiry::Lives(lives.as_slice().to_vec()),
                    None => QuoteExpiry::NotModelled,
                },
            },
            reward: RewardSpec {
                version: REWARD_VERSION,
                kind: RewardKind::NetEquityChange,
                metric: MetricId::NetPnl,
                unit: MetricUnit::Quote,
                mark: RewardMark::LastCompletedRow,
                invalid_action_penalty: config.invalid_action_penalty,
                f32_unit: config.f32_unit,
                policy_input: false,
            },
            episode: EpisodeSpec {
                version: EPISODE_VERSION,
                max_steps: kernel_config.max_steps,
                terminal: config.terminal,
            },
        };
        Ok(Self {
            kernel,
            grid,
            config,
            fee_rate: kernel_config.fee_rate,
            spec,
            cross: None,
        })
    }

    /// Builds an environment over a kernel with a hedge lane on a second instrument
    /// ([`HedgedKernel`]), with the hedge sizes in hedge lots and the hedge features'
    /// scales. Its specification is of observation version
    /// [`HEDGED_OBSERVATION_SPACE_VERSION`], action version [`HEDGED_ACTION_SPACE_VERSION`]
    /// and reward version [`HEDGED_REWARD_VERSION`]; an environment without a lane keeps
    /// the versions before them, and its identity.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::new`], and for whatever [`HedgedKernel::new`] refuses, a hedge scale
    /// that is not positive, or no hedge sizes, more than 64 or one of zero.
    #[allow(clippy::too_many_arguments)]
    pub fn new_hedged(
        kernel_config: &KernelConfig,
        data: &'d KernelDataset,
        lane: &HedgeLaneConfig,
        hedge_data: &'d KernelDataset,
        grid: ActionGrid<K, M>,
        hedge_sizes: Vec<u64>,
        hedge_scales: HedgeScales,
        config: RlConfig,
    ) -> MMResult<Self> {
        hedge_scales.validate()?;
        if hedge_sizes.is_empty() || hedge_sizes.len() > 64 || hedge_sizes.contains(&0) {
            return Err(MMError::InvalidConfiguration(format!(
                "hedge sizes must be 1 to 64 positive lots, got {hedge_sizes:?}"
            )));
        }
        if config.cancel_action {
            return Err(MMError::InvalidConfiguration(
                "the cancel action is not available with a hedge lane".to_string(),
            ));
        }
        if config.quote_lives.is_some() {
            return Err(MMError::InvalidConfiguration(
                "quote lives are not available with a hedge lane".to_string(),
            ));
        }
        let mut env = Self::new(kernel_config, data, grid, config)?;
        let hedged = HedgedKernel::new(kernel_config, data, lane, hedge_data)?;
        env.spec.observation.version = HEDGED_OBSERVATION_SPACE_VERSION;
        env.spec.observation.hedge_features = hedge_scales.features();
        env.spec.action.version = HEDGED_ACTION_SPACE_VERSION;
        env.spec.action.cross_hedge = Some(CrossHedgeSpace {
            tick_size: lane.instrument.tick_size,
            lot_size: lane.instrument.lot_size,
            max_order_size: lane.instrument.max_order_size,
            multiplier: lane.multiplier,
            instrument: match lane.kind {
                HedgeInstrument::Spot => 1,
                _ => 2,
            },
            max_staleness_ns: lane.max_staleness_ns,
            sizes_lots: hedge_sizes.clone(),
        });
        env.spec.reward.version = HEDGED_REWARD_VERSION;
        env.cross = Some(Cross {
            kernel: hedged,
            sizes: hedge_sizes,
            multiplier: lane.multiplier,
            tick: lane.instrument.tick_size,
            lot: lane.instrument.lot_size,
            fee_rate: lane.instrument.fee_rate,
        });
        Ok(env)
    }

    /// What the agent sees of the hedge lane, with one. The lanes' states are read in
    /// place: the kernel's is large, and a copy of it in each frame would fill the stack.
    fn hedge_view(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> Option<HedgeView> {
        let cross = self.cross.as_ref()?;
        Some(cross.kernel.view_parts(&state.kernel, state.lane.as_ref()?))
    }

    /// The net exposure of both lanes' delivered positions, in quoted base units.
    fn net_exposure(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
        observed: &Observation,
    ) -> Option<Decimal> {
        let cross = self.cross.as_ref()?;
        let view = self.hedge_view(state)?;
        observed
            .position
            .checked_add(view.position.checked_mul(cross.multiplier)?)
    }

    /// The side and lots of a hedge of the size at `index` in the second instrument, if it
    /// reduces the net exposure without taking it past zero.
    fn cross_order(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
        observed: &Observation,
        index: usize,
    ) -> Option<(QuoteSide, u64)> {
        let cross = self.cross.as_ref()?;
        let lots = *cross.sizes.get(index)?;
        let net = self.net_exposure(state, observed)?;
        let exposure = Decimal::from(lots)
            .checked_mul(cross.lot)?
            .checked_mul(cross.multiplier)?;
        if net.is_zero() || exposure > net.abs() {
            return None;
        }
        let side = if net.is_sign_positive() {
            QuoteSide::Ask
        } else {
            QuoteSide::Bid
        };
        Some((side, lots))
    }

    /// The sizes [`RlAction::CrossHedge`] may take at `state`, as bits over the hedge sizes:
    /// those the lane would take now (no hedge in flight, a fresh hedge row seen) that reduce
    /// the net exposure of both lanes' delivered positions without taking it past zero.
    /// Zero with no hedge lane. It reads no risk limit: the lane's admission, its own limits
    /// and the combined limits decide on the order.
    #[must_use]
    pub fn cross_hedge_mask(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> u64 {
        let (Some(cross), Some(lane)) = (self.cross.as_ref(), state.lane.as_ref()) else {
            return 0;
        };
        if state.is_done() {
            return 0;
        }
        let lane = cross.kernel.mask_parts(&state.kernel, lane, &cross.sizes);
        let observed = self.kernel.observe(&state.kernel);
        let mut bits = 0_u64;
        for index in 0..cross.sizes.len() {
            if bit(lane, index) && self.cross_order(state, &observed, index).is_some() {
                let one = u32::try_from(index)
                    .ok()
                    .and_then(|shift| 1_u64.checked_shl(shift))
                    .unwrap_or(0);
                bits |= one;
            }
        }
        bits
    }

    /// The hedge features at `state`, with a hedge lane.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for a state of another kernel, and
    /// [`MMError::NumericalError`] if a value does not convert.
    pub fn observe_hedge(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
    ) -> MMResult<Option<HedgeFeatureVector>> {
        self.check_owned(state)?;
        let (Some(cross), Some(view)) = (self.cross.as_ref(), self.hedge_view(state)) else {
            return Ok(None);
        };
        let numeric = |what: &str| MMError::NumericalError(format!("{what} does not convert"));
        let tick = view.tick;
        let spread = tick
            .map(|t| t.ask.checked_sub(t.bid).map(Decimal::from))
            .map(|v| v.ok_or_else(|| numeric("hedge spread")))
            .transpose()?;
        let hedge_mid = tick
            .map(|t| {
                t.mid2()
                    .map(Decimal::from)
                    .and_then(|m| m.checked_mul(cross.tick))
                    .and_then(|m| m.checked_div(Decimal::TWO))
                    .ok_or_else(|| numeric("hedge mid"))
            })
            .transpose()?;
        let observed = self.kernel.observe(&state.kernel);
        let quoted_mid = match observed.row {
            Some(_) => Some(
                self.mid_of(
                    observed
                        .bid
                        .checked_add(observed.ask)
                        .ok_or_else(|| numeric("mid"))?,
                )?,
            ),
            None => None,
        };
        let basis = match (hedge_mid, quoted_mid) {
            (Some(h), Some(q)) => Some(h.checked_sub(q).ok_or_else(|| numeric("basis"))?),
            _ => None,
        };
        let age = tick
            .map(|t| {
                let ns = state.kernel.now_ns().saturating_sub(t.ts_ns);
                Decimal::try_from_i128_with_scale(i128::from(ns), 6)
                    .map_err(|_| numeric("hedge row age"))
            })
            .transpose()?;
        let position = view
            .position
            .checked_mul(cross.multiplier)
            .ok_or_else(|| numeric("hedge position"))?;
        let raw = [
            spread,
            basis,
            age,
            Some(position),
            Some(Decimal::from(u8::from(view.in_flight))),
        ];
        HedgeFeatureVector::new(view.row, raw, &self.spec.observation.hedge_features).map(Some)
    }

    /// The specification of the spaces.
    #[must_use]
    pub fn spec(&self) -> &RlSpec {
        &self.spec
    }

    /// The kernel.
    #[must_use]
    pub fn kernel(&self) -> &FinancialKernel<'d, ORD, MSG, DLV, OBS> {
        &self.kernel
    }

    /// The state at the first decision, or at the end of the run if the data has none.
    ///
    /// # Errors
    ///
    /// Returns an error if the first decision's ledger does not convert, which the
    /// kernel's bounds exclude.
    pub fn reset(&self, seed: u64) -> MMResult<RlState<ORD, MSG, DLV, OBS>> {
        self.reset_with(seed, &mut NoSink)
    }

    /// As [`RlEnv::reset`], sending the kernel's events to `sink`.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::reset`].
    pub fn reset_with<E: EventSink>(
        &self,
        seed: u64,
        sink: &mut E,
    ) -> MMResult<RlState<ORD, MSG, DLV, OBS>> {
        let mut tap = Tap {
            inner: sink,
            row: None,
        };
        if let Some(cross) = &self.cross {
            return self.reset_hedged(cross, seed, tap);
        }
        let kernel = self.kernel.reset_with(seed, &mut tap);
        let mut state = RlState {
            kernel,
            previous: None,
            mark_row: tap.row,
            net: Decimal::ZERO,
            penalties: Decimal::ZERO,
            close: None,
            fault: false,
            lane: None,
            hedge_close: None,
        };
        state.net = self.net_of(&state)?;
        Ok(state)
    }

    /// [`RlEnv::reset_with`] with a hedge lane. Apart, and never inlined, so that an
    /// environment without one does not hold this frame's copies of the kernel's state,
    /// which is large, on its stack.
    #[inline(never)]
    fn reset_hedged<E: EventSink>(
        &self,
        cross: &Cross<'d, ORD, MSG, DLV, OBS>,
        seed: u64,
        mut tap: Tap<'_, E>,
    ) -> MMResult<RlState<ORD, MSG, DLV, OBS>> {
        let both = cross.kernel.reset_with(seed, &mut tap);
        let mut state = RlState {
            kernel: both.kernel,
            previous: None,
            mark_row: tap.row,
            net: Decimal::ZERO,
            penalties: Decimal::ZERO,
            close: None,
            fault: false,
            lane: Some(both.lane),
            hedge_close: None,
        };
        state.net = self.net_of(&state)?;
        Ok(state)
    }

    /// The observation at `state`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for a state of another kernel, and
    /// [`MMError::NumericalError`] if a value does not convert.
    pub fn observe(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> MMResult<FeatureVector> {
        self.check_owned(state)?;
        let obs = self.kernel.observe(&state.kernel);
        let mid = match obs.row {
            Some(_) => {
                let mid2 = obs
                    .bid
                    .checked_add(obs.ask)
                    .ok_or_else(|| inexact("observed mid"))?;
                Some(self.mid_of(mid2)?)
            }
            None => None,
        };
        feature_vector(
            &self.spec.observation,
            &obs,
            state.previous,
            mid,
            self.spec.episode.max_steps,
        )
    }

    /// The mask at `state`. It reads no risk limit.
    #[must_use]
    pub fn mask(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> ActionMask {
        self.kernel.mask(&state.kernel, &self.grid)
    }

    /// The sizes [`RlAction::Hedge`] may take at `state`, as bits over the grid's sizes:
    /// those the [`mask`](Self::mask) allows that are no larger than the delivered
    /// position. Zero when hedge actions are off, nothing has been observed, the position
    /// is flat or the mask allows no order. It reads no risk limit: the kernel's admission
    /// and its gate decide on the order as on any other.
    #[must_use]
    pub fn hedge_mask(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> u64 {
        if !self.config.hedge_actions || state.is_done() {
            return 0;
        }
        let mask = self.mask(state);
        let observed = self.kernel.observe(&state.kernel);
        if !mask.global_quote_ok || observed.row.is_none() {
            return 0;
        }
        let mut bits = 0_u64;
        for (index, lots) in self.grid.sizes().iter().enumerate() {
            if bit(mask.size, index) && self.reduces(&observed, *lots) {
                let one = u32::try_from(index)
                    .ok()
                    .and_then(|shift| 1_u64.checked_shl(shift))
                    .unwrap_or(0);
                bits |= one;
            }
        }
        bits
    }

    /// Returns true if `lots` is a positive quantity no larger than the delivered position.
    fn reduces(&self, observed: &Observation, lots: u64) -> bool {
        self.kernel
            .quantity(lots)
            .is_some_and(|quantity| quantity > Decimal::ZERO && quantity <= observed.position.abs())
    }

    /// Takes one decision.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::step_with`].
    pub fn step(
        &self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        action: RlAction,
    ) -> MMResult<Transition> {
        self.step_with(state, action, &mut NoSink)
    }

    /// As [`RlEnv::step`], sending the kernel's events to `sink`.
    ///
    /// # Errors
    ///
    /// Returns, with the state unchanged, [`MMError::InvalidMarketState`] for a state of
    /// another kernel, or for one whose episode ended or failed, and
    /// [`MMError::InvalidQuoteGeneration`] for an action outside the mask under
    /// [`InvalidActionPolicy::Mask`]. Returns [`MMError::NumericalError`] if the kernel
    /// halts or the ledger does not convert, after which the state refuses every step.
    pub fn step_with<E: EventSink>(
        &self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        action: RlAction,
        sink: &mut E,
    ) -> MMResult<Transition> {
        self.check_owned(state)?;
        if state.is_done() {
            return Err(MMError::InvalidMarketState(
                "the episode has ended".to_string(),
            ));
        }
        let inside = allowed(
            &self.mask(state),
            self.hedge_mask(state),
            self.cross_hedge_mask(state),
            self.config.cancel_action,
            self.config
                .quote_lives
                .map_or(0, |lives| lives.as_slice().len()),
            action,
        );
        if !inside && self.config.invalid_action == InvalidActionPolicy::Mask {
            return Err(MMError::InvalidQuoteGeneration(format!(
                "{action:?} is outside the mask"
            )));
        }
        let decision = self.kernel.observe(&state.kernel);
        let resolved = match action {
            RlAction::CrossHedge { size } => {
                match (inside, self.cross_order(state, &decision, size)) {
                    (true, Some((side, lots))) => Resolved::Cross(side, lots),
                    _ => Resolved::Kernel(KernelAction::QuoteError(ERR_CROSS_HEDGE_REFUSED)),
                }
            }
            _ => Resolved::Kernel(self.resolve(&decision, state, action)),
        };
        let penalty = if inside {
            Decimal::ZERO
        } else {
            self.config.invalid_action_penalty
        };
        self.apply(state, &decision, resolved, penalty, sink)
    }

    /// Steps the kernel with a quote composed outside the grid. Its admission and its gate
    /// decide as for any other action, and nothing is charged.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::step_with`], with no mask to leave.
    pub(super) fn step_composed_with<E: EventSink>(
        &self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        action: KernelAction,
        sink: &mut E,
    ) -> MMResult<Transition> {
        self.check_owned(state)?;
        if state.is_done() {
            return Err(MMError::InvalidMarketState(
                "the episode has ended".to_string(),
            ));
        }
        let decision = self.kernel.observe(&state.kernel);
        self.apply(
            state,
            &decision,
            Resolved::Kernel(action),
            Decimal::ZERO,
            sink,
        )
    }

    fn apply<E: EventSink>(
        &self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        decision: &Observation,
        resolved: Resolved,
        penalty: Decimal,
        sink: &mut E,
    ) -> MMResult<Transition> {
        let mut tap = Tap {
            inner: sink,
            row: None,
        };
        let (outcome, cross_hedge) = match (&self.cross, state.lane.as_mut()) {
            (Some(cross), Some(lane)) => {
                let action = match resolved {
                    Resolved::Kernel(action) => HedgedAction::Kernel(action),
                    Resolved::Cross(side, lots) => HedgedAction::Hedge { side, lots },
                };
                let out = cross
                    .kernel
                    .step_parts(&mut state.kernel, lane, action, &mut tap);
                (out.kernel, out.hedge)
            }
            _ => {
                let action = match resolved {
                    Resolved::Kernel(action) => action,
                    // A hedge in the second instrument resolves only with a lane.
                    Resolved::Cross(..) => KernelAction::QuoteError(ERR_CROSS_HEDGE_REFUSED),
                };
                (
                    self.kernel.step_with(&mut state.kernel, action, &mut tap),
                    None,
                )
            }
        };
        if let Some(row) = tap.row {
            state.mark_row = Some(row);
        }
        if decision.row.is_some()
            && let Some(mid2) = decision.bid.checked_add(decision.ask)
        {
            state.previous = Some(PreviousRow {
                mid2,
                ts_ns: decision.ts_ns,
            });
        }
        let settled = self.settle(state, penalty, outcome, cross_hedge);
        if settled.is_err() {
            state.fault = true;
        }
        settled
    }

    /// An episode's records in the form of the evaluation; with a hedge lane, its records
    /// too, its terminal close included, and a window that covers both lanes (#266).
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] with no completed row or with a hedge
    /// row that has no positive mid, and [`EvaluationError::Inexact`] if a value does
    /// not convert.
    pub fn records(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
        recorder: &EpisodeRecorder,
    ) -> Result<EpisodeRecords, EvaluationError> {
        let convert = |what| EvaluationError::Inexact { what };
        let ticks = self.kernel.data().ticks();
        let mut marks = Vec::with_capacity(recorder.rows.len());
        for row in &recorder.rows {
            let tick = usize::try_from(*row)
                .ok()
                .and_then(|i| ticks.get(i))
                .ok_or(EvaluationError::InvalidInput {
                    reason: format!("row {row} is not in the data"),
                })?;
            let mid = tick
                .mid2()
                .and_then(|mid2| self.mid_of(mid2).ok())
                .ok_or(convert("mark"))?;
            marks.push(Mark {
                timestamp_ms: tick.ts_ns.checked_div(NS_PER_MS).unwrap_or(0),
                mid,
            });
        }
        let mut fills = Vec::with_capacity(recorder.fills.len().saturating_add(1));
        for fill in &recorder.fills {
            fills.push(ExecutedFill {
                timestamp_ms: fill.ts_ns.checked_div(NS_PER_MS).unwrap_or(0),
                side: fill.side,
                price: self.kernel.price(fill.price).ok_or(convert("fill price"))?,
                quantity: self.kernel.quantity(fill.qty).ok_or(convert("fill size"))?,
                quote_fee: self.kernel.money(fill.fee_u).ok_or(convert("fill fee"))?,
                base_fee: Decimal::ZERO,
            });
        }
        if let Some(close) = state.close {
            fills.push(ExecutedFill {
                timestamp_ms: close.ts_ns.checked_div(NS_PER_MS).unwrap_or(0),
                side: close.side,
                price: close.price,
                quantity: close.quantity,
                quote_fee: close.fee,
                base_fee: Decimal::ZERO,
            });
        }
        let (hedge_marks, hedge_fills, hedge_funding) = self.hedge_records(state, recorder)?;
        let Some(first_mark) = marks.first().map(|m| m.timestamp_ms) else {
            return Err(EvaluationError::InvalidInput {
                reason: "the episode has no completed row".to_string(),
            });
        };
        let times = marks
            .iter()
            .map(|m| m.timestamp_ms)
            .chain(fills.iter().map(|f| f.timestamp_ms))
            .chain(hedge_marks.iter().map(|m| m.timestamp_ms))
            .chain(hedge_fills.iter().map(|f| f.timestamp_ms))
            .chain(hedge_funding.iter().map(|f| f.timestamp_ms));
        let (start_ms, last_ms) = times.fold((first_mark, first_mark), |(lo, hi), t| {
            (lo.min(t), hi.max(t))
        });
        let end_ms = last_ms.checked_add(1).ok_or(convert("window"))?;
        Ok(EpisodeRecords {
            window: TimeWindow { start_ms, end_ms },
            marks,
            fills,
            hedge_marks,
            hedge_fills,
            hedge_funding,
        })
    }

    /// The hedge lane's records, its terminal close included; empty without a lane.
    #[allow(clippy::type_complexity)]
    fn hedge_records(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
        recorder: &EpisodeRecorder,
    ) -> Result<(Vec<Mark>, Vec<ExecutedFill>, Vec<FundingPayment>), EvaluationError> {
        let convert = |what| EvaluationError::Inexact { what };
        let Some(cross) = &self.cross else {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        };
        let scale = cross.kernel.lane_money_scale();
        let money = |units: i128| {
            Decimal::try_from_i128_with_scale(units, scale).map_err(|_| convert("hedge money"))
        };
        let ms = |ns: u64| ns.checked_div(NS_PER_MS).unwrap_or(0);
        let (mut marks, mut fills, mut funding) = (Vec::new(), Vec::new(), Vec::new());
        for event in &recorder.hedge {
            match *event {
                HedgeEvent::Row { row, ts_ns, mid2 } => {
                    let mid2 = mid2.ok_or_else(|| EvaluationError::InvalidInput {
                        reason: format!("hedge row {row} has no positive mid"),
                    })?;
                    let mid = Decimal::from(mid2)
                        .checked_mul(cross.tick)
                        .and_then(|m| m.checked_mul(Decimal::new(5, 1)))
                        .ok_or(convert("hedge mark"))?;
                    marks.push(Mark {
                        timestamp_ms: ms(ts_ns),
                        mid,
                    });
                }
                HedgeEvent::Fill {
                    ts_ns,
                    side,
                    price,
                    lots,
                    fee_u,
                } => fills.push(ExecutedFill {
                    timestamp_ms: ms(ts_ns),
                    side,
                    price: Decimal::from(price)
                        .checked_mul(cross.tick)
                        .ok_or(convert("hedge fill price"))?,
                    quantity: Decimal::from(lots)
                        .checked_mul(cross.lot)
                        .ok_or(convert("hedge fill size"))?,
                    quote_fee: money(fee_u)?,
                    base_fee: Decimal::ZERO,
                }),
                HedgeEvent::Funding {
                    at_ns, received_u, ..
                } => funding.push(FundingPayment {
                    timestamp_ms: ms(at_ns),
                    amount: money(received_u)?,
                }),
            }
        }
        if let Some((close, _)) = state.hedge_close {
            fills.push(ExecutedFill {
                timestamp_ms: ms(close.ts_ns),
                side: close.side,
                price: close.price,
                quantity: close.quantity,
                quote_fee: close.fee,
                base_fee: Decimal::ZERO,
            });
        }
        Ok((marks, fills, funding))
    }

    /// Checks the reward against the evaluation: [`RlState::net`] must equal the net
    /// result of [`attribute`] over the same marks and fills to the last digit, and the
    /// position left must agree. Holds at every decision, not only at the end.
    ///
    /// With a hedge lane, each lane is checked against its own records (#266): the
    /// hedge lane's attribution against its ledger's net, its close included, and its
    /// final position against the lane's; the quoted lane's against the rest of the net.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::NotReconciled`] when they differ, and the errors of
    /// [`RlEnv::records`] and [`attribute`].
    pub fn reconcile(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
        recorder: &EpisodeRecorder,
    ) -> Result<(), EvaluationError> {
        let records = self.records(state, recorder)?;
        // The attribution does not read the capital, and a kernel may start with none.
        let input =
            EvaluationInput::new(records.window, &records.marks, &records.fills, Decimal::ONE)?;
        let position = match state.close {
            Some(_) => Decimal::ZERO,
            None => {
                self.kernel
                    .venue_summary(&state.kernel)
                    .map_err(|error| EvaluationError::InvalidInput {
                        reason: error.to_string(),
                    })?
                    .position
            }
        };
        // With a hedge lane, each lane is attributed against its own records (#266): the
        // hedge lane's net, its close included, is its ledger's, checked by its shadow,
        // and the quoted lane's is the rest of the net.
        let lane = self
            .lane_net(state)
            .ok()
            .and_then(|net| net.checked_add(state.hedge_close.map_or(Decimal::ZERO, |(_, c)| c)))
            .ok_or(EvaluationError::Inexact {
                what: "hedge lane net",
            })?;
        let quoted = LedgerTotals {
            net_pnl: state
                .net
                .checked_sub(lane)
                .ok_or(EvaluationError::Inexact { what: "quoted net" })?,
            position,
        };
        let (Some(cross), Some(hedge_lane)) = (&self.cross, state.lane.as_ref()) else {
            return attribute(&input)?.reconcile(quoted);
        };
        let hedge_position = match state.hedge_close {
            Some(_) => Decimal::ZERO,
            None => {
                cross
                    .kernel
                    .summary_parts(&state.kernel, hedge_lane)
                    .map_err(|error| EvaluationError::InvalidInput {
                        reason: error.to_string(),
                    })?
                    .position
            }
        };
        let input = input.with_hedge(
            &records.hedge_marks,
            &records.hedge_fills,
            &records.hedge_funding,
        )?;
        attribute(&input)?.reconcile_lanes(
            quoted,
            LedgerTotals {
                net_pnl: lane,
                position: hedge_position,
            },
        )
    }

    fn check_owned(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> MMResult<()> {
        let owned = match (&self.cross, state.lane.as_ref()) {
            (None, _) => self.kernel.owns(&state.kernel) && state.lane.is_none(),
            (Some(cross), Some(lane)) => cross.kernel.owns_parts(&state.kernel, lane),
            (Some(_), None) => false,
        };
        if owned {
            return Ok(());
        }
        Err(MMError::InvalidMarketState(
            "a state from another dataset or configuration".to_string(),
        ))
    }

    /// The kernel's action for `action`. Off the grid it is a recorded error.
    fn resolve(
        &self,
        decision: &Observation,
        state: &RlState<ORD, MSG, DLV, OBS>,
        action: RlAction,
    ) -> KernelAction {
        let off_grid = KernelAction::QuoteError(ERR_ACTION_OFF_GRID);
        match action {
            RlAction::Hold => KernelAction::Hold,
            RlAction::Cancel if self.config.cancel_action => KernelAction::Cancel,
            RlAction::Cancel => KernelAction::QuoteError(ERR_CANCEL_REFUSED),
            RlAction::ExpiringQuote {
                bid,
                ask,
                bid_size,
                ask_size,
                life,
            } => {
                let Some(lives) = self.config.quote_lives else {
                    return KernelAction::QuoteError(ERR_EXPIRY_REFUSED);
                };
                let Some(life_ns) = lives
                    .as_slice()
                    .get(life)
                    .copied()
                    .and_then(NonZeroU64::new)
                else {
                    return off_grid;
                };
                match self.kernel.grid_action(
                    &state.kernel,
                    &self.grid,
                    (bid, ask, bid_size, ask_size),
                ) {
                    Some(KernelAction::Quote {
                        bid,
                        ask,
                        bid_lots,
                        ask_lots,
                    }) => KernelAction::ExpiringQuote {
                        bid,
                        ask,
                        bid_lots,
                        ask_lots,
                        life_ns,
                    },
                    _ => off_grid,
                }
            }
            RlAction::CrossHedge { .. } => KernelAction::QuoteError(ERR_CROSS_HEDGE_REFUSED),
            RlAction::Hedge { size } => {
                let refused = KernelAction::QuoteError(ERR_HEDGE_REFUSED);
                let Some(lots) = self.grid.sizes().get(size).copied() else {
                    return off_grid;
                };
                if !self.config.hedge_actions
                    || decision.row.is_none()
                    || !self.reduces(decision, lots)
                {
                    return refused;
                }
                if decision.position.is_sign_positive() {
                    KernelAction::OneSided {
                        side: QuoteSide::Ask,
                        price: decision.bid,
                        lots,
                    }
                } else {
                    KernelAction::OneSided {
                        side: QuoteSide::Bid,
                        price: decision.ask,
                        lots,
                    }
                }
            }
            RlAction::Quote {
                bid,
                ask,
                bid_size,
                ask_size,
            } => self
                .kernel
                .grid_action(&state.kernel, &self.grid, (bid, ask, bid_size, ask_size))
                .unwrap_or(off_grid),
            RlAction::OneSided { side, offset, size } => {
                let mid2 = decision.bid.checked_add(decision.ask);
                let price = mid2.and_then(|mid2| match side {
                    QuoteSide::Bid => self.grid.bid_price(mid2, offset),
                    QuoteSide::Ask => self.grid.ask_price(mid2, offset),
                });
                match (price, self.grid.sizes().get(size)) {
                    (Some(price), Some(lots)) => KernelAction::OneSided {
                        side,
                        price,
                        lots: *lots,
                    },
                    _ => off_grid,
                }
            }
        }
    }

    fn mid_of(&self, mid2: u64) -> MMResult<Decimal> {
        let twice = self.kernel.price(mid2).ok_or_else(|| inexact("mid"))?;
        mul(twice, Decimal::new(5, 1), "mid")
    }

    /// The tick of the last completed row.
    fn mark_tick(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> Option<KernelTick> {
        let row = usize::try_from(state.mark_row?).ok()?;
        self.kernel.data().ticks().get(row).copied()
    }

    /// `R - F + p M`, with `M` the mid of the last completed row.
    fn net_of(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> MMResult<Decimal> {
        let summary = self.kernel.venue_summary(&state.kernel)?;
        let cash = sub(summary.realized, summary.total_fees, "net")?;
        let quoted = match self.mark_tick(state) {
            Some(tick) => {
                let mid2 = tick.mid2().ok_or_else(|| inexact("mark"))?;
                let value = mul(summary.position, self.mid_of(mid2)?, "position value")?;
                add(cash, value, "net")?
            }
            // Nothing can have filled: the first order follows the first decision, and
            // the row after it is matched only once this one has completed.
            None => cash,
        };
        add(quoted, self.lane_net(state)?, "net")
    }

    /// The hedge lane's net: cash, less fees, plus funding and its position at the mid of
    /// the last hedge row it ran. Zero with no lane.
    fn lane_net(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> MMResult<Decimal> {
        match (&self.cross, state.lane.as_ref()) {
            (Some(cross), Some(lane)) => Ok(cross.kernel.summary_parts(&state.kernel, lane)?.net),
            _ => Ok(Decimal::ZERO),
        }
    }

    /// The hedge lane's close on termination, at the touch of the last hedge row it ran,
    /// with its fee rate, and what it changes in the net result.
    fn hedge_terminal_close(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
    ) -> MMResult<Option<(TerminalClose, Decimal)>> {
        let (Some(cross), Some(lane)) = (&self.cross, state.lane.as_ref()) else {
            return Ok(None);
        };
        let summary = cross.kernel.summary_parts(&state.kernel, lane)?;
        let position = summary.position;
        if position.is_zero() {
            return Ok(None);
        }
        let tick = lane
            .cursor()
            .checked_sub(1)
            .and_then(|row| usize::try_from(row).ok())
            .and_then(|row| cross.kernel.hedge_data().ticks().get(row).copied())
            .ok_or_else(|| MMError::NumericalError("a hedge position with no row".to_string()))?;
        let long = position.is_sign_positive();
        let (side, touch, shown) = if long {
            (Action::Sell, tick.bid, tick.bid_lots)
        } else {
            (Action::Buy, tick.ask, tick.ask_lots)
        };
        let price = mul(Decimal::from(touch), cross.tick, "hedge close")?;
        let quantity = position.abs();
        let fee = mul(
            mul(quantity, price, "hedge close")?,
            cross.fee_rate,
            "hedge close fee",
        )?;
        let displayed = mul(Decimal::from(shown), cross.lot, "hedge close")?;
        let beyond_touch = sub(quantity, displayed, "hedge close")?.max(Decimal::ZERO);
        let mid2 = tick.mid2().ok_or_else(|| inexact("hedge mark"))?;
        let mid = mul(
            mul(Decimal::from(mid2), cross.tick, "hedge mark")?,
            Decimal::new(5, 1),
            "hedge mark",
        )?;
        let edge = sub(price, mid, "hedge close")?;
        let change = sub(mul(position, edge, "hedge close")?, fee, "hedge close")?;
        Ok(Some((
            TerminalClose {
                side,
                price,
                quantity,
                fee,
                beyond_touch,
                ts_ns: tick.ts_ns,
            },
            change,
        )))
    }

    /// The inventory's close on termination, and what it changes in the net result.
    fn terminal_close(
        &self,
        state: &RlState<ORD, MSG, DLV, OBS>,
    ) -> MMResult<Option<(TerminalClose, Decimal)>> {
        let summary = self.kernel.venue_summary(&state.kernel)?;
        let position = summary.position;
        if position.is_zero() {
            return Ok(None);
        }
        let tick = self.mark_tick(state).ok_or_else(|| {
            MMError::NumericalError("a position with no completed row".to_string())
        })?;
        let long = position.is_sign_positive();
        let (side, touch, shown) = if long {
            (Action::Sell, tick.bid, tick.bid_lots)
        } else {
            (Action::Buy, tick.ask, tick.ask_lots)
        };
        let price = self.kernel.price(touch).ok_or_else(|| inexact("close"))?;
        let quantity = position.abs();
        let fee = mul(mul(quantity, price, "close")?, self.fee_rate, "close fee")?;
        let displayed = self
            .kernel
            .quantity(shown)
            .ok_or_else(|| inexact("close"))?;
        let beyond_touch = sub(quantity, displayed, "close")?.max(Decimal::ZERO);
        let mid2 = tick.mid2().ok_or_else(|| inexact("mark"))?;
        let edge = sub(price, self.mid_of(mid2)?, "close")?;
        let change = sub(mul(position, edge, "close")?, fee, "close")?;
        Ok(Some((
            TerminalClose {
                side,
                price,
                quantity,
                fee,
                beyond_touch,
                ts_ns: tick.ts_ns,
            },
            change,
        )))
    }

    /// Books the step's reward in the state and builds the transition.
    fn settle(
        &self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        penalty: Decimal,
        outcome: StepOutcome,
        cross_hedge: Option<Result<(), HedgeRefusal>>,
    ) -> MMResult<Transition> {
        let (terminated, truncated) = match outcome.status {
            Status::Running => (false, false),
            Status::Terminal(TermReason::EndOfData) => (true, false),
            Status::Truncated => (false, true),
            Status::Halted(reason) => {
                return Err(MMError::NumericalError(format!(
                    "the kernel halted: {reason:?}"
                )));
            }
        };
        let net = self.net_of(state)?;
        let ledger_change = sub(net, state.net, "ledger change")?;
        let closing = match (terminated, self.config.terminal) {
            (true, TerminalValuation::CloseAtTouch) => self.terminal_close(state)?,
            (true, TerminalValuation::MarkAtLastMid) | (false, _) => None,
        };
        let hedge_closing = match (terminated, self.config.terminal) {
            (true, TerminalValuation::CloseAtTouch) => self.hedge_terminal_close(state)?,
            (true, TerminalValuation::MarkAtLastMid) | (false, _) => None,
        };
        let terminal_close = add(
            closing.map_or(Decimal::ZERO, |(_, change)| change),
            hedge_closing.map_or(Decimal::ZERO, |(_, change)| change),
            "terminal close",
        )?;
        let total = sub(
            add(ledger_change, terminal_close, "reward")?,
            penalty,
            "reward",
        )?;
        let as_f32 = total
            .checked_div(self.config.f32_unit)
            .and_then(|value| value.to_f32())
            .ok_or_else(|| MMError::NumericalError(format!("reward {total} does not convert")))?;
        state.net = add(net, terminal_close, "net")?;
        state.penalties = add(state.penalties, penalty, "penalties")?;
        state.close = closing.map(|(close, _)| close);
        state.hedge_close = hedge_closing;
        Ok(Transition {
            observation: self.observe(state)?,
            mask: self.mask(state),
            hedge_mask: self.hedge_mask(state),
            reward: StepReward {
                ledger_change,
                terminal_close,
                penalty,
                total,
                as_f32,
            },
            action_status: outcome.applied,
            terminated,
            truncated,
            close: state.close,
            cross_hedge_mask: self.cross_hedge_mask(state),
            hedge_observation: self.observe_hedge(state)?,
            cross_hedge,
            hedge_close: state.hedge_close(),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::backtest::MarketTick;
    use crate::backtest::kernel::{HaltReason, KernelLatency, KernelTick, RejectReason};
    use crate::dec;

    fn scales() -> ObservationScales {
        ObservationScales {
            spread_ticks: dec!(10),
            mid_change_ticks: dec!(10),
            row_gap_ms: dec!(1000),
            size_lots: dec!(10),
            position: dec!(10),
            entry_gap: dec!(1),
        }
    }

    /// A kernel from `FinancialKernel::new` is sized so that its capacity never refuses,
    /// so this one skips that check.
    fn capacity_run(policy: InvalidActionPolicy) -> Vec<MMResult<Transition>> {
        let latency = KernelLatency::zero(1_000_000).with_delays(0, 0, 0, 100_000);
        let cfg =
            KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_latency(latency);
        let ticks: Vec<MarketTick> = (0..6)
            .map(|i| MarketTick::new(1000 + i, dec!(99.5), dec!(1), dec!(100.5), dec!(1)))
            .collect();
        let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
        let kernel: FinancialKernel<'_, 2, 64, 128, 64> =
            FinancialKernel::new_relaxed(&cfg, &data, false, false, true).unwrap();
        let grid = ActionGrid::new([10], [10], [1]).unwrap();
        let rl = RlConfig::new(scales(), dec!(1)).with_invalid_action(policy, dec!(0.25));
        let env: RlEnv<'_, 1, 1, 2, 64, 128, 64> = RlEnv::assemble(kernel, &cfg, grid, rl).unwrap();
        let mut state = env.reset(1).unwrap();
        let action = RlAction::Quote {
            bid: 0,
            ask: 0,
            bid_size: 0,
            ask_size: 0,
        };
        let first = env.step(&mut state, action);
        assert!(!env.mask(&state).global_quote_ok);
        assert_eq!(env.mask(&state).refusal, Some(RejectReason::CapacityOrders));
        let before = state;
        let second = env.step(&mut state, action);
        if policy == InvalidActionPolicy::Mask {
            assert_eq!(state, before);
        }
        let hold = env.step(&mut state, RlAction::Hold);
        vec![first, second, hold]
    }

    #[test]
    fn test_over_capacity_under_the_mask_policy_is_an_error_and_a_hold_goes_on() {
        let run = capacity_run(InvalidActionPolicy::Mask);
        assert_eq!(
            run[0].as_ref().unwrap().action_status,
            ActionStatus::Applied
        );
        assert!(matches!(run[1], Err(MMError::InvalidQuoteGeneration(_))));
        assert_eq!(
            run[2].as_ref().unwrap().action_status,
            ActionStatus::Applied
        );
    }

    #[test]
    fn test_over_capacity_under_the_reject_policy_is_a_charged_hold() {
        let run = capacity_run(InvalidActionPolicy::Reject);
        assert_eq!(run[0].as_ref().unwrap().reward.penalty, Decimal::ZERO);
        let rejected = run[1].as_ref().unwrap();
        assert_eq!(
            rejected.action_status,
            ActionStatus::Rejected(RejectReason::CapacityOrders)
        );
        assert_eq!(rejected.reward.penalty, dec!(0.25));
        assert_eq!(rejected.reward.total, dec!(-0.25));
        assert_eq!(run[2].as_ref().unwrap().reward.penalty, Decimal::ZERO);
    }

    #[test]
    fn test_a_halt_is_an_error_and_the_state_refuses_every_later_step() {
        // Outside the kernel's bounds: the notional of one fill overflows.
        let p: u64 = 9_000_000_000_000_000_000;
        let cfg = KernelConfig::new(dec!(1), dec!(1), Decimal::from(p), Decimal::from(p), 10)
            .with_fee_rate(dec!(0.1));
        let row = |ts_ns, bid, ask| KernelTick {
            ts_ns,
            bid,
            bid_lots: 1,
            ask,
            ask_lots: 1,
        };
        let data = KernelDataset::from_kernel_ticks(
            vec![row(1, p - 2, p - 1), row(2, p - 4, p - 3)],
            &cfg,
        )
        .unwrap();
        let kernel: FinancialKernel<'_> =
            FinancialKernel::new_relaxed(&cfg, &data, true, false, false).unwrap();
        let grid = ActionGrid::new([0], [0], [p]).unwrap();
        let rl = RlConfig::new(scales(), dec!(1));
        let env: RlEnv<'_, 1, 1> = RlEnv::assemble(kernel, &cfg, grid, rl).unwrap();
        let mut state = env.reset(1).unwrap();
        let action = RlAction::Quote {
            bid: 0,
            ask: 0,
            bid_size: 0,
            ask_size: 0,
        };
        assert!(matches!(
            env.step(&mut state, action),
            Err(MMError::NumericalError(_))
        ));
        assert_eq!(
            state.kernel().status(),
            Status::Halted(HaltReason::LedgerOverflow)
        );
        assert!(state.is_done());
        assert!(matches!(
            env.step(&mut state, RlAction::Hold),
            Err(MMError::InvalidMarketState(_))
        ));
    }

    fn refused() -> ActionMask {
        ActionMask {
            global_quote_ok: false,
            bid: u64::MAX,
            ask: u64::MAX,
            size: u64::MAX,
            refusal: Some(RejectReason::CapacityOrders),
        }
    }

    #[test]
    fn test_allowed_a_refused_mask_blocks_quotes_and_leaves_a_hold() {
        let mask = refused();
        assert!(allowed(&mask, 0, 0, false, 0, RlAction::Hold));
        let quote = RlAction::Quote {
            bid: 0,
            ask: 0,
            bid_size: 0,
            ask_size: 0,
        };
        assert!(!allowed(&mask, 0, 0, false, 0, quote));
        for side in [QuoteSide::Bid, QuoteSide::Ask] {
            let one = RlAction::OneSided {
                side,
                offset: 0,
                size: 0,
            };
            assert!(!allowed(&mask, 0, 0, false, 0, one));
            let open = ActionMask {
                global_quote_ok: true,
                refusal: None,
                ..mask
            };
            assert!(allowed(&open, 0, 0, false, 0, one));
        }
    }

    #[test]
    fn test_allowed_one_sided_reads_only_its_own_side_and_the_size() {
        let mask = ActionMask {
            global_quote_ok: true,
            bid: 0b01,
            ask: 0b10,
            size: 0b01,
            refusal: None,
        };
        let one = |side, offset, size| RlAction::OneSided { side, offset, size };
        assert!(allowed(&mask, 0, 0, false, 0, one(QuoteSide::Bid, 0, 0)));
        assert!(!allowed(&mask, 0, 0, false, 0, one(QuoteSide::Bid, 1, 0)));
        assert!(allowed(&mask, 0, 0, false, 0, one(QuoteSide::Ask, 1, 0)));
        assert!(!allowed(&mask, 0, 0, false, 0, one(QuoteSide::Ask, 0, 0)));
        assert!(!allowed(&mask, 0, 0, false, 0, one(QuoteSide::Bid, 0, 1)));
        assert!(!allowed(&mask, 0, 0, false, 0, one(QuoteSide::Bid, 64, 0)));
    }
}
