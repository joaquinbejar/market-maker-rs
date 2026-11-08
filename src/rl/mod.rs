//! Versioned observation, action and reward spaces (feature `rl`).
//!
//! This module states a learning problem on top of the financial kernel
//! ([`crate::backtest::kernel`]) for an external RL trainer or engine: what a
//! policy sees, what it controls, how it is rewarded and how an episode ends. It
//! holds no training algorithm.
//!
//! [`RlEnv`] drives one kernel, and [`RlSpec`] describes the spaces. A consumer
//! calls [`RlSpec::ensure`] with the [`SpecId`] it was built for, before any step,
//! so that an unknown specification fails at once.
//!
//! # Observation
//!
//! A [`FeatureVector`] is computed from the kernel's
//! [`Observation`](crate::backtest::kernel::Observation), the configuration and
//! the previous decision's row, which the state keeps as its only memory. So it
//! holds only what a real agent could know at the decision: the last row delivered
//! to it, its delivered fills and its own account. It reads no venue position, no
//! undelivered report, no row beyond the observed one and not the length of the
//! data. Each feature declares its unit, its time, its horizon, its memory and
//! its divisor in a [`FeatureSpec`].
//!
//! Normalization divides by a constant from the configuration and clips to
//! `[-1, 1]`, so no statistic of the data enters it. [`FeatureVector::raw`] keeps
//! the `Decimal` value, with `None` for a feature with no value at this decision,
//! and the `f32` is derived from it.
//!
//! # Action
//!
//! An [`RlAction`] holds indices into the [`ActionGrid`](crate::backtest::kernel::ActionGrid):
//! offsets in ticks around the observed mid and sizes in lots. [`ActionSpace`] states
//! the tick, the lot and the bounds. There is no cancel without a new quote.
//!
//! With [`RlConfig::with_hedge_actions`], [`RlAction::Hedge`] reduces the delivered
//! position with a marketable limit order at the observed touch, in the one instrument the
//! kernel holds, as [`HedgeActions::MarketableAtTouch`] states. Its size comes from the
//! order size grid and is never larger than that position at the decision, as
//! [`RlEnv::hedge_mask`] shows. With latency, a resting order can still fill before the
//! hedge replaces it, so the hedge can then take the position past flat. It is an order like any other: the kernel's admission, its gate and its risk
//! limits decide on it whatever the mask said, it fills under the kernel's fill model at
//! its own price with the kernel's fee rate, and its cost reaches the reward only through
//! the kernel's ledger. A hedge in another instrument is not modelled.
//!
//! An action outside the mask follows the declared [`InvalidActionPolicy`]:
//!
//! - [`InvalidActionPolicy::Mask`]: the trainer promised to respect the mask, so
//!   [`RlEnv::step`] returns an error and the state does not change.
//! - [`InvalidActionPolicy::Reject`]: the step is taken as a hold, with the kernel's
//!   reason in [`Transition::action_status`] and the declared penalty in the reward.
//!
//! The mask covers admission only: price range, size range and the capacity the
//! agent knows of, though a kernel from `FinancialKernel::new` is sized so that its
//! capacity never refuses. It never reads the hard limits. Those are checked by the kernel's
//! pretrade gate on the venue position, for any action the mask lets through, and
//! nothing in this module can turn them off.
//!
//! # Reward
//!
//! `Decimal` is the authority, and [`StepReward::as_f32`] is derived from it and read
//! by nothing here. With `R` the cash flows of the fills so far, `F` the quote fees so
//! far, `p` the position and `M` the mid of the last row the kernel has completed:
//!
//! ```text
//! net(t)  = R(t) - F(t) + p(t) M(t)
//! ledger  = net(t + 1) - net(t)
//! reward  = ledger + terminal close - penalty
//! ```
//!
//! This is [`MetricId::NetPnl`](crate::backtest::evaluation::MetricId::NetPnl) of
//! the marks and fills so far: at every decision, the sum of the ledger changes and
//! the close is exactly the net result of
//! [`attribute`](crate::backtest::evaluation::attribute), which
//! [`RlEnv::reconcile`] checks from an [`EpisodeRecorder`]. A fee enters once, in
//! `F`, as a cost of the step in which its fill is booked. A penalty is not
//! accounting: it is kept apart and never enters `net`.
//!
//! The reward follows the venue's clock, not the agent's. A row's mid enters `M` when
//! that row completes, which with no latency is after the decision on that row, so a
//! price move reaches the reward one step after the observation shows it. With latency,
//! a reward can reflect fills and rows the agent has not been told of. It belongs in the
//! training signal: [`RewardSpec::policy_input`] is false, and feeding it to a policy
//! would hand the policy those fills and rows.
//!
//! # End of an episode
//!
//! - `terminated`: the data ended. Reports still queued are delivered, and the
//!   remaining inventory goes through the declared [`TerminalValuation`].
//! - `truncated`: `max_steps` decisions were taken. The position stays open and
//!   valued at `M`, with no close, since the market goes on. Value after the cut is
//!   the trainer's estimate, not part of this ledger.
//! - A halt of the kernel is neither: [`RlEnv::step`] returns an error, and every
//!   later call on that state does too.
//!
//! [`TerminalValuation::CloseAtTouch`] sells a long position at the last row's bid and
//! buys a short one at its ask, for the whole size, paying the kernel's fee rate. That is
//! an assumption, not a fill the venue made: [`TerminalClose::beyond_touch`] reports
//! the size above what the touch displayed, and no impact is charged for it.
//!
//! # Queue-reactive markets
//!
//! [`qr_market`] simulates the rows of an episode with
//! [`QrSimulator`](crate::backtest::queue_reactive::QrSimulator), so that [`RlEnv`] can
//! run on them. The market comes first and does not react: the agent's orders change
//! neither the queues nor the reference price, and its fills come from the kernel's fill
//! model against the sampled touch, not from the simulated market orders. This is no
//! replica of an exchange, and it claims no queue priority of its own. A simulator that
//! stops ends the rows early, with the reason in [`QrMarket::stopped`].
//!
//! # Residual policies
//!
//! [`ResidualEnv`] puts a bounded correction on top of a classical quote from a
//! [`QuoteSource`](crate::backtest::kernel::QuoteSource). That residual policies help is an
//! implementation hypothesis of this crate, not a result of the market making papers. The
//! order is fixed: the classical quote is discretized by
//! [`QuoteAdapter`](crate::backtest::kernel::QuoteAdapter), then [`compose`] applies the
//! residual in ticks and lots and validates the result against the market grid, then the
//! kernel's admission and its gate decide. See [`compose`] for the rules. A zero residual
//! gives the classical action unchanged, and [`ResidualDecision`] logs each component
//! of every decision. There is no mask here: what the correction cannot do is clipped or
//! falls back to the classical action, with the reason logged.
//!
//! # Limits
//!
//! One linear spot instrument with fees in the quote asset, as the kernel. The reward
//! has one kind, the change of net equity. Agreement with an external RL engine's
//! adapter (shapes, masks, `terminated` and `truncated`) cannot be checked from this
//! crate alone.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::MarketTick;
//! use market_maker_rs::backtest::kernel::{ActionGrid, KernelConfig, KernelDataset};
//! use market_maker_rs::dec;
//! use market_maker_rs::rl::{ObservationScales, RlAction, RlConfig, RlEnv};
//!
//! let kernel = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(10), 100);
//! let ticks: Vec<MarketTick> = (0..4)
//!     .map(|i| MarketTick::new(1_000 + i, dec!(100.00), dec!(1), dec!(100.02), dec!(1)))
//!     .collect();
//! let data = KernelDataset::from_ticks(&ticks, &kernel)?;
//! let grid = ActionGrid::new([1, 2], [1, 2], [1, 2])?;
//! let scales = ObservationScales {
//!     spread_ticks: dec!(10),
//!     mid_change_ticks: dec!(10),
//!     row_gap_ms: dec!(1000),
//!     size_lots: dec!(10),
//!     position: dec!(10),
//!     entry_gap: dec!(1),
//! };
//! let env: RlEnv<'_, 2, 2> = RlEnv::new(&kernel, &data, grid, RlConfig::new(scales, dec!(1)))?;
//! let mut state = env.reset(7)?;
//! let mut total = dec!(0);
//! while !state.is_done() {
//!     let action = RlAction::Quote { bid: 0, ask: 0, bid_size: 0, ask_size: 0 };
//!     let transition = env.step(&mut state, action)?;
//!     total += transition.reward.total;
//! }
//! assert_eq!(total, state.net());
//! # Ok::<(), market_maker_rs::types::error::MMError>(())
//! ```

mod audit;
mod env;
mod features;
mod queue_reactive;
mod residual;
mod spec;

/// The version of this crate, for the record of what produced a run.
pub const LIBRARY_VERSION: &str = env!("CARGO_PKG_VERSION");

pub use audit::{EpisodeRecorder, EpisodeRecords};
pub use env::{
    ERR_ACTION_OFF_GRID, ERR_CANCEL_REFUSED, ERR_CROSS_HEDGE_REFUSED, ERR_EXPIRY_REFUSED,
    ERR_HEDGE_REFUSED, RlAction, RlConfig, RlEnv, RlState, StepReward, TerminalClose, Transition,
};
pub use features::{FeatureVector, HedgeFeatureVector};
pub use queue_reactive::{
    MAX_QR_ROWS, MAX_REACTION_MOVES, QrEpisode, QrMarket, ReactiveQrMarket, qr_market,
};
pub use residual::{
    RESIDUAL_VERSION, ResidualBounds, ResidualClips, ResidualDecision, ResidualDelta, ResidualEnv,
    ResidualFallback, ResidualProposal, ResidualStep, compose,
};
pub use spec::{
    ACTION_SPACE_VERSION, ActionSpace, AsOf, CANCEL_ACTION_SPACE_VERSION, CancelActions,
    CrossHedgeSpace, EPISODE_VERSION, EXPIRY_ACTION_SPACE_VERSION, EpisodeSpec, FEATURE_COUNT,
    FeatureHorizon, FeatureId, FeatureSpec, FeatureUnit, HEDGE_FEATURE_COUNT,
    HEDGED_ACTION_SPACE_VERSION, HEDGED_OBSERVATION_SPACE_VERSION, HEDGED_REWARD_VERSION,
    HedgeActions, HedgeScales, InvalidActionPolicy, MAX_QUOTE_LIFE_NS, MAX_QUOTE_LIVES,
    OBSERVATION_SPACE_VERSION, ObservationScales, ObservationSpace, QuoteExpiry, QuoteLives,
    REWARD_VERSION, RewardKind, RewardMark, RewardSpec, RlSpec, SpaceShape, SpecId, StateMemory,
    TerminalValuation,
};
