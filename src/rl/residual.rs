//! Bounded corrections to a classical quote.

use rust_decimal::prelude::ToPrimitive;

use crate::Decimal;
use crate::backtest::kernel::{
    ActionGrid, EventSink, KernelAction, KernelConfig, KernelDataset, NoSink, Observation,
    QuoteAdapter, QuoteSide, QuoteSource,
};
use crate::types::error::{MMError, MMResult};

use super::env::{RlConfig, RlEnv, RlState, Transition};
use super::features::FeatureVector;
use super::spec::SpecId;

/// Version of the residual's rules. Raised when [`compose`] changes meaning.
pub const RESIDUAL_VERSION: u16 = 1;

/// The largest correction, in either direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualBounds {
    /// Largest move of a price, in ticks.
    pub price_ticks: u32,
    /// Largest change of a size, in lots.
    pub size_lots: u32,
}

/// A policy's output, each value in `[-1, 1]` as a fraction of the bound.
///
/// The sides are symmetric: a positive `bid_away` lowers the bid and a positive `ask_away`
/// raises the ask, so a positive value moves its side away from the mid on either side. A
/// positive size adds lots on its side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualProposal {
    /// Move of the bid away from the mid.
    pub bid_away: f64,
    /// Move of the ask away from the mid.
    pub ask_away: f64,
    /// Change of the bid size.
    pub bid_size: f64,
    /// Change of the ask size.
    pub ask_size: f64,
}

impl ResidualProposal {
    /// No correction.
    pub const ZERO: Self = Self {
        bid_away: 0.0,
        ask_away: 0.0,
        bid_size: 0.0,
        ask_size: 0.0,
    };
}

/// A proposal in ticks and lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidualDelta {
    /// Ticks the bid moves down.
    pub bid_away_ticks: i64,
    /// Ticks the ask moves up.
    pub ask_away_ticks: i64,
    /// Lots added to the bid.
    pub bid_lots: i64,
    /// Lots added to the ask.
    pub ask_lots: i64,
}

/// What was clipped on the way to an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ResidualClips {
    /// A proposal value was outside `[-1, 1]`.
    pub proposal: bool,
    /// A price left `1..=max_price` ticks.
    pub price: bool,
    /// A size left `1..=max_order_size` lots.
    pub size: bool,
}

/// Why the classical action stands instead of a composed one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidualFallback {
    /// A proposal value was not finite.
    NonFinite,
    /// The bid would reach the ask.
    Crossed,
    /// There is no classical quote to correct: a hold or a strategy error.
    NoClassicalQuote,
}

/// One decision, component by component.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ResidualDecision {
    /// The classical quote, discretized.
    pub classical: KernelAction,
    /// What the policy said.
    pub proposal: ResidualProposal,
    /// The proposal in ticks and lots, zero if it fell back.
    pub delta: ResidualDelta,
    /// What goes to the kernel.
    pub composed: KernelAction,
    /// What was clipped.
    pub clips: ResidualClips,
    /// Why the classical action stands, if it does.
    pub fallback: Option<ResidualFallback>,
}

fn scaled(value: f64, bound: u32, clips: &mut ResidualClips) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    if !(-1.0..=1.0).contains(&value) {
        clips.proposal = true;
    }
    (value.clamp(-1.0, 1.0) * f64::from(bound)).round().to_i64()
}

fn fallback(
    classical: KernelAction,
    proposal: ResidualProposal,
    clips: ResidualClips,
    reason: ResidualFallback,
) -> ResidualDecision {
    ResidualDecision {
        classical,
        proposal,
        delta: ResidualDelta::default(),
        composed: classical,
        clips,
        fallback: Some(reason),
    }
}

/// Moves `value` by `delta` and keeps it in `1..=max`.
fn moved(value: u64, delta: i64, max: u64, clipped: &mut bool) -> u64 {
    let wide = i128::from(value).saturating_add(i128::from(delta));
    let kept = wide.clamp(1, i128::from(max));
    if kept != wide {
        *clipped = true;
    }
    u64::try_from(kept).unwrap_or(max)
}

/// Applies a proposal to a classical action, in this order:
///
/// 1. A value that is not finite refuses the whole proposal. One outside `[-1, 1]` is
///    clipped, and the value is multiplied by its bound and rounded to a whole tick or lot.
/// 2. A hold or a strategy error has no quote to correct, and stands.
/// 3. Prices move by the delta and stay in `1..=max_price` ticks. Sizes move by the delta and
///    stay in `1..=max_order_size` lots, so a correction cannot remove a side.
/// 4. A two sided quote whose bid would reach its ask falls back to the classical action.
///    So does a nonzero correction of a classical quote that is itself crossed, which the
///    kernel rejects whole.
///
/// A one sided quote takes only its own side's delta. A zero delta gives `classical` itself,
/// so a zero residual reproduces the baseline with no transformation at all. The kernel's
/// admission and its gate come after this, for any result.
#[must_use]
pub fn compose(
    classical: KernelAction,
    proposal: ResidualProposal,
    bounds: ResidualBounds,
    max_price_ticks: u64,
    max_lots: u64,
) -> ResidualDecision {
    let mut clips = ResidualClips::default();
    let parts = [
        scaled(proposal.bid_away, bounds.price_ticks, &mut clips),
        scaled(proposal.ask_away, bounds.price_ticks, &mut clips),
        scaled(proposal.bid_size, bounds.size_lots, &mut clips),
        scaled(proposal.ask_size, bounds.size_lots, &mut clips),
    ];
    let [
        Some(bid_away),
        Some(ask_away),
        Some(bid_more),
        Some(ask_more),
    ] = parts
    else {
        return fallback(classical, proposal, clips, ResidualFallback::NonFinite);
    };
    let delta = ResidualDelta {
        bid_away_ticks: bid_away,
        ask_away_ticks: ask_away,
        bid_lots: bid_more,
        ask_lots: ask_more,
    };
    let composed = match classical {
        // A cancel has no quote to move.
        KernelAction::Hold | KernelAction::QuoteError(_) | KernelAction::Cancel => {
            return fallback(
                classical,
                proposal,
                clips,
                ResidualFallback::NoClassicalQuote,
            );
        }
        KernelAction::Quote { .. }
        | KernelAction::ExpiringQuote { .. }
        | KernelAction::OneSided { .. }
            if delta == ResidualDelta::default() =>
        {
            classical
        }
        KernelAction::Quote {
            bid,
            ask,
            bid_lots,
            ask_lots,
        } => {
            if bid >= ask {
                return fallback(classical, proposal, clips, ResidualFallback::Crossed);
            }
            let new_bid = moved(
                bid,
                bid_away.saturating_neg(),
                max_price_ticks,
                &mut clips.price,
            );
            let new_ask = moved(ask, ask_away, max_price_ticks, &mut clips.price);
            if new_bid >= new_ask {
                return fallback(classical, proposal, clips, ResidualFallback::Crossed);
            }
            KernelAction::Quote {
                bid: new_bid,
                ask: new_ask,
                bid_lots: moved(bid_lots, bid_more, max_lots, &mut clips.size),
                ask_lots: moved(ask_lots, ask_more, max_lots, &mut clips.size),
            }
        }
        // A classical source that quotes with an expiry keeps it: the
        // residual moves the prices and sizes only.
        KernelAction::ExpiringQuote {
            bid,
            ask,
            bid_lots,
            ask_lots,
            life_ns,
        } => {
            if bid >= ask {
                return fallback(classical, proposal, clips, ResidualFallback::Crossed);
            }
            let new_bid = moved(
                bid,
                bid_away.saturating_neg(),
                max_price_ticks,
                &mut clips.price,
            );
            let new_ask = moved(ask, ask_away, max_price_ticks, &mut clips.price);
            if new_bid >= new_ask {
                return fallback(classical, proposal, clips, ResidualFallback::Crossed);
            }
            KernelAction::ExpiringQuote {
                bid: new_bid,
                ask: new_ask,
                bid_lots: moved(bid_lots, bid_more, max_lots, &mut clips.size),
                ask_lots: moved(ask_lots, ask_more, max_lots, &mut clips.size),
                life_ns,
            }
        }
        KernelAction::OneSided { side, price, lots } => {
            let (price_delta, size_delta) = match side {
                QuoteSide::Bid => (bid_away.saturating_neg(), bid_more),
                QuoteSide::Ask => (ask_away, ask_more),
            };
            KernelAction::OneSided {
                side,
                price: moved(price, price_delta, max_price_ticks, &mut clips.price),
                lots: moved(lots, size_delta, max_lots, &mut clips.size),
            }
        }
    };
    ResidualDecision {
        classical,
        proposal,
        delta,
        composed,
        clips,
        fallback: None,
    }
}

/// What a residual step gave.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ResidualStep {
    /// The environment's transition.
    pub transition: Transition,
    /// How the action came to be.
    pub decision: ResidualDecision,
}

/// An environment whose policy corrects a classical quote. See [`compose`].
///
/// It needs `&mut self` to step, as a [`QuoteSource`] may keep state within an episode.
/// Every reset starts the source again from the state it was built with, so what it learnt
/// in one episode does not reach the next, and one seed repeats.
#[derive(Debug, Clone)]
pub struct ResidualEnv<
    'd,
    S,
    const ORD: usize = 16,
    const MSG: usize = 64,
    const DLV: usize = 128,
    const OBS: usize = 64,
> {
    env: RlEnv<'d, 1, 1, ORD, MSG, DLV, OBS>,
    adapter: QuoteAdapter<S>,
    built: QuoteAdapter<S>,
    bounds: ResidualBounds,
    max_price_ticks: u64,
    max_lots: u64,
}

impl<
    'd,
    S: QuoteSource + Clone,
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
> ResidualEnv<'d, S, ORD, MSG, DLV, OBS>
{
    /// Builds the environment. `default_order_size` replaces a classical size that is not
    /// positive, as in [`QuoteAdapter::new`].
    ///
    /// # Errors
    ///
    /// Returns whatever [`RlEnv::new`] and [`QuoteAdapter::new`] refuse.
    pub fn new(
        kernel_config: &KernelConfig,
        data: &'d KernelDataset,
        source: S,
        default_order_size: Decimal,
        bounds: ResidualBounds,
        config: RlConfig,
    ) -> MMResult<Self> {
        if config.quote_lives.is_some() {
            return Err(MMError::InvalidConfiguration(
                "quote lives are not available with a residual policy".to_string(),
            ));
        }
        // The grid of the inner environment is not used: quotes come from `compose`.
        let grid = ActionGrid::new([0], [0], [1])?;
        let env = RlEnv::new(kernel_config, data, grid, config)?;
        let adapter = QuoteAdapter::new(source, env.kernel(), default_order_size)?;
        let whole = |value: Decimal, unit: Decimal, what: &str| {
            value
                .checked_div(unit)
                .and_then(|count| count.to_u64())
                .ok_or_else(|| MMError::InvalidConfiguration(format!("{what} does not convert")))
        };
        Ok(Self {
            max_price_ticks: whole(
                kernel_config.max_price,
                kernel_config.tick_size,
                "max_price",
            )?,
            max_lots: whole(
                kernel_config.max_order_size,
                kernel_config.lot_size,
                "max_order_size",
            )?,
            env,
            built: adapter.clone(),
            adapter,
            bounds,
        })
    }

    /// The inner environment, for its kernel, its records and its reconciliation.
    #[must_use]
    pub fn env(&self) -> &RlEnv<'d, 1, 1, ORD, MSG, DLV, OBS> {
        &self.env
    }

    /// The bounds of the correction.
    #[must_use]
    pub fn bounds(&self) -> ResidualBounds {
        self.bounds
    }

    /// The identity of the spaces. Its fingerprint does not cover the bounds or the classical
    /// strategy, which a consumer states on its own.
    #[must_use]
    pub fn spec_id(&self) -> SpecId {
        self.env.spec().id()
    }

    /// The state at the first decision, with the [`QuoteSource`] as it was built.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::reset`].
    pub fn reset(&mut self, seed: u64) -> MMResult<RlState<ORD, MSG, DLV, OBS>> {
        // A state is large, so it is returned straight from the inner environment.
        self.adapter = self.built.clone();
        self.env.reset(seed)
    }

    /// As [`ResidualEnv::reset`], sending the kernel's events to `sink`.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::reset`].
    pub fn reset_with<E: EventSink>(
        &mut self,
        seed: u64,
        sink: &mut E,
    ) -> MMResult<RlState<ORD, MSG, DLV, OBS>> {
        self.adapter = self.built.clone();
        self.env.reset_with(seed, sink)
    }

    /// The observation at `state`.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::observe`].
    pub fn observe(&self, state: &RlState<ORD, MSG, DLV, OBS>) -> MMResult<FeatureVector> {
        self.env.observe(state)
    }

    /// What the classical strategy would do at `state`, from what the agent knows.
    #[must_use]
    pub fn classical(&mut self, state: &RlState<ORD, MSG, DLV, OBS>) -> KernelAction {
        let observation: Observation = self.env.kernel().observe(state.kernel());
        self.adapter.act(self.env.kernel(), &observation)
    }

    /// Takes one decision.
    ///
    /// # Errors
    ///
    /// As [`ResidualEnv::step_with`].
    pub fn step(
        &mut self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        proposal: ResidualProposal,
    ) -> MMResult<ResidualStep> {
        self.step_with(state, proposal, &mut NoSink)
    }

    /// As [`ResidualEnv::step`], sending the kernel's events to `sink`.
    ///
    /// # Errors
    ///
    /// As [`RlEnv::step_with`]. A proposal is never an error: what it cannot do is clipped or
    /// falls back, and the decision says so.
    pub fn step_with<E: EventSink>(
        &mut self,
        state: &mut RlState<ORD, MSG, DLV, OBS>,
        proposal: ResidualProposal,
        sink: &mut E,
    ) -> MMResult<ResidualStep> {
        let classical = self.classical(state);
        let decision = compose(
            classical,
            proposal,
            self.bounds,
            self.max_price_ticks,
            self.max_lots,
        );
        let transition = self
            .env
            .step_composed_with(state, decision.composed, sink)?;
        Ok(ResidualStep {
            transition,
            decision,
        })
    }
}
