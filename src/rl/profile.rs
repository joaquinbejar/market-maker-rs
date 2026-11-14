//! A specification from a declared profile, with no kernel, and the layout of the action
//! heads.
//!
//! An external environment that reuses these spaces builds its [`RlSpec`] with
//! [`RlProfile::spec`], computes its observations with
//! [`observation_features`](crate::rl::observation_features) and turns the rows of its
//! policy's discrete heads into actions with [`RlSpec::decode_action`]. It keeps no copy of
//! the layout, so its [`SpecId`](crate::rl::SpecId) and its vectors are the crate's.

use crate::Decimal;
use crate::backtest::evaluation::{HedgeNotModelledReason, MetricId, MetricUnit};
use crate::backtest::kernel::QuoteSide;
use crate::types::error::{MMError, MMResult};

use super::env::{RlAction, RlConfig};
use super::spec::{
    ACTION_SPACE_VERSION, ActionSpace, CANCEL_ACTION_SPACE_VERSION, CancelActions, EPISODE_VERSION,
    EXPIRY_ACTION_SPACE_VERSION, EpisodeSpec, HedgeActions, InstrumentRole, InstrumentSpace,
    MAX_PROFILE_INSTRUMENTS, MULTI_INSTRUMENT_ACTION_SPACE_VERSION, ObservationSpace, QuoteExpiry,
    REWARD_VERSION, RewardKind, RewardMark, RewardSpec, RlSpec,
};

/// Number of discrete action heads, in the order of [`ActionHead`].
pub const ACTION_HEAD_COUNT: usize = 7;

/// Number of values of [`ActionHead::Kind`].
pub const ACTION_KIND_COUNT: usize = 8;

/// A discrete action head, at its position in a row of head values.
///
/// [`ActionHead::Kind`] takes these values: 0 [`RlAction::Hold`], 1 [`RlAction::Quote`],
/// 2 [`RlAction::OneSided`] on the bid, 3 [`RlAction::OneSided`] on the ask,
/// 4 [`RlAction::Hedge`], 5 [`RlAction::CrossHedge`], 6 [`RlAction::Cancel`] and
/// 7 [`RlAction::ExpiringQuote`]. A one sided order reads the offset and size heads of its
/// side; a hedge and a hedge in the second instrument read [`ActionHead::BidSize`]. A kind
/// ignores the heads it does not read, and [`RlSpec::encode_action`] writes 0 there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ActionHead {
    /// The kind of action.
    Kind,
    /// The position of the instrument in [`ActionSpace::instruments`]; 0 with one plain
    /// instrument.
    Instrument,
    /// Index of the bid offset.
    Bid,
    /// Index of the ask offset.
    Ask,
    /// Index of the bid size, or of a hedge's size.
    BidSize,
    /// Index of the ask size.
    AskSize,
    /// Index of the quote life.
    Life,
}

impl ActionHead {
    /// Every head, in the order of a row.
    pub const ALL: [Self; ACTION_HEAD_COUNT] = [
        Self::Kind,
        Self::Instrument,
        Self::Bid,
        Self::Ask,
        Self::BidSize,
        Self::AskSize,
        Self::Life,
    ];

    /// Position in a row.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Kind => 0,
            Self::Instrument => 1,
            Self::Bid => 2,
            Self::Ask => 3,
            Self::BidSize => 4,
            Self::AskSize => 5,
            Self::Life => 6,
        }
    }
}

/// An action on one instrument of the specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct InstrumentAction {
    /// The position of the instrument in [`ActionSpace::instruments`]; 0 with one plain
    /// instrument.
    pub instrument: usize,
    /// The action, with indices into that instrument's grid.
    pub action: RlAction,
}

impl InstrumentAction {
    /// `action` on the instrument at `instrument`.
    #[must_use]
    pub fn new(instrument: usize, action: RlAction) -> Self {
        Self { instrument, action }
    }
}

/// A declared profile: everything a specification states, with no kernel and no data.
///
/// With one quoted instrument of depth 1 and one order per side, [`RlProfile::spec`] is the
/// specification that [`RlEnv::new`](crate::rl::RlEnv::new) states for the same kernel
/// configuration (tick, lot, largest price and order size, `max_steps`), grid, `ORD` and
/// [`RlConfig`], to the byte. Any other profile adds the instruments and the index head, of
/// action version [`MULTI_INSTRUMENT_ACTION_SPACE_VERSION`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RlProfile {
    /// The ordered instruments; the first is quoted, and the index head points into them.
    pub instruments: Vec<InstrumentSpace>,
    /// Scales, invalid action policy and penalty, terminal valuation, hedge, cancel and
    /// lives. The hedge lane of [`RlEnv::new_hedged`](crate::rl::RlEnv::new_hedged) is not
    /// declared here: a hedge instrument is.
    pub config: RlConfig,
    /// The order limit: the divisor of the open order features, `ORD` for an environment.
    pub order_capacity: u64,
    /// Decisions before truncation.
    pub max_steps: u64,
    /// What values an open position in the reward.
    pub reward_mark: RewardMark,
}

impl RlProfile {
    /// A profile marked at [`RewardMark::LastCompletedRow`], as an environment of this crate.
    #[must_use]
    pub fn new(
        instruments: Vec<InstrumentSpace>,
        config: RlConfig,
        order_capacity: u64,
        max_steps: u64,
    ) -> Self {
        Self {
            instruments,
            config,
            order_capacity,
            max_steps,
            reward_mark: RewardMark::LastCompletedRow,
        }
    }

    /// Sets what values an open position, such as [`RewardMark::ExternalVenueMark`].
    #[must_use]
    pub fn with_reward_mark(mut self, mark: RewardMark) -> Self {
        self.reward_mark = mark;
        self
    }

    /// The specification of this profile.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a scale, `f32_unit`,
    /// `order_capacity` or `max_steps` that is not positive, a negative penalty, no
    /// instruments or more than [`MAX_PROFILE_INSTRUMENTS`], a first instrument that is not
    /// quoted, and an instrument with a tick, lot, largest price or order size that is not
    /// positive, a zero depth or `max_orders`, empty grids, grids of more than 64 values,
    /// bid and ask offsets of different lengths, or a zero size.
    pub fn spec(&self) -> MMResult<RlSpec> {
        let refuse = |why: String| Err(MMError::InvalidConfiguration(why));
        let cfg = &self.config;
        check_config(cfg)?;
        if self.instruments.is_empty() || self.instruments.len() > MAX_PROFILE_INSTRUMENTS {
            return refuse(format!(
                "a profile holds 1 to {MAX_PROFILE_INSTRUMENTS} instruments, got {}",
                self.instruments.len()
            ));
        }
        for (i, instrument) in self.instruments.iter().enumerate() {
            check_instrument(i, instrument)?;
        }
        let Some(first) = self.instruments.first() else {
            return refuse("a profile needs an instrument".to_string());
        };
        if first.role != InstrumentRole::Quoted {
            return refuse("the first instrument of a profile must be quoted".to_string());
        }
        let observation =
            ObservationSpace::from_scales(&cfg.scales, self.order_capacity, self.max_steps)?;
        let mut spec = build_spec(observation, first, cfg, self.max_steps, self.reward_mark);
        if !(self.instruments.len() == 1 && first.is_plain()) {
            spec.action.version = MULTI_INSTRUMENT_ACTION_SPACE_VERSION;
            spec.action.instruments = self.instruments.clone();
        }
        Ok(spec)
    }
}

fn check_config(cfg: &RlConfig) -> MMResult<()> {
    if cfg.f32_unit <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(format!(
            "f32_unit must be positive, got {}",
            cfg.f32_unit
        )));
    }
    if cfg.invalid_action_penalty < Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(format!(
            "invalid_action_penalty must not be negative, got {}",
            cfg.invalid_action_penalty
        )));
    }
    Ok(())
}

fn check_instrument(index: usize, i: &InstrumentSpace) -> MMResult<()> {
    let refuse = |why: String| {
        Err(MMError::InvalidConfiguration(format!(
            "instrument {index} of the profile: {why}"
        )))
    };
    for (name, value) in [
        ("tick_size", i.tick_size),
        ("lot_size", i.lot_size),
        ("max_price", i.max_price),
        ("max_order_size", i.max_order_size),
    ] {
        if value <= Decimal::ZERO {
            return refuse(format!("{name} must be positive, got {value}"));
        }
    }
    if i.depth == 0 || i.max_orders == 0 {
        return refuse(format!(
            "depth and max_orders must be positive, got {} and {}",
            i.depth, i.max_orders
        ));
    }
    let offsets = i.bid_offsets_ticks.len();
    if offsets == 0 || offsets > 64 || i.ask_offsets_ticks.len() != offsets {
        return refuse(format!(
            "bid and ask offsets must be 1 to 64 values each, of the same length, got {} and {}",
            offsets,
            i.ask_offsets_ticks.len()
        ));
    }
    if i.sizes_lots.is_empty() || i.sizes_lots.len() > 64 || i.sizes_lots.contains(&0) {
        return refuse(format!(
            "sizes must be 1 to 64 positive lots, got {:?}",
            i.sizes_lots
        ));
    }
    Ok(())
}

/// The specification of one instrument under `config`, as an environment states it.
pub(super) fn build_spec(
    observation: ObservationSpace,
    instrument: &InstrumentSpace,
    config: &RlConfig,
    max_steps: u64,
    mark: RewardMark,
) -> RlSpec {
    RlSpec {
        observation,
        action: ActionSpace {
            version: if config.quote_lives.is_some() {
                EXPIRY_ACTION_SPACE_VERSION
            } else if config.cancel_action {
                CANCEL_ACTION_SPACE_VERSION
            } else {
                ACTION_SPACE_VERSION
            },
            tick_size: instrument.tick_size,
            lot_size: instrument.lot_size,
            max_price: instrument.max_price,
            max_order_size: instrument.max_order_size,
            bid_offsets_ticks: instrument.bid_offsets_ticks.clone(),
            ask_offsets_ticks: instrument.ask_offsets_ticks.clone(),
            sizes_lots: instrument.sizes_lots.clone(),
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
            instruments: Vec::new(),
        },
        reward: RewardSpec {
            version: REWARD_VERSION,
            kind: RewardKind::NetEquityChange,
            metric: MetricId::NetPnl,
            unit: MetricUnit::Quote,
            mark,
            invalid_action_penalty: config.invalid_action_penalty,
            f32_unit: config.f32_unit,
            policy_input: false,
        },
        episode: EpisodeSpec {
            version: EPISODE_VERSION,
            max_steps,
            terminal: config.terminal,
        },
    }
}

/// The grid of one instrument: bid offsets, ask offsets and sizes.
type Grid<'a> = (&'a [u16], &'a [u16], &'a [u64]);

impl RlSpec {
    fn grid(&self, instrument: usize) -> MMResult<Grid<'_>> {
        let a = &self.action;
        if a.instruments.is_empty() {
            if instrument == 0 {
                return Ok((&a.bid_offsets_ticks, &a.ask_offsets_ticks, &a.sizes_lots));
            }
        } else if let Some(i) = a.instruments.get(instrument) {
            return Ok((&i.bid_offsets_ticks, &i.ask_offsets_ticks, &i.sizes_lots));
        }
        Err(MMError::InvalidConfiguration(format!(
            "instrument head {instrument} is outside the specification's {} instruments",
            a.instruments.len().max(1)
        )))
    }

    fn lives(&self) -> usize {
        match &self.action.expiry {
            QuoteExpiry::Lives(lives) => lives.len(),
            QuoteExpiry::NotModelled => 0,
        }
    }

    fn cross_sizes(&self) -> usize {
        self.action
            .cross_hedge
            .as_ref()
            .map_or(0, |c| c.sizes_lots.len())
    }

    /// The number of values of each head, in the order of [`ActionHead::ALL`]: what a
    /// multi discrete policy samples. An offset or size head has the largest grid of the
    /// instruments, and a value past the chosen instrument's grid does not decode.
    #[must_use]
    pub fn action_head_sizes(&self) -> [usize; ACTION_HEAD_COUNT] {
        let a = &self.action;
        let (bids, asks, sizes) = if a.instruments.is_empty() {
            (
                a.bid_offsets_ticks.len(),
                a.ask_offsets_ticks.len(),
                a.sizes_lots.len(),
            )
        } else {
            a.instruments.iter().fold((0, 0, 0), |(b, k, s), i| {
                (
                    b.max(i.bid_offsets_ticks.len()),
                    k.max(i.ask_offsets_ticks.len()),
                    s.max(i.sizes_lots.len()),
                )
            })
        };
        [
            ACTION_KIND_COUNT,
            a.instruments.len().max(1),
            bids.max(1),
            asks.max(1),
            sizes.max(self.cross_sizes()).max(1),
            sizes.max(1),
            self.lives().max(1),
        ]
    }

    /// The action of a row of head values, in the order of [`ActionHead::ALL`]. A kind that
    /// the specification does not offer, such as a hedge without hedge actions, decodes all
    /// the same: the mask and the [`InvalidActionPolicy`](crate::rl::InvalidActionPolicy)
    /// decide on it.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a row that is not
    /// [`ACTION_HEAD_COUNT`] long, an unknown kind, an instrument outside the
    /// specification, or a head the kind reads that is outside its range for that
    /// instrument: its grid, the hedge sizes of the second instrument or the quote lives.
    pub fn decode_action(&self, heads: &[usize]) -> MMResult<InstrumentAction> {
        let row: &[usize; ACTION_HEAD_COUNT] = heads.try_into().map_err(|_| {
            MMError::InvalidConfiguration(format!(
                "an action row has {ACTION_HEAD_COUNT} heads, got {}",
                heads.len()
            ))
        })?;
        let [kind, instrument, bid, ask, bid_size, ask_size, life] = *row;
        let (bids, asks, sizes) = self.grid(instrument)?;
        let check = |head: ActionHead, value: usize, len: usize| {
            if value < len {
                Ok(value)
            } else {
                Err(MMError::InvalidConfiguration(format!(
                    "head {head:?} is {value}, outside its {len} values"
                )))
            }
        };
        let action = match kind {
            0 => RlAction::Hold,
            1 => RlAction::Quote {
                bid: check(ActionHead::Bid, bid, bids.len())?,
                ask: check(ActionHead::Ask, ask, asks.len())?,
                bid_size: check(ActionHead::BidSize, bid_size, sizes.len())?,
                ask_size: check(ActionHead::AskSize, ask_size, sizes.len())?,
            },
            2 => RlAction::OneSided {
                side: QuoteSide::Bid,
                offset: check(ActionHead::Bid, bid, bids.len())?,
                size: check(ActionHead::BidSize, bid_size, sizes.len())?,
            },
            3 => RlAction::OneSided {
                side: QuoteSide::Ask,
                offset: check(ActionHead::Ask, ask, asks.len())?,
                size: check(ActionHead::AskSize, ask_size, sizes.len())?,
            },
            4 => RlAction::Hedge {
                size: check(ActionHead::BidSize, bid_size, sizes.len())?,
            },
            5 => RlAction::CrossHedge {
                size: check(ActionHead::BidSize, bid_size, self.cross_sizes())?,
            },
            6 => RlAction::Cancel,
            7 => RlAction::ExpiringQuote {
                bid: check(ActionHead::Bid, bid, bids.len())?,
                ask: check(ActionHead::Ask, ask, asks.len())?,
                bid_size: check(ActionHead::BidSize, bid_size, sizes.len())?,
                ask_size: check(ActionHead::AskSize, ask_size, sizes.len())?,
                life: check(ActionHead::Life, life, self.lives())?,
            },
            other => {
                return Err(MMError::InvalidConfiguration(format!(
                    "head Kind is {other}, outside its {ACTION_KIND_COUNT} values"
                )));
            }
        };
        Ok(InstrumentAction { instrument, action })
    }

    /// The row of head values of `action`, with 0 in the heads its kind does not read: the
    /// inverse of [`RlSpec::decode_action`].
    ///
    /// # Errors
    ///
    /// As [`RlSpec::decode_action`] for the row it would give.
    pub fn encode_action(&self, action: &InstrumentAction) -> MMResult<[usize; ACTION_HEAD_COUNT]> {
        let row = |kind, bid, ask, bid_size, ask_size, life| {
            [kind, action.instrument, bid, ask, bid_size, ask_size, life]
        };
        let heads = match action.action {
            RlAction::Hold => row(0, 0, 0, 0, 0, 0),
            RlAction::Quote {
                bid,
                ask,
                bid_size,
                ask_size,
            } => row(1, bid, ask, bid_size, ask_size, 0),
            RlAction::OneSided {
                side: QuoteSide::Bid,
                offset,
                size,
            } => row(2, offset, 0, size, 0, 0),
            RlAction::OneSided {
                side: QuoteSide::Ask,
                offset,
                size,
            } => row(3, 0, offset, 0, size, 0),
            RlAction::Hedge { size } => row(4, 0, 0, size, 0, 0),
            RlAction::CrossHedge { size } => row(5, 0, 0, size, 0, 0),
            RlAction::Cancel => row(6, 0, 0, 0, 0, 0),
            RlAction::ExpiringQuote {
                bid,
                ask,
                bid_size,
                ask_size,
                life,
            } => row(7, bid, ask, bid_size, ask_size, life),
        };
        self.decode_action(&heads)?;
        Ok(heads)
    }
}
