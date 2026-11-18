//! The spaces of an options quoting agent over a moving window of series, and its net
//! delta.
//!
//! A run that lists many strikes over several expiries, and lists, expires and rolls
//! series inside an episode, cannot name its instruments in a fixed list. The instrument
//! head of [`OptionsProfile`] therefore indexes **relative slots**:
//!
//! - `E_w` expiry ranks ([`OptionsProfile::expiry_ranks`]): rank 0 is the nearest listed
//!   option expiry at the decision, rank 1 the next, over every expiry class; two expiries
//!   at the same instant are ordered by their symbol.
//! - `K_w` strike ranks per expiry ([`OptionsProfile::strike_ranks`]): the strikes of that
//!   expiry ordered by their distance to the agent's last delivered index mark, ties to the
//!   lower strike; rank 0 is the nearest.
//! - Both rights: slot `s = ((e * K_w) + k) * 2 + right`, with call 0 and put 1.
//! - One hedge lane after the option slots, index `2 * E_w * K_w`.
//!
//! Ranking is exact: distances are `Decimal` differences and expiries compare integer
//! instants, so no float takes part in naming a slot ([`OptionsProfile::resolve_slots`]).
//! A slot with no listed series is empty, and before the first delivered index mark every
//! option slot is empty.
//!
//! # Observation
//!
//! A `Box` of `f32`, each feature an exact `Decimal` or integer divided by a profile
//! divisor and clipped to `[-1, 1]`, 0 when it has no value. In order: the
//! [`SLOT_FEATURE_COUNT`] features of each option slot ([`SlotFeature`]) in slot order,
//! then the features of the hedge lane ([`FeatureId::ALL`] of
//! [`observation_features`]), then the [`AGENT_FEATURE_COUNT`] features of the agent
//! ([`AgentFeature`]). An empty slot is all zeros.
//!
//! # Action
//!
//! A `MultiDiscrete` of [`OPTIONS_HEAD_COUNT`] heads: kind ([`OptionsKind`]), instrument
//! (the option slots then the hedge lane), bid offset, ask offset and size, as
//! [`OptionsProfile::action_head_sizes`] gives. One decision acts on one instrument. A
//! hedge on an option slot and a quote kind on the hedge lane are refused by
//! [`OptionsProfile::decode_action`] from the heads alone.
//!
//! # Net delta
//!
//! [`net_delta`] sums the linear positions exactly and the options with a Black Scholes
//! delta at rate 0 and a constant volatility ([`OptionsProfile::sigma_delta`]). The float
//! part is reproducible across architectures: inputs are converted from `Decimal` to `f64`
//! once by correctly rounded parsing, `ln`, `sqrt` and `erfc` come from the `libm` crate
//! (pure Rust, the same code on every target), every other operation is an IEEE basic
//! operation (Rust never contracts a multiply and an add into an FMA, and nothing here
//! calls `mul_add`), and each delta is rounded half to even to `1e-9` from its exact binary
//! value before it returns to `Decimal`. The fixtures of [`RL_OPTIONS_FIXTURES`] are
//! checked on `x86_64` and `aarch64`.
//!
//! The reward is not specified here: each environment computes it from its own
//! accounting, so the reward field of the [`SpecId`] is 0.

use std::cmp::Ordering;

use rust_decimal::RoundingStrategy;
use rust_decimal::prelude::ToPrimitive;

use crate::Decimal;
use crate::backtest::kernel::Observation;
use crate::backtest::kernel::fnv1a128;
use crate::types::action::OptionStyle;
use crate::types::error::{MMError, MMResult};

use super::features::{FeatureVector, PreviousRow, observation_features};
use super::spec::{EPISODE_VERSION, FeatureId, ObservationScales, ObservationSpace, SpecId};

/// Version of the observation of [`OptionsProfile`]: the slot, hedge lane and agent
/// features and their order.
pub const OPTIONS_OBSERVATION_SPACE_VERSION: u16 = 1;

/// Version of the action of [`OptionsProfile`]: the heads, the kind codes and the refusals.
pub const OPTIONS_ACTION_SPACE_VERSION: u16 = 1;

/// The reward version of an [`OptionsProfile`] [`SpecId`]: 0, as the reward is the
/// environment's own.
pub const OPTIONS_REWARD_VERSION: u16 = 0;

/// Version of [`net_delta`] and [`black_scholes_delta`]: the formula, the float path and the
/// rounding. Part of the fingerprint.
pub const NET_DELTA_VERSION: u16 = 1;

/// Number of features per option slot, in the order of [`SlotFeature::ALL`].
pub const SLOT_FEATURE_COUNT: usize = 13;

/// Number of features of the agent, in the order of [`AgentFeature::ALL`].
pub const AGENT_FEATURE_COUNT: usize = 3;

/// Number of action heads, in the order of [`OptionsHead::ALL`].
pub const OPTIONS_HEAD_COUNT: usize = 5;

/// Number of values of [`OptionsHead::Kind`].
pub const OPTIONS_KIND_COUNT: usize = 6;

/// The largest `E_w` or `K_w`.
pub const MAX_WINDOW_RANKS: u16 = 64;

/// Decimal places of a rounded option delta.
pub const DELTA_DECIMALS: u32 = 9;

/// The fixtures of `doc/fixtures/rl_options.json`, for an environment's own tests:
/// profiles with their spec bytes and ids, slot resolutions, observations, head rows and
/// deltas. `doc/fixtures/RL_OPTIONS.md` describes the format.
pub const RL_OPTIONS_FIXTURES: &str = include_str!("../../doc/fixtures/rl_options.json");

const OPTIONS_MAGIC: [u8; 4] = *b"MMOQ";

/// A feature of an option slot, in the order of the slot's part of the vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SlotFeature {
    /// 1 when the slot names a listed series; the slot is all zeros otherwise.
    Present,
    /// Strike minus the delivered index mark, in quote per base.
    StrikeDistance,
    /// Expiry instant minus the decision instant, in milliseconds; negative once the
    /// instant has passed and the series is still listed in the view.
    TimeToExpiry,
    /// Observed mid minus the best bid, in ticks of the series. No value without both
    /// sides.
    BidOffset,
    /// Best ask minus the observed mid, in ticks. No value without both sides.
    AskOffset,
    /// Size at the best bid, in lots. No value without a bid.
    BidSize,
    /// Size at the best ask, in lots. No value without an ask.
    AskSize,
    /// The delivered venue mark of the series, in quote per base. No value without one.
    VenueMark,
    /// The agent's delivered position in the series, in lots, signed.
    Position,
    /// The agent's known resting bids.
    KnownBids,
    /// The agent's known resting asks.
    KnownAsks,
    /// The agent's pending bids: sent, with no outcome delivered.
    PendingBids,
    /// The agent's pending asks.
    PendingAsks,
}

impl SlotFeature {
    /// Every slot feature, in order.
    pub const ALL: [Self; SLOT_FEATURE_COUNT] = [
        Self::Present,
        Self::StrikeDistance,
        Self::TimeToExpiry,
        Self::BidOffset,
        Self::AskOffset,
        Self::BidSize,
        Self::AskSize,
        Self::VenueMark,
        Self::Position,
        Self::KnownBids,
        Self::KnownAsks,
        Self::PendingBids,
        Self::PendingAsks,
    ];
}

/// A feature of the agent, after the hedge lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AgentFeature {
    /// Equity minus the equity at the episode start, in quote units.
    EquityChange,
    /// The net delta of [`net_delta`], in base units. No value before the first index mark.
    NetDelta,
    /// Decisions taken so far, over `max_steps`.
    Decisions,
}

impl AgentFeature {
    /// Every agent feature, in order.
    pub const ALL: [Self; AGENT_FEATURE_COUNT] =
        [Self::EquityChange, Self::NetDelta, Self::Decisions];
}

/// The divisors of the slot and agent features. Each is positive.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OptionsScales {
    /// Divisor of [`SlotFeature::StrikeDistance`], in quote per base.
    pub strike_distance: Decimal,
    /// Divisor of [`SlotFeature::TimeToExpiry`], in milliseconds.
    pub time_to_expiry_ms: Decimal,
    /// Divisor of [`SlotFeature::BidOffset`] and [`SlotFeature::AskOffset`], in ticks.
    pub offset_ticks: Decimal,
    /// Divisor of the book sizes, in lots: the profile depth.
    pub depth_lots: Decimal,
    /// Divisor of [`SlotFeature::VenueMark`], in quote per base.
    pub venue_mark: Decimal,
    /// Divisor of [`SlotFeature::Position`], in lots.
    pub position_lots: Decimal,
    /// Divisor of [`AgentFeature::EquityChange`], in quote units.
    pub equity: Decimal,
    /// Divisor of [`AgentFeature::NetDelta`], in base units.
    pub net_delta: Decimal,
}

impl OptionsScales {
    /// The divisors, in the order of the fields.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        strike_distance: Decimal,
        time_to_expiry_ms: Decimal,
        offset_ticks: Decimal,
        depth_lots: Decimal,
        venue_mark: Decimal,
        position_lots: Decimal,
        equity: Decimal,
        net_delta: Decimal,
    ) -> Self {
        Self {
            strike_distance,
            time_to_expiry_ms,
            offset_ticks,
            depth_lots,
            venue_mark,
            position_lots,
            equity,
            net_delta,
        }
    }

    fn named(&self) -> [(&'static str, Decimal); 8] {
        [
            ("strike_distance", self.strike_distance),
            ("time_to_expiry_ms", self.time_to_expiry_ms),
            ("offset_ticks", self.offset_ticks),
            ("depth_lots", self.depth_lots),
            ("venue_mark", self.venue_mark),
            ("position_lots", self.position_lots),
            ("equity", self.equity),
            ("net_delta", self.net_delta),
        ]
    }
}

/// A profile of the options quoting agent: every constant its spaces and its net delta
/// depend on, fingerprinted in its [`SpecId`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OptionsProfile {
    /// `E_w`: expiry ranks in the window.
    pub expiry_ranks: u16,
    /// `K_w`: strike ranks per expiry.
    pub strike_ranks: u16,
    /// The constant volatility of [`net_delta`], per square root of a year. Positive.
    pub sigma_delta: Decimal,
    /// Divisors of the slot and agent features.
    pub scales: OptionsScales,
    /// Divisors of the hedge lane features, as an [`ObservationSpace::from_scales`].
    pub hedge_scales: ObservationScales,
    /// Bid offsets from the observed mid, in ticks, strictly increasing.
    pub bid_offsets_ticks: Vec<u16>,
    /// Ask offsets from the observed mid, in ticks, strictly increasing.
    pub ask_offsets_ticks: Vec<u16>,
    /// Order sizes, in lots, strictly increasing and positive; the hedge reads the same
    /// grid in the hedge lane's lots.
    pub sizes_lots: Vec<u64>,
    /// The most resting orders per side and series: the divisor of the order counts.
    pub max_orders: u16,
    /// Decisions before truncation.
    pub max_steps: u64,
    /// Time between two decisions, in nanoseconds.
    pub step_ns: u64,
}

impl OptionsProfile {
    /// A profile; [`OptionsProfile::validate`] checks it.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        expiry_ranks: u16,
        strike_ranks: u16,
        sigma_delta: Decimal,
        scales: OptionsScales,
        hedge_scales: ObservationScales,
        bid_offsets_ticks: Vec<u16>,
        ask_offsets_ticks: Vec<u16>,
        sizes_lots: Vec<u64>,
        max_orders: u16,
        max_steps: u64,
        step_ns: u64,
    ) -> Self {
        Self {
            expiry_ranks,
            strike_ranks,
            sigma_delta,
            scales,
            hedge_scales,
            bid_offsets_ticks,
            ask_offsets_ticks,
            sizes_lots,
            max_orders,
            max_steps,
            step_ns,
        }
    }

    /// Checks the profile.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for `sigma_delta <= 0`, `E_w` or `K_w` of
    /// 0 or above [`MAX_WINDOW_RANKS`], a divisor that is not positive, a zero
    /// `max_orders`, `max_steps` or `step_ns`, and a grid that is empty, longer than 64 or
    /// not strictly increasing, or a size of 0 lots.
    pub fn validate(&self) -> MMResult<()> {
        if self.sigma_delta <= Decimal::ZERO {
            return Err(config(format!(
                "sigma_delta must be positive, got {}",
                self.sigma_delta
            )));
        }
        for (name, ranks) in [
            ("expiry_ranks", self.expiry_ranks),
            ("strike_ranks", self.strike_ranks),
        ] {
            if ranks == 0 || ranks > MAX_WINDOW_RANKS {
                return Err(config(format!(
                    "{name} must be in 1..={MAX_WINDOW_RANKS}, got {ranks}"
                )));
            }
        }
        for (name, value) in self.scales.named() {
            if value <= Decimal::ZERO {
                return Err(config(format!(
                    "options scale {name} must be positive, got {value}"
                )));
            }
        }
        if self.max_orders == 0 || self.max_steps == 0 || self.step_ns == 0 {
            return Err(config(format!(
                "max_orders, max_steps and step_ns must be positive, got {}, {} and {}",
                self.max_orders, self.max_steps, self.step_ns
            )));
        }
        let offsets = |name: &str, grid: &[u16]| {
            increasing(
                name,
                &grid.iter().map(|v| u64::from(*v)).collect::<Vec<_>>(),
            )
        };
        offsets("bid_offsets_ticks", &self.bid_offsets_ticks)?;
        offsets("ask_offsets_ticks", &self.ask_offsets_ticks)?;
        increasing("sizes_lots", &self.sizes_lots)?;
        if self.sizes_lots.first() == Some(&0) {
            return Err(config("sizes_lots must be positive".to_string()));
        }
        self.hedge_space().map(|_| ())
    }

    /// The number of option slots, `2 * E_w * K_w`.
    #[must_use]
    pub fn option_slots(&self) -> usize {
        usize::from(self.expiry_ranks)
            .saturating_mul(usize::from(self.strike_ranks))
            .saturating_mul(2)
    }

    /// The index of the hedge lane in the instrument head: [`OptionsProfile::option_slots`].
    #[must_use]
    pub fn hedge_lane(&self) -> usize {
        self.option_slots()
    }

    /// The slot of `(expiry rank, strike rank, style)`, or `None` outside the window.
    #[must_use]
    pub fn slot_index(
        &self,
        expiry_rank: usize,
        strike_rank: usize,
        style: OptionStyle,
    ) -> Option<usize> {
        (expiry_rank < usize::from(self.expiry_ranks)
            && strike_rank < usize::from(self.strike_ranks))
        .then(|| {
            expiry_rank
                .saturating_mul(usize::from(self.strike_ranks))
                .saturating_add(strike_rank)
                .saturating_mul(2)
                .saturating_add(right_code(style))
        })
    }

    /// The `(expiry rank, strike rank, style)` of a slot, or `None` past the option slots.
    #[must_use]
    pub fn slot_ranks(&self, slot: usize) -> Option<(usize, usize, OptionStyle)> {
        if slot >= self.option_slots() {
            return None;
        }
        let k_w = usize::from(self.strike_ranks);
        let style = if slot.is_multiple_of(2) {
            OptionStyle::Call
        } else {
            OptionStyle::Put
        };
        let pair = slot / 2;
        Some((pair.checked_div(k_w)?, pair.checked_rem(k_w)?, style))
    }

    /// The hedge lane's observation space.
    ///
    /// # Errors
    ///
    /// As [`ObservationSpace::from_scales`].
    pub fn hedge_space(&self) -> MMResult<ObservationSpace> {
        ObservationSpace::from_scales(
            &self.hedge_scales,
            u64::from(self.max_orders),
            self.max_steps,
        )
    }

    /// The length of the observation: `option_slots * SLOT_FEATURE_COUNT`, then the hedge
    /// lane features, then [`AGENT_FEATURE_COUNT`].
    #[must_use]
    pub fn observation_len(&self) -> usize {
        self.option_slots()
            .saturating_mul(SLOT_FEATURE_COUNT)
            .saturating_add(FeatureId::ALL.len())
            .saturating_add(AGENT_FEATURE_COUNT)
    }

    /// The number of values of each head, in the order of [`OptionsHead::ALL`].
    #[must_use]
    pub fn action_head_sizes(&self) -> [usize; OPTIONS_HEAD_COUNT] {
        [
            OPTIONS_KIND_COUNT,
            self.option_slots().saturating_add(1),
            self.bid_offsets_ticks.len(),
            self.ask_offsets_ticks.len(),
            self.sizes_lots.len(),
        ]
    }

    /// The canonical bytes of the profile, the input of its fingerprint.
    ///
    /// # Errors
    ///
    /// As [`OptionsProfile::validate`].
    pub fn to_bytes(&self) -> MMResult<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(&OPTIONS_MAGIC);
        for v in [
            OPTIONS_OBSERVATION_SPACE_VERSION,
            OPTIONS_ACTION_SPACE_VERSION,
            OPTIONS_REWARD_VERSION,
            EPISODE_VERSION,
            NET_DELTA_VERSION,
            self.expiry_ranks,
            self.strike_ranks,
            self.max_orders,
        ] {
            put_u16(&mut out, v);
        }
        put_u64(&mut out, SLOT_FEATURE_COUNT as u64);
        put_u64(&mut out, AGENT_FEATURE_COUNT as u64);
        put_decimal(&mut out, self.sigma_delta);
        for (_, value) in self.scales.named() {
            put_decimal(&mut out, value);
        }
        let hedge = self.hedge_space()?;
        put_u64(&mut out, hedge.features.len() as u64);
        for feature in &hedge.features {
            put_u64(&mut out, feature.id.index() as u64);
            put_u16(&mut out, feature.unit.code());
            put_u16(&mut out, feature.as_of.code());
            put_u16(&mut out, feature.horizon.code());
            put_u16(&mut out, feature.memory.code());
            put_decimal(&mut out, feature.divisor);
        }
        for grid in [&self.bid_offsets_ticks, &self.ask_offsets_ticks] {
            put_u64(&mut out, grid.len() as u64);
            for v in grid.iter() {
                put_u16(&mut out, *v);
            }
        }
        put_u64(&mut out, self.sizes_lots.len() as u64);
        for v in &self.sizes_lots {
            put_u64(&mut out, *v);
        }
        put_u64(&mut out, self.max_steps);
        put_u64(&mut out, self.step_ns);
        Ok(out)
    }

    /// The identity of the profile's spaces.
    ///
    /// # Errors
    ///
    /// As [`OptionsProfile::validate`].
    pub fn id(&self) -> MMResult<SpecId> {
        Ok(SpecId {
            observation: OPTIONS_OBSERVATION_SPACE_VERSION,
            action: OPTIONS_ACTION_SPACE_VERSION,
            reward: OPTIONS_REWARD_VERSION,
            episode: EPISODE_VERSION,
            fingerprint: fnv1a128(&self.to_bytes()?),
        })
    }

    /// Checks that this is the profile a consumer was built for.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] when the id differs, or as
    /// [`OptionsProfile::validate`].
    pub fn ensure(&self, expected: SpecId) -> MMResult<()> {
        let id = self.id()?;
        if id == expected {
            return Ok(());
        }
        Err(config(format!(
            "unknown options specification: expected {expected:?}, this is {id:?}"
        )))
    }

    /// The window at a decision: for each option slot, the index into `listed` of the
    /// series it names, or `None`.
    ///
    /// `listed` is what the agent has been delivered as listed and not delisted, whatever
    /// its expiry instant. With no `index_mark` every slot is empty.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] when two entries share an expiry symbol,
    /// a strike and a style, or one expiry symbol has two instants.
    pub fn resolve_slots(
        &self,
        index_mark: Option<Decimal>,
        listed: &[ListedOption],
    ) -> MMResult<Vec<Option<usize>>> {
        let mut slots = vec![None; self.option_slots()];
        let Some(mark) = index_mark else {
            return Ok(slots);
        };
        let mut expiries: Vec<(u64, &str)> = Vec::new();
        for item in listed {
            match expiries
                .iter()
                .find(|(_, s)| *s == item.expiry_symbol.as_str())
            {
                Some((ns, _)) if *ns != item.expiry_ns => {
                    return Err(config(format!(
                        "expiry {} has two instants",
                        item.expiry_symbol
                    )));
                }
                Some(_) => {}
                None => expiries.push((item.expiry_ns, item.expiry_symbol.as_str())),
            }
        }
        expiries.sort_unstable();
        for (e, (_, symbol)) in expiries
            .iter()
            .take(usize::from(self.expiry_ranks))
            .enumerate()
        {
            let mut strikes: Vec<Decimal> = listed
                .iter()
                .filter(|i| i.expiry_symbol == *symbol)
                .map(|i| i.strike)
                .collect();
            strikes.sort_unstable();
            strikes.dedup();
            strikes.sort_by(|a, b| {
                let da = a.checked_sub(mark).map(|d| d.abs());
                let db = b.checked_sub(mark).map(|d| d.abs());
                da.cmp(&db).then(a.cmp(b))
            });
            for (k, strike) in strikes
                .iter()
                .take(usize::from(self.strike_ranks))
                .enumerate()
            {
                for style in [OptionStyle::Call, OptionStyle::Put] {
                    let mut found = listed.iter().enumerate().filter(|(_, i)| {
                        i.expiry_symbol == *symbol && i.strike == *strike && i.style == style
                    });
                    let first = found.next().map(|(n, _)| n);
                    if found.next().is_some() {
                        return Err(config(format!(
                            "series {symbol} {strike} {style:?} is listed twice"
                        )));
                    }
                    if let Some(slot) = self.slot_index(e, k, style)
                        && let Some(cell) = slots.get_mut(slot) {
                            *cell = first;
                        }
                }
            }
        }
        Ok(slots)
    }

    /// The observation at a decision at `decision_ns`.
    ///
    /// `slots` has one entry per option slot, `None` for an empty one; `hedge` is the hedge
    /// lane's observation, previous row and tick, as [`observation_features`] takes them.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a profile that does not validate, a
    /// number of slots other than [`OptionsProfile::option_slots`], a filled slot with no
    /// `index_mark` or a tick that is not positive, and [`MMError::NumericalError`] if a
    /// value does not convert.
    pub fn observation(
        &self,
        decision_ns: u64,
        index_mark: Option<Decimal>,
        slots: &[Option<SlotInput>],
        hedge: (&Observation, Option<PreviousRow>, Decimal),
        agent: &AgentInput,
    ) -> MMResult<OptionsObservation> {
        self.validate()?;
        if slots.len() != self.option_slots() {
            return Err(config(format!(
                "the observation has {} option slots, got {}",
                self.option_slots(),
                slots.len()
            )));
        }
        let s = &self.scales;
        let orders = Decimal::from(self.max_orders);
        let slot_divisors = [
            Decimal::ONE,
            s.strike_distance,
            s.time_to_expiry_ms,
            s.offset_ticks,
            s.offset_ticks,
            s.depth_lots,
            s.depth_lots,
            s.venue_mark,
            s.position_lots,
            orders,
            orders,
            orders,
            orders,
        ];
        let mut raw = Vec::with_capacity(self.observation_len());
        let mut values = Vec::with_capacity(self.observation_len());
        for slot in slots {
            let features = match slot {
                None => [None; SLOT_FEATURE_COUNT],
                Some(input) => {
                    let mark = index_mark
                        .ok_or_else(|| config("a filled slot needs an index mark".to_string()))?;
                    input.raw(decision_ns, mark)?
                }
            };
            for (value, divisor) in features.iter().zip(slot_divisors) {
                raw.push(*value);
                values.push(normalize(*value, divisor)?);
            }
        }
        let (obs, previous, tick) = hedge;
        let lane: FeatureVector =
            observation_features(&self.hedge_space()?, obs, previous, tick, self.max_steps)?;
        raw.extend_from_slice(&lane.raw);
        values.extend_from_slice(&lane.values);
        let agent_raw = [
            Some(agent.equity_change),
            agent.net_delta,
            Some(Decimal::from(agent.decisions)),
        ];
        let agent_divisors = [s.equity, s.net_delta, Decimal::from(self.max_steps)];
        for (value, divisor) in agent_raw.iter().zip(agent_divisors) {
            raw.push(*value);
            values.push(normalize(*value, divisor)?);
        }
        Ok(OptionsObservation { raw, values })
    }

    /// The action of a row of head values, in the order of [`OptionsHead::ALL`].
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidAction`] for a row that is not [`OPTIONS_HEAD_COUNT`]
    /// long, a kind or instrument outside its head, a hedge on an option slot, a quote, a
    /// one sided order on the hedge lane, or a head the kind reads outside its grid.
    pub fn decode_action(&self, heads: &[usize]) -> MMResult<OptionsAction> {
        let row: &[usize; OPTIONS_HEAD_COUNT] = heads.try_into().map_err(|_| {
            invalid(format!(
                "an action row has {OPTIONS_HEAD_COUNT} heads, got {}",
                heads.len()
            ))
        })?;
        let [kind, instrument, bid, ask, size] = *row;
        let kind = OptionsKind::from_code(kind).ok_or_else(|| {
            invalid(format!(
                "kind {kind} is outside its {OPTIONS_KIND_COUNT} values"
            ))
        })?;
        let lane = if instrument < self.option_slots() {
            Lane::Option(instrument)
        } else if instrument == self.hedge_lane() {
            Lane::Hedge
        } else {
            return Err(invalid(format!(
                "instrument {instrument} is outside its {} values",
                self.option_slots().saturating_add(1)
            )));
        };
        match (kind, lane) {
            (OptionsKind::Hedge, Lane::Option(_)) => {
                return Err(invalid("a hedge on an option slot is refused".to_string()));
            }
            (OptionsKind::Quote | OptionsKind::BidOnly | OptionsKind::AskOnly, Lane::Hedge) => {
                return Err(invalid("a quote on the hedge lane is refused".to_string()));
            }
            _ => {}
        }
        let check = |head: OptionsHead, value: usize, len: usize| {
            if value < len {
                Ok(value)
            } else {
                Err(invalid(format!(
                    "head {head:?} is {value}, outside its {len} values"
                )))
            }
        };
        let bids = self.bid_offsets_ticks.len();
        let asks = self.ask_offsets_ticks.len();
        let sizes = self.sizes_lots.len();
        let (bid, ask, size) = match kind {
            OptionsKind::Hold | OptionsKind::Cancel => (0, 0, 0),
            OptionsKind::Quote => (
                check(OptionsHead::Bid, bid, bids)?,
                check(OptionsHead::Ask, ask, asks)?,
                check(OptionsHead::Size, size, sizes)?,
            ),
            OptionsKind::BidOnly => (
                check(OptionsHead::Bid, bid, bids)?,
                0,
                check(OptionsHead::Size, size, sizes)?,
            ),
            OptionsKind::AskOnly => (
                0,
                check(OptionsHead::Ask, ask, asks)?,
                check(OptionsHead::Size, size, sizes)?,
            ),
            OptionsKind::Hedge => (0, 0, check(OptionsHead::Size, size, sizes)?),
        };
        Ok(OptionsAction {
            kind,
            lane,
            bid,
            ask,
            size,
        })
    }

    /// The canonical row of `action`, with 0 in the heads its kind does not read.
    ///
    /// # Errors
    ///
    /// As [`OptionsProfile::decode_action`] for that row.
    pub fn encode_action(&self, action: &OptionsAction) -> MMResult<[usize; OPTIONS_HEAD_COUNT]> {
        let instrument = match action.lane {
            Lane::Option(slot) => slot,
            Lane::Hedge => self.hedge_lane(),
        };
        let row = [
            action.kind.code(),
            instrument,
            action.bid,
            action.ask,
            action.size,
        ];
        let decoded = self.decode_action(&row)?;
        Ok([
            decoded.kind.code(),
            instrument,
            decoded.bid,
            decoded.ask,
            decoded.size,
        ])
    }

    /// [`net_delta`] with this profile's `sigma_delta`.
    ///
    /// # Errors
    ///
    /// As [`net_delta`].
    pub fn net_delta(
        &self,
        index_mark: Decimal,
        decision_ns: u64,
        day_ns: u64,
        linear: &[LinearPosition],
        options: &[OptionPosition],
    ) -> MMResult<Decimal> {
        net_delta(
            self.sigma_delta,
            index_mark,
            decision_ns,
            day_ns,
            linear,
            options,
        )
    }
}

/// A listed option series in the agent's delivered view.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ListedOption {
    /// The expiry's symbol, which orders two expiries at the same instant.
    pub expiry_symbol: String,
    /// The expiry instant, in nanoseconds.
    pub expiry_ns: u64,
    /// The strike, in quote per base.
    pub strike: Decimal,
    /// Call or put.
    pub style: OptionStyle,
}

impl ListedOption {
    /// A listed series.
    #[must_use]
    pub fn new(
        expiry_symbol: impl Into<String>,
        expiry_ns: u64,
        strike: Decimal,
        style: OptionStyle,
    ) -> Self {
        Self {
            expiry_symbol: expiry_symbol.into(),
            expiry_ns,
            strike,
            style,
        }
    }
}

/// What the agent has been delivered about the series of a filled slot.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SlotInput {
    /// The strike.
    pub strike: Decimal,
    /// The expiry instant, in nanoseconds.
    pub expiry_ns: u64,
    /// The series' tick, positive.
    pub tick_size: Decimal,
    /// The best bid: price and size in lots.
    pub best_bid: Option<(Decimal, Decimal)>,
    /// The best ask: price and size in lots.
    pub best_ask: Option<(Decimal, Decimal)>,
    /// The delivered venue mark.
    pub venue_mark: Option<Decimal>,
    /// The agent's position, in lots.
    pub position_lots: Decimal,
    /// Known resting orders, bid then ask.
    pub known_orders: [u64; 2],
    /// Pending orders, bid then ask.
    pub pending_orders: [u64; 2],
}

impl SlotInput {
    /// A slot with no book, no mark, no position and no orders.
    #[must_use]
    pub fn new(strike: Decimal, expiry_ns: u64, tick_size: Decimal) -> Self {
        Self {
            strike,
            expiry_ns,
            tick_size,
            best_bid: None,
            best_ask: None,
            venue_mark: None,
            position_lots: Decimal::ZERO,
            known_orders: [0; 2],
            pending_orders: [0; 2],
        }
    }

    /// Sets the best bid and ask.
    #[must_use]
    pub fn with_book(
        mut self,
        bid: Option<(Decimal, Decimal)>,
        ask: Option<(Decimal, Decimal)>,
    ) -> Self {
        self.best_bid = bid;
        self.best_ask = ask;
        self
    }

    /// Sets the venue mark.
    #[must_use]
    pub fn with_venue_mark(mut self, mark: Decimal) -> Self {
        self.venue_mark = Some(mark);
        self
    }

    /// Sets the position and the order counts.
    #[must_use]
    pub fn with_account(
        mut self,
        position_lots: Decimal,
        known: [u64; 2],
        pending: [u64; 2],
    ) -> Self {
        self.position_lots = position_lots;
        self.known_orders = known;
        self.pending_orders = pending;
        self
    }

    fn raw(
        &self,
        decision_ns: u64,
        mark: Decimal,
    ) -> MMResult<[Option<Decimal>; SLOT_FEATURE_COUNT]> {
        if self.tick_size <= Decimal::ZERO {
            return Err(config(format!(
                "a slot tick must be positive, got {}",
                self.tick_size
            )));
        }
        let to_expiry_ns = i128::from(self.expiry_ns)
            .checked_sub(i128::from(decision_ns))
            .ok_or_else(|| numeric("time to expiry"))?;
        let to_expiry_ms = Decimal::try_from_i128_with_scale(to_expiry_ns, 6)
            .map_err(|_| numeric("time to expiry"))?;
        let (bid_offset, ask_offset) = match (self.best_bid, self.best_ask) {
            (Some((bid, _)), Some((ask, _))) => {
                let mid2 = bid.checked_add(ask).ok_or_else(|| numeric("mid"))?;
                let tick2 = self
                    .tick_size
                    .checked_mul(Decimal::TWO)
                    .ok_or_else(|| numeric("tick"))?;
                let bid2 = bid.checked_add(bid).ok_or_else(|| numeric("bid"))?;
                let ask2 = ask.checked_add(ask).ok_or_else(|| numeric("ask"))?;
                (
                    Some(
                        mid2.checked_sub(bid2)
                            .and_then(|v| v.checked_div(tick2))
                            .ok_or_else(|| numeric("bid offset"))?,
                    ),
                    Some(
                        ask2.checked_sub(mid2)
                            .and_then(|v| v.checked_div(tick2))
                            .ok_or_else(|| numeric("ask offset"))?,
                    ),
                )
            }
            _ => (None, None),
        };
        let count = |v: u64| Some(Decimal::from(v));
        Ok([
            Some(Decimal::ONE),
            Some(
                self.strike
                    .checked_sub(mark)
                    .ok_or_else(|| numeric("strike distance"))?,
            ),
            Some(to_expiry_ms),
            bid_offset,
            ask_offset,
            self.best_bid.map(|(_, size)| size),
            self.best_ask.map(|(_, size)| size),
            self.venue_mark,
            Some(self.position_lots),
            count(self.known_orders[0]),
            count(self.known_orders[1]),
            count(self.pending_orders[0]),
            count(self.pending_orders[1]),
        ])
    }
}

/// The agent's own figures at a decision.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AgentInput {
    /// Equity minus the equity at the episode start.
    pub equity_change: Decimal,
    /// The net delta, `None` before the first index mark.
    pub net_delta: Option<Decimal>,
    /// Decisions taken so far.
    pub decisions: u64,
}

impl AgentInput {
    /// The agent's figures.
    #[must_use]
    pub fn new(equity_change: Decimal, net_delta: Option<Decimal>, decisions: u64) -> Self {
        Self {
            equity_change,
            net_delta,
            decisions,
        }
    }
}

/// The observation of an [`OptionsProfile`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OptionsObservation {
    /// Each feature before normalization, `None` with no value.
    pub raw: Vec<Option<Decimal>>,
    /// `raw / divisor` clipped to `[-1, 1]`, 0 with no value: the policy's input.
    pub values: Vec<f32>,
}

/// An action head of an [`OptionsProfile`], at its position in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptionsHead {
    /// [`OptionsKind`].
    Kind,
    /// An option slot, or the hedge lane after them.
    Instrument,
    /// Index of the bid offset.
    Bid,
    /// Index of the ask offset.
    Ask,
    /// Index of the size, for a quote, a one sided order or a hedge.
    Size,
}

impl OptionsHead {
    /// Every head, in the order of a row.
    pub const ALL: [Self; OPTIONS_HEAD_COUNT] = [
        Self::Kind,
        Self::Instrument,
        Self::Bid,
        Self::Ask,
        Self::Size,
    ];
}

/// The kind of an action, by its code in [`OptionsHead::Kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptionsKind {
    /// 0: no new command.
    Hold,
    /// 1: a two sided quote on an option slot.
    Quote,
    /// 2: a bid only on an option slot.
    BidOnly,
    /// 3: an ask only on an option slot.
    AskOnly,
    /// 4: a hedge on the hedge lane, reducing the net delta.
    Hedge,
    /// 5: cancel the agent's orders on the instrument.
    Cancel,
}

impl OptionsKind {
    /// The kind of `code`.
    #[must_use]
    pub const fn from_code(code: usize) -> Option<Self> {
        Some(match code {
            0 => Self::Hold,
            1 => Self::Quote,
            2 => Self::BidOnly,
            3 => Self::AskOnly,
            4 => Self::Hedge,
            5 => Self::Cancel,
            _ => return None,
        })
    }

    /// Its code.
    #[must_use]
    pub const fn code(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Quote => 1,
            Self::BidOnly => 2,
            Self::AskOnly => 3,
            Self::Hedge => 4,
            Self::Cancel => 5,
        }
    }
}

/// The instrument of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// An option slot.
    Option(usize),
    /// The hedge lane.
    Hedge,
}

/// A decoded action; heads its kind does not read are 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OptionsAction {
    /// The kind.
    pub kind: OptionsKind,
    /// The instrument.
    pub lane: Lane,
    /// Index into the bid offsets.
    pub bid: usize,
    /// Index into the ask offsets.
    pub ask: usize,
    /// Index into the sizes.
    pub size: usize,
}

impl OptionsAction {
    /// An action.
    #[must_use]
    pub fn new(kind: OptionsKind, lane: Lane, bid: usize, ask: usize, size: usize) -> Self {
        Self {
            kind,
            lane,
            bid,
            ask,
            size,
        }
    }
}

/// A perpetual or future position: `lots * lot_size` base units, exact.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LinearPosition {
    /// Signed lots.
    pub lots: Decimal,
    /// Base units per lot.
    pub lot_size: Decimal,
}

impl LinearPosition {
    /// A linear position.
    #[must_use]
    pub fn new(lots: Decimal, lot_size: Decimal) -> Self {
        Self { lots, lot_size }
    }
}

/// An option position.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OptionPosition {
    /// Signed lots.
    pub lots: Decimal,
    /// Base units per lot.
    pub base_per_lot: Decimal,
    /// The strike, positive.
    pub strike: Decimal,
    /// Call or put.
    pub style: OptionStyle,
    /// The expiry instant, in nanoseconds.
    pub expiry_ns: u64,
}

impl OptionPosition {
    /// An option position.
    #[must_use]
    pub fn new(
        lots: Decimal,
        base_per_lot: Decimal,
        strike: Decimal,
        style: OptionStyle,
        expiry_ns: u64,
    ) -> Self {
        Self {
            lots,
            base_per_lot,
            strike,
            style,
            expiry_ns,
        }
    }
}

/// The net delta of an agent, in base units.
///
/// The sum of every linear position's `lots * lot_size` and every option position's
/// `lots * base_per_lot * delta`, with `delta` from [`black_scholes_delta`] at `S` the
/// index mark, `K` the strike, `tau` the time from `decision_ns` to the expiry divided by
/// `365 * day_ns` and `sigma` the given volatility. Positions of zero lots are skipped. The
/// products and the sum are exact `Decimal`.
///
/// # Errors
///
/// As [`black_scholes_delta`], [`MMError::InvalidConfiguration`] for a zero `day_ns`, and
/// [`MMError::NumericalError`] on overflow.
pub fn net_delta(
    sigma: Decimal,
    index_mark: Decimal,
    decision_ns: u64,
    day_ns: u64,
    linear: &[LinearPosition],
    options: &[OptionPosition],
) -> MMResult<Decimal> {
    if day_ns == 0 {
        return Err(config("day_ns must be positive".to_string()));
    }
    let year = Decimal::from(day_ns)
        .checked_mul(Decimal::from(365))
        .ok_or_else(|| numeric("year"))?;
    let mut total = Decimal::ZERO;
    for p in linear {
        let base = p
            .lots
            .checked_mul(p.lot_size)
            .ok_or_else(|| numeric("linear delta"))?;
        total = total
            .checked_add(base)
            .ok_or_else(|| numeric("net delta"))?;
    }
    for p in options.iter().filter(|p| !p.lots.is_zero()) {
        let to_expiry = i128::from(p.expiry_ns)
            .checked_sub(i128::from(decision_ns))
            .ok_or_else(|| numeric("time to expiry"))?;
        let tau = Decimal::try_from_i128_with_scale(to_expiry, 0)
            .map_err(|_| numeric("time to expiry"))?
            .checked_div(year)
            .ok_or_else(|| numeric("tau"))?;
        let delta = black_scholes_delta(index_mark, p.strike, tau, sigma, p.style)?;
        let base = p
            .lots
            .checked_mul(p.base_per_lot)
            .and_then(|b| b.checked_mul(delta))
            .ok_or_else(|| numeric("option delta"))?;
        total = total
            .checked_add(base)
            .ok_or_else(|| numeric("net delta"))?;
    }
    Ok(total)
}

/// The most hedge lots a hedge may trade: `|net_delta| / lot_size`, rounded toward zero.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a lot size that is not positive and
/// [`MMError::NumericalError`] when the result does not fit.
pub fn hedge_cap_lots(net_delta: Decimal, lot_size: Decimal) -> MMResult<u64> {
    if lot_size <= Decimal::ZERO {
        return Err(config(format!("lot size must be positive, got {lot_size}")));
    }
    net_delta
        .abs()
        .checked_div(lot_size)
        .map(|v| v.trunc())
        .and_then(|v| v.to_u64())
        .ok_or_else(|| numeric("hedge cap"))
}

/// The Black Scholes delta at rate 0, rounded half to even to [`DELTA_DECIMALS`].
///
/// `N(d1)` for a call and `N(d1) - 1` for a put, with
/// `d1 = (ln(S / K) + sigma^2 * tau / 2) / (sigma * sqrt(tau))` and
/// `N(x) = erfc(-x / sqrt(2)) / 2`. At or past expiry (`tau <= 0`): 1 for a call with
/// `S > K`, -1 for a put with `S < K`, 0 otherwise, exact. `tau` is in years.
///
/// The float path is fixed: `S`, `K`, `tau` and `sigma` are converted to `f64` once, by
/// correctly rounded parsing of their decimal form; `ln`, `sqrt` and `erfc` are the `libm`
/// crate's; every other operation is one IEEE basic operation in the order written above,
/// with no fused multiply add. The result is rounded from its exact binary value.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for a spot, strike or `sigma` that is not
/// positive, and [`MMError::NumericalError`] for a value that does not convert or a
/// result that is not finite.
pub fn black_scholes_delta(
    spot: Decimal,
    strike: Decimal,
    tau: Decimal,
    sigma: Decimal,
    style: OptionStyle,
) -> MMResult<Decimal> {
    for (name, value) in [("spot", spot), ("strike", strike), ("sigma", sigma)] {
        if value <= Decimal::ZERO {
            return Err(config(format!("{name} must be positive, got {value}")));
        }
    }
    if tau <= Decimal::ZERO {
        return Ok(match (style, spot.cmp(&strike)) {
            (OptionStyle::Call, Ordering::Greater) => Decimal::ONE,
            (OptionStyle::Put, Ordering::Less) => Decimal::NEGATIVE_ONE,
            _ => Decimal::ZERO,
        });
    }
    let s = to_f64(spot)?;
    let k = to_f64(strike)?;
    let t = to_f64(tau)?;
    let v = to_f64(sigma)?;
    let numerator = libm::log(s / k) + ((v * v) * t) / 2.0;
    let denominator = v * libm::sqrt(t);
    let d1 = numerator / denominator;
    let n = libm::erfc(-d1 / std::f64::consts::SQRT_2) / 2.0;
    let delta = match style {
        OptionStyle::Call => n,
        OptionStyle::Put => n - 1.0,
    };
    round_half_even(delta, DELTA_DECIMALS)
}

/// `value` as the nearest `f64`, by Rust's correctly rounded parsing of its decimal form.
fn to_f64(value: Decimal) -> MMResult<f64> {
    let parsed: f64 = value
        .normalize()
        .to_string()
        .parse()
        .map_err(|_| numeric("decimal to f64"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(numeric("decimal to f64"))
    }
}

/// `x` rounded half to even to `decimals` places, from its exact binary value.
///
/// Rust prints the exact decimal expansion of an `f64` with enough digits (an `f64` has at
/// most 1074 fractional digits), so the digit after the last kept one and whether anything
/// follows it decide the rounding with no float arithmetic.
fn round_half_even(x: f64, decimals: u32) -> MMResult<Decimal> {
    if !x.is_finite() {
        return Err(numeric("delta is not finite"));
    }
    let exact = format!("{:.1074}", x.abs());
    let (int_part, frac) = exact.split_once('.').ok_or_else(|| numeric("delta"))?;
    let digits = frac.as_bytes();
    let keep = usize::try_from(decimals).map_err(|_| numeric("delta"))?;
    let kept = frac.get(..keep).ok_or_else(|| numeric("delta"))?;
    let next = digits.get(keep).copied().ok_or_else(|| numeric("delta"))?;
    let rest_nonzero = digits
        .get(keep.saturating_add(1)..)
        .is_some_and(|rest| rest.iter().any(|d| *d != b'0'));
    let truncated =
        Decimal::from_str_exact(&format!("{int_part}.{kept}")).map_err(|_| numeric("delta"))?;
    let odd = |d: &u8| d.checked_sub(b'0').is_some_and(|v| v & 1 == 1);
    let last_odd = kept
        .as_bytes()
        .last()
        .or_else(|| int_part.as_bytes().last())
        .is_some_and(odd);
    let up = match next.cmp(&b'5') {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => rest_nonzero || last_odd,
    };
    let magnitude = if up {
        truncated
            .checked_add(Decimal::new(1, decimals))
            .ok_or_else(|| numeric("delta"))?
    } else {
        truncated
    };
    let signed = if x.is_sign_negative() {
        Decimal::ZERO
            .checked_sub(magnitude)
            .ok_or_else(|| numeric("delta"))?
    } else {
        magnitude
    };
    Ok(signed
        .round_dp_with_strategy(decimals, RoundingStrategy::MidpointNearestEven)
        .normalize())
}

fn normalize(value: Option<Decimal>, divisor: Decimal) -> MMResult<f32> {
    match value {
        None => Ok(0.0),
        Some(v) => v
            .checked_div(divisor)
            .ok_or_else(|| numeric("normalized feature"))?
            .clamp(Decimal::NEGATIVE_ONE, Decimal::ONE)
            .to_f32()
            .ok_or_else(|| numeric("normalized feature")),
    }
}

fn increasing(name: &str, grid: &[u64]) -> MMResult<()> {
    if grid.is_empty() || grid.len() > 64 {
        return Err(config(format!(
            "{name} must hold 1 to 64 values, got {}",
            grid.len()
        )));
    }
    if grid.windows(2).any(|w| w.first() >= w.get(1)) {
        return Err(config(format!("{name} must be strictly increasing")));
    }
    Ok(())
}

const fn right_code(style: OptionStyle) -> usize {
    match style {
        OptionStyle::Call => 0,
        OptionStyle::Put => 1,
    }
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_decimal(out: &mut Vec<u8>, value: Decimal) {
    out.extend_from_slice(&value.normalize().serialize());
}

fn config(message: String) -> MMError {
    MMError::InvalidConfiguration(message)
}

fn invalid(message: String) -> MMError {
    MMError::InvalidAction(message)
}

fn numeric(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} does not convert"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn rounding_is_half_even_on_the_exact_value() {
        // 2^-10 = 0.0009765625 is a tie at 9 places: kept digit 2 is even.
        assert_eq!(
            round_half_even(0.000_976_562_5, 9).unwrap(),
            dec!(0.000976562)
        );
        // 3 * 2^-10 = 0.0029296875: kept digit 7 is odd, so it rounds up.
        assert_eq!(
            round_half_even(0.002_929_687_5, 9).unwrap(),
            dec!(0.002929688)
        );
        assert_eq!(round_half_even(-0.5, 9).unwrap(), dec!(-0.5));
        assert_eq!(round_half_even(0.999_999_999_9, 9).unwrap(), dec!(1));
    }

    #[test]
    fn at_the_money_call_and_put_differ_by_one() {
        let call = black_scholes_delta(
            dec!(100),
            dec!(100),
            dec!(0.25),
            dec!(0.5),
            OptionStyle::Call,
        )
        .unwrap();
        let put = black_scholes_delta(
            dec!(100),
            dec!(100),
            dec!(0.25),
            dec!(0.5),
            OptionStyle::Put,
        )
        .unwrap();
        assert_eq!(call - put, Decimal::ONE);
        assert!(call > dec!(0.5) && call < dec!(0.6));
    }

    #[test]
    fn expired_deltas_are_intrinsic() {
        let d = |s, k, style| black_scholes_delta(s, k, Decimal::ZERO, dec!(0.5), style).unwrap();
        assert_eq!(d(dec!(101), dec!(100), OptionStyle::Call), Decimal::ONE);
        assert_eq!(d(dec!(100), dec!(100), OptionStyle::Call), Decimal::ZERO);
        assert_eq!(
            d(dec!(99), dec!(100), OptionStyle::Put),
            Decimal::NEGATIVE_ONE
        );
        assert_eq!(d(dec!(100), dec!(100), OptionStyle::Put), Decimal::ZERO);
    }

    #[test]
    fn hedge_cap_rounds_toward_zero() {
        assert_eq!(hedge_cap_lots(dec!(-2.7), dec!(1)).unwrap(), 2);
        assert_eq!(hedge_cap_lots(dec!(0.09), dec!(0.1)).unwrap(), 0);
    }
}
