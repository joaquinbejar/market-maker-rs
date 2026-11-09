//! The descriptors of the spaces, their versions and their identity.

use crate::Decimal;
use crate::backtest::evaluation::{HedgeNotModelledReason, MetricId, MetricUnit};
use crate::backtest::kernel::fnv1a128;
use crate::types::error::{MMError, MMResult};

/// Version of the observation space. Raised when a feature, its order or its meaning changes.
pub const OBSERVATION_SPACE_VERSION: u16 = 1;
/// Version of the action space. Raised when an action or its meaning changes.
pub const ACTION_SPACE_VERSION: u16 = 2;
/// Version of the reward. Raised when its definition changes.
pub const REWARD_VERSION: u16 = 1;
/// Version of the observation space with a hedge lane: the features of version 1, then
/// [`FeatureId::HEDGE`].
pub const HEDGED_OBSERVATION_SPACE_VERSION: u16 = 2;
/// Version of the action space with a hedge lane: version 2's actions, and
/// [`CrossHedgeSpace`].
pub const HEDGED_ACTION_SPACE_VERSION: u16 = 3;
/// Version of the action space with the cancel-only action: version 2's actions, and
/// [`CancelActions::CancelOnly`].
pub const CANCEL_ACTION_SPACE_VERSION: u16 = 4;
/// Version of the action space with quotes that expire at the venue: [`QuoteExpiry::Lives`],
/// with or without the cancel-only action.
pub const EXPIRY_ACTION_SPACE_VERSION: u16 = 5;
/// The most quote lives an action space can offer.
pub const MAX_QUOTE_LIVES: usize = 8;
/// The longest quote life, in nanoseconds: 2^62, about 146 years, so that an activation time
/// plus a life does not overflow the kernel's clock.
pub const MAX_QUOTE_LIFE_NS: u64 = 1 << 62;
/// Version of the reward with a hedge lane: version 1's, with the hedge lane's ledger in the
/// net result.
pub const HEDGED_REWARD_VERSION: u16 = 2;
/// Number of hedge features of the observation space with a hedge lane.
pub const HEDGE_FEATURE_COUNT: usize = 5;
/// Version of the episode rules. Raised when an end of episode changes meaning.
pub const EPISODE_VERSION: u16 = 1;

/// Number of features of the observation space.
pub const FEATURE_COUNT: usize = 11;

const SPEC_MAGIC: [u8; 4] = *b"MMRL";

/// A feature of the observation, in the order of the vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FeatureId {
    /// `ask - bid` of the observed row.
    Spread,
    /// The observed mid minus the mid of the previous decision's row. No value at the
    /// first decision.
    MidChange,
    /// The observed row's time minus the previous decision's row's time, which shows
    /// a gap in the data. No value at the first decision.
    RowGap,
    /// Size at the observed bid.
    BidSize,
    /// Size at the observed ask.
    AskSize,
    /// `(bid size - ask size) / (bid size + ask size)`. No value when both are zero.
    Imbalance,
    /// Position from delivered fills.
    Position,
    /// The observed mid minus the average entry of that position. No value when flat.
    EntryGap,
    /// Orders open in the agent's account.
    OpenOrders,
    /// Open orders waiting for the outcome of a cancel.
    AwaitingCancel,
    /// `max_steps` minus the decisions taken.
    StepsLeft,
    /// `ask - bid` of the hedge row the agent sees, in hedge ticks.
    HedgeSpread,
    /// The hedge row's mid minus the observed quoted mid, both in quote per base of their
    /// own instrument.
    HedgeBasis,
    /// The decision's time minus the hedge row's time: how stale it is.
    HedgeRowAge,
    /// The hedge position from delivered reports, times the multiplier: base units of the
    /// quoted instrument.
    HedgePosition,
    /// 1 while a hedge is in flight, from its submission to its report, and 0 otherwise.
    HedgeInFlight,
}

impl FeatureId {
    /// Every feature, in the order of the vector.
    pub const ALL: [Self; FEATURE_COUNT] = [
        Self::Spread,
        Self::MidChange,
        Self::RowGap,
        Self::BidSize,
        Self::AskSize,
        Self::Imbalance,
        Self::Position,
        Self::EntryGap,
        Self::OpenOrders,
        Self::AwaitingCancel,
        Self::StepsLeft,
    ];

    /// The hedge features, after [`FeatureId::ALL`], in the order of the hedge vector.
    pub const HEDGE: [Self; HEDGE_FEATURE_COUNT] = [
        Self::HedgeSpread,
        Self::HedgeBasis,
        Self::HedgeRowAge,
        Self::HedgePosition,
        Self::HedgeInFlight,
    ];

    /// Position in the vector; the hedge features count on after [`FeatureId::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Spread => 0,
            Self::MidChange => 1,
            Self::RowGap => 2,
            Self::BidSize => 3,
            Self::AskSize => 4,
            Self::Imbalance => 5,
            Self::Position => 6,
            Self::EntryGap => 7,
            Self::OpenOrders => 8,
            Self::AwaitingCancel => 9,
            Self::StepsLeft => 10,
            Self::HedgeSpread => 11,
            Self::HedgeBasis => 12,
            Self::HedgeRowAge => 13,
            Self::HedgePosition => 14,
            Self::HedgeInFlight => 15,
        }
    }
}

/// Unit of a feature before normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureUnit {
    /// Price ticks.
    Ticks,
    /// Milliseconds.
    Milliseconds,
    /// Quantity lots.
    Lots,
    /// A dimensionless ratio.
    Ratio,
    /// Base asset.
    Base,
    /// Quote asset per unit of base asset.
    QuotePerBase,
    /// A count.
    Count,
}

impl FeatureUnit {
    const fn code(self) -> u16 {
        match self {
            Self::Ticks => 1,
            Self::Milliseconds => 2,
            Self::Lots => 3,
            Self::Ratio => 4,
            Self::Base => 5,
            Self::QuotePerBase => 6,
            Self::Count => 7,
        }
    }
}

/// The time a feature's inputs are known at. Each is known to a real agent at its decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AsOf {
    /// The exchange time of the last row delivered to the agent.
    ObservedRow,
    /// The reports delivered up to the decision. The venue may know more.
    DeliveredReports,
    /// The agent's own orders, at the decision.
    AgentAccount,
    /// The decisions taken, at the decision.
    DecisionCount,
}

impl AsOf {
    const fn code(self) -> u16 {
        match self {
            Self::ObservedRow => 1,
            Self::DeliveredReports => 2,
            Self::AgentAccount => 3,
            Self::DecisionCount => 4,
        }
    }
}

/// The span a feature describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureHorizon {
    /// One instant.
    Instant,
    /// From the previous decision to this one.
    SincePreviousDecision,
    /// The whole episode so far.
    EpisodeSoFar,
    /// The decisions still allowed by `max_steps`, which says nothing about the data.
    DecisionsAllowed,
}

impl FeatureHorizon {
    const fn code(self) -> u16 {
        match self {
            Self::Instant => 1,
            Self::SincePreviousDecision => 2,
            Self::EpisodeSoFar => 3,
            Self::DecisionsAllowed => 4,
        }
    }
}

/// What a feature remembers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StateMemory {
    /// Nothing: the inputs are at hand at the decision.
    None,
    /// The previous decision's observed row.
    PreviousDecision,
}

impl StateMemory {
    const fn code(self) -> u16 {
        match self {
            Self::None => 1,
            Self::PreviousDecision => 2,
        }
    }
}

/// The declaration of one feature.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FeatureSpec {
    /// The feature.
    pub id: FeatureId,
    /// Unit of its raw value.
    pub unit: FeatureUnit,
    /// When its inputs are known.
    pub as_of: AsOf,
    /// The span it describes.
    pub horizon: FeatureHorizon,
    /// What it remembers.
    pub memory: StateMemory,
    /// The normalized value is `raw / divisor`, clipped to `[-1, 1]`.
    pub divisor: Decimal,
}

/// The constants that normalize the features. Each is positive, and comes from the
/// configuration rather than from the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationScales {
    /// Divisor of [`FeatureId::Spread`], in ticks.
    pub spread_ticks: Decimal,
    /// Divisor of [`FeatureId::MidChange`], in ticks.
    pub mid_change_ticks: Decimal,
    /// Divisor of [`FeatureId::RowGap`], in milliseconds.
    pub row_gap_ms: Decimal,
    /// Divisor of [`FeatureId::BidSize`] and [`FeatureId::AskSize`], in lots.
    pub size_lots: Decimal,
    /// Divisor of [`FeatureId::Position`], in base units.
    pub position: Decimal,
    /// Divisor of [`FeatureId::EntryGap`], in quote per base.
    pub entry_gap: Decimal,
}

impl ObservationScales {
    pub(super) fn validate(&self) -> MMResult<()> {
        for (name, value) in [
            ("spread_ticks", self.spread_ticks),
            ("mid_change_ticks", self.mid_change_ticks),
            ("row_gap_ms", self.row_gap_ms),
            ("size_lots", self.size_lots),
            ("position", self.position),
            ("entry_gap", self.entry_gap),
        ] {
            if value <= Decimal::ZERO {
                return Err(MMError::InvalidConfiguration(format!(
                    "observation scale {name} must be positive, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// The constants that normalize the hedge features. Each is positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedgeScales {
    /// Divisor of [`FeatureId::HedgeSpread`], in hedge ticks.
    pub spread_ticks: Decimal,
    /// Divisor of [`FeatureId::HedgeBasis`], in quote per base.
    pub basis: Decimal,
    /// Divisor of [`FeatureId::HedgeRowAge`], in milliseconds.
    pub row_age_ms: Decimal,
    /// Divisor of [`FeatureId::HedgePosition`], in base units.
    pub position: Decimal,
}

impl HedgeScales {
    pub(super) fn validate(&self) -> MMResult<()> {
        for (name, value) in [
            ("spread_ticks", self.spread_ticks),
            ("basis", self.basis),
            ("row_age_ms", self.row_age_ms),
            ("position", self.position),
        ] {
            if value <= Decimal::ZERO {
                return Err(MMError::InvalidConfiguration(format!(
                    "hedge scale {name} must be positive, got {value}"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn features(&self) -> Vec<FeatureSpec> {
        let spec = |id, unit, as_of, horizon, divisor| FeatureSpec {
            id,
            unit,
            as_of,
            horizon,
            memory: StateMemory::None,
            divisor,
        };
        use AsOf as A;
        use FeatureHorizon as H;
        use FeatureId as F;
        use FeatureUnit as U;
        vec![
            spec(
                F::HedgeSpread,
                U::Ticks,
                A::ObservedRow,
                H::Instant,
                self.spread_ticks,
            ),
            spec(
                F::HedgeBasis,
                U::QuotePerBase,
                A::ObservedRow,
                H::Instant,
                self.basis,
            ),
            spec(
                F::HedgeRowAge,
                U::Milliseconds,
                A::ObservedRow,
                H::Instant,
                self.row_age_ms,
            ),
            spec(
                F::HedgePosition,
                U::Base,
                A::DeliveredReports,
                H::EpisodeSoFar,
                self.position,
            ),
            spec(
                F::HedgeInFlight,
                U::Count,
                A::AgentAccount,
                H::Instant,
                Decimal::ONE,
            ),
        ]
    }
}

/// The observation space.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObservationSpace {
    /// [`OBSERVATION_SPACE_VERSION`], or [`HEDGED_OBSERVATION_SPACE_VERSION`] with a hedge
    /// lane.
    pub version: u16,
    /// The features, in the order of the vector.
    pub features: [FeatureSpec; FEATURE_COUNT],
    /// The hedge features, in the order of [`FeatureId::HEDGE`]; empty with no hedge lane.
    pub hedge_features: Vec<FeatureSpec>,
}

impl ObservationSpace {
    pub(super) fn new(scales: &ObservationScales, orders: Decimal, max_steps: Decimal) -> Self {
        let spec = |id, unit, as_of, horizon, memory, divisor| FeatureSpec {
            id,
            unit,
            as_of,
            horizon,
            memory,
            divisor,
        };
        use AsOf as A;
        use FeatureHorizon as H;
        use FeatureId as F;
        use FeatureUnit as U;
        use StateMemory as M;
        Self {
            version: OBSERVATION_SPACE_VERSION,
            features: [
                spec(
                    F::Spread,
                    U::Ticks,
                    A::ObservedRow,
                    H::Instant,
                    M::None,
                    scales.spread_ticks,
                ),
                spec(
                    F::MidChange,
                    U::Ticks,
                    A::ObservedRow,
                    H::SincePreviousDecision,
                    M::PreviousDecision,
                    scales.mid_change_ticks,
                ),
                spec(
                    F::RowGap,
                    U::Milliseconds,
                    A::ObservedRow,
                    H::SincePreviousDecision,
                    M::PreviousDecision,
                    scales.row_gap_ms,
                ),
                spec(
                    F::BidSize,
                    U::Lots,
                    A::ObservedRow,
                    H::Instant,
                    M::None,
                    scales.size_lots,
                ),
                spec(
                    F::AskSize,
                    U::Lots,
                    A::ObservedRow,
                    H::Instant,
                    M::None,
                    scales.size_lots,
                ),
                spec(
                    F::Imbalance,
                    U::Ratio,
                    A::ObservedRow,
                    H::Instant,
                    M::None,
                    Decimal::ONE,
                ),
                spec(
                    F::Position,
                    U::Base,
                    A::DeliveredReports,
                    H::EpisodeSoFar,
                    M::None,
                    scales.position,
                ),
                spec(
                    F::EntryGap,
                    U::QuotePerBase,
                    A::DeliveredReports,
                    H::Instant,
                    M::None,
                    scales.entry_gap,
                ),
                spec(
                    F::OpenOrders,
                    U::Count,
                    A::AgentAccount,
                    H::Instant,
                    M::None,
                    orders,
                ),
                spec(
                    F::AwaitingCancel,
                    U::Count,
                    A::AgentAccount,
                    H::Instant,
                    M::None,
                    orders,
                ),
                spec(
                    F::StepsLeft,
                    U::Count,
                    A::DecisionCount,
                    H::DecisionsAllowed,
                    M::None,
                    max_steps,
                ),
            ],
            hedge_features: Vec::new(),
        }
    }
}

/// What happens to an action outside the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidActionPolicy {
    /// The caller must respect the mask: the step is an error and changes nothing.
    Mask,
    /// The step is taken as a hold, with the reason reported and the penalty charged.
    Reject,
}

impl InvalidActionPolicy {
    const fn code(self) -> u16 {
        match self {
            Self::Mask => 1,
            Self::Reject => 2,
        }
    }
}

/// Which hedge actions exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgeActions {
    /// There are none, as in the attribution, which reports hedge costs the same way.
    NotModelled {
        /// Why.
        reason: HedgeNotModelledReason,
    },
    /// A hedge in the quoted instrument: a marketable limit order at the observed touch
    /// on the side that reduces the delivered position, of a size from the order size
    /// grid, never larger than that position at the decision. It replaces the resting orders, as a
    /// quote does, and fills under the kernel's fill model at its own price, with the
    /// kernel's fee rate: when a later row's touch reaches it, after the latencies, and
    /// otherwise it rests. A hedge in another instrument is not modelled.
    MarketableAtTouch,
}

impl HedgeActions {
    const fn code(self) -> u16 {
        match self {
            Self::NotModelled {
                reason: HedgeNotModelledReason::SingleInstrument,
            } => 1,
            Self::MarketableAtTouch => 2,
        }
    }
}

/// Which cancel actions exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CancelActions {
    /// None: a policy withdraws its quotes only by replacing them.
    NotModelled,
    /// [`RlAction::Cancel`](crate::rl::RlAction::Cancel): cancel every resting or pending
    /// order and send nothing. Each cancel takes effect after the kernel's cancel latency,
    /// so a fill before it stands, and a cancel that reaches an order already filled is
    /// rejected. A quote sent at a decision and cancelled `d` decisions later, one per row,
    /// rests `d` rows plus the cancel latency less the submission latency.
    CancelOnly,
}

impl CancelActions {
    const fn code(self) -> u16 {
        match self {
            Self::NotModelled => 1,
            Self::CancelOnly => 2,
        }
    }
}

/// The lives a quote can be sent with, in nanoseconds of venue time: 1 to
/// [`MAX_QUOTE_LIVES`] values, strictly increasing, not zero and at most
/// [`MAX_QUOTE_LIFE_NS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteLives {
    values: [u64; MAX_QUOTE_LIVES],
    len: usize,
}

impl QuoteLives {
    /// Checks and keeps `lives`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for no lives, more than
    /// [`MAX_QUOTE_LIVES`], a zero, a life above [`MAX_QUOTE_LIFE_NS`], or values that do not strictly increase.
    pub fn new(lives: &[u64]) -> MMResult<Self> {
        let refuse = |why: &str| {
            Err(MMError::InvalidConfiguration(format!(
                "quote lives {why}, got {lives:?}"
            )))
        };
        if lives.is_empty() || lives.len() > MAX_QUOTE_LIVES {
            return refuse("must be 1 to 8 values");
        }
        if lives.contains(&0) {
            return refuse("must not be zero");
        }
        if lives.iter().any(|life| *life > MAX_QUOTE_LIFE_NS) {
            return refuse("must be at most 2^62 ns");
        }
        if lives.iter().zip(lives.iter().skip(1)).any(|(a, b)| a >= b) {
            return refuse("must strictly increase");
        }
        let mut values = [0; MAX_QUOTE_LIVES];
        for (slot, life) in values.iter_mut().zip(lives) {
            *slot = *life;
        }
        Ok(Self {
            values,
            len: lives.len(),
        })
    }

    /// The lives, in nanoseconds.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        self.values.get(..self.len).unwrap_or(&[])
    }
}

/// Whether a quote can expire by itself at the venue.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuoteExpiry {
    /// No: a quote rests until it fills, is replaced or is cancelled.
    NotModelled,
    /// [`RlAction::ExpiringQuote`](crate::rl::RlAction::ExpiringQuote), with these lives in
    /// nanoseconds. A leg rests its life from its own activation: it fills only on rows
    /// strictly inside `(activation, activation + life)`, and a replaced leg's expiry does
    /// not touch its replacement. [`RlAction::Quote`](crate::rl::RlAction::Quote) still
    /// rests with no expiry.
    Lives(Vec<u64>),
}

/// A hedge in a second instrument, the hedge lane of
/// [`HedgedKernel`](crate::backtest::kernel::HedgedKernel): an immediate-or-cancel order at
/// the hedge instrument's observed touch, on the side that reduces the net exposure of
/// both lanes' delivered positions, of a size from [`CrossHedgeSpace::sizes_lots`] that
/// never takes that exposure past zero. The quoted lane holds. One is in flight at a
/// time, from its submission to its report.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CrossHedgeSpace {
    /// The hedge instrument's tick, in quote per its unit.
    pub tick_size: Decimal,
    /// The hedge instrument's lot.
    pub lot_size: Decimal,
    /// Its largest order size.
    pub max_order_size: Decimal,
    /// Quoted base units per unit of the hedge's quantity.
    pub multiplier: Decimal,
    /// 1 for spot, 2 for a linear perpetual.
    pub instrument: u16,
    /// Oldest hedge row a hedge may trade on, in nanoseconds.
    pub max_staleness_ns: u64,
    /// Hedge sizes, in hedge lots.
    pub sizes_lots: Vec<u64>,
}

/// The action space.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ActionSpace {
    /// [`ACTION_SPACE_VERSION`], [`HEDGED_ACTION_SPACE_VERSION`] with a hedge lane, or
    /// [`CANCEL_ACTION_SPACE_VERSION`] with the cancel-only action, or
    /// [`EXPIRY_ACTION_SPACE_VERSION`] with quote lives.
    pub version: u16,
    /// Value of one tick, in quote per base.
    pub tick_size: Decimal,
    /// Value of one lot, in base units.
    pub lot_size: Decimal,
    /// Largest price, in quote per base.
    pub max_price: Decimal,
    /// Largest order size, in base units.
    pub max_order_size: Decimal,
    /// Bid offsets from the observed mid, in ticks.
    pub bid_offsets_ticks: Vec<u16>,
    /// Ask offsets from the observed mid, in ticks.
    pub ask_offsets_ticks: Vec<u16>,
    /// Order sizes, in lots.
    pub sizes_lots: Vec<u64>,
    /// What happens outside the mask.
    pub invalid_action: InvalidActionPolicy,
    /// Which hedge actions in the quoted instrument exist.
    pub hedge: HedgeActions,
    /// The hedge in a second instrument, with a hedge lane.
    pub cross_hedge: Option<CrossHedgeSpace>,
    /// Which cancel actions exist.
    pub cancel: CancelActions,
    /// Whether a quote can expire by itself.
    pub expiry: QuoteExpiry,
}

/// The one kind of reward of this version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RewardKind {
    /// The change of net equity, after fees, with penalties apart. Not a cash flow.
    NetEquityChange,
}

impl RewardKind {
    const fn code(self) -> u16 {
        match self {
            Self::NetEquityChange => 1,
        }
    }
}

/// The price that values an open position in the reward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RewardMark {
    /// The mid of the last row the kernel completed. The kernel's own unrealized value
    /// moves to the last fill price between rows, which the attribution does not.
    LastCompletedRow,
}

impl RewardMark {
    const fn code(self) -> u16 {
        match self {
            Self::LastCompletedRow => 1,
        }
    }
}

/// The reward's definition.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RewardSpec {
    /// [`REWARD_VERSION`], or [`HEDGED_REWARD_VERSION`] with a hedge lane, whose cash,
    /// fees, funding and position at its last mid join the net result.
    pub version: u16,
    /// What it measures.
    pub kind: RewardKind,
    /// The evaluation metric its ledger part sums to.
    pub metric: MetricId,
    /// Its unit, the metric's.
    pub unit: MetricUnit,
    /// What values an open position.
    pub mark: RewardMark,
    /// The charge for an action outside the mask, under
    /// [`InvalidActionPolicy::Reject`]. Not negative, and never in the ledger.
    pub invalid_action_penalty: Decimal,
    /// The `f32` reward is the exact reward divided by this.
    pub f32_unit: Decimal,
    /// Whether a policy may read the reward. False: see the [module documentation](super).
    pub policy_input: bool,
}

/// What values the inventory left when the data ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminalValuation {
    /// At the last row's mid, with no fee.
    MarkAtLastMid,
    /// Closed at the last row's touch, with the kernel's fee rate.
    CloseAtTouch,
}

impl TerminalValuation {
    const fn code(self) -> u16 {
        match self {
            Self::MarkAtLastMid => 1,
            Self::CloseAtTouch => 2,
        }
    }
}

/// The rules for the end of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct EpisodeSpec {
    /// [`EPISODE_VERSION`].
    pub version: u16,
    /// Decisions before truncation.
    pub max_steps: u64,
    /// What values the inventory on termination. Truncation closes nothing.
    pub terminal: TerminalValuation,
}

/// The sizes of the spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceShape {
    /// Length of the feature vector.
    pub features: usize,
    /// Bid offsets, and bits of the mask's `bid`.
    pub bid_offsets: usize,
    /// Ask offsets, and bits of the mask's `ask`.
    pub ask_offsets: usize,
    /// Sizes, and bits of the mask's `size`.
    pub sizes: usize,
    /// Hedge features; zero with no hedge lane.
    pub hedge_features: usize,
    /// Hedge sizes, and bits of the cross hedge mask; zero with no hedge lane.
    pub hedge_sizes: usize,
}

/// The identity of a specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecId {
    /// [`OBSERVATION_SPACE_VERSION`].
    pub observation: u16,
    /// [`ACTION_SPACE_VERSION`].
    pub action: u16,
    /// [`REWARD_VERSION`].
    pub reward: u16,
    /// [`EPISODE_VERSION`].
    pub episode: u16,
    /// FNV-1a-128 of [`RlSpec::to_bytes`]. It identifies the content, and is not a
    /// security boundary.
    pub fingerprint: u128,
}

/// The whole specification of a learning problem.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RlSpec {
    /// What a policy sees.
    pub observation: ObservationSpace,
    /// What a policy controls.
    pub action: ActionSpace,
    /// How it is rewarded.
    pub reward: RewardSpec,
    /// How an episode ends.
    pub episode: EpisodeSpec,
}

const fn metric_unit_code(unit: MetricUnit) -> u16 {
    match unit {
        MetricUnit::Quote => 1,
        MetricUnit::QuotePerBase => 2,
        MetricUnit::Base => 3,
        MetricUnit::Count => 4,
        MetricUnit::Ratio => 5,
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

fn put_len(out: &mut Vec<u8>, len: usize) {
    put_u64(out, u64::try_from(len).unwrap_or(u64::MAX));
}

impl RlSpec {
    /// The canonical bytes of the specification, the input of its fingerprint.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&SPEC_MAGIC);
        let o = &self.observation;
        put_u16(&mut out, o.version);
        put_len(&mut out, o.features.len());
        for feature in &o.features {
            put_len(&mut out, feature.id.index());
            put_u16(&mut out, feature.unit.code());
            put_u16(&mut out, feature.as_of.code());
            put_u16(&mut out, feature.horizon.code());
            put_u16(&mut out, feature.memory.code());
            put_decimal(&mut out, feature.divisor);
        }
        if !o.hedge_features.is_empty() {
            out.extend_from_slice(b"HF");
            put_len(&mut out, o.hedge_features.len());
            for feature in &o.hedge_features {
                put_len(&mut out, feature.id.index());
                put_u16(&mut out, feature.unit.code());
                put_u16(&mut out, feature.as_of.code());
                put_u16(&mut out, feature.horizon.code());
                put_u16(&mut out, feature.memory.code());
                put_decimal(&mut out, feature.divisor);
            }
        }
        let a = &self.action;
        put_u16(&mut out, a.version);
        put_decimal(&mut out, a.tick_size);
        put_decimal(&mut out, a.lot_size);
        put_decimal(&mut out, a.max_price);
        put_decimal(&mut out, a.max_order_size);
        put_len(&mut out, a.bid_offsets_ticks.len());
        for offset in &a.bid_offsets_ticks {
            put_u16(&mut out, *offset);
        }
        put_len(&mut out, a.ask_offsets_ticks.len());
        for offset in &a.ask_offsets_ticks {
            put_u16(&mut out, *offset);
        }
        put_len(&mut out, a.sizes_lots.len());
        for size in &a.sizes_lots {
            put_u64(&mut out, *size);
        }
        put_u16(&mut out, a.invalid_action.code());
        put_u16(&mut out, a.hedge.code());
        // A section that a specification without a hedge lane does not write, so its bytes
        // and identity stay as they were.
        if let Some(c) = &a.cross_hedge {
            out.extend_from_slice(b"XH");
            put_decimal(&mut out, c.tick_size);
            put_decimal(&mut out, c.lot_size);
            put_decimal(&mut out, c.max_order_size);
            put_decimal(&mut out, c.multiplier);
            put_u16(&mut out, c.instrument);
            put_u64(&mut out, c.max_staleness_ns);
            put_len(&mut out, c.sizes_lots.len());
            for size in &c.sizes_lots {
                put_u64(&mut out, *size);
            }
        }
        // As the hedge lane's: a specification without the cancel action does not write it.
        if a.cancel != CancelActions::NotModelled {
            out.extend_from_slice(b"CX");
            put_u16(&mut out, a.cancel.code());
        }
        // Likewise for quote lives.
        if let QuoteExpiry::Lives(lives) = &a.expiry {
            out.extend_from_slice(b"EX");
            put_len(&mut out, lives.len());
            for life in lives {
                put_u64(&mut out, *life);
            }
        }
        let r = &self.reward;
        put_u16(&mut out, r.version);
        put_u16(&mut out, r.kind.code());
        put_len(
            &mut out,
            MetricId::ALL
                .iter()
                .position(|m| *m == r.metric)
                .unwrap_or(usize::MAX),
        );
        put_u16(&mut out, metric_unit_code(r.unit));
        put_u16(&mut out, r.mark.code());
        put_decimal(&mut out, r.invalid_action_penalty);
        put_decimal(&mut out, r.f32_unit);
        put_u16(&mut out, u16::from(r.policy_input));
        let e = &self.episode;
        put_u16(&mut out, e.version);
        put_u64(&mut out, e.max_steps);
        put_u16(&mut out, e.terminal.code());
        out
    }

    /// The identity of this specification.
    #[must_use]
    pub fn id(&self) -> SpecId {
        SpecId {
            observation: self.observation.version,
            action: self.action.version,
            reward: self.reward.version,
            episode: self.episode.version,
            fingerprint: fnv1a128(&self.to_bytes()),
        }
    }

    /// The sizes of the spaces.
    #[must_use]
    pub fn shape(&self) -> SpaceShape {
        SpaceShape {
            features: self.observation.features.len(),
            bid_offsets: self.action.bid_offsets_ticks.len(),
            ask_offsets: self.action.ask_offsets_ticks.len(),
            sizes: self.action.sizes_lots.len(),
            hedge_features: self.observation.hedge_features.len(),
            hedge_sizes: self
                .action
                .cross_hedge
                .as_ref()
                .map_or(0, |c| c.sizes_lots.len()),
        }
    }

    /// Checks that this is the specification a consumer was built for.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] when `expected` differs in a version
    /// or in the fingerprint.
    pub fn ensure(&self, expected: SpecId) -> MMResult<()> {
        let id = self.id();
        if id == expected {
            return Ok(());
        }
        Err(MMError::InvalidConfiguration(format!(
            "unknown rl specification: expected {expected:?}, this is {id:?}"
        )))
    }
}
