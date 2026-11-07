//! A hedge in a second instrument (#256): see [`HedgedKernel`].

use crate::Decimal;
use crate::risk::RiskLimits;
use crate::types::action::{Action, Dir};
use crate::types::error::{MMError, MMResult};

use super::FinancialKernel;
use super::book::dir_code;
use super::config::{self, KernelConfig, KernelFee, KernelFillModel, Relax, ValidatedConfig};
use super::data::{DatasetId, KernelDataset, KernelTick, NS_PER_MS};
use super::encode::{CONFIG_MAGIC, Reader, Writer, fnv1a128};
use super::gate::{self, GateState, GateVerdict};
use super::grid::{Unit, pow10_i128, scaled_decimal};
use super::ledger::InvDec;
use super::rng::KernelRng;
use super::state::{
    EventSink, HaltReason, KernelAction, KernelState, NoSink, Phase, QuoteSide, StepOutcome,
};
use super::step::{self, Em};

type Halt = HaltReason;

/// Version of the hedge lane's configuration encoding.
pub const HEDGE_FORMAT_VERSION: u16 = 1;

/// A funding payment of a perpetual: at `at_ns`, the venue position pays
/// `position * mark * rate`, so a positive rate means a long pays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FundingRate {
    /// Time, in nanoseconds.
    pub at_ns: u64,
    /// Rate per payment.
    pub rate: Decimal,
}

/// What the hedge instrument is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgeInstrument {
    /// A spot instrument: no funding.
    Spot,
    /// A linear perpetual, with its declared funding schedule in time order.
    LinearPerpetual {
        /// The schedule; strictly increasing times.
        funding: Vec<FundingRate>,
    },
}

/// Limits on both lanes together, independent of the policy and its masks.
///
/// Every order of either lane must keep, with every live order of both
/// lanes filled on its side:
///
/// - the net exposure in the quoted instrument's base units, `quoted
///   quantity + multiplier * hedge quantity`, within `max_net_exposure`, in
///   the order's direction;
/// - the gross notional, the sum over both lanes of
///   `(|position| + live buys + live sells) * mark`, within
///   `max_gross_notional`, each lane at its last mid. A lane with no mark
///   yet holds no position and no live order, as orders follow an observed
///   row, so only the order's own lane can lack one: it is then valued at
///   the order's price.
///
/// A gross limit sees the fully hedged, oversized book that the net limit
/// cannot. Both lanes' prices are taken in one quote currency. An order
/// beyond a limit is refused, never reduced.
///
/// Only admission is gated, at a decision: fills of orders already resting
/// in the quoted lane move the net exposure between decisions without a
/// check, as fills always do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CombinedLimits {
    /// Largest net exposure, in the quoted instrument's base units.
    pub max_net_exposure: Option<Decimal>,
    /// Largest gross notional, in the quote currency.
    pub max_gross_notional: Option<Decimal>,
}

/// The hedge lane.
#[derive(Debug, Clone, PartialEq)]
pub struct HedgeLaneConfig {
    /// The hedge instrument's grid and fees: `tick_size`, `lot_size`,
    /// `fee_rate` (quote fees), `max_price`, `max_order_size` and its own
    /// `limits`. `max_steps` and `latency` must equal the quoted
    /// configuration's; `initial_capital` must be zero (the capital is the
    /// quoted lane's), and the fill model immediate.
    pub instrument: KernelConfig,
    /// Spot or linear perpetual.
    pub kind: HedgeInstrument,
    /// Quoted instrument base units per unit of the hedge's quantity;
    /// positive.
    pub multiplier: Decimal,
    /// Oldest hedge row a hedge may trade on, in nanoseconds.
    pub max_staleness_ns: u64,
    /// Limits on both lanes.
    pub combined: CombinedLimits,
}

impl HedgeLaneConfig {
    /// A lane with no combined limits.
    #[must_use]
    pub fn new(
        instrument: KernelConfig,
        kind: HedgeInstrument,
        multiplier: Decimal,
        max_staleness_ns: u64,
    ) -> Self {
        Self {
            instrument,
            kind,
            multiplier,
            max_staleness_ns,
            combined: CombinedLimits::default(),
        }
    }

    /// Sets the combined limits.
    #[must_use]
    pub fn with_combined(mut self, combined: CombinedLimits) -> Self {
        self.combined = combined;
        self
    }

    /// Canonical encoding (`MMKC` + `HL`), the input of the fingerprint.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&CONFIG_MAGIC);
        out.extend_from_slice(b"HL");
        out.extend_from_slice(&HEDGE_FORMAT_VERSION.to_le_bytes());
        let instrument = self.instrument.encode();
        out.extend_from_slice(&(instrument.len() as u64).to_le_bytes());
        out.extend_from_slice(&instrument);
        let decimal = |out: &mut Vec<u8>, d: Decimal| out.extend_from_slice(&d.serialize());
        match &self.kind {
            HedgeInstrument::Spot => out.push(1),
            HedgeInstrument::LinearPerpetual { funding } => {
                out.push(2);
                out.extend_from_slice(&(funding.len() as u64).to_le_bytes());
                for f in funding {
                    out.extend_from_slice(&f.at_ns.to_le_bytes());
                    decimal(&mut out, f.rate);
                }
            }
        }
        decimal(&mut out, self.multiplier);
        out.extend_from_slice(&self.max_staleness_ns.to_le_bytes());
        for limit in [
            self.combined.max_net_exposure,
            self.combined.max_gross_notional,
        ] {
            match limit {
                None => out.push(0),
                Some(value) => {
                    out.push(1);
                    decimal(&mut out, value);
                }
            }
        }
        out
    }

    /// FNV-1a-128 of [`HedgeLaneConfig::encode`].
    #[must_use]
    pub fn fingerprint(&self) -> u128 {
        fnv1a128(&self.encode())
    }
}

/// A funding payment in integers: the rate's mantissa and scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Funding {
    at_ns: u64,
    rate_m: i128,
    rate_s: u32,
}

/// The validated lane.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ValidatedLane {
    inst: ValidatedConfig,
    funding: Vec<Funding>,
    multiplier: Decimal,
    staleness_ns: u64,
    combined: CombinedLimits,
    fingerprint: u128,
    /// Bound on the lane's cash, fees and funding, in its money units.
    bound: u128,
}

fn invalid(message: String) -> MMError {
    MMError::InvalidConfiguration(message)
}

fn validate_lane(lane: &HedgeLaneConfig, quoted: &KernelConfig) -> MMResult<ValidatedLane> {
    let i = &lane.instrument;
    if !i.initial_capital.is_zero() {
        return Err(invalid(
            "the hedge lane holds no capital: initial_capital must be zero".to_string(),
        ));
    }
    if i.margin.is_some() {
        return Err(invalid("the hedge lane takes no balance gate".to_string()));
    }
    if i.fee_currency != KernelFee::Quote || i.fill_model != KernelFillModel::Immediate {
        return Err(invalid(
            "the hedge lane takes quote fees and fills immediately".to_string(),
        ));
    }
    if i.max_steps != quoted.max_steps || i.latency != quoted.latency {
        return Err(invalid(
            "the hedge lane's max_steps and latency must equal the quoted configuration's"
                .to_string(),
        ));
    }
    if lane.multiplier <= Decimal::ZERO {
        return Err(invalid(format!(
            "multiplier must be positive, got {}",
            lane.multiplier
        )));
    }
    for limit in [
        lane.combined.max_net_exposure,
        lane.combined.max_gross_notional,
    ]
    .into_iter()
    .flatten()
    {
        if limit < Decimal::ZERO {
            return Err(invalid(format!("a combined limit is negative: {limit}")));
        }
    }
    let inst = config::validate(i, Relax::default())?;
    let mut funding = Vec::new();
    if let HedgeInstrument::LinearPerpetual { funding: schedule } = &lane.kind {
        let mut last = None;
        for f in schedule {
            if last.is_some_and(|t| f.at_ns <= t) {
                return Err(invalid(
                    "funding times must be strictly increasing".to_string(),
                ));
            }
            last = Some(f.at_ns);
            let rate = f.rate.normalize();
            // The shadow multiplies lots, half a mid and the rate in a
            // `Decimal`, exactly only within 28 places.
            let places = i
                .lot_size
                .scale()
                .saturating_add(i.tick_size.scale())
                .saturating_add(1)
                .saturating_add(rate.scale());
            if places > 28 {
                return Err(invalid(format!(
                    "funding rate {} has too many places for the lane's grid",
                    f.rate
                )));
            }
            funding.push(Funding {
                at_ns: f.at_ns,
                rate_m: rate.mantissa(),
                rate_s: rate.scale(),
            });
        }
    }
    let bound = lane_bound(&inst, &funding)
        .ok_or_else(|| invalid("the hedge lane's bound overflows".to_string()))?;
    if bound.checked_mul(2).is_none_or(|b| b >= 1u128 << 96) {
        return Err(invalid(format!(
            "the hedge lane's bound {bound} is not below 2^95"
        )));
    }
    Ok(ValidatedLane {
        inst,
        funding,
        multiplier: lane.multiplier,
        staleness_ns: lane.max_staleness_ns,
        combined: lane.combined,
        fingerprint: lane.fingerprint(),
        bound,
    })
}

/// `B` of the instrument, plus every funding payment at the largest
/// position (`2SQ` lots) and the largest mid (`P` ticks), rounded up.
fn lane_bound(inst: &ValidatedConfig, funding: &[Funding]) -> Option<u128> {
    let position = u128::from(inst.s_max)
        .checked_mul(2)?
        .checked_mul(u128::from(inst.q))?;
    let base = position
        .checked_mul(u128::from(inst.p))?
        .checked_mul(u128::from(inst.tick.mantissa))?
        .checked_mul(u128::from(inst.lot.mantissa))?;
    let mut total = inst.bound_b;
    for f in funding {
        let rate = f.rate_m.unsigned_abs();
        // The integers the payment is computed with (`fund`) must fit too:
        // twice the mid, and the rate's scale against the fee scale.
        let (up, down) = funding_scales(inst.fs, f.rate_s);
        let numerator = base
            .checked_mul(2)?
            .checked_mul(rate)?
            .checked_mul(super::grid::pow10_u128(up)?)?;
        if numerator > i128::MAX.unsigned_abs() {
            return None;
        }
        let payment = numerator
            .checked_div(super::grid::pow10_u128(down)?.checked_mul(2)?)?
            .checked_add(1)?;
        total = total.checked_add(payment)?;
    }
    Some(total)
}

/// The powers of ten that put a payment in money units: `10^up` over
/// `2 * 10^down`, from the fee scale and the rate's scale.
fn funding_scales(fs: u32, rate_s: u32) -> (u32, u32) {
    match fs.checked_sub(rate_s) {
        Some(up) => (up, 0),
        None => (0, rate_s.saturating_sub(fs)),
    }
}

/// A money event of the hedge lane, as [`EventSink::hedge`] receives it
/// (#266), in the lane's order: at one time, funding, then the row, then an
/// order's execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgeEvent {
    /// A hedge row ran: the lane's mark is now its mid.
    Row {
        /// Row index in the hedge data.
        row: u64,
        /// Row time, in nanoseconds.
        ts_ns: u64,
        /// `bid + ask` in ticks, the mid in half ticks; `None` when not
        /// positive, which leaves the lane without a mark.
        mid2: Option<u64>,
    },
    /// A fill booked in the lane's venue ledger.
    Fill {
        /// Execution time, in nanoseconds.
        ts_ns: u64,
        /// Our side: [`Action::Buy`] or [`Action::Sell`].
        side: Action,
        /// Fill price, in ticks.
        price: u64,
        /// Size, in lots.
        lots: u64,
        /// Fee, in money units of [`HedgedKernel::lane_money_scale`];
        /// negative for a rebate.
        fee_u: i128,
    },
    /// A funding payment booked at the lane's mark.
    Funding {
        /// Payment time, in nanoseconds.
        at_ns: u64,
        /// Received, in money units of [`HedgedKernel::lane_money_scale`];
        /// negative when paid.
        received_u: i128,
        /// The mark it was valued at, in half ticks.
        mid2: u64,
    },
}

/// How a hedge order ended at the venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgeFill {
    /// Filled whole.
    Filled,
    /// Filled up to the touch's lots; the rest cancelled.
    Partial,
    /// The touch had moved through the limit: cancelled.
    MovedThrough,
    /// The lane had gone stale in flight: cancelled.
    Stale,
    /// The touch at or inside the limit showed no lots: cancelled.
    NoLots,
}

/// A hedge order in flight to the venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HedgeOrder {
    id: u64,
    dir: Dir,
    price: u64,
    lots: u64,
    exec_ns: u64,
}

/// A hedge order's report in flight to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HedgeReport {
    id: u64,
    dir: Dir,
    filled: u64,
    price: u64,
    outcome: HedgeFill,
    due_ns: u64,
}

/// Counts of the hedge lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct HedgeCounters {
    /// Hedge orders submitted.
    pub submitted: u64,
    /// Filled whole.
    pub filled: u64,
    /// Filled in part, at the touch's lots.
    pub partial: u64,
    /// Lots cancelled beyond the touch.
    pub beyond_touch_lots: u64,
    /// Cancelled: the touch moved through the limit.
    pub moved_through: u64,
    /// Cancelled: the lane went stale in flight.
    pub stale_in_flight: u64,
    /// Cancelled: the touch showed no lots.
    pub no_lots: u64,
    /// Hedges refused at the decision, by reason ([`HedgeRefusal::index`]).
    pub refused: [u64; 7],
    /// Quoted-lane orders refused by the combined limits.
    pub combined_quote_rejects: u64,
    /// Funding payments booked.
    pub funding_paid: u64,
    /// Of those, at a mark older than the staleness limit.
    pub funding_stale: u64,
    /// Funding times with no hedge row yet: nothing charged.
    pub funding_without_mark: u64,
    /// Orders or reports still pending when the run ended, dropped.
    pub dropped_at_end: u64,
    /// Reports delivered to the agent.
    pub delivered: u64,
}

/// Why a hedge was refused at the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgeRefusal {
    /// The run is not at a decision.
    Terminal,
    /// A hedge is in flight: one at a time.
    InFlight,
    /// The agent sees no hedge row yet.
    NoTouch,
    /// The hedge row the agent sees is older than the staleness limit.
    Stale,
    /// The size is outside `1..=max_order_size`.
    SizeRange,
    /// The hedge instrument's own limits refused it.
    Limits,
    /// The combined limits refused it.
    Combined,
}

impl HedgeRefusal {
    /// Index in [`HedgeCounters::refused`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Terminal => 0,
            Self::InFlight => 1,
            Self::NoTouch => 2,
            Self::Stale => 3,
            Self::SizeRange => 4,
            Self::Limits => 5,
            Self::Combined => 6,
        }
    }
}

/// The hedge lane's state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedgeLane {
    data_id: DatasetId,
    cfg_fp: u128,
    /// Next hedge row to run.
    row: u64,
    /// Last hedge row run.
    last_row: Option<u64>,
    /// Next funding payment.
    funding: u64,
    rng: KernelRng,
    next_id: u64,
    order: Option<HedgeOrder>,
    report: Option<HedgeReport>,
    /// Venue position and live lots; `mark2` is the last row's mid.
    gate: GateState,
    cash_u: i128,
    fees_u: i128,
    funding_u: i128,
    venue: InvDec,
    cash_d: Decimal,
    fees_d: Decimal,
    funding_d: Decimal,
    agent_lots: i64,
    agent: InvDec,
    /// The run ended and the lane ran to its end.
    closed: bool,
    counters: HedgeCounters,
}

impl HedgeLane {
    /// Counts of the lane.
    #[must_use]
    pub fn counters(&self) -> HedgeCounters {
        self.counters
    }

    /// Venue position, in lots of the hedge instrument.
    #[must_use]
    pub fn venue_position_lots(&self) -> i64 {
        self.gate.venue_pos_lots
    }

    /// Position the agent has been told of, in lots.
    #[must_use]
    pub fn agent_position_lots(&self) -> i64 {
        self.agent_lots
    }

    /// Whether a hedge is in flight, from submission to its report.
    #[must_use]
    pub fn in_flight(&self) -> bool {
        self.order.is_some() || self.report.is_some()
    }

    /// Next hedge row to run.
    #[must_use]
    pub fn cursor(&self) -> u64 {
        self.row
    }
}

/// The two lanes' states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedgedState<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize> {
    /// The quoted lane.
    pub kernel: KernelState<ORD, MSG, DLV, OBS>,
    /// The hedge lane.
    pub lane: HedgeLane,
}

/// One decision: a kernel action, or a hedge with the quotes left as they
/// are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HedgedAction {
    /// An action of the quoted lane.
    Kernel(KernelAction),
    /// An immediate-or-cancel hedge of `lots` hedge lots: `Bid` buys at the
    /// observed ask, `Ask` sells at the observed bid. The quoted lane holds.
    Hedge {
        /// Direction.
        side: QuoteSide,
        /// Size, in hedge lots.
        lots: u64,
    },
}

/// The result of a step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HedgedOutcome {
    /// The quoted lane's outcome; a hedge is a `Hold` there.
    pub kernel: StepOutcome,
    /// For a hedge, whether it was submitted.
    pub hedge: Option<Result<(), HedgeRefusal>>,
    /// Quoted-lane orders the combined limits refused in this step.
    pub combined_rejects: u64,
}

/// The hedge row the agent sees, with its own position.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HedgeView {
    /// The row, if any is visible.
    pub row: Option<u64>,
    /// The row's values.
    pub tick: Option<KernelTick>,
    /// Whether the row is older than the staleness limit at the decision.
    pub stale: bool,
    /// Delivered position, in lots.
    pub position_lots: i64,
    /// Delivered position.
    pub position: Decimal,
    /// Average entry of the delivered position.
    pub avg_entry: Decimal,
    /// Whether a hedge is in flight.
    pub in_flight: bool,
}

/// The hedge lane's venue totals, and both lanes' net.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HedgeSummary {
    /// Venue position.
    pub position: Decimal,
    /// Cash from fills.
    pub cash: Decimal,
    /// Fees paid.
    pub fees: Decimal,
    /// Funding received (negative when paid).
    pub funding: Decimal,
    /// Position at the last mid; zero with no mid.
    pub mark_value: Decimal,
    /// `cash - fees + funding + mark_value`.
    pub net: Decimal,
    /// The quoted lane's `net_pnl` plus this lane's `net`.
    pub combined_net: Decimal,
}

/// The combined limits with the hedge lane's side of them, for the quoted
/// lane's gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CombinedCheck {
    limits: CombinedLimits,
    multiplier: Decimal,
    hedge: GateState,
    hedge_tick: Unit,
    hedge_lot: Unit,
}

/// One lane's exposure in `Decimal`: position, live buys and sells, all in
/// base units of the quoted instrument (`scale` per lot), and its gross
/// notional per lot at `price`.
struct Exposure {
    position: Decimal,
    buys: Decimal,
    sells: Decimal,
    /// `(|position| + buys + sells)` lots times lot times price.
    gross: Decimal,
}

fn exposure(
    gate: &GateState,
    lot: &Unit,
    tick: &Unit,
    per_lot: Decimal,
    price: Option<u64>,
) -> Option<Exposure> {
    let lots = |n: i128| lot.to_decimal(n)?.checked_mul(per_lot);
    let position = lots(i128::from(gate.venue_pos_lots))?;
    let buys = lots(i128::from(gate.live_buy_lots))?;
    let sells = lots(i128::from(gate.live_sell_lots))?;
    let gross = match price {
        None => Decimal::ZERO,
        Some(half_ticks) => {
            let held = u128::from(gate.venue_pos_lots.unsigned_abs())
                .checked_add(u128::from(gate.live_buy_lots))?
                .checked_add(u128::from(gate.live_sell_lots))?;
            let quantity = lot.to_decimal(i128::try_from(held).ok()?)?;
            let mid = tick
                .to_decimal(i128::from(half_ticks))?
                .checked_div(Decimal::TWO)?;
            quantity.checked_mul(mid)?
        }
    };
    Some(Exposure {
        position,
        buys,
        sells,
        gross,
    })
}

impl CombinedCheck {
    /// Whether the combined limits admit `lots` at `price` on the lane
    /// whose gate is `with`, the other lane's being `other`. `None` if a sum
    /// is not representable.
    #[allow(clippy::too_many_arguments)]
    fn admits(
        &self,
        quoted: &GateState,
        quoted_tick: &Unit,
        quoted_lot: &Unit,
        hedge_order: bool,
        dir: Dir,
        price: u64,
        lots: u64,
    ) -> Option<bool> {
        let mut quoted = *quoted;
        let mut hedge = self.hedge;
        let target = if hedge_order { &mut hedge } else { &mut quoted };
        target.reserve(dir, lots)?;
        // A lane with no mark is valued at the order's price, twice in half ticks.
        let order_mark = price.checked_mul(2)?;
        let quoted_mark = quoted.mark2.or((!hedge_order).then_some(order_mark));
        let hedge_mark = hedge.mark2.or(hedge_order.then_some(order_mark));
        let q = exposure(&quoted, quoted_lot, quoted_tick, Decimal::ONE, quoted_mark)?;
        let h = exposure(
            &hedge,
            &self.hedge_lot,
            &self.hedge_tick,
            self.multiplier,
            hedge_mark,
        )?;
        if let Some(max) = self.limits.max_net_exposure {
            let ok = match dir {
                Dir::Buy => q
                    .position
                    .checked_add(q.buys)?
                    .checked_add(h.position)?
                    .checked_add(h.buys)?
                    .le(&max),
                Dir::Sell => q
                    .position
                    .checked_sub(q.sells)?
                    .checked_add(h.position)?
                    .checked_sub(h.sells)?
                    .ge(&max.checked_neg()?),
            };
            if !ok {
                return Some(false);
            }
        }
        if let Some(max) = self.limits.max_gross_notional
            && q.gross.checked_add(h.gross)? > max
        {
            return Some(false);
        }
        Some(true)
    }

    /// For a quoted-lane order, called by the kernel's gate.
    pub(super) fn admits_quoted(
        &self,
        quoted: &GateState,
        tick: &Unit,
        lot: &Unit,
        dir: Dir,
        price: u64,
        lots: u64,
    ) -> Option<bool> {
        self.admits(quoted, tick, lot, false, dir, price, lots)
    }
}

trait CheckedNeg: Sized {
    fn checked_neg(self) -> Option<Self>;
}

impl CheckedNeg for Decimal {
    fn checked_neg(self) -> Option<Self> {
        Decimal::ZERO.checked_sub(self)
    }
}

/// A kernel on the quoted instrument with a hedge lane on a second one.
///
/// [`HedgedKernel`] runs a [`FinancialKernel`] on the quoted instrument and a
/// hedge lane on a second one: its own grid, fees and data, a spot or a
/// linear perpetual on the same underlying, with a declared funding schedule
/// for the perpetual. The agent hedges with an immediate-or-cancel order at
/// the hedge instrument's observed touch ([`HedgedAction::Hedge`]); the lane
/// never rests an order and never quotes.
///
/// # One clock
///
/// The two lanes meet only at decision points: an order of either lane is
/// admitted there, under the instrument's own limits and the combined limits
/// ([`CombinedLimits`]), which read both lanes' positions and live orders;
/// and the ledgers are summed there. Between two decisions, the quoted
/// lane's events depend on nothing in the hedge lane and the hedge lane's on
/// nothing in the quoted one. So the kernel advances the quoted lane to its
/// next decision at time `τ`, and the hedge lane's events before `τ` then run
/// in time order, which is the same as one merged clock. At equal times the
/// quoted lane comes first, its decision included: a hedge row, fill, report
/// or funding at exactly `τ` is not seen by the decision at `τ`. When the run
/// ends, the hedge lane runs to the end time, inclusive, and what is still
/// pending then is dropped and counted.
///
/// Within the hedge lane, at equal times: funding, then the row, then the
/// order's execution, then the report's delivery.
///
/// # The hedge order
///
/// - It is admitted at a decision if the run is deciding, no hedge is in
///   flight (one at a time, from submission to the delivery of its report),
///   the agent sees a hedge row no older than
///   [`HedgeLaneConfig::max_staleness_ns`], its size is in range, and the
///   instrument's limits and the combined limits accept it. Its limit price
///   is the touch the agent sees: the ask to buy, the bid to sell.
/// - Everything above is decided at the decision, on both lanes' venue
///   positions and live orders then. At execution only the hedge lane is
///   read: its touch and its staleness. Nothing of the quoted lane is
///   rechecked then, as the quoted lane has moved on to its next decision.
/// - It reaches the venue after the submission delay of the quoted
///   configuration's [`KernelLatency`](super::KernelLatency), with the
///   lane's own jitter stream. There it executes against the touch standing
///   at that time: at the standing price when that price is at or better
///   than its limit, for at most the touch's lots. A touch that moved through
///   the limit cancels it, and so does a lane gone stale in flight: it never
///   fills at an old price. The lots beyond the touch are cancelled and
///   counted.
/// - Its report reaches the agent after the acknowledgement delay. The agent
///   sees a hedge row after the observation delay, without jitter.
///
/// # Funding
///
/// At each time of the schedule, the venue position pays
/// `position * mark * rate`: a positive rate means a long pays. The mark is
/// the mid of the last hedge row before that time. With no row yet nothing
/// is charged, and a mark older than `max_staleness_ns` is still used and
/// counted as stale. The payment is rounded half to even to the lane's money
/// unit.
///
/// # Exactness
///
/// The lane keeps cash, fees and funding in integers of its money unit
/// `10^-s`, with `s` the instrument's money scale (module docs of the
/// kernel), and checks every fill and payment against a `Decimal` shadow; a
/// mismatch halts the run. Its bound adds the largest funding to the
/// instrument's bound `B`. No engine runs two instruments, so the lane is
/// checked by hand-computed fixtures and the shadow, not against an engine;
/// with no lane, [`FinancialKernel`] is unchanged.
#[derive(Debug, Clone)]
pub struct HedgedKernel<
    'd,
    const ORD: usize = 16,
    const MSG: usize = 64,
    const DLV: usize = 128,
    const OBS: usize = 64,
> {
    kernel: FinancialKernel<'d, ORD, MSG, DLV, OBS>,
    lane: ValidatedLane,
    hedge: &'d KernelDataset,
}

/// Rounds `numerator / denominator` half to even; `denominator > 0`.
fn div_half_even(numerator: i128, denominator: i128) -> Option<i128> {
    let quotient = numerator.checked_div(denominator)?;
    let remainder = numerator.checked_rem(denominator)?;
    let twice = remainder.checked_abs()?.checked_mul(2)?;
    let away = match twice.cmp(&denominator) {
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => quotient.checked_rem(2)? != 0,
    };
    if !away {
        return Some(quotient);
    }
    if numerator < 0 {
        quotient.checked_sub(1)
    } else {
        quotient.checked_add(1)
    }
}

fn signed(dir: Dir, lots: u64) -> Option<i64> {
    let lots = i64::try_from(lots).ok()?;
    match dir {
        Dir::Buy => Some(lots),
        Dir::Sell => lots.checked_neg(),
    }
}

/// Classes of the hedge lane's events at one time, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LaneEvent {
    Funding,
    Row,
    Execute,
    Deliver,
}

impl<'d, const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    HedgedKernel<'d, ORD, MSG, DLV, OBS>
{
    /// Validates both lanes and builds the kernel.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the quoted lane is
    /// refused by [`FinancialKernel::new`], if the lane's instrument breaks
    /// a rule of [`HedgeLaneConfig::instrument`] or a bound, if the
    /// multiplier is not positive, a combined limit negative, or funding
    /// times not strictly increasing, if `hedge` was built on another
    /// grid, or if the quoted lane has a balance gate
    /// ([`KernelConfig::margin`]), which reads one instrument's balances
    /// only.
    pub fn new(
        cfg: &KernelConfig,
        data: &'d KernelDataset,
        lane: &HedgeLaneConfig,
        hedge: &'d KernelDataset,
    ) -> MMResult<Self> {
        if cfg.margin.is_some() {
            return Err(invalid(
                "a hedged kernel does not take a balance gate on its quoted lane".to_string(),
            ));
        }
        let kernel = FinancialKernel::new(cfg, data)?;
        let lane = validate_lane(lane, cfg)?;
        let grid = hedge.grid;
        if grid.tick_m != lane.inst.tick.mantissa
            || grid.tick_s != lane.inst.tick.scale
            || grid.lot_m != lane.inst.lot.mantissa
            || grid.lot_s != lane.inst.lot.scale
            || grid.max_price != lane.inst.p
        {
            return Err(invalid(
                "the hedge dataset was built on another tick, lot or max_price grid".to_string(),
            ));
        }
        let tail = lane
            .inst
            .latency
            .max_total_ns()
            .and_then(|t| t.checked_add(lane.staleness_ns))
            .ok_or_else(|| invalid("hedge latency overflows u64".to_string()))?;
        let last = lane
            .funding
            .last()
            .map_or(hedge.id.last_ts, |f| f.at_ns.max(hedge.id.last_ts))
            .max(data.id.last_ts);
        if last.checked_add(tail).is_none() {
            return Err(invalid(
                "the last hedge time plus latency overflows u64".to_string(),
            ));
        }
        Ok(Self {
            kernel,
            lane,
            hedge,
        })
    }

    /// The quoted lane's kernel.
    #[must_use]
    pub fn kernel(&self) -> &FinancialKernel<'d, ORD, MSG, DLV, OBS> {
        &self.kernel
    }

    /// The hedge dataset.
    #[must_use]
    pub fn hedge_data(&self) -> &'d KernelDataset {
        self.hedge
    }

    /// Fingerprint of the hedge lane's configuration.
    #[must_use]
    pub fn lane_fingerprint(&self) -> u128 {
        self.lane.fingerprint
    }

    /// The lane's money scale.
    #[must_use]
    pub fn lane_money_scale(&self) -> u32 {
        self.lane.inst.s
    }

    /// Whether `s` belongs to this kernel's datasets and configurations.
    #[must_use]
    pub fn owns(&self, s: &HedgedState<ORD, MSG, DLV, OBS>) -> bool {
        self.owns_parts(&s.kernel, &s.lane)
    }

    /// [`HedgedKernel::owns`] on the two lanes' states held apart.
    pub(crate) fn owns_parts(&self, k: &KernelState<ORD, MSG, DLV, OBS>, l: &HedgeLane) -> bool {
        self.kernel.owns(k) && l.data_id == self.hedge.id && l.cfg_fp == self.lane.fingerprint
    }

    /// The state at the first decision point (or the end of the run).
    #[must_use]
    pub fn reset(&self, seed: u64) -> HedgedState<ORD, MSG, DLV, OBS> {
        self.reset_with(seed, &mut NoSink)
    }

    /// As [`HedgedKernel::reset`], sending the quoted lane's events to `sink`, and the
    /// hedge lane's money events to [`EventSink::hedge`].
    pub fn reset_with<E: EventSink>(
        &self,
        seed: u64,
        sink: &mut E,
    ) -> HedgedState<ORD, MSG, DLV, OBS> {
        let kernel = self.kernel.reset_with(seed, sink);
        let mut s = HedgedState {
            kernel,
            lane: HedgeLane {
                data_id: self.hedge.id,
                cfg_fp: self.lane.fingerprint,
                row: 0,
                last_row: None,
                funding: 0,
                // Its own stream: the quoted lane's draws are unchanged.
                rng: KernelRng::new(seed ^ 0x6865_6467_655f_6c6e),
                next_id: 1,
                order: None,
                report: None,
                gate: GateState::NEW,
                cash_u: 0,
                fees_u: 0,
                funding_u: 0,
                venue: InvDec::FLAT,
                cash_d: Decimal::ZERO,
                fees_d: Decimal::ZERO,
                funding_d: Decimal::ZERO,
                agent_lots: 0,
                agent: InvDec::FLAT,
                closed: false,
                counters: HedgeCounters::default(),
            },
        };
        self.catch_up(&mut s.kernel, &mut s.lane, &mut Em::new(sink));
        s
    }

    /// Applies `action` and advances to the next decision point.
    pub fn step(
        &self,
        s: &mut HedgedState<ORD, MSG, DLV, OBS>,
        action: HedgedAction,
    ) -> HedgedOutcome {
        self.step_with(s, action, &mut NoSink)
    }

    /// As [`HedgedKernel::step`], sending the quoted lane's events to `sink`, and the
    /// hedge lane's money events to [`EventSink::hedge`].
    pub fn step_with<E: EventSink>(
        &self,
        s: &mut HedgedState<ORD, MSG, DLV, OBS>,
        action: HedgedAction,
        sink: &mut E,
    ) -> HedgedOutcome {
        self.step_parts(&mut s.kernel, &mut s.lane, action, sink)
    }

    /// [`HedgedKernel::step_with`] on the two lanes' states held apart, so a caller that
    /// holds them in its own state copies neither.
    pub(crate) fn step_parts<E: EventSink>(
        &self,
        k: &mut KernelState<ORD, MSG, DLV, OBS>,
        l: &mut HedgeLane,
        action: HedgedAction,
        sink: &mut E,
    ) -> HedgedOutcome {
        if !self.owns_parts(k, l) {
            let mut em = Em::new(sink);
            if k.status().is_running() {
                step::halt(k, HaltReason::StateMismatch, &mut em);
            }
            return HedgedOutcome {
                kernel: StepOutcome {
                    applied: super::ActionStatus::Rejected(super::RejectReason::Terminal),
                    status: k.status(),
                    events: em.count,
                },
                hedge: None,
                combined_rejects: 0,
            };
        }
        let (action, hedge) = match action {
            HedgedAction::Kernel(action) => (action, None),
            HedgedAction::Hedge { side, lots } => {
                let admitted = self.admit_hedge(k, l, side, lots);
                if let Err(reason) = admitted {
                    let slot = l.counters.refused.get_mut(reason.index());
                    if let Some(slot) = slot {
                        *slot = slot.saturating_add(1);
                    }
                }
                (KernelAction::Hold, Some(admitted))
            }
        };
        let check = self.combined_check(l);
        let mut rejects = 0u64;
        let kernel =
            self.kernel
                .step_combined(k, action, check.as_ref().map(|c| (c, &mut rejects)), sink);
        l.counters.combined_quote_rejects =
            l.counters.combined_quote_rejects.saturating_add(rejects);
        let mut em = Em::new(sink);
        self.catch_up(k, l, &mut em);
        HedgedOutcome {
            kernel: StepOutcome {
                events: kernel.events.saturating_add(em.count),
                status: k.status(),
                ..kernel
            },
            hedge,
            combined_rejects: rejects,
        }
    }

    fn combined_check(&self, lane: &HedgeLane) -> Option<CombinedCheck> {
        let limits = self.lane.combined;
        (limits.max_net_exposure.is_some() || limits.max_gross_notional.is_some()).then_some(
            CombinedCheck {
                limits,
                multiplier: self.lane.multiplier,
                hedge: lane.gate,
                hedge_tick: self.lane.inst.tick,
                hedge_lot: self.lane.inst.lot,
            },
        )
    }

    /// The last hedge row the agent sees at the decision: run before it,
    /// and observed by then.
    fn visible_row(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &HedgeLane,
    ) -> Option<(u64, KernelTick)> {
        let now = k.now_ns();
        let delay = self
            .lane
            .inst
            .latency
            .ns(self.lane.inst.latency.observation_q)?;
        let mut index = l.last_row?;
        loop {
            let tick = *self.hedge.row(index)?;
            if tick
                .ts_ns
                .checked_add(delay)
                .is_some_and(|seen| seen <= now)
            {
                return Some((index, tick));
            }
            index = index.checked_sub(1)?;
        }
    }

    fn stale(&self, tick: &KernelTick, now: u64) -> bool {
        now.saturating_sub(tick.ts_ns) > self.lane.staleness_ns
    }

    /// What the agent sees of the hedge lane.
    #[must_use]
    pub fn hedge_view(&self, s: &HedgedState<ORD, MSG, DLV, OBS>) -> HedgeView {
        self.view_parts(&s.kernel, &s.lane)
    }

    /// [`HedgedKernel::hedge_view`] on the two lanes' states held apart.
    pub(crate) fn view_parts(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &HedgeLane,
    ) -> HedgeView {
        let visible = if self.owns_parts(k, l) {
            self.visible_row(k, l)
        } else {
            None
        };
        let now = k.now_ns();
        let position = self
            .lane
            .inst
            .lot
            .to_decimal(i128::from(l.agent_lots))
            .unwrap_or(Decimal::ZERO);
        HedgeView {
            row: visible.map(|(r, _)| r),
            tick: visible.map(|(_, t)| t),
            stale: visible.is_none_or(|(_, t)| self.stale(&t, now)),
            position_lots: l.agent_lots,
            position,
            avg_entry: l.agent.avg_entry,
            in_flight: l.in_flight(),
        }
    }

    /// The sizes, as bits over `sizes`, a hedge of either direction may take
    /// now: the decision's checks except the instrument's and the combined
    /// limits, which bind at admission, so a size here can still be refused.
    #[must_use]
    pub fn hedge_mask(&self, s: &HedgedState<ORD, MSG, DLV, OBS>, sizes: &[u64]) -> u64 {
        self.mask_parts(&s.kernel, &s.lane, sizes)
    }

    /// [`HedgedKernel::hedge_mask`] on the two lanes' states held apart.
    pub(crate) fn mask_parts(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &HedgeLane,
        sizes: &[u64],
    ) -> u64 {
        if self.precheck(k, l).is_err() {
            return 0;
        }
        let mut mask = 0u64;
        for (i, lots) in sizes.iter().enumerate().take(64) {
            if *lots >= 1 && *lots <= self.lane.inst.q {
                mask |= 1u64
                    .checked_shl(u32::try_from(i).unwrap_or(64))
                    .unwrap_or(0);
            }
        }
        mask
    }

    /// The decision's checks that do not depend on the order.
    fn precheck(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &HedgeLane,
    ) -> Result<KernelTick, HedgeRefusal> {
        if !self.owns_parts(k, l) || !k.status().is_running() || k.phase() != Phase::AwaitDecision {
            return Err(HedgeRefusal::Terminal);
        }
        if l.in_flight() {
            return Err(HedgeRefusal::InFlight);
        }
        let (_, tick) = self.visible_row(k, l).ok_or(HedgeRefusal::NoTouch)?;
        if self.stale(&tick, k.now_ns()) {
            return Err(HedgeRefusal::Stale);
        }
        Ok(tick)
    }

    fn admit_hedge(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &mut HedgeLane,
        side: QuoteSide,
        lots: u64,
    ) -> Result<(), HedgeRefusal> {
        let tick = self.precheck(k, l)?;
        if lots == 0 || lots > self.lane.inst.q {
            return Err(HedgeRefusal::SizeRange);
        }
        let (dir, price) = match side {
            QuoteSide::Bid => (Dir::Buy, tick.ask),
            QuoteSide::Ask => (Dir::Sell, tick.bid),
        };
        let mut lots = lots;
        if let Some(limits) = &self.lane.inst.limits {
            let decision = gate_check(limits, &l.gate, &self.lane.inst, dir, price, lots)
                .ok_or(HedgeRefusal::Limits)?;
            match decision.0 {
                GateVerdict::Accepted | GateVerdict::Reduced => lots = decision.1,
                GateVerdict::ReducedOffLot | GateVerdict::Rejected => {
                    return Err(HedgeRefusal::Limits);
                }
            }
        }
        if let Some(check) = self.combined_check(l) {
            let admitted = check.admits(
                &k.gate,
                &self.kernel.cfg.tick,
                &self.kernel.cfg.lot,
                true,
                dir,
                price,
                lots,
            );
            if admitted != Some(true) {
                return Err(HedgeRefusal::Combined);
            }
        }
        let now = k.now_ns();
        let latency = self.lane.inst.latency;
        let mut rng = l.rng;
        let jitter = rng.draw(latency.jitter_q);
        let exec_ns = latency
            .submit_q
            .checked_add(jitter)
            .and_then(|q| latency.ns(q))
            .and_then(|ns| now.checked_add(ns))
            .ok_or(HedgeRefusal::Terminal)?;
        let id = l.next_id;
        let next_id = id.checked_add(1).ok_or(HedgeRefusal::Terminal)?;
        let submitted = l
            .counters
            .submitted
            .checked_add(1)
            .ok_or(HedgeRefusal::Terminal)?;
        let mut gate = l.gate;
        gate.reserve(dir, lots).ok_or(HedgeRefusal::Terminal)?;
        l.gate = gate;
        l.rng = rng;
        l.next_id = next_id;
        l.counters.submitted = submitted;
        l.order = Some(HedgeOrder {
            id,
            dir,
            price,
            lots,
            exec_ns,
        });
        Ok(())
    }

    /// Runs the hedge lane to the quoted lane's time: before it while
    /// deciding, through it once the run has ended. A failure halts the run.
    fn catch_up<E: EventSink>(
        &self,
        k: &mut KernelState<ORD, MSG, DLV, OBS>,
        l: &mut HedgeLane,
        em: &mut Em<'_, E>,
    ) {
        if l.closed {
            return;
        }
        let now = k.now_ns();
        let running = k.status().is_running();
        if let Err(reason) = self.run_lane(l, now, !running, em) {
            if k.status().is_running() {
                step::halt(k, reason, em);
            }
            l.closed = true;
            return;
        }
        if !k.status().is_running() {
            let pending =
                u64::from(l.order.is_some()).saturating_add(u64::from(l.report.is_some()));
            if let Some(order) = l.order.take() {
                let _ = l.gate.release(order.dir, order.lots);
            }
            l.report = None;
            l.counters.dropped_at_end = l.counters.dropped_at_end.saturating_add(pending);
            l.closed = true;
        }
    }

    /// The next lane event, if it falls before `until` (through it when
    /// `inclusive`).
    fn next_event(
        &self,
        lane: &HedgeLane,
        until: u64,
        inclusive: bool,
    ) -> Option<(u64, LaneEvent)> {
        let funding = self
            .lane
            .funding
            .get(usize::try_from(lane.funding).ok()?)
            .map(|f| (f.at_ns, LaneEvent::Funding));
        let row = self.hedge.row(lane.row).map(|t| (t.ts_ns, LaneEvent::Row));
        let execute = lane.order.map(|o| (o.exec_ns, LaneEvent::Execute));
        let deliver = lane.report.map(|r| (r.due_ns, LaneEvent::Deliver));
        [funding, row, execute, deliver]
            .into_iter()
            .flatten()
            .min()
            .filter(|(t, _)| if inclusive { *t <= until } else { *t < until })
    }

    fn run_lane<E: EventSink>(
        &self,
        lane: &mut HedgeLane,
        until: u64,
        inclusive: bool,
        sink: &mut E,
    ) -> Result<(), Halt> {
        while let Some((at, event)) = self.next_event(lane, until, inclusive) {
            match event {
                LaneEvent::Funding => self.fund(lane, at, sink)?,
                LaneEvent::Row => {
                    let tick = self.hedge.row(lane.row).ok_or(Halt::CounterOverflow)?;
                    lane.gate.mark2 = tick.mid2().filter(|m| *m > 0);
                    lane.last_row = Some(lane.row);
                    sink.hedge(HedgeEvent::Row {
                        row: lane.row,
                        ts_ns: tick.ts_ns,
                        mid2: lane.gate.mark2,
                    });
                    lane.row = lane.row.checked_add(1).ok_or(Halt::CounterOverflow)?;
                }
                LaneEvent::Execute => self.execute(lane, at, sink)?,
                LaneEvent::Deliver => self.deliver(lane)?,
            }
        }
        Ok(())
    }

    fn last_tick(&self, lane: &HedgeLane) -> Option<KernelTick> {
        lane.last_row.and_then(|r| self.hedge.row(r)).copied()
    }

    fn fund<E: EventSink>(&self, lane: &mut HedgeLane, at: u64, sink: &mut E) -> Result<(), Halt> {
        let index = usize::try_from(lane.funding).map_err(|_| Halt::CounterOverflow)?;
        let funding = *self.lane.funding.get(index).ok_or(Halt::CounterOverflow)?;
        lane.funding = lane.funding.checked_add(1).ok_or(Halt::CounterOverflow)?;
        let (Some(tick), Some(mid2)) = (self.last_tick(lane), lane.gate.mark2) else {
            lane.counters.funding_without_mark =
                lane.counters.funding_without_mark.saturating_add(1);
            return Ok(());
        };
        if self.stale(&tick, at) {
            lane.counters.funding_stale = lane.counters.funding_stale.saturating_add(1);
        }
        let inst = &self.lane.inst;
        // pos * lot * (mid2 / 2) * tick * rate, in units of 10^-s, s = ts + ls + fs.
        let (up, down) = funding_scales(inst.fs, funding.rate_s);
        let numerator = i128::from(lane.gate.venue_pos_lots)
            .checked_mul(i128::from(inst.lot.mantissa))
            .and_then(|v| v.checked_mul(i128::from(mid2)))
            .and_then(|v| v.checked_mul(i128::from(inst.tick.mantissa)))
            .and_then(|v| v.checked_mul(funding.rate_m))
            .and_then(|v| v.checked_mul(pow10_i128(up)?))
            .ok_or(Halt::LedgerOverflow)?;
        let denominator = pow10_i128(down)
            .and_then(|d| d.checked_mul(2))
            .ok_or(Halt::LedgerOverflow)?;
        let paid = div_half_even(numerator, denominator).ok_or(Halt::LedgerOverflow)?;
        lane.funding_u = lane
            .funding_u
            .checked_sub(paid)
            .ok_or(Halt::LedgerOverflow)?;
        // The shadow.
        let rate = Decimal::try_from_i128_with_scale(funding.rate_m, funding.rate_s)
            .map_err(|_| Halt::LedgerOverflow)?;
        let shadow = inst
            .lot
            .to_decimal(i128::from(lane.gate.venue_pos_lots))
            .and_then(|p| {
                let mid = inst
                    .tick
                    .to_decimal(i128::from(mid2))?
                    .checked_div(Decimal::TWO)?;
                p.checked_mul(mid)?.checked_mul(rate)
            })
            .map(|v| {
                v.round_dp_with_strategy(
                    inst.s,
                    rust_decimal::RoundingStrategy::MidpointNearestEven,
                )
            })
            .ok_or(Halt::LedgerOverflow)?;
        lane.funding_d = lane
            .funding_d
            .checked_sub(shadow)
            .ok_or(Halt::LedgerOverflow)?;
        lane.counters.funding_paid = lane.counters.funding_paid.saturating_add(1);
        self.check_ledger(lane)?;
        sink.hedge(HedgeEvent::Funding {
            at_ns: at,
            received_u: paid.checked_neg().ok_or(Halt::LedgerOverflow)?,
            mid2,
        });
        Ok(())
    }

    fn execute<E: EventSink>(
        &self,
        lane: &mut HedgeLane,
        at: u64,
        sink: &mut E,
    ) -> Result<(), Halt> {
        let order = lane.order.take().ok_or(Halt::CounterOverflow)?;
        let latency = self.lane.inst.latency;
        let jitter = lane.rng.draw(latency.jitter_q);
        let due_ns = latency
            .ack_q
            .checked_add(jitter)
            .and_then(|q| latency.ns(q))
            .and_then(|ns| at.checked_add(ns))
            .ok_or(Halt::TimeOverflow)?;
        let fresh = self.last_tick(lane).filter(|t| !self.stale(t, at));
        let touch = fresh.map(|t| match order.dir {
            Dir::Buy => (t.ask, t.ask_lots, t.ask <= order.price),
            Dir::Sell => (t.bid, t.bid_lots, t.bid >= order.price),
        });
        let (outcome, filled, price) = match touch {
            None => (HedgeFill::Stale, 0, order.price),
            Some((price, available, true)) if available > 0 => {
                let filled = order.lots.min(available);
                let outcome = if filled == order.lots {
                    HedgeFill::Filled
                } else {
                    HedgeFill::Partial
                };
                (outcome, filled, price)
            }
            Some((_, _, true)) => (HedgeFill::NoLots, 0, order.price),
            Some(_) => (HedgeFill::MovedThrough, 0, order.price),
        };
        let unfilled = order
            .lots
            .checked_sub(filled)
            .ok_or(Halt::CounterOverflow)?;
        lane.gate
            .release(order.dir, unfilled)
            .ok_or(Halt::CounterOverflow)?;
        let c = &mut lane.counters;
        match outcome {
            HedgeFill::Filled => c.filled = c.filled.saturating_add(1),
            HedgeFill::Partial => {
                c.partial = c.partial.saturating_add(1);
                c.beyond_touch_lots = c.beyond_touch_lots.saturating_add(unfilled);
            }
            HedgeFill::MovedThrough => c.moved_through = c.moved_through.saturating_add(1),
            HedgeFill::Stale => c.stale_in_flight = c.stale_in_flight.saturating_add(1),
            HedgeFill::NoLots => c.no_lots = c.no_lots.saturating_add(1),
        }
        if filled > 0 {
            self.book_fill(lane, order.dir, price, filled, at, sink)?;
        }
        lane.report = Some(HedgeReport {
            id: order.id,
            dir: order.dir,
            filled,
            price,
            outcome,
            due_ns,
        });
        Ok(())
    }

    fn book_fill<E: EventSink>(
        &self,
        lane: &mut HedgeLane,
        dir: Dir,
        price: u64,
        lots: u64,
        at: u64,
        sink: &mut E,
    ) -> Result<(), Halt> {
        let inst = &self.lane.inst;
        let notional = inst.notional_u(price, lots).ok_or(Halt::LedgerOverflow)?;
        let fee = inst.fee_u(price, lots).ok_or(Halt::LedgerOverflow)?;
        lane.cash_u = match dir {
            Dir::Buy => lane.cash_u.checked_sub(notional),
            Dir::Sell => lane.cash_u.checked_add(notional),
        }
        .ok_or(Halt::LedgerOverflow)?;
        lane.fees_u = lane.fees_u.checked_add(fee).ok_or(Halt::LedgerOverflow)?;
        lane.gate.fill(dir, lots).ok_or(Halt::CounterOverflow)?;
        let price_d = inst
            .tick
            .to_decimal(i128::from(price))
            .ok_or(Halt::LedgerOverflow)?;
        let lots_d = inst
            .lot
            .to_decimal(i128::from(signed(dir, lots).ok_or(Halt::CounterOverflow)?))
            .ok_or(Halt::LedgerOverflow)?;
        let value = lots_d.checked_mul(price_d).ok_or(Halt::LedgerOverflow)?;
        lane.cash_d = lane.cash_d.checked_sub(value).ok_or(Halt::LedgerOverflow)?;
        let rate = Decimal::try_from_i128_with_scale(inst.rate_num, inst.fs)
            .map_err(|_| Halt::LedgerOverflow)?;
        let fee_d = value.abs().checked_mul(rate).ok_or(Halt::LedgerOverflow)?;
        lane.fees_d = lane.fees_d.checked_add(fee_d).ok_or(Halt::LedgerOverflow)?;
        lane.venue
            .update_fill(lots_d, price_d, at / NS_PER_MS)
            .ok_or(Halt::LedgerOverflow)?;
        self.check_ledger(lane)?;
        sink.hedge(HedgeEvent::Fill {
            ts_ns: at,
            side: Action::from(dir),
            price,
            lots,
            fee_u: fee,
        });
        Ok(())
    }

    fn deliver(&self, lane: &mut HedgeLane) -> Result<(), Halt> {
        let report = lane.report.take().ok_or(Halt::CounterOverflow)?;
        if report.filled > 0 {
            let delta = signed(report.dir, report.filled).ok_or(Halt::CounterOverflow)?;
            lane.agent_lots = lane
                .agent_lots
                .checked_add(delta)
                .ok_or(Halt::CounterOverflow)?;
            let inst = &self.lane.inst;
            let lots_d = inst
                .lot
                .to_decimal(i128::from(delta))
                .ok_or(Halt::LedgerOverflow)?;
            let price_d = inst
                .tick
                .to_decimal(i128::from(report.price))
                .ok_or(Halt::LedgerOverflow)?;
            lane.agent
                .update_fill(lots_d, price_d, report.due_ns / NS_PER_MS)
                .ok_or(Halt::LedgerOverflow)?;
        }
        let _ = report.id;
        lane.counters.delivered = lane.counters.delivered.saturating_add(1);
        Ok(())
    }

    /// The integers within the bound and equal to the shadow.
    fn check_ledger(&self, lane: &HedgeLane) -> Result<(), Halt> {
        let s = self.lane.inst.s;
        for (units, shadow) in [
            (lane.cash_u, lane.cash_d),
            (lane.fees_u, lane.fees_d),
            (lane.funding_u, lane.funding_d),
        ] {
            if units.unsigned_abs() > self.lane.bound {
                return Err(Halt::LedgerOverflow);
            }
            if scaled_decimal(units, s) != Some(shadow) {
                return Err(Halt::LedgerOverflow);
            }
        }
        Ok(())
    }

    /// The hedge lane's venue totals, with both lanes' net.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for a state of another kernel,
    /// and [`MMError::NumericalError`] if a value does not convert.
    pub fn summary(&self, s: &HedgedState<ORD, MSG, DLV, OBS>) -> MMResult<HedgeSummary> {
        self.summary_parts(&s.kernel, &s.lane)
    }

    /// [`HedgedKernel::summary`] on the two lanes' states held apart.
    pub(crate) fn summary_parts(
        &self,
        k: &KernelState<ORD, MSG, DLV, OBS>,
        l: &HedgeLane,
    ) -> MMResult<HedgeSummary> {
        if !self.owns_parts(k, l) {
            return Err(MMError::InvalidMarketState(
                "summary of a state from another kernel".to_string(),
            ));
        }
        let fail = || MMError::NumericalError("hedge summary does not convert".to_string());
        let inst = &self.lane.inst;
        let lane = l;
        let money = |u: i128| scaled_decimal(u, inst.s).ok_or_else(fail);
        let (cash, fees, funding) = (
            money(lane.cash_u)?,
            money(lane.fees_u)?,
            money(lane.funding_u)?,
        );
        let position = inst
            .lot
            .to_decimal(i128::from(lane.gate.venue_pos_lots))
            .ok_or_else(fail)?;
        let mark_value = match lane.gate.mark2 {
            None => Decimal::ZERO,
            Some(mid2) => {
                let mid = inst
                    .tick
                    .to_decimal(i128::from(mid2))
                    .and_then(|m| m.checked_div(Decimal::TWO))
                    .ok_or_else(fail)?;
                position.checked_mul(mid).ok_or_else(fail)?
            }
        };
        let net = cash
            .checked_sub(fees)
            .and_then(|v| v.checked_add(funding))
            .and_then(|v| v.checked_add(mark_value))
            .ok_or_else(fail)?;
        let quoted = self.kernel.venue_summary(k)?;
        Ok(HedgeSummary {
            position,
            cash,
            fees,
            funding,
            mark_value,
            net,
            combined_net: quoted.net_pnl.checked_add(net).ok_or_else(fail)?,
        })
    }
}

/// The instrument's own limits for a hedge order: the verdict and the lots.
fn gate_check(
    limits: &RiskLimits,
    gate: &GateState,
    inst: &ValidatedConfig,
    dir: Dir,
    price: u64,
    lots: u64,
) -> Option<(GateVerdict, u64)> {
    let decision = gate::check(limits, gate, &inst.tick, &inst.lot, dir, price, lots).ok()?;
    Some((decision.verdict, decision.submitted))
}

/// Magic of a hedged state's canonical bytes.
pub const HEDGED_STATE_MAGIC: [u8; 4] = *b"MMHS";

fn decode_error(message: impl Into<String>) -> MMError {
    MMError::InvalidMarketState(format!("hedged state decode: {}", message.into()))
}

impl<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    HedgedState<ORD, MSG, DLV, OBS>
{
    /// Canonical bytes: the quoted lane's [`encode`](super::encode), then the
    /// hedge lane. The lane's `Decimal` shadow is not stored: a decoded
    /// state takes it from the integers.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(&HEDGED_STATE_MAGIC);
        w.u16(HEDGE_FORMAT_VERSION);
        let kernel = super::encode(&self.kernel);
        w.u64(u64::try_from(kernel.len()).unwrap_or(u64::MAX));
        w.bytes(&kernel);
        let l = &self.lane;
        let id = &l.data_id;
        w.u16(id.scheme);
        w.u64(id.len);
        w.u64(id.first_ts);
        w.u64(id.last_ts);
        w.u128(id.digest);
        w.bytes(&id.external);
        w.u128(l.cfg_fp);
        w.u64(l.row);
        w.option_u64(l.last_row);
        w.u64(l.funding);
        w.u8(l.rng.algo);
        w.u64(l.rng.seed);
        w.u64(l.rng.state);
        w.u64(l.next_id);
        match l.order {
            None => w.bool(false),
            Some(o) => {
                w.bool(true);
                w.u64(o.id);
                w.u8(dir_code(o.dir));
                w.u64(o.price);
                w.u64(o.lots);
                w.u64(o.exec_ns);
            }
        }
        match l.report {
            None => w.bool(false),
            Some(r) => {
                w.bool(true);
                w.u64(r.id);
                w.u8(dir_code(r.dir));
                w.u64(r.filled);
                w.u64(r.price);
                w.u8(match r.outcome {
                    HedgeFill::Filled => 1,
                    HedgeFill::Partial => 2,
                    HedgeFill::MovedThrough => 3,
                    HedgeFill::Stale => 4,
                    HedgeFill::NoLots => 5,
                });
                w.u64(r.due_ns);
            }
        }
        w.i64(l.gate.venue_pos_lots);
        w.u64(l.gate.live_buy_lots);
        w.u64(l.gate.live_sell_lots);
        w.option_u64(l.gate.mark2);
        w.i128(l.cash_u);
        w.i128(l.fees_u);
        w.i128(l.funding_u);
        for position in [l.venue, l.agent] {
            w.decimal(position.quantity);
            w.decimal(position.avg_entry);
            w.u64(position.last_update);
        }
        w.i64(l.agent_lots);
        w.bool(l.closed);
        let c = &l.counters;
        for v in [
            c.submitted,
            c.filled,
            c.partial,
            c.beyond_touch_lots,
            c.moved_through,
            c.stale_in_flight,
            c.no_lots,
        ] {
            w.u64(v);
        }
        for v in c.refused {
            w.u64(v);
        }
        for v in [
            c.combined_quote_rejects,
            c.funding_paid,
            c.funding_stale,
            c.funding_without_mark,
            c.dropped_at_end,
            c.delivered,
        ] {
            w.u64(v);
        }
        w.finish()
    }
}

impl<'d, const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    HedgedKernel<'d, ORD, MSG, DLV, OBS>
{
    /// Decodes a state of this kernel, refusing one that breaks an
    /// invariant the lane relies on: its datasets and configuration, its
    /// cursors, one hedge in flight with exactly its lots reserved, the mark
    /// of its last row, positions that agree with their lots, and ledger
    /// values within the bound. As for the quoted lane, this does not prove
    /// the state was reached by stepping.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for bytes that are not such a
    /// state.
    pub fn decode(&self, bytes: &[u8]) -> MMResult<HedgedState<ORD, MSG, DLV, OBS>> {
        let mut r = Reader { bytes, pos: 0 };
        if r.arr::<4>()? != HEDGED_STATE_MAGIC {
            return Err(decode_error("bad magic"));
        }
        let version = r.u16()?;
        if version != HEDGE_FORMAT_VERSION {
            return Err(decode_error(format!("unsupported format {version}")));
        }
        let kernel_len = usize::try_from(r.u64()?).map_err(|_| decode_error("length"))?;
        let kernel = super::decode(r.take(kernel_len)?, &self.kernel)?;
        let data_id = DatasetId {
            scheme: r.u16()?,
            len: r.u64()?,
            first_ts: r.u64()?,
            last_ts: r.u64()?,
            digest: r.u128()?,
            external: r.arr::<32>()?,
        };
        let cfg_fp = r.u128()?;
        if data_id != self.hedge.id || cfg_fp != self.lane.fingerprint {
            return Err(decode_error("another hedge dataset or configuration"));
        }
        let row = r.u64()?;
        let last_row = r.option_u64()?;
        let funding = r.u64()?;
        let rng = KernelRng {
            algo: r.u8()?,
            seed: r.u64()?,
            state: r.u64()?,
        };
        let next_id = r.u64()?;
        let order = if r.bool()? {
            Some(HedgeOrder {
                id: r.u64()?,
                dir: r.dir()?,
                price: r.u64()?,
                lots: r.u64()?,
                exec_ns: r.u64()?,
            })
        } else {
            None
        };
        let report = if r.bool()? {
            Some(HedgeReport {
                id: r.u64()?,
                dir: r.dir()?,
                filled: r.u64()?,
                price: r.u64()?,
                outcome: match r.u8()? {
                    1 => HedgeFill::Filled,
                    2 => HedgeFill::Partial,
                    3 => HedgeFill::MovedThrough,
                    4 => HedgeFill::Stale,
                    5 => HedgeFill::NoLots,
                    code => return Err(decode_error(format!("outcome {code}"))),
                },
                due_ns: r.u64()?,
            })
        } else {
            None
        };
        let gate = GateState {
            venue_pos_lots: r.i64()?,
            live_buy_lots: r.u64()?,
            live_sell_lots: r.u64()?,
            mark2: r.option_u64()?,
        };
        let (cash_u, fees_u, funding_u) = (r.i128()?, r.i128()?, r.i128()?);
        let mut positions = [InvDec::FLAT; 2];
        for position in &mut positions {
            *position = InvDec {
                quantity: r.decimal()?,
                avg_entry: r.decimal()?,
                last_update: r.u64()?,
            };
        }
        let [venue, agent] = positions;
        let agent_lots = r.i64()?;
        let closed = r.bool()?;
        let mut head = [0u64; 7];
        for v in &mut head {
            *v = r.u64()?;
        }
        let mut refused = [0u64; 7];
        for v in &mut refused {
            *v = r.u64()?;
        }
        let mut tail = [0u64; 6];
        for v in &mut tail {
            *v = r.u64()?;
        }
        if r.pos != bytes.len() {
            return Err(decode_error("trailing bytes"));
        }
        let [
            submitted,
            filled,
            partial,
            beyond_touch_lots,
            moved_through,
            stale_in_flight,
            no_lots,
        ] = head;
        let [
            combined_quote_rejects,
            funding_paid,
            funding_stale,
            funding_without_mark,
            dropped_at_end,
            delivered,
        ] = tail;
        let inst = &self.lane.inst;
        let fail = |what: &str| Err(decode_error(what.to_string()));
        if row > self.hedge.id.len || last_row != row.checked_sub(1) {
            return fail("hedge row cursor");
        }
        if usize::try_from(funding).map_or(true, |f| f > self.lane.funding.len()) {
            return fail("funding cursor");
        }
        if rng.algo != super::RNG_ALGO_XORSHIFT64STAR || rng.state == 0 {
            return fail("random stream");
        }
        if order.is_some() && report.is_some() || closed && (order.is_some() || report.is_some()) {
            return fail("more than one hedge in flight");
        }
        let reserved = match order {
            None => (0, 0),
            Some(o) => {
                if o.lots == 0 || o.lots > inst.q || o.price == 0 || o.price > inst.p {
                    return fail("hedge order out of range");
                }
                match o.dir {
                    Dir::Buy => (o.lots, 0),
                    Dir::Sell => (0, o.lots),
                }
            }
        };
        if (gate.live_buy_lots, gate.live_sell_lots) != reserved {
            return fail("reserved lots differ from the hedge in flight");
        }
        if report.is_some_and(|r| r.filled > inst.q || r.price > inst.p) {
            return fail("hedge report out of range");
        }
        let mark = last_row
            .and_then(|r| self.hedge.row(r))
            .and_then(KernelTick::mid2)
            .filter(|m| *m > 0);
        if gate.mark2 != mark {
            return fail("mark differs from the last hedge row");
        }
        let largest = u128::from(inst.s_max)
            .checked_mul(2)
            .and_then(|v| v.checked_mul(u128::from(inst.q)))
            .unwrap_or(u128::MAX);
        for lots in [gate.venue_pos_lots, agent_lots] {
            if u128::from(lots.unsigned_abs()) > largest {
                return fail("position beyond 2SQ lots");
            }
        }
        if inst.lot.to_decimal(i128::from(gate.venue_pos_lots)) != Some(venue.quantity)
            || inst.lot.to_decimal(i128::from(agent_lots)) != Some(agent.quantity)
        {
            return fail("position differs from its lots");
        }
        let mut shadows = [Decimal::ZERO; 3];
        for (units, shadow) in [cash_u, fees_u, funding_u].into_iter().zip(&mut shadows) {
            if units.unsigned_abs() > self.lane.bound {
                return fail("ledger value beyond the bound");
            }
            *shadow = scaled_decimal(units, inst.s).ok_or_else(|| decode_error("ledger value"))?;
        }
        let [cash_d, fees_d, funding_d] = shadows;
        Ok(HedgedState {
            kernel,
            lane: HedgeLane {
                data_id,
                cfg_fp,
                row,
                last_row,
                funding,
                rng,
                next_id,
                order,
                report,
                gate,
                cash_u,
                fees_u,
                funding_u,
                venue,
                cash_d,
                fees_d,
                funding_d,
                agent_lots,
                agent,
                closed,
                counters: HedgeCounters {
                    submitted,
                    filled,
                    partial,
                    beyond_touch_lots,
                    moved_through,
                    stale_in_flight,
                    no_lots,
                    refused,
                    combined_quote_rejects,
                    funding_paid,
                    funding_stale,
                    funding_without_mark,
                    dropped_at_end,
                    delivered,
                },
            },
        })
    }
}
