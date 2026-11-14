//! Accounts of many lanes: spot, perpetuals, futures and options together.
//!
//! See the [module documentation](super) for the attribution of one lane. This
//! file applies the same rule of the mark in force to every lane of an account,
//! each against its own marks, and adds what a derivatives venue needs: a
//! contract multiplier, option lanes, an opening position, funding, expiry
//! settlement, closes that are not fills, account flows outside the lanes, the
//! liquidity of a fill and the orders behind the fills.

use std::collections::BTreeMap;

use crate::Decimal;
use crate::execution::Action;
use crate::strategy::walk_forward::TimeWindow;

use super::EvaluationError;
use super::attribution::{
    EvalResult, ExecutedFill, LedgerTotals, Mark, MetricValue, QualitySums, Reading, SideQuality,
    add, invalid, mul, negated, sub,
};
use super::definitions::MetricId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The unit of every instant of an [`AccountInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum TimeUnit {
    /// Milliseconds, the unit of [`TimeWindow`] and [`Mark`].
    Milliseconds,
    /// Microseconds.
    Microseconds,
    /// Nanoseconds.
    Nanoseconds,
}

/// A half open interval of instants, `start <= t < end`, in a declared unit.
///
/// A caller with a closed interval `[a, b]` passes `end = b + 1` in the same
/// unit. Instants are plain ordered `u64`s: two records at distinct instants of
/// a nanosecond clock stay distinct, which a millisecond window cannot promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Window {
    /// First instant inside the window.
    pub start: u64,
    /// First instant after the window.
    pub end: u64,
    /// Unit of `start`, `end` and every record's time.
    pub unit: TimeUnit,
}

impl Window {
    /// Creates a window.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] when `start >= end`.
    pub fn new(start: u64, end: u64, unit: TimeUnit) -> EvalResult<Self> {
        if start >= end {
            return invalid(format!("window start {start} must precede end {end}"));
        }
        Ok(Self { start, end, unit })
    }

    /// Whether `time` is inside the window.
    #[must_use]
    pub fn contains(&self, time: u64) -> bool {
        self.start <= time && time < self.end
    }
}

impl From<TimeWindow> for Window {
    fn from(window: TimeWindow) -> Self {
        Self {
            start: window.start_ms,
            end: window.end_ms,
            unit: TimeUnit::Milliseconds,
        }
    }
}

/// What a lane is in the account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum LaneRole {
    /// A lane the strategy quotes.
    Quoted,
    /// A lane that hedges the quoted ones.
    Hedge,
}

/// How a lane is valued. Both are linear in the mark: a position of `c`
/// contracts with multiplier `M` is worth `c M mark`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum LaneKind {
    /// Spot, or a linear future or perpetual.
    Linear,
    /// An option valued at its mark, the premium paid or received as the fill's
    /// cash. An option fill has no base fee.
    Option,
}

/// Whether a fill added or took liquidity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Liquidity {
    /// Our resting order was hit.
    Maker,
    /// Our order crossed the book.
    Taker,
}

/// A valuation price of one lane. Never negative; zero is admitted, as an out
/// of the money option is worth nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LaneMark {
    /// Instant, in the window's unit.
    pub time: u64,
    /// Price per unit of the underlying.
    pub price: Decimal,
}

impl From<Mark> for LaneMark {
    fn from(mark: Mark) -> Self {
        Self {
            time: mark.timestamp_ms,
            price: mark.mid,
        }
    }
}

/// A fill of one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LaneFill {
    /// Execution instant, in the window's unit.
    pub time: u64,
    /// Our side: [`Action::Buy`] or [`Action::Sell`].
    pub side: Action,
    /// Price per unit of the underlying, the premium for an option. Never
    /// negative.
    pub price: Decimal,
    /// Contracts, gross of any base fee. Positive.
    pub quantity: Decimal,
    /// Fee fixed in the quote asset; negative for a rebate.
    pub quote_fee: Decimal,
    /// Fee in contracts of the lane. It leaves the position.
    pub base_fee: Decimal,
    /// Maker or taker.
    pub liquidity: Liquidity,
    /// Leave this fill out of fill quality, such as a liquidation. It still
    /// counts in the attribution.
    pub excluded_from_quality: bool,
    /// The [`OrderRecord::id`] this fill belongs to, if orders are recorded.
    pub order_id: Option<u64>,
}

impl LaneFill {
    /// The record of an [`ExecutedFill`], as a maker fill of no recorded order.
    #[must_use]
    pub fn from_executed(fill: &ExecutedFill) -> Self {
        Self {
            time: fill.timestamp_ms,
            side: fill.side,
            price: fill.price,
            quantity: fill.quantity,
            quote_fee: fill.quote_fee,
            base_fee: fill.base_fee,
            liquidity: Liquidity::Maker,
            excluded_from_quality: false,
            order_id: None,
        }
    }
}

/// A funding payment on a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LaneFunding {
    /// Payment instant, in the window's unit.
    pub time: u64,
    /// Received; negative when paid.
    pub amount: Decimal,
}

/// A position change that is not a fill, such as an auto deleveraging close.
/// It has no fee. Its cash is taken as given, as [`Settlement::cash`] is, so a
/// venue's rounding of the close stays in it; its edge against the mark in
/// force is [`LaneAttribution::transfer_edge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LaneTransfer {
    /// Instant, in the window's unit.
    pub time: u64,
    /// Our side: [`Action::Buy`] or [`Action::Sell`].
    pub side: Action,
    /// Contracts. Positive.
    pub quantity: Decimal,
    /// The signed gross cash of the close: received is positive, so a buy
    /// usually carries a negative amount.
    pub cash: Decimal,
}

/// The end of a lane's position without a fill, such as an option or a future
/// at expiry. It is the last record of its lane, apart from funding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Settlement {
    /// Instant, in the window's unit.
    pub time: u64,
    /// Settlement price, or payoff, per unit of the underlying. Never negative.
    pub price: Decimal,
    /// Cash the venue paid for the position; negative when the account paid. It
    /// is taken as given, and its difference to `position * M * price` is
    /// [`LaneAttribution::settlement_rounding`].
    pub cash: Decimal,
}

/// A position held at the start of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Opening {
    /// Contracts held; negative for a short.
    pub position: Decimal,
    /// The price it is valued at when the window starts, the mark in force until
    /// the lane's first mark. Never negative.
    pub price: Decimal,
}

/// Why account cash moved outside the lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum FlowKind {
    /// A top up after an auto deleveraging.
    AutoDeleveragingTopUp,
    /// A draw on an insurance fund.
    InsuranceFundDraw,
    /// Any other flow.
    Other,
}

/// Account cash outside every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccountFlow {
    /// Instant, in the window's unit.
    pub time: u64,
    /// Why.
    pub kind: FlowKind,
    /// Received; negative when paid.
    pub amount: Decimal,
}

/// How an order ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum OrderEnd {
    /// Fully filled.
    Filled,
    /// Cancelled, possibly after partial fills.
    Cancelled,
    /// Expired, possibly after partial fills.
    Expired,
    /// Still open when the window ended.
    Open,
}

/// An order the venue accepted on a lane.
///
/// Its offered quantity is [`OrderRecord::offered_quantity`]: the size first
/// accepted plus every positive increase of the open quantity by an accepted
/// replace. An order accepted for 10, filled for 8 and replaced to an open size
/// of 10 has offered 18. Its fills never exceed it, and an order that ended
/// [`OrderEnd::Filled`] was filled for exactly it, and an order filled for
/// exactly it ended [`OrderEnd::Filled`].
///
/// For an order accepted before the window, `accepted_quantity` is its open
/// quantity at the window start, since fills before the window cannot be
/// recorded. An order accepted before the window, open for 6 at the start and
/// filled for 6 inside it, is recorded with an accepted quantity of 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderRecord {
    /// Identifier, unique within the lane.
    pub id: u64,
    /// Contracts of the order when the venue accepted it, or its open quantity
    /// at the window start for an order accepted before the window. Positive.
    pub accepted_quantity: Decimal,
    /// The sum of every positive increase of the open quantity by an accepted
    /// replace, in contracts. Never negative; a replace that lowers the open
    /// quantity adds nothing.
    pub replace_increase: Decimal,
    /// Replaces the venue accepted. Positive when `replace_increase` is.
    pub accepted_replaces: u64,
    /// How it ended.
    pub end_state: OrderEnd,
    /// A liquidation order, left out of the fill ratios.
    pub liquidation: bool,
}

impl OrderRecord {
    /// Contracts offered: `accepted_quantity + replace_increase`.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Inexact`] if the sum would round.
    pub fn offered_quantity(&self) -> EvalResult<Decimal> {
        add(
            self.accepted_quantity,
            self.replace_increase,
            "offered quantity",
        )
    }
}

/// The records of one lane.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LaneRecords<'a> {
    /// Role.
    pub role: LaneRole,
    /// Valuation.
    pub kind: LaneKind,
    /// Units of the underlying per contract. Positive.
    pub multiplier: Decimal,
    /// The position at the start; flat when `None`.
    pub opening: Option<Opening>,
    /// Marks, in time order.
    pub marks: &'a [LaneMark],
    /// Fills, in time order.
    pub fills: &'a [LaneFill],
    /// Closes that are not fills, in time order.
    pub transfers: &'a [LaneTransfer],
    /// Funding, in time order.
    pub funding: &'a [LaneFunding],
    /// The settlement, if the position ends without a fill.
    pub settlement: Option<Settlement>,
    /// Orders.
    pub orders: &'a [OrderRecord],
}

impl<'a> LaneRecords<'a> {
    /// A flat lane with no records.
    #[must_use]
    pub fn new(role: LaneRole, kind: LaneKind, multiplier: Decimal) -> Self {
        Self {
            role,
            kind,
            multiplier,
            opening: None,
            marks: &[],
            fills: &[],
            transfers: &[],
            funding: &[],
            settlement: None,
            orders: &[],
        }
    }

    /// With an opening position.
    #[must_use]
    pub fn with_opening(mut self, opening: Opening) -> Self {
        self.opening = Some(opening);
        self
    }

    /// With marks.
    #[must_use]
    pub fn with_marks(mut self, marks: &'a [LaneMark]) -> Self {
        self.marks = marks;
        self
    }

    /// With fills.
    #[must_use]
    pub fn with_fills(mut self, fills: &'a [LaneFill]) -> Self {
        self.fills = fills;
        self
    }

    /// With closes that are not fills.
    #[must_use]
    pub fn with_transfers(mut self, transfers: &'a [LaneTransfer]) -> Self {
        self.transfers = transfers;
        self
    }

    /// With funding.
    #[must_use]
    pub fn with_funding(mut self, funding: &'a [LaneFunding]) -> Self {
        self.funding = funding;
        self
    }

    /// With a settlement.
    #[must_use]
    pub fn with_settlement(mut self, settlement: Settlement) -> Self {
        self.settlement = Some(settlement);
        self
    }

    /// With orders.
    #[must_use]
    pub fn with_orders(mut self, orders: &'a [OrderRecord]) -> Self {
        self.orders = orders;
        self
    }

    /// The opening position, zero when flat.
    fn opening_position(&self) -> Decimal {
        self.opening.map_or(Decimal::ZERO, |o| o.position)
    }
}

/// `+q` for a buy, `-q` for a sell.
fn signed(side: Action, quantity: Decimal, what: &str, time: u64) -> EvalResult<Decimal> {
    match side {
        Action::Buy => Ok(quantity),
        Action::Sell => Ok(negated(quantity)),
        _ => invalid(format!("{what} at {time} must be a buy or a sell")),
    }
}

/// Checks that `times` are inside the window and in order.
fn check_times(
    window: &Window,
    owner: &str,
    what: &str,
    times: impl Iterator<Item = u64>,
) -> EvalResult<()> {
    let mut previous = None;
    for t in times {
        if !window.contains(t) {
            return invalid(format!("{owner}: {what} at {t} is outside the window"));
        }
        if previous.is_some_and(|p| t < p) {
            return invalid(format!("{owner}: {what} at {t} is out of order"));
        }
        previous = Some(t);
    }
    Ok(())
}

fn check_lane(window: &Window, index: usize, lane: &LaneRecords<'_>) -> EvalResult<()> {
    let refuse = |reason: String| invalid(format!("lane {index}: {reason}"));
    if lane.multiplier <= Decimal::ZERO {
        return refuse(format!("multiplier {} must be positive", lane.multiplier));
    }
    if let Some(opening) = lane.opening
        && opening.price < Decimal::ZERO
    {
        return refuse(format!("opening price {} is negative", opening.price));
    }
    let owner = format!("lane {index}");
    check_times(window, &owner, "mark", lane.marks.iter().map(|m| m.time))?;
    check_times(window, &owner, "fill", lane.fills.iter().map(|f| f.time))?;
    check_times(
        window,
        &owner,
        "transfer",
        lane.transfers.iter().map(|t| t.time),
    )?;
    check_times(
        window,
        &owner,
        "funding",
        lane.funding.iter().map(|f| f.time),
    )?;
    if let Some(mark) = lane.marks.iter().find(|m| m.price < Decimal::ZERO) {
        return refuse(format!("mark at {} is negative", mark.time));
    }
    for fill in lane.fills {
        let t = fill.time;
        if fill.price < Decimal::ZERO || fill.quantity <= Decimal::ZERO {
            return refuse(format!(
                "fill at {t} needs a price of at least zero and a positive size"
            ));
        }
        if lane.kind == LaneKind::Option && !fill.base_fee.is_zero() {
            return refuse(format!("option fill at {t} has a base fee"));
        }
        signed(fill.side, fill.quantity, "fill", t)?;
    }
    for transfer in lane.transfers {
        let t = transfer.time;
        if transfer.quantity <= Decimal::ZERO {
            return refuse(format!("transfer at {t} needs a positive size"));
        }
        signed(transfer.side, transfer.quantity, "transfer", t)?;
    }
    if let Some(settlement) = lane.settlement {
        let t = settlement.time;
        if !window.contains(t) {
            return refuse(format!("settlement at {t} is outside the window"));
        }
        if settlement.price < Decimal::ZERO {
            return refuse(format!("settlement price {} is negative", settlement.price));
        }
        let last = lane
            .marks
            .iter()
            .map(|m| m.time)
            .chain(lane.fills.iter().map(|f| f.time))
            .chain(lane.transfers.iter().map(|x| x.time))
            .max();
        if last.is_some_and(|last| last > t) {
            return refuse(format!(
                "a mark, fill or transfer comes after the settlement at {t}"
            ));
        }
    }
    let mut ids = BTreeMap::new();
    for order in lane.orders {
        if order.accepted_quantity <= Decimal::ZERO {
            return refuse(format!("order {} needs a positive accepted size", order.id));
        }
        if order.replace_increase < Decimal::ZERO {
            return refuse(format!(
                "order {} has a negative replace increase",
                order.id
            ));
        }
        if order.replace_increase > Decimal::ZERO && order.accepted_replaces == 0 {
            return refuse(format!(
                "order {} has a replace increase but no accepted replace",
                order.id
            ));
        }
        let offered = order.offered_quantity()?;
        if ids
            .insert(order.id, (offered, Decimal::ZERO, order.end_state))
            .is_some()
        {
            return refuse(format!("order {} is recorded twice", order.id));
        }
    }
    for fill in lane.fills {
        let Some(id) = fill.order_id else { continue };
        let Some((offered, filled, _)) = ids.get_mut(&id) else {
            return refuse(format!("fill at {} names unknown order {id}", fill.time));
        };
        *filled = add(*filled, fill.quantity, "order fills")?;
        if *filled > *offered {
            return refuse(format!("order {id} is filled beyond its offered size"));
        }
    }
    for (id, (offered, filled, end)) in &ids {
        if *end == OrderEnd::Filled && filled != offered {
            return refuse(format!(
                "order {id} ended filled with {filled} of {offered} contracts filled"
            ));
        }
        if *end != OrderEnd::Filled && filled == offered {
            return refuse(format!(
                "order {id} is filled for its offered size but did not end filled"
            ));
        }
    }
    Ok(())
}

/// The records of one account in one window, checked.
#[derive(Debug, Clone)]
pub struct AccountInput<'a> {
    window: Window,
    lanes: Vec<LaneRecords<'a>>,
    flows: &'a [AccountFlow],
}

impl<'a> AccountInput<'a> {
    /// Checks the records of an account.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] when a record is outside the
    /// window or out of order, a multiplier or size is not positive, a price or
    /// mark is negative, a fill or transfer is neither a buy nor a sell, an option
    /// fill has a base fee, a record of a lane comes after its settlement, an
    /// order is recorded twice or has a negative replace increase, a fill names
    /// an order its lane does not record, an order is filled beyond its offered
    /// size, or an order that ended filled was filled for less than it offered.
    pub fn new(
        window: Window,
        lanes: Vec<LaneRecords<'a>>,
        flows: &'a [AccountFlow],
    ) -> EvalResult<Self> {
        for (index, lane) in lanes.iter().enumerate() {
            check_lane(&window, index, lane)?;
        }
        check_times(&window, "account", "flow", flows.iter().map(|f| f.time))?;
        Ok(Self {
            window,
            lanes,
            flows,
        })
    }

    /// The window.
    #[must_use]
    pub fn window(&self) -> Window {
        self.window
    }

    /// The lanes.
    #[must_use]
    pub fn lanes(&self) -> &[LaneRecords<'a>] {
        &self.lanes
    }
}

/// Where a lane's result came from, in the quote asset. With `M` the
/// multiplier, `c` contracts and `s` our side:
///
/// ```text
/// spread capture      = Σ s c M (M_f - P)        over fills
/// transfer edge       = Σ (X + s c M M_f)        over transfers, X the
///                       transfer's signed cash, so s c M (M_f - P) when
///                       X = -s c M P exactly
/// inventory           = Σ p_i M (M_i - M_{i-1}), from the opening price, to
///                       the settlement price if any
/// base fees           = Σ b M M_f
/// settlement rounding = settlement cash - p M S
/// net                 = spread capture + transfer edge + inventory
///                       - quote fees - base fees + funding + settlement rounding
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct LaneAttribution {
    /// Earned against the mark in force at each fill.
    pub spread_capture: Decimal,
    /// Earned against the mark in force at each close that is not a fill: its
    /// signed cash plus the value at that mark of the contracts it moved.
    pub transfer_edge: Decimal,
    /// The position times later moves of the lane's price.
    pub inventory_pnl: Decimal,
    /// Fees fixed in the quote asset; negative for rebates.
    pub quote_fees: Decimal,
    /// Fees in contracts, valued at the mark in force at each fill.
    pub base_fees: Decimal,
    /// Funding received; negative when paid.
    pub funding: Decimal,
    /// Settlement cash minus the exact value of the settled position.
    pub settlement_rounding: Decimal,
    /// The sum above.
    pub net_pnl: Decimal,
}

impl LaneAttribution {
    fn with_net(mut self) -> EvalResult<Self> {
        const WHAT: &str = "lane net pnl";
        let mut net = add(self.spread_capture, self.transfer_edge, WHAT)?;
        net = add(net, self.inventory_pnl, WHAT)?;
        net = sub(net, self.quote_fees, WHAT)?;
        net = sub(net, self.base_fees, WHAT)?;
        net = add(net, self.funding, WHAT)?;
        self.net_pnl = add(net, self.settlement_rounding, WHAT)?;
        Ok(self)
    }

    fn plus(mut self, other: &Self) -> EvalResult<Self> {
        self.spread_capture = add(self.spread_capture, other.spread_capture, "spread capture")?;
        self.transfer_edge = add(self.transfer_edge, other.transfer_edge, "transfer edge")?;
        self.inventory_pnl = add(self.inventory_pnl, other.inventory_pnl, "inventory pnl")?;
        self.quote_fees = add(self.quote_fees, other.quote_fees, "quote fees")?;
        self.base_fees = add(self.base_fees, other.base_fees, "base fees")?;
        self.funding = add(self.funding, other.funding, "funding")?;
        self.settlement_rounding = add(
            self.settlement_rounding,
            other.settlement_rounding,
            "settlement rounding",
        )?;
        self.with_net()
    }
}

/// The orders of a lane, liquidation orders left out of every count but their
/// own. Quantities are in units of the underlying, contracts times the lane's
/// multiplier, so they add up across lanes of different multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct OrderStats {
    /// Orders.
    pub orders: u64,
    /// Orders with at least one fill.
    pub filled_orders: u64,
    /// Units offered, [`OrderRecord::offered_quantity`] times the multiplier.
    pub accepted_quantity: Decimal,
    /// Units filled.
    pub filled_quantity: Decimal,
    /// Replaces accepted.
    pub accepted_replaces: u64,
    /// Orders that ended [`OrderEnd::Filled`].
    pub ended_filled: u64,
    /// Orders that ended [`OrderEnd::Cancelled`].
    pub ended_cancelled: u64,
    /// Orders that ended [`OrderEnd::Expired`].
    pub ended_expired: u64,
    /// Orders still open when the window ended, [`OrderEnd::Open`].
    pub ended_open: u64,
    /// Liquidation orders, left out of the above.
    pub liquidation_orders: u64,
}

impl OrderStats {
    /// `filled_orders / orders`; `None` with no order.
    #[must_use]
    pub fn fill_ratio_by_count(&self) -> Option<Decimal> {
        Decimal::from(self.filled_orders).checked_div(Decimal::from(self.orders))
    }

    /// `filled_quantity / accepted_quantity`; `None` with no order. Rounded by
    /// the division.
    #[must_use]
    pub fn fill_ratio_by_quantity(&self) -> Option<Decimal> {
        if self.accepted_quantity.is_zero() {
            return None;
        }
        self.filled_quantity.checked_div(self.accepted_quantity)
    }

    fn plus(mut self, other: &Self) -> EvalResult<Self> {
        self.orders = self.orders.saturating_add(other.orders);
        self.filled_orders = self.filled_orders.saturating_add(other.filled_orders);
        self.accepted_quantity = add(
            self.accepted_quantity,
            other.accepted_quantity,
            "accepted quantity",
        )?;
        self.filled_quantity = add(
            self.filled_quantity,
            other.filled_quantity,
            "filled quantity",
        )?;
        self.accepted_replaces = self
            .accepted_replaces
            .saturating_add(other.accepted_replaces);
        self.ended_filled = self.ended_filled.saturating_add(other.ended_filled);
        self.ended_cancelled = self.ended_cancelled.saturating_add(other.ended_cancelled);
        self.ended_expired = self.ended_expired.saturating_add(other.ended_expired);
        self.ended_open = self.ended_open.saturating_add(other.ended_open);
        self.liquidation_orders = self
            .liquidation_orders
            .saturating_add(other.liquidation_orders);
        Ok(self)
    }
}

fn order_stats(lane: &LaneRecords<'_>) -> EvalResult<OrderStats> {
    let m = lane.multiplier;
    let mut filled: BTreeMap<u64, Decimal> = BTreeMap::new();
    for fill in lane.fills {
        if let Some(id) = fill.order_id {
            let sum = filled.entry(id).or_default();
            *sum = add(*sum, fill.quantity, "filled quantity")?;
        }
    }
    let mut stats = OrderStats::default();
    for order in lane.orders {
        if order.liquidation {
            stats.liquidation_orders = stats.liquidation_orders.saturating_add(1);
            continue;
        }
        let done = filled.get(&order.id).copied().unwrap_or_default();
        stats.orders = stats.orders.saturating_add(1);
        if !done.is_zero() {
            stats.filled_orders = stats.filled_orders.saturating_add(1);
        }
        let offered = mul(order.offered_quantity()?, m, "offered quantity")?;
        stats.accepted_quantity = add(stats.accepted_quantity, offered, "offered quantity")?;
        let done = mul(done, m, "filled quantity")?;
        stats.filled_quantity = add(stats.filled_quantity, done, "filled quantity")?;
        stats.accepted_replaces = stats
            .accepted_replaces
            .saturating_add(order.accepted_replaces);
        let count = match order.end_state {
            OrderEnd::Filled => &mut stats.ended_filled,
            OrderEnd::Cancelled => &mut stats.ended_cancelled,
            OrderEnd::Expired => &mut stats.ended_expired,
            OrderEnd::Open => &mut stats.ended_open,
        };
        *count = count.saturating_add(1);
    }
    Ok(stats)
}

/// The result of one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct LaneOutcome {
    /// Role.
    pub role: LaneRole,
    /// Where the result came from.
    pub attribution: LaneAttribution,
    /// Contracts held at the start.
    pub opening_position: Decimal,
    /// Contracts held at the end, net of base fees; zero after a settlement.
    pub final_position: Decimal,
    /// Whether the lane settled.
    pub settled: bool,
    /// Fills.
    pub fills: u64,
    /// Units of the underlying filled, contracts times the multiplier, gross of
    /// base fees.
    pub volume: Decimal,
    /// Orders.
    pub orders: OrderStats,
}

/// A sum of lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct LaneTotals {
    /// Lanes summed.
    pub lanes: u64,
    /// Their attribution.
    pub attribution: LaneAttribution,
    /// Their fills.
    pub fills: u64,
    /// Their orders.
    pub orders: OrderStats,
}

impl LaneTotals {
    fn plus(mut self, lane: &LaneOutcome) -> EvalResult<Self> {
        self.lanes = self.lanes.saturating_add(1);
        self.attribution = self.attribution.plus(&lane.attribution)?;
        self.fills = self.fills.saturating_add(lane.fills);
        self.orders = self.orders.plus(&lane.orders)?;
        Ok(self)
    }
}

/// The attribution of an account.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct AccountAttribution {
    /// The window.
    pub window: Window,
    /// Each lane, in the order given.
    pub lanes: Vec<LaneOutcome>,
    /// The quoted lanes.
    pub quoted: LaneTotals,
    /// The hedge lanes.
    pub hedge: LaneTotals,
    /// Every lane.
    pub all_lanes: LaneTotals,
    /// Account flows outside the lanes; received is positive.
    pub external_flows: Decimal,
    /// `all_lanes.attribution.net_pnl + external_flows`.
    pub net_pnl: Decimal,
}

/// One record of a lane in the order the attribution applies it.
#[derive(Clone, Copy)]
enum Event<'a> {
    Mark(&'a LaneMark),
    Fill(&'a LaneFill),
    Transfer(&'a LaneTransfer),
    Settle(&'a Settlement),
}

/// The lane's marks, fills, transfers and settlement, by time; at one instant
/// the marks first, in the order given, then the fills, the transfers and the
/// settlement.
fn events<'a>(lane: &'a LaneRecords<'a>) -> Vec<(u64, u8, Event<'a>)> {
    let mut events: Vec<(u64, u8, Event<'a>)> = lane
        .marks
        .iter()
        .map(|m| (m.time, 0, Event::Mark(m)))
        .chain(lane.fills.iter().map(|f| (f.time, 1, Event::Fill(f))))
        .chain(
            lane.transfers
                .iter()
                .map(|t| (t.time, 2, Event::Transfer(t))),
        )
        .chain(
            lane.settlement
                .iter()
                .map(|s| (s.time, 3, Event::Settle(s))),
        )
        .collect();
    events.sort_by_key(|(time, rank, _)| (*time, *rank));
    events
}

fn attribute_lane(index: usize, lane: &LaneRecords<'_>) -> EvalResult<LaneOutcome> {
    let m = lane.multiplier;
    let opening_position = lane.opening_position();
    let mut position = opening_position;
    let mut in_force: Option<Decimal> = lane.opening.map(|o| o.price);
    let mut cash = Decimal::ZERO;
    let mut row = LaneAttribution::default();
    let mut volume = Decimal::ZERO;
    let no_mark = |what: &str, t: u64| {
        invalid::<Decimal>(format!(
            "lane {index}: the {what} at {t} has no mark or opening price at or before it"
        ))
    };
    for (time, _, event) in events(lane) {
        match event {
            Event::Mark(mark) => {
                if let Some(previous) = in_force {
                    let moved = mul(m, sub(mark.price, previous, "price move")?, "price move")?;
                    let earned = mul(position, moved, "inventory pnl")?;
                    row.inventory_pnl = add(row.inventory_pnl, earned, "inventory pnl")?;
                }
                in_force = Some(mark.price);
            }
            Event::Fill(fill) => {
                let mark = match in_force {
                    Some(mark) => mark,
                    None => no_mark("fill", time)?,
                };
                let contracts = signed(fill.side, fill.quantity, "fill", time)?;
                let units = mul(contracts, m, "fill size")?;
                let edge = sub(mark, fill.price, "spread capture")?;
                row.spread_capture = add(
                    row.spread_capture,
                    mul(units, edge, "spread capture")?,
                    "spread capture",
                )?;
                row.quote_fees = add(row.quote_fees, fill.quote_fee, "quote fees")?;
                let fee_units = mul(fill.base_fee, m, "base fees")?;
                row.base_fees = add(
                    row.base_fees,
                    mul(fee_units, mark, "base fees")?,
                    "base fees",
                )?;
                cash = sub(cash, mul(units, fill.price, "cash")?, "cash")?;
                position = add(
                    position,
                    sub(contracts, fill.base_fee, "position")?,
                    "position",
                )?;
                volume = add(volume, units.abs(), "volume")?;
            }
            Event::Transfer(transfer) => {
                let mark = match in_force {
                    Some(mark) => mark,
                    None => no_mark("transfer", time)?,
                };
                let contracts = signed(transfer.side, transfer.quantity, "transfer", time)?;
                let units = mul(contracts, m, "transfer size")?;
                let edge = add(
                    transfer.cash,
                    mul(units, mark, "transfer edge")?,
                    "transfer edge",
                )?;
                row.transfer_edge = add(row.transfer_edge, edge, "transfer edge")?;
                cash = add(cash, transfer.cash, "cash")?;
                position = add(position, contracts, "position")?;
            }
            Event::Settle(settlement) => {
                let units = mul(position, m, "settled size")?;
                if let Some(previous) = in_force {
                    let moved = sub(settlement.price, previous, "settlement move")?;
                    let earned = mul(units, moved, "inventory pnl")?;
                    row.inventory_pnl = add(row.inventory_pnl, earned, "inventory pnl")?;
                } else if !units.is_zero() {
                    return invalid(format!(
                        "lane {index}: a position settles at {time} without a price before it"
                    ));
                }
                let exact = mul(units, settlement.price, "settlement value")?;
                row.settlement_rounding = sub(settlement.cash, exact, "settlement rounding")?;
                cash = add(cash, settlement.cash, "cash")?;
                position = Decimal::ZERO;
                in_force = Some(settlement.price);
            }
        }
    }
    for payment in lane.funding {
        row.funding = add(row.funding, payment.amount, "funding")?;
    }
    let row = row.with_net()?;
    // The same result from the books: cash, the final value, less the opening
    // value, fees and funding.
    const WHAT: &str = "lane direct net";
    let last = in_force.unwrap_or_default();
    let value = mul(mul(position, m, WHAT)?, last, WHAT)?;
    let opening_value = match lane.opening {
        Some(o) => mul(mul(o.position, m, WHAT)?, o.price, WHAT)?,
        None => Decimal::ZERO,
    };
    let mut direct = add(cash, value, WHAT)?;
    direct = sub(direct, opening_value, WHAT)?;
    direct = sub(direct, row.quote_fees, WHAT)?;
    direct = add(direct, row.funding, WHAT)?;
    if direct != row.net_pnl {
        return Err(EvaluationError::NotReconciled {
            what: "lane components and cash",
            attributed: row.net_pnl,
            booked: direct,
        });
    }
    Ok(LaneOutcome {
        role: lane.role,
        attribution: row,
        opening_position,
        final_position: position,
        settled: lane.settlement.is_some(),
        fills: crate::types::decimal::saturating_u64(lane.fills.len()),
        volume,
        orders: order_stats(lane)?,
    })
}

/// Attributes every lane of an account, and the account. See
/// [`LaneAttribution`] for the terms of a lane.
///
/// # Errors
///
/// Returns [`EvaluationError::InvalidInput`] for a fill or transfer with no mark
/// and no opening price at or before it, [`EvaluationError::Inexact`] if a sum or
/// product would round, and [`EvaluationError::NotReconciled`] if a lane's terms
/// disagree with its cash, which is a defect.
pub fn attribute_account(input: &AccountInput<'_>) -> EvalResult<AccountAttribution> {
    let mut lanes = Vec::with_capacity(input.lanes.len());
    let (mut quoted, mut hedge, mut all) = (
        LaneTotals::default(),
        LaneTotals::default(),
        LaneTotals::default(),
    );
    for (index, records) in input.lanes.iter().enumerate() {
        let lane = attribute_lane(index, records)?;
        match lane.role {
            LaneRole::Quoted => quoted = quoted.plus(&lane)?,
            LaneRole::Hedge => hedge = hedge.plus(&lane)?,
        }
        all = all.plus(&lane)?;
        lanes.push(lane);
    }
    let external_flows = input
        .flows
        .iter()
        .try_fold(Decimal::ZERO, |sum, f| add(sum, f.amount, "external flows"))?;
    let net_pnl = add(all.attribution.net_pnl, external_flows, "account net pnl")?;
    Ok(AccountAttribution {
        window: input.window,
        lanes,
        quoted,
        hedge,
        all_lanes: all,
        external_flows,
        net_pnl,
    })
}

/// What the books of an account say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountBooks {
    /// Each lane's net result and final position in contracts, in the order of
    /// the input's lanes.
    pub lanes: Vec<LedgerTotals>,
    /// The account's net result, flows outside the lanes included.
    pub net_pnl: Decimal,
}

/// Residuals against the books, per lane, per role and for the account. Every
/// residual is attributed minus booked: positive when the attribution reports
/// more than the books.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct Residuals {
    /// Each lane's `(net pnl, final position)` residual.
    pub lanes: Vec<(Decimal, Decimal)>,
    /// The quoted lanes' net result residual: their attributed sum minus the
    /// sum of their booked results.
    pub quoted: Decimal,
    /// The hedge lanes' net result residual, likewise.
    pub hedge: Decimal,
    /// The account's net result residual.
    pub net_pnl: Decimal,
}

impl Residuals {
    /// The largest absolute net result residual, of a lane, a role or the
    /// account, [`MetricId::ReconciliationResidual`]. Position residuals are in
    /// contracts and are not part of it.
    #[must_use]
    pub fn largest(&self) -> Decimal {
        self.lanes
            .iter()
            .map(|(net, _)| net.abs())
            .chain([self.quoted.abs(), self.hedge.abs()])
            .fold(self.net_pnl.abs(), Decimal::max)
    }

    /// Whether every residual is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.net_pnl.is_zero()
            && self.quoted.is_zero()
            && self.hedge.is_zero()
            && self
                .lanes
                .iter()
                .all(|(net, position)| net.is_zero() && position.is_zero())
    }
}

impl AccountAttribution {
    /// Attributed minus booked, per lane, per role and for the account.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] when the books hold another
    /// number of lanes, and [`EvaluationError::Inexact`] if a difference would
    /// round.
    pub fn residuals(&self, books: &AccountBooks) -> EvalResult<Residuals> {
        if books.lanes.len() != self.lanes.len() {
            return invalid(format!(
                "the books hold {} lanes, the attribution {}",
                books.lanes.len(),
                self.lanes.len()
            ));
        }
        let lanes = self
            .lanes
            .iter()
            .zip(&books.lanes)
            .map(|(lane, booked)| {
                Ok((
                    sub(lane.attribution.net_pnl, booked.net_pnl, "residual")?,
                    sub(lane.final_position, booked.position, "residual")?,
                ))
            })
            .collect::<EvalResult<Vec<_>>>()?;
        let booked = |role: LaneRole| {
            self.lanes
                .iter()
                .zip(&books.lanes)
                .filter(|(lane, _)| lane.role == role)
                .try_fold(Decimal::ZERO, |sum, (_, b)| {
                    add(sum, b.net_pnl, "booked role")
                })
        };
        let quoted = sub(
            self.quoted.attribution.net_pnl,
            booked(LaneRole::Quoted)?,
            "residual",
        )?;
        let hedge = sub(
            self.hedge.attribution.net_pnl,
            booked(LaneRole::Hedge)?,
            "residual",
        )?;
        Ok(Residuals {
            lanes,
            quoted,
            hedge,
            net_pnl: sub(self.net_pnl, books.net_pnl, "residual")?,
        })
    }

    /// Compares every lane and the account with the books.
    ///
    /// # Errors
    ///
    /// As [`AccountAttribution::residuals`], and
    /// [`EvaluationError::NotReconciled`] for the first lane, in order, whose net
    /// result or final position differ by any amount, then for the account.
    pub fn reconcile(&self, books: &AccountBooks) -> EvalResult<()> {
        self.residuals(books)?;
        for (lane, booked) in self.lanes.iter().zip(&books.lanes) {
            if lane.attribution.net_pnl != booked.net_pnl {
                return Err(EvaluationError::NotReconciled {
                    what: "lane net pnl",
                    attributed: lane.attribution.net_pnl,
                    booked: booked.net_pnl,
                });
            }
            if lane.final_position != booked.position {
                return Err(EvaluationError::NotReconciled {
                    what: "lane final position",
                    attributed: lane.final_position,
                    booked: booked.position,
                });
            }
        }
        if self.net_pnl != books.net_pnl {
            return Err(EvaluationError::NotReconciled {
                what: "account net pnl",
                attributed: self.net_pnl,
                booked: books.net_pnl,
            });
        }
        Ok(())
    }

    /// The account's metrics: its attribution, fills and orders over every lane,
    /// and the largest reconciliation residual when `books` are given.
    /// Quantities are in units of the underlying, so ratios over lanes of
    /// different multipliers weigh each lane by its size.
    ///
    /// # Errors
    ///
    /// As [`AccountAttribution::residuals`].
    pub fn values(&self, books: Option<&AccountBooks>) -> EvalResult<Vec<MetricValue>> {
        let a = &self.all_lanes.attribution;
        let of = |id, value| MetricValue::new(id, None, Reading::Value(value));
        let mut values = vec![
            of(MetricId::NetPnl, self.net_pnl),
            of(MetricId::SpreadCapture, a.spread_capture),
            of(MetricId::InventoryPnl, a.inventory_pnl),
            of(MetricId::QuoteFees, a.quote_fees),
            of(MetricId::BaseFees, a.base_fees),
            of(MetricId::Funding, a.funding),
            of(MetricId::TransferEdge, a.transfer_edge),
            of(MetricId::ExternalFlows, self.external_flows),
            of(MetricId::SettlementRounding, a.settlement_rounding),
            of(MetricId::Fills, Decimal::from(self.all_lanes.fills)),
        ];
        values.extend(order_values(&self.all_lanes.orders));
        let residual = match books {
            Some(books) => Reading::Value(self.residuals(books)?.largest()),
            None => Reading::Undefined,
        };
        values.push(MetricValue::new(
            MetricId::ReconciliationResidual,
            None,
            residual,
        ));
        Ok(values)
    }
}

fn order_values(orders: &OrderStats) -> Vec<MetricValue> {
    let of = |id, value| MetricValue::new(id, None, Reading::Value(value));
    let optional = |id, value: Option<Decimal>| {
        MetricValue::new(id, None, value.map_or(Reading::Undefined, Reading::Value))
    };
    vec![
        of(MetricId::Orders, Decimal::from(orders.orders)),
        of(MetricId::FilledOrders, Decimal::from(orders.filled_orders)),
        of(MetricId::AcceptedQuantity, orders.accepted_quantity),
        of(MetricId::FilledQuantity, orders.filled_quantity),
        optional(MetricId::FillRatioByCount, orders.fill_ratio_by_count()),
        optional(
            MetricId::FillRatioByQuantity,
            orders.fill_ratio_by_quantity(),
        ),
    ]
}

impl LaneOutcome {
    /// The lane's metrics: its attribution, fills, volume in units of the
    /// underlying, and orders.
    #[must_use]
    pub fn values(&self) -> Vec<MetricValue> {
        let a = &self.attribution;
        let of = |id, value| MetricValue::new(id, None, Reading::Value(value));
        let mut values = vec![
            of(MetricId::NetPnl, a.net_pnl),
            of(MetricId::SpreadCapture, a.spread_capture),
            of(MetricId::InventoryPnl, a.inventory_pnl),
            of(MetricId::QuoteFees, a.quote_fees),
            of(MetricId::BaseFees, a.base_fees),
            of(MetricId::Funding, a.funding),
            of(MetricId::TransferEdge, a.transfer_edge),
            of(MetricId::SettlementRounding, a.settlement_rounding),
            of(MetricId::Fills, Decimal::from(self.fills)),
            of(MetricId::Volume, self.volume),
        ];
        values.extend(order_values(&self.orders));
        values
    }
}

/// How a lane's fills are measured after a horizon, in the window's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LaneQualitySpec {
    /// Time after the execution. Positive.
    pub horizon: u64,
    /// As [`FillQualitySpec::max_mark_delay_ms`](super::FillQualitySpec), in the
    /// window's unit.
    pub max_mark_delay: u64,
}

/// Fill quality of one lane at one horizon. Sizes and means are per unit of the
/// underlying, so a fill of `c` contracts weighs `c M`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct LaneQualityReport {
    /// The measurement.
    pub spec: LaneQualitySpec,
    /// Every fill not excluded by its flag.
    pub all: SideQuality,
    /// Our buys.
    pub buys: SideQuality,
    /// Our sells.
    pub sells: SideQuality,
    /// Maker fills.
    pub maker: SideQuality,
    /// Taker fills.
    pub taker: SideQuality,
    /// Fills left out by [`LaneFill::excluded_from_quality`].
    pub flagged: u64,
}

/// Measures the fills of one lane at one horizon, against the lane's own marks.
/// The mark in force before the lane's first mark is its opening price; a
/// settlement is not a mark. To measure a fill against the settlement price,
/// pass a [`LaneMark`] at the settlement instant with that price. Such a mark is
/// admitted and applied before the settlement. It measures a fill whose horizon
/// falls at the settlement, or before it within `max_mark_delay`; a later
/// horizon stays unmeasured. It leaves the net unchanged, and the split too
/// unless a fill or transfer shares the settlement instant.
///
/// # Errors
///
/// Returns [`EvaluationError::InvalidInput`] for a lane index out of range, a
/// zero horizon or one that overflows, or a measured fill with no mark or
/// opening price at or before it, and [`EvaluationError::Inexact`] if a sum or
/// product would round.
pub fn lane_fill_quality(
    input: &AccountInput<'_>,
    lane: usize,
    spec: LaneQualitySpec,
) -> EvalResult<LaneQualityReport> {
    let Some(records) = input.lanes.get(lane) else {
        return invalid(format!("no lane {lane}"));
    };
    if spec.horizon == 0 {
        return invalid("a markout horizon must be positive".to_string());
    }
    let marks = records.marks;
    let opening = records.opening.map(|o| o.price);
    let mut sums: [QualitySums; 5] = Default::default();
    let mut flagged = 0_u64;
    for fill in records.fills {
        if fill.excluded_from_quality {
            flagged = flagged.saturating_add(1);
            continue;
        }
        let t = fill.time;
        let Some(due) = t.checked_add(spec.horizon) else {
            return invalid(format!("horizon overflows at the fill at {t}"));
        };
        let Some(latest) = due.checked_add(spec.max_mark_delay) else {
            return invalid(format!("mark delay overflows at the fill at {t}"));
        };
        let in_force = marks.partition_point(|m| m.time <= t);
        let at_fill = in_force
            .checked_sub(1)
            .and_then(|i| marks.get(i))
            .map(|m| m.price)
            .or(opening)
            .ok_or_else(|| EvaluationError::InvalidInput {
                reason: format!("lane {lane}: the fill at {t} has no mark at or before it"),
            })?;
        let next = marks.partition_point(|m| m.time < due);
        let at_horizon = marks
            .get(next)
            .filter(|m| m.time <= latest)
            .map(|first| marks.partition_point(|m| m.time <= first.time))
            .and_then(|end| end.checked_sub(1))
            .and_then(|last| marks.get(last))
            .map(|m| m.price);
        let side = if signed(fill.side, fill.quantity, "fill", t)? > Decimal::ZERO {
            Decimal::ONE
        } else {
            Decimal::NEGATIVE_ONE
        };
        let units = mul(fill.quantity, records.multiplier, "fill size")?;
        let side_group = if side > Decimal::ZERO { 1 } else { 2 };
        let liquidity_group = match fill.liquidity {
            Liquidity::Maker => 3,
            Liquidity::Taker => 4,
        };
        for group in [0, side_group, liquidity_group] {
            if let Some(sum) = sums.get_mut(group) {
                sum.record_amount(units, fill.price, side, at_fill, at_horizon)?;
            }
        }
    }
    let [all, buys, sells, maker, taker] = sums;
    Ok(LaneQualityReport {
        spec,
        all: all.finish()?,
        buys: buys.finish()?,
        sells: sells.finish()?,
        maker: maker.finish()?,
        taker: taker.finish()?,
        flagged,
    })
}

/// What a contract specification is as a lane: its valuation and multiplier.
#[cfg(feature = "instruments")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct LaneContract {
    /// Valuation.
    pub kind: LaneKind,
    /// Units of the underlying per contract.
    pub multiplier: Decimal,
}

/// Checks that a contract is worth `contracts * M * mark`, and returns its lane
/// kind and multiplier `M`: spot, a linear future or perpetual of any base
/// quantity, or a linear option of any base quantity. Unlike
/// [`check_contract`](super::check_contract), which serves the single lane
/// attribution, a multiplier other than one is admitted.
///
/// # Errors
///
/// Returns [`EvaluationError::UnsupportedContract`] for an inverse contract, or
/// one sized in the quote asset.
#[cfg(feature = "instruments")]
pub fn lane_contract(
    spec: &financial_types::instrument::InstrumentSpec,
) -> EvalResult<LaneContract> {
    use financial_types::instrument::{ContractKind, Margining, SizeKind};
    let linear =
        spec.margining() == Margining::Linear && spec.size().kind() == SizeKind::BaseQuantity;
    let kind = match spec.kind() {
        ContractKind::Spot => Some(LaneKind::Linear),
        ContractKind::Future | ContractKind::Perpetual if linear => Some(LaneKind::Linear),
        ContractKind::Option if linear => Some(LaneKind::Option),
        _ => None,
    };
    match kind {
        Some(kind) if spec.size().amount() > Decimal::ZERO => Ok(LaneContract {
            kind,
            multiplier: spec.size().amount(),
        }),
        _ => Err(EvaluationError::UnsupportedContract {
            reason: format!(
                "{} is {:?}, {:?}, sized {:?} {}",
                spec.id().as_str(),
                spec.kind(),
                spec.margining(),
                spec.size().kind(),
                spec.size().amount()
            ),
        }),
    }
}
