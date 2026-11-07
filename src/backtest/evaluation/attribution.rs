//! The arithmetic of the metrics.

use std::collections::BTreeMap;

use crate::Decimal;
use crate::backtest::engine::SimulatedFill;
use crate::execution::Action;
use crate::strategy::walk_forward::TimeWindow;
use crate::types::decimal::{exact_add, exact_product};

use super::EvaluationError;
use super::definitions::{MetricDefinition, MetricId};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Milliseconds in a UTC day.
const DAY_MS: u64 = 86_400_000;

/// Most days [`attribute_daily`] lists, which bounds its allocation.
const MAX_CALENDAR_DAYS: u64 = 1_000_000;

type EvalResult<T> = Result<T, EvaluationError>;

fn add(a: Decimal, b: Decimal, what: &'static str) -> EvalResult<Decimal> {
    exact_add(a, b).ok_or(EvaluationError::Inexact { what })
}

/// `-value`, which a `Decimal` always holds.
fn negated(mut value: Decimal) -> Decimal {
    value.set_sign_negative(!value.is_sign_negative());
    value
}

pub(super) fn sub(a: Decimal, b: Decimal, what: &'static str) -> EvalResult<Decimal> {
    add(a, negated(b), what)
}

fn mul(a: Decimal, b: Decimal, what: &'static str) -> EvalResult<Decimal> {
    exact_product(a, b).ok_or(EvaluationError::Inexact { what })
}

fn invalid<T>(reason: String) -> EvalResult<T> {
    Err(EvaluationError::InvalidInput { reason })
}

/// A valuation price.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mark {
    /// Time, in milliseconds.
    pub timestamp_ms: u64,
    /// Mid price. Positive.
    pub mid: Decimal,
}

/// A fill, at its execution time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutedFill {
    /// Execution time, in milliseconds.
    pub timestamp_ms: u64,
    /// Our side: [`Action::Buy`] or [`Action::Sell`].
    pub side: Action,
    /// Fill price. Positive.
    pub price: Decimal,
    /// Size, gross of any base fee. Positive.
    pub quantity: Decimal,
    /// Fee fixed in the quote asset; negative for a rebate.
    pub quote_fee: Decimal,
    /// Fee in the base asset, in base units. It leaves the position.
    pub base_fee: Decimal,
}

impl ExecutedFill {
    /// The record of an engine fill.
    ///
    /// [`SimulatedFill::fee`] holds the value at the fill price of a fee in the
    /// base asset, which is not a fee in the quote asset, so with a base fee the
    /// quote fee is zero.
    #[must_use]
    pub fn from_simulated(fill: &SimulatedFill) -> Self {
        Self {
            timestamp_ms: fill.timestamp,
            side: fill.side,
            price: fill.price,
            quantity: fill.quantity,
            quote_fee: if fill.base_fee.is_zero() {
                fill.fee
            } else {
                Decimal::ZERO
            },
            base_fee: fill.base_fee,
        }
    }

    /// The size with our sign.
    fn signed_quantity(&self) -> EvalResult<Decimal> {
        match self.side {
            Action::Buy => Ok(self.quantity),
            Action::Sell => Ok(negated(self.quantity)),
            _ => invalid("a fill must be a buy or a sell".to_string()),
        }
    }
}

/// A funding payment of a hedge instrument (#266).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FundingPayment {
    /// Payment time, in milliseconds.
    pub timestamp_ms: u64,
    /// Received; negative when paid.
    pub amount: Decimal,
}

/// The records of a hedge in a second instrument, checked: its own marks, fills
/// and funding payments. Their being here is what tags them with the instrument.
#[derive(Debug, Clone, Copy)]
struct HedgeRecords<'a> {
    marks: &'a [Mark],
    fills: &'a [ExecutedFill],
    funding: &'a [FundingPayment],
}

/// The records of one window, checked.
#[derive(Debug, Clone, Copy)]
pub struct EvaluationInput<'a> {
    window: TimeWindow,
    marks: &'a [Mark],
    fills: &'a [ExecutedFill],
    capital: Decimal,
    hedge: Option<HedgeRecords<'a>>,
}

/// Checks the marks and fills of one instrument against a window.
fn check_lane(window: TimeWindow, marks: &[Mark], fills: &[ExecutedFill]) -> EvalResult<()> {
    let inside = |t: u64| window.start_ms <= t && t < window.end_ms;
    let mut previous = marks.first().map_or(0, |m| m.timestamp_ms);
    for mark in marks {
        if !inside(mark.timestamp_ms) {
            return invalid(format!(
                "mark at {} is outside the window",
                mark.timestamp_ms
            ));
        }
        if mark.timestamp_ms < previous {
            return invalid(format!("mark at {} is out of order", mark.timestamp_ms));
        }
        if mark.mid <= Decimal::ZERO {
            return invalid(format!("mark at {} is not positive", mark.timestamp_ms));
        }
        previous = mark.timestamp_ms;
    }
    let first = marks.first().map(|m| m.timestamp_ms);
    let mut previous = first.unwrap_or(0);
    for fill in fills {
        let t = fill.timestamp_ms;
        if first.is_none_or(|first| t < first) {
            return Err(EvaluationError::FillBeforeFirstMark { timestamp_ms: t });
        }
        if !inside(t) {
            return invalid(format!("fill at {t} is outside the window"));
        }
        if t < previous {
            return invalid(format!("fill at {t} is out of order"));
        }
        if fill.price <= Decimal::ZERO || fill.quantity <= Decimal::ZERO {
            return invalid(format!("fill at {t} needs a positive price and size"));
        }
        fill.signed_quantity()?;
        previous = t;
    }
    Ok(())
}

impl<'a> EvaluationInput<'a> {
    /// Checks the records of a window.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidInput`] when there is no mark, a mark or
    /// a fill is outside the window or out of order, a mid, price, size or the
    /// capital is not positive, or a fill is neither a buy nor a sell, and
    /// [`EvaluationError::FillBeforeFirstMark`] for a fill before the first mark.
    pub fn new(
        window: TimeWindow,
        marks: &'a [Mark],
        fills: &'a [ExecutedFill],
        capital: Decimal,
    ) -> EvalResult<Self> {
        if capital <= Decimal::ZERO {
            return invalid(format!("capital {capital} must be positive"));
        }
        if marks.is_empty() {
            return invalid("a window needs at least one mark".to_string());
        }
        check_lane(window, marks, fills)?;
        Ok(Self {
            window,
            marks,
            fills,
            capital,
            hedge: None,
        })
    }

    /// Adds the records of a hedge in a second instrument (#266): its marks, its
    /// fills and its funding payments, in the same window. The attribution then
    /// reports the hedge apart, in [`Attribution::hedge_costs`].
    ///
    /// The hedge may have no record at all, a lane that has not traded yet. Its
    /// fees are in the quote asset only.
    ///
    /// # Errors
    ///
    /// As [`EvaluationInput::new`] for the hedge's marks and fills, with
    /// [`EvaluationError::FillBeforeFirstMark`] for a fill before the hedge's first
    /// mark; [`EvaluationError::InvalidInput`] for a hedge fill with a base fee,
    /// or a funding payment outside the window or out of order.
    pub fn with_hedge(
        mut self,
        marks: &'a [Mark],
        fills: &'a [ExecutedFill],
        funding: &'a [FundingPayment],
    ) -> EvalResult<Self> {
        check_lane(self.window, marks, fills)?;
        if let Some(fill) = fills.iter().find(|f| !f.base_fee.is_zero()) {
            return invalid(format!(
                "hedge fill at {} has a base fee",
                fill.timestamp_ms
            ));
        }
        let inside = |t: u64| self.window.start_ms <= t && t < self.window.end_ms;
        let mut previous = funding.first().map_or(0, |f| f.timestamp_ms);
        for payment in funding {
            let t = payment.timestamp_ms;
            if !inside(t) {
                return invalid(format!("funding at {t} is outside the window"));
            }
            if t < previous {
                return invalid(format!("funding at {t} is out of order"));
            }
            previous = t;
        }
        self.hedge = Some(HedgeRecords {
            marks,
            fills,
            funding,
        });
        Ok(self)
    }

    /// The window.
    #[must_use]
    pub fn window(&self) -> TimeWindow {
        self.window
    }
}

/// One step of a window, in the order the attribution needs.
enum Step<'a> {
    /// The price moved to `mark`, from `previous` if there was one.
    Move {
        mark: &'a Mark,
        previous: Option<&'a Mark>,
    },
    /// A fill, at the mark in force.
    Fill {
        fill: &'a ExecutedFill,
        mark: &'a Mark,
    },
    /// All the fills up to `mark` are applied.
    Settled { mark: &'a Mark },
}

/// Walks the quoted instrument's marks and fills in time order.
fn sweep<'a>(
    input: &EvaluationInput<'a>,
    on_step: impl FnMut(Step<'a>) -> EvalResult<()>,
) -> EvalResult<()> {
    sweep_lane(input.marks, input.fills, on_step)
}

/// Walks one instrument's marks and fills in time order.
fn sweep_lane<'a>(
    marks: &'a [Mark],
    fills: &'a [ExecutedFill],
    mut on_step: impl FnMut(Step<'a>) -> EvalResult<()>,
) -> EvalResult<()> {
    let mut fills = fills.iter().peekable();
    let mut in_force: Option<&'a Mark> = None;
    let mut marks = marks.iter().peekable();
    while let Some(mark) = marks.next() {
        while let (Some(fill), Some(current)) = (
            fills.next_if(|fill| fill.timestamp_ms < mark.timestamp_ms),
            in_force,
        ) {
            on_step(Step::Fill {
                fill,
                mark: current,
            })?;
        }
        on_step(Step::Move {
            mark,
            previous: in_force,
        })?;
        in_force = Some(mark);
        let last_at_this_time = marks
            .peek()
            .is_none_or(|next| next.timestamp_ms > mark.timestamp_ms);
        if last_at_this_time {
            while let Some(fill) = fills.next_if(|fill| fill.timestamp_ms == mark.timestamp_ms) {
                on_step(Step::Fill { fill, mark })?;
            }
        }
        on_step(Step::Settled { mark })?;
    }
    if let Some(mark) = in_force {
        for fill in fills {
            on_step(Step::Fill { fill, mark })?;
        }
    }
    Ok(())
}

/// Why hedge costs have no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum HedgeNotModelledReason {
    /// The records hold one instrument, so they have no hedge leg.
    SingleInstrument,
}

/// Costs of a hedge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum HedgeCosts {
    /// There is no value, which is not the same as a cost of zero.
    NotModelled {
        /// Why.
        reason: HedgeNotModelledReason,
    },
    /// The hedge lane's own attribution, from the records of
    /// [`EvaluationInput::with_hedge`] (#266).
    Attributed(HedgeAttribution),
}

/// The attribution of a hedge in a second instrument, against its own marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct HedgeAttribution {
    /// Earned against the hedge's mark in force at each of its fills; negative
    /// when the hedge pays the spread.
    pub spread_capture: Decimal,
    /// The hedge position times later moves of the hedge's mark.
    pub inventory_pnl: Decimal,
    /// Fees in the quote asset; negative for rebates.
    pub fees: Decimal,
    /// Funding received; negative when paid.
    pub funding: Decimal,
    /// `spread_capture + inventory_pnl - fees + funding`.
    pub net_pnl: Decimal,
}

impl HedgeAttribution {
    const ZERO: Self = Self {
        spread_capture: Decimal::ZERO,
        inventory_pnl: Decimal::ZERO,
        fees: Decimal::ZERO,
        funding: Decimal::ZERO,
        net_pnl: Decimal::ZERO,
    };

    /// What the hedge cost to run, apart from its inventory result: `fees -
    /// spread_capture - funding`, positive for a cost, as [`MetricId::HedgeCosts`]
    /// reads.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Inexact`] if the difference would round.
    pub fn costs(&self) -> EvalResult<Decimal> {
        const WHAT: &str = "hedge costs";
        sub(
            sub(self.fees, self.spread_capture, WHAT)?,
            self.funding,
            WHAT,
        )
    }

    fn with_net(mut self) -> EvalResult<Self> {
        const WHAT: &str = "hedge net pnl";
        let gross = add(self.spread_capture, self.inventory_pnl, WHAT)?;
        self.net_pnl = add(sub(gross, self.fees, WHAT)?, self.funding, WHAT)?;
        Ok(self)
    }

    fn plus(mut self, other: &Self) -> EvalResult<Self> {
        self.spread_capture = add(self.spread_capture, other.spread_capture, "hedge spread")?;
        self.inventory_pnl = add(self.inventory_pnl, other.inventory_pnl, "hedge inventory")?;
        self.fees = add(self.fees, other.fees, "hedge fees")?;
        self.funding = add(self.funding, other.funding, "hedge funding")?;
        self.with_net()
    }
}

impl HedgeCosts {
    fn plus(self, other: Self) -> EvalResult<Self> {
        Ok(match (self, other) {
            (Self::Attributed(a), Self::Attributed(b)) => Self::Attributed(a.plus(&b)?),
            (Self::Attributed(a), Self::NotModelled { .. })
            | (Self::NotModelled { .. }, Self::Attributed(a)) => Self::Attributed(a),
            (not_modelled @ Self::NotModelled { .. }, Self::NotModelled { .. }) => not_modelled,
        })
    }
}

/// Where a result came from. See the [module documentation](super).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct Attribution {
    /// Earned against the mark in force at each fill.
    pub spread_capture: Decimal,
    /// The position, net of base fees, times later price moves.
    pub inventory_pnl: Decimal,
    /// Fees fixed in the quote asset; negative for rebates.
    pub quote_fees: Decimal,
    /// Fees in the base asset, valued at the mark in force at each fill.
    pub base_fees: Decimal,
    /// The hedge lane's, when the records hold one; not part of
    /// [`Attribution::net_pnl`].
    pub hedge_costs: HedgeCosts,
    /// `spread_capture + inventory_pnl - quote_fees - base_fees`.
    pub net_pnl: Decimal,
}

impl Attribution {
    fn zero() -> Self {
        Self {
            spread_capture: Decimal::ZERO,
            inventory_pnl: Decimal::ZERO,
            quote_fees: Decimal::ZERO,
            base_fees: Decimal::ZERO,
            hedge_costs: HedgeCosts::NotModelled {
                reason: HedgeNotModelledReason::SingleInstrument,
            },
            net_pnl: Decimal::ZERO,
        }
    }

    fn with_net(mut self) -> EvalResult<Self> {
        const WHAT: &str = "net pnl";
        let gross = add(self.spread_capture, self.inventory_pnl, WHAT)?;
        let after_quote = sub(gross, self.quote_fees, WHAT)?;
        self.net_pnl = sub(after_quote, self.base_fees, WHAT)?;
        Ok(self)
    }

    /// [`Attribution::zero`], with a hedge of zero when the records hold one.
    fn zero_for(input: &EvaluationInput<'_>) -> Self {
        let mut zero = Self::zero();
        if input.hedge.is_some() {
            zero.hedge_costs = HedgeCosts::Attributed(HedgeAttribution::ZERO);
        }
        zero
    }

    fn plus(mut self, other: &Self) -> EvalResult<Self> {
        self.spread_capture = add(self.spread_capture, other.spread_capture, "spread capture")?;
        self.inventory_pnl = add(self.inventory_pnl, other.inventory_pnl, "inventory pnl")?;
        self.quote_fees = add(self.quote_fees, other.quote_fees, "quote fees")?;
        self.base_fees = add(self.base_fees, other.base_fees, "base fees")?;
        self.hedge_costs = self.hedge_costs.plus(other.hedge_costs)?;
        self.with_net()
    }

    /// The hedge's attribution, when there is one.
    fn hedge_mut(&mut self) -> Option<&mut HedgeAttribution> {
        match &mut self.hedge_costs {
            HedgeCosts::Attributed(hedge) => Some(hedge),
            HedgeCosts::NotModelled { .. } => None,
        }
    }
}

/// What the books of a run say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerTotals {
    /// Net result, after fees.
    pub net_pnl: Decimal,
    /// Final position, net of base fees.
    pub position: Decimal,
}

/// The attribution of one bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct BucketAttribution {
    /// The attribution.
    pub attribution: Attribution,
    /// Marks in this bucket. Zero on a calendar day with no observation.
    pub marks: u64,
    /// Fills in this bucket.
    pub fills: u64,
}

/// An attribution split by label.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct BucketedAttribution<K: Ord> {
    /// The whole window; the exact sum of the buckets.
    pub total: Attribution,
    /// Final position, net of base fees.
    pub final_position: Decimal,
    /// The hedge's final position, when the records hold a hedge (#266).
    pub hedge_final_position: Option<Decimal>,
    /// The buckets.
    pub buckets: BTreeMap<K, BucketAttribution>,
}

impl<K: Ord> BucketedAttribution<K> {
    /// Compares the attribution with the books of the same run.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::NotReconciled`] when the net result or the final
    /// position differ by any amount.
    pub fn reconcile(&self, books: LedgerTotals) -> EvalResult<()> {
        if self.total.net_pnl != books.net_pnl {
            return Err(EvaluationError::NotReconciled {
                what: "net pnl",
                attributed: self.total.net_pnl,
                booked: books.net_pnl,
            });
        }
        if self.final_position != books.position {
            return Err(EvaluationError::NotReconciled {
                what: "final position",
                attributed: self.final_position,
                booked: books.position,
            });
        }
        Ok(())
    }

    /// Compares both lanes with their books (#266): the quoted lane as
    /// [`BucketedAttribution::reconcile`], and the hedge's net result and final
    /// position with `hedge`.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::NotReconciled`] when a net result or a final
    /// position differ by any amount, and [`EvaluationError::InvalidInput`] when the
    /// records hold no hedge.
    pub fn reconcile_lanes(&self, quoted: LedgerTotals, hedge: LedgerTotals) -> EvalResult<()> {
        self.reconcile(quoted)?;
        let (HedgeCosts::Attributed(attributed), Some(position)) =
            (self.total.hedge_costs, self.hedge_final_position)
        else {
            return invalid("the records hold no hedge".to_string());
        };
        if attributed.net_pnl != hedge.net_pnl {
            return Err(EvaluationError::NotReconciled {
                what: "hedge net pnl",
                attributed: attributed.net_pnl,
                booked: hedge.net_pnl,
            });
        }
        if position != hedge.position {
            return Err(EvaluationError::NotReconciled {
                what: "hedge final position",
                attributed: position,
                booked: hedge.position,
            });
        }
        Ok(())
    }
}

/// Attributes a window, each mark's label naming its bucket. See the
/// [module documentation](super).
///
/// The label is called once per mark, in time order, so a stateful labeller sees
/// only the past.
///
/// # Errors
///
/// Returns [`EvaluationError::Inexact`] if a sum or product would round, and
/// [`EvaluationError::NotReconciled`] if the components disagree with the direct
/// computation of the net result, which is a defect.
pub fn attribute_by<K: Ord + Clone>(
    input: &EvaluationInput<'_>,
    mut label: impl FnMut(&Mark) -> K,
) -> EvalResult<BucketedAttribution<K>> {
    let mut buckets: BTreeMap<K, BucketAttribution> = BTreeMap::new();
    let mut key: Option<K> = None;
    // The bucket of each quoted mark, in time order, for the hedge's records.
    let mut timeline: Vec<(u64, K)> = Vec::with_capacity(input.marks.len());
    let mut position = Decimal::ZERO;
    let mut cash = Decimal::ZERO;
    let mut last_mid = Decimal::ZERO;
    let empty = Attribution::zero_for(input);
    sweep(input, |step| {
        match step {
            Step::Move { mark, previous } => {
                let this = label(mark);
                if input.hedge.is_some() {
                    timeline.push((mark.timestamp_ms, this.clone()));
                }
                let bucket = buckets.entry(this.clone()).or_insert(BucketAttribution {
                    attribution: empty,
                    marks: 0,
                    fills: 0,
                });
                bucket.marks = bucket.marks.saturating_add(1);
                if let Some(previous) = previous {
                    let moved = sub(mark.mid, previous.mid, "price move")?;
                    let earned = mul(position, moved, "inventory pnl")?;
                    bucket.attribution.inventory_pnl =
                        add(bucket.attribution.inventory_pnl, earned, "inventory pnl")?;
                }
                last_mid = mark.mid;
                key = Some(this);
            }
            Step::Fill { fill, mark } => {
                let Some(bucket) = key.as_ref().and_then(|k| buckets.get_mut(k)) else {
                    return Err(EvaluationError::FillBeforeFirstMark {
                        timestamp_ms: fill.timestamp_ms,
                    });
                };
                let signed = fill.signed_quantity()?;
                let edge = sub(mark.mid, fill.price, "spread capture")?;
                let captured = mul(signed, edge, "spread capture")?;
                let base_value = mul(fill.base_fee, mark.mid, "base fees")?;
                let row = &mut bucket.attribution;
                row.spread_capture = add(row.spread_capture, captured, "spread capture")?;
                row.quote_fees = add(row.quote_fees, fill.quote_fee, "quote fees")?;
                row.base_fees = add(row.base_fees, base_value, "base fees")?;
                bucket.fills = bucket.fills.saturating_add(1);
                let notional = mul(signed, fill.price, "cash")?;
                cash = sub(cash, notional, "cash")?;
                let delta = sub(signed, fill.base_fee, "position")?;
                position = add(position, delta, "position")?;
            }
            Step::Settled { .. } => {}
        }
        Ok(())
    })?;
    let hedge_position = match input.hedge {
        Some(hedge) => Some(attribute_hedge(&hedge, &timeline, &mut buckets)?),
        None => None,
    };
    let mut total = Attribution::zero();
    for bucket in buckets.values_mut() {
        if let Some(hedge) = bucket.attribution.hedge_mut() {
            *hedge = hedge.with_net()?;
        }
        bucket.attribution = bucket.attribution.with_net()?;
        total = total.plus(&bucket.attribution)?;
    }
    let value = mul(position, last_mid, "final value")?;
    let direct = sub(
        add(cash, value, "direct net")?,
        total.quote_fees,
        "direct net",
    )?;
    if direct != total.net_pnl {
        return Err(EvaluationError::NotReconciled {
            what: "components and cash",
            attributed: total.net_pnl,
            booked: direct,
        });
    }
    if let (Some(hedge), Some((h_cash, h_position, h_mid))) = (input.hedge, hedge_position) {
        const WHAT: &str = "hedge direct net";
        let HedgeCosts::Attributed(attributed) = total.hedge_costs else {
            return invalid("a hedge without its attribution".to_string());
        };
        let funding = hedge
            .funding
            .iter()
            .try_fold(Decimal::ZERO, |sum, p| add(sum, p.amount, WHAT))?;
        let value = mul(h_position, h_mid, WHAT)?;
        let direct = add(
            sub(add(h_cash, value, WHAT)?, attributed.fees, WHAT)?,
            funding,
            WHAT,
        )?;
        if direct != attributed.net_pnl {
            return Err(EvaluationError::NotReconciled {
                what: "hedge components and cash",
                attributed: attributed.net_pnl,
                booked: direct,
            });
        }
    }
    Ok(BucketedAttribution {
        total,
        final_position: position,
        hedge_final_position: hedge_position.map(|(_, position, _)| position),
        buckets,
    })
}

/// The bucket in force at `timestamp_ms`: that of the last quoted mark at or
/// before it, or the first bucket for a record before the first quoted mark.
fn bucket_at<'b, K: Ord>(
    timeline: &[(u64, K)],
    buckets: &'b mut BTreeMap<K, BucketAttribution>,
    timestamp_ms: u64,
) -> EvalResult<&'b mut HedgeAttribution> {
    let index = timeline
        .partition_point(|(t, _)| *t <= timestamp_ms)
        .saturating_sub(1);
    timeline
        .get(index)
        .and_then(|(_, key)| buckets.get_mut(key))
        .and_then(|bucket| bucket.attribution.hedge_mut())
        .ok_or(EvaluationError::InvalidInput {
            reason: "a hedge record without a quoted bucket".to_string(),
        })
}

/// Sweeps the hedge's marks and fills in their own order, each term into the
/// quoted bucket in force at its time, and its funding likewise. Returns the
/// hedge's cash, final position and last mark.
fn attribute_hedge<K: Ord>(
    hedge: &HedgeRecords<'_>,
    timeline: &[(u64, K)],
    buckets: &mut BTreeMap<K, BucketAttribution>,
) -> EvalResult<(Decimal, Decimal, Decimal)> {
    let (mut cash, mut position, mut last_mid) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    sweep_lane(hedge.marks, hedge.fills, |step| {
        match step {
            Step::Move { mark, previous } => {
                if let Some(previous) = previous {
                    let moved = sub(mark.mid, previous.mid, "hedge price move")?;
                    let earned = mul(position, moved, "hedge inventory")?;
                    let row = bucket_at(timeline, buckets, mark.timestamp_ms)?;
                    row.inventory_pnl = add(row.inventory_pnl, earned, "hedge inventory")?;
                }
                last_mid = mark.mid;
            }
            Step::Fill { fill, mark } => {
                let signed = fill.signed_quantity()?;
                let edge = sub(mark.mid, fill.price, "hedge spread")?;
                let captured = mul(signed, edge, "hedge spread")?;
                let row = bucket_at(timeline, buckets, fill.timestamp_ms)?;
                row.spread_capture = add(row.spread_capture, captured, "hedge spread")?;
                row.fees = add(row.fees, fill.quote_fee, "hedge fees")?;
                cash = sub(cash, mul(signed, fill.price, "hedge cash")?, "hedge cash")?;
                position = add(position, signed, "hedge position")?;
            }
            Step::Settled { .. } => {}
        }
        Ok(())
    })?;
    for payment in hedge.funding {
        let row = bucket_at(timeline, buckets, payment.timestamp_ms)?;
        row.funding = add(row.funding, payment.amount, "hedge funding")?;
    }
    Ok((cash, position, last_mid))
}

/// Attributes a whole window, as one bucket.
///
/// # Errors
///
/// As [`attribute_by`].
pub fn attribute(input: &EvaluationInput<'_>) -> EvalResult<BucketedAttribution<()>> {
    attribute_by(input, |_| ())
}

/// The attribution of one UTC day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct DailyAttribution {
    /// Whole UTC days since the Unix epoch.
    pub day: u64,
    /// The day. Zero, with `marks == 0`, for a day with no marks. Its `fills` count by
    /// the mark in force and not by execution day, so a fill on a day with no marks
    /// counts on the earlier day that owns that mark.
    pub bucket: BucketAttribution,
}

/// Attributes a window per UTC day, every day of the window listed in order.
///
/// # Errors
///
/// As [`attribute_by`], and [`EvaluationError::InvalidInput`] for a window of more
/// than a million days.
pub fn attribute_daily(
    input: &EvaluationInput<'_>,
) -> EvalResult<(BucketedAttribution<u64>, Vec<DailyAttribution>)> {
    let first = input.window.start_ms / DAY_MS;
    let last = input.window.end_ms.saturating_sub(1) / DAY_MS;
    if last.saturating_sub(first) >= MAX_CALENDAR_DAYS {
        return invalid(format!("the window spans days {first} to {last}"));
    }
    let split = attribute_by(input, |mark| mark.timestamp_ms / DAY_MS)?;
    let calendar = (first..=last)
        .map(|day| DailyAttribution {
            day,
            bucket: split
                .buckets
                .get(&day)
                .copied()
                .unwrap_or(BucketAttribution {
                    attribution: Attribution::zero_for(input),
                    marks: 0,
                    fills: 0,
                }),
        })
        .collect();
    Ok((split, calendar))
}

/// The largest fall of `cash + position * mark - quote fees` from its running peak,
/// sampled at each mark after the fills up to it. The first peak is zero, the flat
/// start.
///
/// # Errors
///
/// Returns [`EvaluationError::Inexact`] if a sum or product would round.
pub fn max_drawdown(input: &EvaluationInput<'_>) -> EvalResult<Decimal> {
    let (mut cash, mut position) = (Decimal::ZERO, Decimal::ZERO);
    let (mut peak, mut worst) = (Decimal::ZERO, Decimal::ZERO);
    sweep(input, |step| {
        match step {
            Step::Move { .. } => {}
            Step::Fill { fill, .. } => {
                let signed = fill.signed_quantity()?;
                let notional = mul(signed, fill.price, "cash")?;
                cash = sub(sub(cash, notional, "cash")?, fill.quote_fee, "cash")?;
                let delta = sub(signed, fill.base_fee, "position")?;
                position = add(position, delta, "position")?;
            }
            Step::Settled { mark } => {
                let value = mul(position, mark.mid, "equity")?;
                let equity = add(cash, value, "equity")?;
                peak = peak.max(equity);
                worst = worst.max(sub(peak, equity, "drawdown")?);
            }
        }
        Ok(())
    })?;
    Ok(worst)
}

/// The position over a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct InventoryStats {
    /// Time weighted mean of the absolute position between the first and the last
    /// mark; `None` when they coincide. Rounded by the division.
    pub mean_abs: Option<Decimal>,
    /// Largest absolute position.
    pub max_abs: Decimal,
    /// Position at the end.
    pub final_position: Decimal,
}

/// The position over a window, net of base fees. It is a step function that
/// changes at each fill, and starts flat at the first mark.
///
/// # Errors
///
/// Returns [`EvaluationError::Inexact`] if a sum or product would round.
pub fn inventory_stats(input: &EvaluationInput<'_>) -> EvalResult<InventoryStats> {
    let Some((first, last)) = input.marks.first().zip(input.marks.last()) else {
        return invalid("a window needs at least one mark".to_string());
    };
    let (start, end) = (first.timestamp_ms, last.timestamp_ms);
    let mut position = Decimal::ZERO;
    let mut max_abs = Decimal::ZERO;
    let mut integral = Decimal::ZERO;
    let mut since = start;
    sweep(input, |step| {
        match step {
            Step::Move { .. } | Step::Settled { .. } => {}
            Step::Fill { fill, .. } => {
                let until = fill.timestamp_ms.min(end);
                let held = Decimal::from(until.saturating_sub(since));
                let area = mul(position.abs(), held, "inventory integral")?;
                integral = add(integral, area, "inventory integral")?;
                since = until;
                let delta = sub(fill.signed_quantity()?, fill.base_fee, "position")?;
                position = add(position, delta, "position")?;
                max_abs = max_abs.max(position.abs());
            }
        }
        Ok(())
    })?;
    let held = Decimal::from(end.saturating_sub(since));
    let area = mul(position.abs(), held, "inventory integral")?;
    integral = add(integral, area, "inventory integral")?;
    let span = Decimal::from(end.saturating_sub(start));
    let mean_abs = if span.is_zero() {
        None
    } else {
        Some(integral.checked_div(span).ok_or(EvaluationError::Inexact {
            what: "mean inventory",
        })?)
    };
    Ok(InventoryStats {
        mean_abs,
        max_abs,
        final_position: position,
    })
}

/// How a fill is measured after a horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FillQualitySpec {
    /// Time after the execution, in milliseconds. Positive.
    pub horizon_ms: u64,
    /// The mark used is the first at or after the horizon, the last of them if several
    /// tie at that time, and it may be at most
    /// this late. A fill with none is excluded and counted: an earlier mark is
    /// never used in its place.
    pub max_mark_delay_ms: u64,
}

/// Fill quality of one group of fills. Means are weighted by size over the measured
/// fills, and rounded by the division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct SideQuality {
    /// Fills in the group.
    pub fills: u64,
    /// Fills with a mark at the horizon.
    pub measured: u64,
    /// Fills without one, excluded from the means.
    pub excluded: u64,
    /// Size of the measured fills.
    pub measured_volume: Decimal,
    /// [`MetricId::SignedMarkout`]; `None` with nothing measured.
    pub signed_markout: Option<Decimal>,
    /// [`MetricId::RealizedSpread`]; `None` with nothing measured.
    pub realized_spread: Option<Decimal>,
    /// [`MetricId::EffectiveSpread`] of the measured fills; `None` with nothing
    /// measured.
    pub effective_spread: Option<Decimal>,
    /// [`MetricId::PriceImpact`]; `None` with nothing measured.
    pub price_impact: Option<Decimal>,
}

/// Fill quality at one horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct FillQualityReport {
    /// The measurement.
    pub spec: FillQualitySpec,
    /// Every fill.
    pub all: SideQuality,
    /// Our buys.
    pub buys: SideQuality,
    /// Our sells.
    pub sells: SideQuality,
}

/// Running sums of a [`SideQuality`].
#[derive(Default)]
struct QualitySums {
    fills: u64,
    measured: u64,
    volume: Decimal,
    /// `Σ q s (M_H - P)`.
    markout: Decimal,
    /// `Σ q s (M_f - P)`.
    capture: Decimal,
}

impl QualitySums {
    fn record(
        &mut self,
        fill: &ExecutedFill,
        side: Decimal,
        at_fill: Decimal,
        at_horizon: Option<Decimal>,
    ) -> EvalResult<()> {
        self.fills = self.fills.saturating_add(1);
        let Some(at_horizon) = at_horizon else {
            return Ok(());
        };
        self.measured = self.measured.saturating_add(1);
        self.volume = add(self.volume, fill.quantity, "measured volume")?;
        let signed = mul(side, fill.quantity, "fill quality")?;
        let later = mul(signed, sub(at_horizon, fill.price, "markout")?, "markout")?;
        self.markout = add(self.markout, later, "markout")?;
        let now = mul(signed, sub(at_fill, fill.price, "capture")?, "capture")?;
        self.capture = add(self.capture, now, "capture")?;
        Ok(())
    }

    fn finish(&self) -> EvalResult<SideQuality> {
        let two = Decimal::TWO;
        let per_unit = |sum: Decimal| -> EvalResult<Option<Decimal>> {
            if self.measured == 0 {
                return Ok(None);
            }
            sum.checked_div(self.volume)
                .map(Some)
                .ok_or(EvaluationError::Inexact {
                    what: "fill quality mean",
                })
        };
        let twice_markout = mul(two, self.markout, "realized spread")?;
        let twice_capture = mul(two, self.capture, "effective spread")?;
        let impact = sub(twice_capture, twice_markout, "price impact")?;
        Ok(SideQuality {
            fills: self.fills,
            measured: self.measured,
            excluded: self.fills.saturating_sub(self.measured),
            measured_volume: self.volume,
            signed_markout: per_unit(self.markout)?,
            realized_spread: per_unit(twice_markout)?,
            effective_spread: per_unit(twice_capture)?,
            price_impact: per_unit(impact)?,
        })
    }
}

/// Measures the fills of a window at one horizon. The formulas are on [`MetricId`].
///
/// # Errors
///
/// Returns [`EvaluationError::InvalidInput`] for a zero horizon or one that
/// overflows, and [`EvaluationError::Inexact`] if a sum or product would round.
pub fn fill_quality(
    input: &EvaluationInput<'_>,
    spec: FillQualitySpec,
) -> EvalResult<FillQualityReport> {
    if spec.horizon_ms == 0 {
        return invalid("a markout horizon must be positive".to_string());
    }
    let (mut all, mut buys, mut sells) = (
        QualitySums::default(),
        QualitySums::default(),
        QualitySums::default(),
    );
    for fill in input.fills {
        let t = fill.timestamp_ms;
        let Some(due) = t.checked_add(spec.horizon_ms) else {
            return invalid(format!("horizon overflows at the fill at {t}"));
        };
        let Some(latest) = due.checked_add(spec.max_mark_delay_ms) else {
            return invalid(format!("mark delay overflows at the fill at {t}"));
        };
        let in_force = input.marks.partition_point(|m| m.timestamp_ms <= t);
        let at_fill = in_force
            .checked_sub(1)
            .and_then(|i| input.marks.get(i))
            .ok_or(EvaluationError::FillBeforeFirstMark { timestamp_ms: t })?
            .mid;
        let next = input.marks.partition_point(|m| m.timestamp_ms < due);
        // Of marks tied at that time the last, as for the mark in force.
        let at_horizon = input
            .marks
            .get(next)
            .filter(|m| m.timestamp_ms <= latest)
            .map(|first| {
                input
                    .marks
                    .partition_point(|m| m.timestamp_ms <= first.timestamp_ms)
            })
            .and_then(|end| end.checked_sub(1))
            .and_then(|last| input.marks.get(last))
            .map(|m| m.mid);
        let side = if fill.signed_quantity()? > Decimal::ZERO {
            Decimal::ONE
        } else {
            Decimal::NEGATIVE_ONE
        };
        all.record(fill, side, at_fill, at_horizon)?;
        if side > Decimal::ZERO {
            buys.record(fill, side, at_fill, at_horizon)?;
        } else {
            sells.record(fill, side, at_fill, at_horizon)?;
        }
    }
    Ok(FillQualityReport {
        spec,
        all: all.finish()?,
        buys: buys.finish()?,
        sells: sells.finish()?,
    })
}

/// What to measure over a window.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricsSpec {
    /// Horizons of the markout and realized spread, in milliseconds.
    pub horizons_ms: Vec<u64>,
    /// See [`FillQualitySpec::max_mark_delay_ms`].
    pub max_mark_delay_ms: u64,
}

/// A reported number, or why there is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Reading {
    /// The value.
    Value(Decimal),
    /// Not defined on this data, such as a mean over no fill.
    Undefined,
    /// Not part of the model.
    NotModelled(HedgeNotModelledReason),
}

/// One reported metric, with its definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct MetricValue {
    /// The metric.
    pub id: MetricId,
    /// What it means.
    pub definition: MetricDefinition,
    /// The horizon, for a metric over the fills measured at one. That includes
    /// [`MetricId::EffectiveSpread`], taken at the fill over those same fills so that
    /// it adds up with the others.
    pub horizon_ms: Option<u64>,
    /// The number.
    pub reading: Reading,
}

impl MetricValue {
    fn new(id: MetricId, horizon_ms: Option<u64>, reading: Reading) -> Self {
        Self {
            id,
            definition: id.definition(),
            horizon_ms,
            reading,
        }
    }
}

/// The metrics of one window.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct WindowMetrics {
    /// The window.
    pub window: TimeWindow,
    /// The capital declared for it.
    pub capital: Decimal,
    /// Where the result came from.
    pub attribution: Attribution,
    /// Number of fills.
    pub fills: u64,
    /// Total size of the fills.
    pub volume: Decimal,
    /// The position over the window.
    pub inventory: InventoryStats,
    /// See [`max_drawdown`].
    pub max_drawdown: Decimal,
    /// Fill quality per horizon.
    pub fill_quality: BTreeMap<u64, FillQualityReport>,
}

impl WindowMetrics {
    /// Every metric of the window with its definition, in the order of
    /// [`MetricId::ALL`], the fill quality ones once per horizon.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Inexact`] if a quotient over the capital
    /// overflows.
    pub fn values(&self) -> EvalResult<Vec<MetricValue>> {
        let over_capital = |value: Decimal| -> EvalResult<Reading> {
            value
                .checked_div(self.capital)
                .map(Reading::Value)
                .ok_or(EvaluationError::Inexact {
                    what: "a quotient over the capital",
                })
        };
        let of = |id, value| MetricValue::new(id, None, Reading::Value(value));
        let hedge = match self.attribution.hedge_costs {
            HedgeCosts::NotModelled { reason } => Reading::NotModelled(reason),
            HedgeCosts::Attributed(hedge) => Reading::Value(hedge.costs()?),
        };
        let mut values = vec![
            of(MetricId::NetPnl, self.attribution.net_pnl),
            of(MetricId::SpreadCapture, self.attribution.spread_capture),
            of(MetricId::InventoryPnl, self.attribution.inventory_pnl),
            of(MetricId::QuoteFees, self.attribution.quote_fees),
            of(MetricId::BaseFees, self.attribution.base_fees),
            MetricValue::new(MetricId::HedgeCosts, None, hedge),
        ];
        let optional = |value: Option<Decimal>| value.map_or(Reading::Undefined, Reading::Value);
        for (id, read) in [
            (
                MetricId::SignedMarkout,
                (|q: &SideQuality| q.signed_markout) as fn(&SideQuality) -> Option<Decimal>,
            ),
            (MetricId::RealizedSpread, |q| q.realized_spread),
            (MetricId::EffectiveSpread, |q| q.effective_spread),
            (MetricId::PriceImpact, |q| q.price_impact),
        ] {
            for (horizon, report) in &self.fill_quality {
                values.push(MetricValue::new(
                    id,
                    Some(*horizon),
                    optional(read(&report.all)),
                ));
            }
        }
        values.push(of(MetricId::Fills, Decimal::from(self.fills)));
        values.push(of(MetricId::Volume, self.volume));
        values.push(MetricValue::new(
            MetricId::MeanAbsInventory,
            None,
            optional(self.inventory.mean_abs),
        ));
        values.push(of(MetricId::MaxAbsInventory, self.inventory.max_abs));
        values.push(of(MetricId::MaxDrawdown, self.max_drawdown));
        values.push(MetricValue::new(
            MetricId::NetReturnOnCapital,
            None,
            over_capital(self.attribution.net_pnl)?,
        ));
        values.push(MetricValue::new(
            MetricId::MaxDrawdownOnCapital,
            None,
            over_capital(self.max_drawdown)?,
        ));
        Ok(values)
    }
}

/// Measures one window.
///
/// # Errors
///
/// As [`attribute`], [`fill_quality`], [`inventory_stats`] and [`max_drawdown`].
pub fn evaluate_window(
    input: &EvaluationInput<'_>,
    spec: &MetricsSpec,
) -> EvalResult<WindowMetrics> {
    let attribution = attribute(input)?.total;
    let mut fill_reports = BTreeMap::new();
    for &horizon_ms in &spec.horizons_ms {
        let report = fill_quality(
            input,
            FillQualitySpec {
                horizon_ms,
                max_mark_delay_ms: spec.max_mark_delay_ms,
            },
        )?;
        fill_reports.insert(horizon_ms, report);
    }
    let mut volume = Decimal::ZERO;
    for fill in input.fills {
        volume = add(volume, fill.quantity, "volume")?;
    }
    Ok(WindowMetrics {
        window: input.window,
        capital: input.capital,
        attribution,
        fills: crate::types::decimal::saturating_u64(input.fills.len()),
        volume,
        inventory: inventory_stats(input)?,
        max_drawdown: max_drawdown(input)?,
        fill_quality: fill_reports,
    })
}

/// Checks that a contract is worth its position times its mark, as the attribution
/// assumes: spot, or a linear future or perpetual of one unit of the underlying.
///
/// # Errors
///
/// Returns [`EvaluationError::UnsupportedContract`] for anything else, such as an
/// inverse contract, an option, or a linear contract with another multiplier.
#[cfg(feature = "instruments")]
pub fn check_contract(spec: &financial_types::instrument::InstrumentSpec) -> EvalResult<()> {
    use financial_types::instrument::{ContractKind, Margining, SizeKind};
    let supported = match spec.kind() {
        ContractKind::Spot => true,
        ContractKind::Future | ContractKind::Perpetual => {
            spec.margining() == Margining::Linear
                && spec.size().kind() == SizeKind::BaseQuantity
                && spec.size().amount() == Decimal::ONE
        }
        _ => false,
    };
    if supported {
        return Ok(());
    }
    Err(EvaluationError::UnsupportedContract {
        reason: format!(
            "{} is {:?}, {:?}, sized {:?} {}",
            spec.id().as_str(),
            spec.kind(),
            spec.margining(),
            spec.size().kind(),
            spec.size().amount()
        ),
    })
}
