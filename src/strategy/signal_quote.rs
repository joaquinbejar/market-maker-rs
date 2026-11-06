//! Causal, bounded quote adjustment from microstructure signals.
//!
//! Composes the inventory quotes of the Avellaneda-Stoikov module with
//! predictive features, and measures how much of the outcome comes from them.
//! The reference for the imbalance effect is Cartea, Donnelly and Jaimungal,
//! *Enhancing Trading Strategies with Order Book Signals* (2018): using the
//! imbalance reduces adverse selection costs. That paper solves a control
//! problem over regimes; this module does not. It is a simple, capped linear
//! rule, to be compared with the same strategy without it.
//!
//! # Composition
//!
//! ```text
//! skew  = clamp(Σ weight · value over imbalance and micro-price, ±max_skew)
//! widen = min(toxicity weight · toxicity, max_widen)
//! bid   = inventory bid + skew - widen, rounded down to the tick
//! ask   = inventory ask + skew + widen, rounded up to the tick
//! size  = base size · (1 - toxicity · (1 - min_size_fraction))
//! ```
//!
//! Then the risk limits scale each side's size, and a side the limits refuse
//! is not quoted. The features are:
//!
//! - imbalance `2 I - 1`, in `[-1, 1]`, from the same quote. Its horizon is the
//!   next mid move, with no fixed duration;
//! - micro-price adjustment over half the spread, in `[-1, 1]`, from a record
//!   whose training ends before the simulated data, which is checked. It has no
//!   horizon by construction: the paper defines it as a limit of expected mids;
//! - toxicity, the VPIN of the trades already seen, in `[0, 1]`. It describes the
//!   configured volume buckets behind it, and predicts no price over a horizon.
//!
//! # Missing and stale values
//!
//! Each enabled feature declares `max_age_ms` and an [`UnavailablePolicy`]. A
//! feature that is absent or older than that applies its policy, and
//! [`FeatureCoverage`] counts how often. A feature timestamped after the
//! decision is an error, not a value.
//!
//! # Simulation
//!
//! [`simulate`] replays quotes and trades in order. A quote decided at `t` rests
//! from `t + latency_ms`. A trade fills a resting quote it reaches: an aggressive
//! sell at or below our bid, an aggressive buy at or above our ask, for up to the
//! trade's size, at our price, with no queue model. So fills are optimistic on
//! priority and their level should not be read as a forecast. The rules are the
//! same for every configuration, so differences between configurations on the
//! same data are what the report is for. The signs of a trade reach the features
//! only after the trade. Events apply in the order given, so to apply a trade
//! before a quote with the same timestamp, list it first.
//!
//! The limits are checked at the decision and again at the fill, against the
//! inventory of that moment, because with latency other fills may come between.
//! A fill is cut to the position limit. An order that does not increase the
//! absolute position is never refused. A side that would cross the quote it was
//! decided on, or the latest book when it arrives, is not posted, as a post-only
//! order, and is counted. A crossed or
//! empty quote, or a trade without a positive price and size, is counted and
//! skipped, the same under every configuration. A micro-price model that falls
//! back to the mid for a quote counts as that feature being missing.
//!
//! Attribution reconciles exactly with the ledger:
//! `total = spread capture + inventory - fees`, with spread capture measured
//! from the last mid at the fill and inventory from later mid changes. The
//! report states results as they come, including negative ones, and says
//! nothing about performance on data other than what it was given.

use financial_types::Action;

use crate::Decimal;
use crate::analytics::micro_price::{MicroPriceModel, MicroPriceRecord, SignalSource, TopOfBook};
use crate::analytics::order_flow::{Trade, TradeSide};
use crate::analytics::vpin::{VPINCalculator, VPINConfig};
use crate::risk::RiskLimits;
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};
use crate::types::fee::FeeAsset;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A predictive feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum FeatureKind {
    /// `2 I - 1` of the top of book.
    Imbalance,
    /// Micro-price adjustment over half the spread.
    MicroPrice,
    /// VPIN of the trades seen.
    Toxicity,
}

/// A feature value at a known time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FeatureValue {
    /// Which feature this is.
    pub kind: FeatureKind,
    /// Normalized value: `[-1, 1]`, or `[0, 1]` for toxicity.
    pub value: Decimal,
    /// Time the value was computed from data, in milliseconds.
    pub as_of_ms: u64,
}

/// What to do when an enabled feature is absent or stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnavailablePolicy {
    /// Treat it as no signal.
    Neutral,
    /// Do not quote.
    StopQuoting,
}

/// Use of one feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct FeaturePolicy {
    /// Price units per unit of the feature. Not negative.
    pub weight: Decimal,
    /// Oldest value used, in milliseconds.
    pub max_age_ms: u64,
    /// Policy when absent or older.
    pub when_unavailable: UnavailablePolicy,
}

/// Which features are on, and the caps of their effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[non_exhaustive]
pub struct SignalConfig {
    /// Imbalance; `None` ablates it.
    pub imbalance: Option<FeaturePolicy>,
    /// Micro-price; `None` ablates it.
    pub micro_price: Option<FeaturePolicy>,
    /// Toxicity; `None` ablates it.
    pub toxicity: Option<FeaturePolicy>,
    /// Largest shift of both quotes, in price units. Not negative.
    pub max_skew: Decimal,
    /// Largest widening per side, in price units. Not negative.
    pub max_widen: Decimal,
    /// Size at toxicity 1, as a fraction of the base size, in `[0, 1]`.
    pub min_size_fraction: Decimal,
}

impl SignalConfig {
    /// Creates a configuration with every feature off, which leaves the
    /// inventory quotes as they are.
    #[must_use]
    pub fn baseline(max_skew: Decimal, max_widen: Decimal, min_size_fraction: Decimal) -> Self {
        Self {
            imbalance: None,
            micro_price: None,
            toxicity: None,
            max_skew,
            max_widen,
            min_size_fraction,
        }
    }

    /// Sets the imbalance feature.
    #[must_use]
    pub fn with_imbalance(mut self, policy: FeaturePolicy) -> Self {
        self.imbalance = Some(policy);
        self
    }

    /// Sets the micro-price feature.
    #[must_use]
    pub fn with_micro_price(mut self, policy: FeaturePolicy) -> Self {
        self.micro_price = Some(policy);
        self
    }

    /// Sets the toxicity feature.
    #[must_use]
    pub fn with_toxicity(mut self, policy: FeaturePolicy) -> Self {
        self.toxicity = Some(policy);
        self
    }

    fn policy(&self, kind: FeatureKind) -> Option<FeaturePolicy> {
        match kind {
            FeatureKind::Imbalance => self.imbalance,
            FeatureKind::MicroPrice => self.micro_price,
            FeatureKind::Toxicity => self.toxicity,
        }
    }

    fn validate(&self) -> MMResult<()> {
        let weights_ok = [self.imbalance, self.micro_price, self.toxicity]
            .iter()
            .flatten()
            .all(|p| p.weight >= Decimal::ZERO);
        if !weights_ok
            || self.max_skew < Decimal::ZERO
            || self.max_widen < Decimal::ZERO
            || self.min_size_fraction < Decimal::ZERO
            || self.min_size_fraction > Decimal::ONE
        {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid signal configuration: {self:?}"
            )));
        }
        Ok(())
    }
}

/// Whether a feature was used at a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum FeatureStatus {
    /// Ablated.
    Off,
    /// Used.
    Fresh,
    /// Older than `max_age_ms`; the policy applied.
    Stale,
    /// Absent; the policy applied.
    Missing,
}

/// The adjustment of a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct Adjustment {
    /// Shift of both quotes.
    pub skew: Decimal,
    /// Widening of each side.
    pub widen: Decimal,
    /// Factor on the base size, in `[min_size_fraction, 1]`.
    pub size_factor: Decimal,
    /// False when a [`UnavailablePolicy::StopQuoting`] applied.
    pub quoting: bool,
    /// Status of the imbalance.
    pub imbalance: FeatureStatus,
    /// Status of the micro-price.
    pub micro_price: FeatureStatus,
    /// Status of the toxicity.
    pub toxicity: FeatureStatus,
}

/// Computes the capped adjustment of `features` at `now_ms`.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid configuration,
/// [`MMError::InvalidMarketState`] for a feature timestamped after `now_ms` or
/// out of its range, and [`MMError::NumericalError`] on overflow.
pub fn compute_adjustment(
    features: &[FeatureValue],
    now_ms: u64,
    config: &SignalConfig,
) -> MMResult<Adjustment> {
    config.validate()?;
    for feature in features {
        let low = match feature.kind {
            FeatureKind::Imbalance | FeatureKind::MicroPrice => Decimal::NEGATIVE_ONE,
            FeatureKind::Toxicity => Decimal::ZERO,
        };
        if feature.as_of_ms > now_ms || feature.value < low || feature.value > Decimal::ONE {
            return Err(MMError::InvalidMarketState(format!(
                "invalid feature at {now_ms}: {feature:?}"
            )));
        }
    }
    let mut quoting = true;
    let mut used = |kind: FeatureKind| -> (FeatureStatus, Decimal) {
        let Some(policy) = config.policy(kind) else {
            return (FeatureStatus::Off, Decimal::ZERO);
        };
        let found = features.iter().find(|f| f.kind == kind);
        let status = match found {
            None => FeatureStatus::Missing,
            Some(f) if now_ms.saturating_sub(f.as_of_ms) > policy.max_age_ms => {
                FeatureStatus::Stale
            }
            Some(_) => FeatureStatus::Fresh,
        };
        match (status, found) {
            (FeatureStatus::Fresh, Some(f)) => (status, f.value),
            _ => {
                match policy.when_unavailable {
                    UnavailablePolicy::Neutral => {}
                    UnavailablePolicy::StopQuoting => quoting = false,
                }
                (status, Decimal::ZERO)
            }
        }
    };
    let (imbalance, imbalance_value) = used(FeatureKind::Imbalance);
    let (micro_price, micro_value) = used(FeatureKind::MicroPrice);
    let (toxicity, toxicity_value) = used(FeatureKind::Toxicity);

    let weighted = |kind: FeatureKind, value: Decimal| -> MMResult<Decimal> {
        let weight = config.policy(kind).map_or(Decimal::ZERO, |p| p.weight);
        weight.checked_mul(value).or_overflow("feature effect")
    };
    let skew = weighted(FeatureKind::Imbalance, imbalance_value)?
        .checked_add(weighted(FeatureKind::MicroPrice, micro_value)?)
        .or_overflow("skew")?;
    let negative_cap = Decimal::ZERO
        .checked_sub(config.max_skew)
        .or_overflow("skew cap")?;
    let skew = skew.clamp(negative_cap, config.max_skew);
    let widen = weighted(FeatureKind::Toxicity, toxicity_value)?.min(config.max_widen);
    let size_factor = Decimal::ONE
        .checked_sub(config.min_size_fraction)
        .and_then(|range| range.checked_mul(toxicity_value))
        .and_then(|cut| Decimal::ONE.checked_sub(cut))
        .or_overflow("size factor")?;
    Ok(Adjustment {
        skew,
        widen,
        size_factor,
        quoting,
        imbalance,
        micro_price,
        toxicity,
    })
}

/// An event of the replay.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MarketEvent {
    /// A top of book update.
    Quote(TopOfBook),
    /// A trade.
    Trade(Trade),
}

impl MarketEvent {
    fn timestamp_ms(&self) -> u64 {
        match self {
            Self::Quote(quote) => quote.timestamp_ms,
            Self::Trade(trade) => trade.timestamp,
        }
    }
}

/// Inventory model, costs and execution rules of a simulation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SimulationConfig {
    /// Risk aversion of the inventory quotes.
    pub risk_aversion: Decimal,
    /// Absolute volatility per √year of the inventory quotes.
    pub volatility: Decimal,
    /// `k` of the inventory quotes.
    pub order_intensity: Decimal,
    /// Terminal time in milliseconds. No quote is decided at or after it.
    pub terminal_ms: u64,
    /// Size per side before the features and the limits. Positive.
    pub base_size: Decimal,
    /// Price grid. Positive.
    pub tick_size: Decimal,
    /// Time from a decision until the quote rests, in milliseconds.
    pub latency_ms: u64,
    /// Fee on the notional of each fill; negative for a rebate.
    pub fee_rate: Decimal,
    /// Horizon of the markout, in milliseconds. Positive.
    pub markout_horizon_ms: u64,
    /// Position limits, applied before quoting.
    pub limits: RiskLimits,
    /// Buckets of the toxicity feature.
    pub vpin: VPINConfig,
}

impl SimulationConfig {
    /// Creates a configuration with no latency, no fees, a markout horizon of
    /// one second and tick 0.01.
    #[must_use]
    pub fn new(
        risk_aversion: Decimal,
        volatility: Decimal,
        order_intensity: Decimal,
        terminal_ms: u64,
        base_size: Decimal,
        limits: RiskLimits,
        vpin: VPINConfig,
    ) -> Self {
        Self {
            risk_aversion,
            volatility,
            order_intensity,
            terminal_ms,
            base_size,
            tick_size: Decimal::new(1, 2),
            latency_ms: 0,
            fee_rate: Decimal::ZERO,
            markout_horizon_ms: 1_000,
            limits,
            vpin,
        }
    }

    /// Sets the time until a quote rests.
    #[must_use]
    pub fn with_latency_ms(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Sets the fee on notional.
    #[must_use]
    pub fn with_fee_rate(mut self, fee_rate: Decimal) -> Self {
        self.fee_rate = fee_rate;
        self
    }

    /// Sets the price grid.
    #[must_use]
    pub fn with_tick_size(mut self, tick_size: Decimal) -> Self {
        self.tick_size = tick_size;
        self
    }

    /// Sets the markout horizon.
    #[must_use]
    pub fn with_markout_horizon_ms(mut self, horizon_ms: u64) -> Self {
        self.markout_horizon_ms = horizon_ms;
        self
    }

    fn validate(&self) -> MMResult<()> {
        if self.base_size <= Decimal::ZERO
            || self.tick_size <= Decimal::ZERO
            || self.markout_horizon_ms == 0
        {
            return Err(MMError::InvalidConfiguration(format!(
                "invalid simulation configuration: base {}, tick {}, horizon {}",
                self.base_size, self.tick_size, self.markout_horizon_ms
            )));
        }
        Ok(())
    }
}

/// How often one feature was available at a decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct FeatureCoverage {
    /// Decisions that used it.
    pub fresh: u64,
    /// Decisions where it was stale.
    pub stale: u64,
    /// Decisions where it was absent.
    pub missing: u64,
}

impl FeatureCoverage {
    fn add(&mut self, status: FeatureStatus) {
        match status {
            FeatureStatus::Off => {}
            FeatureStatus::Fresh => self.fresh = self.fresh.saturating_add(1),
            FeatureStatus::Stale => self.stale = self.stale.saturating_add(1),
            FeatureStatus::Missing => self.missing = self.missing.saturating_add(1),
        }
    }
}

/// Result of a simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct SimulationReport {
    /// Quotes decided.
    pub decisions: u64,
    /// Decisions with no quote: a stop policy, the limits on both sides, inventory quotes
    /// that could not be formed, or a gate that took both sides off.
    pub halted_decisions: u64,
    /// Quotes skipped as crossed or empty.
    pub invalid_quotes: u64,
    /// Trades skipped for a price or size that is not positive.
    pub invalid_trades: u64,
    /// Sides not posted because they would cross the book, at the decision or on
    /// arrival.
    pub crossing_sides: u64,
    /// Sides the policy of [`simulate_gated`] reduced. Zero without one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gate_reductions: u64,
    /// Sides the policy of [`simulate_gated`] rejected, or withheld after it failed on an
    /// event. Zero without one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gate_rejections: u64,
    /// Lifecycle events the policy of [`simulate_gated`] could not account for. After the
    /// first, no new side is sent.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gate_faults: u64,
    /// Fills.
    pub fills: u64,
    /// Volume filled.
    pub volume: Decimal,
    /// Inventory at the end.
    pub final_inventory: Decimal,
    /// Largest absolute inventory reached.
    pub max_abs_inventory: Decimal,
    /// Earned against the last mid at each fill.
    pub spread_capture: Decimal,
    /// Inventory times later mid changes.
    pub inventory_pnl: Decimal,
    /// Fees paid; negative for rebates.
    pub fees: Decimal,
    /// `spread_capture + inventory_pnl - fees`, equal to cash plus marked
    /// inventory minus fees.
    pub total_pnl: Decimal,
    /// Size weighted mean of `side · (mid at horizon - fill price)` per unit;
    /// `None` without a fill having a mid at its horizon.
    pub markout: Option<Decimal>,
    /// Coverage of the imbalance.
    pub imbalance: FeatureCoverage,
    /// Coverage of the micro-price.
    pub micro_price: FeatureCoverage,
    /// Coverage of the toxicity.
    pub toxicity: FeatureCoverage,
}

/// A simulated fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct FillRecord {
    /// Time of the trade, in milliseconds.
    pub timestamp_ms: u64,
    /// Our side.
    pub side: Action,
    /// Our price.
    pub price: Decimal,
    /// Quantity filled.
    pub size: Decimal,
}

/// A quote decided by the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct DecisionRecord {
    /// Time of the decision, in milliseconds.
    pub timestamp_ms: u64,
    /// The adjustment applied.
    pub adjustment: Adjustment,
    /// Price and size of the bid, if any.
    pub bid: Option<(Decimal, Decimal)>,
    /// Price and size of the ask, if any.
    pub ask: Option<(Decimal, Decimal)>,
}

/// A simulation in detail, for audit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct SimulationRun {
    /// The totals.
    pub report: SimulationReport,
    /// Every decision, in order.
    pub decisions: Vec<DecisionRecord>,
    /// Every fill, in order.
    pub fills: Vec<FillRecord>,
}

/// A two-sided quote and when it rests.
#[derive(Clone, Copy)]
struct Resting {
    from_ms: u64,
    bid: Option<(Decimal, Decimal)>,
    ask: Option<(Decimal, Decimal)>,
    /// The decision's number, which names its orders to a gate.
    order: u64,
    bid_filled: Decimal,
    ask_filled: Decimal,
}

impl Resting {
    fn new(from_ms: u64, bid: Option<(Decimal, Decimal)>, ask: Option<(Decimal, Decimal)>) -> Self {
        Self {
            from_ms,
            bid,
            ask,
            order: 0,
            bid_filled: Decimal::ZERO,
            ask_filled: Decimal::ZERO,
        }
    }
}

fn order_id(order: u64, side: Action) -> String {
    let name = if matches!(side, Action::Buy) {
        "bid"
    } else {
        "ask"
    };
    format!("sq-{order}-{name}")
}

/// A policy and what the simulation told it.
struct Gate<'a> {
    policy: &'a mut dyn PretradePolicy,
    next_order: u64,
    next_execution: u64,
    faulted: bool,
}

impl Gate<'_> {
    fn event(&mut self, event: &OrderEvent, report: &mut SimulationReport) {
        if self.policy.on_order_event(event).is_err() {
            self.faulted = true;
            report.gate_faults = report.gate_faults.saturating_add(1);
        }
    }

    /// Puts the sides of a decision to the policy.
    fn admit(&mut self, resting: &mut Resting, now: u64, report: &mut SimulationReport) {
        resting.order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        for (side, level) in [
            (Action::Buy, &mut resting.bid),
            (Action::Sell, &mut resting.ask),
        ] {
            let Some((price, size)) = *level else {
                continue;
            };
            if self.faulted {
                *level = None;
                report.gate_rejections = report.gate_rejections.saturating_add(1);
                continue;
            }
            let request = PretradeRequest::new(
                order_id(resting.order, side),
                side,
                Some(price),
                size,
                OrderPurpose::Quote,
                now,
            );
            match self.policy.check(&request) {
                PretradeDecision::Accepted => {}
                PretradeDecision::Reduced { quantity, .. } if quantity > Decimal::ZERO => {
                    *level = Some((price, quantity.min(size)));
                    report.gate_reductions = report.gate_reductions.saturating_add(1);
                }
                PretradeDecision::Reduced { .. } | PretradeDecision::Rejected { .. } => {
                    *level = None;
                    report.gate_rejections = report.gate_rejections.saturating_add(1);
                }
            }
        }
    }

    /// Orders of a quote that will not rest any longer.
    fn cancel(&mut self, resting: &Resting, report: &mut SimulationReport) {
        for (side, level) in [(Action::Buy, resting.bid), (Action::Sell, resting.ask)] {
            if level.is_some() {
                let order_id = order_id(resting.order, side);
                self.event(&OrderEvent::Canceled { order_id }, report);
            }
        }
    }

    fn fill(
        &mut self,
        resting: &mut Resting,
        fill: &FillRecord,
        fee_rate: Decimal,
        report: &mut SimulationReport,
    ) -> MMResult<()> {
        let filled = if matches!(fill.side, Action::Buy) {
            &mut resting.bid_filled
        } else {
            &mut resting.ask_filled
        };
        *filled = filled.checked_add(fill.size).or_overflow("order filled")?;
        let fee = fill
            .price
            .checked_mul(fill.size)
            .and_then(|notional| notional.checked_mul(fee_rate))
            .or_overflow("fee")?;
        let event = OrderEvent::Fill {
            order_id: order_id(resting.order, fill.side),
            execution_id: format!("sq-fill-{}", self.next_execution),
            side: fill.side,
            price: fill.price,
            quantity: fill.size,
            cumulative_quantity: *filled,
            fee,
            fee_asset: FeeAsset::Quote,
            timestamp: fill.timestamp_ms,
        };
        self.next_execution = self.next_execution.saturating_add(1);
        self.event(&event, report);
        Ok(())
    }
}

/// Replays `events` under `signals`.
///
/// `record` provides the micro-price feature, and its training must end by the
/// first event. Without a model, that feature is absent.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] for an invalid configuration or a
/// record trained on the events,
/// [`MMError::InvalidMarketState`] for events out of order or an invalid quote,
/// and [`MMError::NumericalError`] on arithmetic failure.
pub fn simulate(
    events: &[MarketEvent],
    config: &SimulationConfig,
    signals: &SignalConfig,
    record: Option<&MicroPriceRecord>,
) -> MMResult<SimulationReport> {
    simulate_detailed(events, config, signals, record).map(|run| run.report)
}

/// Replays `events` like [`simulate_detailed`], with `policy` as a pretrade gate after
/// the raw limits of `config`, which still apply.
///
/// The policy is reset first. It then sees each mid, each side of each decision before it
/// can rest, and each side's life: accepted on arrival, rejected if it would cross by
/// then, filled, or cancelled when replaced or at the terminal time and the end of the
/// events. A quote is sent while the last one still rests until its replacement arrives,
/// so a policy counting live orders counts both. A reduced side rests for the smaller size,
/// and a rejected one does not rest. If the policy fails on an event, no new side is sent,
/// and the events still reach it. That is not a stop: what is already out or in flight can
/// still rest and fill until a later decision replaces it. See
/// [`SimulationReport::gate_rejections`].
///
/// # Errors
///
/// As [`simulate`].
pub fn simulate_gated(
    events: &[MarketEvent],
    config: &SimulationConfig,
    signals: &SignalConfig,
    record: Option<&MicroPriceRecord>,
    policy: &mut dyn PretradePolicy,
) -> MMResult<SimulationRun> {
    policy.reset();
    let gate = Gate {
        policy,
        next_order: 0,
        next_execution: 0,
        faulted: false,
    };
    replay(events, config, signals, record, Some(gate))
}

/// Replays `events` like [`simulate`], and also returns every decision and fill.
///
/// # Errors
///
/// As [`simulate`].
pub fn simulate_detailed(
    events: &[MarketEvent],
    config: &SimulationConfig,
    signals: &SignalConfig,
    record: Option<&MicroPriceRecord>,
) -> MMResult<SimulationRun> {
    replay(events, config, signals, record, None)
}

fn replay(
    events: &[MarketEvent],
    config: &SimulationConfig,
    signals: &SignalConfig,
    record: Option<&MicroPriceRecord>,
    mut gate: Option<Gate<'_>>,
) -> MMResult<SimulationRun> {
    let model = record.and_then(MicroPriceRecord::model);
    if let (Some(record), Some(first)) = (record, events.first())
        && first.timestamp_ms() < record.training.end_ms
    {
        return Err(MMError::InvalidConfiguration(format!(
            "events start at {} inside the training of {:?}",
            first.timestamp_ms(),
            record.training
        )));
    }
    config.validate()?;
    signals.validate()?;
    if !events.is_sorted_by_key(MarketEvent::timestamp_ms) {
        return Err(MMError::InvalidMarketState(
            "events are not in chronological order".to_string(),
        ));
    }

    let mut vpin = VPINCalculator::new(config.vpin.clone());
    let mut last_trade_ms = None;
    let mut last_mid: Option<Decimal> = None;
    let mut book: Option<(Decimal, Decimal)> = None;
    let mut inventory = Decimal::ZERO;
    let mut pending: Vec<Resting> = Vec::new();
    let mut live: Option<Resting> = None;
    let mut fills: Vec<FillRecord> = Vec::new();
    let mut decisions: Vec<DecisionRecord> = Vec::new();
    let mut mids: Vec<(u64, Decimal)> = Vec::new();
    let mut report = SimulationReport {
        decisions: 0,
        halted_decisions: 0,
        invalid_quotes: 0,
        invalid_trades: 0,
        crossing_sides: 0,
        gate_reductions: 0,
        gate_rejections: 0,
        gate_faults: 0,
        fills: 0,
        volume: Decimal::ZERO,
        final_inventory: Decimal::ZERO,
        max_abs_inventory: Decimal::ZERO,
        spread_capture: Decimal::ZERO,
        inventory_pnl: Decimal::ZERO,
        fees: Decimal::ZERO,
        total_pnl: Decimal::ZERO,
        markout: None,
        imbalance: FeatureCoverage::default(),
        micro_price: FeatureCoverage::default(),
        toxicity: FeatureCoverage::default(),
    };

    for event in events {
        let now = event.timestamp_ms();
        if now >= config.terminal_ms {
            pull_all(&mut pending, &mut live, &mut gate, &mut report);
        }
        activate(&mut pending, &mut live, now, book, &mut gate, &mut report);
        match event {
            MarketEvent::Trade(trade) => {
                if trade.price <= Decimal::ZERO || trade.size <= Decimal::ZERO {
                    report.invalid_trades = report.invalid_trades.saturating_add(1);
                    continue;
                }
                if let (Some(resting), Some(mid)) = (live.as_mut(), last_mid) {
                    let before = fills.len();
                    apply_trade(
                        trade,
                        resting,
                        mid,
                        config,
                        &mut inventory,
                        &mut report,
                        &mut fills,
                    )?;
                    if let (Some(gate), Some(fill)) = (gate.as_mut(), fills.get(before)) {
                        gate.fill(resting, fill, config.fee_rate, &mut report)?;
                    }
                }
                vpin.add_trade(trade);
                last_trade_ms = Some(trade.timestamp);
            }
            MarketEvent::Quote(quote) => {
                match quote.imbalance() {
                    Ok(_) => {}
                    Err(MMError::InvalidMarketState(_)) => {
                        report.invalid_quotes = report.invalid_quotes.saturating_add(1);
                        continue;
                    }
                    Err(error) => return Err(error),
                }
                let mid = quote.mid()?;
                if let Some(previous) = last_mid {
                    let change = mid.checked_sub(previous).or_overflow("mid change")?;
                    let gain = inventory.checked_mul(change).or_overflow("inventory pnl")?;
                    report.inventory_pnl = report
                        .inventory_pnl
                        .checked_add(gain)
                        .or_overflow("inventory pnl")?;
                }
                last_mid = Some(mid);
                if let Some(gate) = gate.as_mut() {
                    gate.policy.on_market_data(mid, now);
                }
                book = Some((quote.bid, quote.ask));
                mids.push((now, mid));
                if now >= config.terminal_ms {
                    continue;
                }
                let features = features_at(quote, model, &vpin, last_trade_ms, signals)?;
                let adjustment = compute_adjustment(&features, now, signals)?;
                report.decisions = report.decisions.saturating_add(1);
                report.imbalance.add(adjustment.imbalance);
                report.micro_price.add(adjustment.micro_price);
                report.toxicity.add(adjustment.toxicity);
                let mut resting = decide(quote, mid, inventory, &adjustment, config, &mut report)?;
                if let Some(gate) = gate.as_mut() {
                    gate.admit(&mut resting, now, &mut report);
                }
                if resting.bid.is_none() && resting.ask.is_none() {
                    report.halted_decisions = report.halted_decisions.saturating_add(1);
                }
                decisions.push(DecisionRecord {
                    timestamp_ms: now,
                    adjustment,
                    bid: resting.bid,
                    ask: resting.ask,
                });
                pending.push(resting);
                activate(&mut pending, &mut live, now, book, &mut gate, &mut report);
            }
        }
    }

    pull_all(&mut pending, &mut live, &mut gate, &mut report);
    report.final_inventory = inventory;
    report.total_pnl = report
        .spread_capture
        .checked_add(report.inventory_pnl)
        .and_then(|gross| gross.checked_sub(report.fees))
        .or_overflow("total pnl")?;
    report.markout = markout(&fills, &mids, config.markout_horizon_ms)?;
    Ok(SimulationRun {
        report,
        decisions,
        fills,
    })
}

/// Replaces the live quote with the newest one whose time has come, without a
/// side that now crosses the latest book, as post-only. A quote already resting
/// is not touched when the book moves.
fn activate(
    pending: &mut Vec<Resting>,
    live: &mut Option<Resting>,
    now: u64,
    book: Option<(Decimal, Decimal)>,
    gate: &mut Option<Gate<'_>>,
    report: &mut SimulationReport,
) {
    let due = pending.partition_point(|r| r.from_ms <= now);
    if due == 0 {
        return;
    }
    let mut arrived: Vec<Resting> = pending.drain(..due).collect();
    let mut newest = arrived.pop();
    if let Some(gate) = gate.as_mut() {
        // Those replaced before they rested, and the one replaced now.
        for resting in arrived.iter().chain(live.iter()) {
            gate.cancel(resting, report);
        }
    }
    if let Some(resting) = newest.as_mut() {
        let order = resting.order;
        for (side, level) in [
            (Action::Buy, &mut resting.bid),
            (Action::Sell, &mut resting.ask),
        ] {
            let Some((price, _)) = *level else {
                continue;
            };
            let crosses = book.is_some_and(|(best_bid, best_ask)| {
                if matches!(side, Action::Buy) {
                    price >= best_ask
                } else {
                    price <= best_bid
                }
            });
            if crosses {
                *level = None;
                report.crossing_sides = report.crossing_sides.saturating_add(1);
                if let Some(gate) = gate.as_mut() {
                    let order_id = order_id(order, side);
                    gate.event(&OrderEvent::VenueRejected { order_id }, report);
                }
            } else if let Some(gate) = gate.as_mut() {
                let order_id = order_id(order, side);
                gate.event(&OrderEvent::Accepted { order_id }, report);
            }
        }
    }
    *live = newest;
}

/// Takes every quote off, at the terminal time and at the end of the events.
fn pull_all(
    pending: &mut Vec<Resting>,
    live: &mut Option<Resting>,
    gate: &mut Option<Gate<'_>>,
    report: &mut SimulationReport,
) {
    if let Some(gate) = gate.as_mut() {
        for resting in pending.iter().chain(live.iter()) {
            gate.cancel(resting, report);
        }
    }
    pending.clear();
    *live = None;
}

fn features_at(
    quote: &TopOfBook,
    model: Option<&MicroPriceModel>,
    vpin: &VPINCalculator,
    last_trade_ms: Option<u64>,
    signals: &SignalConfig,
) -> MMResult<Vec<FeatureValue>> {
    let mut features = Vec::with_capacity(3);
    let now = quote.timestamp_ms;
    if signals.imbalance.is_some() {
        let value = quote
            .imbalance()?
            .checked_mul(Decimal::TWO)
            .and_then(|v| v.checked_sub(Decimal::ONE))
            .or_overflow("imbalance feature")?;
        features.push(FeatureValue {
            kind: FeatureKind::Imbalance,
            value,
            as_of_ms: now,
        });
    }
    if let (true, Some(model)) = (signals.micro_price.is_some(), model) {
        let signal = match model.signal(quote) {
            Ok(signal) => Some(signal),
            Err(MMError::InvalidMarketState(_)) => None,
            Err(error) => return Err(error),
        };
        if let Some(signal) = signal.filter(|s| matches!(s.source, SignalSource::MicroPrice)) {
            let half_spread = quote
                .ask
                .checked_sub(quote.bid)
                .and_then(|spread| spread.checked_div(Decimal::TWO))
                .or_overflow("half spread")?;
            let value = signal
                .adjustment
                .checked_div(half_spread)
                .or_overflow("micro-price feature")?;
            features.push(FeatureValue {
                kind: FeatureKind::MicroPrice,
                value,
                as_of_ms: now,
            });
        }
    }
    if signals.toxicity.is_some()
        && let (Some(value), Some(as_of_ms)) = (vpin.get_vpin(), last_trade_ms)
    {
        features.push(FeatureValue {
            kind: FeatureKind::Toxicity,
            value,
            as_of_ms,
        });
    }
    Ok(features)
}

/// The quote for a decision, under the risk limits.
fn decide(
    quote: &TopOfBook,
    mid: Decimal,
    inventory: Decimal,
    adjustment: &Adjustment,
    config: &SimulationConfig,
    report: &mut SimulationReport,
) -> MMResult<Resting> {
    let now = quote.timestamp_ms;
    let from_ms = now
        .checked_add(config.latency_ms)
        .or_overflow("quote time")?;
    if !adjustment.quoting {
        return Ok(Resting::new(from_ms, None, None));
    }
    let remaining = config
        .terminal_ms
        .checked_sub(now)
        .or_overflow("time to terminal")?;
    let quotes = calculate_optimal_quotes(
        mid,
        inventory,
        config.risk_aversion,
        config.volatility,
        remaining,
        config.order_intensity,
    );
    let (bid, ask) = match quotes {
        Ok(quotes) => quotes,
        Err(MMError::InvalidQuoteGeneration(_)) => {
            return Ok(Resting::new(from_ms, None, None));
        }
        Err(error) => return Err(error),
    };
    let shift = |price: Decimal, by: Decimal| price.checked_add(by).or_overflow("quote");
    let bid = round_to_tick(
        shift(
            shift(bid, adjustment.skew)?,
            Decimal::ZERO
                .checked_sub(adjustment.widen)
                .or_overflow("widen")?,
        )?,
        config.tick_size,
        false,
    )?;
    let ask = round_to_tick(
        shift(shift(ask, adjustment.skew)?, adjustment.widen)?,
        config.tick_size,
        true,
    )?;
    let size = config
        .base_size
        .checked_mul(adjustment.size_factor)
        .or_overflow("quote size")?;
    let side_size = |buys: bool, price: Decimal| -> MMResult<Option<(Decimal, Decimal)>> {
        if price <= Decimal::ZERO {
            return Ok(None);
        }
        // The limits scale only a fill that moves the position away from zero.
        let increases = inventory.is_zero() || (inventory > Decimal::ZERO) == buys;
        let scaled = if increases {
            config.limits.scale_order_size(inventory, size)
        } else {
            size
        };
        let signed = if buys {
            scaled
        } else {
            Decimal::ZERO
                .checked_sub(scaled)
                .or_overflow("order size")?
        };
        if scaled > Decimal::ZERO && allowed(config, inventory, signed, price)? {
            Ok(Some((price, scaled)))
        } else {
            Ok(None)
        }
    };
    let mut crossing = |crosses: bool| -> bool {
        if crosses {
            report.crossing_sides = report.crossing_sides.saturating_add(1);
        }
        crosses
    };
    let bid_side = if crossing(bid >= quote.ask) {
        None
    } else {
        side_size(true, bid)?
    };
    let ask_side = if crossing(ask <= quote.bid) {
        None
    } else {
        side_size(false, ask)?
    };
    Ok(Resting::new(from_ms, bid_side, ask_side))
}

/// The limits, for an order that increases the absolute position.
fn allowed(
    config: &SimulationConfig,
    inventory: Decimal,
    signed: Decimal,
    price: Decimal,
) -> MMResult<bool> {
    let after = inventory.checked_add(signed).or_overflow("position")?;
    if after.abs() <= inventory.abs() {
        return Ok(true);
    }
    config.limits.check_order(inventory, signed, price)
}

fn round_to_tick(price: Decimal, tick: Decimal, up: bool) -> MMResult<Decimal> {
    let ticks = price.checked_div(tick).or_overflow("quote in ticks")?;
    let whole = if up { ticks.ceil() } else { ticks.floor() };
    whole.checked_mul(tick).or_overflow("rounded quote")
}

/// Fills the side a trade reaches.
fn apply_trade(
    trade: &Trade,
    resting: &mut Resting,
    mid: Decimal,
    config: &SimulationConfig,
    inventory: &mut Decimal,
    report: &mut SimulationReport,
    fills: &mut Vec<FillRecord>,
) -> MMResult<()> {
    let (action, side, level) = match trade.side {
        TradeSide::Sell => (Action::Buy, Decimal::ONE, &mut resting.bid),
        TradeSide::Buy => (Action::Sell, Decimal::NEGATIVE_ONE, &mut resting.ask),
    };
    let Some((price, available)) = *level else {
        return Ok(());
    };
    let reaches = match trade.side {
        TradeSide::Sell => trade.price <= price,
        TradeSide::Buy => trade.price >= price,
    };
    if !reaches || trade.size <= Decimal::ZERO {
        return Ok(());
    }
    // The room to the position limit in this direction, at this moment.
    let room = side
        .checked_mul(*inventory)
        .and_then(|along| config.limits.max_position.checked_sub(along))
        .or_overflow("position room")?;
    let size = available.min(trade.size).min(room);
    if size <= Decimal::ZERO {
        return Ok(());
    }
    let signed = side.checked_mul(size).or_overflow("fill")?;
    if !allowed(config, *inventory, signed, price)? {
        return Ok(());
    }
    let rest = available.checked_sub(size).or_overflow("resting size")?;
    *level = (rest > Decimal::ZERO).then_some((price, rest));

    *inventory = side
        .checked_mul(size)
        .and_then(|signed| inventory.checked_add(signed))
        .or_overflow("inventory")?;
    report.max_abs_inventory = report.max_abs_inventory.max(inventory.abs());
    report.fills = report.fills.saturating_add(1);
    report.volume = report.volume.checked_add(size).or_overflow("volume")?;
    let capture = mid
        .checked_sub(price)
        .and_then(|edge| edge.checked_mul(side))
        .and_then(|edge| edge.checked_mul(size))
        .or_overflow("spread capture")?;
    report.spread_capture = report
        .spread_capture
        .checked_add(capture)
        .or_overflow("spread capture")?;
    let fee = price
        .checked_mul(size)
        .and_then(|notional| notional.checked_mul(config.fee_rate))
        .or_overflow("fee")?;
    report.fees = report.fees.checked_add(fee).or_overflow("fees")?;
    fills.push(FillRecord {
        timestamp_ms: trade.timestamp,
        side: action,
        price,
        size,
    });
    Ok(())
}

/// Size weighted markout, from the last mid at or before each fill's horizon,
/// for the fills whose horizon lies within the mids seen.
fn markout(
    fills: &[FillRecord],
    mids: &[(u64, Decimal)],
    horizon_ms: u64,
) -> MMResult<Option<Decimal>> {
    let Some(last_time) = mids.last().map(|(t, _)| *t) else {
        return Ok(None);
    };
    let mut gained = Decimal::ZERO;
    let mut volume = Decimal::ZERO;
    for fill in fills {
        let target = fill
            .timestamp_ms
            .checked_add(horizon_ms)
            .or_overflow("markout time")?;
        if target > last_time {
            continue;
        }
        let after = mids.partition_point(|(t, _)| *t <= target);
        let Some((_, mid)) = after.checked_sub(1).and_then(|i| mids.get(i)) else {
            continue;
        };
        // `Action` is non-exhaustive, and this module only ever records these two.
        let sign = match fill.side {
            Action::Buy => Decimal::ONE,
            Action::Sell => Decimal::NEGATIVE_ONE,
            other => {
                return Err(MMError::InvalidMarketState(format!(
                    "unsupported fill side {other:?}"
                )));
            }
        };
        let gain = mid
            .checked_sub(fill.price)
            .and_then(|edge| edge.checked_mul(sign))
            .and_then(|edge| edge.checked_mul(fill.size))
            .or_overflow("markout")?;
        gained = gained.checked_add(gain).or_overflow("markout")?;
        volume = volume.checked_add(fill.size).or_overflow("markout")?;
    }
    if volume.is_zero() {
        return Ok(None);
    }
    gained.checked_div(volume).or_overflow("markout").map(Some)
}
