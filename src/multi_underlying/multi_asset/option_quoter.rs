//! Joint quotes for option contracts on one underlying, under one capital budget.
//!
//! [`OptionQuoter`] prices with an [`OptionPricer`] and computes no option value of its own. A
//! contract's delta and vega turn a declared covariance of the underlying and its volatility
//! into a covariance between contracts, and that goes through the same [`MultiAssetModel`] as
//! the linear quotes, with the underlying as one more asset: a hedged book is skewed less than
//! a concentrated one of the same size.
//!
//! # What this approximates
//!
//! The sensitivities are frozen for the round, so the covariance is a first order local one.
//! The closed form assumes a constant covariance, which an option's is not. Gamma enters the
//! limits and the scenarios, but not the skew. Results are model outputs, not a claim about an
//! option market.
//!
//! # Capital is a budget, not a venue's margin
//!
//! `margin-rs` has no option instrument, so these quotes cannot pass a `RiskGate`. Each one
//! holds a reservation in the ledger instead, and a quote exists only if the ledger accepted it:
//!
//! - A buy reserves its premium.
//! - A sell reserves the worst scenario value of the contract less its price, over the declared
//!   scenarios, except for the part that only unwinds a long.
//!
//! The scenario figure is an internal budget. It says nothing about what a venue would demand,
//! it does not reach beyond the declared scenarios, and contracts are summed without netting, so
//! a spread holds more than its real worst case.
//!
//! # Limits
//!
//! One underlying, contracts in units of it, and the option kinds the pricer accepts. Premiums
//! are booked as settled PnL under the contract's name. Expired contracts are the caller's to
//! settle before the next round.

use std::fmt;
use std::sync::Arc;

use margin_rs::ledger::{DerivativeSettlement, Ledger};
use margin_rs::{
    Asset, ExecutionId, Instrument, InstrumentId, MarginError, OrderId, ReservationId,
};
use optionstratlib::OptionStyle;
use rust_decimal::RoundingStrategy;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

use super::model::{AssetQuotingParams, MAX_ASSETS, MultiAssetModel, QuotingObjective, RiskView};
use crate::Decimal;
use crate::options::market_maker::{GreeksLimits, HedgeOrder};
use crate::options::pricing::{
    OptionInputs, OptionPricer, OptionValuation, PRICING_YEAR_DAYS, tau_from_days,
};
use crate::types::action::{Action, Dir};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

/// Milliseconds in a day.
const DAY_MS: Decimal = Decimal::from_parts(86_400_000, 0, 0, false, 0);

/// Decimal places kept of a derived correlation: well inside what the model's check tolerates,
/// so a book with fewer factors than contracts stays positive semidefinite after rounding.
const CORRELATION_DP: u32 = 14;

/// The terms of one contract and how it is quoted. A quantity is in units of the underlying, as
/// [`OptionValuation`] is.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractTerms {
    /// Its name, which premiums are booked under.
    pub name: String,
    /// Strike.
    pub strike: Decimal,
    /// Expiry, in ms.
    pub expiry_ms: u64,
    /// Call or put.
    pub style: OptionStyle,
    /// Decimal places of a price.
    pub price_scale: u32,
    /// Decimal places of a quantity.
    pub quantity_scale: u32,
    /// `A` of `Lambda(delta) = A exp(-k delta)`, per time unit.
    pub intensity_scale: Decimal,
    /// `k` of that intensity, per unit of price.
    pub intensity_decay: Decimal,
    /// Size of a quote.
    pub trade_size: Decimal,
    /// Largest absolute position a quote may lead to.
    pub inventory_limit: Decimal,
}

/// The spot instrument that hedges the contracts. It is an asset of the joint model and is not
/// quoted here, so its flow parameters enter only through the risk matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct UnderlyingTerms {
    /// A spot instrument of the ledger, quoted in the cash asset.
    pub instrument: InstrumentId,
    /// `A` of its intensity.
    pub intensity_scale: Decimal,
    /// `k` of its intensity.
    pub intensity_decay: Decimal,
    /// Its trade size.
    pub trade_size: Decimal,
    /// Its inventory limit in the model.
    pub inventory_limit: Decimal,
}

/// The declared covariance of the two factors, per time unit of the intensities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorRisk {
    /// Volatility of the underlying, in price per square root of the time unit.
    pub spot_volatility: Decimal,
    /// Volatility of a parallel shift of implied volatility, in the units of
    /// [`OptionValuation::vega`], per square root of the time unit.
    pub volatility_of_volatility: Decimal,
    /// Their correlation.
    pub correlation: Decimal,
}

/// One declared scenario, repriced in full: a spot move and the pricer of the shocked surface.
#[derive(Clone)]
pub struct RiskScenario {
    /// Its name, in the log.
    pub name: String,
    /// The spot in the scenario, as a multiple of the round's.
    pub spot_factor: Decimal,
    /// The pricer of the scenario, usually on a `ShockedSurface`.
    pub pricer: Arc<dyn OptionPricer>,
}

impl fmt::Debug for RiskScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiskScenario")
            .field("name", &self.name)
            .field("spot_factor", &self.spot_factor)
            .finish_non_exhaustive()
    }
}

/// What the quoter is built from.
#[derive(Clone)]
pub struct OptionQuoterConfig {
    /// The hedge instrument.
    pub underlying: UnderlyingTerms,
    /// The contracts quoted.
    pub contracts: Vec<ContractTerms>,
    /// The asset of premiums, reservations and the budget.
    pub cash_asset: Asset,
    /// The pricer of the round's market.
    pub pricer: Arc<dyn OptionPricer>,
    /// The declared factor covariance.
    pub factors: FactorRisk,
    /// `gamma` of the joint model.
    pub risk_aversion: Decimal,
    /// The objective of the joint model.
    pub objective: QuotingObjective,
    /// Whether the joint model sees the correlations.
    pub risk_view: RiskView,
    /// The budget on [`OptionCapital::committed`].
    pub capital_budget: Decimal,
    /// The scenarios behind a sale's capital and the loss limit. At least one.
    pub scenarios: Vec<RiskScenario>,
}

impl fmt::Debug for OptionQuoterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptionQuoterConfig")
            .field("underlying", &self.underlying)
            .field("contracts", &self.contracts)
            .field("cash_asset", &self.cash_asset)
            .field("factors", &self.factors)
            .field("risk_aversion", &self.risk_aversion)
            .field("objective", &self.objective)
            .field("risk_view", &self.risk_view)
            .field("capital_budget", &self.capital_budget)
            .field("scenarios", &self.scenarios)
            .finish_non_exhaustive()
    }
}

/// The market of one round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionMarket {
    /// Spot of the underlying.
    pub spot: Decimal,
    /// Risk-free rate, as the pricer takes it.
    pub rate: Decimal,
    /// Carry, as the pricer takes it.
    pub carry: Decimal,
    /// The time of the round, in ms.
    pub now_ms: u64,
}

/// A Greek under a limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitedGreek {
    /// Absolute delta, with the hedge position.
    Delta,
    /// Absolute gamma.
    Gamma,
    /// Absolute vega.
    Vega,
    /// Theta per day, from below.
    Theta,
}

/// Why a stage changed or withheld a side.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptionConstraintKind {
    /// The contract has expired; the side is withheld.
    Expired,
    /// The pricer refused the contract in the market or a scenario; the side is withheld. Only
    /// a contract with no position and no live order reaches this, as the round fails otherwise.
    NotValued {
        /// The pricer's error.
        reason: String,
    },
    /// The contract's delta and vega carry no risk under the declared factors, so the model has
    /// nothing to quote from; the side is withheld.
    NoSensitivity,
    /// The model's price is not positive on the contract's grid; the side is withheld.
    NonPositivePrice {
        /// The price after rounding.
        price: Decimal,
    },
    /// The quote stands on the far side of the pricer's value. Reported, not withheld.
    ThroughValue {
        /// The quote price.
        price: Decimal,
        /// The pricer's value.
        value: Decimal,
    },
    /// With the live orders of this side filled, a trade would pass the inventory limit; the
    /// side is withheld.
    InventoryLimit {
        /// The worst case position after the trade.
        resulting: Decimal,
        /// The limit on its absolute value.
        limit: Decimal,
    },
    /// Some way for the live orders and this side to fill could pass a Greek limit, further than
    /// without the side; the side is withheld.
    GreekLimit {
        /// The Greek.
        greek: LimitedGreek,
        /// The worst value over those fills.
        value: Decimal,
        /// The limit.
        limit: Decimal,
    },
    /// Some way for the live orders and this side to fill could pass the loss limit in a
    /// scenario, further than without the side; the side is withheld.
    ScenarioLoss {
        /// The scenario.
        scenario: String,
        /// The worst loss over those fills.
        loss: Decimal,
        /// The limit.
        limit: Decimal,
    },
    /// The shared budget cut the quantity, to zero when the side is withheld.
    CapitalBudget {
        /// Quantity asked of this stage.
        requested: Decimal,
        /// Quantity it left.
        allowed: Decimal,
        /// The budget.
        budget: Decimal,
    },
    /// The ledger has too little free cash to hold the reservation; the side is withheld.
    ReservationRefused {
        /// What the side needed.
        required: Decimal,
        /// What was free.
        available: Decimal,
    },
}

/// One constraint active on one side in a round.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OptionConstraint {
    /// Index of the contract.
    pub contract: usize,
    /// Buy for the bid, sell for the ask.
    pub side: Action,
    /// The stage that acted, and why.
    pub kind: OptionConstraintKind,
}

/// A quote the ledger holds a reservation for.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdmittedOptionQuote {
    /// Index of the contract.
    pub contract: usize,
    /// Buy for the bid, sell for the ask.
    pub side: Action,
    /// Its id, and the id of its reservation.
    pub order_id: OrderId,
    /// The quote price.
    pub price: Decimal,
    /// The quantity still live.
    pub quantity: Decimal,
    /// What it holds in the cash asset.
    pub reserved: Decimal,
}

/// The capital of the quoter, in the cash asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OptionCapital {
    /// Positions at the round's values: a long at its value, a short at its worst scenario loss
    /// from that value, and the hedge at the spot.
    pub deployed: Decimal,
    /// What the live orders hold.
    pub reserved: Decimal,
    /// `deployed + reserved`, less the premium of the buys that only cover a short. This is what
    /// the budget limits.
    pub committed: Decimal,
}

/// What one round produced. The crate does not print: this is the log, for the caller to keep.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptionQuoteRound {
    /// The quotes admitted this round.
    pub quotes: Vec<AdmittedOptionQuote>,
    /// Every active constraint, in the order the stages met them.
    pub constraints: Vec<OptionConstraint>,
    /// The capital after the round.
    pub capital: OptionCapital,
    /// The delta of the positions and the hedge, before any live order fills.
    pub net_delta: Decimal,
    /// The hedge the delta threshold asks for. It is only sized: the caller decides whether to
    /// send it, through its own gate for the spot instrument.
    pub hedge: Option<HedgeOrder>,
}

/// What a fill booked.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OptionFill {
    /// Index of the contract.
    pub contract: usize,
    /// Premium received, negative when paid.
    pub premium: Decimal,
    /// The contract's position after the fill.
    pub position: Decimal,
    /// What the ledger could not cover, if anything.
    pub shortfall: Option<Decimal>,
}

/// A live order and its reservation.
#[derive(Debug, Clone)]
struct LiveOption {
    order_id: OrderId,
    contract: usize,
    side: Dir,
    price: Decimal,
    remaining: Decimal,
    /// The part that holds nothing per unit: a sale's unwinding part. Zero for a buy.
    uncharged: Decimal,
    /// Cash held per charged unit.
    unit: Decimal,
    reserved: Decimal,
}

/// A contract's values in one round.
#[derive(Debug, Clone)]
struct Valued {
    base: OptionValuation,
    /// `V_s - V`, per scenario.
    moves: Vec<Decimal>,
    /// The highest of the value and the scenario values.
    top: Decimal,
}

impl Valued {
    /// What holding one short costs from the value: never negative.
    fn short_capital(&self) -> MMResult<Decimal> {
        self.top
            .checked_sub(self.base.price)
            .or_overflow("short capital")
    }
}

/// How each contract's position can still move: every live order fills or not, independently.
/// Each Greek and scenario result is linear in the positions, so its range over this box is
/// exact.
#[derive(Debug, Clone)]
struct Reach {
    low: Vec<Decimal>,
    high: Vec<Decimal>,
}

impl Reach {
    /// The range of `sum position_i coefficient_i + fixed`.
    fn range(
        &self,
        valued: &[Option<Valued>],
        coefficient: impl Fn(&Valued) -> Decimal,
        fixed: Decimal,
    ) -> MMResult<(Decimal, Decimal)> {
        let (mut lowest, mut highest) = (fixed, fixed);
        for ((low, high), entry) in self.low.iter().zip(&self.high).zip(valued) {
            let Some(entry) = entry else { continue };
            let c = coefficient(entry);
            let a = low.checked_mul(c).or_overflow("position risk")?;
            let b = high.checked_mul(c).or_overflow("position risk")?;
            lowest = lowest.checked_add(a.min(b)).or_overflow("portfolio risk")?;
            highest = highest
                .checked_add(a.max(b))
                .or_overflow("portfolio risk")?;
        }
        Ok((lowest, highest))
    }

    /// With one more side of `size` that may fill.
    fn with_side(&self, contract: usize, side: Dir, size: Decimal) -> MMResult<Self> {
        let mut next = self.clone();
        let slot = match side {
            Dir::Buy => next.high.get_mut(contract),
            Dir::Sell => next.low.get_mut(contract),
        }
        .ok_or_else(|| out_of_range(contract))?;
        *slot = match side {
            Dir::Buy => slot.checked_add(size),
            Dir::Sell => slot.checked_sub(size),
        }
        .or_overflow("worst case position")?;
        Ok(next)
    }
}

/// A Greek limited in absolute value: which, how to read it, what the hedge adds, and the limit.
type AbsoluteGreek = (LimitedGreek, fn(&Valued) -> Decimal, Decimal, Decimal);

/// What the stages of one round read.
#[derive(Debug)]
struct RoundView<'a> {
    valued: &'a [Option<Valued>],
    assets: &'a [Option<usize>],
    model: &'a MultiAssetModel,
    inventory: &'a [Decimal],
    hedge: Decimal,
    market: &'a OptionMarket,
}

/// A side on its way through the stages.
#[derive(Debug, Clone, Copy)]
struct OptionCandidate {
    contract: usize,
    side: Dir,
    price: Decimal,
    quantity: Decimal,
    /// The part that commits nothing new: it unwinds a position.
    covered: Decimal,
    /// Capital one unit beyond `covered` commits.
    unit_capital: Decimal,
}

fn out_of_range(contract: usize) -> MMError {
    MMError::InvalidConfiguration(format!("contract {contract} is out of range"))
}

fn config(what: String) -> MMError {
    MMError::InvalidConfiguration(what)
}

fn ledger_error(what: &str, error: &MarginError) -> MMError {
    MMError::InvalidPositionUpdate(format!("option quoter {what}: {error}"))
}

fn positive(name: &str, value: Decimal) -> MMResult<()> {
    if value <= Decimal::ZERO {
        return Err(config(format!("{name} must be positive, got {value}")));
    }
    Ok(())
}

fn floor_to(value: Decimal, scale: u32) -> Decimal {
    value.round_dp_with_strategy(scale, RoundingStrategy::ToZero)
}

/// Joint quotes for option contracts on one underlying, under one budget.
///
/// Each round runs these stages, in this order. A later stage can only reduce what an earlier
/// one left, and the ledger has the last word:
///
/// 1. **Quote adjustment.** The pricer's value, less or plus the joint model's offsets for the
///    positions and the hedge, rounded away from the value onto the contract's grid.
/// 2. **Hard limits.** The inventory limit, the Greek limits and the scenario loss limit, on what
///    can still happen: every live order and every side already admitted this round may fill or
///    not, in any combination. A breach withholds the side, and a side that does not worsen the
///    figure is not withheld for it.
/// 3. **Size.** The remaining sides share what the budget leaves after
///    [`OptionCapital::committed`]: each is scaled by the same factor, rounded down to a lot. The
///    part of a side that only unwinds a position commits nothing, so it is not scaled.
/// 4. **Hard rejection.** Each side reserves in the ledger, and a quote exists only if the ledger
///    accepted. If rounding takes the committed capital past the budget, it is released.
/// 5. **Hedge.** A delta beyond the threshold yields a sized [`HedgeOrder`], which is reported
///    and not sent. Withholding a side does not trigger one.
///
/// A round never raises the committed capital above the budget, except for what only unwinds a
/// position. A hedge the caller sends is held at the spot in [`OptionCapital::deployed`], so it is
/// sized by the delta and can pass the budget, and the next rounds then have less left. Values can
/// pass it too, by moving.
///
/// The caller owns the order lifecycle: it reports fills and cancels through
/// [`OptionQuoter::on_fill`] and [`OptionQuoter::on_cancel`], and cancels the previous round
/// before the next if it wants the quotes replaced, since live orders keep their capital and
/// their limits. The hedge position is what the ledger holds of the underlying's instrument.
///
/// The positions are this quoter's, as the ledger has no option instrument. A new quoter starts
/// flat unless told otherwise with [`OptionQuoter::with_restored_state`].
#[derive(Debug, Clone)]
pub struct OptionQuoter {
    config: OptionQuoterConfig,
    greek_limits: Option<GreeksLimits>,
    max_scenario_loss: Option<Decimal>,
    positions: Vec<Decimal>,
    live: Vec<LiveOption>,
    next_round: u64,
}

impl OptionQuoter {
    /// A quoter with no position and no order.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if there is no contract or scenario, more
    /// contracts than the model takes, a repeated or unusable name, a figure that must be positive
    /// and is not, a correlation outside `[-1, 1]`, or a trade size finer than its lot or above its
    /// limit.
    pub fn new(config: OptionQuoterConfig) -> MMResult<Self> {
        if config.contracts.is_empty() {
            return Err(config_empty("contract"));
        }
        if config.scenarios.is_empty() {
            return Err(config_empty("scenario"));
        }
        if config.contracts.len() >= MAX_ASSETS {
            return Err(self::config(format!(
                "{} contracts and the underlying exceed the model's {MAX_ASSETS} assets",
                config.contracts.len()
            )));
        }
        positive("capital budget", config.capital_budget)?;
        positive("risk aversion", config.risk_aversion)?;
        positive("spot volatility", config.factors.spot_volatility)?;
        positive(
            "volatility of volatility",
            config.factors.volatility_of_volatility,
        )?;
        if config.factors.correlation.abs() > Decimal::ONE {
            return Err(self::config(format!(
                "factor correlation must be in [-1, 1], got {}",
                config.factors.correlation
            )));
        }
        for scenario in &config.scenarios {
            positive("scenario spot factor", scenario.spot_factor)?;
        }
        AssetQuotingParams::new(
            config.factors.spot_volatility,
            config.underlying.intensity_scale,
            config.underlying.intensity_decay,
            config.underlying.trade_size,
            config.underlying.inventory_limit,
        )?;
        for (index, terms) in config.contracts.iter().enumerate() {
            check_terms(terms, &config.underlying.instrument)?;
            if config
                .contracts
                .iter()
                .take(index)
                .any(|other| other.name == terms.name)
            {
                return Err(self::config(format!("contract {} repeats", terms.name)));
            }
        }
        let positions = vec![Decimal::ZERO; config.contracts.len()];
        Ok(Self {
            config,
            greek_limits: None,
            max_scenario_loss: None,
            positions,
            live: Vec::new(),
            next_round: 0,
        })
    }

    /// Applies Greek limits, and the delta threshold that sizes a hedge.
    #[must_use]
    pub fn with_greek_limits(mut self, limits: GreeksLimits) -> Self {
        self.greek_limits = Some(limits);
        self
    }

    /// Withholds a side that could take the loss in a scenario past `limit`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `limit` is not positive.
    pub fn with_max_scenario_loss(mut self, limit: Decimal) -> MMResult<Self> {
        positive("scenario loss limit", limit)?;
        self.max_scenario_loss = Some(limit);
        Ok(self)
    }

    /// Starts from these positions, one per contract, and this round, after a restart for
    /// instance: [`OptionQuoter::next_round`] of the quoter it replaces. Orders live at the restart
    /// must have been cancelled, as a new quoter does not know them and their reservations stay
    /// in the ledger. An earlier round would reuse order ids, and the ledger then refuses their
    /// fills as already booked, so nothing is booked twice.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] without one position per contract.
    pub fn with_restored_state(
        mut self,
        positions: Vec<Decimal>,
        next_round: u64,
    ) -> MMResult<Self> {
        if positions.len() != self.config.contracts.len() {
            return Err(config(format!(
                "{} positions for {} contracts",
                positions.len(),
                self.config.contracts.len()
            )));
        }
        self.positions = positions;
        self.next_round = next_round;
        Ok(self)
    }

    /// The next round number.
    #[must_use]
    pub fn next_round(&self) -> u64 {
        self.next_round
    }

    /// The position per contract.
    #[must_use]
    pub fn positions(&self) -> &[Decimal] {
        &self.positions
    }

    /// The live orders.
    #[must_use]
    pub fn live_orders(&self) -> Vec<AdmittedOptionQuote> {
        self.live
            .iter()
            .map(|order| AdmittedOptionQuote {
                contract: order.contract,
                side: Action::from(order.side),
                order_id: order.order_id.clone(),
                price: order.price,
                quantity: order.remaining,
                reserved: order.reserved,
            })
            .collect()
    }

    /// The hedge position and the lot of the underlying's instrument.
    fn hedge_position(&self, ledger: &Ledger) -> MMResult<(Decimal, u32)> {
        let id = &self.config.underlying.instrument;
        let Some(Instrument::Spot(spot)) = ledger.instrument(id) else {
            return Err(config(format!(
                "{} is not a spot instrument in the ledger",
                id.as_str()
            )));
        };
        if *spot.quote() != self.config.cash_asset {
            return Err(config(format!(
                "{} is not quoted in the cash asset",
                id.as_str()
            )));
        }
        let held = ledger
            .position(id)
            .map_or(Decimal::ZERO, |position| position.quantity);
        Ok((held, spot.quantity_scale()))
    }

    fn cash_scale(&self, ledger: &Ledger) -> MMResult<u32> {
        ledger.asset_scale(&self.config.cash_asset).ok_or_else(|| {
            config(format!(
                "cash asset {} is not registered",
                self.config.cash_asset
            ))
        })
    }

    /// Whether anything is held or live on a contract.
    fn exposed(&self, contract: usize) -> bool {
        self.positions
            .get(contract)
            .is_some_and(|position| !position.is_zero())
            || self.live.iter().any(|order| order.contract == contract)
    }

    /// The values of a contract, or the reason a side cannot be quoted. A contract with
    /// exposure must have a value, or nothing about the portfolio could be bounded.
    fn value_contract(
        &self,
        contract: usize,
        terms: &ContractTerms,
        market: &OptionMarket,
    ) -> MMResult<Result<Valued, OptionConstraintKind>> {
        if market.now_ms >= terms.expiry_ms {
            if self.exposed(contract) {
                return Err(MMError::InvalidMarketState(format!(
                    "{} has expired with exposure: settle it first",
                    terms.name
                )));
            }
            return Ok(Err(OptionConstraintKind::Expired));
        }
        let days = Decimal::from(terms.expiry_ms.saturating_sub(market.now_ms))
            .checked_div(DAY_MS)
            .or_overflow("days to expiry")?;
        let inputs = |spot| -> MMResult<OptionInputs> {
            Ok(OptionInputs::european(
                spot,
                terms.strike,
                tau_from_days(days)?,
                market.rate,
                market.carry,
                terms.style,
            ))
        };
        let attempt = || -> MMResult<Valued> {
            let base = self.config.pricer.value(&inputs(market.spot)?)?;
            let mut moves = Vec::with_capacity(self.config.scenarios.len());
            let mut top = base.price;
            for scenario in &self.config.scenarios {
                let spot = market
                    .spot
                    .checked_mul(scenario.spot_factor)
                    .or_overflow("scenario spot")?;
                let value = scenario.pricer.value(&inputs(spot)?)?.price;
                top = top.max(value);
                moves.push(value.checked_sub(base.price).or_overflow("scenario move")?);
            }
            Ok(Valued { base, moves, top })
        };
        match attempt() {
            Ok(valued) => Ok(Ok(valued)),
            Err(error) if self.exposed(contract) => Err(error),
            Err(error) => Ok(Err(OptionConstraintKind::NotValued {
                reason: error.to_string(),
            })),
        }
    }

    /// The joint model of the round, with the underlying as asset 0, and each contract's asset
    /// where it has one.
    fn round_model(
        &self,
        valued: &[Option<Valued>],
    ) -> MMResult<(MultiAssetModel, Vec<Option<usize>>)> {
        let factors = &self.config.factors;
        let to_f64 = |value: Decimal, name: &str| {
            value.to_f64().ok_or_else(|| {
                MMError::NumericalError(format!("{name} {value} has no floating point value"))
            })
        };
        let (s, v, rho) = (
            to_f64(factors.spot_volatility, "spot volatility")?,
            to_f64(factors.volatility_of_volatility, "volatility of volatility")?,
            to_f64(factors.correlation, "factor correlation")?,
        );
        let covariance = |a: (f64, f64), b: (f64, f64)| {
            a.0 * b.0 * s * s + a.1 * b.1 * v * v + (a.0 * b.1 + a.1 * b.0) * rho * s * v
        };

        let underlying = &self.config.underlying;
        let mut loadings = vec![(1.0, 0.0)];
        let mut flows = vec![(
            underlying.intensity_scale,
            underlying.intensity_decay,
            underlying.trade_size,
            underlying.inventory_limit,
        )];
        let mut assets = Vec::with_capacity(valued.len());
        for (entry, terms) in valued.iter().zip(&self.config.contracts) {
            let Some(entry) = entry else {
                assets.push(None);
                continue;
            };
            let loading = (
                to_f64(entry.base.delta, "delta")?,
                to_f64(entry.base.vega, "vega")?,
            );
            if covariance(loading, loading) <= f64::EPSILON * s * s {
                assets.push(None);
                continue;
            }
            assets.push(Some(loadings.len()));
            loadings.push(loading);
            flows.push((
                terms.intensity_scale,
                terms.intensity_decay,
                terms.trade_size,
                terms.inventory_limit,
            ));
        }

        let mut params = Vec::with_capacity(loadings.len());
        for (loading, flow) in loadings.iter().zip(&flows) {
            let sigma = covariance(*loading, *loading).sqrt();
            let sigma = Decimal::from_f64(sigma).ok_or_else(|| {
                MMError::NumericalError(format!("volatility {sigma} is not a decimal"))
            })?;
            params.push(AssetQuotingParams::new(
                sigma, flow.0, flow.1, flow.2, flow.3,
            )?);
        }
        let mut correlation = Vec::with_capacity(loadings.len());
        for (i, a) in loadings.iter().enumerate() {
            let mut row = Vec::with_capacity(loadings.len());
            for (j, b) in loadings.iter().enumerate() {
                if i == j {
                    row.push(Decimal::ONE);
                    continue;
                }
                let (p, q) = if i < j { (a, b) } else { (b, a) };
                let value =
                    covariance(*p, *q) / (covariance(*p, *p).sqrt() * covariance(*q, *q).sqrt());
                let value = Decimal::from_f64(value.clamp(-1.0, 1.0))
                    .ok_or_else(|| {
                        MMError::NumericalError(format!("correlation {value} is not a decimal"))
                    })?
                    .round_dp(CORRELATION_DP);
                row.push(value);
            }
            correlation.push(row);
        }
        let model = MultiAssetModel::new(
            params,
            &correlation,
            self.config.risk_aversion,
            self.config.objective,
            self.config.risk_view,
        )?;
        Ok((model, assets))
    }

    /// What the live orders can do to the positions.
    fn reach(&self) -> MMResult<Reach> {
        let mut reach = Reach {
            low: self.positions.clone(),
            high: self.positions.clone(),
        };
        for order in &self.live {
            reach = reach.with_side(order.contract, order.side, order.remaining)?;
        }
        Ok(reach)
    }

    /// The first limit this widening of the reach breaks, if any.
    fn breach(
        &self,
        before: &Reach,
        after: &Reach,
        valued: &[Option<Valued>],
        hedge: Decimal,
        market: &OptionMarket,
    ) -> MMResult<Option<OptionConstraintKind>> {
        if let Some(limits) = &self.greek_limits {
            let absolute: [AbsoluteGreek; 3] = [
                (
                    LimitedGreek::Delta,
                    |entry| entry.base.delta,
                    hedge,
                    limits.max_delta,
                ),
                (
                    LimitedGreek::Gamma,
                    |entry| entry.base.gamma,
                    Decimal::ZERO,
                    limits.max_gamma,
                ),
                (
                    LimitedGreek::Vega,
                    |entry| entry.base.vega,
                    Decimal::ZERO,
                    limits.max_vega,
                ),
            ];
            for (greek, coefficient, fixed, limit) in absolute {
                let (low, high) = before.range(valued, coefficient, fixed)?;
                let worst_before = low.abs().max(high.abs());
                let (low, high) = after.range(valued, coefficient, fixed)?;
                let worst = low.abs().max(high.abs());
                if worst > limit && worst > worst_before {
                    return Ok(Some(OptionConstraintKind::GreekLimit {
                        greek,
                        value: worst,
                        limit,
                    }));
                }
            }
            let daily = |entry: &Valued| {
                entry
                    .base
                    .theta
                    .checked_div(PRICING_YEAR_DAYS)
                    .unwrap_or(Decimal::ZERO)
            };
            let (lowest_before, _) = before.range(valued, daily, Decimal::ZERO)?;
            let (lowest, _) = after.range(valued, daily, Decimal::ZERO)?;
            if lowest < limits.max_theta && lowest < lowest_before {
                return Ok(Some(OptionConstraintKind::GreekLimit {
                    greek: LimitedGreek::Theta,
                    value: lowest,
                    limit: limits.max_theta,
                }));
            }
        }
        if let Some(limit) = self.max_scenario_loss {
            for (index, scenario) in self.config.scenarios.iter().enumerate() {
                let fixed = hedge
                    .checked_mul(market.spot)
                    .and_then(|value| {
                        scenario
                            .spot_factor
                            .checked_sub(Decimal::ONE)
                            .and_then(|change| value.checked_mul(change))
                    })
                    .or_overflow("hedge scenario result")?;
                let coefficient =
                    |entry: &Valued| entry.moves.get(index).copied().unwrap_or(Decimal::ZERO);
                let (lowest_before, _) = before.range(valued, coefficient, fixed)?;
                let (lowest, _) = after.range(valued, coefficient, fixed)?;
                let loss = Decimal::ZERO
                    .checked_sub(lowest)
                    .or_overflow("scenario loss")?;
                if lowest < lowest_before && loss > limit {
                    return Ok(Some(OptionConstraintKind::ScenarioLoss {
                        scenario: scenario.name.clone(),
                        loss,
                        limit,
                    }));
                }
            }
        }
        Ok(None)
    }

    /// The capital at the round's values.
    fn capital(
        &self,
        valued: &[Option<Valued>],
        hedge: Decimal,
        spot: Decimal,
    ) -> MMResult<OptionCapital> {
        let mut deployed = hedge
            .abs()
            .checked_mul(spot)
            .or_overflow("deployed hedge")?;
        for (position, entry) in self.positions.iter().zip(valued) {
            let Some(entry) = entry else { continue };
            let unit = if position.is_sign_negative() {
                entry.short_capital()?
            } else {
                entry.base.price
            };
            let value = position
                .abs()
                .checked_mul(unit)
                .or_overflow("deployed capital")?;
            deployed = deployed
                .checked_add(value)
                .or_overflow("deployed capital")?;
        }
        let mut reserved = Decimal::ZERO;
        let mut covered = Decimal::ZERO;
        let mut covering = vec![Decimal::ZERO; self.positions.len()];
        for order in &self.live {
            reserved = reserved
                .checked_add(order.reserved)
                .or_overflow("reserved capital")?;
            if order.side != Dir::Buy {
                continue;
            }
            let (Some(position), Some(used)) = (
                self.positions.get(order.contract),
                covering.get_mut(order.contract),
            ) else {
                continue;
            };
            let short = Decimal::ZERO
                .checked_sub(*position)
                .or_overflow("short position")?
                .max(Decimal::ZERO);
            let free = short.checked_sub(*used).or_overflow("short left")?;
            let part = order.remaining.min(free.max(Decimal::ZERO));
            *used = used.checked_add(part).or_overflow("short covered")?;
            let premium = part.checked_mul(order.price).or_overflow("covered buy")?;
            covered = covered.checked_add(premium).or_overflow("covered buys")?;
        }
        let committed = deployed
            .checked_add(reserved)
            .and_then(|sum| sum.checked_sub(covered))
            .or_overflow("committed capital")?;
        Ok(OptionCapital {
            deployed,
            reserved,
            committed,
        })
    }

    /// Stages 1 and 2.
    fn candidates(
        &self,
        view: &RoundView<'_>,
        constraints: &mut Vec<OptionConstraint>,
    ) -> MMResult<Vec<OptionCandidate>> {
        let RoundView {
            valued,
            assets,
            model,
            inventory,
            hedge,
            market,
        } = *view;
        let mut reach = self.reach()?;
        let mut candidates = Vec::new();
        for (contract, terms) in self.config.contracts.iter().enumerate() {
            let (Some(Some(entry)), Some(asset)) = (valued.get(contract), assets.get(contract))
            else {
                continue;
            };
            let Some(asset) = asset else {
                for side in [Action::Buy, Action::Sell] {
                    constraints.push(OptionConstraint {
                        contract,
                        side,
                        kind: OptionConstraintKind::NoSensitivity,
                    });
                }
                continue;
            };
            let offsets = model.offsets(*asset, inventory)?;
            let value = entry.base.price;
            let bid = value
                .checked_sub(offsets.bid)
                .or_overflow("bid price")?
                .round_dp_with_strategy(terms.price_scale, RoundingStrategy::ToNegativeInfinity);
            let ask = value
                .checked_add(offsets.ask)
                .or_overflow("ask price")?
                .round_dp_with_strategy(terms.price_scale, RoundingStrategy::ToPositiveInfinity);
            for (side, price) in [(Dir::Buy, bid), (Dir::Sell, ask)] {
                let mut push = |kind| {
                    constraints.push(OptionConstraint {
                        contract,
                        side: Action::from(side),
                        kind,
                    });
                };
                if price <= Decimal::ZERO {
                    push(OptionConstraintKind::NonPositivePrice { price });
                    continue;
                }
                let widened = reach.with_side(contract, side, terms.trade_size)?;
                let resulting = match side {
                    Dir::Buy => widened.high.get(contract),
                    Dir::Sell => widened.low.get(contract),
                }
                .copied()
                .ok_or_else(|| out_of_range(contract))?;
                if resulting.abs() > terms.inventory_limit {
                    push(OptionConstraintKind::InventoryLimit {
                        resulting,
                        limit: terms.inventory_limit,
                    });
                    continue;
                }
                if let Some(kind) = self.breach(&reach, &widened, valued, hedge, market)? {
                    push(kind);
                    continue;
                }
                let through = match side {
                    Dir::Buy => price >= value,
                    Dir::Sell => price <= value,
                };
                if through {
                    push(OptionConstraintKind::ThroughValue { price, value });
                }
                // What this side can still unwind, after the live orders of its side.
                let unwinds = match side {
                    Dir::Buy => Decimal::ZERO
                        .checked_sub(reach.high.get(contract).copied().unwrap_or_default())
                        .or_overflow("short left")?,
                    Dir::Sell => reach.low.get(contract).copied().unwrap_or_default(),
                }
                .max(Decimal::ZERO);
                let unit_capital = match side {
                    Dir::Buy => price,
                    Dir::Sell => entry
                        .top
                        .checked_sub(price)
                        .or_overflow("sale capital")?
                        .max(Decimal::ZERO),
                };
                candidates.push(OptionCandidate {
                    contract,
                    side,
                    price,
                    quantity: terms.trade_size,
                    covered: unwinds.min(terms.trade_size),
                    unit_capital,
                });
                reach = widened;
            }
        }
        Ok(candidates)
    }

    /// Stage 3.
    fn sized(
        &self,
        committed: Decimal,
        candidates: Vec<OptionCandidate>,
        constraints: &mut Vec<OptionConstraint>,
    ) -> MMResult<Vec<OptionCandidate>> {
        let budget = self.config.capital_budget;
        let left = budget
            .checked_sub(committed)
            .or_overflow("budget left")?
            .max(Decimal::ZERO);
        let mut asked = Decimal::ZERO;
        for candidate in &candidates {
            let charged = candidate
                .quantity
                .checked_sub(candidate.covered)
                .or_overflow("charged quantity")?;
            let capital = charged
                .checked_mul(candidate.unit_capital)
                .or_overflow("capital asked")?;
            asked = asked.checked_add(capital).or_overflow("capital asked")?;
        }
        if asked <= left {
            return Ok(candidates);
        }
        let mut sized = Vec::with_capacity(candidates.len());
        for mut candidate in candidates {
            let terms = self
                .config
                .contracts
                .get(candidate.contract)
                .ok_or_else(|| out_of_range(candidate.contract))?;
            let charged = candidate
                .quantity
                .checked_sub(candidate.covered)
                .or_overflow("charged quantity")?;
            let kept = if candidate.unit_capital.is_zero() {
                charged
            } else {
                floor_to(
                    charged
                        .checked_mul(left)
                        .and_then(|value| value.checked_div(asked))
                        .or_overflow("scaled quantity")?,
                    terms.quantity_scale,
                )
            };
            let allowed = candidate
                .covered
                .checked_add(kept)
                .or_overflow("allowed quantity")?;
            if allowed < candidate.quantity {
                constraints.push(OptionConstraint {
                    contract: candidate.contract,
                    side: Action::from(candidate.side),
                    kind: OptionConstraintKind::CapitalBudget {
                        requested: candidate.quantity,
                        allowed,
                        budget,
                    },
                });
            }
            if allowed > Decimal::ZERO {
                candidate.quantity = allowed;
                sized.push(candidate);
            }
        }
        Ok(sized)
    }

    /// The next round whose ids hold no reservation in the ledger.
    fn free_round(&mut self, ledger: &Ledger, sized: &[OptionCandidate]) -> MMResult<u64> {
        loop {
            let round = self.next_round;
            self.next_round = round.saturating_add(1);
            let mut used = false;
            for candidate in sized {
                let id = reservation_id(&order_id(round, candidate)?)?;
                used = used || ledger.reservation(&id).is_some();
            }
            if !used || round == u64::MAX {
                return Ok(round);
            }
        }
    }

    /// Runs one round and reserves for what is left of it.
    ///
    /// A refusal is not an error: it is logged in the round.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the underlying is not a spot instrument of the
    /// ledger quoted in a registered cash asset, or a contract's premium would be finer than that
    /// asset. Returns [`MMError::InvalidMarketState`] if the
    /// spot is not positive, or a contract with exposure has expired. Returns the pricer's error
    /// for a contract with exposure it cannot value, the model's errors, and
    /// [`MMError::NumericalError`] on overflow.
    pub fn quote(
        &mut self,
        ledger: &mut Ledger,
        market: &OptionMarket,
    ) -> MMResult<OptionQuoteRound> {
        if market.spot <= Decimal::ZERO {
            return Err(MMError::InvalidMarketState(format!(
                "spot must be positive, got {}",
                market.spot
            )));
        }
        let (hedge, spot_scale) = self.hedge_position(ledger)?;
        let cash_scale = self.cash_scale(ledger)?;
        for terms in &self.config.contracts {
            if terms.price_scale.saturating_add(terms.quantity_scale) > cash_scale {
                return Err(config(format!(
                    "a premium of {} is finer than the cash asset's {cash_scale} places",
                    terms.name
                )));
            }
        }
        let mut constraints = Vec::new();
        let mut valued = Vec::with_capacity(self.config.contracts.len());
        for (contract, terms) in self.config.contracts.iter().enumerate() {
            match self.value_contract(contract, terms, market)? {
                Ok(entry) => valued.push(Some(entry)),
                Err(kind) => {
                    for side in [Action::Buy, Action::Sell] {
                        constraints.push(OptionConstraint {
                            contract,
                            side,
                            kind: kind.clone(),
                        });
                    }
                    valued.push(None);
                }
            }
        }
        let (model, assets) = self.round_model(&valued)?;
        let mut inventory = vec![hedge];
        for (position, asset) in self.positions.iter().zip(&assets) {
            if asset.is_some() {
                inventory.push(*position);
            }
        }
        let view = RoundView {
            valued: &valued,
            assets: &assets,
            model: &model,
            inventory: &inventory,
            hedge,
            market,
        };
        let candidates = self.candidates(&view, &mut constraints)?;
        let committed = self.capital(&valued, hedge, market.spot)?.committed;
        let sized = self.sized(committed, candidates, &mut constraints)?;

        let round = self.free_round(ledger, &sized)?;
        let mut quotes = Vec::with_capacity(sized.len());
        for candidate in sized {
            let side = Action::from(candidate.side);
            let id = order_id(round, &candidate)?;
            let uncharged = if candidate.side == Dir::Sell {
                candidate.covered
            } else {
                Decimal::ZERO
            };
            let required = reservation_amount(
                candidate.quantity,
                uncharged,
                candidate.unit_capital,
                cash_scale,
            )?;
            let reservation = reservation_id(&id)?;
            if required > Decimal::ZERO {
                match ledger.reserve(
                    reservation.clone(),
                    self.config.cash_asset.clone(),
                    required,
                ) {
                    Ok(_) => {}
                    Err(MarginError::InsufficientBalance {
                        required,
                        available,
                        ..
                    }) => {
                        constraints.push(OptionConstraint {
                            contract: candidate.contract,
                            side,
                            kind: OptionConstraintKind::ReservationRefused {
                                required,
                                available,
                            },
                        });
                        continue;
                    }
                    Err(error) => return Err(ledger_error("reservation", &error)),
                }
            }
            self.live.push(LiveOption {
                order_id: id.clone(),
                contract: candidate.contract,
                side: candidate.side,
                price: candidate.price,
                remaining: candidate.quantity,
                uncharged,
                unit: candidate.unit_capital,
                reserved: required,
            });
            let committed = self.capital(&valued, hedge, market.spot)?.committed;
            if committed > self.config.capital_budget && candidate.covered < candidate.quantity {
                self.live.pop();
                if required > Decimal::ZERO {
                    ledger
                        .release(&reservation)
                        .map_err(|error| ledger_error("release", &error))?;
                }
                constraints.push(OptionConstraint {
                    contract: candidate.contract,
                    side,
                    kind: OptionConstraintKind::CapitalBudget {
                        requested: candidate.quantity,
                        allowed: Decimal::ZERO,
                        budget: self.config.capital_budget,
                    },
                });
                continue;
            }
            quotes.push(AdmittedOptionQuote {
                contract: candidate.contract,
                side,
                order_id: id,
                price: candidate.price,
                quantity: candidate.quantity,
                reserved: required,
            });
        }

        let net_delta = self.net_delta(&valued, hedge)?;
        Ok(OptionQuoteRound {
            quotes,
            constraints,
            capital: self.capital(&valued, hedge, market.spot)?,
            net_delta,
            hedge: self.hedge_order(net_delta, spot_scale, market.spot),
        })
    }

    fn net_delta(&self, valued: &[Option<Valued>], hedge: Decimal) -> MMResult<Decimal> {
        let mut delta = hedge;
        for (position, entry) in self.positions.iter().zip(valued) {
            let Some(entry) = entry else { continue };
            let part = position
                .checked_mul(entry.base.delta)
                .or_overflow("position delta")?;
            delta = delta.checked_add(part).or_overflow("portfolio delta")?;
        }
        Ok(delta)
    }

    fn hedge_order(&self, net_delta: Decimal, lot: u32, spot: Decimal) -> Option<HedgeOrder> {
        let limits = self.greek_limits.as_ref()?;
        if net_delta.abs() <= limits.delta_hedge_threshold {
            return None;
        }
        let quantity = floor_to(Decimal::ZERO.checked_sub(net_delta)?, lot);
        if quantity.is_zero() {
            return None;
        }
        Some(HedgeOrder::underlying(
            self.config.underlying.instrument.as_str().to_string(),
            quantity,
            spot,
        ))
    }

    /// Books a fill: the reservation shrinks with the order, the premium is booked under the
    /// contract's name, and the position moves. Replaying a fill at the same point of an order is
    /// refused by the ledger. On error nothing changes.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidPositionUpdate`] for an unknown order, a quantity that is not
    /// positive or exceeds what is live, a premium finer than the cash asset, or what the ledger
    /// refuses, and [`MMError::NumericalError`] on overflow.
    pub fn on_fill(
        &mut self,
        ledger: &mut Ledger,
        order: &OrderId,
        quantity: Decimal,
    ) -> MMResult<OptionFill> {
        let index = self.live_index(order)?;
        let live = self
            .live
            .get(index)
            .cloned()
            .ok_or_else(|| unknown(order))?;
        if quantity <= Decimal::ZERO || quantity > live.remaining {
            return Err(MMError::InvalidPositionUpdate(format!(
                "fill of {quantity} on order {order} with {} live",
                live.remaining
            )));
        }
        let terms = self
            .config
            .contracts
            .get(live.contract)
            .ok_or_else(|| out_of_range(live.contract))?;
        let remaining = live
            .remaining
            .checked_sub(quantity)
            .or_overflow("remaining")?;
        let cash_scale = self.cash_scale(ledger)?;
        let held = reservation_amount(remaining, live.uncharged, live.unit, cash_scale)?;
        let premium = quantity.checked_mul(live.price).or_overflow("premium")?;
        let premium = match live.side {
            Dir::Buy => Decimal::ZERO.checked_sub(premium).or_overflow("premium")?,
            Dir::Sell => premium,
        };

        let mut trial = ledger.clone();
        if !live.reserved.is_zero() {
            trial
                .set_reservation(
                    &reservation_id(order)?,
                    self.config.cash_asset.clone(),
                    held,
                )
                .map_err(|error| ledger_error("reservation", &error))?;
        }
        let booking = trial
            .book_settlement(&DerivativeSettlement {
                id: ExecutionId::new(format!("premium:{order}:{remaining}"))
                    .map_err(|error| ledger_error("id", &error))?,
                instrument: InstrumentId::new(terms.name.as_str())
                    .map_err(|error| ledger_error("id", &error))?,
                asset: self.config.cash_asset.clone(),
                realized_pnl: premium,
                fee: Decimal::ZERO,
            })
            .map_err(|error| ledger_error("premium", &error))?;
        if !booking.is_new {
            return Err(MMError::InvalidPositionUpdate(format!(
                "this fill of order {order} was already booked"
            )));
        }
        let change = match live.side {
            Dir::Buy => quantity,
            Dir::Sell => Decimal::ZERO.checked_sub(quantity).or_overflow("sold")?,
        };
        let position = self
            .positions
            .get(live.contract)
            .copied()
            .ok_or_else(|| out_of_range(live.contract))?
            .checked_add(change)
            .or_overflow("position")?;

        *ledger = trial;
        if let Some(slot) = self.positions.get_mut(live.contract) {
            *slot = position;
        }
        if remaining.is_zero() {
            self.live.remove(index);
        } else if let Some(slot) = self.live.get_mut(index) {
            slot.remaining = remaining;
            slot.reserved = held;
        }
        Ok(OptionFill {
            contract: live.contract,
            premium,
            position,
            shortfall: booking.shortfall,
        })
    }

    /// Releases what a cancelled order holds.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidPositionUpdate`] for an unknown order or a reservation the ledger
    /// no longer has.
    pub fn on_cancel(&mut self, ledger: &mut Ledger, order: &OrderId) -> MMResult<()> {
        let index = self.live_index(order)?;
        let held = self
            .live
            .get(index)
            .is_some_and(|live| !live.reserved.is_zero());
        if held {
            ledger
                .release(&reservation_id(order)?)
                .map_err(|error| ledger_error("release", &error))?;
        }
        self.live.remove(index);
        Ok(())
    }

    fn live_index(&self, order: &OrderId) -> MMResult<usize> {
        self.live
            .iter()
            .position(|live| live.order_id == *order)
            .ok_or_else(|| unknown(order))
    }
}

fn unknown(order: &OrderId) -> MMError {
    MMError::InvalidPositionUpdate(format!("order {order} is not live"))
}

fn config_empty(what: &str) -> MMError {
    config(format!("an option quoter needs at least one {what}"))
}

/// `(quantity - uncharged) * unit` on the asset's grid, rounded up so that a reservation is
/// never short of what it stands for.
fn reservation_amount(
    quantity: Decimal,
    uncharged: Decimal,
    unit: Decimal,
    scale: u32,
) -> MMResult<Decimal> {
    let charged = quantity
        .checked_sub(uncharged)
        .or_overflow("charged quantity")?
        .max(Decimal::ZERO);
    Ok(charged
        .checked_mul(unit)
        .or_overflow("reservation")?
        .round_dp_with_strategy(scale, RoundingStrategy::ToPositiveInfinity))
}

fn order_id(round: u64, candidate: &OptionCandidate) -> MMResult<OrderId> {
    let side = match candidate.side {
        Dir::Buy => "b",
        Dir::Sell => "a",
    };
    OrderId::new(format!("oq-{round}-{}-{side}", candidate.contract))
        .map_err(|error| MMError::InvalidQuoteGeneration(format!("invalid order id: {error}")))
}

fn reservation_id(order: &OrderId) -> MMResult<ReservationId> {
    ReservationId::new(order.as_str())
        .map_err(|error| MMError::InvalidQuoteGeneration(format!("invalid order id: {error}")))
}

fn check_terms(terms: &ContractTerms, underlying: &InstrumentId) -> MMResult<()> {
    let id = InstrumentId::new(terms.name.as_str()).map_err(|error| {
        config(format!(
            "contract name {} is not usable: {error}",
            terms.name
        ))
    })?;
    if id == *underlying {
        return Err(config(format!(
            "contract {} is named as the underlying",
            terms.name
        )));
    }
    positive("strike", terms.strike)?;
    positive("intensity scale", terms.intensity_scale)?;
    positive("intensity decay", terms.intensity_decay)?;
    positive("trade size", terms.trade_size)?;
    if terms.trade_size > terms.inventory_limit {
        return Err(config(format!(
            "trade size {} is above the limit {} of {}",
            terms.trade_size, terms.inventory_limit, terms.name
        )));
    }
    if floor_to(terms.trade_size, terms.quantity_scale) != terms.trade_size {
        return Err(config(format!(
            "trade size {} is finer than the lot of {}",
            terms.trade_size, terms.name
        )));
    }
    Ok(())
}
