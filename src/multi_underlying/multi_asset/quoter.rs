//! The only way from the model to a quote: through a `margin-rs` risk gate.

use std::fmt;

use margin_rs::risk::{FeeAsset, OrderPurpose, OrderRequest, RiskDecision, RiskGate, RiskReason};
use margin_rs::{Asset, Instrument, InstrumentId, OrderId};
use rust_decimal::RoundingStrategy;

use super::model::MultiAssetModel;
use crate::Decimal;
use crate::types::action::{Action, Dir};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

/// A quote the gate approved and holds a reservation for.
///
/// Only [`MultiAssetQuoter::quote`] creates one, so holding it means the order
/// is recorded in the gate under `order_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdmittedQuote {
    /// Index of the asset in the model.
    pub asset: usize,
    /// Buy for the bid, sell for the ask.
    pub side: Action,
    /// The order in the gate.
    pub order_id: OrderId,
    /// Limit price, on the price grid of the instrument.
    pub price: Decimal,
    /// The quantity the gate approved.
    pub quantity: Decimal,
}

/// Why a stage changed or withheld a side.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstraintKind {
    /// The model's price is not positive on the instrument's grid; the side is
    /// withheld.
    NonPositivePrice {
        /// The price after rounding.
        price: Decimal,
    },
    /// A trade would take the inventory beyond the limit of the model; the side
    /// is withheld.
    InventoryLimit {
        /// The inventory after the trade.
        resulting: Decimal,
        /// The limit on its absolute value.
        limit: Decimal,
    },
    /// A trade would raise the portfolio variance beyond the limit; the side is
    /// withheld. A side that lowers the variance is never withheld for this.
    PortfolioRisk {
        /// `q' Sigma q` after the trade, under the model's view.
        variance: Decimal,
        /// The limit.
        limit: Decimal,
    },
    /// The capital budget shared by every quote cut the quantity, to zero when
    /// the side is withheld.
    CapitalBudget {
        /// Quantity asked of this stage.
        requested: Decimal,
        /// Quantity it left.
        allowed: Decimal,
        /// The budget, in the quote asset.
        budget: Decimal,
    },
    /// The gate reduced the order; the quote stands for the smaller quantity.
    MarginReduced {
        /// Quantity submitted.
        requested: Decimal,
        /// Quantity approved.
        approved: Decimal,
        /// Every cap the gate applied.
        reasons: Vec<RiskReason>,
    },
    /// The gate rejected the order; the side is withheld.
    MarginRejected {
        /// Every reason the gate gave.
        reasons: Vec<RiskReason>,
    },
}

/// One constraint that was active on one side in a round.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ActiveConstraint {
    /// Index of the asset in the model.
    pub asset: usize,
    /// Buy for the bid, sell for the ask.
    pub side: Action,
    /// The stage that acted, and why.
    pub kind: ConstraintKind,
}

impl fmt::Display for ActiveConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "asset {} {:?}: ", self.asset, self.side)?;
        match &self.kind {
            ConstraintKind::NonPositivePrice { price } => {
                write!(f, "price {price} is not positive")
            }
            ConstraintKind::InventoryLimit { resulting, limit } => {
                write!(f, "inventory {resulting} would exceed the limit {limit}")
            }
            ConstraintKind::PortfolioRisk { variance, limit } => {
                write!(f, "portfolio variance {variance} would exceed {limit}")
            }
            ConstraintKind::CapitalBudget {
                requested,
                allowed,
                budget,
            } => write!(
                f,
                "capital budget {budget} cut quantity {requested} to {allowed}"
            ),
            ConstraintKind::MarginReduced {
                requested,
                approved,
                reasons,
            } => write!(
                f,
                "margin gate cut quantity {requested} to {approved}: {reasons:?}"
            ),
            ConstraintKind::MarginRejected { reasons } => {
                write!(f, "margin gate rejected: {reasons:?}")
            }
        }
    }
}

/// What one round produced: the quotes standing, and the cause of every side
/// changed or withheld. The crate does not print: this is the log, for the
/// caller to keep or forward.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QuoteRound {
    /// The approved quotes, by asset, bid before ask.
    pub quotes: Vec<AdmittedQuote>,
    /// Every active constraint, in the order the stages met them.
    pub constraints: Vec<ActiveConstraint>,
    /// Capital the gate holds after the round, in the quote asset.
    pub reserved_capital: Decimal,
}

/// A side on its way through the stages.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    asset: usize,
    side: Dir,
    price: Decimal,
    quantity: Decimal,
    /// Capital one unit of quantity is expected to hold.
    unit_capital: Decimal,
}

/// The terms of an instrument, read from the gate.
#[derive(Debug, Clone)]
struct SpotView {
    id: InstrumentId,
    base: Asset,
    quote: Asset,
    price_scale: u32,
    quantity_scale: u32,
    fee_rate: Decimal,
    fee_asset: FeeAsset,
}

/// Joint quotes for spot instruments under one account's gate and budget.
///
/// Each round runs these stages, in this order. A later stage can only reduce
/// what an earlier one left, and the gate has the last word:
///
/// 1. **Quote adjustment.** The model's offsets for the inventory the gate's
///    ledger holds, rounded away from the reference price onto the grid. A
///    price that is not positive withholds the side.
/// 2. **Hard model limits.** The inventory limit, then the optional portfolio
///    variance limit. A breach withholds the side; a side that lowers the
///    variance still quotes.
/// 3. **Size.** All remaining sides share the capital budget left after what
///    the gate already holds: when they ask for more, each is scaled by the
///    same factor, rounded down to a lot. A side rounded to nothing is withheld.
/// 4. **Hard rejection.** Each side is submitted to the gate, which reserves
///    for it and may reduce or reject it. Nothing can bypass this stage, and a
///    quote exists only if the gate approved it. If rounding in a reservation
///    takes the capital held past the budget, the order is released.
///
/// There is no hedging stage: a withheld side is reported, and placing a hedge
/// is left to the caller. The offsets keep assuming the model's trade size when
/// a stage cuts the quantity.
///
/// The caller owns the gate and the order lifecycle: it forwards venue reports,
/// and it cancels the previous round before the next if it wants the quotes
/// replaced, since orders still live keep their capital and their limits.
///
/// Short inventory on a spot instrument draws on a deposit of the base asset,
/// and the ledger's net traded quantity is the inventory the model sees.
///
/// # Examples
///
/// ```
/// use margin_rs::ledger::{Ledger, LedgerConfig};
/// use margin_rs::risk::RiskGate;
/// use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};
/// use market_maker_rs::dec;
/// use market_maker_rs::multi_underlying::multi_asset::{
///     AssetQuotingParams, MultiAssetModel, MultiAssetQuoter, QuotingObjective, RiskView,
/// };
///
/// let (base, usd) = (Asset::new("AAA").unwrap(), Asset::new("USD").unwrap());
/// let id = InstrumentId::new("AAA-USD").unwrap();
/// let mut ledger = Ledger::new(LedgerConfig::default());
/// ledger.register_asset(base.clone(), 8).unwrap();
/// ledger.register_asset(usd.clone(), 8).unwrap();
/// ledger
///     .register_instrument(SpotInstrument::new(id.clone(), base.clone(), usd.clone(), 4, 4).unwrap())
///     .unwrap();
/// let mut gate = RiskGate::new(ledger);
/// gate.deposit(TransferId::new("quote").unwrap(), usd, dec!(100000)).unwrap();
/// gate.deposit(TransferId::new("base").unwrap(), base, dec!(100)).unwrap();
///
/// let params = AssetQuotingParams::new(dec!(1), dec!(10), dec!(20), dec!(1), dec!(50)).unwrap();
/// let model = MultiAssetModel::new(
///     vec![params],
///     &[vec![dec!(1)]],
///     dec!(0.08),
///     QuotingObjective::RiskAdjustedExpectation,
///     RiskView::Correlated,
/// )
/// .unwrap();
/// let mut quoter = MultiAssetQuoter::new(model, vec![id], dec!(1000)).unwrap();
///
/// let round = quoter.quote(&mut gate, &[dec!(100)], 0).unwrap();
/// assert_eq!(round.quotes.len(), 2);
/// assert!(round.constraints.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct MultiAssetQuoter {
    model: MultiAssetModel,
    instruments: Vec<InstrumentId>,
    capital_budget: Decimal,
    max_portfolio_variance: Option<Decimal>,
    next_round: u64,
}

impl MultiAssetQuoter {
    /// A quoter for `instruments`, one per asset of `model`, in its order, that
    /// never holds more than `capital_budget` of capital, in the quote asset.
    ///
    /// Capital is what the gate reserves for live orders: quote for a bid, and
    /// base valued at the reference price for an ask.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the budget is not positive,
    /// the number of instruments differs from the model's assets, or an
    /// instrument repeats.
    pub fn new(
        model: MultiAssetModel,
        instruments: Vec<InstrumentId>,
        capital_budget: Decimal,
    ) -> MMResult<Self> {
        if capital_budget <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "capital budget must be positive, got {capital_budget}"
            )));
        }
        if instruments.len() != model.asset_count() {
            return Err(MMError::InvalidConfiguration(format!(
                "{} instruments for {} assets",
                instruments.len(),
                model.asset_count()
            )));
        }
        for (index, instrument) in instruments.iter().enumerate() {
            if instruments.iter().take(index).any(|id| id == instrument) {
                return Err(MMError::InvalidConfiguration(format!(
                    "instrument {} repeats",
                    instrument.as_str()
                )));
            }
        }
        Ok(Self {
            model,
            instruments,
            capital_budget,
            max_portfolio_variance: None,
            next_round: 0,
        })
    }

    /// Withholds any side that would raise `q' Sigma q` above `limit`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `limit` is not positive.
    pub fn with_max_portfolio_variance(mut self, limit: Decimal) -> MMResult<Self> {
        if limit <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "portfolio variance limit must be positive, got {limit}"
            )));
        }
        self.max_portfolio_variance = Some(limit);
        Ok(self)
    }

    /// The model.
    #[must_use]
    pub fn model(&self) -> &MultiAssetModel {
        &self.model
    }

    /// The capital budget, in the quote asset: the denominator of the capital
    /// utilization.
    #[must_use]
    pub fn capital_budget(&self) -> Decimal {
        self.capital_budget
    }

    /// The instruments, one per asset.
    #[must_use]
    pub fn instruments(&self) -> &[InstrumentId] {
        &self.instruments
    }

    fn spots(&self, gate: &RiskGate) -> MMResult<Vec<SpotView>> {
        let mut spots: Vec<SpotView> = Vec::with_capacity(self.instruments.len());
        for id in &self.instruments {
            let spot = match gate.ledger().instrument(id) {
                Some(Instrument::Spot(spot)) => spot,
                Some(_) => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "instrument {} is not a spot instrument",
                        id.as_str()
                    )));
                }
                None => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "instrument {} is not registered in the ledger",
                        id.as_str()
                    )));
                }
            };
            if let Some(first) = spots.first()
                && first.quote != *spot.quote()
            {
                return Err(MMError::InvalidConfiguration(format!(
                    "instrument {} does not share the quote asset of the budget",
                    id.as_str()
                )));
            }
            if spots.iter().any(|other| other.base == *spot.base()) {
                return Err(MMError::InvalidConfiguration(format!(
                    "instrument {} repeats a base asset",
                    id.as_str()
                )));
            }
            let limits = gate.limits(id);
            spots.push(SpotView {
                id: id.clone(),
                base: spot.base().clone(),
                quote: spot.quote().clone(),
                price_scale: spot.price_scale(),
                quantity_scale: spot.quantity_scale(),
                fee_rate: limits.max_fee_rate,
                fee_asset: limits.fee_asset,
            });
        }
        Ok(spots)
    }

    fn check_marks(&self, marks: &[Decimal]) -> MMResult<()> {
        if marks.len() != self.instruments.len() {
            return Err(MMError::InvalidMarketState(format!(
                "{} reference prices for {} assets",
                marks.len(),
                self.instruments.len()
            )));
        }
        match marks.iter().find(|mark| **mark <= Decimal::ZERO) {
            Some(mark) => Err(MMError::InvalidMarketState(format!(
                "reference price must be positive, got {mark}"
            ))),
            None => Ok(()),
        }
    }

    /// The capital the gate holds, read from its ledger: reserved quote, plus
    /// reserved base valued at `marks`, one per asset.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if `marks` does not have one
    /// positive price per asset, [`MMError::InvalidConfiguration`] if the
    /// instruments are not spot instruments with one common quote asset and
    /// distinct base assets, and [`MMError::NumericalError`] on overflow.
    pub fn reserved_capital(&self, gate: &RiskGate, marks: &[Decimal]) -> MMResult<Decimal> {
        self.check_marks(marks)?;
        Self::held_capital(gate, &self.spots(gate)?, marks)
    }

    fn balance_reserved(gate: &RiskGate, asset: &Asset) -> MMResult<Decimal> {
        gate.ledger()
            .balance(asset)
            .map(|balance| balance.reserved)
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!("asset {asset} is not registered"))
            })
    }

    fn held_capital(gate: &RiskGate, spots: &[SpotView], marks: &[Decimal]) -> MMResult<Decimal> {
        let mut held = match spots.first() {
            Some(first) => Self::balance_reserved(gate, &first.quote)?,
            None => Decimal::ZERO,
        };
        for (spot, mark) in spots.iter().zip(marks) {
            let value = Self::balance_reserved(gate, &spot.base)?
                .checked_mul(*mark)
                .or_overflow("reserved base value")?;
            held = held.checked_add(value).or_overflow("reserved capital")?;
        }
        Ok(held)
    }

    /// Runs one round at `marks`, the reference price per asset, and submits
    /// what is left of it to `gate`.
    ///
    /// A rejection is not an error: it is logged in the round. An order that
    /// would take the capital past the budget is released through the gate.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if `marks` does not have one
    /// positive price per asset. Returns [`MMError::InvalidConfiguration`] if
    /// the instruments are not registered spot instruments with one common
    /// quote asset and distinct base assets, or a trade size is finer than the
    /// lot of its instrument. Returns [`MMError::NumericalError`] on overflow,
    /// and [`MMError::InvalidQuoteGeneration`] if the gate refuses an order id
    /// or the release of an order over budget.
    pub fn quote(
        &mut self,
        gate: &mut RiskGate,
        marks: &[Decimal],
        now_ms: u64,
    ) -> MMResult<QuoteRound> {
        self.check_marks(marks)?;
        let spots = self.spots(gate)?;
        let mut constraints = Vec::new();
        let candidates = self.candidates(gate, &spots, marks, &mut constraints)?;
        let sized = self.sized(gate, &spots, marks, candidates, &mut constraints)?;

        let round = self.next_round;
        self.next_round = self.next_round.saturating_add(1);
        let mut quotes = Vec::with_capacity(sized.len());
        for candidate in sized {
            let order_id = Self::order_id(round, &candidate)?;
            let Some(spot) = spots.get(candidate.asset) else {
                continue;
            };
            let request = OrderRequest::new(
                order_id.clone(),
                spot.id.clone(),
                Action::from(candidate.side),
                candidate.price,
                candidate.quantity,
            )
            .reducible()
            .with_purpose(OrderPurpose::Quote);
            let approved = match gate.submit(&request, now_ms) {
                RiskDecision::Accepted { quantity } => quantity,
                RiskDecision::Reduced {
                    requested,
                    quantity,
                    reasons,
                } => {
                    constraints.push(ActiveConstraint {
                        asset: candidate.asset,
                        side: Action::from(candidate.side),
                        kind: ConstraintKind::MarginReduced {
                            requested,
                            approved: quantity,
                            reasons,
                        },
                    });
                    quantity
                }
                RiskDecision::Rejected { reasons } => {
                    constraints.push(ActiveConstraint {
                        asset: candidate.asset,
                        side: Action::from(candidate.side),
                        kind: ConstraintKind::MarginRejected { reasons },
                    });
                    continue;
                }
            };
            if Self::held_capital(gate, &spots, marks)? > self.capital_budget {
                gate.on_venue_rejected(&order_id).map_err(|error| {
                    MMError::InvalidQuoteGeneration(format!(
                        "cannot release order {order_id} over budget: {error}"
                    ))
                })?;
                constraints.push(ActiveConstraint {
                    asset: candidate.asset,
                    side: Action::from(candidate.side),
                    kind: ConstraintKind::CapitalBudget {
                        requested: approved,
                        allowed: Decimal::ZERO,
                        budget: self.capital_budget,
                    },
                });
                continue;
            }
            quotes.push(AdmittedQuote {
                asset: candidate.asset,
                side: Action::from(candidate.side),
                order_id,
                price: candidate.price,
                quantity: approved,
            });
        }
        Ok(QuoteRound {
            quotes,
            constraints,
            reserved_capital: Self::held_capital(gate, &spots, marks)?,
        })
    }

    fn order_id(round: u64, candidate: &Candidate) -> MMResult<OrderId> {
        let side = match candidate.side {
            Dir::Buy => "b",
            Dir::Sell => "a",
        };
        OrderId::new(format!("ma-{round}-{}-{side}", candidate.asset))
            .map_err(|error| MMError::InvalidQuoteGeneration(format!("invalid order id: {error}")))
    }

    /// Stages 1 and 2: the sides that may quote, with their full size.
    fn candidates(
        &self,
        gate: &RiskGate,
        spots: &[SpotView],
        marks: &[Decimal],
        constraints: &mut Vec<ActiveConstraint>,
    ) -> MMResult<Vec<Candidate>> {
        let mut inventory = Vec::with_capacity(spots.len());
        for spot in spots {
            let quantity = gate
                .ledger()
                .position(&spot.id)
                .map_or(Decimal::ZERO, |position| position.quantity);
            inventory.push(quantity);
        }
        let variance_now = self.model.portfolio_variance(&inventory)?;

        let mut candidates = Vec::with_capacity(spots.len().saturating_mul(2));
        for (asset, (spot, mark)) in spots.iter().zip(marks).enumerate() {
            let params = self.model.asset(asset).ok_or_else(|| {
                MMError::InvalidConfiguration(format!("asset {asset} is out of range"))
            })?;
            let size = params.trade_size();
            let lots = size.round_dp_with_strategy(spot.quantity_scale, RoundingStrategy::ToZero);
            if lots != size {
                return Err(MMError::InvalidConfiguration(format!(
                    "trade size {size} is finer than the lot of {}",
                    spot.id.as_str()
                )));
            }
            let offsets = self.model.offsets(asset, &inventory)?;
            let bid = mark
                .checked_sub(offsets.bid)
                .or_overflow("bid price")?
                .round_dp_with_strategy(spot.price_scale, RoundingStrategy::ToNegativeInfinity);
            let ask = mark
                .checked_add(offsets.ask)
                .or_overflow("ask price")?
                .round_dp_with_strategy(spot.price_scale, RoundingStrategy::ToPositiveInfinity);
            for (side, price) in [(Dir::Buy, bid), (Dir::Sell, ask)] {
                let mut push = |kind| {
                    constraints.push(ActiveConstraint {
                        asset,
                        side: Action::from(side),
                        kind,
                    });
                };
                if price <= Decimal::ZERO {
                    push(ConstraintKind::NonPositivePrice { price });
                    continue;
                }
                let held = inventory.get(asset).copied().unwrap_or(Decimal::ZERO);
                let resulting = match side {
                    Dir::Sell => held.checked_sub(size),
                    Dir::Buy => held.checked_add(size),
                }
                .or_overflow("inventory after the trade")?;
                if resulting.abs() > params.inventory_limit() {
                    push(ConstraintKind::InventoryLimit {
                        resulting,
                        limit: params.inventory_limit(),
                    });
                    continue;
                }
                if let Some(limit) = self.max_portfolio_variance {
                    let mut after = inventory.clone();
                    if let Some(slot) = after.get_mut(asset) {
                        *slot = resulting;
                    }
                    let variance = self.model.portfolio_variance(&after)?;
                    if variance > limit && variance > variance_now {
                        push(ConstraintKind::PortfolioRisk { variance, limit });
                        continue;
                    }
                }
                candidates.push(Candidate {
                    asset,
                    side,
                    price,
                    quantity: size,
                    unit_capital: Self::unit_capital(spot, side, price, *mark)?,
                });
            }
        }
        Ok(candidates)
    }

    /// Capital one unit of an order is expected to hold, before rounding.
    fn unit_capital(
        spot: &SpotView,
        side: Dir,
        price: Decimal,
        mark: Decimal,
    ) -> MMResult<Decimal> {
        let (value, fee_in_reservation) = match side {
            Dir::Sell => (mark, spot.fee_asset == FeeAsset::Base),
            Dir::Buy => (price, spot.fee_asset == FeeAsset::Quote),
        };
        if !fee_in_reservation {
            return Ok(value);
        }
        let loaded = Decimal::ONE
            .checked_add(spot.fee_rate)
            .or_overflow("fee loading")?;
        value.checked_mul(loaded).or_overflow("unit capital")
    }

    /// Stage 3: one scale for every side, from the budget left.
    fn sized(
        &self,
        gate: &RiskGate,
        spots: &[SpotView],
        marks: &[Decimal],
        candidates: Vec<Candidate>,
        constraints: &mut Vec<ActiveConstraint>,
    ) -> MMResult<Vec<Candidate>> {
        let held = Self::held_capital(gate, spots, marks)?;
        let left = self
            .capital_budget
            .checked_sub(held)
            .or_overflow("capital left")?
            .max(Decimal::ZERO);
        let mut demand = Decimal::ZERO;
        for candidate in &candidates {
            let capital = candidate
                .unit_capital
                .checked_mul(candidate.quantity)
                .or_overflow("capital demand")?;
            demand = demand.checked_add(capital).or_overflow("capital demand")?;
        }
        if demand <= left {
            return Ok(candidates);
        }
        let scale = left.checked_div(demand).or_overflow("capital scale")?;
        let mut sized = Vec::with_capacity(candidates.len());
        for mut candidate in candidates {
            let lot_scale = spots
                .get(candidate.asset)
                .map_or(0, |spot| spot.quantity_scale);
            let allowed = candidate
                .quantity
                .checked_mul(scale)
                .or_overflow("scaled quantity")?
                .round_dp_with_strategy(lot_scale, RoundingStrategy::ToZero);
            if allowed < candidate.quantity {
                constraints.push(ActiveConstraint {
                    asset: candidate.asset,
                    side: Action::from(candidate.side),
                    kind: ConstraintKind::CapitalBudget {
                        requested: candidate.quantity,
                        allowed,
                        budget: self.capital_budget,
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
}
