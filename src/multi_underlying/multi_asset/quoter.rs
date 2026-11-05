//! The only way from the model to a quote: through a `margin-rs` risk gate.

use std::fmt;

use margin_rs::risk::{FeeAsset, OrderPurpose, OrderRequest, RiskDecision, RiskGate, RiskReason};
use margin_rs::{Asset, Instrument, InstrumentId, OrderId};
use rust_decimal::RoundingStrategy;

use super::model::{AssetQuotingParams, MultiAssetModel};
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
    /// With every live order on this side filled, a trade would take the
    /// inventory beyond the limit of the model; the side is withheld. A side
    /// that moves an inventory beyond the limit back towards it is not.
    InventoryLimit {
        /// The worst case inventory after the trade.
        resulting: Decimal,
        /// The limit on its absolute value.
        limit: Decimal,
    },
    /// Some way for the live orders and this side to fill could take the
    /// portfolio variance beyond the limit; the side is withheld. A side that
    /// cannot raise the variance in any such state is never withheld for this.
    PortfolioRisk {
        /// Upper bound of `q' Sigma q` over those states, under the model's view.
        variance: Decimal,
        /// The limit.
        limit: Decimal,
    },
    /// The quote stands, on the far side of the reference price, as the affine
    /// formula gives at a large inventory: it would trade at once against that
    /// price. Reported, not withheld.
    ThroughReference {
        /// The quote price.
        price: Decimal,
        /// The reference price.
        reference: Decimal,
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
            ConstraintKind::ThroughReference { price, reference } => {
                write!(f, "price {price} is through the reference {reference}")
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

/// The capital of an account at some marks, in the quote asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CapitalUse {
    /// Inventory at the marks: `sum |q_i| mark_i`. A long paid for it in quote,
    /// and a short draws it from a deposit of the base asset.
    pub deployed: Decimal,
    /// What the gate's ledger holds for live orders: reserved quote, plus reserved
    /// base at the marks.
    pub reserved: Decimal,
    /// `deployed + reserved`, less the reservations that only unwind inventory: a
    /// sell of base already held long and a buy that covers a short commit nothing
    /// new. This is what the capital budget limits.
    pub committed: Decimal,
}

/// What one round produced: the quotes standing, and the cause of every side
/// changed, withheld or flagged. The crate does not print: this is the log, for
/// the caller to keep or forward.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QuoteRound {
    /// The approved quotes, by asset, bid before ask.
    pub quotes: Vec<AdmittedQuote>,
    /// Every active constraint, in the order the stages met them.
    pub constraints: Vec<ActiveConstraint>,
    /// The capital after the round.
    pub capital: CapitalUse,
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
    /// The part of `quantity` that only unwinds inventory, and commits nothing.
    covered: Decimal,
}

/// The live orders of one instrument.
#[derive(Debug, Clone, Copy, Default)]
struct LiveOrders {
    buys: Decimal,
    sells: Decimal,
    lowest_bid: Option<Decimal>,
}

impl LiveOrders {
    fn add(&mut self, side: Dir, price: Decimal, quantity: Decimal) -> MMResult<()> {
        match side {
            Dir::Sell => {
                self.sells = self.sells.checked_add(quantity).or_overflow("live sells")?;
            }
            Dir::Buy => {
                self.buys = self.buys.checked_add(quantity).or_overflow("live buys")?;
                self.lowest_bid = Some(self.lowest_bid.map_or(price, |bid| bid.min(price)));
            }
        }
        Ok(())
    }
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
///    variance limit, both on what can still happen and not only on the current
///    inventory: orders fill independently, so every live order and every side
///    already admitted this round may fill or not, in any combination. A breach
///    withholds the side. A side that moves the inventory back towards its limit,
///    or that cannot raise the variance in any of those states, still quotes.
/// 3. **Size.** All remaining sides share the capital budget left after what is
///    already committed ([`CapitalUse::committed`]): when they ask for more, each
///    is scaled by the same factor, rounded down to a lot. A side rounded to
///    nothing is withheld. The part of a side that only unwinds inventory commits
///    nothing, so it is not scaled, and stays when the budget is spent.
/// 4. **Hard rejection.** Each side is submitted to the gate, which reserves
///    for it and may reduce or reject it. Nothing can bypass this stage, and a
///    quote exists only if the gate approved it. If rounding in a reservation
///    takes the committed capital past the budget, the order is released.
///
/// A round never raises the committed capital above the budget, with one
/// exception: the fee an unwinding order reserves, which is kept so that an account
/// over budget can still reduce. Marks can also pass the budget, by revaluing the
/// inventory, and the next round then has less left to commit. An account already
/// over budget can only reduce.
///
/// The variance limit is conservative. The largest variance over those states
/// is a hard problem in many assets, so it uses a bound that is exact when one
/// asset can move and otherwise can exceed the true maximum, withholding a side
/// that would have stayed under the limit: with `q = c + r u` and `|u_i| <= 1`,
/// `q' Sigma q <= c' Sigma c + 2 sum |(Sigma c)_i| r_i + (sum r_i sigma_i)^2`.
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
/// and the ledger's net traded quantity is the inventory the model sees. The
/// deposit itself is outside the model: it carries price risk that the quotes do
/// not skew for and the limits do not see.
///
/// Order ids are `ma-<round>-<asset>-<side>`. A round number the gate has already
/// seen is skipped, so a new quoter on a used gate, after a restart for instance,
/// does not collide with the orders of an earlier one.
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
    /// A quoter for `instruments`, one per asset of `model`, in its order, under
    /// `capital_budget`, in the quote asset, on [`CapitalUse::committed`].
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

    /// Withholds any side that could raise `q' Sigma q` above `limit`, in any
    /// combination of fills of the live orders and the sides already admitted. The
    /// check uses the conservative bound documented on the type, and a side that
    /// cannot raise the variance in any such state is exempt.
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

    /// The capital of the account at `marks`, one per asset, read from the gate.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] if `marks` does not have one
    /// positive price per asset, [`MMError::InvalidConfiguration`] if the
    /// instruments are not spot instruments with one common quote asset and
    /// distinct base assets, [`MMError::NumericalError`] on overflow, and
    /// [`MMError::InvalidQuoteGeneration`] if an order record cannot be read.
    pub fn capital_use(&self, gate: &RiskGate, marks: &[Decimal]) -> MMResult<CapitalUse> {
        self.check_marks(marks)?;
        let spots = self.spots(gate)?;
        let inventory = Self::inventory(gate, &spots);
        let live = Self::live_orders(gate, &spots)?;
        Self::capital(gate, &spots, marks, &inventory, &live)
    }

    fn inventory(gate: &RiskGate, spots: &[SpotView]) -> Vec<Decimal> {
        spots
            .iter()
            .map(|spot| {
                gate.ledger()
                    .position(&spot.id)
                    .map_or(Decimal::ZERO, |position| position.quantity)
            })
            .collect()
    }

    /// The live orders per instrument. Orders with a cancel pending count: they
    /// may still fill, and they keep their reservations.
    fn live_orders(gate: &RiskGate, spots: &[SpotView]) -> MMResult<Vec<LiveOrders>> {
        let mut live = vec![LiveOrders::default(); spots.len()];
        for order in gate.live_orders() {
            let Some(asset) = spots.iter().position(|spot| spot.id == order.instrument) else {
                continue;
            };
            let remaining = order.remaining().map_err(|error| {
                MMError::InvalidQuoteGeneration(format!("cannot read order {}: {error}", order.id))
            })?;
            let Some(slot) = live.get_mut(asset) else {
                continue;
            };
            slot.add(Dir::try_from(order.side)?, order.price, remaining)?;
        }
        Ok(live)
    }

    fn capital(
        gate: &RiskGate,
        spots: &[SpotView],
        marks: &[Decimal],
        inventory: &[Decimal],
        live: &[LiveOrders],
    ) -> MMResult<CapitalUse> {
        let reserved = Self::held_capital(gate, spots, marks)?;
        let mut deployed = Decimal::ZERO;
        let mut covered = Decimal::ZERO;
        for ((held, mark), orders) in inventory.iter().zip(marks).zip(live) {
            let value = held
                .abs()
                .checked_mul(*mark)
                .or_overflow("deployed capital")?;
            deployed = deployed
                .checked_add(value)
                .or_overflow("deployed capital")?;
            // Valued at or below what the gate reserved, so never over-subtracted.
            let long = (*held).max(Decimal::ZERO);
            let short = Decimal::ZERO
                .checked_sub(*held)
                .or_overflow("short inventory")?
                .max(Decimal::ZERO);
            let sold = orders
                .sells
                .min(long)
                .checked_mul(*mark)
                .or_overflow("covered capital")?;
            let bought = orders
                .buys
                .min(short)
                .checked_mul(orders.lowest_bid.unwrap_or(Decimal::ZERO))
                .or_overflow("covered capital")?;
            covered = covered
                .checked_add(sold)
                .and_then(|sum| sum.checked_add(bought))
                .or_overflow("covered capital")?;
        }
        let committed = deployed
            .checked_add(reserved)
            .and_then(|sum| sum.checked_sub(covered))
            .or_overflow("committed capital")?;
        Ok(CapitalUse {
            deployed,
            reserved,
            committed,
        })
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
        let inventory = Self::inventory(gate, &spots);
        // Listing the live orders scans every order the gate has seen, and nothing
        // fills during a round: read them once and keep this copy up to date.
        let mut live = Self::live_orders(gate, &spots)?;
        let mut constraints = Vec::new();
        let candidates = self.candidates(&spots, marks, &inventory, &live, &mut constraints)?;
        let committed = Self::capital(gate, &spots, marks, &inventory, &live)?.committed;
        let sized = self.sized(&spots, committed, candidates, &mut constraints)?;

        let round = self.free_round(gate, &sized)?;
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
            let mut with_order = live.clone();
            if let Some(slot) = with_order.get_mut(candidate.asset) {
                slot.add(candidate.side, candidate.price, approved)?;
            }
            let committed = Self::capital(gate, &spots, marks, &inventory, &with_order)?.committed;
            if committed > self.capital_budget && candidate.covered < approved {
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
            live = with_order;
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
            capital: Self::capital(gate, &spots, marks, &inventory, &live)?,
        })
    }

    /// The next round whose order ids the gate has not seen, for these sides.
    fn free_round(&mut self, gate: &RiskGate, sized: &[Candidate]) -> MMResult<u64> {
        loop {
            let round = self.next_round;
            self.next_round = round.saturating_add(1);
            let mut used = false;
            for candidate in sized {
                used = used || gate.order(&Self::order_id(round, candidate)?).is_some();
            }
            // At the end of the range the ids cannot be told apart any more, and
            // the gate rejects the duplicates, which the round then logs.
            if !used || round == u64::MAX {
                return Ok(round);
            }
        }
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
        spots: &[SpotView],
        marks: &[Decimal],
        inventory: &[Decimal],
        live: &[LiveOrders],
        constraints: &mut Vec<ActiveConstraint>,
    ) -> MMResult<Vec<Candidate>> {
        let (mut low, mut high) = Self::reachable(inventory, live)?;

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
            let offsets = self.model.offsets(asset, inventory)?;
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
                let reach = match side {
                    Dir::Sell => low.get_mut(asset),
                    Dir::Buy => high.get_mut(asset),
                }
                .ok_or_else(|| {
                    MMError::InvalidConfiguration(format!("asset {asset} is out of range"))
                })?;
                let before = *reach;
                let (resulting, breach) = match side {
                    Dir::Sell => {
                        let after = before
                            .checked_sub(size)
                            .or_overflow("worst case inventory")?;
                        let floor = Decimal::ZERO
                            .checked_sub(params.inventory_limit())
                            .or_overflow("inventory limit")?;
                        (after, after < floor)
                    }
                    Dir::Buy => {
                        let after = before
                            .checked_add(size)
                            .or_overflow("worst case inventory")?;
                        (after, after > params.inventory_limit())
                    }
                };
                if breach {
                    push(ConstraintKind::InventoryLimit {
                        resulting,
                        limit: params.inventory_limit(),
                    });
                    continue;
                }
                *reach = resulting;
                if let Some(limit) = self.max_portfolio_variance
                    && !self.cannot_raise_variance(asset, side, &low, &high)?
                {
                    let variance = self.variance_bound(&low, &high)?;
                    if variance > limit {
                        push(ConstraintKind::PortfolioRisk { variance, limit });
                        let reach = match side {
                            Dir::Sell => low.get_mut(asset),
                            Dir::Buy => high.get_mut(asset),
                        };
                        if let Some(reach) = reach {
                            *reach = before;
                        }
                        continue;
                    }
                }
                let through = match side {
                    Dir::Sell => price <= *mark,
                    Dir::Buy => price >= *mark,
                };
                if through {
                    push(ConstraintKind::ThroughReference {
                        price,
                        reference: *mark,
                    });
                }
                let held = inventory.get(asset).copied().unwrap_or(Decimal::ZERO);
                let orders = live.get(asset).copied().unwrap_or_default();
                let unwinding = match side {
                    Dir::Sell => held.checked_sub(orders.sells),
                    Dir::Buy => held
                        .checked_add(orders.buys)
                        .and_then(|net| Decimal::ZERO.checked_sub(net)),
                }
                .or_overflow("inventory left to unwind")?
                .max(Decimal::ZERO);
                candidates.push(Candidate {
                    asset,
                    side,
                    price,
                    quantity: size,
                    unit_capital: Self::unit_capital(spot, side, price, *mark)?,
                    covered: unwinding.min(size),
                });
            }
        }
        Ok(candidates)
    }

    /// The lowest and highest inventory per asset if the live orders fill in any
    /// combination.
    fn reachable(
        inventory: &[Decimal],
        live: &[LiveOrders],
    ) -> MMResult<(Vec<Decimal>, Vec<Decimal>)> {
        let mut low = Vec::with_capacity(inventory.len());
        let mut high = Vec::with_capacity(inventory.len());
        for (held, orders) in inventory.iter().zip(live) {
            low.push(
                held.checked_sub(orders.sells)
                    .or_overflow("worst case inventory")?,
            );
            high.push(
                held.checked_add(orders.buys)
                    .or_overflow("worst case inventory")?,
            );
        }
        Ok((low, high))
    }

    /// The covariance entry, which exists for assets of the model.
    fn entry(&self, row: usize, column: usize) -> MMResult<Decimal> {
        self.model.covariance(row, column).ok_or_else(|| {
            MMError::InvalidConfiguration(format!("asset {row} or {column} is out of range"))
        })
    }

    /// The center of the box of reachable inventories and its half width.
    fn center_and_radius(
        low: &[Decimal],
        high: &[Decimal],
    ) -> MMResult<(Vec<Decimal>, Vec<Decimal>)> {
        let mut center = Vec::with_capacity(low.len());
        let mut radius = Vec::with_capacity(low.len());
        for (bottom, top) in low.iter().zip(high) {
            let sum = top.checked_add(*bottom).or_overflow("inventory center")?;
            let width = top.checked_sub(*bottom).or_overflow("inventory width")?;
            center.push(
                sum.checked_div(Decimal::TWO)
                    .or_overflow("inventory center")?,
            );
            radius.push(
                width
                    .checked_div(Decimal::TWO)
                    .or_overflow("inventory width")?,
            );
        }
        Ok((center, radius))
    }

    /// `(Sigma c)_row`.
    fn risk_gradient(&self, row: usize, center: &[Decimal]) -> MMResult<Decimal> {
        let mut gradient = Decimal::ZERO;
        for (column, position) in center.iter().enumerate() {
            let term = self
                .entry(row, column)?
                .checked_mul(*position)
                .or_overflow("risk gradient")?;
            gradient = gradient.checked_add(term).or_overflow("risk gradient")?;
        }
        Ok(gradient)
    }

    /// Whether a fill on this side lowers or keeps `q' Sigma q` from every
    /// reachable inventory, this side's own fill included. The derivative along
    /// the side is `2 s (Sigma q)_i`, linear in `q`, so its largest value over the
    /// box is `s (Sigma c)_i + sum |Sigma_ij| r_j`. When that is not positive,
    /// admitting the side cannot raise the largest reachable variance.
    fn cannot_raise_variance(
        &self,
        asset: usize,
        side: Dir,
        low: &[Decimal],
        high: &[Decimal],
    ) -> MMResult<bool> {
        let (center, radius) = Self::center_and_radius(low, high)?;
        let gradient = self.risk_gradient(asset, &center)?;
        let mut largest = match side {
            Dir::Sell => Decimal::ZERO
                .checked_sub(gradient)
                .or_overflow("risk gradient")?,
            Dir::Buy => gradient,
        };
        for (column, width) in radius.iter().enumerate() {
            let term = self
                .entry(asset, column)?
                .abs()
                .checked_mul(*width)
                .or_overflow("risk gradient bound")?;
            largest = largest
                .checked_add(term)
                .or_overflow("risk gradient bound")?;
        }
        Ok(largest <= Decimal::ZERO)
    }

    /// The bound of `q' Sigma q` over the box, documented on the type.
    fn variance_bound(&self, low: &[Decimal], high: &[Decimal]) -> MMResult<Decimal> {
        let (center, radius) = Self::center_and_radius(low, high)?;
        let mut bound = self.model.portfolio_variance(&center)?;
        let mut spread = Decimal::ZERO;
        for (asset, width) in radius.iter().enumerate() {
            let linear = self
                .risk_gradient(asset, &center)?
                .abs()
                .checked_mul(*width)
                .and_then(|term| term.checked_mul(Decimal::TWO))
                .or_overflow("variance bound")?;
            bound = bound.checked_add(linear).or_overflow("variance bound")?;
            let volatility = self
                .model
                .asset(asset)
                .map_or(Decimal::ZERO, AssetQuotingParams::volatility);
            let term = width
                .checked_mul(volatility)
                .or_overflow("variance bound")?;
            spread = spread.checked_add(term).or_overflow("variance bound")?;
        }
        let quadratic = spread.checked_mul(spread).or_overflow("variance bound")?;
        bound.checked_add(quadratic).or_overflow("variance bound")
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
        spots: &[SpotView],
        committed: Decimal,
        candidates: Vec<Candidate>,
        constraints: &mut Vec<ActiveConstraint>,
    ) -> MMResult<Vec<Candidate>> {
        let left = self
            .capital_budget
            .checked_sub(committed)
            .or_overflow("capital left")?
            .max(Decimal::ZERO);
        let mut demand = Decimal::ZERO;
        for candidate in &candidates {
            let capital = candidate
                .quantity
                .checked_sub(candidate.covered)
                .and_then(|uncovered| uncovered.checked_mul(candidate.unit_capital))
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
                .checked_sub(candidate.covered)
                .and_then(|uncovered| uncovered.checked_mul(scale))
                .and_then(|scaled| scaled.checked_add(candidate.covered))
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
