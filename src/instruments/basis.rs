//! The net edge of a basis trade between a spot and a linear dated future.
//!
//! [`evaluate_basis`] buys one leg at its ask and sells the other at its bid. The
//! gross edge is `(bid - ask) * quantity`, computed by `arbitrage-rs`, and it is an
//! edge only because the legs converge: a dated future converges to its settlement
//! price at expiry, which [`BasisOpportunity::converges_at_ms`] reports. Fees, exit
//! costs and carry are subtracted to get the net edge.
//!
//! Everything here is an estimate at quoted prices. An accepted
//! [`BasisOpportunity`] is an intent, not an executed or guaranteed profit: the legs
//! can fill partially or not at all, and a residual remains at expiry, because a
//! settlement price is not this venue's spot price.
//!
//! # Scope
//!
//! One spot and one dated future, linear, settled in their common price asset, on
//! the same underlying. Refused with a [`BasisError`]:
//!
//! - an inverse contract, whose PnL is not linear in the price;
//! - a perpetual, which pays funding;
//! - an option;
//! - two futures: they do not converge to each other, because when the near one
//!   settles the far one is still open at an unknown spread, which
//!   [`evaluate_calendar`](super::calendar::evaluate_calendar) takes as an assumption;
//! - selling spot without a declared [`CarryKind::Borrow`] on that leg, even if of
//!   zero: otherwise the cheapest input gives the most attractive answer.
//!
//! # Rates and units
//!
//! Every rate in `_bps` is the caller's, not a venue's, in basis points, within
//! `[-10000, 10000]`, and negative is a rebate or a benefit. It is charged on the
//! leg's entry notional. A
//! [`BasisCarry`] is for the whole period up to the convergence: the caller converts
//! an annual rate, and nothing here annualizes or infers a calendar. An exit cost is
//! required per leg, even if zero, as the exit price is not known and the entry
//! notional stands in for it.
//!
//! Sizes are in the underlying. Liquidity is quantized down to the coarser lot
//! exposure of the two legs (`lot * multiplier`), which is the only rounding here;
//! each leg then falls exactly on its own grid. Two lot exposures where neither
//! divides the other are refused: that is a restriction of this function, not a
//! claim that they have no common multiple.
//!
//! [`BasisOpportunity::capital`] is the spot leg's whole notional plus the future's
//! notional times a rate the caller gives. It is an estimate by the caller, not a
//! venue's validated margin.
//!
//! # Rejected, or refused
//!
//! A [`BasisRejection`] is what the market says, for a caller scanning spreads: an
//! expired contract, a missing, stale or off tick quote, too little size, or too
//! little edge. A [`BasisError`] is a request that cannot be answered. A quote from
//! after `as_of_ms` is one: it is lookahead or a clock fault, not a market state.

use arbitrage_rs::opportunity::{
    self, Carry, CarryCost, CostBreakdown, Evaluation, LegQuote, OpportunityInput, RejectReason,
};
use arbitrage_rs::{
    ArbitrageError, Cost, Currency, FeeRate, Instrument, Level, Money, Price, Quantity, Timestamp,
    TopOfBook,
};
use financial_types::Action;
use rust_decimal::Decimal;

use super::{AssetCode, ContractKind, InstrumentSpec, Margining};

/// What a [`BasisCarry`] pays for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CarryKind {
    /// Financing of a long position.
    Financing,
    /// Borrow fee for the asset sold short.
    Borrow,
    /// Any other holding cost or benefit.
    Other,
}

/// A holding cost on one leg's notional, for the whole period to the convergence, or
/// to the exit of a [calendar](super::calendar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasisCarry {
    /// What it pays for.
    pub kind: CarryKind,
    /// The leg it is charged on: [`Action::Buy`] or [`Action::Sell`]. Any other
    /// action is a [`BasisError::CarryOnNeitherLeg`].
    pub on: Action,
    /// The rate for the whole period, in basis points of that notional.
    pub period_bps: Decimal,
}

/// The executable level of one leg: the ask of a buy, or the bid of a sell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasisLevel {
    /// The price.
    pub price: Decimal,
    /// The size quoted at it, in contracts.
    pub contracts: Decimal,
}

/// One leg's contract and quote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisQuote {
    /// The contract.
    pub spec: InstrumentSpec,
    /// Its executable level, or `None` if that side is not quoted.
    pub level: Option<BasisLevel>,
    /// When it was quoted, in milliseconds since the epoch.
    pub quoted_at_ms: u64,
    /// The entry fee, in basis points of the notional.
    pub fee_bps: Decimal,
    /// The cost of leaving the leg, in basis points of the entry notional: the
    /// fee to unwind a leg, or a settlement fee.
    pub exit_bps: Decimal,
}

/// Input to [`evaluate_basis`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisInput {
    /// The time of the decision, in milliseconds since the epoch.
    pub as_of_ms: u64,
    /// The leg to buy.
    pub buy: BasisQuote,
    /// The leg to sell.
    pub sell: BasisQuote,
    /// The size wanted, in the underlying.
    pub target_quantity: Decimal,
    /// The smallest size worth trading, in the underlying.
    pub min_quantity: Decimal,
    /// The oldest quote to accept: a quote of exactly this age is fresh.
    pub max_quote_age_ms: u64,
    /// The future's capital as a fraction of its notional, within `[0, 1]`.
    pub future_margin_rate: Decimal,
    /// The holding costs.
    pub carry: Vec<BasisCarry>,
    /// The smallest net edge to accept, in the price asset: zero or more. An edge
    /// of exactly this much is accepted, if it is above zero.
    pub min_net_edge: Decimal,
}

/// One leg of a [`BasisOpportunity`]: an intent, not a fill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisLegPlan {
    /// The contract's identifier.
    pub instrument: String,
    /// The price the estimate uses.
    pub price: Decimal,
    /// The contracts to trade, exactly on the contract's lot grid.
    pub contracts: Decimal,
}

/// A holding cost of a [`BasisOpportunity`], in the price asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasisCarryCost {
    /// What it pays for.
    pub kind: CarryKind,
    /// The leg it was charged on.
    pub on: Action,
    /// The amount: positive is paid.
    pub amount: Decimal,
}

/// Every cost subtracted from a gross edge, in the price asset: positive is paid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisCosts {
    /// The entry fee of the buy leg.
    pub buy_fee: Decimal,
    /// The entry fee of the sell leg.
    pub sell_fee: Decimal,
    /// The cost of leaving the buy leg.
    pub buy_exit: Decimal,
    /// The cost of leaving the sell leg.
    pub sell_exit: Decimal,
    /// The holding costs, in the order given.
    pub carry: Vec<BasisCarryCost>,
    /// All of them.
    pub total: Decimal,
}

/// An accepted basis trade, with its expected economics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisOpportunity {
    /// The asset of every amount here.
    pub asset: AssetCode,
    /// The size, in the underlying.
    pub quantity: Decimal,
    /// The leg to buy.
    pub buy: BasisLegPlan,
    /// The leg to sell.
    pub sell: BasisLegPlan,
    /// The future's expiry, in milliseconds since the epoch: the end of the period
    /// that a [`BasisCarry`] covers.
    pub converges_at_ms: u64,
    /// `(bid - ask) * quantity`.
    pub gross: Decimal,
    /// What is subtracted from it.
    pub costs: BasisCosts,
    /// `gross - costs.total`.
    pub net: Decimal,
    /// The caller's estimate of the capital: see the [module documentation](self).
    pub capital: Decimal,
}

/// Why [`evaluate_basis`] discarded a trade.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BasisRejection {
    /// The future expired at or before the decision.
    Expired {
        /// The contract.
        instrument: String,
        /// Its expiry.
        expiry_ms: u64,
    },
    /// The side a leg needs is not quoted.
    MissingQuote {
        /// The contract.
        instrument: String,
    },
    /// A quote older than the maximum.
    StaleQuote {
        /// The contract.
        instrument: String,
        /// Its age.
        age_ms: u64,
    },
    /// A quote off the contract's price grid, which cannot be executed under this
    /// specification.
    OffTick {
        /// The contract.
        instrument: String,
        /// The price.
        price: Decimal,
    },
    /// Less size than the minimum is executable on both legs, on both lot grids.
    InsufficientLiquidity {
        /// What is available, in the underlying; it may be zero.
        available: Decimal,
        /// The minimum asked for.
        required: Decimal,
    },
    /// The sell bid is not above the buy ask; in a [calendar](super::calendar), the
    /// entry and the assumed exit leave nothing.
    NoGrossEdge {
        /// The gross edge, zero or negative.
        gross: Decimal,
    },
    /// Costs leave a net edge that is not above zero, or is below the threshold.
    NetEdgeBelowThreshold {
        /// The gross edge.
        gross: Decimal,
        /// What is subtracted from it.
        costs: BasisCosts,
        /// The net edge.
        net: Decimal,
        /// The threshold asked for.
        min_net_edge: Decimal,
    },
}

/// The result of [`evaluate_basis`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisEvaluation {
    /// The trade passed.
    Accepted(BasisOpportunity),
    /// The trade was discarded, with the reason.
    Rejected(BasisRejection),
}

/// A request [`evaluate_basis`] cannot answer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BasisError {
    /// The legs are not one spot and one linear dated future with the same terms, or
    /// in a [calendar](super::calendar), two linear dated futures of different expiries.
    #[error("unsupported basis trade: {reason}")]
    Unsupported {
        /// Why not.
        reason: String,
    },
    /// Selling spot needs a declared borrow.
    #[error("selling spot needs a declared borrow carry on that leg, even of zero")]
    MissingBorrow,
    /// Neither lot exposure divides the other, which this function does not handle.
    #[error("the lot exposures {buy} and {sell} are not nested")]
    LotsNotNested {
        /// The buy leg's `lot * multiplier`.
        buy: Decimal,
        /// The sell leg's `lot * multiplier`.
        sell: Decimal,
    },
    /// A carry on an action that is neither leg.
    #[error("a carry on {on} is neither leg")]
    CarryOnNeitherLeg {
        /// The action.
        on: Action,
    },
    /// A value out of its range.
    #[error("{field} is {value}: {reason}")]
    InvalidValue {
        /// Which one.
        field: &'static str,
        /// What it was.
        value: Decimal,
        /// What it must be.
        reason: &'static str,
    },
    /// A time beyond what the evaluation can represent.
    #[error("{field} is {value}, beyond the largest time supported")]
    TimeOutOfRange {
        /// Which one.
        field: &'static str,
        /// What it was.
        value: u64,
    },
    /// A quote from after the decision: lookahead, or a clock fault.
    #[error("quote of {instrument} at {quoted_at_ms}, after the decision at {as_of_ms}")]
    QuoteFromTheFuture {
        /// The contract.
        instrument: String,
        /// When it was quoted.
        quoted_at_ms: u64,
        /// The time of the decision.
        as_of_ms: u64,
    },
    /// The evaluation itself refused the input, or a result is not exact.
    #[error("basis evaluation: {reason}")]
    Evaluation {
        /// The reason.
        reason: String,
    },
    /// The exit of a [calendar](super::calendar) is at or before the decision, or at
    /// or after the near contract's expiry.
    #[error(
        "an exit at {exit_at_ms} is not after the decision at {as_of_ms} and before the \
         near expiry at {near_expiry_ms}"
    )]
    ExitOutsideWindow {
        /// The exit asked for.
        exit_at_ms: u64,
        /// The time of the decision.
        as_of_ms: u64,
        /// The near contract's expiry.
        near_expiry_ms: u64,
    },
}

/// Private, so that no `arbitrage-rs` type is in this crate's public API.
pub(super) fn upstream(error: ArbitrageError) -> BasisError {
    BasisError::Evaluation {
        reason: error.to_string(),
    }
}

pub(super) fn unsupported<T>(reason: impl Into<String>) -> Result<T, BasisError> {
    Err(BasisError::Unsupported {
        reason: reason.into(),
    })
}

pub(super) fn time(field: &'static str, value: u64) -> Result<Timestamp, BasisError> {
    i64::try_from(value)
        .map(Timestamp::from_millis)
        .map_err(|_| BasisError::TimeOutOfRange { field, value })
}

pub(super) fn bps(field: &'static str, value: Decimal) -> Result<FeeRate, BasisError> {
    FeeRate::from_bps(value).map_err(|_| BasisError::InvalidValue {
        field,
        value,
        reason: "must be within -10000 and 10000 basis points",
    })
}

pub(super) fn positive_quantity(
    field: &'static str,
    value: Decimal,
) -> Result<Quantity, BasisError> {
    Quantity::new(value).map_err(|_| BasisError::InvalidValue {
        field,
        value,
        reason: "must be positive",
    })
}

/// A contract's underlying per contract.
fn multiplier(spec: &InstrumentSpec) -> Decimal {
    spec.size().amount()
}

/// Checks a leg's terms, and returns the future's expiry if it has one.
pub(super) fn check_contract(spec: &InstrumentSpec) -> Result<Option<u64>, BasisError> {
    let id = spec.id().as_str();
    // A linear specification has a size in the underlying, settled in its price asset.
    if spec.margining() != Margining::Linear {
        return unsupported(format!("{id} is not linear"));
    }
    match (spec.kind(), spec.expiry_ms()) {
        (ContractKind::Spot, _) => Ok(None),
        (ContractKind::Future, Some(expiry_ms)) => Ok(Some(expiry_ms)),
        (kind, _) => unsupported(format!("{id} is a {kind:?}")),
    }
}

/// Checks both legs, and returns the future's expiry.
fn check_legs(input: &BasisInput) -> Result<u64, BasisError> {
    let (buy, sell) = (&input.buy.spec, &input.sell.spec);
    let expiry_ms = match (check_contract(buy)?, check_contract(sell)?) {
        (Some(_), Some(_)) => {
            return unsupported("two futures do not converge to each other");
        }
        (None, None) => return unsupported("two spot legs are not a basis"),
        (Some(expiry_ms), None) | (None, Some(expiry_ms)) => expiry_ms,
    };
    if buy.underlying() != sell.underlying() {
        return unsupported("the legs have different underlyings");
    }
    if buy.price_asset() != sell.price_asset() {
        return unsupported("the legs have different price assets");
    }
    let borrow = input
        .carry
        .iter()
        .any(|c| c.kind == CarryKind::Borrow && c.on == Action::Sell);
    if sell.kind() == ContractKind::Spot && !borrow {
        return Err(BasisError::MissingBorrow);
    }
    Ok(expiry_ms)
}

/// The coarser lot exposure, if the other one divides it.
pub(super) fn common_lot(
    buy: &InstrumentSpec,
    sell: &InstrumentSpec,
) -> Result<Quantity, BasisError> {
    let exposure = |spec: &InstrumentSpec| {
        arbitrage_rs::exact::mul(spec.lot(), multiplier(spec))
            .and_then(Quantity::new)
            .map_err(upstream)
    };
    let (a, b) = (exposure(buy)?, exposure(sell)?);
    let (coarse, fine) = if a.value() >= b.value() {
        (a, b)
    } else {
        (b, a)
    };
    if coarse.floor_to_lot(fine).map_err(upstream)? != Some(coarse) {
        return Err(BasisError::LotsNotNested {
            buy: a.value(),
            sell: b.value(),
        });
    }
    Ok(coarse)
}

/// The contracts worth `quantity` of the underlying, exactly.
fn contracts_for(spec: &InstrumentSpec, quantity: Decimal) -> Result<Decimal, BasisError> {
    let contracts =
        quantity
            .checked_div(multiplier(spec))
            .ok_or_else(|| BasisError::Evaluation {
                reason: format!("{quantity} contracts of {} overflow", spec.id().as_str()),
            })?;
    if arbitrage_rs::exact::mul(contracts, multiplier(spec)).map_err(upstream)? != quantity {
        return Err(BasisError::Evaluation {
            reason: format!(
                "{quantity} of the underlying is not exact in contracts of {}",
                spec.id().as_str()
            ),
        });
    }
    Ok(contracts.normalize())
}

pub(super) fn plan(
    spec: &InstrumentSpec,
    price: Decimal,
    quantity: Decimal,
) -> Result<BasisLegPlan, BasisError> {
    let contracts = contracts_for(spec, quantity)?;
    // The quantity is a multiple of the coarser lot exposure, so this cannot fail
    // unless a contract's terms were not checked.
    spec.check_lot(contracts)
        .map_err(|e| BasisError::Evaluation {
            reason: e.to_string(),
        })?;
    Ok(BasisLegPlan {
        instrument: spec.id().as_str().to_string(),
        price,
        contracts,
    })
}

/// Rejects a leg that cannot be executed now, or returns its executable level.
pub(super) fn check_quote(
    quote: &BasisQuote,
    as_of_ms: u64,
    max_quote_age_ms: u64,
) -> Result<Result<BasisLevel, BasisRejection>, BasisError> {
    let instrument = quote.spec.id().as_str().to_string();
    time("quoted_at_ms", quote.quoted_at_ms)?;
    let Some(age_ms) = as_of_ms.checked_sub(quote.quoted_at_ms) else {
        return Err(BasisError::QuoteFromTheFuture {
            instrument,
            quoted_at_ms: quote.quoted_at_ms,
            as_of_ms,
        });
    };
    let Some(level) = quote.level else {
        return Ok(Err(BasisRejection::MissingQuote { instrument }));
    };
    if level.price <= Decimal::ZERO {
        return Err(BasisError::InvalidValue {
            field: "price",
            value: level.price,
            reason: "must be positive",
        });
    }
    if level.contracts < Decimal::ZERO {
        return Err(BasisError::InvalidValue {
            field: "contracts",
            value: level.contracts,
            reason: "must not be negative",
        });
    }
    if age_ms > max_quote_age_ms {
        return Ok(Err(BasisRejection::StaleQuote { instrument, age_ms }));
    }
    if quote.spec.check_price(level.price).is_err() {
        return Ok(Err(BasisRejection::OffTick {
            instrument,
            price: level.price,
        }));
    }
    Ok(Ok(level))
}

fn upstream_instrument(spec: &InstrumentSpec) -> Result<Instrument, BasisError> {
    let id = arbitrage_rs::InstrumentId::new(spec.id().as_str()).map_err(upstream)?;
    Ok(match spec.expiry_ms() {
        Some(expiry_ms) => Instrument::dated(id, time("expiry_ms", expiry_ms)?),
        None => Instrument::undated(id),
    })
}

fn upstream_book(side: Action, level: BasisLevel, size: Quantity) -> Result<TopOfBook, BasisError> {
    let level = Level::new(Price::new(level.price).map_err(upstream)?, size);
    Ok(match side {
        Action::Buy => TopOfBook {
            bid: None,
            ask: Some(level),
        },
        _ => TopOfBook {
            bid: Some(level),
            ask: None,
        },
    })
}

fn amount(cost: Cost) -> Decimal {
    cost.money().amount()
}

/// Our costs from upstream's: the last two carry items are the exit costs.
fn costs_from(breakdown: &CostBreakdown, carried: &[BasisCarry]) -> Result<BasisCosts, BasisError> {
    // The count is exact, as these pair by position with what was given.
    let Some((held, [buy_exit, sell_exit])) = breakdown.carry.split_at_checked(carried.len())
    else {
        return Err(BasisError::Evaluation {
            reason: "the evaluation did not itemize every carry and exit cost".to_string(),
        });
    };
    let exit = |item: &CarryCost| amount(item.cost);
    Ok(BasisCosts {
        buy_fee: amount(breakdown.buy_fee),
        sell_fee: amount(breakdown.sell_fee),
        buy_exit: exit(buy_exit),
        sell_exit: exit(sell_exit),
        carry: held
            .iter()
            .zip(carried)
            .map(|(item, given)| BasisCarryCost {
                kind: given.kind,
                on: given.on,
                amount: amount(item.cost),
            })
            .collect(),
        total: amount(breakdown.total),
    })
}

/// The spot leg's notional, plus the future's at `rate`.
pub(super) fn capital(
    buy: (&InstrumentSpec, Decimal),
    sell: (&InstrumentSpec, Decimal),
    quantity: Quantity,
    rate: Decimal,
    currency: Currency,
) -> Result<Decimal, BasisError> {
    let leg = |(spec, price): (&InstrumentSpec, Decimal)| -> Result<Money, BasisError> {
        let notional = Price::new(price)
            .and_then(|price| price.notional(quantity, currency))
            .map_err(upstream)?;
        if spec.kind() == ContractKind::Spot {
            Ok(notional)
        } else {
            notional.checked_scale(rate).map_err(upstream)
        }
    };
    Ok(leg(buy)?
        .checked_add(leg(sell)?)
        .map_err(upstream)?
        .amount())
}

/// Checks what a request asks for, and returns its target and minimum sizes.
pub(super) fn check_amounts(
    future_margin_rate: Decimal,
    min_net_edge: Decimal,
    target_quantity: Decimal,
    min_quantity: Decimal,
) -> Result<(Quantity, Quantity), BasisError> {
    if future_margin_rate < Decimal::ZERO || future_margin_rate > Decimal::ONE {
        return Err(BasisError::InvalidValue {
            field: "future_margin_rate",
            value: future_margin_rate,
            reason: "must be within 0 and 1",
        });
    }
    if min_net_edge < Decimal::ZERO {
        return Err(BasisError::InvalidValue {
            field: "min_net_edge",
            value: min_net_edge,
            reason: "must not be negative",
        });
    }
    let target = positive_quantity("target_quantity", target_quantity)?;
    let minimum = positive_quantity("min_quantity", min_quantity)?;
    if minimum.value() > target.value() {
        return Err(BasisError::InvalidValue {
            field: "min_quantity",
            value: minimum.value(),
            reason: "must not be above target_quantity",
        });
    }
    Ok((target, minimum))
}

/// Checks the holding costs, in the order given.
pub(super) fn checked_carry(items: &[BasisCarry]) -> Result<Vec<Carry>, BasisError> {
    let mut carry = Vec::with_capacity(items.len());
    for item in items {
        // Here, as a request that cannot be answered must not depend on the market.
        if !matches!(item.on, Action::Buy | Action::Sell) {
            return Err(BasisError::CarryOnNeitherLeg { on: item.on });
        }
        carry.push(Carry {
            kind: match item.kind {
                CarryKind::Financing => opportunity::CarryKind::Financing,
                CarryKind::Borrow => opportunity::CarryKind::Borrow,
                CarryKind::Other => opportunity::CarryKind::Other,
            },
            leg: item.on,
            rate: bps("period_bps", item.period_bps)?,
        });
    }
    Ok(carry)
}

/// What both legs can execute, in the underlying, on both lot grids, or a rejection
/// if that is below the minimum.
pub(super) fn executable_quantity(
    buy: (&InstrumentSpec, BasisLevel),
    sell: (&InstrumentSpec, BasisLevel),
    lot: Quantity,
    target: Quantity,
    minimum: Quantity,
) -> Result<Result<Quantity, BasisRejection>, BasisError> {
    let units = |(spec, level): (&InstrumentSpec, BasisLevel)| -> Result<Decimal, BasisError> {
        arbitrage_rs::exact::mul(level.contracts, multiplier(spec)).map_err(upstream)
    };
    let available = units(buy)?.min(units(sell)?).min(target.value());
    let floored = match Quantity::new(available) {
        Ok(quantity) => quantity.floor_to_lot(lot).map_err(upstream)?,
        Err(_) => None,
    };
    Ok(match floored {
        Some(quantity) if quantity.value() >= minimum.value() => Ok(quantity),
        other => Err(BasisRejection::InsufficientLiquidity {
            available: other.map_or(Decimal::ZERO, |q| q.value()),
            required: minimum.value(),
        }),
    })
}

/// Evaluates buying one leg and selling the other: see the
/// [module documentation](self).
///
/// # Errors
///
/// [`BasisError`] for a request that cannot be answered.
pub fn evaluate_basis(input: &BasisInput) -> Result<BasisEvaluation, BasisError> {
    let expiry_ms = check_legs(input)?;
    let as_of = time("as_of_ms", input.as_of_ms)?;
    let (buy, sell) = (&input.buy, &input.sell);
    let (buy_spec, sell_spec) = (&buy.spec, &sell.spec);
    let lot = common_lot(buy_spec, sell_spec)?;
    let currency = Currency::new(buy_spec.price_asset().as_str()).map_err(upstream)?;
    let (target, minimum) = check_amounts(
        input.future_margin_rate,
        input.min_net_edge,
        input.target_quantity,
        input.min_quantity,
    )?;
    let mut carry = checked_carry(&input.carry)?;
    carry.reserve_exact(2);
    let exit_item = |leg, value| -> Result<Carry, BasisError> {
        Ok(Carry {
            kind: opportunity::CarryKind::Other,
            leg,
            rate: bps("exit_bps", value)?,
        })
    };
    carry.push(exit_item(Action::Buy, buy.exit_bps)?);
    carry.push(exit_item(Action::Sell, sell.exit_bps)?);
    let (buy_fee, sell_fee) = (bps("fee_bps", buy.fee_bps)?, bps("fee_bps", sell.fee_bps)?);

    // Both quotes are checked for what can only be an error before any rejection.
    let buy_check = check_quote(buy, input.as_of_ms, input.max_quote_age_ms)?;
    let sell_check = check_quote(sell, input.as_of_ms, input.max_quote_age_ms)?;

    if expiry_ms <= input.as_of_ms {
        let future = if buy_spec.kind() == ContractKind::Future {
            buy_spec
        } else {
            sell_spec
        };
        return Ok(BasisEvaluation::Rejected(BasisRejection::Expired {
            instrument: future.id().as_str().to_string(),
            expiry_ms,
        }));
    }
    let buy_level = match buy_check {
        Ok(level) => level,
        Err(rejection) => return Ok(BasisEvaluation::Rejected(rejection)),
    };
    let sell_level = match sell_check {
        Ok(level) => level,
        Err(rejection) => return Ok(BasisEvaluation::Rejected(rejection)),
    };

    let quantity = match executable_quantity(
        (buy_spec, buy_level),
        (sell_spec, sell_level),
        lot,
        target,
        minimum,
    )? {
        Ok(quantity) => quantity,
        Err(rejection) => return Ok(BasisEvaluation::Rejected(rejection)),
    };

    let evaluation = opportunity::evaluate(&OpportunityInput {
        as_of,
        quote_currency: currency,
        buy: LegQuote {
            instrument: upstream_instrument(buy_spec)?,
            book: upstream_book(Action::Buy, buy_level, quantity)?,
            fee: buy_fee,
        },
        sell: LegQuote {
            instrument: upstream_instrument(sell_spec)?,
            book: upstream_book(Action::Sell, sell_level, quantity)?,
            fee: sell_fee,
        },
        target_quantity: quantity,
        min_quantity: quantity,
        carry,
        min_net_edge: Money::new(input.min_net_edge, currency),
    })
    .map_err(upstream)?;
    let opportunity = match evaluation {
        Evaluation::Accepted(opportunity) => opportunity,
        Evaluation::Rejected(RejectReason::NoGrossEdge { gross }) => {
            return Ok(BasisEvaluation::Rejected(BasisRejection::NoGrossEdge {
                gross: gross.amount(),
            }));
        }
        Evaluation::Rejected(RejectReason::NetEdgeBelowThreshold {
            gross,
            costs,
            net,
            min_net_edge,
        }) => {
            return Ok(BasisEvaluation::Rejected(
                BasisRejection::NetEdgeBelowThreshold {
                    gross: gross.amount(),
                    costs: costs_from(&costs, &input.carry)?,
                    net: net.amount(),
                    min_net_edge: min_net_edge.amount(),
                },
            ));
        }
        Evaluation::Rejected(other) => {
            return Err(BasisError::Evaluation {
                reason: format!("unexpected rejection: {other:?}"),
            });
        }
    };
    let asset = buy_spec.price_asset().clone();
    Ok(BasisEvaluation::Accepted(BasisOpportunity {
        asset,
        quantity: quantity.value(),
        buy: plan(buy_spec, buy_level.price, quantity.value())?,
        sell: plan(sell_spec, sell_level.price, quantity.value())?,
        converges_at_ms: expiry_ms,
        gross: opportunity.expected_gross.amount(),
        costs: costs_from(&opportunity.expected_costs, &input.carry)?,
        net: opportunity.expected_net.amount(),
        capital: capital(
            (buy_spec, buy_level.price),
            (sell_spec, sell_level.price),
            quantity,
            input.future_margin_rate,
            currency,
        )?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream items that do not match what was given are an error, in any number.
    #[test]
    fn test_costs_need_every_carry_and_exit_item() {
        let usdt = Currency::new("USDT").unwrap();
        let cost = |amount| Cost::new(Money::new(amount, usdt));
        let item = |amount| CarryCost {
            kind: opportunity::CarryKind::Other,
            leg: Action::Buy,
            cost: cost(amount),
        };
        let breakdown = |items: Vec<CarryCost>| CostBreakdown {
            buy_fee: cost(Decimal::ONE),
            sell_fee: cost(Decimal::TWO),
            carry: items,
            total: cost(Decimal::TEN),
        };
        let given = [BasisCarry {
            kind: CarryKind::Borrow,
            on: Action::Sell,
            period_bps: Decimal::ONE,
        }];
        let costs = costs_from(
            &breakdown(vec![
                item(Decimal::new(3, 0)),
                item(Decimal::new(4, 0)),
                item(Decimal::new(5, 0)),
            ]),
            &given,
        )
        .unwrap();
        assert_eq!(costs.carry[0].amount, Decimal::new(3, 0));
        assert_eq!(costs.carry[0].kind, CarryKind::Borrow);
        assert_eq!(costs.carry[0].on, Action::Sell);
        assert_eq!(costs.buy_exit, Decimal::new(4, 0));
        assert_eq!(costs.sell_exit, Decimal::new(5, 0));
        for count in [0, 1, 2, 4] {
            let items = vec![item(Decimal::ONE); count];
            assert!(matches!(
                costs_from(&breakdown(items), &given),
                Err(BasisError::Evaluation { .. })
            ));
        }
    }
}
