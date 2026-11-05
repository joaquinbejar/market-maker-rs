//! The net result of a calendar spread between two linear dated futures, under an
//! assumed exit.
//!
//! [`evaluate_calendar`] buys one future at its ask and sells the other at its bid.
//! Two futures do not converge to each other: when the near one settles, the far one
//! is still open at a spread nobody knows. So the result depends on an assumption, the
//! required [`CalendarInput::expected_exit_spread`], which is reported apart from the
//! [`CalendarOpportunity::entry_spread`] that was quoted.
//!
//! An accepted [`CalendarOpportunity`] says only that this assumption clears the
//! costs. It is an intent, not an executed or guaranteed profit: the legs can fill
//! partially or not at all, and the exit spread can be anything.
//!
//! # The spread
//!
//! Every spread is the far price minus the near price, whichever leg is bought. With
//! `q` the size in the underlying and `X` the assumed exit spread:
//!
//! - [`CalendarSide::BuyNearSellFar`] earns `q * (entry_spread - X)`, so a unit more
//!   of `X` is `q` less;
//! - [`CalendarSide::SellNearBuyFar`] earns `q * (X - entry_spread)`, so a unit more
//!   of `X` is `q` more.
//!
//! The entry spread may be zero or negative, and so may `X`, which is not a quote and
//! is on no price grid. Buying the near at 100 and selling the far at 99 earns 2 per
//! unit if the spread ends at -3.
//!
//! `X` is about prices and `exit_bps` is about fees, so neither counts a cost twice.
//! `X` includes what the caller expects to lose to the bid and ask on the way out, and
//! [`BasisQuote::exit_bps`] is any fee, charged on the entry notional as the exit price
//! is not known. It is required per leg by its type, and it may be zero or a rebate,
//! as it is itemized.
//!
//! # The exit
//!
//! [`CalendarInput::exit_at_ms`] is after the decision and before the near expiry, and
//! any carry is for the whole period up to it. The assumption is of both legs closed by
//! a trade. An exit at or after the near expiry is refused: the near leg then settles
//! at a price that is not a trade, which this model does not cover, and `X` no longer
//! describes two open legs.
//!
//! # Scope
//!
//! Two dated futures, linear, of different expiries, with the same underlying and price
//! asset. Anything else is a [`BasisError::Unsupported`]. A leg's multiplier and lot
//! may differ: every amount is in the underlying, each leg has its own contracts, and
//! liquidity is quantized as in [`evaluate_basis`](super::basis::evaluate_basis).
//!
//! [`CalendarOpportunity::capital`] is both legs' notional times a rate the caller
//! gives, added with no offset between them. It is an estimate by the caller, not a
//! venue's validated margin, and a venue's calendar offset is the caller's to reflect
//! in that rate.
//!
//! # Rejected, or refused
//!
//! As in [`basis`](super::basis), a [`BasisRejection`] is what the market says and a
//! [`BasisError`] is a request that cannot be answered, which does not depend on the
//! quotes. An exit at or before the decision is refused only while the near contract
//! is alive. Once it expired that is an [`BasisRejection::Expired`], so a caller
//! scanning at a fixed date gets a rejection. A result that does not fit a `Decimal`
//! exactly is an error, and nothing is rounded.

use arbitrage_rs::opportunity::Carry;
use arbitrage_rs::{Cost, Currency, Money, Price, Quantity, exact};
use financial_types::Action;
use rust_decimal::Decimal;

use super::basis::{
    BasisCarry, BasisCarryCost, BasisCosts, BasisError, BasisLegPlan, BasisQuote, BasisRejection,
    bps, capital, check_amounts, check_contract, check_quote, checked_carry, common_lot,
    executable_quantity, plan, time, unsupported, upstream,
};
use super::{AssetCode, InstrumentSpec};

/// Which future is bought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalendarSide {
    /// Buys the near future and sells the far one: gains if the spread falls.
    BuyNearSellFar,
    /// Sells the near future and buys the far one: gains if the spread rises.
    SellNearBuyFar,
}

/// Input to [`evaluate_calendar`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarInput {
    /// The time of the decision, in milliseconds since the epoch.
    pub as_of_ms: u64,
    /// The leg to buy.
    pub buy: BasisQuote,
    /// The leg to sell.
    pub sell: BasisQuote,
    /// The far price minus the near price at the exit: an assumption, and signed.
    pub expected_exit_spread: Decimal,
    /// When the exit is assumed, in milliseconds since the epoch: see the
    /// [module documentation](self).
    pub exit_at_ms: u64,
    /// The size wanted, in the underlying.
    pub target_quantity: Decimal,
    /// The smallest size worth trading, in the underlying.
    pub min_quantity: Decimal,
    /// The oldest quote to accept: a quote of exactly this age is fresh.
    pub max_quote_age_ms: u64,
    /// Each leg's capital as a fraction of its notional, within `[0, 1]`.
    pub future_margin_rate: Decimal,
    /// The holding costs, for the whole period up to the exit.
    pub carry: Vec<BasisCarry>,
    /// The smallest net result to accept, in the price asset: zero or more. A result
    /// of exactly this much is accepted, if it is above zero.
    pub min_net_edge: Decimal,
}

/// An accepted calendar, with its economics under the assumed exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarOpportunity {
    /// The asset of every amount here.
    pub asset: AssetCode,
    /// The size, in the underlying.
    pub quantity: Decimal,
    /// Which future is bought.
    pub side: CalendarSide,
    /// The leg to buy.
    pub buy: BasisLegPlan,
    /// The leg to sell.
    pub sell: BasisLegPlan,
    /// The far price minus the near price, as quoted now.
    pub entry_spread: Decimal,
    /// The far price minus the near price at the exit, as the caller assumed.
    pub expected_exit_spread: Decimal,
    /// The exit the caller assumed, the end of the period that a carry covers.
    pub exit_at_ms: u64,
    /// The result of the two spreads, in the price asset: see the
    /// [module documentation](self).
    pub gross: Decimal,
    /// What is subtracted from it.
    pub costs: BasisCosts,
    /// `gross - costs.total`.
    pub net: Decimal,
    /// The caller's estimate of the capital: see the [module documentation](self).
    pub capital: Decimal,
}

/// The result of [`evaluate_calendar`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarEvaluation {
    /// The calendar passed, under the assumed exit.
    Accepted(CalendarOpportunity),
    /// The calendar was discarded, with the reason.
    Rejected(BasisRejection),
}

/// Who is near, and when it expires.
struct Legs<'a> {
    near: &'a InstrumentSpec,
    near_expiry_ms: u64,
    side: CalendarSide,
}

/// Checks both legs, and tells near from far.
fn check_legs(input: &CalendarInput) -> Result<Legs<'_>, BasisError> {
    let (buy, sell) = (&input.buy.spec, &input.sell.spec);
    let (buy_expiry, sell_expiry) = match (check_contract(buy)?, check_contract(sell)?) {
        (Some(buy_expiry), Some(sell_expiry)) => (buy_expiry, sell_expiry),
        _ => return unsupported("a calendar is two dated futures"),
    };
    if buy.id() == sell.id() {
        return unsupported("the legs are the same contract");
    }
    if buy.underlying() != sell.underlying() {
        return unsupported("the legs have different underlyings");
    }
    if buy.price_asset() != sell.price_asset() {
        return unsupported("the legs have different price assets");
    }
    if buy_expiry == sell_expiry {
        return unsupported("the legs expire together, so neither is the near one");
    }
    Ok(if buy_expiry < sell_expiry {
        Legs {
            near: buy,
            near_expiry_ms: buy_expiry,
            side: CalendarSide::BuyNearSellFar,
        }
    } else {
        Legs {
            near: sell,
            near_expiry_ms: sell_expiry,
            side: CalendarSide::SellNearBuyFar,
        }
    })
}

/// A rate on an entry notional.
fn charge(rate: Decimal, field: &'static str, notional: Money) -> Result<Cost, BasisError> {
    bps(field, rate)?.fee_on(notional).map_err(upstream)
}

/// Every cost, itemized and summed, from the notionals of the two legs.
fn costs_of(
    input: &CalendarInput,
    carry: &[Carry],
    buy_notional: Money,
    sell_notional: Money,
) -> Result<BasisCosts, BasisError> {
    let buy_fee = charge(input.buy.fee_bps, "fee_bps", buy_notional)?;
    let sell_fee = charge(input.sell.fee_bps, "fee_bps", sell_notional)?;
    let buy_exit = charge(input.buy.exit_bps, "exit_bps", buy_notional)?;
    let sell_exit = charge(input.sell.exit_bps, "exit_bps", sell_notional)?;
    let mut total = buy_fee
        .checked_add(sell_fee)
        .and_then(|sum| sum.checked_add(buy_exit))
        .and_then(|sum| sum.checked_add(sell_exit))
        .map_err(upstream)?;
    let mut held = Vec::with_capacity(carry.len());
    for (item, given) in carry.iter().zip(&input.carry) {
        let notional = if given.on == Action::Buy {
            buy_notional
        } else {
            sell_notional
        };
        let cost = item.rate.fee_on(notional).map_err(upstream)?;
        total = total.checked_add(cost).map_err(upstream)?;
        held.push(BasisCarryCost {
            kind: given.kind,
            on: given.on,
            amount: cost.money().amount(),
        });
    }
    Ok(BasisCosts {
        buy_fee: buy_fee.money().amount(),
        sell_fee: sell_fee.money().amount(),
        buy_exit: buy_exit.money().amount(),
        sell_exit: sell_exit.money().amount(),
        carry: held,
        total: total.money().amount(),
    })
}

/// Evaluates buying one future and selling the other, under an assumed exit: see the
/// [module documentation](self).
///
/// # Errors
///
/// [`BasisError`] for a request that cannot be answered.
pub fn evaluate_calendar(input: &CalendarInput) -> Result<CalendarEvaluation, BasisError> {
    let legs = check_legs(input)?;
    time("as_of_ms", input.as_of_ms)?;
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
    let carry = checked_carry(&input.carry)?;
    for quote in [buy, sell] {
        bps("fee_bps", quote.fee_bps)?;
        bps("exit_bps", quote.exit_bps)?;
    }
    let outside_window = BasisError::ExitOutsideWindow {
        exit_at_ms: input.exit_at_ms,
        as_of_ms: input.as_of_ms,
        near_expiry_ms: legs.near_expiry_ms,
    };
    if input.exit_at_ms >= legs.near_expiry_ms {
        return Err(outside_window);
    }

    // Both quotes are checked for what can only be an error before any rejection.
    let buy_check = check_quote(buy, input.as_of_ms, input.max_quote_age_ms)?;
    let sell_check = check_quote(sell, input.as_of_ms, input.max_quote_age_ms)?;

    if legs.near_expiry_ms <= input.as_of_ms {
        return Ok(CalendarEvaluation::Rejected(BasisRejection::Expired {
            instrument: legs.near.id().as_str().to_string(),
            expiry_ms: legs.near_expiry_ms,
        }));
    }
    if input.exit_at_ms <= input.as_of_ms {
        return Err(outside_window);
    }
    let buy_level = match buy_check {
        Ok(level) => level,
        Err(rejection) => return Ok(CalendarEvaluation::Rejected(rejection)),
    };
    let sell_level = match sell_check {
        Ok(level) => level,
        Err(rejection) => return Ok(CalendarEvaluation::Rejected(rejection)),
    };
    let quantity = match executable_quantity(
        (buy_spec, buy_level),
        (sell_spec, sell_level),
        lot,
        target,
        minimum,
    )? {
        Ok(quantity) => quantity,
        Err(rejection) => return Ok(CalendarEvaluation::Rejected(rejection)),
    };

    let (buy_price, sell_price) = (buy_level.price, sell_level.price);
    let exit_spread = input.expected_exit_spread;
    let (entry_spread, per_unit) = match legs.side {
        CalendarSide::BuyNearSellFar => {
            let entry = exact::sub(sell_price, buy_price).map_err(upstream)?;
            (entry, exact::sub(entry, exit_spread).map_err(upstream)?)
        }
        CalendarSide::SellNearBuyFar => {
            let entry = exact::sub(buy_price, sell_price).map_err(upstream)?;
            (entry, exact::sub(exit_spread, entry).map_err(upstream)?)
        }
    };
    let gross = exact::mul(per_unit, quantity.value()).map_err(upstream)?;
    if gross <= Decimal::ZERO {
        return Ok(CalendarEvaluation::Rejected(BasisRejection::NoGrossEdge {
            gross,
        }));
    }

    let notional = |price: Decimal, quantity: Quantity| -> Result<Money, BasisError> {
        Price::new(price)
            .and_then(|price| price.notional(quantity, currency))
            .map_err(upstream)
    };
    let costs = costs_of(
        input,
        &carry,
        notional(buy_price, quantity)?,
        notional(sell_price, quantity)?,
    )?;
    let net = exact::sub(gross, costs.total).map_err(upstream)?;
    if net <= Decimal::ZERO || net < input.min_net_edge {
        return Ok(CalendarEvaluation::Rejected(
            BasisRejection::NetEdgeBelowThreshold {
                gross,
                costs,
                net,
                min_net_edge: input.min_net_edge,
            },
        ));
    }
    Ok(CalendarEvaluation::Accepted(CalendarOpportunity {
        asset: buy_spec.price_asset().clone(),
        quantity: quantity.value(),
        side: legs.side,
        buy: plan(buy_spec, buy_price, quantity.value())?,
        sell: plan(sell_spec, sell_price, quantity.value())?,
        entry_spread,
        expected_exit_spread: exit_spread,
        exit_at_ms: input.exit_at_ms,
        gross,
        costs,
        net,
        capital: capital(
            (buy_spec, buy_price),
            (sell_spec, sell_price),
            quantity,
            input.future_margin_rate,
            currency,
        )?,
    }))
}
