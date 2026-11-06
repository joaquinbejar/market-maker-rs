//! Booking an option's expiry, exercise or assignment.
//!
//! What a position must pay, receive or deliver is computed by OptionStratLib, through
//! `settlement_obligation`. The `margin-rs` ledger holds the option itself: a registered
//! [`OptionInstrument`], its position, the premium booked at each fill, and its settlement,
//! through [`Ledger::settle_option`]. This module joins the two: it maps the obligation to the
//! ledger's [`OptionOutcome`], releases the contract's open orders first, and refuses unless the
//! ledger moved exactly what OptionStratLib computed.
//!
//! # Outcomes
//!
//! - **Cash.** `OptionOutcome::Cash` at the terms' settlement price. Out of the money it pays
//!   zero: the ledger has no lapse for a cash option, so a payoff is never dropped by leaving its
//!   price out. The value is settled PnL of the option, not a transfer.
//! - **Physical, exercised or assigned.** `OptionOutcome::Delivery`: a spot execution at the
//!   strike, booked with the close of the option in one step, so the underlying has the strike
//!   as its cost and the premium stays in the option's PnL.
//! - **Physical, not exercised.** `OptionOutcome::Lapsed`.
//!
//! Either way the settlement stays in the ledger under its identifier, and the position is
//! flat, so a rebuilt book cannot settle the contract again.
//!
//! # Causal order
//!
//! 1. The obligation is computed, and the instrument's terms are checked against the option. A
//!    missing price, a price after the cutoff, an event of the wrong side, an unsupported option
//!    type, terms that disagree, or a position other than the option's are errors here, before
//!    anything changes.
//! 2. The reservations of the contract's open orders are released, so a settlement can use them.
//!    One no longer in the ledger, such as a filled order's, is noted and skipped.
//! 3. The settlement is booked. It is mandatory: a position that cannot pay or deliver is booked
//!    all the same, and the amount short is in [`SettlementRecord::shortfalls`]. What follows,
//!    such as a default or a liquidation, is not modelled.
//! 4. The position leaves the book.
//!
//! All of it is one transaction: on an error, such as an amount finer than an asset's scale, or
//! a ledger that rounds otherwise than the terms, the ledger and the book are as they were.
//!
//! # What is the caller's
//!
//! - **The instrument and the position.** Register the option in the ledger and book its fills
//!   there, with their premium. The ledger values an open option at the marks it is given.
//! - **Open orders.** Releasing a reservation does not cancel its order: cancel those in
//!   [`SettlementRecord::released`] with whatever manages them. Nothing checks who owns a
//!   reservation, so register only the contract's own.
//! - **The event.** Which short position an assignment falls on and whether an American option
//!   should be exercised early are not decided here, and a settlement closes the whole position.
//!   No clock is read: the terms state the price, its source and the cutoff, which is the time
//!   the settlement applies.
//! - **Margin held in a `RiskGate`**, which is not touched here.
//! - **The underlying and the expiry** of the instrument, which are not compared with the
//!   option's: the obligation depends on neither, and the ledger applies its own expiry rules.
//! - **A position reopened after its settlement**, by a fill executed before the expiry and
//!   booked after it. The book settles a contract once, under one identifier, so such a position
//!   is the caller's to settle in the ledger.

use std::collections::BTreeMap;

use margin_rs::instrument::{
    ExerciseStyle, OptionDelivery, OptionInstrument, OptionMargining, OptionRight,
};
use margin_rs::ledger::{Ledger, OptionOutcome, OptionSettlement};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId, MarginError, ReservationId};
use optionstratlib::Options;
use optionstratlib::model::types::{OptionStyle, OptionType, Side};
use optionstratlib::model::{
    SettlementEvent, SettlementObligation, SettlementTerms, SettlementType, settlement_obligation,
};

use crate::Decimal;
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

/// An open option position: the contract as OptionStratLib sees it, and the ledger's instrument
/// that holds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptionPosition {
    /// The contract, with its side, quantity in contracts and contract size.
    pub option: Options,
    /// The option instrument in the ledger.
    pub instrument: InstrumentId,
    /// The spot instrument a physical settlement trades, with the underlying as its base and the
    /// option's settlement asset as its quote.
    pub spot: Option<InstrumentId>,
    reservations: Vec<ReservationId>,
}

impl OptionPosition {
    /// A position with no open orders, which can settle in cash only.
    #[must_use]
    pub fn new(option: Options, instrument: InstrumentId) -> Self {
        Self {
            option,
            instrument,
            spot: None,
            reservations: Vec::new(),
        }
    }

    /// The position, able to settle physically through `spot`.
    #[must_use]
    pub fn with_spot(mut self, spot: InstrumentId) -> Self {
        self.spot = Some(spot);
        self
    }

    /// The reservations of the contract's open orders.
    #[must_use]
    pub fn reservations(&self) -> &[ReservationId] {
        &self.reservations
    }
}

/// What a settlement did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SettlementRecord {
    /// The contract settled: its instrument in the ledger.
    pub contract: String,
    /// What ended it.
    pub event: SettlementEvent,
    /// How it settled.
    pub settlement_type: SettlementType,
    /// What it obliged, seen from the position.
    pub obligation: SettlementObligation,
    /// The outcome booked in the ledger.
    pub outcome: OptionOutcome,
    /// Realized PnL of the option's close, in its settlement asset: the value less the premium.
    pub realized_pnl: Decimal,
    /// Who published the price, if the event needed one.
    pub price_source: Option<String>,
    /// The reservations released, with their assets and amounts. Their orders are the caller's
    /// to cancel.
    pub released: Vec<(ReservationId, Asset, Decimal)>,
    /// Registered reservations no longer in the ledger, such as a filled order's.
    pub already_gone: Vec<ReservationId>,
    /// What the booking left short, per asset: the reservations it could not cover or the
    /// balance it took below zero.
    pub shortfalls: BTreeMap<Asset, Decimal>,
}

/// Open option positions, kept with the ledger that holds their instruments.
#[derive(Debug, Clone, Default)]
pub struct OptionBook {
    open: BTreeMap<String, OptionPosition>,
    settled: BTreeMap<String, SettlementRecord>,
}

fn ledger_error(what: &str, error: &MarginError) -> MMError {
    MMError::InvalidPositionUpdate(format!("settlement {what}: {error}"))
}

fn config_error(what: String) -> MMError {
    MMError::InvalidConfiguration(what)
}

fn execution_id(kind: &str, contract: &str) -> MMResult<ExecutionId> {
    ExecutionId::new(format!("{kind}:{contract}")).map_err(|error| ledger_error("id", &error))
}

fn total(ledger: &Ledger, asset: &Asset) -> Decimal {
    ledger
        .balance(asset)
        .map_or(Decimal::ZERO, |balance| balance.total)
}

/// The ledger's option for `position`, checked against the option's own terms.
fn checked_instrument(ledger: &Ledger, position: &OptionPosition) -> MMResult<OptionInstrument> {
    let id = &position.instrument;
    let instrument = ledger
        .instrument(id)
        .and_then(Instrument::option_terms)
        .ok_or_else(|| config_error(format!("{id} is not an option in the ledger")))?;
    let option = &position.option;
    let contract = instrument.contract();
    if contract.margining != OptionMargining::Linear {
        // OptionStratLib's obligation is in the price asset, and an inverse option pays the
        // underlying.
        return Err(config_error(format!(
            "{id}: inverse options are not settled here"
        )));
    }
    let right = match option.option_style {
        OptionStyle::Call => OptionRight::Call,
        OptionStyle::Put => OptionRight::Put,
    };
    let exercise = match option.option_type {
        OptionType::European => ExerciseStyle::European,
        OptionType::American => ExerciseStyle::American,
        _ => {
            return Err(config_error(format!(
                "{id}: the ledger holds European and American options only"
            )));
        }
    };
    let disagree =
        |what: &str| config_error(format!("{id}: the ledger's {what} is not the option's"));
    if contract.strike != option.strike_price.to_dec() {
        return Err(disagree("strike"));
    }
    if contract.right != right {
        return Err(disagree("right"));
    }
    if contract.exercise != exercise {
        return Err(disagree("exercise style"));
    }
    if instrument.terms().multiplier() != option.contract_size.to_dec() {
        return Err(disagree("multiplier"));
    }
    Ok(instrument.clone())
}

/// The ledger's position in `position`'s instrument must be the option's, side and quantity.
fn check_position(ledger: &Ledger, position: &OptionPosition) -> MMResult<Decimal> {
    let id = &position.instrument;
    let held = ledger
        .position(id)
        .map_or(Decimal::ZERO, |snapshot| snapshot.quantity);
    let quantity = position.option.quantity.to_dec();
    let expected = match position.option.side {
        Side::Long => quantity,
        Side::Short => Decimal::ZERO
            .checked_sub(quantity)
            .or_overflow("short quantity")?,
    };
    if held != expected {
        return Err(config_error(format!(
            "{id}: the ledger holds {held} contracts, and the option is {expected}"
        )));
    }
    Ok(quantity)
}

impl OptionBook {
    /// An empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens `position`, under its instrument's identifier.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the contract is open or was settled, as a
    /// settlement ends a contract once, if its quantity is zero, if its identifier is too long to
    /// name its settlement in the ledger, if the instrument is not an
    /// option in `ledger` whose strike, right, exercise style and multiplier are the option's,
    /// or if the ledger's position in it is not the option's side and quantity.
    pub fn open(&mut self, ledger: &Ledger, position: OptionPosition) -> MMResult<()> {
        let contract = position.instrument.to_string();
        if self.open.contains_key(&contract) || self.settled.contains_key(&contract) {
            return Err(config_error(format!(
                "option contract {contract} is already in the book"
            )));
        }
        if position.option.quantity.is_zero() {
            return Err(config_error(format!(
                "option contract {contract} has no quantity"
            )));
        }
        // The identifiers its settlement will need, so that what opens can settle.
        execution_id("settlement", &contract)
            .and_then(|_| execution_id("delivery", &contract))
            .map_err(|error| config_error(format!("option contract {contract}: {error}")))?;
        checked_instrument(ledger, &position)?;
        check_position(ledger, &position)?;
        self.open.insert(contract, position);
        Ok(())
    }

    /// Registers the reservation of an open order on `contract`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the contract is not open, or if the
    /// reservation is registered already, on this contract or another.
    pub fn add_reservation(&mut self, contract: &str, reservation: ReservationId) -> MMResult<()> {
        if self
            .open
            .values()
            .any(|position| position.reservations.contains(&reservation))
        {
            return Err(config_error(format!(
                "reservation {reservation} is registered already"
            )));
        }
        let position = self
            .open
            .get_mut(contract)
            .ok_or_else(|| config_error(format!("option contract {contract} is not open")))?;
        position.reservations.push(reservation);
        Ok(())
    }

    /// The position open under `contract`, if any.
    #[must_use]
    pub fn position(&self, contract: &str) -> Option<&OptionPosition> {
        self.open.get(contract)
    }

    /// How a settled contract ended, if it did.
    #[must_use]
    pub fn settlement(&self, contract: &str) -> Option<&SettlementRecord> {
        self.settled.get(contract)
    }

    /// Settles `contract` under `terms`. See the [module documentation](self) for its order.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a contract that is not open, terms
    /// OptionStratLib refuses, such as a missing price or an unsupported option type, an
    /// instrument or a position that is no longer the option's, a settlement type other than the
    /// instrument's delivery, or a physical settlement with no spot instrument, and
    /// [`MMError::InvalidPositionUpdate`] if the ledger refuses a step, has booked the
    /// settlement already, or moves other than the obligation. The ledger and the book are then
    /// unchanged.
    pub fn settle(
        &mut self,
        ledger: &mut Ledger,
        contract: &str,
        terms: &SettlementTerms,
    ) -> MMResult<SettlementRecord> {
        let position = self
            .open
            .get(contract)
            .ok_or_else(|| config_error(format!("option contract {contract} is not open")))?;
        let obligation = settlement_obligation(&position.option, terms)
            .map_err(|error| config_error(format!("settling {contract}: {error}")))?;
        let instrument = checked_instrument(ledger, position)?;
        let contracts = check_position(ledger, position)?;
        let delivery = match terms.settlement_type {
            SettlementType::Cash => OptionDelivery::Cash,
            SettlementType::Physical => OptionDelivery::Physical,
        };
        if instrument.contract().delivery != delivery {
            return Err(config_error(format!(
                "{contract} settles {:?} in the ledger, and the terms say {:?}",
                instrument.contract().delivery,
                terms.settlement_type
            )));
        }
        let outcome = match delivery {
            OptionDelivery::Cash => OptionOutcome::Cash {
                price: terms
                    .price
                    .as_ref()
                    .map(|price| price.price.to_dec())
                    .ok_or_else(|| {
                        config_error(format!("{contract} settles in cash with no price"))
                    })?,
            },
            OptionDelivery::Physical if obligation.underlying.is_zero() => OptionOutcome::Lapsed,
            OptionDelivery::Physical => OptionOutcome::Delivery {
                spot: position.spot.clone().ok_or_else(|| {
                    config_error(format!(
                        "{contract} settles physically with no spot instrument"
                    ))
                })?,
                delivery_id: execution_id("delivery", contract)?,
            },
        };
        let as_of_ms = u64::try_from(terms.cutoff.timestamp_millis())
            .map_err(|_| config_error(format!("{contract}: a cutoff before the epoch")))?;
        let cash_asset = instrument.terms().settlement().clone();
        let underlying = instrument.contract().underlying.clone();

        let mut trial = ledger.clone();
        let (mut released, mut already_gone) = (Vec::new(), Vec::new());
        for reservation in &position.reservations {
            if trial.reservation(reservation).is_none() {
                already_gone.push(reservation.clone());
                continue;
            }
            let (asset, amount) = trial
                .release(reservation)
                .map_err(|error| ledger_error("release", &error))?;
            released.push((reservation.clone(), asset, amount));
        }
        let (cash_before, base_before) = (total(&trial, &cash_asset), total(&trial, &underlying));
        let booking = trial
            .settle_option(&OptionSettlement {
                id: execution_id("settlement", contract)?,
                instrument: position.instrument.clone(),
                contracts,
                outcome: outcome.clone(),
                as_of_ms,
            })
            .map_err(|error| ledger_error("booking", &error))?;
        if !booking.is_new {
            return Err(MMError::InvalidPositionUpdate(format!(
                "settlement of {contract} was already booked"
            )));
        }
        let moved = |after: Decimal, before: Decimal| {
            after
                .checked_sub(before)
                .ok_or_else(|| MMError::NumericalError(format!("settlement of {contract}")))
        };
        let cash_moved = moved(total(&trial, &cash_asset), cash_before)?;
        let base_moved = moved(total(&trial, &underlying), base_before)?;
        if cash_moved != obligation.cash || base_moved != obligation.underlying {
            return Err(MMError::InvalidPositionUpdate(format!(
                "settlement of {contract} moved {cash_moved} and {base_moved}, and owes {} and {}",
                obligation.cash, obligation.underlying
            )));
        }
        let record = SettlementRecord {
            contract: contract.to_string(),
            event: terms.event,
            settlement_type: terms.settlement_type,
            obligation,
            outcome,
            realized_pnl: booking.cashflow.realized_pnl,
            price_source: terms.price.as_ref().map(|price| price.source.clone()),
            released,
            already_gone,
            shortfalls: booking.shortfalls,
        };
        *ledger = trial;
        self.open.remove(contract);
        self.settled.insert(contract.to_string(), record.clone());
        Ok(record)
    }
}
