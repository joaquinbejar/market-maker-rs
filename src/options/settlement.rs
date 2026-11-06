//! Booking an option's expiry, exercise or assignment.
//!
//! What a position must pay, receive or deliver is computed by OptionStratLib, through
//! `settlement_obligation`. This module owns what happens around it: the open orders on the
//! contract, the ledger, and the option's own record.
//!
//! The `margin-rs` ledger has no option instrument, so [`OptionBook`] holds option positions
//! beside it, and the ledger never values one. A cash settlement is a derivative settlement in
//! the ledger, under the contract's name, so it counts as settled PnL and not as a transfer. A
//! physical settlement is an execution of the spot instrument at the strike, so the underlying
//! has the strike as its cost. The position then leaves the book, so nothing holds both an
//! extinguished option and its value.
//!
//! # Causal order
//!
//! 1. The obligation is computed. A missing price, a price after the cutoff, an event of the
//!    wrong side and an unsupported option type are errors here, before anything changes.
//! 2. The reservations of the contract's open orders are released, so a settlement can use them.
//!    One no longer in the ledger, such as a filled order's, is noted and skipped.
//! 3. The obligation is booked. It is mandatory: a position that cannot pay or deliver is booked
//!    all the same, and the amount short is in [`SettlementRecord::shortfalls`]. What follows,
//!    such as a default or a liquidation, is not modelled.
//! 4. The position leaves the book.
//!
//! All of it is one transaction: on an error, such as an amount finer than the asset's scale, the
//! ledger and the book are as they were, and the position stays open.
//!
//! # What is the caller's
//!
//! - **Open orders.** Releasing a reservation does not cancel its order: cancel those in
//!   [`SettlementRecord::released`] with whatever manages them. Nothing checks who owns a
//!   reservation, so register only the contract's own.
//! - **The event.** Which short position an assignment falls on and whether an American option
//!   should be exercised early are not decided here, and a settlement ends the whole position. No
//!   clock is read: the terms state the price, its source and the cutoff.
//! - **The contract size**, taken from the `Options` as it is, where 1 is also what a position
//!   deserialized without one gets.
//! - **Margin held in a `RiskGate`**, which is not touched here.

use std::collections::BTreeMap;

use margin_rs::ledger::{Booking, DerivativeSettlement, Execution, Ledger};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId, MarginError, ReservationId};
use optionstratlib::Options;
use optionstratlib::model::{
    SettlementEvent, SettlementObligation, SettlementTerms, SettlementType, settlement_obligation,
};

use crate::Decimal;
use crate::execution::Action;
use crate::types::error::{MMError, MMResult};

/// An open option position, and what its settlement moves in the ledger.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptionPosition {
    /// The contract, with its side, quantity and contract size.
    pub option: Options,
    /// The asset cash settles in.
    pub cash_asset: Asset,
    /// The spot instrument a physical settlement trades, with the underlying as its base and
    /// `cash_asset` as its quote.
    pub spot: Option<InstrumentId>,
    reservations: Vec<ReservationId>,
}

impl OptionPosition {
    /// A position with no open orders, which can settle in cash only.
    #[must_use]
    pub fn new(option: Options, cash_asset: Asset) -> Self {
        Self {
            option,
            cash_asset,
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
    /// The contract settled.
    pub contract: String,
    /// What ended it.
    pub event: SettlementEvent,
    /// How it settled.
    pub settlement_type: SettlementType,
    /// What it obliged, seen from the position.
    pub obligation: SettlementObligation,
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

/// Open option positions, kept beside a ledger.
#[derive(Debug, Clone, Default)]
pub struct OptionBook {
    open: BTreeMap<String, OptionPosition>,
    settled: BTreeMap<String, SettlementRecord>,
}

fn ledger_error(what: &str, error: &MarginError) -> MMError {
    MMError::InvalidPositionUpdate(format!("settlement {what}: {error}"))
}

fn settlement_id(contract: &str) -> Result<ExecutionId, MarginError> {
    ExecutionId::new(format!("settlement:{contract}"))
}

fn config_error(what: String) -> MMError {
    MMError::InvalidConfiguration(what)
}

fn checked_total(ledger: &Ledger, asset: &Asset) -> MMResult<Decimal> {
    ledger
        .balance(asset)
        .map(|balance| balance.total)
        .ok_or_else(|| config_error(format!("settlement asset {asset} is not in the ledger")))
}

/// Books a cash obligation as a derivative settlement of the contract.
fn book_cash(
    ledger: &mut Ledger,
    contract: &str,
    position: &OptionPosition,
    cash: Decimal,
    shortfalls: &mut BTreeMap<Asset, Decimal>,
) -> MMResult<()> {
    let id = settlement_id(contract).map_err(|error| ledger_error("id", &error))?;
    let instrument = InstrumentId::new(contract).map_err(|error| ledger_error("id", &error))?;
    let booking = ledger
        .book_settlement(&DerivativeSettlement {
            id,
            instrument,
            asset: position.cash_asset.clone(),
            realized_pnl: cash,
            fee: Decimal::ZERO,
        })
        .map_err(|error| ledger_error("cash", &error))?;
    if !booking.is_new {
        return Err(MMError::InvalidPositionUpdate(format!(
            "settlement of {contract} was already booked"
        )));
    }
    if let Some(short) = booking.shortfall {
        shortfalls.insert(position.cash_asset.clone(), short);
    }
    Ok(())
}

/// Books a physical obligation as an execution of the spot instrument at the strike, and checks
/// it moved what the obligation says.
fn book_physical(
    ledger: &mut Ledger,
    contract: &str,
    position: &OptionPosition,
    obligation: &SettlementObligation,
    shortfalls: &mut BTreeMap<Asset, Decimal>,
) -> MMResult<()> {
    let spot = position.spot.as_ref().ok_or_else(|| {
        config_error(format!(
            "{contract} settles physically with no spot instrument"
        ))
    })?;
    let Some(Instrument::Spot(instrument)) = ledger.instrument(spot) else {
        return Err(config_error(format!(
            "{spot} is not a spot instrument in the ledger"
        )));
    };
    let (base, quote) = (instrument.base().clone(), instrument.quote().clone());
    if quote != position.cash_asset {
        return Err(config_error(format!(
            "{spot} is quoted in {quote}, and {contract} settles in {}",
            position.cash_asset
        )));
    }
    let (cash_before, base_before) = (
        checked_total(ledger, &quote)?,
        checked_total(ledger, &base)?,
    );
    let side = if obligation.underlying > Decimal::ZERO {
        Action::Buy
    } else {
        Action::Sell
    };
    let id = settlement_id(contract).map_err(|error| ledger_error("id", &error))?;
    let execution = Execution::new(
        id,
        spot.clone(),
        side,
        position.option.strike_price.to_dec(),
        obligation.underlying.abs(),
    );
    let booked = ledger
        .book_execution_allowing_overdraft(&execution, None)
        .map_err(|error| ledger_error("delivery", &error))?;
    if matches!(booked.booking, Booking::Duplicate(_)) {
        return Err(MMError::InvalidPositionUpdate(format!(
            "settlement of {contract} was already booked"
        )));
    }
    shortfalls.extend(booked.shortfalls);
    let moved = |after: Decimal, before: Decimal| {
        after
            .checked_sub(before)
            .ok_or_else(|| MMError::NumericalError(format!("settlement of {contract}")))
    };
    let cash_moved = moved(checked_total(ledger, &quote)?, cash_before)?;
    let base_moved = moved(checked_total(ledger, &base)?, base_before)?;
    if cash_moved != obligation.cash || base_moved != obligation.underlying {
        return Err(MMError::InvalidPositionUpdate(format!(
            "settlement of {contract} moved {cash_moved} and {base_moved}, and owes {} and {}",
            obligation.cash, obligation.underlying
        )));
    }
    Ok(())
}

impl OptionBook {
    /// An empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens a position under `contract`, a name the ledger's identifiers can hold.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the contract is open or was settled, as a
    /// settlement ends a contract once, if its quantity is zero, or if the name cannot be an
    /// identifier in the ledger.
    pub fn open(&mut self, contract: impl Into<String>, position: OptionPosition) -> MMResult<()> {
        let contract = contract.into();
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
        settlement_id(&contract)
            .and_then(|_| InstrumentId::new(contract.as_str()))
            .map_err(|error| config_error(format!("option contract {contract}: {error}")))?;
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
    /// OptionStratLib refuses, such as a missing price or an unsupported option type, or a
    /// physical settlement with no usable spot instrument, and
    /// [`MMError::InvalidPositionUpdate`] if the ledger refuses a step, such as an amount finer
    /// than an asset's scale. The ledger and the book are then unchanged.
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
        let mut shortfalls = BTreeMap::new();
        if !obligation.underlying.is_zero() {
            book_physical(&mut trial, contract, position, &obligation, &mut shortfalls)?;
        } else if !obligation.cash.is_zero() {
            book_cash(
                &mut trial,
                contract,
                position,
                obligation.cash,
                &mut shortfalls,
            )?;
        }
        let record = SettlementRecord {
            contract: contract.to_string(),
            event: terms.event,
            settlement_type: terms.settlement_type,
            obligation,
            price_source: terms.price.as_ref().map(|price| price.source.clone()),
            released,
            already_gone,
            shortfalls,
        };
        *ledger = trial;
        self.open.remove(contract);
        self.settled.insert(contract.to_string(), record.clone());
        Ok(record)
    }
}
