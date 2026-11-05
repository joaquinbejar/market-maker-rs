//! Accounting port at the backtest fill boundary.
//!
//! By default [`BacktestEngine`](crate::backtest::BacktestEngine) keeps its own
//! position, cash flow and fee totals. A [`FillAccountant`] replaces that
//! built in bookkeeping: once selected with
//! [`BacktestEngine::with_accountant`](crate::backtest::BacktestEngine::with_accountant),
//! every fill is posted to it exactly once and the result metrics (position,
//! PnL, fees, equity) are read back from its [`AccountingSnapshot`]. Only one
//! accountant is ever active, so the engine never books a fill twice.
//!
//! With the `margin` feature, `MarginLedgerAccountant` posts fills to the
//! exact ledger of the `margin-rs` crate, on a spot instrument, or on a future
//! or a perpetual through
//! `BacktestEngine::with_instrument`
//! with the `instruments` feature as well.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{AccountingFill, AccountingSnapshot, FillAccountant};
//! use market_maker_rs::types::error::MMResult;
//! use market_maker_rs::Decimal;
//!
//! /// Counts fills and nets quantity; ignores cash.
//! #[derive(Debug, Default)]
//! struct Counter {
//!     snapshot: AccountingSnapshot,
//! }
//!
//! impl FillAccountant for Counter {
//!     fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
//!         self.snapshot.position += fill.signed_quantity()?;
//!         self.snapshot.fills_posted += 1;
//!         Ok(())
//!     }
//!     fn snapshot(&self) -> MMResult<AccountingSnapshot> {
//!         Ok(self.snapshot.clone())
//!     }
//!     fn reset(&mut self) {
//!         self.snapshot = AccountingSnapshot::default();
//!     }
//! }
//! ```

use std::fmt;

use crate::Decimal;
use crate::execution::Action;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

use super::lifecycle::SimOrderId;

pub use crate::types::fee::FeeAsset;

/// One fill as posted to a [`FillAccountant`].
#[derive(Debug, Clone, PartialEq)]
pub struct AccountingFill {
    /// Engine wide unique execution sequence, starting at 1; stable across
    /// replays of the same run, so it can serve as an idempotency key for one
    /// accountant. An accountant handed to an engine again, or a ledger wrapped
    /// again, opens another sequence
    /// ([`FillAccountant::begin_from_current_state`]):
    /// `MarginLedgerAccountant` then books a replay as new fills, not as
    /// duplicates, so it is no way to recover a run onto its own ledger.
    pub execution_id: u64,
    /// Order that was filled.
    pub order_id: SimOrderId,
    /// Action of the order. The engine only posts buy and sell fills.
    pub side: Action,
    /// Execution price, slippage included.
    pub price: Decimal,
    /// Quantity filled by this execution.
    pub quantity: Decimal,
    /// Cumulative filled quantity of the order after this execution.
    pub cumulative_quantity: Decimal,
    /// Fee charged for this execution.
    pub fee: Decimal,
    /// Asset in which [`AccountingFill::fee`] is charged.
    pub fee_asset: FeeAsset,
    /// Venue time of the execution, in milliseconds.
    pub timestamp: u64,
}

impl AccountingFill {
    /// Quantity with sign: positive for a buy, negative for a sell.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`]
    /// if the side is neither buy nor sell.
    pub fn signed_quantity(&self) -> MMResult<Decimal> {
        Ok(match Dir::try_from(self.side)? {
            Dir::Buy => self.quantity,
            // Negating a Decimal flips its sign bit: it cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            Dir::Sell => -self.quantity,
        })
    }

    /// Cash flow of the execution before fees: negative for a buy.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`]
    /// if the side is neither buy nor sell, or
    /// [`MMError::NumericalError`]
    /// if the notional overflows.
    pub fn cash_flow(&self) -> MMResult<Decimal> {
        let dir = Dir::try_from(self.side)?;
        let notional = self.quantity.checked_mul(self.price).ok_or_else(|| {
            MMError::NumericalError(format!(
                "cash flow overflow: {} * {}",
                self.quantity, self.price
            ))
        })?;
        Ok(match dir {
            // Negating a Decimal flips its sign bit: it cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            Dir::Buy => -notional,
            Dir::Sell => notional,
        })
    }
}

/// Immutable view of an accountant's state, read by the engine metrics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccountingSnapshot {
    /// Net position in base units (positive long), or in contracts on a
    /// derivative contract.
    pub position: Decimal,
    /// Average entry price of the open position, if any.
    pub average_entry_price: Option<Decimal>,
    /// Sum of execution cash flows, excluding fees. On a derivative contract,
    /// its realized PnL in the settlement asset, with no notional in it.
    pub cash_flow: Decimal,
    /// Fees fixed in the quote asset (quote fees and third asset fees
    /// converted to quote), or in the settlement asset on a derivative contract.
    pub fees: Decimal,
    /// Fees paid in the base asset, in base units. They are already
    /// deducted from [`AccountingSnapshot::position`]; the engine marks
    /// them at the current price. Zero on a derivative contract.
    pub base_fees: Decimal,
    /// Number of distinct executions booked.
    pub fills_posted: u64,
}

/// Accounting port at the fill boundary.
///
/// # Contract
///
/// - The engine calls [`FillAccountant::post_fill`] exactly once per
///   execution, in venue order, with a unique
///   [`AccountingFill::execution_id`].
/// - After every post the engine reads [`FillAccountant::snapshot`] and uses
///   it for position, PnL, fees and equity.
/// - A post that returns an error leaves the accountant unchanged; the
///   engine records the error (see
///   [`BacktestEngine::accounting_errors`](crate::backtest::BacktestEngine::accounting_errors))
///   and keeps running, so the run shows where the accountant disagreed.
/// - [`FillAccountant::reset`] returns the accountant to its initial state
///   and is called by the engine's reset.
pub trait FillAccountant: fmt::Debug {
    /// Books one execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the accountant cannot book the execution; the
    /// accountant must then be unchanged.
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()>;

    /// Returns the current accounting state.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if a sum over the booked
    /// executions overflows. The snapshot is a running balance, so it never
    /// saturates silently.
    fn snapshot(&self) -> MMResult<AccountingSnapshot>;

    /// Returns to the initial state, which is the one
    /// [`FillAccountant::begin_from_current_state`] last fixed.
    fn reset(&mut self);

    /// Fixes the current state as where a run starts.
    ///
    /// The engine calls this when it is handed the accountant, so that its
    /// first run and a run after [`FillAccountant::reset`] start from the same
    /// state, whatever was posted before. Execution ids restart at 1 with each
    /// engine and each reset: after this call, an accountant must not take them
    /// for ids it booked before. By default nothing happens, for an accountant
    /// that keeps no such state.
    fn begin_from_current_state(&mut self) {}

    /// Human readable description for result reports.
    fn description(&self) -> String {
        "custom accountant".to_string()
    }

    /// The unrealized PnL of the open position at `mark`, for an accountant
    /// whose position is not worth `position * mark`, such as one of a derivative
    /// contract. `None` keeps that formula, which a derivative run refuses to
    /// fall back to.
    fn unrealized_at(&self, _mark: Decimal) -> Option<MMResult<Decimal>> {
        None
    }

    /// Whether this accountant can only run with
    /// `BacktestEngine::with_instrument`.
    fn requires_instrument(&self) -> bool {
        false
    }

    /// Checks that this accountant books `spec`, with the same economic terms and
    /// not only the same identifier, and fees at `fee_scale`.
    ///
    /// # Errors
    ///
    /// By default [`MMError::InvalidConfiguration`]: an accountant must declare
    /// that it can book a contract, or the terms could differ silently.
    #[cfg(feature = "instruments")]
    fn check_instrument(
        &self,
        spec: &financial_types::instrument::InstrumentSpec,
        _fee_scale: u32,
    ) -> MMResult<()> {
        Err(MMError::InvalidConfiguration(format!(
            "accountant {} cannot book instrument {}",
            self.description(),
            spec.id().as_str()
        )))
    }
}

#[cfg(feature = "margin")]
pub use margin_ledger::MarginLedgerAccountant;
#[cfg(all(feature = "margin", feature = "instruments"))]
pub(super) use margin_ledger::largest_execution_number;

#[cfg(feature = "margin")]
mod margin_ledger {
    use std::collections::BTreeMap;

    use super::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
    use crate::Decimal;
    use crate::types::action::Dir;
    use crate::types::decimal::OrOverflow;
    use crate::types::error::{MMError, MMResult};
    use margin_rs::Action;
    use margin_rs::ledger::{Booking, Execution, Ledger};
    use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};

    /// [`FillAccountant`] backed by the exact single account ledger of
    /// `margin-rs`.
    ///
    /// - Each fill becomes a ledger execution with id `sim-<n>`, where `n` is
    ///   its `execution_id` plus the largest such `n` in any journal of the
    ///   ledger when the run
    ///   started, so booking is idempotent and a run never reuses an execution
    ///   the ledger started with. Reset returns to that same start: the state
    ///   when an engine took the accountant, or when it was wrapped if none has.
    /// - Fills already happened at the venue, so they are booked even when a
    ///   balance goes negative (for example selling base the account never
    ///   held); the resulting shortfalls are kept in
    ///   [`MarginLedgerAccountant::overdrafts`] instead of refusing the fill.
    /// - On a future or a perpetual, the position is in contracts, the cash flow
    ///   is the ledger's realized PnL, and a non zero fee must be in the settlement
    ///   asset ([`FeeAsset::Settlement`]); the engine needs
    ///   `BacktestEngine::with_instrument`
    ///   for it. The ledger must have no other open position, because
    ///   [`Ledger::valuation`] needs a mark for each and a run has one. Below, on
    ///   a spot instrument:
    /// - Fees may be negative (rebates). Quote fees are debited from the
    ///   quote balance and base fees from the base balance; the snapshot
    ///   position is the net base held (traded quantity minus base fees)
    ///   and reports the base fees in base units for the engine to mark. A
    ///   fee in a third asset is refused as an accounting error.
    /// - The snapshot covers only executions of the selected instrument, so
    ///   other activity in the ledger never leaks into the backtest.
    /// - The ledger enforces its instrument and asset scales: configure
    ///   [`BacktestConfig::tick_size`](crate::backtest::BacktestConfig::tick_size),
    ///   [`BacktestConfig::lot_size`](crate::backtest::BacktestConfig::lot_size)
    ///   and the fee rate so values fit them, otherwise the post fails and is
    ///   reported as an accounting error.
    #[derive(Debug, Clone)]
    pub struct MarginLedgerAccountant {
        initial: Ledger,
        ledger: Ledger,
        instrument: InstrumentId,
        base: Asset,
        quote: Asset,
        /// A future or a perpetual: `quote` is its settlement asset.
        derivative: bool,
        /// Added to each execution id, past those the ledger started with.
        execution_offset: u64,
        overdrafts: BTreeMap<Asset, Decimal>,
        /// Sums over the journal, so that a snapshot never rescans it: the
        /// engine takes one after every fill.
        totals: Totals,
        initial_totals: Totals,
    }

    impl MarginLedgerAccountant {
        /// Wraps a ledger in which `instrument` is a registered instrument; its
        /// current state is what reset restores.
        ///
        /// # Errors
        ///
        /// Returns [`MMError::InvalidConfiguration`] if the instrument is not
        /// registered, or is a derivative in a ledger with another open position,
        /// or [`MMError::NumericalError`] if the base fees already in the ledger
        /// overflow.
        pub fn new(ledger: Ledger, instrument: InstrumentId) -> MMResult<Self> {
            let (base, quote, derivative) = match ledger.instrument(&instrument) {
                Some(Instrument::Spot(spot)) => (spot.base().clone(), spot.quote().clone(), false),
                // A derivative exchanges no base: only its settlement asset moves.
                Some(other) => (other.quote().clone(), other.quote().clone(), true),
                None => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "instrument {} is not registered in the ledger",
                        instrument.as_str()
                    )));
                }
            };
            if derivative
                && let Some((other, _)) = ledger
                    .snapshot()
                    .positions
                    .iter()
                    .find(|(id, position)| **id != instrument && !position.is_flat())
            {
                return Err(MMError::InvalidConfiguration(format!(
                    "instrument {} cannot be valued with {} open in the same ledger",
                    instrument.as_str(),
                    other.as_str()
                )));
            }
            let execution_offset = largest_execution_number(&ledger);
            // Refused here, before any fill; the other sums fail in the snapshot.
            ledger
                .journal()
                .iter()
                .filter(|e| e.instrument == instrument)
                .filter_map(|e| e.fee.as_ref().filter(|f| f.asset == base))
                .try_fold(Decimal::ZERO, |sum, f| sum.checked_add(f.amount))
                .or_overflow("ledger base fees")?;
            let mut totals = Totals::default();
            for entry in ledger
                .journal()
                .iter()
                .filter(|e| e.instrument == instrument)
            {
                totals.add(entry, derivative, &base, &quote);
            }
            Ok(Self {
                initial: ledger.clone(),
                ledger,
                instrument,
                base,
                quote,
                derivative,
                execution_offset,
                overdrafts: BTreeMap::new(),
                totals,
                initial_totals: totals,
            })
        }

        /// The underlying ledger.
        #[must_use]
        pub fn ledger(&self) -> &Ledger {
            &self.ledger
        }

        /// Largest shortfall per asset created by booking venue fills that
        /// the balance did not cover.
        #[must_use]
        pub fn overdrafts(&self) -> &BTreeMap<Asset, Decimal> {
            &self.overdrafts
        }
    }

    /// Running sums over the executions of one instrument, in booking order.
    #[derive(Debug, Clone, Copy, Default)]
    struct Totals {
        /// The sum of its cash flows, which a derivative's notional of zero leaves
        /// at zero.
        cash_flow: Decimal,
        /// Fees in the quote asset, which on a derivative is its settlement
        /// asset.
        fees: Decimal,
        /// On a spot instrument, fees in its base asset.
        base_fees: Decimal,
        /// Executions booked.
        fills: u64,
        /// The first sum that overflowed. It stays an error until a reset, even
        /// if a later fill would bring the sum back in range, as a sum over the
        /// journal in this order would.
        overflow: Option<&'static str>,
    }

    impl Totals {
        /// Adds one booked execution. `base` equals `quote` on a derivative, whose
        /// fees count once, as fees.
        fn add(
            &mut self,
            entry: &margin_rs::ledger::ExecutionCashflow,
            derivative: bool,
            base: &Asset,
            quote: &Asset,
        ) {
            let fee = entry.fee.as_ref();
            // Apart from the others: a position needs it even after one overflows.
            // Wrapping and posting refuse its own overflow, so it should not happen.
            if let Some(fee) = fee.filter(|f| !derivative && f.asset == *base) {
                match self.base_fees.checked_add(fee.amount) {
                    Some(sum) => self.base_fees = sum,
                    None => self.overflow = self.overflow.or(Some("ledger base fees")),
                }
            }
            if self.overflow.is_some() {
                return;
            }
            let sum = match Dir::try_from(entry.side) {
                Ok(Dir::Buy) => self.cash_flow.checked_sub(entry.notional),
                Ok(Dir::Sell) => self.cash_flow.checked_add(entry.notional),
                // margin-rs books only buys and sells and rejects any other
                // action before booking, so no journal entry has another side.
                Err(_) => Some(self.cash_flow),
            };
            let Some(sum) = sum else {
                self.overflow = Some("ledger cash flow");
                return;
            };
            self.cash_flow = sum;
            if let Some(fee) = fee.filter(|f| f.asset == *quote) {
                let Some(sum) = self.fees.checked_add(fee.amount) else {
                    self.overflow = Some("ledger fees");
                    return;
                };
                self.fees = sum;
            }
            // A count of journal entries: it saturates at `u64::MAX`.
            self.fills = self.fills.saturating_add(1);
        }

        /// The error of a sum that overflowed.
        fn check(&self) -> MMResult<()> {
            match self.overflow {
                Some(what) => None.or_overflow(what),
                None => Ok(()),
            }
        }
    }

    /// The largest `n` among the `sim-<n>` identifiers of `ledger`, in all the
    /// journals that share them. Any other identifier cannot collide with one.
    pub(in crate::backtest) fn largest_execution_number(ledger: &Ledger) -> u64 {
        ledger
            .journal()
            .iter()
            .map(|e| &e.execution_id)
            .chain(ledger.settlement_journal().iter().map(|e| &e.id))
            .chain(ledger.funding_journal().iter().map(|e| &e.id))
            .chain(ledger.expiry_journal().iter().map(|e| &e.id))
            .filter_map(|id| id.as_str().strip_prefix("sim-"))
            .filter_map(|n| n.parse::<u64>().ok())
            .max()
            .unwrap_or(0)
    }

    fn margin_error(context: &str, error: margin_rs::MarginError) -> MMError {
        MMError::InvalidPositionUpdate(format!("margin ledger {context}: {error}"))
    }

    impl MarginLedgerAccountant {
        /// The snapshot of a derivative: contracts, the ledger's realized PnL and
        /// the fees of this instrument, with no notional in it.
        fn derivative_snapshot(
            &self,
            position: Option<&margin_rs::ledger::PositionSnapshot>,
        ) -> MMResult<AccountingSnapshot> {
            self.totals.check()?;
            Ok(AccountingSnapshot {
                position: position.map_or(Decimal::ZERO, |p| p.quantity),
                average_entry_price: position
                    .and_then(margin_rs::ledger::PositionSnapshot::average_price),
                cash_flow: position.map_or(Decimal::ZERO, |p| p.realized_pnl),
                fees: self.totals.fees,
                fills_posted: self.totals.fills,
                ..AccountingSnapshot::default()
            })
        }
    }

    impl FillAccountant for MarginLedgerAccountant {
        fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
            if self.derivative && !fill.fee.is_zero() && fill.fee_asset != FeeAsset::Settlement {
                return Err(MMError::InvalidPositionUpdate(format!(
                    "a derivative fee must be in the settlement asset, got {:?}",
                    fill.fee_asset
                )));
            }
            let fee_asset = match &fill.fee_asset {
                FeeAsset::Settlement if !self.derivative && !fill.fee.is_zero() => {
                    return Err(MMError::InvalidPositionUpdate(
                        "a spot instrument has no settlement asset fee".to_string(),
                    ));
                }
                FeeAsset::Quote | FeeAsset::Settlement => self.quote.clone(),
                FeeAsset::Base => self.base.clone(),
                FeeAsset::Other(code) => {
                    if fill.fee.is_zero() {
                        self.quote.clone()
                    } else {
                        return Err(MMError::InvalidPositionUpdate(format!(
                            "margin ledger adapter cannot book a fee in {code}"
                        )));
                    }
                }
            };
            let number = self
                .execution_offset
                .checked_add(fill.execution_id)
                .or_overflow("execution id")?;
            let id = ExecutionId::new(format!("sim-{number}"))
                .map_err(|e| margin_error("execution id", e))?;
            // Before booking: a rejected post leaves the ledger unchanged.
            let side = Action::from(Dir::try_from(fill.side)?);
            // Only a new execution adds to the sum: an identical replay must book
            // as a duplicate, and one with other terms as a conflict.
            if matches!(fill.fee_asset, FeeAsset::Base)
                && self.ledger.execution_cashflow(&id).is_none()
            {
                self.totals
                    .base_fees
                    .checked_add(fill.fee)
                    .or_overflow("ledger base fees")?;
            }
            let mut execution =
                Execution::new(id, self.instrument.clone(), side, fill.price, fill.quantity);
            if self.derivative {
                execution = execution.with_executed_at(fill.timestamp);
            }
            // Negative fees are rebates and are booked as such.
            if !fill.fee.is_zero() {
                execution = execution.with_fee(fee_asset, fill.fee);
            }
            let booked = self
                .ledger
                .book_execution_allowing_overdraft(&execution, None)
                .map_err(|e| margin_error("booking", e))?;
            if let Booking::Booked(entry) = &booked.booking {
                self.totals
                    .add(entry, self.derivative, &self.base, &self.quote);
            }
            for (asset, shortfall) in booked.shortfalls {
                let worst = self.overdrafts.entry(asset).or_insert(Decimal::ZERO);
                *worst = (*worst).max(shortfall);
            }
            Ok(())
        }

        fn snapshot(&self) -> MMResult<AccountingSnapshot> {
            let position = self.ledger.position(&self.instrument);
            if self.derivative {
                return self.derivative_snapshot(position.as_ref());
            }
            // The account holds the traded quantity minus the base it paid as fees.
            let held = position
                .as_ref()
                .map_or(Decimal::ZERO, |p| p.quantity)
                .checked_sub(self.totals.base_fees)
                .or_overflow("ledger position")?;
            self.totals.check()?;
            Ok(AccountingSnapshot {
                position: held,
                average_entry_price: position.as_ref().and_then(|p| p.average_price()),
                cash_flow: self.totals.cash_flow,
                fees: self.totals.fees,
                base_fees: self.totals.base_fees,
                fills_posted: self.totals.fills,
            })
        }

        fn reset(&mut self) {
            self.ledger = self.initial.clone();
            self.overdrafts.clear();
            self.totals = self.initial_totals;
        }

        fn begin_from_current_state(&mut self) {
            self.execution_offset = largest_execution_number(&self.ledger);
            self.initial = self.ledger.clone();
            self.initial_totals = self.totals;
            self.overdrafts.clear();
        }

        fn description(&self) -> String {
            format!("margin-rs ledger for {}", self.instrument.as_str())
        }

        fn unrealized_at(&self, mark: Decimal) -> Option<MMResult<Decimal>> {
            if !self.derivative {
                return None;
            }
            let marks = BTreeMap::from([(self.instrument.clone(), mark)]);
            Some(
                self.ledger
                    .valuation(&marks)
                    .map_err(|e| margin_error("valuation", e))
                    .map(|valuation| {
                        valuation
                            .positions
                            .get(&self.instrument)
                            .map_or(Decimal::ZERO, |position| position.unrealized_pnl)
                    }),
            )
        }

        fn requires_instrument(&self) -> bool {
            self.derivative
        }

        #[cfg(feature = "instruments")]
        fn check_instrument(
            &self,
            spec: &financial_types::instrument::InstrumentSpec,
            fee_scale: u32,
        ) -> MMResult<()> {
            let refuse = |reason: String| {
                Err(MMError::InvalidConfiguration(format!(
                    "ledger instrument {} and specification {}: {reason}",
                    self.instrument.as_str(),
                    spec.id().as_str()
                )))
            };
            let Some(registered) = self.ledger.instrument(&self.instrument) else {
                return refuse("not registered".to_string());
            };
            let Some(scales) = crate::instruments::margin::registered_scales(registered) else {
                return refuse("not a derivative".to_string());
            };
            // The same scales, so any difference is in the terms.
            let expected = Instrument::from_spec(spec, scales)
                .map_err(|e| margin_error("specification", e))?;
            if expected != *registered {
                return refuse(format!("the terms differ: {registered:?} and {expected:?}"));
            }
            let settlement = self.ledger.snapshot().assets.get(&self.quote).copied();
            if settlement.is_none_or(|scale| fee_scale > scale) {
                return refuse(format!(
                    "fee scale {fee_scale} above the settlement asset's {settlement:?}"
                ));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn fill(side: Action, price: Decimal, quantity: Decimal) -> AccountingFill {
        AccountingFill {
            execution_id: 1,
            order_id: 1,
            side,
            price,
            quantity,
            cumulative_quantity: quantity,
            fee: Decimal::ZERO,
            fee_asset: FeeAsset::Quote,
            timestamp: 0,
        }
    }

    #[test]
    fn test_signed_quantity_and_cash_flow() {
        let buy = fill(Action::Buy, dec!(100), dec!(2));
        assert_eq!(buy.signed_quantity().unwrap(), dec!(2));
        assert_eq!(buy.cash_flow().unwrap(), dec!(-200));
        let sell = fill(Action::Sell, dec!(100), dec!(2));
        assert_eq!(sell.signed_quantity().unwrap(), dec!(-2));
        assert_eq!(sell.cash_flow().unwrap(), dec!(200));
    }

    #[test]
    fn test_cash_flow_notional_overflow_returns_err() {
        let huge = fill(Action::Buy, Decimal::MAX, dec!(2));
        assert!(matches!(
            huge.cash_flow(),
            Err(crate::types::error::MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_signed_quantity_and_cash_flow_unsupported_action_return_err() {
        let other = fill(Action::Other, dec!(100), dec!(2));
        let expected = Err(crate::types::error::MMError::UnsupportedAction {
            action: Action::Other,
        });
        assert_eq!(other.signed_quantity(), expected);
        assert_eq!(other.cash_flow(), expected);
    }
}
