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
//! exact ledger of the `margin-rs` crate.
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
//!     fn snapshot(&self) -> AccountingSnapshot {
//!         self.snapshot.clone()
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
    /// replays of the same run, so it can serve as an idempotency key.
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
    /// Net position in base units (positive long).
    pub position: Decimal,
    /// Average entry price of the open position, if any.
    pub average_entry_price: Option<Decimal>,
    /// Sum of execution cash flows, excluding fees.
    pub cash_flow: Decimal,
    /// Fees fixed in the quote asset (quote fees and third asset fees
    /// converted to quote).
    pub fees: Decimal,
    /// Fees paid in the base asset, in base units. They are already
    /// deducted from [`AccountingSnapshot::position`]; the engine marks
    /// them at the current price.
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
    fn snapshot(&self) -> AccountingSnapshot;

    /// Returns to the initial state.
    fn reset(&mut self);

    /// Human readable description for result reports.
    fn description(&self) -> String {
        "custom accountant".to_string()
    }
}

#[cfg(feature = "margin")]
pub use margin_ledger::MarginLedgerAccountant;

#[cfg(feature = "margin")]
mod margin_ledger {
    use std::collections::BTreeMap;

    use super::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
    use crate::Decimal;
    use crate::types::action::Dir;
    use crate::types::error::{MMError, MMResult};
    use margin_rs::Action;
    use margin_rs::ledger::{Execution, Ledger};
    use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};

    /// [`FillAccountant`] backed by the exact single account ledger of
    /// `margin-rs`.
    ///
    /// - Each fill becomes a ledger execution with id `sim-<execution_id>`,
    ///   so booking is idempotent.
    /// - Fills already happened at the venue, so they are booked even when a
    ///   balance goes negative (for example selling base the account never
    ///   held); the resulting shortfalls are kept in
    ///   [`MarginLedgerAccountant::overdrafts`] instead of refusing the fill.
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
        overdrafts: BTreeMap<Asset, Decimal>,
        /// Base fees booked on this instrument, kept incrementally so a
        /// snapshot never rescans the journal for them.
        base_fees: Decimal,
        initial_base_fees: Decimal,
    }

    impl MarginLedgerAccountant {
        /// Wraps a ledger in which `instrument` is a registered spot
        /// instrument; its current state is what reset restores.
        ///
        /// # Errors
        ///
        /// Returns [`MMError::InvalidConfiguration`] if the instrument is not
        /// a registered spot instrument.
        pub fn new(ledger: Ledger, instrument: InstrumentId) -> MMResult<Self> {
            let (base, quote) = match ledger.instrument(&instrument) {
                Some(Instrument::Spot(spot)) => (spot.base().clone(), spot.quote().clone()),
                Some(_) => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "instrument {} is not a spot instrument",
                        instrument.as_str()
                    )));
                }
                None => {
                    return Err(MMError::InvalidConfiguration(format!(
                        "instrument {} is not registered in the ledger",
                        instrument.as_str()
                    )));
                }
            };
            let base_fees = ledger
                .journal()
                .iter()
                .filter(|e| e.instrument == instrument)
                .filter_map(|e| e.fee.as_ref().filter(|f| f.asset == base))
                .fold(Decimal::ZERO, |sum, f| sum.saturating_add(f.amount));
            Ok(Self {
                initial: ledger.clone(),
                ledger,
                instrument,
                base,
                quote,
                overdrafts: BTreeMap::new(),
                base_fees,
                initial_base_fees: base_fees,
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

    fn margin_error(context: &str, error: margin_rs::MarginError) -> MMError {
        MMError::InvalidPositionUpdate(format!("margin ledger {context}: {error}"))
    }

    impl FillAccountant for MarginLedgerAccountant {
        fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
            let fee_asset = match &fill.fee_asset {
                FeeAsset::Quote => self.quote.clone(),
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
            let id = ExecutionId::new(format!("sim-{}", fill.execution_id))
                .map_err(|e| margin_error("execution id", e))?;
            // Before booking: a rejected post leaves the ledger unchanged.
            let side = Action::from(Dir::try_from(fill.side)?);
            let mut execution =
                Execution::new(id, self.instrument.clone(), side, fill.price, fill.quantity);
            // Negative fees are rebates and are booked as such.
            if !fill.fee.is_zero() {
                execution = execution.with_fee(fee_asset, fill.fee);
            }
            let booked = self
                .ledger
                .book_execution_allowing_overdraft(&execution, None)
                .map_err(|e| margin_error("booking", e))?;
            if fill.fee_asset == FeeAsset::Base {
                self.base_fees = self.base_fees.saturating_add(fill.fee);
            }
            for (asset, shortfall) in booked.shortfalls {
                let worst = self.overdrafts.entry(asset).or_insert(Decimal::ZERO);
                *worst = (*worst).max(shortfall);
            }
            Ok(())
        }

        fn snapshot(&self) -> AccountingSnapshot {
            let position = self.ledger.position(&self.instrument);
            let mut snapshot = AccountingSnapshot {
                // The account holds the traded quantity minus the base it
                // paid as fees.
                position: position
                    .as_ref()
                    .map_or(Decimal::ZERO, |p| p.quantity)
                    .saturating_sub(self.base_fees),
                average_entry_price: position.as_ref().and_then(|p| p.average_price()),
                base_fees: self.base_fees,
                ..AccountingSnapshot::default()
            };
            for entry in self
                .ledger
                .journal()
                .iter()
                .filter(|e| e.instrument == self.instrument)
            {
                // A snapshot cannot fail, so the sums saturate at the
                // `Decimal` bounds instead of panicking.
                snapshot.cash_flow = match Dir::try_from(entry.side) {
                    Ok(Dir::Buy) => snapshot.cash_flow.saturating_sub(entry.notional),
                    Ok(Dir::Sell) => snapshot.cash_flow.saturating_add(entry.notional),
                    // margin-rs books only buys and sells and rejects any
                    // other action before booking, so no journal entry has
                    // another side.
                    Err(_) => snapshot.cash_flow,
                };
                if let Some(fee) = entry.fee.as_ref().filter(|f| f.asset == self.quote) {
                    snapshot.fees = snapshot.fees.saturating_add(fee.amount);
                }
                snapshot.fills_posted = snapshot.fills_posted.saturating_add(1);
            }
            snapshot
        }

        fn reset(&mut self) {
            self.ledger = self.initial.clone();
            self.overdrafts.clear();
            self.base_fees = self.initial_base_fees;
        }

        fn description(&self) -> String {
            format!("margin-rs ledger for {}", self.instrument.as_str())
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
