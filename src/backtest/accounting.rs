//! Accounting port at the backtest fill boundary.
//!
//! By default [`BacktestEngine`](super::BacktestEngine) keeps its own
//! position, cash flow and fee totals. A [`FillAccountant`] replaces that
//! built in bookkeeping: once selected with
//! [`BacktestEngine::with_accountant`](super::BacktestEngine::with_accountant),
//! every fill is posted to it exactly once and the result metrics (position,
//! PnL, fees, equity) are read back from its [`AccountingSnapshot`]. Only one
//! accountant is ever active, so the engine never books a fill twice.
//!
//! With the `margin` feature, [`MarginLedgerAccountant`] posts fills to the
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
use crate::types::error::MMResult;

use super::lifecycle::SimOrderId;

/// Asset in which a fill's fee is charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeeAsset {
    /// The quote asset of the instrument (the engine's convention).
    Quote,
    /// The base asset of the instrument.
    Base,
}

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
    /// Returns [`MMError::UnsupportedAction`](crate::types::error::MMError::UnsupportedAction)
    /// if the side is neither buy nor sell.
    pub fn signed_quantity(&self) -> MMResult<Decimal> {
        Ok(match Dir::try_from(self.side)? {
            Dir::Buy => self.quantity,
            Dir::Sell => -self.quantity,
        })
    }

    /// Cash flow of the execution before fees: negative for a buy.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`](crate::types::error::MMError::UnsupportedAction)
    /// if the side is neither buy nor sell.
    pub fn cash_flow(&self) -> MMResult<Decimal> {
        Ok(-self.signed_quantity()? * self.price)
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
    /// Fees paid, in the quote asset.
    pub fees: Decimal,
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
///   [`BacktestEngine::accounting_errors`](super::BacktestEngine::accounting_errors))
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
    use margin_rs::ledger::{Execution, Ledger, Side as LedgerSide};
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
    /// - Fees may be negative (rebates). Only quote denominated fees are
    ///   supported; a base fee is refused as an accounting error.
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
        quote: Asset,
        overdrafts: BTreeMap<Asset, Decimal>,
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
            let quote = match ledger.instrument(&instrument) {
                Some(Instrument::Spot(spot)) => spot.quote().clone(),
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
            Ok(Self {
                initial: ledger.clone(),
                ledger,
                instrument,
                quote,
                overdrafts: BTreeMap::new(),
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
            if fill.fee_asset == FeeAsset::Base && !fill.fee.is_zero() {
                return Err(MMError::InvalidPositionUpdate(
                    "margin ledger adapter supports quote denominated fees only".to_string(),
                ));
            }
            let id = ExecutionId::new(format!("sim-{}", fill.execution_id))
                .map_err(|e| margin_error("execution id", e))?;
            // Before booking: a rejected post leaves the ledger unchanged.
            let side = match Dir::try_from(fill.side)? {
                Dir::Buy => LedgerSide::Buy,
                Dir::Sell => LedgerSide::Sell,
            };
            let mut execution =
                Execution::new(id, self.instrument.clone(), side, fill.price, fill.quantity);
            // Negative fees are rebates and are booked as such.
            if !fill.fee.is_zero() {
                execution = execution.with_fee(self.quote.clone(), fill.fee);
            }
            let booked = self
                .ledger
                .book_execution_allowing_overdraft(&execution, None)
                .map_err(|e| margin_error("booking", e))?;
            for (asset, shortfall) in booked.shortfalls {
                let worst = self.overdrafts.entry(asset).or_insert(Decimal::ZERO);
                *worst = (*worst).max(shortfall);
            }
            Ok(())
        }

        fn snapshot(&self) -> AccountingSnapshot {
            let position = self.ledger.position(&self.instrument);
            let mut snapshot = AccountingSnapshot {
                position: position.as_ref().map_or(Decimal::ZERO, |p| p.quantity),
                average_entry_price: position.as_ref().and_then(|p| p.average_price()),
                ..AccountingSnapshot::default()
            };
            for entry in self
                .ledger
                .journal()
                .iter()
                .filter(|e| e.instrument == self.instrument)
            {
                snapshot.cash_flow += match entry.side {
                    LedgerSide::Buy => -entry.notional,
                    LedgerSide::Sell => entry.notional,
                };
                if let Some(fee) = entry.fee.as_ref().filter(|f| f.asset == self.quote) {
                    snapshot.fees += fee.amount;
                }
                snapshot.fills_posted += 1;
            }
            snapshot
        }

        fn reset(&mut self) {
            self.ledger = self.initial.clone();
            self.overdrafts.clear();
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
    fn test_signed_quantity_and_cash_flow_unsupported_action_return_err() {
        let other = fill(Action::Other, dec!(100), dec!(2));
        let expected = Err(crate::types::error::MMError::UnsupportedAction {
            action: Action::Other,
        });
        assert_eq!(other.signed_quantity(), expected);
        assert_eq!(other.cash_flow(), expected);
    }
}
