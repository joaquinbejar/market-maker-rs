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
//! use market_maker_rs::execution::Side;
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
//!         self.snapshot.position += fill.signed_quantity();
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
use crate::execution::Side;
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
    /// Side of the order.
    pub side: Side,
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
    #[must_use]
    pub fn signed_quantity(&self) -> Decimal {
        match self.side {
            Side::Buy => self.quantity,
            Side::Sell => -self.quantity,
        }
    }

    /// Cash flow of the execution before fees: negative for a buy.
    #[must_use]
    pub fn cash_flow(&self) -> Decimal {
        -self.signed_quantity() * self.price
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
    use super::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
    use crate::Decimal;
    use crate::execution::Side;
    use crate::types::error::{MMError, MMResult};
    use margin_rs::ledger::{Execution, Ledger, Side as LedgerSide, TransferKind};
    use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId};

    /// [`FillAccountant`] backed by the exact single account ledger of
    /// `margin-rs`.
    ///
    /// Each fill becomes a ledger execution with id `sim-<execution_id>`, so
    /// booking is idempotent. The ledger enforces its instrument and asset
    /// scales: configure [`BacktestConfig::tick_size`](crate::backtest::BacktestConfig::tick_size),
    /// [`BacktestConfig::lot_size`](crate::backtest::BacktestConfig::lot_size)
    /// and the fee rate so prices, quantities and fees fit them, otherwise
    /// the post fails and is reported as an accounting error. Insufficient
    /// balance is also an error: the ledger never overdraws.
    #[derive(Debug, Clone)]
    pub struct MarginLedgerAccountant {
        initial: Ledger,
        ledger: Ledger,
        instrument: InstrumentId,
        base: Asset,
        quote: Asset,
    }

    impl MarginLedgerAccountant {
        /// Wraps a ledger in which `instrument` is a registered spot
        /// instrument; deposits made before wrapping are the starting
        /// balance restored by reset.
        ///
        /// # Errors
        ///
        /// Returns [`MMError::InvalidConfiguration`] if the instrument is not
        /// registered in the ledger.
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
            Ok(Self {
                initial: ledger.clone(),
                ledger,
                instrument,
                base,
                quote,
            })
        }

        /// The underlying ledger.
        #[must_use]
        pub fn ledger(&self) -> &Ledger {
            &self.ledger
        }

        fn fee_asset(&self, asset: FeeAsset) -> &Asset {
            match asset {
                FeeAsset::Quote => &self.quote,
                FeeAsset::Base => &self.base,
            }
        }

        fn net_deposits(&self, asset: &Asset) -> Decimal {
            self.ledger
                .transfers()
                .filter(|t| &t.asset == asset)
                .map(|t| match t.kind {
                    TransferKind::Deposit => t.amount,
                    TransferKind::Withdrawal => -t.amount,
                })
                .sum()
        }
    }

    fn margin_error(context: &str, error: margin_rs::MarginError) -> MMError {
        MMError::InvalidPositionUpdate(format!("margin ledger {context}: {error}"))
    }

    impl FillAccountant for MarginLedgerAccountant {
        fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
            let id = ExecutionId::new(format!("sim-{}", fill.execution_id))
                .map_err(|e| margin_error("execution id", e))?;
            let side = match fill.side {
                Side::Buy => LedgerSide::Buy,
                Side::Sell => LedgerSide::Sell,
            };
            let mut execution =
                Execution::new(id, self.instrument.clone(), side, fill.price, fill.quantity);
            if fill.fee > Decimal::ZERO {
                execution = execution.with_fee(self.fee_asset(fill.fee_asset).clone(), fill.fee);
            }
            self.ledger
                .book_execution(&execution)
                .map_err(|e| margin_error("booking", e))?;
            Ok(())
        }

        fn snapshot(&self) -> AccountingSnapshot {
            let position = self.ledger.position(&self.instrument);
            let quantity = position.as_ref().map_or(Decimal::ZERO, |p| p.quantity);
            let average_entry_price = position.as_ref().and_then(|p| p.average_price());
            let fees = self.ledger.fees_paid(&self.quote);
            let total = self
                .ledger
                .balance(&self.quote)
                .map_or(Decimal::ZERO, |b| b.total);
            AccountingSnapshot {
                position: quantity,
                average_entry_price,
                // Quote balance change from trading, fees added back.
                cash_flow: total - self.net_deposits(&self.quote) + fees,
                fees,
                fills_posted: self.ledger.journal().len() as u64,
            }
        }

        fn reset(&mut self) {
            self.ledger = self.initial.clone();
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

    fn fill(side: Side, price: Decimal, quantity: Decimal) -> AccountingFill {
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
        let buy = fill(Side::Buy, dec!(100), dec!(2));
        assert_eq!(buy.signed_quantity(), dec!(2));
        assert_eq!(buy.cash_flow(), dec!(-200));
        let sell = fill(Side::Sell, dec!(100), dec!(2));
        assert_eq!(sell.signed_quantity(), dec!(-2));
        assert_eq!(sell.cash_flow(), dec!(200));
    }
}
