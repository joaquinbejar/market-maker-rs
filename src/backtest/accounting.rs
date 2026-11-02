//! Accounting port at the backtest fill boundary.
//!
//! Every fill the [`BacktestEngine`] simulates is
//! posted to exactly one [`Accountant`], once, with its fee and its currency
//! context ([`AccountingFill`]). The engine keeps no second copy of the
//! position, the PnL or the fees: [`BacktestEngine::get_state`],
//! [`BacktestEngine::accounting`], the equity curve and the
//! [`BacktestResult`] all read the accountant's
//! immutable [`AccountingSnapshot`].
//!
//! The default accountant is [`LegacyAccountant`], which reproduces the
//! engine's historical figures exactly, so existing results do not change.
//! A single ledger authority (for example margin-rs) plugs in through
//! [`BacktestEngine::with_accountant`].
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::accounting::{
//!     Accountant, AccountingFill, AccountingSnapshot, FillContext, LegacyAccountant,
//! };
//! use market_maker_rs::execution::Side;
//! use market_maker_rs::dec;
//!
//! let mut accountant = LegacyAccountant::new();
//! let fill = AccountingFill {
//!     fill_id: 1,
//!     order_id: 7,
//!     side: Side::Buy,
//!     price: dec!(100),
//!     quantity: dec!(2),
//!     fee: dec!(0.2),
//!     timestamp: 1_000,
//!     context: FillContext::new("BTC-USDC", "BTC", "USDC"),
//! };
//! accountant.post_fill(&fill).unwrap();
//! let snapshot: AccountingSnapshot = accountant.snapshot();
//! assert_eq!(snapshot.position.quantity, dec!(2));
//! assert_eq!(snapshot.total_fees, dec!(0.2));
//! assert_eq!(snapshot.fills_posted, 1);
//! ```

use core::fmt;
use std::panic::{RefUnwindSafe, UnwindSafe};

use crate::Decimal;
use crate::execution::Side;
use crate::position::inventory::InventoryPosition;
use crate::position::pnl::PnL;
use crate::types::error::{MMError, MMResult};

use super::lifecycle::SimOrderId;

/// Instrument and currencies of the fills a backtest posts.
///
/// Set with [`BacktestEngine::with_accountant`](crate::backtest::BacktestEngine::with_accountant).
/// The default is empty (no currency configured), which the
/// [`LegacyAccountant`] does not need.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FillContext {
    /// Instrument traded, for example `BTC-USDC`.
    pub instrument: String,
    /// Currency received by a buy, for example `BTC`.
    pub base_currency: String,
    /// Currency paid by a buy, for example `USDC`.
    pub quote_currency: String,
    /// Currency fees are charged in. The engine computes every fee as
    /// `price * quantity * fee_rate`, an amount in the quote currency, so
    /// this must equal `quote_currency` ([`FillContext::validate`]).
    pub fee_currency: String,
}

impl FillContext {
    /// A context whose fees are charged in the quote currency.
    #[must_use]
    pub fn new(
        instrument: impl Into<String>,
        base_currency: impl Into<String>,
        quote_currency: impl Into<String>,
    ) -> Self {
        let quote_currency = quote_currency.into();
        Self {
            instrument: instrument.into(),
            base_currency: base_currency.into(),
            fee_currency: quote_currency.clone(),
            quote_currency,
        }
    }

    /// Checks that fees are in the quote currency, the only currency the
    /// engine computes them in. Fees charged in another currency (for
    /// example the base) are not supported yet.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] if `fee_currency` differs from
    /// `quote_currency`.
    pub fn validate(&self) -> MMResult<()> {
        if self.fee_currency != self.quote_currency {
            return Err(MMError::InvalidConfiguration(format!(
                "fee currency {} must be the quote currency {}: fees are computed in quote",
                self.fee_currency, self.quote_currency
            )));
        }
        Ok(())
    }
}

/// One fill as posted to an [`Accountant`].
#[derive(Debug, Clone, PartialEq)]
pub struct AccountingFill {
    /// Identifier of the fill, unique within a run: 1 for the first fill,
    /// then increasing by one. An accountant can book idempotently by it.
    pub fill_id: u64,
    /// Simulated order the fill belongs to.
    pub order_id: SimOrderId,
    /// Side of the order.
    pub side: Side,
    /// Execution price, slippage included.
    pub price: Decimal,
    /// Executed quantity, positive.
    pub quantity: Decimal,
    /// Fee charged for the fill, in [`FillContext::fee_currency`].
    pub fee: Decimal,
    /// Simulated time of the fill, in milliseconds.
    pub timestamp: u64,
    /// Instrument and currencies.
    pub context: FillContext,
}

/// Immutable accounting results read by the engine and by metrics.
#[derive(Clone, PartialEq)]
pub struct AccountingSnapshot {
    /// Current position.
    pub position: InventoryPosition,
    /// Current PnL, before fees.
    pub pnl: PnL,
    /// Cumulative fees.
    pub total_fees: Decimal,
    /// Number of fills posted.
    pub fills_posted: u64,
}

impl AccountingSnapshot {
    /// The empty state: flat, no PnL, no fees, no fills.
    #[must_use]
    pub fn new() -> Self {
        Self {
            position: InventoryPosition::new(),
            pnl: PnL::new(),
            total_fees: Decimal::ZERO,
            fills_posted: 0,
        }
    }

    /// `pnl.total - total_fees`.
    #[must_use]
    pub fn net_pnl(&self) -> Decimal {
        self.pnl.total - self.total_fees
    }
}

impl Default for AccountingSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for AccountingSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccountingSnapshot")
            .field("quantity", &self.position.quantity)
            .field("avg_entry_price", &self.position.avg_entry_price)
            .field("realized", &self.pnl.realized)
            .field("unrealized", &self.pnl.unrealized)
            .field("total", &self.pnl.total)
            .field("total_fees", &self.total_fees)
            .field("fills_posted", &self.fills_posted)
            .finish()
    }
}

/// The single accounting authority of a backtest.
///
/// The supertraits keep [`BacktestEngine`](crate::backtest::BacktestEngine)
/// `Send`, `Sync` and unwind safe, as it was before the port.
///
/// The engine calls [`Accountant::post_fill`] once per simulated fill, in
/// fill order, and [`Accountant::mark_to_market`] once per tick, after that
/// tick's fills. It reads results only through [`Accountant::snapshot`].
pub trait Accountant: fmt::Debug + Send + Sync + UnwindSafe + RefUnwindSafe {
    /// Books one fill.
    ///
    /// # Errors
    ///
    /// Accountant specific. The engine records the error (see
    /// [`BacktestEngine::accounting_errors`](crate::backtest::BacktestEngine::accounting_errors))
    /// and the simulation goes on: a fill that happened is never undone.
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()>;

    /// Revalues the position at `mid_price`.
    ///
    /// # Errors
    ///
    /// Accountant specific; recorded like [`Accountant::post_fill`] errors.
    fn mark_to_market(&mut self, mid_price: Decimal, timestamp: u64) -> MMResult<()>;

    /// Current results.
    fn snapshot(&self) -> AccountingSnapshot;

    /// Returns to the empty state for another run.
    fn reset(&mut self);
}

/// The engine's historical accounting, kept as the default so existing
/// results stay identical.
///
/// - Position: [`InventoryPosition::update_fill`] (weighted average entry).
/// - `pnl.realized` accumulates the signed cash flow of each fill (minus
///   the notional of a buy, plus that of a sell); `pnl.unrealized` is
///   `quantity * mid` at the last mark, so `pnl.total` is the cash flow
///   plus the inventory value.
/// - Fees accumulate in `total_fees`, apart from the PnL.
///
/// It ignores the currency context and never fails. Its arithmetic is the
/// historical unchecked `Decimal` arithmetic, kept on purpose so legacy
/// results stay bit for bit identical; an exact, checked ledger plugs in
/// as another [`Accountant`].
#[derive(Debug, Clone, Default)]
pub struct LegacyAccountant {
    state: AccountingSnapshot,
}

impl LegacyAccountant {
    /// A flat accountant.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Accountant for LegacyAccountant {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        let signed_quantity = match fill.side {
            Side::Buy => fill.quantity,
            Side::Sell => -fill.quantity,
        };
        self.state
            .position
            .update_fill(signed_quantity, fill.price, fill.timestamp);
        let notional = fill.price * fill.quantity;
        let cash_flow = match fill.side {
            Side::Buy => -notional,
            Side::Sell => notional,
        };
        self.state.pnl.add_realized(cash_flow);
        self.state.total_fees += fill.fee;
        self.state.fills_posted += 1;
        Ok(())
    }

    fn mark_to_market(&mut self, mid_price: Decimal, _timestamp: u64) -> MMResult<()> {
        let pnl = &mut self.state.pnl;
        pnl.unrealized = self.state.position.quantity * mid_price;
        pnl.total = pnl.realized + pnl.unrealized;
        Ok(())
    }

    fn snapshot(&self) -> AccountingSnapshot {
        self.state.clone()
    }

    fn reset(&mut self) {
        self.state = AccountingSnapshot::new();
    }
}
