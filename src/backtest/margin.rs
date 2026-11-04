//! `margin-rs` risk gate behind the pretrade and accounting ports.
//!
//! [`MarginAccount`] shares one `margin_rs::risk::RiskGate` (which owns the
//! account's ledger) between a [`PretradePolicy`](crate::risk::pretrade::PretradePolicy) and a [`FillAccountant`]:
//!
//! - [`MarginAccount::policy`] submits every order to the gate (reservation
//!   and hard limits), forwards venue accepts, rejects, cancel requests and
//!   confirmations, and books each fill through the gate exactly once;
//! - [`MarginAccount::accountant`] reads position, cash and fees back from
//!   the gate's ledger, so the engine metrics come from the same authority.
//!
//! Gate errors on lifecycle events are kept in [`MarginAccount::errors`];
//! the gate trips itself on accounting breaches and execution anomalies, and
//! then rejects every new order.
//!
//! Fees in the base asset: the gate reserves and checks them only when its
//! limits declare `fee_asset: margin_rs::risk::FeeAsset::Base`; with the
//! default (quote) declaration a base fee is booked from the free balance.
//! The declaration that matches the backtest's
//! `FeeCurrency::Base { scale }`, which rounds each fill half to even at
//! `scale`, is `fee_rounding: FeeRounding::PerFill { mode:
//! Rounding::HalfEven, min_fill: None }`, with `max_fee_rate` at least the
//! fee rate and the base asset registered at `scale`. The default rounding
//! (every fill rounded up) also never breaches, at the cost of a larger
//! reservation. A fill outside the declaration is booked, recorded in
//! [`MarginAccount::errors`] as a fee breach, and the gate rejects new
//! orders on the instrument until `RiskGate::reset_fee_declaration`.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use margin_rs::ledger::{Execution, TransferKind};
use margin_rs::risk::{
    FillReport, OrderPurpose as GateOrderPurpose, OrderRequest, RiskDecision, RiskGate,
};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId, OrderId};

use super::accounting::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
use crate::Decimal;
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::types::action::{Action, Dir};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

#[derive(Debug)]
struct Shared {
    initial: RiskGate,
    gate: RiskGate,
    instrument: InstrumentId,
    base: Asset,
    quote: Asset,
    errors: Vec<String>,
    fault: Option<MMError>,
}

/// One `margin-rs` account (risk gate and ledger) shared by a pretrade
/// policy and an accountant. Cloning shares the same account.
#[derive(Clone)]
pub struct MarginAccount {
    shared: Arc<Mutex<Shared>>,
}

impl fmt::Debug for MarginAccount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shared = self.lock();
        f.debug_struct("MarginAccount")
            .field("instrument", &shared.instrument.as_str())
            .field("errors", &shared.errors.len())
            .finish()
    }
}

impl MarginAccount {
    /// Wraps a configured gate (assets, spot instrument, deposits and
    /// limits already set). Its current state is what reset restores.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `instrument` is not a
    /// spot instrument registered in the gate's ledger.
    pub fn new(gate: RiskGate, instrument: InstrumentId) -> MMResult<Self> {
        let (base, quote) = match gate.ledger().instrument(&instrument) {
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
            shared: Arc::new(Mutex::new(Shared {
                initial: gate.clone(),
                gate,
                instrument,
                base,
                quote,
                errors: Vec::new(),
                fault: None,
            })),
        })
    }

    /// Pretrade policy backed by the gate.
    #[must_use]
    pub fn policy(&self) -> MarginRiskPolicy {
        MarginRiskPolicy {
            account: self.clone(),
        }
    }

    /// Accountant reading the gate's ledger.
    #[must_use]
    pub fn accountant(&self) -> MarginGateAccountant {
        MarginGateAccountant {
            account: self.clone(),
        }
    }

    /// Runs `f` with read access to the gate.
    pub fn with_gate<R>(&self, f: impl FnOnce(&RiskGate) -> R) -> R {
        f(&self.lock().gate)
    }

    /// Gate errors raised while forwarding lifecycle events.
    #[must_use]
    pub fn errors(&self) -> Vec<String> {
        self.lock().errors.clone()
    }

    /// The fault latched by a fill with an unsupported side, if any, or
    /// [`MMError::LockPoisoned`] when a thread panicked while holding the
    /// account. While either is set, the policy rejects every order; reset
    /// clears both.
    #[must_use]
    pub fn fault(&self) -> Option<MMError> {
        match self.shared.lock() {
            Ok(shared) => shared.fault.clone(),
            Err(_) => Some(Self::poisoned()),
        }
    }

    fn poisoned() -> MMError {
        MMError::LockPoisoned("margin account".to_string())
    }

    /// Locks the shared account for a decision or a booking: a poisoned
    /// lock fails closed with [`MMError::LockPoisoned`], since a panic may
    /// have interrupted an update.
    fn lock_checked(&self) -> MMResult<MutexGuard<'_, Shared>> {
        self.shared.lock().map_err(|_| Self::poisoned())
    }

    /// Locks the shared account for a read only view (debug output,
    /// descriptions, snapshots); a poisoned lock is recovered there, and
    /// decisions still fail closed through [`MarginAccount::lock_checked`].
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Restores the initial gate and clears errors, the fault and a
    /// poisoned lock.
    fn reset(&self) {
        let mut shared = self.lock();
        shared.gate = shared.initial.clone();
        shared.errors.clear();
        shared.fault = None;
        // Cleared while the guard is held, so no other thread can observe
        // the restored state as poisoned or a poisoned state as clean.
        self.shared.clear_poison();
    }
}

fn order_id(raw: &str) -> margin_rs::Result<OrderId> {
    OrderId::new(format!("sim-order-{raw}"))
}

/// [`PretradePolicy`] over a shared [`MarginAccount`].
///
/// A fill whose side is neither buy nor sell latches a fault on the account
/// ([`MarginAccount::fault`]): every order is then rejected until reset.
#[derive(Debug, Clone)]
pub struct MarginRiskPolicy {
    account: MarginAccount,
}

impl MarginRiskPolicy {
    fn record<T>(shared: &mut Shared, context: &str, result: margin_rs::Result<T>) {
        if let Err(error) = result {
            shared.errors.push(format!("{context}: {error}"));
        }
    }
}

impl PretradePolicy for MarginRiskPolicy {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        let dir = match Dir::try_from(request.side) {
            Ok(dir) => dir,
            Err(error) => {
                return PretradeDecision::Rejected {
                    reason: error.to_string(),
                };
            }
        };
        let mut shared = match self.account.lock_checked() {
            Ok(shared) => shared,
            Err(error) => {
                return PretradeDecision::Rejected {
                    reason: format!("policy fault: {error}"),
                };
            }
        };
        if let Some(fault) = &shared.fault {
            return PretradeDecision::Rejected {
                reason: format!("policy fault: {fault}"),
            };
        }
        if let Some(asked) = &request.instrument
            && asked != shared.instrument.as_str()
        {
            return PretradeDecision::Rejected {
                reason: format!(
                    "margin account trades {}, not {asked}",
                    shared.instrument.as_str()
                ),
            };
        }
        let Some(price) = request.price else {
            return PretradeDecision::Rejected {
                reason: "margin gate needs a limit price".to_string(),
            };
        };
        let id = match order_id(&request.order_id) {
            Ok(id) => id,
            Err(error) => {
                return PretradeDecision::Rejected {
                    reason: format!("invalid order id: {error}"),
                };
            }
        };
        let purpose = match request.purpose {
            OrderPurpose::Quote => GateOrderPurpose::Quote,
            OrderPurpose::Hedge => GateOrderPurpose::Hedge,
            OrderPurpose::Other => GateOrderPurpose::Other,
        };
        let order = OrderRequest::new(
            id,
            shared.instrument.clone(),
            Action::from(dir),
            price,
            request.quantity,
        )
        .reducible()
        .with_purpose(purpose);
        match shared.gate.submit(&order, request.timestamp) {
            RiskDecision::Accepted { quantity } if quantity >= request.quantity => {
                PretradeDecision::Accepted
            }
            RiskDecision::Accepted { quantity } => PretradeDecision::Reduced {
                quantity,
                reason: "margin gate approved less".to_string(),
            },
            RiskDecision::Reduced {
                quantity, reasons, ..
            } => PretradeDecision::Reduced {
                quantity,
                reason: format!("{reasons:?}"),
            },
            RiskDecision::Rejected { reasons } => PretradeDecision::Rejected {
                reason: format!("{reasons:?}"),
            },
        }
    }

    fn on_market_data(&mut self, mark: Decimal, timestamp: u64) {
        // A poisoned account stays poisoned (orders are rejected) until
        // reset; the mark is not forwarded to a possibly torn gate.
        let Ok(mut shared) = self.account.lock_checked() else {
            return;
        };
        let instrument = shared.instrument.clone();
        let result = shared.gate.on_market_data(&instrument, mark, timestamp);
        Self::record(&mut shared, "market data", result);
    }

    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        let mut shared = self.account.lock_checked()?;
        // A fill with an unsupported side is unknown exposure, whether or
        // not the gate saw the order: latch the fault and fail, booking
        // nothing.
        if let OrderEvent::Fill {
            fee,
            fee_asset: FeeAsset::Other(asset),
            ..
        } = event
            && !fee.is_zero()
        {
            // The account has no such asset: the fee cannot be booked.
            let error = MMError::InvalidPositionUpdate(format!(
                "margin account cannot book a fee in {asset}"
            ));
            shared.fault = Some(error.clone());
            return Err(error);
        }
        if let OrderEvent::Fill {
            fee,
            fee_asset: FeeAsset::Settlement,
            ..
        } = event
            && !fee.is_zero()
        {
            // The account is spot: it has no settlement asset.
            let error = MMError::InvalidPositionUpdate(
                "margin account cannot book a settlement asset fee".to_string(),
            );
            shared.fault = Some(error.clone());
            return Err(error);
        }
        if let OrderEvent::Fill { side, .. } = event
            && let Err(error) = Dir::try_from(*side)
        {
            shared.fault = Some(error.clone());
            return Err(error);
        }
        let raw = match event {
            OrderEvent::Accepted { order_id }
            | OrderEvent::VenueRejected { order_id }
            | OrderEvent::CancelRequested { order_id }
            | OrderEvent::Canceled { order_id }
            | OrderEvent::CancelRejected { order_id }
            | OrderEvent::Fill { order_id, .. } => order_id,
        };
        let Ok(id) = order_id(raw) else {
            return Ok(());
        };
        // Orders the gate never saw (rejected locally) are ignored.
        if shared.gate.order(&id).is_none() {
            return Ok(());
        }
        match event {
            OrderEvent::Accepted { .. } => {
                let result = shared.gate.on_venue_accepted(&id);
                Self::record(&mut shared, "venue accepted", result);
            }
            OrderEvent::VenueRejected { .. } => {
                let result = shared.gate.on_venue_rejected(&id);
                Self::record(&mut shared, "venue rejected", result);
            }
            OrderEvent::CancelRequested { .. } => {
                let result = shared.gate.request_cancel(&id);
                Self::record(&mut shared, "cancel requested", result);
            }
            OrderEvent::Canceled { .. } => {
                let result = shared.gate.on_cancel_confirmed(&id);
                Self::record(&mut shared, "cancel confirmed", result);
            }
            OrderEvent::CancelRejected { .. } => {}
            OrderEvent::Fill {
                execution_id,
                side,
                price,
                quantity,
                cumulative_quantity,
                fee,
                fee_asset,
                ..
            } => {
                // Checked at the top of the method.
                let dir = Dir::try_from(*side)?;
                let execution_id = match ExecutionId::new(execution_id) {
                    Ok(id) => id,
                    Err(error) => {
                        shared.errors.push(format!("execution id: {error}"));
                        return Ok(());
                    }
                };
                let mut execution = Execution::new(
                    execution_id,
                    shared.instrument.clone(),
                    Action::from(dir),
                    *price,
                    *quantity,
                );
                // Negative fees are rebates and are booked as such.
                if !fee.is_zero() {
                    let asset = match fee_asset {
                        FeeAsset::Quote => shared.quote.clone(),
                        FeeAsset::Base => shared.base.clone(),
                        // Refused at the top of the method.
                        FeeAsset::Other(_) | FeeAsset::Settlement => return Ok(()),
                    };
                    execution = execution.with_fee(asset, *fee);
                }
                let report = FillReport::new(execution, *cumulative_quantity);
                let result = shared.gate.on_fill(&id, &report);
                match result {
                    Ok(outcome) => {
                        if !outcome.is_clean() {
                            shared.errors.push(format!(
                                "fill anomalies: {:?}, shortfalls: {:?}",
                                outcome.anomalies, outcome.shortfalls
                            ));
                        }
                        // Not part of `is_clean`: the fill is booked, but
                        // the gate now rejects new orders on the instrument.
                        if !outcome.fee_breaches.is_empty() {
                            shared
                                .errors
                                .push(format!("fee breaches: {:?}", outcome.fee_breaches));
                        }
                    }
                    Err(error) => shared.errors.push(format!("fill: {error}")),
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.account.reset();
    }

    fn description(&self) -> String {
        format!(
            "margin-rs risk gate for {}",
            self.account.lock().instrument.as_str()
        )
    }
}

/// [`FillAccountant`] reading a shared [`MarginAccount`]'s ledger.
///
/// The gate books fills when [`MarginRiskPolicy`] forwards them, before the
/// engine posts to the accountant; posting checks that the execution was
/// booked and fails otherwise (for example when the gate refused it).
#[derive(Debug, Clone)]
pub struct MarginGateAccountant {
    account: MarginAccount,
}

impl FillAccountant for MarginGateAccountant {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        let shared = self.account.lock_checked()?;
        let id = ExecutionId::new(format!("sim-{}", fill.execution_id))
            .map_err(|e| MMError::InvalidPositionUpdate(format!("execution id: {e}")))?;
        if shared.gate.ledger().execution_cashflow(&id).is_none() {
            return Err(MMError::InvalidPositionUpdate(format!(
                "execution {} was not booked by the margin gate",
                id.as_str()
            )));
        }
        Ok(())
    }

    fn snapshot(&self) -> MMResult<AccountingSnapshot> {
        let shared = self.account.lock();
        let ledger = shared.gate.ledger();
        let position = ledger.position(&shared.instrument);
        // The account is dedicated to one instrument, so the fees paid in
        // its base asset are this instrument's base fees (O(1), no journal
        // scan). They are deducted from the position: the account holds
        // the traded quantity minus the base it paid as fees.
        let base_fees = ledger.fees_paid(&shared.base);
        let quote_fees = ledger.fees_paid(&shared.quote);
        let total = ledger
            .balance(&shared.quote)
            .map_or(Decimal::ZERO, |b| b.total);
        let deposits = ledger
            .transfers()
            .filter(|t| t.asset == shared.quote)
            .try_fold(Decimal::ZERO, |acc, t| match t.kind {
                TransferKind::Deposit => acc.checked_add(t.amount),
                TransferKind::Withdrawal => acc.checked_sub(t.amount),
            })
            .or_overflow("margin account deposits")?;
        Ok(AccountingSnapshot {
            position: position
                .as_ref()
                .map_or(Decimal::ZERO, |p| p.quantity)
                .checked_sub(base_fees)
                .or_overflow("margin account position")?,
            average_entry_price: position.as_ref().and_then(|p| p.average_price()),
            cash_flow: total
                .checked_sub(deposits)
                .and_then(|x| x.checked_add(quote_fees))
                .or_overflow("margin account cash flow")?,
            fees: quote_fees,
            base_fees,
            fills_posted: crate::types::decimal::saturating_u64(ledger.journal().len()),
        })
    }

    fn reset(&mut self) {
        self.account.reset();
    }

    fn description(&self) -> String {
        format!(
            "margin-rs gate ledger for {}",
            self.account.lock().instrument.as_str()
        )
    }
}
