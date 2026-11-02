//! `margin-rs` risk gate behind the pretrade and accounting ports.
//!
//! [`MarginAccount`] shares one `margin_rs::risk::RiskGate` (which owns the
//! account's ledger) between a [`PretradePolicy`] and a [`FillAccountant`]:
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

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use margin_rs::ledger::{Execution, Side as LedgerSide, TransferKind};
use margin_rs::risk::{
    FillReport, OrderPurpose as GateOrderPurpose, OrderRequest, RiskDecision, RiskGate,
};
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId, OrderId};

use super::accounting::{AccountingFill, AccountingSnapshot, FillAccountant};
use crate::Decimal;
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

#[derive(Debug)]
struct Shared {
    initial: RiskGate,
    gate: RiskGate,
    instrument: InstrumentId,
    quote: Asset,
    errors: Vec<String>,
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
        let quote = match gate.ledger().instrument(&instrument) {
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
            shared: Arc::new(Mutex::new(Shared {
                initial: gate.clone(),
                gate,
                instrument,
                quote,
                errors: Vec::new(),
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

    /// Locks the shared account; a poisoned lock is recovered, since every
    /// gate operation leaves the gate consistent on error.
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn reset(&self) {
        let mut shared = self.lock();
        shared.gate = shared.initial.clone();
        shared.errors.clear();
    }
}

fn order_id(raw: &str) -> margin_rs::Result<OrderId> {
    OrderId::new(format!("sim-order-{raw}"))
}

fn ledger_side(dir: Dir) -> LedgerSide {
    match dir {
        Dir::Buy => LedgerSide::Buy,
        Dir::Sell => LedgerSide::Sell,
    }
}

/// [`PretradePolicy`] over a shared [`MarginAccount`].
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
        let mut shared = self.account.lock();
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
            ledger_side(dir),
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
        let mut shared = self.account.lock();
        let instrument = shared.instrument.clone();
        let result = shared.gate.on_market_data(&instrument, mark, timestamp);
        Self::record(&mut shared, "market data", result);
    }

    fn on_order_event(&mut self, event: &OrderEvent) {
        let mut shared = self.account.lock();
        let raw = match event {
            OrderEvent::Accepted { order_id }
            | OrderEvent::VenueRejected { order_id }
            | OrderEvent::CancelRequested { order_id }
            | OrderEvent::Canceled { order_id }
            | OrderEvent::CancelRejected { order_id }
            | OrderEvent::Fill { order_id, .. } => order_id,
        };
        let Ok(id) = order_id(raw) else {
            return;
        };
        // Orders the gate never saw (rejected locally) are ignored.
        if shared.gate.order(&id).is_none() {
            return;
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
                ..
            } => {
                let dir = match Dir::try_from(*side) {
                    Ok(dir) => dir,
                    Err(error) => {
                        shared.errors.push(format!("fill: {error}"));
                        return;
                    }
                };
                let execution_id = match ExecutionId::new(execution_id) {
                    Ok(id) => id,
                    Err(error) => {
                        shared.errors.push(format!("execution id: {error}"));
                        return;
                    }
                };
                let mut execution = Execution::new(
                    execution_id,
                    shared.instrument.clone(),
                    ledger_side(dir),
                    *price,
                    *quantity,
                );
                // Negative fees are rebates and are booked as such.
                if !fee.is_zero() {
                    execution = execution.with_fee(shared.quote.clone(), *fee);
                }
                let report = FillReport::new(execution, *cumulative_quantity);
                let result = shared.gate.on_fill(&id, &report);
                match result {
                    Ok(outcome) if !outcome.is_clean() => shared
                        .errors
                        .push(format!("fill anomalies: {:?}", outcome.anomalies)),
                    Ok(_) => {}
                    Err(error) => shared.errors.push(format!("fill: {error}")),
                }
            }
        }
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
        let shared = self.account.lock();
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

    fn snapshot(&self) -> AccountingSnapshot {
        let shared = self.account.lock();
        let ledger = shared.gate.ledger();
        let position = ledger.position(&shared.instrument);
        let fees = ledger.fees_paid(&shared.quote);
        let total = ledger
            .balance(&shared.quote)
            .map_or(Decimal::ZERO, |b| b.total);
        let deposits: Decimal = ledger
            .transfers()
            .filter(|t| t.asset == shared.quote)
            .map(|t| match t.kind {
                TransferKind::Deposit => t.amount,
                TransferKind::Withdrawal => -t.amount,
            })
            .sum();
        AccountingSnapshot {
            position: position.as_ref().map_or(Decimal::ZERO, |p| p.quantity),
            average_entry_price: position.as_ref().and_then(|p| p.average_price()),
            cash_flow: total - deposits + fees,
            fees,
            fills_posted: ledger.journal().len() as u64,
        }
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
