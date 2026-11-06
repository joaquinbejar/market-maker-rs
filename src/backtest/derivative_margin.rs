//! Initial margin for a future or a perpetual, behind the pretrade and accounting ports.
//!
//! [`DerivativeMarginAccount`] shares one `margin_rs::risk::RiskGate`, which owns the ledger,
//! and one `MarginReservationService` between a pretrade policy and an accountant, so that the
//! requirement, the fills and the cash have a single authority:
//!
//! - [`DerivativeMarginAccount::policy`] reserves each order on the requirement it would add,
//!   in contracts and on the contract's notional at the last mark, and keeps that requirement
//!   as one hold in the ledger. It books each fill, which stands whatever the margin. A
//!   requested cancel keeps its margin, as the order can still fill; an effective cancel or a
//!   venue rejection releases it.
//! - [`DerivativeMarginAccount::accountant`] reads position, realized PnL and fees back from
//!   that ledger.
//!
//! It takes every mark as its own contract's, so it runs in `BacktestEngine::with_instrument`
//! and `ContractEngine`, whose marks are of several contracts, refuses it.
//!
//! The policy needs `BacktestEngine::with_instrument` and this account's accountant: it books
//! in contracts, which the engine's own bookkeeping does not.
//!
//! The service accepts or refuses an order, so a proposal beyond the margin is refused whole,
//! never cut. A requirement that a mark move takes beyond the cash keeps the previous hold and
//! is counted in [`DerivativeMarginAccount::hold_shortfalls`]. Nothing liquidates: funding,
//! expiry and liquidation are outside this account.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use margin_rs::initial_margin::{ExposurePosition, ExposureView, InitialMarginModel, ProductKind};
use margin_rs::ledger::{Booking, Execution};
use margin_rs::margin_reservations::{
    BookedFill, MarginDecision, MarginInstrument, MarginOrder, MarginReservationService,
};
use margin_rs::risk::RiskGate;
use margin_rs::{
    AccountId, Asset, ContractSize, ExecutionId, Instrument, InstrumentId, OrderId, ReservationId,
    VenueId,
};

use super::accounting::{AccountingFill, AccountingSnapshot, FeeAsset, FillAccountant};
use crate::Decimal;
use crate::risk::pretrade::{OrderEvent, PretradeDecision, PretradePolicy, PretradeRequest};
use crate::types::action::{Action, Dir};
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};

#[derive(Debug, Clone)]
struct State<M> {
    gate: RiskGate,
    service: MarginReservationService<M>,
    marks: BTreeMap<InstrumentId, Decimal>,
    fees: Decimal,
    fills: u64,
    errors: Vec<String>,
    hold_shortfalls: u64,
    fault: Option<MMError>,
}

#[derive(Debug)]
struct Shared<M> {
    initial: State<M>,
    state: State<M>,
    instrument: InstrumentId,
    settlement: Asset,
    hold: ReservationId,
}

/// One account in a future or a perpetual, shared by a pretrade policy and an accountant.
/// Cloning shares the same account.
pub struct DerivativeMarginAccount<M> {
    shared: Arc<Mutex<Shared<M>>>,
}

impl<M> Clone for DerivativeMarginAccount<M> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<M> fmt::Debug for DerivativeMarginAccount<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shared = self.lock();
        f.debug_struct("DerivativeMarginAccount")
            .field("instrument", &shared.instrument.as_str())
            .field("errors", &shared.state.errors.len())
            .finish()
    }
}

/// The terms the service registers, from what the ledger holds.
fn terms_of(instrument: &Instrument) -> Option<MarginInstrument> {
    let product = match instrument {
        Instrument::LinearFuture(_) => ProductKind::LinearFuture,
        Instrument::LinearPerpetual(_) => ProductKind::LinearPerpetual,
        Instrument::InverseFuture(_) => ProductKind::InverseFuture,
        Instrument::InversePerpetual(_) => ProductKind::InversePerpetual,
        _ => return None,
    };
    if let Some(terms) = instrument.linear_terms() {
        return Some(MarginInstrument {
            product,
            settlement: terms.settlement().clone(),
            size: ContractSize::Multiplier(terms.multiplier()),
        });
    }
    let terms = instrument.inverse_terms()?;
    Some(MarginInstrument {
        product,
        settlement: terms.settlement().clone(),
        size: ContractSize::ContractValue(terms.contract_value()),
    })
}

fn config_error(context: &str, error: margin_rs::MarginError) -> MMError {
    MMError::InvalidConfiguration(format!("derivative margin {context}: {error}"))
}

fn order_id(raw: &str) -> margin_rs::Result<OrderId> {
    OrderId::new(format!("sim-order-{raw}"))
}

impl<M: InitialMarginModel + Clone> DerivativeMarginAccount<M> {
    /// Wraps a configured gate, with `instrument` registered and its deposits made, and no
    /// execution booked yet. Its state now is what a reset restores.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `instrument` is not a future or a
    /// perpetual registered in the gate's ledger, if the ledger already holds an execution,
    /// whose id a run could repeat, or if the service or the model refuses its terms, such
    /// as a model in another currency.
    pub fn new(
        model: M,
        gate: RiskGate,
        instrument: InstrumentId,
        account: AccountId,
        venue: VenueId,
        hold: ReservationId,
    ) -> MMResult<Self> {
        let refuse = |reason: &str| {
            Err(MMError::InvalidConfiguration(format!(
                "instrument {}: {reason}",
                instrument.as_str()
            )))
        };
        let Some(registered) = gate.ledger().instrument(&instrument) else {
            return refuse("not registered in the ledger");
        };
        let Some(terms) = terms_of(registered) else {
            return refuse("not a future or a perpetual");
        };
        if !gate.ledger().journal().is_empty() {
            return refuse("the ledger already holds executions");
        }
        let settlement = terms.settlement.clone();
        model
            .requirement(&ExposureView {
                account: account.clone(),
                venue: venue.clone(),
                positions: vec![ExposurePosition {
                    instrument: instrument.clone(),
                    product: terms.product,
                    settlement: settlement.clone(),
                    contracts: Decimal::ONE,
                    size: terms.size,
                    mark: Decimal::ONE,
                }],
            })
            .map_err(|e| config_error("model", e))?;
        let mut service = MarginReservationService::on_ledger(model, account, venue, hold.clone());
        service
            .register(instrument.clone(), terms)
            .map_err(|e| config_error("terms", e))?;
        let state = State {
            gate,
            service,
            marks: BTreeMap::new(),
            fees: Decimal::ZERO,
            fills: 0,
            errors: Vec::new(),
            hold_shortfalls: 0,
            fault: None,
        };
        Ok(Self {
            shared: Arc::new(Mutex::new(Shared {
                initial: state.clone(),
                state,
                instrument,
                settlement,
                hold,
            })),
        })
    }

    /// Pretrade policy of this account.
    #[must_use]
    pub fn policy(&self) -> DerivativeMarginPolicy<M> {
        DerivativeMarginPolicy {
            account: self.clone(),
        }
    }

    /// Accountant reading this account's ledger.
    #[must_use]
    pub fn accountant(&self) -> DerivativeMarginAccountant<M> {
        DerivativeMarginAccountant {
            account: self.clone(),
        }
    }

    /// Reads the gate and its ledger.
    pub fn with_gate<R>(&self, f: impl FnOnce(&RiskGate) -> R) -> R {
        f(&self.lock().state.gate)
    }

    /// Errors on lifecycle events, in order.
    #[must_use]
    pub fn errors(&self) -> Vec<String> {
        self.lock().state.errors.clone()
    }

    /// The fault that stops new orders, if any.
    #[must_use]
    pub fn fault(&self) -> Option<MMError> {
        self.lock().state.fault.clone()
    }

    /// How many times a requirement could not be held, the previous hold being kept.
    #[must_use]
    pub fn hold_shortfalls(&self) -> u64 {
        self.lock().state.hold_shortfalls
    }
}

impl<M> DerivativeMarginAccount<M> {
    fn poisoned() -> MMError {
        MMError::InvalidPositionUpdate("derivative margin account lock poisoned".to_string())
    }

    fn lock_checked(&self) -> MMResult<MutexGuard<'_, Shared<M>>> {
        self.shared.lock().map_err(|_| Self::poisoned())
    }

    /// A read only view: decisions fail closed through [`Self::lock_checked`].
    fn lock(&self) -> MutexGuard<'_, Shared<M>> {
        self.shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl<M: Clone> DerivativeMarginAccount<M> {
    fn reset(&self) {
        let mut shared = self.lock();
        shared.state = shared.initial.clone();
        self.shared.clear_poison();
    }
}

/// [`PretradePolicy`] over a shared [`DerivativeMarginAccount`].
#[derive(Debug, Clone)]
pub struct DerivativeMarginPolicy<M> {
    account: DerivativeMarginAccount<M>,
}

impl<M: InitialMarginModel + Clone> Shared<M> {
    /// Sets the hold to the requirement now.
    fn sync_hold(&mut self) {
        let state = &mut self.state;
        match state
            .service
            .sync_hold(&mut state.gate, &self.hold, &state.marks)
        {
            Ok(_) => {}
            Err(margin_rs::MarginError::InsufficientBalance { .. }) => {
                state.hold_shortfalls = state.hold_shortfalls.saturating_add(1);
            }
            // Anything else is not about the cash, and new orders must not go on.
            Err(error) => {
                let error = MMError::InvalidPositionUpdate(format!("derivative hold: {error}"));
                state.errors.push(error.to_string());
                state.fault = Some(error);
            }
        }
    }

    fn release(&mut self, raw: &str) -> MMResult<()> {
        let id = order_id(raw)
            .map_err(|e| MMError::InvalidPositionUpdate(format!("order id {raw}: {e}")))?;
        self.state.service.release_on_effective_cancel(&id);
        self.sync_hold();
        Ok(())
    }

    /// Books a fill, mandatory, then moves the order and the hold.
    fn book(&mut self, event: &FillEvent<'_>) -> MMResult<()> {
        if !event.fee.is_zero() && event.fee_asset != &FeeAsset::Settlement {
            return Err(MMError::InvalidPositionUpdate(format!(
                "a derivative fee must be in the settlement asset, got {:?}",
                event.fee_asset
            )));
        }
        let side = Action::from(Dir::try_from(event.side)?);
        let fail = |context: &str, error: margin_rs::MarginError| {
            MMError::InvalidPositionUpdate(format!("derivative margin {context}: {error}"))
        };
        let id = ExecutionId::new(event.execution_id).map_err(|e| fail("execution id", e))?;
        let order = order_id(event.order_id).map_err(|e| fail("order id", e))?;
        let mut execution = Execution::new(
            id.clone(),
            self.instrument.clone(),
            side,
            event.price,
            event.quantity,
        )
        .with_executed_at(event.timestamp);
        if !event.fee.is_zero() {
            execution = execution.with_fee(self.settlement.clone(), event.fee);
        }
        let state = &mut self.state;
        let booked = state
            .gate
            .book_derivative_execution(&execution)
            .map_err(|e| fail("booking", e))?;
        if matches!(booked.booking, Booking::Booked(_)) {
            state.fees = state.fees.checked_add(event.fee).or_overflow("fees")?;
            state.fills = state.fills.saturating_add(1);
        }
        let applied = state
            .service
            .apply_booked_fill(&mut state.gate, &order, &id, &state.marks)
            .map_err(|e| fail("fill", e))?;
        if let BookedFill::Applied {
            overfill: Some(overfill),
            ..
        } = applied
        {
            state.errors.push(format!(
                "fill {} went {overfill} past order {}",
                id.as_str(),
                order.as_str()
            ));
        }
        Ok(())
    }
}

/// The parts of [`OrderEvent::Fill`] that are booked.
struct FillEvent<'a> {
    order_id: &'a str,
    execution_id: &'a str,
    side: Action,
    price: Decimal,
    quantity: Decimal,
    fee: Decimal,
    fee_asset: &'a FeeAsset,
    timestamp: u64,
}

impl<M: InitialMarginModel + Clone + fmt::Debug> PretradePolicy for DerivativeMarginPolicy<M> {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        let reject = |reason: String| PretradeDecision::Rejected { reason };
        let dir = match Dir::try_from(request.side) {
            Ok(dir) => dir,
            Err(error) => return reject(error.to_string()),
        };
        let mut guard = match self.account.lock_checked() {
            Ok(guard) => guard,
            Err(error) => return reject(format!("policy fault: {error}")),
        };
        let shared = &mut *guard;
        if let Some(fault) = &shared.state.fault {
            return reject(format!("policy fault: {fault}"));
        }
        if let Some(asked) = &request.instrument
            && asked != shared.instrument.as_str()
        {
            return reject(format!(
                "this account trades {}, not {asked}",
                shared.instrument.as_str()
            ));
        }
        let id = match order_id(&request.order_id) {
            Ok(id) => id,
            Err(error) => return reject(format!("invalid order id: {error}")),
        };
        let order = MarginOrder {
            id,
            instrument: shared.instrument.clone(),
            side: Action::from(dir),
            contracts: request.quantity,
        };
        let state = &mut shared.state;
        match state
            .service
            .reserve_with_gate(order, &mut state.gate, &shared.hold, &state.marks)
        {
            Ok(MarginDecision::Reserved { .. }) => PretradeDecision::Accepted,
            Ok(MarginDecision::Rejected { reason }) => reject(format!("{reason:?}")),
            Err(error) => reject(format!("margin service: {error}")),
        }
    }

    fn on_market_data(&mut self, mark: Decimal, _timestamp: u64) {
        let Ok(mut shared) = self.account.lock_checked() else {
            return;
        };
        let instrument = shared.instrument.clone();
        shared.state.marks.insert(instrument, mark);
        shared.sync_hold();
    }

    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        let mut shared = self.account.lock_checked()?;
        let result = match event {
            // The order can still fill, so it keeps its margin.
            OrderEvent::Accepted { .. }
            | OrderEvent::CancelRequested { .. }
            | OrderEvent::CancelRejected { .. } => Ok(()),
            OrderEvent::VenueRejected { order_id } | OrderEvent::Canceled { order_id } => {
                shared.release(order_id)
            }
            OrderEvent::Fill {
                order_id,
                execution_id,
                side,
                price,
                quantity,
                fee,
                fee_asset,
                timestamp,
                ..
            } => shared.book(&FillEvent {
                order_id,
                execution_id,
                side: *side,
                price: *price,
                quantity: *quantity,
                fee: *fee,
                fee_asset,
                timestamp: *timestamp,
            }),
        };
        if let Err(error) = &result {
            shared.state.errors.push(error.to_string());
            shared.state.fault = Some(error.clone());
        }
        result
    }

    fn reset(&mut self) {
        self.account.reset();
    }

    fn description(&self) -> String {
        format!(
            "margin-rs initial margin for {}",
            self.account.lock().instrument.as_str()
        )
    }

    fn in_one_contract(&self) -> bool {
        true
    }
}

/// [`FillAccountant`] reading a shared [`DerivativeMarginAccount`]'s ledger. The policy books
/// each fill before the engine posts it; posting checks that it did.
#[derive(Debug, Clone)]
pub struct DerivativeMarginAccountant<M> {
    account: DerivativeMarginAccount<M>,
}

impl<M: InitialMarginModel + Clone + fmt::Debug> FillAccountant for DerivativeMarginAccountant<M> {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        let shared = self.account.lock_checked()?;
        let id = ExecutionId::new(format!("sim-{}", fill.execution_id))
            .map_err(|e| MMError::InvalidPositionUpdate(format!("execution id: {e}")))?;
        if shared.state.gate.ledger().execution_cashflow(&id).is_none() {
            return Err(MMError::InvalidPositionUpdate(format!(
                "execution {} was not booked by the derivative margin policy",
                id.as_str()
            )));
        }
        Ok(())
    }

    fn snapshot(&self) -> MMResult<AccountingSnapshot> {
        let shared = self.account.lock_checked()?;
        let position = shared.state.gate.ledger().position(&shared.instrument);
        Ok(AccountingSnapshot {
            position: position.as_ref().map_or(Decimal::ZERO, |p| p.quantity),
            average_entry_price: position.as_ref().and_then(|p| p.average_price()),
            cash_flow: position.as_ref().map_or(Decimal::ZERO, |p| p.realized_pnl),
            fees: shared.state.fees,
            fills_posted: shared.state.fills,
            ..AccountingSnapshot::default()
        })
    }

    fn reset(&mut self) {
        self.account.reset();
    }

    fn description(&self) -> String {
        format!(
            "margin-rs derivative ledger for {}",
            self.account.lock().instrument.as_str()
        )
    }

    fn unrealized_at(&self, mark: Decimal) -> Option<MMResult<Decimal>> {
        let shared = match self.account.lock_checked() {
            Ok(shared) => shared,
            Err(error) => return Some(Err(error)),
        };
        let marks = BTreeMap::from([(shared.instrument.clone(), mark)]);
        Some(
            shared
                .state
                .gate
                .ledger()
                .valuation(&marks)
                .map_err(|e| MMError::InvalidPositionUpdate(format!("valuation: {e}")))
                .map(|valuation| {
                    valuation
                        .positions
                        .get(&shared.instrument)
                        .map_or(Decimal::ZERO, |position| position.unrealized_pnl)
                }),
        )
    }

    fn requires_instrument(&self) -> bool {
        true
    }

    fn check_instrument(
        &self,
        spec: &financial_types::instrument::InstrumentSpec,
        fee_scale: u32,
    ) -> MMResult<()> {
        let shared = self.account.lock_checked()?;
        let refuse = |reason: String| {
            Err(MMError::InvalidConfiguration(format!(
                "ledger instrument {} and specification {}: {reason}",
                shared.instrument.as_str(),
                spec.id().as_str()
            )))
        };
        let ledger = shared.state.gate.ledger();
        let Some(registered) = ledger.instrument(&shared.instrument) else {
            return refuse("not registered".to_string());
        };
        let Some(scales) = crate::instruments::margin::registered_scales(registered) else {
            return refuse("not a derivative".to_string());
        };
        let expected = Instrument::from_spec(spec, scales)
            .map_err(|e| MMError::InvalidConfiguration(format!("specification: {e}")))?;
        if expected != *registered {
            return refuse(format!("the terms differ: {registered:?} and {expected:?}"));
        }
        let scale = ledger.snapshot().assets.get(&shared.settlement).copied();
        if scale.is_none_or(|scale| fee_scale > scale) {
            return refuse(format!(
                "fee scale {fee_scale} above the settlement asset's {scale:?}"
            ));
        }
        Ok(())
    }
}
