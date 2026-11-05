//! What may be quoted and reserved while a [`RollPlan`] runs.
//!
//! [`RollGuard`] is a [`PretradePolicy`] for the window of a roll, for the live path
//! ([`OrderManager::register_order_checked`](crate::execution::OrderManager::register_order_checked))
//! and for direct callers. `BacktestEngine::with_instrument` refuses a pretrade policy,
//! and `backtest::contracts::ContractEngine` takes this one, through `run::SharedRollGuard`.
//!
//! # The rules
//!
//! Every order goes through the guard, the plan's own legs included: a fill of an order
//! it never saw cannot be attributed, as `OrderEvent` names no instrument, so it is
//! ignored until [`RollGuard::sync`], and quotes could stack on top of it meanwhile.
//!
//! - A request that names no instrument or another one, on another action, or for a
//!   quantity that is not positive, is rejected, as is a repeated order identifier.
//! - From the plan's deadline both legs are rejected, judged on the latest time seen,
//!   in requests, market data, fills or any event the plan saw, so a stale request cannot
//!   reopen the window.
//!   Handing over to a normal policy is the caller's act, after retiring what is live.
//! - An order that reduces a position may do so down to zero, less what already
//!   reduces it, and never flips it. That holds past any limit: the guard must not trap
//!   the risk it limits.
//! - The near contract closes only, except while the plan unwinds, when it may add
//!   back towards its start, as an unwind needs.
//! - The far contract may add only while the plan works, within its start and target.
//! - On both legs together, the exposure if every live order of the order's direction
//!   fills, with this one, may not pass `max_exposure`. Orders of the other direction
//!   are not netted, as they may not fill. So with `max_exposure` equal to the exposure
//!   at the start, the far contract can only follow what the near one has filled.
//!
//! A larger order is reduced to the room, on the lot; no room is a rejection.
//!
//! # Positions
//!
//! A position is the plan's at the last [`RollGuard::sync`], plus the fills the guard
//! saw that the plan had not, by leg and execution identifier, so no order of events
//! counts one twice or loses one. A sync also carries the plan's state and a settlement
//! with its effect, once, as they are in the plan's positions, and the latest time it
//! saw. A copy of the plan that does not descend from what an earlier sync saw is
//! refused, however many events it has, as is a fill with other content.
//!
//! Sync before deciding after a fill the guard could not see or a change in the plan's
//! state. A settlement is published at or after the near contract's expiry, so a sync
//! with one closes the window: nothing new is decided on a position after it. A near
//! fill only the guard has seen may by then have closed at the settlement price, which
//! only the plan's record can tell, so until the plan sees it [`RollGuard::position`] is
//! provisional.
//!
//! A sync clears neither live orders nor a fault of the guard. Only
//! [`PretradePolicy::reset`], which returns to the guard's construction, does.
//!
//! # Reserves
//!
//! `RollReserves` (feature `margin`) holds both legs' margin in a `margin-rs` ledger, with no offset
//! between them. Only a ledger's own positions are supported, as it is the truth for a
//! settled position, which a service with positions of its own could not drop.
//!
//! `Ledger::settle_expiry` moves cash and closes the position, and leaves reservations
//! alone. So after it, a hold stands and a live near order still counts, until
//! `RollReserves::near_settled` releases those orders and syncs the hold. Marks for
//! margin come from the caller, and never settle anything.

use std::collections::{BTreeMap, HashMap};

use arbitrage_rs::Quantity;
use financial_types::Action;
use rust_decimal::Decimal;

use super::{
    LegTerms, RollError, RollEvent, RollFailurePolicy, RollLeg, RollPlan, RollState, exact_add,
    exact_mul, exact_sub, upstream,
};
use crate::instruments::futures::FutureTerms;
use crate::risk::pretrade::{OrderEvent, PretradeDecision, PretradePolicy, PretradeRequest};
use crate::types::error::{MMError, MMResult};

/// What the guard knows of a leg, fixed at construction.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GuardLeg {
    terms: FutureTerms,
    size: Decimal,
    lot: Decimal,
    /// The lowest position that an order adding may reach.
    low: Decimal,
    /// The highest position that an order adding may reach.
    high: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GuardOrder {
    leg: RollLeg,
    direction: Decimal,
    approved: Decimal,
    filled: Decimal,
    open: bool,
}

/// What a plan was made from, which its events do not change.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanOrigin {
    near: LegTerms,
    far: LegTerms,
    policy: RollFailurePolicy,
    created_at_ms: u64,
    deadline_ms: u64,
}

impl PlanOrigin {
    fn of(plan: &RollPlan) -> Self {
        Self {
            near: plan.near.clone(),
            far: plan.far.clone(),
            policy: plan.policy,
            created_at_ms: plan.created_at_ms,
            deadline_ms: plan.deadline_ms,
        }
    }
}

/// What a sync, or the construction, sets.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanView {
    near_position: Decimal,
    far_position: Decimal,
    fills: BTreeMap<(bool, String), Decimal>,
    state: RollState,
    near_settled: bool,
    latest_ms: u64,
    history: Vec<RollEvent>,
}

fn key(leg: RollLeg, exec_id: &str) -> (bool, String) {
    (leg == RollLeg::Near, exec_id.to_string())
}

impl PlanView {
    fn of(plan: &RollPlan) -> Self {
        let signed = |fill: &super::RollFill| match fill.side {
            Action::Buy => fill.contracts,
            _ => arbitrage_rs::exact::neg(fill.contracts),
        };
        Self {
            near_position: plan.position(RollLeg::Near),
            far_position: plan.position(RollLeg::Far),
            fills: plan
                .fills()
                .iter()
                .map(|f| (key(f.leg, &f.exec_id), signed(f)))
                .collect(),
            state: plan.state(),
            near_settled: plan.near_settled().is_some(),
            latest_ms: plan.observed_ms,
            history: plan.history.clone(),
        }
    }
}

/// The pretrade policy of a roll's window: see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct RollGuard {
    near: GuardLeg,
    far: GuardLeg,
    deadline_ms: u64,
    max_exposure: Decimal,
    origin: PlanOrigin,
    initial: PlanView,
    view: PlanView,
    orders: HashMap<String, GuardOrder>,
    fills: Vec<(RollLeg, String, Decimal)>,
    latest_ms: u64,
    fault: Option<MMError>,
}

fn direction_of(side: Action) -> Option<Decimal> {
    match side {
        Action::Buy => Some(Decimal::ONE),
        Action::Sell => Some(Decimal::NEGATIVE_ONE),
        _ => None,
    }
}

/// `value` down to a multiple of `step`, or zero if it is not positive.
fn floored(value: Decimal, step: Decimal) -> Result<Decimal, RollError> {
    if value <= Decimal::ZERO {
        return Ok(Decimal::ZERO);
    }
    let value = Quantity::new(value).map_err(upstream)?;
    let step = Quantity::new(step).map_err(upstream)?;
    Ok(value
        .floor_to_lot(step)
        .map_err(upstream)?
        .map_or(Decimal::ZERO, |q| q.value()))
}

impl RollGuard {
    /// A guard for `plan`, at its positions and state now.
    ///
    /// # Errors
    ///
    /// [`RollError::InvalidExposureLimit`] if `max_exposure` is below the plan's exposure
    /// now, and [`RollError::Plan`] if a figure overflows.
    pub fn new(plan: &RollPlan, max_exposure: Decimal) -> Result<Self, RollError> {
        let exposure = plan.exposure()?.abs();
        if max_exposure < exposure {
            return Err(RollError::InvalidExposureLimit {
                max_exposure,
                exposure,
            });
        }
        let far_signed = if plan.side(RollLeg::Far) == Action::Buy {
            plan.contracts(RollLeg::Far)
        } else {
            arbitrage_rs::exact::neg(plan.contracts(RollLeg::Far))
        };
        let far_target = exact_add(plan.far.start, far_signed)?;
        let view = PlanView::of(plan);
        Ok(Self {
            near: GuardLeg {
                terms: plan.near.terms.clone(),
                size: plan.near.size,
                lot: plan.near.terms.spec().lot(),
                low: plan.near.start.min(Decimal::ZERO),
                high: plan.near.start.max(Decimal::ZERO),
            },
            far: GuardLeg {
                terms: plan.far.terms.clone(),
                size: plan.far.size,
                lot: plan.far.terms.spec().lot(),
                low: plan.far.start.min(far_target),
                high: plan.far.start.max(far_target),
            },
            deadline_ms: plan.deadline_ms,
            max_exposure,
            origin: PlanOrigin::of(plan),
            latest_ms: view.latest_ms,
            initial: view.clone(),
            view,
            orders: HashMap::new(),
            fills: Vec::new(),
            fault: None,
        })
    }

    /// Takes the plan's positions, state and settlement: see the
    /// [module documentation](self).
    ///
    /// # Errors
    ///
    /// [`RollError::OtherPlan`] for another plan, by anything in its request;
    /// [`RollError::StalePlan`] for one whose events do not continue those an
    /// earlier sync saw; and [`RollError::ConflictingFill`] for a fill the plan and
    /// the guard saw with other content. The guard is then unchanged.
    pub fn sync(&mut self, plan: &RollPlan) -> Result<(), RollError> {
        if PlanOrigin::of(plan) != self.origin {
            return Err(RollError::OtherPlan);
        }
        let view = PlanView::of(plan);
        // Only what descends from what was seen: a copy from before a rejection, however
        // many events it has since, could otherwise start far opening again.
        if !view.history.starts_with(&self.view.history) {
            return Err(RollError::StalePlan);
        }
        for (leg, exec_id, signed) in &self.fills {
            if view
                .fills
                .get(&key(*leg, exec_id))
                .is_some_and(|seen| seen != signed)
            {
                return Err(RollError::ConflictingFill {
                    exec_id: exec_id.clone(),
                });
            }
        }
        self.latest_ms = self.latest_ms.max(view.latest_ms);
        self.view = view;
        Ok(())
    }

    /// The latest time seen, which decides the window.
    #[must_use]
    pub fn latest_ms(&self) -> u64 {
        self.latest_ms
    }

    fn leg(&self, leg: RollLeg) -> &GuardLeg {
        match leg {
            RollLeg::Near => &self.near,
            RollLeg::Far => &self.far,
        }
    }

    /// The position on a leg now: see the [module documentation](self).
    ///
    /// # Errors
    ///
    /// [`RollError::Plan`] if it overflows.
    pub fn position(&self, leg: RollLeg) -> Result<Decimal, RollError> {
        let mut position = match leg {
            RollLeg::Near => self.view.near_position,
            RollLeg::Far => self.view.far_position,
        };
        for (fill_leg, exec_id, signed) in &self.fills {
            if *fill_leg == leg && !self.view.fills.contains_key(&key(leg, exec_id)) {
                position = exact_add(position, *signed)?;
            }
        }
        Ok(position)
    }

    /// The contracts of a leg's live orders in a direction.
    fn live(&self, leg: RollLeg, direction: Decimal) -> Result<Decimal, RollError> {
        let mut live = Decimal::ZERO;
        for order in self.orders.values() {
            if order.open && order.leg == leg && order.direction == direction {
                let left = exact_sub(order.approved, order.filled)?;
                live = exact_add(live, left.max(Decimal::ZERO))?;
            }
        }
        Ok(live)
    }

    /// The room on a leg by its own rules, in contracts, not on the lot.
    fn leg_room(&self, leg: RollLeg, direction: Decimal) -> Result<Decimal, RollError> {
        let position = self.position(leg)?;
        let live = self.live(leg, direction)?;
        let reduces =
            if !position.is_zero() && (position > Decimal::ZERO) != (direction > Decimal::ZERO) {
                exact_sub(position.abs(), live)?
            } else {
                Decimal::ZERO
            };
        let adds = match leg {
            RollLeg::Near => matches!(
                self.view.state,
                RollState::Unwinding | RollState::UnwindRequested
            ),
            RollLeg::Far => self.view.state == RollState::Working,
        };
        if !adds {
            return Ok(reduces);
        }
        let terms = self.leg(leg);
        let worst = exact_add(position, exact_mul(direction, live)?)?;
        let band = if direction > Decimal::ZERO {
            exact_sub(terms.high, worst)?
        } else {
            exact_sub(worst, terms.low)?
        };
        Ok(band.max(reduces))
    }

    /// The room in the exposure if everything live in this direction fills.
    fn exposure_room(&self, direction: Decimal) -> Result<Decimal, RollError> {
        let worst_of = |leg: RollLeg| -> Result<Decimal, RollError> {
            let live = exact_mul(direction, self.live(leg, direction)?)?;
            exact_mul(exact_add(self.position(leg)?, live)?, self.leg(leg).size)
        };
        let worst = exact_add(worst_of(RollLeg::Near)?, worst_of(RollLeg::Far)?)?;
        exact_sub(self.max_exposure, exact_mul(direction, worst)?)
    }

    fn decide(
        &self,
        request: &PretradeRequest,
        direction: Decimal,
        leg: RollLeg,
    ) -> Result<PretradeDecision, RollError> {
        let terms = self.leg(leg);
        let wanted = request.quantity;
        let leg_room = self.leg_room(leg, direction)?;
        let exposure_room = self.exposure_room(direction)?;
        if wanted <= leg_room && exact_mul(wanted, terms.size)? <= exposure_room {
            return Ok(PretradeDecision::Accepted);
        }
        // The exposure's room on the grid of this leg, in its contracts.
        let step = exact_mul(terms.lot, terms.size)?;
        let by_exposure = floored(exposure_room, step)?
            .checked_div(terms.size)
            .unwrap_or(Decimal::ZERO);
        let room = floored(leg_room, terms.lot)?.min(by_exposure.normalize());
        if room <= Decimal::ZERO {
            return Ok(PretradeDecision::Rejected {
                reason: format!(
                    "no room on {}: {leg_room} on the leg, {exposure_room} of exposure",
                    terms.terms.spec().id().as_str()
                ),
            });
        }
        Ok(PretradeDecision::Reduced {
            quantity: room,
            reason: format!(
                "reduced on {}: {leg_room} on the leg, {exposure_room} of exposure",
                terms.terms.spec().id().as_str()
            ),
        })
    }
}

impl PretradePolicy for RollGuard {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        self.latest_ms = self.latest_ms.max(request.timestamp);
        let reject = |reason: String| PretradeDecision::Rejected { reason };
        if let Some(fault) = &self.fault {
            return reject(format!("the guard faulted: {fault}"));
        }
        let Some(direction) = direction_of(request.side) else {
            return reject(format!("unsupported side {}", request.side));
        };
        if request.quantity <= Decimal::ZERO {
            return reject(format!("quantity {} is not positive", request.quantity));
        }
        if self.orders.contains_key(&request.order_id) {
            return reject(format!("order {} was already seen", request.order_id));
        }
        let leg = match request.instrument.as_deref() {
            Some(id) if id == self.near.terms.spec().id().as_str() => RollLeg::Near,
            Some(id) if id == self.far.terms.spec().id().as_str() => RollLeg::Far,
            other => return reject(format!("{other:?} is not a contract of this roll")),
        };
        if self.latest_ms >= self.deadline_ms {
            return reject(format!(
                "the roll's window ended at {}, and it is {}",
                self.deadline_ms, self.latest_ms
            ));
        }
        let decision = match self.decide(request, direction, leg) {
            Ok(decision) => decision,
            Err(error) => return reject(error.to_string()),
        };
        let approved = decision.approved_quantity(request.quantity);
        if approved > Decimal::ZERO {
            self.orders.insert(
                request.order_id.clone(),
                GuardOrder {
                    leg,
                    direction,
                    approved,
                    filled: Decimal::ZERO,
                    open: true,
                },
            );
        }
        decision
    }

    fn in_contracts(&self) -> bool {
        true
    }

    fn on_market_data(&mut self, _mark: Decimal, timestamp: u64) {
        self.latest_ms = self.latest_ms.max(timestamp);
    }

    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        match event {
            OrderEvent::VenueRejected { order_id } | OrderEvent::Canceled { order_id } => {
                // The order keeps its leg: a fill may still arrive.
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.open = false;
                }
            }
            OrderEvent::Fill {
                order_id,
                execution_id,
                side,
                quantity,
                timestamp,
                ..
            } => {
                self.latest_ms = self.latest_ms.max(*timestamp);
                let Some(direction) = direction_of(*side) else {
                    let error = MMError::UnsupportedAction { action: *side };
                    self.fault = Some(error.clone());
                    return Err(error);
                };
                let Some(order) = self.orders.get_mut(order_id) else {
                    return Ok(());
                };
                let leg = order.leg;
                if *quantity <= Decimal::ZERO {
                    let error = MMError::InvalidPositionUpdate(format!(
                        "execution {execution_id} has quantity {quantity}"
                    ));
                    self.fault = Some(error.clone());
                    return Err(error);
                }
                // A real execution stays counted, on the side it was on, and the guard
                // decides nothing more until a reset, as its order did not say that.
                let wrong_side = order.direction != direction;
                let figures = exact_add(order.filled, *quantity)
                    .and_then(|filled| Ok((filled, exact_mul(direction, *quantity)?)));
                let (filled, signed) = match figures {
                    Ok(figures) => figures,
                    Err(error) => {
                        let error = MMError::NumericalError(error.to_string());
                        self.fault = Some(error.clone());
                        return Err(error);
                    }
                };
                // By leg and execution, so that a trade reassigned to another order
                // does not count twice. The same execution with other content cannot be
                // accounted for.
                let seen = self
                    .fills
                    .iter()
                    .find(|(fill_leg, id, _)| *fill_leg == leg && id == execution_id);
                if let Some((_, _, before)) = seen {
                    if *before == signed {
                        return Ok(());
                    }
                    let error = MMError::InvalidPositionUpdate(format!(
                        "execution {execution_id} was observed with other content"
                    ));
                    self.fault = Some(error.clone());
                    return Err(error);
                }
                order.filled = filled;
                self.fills.push((leg, execution_id.clone(), signed));
                if wrong_side {
                    let error = MMError::InvalidPositionUpdate(format!(
                        "execution {execution_id} is on the other side of its order"
                    ));
                    self.fault = Some(error.clone());
                    return Err(error);
                }
            }
            OrderEvent::Accepted { .. }
            | OrderEvent::CancelRequested { .. }
            | OrderEvent::CancelRejected { .. } => {}
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.view = self.initial.clone();
        self.orders.clear();
        self.fills.clear();
        self.latest_ms = self.initial.latest_ms;
        self.fault = None;
    }

    fn description(&self) -> String {
        format!(
            "roll guard from {} to {}",
            self.near.terms.spec().id().as_str(),
            self.far.terms.spec().id().as_str()
        )
    }
}

#[cfg(feature = "margin")]
pub use reserves::RollReserves;

#[cfg(feature = "margin")]
mod reserves {
    use std::collections::{BTreeMap, BTreeSet};

    use margin_rs::initial_margin::{InitialMarginModel, ProductKind};
    use margin_rs::instrument::ContractSize;
    use margin_rs::margin_reservations::{
        BookedFill, MarginDecision, MarginInstrument, MarginOrder, MarginReservationService,
    };
    use margin_rs::risk::RiskGate;
    use margin_rs::{
        AccountId, Asset, ExecutionId, InstrumentId, MarginError, OrderId, ReservationId, VenueId,
    };
    use rust_decimal::Decimal;

    use super::super::{RollError, RollLeg};
    use crate::instruments::futures::{FutureError, FutureTerms};
    use crate::instruments::{ContractKind, InstrumentSpec, Margining};

    fn margin(error: MarginError) -> RollError {
        RollError::Future(FutureError::Ledger(error))
    }

    fn registered(spec: &InstrumentSpec) -> Result<MarginInstrument, RollError> {
        let amount = spec.size().amount();
        let (product, size) = match (spec.kind(), spec.margining()) {
            (ContractKind::Future, Margining::Linear) => {
                (ProductKind::LinearFuture, ContractSize::Multiplier(amount))
            }
            _ => (
                ProductKind::InverseFuture,
                ContractSize::ContractValue(amount),
            ),
        };
        Ok(MarginInstrument {
            product,
            settlement: Asset::new(spec.settlement_asset().as_str()).map_err(margin)?,
            size,
        })
    }

    /// Both legs' margin in a ledger: see the [module documentation](crate::instruments::roll::guard).
    #[derive(Debug, Clone)]
    pub struct RollReserves<M> {
        service: MarginReservationService<M>,
        hold: ReservationId,
        near: FutureTerms,
        far: FutureTerms,
        near_orders: BTreeSet<OrderId>,
    }

    impl<M: InitialMarginModel + Clone> RollReserves<M> {
        /// Reserves for a roll from `near` to `far`, held as `hold`.
        ///
        /// # Errors
        ///
        /// [`RollError::Future`] if a contract cannot be registered.
        pub fn new(
            model: M,
            account: AccountId,
            venue: VenueId,
            hold: ReservationId,
            near: &FutureTerms,
            far: &FutureTerms,
        ) -> Result<Self, RollError> {
            let mut service =
                MarginReservationService::on_ledger(model, account, venue, hold.clone());
            for terms in [near, far] {
                service
                    .register(
                        InstrumentId::new(terms.spec().id().as_str()).map_err(margin)?,
                        registered(terms.spec())?,
                    )
                    .map_err(margin)?;
            }
            Ok(Self {
                service,
                hold,
                near: near.clone(),
                far: far.clone(),
                near_orders: BTreeSet::new(),
            })
        }

        fn terms(&self, leg: RollLeg) -> &FutureTerms {
            match leg {
                RollLeg::Near => &self.near,
                RollLeg::Far => &self.far,
            }
        }

        /// Reserves an order on a leg, or says why not in the decision.
        ///
        /// # Errors
        ///
        /// [`RollError::NearAlreadySettled`] on a settled near contract;
        /// [`RollError::Future`] for other terms in the ledger, or what the service
        /// reports.
        pub fn reserve(
            &mut self,
            leg: RollLeg,
            id: OrderId,
            side: financial_types::Action,
            contracts: Decimal,
            gate: &mut RiskGate,
            marks: &BTreeMap<InstrumentId, Decimal>,
        ) -> Result<MarginDecision, RollError> {
            let instrument = self.terms(leg).identify_in_ledger(gate.ledger())?;
            if leg == RollLeg::Near && gate.ledger().is_settled(&instrument) {
                return Err(RollError::NearAlreadySettled);
            }
            let decision = self
                .service
                .reserve_with_gate(
                    MarginOrder {
                        id: id.clone(),
                        instrument,
                        side,
                        contracts,
                    },
                    gate,
                    &self.hold,
                    marks,
                )
                .map_err(margin)?;
            if leg == RollLeg::Near && matches!(decision, MarginDecision::Reserved { .. }) {
                self.near_orders.insert(id);
            }
            Ok(decision)
        }

        /// Reports a fill already booked in the gate's ledger.
        ///
        /// # Errors
        ///
        /// [`RollError::Future`] with what the service reports.
        pub fn apply_booked_fill(
            &mut self,
            order: &OrderId,
            execution: &ExecutionId,
            gate: &mut RiskGate,
            marks: &BTreeMap<InstrumentId, Decimal>,
        ) -> Result<BookedFill, RollError> {
            self.service
                .apply_booked_fill(gate, order, execution, marks)
                .map_err(margin)
        }

        /// Releases what is left of an order once its cancel is effective, and returns
        /// the contracts released. The hold moves at the next [`RollReserves::sync_hold`].
        pub fn release_on_cancel(&mut self, order: &OrderId) -> Decimal {
            self.near_orders.remove(order);
            self.service.release_on_effective_cancel(order)
        }

        /// Sets the hold to the requirement now, and returns it.
        ///
        /// # Errors
        ///
        /// [`RollError::Future`] with what the service reports, with the hold unchanged,
        /// such as a requirement that a mark took beyond the cash.
        pub fn sync_hold(
            &self,
            gate: &mut RiskGate,
            marks: &BTreeMap<InstrumentId, Decimal>,
        ) -> Result<Decimal, RollError> {
            self.service
                .sync_hold(gate, &self.hold, marks)
                .map_err(margin)
        }

        /// After the near contract settled in the gate's ledger, releases its live orders
        /// and syncs the hold, as one step: on an error nothing changed, and it can be
        /// tried again. Repeating it only syncs.
        ///
        /// # Errors
        ///
        /// [`RollError::NearNotSettled`] if the ledger holds no such settlement, and
        /// [`RollError::Future`] for other terms in the ledger or a failed sync.
        pub fn near_settled(
            &mut self,
            gate: &mut RiskGate,
            marks: &BTreeMap<InstrumentId, Decimal>,
        ) -> Result<Decimal, RollError> {
            let instrument = self.near.identify_in_ledger(gate.ledger())?;
            if !gate.ledger().is_settled(&instrument) {
                return Err(RollError::NearNotSettled);
            }
            let mut service = self.service.clone();
            for order in &self.near_orders {
                service.release_on_effective_cancel(order);
            }
            let hold = service.sync_hold(gate, &self.hold, marks).map_err(margin)?;
            self.service = service;
            self.near_orders.clear();
            Ok(hold)
        }
    }
}
