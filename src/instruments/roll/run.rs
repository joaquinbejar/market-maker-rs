//! A [`RollPlan`] run through `backtest::contracts::ContractEngine`.
//!
//! [`RollStrategy`] carries out what the plan asks and tells it what happened. On its
//! first tick at or after the plan's creation it sends attempt 0 of both legs, at the
//! plan's limits, unless the deadline has passed by then, when no order is sent and the
//! plan hears that they were cancelled with nothing filled. The engine wakes it at the
//! deadline, so a live order is cancelled then, before a later tick could fill it.
//! Every fill, rejection, cancellation and settlement of the near contract goes to the
//! plan. It sends nothing the plan did not ask for, and it does not quote.
//!
//! Only a plan that has seen no event yet, for the positions the ledger holds at the start
//! on both legs, is run: anything else is a fault and no order, as selling what is not held
//! would flip a position. This is judged once, on the first event it sees: a settlement can
//! reach it before its first tick, when the position it closed still counts as held, and if
//! that is of the near contract the plan hears that its attempts were cancelled, unfilled,
//! and nothing is sent.
//!
//! When the data ends nothing more can be sent, so a plan may end the run still working,
//! which [`RollStrategy::plan`] shows.
//!
//! The plan's clock is the engine's: the time an event reached the strategy, which a fill's
//! or a report's own time may be behind. In a run the ledger is ahead of the plan until a
//! fill's report arrives, so `RollPlan::reconcile` reads clean only once the run ended.
//!
//! # A reduced order
//!
//! A policy may approve less than a leg asked for. The order is then smaller than the
//! attempt the plan believes is live, so when it ends filled, the plan hears that the
//! attempt was cancelled at what it filled, and treats the rest by its policy: it resends
//! it or gives up on it, and never waits for it. An unwind reduced is a fault.
//!
//! # With a guard
//!
//! [`SharedRollGuard`] is the [`RollGuard`] the engine's policy and this strategy share. It
//! syncs after every event the plan takes, before the plan's commands go out, so an order
//! is judged on the state that asked for it. A poisoned lock fails closed: the guard
//! rejects every order and the run shows a fault, until a reset. Fills that already
//! executed stay in the ledger.
//!
//! Attempt 0 goes out on both legs at once. With `max_exposure` equal to the exposure at
//! the start the guard therefore rejects the far leg, as nothing has closed yet, and the
//! plan treats that by its policy. For both legs to work together, `max_exposure` needs
//! room for both.
//!
//! # Assumption of an unwind
//!
//! The plan does not price an unwind. This strategy crosses the latest book it has
//! observed of that leg, at the ask to buy and the bid to sell. That book must have size on
//! that side for the whole of it and be no older than `max_book_age_ms` at the decision.
//! Another leg's book, or one not observed yet, is never used, and with none nothing is
//! invented or rounded: it is a fault and no order. An unwind rejected, as for a price off
//! the grid, or cancelled with quantity left, is a fault too.
//!
//! # Faults
//!
//! What the plan refuses, what its intents do not take, a guard that cannot sync and a
//! cancel for an attempt that has no order are faults, through
//! `ContractStrategy::take_fault`. The engine then refuses new orders, and the run is not
//! consistent.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use financial_types::Action;
use financial_types::instrument::InstrumentId;
use rust_decimal::Decimal;

use super::{RollAction, RollEvent, RollFill, RollGuard, RollLeg, RollPlan};
use crate::backtest::contracts::{
    ContractCommand, ContractFill, ContractOrder, ContractReport, ContractStrategy, ContractView,
};
use crate::backtest::{ExecutionKind, MarketTick};
use crate::instruments::futures::SettlementPrice;
use crate::risk::pretrade::{
    OrderEvent, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
};
use crate::types::error::{MMError, MMResult};

/// A [`RollGuard`] shared by the engine's policy and a [`RollStrategy`]. Cloning shares
/// the same guard.
#[derive(Debug, Clone)]
pub struct SharedRollGuard {
    guard: Arc<Mutex<RollGuard>>,
}

impl SharedRollGuard {
    /// Shares `guard`.
    #[must_use]
    pub fn new(guard: RollGuard) -> Self {
        Self {
            guard: Arc::new(Mutex::new(guard)),
        }
    }

    fn poisoned() -> MMError {
        MMError::LockPoisoned("roll guard".to_string())
    }

    fn lock(&self) -> MMResult<MutexGuard<'_, RollGuard>> {
        self.guard.lock().map_err(|_| Self::poisoned())
    }

    /// Runs `f` with the guard.
    ///
    /// # Errors
    ///
    /// [`MMError::LockPoisoned`] if a thread panicked while holding it.
    pub fn with_guard<R>(&self, f: impl FnOnce(&RollGuard) -> R) -> MMResult<R> {
        Ok(f(&*self.lock()?))
    }
}

impl PretradePolicy for SharedRollGuard {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        match self.lock() {
            Ok(mut guard) => guard.check(request),
            Err(error) => PretradeDecision::Rejected {
                reason: format!("policy fault: {error}"),
            },
        }
    }

    fn on_market_data(&mut self, mark: Decimal, timestamp: u64) {
        // A poisoned guard stays so until a reset, and rejects every order.
        if let Ok(mut guard) = self.lock() {
            guard.on_market_data(mark, timestamp);
        }
    }

    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        self.lock()?.on_order_event(event)
    }

    fn reset(&mut self) {
        let mut guard = self
            .guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.reset();
        // Cleared while held, so no other thread sees the state restored and poisoned.
        self.guard.clear_poison();
    }

    fn description(&self) -> String {
        "shared roll guard".to_string()
    }

    fn in_contracts(&self) -> bool {
        true
    }
}

/// What an order of this strategy is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Owned {
    leg: RollLeg,
    /// `None` for an unwind.
    attempt: Option<u32>,
}

/// Runs a [`RollPlan`]: see the [module documentation](self).
#[derive(Debug, Clone)]
pub struct RollStrategy {
    initial: RollPlan,
    plan: RollPlan,
    guard: Option<SharedRollGuard>,
    max_book_age_ms: u64,
    started: bool,
    /// Whether the plan can run, once judged.
    fit_to_run: Option<bool>,
    now_ms: u64,
    next_tag: u64,
    owned: HashMap<u64, Owned>,
    tags: HashMap<(RollLeg, u32), u64>,
    /// What each order asked for.
    asked: HashMap<u64, Decimal>,
    /// The orders a policy reduced, and what it approved.
    reduced: HashMap<u64, Decimal>,
    near_book: Option<MarketTick>,
    far_book: Option<MarketTick>,
    faults: VecDeque<MMError>,
}

impl RollStrategy {
    /// A strategy for `plan`, whose unwinds accept a book up to `max_book_age_ms` old.
    #[must_use]
    pub fn new(plan: RollPlan, max_book_age_ms: u64) -> Self {
        Self {
            initial: plan.clone(),
            plan,
            guard: None,
            max_book_age_ms,
            started: false,
            fit_to_run: None,
            now_ms: 0,
            next_tag: 0,
            owned: HashMap::new(),
            tags: HashMap::new(),
            asked: HashMap::new(),
            reduced: HashMap::new(),
            near_book: None,
            far_book: None,
            faults: VecDeque::new(),
        }
    }

    /// Keeps `guard` in sync with the plan. Give the engine the same guard as its policy.
    #[must_use]
    pub fn with_guard(mut self, guard: SharedRollGuard) -> Self {
        self.guard = Some(guard);
        self
    }

    /// The plan, as far as this run took it.
    #[must_use]
    pub fn plan(&self) -> &RollPlan {
        &self.plan
    }

    fn leg_of(&self, contract: &InstrumentId) -> Option<RollLeg> {
        if self.plan.near.terms.spec().id() == contract {
            Some(RollLeg::Near)
        } else if self.plan.far.terms.spec().id() == contract {
            Some(RollLeg::Far)
        } else {
            None
        }
    }

    fn fault(&mut self, reason: String) {
        self.faults
            .push_back(MMError::InvalidMarketState(format!("roll: {reason}")));
    }

    fn see_time(&mut self, at_ms: u64) {
        self.now_ms = self.now_ms.max(at_ms);
    }

    /// A new order: its tag is what its reports will carry.
    fn order(
        &mut self,
        owned: Owned,
        side: Action,
        price: Decimal,
        quantity: Decimal,
    ) -> ContractCommand {
        let tag = self.next_tag;
        self.next_tag = self.next_tag.saturating_add(1);
        self.owned.insert(tag, owned);
        self.asked.insert(tag, quantity);
        if let Some(attempt) = owned.attempt {
            self.tags.insert((owned.leg, attempt), tag);
        }
        let contract = self.plan.leg(owned.leg).terms.spec().id().clone();
        ContractCommand::Submit(ContractOrder {
            tag,
            contract,
            side,
            price,
            quantity,
            purpose: OrderPurpose::Other,
        })
    }

    fn attempt_zero(&mut self, leg: RollLeg) -> ContractCommand {
        let (side, price, contracts) = {
            let terms = self.plan.leg(leg);
            (terms.side, terms.price, terms.contracts)
        };
        self.order(
            Owned {
                leg,
                attempt: Some(0),
            },
            side,
            price,
            contracts,
        )
    }

    /// Whether this plan can run on the positions at the start, of which a settlement before
    /// this first event may have closed `closed` on a leg.
    fn judge(&mut self, view: &ContractView, closed: Option<(RollLeg, Decimal)>) -> bool {
        if let Some(fit) = self.fit_to_run {
            return fit;
        }
        let fit = self.fit_now(view, closed);
        self.fit_to_run = Some(fit);
        fit
    }

    fn fit_now(&mut self, view: &ContractView, closed: Option<(RollLeg, Decimal)>) -> bool {
        if !self.initial.history.is_empty() {
            self.fault("the plan has seen events already, so its attempts are not new".to_string());
            return false;
        }
        for leg in [RollLeg::Near, RollLeg::Far] {
            let terms = self.plan.leg(leg);
            let (planned, mut held) = (terms.start, view.position(terms.terms.spec().id()));
            if let Some((_, contracts)) = closed.filter(|(on, _)| *on == leg) {
                held = held.saturating_add(contracts);
            }
            if planned != held {
                self.fault(format!(
                    "the plan starts at {planned} on the {leg:?} leg, and the ledger holds {held}"
                ));
                return false;
            }
        }
        true
    }

    /// The first orders, if the plan can still send them.
    fn start(&mut self, view: &ContractView) -> Vec<ContractCommand> {
        self.started = true;
        if !self.judge(view, None) {
            return Vec::new();
        }
        if self.now_ms >= self.plan.deadline_ms {
            self.never_sent(&RollEvent::Tick { at_ms: self.now_ms });
            return Vec::new();
        }
        vec![
            self.attempt_zero(RollLeg::Near),
            self.attempt_zero(RollLeg::Far),
        ]
    }

    /// `first` came before anything was sent, so the plan asks to cancel what it believes is
    /// live, and it is told that was done, with nothing filled.
    fn never_sent(&mut self, first: &RollEvent) {
        let mut asked = match self.plan.apply(first) {
            Ok(report) => report.actions,
            Err(error) => {
                self.fault(format!("the plan refused an event: {error}"));
                return;
            }
        };
        while let Some(action) = asked.pop() {
            let RollAction::Cancel { leg, attempt } = action else {
                self.fault(format!("the plan asked for {action:?} with nothing sent"));
                continue;
            };
            match self.plan.apply(&RollEvent::CancelAck {
                leg,
                attempt,
                cum_contracts: Decimal::ZERO,
                at_ms: self.now_ms,
            }) {
                Ok(report) => asked.extend(report.actions),
                Err(error) => self.fault(format!("the plan refused an event: {error}")),
            }
        }
        self.sync_guard();
    }

    fn sync_guard(&mut self) {
        if let Some(guard) = &self.guard {
            let synced = guard.lock().and_then(|mut held| {
                held.sync(&self.plan)
                    .map_err(|error| MMError::InvalidMarketState(error.to_string()))
            });
            if let Err(error) = synced {
                self.faults.push_back(error);
            }
        }
    }

    /// The price an unwind crosses at: see the [module documentation](self).
    fn unwind_price(
        &self,
        leg: RollLeg,
        side: Action,
        contracts: Decimal,
    ) -> Result<Decimal, String> {
        let book = match leg {
            RollLeg::Near => self.near_book.as_ref(),
            RollLeg::Far => self.far_book.as_ref(),
        };
        let Some(book) = book else {
            return Err(format!("no book of the {leg:?} leg to unwind against"));
        };
        let age = self.now_ms.saturating_sub(book.timestamp);
        if age > self.max_book_age_ms {
            return Err(format!(
                "the book of the {leg:?} leg is {age} ms old, past {}",
                self.max_book_age_ms
            ));
        }
        let (price, size) = match side {
            Action::Buy => (book.ask_price, book.ask_size),
            Action::Sell => (book.bid_price, book.bid_size),
            other => return Err(format!("an unwind on {other}")),
        };
        if price <= Decimal::ZERO {
            return Err(format!(
                "the book of the {leg:?} leg has no price to {side}"
            ));
        }
        if size < contracts {
            return Err(format!(
                "the book of the {leg:?} leg has {size} to {side}, short of {contracts}"
            ));
        }
        Ok(price)
    }

    fn command_for(&mut self, action: RollAction) -> Option<ContractCommand> {
        match action {
            RollAction::Cancel { leg, attempt } => match self.tags.get(&(leg, attempt)) {
                Some(tag) => Some(ContractCommand::Cancel { tag: *tag }),
                None => {
                    self.fault(format!(
                        "asked to cancel attempt {attempt} of the {leg:?} leg, which has no order"
                    ));
                    None
                }
            },
            RollAction::Resubmit {
                leg,
                attempt,
                contracts,
                price,
            } => {
                let side = self.plan.leg(leg).side;
                let owned = Owned {
                    leg,
                    attempt: Some(attempt),
                };
                Some(self.order(owned, side, price, contracts))
            }
            RollAction::Unwind {
                leg,
                side,
                contracts,
            } => match self.unwind_price(leg, side, contracts) {
                Ok(price) => {
                    let owned = Owned { leg, attempt: None };
                    Some(self.order(owned, side, price, contracts))
                }
                Err(reason) => {
                    self.fault(reason);
                    None
                }
            },
        }
    }

    /// Gives the plan one event, syncs the guard, and returns what the plan asks.
    fn apply(&mut self, event: &RollEvent) -> Vec<ContractCommand> {
        let report = match self.plan.apply(event) {
            Ok(report) => report,
            Err(error) => {
                self.fault(format!("the plan refused an event: {error}"));
                return Vec::new();
            }
        };
        if let Some(refused) = &report.refused {
            self.fault(format!("the plan's intents did not take a fill: {refused}"));
        }
        self.sync_guard();
        report
            .actions
            .into_iter()
            .filter_map(|action| self.command_for(action))
            .collect()
    }
}

impl ContractStrategy for RollStrategy {
    fn on_tick(
        &mut self,
        contract: &InstrumentId,
        tick: &MarketTick,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        let Some(leg) = self.leg_of(contract) else {
            return Vec::new();
        };
        self.see_time(view.now_ms());
        let book = match leg {
            RollLeg::Near => &mut self.near_book,
            RollLeg::Far => &mut self.far_book,
        };
        // An out of order tick is not the latest book.
        if book.as_ref().is_none_or(|b| b.timestamp <= tick.timestamp) {
            *book = Some(tick.clone());
        }
        if self.now_ms < self.plan.created_at_ms {
            return Vec::new();
        }
        if self.started {
            return Vec::new();
        }
        self.start(view)
    }

    fn on_report(&mut self, report: &ContractReport, view: &ContractView) -> Vec<ContractCommand> {
        let Some(owned) = self.owned.get(&report.tag).copied() else {
            self.fault(format!(
                "a report of order tag {}, not of this roll",
                report.tag
            ));
            return Vec::new();
        };
        self.see_time(view.now_ms());
        let at_ms = self.now_ms;
        let remaining = report.report.remaining;
        match report.report.kind {
            ExecutionKind::Rejected => {
                if let Some(attempt) = owned.attempt {
                    return self.apply(&RollEvent::Rejected {
                        leg: owned.leg,
                        attempt,
                        at_ms,
                    });
                }
                self.fault(format!("an unwind of the {:?} leg was rejected", owned.leg));
            }
            ExecutionKind::Canceled => {
                if let Some(attempt) = owned.attempt {
                    return self.apply(&RollEvent::CancelAck {
                        leg: owned.leg,
                        attempt,
                        cum_contracts: report.report.cumulative_quantity,
                        at_ms,
                    });
                }
                if remaining > Decimal::ZERO {
                    self.fault(format!(
                        "an unwind of the {:?} leg was cancelled with {remaining} left",
                        owned.leg
                    ));
                }
            }
            ExecutionKind::Accepted => {
                let reduced = self
                    .asked
                    .get(&report.tag)
                    .is_some_and(|asked| remaining < *asked);
                if reduced && owned.attempt.is_some() {
                    self.reduced.insert(report.tag, remaining);
                } else if reduced {
                    self.fault(format!(
                        "an unwind of the {:?} leg was reduced to {remaining}",
                        owned.leg
                    ));
                }
            }
            // A fill reaches the plan in its own callback, and a cancel that came too
            // late changes nothing it does not hear from that fill.
            ExecutionKind::PartialFill | ExecutionKind::Fill | ExecutionKind::CancelRejected => {}
        }
        Vec::new()
    }

    fn on_fill(&mut self, fill: &ContractFill, view: &ContractView) -> Vec<ContractCommand> {
        let Some(owned) = self.owned.get(&fill.tag).copied() else {
            self.fault(format!(
                "a fill of order tag {}, not of this roll",
                fill.tag
            ));
            return Vec::new();
        };
        self.see_time(view.now_ms());
        let fee_asset = self
            .plan
            .leg(owned.leg)
            .terms
            .spec()
            .settlement_asset()
            .clone();
        let mut commands = self.apply(&RollEvent::Fill(RollFill {
            leg: owned.leg,
            attempt: owned.attempt,
            exec_id: fill.execution_id.clone(),
            side: fill.side,
            contracts: fill.quantity,
            cum_contracts: fill.cumulative_quantity,
            price: fill.price,
            fee: fill.fee,
            fee_asset,
            at_ms: fill.timestamp,
        }));
        // The end of an order smaller than its attempt, which nothing else would tell.
        if let Some(attempt) = owned.attempt
            && self.reduced.get(&fill.tag) == Some(&fill.cumulative_quantity)
        {
            commands.extend(self.apply(&RollEvent::CancelAck {
                leg: owned.leg,
                attempt,
                cum_contracts: fill.cumulative_quantity,
                at_ms: self.now_ms,
            }));
        }
        commands
    }

    fn on_settlement(
        &mut self,
        contract: &InstrumentId,
        price: &SettlementPrice,
        contracts: Decimal,
        view: &ContractView,
    ) -> Vec<ContractCommand> {
        self.see_time(view.now_ms());
        let Some(leg) = self.leg_of(contract) else {
            return Vec::new();
        };
        if !self.started && !self.judge(view, Some((leg, contracts))) {
            return Vec::new();
        }
        if leg == RollLeg::Far {
            return Vec::new();
        }
        let event = RollEvent::NearSettled {
            price: price.clone(),
            contracts,
        };
        if !self.started {
            self.started = true;
            self.never_sent(&event);
            return Vec::new();
        }
        if self.fit_to_run == Some(false) {
            return Vec::new();
        }
        self.apply(&event)
    }

    fn next_wake_ms(&self) -> Option<u64> {
        // The engine wakes it once, and ignores this once the time has passed. A plan not
        // fit to run has nothing live to end.
        (self.started && self.fit_to_run == Some(true)).then_some(self.plan.deadline_ms)
    }

    fn on_wake(&mut self, view: &ContractView) -> Vec<ContractCommand> {
        self.see_time(view.now_ms());
        self.apply(&RollEvent::Tick { at_ms: self.now_ms })
    }

    fn take_fault(&mut self) -> Option<MMError> {
        self.faults.pop_front()
    }

    fn reset(&mut self) {
        self.plan = self.initial.clone();
        self.started = false;
        self.fit_to_run = None;
        self.now_ms = 0;
        self.next_tag = 0;
        self.owned.clear();
        self.tags.clear();
        self.asked.clear();
        self.reduced.clear();
        self.near_book = None;
        self.far_book = None;
        self.faults.clear();
    }
}
