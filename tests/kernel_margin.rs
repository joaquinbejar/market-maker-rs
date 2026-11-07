//! Margin-rs ledger reservations beside the kernel: a ledger driven by the
//! agent's own actions and the kernel's events, through the replay, snapshot
//! and batch gate.
//!
//! These reservations shadow the kernel and do not admit or refuse orders; a
//! reservation the deposits cannot cover fails the test. The kernel's own
//! balance gate, which does refuse them (`KernelMargin`), is compared with the
//! engine's `MarginAccount` in `kernel_margin_gate.rs`.
//!
//! An order reserves when it is submitted and the delivered reports consume the
//! reservation, so it stays pending across every latency. A buy reserves
//! `remaining * price` of quote, a sell `remaining` of base. Selling short draws
//! on the base deposit, so the position is the base balance minus the deposit.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

/// Deliveries capacity of the harness kernels: `(3 + R) * ORD` for `R` up to 13.
const HARNESS_DLV: usize = 1024;

#[path = "support/kernel_harness.rs"]
mod harness;

#[path = "support/kernel_gate.rs"]
mod gate;

use gate::*;
use harness::*;
use margin_rs::Decimal;
use margin_rs::ledger::{Execution, ExecutionCashflow, Ledger, LedgerConfig, LedgerSnapshot};
use margin_rs::{Asset, ExecutionId, InstrumentId, ReservationId, SpotInstrument, TransferId};
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    ExecKind, KernelAction, KernelConfig, KernelEvent, KernelFillModel, KernelLatency, Observation,
    Origin, QuoteSide, ReportRec, Status, TermReason, decode, encode, fingerprint,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;
use std::collections::BTreeMap;

/// Deposited in each asset; far above what the scenarios trade.
const DEPOSIT: i64 = 1_000_000_000;

/// An order with an open reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenOrder {
    side: Action,
    price: Decimal,
    remaining: Decimal,
}

/// A deliberate bug, for the negative controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    SkipRelease,
    SkipBooking,
    ReleaseOnAccepted,
}

/// A margin-rs ledger and the orders it holds reservations for.
#[derive(Debug, Clone)]
struct MarginShadow {
    ledger: Ledger,
    instrument: InstrumentId,
    base: Asset,
    quote: Asset,
    tick: Decimal,
    lot: Decimal,
    open: BTreeMap<u64, OpenOrder>,
    fault: Option<Fault>,
}

/// The ledger as one value: compared by equality, not through a hash of its
/// `Debug` output.
type LedgerImage = (LedgerSnapshot, Vec<ExecutionCashflow>);

fn scale_of(value: Decimal) -> u32 {
    value.normalize().scale()
}

fn reservation_id(order_id: u64) -> ReservationId {
    ReservationId::new(format!("ord-{order_id}")).unwrap()
}

/// The legs an action submits, in id order: `(side, ticks, lots)`.
fn legs(action: &KernelAction) -> Vec<(Action, u64, u64)> {
    match *action {
        KernelAction::Quote {
            bid,
            ask,
            bid_lots,
            ask_lots,
        } => vec![(Action::Buy, bid, bid_lots), (Action::Sell, ask, ask_lots)],
        KernelAction::OneSided { side, price, lots } => {
            let side = match side {
                QuoteSide::Bid => Action::Buy,
                QuoteSide::Ask => Action::Sell,
            };
            vec![(side, price, lots)]
        }
        _ => Vec::new(),
    }
}

impl MarginShadow {
    fn new(cfg: &KernelConfig) -> Self {
        let (tick, lot) = (cfg.tick_size, cfg.lot_size);
        let (tick_scale, lot_scale) = (scale_of(tick), scale_of(lot));
        // The kernel's money scale, so every notional and fee fits exactly.
        let money_scale = tick_scale + lot_scale + scale_of(cfg.fee_rate);
        let base = Asset::new("BASE").unwrap();
        let quote = Asset::new("QUOTE").unwrap();
        let instrument = InstrumentId::new("BASE-QUOTE").unwrap();
        let mut ledger = Ledger::new(LedgerConfig::default());
        ledger.register_asset(base.clone(), lot_scale).unwrap();
        ledger.register_asset(quote.clone(), money_scale).unwrap();
        ledger
            .register_instrument(
                SpotInstrument::new(
                    instrument.clone(),
                    base.clone(),
                    quote.clone(),
                    tick_scale,
                    lot_scale,
                )
                .unwrap(),
            )
            .unwrap();
        for (id, asset) in [("base", &base), ("quote", &quote)] {
            ledger
                .deposit(
                    TransferId::new(id).unwrap(),
                    asset.clone(),
                    Decimal::from(DEPOSIT),
                )
                .unwrap();
        }
        Self {
            ledger,
            instrument,
            base,
            quote,
            tick,
            lot,
            open: BTreeMap::new(),
            fault: None,
        }
    }

    fn with_fault(mut self, fault: Fault) -> Self {
        self.fault = Some(fault);
        self
    }

    /// The asset and amount an order reserves.
    fn reserved(&self, order: &OpenOrder) -> (Asset, Decimal) {
        match order.side {
            Action::Buy => (
                self.quote.clone(),
                (order.price * order.remaining).normalize(),
            ),
            _ => (self.base.clone(), order.remaining.normalize()),
        }
    }

    fn lots(&self, lots: u64) -> Decimal {
        (Decimal::from(lots) * self.lot).normalize()
    }

    fn price(&self, ticks: u64) -> Decimal {
        (Decimal::from(ticks) * self.tick).normalize()
    }

    /// Reserves for the orders `action` submitted with ids from `first_id` to
    /// before `next_id`, using the step's events only for what the action
    /// itself decides: a local reject, and a size the risk gate reduced.
    fn submit(
        &mut self,
        action: &KernelAction,
        first_id: u64,
        next_id: u64,
        events: &[KernelEvent],
    ) -> Result<(), String> {
        for (offset, (side, price, lots)) in (0u64..).zip(legs(action)) {
            let id = first_id + offset;
            if id >= next_id {
                break;
            }
            let locally_rejected = events.iter().any(|e| {
                matches!(e, KernelEvent::Report(r)
                    if r.order_id == id && r.kind == ExecKind::Rejected && r.origin == Origin::Local)
            });
            if locally_rejected {
                continue;
            }
            let gated = events.iter().find_map(|e| match e {
                KernelEvent::Gate(g) if g.order_id == id => Some(g.submitted),
                _ => None,
            });
            let order = OpenOrder {
                side,
                price: self.price(price),
                remaining: self.lots(gated.unwrap_or(lots)),
            };
            let (asset, amount) = self.reserved(&order);
            let opened = self
                .ledger
                .reserve(reservation_id(id), asset, amount)
                .map_err(|e| format!("reserve order {id}: {e}"))?;
            if !opened {
                return Err(format!("order {id} reserved twice"));
            }
            self.open.insert(id, order);
        }
        Ok(())
    }

    /// The venue accepted an order: what was reserved at submission must be it.
    fn check_accepted(&self, r: &ReportRec) -> Result<(), String> {
        let expected = OpenOrder {
            side: r.side,
            price: self.price(r.price),
            remaining: self.lots(r.qty),
        };
        match self.open.get(&r.order_id) {
            Some(order) if *order == expected => Ok(()),
            other => Err(format!(
                "order {} accepted as {expected:?}, reserved as {other:?}",
                r.order_id
            )),
        }
    }

    fn book_fill(&mut self, r: &ReportRec, fee: Decimal) -> Result<(), String> {
        if self.fault == Some(Fault::SkipBooking) {
            self.fault = None;
            return Ok(());
        }
        // The order and the cumulative quantity name a fill. A repeat changes
        // nothing, even after later fills or once the order has closed.
        let id = ExecutionId::new(format!("fill-{}-{}", r.order_id, r.cum)).unwrap();
        if let Some(booked) = self
            .ledger
            .journal()
            .iter()
            .find(|entry| entry.execution_id == id)
        {
            return if booked.side == r.side && booked.quantity == self.lots(r.last) {
                Ok(())
            } else {
                Err(format!("fill {id:?} repeated with other terms"))
            };
        }
        let Some(order) = self.open.get(&r.order_id).copied() else {
            return Err(format!(
                "fill of order {} without a reservation",
                r.order_id
            ));
        };
        let next = OpenOrder {
            remaining: self.lots(r.qty - r.cum),
            ..order
        };
        let (_, reserved) = self.reserved(&next);
        let mut execution = Execution::new(
            id,
            self.instrument.clone(),
            r.side,
            order.price,
            self.lots(r.last),
        );
        if !fee.is_zero() {
            execution = execution.with_fee(self.quote.clone(), fee);
        }
        let booked = self
            .ledger
            .book_execution_allowing_overdraft(
                &execution,
                Some((&reservation_id(r.order_id), reserved)),
            )
            .map_err(|e| format!("book order {}: {e}", r.order_id))?;
        if !booked.shortfalls.is_empty() {
            return Err(format!("shortfall: {:?}", booked.shortfalls));
        }
        if next.remaining.is_zero() {
            self.open.remove(&r.order_id);
        } else {
            self.open.insert(r.order_id, next);
        }
        Ok(())
    }

    fn close(&mut self, order_id: u64) -> Result<(), String> {
        if self.fault == Some(Fault::SkipRelease) {
            self.fault = None;
            self.open.remove(&order_id);
            return Ok(());
        }
        self.open
            .remove(&order_id)
            .ok_or_else(|| format!("close of order {order_id} without a reservation"))?;
        self.ledger
            .release(&reservation_id(order_id))
            .map(|_| ())
            .map_err(|e| format!("release order {order_id}: {e}"))
    }

    /// Applies events, in order. A `CancelRejected` closes nothing here: the
    /// fill before it already consumed the reservation.
    fn apply(&mut self, k: &Kernel<'_>, events: &[KernelEvent]) -> Result<(), String> {
        for event in events {
            match event {
                KernelEvent::Report(r) if r.kind == ExecKind::Accepted => {
                    self.check_accepted(r)?;
                    if self.fault == Some(Fault::ReleaseOnAccepted) {
                        self.fault = None;
                        self.close(r.order_id)?;
                    }
                }
                KernelEvent::Delivered(r) => match r.kind {
                    ExecKind::Fill | ExecKind::PartialFill => {
                        self.book_fill(r, k.money(r.fee_u).ok_or("fee")?)?;
                    }
                    ExecKind::Canceled => self.close(r.order_id)?,
                    ExecKind::Rejected if r.origin == Origin::Venue => self.close(r.order_id)?,
                    _ => {}
                },
                _ => {}
            }
        }
        Ok(())
    }

    /// One step: the action's reservations, then its events.
    fn step(
        &mut self,
        k: &Kernel<'_>,
        action: &KernelAction,
        first_id: u64,
        next_id: u64,
        events: &[KernelEvent],
    ) -> Result<(), String> {
        self.submit(action, first_id, next_id, events)?;
        self.apply(k, events)
    }

    /// Every reservation against its order, and the totals against them.
    fn check(&self) -> Result<(), String> {
        let snapshot = self.ledger.snapshot();
        if snapshot.reservations.len() != self.open.len() {
            return Err(format!(
                "{} reservations, {} open orders",
                snapshot.reservations.len(),
                self.open.len()
            ));
        }
        let mut base = Decimal::ZERO;
        let mut quote = Decimal::ZERO;
        for (id, order) in &self.open {
            let (asset, amount) = self.reserved(order);
            let held = self.ledger.reservation(&reservation_id(*id));
            if held != Some((&asset, amount)) {
                return Err(format!("order {id}: holds {held:?}, wants {amount}"));
            }
            if asset == self.base {
                base += amount;
            } else {
                quote += amount;
            }
        }
        for (asset, total) in [(&self.base, base), (&self.quote, quote)] {
            let balance = self.ledger.balance(asset).unwrap();
            if balance.reserved != total || balance.available + balance.reserved != balance.total {
                return Err(format!(
                    "{asset:?}: {balance:?}, reserved should be {total}"
                ));
            }
        }
        if !self.ledger.overdrafts().is_empty() {
            return Err(format!("overdraft: {:?}", self.ledger.overdrafts()));
        }
        if !self
            .ledger
            .reconcile()
            .map_err(|e| e.to_string())?
            .is_clean()
        {
            return Err("reconcile is not clean".to_string());
        }
        Ok(())
    }

    fn image(&self) -> LedgerImage {
        (self.ledger.snapshot(), self.ledger.journal().to_vec())
    }

    /// An open reservation for less than it started with.
    fn has_partly_consumed(&self, sizes: &BTreeMap<u64, Decimal>) -> bool {
        self.open
            .iter()
            .any(|(id, order)| sizes.get(id).is_some_and(|size| *size > order.remaining))
    }
}

/// A finished run: every report delivered, the queues really empty, the ledger
/// equal to the kernel's venue totals, and reservations and open orders agree
/// both ways. Every open reservation is an order the kernel's account holds
/// open. An order open there without a reservation was filled in full with its
/// cancel dropped at the end of the data: it still counts for capacity, but
/// nothing of it is left to reserve. Returns how many of those there were.
fn check_finished(
    shadow: &MarginShadow,
    last: &State,
    k: &Kernel<'_>,
    events: &[KernelEvent],
) -> Result<u64, String> {
    let q = last.queue_lengths();
    if q.deliveries != 0 || q.msgs != 0 || q.observed != 0 {
        return Err(format!("queues not empty: {q:?}"));
    }
    let venue = k.venue_summary(last).map_err(|e| e.to_string())?;
    let deposit = Decimal::from(DEPOSIT);
    let base = shadow.ledger.balance(&shadow.base).unwrap();
    if base.total != deposit + venue.position {
        return Err(format!(
            "base {} for position {}",
            base.total, venue.position
        ));
    }
    let quote = shadow.ledger.balance(&shadow.quote).unwrap();
    if quote.total != deposit + venue.realized - venue.total_fees {
        return Err(format!(
            "quote {} for realized {} and fees {}",
            quote.total, venue.realized, venue.total_fees
        ));
    }
    let fees = shadow.ledger.snapshot().fees_paid;
    let paid = fees.get(&shadow.quote).copied().unwrap_or(Decimal::ZERO);
    if paid != venue.total_fees {
        return Err(format!("fees {paid}, venue {}", venue.total_fees));
    }
    for id in shadow.open.keys() {
        if !last.is_open(*id) {
            return Err(format!("reservation of order {id}, closed in the kernel"));
        }
    }
    let mut filled_with_cancel_dropped = 0;
    for id in (1..last.next_order_id()).filter(|id| last.is_open(*id)) {
        if shadow.open.contains_key(&id) {
            continue;
        }
        let last_report = events.iter().rev().find_map(|e| match e {
            KernelEvent::Delivered(r) if r.order_id == id => Some(r),
            _ => None,
        });
        let filled = last_report.is_some_and(|r| r.kind == ExecKind::Fill && r.cum == r.qty);
        let cancel_dropped = events.iter().any(
            |e| matches!(e, KernelEvent::DroppedMsg { order_id, cancel: true } if *order_id == id),
        );
        if !(filled && cancel_dropped) {
            return Err(format!(
                "order {id} open in the kernel without a reservation: {last_report:?}"
            ));
        }
        filled_with_cancel_dropped += 1;
    }
    Ok(filled_with_cancel_dropped)
}

/// A run with the shadow at every decision.
struct ShadowRun {
    at_decision: Vec<MarginShadow>,
    last: MarginShadow,
    /// Starting size of every order that reserved.
    sizes: BTreeMap<u64, Decimal>,
}

/// Feeds a recorded run to a shadow, step by step, checking it at each step.
fn shadow_run(
    k: &Kernel<'_>,
    start: MarginShadow,
    states: &[State],
    last: &State,
    marks: &[usize],
    actions: &[KernelAction],
    events: &[KernelEvent],
) -> Result<ShadowRun, String> {
    let mut shadow = start;
    let mut sizes = BTreeMap::new();
    shadow.apply(k, &events[..marks.first().copied().unwrap_or(events.len())])?;
    let mut at_decision = Vec::with_capacity(states.len());
    for (i, state) in states.iter().enumerate() {
        shadow.check()?;
        at_decision.push(shadow.clone());
        let next = states.get(i + 1).unwrap_or(last);
        let end = marks.get(i + 1).copied().unwrap_or(events.len());
        let segment = &events[marks[i]..end];
        shadow.submit(
            &actions[i],
            state.next_order_id(),
            next.next_order_id(),
            segment,
        )?;
        for (id, order) in &shadow.open {
            sizes.entry(*id).or_insert(order.remaining);
        }
        shadow.apply(k, segment)?;
    }
    shadow.check()?;
    Ok(ShadowRun {
        at_decision,
        last: shadow,
        sizes,
    })
}

/// Replays `actions` from a snapshot of the state and the shadow.
fn replay_both(
    k: &Kernel<'_>,
    start: State,
    start_shadow: MarginShadow,
    actions: &[KernelAction],
) -> Result<(State, MarginShadow, Vec<KernelEvent>, Vec<Observation>), String> {
    let mut s = start;
    let mut shadow = start_shadow;
    let mut events = Vec::new();
    let mut observations = Vec::new();
    for action in actions {
        observations.push(k.observe(&s));
        let first_id = s.next_order_id();
        let mut step_events = Vec::new();
        k.step_with(&mut s, *action, &mut step_events);
        shadow.step(k, action, first_id, s.next_order_id(), &step_events)?;
        events.extend(step_events);
    }
    Ok((s, shadow, events, observations))
}

/// The scenario under both fill models.
fn both_models(seed: u64, index: u64) -> Vec<Scenario> {
    let mut scenarios = vec![scenario(seed, index)];
    scenarios.extend(partial_scenario(seed, index));
    scenarios
}

fn trajectory_run(k: &Kernel<'_>, sc: &Scenario, t: &Trajectory) -> ShadowRun {
    shadow_run(
        k,
        MarginShadow::new(&sc.cfg),
        &t.states,
        &t.last,
        &t.marks,
        &t.actions,
        &t.events,
    )
    .unwrap()
}

/// A fixed scenario with a through market: one lot clears per millisecond.
fn fixed(
    latency: KernelLatency,
    limits: Option<RiskLimits>,
    actions: Vec<KernelAction>,
    rows: u64,
) -> Scenario {
    let mut cfg = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100)
        .with_initial_capital(dec!(10000))
        .with_latency(latency)
        .with_fill_model(KernelFillModel::QueuePosition {
            rate: dec!(1),
            min_queue_ms: 0,
        });
    cfg.limits = limits;
    let ticks = (0..rows)
        .map(|i| MarketTick::new(1000 + i, dec!(13), dec!(1), dec!(14), dec!(1)))
        .collect();
    Scenario::fixed(cfg, ticks, actions)
}

fn bid(price: u64, lots: u64) -> KernelAction {
    KernelAction::OneSided {
        side: QuoteSide::Bid,
        price,
        lots,
    }
}

fn run_fixed(sc: &Scenario) -> (KernelRun, ShadowRun) {
    let data = sc.data();
    let k = kernel(sc, &data);
    let kr = run_kernel(&k, sc, true);
    let run = shadow_run(
        &k,
        MarginShadow::new(&sc.cfg),
        &kr.states,
        &kr.last,
        &kr.marks,
        &kr.actions,
        &kr.events,
    )
    .unwrap();
    (kr, run)
}

#[test]
fn test_margin_reservations_follow_the_orders_and_end_equal_to_the_kernel() {
    on_large_stack(|| {
        let (mut runs, mut finished, mut with_partials, mut partly, mut still_open) =
            (0, 0, 0, 0, 0);
        let mut filled_cancel_dropped = 0u64;
        for seed in 0..seeds().min(100) {
            for index in 0..DATASETS {
                for sc in both_models(seed, index) {
                    let ctx = format!("seed {seed} dataset {index} {:?}", sc.cfg.fill_model);
                    let data = sc.data();
                    let k = kernel(&sc, &data);
                    let kr = run_kernel(&k, &sc, true);
                    let run = shadow_run(
                        &k,
                        MarginShadow::new(&sc.cfg),
                        &kr.states,
                        &kr.last,
                        &kr.marks,
                        &kr.actions,
                        &kr.events,
                    )
                    .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                    runs += 1;
                    with_partials += u64::from(kr.events.iter().any(|e| {
                        matches!(e, KernelEvent::Delivered(r) if r.kind == ExecKind::PartialFill)
                    }));
                    partly += run
                        .at_decision
                        .iter()
                        .filter(|shadow| shadow.has_partly_consumed(&run.sizes))
                        .count();
                    assert_eq!(
                        kr.last.status(),
                        Status::Terminal(TermReason::EndOfData),
                        "{ctx}"
                    );
                    filled_cancel_dropped += check_finished(&run.last, &kr.last, &k, &kr.events)
                        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                    finished += 1;
                    still_open += run.last.open.len();
                }
            }
        }
        println!(
            "runs {runs}, finished {finished}, with partial fills {with_partials}, decisions with \
             a partly consumed reservation {partly}, reservations open at the end {still_open}, \
             open without a reservation after a fill and a dropped cancel \
             {filled_cancel_dropped}"
        );
        assert!(finished > 0, "runs to the end of the data");
        assert!(with_partials > 0, "runs with partial fills delivered");
        assert!(partly > 0, "partly consumed reservations");
        assert!(still_open > 0, "reservations open at the end");
        assert!(
            filled_cancel_dropped > 0,
            "orders filled with their cancel dropped"
        );
    });
}

#[test]
fn test_margin_reserves_at_submission_without_limits_before_any_report() {
    on_large_stack(|| {
        let latency = KernelLatency::zero(MS).with_delays(0, 3, 0, 2);
        let (kr, run) = run_fixed(&fixed(latency, None, vec![bid(15, 5)], 10));
        // At the next decision the order is reserved and nothing was reported.
        let events_so_far = &kr.events[..kr.marks[1]];
        assert!(
            !events_so_far.iter().any(|e| matches!(
                e,
                KernelEvent::Report(r) | KernelEvent::Delivered(r) if r.order_id == 1
            )),
            "{events_so_far:?}"
        );
        let shadow = &run.at_decision[1];
        assert_eq!(
            shadow.ledger.reservation(&reservation_id(1)),
            Some((&shadow.quote, dec!(75)))
        );
    });
}

#[test]
fn test_margin_reservation_amounts_are_in_the_reserved_asset() {
    on_large_stack(|| {
        let mut cfg = KernelConfig::new(dec!(0.25), dec!(0.1), dec!(1000), dec!(10), 100)
            .with_initial_capital(dec!(10000));
        cfg.latency = KernelLatency::zero(MS);
        let ticks = (0..4u64)
            .map(|i| MarketTick::new(1000 + i, dec!(1), dec!(1), dec!(10), dec!(1)))
            .collect();
        let action = KernelAction::Quote {
            bid: 15,
            ask: 41,
            bid_lots: 3,
            ask_lots: 7,
        };
        let sc = Scenario::fixed(cfg, ticks, vec![action]);
        let (_, run) = run_fixed(&sc);
        let shadow = &run.at_decision[1];
        // 3 lots of 0.1 at 15 ticks of 0.25, in quote; 7 lots of 0.1, in base.
        assert_eq!(
            shadow.ledger.reservation(&reservation_id(1)),
            Some((&shadow.quote, dec!(1.125)))
        );
        assert_eq!(
            shadow.ledger.reservation(&reservation_id(2)),
            Some((&shadow.base, dec!(0.7)))
        );
    });
}

#[test]
fn test_margin_reservation_shrinks_with_each_partial_and_cancel_releases_the_rest() {
    on_large_stack(|| {
        let latency = KernelLatency::zero(MS).with_delays(0, 0, 3, 1);
        let sc = fixed(
            latency,
            None,
            vec![bid(15, 5), KernelAction::Hold, bid(1, 2)],
            10,
        );
        let data = sc.data();
        let k = kernel(&sc, &data);
        let (kr, run) = run_fixed(&sc);
        // The delivered reports of order 1, one by one.
        let mut shadow = MarginShadow::new(&sc.cfg);
        shadow.submit(&kr.actions[0], 1, 2, &[]).unwrap();
        let mut held = Vec::new();
        for event in &kr.events {
            let KernelEvent::Delivered(r) = event else {
                continue;
            };
            if r.order_id != 1 {
                continue;
            }
            shadow.apply(&k, std::slice::from_ref(event)).unwrap();
            held.push((
                r.kind,
                shadow
                    .ledger
                    .reservation(&reservation_id(1))
                    .map(|(_, amount)| amount),
            ));
        }
        // 15 per lot; the rest goes back with the cancel.
        assert_eq!(
            held,
            vec![
                (ExecKind::Accepted, Some(dec!(75))),
                (ExecKind::PartialFill, Some(dec!(60))),
                (ExecKind::PartialFill, Some(dec!(45))),
                (ExecKind::PartialFill, Some(dec!(30))),
                (ExecKind::PartialFill, Some(dec!(15))),
                (ExecKind::Canceled, None),
            ]
        );
        run.last.check().unwrap();
        check_finished(&run.last, &kr.last, &k, &kr.events).unwrap();
    });
}

#[test]
fn test_margin_duplicate_fill_changes_nothing() {
    on_large_stack(|| {
        let sc = fixed(KernelLatency::zero(MS), None, vec![bid(15, 5)], 8);
        let data = sc.data();
        let k = kernel(&sc, &data);
        let kr = run_kernel(&k, &sc, true);
        let fills: Vec<KernelEvent> = kr
            .events
            .iter()
            .filter(|e| {
                matches!(e, KernelEvent::Delivered(r)
                    if matches!(r.kind, ExecKind::PartialFill | ExecKind::Fill))
            })
            .copied()
            .collect();
        assert_eq!(fills.len(), 5, "four partials and the fill");
        let mut shadow = MarginShadow::new(&sc.cfg);
        shadow.submit(&kr.actions[0], 1, 2, &[]).unwrap();
        let assert_repeats_change_nothing = |shadow: &mut MarginShadow, repeats: &[KernelEvent]| {
            let (image, open) = (shadow.image(), shadow.open.clone());
            shadow.apply(&k, repeats).unwrap();
            assert_eq!(shadow.image(), image);
            assert_eq!(shadow.open, open);
            shadow.check().unwrap();
        };
        shadow.apply(&k, &fills[..2]).unwrap();
        // The latest partial, then an older one.
        assert_repeats_change_nothing(&mut shadow, &fills[1..2]);
        assert_repeats_change_nothing(&mut shadow, &fills[..1]);
        shadow.apply(&k, &fills[2..]).unwrap();
        assert!(shadow.open.is_empty());
        // The terminal fill and a partial, once the order has closed.
        assert_repeats_change_nothing(&mut shadow, &fills[4..]);
        assert_repeats_change_nothing(&mut shadow, &fills[..1]);

        // The same fill with other terms is an error, not a repeat.
        let KernelEvent::Delivered(mut other) = fills[0] else {
            panic!("not a delivered fill");
        };
        other.last += 1;
        assert!(shadow.apply(&k, &[KernelEvent::Delivered(other)]).is_err());
    });
}

#[test]
fn test_margin_no_reservation_for_what_is_refused_and_a_reduced_size_reserves_less() {
    on_large_stack(|| {
        // A crossed quote is rejected locally on both sides.
        let crossed = KernelAction::Quote {
            bid: 20,
            ask: 10,
            bid_lots: 1,
            ask_lots: 1,
        };
        let (_, run) = run_fixed(&fixed(KernelLatency::zero(MS), None, vec![crossed], 3));
        assert!(run.at_decision[1].open.is_empty());
        assert!(run.last.ledger.snapshot().reservations.is_empty());

        // A position limit of 2 lots reduces an order of 5 lots to 2.
        let limits = RiskLimits::new(dec!(2), Decimal::MAX, dec!(0)).unwrap();
        let far = bid(1, 5);
        let (kr, run) = run_fixed(&fixed(KernelLatency::zero(MS), Some(limits), vec![far], 3));
        assert!(kr.events.iter().any(|e| matches!(
            e,
            KernelEvent::Gate(g) if g.order_id == 1 && g.submitted == 2
        )));
        let shadow = &run.at_decision[1];
        assert_eq!(
            shadow.ledger.reservation(&reservation_id(1)),
            Some((&shadow.quote, dec!(2)))
        );
    });
}

#[test]
fn test_margin_dropped_activation_keeps_its_reservation_at_the_end() {
    on_large_stack(|| {
        // The activation is still in flight when the data ends.
        let latency = KernelLatency::zero(MS).with_delays(0, 100, 0, 0);
        let sc = fixed(latency, None, vec![bid(15, 5)], 4);
        let data = sc.data();
        let k = kernel(&sc, &data);
        let (kr, run) = run_fixed(&sc);
        assert!(kr.events.iter().any(|e| matches!(
            e,
            KernelEvent::DroppedMsg {
                order_id: 1,
                cancel: false
            }
        )));
        assert_eq!(kr.last.live_lots(), (5, 0));
        assert!(kr.last.is_open(1));
        assert_eq!(
            run.last.ledger.reservation(&reservation_id(1)),
            Some((&run.last.quote, dec!(75)))
        );
        check_finished(&run.last, &kr.last, &k, &kr.events).unwrap();
    });
}

/// A partial fill and a cancel, for the negative controls.
fn control_scenario() -> Scenario {
    fixed(
        KernelLatency::zero(MS).with_delays(0, 0, 3, 1),
        None,
        vec![bid(15, 5), bid(1, 2)],
        10,
    )
}

fn control_run(start: MarginShadow, sc: &Scenario) -> Result<(), String> {
    let data = sc.data();
    let k = kernel(sc, &data);
    let kr = run_kernel(&k, sc, true);
    let run = shadow_run(
        &k,
        start,
        &kr.states,
        &kr.last,
        &kr.marks,
        &kr.actions,
        &kr.events,
    )?;
    check_finished(&run.last, &kr.last, &k, &kr.events).map(|_| ())
}

#[test]
fn test_margin_negative_controls_are_caught() {
    on_large_stack(|| {
        let sc = control_scenario();
        let clean = MarginShadow::new(&sc.cfg);
        assert_eq!(control_run(clean.clone(), &sc), Ok(()));
        // A reservation released while the order rests, with no fill to expose
        // it: only the check from the kernel's open orders can catch this.
        let resting = fixed(KernelLatency::zero(MS), None, vec![bid(1, 2)], 4);
        let early = MarginShadow::new(&resting.cfg).with_fault(Fault::ReleaseOnAccepted);
        let error = control_run(early, &resting).unwrap_err();
        assert!(error.contains("without a reservation"), "{error}");

        let release = control_run(clean.clone().with_fault(Fault::SkipRelease), &sc);
        assert!(release.is_err(), "a skipped release passed");
        let booking = control_run(clean.clone().with_fault(Fault::SkipBooking), &sc);
        assert!(booking.is_err(), "a skipped booking passed");

        // Restoring the state without the shadow: the fills of the order that
        // was open at the snapshot have nothing to consume.
        let data = sc.data();
        let k = kernel(&sc, &data);
        let kr = run_kernel(&k, &sc, true);
        let resumed = replay_both(&k, kr.states[1], clean, &kr.actions[1..]);
        assert!(resumed.is_err(), "a restore without the shadow passed");
    });
}

#[test]
fn test_margin_restore_with_reservations_pending_reproduces_state_and_ledger() {
    on_large_stack(|| {
        let (mut restored, mut partly, mut with_messages, mut with_cancel) =
            (0u64, 0u64, 0u64, 0u64);
        for seed in 0..seeds().min(25) {
            for index in 0..DATASETS {
                for sc in both_models(seed, index) {
                    let data = sc.data();
                    let k = kernel(&sc, &data);
                    let t = trajectory(&k, &sc);
                    let run = trajectory_run(&k, &sc, &t);
                    // As a new process would: its own dataset and kernel.
                    let fresh_data = sc.data();
                    let fresh = kernel(&sc, &fresh_data);
                    for (i, snapshot) in t.states.iter().enumerate() {
                        let shadow = &run.at_decision[i];
                        if shadow.open.is_empty() {
                            continue;
                        }
                        let ctx = format!("seed {seed} dataset {index} decision {i}");
                        let back = decode(&encode(snapshot), &fresh).unwrap();
                        assert_eq!(back, *snapshot, "{ctx}: decoded");
                        let (last, resumed, events, observations) =
                            replay_both(&fresh, back, shadow.clone(), &t.actions[i..])
                                .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                        assert_eq!(last, t.last, "{ctx}: state");
                        assert_eq!(events, t.events[t.marks[i]..], "{ctx}: events");
                        assert_eq!(observations, t.observations[i..], "{ctx}: observations");
                        assert_eq!(resumed.image(), run.last.image(), "{ctx}: ledger");
                        assert_eq!(resumed.open, run.last.open, "{ctx}: open orders");
                        restored += 1;
                        partly += u64::from(shadow.has_partly_consumed(&run.sizes));
                        with_messages += u64::from(snapshot.queue_lengths().msgs > 0);
                        with_cancel += u64::from(events.iter().any(|e| {
                            matches!(e, KernelEvent::Delivered(r) if r.kind == ExecKind::Canceled)
                        }));
                    }
                }
            }
        }
        println!(
            "restores with reservations pending {restored}, partly consumed {partly}, with venue \
             messages {with_messages}, followed by a delivered cancel {with_cancel}"
        );
        assert!(restored > 0, "restores with reservations pending");
        assert!(partly > 0, "restores of a partly consumed reservation");
        assert!(with_messages > 0, "restores with venue messages pending");
        assert!(with_cancel > 0, "restores followed by a release");
    });
}

#[test]
fn test_margin_forks_keep_their_ledgers_apart() {
    on_large_stack(|| {
        let (mut forks, mut diverged) = (0u64, 0u64);
        for seed in 0..seeds().min(25) {
            for index in 0..DATASETS {
                for sc in both_models(seed, index) {
                    let data = sc.data();
                    let k = kernel(&sc, &data);
                    let t = trajectory(&k, &sc);
                    let run = trajectory_run(&k, &sc, &t);
                    let i = t.states.len() / 2;
                    let parent = t.states[i];
                    let base = &run.at_decision[i];
                    let ctx = format!("seed {seed} dataset {index}");

                    // The other fork runs first, on its own copy of the ledger.
                    let other = KernelAction::Quote {
                        bid: parent_bid(&t.observations[i]),
                        ask: t.observations[i].ask + 1,
                        bid_lots: 1,
                        ask_lots: 1,
                    };
                    let others = vec![other; t.actions.len() - i];
                    let (other_last, other_ledger, ..) =
                        replay_both(&k, parent, base.clone(), &others).unwrap();
                    let (last, ledger, ..) =
                        replay_both(&k, parent, base.clone(), &t.actions[i..]).unwrap();
                    assert_eq!(last, t.last, "{ctx}: reference fork");
                    assert_eq!(ledger.image(), run.last.image(), "{ctx}: reference ledger");

                    let (again, again_ledger, ..) =
                        replay_both(&k, parent, base.clone(), &others).unwrap();
                    assert_eq!(again, other_last, "{ctx}: other fork");
                    assert_eq!(
                        again_ledger.image(),
                        other_ledger.image(),
                        "{ctx}: other ledger"
                    );

                    forks += 1;
                    diverged += u64::from(other_ledger.image() != ledger.image());
                }
            }
        }
        println!("forks {forks}, with diverging ledgers {diverged}");
        assert!(forks > 0, "forks");
        assert!(diverged > 0, "forks whose ledgers diverge");
    });
}

/// Final fingerprint, event hash and ledger per instance.
fn run_instances(indices: &[u64]) -> Vec<(u64, u128, u128, LedgerImage)> {
    indices
        .iter()
        .filter_map(|i| {
            let sc = partial_scenario(i / DATASETS, i % DATASETS)?;
            let data = sc.data();
            let k = kernel(&sc, &data);
            let t = trajectory(&k, &sc);
            let run = trajectory_run(&k, &sc, &t);
            Some((
                *i,
                fingerprint(&t.last),
                events_hash(&t.events),
                run.last.image(),
            ))
        })
        .collect()
}

#[test]
fn test_margin_ledgers_do_not_depend_on_worker_assignment() {
    let order: Vec<u64> = (0..120).collect();
    let mut reference = Vec::new();
    on_large_stack(|| reference = run_instances(&order));
    assert!(reference.len() > 60, "instances: {}", reference.len());
    let mut distinct: Vec<&LedgerImage> = Vec::new();
    for instance in &reference {
        if !distinct.contains(&&instance.3) {
            distinct.push(&instance.3);
        }
    }
    assert!(
        distinct.len() > 30,
        "ledgers must differ: {}",
        distinct.len()
    );
    for workers in [1usize, 2, 7] {
        let mut reversed = order.clone();
        reversed.reverse();
        let chunk = reversed.len().div_ceil(workers);
        let mut got: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = reversed
                .chunks(chunk)
                .map(|part| {
                    std::thread::Builder::new()
                        .stack_size(STACK_BYTES)
                        .spawn_scoped(scope, move || run_instances(part))
                        .unwrap()
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        got.sort_unstable_by_key(|instance| instance.0);
        assert_eq!(got, reference, "{workers} workers");
    }
}
