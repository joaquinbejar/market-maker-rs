//! The action transaction and the resumable common advance.
//!
//! Every function mirrors one engine routine call for call (`replace_quotes`,
//! `advance_events`, `apply_next_venue_message`, `apply_cancel`,
//! `deliver_next`, `match_active_orders`, `process_fill`, `mark_to`,
//! `queue_delivery`), so stamps, message sequences, execution ids and
//! jitter draws are consumed at the same call sites.

use crate::Decimal;
use crate::types::action::Dir;
use core::num::NonZeroU64;

use super::account;
use super::book::{
    Blank, Delivery, ExecKind, FillRec, MsgKind, ObsEntry, OrderState, Origin, ReportRec, VenueMsg,
    VenueOrder,
};
use super::config::{ValidFill, ValidatedConfig};
use super::data::{KernelDataset, KernelTick, NS_PER_MS};
use super::encode::insert_by_id;
use super::gate::{self, Balances, GateCheck, GateVerdict, MarginTerms};
use super::grid::{half_units_decimal, scaled_decimal};
use super::state::{
    ActionStatus, EventSink, GateRecord, HaltReason, KernelEvent, KernelState, Phase, QueueLengths,
    RejectReason, Status, TermReason,
};

type Halt = HaltReason;

/// The combined limits of a hedged kernel, with the counter of the orders
/// they refuse; `None` for a kernel alone.
pub(super) type Combined<'a> = Option<(&'a super::hedge::CombinedCheck, &'a mut u64)>;

/// Counts the agent-facing events forwarded to the caller's sink.
///
/// A run delivers at most three status reports per order, a fill report per
/// lot and `len` observations: `3 * 2S + 2S * Q + len`, below `2^64` for
/// `S <= 2^60`, `2S * Q` within `i64` and `len <= u32::MAX`, so the count cannot
/// overflow; if it did, `overflow` halts the step.
pub(super) struct Em<'a, E: EventSink> {
    sink: &'a mut E,
    pub(super) count: u64,
    pub(super) overflow: bool,
}

impl<'a, E: EventSink> Em<'a, E> {
    pub(super) fn new(sink: &'a mut E) -> Self {
        Self {
            sink,
            count: 0,
            overflow: false,
        }
    }
}

impl<E: EventSink> EventSink for Em<'_, E> {
    fn event(&mut self, event: KernelEvent) {
        if event.is_agent_facing() {
            match self.count.checked_add(1) {
                Some(count) => self.count = count,
                None => self.overflow = true,
            }
        }
        self.sink.event(event);
    }

    fn hedge(&mut self, event: super::hedge::HedgeEvent) {
        self.sink.hedge(event);
    }

    fn queues(&mut self, lengths: QueueLengths) {
        self.sink.queues(lengths);
    }
}

/// Events of one action transaction, forwarded only if it commits. Two
/// orders produce at most a gate record and a report each, so four slots
/// suffice; a fifth event fails the transaction.
struct Buf {
    items: [Option<KernelEvent>; 4],
    len: usize,
    full: bool,
}

impl Buf {
    fn new() -> Self {
        Self {
            items: [None; 4],
            len: 0,
            full: false,
        }
    }
}

impl EventSink for Buf {
    fn event(&mut self, event: KernelEvent) {
        match (self.items.get_mut(self.len), self.len.checked_add(1)) {
            (Some(slot), Some(next)) => {
                *slot = Some(event);
                self.len = next;
            }
            (None, _) | (_, None) => self.full = true,
        }
    }
}

/// Reports the queue lengths to the sink.
fn probe<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize, E: EventSink>(
    s: &St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) {
    em.queues(s.queue_lengths());
}

type St<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize> =
    KernelState<ORD, MSG, DLV, OBS>;

fn bump(v: &mut u64) -> Result<(), Halt> {
    *v = v.checked_add(1).ok_or(Halt::CounterOverflow)?;
    Ok(())
}

fn stamp<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &mut St<ORD, MSG, DLV, OBS>,
) -> Result<(u64, u64), Halt> {
    let seq = s.stamp_seq;
    bump(&mut s.stamp_seq)?;
    Ok((s.now_ns, seq))
}

/// `at + (base + jitter) * quantum`, drawing the jitter like the engine's
/// `delayed`.
fn delayed<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    at: u64,
    base_q: u64,
) -> Result<u64, Halt> {
    let jitter = s.rng_latency.draw(cfg.latency.jitter_q);
    let quanta = base_q.checked_add(jitter).ok_or(Halt::TimeOverflow)?;
    let ns = cfg.latency.ns(quanta).ok_or(Halt::TimeOverflow)?;
    at.checked_add(ns).ok_or(Halt::TimeOverflow)
}

fn status_report(
    ts_ns: u64,
    seq: u64,
    order: &VenueOrder,
    kind: ExecKind,
    origin: Origin,
    cancel_pending: bool,
) -> ReportRec {
    ReportRec {
        ts_ns,
        seq,
        order_id: order.id,
        side: order.side.into(),
        kind,
        origin,
        cancel_pending,
        price: order.price,
        qty: order.qty,
        last: 0,
        cum: order.filled,
        fee_u: 0,
    }
}

fn queue_delivery<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    report: ReportRec,
    fill: Option<FillRec>,
    em: &mut E,
) -> Result<(), Halt> {
    // FIFO: jitter delays a report but never lets it overtake an earlier one.
    let at = delayed(cfg, s, report.ts_ns, cfg.latency.ack_q)?.max(s.last_delivery_ns);
    s.last_delivery_ns = at;
    em.event(KernelEvent::Report(report));
    s.deliveries
        .push(Delivery {
            at_ns: at,
            report,
            fill,
        })
        .map_err(|_| Halt::CounterOverflow)
}

fn schedule<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &mut St<ORD, MSG, DLV, OBS>,
    eff_ns: u64,
    kind: MsgKind,
) -> Result<(), Halt> {
    let msg = VenueMsg {
        eff_ns,
        seq: s.next_msg_seq,
        kind,
    };
    bump(&mut s.next_msg_seq)?;
    let index = s
        .msgs
        .position(|m| m.key() > msg.key())
        .unwrap_or(s.msgs.len());
    s.msgs.insert(index, msg).map_err(|_| Halt::CounterOverflow)
}

fn apply_next_msg<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) -> Result<(), Halt> {
    // The expiry's time, and room for its sequence number, are checked
    // before the message is taken: a halt leaves it in the queue, and an
    // active order never lacks the expiry its quote asked for.
    let expiry = match s.msgs.first().map(|m| m.kind) {
        Some(MsgKind::Activate {
            order,
            life: Some(life),
        }) if order.price > 0 && order.qty > 0 => {
            let at = order
                .active_after_ns
                .checked_add(life.get())
                .ok_or(Halt::TimeOverflow)?;
            s.next_msg_seq.checked_add(1).ok_or(Halt::CounterOverflow)?;
            Some(at)
        }
        _ => None,
    };
    let Some(msg) = s.msgs.pop_front() else {
        return Ok(());
    };
    match msg.kind {
        MsgKind::Activate { mut order, .. } => {
            let valid = order.price > 0 && order.qty > 0;
            let (ts, seq) = stamp(s)?;
            let report = if valid {
                order.state = OrderState::Active;
                status_report(ts, seq, &order, ExecKind::Accepted, Origin::Venue, false)
            } else {
                order.state = OrderState::Rejected;
                // The venue rejected it: the reservation is released.
                s.gate
                    .release(order.side, order.remaining())
                    .ok_or(Halt::CounterOverflow)?;
                status_report(
                    ts,
                    seq,
                    &order,
                    ExecKind::Rejected,
                    Origin::Venue,
                    order.cancel_inflight,
                )
            };
            queue_delivery(cfg, s, report, None, em)?;
            if valid {
                s.orders.push(order).map_err(|_| Halt::CounterOverflow)?;
                // The expiry takes the venue message the activation held.
                if let Some(at) = expiry {
                    schedule(s, at, MsgKind::Expire { id: order.id })?;
                }
            } else if order.cancel_inflight {
                insert_by_id(&mut s.awaiting, order).map_err(|_| Halt::CounterOverflow)?;
            }
            Ok(())
        }
        MsgKind::Cancel { id } => apply_cancel(cfg, s, id, em),
        MsgKind::Expire { id } => apply_expire(cfg, s, id, em),
    }
}

/// The venue's cancel of a resting order at its expiry. An order with the
/// agent's cancel in flight then awaits it, which the venue rejects as for a
/// fill before a cancel.
fn apply_expire<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    id: u64,
    em: &mut E,
) -> Result<(), Halt> {
    let Some(index) = s.orders.position(|o| o.id == id && !o.state.is_terminal()) else {
        return Ok(());
    };
    let (ts, seq) = stamp(s)?;
    let Some(mut order) = s.orders.remove(index) else {
        return Ok(());
    };
    order.state = OrderState::Canceled;
    s.gate
        .release(order.side, order.remaining())
        .ok_or(Halt::CounterOverflow)?;
    em.event(KernelEvent::Expired {
        order_id: id,
        at_ns: s.now_ns,
    });
    let report = status_report(
        ts,
        seq,
        &order,
        ExecKind::Canceled,
        Origin::Venue,
        order.cancel_inflight,
    );
    queue_delivery(cfg, s, report, None, em)?;
    if order.cancel_inflight {
        insert_by_id(&mut s.awaiting, order).map_err(|_| Halt::CounterOverflow)?;
    }
    Ok(())
}

/// Removes the pending expiry of order `id`, if any: an order that ended
/// otherwise keeps no venue message for its expiry.
fn remove_expiry<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &mut St<ORD, MSG, DLV, OBS>,
    id: u64,
) {
    if let Some(index) = s
        .msgs
        .position(|m| matches!(m.kind, MsgKind::Expire { id: target } if target == id))
    {
        let _ = s.msgs.remove(index);
    }
}

fn apply_cancel<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    id: u64,
    em: &mut E,
) -> Result<(), Halt> {
    let (ts, seq) = stamp(s)?;
    if let Some(index) = s.orders.position(|o| o.id == id && !o.state.is_terminal())
        && let Some(mut order) = s.orders.remove(index)
    {
        order.state = OrderState::Canceled;
        s.gate
            .release(order.side, order.remaining())
            .ok_or(Halt::CounterOverflow)?;
        remove_expiry(s, id);
        let report = status_report(ts, seq, &order, ExecKind::Canceled, Origin::Venue, false);
        return queue_delivery(cfg, s, report, None, em);
    }
    let pending = s.msgs.position(|m| match m.kind {
        MsgKind::Activate { order, .. } => order.id == id,
        MsgKind::Cancel { .. } | MsgKind::Expire { .. } => false,
    });
    if let Some(index) = pending {
        if let Some(VenueMsg {
            kind: MsgKind::Activate { mut order, .. },
            ..
        }) = s.msgs.remove(index)
        {
            order.state = OrderState::Canceled;
            s.gate
                .release(order.side, order.remaining())
                .ok_or(Halt::CounterOverflow)?;
            let report = status_report(ts, seq, &order, ExecKind::Canceled, Origin::Venue, false);
            queue_delivery(cfg, s, report, None, em)?;
        }
        return Ok(());
    }
    if let Some(index) = s.awaiting.position(|o| o.id == id)
        && let Some(order) = s.awaiting.remove(index)
    {
        let report = status_report(
            ts,
            seq,
            &order,
            ExecKind::CancelRejected,
            Origin::Venue,
            false,
        );
        queue_delivery(cfg, s, report, None, em)?;
    }
    Ok(())
}

/// Signed lots of a fill as a base delta, and the price, in `Decimal`.
fn fill_decimals(
    cfg: &ValidatedConfig,
    dir: Dir,
    price: u64,
    qty: u64,
) -> Option<(Decimal, Decimal)> {
    let qty = i128::from(qty);
    let signed = match dir {
        Dir::Buy => qty,
        Dir::Sell => qty.checked_neg()?,
    };
    Some((
        cfg.lot.to_decimal(signed)?,
        cfg.tick.to_decimal(i128::from(price))?,
    ))
}

fn deliver_next<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) -> Result<(), Halt> {
    let Some(delivery) = s.deliveries.pop_front() else {
        return Ok(());
    };
    em.event(KernelEvent::Delivered(delivery.report));
    if let Some(fill) = delivery.fill {
        let dir = Dir::try_from(fill.side).map_err(|_| Halt::LedgerOverflow)?;
        let (delta, price) =
            fill_decimals(cfg, dir, fill.price, fill.qty).ok_or(Halt::LedgerOverflow)?;
        s.agent
            .pos
            .update_fill(delta, price, fill.ts_ns)
            .ok_or(Halt::LedgerOverflow)?;
        bump(&mut s.agent.delivered_fills)?;
    }
    match (delivery.report.kind, delivery.report.origin) {
        (ExecKind::Rejected, Origin::Local) => bump(&mut s.agent.delivered_local_rejects)?,
        (
            ExecKind::Rejected
            | ExecKind::Accepted
            | ExecKind::Canceled
            | ExecKind::CancelRejected
            | ExecKind::PartialFill
            | ExecKind::Fill,
            Origin::Local | Origin::Venue,
        ) => {}
    }
    account::on_delivery(&mut s.account, &delivery.report);
    Ok(())
}

/// Pops an observation. Returns true when the run pauses (decision point)
/// or stops (truncation).
fn pop_observation<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) -> bool {
    let Some(entry) = s.observed.pop_front() else {
        return false;
    };
    if s.step >= cfg.s_max {
        s.status = Status::Truncated;
        s.phase = Phase::Terminal;
        return true;
    }
    s.agent.last_obs_row = Some(entry.row);
    em.event(KernelEvent::Observed {
        row: entry.row,
        at_ns: s.now_ns,
    });
    s.resume = s.phase;
    s.phase = Phase::AwaitDecision;
    true
}

/// Event classes, in their order at equal times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Venue,
    Delivery,
    Observation,
}

/// Runs every event with time at or before `until`, venue messages first,
/// then deliveries, then observations at equal times. Returns true when an
/// observation paused the run.
fn run_events<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    until: u64,
    em: &mut E,
) -> Result<bool, Halt> {
    loop {
        let candidates = [
            (s.msgs.first().map(|m| m.eff_ns), Class::Venue),
            (s.deliveries.first().map(|d| d.at_ns), Class::Delivery),
            (s.observed.first().map(|o| o.ready_ns), Class::Observation),
        ];
        let mut next: Option<(u64, Class)> = None;
        for (time, class) in candidates {
            if let Some(t) = time
                && t <= until
                && next.is_none_or(|n| (t, class) < n)
            {
                next = Some((t, class));
            }
        }
        let Some((time, class)) = next else {
            return Ok(false);
        };
        s.now_ns = time.max(s.now_ns);
        match class {
            Class::Venue => apply_next_msg(cfg, s, em)?,
            Class::Delivery => deliver_next(cfg, s, em)?,
            Class::Observation => {
                let paused = pop_observation(cfg, s, em);
                probe(s, em);
                if paused {
                    return Ok(true);
                }
                continue;
            }
        }
        probe(s, em);
    }
}

/// Lots of `order` the queue position model fills on `row` it crosses, as
/// `QueuePositionFillModel::fill_for`: the model's cumulative quantity less
/// what is already filled, which can leave nothing when the touch grew. The
/// cleared volume is `floor(rate_lots * ms in queue / per_ms)`, from the
/// order's entry to the queue.
fn queue_fill_lots(
    order: &VenueOrder,
    row: &KernelTick,
    (rate_lots, per_ms): (u64, u64),
    min_queue_ms: u64,
) -> Option<u64> {
    let in_queue_ns = row.ts_ns.saturating_sub(order.active_after_ns);
    let in_queue_ms = in_queue_ns.checked_div(NS_PER_MS)?;
    if in_queue_ms < min_queue_ms {
        return Some(0);
    }
    // `bid < ask` on every row, so a price is one level only.
    let (through, touch_lots) = match order.side {
        Dir::Buy => (row.ask < order.price, row.ask_lots),
        Dir::Sell => (row.bid > order.price, row.bid_lots),
    };
    let ahead = if through { 0 } else { touch_lots };
    let cleared = u128::from(rate_lots)
        .checked_mul(u128::from(in_queue_ms))?
        .checked_div(u128::from(per_ms))?;
    let Some(cumulative) = cleared.checked_sub(u128::from(ahead)).filter(|v| *v > 0) else {
        return Some(0);
    };
    if cumulative >= u128::from(order.qty) {
        return Some(order.remaining());
    }
    Some(u64::try_from(cumulative).ok()?.saturating_sub(order.filled))
}

/// Matches the resting orders against `row`, in book order, as
/// `match_active_orders` with the fill model.
fn match_row<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    row: &KernelTick,
    em: &mut E,
) -> Result<(), Halt> {
    let mid2 = row.mid2().ok_or(Halt::LedgerOverflow)?;
    if mid2 > 0 {
        s.gate.mark2 = Some(mid2);
    }
    for index in 0..s.orders.len() {
        let Some(mut order) = s.orders.get(index).copied() else {
            continue;
        };
        if order.state != OrderState::Active || row.ts_ns <= order.active_after_ns {
            continue;
        }
        let crosses = match order.side {
            Dir::Buy => row.ask <= order.price,
            Dir::Sell => row.bid >= order.price,
        };
        let remaining = order.remaining();
        if !crosses || remaining == 0 || order.price == 0 {
            continue;
        }
        let qty = match cfg.fill {
            ValidFill::Immediate => remaining,
            ValidFill::QueuePosition {
                rate_lots,
                per_ms,
                min_queue_ms,
            } => queue_fill_lots(&order, row, (rate_lots, per_ms), min_queue_ms)
                .ok_or(Halt::CounterOverflow)?,
        };
        if qty == 0 {
            continue;
        }
        let fee_u = cfg.fee_u(order.price, qty).ok_or(Halt::LedgerOverflow)?;
        order.filled = order.filled.checked_add(qty).ok_or(Halt::CounterOverflow)?;
        let kind = if qty == remaining {
            order.state = OrderState::Filled;
            ExecKind::Fill
        } else {
            ExecKind::PartialFill
        };
        let (ts, seq) = stamp(s)?;
        let exec_id = s.next_exec_id;
        let report = ReportRec {
            ts_ns: ts,
            seq,
            order_id: order.id,
            side: order.side.into(),
            kind,
            origin: Origin::Venue,
            cancel_pending: order.cancel_inflight,
            price: order.price,
            qty: order.qty,
            last: qty,
            cum: order.filled,
            fee_u,
        };
        let fill = FillRec {
            ts_ns: s.now_ns,
            exec_id,
            order_id: order.id,
            side: order.side.into(),
            price: order.price,
            qty,
            fee_u,
        };
        let mut gate = s.gate;
        gate.fill(order.side, qty).ok_or(Halt::CounterOverflow)?;
        process_fill(cfg, s, gate, order.side, &fill)?;
        bump(&mut s.counters.venue_fills)?;
        em.event(KernelEvent::Fill(fill));
        if let Some(slot) = s.orders.get_mut(index) {
            *slot = order;
        }
        queue_delivery(cfg, s, report, Some(fill), em)?;
        probe(s, em);
    }
    let mut index = 0;
    while let Some(order) = s.orders.get(index).copied() {
        if !order.state.is_terminal() {
            index = index.saturating_add(1);
            continue;
        }
        let _ = s.orders.remove(index);
        remove_expiry(s, order.id);
        if order.cancel_inflight {
            insert_by_id(&mut s.awaiting, order).map_err(|_| Halt::CounterOverflow)?;
        }
    }
    Ok(())
}

/// Books a fill in the venue ledger and the gate (`gate` already holds
/// the fill), atomically, as `process_fill`.
fn process_fill<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    gate: super::gate::GateState,
    dir: Dir,
    fill: &FillRec,
) -> Result<(), Halt> {
    let overflow = Halt::LedgerOverflow;
    let mut v = s.venue;
    let notional_u = cfg.notional_u(fill.price, fill.qty).ok_or(overflow)?;
    let (delta, price) = fill_decimals(cfg, dir, fill.price, fill.qty).ok_or(overflow)?;
    let notional = price.checked_mul(delta.abs()).ok_or(overflow)?;
    match dir {
        Dir::Buy => {
            v.cash_u = v.cash_u.checked_sub(notional_u).ok_or(overflow)?;
            v.realized = v.realized.checked_sub(notional).ok_or(overflow)?;
        }
        Dir::Sell => {
            v.cash_u = v.cash_u.checked_add(notional_u).ok_or(overflow)?;
            v.realized = v.realized.checked_add(notional).ok_or(overflow)?;
        }
    }
    v.pos
        .update_fill(delta, price, fill.ts_ns)
        .ok_or(overflow)?;
    v.fixed_fees_u = v.fixed_fees_u.checked_add(fill.fee_u).ok_or(overflow)?;
    v.last_mark2 = Some(fill.price.checked_mul(2).ok_or(overflow)?);
    v.max_abs_pos = v.max_abs_pos.max(gate.venue_pos_lots.unsigned_abs());
    let next_exec_id = s.next_exec_id.checked_add(1).ok_or(Halt::CounterOverflow)?;
    // Cross-check of the exact integer shadow against the Decimal values.
    let cash = scaled_decimal(v.cash_u, cfg.s).ok_or(overflow)?;
    let lots = cfg
        .lot
        .to_decimal(i128::from(gate.venue_pos_lots))
        .ok_or(overflow)?;
    if cash != v.realized
        || lots != v.pos.quantity
        || !within(cfg, v.cash_u)
        || !within(cfg, v.fixed_fees_u)
    {
        return Err(overflow);
    }
    s.venue = v;
    s.gate = gate;
    s.next_exec_id = next_exec_id;
    Ok(())
}

/// `|value| <= B`.
fn within(cfg: &ValidatedConfig, value: i128) -> bool {
    value.unsigned_abs() <= cfg.bound_b
}

/// Equity at `mark2`, in half-U.
pub(super) fn equity_h<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    cfg: &ValidatedConfig,
    s: &St<ORD, MSG, DLV, OBS>,
    mark2: u64,
) -> Option<i128> {
    let v = &s.venue;
    cfg.c0_u
        .checked_add(v.cash_u)?
        .checked_sub(v.fixed_fees_u)?
        .checked_mul(2)?
        .checked_add(cfg.position_value_h(s.gate.venue_pos_lots, mark2)?)
}

/// Marks the venue ledger at the row's mid and updates peak and drawdown.
fn row_end<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize, E: EventSink>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    index: u64,
    row: &KernelTick,
    em: &mut E,
) -> Result<(), Halt> {
    let overflow = Halt::LedgerOverflow;
    let mid2 = row.mid2().ok_or(overflow)?;
    let equity = equity_h(cfg, s, mid2).ok_or(overflow)?;
    let mut v = s.venue;
    v.last_mark2 = Some(mid2);
    if equity > v.peak_h {
        v.peak_h = equity;
    }
    let drawdown = v.peak_h.checked_sub(equity).ok_or(overflow)?;
    if drawdown > v.max_dd_h {
        v.max_dd_h = drawdown;
    }
    // Cross-check: the engine's Decimal equity equals the integer one.
    let capital = scaled_decimal(cfg.c0_u, cfg.s).ok_or(overflow)?;
    let fees = scaled_decimal(v.fixed_fees_u, cfg.s).ok_or(overflow)?;
    let mid = cfg
        .tick
        .to_decimal(i128::from(mid2))
        .and_then(|m| m.checked_div(Decimal::TWO))
        .ok_or(overflow)?;
    let decimal_equity = v
        .pos
        .quantity
        .checked_mul(mid)
        .and_then(|u| v.realized.checked_add(u))
        .and_then(|t| capital.checked_add(t))
        .and_then(|e| e.checked_sub(fees))
        .ok_or(overflow)?;
    let half = cfg.half_bound();
    if half_units_decimal(equity, cfg.s) != Some(decimal_equity)
        || equity.unsigned_abs() > half
        || v.max_dd_h.unsigned_abs() > cfg.drawdown_bound()
        || v.peak_h.unsigned_abs() > half
    {
        return Err(overflow);
    }
    s.venue = v;
    em.event(KernelEvent::RowEnd {
        row: index,
        ts_ns: row.ts_ns,
        equity_half_u: equity,
    });
    Ok(())
}

fn end_of_data<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) -> Result<(), Halt> {
    while !s.deliveries.is_empty() {
        deliver_next(cfg, s, em)?;
    }
    let dropped_msgs = u64::try_from(s.msgs.len()).map_err(|_| Halt::CounterOverflow)?;
    let dropped_obs = u64::try_from(s.observed.len()).map_err(|_| Halt::CounterOverflow)?;
    for msg in s.msgs.as_slice() {
        em.event(match msg.kind {
            MsgKind::Activate { order, .. } => KernelEvent::DroppedMsg {
                order_id: order.id,
                cancel: false,
            },
            MsgKind::Cancel { id } => KernelEvent::DroppedMsg {
                order_id: id,
                cancel: true,
            },
            MsgKind::Expire { id } => KernelEvent::DroppedExpiry { order_id: id },
        });
    }
    s.msgs.clear();
    s.observed.clear();
    s.counters.eof_dropped_msgs = s
        .counters
        .eof_dropped_msgs
        .checked_add(dropped_msgs)
        .ok_or(Halt::CounterOverflow)?;
    s.counters.eof_dropped_obs = s
        .counters
        .eof_dropped_obs
        .checked_add(dropped_obs)
        .ok_or(Halt::CounterOverflow)?;
    s.status = Status::Terminal(TermReason::EndOfData);
    s.phase = Phase::Terminal;
    probe(s, em);
    em.event(KernelEvent::EndOfData {
        dropped_msgs,
        dropped_obs,
    });
    Ok(())
}

/// Stops the run with `reason`.
pub(super) fn halt<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    s: &mut St<ORD, MSG, DLV, OBS>,
    reason: HaltReason,
    em: &mut E,
) {
    s.status = Status::Halted(reason);
    s.phase = Phase::Terminal;
    s.counters.halts = 1;
    em.event(KernelEvent::Halted(reason));
}

/// Runs the advance until the next decision point or the end of the run.
pub(super) fn advance<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    data: &KernelDataset,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) {
    if let Err(reason) = advance_inner(cfg, data, s, em) {
        halt(s, reason, em);
    }
}

fn advance_inner<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    data: &KernelDataset,
    s: &mut St<ORD, MSG, DLV, OBS>,
    em: &mut E,
) -> Result<(), Halt> {
    loop {
        match s.phase {
            Phase::AwaitDecision | Phase::Terminal => return Ok(()),
            Phase::BeforeMatch => {
                let index = s.cur_row.ok_or(Halt::CounterOverflow)?;
                let row = *data.row(index).ok_or(Halt::CounterOverflow)?;
                if run_events(cfg, s, row.ts_ns, em)? {
                    return Ok(());
                }
                s.now_ns = s.now_ns.max(row.ts_ns);
                match_row(cfg, s, &row, em)?;
                let ready =
                    delayed(cfg, s, row.ts_ns, cfg.latency.observation_q)?.max(s.last_observed_ns);
                s.last_observed_ns = ready;
                s.observed
                    .push(ObsEntry {
                        ready_ns: ready,
                        row: index,
                    })
                    .map_err(|_| Halt::CounterOverflow)?;
                probe(s, em);
                s.phase = Phase::ObsQueued;
            }
            Phase::ObsQueued => {
                if let Some(index) = s.cur_row {
                    let row = *data.row(index).ok_or(Halt::CounterOverflow)?;
                    if run_events(cfg, s, row.ts_ns, em)? {
                        return Ok(());
                    }
                    row_end(cfg, s, index, &row, em)?;
                }
                if data.row(s.cursor).is_none() {
                    return end_of_data(cfg, s, em);
                }
                if s.step >= cfg.s_max {
                    s.status = Status::Truncated;
                    s.phase = Phase::Terminal;
                    return Ok(());
                }
                s.cur_row = Some(s.cursor);
                bump(&mut s.cursor)?;
                s.phase = Phase::BeforeMatch;
            }
        }
    }
}

/// Admission: ranges and the agent-known capacity account. Reads only the
/// configuration and the account.
pub(super) fn admit<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    cfg: &ValidatedConfig,
    s: &St<ORD, MSG, DLV, OBS>,
    bid: u64,
    ask: u64,
    bid_lots: u64,
    ask_lots: u64,
) -> Result<(), RejectReason> {
    let price_ok = |p: u64| p >= cfg.min_price && p <= cfg.p;
    if !price_ok(bid) || !price_ok(ask) {
        return Err(RejectReason::PriceRange);
    }
    let size_ok = |q: u64| q >= 1 && q <= cfg.q;
    if !size_ok(bid_lots) || !size_ok(ask_lots) {
        return Err(RejectReason::SizeRange);
    }
    capacity_ok::<ORD, MSG, DLV>(account::held(&s.account))
}

/// The capacity part of admission for an account holding `held` orders.
///
/// The deliveries check is on what the agent knows, three per open order. It
/// does not count partial fill reports, which [`super::FinancialKernel::new`]
/// makes room for: with `DLV >= (3 + R) * ORD` it never binds before the orders
/// check.
pub(super) fn capacity_ok<const ORD: usize, const MSG: usize, const DLV: usize>(
    held: u64,
) -> Result<(), RejectReason> {
    let cap = |n: usize| u128::try_from(n).unwrap_or(u128::MAX);
    let next = u128::from(held).saturating_add(2);
    if next > cap(ORD) {
        return Err(RejectReason::CapacityOrders);
    }
    if next.saturating_mul(2) > cap(MSG) {
        return Err(RejectReason::CapacityMsgs);
    }
    if next.saturating_mul(3) > cap(DLV) {
        return Err(RejectReason::CapacityDeliveries);
    }
    Ok(())
}

/// The orders of an admitted action: one or two legs `(side, price,
/// lots)`, and whether the quote is crossed (both legs rejected locally).
#[derive(Debug, Clone, Copy)]
pub(super) struct Legs {
    pub(super) legs: [Option<(Dir, u64, u64)>; 2],
    pub(super) crossed: bool,
    /// The venue time each leg rests from its activation, for an expiring
    /// quote.
    pub(super) life: Option<NonZeroU64>,
}

/// What the transaction may change, saved before it runs. Every change is
/// an append (deliveries, account), an insert of a message with a fresh
/// sequence number, a cancel flag set on the targets of those messages, or
/// a scalar, so restoring these values rolls the state back exactly.
pub(super) struct Undo {
    rng_latency: super::rng::KernelRng,
    stamp_seq: u64,
    next_msg_seq: u64,
    next_order_id: u64,
    last_delivery_ns: u64,
    local_rejects: u64,
    gate: super::gate::GateState,
    deliveries: usize,
    account: usize,
}

impl Undo {
    pub(super) fn save<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
        s: &St<ORD, MSG, DLV, OBS>,
    ) -> Self {
        Self {
            rng_latency: s.rng_latency,
            stamp_seq: s.stamp_seq,
            next_msg_seq: s.next_msg_seq,
            next_order_id: s.next_order_id,
            last_delivery_ns: s.last_delivery_ns,
            local_rejects: s.counters.local_rejects,
            gate: s.gate,
            deliveries: s.deliveries.len(),
            account: s.account.len(),
        }
    }

    pub(super) fn restore<
        const ORD: usize,
        const MSG: usize,
        const DLV: usize,
        const OBS: usize,
    >(
        self,
        s: &mut St<ORD, MSG, DLV, OBS>,
    ) {
        // New messages carry sequence numbers from the saved one on; the
        // cancels among them name the orders whose flag the transaction set.
        let mut i = 0;
        while let Some(msg) = s.msgs.get(i).copied() {
            if msg.seq < self.next_msg_seq {
                i = i.saturating_add(1);
                continue;
            }
            if let MsgKind::Cancel { id } = msg.kind {
                set_cancel(s, id, false);
            }
            let _ = s.msgs.remove(i);
        }
        s.deliveries.truncate(self.deliveries);
        s.account.truncate(self.account);
        s.rng_latency = self.rng_latency;
        s.stamp_seq = self.stamp_seq;
        s.next_msg_seq = self.next_msg_seq;
        s.next_order_id = self.next_order_id;
        s.last_delivery_ns = self.last_delivery_ns;
        s.counters.local_rejects = self.local_rejects;
        s.gate = self.gate;
    }
}

/// Applies an admitted action as `replace_quotes`, in place; on failure
/// the undo record restores the pre-action state exactly, without copying
/// the whole state.
pub(super) fn apply_quote<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    s: &mut St<ORD, MSG, DLV, OBS>,
    legs: Legs,
    combined: Combined<'_>,
    em: &mut E,
) -> Result<(), Halt> {
    let undo = Undo::save(s);
    let mut buf = Buf::new();
    let mut result = quote_transaction(cfg, s, legs, combined, &mut buf);
    if result.is_ok() && buf.full {
        result = Err(Halt::CounterOverflow);
    }
    if let Err(reason) = result {
        undo.restore(s);
        return Err(reason);
    }
    for event in buf.items.iter().take(buf.len).flatten() {
        em.event(*event);
    }
    probe(s, em);
    Ok(())
}

/// Smallest cancel target above `after`: a resting or pending order with no
/// cancel in flight.
fn next_target<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &St<ORD, MSG, DLV, OBS>,
    after: Option<u64>,
) -> Option<u64> {
    let above = |id: u64| after.is_none_or(|a| id > a);
    let resting = s
        .orders
        .as_slice()
        .iter()
        .filter(|o| !o.state.is_terminal() && !o.cancel_inflight)
        .map(|o| o.id);
    let pending = s.msgs.as_slice().iter().filter_map(|m| match m.kind {
        MsgKind::Activate { order, .. } if !order.cancel_inflight => Some(order.id),
        MsgKind::Activate { .. } | MsgKind::Cancel { .. } | MsgKind::Expire { .. } => None,
    });
    resting.chain(pending).filter(|id| above(*id)).min()
}

fn set_cancel<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &mut St<ORD, MSG, DLV, OBS>,
    id: u64,
    inflight: bool,
) {
    for order in s.orders.as_mut_slice() {
        if order.id == id {
            order.cancel_inflight = inflight;
        }
    }
    for msg in s.msgs.as_mut_slice() {
        if let MsgKind::Activate { order, .. } = &mut msg.kind
            && order.id == id
        {
            order.cancel_inflight = inflight;
        }
    }
}

pub(super) fn quote_transaction<
    const ORD: usize,
    const MSG: usize,
    const DLV: usize,
    const OBS: usize,
    E: EventSink,
>(
    cfg: &ValidatedConfig,
    t: &mut St<ORD, MSG, DLV, OBS>,
    legs: Legs,
    mut combined: Combined<'_>,
    em: &mut E,
) -> Result<(), Halt> {
    let now = t.now_ns;
    let mut after = None;
    while let Some(id) = next_target(t, after) {
        let eff = delayed(cfg, t, now, cfg.latency.cancel_q)?;
        // The message first: the undo record finds every flag it sets
        // through the cancel messages it inserted.
        schedule(t, eff, MsgKind::Cancel { id })?;
        set_cancel(t, id, true);
        after = Some(id);
    }
    let crossed = legs.crossed;
    for (dir, price, lots) in legs.legs.into_iter().flatten() {
        let id = t.next_order_id;
        bump(&mut t.next_order_id)?;
        let _submitted = stamp(t)?;
        account::open(&mut t.account, id).map_err(|_| Halt::CounterOverflow)?;
        let mut refused = crossed;
        let mut size = lots;
        let mut record = None;
        if !refused && let Some(limits) = &cfg.limits {
            let decision = gate::check(limits, &t.gate, &cfg.tick, &cfg.lot, dir, price, lots)
                .map_err(|_| Halt::GateNumeric)?;
            record = Some(GateRecord {
                order_id: id,
                side: dir.into(),
                requested: lots,
                verdict: decision.verdict,
                submitted: decision.submitted,
                check: GateCheck::Limits,
            });
            match decision.verdict {
                GateVerdict::Accepted | GateVerdict::Reduced => size = decision.submitted,
                GateVerdict::ReducedOffLot | GateVerdict::Rejected => {
                    refused = true;
                }
            }
        }
        // The balance gate binds after the limits, on what they approved:
        // their record is emitted first, then its own.
        if !refused && let Some(margin) = &cfg.margin {
            if let Some(record) = record.take() {
                em.event(KernelEvent::Gate(record));
            }
            let terms = MarginTerms {
                tick: &cfg.tick,
                lot: &cfg.lot,
                fee_scale: cfg.fee_scale().ok_or(Halt::GateNumeric)?,
                fee_num: margin.fee_num,
            };
            let balances = balances(cfg, t, &terms).ok_or(Halt::GateNumeric)?;
            let decision =
                gate::margin_check(&terms, &balances, dir, price, size).ok_or(Halt::GateNumeric)?;
            record = Some(GateRecord {
                order_id: id,
                side: dir.into(),
                requested: size,
                verdict: decision.verdict,
                submitted: decision.submitted,
                check: GateCheck::Margin,
            });
            match decision.verdict {
                GateVerdict::Accepted | GateVerdict::Reduced => size = decision.submitted,
                GateVerdict::ReducedOffLot | GateVerdict::Rejected => {
                    refused = true;
                }
            }
        }
        // The combined limits of a hedged kernel bind after the instrument's
        // own, on what it would submit; without them nothing here changes.
        if !refused && let Some((check, rejects)) = combined.as_mut() {
            let admitted = check
                .admits_quoted(&t.gate, &cfg.tick, &cfg.lot, dir, price, size)
                .ok_or(Halt::GateNumeric)?;
            if !admitted {
                refused = true;
                bump(rejects)?;
                record = Some(GateRecord {
                    order_id: id,
                    side: dir.into(),
                    requested: lots,
                    verdict: GateVerdict::Rejected,
                    submitted: 0,
                    check: GateCheck::Combined,
                });
            }
        }
        if let Some(record) = record {
            em.event(KernelEvent::Gate(record));
        }
        let order = VenueOrder {
            id,
            side: dir,
            price,
            qty: size,
            ..VenueOrder::BLANK
        };
        if refused {
            let (ts, seq) = stamp(t)?;
            let mut rejected = order;
            rejected.state = OrderState::Rejected;
            let report =
                status_report(ts, seq, &rejected, ExecKind::Rejected, Origin::Local, false);
            queue_delivery(cfg, t, report, None, em)?;
            bump(&mut t.counters.local_rejects)?;
            continue;
        }
        t.gate.reserve(dir, size).ok_or(Halt::CounterOverflow)?;
        let eff = delayed(cfg, t, now, cfg.latency.submit_q)?;
        schedule(
            t,
            eff,
            MsgKind::Activate {
                order: VenueOrder {
                    active_after_ns: eff,
                    ..order
                },
                life: legs.life,
            },
        )?;
    }
    Ok(())
}

/// The balances of the balance gate: the quote balance (initial capital,
/// fill cash flows and fees) less the reservations of the live buys (resting
/// or in flight to the venue, as the gate's live lots), and the base deposit
/// plus the venue position less the live sells.
fn balances<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    cfg: &ValidatedConfig,
    t: &St<ORD, MSG, DLV, OBS>,
    terms: &MarginTerms<'_>,
) -> Option<Balances> {
    let margin = cfg.margin.as_ref()?;
    let activations = t.msgs.as_slice().iter().filter_map(|m| match &m.kind {
        MsgKind::Activate { order, .. } => Some(order),
        MsgKind::Cancel { .. } | MsgKind::Expire { .. } => None,
    });
    let reserved = t
        .orders
        .as_slice()
        .iter()
        .chain(activations)
        .filter(|o| o.side == Dir::Buy)
        .try_fold(0i128, |sum, o| {
            sum.checked_add(terms.buy_reservation_u(o.price, o.remaining())?)
        })?;
    let quote_free_u = cfg
        .c0_u
        .checked_add(t.venue.cash_u)?
        .checked_sub(t.venue.fixed_fees_u)?
        .checked_sub(reserved)?;
    let base_free_lots = i128::from(margin.base_lots)
        .checked_add(i128::from(t.gate.venue_pos_lots))?
        .checked_sub(i128::from(t.gate.live_sell_lots))?;
    Some(Balances {
        quote_free_u,
        base_free_lots,
    })
}

/// Records the outcome of an action in the counters.
pub(super) fn record<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &mut St<ORD, MSG, DLV, OBS>,
    applied: ActionStatus,
) -> Result<(), Halt> {
    let counted = match applied {
        ActionStatus::Applied => None,
        ActionStatus::Rejected(reason) => Some(reason),
        ActionStatus::QuoteError(_) => Some(RejectReason::QuoteError),
    };
    if let Some(reason) = counted {
        let slot = s
            .counters
            .rejected
            .get_mut(reason.index())
            .ok_or(Halt::CounterOverflow)?;
        bump(slot)?;
    }
    s.last = Some(applied);
    Ok(())
}
