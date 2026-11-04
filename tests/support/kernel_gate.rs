//! The hedged driver and the replay helpers of the kernel's replay, snapshot
//! and batch gates.

#![allow(dead_code)]

use super::harness::*;
use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionStatus, KernelAction, KernelEvent, KernelFillModel, Observation, Phase, QuoteSide,
    capacity_need, fnv1a128,
};

/// Absolute position, in lots, at which the driver hedges.
pub const HEDGE_AT: u64 = 2;

/// The script, except that a position of `HEDGE_AT` lots or more is reduced
/// by an order that crosses the touch, one bounded hedge at a time.
pub fn hedged(sc: &Scenario, decision: u64, obs: &Observation) -> KernelAction {
    let lots = u64::try_from(
        ((obs.position / sc.cfg.lot_size).abs().round())
            .normalize()
            .mantissa(),
    )
    .unwrap();
    if lots < HEDGE_AT {
        return sc.action(decision, obs.bid, obs.ask);
    }
    let (side, price) = if obs.position.is_sign_positive() {
        (QuoteSide::Ask, obs.bid)
    } else {
        (QuoteSide::Bid, obs.ask)
    };
    KernelAction::OneSided {
        side,
        price,
        lots: lots.min(max_lots(sc)),
    }
}

/// One full run of the hedged driver.
pub struct Trajectory {
    pub states: Vec<State>,
    pub observations: Vec<Observation>,
    pub actions: Vec<KernelAction>,
    pub outcomes: Vec<ActionStatus>,
    /// Events before each decision.
    pub marks: Vec<usize>,
    pub events: Vec<KernelEvent>,
    pub last: State,
}

pub fn trajectory(k: &Kernel<'_>, sc: &Scenario) -> Trajectory {
    let mut events = Vec::new();
    let mut s = k.reset_with(sc.latency_seed, &mut events);
    let mut t = Trajectory {
        states: Vec::new(),
        observations: Vec::new(),
        actions: Vec::new(),
        outcomes: Vec::new(),
        marks: Vec::new(),
        events: Vec::new(),
        last: s,
    };
    let mut decision = 0;
    while s.phase() == Phase::AwaitDecision {
        let obs = k.observe(&s);
        let action = hedged(sc, decision, &obs);
        t.states.push(s);
        t.observations.push(obs);
        t.actions.push(action);
        t.marks.push(events.len());
        t.outcomes
            .push(k.step_with(&mut s, action, &mut events).applied);
        decision += 1;
    }
    t.events = events;
    t.last = s;
    t
}

pub fn events_hash(events: &[KernelEvent]) -> u128 {
    fnv1a128(format!("{events:?}").as_bytes())
}

/// Whether replayed events include a new report, which means venue messages
/// were scheduled and, with jitter, generator draws consumed.
pub fn scheduled(events: &[KernelEvent]) -> bool {
    events.iter().any(|e| matches!(e, KernelEvent::Report(_)))
}

/// Replays `actions` from `start`, returning the final state, the events and
/// the observation at each decision.
pub fn replay(
    k: &Kernel<'_>,
    start: State,
    actions: &[KernelAction],
) -> (State, Vec<KernelEvent>, Vec<Observation>) {
    let mut s = start;
    let mut events = Vec::new();
    let mut observations = Vec::new();
    for action in actions {
        observations.push(k.observe(&s));
        k.step_with(&mut s, *action, &mut events);
    }
    (s, events, observations)
}

pub fn parent_bid(obs: &Observation) -> u64 {
    obs.bid.saturating_sub(1).max(1)
}

/// Stack of the threads that hold these states by value: their delivery queue
/// is several times the size of the other kernel tests'.
pub const STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread with [`STACK_BYTES`] of stack.
pub fn on_large_stack(body: impl FnOnce() + Send) {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, body)
            .unwrap()
            .join()
            .unwrap();
    });
}

/// The scenario under the queue position model, when the harness capacity
/// has room for its partial fill reports.
pub fn partial_scenario(seed: u64, index: u64) -> Option<Scenario> {
    let mut sc = scenario(seed, index);
    let mix = seed.wrapping_mul(31).wrapping_add(index);
    let rate_lots = [1u64, 1, 2, 3][usize::try_from(mix % 4).unwrap()];
    let rate = (Decimal::from(rate_lots) * sc.cfg.lot_size).normalize();
    let min_queue_ms = [0u64, 0, 1, 3][usize::try_from((mix / 4) % 4).unwrap()];
    sc.cfg.fill_model = KernelFillModel::QueuePosition { rate, min_queue_ms };
    fits(&sc).then_some(sc)
}

pub fn fits(sc: &Scenario) -> bool {
    let need = capacity_need(&sc.cfg, &sc.data()).unwrap();
    (3 + usize::try_from(need.partial_reports_per_order).unwrap()) * ORD <= DLV
}
