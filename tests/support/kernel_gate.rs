//! The hedged driver and the replay helpers of the kernel's replay, snapshot
//! and batch gates.

#![allow(dead_code)]

use super::harness::*;
use arbitrage_rs::hedge::{
    Band, HedgeDecision, HedgePlan, NoHedgeReason, PositionView, plan_band_hedge,
};
use arbitrage_rs::{
    Currency, FeeRate, Instrument, InstrumentId, Level, Price, Quantity, Timestamp, TopOfBook,
};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionStatus, KernelAction, KernelEvent, KernelFillModel, Observation, Phase, QuoteSide,
    capacity_need, fnv1a128,
};
use market_maker_rs::execution::Action;

/// Absolute position, in lots, at which the driver hedges.
pub const HEDGE_AT: u64 = 2;

/// Whole units of `unit` in `value`.
fn whole(value: Decimal, unit: Decimal) -> u64 {
    let count = (value / unit).normalize();
    assert_eq!(count.scale(), 0, "{value} is not a multiple of {unit}");
    u64::try_from(count.mantissa()).unwrap()
}

/// What arbitrage-rs plans for the observed position: a band that hedges from
/// `HEDGE_AT` lots back to zero, at the observed touch.
///
/// The level size is the largest hedge, not the row's touch size. It does not
/// stand for observed liquidity: the fill models here do not limit an order by
/// the touch, and this keeps the hedges the recorded trajectories were made
/// with. `touch_lots` gives a real size, for a direct test of that cap. The fee
/// is zero because it only changes the plan's estimates, which are not used.
pub fn band_plan(sc: &Scenario, obs: &Observation, touch_lots: Option<u64>) -> HedgeDecision {
    let lot = sc.cfg.lot_size;
    let tick = sc.cfg.tick_size;
    let quantity = |lots: u64| Quantity::new(Decimal::from(lots) * lot).unwrap();
    let largest = max_lots(sc);
    // Zero ticks: no row observed yet, so that side is not quoted.
    let level = |ticks: u64| {
        (ticks > 0).then(|| {
            Level::new(
                Price::new(Decimal::from(ticks) * tick).unwrap(),
                quantity(touch_lots.unwrap_or(largest)),
            )
        })
    };
    plan_band_hedge(&PositionView {
        as_of: Timestamp::from_millis(i64::try_from(obs.ts_ns / MS).unwrap()),
        quote_currency: Currency::new("USD").unwrap(),
        exposure: obs.position,
        band: Band::symmetric(Decimal::from(HEDGE_AT - 1) * lot).unwrap(),
        instrument: Instrument {
            id: InstrumentId::new("BASE-QUOTE").unwrap(),
            expiry: None,
        },
        book: TopOfBook {
            bid: level(obs.bid),
            ask: level(obs.ask),
        },
        fee: FeeRate::zero(),
        lot_size: Some(quantity(1)),
        max_quantity: Some(quantity(largest)),
    })
    .unwrap()
}

/// The order a hedge leg sends: at the leg's price, which is the touch.
pub fn hedge_action(sc: &Scenario, plan: &HedgePlan) -> KernelAction {
    let side = match plan.leg.side {
        Action::Sell => QuoteSide::Ask,
        Action::Buy => QuoteSide::Bid,
        other => panic!("hedge side {other:?}"),
    };
    KernelAction::OneSided {
        side,
        price: whole(plan.leg.price.value(), sc.cfg.tick_size),
        lots: whole(plan.leg.quantity.value(), sc.cfg.lot_size),
    }
}

/// The script, except that arbitrage-rs reduces a position outside its band by
/// an order that crosses the touch, one bounded hedge at a time.
pub fn hedged(sc: &Scenario, decision: u64, obs: &Observation) -> KernelAction {
    // The band only means "from `HEDGE_AT` lots" for a whole number of lots.
    whole(obs.position.abs(), sc.cfg.lot_size);
    match band_plan(sc, obs, None) {
        HedgeDecision::Hedge(plan) => hedge_action(sc, &plan),
        HedgeDecision::NoHedge {
            reason: NoHedgeReason::InsideBand,
            ..
        } => sc.action(decision, obs.bid, obs.ask),
        other => panic!("no hedge for an observed position: {other:?}"),
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

/// FNV-1a-128 of the events' `Debug` output, without the gate records'
/// `check` field (added after these hashes were recorded): the hashes then
/// show that a run without the balance gate emits the events it emitted.
pub fn events_hash(events: &[KernelEvent]) -> u128 {
    let text = format!("{events:?}")
        .replace(", check: Limits", "")
        .replace(", check: Combined", "");
    assert!(
        !text.contains("check: "),
        "a gate check outside the recorded hashes"
    );
    fnv1a128(text.as_bytes())
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
