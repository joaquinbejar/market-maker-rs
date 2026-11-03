//! Shared fixtures for the kernel integration tests: a SplitMix64
//! generator, seeded scenarios, a scripted strategy for the engine and the
//! same script for the kernel, and an exact engine comparison.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::cell::RefCell;
use std::rc::Rc;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::engine::FeeCurrency;
use market_maker_rs::backtest::kernel::{
    ActionStatus, ExecKind, FinancialKernel, GateVerdict, KernelAction, KernelConfig,
    KernelDataset, KernelEvent, KernelLatency, KernelState, Observation, Phase, ReportRec, Status,
    TermReason,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, ExecutionKind,
    ExecutionReport, MarketTick, SimLatencyConfig, SimulatedFill, SlippageModel, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::{LimitsPolicy, PretradeDecision};
use market_maker_rs::strategy::quote::Quote;

/// Capacities used by the seeded runs.
pub const ORD: usize = 64;
pub const MSG: usize = 128;
pub const DLV: usize = 192;
pub const OBS: usize = 128;

pub type Kernel<'d> = FinancialKernel<'d, ORD, MSG, DLV, OBS>;
pub type State = KernelState<ORD, MSG, DLV, OBS>;

/// Nanoseconds per millisecond.
pub const MS: u64 = 1_000_000;

/// SplitMix64.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[usize::try_from(self.below(items.len() as u64)).unwrap()]
    }
}

/// Number of seeds of the property loops: `KERNEL_SEEDS` or 500.
pub fn seeds() -> u64 {
    std::env::var("KERNEL_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}

/// Datasets per seed.
pub const DATASETS: u64 = 20;

/// Limits of a scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitsKind {
    None,
    Small,
    Notional,
    Max,
}

/// A generated scenario.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub cfg: KernelConfig,
    pub ticks: Vec<MarketTick>,
    pub latency_seed: u64,
    pub script_seed: u64,
    pub limits: LimitsKind,
    /// Fixed actions per decision (then `Hold`) instead of the script.
    pub fixed: Option<Vec<KernelAction>>,
}

impl Scenario {
    /// A hand-written scenario with fixed actions.
    pub fn fixed(cfg: KernelConfig, ticks: Vec<MarketTick>, actions: Vec<KernelAction>) -> Self {
        Self {
            cfg,
            ticks,
            latency_seed: 0,
            script_seed: 0,
            limits: LimitsKind::None,
            fixed: Some(actions),
        }
    }

    /// The action for decision `k` at the observed `(bid, ask)`.
    pub fn action(&self, k: u64, bid: u64, ask: u64) -> KernelAction {
        match &self.fixed {
            Some(actions) => actions
                .get(usize::try_from(k).unwrap())
                .copied()
                .unwrap_or(KernelAction::Hold),
            None => script(self.script_seed, k, bid, ask, max_lots(self)),
        }
    }
    pub fn data(&self) -> KernelDataset {
        KernelDataset::from_ticks(&self.ticks, &self.cfg).unwrap()
    }

    pub fn engine_latency(&self) -> SimLatencyConfig {
        let l = self.cfg.latency;
        SimLatencyConfig::new()
            .with_observation_ms(l.observation_q)
            .with_submit_ms(l.submit_q)
            .with_cancel_ms(l.cancel_q)
            .with_ack_ms(l.ack_q)
            .with_jitter(l.jitter_q, self.latency_seed)
    }
}

fn latency(rng: &mut SplitMix64) -> KernelLatency {
    let class = rng.below(3);
    let delay = |rng: &mut SplitMix64| match class {
        0 => 0,
        1 => rng.below(4),
        _ => 3 + rng.below(12),
    };
    let lat = KernelLatency::zero(MS).with_delays(delay(rng), delay(rng), delay(rng), delay(rng));
    let jitter = match rng.below(10) {
        0..=4 => 0,
        5..=8 => 1 + rng.below(3),
        _ => 8 + rng.below(8),
    };
    lat.with_jitter(jitter)
}

/// Generates scenario `index` of `seed`.
pub fn scenario(seed: u64, index: u64) -> Scenario {
    let mut rng = SplitMix64(
        seed.wrapping_mul(0x1_0000)
            .wrapping_add(index)
            .wrapping_add(17),
    );
    let tick = rng.pick(&[dec!(0.01), dec!(0.25), dec!(1), dec!(0.5)]);
    let lot = rng.pick(&[dec!(1), dec!(0.1), dec!(0.001)]);
    let fee = rng.pick(&[dec!(0), dec!(0.0001), dec!(-0.00005), dec!(0.001)]);
    let lat = latency(&mut rng);
    let rows = 20 + rng.below(30);
    let window = lat.submit_q + lat.cancel_q + lat.ack_q + 6 * lat.jitter_q + lat.observation_q;
    let gap_max = 1 + window / 4;
    let mut ts = 1_000u64;
    let mut mid = 1_000i64 + i64::try_from(rng.below(200)).unwrap();
    let mut ticks = Vec::new();
    for _ in 0..rows {
        if !rng.chance(30) {
            ts += 1 + rng.below(gap_max);
        }
        mid += i64::try_from(rng.below(5)).unwrap() - 2;
        mid = mid.clamp(50, 5_000);
        let half = 1 + i64::try_from(rng.below(2)).unwrap();
        let bid = u64::try_from(mid - half).unwrap();
        let ask = u64::try_from(mid + half).unwrap();
        let price = |t: u64| Decimal::from(t) * tick;
        let size = |l: u64| Decimal::from(l) * lot;
        ticks.push(MarketTick::new(
            ts,
            price(bid),
            size(1 + rng.below(9)),
            price(ask),
            size(1 + rng.below(9)),
        ));
    }
    let limits = match rng.below(4) {
        0 => LimitsKind::None,
        1 => LimitsKind::Small,
        2 => LimitsKind::Notional,
        _ => LimitsKind::Max,
    };
    let mut cfg = KernelConfig::new(
        tick,
        lot,
        Decimal::from(10_000) * tick,
        Decimal::from(10) * lot,
        rows + 5,
    )
    .with_initial_capital(Decimal::from(1_000_000 + rng.below(1_000)))
    .with_fee_rate(fee)
    .with_latency(lat);
    cfg.limits = match limits {
        LimitsKind::None => None,
        LimitsKind::Small => Some(
            RiskLimits::new(Decimal::from(2 + rng.below(6)) * lot, Decimal::MAX, dec!(0)).unwrap(),
        ),
        LimitsKind::Notional => {
            // Room for a few lots at the mid, usually not a whole number of lots.
            let notional = Decimal::from(mid) * tick * lot * Decimal::from(3 + rng.below(4))
                + tick * lot / dec!(3);
            Some(RiskLimits::new(Decimal::from(50) * lot, notional, dec!(0)).unwrap())
        }
        LimitsKind::Max => Some(RiskLimits::new(Decimal::MAX, Decimal::MAX, dec!(0)).unwrap()),
    };
    Scenario {
        cfg,
        ticks,
        latency_seed: rng.next(),
        script_seed: rng.next(),
        limits,
        fixed: None,
    }
}

/// The scripted action for decision `k` at the observed `(bid, ask)`.
pub fn script(seed: u64, k: u64, bid: u64, ask: u64, max_lots: u64) -> KernelAction {
    let mut rng = SplitMix64(seed ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let roll = rng.below(100);
    let lots = |rng: &mut SplitMix64| {
        if rng.chance(10) {
            max_lots
        } else {
            1 + rng.below(4.min(max_lots))
        }
    };
    if roll < 25 {
        return KernelAction::Hold;
    }
    if roll < 30 {
        return KernelAction::QuoteError(market_maker_rs::backtest::kernel::ErrCode(5));
    }
    if roll < 38 {
        let b = ask + rng.below(2);
        let a = b.saturating_sub(rng.below(3)).max(1);
        return KernelAction::Quote {
            bid: b,
            ask: a,
            bid_lots: lots(&mut rng),
            ask_lots: lots(&mut rng),
        };
    }
    // Around the touch, sometimes crossing it so that later rows fill.
    let b = if rng.chance(40) {
        ask + rng.below(2)
    } else {
        bid.saturating_sub(rng.below(3)).max(1)
    };
    let a = if rng.chance(40) {
        bid.saturating_sub(rng.below(2)).max(1)
    } else {
        ask + rng.below(3)
    };
    let (b, a) = if b >= a {
        (a.saturating_sub(1).max(1), b + 1)
    } else {
        (b, a)
    };
    KernelAction::Quote {
        bid: b,
        ask: a,
        bid_lots: lots(&mut rng),
        ask_lots: lots(&mut rng),
    }
}

/// What the scripted engine strategy saw and received.
#[derive(Debug, Default)]
pub struct EngineLog {
    /// Per decision: observed tick timestamp and the position passed.
    pub decisions: Vec<(u64, Decimal, Decimal)>,
    /// Delivered reports, in delivery order.
    pub delivered: Vec<ExecutionReport>,
}

/// The engine side of the script.
pub struct Scripted {
    scenario: Scenario,
    k: u64,
    tick: Decimal,
    lot: Decimal,
    pub log: Rc<RefCell<EngineLog>>,
}

impl BacktestStrategy for Scripted {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        let bid = u64::try_from((tick.bid_price / self.tick).normalize().mantissa()).unwrap();
        let ask = u64::try_from((tick.ask_price / self.tick).normalize().mantissa()).unwrap();
        self.log.borrow_mut().decisions.push((
            tick.timestamp,
            position.quantity,
            position.avg_entry_price,
        ));
        let action = self.scenario.action(self.k, bid, ask);
        self.k += 1;
        match action {
            KernelAction::Hold | KernelAction::QuoteError(_) => None,
            KernelAction::Quote {
                bid,
                ask,
                bid_lots,
                ask_lots,
            } => Some(Quote {
                bid_price: Decimal::from(bid) * self.tick,
                bid_size: Decimal::from(bid_lots) * self.lot,
                ask_price: Decimal::from(ask) * self.tick,
                ask_size: Decimal::from(ask_lots) * self.lot,
                timestamp: tick.timestamp,
            }),
        }
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn on_execution_report(&mut self, report: &ExecutionReport) {
        self.log.borrow_mut().delivered.push(report.clone());
    }

    fn reset(&mut self) {}
}

/// An engine run of the scenario.
pub struct EngineRun {
    pub result: BacktestResult,
    pub reports: Vec<ExecutionReport>,
    pub pretrade: Vec<(u64, PretradeDecision, Decimal)>,
    pub state: (InventoryPosition, Decimal, Decimal),
    pub log: EngineLog,
}

pub fn run_engine(sc: &Scenario) -> EngineRun {
    let log = Rc::new(RefCell::new(EngineLog::default()));
    let strategy = Scripted {
        scenario: sc.clone(),
        k: 0,
        tick: sc.cfg.tick_size,
        lot: sc.cfg.lot_size,
        log: Rc::clone(&log),
    };
    let config = BacktestConfig {
        initial_capital: sc.cfg.initial_capital,
        fee_rate: sc.cfg.fee_rate,
        fee_currency: FeeCurrency::Quote,
        tick_size: sc.cfg.tick_size,
        lot_size: sc.cfg.lot_size,
        slippage: SlippageModel::None,
        default_order_size: sc.cfg.lot_size,
        record_equity_curve: true,
        record_trades: true,
    };
    let engine = BacktestEngine::new(config, strategy, VecDataSource::new(sc.ticks.clone()))
        .with_latency(sc.engine_latency())
        .unwrap();
    let mut engine = match &sc.cfg.limits {
        Some(limits) => engine.with_pretrade_policy(LimitsPolicy::new(limits.clone())),
        None => engine,
    };
    let result = engine.run();
    let reports = engine.execution_reports().to_vec();
    let pretrade = engine
        .pretrade_decisions()
        .iter()
        .map(|r| (r.order_id, r.decision.clone(), r.submitted))
        .collect();
    let (position, pnl) = engine.get_state();
    let state = (position.clone(), pnl.realized, pnl.total);
    drop(engine);
    let log = Rc::try_unwrap(log).unwrap().into_inner();
    EngineRun {
        result,
        reports,
        pretrade,
        state,
        log,
    }
}

/// A kernel run: events, per decision observations, actions and states.
pub struct KernelRun {
    pub events: Vec<KernelEvent>,
    /// Number of events before each decision.
    pub marks: Vec<usize>,
    pub observations: Vec<Observation>,
    pub actions: Vec<KernelAction>,
    pub states: Vec<State>,
    pub outcomes: Vec<ActionStatus>,
    pub last: State,
}

pub fn max_lots(sc: &Scenario) -> u64 {
    u64::try_from(
        (sc.cfg.max_order_size / sc.cfg.lot_size)
            .normalize()
            .mantissa(),
    )
    .unwrap()
}

/// Runs the script on the kernel; `keep_states` keeps the state at every
/// decision.
pub fn run_kernel(k: &Kernel<'_>, sc: &Scenario, keep_states: bool) -> KernelRun {
    let mut events = Vec::new();
    let mut s = k.reset_with(sc.latency_seed, &mut events);
    let mut run = KernelRun {
        events: Vec::new(),
        marks: Vec::new(),
        observations: Vec::new(),
        actions: Vec::new(),
        states: Vec::new(),
        outcomes: Vec::new(),
        last: s,
    };
    let mut decision = 0u64;
    while s.phase() == Phase::AwaitDecision {
        let obs = k.observe(&s);
        let action = sc.action(decision, obs.bid, obs.ask);
        if keep_states {
            run.states.push(s);
        }
        run.marks.push(events.len());
        run.observations.push(obs);
        run.actions.push(action);
        let out = k.step_with(&mut s, action, &mut events);
        run.outcomes.push(out.applied);
        decision += 1;
    }
    run.events = events;
    run.last = s;
    run
}

fn kind(kind: ExecKind) -> ExecutionKind {
    match kind {
        ExecKind::Accepted => ExecutionKind::Accepted,
        ExecKind::Rejected => ExecutionKind::Rejected,
        ExecKind::Canceled => ExecutionKind::Canceled,
        ExecKind::CancelRejected => ExecutionKind::CancelRejected,
        ExecKind::PartialFill => ExecutionKind::PartialFill,
        ExecKind::Fill => ExecutionKind::Fill,
    }
}

/// Asserts the kernel report equals the engine report field by field.
pub fn assert_report(k: &Kernel<'_>, ours: &ReportRec, theirs: &ExecutionReport, ctx: &str) {
    assert_eq!(ours.ts_ns, theirs.stamp.timestamp * MS, "{ctx}: ts");
    assert_eq!(ours.seq, theirs.stamp.sequence, "{ctx}: sequence");
    assert_eq!(ours.order_id, theirs.order_id, "{ctx}: order id");
    assert_eq!(ours.side, theirs.side, "{ctx}: side");
    assert_eq!(kind(ours.kind), theirs.kind, "{ctx}: kind");
    assert_eq!(k.price(ours.price).unwrap(), theirs.price, "{ctx}: price");
    assert_eq!(
        k.quantity(ours.last).unwrap(),
        theirs.last_quantity,
        "{ctx}: last qty"
    );
    assert_eq!(
        k.quantity(ours.cum).unwrap(),
        theirs.cumulative_quantity,
        "{ctx}: cum"
    );
    assert_eq!(
        k.quantity(ours.remaining()).unwrap(),
        theirs.remaining,
        "{ctx}: remaining"
    );
    assert_eq!(k.money(ours.fee_u).unwrap(), theirs.fee, "{ctx}: fee");
}

/// Compares a kernel run with the engine run of the same scenario.
pub fn assert_equivalent(k: &Kernel<'_>, sc: &Scenario, kr: &KernelRun, er: &EngineRun, ctx: &str) {
    assert_eq!(
        kr.last.status(),
        Status::Terminal(TermReason::EndOfData),
        "{ctx}: status"
    );
    // 1. Execution reports, in creation order.
    let created: Vec<&ReportRec> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Report(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(created.len(), er.reports.len(), "{ctx}: report count");
    for (i, (ours, theirs)) in created.iter().zip(&er.reports).enumerate() {
        assert_report(k, ours, theirs, &format!("{ctx}: report {i}"));
    }
    // Delivered reports, in delivery order.
    let delivered: Vec<&ReportRec> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Delivered(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(
        delivered.len(),
        er.log.delivered.len(),
        "{ctx}: delivered count"
    );
    for (i, (ours, theirs)) in delivered.iter().zip(&er.log.delivered).enumerate() {
        assert_report(k, ours, theirs, &format!("{ctx}: delivered {i}"));
    }
    // 2. Fills.
    let fills: Vec<_> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Fill(f) => Some(*f),
            _ => None,
        })
        .collect();
    assert_eq!(fills.len(), er.result.trades.len(), "{ctx}: trade count");
    for (f, t) in fills.iter().zip(&er.result.trades) {
        assert_eq!(f.ts_ns, t.timestamp * MS, "{ctx}: trade ts");
        assert_eq!(k.price(f.price).unwrap(), t.price, "{ctx}: trade price");
        assert_eq!(k.quantity(f.qty).unwrap(), t.quantity, "{ctx}: trade qty");
        assert_eq!(f.side, t.side, "{ctx}: trade side");
        assert_eq!(k.money(f.fee_u).unwrap(), t.fee, "{ctx}: trade fee");
        assert_eq!(Some(f.exec_id), t.execution_id, "{ctx}: execution id");
    }
    // 3. Pretrade decisions.
    let gates: Vec<_> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Gate(g) => Some(*g),
            _ => None,
        })
        .collect();
    assert_eq!(gates.len(), er.pretrade.len(), "{ctx}: pretrade count");
    for (g, (id, decision, submitted)) in gates.iter().zip(&er.pretrade) {
        assert_eq!(g.order_id, *id, "{ctx}: pretrade id");
        let matches = match (g.verdict, decision) {
            (GateVerdict::Accepted, PretradeDecision::Accepted) => true,
            (GateVerdict::Reduced, PretradeDecision::Reduced { quantity, .. }) => {
                k.quantity(g.submitted).unwrap() == *quantity
            }
            (GateVerdict::ReducedOffLot, PretradeDecision::Reduced { .. }) => true,
            (GateVerdict::Rejected, PretradeDecision::Rejected { .. }) => true,
            _ => false,
        };
        assert!(matches, "{ctx}: pretrade {g:?} vs {decision:?}");
        assert_eq!(
            k.quantity(g.submitted).unwrap(),
            *submitted,
            "{ctx}: submitted"
        );
    }
    // 4. Venue ledger: equity per row and final state.
    let rows: Vec<_> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::RowEnd {
                ts_ns,
                equity_half_u,
                ..
            } => Some((*ts_ns, *equity_half_u)),
            _ => None,
        })
        .collect();
    assert_eq!(
        rows.len(),
        er.result.equity_curve.len(),
        "{ctx}: equity points"
    );
    for ((ts, eq), (ets, eeq)) in rows.iter().zip(&er.result.equity_curve) {
        assert_eq!(*ts, ets * MS, "{ctx}: equity ts");
        assert_eq!(k.half_money(*eq).unwrap(), *eeq, "{ctx}: equity at {ets}");
    }
    let summary = k.venue_summary(&kr.last).unwrap();
    let (position, realized, total) = &er.state;
    assert_eq!(summary.position, position.quantity, "{ctx}: position");
    assert_eq!(
        summary.avg_entry, position.avg_entry_price,
        "{ctx}: avg entry"
    );
    assert_eq!(summary.realized, *realized, "{ctx}: realized");
    assert_eq!(summary.total_pnl, *total, "{ctx}: total pnl");
    // 5. Agent view at each decision.
    assert_eq!(
        kr.observations.len(),
        er.log.decisions.len(),
        "{ctx}: decisions"
    );
    for (obs, (ts, qty, avg)) in kr.observations.iter().zip(&er.log.decisions) {
        assert_eq!(obs.ts_ns, ts * MS, "{ctx}: observed ts");
        assert_eq!(obs.position, *qty, "{ctx}: agent position");
        assert_eq!(obs.avg_entry, *avg, "{ctx}: agent avg entry");
    }
    // 6. Result.
    let r = &er.result;
    assert_eq!(summary.total_pnl, r.total_pnl, "{ctx}: result total");
    assert_eq!(summary.total_fees, r.total_fees, "{ctx}: result fees");
    assert_eq!(summary.net_pnl, r.net_pnl, "{ctx}: result net");
    assert_eq!(
        kr.last.counters().venue_fills,
        r.num_trades,
        "{ctx}: trades"
    );
    assert_eq!(kr.last.cursor(), r.num_ticks, "{ctx}: ticks");
    assert_eq!(summary.max_position, r.max_position, "{ctx}: max position");
    assert_eq!(summary.position, r.final_position, "{ctx}: final position");
    assert_eq!(summary.max_drawdown, r.max_drawdown, "{ctx}: drawdown");
    if let (Some(first), Some(last)) = (sc.ticks.first(), sc.ticks.last()) {
        assert_eq!(r.start_time, first.timestamp);
        assert_eq!(r.end_time, last.timestamp);
    }
}

/// `cancel_pending` agreement: for every delivered kernel report with the
/// flag set the engine later emits exactly one cancel outcome for the order;
/// with the flag clear, none. A flagged report with no outcome is only
/// allowed when the run ended at the end of the data with venue messages
/// dropped. Returns the number of flagged reports.
pub fn assert_cancel_pending_agreement(kr: &KernelRun, er: &EngineRun, ctx: &str) -> (u64, u64) {
    let dropped = kr
        .events
        .iter()
        .any(|e| matches!(e, KernelEvent::EndOfData { dropped_msgs, .. } if *dropped_msgs > 0));
    let mut flagged = 0;
    let mut dropped_cases = 0;
    let created: Vec<&ReportRec> = kr
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Report(r) => Some(r),
            _ => None,
        })
        .collect();
    for e in &kr.events {
        let KernelEvent::Delivered(r) = e else {
            continue;
        };
        if !matches!(r.kind, ExecKind::Fill | ExecKind::Rejected) {
            continue;
        }
        let at = created
            .iter()
            .position(|c| c.seq == r.seq && c.order_id == r.order_id)
            .unwrap();
        let later = er.reports[at + 1..]
            .iter()
            .filter(|x| {
                x.order_id == r.order_id
                    && matches!(
                        x.kind,
                        ExecutionKind::Canceled | ExecutionKind::CancelRejected
                    )
            })
            .count();
        if r.cancel_pending {
            flagged += 1;
            if later == 0 {
                assert!(
                    dropped,
                    "{ctx}: cancel_pending without outcome and nothing dropped: {r:?}"
                );
                dropped_cases += 1;
            } else {
                assert_eq!(
                    later, 1,
                    "{ctx}: cancel_pending outcomes for {}",
                    r.order_id
                );
            }
        } else {
            assert_eq!(
                later, 0,
                "{ctx}: outcome without cancel_pending for {}",
                r.order_id
            );
        }
    }
    (flagged, dropped_cases)
}

/// Builds the kernel of a scenario, asserting `new()` accepts it.
pub fn kernel<'d>(sc: &Scenario, data: &'d KernelDataset) -> Kernel<'d> {
    match Kernel::new(&sc.cfg, data) {
        Ok(k) => k,
        Err(e) => panic!(
            "new() rejected the scenario: {e}; need {:?}",
            market_maker_rs::backtest::kernel::capacity_need(&sc.cfg, data)
        ),
    }
}

/// Runs `actions` on the kernel until decision `upto` (or the end) and
/// returns the state there, the delivered reports so far and the outcomes.
pub fn run_prefix(
    k: &Kernel<'_>,
    seed: u64,
    actions: &[KernelAction],
    upto: usize,
) -> (State, Vec<ReportRec>, Vec<ActionStatus>) {
    let mut events = Vec::new();
    let mut s = k.reset_with(seed, &mut events);
    let mut outcomes = Vec::new();
    for action in actions.iter().take(upto) {
        if s.phase() != Phase::AwaitDecision {
            break;
        }
        outcomes.push(k.step_with(&mut s, *action, &mut events).applied);
    }
    let delivered = events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Delivered(r) => Some(*r),
            _ => None,
        })
        .collect();
    (s, delivered, outcomes)
}

/// Decision times, from the `Observed` events.
pub fn decision_times(events: &[KernelEvent]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Observed { at_ns, .. } => Some(*at_ns),
            _ => None,
        })
        .collect()
}

/// Replaces the prices of the rows selected by `pick` with random grid data.
pub fn perturb(
    sc: &Scenario,
    rng: &mut SplitMix64,
    pick: impl Fn(usize, &MarketTick) -> bool,
) -> (Scenario, usize) {
    let mut out = sc.clone();
    let mut changed = 0;
    for (i, t) in out.ticks.iter_mut().enumerate() {
        if !pick(i, &sc.ticks[i]) {
            continue;
        }
        let tick = sc.cfg.tick_size;
        let mid = (t.bid_price / tick).normalize().mantissa() + i128::from(rng.below(9)) - 4;
        let half = 1 + i128::from(rng.below(3));
        let bid = (mid - half).max(1);
        t.bid_price = Decimal::from(bid) * tick;
        t.ask_price = Decimal::from(bid + 2 * half) * tick;
        changed += 1;
    }
    (out, changed)
}
