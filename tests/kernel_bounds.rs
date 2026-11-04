//! Bounds, capacity, mask, look-ahead fixtures, decode and dataset checks
//! of the financial kernel.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

/// Deliveries capacity of the harness kernels.
const HARNESS_DLV: usize = 192;

#[path = "support/kernel_harness.rs"]
mod harness;

use harness::*;
use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionGrid, ActionStatus, ExecKind, FinancialKernel, KernelAction, KernelConfig, KernelDataset,
    KernelEvent, KernelLatency, KernelTick, Phase, RejectReason, ReportRec, Status, TermReason,
    capacity_need, decode, encode,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestStrategy, MarketTick, SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};

type Small<'d> = FinancialKernel<'d>;

fn empty(cfg: &KernelConfig) -> KernelDataset {
    KernelDataset::from_kernel_ticks(Vec::new(), cfg).unwrap()
}

fn build(cfg: &KernelConfig) -> MMResult<u128> {
    let data = empty(cfg);
    let k: MMResult<Small<'_>> = FinancialKernel::new(cfg, &data);
    k.map(|k| k.bound())
}

fn cfg(
    tick: Decimal,
    lot: Decimal,
    fee: Decimal,
    steps: u64,
    max_price: Decimal,
    max_order: Decimal,
    c0: Decimal,
) -> KernelConfig {
    KernelConfig::new(tick, lot, max_price, max_order, steps)
        .with_fee_rate(fee)
        .with_initial_capital(c0)
}

fn close(value: u128, expected: f64) -> bool {
    let v = value as f64;
    (v - expected).abs() <= expected * 0.001
}

#[test]
fn test_kernel_bounds_table_examples_accept_and_reject() {
    let rows: [(KernelConfig, Option<f64>); 7] = [
        (
            cfg(
                dec!(0.01),
                dec!(0.0001),
                dec!(0.0001),
                10_000_000,
                dec!(100000),
                dec!(100),
                dec!(1000000),
            ),
            Some(4.0e24),
        ),
        (
            cfg(
                dec!(0.01),
                dec!(1),
                dec!(0.0001),
                1_000_000,
                dec!(10000),
                dec!(100000),
                dec!(10000000),
            ),
            Some(4.0e21),
        ),
        (
            cfg(
                dec!(0.01),
                dec!(0.0001),
                dec!(0.00001),
                100_000_000,
                dec!(10000),
                dec!(10),
                dec!(1000000),
            )
            .with_limits(RiskLimits::new(dec!(100), Decimal::MAX, dec!(0)).unwrap()),
            Some(2.0e24),
        ),
        (
            cfg(
                dec!(0.5),
                dec!(0.001),
                dec!(0),
                1_000_000,
                dec!(50000000),
                dec!(100000),
                dec!(1000),
            ),
            Some(2.0e23),
        ),
        (
            cfg(
                dec!(0.000001),
                dec!(0.00000001),
                dec!(0.00025),
                1_000_000,
                dec!(100000),
                dec!(10),
                dec!(1000000),
            ),
            None,
        ),
        (
            cfg(
                dec!(0.00000001),
                dec!(0.00000001),
                dec!(0.000000000001),
                1_000,
                dec!(100),
                dec!(1),
                dec!(10000),
            ),
            None,
        ),
        (
            cfg(
                dec!(1),
                dec!(1),
                dec!(0),
                1_000_000_000,
                dec!(1000000000),
                dec!(1000000000),
                dec!(0),
            ),
            None,
        ),
    ];
    for (i, (config, expected)) in rows.iter().enumerate() {
        match (build(config), expected) {
            (Ok(b), Some(e)) => assert!(close(b, *e), "row {i}: B = {b}, expected {e}"),
            (Err(_), None) => {}
            (got, want) => panic!("row {i}: got {got:?}, expected {want:?}"),
        }
    }
    // Money scales of the accepted rows.
    let scales = [10u32, 6, 11, 4];
    for (i, s) in scales.iter().enumerate() {
        let data = empty(&rows[i].0);
        let k: Small<'_> = FinancialKernel::new(&rows[i].0, &data).unwrap();
        assert_eq!(k.money_scale(), *s, "row {i}");
    }
}

#[test]
fn test_kernel_bounds_mantissa_example_notional_is_125_units() {
    // tick 0.25, lot 0.5, fs = 0: one lot at one tick is 0.125 = 125 U at s = 3.
    let config = KernelConfig::new(dec!(0.25), dec!(0.5), dec!(10), dec!(5), 10);
    let ticks = vec![
        MarketTick::new(1, dec!(0.25), dec!(0.5), dec!(0.5), dec!(0.5)),
        MarketTick::new(2, dec!(0.25), dec!(0.5), dec!(0.5), dec!(0.5)),
    ];
    let data = KernelDataset::from_ticks(&ticks, &config).unwrap();
    let k: Small<'_> = FinancialKernel::new(&config, &data).unwrap();
    assert_eq!(k.money_scale(), 3);
    assert_eq!(k.money(125), Some(dec!(0.125)));
    let mut s = k.reset(0);
    // Buy one lot at 2 ticks (0.5): crosses the ask of the next row.
    k.step(
        &mut s,
        KernelAction::Quote {
            bid: 2,
            ask: 3,
            bid_lots: 1,
            ask_lots: 1,
        },
    );
    while s.phase() == Phase::AwaitDecision {
        k.step(&mut s, KernelAction::Hold);
    }
    assert_eq!(s.venue_realized(), dec!(-0.25));
}

#[test]
fn test_kernel_bounds_boundary_factor_twenty_accepts_largest_b() {
    // tick = lot = 1, fs = 0, P = Q = S = 1: B = C0 + 4.
    let limit = ((1u128 << 96) - 1) / 20;
    let at = |c0: u128, steps: u64| {
        cfg(
            dec!(1),
            dec!(1),
            dec!(0),
            steps,
            dec!(1),
            dec!(1),
            Decimal::from_i128_with_scale(i128::try_from(c0).unwrap(), 0),
        )
    };
    assert_eq!(build(&at(limit - 4, 1)).unwrap(), limit);
    assert!(build(&at(limit - 3, 1)).is_err(), "C0 + 1 U");
    assert!(build(&at(limit - 4, 2)).is_err(), "S + 1");
}

#[test]
fn test_kernel_bounds_boundary_factor_two_accepts_largest_b() {
    // tick = lot = 1, fee 0.1 (fs = 1, U = 0.1), P = Q = S = 1: B = C0 + 42.
    let limit = ((1u128 << 96) - 1) / 2;
    let at = |c0: u128, steps: u64| {
        cfg(
            dec!(1),
            dec!(1),
            dec!(0.1),
            steps,
            dec!(1),
            dec!(1),
            Decimal::try_from_i128_with_scale(i128::try_from(c0).unwrap(), 1).unwrap(),
        )
    };
    assert_eq!(build(&at(limit - 42, 1)).unwrap(), limit);
    assert!(build(&at(limit - 41, 1)).is_err(), "C0 + 1 U");
    assert!(build(&at(limit - 42, 2)).is_err(), "S + 1");
}

/// Runs the most fills the truncation rule allows, `2S - 2` fills of `Q`
/// lots next to `P` (both sides of every quote but the last), with `B`
/// exactly at the limit for `fee` (factor 20 when the fee scale is 0, else
/// 2), and checks every report and ledger value against the engine.
///
/// `2S` fills, the bound's worst case, are not reachable: the run stops at
/// the end of the row in which decision `S` is taken (end of the data or
/// truncation at `max_steps`), and an order only fills on a later row, so
/// the last quote never fills. `2S - 2` is the reachable maximum under
/// the specified truncation semantics, and the `2S` in `B` remains a valid
/// upper bound.
fn boundary_run(fee: Decimal, fee_num: u128, fs: u32) {
    let (p, q, steps) = (1_000u64, 1_000u64, 40u64);
    let f = 10u128.pow(fs);
    let n = u128::from(p * q) * f;
    let fee_u = u128::from(p * q) * fee_num;
    let x = 2 * u128::from(steps * q);
    let rest = 2 * u128::from(steps) * (n + fee_u) + x * u128::from(p) * f;
    let factor = if fs == 0 { 20 } else { 2 };
    let limit = ((1u128 << 96) - 1) / factor;
    let c0 = i128::try_from(limit - rest).unwrap();
    let config = cfg(
        dec!(1),
        dec!(1),
        fee,
        steps,
        Decimal::from(p),
        Decimal::from(q),
        Decimal::try_from_i128_with_scale(c0, fs).unwrap(),
    )
    .with_latency(KernelLatency::zero(MS).with_delays(3, 0, 5, 0));
    // Rows alternate (1, 2), which fills a buy at P - 2, and (P - 1, P),
    // which fills a sell at P - 1 (the highest a sell can fill, as every
    // ask is at most P). The gap before row S - 1 leaves the second to last
    // quote two rows before the last decision.
    let rows: Vec<(u64, u64, u64)> = (0..steps + 3)
        .map(|i| {
            let ts = 1_000 + i + if i >= steps - 1 { 2 } else { 0 };
            if i % 2 == 0 {
                (ts, 1, 2)
            } else {
                (ts, p - 1, p)
            }
        })
        .collect();
    let ticks = rows
        .iter()
        .map(|(ts, b, a)| {
            MarketTick::new(*ts, Decimal::from(*b), dec!(1), Decimal::from(*a), dec!(1))
        })
        .collect();
    let actions = (0..steps)
        .map(|_| KernelAction::Quote {
            bid: p - 2,
            ask: p - 1,
            bid_lots: q,
            ask_lots: q,
        })
        .collect();
    let sc = Scenario::fixed(config, ticks, actions);
    let data = sc.data();
    let k = kernel(&sc, &data);
    assert_eq!(k.bound(), limit, "B at the factor {factor} limit");
    let kr = run_kernel(&k, &sc, false);
    assert_eq!(kr.last.status(), Status::Terminal(TermReason::EndOfData));
    assert_eq!(kr.observations.len(), usize::try_from(steps).unwrap());
    assert_eq!(
        kr.last.counters().venue_fills,
        2 * steps - 2,
        "2S - 2 fills"
    );
    assert_eq!(kr.last.counters().halts, 0);
    let er = run_engine(&sc);
    assert_equivalent(&k, &sc, &kr, &er, &format!("boundary fs {fs}"));
}

#[test]
fn test_kernel_bounds_fills_at_max_price_and_size_never_halt() {
    boundary_run(dec!(0.001), 1, 3);
}

#[test]
fn test_kernel_bounds_factor_twenty_boundary_full_run_never_halts() {
    boundary_run(dec!(0), 0, 0);
}

#[test]
fn test_kernel_bounds_finest_grid_config_rejected_and_dropped_fee_probe() {
    let finest = cfg(
        dec!(0.00000001),
        dec!(0.00000001),
        dec!(0.000000000001),
        10,
        dec!(1),
        dec!(1),
        dec!(10000),
    );
    assert!(matches!(
        build(&finest),
        Err(MMError::InvalidConfiguration(_))
    ));
    // Regression probe: outside the bounds a Decimal subtraction silently
    // drops the 1e-28 fee, which is why this grid is rejected.
    let fee = dec!(0.00000001) * dec!(0.00000001) * dec!(0.000000000001);
    assert_eq!(fee, Decimal::new(1, 28));
    assert_eq!(dec!(10000) - fee, dec!(10000));
}

#[test]
fn test_kernel_bounds_mid2_width_overflow_rejected() {
    // P = 1e19 fits u64 and 20B fits 96 bits, but 2P - 1 does not fit u64.
    let config = cfg(
        Decimal::new(1, 20),
        dec!(0.1),
        dec!(0),
        1,
        dec!(0.1),
        dec!(0.1),
        dec!(0),
    );
    let err = build(&config).unwrap_err();
    assert!(err.to_string().contains("2 * max_price"), "{err}");
}

#[test]
fn test_kernel_dataset_out_of_order_rejected_while_engine_clamps() {
    let config = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 10);
    let ticks = vec![
        MarketTick::new(1000, dec!(98), dec!(1), dec!(102), dec!(1)),
        MarketTick::new(1002, dec!(98), dec!(1), dec!(102), dec!(1)),
        MarketTick::new(1001, dec!(98), dec!(1), dec!(102), dec!(1)),
    ];
    assert!(matches!(
        KernelDataset::from_ticks(&ticks, &config),
        Err(MMError::InvalidMarketState(_))
    ));
    struct Passive;
    impl BacktestStrategy for Passive {
        fn on_tick(&mut self, _: &MarketTick, _: &InventoryPosition) -> Option<Quote> {
            None
        }
        fn on_fill(&mut self, _: &SimulatedFill) {}
        fn reset(&mut self) {}
    }
    let mut engine = BacktestEngine::new(
        BacktestConfig::default(),
        Passive,
        VecDataSource::new(ticks),
    );
    assert_eq!(
        engine.run().num_ticks,
        3,
        "the engine clamps instead (documented divergence)"
    );
}

#[test]
fn test_kernel_dataset_invalid_rows_rejected() {
    let config = KernelConfig::new(dec!(0.5), dec!(1), dec!(100), dec!(10), 10);
    let bad = [
        MarketTick::new(1, dec!(10.25), dec!(1), dec!(11), dec!(1)),
        MarketTick::new(1, dec!(10), dec!(1.5), dec!(11), dec!(1)),
        MarketTick::new(1, dec!(11), dec!(1), dec!(11), dec!(1)),
        MarketTick::new(1, dec!(0), dec!(1), dec!(11), dec!(1)),
        MarketTick::new(1, dec!(10), dec!(1), dec!(100.5), dec!(1)),
    ];
    for tick in bad {
        assert!(
            KernelDataset::from_ticks(std::slice::from_ref(&tick), &config).is_err(),
            "{tick:?}"
        );
    }
}

#[test]
fn test_kernel_new_capacity_checks_reject_small_capacities() {
    // Ten rows at one timestamp, zero latency: D = 10 so ORD >= 24.
    let config = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100);
    let rows = (0..10)
        .map(|_| KernelTick {
            ts_ns: MS,
            bid: 98,
            bid_lots: 1,
            ask: 102,
            ask_lots: 1,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &config).unwrap();
    let need = capacity_need(&config, &data).unwrap();
    assert_eq!(need.ord, 24);
    let small: MMResult<FinancialKernel<'_>> = FinancialKernel::new(&config, &data);
    assert!(small.is_err());
    let fits: MMResult<FinancialKernel<'_, 24, 48, 72, 4>> = FinancialKernel::new(&config, &data);
    assert!(fits.is_ok());
    // Observations in flight: rows 1 ms apart with a 5 ms delay.
    let lagged = config
        .clone()
        .with_latency(KernelLatency::zero(MS).with_delays(5, 0, 0, 0));
    let rows = (0..10)
        .map(|i| KernelTick {
            ts_ns: (i + 1) * MS,
            bid: 98,
            bid_lots: 1,
            ask: 102,
            ask_lots: 1,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &lagged).unwrap();
    assert_eq!(capacity_need(&lagged, &data).unwrap().obs, 5);
    let tight: MMResult<FinancialKernel<'_, 64, 128, 192, 4>> =
        FinancialKernel::new(&lagged, &data);
    assert!(tight.is_err());
    let enough: MMResult<FinancialKernel<'_, 64, 128, 192, 5>> =
        FinancialKernel::new(&lagged, &data);
    assert!(enough.is_ok());
}

/// Observation burst fixture: heavy observation jitter clamps a backlog
/// that pops inside one window, every decision quotes, with worst-case
/// cancel and ack delays.
fn burst() -> (KernelConfig, KernelDataset) {
    let config = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 1_000).with_latency(
        KernelLatency::zero(MS)
            .with_delays(0, 1, 4, 3)
            .with_jitter(6),
    );
    let rows = (0..60)
        .map(|i| KernelTick {
            ts_ns: (1_000 + 2 * i) * MS,
            bid: 98 + i % 3,
            bid_lots: 1,
            ask: 103 - i % 2,
            ask_lots: 1,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &config).unwrap();
    (config, data)
}

#[test]
fn test_kernel_observation_burst_with_required_capacities_has_no_capacity_reject() {
    let (config, data) = burst();
    let need = capacity_need(&config, &data).unwrap();
    assert_eq!((need.ord, need.obs), (50, 3), "update the generics below");
    let k: FinancialKernel<'_, 50, 100, 150, 3> = FinancialKernel::new(&config, &data).unwrap();
    for seed in 0..200 {
        let mut s = k.reset(seed);
        let mut i = 0u64;
        let mut times = Vec::new();
        let mut events = Vec::new();
        while s.phase() == Phase::AwaitDecision {
            let obs = k.observe(&s);
            let action = KernelAction::Quote {
                bid: obs.bid - i % 2,
                ask: obs.ask + i % 3,
                bid_lots: 1,
                ask_lots: 1,
            };
            let out = k.step_with(&mut s, action, &mut events);
            assert_eq!(
                out.applied,
                ActionStatus::Applied,
                "seed {seed} decision {i}"
            );
            times.push(s.now_ns());
            i += 1;
        }
        assert_eq!(s.status(), Status::Terminal(TermReason::EndOfData));
        // The backlog popped as bursts: several decisions at one time.
        let bursts = decision_times(&events)
            .windows(2)
            .filter(|w| w[0] == w[1])
            .count();
        assert!(bursts > 0 || seed > 0);
    }
}

/// Engine-checked fixture with fixed actions.
fn fixture(
    latency: KernelLatency,
    rows: &[(u64, u64, u64)],
    actions: Vec<KernelAction>,
    limits: Option<RiskLimits>,
) -> (Scenario, KernelDataset) {
    let mut config = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100)
        .with_initial_capital(dec!(10000))
        .with_fee_rate(dec!(0.001))
        .with_latency(latency);
    config.limits = limits;
    let ticks = rows
        .iter()
        .map(|(ts, bid, ask)| {
            MarketTick::new(
                *ts,
                Decimal::from(*bid),
                dec!(1),
                Decimal::from(*ask),
                dec!(1),
            )
        })
        .collect();
    let sc = Scenario::fixed(config, ticks, actions);
    let data = sc.data();
    (sc, data)
}

fn quote(bid: u64, ask: u64) -> KernelAction {
    KernelAction::Quote {
        bid,
        ask,
        bid_lots: 1,
        ask_lots: 1,
    }
}

fn order_reports(events: &[KernelEvent], id: u64) -> Vec<ReportRec> {
    events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Delivered(r) if r.order_id == id => Some(*r),
            _ => None,
        })
        .collect()
}

/// Runs a fixture on the kernel and the engine, asserting equivalence, and
/// checks that order `id` is open in the account exactly until its last
/// report is delivered.
fn run_fixture(sc: &Scenario, data: &KernelDataset, id: u64) -> Vec<(u64, ExecKind, bool)> {
    let k = kernel(sc, data);
    let kr = run_kernel(&k, sc, true);
    let er = run_engine(sc);
    assert_equivalent(&k, sc, &kr, &er, "fixture");
    assert_cancel_pending_agreement(&kr, &er, "fixture");
    let reports = order_reports(&kr.events, id);
    let last_seq = reports.last().unwrap().seq;
    for (j, s) in kr.states.iter().enumerate() {
        let delivered = kr.events[..kr.marks[j]].iter().any(
            |e| matches!(e, KernelEvent::Delivered(r) if r.order_id == id && r.seq == last_seq),
        );
        let submitted = s.next_order_id() > id;
        assert_eq!(
            s.is_open(id),
            submitted && !delivered,
            "order {id} must be open exactly until its last report (decision {j})"
        );
        let q = s.queue_lengths();
        assert!(
            q.orders + q.awaiting <= q.held && q.msgs <= 2 * q.held && q.deliveries <= 3 * q.held
        );
    }
    reports
        .iter()
        .map(|r| (r.ts_ns / MS, r.kind, r.cancel_pending))
        .collect()
}

fn race_rows() -> Vec<(u64, u64, u64)> {
    // The row at 1003 crosses a resting bid at 99.
    (0..10)
        .map(|i| {
            if 1000 + i == 1003 {
                (1003, 97, 99)
            } else {
                (1000 + i, 98, 102)
            }
        })
        .collect()
}

#[test]
fn test_kernel_race_cancel_at_fill_time_cancel_wins() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 2, 0);
    let (sc, data) = fixture(
        latency,
        &race_rows(),
        vec![quote(99, 103), quote(90, 110)],
        None,
    );
    assert_eq!(
        run_fixture(&sc, &data, 1),
        vec![
            (1000, ExecKind::Accepted, false),
            (1003, ExecKind::Canceled, false)
        ]
    );
}

#[test]
fn test_kernel_race_fill_before_cancel_reports_cancel_pending() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 3, 0);
    let (sc, data) = fixture(
        latency,
        &race_rows(),
        vec![quote(99, 103), quote(90, 110)],
        None,
    );
    assert_eq!(
        run_fixture(&sc, &data, 1),
        vec![
            (1000, ExecKind::Accepted, false),
            (1003, ExecKind::Fill, true),
            (1004, ExecKind::CancelRejected, false)
        ]
    );
}

#[test]
fn test_kernel_race_fill_without_cancel_closes_on_fill() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 3, 0);
    let (sc, data) = fixture(latency, &race_rows(), vec![quote(99, 103)], None);
    assert_eq!(
        run_fixture(&sc, &data, 1),
        vec![
            (1000, ExecKind::Accepted, false),
            (1003, ExecKind::Fill, false)
        ]
    );
}

#[test]
fn test_kernel_fill_delivered_late_updates_venue_then_agent() {
    // Fill at 1003, ack 5 ms: delivered at 1008, after several rows.
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 5);
    let (sc, data) = fixture(latency, &race_rows(), vec![quote(99, 103)], None);
    let k = kernel(&sc, &data);
    let kr = run_kernel(&k, &sc, true);
    let er = run_engine(&sc);
    assert_equivalent(&k, &sc, &kr, &er, "late fill");
    let times = decision_times(&kr.events);
    let mut seen_gap = false;
    for (s, t) in kr.states.iter().zip(&times) {
        let ms = t / MS;
        let obs = k.observe(s);
        if (1004..1008).contains(&ms) {
            assert_eq!(s.venue_position(), dec!(1), "venue books the fill at 1003");
            assert_eq!(obs.position, dec!(0), "the agent learns it at 1008");
            seen_gap = true;
        }
        if ms >= 1008 {
            assert_eq!(obs.position, dec!(1));
        }
    }
    assert!(seen_gap);
}

#[test]
fn test_kernel_gate_rejects_on_undelivered_fill_and_view_waits_for_report() {
    let latency = KernelLatency::zero(MS).with_delays(0, 0, 0, 5);
    let limits = RiskLimits::new(dec!(1), Decimal::MAX, dec!(0)).unwrap();
    let actions = vec![
        quote(99, 110),
        KernelAction::Hold,
        KernelAction::Hold,
        KernelAction::Hold,
        quote(99, 111),
    ];
    let (sc, data) = fixture(latency, &race_rows(), actions, Some(limits));
    let k = kernel(&sc, &data);
    let kr = run_kernel(&k, &sc, true);
    let er = run_engine(&sc);
    assert_equivalent(&k, &sc, &kr, &er, "gate on undelivered fill");
    // Decision 4 at 1004: the fill of 1003 is not delivered, the gate
    // refuses the bid (order 3) locally.
    let reject = kr
        .events
        .iter()
        .find_map(|e| match e {
            KernelEvent::Report(r) if r.order_id == 3 => Some(*r),
            _ => None,
        })
        .unwrap();
    assert_eq!(reject.kind, ExecKind::Rejected);
    assert_eq!(reject.ts_ns / MS, 1004);
    let times = decision_times(&kr.events);
    for (j, (s, t)) in kr.states.iter().zip(&times).enumerate() {
        let obs = k.observe(s);
        if j > 4 && t / MS < 1009 {
            assert_eq!(
                obs.delivered_local_rejects,
                0,
                "reject not yet delivered at {}",
                t / MS
            );
            assert_eq!(obs.position, if t / MS >= 1008 { dec!(1) } else { dec!(0) });
        }
        if t / MS >= 1009 {
            assert_eq!(obs.delivered_local_rejects, 1);
        }
    }
}

fn grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 4], [0, 2, 3], [1, 2]).unwrap()
}

/// 4 x 4 offsets and 4 sizes: 256 quotes, some out of range.
fn big_grid() -> ActionGrid<4, 4> {
    ActionGrid::new([0, 1, 4, 9_000], [0, 2, 3, 12_000], [1, 2, 10, 11]).unwrap()
}

fn admissions(k: &Kernel<'_>, s: &State) -> Vec<ActionStatus> {
    admissions_on(k, s, &grid())
}

fn admissions_on<const K: usize, const M: usize>(
    k: &Kernel<'_>,
    s: &State,
    g: &ActionGrid<K, M>,
) -> Vec<ActionStatus> {
    let mut out = Vec::new();
    for b in 0..K {
        for a in 0..K {
            for x in 0..M {
                for y in 0..M {
                    let mut c = *s;
                    out.push(match k.grid_action(s, g, (b, a, x, y)) {
                        Some(action) => k.step(&mut c, action).applied,
                        None => ActionStatus::Rejected(RejectReason::PriceRange),
                    });
                }
            }
        }
    }
    out
}

#[test]
fn test_kernel_hidden_progress_leaves_mask_and_admission_identical() {
    // Same enriched delivered history, different hidden venue progress: an
    // undelivered fill vs none, an applied vs a pending activation, an
    // effective vs a pending cancel.
    let mut kinds = [0u64; 3];
    for seed in 0..40 {
        for index in 0..DATASETS {
            let mut base = scenario(seed, index);
            base.cfg.latency.jitter_q = 0;
            base.cfg.latency.ack_q += 3;
            let mut fill_variant = base.clone();
            fill_variant.ticks = perturb(&base, &mut SplitMix64(seed ^ index), |i, _| i % 2 == 1)
                .0
                .ticks;
            let mut submit_variant = base.clone();
            submit_variant.cfg.latency.submit_q += 1;
            let mut cancel_variant = base.clone();
            cancel_variant.cfg.latency.cancel_q += 1;
            let data = base.data();
            let k = kernel(&base, &data);
            let kr = run_kernel(&k, &base, true);
            for (kind, variant) in [fill_variant, submit_variant, cancel_variant]
                .iter()
                .enumerate()
            {
                let vdata = variant.data();
                let vk = kernel(variant, &vdata);
                for j in 0..kr.states.len() {
                    let (a, ha, oa) = run_prefix(&k, base.latency_seed, &kr.actions, j);
                    let (b, hb, ob) = run_prefix(&vk, variant.latency_seed, &kr.actions, j);
                    if b.phase() != Phase::AwaitDecision
                        || enriched(&ha) != enriched(&hb)
                        || oa != ob
                    {
                        continue;
                    }
                    // Same observed rows: every row up to the observed one.
                    let row = usize::try_from(k.observe(&a).row.unwrap()).unwrap();
                    if base.ticks[..=row] != variant.ticks[..=row] {
                        continue;
                    }
                    let hidden = a.next_exec_id() != b.next_exec_id()
                        || a.stamp_seq() != b.stamp_seq()
                        || a.queue_lengths() != b.queue_lengths();
                    if !hidden {
                        continue;
                    }
                    assert_eq!(
                        k.mask(&a, &grid()).to_bytes(),
                        vk.mask(&b, &grid()).to_bytes()
                    );
                    assert_eq!(k.observe(&a).to_bytes(), vk.observe(&b).to_bytes());
                    assert_eq!(admissions(&k, &a), admissions(&vk, &b));
                    kinds[kind] += 1;
                }
            }
        }
    }
    for (i, count) in kinds.iter().enumerate() {
        assert!(*count > 0, "hidden progress kind {i} never reached");
    }
}

#[test]
fn test_kernel_mask_exactness_full_grid_matches_admission() {
    // Every one of the 256 grid quotes at every decision.
    let mut admitted = [0u64; 2];
    for seed in 0..5 {
        for index in 0..DATASETS {
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            let g = big_grid();
            for s in &kr.states {
                let mask = k.mask(s, &g);
                let all = admissions_on(&k, s, &g);
                assert_eq!(all.len(), 256);
                let mut i = 0;
                for b in 0..4 {
                    for a in 0..4 {
                        for x in 0..4 {
                            for y in 0..4 {
                                let applied = all[i] == ActionStatus::Applied;
                                assert_eq!(applied, mask.allows(b, a, x, y));
                                admitted[usize::from(applied)] += 1;
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(admitted[0] > 0 && admitted[1] > 0, "{admitted:?}");
}

#[test]
fn test_kernel_decode_rejects_tampered_states() {
    let sc = scenario(3, 4);
    let data = sc.data();
    let k = kernel(&sc, &data);
    let kr = run_kernel(&k, &sc, true);
    let s = kr.states[kr.states.len() / 2];
    let bytes = encode(&s);
    assert_eq!(decode(&bytes, &k).unwrap(), s);
    let mut tampered = Vec::new();
    let mut magic = bytes.clone();
    magic[0] = b'X';
    tampered.push(magic);
    let mut version = bytes.clone();
    version[4] = 9;
    tampered.push(version);
    let mut capacity = bytes.clone();
    capacity[8] ^= 1;
    tampered.push(capacity);
    tampered.push(bytes[..bytes.len() - 1].to_vec());
    let mut trailing = bytes.clone();
    trailing.push(0);
    tampered.push(trailing);
    // Every single byte flip in the counters tail breaks a consistency check
    // (actions equal steps, rejects bounded by steps, fills equal exec ids).
    let n = bytes.len();
    let mut actions = bytes.clone();
    actions[n - 120] ^= 1;
    tampered.push(actions);
    let mut fills = bytes.clone();
    fills[n - 16] ^= 1;
    tampered.push(fills);
    for (i, t) in tampered.iter().enumerate() {
        assert!(decode(t, &k).is_err(), "tampered state {i} decoded");
    }
    // A state from another dataset.
    let other = scenario(3, 5);
    let odata = other.data();
    let ok = kernel(&other, &odata);
    assert!(decode(&bytes, &ok).is_err());
    // Random flips never panic.
    let mut rng = SplitMix64(7);
    for _ in 0..2_000 {
        let mut b = bytes.clone();
        let i = usize::try_from(rng.below(n as u64)).unwrap();
        b[i] ^= 1 << rng.below(8);
        let _ = decode(&b, &k);
    }
}

#[test]
fn test_kernel_step_after_termination_is_idempotent_and_uncounted() {
    let sc = scenario(5, 2);
    let data = sc.data();
    let k = kernel(&sc, &data);
    let mut s = run_kernel(&k, &sc, false).last;
    assert!(!s.status().is_running());
    let bytes = encode(&s);
    let (steps, actions, rejected) = (s.step_count(), s.counters().actions, s.counters().rejected);
    for i in 0..1000u64 {
        let action = if i % 2 == 0 {
            KernelAction::Hold
        } else {
            quote(10, 20)
        };
        let out = k.step(&mut s, action);
        assert_eq!(out.applied, ActionStatus::Rejected(RejectReason::Terminal));
        assert_eq!(out.events, 0);
    }
    assert_eq!(encode(&s), bytes, "terminal state must stay byte-identical");
    assert_eq!(
        (s.step_count(), s.counters().actions, s.counters().rejected),
        (steps, actions, rejected)
    );
}

/// Byte offsets of the sequence fields in a state encoding (see the field
/// order table of `encode`): after the header, capacities, dataset id,
/// fingerprint, phases, cursor, current row, now, step, status and the two
/// random streams.
fn sequence_offset(bytes: &[u8]) -> usize {
    let cur_row_tag = 4 + 2 + 1 + 1 + 16 + 74 + 16 + 1 + 1 + 8;
    let cur_row = if bytes[cur_row_tag] == 1 { 9 } else { 1 };
    cur_row_tag + cur_row + 8 + 8 + 3 + 32
}

fn patch(bytes: &[u8], at: usize, value: u64) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
    out
}

#[test]
fn test_kernel_decode_public_counter_tamper_rejected() {
    for (seed, index) in [(0, 0), (3, 4), (7, 11)] {
        let sc = scenario(seed, index);
        let data = sc.data();
        let k = kernel(&sc, &data);
        let kr = run_kernel(&k, &sc, true);
        // The reset state (first decision) and a mid-run state.
        for s in [kr.states[0], kr.states[kr.states.len() / 2]] {
            let bytes = encode(&s);
            assert_eq!(decode(&bytes, &k).unwrap(), s);
            let seq = sequence_offset(&bytes);
            let n = bytes.len();
            // stamp_seq, next_msg_seq, next_order_id, next_exec_id,
            // last_delivery_ns, last_observed_ns.
            let mut cases: Vec<(String, Vec<u8>)> = (0..6)
                .map(|i| {
                    (
                        format!("sequence field {i}"),
                        patch(&bytes, seq + 8 * i, u64::MAX),
                    )
                })
                .collect();
            cases.push((
                "stamp_seq + 1e6".into(),
                patch(&bytes, seq, s.stamp_seq() + 1_000_000),
            ));
            cases.push((
                "next_exec_id + 1".into(),
                patch(&bytes, seq + 24, s.next_exec_id() + 1),
            ));
            // Counters: actions, then after the rejects halts, the two
            // drop counters, quote errors, venue fills and local rejects.
            cases.push(("actions".into(), patch(&bytes, n - 120, u64::MAX)));
            for (i, name) in [
                "halts",
                "eof msgs",
                "eof obs",
                "quote errors",
                "venue fills",
                "local rejects",
            ]
            .iter()
            .enumerate()
            {
                cases.push(((*name).into(), patch(&bytes, n - 48 + 8 * i, u64::MAX)));
            }
            cases.push((
                "rejected[PriceRange]".into(),
                patch(&bytes, n - 112 + 8, u64::MAX),
            ));
            for (name, tampered) in cases {
                assert!(
                    decode(&tampered, &k).is_err(),
                    "seed {seed} dataset {index}: {name} decoded"
                );
            }
        }
    }
}

#[test]
fn test_kernel_decode_reset_state_with_max_stamp_seq_rejected() {
    // The reported repro: only stamp_seq set to u64::MAX on a reset state.
    let config = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 100);
    let rows = (0..4)
        .map(|i| KernelTick {
            ts_ns: (1 + i) * MS,
            bid: 98,
            bid_lots: 1,
            ask: 102,
            ask_lots: 1,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &config).unwrap();
    let k: Small<'_> = FinancialKernel::new(&config, &data).unwrap();
    let s = k.reset(0);
    let bytes = encode(&s);
    assert_eq!(
        &bytes[sequence_offset(&bytes)..sequence_offset(&bytes) + 8],
        &s.stamp_seq().to_le_bytes()
    );
    assert!(decode(&patch(&bytes, sequence_offset(&bytes), u64::MAX), &k).is_err());
    let mut next = decode(&bytes, &k).unwrap();
    k.step(&mut next, quote(97, 103));
    assert!(!matches!(next.status(), Status::Halted(_)));
}

/// Number of fuzz seeds: `KERNEL_FUZZ_SEEDS` or 60.
fn fuzz_seeds() -> u64 {
    std::env::var("KERNEL_FUZZ_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60)
}

/// Decodes `bytes` and, if accepted, steps the state with Hold and with the
/// scenario's action: neither may halt (or panic).
fn step_accepted(k: &Kernel<'_>, sc: &Scenario, bytes: &[u8], j: u64, ctx: &str) -> bool {
    let Ok(s) = decode(bytes, k) else {
        return false;
    };
    let obs = k.observe(&s);
    for action in [KernelAction::Hold, sc.action(j, obs.bid, obs.ask)] {
        let mut next = s;
        k.step(&mut next, action);
        assert!(
            !matches!(next.status(), Status::Halted(_)),
            "{ctx}: accepted state halts on {action:?}: {:?}",
            next.status()
        );
    }
    true
}

#[test]
fn test_kernel_decode_fuzz_accepted_states_never_halt() {
    // Every 7th decision state of seeds 0..KERNEL_FUZZ_SEEDS (60 by default)
    // and datasets 0..4 gets 40 mutations of one to three bytes (a bit
    // flip, 0xFF or 0x00). Whatever decode accepts is stepped with Hold and
    // with the action the run took there: no state may halt or panic.
    let (mut tried, mut accepted) = (0u64, 0u64);
    let mut halts = Vec::new();
    let mut rng = SplitMix64(0xC0FFEE);
    for seed in 0..fuzz_seeds() {
        for index in 0..4u64 {
            let sc = scenario(seed, index);
            let data = sc.data();
            let k = kernel(&sc, &data);
            let kr = run_kernel(&k, &sc, true);
            for (j, state) in kr.states.iter().enumerate().step_by(7) {
                let bytes = encode(state);
                for _ in 0..40 {
                    let mut b = bytes.clone();
                    let flips = 1 + rng.next() % 3;
                    for _ in 0..flips {
                        let at = usize::try_from(rng.next() % b.len() as u64).unwrap();
                        b[at] = match rng.next() % 3 {
                            0 => b[at] ^ (1 << (rng.next() % 8)),
                            1 => 0xFF,
                            _ => 0x00,
                        };
                    }
                    tried += 1;
                    let Ok(decoded) = decode(&b, &k) else {
                        continue;
                    };
                    accepted += 1;
                    let taken = kr.actions.get(j).copied().unwrap_or(KernelAction::Hold);
                    for action in [KernelAction::Hold, taken] {
                        let mut next = decoded;
                        k.step(&mut next, action);
                        if let Status::Halted(reason) = next.status() {
                            let diffs: Vec<(usize, u8, u8)> = bytes
                                .iter()
                                .zip(&b)
                                .enumerate()
                                .filter(|(_, (x, y))| x != y)
                                .map(|(i, (x, y))| (i, *x, *y))
                                .collect();
                            halts.push(format!(
                                "seed {seed} dataset {index} decision {j} {action:?}: {reason:?}, bytes {diffs:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        halts.is_empty(),
        "{} halts in {tried} tries ({accepted} accepted):\n{}",
        halts.len(),
        halts.join("\n")
    );
    assert!(accepted > 0, "no mutation accepted in {tried} tries");
}

#[test]
fn test_kernel_decode_fuzz_regression_fee_byte_halts_never() {
    // Seed 55, dataset 0, decision 21: byte 320 from 0xFF to 0x7F moved
    // the venue fees far from the fills; decode must not accept a state
    // whose next step halts.
    let sc = scenario(55, 0);
    let data = sc.data();
    let k = kernel(&sc, &data);
    let kr = run_kernel(&k, &sc, true);
    let mut bytes = encode(&kr.states[21]);
    assert_eq!(bytes[320], 0xFF);
    bytes[320] = 0x7F;
    step_accepted(&k, &sc, &bytes, 21, "seed 55 regression");
}
