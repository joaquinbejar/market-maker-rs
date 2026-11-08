//! The kernel's balance gate against `BacktestEngine` with a `MarginAccount`:
//! the same scenarios, decided alike order by order, with the same reports,
//! fills, ledger and margin balances. All data here is synthetic.

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

use harness::*;
use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    ExecKind, GateCheck, GateVerdict, KernelAction, KernelConfig, KernelEvent, KernelFillModel,
    KernelMargin, Origin,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;

/// `sc` with a balance gate that binds: its lot replaced by one of several
/// grid units (so that a reduction can fall between two lots), no limits, a
/// quote balance of a few lots' notional at the first mid plus a fraction of
/// a tick's, and a base deposit of a few lots.
fn margin_scenario(mut sc: Scenario, seed: u64, index: u64) -> Scenario {
    let mut rng = SplitMix64(
        seed.wrapping_mul(0x9E37)
            .wrapping_add(index)
            .wrapping_add(3),
    );
    let old_lot = sc.cfg.lot_size;
    let lot = rng.pick(&[dec!(1), dec!(0.5), dec!(0.25), dec!(0.03)]);
    let relot = |size: Decimal| ((size / old_lot).normalize() * lot).normalize();
    sc.ticks = sc
        .ticks
        .iter()
        .map(|t| {
            MarketTick::new(
                t.timestamp,
                t.bid_price,
                relot(t.bid_size),
                t.ask_price,
                relot(t.ask_size),
            )
        })
        .collect();
    let first = sc.ticks[0].clone();
    let mid = (first.bid_price + first.ask_price) / dec!(2);
    let tick = sc.cfg.tick_size;
    let lots = Decimal::from(rng.below(8));
    let fraction = Decimal::from(rng.below(1_000)) * tick * lot / dec!(10);
    let capital = if rng.chance(15) {
        // Room for everything: the gate accepts every order.
        dec!(10_000_000)
    } else {
        (mid * lot * lots + fraction).round_dp(tick.scale() + lot.scale())
    };
    let fill_model = match sc.cfg.fill_model {
        KernelFillModel::QueuePosition { rate, min_queue_ms } => KernelFillModel::QueuePosition {
            rate: relot(rate),
            min_queue_ms,
        },
        other => other,
    };
    let mut cfg = KernelConfig::new(
        tick,
        lot,
        sc.cfg.max_price,
        Decimal::from(10) * lot,
        sc.cfg.max_steps,
    )
    .with_initial_capital(capital)
    .with_fee_rate(sc.cfg.fee_rate)
    .with_latency(sc.cfg.latency)
    .with_fill_model(fill_model)
    .with_margin(KernelMargin::new(Decimal::from(rng.below(5)) * lot));
    cfg.limits = None;
    sc.cfg = cfg;
    sc.limits = LimitsKind::None;
    sc
}

/// What the parity loop saw, to show that every path was taken.
#[derive(Debug, Default)]
struct Seen {
    verdicts: [u64; 4],
    sells_refused: u64,
    buys_refused: u64,
    fills: u64,
    cancels: u64,
}

fn verdict_index(verdict: GateVerdict) -> usize {
    match verdict {
        GateVerdict::Accepted => 0,
        GateVerdict::Reduced => 1,
        GateVerdict::ReducedOffLot => 2,
        GateVerdict::Rejected => 3,
    }
}

/// Runs `sc` on the kernel and the engine and compares everything, the margin
/// account's balances included.
fn assert_parity(sc: &Scenario, ctx: &str, seen: &mut Seen) {
    let data = sc.data();
    let k = kernel(sc, &data);
    let kr = run_kernel(&k, sc, false);
    let er = run_engine(sc);
    assert_equivalent(&k, sc, &kr, &er, ctx);
    assert_cancel_pending_agreement(&kr, &er, ctx);
    // The quote and base balances the account ends with are the kernel's.
    let summary = k.venue_summary(&kr.last).unwrap();
    let end = er.margin.unwrap();
    assert_eq!(
        end.quote.0,
        sc.cfg.initial_capital + summary.realized - summary.total_fees,
        "{ctx}: quote balance"
    );
    assert_eq!(
        end.base.0,
        sc.cfg.margin.unwrap().base_deposit + summary.position,
        "{ctx}: base balance"
    );
    for event in &kr.events {
        match event {
            KernelEvent::Gate(g) => {
                assert_eq!(g.check, GateCheck::Margin, "{ctx}: check");
                seen.verdicts[verdict_index(g.verdict)] += 1;
                if g.submitted == 0 {
                    match g.side {
                        Action::Buy => seen.buys_refused += 1,
                        _ => seen.sells_refused += 1,
                    }
                }
            }
            KernelEvent::Fill(_) => seen.fills += 1,
            KernelEvent::Report(r) if r.kind == ExecKind::Canceled => seen.cancels += 1,
            _ => {}
        }
    }
}

#[test]
fn test_balance_gate_decides_as_the_margin_account_in_the_engine() {
    let mut seen = Seen::default();
    let runs = seeds() / 5;
    for seed in 0..runs {
        for index in 0..DATASETS {
            let ctx = format!("seed {seed} dataset {index}");
            let sc = margin_scenario(scenario(seed, index), seed, index);
            assert_parity(&sc, &ctx, &mut seen);
        }
    }
    assert!(
        seen.verdicts.iter().all(|n| *n > 0),
        "every verdict decided: {seen:?}"
    );
    assert!(seen.buys_refused > 0 && seen.sells_refused > 0, "{seen:?}");
    assert!(seen.fills > 0 && seen.cancels > 0, "{seen:?}");
}

#[test]
fn test_balance_gate_decides_as_the_margin_account_under_partial_fills() {
    let mut seen = Seen::default();
    let mut runs = 0;
    for seed in 0..seeds() / 5 {
        for index in 0..DATASETS {
            let Some(base) = gate::partial_scenario(seed, index) else {
                continue;
            };
            let sc = margin_scenario(base, seed, index);
            if !gate::fits(&sc) {
                continue;
            }
            runs += 1;
            assert_parity(
                &sc,
                &format!("partial seed {seed} dataset {index}"),
                &mut seen,
            );
        }
    }
    assert!(runs > 100, "{runs} partial runs");
    assert!(seen.verdicts.iter().all(|n| *n > 0), "{seen:?}");
}

/// Tick 1, lot 1, no fee, no latency: rows at 100/101.
fn hand(capital: Decimal, base: Decimal, actions: Vec<KernelAction>) -> Scenario {
    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(10_000), dec!(10), 10)
        .with_initial_capital(capital)
        .with_margin(KernelMargin::new(base));
    let ticks = (0..4)
        .map(|i| MarketTick::new(1_000 + i, dec!(100), dec!(5), dec!(101), dec!(5)))
        .collect();
    Scenario::fixed(cfg, ticks, actions)
}

fn quote(bid: u64, ask: u64, bid_lots: u64, ask_lots: u64) -> KernelAction {
    KernelAction::Quote {
        bid,
        ask,
        bid_lots,
        ask_lots,
    }
}

fn gates(sc: &Scenario) -> Vec<(Action, u64, GateVerdict, u64)> {
    let data = sc.data();
    let k = kernel(sc, &data);
    run_kernel(&k, sc, false)
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Gate(g) => Some((g.side, g.requested, g.verdict, g.submitted)),
            _ => None,
        })
        .collect()
}

#[test]
fn test_a_buy_is_reduced_to_the_lots_the_capital_covers_and_a_sell_to_the_base() {
    let mut seen = Seen::default();
    // 350 of quote covers 3 lots at 100; 2 lots of base cover a sell of 2.
    let sc = hand(dec!(350), dec!(2), vec![quote(100, 102, 5, 4)]);
    assert_eq!(
        gates(&sc),
        vec![
            (Action::Buy, 5, GateVerdict::Reduced, 3),
            (Action::Sell, 4, GateVerdict::Reduced, 2),
        ]
    );
    assert_parity(&sc, "reduced", &mut seen);
    // Below one lot of either: refused, as a local reject the agent receives.
    let sc = hand(dec!(99), dec!(0), vec![quote(100, 102, 1, 1)]);
    assert_eq!(
        gates(&sc),
        vec![
            (Action::Buy, 1, GateVerdict::Rejected, 0),
            (Action::Sell, 1, GateVerdict::Rejected, 0),
        ]
    );
    assert_parity(&sc, "refused", &mut seen);
    let data = sc.data();
    let k = kernel(&sc, &data);
    let run = run_kernel(&k, &sc, false);
    let rejects = run
        .events
        .iter()
        .filter(|e| {
            matches!(e, KernelEvent::Delivered(r)
            if r.kind == ExecKind::Rejected && r.origin == Origin::Local)
        })
        .count();
    assert_eq!(rejects, 2);
    assert_eq!(run.last.counters().venue_fills, 0);
}

#[test]
fn test_a_live_order_keeps_its_reservation_until_its_cancel_is_applied() {
    // 250 covers 2 lots at 100. The first bid rests: no ask reaches 100. The
    // second quote cancels it, but the cancel is applied at the venue after
    // the new bid is checked, so the new bid meets the old one's reservation
    // and is refused. By the third decision the cancel has been applied.
    let sc = hand(
        dec!(250),
        dec!(0),
        vec![
            quote(100, 200, 2, 1),
            quote(100, 200, 2, 1),
            quote(100, 200, 2, 1),
        ],
    );
    let buys: Vec<_> = gates(&sc)
        .into_iter()
        .filter(|g| g.0 == Action::Buy)
        .collect();
    assert_eq!(
        buys,
        vec![
            (Action::Buy, 2, GateVerdict::Accepted, 2),
            (Action::Buy, 2, GateVerdict::Rejected, 0),
            (Action::Buy, 2, GateVerdict::Accepted, 2),
        ]
    );
    assert_parity(&sc, "reservation", &mut Seen::default());
}

#[test]
fn test_a_reduction_between_two_lots_is_refused_as_the_engine_refuses_it() {
    // Lot 0.5 (five grid units of 0.1), price 100: 120 of quote covers 1.2,
    // twelve grid units, which is not a whole number of lots.
    let cfg = KernelConfig::new(dec!(1), dec!(0.5), dec!(10_000), dec!(5), 10)
        .with_initial_capital(dec!(120))
        .with_margin(KernelMargin::new(dec!(0)));
    let ticks = (0..3)
        .map(|i| MarketTick::new(1_000 + i, dec!(100), dec!(5), dec!(101), dec!(5)))
        .collect();
    let sc = Scenario::fixed(cfg, ticks, vec![quote(100, 102, 4, 1)]);
    assert_eq!(
        gates(&sc)[0],
        (Action::Buy, 4, GateVerdict::ReducedOffLot, 0)
    );
    assert_parity(&sc, "off lot", &mut Seen::default());
}

#[test]
fn test_a_kernel_without_the_gate_keeps_its_fingerprint_and_runs_alike() {
    let sc = margin_scenario(scenario(3, 1), 3, 1);
    let mut plain = sc.cfg.clone();
    plain.margin = None;
    // The section is written only with the gate: without it the bytes are the
    // ones a configuration from before the gate writes.
    let bytes = plain.encode();
    assert!(!bytes.windows(2).any(|w| w == b"MG"));
    assert_ne!(plain.fingerprint(), sc.cfg.fingerprint());
    assert_eq!(&sc.cfg.encode()[..bytes.len()], &bytes[..]);
    // A gate whose balances are never reached changes no report, fill or
    // counter.
    let run = |cfg: KernelConfig| {
        let sc = Scenario { cfg, ..sc.clone() };
        let data = sc.data();
        let k = kernel(&sc, &data);
        let kr = run_kernel(&k, &sc, false);
        let events: Vec<_> = kr
            .events
            .into_iter()
            .filter(|e| !matches!(e, KernelEvent::Gate(_)))
            .collect();
        (events, kr.last.counters())
    };
    let rich = KernelConfig {
        initial_capital: dec!(10_000_000),
        ..plain
    };
    let gated = rich
        .clone()
        .with_margin(KernelMargin::new(dec!(1_000) * rich.lot_size));
    let (with, without) = (run(gated), run(rich));
    assert!(with.1.venue_fills > 0);
    assert_eq!(with, without);
}

#[test]
fn test_a_base_deposit_off_the_lot_grid_or_negative_is_refused() {
    let base = hand(dec!(100), dec!(0), vec![]);
    let data = base.data();
    for deposit in [dec!(-1), dec!(0.5)] {
        let cfg = base.cfg.clone().with_margin(KernelMargin::new(deposit));
        let error = Kernel::new(&cfg, &data).unwrap_err();
        assert!(error.to_string().contains("base_deposit"), "{error}");
    }
    let cfg = base.cfg.clone().with_margin(KernelMargin::new(dec!(3)));
    assert!(Kernel::new(&cfg, &data).is_ok());
}

#[test]
fn test_a_rebate_reserves_the_whole_notional_as_the_account_does() {
    // A rebate of 0.1 %: the account reserves at a fee rate of 0, so 199.9 of quote covers
    // one lot at 100, not the two that 99.9 a lot would.
    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(10_000), dec!(10), 10)
        .with_initial_capital(dec!(199.9))
        .with_fee_rate(dec!(-0.001))
        .with_margin(KernelMargin::new(dec!(0)));
    let ticks = (0..3)
        .map(|i| MarketTick::new(1_000 + i, dec!(100), dec!(5), dec!(101), dec!(5)))
        .collect();
    let sc = Scenario::fixed(cfg, ticks, vec![quote(100, 102, 2, 1)]);
    assert_eq!(gates(&sc)[0], (Action::Buy, 2, GateVerdict::Reduced, 1));
    assert_parity(&sc, "rebate", &mut Seen::default());
}

#[test]
fn test_with_limits_the_gate_decides_on_what_they_approved_and_both_are_recorded() {
    // Limits of 3 lots cut a bid of 5 to 3; 250 of quote then covers 2.
    let mut sc = hand(dec!(250), dec!(5), vec![quote(100, 102, 5, 1)]);
    sc.cfg.limits = Some(RiskLimits::new(dec!(3), Decimal::MAX, dec!(0)).unwrap());
    let data = sc.data();
    let k = kernel(&sc, &data);
    let records: Vec<_> = run_kernel(&k, &sc, false)
        .events
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Gate(g) => Some((g.check, g.side, g.requested, g.verdict, g.submitted)),
            _ => None,
        })
        .collect();
    assert_eq!(
        records,
        vec![
            (GateCheck::Limits, Action::Buy, 5, GateVerdict::Reduced, 3),
            (GateCheck::Margin, Action::Buy, 3, GateVerdict::Reduced, 2),
            (GateCheck::Limits, Action::Sell, 1, GateVerdict::Accepted, 1),
            (GateCheck::Margin, Action::Sell, 1, GateVerdict::Accepted, 1),
        ]
    );
}
