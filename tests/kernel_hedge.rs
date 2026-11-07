//! The hedge lane of `HedgedKernel` (#256), on hand-computed fixtures.
//!
//! No engine runs two instruments, so the lane is checked here by hand and
//! by its own `Decimal` shadow; the quoted lane is checked against a kernel
//! alone.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    CombinedLimits, EventSink, FinancialKernel, FundingRate, HedgeEvent, HedgeInstrument,
    HedgeLaneConfig, HedgeRefusal, HedgedAction, HedgedKernel, HedgedState, KernelAction,
    KernelConfig, KernelDataset, KernelEvent, KernelLatency, KernelMargin, QuoteSide,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::risk::RiskLimits;

const MS: u64 = 1_000_000;

type K<'d> = HedgedKernel<'d>;
type S = HedgedState<16, 64, 128, 64>;

/// Submission and acknowledgement delays of 1 ms, no jitter.
fn latency(submit_q: u64, ack_q: u64, jitter_q: u64) -> KernelLatency {
    let mut l = KernelLatency::zero(MS).with_delays(0, submit_q, 1, ack_q);
    l.jitter_q = jitter_q;
    l
}

/// The quoted lane: tick 0.01, lot 1, rows every second from 1 s at
/// 100.00 / 100.02.
fn quoted(l: KernelLatency) -> (KernelConfig, KernelDataset) {
    let cfg = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(10), 100)
        .with_initial_capital(dec!(10000))
        .with_latency(l);
    let ticks: Vec<MarketTick> = (1..=10)
        .map(|i| MarketTick::new(i * 1_000, dec!(100.00), dec!(5), dec!(100.02), dec!(5)))
        .collect();
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    (cfg, data)
}

/// The hedge instrument: tick 0.01, lot 1, fee 5 bp.
fn instrument(l: KernelLatency) -> KernelConfig {
    KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(10), 100)
        .with_fee_rate(dec!(0.0005))
        .with_latency(l)
}

/// Hedge rows `(ms, bid, ask, lots)`.
fn hedge_rows(cfg: &KernelConfig, rows: &[(u64, Decimal, Decimal, Decimal)]) -> KernelDataset {
    let ticks: Vec<MarketTick> = rows
        .iter()
        .map(|(ms, bid, ask, lots)| MarketTick::new(*ms, *bid, *lots, *ask, *lots))
        .collect();
    KernelDataset::from_ticks(&ticks, cfg).unwrap()
}

fn spot(cfg: KernelConfig) -> HedgeLaneConfig {
    HedgeLaneConfig::new(cfg, HedgeInstrument::Spot, dec!(1), 10_000 * MS)
}

fn hedge(side: QuoteSide, lots: u64) -> HedgedAction {
    HedgedAction::Hedge { side, lots }
}

fn hold() -> HedgedAction {
    HedgedAction::Kernel(KernelAction::Hold)
}

#[test]
fn test_a_hedge_fills_at_the_standing_touch_and_reaches_the_agent_after_its_report() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(s.kernel.now_ns(), 1_000 * MS);
    let view = k.hedge_view(&s);
    assert_eq!(view.row, Some(0));
    assert!(!view.stale);
    let out = k.step(&mut s, hedge(QuoteSide::Bid, 2));
    assert_eq!(out.hedge, Some(Ok(())));
    // Executed at 1 001 ms at the ask, 100.04, and delivered at 1 002 ms: by
    // the decision at 2 000 ms the agent holds it.
    assert_eq!(s.kernel.now_ns(), 2_000 * MS);
    assert_eq!(s.lane.venue_position_lots(), 2);
    assert_eq!(s.lane.agent_position_lots(), 2);
    assert!(!s.lane.in_flight());
    let summary = k.summary(&s).unwrap();
    assert_eq!(summary.position, dec!(2));
    assert_eq!(summary.cash, dec!(-200.08));
    // 200.08 * 0.0005.
    assert_eq!(summary.fees, dec!(0.10004));
    assert_eq!(summary.funding, dec!(0));
    // Marked at the mid 100.02.
    assert_eq!(summary.mark_value, dec!(200.04));
    assert_eq!(summary.net, dec!(-0.14004));
    let quoted_net = k.kernel().venue_summary(&s.kernel).unwrap().net_pnl;
    assert_eq!(summary.combined_net, quoted_net + dec!(-0.14004));
    let c = s.lane.counters();
    assert_eq!((c.submitted, c.filled, c.delivered), (1, 1, 1));
    assert_eq!(k.hedge_view(&s).avg_entry, dec!(100.04));
}

#[test]
fn test_a_sell_hedge_takes_the_bid_and_a_better_touch_fills_at_its_price() {
    let l = latency(500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    // The decision sees 100.00 / 100.04; by execution, at 1 500 ms, the bid
    // has risen to 100.03.
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_200, dec!(100.03), dec!(100.05), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Ask, 3)).hedge, Some(Ok(())));
    let summary = k.summary(&s).unwrap();
    assert_eq!(summary.position, dec!(-3));
    assert_eq!(summary.cash, dec!(300.09));
}

#[test]
fn test_a_touch_moved_through_the_limit_cancels_and_releases_the_lots() {
    let l = latency(500, 1, 0);
    let (cfg, data) = quoted(l);
    // A position limit of 2: lots left reserved would refuse the next hedge.
    let inst = instrument(l).with_limits(RiskLimits::new(dec!(2), dec!(100000), dec!(1)).unwrap());
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_200, dec!(100.04), dec!(100.06), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hedge(QuoteSide::Bid, 2));
    assert_eq!(s.lane.venue_position_lots(), 0);
    let c = s.lane.counters();
    assert_eq!((c.filled, c.moved_through, c.delivered), (0, 1, 1));
    assert_eq!(k.summary(&s).unwrap().cash, dec!(0));
    // The lots were released: a new hedge goes through, at the new ask.
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 2)).hedge, Some(Ok(())));
    assert_eq!(s.lane.venue_position_lots(), 2);
}

#[test]
fn test_a_hedge_beyond_the_touch_fills_its_lots_and_counts_the_rest() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(3))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hedge(QuoteSide::Bid, 5));
    assert_eq!(s.lane.venue_position_lots(), 3);
    let c = s.lane.counters();
    assert_eq!((c.partial, c.beyond_touch_lots), (1, 2));
}

#[test]
fn test_one_hedge_at_a_time_and_the_size_and_touch_checks() {
    // Submission takes 1.5 s, so the hedge is still in flight at 2 s.
    let l = latency(1_500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(1_500, dec!(100.00), dec!(100.04), dec!(5))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    // At 1 s no hedge row is visible yet.
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge,
        Some(Err(HedgeRefusal::NoTouch))
    );
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 0)).hedge,
        Some(Err(HedgeRefusal::SizeRange))
    );
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 11)).hedge,
        Some(Err(HedgeRefusal::SizeRange))
    );
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge, Some(Ok(())));
    assert!(s.lane.in_flight());
    assert_eq!(k.hedge_mask(&s, &[1, 2, 11]), 0);
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge,
        Some(Err(HedgeRefusal::InFlight))
    );
    let refused = s.lane.counters().refused;
    assert_eq!(refused[HedgeRefusal::NoTouch.index()], 1);
    assert_eq!(refused[HedgeRefusal::SizeRange.index()], 2);
    assert_eq!(refused[HedgeRefusal::InFlight.index()], 1);
    // Out of flight once its report is delivered.
    k.step(&mut s, hold());
    assert!(!s.lane.in_flight());
    assert_eq!(k.hedge_mask(&s, &[1, 2, 11]), 0b011);
}

#[test]
fn test_at_equal_times_the_decision_does_not_see_the_hedge_lane() {
    // A hedge row at exactly the decision's 1 000 ms, and a report due at
    // exactly the next decision's 2 000 ms.
    let l = latency(1, 999, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_000, dec!(100.01), dec!(100.03), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    // The row at 1 000 ms is not visible to the decision at 1 000 ms.
    assert_eq!(k.hedge_view(&s).row, Some(0));
    k.step(&mut s, hedge(QuoteSide::Bid, 2));
    // It executed at 1 001 ms against the touch standing then: 100.03.
    assert_eq!(k.summary(&s).unwrap().cash, dec!(-200.06));
    // Its report, due at 2 000 ms, is not seen by the decision at 2 000 ms.
    assert_eq!(s.kernel.now_ns(), 2_000 * MS);
    assert_eq!(s.lane.venue_position_lots(), 2);
    assert_eq!(s.lane.agent_position_lots(), 0);
    assert!(k.hedge_view(&s).in_flight);
    k.step(&mut s, hold());
    assert_eq!(s.lane.agent_position_lots(), 2);
}

#[test]
fn test_the_agent_sees_a_hedge_row_after_the_observation_delay() {
    // An observation delay of 200 ms: the row at 900 ms is seen from 1 100 ms.
    let l = KernelLatency::zero(MS).with_delays(200, 1, 1, 1);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (900, dec!(100.01), dec!(100.03), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let s: S = k.reset(7);
    // The quoted row at 1 000 ms is decided on at 1 200 ms, which sees both.
    assert_eq!(s.kernel.now_ns(), 1_200 * MS);
    assert_eq!(k.hedge_view(&s).row, Some(1));
    let rows = hedge_rows(
        &instrument(l),
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_100, dec!(100.01), dec!(100.03), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(instrument(l)), &rows).unwrap();
    let s: S = k.reset(7);
    // Run at 1 100 ms, before the decision, but seen only at 1 300 ms.
    assert_eq!(k.hedge_view(&s).row, Some(0));
    assert_eq!(k.hedge_view(&s).tick.unwrap().ask, 10_004);
}

#[test]
fn test_a_stale_lane_refuses_at_the_decision_and_cancels_in_flight() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    // Stale after 600 ms: fresh at the decision at 1 000 ms, stale at 2 000 ms.
    let lane = HedgeLaneConfig::new(inst.clone(), HedgeInstrument::Spot, dec!(1), 600 * MS);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hold());
    assert!(k.hedge_view(&s).stale);
    assert_eq!(k.hedge_mask(&s, &[1]), 0);
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge,
        Some(Err(HedgeRefusal::Stale))
    );
    // In flight: fresh at 1 000 ms, stale by its execution at 1 500 ms.
    let l = latency(500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let lane = HedgeLaneConfig::new(inst, HedgeInstrument::Spot, dec!(1), 600 * MS);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge, Some(Ok(())));
    assert_eq!(s.lane.venue_position_lots(), 0);
    assert_eq!(s.lane.counters().stale_in_flight, 1);
    // The quoted lane is not affected by the hedge lane's staleness.
    let quote = KernelAction::Quote {
        bid: 9_999,
        ask: 10_003,
        bid_lots: 1,
        ask_lots: 1,
    };
    let out = k.step(&mut s, HedgedAction::Kernel(quote));
    assert_eq!(
        out.kernel.applied,
        market_maker_rs::backtest::kernel::ActionStatus::Applied
    );
}

fn perpetual(inst: KernelConfig, rates: &[(u64, Decimal)], staleness_ms: u64) -> HedgeLaneConfig {
    let funding = rates
        .iter()
        .map(|(ms, rate)| FundingRate {
            at_ns: ms * MS,
            rate: *rate,
        })
        .collect();
    HedgeLaneConfig::new(
        inst,
        HedgeInstrument::LinearPerpetual { funding },
        dec!(1),
        staleness_ms * MS,
    )
}

#[test]
fn test_funding_charges_the_venue_position_at_the_last_mid_before_it() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    // Mid 100.02 at 500 ms, 100.06 at 1 500 ms; funding at 1 500 ms uses the
    // row before it, as funding runs first at equal times.
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_500, dec!(100.04), dec!(100.08), dec!(5)),
        ],
    );
    for (side, paid) in [
        (QuoteSide::Bid, dec!(-0.020004)),
        (QuoteSide::Ask, dec!(0.020004)),
    ] {
        let lane = perpetual(inst.clone(), &[(1_500, dec!(0.0001))], 10_000);
        let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
        let mut s: S = k.reset(7);
        k.step(&mut s, hedge(side, 2));
        // A long of 2 pays 2 * 100.02 * 0.0001; a short of 2 receives it.
        assert_eq!(k.summary(&s).unwrap().funding, paid);
        assert_eq!(s.lane.counters().funding_paid, 1);
    }
    // The fill's report reaches the agent at 1 901 ms, after the payment: it
    // is charged on the venue position all the same.
    let l = latency(1, 900, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_500, dec!(100.04), dec!(100.08), dec!(5)),
        ],
    );
    let lane = perpetual(inst, &[(1_500, dec!(0.0001))], 10_000);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hedge(QuoteSide::Bid, 2));
    assert_eq!(k.summary(&s).unwrap().funding, dec!(-0.020004));
}

#[test]
fn test_funding_rounds_half_to_even_and_flags_what_it_cannot_mark() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(99.99), dec!(100.01), dec!(5))]);
    // A long of 1 at mid 100: 5e-7 rounds to 0 at the money unit 1e-6, and
    // 1.5e-6 to 2e-6. A payment before any row marks nothing. Those at 1.6 s
    // and 3 s, over 1 s after the last row, are stale and still charged.
    let lane = perpetual(
        inst,
        &[
            (100, dec!(0.0001)),
            (1_500, dec!(0.000000005)),
            (1_600, dec!(0.000000015)),
            (3_000, dec!(0.0001)),
        ],
        1_000,
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hedge(QuoteSide::Bid, 1));
    assert_eq!(k.summary(&s).unwrap().funding, dec!(-0.000002));
    // The payment at exactly 3 000 ms runs after the decision at 3 000 ms.
    k.step(&mut s, hold());
    assert_eq!(k.summary(&s).unwrap().funding, dec!(-0.000002));
    k.step(&mut s, hold());
    assert_eq!(k.summary(&s).unwrap().funding, dec!(-0.010002));
    let c = s.lane.counters();
    assert_eq!(
        (c.funding_without_mark, c.funding_paid, c.funding_stale),
        (1, 3, 2)
    );
}

#[test]
fn test_combined_limits_bind_both_lanes_with_the_hedge_in_flight_at_its_worst() {
    // The hedge takes 1.5 s to execute, so it is in flight at 2 s.
    let l = latency(1_500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let lane = spot(inst).with_combined(CombinedLimits {
        max_net_exposure: Some(dec!(3)),
        max_gross_notional: None,
    });
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 4)).hedge,
        Some(Err(HedgeRefusal::Combined))
    );
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 2)).hedge, Some(Ok(())));
    // At 3 s the buy of 2 is in flight: a quoted buy of 2 would make 4 if
    // both filled, so it is refused; the sell of 2, at worst -2, is not.
    assert!(s.lane.in_flight());
    let quote = KernelAction::Quote {
        bid: 9_999,
        ask: 10_003,
        bid_lots: 2,
        ask_lots: 2,
    };
    let out = k.step(&mut s, HedgedAction::Kernel(quote));
    assert_eq!(out.combined_rejects, 1);
    assert_eq!(s.lane.counters().combined_quote_rejects, 1);
    assert_eq!(s.kernel.live_lots(), (0, 2));
    // Once the hedge is long 2, a quoted sell of 3, with the resting ask of 2
    // still live while its cancel is in flight, nets 0 - (2 + 3) + 2 = -3 at worst:
    // admitted.
    k.step(&mut s, hold());
    assert_eq!(s.lane.venue_position_lots(), 2);
    let sell = KernelAction::OneSided {
        side: QuoteSide::Ask,
        price: 10_003,
        lots: 3,
    };
    assert_eq!(
        k.step(&mut s, HedgedAction::Kernel(sell)).combined_rejects,
        0
    );
    // A sell of 4 now, with the 3 still live, would net -5: refused.
    let sell = KernelAction::OneSided {
        side: QuoteSide::Ask,
        price: 10_003,
        lots: 4,
    };
    assert_eq!(
        k.step(&mut s, HedgedAction::Kernel(sell)).combined_rejects,
        1
    );
}

#[test]
fn test_the_gross_limit_sees_a_fully_hedged_book_the_net_limit_cannot() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(10))]);
    // Short 4 hedged, quoting a buy of 4 is net flat at worst, but 8 lots at
    // about 100 are 800 of gross notional against a limit of 700.
    let lane = spot(inst).with_combined(CombinedLimits {
        max_net_exposure: Some(dec!(1)),
        max_gross_notional: Some(dec!(700)),
    });
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    // A short of 4 alone breaks the net limit.
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Ask, 4)).hedge,
        Some(Err(HedgeRefusal::Combined))
    );
    let lane = spot(instrument(l)).with_combined(CombinedLimits {
        max_net_exposure: None,
        max_gross_notional: Some(dec!(700)),
    });
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Ask, 4)).hedge, Some(Ok(())));
    let buy = KernelAction::OneSided {
        side: QuoteSide::Bid,
        price: 9_999,
        lots: 4,
    };
    assert_eq!(
        k.step(&mut s, HedgedAction::Kernel(buy)).combined_rejects,
        1
    );
    let buy = KernelAction::OneSided {
        side: QuoteSide::Bid,
        price: 9_999,
        lots: 2,
    };
    assert_eq!(
        k.step(&mut s, HedgedAction::Kernel(buy)).combined_rejects,
        0
    );
}

#[test]
fn test_the_instruments_own_limits_bind_the_hedge() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l).with_limits(RiskLimits::new(dec!(2), dec!(100000), dec!(1)).unwrap());
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(10))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    // Reduced to the 2 lots the position limit leaves.
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 3)).hedge, Some(Ok(())));
    assert_eq!(s.lane.venue_position_lots(), 2);
    assert_eq!(
        k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge,
        Some(Err(HedgeRefusal::Limits))
    );
}

#[test]
fn test_the_quoted_lane_runs_as_a_kernel_alone_with_or_without_hedges() {
    // Jitter on: the lane's own stream leaves the quoted lane's draws as they
    // are.
    let l = latency(2, 3, 4);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (300, dec!(100.00), dec!(100.04), dec!(5)),
            (2_300, dec!(100.02), dec!(100.06), dec!(5)),
            (4_300, dec!(100.01), dec!(100.03), dec!(5)),
        ],
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let alone: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut s: S = k.reset(11);
    let mut a = alone.reset(11);
    assert_eq!(s.kernel, a);
    let quote = KernelAction::Quote {
        bid: 10_000,
        ask: 10_002,
        bid_lots: 1,
        ask_lots: 1,
    };
    let plan = [
        hedge(QuoteSide::Bid, 1),
        HedgedAction::Kernel(quote),
        hedge(QuoteSide::Ask, 1),
        hold(),
        HedgedAction::Kernel(quote),
        hedge(QuoteSide::Bid, 2),
        hold(),
    ];
    for action in plan {
        let as_kernel = match action {
            HedgedAction::Kernel(a) => a,
            _ => KernelAction::Hold,
        };
        let out = k.step(&mut s, action);
        let alone_out = alone.step(&mut a, as_kernel);
        assert_eq!(out.kernel.applied, alone_out.applied);
        assert_eq!(s.kernel, a);
    }
    assert_eq!(s.lane.counters().submitted, 3);
}

#[test]
fn test_what_is_pending_at_the_end_is_dropped_and_counted() {
    // Submission takes 2.5 s, and the hedge is sent at the last row: the run
    // ends first.
    let l = latency(2_500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst), &rows).unwrap();
    let mut s: S = k.reset(7);
    while s.kernel.now_ns() < 10_000 * MS {
        k.step(&mut s, hold());
    }
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge, Some(Ok(())));
    while s.kernel.status().is_running() {
        k.step(&mut s, hold());
    }
    assert_eq!(s.lane.counters().dropped_at_end, 1);
    assert!(!s.lane.in_flight());
    assert_eq!(s.lane.venue_position_lots(), 0);
}

#[test]
fn test_the_lane_is_validated() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let refused = |lane: HedgeLaneConfig| K::new(&cfg, &data, &lane, &rows).is_err();
    let mut lane = spot(inst.clone());
    lane.multiplier = dec!(0);
    assert!(refused(lane));
    assert!(refused(spot(inst.clone().with_initial_capital(dec!(1)))));
    assert!(refused(spot(inst.clone().with_latency(latency(2, 1, 0)))));
    assert!(refused(perpetual(
        inst.clone(),
        &[(2_000, dec!(0.0001)), (2_000, dec!(0.0001))],
        1_000
    )));
    assert!(refused(spot(inst.clone()).with_combined(CombinedLimits {
        max_net_exposure: Some(dec!(-1)),
        max_gross_notional: None,
    })));
    // The hedge data on another grid.
    let other = instrument(l);
    let other = KernelConfig::new(dec!(0.1), dec!(1), dec!(1000), dec!(10), 100)
        .with_latency(other.latency);
    let off_grid = hedge_rows(&other, &[(500, dec!(100.0), dec!(100.1), dec!(5))]);
    assert!(K::new(&cfg, &data, &spot(inst.clone()), &off_grid).is_err());
    // A state of another lane is refused.
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst.clone()), &rows).unwrap();
    let k2: K<'_> = HedgedKernel::new(
        &cfg,
        &data,
        &spot(inst).with_combined(CombinedLimits {
            max_net_exposure: Some(dec!(5)),
            max_gross_notional: None,
        }),
        &rows,
    )
    .unwrap();
    let mut s: S = k.reset(7);
    assert!(k.owns(&s));
    assert!(!k2.owns(&s));
    assert!(k2.summary(&s).is_err());
    k2.step(&mut s, hold());
    assert!(!s.kernel.status().is_running());
}

#[test]
fn test_a_snapshot_replays_the_same_transitions_and_its_bytes_decode_to_it() {
    let l = latency(1_500, 300, 2);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (2_100, dec!(100.01), dec!(100.03), dec!(5)),
            (4_100, dec!(100.02), dec!(100.06), dec!(5)),
        ],
    );
    let lane = perpetual(
        inst,
        &[(1_200, dec!(0.0001)), (3_200, dec!(-0.0002))],
        10_000,
    );
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut s: S = k.reset(3);
    k.step(&mut s, hedge(QuoteSide::Bid, 2));
    // A hedge in flight: copied, and through its bytes.
    assert!(s.lane.in_flight());
    let snapshot = s;
    let bytes = s.encode();
    let decoded = k.decode(&bytes).unwrap();
    assert_eq!(decoded.encode(), bytes);
    let quote = KernelAction::Quote {
        bid: 10_000,
        ask: 10_002,
        bid_lots: 1,
        ask_lots: 1,
    };
    let plan = [
        hold(),
        HedgedAction::Kernel(quote),
        hedge(QuoteSide::Ask, 1),
        hold(),
        hold(),
    ];
    let mut runs = Vec::new();
    for start in [snapshot, decoded] {
        let mut t = start;
        let mut trace = Vec::new();
        for action in plan {
            let out = k.step(&mut t, action);
            trace.push((out.kernel.applied, out.hedge, t.encode()));
        }
        runs.push((trace, k.summary(&t).unwrap()));
    }
    assert_eq!(runs[0], runs[1]);
    assert!(runs[0].1.funding != dec!(0));
}

#[test]
fn test_bytes_that_are_not_a_state_of_this_kernel_are_refused() {
    let l = latency(1_500, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst.clone()), &rows).unwrap();
    let mut s: S = k.reset(7);
    k.step(&mut s, hedge(QuoteSide::Bid, 2));
    let bytes = s.encode();
    assert!(k.decode(&bytes).is_ok());
    // Truncated, or with a byte more.
    assert!(k.decode(&bytes[..bytes.len() - 1]).is_err());
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(k.decode(&longer).is_err());
    // Another lane's kernel.
    let other: K<'_> = HedgedKernel::new(
        &cfg,
        &data,
        &spot(inst).with_combined(CombinedLimits {
            max_net_exposure: Some(dec!(9)),
            max_gross_notional: None,
        }),
        &rows,
    )
    .unwrap();
    assert!(other.decode(&bytes).is_err());
    // Reserved lots that differ from the hedge in flight. From the end: 20
    // counters, the closed flag, the agent's lots, two positions (two
    // decimals of 17 bytes and a u64 each), three ledger values, the mark
    // (flag and u64), the live sell lots, then the live buy lots.
    let live_buy = bytes.len() - (8 * 20 + 1 + 8 + 2 * (17 + 17 + 8) + 3 * 16 + 9 + 8 + 8);
    let mut broken = bytes.clone();
    broken[live_buy] ^= 1;
    let error = k.decode(&broken).unwrap_err().to_string();
    assert!(error.contains("reserved lots"), "{error}");
}

#[test]
fn test_a_touch_with_no_lots_cancels_and_a_rate_with_too_many_places_is_refused() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(0))]);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &spot(inst.clone()), &rows).unwrap();
    let mut s: S = k.reset(7);
    assert_eq!(k.step(&mut s, hedge(QuoteSide::Bid, 1)).hedge, Some(Ok(())));
    let c = s.lane.counters();
    assert_eq!((c.no_lots, c.moved_through, c.filled), (1, 0, 0));
    assert_eq!(s.lane.venue_position_lots(), 0);
    // Lot 1 and tick 0.01 leave 25 places for a rate: 26 are refused.
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let ok = perpetual(
        inst.clone(),
        &[(1_500, dec!(0.0000000000000000000000001))],
        1_000,
    );
    assert!(K::new(&cfg, &data, &ok, &rows).is_ok());
    let fine = perpetual(inst, &[(1_500, dec!(0.00000000000000000000000001))], 1_000);
    assert!(K::new(&cfg, &data, &fine, &rows).is_err());
}

/// Keeps the hedge lane's events and counts the quoted lane's.
#[derive(Default)]
struct HedgeSink {
    hedge: Vec<HedgeEvent>,
    quoted: usize,
}

impl EventSink for HedgeSink {
    fn event(&mut self, _event: KernelEvent) {
        self.quoted += 1;
    }

    fn hedge(&mut self, event: HedgeEvent) {
        self.hedge.push(event);
    }
}

#[test]
fn test_the_lane_sends_its_rows_fills_and_funding_in_its_own_order() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(
        &inst,
        &[
            (500, dec!(100.00), dec!(100.04), dec!(5)),
            (1_500, dec!(100.04), dec!(100.08), dec!(5)),
        ],
    );
    let lane = perpetual(inst, &[(1_500, dec!(0.0001))], 10_000);
    let k: K<'_> = HedgedKernel::new(&cfg, &data, &lane, &rows).unwrap();
    let mut sink = HedgeSink::default();
    let mut s: S = k.reset_with(7, &mut sink);
    k.step_with(&mut s, hedge(QuoteSide::Bid, 2), &mut sink);
    // Money scale 6 (tick 2, lot 0, fee 4). Buy 2 at the ask 100.04 at 1 001 ms, fee
    // 2 * 100.04 * 0.0005 = 0.10004. At 1 500 ms, funding first, at the mark of the row at
    // 500 ms, then that row: the long of 2 pays 0.020004.
    assert_eq!(
        sink.hedge,
        vec![
            HedgeEvent::Row {
                row: 0,
                ts_ns: 500 * MS,
                mid2: Some(20_004),
            },
            HedgeEvent::Fill {
                ts_ns: 1_001 * MS,
                side: Action::Buy,
                price: 10_004,
                lots: 2,
                fee_u: 100_040,
            },
            HedgeEvent::Funding {
                at_ns: 1_500 * MS,
                received_u: -20_004,
                mid2: 20_004,
            },
            HedgeEvent::Row {
                row: 1,
                ts_ns: 1_500 * MS,
                mid2: Some(20_012),
            },
        ]
    );
    assert!(sink.quoted > 0);
    // The ledger agrees with the events: fees and funding are their sums.
    let summary = k.summary(&s).unwrap();
    assert_eq!(summary.funding, dec!(-0.020004));
    assert_eq!(summary.fees, dec!(0.10004));
}

#[test]
fn test_a_hedged_kernel_refuses_a_balance_gate_on_either_lane() {
    let l = latency(1, 1, 0);
    let (cfg, data) = quoted(l);
    let inst = instrument(l);
    let rows = hedge_rows(&inst, &[(500, dec!(100.00), dec!(100.04), dec!(5))]);
    let gated = cfg.clone().with_margin(KernelMargin::new(dec!(0)));
    let error = K::new(&gated, &data, &spot(inst.clone()), &rows).unwrap_err();
    assert!(error.to_string().contains("balance gate"), "{error}");
    let lane = spot(inst.with_margin(KernelMargin::new(dec!(0))));
    let error = K::new(&cfg, &data, &lane, &rows).unwrap_err();
    assert!(error.to_string().contains("balance gate"), "{error}");
}
