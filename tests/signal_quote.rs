//! Signal composition and its ablation on seeded synthetic data.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use financial_types::Action;
use market_maker_rs::analytics::micro_price::{MicroPriceConfig, TopOfBook, calibrate_micro_price};
use market_maker_rs::analytics::micro_price::{SignalSource, StateStatus};
use market_maker_rs::analytics::order_flow::{Trade, TradeSide};
use market_maker_rs::analytics::vpin::VPINConfig;
use market_maker_rs::dec;
use market_maker_rs::prelude::Decimal;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use market_maker_rs::strategy::signal_quote::*;
use market_maker_rs::strategy::walk_forward::TimeWindow;

const TERMINAL: u64 = 1_000_000_000;

fn policy(weight: Decimal) -> FeaturePolicy {
    FeaturePolicy {
        weight,
        max_age_ms: 1_000,
        when_unavailable: UnavailablePolicy::Neutral,
    }
}

fn caps() -> SignalConfig {
    SignalConfig::baseline(dec!(0.05), dec!(0.03), dec!(0.5))
}

fn feature(kind: FeatureKind, value: Decimal, as_of_ms: u64) -> FeatureValue {
    FeatureValue {
        kind,
        value,
        as_of_ms,
    }
}

#[test]
fn test_adjustment_applies_weights_and_caps() {
    let cfg = caps()
        .with_imbalance(policy(dec!(0.02)))
        .with_micro_price(policy(dec!(0.01)))
        .with_toxicity(policy(dec!(0.02)));
    let values = [
        feature(FeatureKind::Imbalance, dec!(0.5), 10),
        feature(FeatureKind::MicroPrice, dec!(-0.5), 10),
        feature(FeatureKind::Toxicity, dec!(0.5), 10),
    ];
    let a = compute_adjustment(&values, 10, &cfg).unwrap();
    assert_eq!(a.skew, dec!(0.005));
    assert_eq!(a.widen, dec!(0.01));
    assert_eq!(a.size_factor, dec!(0.75));
    assert!(a.quoting);
    assert_eq!(a.imbalance, FeatureStatus::Fresh);

    // The caps bind in both directions.
    let strong = cfg
        .with_imbalance(policy(dec!(1)))
        .with_toxicity(policy(dec!(1)));
    let up = [
        feature(FeatureKind::Imbalance, dec!(1), 10),
        feature(FeatureKind::Toxicity, dec!(1), 10),
    ];
    let a = compute_adjustment(&up, 10, &strong).unwrap();
    assert_eq!(
        (a.skew, a.widen, a.size_factor),
        (dec!(0.05), dec!(0.03), dec!(0.5))
    );
    let down = [feature(FeatureKind::Imbalance, dec!(-1), 10)];
    assert_eq!(
        compute_adjustment(&down, 10, &strong).unwrap().skew,
        dec!(-0.05)
    );
}

#[test]
fn test_adjustment_ablated_feature_has_no_effect() {
    let values = [
        feature(FeatureKind::Imbalance, dec!(1), 10),
        feature(FeatureKind::MicroPrice, dec!(1), 10),
        feature(FeatureKind::Toxicity, dec!(1), 10),
    ];
    let a = compute_adjustment(&values, 10, &caps()).unwrap();
    assert_eq!(
        (a.skew, a.widen, a.size_factor),
        (dec!(0), dec!(0), dec!(1))
    );
    assert_eq!(a.imbalance, FeatureStatus::Off);
    assert_eq!(a.micro_price, FeatureStatus::Off);
    assert_eq!(a.toxicity, FeatureStatus::Off);
}

#[test]
fn test_adjustment_missing_and_stale_apply_the_policy() {
    let neutral = caps().with_imbalance(policy(dec!(0.02)));
    let old = [feature(FeatureKind::Imbalance, dec!(1), 10)];
    // Exactly at the limit is still fresh.
    let a = compute_adjustment(&old, 1_010, &neutral).unwrap();
    assert_eq!((a.imbalance, a.skew), (FeatureStatus::Fresh, dec!(0.02)));
    let a = compute_adjustment(&old, 1_011, &neutral).unwrap();
    assert_eq!(
        (a.imbalance, a.skew, a.quoting),
        (FeatureStatus::Stale, dec!(0), true)
    );
    let a = compute_adjustment(&[], 10, &neutral).unwrap();
    assert_eq!(
        (a.imbalance, a.skew, a.quoting),
        (FeatureStatus::Missing, dec!(0), true)
    );

    let stop = caps().with_imbalance(FeaturePolicy {
        when_unavailable: UnavailablePolicy::StopQuoting,
        ..policy(dec!(0.02))
    });
    assert!(!compute_adjustment(&old, 1_011, &stop).unwrap().quoting);
    assert!(!compute_adjustment(&[], 10, &stop).unwrap().quoting);
    assert!(compute_adjustment(&old, 1_010, &stop).unwrap().quoting);
}

#[test]
fn test_adjustment_future_or_out_of_range_feature_is_refused() {
    let cfg = caps().with_imbalance(policy(dec!(0.02)));
    let future = [feature(FeatureKind::Imbalance, dec!(1), 11)];
    assert!(compute_adjustment(&future, 10, &cfg).is_err());
    for bad in [
        feature(FeatureKind::Imbalance, dec!(1.01), 1),
        feature(FeatureKind::Imbalance, dec!(-1.01), 1),
        feature(FeatureKind::MicroPrice, dec!(1.01), 1),
        feature(FeatureKind::Toxicity, dec!(-0.01), 1),
        feature(FeatureKind::Toxicity, dec!(1.01), 1),
    ] {
        assert!(compute_adjustment(&[bad], 10, &cfg).is_err(), "{bad:?}");
    }
}

#[test]
fn test_adjustment_invalid_config_is_refused() {
    for bad in [
        SignalConfig::baseline(dec!(-1), dec!(0), dec!(0.5)),
        SignalConfig::baseline(dec!(0), dec!(-1), dec!(0.5)),
        SignalConfig::baseline(dec!(0), dec!(0), dec!(-0.1)),
        SignalConfig::baseline(dec!(0), dec!(0), dec!(1.1)),
        caps().with_toxicity(policy(dec!(-1))),
    ] {
        assert!(compute_adjustment(&[], 0, &bad).is_err(), "{bad:?}");
    }
}

fn top(timestamp_ms: u64, bid: Decimal, bid_size: Decimal, ask_size: Decimal) -> MarketEvent {
    MarketEvent::Quote(TopOfBook {
        timestamp_ms,
        bid,
        ask: bid + dec!(0.01),
        bid_size,
        ask_size,
    })
}

fn trade(timestamp: u64, price: Decimal, size: Decimal, side: TradeSide) -> MarketEvent {
    MarketEvent::Trade(Trade::new(price, size, side, timestamp))
}

fn sim() -> SimulationConfig {
    SimulationConfig::new(
        dec!(0.1),
        dec!(1),
        dec!(20),
        TERMINAL,
        dec!(1),
        RiskLimits::new(dec!(10), dec!(100000), dec!(1)).unwrap(),
        VPINConfig::new(dec!(5), 2, dec!(0.7)).unwrap(),
    )
}

#[test]
fn test_simulate_attribution_matches_a_hand_ledger() {
    // Buy 1 at our bid, the mid then moves up 0.5, and the fee is 10 bps.
    let (a_bid, _) = calculate_optimal_quotes(
        dec!(100.005),
        dec!(0),
        dec!(0.1),
        dec!(1),
        TERMINAL,
        dec!(20),
    )
    .unwrap();
    let bid = (a_bid / dec!(0.01)).floor() * dec!(0.01);
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(10, bid, dec!(3), TradeSide::Sell),
        top(20, dec!(100.50), dec!(1), dec!(1)),
    ];
    let cfg = sim().with_fee_rate(dec!(0.001)).with_markout_horizon_ms(5);
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!(r.decisions, 2);
    assert_eq!(
        (r.fills, r.volume, r.final_inventory),
        (1, dec!(1), dec!(1))
    );
    assert_eq!(r.max_abs_inventory, dec!(1));
    assert_eq!(r.spread_capture, dec!(100.005) - bid);
    assert_eq!(r.inventory_pnl, dec!(0.5));
    assert_eq!(r.fees, bid * dec!(0.001));
    // Cash -bid plus 1 at the last mid, less the fee.
    assert_eq!(r.total_pnl, dec!(100.505) - bid - bid * dec!(0.001));
    // The mid 5 ms after the fill is still the first one.
    assert_eq!(r.markout, Some(dec!(100.005) - bid));

    let later = sim().with_markout_horizon_ms(10);
    let r = simulate(&events, &later, &caps(), None).unwrap();
    assert_eq!(r.markout, Some(dec!(100.505) - bid));
    let past = sim().with_markout_horizon_ms(11);
    assert_eq!(
        simulate(&events, &past, &caps(), None).unwrap().markout,
        None
    );
}

#[test]
fn test_simulate_sell_side_and_partial_fill_use_the_ask() {
    let (_, a_ask) = calculate_optimal_quotes(
        dec!(100.005),
        dec!(0),
        dec!(0.1),
        dec!(1),
        TERMINAL,
        dec!(20),
    )
    .unwrap();
    let ask = (a_ask / dec!(0.01)).ceil() * dec!(0.01);
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        // Below the ask: not reached. Then two partial fills, then nothing left.
        trade(1, ask - dec!(0.01), dec!(1), TradeSide::Buy),
        trade(2, ask, dec!(0.4), TradeSide::Buy),
        trade(3, ask, dec!(5), TradeSide::Buy),
        trade(4, ask, dec!(5), TradeSide::Buy),
    ];
    let r = simulate(&events, &sim(), &caps(), None).unwrap();
    assert_eq!(
        (r.fills, r.volume, r.final_inventory),
        (2, dec!(1), dec!(-1))
    );
    assert_eq!(r.spread_capture, ask - dec!(100.005));
}

#[test]
fn test_simulate_latency_delays_the_quote_and_the_replacement() {
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(50, dec!(1), dec!(1), TradeSide::Sell),
        trade(100, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let none = simulate(&events, &sim(), &caps(), None).unwrap();
    assert_eq!(none.fills, 1, "the size is used up by the first trade");
    let slow = simulate(&events, &sim().with_latency_ms(100), &caps(), None).unwrap();
    assert_eq!(slow.fills, 1);
    assert_eq!(slow.volume, dec!(1));
    let slower = simulate(&events, &sim().with_latency_ms(101), &caps(), None).unwrap();
    assert_eq!(slower.fills, 0);
}

#[test]
fn test_simulate_limits_scale_and_refuse_the_side_that_adds_to_the_position() {
    let mut cfg = sim();
    cfg.limits = RiskLimits::new(dec!(1), dec!(100000), dec!(1)).unwrap();
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(5), TradeSide::Sell),
        top(2, dec!(100.00), dec!(1), dec!(1)),
        trade(3, dec!(1), dec!(5), TradeSide::Sell),
        trade(4, dec!(1000), dec!(5), TradeSide::Buy),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    // At the limit the bid is gone and the ask, which reduces, stays.
    assert_eq!((r.max_abs_inventory, r.final_inventory), (dec!(1), dec!(0)));
    assert_eq!(r.fills, 2);
}

#[test]
fn test_simulate_limits_scale_the_size_with_the_position() {
    // Scaling 1 at position 1 of 10 leaves 0.9 on the bid.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(5), TradeSide::Sell),
        top(2, dec!(100.00), dec!(1), dec!(1)),
        trade(3, dec!(1), dec!(5), TradeSide::Sell),
    ];
    let r = simulate(&events, &sim(), &caps(), None).unwrap();
    assert_eq!(r.volume, dec!(1.9));
}

#[test]
fn test_simulate_notional_limit_refuses_the_quote() {
    let mut cfg = sim();
    cfg.limits = RiskLimits::new(dec!(10), dec!(50), dec!(1)).unwrap();
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(5), TradeSide::Sell),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!((r.halted_decisions, r.fills), (1, 0));
}

#[test]
fn test_simulate_no_quote_at_terminal_and_the_resting_one_is_pulled() {
    let mut cfg = sim();
    cfg.terminal_ms = 100;
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(100, dec!(100.00), dec!(1), dec!(1)),
        trade(101, dec!(1), dec!(5), TradeSide::Sell),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!((r.decisions, r.fills), (1, 0));
}

#[test]
fn test_simulate_quotes_round_away_from_the_mid() {
    // At tick 1 the A-S bid near 99.95 rounds down to 99 and the ask to 101.
    let cfg = sim().with_tick_size(dec!(1));
    let run = |price, side| {
        let events = [
            top(0, dec!(100.00), dec!(1), dec!(1)),
            trade(1, price, dec!(1), side),
        ];
        simulate(&events, &cfg, &caps(), None).unwrap().fills
    };
    assert_eq!(run(dec!(99.5), TradeSide::Sell), 0);
    assert_eq!(run(dec!(99), TradeSide::Sell), 1);
    assert_eq!(run(dec!(100.5), TradeSide::Buy), 0);
    assert_eq!(run(dec!(101), TradeSide::Buy), 1);
}

#[test]
fn test_simulate_toxicity_reduces_the_size() {
    let mut cfg = sim();
    cfg.vpin = VPINConfig::new(dec!(1), 1, dec!(0.7)).unwrap();
    // Weight 0, so the toxicity cuts the size and does not widen. One sided
    // flow gives VPIN 1, so the factor is min_size_fraction.
    let on = caps().with_toxicity(policy(dec!(0)));
    let events = [
        trade(0, dec!(100), dec!(1), TradeSide::Buy),
        top(1, dec!(100.00), dec!(1), dec!(1)),
        trade(2, dec!(1), dec!(5), TradeSide::Sell),
    ];
    let r = simulate(&events, &cfg, &on, None).unwrap();
    assert_eq!(r.volume, dec!(0.5));
}

#[test]
fn test_simulate_total_matches_a_ledger_of_the_fills() {
    let (_, events) = scenario(13, 5_000);
    let cfg = sim().with_fee_rate(dec!(0.0001));
    let on = caps().with_imbalance(policy(dec!(0.03)));
    let run = simulate_detailed(&events, &cfg, &on, None).unwrap();
    let last_mid = events
        .iter()
        .rev()
        .find_map(|e| match e {
            MarketEvent::Quote(q) => Some(q.mid().unwrap()),
            _ => None,
        })
        .unwrap();
    let mut cash = dec!(0);
    let mut inventory = dec!(0);
    let mut fees = dec!(0);
    for fill in &run.fills {
        let sign = match fill.side {
            Action::Buy => dec!(1),
            Action::Sell => dec!(-1),
            other => panic!("{other:?}"),
        };
        cash -= sign * fill.size * fill.price;
        inventory += sign * fill.size;
        fees += fill.size * fill.price * dec!(0.0001);
    }
    assert!(run.fills.len() > 100);
    assert_eq!(inventory, run.report.final_inventory);
    assert_eq!(fees, run.report.fees);
    let total = cash + inventory * last_mid - fees;
    assert!((total - run.report.total_pnl).abs() < dec!(0.000000001));
}

#[test]
fn test_simulate_limit_holds_under_latency_and_the_reducing_side_stays() {
    let mut cfg = sim().with_latency_ms(150);
    cfg.limits = RiskLimits::new(dec!(1), dec!(100000), dec!(0)).unwrap();
    // Two decisions at inventory 0, each with a full bid, go live in turn.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(100, dec!(100.00), dec!(1), dec!(1)),
        trade(200, dec!(1), dec!(1), TradeSide::Sell),
        top(250, dec!(100.00), dec!(1), dec!(1)),
        trade(260, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let run = simulate_detailed(&events, &cfg, &caps(), None).unwrap();
    assert_eq!(run.report.final_inventory, dec!(1), "{:?}", run.fills);
    assert_eq!(run.report.max_abs_inventory, dec!(1));

    // A fill is cut to the room left, not refused: 0.5 of the second bid.
    let mut cfg = sim().with_latency_ms(150);
    cfg.limits = RiskLimits::new(dec!(1.5), dec!(100000), dec!(0)).unwrap();
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(100, dec!(100.00), dec!(1), dec!(1)),
        trade(200, dec!(1), dec!(1), TradeSide::Sell),
        top(250, dec!(100.00), dec!(1), dec!(1)),
        trade(260, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!(r.final_inventory, dec!(1.5));

    // Above the notional limit even after selling 1, the side that reduces is
    // still quoted, and the side that adds is not.
    let mut cfg = sim();
    cfg.limits = RiskLimits::new(dec!(10), dec!(250), dec!(0)).unwrap();
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(1), TradeSide::Sell),
        top(2, dec!(100.00), dec!(1), dec!(1)),
        trade(3, dec!(1), dec!(1), TradeSide::Sell),
        top(4, dec!(300.00), dec!(1), dec!(1)),
    ];
    let run = simulate_detailed(&events, &cfg, &caps(), None).unwrap();
    assert_eq!(run.report.final_inventory, dec!(2));
    let last = run.decisions.last().unwrap();
    assert!(last.ask.is_some() && last.bid.is_none(), "{last:?}");
}

#[test]
fn test_simulate_invalid_quote_is_counted_under_every_configuration() {
    let crossed = MarketEvent::Quote(TopOfBook {
        timestamp_ms: 1,
        bid: dec!(101),
        ask: dec!(100),
        bid_size: dec!(1),
        ask_size: dec!(1),
    });
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        crossed,
        top(2, dec!(100.00), dec!(1), dec!(1)),
    ];
    let (base, lean) = (
        simulate(&events, &sim(), &caps(), None).unwrap(),
        simulate(
            &events,
            &sim(),
            &caps().with_imbalance(policy(dec!(0.01))),
            None,
        )
        .unwrap(),
    );
    for r in [base, lean] {
        assert_eq!((r.decisions, r.invalid_quotes), (2, 1));
    }
}

#[test]
fn test_simulate_unformable_inventory_quote_is_a_halted_decision() {
    // Volatility so large that the A-S bid is not positive.
    let mut cfg = sim();
    cfg.volatility = dec!(100000);
    let events = [top(0, dec!(100.00), dec!(1), dec!(1))];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!((r.decisions, r.halted_decisions), (1, 1));
}

#[test]
fn test_simulate_micro_price_feature_moves_the_quotes_when_calibrated() {
    let quotes = small_move_quotes(5, 30_000);
    let split = 20_000usize;
    let window = TimeWindow::new(0, split as u64 * 100).unwrap();
    let record = calibrate_micro_price(
        &quotes,
        &window,
        &MicroPriceConfig::new(dec!(0.01), 5, 1),
        "d",
    )
    .unwrap();
    assert!(record.model().unwrap().calibrated_share() > dec!(0.9));
    let events: Vec<_> = quotes[split..]
        .iter()
        .map(|q| MarketEvent::Quote(*q))
        .collect();
    let on = caps().with_micro_price(policy(dec!(0.05)));
    let run = simulate_detailed(&events, &sim(), &on, Some(&record)).unwrap();
    // Normalization and coverage, against the model directly.
    let model = record.model().unwrap();
    let mut fresh = 0u64;
    for (decision, quote) in run.decisions.iter().zip(&quotes[split..]) {
        let signal = model.signal(quote).unwrap();
        let expected = if signal.source == SignalSource::MicroPrice {
            fresh += 1;
            (dec!(0.05) * signal.adjustment / dec!(0.005)).clamp(dec!(-0.05), dec!(0.05))
        } else {
            dec!(0)
        };
        assert_eq!(decision.adjustment.skew.round_dp(12), expected.round_dp(12));
    }
    assert_eq!(run.report.micro_price.fresh, fresh);
    assert_eq!(run.report.micro_price.missing, run.report.decisions - fresh);
    assert!(run.report.micro_price.fresh > 9_000);
    let skewed = run
        .decisions
        .iter()
        .filter(|d| !d.adjustment.skew.is_zero())
        .count();
    assert!(skewed > 1_000, "{skewed}");
    let base = simulate_detailed(&events, &sim(), &caps(), Some(&record)).unwrap();
    assert_ne!(run.decisions[0].bid, None);
    assert_ne!(run.decisions, base.decisions);
}

#[test]
fn test_simulate_terminal_pulls_quotes_whatever_the_event() {
    let mut cfg = sim();
    cfg.terminal_ms = 100;
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(100, dec!(1), dec!(1), TradeSide::Sell),
    ];
    assert_eq!(simulate(&events, &cfg, &caps(), None).unwrap().fills, 0);
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(102, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let slow = cfg.clone().with_latency_ms(101);
    assert_eq!(simulate(&events, &slow, &caps(), None).unwrap().fills, 0);
}

#[test]
fn test_simulate_record_trained_on_the_events_is_refused() {
    let quotes = vec![TopOfBook {
        timestamp_ms: 0,
        bid: dec!(100),
        ask: dec!(100.01),
        bid_size: dec!(1),
        ask_size: dec!(1),
    }];
    let window = TimeWindow::new(0, 100).unwrap();
    let record = calibrate_micro_price(
        &quotes,
        &window,
        &MicroPriceConfig::new(dec!(0.01), 5, 1),
        "d",
    )
    .unwrap();
    let on = caps().with_micro_price(policy(dec!(0.01)));
    let inside = [top(99, dec!(100.00), dec!(1), dec!(1))];
    assert!(simulate(&inside, &sim(), &on, Some(&record)).is_err());
    let after = [top(100, dec!(100.00), dec!(1), dec!(1))];
    assert!(simulate(&after, &sim(), &on, Some(&record)).is_ok());
}

#[test]
fn test_simulate_side_crossing_the_quote_is_not_posted_and_is_counted() {
    // Up: a skew of 0.5 puts the bid above the best ask; the ask stays.
    let huge = SignalConfig::baseline(dec!(0.5), dec!(0), dec!(1));
    let events = [
        top(0, dec!(100.00), dec!(9), dec!(1)),
        trade(1, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let up = huge.with_imbalance(policy(dec!(10)));
    let run = simulate_detailed(&events, &sim(), &up, None).unwrap();
    assert_eq!((run.report.crossing_sides, run.report.fills), (1, 0));
    assert!(run.decisions[0].bid.is_none() && run.decisions[0].ask.is_some());
    // Down: the mirror.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(9)),
        trade(1, dec!(1000), dec!(1), TradeSide::Buy),
    ];
    let run = simulate_detailed(&events, &sim(), &up, None).unwrap();
    assert_eq!((run.report.crossing_sides, run.report.fills), (1, 0));
    assert!(run.decisions[0].ask.is_none() && run.decisions[0].bid.is_some());
    // Touching counts: a bid exactly at the ask.
    let (base_bid, _) = calculate_optimal_quotes(
        dec!(100.005),
        dec!(0),
        dec!(0.1),
        dec!(1),
        TERMINAL,
        dec!(20),
    )
    .unwrap();
    let floor = (base_bid / dec!(0.01)).floor() * dec!(0.01);
    let exact = SignalConfig::baseline(dec!(100.01) - floor, dec!(0), dec!(1))
        .with_imbalance(policy(dec!(10)));
    let events = [top(0, dec!(100.00), dec!(9), dec!(1))];
    let run = simulate_detailed(&events, &sim(), &exact, None).unwrap();
    assert_eq!(run.report.crossing_sides, 1, "{:?}", run.decisions);
    let shifted_ask = TopOfBook {
        timestamp_ms: 0,
        bid: dec!(100.00),
        ask: dec!(100.01),
        bid_size: dec!(1),
        ask_size: dec!(9),
    };
    let (_, base_ask) = calculate_optimal_quotes(
        dec!(100.005),
        dec!(0),
        dec!(0.1),
        dec!(1),
        TERMINAL,
        dec!(20),
    )
    .unwrap();
    let ceil = (base_ask / dec!(0.01)).ceil() * dec!(0.01);
    let exact = SignalConfig::baseline(ceil - dec!(100.00), dec!(0), dec!(1))
        .with_imbalance(policy(dec!(10)));
    let run = simulate_detailed(&[MarketEvent::Quote(shifted_ask)], &sim(), &exact, None).unwrap();
    assert_eq!(run.report.crossing_sides, 1, "{:?}", run.decisions);
}

#[test]
fn test_simulate_post_only_applies_on_arrival_against_the_latest_book() {
    let cfg = sim().with_latency_ms(100);
    // The bid decided at 0 arrives at 100, above the ask of the book at 50.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(50, dec!(99.00), dec!(1), dec!(1)),
        trade(100, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!((r.fills, r.crossing_sides), (0, 1));
    // The mirror, for the ask.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(50, dec!(101.00), dec!(1), dec!(1)),
        trade(100, dec!(1000), dec!(1), TradeSide::Buy),
    ];
    let r = simulate(&events, &cfg, &caps(), None).unwrap();
    assert_eq!((r.fills, r.crossing_sides), (0, 1));
    // A quote that rests before the book moves is left as it is.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(50, dec!(99.00), dec!(1), dec!(1)),
        trade(60, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let r = simulate(&events, &sim().with_latency_ms(10), &caps(), None).unwrap();
    assert_eq!(r.crossing_sides, 0);
    assert_eq!(r.fills, 1);
}

#[test]
fn test_simulate_invalid_trade_is_counted_and_reaches_nothing() {
    let mut cfg = sim();
    cfg.vpin = VPINConfig::new(dec!(1), 1, dec!(0.7)).unwrap();
    let on = caps().with_toxicity(policy(dec!(0.02)));
    let clean = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(5, dec!(100.00), dec!(1), dec!(1)),
    ];
    for bad in [
        trade(1, dec!(1), dec!(-5), TradeSide::Sell),
        trade(1, dec!(1), dec!(0), TradeSide::Sell),
        trade(1, dec!(0), dec!(5), TradeSide::Sell),
        trade(1, dec!(-1), dec!(5), TradeSide::Sell),
    ] {
        let events = [clean[0].clone(), bad.clone(), clean[1].clone()];
        let r = simulate(&events, &cfg, &on, None).unwrap();
        assert_eq!((r.invalid_trades, r.fills), (1, 0), "{bad:?}");
        assert_eq!(r.toxicity.fresh, 0, "{bad:?}");
    }
}

#[test]
fn test_simulate_newest_due_quote_rests() {
    // With latency 100, both are due at 200; the fill is at the second's price.
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        top(50, dec!(110.00), dec!(1), dec!(1)),
        trade(200, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let run = simulate_detailed(&events, &sim().with_latency_ms(100), &caps(), None).unwrap();
    assert_eq!(run.fills[0].price, run.decisions[1].bid.unwrap().0);
    assert_ne!(
        run.decisions[0].bid.unwrap().0,
        run.decisions[1].bid.unwrap().0
    );
}

#[test]
fn test_simulate_reduction_flipping_to_the_same_size_is_exempt() {
    // From +1, selling 2 ends at -1: not an increase, at a price where a marked
    // position of 1 is already above the notional limit.
    let mut cfg = sim();
    cfg.base_size = dec!(2);
    cfg.limits = RiskLimits::new(dec!(10), dec!(250), dec!(0)).unwrap();
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(1), TradeSide::Sell),
        top(2, dec!(300.00), dec!(1), dec!(1)),
    ];
    let run = simulate_detailed(&events, &cfg, &caps(), None).unwrap();
    assert_eq!(run.report.final_inventory, dec!(1));
    let last = run.decisions.last().unwrap();
    assert_eq!(last.ask.unwrap().1, dec!(2), "{last:?}");
}

#[test]
fn test_simulate_stop_policy_halts_quoting() {
    let stop = caps().with_micro_price(FeaturePolicy {
        when_unavailable: UnavailablePolicy::StopQuoting,
        ..policy(dec!(0.01))
    });
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(1, dec!(1), dec!(1), TradeSide::Sell),
    ];
    let r = simulate(&events, &sim(), &stop, None).unwrap();
    assert_eq!((r.decisions, r.halted_decisions, r.fills), (1, 1, 0));
    assert_eq!(r.micro_price.missing, 1);
}

#[test]
fn test_simulate_toxicity_uses_only_trades_already_seen() {
    // One bucket of 1 per trade: a single seen trade gives a VPIN.
    let mut cfg = sim();
    cfg.vpin = VPINConfig::new(dec!(1), 1, dec!(0.7)).unwrap();
    let on = caps().with_toxicity(policy(dec!(0.02)));
    let events = [
        top(0, dec!(100.00), dec!(1), dec!(1)),
        trade(10, dec!(100), dec!(1), TradeSide::Buy),
        top(10, dec!(100.00), dec!(1), dec!(1)),
        top(2_000, dec!(100.00), dec!(1), dec!(1)),
    ];
    let r = simulate(&events, &cfg, &on, None).unwrap();
    assert_eq!(r.toxicity.missing, 1, "before the trade");
    assert_eq!(r.toxicity.fresh, 1, "after the trade");
    assert_eq!(r.toxicity.stale, 1, "two seconds later");

    // Flipping the trade reaches only the decisions after it.
    let mut flipped = events.to_vec();
    flipped[1] = trade(10, dec!(100), dec!(5), TradeSide::Buy);
    let one = simulate_detailed(&events, &cfg, &on, None).unwrap();
    let two = simulate_detailed(&flipped, &cfg, &on, None).unwrap();
    assert_eq!(one.decisions[0], two.decisions[0]);
}

#[test]
fn test_simulate_later_events_do_not_change_earlier_decisions_or_fills() {
    let events = scenario(11, 2_000).1;
    let cut = 1_000;
    let cut_ms = match &events[cut] {
        MarketEvent::Quote(q) => q.timestamp_ms,
        MarketEvent::Trade(t) => t.timestamp,
        _ => panic!("unknown event"),
    };
    let on = caps()
        .with_imbalance(policy(dec!(0.03)))
        .with_toxicity(FeaturePolicy {
            max_age_ms: 60_000,
            ..policy(dec!(0.02))
        });
    let mut changed = events.clone();
    for event in changed.iter_mut().skip(cut) {
        if let MarketEvent::Trade(t) = event {
            // Out of reach of any quote, and on the other side.
            (t.side, t.price) = match t.side {
                TradeSide::Buy => (TradeSide::Sell, dec!(1000000)),
                TradeSide::Sell => (TradeSide::Buy, dec!(1)),
            };
        }
    }
    let one = simulate_detailed(&events, &sim(), &on, None).unwrap();
    let two = simulate_detailed(&changed, &sim(), &on, None).unwrap();
    let before = |run: &SimulationRun| {
        let d: Vec<_> = run
            .decisions
            .iter()
            .filter(|d| d.timestamp_ms < cut_ms)
            .copied()
            .collect();
        let f: Vec<_> = run
            .fills
            .iter()
            .filter(|f| f.timestamp_ms < cut_ms)
            .copied()
            .collect();
        (d, f)
    };
    assert!(!before(&one).0.is_empty() && !before(&one).1.is_empty());
    assert_eq!(before(&one), before(&two));
    assert_ne!(one.fills, two.fills, "the change does reach later fills");
}

#[test]
fn test_simulate_invalid_inputs_are_refused() {
    let quote = top(5, dec!(100.00), dec!(1), dec!(1));
    let earlier = top(1, dec!(100.00), dec!(1), dec!(1));
    assert!(simulate(&[quote, earlier], &sim(), &caps(), None).is_err());
    assert!(simulate(&[], &sim().with_tick_size(dec!(0)), &caps(), None).is_err());
    assert!(simulate(&[], &sim().with_markout_horizon_ms(0), &caps(), None).is_err());
    let bad = SignalConfig::baseline(dec!(-1), dec!(0), dec!(0));
    assert!(simulate(&[], &sim(), &bad, None).is_err());
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// An adverse selection scenario. Between jumps of 12 ticks, the book leans
/// toward the coming jump for 20 quotes; just before the jump an informed trader
/// sweeps resting quotes up to 6 ticks from the mid, and in the lean some
/// one-sided trades arrive. Otherwise uninformed trades reach 7 ticks both ways.
/// Returns the top of book for fitting and the events.
fn scenario(seed: u64, steps: u64) -> (Vec<TopOfBook>, Vec<MarketEvent>) {
    let mut rng = Rng(seed);
    let (mut quotes, mut events) = (Vec::new(), Vec::new());
    let mut bid = 10_000i64;
    let mut lean = 0.0f64;
    let mut countdown = 0u64;
    for step in 0..steps {
        let time = step * 100;
        if countdown == 0 && rng.unit() < 0.02 {
            lean = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
            countdown = 20;
        }
        let imbalance = 0.5 + 0.4 * lean + 0.1 * (rng.unit() - 0.5);
        let bid_size = (imbalance * 100.0).round().clamp(1.0, 99.0) as u64;
        let bid_price = Decimal::from(bid) * dec!(0.01);
        let q = TopOfBook {
            timestamp_ms: time,
            bid: bid_price,
            ask: bid_price + dec!(0.01),
            bid_size: Decimal::from(bid_size),
            ask_size: Decimal::from(100 - bid_size),
        };
        quotes.push(q);
        events.push(MarketEvent::Quote(q));
        let mid = bid_price + dec!(0.005);
        let at = |ticks: i64| mid + Decimal::from(ticks) * dec!(0.01);
        if rng.unit() < 0.5 {
            let (price, side) = if rng.unit() < 0.5 {
                (at(7), TradeSide::Buy)
            } else {
                (at(-7), TradeSide::Sell)
            };
            events.push(trade(time + 10, price, dec!(1), side));
        }
        if countdown > 0 {
            countdown -= 1;
            if rng.unit() < 0.3 {
                let (price, side) = if lean > 0.0 {
                    (at(1), TradeSide::Buy)
                } else {
                    (at(-1), TradeSide::Sell)
                };
                events.push(trade(time + 20, price, dec!(1), side));
            }
            if countdown == 0 {
                let (price, side, jump) = if lean > 0.0 {
                    (at(6), TradeSide::Buy, 12)
                } else {
                    (at(-6), TradeSide::Sell, -12)
                };
                events.push(trade(time + 30, price, dec!(3), side));
                bid += jump;
                lean = 0.0;
            }
        }
    }
    (quotes, events)
}

fn ablations(model: bool) -> Vec<(&'static str, SignalConfig)> {
    let i = policy(dec!(0.03));
    let m = policy(dec!(0.03));
    let t = FeaturePolicy {
        max_age_ms: 60_000,
        ..policy(dec!(0.02))
    };
    let mut out = vec![
        ("baseline", caps()),
        ("imbalance", caps().with_imbalance(i)),
    ];
    if model {
        out.push(("micro_price", caps().with_micro_price(m)));
    }
    out.push(("toxicity", caps().with_toxicity(t)));
    if model {
        out.push((
            "all",
            caps()
                .with_imbalance(i)
                .with_micro_price(m)
                .with_toxicity(t),
        ));
    }
    out
}

#[test]
fn test_ablation_report_on_an_adverse_selection_scenario() {
    let (fit, _) = scenario(21, 20_000);
    let window = TimeWindow::new(0, 2_000_000).unwrap();
    let record = calibrate_micro_price(
        &fit,
        &window,
        &MicroPriceConfig::new(dec!(0.01), 5, 1),
        "fit",
    )
    .unwrap();
    let (_, events) = scenario(22, 20_000);
    let events: Vec<_> = events
        .into_iter()
        .map(|e| match e {
            MarketEvent::Quote(q) => MarketEvent::Quote(TopOfBook {
                timestamp_ms: q.timestamp_ms + 2_000_000,
                ..q
            }),
            MarketEvent::Trade(t) => MarketEvent::Trade(Trade {
                timestamp: t.timestamp + 2_000_000,
                ..t
            }),
            other => other,
        })
        .collect();

    println!(
        "| latency ms | fee bps | config | fills | final inv | max inv | capture | inventory | fees | total | markout | fresh i/m/t |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    // Stated result of this scenario: 12 tick jumps put the informative buckets
    // outside the quotes, so they fall back, and the rest carry no adjustment.
    assert!(
        record
            .model()
            .unwrap()
            .states()
            .iter()
            .any(|m| m.status == StateStatus::OutsideQuotes)
    );
    println!(
        "calibrated_share={}",
        record.model().unwrap().calibrated_share().round_dp(4)
    );
    for state in record.model().unwrap().states() {
        println!(
            "bucket={} status={:?} adjustment_ticks={} observations={}",
            state.imbalance_bucket,
            state.status,
            (state.adjustment / dec!(0.01)).round_dp(3),
            state.observations
        );
    }
    let mut results = Vec::new();
    for latency in [0u64, 50, 200] {
        for fee_bps in [0i64, 1] {
            let cfg = sim()
                .with_latency_ms(latency)
                .with_fee_rate(Decimal::from(fee_bps) / dec!(10000));
            for (name, signals) in ablations(true) {
                let r = simulate(&events, &cfg, &signals, Some(&record)).unwrap();
                println!(
                    "| {latency} | {fee_bps} | {name} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} |",
                    r.fills,
                    r.final_inventory,
                    r.max_abs_inventory,
                    r.spread_capture.round_dp(2),
                    r.inventory_pnl.round_dp(2),
                    r.fees.round_dp(2),
                    r.total_pnl.round_dp(2),
                    r.markout.unwrap().round_dp(4),
                    r.imbalance.fresh,
                    r.micro_price.fresh,
                    r.toxicity.fresh,
                );
                results.push((latency, fee_bps, name, r));
            }
        }
    }
    for seed in 30..35 {
        let (_, other) = scenario(seed, 20_000);
        let run = |signals: &SignalConfig| simulate(&other, &sim(), signals, None).unwrap();
        let (b, i) = (
            run(&caps()),
            run(&caps().with_imbalance(policy(dec!(0.03)))),
        );
        println!(
            "seed={seed} baseline_total={} imbalance_total={} baseline_markout={} imbalance_markout={}",
            b.total_pnl.round_dp(2),
            i.total_pnl.round_dp(2),
            b.markout.unwrap().round_dp(4),
            i.markout.unwrap().round_dp(4)
        );
    }
    let at = |latency, name: &str| {
        results
            .iter()
            .find(|(l, f, n, _)| *l == latency && *f == 0 && *n == name)
            .map(|(_, _, _, r)| *r)
            .unwrap()
    };
    // The fixed scenario, stated before the run: leaning away from the informed
    // sweep must reduce adverse selection compared with the same strategy.
    // A model's fallback counts as the feature missing, not as a zero signal.
    let micro = at(0, "micro_price");
    let usable = events
        .iter()
        .filter(|e| match e {
            MarketEvent::Quote(q) => {
                record.model().unwrap().signal(q).unwrap().source == SignalSource::MicroPrice
            }
            _ => false,
        })
        .count() as u64;
    assert_eq!(micro.micro_price.fresh, usable);
    assert_eq!(micro.micro_price.missing, micro.decisions - usable);
    assert!(micro.micro_price.missing > 0);
    let (base, lean) = (at(0, "baseline"), at(0, "imbalance"));
    assert!(
        lean.markout.unwrap() > base.markout.unwrap(),
        "{lean:?} {base:?}"
    );
    assert_eq!(lean.imbalance.fresh, lean.decisions);
    // Stale quotes: latency must not help the naive strategy on this data.
    assert!(at(200, "baseline").markout.unwrap() <= base.markout.unwrap());
}

/// A chain where imbalance predicts a one tick move, inside the quotes.
fn small_move_quotes(seed: u64, count: u64) -> Vec<TopOfBook> {
    let mut rng = Rng(seed);
    let (mut bid, mut imbalance) = (10_000i64, 0.5f64);
    let mut quotes = Vec::new();
    for step in 0..count {
        let bid_size = (imbalance * 100.0).round().clamp(1.0, 99.0) as u64;
        let price = Decimal::from(bid) * dec!(0.01);
        quotes.push(TopOfBook {
            timestamp_ms: step * 100,
            bid: price,
            ask: price + dec!(0.01),
            bid_size: Decimal::from(bid_size),
            ask_size: Decimal::from(100 - bid_size),
        });
        if rng.unit() < 0.1 {
            if rng.unit() < imbalance {
                bid += 1;
                imbalance = 0.3 + 0.2 * rng.unit();
            } else {
                bid -= 1;
                imbalance = 0.5 + 0.2 * rng.unit();
            }
        } else {
            imbalance = (imbalance + 0.08 * (rng.unit() - 0.5)).clamp(0.01, 0.99);
        }
    }
    quotes
}

#[test]
fn test_ablation_report_with_moves_inside_the_quotes() {
    // Fitted on the first 40,000 quotes. Uninformed trades reach 6 ticks either
    // way, so that the features act through the quotes' own distance.
    let quotes = small_move_quotes(41, 60_000);
    let split = 40_000usize;
    let window = TimeWindow::new(0, split as u64 * 100).unwrap();
    let record = calibrate_micro_price(
        &quotes,
        &window,
        &MicroPriceConfig::new(dec!(0.01), 5, 1),
        "d",
    )
    .unwrap();
    let share = record.model().unwrap().calibrated_share();
    println!("small_moves calibrated_share={}", share.round_dp(4));
    for state in record.model().unwrap().states() {
        println!(
            "small_moves bucket={} status={:?} adjustment_ticks={}",
            state.imbalance_bucket,
            state.status,
            (state.adjustment / dec!(0.01)).round_dp(3)
        );
    }
    let mut rng = Rng(43);
    let mut events = Vec::new();
    for q in &quotes[split..] {
        events.push(MarketEvent::Quote(*q));
        if rng.unit() < 0.5 {
            let mid = q.mid().unwrap();
            let (price, side) = if rng.unit() < 0.5 {
                (mid + dec!(0.06), TradeSide::Buy)
            } else {
                (mid - dec!(0.06), TradeSide::Sell)
            };
            events.push(trade(q.timestamp_ms + 10, price, dec!(1), side));
        }
    }
    let mut baseline = None;
    for (name, signals) in ablations(true) {
        for latency in [0u64, 200] {
            let cfg = sim().with_latency_ms(latency);
            let r = simulate(&events, &cfg, &signals, Some(&record)).unwrap();
            println!(
                "small_moves latency={latency} {name} fills={} max_inv={} capture={} inventory={} total={} markout={} fresh_micro={} crossing={}",
                r.fills,
                r.max_abs_inventory.round_dp(2),
                r.spread_capture.round_dp(2),
                r.inventory_pnl.round_dp(2),
                r.total_pnl.round_dp(2),
                r.markout.unwrap().round_dp(4),
                r.micro_price.fresh,
                r.crossing_sides,
            );
            if latency == 0 {
                match name {
                    "baseline" => baseline = Some(r),
                    "micro_price" => {
                        assert!(r.micro_price.fresh > 0);
                        assert_ne!(r.total_pnl, baseline.unwrap().total_pnl);
                    }
                    _ => {}
                }
            }
        }
    }
}

use market_maker_rs::risk::pretrade::{
    LimitsPolicy, OrderEvent, PretradeDecision, PretradePolicy, PretradeRequest,
};

/// Decides from a script, and remembers the life of each order.
#[derive(Debug, Default)]
struct Scripted {
    decide: Option<fn(&PretradeRequest) -> PretradeDecision>,
    fail_on_fill: bool,
    checks: Vec<String>,
    requests: Vec<PretradeRequest>,
    fills: Vec<OrderEvent>,
    events: Vec<String>,
    marks: u32,
    resets: u32,
    executions: Vec<String>,
    cumulative: Vec<Decimal>,
}

impl PretradePolicy for Scripted {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        self.checks.push(request.order_id.clone());
        self.requests.push(request.clone());
        self.decide
            .map_or(PretradeDecision::Accepted, |decide| decide(request))
    }
    fn on_market_data(&mut self, _mark: Decimal, _timestamp: u64) {
        self.marks += 1;
    }
    fn on_order_event(
        &mut self,
        event: &OrderEvent,
    ) -> market_maker_rs::types::error::MMResult<()> {
        if matches!(event, OrderEvent::Fill { .. }) {
            self.fills.push(event.clone());
        }
        let text = match event {
            OrderEvent::Accepted { order_id } => format!("accepted {order_id}"),
            OrderEvent::VenueRejected { order_id } => format!("rejected {order_id}"),
            OrderEvent::Canceled { order_id } => format!("canceled {order_id}"),
            OrderEvent::Fill {
                order_id,
                execution_id,
                quantity,
                cumulative_quantity,
                fee,
                ..
            } => {
                self.executions.push(execution_id.clone());
                self.cumulative.push(*cumulative_quantity);
                if self.fail_on_fill {
                    return Err(
                        market_maker_rs::types::error::MMError::InvalidPositionUpdate(
                            "no".to_string(),
                        ),
                    );
                }
                format!("fill {order_id} {quantity} {cumulative_quantity} {fee}")
            }
            other => format!("{other:?}"),
        };
        self.events.push(text);
        Ok(())
    }
    fn reset(&mut self) {
        self.resets += 1;
    }
}

fn gated_events() -> Vec<MarketEvent> {
    vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        trade(10, dec!(90), dec!(0.5), TradeSide::Sell),
        top(20, dec!(100), dec!(1), dec!(1)),
        trade(30, dec!(90), dec!(100), TradeSide::Sell),
        top(40, dec!(100), dec!(1), dec!(1)),
    ]
}

#[test]
fn test_simulate_gated_with_a_permissive_policy_equals_simulate() {
    {
        let events = gated_events();
        let plain = simulate_detailed(&events, &sim(), &caps(), None).unwrap();
        let mut policy = Scripted::default();
        let gated = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
        assert_eq!(plain, gated);
        assert!(plain.report.fills > 0);
        assert_eq!(policy.resets, 1);
        assert_eq!(policy.marks, 3);
    }
}

#[test]
fn test_simulate_gated_rejection_takes_the_side_off() {
    let events = gated_events();
    let plain = simulate(&events, &sim(), &caps(), None).unwrap();
    let mut policy = Scripted {
        decide: Some(|r| {
            if matches!(r.side, Action::Buy) {
                PretradeDecision::Rejected {
                    reason: "no".to_string(),
                }
            } else {
                PretradeDecision::Accepted
            }
        }),
        ..Scripted::default()
    };
    let run = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(run.report.fills, 0);
    assert_ne!(run.report.total_pnl, plain.total_pnl);
    assert_eq!(run.report.gate_rejections, run.report.decisions);
    assert!(
        run.decisions
            .iter()
            .all(|d| d.bid.is_none() && d.ask.is_some())
    );
}

#[test]
fn test_simulate_gated_reduction_changes_the_fill() {
    let events = gated_events();
    let plain = simulate(&events, &sim(), &caps(), None).unwrap();
    let mut policy = Scripted {
        decide: Some(|_| PretradeDecision::Reduced {
            quantity: dec!(0.25),
            reason: "smaller".to_string(),
        }),
        ..Scripted::default()
    };
    let run = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert!(run.report.volume < plain.volume, "{:?}", run.report);
    assert!(run.fills.iter().all(|f| f.size <= dec!(0.25)));
    assert_eq!(run.report.gate_reductions, 2 * run.report.decisions);
    assert!(run.report.gate_rejections == 0);
}

#[test]
fn test_simulate_gated_reduction_to_nothing_is_a_rejection() {
    let mut policy = Scripted {
        decide: Some(|_| PretradeDecision::Reduced {
            quantity: dec!(0),
            reason: "none".to_string(),
        }),
        ..Scripted::default()
    };
    let run = simulate_gated(&gated_events(), &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(run.report.fills, 0);
    assert_eq!(run.report.gate_reductions, 0);
    assert_eq!(run.report.gate_rejections, 2 * run.report.decisions);
}

#[test]
fn test_simulate_gated_tells_the_policy_each_order_life() {
    let events = vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        trade(10, dec!(90), dec!(0.5), TradeSide::Sell),
        top(20, dec!(100), dec!(1), dec!(1)),
    ];
    let mut policy = Scripted::default();
    let run = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(run.report.fills, 1);
    assert_eq!(
        policy.checks,
        ["sq-0-bid", "sq-0-ask", "sq-1-bid", "sq-1-ask"]
    );
    let fee = run.report.fees;
    assert_eq!(
        policy.events,
        vec![
            "accepted sq-0-bid".to_string(),
            "accepted sq-0-ask".to_string(),
            format!("fill sq-0-bid 0.5 0.5 {fee}"),
            "canceled sq-0-bid".to_string(),
            "canceled sq-0-ask".to_string(),
            "accepted sq-1-bid".to_string(),
            "accepted sq-1-ask".to_string(),
            "canceled sq-1-bid".to_string(),
            "canceled sq-1-ask".to_string(),
        ]
    );
}

#[test]
fn test_simulate_gated_a_quote_replaced_before_it_arrives_is_cancelled() {
    let events = vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        top(10, dec!(100), dec!(1), dec!(1)),
        top(200, dec!(100), dec!(1), dec!(1)),
    ];
    let mut policy = Scripted::default();
    simulate_gated(
        &events,
        &sim().with_latency_ms(100),
        &caps(),
        None,
        &mut policy,
    )
    .unwrap();
    let position = |text: &str| policy.events.iter().position(|e| e == text).unwrap();
    assert!(position("canceled sq-0-bid") < position("accepted sq-1-bid"));
    assert!(!policy.events.iter().any(|e| e == "accepted sq-0-bid"));
}

#[test]
fn test_simulate_gated_stops_sending_after_a_policy_fault() {
    let mut policy = Scripted {
        fail_on_fill: true,
        ..Scripted::default()
    };
    let run = simulate_gated(&gated_events(), &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(run.report.gate_faults, 1);
    assert_eq!(policy.checks.len(), 2);
    assert_eq!(run.report.gate_rejections, 2 * (run.report.decisions - 1));
    assert_eq!(run.report.fills, 1);
    assert!(run.decisions[1].bid.is_none() && run.decisions[1].ask.is_none());
}

#[test]
fn test_simulate_gated_limits_policy_holds_its_own_tighter_limit() {
    let events = gated_events();
    let mut policy = LimitsPolicy::new(RiskLimits::new(dec!(0.75), dec!(100000), dec!(1)).unwrap());
    let run = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert!(
        run.report.max_abs_inventory <= dec!(0.75),
        "{:?}",
        run.report
    );
    assert!(run.report.gate_reductions + run.report.gate_rejections > 0);
    // Nothing is left live, whatever happened.
    assert_eq!(policy.live_quantity(Action::Buy).unwrap(), dec!(0));
    assert_eq!(policy.live_quantity(Action::Sell).unwrap(), dec!(0));
    assert_eq!(policy.position(), run.report.final_inventory);
}

#[test]
fn test_simulate_gated_partial_fills_accumulate_under_their_own_executions() {
    let events = vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        trade(10, dec!(90), dec!(0.25), TradeSide::Sell),
        trade(11, dec!(90), dec!(0.25), TradeSide::Sell),
    ];
    let mut policy = Scripted::default();
    simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(policy.cumulative, [dec!(0.25), dec!(0.5)]);
    assert_eq!(policy.executions, ["sq-fill-0", "sq-fill-1"]);
}

#[test]
fn test_simulate_gated_a_side_that_crosses_on_arrival_is_rejected_to_the_policy() {
    let events = vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        top(50, dec!(50), dec!(1), dec!(1)),
        top(100, dec!(50), dec!(1), dec!(1)),
    ];
    let mut policy = Scripted::default();
    let cfg = sim().with_latency_ms(100);
    let run = simulate_gated(&events, &cfg, &caps(), None, &mut policy).unwrap();
    assert!(run.report.crossing_sides > 0);
    assert!(policy.events.iter().any(|e| e == "rejected sq-0-bid"));
    assert!(policy.events.iter().any(|e| e == "accepted sq-0-ask"));
    assert!(!policy.events.iter().any(|e| e == "accepted sq-0-bid"));
}

#[test]
fn test_simulate_gated_the_terminal_time_cancels_what_rests() {
    let events = vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        top(TERMINAL, dec!(100), dec!(1), dec!(1)),
    ];
    let mut policy = Scripted::default();
    simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(policy.checks.len(), 2);
    assert_eq!(
        policy.events,
        [
            "accepted sq-0-bid",
            "accepted sq-0-ask",
            "canceled sq-0-bid",
            "canceled sq-0-ask",
        ]
    );
}

#[cfg(feature = "margin")]
#[test]
fn test_simulate_gated_margin_policy_books_fills_and_cancels_cleanly() {
    use margin_rs::ledger::Ledger;
    use margin_rs::risk::{InstrumentLimits, RiskGate};
    use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};
    use market_maker_rs::backtest::MarginAccount;
    let quote = Asset::new("USDC").unwrap();
    let base = Asset::new("BTC").unwrap();
    let pair = InstrumentId::new("BTC-USDC").unwrap();
    let mut ledger = Ledger::default();
    ledger.register_asset(quote.clone(), 8).unwrap();
    ledger.register_asset(base.clone(), 8).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(pair.clone(), base.clone(), quote.clone(), 4, 4).unwrap(),
        )
        .unwrap();
    let mut gate = RiskGate::new(ledger);
    gate.deposit(TransferId::new("quote").unwrap(), quote, dec!(100000))
        .unwrap();
    gate.deposit(TransferId::new("base").unwrap(), base, dec!(100))
        .unwrap();
    gate.set_limits(
        &pair,
        InstrumentLimits {
            max_fee_rate: dec!(0.001),
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    let account = MarginAccount::new(gate, pair).unwrap();
    let mut policy = account.policy();
    let run = simulate_gated(&gated_events(), &sim(), &caps(), None, &mut policy).unwrap();
    assert!(run.report.fills > 0, "{:?}", run.report);
    assert_eq!(run.report.gate_faults, 0);
    assert!(account.errors().is_empty(), "{:?}", account.errors());
    assert!(account.fault().is_none());
    assert_eq!(
        account.with_gate(|g| g.ledger().journal().len()),
        run.fills.len()
    );
}

/// A bid and an ask filled, the ask twice, and the fee on each.
fn two_sided_events() -> Vec<MarketEvent> {
    vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        trade(10, dec!(90), dec!(0.5), TradeSide::Sell),
        trade(11, dec!(200), dec!(0.25), TradeSide::Buy),
        trade(12, dec!(200), dec!(0.25), TradeSide::Buy),
        top(20, dec!(100), dec!(1), dec!(1)),
    ]
}

#[test]
fn test_simulate_gated_tells_the_policy_each_whole_fill_on_both_sides() {
    let cfg = sim().with_fee_rate(dec!(0.0005));
    let mut policy = Scripted::default();
    let run = simulate_gated(&two_sided_events(), &cfg, &caps(), None, &mut policy).unwrap();
    assert_eq!(run.fills.len(), 3);
    let at = |i: usize| run.fills[i].price;
    let event = |i: usize, order_id: &str, side, quantity, cumulative_quantity, timestamp| {
        OrderEvent::Fill {
            order_id: order_id.to_string(),
            execution_id: format!("sq-fill-{i}"),
            side,
            price: at(i),
            quantity,
            cumulative_quantity,
            fee: at(i) * quantity * dec!(0.0005),
            fee_asset: market_maker_rs::types::fee::FeeAsset::Quote,
            timestamp,
        }
    };
    assert!(at(0) < dec!(100.005) && at(1) > dec!(100.005));
    assert_eq!(
        policy.fills,
        [
            event(0, "sq-0-bid", Action::Buy, dec!(0.5), dec!(0.5), 10),
            event(1, "sq-0-ask", Action::Sell, dec!(0.25), dec!(0.25), 11),
            event(2, "sq-0-ask", Action::Sell, dec!(0.25), dec!(0.5), 12),
        ]
    );
    let fees: Decimal = (0..3)
        .map(|i| at(i) * run.fills[i].size * dec!(0.0005))
        .sum();
    assert_eq!(fees, run.report.fees);
    assert!(fees > dec!(0));
}

#[test]
fn test_simulate_gated_each_request_is_a_quote_at_its_decision() {
    let mut policy = Scripted::default();
    let run = simulate_gated(&two_sided_events(), &sim(), &caps(), None, &mut policy).unwrap();
    let times: Vec<u64> = policy.requests.iter().map(|r| r.timestamp).collect();
    assert_eq!(times, [0, 0, 20, 20]);
    for request in &policy.requests {
        assert_eq!(
            request.purpose,
            market_maker_rs::risk::pretrade::OrderPurpose::Quote
        );
        assert_eq!(request.quantity, dec!(1));
    }
    let (bid, _) = run.decisions[0].bid.unwrap();
    let (ask, _) = run.decisions[0].ask.unwrap();
    assert_eq!(policy.requests[0].price, Some(bid));
    assert_eq!(policy.requests[1].price, Some(ask));
    assert_eq!(policy.requests[0].side, Action::Buy);
    assert_eq!(policy.requests[1].side, Action::Sell);
}

#[test]
fn test_simulate_gated_a_reduction_above_the_size_keeps_the_size() {
    let mut policy = Scripted {
        decide: Some(|_| PretradeDecision::Reduced {
            quantity: dec!(100),
            reason: "more".to_string(),
        }),
        ..Scripted::default()
    };
    let run = simulate_gated(&two_sided_events(), &sim(), &caps(), None, &mut policy).unwrap();
    assert_eq!(run.decisions[0].bid.unwrap().1, dec!(1));
    assert_eq!(run.decisions[0].ask.unwrap().1, dec!(1));
}

#[test]
fn test_simulate_gated_a_quote_in_flight_at_the_terminal_time_or_the_end_is_cancelled() {
    let mut policy = Scripted::default();
    let cfg = sim().with_latency_ms(100);
    let events = [
        top(0, dec!(100), dec!(1), dec!(1)),
        top(TERMINAL, dec!(100), dec!(1), dec!(1)),
    ];
    simulate_gated(&events, &cfg, &caps(), None, &mut policy).unwrap();
    assert_eq!(policy.events, ["canceled sq-0-bid", "canceled sq-0-ask"]);
    let mut policy = Scripted::default();
    simulate_gated(&events[..1], &cfg, &caps(), None, &mut policy).unwrap();
    assert_eq!(policy.events, ["canceled sq-0-bid", "canceled sq-0-ask"]);
}

#[test]
fn test_simulate_gated_after_a_fault_a_quote_already_out_can_still_fill() {
    let mut policy = Scripted {
        fail_on_fill: true,
        ..Scripted::default()
    };
    let events = [
        top(0, dec!(100), dec!(1), dec!(1)),
        trade(10, dec!(90), dec!(0.25), TradeSide::Sell),
        trade(11, dec!(90), dec!(0.25), TradeSide::Sell),
        top(20, dec!(100), dec!(1), dec!(1)),
        trade(21, dec!(90), dec!(0.25), TradeSide::Sell),
    ];
    let run = simulate_gated(&events, &sim(), &caps(), None, &mut policy).unwrap();
    // Both before the next decision, and that decision replaced it with nothing.
    assert_eq!(run.report.fills, 2);
    assert_eq!(run.report.gate_faults, 2);
    assert_eq!(policy.checks.len(), 2);
}

/// A quote replaced before it arrives, an ask filled, and one still in flight at the end.
fn latency_events() -> Vec<MarketEvent> {
    vec![
        top(0, dec!(100), dec!(1), dec!(1)),
        top(10, dec!(100), dec!(1), dec!(1)),
        top(40, dec!(100), dec!(1), dec!(1)),
        trade(45, dec!(200), dec!(0.5), TradeSide::Buy),
        top(70, dec!(100), dec!(1), dec!(1)),
        trade(75, dec!(90), dec!(0.25), TradeSide::Sell),
        top(80, dec!(100), dec!(1), dec!(1)),
    ]
}

#[test]
fn test_simulate_gated_limits_policy_stays_clean_under_latency() {
    let events = latency_events();
    let cfg = sim().with_latency_ms(30).with_fee_rate(dec!(0.0005));
    let plain = simulate_detailed(&events, &cfg, &caps(), None).unwrap();
    let mut policy = LimitsPolicy::new(RiskLimits::new(dec!(10), dec!(100000), dec!(1)).unwrap());
    let run = simulate_gated(&events, &cfg, &caps(), None, &mut policy).unwrap();
    assert_eq!(run, plain);
    assert!(run.fills.iter().any(|f| f.side == Action::Sell));
    assert_eq!(run.report.gate_faults, 0);
    assert_eq!(policy.live_quantity(Action::Buy).unwrap(), dec!(0));
    assert_eq!(policy.live_quantity(Action::Sell).unwrap(), dec!(0));
    assert_eq!(policy.position(), run.report.final_inventory);
}

#[cfg(feature = "margin")]
#[test]
fn test_simulate_gated_margin_policy_stays_clean_under_latency() {
    use margin_rs::ledger::Ledger;
    use margin_rs::risk::{InstrumentLimits, RiskGate};
    use margin_rs::{Asset, InstrumentId, SpotInstrument, TransferId};
    use market_maker_rs::backtest::MarginAccount;
    let quote = Asset::new("USDC").unwrap();
    let base = Asset::new("BTC").unwrap();
    let pair = InstrumentId::new("BTC-USDC").unwrap();
    let mut ledger = Ledger::default();
    ledger.register_asset(quote.clone(), 8).unwrap();
    ledger.register_asset(base.clone(), 8).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(pair.clone(), base.clone(), quote.clone(), 4, 4).unwrap(),
        )
        .unwrap();
    let mut gate = RiskGate::new(ledger);
    gate.deposit(TransferId::new("quote").unwrap(), quote, dec!(100000))
        .unwrap();
    gate.deposit(TransferId::new("base").unwrap(), base, dec!(100))
        .unwrap();
    gate.set_limits(
        &pair,
        InstrumentLimits {
            max_fee_rate: dec!(0.001),
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    let account = MarginAccount::new(gate, pair).unwrap();
    let mut policy = account.policy();
    let events = latency_events();
    let cfg = sim().with_latency_ms(30).with_fee_rate(dec!(0.0005));
    let run = simulate_gated(&events, &cfg, &caps(), None, &mut policy).unwrap();
    assert_eq!(
        run,
        simulate_detailed(&events, &cfg, &caps(), None).unwrap()
    );
    assert!(run.fills.iter().any(|f| f.side == Action::Sell));
    assert_eq!(run.report.gate_faults, 0);
    assert!(account.errors().is_empty(), "{:?}", account.errors());
    assert!(account.fault().is_none());
    assert_eq!(
        account.with_gate(|g| g.ledger().journal().len()),
        run.fills.len()
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_a_report_from_before_the_gate_reads_with_zero_gate_counts() {
    let report = simulate(&gated_events(), &sim(), &caps(), None).unwrap();
    let mut older = serde_json::to_value(report).unwrap();
    for field in ["gate_reductions", "gate_rejections", "gate_faults"] {
        older.as_object_mut().unwrap().remove(field).unwrap();
    }
    let read: SimulationReport = serde_json::from_value(older).unwrap();
    assert_eq!(read, report);
}
