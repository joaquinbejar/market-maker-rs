//! Signal composition and its ablation on seeded synthetic data.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(clippy::arithmetic_side_effects)]

use market_maker_rs::analytics::micro_price::StateStatus;
use market_maker_rs::analytics::micro_price::{MicroPriceConfig, TopOfBook, calibrate_micro_price};
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
        cash -= fill.side * fill.size * fill.price;
        inventory += fill.side * fill.size;
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
    // A chain where imbalance predicts a one tick move, inside the quotes.
    let mut rng = Rng(5);
    let (mut bid, mut imbalance) = (10_000i64, 0.5f64);
    let mut quotes = Vec::new();
    for step in 0..30_000u64 {
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
    let split = 20_000usize;
    let window = TimeWindow::new(0, split as u64 * 100).unwrap();
    let record = calibrate_micro_price(
        &quotes,
        &window,
        &MicroPriceConfig::new(dec!(0.01), 5, 1),
        "d",
    )
    .unwrap();
    let model = record.model().expect("fitted");
    assert!(model.calibrated_share() > dec!(0.9));
    let events: Vec<_> = quotes[split..]
        .iter()
        .map(|q| MarketEvent::Quote(*q))
        .collect();
    let on = caps().with_micro_price(policy(dec!(0.05)));
    let run = simulate_detailed(&events, &sim(), &on, Some(model)).unwrap();
    assert!(run.report.micro_price.fresh > 9_000);
    let skewed = run
        .decisions
        .iter()
        .filter(|d| !d.adjustment.skew.is_zero())
        .count();
    assert!(skewed > 1_000, "{skewed}");
    let base = simulate_detailed(&events, &sim(), &caps(), Some(model)).unwrap();
    assert_ne!(run.decisions[0].bid, None);
    assert_ne!(run.decisions, base.decisions);
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
    let model = record.model().expect("fitted");
    let (_, events) = scenario(22, 20_000);

    println!(
        "| latency ms | fee bps | config | fills | final inv | max inv | capture | inventory | fees | total | markout | fresh i/m/t |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    // Stated result of this scenario: 12 tick jumps put the informative buckets
    // outside the quotes, so they fall back, and the rest carry no adjustment.
    assert!(
        model
            .states()
            .iter()
            .any(|m| m.status == StateStatus::OutsideQuotes)
    );
    println!("calibrated_share={}", model.calibrated_share().round_dp(4));
    for state in model.states() {
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
                let r = simulate(&events, &cfg, &signals, Some(model)).unwrap();
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
    let (base, lean) = (at(0, "baseline"), at(0, "imbalance"));
    assert!(
        lean.markout.unwrap() > base.markout.unwrap(),
        "{lean:?} {base:?}"
    );
    assert_eq!(lean.imbalance.fresh, lean.decisions);
    // Stale quotes: latency must not help the naive strategy on this data.
    assert!(at(200, "baseline").markout.unwrap() <= base.markout.unwrap());
}
