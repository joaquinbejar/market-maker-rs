//! Residual corrections to a classical quote. All data here is synthetic.

#![cfg(feature = "rl")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    AvellanedaQuoter, ErrCode, FinancialKernel, GlftQuoter, KernelAction, KernelConfig,
    KernelDataset, Observation, QuoteAdapter, QuoteSide, QuoteSource, SourceQuote, encode,
};
use market_maker_rs::dec;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::rl::{
    EpisodeRecorder, ObservationScales, ResidualBounds, ResidualClips, ResidualDelta, ResidualEnv,
    ResidualFallback, ResidualProposal, RlConfig, compose,
};
use market_maker_rs::strategy::glft::GLFTConfig;
use market_maker_rs::types::error::MMResult;

const BOUNDS: ResidualBounds = ResidualBounds {
    price_ticks: 10,
    size_lots: 2,
};

const QUOTE: KernelAction = KernelAction::Quote {
    bid: 100,
    ask: 110,
    bid_lots: 3,
    ask_lots: 4,
};

fn proposal(bid_away: f64, ask_away: f64, bid_size: f64, ask_size: f64) -> ResidualProposal {
    ResidualProposal {
        bid_away,
        ask_away,
        bid_size,
        ask_size,
    }
}

fn run(action: KernelAction, p: ResidualProposal) -> market_maker_rs::rl::ResidualDecision {
    compose(action, p, BOUNDS, 1000, 5)
}

#[test]
fn test_compose_a_zero_residual_is_the_classical_action() {
    let quotes = [
        QUOTE,
        // Crossed, as the adapter keeps a crossed quote.
        KernelAction::Quote {
            bid: 110,
            ask: 100,
            bid_lots: 3,
            ask_lots: 4,
        },
        // Outside what compose would keep, untouched all the same.
        KernelAction::Quote {
            bid: 5000,
            ask: 6000,
            bid_lots: 50,
            ask_lots: 60,
        },
        KernelAction::OneSided {
            side: QuoteSide::Bid,
            price: 100,
            lots: 1,
        },
    ];
    for action in quotes {
        let decision = run(action, ResidualProposal::ZERO);
        assert_eq!(decision.composed, action);
        assert_eq!(decision.classical, action);
        assert_eq!(decision.delta, ResidualDelta::default());
        assert_eq!(decision.fallback, None);
        assert_eq!(decision.clips, ResidualClips::default());
    }
}

#[test]
fn test_compose_moves_each_side_away_from_the_mid_and_rounds_to_whole_units() {
    let decision = run(QUOTE, proposal(0.5, 0.26, 1.0, -0.5));
    assert_eq!(
        decision.delta,
        ResidualDelta {
            bid_away_ticks: 5,
            ask_away_ticks: 3,
            bid_lots: 2,
            ask_lots: -1,
        }
    );
    assert_eq!(
        decision.composed,
        KernelAction::Quote {
            bid: 95,
            ask: 113,
            bid_lots: 5,
            ask_lots: 3,
        }
    );
    assert_eq!(decision.fallback, None);
    assert_eq!(decision.clips, ResidualClips::default());
    // Negative values tighten, symmetrically.
    let tighter = run(QUOTE, proposal(-0.3, -0.3, 0.0, 0.0));
    assert_eq!(
        tighter.composed,
        KernelAction::Quote {
            bid: 103,
            ask: 107,
            bid_lots: 3,
            ask_lots: 4,
        }
    );
}

#[test]
fn test_compose_clips_a_proposal_beyond_its_bound_and_says_so() {
    let decision = run(QUOTE, proposal(7.0, -7.0, 0.0, 0.0));
    assert!(decision.clips.proposal);
    assert_eq!(decision.delta.bid_away_ticks, 10);
    assert_eq!(decision.delta.ask_away_ticks, -10);
    assert_eq!(decision.fallback, None);
    assert_eq!(
        decision.composed,
        KernelAction::Quote {
            bid: 90,
            ask: 100,
            bid_lots: 3,
            ask_lots: 4,
        }
    );
    for slot in 0..4 {
        let mut values = [0.0; 4];
        values[slot] = 1.5;
        let p = proposal(values[0], values[1], values[2], values[3]);
        assert!(run(QUOTE, p).clips.proposal, "slot {slot}");
    }
    assert!(!run(QUOTE, proposal(1.0, -1.0, 1.0, -1.0)).clips.proposal);
}

#[test]
fn test_compose_refuses_a_value_that_is_not_finite() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for slot in 0..4 {
            let mut values = [0.25; 4];
            values[slot] = bad;
            let p = proposal(values[0], values[1], values[2], values[3]);
            let decision = run(QUOTE, p);
            assert_eq!(decision.composed, QUOTE, "slot {slot}");
            assert_eq!(decision.fallback, Some(ResidualFallback::NonFinite));
            assert_eq!(decision.delta, ResidualDelta::default());
        }
    }
}

#[test]
fn test_compose_keeps_prices_and_sizes_in_range() {
    let low = KernelAction::Quote {
        bid: 3,
        ask: 998,
        bid_lots: 1,
        ask_lots: 5,
    };
    let decision = run(low, proposal(1.0, 1.0, -1.0, 1.0));
    assert_eq!(
        decision.composed,
        KernelAction::Quote {
            bid: 1,
            ask: 1000,
            bid_lots: 1,
            ask_lots: 5,
        }
    );
    assert!(decision.clips.price && decision.clips.size);
    assert!(!run(QUOTE, proposal(0.5, 0.5, 0.0, 0.0)).clips.price);
    let price_only = run(low, proposal(1.0, 0.0, 0.0, 0.0));
    assert!(price_only.clips.price && !price_only.clips.size);
    let ask_price = run(low, proposal(0.0, 1.0, 0.0, 0.0));
    assert!(ask_price.clips.price && !ask_price.clips.size);
    let bid_size = run(low, proposal(0.0, 0.0, -1.0, 0.0));
    assert!(bid_size.clips.size && !bid_size.clips.price);
    let ask_size = run(low, proposal(0.0, 0.0, 0.0, 1.0));
    assert!(ask_size.clips.size && !ask_size.clips.price);
}

#[test]
fn test_compose_falls_back_when_the_bid_would_reach_the_ask() {
    let tight = KernelAction::Quote {
        bid: 100,
        ask: 104,
        bid_lots: 3,
        ask_lots: 4,
    };
    let decision = run(tight, proposal(-0.2, -0.2, 0.5, 0.0));
    assert_eq!(decision.composed, tight);
    assert_eq!(decision.fallback, Some(ResidualFallback::Crossed));
    // One tick short of it.
    let close = run(tight, proposal(-0.2, -0.1, 0.0, 0.0));
    assert_eq!(close.fallback, None);
    assert_eq!(
        close.composed,
        KernelAction::Quote {
            bid: 102,
            ask: 103,
            bid_lots: 3,
            ask_lots: 4,
        }
    );
    let crossed = KernelAction::Quote {
        bid: 110,
        ask: 100,
        bid_lots: 3,
        ask_lots: 4,
    };
    let kept = run(crossed, proposal(1.0, 1.0, 0.0, 0.0));
    assert_eq!(kept.composed, crossed);
    assert_eq!(kept.fallback, Some(ResidualFallback::Crossed));
}

#[test]
fn test_compose_one_sided_takes_only_its_own_side() {
    let bid = KernelAction::OneSided {
        side: QuoteSide::Bid,
        price: 100,
        lots: 2,
    };
    assert_eq!(
        run(bid, proposal(0.5, 1.0, 0.5, -1.0)).composed,
        KernelAction::OneSided {
            side: QuoteSide::Bid,
            price: 95,
            lots: 3,
        }
    );
    let ask = KernelAction::OneSided {
        side: QuoteSide::Ask,
        price: 100,
        lots: 2,
    };
    assert_eq!(
        run(ask, proposal(1.0, 0.5, -1.0, 0.5)).composed,
        KernelAction::OneSided {
            side: QuoteSide::Ask,
            price: 105,
            lots: 3,
        }
    );
    let low = KernelAction::OneSided {
        side: QuoteSide::Ask,
        price: 999,
        lots: 1,
    };
    let clipped = run(low, proposal(0.0, 1.0, 0.0, -1.0));
    assert!(clipped.clips.price && clipped.clips.size);
}

#[test]
fn test_compose_has_nothing_to_correct_on_a_hold_or_a_strategy_error() {
    for action in [KernelAction::Hold, KernelAction::QuoteError(ErrCode(3))] {
        let decision = run(action, proposal(1.0, 1.0, 1.0, 1.0));
        assert_eq!(decision.composed, action);
        assert_eq!(decision.fallback, Some(ResidualFallback::NoClassicalQuote));
    }
}

fn scales() -> ObservationScales {
    ObservationScales {
        spread_ticks: dec!(10),
        mid_change_ticks: dec!(10),
        row_gap_ms: dec!(1000),
        size_lots: dec!(10),
        position: dec!(10),
        entry_gap: dec!(1),
    }
}

fn rows(count: u64) -> Vec<MarketTick> {
    (0..count)
        .map(|i| {
            let bid = Decimal::from(990 + (i64::try_from(i).unwrap() * 7) % 21);
            MarketTick::new(1_000 * (i + 1), bid, dec!(5), bid + dec!(2), dec!(5))
        })
        .collect()
}

fn avellaneda() -> AvellanedaQuoter {
    AvellanedaQuoter {
        risk_aversion: dec!(0.1),
        volatility: dec!(0.2),
        order_intensity: dec!(1.5),
        session_start_ms: 0,
        horizon_ms: 1_000_000,
        size: dec!(1),
    }
}

fn glft() -> GlftQuoter {
    GlftQuoter {
        config: GLFTConfig::new(
            dec!(0.01),
            dec!(0.3),
            dec!(0.9),
            30,
            dec!(1),
            1_000_000,
            dec!(0),
        )
        .unwrap(),
        volatility: dec!(5),
        size: dec!(1),
        session_start_ms: 0,
        form: market_maker_rs::backtest::kernel::GlftForm::Exact,
        counts: market_maker_rs::backtest::kernel::GlftCounts::default(),
    }
}

fn kernel_config() -> KernelConfig {
    KernelConfig::new(dec!(1), dec!(1), dec!(5000), dec!(10), 10_000).with_fee_rate(dec!(0.001))
}

/// A zero residual steps the kernel as the adapter alone does, state by state.
fn zero_reproduces<S: QuoteSource + Clone>(source: S) {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(150), &cfg).unwrap();
    let mut env: ResidualEnv<'_, S> = ResidualEnv::new(
        &cfg,
        &data,
        source.clone(),
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut adapter = QuoteAdapter::new(source, &kernel, dec!(1)).unwrap();
    let mut state = env.reset(1).unwrap();
    let mut direct = kernel.reset(1);
    let mut quotes = 0;
    while !state.is_done() {
        let action = adapter.act(&kernel, &kernel.observe(&direct));
        let step = env.step(&mut state, ResidualProposal::ZERO).unwrap();
        assert_eq!(step.decision.classical, action);
        assert_eq!(step.decision.composed, action);
        assert_eq!(step.decision.delta, ResidualDelta::default());
        kernel.step(&mut direct, action);
        assert_eq!(encode(state.kernel()), encode(&direct));
        quotes += usize::from(matches!(action, KernelAction::Quote { .. }));
    }
    assert!(quotes > 100, "only {quotes} quotes");
    assert!(state.kernel().counters().venue_fills > 0);
}

/// A source that remembers how often it quoted, and quotes further away each time.
#[derive(Debug, Clone, Default)]
struct Drifting {
    calls: u32,
}

impl QuoteSource for Drifting {
    fn quote(
        &mut self,
        _obs: &Observation,
        mid: Decimal,
        _inventory: Decimal,
    ) -> MMResult<SourceQuote> {
        self.calls += 1;
        let away = Decimal::from(1 + self.calls % 40);
        Ok(SourceQuote::Bid(mid - away, dec!(1)))
    }
}

/// One episode's classical actions, reset with or without a sink, as the bridge's worker
/// resets with one.
fn classical_actions(
    env: &mut ResidualEnv<'_, Drifting>,
    seed: u64,
    with_sink: bool,
) -> Vec<KernelAction> {
    let mut recorder = EpisodeRecorder::default();
    let mut state = if with_sink {
        env.reset_with(seed, &mut recorder).unwrap()
    } else {
        env.reset(seed).unwrap()
    };
    let mut actions = Vec::new();
    while !state.is_done() {
        let step = env.step(&mut state, ResidualProposal::ZERO).unwrap();
        actions.push(step.decision.classical);
    }
    actions
}

#[test]
fn test_env_reset_starts_a_stateful_source_again_so_a_seed_repeats() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(60), &cfg).unwrap();
    let mut env: ResidualEnv<'_, Drifting> = ResidualEnv::new(
        &cfg,
        &data,
        Drifting::default(),
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    let first = classical_actions(&mut env, 1, false);
    // The source does keep state within an episode.
    assert!(first.len() > 2);
    assert_ne!(first.first(), first.get(1));
    for with_sink in [false, true, true, false] {
        assert_eq!(classical_actions(&mut env, 1, with_sink), first);
    }
    // A reset in the middle of an episode starts it again too, either way.
    for with_sink in [false, true] {
        let mut state = env.reset(1).unwrap();
        env.step(&mut state, ResidualProposal::ZERO).unwrap();
        assert_eq!(classical_actions(&mut env, 1, with_sink), first);
    }
}

#[test]
fn test_env_zero_residual_reproduces_avellaneda_stoikov() {
    zero_reproduces(avellaneda());
}

#[test]
fn test_env_zero_residual_reproduces_glft() {
    zero_reproduces(glft());
}

#[test]
fn test_env_logs_every_component_and_reconciles_with_the_evaluation() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(100), &cfg).unwrap();
    let mut env: ResidualEnv<'_, AvellanedaQuoter> = ResidualEnv::new(
        &cfg,
        &data,
        avellaneda(),
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    assert_eq!(env.bounds(), BOUNDS);
    assert_eq!(env.spec_id(), env.env().spec().id());
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder).unwrap();
    assert_eq!(env.observe(&state).unwrap().row, Some(0));
    let p = proposal(-0.2, 0.2, 0.5, 0.0);
    while !state.is_done() {
        let expected = env.classical(&state);
        let step = env.step_with(&mut state, p, &mut recorder).unwrap();
        assert_eq!(step.decision.classical, expected);
        assert_eq!(step.decision.proposal, p);
        assert_eq!(step.decision.delta.bid_away_ticks, -2);
        assert_ne!(step.decision.composed, expected);
        assert_eq!(step.transition.reward.penalty, dec!(0));
        env.env().reconcile(&state, &recorder).unwrap();
    }
    assert!(env.step(&mut state, p).is_err());
}

#[test]
fn test_env_sends_the_composed_action_to_the_kernel() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(100), &cfg).unwrap();
    let mut env: ResidualEnv<'_, AvellanedaQuoter> = ResidualEnv::new(
        &cfg,
        &data,
        avellaneda(),
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut state = env.reset(1).unwrap();
    let mut direct = kernel.reset(1);
    let p = proposal(-0.2, 0.2, 0.5, 0.0);
    while !state.is_done() {
        let step = env.step(&mut state, p).unwrap();
        assert_ne!(step.decision.composed, step.decision.classical);
        kernel.step(&mut direct, step.decision.composed);
        assert_eq!(encode(state.kernel()), encode(&direct));
    }
}

#[test]
fn test_env_a_strategy_error_is_logged_and_taken_as_a_hold() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(3), &cfg).unwrap();
    let mut late = avellaneda();
    late.horizon_ms = 1;
    let mut env: ResidualEnv<'_, AvellanedaQuoter> = ResidualEnv::new(
        &cfg,
        &data,
        late,
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1)),
    )
    .unwrap();
    let mut state = env.reset(1).unwrap();
    let step = env.step(&mut state, proposal(1.0, 1.0, 1.0, 1.0)).unwrap();
    assert!(matches!(
        step.decision.composed,
        KernelAction::QuoteError(_)
    ));
    assert_eq!(
        step.decision.fallback,
        Some(ResidualFallback::NoClassicalQuote)
    );
    assert_eq!(state.kernel().venue_position(), dec!(0));
}

/// A stream of hostile values: extremes, values that are not finite and a walk.
fn hostile(step: u64) -> f64 {
    match step % 7 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => -1e300,
        3 => 1e300,
        4 => 1.0,
        5 => -1.0,
        _ => f64::from(u32::try_from(step % 97).unwrap()) / 97.0,
    }
}

#[test]
fn test_env_no_output_bypasses_the_limits() {
    let limits = RiskLimits::new(dec!(3), dec!(3200), dec!(1)).unwrap();
    let capped = kernel_config().with_limits(limits);
    let open = kernel_config();
    let ticks = rows(400);
    let capped_data = KernelDataset::from_ticks(&ticks, &capped).unwrap();
    let open_data = KernelDataset::from_ticks(&ticks, &open).unwrap();
    let wide = ResidualBounds {
        price_ticks: 40,
        size_lots: 100,
    };
    let rl = RlConfig::new(scales(), dec!(1));
    let mut limited: ResidualEnv<'_, AvellanedaQuoter> =
        ResidualEnv::new(&capped, &capped_data, avellaneda(), dec!(1), wide, rl).unwrap();
    let mut free: ResidualEnv<'_, AvellanedaQuoter> =
        ResidualEnv::new(&open, &open_data, avellaneda(), dec!(1), wide, rl).unwrap();
    let mut state = limited.reset(1).unwrap();
    let mut control = free.reset(1).unwrap();
    let mut step = 0;
    let mut largest_free = dec!(0);
    let mut largest_size = 0;
    while !state.is_done() {
        // No move of the prices, which a large one would turn into a crossed fallback.
        let p = proposal(
            0.0,
            0.0,
            hostile(step),
            hostile(step.wrapping_mul(3).wrapping_add(1)),
        );
        let t = limited.step(&mut state, p).unwrap();
        free.step(&mut control, p).unwrap();
        match t.decision.composed {
            KernelAction::Quote {
                bid_lots, ask_lots, ..
            } => {
                assert!(bid_lots <= 10 && ask_lots <= 10);
                largest_size = largest_size.max(bid_lots).max(ask_lots);
            }
            KernelAction::OneSided { lots, .. } => assert!(lots <= 10),
            KernelAction::Hold | KernelAction::QuoteError(_) | KernelAction::Cancel => {}
            other => panic!("the residual composed {other:?}"),
        }
        let position = state.kernel().venue_position().abs();
        assert!(position <= dec!(3), "position {position} at step {step}");
        largest_free = largest_free.max(control.kernel().venue_position().abs());
        step += 1;
    }
    assert!(largest_free > dec!(3), "the fixture never tests the limit");
    assert_eq!(
        largest_size, 10,
        "the fixture never reaches the largest order"
    );
}

#[test]
fn test_env_no_output_bypasses_the_notional_limit() {
    // Every mid is above 990, so a notional of 1500 allows one lot and not two.
    let limits = RiskLimits::new(dec!(100), dec!(1500), dec!(1)).unwrap();
    let capped = kernel_config().with_limits(limits);
    let data = KernelDataset::from_ticks(&rows(200), &capped).unwrap();
    let wide = ResidualBounds {
        price_ticks: 40,
        size_lots: 100,
    };
    let rl = RlConfig::new(scales(), dec!(1));
    let mut env: ResidualEnv<'_, AvellanedaQuoter> =
        ResidualEnv::new(&capped, &data, avellaneda(), dec!(1), wide, rl).unwrap();
    let mut state = env.reset(1).unwrap();
    let mut filled = false;
    while !state.is_done() {
        env.step(&mut state, proposal(-1.0, -1.0, 1.0, 1.0))
            .unwrap();
        let position = state.kernel().venue_position().abs();
        assert!(position <= dec!(1), "position {position}");
        filled |= !position.is_zero();
    }
    assert!(filled, "the fixture never reaches the limit");
}

#[test]
fn test_a_residual_environment_refuses_quote_lives_at_construction() {
    let cfg = kernel_config();
    let data = KernelDataset::from_ticks(&rows(3), &cfg).unwrap();
    let refused = ResidualEnv::<'_, AvellanedaQuoter>::new(
        &cfg,
        &data,
        avellaneda(),
        dec!(1),
        BOUNDS,
        RlConfig::new(scales(), dec!(1))
            .with_quote_lives(market_maker_rs::rl::QuoteLives::new(&[5]).unwrap()),
    );
    assert!(matches!(refused, Err(e) if e.to_string().contains("quote lives")));
}
