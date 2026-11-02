//! Behaviour of the Avellaneda-Stoikov baseline inside the causal simulator
//! (#85), reported separately from the mathematical validation in
//! `avellaneda_stoikov_fixtures.rs`.
//!
//! These checks are about execution, not profitability: with fees, a
//! pretrade position limit, latency and tick/lot rounding configured, every
//! submitted quote lies on the tick grid, quotes are never crossed, the
//! position never exceeds the limit, and the run replays identically. The
//! synthetic price path is deterministic; no claim is made about PnL.
//!
//! Scope note: the formal prerequisite #84 (validated replay, snapshot and
//! end to end equivalence) is not done yet. These tests rely on the causal
//! engine semantics already merged (persistent orders, fill model, latency,
//! accounting and pretrade ports) and on its own replay determinism; they do
//! not validate snapshot/restore or cross library equivalence, which remain
//! pending on #84.

use market_maker_rs::Decimal;
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestStrategy, ExecutionKind, MarketTick, SimLatencyConfig,
    SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::LimitsPolicy;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use market_maker_rs::strategy::quote::Quote;

const TICK: Decimal = dec!(0.01);
const MAX_POSITION: Decimal = dec!(3);

/// A-S quoting with a fixed horizon, inventory from the delivered position.
struct AsStrategy {
    gamma: Decimal,
    sigma: Decimal,
    k: Decimal,
    horizon_ms: u64,
}

impl BacktestStrategy for AsStrategy {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        let (bid, ask) = calculate_optimal_quotes(
            tick.mid_price(),
            position.quantity,
            self.gamma,
            self.sigma,
            self.horizon_ms,
            self.k,
        )
        .ok()?;
        Some(Quote {
            bid_price: bid,
            bid_size: dec!(1),
            ask_price: ask,
            ask_size: dec!(1),
            timestamp: tick.timestamp,
        })
    }
    fn on_fill(&mut self, _fill: &SimulatedFill) {}
    fn reset(&mut self) {}
}

/// Deterministic zig-zag path around 100 with a 0.10 wide book; every few
/// ticks the market jumps far enough to cross resting quotes.
fn ticks() -> Vec<MarketTick> {
    let offsets = [
        dec!(0),
        dec!(0.4),
        dec!(0.9),
        dec!(0.3),
        dec!(-0.5),
        dec!(-1.1),
        dec!(-0.4),
        dec!(0.2),
    ];
    (0..400u64)
        .map(|i| {
            let mid = dec!(100) + offsets[(i % 8) as usize];
            MarketTick::new(
                1_000 + i * 10,
                mid - dec!(0.05),
                dec!(5),
                mid + dec!(0.05),
                dec!(5),
            )
        })
        .collect()
}

fn engine() -> BacktestEngine<AsStrategy, VecDataSource> {
    let limits = RiskLimits::new(MAX_POSITION, dec!(1_000_000), dec!(0)).unwrap();
    BacktestEngine::new(
        BacktestConfig::default()
            .with_fee_rate(dec!(0.0005))
            .with_tick_size(TICK)
            .with_lot_size(dec!(0.001))
            .with_record_trades(true),
        AsStrategy {
            gamma: dec!(0.1),
            sigma: dec!(2),
            k: dec!(1.5),
            horizon_ms: 3_600_000,
        },
        VecDataSource::new(ticks()),
    )
    .with_latency(
        SimLatencyConfig::new()
            .with_observation_ms(2)
            .with_submit_ms(3)
            .with_cancel_ms(3)
            .with_ack_ms(2)
            .with_jitter(2, 7),
    )
    .unwrap()
    .with_pretrade_policy(LimitsPolicy::new(limits))
}

#[test]
fn quotes_respect_tick_grid_and_never_cross() {
    let mut engine = engine();
    let result = engine.run();
    assert!(result.num_trades > 0, "the path must exercise fills");
    let accepted: Vec<_> = engine
        .execution_reports()
        .iter()
        .filter(|r| r.kind == ExecutionKind::Accepted)
        .collect();
    assert!(!accepted.is_empty());
    for report in &accepted {
        assert_eq!(
            report.price % TICK,
            Decimal::ZERO,
            "off tick: {}",
            report.price
        );
    }
    // Each decision submits a bid below its ask.
    for pair in engine.pretrade_decisions().chunks(2) {
        if let [bid, ask] = pair {
            assert!(bid.request.price < ask.request.price);
        }
    }
}

#[test]
fn position_never_exceeds_the_pretrade_limit() {
    let mut engine = engine();
    let result = engine.run();
    assert!(
        result.max_position <= MAX_POSITION,
        "max {}",
        result.max_position
    );
    // Walk the fills to check the bound at every step, not only the peak
    // the engine tracks.
    let mut position = Decimal::ZERO;
    for fill in &result.trades {
        position += match fill.side {
            market_maker_rs::execution::Side::Buy => fill.quantity,
            market_maker_rs::execution::Side::Sell => -fill.quantity,
        };
        assert!(position.abs() <= MAX_POSITION);
    }
    assert_eq!(position, result.final_position);
}

#[test]
fn fees_are_charged_on_every_fill() {
    let mut engine = engine();
    let result = engine.run();
    let fees: Decimal = result.trades.iter().map(|f| f.fee).sum();
    assert_eq!(fees, result.total_fees);
    assert!(result.trades.iter().all(|f| f.fee > Decimal::ZERO));
}

#[test]
fn run_replays_identically() {
    let mut engine = engine();
    let first = engine.run();
    let first_reports = engine.execution_reports().to_vec();
    engine.reset();
    let second = engine.run();
    assert_eq!(first.net_pnl, second.net_pnl);
    assert_eq!(first.num_trades, second.num_trades);
    assert_eq!(first_reports, engine.execution_reports());
}
