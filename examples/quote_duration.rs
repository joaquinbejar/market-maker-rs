//! How long to hold a quote, in the continuous causal engine, under delays.
//!
//! Everything here is synthetic, and it decides nothing about any market. This is
//! a continuous order book with its own execution rules: it does not reproduce a
//! batch market, so its results must not be compared with results from one. No
//! teacher or predictor is learned, and the delays are inputs, not measurements
//! of any venue.
//!
//! # Duration as an action
//!
//! A classical quoter sets its prices by Avellaneda-Stoikov, with fixed
//! parameters. [`Hold::EveryTick`] requotes at every tick it sees, which is the
//! arm without duration control. [`Hold::For`] keeps a quote for a chosen time,
//! then requotes. A requote goes through the engine like any other, so the old
//! quote's cancel takes the simulated delay, and a fill before that stands.
//!
//! So [`Hold::For`] tests how often a quote is refreshed, not an expiry. The book
//! is not left empty when a duration ends, as a new quote replaces the old one, and
//! the replacement loses its place in time. A pricer with no price keeps the old
//! quote live.
//!
//! # Expiry
//!
//! [`Hold::Expire`] is a true expiry: a decision every `every_ms`, as under
//! `Hold::For(every_ms)`, whose quote lives for `live_ms` only. At the first tick
//! seen at or after that, the quoter stays silent and asks for its orders to be
//! cancelled, through [`BacktestStrategy::cancels_when_silent`], and nothing replaces
//! them until the next decision. So the two arms decide at the same rate, and differ
//! in how long the book holds a quote. The cancel takes the simulated delay like any
//! other, so a fill before it takes effect stands, and an order still in flight is
//! cancelled before it rests. Expiry is a request from the quoter, seen on ticks: it
//! is not an instruction held by the venue, which would need no cancel message.
//!
//! A quote rests for its life plus the cancel delay less the submit delay, so for
//! `live_ms` when these are equal, as in the reported runs. From a delay of `live_ms`
//! the cancel is requested by the time the quote is active, so every fill comes with
//! a cancel already on its way: late cancels then equal fills by construction, and
//! say nothing more.
//!
//! # Tape
//!
//! The mid is an efficient price plus a transient bounce, on a fixed clock. A fill
//! on a bounce earns the spread and a fill on a jump of the efficient price is
//! picked off, which gives each arm something to trade off. The parameters are
//! chosen to show that trade-off, not fitted to anything.
//!
//! # Protocol
//!
//! Every arm under a delay has the same tape, fees, fill model, limits and delay,
//! and no arm is selected, so no holdout is needed. All planned seeds are reported.
//! The net result's interval is a moving block bootstrap over fixed time buckets,
//! of the complete ones, and it is statistical only.
//!
//! # A limit of the design
//!
//! Every message has the same delay, on a fixed clock of ticks. Requoting at every
//! tick then leaves a quote live for one tick interval, from one round trip after
//! the tick it came from, with the end excluded and a venue message coming before
//! a tick at the same time. When twice the delay is a multiple of the clock, that
//! interval holds no tick, so the quote never fills. All the delays above zero
//! reported here are such delays, so that arm's zero fills come from this alignment
//! and do not hold for every delay.
//!
//! Run with: `cargo run --release --example quote_duration`

use market_maker_rs::Decimal;
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, ExecutionKind,
    ExecutionReport, MarketTick, SimLatencyConfig, SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::RiskLimits;
use market_maker_rs::risk::pretrade::LimitsPolicy;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use market_maker_rs::strategy::quote::Quote;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Milliseconds between ticks.
const TICK_MS: u64 = 10;
/// Ticks of a run: ten minutes.
const TAPE_TICKS: usize = 60_000;
/// Standard deviation of the efficient price per tick, in price units.
const DIFFUSION: f64 = 0.002;
/// Standard deviation of the transient bounce, in price units.
const BOUNCE: f64 = 0.03;
/// Probability of a jump of the efficient price per tick.
const JUMP_PROBABILITY: f64 = 0.002;
/// Size of a jump, in price units.
const JUMP: f64 = 0.2;
/// Half of the width of the touch, in price units.
const HALF_TOUCH: Decimal = dec!(0.005);
/// Risk aversion, in 1 / price units.
const GAMMA: Decimal = dec!(0.1);
/// Absolute volatility per √year, for a diffusion of `DIFFUSION` per tick.
const SIGMA: Decimal = dec!(112);
/// `k`, in 1 / price units.
const K: Decimal = dec!(30);
/// Time to terminal at each decision, in milliseconds.
const TERMINAL_MS: u64 = 60_000;
/// Size of a quote.
const ORDER_SIZE: Decimal = dec!(1);
/// Largest absolute position the limits allow.
const MAX_POSITION: Decimal = dec!(5);
/// Fee per unit of notional.
const FEE: Decimal = dec!(0.00001);
/// Horizon of the markout, in milliseconds.
const MARKOUT_MS: u64 = 1_000;
/// Length of a bucket of the net result, in milliseconds.
const BUCKET_MS: u64 = 10_000;
/// Planned seeds, all reported.
const SEEDS: [u64; 5] = [11, 23, 37, 53, 71];
/// Delays of every message, in milliseconds.
const DELAYS_MS: [u64; 4] = [0, 10, 50, 200];
/// Arms.
const HOLDS: [Hold; 4] = [
    Hold::EveryTick,
    Hold::For(50),
    Hold::For(250),
    Hold::For(1_000),
];
/// Arms with a true expiry, each at the decision rate of an arm of [`HOLDS`].
const EXPIRIES: [Hold; 2] = [
    Hold::Expire {
        live_ms: 50,
        every_ms: 250,
    },
    Hold::Expire {
        live_ms: 250,
        every_ms: 1_000,
    },
];
/// Bootstrap replicates.
const REPLICATES: usize = 200;
/// Buckets per bootstrap block.
const BLOCK_BUCKETS: usize = 6;

/// A small seeded generator, so that a run reproduces without a dependency.
struct Rng(u64);

impl Rng {
    /// SplitMix64.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize
    }

    /// A standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.unit()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// A tick from its mid, with the touch around it.
fn tick_at(timestamp: u64, mid: Decimal) -> MarketTick {
    MarketTick::new(
        timestamp,
        mid - HALF_TOUCH,
        dec!(5),
        mid + HALF_TOUCH,
        dec!(5),
    )
}

/// The tape of a seed, to the cent.
fn generate_tape(seed: u64, ticks: usize) -> Vec<MarketTick> {
    let mut rng = Rng(seed);
    let mut efficient = 100.0f64;
    (0..ticks)
        .map(|index| {
            efficient += DIFFUSION * rng.normal();
            if rng.unit() < JUMP_PROBABILITY {
                efficient += if rng.unit() < 0.5 { JUMP } else { -JUMP };
            }
            let mid = efficient + BOUNCE * rng.normal();
            let mid = Decimal::from_f64(mid).map_or(dec!(100), |m| m.round_dp(2));
            tick_at(index as u64 * TICK_MS, mid)
        })
        .collect()
}

/// How long a quote is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hold {
    /// Requote at every tick seen: no duration control.
    EveryTick,
    /// Keep a quote this long, in milliseconds of the ticks' time, then requote.
    For(u64),
    /// Decide every `every_ms`, and ask to cancel each quote after `live_ms` of the
    /// ticks' time, leaving the book empty from then until the next decision.
    Expire { live_ms: u64, every_ms: u64 },
}

/// Prices a quote from the mid and the delivered position.
type Pricer = fn(Decimal, Decimal) -> Option<(Decimal, Decimal)>;

/// The classical arm: Avellaneda-Stoikov with fixed parameters.
fn classical_prices(mid: Decimal, position: Decimal) -> Option<(Decimal, Decimal)> {
    calculate_optimal_quotes(mid, position, GAMMA, SIGMA, TERMINAL_MS, K).ok()
}

/// A quoter that holds its quotes, and what it heard from the engine.
#[derive(Clone)]
struct DurationQuoter {
    hold: Hold,
    pricer: Pricer,
    /// Time of the tick of the last quote.
    last_quote: Option<u64>,
    /// Time of the last tick seen.
    seen: u64,
    quotes: usize,
    reports: Vec<ExecutionKind>,
}

impl DurationQuoter {
    fn new(hold: Hold, pricer: Pricer) -> Self {
        Self {
            hold,
            pricer,
            last_quote: None,
            seen: 0,
            quotes: 0,
            reports: Vec::new(),
        }
    }

    fn count(&self, kind: ExecutionKind) -> usize {
        self.reports.iter().filter(|k| **k == kind).count()
    }

    fn due(&self, timestamp: u64) -> bool {
        match (self.hold, self.last_quote) {
            (Hold::EveryTick, _) | (_, None) => true,
            (Hold::For(hold_ms), Some(last)) => timestamp.saturating_sub(last) >= hold_ms,
            (Hold::Expire { every_ms, .. }, Some(last)) => {
                timestamp.saturating_sub(last) >= every_ms
            }
        }
    }

    /// Whether the last quote's life is over at this time.
    fn expired(&self, timestamp: u64) -> bool {
        match (self.hold, self.last_quote) {
            (Hold::Expire { live_ms, .. }, Some(last)) => timestamp.saturating_sub(last) >= live_ms,
            (Hold::EveryTick | Hold::For(_) | Hold::Expire { .. }, _) => false,
        }
    }
}

impl BacktestStrategy for DurationQuoter {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        self.seen = tick.timestamp;
        if !self.due(tick.timestamp) {
            return None;
        }
        let (bid_price, ask_price) = (self.pricer)(tick.mid_price(), position.quantity)?;
        self.last_quote = Some(tick.timestamp);
        self.quotes += 1;
        Some(Quote {
            bid_price,
            bid_size: ORDER_SIZE,
            ask_price,
            ask_size: ORDER_SIZE,
            timestamp: tick.timestamp,
        })
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn on_execution_report(&mut self, report: &ExecutionReport) {
        self.reports.push(report.kind);
    }

    fn cancels_when_silent(&self) -> bool {
        self.expired(self.seen)
    }

    fn reset(&mut self) {
        self.last_quote = None;
        self.seen = 0;
        self.quotes = 0;
        self.reports.clear();
    }
}

/// The same delay on every message.
fn delays(ms: u64) -> SimLatencyConfig {
    SimLatencyConfig::new()
        .with_observation_ms(ms)
        .with_submit_ms(ms)
        .with_cancel_ms(ms)
        .with_ack_ms(ms)
}

/// Runs one arm. Every arm has the same fees, tick, lot, limits and fill model.
fn run(
    ticks: Vec<MarketTick>,
    hold: Hold,
    pricer: Pricer,
    latency: SimLatencyConfig,
) -> (BacktestResult, DurationQuoter) {
    let config = BacktestConfig::default()
        .with_fee_rate(FEE)
        .with_tick_size(dec!(0.01))
        .with_lot_size(dec!(0.001))
        .with_record_trades(true)
        .with_record_equity_curve(true);
    let limits = RiskLimits::new(MAX_POSITION, dec!(1_000_000), dec!(0)).ok();
    let engine = BacktestEngine::new(
        config,
        DurationQuoter::new(hold, pricer),
        VecDataSource::new(ticks),
    );
    let engine = match limits {
        Some(limits) => engine.with_pretrade_policy(LimitsPolicy::new(limits)),
        None => engine,
    };
    let mut engine = match engine.with_latency(latency) {
        Ok(engine) => engine,
        Err(error) => panic!("delays within the limit are accepted: {error}"),
    };
    let result = engine.run();
    (result, engine.strategy().clone())
}

/// What a report shows of one run.
#[derive(Debug, Clone, PartialEq)]
struct Measures {
    fills: usize,
    quotes: usize,
    cancels: usize,
    late_cancels: usize,
    /// Mean of the side-signed move from the fill price to the mid `MARKOUT_MS`
    /// later, in price units; positive is good. Fills with no later mid are left out.
    markout: f64,
    /// Mean absolute position over the ticks.
    mean_inventory: f64,
    max_inventory: f64,
    net_result: f64,
    /// Net result over the complete buckets, which is what the interval is for.
    buckets_total: f64,
    low: f64,
    high: f64,
}

/// The mid at the first tick at or after `time`, if any. `ticks` are in order.
fn mid_at_or_after(ticks: &[MarketTick], time: u64) -> Option<Decimal> {
    let index = ticks.partition_point(|t| t.timestamp < time);
    ticks.get(index).map(MarketTick::mid_price)
}

/// Mean side-signed markout of the fills.
fn mean_markout(trades: &[SimulatedFill], ticks: &[MarketTick]) -> f64 {
    let moves: Vec<f64> = trades
        .iter()
        .filter_map(|trade| {
            let later = mid_at_or_after(ticks, trade.timestamp + MARKOUT_MS)?;
            let signed = match trade.side {
                Action::Buy => later - trade.price,
                _ => trade.price - later,
            };
            signed.to_f64()
        })
        .collect();
    mean_of(&moves)
}

/// Mean and largest absolute position over the ticks, from the fills.
fn inventory_over_ticks(trades: &[SimulatedFill], ticks: &[MarketTick]) -> (f64, f64) {
    let mut position = Decimal::ZERO;
    let mut next = 0;
    let (mut sum, mut max) = (0.0f64, 0.0f64);
    for tick in ticks {
        while let Some(trade) = trades.get(next).filter(|t| t.timestamp <= tick.timestamp) {
            position += match trade.side {
                Action::Buy => trade.quantity,
                _ => -trade.quantity,
            };
            next += 1;
        }
        let held = position.abs().to_f64().unwrap_or(f64::NAN);
        sum += held;
        max = max.max(held);
    }
    (sum / ticks.len().max(1) as f64, max)
}

/// The change of equity over each complete bucket of time, by the last equity
/// inside it. A partial last bucket is left out, so these sum to a little less
/// than the whole run.
fn bucket_changes(curve: &[(u64, Decimal)]) -> Vec<f64> {
    let mut changes = Vec::new();
    let Some((start, first)) = curve.first() else {
        return changes;
    };
    let (mut opening, mut latest) = (*first, *first);
    let mut bucket_end = start + BUCKET_MS;
    for (time, equity) in curve {
        if *time >= bucket_end {
            changes.push((latest - opening).to_f64().unwrap_or(f64::NAN));
            opening = latest;
            bucket_end += BUCKET_MS;
        }
        latest = *equity;
    }
    changes
}

/// A 95% interval of the sum, by a moving block bootstrap.
fn block_interval(values: &[f64], seed: u64) -> (f64, f64) {
    if values.len() < BLOCK_BUCKETS {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng(seed);
    let last_start = values.len() - BLOCK_BUCKETS + 1;
    let mut sums = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let mut sum = 0.0;
        let mut count = 0;
        while count < values.len() {
            let start = rng.below(last_start);
            for value in values[start..start + BLOCK_BUCKETS]
                .iter()
                .take(values.len() - count)
            {
                sum += value;
                count += 1;
            }
        }
        sums.push(sum);
    }
    sums.sort_by(f64::total_cmp);
    let at = |quantile: f64| sums[((REPLICATES - 1) as f64 * quantile).round() as usize];
    (at(0.025), at(0.975))
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Runs one arm under one delay on a tape, and measures it.
fn measure(ticks: &[MarketTick], hold: Hold, delay_ms: u64, seed: u64) -> Measures {
    let (result, strategy) = run(ticks.to_vec(), hold, classical_prices, delays(delay_ms));
    let (mean_inventory, max_inventory) = inventory_over_ticks(&result.trades, ticks);
    let buckets = bucket_changes(&result.equity_curve);
    let (low, high) = block_interval(&buckets, seed);
    Measures {
        fills: result.trades.len(),
        quotes: strategy.quotes,
        cancels: strategy.count(ExecutionKind::Canceled),
        late_cancels: strategy.count(ExecutionKind::CancelRejected),
        markout: mean_markout(&result.trades, ticks),
        mean_inventory,
        max_inventory,
        net_result: result.net_pnl.to_f64().unwrap_or(f64::NAN),
        buckets_total: buckets.iter().sum(),
        low,
        high,
    }
}

fn main() {
    println!(
        "SYNTHETIC ONLY: a continuous book, no market data, no batch market, nothing learned."
    );
    println!(
        "Config: {TAPE_TICKS} ticks of {TICK_MS} ms, diffusion {DIFFUSION}, bounce {BOUNCE}, jump \
         {JUMP} with probability {JUMP_PROBABILITY}, gamma {GAMMA}, sigma {SIGMA}, k {K}, terminal \
         {TERMINAL_MS} ms, size {ORDER_SIZE}, limit {MAX_POSITION}, fee {FEE}, markout {MARKOUT_MS} \
         ms, seeds {SEEDS:?}, {REPLICATES} replicates, blocks of {BLOCK_BUCKETS} buckets of \
         {BUCKET_MS} ms."
    );
    for delay_ms in DELAYS_MS {
        println!("\n== every message delayed {delay_ms} ms ==");
        for hold in HOLDS.into_iter().chain(EXPIRIES) {
            println!("{hold:?}");
            for seed in SEEDS {
                let ticks = generate_tape(seed, TAPE_TICKS);
                let m = measure(&ticks, hold, delay_ms, seed);
                println!(
                    "  seed {seed}: quotes {}, fills {}, cancels {}, late cancels {}, markout \
                     {:.4}, mean inventory {:.2}, max inventory {:.1}, net {:.2}, net over \
                     complete buckets {:.2} [{:.2}, {:.2}]",
                    m.quotes,
                    m.fills,
                    m.cancels,
                    m.late_cancels,
                    m.markout,
                    m.mean_inventory,
                    m.max_inventory,
                    m.net_result,
                    m.buckets_total,
                    m.low,
                    m.high,
                );
            }
        }
    }
    println!(
        "\nNo teacher or predictor was learned, nothing here is a batch market, and no real data \
         was read."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quotes one unit away from the mid, whatever the position.
    fn one_away(mid: Decimal, _position: Decimal) -> Option<(Decimal, Decimal)> {
        Some((mid - dec!(1), mid + dec!(1)))
    }

    fn never(_mid: Decimal, _position: Decimal) -> Option<(Decimal, Decimal)> {
        None
    }

    /// Quotes 99 and 101 at 0, then requotes at 100 to 98.5 and 100.5, so a mid of
    /// 98.9 crosses only the old bid. It comes at `cross_at`.
    fn race_tape(cross_at: u64) -> Vec<MarketTick> {
        vec![
            tick_at(0, dec!(100)),
            tick_at(100, dec!(99.5)),
            tick_at(cross_at, dec!(98.9)),
        ]
    }

    fn cancel_after(ms: u64) -> SimLatencyConfig {
        SimLatencyConfig::new().with_cancel_ms(ms)
    }

    #[test]
    fn test_fill_before_the_effective_cancel_stands() {
        for cross_at in [120, 149] {
            let (result, strategy) = run(
                race_tape(cross_at),
                Hold::For(100),
                one_away,
                cancel_after(50),
            );
            assert_eq!(result.trades.len(), 1, "{cross_at}");
            assert_eq!(result.trades[0].side, Action::Buy);
            assert_eq!(result.trades[0].price, dec!(99));
            assert_eq!(result.trades[0].timestamp, cross_at);
            assert_eq!(result.final_position, ORDER_SIZE);
            // The duration ran out first, and the fill still counts.
            assert_eq!(strategy.quotes, 2);
        }
    }

    #[test]
    fn test_a_tick_at_or_after_the_effective_cancel_finds_no_old_quote() {
        for (cross_at, cancel_ms) in [(150, 50), (151, 50), (120, 10), (110, 10)] {
            let (result, strategy) = run(
                race_tape(cross_at),
                Hold::For(100),
                one_away,
                cancel_after(cancel_ms),
            );
            assert!(result.trades.is_empty(), "{cross_at} {cancel_ms}");
            assert_eq!(strategy.count(ExecutionKind::Canceled), 2);
            assert_eq!(strategy.count(ExecutionKind::CancelRejected), 0);
        }
    }

    #[test]
    fn test_a_cancel_too_late_for_a_filled_order_is_rejected_and_the_fill_stays() {
        let (result, strategy) = run(race_tape(120), Hold::For(100), one_away, cancel_after(50));
        assert_eq!(result.trades.len(), 1);
        // A later tick lets the cancel's answer happen and be delivered.
        let mut ticks = race_tape(120);
        ticks.push(tick_at(200, dec!(99.5)));
        let (later, with_answer) = run(ticks, Hold::For(100), one_away, cancel_after(50));
        assert_eq!(later.trades.len(), 1);
        assert_eq!(later.trades[0].timestamp, 120);
        assert_eq!(later.final_position, ORDER_SIZE);
        assert_eq!(with_answer.count(ExecutionKind::CancelRejected), 1);
        assert_eq!(with_answer.count(ExecutionKind::Fill), 1);
        assert_eq!(strategy.count(ExecutionKind::Fill), 1);
    }

    #[test]
    fn test_a_requote_cancels_an_order_still_in_flight_before_it_rests() {
        // The first quote is active at 200, and its cancel at 110 comes before that.
        let latency = SimLatencyConfig::new()
            .with_submit_ms(200)
            .with_cancel_ms(10);
        let ticks = vec![
            tick_at(0, dec!(100)),
            tick_at(100, dec!(100)),
            tick_at(250, dec!(90)),
        ];
        let (result, strategy) = run(ticks, Hold::For(100), one_away, latency);
        // At 250 the first quote would cross, and it is gone. The second would be
        // active only from 300, and 250 also requotes.
        assert!(result.trades.is_empty(), "{:?}", result.trades);
        assert_eq!(strategy.quotes, 3);
    }

    #[test]
    fn test_a_quote_is_live_until_its_duration_runs_out() {
        // No requote before the hold is over, so the first quote takes this fill.
        let ticks = vec![
            tick_at(0, dec!(100)),
            tick_at(50, dec!(100)),
            tick_at(99, dec!(98.9)),
        ];
        let (result, strategy) = run(ticks, Hold::For(100), one_away, SimLatencyConfig::new());
        assert_eq!(strategy.quotes, 1);
        assert_eq!(result.trades.len(), 1);
        assert_eq!(strategy.count(ExecutionKind::Canceled), 0);
    }

    #[test]
    fn test_requote_comes_exactly_when_the_hold_is_over() {
        let mut quoter = DurationQuoter::new(Hold::For(100), one_away);
        let position = InventoryPosition::new();
        assert!(
            quoter
                .on_tick(&tick_at(1_000, dec!(100)), &position)
                .is_some()
        );
        assert!(
            quoter
                .on_tick(&tick_at(1_099, dec!(100)), &position)
                .is_none()
        );
        assert!(
            quoter
                .on_tick(&tick_at(1_100, dec!(100)), &position)
                .is_some()
        );
        assert!(
            quoter
                .on_tick(&tick_at(1_199, dec!(100)), &position)
                .is_none()
        );
        assert_eq!(quoter.quotes, 2);
        let mut every = DurationQuoter::new(Hold::EveryTick, one_away);
        for time in [1, 2, 3] {
            assert!(
                every
                    .on_tick(&tick_at(time, dec!(100)), &position)
                    .is_some()
            );
        }
        assert_eq!(every.quotes, 3);
    }

    const EXPIRE: Hold = Hold::Expire {
        live_ms: 100,
        every_ms: 1_000,
    };

    #[test]
    fn test_an_expired_quote_leaves_the_book_empty_until_the_next_decision() {
        // Quotes 99 and 101 at 0, cancelled from 100, and 90 would cross the old bid.
        let ticks = vec![
            tick_at(0, dec!(100)),
            tick_at(100, dec!(100)),
            tick_at(300, dec!(90)),
            tick_at(999, dec!(90)),
        ];
        let (result, strategy) = run(ticks.clone(), EXPIRE, one_away, cancel_after(10));
        assert!(result.trades.is_empty(), "{:?}", result.trades);
        assert_eq!(strategy.quotes, 1);
        assert_eq!(strategy.count(ExecutionKind::Canceled), 2);
        // A refresh at the same rate has its first quote live on the same ticks.
        let (refresh, _) = run(ticks, Hold::For(1_000), one_away, cancel_after(10));
        assert_eq!(refresh.trades.len(), 1);
    }

    #[test]
    fn test_an_expiring_quote_is_live_until_its_life_runs_out() {
        let ticks = vec![tick_at(0, dec!(100)), tick_at(99, dec!(98.9))];
        let (result, strategy) = run(ticks, EXPIRE, one_away, SimLatencyConfig::new());
        assert_eq!(result.trades.len(), 1);
        assert_eq!(strategy.count(ExecutionKind::Canceled), 0);
    }

    #[test]
    fn test_a_fill_before_the_cancel_of_an_expiry_takes_effect_stands() {
        for cross_at in [120, 149] {
            let (result, strategy) = run(race_tape(cross_at), EXPIRE, one_away, cancel_after(50));
            assert_eq!(result.trades.len(), 1, "{cross_at}");
            assert_eq!(result.trades[0].side, Action::Buy);
            assert_eq!(result.trades[0].price, dec!(99));
            assert_eq!(result.trades[0].timestamp, cross_at);
            assert_eq!(result.final_position, ORDER_SIZE);
            // Nothing replaced it.
            assert_eq!(strategy.quotes, 1);
        }
        for (cross_at, cancel_ms) in [(150, 50), (151, 50), (120, 10), (110, 10)] {
            let (result, _) = run(
                race_tape(cross_at),
                EXPIRE,
                one_away,
                cancel_after(cancel_ms),
            );
            assert!(result.trades.is_empty(), "{cross_at} {cancel_ms}");
        }
    }

    #[test]
    fn test_a_cancel_of_an_expiry_too_late_for_a_fill_is_rejected_and_the_fill_stays() {
        let mut ticks = race_tape(120);
        ticks.push(tick_at(200, dec!(99.5)));
        let (result, strategy) = run(ticks, EXPIRE, one_away, cancel_after(50));
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].timestamp, 120);
        assert_eq!(result.final_position, ORDER_SIZE);
        assert_eq!(strategy.count(ExecutionKind::CancelRejected), 1);
        // The other side of the quote was still live, and goes.
        assert_eq!(strategy.count(ExecutionKind::Canceled), 1);
    }

    #[test]
    fn test_an_expiry_cancels_an_order_still_in_flight_before_it_rests() {
        // The quote is active at 200, and its cancel at 110 comes before that.
        let latency = SimLatencyConfig::new()
            .with_submit_ms(200)
            .with_cancel_ms(10);
        let ticks = vec![
            tick_at(0, dec!(100)),
            tick_at(100, dec!(100)),
            tick_at(250, dec!(90)),
        ];
        let (result, strategy) = run(ticks.clone(), EXPIRE, one_away, latency);
        assert!(result.trades.is_empty(), "{:?}", result.trades);
        assert_eq!(strategy.quotes, 1);
        // With a cancel too slow to arrive before that tick, the same quote does fill.
        let slow = latency.with_cancel_ms(200);
        let (filled, _) = run(ticks, EXPIRE, one_away, slow);
        assert_eq!(filled.trades.len(), 1);
    }

    #[test]
    fn test_expiry_and_the_next_decision_come_exactly_when_due() {
        let mut quoter = DurationQuoter::new(EXPIRE, one_away);
        let position = InventoryPosition::new();
        let at = |quoter: &mut DurationQuoter, time| {
            let quote = quoter.on_tick(&tick_at(time, dec!(100)), &position);
            (quote.is_some(), quoter.cancels_when_silent())
        };
        assert!(!quoter.cancels_when_silent());
        assert_eq!(at(&mut quoter, 1_000), (true, false));
        assert_eq!(at(&mut quoter, 1_099), (false, false));
        assert_eq!(at(&mut quoter, 1_100), (false, true));
        assert_eq!(at(&mut quoter, 1_999), (false, true));
        assert_eq!(at(&mut quoter, 2_000), (true, false));
        assert_eq!(at(&mut quoter, 2_099), (false, false));
        assert_eq!(quoter.quotes, 2);
        quoter.reset();
        assert_eq!(quoter.seen, 0);
        assert!(!quoter.cancels_when_silent());
    }

    #[test]
    fn test_only_an_expiring_arm_cancels_when_silent() {
        let position = InventoryPosition::new();
        // A price never comes after the first, so the later ticks are silent, past any hold.
        for hold in [Hold::EveryTick, Hold::For(100)] {
            let mut quoter = DurationQuoter::new(hold, one_away);
            let _ = quoter.on_tick(&tick_at(0, dec!(100)), &position);
            quoter.pricer = never;
            for time in [99, 100, 5_000] {
                assert!(
                    quoter
                        .on_tick(&tick_at(time, dec!(100)), &position)
                        .is_none()
                );
                assert!(!quoter.cancels_when_silent(), "{hold:?} {time}");
            }
        }
    }

    #[test]
    fn test_a_quote_rests_for_its_life_plus_the_cancel_less_the_submit() {
        // Active after 30, with the cancel asked at 50 taking effect at 60: 30 ms of rest.
        let hold = Hold::Expire {
            live_ms: 50,
            every_ms: 1_000,
        };
        let latency = SimLatencyConfig::new()
            .with_submit_ms(30)
            .with_cancel_ms(10);
        let fills = |ticks: Vec<MarketTick>| run(ticks, hold, one_away, latency).0.trades;
        let (start, seen) = (tick_at(0, dec!(100)), tick_at(50, dec!(100)));
        let cross = |at| tick_at(at, dec!(98.9));
        assert!(fills(vec![start.clone(), cross(30), seen.clone()]).is_empty());
        assert_eq!(fills(vec![start.clone(), cross(31), seen.clone()]).len(), 1);
        assert_eq!(fills(vec![start.clone(), seen.clone(), cross(59)]).len(), 1);
        assert!(fills(vec![start, seen, cross(60)]).is_empty());
    }

    #[test]
    fn test_from_a_delay_of_its_life_every_fill_of_an_expiry_has_a_late_cancel() {
        let ticks = short_tape(SEEDS[0]);
        for delay_ms in [50, 200] {
            let expiring = measure(&ticks, EXPIRIES[0], delay_ms, SEEDS[0]);
            assert!(expiring.fills > 0);
            assert_eq!(expiring.late_cancels, expiring.fills, "{delay_ms}");
        }
        // Under its life they part: most fills come before the cancel is asked.
        let expiring = measure(&ticks, EXPIRIES[0], 10, SEEDS[0]);
        assert!(expiring.late_cancels * 2 < expiring.fills, "{expiring:?}");
    }

    #[test]
    fn test_no_price_at_a_decision_leaves_an_expired_quote_cancelled() {
        let mut quoter = DurationQuoter::new(EXPIRE, one_away);
        let position = InventoryPosition::new();
        assert!(quoter.on_tick(&tick_at(0, dec!(100)), &position).is_some());
        quoter.pricer = never;
        assert!(
            quoter
                .on_tick(&tick_at(1_000, dec!(100)), &position)
                .is_none()
        );
        assert!(quoter.cancels_when_silent());
        assert_eq!(quoter.quotes, 1);
    }

    #[test]
    fn test_a_life_as_long_as_the_decision_interval_is_the_refresh() {
        // The control of the expiry's decisions. With a price always there, a quote is
        // always replaced at its end, so this does not reach the silent cancel.
        let ticks = short_tape(SEEDS[0]);
        for delay_ms in DELAYS_MS {
            let measure_of = |hold| measure(&ticks, hold, delay_ms, SEEDS[0]);
            let expiring = measure_of(Hold::Expire {
                live_ms: 250,
                every_ms: 250,
            });
            assert_eq!(expiring, measure_of(Hold::For(250)), "{delay_ms}");
        }
    }

    #[test]
    fn test_an_expiry_decides_at_the_rate_of_its_refresh_and_fills_less() {
        for seed in SEEDS {
            let ticks = short_tape(seed);
            for delay_ms in DELAYS_MS {
                for (expiring, every_ms) in [(EXPIRIES[0], 250), (EXPIRIES[1], 1_000)] {
                    let expiring = measure(&ticks, expiring, delay_ms, seed);
                    let refresh = measure(&ticks, Hold::For(every_ms), delay_ms, seed);
                    assert_eq!(expiring.quotes, refresh.quotes, "{seed} {delay_ms}");
                    assert!(expiring.fills < refresh.fills, "{seed} {delay_ms}");
                }
            }
        }
    }

    #[test]
    fn test_on_the_reported_tapes_an_expiry_has_the_worse_markout() {
        // The negative result of the report, on its own tapes: a shorter life does not
        // give cleaner fills here, on any seed or delay. Shorter tapes need not agree.
        for seed in SEEDS {
            let ticks = generate_tape(seed, TAPE_TICKS);
            for delay_ms in DELAYS_MS {
                for (expiring, every_ms) in [(EXPIRIES[0], 250), (EXPIRIES[1], 1_000)] {
                    let expiring = measure(&ticks, expiring, delay_ms, seed);
                    let refresh = measure(&ticks, Hold::For(every_ms), delay_ms, seed);
                    assert!(expiring.markout < refresh.markout, "{seed} {delay_ms}");
                }
            }
        }
    }

    #[test]
    fn test_the_pricer_gets_the_mid_and_the_delivered_position() {
        fn shifted(mid: Decimal, position: Decimal) -> Option<(Decimal, Decimal)> {
            Some((mid + position - dec!(1), mid + position + dec!(1)))
        }
        let mut quoter = DurationQuoter::new(Hold::EveryTick, shifted);
        let mut position = InventoryPosition::new();
        position.quantity = dec!(3);
        let quote = quoter.on_tick(&tick_at(5, dec!(100)), &position).unwrap();
        assert_eq!(quote.bid_price, dec!(102));
        assert_eq!(quote.ask_price, dec!(104));
        assert_eq!(quote.bid_size, ORDER_SIZE);
        assert_eq!(quote.ask_size, ORDER_SIZE);
        assert_eq!(quote.timestamp, 5);
    }

    #[test]
    fn test_tape_has_jumps_both_ways_on_a_fixed_clock() {
        let ticks = short_tape(SEEDS[0]);
        let moves: Vec<Decimal> = ticks
            .windows(2)
            .map(|pair| pair[1].mid_price() - pair[0].mid_price())
            .collect();
        // Over four standard deviations of two bounces of 0.03: those are the jumps of
        // 0.2, of which a run this long has some each way.
        let count = |pass: fn(&Decimal) -> bool| moves.iter().filter(|m| pass(m)).count();
        assert!(count(|m| *m > dec!(0.18)) >= 4);
        assert!(count(|m| *m < dec!(-0.18)) >= 4);
        assert!(
            ticks
                .windows(2)
                .all(|pair| pair[1].timestamp == pair[0].timestamp + TICK_MS)
        );
        assert_eq!(
            ticks[0].ask_price - ticks[0].bid_price,
            dec!(2) * HALF_TOUCH
        );
    }

    #[test]
    fn test_no_price_means_no_quote_and_the_hold_does_not_start() {
        let mut quoter = DurationQuoter::new(Hold::For(100), never);
        let position = InventoryPosition::new();
        assert!(quoter.on_tick(&tick_at(1, dec!(100)), &position).is_none());
        assert_eq!(quoter.quotes, 0);
        assert_eq!(quoter.last_quote, None);
        quoter.pricer = one_away;
        assert!(quoter.on_tick(&tick_at(2, dec!(100)), &position).is_some());
        quoter.reset();
        assert_eq!(quoter.quotes, 0);
        assert_eq!(quoter.last_quote, None);
        assert!(quoter.reports.is_empty());
    }

    #[test]
    fn test_classical_prices_use_the_baseline_and_the_position() {
        let flat = classical_prices(dec!(100), dec!(0)).unwrap();
        assert_eq!(
            flat,
            calculate_optimal_quotes(dec!(100), dec!(0), GAMMA, SIGMA, TERMINAL_MS, K).unwrap()
        );
        let long = classical_prices(dec!(100), dec!(3)).unwrap();
        assert!(long.0 < flat.0 && long.1 < flat.1);
    }

    fn short_tape(seed: u64) -> Vec<MarketTick> {
        generate_tape(seed, TAPE_TICKS / 5)
    }

    #[test]
    fn test_every_fill_reported_is_in_the_result_and_the_position_adds_up() {
        let ticks = short_tape(SEEDS[0]);
        for delay_ms in DELAYS_MS {
            for hold in HOLDS.into_iter().chain(EXPIRIES) {
                let (result, strategy) =
                    run(ticks.clone(), hold, classical_prices, delays(delay_ms));
                assert_eq!(
                    strategy.count(ExecutionKind::Fill)
                        + strategy.count(ExecutionKind::PartialFill),
                    result.trades.len(),
                    "{delay_ms} {hold:?}"
                );
                let position: Decimal = result
                    .trades
                    .iter()
                    .map(|t| match t.side {
                        Action::Buy => t.quantity,
                        _ => -t.quantity,
                    })
                    .sum();
                assert_eq!(position, result.final_position, "{delay_ms} {hold:?}");
                assert!(result.max_position <= MAX_POSITION, "{delay_ms} {hold:?}");
                assert!(result.accounting.is_consistent(), "{delay_ms} {hold:?}");
            }
        }
    }

    #[test]
    fn test_requoting_every_tick_starves_when_twice_the_delay_fits_the_clock() {
        // The stated limit of the design: a quote is live for one tick interval from
        // one round trip after its tick, and these delays leave no tick inside it.
        let ticks = short_tape(SEEDS[0]);
        let fills = |delay_ms: u64| {
            let (result, strategy) = run(
                ticks.clone(),
                Hold::EveryTick,
                classical_prices,
                delays(delay_ms),
            );
            assert!(strategy.quotes > 0);
            result.trades.len()
        };
        for delay_ms in [5, 10, 15, 50, 200] {
            assert_eq!(fills(delay_ms), 0, "{delay_ms}");
        }
        // Otherwise it does fill, under a tick of delay or over it.
        for delay_ms in [0, 4, 12, 53] {
            assert!(fills(delay_ms) > 0, "{delay_ms}");
        }
    }

    #[test]
    fn test_a_longer_hold_quotes_and_cancels_less_at_each_delay() {
        let ticks = short_tape(SEEDS[0]);
        for delay_ms in DELAYS_MS {
            let measures: Vec<Measures> = HOLDS
                .iter()
                .map(|hold| measure(&ticks, *hold, delay_ms, SEEDS[0]))
                .collect();
            for pair in measures.windows(2) {
                assert!(pair[1].quotes < pair[0].quotes, "{delay_ms} {pair:?}");
                assert!(pair[1].cancels < pair[0].cancels, "{delay_ms} {pair:?}");
            }
        }
    }

    #[test]
    fn test_a_longer_delay_worsens_markout_and_brings_more_late_cancels_on_every_seed() {
        // On each planned tape, for arms that do fill at every delay here.
        for seed in SEEDS {
            let ticks = short_tape(seed);
            for hold in [Hold::For(50), Hold::For(250)] {
                let zero = measure(&ticks, hold, 0, seed);
                let long = measure(&ticks, hold, 200, seed);
                assert!(
                    long.markout < zero.markout,
                    "{seed} {hold:?} {zero:?} {long:?}"
                );
                assert!(long.late_cancels > zero.late_cancels, "{seed} {hold:?}");
            }
        }
    }

    #[test]
    fn test_measures_reproduce_by_seed_and_tapes_differ() {
        let ticks = short_tape(SEEDS[0]);
        assert_eq!(ticks, short_tape(SEEDS[0]));
        assert_ne!(ticks, short_tape(SEEDS[1]));
        let a = measure(&ticks, Hold::For(50), 10, SEEDS[0]);
        let b = measure(&ticks, Hold::For(50), 10, SEEDS[0]);
        assert_eq!(a, b);
        assert!(a.low <= a.high);
        // The interval is for the complete buckets, whose total is reported apart.
        let (result, _) = run(ticks.clone(), Hold::For(50), classical_prices, delays(10));
        let expected: f64 = bucket_changes(&result.equity_curve).iter().sum();
        assert_eq!(a.buckets_total, expected);
        assert_ne!(a.buckets_total, a.net_result);
    }

    #[test]
    fn test_every_arm_reads_the_same_tape_at_the_same_delay() {
        // What differs between two arms is the hold: the ticks seen are the same.
        let ticks = short_tape(SEEDS[0]);
        for hold in HOLDS.into_iter().chain(EXPIRIES) {
            let (result, _) = run(ticks.clone(), hold, classical_prices, delays(50));
            assert_eq!(result.num_ticks, ticks.len() as u64);
            assert_eq!(result.start_time, ticks[0].timestamp);
            assert_eq!(result.end_time, ticks[ticks.len() - 1].timestamp);
        }
    }

    #[test]
    fn test_markout_is_signed_by_side_and_needs_a_later_mid() {
        let ticks = vec![tick_at(0, dec!(100)), tick_at(MARKOUT_MS, dec!(101))];
        let fill = |side, price, timestamp| SimulatedFill::new(side, price, dec!(1), timestamp);
        // Bought at 100.5 and sold at 100.5, against a later mid of 101.
        let buy = fill(Action::Buy, dec!(100.5), 0);
        let sell = fill(Action::Sell, dec!(100.5), 0);
        assert_eq!(mean_markout(std::slice::from_ref(&buy), &ticks), 0.5);
        assert_eq!(mean_markout(std::slice::from_ref(&sell), &ticks), -0.5);
        // With no mid that late it is left out.
        let late = fill(Action::Buy, dec!(1), 1);
        assert_eq!(mean_markout(&[buy.clone(), late], &ticks), 0.5);
        assert_eq!(mid_at_or_after(&ticks, MARKOUT_MS), Some(dec!(101)));
        assert_eq!(mid_at_or_after(&ticks, MARKOUT_MS + 1), None);
    }

    #[test]
    fn test_inventory_counts_fills_up_to_each_tick() {
        let ticks = vec![
            tick_at(0, dec!(100)),
            tick_at(10, dec!(100)),
            tick_at(20, dec!(100)),
            tick_at(30, dec!(100)),
        ];
        let trades = vec![
            SimulatedFill::new(Action::Buy, dec!(100), dec!(2), 10),
            SimulatedFill::new(Action::Sell, dec!(100), dec!(3), 20),
        ];
        // Position 0, 2, -1, -1.
        assert_eq!(inventory_over_ticks(&trades, &ticks), (1.0, 2.0));
    }

    #[test]
    fn test_bucket_changes_use_the_last_equity_of_each_complete_bucket() {
        let curve = vec![
            (0, dec!(100)),
            (BUCKET_MS - 1, dec!(103)),
            (BUCKET_MS, dec!(104)),
            (2 * BUCKET_MS - 1, dec!(101)),
            (2 * BUCKET_MS, dec!(500)),
        ];
        assert_eq!(bucket_changes(&curve), vec![3.0, -2.0]);
        assert!(bucket_changes(&[]).is_empty());
    }

    #[test]
    fn test_block_interval_brackets_the_sum_and_needs_enough_values() {
        let values: Vec<f64> = (0..120).map(|i| f64::from(i % 7) - 3.0).collect();
        let (low, high) = block_interval(&values, 3);
        let sum: f64 = values.iter().sum();
        assert!(low < sum && sum < high, "{low} {sum} {high}");
        let (low, high) = block_interval(&values[..BLOCK_BUCKETS - 1], 3);
        assert!(low.is_nan() && high.is_nan());
        assert_eq!(mean_of(&[]), 0.0);
    }

    #[test]
    fn test_delays_set_every_message() {
        let latency = delays(7);
        assert_eq!(latency.observation_ms, 7);
        assert_eq!(latency.submit_ms, 7);
        assert_eq!(latency.cancel_ms, 7);
        assert_eq!(latency.ack_ms, 7);
        assert_eq!(latency.jitter_ms, 0);
    }
}
