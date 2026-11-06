//! Constraint costs measured apart from reward and from the hard gate (#105).
//!
//! Everything here is synthetic, with made-up budgets and a tape chosen to make positions
//! move. No policy is learned and no constrained optimizer runs: the proposers are scripted,
//! so nothing here compares constrained learning with a penalty, and no result here is a
//! safety guarantee for any algorithm.
//!
//! Three questions are kept apart:
//!
//! - **What a cost is.** Each cost is observable from the position, the equity and the mark at
//!   a tick, is never negative, and is zero inside its budget. See [`inventory_cost`],
//!   [`margin_cost`] and [`drawdown_cost`].
//! - **What a constraint would see.** The expected cost of a tick, and the ticks in violation
//!   in an episode.
//! - **What the hard gate did.** The decisions it rejected or cut, which no cost changes.
//!
//! The gate is `margin-rs`'s pretrade gate, through [`MarginAccount`], unchanged, on a spot
//! account: it reserves what an order could need from the free balance. The margin cost reads
//! the position as a linear perpetual under `margin-rs`'s `linear-im/1` maintenance rule, with
//! [`CAPITAL`], a made-up figure unrelated to the deposits, so it measures a different account
//! than the one the gate enforces. It is an observation, it stops nothing, and the gate never
//! reads it. A spot account is the only one this gate can run: it has no derivative path, so
//! no margin requirement is enforced here, only the balance. A short here is a sale of base
//! held on deposit, costed as a perpetual short.
//!
//! The control has no gate, so it can hold positions no such account could, and only shows
//! what the gate stops. It is not a policy to compare with.
//!
//! A proposer quotes on every other tick and cancels on the ticks between. A requested cancel
//! keeps its reservation until it is confirmed, so quoting on every tick would meet the
//! reservation of its own previous quote, and a refusal would not be about the balance.
//!
//! Evidence: `cargo test --release --features margin --test constraint_costs -- --ignored
//! --nocapture`.

use margin_rs::initial_margin::{ExposurePosition, ExposureView, InitialMarginModel, ProductKind};
use margin_rs::ledger::Ledger;
use margin_rs::maintenance::{AccountState, CollateralView, assess};
use margin_rs::risk::{InstrumentLimits, RiskGate};
use margin_rs::rules::TieredLinearIm;
use margin_rs::{
    AccountId, Asset, ContractSize, InstrumentId, SpotInstrument, TransferId, VenueId,
};
use market_maker_rs::Decimal;
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, MarginAccount, MarketTick,
    PretradeRecord, SimulatedFill, VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Action;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::risk::pretrade::{OrderPurpose, PretradeDecision, PretradeRequest};
use market_maker_rs::strategy::quote::Quote;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Ticks of an episode.
const TICKS: usize = 200;
/// Milliseconds between ticks.
const TICK_MS: u64 = 100;
/// Standard deviation of a move of the mid per tick, in price units.
const STEP: f64 = 0.7;
/// A proposer's distance from the mid, in price units.
const HALF_SPREAD: Decimal = dec!(1);
/// What a modest proposer asks for.
const MODEST: Decimal = dec!(1);
/// What an oversized proposer asks for, beyond any balance of the account.
const OVERSIZED: Decimal = dec!(50);
/// Quote deposit of the account.
const QUOTE_DEPOSIT: Decimal = dec!(1000);
/// Base deposit of the account.
const BASE_DEPOSIT: Decimal = dec!(5);
/// Fee per unit of notional, for every arm.
const FEE: Decimal = dec!(0.0001);
/// Budget of the inventory cost, in base units.
const INVENTORY_BUDGET: Decimal = dec!(5);
/// Budget of the drawdown cost, in quote units.
const DRAWDOWN_BUDGET: Decimal = dec!(20);
/// Capital of the account in the margin cost, in quote units.
const CAPITAL: Decimal = dec!(40);
/// Planned episodes, all reported.
const SEEDS: std::ops::Range<u64> = 1..41;

/// How far the position is outside its budget, on either side, in base units.
fn inventory_cost(position: Decimal, budget: Decimal) -> Decimal {
    (position.abs() - budget).max(Decimal::ZERO)
}

/// How far the equity is under its running peak, beyond the budget, in quote units.
fn drawdown_cost(peak: Decimal, equity: Decimal, budget: Decimal) -> Decimal {
    (peak - equity - budget).max(Decimal::ZERO)
}

/// The shortfall below maintenance, in quote units, of `position` at `mark` with `pnl`, read
/// as a linear perpetual with [`CAPITAL`].
fn margin_cost(position: Decimal, pnl: Decimal, mark: Decimal) -> Decimal {
    let usd = Asset::new("USD").unwrap();
    let rule = TieredLinearIm::linear_v1(usd.clone(), 6).unwrap();
    let view = ExposureView {
        account: AccountId::new("costs").unwrap(),
        venue: VenueId::new("synthetic").unwrap(),
        positions: if position.is_zero() {
            vec![]
        } else {
            vec![ExposurePosition {
                instrument: InstrumentId::new("PERP").unwrap(),
                product: ProductKind::LinearPerpetual,
                settlement: usd.clone(),
                contracts: position,
                size: ContractSize::Multiplier(Decimal::ONE),
                mark,
            }]
        },
    };
    let collateral = CAPITAL + pnl;
    assess(
        AccountState::Healthy,
        &CollateralView {
            currency: usd,
            cash: collateral,
            collateral: Some(collateral),
            equity: Some(collateral),
        },
        &rule.requirement(&view).unwrap(),
    )
    .unwrap()
    .maintenance_shortfall
    .unwrap()
}

/// Sums of each cost over an episode's ticks, and the ticks in violation of each.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Costs {
    inventory: Decimal,
    margin: Decimal,
    drawdown: Decimal,
    inventory_violations: usize,
    margin_violations: usize,
    drawdown_violations: usize,
    ticks: usize,
    /// Cash and position at the last mark, including fees.
    pnl: Decimal,
}

/// The costs of an episode, from its fills and its marks. `pnl` at a tick is cash and position
/// at that mark, which is the same with or without the gate as long as no fill is refused.
fn episode_costs(trades: &[SimulatedFill], ticks: &[MarketTick]) -> Costs {
    let mut costs = Costs::default();
    let (mut position, mut cash, mut peak) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    let mut next = 0;
    for tick in ticks {
        while let Some(trade) = trades.get(next).filter(|t| t.timestamp <= tick.timestamp) {
            let notional = trade.price * trade.quantity;
            match trade.side {
                Action::Buy => {
                    position += trade.quantity;
                    cash -= notional + trade.fee;
                }
                Action::Sell => {
                    position -= trade.quantity;
                    cash += notional - trade.fee;
                }
                other => panic!("a fill is a buy or a sell, not {other:?}"),
            }
            next += 1;
        }
        let mark = tick.mid_price();
        let pnl = cash + position * mark;
        peak = peak.max(pnl);
        let inventory = inventory_cost(position, INVENTORY_BUDGET);
        let margin = margin_cost(position, pnl, mark);
        let drawdown = drawdown_cost(peak, pnl, DRAWDOWN_BUDGET);
        costs.inventory += inventory;
        costs.margin += margin;
        costs.drawdown += drawdown;
        costs.inventory_violations += usize::from(inventory > Decimal::ZERO);
        costs.margin_violations += usize::from(margin > Decimal::ZERO);
        costs.drawdown_violations += usize::from(drawdown > Decimal::ZERO);
        costs.ticks += 1;
        costs.pnl = pnl;
    }
    costs
}

/// What the hard gate did with an episode's proposals.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct GateCounts {
    proposals: usize,
    rejected: usize,
    reduced: usize,
}

fn gate_counts(records: &[PretradeRecord]) -> GateCounts {
    let mut counts = GateCounts {
        proposals: records.len(),
        ..GateCounts::default()
    };
    for record in records {
        match record.decision {
            PretradeDecision::Rejected { .. } => counts.rejected += 1,
            PretradeDecision::Reduced { .. } => counts.reduced += 1,
            PretradeDecision::Accepted => {}
        }
    }
    counts
}

/// Quotes a fixed size around the mid on every other tick, whatever it holds or is told, and
/// cancels on the ticks between.
#[derive(Debug, Clone)]
struct Proposer {
    size: Decimal,
    quoted: bool,
}

impl Proposer {
    fn new(size: Decimal) -> Self {
        Self {
            size,
            quoted: false,
        }
    }
}

impl BacktestStrategy for Proposer {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        self.quoted = !self.quoted;
        if !self.quoted {
            return None;
        }
        let mid = tick.mid_price().round_dp(2);
        Some(Quote {
            bid_price: mid - HALF_SPREAD,
            bid_size: self.size,
            ask_price: mid + HALF_SPREAD,
            ask_size: self.size,
            timestamp: tick.timestamp,
        })
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn cancels_when_silent(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.quoted = false;
    }
}

fn tick_at(timestamp: u64, mid: Decimal) -> MarketTick {
    let half = dec!(0.005);
    MarketTick::new(timestamp, mid - half, dec!(5), mid + half, dec!(5))
}

/// A seeded walk of the mid, so that proposals on both sides fill.
fn tape(seed: u64) -> Vec<MarketTick> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut mid = 100.0f64;
    (0..TICKS)
        .map(|index| {
            let (a, b) = (1.0 - unit(), unit());
            mid += STEP * (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos();
            mid = mid.max(10.0);
            let price = Decimal::from_f64(mid).unwrap().round_dp(2);
            tick_at(index as u64 * TICK_MS, price)
        })
        .collect()
}

/// A spot account with these deposits, and only fees declared as limits.
fn account(quote: Decimal, base: Decimal) -> MarginAccount {
    let usdc = Asset::new("USDC").unwrap();
    let btc = Asset::new("BTC").unwrap();
    let pair = InstrumentId::new("BTC-USDC").unwrap();
    let mut ledger = Ledger::default();
    // Eight digits, so that a fee on two digits of price and of size is on the grid.
    ledger.register_asset(usdc.clone(), 8).unwrap();
    ledger.register_asset(btc.clone(), 8).unwrap();
    ledger
        .register_instrument(
            SpotInstrument::new(pair.clone(), btc.clone(), usdc.clone(), 2, 2).unwrap(),
        )
        .unwrap();
    let mut gate = RiskGate::new(ledger);
    if quote > Decimal::ZERO {
        gate.deposit(TransferId::new("seed-usdc").unwrap(), usdc, quote)
            .unwrap();
    }
    if base > Decimal::ZERO {
        gate.deposit(TransferId::new("seed-btc").unwrap(), btc, base)
            .unwrap();
    }
    gate.set_limits(
        &pair,
        InstrumentLimits {
            max_fee_rate: FEE,
            ..InstrumentLimits::default()
        },
    )
    .unwrap();
    MarginAccount::new(gate, pair).unwrap()
}

fn config() -> BacktestConfig {
    BacktestConfig::default()
        .with_fee_rate(FEE)
        .with_tick_size(dec!(0.01))
        .with_lot_size(dec!(0.01))
        .with_record_trades(true)
}

fn gated(
    ticks: Vec<MarketTick>,
    size: Decimal,
    account: &MarginAccount,
) -> (BacktestResult, GateCounts, Vec<PretradeRecord>) {
    let mut engine = BacktestEngine::new(config(), Proposer::new(size), VecDataSource::new(ticks))
        .with_pretrade_policy(account.policy())
        .with_accountant(account.accountant());
    let result = engine.run();
    assert!(result.accounting.is_consistent(), "{:?}", result.accounting);
    assert!(account.errors().is_empty(), "{:?}", account.errors());
    let records = engine.pretrade_decisions().to_vec();
    (result, gate_counts(&records), records)
}

/// The same proposer with no gate at all: the control of what the gate stops.
fn ungated(ticks: Vec<MarketTick>, size: Decimal) -> BacktestResult {
    BacktestEngine::new(config(), Proposer::new(size), VecDataSource::new(ticks)).run()
}

/// The balances of an account with these deposits after each fill, from the fills alone: the
/// final quote and base, and the lowest each reached.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Balances {
    quote: Decimal,
    base: Decimal,
    lowest_quote: Decimal,
    lowest_base: Decimal,
}

fn balances_after(trades: &[SimulatedFill], quote: Decimal, base: Decimal) -> Balances {
    let mut balances = Balances {
        quote,
        base,
        lowest_quote: quote,
        lowest_base: base,
    };
    for trade in trades {
        let notional = trade.price * trade.quantity;
        match trade.side {
            Action::Buy => {
                balances.quote -= notional + trade.fee;
                balances.base += trade.quantity;
            }
            Action::Sell => {
                balances.quote += notional - trade.fee;
                balances.base -= trade.quantity;
            }
            other => panic!("a fill is a buy or a sell, not {other:?}"),
        }
        balances.lowest_quote = balances.lowest_quote.min(balances.quote);
        balances.lowest_base = balances.lowest_base.min(balances.base);
    }
    balances
}

/// Whether the gate refused this for what the balance could pay.
fn refused_for_the_balance(record: &PretradeRecord) -> bool {
    match &record.decision {
        PretradeDecision::Rejected { reason } => reason.contains("InsufficientBalance"),
        PretradeDecision::Accepted | PretradeDecision::Reduced { .. } => false,
    }
}

fn bought(trades: &[SimulatedFill]) -> Decimal {
    trades
        .iter()
        .filter(|t| t.side == Action::Buy)
        .map(|t| t.quantity)
        .sum()
}

#[test]
fn test_inventory_cost_by_hand() {
    assert_eq!(inventory_cost(dec!(7), dec!(5)), dec!(2));
    // A short costs the same as a long, and inside the budget nothing.
    assert_eq!(inventory_cost(dec!(-7), dec!(5)), dec!(2));
    assert_eq!(inventory_cost(dec!(5), dec!(5)), dec!(0));
    assert_eq!(inventory_cost(dec!(-3), dec!(5)), dec!(0));
}

#[test]
fn test_drawdown_cost_by_hand() {
    assert_eq!(drawdown_cost(dec!(100), dec!(70), dec!(20)), dec!(10));
    assert_eq!(drawdown_cost(dec!(100), dec!(80), dec!(20)), dec!(0));
    // An equity above its peak is no credit.
    assert_eq!(drawdown_cost(dec!(100), dec!(150), dec!(20)), dec!(0));
}

#[test]
fn test_margin_cost_by_hand() {
    // Maintenance is 5% of the notional: 50 on 10 at 100, against capital 40.
    assert_eq!(margin_cost(dec!(10), dec!(0), dec!(100)), dec!(10));
    // A short counts the same, a profit covers it, and a loss adds to it.
    assert_eq!(margin_cost(dec!(-10), dec!(0), dec!(100)), dec!(10));
    assert_eq!(margin_cost(dec!(10), dec!(10), dec!(100)), dec!(0));
    assert_eq!(margin_cost(dec!(10), dec!(-5), dec!(100)), dec!(15));
    // The mark scales it, and no position needs nothing until capital is gone.
    assert_eq!(margin_cost(dec!(10), dec!(0), dec!(200)), dec!(60));
    assert_eq!(margin_cost(dec!(0), dec!(0), dec!(100)), dec!(0));
    assert_eq!(margin_cost(dec!(0), dec!(-41), dec!(100)), dec!(1));
}

#[test]
fn test_episode_costs_by_hand() {
    let ticks = [
        tick_at(0, dec!(100)),
        tick_at(100, dec!(100)),
        tick_at(200, dec!(90)),
        tick_at(300, dec!(90)),
    ];
    let fill = |side, quantity, at| SimulatedFill::new(side, dec!(100), quantity, at);
    let trades = [
        fill(Action::Buy, dec!(7), 100),
        fill(Action::Sell, dec!(7), 300),
    ];
    let costs = episode_costs(&trades, &ticks);
    // Long 7 on the two middle ticks, 2 over budget: 4. Sold at 100 above a mark of 90, the
    // last tick is back to zero at a pnl of 0.
    assert_eq!(costs.inventory, dec!(4));
    assert_eq!(costs.inventory_violations, 2);
    // At 90 the pnl is -70, the peak 0: 50 over the budget of 20, on one tick.
    assert_eq!(costs.drawdown, dec!(50));
    assert_eq!(costs.drawdown_violations, 1);
    // Capital 40 less 70 is -30, maintenance 5% of 630 is 31.5: 61.5 at that tick. Long 7 at
    // 100 needs 35 against 40, which is nothing.
    assert_eq!(costs.margin, dec!(61.5));
    assert_eq!(costs.margin_violations, 1);
    assert_eq!(costs.ticks, 4);
}

#[test]
fn test_episode_costs_charge_a_fee_on_either_side() {
    let ticks = [tick_at(0, dec!(100)), tick_at(100, dec!(100))];
    for side in [Action::Buy, Action::Sell] {
        let mut trade = SimulatedFill::new(side, dec!(100), dec!(1), 100);
        trade.fee = dec!(30);
        // A fee of 30 is a pnl of -30, under a peak of 0: 10 over the budget.
        let costs = episode_costs(&[trade], &ticks);
        assert_eq!(costs.drawdown, dec!(10));
        assert_eq!(costs.pnl, dec!(-30));
    }
}

#[test]
fn test_a_drawdown_is_from_the_running_peak() {
    let ticks = [
        tick_at(0, dec!(100)),
        tick_at(100, dec!(150)),
        tick_at(200, dec!(120)),
    ];
    let trades = [SimulatedFill::new(Action::Buy, dec!(100), dec!(1), 0)];
    // The pnl goes 0, 50, 20: 30 under its peak, which is 10 over the budget.
    let costs = episode_costs(&trades, &ticks);
    assert_eq!(costs.drawdown, dec!(10));
    assert_eq!(costs.drawdown_violations, 1);
}

#[test]
fn test_every_fill_pays_the_declared_fee_with_or_without_the_gate() {
    // Otherwise the costs of the two arms would not compare.
    let ticks = vec![tick_at(0, dec!(100)), tick_at(100, dec!(98.9))];
    let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
    let (with, _, _) = gated(ticks.clone(), MODEST, &account);
    let without = ungated(ticks, MODEST);
    for trades in [&with.trades, &without.trades] {
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].fee, dec!(99) * MODEST * FEE);
    }
}

#[test]
fn test_a_proposal_beyond_the_balance_is_cut_to_what_the_account_can_pay() {
    let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
    let ticks = vec![tick_at(0, dec!(100)), tick_at(100, dec!(98.9))];
    let (result, counts, records) = gated(ticks, OVERSIZED, &account);
    // The proposal stays on the record as it was, with what was submitted beside it.
    let bid = records
        .iter()
        .find(|r| r.request.side == Action::Buy)
        .unwrap();
    assert_eq!(bid.request.quantity, OVERSIZED);
    assert!(bid.submitted < OVERSIZED);
    assert!(bid.submitted > Decimal::ZERO);
    // 99 for each unit, and the fee, from a balance of 1000.
    assert!(bid.submitted * dec!(99) * (Decimal::ONE + FEE) <= QUOTE_DEPOSIT);
    assert!((bid.submitted + dec!(0.01)) * dec!(99) * (Decimal::ONE + FEE) > QUOTE_DEPOSIT);
    assert!(matches!(bid.decision, PretradeDecision::Reduced { .. }));
    assert!(counts.reduced > 0);
    let ask = records
        .iter()
        .find(|r| r.request.side == Action::Sell)
        .unwrap();
    assert_eq!(ask.submitted, BASE_DEPOSIT);
    // What filled is what was submitted, not what was proposed.
    assert_eq!(bought(&result.trades), bid.submitted);
    assert!(account.errors().is_empty(), "{:?}", account.errors());
}

#[test]
fn test_with_nothing_free_a_proposal_gets_no_order_executed() {
    let account = account(Decimal::ZERO, Decimal::ZERO);
    let ticks = vec![
        tick_at(0, dec!(100)),
        tick_at(100, dec!(98.9)),
        tick_at(200, dec!(101.1)),
    ];
    let (result, counts, records) = gated(ticks.clone(), OVERSIZED, &account);
    assert!(result.trades.is_empty(), "{:?}", result.trades);
    assert_eq!(result.final_position, Decimal::ZERO);
    assert_eq!(counts.rejected, counts.proposals);
    assert!(counts.proposals >= 2);
    assert!(records.iter().all(|r| r.submitted.is_zero()));
    assert!(records.iter().all(refused_for_the_balance), "{records:?}");
    // The same proposer with no gate does get it executed, so the gate is what stopped it.
    assert!(!ungated(ticks, OVERSIZED).trades.is_empty());
}

#[test]
fn test_the_ledger_books_each_fill_by_hand() {
    // Bought 1 at 99 for a fee of 0.0099, and sold 1 at 101 for a fee of 0.0101.
    let ticks = vec![
        tick_at(0, dec!(100)),
        tick_at(100, dec!(98.9)),
        tick_at(200, dec!(100)),
        tick_at(300, dec!(101.1)),
    ];
    let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
    let before = account.with_gate(|gate| gate.ledger().journal().len());
    let (result, _, _) = gated(ticks, MODEST, &account);
    assert_eq!(result.trades.len(), 2);
    // The policy books each fill, and the engine's figures read that same ledger back.
    assert_eq!(
        result.accounting.accountant.as_deref(),
        Some("margin-rs gate ledger for BTC-USDC")
    );
    account.with_gate(|gate| {
        let ledger = gate.ledger();
        assert_eq!(ledger.journal().len(), before + 2);
        let total = |asset| ledger.balance(&Asset::new(asset).unwrap()).unwrap().total;
        assert_eq!(total("USDC"), dec!(1001.98));
        assert_eq!(total("BTC"), BASE_DEPOSIT);
    });
    let hand = balances_after(&result.trades, QUOTE_DEPOSIT, BASE_DEPOSIT);
    assert_eq!(hand.quote, dec!(1001.98));
    assert_eq!(hand.base, BASE_DEPOSIT);
    // The first fill spent 99.0099 of quote and the second gave back the base.
    assert_eq!(hand.lowest_quote, dec!(900.9901));
    assert_eq!(hand.lowest_base, BASE_DEPOSIT);
}

#[test]
fn test_over_the_planned_episodes_the_gate_keeps_each_balance_from_going_under_zero() {
    let mut control_overdrawn = 0;
    for seed in SEEDS {
        let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
        let before = account.with_gate(|gate| gate.ledger().journal().len());
        let (result, counts, records) = gated(tape(seed), OVERSIZED, &account);
        assert!(!result.trades.is_empty(), "{seed}");
        assert!(counts.reduced > 0, "{seed}");
        assert!(counts.rejected > 0, "{seed}");
        // Every refusal is about the balance, once fills have used it.
        let refused = records.iter().filter(|r| r.decision.is_rejected());
        assert!(refused.clone().all(refused_for_the_balance), "{seed}");
        assert!(
            account.errors().is_empty(),
            "{seed}: {:?}",
            account.errors()
        );
        assert_eq!(account.fault().map(|e| e.to_string()), None, "{seed}");
        // The balances over the whole path, from the fills, and the ledger agrees at the end.
        let path = balances_after(&result.trades, QUOTE_DEPOSIT, BASE_DEPOSIT);
        assert!(path.lowest_quote >= Decimal::ZERO, "{seed}");
        assert!(path.lowest_base >= Decimal::ZERO, "{seed}");
        account.with_gate(|gate| {
            let ledger = gate.ledger();
            assert!(ledger.reconcile().unwrap().is_clean(), "{seed}");
            assert_eq!(
                ledger.journal().len(),
                before + result.trades.len(),
                "{seed}"
            );
            let total = |asset| ledger.balance(&Asset::new(asset).unwrap()).unwrap().total;
            assert_eq!(total("USDC"), path.quote, "{seed}");
            assert_eq!(total("BTC"), path.base, "{seed}");
        });
        // So the check can fail: the same proposer with no gate goes under on the same tape.
        let control = ungated(tape(seed), OVERSIZED);
        let path = balances_after(&control.trades, QUOTE_DEPOSIT, BASE_DEPOSIT);
        control_overdrawn +=
            usize::from(path.lowest_quote < Decimal::ZERO || path.lowest_base < Decimal::ZERO);
    }
    assert_eq!(control_overdrawn, SEEDS.count());
}

#[test]
fn test_the_gate_stops_what_the_control_executes_and_the_position_is_smaller() {
    for seed in SEEDS.take(10) {
        let ticks = tape(seed);
        let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
        let (gate, _, _) = gated(ticks.clone(), OVERSIZED, &account);
        let control = ungated(ticks.clone(), OVERSIZED);
        assert!(bought(&gate.trades) < bought(&control.trades), "{seed}");
        // One fill of the control is already more than the base deposit could cover.
        assert!(control.max_position > gate.max_position, "{seed}");
        assert!(control.max_position >= OVERSIZED, "{seed}");
        // This follows from the smaller position, and is no evidence of a margin limit.
        let (with, without) = (
            episode_costs(&gate.trades, &ticks),
            episode_costs(&control.trades, &ticks),
        );
        assert!(with.inventory < without.inventory, "{seed}");
    }
}

#[test]
fn test_the_costs_read_the_same_pnl_as_the_engine() {
    for seed in SEEDS.take(10) {
        let ticks = tape(seed);
        let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
        let (gate, _, _) = gated(ticks.clone(), OVERSIZED, &account);
        let control = ungated(ticks.clone(), OVERSIZED);
        assert_eq!(
            episode_costs(&gate.trades, &ticks).pnl,
            gate.net_pnl,
            "{seed}"
        );
        assert_eq!(
            episode_costs(&control.trades, &ticks).pnl,
            control.net_pnl,
            "{seed}"
        );
    }
}

#[test]
fn test_a_violation_and_a_refusal_are_separate_things() {
    let mut refused_without_violation = 0;
    for seed in SEEDS.take(10) {
        let ticks = tape(seed);
        // Plenty of balance, and one fill is over the inventory budget, with nothing refused
        // or cut. The gate reads the account, not these costs.
        let rich = account(dec!(10000000), dec!(10000));
        let (result, counts, _) = gated(ticks.clone(), dec!(10), &rich);
        assert!(
            episode_costs(&result.trades, &ticks).inventory_violations > 0,
            "{seed}"
        );
        assert_eq!(counts.rejected + counts.reduced, 0, "{seed}");
        // So little balance that the position stays inside its budget, with orders refused.
        let poor = account(dec!(100), dec!(1));
        let (result, counts, _) = gated(ticks.clone(), OVERSIZED, &poor);
        assert!(!result.trades.is_empty(), "{seed}");
        assert!(counts.reduced > 0, "{seed}");
        assert_eq!(
            episode_costs(&result.trades, &ticks).inventory_violations,
            0,
            "{seed}"
        );
        refused_without_violation += counts.rejected + counts.reduced;
    }
    assert!(refused_without_violation > 0);
}

#[test]
fn test_gate_counts_keep_each_outcome_apart() {
    let record = |decision| {
        let request = PretradeRequest::new(
            "o",
            Action::Buy,
            Some(dec!(1)),
            dec!(1),
            OrderPurpose::Quote,
            0,
        );
        PretradeRecord {
            order_id: Default::default(),
            request,
            decision,
            submitted: dec!(0),
        }
    };
    let counts = gate_counts(&[
        record(PretradeDecision::Accepted),
        record(PretradeDecision::Reduced {
            quantity: dec!(1),
            reason: String::new(),
        }),
        record(PretradeDecision::Rejected {
            reason: String::new(),
        }),
        record(PretradeDecision::Rejected {
            reason: String::new(),
        }),
    ]);
    assert_eq!(
        counts,
        GateCounts {
            proposals: 4,
            rejected: 2,
            reduced: 1,
        }
    );
}

#[test]
fn test_a_proposal_does_not_meet_the_reservation_of_its_own_last_quote() {
    // With no fill the balance is the same at each proposal, and so is the decision.
    let ticks: Vec<MarketTick> = (0..6)
        .map(|index| tick_at(index * TICK_MS, dec!(100)))
        .collect();
    let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
    let (result, counts, records) = gated(ticks, OVERSIZED, &account);
    assert!(result.trades.is_empty());
    // Bid and ask on three ticks of the six.
    assert_eq!(counts.proposals, 6);
    assert_eq!(counts.reduced, 6);
    assert_eq!(records[0].submitted, records[2].submitted);
    assert_eq!(records[1].submitted, records[3].submitted);
}

#[test]
fn test_a_proposer_alternates_and_restarts_on_a_reset() {
    let mut proposer = Proposer::new(MODEST);
    let position = InventoryPosition::new();
    let tick = tick_at(0, dec!(100));
    assert!(proposer.on_tick(&tick, &position).is_some());
    assert!(proposer.on_tick(&tick, &position).is_none());
    assert!(proposer.on_tick(&tick, &position).is_some());
    assert!(proposer.cancels_when_silent());
    proposer.reset();
    assert!(proposer.on_tick(&tick, &position).is_some());
}

#[test]
fn test_only_a_balance_refusal_counts_as_one() {
    let record = |reason: &str| PretradeRecord {
        order_id: Default::default(),
        request: PretradeRequest::new(
            "o",
            Action::Buy,
            Some(dec!(1)),
            dec!(1),
            OrderPurpose::Quote,
            0,
        ),
        decision: PretradeDecision::Rejected {
            reason: reason.to_string(),
        },
        submitted: dec!(0),
    };
    assert!(refused_for_the_balance(&record("[InsufficientBalance {}]")));
    assert!(!refused_for_the_balance(&record("[DuplicateOrderId]")));
}

#[test]
fn test_a_tape_reproduces_by_seed() {
    assert_eq!(tape(1), tape(1));
    assert_ne!(tape(1), tape(2));
    assert_eq!(tape(1).len(), TICKS);
}

/// One line of the table: the mean of a figure over the episodes, and its range.
fn line(name: &str, values: &[f64]) -> String {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{name} {mean:.3} [{low:.3}, {high:.3}]")
}

/// An arm of the table.
#[derive(Debug, Clone, Copy)]
struct Arm {
    name: &'static str,
    size: Decimal,
    gate: bool,
}

const ARMS: [Arm; 3] = [
    Arm {
        name: "modest, gated",
        size: MODEST,
        gate: true,
    },
    Arm {
        name: "oversized, gated",
        size: OVERSIZED,
        gate: true,
    },
    Arm {
        name: "oversized, no gate (control)",
        size: OVERSIZED,
        gate: false,
    },
];

/// The costs and the gate's decisions of an arm on a tape. A control has no gate, so it has
/// no decisions at all.
fn run_arm(arm: Arm, ticks: &[MarketTick]) -> (Costs, GateCounts) {
    let (result, counts) = if arm.gate {
        let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
        let (result, counts, _) = gated(ticks.to_vec(), arm.size, &account);
        (result, counts)
    } else {
        (ungated(ticks.to_vec(), arm.size), GateCounts::default())
    };
    (episode_costs(&result.trades, ticks), counts)
}

/// The table's text for an arm, from each episode's costs and decisions, kept apart.
fn report(name: &str, episodes: &[(Costs, GateCounts)]) -> String {
    let cost = |pick: fn(&Costs) -> Decimal| -> Vec<f64> {
        episodes
            .iter()
            .map(|(c, _)| (pick(c) / Decimal::from(c.ticks)).to_f64().unwrap())
            .collect()
    };
    let ticks = |pick: fn(&Costs) -> usize| -> Vec<f64> {
        episodes.iter().map(|(c, _)| pick(c) as f64).collect()
    };
    let gate = |pick: fn(&GateCounts) -> usize| -> Vec<f64> {
        episodes.iter().map(|(_, g)| pick(g) as f64).collect()
    };
    format!(
        "{name}\n  expected cost: {}, {}, {}\n  violations: {}, {}, {}\n  gate: {}, {}, {}",
        line("inventory", &cost(|c| c.inventory)),
        line("margin", &cost(|c| c.margin)),
        line("drawdown", &cost(|c| c.drawdown)),
        line("inventory", &ticks(|c| c.inventory_violations)),
        line("margin", &ticks(|c| c.margin_violations)),
        line("drawdown", &ticks(|c| c.drawdown_violations)),
        line("rejected", &gate(|g| g.rejected)),
        line("reduced", &gate(|g| g.reduced)),
        line("proposals", &gate(|g| g.proposals)),
    )
}

#[test]
fn test_a_report_prints_each_figure_under_its_name() {
    let costs = Costs {
        inventory: dec!(10),
        margin: dec!(20),
        drawdown: dec!(30),
        inventory_violations: 1,
        margin_violations: 2,
        drawdown_violations: 3,
        ticks: 10,
        pnl: dec!(0),
    };
    let counts = GateCounts {
        proposals: 6,
        rejected: 4,
        reduced: 5,
    };
    let double = Costs {
        inventory: dec!(30),
        inventory_violations: 3,
        ..costs
    };
    assert_eq!(
        report("arm", &[(costs, counts), (double, counts)]),
        "arm\n  \
         expected cost: inventory 2.000 [1.000, 3.000], margin 2.000 [2.000, 2.000], drawdown \
         3.000 [3.000, 3.000]\n  \
         violations: inventory 2.000 [1.000, 3.000], margin 2.000 [2.000, 2.000], drawdown 3.000 \
         [3.000, 3.000]\n  \
         gate: rejected 4.000 [4.000, 4.000], reduced 5.000 [5.000, 5.000], proposals 6.000 \
         [6.000, 6.000]"
    );
}

#[test]
fn test_the_arms_are_as_named_and_only_the_control_has_no_gate() {
    let ticks = tape(1);
    let run = |index: usize| run_arm(ARMS[index], &ticks);
    assert_eq!(ARMS[0].size, MODEST);
    let (modest, modest_counts) = run(0);
    let (oversized, oversized_counts) = run(1);
    let (control, control_counts) = run(2);
    assert!(modest_counts.proposals > 0);
    assert!(oversized_counts.reduced > modest_counts.reduced);
    assert!(oversized.inventory > modest.inventory);
    assert_eq!(control_counts, GateCounts::default());
    // Same size, and only the gate is between them.
    assert_eq!(ARMS[1].size, ARMS[2].size);
    assert!(control.inventory > oversized.inventory);
}

/// The evidence table. `--ignored --nocapture`, in release, with the `margin` feature.
#[test]
#[ignore = "prints the evidence table"]
fn evidence() {
    println!(
        "SYNTHETIC: scripted proposers, made-up budgets, nothing learned. Inventory budget \
         {INVENTORY_BUDGET}, drawdown budget {DRAWDOWN_BUDGET}, margin capital {CAPITAL}, deposits \
         {QUOTE_DEPOSIT} quote and {BASE_DEPOSIT} base, fee {FEE}, {TICKS} ticks, episodes {SEEDS:?}."
    );
    println!("Mean per episode [lowest, highest]. Costs are per tick; violations are ticks.");
    for arm in ARMS {
        let episodes: Vec<(Costs, GateCounts)> =
            SEEDS.map(|seed| run_arm(arm, &tape(seed))).collect();
        println!("\n{}", report(arm.name, &episodes));
    }
}
