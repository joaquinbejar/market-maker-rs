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
                _ => {
                    position -= trade.quantity;
                    cash += notional - trade.fee;
                }
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
fn test_over_the_planned_episodes_the_gate_ends_each_with_a_solvent_account() {
    for seed in SEEDS {
        let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
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
        account.with_gate(|gate| {
            let ledger = gate.ledger();
            assert!(ledger.reconcile().unwrap().is_clean(), "{seed}");
            for asset in ["USDC", "BTC"] {
                let balance = ledger.balance(&Asset::new(asset).unwrap()).unwrap();
                assert!(balance.total >= Decimal::ZERO, "{seed} {asset}");
            }
        });
        // It cannot sell more than it holds, however much it proposes.
        assert!(result.final_position >= -BASE_DEPOSIT, "{seed}");
    }
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

/// The mean of a figure over the episodes, and its range.
fn line(name: &str, values: &[f64]) -> String {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{name} {mean:.3} [{low:.3}, {high:.3}]")
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
    let per_tick = |sum: Decimal, ticks: usize| (sum / Decimal::from(ticks)).to_f64().unwrap();
    for (name, size, gate) in [
        ("modest, gated", MODEST, true),
        ("oversized, gated", OVERSIZED, true),
        ("oversized, no gate (control)", OVERSIZED, false),
    ] {
        let mut rows: Vec<[f64; 9]> = Vec::new();
        for seed in SEEDS {
            let ticks = tape(seed);
            let (result, counts) = if gate {
                let account = account(QUOTE_DEPOSIT, BASE_DEPOSIT);
                let (result, counts, _) = gated(ticks.clone(), size, &account);
                assert!(account.errors().is_empty(), "{:?}", account.errors());
                (result, counts)
            } else {
                (ungated(ticks.clone(), size), GateCounts::default())
            };
            let costs = episode_costs(&result.trades, &ticks);
            rows.push([
                per_tick(costs.inventory, costs.ticks),
                per_tick(costs.margin, costs.ticks),
                per_tick(costs.drawdown, costs.ticks),
                costs.inventory_violations as f64,
                costs.margin_violations as f64,
                costs.drawdown_violations as f64,
                counts.rejected as f64,
                counts.reduced as f64,
                counts.proposals as f64,
            ]);
        }
        let column = |index: usize| rows.iter().map(|row| row[index]).collect::<Vec<f64>>();
        println!("\n{name}");
        println!(
            "  expected cost: {}, {}, {}",
            line("inventory", &column(0)),
            line("margin", &column(1)),
            line("drawdown", &column(2))
        );
        println!(
            "  violations: {}, {}, {}",
            line("inventory", &column(3)),
            line("margin", &column(4)),
            line("drawdown", &column(5))
        );
        println!(
            "  gate: {}, {}, {}",
            line("rejected", &column(6)),
            line("reduced", &column(7)),
            line("proposals", &column(8))
        );
    }
}
