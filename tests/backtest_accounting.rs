//! Accounting port at the backtest fill boundary (M1-04-INT).
//!
//! Hand-checked fixture: the strategy always quotes bid 99 and ask 101 for
//! 1 unit, with a fee rate of 0.001. Quotes made on one tick rest from the
//! next tick on.
//!
//! - t=1000: bid 99.5 / ask 100.5; the first quotes are sent.
//! - t=1001: ask 98.9 <= 99, so the resting bid fills: buy 1 at 99, fee
//!   0.099.
//! - t=1002: bid 101.2 >= 101, so the resting ask fills: sell 1 at 101, fee
//!   0.101.
//!
//! Legacy results: cash flow -99 + 101 = 2, flat, fees 0.2, net 1.8.

use std::sync::{Arc, Mutex};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::accounting::{
    Accountant, AccountingFill, AccountingSnapshot, FillContext, LegacyAccountant,
};
use market_maker_rs::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, MarketTick, SimulatedFill,
    VecDataSource,
};
use market_maker_rs::dec;
use market_maker_rs::execution::Side;
use market_maker_rs::position::inventory::InventoryPosition;
use market_maker_rs::strategy::quote::Quote;
use market_maker_rs::types::error::{MMError, MMResult};

struct FixedQuotes;

impl BacktestStrategy for FixedQuotes {
    fn on_tick(&mut self, tick: &MarketTick, _position: &InventoryPosition) -> Option<Quote> {
        Some(Quote {
            bid_price: dec!(99),
            bid_size: Decimal::ONE,
            ask_price: dec!(101),
            ask_size: Decimal::ONE,
            timestamp: tick.timestamp,
        })
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn reset(&mut self) {}
}

fn ticks() -> Vec<MarketTick> {
    vec![
        MarketTick::new(1000, dec!(99.5), dec!(1), dec!(100.5), dec!(1)),
        MarketTick::new(1001, dec!(98.5), dec!(1), dec!(98.9), dec!(1)),
        MarketTick::new(1002, dec!(101.2), dec!(1), dec!(101.5), dec!(1)),
    ]
}

fn config() -> BacktestConfig {
    BacktestConfig::default()
        .with_initial_capital(dec!(1000))
        .with_fee_rate(dec!(0.001))
        .with_record_trades(true)
        .with_record_equity_curve(true)
}

fn engine() -> BacktestEngine<FixedQuotes, VecDataSource> {
    BacktestEngine::new(config(), FixedQuotes, VecDataSource::new(ticks()))
}

fn context() -> FillContext {
    FillContext::new("BTC-USDC", "BTC", "USDC")
}

/// Test double: records every posting and every mark, and delegates the
/// figures to a legacy accountant.
#[derive(Debug, Clone, Default)]
struct Recording {
    fills: Arc<Mutex<Vec<AccountingFill>>>,
    marks: Arc<Mutex<Vec<(Decimal, u64)>>>,
    resets: Arc<Mutex<u32>>,
    inner: LegacyAccountant,
}

impl Accountant for Recording {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        self.fills.lock().expect("lock").push(fill.clone());
        self.inner.post_fill(fill)
    }

    fn mark_to_market(&mut self, mid_price: Decimal, timestamp: u64) -> MMResult<()> {
        self.marks
            .lock()
            .expect("lock")
            .push((mid_price, timestamp));
        self.inner.mark_to_market(mid_price, timestamp)
    }

    fn snapshot(&self) -> AccountingSnapshot {
        self.inner.snapshot()
    }

    fn reset(&mut self) {
        self.fills.lock().expect("lock").clear();
        self.marks.lock().expect("lock").clear();
        *self.resets.lock().expect("lock") += 1;
        self.inner.reset();
    }
}

fn same_result(a: &BacktestResult, b: &BacktestResult) {
    assert_eq!(a.total_pnl, b.total_pnl);
    assert_eq!(a.total_fees, b.total_fees);
    assert_eq!(a.net_pnl, b.net_pnl);
    assert_eq!(a.num_trades, b.num_trades);
    assert_eq!(a.max_position, b.max_position);
    assert_eq!(a.final_position, b.final_position);
    assert_eq!(a.equity_curve, b.equity_curve);
    assert_eq!(a.max_drawdown, b.max_drawdown);
}

#[test]
fn legacy_default_results_are_unchanged() {
    let result = engine().run();
    assert_eq!(result.num_trades, 2);
    assert_eq!(result.total_pnl, dec!(2));
    assert_eq!(result.total_fees, dec!(0.2));
    assert_eq!(result.net_pnl, dec!(1.8));
    assert_eq!(result.final_position, dec!(0));
    assert_eq!(result.max_position, dec!(1));

    // Selecting the legacy accountant explicitly changes nothing.
    let explicit = engine()
        .with_accountant(LegacyAccountant::new(), context())
        .run();
    same_result(&result, &explicit);
}

#[test]
fn the_accountant_receives_each_fill_once_with_fee_and_context() {
    let double = Recording::default();
    let fills = Arc::clone(&double.fills);
    let marks = Arc::clone(&double.marks);
    let mut engine = engine().with_accountant(double, context());
    let result = engine.run();

    let fills = fills.lock().expect("lock").clone();
    assert_eq!(fills.len() as u64, result.num_trades);
    let summary: Vec<_> = fills
        .iter()
        .map(|f| (f.fill_id, f.side, f.price, f.quantity, f.fee, f.timestamp))
        .collect();
    assert_eq!(
        summary,
        vec![
            (1, Side::Buy, dec!(99), dec!(1), dec!(0.099), 1001),
            (2, Side::Sell, dec!(101), dec!(1), dec!(0.101), 1002),
        ]
    );
    assert!(fills.iter().all(|f| f.context == context()));
    assert_eq!(fills[0].context.fee_currency, "USDC");
    // Each posting matches a fill report of the same order.
    let reports: Vec<_> = engine
        .execution_reports()
        .iter()
        .filter(|r| r.last_quantity > Decimal::ZERO)
        .map(|r| (r.order_id, r.price, r.last_quantity, r.fee))
        .collect();
    let posted: Vec<_> = fills
        .iter()
        .map(|f| (f.order_id, f.price, f.quantity, f.fee))
        .collect();
    assert_eq!(posted, reports);

    // One mark per tick, after that tick's fills.
    assert_eq!(
        marks.lock().expect("lock").clone(),
        vec![(dec!(100), 1000), (dec!(98.7), 1001), (dec!(101.35), 1002)]
    );
    assert_eq!(engine.accounting().fills_posted, 2);
    assert!(engine.accounting_errors().is_empty());
    same_result(&result, &self::engine().run());
}

/// An accountant whose figures differ from the legacy ones: the engine
/// reports exactly what it says, so it is the single authority.
#[derive(Debug, Default)]
struct FixedFigures {
    fills: u64,
}

impl Accountant for FixedFigures {
    fn post_fill(&mut self, _fill: &AccountingFill) -> MMResult<()> {
        self.fills += 1;
        Ok(())
    }

    fn mark_to_market(&mut self, _mid_price: Decimal, _timestamp: u64) -> MMResult<()> {
        Ok(())
    }

    fn snapshot(&self) -> AccountingSnapshot {
        let mut snapshot = AccountingSnapshot::new();
        snapshot.pnl.update(dec!(7), dec!(3));
        snapshot.total_fees = dec!(0.5);
        snapshot.position.quantity = Decimal::from(self.fills);
        snapshot.fills_posted = self.fills;
        snapshot
    }

    fn reset(&mut self) {
        self.fills = 0;
    }
}

#[test]
fn results_come_only_from_the_accountant() {
    let mut engine = engine().with_accountant(FixedFigures::default(), context());
    let result = engine.run();
    assert_eq!(result.total_pnl, dec!(10));
    assert_eq!(result.total_fees, dec!(0.5));
    assert_eq!(result.net_pnl, dec!(9.5));
    assert_eq!(result.final_position, dec!(2));
    assert_eq!(result.max_position, dec!(2));
    assert_eq!(engine.get_state().0.quantity, dec!(2));
    assert_eq!(engine.get_state().1.total, dec!(10));
    // Equity = capital + total - fees on every tick.
    assert!(
        result
            .equity_curve
            .iter()
            .all(|(_, equity)| *equity == dec!(1009.5))
    );
}

#[derive(Debug, Default)]
struct Failing {
    inner: LegacyAccountant,
}

impl Accountant for Failing {
    fn post_fill(&mut self, fill: &AccountingFill) -> MMResult<()> {
        if fill.fill_id == 2 {
            return Err(MMError::InvalidConfiguration(
                "rejected by the ledger".into(),
            ));
        }
        self.inner.post_fill(fill)
    }

    fn mark_to_market(&mut self, mid_price: Decimal, timestamp: u64) -> MMResult<()> {
        self.inner.mark_to_market(mid_price, timestamp)
    }

    fn snapshot(&self) -> AccountingSnapshot {
        self.inner.snapshot()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[test]
fn posting_errors_are_recorded_and_the_fill_still_happens() {
    let mut engine = engine().with_accountant(Failing::default(), context());
    let result = engine.run();
    assert_eq!(result.num_trades, 2);
    assert_eq!(
        engine.accounting_errors(),
        &[MMError::InvalidConfiguration(
            "rejected by the ledger".into()
        )]
    );
    assert_eq!(engine.accounting().fills_posted, 1);
    assert_eq!(result.final_position, dec!(1));
}

#[test]
fn reset_restarts_the_accountant_and_the_fill_ids() {
    let double = Recording::default();
    let fills = Arc::clone(&double.fills);
    let resets = Arc::clone(&double.resets);
    let mut engine = engine().with_accountant(double, context());
    let first = engine.run();
    engine.reset();
    assert_eq!(*resets.lock().expect("lock"), 1);
    assert_eq!(engine.accounting(), &AccountingSnapshot::new());
    assert!(engine.accounting_errors().is_empty());
    let second = engine.run();
    same_result(&first, &second);
    let ids: Vec<u64> = fills
        .lock()
        .expect("lock")
        .iter()
        .map(|f| f.fill_id)
        .collect();
    assert_eq!(ids, vec![1, 2]);
}
