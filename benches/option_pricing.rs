//! Benchmarks for option valuation (#95).
//!
//! - `value/*`: one European option valued on a flat and on a skewed
//!   surface, under sticky moneyness;
//! - `chain_refresh/100_strikes`: valuing and quoting a 100-strike chain
//!   (200 contracts) on a flat surface.
//!
//! Inputs and outputs go through `black_box`. Results depend on machine,
//! toolchain and build profile; record them with the commit SHA. There is no
//! earlier pricing path to compare against (the simplified theo it replaces
//! was not a valuation), so these are absolute numbers.
//!
//! Run with `cargo bench --bench option_pricing --features chain`.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use market_maker_rs::chain::{ChainMarketMaker, ChainPricing};
use market_maker_rs::dec;
use market_maker_rs::options::pricing::{Dynamics, OptionInputs, value};
use market_maker_rs::options::surface::{FlatSurface, TotalVariance, VolSurface};
use market_maker_rs::options::{SurfacePricer, tau_from_days};
use market_maker_rs::types::error::MMResult;
use option_chain_orderbook::orderbook::ExpirationOrderBook;
use optionstratlib::OptionStyle;
use optionstratlib::model::ExpirationDate;
use optionstratlib::prelude::pos_or_panic;

/// `sigma(k) = sqrt(0.04 - 0.05 k + 0.1 k^2)`, a smooth skew.
struct Skew;

impl VolSurface for Skew {
    fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
        let p = 0.04 - 0.05 * k + 0.1 * k * k;
        let p1 = -0.05 + 0.2 * k;
        TotalVariance::new(p * tau, p1 * tau, 0.2 * tau, p)
    }
}

fn single(c: &mut Criterion) {
    let flat = FlatSurface::new(dec!(0.2)).expect("valid volatility");
    let inputs = OptionInputs::european(
        dec!(100),
        dec!(105),
        dec!(0.25),
        dec!(0.03),
        dec!(0.01),
        OptionStyle::Call,
    );
    let mut group = c.benchmark_group("value");
    group.bench_function("flat", |b| {
        b.iter(|| value(black_box(&inputs), &flat, Dynamics::StickyMoneyness))
    });
    group.bench_function("skew", |b| {
        b.iter(|| value(black_box(&inputs), &Skew, Dynamics::StickyMoneyness))
    });
    group.finish();
}

fn chain(c: &mut Criterion) {
    let book = ExpirationOrderBook::new("BTC", ExpirationDate::Days(pos_or_panic!(30.0)));
    for i in 0..100u64 {
        book.get_or_create_strike(25_000 + 500 * i);
    }
    let mm = ChainMarketMaker::with_defaults(Arc::new(book));
    let pricer = SurfacePricer::new(
        Arc::new(FlatSurface::new(dec!(0.6)).expect("valid volatility")),
        Dynamics::StickyMoneyness,
    )
    .expect("valid dynamics");
    let pricing = ChainPricing::new(
        Arc::new(pricer),
        tau_from_days(dec!(30)).expect("valid days"),
        dec!(0),
        dec!(0),
    )
    .expect("valid pricing");
    c.bench_function("chain_refresh/100_strikes", |b| {
        b.iter(|| mm.refresh_all_quotes(black_box(50_000), &pricing))
    });
}

criterion_group!(benches, single, chain);
criterion_main!(benches);
