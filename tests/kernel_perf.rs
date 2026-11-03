//! Release timing of the quote transaction at large capacities (ignored by
//! default): `cargo test --release --features kernel --test kernel_perf --
//! --ignored --nocapture`.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::backtest::kernel::{
    FinancialKernel, KernelAction, KernelConfig, KernelDataset, KernelTick,
};
use market_maker_rs::dec;
use std::time::Instant;

#[test]
#[ignore = "timing; run in release with --ignored"]
fn test_kernel_perf_quote_transaction_large_capacities() {
    // A large stack so the by-value state fits; the step's own stack use
    // is what is measured.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    // 400 rows at one timestamp: capacities in the thousands.
    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(1000), dec!(10), 1_000);
    let rows = (0..400)
        .map(|i| KernelTick {
            ts_ns: 1_000_000 + i / 400,
            bid: 98,
            bid_lots: 1,
            ask: 102,
            ask_lots: 1,
        })
        .collect();
    let data = KernelDataset::from_kernel_ticks(rows, &cfg).unwrap();
    let k: Box<FinancialKernel<'_, 1024, 2048, 3072, 512>> =
        Box::new(FinancialKernel::new(&cfg, &data).unwrap());
    let size = std::mem::size_of::<
        market_maker_rs::backtest::kernel::KernelState<1024, 2048, 3072, 512>,
    >();
    let mut quotes = 0u64;
    let start = Instant::now();
    for seed in 0..20 {
        let mut s = Box::new(k.reset(seed));
        while s.status().is_running() {
            k.step(
                &mut s,
                KernelAction::Quote {
                    bid: 97,
                    ask: 103,
                    bid_lots: 1,
                    ask_lots: 1,
                },
            );
            quotes += 1;
        }
    }
    let per = start.elapsed().as_nanos() / u128::from(quotes);
    println!("state {size} bytes, {quotes} quotes, {per} ns per quote step");
}
